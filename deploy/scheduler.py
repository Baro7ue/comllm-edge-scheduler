"""调度服务：跑在 Jetson 上，收环境节点发来的决策请求，回一个动作。

整体结构（一台 Jetson 上两个进程）：

  环境节点 ──HTTP──▶ 本服务（端口 8100）──HTTP──▶ llama-server（端口 8080，只听本机）
                     │  1. 还原状态、拼提示词
                     │  2. 交给 LLM 工作线程，最多等 timeout_ms
                     │  3. 等到了：解析字母 → 动作
                     │  4. 没等到 / 出错：用单步最优公式回答，并标记"回退"
                     │  5. 上一个超时的请求还在算：不排队，马上用公式回答
                     └─ 每个请求在 logs/ 里记一行 JSON

请求 POST /decide：
  {"state": {...}, "fmt": "compact" | "paper" | "formula", "decode": "probs" | "grammar",
   "cache_prompt": true, "timeout_ms": 100, "req_id": 7}
  fmt = formula 时不碰 LLM，直接用公式，用来量"网络 + 服务本身"的开销。
  decode 只对 compact 有效，含义见 deploy/prompts.py。
回复：
  {"action": 2, "source": "llm" | "fallback" | "formula", "reason": "" | "timeout" | "busy" | "invalid" | "error",
   "service_ms": 总耗时, "queue_ms": 在 LLM 队列里等了多久, "llm_ms": 调 llama-server 的往返,
   "prompt_n": 新算的提示词 token, "cache_n": 前缀缓存命中的 token, "prompt_ms": 预填充, "predicted_n": 生成 token 数,
   "predicted_ms": 解码, "text": 模型原文, "probs": 每个动作的概率, "p_chosen": 被选中 token 的概率}

在仓库根目录运行（Jetson 上）：
  python -m deploy.scheduler --model ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
它会自己把 llama-server 拉起来；按 Ctrl-C 两个一起退出。
"""
import argparse
import json
import os
import queue
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from deploy import testbed
from deploy.http_json import JsonConnection
from deploy.prompts import build_llm_request, parse_action
from deploy.wire import state_from_dict
from eval.closed_loop import git_version
from sim.cost import ServerView, State, best_action, one_step_costs

LOG_LOCK = threading.Lock()


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ===== 第 1 段：日志（每个请求一行 JSON） =====
class JsonlLog:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "a", encoding="utf-8")

    def write(self, record):
        """多个线程会同时写，用锁保证每行完整。"""
        line = json.dumps(record, ensure_ascii=False)
        with LOG_LOCK:
            self.f.write(line + "\n")
            self.f.flush()


# ===== 第 2 段：LLM 工作线程 =====
# llama-server 同一时间只算一个请求（--parallel 1），所以这里也只开一个线程排队调用它。
# 请求方（HTTP 处理线程）把任务放进队列后最多等 timeout_ms：
#   - 等到了就用 LLM 的答案；
#   - 没等到就回退公式，并把任务标成"作废"：还没开始的直接跳过，已经在算的算完后记一条"迟到"日志；
#   - 作废的任务还在算的时候又来新请求：新请求不排队，马上回退公式（reason = busy）。
#     不这样做的话，新请求会排在作废任务后面，大概率也超时，每个都白等 timeout_ms，
#     而 timeout_ms 和两个请求的平均间隔差不多长，环境节点那边待决策的任务就会越积越多。
class LlmJob:
    def __init__(self, body, req_id):
        self.body = body
        self.req_id = req_id
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.abandoned = False
        self.started = False
        self.t_submit = time.perf_counter()
        self.t_start = None
        self.t_end = None
        self.resp = None
        self.error = ""


class LlmWorker:
    def __init__(self, host, port, log):
        self.conn = JsonConnection(host, port, timeout_s=120.0)
        self.jobs = queue.Queue()
        self.log = log
        self.current_lock = threading.Lock()
        self.current = None          # 正在调 llama-server 的那个任务，空闲时为 None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, job):
        self.jobs.put(job)

    def busy_with_late_job(self):
        """正在算的任务已经被请求方放弃（超时了还没算完）时返回 True。"""
        with self.current_lock:
            job = self.current
        if job is None:
            return False
        with job.lock:
            return job.abandoned

    def run(self):
        """工作线程的主循环。

        1. 从队列里取一个任务
        2. 请求方已经放弃（超时走了）且还没开始：跳过
        3. 记下"正在算这个任务"，调 llama-server，记下开始和结束时刻
        4. 把结果交回给请求方，清掉"正在算"；如果请求方已经放弃，记一条"迟到"日志
        """
        while True:
            job = self.jobs.get()
            with job.lock:
                if job.abandoned:
                    continue
                job.started = True
                job.t_start = time.perf_counter()
            with self.current_lock:
                self.current = job
            resp = None
            error = ""
            try:
                resp, _ = self.conn.request("POST", "/completion", job.body)
            except Exception as e:      # llama-server 没起来、崩了等等，统一当成"出错"
                error = repr(e)
            with job.lock:
                job.resp = resp
                job.error = error
                job.t_end = time.perf_counter()
                job.done.set()
                late = job.abandoned
            with self.current_lock:
                self.current = None
            if late:
                rec = {"event": "late", "req_id": job.req_id,
                       "llm_ms": round((job.t_end - job.t_start) * 1000.0, 3), "error": error}
                if resp is not None:
                    rec["text"] = resp.get("content", "")
                self.log.write(rec)


# ===== 第 3 段：一次决策 =====
class SchedulerService:
    def __init__(self, worker, log, default_timeout_ms):
        self.worker = worker
        self.log = log
        self.default_timeout_ms = default_timeout_ms
        self.stats_lock = threading.Lock()
        self.n_requests = 0
        self.n_fallback = 0
        self.info = {}

    def decide(self, req, t_in):
        """处理一个 /decide 请求，返回回复字典。t_in 是收到请求的时刻（perf_counter）。

        1. 还原状态，读格式、读法、前缀缓存开关、超时阈值
        2. fmt = formula：直接用公式回答
        3. 工作线程还在算上一个超时作废的任务：不排队，马上回退公式（reason = busy）
        4. 拼 llama-server 请求，交给工作线程，最多等到"收到请求后 timeout_ms"
        5. 没等到：标记作废，回退公式（reason = timeout）
        6. 等到了但出错或答案认不出来：回退公式（reason = error / invalid）
        7. 等到了且答案合法：用 LLM 的答案，附上 llama-server 报的各段耗时和概率
        8. 记日志，返回
        """
        s = state_from_dict(req["state"])
        fmt = req.get("fmt", "compact")
        decode = req.get("decode", "probs")
        cache_prompt = bool(req.get("cache_prompt", True))
        timeout_ms = float(req.get("timeout_ms", self.default_timeout_ms))
        out = {"req_id": req.get("req_id"), "fmt": fmt, "decode": decode, "cache_prompt": cache_prompt,
               "n_actions": s.n_actions, "source": "", "reason": "", "action": None, "queue_ms": 0.0, "llm_ms": 0.0,
               "prompt_n": 0, "cache_n": 0, "prompt_ms": 0.0, "predicted_n": 0, "predicted_ms": 0.0,
               "text": "", "probs": None, "p_chosen": None}

        if fmt == "formula":
            out["action"] = best_action(one_step_costs(s))
            out["source"] = "formula"
            return self.finish(out, t_in)

        if self.worker.busy_with_late_job():
            return self.fallback(out, s, "busy", t_in)

        job = LlmJob(build_llm_request(s, fmt, cache_prompt, decode), out["req_id"])
        self.worker.submit(job)
        remaining_s = timeout_ms / 1000.0 - (time.perf_counter() - t_in)
        if remaining_s > 0:
            job.done.wait(remaining_s)
        with job.lock:
            finished = job.done.is_set()
            if not finished:
                job.abandoned = True
            if job.t_start is not None:
                out["queue_ms"] = (job.t_start - job.t_submit) * 1000.0
            else:
                out["queue_ms"] = (time.perf_counter() - job.t_submit) * 1000.0   # 一直在排队，没轮到
            if finished:
                out["llm_ms"] = (job.t_end - job.t_start) * 1000.0

        if not finished:
            return self.fallback(out, s, "timeout", t_in)
        if job.resp is None:
            out["error"] = job.error
            return self.fallback(out, s, "error", t_in)

        resp = job.resp
        timings = resp.get("timings", {})
        out["prompt_n"] = int(timings.get("prompt_n", 0))
        out["cache_n"] = int(timings.get("cache_n", 0))
        out["prompt_ms"] = float(timings.get("prompt_ms", 0.0))
        out["predicted_n"] = int(timings.get("predicted_n", 0))
        out["predicted_ms"] = float(timings.get("predicted_ms", 0.0))
        out["text"] = resp.get("content", "")
        a, probs, p_chosen = parse_action(fmt, decode, resp, s.n_actions)
        out["probs"] = probs
        out["p_chosen"] = p_chosen
        if a is None:
            return self.fallback(out, s, "invalid", t_in)
        out["action"] = a
        out["source"] = "llm"
        return self.finish(out, t_in)

    def fallback(self, out, s, reason, t_in):
        """LLM 没给出可用答案：用单步最优公式回答（几十微秒），标记来源和原因。"""
        out["action"] = best_action(one_step_costs(s))
        out["source"] = "fallback"
        out["reason"] = reason
        return self.finish(out, t_in)

    def finish(self, out, t_in):
        """补上总耗时，记日志，更新计数。"""
        out["service_ms"] = (time.perf_counter() - t_in) * 1000.0
        self.log.write(out)
        with self.stats_lock:
            self.n_requests += 1
            if out["source"] == "fallback":
                self.n_fallback += 1
        return out


# ===== 第 4 段：HTTP 服务 =====
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"       # 支持长连接：环境节点一条连接一直用
    disable_nagle_algorithm = True      # 小包立即发出，避免和对方的延迟确认凑出几十毫秒的等待
    service = None                      # 启动时填进来

    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        t_in = time.perf_counter()
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n))
        if self.path == "/decide":
            self.send_json(self.service.decide(req, t_in))
        else:
            self.send_json({"error": "unknown path"}, status=404)

    def do_GET(self):
        if self.path == "/info":
            self.send_json(self.service.info)
        elif self.path == "/health":
            self.send_json({"status": "ok"})
        else:
            self.send_json({"error": "unknown path"}, status=404)

    def log_message(self, fmt, *args):
        pass    # 关掉默认的"每个请求打一行"，日志统一写进 logs/


# ===== 第 5 段：拉起 llama-server =====
def start_llama_server(bin_path, model, port, ctx, log_path):
    """后台启动 llama-server，返回进程对象。参数含义见运行步骤文档。"""
    cmd = [os.path.expanduser(bin_path), "-m", os.path.expanduser(model),
           "--host", "127.0.0.1", "--port", str(port),
           "-ngl", "99", "-c", str(ctx), "--parallel", "1", "--cache-ram", "0"]
    log_f = open(log_path, "w", encoding="utf-8")
    print("启动 llama-server：", " ".join(cmd))
    return subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)


def wait_ready(port, proc, timeout_s=180.0):
    """每 0.5 s 问一次 /health，直到 llama-server 加载完模型。

    1. 进程已经退出：说明启动失败，报错
    2. /health 回 ok：返回
    3. 超过 timeout_s 还没好：报错
    """
    conn = JsonConnection("127.0.0.1", port, timeout_s=2.0)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("llama-server 启动失败，看 logs/ 里的 llama_server_*.log")
        try:
            d, _ = conn.request("GET", "/health")
            if d.get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("等了 %d 秒 llama-server 还没准备好" % timeout_s)


def read_props(port):
    """问 llama-server 加载的是哪个模型、哪个版本，写进 /info。"""
    conn = JsonConnection("127.0.0.1", port, timeout_s=5.0)
    d, _ = conn.request("GET", "/props")
    keep = {}
    for k in ("model_path", "build_info", "total_slots"):
        keep[k] = d.get(k)
    keep["n_ctx"] = d.get("default_generation_settings", {}).get("n_ctx")
    return keep


def warm_up(worker):
    """先空跑 3 次，让 CUDA 内核、显存分配都就位，第一批正式请求就不会特别慢。"""
    s = State(d=3.0, local_wait=0.0, net_wait=0.0,
              servers=[ServerView(ghz=41.8, active=0, backlog=0.0), ServerView(ghz=30.0, active=0, backlog=0.0),
                       ServerView(ghz=22.0, active=0, backlog=0.0)])
    for fmt in ("compact", "paper", "compact"):
        job = LlmJob(build_llm_request(s, fmt, cache_prompt=False), "warmup")
        worker.submit(job)
        job.done.wait(60.0)


def main():
    cfg = testbed.load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="GGUF 模型路径；给了就由本服务拉起 llama-server")
    ap.add_argument("--external", action="store_true", help="llama-server 已经单独起好了，不要再拉起")
    ap.add_argument("--host", default="0.0.0.0", help="本服务监听的地址，0.0.0.0 = 所有网卡")
    ap.add_argument("--port", type=int, default=cfg["scheduler_node"]["port"])
    ap.add_argument("--llama-bin", default=cfg["llama_server"]["bin"])
    ap.add_argument("--llama-port", type=int, default=cfg["llama_server"]["port"])
    ap.add_argument("--ctx", type=int, default=cfg["llama_server"]["ctx"])
    ap.add_argument("--timeout-ms", type=float, default=cfg["timeout_ms"], help="请求里没写超时就用这个")
    ap.add_argument("--log-dir", default="logs")
    args = ap.parse_args()
    if args.model is None and not args.external:
        ap.error("要么给 --model，要么加 --external")

    os.makedirs(args.log_dir, exist_ok=True)
    stamp = utc_stamp()
    log = JsonlLog(os.path.join(args.log_dir, f"scheduler_{stamp}.jsonl"))
    proc = None
    if not args.external:
        proc = start_llama_server(args.llama_bin, args.model, args.llama_port, args.ctx,
                                  os.path.join(args.log_dir, f"llama_server_{stamp}.log"))
    try:
        wait_ready(args.llama_port, proc)
        worker = LlmWorker("127.0.0.1", args.llama_port, log)
        warm_up(worker)
        service = SchedulerService(worker, log, args.timeout_ms)
        service.info = {"llama": read_props(args.llama_port), "default_timeout_ms": args.timeout_ms,
                        "started_utc": stamp, "log": log.path, "git": git_version()}
        Handler.service = service
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
        httpd.daemon_threads = True
        print(f"调度服务就绪：{args.host}:{args.port}，模型 {service.info['llama']['model_path']}，日志 {log.path}")
        threading.Thread(target=print_stats, args=(service,), daemon=True).start()
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C，退出")
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=10)


def print_stats(service):
    """每 10 秒打印一行累计请求数和回退数，确认服务活着。没有新请求就不打印。"""
    last = 0
    while True:
        time.sleep(10.0)
        with service.stats_lock:
            n, nf = service.n_requests, service.n_fallback
        if n != last:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 累计请求 {n}，其中回退 {nf}")
            last = n


if __name__ == "__main__":
    main()
