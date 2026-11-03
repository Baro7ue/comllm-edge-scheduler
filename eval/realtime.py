"""实时闭环：环境节点（开发期是 Mac）按真实时钟推进仿真，每个新任务都向 Jetson 要决策。

和离线闭环（eval/closed_loop.py）的区别：
  离线：时间是假的，决策瞬间完成。
  实时：一个槽就是真实的 0.1 秒；决策要等 Jetson 回复，等待的时间算进任务延迟；
        Jetson 忙不过来时，请求在环境节点排队，排到截止期还没轮到的任务直接丢弃。

两个线程：
  主线程   每 0.1 秒开始一个新槽：推进服务器、产生新任务、把新任务放进"待决策队列"。
  发送线程 一次只处理一个请求：从队列里取任务 → 看此刻的状态 → 发请求 → 等回复 → 按收到的时刻执行动作。
           所以每个请求看到的状态都包含了之前所有决策的结果（顺序决策），Jetson 那边也不会同时收到多个请求。
两个线程共用一个仿真器，碰仿真器的地方都加锁。

在仓库根目录运行（Mac 上）：
  python -m eval.realtime --policy formula --n 5 --minutes 10 --name smoke_formula
policy：
  local    在 Mac 上直接用公式，不连 Jetson（检查实时循环本身；--fake-ms 可以假装每次决策要花多久）
  formula  让 Jetson 调度服务用公式回答（量网络和服务开销）
  compact  Jetson 上的 LLM，压缩提示词，单 token
  paper    Jetson 上的 LLM，论文格式长提示词
结果在 runs/<UTC>_<名字>/：tasks.csv 每行一个任务，summary.json 是汇总，config.json 是参数。
"""
import argparse
import csv
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone

from deploy import testbed
from deploy.client import SchedulerClient, parse_host_port
from eval.bench_latency import mean, percentile
from eval.closed_loop import git_version
from sim import config as C
from sim.cost import best_action, one_step_costs
from sim.env import EPS, summarize
from sim.realtime_env import RealtimeEnv

FIELDS = ["tid", "terminal", "t_arrive", "size", "t_sent", "t_recv", "wait_ms", "e2e_ms", "decision_ms",
          "source", "reason", "action", "service_ms", "queue_ms", "llm_ms", "prompt_n", "cache_n", "prompt_ms",
          "predicted_ms", "t_done", "dropped", "drop_at", "delay"]


# ===== 第 1 段：真实时钟 =====
class SlotClock:
    """把 time.perf_counter()（单调时钟，不受系统改时间影响）换算成"第几槽"。"""

    def __init__(self):
        self.t0 = time.perf_counter()

    def now_slots(self):
        return (time.perf_counter() - self.t0) / C.SLOT_S

    def sleep_until_slot(self, t):
        """睡到第 t 槽开始的真实时刻，返回晚了多少毫秒（理想是 0）。"""
        target = self.t0 + t * C.SLOT_S
        wait = target - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        return (time.perf_counter() - target) * 1000.0


# ===== 第 2 段：两种"要决策"的方式 =====
class LocalDecider:
    """在环境节点本机用公式决策。fake_ms > 0 时每次先睡这么久，假装决策很慢（不用 Jetson 也能看排队）。"""

    def __init__(self, fake_ms=0.0):
        self.fake_ms = fake_ms

    def decide(self, state, req_id):
        t0 = time.perf_counter()
        if self.fake_ms > 0:
            time.sleep(self.fake_ms / 1000.0)
        a = best_action(one_step_costs(state))
        ms = (time.perf_counter() - t0) * 1000.0
        return {"action": a, "source": "local", "reason": "", "service_ms": ms, "queue_ms": 0.0, "llm_ms": 0.0,
                "prompt_n": 0, "cache_n": 0, "prompt_ms": 0.0, "predicted_ms": 0.0}, ms


class RemoteDecider:
    """向 Jetson 调度服务要决策。"""

    def __init__(self, client, fmt, cache_prompt, timeout_ms):
        self.client = client
        self.fmt = fmt
        self.cache_prompt = cache_prompt
        self.timeout_ms = timeout_ms

    def decide(self, state, req_id):
        return self.client.decide(state, self.fmt, self.cache_prompt, self.timeout_ms, req_id)


# ===== 第 3 段：发送线程 =====
class Loop:
    """两个线程共用的东西放在一起。"""

    def __init__(self, env, decider, clock):
        self.env = env
        self.decider = decider
        self.clock = clock
        self.lock = threading.Lock()
        self.pending = queue.Queue()
        self.records = {}          # tid → 这个任务的决策记录
        self.n_fallback = 0
        self.n_error = 0
        self.last_error = ""


def sender_loop(loop):
    """一次处理一个待决策任务，直到取到 None。

    1. 取下一个任务
    2. 已经过了截止期：记为在"决策"环节丢弃，不再发请求
    3. 加锁：服务器推进到此刻，看此刻的状态
    4. 不加锁：发请求、等回复（这段时间主线程照常走时钟、产生新任务）
    5. 加锁：服务器推进到收到回复的时刻
       请求失败，或回复到的时候已经过了截止期 → 丢弃；否则从这一刻开始执行动作
    """
    env = loop.env
    while True:
        task = loop.pending.get()
        if task is None:
            return
        rec = loop.records[task.tid]
        now = loop.clock.now_slots()
        with loop.lock:
            if now >= task.deadline - EPS:
                env.expire(task)
                rec["source"] = "expired"
                continue
            env.advance_to(now)
            state = env.observe(task, now)
        rec["t_sent"] = now
        resp = None
        try:
            resp, e2e_ms = loop.decider.decide(state, task.tid)
        except Exception as e:
            loop.n_error += 1
            loop.last_error = repr(e)
        t_recv = loop.clock.now_slots()
        rec["t_recv"] = t_recv
        with loop.lock:
            env.advance_to(t_recv)
            if resp is None:
                env.expire(task)
                rec["source"] = "error"
                continue
            rec["e2e_ms"] = e2e_ms
            for k in ("source", "reason", "service_ms", "queue_ms", "llm_ms", "prompt_n", "cache_n",
                      "prompt_ms", "predicted_ms"):
                rec[k] = resp.get(k)
            if resp.get("source") == "fallback":
                loop.n_fallback += 1
            if t_recv >= task.deadline - EPS:
                env.expire(task)
                continue
            env.apply(task, int(resp["action"]), state, t_recv)


# ===== 第 4 段：主线程 =====
def run_realtime(env, decider, slots, progress_every=100):
    """按真实时钟跑 slots 个槽，返回 (Loop, 每槽的时钟滞后毫秒列表)。

    1. 起发送线程
    2. 每个槽：睡到它的真实时刻（记下晚了多少）→ 加锁开始新槽、产生任务 → 新任务进待决策队列
    3. 每 progress_every 槽打印一行进度
    4. 跑完：通知发送线程退出，等它把手上的请求做完；服务器推进到结束时刻
    """
    clock = SlotClock()
    loop = Loop(env, decider, clock)
    th = threading.Thread(target=sender_loop, args=(loop,), daemon=True)
    th.start()
    lags = []
    for t in range(slots):
        lags.append(clock.sleep_until_slot(t))
        with loop.lock:
            new = env.begin_slot(t)
            for task in new:
                loop.records[task.tid] = {"source": "", "reason": ""}
        for task in new:
            loop.pending.put(task)
        if (t + 1) % progress_every == 0:
            print_progress(loop, t + 1, lags)
    loop.pending.put(None)
    th.join(timeout=30.0)
    with loop.lock:
        env.advance_to(float(slots))
    return loop, lags


def print_progress(loop, t, lags):
    """一行进度：跑了多少秒、任务数、还在排队等决策的、丢弃、回退、最近 100 个决策的 p50、时钟最大滞后。"""
    with loop.lock:
        tasks = list(loop.env.tasks)
    n_drop = 0
    for task in tasks:
        if task.dropped:
            n_drop += 1
    recent = []
    for task in tasks[-100:]:
        rec = loop.records.get(task.tid, {})
        if rec.get("t_recv") is not None:
            recent.append((rec["t_recv"] - task.t_arrive) * C.SLOT_S * 1000.0)
    p50 = percentile(recent, 50) if recent else float("nan")
    print(f"[{t * C.SLOT_S:6.0f} s] 任务 {len(tasks):6d}  待决策 {loop.pending.qsize():4d}  丢弃 {n_drop:5d}  "
          f"回退 {loop.n_fallback:5d}  出错 {loop.n_error}  最近决策 p50 {p50:6.1f} ms  "
          f"时钟最大滞后 {max(lags):5.1f} ms", flush=True)     # 接了 tee 也立刻显示


# ===== 第 5 段：汇总 =====
def build_summary(env, loop, lags, n_servers, slots, warmup):
    """只统计在 [warmup, slots − 11) 之间到达的任务（和离线评测同一个口径），返回汇总字典。

    1. 延迟、丢弃率、负载均衡：和离线评测用同一个函数
    2. 丢弃发生在哪个环节：决策 / 本地 / 上传 / 服务器
    3. 决策延迟（任务视角，= 到达到收到决策，含在环境节点排队）和请求往返（网络 + Jetson）
    4. 决策来源计数、回退原因计数
    5. 时钟滞后：环境节点自己有没有跟上真实时钟
    6. 排队估算：ρ = 每秒请求数 × 平均往返秒数；N_max = 1 / (3 × 平均往返秒数)
    """
    counted = []
    for task in env.tasks:
        if warmup <= task.t_arrive < slots - C.DEADLINE_SLOTS - 1:
            counted.append(task)
    m = summarize(counted, n_servers)
    unresolved = 0
    drop_at = {}
    for task in counted:
        if not task.resolved:
            unresolved += 1
        if task.dropped:
            drop_at[task.drop_at] = drop_at.get(task.drop_at, 0) + 1
    decision_ms = []
    e2e = []
    sources = {}
    reasons = {}
    for task in counted:
        rec = loop.records[task.tid]
        src = rec.get("source", "")
        sources[src] = sources.get(src, 0) + 1
        if rec.get("reason"):
            reasons[rec["reason"]] = reasons.get(rec["reason"], 0) + 1
        if rec.get("t_recv") is not None:
            decision_ms.append((rec["t_recv"] - task.t_arrive) * C.SLOT_S * 1000.0)
        if rec.get("e2e_ms") is not None:
            e2e.append(rec["e2e_ms"])
    seconds = (slots - C.DEADLINE_SLOTS - 1 - warmup) * C.SLOT_S
    req_per_s = len(counted) / seconds
    e2e_mean_s = mean(e2e) / 1000.0 if e2e else float("nan")
    m.update({
        "unresolved": unresolved, "drop_at": drop_at, "sources": sources, "fallback_reasons": reasons,
        "decision_ms_p50": percentile(decision_ms, 50), "decision_ms_p99": percentile(decision_ms, 99),
        "decision_ms_mean": mean(decision_ms),
        "e2e_ms_p50": percentile(e2e, 50), "e2e_ms_p99": percentile(e2e, 99), "e2e_ms_mean": mean(e2e),
        "lag_ms_max": max(lags), "lag_ms_p99": percentile(lags, 99),
        "req_per_s": req_per_s, "rho": req_per_s * e2e_mean_s, "n_max": 1.0 / (3.0 * e2e_mean_s),
        "errors": loop.n_error, "last_error": loop.last_error,
    })
    return m


def write_tasks_csv(path, env, loop):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for task in env.tasks:
            rec = loop.records.get(task.tid, {})
            row = {"tid": task.tid, "terminal": task.terminal, "t_arrive": task.t_arrive, "size": task.size,
                   "t_sent": rec.get("t_sent"), "t_recv": rec.get("t_recv"), "e2e_ms": rec.get("e2e_ms"),
                   "source": rec.get("source", ""), "reason": rec.get("reason", ""), "action": task.action,
                   "t_done": task.t_done, "dropped": task.dropped, "drop_at": task.drop_at}
            for k in ("service_ms", "queue_ms", "llm_ms", "prompt_n", "cache_n", "prompt_ms", "predicted_ms"):
                row[k] = rec.get(k)
            row["wait_ms"] = None
            row["decision_ms"] = None
            if rec.get("t_sent") is not None:
                row["wait_ms"] = (rec["t_sent"] - task.t_arrive) * C.SLOT_S * 1000.0
            if rec.get("t_recv") is not None:
                row["decision_ms"] = (rec["t_recv"] - task.t_arrive) * C.SLOT_S * 1000.0
            row["delay"] = task.delay if task.t_done is not None else None
            out = {}
            for k in FIELDS:
                v = row[k]
                out[k] = round(v, 4) if isinstance(v, float) else v
            w.writerow(out)


def main():
    cfg = testbed.load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="这次运行的名字，会出现在目录名里")
    ap.add_argument("--policy", required=True, choices=["local", "formula", "compact", "paper"])
    ap.add_argument("--n", type=int, default=5, help="终端数 N")
    ap.add_argument("--servers", type=int, default=3, help="服务器台数")
    ap.add_argument("--minutes", type=float, default=10.0, help="跑多少分钟（真实时间）")
    ap.add_argument("--warmup", type=int, default=200, help="开头多少槽不计入统计")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="on", choices=["on", "off"])
    ap.add_argument("--timeout-ms", type=float, default=cfg["timeout_ms"], help="Jetson 上等 LLM 的最长时间")
    ap.add_argument("--fake-ms", type=float, default=0.0, help="只对 local 有效：假装每次决策要花这么久")
    ap.add_argument("--scheduler", default=None, help="调度服务地址 IP:端口；不给就用 deploy/testbed.json")
    args = ap.parse_args()

    slots = int(round(args.minutes * 60.0 / C.SLOT_S))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join("runs", f"{stamp}_{args.name}")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 决策方式
    info = {}
    if args.policy == "local":
        decider = LocalDecider(args.fake_ms)
    else:
        host, port = parse_host_port(args.scheduler)
        client = SchedulerClient(host, port)
        info = client.info()
        decider = RemoteDecider(client, args.policy, args.cache == "on", args.timeout_ms)
    config = {"args": vars(args), "slots": slots, "git": git_version(), "started_utc": stamp,
              "env_node": cfg["env_node"], "scheduler_info": info}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 2. 跑
    print(f"实时闭环：{args.policy}，N={args.n}，{args.servers} 台服务器，{args.minutes} 分钟（{slots} 槽）")
    env = RealtimeEnv(args.n, args.servers, seed=args.seed)
    loop, lags = run_realtime(env, decider, slots)

    # 3. 写结果
    write_tasks_csv(os.path.join(out_dir, "tasks.csv"), env, loop)
    s = build_summary(env, loop, lags, args.servers, slots, args.warmup)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

    # 4. 打印汇总
    print(f"\n统计 {s['n_tasks']} 个任务：平均延迟 {s['mean_delay']:.2f} 槽，丢弃率 {100 * s['drop_rate']:.1f}%"
          f"（按环节 {s['drop_at']}），本地 {100 * s['local_frac']:.0f}%")
    print(f"决策延迟（到达 → 收到决策）：p50 {s['decision_ms_p50']:.1f} ms  p99 {s['decision_ms_p99']:.1f} ms")
    print(f"请求往返（Mac → Jetson → Mac）：p50 {s['e2e_ms_p50']:.1f} ms  p99 {s['e2e_ms_p99']:.1f} ms")
    print(f"决策来源 {s['sources']}，回退原因 {s['fallback_reasons']}，请求出错 {s['errors']}")
    print(f"每秒请求 {s['req_per_s']:.1f}，ρ ≈ {s['rho']:.2f}，N_max ≈ {s['n_max']:.1f} 台终端")
    print(f"时钟滞后：最大 {s['lag_ms_max']:.1f} ms，p99 {s['lag_ms_p99']:.1f} ms；没结束的任务 {s['unresolved']}")
    print("写好了：", out_dir)


if __name__ == "__main__":
    main()
