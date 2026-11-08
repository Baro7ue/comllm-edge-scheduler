"""Colab 闭环：和 Jetson 上的虚拟时间闭环用同一个仿真器、同样的点（台数、N、种子、槽数），只是作答的换成
Colab 上的 PyTorch 模型（原精度 bf16，合并 LoRA）。两边的差别就是部署带来的变化（量化到 Q8_0、换成 llama.cpp）。

用法：Colab 上的 train/S6_Colab闭环.ipynb 解压仓库代码后 import 这个文件。Mac 上只用来打包和测试：
  python -m eval.colab_loop --pack      # 把 sim/、deploy/、eval/ 的代码打成 data/s6_code/，整个文件夹上传到 Drive 的 MyDrive/COMLLM/s6_code/

为什么能并行：一个点里的决策必须一个接一个（每个决策都改变后面任务看到的队列），但不同的点互不相干。
所以同一个模型的 6 个点各开一个线程跑仿真器，线程要决策时把状态交给 Batcher，主线程把手上攒到的状态
拼成一批，GPU 前向一次全部答完，再把答案分回各线程。仿真器和打分的代码和 Jetson 那边完全是同一份。

答案读法和部署时一样：
  compact：只算答案位置的 1 个 token，在整个词表概率最高的 20 个 token 里找概率最大的合法字母；一个都没有算非法
  paper：  贪心生成最多 20 个 token，遇到 <|im_end|> 停，去掉首尾空白后必须正好是一个动作名
非法时和 Jetson 上的调度服务一样回退单步公式，并记一次 fallback。
"""
import argparse
import csv
import hashlib
import json
import os
import re
import threading
import time
import zipfile

from eval.virtual_loop import FIELDS, Recorder, run_point
from sim.cost import best_action, one_step_costs

SERVERS = [3]
NS = [20, 40, 60]
SEEDS = [0, 1]
SLOTS = 1000
WARMUP = 200
STUDENTS = ["main_0.5b", "soft_1.5b", "hard_1.5b", "ps_0.5b", "ps_soft_1.5b", "ps_hard_1.5b"]
CODE_DIRS = ("sim", "deploy", "eval")
CHECK_POINT = {"point": "greedy_k3_n20_sequential_s0_t1000", "servers": 3, "n": 20, "policy": "greedy",
               "mode": "sequential", "seed": 0, "slots": 1000, "warmup": 200}


# ===== 第 1 段：点 =====
def points_for(key):
    """一个模型要跑的点。点的名字和 eval.virtual_loop 的一样，Colab 和 Jetson 的结果能按名字一一对上。"""
    out = []
    for k in SERVERS:
        for n in NS:
            for seed in SEEDS:
                out.append({"point": f"llm_{key}_k{k}_n{n}_sequential_s{seed}_t{SLOTS}", "servers": k, "n": n,
                            "policy": "llm:" + key, "mode": "sequential", "seed": seed, "slots": SLOTS,
                            "warmup": WARMUP})
    return out


# ===== 第 2 段：攒批 =====
class Batcher:
    """几个仿真线程交来的状态攒成一批，主线程一次答完。

    ask（仿真线程调用）：把状态放进队列，等主线程填上答案再返回
    serve（主线程调用）：反复做 ① 等到队列里的状态数 = 还在跑的线程数（或等了 wait_s 秒）
                         ② 整批交给 decide_batch ③ 把答案填回去、叫醒对应线程；所有线程都结束就返回
    """

    def __init__(self, decide_batch, n_threads, wait_s=0.005):
        self.decide_batch = decide_batch
        self.active = n_threads
        self.wait_s = wait_s
        self.queue = []
        self.cv = threading.Condition()
        self.n_calls = 0
        self.n_states = 0
        self.busy_s = 0.0

    def ask(self, s):
        slot = {"s": s, "out": None}
        with self.cv:
            self.queue.append(slot)
            self.cv.notify_all()
            while slot["out"] is None:
                self.cv.wait()
        return slot["out"]

    def finish_thread(self):
        with self.cv:
            self.active -= 1
            self.cv.notify_all()

    def serve(self):
        while True:
            with self.cv:
                while not self.queue and self.active > 0:
                    self.cv.wait()
                if not self.queue and self.active == 0:
                    return
                deadline = time.time() + self.wait_s
                while len(self.queue) < self.active and time.time() < deadline:
                    self.cv.wait(max(0.0, deadline - time.time()))
                batch = self.queue
                self.queue = []
            t0 = time.perf_counter()
            outs = self.decide_batch([slot["s"] for slot in batch])
            self.busy_s += time.perf_counter() - t0
            self.n_calls += 1
            self.n_states += len(batch)
            with self.cv:
                for slot, out in zip(batch, outs):
                    slot["out"] = out
                self.cv.notify_all()


class BatchPolicy:
    """仿真线程里用的策略：问 Batcher 要答案；非法就回退单步公式（和 Jetson 上的调度服务一样），记一次 fallback。"""

    def __init__(self, batcher):
        self.batcher = batcher
        self.n = 0
        self.n_fallback = 0

    def __call__(self, s):
        a = self.batcher.ask(s)
        self.n += 1
        if a is None or a < 0 or a >= s.n_actions:
            self.n_fallback += 1
            return best_action(one_step_costs(s))
        return int(a)


# ===== 第 3 段：跑一个模型的全部点 =====
def finished_points(results_path):
    """results.csv 里已经有的点的名字；文件不存在返回空集合。"""
    done = set()
    if os.path.exists(results_path):
        with open(results_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(r["point"])
    return done


def run_points(points, decide_batch, results_path, sha="", log=print):
    """一个模型的几个点同时跑，结果追加进 results.csv（列和 eval.virtual_loop 的一样）。

    1. 去掉 results.csv 里已经有的点（断了可以接着跑）
    2. 每个点开一个线程：BatchPolicy → Recorder → run_point，跑完记下指标和用时
    3. 主线程当 Batcher，直到所有线程结束
    4. 每个点写一行；ms_per_decision = GPU 忙的总时间 ÷ 决策总数（一批里的几个决策平摊）
    """
    done = finished_points(results_path)
    todo = [p for p in points if p["point"] not in done]
    if not todo:
        log("这些点都跑过了：" + results_path)
        return []
    batcher = Batcher(decide_batch, len(todo))
    results = [None] * len(todo)
    errors = []

    def worker(i, p):
        pol = BatchPolicy(batcher)
        rec = Recorder(pol)
        t0 = time.time()
        try:
            m, share = run_point(p, rec)
            results[i] = (m, share, rec, pol, time.time() - t0)
        except Exception as e:      # noqa: BLE001  一个点出错不拖死其他点
            errors.append((p["point"], repr(e)))
        finally:
            batcher.finish_thread()

    threads = []
    for i, p in enumerate(todo):
        th = threading.Thread(target=worker, args=(i, p), daemon=True)
        th.start()
        threads.append(th)
    t_all = time.time()
    batcher.serve()
    for th in threads:
        th.join()
    ms_per = 1000.0 * batcher.busy_s / max(1, batcher.n_states)
    new_file = not os.path.exists(results_path)
    rows = []
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, p in enumerate(todo):
            if results[i] is None:
                continue
            m, share, rec, pol, wall = results[i]
            row = dict(p)
            row.update(m)
            row["server_share"] = ";".join(str(x) for x in share)
            row["n_decisions"] = rec.n
            row["agree_greedy"] = rec.agree / rec.n if rec.n else float("nan")
            row["fallback"] = pol.n_fallback
            row["ms_per_decision"] = ms_per
            row["wall_s"] = wall
            row["gguf_sha256"] = sha
            out = {}
            for k in FIELDS:
                v = row[k]
                out[k] = round(v, 5) if isinstance(v, float) else v
            w.writerow(out)
            rows.append(out)
            log(f"{p['point']:48s} 平均延迟 {m['mean_delay']:5.2f} 槽  丢弃 {100 * m['drop_rate']:5.1f}%  "
                f"和单步最优一致 {100 * out['agree_greedy']:5.1f}%  回退 {pol.n_fallback}")
    for name, err in errors:
        log(f"{name} 出错：{err}")
    log(f"共 {batcher.n_states} 次决策，{batcher.n_calls} 批（平均每批 {batcher.n_states / max(1, batcher.n_calls):.1f} 个），"
        f"GPU 每次决策 {ms_per:.1f} ms，用时 {(time.time() - t_all) / 60:.1f} 分钟")
    return rows


# ===== 第 4 段：PyTorch 作答（只在 Colab 上用；torch 在函数里才 import，Mac 上测试不需要它） =====
def letter_token_ids(tok):
    """12 个动作字母各自的 token 编号；任何一个不是单 token 就报错（单 token 读概率的前提）。"""
    from sim.render import LETTERS
    ids = []
    for c in LETTERS:
        x = tok.encode(c, add_special_tokens=False)
        assert len(x) == 1, f"动作 {c} 不是单 token：{x}"
        ids.append(x[0])
    return ids


def left_pad(tok, texts, device):
    """左补齐：所有行的最后一个位置都是该出答案的地方；position_ids 让每行真实的第一个 token 从 0 开始。"""
    import torch
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    width = 0
    for x in enc:
        width = max(width, len(x))
    ids = torch.full((len(enc), width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(enc), width), dtype=torch.long)
    for i, x in enumerate(enc):
        ids[i, width - len(x):] = torch.tensor(x, dtype=torch.long)
        mask[i, width - len(x):] = 1
    pos = (mask.cumsum(dim=1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), pos.to(device)


def make_compact_decider(model, tok, device):
    """压缩格式：一批状态 → 一批动作（非法记 −1）。

    1. 每个状态用部署时的同一个函数拼提示词（deploy.prompts.build_llm_request 的 prompt）
    2. 左补齐，前向一次，只取最后一个位置的 logits
    3. 每行：第 20 名的概率当门槛；合法字母里概率最大的那个够门槛就是答案，否则非法
    """
    import torch
    from deploy.prompts import build_llm_request
    letter_ids = letter_token_ids(tok)

    @torch.no_grad()
    def decide(states):
        texts = []
        for s in states:
            texts.append(build_llm_request(s, "compact", True)["prompt"])
        ids, mask, pos = left_pad(tok, texts, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            out = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1)
        probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
        p20 = torch.topk(probs, 20, dim=-1).values[:, -1].cpu().tolist()
        lp = probs[:, letter_ids].cpu().tolist()
        acts = []
        for i, s in enumerate(states):
            best = 0
            for a in range(s.n_actions):
                if lp[i][a] > lp[i][best]:
                    best = a
            acts.append(best if lp[i][best] >= p20[i] else -1)
        return acts

    return decide


ACTION_RE = re.compile(r"本地执行|卸载到边缘服务器(\d+)")     # 和 train/grpo_reward.py 的 ACTION_RE 一样


def parse_paper(text, n_actions, mode):
    """论文格式的回答 → 动作编号；认不出返回 −1。

    strict：去掉首尾空白后必须正好是一个动作名（Jetson 上部署的读法，deploy.prompts.parse_action）
    search：在回答里找第一个动作名（复现 7B 的开环评测用 train/grpo_reward.py 的 parse，就是这种读法）
    服务器编号超出台数都算认不出。
    """
    if mode == "strict":
        from deploy.prompts import parse_action
        a, _, _ = parse_action("paper", "probs", {"content": text}, n_actions)
        return -1 if a is None else a
    m = ACTION_RE.search(text)
    if m is None:
        return -1
    if m.group(0) == "本地执行":
        return 0
    k = int(m.group(1))
    return k if 1 <= k < n_actions else -1


def make_paper_decider(model, tok, device, max_new_tokens=20, parse="strict"):
    """论文格式：一批状态 → 一批动作（认不出记 −1）。贪心生成，遇到 <|im_end|> 停。

    提示词和 Jetson 上部署的论文格式是同一个函数拼的；读法见 parse_paper。
    """
    import torch
    from deploy.prompts import build_llm_request
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    @torch.no_grad()
    def decide(states):
        texts = []
        for s in states:
            texts.append(build_llm_request(s, "paper", True)["prompt"])
        ids, mask, _ = left_pad(tok, texts, device)
        gen = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=max_new_tokens,
                             eos_token_id=eos, pad_token_id=tok.pad_token_id)
        acts = []
        for i, s in enumerate(states):
            text = tok.decode(gen[i, ids.shape[1]:], skip_special_tokens=True)
            acts.append(parse_paper(text, s.n_actions, parse))
        return acts

    return decide


# ===== 第 5 段：打包（Mac 上） =====
def check_value():
    """核对点（单步公式、3 台、N = 20、种子 0）的平均延迟。Colab 上重算一遍，必须和 Mac 上一模一样。"""
    m, _ = run_point(CHECK_POINT, best_policy)
    return m["mean_delay"]


def best_policy(s):
    return best_action(one_step_costs(s))


def pack(out_dir):
    """sim/、deploy/、eval/ 下的 .py 打成 code.zip，meta.json 记校验值、提交号和核对点的数。"""
    from eval.closed_loop import git_version
    os.makedirs(out_dir, exist_ok=True)
    zpath = os.path.join(out_dir, "code.zip")
    names = []
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for d in CODE_DIRS:
            for root, _, files in os.walk(d):
                if "__pycache__" in root:
                    continue
                for fn in sorted(files):
                    if fn.endswith(".py") or fn.endswith(".json"):
                        path = os.path.join(root, fn)
                        z.write(path, path)
                        names.append(path)
    h = hashlib.md5()
    with open(zpath, "rb") as f:
        h.update(f.read())
    meta = {"md5": h.hexdigest(), "git": git_version(), "files": sorted(names),
            "check_point": CHECK_POINT, "check_mean_delay": check_value()}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"打包好了：{zpath}（{len(names)} 个文件，md5 {meta['md5']}，代码版本 {meta['git']}）")
    print(f"核对点平均延迟 {meta['check_mean_delay']}：Colab 上重算必须一样")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", action="store_true", help="打包代码给 Colab 用")
    ap.add_argument("--out", default=os.path.join("data", "s6_code"))
    args = ap.parse_args()
    if args.pack:
        pack(args.out)
    else:
        for key in STUDENTS:
            print(key, len(points_for(key)), "个点")


if __name__ == "__main__":
    main()
