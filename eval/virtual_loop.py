"""虚拟时间闭环：仿真器停下来等决策，决策花的时间不算进任务延迟。只比"决策质量"。

为什么要它：实时闭环里 0.5B 一次决策约 55 ms，最多撑 6–7 台终端。这么轻的负载下，
单步最优和 PS 感知贪心几乎一样（3 台服务器、N = 5：3.44 vs 3.41 槽，云端实测），好坏策略分不出来。
这里时间是假的，可以在 N = 20–80 的高负载下比；LLM 照样在 Jetson 上逐个作答，答案和部署时一模一样。

计划（--plan，可以逗号分隔一次跑几个；几个计划里重复的点只跑一次）：
  e3      羊群效应：公式两种老师，3/7/11 台 × N = 20/40/60/80，批量 vs 顺序。不用 Jetson，几分钟
  e3_llm  羊群效应的 LLM 确认：11 台、N = 80，main_0.5b 批量 vs 顺序（可选）
  e4a     平分算力标签消融：3 台、N = 20/40/60，main_0.5b vs ps_0.5b，加两个老师和 LACS-greedy
  e4b     泛化：N = 40，3/4/6/7/8/10/11 台（4/6/8/10 训练时没见过），main_0.5b 和老师
  e5      开环 vs 闭环：3 台、N = 40，各种策略各跑一次（和 e4a 重复的点不重跑）
  e5_paper 论文原版 1.5B 闭环（可选，一次决策约 1 秒，只跑 400 槽）
  e6      LACS 往后看在闭环里有没有用：3/7/11 台 × N = 20/40/60/80，单步最优 vs LACS（λ = 0.3/1/2），
          另加 PS 感知贪心当"改公式"的参照，顺序决策 3 个种子；批量决策只比单步最优和 λ = 0.3。每点 2000 槽，和 e3 的点不重名，图 3 不受影响。
          只有公式策略，不用 Jetson，约 25 分钟（估算）

在仓库根目录运行（环境节点上，Jetson 的调度服务已经起来）：
  python -m eval.virtual_loop --plan e4a,e4b,e5 --dry-run
  python -m eval.virtual_loop --plan e4a,e4b,e5               # 结果在 runs/<UTC>_s5_virtual_<计划>/
  python -m eval.virtual_loop --plan e4a,e4b,e5 --resume runs/<目录>
  python -m eval.virtual_loop --plan e3                       # 只有公式策略，不连 Jetson
  python -m eval.virtual_loop --plan e6                       # 同上
  python -m eval.virtual_loop --plan e4a --rehearse           # 预演：LLM 换成本机公式，不连 Jetson

策略名：random / greedy / lacs / lacs:<λ> / po2 / psgreedy / smallnet 在本机算；llm:<模型键> 发给 Jetson。
"""
import argparse
import csv
import glob
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

from deploy import testbed
from deploy.client import SchedulerClient, parse_host_port
from eval.closed_loop import git_version
from sim import config as C
from sim.baselines import make_policy
from sim.cost import best_action, one_step_costs
from sim.env import MecEnv, summarize
from sim.render import compact_state

# ===== 第 1 段：计划 =====
# 每项：(服务器台数列表, N 列表, 策略列表, 决策方式列表, 种子列表, 槽数, 热身槽数)
TEACHERS = ["greedy", "psgreedy", "lacs"]
PLANS = {
    "e3": [([3, 7, 11], [20, 40, 60, 80], ["greedy", "psgreedy"], ["sequential", "batch"], [0, 1], 1500, 200)],
    "e3_llm": [([11], [80], ["greedy", "llm:main_0.5b"], ["sequential", "batch"], [0], 1000, 200)],
    "e4a": [([3], [20, 40, 60], TEACHERS, ["sequential"], [0, 1, 2, 3], 1000, 200),
            ([3], [20, 40, 60], ["llm:main_0.5b", "llm:ps_0.5b"], ["sequential"], [0], 1000, 200)],
    "e4b": [([3, 4, 6, 7, 8, 10, 11], [40], TEACHERS + ["llm:main_0.5b"], ["sequential"], [0], 1000, 200)],
    "e5": [([3], [40], ["random", "po2", "smallnet"] + TEACHERS, ["sequential"], [0], 1000, 200),
           ([3], [40], ["llm:main_0.5b", "llm:ps_0.5b", "llm:soft_1.5b", "llm:base_0.5b"], ["sequential"], [0],
            1000, 200)],
    "e5_paper": [([3], [40], ["llm:paper_sft_1.5b"], ["sequential"], [0], 400, 100)],
    "e6": [([3, 7, 11], [20, 40, 60, 80], ["greedy", "lacs", "lacs:1", "lacs:2", "psgreedy"], ["sequential"], [0, 1, 2],
            2000, 200),
           ([3, 7, 11], [20, 40, 60, 80], ["greedy", "lacs"], ["batch"], [0, 1, 2], 2000, 200)],
}
EST_MS = {"main_0.5b": 55.0, "ps_0.5b": 55.0, "base_0.5b": 55.0, "main_0.5b_q4": 60.0, "soft_1.5b": 120.0,
          "paper_sft_1.5b": 900.0}      # 每次决策约多少毫秒（估算），只用来报预计时长

FIELDS = ["point", "servers", "n", "policy", "mode", "seed", "slots", "warmup", "n_tasks", "mean_delay",
          "p95_delay", "drop_rate", "local_frac", "top_server_frac", "jain", "pred_gap", "server_share",
          "n_decisions", "agree_greedy", "fallback", "ms_per_decision", "wall_s", "gguf_sha256"]


def expand(plan_names):
    """计划名 → 点的列表。本机策略排前面（快），LLM 按模型键排在一起（少换几次模型）；重复的点只留一个。"""
    points = []
    seen = set()
    for plan in plan_names.split(","):
        for (servers, ns, policies, modes, seeds, slots, warmup) in PLANS[plan]:
            for k in servers:
                for n in ns:
                    for pol in policies:
                        for mode in modes:
                            for seed in seeds:
                                name = f"{pol.replace(':', '_')}_k{k}_n{n}_{mode}_s{seed}_t{slots}"
                                if name in seen:
                                    continue
                                seen.add(name)
                                points.append({"point": name, "servers": k, "n": n, "policy": pol, "mode": mode,
                                               "seed": seed, "slots": slots, "warmup": warmup})
    points.sort(key=lambda p: (p["policy"].startswith("llm:"), p["policy"], p["servers"], p["n"], p["mode"],
                               p["seed"]))
    return points


def est_decisions(p):
    return C.ARRIVAL_PROB * p["n"] * p["slots"]


# ===== 第 2 段：策略 =====
class SmallNetPolicy:
    """阶段 3 训练好的 1313 参数小网络（train/small_net.py）。特征和压缩提示词里的数字一样（同样的四舍五入）。

    1. 状态按压缩格式的精度四舍五入
    2. 每个选项一行特征：[是否本地, 任务大小, 要排的队, CPU 频率, 活跃任务数, 积压]
    3. 标准化 → 两层 ReLU → 一个分数；分数最高的选项就是动作
    """

    def __init__(self, path):
        w = np.load(path)
        self.p = {}
        for k in w.files:
            self.p[k] = w[k]

    def __call__(self, s):
        s = compact_state(s)
        rows = [[1.0, s.d, s.local_wait, C.LOCAL_GHZ, 0.0, 0.0]]
        for v in s.servers:
            rows.append([0.0, s.d, s.net_wait, v.ghz, float(v.active), v.backlog])
        x = (np.array(rows) - self.p["mu"]) / self.p["sd"]
        h1 = np.maximum(x @ self.p["W1"] + self.p["b1"], 0)
        h2 = np.maximum(h1 @ self.p["W2"] + self.p["b2"], 0)
        score = (h2 @ self.p["W3"] + self.p["b3"])[:, 0]
        return int(np.argmax(score))


class RemoteLlmPolicy:
    """每个状态发给 Jetson 调度服务，用 LLM 的答案。超时给 10 秒：这里比的是答案，不能让超时回退的公式混进来。"""

    def __init__(self, client, fmt):
        self.client = client
        self.fmt = fmt
        self.n = 0
        self.n_fallback = 0
        self.total_ms = 0.0

    def __call__(self, s):
        resp, ms = self.client.decide(s, self.fmt, True, 10000.0, req_id=self.n)
        self.n += 1
        self.total_ms += ms
        if resp.get("source") != "llm":
            self.n_fallback += 1
        return int(resp["action"])


class Recorder:
    """包在任何策略外面：数决策次数、和单步最优一致的次数。"""

    def __init__(self, policy):
        self.policy = policy
        self.n = 0
        self.agree = 0

    def __call__(self, s):
        a = self.policy(s)
        self.n += 1
        if a == best_action(one_step_costs(s)):
            self.agree += 1
        return a


def latest_smallnet():
    paths = sorted(glob.glob(os.path.join("runs", "s3", "small_net_*", "weights.npz")))
    return paths[-1] if paths else None


# ===== 第 3 段：跑一个点 =====
def run_point(p, policy):
    """跑一次离线闭环（虚拟时间），返回 (指标字典, 每台服务器分到的任务占比)。

    1. 建仿真器，逐槽推进 slots 个槽（顺序或批量决策由 mode 决定）
    2. 只统计 [warmup, slots − 11) 之间到达的任务，和其他评测同一个口径
    3. 每台服务器分到的任务数 ÷ 卸载的任务总数（画羊群效应的直方图用）
    """
    env = MecEnv(p["n"], p["servers"], seed=p["seed"], mode=p["mode"])
    for _ in range(p["slots"]):
        env.step(policy)
    counted = []
    for t in env.tasks:
        if p["warmup"] <= t.t_arrive < p["slots"] - C.DEADLINE_SLOTS - 1:
            counted.append(t)
    m = summarize(counted, p["servers"])
    counts = [0] * p["servers"]
    off = 0
    for t in counted:
        if t.action >= 1:
            counts[t.action - 1] += 1
            off += 1
    share = []
    for c in counts:
        share.append(round(c / off, 4) if off else 0.0)
    return m, share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="计划名，可以逗号分隔：" + ", ".join(PLANS))
    ap.add_argument("--resume", default=None, help="接着跑的目录")
    ap.add_argument("--only", default=None, help="只跑这些策略（逗号分隔，如 llm:main_0.5b,greedy）")
    ap.add_argument("--scheduler", default=None, help="调度服务地址 IP:端口；不给就用 deploy/testbed.json")
    ap.add_argument("--smallnet", default=None, help="小网络权重 weights.npz；不给就用 runs/s3/ 下最新的")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rehearse", action="store_true", help="不连 Jetson：llm:* 换成本机公式")
    args = ap.parse_args()

    # 1. 展开计划、去掉已完成的点、估时长
    points = expand(args.plan)
    if args.only:
        keep = args.only.split(",")
        points = [p for p in points if p["policy"] in keep]
    if args.resume:
        out_dir = args.resume
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = os.path.join("runs", f"{stamp}_s5_virtual_{args.plan.replace(',', '+')}"
                               + ("_rehearse" if args.rehearse else ""))
    results = os.path.join(out_dir, "results.csv")
    finished = set()
    if os.path.exists(results):
        with open(results, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                finished.add(r["point"])
    todo = [p for p in points if p["point"] not in finished]
    eta_s = 0.0
    for p in todo:
        if p["policy"].startswith("llm:") and not args.rehearse:
            eta_s += est_decisions(p) * EST_MS.get(p["policy"][4:], 60.0) / 1000.0
        else:
            eta_s += 0.003 * p["n"] * p["slots"] / 40.0      # 本机策略约每 40 台终端·槽 3 ms（估算）
    print(f"计划 {args.plan}：共 {len(points)} 个点，要跑 {len(todo)} 个，预计约 {eta_s / 3600:.1f} 小时（估算）")
    for p in todo:
        tag = f"约 {est_decisions(p):6.0f} 次决策" if p["policy"].startswith("llm:") else ""
        print(f"  {p['point']:48s} {tag}")
    if args.dry_run:
        return

    # 2. 准备：目录、参数记录、调度服务连接、小网络
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sweep_config.json"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"args": vars(args), "git": git_version(), "n_todo": len(todo),
                            "started_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")},
                           ensure_ascii=False) + "\n")
    client = None
    need_llm = False
    for p in todo:
        if p["policy"].startswith("llm:"):
            need_llm = True
    if need_llm and not args.rehearse:
        host, port = parse_host_port(args.scheduler)
        client = SchedulerClient(host, port)
    smallnet_path = args.smallnet or latest_smallnet()
    new_file = not os.path.exists(results)
    f = open(results, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    # 3. 逐点跑
    loaded = None
    info = {}
    t_all = time.time()
    for i, p in enumerate(todo):
        pol = p["policy"]
        remote = None
        sha = ""
        if pol.startswith("llm:") and not args.rehearse:
            key = pol[4:]
            if loaded != key:
                print(f"\n换模型 → {key}（十几秒到一分钟）", flush=True)
                try:
                    info = client.switch_model(key)
                    loaded = key
                except Exception as e:
                    print("  换模型失败，跳过：", repr(e), flush=True)
                    loaded = None
                    continue
            remote = RemoteLlmPolicy(client, testbed.model_entry(key)["fmt"])
            base = remote
            sha = info.get("gguf_sha256") or ""
        elif pol.startswith("llm:"):
            base = make_policy("greedy")            # 预演：用公式顶替
        elif pol == "smallnet":
            if smallnet_path is None:
                print("没找到小网络权重（runs/s3/small_net_*/weights.npz），跳过", p["point"])
                continue
            base = SmallNetPolicy(smallnet_path)
        else:
            base = make_policy(pol, p["seed"])
        rec = Recorder(base)
        t0 = time.time()
        try:
            m, share = run_point(p, rec)
        except Exception as e:
            print(f"[{i + 1}/{len(todo)}] {p['point']} 出错，跳过：{e!r}", flush=True)
            with open(os.path.join(out_dir, "errors.txt"), "a", encoding="utf-8") as ef:
                ef.write(f"{p['point']}\t{e!r}\n")
            continue
        wall = time.time() - t0
        row = dict(p)
        row.update(m)
        row["server_share"] = ";".join(str(x) for x in share)
        row["n_decisions"] = rec.n
        row["agree_greedy"] = rec.agree / rec.n if rec.n else float("nan")
        row["fallback"] = remote.n_fallback if remote else 0
        row["ms_per_decision"] = remote.total_ms / remote.n if remote and remote.n else 0.0
        row["wall_s"] = wall
        row["gguf_sha256"] = sha
        out = {}
        for k in FIELDS:
            v = row[k]
            out[k] = round(v, 5) if isinstance(v, float) else v
        w.writerow(out)
        f.flush()
        print(f"[{i + 1}/{len(todo)}] {p['point']:48s} 平均延迟 {m['mean_delay']:5.2f} 槽  丢弃 "
              f"{100 * m['drop_rate']:5.1f}%  本地 {100 * m['local_frac']:4.0f}%  和单步最优一致 "
              f"{100 * row['agree_greedy']:5.1f}%  用时 {wall / 60:5.1f} 分钟  累计 {(time.time() - t_all) / 60:.0f} 分钟",
              flush=True)
    f.close()
    print("写好了：", results)


if __name__ == "__main__":
    main()
