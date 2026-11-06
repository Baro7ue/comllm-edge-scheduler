"""阶段 5 实时闭环的成批扫描：实验 1（稳定边界）和两板复测。夜里自己跑完，不用人守着。

在仓库根目录运行（环境节点上，Jetson 的调度服务已经起来）：
  python -m eval.s5_sweep --plan e1 --dry-run                 # 只列出要跑的点和预计时长，不连 Jetson
  python -m eval.s5_sweep --plan e1                           # 正式跑，结果在 runs/<UTC>_s5_e1/
  python -m eval.s5_sweep --plan e1 --resume runs/<UTC>_s5_e1 # 中途断了：接着跑还没跑完的点
  python -m eval.s5_sweep --plan e1 --rehearse                # 不连 Jetson 的预演：LLM 换成"本机公式 + 假装耗时"

每个点 = (模型键, 决策方式, 终端数 N, 种子, 超时, 分流)。
  同一个模型的点排在一起；轮到新模型时，先让 Jetson 换模型（POST /model），换好了再跑。
  每跑完一个点，在 results.csv 里追加一行，并把这个点的运行目录（config.json、summary.json、tasks.csv.gz）
  放在扫描目录下面。断了以后用 --resume 重跑，results.csv 里已有的点会跳过。
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone

from deploy import testbed
from deploy.client import SchedulerClient, parse_host_port
from eval.closed_loop import git_version
from eval.realtime import run_one

# ===== 第 1 段：扫描计划 =====
# 每行：(模型键, 决策方式, N 列表, 种子列表, Jetson 超时 ms, 分流 ms)
#   决策方式 formula = Jetson 用公式回答（不碰 LLM，量网络和服务开销）；compact / paper = Jetson 上的 LLM
#   超时：部署配置是 100 ms。1.5B 一次约 115 ms，用 100 ms 会大半回退公式，量不到它本身，所以给 1000 ms
#   分流：0 = 关；100 = 等了超过 100 ms 的任务在环境节点上改用公式（eval/realtime.py 开头有说明）
PLANS = {
    # 实验 1 稳定边界（必做）
    "e1": [
        ("formula", "formula", [5, 10, 20, 40, 60, 80], [0, 1], 100, 0),
        ("main_0.5b", "compact", [1, 2, 4, 5, 6, 7, 8, 10], [0, 1], 100, 0),
        ("soft_1.5b", "compact", [1, 2, 3, 4], [0, 1], 1000, 0),
    ],
    # 实验 1 的超载分流对照
    "e1_shed": [
        ("main_0.5b", "compact", [6, 8, 10, 15, 20], [0, 1], 100, 100),
    ],
    # 论文原版 1.5B 一次决策约 1 秒，1 台终端每秒就有 3 个请求，N = 1 已经撑不住；只跑这一个点
    "e1_paper": [
        ("paper_sft_1.5b", "paper", [1], [0], 3000, 0),
    ],
    # 两板复测：RK3588 当环境节点，部署模型在 N_max 附近的 3 个点
    "twoboard": [
        ("main_0.5b", "compact", [5, 6, 7], [0], 100, 0),
    ],
}

# 预演和预计时长用：每个模型一次决策大约多少毫秒（估算，取自阶段 2、3 的 Jetson 实测中位数）
EST_MS = {"formula": 1.2, "main_0.5b": 55.0, "main_0.5b_q4": 60.0, "ps_0.5b": 55.0, "base_0.5b": 55.0,
          "soft_1.5b": 120.0, "paper_sft_1.5b": 900.0}

FIELDS = ["point", "model_key", "policy", "servers", "n", "seed", "minutes", "timeout_ms", "shed_ms", "run_dir",
          "n_tasks", "mean_delay", "p95_delay", "drop_rate", "drop_decision", "local_frac", "llm_share",
          "fallback", "shed", "decision_ms_p50", "decision_ms_p99", "e2e_ms_p50", "e2e_ms_p99", "e2e_ms_mean",
          "req_per_s", "rho", "n_max", "lag_ms_max", "errors", "gguf_sha256"]


def expand(plan_names):
    """计划名（可以逗号分隔几个）→ 点的列表，formula 排最前，其余按模型键排在一起（少换几次模型）。

    1. 逐个计划、逐行、逐个 N、逐个种子展开成点
    2. 同名的点只留一个
    3. 排序：先 formula，再按模型在计划里第一次出现的顺序；同一模型内按 (分流, N, 种子)
    """
    points = []
    seen = set()
    order = {"formula": -1}
    for plan in plan_names.split(","):
        for (key, policy, ns, seeds, timeout_ms, shed_ms) in PLANS[plan]:
            if key not in order:
                order[key] = len(order)
            for n in ns:
                for seed in seeds:
                    name = f"{key}_{policy}_n{n}_s{seed}"
                    if shed_ms > 0:
                        name = name + f"_shed{int(shed_ms)}"
                    if name in seen:
                        continue
                    seen.add(name)
                    points.append({"point": name, "model_key": key, "policy": policy, "n": n, "seed": seed,
                                   "timeout_ms": timeout_ms, "shed_ms": shed_ms})
    points.sort(key=lambda p: (order[p["model_key"]], p["shed_ms"], p["n"], p["seed"]))
    return points


def done_points(path):
    """results.csv 里已经有的点名。"""
    names = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                names.add(r["point"])
    return names


def summary_row(p, servers, minutes, run_dir, s, info):
    """一个点的汇总字典 → results.csv 的一行。"""
    n = s["n_tasks"]
    row = {"point": p["point"], "model_key": p["model_key"], "policy": p["policy"], "servers": servers,
           "n": p["n"], "seed": p["seed"], "minutes": minutes, "timeout_ms": p["timeout_ms"],
           "shed_ms": p["shed_ms"], "run_dir": os.path.basename(run_dir),
           "drop_decision": s["drop_at"].get("decision", 0) / n if n else float("nan"),
           "fallback": s["sources"].get("fallback", 0), "shed": s["sources"].get("shed", 0),
           "gguf_sha256": "" if p["policy"] == "formula" else ((info or {}).get("gguf_sha256") or "")}
    for k in ("n_tasks", "mean_delay", "p95_delay", "drop_rate", "local_frac", "llm_share", "decision_ms_p50",
              "decision_ms_p99", "e2e_ms_p50", "e2e_ms_p99", "e2e_ms_mean", "req_per_s", "rho", "n_max",
              "lag_ms_max", "errors"):
        row[k] = s[k]
    out = {}
    for k in FIELDS:
        v = row[k]
        out[k] = round(v, 5) if isinstance(v, float) else v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="计划名，可以逗号分隔：" + ", ".join(PLANS))
    ap.add_argument("--minutes", type=float, default=3.0, help="每个点跑多少分钟（真实时间）")
    ap.add_argument("--servers", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--only", default=None, help="只跑这些模型键（逗号分隔），其余跳过")
    ap.add_argument("--resume", default=None, help="接着跑的扫描目录")
    ap.add_argument("--scheduler", default=None, help="调度服务地址 IP:端口；不给就用 deploy/testbed.json")
    ap.add_argument("--dry-run", action="store_true", help="只列出点和预计时长")
    ap.add_argument("--rehearse", action="store_true", help="不连 Jetson：每个点改成本机公式 + 假装耗时")
    args = ap.parse_args()

    # 1. 展开计划，筛模型，去掉已经跑完的点
    points = expand(args.plan)
    if args.only:
        keep = args.only.split(",")
        points = [p for p in points if p["model_key"] in keep]
    if args.resume:
        out_dir = args.resume
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tag = args.plan.replace(",", "+")
        out_dir = os.path.join("runs", f"{stamp}_s5_{tag}" + ("_rehearse" if args.rehearse else ""))
    results = os.path.join(out_dir, "results.csv")
    finished = done_points(results)
    todo = [p for p in points if p["point"] not in finished]

    # 2. 列出要跑的点和预计时长：每点 = 跑的分钟数 + 约 0.3 分钟开销；每换一次模型约 1 分钟
    switches = 0
    last = None
    for p in todo:
        if p["policy"] != "formula" and p["model_key"] != last:
            switches += 1
            last = p["model_key"]
    eta_min = len(todo) * (args.minutes + 0.3) + switches
    print(f"计划 {args.plan}：共 {len(points)} 个点，已完成 {len(points) - len(todo)}，要跑 {len(todo)}；"
          f"换模型 {switches} 次；预计约 {eta_min / 60:.1f} 小时（估算）")
    for p in todo:
        print(f"  {p['point']:40s} 超时 {p['timeout_ms']:5.0f} ms  分流 {p['shed_ms']:4.0f} ms")
    if args.dry_run:
        return

    # 3. 准备：扫描目录、记下这次扫描的参数；连调度服务（预演不连）
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sweep_config.json"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"args": vars(args), "git": git_version(), "env_node": testbed.load()["env_node"],
                            "started_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                            "n_todo": len(todo)}, ensure_ascii=False) + "\n")
    client = None
    if not args.rehearse:
        host, port = parse_host_port(args.scheduler)
        client = SchedulerClient(host, port)
        print("调度服务现在加载的模型：", client.info().get("model_key"))
    new_file = not os.path.exists(results)
    f = open(results, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    # 4. 逐点跑：轮到新模型先换模型；一个点出错就记下来，接着跑下一个
    t_start = time.time()
    info = {}
    loaded = None
    for i, p in enumerate(todo):
        policy = p["policy"]
        fake_ms = 0.0
        expect = None
        if args.rehearse:
            fake_ms = EST_MS.get(p["model_key"], 50.0) if policy != "formula" else 0.0
            policy = "local"
        elif policy != "formula":
            expect = p["model_key"]
            if loaded != p["model_key"]:
                print(f"\n换模型 → {p['model_key']}（十几秒到一分钟）", flush=True)
                try:
                    info = client.switch_model(p["model_key"])
                    loaded = p["model_key"]
                except Exception as e:
                    print("  换模型失败，跳过这个模型的点：", repr(e), flush=True)
                    loaded = None
                    continue
        elif client is not None:
            info = client.info()
        try:
            run_dir, s = run_one(p["point"], policy, p["n"], args.servers, args.minutes, args.warmup, p["seed"],
                                 "on", p["timeout_ms"], fake_ms, args.scheduler, p["shed_ms"], expect,
                                 out_root=out_dir, tasks_gz=True, quiet=True)
        except Exception as e:
            print(f"[{i + 1}/{len(todo)}] {p['point']} 出错，跳过：{e!r}", flush=True)
            with open(os.path.join(out_dir, "errors.txt"), "a", encoding="utf-8") as ef:
                ef.write(f"{p['point']}\t{e!r}\n")
            continue
        w.writerow(summary_row(p, args.servers, args.minutes, run_dir, s, info))
        f.flush()
        left = (time.time() - t_start) / (i + 1) * (len(todo) - i - 1) / 60
        print(f"[{i + 1}/{len(todo)}] {p['point']:40s} 平均延迟 {s['mean_delay']:5.2f} 槽  "
              f"丢弃 {100 * s['drop_rate']:5.1f}%  决策 p50 {s['decision_ms_p50']:7.1f} ms  "
              f"LLM 占 {100 * s['llm_share']:5.1f}%  还剩约 {left:.0f} 分钟", flush=True)
    f.close()
    print("写好了：", results)


if __name__ == "__main__":
    main()
