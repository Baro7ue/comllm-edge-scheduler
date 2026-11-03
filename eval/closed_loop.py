"""离线闭环评测：扫终端数 N，每个 (服务器台数, N, 策略, 种子) 跑一次仿真，结果写进一个 CSV。

在仓库根目录运行（-m 的写法让 Python 能找到 sim 包）：
  python -m eval.closed_loop --name s1_sweep_3srv
结果在 runs/<UTC 时间>_<名字>/：results.csv 是数据，config.json 记下这次的参数和代码版本。
"""
import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone

from sim import config as C
from sim.baselines import EXPLORE_NAMES, POLICY_NAMES, make_policy
from sim.env import run

FIELDS = ["servers", "n_terminals", "policy", "mode", "seed", "n_tasks", "mean_delay", "p95_delay",
          "drop_rate", "local_frac", "top_server_frac", "jain", "pred_gap", "wall_s"]


def int_list(text):
    return [int(x) for x in text.split(",")]


def git_version():
    """当前代码的提交号；工作区有没提交的改动就加 -dirty（这样结果能追溯到哪一版代码）。"""
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "sim", "eval", "deploy"],
                                        text=True).strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="这次运行的名字，会出现在目录名里")
    ap.add_argument("--servers", type=int_list, default=[3], help="服务器台数，逗号分隔，如 3,7,11")
    ap.add_argument("--n", type=int_list, default=[10, 20, 30, 40, 50, 60, 80], help="终端数 N 的列表")
    ap.add_argument("--policies", default=",".join(POLICY_NAMES + EXPLORE_NAMES))
    ap.add_argument("--seeds", type=int_list, default=[0, 1, 2])
    ap.add_argument("--slots", type=int, default=3000, help="每次仿真多少个时间槽（3000 槽 = 300 秒）")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--mode", default="sequential", choices=["sequential", "batch"])
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join("runs", f"{stamp}_{args.name}")
    os.makedirs(out_dir, exist_ok=True)
    config = {k: v for k, v in vars(args).items()}
    config.update(git=git_version(), started_utc=stamp,
                  sim={k: getattr(C, k) for k in ("SLOT_S", "DEADLINE_SLOTS", "ARRIVAL_PROB", "TASK_MBITS",
                                                  "UPLINK_MBITS_PER_SLOT", "LOCAL_GHZ", "CYCLES_PER_BIT")})
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    policies = args.policies.split(",")
    total = len(args.servers) * len(args.n) * len(policies) * len(args.seeds)
    done = 0
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in args.servers:
            for n in args.n:
                for p in policies:
                    for seed in args.seeds:
                        t0 = time.perf_counter()
                        m, _ = run(make_policy(p, seed), n, k, slots=args.slots, warmup=args.warmup,
                                   seed=seed, mode=args.mode)
                        row = dict(servers=k, n_terminals=n, policy=p, mode=args.mode, seed=seed,
                                   wall_s=round(time.perf_counter() - t0, 2), **m)
                        w.writerow({key: (round(v, 5) if isinstance(v, float) else v) for key, v in row.items()})
                        f.flush()
                        done += 1
                        print(f"[{done}/{total}] {k} 台服务器 N={n:3d} {p:8s} 种子{seed}  "
                              f"平均延迟 {m['mean_delay']:.2f} 槽  丢弃率 {100 * m['drop_rate']:5.1f}%  "
                              f"本地 {100 * m['local_frac']:4.0f}%")
    print("写好了：", out_dir)


if __name__ == "__main__":
    main()
