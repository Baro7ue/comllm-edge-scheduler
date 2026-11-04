"""在仿真器的真实状态分布上重跑 10/23 的 LACS 检查，用结果决定阶段 3 的软标签用
"单步成本"还是"单步 + LACS"。

10/23 那次用的是我手定范围的随机状态；这次的状态来自闭环仿真（调度器真会遇到的状态）。
做法：用某个行为策略（greedy 或 random）跑闭环，把每次决策时的状态记下来，
再对每个状态比较"只看 J 的选择"和"看 J + λ·C 的选择"是否不同。

  python -m eval.lacs_check_sim
  python -m eval.lacs_check_sim --sweep runs/<目录>/results.csv    # 顺便对比闭环里两种基线的结果

判定规则（运行前就定好，不看结果再改）：
  λ=0.3 时在所有配置里改变决策的状态都 < 1%，并且闭环扫描里 LACS-greedy 和单步最优的
  平均延迟相差 < 2%、丢弃率相差 < 1 个百分点 → 软标签只用单步成本；否则用单步 + LACS。
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

from sim.baselines import make_policy
from sim.cost import best_action, lacs_impact, one_step_costs
from sim.env import MecEnv

LAMBDAS = (0.3, 1.0, 2.0)


def collect_states(behavior, n_servers, n_terminals, n_states, seed=0, warmup=200):
    """用行为策略跑闭环，返回热身之后的 n_states 个决策状态。"""
    env = MecEnv(n_terminals, n_servers, seed=seed)
    pol, states = make_policy(behavior, seed), []

    def record(s):
        if env.t >= warmup:
            states.append(s)
        return pol(s)

    while len(states) < n_states:
        env.step(record)
    return states[:n_states]


def check(states):
    """返回 {λ: (改变决策的比例, 改变时 J+λC 平均好了多少槽)}"""
    flips = {lam: [] for lam in LAMBDAS}
    for s in states:
        J = np.array(one_step_costs(s))
        Cs = np.array([lacs_impact(s, a) for a in range(s.n_actions)])
        g = best_action(J)
        for lam in LAMBDAS:
            total = J + lam * Cs
            a = best_action(total)
            if a != g:
                flips[lam].append(total[g] - total[a])
    return {lam: (len(v) / len(states), float(np.mean(v)) if v else 0.0) for lam, v in flips.items()}


def sweep_compare(path):
    rows = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["policy"] in ("greedy", "lacs"):
                rows[(int(r["servers"]), int(r["n_terminals"]), r["policy"])].append(r)
    print("\n闭环扫描里 LACS-greedy 对比单步最优（几个种子的平均）：")
    print("服务器  N    平均延迟 单步/LACS（槽）  相差     丢弃率 单步/LACS    相差（百分点）")
    worst_delay, worst_drop = 0.0, 0.0
    for k, n in sorted({(k, n) for k, n, _ in rows}):
        g = rows[(k, n, "greedy")]
        lc = rows[(k, n, "lacs")]
        gd = np.mean([float(r["mean_delay"]) for r in g])
        ld = np.mean([float(r["mean_delay"]) for r in lc])
        gp = 100 * np.mean([float(r["drop_rate"]) for r in g])
        lp = 100 * np.mean([float(r["drop_rate"]) for r in lc])
        rel = 100 * (ld - gd) / gd
        worst_delay, worst_drop = max(worst_delay, abs(rel)), max(worst_drop, abs(lp - gp))
        print(f"{k:4d} {n:4d}    {gd:6.3f} / {ld:6.3f}        {rel:+5.2f}%    {gp:5.2f}% / {lp:5.2f}%   {lp - gp:+5.2f}")
    return worst_delay, worst_drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", default="3,7,11")
    ap.add_argument("--n", default="20,40")
    ap.add_argument("--states", type=int, default=3000, help="每个配置收集多少个状态")
    ap.add_argument("--sweep", help="closed_loop.py 输出的 results.csv（可选）")
    args = ap.parse_args()

    worst_flip = 0.0
    print("行为策略  服务器  N   状态数  平均活跃数  平均积压(Mbits)   " +
          "   ".join(f"λ={lam}: 改变比例 / 平均好多少" for lam in LAMBDAS))
    for behavior in ("greedy", "random"):
        for k in map(int, args.servers.split(",")):
            for n in map(int, args.n.split(",")):
                states = collect_states(behavior, k, n, args.states)
                act = np.mean([v.active for s in states for v in s.servers])
                bl = np.mean([v.backlog for s in states for v in s.servers])
                res = check(states)
                worst_flip = max(worst_flip, res[0.3][0])
                cells = "   ".join(f"{100 * f:6.2f}% / {g:.3f} 槽" for f, g in res.values())
                print(f"{behavior:8s} {k:4d} {n:4d} {len(states):7d}   {act:8.2f}   {bl:10.2f}        {cells}")
    print("\n参照：原作者数据（按答案配平过）里服务器平均活跃数 1.44、平均积压 4.5 Mbits")

    print(f"\nλ=0.3 时最多有 {100 * worst_flip:.2f}% 的状态被 LACS 改变决策（阈值 1%）")
    if args.sweep:
        wd, wp = sweep_compare(args.sweep)
        print(f"闭环里平均延迟最大相差 {wd:.2f}%（阈值 2%），丢弃率最大相差 {wp:.2f} 个百分点（阈值 1）")
        ok = worst_flip < 0.01 and wd < 2 and wp < 1
        print("结论：" + ("软标签只用单步成本" if ok else "软标签用单步 + LACS"))
    else:
        print("（加 --sweep 参数才能给出最终结论）")


if __name__ == "__main__":
    main()
