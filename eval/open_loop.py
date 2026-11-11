"""阶段 3 开环评测：把所有模型和公式基线放在同一份测试集上，用同一套指标打分。

"开环"的意思：每条测试状态单独打分，决策不影响后面的状态（闭环在阶段 5 做）。
Colab 上的 notebook 只负责让模型给每条状态选一个动作，写进 preds.csv；
指标全部在这里算，所以我的模型、论文原版模型、公式基线用的是完全同一套定义。

在仓库根目录运行（Mac 上）：
  python -m eval.open_loop runs/s3/*
每个参数是一个从 Drive 下载下来的运行目录，里面要有 config.json 和 preds.csv
（训练 notebook 还会有 preds_untrained.csv，就是训练前的同一个基座模型）。
论文原版基线的目录 paper_<时间>/ 下面每个模型一个子目录，会自动往下找一层；
没有 config.json 和 preds.csv 的目录（比如扫温度的 sweep_0.5b_*）跳过。
一个策略只答了一部分测试状态时（论文原版基线可以截取），只在它答了的那些状态上打分，"条数"一栏会变少。
结果写进 evidence/G3/open_loop.md 和 evidence/G3/open_loop.csv。

指标（单位：槽；论文式号见 README）：
  准确率      选的动作 = 单步最优（best）的比例
  多花延迟    选的动作的成本 − 最优动作的成本，只算合法动作
  AL          论文式 26：平均成本；超过截止期 10 槽的加罚 ρ = 10（论文表 I）；非法动作记 截止期 + ρ = 20
  PR          论文式 28–29：最优动作的 AL ÷ 这个策略的 AL
  丢弃率      论文式 27 的开环版：选的动作成本超过截止期，或者动作非法
  非法率      压缩格式：前 20 名候选里没有合法字母；论文格式：认不出动作，或者服务器编号超出台数
  本地比例    选本地执行的比例
  LBI         论文式 31：只看卸载的任务，按服务器分的条数算 Jain 指数，n = 真实台数；按台数分别算再平均
"""
import argparse
import csv
import json
import os
from collections import Counter

import numpy as np

from deploy.wire import state_from_dict
from sim.baselines import make_policy

RHO = 10.0             # 论文表 I 的超时惩罚
DEADLINE = 10.0        # 截止期（槽）
SPLITS = ("test_seen", "test_unseen", "friend_clean")
SPLIT_NAMES = {"test_seen": "见过的台数（3/5/7/9/11）", "test_unseen": "没见过的台数（4/6/8/10）",
               "friend_clean": "原作者干净测试集（245 条）"}
FORMULA_POLICIES = (("oracle_compact", "公式·压缩精度（学生上限）"), ("lacs", "LACS-greedy"),
                    ("psgreedy", "PS 感知贪心"), ("po2", "power-of-2"), ("random", "随机"))


# ===== 第 1 段：读数据和预测 =====
def load_split(data_dir, split):
    """读 data/s3/<split>.jsonl，返回行列表；文件不存在返回空列表。"""
    path = os.path.join(data_dir, split + ".jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_preds(path):
    """preds.csv → {(split, idx): 动作}；动作 −1 表示非法。"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["split"], int(r["idx"]))] = int(r["pred"])
    return out


def formula_preds(policy_name, data, seed=0):
    """公式基线在每条测试状态上的动作。oracle_compact = 压缩精度状态上的单步最优（学生能看到的信息的上限）。"""
    out = {}
    policy = None
    if policy_name != "oracle_compact":
        policy = make_policy(policy_name, seed)
    for split, rows in data.items():
        for i, r in enumerate(rows):
            if policy is None:
                c = r["costs_compact"]
                best = 0
                for a in range(len(c)):
                    if c[a] < c[best]:
                        best = a
                out[(split, i)] = best
            else:
                out[(split, i)] = int(policy(state_from_dict(r["state"])))
    return out


# ===== 第 2 段：指标 =====
def effective_cost(costs, a):
    """论文式 26 的单条成本：合法动作按成本，超过截止期加 ρ；非法动作记 截止期 + ρ。"""
    if a < 0 or a >= len(costs):
        return DEADLINE + RHO
    c = costs[a]
    if c > DEADLINE:
        c = c + RHO
    return c


def jain(counts):
    """Jain 指数：(Σn)² ÷ (台数 × Σn²)；一次都没卸载时返回 nan。"""
    total = 0.0
    sq = 0.0
    for n in counts:
        total += n
        sq += n * n
    if total == 0:
        return float("nan")
    return total * total / (len(counts) * sq)


def score(rows, preds, split):
    """一组行（同一个集合里的某些状态）上的全部指标。preds 里没有的状态不算（这个策略没答）。
    动作 −1 才是非法。这组行里一条都没答时返回 None。

    1. 逐条：是否命中最优、多花延迟、论文式 26 的成本、是否丢弃 / 非法 / 本地
    2. 汇总成百分比和平均值
    3. LBI：按服务器台数分组，各算一次 Jain 再平均
    4. 本地动作单独看：本地最优占比、本地召回（本地最优的状态里选了本地的比例）、
       本地精确（选了本地的状态里本地确实最优的比例）、卸载状态准确率（最优是某台服务器的状态里选对的比例）。
       分母是 0 时记 nan
    """
    answered = []
    for r in rows:
        if (split, r["_idx"]) in preds:
            answered.append(r)
    rows = answered
    n = len(rows)
    if n == 0:
        return None
    hit = 0
    regrets = []
    al = 0.0
    al_oracle = 0.0
    drops = 0
    invalid = 0
    local = 0
    best_local = 0
    local_hit = 0
    offload_hit = 0
    counts_by_k = {}
    for i, r in enumerate(rows):
        a = preds[(split, r["_idx"])]
        costs = r["costs"]
        best = r["best"]
        if best == 0:
            best_local += 1
            if a == 0:
                local_hit += 1
        elif a == best:
            offload_hit += 1
        legal = 0 <= a < r["n_actions"]
        if not legal:
            invalid += 1
        else:
            regrets.append(costs[a] - costs[best])
            if a == best:
                hit += 1
            if a == 0:
                local += 1
            else:
                k = r["servers"]
                if k not in counts_by_k:
                    counts_by_k[k] = [0] * k
                counts_by_k[k][a - 1] += 1
        ce = effective_cost(costs, a)
        al += ce
        al_oracle += effective_cost(costs, best)
        if ce > DEADLINE:
            drops += 1
    lbis = []
    for k in counts_by_k:
        v = jain(counts_by_k[k])
        if not np.isnan(v):
            lbis.append(v)
    return {
        "n": n,
        "acc_pct": 100.0 * hit / n,
        "regret": float(np.mean(regrets)) if regrets else float("nan"),
        "al": al / n,
        "pr_pct": 100.0 * al_oracle / al,
        "drop_pct": 100.0 * drops / n,
        "invalid_pct": 100.0 * invalid / n,
        "local_pct": 100.0 * local / n,
        "lbi_pct": 100.0 * float(np.mean(lbis)) if lbis else float("nan"),
        "best_local_pct": 100.0 * best_local / n,
        "local_recall_pct": 100.0 * local_hit / best_local if best_local else float("nan"),
        "local_precision_pct": 100.0 * local_hit / local if local else float("nan"),
        "offload_acc_pct": 100.0 * offload_hit / (n - best_local) if n > best_local else float("nan"),
    }


# ===== 第 3 段：收集所有策略 =====
def find_run_dirs(args):
    """命令行给的目录 → 真正的运行目录列表：本身有 config.json 和 preds.csv 的直接用；
    没有的往下找一层子目录；都没有的打印一行跳过。"""
    out = []
    for d in args:
        if not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, "config.json")) and os.path.exists(os.path.join(d, "preds.csv")):
            out.append(d)
            continue
        found = False
        for sub in sorted(os.listdir(d)):
            p = os.path.join(d, sub)
            if os.path.exists(os.path.join(p, "config.json")) and os.path.exists(os.path.join(p, "preds.csv")):
                out.append(p)
                found = True
        if not found:
            print("跳过（没有 config.json 和 preds.csv）：", d)
    return out


def run_name(run_dir):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("name", os.path.basename(run_dir.rstrip("/"))), cfg


def collect_policies(run_dirs, data):
    """返回 [(显示名, preds 字典)]：先放公式基线，再放每个运行目录里的模型（以及训练前的基座，每种基座只放一次）。"""
    policies = []
    for name, label in FORMULA_POLICIES:
        policies.append((label, formula_preds(name, data)))
    untrained_seen = set()
    for d in run_dirs:
        name, cfg = run_name(d)
        path_u = os.path.join(d, "preds_untrained.csv")
        base = cfg.get("model", "")
        if os.path.exists(path_u) and base not in untrained_seen:
            untrained_seen.add(base)
            policies.append(("未训练 " + cfg.get("model_tag", base), load_preds(path_u)))
        policies.append((name, load_preds(os.path.join(d, "preds.csv"))))
    return policies


# ===== 第 4 段：出表 =====
def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="运行目录（从 Drive 下载的），每个里面有 config.json 和 preds.csv")
    ap.add_argument("--data", default=os.path.join("data", "s3"))
    ap.add_argument("--out", default=os.path.join("evidence", "G3"))
    ap.add_argument("--title", default="阶段 3 开环评测", help="open_loop.md 第一行的标题")
    args = ap.parse_args()

    # 1. 读三份测试集，每行记下它在文件里的行号（preds.csv 用行号对应）
    data = {}
    for split in SPLITS:
        rows = load_split(args.data, split)
        for i, r in enumerate(rows):
            r["_idx"] = i
        if rows:
            data[split] = rows
    policies = collect_policies(find_run_dirs(args.runs), data)

    # 2. 每个策略 × 每个集合算一遍，再按服务器台数各算一遍
    table = []
    for label, preds in policies:
        for split, rows in data.items():
            m = score(rows, preds, split)
            if m is None:
                continue
            table.append(dict(policy=label, split=split, servers="全部", **m))
            ks = set()
            for r in rows:
                ks.add(r["servers"])
            for k in sorted(ks):
                sub = []
                for r in rows:
                    if r["servers"] == k:
                        sub.append(r)
                m = score(sub, preds, split)
                if m is not None:
                    table.append(dict(policy=label, split=split, servers=k, **m))

    # 2b. 再按终端数 N 各算一遍（原作者测试集没有 N，记 −1，跳过）；单独写一个 CSV，
    #     不和上面按台数的表混在一起（figures.py 按 (策略, 集合, 台数) 读 open_loop.csv）
    by_n = []
    for label, preds in policies:
        for split, rows in data.items():
            ns = set()
            for r in rows:
                if r.get("n_terminals", -1) > 0:
                    ns.add(r["n_terminals"])
            for nt in sorted(ns):
                sub = []
                for r in rows:
                    if r.get("n_terminals") == nt:
                        sub.append(r)
                m = score(sub, preds, split)
                if m is not None:
                    by_n.append(dict(policy=label, split=split, terminals=nt, **m))

    # 3. 写 CSV（全部数字）和 Markdown（给人看的表）
    os.makedirs(args.out, exist_ok=True)
    metric_fields = ["n", "acc_pct", "regret", "al", "pr_pct", "drop_pct", "invalid_pct", "local_pct", "lbi_pct",
                     "best_local_pct", "local_recall_pct", "local_precision_pct", "offload_acc_pct"]
    for fn, rows_out, key in (("open_loop.csv", table, "servers"), ("open_loop_by_n.csv", by_n, "terminals")):
        with open(os.path.join(args.out, fn), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["policy", "split", key] + metric_fields)
            w.writeheader()
            for row in rows_out:
                out = {}
                for k, v in row.items():
                    out[k] = round(v, 4) if isinstance(v, float) else v
                w.writerow(out)
    lines = ["# " + args.title, "",
             "由 `python -m eval.open_loop` 生成。单位：槽。AL、PR 按论文式 26–29，ρ = 10（论文表 I）。",
             "本地召回 = 本地最优的状态里选了本地的比例；本地精确 = 选了本地的状态里本地确实最优的比例；"
             "卸载准确率 = 最优是某台服务器的状态里选对的比例。分母为 0 记 –。", ""]
    for split in data:
        n_rows = len(data[split])
        n_local = 0
        for r in data[split]:
            if r["best"] == 0:
                n_local += 1
        lines += [f"## {SPLIT_NAMES[split]}", "",
                  f"{n_rows} 条，本地最优 {n_local} 条（{100.0 * n_local / n_rows:.2f}%）；"
                  f"从不选本地的策略准确率最多 {100.0 * (n_rows - n_local) / n_rows:.2f}%。", "",
                  "| 策略 | 条数 | 准确率 % | 多花延迟 | AL | PR % | 丢弃率 % | 非法率 % | 本地 % | "
                  "本地召回 % | 本地精确 % | 卸载准确率 % | LBI % |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in table:
            if row["split"] == split and row["servers"] == "全部":
                lines.append(f"| {row['policy']} | {row['n']} | {fmt(row['acc_pct'], 1)} | {fmt(row['regret'], 3)} | "
                             f"{fmt(row['al'], 3)} | {fmt(row['pr_pct'], 1)} | {fmt(row['drop_pct'], 1)} | "
                             f"{fmt(row['invalid_pct'], 1)} | {fmt(row['local_pct'], 1)} | "
                             f"{fmt(row['local_recall_pct'], 1)} | {fmt(row['local_precision_pct'], 1)} | "
                             f"{fmt(row['offload_acc_pct'], 1)} | {fmt(row['lbi_pct'], 1)} |")
        lines.append("")
    ks_all = set()
    for split in ("test_seen", "test_unseen"):
        for r in data.get(split, []):
            ks_all.add(r["servers"])
    ks_all = sorted(ks_all)
    lines += ["## 按服务器台数的 PR %（图 4 的数据）", "",
              "| 策略 | " + " | ".join(f"{k} 台" for k in ks_all) + " |",
              "|---|" + "---|" * len(ks_all)]
    for label, _ in policies:
        cells = []
        for k in ks_all:
            v = None
            for row in table:
                if row["policy"] == label and row["servers"] == k and row["split"] in ("test_seen", "test_unseen"):
                    v = row["pr_pct"]
            cells.append(fmt(v, 1))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    # 按台数、按终端数 N 的本地召回（见过 + 没见过的台数；N 两份集合各一张）
    lines += ["## 按服务器台数的本地召回 %", "",
              "| 策略 | " + " | ".join(f"{k} 台" for k in ks_all) + " |",
              "|---|" + "---|" * len(ks_all)]
    for label, _ in policies:
        cells = []
        for k in ks_all:
            v = None
            for row in table:
                if row["policy"] == label and row["servers"] == k and row["split"] in ("test_seen", "test_unseen"):
                    v = row["local_recall_pct"]
            cells.append(fmt(v, 1))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    for split in ("test_seen", "test_unseen"):
        ns_all = sorted({row["terminals"] for row in by_n if row["split"] == split})
        if not ns_all:
            continue
        for metric, name in (("pr_pct", "PR %"), ("local_recall_pct", "本地召回 %")):
            lines += [f"## 按终端数 N 的{name}（{SPLIT_NAMES[split]}）", "",
                      "| 策略 | " + " | ".join(f"N = {nt}" for nt in ns_all) + " |",
                      "|---|" + "---|" * len(ns_all)]
            for label, _ in policies:
                cells = []
                for nt in ns_all:
                    v = None
                    for row in by_n:
                        if row["policy"] == label and row["split"] == split and row["terminals"] == nt:
                            v = row[metric]
                    cells.append(fmt(v, 1))
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
            lines.append("")
    with open(os.path.join(args.out, "open_loop.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("写好了：", os.path.join(args.out, "open_loop.md"))


if __name__ == "__main__":
    main()
