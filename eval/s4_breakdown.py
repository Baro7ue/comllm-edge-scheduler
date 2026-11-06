"""阶段 4 分析里用到、但 open_loop.md 和 grpo_report.md 没有直接给出的数字。

在仓库根目录运行（Mac 上）：
  python -m eval.s4_breakdown --diagnose runs/s4/diagnose_* --fixed runs/s4/fixed_* --repo runs/s4/repo_* \
      --hard runs/s3/hard_1.5b_* --hard-grpo runs/s4/hard_grpo_*
结果写进 evidence/G4/breakdown.md，四节：
  1. 原作者三个模型发给服务器 1 的比例：0.7 采样期望 / 贪心 / 整句概率最大，和公式最优对照，分台数
  2. 原作者 sft 标签：动作分布，以及"卸载到边缘服务器"后面第一个数字的分布
  3. fixed / repo：本地最优的题占多花延迟的多少；repo 回答的格式问题，严格解析后的准确率
  4. B：hard_1.5b 加 GRPO 前后答案变了的题
"""
import argparse
import csv
import json
import math
import os
import re
from collections import Counter

from eval.open_loop import effective_cost, load_split
from sim.friend_data import DEFAULT_DIR

MODELS = ("sft", "grpo", "score-grpo")
ACTION_RE = re.compile(r"本地执行|卸载到边缘服务器(\d+)")


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def read_preds(path):
    """preds.csv → {(split, idx): 整行字典}"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["split"], int(r["idx"]))] = r
    return out


def pct(a, b):
    if b == 0:
        return "–"
    return f"{100.0 * a / b:.1f}"


# ===== 第 1 节 =====
def server1_section(diag_dir):
    """原作者三个模型在干净测试集上发给服务器 1 的比例。

    1. 每个模型读 dist.jsonl 里 friend_clean 的行
    2. 每行：0.7 采样期望 = 服务器 1 的 exp(logp_T0.7) ÷ 所有合法动作的和；
       贪心 = greedy 是否为 1；整句最大 = logp_T1.0 最大的动作是否为 1；公式最优 = best 是否为 1
    3. 全部一起算一行，再按台数各算一行
    """
    lines = ["## 1. 原作者模型发给服务器 1 的比例（原作者干净测试集）", "",
             "| 模型 | 台数 | 条数 | 0.7 采样期望 % | 贪心 % | 整句概率最大 % | 公式最优 % |", "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        rows = []
        for r in read_jsonl(os.path.join(diag_dir, m, "dist.jsonl")):
            if r["split"] == "friend_clean":
                rows.append(r)
        groups = {"全部": rows}
        for r in rows:
            key = str(r["servers"])
            if key not in groups:
                groups[key] = []
            groups[key].append(r)
        order = ["全部"] + sorted([k for k in groups if k != "全部"], key=int)
        for key in order:
            g = groups[key]
            sampled = 0.0
            greedy = 0
            argmax = 0
            best = 0
            for r in g:
                lp = r["logp_T0.7"]
                z = 0.0
                for x in lp:
                    z += math.exp(x)
                sampled += math.exp(lp[1]) / z
                if r["greedy"] == 1:
                    greedy += 1
                lp1 = r["logp_T1.0"]
                top = 0
                for a in range(len(lp1)):
                    if lp1[a] > lp1[top]:
                        top = a
                if top == 1:
                    argmax += 1
                if r["best"] == 1:
                    best += 1
            n = len(g)
            lines.append(f"| {m} | {key} | {n} | {pct(sampled, n)} | {pct(greedy, n)} | {pct(argmax, n)} | {pct(best, n)} |")
    return lines


# ===== 第 2 节 =====
def label_section(friend_dir):
    """原作者 sft 训练数据的标签分布。

    1. 读 sft_train_data_610.json，每条从 output 里找动作名
    2. 统计动作（本地 / 服务器 k）的条数
    3. 统计"卸载到边缘服务器"后面第一个数字（本地单独一类）的条数
    """
    with open(os.path.join(friend_dir, "sft_train_data_610.json"), encoding="utf-8") as f:
        data = json.load(f)
    actions = Counter()
    first = Counter()
    for e in data:
        m = ACTION_RE.search(str(e["output"]))
        if m is None:
            actions["认不出"] += 1
            first["认不出"] += 1
        elif m.group(0) == "本地执行":
            actions["本地"] += 1
            first["本地"] += 1
        else:
            actions["服务器" + m.group(1)] += 1
            first["数字 " + m.group(1)[0]] += 1
    n = len(data)
    lines = ["## 2. 原作者 sft 标签（610 条）", "", "| 动作 | 占比 % |", "|---|---|"]
    for k, v in actions.most_common():
        lines.append(f"| {k} | {pct(v, n)} |")
    lines += ["", "| 动作名第一个区分字 | 占比 % |", "|---|---|"]
    for k, v in first.most_common():
        lines.append(f"| {k} | {pct(v, n)} |")
    return lines


# ===== 第 3 节 =====
def canonical(a):
    if a == 0:
        return "本地执行"
    return f"卸载到边缘服务器{a}"


def grpo_a_section(data, runs):
    """fixed / repo：从不选本地的代价，以及回答格式。

    1. 每个运行读 preds.csv（有 raw 一列 = 模型原话）
    2. 每个测试集：本地最优的条数、模型选本地的条数、本地最优的题占全部多花延迟的比例
    3. raw 和标准动作名不完全一样的条数；把这些算非法后的准确率（严格解析）
    """
    lines = ["## 3. 从原作者 sft 起跑的 GRPO（贪心）", "",
             "| 运行 | 测试集 | 条数 | 本地最优 % | 选本地 % | 本地最优的题占多花延迟 % | 回答不规范 条 | 准确率 宽松 % | 准确率 严格 % |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, run_dir in runs:
        preds = read_preds(os.path.join(run_dir, "preds.csv"))
        for split in ("test_seen", "friend_clean"):
            n = 0
            local_best = 0
            local_pick = 0
            extra_all = 0.0
            extra_local = 0.0
            odd = 0
            acc = 0
            acc_strict = 0
            for (s, i), p in preds.items():
                if s != split:
                    continue
                row = data[split][i]
                a = int(p["pred"])
                b = row["best"]
                extra = effective_cost(row["costs"], a) - effective_cost(row["costs"], b)
                n += 1
                extra_all += extra
                if b == 0:
                    local_best += 1
                    extra_local += extra
                if a == 0:
                    local_pick += 1
                ok = p.get("raw", "").strip() == canonical(a)
                if not ok:
                    odd += 1
                if a == b:
                    acc += 1
                    if ok:
                        acc_strict += 1
            lines.append(f"| {name} | {split} | {n} | {pct(local_best, n)} | {pct(local_pick, n)} | "
                         f"{pct(extra_local, extra_all)} | {odd} | {pct(acc, n)} | {pct(acc_strict, n)} |")
    return lines


# ===== 第 4 节 =====
def grpo_b_section(data, hard_dir, grpo_dir):
    """B：hard_1.5b 和它加 GRPO 之后，答案变了的题。

    1. 两份 preds.csv 按 (split, idx) 对齐
    2. 每个测试集数：hard 对 GRPO 错、hard 错 GRPO 对；前者里改选本地的条数
    3. 前者多花的延迟（评测精度成本）的中位数和最大值；压缩精度成本差 < 0.05 槽的条数
    """
    hard = read_preds(os.path.join(hard_dir, "preds.csv"))
    grpo = read_preds(os.path.join(grpo_dir, "preds.csv"))
    lines = ["## 4. B：hard_1.5b 加 GRPO 前后（部署规则）", "",
             "| 测试集 | 条数 | hard 对 → GRPO 错 | hard 错 → GRPO 对 | 其中改选本地 | 多花延迟 中位 / 最大（槽） | 压缩精度差 < 0.05 槽 | 选本地 % hard → GRPO | 本地最优 % |",
             "|---|---|---|---|---|---|---|---|---|"]
    for split in ("test_seen", "test_unseen", "friend_clean"):
        n = 0
        worse = 0
        better = 0
        to_local = 0
        near = 0
        gaps = []
        local_h = 0
        local_g = 0
        local_best = 0
        for key, pg in grpo.items():
            if key[0] != split or key not in hard:
                continue
            row = data[split][key[1]]
            b = row["best"]
            h = int(hard[key]["pred"])
            g = int(pg["pred"])
            n += 1
            if h == 0:
                local_h += 1
            if g == 0:
                local_g += 1
            if b == 0:
                local_best += 1
            if h == b and g != b:
                worse += 1
                if g == 0:
                    to_local += 1
                gaps.append(row["costs"][g] - row["costs"][b])
                if row["costs_compact"][g] - row["costs_compact"][b] < 0.05:
                    near += 1
            if h != b and g == b:
                better += 1
        gaps.sort()
        gap_text = "–"
        if gaps:
            gap_text = f"{gaps[len(gaps) // 2]:.3f} / {gaps[-1]:.3f}"
        lines.append(f"| {split} | {n} | {worse} | {better} | {to_local} | {gap_text} | {near} | "
                     f"{pct(local_h, n)} → {pct(local_g, n)} | {pct(local_best, n)} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnose", required=True)
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--hard", required=True)
    ap.add_argument("--hard-grpo", required=True)
    ap.add_argument("--data", default=os.path.join("data", "s3"))
    ap.add_argument("--friend", default=DEFAULT_DIR)
    ap.add_argument("--out", default=os.path.join("evidence", "G4"))
    args = ap.parse_args()

    data = {}
    for split in ("test_seen", "test_unseen", "friend_clean"):
        data[split] = load_split(args.data, split)
    lines = ["# 阶段 4 补充数字", "", "由 `python -m eval.s4_breakdown` 生成。单位：槽。", ""]
    lines += server1_section(args.diagnose) + [""]
    lines += label_section(args.friend) + [""]
    lines += grpo_a_section(data, [("fixed", args.fixed), ("repo", args.repo)]) + [""]
    lines += grpo_b_section(data, args.hard, args.hard_grpo) + [""]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "breakdown.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("写好了：", path)


if __name__ == "__main__":
    main()
