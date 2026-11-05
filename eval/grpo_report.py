"""阶段 4 的汇总：原作者三个模型的完整动作分布（对照论文表 V），以及 GRPO 训练曲线。

在仓库根目录运行（Mac 上）：
  python -m eval.grpo_report runs/s4/*
参数是从 Drive 下载的阶段 4 运行目录：
  diagnose_<时间>/   下面 sft、grpo、score-grpo 三个子目录，每个里面有 dist.jsonl
  smoke_<时间>/ fixed_<时间>/ repo_<时间>/   里面有 summary.json、train_log.csv、val_log.json，正式训练的还有 dist.jsonl
  hard_grpo_<时间>/   B（从我们的硬标签 1.5B 出发）：同样有这三个文件，没有 dist.jsonl（压缩格式的动作分布看 preds.csv）
结果写进 evidence/G4/grpo_report.md 和 evidence/G4/grpo_curves.png。
开环指标（准确率、PR、LBI）仍用 python -m eval.open_loop 算，见运行文档。
"""
import argparse
import csv
import json
import math
import os

# 论文表 V（11 台服务器，306 个决策）：本地、服务器 1 … 11 的条数
PAPER_TABLE5 = {
    "SFT-1.5B": [41, 59, 35, 27, 28, 26, 4, 0, 14, 16, 2, 8],
    "GRPO-1.5B": [12, 192, 43, 10, 19, 10, 1, 0, 4, 10, 0, 5],
}


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def summarize_dist(rows):
    """dist.jsonl 的行 → 一组数字。

    1. 整句最大概率的动作（T=1）、贪心答案，各自的准确率，以及两者相同的比例
    2. 合法动作概率之和（T=1）；温度 0.7 采样时抽到最优动作的概率（期望准确率）
    3. 服务器 1 的平均概率（T=1）
    """
    n = len(rows)
    seq_hit = 0
    greedy_hit = 0
    agree = 0
    mass = 0.0
    exp07 = 0.0
    p_s1 = 0.0
    for r in rows:
        lp1 = r["logp_T1.0"]
        best_seq = 0
        for a in range(len(lp1)):
            if lp1[a] > lp1[best_seq]:
                best_seq = a
        if best_seq == r["best"]:
            seq_hit += 1
        if r["greedy"] == r["best"]:
            greedy_hit += 1
        if r["greedy"] == best_seq:
            agree += 1
        for x in lp1:
            mass += math.exp(x)
        exp07 += math.exp(r["logp_T0.7"][r["best"]])
        p_s1 += math.exp(lp1[1])
    return {"n": n, "seq_acc": 100.0 * seq_hit / n, "greedy_acc": 100.0 * greedy_hit / n,
            "agree": 100.0 * agree / n, "mass": 100.0 * mass / n, "exp_acc_07": 100.0 * exp07 / n,
            "p_s1": 100.0 * p_s1 / n}


def table5_shares(rows):
    """11 台的行：温度 0.7 采样时每个动作的期望占比，和贪心答案的占比（%）。"""
    share07 = [0.0] * 12
    share_g = [0.0] * 12
    n = 0
    for r in rows:
        if r["servers"] != 11:
            continue
        n += 1
        p = []
        total = 0.0
        for x in r["logp_T0.7"]:
            p.append(math.exp(x))
            total += math.exp(x)
        for a in range(12):
            share07[a] += p[a] / total
        if r["greedy"] >= 0:
            share_g[r["greedy"]] += 1
    if n == 0:
        return None
    for a in range(12):
        share07[a] = 100.0 * share07[a] / n
        share_g[a] = 100.0 * share_g[a] / n
    return {"n": n, "t07": share07, "greedy": share_g}


def find_runs(args):
    """返回 (diagnose 目录列表, 训练目录列表)。"""
    diag = []
    train = []
    for d in args:
        if not os.path.isdir(d):
            continue
        base = os.path.basename(d.rstrip("/"))
        if base.startswith("diagnose_"):
            diag.append(d)
        elif os.path.exists(os.path.join(d, "train_log.csv")):
            train.append(d)
    return diag, train


def dist_lines(title_rows):
    """[(名字, dist 行列表)] → Markdown 表格的行，按集合分开。"""
    lines = []
    for split, split_name in (("friend_clean", "原作者干净测试集"), ("test_seen", "test_seen 里 11 台")):
        lines += [f"### {split_name}", "",
                  "| 模型 | 条数 | 整句最大概率 准确率 % | 贪心 准确率 % | 两者相同 % | 合法动作概率和 % | 0.7 采样期望准确率 % | 服务器 1 平均概率 % |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, rows in title_rows:
            sub = []
            for r in rows:
                if r["split"] == split:
                    sub.append(r)
            if not sub:
                continue
            s = summarize_dist(sub)
            lines.append(f"| {name} | {s['n']} | {s['seq_acc']:.1f} | {s['greedy_acc']:.1f} | {s['agree']:.1f} | "
                         f"{s['mass']:.1f} | {s['exp_acc_07']:.1f} | {s['p_s1']:.1f} |")
        lines.append("")
    return lines


def table5_lines(title_rows):
    head = "| 来源 | 条数 | 本地 | " + " | ".join(f"S{k}" for k in range(1, 12)) + " |"
    lines = ["### 动作占比 %（11 台，对照论文表 V）", "", head, "|---|---|" + "---|" * 12]
    for name, counts in PAPER_TABLE5.items():
        total = 0
        for c in counts:
            total += c
        cells = []
        for c in counts:
            cells.append(f"{100.0 * c / 306:.1f}")
        lines.append(f"| 论文表 V {name} | 306（合计 {total}） | " + " | ".join(cells) + " |")
    for name, rows in title_rows:
        sub = []
        for r in rows:
            if r["split"] == "test_seen":
                sub.append(r)
        t = table5_shares(sub)
        if t is None:
            continue
        for kind, label in (("t07", "0.7 采样期望"), ("greedy", "贪心")):
            cells = []
            for x in t[kind]:
                cells.append(f"{x:.1f}")
            lines.append(f"| {name} · {label} | {t['n']} | " + " | ".join(cells) + " |")
    lines += ["", "论文表 V 的 SFT-1.5B 一行只有 260 个决策，剩下 46 个应该是认不出的回答；占比仍按 306 算。", ""]
    return lines


def read_train_log(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                out[k] = float(v)
            rows.append(out)
    return rows


def plot_curves(train_dirs, path):
    """四张小图：采样回答的平均奖励、奖励全相同的组的比例、采样里服务器 1 的占比、验证集贪心准确率。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = (("reward_mean", "采样回答的平均奖励"), ("zero_std_frac", "奖励全相同的组的比例"),
              ("s1_pct", "采样里选服务器 1 的比例 %"), (None, "验证集贪心准确率 %"))
    for d in train_dirs:
        name = os.path.basename(d.rstrip("/"))
        log = read_train_log(os.path.join(d, "train_log.csv"))
        steps = []
        for r in log:
            steps.append(r["step"])
        for ax, (key, title) in zip(axes.flat, panels):
            if key is None:
                with open(os.path.join(d, "val_log.json"), encoding="utf-8") as f:
                    val = json.load(f)
                xs = []
                ys = []
                for v in val:
                    xs.append(v["step"])
                    ys.append(v["acc_pct"])
                ax.plot(xs, ys, marker="o", label=name)
            else:
                ys = []
                for r in log:
                    ys.append(r[key])
                ax.plot(steps, ys, label=name, alpha=0.8)
            ax.set_title(title)
            ax.set_xlabel("步")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=os.path.join("evidence", "G4"))
    args = ap.parse_args()
    diag, train = find_runs(args.runs)
    os.makedirs(args.out, exist_ok=True)

    named = []
    for d in diag:
        for m in ("sft", "grpo", "score-grpo"):
            p = os.path.join(d, m, "dist.jsonl")
            if os.path.exists(p):
                named.append(("论文原版 " + m, read_jsonl(p)))
    for d in train:
        p = os.path.join(d, "dist.jsonl")
        if os.path.exists(p):
            named.append((os.path.basename(d.rstrip("/")), read_jsonl(p)))

    lines = ["# 阶段 4：动作分布和 GRPO 训练", "",
             "由 `python -m eval.grpo_report` 生成。整句概率含结尾的 <|im_end|>；0.7 是原作者评测脚本的采样温度。", ""]
    lines += dist_lines(named)
    lines += table5_lines(named)
    if train:
        lines += ["## 训练", "", "| 运行 | 步数 | 用时（分钟） | 验证集准确率 % 第 0 步 → 最后 | 最后 10 步 全相同组比例 | 最后 10 步 非法 % |",
                  "|---|---|---|---|---|---|"]
        for d in train:
            with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
                s = json.load(f)
            log = read_train_log(os.path.join(d, "train_log.csv"))
            last = log[-10:]
            z = 0.0
            bad = 0.0
            for r in last:
                z += r["zero_std_frac"]
                bad += r["bad_server_pct"] + r["unparsed_pct"]
            val = s["val_log"]
            lines.append(f"| {os.path.basename(d.rstrip('/'))} | {len(log)} | {s.get('minutes_train', '–')} | "
                         f"{val[0]['acc_pct']} → {val[-1]['acc_pct']} | {z / len(last):.2f} | {bad / len(last):.1f} |")
        lines.append("")
        plot_curves(train, os.path.join(args.out, "grpo_curves.png"))
        lines += ["![训练曲线](grpo_curves.png)", ""]
    with open(os.path.join(args.out, "grpo_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("写好了：", os.path.join(args.out, "grpo_report.md"))


if __name__ == "__main__":
    main()
