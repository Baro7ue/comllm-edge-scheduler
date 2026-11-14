"""Draw the four README figures in English and Chinese from the archived results.

  python -m eval.readme_figures      # writes docs/media/en/*.png, docs/media/zh/*.png and docs/media/numbers.md

Every value plotted here is read from a file that is already in the repository, and the same values are
written to docs/media/numbers.md with their source, so README numbers can be checked against them.
The script does not touch figures/ (that is eval.figures).

1. stability.png    real-time closed loop on the Jetson: drop rate against the number of terminals
2. latency.png      one decision on the Jetson, 500 requests per model: p50 and p99
3. open_loop.png    paper's PR on the three fixed test sets: students, reproduced 7B, original-author models
4. closed_loop.png  virtual-time closed loop, 3 servers: Colab (bf16) and Jetson (Q8_0), drop rate and mean delay
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from eval import figures as F  # noqa: E402

OUT = os.path.join("docs", "media")
INK, MUTED, GRID = "#1f1f1f", "#5b5a56", "#e6e5e1"
BLUE, BLUE_L = "#2f6db5", "#8fb3de"
GREEN, GREEN_L = "#2e8b57", "#93c9a8"
ORANGE, RED = "#d9822b", "#c0392b"
GREY, GREY_L = "#7d7d7d", "#b9b9b9"
SPLIT_COLORS = ("#2f6db5", "#79a6d9", "#d9822b")

LATENCY = os.path.join("evidence", "G6", "jetson_open", "latency_coverage.csv")
OPEN = os.path.join("evidence", "G6", "open_loop.csv")
COLAB_STUDENTS = os.path.join("runs", "20261112T064150Z_s6_colab_loop", "results.csv")
COLAB_7B = os.path.join("runs", "20261113T104604Z_s7_colab_closed_loop", "results.csv")
JETSON_VIRTUAL = [os.path.join("runs", d, "results.csv") for d in
                  ("20261107T071637Z_s5_virtual_e4a+e4b+e5", "20261108T044444Z_s5_virtual_e4ps_a",
                   "20261108T094747Z_s5_virtual_e4ps_b")]

T = {
    "en": {
        "font": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "terminals": "terminals N (log scale)", "drop": "tasks dropped (%)", "delay": "mean delay of completed tasks (slots)",
        "stab_title": "Real-time closed loop on the Jetson, 3 servers",
        "stab_note": "Decision time counts toward task delay. Mean of 2 seeds, 3 min per point (1 seed for the paper-format 1.5B).",
        "formula": "one-step formula (no LLM)", "main": "our 0.5B, Q8_0 (deployed)", "shed": "our 0.5B + overload shedding",
        "soft15": "our 1.5B, Q8_0", "paper15": "original-author 1.5B, paper format",
        "lat_title": "One decision on the Jetson Orin Nano: end-to-end, measured on the environment node",
        "lat_x": "milliseconds (log scale)", "p50": "p50", "p99": "p99",
        "lat_note": "500 requests per model after 10 warm-up requests, servers 3/7/11 in turn. Students: compact prompt, one token, prefix cache. Original-author models: paper format, generated answer.",
        "open_title": "Open loop: the paper's PR on three fixed test sets",
        "open_x": "PR = oracle cost / policy cost (%)",
        "splits": ("seen server counts (3010 states, 48% local-optimal)", "unseen server counts (2408, 46%)",
                   "authors' cleaned set (245, 2.4%)"),
        "open_note": "Colab, full precision, greedy decoding. Reproduced 7B: the paper's GRPO recipe from the authors' SFT-7B; it is not the original COMLLM-7B.",
        "closed_title": "Virtual-time closed loop, 3 servers: Colab bf16 (solid) and Jetson Q8_0 (hollow)",
        "closed_note": "Mean of seeds 0 and 1, 1000 slots per point. Mean delay counts completed tasks only; read it with the drop rate. The 7B models ran on Colab only.",
        "f_paper": "paper formula", "f_ps": "PS-aware formula",
        "s_main": "our 0.5B, paper teacher (deployed)", "s_ps": "our 0.5B, PS teacher",
        "r7_03": "reproduced 7B, λ = 0.3", "r7_0": "reproduced 7B, λ = 0",
        "models": {
            "main_0.5b": "our 0.5B (deployed)", "soft_1.5b": "our 1.5B soft labels", "hard_1.5b": "our 1.5B hard labels",
            "ps_0.5b": "our 0.5B, PS teacher", "ps_soft_1.5b": "our 1.5B soft, PS teacher",
            "ps_hard_1.5b": "our 1.5B hard, PS teacher", "paper_sft_1.5b": "original-author 1.5B SFT",
            "paper_grpo_1.5b": "original-author 1.5B GRPO", "paper_score_grpo_1.5b": "original-author 1.5B score-GRPO",
            "7b_03": "reproduced 7B, λ = 0.3", "7b_0": "reproduced 7B, λ = 0", "sft7b": "authors' SFT-7B (start point)",
        },
    },
    "zh": {
        "font": ["PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"],
        "terminals": "终端数 N（对数刻度）", "drop": "丢弃的任务（%）", "delay": "完成任务的平均延迟（槽）",
        "stab_title": "Jetson 上的实时闭环，3 台服务器",
        "stab_note": "决策耗时算进任务延迟。每点 3 分钟、2 个种子的平均（论文格式 1.5B 只有 1 个种子）。",
        "formula": "单步公式（不用 LLM）", "main": "我们的 0.5B，Q8_0（部署）", "shed": "我们的 0.5B + 超载分流",
        "soft15": "我们的 1.5B，Q8_0", "paper15": "原作者 1.5B，论文格式",
        "lat_title": "Jetson Orin Nano 上的单次决策：环境节点量到的端到端时间",
        "lat_x": "毫秒（对数刻度）", "p50": "p50", "p99": "p99",
        "lat_note": "每个模型 10 次热身后计 500 次，3/7/11 台服务器轮流。学生：压缩提示词、1 个 token、前缀缓存。原作者模型：论文格式，生成回答。",
        "open_title": "开环：三套固定测试集上的论文 PR",
        "open_x": "PR = 最优成本 ÷ 策略成本（%）",
        "splits": ("见过的台数（3010 条，48% 本地最优）", "没见过的台数（2408 条，46%）", "原作者干净测试集（245 条，2.4%）"),
        "open_note": "Colab，原精度，贪心解码。复现的 7B：从原作者 SFT-7B 出发按论文的 GRPO 方法训练，不等于原版 COMLLM-7B。",
        "closed_title": "虚拟时间闭环，3 台服务器：Colab bf16（实心）和 Jetson Q8_0（空心）",
        "closed_note": "种子 0、1 的平均，每点 1000 槽。平均延迟只算完成的任务，要和丢弃率一起看。7B 只在 Colab 上跑。",
        "f_paper": "论文公式", "f_ps": "PS 感知公式",
        "s_main": "我们的 0.5B，论文公式老师（部署）", "s_ps": "我们的 0.5B，PS 老师",
        "r7_03": "复现的 7B，λ = 0.3", "r7_0": "复现的 7B，λ = 0",
        "models": {
            "main_0.5b": "我们的 0.5B（部署）", "soft_1.5b": "我们的 1.5B 软标签", "hard_1.5b": "我们的 1.5B 硬标签",
            "ps_0.5b": "我们的 0.5B，PS 老师", "ps_soft_1.5b": "我们的 1.5B 软标签，PS 老师",
            "ps_hard_1.5b": "我们的 1.5B 硬标签，PS 老师", "paper_sft_1.5b": "原作者 1.5B SFT",
            "paper_grpo_1.5b": "原作者 1.5B GRPO", "paper_score_grpo_1.5b": "原作者 1.5B score-GRPO",
            "7b_03": "复现的 7B，λ = 0.3", "7b_0": "复现的 7B，λ = 0", "sft7b": "原作者 SFT-7B（复现的起点）",
        },
    },
}

OPEN_POLICIES = [("main_0.5b", "0.5b_soft_T0.3"), ("soft_1.5b", "1.5b_soft_T0.3"), ("hard_1.5b", "1.5b_hard"),
                 ("7b_03", "COMLLM-7B复现"), ("7b_0", "GRPO-7B复现（同奖励λ=0）"), ("sft7b", "SFT-7B训练前基线"),
                 ("paper_sft_1.5b", "论文原版 sft"), ("paper_grpo_1.5b", "论文原版 grpo"),
                 ("paper_score_grpo_1.5b", "论文原版 score-grpo")]
SPLITS = ("test_seen", "test_unseen", "friend_clean")
NUMBER_ONLY_POLICIES = [("ps_0.5b", "0.5b_soft_T0.3_ps"), ("ps_soft_1.5b", "1.5b_soft_T0.3_ps"), ("ps_hard_1.5b", "1.5b_hard_ps")]


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def note(fig, text):
    fig.text(0.01, 0.01, text, fontsize=7.8, color=MUTED, ha="left", va="bottom", wrap=True)


# ===== data =====
def stability_data():
    """Real-time closed loop (stage 5, experiment 1): (model, shedding) -> {N: [rows]}."""
    src = F.discover()
    rows = F.load_rows(src["e1"])
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["model_key"], float(r["shed_ms"]) > 0)][int(r["n"])].append(r)
    return groups, src["e1"]


def latency_data():
    return read(LATENCY)


def open_data():
    rows = read(OPEN)
    out = {}
    for key, policy in OPEN_POLICIES + NUMBER_ONLY_POLICIES:
        for split in SPLITS:
            hit = [r for r in rows if r["policy"] == policy and r["split"] == split and r["servers"] == "全部"]
            assert len(hit) == 1, (policy, split)
            out[(key, split)] = hit[0]
    return out


def closed_data():
    """(policy, platform) -> {N: (mean delay, drop %)} for 3 servers, seeds 0 and 1, sequential, 1000 slots."""
    want = {("llm:main_0.5b", "colab"), ("llm:ps_0.5b", "colab"), ("llm:main_0.5b", "jetson"),
            ("llm:ps_0.5b", "jetson"), ("greedy", "jetson"), ("psgreedy", "jetson"),
            ("llm:comllm_7b_repro", "colab"), ("llm:grpo_7b_repro", "colab")}
    cells = defaultdict(list)
    for path, platform in [(COLAB_STUDENTS, "colab"), (COLAB_7B, "colab")] + [(p, "jetson") for p in JETSON_VIRTUAL]:
        for r in read(path):
            if (r["policy"], platform) not in want:
                continue
            if r["servers"] != "3" or r["mode"] != "sequential" or r["slots"] != "1000" or r["seed"] not in ("0", "1"):
                continue
            if int(r["n"]) not in (20, 40, 60):
                continue
            cells[(r["policy"], platform, int(r["n"]))].append(r)
    out = defaultdict(dict)
    for (pol, platform, n), rs in cells.items():
        assert len(rs) == 2, (pol, platform, n, len(rs))
        out[(pol, platform)][n] = (F.mean([float(r["mean_delay"]) for r in rs]),
                                    100 * F.mean([float(r["drop_rate"]) for r in rs]))
    return out


# ===== figures =====
def fig_stability(lang, groups):
    t = T[lang]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    lines = [(("formula", False), t["formula"], GREY, "s", "--"), (("main_0.5b", False), t["main"], BLUE, "o", "-"),
             (("main_0.5b", True), t["shed"], BLUE_L, "o", "--"), (("soft_1.5b", False), t["soft15"], GREEN, "^", "-")]
    for key, label, color, marker, ls in lines:
        by_n = groups[key]
        ns = sorted(by_n)
        ys = [100 * F.mean([float(r["drop_rate"]) for r in by_n[n]]) for n in ns]
        ax.plot(ns, ys, color=color, marker=marker, linestyle=ls, linewidth=2, markersize=5, label=label)
    p = groups[("paper_sft_1.5b", False)][1]
    y = 100 * F.mean([float(r["drop_rate"]) for r in p])
    ax.plot([1], [y], color=RED, marker="X", markersize=9, linestyle="none", label=t["paper15"])
    ax.annotate(f"{y:.1f}%", (1, y), xytext=(8, -4), textcoords="offset points", fontsize=8, color=RED, va="top")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 7, 10, 20, 40, 80])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(-3, 103)
    style(ax, t["stab_title"], t["terminals"], t["drop"])
    ax.legend(frameon=False, fontsize=8, loc="center right")
    note(fig, t["stab_note"])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_latency(lang, rows):
    t = T[lang]
    order = ["main_0.5b", "ps_0.5b", "soft_1.5b", "hard_1.5b", "ps_soft_1.5b", "ps_hard_1.5b",
             "paper_sft_1.5b", "paper_grpo_1.5b", "paper_score_grpo_1.5b"]
    by = {r["model_key"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, key in enumerate(order):
        r = by[key]
        p50, p99 = float(r["p50_ms"]), float(r["p99_ms"])
        color = RED if key.startswith("paper_") else (GREEN if key.startswith("ps_") else BLUE)
        ax.barh(i, p50, color=color, height=0.6)
        ax.plot([p50, p99], [i, i], color=INK, linewidth=1)
        ax.plot([p99], [i], marker="|", color=INK, markersize=9)
        ax.text(p99 * 1.08, i, f"{p50:.0f} / {p99:.0f} ms", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([t["models"][k] for k in order], fontsize=8.5)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(20, 4000)
    ax.set_xticks([20, 50, 100, 200, 500, 1000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    style(ax, t["lat_title"], t["lat_x"], "")
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MUTED), Line2D([], [], color=INK, marker="|", markersize=9)]
    ax.legend(handles, [t["p50"], t["p99"]], frameon=False, fontsize=8, loc="upper right")
    note(fig, t["lat_note"])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def fig_open(lang, data):
    t = T[lang]
    keys = [k for k, _ in OPEN_POLICIES]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    h = 0.26
    for j, split in enumerate(SPLITS):
        ys = [i + (j - 1) * h for i in range(len(keys))]
        vals = [float(data[(k, split)]["pr_pct"]) for k in keys]
        ax.barh(ys, vals, height=h, color=SPLIT_COLORS[j], label=t["splits"][j])
        for y, v in zip(ys, vals):
            ax.text(v + 0.3, y, f"{v:.2f}", va="center", fontsize=6.8, color=INK)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([t["models"][k] for k in keys], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(60, 104)
    for sep in (2.5, 5.5):
        ax.axhline(sep, color=GRID, linewidth=1)
    style(ax, t["open_title"], t["open_x"], "")
    ax.grid(False, axis="y")
    ax.legend(frameon=False, fontsize=7.8, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=1)
    note(fig, t["open_note"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_closed(lang, data):
    t = T[lang]
    series = [("greedy", "jetson", t["f_paper"], GREY, "s", "--", True),
              ("psgreedy", "jetson", t["f_ps"], GREEN_L, "s", "--", True),
              ("llm:main_0.5b", "colab", t["s_main"], BLUE, "o", "-", True),
              ("llm:main_0.5b", "jetson", None, BLUE, "o", "none", False),
              ("llm:ps_0.5b", "colab", t["s_ps"], GREEN, "^", "-", True),
              ("llm:ps_0.5b", "jetson", None, GREEN, "^", "none", False),
              ("llm:comllm_7b_repro", "colab", t["r7_03"], ORANGE, "D", "-", True),
              ("llm:grpo_7b_repro", "colab", t["r7_0"], RED, "v", "-", True)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    ns = [20, 40, 60]
    for pol, platform, label, color, marker, ls, filled in series:
        cell = data[(pol, platform)]
        for ax, idx in ((axes[0], 1), (axes[1], 0)):
            ys = [cell[n][idx] for n in ns]
            ax.plot(ns, ys, color=color, marker=marker, linestyle=ls, linewidth=2 if ls != "none" else 0,
                    markersize=8 if not filled else 5, markerfacecolor=color if filled else "white",
                    markeredgewidth=1.4, label=label)
    style(axes[0], t["drop"], t["terminals"].split("（")[0].split(" (")[0], "%")
    style(axes[1], t["delay"], t["terminals"].split("（")[0].split(" (")[0], "")
    for ax in axes:
        ax.set_xticks(ns)
    axes[0].set_ylim(-3, 90)
    handles, labels = axes[0].get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
    fig.legend([h for h, _ in keep], [l for _, l in keep], loc="upper center", bbox_to_anchor=(0.5, 0.17),
               ncol=3, frameon=False, fontsize=8)
    fig.suptitle(t["closed_title"], x=0.01, ha="left", fontsize=10.5, color=INK)
    note(fig, t["closed_note"])
    fig.tight_layout(rect=(0, 0.2, 1, 0.96))
    return fig


# ===== numbers =====
def write_numbers(groups, e1_dirs, lat, open_rows, closed):
    lines = ["# Numbers behind the README figures", "",
             "Generated by `python -m eval.readme_figures`. Every value is read from the file named in the last column.", "",
             "| What | Value | Source |", "|---|---|---|"]
    src_e1 = ", ".join(os.path.basename(d) for d in e1_dirs)
    for key in sorted(groups):
        for n in sorted(groups[key]):
            rs = groups[key][n]
            lines.append(f"| real-time drop rate, {key[0]}{' + shedding' if key[1] else ''}, N = {n} | "
                         f"{100 * F.mean([float(r['drop_rate']) for r in rs]):.1f}% "
                         f"(mean delay {F.mean([float(r['mean_delay']) for r in rs]):.2f} slots, {len(rs)} seed(s)) | {src_e1} |")
    for r in lat:
        lines.append(f"| Jetson decision latency, {r['model_key']} | p50 {float(r['p50_ms']):.1f} ms, "
                     f"p99 {float(r['p99_ms']):.1f} ms | {r['source_run']} |")
    for (key, split), r in sorted(open_rows.items()):
        lines.append(f"| open loop {split}, {key} | accuracy {float(r['acc_pct']):.1f}%, PR {float(r['pr_pct']):.2f}%, "
                     f"local recall {r['local_recall_pct']} | {OPEN} |")
    for (pol, platform), cell in sorted(closed.items()):
        for n in sorted(cell):
            d, drop = cell[n]
            lines.append(f"| closed loop {platform}, {pol}, 3 servers, N = {n} | mean delay {d:.3f} slots, "
                         f"drop {drop:.2f}% | {'colab' if platform == 'colab' else 'jetson'} run CSVs |")
    lines += deployment_lines()
    lines.append("")
    lines.append("Closed-loop sources: " + ", ".join([COLAB_STUDENTS, COLAB_7B] + JETSON_VIRTUAL) + ".")
    with open(os.path.join(OUT, "numbers.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


STUDENTS = ("main_0.5b", "soft_1.5b", "hard_1.5b", "ps_0.5b", "ps_soft_1.5b", "ps_hard_1.5b")
AGREEMENT = os.path.join("evidence", "G6", "jetson_open", "platform_comparison.csv")


def deployment_lines():
    """Colab bf16 vs Jetson Q8_0 for the six students.

    1. closed loop: per student and N, |Δ mean delay| and |Δ drop rate| (seed means); the largest gaps over the
       17 settings other than ps_hard_1.5b at N = 60, which is reported on its own
    2. open loop: per-state agreement range for the paper-teacher students and the two PS soft-label students
    """
    cells = defaultdict(list)
    for path, platform in [(COLAB_STUDENTS, "colab")] + [(p, "jetson") for p in JETSON_VIRTUAL]:
        for r in read(path):
            key = r["policy"].replace("llm:", "")
            if key not in STUDENTS or r["servers"] != "3" or r["slots"] != "1000" or r["seed"] not in ("0", "1"):
                continue
            if r["mode"] == "sequential" and int(r["n"]) in (20, 40, 60):
                cells[(key, platform, int(r["n"]))].append(r)
    out, dd, dr = [], [], []
    for key in STUDENTS:
        for n in (20, 40, 60):
            c, j = cells[(key, "colab", n)], cells[(key, "jetson", n)]
            assert len(c) == 2 and len(j) == 2, (key, n)
            d = abs(F.mean([float(r["mean_delay"]) for r in c]) - F.mean([float(r["mean_delay"]) for r in j]))
            p = 100 * abs(F.mean([float(r["drop_rate"]) for r in c]) - F.mean([float(r["drop_rate"]) for r in j]))
            out.append(f"| Colab vs Jetson closed loop, {key}, N = {n} | |Δ mean delay| {d:.3f} slots, "
                       f"|Δ drop| {p:.2f} points | colab and jetson run CSVs |")
            if (key, n) != ("ps_hard_1.5b", 60):
                dd.append(d)
                dr.append(p)
    out.append(f"| Colab vs Jetson closed loop, largest gap over 17 settings (all but ps_hard_1.5b at N = 60) | "
               f"|Δ mean delay| {max(dd):.2f} slots ({max(dd):.3f}), |Δ drop| {max(dr):.2f} points | colab and jetson run CSVs |")
    rows = read(AGREEMENT)
    for label, keys in (("paper-teacher students", ("main_0.5b", "soft_1.5b", "hard_1.5b")),
                        ("PS-teacher soft-label students", ("ps_0.5b", "ps_soft_1.5b")),
                        ("PS-teacher hard-label 1.5B", ("ps_hard_1.5b",))):
        vals = [float(r["agreement_pct"]) for r in rows if r["model_key"] in keys]
        out.append(f"| open loop, same answer per state on Colab and Jetson, {label} | {min(vals):.2f}%–{max(vals):.2f}% "
                   f"over the three test sets | {AGREEMENT} |")
    return out


def main():
    groups, e1_dirs = stability_data()
    lat = latency_data()
    open_rows = open_data()
    closed = closed_data()
    for lang in ("en", "zh"):
        plt.rcParams["font.sans-serif"] = T[lang]["font"]
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["axes.unicode_minus"] = False
        os.makedirs(os.path.join(OUT, lang), exist_ok=True)
        for name, fig in (("stability", fig_stability(lang, groups)), ("latency", fig_latency(lang, lat)),
                          ("open_loop", fig_open(lang, open_rows)), ("closed_loop", fig_closed(lang, closed))):
            path = os.path.join(OUT, lang, name + ".png")
            fig.savefig(path, dpi=200, facecolor="white")
            plt.close(fig)
            print("wrote", path)
    write_numbers(groups, e1_dirs, lat, open_rows, closed)
    print("wrote", os.path.join(OUT, "numbers.md"))


if __name__ == "__main__":
    main()
