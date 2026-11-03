"""把 closed_loop.py 的 CSV 画成草图：横轴终端数 N，左图平均延迟，右图丢弃率，每种策略一条线。
线上的点是几个种子的平均值，竖线是最小到最大值。

  python -m eval.plot_sweep runs/<目录>/results.csv
图存在 CSV 旁边，名字是 sweep.png（服务器台数不止一种时，每种一行）。
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")                  # 不弹窗口，直接存文件
import matplotlib.pyplot as plt        # noqa: E402

# Mac 上用苹方，Linux 上用文泉驿，都没有就退回英文字体（中文会显示成方框）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC",
                                   "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 每种策略固定一种颜色和点形（换数据也不变色）；颜色组已过色盲区分度检查
# 画的顺序就是字典顺序：LACS 先画、点大一号，单步最优后画在上面，两条线几乎重合时都能看见
STYLE = {
    "lacs":     ("#eb6834", "s", "-", "LACS-greedy", 8),
    "greedy":   ("#2a78d6", "o", "-", "单步最优", 5),
    "po2":      ("#1baf7a", "^", "-", "power-of-2", 6),
    "random":   ("#eda100", "D", "-", "随机", 5),
    "psgreedy": ("#e87ba4", "v", "--", "PS 感知（探索）", 6),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(path):
    """返回 {(服务器台数, 策略): {N: [每个种子的行]}}"""
    data = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            data[(int(r["servers"]), r["policy"])][int(r["n_terminals"])].append(r)
    return data


def main(path):
    data = load(path)
    server_counts = sorted({k for k, _ in data})
    fig, axes = plt.subplots(len(server_counts), 2, figsize=(11, 3.8 * len(server_counts)), squeeze=False)
    metrics = [("mean_delay", "平均延迟（槽，只算完成的任务）", 1.0), ("drop_rate", "丢弃率（%）", 100.0)]
    for row, k in enumerate(server_counts):
        for col, (key, title, scale) in enumerate(metrics):
            ax = axes[row][col]
            for p, (color, marker, ls, label, size) in STYLE.items():
                if (k, p) not in data:
                    continue
                ns = sorted(data[(k, p)])
                vals = [[float(r[key]) * scale for r in data[(k, p)][n]] for n in ns]
                mean = [sum(v) / len(v) for v in vals]
                lo = [m - min(v) for m, v in zip(mean, vals)]
                hi = [max(v) - m for m, v in zip(mean, vals)]
                ax.errorbar(ns, mean, yerr=[lo, hi], color=color, marker=marker, markersize=size, linestyle=ls,
                            linewidth=2, capsize=0, elinewidth=1, label=label)
            ax.set_title(f"{k} 台服务器 · {title}", fontsize=11, color=INK, loc="left")
            ax.set_xlabel("终端数 N", color=MUTED)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=MUTED)
            ax.set_ylim(bottom=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))   # 顶上留一行给图例
    out = os.path.join(os.path.dirname(path), "sweep.png")
    fig.savefig(out, dpi=150)
    print("图已保存：", out)


if __name__ == "__main__":
    main(sys.argv[1])
