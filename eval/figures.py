"""一条命令从 CSV 重画全部图（8 张，跑过 Colab 闭环再多 1 张），并把简历要用的几个数字连同出处写成 figures/numbers.md（G5 的要求）。

在仓库根目录运行（Mac 上）：
  python -m eval.figures                 # 读 evidence/G5/sources.json 里登记的运行目录；没有这个文件就自动找最新的并写出来
  python -m eval.figures --rediscover    # 重新找最新的运行目录，覆盖 sources.json
结果：figures/fig1_stability.png … fig8_students.png、figures/numbers.md；跑过两板复测时还有 figures/twoboard.md。
某张图的数据还没跑出来时，那张图跳过，屏幕上会说缺什么。

数据从哪来（sources.json 的键）：
  e1          eval/s5_sweep.py 的扫描目录（results.csv）：实验 1 稳定边界，可以列几个目录
  twoboard    eval/s5_sweep.py --plan twoboard 的扫描目录：RK3588 当环境节点时的复测，和 e1 里同名的点对照
  bench       eval/bench_latency.py 的运行目录：实验 2 延迟分解，同一配置只取最新一次
  virtual     eval/virtual_loop.py 的目录：实验 3 羊群效应、实验 4 消融和泛化（含两个老师 × 三种学生）、实验 5 开环 vs 闭环、实验 6 LACS
  open_loop   阶段 3、4 的开环评测表 evidence/G3/open_loop.csv、evidence/G4/open_loop.csv
  colab       Colab 闭环的目录 runs/*_s6_colab_loop*/（train/S6_Colab闭环.ipynb 的结果，点的名字和 virtual 里一样）
"""
import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import matplotlib.transforms as mtransforms     # noqa: E402
from matplotlib.lines import Line2D             # noqa: E402
from matplotlib.patches import Patch            # noqa: E402

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC",
                                   "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SOURCES = os.path.join("evidence", "G5", "sources.json")
OUT = "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# 模型键 / 策略名 → 图上的名字、颜色、点形。颜色和 eval/plot_sweep.py 一致：同一策略在所有图里同一个颜色
STYLE = {
    "formula":        ("公式（Jetson 上算）", "#2a78d6", "o"),
    "greedy":         ("单步最优（公式）", "#2a78d6", "o"),
    "lacs":           ("LACS-greedy", "#eb6834", "s"),
    "lacs:1":         ("LACS λ = 1", "#0f8b8d", "D"),
    "lacs:2":         ("LACS λ = 2", "#8a3412", "^"),
    "psgreedy":       ("PS 感知贪心", "#e87ba4", "v"),
    "po2":            ("power-of-2", "#1baf7a", "^"),
    "random":         ("随机", "#eda100", "D"),
    "smallnet":       ("小网络 1313 参数", "#7a5af5", "P"),
    "main_0.5b":      ("0.5B 软标签（部署）", "#0b0b0b", "o"),
    "main_0.5b_q4":   ("0.5B 软标签 Q4_K_M", "#52514e", "o"),
    "soft_1.5b":      ("1.5B 软标签", "#8c564b", "s"),
    "hard_1.5b":      ("1.5B 硬标签", "#5d3a1a", "^"),
    "ps_0.5b":        ("0.5B 平分算力标签", "#c2185b", "v"),
    "ps_soft_1.5b":   ("1.5B 平分算力软标签", "#8e0e47", "s"),
    "ps_hard_1.5b":   ("1.5B 平分算力硬标签", "#f08cb4", "^"),
    "base_0.5b":      ("未训练 0.5B", "#9e9e9e", "x"),
    "paper_sft_1.5b": ("论文原版 1.5B（sft）", "#d62728", "X"),
}

# 虚拟时间闭环的策略名 → 开环表（evidence/G3/open_loop.csv）里的名字
OPEN_NAME = {"greedy": "公式·压缩精度（学生上限）", "lacs": "LACS-greedy", "psgreedy": "PS 感知贪心",
             "po2": "power-of-2", "random": "随机", "smallnet": "小网络 1313 参数",
             "llm:main_0.5b": "0.5b_soft_T0.3", "llm:ps_0.5b": "0.5b_soft_T0.3_ps",
             "llm:soft_1.5b": "1.5b_soft_T0.3", "llm:base_0.5b": "未训练 0.5B", "llm:paper_sft_1.5b": "论文原版 sft",
             "llm:hard_1.5b": "1.5b_hard", "llm:ps_soft_1.5b": "1.5b_soft_T0.3_ps", "llm:ps_hard_1.5b": "1.5b_hard_ps"}

PLATFORM_NAME = {"jetson": "Jetson Q8_0", "colab": "Colab 原精度"}
# 图 8 上另外画的论文模型（只在 Colab 上跑闭环）：(策略名, 名字, 颜色)
PAPER_MODELS_CLOSED = [("llm:comllm_7b_repro", "复现的 COMLLM-7B", "论文的模型", "#0b0b0b"),
                       ("llm:grpo_7b_repro", "复现的 GRPO-7B", "论文的模型，不带 LACS", "#7a7974")]

# 图 8：两个老师 × 三种学生。每个老师：(名字, 策略名, 虚线颜色, [(学生策略名, 学生名字, 颜色, 点形), ...])
TEACHER_GRID = [
    ("论文公式", "greedy", "#2a78d6",
     [("llm:main_0.5b", "0.5B 软标签", "#0b2e63", "o"), ("llm:soft_1.5b", "1.5B 软标签", "#1f6fd1", "s"),
      ("llm:hard_1.5b", "1.5B 硬标签", "#7fb2ee", "^")]),
    ("PS 感知公式", "psgreedy", "#e87ba4",
     [("llm:ps_0.5b", "0.5B 软标签", "#7a0b3a", "o"), ("llm:ps_soft_1.5b", "1.5B 软标签", "#c2185b", "s"),
      ("llm:ps_hard_1.5b", "1.5B 硬标签", "#f08cb4", "^")]),
]


def style_of(key):
    k = key[4:] if key.startswith("llm:") else key
    return STYLE.get(k, (k, "#52514e", "o"))


def tidy(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finish(fig, legends, note=None, h_pad=None):
    """排版收尾：各格排好以后，图例放到格子正下方（不压数据），注释写在最下面。

    legends：[(格子列表, 图例条目, 每行几个), ...]。图例条目是 [(句柄, 文字), ...]；写 None 就用第一格自己的。
    1. tight_layout 排好各格（这时还没有图例）
    2. 逐个图例：量出这组格子连同横轴标题的左右范围和下边缘，图例居中、上边缘贴在下边缘下面
    3. 注释写在所有格子和图例的更下面
    保存时用 bbox_inches="tight"，伸到图外的图例和注释会一起存进去
    """
    if h_pad is None:
        fig.tight_layout()
    else:
        fig.tight_layout(h_pad=h_pad)
    renderer = fig.canvas.get_renderer()
    w = fig.bbox.width
    h = fig.bbox.height
    lowest = 1.0
    for ax in fig.axes:
        lowest = min(lowest, ax.get_tightbbox(renderer).y0 / h)
    for axes, items, ncol in legends:
        x0, x1, y0 = None, None, None
        for ax in axes:
            bb = ax.get_tightbbox(renderer)
            x0 = bb.x0 if x0 is None else min(x0, bb.x0)
            x1 = bb.x1 if x1 is None else max(x1, bb.x1)
            y0 = bb.y0 if y0 is None else min(y0, bb.y0)
        if items is None:
            handles, labels = axes[0].get_legend_handles_labels()
        else:
            handles = []
            labels = []
            for handle, text in items:
                handles.append(handle)
                labels.append(text)
        leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=((x0 + x1) / 2 / w, y0 / h - 0.01),
                         ncol=ncol, frameon=False, fontsize=8, handlelength=3)
        lowest = min(lowest, leg.get_window_extent(renderer).y0 / h)
    if note:
        fig.text(0.01, lowest - 0.02, note, fontsize=8, color=MUTED, va="top", ha="left", linespacing=1.6)


# ===== 第 1 段：找数据 =====
def discover(include_rehearse=False):
    """自动找最新的运行目录，返回和 sources.json 同样结构的字典。

    1. e1：runs/*_s5_*/results.csv 所在的扫描目录；名字带 twoboard 的归到 twoboard
    2. virtual：runs/*_s5_virtual_*/ 全部（不同计划可能分几次跑）
    3. bench：runs/*bench*/summary.json 全部（画图时同一配置只取最新）
    4. 开环表：evidence/G3、G4 的 open_loop.csv
    5. colab：runs/*_s6_colab_loop*/ 全部
    预演目录（名字带 rehearse）默认不算
    """
    src = {"e1": [], "twoboard": [], "virtual": [], "bench": [], "open_loop": [], "colab": []}
    for d in sorted(glob.glob(os.path.join("runs", "*_s5_*"))):
        if "rehearse" in d and not include_rehearse:
            continue
        if not os.path.exists(os.path.join(d, "results.csv")):
            continue
        if "_s5_virtual_" in d:
            src["virtual"].append(d)
        elif "twoboard" in d:
            src["twoboard"].append(d)
        else:
            src["e1"].append(d)
    for p in sorted(glob.glob(os.path.join("runs", "*bench*", "summary.json"))):
        src["bench"].append(os.path.dirname(p))
    for p in sorted(glob.glob(os.path.join("runs", "*_s6_colab_loop*", "results.csv"))):
        src["colab"].append(os.path.dirname(p))
    for p in (os.path.join("evidence", "G3", "open_loop.csv"), os.path.join("evidence", "G4", "open_loop.csv")):
        if os.path.exists(p):
            src["open_loop"].append(p)
    return src


def load_rows(dirs):
    """几个目录的 results.csv 合在一起；同一个点出现多次，后面的目录覆盖前面的。"""
    by_point = {}
    for d in dirs:
        for r in read_csv(os.path.join(d, "results.csv")):
            r["_dir"] = os.path.basename(d)
            by_point[r["point"]] = r
    return list(by_point.values())


def load_bench(dirs):
    """延迟基准：每个 (环境节点, 模型, 格式, 缓存) 只留最新一次。模型优先用调度服务报的模型键。

    环境节点取自 config.json 的 env_node；阶段 2、3 的旧目录没有这一项，都是在 Mac 上测的，记为 mac。
    """
    latest = {}
    for d in sorted(dirs):
        with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
            s = json.load(f)
        env = "mac"
        cfg_path = os.path.join(d, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                env = (json.load(f).get("env_node") or {}).get("name", "mac")
        info = s.get("scheduler_info") or {}
        key = info.get("model_key") or s["model"]
        if s["fmt"] == "formula":
            key = "formula"
        s["_key"] = key
        s["_env"] = env
        s["_dir"] = os.path.basename(d)
        latest[(env, key, s["fmt"], s["cache"], s.get("decode", "-"))] = s
    return list(latest.values())


def load_open(paths):
    """开环表 → {(策略名, 数据集, 台数): 行}。G4 表里和 G3 重名的行以 G3 为准。"""
    out = {}
    for p in reversed(paths):
        for r in read_csv(p):
            out[(r["policy"], r["split"], r["servers"])] = r
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def twoboard_table(e1_rows, tb_rows, bench, path, numbers):
    """两板复测对照表（Markdown）：同一个点，Mac 当环境节点 vs RK3588 当环境节点。

    1. 闭环：twoboard 里的每个点，在 e1 里找同名的点，两边并排列 平均延迟 / 丢弃率 / 往返 p50、p99
    2. 延迟基准：同一 (模型, 格式, 缓存) 在两个环境节点上各测过的，并排列 端到端 p50、p99 和 网络+HTTP 平均
    3. 两样都没有就不写文件
    """
    mac = {}
    for r in e1_rows:
        mac[r["point"]] = r
    lines = []
    for r in sorted(tb_rows, key=lambda r: (r["model_key"], int(r["n"]), int(r["seed"]))):
        m = mac.get(r["point"])
        if m is None:
            continue
        lines.append(f"| {r['point']} | {float(m['mean_delay']):.2f} / {float(r['mean_delay']):.2f} | "
                     f"{100 * float(m['drop_rate']):.1f} / {100 * float(r['drop_rate']):.1f} | "
                     f"{float(m['e2e_ms_p50']):.1f} / {float(r['e2e_ms_p50']):.1f} | "
                     f"{float(m['e2e_ms_p99']):.1f} / {float(r['e2e_ms_p99']):.1f} | {m['_dir']} / {r['_dir']} |")
    by_cfg = {}
    for b in bench:
        by_cfg.setdefault((b["_key"], b["fmt"], b["cache"], b.get("decode", "-")), {})[b["_env"]] = b
    blines = []
    for cfg in sorted(by_cfg):
        envs = by_cfg[cfg]
        if "mac" not in envs or len(envs) < 2:
            continue
        for env in sorted(envs):
            if env == "mac":
                continue
            a = envs["mac"]
            b = envs[env]
            blines.append(f"| {cfg[0]} · {cfg[1]} · 缓存{cfg[2]} | {env} | {a['e2e_p50']:.1f} / {b['e2e_p50']:.1f} | "
                          f"{a['e2e_p99']:.1f} / {b['e2e_p99']:.1f} | {a['net_mean']:.1f} / {b['net_mean']:.1f} | "
                          f"{a['_dir']} / {b['_dir']} |")
            if cfg[0] == "main_0.5b" and cfg[1] == "compact" and cfg[2] == "on":
                numbers.append((f"两板：单次决策（环境节点 {env}）", f"p50 {b['e2e_p50']:.1f} ms，p99 {b['e2e_p99']:.1f} ms（实测）；"
                                f"Mac 上 p50 {a['e2e_p50']:.1f} ms", b["_dir"]))
    if not lines and not blines:
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 两板复测：Mac vs RK3588 当环境节点\n\n由 `python -m eval.figures` 生成。每格是“Mac / RK3588”，全部实测。\n\n")
        if lines:
            f.write("## 实时闭环（同一个点）\n\n")
            f.write("| 点 | 平均延迟（槽） | 丢弃率 % | 往返 p50 ms | 往返 p99 ms | 运行目录 |\n|---|---|---|---|---|---|\n")
            f.write("\n".join(lines) + "\n\n")
        if blines:
            f.write("## 单次决策延迟基准\n\n")
            f.write("| 配置 | 对照节点 | 端到端 p50 ms | 端到端 p99 ms | 网络+HTTP 平均 ms | 运行目录 |\n|---|---|---|---|---|---|\n")
            f.write("\n".join(blines) + "\n")
    return True


# ===== 第 2 段：八张图 =====
def fig1_stability(rows, bench, numbers):
    """图 1 稳定边界：横轴 N，左平均延迟、右丢弃率；每个 (模型, 是否分流) 一条线，竖虚线是理论 N_max。

    1. 按 (模型, 分流) 分组，同一 N 的几个种子取平均，竖线画最小到最大
    2. 理论 N_max = 1 / (3 × 单次决策平均秒数)：优先用实验 1 里最小 N 那次的往返（和实验 1 同样 3 台服务器），
       没有才用实验 2 的延迟基准（基准混了 3/7/11 台，提示词更长，算出的 N_max 偏小）；竖虚线顶上标出数值
    3. 稳定边界 M = 决策环节丢弃（过了截止期还没拿到决策）在每个种子上都低于 1% 的最大 N；一个都不稳定就写最小 N 时丢了多少；
       再写上测到的最大 N 时的总丢弃率（含服务器排满丢的）
    4. 只测了一个 N 的模型（论文原版 1.5B 只测 N = 1），在右图点旁直接写丢弃率
    5. 数每个模型每个 N 有几个种子，写进注释
    6. 图例放到两格下方
    """
    groups = {}
    for r in rows:
        key = (r["model_key"], float(r["shed_ms"]) > 0)
        groups.setdefault(key, {}).setdefault(int(r["n"]), []).append(r)
    seed_count = {}
    for key, by_n in groups.items():
        fewest = None
        for n in by_n:
            k = len(by_n[n])
            if fewest is None or k < fewest:
                fewest = k
        seed_count[key] = fewest
    bench_mean = {}
    for s in bench:
        if s["fmt"] in ("compact", "formula", "paper") and s["cache"] == ("off" if s["fmt"] == "paper" else "on"):
            bench_mean[s["_key"]] = s["e2e_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    marks = []
    for (key, shed), by_n in sorted(groups.items(), key=lambda kv: (kv[0][0] != "formula", kv[0])):
        name, color, marker = style_of(key)
        if shed:
            name = name + " + 超载分流"
        ns = sorted(by_n)
        for ax, field, scale in ((axes[0], "mean_delay", 1.0), (axes[1], "drop_rate", 100.0)):
            m, lo, hi = [], [], []
            for n in ns:
                vals = []
                for r in by_n[n]:
                    vals.append(float(r[field]) * scale)
                m.append(mean(vals))
                lo.append(m[-1] - min(vals))
                hi.append(max(vals) - m[-1])
            ax.errorbar(ns, m, yerr=[lo, hi], color=color, marker=marker, linestyle="--" if shed else "-",
                        linewidth=2, markersize=5, label=name)
        stable = []
        for n in ns:
            if all(float(r["drop_decision"]) < 0.01 for r in by_n[n]):
                stable.append(n)
        e2e = mean([float(r["e2e_ms_mean"]) for r in by_n[ns[0]]])
        if e2e != e2e:
            e2e = bench_mean.get(key, float("nan"))
        nmax = 1000.0 / (3.0 * e2e) if e2e and e2e == e2e else float("nan")
        if not shed and key != "formula" and nmax < 100:
            for ax in axes:
                ax.axvline(nmax, color=color, linestyle=":", linewidth=1)
            marks.append((nmax, color))
        if len(ns) == 1:
            drop = mean([float(r["drop_rate"]) for r in by_n[ns[0]]])
            axes[1].annotate(f"{name}\nN = {ns[0]} 就丢弃 {100 * drop:.0f}%", (ns[0], 100 * drop), xytext=(8, -6),
                             textcoords="offset points", fontsize=8, color=color, va="top")
        if stable and max(stable) == ns[-1]:
            m_text = f"≥ {max(stable)} 台终端（实测，测到的最大 N 也稳定）"
        elif stable:
            m_text = f"{max(stable)} 台终端（实测，测过的 N 里决策环节丢弃 < 1% 的最大值）"
        else:
            first = mean([float(r["drop_decision"]) for r in by_n[ns[0]]])
            m_text = f"1 台都撑不住（实测：N = {ns[0]} 时决策环节就丢弃 {100 * first:.1f}%）"
        last_drop = mean([float(r["drop_rate"]) for r in by_n[ns[-1]]])
        m_text += f"；测到的最大 N = {ns[-1]} 时总丢弃率 {100 * last_drop:.1f}%"
        dirs = set()
        for v in by_n.values():
            for r in v:
                dirs.add(r["_dir"])
        numbers.append((f"稳定边界 M（{name}）", m_text + f"；理论 N_max = 1/(3 × {e2e:.1f} ms) = {nmax:.1f}",
                        ", ".join(sorted(dirs))))
    tidy(axes[0], "实验 1 · 平均延迟（只算完成的任务）", "终端数 N", "槽（0.1 s）")
    tidy(axes[1], "实验 1 · 丢弃率", "终端数 N", "%")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 40, 80])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for nmax, color in marks:
        ax = axes[0]
        tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(nmax, 0.98, f"N_max {nmax:.1f} ", transform=tr, rotation=90, va="top", ha="right", fontsize=7,
                color=color)
    handles, labels = axes[0].get_legend_handles_labels()
    items = []
    for i in range(len(handles)):
        items.append((handles[i], labels[i]))
    items.append((Line2D([], [], color=MUTED, linestyle=":"), "理论容量 N_max = 1/(3 × 单次决策秒数)"))
    most = max(seed_count.values())
    fewer = []
    for key in sorted(seed_count):
        if seed_count[key] < most:
            name = style_of(key[0])[0]
            if key[1]:
                name = name + " + 超载分流"
            fewer.append(f"{name} {seed_count[key]} 个")
    seeds_text = f"每点 3 分钟、{most} 个种子"
    if fewer:
        seeds_text += "（" + "，".join(fewer) + "）"
    finish(fig, [(list(axes), items, 3)],
           f"实时闭环，3 台服务器，{seeds_text}；竖线是种子之间的最小到最大。超载分流：等决策超过 100 ms 就改用公式。")
    return fig


def fig2_latency(bench, e1_rows, numbers):
    """图 2 决策延迟分解：每个配置一根横条，按 网络+HTTP / Python 服务 / llama-server 其他 / 预填充 / 解码 堆起来，
    条末标端到端 p50 / p99。从慢到快排，读起来就是"一步步省下来"。

    1. 只画调度服务报了模型键的运行（阶段 5 起：训练好的模型和公式）；阶段 2 测的是未训练的原版 Qwen，表在 evidence/G2/latency_table.md
    2. 每个配置一根横条，从慢到快排；部署配置的名字加粗
    3. 论文格式那条是关缓存测的；实验 1 里同一个模型开着缓存（3 台服务器），把那次的往返 p50 写进注释和数字表
    """
    rows = []
    for s in bench:
        if s["_key"] in STYLE:
            rows.append(s)
    rows.sort(key=lambda s: -s["e2e_p50"])
    if not rows:
        return None
    parts = [("net_mean", "网络+HTTP", "#9e9e9e"), ("python_mean", "Python 服务", "#7a5af5"),
             ("llama_other_mean", "llama-server 其他", "#eda100"), ("prompt_ms_mean", "预填充", "#2a78d6"),
             ("predicted_ms_mean", "解码", "#eb6834")]
    fmt_name = {"paper": "论文格式", "compact": "压缩格式", "formula": "公式"}
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(rows) + 1.4))
    labels = []
    deployed = None
    longest = 0.0
    for i in range(len(rows)):
        s = rows[i]
        left = 0.0
        for k, name, color in parts:
            v = max(0.0, float(s.get(k, 0.0)))
            ax.barh(i, v, left=left, color=color, label=name if i == 0 else None)
            left += v
        longest = max(longest, left)
        ax.text(left + 8, i, f"p50 {s['e2e_p50']:.1f} / p99 {s['e2e_p99']:.1f} ms", va="center", fontsize=8,
                color=INK)
        name = style_of(s["_key"])[0]
        if s["fmt"] == "formula":
            labels.append(name)
        else:
            cache = "开" if s["cache"] == "on" else "关"
            labels.append(f"{name} · {fmt_name.get(s['fmt'], s['fmt'])} · 缓存{cache}")
        if s["_key"] == "main_0.5b" and s["fmt"] == "compact" and s["cache"] == "on":
            deployed = i
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    if deployed is not None:
        ax.get_yticklabels()[deployed].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, longest * 1.32)
    tidy(ax, "实验 2 · 单次决策延迟分解（Jetson Orin Nano，环境节点上计时，500 次的平均）", "毫秒", "")
    note = "每条的长度是 500 次的平均，分成五段；条末是端到端 p50 / p99。3/7/11 台服务器轮流。加粗的是部署配置。"
    paper = {}
    for r in e1_rows:
        if r["model_key"] == "paper_sft_1.5b" and float(r["shed_ms"]) == 0:
            paper.setdefault(int(r["n"]), []).append(r)
    if paper:
        n = min(paper)
        p50 = mean([float(r["e2e_ms_p50"]) for r in paper[n]])
        note += f"\n论文格式那条关着提示词缓存；实验 1 里同一个模型开着缓存、3 台服务器，往返 p50 {p50:.0f} ms（实测，N = {n}）"
        if deployed is not None:
            note += f"，仍是部署配置的 {p50 / rows[deployed]['e2e_p50']:.1f} 倍"
        note += "。"
        numbers.append(("论文原版 1.5B 单次决策（开缓存、3 台服务器）", f"往返 p50 {p50:.1f} ms（实测，实验 1、N = {n}）",
                        paper[n][0]["_dir"]))
    finish(fig, [([ax], None, 5)], note)
    for i in range(len(rows)):
        s = rows[i]
        value = f"p50 {s['e2e_p50']:.1f} ms，p99 {s['e2e_p99']:.1f} ms"
        if float(s.get("prompt_tokens_mean", 0) or 0) > 0:
            value += f"，平均提示词 {float(s['prompt_tokens_mean']):.0f} token"
        value += "（实测）"
        if s["_key"] == "main_0.5b" and s["fmt"] == "compact" and s["cache"] == "on":
            numbers.append(("单次决策 X（部署配置）", value, s["_dir"]))
        elif s["_key"] == "paper_sft_1.5b" and s["fmt"] == "paper" and s["cache"] == "off":
            numbers.append(("论文原版 1.5B 单次决策 Y（关缓存、3/7/11 台轮流）", value, s["_dir"]))
        else:
            numbers.append((f"单次决策（{labels[i]}）", value, s["_dir"]))
    return fig


def fig3_herding(rows, numbers):
    """图 3 羊群效应：(a) 改成顺序决策后平均延迟降低多少；(b) 丢弃率降低多少；(c) 11 台、N 最大时每台服务器分到的卸载任务。

    1. 只用实验 3 的行（1500 槽：两种老师 × 两种决策方式）
    2. 同一 (老师, 台数, N, 决策方式) 几个种子取平均
       (a) = (批量 − 顺序) / 批量 × 100%，(b) = 批量丢弃率 − 顺序丢弃率（百分点）；两格都是正数 = 顺序更好
    3. (c) 读 server_share 列，两种决策方式并排
    4. 颜色 = 台数，实线 = 单步最优，虚线 = PS 感知贪心；图例只列这 5 样，放在 (a)(b) 下方
    5. 注释：平均延迟只算完成的任务，所以 (a)(b) 要一起看；3 台时单步最优改顺序为什么变差（本地占比现算）
    """
    cell = {}
    for r in rows:
        if r["policy"] not in ("greedy", "psgreedy") or int(r["slots"]) != 1500:
            continue
        cell.setdefault((r["policy"], int(r["servers"]), int(r["n"]), r["mode"]), []).append(r)
    if not cell:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    colors = {3: "#2a78d6", 7: "#1baf7a", 11: "#eb6834"}
    best = None
    worst3 = None
    for pol, ls in (("greedy", "-"), ("psgreedy", "--")):
        for k in (3, 7, 11):
            ns = sorted({n for (p, kk, n, m) in cell if p == pol and kk == k})
            d_delay = []
            d_drop = []
            for n in ns:
                seq = cell.get((pol, k, n, "sequential"), [])
                bat = cell.get((pol, k, n, "batch"), [])
                s_delay = mean([float(r["mean_delay"]) for r in seq])
                b_delay = mean([float(r["mean_delay"]) for r in bat])
                s_drop = mean([float(r["drop_rate"]) for r in seq])
                b_drop = mean([float(r["drop_rate"]) for r in bat])
                d_delay.append(100.0 * (b_delay - s_delay) / b_delay)
                d_drop.append(100.0 * (b_drop - s_drop))
                if pol == "greedy" and k == 11 and (best is None or n > best[0]):
                    best = (n, s_delay, b_delay, s_drop, b_drop)
                if pol == "greedy" and k == 3 and (worst3 is None or d_delay[-1] < worst3[1]):
                    worst3 = (n, d_delay[-1], s_drop, b_drop)
            axes[0].plot(ns, d_delay, color=colors[k], linestyle=ls, marker="o")
            axes[1].plot(ns, d_drop, color=colors[k], linestyle=ls, marker="o")
    tidy(axes[0], "(a) 改成顺序决策后，平均延迟降低多少", "终端数 N", "(批量 − 顺序) / 批量，%")
    tidy(axes[1], "(b) 改成顺序决策后，丢弃率降低多少", "终端数 N", "批量 − 顺序，百分点")
    for ax in axes[:2]:
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks([20, 40, 60, 80])
    items = []
    for k in (3, 7, 11):
        items.append((Line2D([], [], color=colors[k], marker="o"), f"{k} 台服务器"))
    items.append((Line2D([], [], color=MUTED), "单步最优（论文公式）"))
    items.append((Line2D([], [], color=MUTED, linestyle="--"), "PS 感知贪心（平分算力公式）"))
    legends = [([axes[0], axes[1]], items, 5)]
    if best is not None:
        n, seq, bat, s_drop, b_drop = best
        share = {}
        for mode in ("sequential", "batch"):
            rs = cell.get(("greedy", 11, n, mode), [])
            vals = None
            for r in rs:
                v = [float(x) for x in r["server_share"].split(";")]
                vals = v if vals is None else [a + b for a, b in zip(vals, v)]
            share[mode] = [x / len(rs) for x in vals]
        idx = list(range(1, 12))
        axes[2].bar([i - 0.2 for i in idx], [100 * x for x in share["sequential"]], width=0.4, color="#2a78d6",
                    label="顺序决策 + 虚拟状态更新")
        axes[2].bar([i + 0.2 for i in idx], [100 * x for x in share["batch"]], width=0.4, color="#eb6834",
                    label="批量决策（同一槽共用过时状态）")
        axes[2].set_xticks(idx)
        tidy(axes[2], f"(c) 11 台、N = {n}、单步最优：每台分到的卸载任务", "服务器编号", "占卸载任务 %")
        legends.append(([axes[2]], None, 1))
        dirs = ", ".join(sorted({r["_dir"] for v in cell.values() for r in v}))
        numbers.append((f"羊群效应修复 Z（11 台、N = {n}、单步最优）",
                        f"平均延迟 {bat:.2f} → {seq:.2f} 槽，降低 {100 * (bat - seq) / bat:.1f}%；"
                        f"丢弃率 {100 * b_drop:.1f}% → {100 * s_drop:.1f}%（仿真实测）", dirs))
        if worst3 is not None and worst3[1] < 0:
            n3, d3, s3, b3 = worst3
            numbers.append((f"反例：3 台、N = {n3}、单步最优改顺序决策",
                            f"平均延迟反而高 {-d3:.1f}%，丢弃率 {100 * b3:.1f}% → {100 * s3:.1f}%（仿真实测）", dirs))
    fig.suptitle("实验 3 · 羊群效应：同一槽里一起决策（批量） vs 一个接一个、每次更新虚拟状态（顺序）", x=0.01, ha="left",
                 fontsize=12, color=INK)
    loc = {}
    for mode in ("batch", "sequential"):
        loc[mode] = mean([float(r["local_frac"]) for r in cell.get(("greedy", 3, 40, mode), [])])
    note = ("虚拟时间闭环，公式老师，每点 1500 槽、2 个种子取平均。平均延迟只算完成的任务，丢得多时会显得低，所以 (a)(b) 要一起看。\n"
            "3 台时单步最优改成顺序反而变差：论文公式高估平分算力服务器的排队，顺序决策把同一槽刚分出去的任务也算进去，高估叠加，"
            f"更多任务留在本地（N = 40：本地 {100 * loc['batch']:.0f}% → {100 * loc['sequential']:.0f}%），本地排满就丢弃。")
    finish(fig, legends, note)
    return fig


def fig4_generalization(rows, open_rows, numbers):
    """图 4 泛化：左 = 开环准确率按服务器台数；右 = 虚拟时间闭环（N = 40）平均延迟按台数。4/6/8/10 台训练时没见过，灰底标出。

    1. 左图纵轴断开：上段 86–100 看我们的模型，下段 0–45 看论文原版和随机。每条线在上下两段各画一次，各段只显示自己的范围
       （单数台数的格子在 test_seen，双数在 test_unseen）
    2. PR（论文式 28–29）在我们的模型上全都接近 100%，几条线叠成一条，所以左图画准确率，PR 的范围写进注释
    3. 右图：N = 40、顺序决策、种子 0 的点；单步最优画粗浅线，0.5B 叠在上面，两者差多少现算写在图里
    """
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1])
    ax_hi = fig.add_subplot(gs[0, 0])
    ax_lo = fig.add_subplot(gs[1, 0], sharex=ax_hi)
    ax_r = fig.add_subplot(gs[:, 1])
    ks = list(range(3, 12))
    lines = [("公式·压缩精度（学生上限）", "greedy", "单步最优（公式，压缩精度 = 学生上限）"),
             ("0.5b_soft_T0.3", "main_0.5b", None), ("1.5b_soft_T0.3", "soft_1.5b", None),
             ("论文原版 sft", "paper_sft_1.5b", None), ("随机", "random", None)]
    pr = {}
    for name, key, label in lines:
        xs, ys, prs = [], [], []
        for k in ks:
            split = "test_seen" if k % 2 == 1 else "test_unseen"
            r = open_rows.get((name, split, str(k)))
            if r is not None:
                xs.append(k)
                ys.append(float(r["acc_pct"]))
                prs.append(float(r["pr_pct"]))
        if not xs:
            continue
        pr[key] = prs
        default_label, color, marker = style_of(key)
        for ax in (ax_hi, ax_lo):
            ax.plot(xs, ys, color=color, marker=marker, label=label or default_label,
                    linestyle="--" if key == "greedy" else "-", linewidth=1.8)
    ax_hi.set_ylim(86, 100.8)
    ax_lo.set_ylim(0, 45)
    tidy(ax_hi, "开环准确率（和单步最优选得一样的比例，阶段 3）", "", "准确率 %")
    tidy(ax_lo, "", "服务器台数", "")
    ax_hi.tick_params(labelbottom=False, bottom=False)
    ax_hi.spines["bottom"].set_visible(False)
    for ax, y in ((ax_hi, 0.0), (ax_lo, 1.0)):
        ax.plot([-0.015, 0.015], [y - 0.03, y + 0.03], transform=ax.transAxes, color=INK, clip_on=False,
                linewidth=1)
    cl = {}
    for r in rows:
        if int(r["n"]) == 40 and r["mode"] == "sequential" and int(r["seed"]) == 0 and int(r["slots"]) == 1000:
            cl.setdefault(r["policy"], {})[int(r["servers"])] = r
    draw = {"greedy": {"linewidth": 7, "alpha": 0.3, "marker": None},
            "lacs": {"linewidth": 1, "markersize": 4},
            "psgreedy": {"linewidth": 1.8, "linestyle": "--"},
            "llm:main_0.5b": {"linewidth": 1.8, "markersize": 5},
            "llm:ps_0.5b": {"linewidth": 1.8, "markersize": 5}}
    if len(cl.get("llm:ps_0.5b", {})) < 2:
        cl.pop("llm:ps_0.5b", None)
    for pol in ("greedy", "lacs", "psgreedy", "llm:main_0.5b", "llm:ps_0.5b"):
        if pol not in cl:
            continue
        xs = sorted(cl[pol])
        label, color, marker = style_of(pol)
        kw = {"marker": marker}
        kw.update(draw[pol])
        ax_r.plot(xs, [float(cl[pol][k]["mean_delay"]) for k in xs], color=color, label=label, **kw)
    tidy(ax_r, "虚拟时间闭环，N = 40 台终端", "服务器台数", "平均延迟（槽）")
    for ax in (ax_hi, ax_lo, ax_r):
        for k in (4, 6, 8, 10):
            ax.axvspan(k - 0.5, k + 0.5, color=GRID, alpha=0.6, linewidth=0)
        ax.set_xticks(ks)
    if "llm:main_0.5b" in cl and "greedy" in cl:
        gaps = []
        rel = []
        for k in sorted(cl["llm:main_0.5b"]):
            if k in cl["greedy"]:
                a = float(cl["llm:main_0.5b"][k]["mean_delay"])
                b = float(cl["greedy"][k]["mean_delay"])
                rel.append(100 * (a - b) / b)
                gaps.append(f"{k} 台 {rel[-1]:+.1f}%")
        ax_r.text(0.97, 0.62, f"0.5B 和单步最优、LACS 几乎重合：\n0.5B 每个台数相差 {min(rel):+.1f}% ~ {max(rel):+.1f}%",
                  transform=ax_r.transAxes, ha="right", va="top", fontsize=8, color=INK)
        numbers.append(("0.5B 闭环平均延迟 vs 单步最优（N = 40）", "；".join(gaps) + "（仿真实测）",
                        cl["llm:main_0.5b"][min(cl["llm:main_0.5b"])]["_dir"]))
    if "llm:ps_0.5b" in cl and "psgreedy" in cl:
        gaps = []
        dirs = set()
        for k in sorted(cl["llm:ps_0.5b"]):
            if k in cl["psgreedy"]:
                a = float(cl["llm:ps_0.5b"][k]["mean_delay"])
                b = float(cl["psgreedy"][k]["mean_delay"])
                gaps.append(f"{k} 台 {100 * (a - b) / b:+.1f}%")
                dirs.add(cl["llm:ps_0.5b"][k]["_dir"])
        numbers.append(("PS 老师 0.5B 闭环平均延迟 vs PS 感知贪心（N = 40）", "；".join(gaps) + "（仿真实测）",
                        ", ".join(sorted(dirs))))
    handles, labels = ax_hi.get_legend_handles_labels()
    items = []
    for i in range(len(handles)):
        items.append((handles[i], labels[i]))
    items.append((Patch(color=GRID), "灰底：训练时没见过的台数"))
    fig.suptitle("实验 4 · 台数泛化：训练时没见过 4/6/8/10 台服务器（灰底）", x=0.01, ha="left", fontsize=12, color=INK)
    note = "灰底 4/6/8/10 台：训练集里没有，只用来测泛化。右图每点 1000 槽、种子 0，公式和 0.5B 面对同一串任务。"
    if "main_0.5b" in pr and "soft_1.5b" in pr and "paper_sft_1.5b" in pr:
        ours = min(min(pr["main_0.5b"]), min(pr["soft_1.5b"]))
        note += (f"\n开环 PR（论文式 28–29）：0.5B、1.5B 每个台数都 ≥ {ours:.1f}%，"
                 f"论文原版 sft {min(pr['paper_sft_1.5b']):.0f}–{max(pr['paper_sft_1.5b']):.0f}%；PR 分不开我们的模型，所以这里画准确率。")
    finish(fig, [([ax_lo], items, 2), ([ax_r], None, 2)], note, h_pad=0.4)
    return fig


def fig5_open_vs_closed(rows, open_rows, numbers):
    """图 5 开环 vs 闭环：横轴 = 开环准确率（3 台服务器，test_seen）；左 = 虚拟时间闭环平均延迟，右 = 丢弃率（3 台、N = 40）。

    只取实验 5 的点：1000 槽（e5），论文原版 1.5B 太慢只跑了 400 槽（e5_paper）；实验 3 的 1500 槽点不算，免得同一策略出现两次。
    1. 每个策略一个点；名字不写在点旁（挤在一起看不清），统一放到图例里，图例里带上三个数
    2. 准确率 > 95% 的几个点挤在一起：左图放一个放大的小图
    3. 注释说明为什么准确率低的策略反而快、随机为什么平均延迟不高
    """
    pts = []
    for r in rows:
        if int(r["slots"]) not in (1000, 400):
            continue
        if int(r["servers"]) == 3 and int(r["n"]) == 40 and r["mode"] == "sequential" and int(r["seed"]) == 0:
            o = open_rows.get((OPEN_NAME.get(r["policy"], ""), "test_seen", "3"))
            if o is not None:
                pts.append((r["policy"], float(o["acc_pct"]), float(r["mean_delay"]), float(r["drop_rate"]), r["_dir"]))
    if not pts:
        return None
    pts.sort(key=lambda p: -p[1])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    items = []
    for pol, acc, delay, drop, _ in pts:
        label, color, marker = style_of(pol)
        h = axes[0].scatter([acc], [delay], color=color, marker=marker, s=60, zorder=3)
        axes[1].scatter([acc], [100 * drop], color=color, marker=marker, s=60, zorder=3)
        items.append((h, f"{label}：准确率 {acc:.1f}%，延迟 {delay:.2f} 槽，丢弃 {100 * drop:.1f}%"))
    tidy(axes[0], "闭环平均延迟（只算完成的任务）", "开环准确率 %", "槽")
    tidy(axes[1], "闭环丢弃率", "开环准确率 %", "%")
    cluster = []
    for p in pts:
        if p[1] > 95:
            cluster.append(p)
    if len(cluster) >= 2:
        ins = axes[0].inset_axes([0.5, 0.08, 0.42, 0.42])
        for pol, acc, delay, drop, _ in cluster:
            label, color, marker = style_of(pol)
            ins.scatter([acc], [delay], color=color, marker=marker, s=40, zorder=3)
        accs = [p[1] for p in cluster]
        delays = [p[2] for p in cluster]
        pad_y = max(0.01, 0.15 * (max(delays) - min(delays)))
        ins.set_xlim(min(accs) - 0.3, max(accs) + 0.3)
        ins.set_ylim(min(delays) - pad_y, max(delays) + pad_y)
        ins.tick_params(labelsize=7, colors=MUTED)
        ins.grid(True, color=GRID, linewidth=0.6)
        axes[0].indicate_inset_zoom(ins, edgecolor=MUTED)
    fig.suptitle("实验 5 · 开环准确率高，闭环不一定好（3 台服务器，N = 40，虚拟时间闭环）", x=0.01, ha="left", fontsize=12,
                 color=INK)
    note = ("每点 1000 槽、种子 0。开环准确率 = 和单步最优（论文公式）选得一样的比例。PS 感知贪心、0.5B 平分算力标签和论文公式常常选得不一样，"
            "开环准确率只有约 20%，闭环里却最快：论文公式高估平分算力服务器的排队。")
    for pol, acc, delay, drop, _ in pts:
        if pol == "random":
            note += f"\n随机的平均延迟看着和单步最优差不多，是因为丢了 {100 * drop:.0f}% 的任务（平均延迟只算完成的），看右图。"
    finish(fig, [(list(axes), items, 2)], note)
    numbers.append(("开环 vs 闭环", "；".join(f"{style_of(p)[0]} 准确率 {a:.1f}% → 延迟 {d:.2f} 槽、丢弃 {100 * x:.1f}%"
                                         for p, a, d, x, _ in pts), pts[0][4]))
    return fig


# 图 6 左图：开环表里的名字 → 图上的名字，从上往下的顺序
ABLATION_MODELS = [
    ("公式·压缩精度（学生上限）", "公式（学生上限）"),
    ("0.5b_soft_T0.3", "0.5B 软标签（部署）"),
    ("1.5b_soft_T0.3", "1.5B 软标签"),
    ("1.5b_hard", "1.5B 硬标签"),
    ("1.5b_hard + GRPO（fixed 奖励）", "1.5B 硬标签 + GRPO"),
    ("论文原版 sft", "论文原版 sft"),
    ("论文原版 grpo", "论文原版 grpo"),
    ("论文原版 score-grpo", "论文原版 score-grpo"),
    ("GRPO-fixed（从论文原版 sft）", "论文原版 sft + GRPO（fixed 奖励）"),
    ("GRPO-repo（从论文原版 sft）", "论文原版 sft + GRPO（原作者仓库奖励）"),
]


def fig6_ablation(rows, open_rows, numbers):
    """图 6 消融：左 = 开环准确率（软 vs 硬标签、GRPO、论文原版几个模型），横条；右 = 平分算力标签消融的闭环平均延迟（3 台，按 N）。

    1. 左图每个模型一组横条：见过的台数 / 没见过的台数 / 原作者干净测试集，条末写数值；没测的写"未测"
    2. 某个模型在某个数据集上只测了一部分（条数和公式那一行不一样），画成斜线空心条，图例里写清楚条数，不能和整套直接比
    3. 每个模型三个数据集的准确率写进数字表（README 结果表要用）
    4. 右图：3 台、顺序决策、1000 槽，同一 N 几个种子取平均
    """
    fig = plt.figure(figsize=(15, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    splits = (("test_seen", "见过的台数", "#2a78d6"), ("test_unseen", "没见过的台数", "#1baf7a"),
              ("friend_clean", "原作者干净测试集", "#eb6834"))
    full = {}
    for split, _, _ in splits:
        r = open_rows.get(("公式·压缩精度（学生上限）", split, "全部"))
        full[split] = r["n"] if r is not None else None
    present = []
    for name, label in ABLATION_MODELS:
        for split, _, _ in splits:
            if (name, split, "全部") in open_rows:
                present.append((name, label))
                break
    bar_h = 0.26
    subsets = []
    for i in range(len(present)):
        name, label = present[i]
        for j in range(len(splits)):
            split, slabel, color = splits[j]
            y = i + (j - 1) * bar_h
            r = open_rows.get((name, split, "全部"))
            if r is None:
                ax.text(1, y, "未测", va="center", fontsize=7, color=MUTED)
                continue
            v = float(r["acc_pct"])
            if full[split] is not None and r["n"] != full[split]:
                ax.barh(y, v, height=bar_h, color="white", edgecolor=color, hatch="///", linewidth=0.8)
                subsets.append((slabel, r["n"], full[split], color))
            else:
                ax.barh(y, v, height=bar_h, color=color)
            ax.text(v + 0.8, y, f"{v:.1f}", va="center", fontsize=7, color=INK)
    for name, label in present:
        parts = []
        for split, slabel, _ in splits:
            r = open_rows.get((name, split, "全部"))
            if r is not None:
                parts.append(f"{slabel} {float(r['acc_pct']):.1f}%（{r['n']} 条）")
        numbers.append((f"开环准确率（{label}）", "；".join(parts) + "（实测，贪心解码）",
                        "evidence/G3/open_loop.csv、evidence/G4/open_loop.csv"))
    ax.set_yticks(range(len(present)))
    labels = []
    for name, label in present:
        labels.append(label)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    tidy(ax, "开环准确率（阶段 3、4，和单步最优选得一样的比例）", "%", "")
    ax.grid(False, axis="y")
    items = []
    for split, slabel, color in splits:
        items.append((Patch(color=color), slabel))
    done = set()
    for slabel, n, n_full, color in subsets:
        if (slabel, n) in done:
            continue
        done.add((slabel, n))
        items.append((Patch(facecolor="white", edgecolor=color, hatch="///"),
                      f"斜线：{slabel}只测了 {n} 条（整套 {n_full} 条），不能和实心条直接比"))
    by = {}
    drop = {}
    per_seed = {}
    dirs = set()
    for r in rows:
        if int(r["servers"]) == 3 and r["mode"] == "sequential" and int(r["slots"]) == 1000 and \
                r["policy"] in ("greedy", "psgreedy", "llm:main_0.5b", "llm:ps_0.5b"):
            by.setdefault(r["policy"], {}).setdefault(int(r["n"]), []).append(float(r["mean_delay"]))
            drop.setdefault(r["policy"], {}).setdefault(int(r["n"]), []).append(float(r["drop_rate"]))
            per_seed.setdefault(r["policy"], {}).setdefault(int(r["n"]), {})[int(r["seed"])] = float(r["mean_delay"])
            dirs.add(r["_dir"])
    for pol in ("greedy", "psgreedy", "llm:main_0.5b", "llm:ps_0.5b"):
        if pol not in by:
            continue
        ns = sorted(by[pol])
        label, color, marker = style_of(pol)
        ax_r.plot(ns, [mean(by[pol][n]) for n in ns], color=color, marker=marker,
                  linestyle="--" if not pol.startswith("llm:") else "-", label=label, linewidth=1.8)
    tidy(ax_r, "实验 4 · 平分算力标签消融（虚拟时间闭环，3 台）", "终端数 N", "平均延迟（槽）")
    if by:
        ax_r.set_xticks(sorted({n for v in by.values() for n in v}))
    if "llm:main_0.5b" in by and "llm:ps_0.5b" in by:
        parts = []
        for n in sorted(by["llm:main_0.5b"]):
            if n in by["llm:ps_0.5b"]:
                a = mean(by["llm:main_0.5b"][n])
                b = mean(by["llm:ps_0.5b"][n])
                da = 100 * mean(drop["llm:main_0.5b"][n])
                db = 100 * mean(drop["llm:ps_0.5b"][n])
                parts.append(f"N = {n}：{a:.2f} → {b:.2f} 槽（{100 * (b - a) / a:+.1f}%），丢弃 {da:.1f}% → {db:.1f}%")
        seeds = len(by["llm:ps_0.5b"][min(by["llm:ps_0.5b"])])
        numbers.append(("平分算力标签 0.5B vs 主线 0.5B（模拟器按论文式 4 的平分算力排队）",
                        "；".join(parts) + f"（仿真实测，{seeds} 个种子）", ", ".join(sorted(dirs))))
        parts = []
        used = set()
        for stu, tea, name in (("llm:main_0.5b", "greedy", "论文公式老师"), ("llm:ps_0.5b", "psgreedy", "PS 老师")):
            gaps = []
            for n in sorted(per_seed.get(stu, {})):
                common = sorted(set(per_seed[stu][n]) & set(per_seed.get(tea, {}).get(n, {})))
                if common:
                    a = mean([per_seed[stu][n][k] for k in common])
                    b = mean([per_seed[tea][n][k] for k in common])
                    gaps.append(100 * (a - b) / b)
                    used.update(common)
            if gaps:
                parts.append(f"{name}：0.5B 比老师 {min(gaps):+.1f}% ~ {max(gaps):+.1f}%")
        seed_text = "同一个种子 0" if used == {0} else f"同样的 {len(used)} 个种子"
        numbers.append((f"两个 0.5B 学生 vs 各自的老师（3 台，N = 20/40/60，{seed_text}）",
                        "；".join(parts) + "（仿真实测）", ", ".join(sorted(dirs))))
    n_llm = 0
    for pol in ("llm:main_0.5b", "llm:ps_0.5b"):
        for n in per_seed.get(pol, {}):
            n_llm = max(n_llm, len(per_seed[pol][n]))
    n_formula = 0
    for pol in ("greedy", "psgreedy"):
        for n in per_seed.get(pol, {}):
            n_formula = max(n_formula, len(per_seed[pol][n]))
    note = ("右图：虚线是两种公式老师，实线是照着它们训练的 0.5B 学生；模拟器按论文式 4 的平分算力（PS）算服务时间，结论只适用于这种服务器。\n"
            f"公式点是 {n_formula} 个种子的平均，LLM 点是 {n_llm} 个种子的平均（LLM 点要在 Jetson 上逐个作答）。")
    finish(fig, [([ax], items, 2), ([ax_r], None, 2)], note)
    return fig


def fig7_lacs(rows, numbers):
    """图 7 LACS 往后看在闭环里有没有用：3/7/11 台各一列，横轴 N；上排 = 顺序决策平均延迟相对单步最优的变化（%），下排 = 丢弃率。

    1. 只用实验 6 的行（2000 槽），同一 (台数, N, 策略, 决策方式) 几个种子取平均
    2. 上排画 λ = 0.3/1/2 和 PS 感知贪心相对单步最优（λ = 0）的变化，单步最优就是 0 那条线；
       λ = 0.3 和单步最优几乎选得一样，画绝对延迟时几条线叠成一条，所以画相对变化
       下排画各策略（含单步最优）的丢弃率，因为平均延迟只算完成的任务
    3. 数字：λ = 0.3 闭环里和单步最优选得一样的比例；每个 λ 相对单步最优的平均延迟变化（全部点平均）；
       批量决策下 λ = 0.3 有没有减轻羊群效应（11 台、N 最大）
    """
    cell = {}
    for r in rows:
        if int(r["slots"]) != 2000 or not (r["policy"] in ("greedy", "psgreedy") or r["policy"].startswith("lacs")):
            continue
        cell.setdefault((r["policy"], int(r["servers"]), int(r["n"]), r["mode"]), []).append(r)
    if not cell:
        return None
    pols = ["greedy", "lacs", "lacs:1", "lacs:2"]
    names = {"lacs": "LACS λ = 0.3（论文）", "lacs:1": "LACS λ = 1", "lacs:2": "LACS λ = 2",
             "psgreedy": "PS 感知贪心（改公式，参照）"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [1.5, 1]})
    missing = False
    for i, k in enumerate((3, 7, 11)):
        top = axes[0][i]
        bottom = axes[1][i]
        top.axhline(0, color="#2a78d6", linewidth=2, label="单步最优（λ = 0，基准）")
        g_ns = sorted({n for (p, kk, n, m) in cell if p == "greedy" and kk == k and m == "sequential"})
        g_drop = []
        for n in g_ns:
            g_drop.append(100.0 * mean([float(r["drop_rate"]) for r in cell[("greedy", k, n, "sequential")]]))
        bottom.plot(g_ns, g_drop, color="#2a78d6", linewidth=2, marker="o", markersize=4)
        for pol in pols[1:] + ["psgreedy"]:
            ns = []
            ys = []
            drops = []
            for n in sorted({n for (p, kk, n, m) in cell if p == pol and kk == k and m == "sequential"}):
                g = cell.get(("greedy", k, n, "sequential"))
                if not g:
                    continue
                v = cell[(pol, k, n, "sequential")]
                base = mean([float(r["mean_delay"]) for r in g])
                ns.append(n)
                ys.append(100.0 * (mean([float(r["mean_delay"]) for r in v]) - base) / base)
                drops.append(100.0 * mean([float(r["drop_rate"]) for r in v]))
            if not ns:
                missing = True
                continue
            label, color, marker = style_of(pol)
            ls = "--" if pol == "psgreedy" else "-"
            top.plot(ns, ys, color=color, marker=marker, label=names[pol], linewidth=1.8, linestyle=ls)
            bottom.plot(ns, drops, color=color, marker=marker, linewidth=1.4, linestyle=ls, markersize=4)
        tidy(top, f"{k} 台服务器", "", "平均延迟相对单步最优，%" if i == 0 else "")
        tidy(bottom, "", "终端数 N", "丢弃率 %" if i == 0 else "")
        bottom.set_xticks([20, 40, 60, 80])
    fig.suptitle("实验 6 · LACS 往后看在闭环里有没有用（虚拟时间闭环，顺序决策；上排负数 = 比单步最优快，下排是丢弃率）",
                 x=0.01, ha="left", fontsize=12, color=INK)
    note = ("每点 2000 槽、3 个种子取平均。平均延迟只算完成的任务，丢得多时会显得低，所以上下两排要一起看。\n"
            "λ = 0.3 和单步最优几乎每次选得一样（两条线重合在 0）；λ = 1 时两边打平的情况很多，平局时选法不同，结果在 0 附近小幅摆动。")
    if missing:
        note += "\n实验 6 还没跑完，缺的线没画（python -m eval.virtual_loop --plan e6 --resume runs/<目录> 接着跑）。"
    items = []
    seen = set()
    for ax in axes[0]:
        handles, labels = ax.get_legend_handles_labels()
        for j in range(len(handles)):
            if labels[j] not in seen:
                seen.add(labels[j])
                items.append((handles[j], labels[j]))
    finish(fig, [(list(axes[1]), items, 5)], note)
    dirs = ", ".join(sorted({r["_dir"] for v in cell.values() for r in v}))
    agree = [float(r["agree_greedy"]) for (p, k, n, m), v in cell.items() if p == "lacs" and m == "sequential"
             for r in v]
    if agree:
        numbers.append(("LACS λ = 0.3 闭环里和单步最优选得一样", f"{100 * mean(agree):.1f}%（仿真实测，顺序决策全部点）", dirs))
    for pol in pols[1:] + ["psgreedy"]:
        diffs = []
        for (p, k, n, m), v in cell.items():
            g = cell.get(("greedy", k, n, m))
            if p != pol or m != "sequential" or not g:
                continue
            base = mean([float(r["mean_delay"]) for r in g])
            diffs.append(100.0 * (mean([float(r["mean_delay"]) for r in v]) - base) / base)
        if diffs:
            numbers.append((f"{names[pol]} 相对单步最优的平均延迟变化",
                            f"平均 {mean(diffs):+.2f}%，范围 {min(diffs):+.2f}% ~ {max(diffs):+.2f}%（仿真实测，顺序决策 {len(diffs)} 个点）",
                            dirs))
    ns11 = sorted({n for (p, k, n, m) in cell if k == 11 and m == "batch"})
    if ns11:
        n = ns11[-1]
        g = mean([float(r["mean_delay"]) for r in cell.get(("greedy", 11, n, "batch"), [])])
        lam = mean([float(r["mean_delay"]) for r in cell.get(("lacs", 11, n, "batch"), [])])
        numbers.append((f"批量决策下 LACS 能否减轻羊群效应（11 台、N = {n}）",
                        f"单步最优 {g:.2f} 槽，LACS λ = 0.3 {lam:.2f} 槽（仿真实测）", dirs))
    return fig

def fig8_students(rows, open_rows, numbers, platform="jetson"):
    """图 8 两个老师 × 三种学生（3 台，N = 20/40/60，顺序决策，1000 槽）：左 = 平均延迟，右 = 丢弃率。

    platform：jetson = 学生在 Jetson 上用 GGUF Q8_0 作答（eval.virtual_loop）；colab = 在 Colab 上用 PyTorch 原精度作答
    （eval.colab_loop）。老师是公式，在哪儿算都一样，两张图用同一份老师的点。Colab 那张图有复现的 COMLLM-7B 时也画上。

    1. 收集 3 台、顺序决策、1000 槽的点：每个 (策略, N) 下按种子存平均延迟和丢弃率
    2. 只用学生们都跑过的种子（老师跑过的种子更多，也只取这几个），所有线在同一批任务上比
    3. 学生少于 3 个，或学生的 N 不全，就不画（返回 None），等 e4ps 跑完
    4. 每个老师画一条虚线，三个学生画实线；同一个老师的学生用同一个色系，点形区分 0.5B 软 / 1.5B 软 / 1.5B 硬
    5. 数字表：每个老师一行（闭环）；每个学生一行（开环准确率和 PR + 闭环 + 比老师）；同样的学生换老师差多少一行
    """
    # 1. 收集
    cell = {}
    dirs = set()
    for r in rows:
        if int(r["servers"]) != 3 or r["mode"] != "sequential" or int(r["slots"]) != 1000:
            continue
        cell.setdefault(r["policy"], {}).setdefault(int(r["n"]), {})[int(r["seed"])] = \
            (float(r["mean_delay"]), float(r["drop_rate"]), r["_dir"])
    ns = [20, 40, 60]
    students = []
    for _, _, _, studs in TEACHER_GRID:
        for key, _, _, _ in studs:
            if key in cell and all(n in cell[key] for n in ns):
                students.append(key)
    if len(students) < 3:
        return None
    # 2. 共同的种子
    seeds = None
    for key in students:
        for n in ns:
            have = set(cell[key][n])
            seeds = have if seeds is None else seeds & have
    seeds = sorted(seeds or [])
    if not seeds:
        return None

    def avg(key, n):
        """这个策略在 N 下、共同种子上的 (平均延迟, 丢弃率)；缺任何一个种子就返回 None。"""
        got = cell.get(key, {}).get(n, {})
        for k in seeds:
            if k not in got:
                return None
        for k in seeds:
            dirs.add(got[k][2])
        return mean([got[k][0] for k in seeds]), mean([got[k][1] for k in seeds])

    # 3/4. 画
    fig, (ax_d, ax_x) = plt.subplots(1, 2, figsize=(13, 5.2))
    items = []
    extra = []
    for tname, tkey, tcolor, studs in TEACHER_GRID:
        vals = [avg(tkey, n) for n in ns]
        if all(v is not None for v in vals):
            ax_d.plot(ns, [v[0] for v in vals], color=tcolor, linestyle="--", linewidth=2.4, alpha=0.8)
            ax_x.plot(ns, [100 * v[1] for v in vals], color=tcolor, linestyle="--", linewidth=2.4, alpha=0.8)
            items.append((Line2D([], [], color=tcolor, linestyle="--", linewidth=2.4), f"{tname}（老师）"))
        else:
            items.append((Line2D([], [], color=tcolor, linestyle="--", linewidth=2.4, alpha=0.3), f"{tname}（老师，没测）"))
        for key, slabel, color, marker in studs:
            if key not in students:
                items.append((Line2D([], [], color=color, marker=marker, linewidth=1.6, alpha=0.3),
                              f"{tname}的学生 · {slabel}（没测）"))
                continue
            vals = [avg(key, n) for n in ns]
            ax_d.plot(ns, [v[0] for v in vals], color=color, marker=marker, linewidth=1.6, markersize=5)
            ax_x.plot(ns, [100 * v[1] for v in vals], color=color, marker=marker, linewidth=1.6, markersize=5)
            items.append((Line2D([], [], color=color, marker=marker, linewidth=1.6), f"{tname}的学生 · {slabel}"))
    plat = PLATFORM_NAME[platform]
    for key, short, tag, color in PAPER_MODELS_CLOSED:
        vals = [avg(key, n) for n in ns]
        if all(v is not None for v in vals):
            ax_d.plot(ns, [v[0] for v in vals], color=color, linestyle=":", marker="D", linewidth=1.8)
            ax_x.plot(ns, [100 * v[1] for v in vals], color=color, linestyle=":", marker="D", linewidth=1.8)
            extra.append((Line2D([], [], color=color, linestyle=":", marker="D", linewidth=1.8), short, tag, vals))
    tidy(ax_d, f"两个老师 × 三种学生 · 平均延迟（虚拟时间闭环，3 台，{plat}）", "终端数 N", "平均延迟（槽）")
    tidy(ax_x, f"两个老师 × 三种学生 · 丢弃率（{plat}）", "终端数 N", "丢弃率（%）")
    for ax in (ax_d, ax_x):
        ax.set_xticks(ns)
    # 图例两行：第一行论文公式那一组，第二行 PS 那一组（matplotlib 按列填，所以交错排）
    half = len(items) // 2
    ordered = []
    for i in range(half):
        ordered.append(items[i])
        ordered.append(items[half + i])
    for handle, short, tag, _ in extra:
        ordered.append((handle, f"{short}（{tag}）"))
    who = "LLM 在 Jetson 上用 GGUF Q8_0 逐个作答" if platform == "jetson" else "LLM 在 Colab 上用 PyTorch 原精度（bf16）作答"
    note = (f"每条线是同样 {len(seeds)} 个种子（{', '.join(str(k) for k in seeds)}）的平均，所有线在同一批任务上比。"
            f"虚线是公式老师，实线是照着它训练的学生（{who}）"
            + ("，点线是论文的模型" if extra else "") + "。\n"
            "模拟器按论文式 4 的平分算力（PS）算服务时间，结论只适用于这种服务器。")
    finish(fig, [([ax_d, ax_x], ordered, half)], note)

    # 5. 数字表
    src = ", ".join(sorted(dirs))
    seed_text = f"{len(seeds)} 个种子"
    for tname, tkey, _, studs in TEACHER_GRID:
        tv = [avg(tkey, n) for n in ns]
        t_ok = all(v is not None for v in tv)
        if t_ok and platform == "jetson":
            numbers.append((f"老师：{tname}（3 台，N = 20/40/60）",
                            "平均延迟 " + " / ".join(f"{v[0]:.2f}" for v in tv) + " 槽，丢弃 "
                            + " / ".join(f"{100 * v[1]:.1f}%" for v in tv) + f"（仿真实测，{seed_text}）", src))
        for key, slabel, _, _ in studs:
            parts = []
            r = open_rows.get((OPEN_NAME.get(key, ""), "friend_clean", "全部"))
            if r is not None:
                parts.append(f"开环准确率 {float(r['acc_pct']):.1f}%、PR {float(r['pr_pct']):.2f}%"
                             "（原作者干净测试集，尺子是论文公式）")
            if key in students:
                sv = [avg(key, n) for n in ns]
                parts.append("闭环平均延迟 " + " / ".join(f"{v[0]:.2f}" for v in sv) + " 槽，丢弃 "
                             + " / ".join(f"{100 * v[1]:.1f}%" for v in sv) + f"（仿真实测，{seed_text}）")
                if t_ok:
                    gaps = []
                    for i in range(len(ns)):
                        gaps.append(100 * (sv[i][0] - tv[i][0]) / tv[i][0])
                    parts.append(f"比老师 {min(gaps):+.1f}% ~ {max(gaps):+.1f}%")
            else:
                parts.append("闭环没测")
            numbers.append((f"学生：{tname}老师 · {slabel}（3 台，N = 20/40/60，{plat}）", "；".join(parts), src))
    paper_studs = TEACHER_GRID[0][3]
    ps_studs = TEACHER_GRID[1][3]
    parts = []
    for i in range(len(paper_studs)):
        a_key, slabel = paper_studs[i][0], paper_studs[i][1]
        b_key = ps_studs[i][0]
        if a_key in students and b_key in students:
            cells = []
            for n in ns:
                a = avg(a_key, n)
                b = avg(b_key, n)
                cells.append(f"N = {n} 延迟 {100 * (b[0] - a[0]) / a[0]:+.1f}%、丢弃 {100 * a[1]:.1f}% → {100 * b[1]:.1f}%")
            parts.append(f"{slabel}：" + "，".join(cells))
    if parts:
        numbers.append((f"同样的学生，PS 老师 vs 论文公式老师（3 台，{plat}）", "；".join(parts) + f"（仿真实测，{seed_text}）",
                        src))
    for _, short, tag, vals in extra:
        numbers.append((f"{short}（{tag}，3 台，N = 20/40/60，{plat}）",
                        "闭环平均延迟 " + " / ".join(f"{v[0]:.2f}" for v in vals) + " 槽，丢弃 "
                        + " / ".join(f"{100 * v[1]:.1f}%" for v in vals) + f"（仿真实测，{seed_text}）", src))
    return fig


def deploy_gap(virtual, colab, numbers):
    """同一个学生、同一个点，Jetson（GGUF Q8_0）比 Colab（PyTorch 原精度）的闭环平均延迟差多少。

    1. 两边按点的名字对上（eval.colab_loop 的点名和 eval.virtual_loop 的一样）
    2. 每个学生、每个 N：两边都有的种子取平均，算 (Jetson − Colab) ÷ Colab
    3. 每个学生一段，写进一行数字
    """
    by_point = {}
    for r in virtual:
        by_point[r["point"]] = r
    parts = []
    dirs = set()
    for tname, _, _, studs in TEACHER_GRID:
        for key, slabel, _, _ in studs:
            per_n = {}
            for r in colab:
                if r["policy"] != key or r["point"] not in by_point:
                    continue
                j = by_point[r["point"]]
                per_n.setdefault(int(r["n"]), []).append((float(j["mean_delay"]), float(r["mean_delay"])))
                dirs.add(r["_dir"])
                dirs.add(j["_dir"])
            if not per_n:
                continue
            cells = []
            for n in sorted(per_n):
                a = mean([x[0] for x in per_n[n]])
                b = mean([x[1] for x in per_n[n]])
                cells.append(f"N = {n} {100 * (a - b) / b:+.1f}%")
            parts.append(f"{tname}老师 · {slabel}：" + "，".join(cells))
    if parts:
        numbers.append(("部署前后的闭环平均延迟（Jetson Q8_0 比 Colab 原精度，3 台）", "；".join(parts) + "（仿真实测）",
                        ", ".join(sorted(dirs))))
    return bool(parts)


# ===== 第 3 段：主程序 =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rediscover", action="store_true", help="重新找最新的运行目录，覆盖 sources.json")
    ap.add_argument("--include-rehearse", action="store_true", help="预演目录也算（只用来试脚本）")
    ap.add_argument("--sources", default=SOURCES)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    # 1. 定数据来源：有 sources.json 就照它，没有（或 --rediscover）就自动找并写出来
    if os.path.exists(args.sources) and not args.rediscover:
        with open(args.sources, encoding="utf-8") as f:
            src = json.load(f)
    else:
        src = discover(args.include_rehearse)
        os.makedirs(os.path.dirname(args.sources), exist_ok=True)
        with open(args.sources, "w", encoding="utf-8") as f:
            json.dump(src, f, ensure_ascii=False, indent=2)
        print("写好了数据来源清单：", args.sources)

    # 2. 读数据
    e1 = load_rows(src.get("e1", []))
    twoboard = load_rows(src.get("twoboard", []))
    virtual = load_rows(src.get("virtual", []))
    bench_all = load_bench(src.get("bench", []))
    bench = [b for b in bench_all if b["_env"] == "mac"]       # 图 1、图 2 只用 Mac 当环境节点时测的
    open_rows = load_open(src.get("open_loop", []))
    colab = load_rows(src.get("colab", []))
    teachers = [r for r in virtual if not r["policy"].startswith("llm:")]

    # 3. 逐张画；缺数据的跳过
    os.makedirs(args.out, exist_ok=True)
    numbers = []
    jobs = [
        ("fig1_stability.png", lambda: fig1_stability(e1, bench, numbers) if e1 else None, "实验 1 扫描（eval.s5_sweep）"),
        ("fig2_latency.png", lambda: fig2_latency(bench, e1, numbers), "延迟基准（eval.bench_latency）"),
        ("fig3_herding.png", lambda: fig3_herding(virtual, numbers), "实验 3（eval.virtual_loop --plan e3）"),
        ("fig4_generalization.png", lambda: fig4_generalization(virtual, open_rows, numbers), "开环表 / e4b"),
        ("fig5_open_vs_closed.png", lambda: fig5_open_vs_closed(virtual, open_rows, numbers), "e5 + 开环表"),
        ("fig6_ablation.png", lambda: fig6_ablation(virtual, open_rows, numbers), "开环表 / e4a"),
        ("fig7_lacs.png", lambda: fig7_lacs(virtual, numbers), "实验 6（eval.virtual_loop --plan e6）"),
        ("fig8_students.png", lambda: fig8_students(virtual, open_rows, numbers),
         "两个老师 × 三种学生（eval.virtual_loop --plan e4ps）"),
        ("fig8_students_colab.png",
         lambda: fig8_students(teachers + colab, open_rows, numbers, platform="colab") if colab else None,
         "Colab 闭环（train/S6_Colab闭环.ipynb）"),
    ]
    for fname, make, need in jobs:
        fig = make()
        if fig is None:
            print(f"跳过 {fname}：还没有 {need} 的数据")
            continue
        fig.savefig(os.path.join(args.out, fname), dpi=150, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print("画好了：", os.path.join(args.out, fname))

    if colab and deploy_gap(virtual, colab, numbers):
        print("算好了部署前后的闭环差别（Jetson vs Colab）")

    # 4. 两板对照表（跑过两板复测才有）
    if twoboard_table(e1, twoboard, bench_all, os.path.join(args.out, "twoboard.md"), numbers):
        print("写好了：", os.path.join(args.out, "twoboard.md"))

    # 5. 简历数字和出处
    with open(os.path.join(args.out, "numbers.md"), "w", encoding="utf-8") as f:
        f.write("# 关键数字\n\n由 `python -m eval.figures` 生成，README 和简历里的数字都从这里取。每行最后一列是数据所在的运行目录（runs/ 下）或开环表。\n\n")
        f.write("| 数字 | 值 | 运行目录 |\n|---|---|---|\n")
        for name, value, where in numbers:
            f.write(f"| {name} | {value} | {where} |\n")
    print("写好了：", os.path.join(args.out, "numbers.md"))


if __name__ == "__main__":
    main()
