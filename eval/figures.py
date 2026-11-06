"""一条命令从 CSV 重画全部 7 张图，并把简历要用的几个数字连同出处写成 figures/numbers.md（G5 的要求）。

在仓库根目录运行（Mac 上）：
  python -m eval.figures                 # 读 evidence/G5/sources.json 里登记的运行目录；没有这个文件就自动找最新的并写出来
  python -m eval.figures --rediscover    # 重新找最新的运行目录，覆盖 sources.json
结果：figures/fig1_stability.png … fig7_lacs.png、figures/numbers.md；跑过两板复测时还有 figures/twoboard.md。
某张图的数据还没跑出来时，那张图跳过，屏幕上会说缺什么。

数据从哪来（sources.json 的键）：
  e1          eval/s5_sweep.py 的扫描目录（results.csv）：实验 1 稳定边界，可以列几个目录
  twoboard    eval/s5_sweep.py --plan twoboard 的扫描目录：RK3588 当环境节点时的复测，和 e1 里同名的点对照
  bench       eval/bench_latency.py 的运行目录：实验 2 延迟分解，同一配置只取最新一次
  virtual     eval/virtual_loop.py 的目录：实验 3 羊群效应、实验 4 消融和泛化、实验 5 开环 vs 闭环、实验 6 LACS
  open_loop   阶段 3、4 的开环评测表 evidence/G3/open_loop.csv、evidence/G4/open_loop.csv
"""
import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

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
    "lacs:1":         ("LACS λ = 1", "#c4501f", "s"),
    "lacs:2":         ("LACS λ = 2", "#8a3412", "s"),
    "psgreedy":       ("PS 感知贪心", "#e87ba4", "v"),
    "po2":            ("power-of-2", "#1baf7a", "^"),
    "random":         ("随机", "#eda100", "D"),
    "smallnet":       ("小网络 1313 参数", "#7a5af5", "P"),
    "main_0.5b":      ("0.5B 软标签（部署）", "#0b0b0b", "o"),
    "main_0.5b_q4":   ("0.5B 软标签 Q4_K_M", "#52514e", "o"),
    "soft_1.5b":      ("1.5B 软标签", "#8c564b", "s"),
    "ps_0.5b":        ("0.5B 平分算力标签", "#c2185b", "v"),
    "base_0.5b":      ("未训练 0.5B", "#9e9e9e", "x"),
    "paper_sft_1.5b": ("论文原版 1.5B（sft）", "#d62728", "X"),
}

# 虚拟时间闭环的策略名 → 开环表（evidence/G3/open_loop.csv）里的名字
OPEN_NAME = {"greedy": "公式·压缩精度（学生上限）", "lacs": "LACS-greedy", "psgreedy": "PS 感知贪心",
             "po2": "power-of-2", "random": "随机", "smallnet": "小网络 1313 参数",
             "llm:main_0.5b": "0.5b_soft_T0.3", "llm:ps_0.5b": "0.5b_soft_T0.3_ps",
             "llm:soft_1.5b": "1.5b_soft_T0.3", "llm:base_0.5b": "未训练 0.5B", "llm:paper_sft_1.5b": "论文原版 sft"}


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


# ===== 第 1 段：找数据 =====
def discover(include_rehearse=False):
    """自动找最新的运行目录，返回和 sources.json 同样结构的字典。

    1. e1：runs/*_s5_*/results.csv 所在的扫描目录；名字带 twoboard 的归到 twoboard
    2. virtual：runs/*_s5_virtual_*/ 全部（不同计划可能分几次跑）
    3. bench：runs/*bench*/summary.json 全部（画图时同一配置只取最新）
    4. 开环表：evidence/G3、G4 的 open_loop.csv
    预演目录（名字带 rehearse）默认不算
    """
    src = {"e1": [], "twoboard": [], "virtual": [], "bench": [], "open_loop": []}
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


# ===== 第 2 段：六张图 =====
def fig1_stability(rows, bench, numbers):
    """图 1 稳定边界：横轴 N，左平均延迟、右丢弃率；每个 (模型, 是否分流) 一条线，竖虚线是理论 N_max。

    1. 按 (模型, 分流) 分组，同一 N 的几个种子取平均，竖线画最小到最大
    2. 理论 N_max = 1 / (3 × 单次决策平均秒数)：优先用实验 1 里最小 N 那次的往返（和实验 1 同样 3 台服务器），
       没有才用实验 2 的延迟基准（基准混了 3/7/11 台，提示词更长，算出的 N_max 偏小）
    3. 稳定边界 M = 决策环节丢弃（过了截止期还没拿到决策）在每个种子上都低于 1% 的最大 N
    """
    groups = {}
    for r in rows:
        key = (r["model_key"], float(r["shed_ms"]) > 0)
        groups.setdefault(key, {}).setdefault(int(r["n"]), []).append(r)
    bench_mean = {}
    for s in bench:
        if s["fmt"] in ("compact", "formula", "paper") and s["cache"] == ("off" if s["fmt"] == "paper" else "on"):
            bench_mean[s["_key"]] = s["e2e_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for (key, shed), by_n in sorted(groups.items(), key=lambda kv: (kv[0][0] != "formula", kv[0])):
        name, color, marker = style_of(key)
        if shed:
            name = name + " + 超载分流"
        ns = sorted(by_n)
        for ax, field, scale in ((axes[0], "mean_delay", 1.0), (axes[1], "drop_rate", 100.0)):
            vals = [[float(r[field]) * scale for r in by_n[n]] for n in ns]
            m = [mean(v) for v in vals]
            lo = [m[i] - min(vals[i]) for i in range(len(ns))]
            hi = [max(vals[i]) - m[i] for i in range(len(ns))]
            ax.errorbar(ns, m, yerr=[lo, hi], color=color, marker=marker, linestyle="--" if shed else "-",
                        linewidth=2, markersize=5, label=name)
        stable = [n for n in ns if all(float(r["drop_decision"]) < 0.01 for r in by_n[n])]
        e2e = mean([float(r["e2e_ms_mean"]) for r in by_n[ns[0]]])
        if e2e != e2e:
            e2e = bench_mean.get(key, float("nan"))
        nmax = 1000.0 / (3.0 * e2e) if e2e and e2e == e2e else float("nan")
        if not shed and key != "formula" and nmax < 100:
            for ax in axes:
                ax.axvline(nmax, color=color, linestyle=":", linewidth=1)
        numbers.append((f"稳定边界 M（{name}）", f"{max(stable) if stable else '无'} 台终端（实测，测过的 N 里决策环节丢弃 < 1% 的最大值）；"
                        f"理论 N_max = 1/(3 × {e2e:.1f} ms) = {nmax:.1f}", ", ".join(sorted({r['_dir'] for v in by_n.values() for r in v}))))
    tidy(axes[0], "实验 1 · 平均延迟（只算完成的任务）", "终端数 N", "槽（0.1 s）")
    tidy(axes[1], "实验 1 · 丢弃率", "终端数 N", "%")
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_xticks([1, 2, 5, 10, 20, 40, 80])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1].legend(frameon=False, fontsize=9)
    fig.text(0.01, 0.01, "竖虚线：理论容量 N_max = 1/(3 × 单次决策平均秒数)。实时闭环，3 台服务器，每点 3 分钟。",
             fontsize=8, color=MUTED)
    return fig


def fig2_latency(bench, numbers):
    """图 2 决策延迟分解：每个配置一根横条，按 网络+HTTP / Python 服务 / llama-server 其他 / 预填充 / 解码 堆起来，
    条末标端到端 p50 / p99。从慢到快排，读起来就是"一步步省下来"。"""
    rows = sorted(bench, key=lambda s: -s["e2e_p50"])
    if not rows:
        return None
    parts = [("net_mean", "网络+HTTP", "#9e9e9e"), ("python_mean", "Python 服务", "#7a5af5"),
             ("llama_other_mean", "llama-server 其他", "#eda100"), ("prompt_ms_mean", "预填充", "#2a78d6"),
             ("predicted_ms_mean", "解码", "#eb6834")]
    fmt_name = {"paper": "论文格式", "compact": "压缩格式", "formula": "公式"}
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 1.6))
    labels = []
    for i, s in enumerate(rows):
        left = 0.0
        for k, name, color in parts:
            v = max(0.0, float(s.get(k, 0.0)))
            ax.barh(i, v, left=left, color=color, label=name if i == 0 else None)
            left += v
        ax.text(left + 5, i, f"p50 {s['e2e_p50']:.1f} / p99 {s['e2e_p99']:.1f} ms", va="center", fontsize=8,
                color=INK)
        labels.append(f"{style_of(s['_key'])[0]} · {fmt_name.get(s['fmt'], s['fmt'])} · 缓存{s['cache']}")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    tidy(ax, "实验 2 · 单次决策延迟分解（Jetson Orin Nano，环境节点上计时，500 次的平均）", "毫秒", "")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="lower right")
    for s in rows:
        if s["_key"] == "main_0.5b" and s["fmt"] == "compact" and s["cache"] == "on":
            numbers.append(("单次决策 X（部署配置）", f"p50 {s['e2e_p50']:.1f} ms，p99 {s['e2e_p99']:.1f} ms（实测）", s["_dir"]))
        if s["_key"] == "paper_sft_1.5b" and s["fmt"] == "paper" and s["cache"] == "off":
            numbers.append(("论文原版 1.5B 单次决策 Y", f"p50 {s['e2e_p50']:.1f} ms，p99 {s['e2e_p99']:.1f} ms（实测）", s["_dir"]))
    return fig


def fig3_herding(rows, numbers):
    """图 3 羊群效应：左图 = 批量决策比顺序决策多出的平均延迟（%），按台数和 N；右图 = 11 台、N 最大时每台服务器分到的任务占比。

    1. 只用实验 3 的行（公式策略，两种决策方式都有）
    2. 同一 (台数, N, 策略) 下几个种子取平均，算 (批量 − 顺序) / 批量，也就是"改成顺序以后降低了多少"
    3. 右图读 server_share 列
    """
    cell = {}
    for r in rows:
        if r["policy"] not in ("greedy", "psgreedy") or int(r["slots"]) != 1500:
            continue
        cell.setdefault((r["policy"], int(r["servers"]), int(r["n"]), r["mode"]), []).append(r)
    if not cell:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = {3: "#2a78d6", 7: "#1baf7a", 11: "#eb6834"}
    best = None
    for pol, ls in (("greedy", "-"), ("psgreedy", "--")):
        for k in (3, 7, 11):
            ns = sorted({n for (p, kk, n, m) in cell if p == pol and kk == k})
            ys = []
            for n in ns:
                seq = mean([float(r["mean_delay"]) for r in cell.get((pol, k, n, "sequential"), [])])
                bat = mean([float(r["mean_delay"]) for r in cell.get((pol, k, n, "batch"), [])])
                ys.append(100.0 * (bat - seq) / bat)
                if pol == "greedy" and k == 11 and (best is None or n > best[0]):
                    best = (n, seq, bat)
            axes[0].plot(ns, ys, color=colors[k], linestyle=ls, marker="o",
                         label=f"{k} 台 · {style_of(pol)[0]}")
    axes[0].axhline(0, color=MUTED, linewidth=1)
    tidy(axes[0], "实验 3 · 改成顺序决策后平均延迟降低多少", "终端数 N", "(批量 − 顺序) / 批量，%")
    axes[0].legend(frameon=False, fontsize=8)
    if best is not None:
        n, seq, bat = best
        share = {}
        for mode in ("sequential", "batch"):
            rs = cell.get(("greedy", 11, n, mode), [])
            vals = None
            for r in rs:
                v = [float(x) for x in r["server_share"].split(";")]
                vals = v if vals is None else [a + b for a, b in zip(vals, v)]
            share[mode] = [x / len(rs) for x in vals]
        idx = list(range(1, 12))
        axes[1].bar([i - 0.2 for i in idx], [100 * x for x in share["sequential"]], width=0.4, color="#2a78d6",
                    label="顺序决策 + 虚拟状态更新")
        axes[1].bar([i + 0.2 for i in idx], [100 * x for x in share["batch"]], width=0.4, color="#eb6834",
                    label="批量决策（共用过时状态）")
        axes[1].set_xticks(idx)
        tidy(axes[1], f"11 台、N = {n}、单步最优：每台服务器分到的卸载任务", "服务器编号", "占卸载任务 %")
        axes[1].legend(frameon=False, fontsize=8)
        dirs = ", ".join(sorted({r["_dir"] for v in cell.values() for r in v}))
        numbers.append((f"羊群效应修复 Z（11 台、N = {n}、单步最优）",
                        f"平均延迟 {bat:.2f} → {seq:.2f} 槽，降低 {100 * (bat - seq) / bat:.1f}%（仿真实测）", dirs))
    return fig


def fig4_generalization(rows, open_rows, numbers):
    """图 4 泛化：左 = 开环 PR（%）按服务器台数；右 = 虚拟时间闭环（N = 40）平均延迟按台数。4/6/8/10 台训练时没见过，灰底标出。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ks = list(range(3, 12))
    for name, key in (("0.5b_soft_T0.3", "main_0.5b"), ("1.5b_soft_T0.3", "soft_1.5b"),
                      ("论文原版 sft", "paper_sft_1.5b"), ("LACS-greedy", "lacs"), ("随机", "random")):
        xs, ys = [], []
        for k in ks:
            split = "test_seen" if k % 2 == 1 else "test_unseen"
            r = open_rows.get((name, split, str(k)))
            if r is not None:
                xs.append(k)
                ys.append(float(r["pr_pct"]))
        if xs:
            label, color, marker = style_of(key)
            axes[0].plot(xs, ys, color=color, marker=marker, label=label)
    tidy(axes[0], "开环 PR（阶段 3，论文式 28–29）", "服务器台数", "PR %")
    axes[0].legend(frameon=False, fontsize=8)
    cl = {}
    for r in rows:
        if int(r["n"]) == 40 and r["mode"] == "sequential" and int(r["seed"]) == 0 and int(r["slots"]) == 1000:
            cl.setdefault(r["policy"], {})[int(r["servers"])] = r
    for pol in ("greedy", "lacs", "psgreedy", "llm:main_0.5b"):
        if pol not in cl:
            continue
        xs = sorted(cl[pol])
        label, color, marker = style_of(pol)
        axes[1].plot(xs, [float(cl[pol][k]["mean_delay"]) for k in xs], color=color, marker=marker, label=label)
    tidy(axes[1], "虚拟时间闭环，N = 40 台终端", "服务器台数", "平均延迟（槽）")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        for k in (4, 6, 8, 10):
            ax.axvspan(k - 0.5, k + 0.5, color=GRID, alpha=0.6, linewidth=0)
        ax.set_xticks(ks)
    if "llm:main_0.5b" in cl and "greedy" in cl:
        gaps = []
        for k in sorted(cl["llm:main_0.5b"]):
            if k in cl["greedy"]:
                a = float(cl["llm:main_0.5b"][k]["mean_delay"])
                b = float(cl["greedy"][k]["mean_delay"])
                gaps.append(f"{k} 台 {100 * (a - b) / b:+.1f}%")
        numbers.append(("0.5B 闭环平均延迟 vs 单步最优（N = 40）", "；".join(gaps) + "（仿真实测）",
                        cl["llm:main_0.5b"][min(cl["llm:main_0.5b"])]["_dir"]))
    return fig


def fig5_open_vs_closed(rows, open_rows, numbers):
    """图 5 开环 vs 闭环：横轴 = 开环准确率（3 台服务器，test_seen），纵轴 = 虚拟时间闭环平均延迟（3 台、N = 40）。

    只取实验 5 的点：1000 槽（e5），论文原版 1.5B 太慢只跑了 400 槽（e5_paper）；实验 3 的 1500 槽点不算，免得同一策略出现两次。
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
    fig, ax = plt.subplots(figsize=(8, 5))
    for pol, acc, delay, drop, _ in pts:
        label, color, marker = style_of(pol)
        ax.scatter([acc], [delay], color=color, marker=marker, s=60, zorder=3)
        ax.annotate(f"{label}\n丢弃 {100 * drop:.1f}%", (acc, delay), textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=INK)
    tidy(ax, "实验 5 · 开环准确率高，闭环延迟不一定低（3 台服务器）", "开环准确率 %（和单步最优一致的比例）",
         "虚拟时间闭环平均延迟（槽，N = 40）")
    numbers.append(("开环 vs 闭环", "；".join(f"{style_of(p)[0]} 准确率 {a:.1f}% → 延迟 {d:.2f} 槽" for p, a, d, _, _ in pts),
                    pts[0][4]))
    return fig


def fig6_ablation(rows, open_rows, numbers):
    """图 6 消融：左 = 开环准确率（软 vs 硬标签、GRPO、论文原版三个模型）；右 = 平分算力标签消融的闭环平均延迟（3 台，按 N）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    names = ["0.5b_soft_T0.3", "1.5b_soft_T0.3", "1.5b_hard", "1.5b_hard + GRPO（fixed 奖励）",
             "论文原版 sft", "论文原版 grpo", "论文原版 score-grpo", "GRPO-fixed（从论文原版 sft）",
             "GRPO-repo（从论文原版 sft）"]
    splits = (("test_seen", "见过的台数", "#2a78d6"), ("test_unseen", "没见过的台数", "#1baf7a"),
              ("friend_clean", "原作者干净测试集", "#eb6834"))
    present = [n for n in names if (n, "test_seen", "全部") in open_rows or (n, "friend_clean", "全部") in open_rows]
    for j, (split, label, color) in enumerate(splits):
        xs, ys = [], []
        for i, n in enumerate(present):
            r = open_rows.get((n, split, "全部"))
            if r is not None:
                xs.append(i + (j - 1) * 0.27)
                ys.append(float(r["acc_pct"]))
        axes[0].bar(xs, ys, width=0.27, color=color, label=label)
    axes[0].set_xticks(range(len(present)))
    axes[0].set_xticklabels(present, rotation=30, ha="right", fontsize=7)
    tidy(axes[0], "开环准确率（阶段 3、4）", "", "%")
    axes[0].legend(frameon=False, fontsize=8)
    by = {}
    dirs = set()
    for r in rows:
        if int(r["servers"]) == 3 and r["mode"] == "sequential" and int(r["slots"]) == 1000 and \
                r["policy"] in ("greedy", "psgreedy", "llm:main_0.5b", "llm:ps_0.5b"):
            by.setdefault(r["policy"], {}).setdefault(int(r["n"]), []).append(float(r["mean_delay"]))
            dirs.add(r["_dir"])
    for pol in ("greedy", "psgreedy", "llm:main_0.5b", "llm:ps_0.5b"):
        if pol not in by:
            continue
        ns = sorted(by[pol])
        label, color, marker = style_of(pol)
        axes[1].plot(ns, [mean(by[pol][n]) for n in ns], color=color, marker=marker,
                     linestyle="--" if not pol.startswith("llm:") else "-", label=label)
    tidy(axes[1], "实验 4 · 平分算力标签消融（虚拟时间闭环，3 台）", "终端数 N", "平均延迟（槽）")
    axes[1].legend(frameon=False, fontsize=8)
    if "llm:main_0.5b" in by and "llm:ps_0.5b" in by:
        parts = []
        for n in sorted(by["llm:main_0.5b"]):
            if n in by["llm:ps_0.5b"]:
                a = mean(by["llm:main_0.5b"][n])
                b = mean(by["llm:ps_0.5b"][n])
                parts.append(f"N = {n}：{a:.2f} → {b:.2f} 槽（{100 * (b - a) / a:+.1f}%）")
        numbers.append(("平分算力标签 0.5B vs 主线 0.5B（模拟器按平分算力写，主场优势）", "；".join(parts) + "（仿真实测）",
                        ", ".join(sorted(dirs))))
    return fig


# ===== 第 3 段：主程序 =====
def fig7_lacs(rows, numbers):
    """图 7 LACS 往后看在闭环里有没有用：3/7/11 台各一格，横轴 N，纵轴顺序决策的平均延迟；每条线一个 λ。

    1. 只用实验 6 的行（2000 槽），同一 (台数, N, 策略, 决策方式) 几个种子取平均
    2. 每格画单步最优和 λ = 0.3/1/2 四条线，PS 感知贪心画虚线当参照
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
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for i, k in enumerate((3, 7, 11)):
        for pol in pols + ["psgreedy"]:
            ns = sorted({n for (p, kk, n, m) in cell if p == pol and kk == k and m == "sequential"})
            if not ns:
                continue
            ys = []
            for n in ns:
                ys.append(mean([float(r["mean_delay"]) for r in cell[(pol, k, n, "sequential")]]))
            label, color, marker = style_of(pol)
            if pol == "greedy":
                label, color = "单步最优（λ = 0）", "#2a78d6"
            elif pol == "lacs":
                label = "LACS λ = 0.3（论文）"
            elif pol == "psgreedy":
                label = "PS 感知贪心（改公式，参照）"
            axes[i].plot(ns, ys, color=color, marker=marker, label=label, linewidth=1.8,
                         linestyle="--" if pol == "psgreedy" else "-")
        tidy(axes[i], f"{k} 台服务器", "终端数 N", "平均延迟（槽，顺序决策）" if i == 0 else "")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("实验 6 · LACS 往后看在闭环里有没有用", x=0.01, ha="left", fontsize=12, color=INK)
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
            numbers.append((f"{style_of(pol)[0]} 相对单步最优的平均延迟变化",
                            f"平均 {mean(diffs):+.2f}%，范围 {min(diffs):+.2f}% ~ {max(diffs):+.2f}%（仿真实测，顺序决策 {len(diffs)} 个点）",
                            dirs))
    ns11 = sorted({n for (p, k, n, m) in cell if k == 11 and m == "batch"})
    if ns11:
        n = ns11[-1]
        g = mean([float(r["mean_delay"]) for r in cell.get(("greedy", 11, n, "batch"), [])])
        l = mean([float(r["mean_delay"]) for r in cell.get(("lacs", 11, n, "batch"), [])])
        numbers.append((f"批量决策下 LACS 能否减轻羊群效应（11 台、N = {n}）",
                        f"单步最优 {g:.2f} 槽，LACS λ = 0.3 {l:.2f} 槽（仿真实测）", dirs))
    return fig


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

    # 3. 逐张画；缺数据的跳过
    os.makedirs(args.out, exist_ok=True)
    numbers = []
    jobs = [
        ("fig1_stability.png", lambda: fig1_stability(e1, bench, numbers) if e1 else None, "实验 1 扫描（eval.s5_sweep）"),
        ("fig2_latency.png", lambda: fig2_latency(bench, numbers), "延迟基准（eval.bench_latency）"),
        ("fig3_herding.png", lambda: fig3_herding(virtual, numbers), "实验 3（eval.virtual_loop --plan e3）"),
        ("fig4_generalization.png", lambda: fig4_generalization(virtual, open_rows, numbers), "开环表 / e4b"),
        ("fig5_open_vs_closed.png", lambda: fig5_open_vs_closed(virtual, open_rows, numbers), "e5 + 开环表"),
        ("fig6_ablation.png", lambda: fig6_ablation(virtual, open_rows, numbers), "开环表 / e4a"),
        ("fig7_lacs.png", lambda: fig7_lacs(virtual, numbers), "实验 6（eval.virtual_loop --plan e6）"),
    ]
    for fname, make, need in jobs:
        fig = make()
        if fig is None:
            print(f"跳过 {fname}：还没有 {need} 的数据")
            continue
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(os.path.join(args.out, fname), dpi=150)
        plt.close(fig)
        print("画好了：", os.path.join(args.out, fname))

    # 4. 两板对照表（跑过两板复测才有）
    if twoboard_table(e1, twoboard, bench_all, os.path.join(args.out, "twoboard.md"), numbers):
        print("写好了：", os.path.join(args.out, "twoboard.md"))

    # 5. 简历数字和出处
    with open(os.path.join(args.out, "numbers.md"), "w", encoding="utf-8") as f:
        f.write("# 阶段 5 关键数字\n\n由 `python -m eval.figures` 生成。每行最后一列是数据所在的运行目录（runs/ 下）。\n\n")
        f.write("| 数字 | 值 | 运行目录 |\n|---|---|---|\n")
        for name, value, where in numbers:
            f.write(f"| {name} | {value} | {where} |\n")
    print("写好了：", os.path.join(args.out, "numbers.md"))


if __name__ == "__main__":
    main()
