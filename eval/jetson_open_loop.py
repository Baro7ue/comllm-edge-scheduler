"""Jetson 开环评测：开环测试集的每条状态都发给 Jetson 上的调度服务，记下 GGUF 模型（Q8_0）的答案。

Colab 那边的开环（阶段 3、4）是 PyTorch 原精度模型答的；这里是部署后的同一个模型在板子上答的。
两边写成同样格式的 preds.csv，再交给 eval.open_loop 用同一把尺子（论文的单步最优）打分，
两边的差别就是部署带来的变化（量化到 Q8_0、换成 llama.cpp）。

在仓库根目录运行（Mac 上，Jetson 上的调度服务已经开着；换模型由脚本自己做）：
  python -m eval.jetson_open_loop --models main_0.5b,soft_1.5b --dry-run     # 只看要答多少条、大概多久
  python -m eval.jetson_open_loop --models main_0.5b,soft_1.5b               # 真跑
  python -m eval.jetson_open_loop --models main_0.5b --resume runs/jetson_open/<目录>   # 断了接着跑
  python -m eval.open_loop runs/jetson_open/* --out evidence/G6/jetson_open --title "Jetson 开环评测（GGUF Q8_0）"
  python -m eval.jetson_open_loop --compare runs/jetson_open/*               # 和 Colab 的答案逐条比

结果：runs/jetson_open/<UTC>_<模型键>/config.json 和 preds.csv（列和 Colab 的一样：split, idx, pred），
另外多记 source、reason、ms 三列。答案不是 LLM 自己给的（认不出、出错），pred 记 −1，按非法算，
不让公式兜底的答案混进来。超时给 10 秒，也是这个原因。
"""
import argparse
import csv
import json
import os
from datetime import datetime, timezone

from deploy import testbed
from deploy.client import SchedulerClient, parse_host_port
from deploy.wire import state_from_dict
from eval.closed_loop import git_version
from eval.open_loop import SPLITS, find_run_dirs, load_preds, load_split, run_name

OUT_ROOT = os.path.join("runs", "jetson_open")
# 模型键 → Colab 开环表里同一个模型的名字（eval.open_loop 用 config.json 的 name 当表里的名字）
COLAB_NAME = {"main_0.5b": "0.5b_soft_T0.3", "soft_1.5b": "1.5b_soft_T0.3", "hard_1.5b": "1.5b_hard",
              "ps_0.5b": "0.5b_soft_T0.3_ps", "ps_soft_1.5b": "1.5b_soft_T0.3_ps", "ps_hard_1.5b": "1.5b_hard_ps",
              "paper_sft_1.5b": "论文原版 sft", "main_0.5b_q4": "0.5b_soft_T0.3"}
EST_MS = {"compact_0.5b": 45.0, "compact_1.5b": 80.0, "paper": 950.0}   # 每条约多少毫秒（估算，按 11/6–11/7 实测）
FIELDS = ["split", "idx", "pred", "source", "reason", "ms"]


def jetson_name(key):
    """Jetson 那一行在开环表里的名字：Colab 的名字后面加"（Jetson Q8_0）"。"""
    base = COLAB_NAME.get(key, key)
    quant = "Q4_K_M" if key.endswith("_q4") else "Q8_0"
    return f"{base}（Jetson {quant}）"


def est_ms(key, fmt):
    if fmt == "paper":
        return EST_MS["paper"]
    if "1.5b" in key:
        return EST_MS["compact_1.5b"]
    return EST_MS["compact_0.5b"]


def answer_one(client, row, fmt, req_id):
    """一条状态发给调度服务，返回 (pred, source, reason, 往返毫秒)。不是 LLM 给的答案，pred 记 −1。"""
    resp, ms = client.decide(state_from_dict(row["state"]), fmt, True, 10000.0, req_id=req_id)
    if resp.get("source") == "llm":
        return int(resp["action"]), "llm", "", ms
    return -1, resp.get("source", ""), resp.get("reason", ""), ms


def run_model(client, key, data, out_dir, cfg):
    """一个模型答完所有测试集。

    1. 换模型（调度服务重启 llama-server，十几秒到一分钟）
    2. 读 preds.csv 里已经答过的 (split, idx)，跳过它们（断了可以接着跑）
    3. 逐条发给调度服务，答一条写一行，每 200 条打印一次进度
    4. 写 config.json：名字、模型键、GGUF 的 sha256、git 版本、条数
    """
    entry = testbed.model_entry(key, cfg)
    fmt = entry["fmt"]
    info = client.switch_model(key)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "preds.csv")
    done = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["split"], int(r["idx"])))
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()
    n_total = 0
    for split in data:
        n_total += len(data[split])
    n = len(done)
    n_bad = 0
    t_ms = 0.0
    for split in data:
        for r in data[split]:
            if (split, r["_idx"]) in done:
                continue
            pred, source, reason, ms = answer_one(client, r, fmt, n)
            w.writerow({"split": split, "idx": r["_idx"], "pred": pred, "source": source, "reason": reason,
                        "ms": round(ms, 2)})
            n += 1
            t_ms += ms
            if pred < 0:
                n_bad += 1
            if n % 200 == 0:
                f.flush()
                print(f"  {key}：{n}/{n_total} 条，非 LLM 答案 {n_bad} 条，平均 {t_ms / max(1, n - len(done)):.0f} ms",
                      flush=True)
    f.close()
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fc:
        json.dump({"name": jetson_name(key), "model_key": key, "fmt": fmt, "platform": "jetson",
                   "gguf": entry["gguf"], "gguf_sha256": info.get("gguf_sha256"), "git": git_version(),
                   "splits": list(data), "n_rows": n_total,
                   "finished_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")},
                  fc, ensure_ascii=False, indent=2)
    print(f"{key} 答完：{n_total} 条，这次新答 {n - len(done)} 条，其中非 LLM 答案 {n_bad} 条 → {out_dir}", flush=True)


def compare(jetson_dirs, colab_roots):
    """逐个 Jetson 目录，找 Colab 上同名模型的 preds.csv，按测试集算两边答案一样的比例。

    1. Colab 那边的运行目录按 config.json 的 name 建索引（runs/s3、runs/s4 下面，自动往下找一层）
    2. 每个 Jetson 目录：取它 config.json 的 model_key → Colab 名字 → 找到对应的 preds.csv
    3. 两边都答了的 (split, idx) 里数一样的条数；Jetson 非 LLM 答案单独数
    """
    colab = {}
    for d in find_run_dirs(colab_roots):
        name, _ = run_name(d)
        colab[name] = os.path.join(d, "preds.csv")
    lines = ["| 模型 | 测试集 | 条数 | 两边答案一样 % | Jetson 非 LLM 答案 |", "|---|---|---|---|---|"]
    for d in jetson_dirs:
        cfg_path = os.path.join(d, "config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path, encoding="utf-8") as fc:
            cfg = json.load(fc)
        cname = COLAB_NAME.get(cfg.get("model_key", ""), "")
        if cname not in colab:
            lines.append(f"| {cfg.get('name')} | – | – | Colab 那边没找到 {cname} | – |")
            continue
        ref = load_preds(colab[cname])
        got = load_preds(os.path.join(d, "preds.csv"))
        for split in SPLITS:
            n = 0
            same = 0
            bad = 0
            for (s, i), a in got.items():
                if s != split or (s, i) not in ref:
                    continue
                n += 1
                if a == ref[(s, i)]:
                    same += 1
                if a < 0:
                    bad += 1
            if n:
                lines.append(f"| {cfg.get('name')} | {split} | {n} | {100.0 * same / n:.1f} | {bad} |")
    print("\n".join(lines))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="模型键，逗号分隔（deploy/testbed.json 的 models）")
    ap.add_argument("--splits", default=",".join(SPLITS), help="答哪几套测试集，默认三套都答")
    ap.add_argument("--data", default=os.path.join("data", "s3"))
    ap.add_argument("--resume", default=None, help="接着跑的目录（只能配一个模型）")
    ap.add_argument("--scheduler", default=None, help="调度服务地址 IP:端口；不给就用 deploy/testbed.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--compare", nargs="*", default=None, help="Jetson 运行目录：和 Colab 的答案逐条比，不连 Jetson")
    ap.add_argument("--colab", default="runs/s3,runs/s4", help="--compare 时 Colab 运行目录的上一级，逗号分隔")
    args = ap.parse_args()

    if args.compare is not None:
        compare(args.compare, args.colab.split(","))
        return

    # 1. 读测试集，每行记下行号（和 Colab 的 preds.csv 用同样的行号）
    data = {}
    for split in args.splits.split(","):
        rows = load_split(args.data, split)
        for i, r in enumerate(rows):
            r["_idx"] = i
        if rows:
            data[split] = rows
        else:
            print("没找到测试集：", os.path.join(args.data, split + ".jsonl"))
    if not data:
        return
    keys = args.models.split(",") if args.models else []
    if args.resume and len(keys) != 1:
        raise SystemExit("--resume 只能配一个模型")

    # 2. 估时长
    cfg = testbed.load()
    n_rows = 0
    for split in data:
        n_rows += len(data[split])
    total_s = 0.0
    for key in keys:
        fmt = testbed.model_entry(key, cfg)["fmt"]
        sec = n_rows * est_ms(key, fmt) / 1000.0
        total_s += sec
        print(f"  {key:16s} 格式 {fmt:8s} {n_rows} 条，约 {sec / 60:.0f} 分钟（估算）")
    print(f"共 {len(keys)} 个模型，约 {total_s / 3600:.1f} 小时（估算）")
    if args.dry_run or not keys:
        return

    # 3. 逐个模型答
    host, port = parse_host_port(args.scheduler)
    client = SchedulerClient(host, port)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for key in keys:
        out_dir = args.resume or os.path.join(OUT_ROOT, f"{stamp}_{key}")
        print(f"\n换模型 → {key}", flush=True)
        run_model(client, key, data, out_dir, cfg)


if __name__ == "__main__":
    main()
