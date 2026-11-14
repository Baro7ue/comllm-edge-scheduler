"""Validate saved suite outputs and compare fixed-state PyTorch/GGUF results."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from eval import open_loop as O
from eval.jetson_open_loop import COLAB_NAME
from run_suite import KEYS, HASHES

REFERENCES = dict(zip(KEYS, [
    "runs/s3/main_0.5b_20261104T131423Z", "runs/s3/soft_1.5b_20261104T132257Z",
    "runs/s3/hard_1.5b_20261104T134648Z", "runs/s3/ps_0.5b_20261104T140119Z",
    "runs/s3/ps_soft_1.5b_20261108T044612Z", "runs/s3/ps_hard_1.5b_20261108T064436Z",
    "runs/s3/paper_20261105T040018Z/sft", "runs/s3/paper_20261105T040018Z/grpo",
    "runs/s3/paper_20261105T040018Z/score-grpo",
]))
REUSED_BENCHMARKS = {
    "main_0.5b": "runs/20261106T130530Z_bench_comllm-0.5b-soft-T0.3-q8_0_compact_cacheon",
    "soft_1.5b": "runs/20261106T130817Z_bench_comllm-1.5b-soft-T0.3-q8_0_compact_cacheon",
}


def read_json(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def predictions(p):
    with Path(p).open(newline="") as inp:
        rows = list(csv.DictReader(inp))
    keys = [(r["split"], int(r["idx"])) for r in rows]
    assert len(keys) == len(set(keys)), f"duplicate keys in {p}"
    return {(r["split"], int(r["idx"])): int(r["pred"]) for r in rows}, rows


def write_csv(path, rows):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(v):
    return "–" if isinstance(v, float) and not math.isfinite(v) else f"{v:.2f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="Inspect complete models without declaring suite completion")
    args = ap.parse_args()
    state = read_json(HERE / "suite_status.json")
    if not args.partial:
        assert state["status"] == "complete"
    preflight = read_json(HERE / "preflight.json")
    interruption = read_json(HERE / "interruption_20261113T0749.json")
    interrupted_csv = ROOT / "runs/jetson_open/20261113T034033Z_paper_score_grpo_1.5b/preds.csv"
    prefix = interrupted_csv.read_bytes()[:interruption["prefix_bytes"]]
    assert hashlib.sha256(prefix).hexdigest() == interruption["prefix_sha256"]
    for relative, expected_hash in preflight["source_files"].items():
        assert sha(ROOT / relative) == expected_hash, relative
    data = {s: O.load_split(str(ROOT / "data/s3"), s) for s in O.SPLITS}
    for s, rows in data.items():
        assert sha(ROOT / "data/s3" / (s + ".jsonl")) == preflight["test_sets"][s]["sha256"]
        for i, row in enumerate(rows):
            row["_idx"] = i
    expected = {(s, i) for s, rows in data.items() for i in range(len(rows))}
    metrics, provenance, benches = [], [], []
    for key in KEYS:
        task = state["tasks"].get("open_" + key, {})
        if task.get("status") != "complete":
            assert args.partial
            continue
        out = ROOT / task["output"]
        pred_path = out / "preds.csv"
        cfg = read_json(out / "config.json")
        preds, records = predictions(pred_path)
        assert len(records) == len(preds) == 5663 and set(preds) == expected
        assert sha(pred_path) == task["prediction_sha256"]
        assert cfg["gguf_sha256"] == HASHES[key]
        for rec in records:
            assert math.isfinite(float(rec["ms"])) and float(rec["ms"]) >= 0
        actual_backend = task["scheduler_info_after"]
        assert actual_backend["model_key"] == key and actual_backend["gguf_sha256"] == HASHES[key]
        assert actual_backend["llama"]["n_ctx"] == 4096
        assert actual_backend["llama"]["build_info"] == "b11719-5aeac37a3"
        reference_path = ROOT / REFERENCES[key] / "preds.csv"
        reference_config = ROOT / REFERENCES[key] / "config.json"
        assert read_json(reference_config)["name"] == COLAB_NAME[key]
        reference, _ = predictions(reference_path)
        assert expected <= set(reference)
        provenance.append({"model_key": key, "jetson_run": task["output"],
                           "jetson_predictions_sha256": sha(pred_path), "gguf_sha256": HASHES[key],
                           "pytorch_run": REFERENCES[key], "pytorch_predictions_sha256": sha(reference_path),
                           "pytorch_config_sha256": sha(reference_config)})
        for split, rows in data.items():
            m, ref = O.score(rows, preds, split), O.score(rows, reference, split)
            row = {"model_key": key, "split": split, "n": len(rows),
                   "agreement_pct": 100 * sum(preds[(split, i)] == reference[(split, i)]
                                               for i in range(len(rows))) / len(rows),
                   "non_llm": sum(r["source"] != "llm" for r in records if r["split"] == split)}
            for field in ("acc_pct", "pr_pct", "invalid_pct", "local_pct", "local_recall_pct",
                          "local_precision_pct", "offload_acc_pct", "al"):
                row["jetson_" + field], row["pytorch_" + field] = m[field], ref[field]
            metrics.append(row)
    if not args.partial:
        assert len(metrics) == 27 and len(provenance) == 9
    report = ROOT / "evidence/G6/jetson_open"
    if args.partial:
        report = HERE / "partial_analysis"
    report.mkdir(parents=True, exist_ok=True)
    if metrics:
        write_csv(report / "platform_comparison.csv", metrics)
    for key in KEYS:
        task = state["tasks"].get("bench_" + key, {})
        folder = task.get("output", REUSED_BENCHMARKS.get(key))
        if not folder:
            assert args.partial
            continue
        summary = read_json(ROOT / folder / "summary.json")
        cfg = read_json(ROOT / folder / "config.json")
        with (ROOT / folder / "requests.csv").open(newline="") as inp:
            records = list(csv.DictReader(inp))
        assert len(records) == summary["n"] == cfg["args"]["n"] == 500
        assert [int(r["i"]) for r in records] == list(range(500))
        assert all(r["source"] == "llm" and math.isfinite(float(r["e2e_ms"])) and
                   float(r["e2e_ms"]) > 0 for r in records)
        assert cfg["args"]["warmup"] == 10 and cfg["args"]["servers"] == [3, 7, 11]
        assert [int(r["servers"]) for r in records] == [
            [3, 7, 11][(i + 10) % 3] for i in range(500)]
        assert summary["sources"] == {"llm": 500}
        assert summary["scheduler_info"]["gguf_sha256"] == HASHES[key]
        expected_fmt = "paper" if key.startswith("paper_") else "compact"
        assert cfg["args"]["fmt"] == summary["fmt"] == expected_fmt
        assert summary["cache"] == ("off" if expected_fmt == "paper" else "on")
        assert cfg["args"]["model_key"] == key
        if expected_fmt == "compact":
            assert summary["decode"] == cfg["args"]["decode"] == "probs"
        benches.append({"model_key": key, "source_run": folder, "reused": key in REUSED_BENCHMARKS,
                        "p50_ms": summary["e2e_p50"], "p99_ms": summary["e2e_p99"],
                        "mean_ms": summary["e2e_mean"], "fmt": summary["fmt"], "cache": summary["cache"],
                        "prompt_tokens_mean": summary["prompt_tokens_mean"],
                        "requests_sha256": sha(ROOT / folder / "requests.csv")})
    write_csv(report / "latency_coverage.csv", benches)
    text = ["# Jetson 固定测试集开环与延迟基准", "",
            "每套状态独立作答。准确率是与数据集动作标签的一致率；三套分别报告。PR = 最优 AL / 策略 AL × 100%。",
            "本地召回、本地精确以论文动作标签为参照；PS 学生的训练目标不同。",
            "两平台逐条一致率衡量部署前后答案变化，不代表任务完成准确率。", "",
            "score-GRPO 曾发生一次 HTTP 传输等待超时：板端已完成该次推理，客户端未落盘回复。保留已有 4472 条，再续跑缺失的 1191 条；没有修改请求设置或重启服务。那次未收到回复的尝试不计入固定测试输出，原始中断和断点校验记录见运行目录的 interruption_20261113T0749.json。最终每状态仍恰好一条。", "",
            "本轮一致率包含全部状态，未剔除成本打平的状态。开环均开启前缀缓存，每请求给 10 秒作答；实时调度的默认 100 毫秒不适用于本表。", "",
            "| 模型 | 测试集 | 样本数 | Jetson 准确率 % | Jetson PR % | PyTorch PR % | 逐条一致 % | 非 LLM | 本地召回 % | 本地精确 % |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in metrics:
        values = [r["jetson_acc_pct"], r["jetson_pr_pct"], r["pytorch_pr_pct"], r["agreement_pct"]]
        text.append(f"| {r['model_key']} | {r['split']} | {r['n']} | " +
                    " | ".join(fmt(v) for v in values) + f" | {r['non_llm']} | " +
                    fmt(r["jetson_local_recall_pct"]) + " | " + fmt(r["jetson_local_precision_pct"]) + " |")
    text += ["", "延迟基准固定 500 次请求、10 次热身、3/7/11 台轮流。学生为 compact/cache on，原作者为 paper/cache off。",
             "不同格式代表各自的完整部署配置。复用两个学生旧基准，新增缺失六组；另补测 SFT 使三个原作者模型使用同一版贪心参数。旧 SFT 结果仍保留。", "",
             "| 模型 | p50 ms | p99 ms | 复用旧结果 | 运行目录 |", "|---|---:|---:|---|---|"]
    for r in benches:
        text.append(f"| {r['model_key']} | {r['p50_ms']:.2f} | {r['p99_ms']:.2f} | {r['reused']} | {r['source_run']} |")
    text += ["", "原始预测、配置和校验值见各运行目录；来源映射见 provenance.json。", ""]
    (report / "comparison.md").write_text("\n".join(text))
    (report / "provenance.json").write_text(json.dumps({"complete": not args.partial,
        "suite": str(HERE.relative_to(ROOT)), "models": provenance,
        "benchmarks": benches, "test_sets": preflight["test_sets"],
        "transport_interruption": {"record": str((HERE / "interruption_20261113T0749.json").relative_to(ROOT)),
            "saved_rows": interruption["saved_rows"], "resumed_rows": interruption["remaining_rows"],
            "completed_prefix_sha256": interruption["prefix_sha256"], "prefix_preserved": True}}, ensure_ascii=False, indent=2) + "\n")
    print(f"Validated {len(provenance)}/9 models, {len(metrics)} split rows, {len(benches)}/9 benchmarks; {report}")
