"""Serial, resumable launcher for the agreed nine-model Jetson evaluation.

Uses the existing evaluation commands unchanged. Run from this checkout;
progress and each command's console output stay next to this launcher.
"""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from deploy import testbed
from deploy.client import SchedulerClient

KEYS = ["main_0.5b", "soft_1.5b", "hard_1.5b", "ps_0.5b",
        "ps_soft_1.5b", "ps_hard_1.5b", "paper_sft_1.5b",
        "paper_grpo_1.5b", "paper_score_grpo_1.5b"]
BENCH_KEYS = {"hard_1.5b", "ps_0.5b", "ps_soft_1.5b", "ps_hard_1.5b",
              "paper_grpo_1.5b", "paper_score_grpo_1.5b"}
HASHES = dict(zip(KEYS, [
    "3b4c3232d725792b63be2ea4607a21ff6e97c51e5fbfffd05e300de380ba3783",
    "79680e01422f26be21f8686d2018387330fca532d2ca736290a9f4a292645810",
    "c1103a5521a2afb5ed4c54d8d28e140eb07b09161b41ab8ec4c4523811c22dd3",
    "7eebe06bd6470d7dcd27da1713f5fc62def3fe5ed81f5e253b7c459c86a29bcc",
    "ce3719c126785b7597dbc3ff5a4508fa75f96b532aa7bb490a834eb3a28dcd39",
    "96829485ba6013d6479ec8669f0aba7cc813b80e06ac013444eaac9301fcfeb9",
    "40c914087efaaa4becd29bb17d18ee2dd70366fd84f47d57072ded3c2a1280c8",
    "9cb76a1c3373d0f409060d257f6e0065b10fe21e4096dd530ef3a6d4500af10e",
    "57f52fb0d89bcb7985ff24aa818ed6fb355accf95d9b408189e0c25818b4aff6",
]))


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save():
    state["updated_utc"] = utc()
    tmp = HERE / "suite_status.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(HERE / "suite_status.json")


def run_command(task, args):
    state["current_task"] = task
    previous_attempt = state["tasks"].get(task)
    record = {"started_utc": utc(), "command": args, "status": "running"}
    if previous_attempt is not None:
        history = list(previous_attempt.get("previous_attempts", []))
        old = {k: v for k, v in previous_attempt.items() if k != "previous_attempts"}
        if old.get("returncode"):
            old["status"] = "failed"
        record["previous_attempts"] = history + [old]
    state["tasks"][task] = record
    save()
    print(utc(), task, "started", flush=True)
    with (HERE / (task + ".log")).open("a") as log:
        result = subprocess.run([sys.executable, "-u", "-m"] + args,
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    record["returncode"] = result.returncode
    if result.returncode:
        raise RuntimeError(f"{task} failed with {result.returncode}; inspect its log")
    return record


if __name__ == "__main__":
    os.chdir(ROOT)
    lock = open("/tmp/comllm-jetson-192.168.50.20-8100.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()) + "\n")
    lock.flush()
    previous = HERE / "suite_status.json"
    state = json.loads(previous.read_text()) if previous.exists() else {
        "started_utc": utc(), "tasks": {}, "root": str(ROOT)}
    state.update(status="running", pid=os.getpid())
    state.pop("error", None)
    preflight = json.loads((HERE / "preflight.json").read_text())
    for split, expected in preflight["test_sets"].items():
        assert digest(ROOT / "data/s3" / (split + ".jsonl")) == expected["sha256"]
    for relative, expected in preflight["source_files"].items():
        assert digest(ROOT / relative) == expected, relative
    save()
    try:
        for key in KEYS:
            task = "open_" + key
            if state["tasks"].get(task, {}).get("status") != "complete":
                record = run_command(task, ["eval.jetson_open_loop", "--models", key,
                    "--resume", "runs/jetson_open/" + HERE.name.split("_s6_")[0] + "_" + key])
                out = ROOT / "runs/jetson_open" / (HERE.name.split("_s6_")[0] + "_" + key)
                cfg = json.loads((out / "config.json").read_text())
                assert cfg["gguf_sha256"] == HASHES[key]
                rows = list(csv.DictReader((out / "preds.csv").open()))
                expected_keys = {(s, i) for s, d in preflight["test_sets"].items()
                                 for i in range(d["n"])}
                got = [(r["split"], int(r["idx"])) for r in rows]
                assert len(got) == len(set(got)) == 5663 and set(got) == expected_keys
                record.update(status="complete", finished_utc=utc(), output=str(out.relative_to(ROOT)),
                              prediction_sha256=digest(out / "preds.csv"),
                              n_rows=len(rows), non_llm=sum(r["source"] != "llm" for r in rows))
                record["scheduler_info_after"] = SchedulerClient().info()
                save()
                print(utc(), task, "verified", record["n_rows"], "rows", flush=True)
            task = "bench_" + key
            if key in BENCH_KEYS and state["tasks"].get(task, {}).get("status") != "complete":
                before = set((ROOT / "runs").glob("*_bench_*/summary.json"))
                fmt = testbed.model_entry(key)["fmt"]
                record = run_command(task, ["eval.bench_latency", "--fmt", fmt,
                    "--cache", "off" if fmt == "paper" else "on", "--model-key", key,
                    "--n", "500", "--warmup", "10", "--servers", "3,7,11"])
                outputs = set((ROOT / "runs").glob("*_bench_*/summary.json")) - before
                assert len(outputs) == 1
                output = outputs.pop()
                summary = json.loads(output.read_text())
                assert summary["n"] == 500 and summary["sources"] == {"llm": 500}
                assert summary["scheduler_info"]["gguf_sha256"] == HASHES[key]
                record.update(status="complete", finished_utc=utc(), output=str(output.parent.relative_to(ROOT)),
                              summary_sha256=digest(output))
                save()
                print(utc(), task, "verified 500 requests", flush=True)
        state.update(status="complete", finished_utc=utc(), current_task=None)
        save()
        print(utc(), "All nine open-loop runs and six missing benchmarks verified", flush=True)
    except BaseException as exc:
        current = state.get("current_task")
        if current in state["tasks"]:
            state["tasks"][current].update(status="failed", failed_utc=utc())
        state.update(status="failed", error=repr(exc))
        save()
        raise
