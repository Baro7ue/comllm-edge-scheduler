"""After the serial suite, measure SFT with the fixed greedy sampler and score."""
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from run_suite import HASHES, digest


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(state):
    temp = HERE / "suite_status.tmp"
    state["updated_utc"] = utc()
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    temp.replace(HERE / "suite_status.json")


while True:
    state = json.loads((HERE / "suite_status.json").read_text())
    if state["status"] == "failed":
        raise RuntimeError(state.get("error", "Suite stopped"))
    if state["status"] == "complete":
        break
    time.sleep(20)

lock = open("/tmp/comllm-jetson-192.168.50.20-8100.lock", "a+")
fcntl.flock(lock, fcntl.LOCK_EX)
task = "bench_paper_sft_1.5b"
try:
    if state["tasks"].get(task, {}).get("status") != "complete":
        state.update(status="running", current_task=task, pid=__import__("os").getpid())
        record = {"status": "running", "started_utc": utc(),
                  "reason": "Historical SFT benchmark predates explicit greedy-sampler fix 4567c59; same sampler as the two new GRPO benchmarks is needed."}
        state["tasks"][task] = record
        write(state)
        before = set((ROOT / "runs").glob("*_bench_*/summary.json"))
        cmd = [sys.executable, "-u", "-m", "eval.bench_latency", "--fmt", "paper", "--cache", "off",
               "--model-key", "paper_sft_1.5b", "--n", "500", "--warmup", "10", "--servers", "3,7,11"]
        record["command"] = cmd
        print(utc(), "Starting supplemental SFT latency control", flush=True)
        with (HERE / (task + ".log")).open("a") as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        outputs = set((ROOT / "runs").glob("*_bench_*/summary.json")) - before
        assert len(outputs) == 1
        output = outputs.pop()
        summary = json.loads(output.read_text())
        assert summary["n"] == 500 and summary["sources"] == {"llm": 500}
        assert summary["scheduler_info"]["gguf_sha256"] == HASHES["paper_sft_1.5b"]
        record.update(status="complete", finished_utc=utc(), output=str(output.parent.relative_to(ROOT)),
                      summary_sha256=digest(output))
        state.update(status="complete", finished_utc=utc(), current_task=None)
        write(state)
    subprocess.run([sys.executable, str(HERE / "summarize_results.py")], cwd=ROOT, check=True)
    dirs = [str(ROOT / state["tasks"]["open_" + key]["output"]) for key in HASHES]
    subprocess.run([sys.executable, "-m", "eval.open_loop", *dirs, "--out", "evidence/G6/jetson_open",
                    "--title", "Jetson 全量开环（九模型，各 5663 状态，GGUF Q8_0）"],
                   cwd=ROOT, stdout=(HERE / "scoring_console.log").open("w"), stderr=subprocess.STDOUT, check=True)
    with (ROOT / "evidence/G6/jetson_open/prediction_agreement.md").open("w") as log:
        subprocess.run([sys.executable, "-m", "eval.jetson_open_loop", "--compare", *dirs,
                        "--colab", "runs/s3,runs/s4,runs/s3/paper_20261105T040018Z"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (ROOT / "evidence/G2/latency_table.md").open("w") as log:
        subprocess.run([sys.executable, "-m", "eval.latency_table"], cwd=ROOT, stdout=log, check=True)
    (HERE / "postprocessing.json").write_text(json.dumps({"status": "complete", "finished_utc": utc(),
        "report": "evidence/G6/jetson_open/comparison.md", "open_loop_models": 9,
        "new_benchmarks": 7, "reused_benchmarks": 2}, indent=2) + "\n")
    print(utc(), "Full suite and scoring complete", flush=True)
except BaseException as exc:
    state.update(status="failed", error=repr(exc))
    write(state)
    raise
