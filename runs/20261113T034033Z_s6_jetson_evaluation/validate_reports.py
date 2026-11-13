"""Cross-check the two report paths and record final offline acceptance."""
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT / "evidence/G6/jetson_open"


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as inp:
        return list(csv.DictReader(inp))


state = read_json(HERE / "suite_status.json")
assert state["status"] == "complete" and len(state["tasks"]) == 16
assert all(task["status"] == "complete" for task in state["tasks"].values())
assert read_json(HERE / "postprocessing.json")["status"] == "complete"
metrics = read_csv(REPORT / "platform_comparison.csv")
lookup = {(r["model_key"], r["split"]): r for r in metrics}
assert len(metrics) == len(lookup) == 27
names = {
    read_json(ROOT / task["output"] / "config.json")["name"]: name[5:]
    for name, task in state["tasks"].items() if name.startswith("open_")
}
# The shared scorer also adds five formula/random baselines. Check the nine
# measured models here; its CSV rounds floats to four decimal places.
scored = [r for r in read_csv(REPORT / "open_loop.csv")
          if r["servers"] == "全部" and r["policy"] in names]
assert len(scored) == 27
for row in scored:
    ref = lookup[names[row["policy"]], row["split"]]
    assert int(row["n"]) == int(ref["n"])
    for field in ("acc_pct", "pr_pct", "invalid_pct", "local_pct", "local_recall_pct",
                  "local_precision_pct", "offload_acc_pct", "al"):
        a, b = float(row[field]), float(ref["jetson_" + field])
        assert (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 0.000050001, (
            row["policy"], row["split"], field, a, b)

agreement_rows = 0
for line in (REPORT / "prediction_agreement.md").read_text().splitlines():
    if not line.startswith("| "):
        continue
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    if len(cells) != 5 or cells[1] not in ("test_seen", "test_unseen", "friend_clean"):
        continue
    name, split, n, agreement, non_llm = cells
    ref = lookup[names[name], split]
    assert int(n) == int(ref["n"]) and int(non_llm) == int(ref["non_llm"])
    assert abs(float(agreement) - round(float(ref["agreement_pct"]), 1)) < 1e-9
    agreement_rows += 1
assert agreement_rows == 27

for file, count in (("unit_tests.log", 86), ("readme_checks.log", 3)):
    log = (HERE / file).read_text()
    assert re.search(rf"^Ran {count} tests ", log, re.M)
    assert re.search(r"^OK$", log, re.M)
board = read_json(HERE / "board_final_state.json")
assert board["remote_source_hashes_match"] and all(board["ports_closed"].values())
assert all(float(r["jetson_invalid_pct"]) == 0 for r in metrics)
benches = read_csv(REPORT / "latency_coverage.csv")
assert len(benches) == 9 and sum(r["reused"] == "True" for r in benches) == 2
qa = {
    "status": "complete", "verified_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Nine small-model Jetson fixed-test open-loop runs and latency coverage",
    "open_models": 9, "saved_predictions": sum(int(r["n"]) for r in metrics),
    "states_per_model": 5663,
    "split_counts_per_model": {"test_seen": 3010, "test_unseen": 2408, "friend_clean": 245},
    "non_llm_predictions": sum(int(r["non_llm"]) for r in metrics), "invalid_predictions": 0,
    "all_metrics_cross_checked_rows": 27, "legacy_agreement_cross_checked_rows": agreement_rows,
    "new_500_request_benchmarks": 7, "reused_500_request_benchmarks": 2,
    "required_missing_benchmarks_completed": 6, "full_unit_tests_passed": 86,
    "readme_source_tests_passed_after_update": 3, "transport_interruptions": 1,
    "original_prediction_prefix_preserved": True,
    "remote_code_hashes_match": board["remote_source_hashes_match"],
    "owned_service_ports_closed": board["ports_closed"],
    "served_requests_including_benchmark_warmups_and_unreceived_attempt": board["served_requests"],
    "measurement_commit": "cafd7e4",
}
assert qa["saved_predictions"] == 50967 and qa["non_llm_predictions"] == 0
(HERE / "final_validation.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(qa, ensure_ascii=False))
