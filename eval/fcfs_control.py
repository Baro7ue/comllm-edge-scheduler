"""FCFS control: does the PS-aware formula still beat the paper's formula when servers are first-come-first-served?

The PS-aware formula was derived from the paper's processor-sharing server model (eq. 4), and the simulator
uses the same model, so its advantage over the paper's formula could come from that shared assumption.
This script repeats the formula-against-formula comparison of stage 5 experiment 6 with the servers switched
to FCFS, where a server works on one task at a time. No LLM and no Jetson; formulas only.

  python -m eval.fcfs_control              # results in runs/<UTC>_s6_fcfs_control/
  python -m eval.fcfs_control --jobs 1     # no worker processes

Points: 3/7/11 servers x N = 20/40/60/80 x seeds 0, 1, 2, 2000 slots, tasks arriving in the first 200 slots
not counted, decisions one at a time. This is the grid of experiment 6, so the PS rows must reproduce it.

Policies:
  greedy      the paper's one-step formula
  psgreedy    the PS-aware formula
  fcfsgreedy  an FCFS-aware formula, as the reference for what a formula matched to FCFS servers reaches

Outputs: results.csv (one row per run), summary.md (the comparison) and config.json.
"""
import argparse
import csv
import json
import os
from datetime import datetime, timezone
from multiprocessing import Pool

from eval.closed_loop import git_version
from sim import config as C
from sim.baselines import make_policy
from sim.env import MecEnv, summarize

SERVERS = [3, 7, 11]
NS = [20, 40, 60, 80]
SEEDS = [0, 1, 2]
SLOTS = 2000
WARMUP = 200
POLICIES = ["greedy", "psgreedy", "fcfsgreedy"]
DISCIPLINES = ["ps", "fcfs"]
FIELDS = ["point", "discipline", "servers", "n", "policy", "seed", "slots", "warmup", "n_tasks", "mean_delay",
          "p95_delay", "drop_rate", "local_frac", "top_server_frac", "jain", "pred_gap"]
E6 = os.path.join("runs", "20261106T144248Z_s5_virtual_e6", "results.csv")
NAMES = {"greedy": "paper formula", "psgreedy": "PS-aware formula", "fcfsgreedy": "FCFS-aware formula"}


def points():
    out = []
    for disc in DISCIPLINES:
        for k in SERVERS:
            for n in NS:
                for pol in POLICIES:
                    for seed in SEEDS:
                        out.append({"point": f"{disc}_{pol}_k{k}_n{n}_sequential_s{seed}_t{SLOTS}", "discipline": disc,
                                    "servers": k, "n": n, "policy": pol, "seed": seed, "slots": SLOTS,
                                    "warmup": WARMUP})
    return out


def run_point(p):
    """One closed-loop run in virtual time; the counting window is the same as in eval.virtual_loop."""
    env = MecEnv(p["n"], p["servers"], seed=p["seed"], mode="sequential", discipline=p["discipline"])
    policy = make_policy(p["policy"], p["seed"])
    for _ in range(p["slots"]):
        env.step(policy)
    counted = [t for t in env.tasks if p["warmup"] <= t.t_arrive < p["slots"] - C.DEADLINE_SLOTS - 1]
    assert all(t.resolved for t in counted), p["point"]
    row = dict(p)
    row.update(summarize(counted, p["servers"]))
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}


def mean(xs):
    return sum(xs) / len(xs)


def compare(rows):
    """Seed means per (discipline, servers, N, policy), then each formula against the paper's formula.

    Returns {discipline: {policy: [(servers, N, delay change %, drop % of the policy, drop % of greedy), ...]}}.
    """
    cell = {}
    for r in rows:
        cell.setdefault((r["discipline"], int(r["servers"]), int(r["n"]), r["policy"]), []).append(r)
    out = {}
    for disc in DISCIPLINES:
        out[disc] = {}
        for pol in POLICIES[1:]:
            lines = []
            for k in SERVERS:
                for n in NS:
                    g, v = cell[(disc, k, n, "greedy")], cell[(disc, k, n, pol)]
                    assert len(g) == len(SEEDS) and len(v) == len(SEEDS)
                    base = mean([float(r["mean_delay"]) for r in g])
                    lines.append((k, n, 100.0 * (mean([float(r["mean_delay"]) for r in v]) - base) / base,
                                  100.0 * mean([float(r["drop_rate"]) for r in v]),
                                  100.0 * mean([float(r["drop_rate"]) for r in g])))
            out[disc][pol] = lines
    return out


def check_against_e6(rows):
    """PS rows of greedy and psgreedy must equal experiment 6 (same simulator, same points). Returns (n compared, n different)."""
    if not os.path.exists(E6):
        return 0, 0
    with open(E6, newline="", encoding="utf-8") as f:
        old = {r["point"]: r for r in csv.DictReader(f)}
    n = bad = 0
    for r in rows:
        if r["discipline"] != "ps" or r["policy"] == "fcfsgreedy":
            continue
        o = old.get(r["point"][3:])
        if o is None:
            continue
        n += 1
        if any(abs(float(o[k]) - float(r[k])) > 1e-9 for k in ("mean_delay", "drop_rate", "n_tasks")):
            bad += 1
    return n, bad


def write_summary(path, rows, cmp, e6):
    disc_name = {"ps": "Processor-sharing servers (the paper's eq. 4, used everywhere else in this project)",
                 "fcfs": "First-come-first-served servers (this control)"}
    lines = ["# FCFS control: formula against formula", "",
             "Generated by `python -m eval.fcfs_control`. Simulation, virtual time, decisions one at a time, "
             f"{len(SERVERS) * len(NS)} points ({'/'.join(map(str, SERVERS))} servers x N = "
             f"{'/'.join(map(str, NS))}), mean of seeds {', '.join(map(str, SEEDS))}, {SLOTS} slots per run.",
             "Mean delay counts completed tasks only, so read it with the drop rate.", ""]
    if e6[0]:
        lines += [f"Check: the {e6[0]} processor-sharing runs of the paper and PS-aware formulas were compared with "
                  f"`{E6}`; {e6[1]} differ.", ""]
    local = {}
    for r in rows:
        local.setdefault((r["discipline"], r["policy"]), []).append(100.0 * float(r["local_frac"]))
    lines += ["## Mean delay against the paper's formula, over the 12 points", "",
              "| Servers are | Formula | Mean change | Range | Points where it is faster | Mean drop rate (formula / paper formula) "
              "| Tasks run locally (formula / paper formula) |",
              "|---|---|---|---|---|---|---|"]
    for disc in DISCIPLINES:
        for pol in POLICIES[1:]:
            ls = cmp[disc][pol]
            ch = [x[2] for x in ls]
            lines.append(f"| {disc.upper()} | {NAMES[pol]} | {mean(ch):+.2f}% | {min(ch):+.2f}% to {max(ch):+.2f}% | "
                         f"{sum(1 for c in ch if c < 0)} of {len(ch)} | {mean([x[3] for x in ls]):.2f}% / "
                         f"{mean([x[4] for x in ls]):.2f}% | {mean(local[(disc, pol)]):.1f}% / "
                         f"{mean(local[(disc, 'greedy')]):.1f}% |")
    key = lambda r: (r["discipline"], r["servers"], r["n"], r["seed"])
    ps = {key(r): r for r in rows if r["policy"] == "psgreedy"}
    same = sum(1 for r in rows if r["policy"] == "fcfsgreedy"
               and all(r[k] == ps[key(r)][k] for k in ("mean_delay", "drop_rate", "local_frac")))
    lines += ["", f"The PS-aware and FCFS-aware formulas give the same mean delay, drop rate and local share in {same} of "
              f"{len(ps)} runs."]
    for disc in DISCIPLINES:
        lines += ["", f"## {disc_name[disc]}", "",
                  "Mean delay change against the paper's formula (%), then drop rate (%) of the formula / of the paper's formula.", "",
                  "| Servers | N | PS-aware formula | FCFS-aware formula |", "|---|---|---|---|"]
        a, b = cmp[disc]["psgreedy"], cmp[disc]["fcfsgreedy"]
        for x, y in zip(a, b):
            lines.append(f"| {x[0]} | {x[1]} | {x[2]:+.2f}, {x[3]:.2f} / {x[4]:.2f} | {y[2]:+.2f}, {y[3]:.2f} / {y[4]:.2f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="worker processes; 1 runs in this process")
    ap.add_argument("--out", default=None, help="output folder; default runs/<UTC>_s6_fcfs_control")
    args = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or os.path.join("runs", f"{stamp}_s6_fcfs_control")
    os.makedirs(out_dir, exist_ok=True)
    todo = points()
    print(f"{len(todo)} runs, {args.jobs} worker(s)", flush=True)
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            rows = pool.map(run_point, todo, chunksize=1)
    else:
        rows = [run_point(p) for p in todo]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    e6 = check_against_e6(rows)
    cmp = compare(rows)
    write_summary(os.path.join(out_dir, "summary.md"), rows, cmp, e6)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"git": git_version(), "started_utc": stamp, "servers": SERVERS, "n": NS, "seeds": SEEDS,
                   "slots": SLOTS, "warmup": WARMUP, "policies": POLICIES, "disciplines": DISCIPLINES,
                   "mode": "sequential", "e6_check": {"compared": e6[0], "different": e6[1]}}, f, indent=2)
    print(f"experiment 6 check: {e6[0]} runs compared, {e6[1]} differ")
    for disc in DISCIPLINES:
        for pol in POLICIES[1:]:
            ch = [x[2] for x in cmp[disc][pol]]
            print(f"{disc:4s} {pol:10s} vs greedy: mean {mean(ch):+.2f}%  range {min(ch):+.2f}% .. {max(ch):+.2f}%")
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
