"""Pinned COMLLM reward for the controlled S7 lambda=0 / lambda=.3 arms.

Reference: Cassandra-Cheng/COMLLM-MEC-Task-Offloading, commit
17adedef587d5dbc02c167d147692a61179cdf4b: comllm_train.py,
grpo_new_7B.py, cost_calculator.py and mec_utils.py.

Both arms use COMLLM's task*.8 + format*.2, parser and thresholds. This is
not a literal reproduction of both training entry points: upstream plain
GRPO also changes format reward (number keywords and a -2 rather than -1
miss penalty), and defaults to group size 8 instead of COMLLM's 4.

The future term is K=3 random.uniform(.5*d, 1.5*d), including local actions;
the upstream comment excluding local actions disagrees with its actual code.
There is no one-slot service subtraction or sequential rollout of those three
future tasks. No PS teacher or 41-point deterministic grid is used here.

reward(text, row, lambda_lacs, seed=..., sample_key=...) is independent of
call order. Use the same seed and (step, prompt id) key for both arms. A
caller-owned random.Random can instead be passed as rng; saving/restoring
its getstate()/setstate() is then the caller's checkpoint responsibility.
The deterministic keyed stream is an explicit reproducibility adaptation
of upstream's process-global random stream, not a claim of bitwise training
reproduction. Only Python's standard library is required.
"""
import hashlib
import json
import math
import random
import re

UPSTREAM_COMMIT = "17adedef587d5dbc02c167d147692a61179cdf4b"
LOOKAHEAD_STEPS = 3
TASK_WEIGHT = 0.8
FORMAT_WEIGHT = 0.2
INVALID_ACTION_COST = 5.0


def parse_action(text):
    """Upstream parser: server mention first, then Chinese/English local."""
    if not isinstance(text, str):
        return None
    clean = text.lower().strip()
    match = re.search(r"(?:边缘服务器|服务器|server)\s*(\d+)", clean)
    if match and int(match.group(1)) > 0:
        return int(match.group(1))
    return 0 if "本地" in clean or "local" in clean else None


def format_reward(text):
    """COMLLM's format reward, deliberately shared by both controlled arms."""
    if not isinstance(text, str) or not text.strip():
        return -2.0
    clean = text.strip()
    lower = clean.lower()
    value = 0.5
    for keyword in ("json", "{", "}", "```", "action_id", "action_name"):
        if keyword in lower:
            value -= 0.5
    if not 4 <= len(clean) <= 30:
        value -= abs(len(clean) - 17.0) / 10.0
    value += 1.0 if any(k in lower for k in ("本地", "边缘", "local", "server")) else -1.0
    return value


def _number(text, pattern=r"([\d\.]+)", default=0.0):
    match = re.search(pattern, str(text)) if text else None
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return default


def _field(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _params(row):
    """Normalize raw friend input or sim.State without importing numpy/torch.

    S4 rows include rounded costs and state, but original paper_input wins.
    Raw sparse server IDs are retained instead of silently reindexing them.
    """
    if isinstance(row, dict):
        for key in ("paper_input", "friend_input", "raw", "input", "state_dict", "state"):
            if row.get(key) is not None:
                found = _params(row[key])
                if found is not None:
                    return found
        if "任务大小" in row:
            servers = []
            for key in sorted(row, key=lambda k: int(k[5:]) if re.fullmatch(r"边缘服务器\d+", k) else -1):
                if not re.fullmatch(r"边缘服务器\d+", key):
                    continue
                info = row[key]
                if not info or "None" in str(info):
                    continue
                servers.append({"id": int(key[5:]),
                                "active_tasks": int(_number(info, r"活跃任务数[：:]?\s*(\d+)")),
                                "queue_len": _number(info, r"服务器总队列长度[：:]?\s*([\d\.]+)"),
                                "cpu_freq": _number(info, r"CPU频率[：:]?\s*([\d\.]+)")})
            local = row.get("本地设备", "")
            return {"task_size": _number(row.get("任务大小")),
                    "local_wait": _number(row.get("本地CPU队列等待时间")),
                    "net_wait": _number(row.get("网络队列等待时间")),
                    "trans_rate": _number(row.get("网络上传速率")),
                    "local_cpu_freq": _number(local, r"CPU频率\s*([\d\.]+)"),
                    "cycles_per_bit": _number(local, r"每比特周期数\s*([\d\.]+)", 297.0),
                    "time_slot_duration": _number(row.get("时间槽时长", "0.1"), default=0.1),
                    "servers": servers}
    if _field(row, "d") is not None:
        return {"task_size": float(_field(row, "d")),
                "local_wait": float(_field(row, "local_wait", 0)),
                "net_wait": float(_field(row, "net_wait", 0)),
                "trans_rate": float(_field(row, "rate", 1.4)),
                "local_cpu_freq": float(_field(row, "local_ghz", 2.5)),
                "cycles_per_bit": float(_field(row, "cycles_per_bit", 297.0)),
                "time_slot_duration": float(_field(row, "time_slot_duration", 0.1)),
                "servers": [{"id": int(_field(v, "id", i + 1)),
                             "active_tasks": int(_field(v, "active")),
                             "queue_len": float(_field(v, "backlog")),
                             "cpu_freq": float(_field(v, "ghz"))}
                            for i, v in enumerate(_field(row, "servers", []))]}
    return None


def _local_rate(p, fallback=0.0):
    density = p["cycles_per_bit"] / 1000.0
    capacity = p["local_cpu_freq"] * p["time_slot_duration"]
    return capacity / density if capacity > 0 and density > 0 else fallback


def calculate_cost(action, row):
    """Paper one-step cost in slots; nonexistent positive server ID costs 5."""
    if action is None or action < 0:
        return float("inf")
    p = _params(row)
    if p is None:
        if isinstance(row, dict) and "costs" in row:
            return float(row["costs"][action]) if action < len(row["costs"]) else INVALID_ACTION_COST
        raise ValueError("reward row requires original paper input, state, or costs")
    density = p["cycles_per_bit"] / 1000.0
    if density <= 0:
        return float("inf")
    if action == 0:
        rate = _local_rate(p)
        return p["local_wait"] + p["task_size"] / rate if rate > 0 else float("inf")
    server = next((v for v in p["servers"] if v["id"] == action), None)
    if server is None:
        return INVALID_ACTION_COST
    if p["trans_rate"] <= 0:
        return float("inf")
    share = (server["cpu_freq"] * p["time_slot_duration"] / (server["active_tasks"] + 1)) / density
    if share <= 0:
        return float("inf")
    return (p["net_wait"] + p["task_size"] / p["trans_rate"]
            + server["queue_len"] / share + p["task_size"] / share)


def _rng_for(p, seed, sample_key):
    blob = json.dumps({"seed": seed, "sample_key": sample_key, "input": p},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return random.Random(int.from_bytes(hashlib.sha256(blob.encode("utf-8")).digest(), "big"))


def predict_future(action, row, *, rng=None, seed=0, sample_key=None, lookahead_steps=LOOKAHEAD_STEPS):
    """K independent hypothetical next-task minima after applying this action.

    rng must expose uniform(a,b). No process-global RNG is read or changed.
    The three hypothetical tasks do not themselves mutate the future state.
    """
    if not isinstance(lookahead_steps, int) or lookahead_steps <= 0:
        raise ValueError("lookahead_steps must be a positive integer")
    p = _params(row)
    if p is None:
        if isinstance(row, dict) and "future_cost" in row:
            return float(row["future_cost"])
        if isinstance(row, dict) and "future_costs" in row:
            return float(row["future_costs"][action])
        raise ValueError("future reward requires original state or explicit future_cost")
    density = p["cycles_per_bit"] / 1000.0
    if density <= 0:
        return 100.0
    stream = rng if rng is not None else _rng_for(p, seed, sample_key)
    servers = [dict(v) for v in p["servers"]]
    local_wait = p["local_wait"]
    if action == 0:
        local_wait += p["task_size"] / _local_rate(p, fallback=0.1)
    elif action > 0:
        for v in servers:
            if v["id"] == action:
                v["queue_len"] += p["task_size"]
                v["active_tasks"] += 1
                break
    local_rate = _local_rate(p, fallback=0.001)
    total = 0.0
    for _ in range(lookahead_steps):
        d = stream.uniform(p["task_size"] * 0.5, p["task_size"] * 1.5)
        local = local_wait + d / local_rate if local_rate > 0 else 100.0
        trans = d / p["trans_rate"] if p["trans_rate"] > 0 else 100.0
        edges = []
        for v in servers:
            active = v["active_tasks"] + 1
            share = (v["cpu_freq"] * p["time_slot_duration"] / active) / density if active > 0 else 0.001
            edges.append(p["net_wait"] + trans + v["queue_len"] / share + d / share if share > 0 else 100.0)
        total += min(local, min(edges) if edges else 100.0)
    return total / lookahead_steps


def reward_details(text, row, lambda_lacs=0.3, *, seed=0, sample_key=None, rng=None):
    """Return scalar reward and its serializable diagnostic components.

    The original raw input takes priority over row['costs'] (rounded by S4).
    A costs-only caller must provide future_cost or per-action future_costs
    whenever current cost < 10, even in the lambda=0 arm, so both arms use
    the same path. Missing state is a data/configuration error, not silently
    an alternative zero-future objective. Non-finite rewards fail early.
    """
    if not math.isfinite(lambda_lacs) or lambda_lacs < 0:
        raise ValueError("lambda_lacs must be finite and nonnegative")
    action = parse_action(text)
    fmt = format_reward(text)
    current = None
    future = 0.0
    total = None
    task = -10.0
    if action is not None:
        current = calculate_cost(action, row)
        if current < 10.0:
            future = predict_future(action, row, rng=rng, seed=seed, sample_key=sample_key)
        total = current + lambda_lacs * future
        task = 1.0 - total - (10.0 if total > 10.0 else 0.0)
    value = TASK_WEIGHT * task + FORMAT_WEIGHT * fmt
    if not math.isfinite(value):
        raise ValueError("non-finite reward: check source input units and CPU/upload rates")
    return {"reward": value, "task_reward": task, "format_reward": fmt,
            "action": action, "current_cost": current, "future_cost": future,
            "total_cost": total, "lambda_lacs": float(lambda_lacs), "lookahead_steps": LOOKAHEAD_STEPS}


def reward(text, row, lambda_lacs=0.3, *, seed=0, sample_key=None, rng=None):
    """Pure-Python scalar API for the S7 training runner."""
    return reward_details(text, row, lambda_lacs, seed=seed, sample_key=sample_key, rng=rng)["reward"]
