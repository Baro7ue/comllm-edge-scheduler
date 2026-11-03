"""决策延迟基准：从环境节点（Mac）向 Jetson 调度服务连发 N 次请求，记每次的端到端延迟和各段耗时。

一次只发一个请求，收到回复再发下一个，所以量到的是"没有排队时"单次决策要多久。
状态取自离线闭环仿真（贪心策略、30 台终端，系统已经有负载），3/7/11 台服务器轮流，
每种配置用的是同一串状态，配置之间可以直接比。

在仓库根目录运行（Mac 上，Jetson 的调度服务已经起来）：
  python -m eval.bench_latency --fmt compact --cache on
结果在 runs/<UTC>_bench_<模型>_<格式>_<缓存>/：requests.csv 每行一次请求，summary.json 是汇总。

各段耗时的关系（毫秒）：
  端到端（Mac 上量） = 网络和 HTTP + Jetson 服务内
  Jetson 服务内      = Python 服务（还原状态、拼提示词、解析） + 调 llama-server 的往返
  调 llama-server    = 预填充 + 解码 + llama-server 其他（分词、采样、JSON）
"""
import argparse
import csv
import json
import os
from datetime import datetime, timezone

from deploy.client import SchedulerClient, parse_host_port
from eval.closed_loop import git_version, int_list
from sim.baselines import make_policy
from sim.env import MecEnv

FIELDS = ["i", "servers", "e2e_ms", "service_ms", "net_ms", "queue_ms", "llm_ms", "prompt_n", "cache_n",
          "prompt_ms", "predicted_n", "predicted_ms", "source", "reason", "action", "text", "p_chosen",
          "letter_mass", "probs_argmax"]


def collect_states(server_counts, per_count, n_terminals=30, every=7, skip=300, seed=0):
    """从离线闭环里取状态。返回列表，按服务器台数轮流排：3, 7, 11, 3, 7, 11, ……

    1. 每种服务器台数各跑一段贪心闭环
    2. 跳过前 skip 个决策（系统还没热起来），之后每隔 every 个决策取一个状态，取够 per_count 个
    3. 按台数轮流交错排好
    """
    pools = []
    for k in server_counts:
        env = MecEnv(n_terminals, k, seed=seed)
        greedy = make_policy("greedy")
        seen = []
        count = [0]

        def record(s):
            count[0] += 1
            if count[0] > skip and count[0] % every == 0:
                seen.append(s)
            return greedy(s)

        while len(seen) < per_count:
            env.step(record)
        pools.append(seen[:per_count])
    out = []
    for i in range(per_count):
        for pool in pools:
            out.append(pool[i])
    return out


def percentile(values, q):
    """第 q 百分位（线性插值，和 numpy.percentile 的默认算法一样）。"""
    v = sorted(values)
    if len(v) == 0:
        return float("nan")
    pos = (len(v) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def mean(values):
    if len(values) == 0:
        return float("nan")
    total = 0.0
    for x in values:
        total += x
    return total / len(values)


def short_model_name(path):
    """"/home/ethan/models/qwen2.5-1.5b-instruct-q4_k_m.gguf" → "1.5b-q4_k_m"。认不出来就用文件名。"""
    name = os.path.basename(path or "unknown")
    if name.endswith(".gguf"):
        name = name[:-5]
    name = name.replace("qwen2.5-", "").replace("-instruct", "")
    return name


def summarize(rows, servers_list):
    """把每次请求的记录汇总成一个字典（毫秒）。

    1. 端到端延迟的 p50 / p90 / p99 / 平均 / 最大
    2. 各段平均：网络和 HTTP、Python 服务、llama-server 其他、预填充、解码（关系见文件开头）
    3. token 数平均：提示词总长（新算 + 缓存命中）、缓存命中、生成
    4. 来源计数（llm / fallback / formula）和压缩格式的概率检查：
       前 20 名候选里有没有合法字母、按概率取最大是否等于实际答案、合法字母一共占多少概率
    5. 理论容量 N_max = 1 / (3 × 平均端到端秒数)：每台终端每秒 3 个请求（0.3 个/槽 × 10 槽/秒）
    6. 按服务器台数分开的 p50
    """
    e2e = [r["e2e_ms"] for r in rows]
    s = {
        "n": len(rows),
        "e2e_p50": percentile(e2e, 50), "e2e_p90": percentile(e2e, 90), "e2e_p99": percentile(e2e, 99),
        "e2e_mean": mean(e2e), "e2e_max": max(e2e),
        "service_mean": mean([r["service_ms"] for r in rows]),
        "net_mean": mean([r["net_ms"] for r in rows]),
        "llm_mean": mean([r["llm_ms"] for r in rows]),
        "prompt_ms_mean": mean([r["prompt_ms"] for r in rows]),
        "predicted_ms_mean": mean([r["predicted_ms"] for r in rows]),
        "prompt_tokens_mean": mean([r["prompt_n"] + r["cache_n"] for r in rows]),
        "cache_n_mean": mean([r["cache_n"] for r in rows]),
        "predicted_n_mean": mean([r["predicted_n"] for r in rows]),
    }
    s["python_mean"] = s["service_mean"] - s["llm_mean"]
    s["llama_other_mean"] = s["llm_mean"] - s["prompt_ms_mean"] - s["predicted_ms_mean"]
    sources = {}
    for r in rows:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    s["sources"] = sources
    with_mass = []
    with_argmax = []
    for r in rows:
        if r["letter_mass"] != "":
            with_mass.append(r)
        if r["probs_argmax"] != "":
            with_argmax.append(r)
    if len(with_mass) > 0:
        agree = 0
        for r in with_argmax:
            if r["probs_argmax"] == r["action"]:
                agree += 1
        s["letters_found"] = len(with_argmax) / len(with_mass)       # 前 20 名候选里至少有一个合法字母的比例
        s["probs_agree"] = agree / len(with_argmax) if with_argmax else float("nan")
        s["letter_mass_mean"] = mean([r["letter_mass"] for r in with_mass])
    s["n_max"] = 1.0 / (3.0 * s["e2e_mean"] / 1000.0)
    by_servers = {}
    for k in servers_list:
        vals = [r["e2e_ms"] for r in rows if r["servers"] == k]
        by_servers[str(k)] = {"p50": percentile(vals, 50), "p99": percentile(vals, 99),
                              "prompt_tokens": mean([r["prompt_n"] + r["cache_n"] for r in rows if r["servers"] == k])}
    s["by_servers"] = by_servers
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", required=True, choices=["compact", "paper", "formula"])
    ap.add_argument("--cache", default="on", choices=["on", "off"], help="llama-server 前缀缓存开还是关")
    ap.add_argument("--decode", default="probs", choices=["probs", "grammar"], help="压缩格式怎么读答案")
    ap.add_argument("--n", type=int, default=500, help="计入统计的请求数")
    ap.add_argument("--warmup", type=int, default=10, help="开头先发几次不计入")
    ap.add_argument("--servers", type=int_list, default=[3, 7, 11])
    ap.add_argument("--scheduler", default=None, help="调度服务地址 IP:端口；不给就用 deploy/testbed.json")
    ap.add_argument("--name", default=None, help="目录名里的名字；不给就按 模型_格式_缓存 自动起")
    args = ap.parse_args()

    # 1. 连调度服务，问它加载的是哪个模型
    host, port = parse_host_port(args.scheduler)
    client = SchedulerClient(host, port)
    info = client.info()
    model = short_model_name(info.get("llama", {}).get("model_path"))
    if args.fmt == "formula":
        model = "formula"
    decode = args.decode if args.fmt == "compact" else "-"
    name = args.name
    if name is None:
        name = f"bench_{model}_{args.fmt}_cache{args.cache}"
        if decode == "grammar":
            name = name + "_grammar"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join("runs", f"{stamp}_{name}")
    os.makedirs(out_dir, exist_ok=True)

    # 2. 准备状态（每种配置都是同一串）
    total = args.warmup + args.n
    per_count = total // len(args.servers) + 1
    states = collect_states(args.servers, per_count)[:total]
    cache_on = args.cache == "on"

    # 3. 逐个发请求：前 warmup 个不记，之后每个记一行
    rows = []
    for i in range(total):
        s = states[i]
        resp, e2e_ms = client.decide(s, args.fmt, cache_on, timeout_ms=10000, req_id=i, decode=args.decode)
        if i < args.warmup:
            continue
        probs = resp.get("probs")
        letter_mass = ""
        probs_argmax = ""
        if probs is not None:
            letter_mass = 0.0
            best = 0
            for a in range(len(probs)):
                letter_mass += probs[a]
                if probs[a] > probs[best]:
                    best = a
            if letter_mass > 0:          # 前 20 名里一个合法字母都没有时，没法按概率取最大
                probs_argmax = best
        rows.append({
            "i": i - args.warmup, "servers": len(s.servers), "e2e_ms": e2e_ms,
            "service_ms": resp["service_ms"], "net_ms": e2e_ms - resp["service_ms"],
            "queue_ms": resp["queue_ms"], "llm_ms": resp["llm_ms"], "prompt_n": resp["prompt_n"],
            "cache_n": resp["cache_n"], "prompt_ms": resp["prompt_ms"], "predicted_n": resp["predicted_n"],
            "predicted_ms": resp["predicted_ms"], "source": resp["source"], "reason": resp["reason"],
            "action": resp["action"], "text": resp["text"], "p_chosen": resp["p_chosen"],
            "letter_mass": letter_mass, "probs_argmax": probs_argmax,
        })
        done = i - args.warmup + 1
        if done % 100 == 0:
            print(f"[{done}/{args.n}] 最近一次 {e2e_ms:.1f} ms，来源 {resp['source']}", flush=True)   # 接了 tee 也立刻显示

    # 4. 写结果
    with open(os.path.join(out_dir, "requests.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            out = {}
            for k in FIELDS:
                v = r[k]
                out[k] = round(v, 4) if isinstance(v, float) else v
            w.writerow(out)
    summary = summarize(rows, args.servers)
    summary.update(model=model, fmt=args.fmt, cache=args.cache, decode=decode, scheduler_info=info)
    config = {"args": vars(args), "git": git_version(), "started_utc": stamp,
              "scheduler": f"{client.host}:{client.port}", "scheduler_info": info}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 5. 打印汇总
    print(f"\n配置：{model}  格式 {args.fmt}  读法 {decode}  前缀缓存 {args.cache}  共 {summary['n']} 次")
    print(f"端到端（Mac 上量）：p50 {summary['e2e_p50']:.1f} ms  p99 {summary['e2e_p99']:.1f} ms  "
          f"平均 {summary['e2e_mean']:.1f} ms  最大 {summary['e2e_max']:.1f} ms")
    print(f"平均拆分（ms）：网络+HTTP {summary['net_mean']:.1f} | Python 服务 {summary['python_mean']:.1f} | "
          f"llama-server 其他 {summary['llama_other_mean']:.1f} | 预填充 {summary['prompt_ms_mean']:.1f} | "
          f"解码 {summary['predicted_ms_mean']:.1f}")
    print(f"提示词平均 {summary['prompt_tokens_mean']:.0f} token，缓存命中 {summary['cache_n_mean']:.0f}，"
          f"生成 {summary['predicted_n_mean']:.1f}")
    print(f"来源：{summary['sources']}")
    if "probs_agree" in summary:
        print(f"概率检查：前 20 名候选里有合法字母 {100 * summary['letters_found']:.1f}%；"
              f"其中按概率取最大和实际答案一致 {100 * summary['probs_agree']:.1f}%；"
              f"合法字母平均共占概率 {100 * summary['letter_mass_mean']:.1f}%")
    print(f"理论容量 N_max = 1/(3 × {summary['e2e_mean'] / 1000:.4f} s) = {summary['n_max']:.1f} 台终端")
    for k, v in summary["by_servers"].items():
        print(f"  {k} 台服务器：p50 {v['p50']:.1f} ms  p99 {v['p99']:.1f} ms  提示词 {v['prompt_tokens']:.0f} token")
    print("写好了：", out_dir)


if __name__ == "__main__":
    main()
