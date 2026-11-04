"""GGUF 比对：同样 200 条状态，Jetson 上的 llama.cpp 选的动作和 Colab 上 PyTorch 选的是否一样。

Colab 训练 notebook 最后一步会写 expected_200.jsonl：每行一条测试状态，带 PyTorch（合并 LoRA 后的 bf16 模型）
按部署规则选出的动作（前 20 名候选里概率最大的合法字母）和它领先第二名的幅度。
这里把每条状态原样发给 Jetson 上的调度服务，走的是部署时的同一条路（拼提示词 → llama-server → 读概率）。

在仓库根目录运行（Mac 上，Jetson 的调度服务已经用要比对的那个 .gguf 起来）：
  python -m eval.check_gguf --expected <expected_200.jsonl 的路径> --name 1.5b_soft_q4
结果在 runs/<UTC>_gguf_<名字>/：rows.csv 每条一行，summary.json 是汇总。

超时阈值默认给 5000 ms：这里比的是"答案一不一样"，不能让超时回退的公式答案混进来。
"""
import argparse
import csv
import json
import os
from datetime import datetime, timezone

from deploy.client import SchedulerClient, parse_host_port
from deploy.wire import state_from_dict
from eval.closed_loop import git_version
from sim.render import render_compact

FIELDS = ["i", "split", "idx", "servers", "pt_action", "jetson_action", "agree", "pt_margin", "extra_cost",
          "source", "reason", "text_ok", "llm_ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected", required=True, help="Colab 写的 expected_200.jsonl")
    ap.add_argument("--name", required=True, help="这次比对的名字，比如 1.5b_soft_q4")
    ap.add_argument("--timeout-ms", type=int, default=5000)
    ap.add_argument("--scheduler", default=None, help="调度服务地址 IP:端口，不给就用 deploy/testbed.json")
    args = ap.parse_args()

    # 1. 读期望答案；连调度服务，记下它加载的是哪个模型文件
    rows = []
    with open(args.expected, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    host, port = parse_host_port(args.scheduler)
    client = SchedulerClient(host, port)
    info = client.info()
    llama = info.get("llama") or {}
    model_path = llama.get("model_path")
    print("调度服务加载的模型:", model_path, "，llama.cpp 版本:", llama.get("build_info"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join("runs", f"{stamp}_gguf_{args.name}")
    os.makedirs(out_dir, exist_ok=True)

    # 2. 逐条发请求，比动作
    #    text_ok：Mac 这边用同一个状态渲染出的提示词和训练时的完全一样（不一样说明数据或代码版本对不上）
    #    extra_cost：Jetson 选的动作比 PyTorch 选的多花多少槽（一样时为 0）
    results = []
    for i, r in enumerate(rows):
        s = state_from_dict(r["state"])
        text_ok = render_compact(s)[0][1]["content"] == r["user"]
        resp, _ = client.decide(s, "compact", True, args.timeout_ms, i)
        a = resp.get("action")
        agree = a == r["pt_action"]
        extra = 0.0
        if a is not None and 0 <= a < len(r["costs"]):
            extra = r["costs"][a] - r["costs"][r["pt_action"]]
        results.append({"i": i, "split": r["split"], "idx": r["idx"], "servers": r["servers"],
                        "pt_action": r["pt_action"], "jetson_action": a, "agree": int(agree),
                        "pt_margin": round(r["pt_margin"], 4), "extra_cost": round(extra, 4),
                        "source": resp.get("source"), "reason": resp.get("reason"), "text_ok": int(text_ok),
                        "llm_ms": resp.get("llm_ms")})

    # 3. 汇总：一致率；不一致的那几条 PyTorch 领先多少、Jetson 多花多少
    n = len(results)
    n_agree = 0
    n_llm = 0
    n_text_bad = 0
    disagree = []
    for x in results:
        n_agree += x["agree"]
        if x["source"] == "llm":
            n_llm += 1
        if not x["text_ok"]:
            n_text_bad += 1
        if not x["agree"]:
            disagree.append(x)
    extra_total = 0.0
    extra_max = 0.0
    for x in disagree:
        extra_total += x["extra_cost"]
        if x["extra_cost"] > extra_max:
            extra_max = x["extra_cost"]
    summary = {
        "name": args.name, "expected": os.path.abspath(args.expected), "model_path": model_path,
        "llama_build": llama.get("build_info"),
        "n": n, "agree": n_agree, "agree_pct": round(100.0 * n_agree / n, 2),
        "answered_by_llm": n_llm, "prompt_text_mismatch": n_text_bad,
        "disagree_mean_extra_cost": round(extra_total / len(disagree), 4) if disagree else 0.0,
        "disagree_max_extra_cost": round(extra_max, 4),
        "mean_extra_cost_all": round(extra_total / n, 5),
        "git": git_version(), "started_utc": stamp, "timeout_ms": args.timeout_ms,
    }
    with open(os.path.join(out_dir, "rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for x in results:
            w.writerow(x)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"一致 {n_agree}/{n}（{summary['agree_pct']}%），LLM 自己答的 {n_llm}/{n}，提示词对不上的 {n_text_bad} 条")
    if disagree:
        print(f"不一致的 {len(disagree)} 条：Jetson 平均多花 {summary['disagree_mean_extra_cost']} 槽，"
              f"最多 {summary['disagree_max_extra_cost']} 槽；平摊到 {n} 条是 {summary['mean_extra_cost_all']} 槽/条")
        for x in disagree:
            print(f"  第 {x['i']} 条（{x['servers']} 台）：PyTorch 选 {x['pt_action']}（领先 {x['pt_margin']}），"
                  f"Jetson 选 {x['jetson_action']}，多花 {x['extra_cost']} 槽，来源 {x['source']} {x['reason']}")
    print("写好了：", out_dir)


if __name__ == "__main__":
    main()
