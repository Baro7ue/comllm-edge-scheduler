"""GGUF 比对：同样 200 条状态，Jetson 上的 llama.cpp 选的动作和 Colab 上 PyTorch 选的是否一样。

Colab 训练 notebook 最后一步会写 expected_200.jsonl：每行一条测试状态，带 PyTorch（合并 LoRA 后的 bf16 模型）
按部署规则选出的动作（前 20 名候选里概率最大的合法字母）和它领先第二名的幅度。
这里把每条状态原样发给 Jetson 上的调度服务，走的是部署时的同一条路（拼提示词 → llama-server → 读概率）。

在仓库根目录运行（Mac 上，Jetson 的调度服务已经用要比对的那个 .gguf 起来）：
  python -m eval.check_gguf --expected <expected_200.jsonl 的路径> --name 1.5b_soft_q4
结果在 runs/<UTC>_gguf_<名字>/：rows.csv 每条一行，summary.json 是汇总。

超时阈值默认给 5000 ms：这里比的是"答案一不一样"，不能让超时回退的公式答案混进来。

11/4 的训练 notebook 写 expected_200.jsonl 时有个 bug：用"行号"当字典的键，两份测试集的行号会重复，
后算的那行把先算的覆盖了，200 条里约 89 条的参考答案其实是另一行的。这里会自动认出这些行，
改用同一个运行目录里 preds.csv 的答案（合并前的 LoRA 模型，同样是 PyTorch）。preds.csv 不在
expected_200.jsonl 的上两级目录时，用 --preds 指定。
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

FIELDS = ["i", "split", "idx", "servers", "ref", "pt_action", "jetson_action", "agree", "pt_margin", "extra_cost",
          "source", "reason", "text_ok", "llm_ms"]


def load_preds_csv(path):
    """preds.csv → {(split, 行号): (动作, 领先幅度)}；文件不存在返回空字典。"""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["split"], int(r["idx"]))] = (int(r["pred"]), float(r["margin"]))
    return out


def repair_reference(rows, preds):
    """认出参考答案被别的行覆盖了的行，改用 preds.csv 的答案。返回 (修了几条, 没修的行里和 preds.csv 一致几条)。

    被覆盖的行，pt_letter_probs 的长度是另一行的动作数；两份测试集的台数一奇一偶，长度一定对不上。
    1. 逐行看 pt_letter_probs 的长度是不是等于 n_actions
    2. 等于 → 参考答案就是这一行的（合并后的模型），ref = "merged"；顺便和 preds.csv 比一下，应该几乎全一样
    3. 不等 → 动作和领先幅度换成 preds.csv 里同一行的，ref = "lora"；preds.csv 里没有这一行就停下
    """
    n_fixed = 0
    n_same = 0
    for r in rows:
        key = (r["split"], r["idx"])
        if len(r["pt_letter_probs"]) == r["n_actions"]:
            r["ref"] = "merged"
            if key in preds and preds[key][0] == r["pt_action"]:
                n_same += 1
            continue
        if key not in preds:
            raise SystemExit(f"第 {r['split']} 第 {r['idx']} 行的参考答案是坏的，要用 preds.csv 补，"
                             "但没找到 preds.csv 或里面没有这一行：用 --preds 指定这个运行目录里的 preds.csv")
        r["pt_action"] = preds[key][0]
        r["pt_margin"] = preds[key][1]
        r["ref"] = "lora"
        n_fixed += 1
    return n_fixed, n_same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected", required=True, help="Colab 写的 expected_200.jsonl")
    ap.add_argument("--name", required=True, help="这次比对的名字，比如 1.5b_soft_q4")
    ap.add_argument("--timeout-ms", type=int, default=5000)
    ap.add_argument("--scheduler", default=None, help="调度服务地址 IP:端口，不给就用 deploy/testbed.json")
    ap.add_argument("--preds", default=None, help="同一个运行目录的 preds.csv，不给就在 expected 的上两级目录找")
    args = ap.parse_args()

    # 1. 读期望答案，修好被覆盖的参考答案；连调度服务，记下它加载的是哪个模型文件
    rows = []
    with open(args.expected, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    preds_path = args.preds
    if preds_path is None:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.expected)))
        preds_path = os.path.join(run_dir, "preds.csv")
    n_fixed, n_same = repair_reference(rows, load_preds_csv(preds_path))
    n_merged = len(rows) - n_fixed
    print(f"参考答案：合并后的模型 {n_merged} 条（其中和 preds.csv 一致 {n_same} 条），用 preds.csv 补的 {n_fixed} 条")
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
        pt_ok = 0 <= r["pt_action"] < len(r["costs"])
        if a is not None and 0 <= a < len(r["costs"]) and pt_ok:
            extra = r["costs"][a] - r["costs"][r["pt_action"]]
        results.append({"i": i, "split": r["split"], "idx": r["idx"], "servers": r["servers"], "ref": r["ref"],
                        "pt_action": r["pt_action"], "jetson_action": a, "agree": int(agree),
                        "pt_margin": round(r["pt_margin"], 4), "extra_cost": round(extra, 4),
                        "source": resp.get("source"), "reason": resp.get("reason"), "text_ok": int(text_ok),
                        "llm_ms": resp.get("llm_ms")})

    # 3. 汇总：一致率；不一致的那几条 PyTorch 领先多少、Jetson 多花多少
    n = len(results)
    n_agree = 0
    n_agree_merged = 0
    n_llm = 0
    n_text_bad = 0
    disagree = []
    for x in results:
        n_agree += x["agree"]
        if x["ref"] == "merged":
            n_agree_merged += x["agree"]
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
        "ref_merged": n_merged, "ref_merged_same_as_preds_csv": n_same, "ref_from_preds_csv": n_fixed,
        "agree_on_merged_ref": n_agree_merged, "agree_on_preds_csv_ref": n_agree - n_agree_merged,
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
    print(f"  按参考答案来源：合并后的模型 {n_agree_merged}/{n_merged}，preds.csv 补的 {n_agree - n_agree_merged}/{n_fixed}")
    if disagree:
        print(f"不一致的 {len(disagree)} 条：Jetson 平均多花 {summary['disagree_mean_extra_cost']} 槽，"
              f"最多 {summary['disagree_max_extra_cost']} 槽；平摊到 {n} 条是 {summary['mean_extra_cost_all']} 槽/条")
        for x in disagree:
            print(f"  第 {x['i']} 条（{x['servers']} 台，参考 {x['ref']}）：PyTorch 选 {x['pt_action']}（领先 {x['pt_margin']}），"
                  f"Jetson 选 {x['jetson_action']}，多花 {x['extra_cost']} 槽，来源 {x['source']} {x['reason']}")
    print("写好了：", out_dir)


if __name__ == "__main__":
    main()
