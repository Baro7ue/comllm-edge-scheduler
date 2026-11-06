"""论文原版 1.5B 的 GGUF 比对：Jetson 上 llama.cpp（Q8_0）的回答，和阶段 3 在 Colab 上用 PyTorch 跑的回答是否一样。

阶段 3 的 runs/s3/paper_<时间>/sft/preds.csv 每行有 split、idx、pred（动作）、raw（原文）。
这里按同样的行号从 data/s3/<split>.jsonl 取出状态，用论文格式发给 Jetson 上的调度服务，比动作和原文。

在仓库根目录运行（Mac 上，Jetson 的调度服务已经加载 paper_sft_1.5b）：
  python -m eval.check_paper_gguf --preds runs/s3/paper_20261105T040018Z/sft/preds.csv
结果在 runs/<UTC>_gguf_paper_sft/：rows.csv 每条一行，summary.json 是汇总。
"""
import argparse
import csv
import json
import os
from datetime import datetime, timezone

from deploy.client import SchedulerClient, parse_host_port
from deploy.wire import state_from_dict
from eval.closed_loop import git_version

FIELDS = ["i", "split", "idx", "servers", "pt_action", "jetson_action", "agree", "pt_raw", "jetson_raw",
          "same_text", "source", "llm_ms"]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="阶段 3 论文原版基线的 preds.csv（sft 那个）")
    ap.add_argument("--data", default=os.path.join("data", "s3"), help="make_data 生成的测试集目录")
    ap.add_argument("--n", type=int, default=200, help="比多少条（两份测试集均匀抽）")
    ap.add_argument("--scheduler", default=None)
    ap.add_argument("--name", default="paper_sft")
    args = ap.parse_args()

    # 1. 读 PyTorch 的答案，只留带状态的两份测试集，均匀抽 n 条
    pt = []
    with open(args.preds, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] in ("test_seen", "test_unseen"):
                pt.append(r)
    step = max(1, len(pt) // args.n)
    picked = pt[::step][:args.n]
    data = {}
    for split in ("test_seen", "test_unseen"):
        data[split] = read_jsonl(os.path.join(args.data, split + ".jsonl"))

    # 2. 连调度服务，确认加载的是论文原版模型
    host, port = parse_host_port(args.scheduler)
    client = SchedulerClient(host, port)
    info = client.info()
    print("调度服务加载的模型：", info.get("model_key"), info.get("gguf"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join("runs", f"{stamp}_gguf_{args.name}")
    os.makedirs(out_dir, exist_ok=True)

    # 3. 逐条发请求（论文格式、超时 10 秒，不让公式回退混进来），比动作和原文
    rows = []
    agree = 0
    same = 0
    for i, r in enumerate(picked):
        row = data[r["split"]][int(r["idx"])]
        s = state_from_dict(row["state"])
        resp, _ = client.decide(s, "paper", True, 10000.0, req_id=i)
        a = resp["action"] if resp["source"] == "llm" else -1
        ok = int(a == int(r["pred"]))
        text_ok = int(resp.get("text", "").strip() == r["raw"].strip())
        agree += ok
        same += text_ok
        rows.append({"i": i, "split": r["split"], "idx": r["idx"], "servers": len(s.servers),
                     "pt_action": r["pred"], "jetson_action": a, "agree": ok, "pt_raw": r["raw"],
                     "jetson_raw": resp.get("text", ""), "same_text": text_ok, "source": resp["source"],
                     "llm_ms": round(resp.get("llm_ms", 0.0), 2)})
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(picked)}] 动作一致 {agree}/{i + 1}", flush=True)

    # 4. 写结果
    with open(os.path.join(out_dir, "rows.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary = {"n": len(rows), "action_agree": agree / len(rows), "text_same": same / len(rows),
               "preds": args.preds, "scheduler_info": info, "git": git_version()}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n{len(rows)} 条：动作一致 {100 * summary['action_agree']:.1f}%，回答原文完全一样 "
          f"{100 * summary['text_same']:.1f}%")
    print("写好了：", out_dir)


if __name__ == "__main__":
    main()
