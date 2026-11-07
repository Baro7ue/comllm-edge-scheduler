"""原作者 1.5B 的 GGUF 和 PyTorch 回答不一样时，看 Jetson 上的模型在两边分岔的那一步有多犹豫。

check_paper_gguf 发现动作一致率低时用。这里不经过调度服务，直接问 llama-server，要每个生成 token 的前 10 名候选概率，
在 Jetson 的回答和 PyTorch 的回答第一次分岔的那一步，比两边各自选的那个 token 的概率：
  两个概率很接近        → 模型本来就在两个动作之间犹豫，量化带来的一点点误差就能翻过去
  PyTorch 选的不在前 10 名 → 不像同一个模型，要查 GGUF 转换
另外每条都关掉提示词缓存重答一遍，和 check_paper_gguf（开着缓存）的回答比：同一台机器、同一个文件，开关缓存就翻的，也是犹豫。

llama-server 只听 Jetson 本机，所以先在 Mac 上另开一个终端开 ssh 隧道（跑完 Ctrl-C 关掉）：
  ssh -N -L 18080:127.0.0.1:8080 Jetson
再在仓库根目录运行（Jetson 上加载的还是 paper_sft_1.5b）：
  python -m eval.probe_paper_gguf --rows runs/<时间>_gguf_paper_sft/rows.csv
结果写在 rows.csv 旁边：probe.csv 每条一行，probe_summary.json 是汇总。
"""
import argparse
import csv
import json
import math
import os

from deploy.http_json import JsonConnection
from deploy.prompts import GREEDY, PAPER_N_PREDICT, chatml, paper_messages
from deploy.wire import state_from_dict
from eval.check_paper_gguf import read_jsonl

N_PROBS = 10
END_TOKENS = ("", "<|im_end|>", "<|endoftext|>")
FIELDS = ["i", "split", "idx", "servers", "agree", "pt_raw", "jetson_raw", "probe_raw", "same_as_cached",
          "split_step", "p_jetson", "p_pytorch", "pytorch_rank", "min_margin", "min_margin_step"]


def split_point(cp, pt_text):
    """Jetson 逐个生成的 token 和 PyTorch 的回答原文对齐，找第一次分岔的那一步。

    返回 (步号, Jetson 那个 token 的概率, PyTorch 那条路的 token 在前 10 名里的概率, 名次)。
    PyTorch 那条路不在前 10 名：概率记 0、名次记 None。一直没分岔：返回 (None, None, None, None)。

    1. 已对齐的前缀从空开始，逐步取 Jetson 生成的 token
    2. PyTorch 原文已经对完、Jetson 这一步是结束符 → 两边一样，没分岔
    3. PyTorch 原文剩下的部分以这个 token 开头 → 前缀接上，看下一步
    4. 否则就是分岔点：在这一步的前 10 名候选里找能接上 PyTorch 原文的 token（剩下为空时找结束符），取概率最大的
    """
    prefix = ""
    for step in range(len(cp)):
        rec = cp[step]
        tok = rec.get("token", "")
        rest = pt_text[len(prefix):]
        if rest == "" and tok in END_TOKENS:
            return None, None, None, None
        if tok != "" and rest.startswith(tok):
            prefix = prefix + tok
            continue
        p_jetson = math.exp(rec["logprob"])
        p_pt = 0.0
        rank = None
        cands = rec.get("top_logprobs", [])
        for r in range(len(cands)):
            c = cands[r].get("token", "")
            if rest == "":
                hit = c in END_TOKENS
            else:
                hit = c != "" and rest.startswith(c)
            p = math.exp(cands[r]["logprob"])
            if hit and p > p_pt:
                p_pt = p
                rank = r + 1
        return step, p_jetson, p_pt, rank
    return None, None, None, None


def min_margin(cp):
    """每一步前两名候选的概率差，取最小的那一步：返回 (差, 步号)。这一步就是模型最犹豫的地方。

    1. 逐步取前 10 名候选，按概率从大到小排
    2. 第一名减第二名（只有一个候选时第二名记 0）
    3. 记下差最小的那一步
    """
    best = None
    best_step = None
    for step in range(len(cp)):
        ps = []
        for cand in cp[step].get("top_logprobs", []):
            ps.append(math.exp(cand["logprob"]))
        ps.sort(reverse=True)
        if len(ps) == 0:
            continue
        second = ps[1] if len(ps) > 1 else 0.0
        m = ps[0] - second
        if best is None or m < best:
            best = m
            best_step = step
    return best, best_step


def median(xs):
    if len(xs) == 0:
        return None
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="check_paper_gguf 写的 rows.csv")
    ap.add_argument("--data", default=os.path.join("data", "s3"))
    ap.add_argument("--llama", default="127.0.0.1:18080", help="llama-server 地址（经 ssh 隧道）")
    args = ap.parse_args()

    # 1. 读 check_paper_gguf 的逐条结果和测试集
    with open(args.rows, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    data = {}
    for split in ("test_seen", "test_unseen"):
        data[split] = read_jsonl(os.path.join(args.data, split + ".jsonl"))
    host, port = args.llama.rsplit(":", 1)
    conn = JsonConnection(host, int(port), timeout_s=60.0)

    # 2. 每条：同样的提示词和采样参数，关掉缓存、要前 10 名候选概率
    out = []
    for i in range(len(rows)):
        r = rows[i]
        s = state_from_dict(data[r["split"]][int(r["idx"])]["state"])
        body = {"prompt": chatml(paper_messages(s)), "n_predict": PAPER_N_PREDICT,
                "cache_prompt": False, "n_probs": N_PROBS, "stream": False,
                "response_fields": ["content", "completion_probabilities"]}
        for key in GREEDY:
            body[key] = GREEDY[key]
        resp, _ = conn.request("POST", "/completion", body)
        cp = resp.get("completion_probabilities") or []
        text = resp.get("content", "").strip()

        # 3. 和 PyTorch 原文找分岔点；再找最犹豫的那一步
        step, p_j, p_pt, rank = split_point(cp, r["pt_raw"].strip())
        mm, mm_step = min_margin(cp)
        out.append({"i": r["i"], "split": r["split"], "idx": r["idx"], "servers": r["servers"], "agree": r["agree"],
                    "pt_raw": r["pt_raw"], "jetson_raw": r["jetson_raw"], "probe_raw": text,
                    "same_as_cached": int(text == r["jetson_raw"].strip()),
                    "split_step": "" if step is None else step,
                    "p_jetson": "" if p_j is None else round(p_j, 4),
                    "p_pytorch": "" if p_pt is None else round(p_pt, 4),
                    "pytorch_rank": "" if rank is None else rank,
                    "min_margin": "" if mm is None else round(mm, 4),
                    "min_margin_step": "" if mm_step is None else mm_step})
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(rows)}]", flush=True)

    # 4. 汇总：只看和 PyTorch 分岔的那些条
    diverged = []
    for o in out:
        if o["split_step"] != "":
            diverged.append(o)
    p_j_list = []
    p_pt_list = []
    gap_list = []
    in_top = 0
    close = 0
    for o in diverged:
        p_j_list.append(o["p_jetson"])
        p_pt_list.append(o["p_pytorch"])
        gap_list.append(o["p_jetson"] - o["p_pytorch"])
        if o["pytorch_rank"] != "":
            in_top += 1
            if o["p_jetson"] - o["p_pytorch"] < 0.2:
                close += 1
    same_cached = 0
    margins = []
    for o in out:
        same_cached += o["same_as_cached"]
        if o["min_margin"] != "":
            margins.append(o["min_margin"])
    summary = {"n": len(out), "same_as_cached": same_cached, "diverged_from_pytorch": len(diverged),
               "pytorch_token_in_top10": in_top, "gap_below_0.2": close,
               "median_p_jetson": median(p_j_list), "median_p_pytorch": median(p_pt_list),
               "median_gap": median(gap_list), "median_min_margin_all_rows": median(margins)}
    out_dir = os.path.dirname(os.path.abspath(args.rows))
    with open(os.path.join(out_dir, "probe.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for o in out:
            w.writerow(o)
    with open(os.path.join(out_dir, "probe_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print()
    print(f"{len(out)} 条：关缓存重答和 check_paper_gguf 一样 {same_cached} 条")
    print(f"和 PyTorch 分岔 {len(diverged)} 条，其中 PyTorch 那条路在前 10 名 {in_top} 条，两边概率差 < 0.2 的 {close} 条")
    print(f"分岔那一步的中位概率：Jetson 选的 {summary['median_p_jetson']}，PyTorch 选的 {summary['median_p_pytorch']}")
    print(f"所有条最犹豫那一步的前两名概率差（中位）：{summary['median_min_margin_all_rows']}")
    print("写好了：", os.path.join(out_dir, "probe.csv"))


if __name__ == "__main__":
    main()
