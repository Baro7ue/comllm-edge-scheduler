"""阶段 3 的训练、验证、测试数据：在离线闭环里收集调度器看到的状态，算出每个动作的成本。

在仓库根目录运行（Mac 上，激活 .venv）：
  python -m train.make_data
输出在 data/s3/（不进仓库；整个文件夹上传到 Google Drive 的 MyDrive/COMLLM/s3_data/）：
  train.jsonl         训练用，3/5/7/9/11 台服务器
  val.jsonl           验证用（选温度 T、看训练曲线），3/5/7/9/11 台
  test_seen.jsonl     测试：3/5/7/9/11 台，每种台数 600 条
  test_unseen.jsonl   测试：4/6/8/10 台，训练时没出现过，每种台数 600 条
  friend_clean.jsonl  原作者测试集里和他的训练集不重复的 245 条（找不到原作者数据时跳过）
  meta.json           参数、条数、校验值，以及给 Colab 核对提示词用的样例全文

每行一个 JSON，字段：
  user           压缩格式提示词的 user 部分。system 部分每行都一样，写在 meta.json 里
  n_actions      动作个数 = 服务器台数 + 1
  costs          论文单步公式算出的每个动作的延迟（槽），按 4 位小数的状态算，评测用
  costs_compact  同一个公式按压缩格式精度（任务大小 2 位、其余 1 位小数）的状态算，训练标签用
  ps_costs       平分算力版的估计，也按压缩格式精度算，可选消融用
  best           costs 最小的动作
  state          状态本身（4 位小数），Jetson 比对时原样发给调度服务。只有测试集和 friend_clean 有
  paper_input    论文格式的 input 字典，只有测试集和 friend_clean 有，论文原版模型评测用
  split, servers, n_terminals, policy, seed   这条状态属于哪个集合、从哪次仿真来

状态从哪来：每种 (服务器台数, 终端数 N, 行为策略, 种子) 跑一次闭环仿真，
热身 200 槽以后，每隔 stride 个决策记一个状态（隔开取，相邻状态太像）。
  训练集的行为策略：60% 贪心、20% 带 20% 随机动作的贪心、20% PS 感知贪心，
                    后两种让数据里也有"队列不均衡、服务器很忙"的状态，模型犯错以后也见过类似场面
  验证集和测试集：  只用贪心，和部署时会遇到的状态分布最接近
标签只看状态本身，和是哪个策略把系统带到这个状态无关。
"""
import argparse
import hashlib
import json
import os
from collections import Counter

import numpy as np

from deploy.prompts import QWEN_DEFAULT_SYSTEM, chatml, paper_messages_json
from deploy.wire import state_from_dict, state_to_dict
from eval.closed_loop import git_version
from sim import config as C
from sim.baselines import make_policy
from sim.cost import best_action, one_step_costs, ps_costs, rounded
from sim.env import MecEnv
from sim.render import COMPACT_SYSTEM, PAPER_INSTRUCTION, compact_state, render_compact, render_paper

N_LIST = (5, 10, 20, 30, 40, 60, 80)        # 终端数 N：从很闲到 3 台服务器明显过载
WARMUP_SLOTS = 200                           # 开头系统还空着，这段的状态不要
TRAIN_MIX = (("greedy", 0.6), ("eps_greedy", 0.2), ("psgreedy", 0.2))
EPS = 0.2                                    # eps_greedy 里随机选动作的概率

# 每个集合：用哪些服务器台数、每种台数要多少条、种子从几开始、用哪些行为策略
SPLITS = {
    "train":       {"servers": C.TRAIN_SERVER_COUNTS, "per_count": 20000, "seed0": 1000, "mix": TRAIN_MIX},
    "val":         {"servers": C.TRAIN_SERVER_COUNTS, "per_count": 400, "seed0": 2000, "mix": (("greedy", 1.0),)},
    "test_seen":   {"servers": C.TRAIN_SERVER_COUNTS, "per_count": 600, "seed0": 3000, "mix": (("greedy", 1.0),)},
    "test_unseen": {"servers": C.UNSEEN_SERVER_COUNTS, "per_count": 600, "seed0": 4000, "mix": (("greedy", 1.0),)},
}
EVAL_SPLITS = ("val", "test_seen", "test_unseen")   # 这些集合的状态不许出现在训练集里


# ===== 第 1 段：行为策略 =====
class EpsGreedy:
    """以 EPS 的概率随机选一个动作，否则选单步最优。"""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.greedy = make_policy("greedy")

    def __call__(self, s):
        if self.rng.random() < EPS:
            return int(self.rng.integers(s.n_actions))
        return self.greedy(s)


def make_behavior(name, seed):
    if name == "eps_greedy":
        return EpsGreedy(seed)
    return make_policy(name, seed)


# ===== 第 2 段：从一次闭环仿真里取状态 =====
def collect(n_servers, n_terminals, policy_name, seed, want):
    """跑一次闭环，返回 want 个状态（已经四舍五入到 4 位小数）。

    1. 每隔 stride 个决策取一个：stride 约等于 2 个时间槽里的决策数，最少 3
    2. 前 WARMUP_SLOTS 槽的决策不取
    3. 行为策略照常做决策，仿真照常推进，取够 want 个就停
    """
    stride = max(3, int(round(2 * C.ARRIVAL_PROB * n_terminals)))
    env = MecEnv(n_terminals, n_servers, seed=seed)
    behavior = make_behavior(policy_name, seed)
    picked = []
    count = [0]

    def record(s):
        if env.t >= WARMUP_SLOTS:
            count[0] += 1
            if count[0] % stride == 0 and len(picked) < want:
                picked.append(rounded(s, 4, 4))
        return behavior(s)

    while len(picked) < want:
        env.step(record)
    return picked


# ===== 第 3 段：一个状态 → 一行数据 =====
def round4(xs):
    out = []
    for x in xs:
        out.append(round(x, 4))
    return out


def make_row(s4, split, meta, paper_input=None):
    """s4：已经四舍五入到 4 位小数的状态。

    1. 压缩格式的 user 文本，由 s4 渲染（Jetson 上的调度服务收到 s4 也会渲染出同一段文本）
    2. 三组成本：4 位小数状态的论文公式（评测）、压缩精度状态的论文公式和 PS 版（训练标签）
    3. best 用未取整的成本选，再把成本取 4 位小数写出
    4. 测试集（给了 paper_input 的）额外带状态本身和论文格式的 input 字典
    """
    messages, sc = render_compact(s4)
    costs = one_step_costs(s4)
    row = {
        "split": split,
        "user": messages[1]["content"],
        "n_actions": s4.n_actions,
        "best": best_action(costs),
        "costs": round4(costs),
        "costs_compact": round4(one_step_costs(sc)),
        "ps_costs": round4(ps_costs(sc)),
    }
    for k, v in meta.items():
        row[k] = v
    if paper_input is not None:
        row["state"] = state_to_dict(s4)
        row["paper_input"] = paper_input
    return row


# ===== 第 4 段：生成一个集合 =====
def build_split(name, spec, n_list, scale):
    """返回这个集合的行列表（已在集合内去重）和去掉的重复条数。

    1. 每种服务器台数要 per_count 条，平均分给每个 N，再按比例分给每个行为策略
    2. 每个 (台数, N, 策略) 用一个不同的种子跑一次闭环，取状态
    3. 按 user 文本去重：完全一样的提示词只留第一条
    """
    rows = []
    seen = set()
    dup = 0
    seed = spec["seed0"]
    want_per_count = max(1, int(spec["per_count"] * scale))
    for k in spec["servers"]:
        for n in n_list:
            for policy_name, share in spec["mix"]:
                want = max(1, int(round(want_per_count / len(n_list) * share)))
                states = collect(k, n, policy_name, seed, want)
                meta = {"servers": k, "n_terminals": n, "policy": policy_name, "seed": seed}
                for s4 in states:
                    paper_input = None
                    if name != "train" and name != "val":
                        paper_input = render_paper(s4)["input"]
                    row = make_row(s4, name, meta, paper_input)
                    if row["user"] in seen:
                        dup += 1
                        continue
                    seen.add(row["user"])
                    rows.append(row)
                seed += 1
            print(f"  {name}: {k} 台服务器 N={n:2d} 完成，累计 {len(rows)} 条")
    return rows, dup


# ===== 第 5 段：原作者测试集里干净的那部分 =====
def build_friend_clean(data_dir):
    """原作者的测试集去掉和他的两份训练集完全相同的样本（按 input 比较），剩下的做成行。

    1. 把两份训练集的 input 转成文本放进集合
    2. 测试集里 input 不在集合里的，才算干净
    3. 状态用 4 位小数（原作者数据本来就是 4 位），论文格式 input 原样保留
    找不到原作者数据时返回 None。
    """
    from sim.friend_data import FILES, load_all, parse_entry
    try:
        entries = load_all(data_dir)
    except FileNotFoundError:
        return None
    train_inputs = set()
    for fname, e in entries:
        if fname != FILES[2]:
            train_inputs.add(json.dumps(e["input"], ensure_ascii=False, sort_keys=True))
    rows = []
    for fname, e in entries:
        if fname != FILES[2]:
            continue
        if json.dumps(e["input"], ensure_ascii=False, sort_keys=True) in train_inputs:
            continue
        s, label = parse_entry(e)
        s4 = rounded(s, 4, 4)
        meta = {"servers": len(s4.servers), "n_terminals": -1, "policy": "friend", "seed": -1}
        row = make_row(s4, "friend_clean", meta, e["input"])
        row["friend_label"] = label
        rows.append(row)
    return rows


# ===== 第 6 段：统计、写文件 =====
def label_stats(rows):
    """每种服务器台数下，最优动作的分布（百分比，0 = 本地）。"""
    out = {}
    by_k = {}
    for r in rows:
        by_k.setdefault(r["servers"], []).append(r["best"])
    for k in sorted(by_k):
        cnt = Counter(by_k[k])
        n = len(by_k[k])
        dist = {}
        for a in sorted(cnt):
            dist[str(a)] = round(100.0 * cnt[a] / n, 1)
        out[str(k)] = {"n": n, "best_pct": dist}
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def golden_texts(rows_by_split):
    """给 Colab 核对用的提示词全文：Colab 上 tokenizer.apply_chat_template 拼出来的必须和这里一字不差。"""
    compact = []
    for r in rows_by_split["train"][:3]:
        compact.append(chatml([{"role": "system", "content": COMPACT_SYSTEM}, {"role": "user", "content": r["user"]}]))
    paper = []
    for r in rows_by_split["test_seen"][:2]:
        paper.append(chatml(paper_messages_json(state_from_dict(r["state"]))))
    return {"compact": compact, "paper": paper}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "s3"), help="输出目录")
    ap.add_argument("--scale", type=float, default=1.0, help="条数乘这个系数，测试时用小数，比如 0.01")
    ap.add_argument("--friend-dir", default=None, help="原作者数据目录，不给就用 sim/friend_data.py 里的默认位置")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1. 先生成验证集和测试集，再生成训练集
    rows_by_split = {}
    dup_within = {}
    for name in ("val", "test_seen", "test_unseen", "train"):
        print(f"生成 {name} ……")
        rows, dup = build_split(name, SPLITS[name], N_LIST, args.scale)
        rows_by_split[name] = rows
        dup_within[name] = dup

    # 2. 原作者的干净测试集
    from sim.friend_data import DEFAULT_DIR
    friend_rows = build_friend_clean(args.friend_dir or DEFAULT_DIR)
    if friend_rows is None:
        print("没找到原作者数据，跳过 friend_clean")
    else:
        rows_by_split["friend_clean"] = friend_rows

    # 3. 训练集里去掉在任何评测集里出现过的提示词，保证训练和测试没有重复状态
    held_out = set()
    for name in rows_by_split:
        if name != "train":
            for r in rows_by_split[name]:
                held_out.add(r["user"])
    kept = []
    removed = 0
    for r in rows_by_split["train"]:
        if r["user"] in held_out:
            removed += 1
        else:
            kept.append(r)
    rows_by_split["train"] = kept

    # 4. 再核对一遍：训练集和每个评测集的交集都必须是 0
    train_users = set()
    for r in rows_by_split["train"]:
        train_users.add(r["user"])
    overlap = {}
    for name in rows_by_split:
        if name == "train":
            continue
        n = 0
        for r in rows_by_split[name]:
            if r["user"] in train_users:
                n += 1
        overlap[name] = n
        assert n == 0, f"{name} 和训练集有 {n} 条重复"

    # 5. 写文件，记统计和校验值
    files = {}
    for name, rows in rows_by_split.items():
        path = os.path.join(args.out, name + ".jsonl")
        write_jsonl(path, rows)
        files[name] = {"rows": len(rows), "md5": md5_of(path), "bytes": os.path.getsize(path)}
    splits_info = {}
    for k, v in SPLITS.items():
        mix = []
        for m in v["mix"]:
            mix.append(list(m))
        splits_info[k] = {"servers": list(v["servers"]), "per_count": v["per_count"], "seed0": v["seed0"], "mix": mix}
    labels = {}
    for name, rows in rows_by_split.items():
        labels[name] = label_stats(rows)
    meta = {
        "version": "s3-v1",
        "git": git_version(),
        "scale": args.scale,
        "n_list": list(N_LIST),
        "splits": splits_info,
        "eps": EPS,
        "files": files,
        "dup_within_split": dup_within,
        "train_removed_as_heldout": removed,
        "train_overlap_after": overlap,
        "labels": labels,
        "compact_system": COMPACT_SYSTEM,
        "qwen_default_system": QWEN_DEFAULT_SYSTEM,
        "paper_instruction": PAPER_INSTRUCTION,
        "letters": "ABCDEFGHIJKL",
        "deadline_slots": C.DEADLINE_SLOTS,
        "golden": golden_texts(rows_by_split),
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    # 6. 打印摘要（运行文档让你把这段存进 evidence/G3/data_stats.txt）
    print()
    print("代码版本:", meta["git"])
    for name, info in files.items():
        print(f"{name:13s} {info['rows']:6d} 条  md5 {info['md5']}  {info['bytes'] / 1e6:.1f} MB")
    print("集合内去掉的重复:", dup_within)
    print("训练集里因为和评测集相同而去掉的:", removed)
    print("训练集和各评测集的重复数:", overlap)
    for name in ("train", "test_seen", "test_unseen"):
        print(f"{name} 每种台数下最优动作的分布（%，0 = 本地）：")
        for k, st in meta["labels"][name].items():
            top = sorted(st["best_pct"].items(), key=lambda kv: -kv[1])[:4]
            print(f"  {k:>2s} 台：{st['n']} 条，本地 {st['best_pct'].get('0', 0.0)}%，最多的四个 {top}")
    print("写好了：", args.out)


if __name__ == "__main__":
    main()
