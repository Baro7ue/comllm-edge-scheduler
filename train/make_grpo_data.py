"""阶段 4 GRPO 的训练题和验证题：原作者 GRPO 数据集（998 条）里的状态，算好每个动作的成本。

在仓库根目录运行（Mac 上，激活 .venv）：
  python -m train.make_grpo_data
输出在 data/s4/（不进仓库；整个文件夹上传到 Google Drive 的 MyDrive/COMLLM/s4_data/）：
  grpo_train.jsonl   GRPO 训练用的题
  grpo_val.jsonl     训练中途看效果用的题（VAL_ROWS 条，不参与训练）
  grpo_reward.py     奖励函数，原样复制一份：Colab 上直接 import 它，和仓库里用的是同一个文件
  meta.json          条数、校验值、每种台数下最优动作的分布

每行的字段和阶段 3 的测试集一样（train/make_data.py 的 make_row），另加 friend_label = 原作者数据里的标签。
测试仍用阶段 3 的 friend_clean 和 test_seen（Drive 上的 s3_data），这里核对 GRPO 题和它们没有重复。
"""
import argparse
import json
import os
import shutil

import numpy as np

from eval.closed_loop import git_version
from sim.cost import rounded
from sim.friend_data import DEFAULT_DIR, FILES, load_all, parse_entry
from train.make_data import label_stats, make_row, md5_of, write_jsonl

VAL_ROWS = 100
SEED = 4


def input_key(e):
    """一条原作者样本的 input 转成文字，用来判断两条样本是不是同一个状态。"""
    return json.dumps(e["input"], ensure_ascii=False, sort_keys=True)


def build(data_dir):
    """返回 (训练行, 验证行, 统计)。

    1. 读原作者的三份数据，取出 GRPO 那一份
    2. 去掉 GRPO 数据内部的重复，以及和原作者测试集相同的状态（干净测试集本来就不含它们，这里再保险一次）
    3. 每条：解析成状态 → 四舍五入到 4 位小数 → 算成本，做成一行
    4. 按固定种子打乱，前 VAL_ROWS 条做验证，其余做训练
    """
    entries = load_all(data_dir)
    test_keys = set()
    for fname, e in entries:
        if fname == FILES[2]:
            test_keys.add(input_key(e))
    seen = set()
    rows = []
    dup = 0
    in_test = 0
    for fname, e in entries:
        if fname != FILES[1]:
            continue
        key = input_key(e)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        if key in test_keys:
            in_test += 1
            continue
        s, label = parse_entry(e)
        s4 = rounded(s, 4, 4)
        meta = {"servers": len(s4.servers), "n_terminals": -1, "policy": "friend", "seed": -1}
        row = make_row(s4, "grpo", meta, e["input"])
        row["friend_label"] = label
        rows.append(row)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    val = []
    train = []
    for j, i in enumerate(order):
        r = rows[int(i)]
        if j < VAL_ROWS:
            r["split"] = "grpo_val"
            val.append(r)
        else:
            r["split"] = "grpo_train"
            train.append(r)
    label_match = 0
    for r in rows:
        if r["friend_label"] == r["best"]:
            label_match += 1
    stats = {"dup_removed": dup, "in_friend_test_removed": in_test,
             "friend_label_equals_best_pct": round(100.0 * label_match / len(rows), 2)}
    return train, val, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "s4"), help="输出目录")
    ap.add_argument("--friend-dir", default=None, help="原作者数据目录，不给就用 sim/friend_data.py 里的默认位置")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train, val, stats = build(args.friend_dir or DEFAULT_DIR)
    files = {}
    for name, rows in (("grpo_train", train), ("grpo_val", val)):
        path = os.path.join(args.out, name + ".jsonl")
        write_jsonl(path, rows)
        files[name] = {"rows": len(rows), "md5": md5_of(path), "bytes": os.path.getsize(path)}
    reward_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grpo_reward.py")
    reward_dst = os.path.join(args.out, "grpo_reward.py")
    shutil.copyfile(reward_src, reward_dst)
    meta = {
        "version": "s4-v1",
        "git": git_version(),
        "seed": SEED,
        "files": files,
        "reward_md5": md5_of(reward_dst),
        "stats": stats,
        "labels": {"grpo_train": label_stats(train), "grpo_val": label_stats(val)},
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print("代码版本:", meta["git"])
    for name, info in files.items():
        print(f"{name:11s} {info['rows']:4d} 条  md5 {info['md5']}")
    print("去掉的重复:", stats["dup_removed"], " 去掉的和原作者测试集相同的:", stats["in_friend_test_removed"])
    print("原作者标签 = 单步最优的比例:", stats["friend_label_equals_best_pct"], "%")
    print("训练题每种台数的条数:", {k: v["n"] for k, v in meta["labels"]["grpo_train"].items()})
    print("写好了：", args.out)


if __name__ == "__main__":
    main()
