"""把 runs/ 下所有延迟基准的 summary.json 汇总成一张 Markdown 表（图 2 的初版）。

  python -m eval.latency_table                 # 打印到屏幕
  python -m eval.latency_table > evidence/G2/latency_table.md

同一个配置（模型 + 格式 + 读法 + 缓存）跑过多次时，只取最新的一次。
"""
import glob
import json
import os

FMT_NAME = {"paper": "论文格式长提示词，生成动作名", "compact": "压缩提示词，单 token", "formula": "公式（不调 LLM）"}


def load_latest():
    """读 runs/*bench*/summary.json，按 (模型, 格式, 缓存, 读法) 只留最新一次，返回列表。

    1. 按目录名排序（目录名以 UTC 时间开头，排序就是时间顺序）
    2. 后读到的覆盖先读到的
    """
    paths = sorted(glob.glob(os.path.join("runs", "*bench*", "summary.json")))
    latest = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        s["run_dir"] = os.path.basename(os.path.dirname(p))
        latest[(s["model"], s["fmt"], s["cache"], s.get("decode", "-"))] = s
    return list(latest.values())


def p50_of(s):
    return s["e2e_p50"]


def main():
    rows = load_latest()
    rows.sort(key=p50_of, reverse=True)     # 从慢到快，读起来就是"一步步省下来"
    print("| 模型 | 提示词与输出 | 前缀缓存 | 提示词 token（缓存命中） | 生成 token | p50 | p99 | "
          "预填充 | 解码 | llama-server 其他 | Python 服务 | 网络+HTTP | N_max | 次数 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in rows:
        how = FMT_NAME.get(s["fmt"], s["fmt"])
        if s.get("decode") == "grammar":
            how = how + " + GBNF"
        print(f"| {s['model']} | {how} | {s['cache']} | "
              f"{s['prompt_tokens_mean']:.0f}（{s['cache_n_mean']:.0f}） | {s['predicted_n_mean']:.1f} | "
              f"{s['e2e_p50']:.1f} | {s['e2e_p99']:.1f} | {s['prompt_ms_mean']:.1f} | {s['predicted_ms_mean']:.1f} | "
              f"{s['llama_other_mean']:.1f} | {s['python_mean']:.1f} | {s['net_mean']:.1f} | "
              f"{s['n_max']:.1f} | {s['n']} |")
    print()
    print("时间单位都是 ms。端到端 p50/p99 在环境节点上计时；后面五列是平均值，加起来等于端到端平均。"
          "预填充、解码是 llama-server 自报的；llama-server 其他 = 分词、采样、拼 JSON；"
          "Python 服务 = 还原状态、拼提示词、解析答案；网络+HTTP = 网线往返和两端的 HTTP 处理。"
          "N_max = 1 / (3 × 端到端平均秒数)，即每台终端每秒 3 个请求时调度节点能撑住的终端数。")
    print()
    print("运行目录：")
    for s in rows:
        print(f"- {s['run_dir']}")


if __name__ == "__main__":
    main()
