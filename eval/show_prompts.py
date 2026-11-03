"""看看模型到底会看到什么：从仿真器里取一个状态，分别用论文格式和压缩格式渲染出来，并统计字符数。

  python -m eval.show_prompts

token 数要用和部署时同一个分词器数才准，阶段 2 在 Jetson 上用 llama.cpp 的 llama-tokenize 实测。
"""
import json

from sim.baselines import make_policy
from sim.env import MecEnv
from sim.render import render_compact, render_paper


def sample_state(n_servers, n_terminals=30, seed=0, skip=300):
    """跑一段闭环，取第 skip 个决策时的状态（系统已经"热起来"了，不是空的）。"""
    env = MecEnv(n_terminals, n_servers, seed=seed)
    greedy, seen = make_policy("greedy"), []

    def record(s):
        seen.append(s)
        return greedy(s)

    while len(seen) <= skip:
        env.step(record)
    return seen[skip]


def main():
    for k in (3, 7, 11):
        s = sample_state(k)
        paper = render_paper(s)
        paper_text = paper["instruction"] + "\n" + json.dumps(paper["input"], ensure_ascii=False)
        msgs, _ = render_compact(s)
        print(f"===== {k} 台服务器 =====")
        print("--- 压缩格式 user 部分（每次都变）：")
        print(msgs[1]["content"])
        print(f"字符数：论文格式约 {len(paper_text)}；压缩格式 system {len(msgs[0]['content'])} + "
              f"user {len(msgs[1]['content'])}\n")
    print("--- 压缩格式 system 部分（不变，可被前缀缓存复用）：")
    print(msgs[0]["content"])


if __name__ == "__main__":
    main()
