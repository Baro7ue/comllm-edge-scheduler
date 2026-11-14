"""读测试床配置 deploy/testbed.json。

两块板子的地址、端口、超时阈值、模型清单都只写在这一个文件里。
阶段 5 把环境节点从 Mac 换成 RK3588 时，只改 env_node 那一项。

模型清单 models：模型键 → {"gguf": Jetson 上的文件路径, "fmt": 提示词格式}。
  main_0.5b       部署模型：0.5B 软标签（T = 0.3），Q8_0
  main_0.5b_q4    同一个模型的 Q4_K_M，只用于延迟表
  soft_1.5b       1.5B 软标签（T = 0.3），Q8_0
  ps_0.5b         0.5B，标签按平分算力的估计来算（实验 4 消融）
  base_0.5b       没训练过的 Qwen2.5-0.5B-Instruct，Q8_0
  paper_sft_1.5b  原作者的 sft 1.5B，转成 Q8_0，用原作者仓库的提示词拼法
  paper_grpo_1.5b / paper_score_grpo_1.5b  原作者两个 GRPO 1.5B，同样用论文格式
"""
import json
import os

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testbed.json")


def load(path=None):
    """返回配置字典。path 不给就读 deploy/testbed.json。"""
    with open(path or DEFAULT_PATH, encoding="utf-8") as f:
        return json.load(f)


def model_entry(key, cfg=None):
    """模型键 → {"gguf": ..., "fmt": ...}。键不在清单里就报错，并列出所有合法的键。"""
    cfg = cfg or load()
    models = cfg.get("models", {})
    if key not in models:
        raise KeyError("模型键 " + key + " 不在 deploy/testbed.json 的 models 里；可选：" + ", ".join(sorted(models)))
    return models[key]
