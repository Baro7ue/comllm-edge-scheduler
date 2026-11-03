"""读测试床配置 deploy/testbed.json。

两块板子的地址、端口、超时阈值都只写在这一个文件里。
阶段 5 把环境节点从 Mac 换成 RK3588 时，只改 env_node 那一项。
"""
import json
import os

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testbed.json")


def load(path=None):
    """返回配置字典。path 不给就读 deploy/testbed.json。"""
    with open(path or DEFAULT_PATH, encoding="utf-8") as f:
        return json.load(f)
