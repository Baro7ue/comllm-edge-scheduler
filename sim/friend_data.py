"""读朋友的三份数据集，把每条样本解析成 State + 标签动作。

数据不进仓库。默认从 Mac 上的项目资料文件夹读，
也可以用环境变量 COMLLM_FRIEND_DATA 指定别的目录。
"""
import json
import os
import re

from sim.cost import ServerView, State
from sim.render import paper_action_index

DEFAULT_DIR = os.path.expanduser(
    os.environ.get("COMLLM_FRIEND_DATA", "~/Documents/COMLLM训练部署及改良/from_friend/data"))
FILES = ("sft_train_data_610.json", "grpo_train_data_1000.json", "test_data_1000.json")

_SERVER_RE = re.compile(
    r"类型 (\S+), CPU频率 ([\d.]+) GHz, 每比特周期数 ([\d.]+), 活跃任务数(\d+)，"
    r"服务器总队列长度([\d.]+) Mbits, 过去10个时间槽负载: \[(.*)\]")


def _num(text):
    """"3.6995 Mbits" → 3.6995"""
    return float(re.match(r"-?[\d.]+", text).group())


def parse_entry(e):
    """一条样本 → (State, 标签动作编号)。"""
    x = e["input"]
    servers, k = [], 1
    while f"边缘服务器{k}" in x:
        m = _SERVER_RE.fullmatch(x[f"边缘服务器{k}"])
        servers.append(ServerView(
            ghz=float(m.group(2)), active=int(m.group(4)), backlog=float(m.group(5)),
            history=[float(h) for h in m.group(6).split(", ")], kind=m.group(1)))
        k += 1
    s = State(d=_num(x["任务大小"]), local_wait=_num(x["本地CPU队列等待时间"]),
              net_wait=_num(x["网络队列等待时间"]), servers=servers,
              rate=_num(x["网络上传速率"]), deadline=int(_num(x["最大延迟"])),
              local_ghz=float(re.search(r"CPU频率 ([\d.]+) GHz", x["本地设备"]).group(1)))
    return s, paper_action_index(e["output"]["action_name"])


def load_all(data_dir=DEFAULT_DIR):
    """返回 [(文件名, 原始样本字典), ...]，三份文件按顺序拼在一起。目录不存在时抛 FileNotFoundError。"""
    rows = []
    for name in FILES:
        with open(os.path.join(data_dir, name), encoding="utf-8") as f:
            rows += [(name, e) for e in json.load(f)]
    return rows
