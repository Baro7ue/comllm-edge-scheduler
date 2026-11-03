"""环境节点和调度服务之间传的"状态"：State 和 JSON 字典互相转换。

请求里只有调度器本来就能看到的东西（新任务大小、终端两条队列的等待、每台服务器的状态），
没有仿真里的任何真值（比如任务实际什么时候算完）。
"""
from sim.cost import ServerView, State


def state_to_dict(s):
    """State → 可以 json.dumps 的字典。

    1. 任务和终端的字段原样写出
    2. 每台服务器写成一个小字典，按服务器编号顺序放进列表
    """
    servers = []
    for v in s.servers:
        servers.append({"ghz": v.ghz, "active": v.active, "backlog": v.backlog, "history": list(v.history)})
    return {
        "d": s.d,
        "local_wait": s.local_wait,
        "net_wait": s.net_wait,
        "rate": s.rate,
        "deadline": s.deadline,
        "local_ghz": s.local_ghz,
        "servers": servers,
    }


def state_from_dict(x):
    """state_to_dict 的逆过程。

    1. 逐台还原 ServerView（服务器类型由频率查表得到）
    2. 拼回 State
    """
    servers = []
    for v in x["servers"]:
        servers.append(ServerView(ghz=float(v["ghz"]), active=int(v["active"]),
                                  backlog=float(v["backlog"]), history=[float(h) for h in v["history"]]))
    return State(d=float(x["d"]), local_wait=float(x["local_wait"]), net_wait=float(x["net_wait"]),
                 servers=servers, rate=float(x["rate"]), deadline=int(x["deadline"]),
                 local_ghz=float(x["local_ghz"]))
