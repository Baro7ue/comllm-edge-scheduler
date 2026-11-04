"""状态的数据结构 + 单步延迟公式 + LACS 未来影响。

动作编号约定（全项目统一）：a = 0 本地执行，a = k 卸载到第 k 台服务器（k 从 1 开始）。
所有时间单位都是"时间槽"（0.1 s）。
"""
from dataclasses import dataclass, field, replace

import numpy as np

from sim import config as C


# ===== 第 1 段：调度器看到的"状态" =====
# 一个 State = 一个新任务到达时，调度器做决策所需的全部信息。
# 字段和原作者数据里的 input 一一对应，所以论文格式渲染器可以逐字还原。
@dataclass
class ServerView:
    ghz: float                 # CPU 频率
    active: int                # 活跃任务数（调度器视角，含正在上传、还没到服务器的任务）
    backlog: float             # 服务器总队列长度：这些任务还没算完的总数据量，Mbits
    history: list = field(default_factory=lambda: [0.0] * C.HISTORY_LEN)   # 过去 10 槽的活跃任务数
    kind: str = ""             # 高性能型 / 标准型 / 节能型

    def __post_init__(self):
        if not self.kind:
            self.kind = C.SERVER_KIND[self.ghz]


@dataclass
class State:
    d: float                   # 新任务大小，Mbits
    local_wait: float          # 本地 CPU 队列等待时间，槽
    net_wait: float            # 上传队列等待时间，槽
    servers: list              # ServerView 列表，下标 0 是服务器 1
    rate: float = C.UPLINK_MBITS_PER_SLOT
    deadline: int = C.DEADLINE_SLOTS
    local_ghz: float = C.LOCAL_GHZ

    @property
    def n_actions(self):
        return len(self.servers) + 1


def rounded(s, nd_size=4, nd_other=4):
    """把状态里的数四舍五入（渲染成提示词前先做这一步，标签也用四舍五入后的值来算，
    保证模型看到的数字和标签用的数字完全一样）。"""
    return replace(
        s,
        d=round(s.d, nd_size),
        local_wait=round(s.local_wait, nd_other),
        net_wait=round(s.net_wait, nd_other),
        servers=[replace(v, backlog=round(v.backlog, nd_other)) for v in s.servers],
    )


# ===== 第 2 段：单步延迟公式 =====
# 关键观察：对固定的状态，每个动作的延迟都是任务大小 x 的一条直线  J_a(x) = alpha_a + beta_a * x
#   本地：      J_0(x) = 本地排队 + x / 本地速率
#   服务器 k：  J_k(x) = 上传排队 + x / 上传速率 + (积压 + x) / 分到的速率
#               分到的速率 = 服务器速率 / (活跃数 + 1)    ← 假设新任务加入后大家平分（processor sharing）
# 所以先算出每个动作的 (alpha, beta)，任何大小的任务都能直接代进去。
def action_lines(s):
    """返回两个长度为 n_actions 的数组 alpha, beta。"""
    alpha = [s.local_wait]
    beta = [1.0 / C.mbits_per_slot(s.local_ghz)]
    for v in s.servers:
        share = C.mbits_per_slot(v.ghz) / (v.active + 1)
        alpha.append(s.net_wait + v.backlog / share)
        beta.append(1.0 / s.rate + 1.0 / share)
    return np.array(alpha), np.array(beta)


def one_step_costs(s):
    """每个动作的预估总延迟（槽），返回 Python 列表。"""
    alpha, beta = action_lines(s)
    return (alpha + beta * s.d).tolist()


def best_action(costs):
    """延迟最小的动作。并列时取编号小的。"""
    return int(np.argmin(costs))


# ===== 第 3 段：LACS 的"未来影响" =====
# 做法：假装执行动作 a（本地队列变长，或目标服务器活跃数 +1、积压 +d），
# 再看"下一个任务"在新状态下最好能多快，对未来任务大小取平均。
# 论文是随机抽 3 个未来任务，有噪声；这里对 0.5d–1.5d 均匀取 41 个点求平均，相当于期望值，没有噪声。
FUTURE_GRID = np.linspace(0.5, 1.5, 41)


def lacs_impact(s, a, grid=FUTURE_GRID):
    """动作 a 的未来影响 C_a（槽）。"""
    lw, servers = s.local_wait, list(s.servers)
    if a == 0:
        lw = lw + s.d / C.mbits_per_slot(s.local_ghz)
    else:
        v = servers[a - 1]
        servers[a - 1] = replace(v, active=v.active + 1, backlog=v.backlog + s.d)
    alpha, beta = action_lines(replace(s, local_wait=lw, servers=servers))
    x = s.d * grid                                     # 41 个假想的未来任务大小
    future = alpha[:, None] + beta[:, None] * x[None, :]   # 形状 (动作数, 41)：每个未来任务选每个动作的延迟
    return float(future.min(axis=0).mean())            # 每个未来任务取最好的动作，再平均


def lacs_scores(s, lam=C.LACS_LAMBDA):
    """J_a + λ·C_a，越小越好。返回 (J 列表, C 列表, 总分列表)。"""
    J = one_step_costs(s)
    Cs = [lacs_impact(s, a) for a in range(s.n_actions)]
    return J, Cs, [j + lam * c for j, c in zip(J, Cs)]


# ===== 第 4 段（探索，不属于 G1）：更贴近 processor sharing 的单步预估 =====
# 论文公式里服务器那一项是 (积压 + d) × (n+1) / R：先按"排队"把积压算一遍，又按"平分"再乘 (n+1)，
# 对 PS 服务器来说算了两遍，负载越高越悲观。这里换成 PS 下的近似：
#   (1) 上传要 t_up 槽，这段时间服务器已经在算，积压先减掉 R × t_up；
#   (2) 新任务到达后，假设 n 个老任务剩余量相同（各 q/n），大家各分 R/(n+1)：
#       d 比老任务小 → 它先算完，用时 d(n+1)/R；d 比老任务大 → 老任务先走完，总用时 (q+d)/R。
#       两种情况合起来正好是 min(d(n+1), q+d) / R。
def ps_costs(s):
    costs = [s.local_wait + s.d / C.mbits_per_slot(s.local_ghz)]
    for v in s.servers:
        R = C.mbits_per_slot(v.ghz)
        t_up = s.net_wait + s.d / s.rate
        q = max(0.0, v.backlog - R * t_up)
        n = v.active if q > 0 else 0
        costs.append(t_up + min(s.d * (n + 1), q + s.d) / R)
    return costs
