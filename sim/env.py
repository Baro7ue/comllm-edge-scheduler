"""闭环仿真器：N 台终端 + K 台边缘服务器，按时间槽推进。

"闭环"的意思：调度器的每个决策都会改变后面的队列，后面的任务看到的是被前面决策改变过的状态。
对比"开环"评测：每条测试样本单独打分，决策不影响任何后续状态。

每个时间槽 t 内按顺序发生：
  1. 记录每台服务器的活跃任务数（负载历史）
  2. 每台终端以 0.3 的概率产生一个新任务
  3. 调度策略给每个新任务选动作（本地 / 服务器 k）
  4. 服务器从 t 推进到 t+1：上传完的任务到达、按 processor sharing 计算、超过截止期的丢弃

三种排队（单位都是槽）：
  - 本地 CPU：每台终端一条先来先服务（FIFO）队列，速率 0.842 Mbits/槽
  - 上传链路：每台终端一条 FIFO 队列，速率 1.4 Mbits/槽
  - 边缘服务器：processor sharing，n 个任务同时在算时各分到 1/n 的算力
任务从到达起超过 10 槽还没算完就丢弃，丢弃时立刻离开队列，不再占资源。
"""
import heapq
from collections import deque
from dataclasses import dataclass

import numpy as np

from sim import config as C
from sim.cost import ServerView, State, one_step_costs
from sim.terminals import RandomTerminal

EPS = 1e-9   # 浮点比较的容差


# ===== 第 1 段：一个任务的完整记录 =====
@dataclass
class Task:
    tid: int                 # 任务编号
    terminal: int            # 来自哪台终端
    size: float              # Mbits
    t_arrive: float          # 到达时刻（绝对时间，槽）
    deadline: float          # 截止时刻（绝对时间）= 到达 + 10
    action: int = -1         # 调度器选的动作
    pred: float = 0.0        # 决策时公式预估的延迟（用来对比"公式说多少"和"实际多少"）
    t_done: float = None     # 完成时刻；None 表示还没完成
    dropped: bool = False
    drop_at: str = ""        # 在哪里被丢弃：local / uplink / server

    @property
    def delay(self):
        return self.t_done - self.t_arrive

    @property
    def resolved(self):
        return self.dropped or self.t_done is not None


# ===== 第 2 段：一台边缘服务器（processor sharing） =====
class PsServer:
    def __init__(self, ghz):
        self.ghz = ghz
        self.rate = C.mbits_per_slot(ghz)    # 整台服务器每槽能算多少 Mbits
        self.jobs = []                       # 正在算的任务：[task, 剩余 Mbits]
        self.incoming = []                   # 已经分到这台、还在上传的任务：小顶堆，元素 (到达服务器的时刻, tid, task)
        self.history = deque([0.0] * C.HISTORY_LEN, maxlen=C.HISTORY_LEN)

    # 调度器视角的状态：把"正在上传、还没到"的任务也算上。
    # 理由：调度器知道自己刚派过去哪些任务；它们很快就会到，新任务到服务器时要和它们一起分算力。
    # 这也是后面修复羊群效应（顺序决策 + 虚拟状态更新）能起作用的前提。
    def view_active(self):
        return len(self.jobs) + len(self.incoming)

    def view_backlog(self):
        return sum(rem for _, rem in self.jobs) + sum(task.size for _, _, task in self.incoming)

    def add_incoming(self, t_arrive_server, task):
        heapq.heappush(self.incoming, (t_arrive_server, task.tid, task))

    def advance(self, t0, t1):
        """把这台服务器从时刻 t0 推进到 t1。用"下一个事件"的方式走：
        两个事件之间活跃任务数不变，每个任务的速度也不变，可以直接算出走了多少。"""
        now = t0
        while now < t1 - EPS:
            # (1) 上传完的任务进入服务器
            while self.incoming and self.incoming[0][0] <= now + EPS:
                _, _, task = heapq.heappop(self.incoming)
                self.jobs.append([task, task.size])
            t_arr = self.incoming[0][0] if self.incoming else float("inf")
            if not self.jobs:                          # 没有任务在算：直接跳到下一个到达，或者跳到 t1
                now = min(t_arr, t1)
                continue
            # (2) 下一个事件：有任务算完 / 有新任务到达 / 有任务到截止期 / 到 t1，取最早的
            per_job = self.rate / len(self.jobs)       # processor sharing：每个任务分到 1/n
            t_fin = now + min(rem for _, rem in self.jobs) / per_job
            t_dl = min(task.deadline for task, _ in self.jobs)
            t_next = min(t_fin, t_arr, t_dl, t1)
            # (3) 每个任务都按 per_job 的速度算了 (t_next - now) 这么久
            work = per_job * (t_next - now)
            for job in self.jobs:
                job[1] -= work
            now = t_next
            # (4) 算完的记完成时刻；没算完但已到截止期的丢弃
            keep = []
            for task, rem in self.jobs:
                if rem <= EPS:
                    task.t_done = now
                elif now >= task.deadline - EPS:
                    task.dropped, task.drop_at = True, "server"
                else:
                    keep.append([task, rem])
            self.jobs = keep


# ===== 第 3 段：整个系统 =====
class MecEnv:
    def __init__(self, n_terminals, n_servers=3, seed=0, mode="sequential", terminals=None):
        """mode：同一个时间槽里的多个新任务怎么决策
          sequential：一个接一个，后一个看到前一个决策之后的状态（每个决策只要几微秒的公式基线用这个）
          batch：     同一槽的任务都看槽开始时的同一份状态（模拟 LLM 批量推理，会出现羊群效应）"""
        assert mode in ("sequential", "batch")
        self.mode = mode
        self.lineup = C.SERVER_LINEUPS[n_servers]
        self.servers = [PsServer(g) for g in self.lineup]
        self.terminals = terminals or [RandomTerminal() for _ in range(n_terminals)]
        n = len(self.terminals)
        self.local_free = [0.0] * n    # 每台终端的本地 CPU 在哪个时刻空出来
        self.up_free = [0.0] * n       # 每台终端的上传链路在哪个时刻空出来
        self.rng = np.random.default_rng(seed)   # 只管任务的产生和顺序，和策略的随机数分开
        self.t = 0
        self.tasks = []

    def observe(self, task):
        """新任务 task 到达时，调度器看到的状态。"""
        i, t = task.terminal, self.t
        return State(
            d=task.size,
            local_wait=max(0.0, self.local_free[i] - t),
            net_wait=max(0.0, self.up_free[i] - t),
            servers=[ServerView(ghz=s.ghz, active=s.view_active(), backlog=s.view_backlog(),
                                history=list(s.history)) for s in self.servers],
        )

    def apply(self, task, a, state):
        """执行动作 a。本地和上传都是 FIFO，所以完成时刻在决策这一刻就能算出来。"""
        task.action = a
        task.pred = one_step_costs(state)[a]
        i, t = task.terminal, self.t
        if a == 0:                                         # 本地执行
            start = max(t, self.local_free[i])             # 前面的任务做完才轮到它
            finish = start + task.size / C.LOCAL_MBITS_PER_SLOT
            if finish <= task.deadline + EPS:
                task.t_done = finish
                self.local_free[i] = finish
            else:                                          # 赶不上截止期：算到截止期为止，然后丢弃
                task.dropped, task.drop_at = True, "local"
                self.local_free[i] = max(start, task.deadline)
        else:                                              # 卸载到服务器 a：先上传
            start = max(t, self.up_free[i])
            up_done = start + task.size / C.UPLINK_MBITS_PER_SLOT
            if up_done <= task.deadline + EPS:
                self.up_free[i] = up_done
                self.servers[a - 1].add_incoming(up_done, task)
            else:                                          # 传都传不完：在上传队列里丢弃
                task.dropped, task.drop_at = True, "uplink"
                self.up_free[i] = max(start, task.deadline)

    def step(self, policy):
        """推进一个时间槽。policy 是一个函数：输入 State，输出动作编号。"""
        t = self.t
        for s in self.servers:
            s.history.append(float(s.view_active()))
        new = []
        for i, term in enumerate(self.terminals):
            size = term.maybe_task(t, self.rng)
            if size is not None:
                new.append(Task(tid=len(self.tasks) + len(new), terminal=i, size=size,
                                t_arrive=float(t), deadline=float(t + C.DEADLINE_SLOTS)))
        new = [new[k] for k in self.rng.permutation(len(new))]   # 打乱同一槽内的决策顺序
        if self.mode == "sequential":
            for task in new:
                st = self.observe(task)
                self.apply(task, policy(st), st)
        else:
            states = [self.observe(task) for task in new]          # 先全部观测（同一份旧状态）
            actions = [policy(st) for st in states]                # 再全部决策
            for task, st, a in zip(new, states, actions):
                self.apply(task, a, st)
        self.tasks.extend(new)
        for s in self.servers:
            s.advance(t, t + 1)
        self.t += 1
        return new


# ===== 第 4 段：跑一次、出指标 =====
def summarize(tasks, n_servers):
    """平均延迟只算完成的任务；丢弃率单独算；负载均衡用 Jain 指数（按卸载到每台服务器的任务数，n = 真实台数）。"""
    n = len(tasks)
    done = [t for t in tasks if not t.dropped]
    delays = np.array([t.delay for t in done]) if done else np.array([np.nan])
    gaps = np.array([t.delay - t.pred for t in done]) if done else np.array([np.nan])
    counts = np.bincount([t.action for t in tasks], minlength=n_servers + 1)
    srv = counts[1:].astype(float)
    off = srv.sum()
    return {
        "n_tasks": n,
        "mean_delay": float(np.mean(delays)),            # 槽
        "p95_delay": float(np.percentile(delays, 95)),
        "drop_rate": (n - len(done)) / n if n else float("nan"),
        "local_frac": float(counts[0] / n) if n else float("nan"),
        "top_server_frac": float(srv.max() / off) if off else float("nan"),   # 卸载的任务里，最热门那台占多少
        "jain": float(off ** 2 / (n_servers * (srv ** 2).sum())) if off else float("nan"),
        "pred_gap": float(np.mean(gaps)),                 # 实际延迟 − 公式预估延迟（只算完成的任务）
    }


def run(policy, n_terminals, n_servers=3, slots=3000, warmup=200, seed=0, mode="sequential"):
    """跑 slots 个时间槽，返回 (指标字典, env)。
    只统计在 [warmup, slots − 11) 之间到达的任务：开头系统还空着（没热起来），
    结尾到达的任务可能来不及结束。中间这段任务到 slots 时一定已经完成或被丢弃。"""
    env = MecEnv(n_terminals, n_servers, seed=seed, mode=mode)
    for _ in range(slots):
        env.step(policy)
    counted = [t for t in env.tasks if warmup <= t.t_arrive < slots - C.DEADLINE_SLOTS - 1]
    assert all(t.resolved for t in counted), "有任务没结束，检查仿真逻辑"
    return summarize(counted, n_servers), env
