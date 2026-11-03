"""实时模式用的仿真器：时间是带小数的"槽"，决策在收到的那一刻才生效。

和离线的 MecEnv 比，只多了两件事：
  1. 服务器可以推进到任意时刻（比如 12.37 槽），不必等到整槽。
  2. 决策用"收到决策的时刻"去执行：本地计算或上传从那一刻才开始，
     所以决策花掉的时间（网络 + 排队 + LLM 推理）会自然地算进任务延迟。

为什么推进到小数时刻不会出错：
  决策只会往服务器里加"将来才到"的任务。上传最快也要 2 Mbits ÷ 1.4 Mbits/槽 ≈ 1.43 槽，
  所以在时刻 r 做出的卸载决策，任务最早 r + 1.43 才到服务器，而服务器此时只推进到了 r。
"""
from sim.env import MecEnv


class RealtimeEnv(MecEnv):
    def __init__(self, n_terminals, n_servers=3, seed=0):
        super().__init__(n_terminals, n_servers, seed=seed, mode="sequential")
        self.now = 0.0   # 所有服务器已经推进到的时刻（槽）

    def advance_to(self, t):
        """把所有服务器从 self.now 推进到时刻 t。

        1. t 不比 self.now 晚：什么都不做（主线程和发送线程都会调用，时间只能往前走）
        2. 每台服务器从 self.now 推进到 t
        3. 记下 self.now = t
        """
        if t <= self.now:
            return
        for s in self.servers:
            s.advance(self.now, t)
        self.now = t

    def begin_slot(self, t):
        """第 t 槽开始时调用，返回这一槽新到达的任务（已打乱顺序）。

        1. 服务器推进到槽起点 t
        2. 记录每台服务器的活跃任务数（负载历史，论文格式提示词要用）
        3. 产生新任务：和离线仿真调用同一个函数，同一个种子得到同一串任务
        4. 新任务记进 self.tasks
        """
        self.advance_to(float(t))
        self.t = t
        for s in self.servers:
            s.history.append(float(s.view_active()))
        new = self.new_tasks(t)
        for task in new:
            self.tasks.append(task)
        return new

    def expire(self, task):
        """任务过了截止期还没拿到决策（或者决策请求失败）：记为在"决策"环节丢弃。"""
        task.dropped = True
        task.drop_at = "decision"
