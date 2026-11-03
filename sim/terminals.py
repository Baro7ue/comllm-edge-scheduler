"""终端（产生任务的一方）。

主线只用 RandomTerminal。DuckTerminal 是第 3 周可选演示（RK3588 上用 MuJoCo 仿真的机器鸭）留的接口，
现在先继承同一套任务分布，保证以后换成机器鸭时，仿真器和评测脚本一行都不用改。
"""
from sim import config as C


class RandomTerminal:
    """每个时间槽以 p 的概率产生一个任务，大小在 size_range 里均匀分布。"""

    def __init__(self, p=C.ARRIVAL_PROB, size_range=C.TASK_MBITS):
        self.p = p
        self.size_range = size_range

    def maybe_task(self, t, rng):
        """第 t 槽有没有新任务：有就返回任务大小（Mbits），没有返回 None。
        两个随机数每槽都抽，不管有没有任务：每台终端每槽固定用掉两个随机数。
        不同策略面对同一串任务，靠的是 env.rng 只用来产生任务和打乱顺序，策略用的是自己的随机数。"""
        u = rng.random()
        size = rng.uniform(*self.size_range)
        return size if u < self.p else None


class DuckTerminal(RandomTerminal):
    """演示用的机器鸭终端（可选，第 3 周）。现在只是空壳：任务分布和 RandomTerminal 完全一样。
    以后接 MuJoCo 时，改写 maybe_task（例如每当鸭子的摄像头出一帧就产生一次检测任务），
    函数签名保持不变。"""

    def __init__(self, duck_id, **kwargs):
        super().__init__(**kwargs)
        self.duck_id = duck_id
