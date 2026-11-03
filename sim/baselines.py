"""四种不用 LLM 的基线策略。每个策略都是"可调用对象"：policy(state) → 动作编号。

  random  随机选一个动作                      下界
  greedy  单步最优：公式延迟最小的动作         监督训练的标签按它来算
  lacs    LACS-greedy：J + 0.3·C 最小的动作    论文 COMLLM 训练奖励的最优解，当上界参照
  po2     power-of-2：随机挑两个动作，选公式延迟小的那个    便宜的经典负载均衡方法

另有一个探索用的 psgreedy（不在计划的四种里，不属于 G1）：用 cost.ps_costs 代替论文公式取最小。
它用来回答"论文公式本身是不是闭环里的好老师"。
"""
import numpy as np

from sim import config as C
from sim.cost import best_action, lacs_scores, one_step_costs, ps_costs


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, s):
        return int(self.rng.integers(s.n_actions))


class GreedyPolicy:
    def __call__(self, s):
        return best_action(one_step_costs(s))


class LacsGreedyPolicy:
    def __init__(self, lam=C.LACS_LAMBDA):
        self.lam = lam

    def __call__(self, s):
        _, _, total = lacs_scores(s, self.lam)
        return best_action(total)


class PowerOfTwoPolicy:
    """在"本地 + 全部服务器"里不重复地随机抽两个，选公式延迟小的那个。
    经典 power-of-d 只在服务器之间抽；这里把本地也算一个选项，因为本地本来就是合法动作。"""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, s):
        a, b = self.rng.choice(s.n_actions, size=2, replace=False)
        costs = one_step_costs(s)
        return int(a) if costs[a] <= costs[b] else int(b)


class PsGreedyPolicy:
    def __call__(self, s):
        return best_action(ps_costs(s))


def make_policy(name, seed=0):
    """按名字造策略。seed 只影响策略自己的随机数（random、po2），和仿真器的随机数分开。"""
    return {
        "random": lambda: RandomPolicy(seed),
        "greedy": lambda: GreedyPolicy(),
        "lacs": lambda: LacsGreedyPolicy(),
        "po2": lambda: PowerOfTwoPolicy(seed),
        "psgreedy": lambda: PsGreedyPolicy(),
    }[name]()


POLICY_NAMES = ("random", "greedy", "lacs", "po2")   # 计划里的四种基线
EXPLORE_NAMES = ("psgreedy",)                        # 探索用
