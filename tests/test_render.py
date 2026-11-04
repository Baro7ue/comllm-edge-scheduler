"""两种提示词渲染器的测试。"""
import json
import unittest

import numpy as np

from sim.baselines import make_policy
from sim.cost import best_action, one_step_costs
from sim.env import MecEnv
from sim.friend_data import DEFAULT_DIR, load_all, parse_entry
from sim.render import (LETTERS, paper_action_index, paper_action_name, parse_compact,
                        render_compact, render_paper)


class TestPaperFormat(unittest.TestCase):
    """G1 判据之一：论文格式渲染原作者数据的状态，结果逐字相同。"""

    @classmethod
    def setUpClass(cls):
        try:
            cls.rows = load_all()
        except FileNotFoundError:
            raise unittest.SkipTest(f"找不到原作者数据目录 {DEFAULT_DIR}")

    def test_verbatim(self):
        same = 0
        for _, e in self.rows:
            s, _ = parse_entry(e)
            r = render_paper(s)
            # 用 json.dumps 比较字符串，连键的顺序、整数还是小数（"丢弃惩罚": 100）都要一样
            same += (r["instruction"] == e["instruction"] and
                     json.dumps(r["input"], ensure_ascii=False) == json.dumps(e["input"], ensure_ascii=False))
        print(f"\n  [G1] 论文格式逐字一致：{same}/{len(self.rows)}")
        self.assertEqual(same, len(self.rows))

    def test_action_names(self):
        for _, e in self.rows:
            name = e["output"]["action_name"]
            self.assertEqual(paper_action_name(paper_action_index(name)), name)


def _sim_states(n_servers, n=200, seed=0):
    """从仿真器里取一些真实状态。"""
    env = MecEnv(n_terminals=30, n_servers=n_servers, seed=seed)
    greedy, states = make_policy("greedy"), []

    def record(s):
        states.append(s)
        return greedy(s)

    while len(states) < n:
        env.step(record)
    return states[:n]


class TestCompactFormat(unittest.TestCase):
    def test_roundtrip_keeps_label(self):
        """压缩格式里的信息足够算出标签：解析回来的状态，算出的每个动作延迟和渲染用的状态完全一样。"""
        for k in (3, 4, 7, 11):
            for s in _sim_states(k):
                msgs, used = render_compact(s)
                back = parse_compact(msgs[1]["content"])
                np.testing.assert_allclose(one_step_costs(back), one_step_costs(used), rtol=0, atol=1e-12)

    def test_letters(self):
        s = _sim_states(11, n=1)[0]
        msgs, _ = render_compact(s)
        lines = msgs[1]["content"].split("\n")
        self.assertEqual(len(lines), 1 + 12)                     # 1 行任务 + A–L 共 12 个动作
        self.assertEqual([ln[0] for ln in lines[1:]], list(LETTERS))

    def test_label_changes_are_rare(self):
        """四舍五入到压缩精度后，标签只在极少数（几乎并列的）状态上改变。"""
        states = _sim_states(7, n=500)
        changed = sum(best_action(one_step_costs(s)) != best_action(one_step_costs(render_compact(s)[1]))
                      for s in states)
        self.assertLess(changed / len(states), 0.05)


if __name__ == "__main__":
    unittest.main()
