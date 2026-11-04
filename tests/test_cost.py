"""单步延迟公式和 LACS 的测试。运行：python -m unittest discover -s tests -v"""
import unittest

from sim.cost import ServerView, State, best_action, lacs_scores, one_step_costs
from sim.friend_data import DEFAULT_DIR, load_all, parse_entry

# 原作者公开仓库里唯一的样例 MEC_data.json（10/18 手算过：本地约 7.98 槽，服务器 1 约 3.29 槽，标签是服务器 1）
REPO_SAMPLE = State(d=4.1934, local_wait=3.0, net_wait=0.0,
                    servers=[ServerView(41.8, 0, 0.0), ServerView(30.0, 0, 0.0), ServerView(22.0, 1, 7.2721)])


class TestRepoSample(unittest.TestCase):
    def test_one_step_costs(self):
        J = one_step_costs(REPO_SAMPLE)
        for got, want in zip(J, [7.98, 3.29, 3.41, 6.09]):
            self.assertAlmostEqual(got, want, places=2)
        self.assertEqual(best_action(J), 1)

    def test_lacs(self):
        # 10/23 的 LACS 检查算出的期望值：C = 3.29/3.41/3.29/3.29，J+0.3C = 8.97/4.32/4.40/7.08
        _, Cs, total = lacs_scores(REPO_SAMPLE)
        for got, want in zip(Cs, [3.29, 3.41, 3.29, 3.29]):
            self.assertAlmostEqual(got, want, places=2)
        for got, want in zip(total, [8.97, 4.32, 4.40, 7.08]):
            self.assertAlmostEqual(got, want, places=2)


class TestFriendData(unittest.TestCase):
    """G1 判据之一：公式在原作者的三份数据（1914 条）上 100% 复现标签。"""

    @classmethod
    def setUpClass(cls):
        try:
            cls.rows = load_all()
        except FileNotFoundError:
            raise unittest.SkipTest(f"找不到原作者数据目录 {DEFAULT_DIR}，用环境变量 COMLLM_FRIEND_DATA 指定")

    def test_argmin_matches_label(self):
        wrong = []
        for name, e in self.rows:
            s, label = parse_entry(e)
            if best_action(one_step_costs(s)) != label:
                wrong.append(name)
        print(f"\n  [G1] 公式复现标签：{len(self.rows) - len(wrong)}/{len(self.rows)}")
        self.assertEqual(len(self.rows), 1914)
        self.assertEqual(wrong, [])


if __name__ == "__main__":
    unittest.main()
