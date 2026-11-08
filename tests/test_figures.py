"""图 8（两个老师 × 三种学生）的测试：用手造的几行结果，不需要真实运行目录。"""
import unittest

from eval import figures


def _row(policy, n, seed, delay, drop=0.0, servers=3, mode="sequential", slots=1000):
    return {"policy": policy, "n": str(n), "seed": str(seed), "servers": str(servers), "mode": mode,
            "slots": str(slots), "mean_delay": str(delay), "drop_rate": str(drop), "_dir": "run_x"}


def _grid(students, seeds=(0, 1), teacher_seeds=(0, 1, 2, 3)):
    """老师跑 4 个种子、学生跑 2 个种子；延迟 = 基准 + 种子号，方便算平均。"""
    rows = []
    for pol, base in (("greedy", 5.0), ("psgreedy", 4.0)):
        for n in (20, 40, 60):
            for k in teacher_seeds:
                rows.append(_row(pol, n, k, base + k))
    for pol in students:
        for n in (20, 40, 60):
            for k in seeds:
                rows.append(_row(pol, n, k, 4.5 + k, drop=0.01))
    return rows


class TestFig8Students(unittest.TestCase):
    def test_skips_until_three_students(self):
        numbers = []
        rows = _grid(["llm:main_0.5b", "llm:ps_0.5b"])
        self.assertIsNone(figures.fig8_students(rows, {}, numbers))
        self.assertEqual(numbers, [])

    def test_teacher_uses_only_student_seeds(self):
        numbers = []
        rows = _grid(["llm:main_0.5b", "llm:ps_0.5b", "llm:soft_1.5b", "llm:ps_soft_1.5b"])
        fig = figures.fig8_students(rows, {}, numbers)
        self.assertIsNotNone(fig)
        figures.plt.close(fig)
        by_name = {name: value for name, value, _ in numbers}
        # 老师只取种子 0 和 1：(5 + 6) / 2 = 5.50，不是 4 个种子的 6.50
        self.assertIn("平均延迟 5.50 / 5.50 / 5.50 槽", by_name["老师：论文公式（3 台，N = 20/40/60）"])
        self.assertIn("2 个种子", by_name["老师：论文公式（3 台，N = 20/40/60）"])
        self.assertIn("闭环没测", by_name["学生：论文公式老师 · 1.5B 硬标签（3 台，N = 20/40/60，Jetson Q8_0）"])
        pair = by_name["同样的学生，PS 老师 vs 论文公式老师（3 台，Jetson Q8_0）"]
        self.assertIn("0.5B 软标签", pair)
        self.assertIn("1.5B 软标签", pair)
        self.assertNotIn("1.5B 硬标签", pair)

    def test_open_loop_numbers_from_table(self):
        numbers = []
        rows = _grid(["llm:main_0.5b", "llm:ps_0.5b", "llm:soft_1.5b"])
        open_rows = {("1.5b_hard_ps", "friend_clean", "全部"): {"acc_pct": "20.0", "pr_pct": "75.5"}}
        fig = figures.fig8_students(rows, open_rows, numbers)
        figures.plt.close(fig)
        by_name = {name: value for name, value, _ in numbers}
        value = by_name["学生：PS 感知公式老师 · 1.5B 硬标签（3 台，N = 20/40/60，Jetson Q8_0）"]
        self.assertIn("开环准确率 20.0%、PR 75.50%", value)
        self.assertIn("闭环没测", value)

    def test_colab_platform(self):
        """Colab 那张图：老师行不重复写；复现的 7B 有全部 N 就多一行；行名带"Colab 原精度"。"""
        numbers = []
        rows = _grid(["llm:main_0.5b", "llm:ps_0.5b", "llm:soft_1.5b"])
        for n in (20, 40, 60):
            for k in (0, 1):
                rows.append(_row("llm:comllm_7b_repro", n, k, 4.2 + k))
        fig = figures.fig8_students(rows, {}, numbers, platform="colab")
        self.assertIsNotNone(fig)
        figures.plt.close(fig)
        names = [name for name, _, _ in numbers]
        for name in names:
            self.assertFalse(name.startswith("老师："), name)
        by_name = {name: value for name, value, _ in numbers}
        self.assertIn("学生：论文公式老师 · 0.5B 软标签（3 台，N = 20/40/60，Colab 原精度）", by_name)
        self.assertIn("平均延迟 4.70 / 4.70 / 4.70 槽", by_name["复现的 COMLLM-7B（论文的模型，3 台，N = 20/40/60，Colab 原精度）"])


class TestDeployGap(unittest.TestCase):
    def test_gap_by_point_name(self):
        """同一个点名：Jetson 5.0 槽、Colab 4.0 槽 → +25.0%；Colab 没有的学生不写。"""
        virtual = []
        colab = []
        for n in (20, 40):
            j = _row("llm:main_0.5b", n, 0, 5.0)
            j["point"] = f"llm_main_0.5b_k3_n{n}_sequential_s0_t1000"
            virtual.append(j)
            c = dict(j)
            c["mean_delay"] = "4.0"
            c["_dir"] = "run_colab"
            colab.append(c)
        other = _row("llm:soft_1.5b", 20, 0, 6.0)
        other["point"] = "llm_soft_1.5b_k3_n20_sequential_s0_t1000"
        virtual.append(other)
        numbers = []
        self.assertTrue(figures.deploy_gap(virtual, colab, numbers))
        self.assertEqual(len(numbers), 1)
        name, value, src = numbers[0]
        self.assertIn("Jetson Q8_0 比 Colab 原精度", name)
        self.assertIn("N = 20 +25.0%", value)
        self.assertIn("N = 40 +25.0%", value)
        self.assertNotIn("1.5B", value)
        self.assertEqual(src, "run_colab, run_x")

    def test_no_overlap(self):
        numbers = []
        self.assertFalse(figures.deploy_gap([], [], numbers))
        self.assertEqual(numbers, [])


if __name__ == "__main__":
    unittest.main()
