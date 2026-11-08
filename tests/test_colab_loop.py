"""Colab 闭环的测试：不需要 GPU 和 torch。作答换成单步公式，结果必须和 Jetson 那边同一个点用公式跑的一模一样。"""
import csv
import json
import os
import tempfile
import unittest

from eval import colab_loop, virtual_loop
from sim.baselines import make_policy
from sim.cost import best_action, one_step_costs


def formula_batch(states):
    out = []
    for s in states:
        out.append(best_action(one_step_costs(s)))
    return out


class TestColabLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "results.csv")
        self.points = []
        for p in colab_loop.points_for("main_0.5b")[:3]:
            p = dict(p)
            p["slots"] = 120
            p["warmup"] = 20
            p["point"] = p["point"].replace("t1000", "t120")
            self.points.append(p)

    def tearDown(self):
        self.tmp.cleanup()

    def test_points_match_jetson_names(self):
        names = {p["point"] for p in colab_loop.points_for("ps_soft_1.5b")}
        jetson = {p["point"] for p in virtual_loop.expand("e4ps") if p["policy"] == "llm:ps_soft_1.5b"}
        self.assertEqual(names, jetson)

    def test_batched_equals_one_by_one(self):
        """几个点拼批作答，结果和一个一个单独跑完全一样；拼批确实减少了前向次数。"""
        calls = []

        def batch(states):
            calls.append(len(states))
            return formula_batch(states)

        rows = colab_loop.run_points(self.points, batch, self.path, log=lambda *_: None)
        self.assertEqual(len(rows), 3)
        for p, row in zip(self.points, rows):
            m, _ = virtual_loop.run_point(p, make_policy("greedy"))
            self.assertAlmostEqual(float(row["mean_delay"]), round(m["mean_delay"], 5))
            self.assertEqual(int(row["fallback"]), 0)
            self.assertAlmostEqual(float(row["agree_greedy"]), 1.0)
        self.assertLess(len(calls), sum(int(r["n_decisions"]) for r in rows))
        self.assertGreater(max(calls), 1)
        # 再跑一次：都跑过了，不重复写
        self.assertEqual(colab_loop.run_points(self.points, batch, self.path, log=lambda *_: None), [])
        with open(self.path, newline="", encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 3)

    def test_invalid_falls_back_to_formula(self):
        rows = colab_loop.run_points(self.points[:1], lambda states: [-1] * len(states), self.path,
                                     log=lambda *_: None)
        self.assertEqual(int(rows[0]["fallback"]), int(rows[0]["n_decisions"]))
        m, _ = virtual_loop.run_point(self.points[0], make_policy("greedy"))
        self.assertAlmostEqual(float(rows[0]["mean_delay"]), round(m["mean_delay"], 5))

    def test_parse_paper(self):
        self.assertEqual(colab_loop.parse_paper("卸载到边缘服务器2", 4, "strict"), 2)
        self.assertEqual(colab_loop.parse_paper(" 本地执行\n", 4, "strict"), 0)
        self.assertEqual(colab_loop.parse_paper("选择：卸载到边缘服务器2", 4, "strict"), -1)
        self.assertEqual(colab_loop.parse_paper("选择：卸载到边缘服务器2", 4, "search"), 2)
        self.assertEqual(colab_loop.parse_paper("卸载到边缘服务器4", 4, "search"), -1)     # 3 台服务器，没有 4 号
        self.assertEqual(colab_loop.parse_paper("服务器", 4, "search"), -1)

    def test_pack(self):
        out = os.path.join(self.tmp.name, "s6_code")
        meta = colab_loop.pack(out)
        self.assertTrue(os.path.exists(os.path.join(out, "code.zip")))
        self.assertIn(os.path.join("eval", "colab_loop.py"), meta["files"])
        self.assertIn(os.path.join("sim", "env.py"), meta["files"])
        with open(os.path.join(out, "meta.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["check_mean_delay"], colab_loop.check_value())


if __name__ == "__main__":
    unittest.main()
