"""阶段 3 训练数据和开环评测的测试。

数据用很小的比例（--scale 0.01）现生成到临时目录，几秒钟就跑完。
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from deploy.prompts import chatml, paper_messages_json
from deploy.wire import state_from_dict
from eval.check_gguf import repair_reference
from eval.open_loop import effective_cost, jain, score
from sim.cost import one_step_costs
from sim.render import COMPACT_SYSTEM, render_compact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_data(out_dir):
    """在 out_dir 里跑一次 python -m train.make_data --scale 0.01，返回 meta.json 的内容。"""
    subprocess.run([sys.executable, "-m", "train.make_data", "--out", out_dir, "--scale", "0.01"],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def read_rows(out_dir, split):
    rows = []
    with open(os.path.join(out_dir, split + ".jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class TestMakeData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = cls.tmp.name
        cls.meta = make_data(cls.dir)
        cls.rows = {}
        for split in cls.meta["files"]:
            cls.rows[split] = read_rows(cls.dir, split)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_no_train_test_overlap(self):
        """训练集的提示词和任何评测集都不重复。"""
        train = set()
        for r in self.rows["train"]:
            train.add(r["user"])
        for split, rows in self.rows.items():
            if split == "train":
                continue
            for r in rows:
                self.assertNotIn(r["user"], train, split)

    def test_server_counts(self):
        """训练、验证、见过的测试集只有奇数台；没见过的测试集只有 4/6/8/10 台。"""
        for split in ("train", "val", "test_seen"):
            for r in self.rows[split]:
                self.assertIn(r["servers"], (3, 5, 7, 9, 11))
        for r in self.rows["test_unseen"]:
            self.assertIn(r["servers"], (4, 6, 8, 10))

    def test_row_fields(self):
        """动作个数 = 台数 + 1；best 是 costs 里最小的那个（costs 已取 4 位小数，允许 1e-4 误差）。"""
        for split, rows in self.rows.items():
            for r in rows:
                self.assertEqual(r["n_actions"], r["servers"] + 1)
                self.assertEqual(len(r["costs"]), r["n_actions"])
                self.assertEqual(len(r["costs_compact"]), r["n_actions"])
                self.assertLessEqual(r["costs"][r["best"]], min(r["costs"]) + 1e-4)

    def test_state_renders_same_text(self):
        """测试集里存的状态，在 Mac / Jetson 上重新渲染，得到和数据里一字不差的提示词和成本。"""
        for split in ("test_seen", "test_unseen"):
            for r in self.rows[split]:
                s = state_from_dict(r["state"])
                messages, _ = render_compact(s)
                self.assertEqual(messages[1]["content"], r["user"])
                costs = one_step_costs(s)
                for a in range(len(costs)):
                    self.assertAlmostEqual(costs[a], r["costs"][a], places=3)

    def test_golden_texts(self):
        """meta.json 里给 Colab 核对的样例 = deploy/prompts.py 拼出来的全文（论文格式是 JSON 拼法那种）。"""
        for i, golden in enumerate(self.meta["golden"]["compact"]):
            r = self.rows["train"][i]
            msgs = [{"role": "system", "content": COMPACT_SYSTEM}, {"role": "user", "content": r["user"]}]
            self.assertEqual(chatml(msgs), golden)
        for i, golden in enumerate(self.meta["golden"]["paper"]):
            s = state_from_dict(self.rows["test_seen"][i]["state"])
            self.assertEqual(chatml(paper_messages_json(s)), golden)

    def test_deterministic(self):
        """同样的参数再生成一次，每个文件的校验值都一样（Ethan 在 Mac 上生成的和这里的能对上）。"""
        with tempfile.TemporaryDirectory() as d:
            meta2 = make_data(d)
        for split, info in self.meta["files"].items():
            self.assertEqual(meta2["files"][split]["md5"], info["md5"], split)


class TestOpenLoopMetrics(unittest.TestCase):
    def test_effective_cost(self):
        """论文式 26：没超截止期按原值；超了加 ρ = 10；非法动作记 20。"""
        costs = [3.0, 12.0]
        self.assertEqual(effective_cost(costs, 0), 3.0)
        self.assertEqual(effective_cost(costs, 1), 22.0)
        self.assertEqual(effective_cost(costs, -1), 20.0)
        self.assertEqual(effective_cost(costs, 5), 20.0)

    def test_jain(self):
        self.assertAlmostEqual(jain([5, 5, 5]), 1.0)
        self.assertAlmostEqual(jain([9, 0, 0]), 1.0 / 3.0)

    def test_score(self):
        """两条状态：一条选对，一条非法。没答的状态（第 3 条）不计入。"""
        rows = [{"_idx": 0, "costs": [2.0, 1.0], "best": 1, "n_actions": 2, "servers": 1},
                {"_idx": 1, "costs": [2.0, 1.0], "best": 1, "n_actions": 2, "servers": 1},
                {"_idx": 2, "costs": [2.0, 1.0], "best": 1, "n_actions": 2, "servers": 1}]
        preds = {("t", 0): 1, ("t", 1): -1}
        m = score(rows, preds, "t")
        self.assertEqual(m["n"], 2)
        self.assertAlmostEqual(m["acc_pct"], 50.0)
        self.assertAlmostEqual(m["invalid_pct"], 50.0)
        self.assertAlmostEqual(m["al"], (1.0 + 20.0) / 2)
        self.assertAlmostEqual(m["pr_pct"], 100.0 * 1.0 / 10.5)
        self.assertIsNone(score(rows, {}, "t"))

    def test_score_local(self):
        """本地召回 / 精确 / 卸载准确率：4 条，2 条本地最优。
        第 0 条本地最优选了本地（命中），第 1 条本地最优选了服务器（漏掉），
        第 2 条服务器最优选了本地（误选），第 3 条服务器最优选对。"""
        rows = []
        for i, best in enumerate((0, 0, 1, 1)):
            costs = [1.0, 2.0] if best == 0 else [2.0, 1.0]
            rows.append({"_idx": i, "costs": costs, "best": best, "n_actions": 2, "servers": 1})
        preds = {("t", 0): 0, ("t", 1): 1, ("t", 2): 0, ("t", 3): 1}
        m = score(rows, preds, "t")
        self.assertAlmostEqual(m["best_local_pct"], 50.0)
        self.assertAlmostEqual(m["local_recall_pct"], 50.0)
        self.assertAlmostEqual(m["local_precision_pct"], 50.0)
        self.assertAlmostEqual(m["offload_acc_pct"], 50.0)
        # 从不选本地：召回 0，精确没有分母
        m = score(rows, {("t", i): 1 for i in range(4)}, "t")
        self.assertAlmostEqual(m["local_recall_pct"], 0.0)
        self.assertTrue(np.isnan(m["local_precision_pct"]))
        self.assertAlmostEqual(m["offload_acc_pct"], 100.0)


class TestGgufReference(unittest.TestCase):
    """expected_200.jsonl 里被别的行覆盖的参考答案，要认出来并用 preds.csv 补回。"""

    def rows(self):
        good = {"split": "test_seen", "idx": 7, "n_actions": 4, "pt_action": 1, "pt_margin": 2.0,
                "pt_letter_probs": [0.1, 0.8, 0.05, 0.05]}
        bad = {"split": "test_unseen", "idx": 7, "n_actions": 5, "pt_action": 3, "pt_margin": 2.0,
               "pt_letter_probs": [0.1, 0.1, 0.1, 0.7]}
        return [good, bad]

    def test_repair(self):
        rows = self.rows()
        preds = {("test_seen", 7): (1, 2.0), ("test_unseen", 7): (4, 0.5)}
        n_fixed, n_same = repair_reference(rows, preds)
        self.assertEqual(n_fixed, 1)
        self.assertEqual(n_same, 1)
        self.assertEqual([r["ref"] for r in rows], ["merged", "lora"])
        self.assertEqual(rows[1]["pt_action"], 4)
        self.assertEqual(rows[1]["pt_margin"], 0.5)
        self.assertEqual(rows[0]["pt_action"], 1)

    def test_missing_preds(self):
        with self.assertRaises(SystemExit):
            repair_reference(self.rows(), {})


if __name__ == "__main__":
    unittest.main()
