"""阶段 4 GRPO 奖励和数据的测试。

原作者数据不在仓库里：找不到时跳过数据那一组测试（Mac 上默认能找到）。
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from eval.open_loop import effective_cost as eval_effective_cost
from sim.friend_data import DEFAULT_DIR
from train.grpo_reward import (BAD_SERVER, NONE, OK, effective_cost, group_advantages, parse, parse_letter,
                               reward_fixed, reward_fixed_action, reward_repo)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRIEND_DIR = os.environ.get("COMLLM_FRIEND_DATA", DEFAULT_DIR)


class TestReward(unittest.TestCase):
    COSTS = [7.98, 3.29, 3.41, 6.09]       # 本地、服务器 1、2、3

    def test_parse(self):
        self.assertEqual(parse("本地执行", 3), (OK, 0))
        self.assertEqual(parse("卸载到边缘服务器2", 3), (OK, 2))
        self.assertEqual(parse("我选卸载到边缘服务器3。", 3), (OK, 3))
        self.assertEqual(parse("卸载到边缘服务器4", 3), (BAD_SERVER, -1))
        self.assertEqual(parse("卸载到边缘服务器0", 3), (BAD_SERVER, -1))
        self.assertEqual(parse("卸载到服务器2", 3), (NONE, -1))
        self.assertEqual(parse("", 3), (NONE, -1))

    def test_parse_letter(self):
        # 压缩格式：4 个动作时 A–D 合法；E 是"服务器 4"，不存在
        self.assertEqual(parse_letter("A", 4), (OK, 0))
        self.assertEqual(parse_letter("D", 4), (OK, 3))
        self.assertEqual(parse_letter(" C", 4), (OK, 2))
        self.assertEqual(parse_letter("E", 4), (BAD_SERVER, -1))
        self.assertEqual(parse_letter("L", 12), (OK, 11))
        self.assertEqual(parse_letter("a", 4), (NONE, -1))
        self.assertEqual(parse_letter("AB", 4), (NONE, -1))
        self.assertEqual(parse_letter("", 4), (NONE, -1))

    def test_fixed_letter_same_as_text(self):
        # 同一个动作，两种格式的 fixed 奖励一样
        for letter, text in (("A", "本地执行"), ("C", "卸载到边缘服务器2"), ("F", "卸载到边缘服务器5")):
            kind, a = parse_letter(letter, len(self.COSTS))
            self.assertAlmostEqual(reward_fixed_action(kind, a, self.COSTS, 10), reward_fixed(text, self.COSTS, 10))

    def test_effective_cost_same_as_eval(self):
        for c in (3.0, 9.99, 10.0, 10.01, 25.0):
            self.assertEqual(effective_cost(c, 10.0), eval_effective_cost([c], 0))

    def test_fixed_valid(self):
        self.assertAlmostEqual(reward_fixed("卸载到边缘服务器1", self.COSTS, 10), -3.29)
        self.assertAlmostEqual(reward_fixed("本地执行", self.COSTS, 10), -7.98)

    def test_fixed_invalid_is_worst(self):
        costs = [12.0, 3.0, 25.0]           # 本地超时 → 22；服务器 2 超时 → 35
        r_bad = reward_fixed("卸载到边缘服务器5", costs, 10)
        r_none = reward_fixed("不知道", costs, 10)
        self.assertAlmostEqual(r_bad, -40.0)
        self.assertAlmostEqual(r_none, -40.0)
        for a, text in enumerate(("本地执行", "卸载到边缘服务器1", "卸载到边缘服务器2")):
            self.assertLess(r_bad, reward_fixed(text, costs, 10), a)

    def test_repo_rewards_hallucination(self):
        # repo：编一个不存在的服务器得 −4，比选合法的本地（−7.98 + 1）还高，这就是它的漏洞
        self.assertAlmostEqual(reward_repo("卸载到边缘服务器9", self.COSTS), -4.0)
        self.assertAlmostEqual(reward_repo("看不懂", self.COSTS), -5.0)
        self.assertAlmostEqual(reward_repo("本地执行", self.COSTS), -6.98)
        self.assertGreater(reward_repo("卸载到边缘服务器9", self.COSTS), reward_repo("本地执行", self.COSTS))
        self.assertLess(reward_fixed("卸载到边缘服务器9", self.COSTS, 10), reward_fixed("本地执行", self.COSTS, 10))

    def test_repo_valid_differs_by_constant(self):
        # 合法动作之间，repo 和 fixed 只差一个常数 +1，组内归一化后优势完全一样
        a1, _ = group_advantages([reward_fixed(t, self.COSTS, 10) for t in ("本地执行", "卸载到边缘服务器1")], 2)
        a2, _ = group_advantages([reward_repo(t, self.COSTS) for t in ("本地执行", "卸载到边缘服务器1")], 2)
        for x, y in zip(a1, a2):
            self.assertAlmostEqual(x, y)

    def test_group_advantages(self):
        adv, zero = group_advantages([1.0, 3.0, 2.0, 2.0], 2)
        self.assertEqual(zero, 1)
        self.assertAlmostEqual(adv[0], -1.0, places=3)
        self.assertAlmostEqual(adv[1], 1.0, places=3)
        self.assertEqual(adv[2], 0.0)
        self.assertEqual(adv[3], 0.0)
        adv, zero = group_advantages([-3.0, -4.0, -5.0, -4.0], 4)
        self.assertEqual(zero, 0)
        self.assertAlmostEqual(sum(adv), 0.0, places=9)
        self.assertGreater(adv[0], adv[1])


@unittest.skipUnless(os.path.isdir(FRIEND_DIR), "找不到原作者数据")
class TestGrpoData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        subprocess.run([sys.executable, "-m", "train.make_grpo_data", "--out", cls.tmp.name, "--friend-dir", FRIEND_DIR],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        with open(os.path.join(cls.tmp.name, "meta.json"), encoding="utf-8") as f:
            cls.meta = json.load(f)
        cls.rows = {}
        for name in ("grpo_train", "grpo_val"):
            cls.rows[name] = []
            with open(os.path.join(cls.tmp.name, name + ".jsonl"), encoding="utf-8") as f:
                for line in f:
                    cls.rows[name].append(json.loads(line))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_sizes(self):
        self.assertEqual(len(self.rows["grpo_val"]), 100)
        self.assertGreater(len(self.rows["grpo_train"]), 700)

    def test_no_overlap(self):
        keys = {}
        for name in self.rows:
            keys[name] = set()
            for r in self.rows[name]:
                keys[name].add(json.dumps(r["paper_input"], ensure_ascii=False, sort_keys=True))
        self.assertEqual(len(keys["grpo_train"] & keys["grpo_val"]), 0)
        from sim.friend_data import FILES, load_all
        test_keys = set()
        for fname, e in load_all(FRIEND_DIR):
            if fname == FILES[2]:
                test_keys.add(json.dumps(e["input"], ensure_ascii=False, sort_keys=True))
        for name in keys:
            self.assertEqual(len(keys[name] & test_keys), 0, name)

    def test_labels_are_one_step_best(self):
        self.assertEqual(self.meta["stats"]["friend_label_equals_best_pct"], 100.0)
        for r in self.rows["grpo_val"]:
            self.assertEqual(len(r["costs"]), r["n_actions"])
            self.assertEqual(r["n_actions"], r["servers"] + 1)


if __name__ == "__main__":
    unittest.main()
