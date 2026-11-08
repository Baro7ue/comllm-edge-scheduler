"""S7 reward contracts, with numeric fixtures run against upstream 17adedef.

The fixture values come from the four pinned upstream files, using
random.seed(2026); they do not use S4's fixed reward or the 41-point grid.
"""
import copy
import json
import random
import unittest

from train import s7_reward as R


RAW = {
    "任务大小": "2 Mbits", "最大延迟": "10",
    "本地CPU队列等待时间": "1", "网络队列等待时间": "0.5",
    "网络上传速率": "1 Mbits/槽",
    "本地设备": "CPU频率 2.97 GHz, 每比特周期数 297",
    "边缘服务器1": "CPU频率 5.94 GHz, 活跃任务数1，服务器总队列长度2 Mbits",
    "边缘服务器3": "CPU频率 11.88 GHz, 活跃任务数0，服务器总队列长度0 Mbits",
}


class TestS7Reward(unittest.TestCase):
    def test_upstream_server_parser_takes_priority_over_local(self):
        for text, want in (("不是本地，选择服务器2", 2), ("LOCAL", 0),
                           ("server 3", 3), ("服务器0", None), ("不知道", None)):
            self.assertEqual(R.parse_action(text), want)

    def test_comllm_format_reward_does_not_reward_number_only(self):
        for text, want in (("本地执行", 1.5), ("服务器9", -0.5),
                           ("看不懂", -1.9), ("", -2.0), ("1", -2.1),
                           ('{"action_id":1}', -2.0)):
            self.assertAlmostEqual(R.format_reward(text), want)

    def test_upstream_current_costs_use_real_server_ids(self):
        for action, want in ((0, 3.0), (1, 6.5), (3, 3.0), (2, 5.0), (99, 5.0)):
            self.assertAlmostEqual(R.calculate_cost(action, RAW), want)

    def test_upstream_monte_carlo_future_regression(self):
        for action, want in ((0, 2.6945486274735706), (1, 2.692158882806183),
                             (3, 2.7556389019788567), (9, 2.692158882806183)):
            self.assertAlmostEqual(R.predict_future(action, RAW, rng=random.Random(2026)), want)

    def test_local_action_also_changes_future_local_queue(self):
        raw = {k: v for k, v in RAW.items() if not k.startswith("边缘服务器")}
        self.assertAlmostEqual(R.predict_future(0, raw, rng=random.Random(2026)), 4.755638901978856)

    def test_pinned_comllm_total_reward_examples(self):
        for text, want in (("本地执行", -1.9466916705936568),
                           ("卸载到边缘服务器1", -4.746118131873485),
                           ("卸载到边缘服务器3", -1.9613533364749254),
                           ("服务器9", -3.946118131873484),
                           ("看不懂", -8.38), ("", -8.4)):
            self.assertAlmostEqual(R.reward(text, {"paper_input": RAW}, rng=random.Random(2026)), want)

    def test_only_lambda_changes_between_controlled_arms(self):
        a = R.reward_details("本地执行", RAW, 0.0, rng=random.Random(2026))
        b = R.reward_details("本地执行", RAW, 0.3, rng=random.Random(2026))
        self.assertAlmostEqual(a["reward"], -1.3)
        self.assertEqual(a["future_cost"], b["future_cost"])
        self.assertAlmostEqual(b["reward"] - a["reward"], -0.24 * 2.6945486274735706)
        json.dumps(b, allow_nan=False)

    def test_current_cost_at_ten_disables_future_and_total_penalty_is_strict(self):
        d = R.reward_details("本地执行", {"costs": [10.0], "future_cost": 100.0}, 0.3)
        self.assertEqual(d["future_cost"], 0.0)
        self.assertAlmostEqual(d["reward"], -6.9)

    def test_future_can_cross_total_cost_penalty_threshold(self):
        d = R.reward_details("本地执行", {"costs": [9.0], "future_cost": 10.0}, 0.3)
        self.assertAlmostEqual(d["total_cost"], 12.0)
        self.assertAlmostEqual(d["task_reward"], -21.0)
        self.assertAlmostEqual(d["reward"], -16.5)

    def test_current_cost_over_ten_skips_future(self):
        d = R.reward_details("本地执行", {"costs": [11.0], "future_cost": 100.0}, 0.3)
        self.assertEqual(d["future_cost"], 0.0)
        self.assertAlmostEqual(d["reward"], -15.7)

    def test_rng_state_can_be_restored_without_global_random_side_effect(self):
        rng = random.Random(42)
        saved = rng.getstate()
        global_saved = random.getstate()
        a = R.reward("本地执行", RAW, rng=rng)
        rng.setstate(saved)
        self.assertEqual(a, R.reward("本地执行", RAW, rng=rng))
        self.assertEqual(random.getstate(), global_saved)

    def test_per_prompt_sampling_survives_order_and_json_roundtrip(self):
        row = {"paper_input": RAW, "costs": [999.0]}
        a = R.reward("本地执行", row, seed=12, sample_key=[7, "prompt-2"])
        R.reward("本地执行", RAW, seed=999, sample_key="another")
        b = R.reward("本地执行", json.loads(json.dumps(row)), seed=12, sample_key=[7, "prompt-2"])
        self.assertEqual(a, b)
        self.assertNotEqual(a, R.reward("本地执行", row, seed=12, sample_key=[8, "prompt-2"]))

    def test_row_containers_preserve_raw_input_over_rounded_costs(self):
        want = -1.9466916705936568
        for key in ("paper_input", "friend_input", "raw", "input"):
            row = {key: RAW, "costs": [999.0]}
            original = copy.deepcopy(row)
            self.assertAlmostEqual(R.reward("本地执行", row, rng=random.Random(2026)), want)
            self.assertEqual(row, original)

    def test_state_dict_and_state_objects_use_same_cost_units(self):
        state = {"d": 2, "local_wait": 1, "net_wait": .5, "rate": 1,
                 "deadline": 10, "local_ghz": 2.97,
                 "servers": [{"ghz": 5.94, "active": 1, "backlog": 2}]}
        self.assertAlmostEqual(R.calculate_cost(1, {"state_dict": state}), 6.5)
        self.assertAlmostEqual(R.calculate_cost(0, {"state": state}), 3.0)
        from types import SimpleNamespace
        obj = SimpleNamespace(**{**state, "servers": [SimpleNamespace(**state["servers"][0])]})
        self.assertAlmostEqual(R.calculate_cost(1, obj), 6.5)

    def test_missing_future_state_is_not_silently_treated_as_zero(self):
        with self.assertRaises(ValueError):
            R.reward("本地执行", {"costs": [3.0]}, 0.3)

    def test_invalid_lookahead_and_lambda_fail_before_training(self):
        with self.assertRaises(ValueError):
            R.predict_future(0, RAW, rng=random.Random(1), lookahead_steps=0)
        with self.assertRaises(ValueError):
            R.reward("本地执行", RAW, float("nan"))


if __name__ == "__main__":
    unittest.main()
