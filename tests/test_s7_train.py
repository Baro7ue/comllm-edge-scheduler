"""CPU-only S7 orchestration tests; no model download or GPU is used."""
import csv
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest.mock import patch

TRAINER_SPEC = importlib.util.find_spec("train.s7_train")
if TRAINER_SPEC:
    from train import s7_train as S
else:
    S = None


class FakeEngine:
    """A stateful optimizer substitute at the expensive model boundary."""
    def __init__(self, config, instruction):
        self.value = random.random()
        self.updates = []

    def prepare(self, rows):
        return list(rows), 0

    def step(self, batch, lambda_lacs, step, total_steps):
        # Both optimizer momentum and random sampling must survive a restart.
        self.value = self.value * .8 + random.random() + lambda_lacs
        self.updates.append([step, [r["_idx"] for r in batch], self.value])
        return {"loss": self.value, "reward_mean": -self.value}

    def answers(self, rows):
        return ["我选卸载到边缘服务器1。" for _ in rows]

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.value, self.updates), f)

    def load_state(self, path):
        with open(path, "rb") as f:
            self.value, self.updates = pickle.load(f)

    def export_adapter(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_config.json").write_text("{}")
        (Path(path) / "fake_state.json").write_text(json.dumps(self.updates))

    def close(self):
        pass


class S7TrainTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(S, "S7 training runner has not been implemented")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "config.json").write_text('{"hidden_size":3584}')
        (self.base / "model.safetensors").write_bytes(b"identity fixture")
        self.s3, self.s4 = self.root / "s3", self.root / "s4"
        self.s3.mkdir()
        self.s4.mkdir()
        self.rows = [self.row(i) for i in range(6)]
        self.write_rows(self.s4 / "grpo_train.jsonl", self.rows)
        self.write_rows(self.s4 / "grpo_val.jsonl", [self.row(6)])
        for name, i in (("test_seen", 10), ("test_unseen", 11), ("friend_clean", 12)):
            self.write_rows(self.s3 / (name + ".jsonl"), [self.row(i)])
        (self.s3 / "meta.json").write_text(json.dumps({"paper_instruction": "决定任务去向"}))
        self.cfg = S.TrainConfig(base=str(self.base), s3_data=str(self.s3), s4_data=str(self.s4),
                                out=str(self.root / "out"), group="lacs", mode="full",
                                prompts_per_step=2, checkpoint_every=1, eval_batch=2,
                                max_minutes=120, max_steps=3)

    @staticmethod
    def row(i):
        return {"paper_input": {"任务大小": str(i), "本地设备": "CPU频率2.5"},
                "user": "state-" + str(i), "servers": 3, "n_actions": 4,
                "costs": [4., 2., 3., 5.], "best": 1,
                "state": {"deadline": 10, "task_size": i}}

    @staticmethod
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    def test_heldout_states_removed_even_when_input_dict_order_differs(self):
        duplicate = dict(self.rows[0])
        duplicate["paper_input"] = dict(reversed(list(duplicate["paper_input"].items())))
        self.write_rows(self.s3 / "test_seen.jsonl", [duplicate])
        data = S.load_data(self.cfg)
        self.assertEqual([r["_idx"] for r in data["train"]], [1, 2, 3, 4, 5])
        self.assertEqual(data["audit"]["train_heldout_removed"], 1)
        self.assertEqual(data["audit"]["remaining_overlap"], 0)

    def test_compact_rounding_collision_does_not_remove_distinct_paper_states(self):
        self.rows[1]["user"] = self.rows[0]["user"]
        self.write_rows(self.s4 / "grpo_train.jsonl", self.rows)
        data = S.load_data(self.cfg)
        self.assertEqual(len(data["train"]), 6)
        self.assertEqual(data["audit"]["train_duplicates_removed"], 0)

    def test_resume_replays_optimizer_and_rng_exactly(self):
        S.run(self.cfg, engine_factory=FakeEngine)
        whole = json.loads((self.root / "out/lacs/lora/fake_state.json").read_text())
        resumed = S.dataclasses.replace(self.cfg, out=str(self.root / "restart"))
        ticks = iter([0., 0., 0., 121., 122., 123., 124., 125.])
        short = S.dataclasses.replace(resumed, max_minutes=2)
        stopped = S.run(short, engine_factory=FakeEngine, clock=lambda: next(ticks, 125.))
        self.assertEqual(stopped["status"], "time_limit")
        self.assertGreater(stopped["step"], 0)
        self.assertLess(stopped["step"], 3)
        result = S.run(S.dataclasses.replace(resumed, resume=True), engine_factory=FakeEngine)
        self.assertEqual(result["status"], "complete")
        actual = json.loads((self.root / "restart/lacs/lora/fake_state.json").read_text())
        self.assertEqual(actual, whole)

    def test_resume_rejects_changed_base_weights(self):
        S.run(self.cfg, engine_factory=FakeEngine)
        (self.base / "model.safetensors").write_bytes(b"different base")
        with self.assertRaisesRegex(ValueError, "identity|身份"):
            S.run(S.dataclasses.replace(self.cfg, resume=True), engine_factory=FakeEngine)

    def test_paired_groups_reject_different_seed_and_keep_outputs_separate(self):
        S.run(self.cfg, engine_factory=FakeEngine)
        other = S.dataclasses.replace(self.cfg, group="nolacs", seed=17)
        with self.assertRaisesRegex(ValueError, "pair|配对"):
            S.run(other, engine_factory=FakeEngine)
        result = S.run(S.dataclasses.replace(other, seed=self.cfg.seed), engine_factory=FakeEngine)
        self.assertEqual(result["status"], "complete")
        self.assertTrue((self.root / "out/lacs/lora/adapter_config.json").is_file())
        self.assertTrue((self.root / "out/nolacs/lora/adapter_config.json").is_file())

    def test_predictions_cover_all_test_rows_and_record_parser_difference(self):
        S.run(self.cfg, engine_factory=FakeEngine)
        with open(self.root / "out/lacs/preds.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({r["split"] for r in rows}, {"test_seen", "test_unseen", "friend_clean"})
        self.assertEqual([(r["pred"], r["pred_strict"]) for r in rows], [("1", "-1")] * 3)
        S.run(S.dataclasses.replace(self.cfg, resume=True), engine_factory=FakeEngine)
        with open(self.root / "out/lacs/preds.csv", newline="") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 3)

    def test_smoke_cannot_overwrite_formal_adapter(self):
        S.run(S.dataclasses.replace(self.cfg, mode="smoke"), engine_factory=FakeEngine)
        self.assertTrue((self.root / "out/smoke/lacs/lora/adapter_config.json").is_file())
        self.assertFalse((self.root / "out/lacs").exists())

    def test_fresh_run_refuses_existing_group(self):
        S.run(self.cfg, engine_factory=FakeEngine)
        with self.assertRaisesRegex(ValueError, "resume|恢复"):
            S.run(self.cfg, engine_factory=FakeEngine)

    def test_prompt_has_same_paper_structure_for_both_groups(self):
        prompt = S.paper_messages(self.rows[0], "决定任务去向")
        self.assertEqual(prompt[1]["content"], "决定任务去向\n\n**输入数据**：\n任务大小: 0\n本地设备: CPU频率2.5")
        self.assertEqual(S.parse_prediction("卸载到边缘服务器4", 3), -1)

    def test_default_effective_prompt_batch_covers_remainder(self):
        cfg = S.TrainConfig(base="unused", s3_data="unused", s4_data="unused", out="unused", mode="full")
        batches = S.training_batches([self.row(i) for i in range(830)], cfg)
        self.assertEqual(len(batches), 208)
        self.assertEqual(len(batches[-1]), 2)
        self.assertEqual(sum(map(len, batches)), 830)

    def test_interrupted_checkpoint_publish_retains_previous_checkpoint(self):
        folder = self.root / "atomic"
        folder.mkdir()
        engine = FakeEngine(self.cfg, "unused")
        progress = {"step": 0, "target_steps": 3}
        S.save_checkpoint(folder, engine, progress)
        old = folder / json.loads((folder / "latest.json").read_text())["checkpoint"]
        replace = S.os.replace
        def fail_directory_publish(src, dst):
            if Path(src).is_dir():
                raise OSError("simulated disconnected Drive")
            return replace(src, dst)
        with patch.object(S.os, "replace", side_effect=fail_directory_publish):
            with self.assertRaises(OSError):
                S.save_checkpoint(folder, engine, progress)
        self.assertTrue(old.is_dir(), "a failed publish removed the previously usable checkpoint")

    def test_validation_uses_configured_batches(self):
        class BoundedEngine(FakeEngine):
            def answers(engine, rows):
                if len(rows) > 2:
                    raise AssertionError("oversized GPU evaluation batch")
                return super().answers(rows)
        self.write_rows(self.s4 / "grpo_val.jsonl", [self.row(i) for i in range(20, 25)])
        result = S.run(S.dataclasses.replace(self.cfg, eval_every=1), engine_factory=BoundedEngine)
        self.assertEqual(result["status"], "complete")
        vals = [json.loads(x) for x in (self.root / "out/lacs/val_log.jsonl").read_text().splitlines()]
        self.assertEqual([x["n"] for x in vals], [5, 5, 5])

    def test_validate_only_does_not_create_outputs_or_import_torch(self):
        import sys
        before = "torch" in sys.modules
        args = ["--base", str(self.base), "--s3-data", str(self.s3), "--s4-data", str(self.s4),
                "--out", str(self.root / "preflight"), "--group", "lacs", "--validate-only"]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(S.main(args), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "validated")
        self.assertFalse((self.root / "preflight").exists())
        self.assertEqual("torch" in sys.modules, before)

    def test_resume_without_checkpoint_starts_fresh(self):
        result = S.run(S.dataclasses.replace(self.cfg, resume=True), engine_factory=FakeEngine)
        self.assertEqual((result["status"], result["step"]), ("complete", 3))

    def test_evaluation_resume_does_not_repeat_training_or_predictions(self):
        class Clock:
            value = 0.
            def __call__(self):
                return self.value
        clock = Clock()
        class SlowEvalEngine(FakeEngine):
            def answers(engine, rows):
                clock.value = 121.
                return super().answers(rows)
        paused = S.run(S.dataclasses.replace(self.cfg, max_minutes=2), engine_factory=SlowEvalEngine, clock=clock)
        self.assertEqual((paused["status"], paused["step"], paused["training_complete"]), ("time_limit", 3, True))
        result = S.run(S.dataclasses.replace(self.cfg, resume=True), engine_factory=FakeEngine)
        self.assertEqual(result["status"], "complete")
        updates = json.loads((self.root / "out/lacs/lora/fake_state.json").read_text())
        self.assertEqual(len(updates), 3)
        with open(self.root / "out/lacs/preds.csv", newline="") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 3)

    def test_resume_recovers_only_last_truncated_json_log_record(self):
        S.run(self.cfg, engine_factory=FakeEngine)
        log = self.root / "out/lacs/train_log.jsonl"
        with log.open("a") as f:
            f.write('{"step":4,"loss":')
        result = S.run(S.dataclasses.replace(self.cfg, resume=True), engine_factory=FakeEngine)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(log.read_text().splitlines()), 3)
        log.write_text('{"step":1}\nBROKEN\n{"step":2}\n')
        with self.assertRaises(ValueError):
            S.run(S.dataclasses.replace(self.cfg, resume=True), engine_factory=FakeEngine)

    def test_resume_reanswers_only_incomplete_final_prediction_record(self):
        S.run(self.cfg, engine_factory=FakeEngine)
        path = self.root / "out/lacs/preds.csv"
        lines = path.read_text().splitlines(keepends=True)
        path.write_text("".join(lines[:-1]) + 'friend_clean,0,1,"unfinished')
        result = S.run(S.dataclasses.replace(self.cfg, resume=True), engine_factory=FakeEngine)
        self.assertEqual(result["status"], "complete")
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]["pred_strict"], "-1")


if __name__ == "__main__":
    unittest.main()
