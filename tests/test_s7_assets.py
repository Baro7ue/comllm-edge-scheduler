"""Small real fixtures catch contaminated datasets and incomplete weight uploads."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile


class S7AssetsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("train.s7_assets"), "asset implementation is missing")
        self.assets = importlib.import_module("train.s7_assets")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def model(self):
        d = self.root / "model"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"hidden_size": 3584, "torch_dtype": "float16"}))
        header = {"a": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
                  "b": {"dtype": "F16", "shape": [1], "data_offsets": [4, 6]}}
        raw = json.dumps(header).encode()
        shard = d / "model-00001-of-00001.safetensors"
        shard.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 6)
        index = {"metadata": {"total_size": 6, "total_parameters": 3},
                 "weight_map": {"a": shard.name, "b": shard.name}}
        (d / "model.safetensors.index.json").write_text(json.dumps(index))
        (d / "tokenizer_config.json").write_text('{"chat_template":"test"}')
        return d, shard

    def data(self):
        d = self.root / "source"
        paths = ["data/s4/grpo_train.jsonl", "data/s4/grpo_val.jsonl", "data/s3/friend_clean.jsonl",
                 "data/s3/test_seen.jsonl", "data/s3/test_unseen.jsonl"]
        for i, rel in enumerate(paths):
            row = {"split": Path(rel).stem, "n_actions": 2, "servers": 1, "best": 0,
                   "costs": [1.0, 2.0], "costs_compact": [1.0, 2.0], "ps_costs": [1.0, 2.0],
                   "state": {"d": i + 1, "servers": [{"ghz": 1}]}, "paper_input": {"任务大小": str(i + 1)}}
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(row) + "\n")
        (d / "data/s4/meta.json").write_text('{"source":"original-cleaning-record"}')
        return d

    def test_header_validation_reports_real_payload_without_claiming_full_sha(self):
        d, _ = self.model()
        got = self.assets.model_manifest(d)
        self.assertEqual(got["tensor_count"], 2)
        self.assertEqual(got["total_parameters"], 3)
        self.assertEqual(got["total_payload_bytes"], 6)
        self.assertEqual(got["dtypes"], {"F16": 2})
        self.assertNotIn("sha256", got["shards"][0])
        self.assertEqual(len(got["shards"][0]["header_sha256"]), 64)

    def test_truncated_payload_is_rejected_without_loading_tensors(self):
        d, p = self.model()
        p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "payload|offset|size"):
            self.assets.model_manifest(d)

    def test_opt_in_full_sha_covers_the_payload_as_well_as_header(self):
        d, p = self.model()
        got = self.assets.model_manifest(d, full_sha256=True)
        self.assertEqual(got["shards"][0]["sha256"], hashlib.sha256(p.read_bytes()).hexdigest())

    def test_repack_can_reuse_a_named_previous_payload_hash_after_header_checks(self):
        d, p = self.model()
        previous = self.root / "previous.json"
        previous.write_text(json.dumps(self.assets.model_manifest(d, full_sha256=True)))
        got = self.assets.model_manifest(d, previous_manifest=previous)
        self.assertEqual(got["shards"][0]["sha256"], hashlib.sha256(p.read_bytes()).hexdigest())
        self.assertIn("payload_hash_source", got)
        old = json.loads(previous.read_text())
        old["shards"][0]["bytes"] += 1
        previous.write_text(json.dumps(old))
        with self.assertRaisesRegex(ValueError, "previous|manifest|identity"):
            self.assets.model_manifest(d, previous_manifest=previous)

    def test_index_cannot_silently_omit_header_tensors(self):
        d, _ = self.model()
        p = d / "model.safetensors.index.json"
        idx = json.loads(p.read_text())
        del idx["weight_map"]["b"]
        p.write_text(json.dumps(idx))
        with self.assertRaisesRegex(ValueError, "index|tensor"):
            self.assets.model_manifest(d)

    def test_shape_byte_disagreement_is_rejected(self):
        d, p = self.model()
        with p.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h["a"]["shape"] = [3]
        raw = json.dumps(h).encode()
        p.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 6)
        with self.assertRaisesRegex(ValueError, "shape|bytes"):
            self.assets.model_manifest(d)

    def test_dataset_audit_rejects_train_test_overlap_even_when_split_differs(self):
        d = self.data()
        p = d / "data/s4/grpo_train.jsonl"
        train = json.loads(p.read_text())
        train["split"] = "friend_clean"
        (d / "data/s3/friend_clean.jsonl").write_text(json.dumps(train) + "\n")
        with self.assertRaisesRegex(ValueError, "overlap|重复"):
            self.assets.audit_data(d)

    def test_pack_contains_current_code_clean_data_and_provenance_but_no_weights(self):
        source = self.data()
        model, _ = self.model()
        repo = self.root / "code"
        (repo / "train").mkdir(parents=True)
        (repo / "train/new.py").write_text("VALUE = 'current-worktree'\n")
        (repo / "train/private.bin").write_bytes(b"not-code")
        out = self.root / "assets.zip"
        got = self.assets.pack_assets(repo, source, model, out)
        with zipfile.ZipFile(out) as z:
            self.assertIn("train/new.py", z.namelist())
            self.assertIn("data/s4/meta.json", z.namelist())
            self.assertIn("model_metadata/tokenizer_config.json", z.namelist())
            self.assertIn("model_manifest.json", z.namelist())
            self.assertFalse(any(n.endswith((".safetensors", ".bin")) for n in z.namelist()))
            self.assertEqual(z.read("train/new.py"), b"VALUE = 'current-worktree'\n")
            manifest = json.loads(z.read("s7_assets_manifest.json"))
            self.assertEqual(manifest["data_audit"]["files"]["data/s4/grpo_train.jsonl"]["rows"], 1)
            self.assertIn("code_source", manifest)
        self.assertEqual(got["archive_bytes"], out.stat().st_size)


if __name__ == "__main__":
    unittest.main()
