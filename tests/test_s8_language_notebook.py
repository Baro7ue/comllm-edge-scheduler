"""Checks for train/S8_语言能力检查.ipynb, the language-ability check of the students.

The notebook needs a GPU, so it cannot run here. These tests cover what can be checked without one:
1. every code cell compiles and the notebook is saved without outputs
2. the pure-Python cell (built-in data and helpers): prompt layout equals the deployed one, the scheduling
   examples match the simulator, the multiple-choice answers are balanced, responses are classified correctly
3. a dry run of cells 2-9 with small fakes for torch, transformers, peft, datasets and Google Drive, to catch
   wrong indices, keys and paths in the model loop and the report
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from deploy.prompts import chatml
from deploy.wire import state_from_dict
from sim.cost import best_action, one_step_costs
from sim.render import COMPACT_SYSTEM, LETTERS, render_compact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(ROOT, "train", "S8_语言能力检查.ipynb")
VOCAB = 400
STUDENTS = {"main_0.5b": "0.5b_soft_T0.3", "soft_1.5b": "1.5b_soft_T0.3", "hard_1.5b": "1.5b_hard",
            "ps_0.5b": "0.5b_soft_T0.3_ps", "ps_soft_1.5b": "1.5b_soft_T0.3_ps", "ps_hard_1.5b": "1.5b_hard_ps"}


def load_cells():
    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)
    return nb, ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def run_quiet(src, ns):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, "cell", "exec"), ns)
    return buf.getvalue()


# ===== fakes for the dry run =====
class FakeTensor:
    def __init__(self, a):
        self.a = np.asarray(a)

    def float(self):
        return FakeTensor(self.a.astype(np.float64))

    def __getitem__(self, k):
        return FakeTensor(self.a[k])

    def item(self):
        return self.a.item()


def fake_torch():
    t = types.ModuleType("torch")
    t.__version__ = "fake"
    t.long, t.bfloat16, t.float32 = "torch.long", "torch.bfloat16", "torch.float32"
    t.tensor = lambda data, dtype=None, device=None: FakeTensor(data)
    t.ones_like = lambda x: FakeTensor(np.ones_like(x.a))
    t.no_grad = lambda: (lambda f: f)

    def log_softmax(x, dim=-1):
        a = x.a - x.a.max(axis=dim, keepdims=True)
        return FakeTensor(a - np.log(np.exp(a).sum(axis=dim, keepdims=True)))

    t.log_softmax = log_softmax
    t.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda: "fake GPU",
                                   is_bf16_supported=lambda: True, empty_cache=lambda: None)
    return t


class FakeTokenizer:
    pad_token_id = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % VOCAB for c in text]}

    def convert_tokens_to_ids(self, token):
        return 1

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids.a if int(i) != 1)


class FakeModel:
    """Logits depend only on the input, so results are repeatable. A merged student answers "B" to everything."""

    def __init__(self):
        self.student = False

    def to(self, device):
        return self

    def eval(self):
        return self

    def merge_and_unload(self):
        self.student = True
        return self

    def __call__(self, input_ids):
        ids = input_ids.a[0]
        rng = np.random.default_rng(int(ids.sum()))
        logits = rng.normal(size=(1, len(ids), VOCAB))
        if self.student:
            logits[0, :, ord("B")] += 20.0
        return types.SimpleNamespace(logits=FakeTensor(logits))

    def generate(self, input_ids, attention_mask, do_sample, max_new_tokens, eos_token_id, pad_token_id,
                 repetition_penalty):
        assert not do_sample and repetition_penalty == 1.0 and max_new_tokens > 0
        reply = "B" if self.student else "好的。"
        new = [ord(c) for c in reply] + [eos_token_id]
        return FakeTensor(np.concatenate([input_ids.a, np.array([new])], axis=1))


def fake_modules(drive_root):
    tf = types.ModuleType("transformers")
    tf.__version__ = "fake"
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path, revision=None: FakeTokenizer())
    tf.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda path, revision=None, dtype=None: FakeModel())
    pf = types.ModuleType("peft")
    pf.__version__ = "fake"

    def from_pretrained(model, adapter):
        assert os.path.isdir(adapter), adapter
        return model

    pf.PeftModel = types.SimpleNamespace(from_pretrained=from_pretrained)
    ds = types.ModuleType("datasets")

    def load_dataset(name, config, split):
        if "wikitext" in name:
            return {"text": ["plain text line %d" % i for i in range(40)] + ["", " "]}
        return [{"question": "q%d" % i, "choices": ["a", "b", "c", "d"], "answer": i % 4} for i in range(300)]

    ds.load_dataset = load_dataset
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path: None)
    google.colab = colab
    return {"torch": fake_torch(), "transformers": tf, "peft": pf, "datasets": ds, "google": google,
            "google.colab": colab}


class TestNotebookStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb, cls.codes = load_cells()
        cls.ns = {}
        run_quiet(cls.codes[3], cls.ns)

    def test_cells_compile_and_no_outputs(self):
        self.assertEqual(len(self.codes), 9)
        for i, src in enumerate(self.codes):
            compile(src, f"cell {i + 1}", "exec")
        for c in self.nb["cells"]:
            if c["cell_type"] == "code":
                self.assertEqual(c["outputs"], [])
                self.assertIsNone(c["execution_count"])

    def test_prompt_layout_matches_deployment(self):
        self.assertEqual(self.ns["SCHEDULER_SYSTEM"], COMPACT_SYSTEM)
        self.assertEqual(self.ns["LETTERS"], LETTERS)
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        self.assertEqual(self.ns["chatml"](messages), chatml(messages))
        self.assertEqual(self.ns["chat_prompt"]("s", "u"), chatml(messages))

    def test_scheduling_examples_match_simulator(self):
        # each example renders to the stored prompt and its best action is the one-step optimum by at least 1 slot
        self.assertEqual(len(self.ns["SCHED"]), 8)
        for ex in self.ns["SCHED"]:
            s = state_from_dict(ex["state"])
            self.assertEqual(render_compact(s)[0][1]["content"], ex["user"])
            costs = one_step_costs(s)
            self.assertEqual(best_action(costs), ex["best"])
            self.assertEqual(len(costs), ex["n_actions"])
            self.assertGreaterEqual(sorted(costs)[1] - sorted(costs)[0], 0.99)
        self.assertEqual(sum(1 for ex in self.ns["SCHED"] if ex["best"] == 0), 4)

    def test_multiple_choice_is_balanced(self):
        for lang in ("zh", "en"):
            items = self.ns["MCQ"][lang]
            self.assertEqual(len(items), 16)
            self.assertEqual(len({q for q, _, _ in items}), 16)
            for _, options, answer in items:
                self.assertEqual(len(set(options)), 4)
                self.assertIn(answer, (0, 1, 2, 3))
            for a in range(4):
                self.assertEqual(sum(1 for _, _, x in items if x == a), 4)

    def test_helpers(self):
        ns = self.ns
        kinds = [ns["classify_response"](x) for x in ("A", " b。", "", "  ", "A 本地", "我是通义千问。", "M", "L.")]
        self.assertEqual(kinds, ["letter", "letter", "empty", "empty", "text", "text", "text", "letter"])
        stats = ns["chat_stats"]([{"response": "A", "letter_mass": 1.0}, {"response": "你好", "letter_mass": 0.0}])
        self.assertEqual((stats["n"], stats["letter"], stats["text"], stats["empty"]), (2, 1, 1, 0))
        self.assertAlmostEqual(stats["letter_mass"], 0.5)
        self.assertAlmostEqual(ns["perplexity"]([np.log(0.5), np.log(0.5)]), 2.0)
        self.assertEqual(ns["argmax"]([0.1, 0.9, 0.9]), 1)
        self.assertEqual(ns["mcq_cloze"]("en", "Q?", "Paris"), ("Question: Q?\nAnswer:", " Paris"))
        self.assertEqual(ns["mcq_cloze"]("zh", "问？", "北京"), ("问题：问？\n答案：", "北京"))
        self.assertEqual(ns["base_of"]("ps_soft_1.5b"), "base_1.5b")
        self.assertEqual(ns["base_of"]("main_0.5b"), "base_0.5b")


class TestNotebookDryRun(unittest.TestCase):
    def test_cells_2_to_9_with_fakes(self):
        _, codes = load_cells()
        with tempfile.TemporaryDirectory() as drive:
            for key, name in STUDENTS.items():
                run = os.path.join(drive, "s3_runs", key + "_20261104T000000Z")
                os.makedirs(os.path.join(run, "adapter"))
                with open(os.path.join(run, "config.json"), "w", encoding="utf-8") as f:
                    json.dump({"name": name}, f)
            ns = {}
            with mock.patch.dict(sys.modules, fake_modules(drive)):
                for i in range(1, 9):            # cell 1 installs packages; skip it
                    src = codes[i].replace("/content/drive/MyDrive/COMLLM", drive)
                    out = run_quiet(src, ns)
                out_dir = os.path.join(drive, "s8_runs", "language_check")
                results = {}
                for key in ["base_0.5b", "base_1.5b"] + list(STUDENTS):
                    with open(os.path.join(out_dir, f"model_{key}.json"), encoding="utf-8") as f:
                        results[key] = json.load(f)
                    self.assertTrue(os.path.exists(os.path.join(out_dir, f"items_{key}.json")))
                # the fake student answers "B" to everything, the fake base answers with a sentence
                base, student = results["base_0.5b"], results["main_0.5b"]
                self.assertEqual(base["adapter_run"], "")
                self.assertTrue(student["adapter_run"].endswith("main_0.5b_20261104T000000Z"))
                self.assertEqual(student["chat"]["default"]["letter"], 20)
                self.assertEqual(base["chat"]["default"]["letter"], 0)
                self.assertEqual(base["chat"]["default"]["text"], 20)
                self.assertGreater(student["chat"]["scheduler"]["letter_mass"], 0.99)
                self.assertEqual(student["mcq"]["zh"]["letter_acc"], 0.25)       # always "B": 4 of 16
                self.assertEqual(student["mcq"]["mmlu"]["n"], 200)
                self.assertAlmostEqual(student["mcq"]["mmlu"]["letter_acc"], 0.25, delta=0.1)   # a random 200 of 300
                self.assertEqual(student["sched"]["n"], 8)
                for r in results.values():
                    self.assertGreater(r["ppl"]["zh"]["tokens"], 300)
                    self.assertGreater(r["ppl"]["wikitext"]["tokens"], 100)
                    self.assertTrue(np.isfinite(r["ppl"]["en"]["ppl"]))
                with open(os.path.join(out_dir, "report.md"), encoding="utf-8") as f:
                    report = f.read()
                self.assertEqual(report, out[:len(report)])
                self.assertIn("| main_0.5b |", report)
                self.assertIn("MMLU", report)
                # a second run skips every finished model
                again = run_quiet(codes[7].replace("/content/drive/MyDrive/COMLLM", drive), ns)
                self.assertEqual(again.count("已经跑过，跳过"), 8)


if __name__ == "__main__":
    unittest.main()
