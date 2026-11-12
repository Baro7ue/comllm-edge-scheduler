"""Exercise the 7B notebook guard without loading a model or importing torch."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "train" / "S7_7B闭环_校验修复_v1.ipynb"
ORIGINAL = ROOT / "runs" / "20261112T073239Z_s7_colab_closed_loop" / "S7_7B闭环.executed.ipynb"


class FakeGenerationConfig:
    def __init__(self, beams=None, penalty=1.05):
        self.num_beams = beams
        self.repetition_penalty = penalty

    @staticmethod
    def _get_default_generation_params():
        return {"num_beams": 1, "repetition_penalty": 1.0}


class TestS7ClosedNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text())
        cls.original = json.loads(ORIGINAL.read_text())
        cls.cells = [c for c in cls.notebook["cells"] if c["cell_type"] == "code"]
        cls.source = "".join(cls.cells[5]["source"])
        tree = ast.parse(cls.source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "checked_generation_settings")
        cls.function = function
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
        cls.check = staticmethod(namespace[function.name])

    def call_guard(self, config):
        before = copy.deepcopy(vars(config))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.check(config)
        self.assertEqual(vars(config), before)
        self.assertIn("raw", output.getvalue())
        self.assertIn("effective", output.getvalue())
        return result

    def test_absent_beams_use_library_default_without_mutating_config(self):
        config = FakeGenerationConfig()
        result = self.call_guard(config)
        self.assertIsNone(result["raw"]["num_beams"])
        self.assertEqual(result["effective"]["num_beams"], 1)
        self.assertIsNone(config.num_beams)

    def test_explicit_single_beam_is_valid(self):
        self.assertEqual(self.call_guard(FakeGenerationConfig(1))["effective"]["num_beams"], 1)

    def test_invalid_beam_values_remain_rejected(self):
        for value in (True, False, 1.0, "1", 0, 2, -1):
            with self.subTest(value=value):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(AssertionError, "num_beams"):
                    self.check(FakeGenerationConfig(value))

    def test_repetition_penalty_must_still_be_105(self):
        for value in (None, 1.0, True, "1.05", float("nan"), float("inf")):
            with self.subTest(value=value):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(AssertionError, "repetition_penalty"):
                    self.check(FakeGenerationConfig(penalty=value))

    def test_rest_of_execution_source_is_identical(self):
        old_cells = [c for c in self.original["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(self.cells), 7)
        for i, (old, new) in enumerate(zip(old_cells, self.cells)):
            if i != 5:
                self.assertEqual(old["source"], new["source"])
        lines = self.source.splitlines(keepends=True)
        without_function = "".join(lines[self.function.end_lineno:]).lstrip("\n")
        restored = without_function.replace(
            "            checked_generation_settings(model.generation_config)\n",
            "            assert model.generation_config.repetition_penalty == 1.05 and model.generation_config.num_beams == 1\n",
        )
        self.assertEqual(restored, "".join(old_cells[5]["source"]))


if __name__ == "__main__":
    unittest.main()
