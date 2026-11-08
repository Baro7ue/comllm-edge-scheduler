"""README 里的数字和运行目录能不能找到出处（阶段 6 的 G6 自查）。

1. README.md、docs/stages.md 里写到的运行目录（runs/ 后面跟 UTC 时间的）都要在仓库里；带 * 的按通配符找，至少找到一个
2. figures/numbers.md 最后一列的运行目录和开环表都要在仓库里
3. README "结果一览"表"结果"那一列里每个带小数点或百分号的数，都要在 figures/numbers.md 里出现过（正负号不算）
"""
import glob
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PATTERN = re.compile(r"runs/(\d{8}T[0-9A-Za-z_*+.\-]*)")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?")


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def results_column(readme):
    """取出 README "结果一览"表每一行的第 2 列（结果）。

    1. 找到"## 结果一览"这一节，到下一个"## "为止
    2. 以"|"开头的行是表格；跳过表头和分隔行
    3. 按"|"切开，第 2 格就是结果
    """
    lines = readme.split("\n")
    inside = False
    cells = []
    seen_header = False
    for line in lines:
        if line.startswith("## "):
            inside = line.startswith("## 结果一览")
            continue
        if not inside or not line.startswith("|"):
            continue
        if not seen_header:
            seen_header = True
            continue
        if line.startswith("|---"):
            continue
        parts = line.split("|")
        cells.append(parts[2].strip())
    return cells


class TestReadmeSources(unittest.TestCase):
    def test_run_dirs_exist(self):
        missing = []
        for rel in ("README.md", "docs/stages.md"):
            text = read(rel)
            for name in RUN_PATTERN.findall(text):
                name = name.rstrip(".")
                if not glob.glob(os.path.join(ROOT, "runs", name)):
                    missing.append(f"{rel}: runs/{name}")
        self.assertEqual(missing, [], "这些运行目录在仓库里找不到")

    def test_numbers_md_sources_exist(self):
        missing = []
        for line in read("figures/numbers.md").split("\n"):
            if not line.startswith("| ") or line.startswith("| 数字"):
                continue
            where = line.rstrip(" |").split("|")[-1].strip()
            for item in re.split(r", |、", where):
                item = item.strip()
                if item.startswith("evidence/"):
                    path = os.path.join(ROOT, item)
                else:
                    path = os.path.join(ROOT, "runs", item)
                if not os.path.exists(path):
                    missing.append(item)
        self.assertEqual(missing, [], "numbers.md 里写的出处在仓库里找不到")

    def test_headline_numbers_in_numbers_md(self):
        numbers = read("figures/numbers.md")
        cells = results_column(read("README.md"))
        self.assertGreater(len(cells), 0, "README 里没找到结果一览表")
        missing = []
        for cell in cells:
            for token in NUMBER_PATTERN.findall(cell):
                if "." not in token and "%" not in token:
                    continue
                if token not in numbers:
                    missing.append(token)
        self.assertEqual(missing, [], "README 结果表里这些数在 figures/numbers.md 里找不到")


if __name__ == "__main__":
    unittest.main()
