"""Can every number and run folder in the two READMEs be traced to a file in the repository?

1. Run folders named in README.md, README.zh-CN.md, docs/stages.md and docs/stages.zh-CN.md (runs/ followed by a UTC time) exist;
   names with * are globs and must match at least one folder
2. The run folders and evidence tables in the last column of figures/numbers.md exist
3. Every number with a decimal point or a percent sign in the results table of each README
   (the first table under "## Results" / "## 结果") appears in figures/numbers.md or docs/media/numbers.md
4. Every image the READMEs show exists, and the two READMEs show the same figures, each in its own language
"""
import glob
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PATTERN = re.compile(r"runs/(\d{8}T[0-9A-Za-z_*+.\-]*)")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?")
IMAGE_PATTERN = re.compile(r'<img src="([^"]+)"|!\[[^\]]*\]\(([^)]+)\)')
READMES = {"README.md": "## Results", "README.zh-CN.md": "## 结果"}
NUMBER_FILES = ("figures/numbers.md", "docs/media/numbers.md")


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def results_column(readme, heading):
    """Column 2 (the result) of the first table under the given heading.

    1. find the line that is exactly the heading
    2. skip to the first line that starts with "|"; that is the header row, the next one the separator
    3. collect column 2 of every following row until the first line that does not start with "|"
    """
    lines = readme.split("\n")
    start = lines.index(heading)
    i = start + 1
    while not lines[i].startswith("|"):
        i += 1
    cells = []
    for line in lines[i + 2:]:
        if not line.startswith("|"):
            break
        cells.append(line.split("|")[2].strip())
    return cells


def images(text):
    out = []
    for a, b in IMAGE_PATTERN.findall(text):
        out.append(a or b)
    return out


class TestReadmeSources(unittest.TestCase):
    def test_run_dirs_exist(self):
        missing = []
        for rel in list(READMES) + ["docs/stages.md", "docs/stages.zh-CN.md"]:
            for name in RUN_PATTERN.findall(read(rel)):
                name = name.rstrip(".")
                if not glob.glob(os.path.join(ROOT, "runs", name)):
                    missing.append(f"{rel}: runs/{name}")
        self.assertEqual(missing, [], "these run folders are not in the repository")

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
        self.assertEqual(missing, [], "sources named in figures/numbers.md are not in the repository")

    def test_headline_numbers_have_sources(self):
        numbers = "\n".join(read(p) for p in NUMBER_FILES)
        for rel, heading in READMES.items():
            cells = results_column(read(rel), heading)
            self.assertGreater(len(cells), 5, f"{rel}: results table not found")
            missing = []
            for cell in cells:
                for token in NUMBER_PATTERN.findall(cell):
                    if "." not in token and "%" not in token:
                        continue
                    if token not in numbers:
                        missing.append(token)
            self.assertEqual(missing, [], f"{rel}: these numbers are not in {' or '.join(NUMBER_FILES)}")

    def test_images_exist_and_match(self):
        shown = {}
        for rel in READMES:
            paths = images(read(rel))
            self.assertGreater(len(paths), 0, rel)
            for p in paths:
                self.assertTrue(os.path.exists(os.path.join(ROOT, p)), f"{rel}: {p}")
            shown[rel] = paths
        en = [p.replace("/en/", "/") for p in shown["README.md"]]
        zh = [p.replace("/zh/", "/") for p in shown["README.zh-CN.md"]]
        self.assertEqual(en, zh, "the two READMEs should show the same figures in the same order")
        self.assertTrue(all("/en/" in p for p in shown["README.md"]))
        self.assertTrue(all("/zh/" in p for p in shown["README.zh-CN.md"]))


if __name__ == "__main__":
    unittest.main()
