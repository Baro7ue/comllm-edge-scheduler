"""Render the verified deployment tables into README after postprocessing."""
import csv
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
assert json.loads((HERE / "postprocessing.json").read_text())["status"] == "complete"
REPORT = ROOT / "evidence/G6/jetson_open"
with (REPORT / "platform_comparison.csv").open(newline="") as inp:
    rows = list(csv.DictReader(inp))
with (REPORT / "latency_coverage.csv").open(newline="") as inp:
    benches = list(csv.DictReader(inp))
assert len(rows) == 27 and len(benches) == 9
names = {
    "main_0.5b": "论文老师 0.5B 软标签", "soft_1.5b": "论文老师 1.5B 软标签",
    "hard_1.5b": "论文老师 1.5B 硬标签", "ps_0.5b": "PS 老师 0.5B 软标签",
    "ps_soft_1.5b": "PS 老师 1.5B 软标签", "ps_hard_1.5b": "PS 老师 1.5B 硬标签",
    "paper_sft_1.5b": "原作者 1.5B SFT", "paper_grpo_1.5b": "原作者 1.5B GRPO",
    "paper_score_grpo_1.5b": "原作者 1.5B score-GRPO",
}
splits = ["test_seen", "test_unseen", "friend_clean"]
index = {(r["model_key"], r["split"]): r for r in rows}
assert len(index) == 27 and set(r["model_key"] for r in rows) == set(names)
assert all(int(r["n"]) == {"test_seen":3010, "test_unseen":2408, "friend_clean":245}[r["split"]] for r in rows)
bad = sum(int(r["non_llm"]) for r in rows)
lines = ["## 部署后的开环：九个模型的固定测试集", "",
    f"九个小模型各在 Jetson 上重新回答 3010 条 Seen、2408 条 Unseen 和 245 条原作者干净测试状态，共 50,967 条预测。每个 `(split, idx)` 恰好一条，非 LLM 答案共 {bad} 条；调度服务返回的超时、解析失败等非 LLM 回答以 −1 计入非法率，不把公式回退答案当成模型答案。",
    "本轮开环与延迟测试给每次请求 10 秒作答；实时调度服务的默认 100 毫秒是另一项设置。模型为 Q8_0，llama.cpp 固定 5aeac37a3，上下文 4096，串行单请求；开环均开启前缀缓存，compact 取一个 token，paper 最多生成 20 个 token。", "",
    "score-GRPO 中途遇到一次 HTTP 等待超时，板端已完成该次推理而客户端未收到回复。已有 4472 条原样保留，只续跑缺失的 1191 条；该次未落盘的尝试不计入表中，测试设置与服务均未改变。中断日志和已存前缀的校验值一并保留。", "",
    "上面的开环表复用 Colab/PyTorch 预测；这里是板端结果。下面每格为准确率 / PR（%），最后一列是和既有 PyTorch 预测逐条相同的比例，包含全部状态，不剔除成本打平的状态。PS 学生也按论文成本评分，它们的训练目标不同。", "",
    "| 模型 | Seen（3010） | Unseen（2408） | 原作者干净测试集（245） | 两平台逐条一致率：Seen / Unseen / 原作者（%） |",
    "|---|---|---|---|---|"]
for key, name in names.items():
    cells = [f"{float(index[key,s]['jetson_acc_pct']):.2f} / {float(index[key,s]['jetson_pr_pct']):.2f}" for s in splits]
    agreement = " / ".join(f"{float(index[key,s]['agreement_pct']):.2f}" for s in splits)
    lines.append(f"| {name} | " + " | ".join(cells) + " | " + agreement + " |")
soft_ps = [r for r in rows if r["model_key"] in ("ps_0.5b", "ps_soft_1.5b")]
agreement = [float(r["agreement_pct"]) for r in soft_ps]
pr_gap = max(abs(float(r["jetson_pr_pct"]) - float(r["pytorch_pr_pct"])) for r in soft_ps)
hard_ps = [float(r["agreement_pct"]) for r in rows if r["model_key"] == "ps_hard_1.5b"]
lines += ["", f"PS 软标签两学生的逐条一致率为 {min(agreement):.2f}%–{max(agreement):.2f}%，论文 PR 相差最多 {pr_gap:.2f} 个百分点。PS 硬标签 1.5B 的静态一致率为 {min(hard_ps):.2f}%–{max(hard_ps):.2f}%，仍存在上一节 N = 60 的闭环不稳定现象。汇总分数和逐状态答案提供不同信息；这些结果不建立 PyTorch 等价性，差异也不能全部归因于 Q8。进一步等价性归因调查继续暂缓。", "",
    "完整的本地召回、本地精确、按台数和 N 分解、预测来源及 SHA 见 [板端开环报告](evidence/G6/jetson_open/comparison.md)、[逐条一致率](evidence/G6/jetson_open/prediction_agreement.md) 和 [完整打分表](evidence/G6/jetson_open/open_loop.md)。", "",
    "**单次决策延迟基准。** 每组 500 次计入、10 次热身，3/7/11 台服务器轮流。学生用 compact / cache on，原作者模型用 paper / cache off。两个已有学生基准复用；新增缺失六组，并额外重测 SFT，使原作者三模型使用同一版明确固定的贪心参数（历史 SFT 基准早于该修正）。图 2 保留 11/06 的历史基准，最新覆盖见下表。不同日期、提示词和回答长度仍需结合配置解读，细小差异不作统计显著性判断。", "",
    "| 模型 | p50（ms） | p99（ms） | 来源 |", "|---|---:|---:|---|"]
bench_index = {r["model_key"]: r for r in benches}
for key, name in names.items():
    r = bench_index[key]
    origin = "复用 11/06" if r["reused"] == "True" else "本轮测量"
    lines.append(f"| {name} | {float(r['p50_ms']):.2f} | {float(r['p99_ms']):.2f} | {origin} |")
lines += ["", "延迟是环境节点量到的端到端往返时间，与开环准确率分开测。原始目录、代码与数据指纹及自动验收记录在 `runs/20261113T034033Z_s6_jetson_evaluation`；这次的模型登记代码版本为 cafd7e4。两个原作者 GRPO 文件在本机使用同版本转换器生成，产物与 Colab 记录的完整 SHA-256 相同，传到板端后再次核验通过。", ""]
section = "\n".join(lines)
readme_path = ROOT / "README.md"
readme = readme_path.read_text()
heading = "## 部署后的开环：九个模型的固定测试集"
if heading in readme:
    readme = re.sub(re.escape(heading) + r".*?(?=## 系统\n)", section + "\n", readme, flags=re.S)
else:
    assert "\n## 系统\n" in readme
    readme = readme.replace("\n## 系统\n", "\n" + section + "\n## 系统\n", 1)
date = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
status = f"> 状态（{date}）：九个小模型在 Colab/PyTorch 和 Jetson/Q8_0 上的全量开环已齐，六学生两平台核心闭环各 36/36，Jetson 九模型 500 请求延迟基准已覆盖（部分复用）。7B 开环结果与复现边界见对应章节，7B 闭环进度另行验收。RK3588 两板复测仍按原计划安排；当前实时数字均使用 Mac 作为环境节点。"
readme, count = re.subn(r"^> 状态[^\n]*", status, readme, count=1, flags=re.M)
assert count == 1
sentence = "All nine small models also have full 5,663-state Jetson Q8_0 open-loop predictions and 500-request latency coverage; deployment results are reported separately below. "
needle = "This is an independent project based on the paper, not an official COMLLM release."
if sentence not in readme:
    assert needle in readme
    readme = readme.replace(needle, sentence + needle, 1)
old_ps_latency = "| Jetson 实时：单次决策、稳定边界 | 55.0 ms，7 台终端 | 没测 |"
new_ps_latency = f"| Jetson：单次决策 / 实时稳定边界 | 55.0 ms / 7 台终端 | {float(bench_index['ps_0.5b']['p50_ms']):.2f} ms（延迟基准）/ 实时稳定边界未测 |"
readme = readme.replace(old_ps_latency, new_ps_latency)
readme_path.write_text(readme)
stages_path = ROOT / "docs/stages.md"
stages = stages_path.read_text()
old_probe = "- **原作者 1.5B 的 GGUF。** Jetson 上的 Q8_0 和 PyTorch 选的动作一致 91.5%，不一致的都在本地和卸载成本很接近的状态上（`runs/20261107T042021Z_gguf_paper_sft`）。所以闭环里原作者 1.5B 的数字代表的是 Jetson 上的 Q8_0。"
new_probe = "- **原作者 SFT 1.5B 的早期 GGUF 抽样。** 200 条状态中，Jetson Q8_0 和 PyTorch 的动作一致 183 条（91.5%，`runs/20261107T042021Z_gguf_paper_sft`）。这是抽样结果，不建立全量等价性；本轮九模型的完整 5663 状态结果见下方阶段 6。Jetson 闭环数字反映 Q8_0 部署策略。"
stages = stages.replace(old_probe, new_probe)
heading = "## Jetson 全量开环与补充延迟（阶段 6）"
if heading not in stages:
    stages += "\n" + heading + "\n\n" + """六学生和原作者 SFT、GRPO、score-GRPO 均使用 Q8_0，每模型完整 5663 状态。先检查板端及环境节点进程，再串行运行；开环与延迟扫描不能同时切换同一调度服务。以下是复现入口，已有结果直接复算，无需重复推理。

```bash
python -m eval.jetson_open_loop --models main_0.5b,soft_1.5b,hard_1.5b,ps_0.5b,ps_soft_1.5b,ps_hard_1.5b,paper_sft_1.5b,paper_grpo_1.5b,paper_score_grpo_1.5b --dry-run
# 正式运行去掉 --dry-run；中断后对单模型添加 --resume runs/jetson_open/<目录>
python -m eval.open_loop runs/jetson_open/* --out evidence/G6/jetson_open --title "Jetson 全量开环（GGUF Q8_0）"
python -m eval.jetson_open_loop --compare runs/jetson_open/* --colab runs/s3,runs/s4,runs/s3/paper_20261105T040018Z
python -m eval.bench_latency --fmt compact --cache on --model-key ps_soft_1.5b --n 500 --warmup 10 --servers 3,7,11
python -m eval.bench_latency --fmt paper --cache off --model-key paper_grpo_1.5b --n 500 --warmup 10 --servers 3,7,11
python -m eval.latency_table > evidence/G2/latency_table.md
```

本轮记录在 `runs/20261113T034033Z_s6_jetson_evaluation`：preflight/protocol、模型准备与 SHA、每项日志和 suite_status、离线评分脚本及验收结果。每模型检查 5663 个唯一键与精确分集覆盖；非 LLM 答案记为 −1。两个既有学生延迟基准复用，缺失六组补齐；原作者 SFT 另做一组，以匹配固定贪心后的解码条件。开环与延迟测试每请求均为 10 秒超时，实时服务默认 100 毫秒不用于这些测试。表格和来源索引在 `evidence/G6/jetson_open/`。

PS 0.5B 的台数泛化已在 Jetson 测过 3/4/6/7/8/10/11 台、N = 40、种子 0；和自己的 PS 老师相比平均延迟为 −1.06% 到 +3.95%，不代表多种子统计结论。出处为已归档 Stage 5 与 e4ps_a 结果。
"""
stages_path.write_text(stages)
print("Rendered verified README tables and stage commands")
