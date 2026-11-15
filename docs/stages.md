# 各阶段的做法、命令和完整结果

[README](../README.zh-CN.md)（[English](../README.md)）只放结论。这里按阶段记每一步怎么做、用什么命令、完整结果和要注意的地方，和原论文不一样的地方也在这里。数字全部实测，每个数字来自哪个运行目录见 [figures/numbers.md](../figures/numbers.md) 和 [docs/media/numbers.md](media/numbers.md)。下面的命令都在仓库根目录、激活 `.venv` 后运行。

## 和原论文的差别

下表列出本仓库和原论文（arXiv:2604.07148v2）不一样的地方，以及为什么不一样。公式号和表号都指论文。

| 项目 | 论文 | 本仓库 | 原因和影响 |
|---|---|---|---|
| 边缘服务器的排队时间 | 式 4 说服务器在活跃任务之间平分算力；式 5–7 把排队时间近似为 (积压 + 任务大小) × (活跃数 + 1) × γ ÷ F，积压是整台服务器的 | 仿真器按"平分算力"逐个任务推进；调度决策和训练标签仍用论文公式 | 平分算力时积压等于被算了两遍（推导见 [paper-delay-formula.md](paper-delay-formula.md)），所以这个近似在闭环高负载下偏悲观，单步最优会把太多任务留在本地。3 台服务器、40 台终端时，单步最优丢弃 6.3%；换成按平分算力推导的估计后只丢 0.01%（仿真实测） |
| 评测方式 | 式 26–27：在 1000 条互相独立的测试状态上逐条打分再平均（开环） | 闭环：每个决策都会改变后面的队列状态，延迟和丢弃按仿真里真实发生的排队统计 | 开环看不到估计公式的偏差、多个决策扎堆选同一台服务器的羊群效应，也看不到决策本身的耗时 |
| 训练用的服务器台数 | 第 V-B 节：只在 6 台上训练，直接测 3/5/7/9/11 台 | 在 3/5/7/9/11 台的混合上训练；4/6/8/10 台训练时不出现，专门测泛化 | 原作者提供的数据集本身就是 3/5/7/9/11 台混合 |
| 本地 CPU 频率 | 表 I：2 GHz | 2.5 GHz | 以原作者数据集为准 |
| 超时惩罚 ρ | 表 I：10 | 数据里写的是 100，只在论文格式的提示词里原样写出；评测时平均延迟只算完成的任务，丢弃率单独报 | 延迟和丢弃分开报，结果不受 ρ 取值影响 |
| 任务到达 | 第 III-A 节正文写泊松到达；表 I 写到达概率 0.3 | 每台终端每个时间槽以 0.3 的概率产生一个任务 | 和表 I、数据集一致 |
| 上传速率 | 式 1 按香农公式随信道变化；表 I 给的平均值是 14 Mbps | 固定 14 Mbps（1.4 Mbits/槽） | 数据集里的上传速率都是 1.4 Mbits/槽 |
| 提示词和答案 | 第 IV-C 节：约 500 token 的说明 + 每台服务器约 115 token 的状态；答案是中文动作名（约 12 个 token） | 压缩格式：系统说明约 130 token + 每台服务器一行数字；答案是 1 个字母（A = 本地，B = 服务器 1 ……） | 决策只要 1 个 token、不用生成。1.5B Q4_K_M 在 Jetson 上从 1064 ms 降到 119 ms（阶段 2 实测） |
| 训练目标和数据量 | 式 16：对整段答案的每个 token 算负对数似然；SFT 用 1000 条（第 V-A 节） | 只在答案那 1 个位置算 KL(软标签 ‖ 模型)，软标签 = softmax(−成本 / T)；4 万条 | 软标签把"差不多一样好"的动作也告诉模型；数据由自己的仿真器生成，数量不受限 |
| LACS 的未来影响 | 式 20 假装执行时先减掉一个时间槽的服务量；式 21 随机抽 K = 3 个未来任务 | 不减，和原作者仓库的实现一致；在 0.5–1.5 倍任务大小之间均匀取 41 个点求平均 | 去掉抽样噪声。λ = 0.3 时 LACS 最多只在 0.80% 的仿真状态上改变决策，闭环延迟最多差 0.18%（阶段 1 实测），所以软标签只用单步成本 |

丢弃的判定和论文相同：按式 27，延迟超过截止期 10 槽就算丢弃。原作者仓库的评测脚本用的是 12.5 槽，对比数字时要注意。

## 仿真器（阶段 1）

在仓库根目录、激活 `.venv` 后运行：

```bash
python -m unittest discover -s tests -v                 # 42 个测试，约 40 秒；原作者数据默认从 ~/Documents/COMLLM训练部署及改良/from_friend/data 读
python -m eval.closed_loop --name s1_sweep_3srv         # 扫终端数 N，结果写进 runs/<UTC>_s1_sweep_3srv/results.csv
python -m eval.plot_sweep runs/<目录>/results.csv       # 画平均延迟和丢弃率随 N 的变化
python -m eval.lacs_check_sim --sweep runs/<目录>/results.csv   # 在仿真器状态分布上重跑 LACS 检查
python -m eval.show_prompts                             # 看两种提示词格式
```

参数全部在 `sim/config.py`，数值和原作者的三份数据集（1914 条）一致：时间槽 0.1 s，截止期 10 槽，任务 2–5 Mbits 均匀分布，每台终端每槽以 0.3 的概率产生任务，上传 1.4 Mbits/槽（14 Mbps），本地 CPU 2.5 GHz（论文表 I 写 2 GHz，以数据为准），每比特 297 周期。服务器阵容 3/5/7/9/11 台取自数据；4/6/8/10 台是自定的、训练时不出现的阵容，只用来测泛化。

建模选择：

- **服务器按任务平分算力（processor sharing）。** n 个任务同时在一台频率 F 的服务器上时，每个任务分到 F/n。调度决策用的单步公式沿用论文：假设新任务加入后活跃数加 1，分到 F/(n+1)。这和 Tang & Wong（IEEE TMC 2022，论文的环境来源）不同：他们在每台边缘服务器上为每个终端开一条 FIFO 队列，算力在"非空的队列"之间平分，同一终端发到同一台服务器的多个任务要排队依次算；这里是"每个任务"平分，同一终端的任务也并行。
- **本地 CPU 和上传链路是每台终端各一条 FIFO 队列。** 任务从到达起超过 10 槽还没完成就丢弃，丢弃时立刻离开队列。
- **调度器看到的服务器状态包含"已派出、还在上传"的任务。** 调度器知道自己派过哪些任务，这些任务马上会到服务器上分算力。
- **同一时间槽的多个任务有两种决策方式：** `sequential`（逐个决策，后一个看到前一个的结果）和 `batch`（都看槽开始时的同一份状态，模拟 LLM 批量推理，会出现羊群效应）。公式基线默认用 `sequential`。
- **指标：** 平均延迟只算完成的任务（单位：槽）；丢弃率单独算；负载均衡用 Jain 指数，按卸载到每台服务器的任务数算，n 取真实服务器台数（原作者仓库的脚本取的是"被选中过的最大编号"）。

## 调度节点与实时闭环（阶段 2）

```
环境节点（开发期 Mac，阶段 5 换 RK3588）              调度节点（Jetson Orin Nano）
  eval/realtime.py：按真实时钟推进仿真      ──HTTP──▶  deploy/scheduler.py（端口 8100）
  eval/bench_latency.py：连发 N 次量延迟    ◀────────    │ 拼提示词 → llama-server（端口 8080，只听本机）
  所有时间戳都在这一侧记                                  │ 超时（默认 100 ms）或答案不可用 → 单步公式回答
```

两块板子的地址、端口、超时阈值只写在 `deploy/testbed.json`。

Jetson 上（仓库根目录，激活 `.venv`）：

```bash
python -m deploy.scheduler --model ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf   # 自动拉起 llama-server
```

Mac 上：

```bash
python -m eval.bench_latency --fmt compact --cache on     # 500 次请求的端到端延迟和分段耗时
python -m eval.latency_table                              # 把所有基准汇总成一张表
python -m eval.realtime --policy compact --n 5 --minutes 10 --name smoke_llm   # 实时闭环
```

设计选择：

- **压缩格式只算 1 个 token，读概率而不是加语法。** 请求 `n_predict=1`、`n_probs=20`，在前 20 个候选里取概率最大的合法字母。GBNF 语法也能保证答案合法，但它要逐个检查 15 万个词表 token，模型首选不是合法字母时每个 token 要多花约 20 ms（云端 CPU 实测，Jetson 上待测），所以只作为对照。
- **论文格式按原作者仓库的拼法，生成到结束符为止，最多 20 个 token。** 系统提示和用户内容和阶段 3 的论文原版基线一样（见下面"论文原版基线"）；回答去掉首尾空白后必须正好是"本地执行"或"卸载到边缘服务器k"，否则按不可用处理、回退公式。阶段 2 的论文格式延迟是按旧拼法（状态字典的 JSON、固定生成 12 个 token）量的，阶段 5 用新拼法重量。
- **决策在收到回复的那一刻生效。** 实时模式下，任务从到达开始计时，等决策的时间（环境节点排队 + 网络 + Jetson）都算进任务延迟；过了截止期还没拿到决策的任务记为在"决策"环节丢弃。
- **环境节点一次只发一个请求。** 每个请求看到的状态包含之前所有决策的结果；Jetson 忙时请求在环境节点排队。理论容量 N_max = 1 / (3 × 平均往返秒数)：每台终端每秒 0.3 × 10 = 3 个请求。
- **超时回退。** Jetson 上的服务最多等 LLM `timeout_ms`；等不到就用单步公式回答，在日志里记 `source = fallback`、`reason = timeout`。已经开始算的 LLM 请求会算完，结果记为"迟到"，不影响已经给出的答案。迟到的请求还在算时又来新请求，新请求不排队，马上用公式回答（`reason = busy`）：否则新请求排在后面也会超时，每个都白等 `timeout_ms`，而 100 ms 比 5 台终端时两个请求的平均间隔（约 68 ms）还长，待决策的任务会越积越多。云端预演时 CPU 卡顿过一次，就出现了这样的连锁超时，决策延迟 p99 涨到约 300 ms（云端实测），这条规则就是为它加的。

## 训练与开环评测（阶段 3）

```
Mac        python -m train.make_data          → data/s3/（6 个文件，不进仓库）→ 上传 Drive 的 MyDrive/COMLLM/s3_data/
Colab      train/S3_训练.ipynb                 RUN = sweep_0.5b → main_0.5b → soft_1.5b → hard_1.5b（可选 ps_0.5b）
           train/S3_论文原版基线.ipynb          原作者的 sft / grpo / score-grpo，论文格式
           结果在 MyDrive/COMLLM/s3_runs/，下载到 runs/s3/
Mac        python -m train.small_net（可选）   → runs/s3/small_net_<UTC>/
Mac        python -m eval.open_loop runs/s3/*  → evidence/G3/open_loop.md 和 open_loop.csv
Jetson+Mac python -m eval.check_gguf --expected runs/s3/<目录>/jetson_check/expected_200.jsonl --name <名字>
```

**数据。** 每种 (服务器台数, 终端数 N, 行为策略, 种子) 跑一次闭环仿真，热身 200 槽后隔开取状态，按提示词全文去重；训练集里出现在任何评测集的状态都删掉（交集为 0，`meta.json` 里有记录）。

| 文件 | 服务器台数 | 每种台数 | 行为策略 | 用途 |
|---|---|---|---|---|
| train | 3/5/7/9/11 | 约 2 万 | 60% 贪心、20% 带 20% 随机动作的贪心、20% PS 感知贪心 | 训练（每次抽 4 万条） |
| val | 3/5/7/9/11 | 约 400 | 贪心 | 选温度 T、看训练曲线 |
| test_seen | 3/5/7/9/11 | 约 600 | 贪心 | 测试 |
| test_unseen | 4/6/8/10 | 约 600 | 贪心 | 测试泛化：训练时没出现过的台数 |
| friend_clean | 3/5/7/9/11 | 共 245 | 原作者 | 原作者测试集 306 条去掉和他训练集重复的 61 条 |

**标签和损失。**

- 软标签 p_a = softmax(−(c_a − c_min) / T)，只在合法动作上；T 在验证集上从 0.1 / 0.3 / 1.0 里选"多花延迟"最小的，硬标签（只给最优动作）做对照。
- 标签用压缩格式精度（任务大小 2 位、其余 1 位小数）的状态算，因为学生看到的就是这个精度；评测用 4 位小数状态的成本。两者最优动作不同的状态就是学生的上限，开环表里"公式·压缩精度"一行。
- 损失 = KL(p_老师 ‖ p_学生)，只在答案那 1 个位置算。学生的概率先在整个词表上做 softmax，再取 12 个字母那几项：概率漏到非字母 token 上，字母上的概率就变小，损失跟着变大，所以模型也学会了"只答字母"。
- LoRA：r = 16，α = 32，dropout 0.05，所有线性层；学习率 2e-4，前 3% 线性升温后余弦下降；每次更新 16 条，bf16。
- 训练和评测都用部署时同一条规则选动作：整个词表概率最高的 20 个 token 里，取概率最大的合法字母；一个都没有就算非法。notebook 会核对 Hugging Face 对话模板拼出的提示词和 `deploy/prompts.py` 拼的一字不差。

**论文原版基线。** 原作者的三个 1.5B 模型用原作者仓库训练和评测脚本（`stf_7B.py`、`comllm_train.py`、`evaluate_model.py`）的拼法：系统提示"你是一个MEC任务卸载专家……"，用户内容 = 说明 + 空一行 + "**输入数据**：" + 每行一个"键: 值"。贪心生成最多 20 个 token，在回答里找第一个"本地执行 / 卸载到边缘服务器k"，k 超出台数算非法。原作者的评测脚本是 temperature 0.7 采样，这里统一用贪心，结果可以复现。notebook 先用 sft 模型在 60 条上把这种拼法和另外 4 种（Qwen 默认系统提示 + 状态字典的 JSON 等）都试一遍，和原作者标签一致最多的那种用于全部评测（`probe.json`）。

**开环指标**（`eval/open_loop.py`，单位：槽）。和论文表 II 对齐，这里的 AL 按式 26 计入超时惩罚 ρ = 10（表 I），非法动作记 截止期 + ρ = 20；PR = 最优动作的 AL ÷ 策略的 AL（式 28–29）；丢弃 = 成本超过截止期或动作非法（式 27 的开环版）；LBI 按式 31，只看卸载的任务，n 取真实台数，按台数分别算再平均。闭环评测（阶段 5）仍按上面"超时惩罚 ρ"一行，延迟和丢弃分开报。

**小网络对照（可选）。** `python -m train.small_net`（Mac，numpy，约 20 秒）：同一个 6 → 32 → 32 → 1 的 MLP 给每个选项打分，共 1313 个参数；特征从压缩提示词还原，和 LLM 学生看到的信息一样；用同样的软标签训练，结果写成同样的 `preds.csv`，开环表里多一行。

**GGUF。** 合并 LoRA 后用和 Jetson 同一提交（5aeac37a3）的 `convert_hf_to_gguf.py` 直接出 Q8_0；Q4_K_M 先出 F16 再用 `llama-quantize` 量化。`eval/check_gguf.py` 把 200 条测试状态发给 Jetson 上的调度服务，和 Colab 上合并后模型（PyTorch）选的动作比；不一致时同时报 Jetson 多花了多少槽，因为领先幅度很小的状态最容易被量化翻过来，而这些状态上选错代价也小。

## GRPO 实验（阶段 4）

```
Mac    python -m train.make_grpo_data      → data/s4/（不进仓库）→ 上传 Drive 的 MyDrive/COMLLM/s4_data/
Colab  train/S4_A_论文原版.ipynb            RUN = diagnose → smoke → fixed → repo（从原作者 sft 1.5B 出发，论文格式）
       train/S4_B_硬标签起点.ipynb           从阶段 3 的 hard_1.5b 出发，压缩格式
       结果在 MyDrive/COMLLM/s4_runs/，下载到 runs/s4/
Mac    python -m eval.open_loop <阶段 3 和 4 的运行目录> --out evidence/G4 --title "阶段 4 开环评测"
Mac    python -m eval.grpo_report runs/s4/*  → evidence/G4/grpo_report.md 和 grpo_curves.png
Mac    python -m eval.s4_breakdown ...       → evidence/G4/breakdown.md
```

**做法。** 自己写的 GRPO 循环（式 17–19），不用现成的训练器。每道题采 8 个回答（温度 1.3，top-p 0.95），组内归一化成优势（式 18）。每批回答只更新一次，所以 ρ = 1，clip 不起作用。KL 用 k3 估计，β = 0.04，参考模型是关掉 LoRA 的起点。LoRA r = 16，学习率 5e-5；λ = 0（不用 LACS），和论文 1.5B 的 GRPO 一样（V-A）。两种奖励对照：fixed 是非法回答 = 最差合法动作的有效成本再差 5 槽；repo 按原作者仓库的规则重写，非法记 5，合法额外加 1。A 的训练题是原作者 GRPO 数据去重、去掉测试集状态后的 830 条，另留 100 条做验证；B 用阶段 3 的训练集，奖励按压缩精度的成本算。

**结论（全部实测；括号里是解码方式）。**

- 原作者三个 1.5B 模型（sft / grpo / score-grpo）在原作者干净测试集 245 条上的准确率：13.5% / 13.9% / 13.5%（贪心），10.6% / 11.0% / 11.0%（整句概率最大），12.4% / 12.7% / 12.6%（0.7 采样期望）。随机选是 15.9%。
- 服务器 1 偏向在 sft 里就有。发给服务器 1 的比例：sft 38.3%、grpo 39.9%（0.7 采样期望），83.7%、98.4%（贪心）。服务器 1 真正最优的只有 13.9%。sft 标签里服务器 1、2、3 各占 12.3%，但动作名第一个数字是"1"的占 22.1%，因为服务器 1、10、11 共用这一位。
- 论文表 V 的 GRPO-1.5B（服务器 1 占 62.7%、本地 3.9%）没有得到同一个数。原作者干净测试集 11 台的 36 条上，原作者 grpo 发给服务器 1 的比例是 18.3%（0.7 采样期望）和 91.7%（贪心）。两边的测试状态不同。
- 从原作者 sft 用 GRPO 训练 204 步（约 40 分钟）。原作者干净测试集 PR 70.4% → 96.5%（fixed）/ 99.1%（repo），准确率 13.5% → 81.2% / 87.8%，都是贪心。第 25 步验证集上服务器 1 占 88–89%、本地 0%（贪心），之后自己走出来。同一个 1.5B 能到这个水平，所以我们的测量不支持"1.5B 容量不足"的解释。
- repo 奖励在这里不比 fixed 差。repo 的漏洞是越界的服务器号比本地得分高，但这次没有被触发，两次训练的越界回答都是 0%。repo 模型有 45/1245 条回答多了"服务器"尾巴，严格解析后干净测试集准确率是 86.9%（贪心）。
- 两个模型都从不选本地，而训练题里本地最优的只有 5.9%。在阶段 3 见过台数的 1000 条上，PR 是 86.0% / 89.3%（贪心），本地最优的题占了多花延迟的 77.9% / 92.5%。
- 阶段 3 的 hard_1.5b 再做 1000 步 GRPO（fixed 奖励）：见过台数准确率 96.0% → 92.9%，没见过台数 95.2% → 92.1%，PR 99.9% → 99.7%（部署规则）。学生已在公式上限，GRPO 没有收益。部署仍是阶段 3 的软标签 0.5B Q8_0。

## 闭环评测（阶段 5）

```
Colab   train/S5_论文原版转GGUF.ipynb        原作者 sft 1.5B → comllm-paper-sft-1.5b-q8_0.gguf（MyDrive/COMLLM/s5/gguf/）
Jetson  python -m deploy.scheduler --model-key main_0.5b       模型表在 deploy/testbed.json 的 models；POST /model 换模型
Mac     python -m eval.bench_latency --fmt compact --cache on --model-key main_0.5b      实验 2 延迟分解
Mac     python -m eval.check_paper_gguf --preds runs/s3/paper_<UTC>/sft/preds.csv       论文原版 GGUF 和 PyTorch 比对
Mac     python -m eval.s5_sweep --plan e1,e1_paper,e1_shed      实验 1 稳定边界 + 论文原版 1 点 + 分流对照（实时闭环，51 个点，约 2.9 小时，估算）
Mac     python -m eval.virtual_loop --plan e3                   实验 3 羊群效应（只用公式，本机几分钟）
Mac     python -m eval.virtual_loop --plan e4a,e4b,e5           实验 4、5（每个决策问 Jetson，71 个点，约 3 小时，估算）
Mac     python -m eval.virtual_loop --plan e6                   实验 6 LACS 往后看在闭环里有没有用（只用公式，252 个点，约 25 分钟，估算）
Mac     python -m eval.figures                                  7 张图 + figures/numbers.md（跑过两板复测还有 twoboard.md）
RK3588  python -m eval.s5_sweep --plan twoboard                 两板复测：testbed.json 的 env_node 改成 RK3588
```

`s5_sweep` 和 `virtual_loop` 都有 `--dry-run`（只列点和预计时长）、`--resume <目录>`（断了接着跑）、`--only <模型键>`、`--rehearse`（不连 Jetson，LLM 换成本机公式加假装的耗时，用来试脚本）。

设计选择：

- **两种闭环。** 实时闭环（`eval/realtime.py`）按真实时钟推进，决策延迟算进任务延迟，量"部署后撑得住多少终端"。部署模型单次约 57 ms（阶段 3 实测），只撑得住 N ≈ 6，这么轻的负载下各策略的延迟差不多，比不出决策好坏。所以质量对比用虚拟时间闭环（`eval/virtual_loop.py`）：仿真每个决策都等 Jetson 答完再往下走，决策耗时不计入，可以在 N = 20–80 的重负载下比。只用公式时它和 `sim.env.run` 结果完全一样（单元测试核对）。
- **换模型不用人守着。** `deploy/testbed.json` 的 `models` 把模型键对到 GGUF 文件和提示词格式。`POST /model {"key": "<模型键>"}` 让调度服务停掉 llama-server、换文件重启、预热；`/info` 报模型键、文件路径和文件的 sha256，每次运行的 config.json 都记下来。扫描脚本按模型把点排在一起，轮到新模型才换一次。
- **LACS 只能在闭环里检验（实验 6）。** 开环 AL 只算这一步的成本，单步最优就是它的下限，LACS"这步多花一点、给后面留余地"的好处在开环里看不出来。实验 6 在虚拟时间闭环里比单步最优和 LACS-greedy（λ = 0.3、1、2；`lacs:<λ>` 换 λ），顺序决策，另加 PS 感知贪心当"改公式"的参照；批量决策下再比一次 λ = 0.3，看往后看能不能减轻羊群效应。COMLLM-7B 的权重没有，这里用 LACS 奖励的最优解代替它。
- **超载分流（`--shed-ms`）。** 环境节点一次只发一个请求，终端数超过 N_max 后待决策的任务越排越长，每个任务都等过截止期，丢弃率一下跳到接近 100%。分流：在环境节点排队超过 `shed_ms` 的任务不再等 Jetson，当场用公式决策，记 `source = shed`。默认关；`--plan e1_shed` 是开分流的对照，实验 1 一起跑。
- **论文原版 1.5B 只跑一个实时点。** 它一次决策约 1 秒（阶段 2 用旧拼法实测，阶段 5 重量），1 台终端每秒就有 3 个请求，N = 1 已经超载；实时闭环只跑 N = 1（`e1_paper`），质量对比放在虚拟时间闭环（`e5_paper`，400 槽）。1.5B 软标签学生一次约 120 ms，用部署的 100 ms 超时会大半回退公式，所以它的超时给 1000 ms。
- **一条命令重画。** `evidence/G5/sources.json` 登记每张图用哪些运行目录；`python -m eval.figures` 照它重画全部图，并把简历要用的数字连同运行目录写进 `figures/numbers.md`。没有这个文件就自动找最新的目录并写出来。
- **RK3588 当环境节点。** 环境节点的代码只依赖 numpy（`deploy/requirements-envnode.txt`），在 Python 3.8 上测过。

**结果（11/7，全部实测；实验 3–6 是虚拟时间闭环里的仿真实测）。** 图在 `figures/`，每个数字和它来自哪个运行目录写在 `figures/numbers.md`。

```
实验 1     runs/20261107T042911Z_s5_e1+e1_paper+e1_shed      Jetson 实时闭环，51 个点
实验 2     runs/20261106T1*_bench_*                          单次决策延迟，6 组配置
实验 3     runs/20261107T071129Z_s5_virtual_e3               羊群效应，96 个点
实验 4、5  runs/20261107T071637Z_s5_virtual_e4a+e4b+e5       泛化、PS 标签、开环对闭环，71 个点
实验 6     runs/20261106T144248Z_s5_virtual_e6               LACS，252 个点
```

- **单次决策（实验 2，图 2）。** 部署配置（0.5B Q8_0 + 提示词缓存）往返 p50 55.0 ms、p99 72.0 ms，3/7/11 台服务器的状态轮流发。原作者 1.5B（论文拼法、关缓存）p50 945.2 ms、p99 1043.9 ms。实验 1 只有 3 台服务器，提示词更短：0.5B 平均 42.1 ms，原作者 1.5B 开缓存 p50 472.9 ms。
- **稳定边界（实验 1，图 1）。** M = 每个种子的决策环节丢弃（过了截止期还没拿到决策）都低于 1% 的最大终端数 N。
  - 公式 M ≥ 80（测到的最大 N 也稳）。N = 40 起的丢弃（N = 40 时 6.1%）是 3 台服务器不够用，决策环节丢弃是 0。
  - 部署的 0.5B M = 7，理论 N_max = 1 ÷（3 × 42.1 ms）= 7.9。N = 8 时丢弃率 24.5%，N = 10 时 96.0%。
  - 0.5B 加超载分流，测到的最大 N = 20 也稳：丢弃率 1.0%，39.5% 的任务由 LLM 决策，其余当场用公式。
  - 1.5B 软标签 M = 3，N_max 3.9。
  - 原作者 1.5B 在 N = 1 就丢 93.4%，其中决策环节 82.4%，N_max 0.7。
  - N = 5 时 0.5B 的平均延迟 4.30 槽，公式 3.53 槽。多出的 0.77 槽主要是排队等决策（估算）。
- **羊群效应（实验 3，图 3）。** 11 台、N = 80，单步最优从批量决策改成顺序决策：平均延迟 5.11 → 4.19 槽（−18.1%），丢弃率 2.4% → 0.1%。反例是 3 台、N = 40：改顺序后平均延迟高 9.3%，丢弃率 1.7% → 6.2%，本地比例 56.2% → 70.7%。原因是论文公式高估了按平分算力（PS）排队的服务器，顺序决策把这个高估一个个叠上去，任务被推回本地。PS 感知贪心改成顺序后，每个点的丢弃率都下降。
- **台数泛化（实验 4b，图 4）。** N = 40，同一个种子。部署的 0.5B 和单步最优的平均延迟差 −0.1% ~ +0.5%，覆盖 3–11 台，包括训练时没见过的 4/6/8/10 台。闭环里和单步最优选得一样的比例是 94.2%–99.1%。
- **开环对闭环（实验 5，图 5）。** 3 台、N = 40。
  - 开环准确率 ≥ 98.8% 的五个策略（单步最优、LACS-greedy、小网络、0.5B、1.5B）闭环平均延迟都在 5.26–5.29 槽，丢弃 5.9%–6.1%，彼此分不开。
  - PS 感知贪心开环准确率只有 19.6%，闭环却最快：4.12 槽，丢弃 0.0%。
  - 随机 5.31 槽看着不差，因为丢了 22.9%，丢掉的任务不算延迟。
  - 未训练的 0.5B 6.14 槽，丢 51.6%。
- **PS 标签（实验 4a，图 6 右）。** 3 台，用 PS 感知贪心当老师训练的 0.5B 和主线 0.5B 比：
  - 平均延迟低 13.4% / 19.6% / 9.9%（N = 20 / 40 / 60）。
  - 丢弃率从 0.1% / 5.9% / 17.3% 降到 0.0% / 0.0% / 0.9%。
  - 模拟器本身就按 PS 排队，这个模型有主场优势，换真实服务器要重测。
- **LACS（实验 6，图 7）。** 顺序决策，3/7/11 台 × N = 20–80 共 12 个点，每点 3 个种子。
  - λ = 0.3（论文）和单步最优选得一样 99.7%。平均延迟平均差 +0.04%（−0.07% ~ +0.21%），等于没变。
  - λ = 1 平均 +0.18%（−1.72% ~ +1.99%），差别来自两边正好打平时选法不同。
  - λ = 2 在 −11.78% ~ +26.92% 之间：负载重时帮忙，负载轻时（7、11 台，N = 20）更慢。
  - PS 感知贪心平均 −9.27%（−22.73% ~ +3.79%）。
  - 批量决策，11 台、N = 80：λ = 0.3 是 5.11 槽，单步最优 5.09 槽。往后看减轻不了羊群效应。
- **原作者 SFT 1.5B 的早期 GGUF 抽样。** 200 条状态中，Jetson Q8_0 和 PyTorch 的动作一致 183 条（91.5%，`runs/20261107T042021Z_gguf_paper_sft`）。这是抽样结果，不建立全量等价性；本轮九模型的完整 5663 状态结果见下方阶段 6。Jetson 闭环数字反映 Q8_0 部署策略。

**看数字时注意。**

- 闭环平均延迟只算完成了的任务，丢得多的策略会显得快，一定要和丢弃率一起看。
- 实验 1 每点 2 个种子、3 分钟，原作者 1.5B 只有 1 个种子。实验 4、5 里 LLM 的点只跑了种子 0，和公式比时用同一个种子，面对的是同一串任务。
- λ = 1 时两边正好打平的状态很多。Mac（ARM）和云上（x86）的浮点舍入不同，打平时选法就不同。实验 6 在两边各跑一遍，252 行里 216 行逐位一样，不一样的 36 行全是 λ = 1，平均变化 +0.18%（Mac）对 +0.15%（云上）。
- 实验 6 跑到第 31 个点时停过一次，11/7 用 `--resume` 接着跑完。续跑那次 `sweep_config.json` 记的版本是 `4567c59-dirty`（改写历史后为 `1957819`，见 [commit-map](commit-map.md)），改动只有 `eval/figures.py`，仿真代码没变。
- 图 6 里"原作者 sft + GRPO"两根柱子在见过台数上只测了 1000 条子集，画成斜线。

**还没跑。** `e5_paper`（原作者 1.5B 进虚拟时间闭环，400 槽）和两板复测（RK3588 当环境节点）。两个都是可选项。

## 开环：三套测试集分开报（阶段 6）

```
Mac        python -m eval.open_loop runs/s3/main_0.5b_20261104T131423Z runs/s3/soft_1.5b_20261104T132257Z \
             runs/s3/hard_1.5b_20261104T134648Z runs/s3/ps_0.5b_20261104T140119Z \
             runs/s3/ps_soft_1.5b_20261108T044612Z runs/s3/ps_hard_1.5b_20261108T064436Z \
             runs/s3/paper_20261105T040018Z runs/20261108_s7_sft_baseline runs/20261108T103821Z_s7_colab_7b \
             --out evidence/G6 --title "阶段 6 开环：三套测试集分开看（含本地召回、本地精确）"
           → evidence/G6/open_loop.md、open_loop.csv（按台数）、open_loop_by_n.csv（按终端数 N）
```

只用已有的 `preds.csv` 重新打分，没有重跑模型。三套测试集的本地最优占比是 47.97% / 45.93% / 2.45%，原作者干净测试集上从不选本地也能有 97.55% 的准确率，所以三套分开报，并加了四列：

- 本地最优占比：这一组状态里本地最优的比例，和策略无关。
- 本地召回：本地最优的状态里选了本地的比例。
- 本地精确：选了本地的状态里本地确实最优的比例。
- 卸载准确率：最优是某台服务器的状态里选对的比例。

分母是 0 时记"–"。原作者干净测试集没有终端数 N，不进按 N 的表。复现的 7B、它的起点 SFT-7B 和原作者的三个 1.5B 用论文格式贪心生成，我们的学生用压缩格式取 1 个 token，所以这张表比的是"整套做法"，不是只差模型大小。结论见 README 的"开环：三套测试集"。

## 闭环：Colab 和 Jetson 对比（阶段 6）

```
Mac     python -m eval.colab_loop --pack        打包 sim/、deploy/、eval/ → data/s6_code/（code.zip + meta.json），整个文件夹传到 MyDrive/COMLLM/s6_code/
Colab   train/S6_Colab闭环.ipynb，RUN = "students"   6 个学生 × 6 个点（3 台、N = 20/40/60、种子 0/1、1000 槽），结果写到 MyDrive/COMLLM/s6_runs/colab_loop/
Mac     把 results.csv 和 sweep_config.json 下载到 runs/<UTC>_s6_colab_loop/
```

点的名字和 `eval.virtual_loop` 的一样，Colab 和 Jetson 的结果按名字一一对上。同一个模型的 6 个点各开一个线程跑仿真器，要决策时把状态交给主线程拼成一批，GPU 前向一次答完，所以 `ms_per_decision` 是批量摊下来的时间。notebook 第 4 格在 Colab 上重算核对点（单步公式、3 台、N = 20、种子 0），平均延迟必须和打包时 Mac 上算的完全一样，才往下跑。

这次（`runs/20261112T064150Z_s6_colab_loop`）：代码 `1ae403e`（运行记录里写的是改写历史前的 `f9aad84`，见 [commit-map](commit-map.md)），A100，transformers 5.20.2、peft 0.21.2，36 个点一次跑完，约 70 分钟，回退 0 次。对应的 Jetson 点在 `runs/20261107T071637Z_s5_virtual_e4a+e4b+e5`、`runs/20261108T044444Z_s5_virtual_e4ps_a` 和 `runs/20261108T094747Z_s5_virtual_e4ps_b`。两边对比和 PS 公式 1.5B 硬标签在 N = 60 的例外见 README 的"闭环：Colab 和 Jetson"。

## Jetson 全量开环与补充延迟（阶段 6）

六学生和原作者 SFT、GRPO、score-GRPO 均使用 Q8_0，每模型完整 5663 状态。先检查板端及环境节点进程，再串行运行；开环与延迟扫描不能同时切换同一调度服务。以下是复现入口，已有结果直接复算，无需重复推理。

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

## 两组复现 7B 的 Colab 闭环（阶段 7）

```text
Colab   train/S7_7B闭环_A100普通档_v1.ipynb（原云端 notebook ID 不变），A100、支持 bf16
        现有依赖 transformers==5.20.2、peft==0.21.2；notebook 本身不安装库
        代码执行程序 → 全部运行；Drive 挂载由用户授权
        → MyDrive/COMLLM/s7_closed_runs_20261113_a100_standard/
Mac     下载 results.csv、provenance.json、runtime_identity.json、invocation_*.json、closed_acceptance_*.json
        → runs/20261113T104604Z_s7_colab_closed_loop/
```

入口仍为 [原 Drive notebook](https://colab.research.google.com/drive/1NzefdqRgIiR2a8bLc4AiGRlUu_Wr6iQu)。固定 SFT 目录 `reference_model/qwen2.5-7b-mec-sft-final-all`；`grpo_7b_repro` 合并 `s7_runs/nolacs/lora`（λ = 0），`comllm_7b_repro` 合并 `s7_runs/lacs/lora`（λ = 0.3）。运行前核对 36 文件闭环包、SFT 与 adapter SHA、训练联合验收及 CPU 公式核对点 4.197929414659039；续跑要求 provenance 和硬件身份一致，只跳过完整写入的点。

本次在 A100-SXM4-40GB 完成，torch 2.14.1+cu130、transformers 5.20.2、peft 0.21.2、accelerate 1.15.0、numpy 2.4.6。最终 `closed_acceptance_20261113T092611.183871Z.json` 为 complete，12 个唯一点齐全，CSV SHA 与报告一致。两个模型各 6 点：3 台服务器、N = 20/40/60、种子 0/1、sequential、1000 槽、200 槽热身；计分到达区间为 [200, 989)。12 点的 fallback 分别全为 0，local_frac 分别全为 0。

闭环采用 paper / search / greedy，max_new_tokens = 16、repetition_penalty = 1.05；开环为 15 和 1.0，不能称解码完全相同。README 闭环图里的 7B 按同模型、同 N 的两个种子等权平均，原始逐点数据保留在 CSV。学生参照取 `runs/20261112T064150Z_s6_colab_loop`，greedy、psgreedy、lacs（λ = 0.3）取 `runs/20261107T071637Z_s5_virtual_e4a+e4b+e5`；按 servers、n、mode、seed、slots、warmup 对齐，每个对应点的 n_tasks 和 n_decisions 相同。

虚拟时间不计推理等待；mean_delay 和 p95_delay 只统计完成任务，必须与 drop_rate 一起读。agree_greedy 是所访问状态上的单步公式一致率，不是准确率，和 fallback / n_decisions 一样覆盖全部决策；local_frac 按计分任务统计。动作指标通常包含回退后的动作，本次没有回退。结果仅适用于两份复现权重，不是原版 COMLLM-7B 的成绩，也不是只差模型大小的控制实验；7B 未在 Jetson 上运行。

## 先来先服务服务器的对照（阶段 6）

```bash
python -m eval.fcfs_control        # 只用公式，不连 Jetson，本机约 20 秒；结果在 runs/<UTC>_s6_fcfs_control/
```

PS 感知公式是按论文式 4 的平分算力推出来的，仿真器的服务器也按式 4 排队，所以它比论文公式好，有可能只是因为两边假设一致。这个对照把服务器换成先来先服务（`sim/env.py` 的 `FcfsServer`，一次只算最早到的那个任务），其余不变，重跑实验 6 的公式对公式：3/7/11 台 × N = 20/40/60/80、种子 0/1/2、每点 2000 槽、顺序决策。另加一个按先来先服务写的公式（`sim/cost.py` 的 `fcfs_costs`）当参照。`MecEnv` 默认仍是平分算力，已有结果不受影响。

这次（`runs/20261114T120231Z_s6_fcfs_control`，代码 `a8f2d20`）：

| 服务器 | 公式 | 平均延迟比论文公式 | 范围 | 更快的点 | 丢弃率（它 / 论文公式） | 留在本地的任务（它 / 论文公式） |
|---|---|---|---|---|---|---|
| 平分算力 | PS 感知公式 | −9.27% | −22.73% ~ +3.79% | 10/12 | 0.82% / 4.22% | 28.2% / 58.3% |
| 平分算力 | 先来先服务公式 | −9.26% | −22.74% ~ +3.79% | 10/12 | 0.81% / 4.22% | 28.3% / 58.3% |
| 先来先服务 | PS 感知公式 | −12.88% | −26.40% ~ +1.32% | 11/12 | 0.05% / 4.14% | 28.3% / 58.3% |
| 先来先服务 | 先来先服务公式 | −12.89% | −26.34% ~ +1.32% | 11/12 | 0.04% / 4.14% | 28.4% / 58.3% |

- 平分算力那 72 次运行（论文公式和 PS 感知公式）和实验 6 的结果逐项相同，说明加了 `discipline` 参数后原来的仿真没变。
- PS 感知公式的优势不依赖平分算力这个假设：换成先来先服务，优势更大。
- 两个新公式在 72 次运行里有 48 次结果完全一样。它们的共同点是扣掉上传期间服务器已经算完的积压、不把积压再乘 (n + 1)；起作用的是这两处，而不是"平分"这一项。论文公式把服务器估得太慢，所以把更多任务留在本地。
- 只比了公式。照 PS 感知公式训练的学生没有在先来先服务的服务器上跑；逐点的表在运行目录的 `summary.md`。
