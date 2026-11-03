# comllm-edge-scheduler

基于开源论文 COMLLM（arXiv:2604.07148）的任务卸载调度器端侧部署：Jetson Orin Nano 8GB 做调度节点（llama.cpp/CUDA），RK3588 做环境节点（开发期用 Mac 代替），把决策延迟计入闭环评测。

> 开发中（2026 年 11 月）。结果表只放实测值，完成后补全。

## 目录

- sim/：仿真器和基线
- deploy/：Jetson 调度服务
- train/：Colab 训练 notebook
- eval/：评测与出图
- runs/：每次运行的结果，目录名为 UTC 时间加名字
- docs/：版本记录等文档
- evidence/：各阶段验收证据
- tests/：单元测试（`python -m unittest discover -s tests -v`）

## 仿真器（阶段 1）

在仓库根目录、激活 `.venv` 后运行：

```bash
python -m unittest discover -s tests -v                 # 17 个测试；朋友数据默认从 ~/Documents/COMLLM训练部署及改良/from_friend/data 读
python -m eval.closed_loop --name s1_sweep_3srv         # 扫终端数 N，结果写进 runs/<UTC>_s1_sweep_3srv/results.csv
python -m eval.plot_sweep runs/<目录>/results.csv       # 画平均延迟和丢弃率随 N 的变化
python -m eval.lacs_check_sim --sweep runs/<目录>/results.csv   # 在仿真器状态分布上重跑 LACS 检查
python -m eval.show_prompts                             # 看两种提示词格式
```

参数全部在 `sim/config.py`，数值和朋友的三份数据集（1914 条）一致：时间槽 0.1 s，截止期 10 槽，任务 2–5 Mbits 均匀分布，每台终端每槽以 0.3 的概率产生任务，上传 1.4 Mbits/槽（14 Mbps），本地 CPU 2.5 GHz（论文表 I 写 2 GHz，以数据为准），每比特 297 周期。服务器阵容 3/5/7/9/11 台取自数据；4/6/8/10 台是自定的、训练时不出现的阵容，只用来测泛化。

建模选择：

- **服务器按任务平分算力（processor sharing）。** n 个任务同时在一台频率 F 的服务器上时，每个任务分到 F/n。调度决策用的单步公式沿用论文：假设新任务加入后活跃数加 1，分到 F/(n+1)。这和 Tang & Wong（IEEE TMC 2022，论文的环境来源）不同：他们在每台边缘服务器上为每个终端开一条 FIFO 队列，算力在"非空的队列"之间平分，同一终端发到同一台服务器的多个任务要排队依次算；这里是"每个任务"平分，同一终端的任务也并行。
- **本地 CPU 和上传链路是每台终端各一条 FIFO 队列。** 任务从到达起超过 10 槽还没完成就丢弃，丢弃时立刻离开队列。
- **调度器看到的服务器状态包含"已派出、还在上传"的任务。** 调度器知道自己派过哪些任务，这些任务马上会到服务器上分算力。
- **同一时间槽的多个任务有两种决策方式：** `sequential`（逐个决策，后一个看到前一个的结果）和 `batch`（都看槽开始时的同一份状态，模拟 LLM 批量推理，会出现羊群效应）。公式基线默认用 `sequential`。
- **指标：** 平均延迟只算完成的任务（单位：槽）；丢弃率单独算；负载均衡用 Jain 指数，按卸载到每台服务器的任务数算，n 取真实服务器台数（朋友仓库的脚本取的是"被选中过的最大编号"）。

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
- **论文格式固定生成 12 个 token。** 用来量"论文原版做法"（长提示词 + 生成答案）在同一套部署上的耗时。论文原版模型的答案是 `{"action_name": "卸载到边缘服务器1"}` 这样一段 JSON，朋友数据集里的答案用 Qwen 分词平均约 12 个 token。
- **决策在收到回复的那一刻生效。** 实时模式下，任务从到达开始计时，等决策的时间（环境节点排队 + 网络 + Jetson）都算进任务延迟；过了截止期还没拿到决策的任务记为在"决策"环节丢弃。
- **环境节点一次只发一个请求。** 每个请求看到的状态包含之前所有决策的结果；Jetson 忙时请求在环境节点排队。理论容量 N_max = 1 / (3 × 平均往返秒数)：每台终端每秒 0.3 × 10 = 3 个请求。
- **超时回退。** Jetson 上的服务最多等 LLM `timeout_ms`；等不到就用单步公式回答，在日志里记 `source = fallback`、`reason = timeout`。已经开始算的 LLM 请求会算完，结果记为"迟到"，不影响已经给出的答案。迟到的请求还在算时又来新请求，新请求不排队，马上用公式回答（`reason = busy`）：否则新请求排在后面也会超时，每个都白等 `timeout_ms`，而 100 ms 比 5 台终端时两个请求的平均间隔（约 68 ms）还长，待决策的任务会越积越多。云端预演时 CPU 卡顿过一次，就出现了这样的连锁超时，决策延迟 p99 涨到约 300 ms（云端实测），这条规则就是为它加的。
