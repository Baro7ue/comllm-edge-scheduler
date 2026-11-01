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
