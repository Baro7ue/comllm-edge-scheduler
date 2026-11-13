# Jetson 固定测试集开环与延迟基准

每套状态独立作答。准确率是与数据集动作标签的一致率；三套分别报告。PR = 最优 AL / 策略 AL × 100%。
本地召回、本地精确以论文动作标签为参照；PS 学生的训练目标不同。
两平台逐条一致率衡量部署前后答案变化，不代表任务完成准确率。

score-GRPO 曾发生一次 HTTP 传输等待超时：板端已完成该次推理，客户端未落盘回复。保留已有 4472 条，再续跑缺失的 1191 条；没有修改请求设置或重启服务。那次未收到回复的尝试不计入固定测试输出，原始中断和断点校验记录见运行目录的 interruption_20261113T0749.json。最终每状态仍恰好一条。

本轮一致率包含全部状态，未剔除成本打平的状态。开环均开启前缀缓存，每请求给 10 秒作答；实时调度的默认 100 毫秒不适用于本表。

| 模型 | 测试集 | 样本数 | Jetson 准确率 % | Jetson PR % | PyTorch PR % | 逐条一致 % | 非 LLM | 本地召回 % | 本地精确 % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| main_0.5b | test_seen | 3010 | 97.11 | 99.98 | 99.98 | 97.28 | 0 | 99.38 | 98.36 |
| main_0.5b | test_unseen | 2408 | 95.64 | 99.96 | 99.95 | 97.13 | 0 | 99.28 | 97.51 |
| main_0.5b | friend_clean | 245 | 97.55 | 99.96 | 99.94 | 97.55 | 0 | 83.33 | 100.00 |
| soft_1.5b | test_seen | 3010 | 96.51 | 99.98 | 99.98 | 97.14 | 0 | 99.52 | 98.22 |
| soft_1.5b | test_unseen | 2408 | 94.14 | 99.94 | 99.94 | 97.38 | 0 | 99.19 | 97.77 |
| soft_1.5b | friend_clean | 245 | 97.55 | 99.97 | 99.98 | 98.78 | 0 | 83.33 | 100.00 |
| hard_1.5b | test_seen | 3010 | 95.85 | 99.88 | 99.88 | 99.57 | 0 | 99.03 | 96.95 |
| hard_1.5b | test_unseen | 2408 | 95.22 | 99.93 | 99.93 | 99.63 | 0 | 99.28 | 97.17 |
| hard_1.5b | friend_clean | 245 | 99.18 | 99.99 | 99.99 | 100.00 | 0 | 83.33 | 100.00 |
| ps_0.5b | test_seen | 3010 | 18.17 | 65.82 | 65.67 | 73.82 | 0 | 16.76 | 99.59 |
| ps_0.5b | test_unseen | 2408 | 18.77 | 66.93 | 66.55 | 69.77 | 0 | 18.35 | 99.02 |
| ps_0.5b | friend_clean | 245 | 13.47 | 74.40 | 74.06 | 72.24 | 0 | 0.00 | – |
| ps_soft_1.5b | test_seen | 3010 | 18.70 | 65.90 | 65.57 | 72.33 | 0 | 17.11 | 99.60 |
| ps_soft_1.5b | test_unseen | 2408 | 18.56 | 66.46 | 66.40 | 76.95 | 0 | 17.99 | 100.00 |
| ps_soft_1.5b | friend_clean | 245 | 14.69 | 75.07 | 75.80 | 66.53 | 0 | 0.00 | – |
| ps_hard_1.5b | test_seen | 3010 | 18.50 | 65.89 | 65.89 | 99.97 | 0 | 16.62 | 99.59 |
| ps_hard_1.5b | test_unseen | 2408 | 18.56 | 67.24 | 67.23 | 99.17 | 0 | 17.00 | 100.00 |
| ps_hard_1.5b | friend_clean | 245 | 15.10 | 75.48 | 75.48 | 100.00 | 0 | 0.00 | – |
| paper_sft_1.5b | test_seen | 3010 | 23.12 | 65.43 | 65.20 | 94.42 | 0 | 32.34 | 40.05 |
| paper_sft_1.5b | test_unseen | 2408 | 16.32 | 65.10 | 64.50 | 86.13 | 0 | 17.45 | 37.48 |
| paper_sft_1.5b | friend_clean | 245 | 11.84 | 66.15 | 70.44 | 82.86 | 0 | 0.00 | 0.00 |
| paper_grpo_1.5b | test_seen | 3010 | 11.06 | 64.05 | 64.37 | 87.14 | 0 | 1.32 | 34.55 |
| paper_grpo_1.5b | test_unseen | 2408 | 10.17 | 64.23 | 64.28 | 90.74 | 0 | 0.00 | 0.00 |
| paper_grpo_1.5b | friend_clean | 245 | 13.88 | 74.58 | 74.19 | 97.96 | 0 | 0.00 | – |
| paper_score_grpo_1.5b | test_seen | 3010 | 15.08 | 64.61 | 64.82 | 86.31 | 0 | 11.84 | 40.52 |
| paper_score_grpo_1.5b | test_unseen | 2408 | 11.25 | 64.36 | 64.38 | 88.87 | 0 | 3.71 | 31.78 |
| paper_score_grpo_1.5b | friend_clean | 245 | 13.06 | 73.28 | 72.53 | 93.06 | 0 | 0.00 | 0.00 |

延迟基准固定 500 次请求、10 次热身、3/7/11 台轮流。学生为 compact/cache on，原作者为 paper/cache off。
不同格式代表各自的完整部署配置。复用两个学生旧基准，新增缺失六组；另补测 SFT 使三个原作者模型使用同一版贪心参数。旧 SFT 结果仍保留。

| 模型 | p50 ms | p99 ms | 复用旧结果 | 运行目录 |
|---|---:|---:|---|---|
| main_0.5b | 55.04 | 71.97 | True | runs/20261106T130530Z_bench_comllm-0.5b-soft-T0.3-q8_0_compact_cacheon |
| soft_1.5b | 117.98 | 151.04 | True | runs/20261106T130817Z_bench_comllm-1.5b-soft-T0.3-q8_0_compact_cacheon |
| hard_1.5b | 112.48 | 145.20 | False | runs/20261113T040906Z_bench_comllm-1.5b-hard-q8_0_compact_cacheon |
| ps_0.5b | 51.55 | 105.30 | False | runs/20261113T041530Z_bench_comllm-0.5b-soft-T0.3-ps-q8_0_compact_cacheon |
| ps_soft_1.5b | 112.39 | 145.38 | False | runs/20261113T042652Z_bench_comllm-1.5b-soft-T0.3-ps-q8_0_compact_cacheon |
| ps_hard_1.5b | 112.55 | 145.29 | False | runs/20261113T043835Z_bench_comllm-1.5b-hard-ps-q8_0_compact_cacheon |
| paper_sft_1.5b | 911.40 | 1024.89 | False | runs/20261113T081529Z_bench_comllm-paper-sft-1.5b-q8_0_paper_cacheoff |
| paper_grpo_1.5b | 911.51 | 1161.18 | False | runs/20261113T065012Z_bench_comllm-paper-grpo-1.5b-q8_0_paper_cacheoff |
| paper_score_grpo_1.5b | 910.64 | 1161.08 | False | runs/20261113T080731Z_bench_comllm-paper-score-grpo-1.5b-q8_0_paper_cacheoff |

原始预测、配置和校验值见各运行目录；来源映射见 provenance.json。
