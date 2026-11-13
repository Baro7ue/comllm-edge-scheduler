| 模型 | 提示词与输出 | 前缀缓存 | 提示词 token（缓存命中） | 生成 token | p50 | p99 | 预填充 | 解码 | llama-server 其他 | Python 服务 | 网络+HTTP | N_max | 次数 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5b-q4_k_m | 论文格式长提示词，生成动作名 | off | 1351（0） | 12.0 | 1064.2 | 1339.5 | 778.4 | 272.2 | 9.1 | 0.9 | 2.4 | 0.3 | 500 |
| comllm-paper-grpo-1.5b-q8_0 | 论文格式长提示词，生成动作名 | off | 1358（0） | 6.9 | 911.5 | 1161.2 | 701.9 | 199.7 | 9.1 | 0.9 | 4.4 | 0.4 | 500 |
| comllm-paper-sft-1.5b-q8_0 | 论文格式长提示词，生成动作名 | off | 1358（0） | 5.7 | 911.4 | 1024.9 | 701.2 | 160.9 | 9.2 | 0.8 | 2.6 | 0.4 | 500 |
| comllm-paper-score-grpo-1.5b-q8_0 | 论文格式长提示词，生成动作名 | off | 1358（0） | 6.9 | 910.6 | 1161.1 | 702.0 | 201.2 | 9.1 | 0.9 | 2.4 | 0.4 | 500 |
| 1.5b-q4_k_m | 压缩提示词，单 token | off | 278（0） | 1.0 | 190.3 | 240.7 | 179.3 | 0.0 | 7.7 | 0.7 | 1.7 | 1.8 | 500 |
| 1.5b-q4_k_m | 压缩提示词，单 token | on | 278（133） | 1.0 | 119.0 | 164.8 | 105.1 | 0.0 | 7.8 | 0.7 | 1.6 | 2.9 | 500 |
| comllm-1.5b-soft-T0.3-q8_0 | 压缩提示词，单 token | on | 278（133） | 1.0 | 118.0 | 151.0 | 101.8 | 0.0 | 7.7 | 0.7 | 1.4 | 3.0 | 500 |
| 1.5b-q8_0 | 压缩提示词，单 token | on | 278（133） | 1.0 | 115.7 | 148.0 | 99.2 | 0.0 | 7.9 | 0.6 | 1.5 | 3.1 | 500 |
| comllm-1.5b-hard-ps-q8_0 | 压缩提示词，单 token | on | 278（133） | 1.0 | 112.6 | 145.3 | 96.5 | 0.0 | 7.6 | 0.6 | 1.7 | 3.1 | 500 |
| comllm-1.5b-hard-q8_0 | 压缩提示词，单 token | on | 278（133） | 1.0 | 112.5 | 145.2 | 96.6 | 0.0 | 7.6 | 0.6 | 1.5 | 3.1 | 500 |
| comllm-1.5b-soft-T0.3-ps-q8_0 | 压缩提示词，单 token | on | 278（133） | 1.0 | 112.4 | 145.4 | 96.4 | 0.0 | 7.6 | 0.7 | 1.6 | 3.1 | 500 |
| comllm-0.5b-soft-T0.3-q8_0 | 压缩提示词，单 token | off | 278（0） | 1.0 | 77.2 | 95.2 | 68.1 | 0.0 | 7.6 | 0.7 | 1.6 | 4.3 | 500 |
| comllm-0.5b-soft-T0.3-q4_k_m | 压缩提示词，单 token | on | 278（133） | 1.0 | 57.4 | 80.7 | 48.2 | 0.0 | 7.7 | 0.7 | 1.4 | 5.7 | 500 |
| comllm-0.5b-soft-T0.3-q8_0 | 压缩提示词，单 token | on | 278（133） | 1.0 | 55.0 | 72.0 | 44.2 | 0.0 | 7.4 | 0.6 | 1.5 | 6.2 | 500 |
| 0.5b-q8_0 | 压缩提示词，单 token | on | 278（133） | 1.0 | 52.5 | 69.0 | 41.9 | 0.0 | 7.5 | 0.6 | 1.3 | 6.5 | 500 |
| comllm-0.5b-soft-T0.3-ps-q8_0 | 压缩提示词，单 token | on | 278（133） | 1.0 | 51.6 | 105.3 | 41.6 | 0.0 | 7.4 | 0.6 | 4.3 | 6.2 | 500 |
| formula | 公式（不调 LLM） | on | 0（0） | 0.0 | 1.1 | 1.3 | 0.0 | 0.0 | 0.0 | 0.3 | 0.9 | 291.5 | 500 |

时间单位都是 ms。端到端 p50/p99 在环境节点上计时；后面五列是平均值，加起来等于端到端平均。预填充、解码是 llama-server 自报的；llama-server 其他 = 分词、采样、拼 JSON；Python 服务 = 还原状态、拼提示词、解析答案；网络+HTTP = 网线往返和两端的 HTTP 处理。N_max = 1 / (3 × 端到端平均秒数)，即每台终端每秒 3 个请求时调度节点能撑住的终端数。

运行目录：
- 20261103T135032Z_bench_1.5b-q4_k_m_paper_cacheoff
- 20261113T065012Z_bench_comllm-paper-grpo-1.5b-q8_0_paper_cacheoff
- 20261113T081529Z_bench_comllm-paper-sft-1.5b-q8_0_paper_cacheoff
- 20261113T080731Z_bench_comllm-paper-score-grpo-1.5b-q8_0_paper_cacheoff
- 20261103T135957Z_bench_1.5b-q4_k_m_compact_cacheoff
- 20261103T140258Z_bench_1.5b-q4_k_m_compact_cacheon
- 20261106T130817Z_bench_comllm-1.5b-soft-T0.3-q8_0_compact_cacheon
- 20261103T140628Z_bench_1.5b-q8_0_compact_cacheon
- 20261113T043835Z_bench_comllm-1.5b-hard-ps-q8_0_compact_cacheon
- 20261113T040906Z_bench_comllm-1.5b-hard-q8_0_compact_cacheon
- 20261113T042652Z_bench_comllm-1.5b-soft-T0.3-ps-q8_0_compact_cacheon
- 20261106T130641Z_bench_comllm-0.5b-soft-T0.3-q8_0_compact_cacheoff
- 20261106T130733Z_bench_comllm-0.5b-soft-T0.3-q4_k_m_compact_cacheon
- 20261106T130530Z_bench_comllm-0.5b-soft-T0.3-q8_0_compact_cacheon
- 20261103T141102Z_bench_0.5b-q8_0_compact_cacheon
- 20261113T041530Z_bench_comllm-0.5b-soft-T0.3-ps-q8_0_compact_cacheon
- 20261106T125957Z_bench_formula_formula_cacheon
