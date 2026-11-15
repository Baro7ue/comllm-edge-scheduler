# 版本钉死

[English](versions.md)

记录日期 2026-11-01。11/21 前这些版本不升级；要换，先停下来商量。

| 组件 | 版本 | 用在哪 | 怎么查 |
|---|---|---|---|
| JetPack | 7.2.1（L4T R39.2.1） | Jetson | head -n 1 /etc/nv_tegra_release |
| CUDA | 13.2（nvcc V13.2.86） | Jetson | /usr/local/cuda/bin/nvcc --version |
| llama.cpp | 5aeac37a3 | Jetson 推理；Colab 上转 GGUF 也用这个提交 | git -C ~/llama.cpp rev-parse --short HEAD |
| transformers | 5.20.2 | Colab 训练 | pip show transformers |
| trl | 1.16.0 | 阶段 3 没用到：训练循环是 notebook 里手写的 | pip show trl |
| peft | 0.21.2 | Colab 训练 | pip show peft |
| Qwen2.5-0.5B-Instruct | 7ae557604adf67be50417f59c2c2f167def9a775 | Colab 训练基座（Hugging Face 提交号） | 模型信息接口的 sha 字段 |
| Qwen2.5-1.5B-Instruct | 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 | Colab 训练基座（Hugging Face 提交号） | 同上 |
| Python（Mac 环境节点） | Python 3.14.8 | 仿真器、环境节点 | python --version |
| numpy、requests 等 | 见 requirements.txt | Mac；阶段 5 的 RK3588 照装 | pip freeze |
| Python（RK3588 环境节点） | 阶段 5 预检时填（代码在 3.8.20 + numpy 1.24 上测过） | 两板复测 | python3 --version |

Jetson 的两项已在 evidence/G0/jetson_versions.txt 里现场核对。Colab 上的 torch 用预装版本，阶段 3 开训时补进这张表：

| 组件 | 版本 | 用在哪 | 怎么查 |
|---|---|---|---|
| torch（Colab 预装） | 2.11.0+cu130 | Colab 训练、论文原版基线 | 训练 notebook 第 2 格的输出 |
| Colab GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition（运行时选 G4），bf16 | 同上 | 训练 notebook 第 2 格的输出 |
| Colab GPU（阶段 4 的 B） | NVIDIA A100-SXM4-80GB，bf16，torch 同上 | hard_1.5b 加 GRPO（runs/s4/hard_grpo_20261105T123733Z） | 该运行 summary.json 的 gpu 字段 |
