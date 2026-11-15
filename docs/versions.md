# Pinned versions

[简体中文](versions.zh-CN.md)

Recorded on 2026-11-01. These versions are not upgraded before 11/21; to change one, stop and discuss it first.

| Component | Version | Used for | How to check |
|---|---|---|---|
| JetPack | 7.2.1 (L4T R39.2.1) | Jetson | head -n 1 /etc/nv_tegra_release |
| CUDA | 13.2 (nvcc V13.2.86) | Jetson | /usr/local/cuda/bin/nvcc --version |
| llama.cpp | 5aeac37a3 | inference on the Jetson; the GGUF conversion on Colab uses the same commit | git -C ~/llama.cpp rev-parse --short HEAD |
| transformers | 5.20.2 | training on Colab | pip show transformers |
| trl | 1.16.0 | not used in stage 3: the training loop is written by hand in the notebook | pip show trl |
| peft | 0.21.2 | training on Colab | pip show peft |
| Qwen2.5-0.5B-Instruct | 7ae557604adf67be50417f59c2c2f167def9a775 | base model for training on Colab (Hugging Face commit) | the sha field of the model info API |
| Qwen2.5-1.5B-Instruct | 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 | base model for training on Colab (Hugging Face commit) | same as above |
| Python (Mac environment node) | Python 3.14.8 | simulator, environment node | python --version |
| numpy, requests and others | see requirements.txt | Mac; the RK3588 of stage 5 installs the same | pip freeze |
| Python (RK3588 environment node) | to be filled in at the stage 5 preflight (the code was tested on 3.8.20 + numpy 1.24) | two-board rerun | python3 --version |

The two Jetson entries were checked on the board in evidence/G0/jetson_versions.txt. torch on Colab is the preinstalled version, added to this table when stage 3 training started:

| Component | Version | Used for | How to check |
|---|---|---|---|
| torch (preinstalled on Colab) | 2.11.0+cu130 | training on Colab, the paper's original baseline | output of cell 2 of the training notebook |
| Colab GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (runtime type G4), bf16 | same as above | output of cell 2 of the training notebook |
| Colab GPU (B of stage 4) | NVIDIA A100-SXM4-80GB, bf16, same torch | GRPO on hard_1.5b (runs/s4/hard_grpo_20261105T123733Z) | the gpu field of that run's summary.json |
