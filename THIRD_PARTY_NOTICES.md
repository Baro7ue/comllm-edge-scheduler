# Third-party material

The project's own code, documents, figures and run results are under the Apache License 2.0
([LICENSE](LICENSE)). The material below comes from other people and keeps its own terms.

## In this repository

| Path | What | Source | Terms |
|---|---|---|---|
| `sim/render.py` (`PAPER_INSTRUCTION` and the field layout of the paper format), `deploy/prompts.py` (the paper-format message layout) | the prompt template of the paper: the instruction text and the way the input fields are laid out, reproduced so that the original authors' models are asked exactly as they were trained | Yang, Cheng and Zhang, [arXiv:2604.07148](https://arxiv.org/abs/2604.07148); [Cassandra-Cheng/COMLLM-MEC-Task-Offloading](https://github.com/Cassandra-Cheng/COMLLM-MEC-Task-Offloading) | the authors'; see their repository |
| `train/grpo_reward.py` (the `repo` reward), `train/s7_reward.py` | rewards written for this project to reproduce the behaviour of the authors' training code: its parser, thresholds, weights and constants. `train/s7_reward.py` names the files and the commit (`17adede`) it follows | same repository | this project's code is Apache-2.0; the method and constants are the authors' |
| `sim/cost.py`, `sim/env.py` | the delay and cost formulas of the paper (eq. 4–7, 26–29) and its Table I parameters, implemented for this project | the paper | Apache-2.0 (this project's code) |

## Used but not included

| What | Used for | Source | Licence |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct | base models of the students; the LoRA adapters and GGUF files made from them are not in the repository | [Qwen2.5](https://github.com/QwenLM/Qwen2.5), Hugging Face `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct`, revisions in [docs/versions.md](docs/versions.md) | Apache-2.0 |
| The original authors' models: 1.5B SFT, GRPO and score-GRPO, and SFT-7B | comparison, and the start point of the 7B reproduction | shared by the authors of the paper for this comparison | not public; not distributed here |
| The original authors' dataset | the cleaned test set and the GRPO training prompts | shared by the authors of the paper | not public; not distributed here |
| llama.cpp (commit `5aeac37a3`) | inference on the Jetson and GGUF conversion | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT |
| Hugging Face Transformers, PEFT | training and full-precision evaluation on Colab | [huggingface/transformers](https://github.com/huggingface/transformers), [huggingface/peft](https://github.com/huggingface/peft) | Apache-2.0 |
| Python packages in `requirements.txt` | figures and the local tools | PyPI | each package's own licence |

Results computed with the authors' models and dataset are in `runs/` and `evidence/` as predictions,
scores and logs. The models, the dataset and its prompts are not. Tests that need the dataset skip
themselves when it is missing.

## Read but not copied

- The queueing model of the simulator follows Tang and Wong, [arXiv:2005.02459](https://arxiv.org/abs/2005.02459).
