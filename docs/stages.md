# Method, commands and full results for every stage

[简体中文](stages.zh-CN.md)

The [README](../README.md) only gives the conclusions. This document records, stage by stage, how each step was done, the commands, the full results and what to watch for, and it lists where this repository differs from the paper. Every number is measured; the run folder each number comes from is listed in [figures/numbers.md](../figures/numbers.md) and [docs/media/numbers.md](media/numbers.md). The commands below are run from the repository root with `.venv` activated.

## Differences from the paper

The table lists where this repository differs from the paper (arXiv:2604.07148v2) and why. Equation and table numbers refer to the paper.

| Item | Paper | This repository | Reason and effect |
|---|---|---|---|
| Queueing time on an edge server | Eq. 4 says a server shares its capacity among the active tasks; eq. 5–7 approximate the queueing time as (backlog + task size) × (active tasks + 1) × γ ÷ F, where the backlog is that of the whole server | The simulator advances every task under processor sharing; scheduling decisions and training labels still use the paper's formula | Under processor sharing the backlog is counted twice (derivation in [paper-delay-formula.md](paper-delay-formula.md)), so the approximation is pessimistic in the closed loop at high load and the one-step optimum keeps too many tasks on the terminal. With 3 servers and 40 terminals the one-step optimum drops 6.3%; an estimate derived from processor sharing drops 0.01% (measured in simulation) |
| Evaluation | Eq. 26–27: score 1000 independent test states one by one and average (open loop) | Closed loop: every decision changes the queues seen later, and delay and drops are counted from the queueing that actually happens in the simulation | The open loop cannot see the bias of the estimate, the herding of several decisions onto one server, or the time the decision itself takes |
| Server counts in training | Section V-B: train on 6 servers only and test on 3/5/7/9/11 | Train on a mix of 3/5/7/9/11 servers; 4/6/8/10 never appear in training and are used to test generalization | The data set provided by the original authors is itself a mix of 3/5/7/9/11 servers |
| Local CPU frequency | Table I: 2 GHz | 2.5 GHz | Follows the original authors' data set |
| Deadline penalty ρ | Table I: 10 | The data say 100, which is only written out as is in the paper-format prompt; in evaluation the mean delay counts completed tasks only and the drop rate is reported separately | Delay and drops are reported separately, so the results do not depend on ρ |
| Task arrival | The text of Section III-A says Poisson arrivals; Table I gives an arrival probability of 0.3 | Each terminal produces a task with probability 0.3 in each time slot | Consistent with Table I and the data set |
| Uplink rate | Eq. 1 varies with the channel by the Shannon formula; Table I gives a mean of 14 Mbps | Fixed 14 Mbps (1.4 Mbits per slot) | Every uplink rate in the data set is 1.4 Mbits per slot |
| Prompt and answer | Section IV-C: about 500 tokens of instructions + about 115 tokens of state per server; the answer is an action name in Chinese (about 12 tokens) | Compact format: about 130 tokens of system instructions + one line of numbers per server; the answer is 1 letter (A = local, B = server 1, ...) | A decision takes 1 token and no generation. The 1.5B Q4_K_M on the Jetson goes from 1064 ms to 119 ms (measured in stage 2) |
| Training objective and data size | Eq. 16: negative log-likelihood on every token of the whole answer; SFT uses 1000 samples (Section V-A) | KL(soft label ‖ model) at the single answer position only, soft label = softmax(−cost / T); 40,000 samples | Soft labels also tell the model which actions are nearly as good; the data come from our own simulator, so the amount is not limited |
| Future impact in LACS | Eq. 20 first subtracts one time slot of service in the virtual transition; eq. 21 samples K = 3 future tasks at random | No subtraction, as in the original authors' repository; the average over 41 evenly spaced points between 0.5 and 1.5 times the task size | Removes the sampling noise. With λ = 0.3, LACS changes the decision in at most 0.80% of the simulated states and the closed-loop delay differs by at most 0.18% (measured in stage 1), so the soft labels use the one-step cost only |

Drops are decided as in the paper: by eq. 27, a task whose delay exceeds the deadline of 10 slots counts as dropped. The evaluation script in the original authors' repository uses 12.5 slots; keep this in mind when comparing numbers.

## Simulator (stage 1)

Run from the repository root with `.venv` activated:

```bash
python -m unittest discover -s tests -v                 # 42 tests, about 40 seconds; the original authors' data are read from ~/Documents/COMLLM训练部署及改良/from_friend/data by default
python -m eval.closed_loop --name s1_sweep_3srv         # sweep the number of terminals N; results go to runs/<UTC>_s1_sweep_3srv/results.csv
python -m eval.plot_sweep runs/<folder>/results.csv     # plot mean delay and drop rate against N
python -m eval.lacs_check_sim --sweep runs/<folder>/results.csv   # rerun the LACS check on the simulator's state distribution
python -m eval.show_prompts                             # show the two prompt formats
```

All parameters are in `sim/config.py`, with the values of the original authors' three data sets (1914 samples): a time slot of 0.1 s, a deadline of 10 slots, tasks uniform between 2 and 5 Mbits, a task with probability 0.3 per terminal per slot, an uplink of 1.4 Mbits per slot (14 Mbps), a local CPU of 2.5 GHz (Table I of the paper says 2 GHz; the data take precedence), and 297 cycles per bit. The server line-ups for 3/5/7/9/11 servers come from the data; those for 4/6/8/10 are our own, never appear in training and are used only to test generalization.

Modelling choices:

- **Servers share their capacity among tasks (processor sharing).** With n tasks on a server of frequency F at the same time, each task gets F/n. The one-step formula used for scheduling decisions follows the paper: it assumes the number of active tasks grows by 1 when the new task joins, so the task gets F/(n+1). This differs from Tang & Wong (IEEE TMC 2022, the source of the paper's environment): they keep one FIFO queue per terminal on each edge server and share the capacity among the non-empty queues, so several tasks from one terminal to one server are processed one after another; here the capacity is shared per task, and tasks from the same terminal also run in parallel.
- **The local CPU and the uplink are one FIFO queue each per terminal.** A task not finished within 10 slots of its arrival is dropped and leaves its queue at once.
- **The server state the scheduler sees includes tasks that are assigned but still uploading.** The scheduler knows which tasks it has sent, and they will soon share the server's capacity.
- **Several tasks in one time slot can be decided in two ways:** `sequential` (one at a time, each seeing the result of the previous one) and `batch` (all see the same state from the start of the slot, which imitates batched LLM inference and produces herding). The formula baselines use `sequential` by default.
- **Metrics:** the mean delay counts completed tasks only (unit: slots); the drop rate is computed separately; load balance is the Jain index over the number of tasks offloaded to each server, with n the real number of servers (the script in the original authors' repository uses the largest server index that was ever chosen).

## Scheduler node and real-time closed loop (stage 2)

```
Environment node (Mac in development, RK3588 in stage 5)       Scheduler node (Jetson Orin Nano)
  eval/realtime.py: advances the simulation by wall clock ──HTTP──▶  deploy/scheduler.py (port 8100)
  eval/bench_latency.py: sends N requests, measures latency ◀──────    │ build prompt → llama-server (port 8080, localhost only)
  all timestamps are taken on this side                                │ timeout (100 ms by default) or unusable answer → one-step formula
```

The addresses, ports and timeout of the two boards are written only in `deploy/testbed.json`.

On the Jetson (repository root, `.venv` activated):

```bash
python -m deploy.scheduler --model ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf   # starts llama-server by itself
```

On the Mac:

```bash
python -m eval.bench_latency --fmt compact --cache on     # end-to-end latency and per-segment time of 500 requests
python -m eval.latency_table                              # collect all benchmarks into one table
python -m eval.realtime --policy compact --n 5 --minutes 10 --name smoke_llm   # real-time closed loop
```

Design choices:

- **The compact format computes 1 token and reads probabilities instead of using a grammar.** The request sets `n_predict=1` and `n_probs=20` and takes the most probable legal letter among the top 20 candidates. A GBNF grammar also guarantees a legal answer, but it checks all 150,000 vocabulary tokens one by one, and when the model's first choice is not a legal letter each token costs about 20 ms more (measured on a cloud CPU, not yet on the Jetson), so it is kept only as a control.
- **The paper format follows the prompt layout of the original authors' repository and generates until the end token, at most 20 tokens.** The system prompt and user content are the same as for the paper's original baseline in stage 3 (see "The paper's original baseline" below); after stripping whitespace the answer must be exactly "本地执行" or "卸载到边缘服务器k", otherwise it is treated as unusable and the formula answers. The paper-format latency of stage 2 was measured with the old layout (the state dictionary as JSON, 12 generated tokens fixed); stage 5 measures it again with the new layout.
- **A decision takes effect at the moment the reply arrives.** In real-time mode a task's clock starts at its arrival, and the time spent waiting for the decision (queueing at the environment node + network + Jetson) counts towards the task's delay; a task that has no decision by its deadline is recorded as dropped at the "decision" stage.
- **The environment node sends one request at a time.** The state each request sees includes the results of all earlier decisions; while the Jetson is busy, requests queue at the environment node. The theoretical capacity is N_max = 1 / (3 × mean round trip in seconds): each terminal produces 0.3 × 10 = 3 requests per second.
- **Fallback on timeout.** The service on the Jetson waits for the LLM for at most `timeout_ms`; after that it answers with the one-step formula and logs `source = fallback`, `reason = timeout`. An LLM request that has already started runs to the end and its result is recorded as "late", without affecting the answer already given. If a new request arrives while a late one is still running, it does not queue and is answered by the formula at once (`reason = busy`): otherwise it would time out behind the late one too, every such request would wait `timeout_ms` for nothing, and since 100 ms is longer than the mean gap between two requests with 5 terminals (about 68 ms), the tasks waiting for a decision would pile up. In a cloud rehearsal the CPU stalled once and exactly this chain of timeouts appeared, raising the p99 decision latency to about 300 ms (measured in the cloud); this rule was added for it.

## Training and open-loop evaluation (stage 3)

```
Mac        python -m train.make_data          → data/s3/ (6 files, not in the repository) → upload to MyDrive/COMLLM/s3_data/ on Drive
Colab      train/S3_训练.ipynb                 RUN = sweep_0.5b → main_0.5b → soft_1.5b → hard_1.5b (optionally ps_0.5b)
           train/S3_论文原版基线.ipynb          the original authors' sft / grpo / score-grpo, paper format
           results in MyDrive/COMLLM/s3_runs/, downloaded to runs/s3/
Mac        python -m train.small_net (optional) → runs/s3/small_net_<UTC>/
Mac        python -m eval.open_loop runs/s3/*  → evidence/G3/open_loop.md and open_loop.csv
Jetson+Mac python -m eval.check_gguf --expected runs/s3/<folder>/jetson_check/expected_200.jsonl --name <name>
```

**Data.** One closed-loop simulation is run for each (server count, number of terminals N, behaviour policy, seed); after a warm-up of 200 slots, states are taken at intervals and deduplicated by the full prompt text. Any training state that appears in an evaluation set is removed (the overlap is 0, as recorded in `meta.json`).

| File | Server counts | Per server count | Behaviour policy | Use |
|---|---|---|---|---|
| train | 3/5/7/9/11 | about 20,000 | 60% greedy, 20% greedy with 20% random actions, 20% PS-aware greedy | training (40,000 drawn per run) |
| val | 3/5/7/9/11 | about 400 | greedy | choosing the temperature T, watching the training curve |
| test_seen | 3/5/7/9/11 | about 600 | greedy | test |
| test_unseen | 4/6/8/10 | about 600 | greedy | generalization test: server counts never seen in training |
| friend_clean | 3/5/7/9/11 | 245 in total | original authors | the original authors' 306 test samples minus the 61 that also appear in their training set |

**Labels and loss.**

- Soft label p_a = softmax(−(c_a − c_min) / T), over legal actions only; T is chosen from 0.1 / 0.3 / 1.0 as the one with the smallest extra delay on the validation set, with hard labels (the best action only) as the control.
- Labels are computed from the state at the precision of the compact format (2 decimals for the task size, 1 for the rest), because that is the precision the student sees; evaluation uses the cost of the state with 4 decimals. The states where the two give different best actions are the student's ceiling, shown as the row "formula · compact precision" in the open-loop table.
- Loss = KL(p_teacher ‖ p_student), at the single answer position only. The student's probabilities are a softmax over the whole vocabulary, of which the 12 letter entries are taken: probability that leaks to non-letter tokens makes the letter probabilities smaller and the loss larger, so the model also learns to answer with a letter only.
- LoRA: r = 16, α = 32, dropout 0.05, all linear layers; learning rate 2e-4, linear warm-up over the first 3% followed by cosine decay; 16 samples per update, bf16.
- Training and evaluation pick the action by the same rule as deployment: among the 20 most probable tokens of the whole vocabulary, the most probable legal letter; if there is none, the answer is illegal. The notebook checks that the prompt built by the Hugging Face chat template is character for character the one built by `deploy/prompts.py`.

**The paper's original baseline.** The original authors' three 1.5B models use the prompt layout of the training and evaluation scripts in their repository (`stf_7B.py`, `comllm_train.py`, `evaluate_model.py`): the system prompt "你是一个MEC任务卸载专家……", and user content = instructions + an empty line + "**输入数据**：" + one "key: value" per line. Generation is greedy with at most 20 tokens, and the first "本地执行 / 卸载到边缘服务器k" found in the answer is the action; a k beyond the number of servers is illegal. The original authors' evaluation script samples at temperature 0.7; here everything is greedy so that the results can be reproduced. The notebook first tries this layout and 4 others (Qwen's default system prompt + the state dictionary as JSON, and so on) with the sft model on 60 samples, and the one that agrees most often with the original authors' labels is used for the whole evaluation (`probe.json`).

**Open-loop metrics** (`eval/open_loop.py`, unit: slots). Aligned with Table II of the paper: AL here includes the deadline penalty ρ = 10 (Table I) as in eq. 26, and an illegal action counts as deadline + ρ = 20; PR = AL of the best action ÷ AL of the policy (eq. 28–29); a drop = cost above the deadline or an illegal action (the open-loop version of eq. 27); LBI follows eq. 31, over offloaded tasks only, with n the real number of servers, computed per server count and then averaged. The closed-loop evaluation (stage 5) still follows the row "Deadline penalty ρ" above and reports delay and drops separately.

**Small-network control (optional).** `python -m train.small_net` (Mac, numpy, about 20 seconds): one 6 → 32 → 32 → 1 MLP scores every option, 1313 parameters in total; its features are recovered from the compact prompt, so it sees the same information as the LLM students; it is trained on the same soft labels and writes the same `preds.csv`, which adds one row to the open-loop table.

**GGUF.** After merging the LoRA, `convert_hf_to_gguf.py` at the same commit as the Jetson (5aeac37a3) writes Q8_0 directly; for Q4_K_M it writes F16 first and `llama-quantize` quantizes it. `eval/check_gguf.py` sends 200 test states to the scheduler service on the Jetson and compares the actions with those of the merged model on Colab (PyTorch); where they differ it also reports how many extra slots the Jetson's choice costs, because states with a very small lead are the easiest for quantization to flip, and a wrong choice costs little on those states.

## GRPO experiments (stage 4)

```
Mac    python -m train.make_grpo_data      → data/s4/ (not in the repository) → upload to MyDrive/COMLLM/s4_data/ on Drive
Colab  train/S4_A_论文原版.ipynb            RUN = diagnose → smoke → fixed → repo (from the original authors' sft 1.5B, paper format)
       train/S4_B_硬标签起点.ipynb           from hard_1.5b of stage 3, compact format
       results in MyDrive/COMLLM/s4_runs/, downloaded to runs/s4/
Mac    python -m eval.open_loop <run folders of stages 3 and 4> --out evidence/G4 --title "阶段 4 开环评测"
Mac    python -m eval.grpo_report runs/s4/*  → evidence/G4/grpo_report.md and grpo_curves.png
Mac    python -m eval.s4_breakdown ...       → evidence/G4/breakdown.md
```

**Method.** A GRPO loop written by hand (eq. 17–19), without a ready-made trainer. 8 answers are sampled per prompt (temperature 1.3, top-p 0.95) and normalized within the group into advantages (eq. 18). Each batch of answers is used for one update only, so ρ = 1 and the clip has no effect. The KL uses the k3 estimator with β = 0.04, and the reference model is the starting point with the LoRA switched off. LoRA r = 16, learning rate 5e-5; λ = 0 (no LACS), as for the paper's 1.5B GRPO (V-A). Two rewards are compared: fixed gives an illegal answer the effective cost of the worst legal action plus 5 slots; repo is rewritten after the rule in the original authors' repository, where an illegal answer scores 5 and a legal one gets an extra 1. The training prompts of A are the 830 left from the original authors' GRPO data after deduplication and removal of test-set states, with another 100 kept for validation; B uses the training set of stage 3, with the reward computed from the cost at compact precision.

**Findings (all measured; the decoding is in parentheses).**

- Accuracy of the original authors' three 1.5B models (sft / grpo / score-grpo) on the 245 samples of their cleaned test set: 13.5% / 13.9% / 13.5% (greedy), 10.6% / 11.0% / 11.0% (most probable whole sentence), 12.4% / 12.7% / 12.6% (expectation under sampling at 0.7). Choosing at random gives 15.9%.
- The bias towards server 1 is already in sft. Share sent to server 1: sft 38.3%, grpo 39.9% (expectation under sampling at 0.7), 83.7%, 98.4% (greedy). Server 1 is really the best in only 13.9%. In the sft labels servers 1, 2 and 3 each take 12.3%, but action names whose first digit is "1" take 22.1%, because servers 1, 10 and 11 share that digit.
- The GRPO-1.5B of Table V in the paper (62.7% to server 1, 3.9% local) was not reproduced with the same number. On the 36 samples with 11 servers in the authors' cleaned test set, the original authors' grpo sends 18.3% (expectation under sampling at 0.7) and 91.7% (greedy) to server 1. The test states of the two sides differ.
- GRPO from the original authors' sft for 204 steps (about 40 minutes). PR on the authors' cleaned test set 70.4% → 96.5% (fixed) / 99.1% (repo), accuracy 13.5% → 81.2% / 87.8%, all greedy. At step 25 server 1 takes 88–89% and local 0% on the validation set (greedy), and the model leaves that state by itself afterwards. The same 1.5B reaches this level, so our measurements do not support the explanation that 1.5B lacks capacity.
- The repo reward is not worse than fixed here. Its loophole is that an out-of-range server number scores higher than local, but it was not triggered this time: out-of-range answers are 0% in both training runs. The repo model has 45/1245 answers with an extra "服务器" tail, and with strict parsing its accuracy on the cleaned test set is 86.9% (greedy).
- Neither model ever chooses local, and local is best in only 5.9% of the training prompts. On the 1000 samples with server counts seen in stage 3, PR is 86.0% / 89.3% (greedy), and the prompts where local is best account for 77.9% / 92.5% of the extra delay.
- 1000 more steps of GRPO (fixed reward) on hard_1.5b of stage 3: accuracy on seen server counts 96.0% → 92.9%, on unseen ones 95.2% → 92.1%, PR 99.9% → 99.7% (deployment rule). The student is already at the formula ceiling and GRPO brings no gain. The deployed model is still the soft-label 0.5B Q8_0 of stage 3.

## Closed-loop evaluation (stage 5)

```
Colab   train/S5_论文原版转GGUF.ipynb        the original authors' sft 1.5B → comllm-paper-sft-1.5b-q8_0.gguf (MyDrive/COMLLM/s5/gguf/)
Jetson  python -m deploy.scheduler --model-key main_0.5b       the model table is `models` in deploy/testbed.json; POST /model switches models
Mac     python -m eval.bench_latency --fmt compact --cache on --model-key main_0.5b      experiment 2, latency breakdown
Mac     python -m eval.check_paper_gguf --preds runs/s3/paper_<UTC>/sft/preds.csv       the paper-original GGUF against PyTorch
Mac     python -m eval.s5_sweep --plan e1,e1_paper,e1_shed      experiment 1, stability boundary + 1 point for the paper original + the shedding control (real-time closed loop, 51 points, about 2.9 hours, estimated)
Mac     python -m eval.virtual_loop --plan e3                   experiment 3, herding (formulas only, a few minutes locally)
Mac     python -m eval.virtual_loop --plan e4a,e4b,e5           experiments 4 and 5 (every decision asks the Jetson, 71 points, about 3 hours, estimated)
Mac     python -m eval.virtual_loop --plan e6                   experiment 6, whether the look-ahead of LACS helps in the closed loop (formulas only, 252 points, about 25 minutes, estimated)
Mac     python -m eval.figures                                  7 figures + figures/numbers.md (and twoboard.md once the two-board rerun has been done)
RK3588  python -m eval.s5_sweep --plan twoboard                 two-board rerun: env_node in testbed.json is changed to the RK3588
```

`s5_sweep` and `virtual_loop` both have `--dry-run` (only list the points and the expected duration), `--resume <folder>` (continue after an interruption), `--only <model key>` and `--rehearse` (no Jetson; the LLM is replaced by the local formula plus a pretended latency, to try the scripts).

Design choices:

- **Two closed loops.** The real-time closed loop (`eval/realtime.py`) runs on the wall clock and counts the decision latency in the task delay; it measures how many terminals the deployment can serve. The deployed model takes about 57 ms per decision (measured in stage 3) and only serves N ≈ 6, a load so light that all policies have about the same delay and decision quality cannot be told apart. Quality is therefore compared in the virtual-time closed loop (`eval/virtual_loop.py`): the simulation waits for the Jetson to answer every decision before moving on and does not count the decision time, so policies can be compared under the heavy load of N = 20–80. With formulas only, it gives exactly the results of `sim.env.run` (checked by a unit test).
- **Switching models needs nobody at the board.** `models` in `deploy/testbed.json` maps a model key to a GGUF file and a prompt format. `POST /model {"key": "<model key>"}` makes the scheduler service stop llama-server, restart it with the other file and warm it up; `/info` reports the model key, the file path and the file's sha256, and the config.json of every run records them. The sweep script groups the points by model and switches only when a new model's turn comes.
- **LACS can only be tested in the closed loop (experiment 6).** The open-loop AL counts the cost of this step only, so the one-step optimum is its lower bound, and the benefit of LACS, paying a little more now to leave room for later tasks, cannot show in the open loop. Experiment 6 compares the one-step optimum with LACS-greedy (λ = 0.3, 1, 2; `lacs:<λ>` changes λ) in the virtual-time closed loop with sequential decisions, plus PS-aware greedy as the reference for changing the formula; with batch decisions λ = 0.3 is compared once more, to see whether looking ahead reduces herding. The weights of COMLLM-7B are not available, so the optimum of the LACS reward stands in for it here.
- **Overload shedding (`--shed-ms`).** The environment node sends one request at a time, so once the number of terminals exceeds N_max the queue of tasks waiting for a decision keeps growing, every task waits past its deadline, and the drop rate jumps to nearly 100%. Shedding: a task that has queued at the environment node for longer than `shed_ms` no longer waits for the Jetson and is decided by the formula on the spot, recorded as `source = shed`. Off by default; `--plan e1_shed` is the control with shedding on, run together with experiment 1.
- **The paper's original 1.5B runs only one real-time point.** One decision takes about 1 second (measured in stage 2 with the old layout, measured again in stage 5), and 1 terminal already produces 3 requests per second, so N = 1 is overloaded; the real-time closed loop runs N = 1 only (`e1_paper`), and the quality comparison is left to the virtual-time closed loop (`e5_paper`, 400 slots). The soft-label 1.5B student takes about 120 ms per decision and would fall back to the formula most of the time with the deployed 100 ms timeout, so its timeout is 1000 ms.
- **One command redraws everything.** `evidence/G5/sources.json` records which run folders each figure uses; `python -m eval.figures` redraws all figures from it and writes the numbers needed for the résumé, with their run folders, to `figures/numbers.md`. Without this file it finds the latest folders by itself and writes the file.
- **RK3588 as the environment node.** The environment node's code depends on numpy only (`deploy/requirements-envnode.txt`) and was tested on Python 3.8.

**Results (11/7, all measured; experiments 3–6 are measured in simulation in the virtual-time closed loop).** The figures are in `figures/`, and every number with the run folder it comes from is in `figures/numbers.md`.

```
Experiment 1         runs/20261107T042911Z_s5_e1+e1_paper+e1_shed      real-time closed loop on the Jetson, 51 points
Experiment 2         runs/20261106T1*_bench_*                          latency of one decision, 6 configurations
Experiment 3         runs/20261107T071129Z_s5_virtual_e3               herding, 96 points
Experiments 4 and 5  runs/20261107T071637Z_s5_virtual_e4a+e4b+e5       generalization, PS labels, open loop against closed loop, 71 points
Experiment 6         runs/20261106T144248Z_s5_virtual_e6               LACS, 252 points
```

- **One decision (experiment 2, figure 2).** The deployed setup (0.5B Q8_0 + prompt cache) has a round trip of p50 55.0 ms and p99 72.0 ms, with states of 3/7/11 servers sent in turn. The original authors' 1.5B (paper layout, cache off) has p50 945.2 ms and p99 1043.9 ms. Experiment 1 has only 3 servers and a shorter prompt: the 0.5B averages 42.1 ms, and the original authors' 1.5B with the cache on has p50 472.9 ms.
- **Stability boundary (experiment 1, figure 1).** M = the largest number of terminals N at which the drops at the decision stage (no decision by the deadline) are below 1% for every seed.
  - Formula: M ≥ 80 (stable at the largest N measured). The drops from N = 40 on (6.1% at N = 40) come from 3 servers not being enough; the drops at the decision stage are 0.
  - The deployed 0.5B: M = 7, theoretical N_max = 1 ÷ (3 × 42.1 ms) = 7.9. The drop rate is 24.5% at N = 8 and 96.0% at N = 10.
  - The 0.5B with overload shedding is stable at the largest N measured, N = 20: drop rate 1.0%, with 39.5% of the tasks decided by the LLM and the rest by the formula on the spot.
  - The soft-label 1.5B: M = 3, N_max 3.9.
  - The original authors' 1.5B drops 93.4% at N = 1, of which 82.4% at the decision stage; N_max 0.7.
  - At N = 5 the mean delay of the 0.5B is 4.30 slots and that of the formula 3.53 slots. The extra 0.77 slots are mainly queueing for the decision (estimated).
- **Herding (experiment 3, figure 3).** With 11 servers and N = 80, changing the one-step optimum from batch to sequential decisions: mean delay 5.11 → 4.19 slots (−18.1%), drop rate 2.4% → 0.1%. The counterexample is 3 servers and N = 40: with sequential decisions the mean delay is 9.3% higher, the drop rate goes 1.7% → 6.2% and the local share 56.2% → 70.7%. The reason is that the paper's formula overestimates servers that queue by processor sharing (PS), and sequential decisions stack this overestimate one on another, pushing tasks back to the terminal. For PS-aware greedy, switching to sequential decisions lowers the drop rate at every point.
- **Generalization over server counts (experiment 4b, figure 4).** N = 40, the same seed. The mean delay of the deployed 0.5B differs from the one-step optimum by −0.1% to +0.5% over 3–11 servers, including the 4/6/8/10 never seen in training. In the closed loop it makes the same choice as the one-step optimum in 94.2%–99.1% of the decisions.
- **Open loop against closed loop (experiment 5, figure 5).** 3 servers, N = 40.
  - The five policies with an open-loop accuracy ≥ 98.8% (one-step optimum, LACS-greedy, small network, 0.5B, 1.5B) all have a closed-loop mean delay of 5.26–5.29 slots and drops of 5.9%–6.1%, and cannot be told apart.
  - PS-aware greedy has an open-loop accuracy of only 19.6% but is the fastest in the closed loop: 4.12 slots, 0.0% dropped.
  - Random looks no worse at 5.31 slots because it drops 22.9%, and dropped tasks do not count towards the delay.
  - The untrained 0.5B: 6.14 slots, 51.6% dropped.
- **PS labels (experiment 4a, figure 6 right).** 3 servers, the 0.5B trained with PS-aware greedy as the teacher against the main 0.5B:
  - Mean delay lower by 13.4% / 19.6% / 9.9% (N = 20 / 40 / 60).
  - Drop rate from 0.1% / 5.9% / 17.3% down to 0.0% / 0.0% / 0.9%.
  - The simulator itself queues by PS, so this model has a home advantage and must be measured again on real servers.
- **LACS (experiment 6, figure 7).** Sequential decisions, 12 points of 3/7/11 servers × N = 20–80, 3 seeds per point.
  - λ = 0.3 (the paper) makes the same choice as the one-step optimum 99.7% of the time. The mean delay differs by +0.04% on average (−0.07% to +0.21%), which is no change.
  - λ = 1: +0.18% on average (−1.72% to +1.99%); the difference comes from how exact ties between the two are broken.
  - λ = 2: between −11.78% and +26.92%: it helps at heavy load and is slower at light load (7 and 11 servers, N = 20).
  - PS-aware greedy: −9.27% on average (−22.73% to +3.79%).
  - Batch decisions, 11 servers, N = 80: λ = 0.3 gives 5.11 slots, the one-step optimum 5.09 slots. Looking ahead does not reduce herding.
- **An early GGUF sample of the original authors' SFT 1.5B.** On 200 states, the Q8_0 on the Jetson and PyTorch choose the same action in 183 (91.5%, `runs/20261107T042021Z_gguf_paper_sft`). This is a sample and does not establish equivalence on the full set; the complete results of this round's nine models on 5663 states are under stage 6 below. The closed-loop numbers on the Jetson reflect the deployed Q8_0 policy.

**When reading the numbers.**

- The closed-loop mean delay counts completed tasks only, so a policy that drops a lot looks fast; always read it together with the drop rate.
- Experiment 1 has 2 seeds and 3 minutes per point, and the original authors' 1.5B has only 1 seed. In experiments 4 and 5 the LLM points ran seed 0 only; they are compared with the formulas on the same seed, facing the same sequence of tasks.
- With λ = 1 there are many states where the two are exactly tied. Floating-point rounding differs between the Mac (ARM) and the cloud (x86), so ties are broken differently. Experiment 6 was run once on each: 216 of the 252 rows are identical bit for bit, the 36 that differ are all λ = 1, and the mean change is +0.18% (Mac) against +0.15% (cloud).
- Experiment 6 stopped once at point 31 and was finished on 11/7 with `--resume`. The version recorded in `sweep_config.json` for the resumed run is `4567c59-dirty` (`1957819` after the history rewrite, see [commit-map](commit-map.md)); the only change was in `eval/figures.py`, and the simulation code was the same.
- In figure 6 the two bars for "original authors' sft + GRPO" on seen server counts were measured on a subset of 1000 samples only and are hatched.

**Not run yet.** `e5_paper` (the original authors' 1.5B in the virtual-time closed loop, 400 slots) and the two-board rerun (RK3588 as the environment node). Both are optional.

## Open loop: the three test sets reported separately (stage 6)

```
Mac        python -m eval.open_loop runs/s3/main_0.5b_20261104T131423Z runs/s3/soft_1.5b_20261104T132257Z \
             runs/s3/hard_1.5b_20261104T134648Z runs/s3/ps_0.5b_20261104T140119Z \
             runs/s3/ps_soft_1.5b_20261108T044612Z runs/s3/ps_hard_1.5b_20261108T064436Z \
             runs/s3/paper_20261105T040018Z runs/20261108_s7_sft_baseline runs/20261108T103821Z_s7_colab_7b \
             --out evidence/G6 --title "阶段 6 开环：三套测试集分开看（含本地召回、本地精确）"
           → evidence/G6/open_loop.md, open_loop.csv (by server count), open_loop_by_n.csv (by number of terminals N)
```

Only the existing `preds.csv` files were scored again; no model was rerun. The share of states where local is best is 47.97% / 45.93% / 2.45% in the three test sets, and on the authors' cleaned test set a policy that never chooses local still reaches an accuracy of 97.55%, so the three sets are reported separately, with four more columns:

- Local-optimal share: the share of states in the group where local is best; it does not depend on the policy.
- Local recall: among the states where local is best, the share where local was chosen.
- Local precision: among the states where local was chosen, the share where local really is best.
- Offloading accuracy: among the states where some server is best, the share chosen correctly.

A zero denominator is written as "–". The authors' cleaned test set has no number of terminals N and is left out of the table by N. The reproduced 7B, its starting point SFT-7B and the original authors' three 1.5B models generate greedily in the paper format, while our students take 1 token in the compact format, so this table compares whole approaches, not model size alone. The findings are in the README under "Open loop: three test sets".

## Closed loop: Colab against Jetson (stage 6)

```
Mac     python -m eval.colab_loop --pack        pack sim/, deploy/, eval/ → data/s6_code/ (code.zip + meta.json); upload the whole folder to MyDrive/COMLLM/s6_code/
Colab   train/S6_Colab闭环.ipynb, RUN = "students"   6 students × 6 points (3 servers, N = 20/40/60, seeds 0/1, 1000 slots); results go to MyDrive/COMLLM/s6_runs/colab_loop/
Mac     download results.csv and sweep_config.json to runs/<UTC>_s6_colab_loop/
```

The points have the same names as in `eval.virtual_loop`, so the Colab and Jetson results match one to one by name. The 6 points of one model each run the simulator in their own thread; when decisions are needed, the states are handed to the main thread, put into one batch and answered by a single GPU forward pass, so `ms_per_decision` is the time per decision averaged over the batch. Cell 4 of the notebook recomputes the check point on Colab (one-step formula, 3 servers, N = 20, seed 0); its mean delay must be exactly the one computed on the Mac at packing time before the run goes on.

This run (`runs/20261112T064150Z_s6_colab_loop`): code `1ae403e` (the run record says `f9aad84`, from before the history rewrite, see [commit-map](commit-map.md)), A100, transformers 5.20.2, peft 0.21.2, all 36 points in one go, about 70 minutes, 0 fallbacks. The corresponding Jetson points are in `runs/20261107T071637Z_s5_virtual_e4a+e4b+e5`, `runs/20261108T044444Z_s5_virtual_e4ps_a` and `runs/20261108T094747Z_s5_virtual_e4ps_b`. The comparison of the two sides and the exception of the PS-formula hard-label 1.5B at N = 60 are in the README under "Closed loop: Colab and Jetson".

## Full open loop on the Jetson and additional latency (stage 6)

The six students and the original authors' SFT, GRPO and score-GRPO all use Q8_0, with the complete 5663 states per model. The processes on the board and on the environment node are checked first, and the runs are then done one after another; the open loop and the latency sweep must not switch the same scheduler service at the same time. The commands below are the entry points for reproduction; existing results are scored again directly without repeating inference.

```bash
python -m eval.jetson_open_loop --models main_0.5b,soft_1.5b,hard_1.5b,ps_0.5b,ps_soft_1.5b,ps_hard_1.5b,paper_sft_1.5b,paper_grpo_1.5b,paper_score_grpo_1.5b --dry-run
# for the real run drop --dry-run; after an interruption add --resume runs/jetson_open/<folder> for the single model
python -m eval.open_loop runs/jetson_open/* --out evidence/G6/jetson_open --title "Jetson 全量开环（GGUF Q8_0）"
python -m eval.jetson_open_loop --compare runs/jetson_open/* --colab runs/s3,runs/s4,runs/s3/paper_20261105T040018Z
python -m eval.bench_latency --fmt compact --cache on --model-key ps_soft_1.5b --n 500 --warmup 10 --servers 3,7,11
python -m eval.bench_latency --fmt paper --cache off --model-key paper_grpo_1.5b --n 500 --warmup 10 --servers 3,7,11
python -m eval.latency_table > evidence/G2/latency_table.md
```

This round is recorded in `runs/20261113T034033Z_s6_jetson_evaluation`: preflight/protocol, model preparation and SHA, the log of every item and suite_status, the offline scoring script and the acceptance result. Each model is checked for 5663 unique keys and exact coverage of the splits; an answer that does not come from the LLM is recorded as −1. The two existing student latency benchmarks are reused and the six missing ones were added; one more was run for the original authors' SFT, to match the decoding conditions after greedy was fixed. Every request in the open-loop and latency tests has a timeout of 10 seconds; the default 100 ms of the real-time service is not used in these tests. The tables and the source index are in `evidence/G6/jetson_open/`.

The server-count generalization of the PS 0.5B was measured on the Jetson for 3/4/6/7/8/10/11 servers, N = 40, seed 0; against its own PS teacher the mean delay is −1.06% to +3.95%, which is not a statistical conclusion over several seeds. The sources are the archived Stage 5 and e4ps_a results.

## Colab closed loop of the two reproduced 7B models (stage 7)

```text
Colab   train/S7_7B闭环_A100普通档_v1.ipynb (the ID of the original cloud notebook is unchanged), A100 with bf16
        existing dependencies transformers==5.20.2, peft==0.21.2; the notebook itself installs no library
        Runtime → Run all; the Drive mount is authorized by the user
        → MyDrive/COMLLM/s7_closed_runs_20261113_a100_standard/
Mac     download results.csv, provenance.json, runtime_identity.json, invocation_*.json, closed_acceptance_*.json
        → runs/20261113T104604Z_s7_colab_closed_loop/
```

The entry point is still the [original Drive notebook](https://colab.research.google.com/drive/1NzefdqRgIiR2a8bLc4AiGRlUu_Wr6iQu). The SFT folder is fixed to `reference_model/qwen2.5-7b-mec-sft-final-all`; `grpo_7b_repro` merges `s7_runs/nolacs/lora` (λ = 0) and `comllm_7b_repro` merges `s7_runs/lacs/lora` (λ = 0.3). Before the run, the 36-file closed-loop package, the SFT and adapter SHA, the joint training acceptance and the CPU formula check point 4.197929414659039 are verified; a resumed run requires the same provenance and hardware identity and skips only points that were written completely.

This run was done on an A100-SXM4-40GB with torch 2.14.1+cu130, transformers 5.20.2, peft 0.21.2, accelerate 1.15.0 and numpy 2.4.6. The final `closed_acceptance_20261113T092611.183871Z.json` is complete, all 12 unique points are present, and the CSV SHA agrees with the report. Each of the two models has 6 points: 3 servers, N = 20/40/60, seeds 0/1, sequential, 1000 slots, 200 slots of warm-up; the scored arrival interval is [200, 989). Over the 12 points, fallback is 0 at every point and local_frac is 0 at every point.

The closed loop uses paper / search / greedy with max_new_tokens = 16 and repetition_penalty = 1.05; the open loop uses 15 and 1.0, so the decoding cannot be called identical. The 7B in the README's closed-loop figure is the equal-weight mean of the two seeds for the same model and the same N; the raw data per point are kept in the CSV. The student reference is `runs/20261112T064150Z_s6_colab_loop`, and greedy, psgreedy and lacs (λ = 0.3) are from `runs/20261107T071637Z_s5_virtual_e4a+e4b+e5`; they are aligned by servers, n, mode, seed, slots and warmup, and every pair of corresponding points has the same n_tasks and n_decisions.

Virtual time does not count the wait for inference; mean_delay and p95_delay cover completed tasks only and must be read together with drop_rate. agree_greedy is the agreement with the one-step formula on the visited states, not an accuracy, and like fallback / n_decisions it covers all decisions; local_frac is over the scored tasks. The action metrics normally include the actions taken after a fallback; there was no fallback this time. The results apply only to the two reproduced sets of weights; they are not the score of the original COMLLM-7B, and not a controlled experiment that differs in model size alone; the 7B was not run on the Jetson.

## Control with first-come-first-served servers (stage 6)

```bash
python -m eval.fcfs_control        # formulas only, no Jetson, about 20 seconds locally; results in runs/<UTC>_s6_fcfs_control/
```

The PS-aware formula was derived from the processor sharing of the paper's eq. 4, and the simulator's servers queue by eq. 4 as well, so its advantage over the paper's formula could come only from the two sharing an assumption. This control switches the servers to first-come-first-served (`FcfsServer` in `sim/env.py`, which works only on the task that arrived earliest), keeps everything else and repeats the formula-against-formula comparison of experiment 6: 3/7/11 servers × N = 20/40/60/80, seeds 0/1/2, 2000 slots per point, sequential decisions. A formula written for first-come-first-served (`fcfs_costs` in `sim/cost.py`) is added as a reference. `MecEnv` still defaults to processor sharing, so existing results are not affected.

This run (`runs/20261114T120231Z_s6_fcfs_control`, code `a8f2d20`):

| Servers | Formula | Mean delay against the paper's formula | Range | Points where it is faster | Drop rate (it / paper's formula) | Tasks kept local (it / paper's formula) |
|---|---|---|---|---|---|---|
| Processor sharing | PS-aware formula | −9.27% | −22.73% to +3.79% | 10/12 | 0.82% / 4.22% | 28.2% / 58.3% |
| Processor sharing | FCFS formula | −9.26% | −22.74% to +3.79% | 10/12 | 0.81% / 4.22% | 28.3% / 58.3% |
| First-come-first-served | PS-aware formula | −12.88% | −26.40% to +1.32% | 11/12 | 0.05% / 4.14% | 28.3% / 58.3% |
| First-come-first-served | FCFS formula | −12.89% | −26.34% to +1.32% | 11/12 | 0.04% / 4.14% | 28.4% / 58.3% |

- The 72 processor-sharing runs (the paper's formula and the PS-aware formula) are identical item by item to the results of experiment 6, so adding the `discipline` parameter did not change the original simulation.
- The advantage of the PS-aware formula does not depend on the processor-sharing assumption: with first-come-first-served servers it is larger.
- The two new formulas give exactly the same result in 48 of the 72 runs. What they have in common is that they subtract the backlog the server finishes during the upload and do not multiply the backlog by (n + 1) again; these two points are what matters, not the sharing term. The paper's formula estimates the servers as too slow and therefore keeps more tasks on the terminal.
- Only formulas were compared. The students trained on the PS-aware formula were not run on first-come-first-served servers; the table per point is in `summary.md` in the run folder.

## Language ability of the students (stage 8)

```text
Colab   train/S8_语言能力检查.ipynb, any GPU runtime, Run all      2 base models and 6 students, about 11 minutes on an L4
        -> MyDrive/COMLLM/s8_runs/language_check/ (model_<key>.json, items_<key>.json, report.md)
Mac     copy that folder to runs/<UTC>_s8_language_check/
Mac     python -m eval.language_report runs/<UTC>_s8_language_check      -> analysis.md in that folder
```

The students were trained only on scheduling prompts with a one-letter answer. The notebook compares each untrained base with the merged students; all required data is inside the notebook.

- **Scheduling self-check.** Eight scheduling prompts, the answer is the most likely legal letter. It confirms that the adapter loaded.
- **Perplexity.** Eight built-in passages each in Chinese and English, plain text without the chat format, and the first 80,000 characters of the wikitext-2 test set.
- **Chat.** Twenty ordinary questions, greedy decoding, at most 48 new tokens, once with Qwen's default system prompt and once with the scheduler's. An answer counts as "one letter" when nothing else is left after stripping whitespace and a final full stop.
- **Multiple choice.** 32 built-in questions and a random 200 of the MMLU test set (seed 0), read two ways: the most likely of the letters A–D as the first answer token, and the option whose text is most likely after "Question: … Answer:".

This run (`runs/20261115T123413Z_s8_language_check`): L4, bf16, torch 2.14.1, transformers 5.20.2, peft 0.21.2, 10.6 minutes of model time. The README table and `analysis.md` give the numbers.

- The notebook that produced this run scored every model's self-check against the paper formula's optimum, so the three PS-teacher students show 4/8 in `report.md`. Against their own teacher they score 5/8, 6/8 and 8/8 (`analysis.md`); on three of the eight states the PS formula's two best servers are within 0.02 slots. The notebook now scores each student against its own teacher.
- The untrained 0.5B answers 7 of the 20 ordinary questions with a bare letter when given the scheduler's system prompt, which tells it to output one letter. The students do not (0/20, one of them 1/20).
- Chat answers are classified by form only. The check ran on full-precision PyTorch models, not on the Q8_0 models on the Jetson.
