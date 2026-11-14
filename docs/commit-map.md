# Commit ids before and after the history rewrite

On 2026-11-14 the history of `main` was rewritten for two reasons: every commit message was put
into English in one format (`type(stage): summary`), and the Colab account display name and Google
user ID were removed from every version of the archived notebooks (see the `redaction_note.json`
files under `runs/`). File contents are otherwise unchanged and the final tree is identical.

Run records under `runs/` and `evidence/` (for example `git` in `config.json`, `sweep_config.json`
or a code pack's `meta.json`) still name the old commit ids. Use this table to find the commit they
refer to. An id with `-dirty` means the run had uncommitted changes on top of that commit.

| Old | New | Date | Commit |
|---|---|---|---|
| `b38b00e` | `9a6a3f4` | 2026-11-01 | chore(s0): scaffold the repository, pin versions and add G0 evidence |
| `1cc7038` | `5e200a5` | 2026-11-03 | feat(s1): add the closed-loop simulator, four baselines, prompt renderers and unit tests |
| `1506e2e` | `59cfce8` | 2026-11-03 | chore(s1): add G1 evidence (test output, N sweep, LACS check) |
| `e86a183` | `307327d` | 2026-11-03 | feat(s2): add the Jetson scheduler service, latency benchmark, real-time closed loop and tests |
| `47fe3c6` | `e94f14c` | 2026-11-04 | chore(s2): add G2 evidence (latency table, real-time loop, timeout fallback log, N_max check) |
| `598c8e7` | `c1a151d` | 2026-11-04 | docs(s2): list the differences from the paper in the README |
| `322fd63` | `06a0a04` | 2026-11-04 | feat(s3): add training data generation, Colab training and paper-baseline notebooks, open-loop scoring and GGUF comparison |
| `7671eb4` | `7080529` | 2026-11-05 | fix(s3): build the paper-baseline prompt the way the authors' repository does |
| `5ffd6e6` | `2a7545a` | 2026-11-05 | fix(s3): fix misaligned reference answers in the GGUF comparison and use the authors' prompt layout for the paper baseline |
| `e2c0b24` | `af26695` | 2026-11-05 | chore(s3): add G3 evidence (data statistics, open-loop scores, GGUF comparison) |
| `03805fd` | `24ec550` | 2026-11-05 | fix(s3): use the 11/5 rerun for the paper baseline, add the small-network control and fix the versions table |
| `2c27531` | `d4a9a02` | 2026-11-05 | feat(s4): add GRPO data, rewards, two Colab notebooks and a summary script |
| `4314fcd` | `2ad0541` | 2026-11-06 | chore(s4): add G4 evidence (diagnose, smoke, B, fixed, repo) |
| `6cee24b` | `51b8c16` | 2026-11-06 | docs(s4): add the extra analysis and the stage 4 conclusions to the README |
| `9e1ffb8` | `549bad1` | 2026-11-06 | docs(s4): add a stage 4 section to the README and the A100 to the versions table |
| `1cb978a` | `5b9db7a` | 2026-11-06 | feat(s5): add the closed-loop evaluation scripts (model switching, overload shedding, real-time sweeps, virtual-time loop, figures) and use the authors' layout for the paper format |
| `87c193e` | `9f1e4f7` | 2026-11-06 | chore(s5): record checksums of the frozen model files |
| `b6e076d` | `fda81d0` | 2026-11-06 | feat(s5): add the LACS closed-loop experiment (e6) and extra overload points for experiment 1 |
| `4567c59` | `1957819` | 2026-11-07 | fix(s5): pin llama-server sampling to greedy and add the paper GGUF probe |
| `74936ab` | `0cd25b1` | 2026-11-07 | fix(s5): make the stage 5 figures readable and stop misleading comparisons |
| `641b39b` | `835e4b3` | 2026-11-07 | docs(s5): add the stage 5 closed-loop results to the README |
| `657a40a` | `4a2ed83` | 2026-11-07 | docs(s5): fix swapped experiment 4a/4b labels in the stage 5 results |
| `1008a76` | `6259bdd` | 2026-11-08 | feat(s6): add both teacher lines to the README and the Colab and Jetson evaluation runners |
| `f9aad84` | `1ae403e` | 2026-11-11 | feat(s6): report the open loop per test set with local recall and precision |
| `98fe050` | `8d9cf38` | 2026-11-11 | docs(s6): add the Colab against Jetson closed loop for the six students |
| `d97e492` | `43d8ea2` | 2026-11-11 | chore(s7): archive the stopped 7B closed-loop run and the paper GGUF conversions |
| `2852269` | `bd40b71` | 2026-11-12 | fix(s7): validate the effective generation defaults in the 7B closed loop |
| `0dbe08f` | `3727703` | 2026-11-12 | chore(s7): archive the 7B run stopped by the runtime identity check |
| `0dcfaec` | `b7c10ef` | 2026-11-12 | fix(s7): write the 7B A100 rerun to its own output folder |
| `7f80188` | `378d13e` | 2026-11-12 | chore(s7): keep the state of the 7B closed loop paused by the user |
| `e2a0825` | `f226ae1` | 2026-11-12 | chore(s7): give the standard A100 closed-loop run its own output folder |
| `e95beea` | `f3d3471` | 2026-11-13 | chore(s7): archive the verified 7B closed-loop run |
| `cafd7e4` | `968d3a7` | 2026-11-12 | feat(s6): register the original-author GRPO models for the Jetson |
| `82f17f6` | `adbdb4c` | 2026-11-13 | feat(s6): complete the nine-model Jetson open loop and latency coverage |
| `bb721cb` | `424cec1` | 2026-11-13 | chore(s7): merge the stage 6 Jetson evaluation into the 7B closed-loop branch |
| `f66b986` | `b274498` | 2026-11-13 | chore(repo): record what was redacted in the archived Colab notebooks |
| `602f59d` | `28670f1` | 2026-11-08 | feat(s7): add paired 7B training and validation |
| `d366940` | `89cf2c6` | 2026-11-14 | docs(repo): split the README into English and Chinese with figures |

The tag `v1.0-freeze` moved from `87c193e` to `9f1e4f7` (the same tree).
