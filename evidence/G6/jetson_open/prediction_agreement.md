| 模型 | 测试集 | 条数 | 两边答案一样 % | Jetson 非 LLM 答案 |
|---|---|---|---|---|
| 0.5b_soft_T0.3（Jetson Q8_0） | test_seen | 3010 | 97.3 | 0 |
| 0.5b_soft_T0.3（Jetson Q8_0） | test_unseen | 2408 | 97.1 | 0 |
| 0.5b_soft_T0.3（Jetson Q8_0） | friend_clean | 245 | 97.6 | 0 |
| 1.5b_soft_T0.3（Jetson Q8_0） | test_seen | 3010 | 97.1 | 0 |
| 1.5b_soft_T0.3（Jetson Q8_0） | test_unseen | 2408 | 97.4 | 0 |
| 1.5b_soft_T0.3（Jetson Q8_0） | friend_clean | 245 | 98.8 | 0 |
| 1.5b_hard（Jetson Q8_0） | test_seen | 3010 | 99.6 | 0 |
| 1.5b_hard（Jetson Q8_0） | test_unseen | 2408 | 99.6 | 0 |
| 1.5b_hard（Jetson Q8_0） | friend_clean | 245 | 100.0 | 0 |
| 0.5b_soft_T0.3_ps（Jetson Q8_0） | test_seen | 3010 | 73.8 | 0 |
| 0.5b_soft_T0.3_ps（Jetson Q8_0） | test_unseen | 2408 | 69.8 | 0 |
| 0.5b_soft_T0.3_ps（Jetson Q8_0） | friend_clean | 245 | 72.2 | 0 |
| 1.5b_soft_T0.3_ps（Jetson Q8_0） | test_seen | 3010 | 72.3 | 0 |
| 1.5b_soft_T0.3_ps（Jetson Q8_0） | test_unseen | 2408 | 77.0 | 0 |
| 1.5b_soft_T0.3_ps（Jetson Q8_0） | friend_clean | 245 | 66.5 | 0 |
| 1.5b_hard_ps（Jetson Q8_0） | test_seen | 3010 | 100.0 | 0 |
| 1.5b_hard_ps（Jetson Q8_0） | test_unseen | 2408 | 99.2 | 0 |
| 1.5b_hard_ps（Jetson Q8_0） | friend_clean | 245 | 100.0 | 0 |
| 论文原版 sft（Jetson Q8_0） | test_seen | 3010 | 94.4 | 0 |
| 论文原版 sft（Jetson Q8_0） | test_unseen | 2408 | 86.1 | 0 |
| 论文原版 sft（Jetson Q8_0） | friend_clean | 245 | 82.9 | 0 |
| 论文原版 grpo（Jetson Q8_0） | test_seen | 3010 | 87.1 | 0 |
| 论文原版 grpo（Jetson Q8_0） | test_unseen | 2408 | 90.7 | 0 |
| 论文原版 grpo（Jetson Q8_0） | friend_clean | 245 | 98.0 | 0 |
| 论文原版 score-grpo（Jetson Q8_0） | test_seen | 3010 | 86.3 | 0 |
| 论文原版 score-grpo（Jetson Q8_0） | test_unseen | 2408 | 88.9 | 0 |
| 论文原版 score-grpo（Jetson Q8_0） | friend_clean | 245 | 93.1 | 0 |
