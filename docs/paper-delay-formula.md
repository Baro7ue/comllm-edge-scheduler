# The paper's edge delay formula multiplies by (N + 1) once too often

[简体中文](paper-delay-formula.zh-CN.md)

This document starts from eq. 4 of the paper (arXiv:2604.07148v2), derives how long a task really spends on a processor-sharing server, and compares the result with the paper's eq. 5–7. Equation numbers refer to the paper.

Summary:

- The paper's delay on the server is $`(L + D)(N + 1)\gamma / F`$.
- Under the processor-sharing model of eq. 4 itself, the delay on the server is at most $`\min\{D(N + 1),\ L + D\}\,\gamma / F`$.
- So whenever the server already holds tasks ($`N \ge 1`$), the paper's value is at least $`N + 1`$ times the real one.
- The extra factor is in eq. 6, which divides the backlog $`L`$ of the whole server by the share $`F/(N + 1)`$ of a single task.

Every step of the derivation was checked against the simulator; see [Numerical check](#numerical-check).

## Notation

| Symbol | Meaning | In the paper |
|---|---|---|
| $`D`$ | size of the new task, bits | $`D_m`$ |
| $`\gamma`$ | CPU cycles per bit | $`\gamma`$ |
| $`F`$ | CPU frequency of the server | $`F_e`$ |
| $`N`$ | active tasks already on the server when the new task arrives | $`N_e^m`$ |
| $`L`$ | backlog of the whole server when the new task arrives: the unfinished work of the $`N`$ tasks together, bits | $`L_e^m`$ |
| $`r_i`$ | unfinished work of the $`i`$-th existing task, $`r_1 + \dots + r_N = L`$ | not in the paper |
| $`\mu = F/\gamma`$ | bits the whole server processes per second | — |

## The paper's formula

Eq. 4, the computation rate given to the new task (the $`+1`$ is the new task itself):

```math
f^{\mathrm{eff}} = \frac{F}{N + 1}
```

Eq. 5, the execution time:

```math
T^{\mathrm{exec}} = \frac{D\gamma}{f^{\mathrm{eff}}} = \frac{D\gamma\,(N + 1)}{F}
```

Eq. 6, the queueing time:

```math
T^{\mathrm{wait}} = \frac{L\gamma}{f^{\mathrm{eff}}} = \frac{L\gamma\,(N + 1)}{F}
```

Eq. 7 adds the upload time to these two terms. Without the upload, the time on the server is:

```math
T^{\mathrm{paper}} = T^{\mathrm{wait}} + T^{\mathrm{exec}} = \frac{(L + D)(N + 1)\,\gamma}{F}
```

`one_step_costs` in `sim/cost.py` implements this formula, and it reproduces 100% of the labels in the authors' 1914 samples (`tests/test_cost.py`).

## The real time under processor sharing

Setting: the new task reaches the server at time 0, no other task arrives afterwards, and deadlines are ignored.

**Step 1: tasks that are present receive the same amount of service.** Processor sharing means every task on the server runs at the same speed at any moment. So from time 0 on, as long as a task is still present, the amount of its work that has been processed equals that of every other present task. Call this amount $`s(t)`$.

**Step 2: how much of each existing task is processed by the time the new task finishes.** The new task finishes when $`s = D`$. At that moment the $`i`$-th existing task:

- is still present if $`r_i \ge D`$, with $`D`$ of its work processed;
- has already finished and left if $`r_i < D`$, with $`r_i`$ of its work processed.

Both cases together give $`\min(r_i, D)`$.

**Step 3: a non-empty server runs at full speed.** Tasks only share the capacity; none of it is wasted, so the server as a whole works at rate $`\mu`$ all the time. By the time $`T`$ at which the new task finishes, the server has processed $`\mu T`$, which is the new task's own $`D`$ plus what was processed of each existing task:

```math
\mu T = D + \sum_{i=1}^{N} \min(r_i, D)
```

Therefore (call this eq. A):

```math
T^{\mathrm{PS}} = \frac{\gamma}{F}\left(D + \sum_{i=1}^{N} \min(r_i, D)\right)
```

**Step 4: an upper bound when only $`N`$ and $`L`$ are known.** The scheduler does not see each $`r_i`$, only $`N`$ and $`L`$. The sum has two upper bounds:

- each term $`\min(r_i, D) \le r_i`$, so the sum is at most $`L`$;
- each term $`\min(r_i, D) \le D`$, so the sum is at most $`ND`$.

Substituting into eq. A (call the result eq. B):

```math
T^{\mathrm{PS}} \le \frac{\gamma}{F}\,\min\{D(N + 1),\ L + D\}
```

When the existing tasks all have the same unfinished work ($`r_i = L/N`$), the sum is exactly $`\min(L, ND)`$ and eq. B holds with equality. So eq. B is the worst case when only $`N`$ and $`L`$ are known. `ps_costs` in `sim/cost.py` uses eq. B.

The two branches of eq. B each have a meaning:

- $`D(N + 1)`$: every existing task is larger than the new one, so the new task shares the server with $`N`$ tasks the whole time. This is exactly eq. 5.
- $`L + D`$: the existing tasks finish first, and the server becomes empty just as the new task finishes.

## Comparison with the paper's formula

The right-hand side of eq. B is at most $`(L + D)\gamma/F`$, so:

```math
\frac{T^{\mathrm{paper}}}{T^{\mathrm{PS}}} \ \ge\ \frac{(L + D)(N + 1)}{\min\{D(N + 1),\ L + D\}} \ \ge\ N + 1
```

What the paper adds, in two cases:

| Case | Real time | Paper | Extra part |
|---|---|---|---|
| No existing task is smaller than the new one ($`r_i \ge D`$) | $`D(N + 1)\gamma/F`$, which is eq. 5 | eq. 5 + eq. 6 | all of eq. 6 |
| Existing tasks have equal unfinished work and finish first ($`L \le ND`$) | $`(L + D)\gamma/F`$ | $`(L + D)(N + 1)\gamma/F`$ | a full factor of $`N + 1`$ |

## Where the extra factor comes from

Eq. 5 is correct: the new task gets $`F/(N + 1)`$ of the capacity, and this already contains the whole cost of sharing the server with the others.

The problem is eq. 6. Its numerator $`L`$ is the backlog of the whole server, that is, the work of $`N`$ tasks together; its denominator $`F/(N + 1)`$ is the share of one task. $`L`$ already counts the $`N`$ tasks once, and multiplying by $`N + 1`$ counts them a second time. Only two forms are consistent:

| Server model | Queueing | Execution | Total |
|---|---|---|---|
| First-come-first-served: wait for the tasks ahead, then use the whole server | $`L\gamma/F`$ | $`D\gamma/F`$ | $`(L + D)\gamma/F`$ |
| Processor sharing: start on arrival, with no queueing phase | 0 | at most $`\min\{D(N + 1),\ L + D\}\gamma/F`$ | same as execution |
| Paper: queueing and execution both divided by $`F/(N + 1)`$ | $`L\gamma(N + 1)/F`$ | $`D\gamma(N + 1)/F`$ | $`(L + D)(N + 1)\gamma/F`$ |

The paper's formula puts the structure of first-come-first-served (queue first, then execute) on top of the rate of processor sharing ($`F/(N + 1)`$).

One more check does not depend on the queueing discipline: a non-empty server runs at full speed, so $`L + D`$ of work is finished by $`(L + D)\gamma/F`$ at the latest, and the new task cannot finish later than that. The paper's value is $`N + 1`$ times this limit.

## Numerical check

```bash
python -m eval.ps_formula_check        # uses only the simulator's PsServer; a few seconds
```

The script puts $`N`$ existing tasks and one new task on the `PsServer` of `sim/env.py`, measures how long the new task takes, and compares it with the three formulas. The unit is one time slot (0.1 s).

A 30 GHz server (10.10 Mbits per slot), a new task of 3 Mbits, and 3 tasks already on the server:

| Unfinished work of the existing tasks, Mbits | Simulated | Eq. A | Eq. B | Paper | Paper ÷ simulated |
|---|---|---|---|---|---|
| 2, 2, 2 | 0.891 | 0.891 | 0.891 | 3.564 | 4.00 |
| 4, 4, 4 | 1.188 | 1.188 | 1.188 | 5.940 | 5.00 |
| 0.5, 1, 4.5 | 0.742 | 0.742 | 0.891 | 3.564 | 4.80 |

Over 20000 random states (server frequencies from `sim/config.py`, 0 to 8 existing tasks with 0.05 to 5 Mbits of unfinished work, a new task of 2 to 5 Mbits):

- the simulated time and eq. A differ by at most 8.88 × 10⁻¹⁶ slots, which is floating-point error;
- in no state does the simulated time exceed eq. B;
- over the states with existing tasks, the smallest value of paper ÷ (simulated × (N + 1)) is 1.000000, so the paper's value is at least $`N + 1`$ times the simulated one.

## Scope

- **Only the time on the server is covered.** The upload time is the same on both sides and is left out.
- **No task arrives afterwards.** Later arrivals take part of the capacity, so the real time is longer than eq. A. The paper's formula and eq. B are both one-step estimates, and neither includes this.
- **Deadlines are ignored.** In the simulator a task that passes 10 slots is dropped and leaves the server, which only makes the other tasks faster.
- **The paper's second deviation is not part of this derivation.** The server keeps working during the upload, so the backlog is smaller by $`\mu \times`$ upload time when the new task reaches the server; eq. 7 of the paper uses $`L`$ at decision time. `ps_costs` subtracts this part first and then applies eq. B.
- **The paper calls eq. 6 an approximation.** This document shows how far that approximation is from eq. 4, and says nothing about the paper's other results.

The closed-loop effect of this deviation (the paper's formula keeps more tasks on the terminal and drops more of them) is reported in the README under "Two teachers", and in [stages.md](stages.md) (in Chinese) in the table of differences from the paper and in the stage 6 first-come-first-served control.
