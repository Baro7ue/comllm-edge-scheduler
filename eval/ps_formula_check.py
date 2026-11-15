"""Check the derivation in docs/paper-delay-formula.md against the simulator's processor-sharing server.

For random server states it puts N tasks with remaining sizes r_1..r_N on one PsServer, adds a new task of
size D at time 0 with nothing arriving afterwards, and compares the time the simulator takes with:

  exact   (D + sum_i min(r_i, D)) / R          the closed form derived in the document
  bound   min(D * (N + 1), L + D) / R          the largest the exact value can be when only N and L are known
  paper   (L + D) * (N + 1) / R                the paper's eq. 5 + eq. 6, without the upload term

R is the server's rate in Mbits per slot and L = r_1 + ... + r_N. Upload time and deadlines are left out:
this is only about the time spent on the server.

  python -m eval.ps_formula_check
"""
import random

from sim import config as C
from sim.env import PsServer, Task

NO_DEADLINE = 1e9


def simulate(ghz, rems, d):
    """Slots the new task spends on a PS server that already holds tasks with the remaining sizes in rems."""
    server = PsServer(ghz)
    for i, r in enumerate(rems):
        server.jobs.append([Task(i, 0, r, 0.0, NO_DEADLINE), r])
    new = Task(len(rems), 0, d, 0.0, NO_DEADLINE)
    server.jobs.append([new, d])
    server.advance(0.0, 1e6)
    return new.t_done


def formulas(ghz, rems, d):
    """(exact, bound, paper) in slots."""
    R, n, L = C.mbits_per_slot(ghz), len(rems), sum(rems)
    return ((d + sum(min(r, d) for r in rems)) / R, min(d * (n + 1), L + d) / R, (L + d) * (n + 1) / R)


def random_check(trials=20000, seed=0):
    rng = random.Random(seed)
    ghz_all = sorted(C.SERVER_KIND)
    max_err, over_bound, min_ratio = 0.0, 0, float("inf")
    for _ in range(trials):
        ghz = rng.choice(ghz_all)
        rems = [rng.uniform(0.05, 5.0) for _ in range(rng.randint(0, 8))]
        d = rng.uniform(*C.TASK_MBITS)
        t = simulate(ghz, rems, d)
        exact, bound, paper = formulas(ghz, rems, d)
        max_err = max(max_err, abs(t - exact))
        over_bound += t > bound + 1e-9
        if rems:
            min_ratio = min(min_ratio, paper / t / (len(rems) + 1))
    return max_err, over_bound, min_ratio


def main():
    print("worked examples (30 GHz server, new task D = 3 Mbits), slots on the server")
    print("  remaining sizes     simulated   exact   bound   paper   paper / simulated")
    for rems in ([2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [0.5, 1.0, 4.5]):
        t = simulate(30.0, rems, 3.0)
        exact, bound, paper = formulas(30.0, rems, 3.0)
        print(f"  {str(rems):18s}  {t:9.3f}  {exact:6.3f}  {bound:6.3f}  {paper:6.3f}  {paper / t:8.2f}")
    trials = 20000
    max_err, over_bound, min_ratio = random_check(trials)
    print(f"\n{trials} random states, 0 to 8 tasks already on the server")
    print(f"  largest |simulated - exact|: {max_err:.2e} slots")
    print(f"  states where simulated > bound: {over_bound}")
    print(f"  smallest paper / (simulated * (N + 1)) over states with N >= 1: {min_ratio:.6f}")


if __name__ == "__main__":
    main()
