"""闭环仿真器的测试：先用能手算的小例子核对排队逻辑，再查可复现性。"""
import unittest

from sim import config as C
from sim.baselines import POLICY_NAMES, make_policy
from sim.env import MecEnv, PsServer, Task, run


def _task(tid, size, t=0.0):
    return Task(tid=tid, terminal=0, size=size, t_arrive=t, deadline=t + C.DEADLINE_SLOTS)


class TestPsServer(unittest.TestCase):
    """41.8 GHz 的服务器每槽能算 R ≈ 14.07 Mbits。"""
    R = C.mbits_per_slot(41.8)

    def test_two_jobs_share(self):
        # 两个各 R/2 的任务同时到：各分一半算力，都在第 1.0 槽算完
        s = PsServer(41.8)
        a, b = _task(0, self.R / 2), _task(1, self.R / 2)
        s.add_incoming(0.0, a)
        s.add_incoming(0.0, b)
        s.advance(0, 1)
        self.assertAlmostEqual(a.t_done, 1.0)
        self.assertAlmostEqual(b.t_done, 1.0)

    def test_late_arrival(self):
        # A（大小 R）0 时刻到，B（大小 R）0.5 时刻到：
        #   0–0.5 A 独占，算掉 R/2；之后两个平分，A 剩的 R/2 要 1 槽 → A 在 1.5 完成；
        #   B 在 0.5–1.5 算了 R/2，剩 R/2 独占 0.5 槽 → B 在 2.0 完成（总工作量 2R，正好 2 槽，算力没浪费）
        s = PsServer(41.8)
        a, b = _task(0, self.R), _task(1, self.R, t=0.5)
        s.add_incoming(0.0, a)
        s.add_incoming(0.5, b)
        s.advance(0, 1)
        s.advance(1, 2)
        self.assertAlmostEqual(a.t_done, 1.5)
        self.assertAlmostEqual(b.t_done, 2.0)

    def test_deadline_drop(self):
        # 11R 的任务独占也要 11 槽，超过 10 槽截止期 → 第 10 槽被丢弃
        s = PsServer(41.8)
        a = _task(0, 11 * self.R)
        s.add_incoming(0.0, a)
        for t in range(11):
            s.advance(t, t + 1)
        self.assertTrue(a.dropped)
        self.assertEqual(a.drop_at, "server")
        self.assertEqual(s.jobs, [])


class TestLocalAndUplink(unittest.TestCase):
    def test_local_fifo(self):
        # 同一台终端连续两个本地任务：第二个要等第一个做完
        env = MecEnv(n_terminals=1, n_servers=3)
        a, b = _task(0, 2.0), _task(1, 2.0)
        env.apply(a, 0, env.observe(a))
        st = env.observe(b)
        self.assertAlmostEqual(st.local_wait, 2.0 / C.LOCAL_MBITS_PER_SLOT)   # 看到的排队 = 第一个的计算时间
        env.apply(b, 0, st)
        self.assertAlmostEqual(b.delay, 4.0 / C.LOCAL_MBITS_PER_SLOT)
        self.assertAlmostEqual(b.delay, b.pred)          # 本地是 FIFO，公式预估和实际完全一样

    def test_offload_single(self):
        # 2.8 Mbits 卸载到空的服务器 1：上传 2 槽 + 计算 2.8/R 槽
        env = MecEnv(n_terminals=1, n_servers=3)
        a = _task(0, 2.8)
        env.apply(a, 1, env.observe(a))
        for _ in range(4):
            env.servers[0].advance(env.t, env.t + 1)
            env.t += 1
        self.assertAlmostEqual(a.delay, 2.0 + 2.8 / C.mbits_per_slot(41.8))

    def test_inflight_visible(self):
        # 刚派出去、还在上传的任务，调度器视角里已经算进服务器的活跃数和积压
        env = MecEnv(n_terminals=2, n_servers=3)
        a = Task(tid=0, terminal=0, size=3.0, t_arrive=0.0, deadline=10.0)
        env.apply(a, 1, env.observe(a))
        b = Task(tid=1, terminal=1, size=3.0, t_arrive=0.0, deadline=10.0)
        v = env.observe(b).servers[0]
        self.assertEqual(v.active, 1)
        self.assertAlmostEqual(v.backlog, 3.0)


class TestReproducible(unittest.TestCase):
    """G1 判据之一：同一随机种子跑两次结果完全一致。"""

    def test_same_seed_same_result(self):
        for name in POLICY_NAMES:
            m1, e1 = run(make_policy(name, seed=7), n_terminals=20, n_servers=3, slots=400, warmup=50, seed=7)
            m2, e2 = run(make_policy(name, seed=7), n_terminals=20, n_servers=3, slots=400, warmup=50, seed=7)
            self.assertEqual(m1, m2, name)
            self.assertEqual([t.action for t in e1.tasks], [t.action for t in e2.tasks], name)

    def test_different_seed_differs(self):
        m1, _ = run(make_policy("greedy"), 20, 3, slots=400, warmup=50, seed=1)
        m2, _ = run(make_policy("greedy"), 20, 3, slots=400, warmup=50, seed=2)
        self.assertNotEqual(m1, m2)

    def test_same_tasks_for_every_policy(self):
        # 同一个种子下，不同策略面对的是同一串任务（到达时刻、终端、大小都一样），比较才公平
        seqs = []
        for name in POLICY_NAMES:
            _, env = run(make_policy(name), 20, 3, slots=300, warmup=50, seed=3)
            seqs.append([(t.t_arrive, t.terminal, t.size) for t in env.tasks])
        for s in seqs[1:]:
            self.assertEqual(s, seqs[0])


if __name__ == "__main__":
    unittest.main()
