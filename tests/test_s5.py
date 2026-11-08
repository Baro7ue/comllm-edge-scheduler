"""阶段 5 的测试：论文格式（原作者拼法）、换模型接口、超载分流、虚拟时间闭环、扫描计划。

都不需要 Jetson：换模型用 tests/fake_llama_server.py 冒充 llama-server。
"""
import glob
import hashlib
import json
import math
import os
import socket
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

from deploy import testbed
from deploy.client import SchedulerClient
from deploy.prompts import PAPER_SYSTEM, build_llm_request, parse_action, paper_messages
from deploy.scheduler import Handler, JsonlLog, LlamaProcess, LlmWorker, SchedulerService, model_info, warm_up
from deploy.wire import state_to_dict
from eval import jetson_open_loop, probe_paper_gguf, s5_sweep, virtual_loop
from eval.open_loop import load_preds
from eval.realtime import LocalDecider, build_summary, run_realtime
from sim import config as C
from sim.baselines import make_policy
from sim.cost import ServerView, State
from sim.env import MecEnv, run
from sim.realtime_env import RealtimeEnv
from sim.render import PAPER_INSTRUCTION

HERE = os.path.dirname(os.path.abspath(__file__))


def _state(n_servers=3):
    servers = []
    for ghz in C.SERVER_LINEUPS[n_servers]:
        servers.append(ServerView(ghz=ghz, active=2, backlog=5.5))
    return State(d=3.0, local_wait=1.2, net_wait=0.4, servers=servers)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestPaperFormat(unittest.TestCase):
    def test_repo_layout(self):
        msgs = paper_messages(_state(3))
        self.assertEqual(msgs[0]["content"], PAPER_SYSTEM)
        user = msgs[1]["content"]
        self.assertTrue(user.startswith(PAPER_INSTRUCTION + "\n\n**输入数据**：\n任务大小: 3.0000 Mbits\n"))
        self.assertIn("\n丢弃惩罚: 100\n", user)
        self.assertIn("\n边缘服务器3: 类型 节能型, CPU频率 22.0 GHz", user)

    def test_request(self):
        b = build_llm_request(_state(3), "paper", False)
        self.assertEqual(b["n_predict"], 20)
        self.assertNotIn("ignore_eos", b)
        self.assertEqual(b["top_k"], 1)
        self.assertEqual(b["repeat_penalty"], 1.0)
        self.assertEqual(b["temperature"], 0)

    def test_strict_parse(self):
        self.assertEqual(parse_action("paper", "-", {"content": "卸载到边缘服务器2"}, 4)[0], 2)
        self.assertEqual(parse_action("paper", "-", {"content": " 本地执行\n"}, 4)[0], 0)
        self.assertIsNone(parse_action("paper", "-", {"content": "卸载到边缘服务器1服务器"}, 12)[0])
        self.assertIsNone(parse_action("paper", "-", {"content": "卸载到边缘服务器9"}, 4)[0])
        self.assertEqual(parse_action("paper", "-", {"content": "卸载到边缘服务器11"}, 12)[0], 11)


class TestProbeSplit(unittest.TestCase):
    """GGUF 和 PyTorch 回答的分岔点：Jetson 答服务器 4、PyTorch 答服务器 1，分岔在数字那一步。"""

    @staticmethod
    def _rec(tok, p, cands):
        return {"token": tok, "logprob": math.log(p),
                "top_logprobs": [{"token": t, "logprob": math.log(q)} for t, q in cands]}

    def test_split(self):
        cp = [self._rec("卸载", 0.6, [("卸载", 0.6), ("本地", 0.4)]), self._rec("到边缘", 0.99, [("到边缘", 0.99)]),
              self._rec("服务器", 0.99, [("服务器", 0.99)]),
              self._rec("4", 0.35, [("4", 0.35), ("1", 0.30), ("3", 0.2)]), self._rec("", 0.98, [("", 0.98)])]
        step, p_j, p_pt, rank = probe_paper_gguf.split_point(cp, "卸载到边缘服务器1")
        self.assertEqual((step, rank), (3, 2))
        self.assertAlmostEqual(p_j, 0.35)
        self.assertAlmostEqual(p_pt, 0.30)
        self.assertEqual(probe_paper_gguf.split_point(cp, "卸载到边缘服务器4"), (None, None, None, None))
        step, _, p_pt, rank = probe_paper_gguf.split_point(cp, "本地执行")
        self.assertEqual((step, rank), (0, 2))
        self.assertAlmostEqual(p_pt, 0.4)
        m, at = probe_paper_gguf.min_margin(cp)
        self.assertAlmostEqual(m, 0.05)
        self.assertEqual(at, 3)


class TestModelSwitch(unittest.TestCase):
    """真的拉起两次"假 llama-server"，走一遍 POST /model。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.models = {}
        for key in ("a", "b"):
            path = os.path.join(cls.tmp.name, key + ".gguf")
            with open(path, "wb") as f:
                f.write(key.encode() * 1000)
            cls.models[key] = {"gguf": path, "fmt": "compact"}
        cls.log = JsonlLog(os.path.join(cls.tmp.name, "s.jsonl"))
        port = _free_port()
        cls.llama = LlamaProcess(os.path.join(HERE, "fake_llama_server.py"), port, 4096, cls.tmp.name)
        cls.llama.start(cls.models["a"]["gguf"])
        worker = LlmWorker("127.0.0.1", port, cls.log)
        warm_up(worker)
        cls.service = SchedulerService(worker, cls.log, 100, cls.llama, cls.models)
        cls.service.info = {"model_key": "a"}
        cls.service.info.update(model_info("a", cls.models["a"]["gguf"]))
        Handler.service = cls.service
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.client = SchedulerClient("127.0.0.1", cls.httpd.server_address[1])

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.llama.stop()
        cls.log.f.close()
        cls.tmp.cleanup()

    def test_switch_and_decide(self):
        info = self.client.switch_model("b")
        self.assertEqual(info["model_key"], "b")
        self.assertEqual(info["gguf_sha256"], hashlib.sha256(b"b" * 1000).hexdigest())
        self.assertEqual(info["llama"]["model_path"], self.models["b"]["gguf"])
        resp, _ = self.client.decide(_state(3), "compact", True, 1000, req_id=1)
        self.assertEqual(resp["source"], "llm")
        self.assertEqual(resp["action"], 1)
        with self.assertRaises(RuntimeError):
            self.client.switch_model("no_such_model")
        # 文件不存在：报错，原来的模型照常工作
        self.service.models["gone"] = {"gguf": os.path.join(self.tmp.name, "gone.gguf"), "fmt": "compact"}
        with self.assertRaises(RuntimeError):
            self.client.switch_model("gone")
        self.assertEqual(self.client.info()["model_key"], "b")
        resp, _ = self.client.decide(_state(3), "compact", True, 1000, req_id=2)
        self.assertEqual(resp["source"], "llm")

    def test_jetson_open_loop(self):
        """Jetson 开环：每条状态答一次写进 preds.csv；断了接着跑不重答；--compare 和 Colab 的答案逐条比。"""
        rows = []
        for i in range(5):
            rows.append({"state": state_to_dict(_state(3)), "_idx": i})
        data = {"friend_clean": rows}
        cfg = {"models": self.models}
        out = os.path.join(self.tmp.name, "jetson_open", "x_a")
        jetson_open_loop.run_model(self.client, "a", data, out, cfg)
        jetson_open_loop.run_model(self.client, "a", data, out, cfg)      # 第二次：全部已答，不会重复写
        preds = load_preds(os.path.join(out, "preds.csv"))
        self.assertEqual(len(preds), 5)
        self.assertEqual(set(preds.values()), {1})
        with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
            info = json.load(f)
        self.assertEqual(info["name"], "a（Jetson Q8_0）")
        self.assertEqual(info["gguf_sha256"], hashlib.sha256(b"a" * 1000).hexdigest())
        # Colab 那边同名模型：5 条里 4 条一样
        colab = os.path.join(self.tmp.name, "colab", "run_a")
        os.makedirs(colab)
        with open(os.path.join(colab, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"name": "a"}, f)
        with open(os.path.join(colab, "preds.csv"), "w", encoding="utf-8") as f:
            f.write("split,idx,pred\n")
            for i in range(5):
                f.write(f"friend_clean,{i},{1 if i else 0}\n")
        jetson_open_loop.COLAB_NAME["a"] = "a"
        try:
            lines = jetson_open_loop.compare([out], [os.path.join(self.tmp.name, "colab")])
        finally:
            del jetson_open_loop.COLAB_NAME["a"]
        self.assertIn("| a（Jetson Q8_0） | friend_clean | 5 | 80.0 | 0 |", lines)


class TestShed(unittest.TestCase):
    def test_overload_is_shed(self):
        # N = 12、每次决策假装 40 ms：每秒 36 个请求，一个决策器最多 25 个/秒，超载。开分流后不会有任务因为等不到决策而丢弃
        slots = 60
        env = RealtimeEnv(12, 3, seed=1)
        loop, lags = run_realtime(env, LocalDecider(40.0), slots, progress_every=10 ** 9, shed_ms=100.0)
        s = build_summary(env, loop, lags, 3, slots, 10)
        self.assertGreater(s["sources"].get("shed", 0), 0)
        self.assertEqual(s["drop_at"].get("decision", 0), 0)


class TestVirtualLoop(unittest.TestCase):
    def test_same_as_offline(self):
        p = {"point": "x", "servers": 3, "n": 20, "policy": "greedy", "mode": "batch", "seed": 2, "slots": 300,
             "warmup": 50}
        m, share = virtual_loop.run_point(p, virtual_loop.Recorder(make_policy("greedy")))
        m_off, _ = run(make_policy("greedy"), 20, 3, slots=300, warmup=50, seed=2, mode="batch")
        self.assertEqual(m, m_off)
        self.assertAlmostEqual(sum(share), 1.0, places=3)

    def test_plans_dedupe(self):
        both = virtual_loop.expand("e4a,e5")
        names = [p["point"] for p in both]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("llm_main_0.5b_k3_n40_sequential_s0_t1000", names)
        self.assertLess(len(both), len(virtual_loop.expand("e4a")) + len(virtual_loop.expand("e5")))

    def test_e6_lacs_lambda(self):
        pts = virtual_loop.expand("e6")
        self.assertEqual(len(pts), 3 * 4 * (5 * 3 + 2 * 3))
        self.assertFalse(any(p["policy"].startswith("llm:") for p in pts))
        self.assertAlmostEqual(make_policy("lacs:2").lam, 2.0)
        self.assertAlmostEqual(make_policy("lacs").lam, 0.3)
        env = MecEnv(30, 5, seed=0)
        for _ in range(50):
            env.step(make_policy("lacs:0"))
        rec = virtual_loop.Recorder(make_policy("lacs:0"))
        for _ in range(50):
            env.step(rec)
        self.assertEqual(rec.agree, rec.n)

    def test_e4ps_two_teachers_three_students(self):
        """e4ps 补齐 2 个老师 × 3 种学生：3 台、N = 20/40/60、种子 0 和 1，已经跑过的点不重跑，模型键都登记过。"""
        pts = virtual_loop.expand("e4ps")
        names = {p["point"] for p in pts}
        self.assertEqual(len(pts), 35)
        for key in ("main_0.5b", "soft_1.5b", "hard_1.5b", "ps_0.5b", "ps_soft_1.5b", "ps_hard_1.5b"):
            for n in (20, 40, 60):
                for seed in (0, 1):
                    name = f"llm_{key}_k3_n{n}_sequential_s{seed}_t1000"
                    done_before = (key in ("main_0.5b", "ps_0.5b") and seed == 0) or \
                        (key == "soft_1.5b" and n == 40 and seed == 0)
                    self.assertEqual(name in names, not done_before, name)
        cfg = testbed.load()
        for p in pts:
            self.assertIn(p["policy"][4:], cfg["models"])
            self.assertIn(p["policy"][4:], virtual_loop.EST_MS)

    def test_e4ps_split(self):
        """e4ps_a + e4ps_b 正好是 e4ps，不重叠；B 段只有 PS 老师的两个 1.5B。"""
        a = {p["point"] for p in virtual_loop.expand("e4ps_a")}
        b = virtual_loop.expand("e4ps_b")
        b_names = {p["point"] for p in b}
        whole = {p["point"] for p in virtual_loop.expand("e4ps")}
        self.assertEqual(len(a), 23)
        self.assertEqual(len(b_names), 12)
        self.assertEqual(a & b_names, set())
        self.assertEqual(a | b_names, whole)
        for p in b:
            self.assertIn(p["policy"], ("llm:ps_soft_1.5b", "llm:ps_hard_1.5b"))

    def test_smallnet_policy(self):
        paths = sorted(glob.glob(os.path.join("runs", "s3", "small_net_*", "weights.npz")))
        if not paths:
            self.skipTest("没有 runs/s3/small_net_*/weights.npz")
        net = virtual_loop.SmallNetPolicy(paths[-1])
        rec = virtual_loop.Recorder(net)
        env = MecEnv(30, 5, seed=0)
        for _ in range(100):
            env.step(rec)
        self.assertGreater(rec.agree / rec.n, 0.95)


class TestSweepPlan(unittest.TestCase):
    def test_e1(self):
        pts = s5_sweep.expand("e1")
        self.assertEqual(len(pts), (6 + 8 + 6) * 2)
        self.assertEqual(pts[0]["model_key"], "formula")
        keys = []
        for p in pts:
            if not keys or keys[-1] != p["model_key"]:
                keys.append(p["model_key"])
        self.assertEqual(keys, ["formula", "main_0.5b", "soft_1.5b"])      # 同一模型的点挨在一起

    def test_shed_names(self):
        pts = s5_sweep.expand("e1,e1_shed")
        names = [p["point"] for p in pts]
        self.assertIn("main_0.5b_compact_n8_s0", names)
        self.assertIn("main_0.5b_compact_n8_s0_shed100", names)
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
