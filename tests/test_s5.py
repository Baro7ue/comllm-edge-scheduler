"""阶段 5 的测试：论文格式（原作者拼法）、换模型接口、超载分流、虚拟时间闭环、扫描计划。

都不需要 Jetson：换模型用 tests/fake_llama_server.py 冒充 llama-server。
"""
import glob
import hashlib
import os
import socket
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

from deploy.client import SchedulerClient
from deploy.prompts import PAPER_SYSTEM, build_llm_request, parse_action, paper_messages
from deploy.scheduler import Handler, JsonlLog, LlamaProcess, LlmWorker, SchedulerService, model_info, warm_up
from eval import s5_sweep, virtual_loop
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

    def test_strict_parse(self):
        self.assertEqual(parse_action("paper", "-", {"content": "卸载到边缘服务器2"}, 4)[0], 2)
        self.assertEqual(parse_action("paper", "-", {"content": " 本地执行\n"}, 4)[0], 0)
        self.assertIsNone(parse_action("paper", "-", {"content": "卸载到边缘服务器1服务器"}, 12)[0])
        self.assertIsNone(parse_action("paper", "-", {"content": "卸载到边缘服务器9"}, 4)[0])
        self.assertEqual(parse_action("paper", "-", {"content": "卸载到边缘服务器11"}, 12)[0], 11)


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
        self.assertEqual(len(pts), (6 + 8 + 4) * 2)
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
