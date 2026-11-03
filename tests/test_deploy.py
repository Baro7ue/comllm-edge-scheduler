"""阶段 2 的测试：请求格式、答案解析、调度服务的超时回退、实时仿真器。

调度服务的测试不需要真的 llama-server：这里起一个"假 llama-server"，按设定的延迟回一个写好的答案。
"""
import json
import math
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from deploy.client import SchedulerClient
from deploy.prompts import build_llm_request, chatml, compact_grammar, parse_action
from deploy.scheduler import Handler, JsonlLog, LlmWorker, SchedulerService
from deploy.wire import state_from_dict, state_to_dict
from sim import config as C
from sim.baselines import make_policy
from sim.cost import ServerView, State, best_action, one_step_costs
from sim.env import Task, run, summarize
from sim.realtime_env import RealtimeEnv


def _state(n_servers=3, d=3.0):
    servers = []
    for ghz in C.SERVER_LINEUPS[n_servers]:
        servers.append(ServerView(ghz=ghz, active=2, backlog=5.5))
    return State(d=d, local_wait=1.2, net_wait=0.4, servers=servers)


def _canned(letter_probs, content):
    """假 llama-server 的回复：letter_probs 是 [(token 文本, 概率), ...]，按概率从大到小。"""
    top = []
    for tok, p in letter_probs:
        top.append({"token": tok, "logprob": math.log(p)})
    return {"content": content,
            "completion_probabilities": [{"token": content, "logprob": math.log(letter_probs[0][1]),
                                          "top_logprobs": top}],
            "timings": {"prompt_n": 90, "cache_n": 110, "prompt_ms": 12.5, "predicted_n": 1, "predicted_ms": 0.01}}


class TestWire(unittest.TestCase):
    def test_roundtrip(self):
        s = _state(7)
        s.servers[2].history = [1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        back = state_from_dict(json.loads(json.dumps(state_to_dict(s))))
        self.assertEqual(back, s)


class TestPrompts(unittest.TestCase):
    def test_chatml(self):
        text = chatml([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
        self.assertEqual(text, "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n")

    def test_grammar(self):
        self.assertEqual(compact_grammar(4), "root ::= [A-D]")
        self.assertEqual(compact_grammar(12), "root ::= [A-L]")

    def test_request_bodies(self):
        s = _state(3)
        b = build_llm_request(s, "compact", True)
        self.assertEqual(b["n_predict"], 1)
        self.assertEqual(b["n_probs"], 20)
        self.assertNotIn("grammar", b)
        self.assertTrue(b["prompt"].startswith("<|im_start|>system\n你是边缘计算任务卸载调度器"))
        self.assertTrue(b["prompt"].endswith("<|im_start|>assistant\n"))
        g = build_llm_request(s, "compact", False, decode="grammar")
        self.assertEqual(g["grammar"], "root ::= [A-D]")
        self.assertFalse(g["cache_prompt"])
        p = build_llm_request(s, "paper", True)
        self.assertEqual(p["n_predict"], 12)
        self.assertTrue(p["ignore_eos"])
        self.assertIn("你是一个MEC任务卸载决策智能体", p["prompt"])

    def test_parse_probs(self):
        # 首选 token 不是字母；合法字母里 C 概率最大；E 超出 3 台服务器的范围，要忽略
        resp = _canned([("根据", 0.5), ("E", 0.2), ("C", 0.15), ("A", 0.1)], "根据")
        a, probs, p_chosen = parse_action("compact", "probs", resp, 4)
        self.assertEqual(a, 2)
        self.assertAlmostEqual(probs[2], 0.15)
        self.assertAlmostEqual(probs[0], 0.1)
        self.assertAlmostEqual(p_chosen, 0.5)

    def test_parse_no_letter(self):
        resp = _canned([("根据", 0.5), ("The", 0.2)], "根据")
        a, _, _ = parse_action("compact", "probs", resp, 4)
        self.assertIsNone(a)

    def test_parse_grammar_and_paper(self):
        a, _, _ = parse_action("compact", "grammar", _canned([("D", 0.9)], "D"), 4)
        self.assertEqual(a, 3)
        a, _, _ = parse_action("paper", "-", {"content": "卸载到边缘服务器2。理由是"}, 4)
        self.assertEqual(a, 2)
        a, _, _ = parse_action("paper", "-", {"content": "卸载到边缘服务器9"}, 4)
        self.assertIsNone(a)


# ===== 假 llama-server =====
class FakeLlama(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    delay_s = 0.0
    reply = None

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(n)
        time.sleep(FakeLlama.delay_s)
        body = json.dumps(FakeLlama.reply).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class TestSchedulerService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlama)
        threading.Thread(target=cls.fake.serve_forever, daemon=True).start()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.log = JsonlLog(os.path.join(cls.tmp.name, "s.jsonl"))
        worker = LlmWorker("127.0.0.1", cls.fake.server_address[1], cls.log)
        cls.service = SchedulerService(worker, cls.log, default_timeout_ms=100)
        Handler.service = cls.service
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.client = SchedulerClient("127.0.0.1", cls.httpd.server_address[1])

    @classmethod
    def tearDownClass(cls):
        cls.fake.shutdown()
        cls.httpd.shutdown()
        cls.log.f.close()
        cls.tmp.cleanup()

    def setUp(self):
        FakeLlama.delay_s = 0.0
        FakeLlama.reply = _canned([("B", 0.7), ("A", 0.2)], "B")

    def test_llm_answer(self):
        resp, e2e_ms = self.client.decide(_state(3), "compact", True, 100, req_id=1)
        self.assertEqual(resp["source"], "llm")
        self.assertEqual(resp["action"], 1)
        self.assertEqual(resp["cache_n"], 110)
        self.assertGreaterEqual(e2e_ms, resp["service_ms"])

    def test_timeout_fallback(self):
        # 假 llama-server 要 0.3 s，阈值 50 ms → 回退公式，原因 timeout
        FakeLlama.delay_s = 0.3
        s = _state(3)
        t0 = time.perf_counter()
        resp, _ = self.client.decide(s, "compact", True, 50, req_id=2)
        self.assertLess(time.perf_counter() - t0, 0.25)
        self.assertEqual(resp["source"], "fallback")
        self.assertEqual(resp["reason"], "timeout")
        self.assertEqual(resp["action"], best_action(one_step_costs(s)))
        time.sleep(0.35)      # 等那次迟到的调用做完，免得影响下一个测试

    def test_busy_fallback(self):
        # 第一个请求超时作废后 llama-server 还在算；紧接着的第二个请求不排队，马上回退，原因 busy
        FakeLlama.delay_s = 0.3
        resp1, _ = self.client.decide(_state(3), "compact", True, 50, req_id=6)
        self.assertEqual(resp1["reason"], "timeout")
        t0 = time.perf_counter()
        resp2, _ = self.client.decide(_state(3), "compact", True, 50, req_id=7)
        self.assertLess(time.perf_counter() - t0, 0.03)
        self.assertEqual(resp2["source"], "fallback")
        self.assertEqual(resp2["reason"], "busy")
        time.sleep(0.35)      # 等迟到的调用做完，工作线程空下来
        FakeLlama.delay_s = 0.0
        resp3, _ = self.client.decide(_state(3), "compact", True, 100, req_id=8)
        self.assertEqual(resp3["source"], "llm")

    def test_invalid_fallback(self):
        FakeLlama.reply = _canned([("根据", 0.9)], "根据")
        resp, _ = self.client.decide(_state(3), "compact", True, 100, req_id=3)
        self.assertEqual(resp["source"], "fallback")
        self.assertEqual(resp["reason"], "invalid")

    def test_formula(self):
        s = _state(11)
        resp, _ = self.client.decide(s, "formula", True, 100, req_id=4)
        self.assertEqual(resp["source"], "formula")
        self.assertEqual(resp["action"], best_action(one_step_costs(s)))

    def test_llama_down(self):
        log = JsonlLog(os.path.join(self.tmp.name, "down.jsonl"))
        worker = LlmWorker("127.0.0.1", 1, log)       # 端口 1 上没有服务
        service = SchedulerService(worker, log, default_timeout_ms=500)
        req = {"state": state_to_dict(_state(3)), "fmt": "compact", "timeout_ms": 500, "req_id": 5}
        out = service.decide(req, time.perf_counter())
        self.assertEqual(out["source"], "fallback")
        self.assertEqual(out["reason"], "error")
        log.f.close()


class TestRealtimeEnv(unittest.TestCase):
    def test_instant_decisions_match_offline(self):
        # 决策瞬间完成、在槽起点生效时，实时仿真器和离线仿真器的结果一模一样
        slots, warmup = 400, 50
        m_off, _ = run(make_policy("greedy"), 8, 3, slots=slots, warmup=warmup, seed=4)
        env = RealtimeEnv(8, 3, seed=4)
        greedy = make_policy("greedy")
        for t in range(slots):
            for task in env.begin_slot(t):
                st = env.observe(task, float(t))
                env.apply(task, greedy(st), st, float(t))
        env.advance_to(float(slots))
        counted = []
        for task in env.tasks:
            if warmup <= task.t_arrive < slots - C.DEADLINE_SLOTS - 1:
                counted.append(task)
        self.assertEqual(summarize(counted, 3), m_off)

    def test_decision_delay_adds_to_task_delay(self):
        # 同一个本地任务，决策晚 0.4 槽生效，完成时刻也晚 0.4 槽
        env = RealtimeEnv(1, 3)
        task = Task(tid=0, terminal=0, size=2.0, t_arrive=0.0, deadline=10.0)
        st = env.observe(task, 0.0)
        env.apply(task, 0, st, 0.4)
        self.assertAlmostEqual(task.delay, 0.4 + 2.0 / C.LOCAL_MBITS_PER_SLOT)

    def test_expired_counted_as_drop(self):
        env = RealtimeEnv(1, 3)
        a = Task(tid=0, terminal=0, size=2.0, t_arrive=0.0, deadline=10.0)
        b = Task(tid=1, terminal=0, size=2.0, t_arrive=0.0, deadline=10.0)
        env.apply(a, 0, env.observe(a, 0.0), 0.0)
        env.expire(b)
        m = summarize([a, b], 3)
        self.assertEqual(m["drop_rate"], 0.5)
        self.assertEqual(b.drop_at, "decision")


if __name__ == "__main__":
    unittest.main()
