#!/usr/bin/env python3
"""测试用的假 llama-server：命令行参数和真的一样（-m 模型路径、--port 端口），
/health 回 ok，/props 回模型路径，/completion 总是答字母 B（概率 0.7）。只给 tests/test_s5.py 换模型的测试用。"""
import json
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = sys.argv[sys.argv.index("-m") + 1]
PORT = int(sys.argv[sys.argv.index("--port") + 1])
REPLY = {"content": "B",
         "completion_probabilities": [{"token": "B", "logprob": math.log(0.7),
                                       "top_logprobs": [{"token": "B", "logprob": math.log(0.7)},
                                                        {"token": "A", "logprob": math.log(0.2)}]}],
         "timings": {"prompt_n": 80, "cache_n": 120, "prompt_ms": 5.0, "predicted_n": 1, "predicted_ms": 0.01}}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send({"status": "ok"})
        else:
            self.send({"model_path": MODEL, "build_info": "fake", "total_slots": 1,
                       "default_generation_settings": {"n_ctx": 4096}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(n)
        self.send(REPLY)

    def log_message(self, *args):
        pass


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
