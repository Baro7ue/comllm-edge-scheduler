"""一条长连接上发 JSON、收 JSON 的小工具。调度服务调 llama-server、环境节点调调度服务都用它。

为什么用长连接：每次新建 TCP 连接要多一次握手。局域网里一次握手约 0.2–0.3 ms（估算），
本地回环更少，但 500 次请求累计起来会混进延迟表，所以连接建好后一直复用。
同一个 JsonConnection 只能在一个线程里用。
"""
import http.client
import json
import time


class JsonConnection:
    def __init__(self, host, port, timeout_s=30.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.conn = None

    def _open(self):
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def request(self, method, path, body=None):
        """发一次请求，返回 (回复的字典, 往返毫秒)。

        1. 还没有连接就先建
        2. 发请求、读完整个回复、解析 JSON
        3. 连接被对方关掉之类的错误：重建连接再试一次；第二次还失败就把异常抛给调用方
        4. 等回复超时不重试：对方可能已经在算了，重发只会让它多算一遍
        """
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        last_error = None
        for attempt in range(2):
            if self.conn is None:
                self._open()
            t0 = time.perf_counter()
            try:
                self.conn.request(method, path, body=data, headers=headers)
                resp = self.conn.getresponse()
                raw = resp.read()
                ms = (time.perf_counter() - t0) * 1000.0
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {raw[:200]!r}")
                return json.loads(raw), ms
            except TimeoutError:
                self.close()
                raise
            except (http.client.HTTPException, ConnectionError, OSError) as e:
                last_error = e
                self.close()
        raise last_error
