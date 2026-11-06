"""环境节点这一侧：调用 Jetson 上的调度服务。延迟基准和实时闭环共用。

所有时间戳都在环境节点上记（time.perf_counter，单调时钟），Jetson 只回报它自己内部各段花了多久，
所以两块板子不用对时。
"""
from deploy import testbed
from deploy.http_json import JsonConnection
from deploy.wire import state_to_dict


class SchedulerClient:
    def __init__(self, host=None, port=None, http_timeout_s=30.0):
        cfg = testbed.load()
        self.host = host or cfg["scheduler_node"]["ip"]
        self.port = port or cfg["scheduler_node"]["port"]
        self.conn = JsonConnection(self.host, self.port, timeout_s=http_timeout_s)

    def info(self):
        d, _ = self.conn.request("GET", "/info")
        return d

    def switch_model(self, key, timeout_s=300.0):
        """让调度服务换成模型清单里的 key，返回新的 /info。换模型要十几秒到一分钟，所以单独开一条等得久的连接。"""
        conn = JsonConnection(self.host, self.port, timeout_s=timeout_s)
        try:
            d, _ = conn.request("POST", "/model", {"key": key})
        finally:
            conn.close()
        return d

    def decide(self, state, fmt, cache_prompt, timeout_ms, req_id, decode="probs"):
        """发一次决策请求，返回 (调度服务的回复字典, 环境节点量到的往返毫秒)。

        1. 状态转成字典，拼请求体
        2. 发 POST /decide，计时从发出前一刻到读完回复
        """
        body = {"state": state_to_dict(state), "fmt": fmt, "decode": decode, "cache_prompt": bool(cache_prompt),
                "timeout_ms": timeout_ms, "req_id": req_id}
        return self.conn.request("POST", "/decide", body)


def parse_host_port(text):
    """"192.168.50.20:8100" → ("192.168.50.20", 8100)；None → (None, None)，表示用 testbed.json 里的。"""
    if not text:
        return None, None
    host, port = text.rsplit(":", 1)
    return host, int(port)
