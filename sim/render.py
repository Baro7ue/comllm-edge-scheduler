"""把 State 渲染成提示词。两种格式：

  论文格式（paper）：和原作者数据集逐字一致（instruction + input 字典），给原作者的论文原版 1.5B 用。
                     约 1300–2400 个字符，动作是"本地执行 / 卸载到边缘服务器k"这样的多 token 中文。
  压缩格式（compact）：给我们自己训练的模型用。不变的说明放进 system（前缀，可被 KV 缓存复用），
                     每次变的数字放进 user，目标 150–250 个 token；动作只用一个字母 A–L。
"""
import re

from sim import config as C
from sim.cost import ServerView, State, rounded

# ===== 第 1 段：论文格式 =====
# 这是原作者数据里 instruction 字段的原文，是提示词模板，不是数据。
PAPER_INSTRUCTION = "\n".join([
    "你是一个MEC任务卸载决策智能体。你的目标是为新到达的任务制定最优卸载策略，以最小化任务的总处理延迟，"
    "同时避免任务丢弃（即总延迟不超过最大容忍延迟）。若延迟超过最大容忍延迟，任务被丢弃，成本为常数惩罚。",
    "",
    "决策分析方法：",
    "对每个动作（本地执行；卸载到各边缘服务器）预估总延迟，选择延迟最低且不超过最大容忍延迟的动作。",
    "1. 本地执行延迟 = 本地CPU队列等待时间 + 本地计算时间。",
    "    - 本地计算时间取决于任务大小、任务的计算密度（每比特周期数）和本地CPU的处理能力（频率）。",
    "    - 本地CPU队列等待时间取决于本地队列中已有任务的完成时间。",
    "2. 边缘卸载延迟 = 传输队列等待时间 + 传输时间 + 边缘服务器队列等待时间 + 边缘服务器执行时间。",
    "    - 传输时间取决于任务大小和网络上传速率。",
    "    - 边缘服务器队列等待时间取决于服务器上所有任务的总队列长度、任务的计算密度和服务器被所有活跃任务共享后的处理能力。",
    "    - 边缘服务器执行时间取决于任务大小、任务的计算密度和服务器被所有活跃任务共享后的处理能力。",
    "    - 活跃任务数表示边缘服务器当前处理的并发任务数。在预估延迟时，我们假设新任务加入后，活跃任务数会加1。",
    "3. 负载历史：利用过去10个时间槽的活跃任务数来理解服务器负载的动态趋势。",
    "请基于以下输入数据，执行分析，选择总延迟最低且不超过最大容忍延迟的决策，并只输出决策名称。",
])


def render_paper(s):
    """返回 {"instruction": ..., "input": {...}}，键的顺序、标点（注意"活跃任务数"后面是全角逗号）、
    小数位数都和原作者数据一致。仿真器的状态先四舍五入到 4 位，标签也用四舍五入后的状态算。"""
    s = rounded(s, 4, 4)
    inp = {
        "任务大小": f"{s.d:.4f} Mbits",
        "最大延迟": f"{int(s.deadline)} 时间槽",
        "本地CPU队列等待时间": f"{s.local_wait:.4f} 时间槽",
        "网络队列等待时间": f"{s.net_wait:.4f} 时间槽",
        "网络上传速率": f"{s.rate:.4f} Mbits/时间槽",
        "时间槽时长": f"{C.SLOT_S}秒",
        "丢弃惩罚": C.DROP_PENALTY,
        "本地设备": (f"类型 移动设备, CPU频率 {s.local_ghz} GHz, 每比特周期数 {C.CYCLES_PER_BIT}, "
                 f"传输速率 {s.rate / C.SLOT_S:.1f} Mbps"),
    }
    for k, v in enumerate(s.servers, start=1):
        hist = ", ".join(f"{h:.1f}" for h in v.history)
        inp[f"边缘服务器{k}"] = (f"类型 {v.kind}, CPU频率 {v.ghz} GHz, 每比特周期数 {C.CYCLES_PER_BIT}, "
                             f"活跃任务数{v.active}，服务器总队列长度{v.backlog:.4f} Mbits, "
                             f"过去10个时间槽负载: [{hist}]")
    return {"instruction": PAPER_INSTRUCTION, "input": inp}


def paper_action_name(a):
    return "本地执行" if a == 0 else f"卸载到边缘服务器{a}"


def paper_action_index(name):
    """"本地执行" → 0，"卸载到边缘服务器7" → 7；认不出来返回 None。"""
    if name == "本地执行":
        return 0
    m = re.fullmatch(r"卸载到边缘服务器(\d+)", name)
    return int(m.group(1)) if m else None


# ===== 第 2 段：压缩格式 =====
LETTERS = "ABCDEFGHIJKL"          # A = 本地，B = 服务器 1，……，L = 服务器 11
COMPACT_SYSTEM = (
    "你是边缘计算任务卸载调度器。时间单位是槽（0.1秒），截止期10槽。"
    "A=本地执行（2.5GHz）；B、C、D…=卸载到服务器1、2、3…，要先上传（1.4Mbits/槽）。"
    "每比特297个CPU周期；服务器由活跃任务平分算力，新任务加入后活跃数加1。"
    "选总延迟最小且不超过截止期的动作，只输出一个字母。"
)


def compact_state(s):
    """压缩格式用的精度：任务大小 2 位小数，其余 1 位。训练标签必须用这个函数的输出来算。"""
    return rounded(s, nd_size=2, nd_other=1)


def render_compact(s):
    """返回 (messages, 四舍五入后的状态)。messages 是对话格式，Colab 训练和 Jetson 推理都直接用。"""
    s = compact_state(s)
    lines = [f"任务{s.d:.2f}Mbits 上传排队{s.net_wait:.1f}", f"A 本地 排队{s.local_wait:.1f}"]
    for k, v in enumerate(s.servers, start=1):
        lines.append(f"{LETTERS[k]} {v.ghz}GHz 活跃{v.active} 积压{v.backlog:.1f}")
    messages = [{"role": "system", "content": COMPACT_SYSTEM},
                {"role": "user", "content": "\n".join(lines)}]
    return messages, s


def parse_compact(user_text):
    """render_compact 的逆过程：从 user 文本还原状态。测试用它证明"提示词里的信息足够算出标签"。"""
    lines = user_text.split("\n")
    m = re.fullmatch(r"任务([\d.]+)Mbits 上传排队([\d.]+)", lines[0])
    d, nw = float(m.group(1)), float(m.group(2))
    lw = float(re.fullmatch(r"A 本地 排队([\d.]+)", lines[1]).group(1))
    servers = []
    for k, line in enumerate(lines[2:], start=1):
        m = re.fullmatch(r"([A-L]) ([\d.]+)GHz 活跃(\d+) 积压([\d.]+)", line)
        assert m.group(1) == LETTERS[k]
        servers.append(ServerView(ghz=float(m.group(2)), active=int(m.group(3)), backlog=float(m.group(4))))
    return State(d=d, local_wait=lw, net_wait=nw, servers=servers)
