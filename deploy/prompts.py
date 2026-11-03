"""把状态拼成发给 llama-server 的请求，再把回答解析成动作编号。

两种提示词格式：
  compact：压缩提示词，只算 1 个 token（A=本地，B=服务器 1，……）。读法有两种：
           probs   （默认）不加语法，读首个 token 概率最高的 20 个候选，取其中概率最大的合法字母。
           grammar 用 GBNF 语法只允许合法字母，直接读生成的那个字母。
           两种读法选出的动作相同（只要前 20 名里有合法字母）；区别只在耗时：
           语法要逐个检查 15 万个词表 token 能不能接上，模型首选不是合法字母时这一步很贵。
  paper：  论文格式长提示词，固定生成 12 个 token，不加语法。论文原版模型的答案是
           {"action_name": "卸载到边缘服务器3"} 这样一段 JSON，用 Qwen 分词约 12 个 token。
           用来量"论文原版做法"在同一套部署上要花多久。

提示词用 Qwen2.5 的对话模板（ChatML）手工拼出来，和训练时 tokenizer.apply_chat_template 得到的文本一致：
  <|im_start|>system\n系统提示<|im_end|>\n<|im_start|>user\n用户内容<|im_end|>\n<|im_start|>assistant\n
"""
import json
import math
import re

from sim.render import LETTERS, paper_action_index, render_compact, render_paper

QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
COMPACT_N_PROBS = 20      # 读首个 token 概率最高的 20 个候选
PAPER_N_PREDICT = 12      # 论文格式固定生成 12 个 token
PAPER_ACTION_RE = re.compile(r"本地执行|卸载到边缘服务器\d+")


def chatml(messages):
    """messages（[{"role": ..., "content": ...}, ...]）→ 一整段提示词文本，末尾停在 assistant 开头。

    1. 每条消息写成 <|im_start|>角色\n内容<|im_end|>\n
    2. 最后加上 <|im_start|>assistant\n，模型从这里开始回答
    """
    text = ""
    for m in messages:
        text = text + "<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n"
    text = text + "<|im_start|>assistant\n"
    return text


def paper_messages(s):
    """论文格式的对话：系统提示用 Qwen 默认的那句，用户内容 = 说明文字 + 换行 + 状态字典的 JSON。"""
    p = render_paper(s)
    user = p["instruction"] + "\n" + json.dumps(p["input"], ensure_ascii=False)
    return [{"role": "system", "content": QWEN_DEFAULT_SYSTEM}, {"role": "user", "content": user}]


def compact_grammar(n_actions):
    """压缩格式的 GBNF 语法：只允许 A 到第 n_actions 个字母之一。例：3 台服务器 → root ::= [A-D]"""
    return "root ::= [A-" + LETTERS[n_actions - 1] + "]"


def build_llm_request(s, fmt, cache_prompt, decode="probs"):
    """状态 → 发给 llama-server /completion 的请求体。

    1. 按格式拼对话，转成 ChatML 文本
    2. compact：只算 1 个 token，要前 20 名候选的概率；decode = grammar 时再加 GBNF 语法
       paper：  固定生成 12 个 token（ignore_eos：遇到结束符也不停）
    3. temperature = 0：每次取概率最高的 token，同一个状态永远给同一个答案
    4. 只让 llama-server 回内容、概率、计时三项
    """
    body = {"temperature": 0, "cache_prompt": bool(cache_prompt), "stream": False,
            "response_fields": ["content", "completion_probabilities", "timings"]}   # 只要这三项，回复小一些
    if fmt == "compact":
        messages, _ = render_compact(s)
        body["n_predict"] = 1
        body["n_probs"] = COMPACT_N_PROBS
        if decode == "grammar":
            body["grammar"] = compact_grammar(s.n_actions)
    elif fmt == "paper":
        messages = paper_messages(s)
        body["n_predict"] = PAPER_N_PREDICT
        body["ignore_eos"] = True
    else:
        raise ValueError("未知格式：" + str(fmt))
    body["prompt"] = chatml(messages)
    return body


def action_probs(resp, n_actions):
    """从 llama-server 回复里读首个 token 的概率，换算成每个动作的概率。

    返回 (probs, p_chosen)：
      probs：长度 n_actions 的列表，第 a 项是字母 LETTERS[a] 在原始分布（没有语法过滤）里的概率；
             不在前 20 名候选里的记 0
      p_chosen：实际生成的那个 token 的原始概率
    回复里没有概率时返回 (None, None)。

    1. 取第一个生成 token 的记录
    2. 遍历前 20 名候选：文本正好是某个合法字母，就把 exp(logprob) 记到那个动作上
    3. 实际生成的 token 自己的 logprob 换成概率
    """
    cp = resp.get("completion_probabilities") or []
    if len(cp) == 0:
        return None, None
    first = cp[0]
    probs = [0.0] * n_actions
    for cand in first.get("top_logprobs", []):
        tok = cand.get("token", "")
        if len(tok) == 1 and tok in LETTERS:
            a = LETTERS.index(tok)
            if a < n_actions:
                probs[a] = math.exp(cand["logprob"])
    p_chosen = None
    if "logprob" in first:
        p_chosen = math.exp(first["logprob"])
    return probs, p_chosen


def parse_action(fmt, decode, resp, n_actions):
    """llama-server 的回复 → (动作编号, 每个动作的概率, 生成 token 的概率)；认不出动作时编号为 None。

    1. compact + probs：在前 20 名候选里找概率最大的合法字母；一个都没有就认不出来
    2. compact + grammar：生成的文本必须正好是一个合法字母
    3. paper：在生成的文本里找第一个"本地执行 / 卸载到边缘服务器k"
    4. 最后检查编号在 0 到 n_actions-1 之间
    """
    text = resp.get("content", "").strip()
    probs, p_chosen = None, None
    a = None
    if fmt == "compact":
        probs, p_chosen = action_probs(resp, n_actions)
        if decode == "grammar":
            if len(text) == 1 and text in LETTERS:
                a = LETTERS.index(text)
        elif probs is not None:
            best = -1
            for k in range(n_actions):
                if probs[k] > 0 and (best < 0 or probs[k] > probs[best]):
                    best = k
            if best >= 0:
                a = best
    elif fmt == "paper":
        m = PAPER_ACTION_RE.search(text)
        if m is not None:
            a = paper_action_index(m.group(0))
    if a is not None and (a < 0 or a >= n_actions):
        a = None
    return a, probs, p_chosen
