"""阶段 4 GRPO 的奖励：把模型的一段回答变成一个分数，越大越好。

两种奖励都只看这一步的成本（λ = 0，不带 LACS），区别只在非法回答怎么算：
  fixed  合法动作：−有效成本（成本超过截止期再加 ρ）
         非法回答（认不出动作，或服务器编号超出台数）：比这个状态下最差的合法动作再差 INVALID_MARGIN 槽
  repo   原作者仓库的写法：
         合法动作：−有效成本（截止期固定按 10 槽）+ 解析成功奖励 1
         服务器编号超出台数：成本按 5.0 算，再 + 解析成功奖励 1
         认不出动作：成本按 5.0 算
repo 的问题：5.0 槽比很多合法动作还便宜，编一个不存在的服务器反而得分更高。

两种回答格式：
  论文格式（A：原作者模型）  回答是一段中文，"本地执行" 或 "卸载到边缘服务器k"，用 parse 解析
  压缩格式（B：我们的模型）  回答是一个字母，A = 本地，B = 服务器 1 ……，用 parse_letter 解析
压缩格式只用 fixed 奖励。

动作编号约定和全项目一样：0 = 本地执行，k = 卸载到第 k 台服务器。
"""
import math
import re

RHO = 10.0                  # 超过截止期的惩罚（槽）
INVALID_MARGIN = 5.0        # fixed：非法回答比最差的合法动作再差这么多（槽）
REPO_INVALID_COST = 5.0     # repo：非法回答的成本（槽）
REPO_PARSE_BONUS = 1.0      # repo：认出一个动作（不管合不合法）就加的分
REPO_DEADLINE = 10.0        # repo：截止期写死的值（槽）

ACTION_RE = re.compile(r"本地执行|卸载到边缘服务器(\d+)")
LETTERS = "ABCDEFGHIJKL"       # 压缩格式的动作字母，下标就是动作编号

# 解析结果的三种情况
OK = "ok"                   # 合法动作
BAD_SERVER = "bad_server"   # 认出了动作，但服务器编号（或字母）超出台数
NONE = "none"               # 认不出动作


def parse(text, n_servers):
    """一段回答 → (情况, 动作编号)。不是 OK 时动作编号是 −1。

    1. 在文字里找第一个"本地执行"或"卸载到边缘服务器k"
    2. 找不到 → (NONE, −1)
    3. 本地执行 → (OK, 0)
    4. 服务器编号在 1 到 n_servers 之间 → (OK, k)，否则 → (BAD_SERVER, −1)
    """
    m = ACTION_RE.search(text)
    if m is None:
        return NONE, -1
    if m.group(0) == "本地执行":
        return OK, 0
    k = int(m.group(1))
    if 1 <= k <= n_servers:
        return OK, k
    return BAD_SERVER, -1


def parse_letter(text, n_actions):
    """压缩格式的回答（一个字母）→ (情况, 动作编号)。

    1. 去掉首尾空白后不是 LETTERS 里的单个字母 → (NONE, −1)
    2. 字母的下标小于动作个数 → (OK, 下标)，否则 → (BAD_SERVER, −1)
    """
    t = text.strip()
    if len(t) != 1 or t not in LETTERS:
        return NONE, -1
    a = LETTERS.index(t)
    if a < n_actions:
        return OK, a
    return BAD_SERVER, -1


def effective_cost(cost, deadline):
    """一个合法动作的有效成本：超过截止期加 ρ。和开环评测 AL 的单条成本一样。"""
    if cost > deadline:
        return cost + RHO
    return cost


def reward_fixed(text, costs, deadline):
    """fixed 奖励（论文格式的回答）。costs：每个动作的成本（槽），长度 = 服务器台数 + 1。"""
    kind, a = parse(text, len(costs) - 1)
    return reward_fixed_action(kind, a, costs, deadline)


def reward_fixed_action(kind, a, costs, deadline):
    """fixed 奖励，输入是已经解析好的 (情况, 动作编号)，两种格式共用。

    1. 合法 → −有效成本
    2. 非法 → −(所有合法动作里最大的有效成本 + INVALID_MARGIN)
    """
    if kind == OK:
        return -effective_cost(costs[a], deadline)
    worst = 0.0
    for c in costs:
        e = effective_cost(c, deadline)
        if e > worst:
            worst = e
    return -(worst + INVALID_MARGIN)


def reward_repo(text, costs):
    """repo 奖励。

    1. 解析回答
    2. 认不出动作 → −5.0
    3. 服务器编号超出台数 → −5.0 + 1
    4. 合法 → −有效成本（截止期按 10 槽）+ 1
    """
    kind, a = parse(text, len(costs) - 1)
    if kind == NONE:
        return -REPO_INVALID_COST
    if kind == BAD_SERVER:
        return -REPO_INVALID_COST + REPO_PARSE_BONUS
    return -effective_cost(costs[a], REPO_DEADLINE) + REPO_PARSE_BONUS


def reward(style, text, costs, deadline):
    if style == "fixed":
        return reward_fixed(text, costs, deadline)
    if style == "repo":
        return reward_repo(text, costs)
    raise ValueError(style)


def group_advantages(rewards, group_size, eps=1e-4):
    """论文式 18：每组（同一道题的 group_size 个回答）内，(奖励 − 组内均值) ÷ (组内标准差 + eps)。

    返回 (优势列表, 奖励全相同的组数)。奖励全相同的组，优势全是 0，这道题这一步学不到东西。
    1. 按顺序每 group_size 个切成一组
    2. 算组内均值和标准差（除以 group_size，不是 group_size − 1）
    3. 每个回答的优势 = (奖励 − 均值) ÷ (标准差 + eps)
    4. 标准差小于 1e-6 的组记为"全相同"
    """
    assert len(rewards) % group_size == 0
    adv = []
    zero_std = 0
    for start in range(0, len(rewards), group_size):
        g = rewards[start:start + group_size]
        mean = 0.0
        for r in g:
            mean += r
        mean = mean / group_size
        var = 0.0
        for r in g:
            var += (r - mean) * (r - mean)
        std = math.sqrt(var / group_size)
        if std < 1e-6:
            zero_std += 1
        for r in g:
            adv.append((r - mean) / (std + eps))
    return adv, zero_std
