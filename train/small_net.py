"""可选对照：一个约 1300 个参数的"逐选项打分"小网络，学同样的软标签，在同样的测试集上作答。

在仓库根目录运行（Mac 上，激活 .venv，先跑过 python -m train.make_data）：
  python -m train.small_net
结果写进 runs/s3/small_net_<UTC>/：config.json、preds.csv（和 Colab 的格式一样），eval.open_loop 直接能读。

做法：
  1. 每条状态从压缩提示词还原（和 LLM 学生看到的信息、精度完全一样），拆成"每个选项一行特征"：
       [是否本地, 任务大小, 要排的队, CPU 频率, 活跃任务数, 积压]
     本地那行"要排的队"是本地 CPU 队列，服务器那几行是上传队列
  2. 同一个小 MLP（6 → 32 → 32 → 1）给每个选项打一个分；分最高的就是决策。选项数可变，3 台、11 台都能用
  3. 目标和 LLM 一样：软标签 softmax(−(c − c_min) / T)，损失 = 交叉熵（和 KL 只差一个与模型无关的常数）
  4. 训练完在 4 份评测集上作答，记下单次决策的推理耗时
只用 numpy。
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

from eval.closed_loop import git_version
from sim import config as C
from sim.render import parse_compact

MAX_OPT = 12          # 本地 + 最多 11 台服务器
N_FEAT = 6
HIDDEN = 32
SPLITS_EVAL = ("val", "test_seen", "test_unseen", "friend_clean")


# ===== 第 1 段：数据 → 特征 =====
def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def option_features(user_text):
    """压缩提示词 → 每个选项一行特征（列表的列表）。"""
    s = parse_compact(user_text)
    rows = [[1.0, s.d, s.local_wait, C.LOCAL_GHZ, 0.0, 0.0]]
    for v in s.servers:
        rows.append([0.0, s.d, s.net_wait, v.ghz, float(v.active), v.backlog])
    return rows


def pack(rows):
    """一组行 → X（条数 × 12 × 6 的特征）、M（哪些选项存在，1/0）、COST（压缩精度的成本，不存在的选项为 inf）。"""
    n = len(rows)
    X = np.zeros((n, MAX_OPT, N_FEAT))
    M = np.zeros((n, MAX_OPT))
    COST = np.full((n, MAX_OPT), np.inf)
    for i, r in enumerate(rows):
        f = option_features(r["user"])
        for a in range(len(f)):
            for j in range(N_FEAT):
                X[i, a, j] = f[a][j]
            M[i, a] = 1.0
            COST[i, a] = r["costs_compact"][a]
    return X, M, COST


def soft_targets(COST, M, T):
    """软标签：每行 p_a = exp(−(c_a − c_min) / T) 再归一化；不存在的选项为 0。"""
    z = np.where(M > 0, -COST / T, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z) * M
    return p / p.sum(axis=1, keepdims=True)


# ===== 第 2 段：小网络 =====
class ScorerMLP:
    """同一个 MLP 给每个选项打分：6 → 32 → 32 → 1，ReLU。特征先按训练集的均值和标准差标准化。"""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / N_FEAT), (N_FEAT, HIDDEN))
        self.b1 = np.zeros(HIDDEN)
        self.W2 = rng.normal(0, np.sqrt(2 / HIDDEN), (HIDDEN, HIDDEN))
        self.b2 = np.zeros(HIDDEN)
        self.W3 = rng.normal(0, np.sqrt(1 / HIDDEN), (HIDDEN, 1))
        self.b3 = np.zeros(1)
        self.names = ("W1", "b1", "W2", "b2", "W3", "b3")
        self.m = {}
        self.v = {}
        for p in self.names:
            self.m[p] = 0.0
            self.v[p] = 0.0
        self.t = 0
        self.mu = np.zeros(N_FEAT)
        self.sd = np.ones(N_FEAT)

    def n_params(self):
        total = 0
        for p in self.names:
            total += getattr(self, p).size
        return total

    def fit_scaler(self, X, M):
        """只用存在的选项算每个特征的均值和标准差。"""
        flat = X[M > 0]
        self.mu = flat.mean(axis=0)
        self.sd = flat.std(axis=0) + 1e-6

    def forward(self, X):
        """X：条数 × 12 × 6 → 每个选项的分数（条数 × 12），以及反向传播要用的中间量。"""
        xn = (X - self.mu) / self.sd
        a1 = xn @ self.W1 + self.b1
        h1 = np.maximum(a1, 0)
        a2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(a2, 0)
        score = (h2 @ self.W3 + self.b3)[..., 0]
        return score, (xn, a1, h1, a2, h2)

    def train_step(self, X, M, P, lr):
        """一批数据上的一步 Adam。

        1. 前向：分数 → 只在存在的选项上做 softmax，得到 q
        2. 损失 = −Σ p × log q（交叉熵），对分数的梯度是 (q − p)
        3. 链式法则逐层往回传，得到每个参数的梯度
        4. Adam 更新
        """
        score, (xn, a1, h1, a2, h2) = self.forward(X)
        z = np.where(M > 0, score, -1e9)
        z = z - z.max(axis=1, keepdims=True)
        q = np.exp(z) * M
        q = q / q.sum(axis=1, keepdims=True)
        loss = -np.sum(P * np.log(q + 1e-12)) / len(X)
        ds = (q - P) * M / len(X)
        g = {}
        g["W3"] = np.einsum("boh,bo->h", h2, ds)[:, None]
        g["b3"] = np.array([ds.sum()])
        da2 = ds[..., None] * self.W3[:, 0] * (a2 > 0)
        g["W2"] = np.einsum("boi,boj->ij", h1, da2)
        g["b2"] = da2.sum(axis=(0, 1))
        da1 = (da2 @ self.W2.T) * (a1 > 0)
        g["W1"] = np.einsum("boi,boj->ij", xn, da1)
        g["b1"] = da1.sum(axis=(0, 1))
        self.t += 1
        for p in self.names:
            self.m[p] = 0.9 * self.m[p] + 0.1 * g[p]
            self.v[p] = 0.999 * self.v[p] + 0.001 * g[p] ** 2
            m_hat = self.m[p] / (1 - 0.9 ** self.t)
            v_hat = self.v[p] / (1 - 0.999 ** self.t)
            setattr(self, p, getattr(self, p) - lr * m_hat / (np.sqrt(v_hat) + 1e-8))
        return float(loss)

    def pick(self, X, M):
        """每条选分数最高的存在的选项。"""
        score, _ = self.forward(X)
        score = np.where(M > 0, score, -np.inf)
        return score.argmax(axis=1)


# ===== 第 3 段：主程序 =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "s3"))
    ap.add_argument("--out", default=os.path.join("runs", "s3"))
    ap.add_argument("--T", type=float, default=0.3, help="软标签温度（槽），和 LLM 用同一个 T_BEST")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # 1. 读数据，拆特征，算软标签
    t0 = time.time()
    data = {}
    for split in ("train",) + SPLITS_EVAL:
        path = os.path.join(args.data, split + ".jsonl")
        if os.path.exists(path):
            data[split] = read_jsonl(path)
    Xtr, Mtr, Ctr = pack(data["train"])
    Ptr = soft_targets(Ctr, Mtr, args.T)
    print(f"训练集 {len(Xtr)} 条，拆特征用了 {time.time() - t0:.0f} 秒")

    # 2. 训练：每轮打乱，按批做 Adam；学习率按余弦从 lr 降到 0
    net = ScorerMLP(args.seed)
    net.fit_scaler(Xtr, Mtr)
    rng = np.random.default_rng(args.seed)
    steps_per_epoch = (len(Xtr) + args.batch - 1) // args.batch
    total = steps_per_epoch * args.epochs
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(len(Xtr))
        losses = []
        for s in range(0, len(Xtr), args.batch):
            idx = order[s:s + args.batch]
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * step / total))
            losses.append(net.train_step(Xtr[idx], Mtr[idx], Ptr[idx], lr))
            step += 1
        print(f"第 {epoch + 1}/{args.epochs} 轮  损失 {np.mean(losses):.4f}  累计 {time.time() - t0:.0f} 秒")

    # 3. 在评测集上作答，写 preds.csv（和 Colab 的格式一样：split, idx, pred）
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(args.out, f"small_net_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    quick = {}
    with open(os.path.join(out_dir, "preds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "idx", "pred"])
        for split in SPLITS_EVAL:
            if split not in data:
                continue
            rows = data[split]
            X, M, _ = pack(rows)
            picks = net.pick(X, M)
            hit = 0
            regret = 0.0
            for i, r in enumerate(rows):
                a = int(picks[i])
                w.writerow([split, i, a])
                if a == r["best"]:
                    hit += 1
                regret += r["costs"][a] - r["costs"][r["best"]]
            quick[split] = {"acc_pct": round(100.0 * hit / len(rows), 2), "regret": round(regret / len(rows), 4)}
            print(split, quick[split])

    # 4. 单次决策耗时：一条一条地算（部署时就是一条一条来的），取 2000 次的中位数
    rows = data["test_seen"][:2000]
    X, M, _ = pack(rows)
    times = []
    for i in range(len(rows)):
        t1 = time.perf_counter()
        net.pick(X[i:i + 1], M[i:i + 1])
        times.append(time.perf_counter() - t1)
    us = float(np.median(times)) * 1e6
    print(f"参数个数 {net.n_params()}，单次决策（只算网络，不含拆特征）中位数 {us:.1f} 微秒（本机实测）")

    params = {}
    for p in net.names:
        params[p] = getattr(net, p)
    np.savez(os.path.join(out_dir, "weights.npz"), mu=net.mu, sd=net.sd, **params)
    config = {"name": f"小网络 {net.n_params()} 参数", "model": "mlp-6-32-32-1", "model_tag": "小网络",
              "T": args.T, "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "seed": args.seed,
              "n_train": len(Xtr), "n_params": net.n_params(), "decide_us_median": round(us, 1),
              "quick_metrics": quick, "git": git_version(), "started_utc": stamp}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print("写好了：", out_dir)


if __name__ == "__main__":
    main()
