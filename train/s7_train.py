"""Paired COMLLM-7B reward ablation, without downloading or training at import time.

    python -m train.s7_train --base /path/to/SFT-7B --s3-data /path/to/s3_data \
      --s4-data /path/to/s4_data --out /path/to/s7_runs --group lacs --mode smoke --resume

Repeat with ``--group nolacs``; formal adapters are s7_runs/{lacs,nolacs}/lora.
Smoke artifacts live under s7_runs/smoke instead. ``--validate-only`` never imports torch.
This is S4's single-update, hand-written GRPO loop adapted to the pinned COMLLM reward;
it is a controlled bf16 LoRA reimplementation, not a claim of exact nf4/TRL reproduction.
Wall-time limits are checked between optimizer updates and evaluation batches, not mid-kernel.
Only load checkpoints created by this runner: optimizer/RNG files use Python/torch serialization.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import shutil
import signal
import time

UPSTREAM = "17adedef587d5dbc02c167d147692a61179cdf4b"
SYSTEM = "你是一个MEC任务卸载专家。请严格按照用户的指示分析输入数据，输出卸载策略。不要包含任何解释、代码块标记或额外文本。"
EVAL_SPLITS = ("test_seen", "test_unseen", "friend_clean")
ACTION_RE = re.compile(r"本地执行|卸载到边缘服务器(\d+)")


@dataclass
class TrainConfig:
    base: str
    s3_data: str
    s4_data: str
    out: str
    group: str = "lacs"
    mode: str = "smoke"
    resume: bool = False
    eval_only: bool = False
    seed: int = 0
    epochs: int = 1
    group_size: int = 4
    prompts_per_step: int = 4
    lr: float = 5e-6
    beta: float = .005
    temperature: float = 1.3
    top_p: float = .95
    max_new_tokens: int = 15
    max_seq_length: int = 2048
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = .05
    loss_chunk: int = 1
    eval_batch: int = 4
    eval_every: int = 25
    checkpoint_every: int = 10
    max_steps: int | None = None
    max_minutes: float = 120.

    @property
    def lambda_lacs(self):
        return .3 if self.group == "lacs" else 0.

    @property
    def run_dir(self):
        root = Path(self.out) / "smoke" if self.mode == "smoke" else Path(self.out)
        return root / self.group

    def validate(self):
        if self.group not in ("lacs", "nolacs") or self.mode not in ("smoke", "full"):
            raise ValueError("group must be lacs/nolacs and mode smoke/full")
        for k in ("epochs", "group_size", "prompts_per_step", "loss_chunk", "eval_batch",
                  "eval_every", "checkpoint_every", "max_seq_length", "max_new_tokens"):
            if getattr(self, k) <= 0:
                raise ValueError(k + " must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_minutes <= 0 or self.lr <= 0 or self.temperature <= 0 or self.beta < 0:
            raise ValueError("invalid time or optimizer budget")
        if self.group_size < 2 or not 0 < self.top_p <= 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("invalid sampling/LoRA settings")


def digest_file(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_rows(directory, split):
    path = Path(directory) / (split + ".jsonl")
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                r = json.loads(line)
                r["_idx"], r["split"] = i, split
                rows.append(r)
    meta_path = Path(directory) / "meta.json"
    if meta_path.exists():
        expected = json.loads(meta_path.read_text()).get("files", {}).get(split, {})
        if expected.get("md5") and digest_file(path, "md5") != expected["md5"]:
            raise ValueError(f"dataset checksum mismatch: {path}")
        if "rows" in expected and len(rows) != expected["rows"]:
            raise ValueError(f"dataset row count mismatch: {path}")
    return rows


def state_keys(row):
    """Paper training preserves exact states; compact rounding is not a duplicate criterion."""
    keys = set()
    if "paper_input" in row:
        keys.add(("paper", json.dumps(row["paper_input"], sort_keys=True, ensure_ascii=False)))
    if isinstance(row.get("state"), dict):
        keys.add(("state", json.dumps(row["state"], sort_keys=True, ensure_ascii=False)))
    if not keys and "user" in row:
        keys.add(("compact", row["user"]))
    if not keys:
        raise ValueError("row needs paper_input or compact user for held-out checking")
    return keys


def load_data(cfg):
    """Recheck S4 against *all* three S3 tests; retain original row indices for scoring."""
    evaluation = {s: read_rows(cfg.s3_data, s) for s in EVAL_SPLITS}
    held = set().union(*(state_keys(r) for rows in evaluation.values() for r in rows))
    val_raw, train_raw = read_rows(cfg.s4_data, "grpo_val"), read_rows(cfg.s4_data, "grpo_train")
    val = [r for r in val_raw if not state_keys(r) & held]
    train_held = held | set().union(*(state_keys(r) for r in val))
    train, seen = [], set()
    removed, duplicated = 0, 0
    for r in train_raw:
        k = state_keys(r)
        if k & train_held:
            removed += 1
        elif k & seen:
            duplicated += 1
        else:
            seen.update(k)
            train.append(r)
    if not train or not val:
        raise ValueError("training or validation set empty after held-out filtering")
    instruction = json.loads((Path(cfg.s3_data) / "meta.json").read_text())["paper_instruction"]
    files = [Path(cfg.s3_data) / (s + ".jsonl") for s in EVAL_SPLITS]
    files += [Path(cfg.s4_data) / (s + ".jsonl") for s in ("grpo_train", "grpo_val")]
    files += [Path(cfg.s3_data) / "meta.json"]
    audit = {"source_train": len(train_raw), "source_val": len(val_raw), "train": len(train), "val": len(val),
             "train_heldout_removed": removed, "train_duplicates_removed": duplicated,
             "val_heldout_removed": len(val_raw) - len(val), "remaining_overlap": 0,
             "eval_rows": {s: len(v) for s, v in evaluation.items()},
             "sha256": {p.name: digest_file(p) for p in files}}
    return {"train": train, "val": val, "eval": evaluation, "instruction": instruction, "audit": audit}


def model_identity(base):
    base = Path(base)
    config = json.loads((base / "config.json").read_text())
    if config.get("hidden_size") != 3584:
        raise ValueError("expected original Qwen2.5 SFT-7B (hidden_size=3584)")
    weights = sorted(list(base.glob("*.safetensors")) + list(base.glob("pytorch_model*.bin")))
    if not weights:
        raise ValueError("base directory contains no model weights")
    files = sorted({*weights, *base.glob("*.json"), *base.glob("*.model"), *base.glob("*.txt")})
    return {"base": str(base.resolve()), "config": config,
            "files": {p.name: {"bytes": p.stat().st_size, "sha256": digest_file(p)} for p in files}}


def experiment_identity(cfg, data):
    common = dataclasses.asdict(cfg)
    for k in ("group", "resume", "eval_only", "max_minutes", "out", "base", "s3_data", "s4_data"):
        common.pop(k)
    code_files = [Path(__file__), Path(__file__).with_name("s7_reward.py")]
    return {"settings": common, "base_model": model_identity(cfg.base), "data": data["audit"],
            "upstream_commit": UPSTREAM,
            "implementation_sha256": {p.name: digest_file(p) for p in code_files if p.exists()}}


def paper_messages(row, instruction):
    lines = []
    for k, v in row["paper_input"].items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        lines.append(k + ": " + str(v))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction +
            "\n\n**输入数据**：\n" + "\n".join(lines)}]


def parse_prediction(text, servers, strict=False):
    text = text.strip()
    m = ACTION_RE.fullmatch(text) if strict else ACTION_RE.search(text)
    if m is None:
        return -1
    if m.group(0) == "本地执行":
        return 0
    k = int(m.group(1))
    return k if 1 <= k <= servers else -1


def training_batches(rows, cfg):
    rng = random.Random(cfg.seed)
    batches = []
    for _ in range(cfg.epochs):
        ids = list(range(len(rows)))
        rng.shuffle(ids)
        for start in range(0, len(ids), cfg.prompts_per_step):
            batches.append([rows[i] for i in ids[start:start + cfg.prompts_per_step]])
    limit = cfg.max_steps if cfg.max_steps is not None else (8 if cfg.mode == "smoke" else len(batches))
    return batches[:limit]


def rng_state():
    state = {"python": random.getstate()}
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def restore_rng(state):
    random.setstate(state["python"])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])


def save_checkpoint(run_dir, engine, progress):
    """Publish latest only after all adapter/optimizer/RNG state is on disk."""
    folder = run_dir / "checkpoints"
    folder.mkdir(exist_ok=True)
    # Never replace/delete the target of latest.json before its successor is published.
    name = f"step_{progress['step']:06d}_{time.time_ns()}"
    tmp = folder / (name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    engine.save_state(tmp / "engine.pt")
    with (tmp / "rng.pkl").open("wb") as f:
        pickle.dump(rng_state(), f)
    atomic_json(tmp / "progress.json", progress)
    target = folder / name
    os.replace(tmp, target)
    atomic_json(run_dir / "latest.json", {"checkpoint": "checkpoints/" + name})
    # Keep two complete generations, so an interrupted save does not remove the last usable one.
    older = sorted(p for p in folder.glob("step_*") if p.is_dir() and not p.name.endswith(".tmp"))[:-2]
    for p in older:
        shutil.rmtree(p)


def load_checkpoint(run_dir, engine):
    path = run_dir / json.loads((run_dir / "latest.json").read_text())["checkpoint"]
    engine.load_state(path / "engine.pt")
    with (path / "rng.pkl").open("rb") as f:
        restore_rng(pickle.load(f))
    return json.loads((path / "progress.json").read_text())


def log_prefix(path):
    """Discard an interrupted final JSONL write, never an internally corrupt record."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row.get("step"), int):
                raise ValueError("log record has no integer step")
        except (ValueError, AttributeError):
            if i == len(lines) - 1 and not line.endswith("\n"):
                break
            raise ValueError(f"corrupt internal training log record {i + 1}") from None
        rows.append(row)
    return rows


def prediction_prefix(path, fields):
    """Read complete CSV records, including quoted multiline text; repair only an unfinished tail."""
    text = path.read_text(encoding="utf-8")
    reader = csv.reader(io.StringIO(text), strict=True)
    if next(reader, None) != fields:
        raise ValueError("prediction CSV header does not match this runner")
    rows, done = [], set()
    while True:
        try:
            row = next(reader)
        except StopIteration:
            break
        except csv.Error:
            if reader.line_num == len(text.splitlines()) and not text.endswith("\n"):
                break
            raise ValueError("corrupt internal prediction CSV") from None
        try:
            if len(row) != len(fields):
                raise ValueError("incomplete prediction record")
            r = dict(zip(fields, row))
            key = r["split"], int(r["idx"])
            int(r["pred"])
            int(r["pred_strict"])
            if key in done:
                raise ValueError("duplicate prediction record")
        except ValueError:
            if reader.line_num == len(text.splitlines()) and not text.endswith("\n"):
                break
            raise ValueError("corrupt internal prediction CSV record") from None
        rows.append(r)
        done.add(key)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return done


def write_predictions(run_dir, engine, data, cfg, expired):
    path = run_dir / "preds.csv"
    fields = ["split", "idx", "pred", "raw", "pred_strict"]
    done = prediction_prefix(path, fields) if path.exists() else set()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0:
            w.writeheader()
        for split, rows in data["eval"].items():
            todo = [r for r in rows if (split, r["_idx"]) not in done]
            for start in range(0, len(todo), cfg.eval_batch):
                if expired():
                    return False
                batch = todo[start:start + cfg.eval_batch]
                texts = engine.answers(batch)
                if len(texts) != len(batch):
                    raise ValueError("evaluation output count differs from input")
                for r, text in zip(batch, texts):
                    w.writerow({"split": split, "idx": r["_idx"], "raw": text,
                                "pred": parse_prediction(text, r["servers"]),
                                "pred_strict": parse_prediction(text, r["servers"], strict=True)})
                f.flush()
    return True


def run(cfg, engine_factory=None, clock=None):
    cfg.validate()
    clock = clock or time.monotonic
    start = clock()
    data = load_data(cfg)
    identity = experiment_identity(cfg, data)
    run_dir = cfg.run_dir
    pair_dir = run_dir.parent
    pair_dir.mkdir(parents=True, exist_ok=True)
    pair_path = pair_dir / "pair.json"
    if pair_path.exists() and json.loads(pair_path.read_text()) != identity:
        raise ValueError("paired experiment identity changed (seed/base/data/settings); use a separate output root")
    if run_dir.exists() and not cfg.resume:
        raise ValueError("run already exists; use --resume to restore it")
    run_dir.mkdir(exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()).get("identity") != identity:
        raise ValueError("resume identity differs from the saved experiment")
    atomic_json(pair_path, identity)
    atomic_json(run_dir / "data_audit.json", data["audit"])
    config = {"name": "COMLLM-7B复现" if cfg.group == "lacs" else "GRPO-7B复现（同奖励λ=0）",
              "model_tag": "复现 7B", "group": cfg.group, "lambda_lacs": cfg.lambda_lacs,
              "format": "repo", "decode": "greedy_search", "max_new_tokens": cfg.max_new_tokens,
              "precision": "bf16 LoRA; upstream uses nf4 QLoRA", "identity": identity,
              "algorithm": "S4 single-update GRPO, ratio=1, k3 KL, cosine LR; no TRL dependency",
              "source_deviations": ["bf16 base instead of nf4 QLoRA", "hand-written single-update GRPO instead of TRL",
                                    "AdamW instead of paged_adamw_32bit", "cleaned held-out S4 rows instead of upstream raw data"],
              "effective_prompts_per_update": cfg.prompts_per_step,
              "completions_per_update": cfg.prompts_per_step * cfg.group_size}
    atomic_json(config_path, config)
    random.seed(cfg.seed)
    try:
        import numpy as np
        np.random.seed(cfg.seed)
    except ImportError:
        pass
    engine = (engine_factory or TorchEngine)(cfg, data["instruction"])
    config["runtime"] = getattr(engine, "versions", {})
    atomic_json(config_path, config)
    interrupted = [False]
    previous_sigint = None
    if __import__("threading").current_thread() is __import__("threading").main_thread():
        previous_sigint = signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
    expired = lambda: interrupted[0] or clock() - start >= cfg.max_minutes * 60
    progress = {"step": 0, "elapsed_train_s": 0., "training_complete": False}
    status = "failed"
    try:
        train, dropped = engine.prepare(data["train"])
        val, val_dropped = engine.prepare(data["val"])
        if not train or not val:
            raise ValueError("all training/validation prompts exceed the token limit")
        schedule = training_batches(train, cfg)
        progress["target_steps"] = len(schedule)
        if (run_dir / "latest.json").exists():
            progress = load_checkpoint(run_dir, engine)
            if progress["target_steps"] != len(schedule):
                raise ValueError("tokenized training schedule changed since checkpoint")
        else:
            if cfg.eval_only:
                raise ValueError("--eval-only requires a completed training checkpoint")
            save_checkpoint(run_dir, engine, progress)
        atomic_json(run_dir / "token_audit.json", {"train_dropped": dropped, "val_dropped": val_dropped,
                                                   "train_rows": len(train), "target_steps": len(schedule)})
        if cfg.eval_only and not progress["training_complete"]:
            raise ValueError("training is incomplete; resume training before --eval-only")
        # A resumed step can only appear once in the published log (crashes may leave a newer row).
        log_path = run_dir / "train_log.jsonl"
        if log_path.exists():
            kept = log_prefix(log_path)
            log_path.write_text("".join(json.dumps(r) + "\n" for r in kept if r["step"] <= progress["step"]))
        with log_path.open("a", encoding="utf-8") as log:
            for step in range(progress["step"], len(schedule)):
                if expired():
                    break
                t = clock()
                st = engine.step(schedule[step], cfg.lambda_lacs, step, len(schedule))
                progress["elapsed_train_s"] += max(0., clock() - t)
                progress["step"] = step + 1
                log.write(json.dumps({**st, "step": step + 1, "elapsed_train_s": progress["elapsed_train_s"]}) + "\n")
                log.flush()
                print(f"{cfg.group} step {step + 1}/{len(schedule)} loss={st.get('loss')}", flush=True)
                if (step + 1) % cfg.checkpoint_every == 0 or step + 1 == len(schedule):
                    save_checkpoint(run_dir, engine, progress)
                if (step + 1) % cfg.eval_every == 0 and not expired():
                    # Evaluation consumes no training RNG: preserve the next GRPO sample sequence.
                    state = rng_state()
                    selected = val[:8] if cfg.mode == "smoke" else val
                    texts = []
                    try:
                        for start_val in range(0, len(selected), cfg.eval_batch):
                            if expired():
                                break
                            texts.extend(engine.answers(selected[start_val:start_val + cfg.eval_batch]))
                    finally:
                        restore_rng(state)
                    with (run_dir / "val_log.jsonl").open("a", encoding="utf-8") as f:
                        hits = sum(parse_prediction(t, r["servers"]) == r["best"] for r, t in zip(selected, texts))
                        f.write(json.dumps({"step": step + 1, "n": len(texts), "correct": hits,
                                            "complete": len(texts) == len(selected)}) + "\n")
        progress["training_complete"] = progress["step"] == len(schedule)
        save_checkpoint(run_dir, engine, progress)
        if not progress["training_complete"]:
            status = "interrupted" if interrupted[0] else "time_limit"
        else:
            engine.export_adapter(run_dir / "lora")
            if cfg.mode == "smoke":
                status = "smoke_complete"
            elif write_predictions(run_dir, engine, data, cfg, expired):
                status = "complete"
            else:
                status = "interrupted" if interrupted[0] else "time_limit"
        return {**progress, "status": status, "run_dir": str(run_dir),
                "expected_eval_rows": sum(map(len, data["eval"].values()))}
    finally:
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        atomic_json(run_dir / "summary.json", {**progress, "status": status, "group": cfg.group,
                                               "invocation_seconds": max(0., clock() - start)})
        engine.close()


class TorchEngine:
    """The costly model boundary; imported/constructed only for an actual Colab run."""
    def __init__(self, cfg, instruction):
        import torch
        import importlib.metadata
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        self.torch, self.cfg, self.instruction = torch, cfg, instruction
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("S7 requires a bf16-capable CUDA GPU; validate-only is CPU safe")
        self.versions = {"torch": torch.__version__, "cuda": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(0), "bf16": True,
                         "transformers": importlib.metadata.version("transformers"),
                         "peft": importlib.metadata.version("peft")}
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        self.tok = AutoTokenizer.from_pretrained(cfg.base)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(cfg.base, dtype=torch.bfloat16).to("cuda")
        base.config.use_cache = False
        lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                          bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lora)
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=0.)
        self.stops = list({self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")})
        self.stops = [s for s in self.stops if s is not None and s >= 0]

    def prepare(self, rows):
        kept = []
        for r in rows:
            text = self.tok.apply_chat_template(paper_messages(r, self.instruction), tokenize=False,
                                                add_generation_prompt=True)
            r["p_ids"] = self.tok(text, add_special_tokens=False)["input_ids"]
            if len(r["p_ids"]) + self.cfg.max_new_tokens <= self.cfg.max_seq_length:
                kept.append(r)
        return kept, len(rows) - len(kept)

    def padded(self, rows):
        torch = self.torch
        width = max(len(r["p_ids"]) for r in rows)
        ids = torch.full((len(rows), width), self.tok.pad_token_id, dtype=torch.long, device="cuda")
        mask = torch.zeros_like(ids)
        for i, r in enumerate(rows):
            ids[i, -len(r["p_ids"]):] = torch.tensor(r["p_ids"], device="cuda")
            mask[i, -len(r["p_ids"]):] = 1
        return ids, mask

    def logps(self, p, pm, c, cm):
        torch = self.torch
        ids, mask = torch.cat([p, c], 1), torch.cat([pm, cm], 1)
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        logits = self.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                            logits_to_keep=c.shape[1] + 1).logits[:, :-1, :].float()
        lp = torch.log_softmax(logits / self.cfg.temperature, dim=-1)
        return lp.gather(-1, c.unsqueeze(-1)).squeeze(-1)

    def step(self, batch, lambda_lacs, step, total_steps):
        from train.s7_reward import reward
        from train.grpo_reward import group_advantages
        torch, cfg = self.torch, self.cfg
        self.model.eval()
        p, pm = self.padded(batch)
        with torch.no_grad():
            gen = self.model.generate(input_ids=p, attention_mask=pm, do_sample=True,
                                      temperature=cfg.temperature, top_p=cfg.top_p, top_k=0,
                                      num_return_sequences=cfg.group_size, max_new_tokens=cfg.max_new_tokens,
                                      eos_token_id=self.stops, pad_token_id=self.tok.pad_token_id, use_cache=True)
        c = gen[:, p.shape[1]:]
        cm = torch.zeros_like(c)
        texts = []
        for i in range(len(c)):
            n = c.shape[1]
            for j, token in enumerate(c[i].tolist()):
                if token in self.stops:
                    n = j + 1
                    break
            cm[i, :n] = 1
            texts.append(self.tok.decode(c[i, :n], skip_special_tokens=True).strip())
        rewards = [reward(text, batch[i // cfg.group_size], lambda_lacs, seed=cfg.seed,
                          sample_key=(step, batch[i // cfg.group_size]["_idx"])) for i, text in enumerate(texts)]
        advantages, zero = group_advantages(rewards, cfg.group_size)
        adv = torch.tensor(advantages, dtype=torch.float32, device="cuda")
        p, pm = p.repeat_interleave(cfg.group_size, 0), pm.repeat_interleave(cfg.group_size, 0)
        self.opt.zero_grad(set_to_none=True)
        loss_sum, kl_sum = 0., 0.
        refs = []
        if cfg.beta:
            with torch.no_grad(), self.model.disable_adapter():
                for a in range(0, len(c), cfg.loss_chunk):
                    z = slice(a, a + cfg.loss_chunk)
                    refs.append(self.logps(p[z], pm[z], c[z], cm[z]))
        self.model.train()
        for k, a in enumerate(range(0, len(c), cfg.loss_chunk)):
            z = slice(a, a + cfg.loss_chunk)
            lp = self.logps(p[z], pm[z], c[z], cm[z])
            ratio = (lp - lp.detach()).exp()
            objective = -ratio * adv[z, None]
            kl = torch.zeros_like(lp)
            if cfg.beta:
                delta = refs[k] - lp
                kl = delta.exp() - delta - 1
                objective = objective + cfg.beta * kl
            count = cm[z].sum(1).clamp(min=1)
            loss = ((objective * cm[z]).sum(1) / count).sum() / len(c)
            loss.backward()
            loss_sum += loss.item()
            kl_sum += ((kl.detach() * cm[z]).sum(1) / count).sum().item() / len(c)
        torch.nn.utils.clip_grad_norm_(self.params, 1.)
        lr = cfg.lr * .5 * (1 + math.cos(math.pi * step / max(1, total_steps)))
        for group in self.opt.param_groups:
            group["lr"] = lr
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return {"loss": loss_sum, "kl": kl_sum, "lr": lr,
                "reward_mean": sum(rewards) / len(rewards), "zero_std_frac": zero / len(batch)}

    def answers(self, rows):
        torch = self.torch
        rows, dropped = self.prepare(rows)
        if dropped:
            raise ValueError("evaluation prompt exceeds max_seq_length; no silent test-row filtering")
        p, pm = self.padded(rows)
        self.model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            gen = self.model.generate(input_ids=p, attention_mask=pm, do_sample=False,
                                      max_new_tokens=self.cfg.max_new_tokens, eos_token_id=self.stops,
                                      pad_token_id=self.tok.pad_token_id, use_cache=True, repetition_penalty=1.)
        return [self.tok.decode(c, skip_special_tokens=True).strip() for c in gen[:, p.shape[1]:]]

    def save_state(self, path):
        from peft import get_peft_model_state_dict
        torch = self.torch
        torch.save({"adapter": get_peft_model_state_dict(self.model), "optimizer": self.opt.state_dict(),
                    "torch_cpu_rng": torch.get_rng_state(), "torch_cuda_rng": torch.cuda.get_rng_state_all()}, path)

    def load_state(self, path):
        from peft import set_peft_model_state_dict
        torch = self.torch
        state = torch.load(path, map_location="cuda", weights_only=False)
        set_peft_model_state_dict(self.model, state["adapter"])
        self.opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_cpu_rng"].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in state["torch_cuda_rng"]])

    def export_adapter(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)

    def close(self):
        del self.model, self.opt
        self.torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ("base", "s3-data", "s4-data", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--group", choices=("lacs", "nolacs"), required=True)
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    for name in ("resume", "eval-only", "validate-only"):
        ap.add_argument("--" + name, action="store_true")
    ap.add_argument("--max-minutes", type=float, default=120.)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--loss-chunk", type=int, default=1)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    args = vars(ap.parse_args(argv))
    validate_only = args.pop("validate_only")
    cfg = TrainConfig(**args)
    if validate_only:
        cfg.validate()
        data = load_data(cfg)
        identity = experiment_identity(cfg, data)
        print(json.dumps({"status": "validated", "run_dir": str(cfg.run_dir), "data": data["audit"],
                          "base_model": identity["base_model"], "lambda_lacs": cfg.lambda_lacs}, ensure_ascii=False, indent=2))
        return 0
    result = run(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in ("complete", "smoke_complete") else 75


if __name__ == "__main__":
    raise SystemExit(main())
