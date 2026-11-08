"""Prepare a small Colab code/data archive; inspect large weights by headers only.

No model loading, GPU imports, training, network requests, or source-repo writes.
Weight payload hashes are deliberately opt-in; headers do not prove payload identity.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import zipfile


DATA_FILES = ("data/s4/grpo_train.jsonl", "data/s4/grpo_val.jsonl", "data/s3/friend_clean.jsonl",
              "data/s3/test_seen.jsonl", "data/s3/test_unseen.jsonl")
SOURCE_META = ("data/s3/meta.json", "data/s4/meta.json", "data/s4/grpo_reward.py")
CODE_DIRS = ("sim", "deploy", "eval", "train")
MODEL_META = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
              "chat_template.jinja")
DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "BOOL": 1,
               "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8,
               "F8_E4M3": 1, "F8_E5M2": 1}


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _key(row, field):
    return json.dumps(row[field], ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def audit_data(source_repo):
    """Validate dimensions and detect duplicate states/prompts across train, val, tests."""
    source_repo = Path(source_repo)
    rows_by_file = {}
    files = {}
    required = {"state", "paper_input", "servers", "n_actions", "best", "costs", "costs_compact", "ps_costs"}
    for rel in DATA_FILES:
        path = source_repo / rel
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"empty dataset: {rel}")
        for i, row in enumerate(rows):
            if not required <= row.keys():
                raise ValueError(f"missing schema fields: {rel}:{i}")
            n = row["n_actions"]
            if not isinstance(n, int) or n < 2 or row["servers"] != n - 1 or len(row["state"]["servers"]) != n - 1:
                raise ValueError(f"action/server dimensions disagree: {rel}:{i}")
            for field in ("costs", "costs_compact", "ps_costs"):
                if len(row[field]) != n or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in row[field]):
                    raise ValueError(f"invalid costs: {rel}:{i}:{field}")
            if not isinstance(row["best"], int) or not 0 <= row["best"] < n:
                raise ValueError(f"invalid best action: {rel}:{i}")
            if abs(row["costs"][row["best"]] - min(row["costs"])) > 1e-8:
                raise ValueError(f"best action is not minimum cost: {rel}:{i}")
            if not isinstance(row["paper_input"], dict):
                raise ValueError(f"invalid paper_input: {rel}:{i}")
        rows_by_file[rel] = rows
        unique = {field: len({_key(r, field) for r in rows}) for field in ("state", "paper_input")}
        if any(count != len(rows) for count in unique.values()):
            raise ValueError(f"duplicate states/prompts: {rel}")
        files[rel] = {"rows": len(rows), "bytes": path.stat().st_size, "sha256": file_sha256(path),
                      "unique": unique, "fields": sorted(set.intersection(*(set(r) for r in rows)))}
    overlaps = {}
    for i, train in enumerate(DATA_FILES[:2]):
        for other in DATA_FILES[i + 1:]:
            counts = {field: len({_key(r, field) for r in rows_by_file[train]} &
                                 {_key(r, field) for r in rows_by_file[other]}) for field in ("state", "paper_input")}
            overlaps[train + " :: " + other] = counts
            if any(counts.values()):
                raise ValueError(f"dataset overlap: {train} / {other}: {counts}")
    return {"key_definition": "canonical JSON of complete state (including history) and paper_input separately",
            "files": files, "overlaps": overlaps}


def model_manifest(model_dir, full_sha256=False, previous_manifest=None):
    """Check shard membership, dtype/shape byte counts and exact payload coverage."""
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    if not shard_names or any(Path(n).name != n for n in shard_names):
        raise ValueError("invalid index shard names")
    actual_files = {p.name for p in model_dir.glob("*.safetensors")}
    if set(shard_names) != actual_files:
        raise ValueError(f"index shard files disagree: missing={set(shard_names) - actual_files}, extra={actual_files - set(shard_names)}")
    found = {}
    shards = []
    dtypes = Counter()
    parameters = payload_total = 0
    for name in shard_names:
        path = model_dir / name
        size = path.stat().st_size
        with path.open("rb") as f:
            first = f.read(8)
            if len(first) != 8:
                raise ValueError(f"truncated shard header: {name}")
            header_bytes = struct.unpack("<Q", first)[0]
            if not 0 < header_bytes <= min(100 * 1024 * 1024, size - 8):
                raise ValueError(f"invalid shard header size: {name}")
            raw = f.read(header_bytes)
        header = json.loads(raw)
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        payload = size - 8 - header_bytes
        spans = []
        for tensor, spec in tensors.items():
            if tensor in found:
                raise ValueError(f"duplicate tensor across shards: {tensor}")
            found[tensor] = name
            dtype = spec["dtype"]
            shape = spec["shape"]
            if dtype not in DTYPE_BYTES or any(not isinstance(d, int) or d < 0 for d in shape):
                raise ValueError(f"invalid tensor dtype/shape: {tensor}")
            count = math.prod(shape)
            start, end = spec["data_offsets"]
            if not all(isinstance(x, int) for x in (start, end)) or not 0 <= start <= end <= payload:
                raise ValueError(f"tensor offsets exceed payload size: {tensor}")
            if end - start != count * DTYPE_BYTES[dtype]:
                raise ValueError(f"tensor shape/bytes disagree: {tensor}")
            spans.append((start, end))
            parameters += count
            dtypes[dtype] += 1
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ValueError(f"payload offset gap/overlap: {name}")
            cursor = end
        if cursor != payload:
            raise ValueError(f"payload size disagrees with tensor offsets: {name}")
        payload_total += payload
        shard = {"file": name, "bytes": size, "header_bytes": header_bytes,
                 "header_sha256": hashlib.sha256(raw).hexdigest(), "tensor_count": len(tensors),
                 "payload_bytes": payload, "dtypes": dict(Counter(s["dtype"] for s in tensors.values()))}
        if full_sha256:
            shard["sha256"] = file_sha256(path)
        shards.append(shard)
    if found != weight_map:
        raise ValueError("index weight_map disagrees with shard tensor headers")
    meta = index.get("metadata", {})
    if meta.get("total_size", payload_total) != payload_total or meta.get("total_parameters", parameters) != parameters:
        raise ValueError("index metadata totals disagree with tensor shape/bytes")
    result = {"status": "local_headers_validated_not_colab_training", "source_model_dir": str(model_dir.resolve()),
            "integrity_scope": "file sizes, headers, index membership, dtype/shape/offsets; payload SHA256 only when present",
            "config": json.loads(config_path.read_text(encoding="utf-8")), "config_sha256": file_sha256(config_path),
            "index_sha256": file_sha256(index_path), "tensor_count": len(found), "total_parameters": parameters,
            "total_payload_bytes": payload_total, "dtypes": dict(dtypes), "shards": shards}
    if previous_manifest is not None and not full_sha256:
        previous_path = Path(previous_manifest)
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        identity = ("config_sha256", "index_sha256", "tensor_count", "total_parameters", "total_payload_bytes")
        old_shards = {s["file"]: s for s in previous["shards"]}
        if any(previous.get(k) != result[k] for k in identity) or set(old_shards) != set(shard_names):
            raise ValueError("previous model manifest identity disagrees")
        for shard in shards:
            old = old_shards[shard["file"]]
            if any(old.get(k) != shard[k] for k in ("bytes", "header_sha256", "tensor_count", "payload_bytes")):
                raise ValueError("previous shard manifest identity disagrees")
            digest = old.get("sha256", "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("previous model manifest has no complete payload SHA256")
            shard["sha256"] = digest
        result["payload_hash_source"] = {"file": str(previous_path.resolve()), "sha256": file_sha256(previous_path),
                                         "verification": "previous payload hashes reused; current sizes and headers rechecked, payload not rehashed"}
    return result


def _git_record(path):
    def run(*args):
        result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"path": str(Path(path).resolve()), "commit": run("rev-parse", "HEAD"),
            "status_porcelain": run("status", "--porcelain")}


def pack_assets(repo_root, source_repo, model_dir, out, full_sha256=False, previous_model_manifest=None):
    """Archive current isolated code plus existing cleaned data; keep weights separate."""
    repo_root, source_repo, model_dir, out = map(Path, (repo_root, source_repo, model_dir, out))
    audit = audit_data(source_repo)
    model = model_manifest(model_dir, full_sha256=full_sha256, previous_manifest=previous_model_manifest)
    entries = {}
    for rel in DATA_FILES + SOURCE_META:
        p = source_repo / rel
        if p.is_file():
            entries[rel] = p.read_bytes()
    for directory in CODE_DIRS:
        for p in sorted((repo_root / directory).rglob("*.py")):
            if "__pycache__" not in p.parts:
                entries[p.relative_to(repo_root).as_posix()] = p.read_bytes()
    for name in MODEL_META:
        p = model_dir / name
        if p.is_file():
            entries["model_metadata/" + name] = p.read_bytes()
    entries["model_metadata/model.safetensors.index.json"] = (model_dir / "model.safetensors.index.json").read_bytes()
    entries["model_manifest.json"] = (json.dumps(model, ensure_ascii=False, indent=2) + "\n").encode()
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "prepared_locally_not_colab_execution", "code_source": _git_record(repo_root),
                "data_source": _git_record(source_repo), "data_audit": audit,
                "weight_upload": "Upload each listed .safetensors shard separately, without compression; copy model_metadata into that model folder.",
                "files": {n: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in sorted(entries.items())}}
    entries["s7_assets_manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr(name, data)
    return {"archive": str(out.resolve()), "archive_bytes": out.stat().st_size, "archive_sha256": file_sha256(out),
            "files": len(entries), "data_rows": {p: m["rows"] for p, m in audit["files"].items()},
            "model_tensors": model["tensor_count"], "weight_shards": model["shards"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--source-repo", required=True, help="Existing data/s3 and data/s4; read only")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True, help="Small code/data ZIP; does not contain model weight shards")
    ap.add_argument("--full-model-sha256", action="store_true", help="Also read every weight payload to hash it (slow)")
    ap.add_argument("--model-manifest", default=None, help="Reuse earlier payload hashes after checking current sizes/headers")
    args = ap.parse_args()
    print(json.dumps(pack_assets(args.repo_root, args.source_repo, args.model_dir, args.out,
                                full_sha256=args.full_model_sha256,
                                previous_model_manifest=args.model_manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
