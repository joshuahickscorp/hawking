#!/usr/bin/env python3.12
"""Restart-safe, execution-free source census for the Doctor-v5 scale chain.

This worker reads safetensors headers and hashes source bytes.  It never imports a
model framework, allocates a tensor, trains a model, or produces quality evidence.
Its output is a bootstrap inventory that an independent parameter-manifest
authority may later review and sign.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA = "hawking.doctor_v5_source_census.v1"
CHECKPOINT_SCHEMA = "hawking.doctor_v5_source_census_checkpoint.v1"
REPORT_VERSION = "2026-07-12.1"
MAX_HEADER_BYTES = 256 * 1024 * 1024
HASH_CHUNK_BYTES = 8 * 1024 * 1024
_STOP = False
CLASSIFICATION_NOTE = "name-based conservative bootstrap; independent ownership review required"
AUTHORITY_BLOCKER = "role-separated parameter-manifest review and receipt required"
PROMOTION_BLOCKERS = (
    "no role-separated parameter-manifest receipt",
    "no diagnostic probe receipt",
    "no executable Doctor-v5 program or adapter allowlist",
    "no quality observation, artifact, sealed evaluation, or independent reproduction",
)

# Safetensors has added low-bit and float8 dtypes over time.  Unknown dtypes are
# rejected rather than guessed; packed formats are counted by declared shape but
# never promoted to an exact logical model-parameter claim.
DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8, "I8": 8,
    "U16": 16, "I16": 16,
    "U32": 32, "I32": 32,
    "U64": 64, "I64": 64,
    "F16": 16, "BF16": 16, "F32": 32, "F64": 64,
    "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8,
    "C64": 64, "C128": 128,
    "I4": 4, "U4": 4, "F4": 4,
    "F6_E2M3": 6, "F6_E3M2": 6,
}


class CensusError(ValueError):
    pass


class StopRequested(RuntimeError):
    pass


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without_hash(document: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != field}


def _stamp(document: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(document)
    result[field] = _hash_value(_without_hash(result, field))
    return result


def _is_sha(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, value: Any) -> None:
    """Write a durable checkpoint; inability to fsync is a hard failure."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CensusError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_bytes(raw: bytes, source: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CensusError(f"invalid JSON in {source}: {exc}") from exc


def _read_json(path: Path) -> Any:
    return _json_bytes(path.read_bytes(), str(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            if _STOP:
                raise StopRequested("stop requested while hashing")
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _checked_product(shape: list[int]) -> int:
    product = 1
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise CensusError("tensor shape dimensions must be nonnegative integers")
        product *= dim
        if product > 2**127:
            raise CensusError("tensor shape product is implausibly large")
    return product


def _ownership(name: str) -> str:
    lower = name.lower()
    if "embed_tokens" in lower or "word_embeddings" in lower or lower.endswith(".wte.weight"):
        return "token_embedding"
    if "lm_head" in lower or "output.weight" in lower:
        return "output_head"
    if ".experts." in lower or "expert" in lower:
        return "moe_expert"
    if "self_attn" in lower or "attention" in lower:
        return "attention"
    if ".mlp." in lower or "feed_forward" in lower:
        return "dense_mlp"
    if "norm" in lower:
        return "normalization"
    if lower.endswith(".bias"):
        return "bias"
    return "other_or_unclassified"


def _read_header(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise CensusError(f"source shard is not a regular non-symlink file: {path}")
    identity = _stat_identity(path)
    if identity["bytes"] < 10:
        raise CensusError(f"truncated safetensors file: {path}")
    with open(path, "rb") as handle:
        opened_before = os.fstat(handle.fileno())
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise CensusError(f"truncated safetensors length: {path}")
        header_length = int.from_bytes(raw_length, "little", signed=False)
        if not (1 < header_length <= MAX_HEADER_BYTES):
            raise CensusError(f"invalid safetensors header length {header_length}: {path}")
        if 8 + header_length > identity["bytes"]:
            raise CensusError(f"safetensors header exceeds file: {path}")
        raw_header = handle.read(header_length)
        handle.seek(0)
        digest = hashlib.sha256()
        while True:
            if _STOP:
                raise StopRequested("stop requested while hashing")
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        opened_after = os.fstat(handle.fileno())
    opened_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(opened_before, key) != getattr(opened_after, key) for key in opened_fields):
        raise CensusError(f"source shard changed while being inspected: {path}")
    header = _json_bytes(raw_header, str(path))
    if not isinstance(header, dict):
        raise CensusError(f"safetensors header is not an object: {path}")
    data_bytes = identity["bytes"] - 8 - header_length
    tensors: list[dict[str, Any]] = []
    intervals: list[tuple[int, int, str]] = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict):
                raise CensusError(f"__metadata__ is not an object: {path}")
            continue
        if not isinstance(name, str) or not name:
            raise CensusError(f"invalid tensor name in {path}")
        if not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise CensusError(f"invalid descriptor keys for {name} in {path}")
        dtype = descriptor["dtype"]
        shape = descriptor["shape"]
        offsets = descriptor["data_offsets"]
        if dtype not in DTYPE_BITS:
            raise CensusError(f"unknown dtype {dtype!r} for {name} in {path}")
        if not isinstance(shape, list):
            raise CensusError(f"shape is not a list for {name} in {path}")
        elements = _checked_product(shape)
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)):
            raise CensusError(f"invalid offsets for {name} in {path}")
        start, end = offsets
        if not (0 <= start <= end <= data_bytes):
            raise CensusError(f"out-of-range offsets for {name} in {path}")
        expected_bytes = math.ceil(elements * DTYPE_BITS[dtype] / 8)
        if end - start != expected_bytes:
            raise CensusError(
                f"dtype/shape byte count mismatch for {name} in {path}: "
                f"expected {expected_bytes}, found {end - start}"
            )
        intervals.append((start, end, name))
        tensors.append({
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "elements": elements,
            "data_offsets": offsets,
            "stored_bytes": end - start,
            "ownership_class": _ownership(name),
        })
    intervals.sort()
    cursor = 0
    for start, end, name in intervals:
        if start != cursor:
            kind = "overlap" if start < cursor else "gap"
            raise CensusError(f"{kind} before tensor {name} in {path}")
        cursor = end
    if cursor != data_bytes:
        raise CensusError(f"unaccounted trailing data in {path}")
    if not tensors:
        raise CensusError(f"safetensors shard contains no tensors: {path}")
    return {
        **identity,
        "header_bytes": header_length,
        "header_sha256": hashlib.sha256(raw_header).hexdigest(),
        "data_bytes": data_bytes,
        "tensor_count": len(tensors),
        "tensors": tensors,
        "file_sha256": digest.hexdigest(),
    }


def _source_layout(
    model_dir: Path,
) -> tuple[list[Path], dict[str, str] | None, Path | None, Path]:
    def relevant(path: Path) -> bool:
        return ".cache" not in path.relative_to(model_dir).parts

    root_index = model_dir / "model.safetensors.index.json"
    index_candidates = [root_index] if root_index.is_file() else [
        path for path in model_dir.rglob("model.safetensors.index.json") if relevant(path)
    ]
    if len(index_candidates) > 1:
        raise CensusError("multiple safetensors weight roots are ambiguous")
    index_path = index_candidates[0] if index_candidates else None
    weights_root = index_path.parent if index_path else model_dir
    actual = sorted(weights_root.glob("*.safetensors"))
    if any(path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode) for path in actual):
        raise CensusError("source shard inventory contains a symlink or non-regular file")
    if index_path is not None:
        if index_path.is_symlink() or not stat.S_ISREG(index_path.lstat().st_mode):
            raise CensusError("safetensors index is not a regular non-symlink file")
        index = _read_json(index_path)
        if not isinstance(index, dict) or set(index) != {"metadata", "weight_map"}:
            raise CensusError("invalid safetensors index keys")
        metadata = index.get("metadata")
        if (not isinstance(metadata, dict) or set(metadata) != {"total_size"}
                or isinstance(metadata.get("total_size"), bool)
                or not isinstance(metadata.get("total_size"), int)
                or metadata["total_size"] < 0):
            raise CensusError("safetensors index metadata.total_size is invalid")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise CensusError("safetensors index weight_map is absent or empty")
        for tensor, shard in weight_map.items():
            if not isinstance(tensor, str) or not tensor or not isinstance(shard, str) or not shard:
                raise CensusError("invalid safetensors index mapping")
            if Path(shard).name != shard:
                raise CensusError("safetensors index shard must be a basename")
        expected_names = sorted(set(weight_map.values()))
        actual_names = [path.name for path in actual]
        if actual_names != expected_names:
            raise CensusError(
                f"index/shard inventory mismatch: expected {expected_names}, found {actual_names}"
            )
        return [weights_root / name for name in expected_names], weight_map, index_path, weights_root
    single_candidates = [
        path for path in model_dir.rglob("model.safetensors") if relevant(path)
    ]
    if len(single_candidates) != 1:
        raise CensusError("unindexed source must contain exactly one model.safetensors weight root")
    single = single_candidates[0]
    weights_root = single.parent
    actual = sorted(weights_root.glob("*.safetensors"))
    if not single.is_file() or actual != [single]:
        raise CensusError("unindexed source must contain exactly model.safetensors")
    return [single], None, None, weights_root


def _resource_snapshot(model_dir: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(model_dir)

    def command(argv: list[str]) -> dict[str, Any]:
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=5, check=False)
            return {"ok": result.returncode == 0, "returncode": result.returncode,
                    "output": (result.stdout + result.stderr).strip()[-1000:]}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "sampled_at": _now(),
        "disk_free_bytes": usage.free,
        "disk_total_bytes": usage.total,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_pressure": command(["memory_pressure", "-Q"]),
        "swap": command(["sysctl", "vm.swapusage"]),
        "power": command(["pmset", "-g", "batt"]),
        "thermal": command(["pmset", "-g", "therm"]),
    }


def _load_checkpoint(path: Path, label: str, hf_id: str, model_dir: Path) -> dict[str, Any]:
    try:
        value = _read_json(path)
    except FileNotFoundError:
        value = {}
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "label": label,
        "hf_id": hf_id,
        "model_dir": str(model_dir),
    }
    if not value:
        return {**expected, "created_at": _now(), "updated_at": _now(), "shards": {}}
    if not isinstance(value, dict) or any(value.get(key) != item for key, item in expected.items()):
        raise CensusError("checkpoint identity does not match requested census")
    if not isinstance(value.get("shards"), dict):
        raise CensusError("checkpoint shard map is invalid")
    return value


def _checkpoint_shard_reusable(row: Any, shard: Path, model_dir: Path,
                               ordinal: int) -> bool:
    """Accept a journal row only as resumable work, never as final authority."""
    keys = {
        "bytes", "device", "inode", "mtime_ns", "ctime_ns", "header_bytes",
        "header_sha256", "data_bytes", "tensor_count", "tensors",
        "file_sha256", "name", "ordinal",
    }
    if not isinstance(row, dict) or set(row) != keys:
        return False
    if row.get("name") != shard.relative_to(model_dir).as_posix() \
            or row.get("ordinal") != ordinal:
        return False
    if shard.is_symlink() or not shard.is_file():
        return False
    try:
        identity = _stat_identity(shard)
    except OSError:
        return False
    if any(row.get(key) != identity[key]
           for key in ("bytes", "device", "inode", "mtime_ns", "ctime_ns")):
        return False
    if not _is_sha(row.get("header_sha256")) or not _is_sha(row.get("file_sha256")):
        return False
    tensors = row.get("tensors")
    if not isinstance(tensors, list) or len(tensors) != row.get("tensor_count") or not tensors:
        return False
    names: set[str] = set()
    for tensor in tensors:
        if not isinstance(tensor, dict) or set(tensor) != {
                "name", "dtype", "shape", "elements", "data_offsets",
                "stored_bytes", "ownership_class"}:
            return False
        name = tensor.get("name")
        if not isinstance(name, str) or not name or name in names:
            return False
        names.add(name)
        if tensor.get("dtype") not in DTYPE_BITS or not isinstance(tensor.get("shape"), list):
            return False
        try:
            elements = _checked_product(tensor["shape"])
        except CensusError:
            return False
        offsets = tensor.get("data_offsets")
        if (tensor.get("elements") != elements or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in offsets)
                or offsets[1] - offsets[0] != tensor.get("stored_bytes")):
            return False
    return True


def _auxiliary_manifest(
    model_dir: Path, weights_root: Path,
) -> tuple[list[dict[str, Any]], str | None, dict[str, Any], Path]:
    """Bind every non-weight file below the source root, including custom code/templates."""
    files: list[dict[str, Any]] = []
    for path in sorted(model_dir.rglob("*")):
        if ".cache" in path.relative_to(model_dir).parts:
            continue
        if path.is_symlink():
            raise CensusError(f"source inventory contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(model_dir).as_posix()
        if relative.endswith(".safetensors"):
            continue
        before = _stat_identity(path)
        digest = _sha256_file(path)
        after = _stat_identity(path)
        if before != after:
            raise CensusError(f"source auxiliary file changed while hashing: {path}")
        files.append({"name": relative, "bytes": before["bytes"], "sha256": digest})
    config_candidates = [path for path in (weights_root / "config.json", model_dir / "config.json")
                         if path.is_file()]
    config_candidates = list(dict.fromkeys(config_candidates))
    if not config_candidates:
        config_candidates = [path for path in model_dir.rglob("config.json")
                             if ".cache" not in path.relative_to(model_dir).parts]
    if len(config_candidates) != 1:
        raise CensusError("exactly one applicable config.json is required")
    config_path = config_candidates[0]
    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise CensusError("config.json is not an object")
    tokenizer_candidates = [path for path in (weights_root / "tokenizer_config.json",
                                               model_dir / "tokenizer_config.json")
                            if path.is_file()]
    tokenizer_candidates = list(dict.fromkeys(tokenizer_candidates))
    tokenizer_config = tokenizer_candidates[0] if len(tokenizer_candidates) == 1 else None
    chat_hash: str | None = None
    if tokenizer_config is not None:
        token_doc = _read_json(tokenizer_config)
        template = token_doc.get("chat_template") if isinstance(token_doc, dict) else None
        if template is not None:
            chat_hash = hashlib.sha256(str(template).encode("utf-8")).hexdigest()
    return files, chat_hash, config, config_path


def _download_binding(marker: Path | None, label: str, hf_id: str,
                      model_dir: Path) -> dict[str, Any] | None:
    if marker is None:
        return None
    document = _read_json(marker)
    if not isinstance(document, dict):
        raise CensusError("download marker is not an object")
    if document.get("schema") != "hawking.frontier_download_verified.v1":
        raise CensusError("download marker schema mismatch")
    if document.get("status") != "verified" or document.get("verified_complete") is not True:
        raise CensusError("download marker is not verified-complete")
    if document.get("hf_download_returncode") != 0:
        raise CensusError("download marker records a failed download")
    verification = document.get("verification")
    if (not isinstance(verification, dict) or verification.get("requested") is not True
            or verification.get("returncode") != 0):
        raise CensusError("download marker lacks successful requested verification")
    if document.get("label") != label or document.get("hf_id") != hf_id:
        raise CensusError("download marker model identity mismatch")
    recorded = Path(str(document.get("local_dir", "")))
    if not recorded.is_absolute():
        recorded = (Path.cwd() / recorded)
    if recorded.resolve() != model_dir:
        raise CensusError("download marker local_dir mismatch")
    expected_patterns = ["original/*"] if label == "120B" else []
    if document.get("include_patterns") != expected_patterns:
        raise CensusError("download marker include-pattern policy mismatch")
    return {"path": str(marker.resolve()), "sha256": _sha256_file(marker),
            "schema": document.get("schema")}


def run_census(*, label: str, hf_id: str, model_dir: Path, output: Path,
               checkpoint: Path, expected_download_marker: Path | None = None) -> dict[str, Any]:
    model_dir = model_dir.resolve(strict=True)
    if not model_dir.is_dir():
        raise CensusError("model_dir is not a directory")
    output = output.resolve(strict=False)
    checkpoint = checkpoint.resolve(strict=False)
    protected = {output, checkpoint}
    if expected_download_marker is not None:
        protected.add(expected_download_marker.resolve(strict=False))
    if len(protected) != (3 if expected_download_marker is not None else 2):
        raise CensusError("output, checkpoint, and marker paths must be distinct")
    for path in (output, checkpoint):
        try:
            path.relative_to(model_dir)
        except ValueError:
            pass
        else:
            raise CensusError("output and checkpoint must be outside the source tree")
    before = _resource_snapshot(model_dir)
    shards, weight_map, index_path, weights_root = _source_layout(model_dir)
    download = _download_binding(expected_download_marker, label, hf_id, model_dir)
    state = _load_checkpoint(checkpoint, label, hf_id, model_dir)
    expected_shard_names = {shard.relative_to(model_dir).as_posix() for shard in shards}
    state["shards"] = {name: row for name, row in state["shards"].items()
                       if name in expected_shard_names}
    seen: dict[str, str] = {}
    completed: list[dict[str, Any]] = []
    reused_checkpoint_work = False
    for ordinal, shard in enumerate(shards):
        if _STOP:
            raise StopRequested("stop requested at shard boundary")
        relative_name = shard.relative_to(model_dir).as_posix()
        journal_row = state["shards"].get(relative_name)
        if _checkpoint_shard_reusable(journal_row, shard, model_dir, ordinal):
            row = journal_row
            reused_checkpoint_work = True
        else:
            inspected = _read_header(shard)
            row = {**inspected, "name": relative_name, "ordinal": ordinal}
        state["shards"][relative_name] = row
        state["updated_at"] = _now()
        state["completed_shards"] = ordinal + 1
        state["total_shards"] = len(shards)
        _atomic_json(checkpoint, state)
        for tensor in row["tensors"]:
            name = tensor["name"]
            if name in seen:
                raise CensusError(f"duplicate tensor {name} in {shard.name} and {seen[name]}")
            seen[name] = shard.name
        completed.append(row)
    if reused_checkpoint_work:
        # Journal rows save completed-shard work after an interruption, but never
        # become publication authority. Re-open and re-hash the full source once
        # at the final boundary before constructing any aggregate or report.
        seen = {}
        completed = []
        for ordinal, shard in enumerate(shards):
            if _STOP:
                raise StopRequested("stop requested during final source revalidation")
            relative_name = shard.relative_to(model_dir).as_posix()
            row = {**_read_header(shard), "name": relative_name, "ordinal": ordinal}
            for tensor in row["tensors"]:
                name = tensor["name"]
                if name in seen:
                    raise CensusError(
                        f"duplicate tensor {name} in {shard.name} and {seen[name]}"
                    )
                seen[name] = shard.name
            completed.append(row)
            state["shards"][relative_name] = row
            state["updated_at"] = _now()
            state["completed_shards"] = ordinal + 1
            state["total_shards"] = len(shards)
            _atomic_json(checkpoint, state)
    if weight_map is not None:
        if set(weight_map) != set(seen):
            missing = sorted(set(weight_map) - set(seen))[:5]
            extra = sorted(set(seen) - set(weight_map))[:5]
            raise CensusError(f"index/tensor coverage mismatch; missing={missing}, extra={extra}")
        mismatched = [name for name, shard in weight_map.items() if seen.get(name) != shard]
        if mismatched:
            raise CensusError(f"index maps tensors to wrong shards: {mismatched[:5]}")
        index_document = _read_json(index_path) if index_path else {}
        expected_total = index_document["metadata"]["total_size"]
        observed_total = sum(row["data_bytes"] for row in completed)
        if expected_total != observed_total:
            raise CensusError(
                f"index metadata.total_size mismatch: expected {expected_total}, observed {observed_total}"
            )
    auxiliary, chat_hash, config, config_path = _auxiliary_manifest(model_dir, weights_root)
    dtype_elements: dict[str, int] = {}
    ownership: dict[str, dict[str, int]] = {}
    stored_elements = 0
    tensor_count = 0
    for shard in completed:
        for tensor in shard["tensors"]:
            elements = tensor["elements"]
            stored_elements += elements
            tensor_count += 1
            dtype_elements[tensor["dtype"]] = dtype_elements.get(tensor["dtype"], 0) + elements
            group = ownership.setdefault(tensor["ownership_class"], {"tensors": 0, "elements": 0})
            group["tensors"] += 1
            group["elements"] += elements
    compact_shards = []
    for shard in completed:
        compact_shards.append({key: shard[key] for key in (
            "name", "ordinal", "bytes", "device", "inode", "mtime_ns", "ctime_ns",
            "header_bytes", "header_sha256", "data_bytes", "tensor_count", "file_sha256",
        )})
    source_payload = {
        "hf_id": hf_id,
        "model_dir": str(model_dir),
        "weights_root": weights_root.relative_to(model_dir).as_posix() or ".",
        "inventory_exclusions": [".cache/**"],
        "shards": compact_shards,
        "auxiliary_files": auxiliary,
        "download_binding": download,
    }
    report = {
        "schema": SCHEMA,
        "report_version": REPORT_VERSION,
        "label": label,
        "hf_id": hf_id,
        "model_dir": str(model_dir),
        "status": "complete",
        "completed_at": _now(),
        "producer": {
            "path": "tools/condense/doctor_v5_census.py",
            "sha256": _sha256_file(Path(__file__).resolve()),
            "report_version": REPORT_VERSION,
        },
        "evidence_class": "bootstrap_source_census_not_quality_evidence",
        "quality_observation": None,
        "diagnosis": "undetermined",
        "launch_permitted": False,
        "dominance_proven": False,
        "source": {
            **source_payload,
            "source_manifest_sha256": _hash_value(source_payload),
            "index_path": str(index_path.resolve()) if index_path else None,
            "index_sha256": _sha256_file(index_path) if index_path else None,
            "total_shard_bytes": sum(row["bytes"] for row in compact_shards),
            "shard_count": len(compact_shards),
        },
        "tensor_census": {
            "tensor_count": tensor_count,
            "stored_tensor_element_count": stored_elements,
            "dtype_element_counts": dict(sorted(dtype_elements.items())),
            "ownership_class_counts": dict(sorted(ownership.items())),
            "classification_complete": False,
            "classification_note": CLASSIFICATION_NOTE,
            "tied_or_shared_policy": {
                "config_tie_word_embeddings": config.get("tie_word_embeddings"),
                "alias_storage_proven": False,
            },
            "logical_parameter_count": None,
            "exact_parameter_count_authoritative": False,
            "authority_blocker": AUTHORITY_BLOCKER,
        },
        "identity": {
            "config_sha256": _sha256_file(config_path),
            "tokenizer_manifest_sha256": _hash_value(
                [row for row in auxiliary if "tokenizer" in row["name"] or row["name"] in {
                    "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
                    "spiece.model", "sentencepiece.bpe.model",
                }]
            ),
            "chat_template_sha256": chat_hash,
            "dtype_manifest_sha256": next(
                (row["sha256"] for row in auxiliary if row["name"].endswith("dtypes.json")), None
            ),
            "download_binding": download,
        },
        "architecture_bootstrap": {
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "hidden_size": config.get("hidden_size"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "num_local_experts": config.get("num_local_experts", config.get("num_experts")),
            "num_experts_per_tok": config.get("num_experts_per_tok", config.get("experts_per_token")),
            "packed_dtype_manifest_bound": any(
                row["name"].endswith("dtypes.json") for row in auxiliary
            ),
            "active_parameter_count": None,
            "review_required": True,
        },
        "resource_measurements": {"before": before, "after": _resource_snapshot(model_dir)},
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "schema": CHECKPOINT_SCHEMA,
            "resume_unit": "durable_shard_journal_with_final_full_source_revalidation",
            "bit_exact_training_resume_claimed": False,
            "durable_fsync_required": True,
        },
        "parameter_manifest_receipt": None,
        "promotion_blockers": list(PROMOTION_BLOCKERS),
    }
    report = _stamp(report, "report_sha256")
    errors = validate_report(report)
    if errors:
        raise CensusError("internal report validation failed: " + "; ".join(errors))
    _atomic_json(output, report)
    state["status"] = "complete"
    state["report_path"] = str(output.resolve())
    state["report_sha256"] = report["report_sha256"]
    state["updated_at"] = _now()
    _atomic_json(checkpoint, state)
    return report


def validate_report(document: Any, *, verify_files: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["report is not an object"]
    required = {
        "schema", "report_version", "label", "hf_id", "model_dir", "status",
        "completed_at", "producer", "evidence_class", "quality_observation", "diagnosis",
        "launch_permitted", "dominance_proven", "source", "tensor_census", "identity",
        "architecture_bootstrap", "resource_measurements", "checkpoint",
        "parameter_manifest_receipt", "promotion_blockers", "report_sha256",
    }
    if set(document) != required:
        errors.append("report keys are not exact")
    if document.get("schema") != SCHEMA or document.get("status") != "complete":
        errors.append("schema/status mismatch")
    if document.get("evidence_class") != "bootstrap_source_census_not_quality_evidence":
        errors.append("evidence class is not bootstrap-only")
    if document.get("quality_observation") is not None or document.get("diagnosis") != "undetermined":
        errors.append("census must not contain a quality observation or diagnosis")
    if document.get("launch_permitted") is not False or document.get("dominance_proven") is not False:
        errors.append("census must not authorize launch or dominance")
    if document.get("parameter_manifest_receipt") is not None:
        errors.append("census may not self-issue parameter authority")
    def exact(value: Any, keys: set[str], path: str) -> bool:
        if not isinstance(value, dict):
            errors.append(f"{path} is not an object")
            return False
        if set(value) != keys:
            errors.append(f"{path} keys are not exact")
            return False
        return True

    census = document.get("tensor_census")
    producer = document.get("producer")
    if exact(producer, {"path", "sha256", "report_version"}, "producer"):
        if producer.get("path") != "tools/condense/doctor_v5_census.py" \
                or producer.get("report_version") != REPORT_VERSION \
                or not _is_sha(producer.get("sha256")):
            errors.append("producer identity is invalid")
    if exact(census, {
        "tensor_count", "stored_tensor_element_count", "dtype_element_counts",
        "ownership_class_counts", "classification_complete", "classification_note",
        "tied_or_shared_policy", "logical_parameter_count",
        "exact_parameter_count_authoritative", "authority_blocker",
    }, "tensor_census"):
        if census.get("logical_parameter_count") is not None:
            errors.append("bootstrap census may not assert logical_parameter_count")
        if census.get("exact_parameter_count_authoritative") is not False:
            errors.append("bootstrap census may not assert parameter authority")
        if census.get("classification_complete") is not False \
                or census.get("classification_note") != CLASSIFICATION_NOTE \
                or census.get("authority_blocker") != AUTHORITY_BLOCKER:
            errors.append("tensor census fail-closed literals are invalid")
        for key in ("tensor_count", "stored_tensor_element_count"):
            if isinstance(census.get(key), bool) or not isinstance(census.get(key), int) \
                    or census.get(key) < 0:
                errors.append(f"invalid {key}")
        if not exact(census.get("tied_or_shared_policy"), {
                "config_tie_word_embeddings", "alias_storage_proven"},
                "tensor_census.tied_or_shared_policy"):
            pass
        elif census["tied_or_shared_policy"].get("alias_storage_proven") is not False:
            errors.append("bootstrap census may not assert alias-storage proof")
        dtypes = census.get("dtype_element_counts")
        if not isinstance(dtypes, dict) or any(
                key not in DTYPE_BITS or isinstance(value, bool) or not isinstance(value, int)
                or value < 0 for key, value in (dtypes.items() if isinstance(dtypes, dict) else [])):
            errors.append("dtype_element_counts is invalid")
        elif sum(dtypes.values()) != census.get("stored_tensor_element_count"):
            errors.append("dtype element aggregate mismatch")
        owners = census.get("ownership_class_counts")
        if not isinstance(owners, dict):
            errors.append("ownership_class_counts is invalid")
        else:
            owner_tensors = 0
            owner_elements = 0
            for owner, row in owners.items():
                if not isinstance(owner, str) or not exact(row, {"tensors", "elements"},
                                                           f"ownership.{owner}"):
                    continue
                if any(isinstance(row[key], bool) or not isinstance(row[key], int)
                       or row[key] < 0 for key in ("tensors", "elements")):
                    errors.append(f"ownership.{owner} counts are invalid")
                else:
                    owner_tensors += row["tensors"]
                    owner_elements += row["elements"]
            if owner_tensors != census.get("tensor_count") \
                    or owner_elements != census.get("stored_tensor_element_count"):
                errors.append("ownership aggregate mismatch")
    source = document.get("source")
    source_ok = exact(source, {
        "hf_id", "model_dir", "weights_root", "inventory_exclusions", "shards",
        "auxiliary_files", "download_binding",
        "source_manifest_sha256", "index_path", "index_sha256", "total_shard_bytes",
        "shard_count",
    }, "source")
    if not source_ok or not _is_sha(source.get("source_manifest_sha256")):
        errors.append("source manifest is invalid")
    elif _hash_value({key: source[key] for key in (
            "hf_id", "model_dir", "weights_root", "inventory_exclusions", "shards",
            "auxiliary_files", "download_binding")}) \
            != source["source_manifest_sha256"]:
        errors.append("source manifest hash mismatch")
    if source_ok:
        if source.get("hf_id") != document.get("hf_id") \
                or source.get("model_dir") != document.get("model_dir"):
            errors.append("top-level/source identity mismatch")
        if not isinstance(source.get("weights_root"), str) or not source.get("weights_root"):
            errors.append("source weights_root is invalid")
        if source.get("inventory_exclusions") != [".cache/**"]:
            errors.append("source inventory exclusion policy is invalid")
        shards = source.get("shards")
        if not isinstance(shards, list) or not shards:
            errors.append("source.shards is empty or invalid")
        else:
            for position, row in enumerate(shards):
                if exact(row, {
                    "name", "ordinal", "bytes", "device", "inode", "mtime_ns", "ctime_ns",
                    "header_bytes", "header_sha256", "data_bytes", "tensor_count", "file_sha256",
                }, f"source.shards[{position}]"):
                    if row.get("ordinal") != position:
                        errors.append("source shard ordinals are not canonical")
                    if not _is_sha(row.get("header_sha256")) or not _is_sha(row.get("file_sha256")):
                        errors.append(f"source.shards[{position}] has an invalid hash")
            if source.get("shard_count") != len(shards):
                errors.append("source shard_count mismatch")
            byte_values = [row.get("bytes") for row in shards if isinstance(row, dict)]
            if (len(byte_values) != len(shards)
                    or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                           for value in byte_values)
                    or source.get("total_shard_bytes") != sum(byte_values)):
                errors.append("source total_shard_bytes mismatch")
        auxiliary = source.get("auxiliary_files")
        if not isinstance(auxiliary, list):
            errors.append("source.auxiliary_files is invalid")
        else:
            names: set[str] = set()
            for position, row in enumerate(auxiliary):
                if exact(row, {"name", "bytes", "sha256"},
                         f"source.auxiliary_files[{position}]"):
                    name = row.get("name")
                    if not isinstance(name, str) or not name:
                        errors.append("invalid auxiliary file name")
                        continue
                    if name in names:
                        errors.append("duplicate auxiliary file")
                    names.add(name)
                    if not _is_sha(row.get("sha256")):
                        errors.append("invalid auxiliary file hash")
        binding = source.get("download_binding")
        if binding is not None:
            if exact(binding, {"path", "sha256", "schema"}, "source.download_binding"):
                if binding.get("schema") != "hawking.frontier_download_verified.v1" \
                        or not _is_sha(binding.get("sha256")):
                    errors.append("download binding semantics are invalid")
        if (source.get("index_path") is None) != (source.get("index_sha256") is None):
            errors.append("source index path/hash presence mismatch")
        if source.get("index_sha256") is not None and not _is_sha(source.get("index_sha256")):
            errors.append("source index hash is invalid")
    identity = document.get("identity")
    if exact(identity, {"config_sha256", "tokenizer_manifest_sha256",
                        "chat_template_sha256", "dtype_manifest_sha256",
                        "download_binding"}, "identity"):
        if not _is_sha(identity.get("config_sha256")) \
                or not _is_sha(identity.get("tokenizer_manifest_sha256")):
            errors.append("identity hashes are invalid")
        if identity.get("chat_template_sha256") is not None \
                and not _is_sha(identity.get("chat_template_sha256")):
            errors.append("chat template hash is invalid")
        if identity.get("dtype_manifest_sha256") is not None \
                and not _is_sha(identity.get("dtype_manifest_sha256")):
            errors.append("dtype manifest hash is invalid")
        source_binding = source.get("download_binding") if isinstance(source, dict) else None
        if identity.get("download_binding") != source_binding:
            errors.append("download binding is inconsistent")
    architecture = document.get("architecture_bootstrap")
    if exact(architecture, {
        "model_type", "architectures", "num_hidden_layers", "hidden_size",
        "num_attention_heads", "num_key_value_heads", "num_local_experts",
        "num_experts_per_tok", "packed_dtype_manifest_bound", "active_parameter_count",
        "review_required",
    }, "architecture_bootstrap"):
        if architecture.get("active_parameter_count") is not None \
                or architecture.get("review_required") is not True:
            errors.append("architecture bootstrap must remain unreviewed")
    resources = document.get("resource_measurements")
    if exact(resources, {"before", "after"}, "resource_measurements"):
        for phase in ("before", "after"):
            snapshot = resources[phase]
            if exact(snapshot, {"sampled_at", "disk_free_bytes", "disk_total_bytes",
                                "max_rss_bytes", "memory_pressure", "swap", "power", "thermal"},
                     f"resource_measurements.{phase}"):
                for probe_name in ("memory_pressure", "swap", "power", "thermal"):
                    probe = snapshot[probe_name]
                    if isinstance(probe, dict) and probe.get("ok") is True:
                        exact(probe, {"ok", "returncode", "output"},
                              f"resource_measurements.{phase}.{probe_name}")
                    else:
                        exact(probe, {"ok", "error"},
                              f"resource_measurements.{phase}.{probe_name}")
    checkpoint = document.get("checkpoint")
    if exact(checkpoint, {
        "path", "schema", "resume_unit", "bit_exact_training_resume_claimed",
        "durable_fsync_required",
    }, "checkpoint"):
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA \
                or checkpoint.get("resume_unit") != \
                "durable_shard_journal_with_final_full_source_revalidation" \
                or checkpoint.get("bit_exact_training_resume_claimed") is not False \
                or checkpoint.get("durable_fsync_required") is not True:
            errors.append("checkpoint semantics are invalid")
    blockers = document.get("promotion_blockers")
    if blockers != list(PROMOTION_BLOCKERS):
        errors.append("promotion_blockers is invalid")
    expected_hash = _hash_value(_without_hash(document, "report_sha256"))
    if document.get("report_sha256") != expected_hash:
        errors.append("report hash mismatch")
    if verify_files and source_ok:
        try:
            root = Path(source["model_dir"]).resolve(strict=True)
            if root != Path(str(document.get("model_dir", ""))).resolve(strict=True):
                errors.append("live source root identity mismatch")
            expected_names = {
                row["name"] for row in source["shards"] if isinstance(row, dict)
                and isinstance(row.get("name"), str)
            } | {
                row["name"] for row in source["auxiliary_files"] if isinstance(row, dict)
                and isinstance(row.get("name"), str)
            }
            actual_names: set[str] = set()
            for path in root.rglob("*"):
                relative = path.relative_to(root)
                if ".cache" in relative.parts:
                    continue
                if path.is_symlink():
                    errors.append(f"live source contains symlink: {relative.as_posix()}")
                elif path.is_file():
                    actual_names.add(relative.as_posix())
            if actual_names != expected_names:
                errors.append("live source inventory differs from report")
            for row in source["shards"]:
                path = root / row["name"]
                if (not path.is_file() or path.is_symlink() or path.stat().st_size != row["bytes"]
                        or _sha256_file(path) != row["file_sha256"]):
                    errors.append(f"live source shard mismatch: {row['name']}")
            for row in source["auxiliary_files"]:
                path = root / row["name"]
                if (not path.is_file() or path.is_symlink() or path.stat().st_size != row["bytes"]
                        or _sha256_file(path) != row["sha256"]):
                    errors.append(f"live auxiliary mismatch: {row['name']}")
            binding = source.get("download_binding")
            if isinstance(binding, dict):
                marker = Path(binding["path"])
                if not marker.is_file() or _sha256_file(marker) != binding["sha256"]:
                    errors.append("live download binding mismatch")
        except Exception as exc:
            errors.append(f"live source verification failed: {type(exc).__name__}: {exc}")
    return errors


def _write_safetensor(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    cursor = 0
    header: dict[str, Any] = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [cursor, cursor + len(data)]}
        payload.extend(data)
        cursor += len(data)
    raw = _canonical(header)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + payload)


def selftest() -> int:
    global _STOP
    _STOP = False
    with tempfile.TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"model_type": "synthetic", "tie_word_embeddings": False}),
            encoding="utf-8",
        )
        (model / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{{ test }}"}), encoding="utf-8"
        )
        _write_safetensor(model / "model.safetensors", [
            ("model.embed_tokens.weight", "F16", [2, 2], b"\0" * 8),
            ("lm_head.weight", "F16", [2, 2], b"\0" * 8),
        ])
        output, checkpoint = root / "report.json", root / "checkpoint.json"
        report = run_census(label="tiny", hf_id="local/tiny", model_dir=model,
                            output=output, checkpoint=checkpoint)
        assert not validate_report(report)
        assert report["tensor_census"]["stored_tensor_element_count"] == 8
        again = run_census(label="tiny", hf_id="local/tiny", model_dir=model,
                           output=output, checkpoint=checkpoint)
        assert again["source"]["source_manifest_sha256"] == report["source"]["source_manifest_sha256"]
        journal = _read_json(checkpoint)
        assert _checkpoint_shard_reusable(
            journal["shards"]["model.safetensors"],
            model.resolve() / "model.safetensors", model.resolve(), 0,
        )
        damaged = json.loads(json.dumps(report))
        damaged["launch_permitted"] = True
        assert validate_report(damaged)
        for path, key, value in (
            (("source",), "execution_authorized", True),
            (("identity",), "parameter_manifest_receipt", {"forged": True}),
            (("tensor_census",), "quality_evidence", "pass"),
        ):
            damaged = json.loads(json.dumps(report))
            target = damaged
            for part in path:
                target = target[part]
            target[key] = value
            damaged = _stamp(damaged, "report_sha256")
            assert validate_report(damaged), (path, key)
        forged_checkpoint = _read_json(checkpoint)
        forged_checkpoint["shards"]["model.safetensors"]["file_sha256"] = "f" * 64
        _atomic_json(checkpoint, forged_checkpoint)
        repaired = run_census(label="tiny", hf_id="local/tiny", model_dir=model,
                              output=output, checkpoint=checkpoint)
        assert repaired["source"]["shards"][0]["file_sha256"] != "f" * 64
        bad = root / "bad"
        bad.mkdir()
        (bad / "config.json").write_text("{}", encoding="utf-8")
        raw_header = _canonical({"a": {"dtype": "F16", "shape": [1], "data_offsets": [1, 3]}})
        (bad / "model.safetensors").write_bytes(
            len(raw_header).to_bytes(8, "little") + raw_header + b"\0" * 3
        )
        try:
            run_census(label="bad", hf_id="local/bad", model_dir=bad,
                       output=root / "bad.json", checkpoint=root / "bad.cp.json")
        except CensusError:
            pass
        else:
            raise AssertionError("malformed safetensors was accepted")
    print("doctor_v5_census.py selftest OK")
    return 0


def _request_stop(_signal: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--label", required=True)
    run.add_argument("--hf-id", required=True)
    run.add_argument("--model-dir", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--checkpoint", required=True, type=Path)
    run.add_argument("--expected-download-marker", type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("path", type=Path)
    commands.add_parser("selftest")
    args = parser.parse_args()
    if args.command == "selftest":
        return selftest()
    if args.command == "validate":
        errors = validate_report(_read_json(args.path))
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print(f"valid {SCHEMA}: {args.path}")
        return 0
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        report = run_census(
            label=args.label,
            hf_id=args.hf_id,
            model_dir=args.model_dir,
            output=args.output,
            checkpoint=args.checkpoint,
            expected_download_marker=args.expected_download_marker,
        )
    except StopRequested as exc:
        print(f"[doctor-v5-census] checkpointed stop: {exc}", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[doctor-v5-census] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"label": report["label"], "status": report["status"],
                      "report_sha256": report["report_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
