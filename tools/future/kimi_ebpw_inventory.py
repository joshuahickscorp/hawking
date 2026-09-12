#!/usr/bin/env python3
"""Inventory KIMI's complete model-package EBPW without loading weights.

This is a read-only Gravity.Discover/Gravity.Reduce instrument.  It reads the
index and safetensors headers, bills tensor payload, container/header overhead,
execution metadata, and model-specific loader code, then delegates the final
arithmetic to ``complete_ebpw``.  It does not decode a tensor, materialize a
second model, edit KIMI_BASE, or claim that raw BF16 is a reduced/direct Noetic
representation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from math import prod
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.future.complete_ebpw import (
    STREAM_BROADCAST_AUX,
    STREAM_WEIGHT_CODES,
    cost,
    stream_rates,
)

DEFAULT_SPEC = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "moonshotai--Kimi-VL-A3B-Instruct@398eede0903c"
)
SCHEMA = "hawking.future.kimi_ebpw_inventory.v1"

# These files are needed to resolve and execute the published KIMI package.
# README/.gitattributes remain provenance material but are not runtime bytes.
REQUIRED_METADATA = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tiktoken.model",
    "model.safetensors.index.json",
)
MODEL_CODE_SUFFIXES = (".py",)


class InventoryRefused(RuntimeError):
    """The package cannot be accounted for without guessing."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _need_file(spec: Path, name: str) -> Path:
    path = spec / name
    if not path.is_file():
        raise InventoryRefused(f"required execution file is missing: {path}")
    return path


def _shard_header(path: Path) -> tuple[int, int, dict[str, Any]]:
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise InventoryRefused(f"{path} has no safetensors header length")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 0 or header_len > 128 * 1024 * 1024:
            raise InventoryRefused(f"{path} has unreasonable header length {header_len}")
        raw_header = fh.read(header_len)
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise InventoryRefused(f"{path} header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise InventoryRefused(f"{path} header is not an object")
    payload = 0
    params = 0
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict):
            raise InventoryRefused(f"{path}:{name} tensor entry is not an object")
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        if not isinstance(shape, list) or not all(
            isinstance(x, int) and x >= 0 for x in shape
        ):
            raise InventoryRefused(f"{path}:{name} has invalid shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(
            isinstance(x, int) and x >= 0 for x in offsets
        ) or offsets[1] < offsets[0]:
            raise InventoryRefused(f"{path}:{name} has invalid data_offsets")
        payload += int(offsets[1] - offsets[0])
        params += int(prod(shape))
    return int(header_len), int(payload), {"tensor_count": len(header) - (1 if "__metadata__" in header else 0), "params": params}


def inventory(spec: Path) -> dict[str, Any]:
    index_path = _need_file(spec, "model.safetensors.index.json")
    index = json.loads(index_path.read_text())
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise InventoryRefused(f"{index_path} has no weight_map")
    mapped_shards = sorted(set(str(x) for x in index["weight_map"].values()))
    if not mapped_shards:
        raise InventoryRefused("weight_map is empty")

    shard_rows = []
    payload_bytes = 0
    parent_params = 0
    tensor_count = 0
    indexed_names = set(str(x) for x in index["weight_map"])
    header_names: set[str] = set()
    for shard_name in mapped_shards:
        path = _need_file(spec, shard_name)
        header_len, shard_payload, stats = _shard_header(path)
        with path.open("rb") as fh:
            raw = fh.read(8)
        expected_size = 8 + header_len + shard_payload
        actual_size = path.stat().st_size
        if expected_size != actual_size:
            raise InventoryRefused(
                f"{path} size {actual_size} != safetensors framing {expected_size}"
            )
        with path.open("rb") as fh:
            fh.seek(8)
            header = json.loads(fh.read(header_len))
        names = {str(k) for k in header if k != "__metadata__"}
        header_names |= names
        shard_rows.append({
            "name": shard_name,
            "bytes": actual_size,
            "header_bytes": 8 + header_len,
            "tensor_payload_bytes": shard_payload,
            "tensor_count": stats["tensor_count"],
            "sha256": _sha256(path),
        })
        payload_bytes += shard_payload
        parent_params += stats["params"]
        tensor_count += stats["tensor_count"]
    if header_names != indexed_names:
        missing = sorted(indexed_names - header_names)[:5]
        extra = sorted(header_names - indexed_names)[:5]
        raise InventoryRefused(
            f"index/header tensor names disagree; missing={missing}, extra={extra}"
        )

    metadata_rows = []
    metadata_bytes = 0
    for name in REQUIRED_METADATA:
        path = _need_file(spec, name)
        n = path.stat().st_size
        metadata_rows.append({"name": name, "bytes": n, "sha256": _sha256(path)})
        metadata_bytes += n
    code_rows = []
    code_bytes = 0
    for path in sorted(spec.iterdir()):
        if path.is_file() and path.name.endswith(MODEL_CODE_SUFFIXES):
            n = path.stat().st_size
            code_rows.append({"name": path.name, "bytes": n, "sha256": _sha256(path)})
            code_bytes += n

    container_bytes = sum(int(row["bytes"]) for row in shard_rows)
    framing_bytes = container_bytes - payload_bytes
    stated_total_bytes = payload_bytes + framing_bytes + metadata_bytes + code_bytes
    candidate = {
        "id": "KIMI_BASE_RAW_BF16_MODEL_PACKAGE",
        "parent_params": parent_params,
        "stated_total_bytes": stated_total_bytes,
        "regions": [{
            "name": "bf16_tensor_payload",
            "bytes": payload_bytes,
            "stream_class": STREAM_WEIGHT_CODES,
        }],
        "generators": [],
        "metadata": [
            {"name": "safetensors_framing", "bytes": framing_bytes, "stream_class": STREAM_BROADCAST_AUX},
            {"name": "execution_metadata", "bytes": metadata_bytes, "stream_class": STREAM_BROADCAST_AUX},
        ],
        "tables": [],
        "residuals": [],
        "runtime_auxiliaries": [],
        "representation": [],
        "model_specific_code": [{"name": "published_loader_code", "bytes": code_bytes, "stream_class": STREAM_BROADCAST_AUX}],
        "reconstructs_dense_parent": False,
        "consumes_representation_directly": True,
        "source": str(spec),
    }
    billed = cost(candidate, rates=stream_rates())
    return {
        "schema": SCHEMA,
        "status": "BASELINE_ACCOUNTED",
        "source_model": "KIMI_BASE",
        "specimen": str(spec),
        "immutable_base": True,
        "parent_params": parent_params,
        "tensor_count": tensor_count,
        "index_tensor_count": len(indexed_names),
        "weight_payload_bytes": payload_bytes,
        "safetensors_container_bytes": container_bytes,
        "safetensors_framing_bytes": framing_bytes,
        "execution_metadata_bytes": metadata_bytes,
        "model_specific_code_bytes": code_bytes,
        "complete_model_package_bytes": stated_total_bytes,
        "weight_payload_bpw": payload_bytes * 8.0 / parent_params,
        "complete_ebpw": billed["complete_ebpw"],
        "stored_bpw": billed["stored_bpw"],
        "billed_ms": billed["billed_ms"],
        "complete_accounting": billed,
        "shards": shard_rows,
        "metadata": metadata_rows,
        "model_specific_code": code_rows,
        "direct_execution": {
            "raw_bf16_native_loader": "AVAILABLE",
            "target_reduced_noetic_representation": "NOT_BUILT",
            "dense_parent_dependency": False,
            "note": "Raw BF16 is directly consumed by the native loader; this is not evidence of a reduced Noetic representation.",
        },
        "next_discriminator": {
            "hypothesis": "W ~= G(z) + Q(R) + S can reduce both KIMI MLP and attention/DeltaNet while retaining direct execution.",
            "first_organ": "representative routed-expert down_proj plus shared expert down_proj",
            "cheap_measurement": "header-derived payload/overhead break-even, then small-org capability/perplexity screen",
            "falsifier": "complete accounting cannot reach <=1.0 EBPW or direct decoder requires dense rematerialization",
        },
        "claim_boundary": "Static read-only package inventory. No reduced candidate, capability preservation, direct Noetic execution, TPS, or promotion claim is made. KIMI_BASE was not modified.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.spec)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["source_script_sha256"] = _sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "parent_params": result["parent_params"],
        "complete_ebpw": result["complete_ebpw"],
        "complete_model_package_bytes": result["complete_model_package_bytes"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
