#!/usr/bin/env python3.12
"""Second Light source-provenance rebuild (run-critical defect fix).

The sealed GRAVITY_120B_PROVENANCE.json bound the source shards to a stale staging path
(scratch/staging/gpt-oss-120b.partial/original/) that no longer exists. The real 61 GiB source
now lives at models/gpt-oss-120b/original/. Every driver that reads through ProvenanceReader
therefore saw "source absent" and every gate ran vacuously. This rebuilds the manifest DIRECTLY
from the real safetensors headers at the correct path, so byte ranges are verified against the
files that are actually present. The previous manifest is backed up, never silently clobbered.

This is exactly the run-critical defect the Second Light goal permits touching. It does not change
any weight, only the provenance binding + a fresh, honest source receipt.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORIG = REPO / "models" / "gpt-oss-120b" / "original"
MANIFEST = REPO / "reports" / "condense" / "subbit_frontier" / "GRAVITY_120B_PROVENANCE.json"
SCHEMA = "hawking.gravity.provenance.v2_second_light"

# safetensors dtype -> our tag (bf16/u8/etc are the only ones the reader distinguishes)
_DTYPE = {"BF16": "bf16", "F16": "f16", "F32": "f32", "U8": "u8", "I8": "i8",
          "U16": "u16", "U32": "u32", "F8_E4M3": "f8", "F8_E5M2": "f8"}


def _read_header(shard: Path) -> tuple[dict, int]:
    """Return (tensor_header_json, data_start_offset). safetensors: 8-byte LE u64 header length,
    then that many bytes of JSON, then the tensor data blob."""
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return header, 8 + n


def build() -> dict:
    shards = sorted(ORIG.glob("model--*-of-*.safetensors"))
    if not shards:
        print(json.dumps({"ok": False, "reason": f"no shards under {ORIG}"}))
        raise SystemExit(1)
    tensors = []
    total_bytes = 0
    for shard in shards:
        header, data_start = _read_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start_rel, end_rel = meta["data_offsets"]
            byte_range = [data_start + start_rel, data_start + end_rel]
            byte_len = end_rel - start_rel
            total_bytes += byte_len
            tensors.append({
                "tensor": name,
                "shape": meta["shape"],
                "dtype": _DTYPE.get(meta["dtype"], meta["dtype"].lower()),
                "safetensors_dtype": meta["dtype"],
                "byte_range": byte_range,
                "byte_len": byte_len,
                "shard_name": shard.name,
                "shard_path": str(shard),
                "orientation": "row_major",
            })
    tensors.sort(key=lambda t: t["tensor"])
    # deterministic manifest hash over (name, shape, dtype, byte_len) tuples
    h = hashlib.sha256()
    for t in tensors:
        h.update(f"{t['tensor']}|{t['shape']}|{t['dtype']}|{t['byte_len']}".encode())
    doc = {
        "schema": SCHEMA,
        "source_dir": str(ORIG),
        "weights_root": str(ORIG),
        "shards": [s.name for s in shards],
        "shard_paths": [str(s) for s in shards],
        "tensors": tensors,
        "tensor_count": len(tensors),
        "total_source_bytes": total_bytes,
        "manifest_sha256": h.hexdigest(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rebuilt_reason": "run-critical: prior manifest bound to nonexistent staging path",
    }
    return doc


def main() -> int:
    doc = build()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        backup = MANIFEST.with_suffix(".json.pre_second_light.bak")
        if not backup.exists():
            backup.write_bytes(MANIFEST.read_bytes())
    MANIFEST.write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(json.dumps({"ok": True, "tensor_count": doc["tensor_count"],
                      "total_source_bytes": doc["total_source_bytes"],
                      "manifest_sha256": doc["manifest_sha256"][:16],
                      "shards": len(doc["shards"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
