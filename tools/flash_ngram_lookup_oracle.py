#!/usr/bin/env python3
"""Bounded native-lookup preparation oracle for Flash's 128-shard n-gram bank.

This reads only selected rows from real safetensors shards.  It measures the
source row, packed Q4/Q3 row size, decode parity, and lookup cost; it does not
claim a Metal kernel, complete-token EBPW/TPS, or capability preservation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np


def read_row(path: Path, name: str, row: int) -> tuple[np.ndarray, dict]:
    with path.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        meta = header[name]
        shape = tuple(meta["shape"])
        if len(shape) != 2 or row < 0 or row >= shape[0]:
            raise ValueError(f"invalid row {row} for {name} {shape}")
        width = shape[1] * 2
        offset = 8 + hlen + meta["data_offsets"][0] + row * width
        f.seek(offset)
        raw = f.read(width)
        if len(raw) != width:
            raise ValueError("short row read")
    # Flash's indexed n-gram source rows are BF16.
    if meta.get("dtype") != "BF16":
        raise ValueError(f"expected BF16, got {meta.get('dtype')}")
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view("<f4"), meta


def quantize(row: np.ndarray, bits: int, group: int = 32) -> tuple[bytes, np.ndarray]:
    usable = (row.size // group) * group
    x = row[:usable].reshape(-1, group)
    qmax = (1 << (bits - 1)) - 1
    scale = np.maximum(np.max(np.abs(x), axis=1) / qmax, 1e-30).astype("<f4")
    q = np.clip(np.rint(x / scale[:, None]), -qmax - 1, qmax).astype(np.int16)
    # Unsigned little-endian bit packing is the actual compact payload layout.
    codes = (q + (qmax + 1)).astype(np.uint16).ravel()
    bits_out = 0; nbits = 0; payload = bytearray()
    for code in codes:
        bits_out |= int(code) << nbits; nbits += bits
        while nbits >= 8:
            payload.append(bits_out & 0xFF); bits_out >>= 8; nbits -= 8
    if nbits: payload.append(bits_out & 0xFF)
    packed = scale.tobytes() + bytes(payload)
    # Decode from the packed stream, so parity covers the storage format itself.
    decoded_codes = []; bits_out = 0; nbits = 0
    for byte in payload:
        bits_out |= int(byte) << nbits; nbits += 8
        while nbits >= bits:
            decoded_codes.append((bits_out & ((1 << bits) - 1)) - (qmax + 1)); bits_out >>= bits; nbits -= bits
    decoded = (np.asarray(decoded_codes[: usable], dtype=np.float32).reshape(-1, group) * scale[:, None]).reshape(-1)
    return packed, decoded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc"))
    ap.add_argument("--out", type=Path, default=Path("receipts/headless/FLASH_NGRAM_LOOKUP_ORACLE.json"))
    ap.add_argument("--rows", type=int, default=3, help="first/middle/last rows per shard")
    a = ap.parse_args(); started = time.perf_counter_ns(); root = a.root.resolve()
    doc = {
        "schema": "hawking.flash.ngram_lookup_oracle.v1",
        "model": "Qwen/Qwen3.8-Flash-Next",
        "source": {"root": str(root), "rows_per_shard_requested": max(1, a.rows)},
        "claim_boundary": "Selected real n-gram shard rows and a portable packed-row decode oracle only; no native Metal lookup, complete model, accepted TPS, EBPW, capability, or promotion claim.",
        "promotion_allowed": False,
    }
    if not root.is_dir() or not (root / "model.safetensors.index.json").is_file():
        doc.update({"status": "BLOCKED_SOURCE_UNAVAILABLE", "error": "pinned Flash safetensors root or index is not mounted"})
        doc["bench"] = {"state": "UNKNOWN", "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "machine": "Apple host; source unavailable", "rule": "S032 §3 -- no physical timing without source"}
    else:
        idx = json.loads((root / "model.safetensors.index.json").read_text())
        names = sorted(n for n in idx.get("weight_map", {}) if "ngram_embedding.shard_" in n)
        samples = []; totals = {"source_bytes": 0, "q4_g32_bytes": 0, "q3_g32_bytes": 0}; lookup_ns = []
        for name in names:
            path = root / idx["weight_map"][name]
            # Header-only inspection avoids touching the 102 GB table.
            with path.open("rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(hlen)); meta = header[name]
            rows = sorted(set([0, max(0, meta["shape"][0] // 2), meta["shape"][0] - 1]))[: max(1, a.rows)]
            totals["source_bytes"] += (meta["data_offsets"][1] - meta["data_offsets"][0])
            for row_no in rows:
                t0 = time.perf_counter_ns(); row, _ = read_row(path, name, row_no); lookup_ns.append(time.perf_counter_ns() - t0)
                entry = {"shard": name, "row": row_no, "source_sha256": hashlib.sha256(row.astype("<f4").tobytes()).hexdigest(), "width": int(row.size)}
                for bits, key in ((4, "q4_g32"), (3, "q3_g32")):
                    packed, decoded = quantize(row, bits)
                    aa = row[:decoded.size]; cosine = float(np.dot(aa, decoded) / max(np.linalg.norm(aa) * np.linalg.norm(decoded), 1e-30))
                    entry[key] = {"packed_bytes": len(packed), "cosine": cosine, "mae": float(np.mean(np.abs(aa - decoded)))}
                    totals[f"{key}_bytes"] += len(packed)
                samples.append(entry)
        doc.update({"status": "REAL_ROW_PACKED_LOOKUP_ORACLE", "source": {**doc["source"], "shards": len(names), "sample_rows": len(samples), "indexed_bytes": totals["source_bytes"]}, "samples": samples, "totals": totals})
        doc["bench"] = {"state": "UNKNOWN", "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "lookup_ns_mean": float(np.mean(lookup_ns)) if lookup_ns else None, "lookup_ns_p95": float(np.percentile(lookup_ns, 95)) if lookup_ns else None, "machine": "Apple host; CPU source-row reads", "rule": "S032 §3 -- portable oracle, not native execution; no quiescence claim"}
    doc["elapsed_ns"] = time.perf_counter_ns() - started
    doc["seal_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "out": str(a.out), "sample_rows": len(doc.get("samples", []))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
