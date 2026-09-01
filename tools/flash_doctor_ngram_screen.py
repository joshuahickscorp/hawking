#!/usr/bin/env python3
"""Stage-A n-gram table screen using one real row per shard.

The table is ~102 GB, so this intentionally performs range reads only. It tests
whether shard-local rows show obvious duplication/low-dimensional structure
before any factorization or lookup-kernel work is attempted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc"))
    ap.add_argument("--out", type=Path, default=Path("receipts/headless/FLASH_DOCTOR_NGRAM_SCREEN.json"))
    a = ap.parse_args()
    start = time.perf_counter_ns()
    root = a.root.resolve(); idx = json.loads((root / "model.safetensors.index.json").read_text())
    names = sorted(n for n in idx["weight_map"] if "ngram_embedding.shard_" in n)
    rows = []; shard_meta = []
    for name in names:
        shard = root / idx["weight_map"][name]
        with shard.open("rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(hlen)); m = header[name]
            begin = 8 + hlen + m["data_offsets"][0]; shape = tuple(m["shape"])
            f.seek(begin); raw = f.read(shape[1] * 2)
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32); row = (u16 << 16).view("<f4")
        rows.append(row); shard_meta.append({"name": name, "shape": shape, "dtype": m["dtype"], "bytes": m["data_offsets"][1]-m["data_offsets"][0]})
    x = np.stack(rows) if rows else np.empty((0, 0), dtype=np.float32)
    norms = np.linalg.norm(x, axis=1) if len(x) else np.array([])
    sim = []
    for i in range(len(x)):
        for j in range(i): sim.append(float(np.dot(x[i], x[j]) / (norms[i] * norms[j])))
    centered = x - x.mean(axis=0, keepdims=True) if len(x) else x
    sv = np.linalg.svd(centered, compute_uv=False, full_matrices=False) if len(x) else np.array([])
    energy = np.cumsum(sv * sv) / max(float(np.sum(sv * sv)), 1e-30) if len(sv) else np.array([])
    candidates = []
    for bits, group in ((4, 32), (4, 64), (3, 32), (3, 64)):
        qmax = (1 << (bits - 1)) - 1
        usable = (x.shape[1] // group) * group
        z = x[:, :usable].reshape(x.shape[0], usable // group, group)
        scale = np.maximum(np.abs(z).max(axis=2, keepdims=True) / qmax, 1e-30)
        q = np.clip(np.rint(z / scale), -qmax - 1, qmax).astype(np.int8)
        deq = (q.astype(np.float32) * scale).reshape(x.shape[0], usable)
        aa = x[:, :usable].ravel(); bb = deq.ravel()
        cosine = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb)))
        candidates.append({"candidate": f"uniform_q{bits}_g{group}", "sample_cosine": cosine, "sample_mae": float(np.mean(np.abs(aa - bb))), "nominal_bpw": bits + 16 / group, "native_ready": False})
    doc = {
        "schema": "hawking.flash.doctor_ngram_screen.v1", "status": "REAL_WEIGHT_STAGE_A_NGRAM_SCREEN", "model": "Qwen/Qwen3.8-Flash-Next", "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
        "source": {"root": str(root), "shards": len(names), "rows_read": len(rows), "row_width": int(x.shape[1]) if len(x) else 0, "sample": "first row of every ngram shard"},
        "population": {"total_indexed_bytes": sum(m["bytes"] for m in shard_meta), "mean_pairwise_row_cosine": float(np.mean(sim)) if sim else None, "min_pairwise_row_cosine": float(np.min(sim)) if sim else None, "rank8_energy": float(energy[min(7, len(energy)-1)]) if len(energy) else None, "mean_row_norm": float(np.mean(norms)) if len(norms) else None, "row_sha256": hashlib.sha256(x.astype("<f4").tobytes()).hexdigest()},
        "candidate_hypotheses": [{"name": "factorized_lookup", "status": "UNTESTED", "risk": "row sample does not represent token-frequency distribution"}, {"name": "product_codebook", "status": "UNTESTED", "risk": "collision semantics and lookup fidelity"}, {"name": "shard-local_quantization", "status": "NEXT", "risk": "activation/output sensitivity unknown"}],
        "quantization_screen": candidates,
        "bench": {"state": "UNKNOWN", "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "recorded_by": "tools/flash_doctor_ngram_screen.py", "machine": "Apple M3 Ultra (range-read screen)", "rule": "S032 §3 -- no native timing or capability claim"},
        "claim_boundary": "One real row from each n-gram shard; this is a structural screen only. It does not measure token-frequency coverage, lookup fidelity, native execution, complete EBPW, TPS, or capability.", "promotion_allowed": False, "elapsed_ns": time.perf_counter_ns() - start,
    }
    doc["seal_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest(); a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "shards": len(names), "mean_cosine": doc["population"]["mean_pairwise_row_cosine"], "rank8_energy": doc["population"]["rank8_energy"], "out": str(a.out)}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
