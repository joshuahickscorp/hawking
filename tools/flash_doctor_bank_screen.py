#!/usr/bin/env python3
"""Bounded real-weight Doctor screen for Flash's routed expert population.

Reads small contiguous slices from many real experts without materializing the
multi-GB bank.  It tests cross-expert similarity, sampled rank, and compact
quantization candidates.  Results are hypotheses for NR; no native or
complete-token promotion is implied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np


def locations(root: Path) -> dict[str, tuple[Path, int, tuple[int, ...], str]]:
    idx = json.loads((root / "model.safetensors.index.json").read_text())
    out = {}
    for shard in sorted(set(idx["weight_map"].values())):
        path = root / shard
        with path.open("rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        for name, info in header.items():
            if not isinstance(info, dict) or "data_offsets" not in info:
                continue
            out[name] = (path, 8 + hlen + int(info["data_offsets"][0]), tuple(info["shape"]), info["dtype"])
    return out


def read_bf16_slice(loc, expert: int, rows: int, cols: int) -> np.ndarray:
    path, offset, shape, dtype = loc
    if dtype != "BF16" or len(shape) != 3 or expert >= shape[0] or rows > shape[1] or cols > shape[2]:
        raise ValueError(f"unsupported geometry: shape={shape} dtype={dtype}")
    row_stride = shape[2] * 2
    start = offset + expert * shape[1] * row_stride
    raw = bytearray()
    with path.open("rb") as f:
        for row in range(rows):
            f.seek(start + row * row_stride)
            raw.extend(f.read(cols * 2))
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u16 << 16).view("<f4").reshape(rows, cols)


def quant_error(w: np.ndarray, bits: int, group: int) -> tuple[float, float]:
    flat = w.reshape(-1, w.shape[-1])
    n = flat.shape[1] // group * group
    x = flat[:, :n].reshape(flat.shape[0], n // group, group)
    qmax = (1 << (bits - 1)) - 1
    scale = np.maximum(np.abs(x).max(axis=2, keepdims=True) / qmax, 1e-30)
    q = np.clip(np.rint(x / scale), -qmax - 1, qmax).astype(np.int8)
    y = (q.astype(np.float32) * scale).reshape(flat.shape[0], n)
    a = flat[:, :n].ravel(); b = y.ravel()
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return cosine, float(np.mean(np.abs(a - b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc"))
    ap.add_argument("--layer", type=int, default=44)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--cols", type=int, default=512)
    ap.add_argument("--out", type=Path, default=Path("receipts/headless/FLASH_DOCTOR_EXPERT_BANK_SCREEN.json"))
    a = ap.parse_args()
    started = time.perf_counter_ns()
    root = a.root.resolve()
    loc = locations(root)
    gate_name = f"model.language_model.layers.{a.layer}.mlp.experts.gate_up_proj"
    down_name = f"model.language_model.layers.{a.layer}.mlp.experts.down_proj"
    gate = loc[gate_name]; down = loc[down_name]
    n_experts = gate[2][0]
    experts = np.linspace(0, n_experts - 1, min(a.experts, n_experts), dtype=int).tolist()
    gate_samples = [read_bf16_slice(gate, e, a.rows, a.cols) for e in experts]
    down_cols = min(a.cols, down[2][2])
    down_samples = [read_bf16_slice(down, e, a.rows, down_cols) for e in experts]
    def cosine(x, y):
        return float(np.dot(x.ravel(), y.ravel()) / (np.linalg.norm(x) * np.linalg.norm(y)))
    pairwise = [cosine(gate_samples[i], gate_samples[j]) for i in range(len(experts)) for j in range(i)]
    gate_stack = np.stack([x.ravel() for x in gate_samples])
    centered = gate_stack - gate_stack.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    energy = np.cumsum(sv * sv) / max(float(np.sum(sv * sv)), 1e-30)
    candidates = []
    for label, sample in (("gate_up_proj", gate_samples[0]), ("down_proj", down_samples[0])):
        for bits, group in ((16, 0), (4, 64), (4, 128), (3, 64), (3, 128)):
            if bits == 16:
                cosine_score, mae = 1.0, 0.0
                bpw = 16.0
            else:
                cosine_score, mae = quant_error(sample, bits, group)
                bpw = bits + 16 / group
            candidates.append({"tensor": label, "candidate": f"uniform_q{bits}_g{group}" if bits < 16 else "source_bf16_exact", "sample_cosine": cosine_score, "sample_mae": mae, "active_bpw": bpw, "native_ready": False})
    doc = {
        "schema": "hawking.flash.doctor_expert_bank_screen.v1",
        "status": "REAL_WEIGHT_STAGE_A_SCREEN",
        "model": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
        "source": {"root": str(root), "layer": a.layer, "gate_up_tensor": gate_name, "down_tensor": down_name, "expert_count": n_experts, "experts_sampled": experts, "rows": a.rows, "cols": a.cols},
        "population": {"cross_expert_gate_up_mean_cosine": float(np.mean(pairwise)) if pairwise else None, "cross_expert_gate_up_min_cosine": float(np.min(pairwise)) if pairwise else None, "sampled_population_rank": {"rank_1_energy": float(energy[0]) if len(energy) else None, "rank_4_energy": float(energy[min(3, len(energy)-1)]) if len(energy) else None, "rank_8_energy": float(energy[min(7, len(energy)-1)]) if len(energy) else None, "singular_values": sv.tolist()}}, "candidate_rows": candidates,
        "doctor_funnel": {"stage": "A", "next": "fit shared-basis/archetype candidates over the bank, then native-organ qualify only survivors", "early_rejection": "any candidate below 0.99 sampled cosine or requiring dense rematerialization is not promoted"},
        "bench": {"state": "UNKNOWN", "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "recorded_by": "tools/flash_doctor_bank_screen.py", "machine": "Apple M3 Ultra (CPU/header/range-read screen)", "rule": "S032 §3 -- no native timing or capability claim"},
        "claim_boundary": "Stage-A sampled real-weight population screen across a bounded expert subset. It does not establish all-expert similarity, activation fidelity, native compact execution, complete EBPW, TPS, capability, or promotion.",
        "promotion_allowed": False,
        "elapsed_ns": time.perf_counter_ns() - started,
    }
    doc["seal_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "experts": len(experts), "mean_cosine": doc["population"]["cross_expert_gate_up_mean_cosine"], "rank8_energy": doc["population"]["sampled_population_rank"]["rank_8_energy"], "out": str(a.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
