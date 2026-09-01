#!/usr/bin/env python3
"""Stage-B shared-output basis / latent routed aggregation screen.

For the real layer-4 routed expert union, test W_i ~= U C_i with one shared
output basis U.  If this survives function-space checks, ten routed outputs
could be accumulated in latent space and expanded once.  This artifact is an
NR hypothesis only; it deliberately has no native or capability claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc")


def locs(root: Path):
    idx = json.loads((root / "model.safetensors.index.json").read_text())
    out = {}
    for shard in sorted(set(idx["weight_map"].values())):
        path = root / shard
        with path.open("rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(hlen))
        for name, meta in header.items():
            if isinstance(meta, dict) and "data_offsets" in meta:
                b, e = meta["data_offsets"]
                out[name] = (path, 8 + hlen + int(b), int(e - b), tuple(meta["shape"]), meta["dtype"])
    return out


def read(loc, experts):
    path, off, total, shape, dtype = loc
    if dtype != "BF16" or len(shape) != 3:
        raise ValueError((shape, dtype))
    per = int(np.prod(shape[1:])) * 2
    raw = bytearray()
    with path.open("rb") as fh:
        for expert in experts:
            fh.seek(off + expert * per)
            payload = fh.read(per)
            if len(payload) != per:
                raise IOError("short expert read")
            raw.extend(payload)
    vals = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (vals << 16).view("<f4").reshape(len(experts), shape[1], shape[2])


def q4(x, group=64):
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[1] % group:
        raise ValueError((x.shape, group))
    g = flat.reshape(flat.shape[0], flat.shape[1] // group, group)
    scale = np.maximum(np.max(np.abs(g), axis=2, keepdims=True) / 7.0, 1e-30)
    code = np.clip(np.rint(g / scale), -8, 7).astype(np.int8)
    return (code.astype(np.float32) * scale).reshape(x.shape), float(np.mean(np.abs(g - code * scale)))


def left_basis(mats, rank, seed=3807):
    s, m, n = mats.shape
    a = mats.transpose(1, 0, 2).reshape(m, s * n)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((a.shape[1], rank + 8), dtype=np.float32)
    y = a @ omega
    q, _ = np.linalg.qr(y, mode="reduced")
    b = q.T @ a
    ub, _, _ = np.linalg.svd(b, full_matrices=False)
    return (q @ ub[:, :rank]).astype(np.float32, copy=False)


def evaluate(mats, basis, probes, rank):
    u = basis[:, :rank].astype(np.float16).astype(np.float32)
    coeff = np.einsum("mr,smn->srn", u, mats, optimize=True)
    coeff_q, coeff_mae = q4(coeff, 64)
    recon = np.einsum("mr,srn->smn", u, coeff_q, optimize=True)
    base, base_mae = q4(mats, 64)
    true_y = np.einsum("pn,smn->smp", probes, mats, optimize=True)
    shared_y = np.einsum("mr,srp->smp", u, np.einsum("pn,srn->srp", probes, coeff_q, optimize=True), optimize=True)
    base_y = np.einsum("pn,smn->smp", probes, base, optimize=True)
    # A direct latent aggregation probe: the same shared expansion is applied
    # after a fixed positive top-k weighting, avoiding ten full output buffers.
    weights = np.linspace(1.0, 0.1, mats.shape[0], dtype=np.float32)
    weights /= weights.sum()
    true_sum = np.tensordot(weights, true_y, axes=(0, 0))
    shared_sum = np.tensordot(weights, shared_y, axes=(0, 0))
    denom_w = max(float(np.linalg.norm(mats)), 1e-30)
    denom_y = max(float(np.linalg.norm(true_y)), 1e-30)
    denom_sum = max(float(np.linalg.norm(true_sum)), 1e-30)
    s, m, n = mats.shape
    bits = (m * rank * 16 + s * rank * n * (4 + 16 / 64)) / (s * m * n)
    return {"rank": rank, "active_bpw": bits, "shared_weight_rel_fro": float(np.linalg.norm(mats - recon) / denom_w), "baseline_q4_weight_rel_fro": float(np.linalg.norm(mats - base) / denom_w), "shared_function_rel_fro": float(np.linalg.norm(true_y - shared_y) / denom_y), "baseline_q4_function_rel_fro": float(np.linalg.norm(true_y - base_y) / denom_y), "latent_aggregate_rel_fro": float(np.linalg.norm(true_sum - shared_sum) / denom_sum), "coefficient_mean_abs_error": coeff_mae, "baseline_q4_mean_abs_error": base_mae, "stored_bytes": int(m * rank * 2 + s * rank * n * (0.5 + 2 / 64)), "selected_dense_bytes": int(mats.nbytes), "shared_beats_baseline_function": float(np.linalg.norm(true_y - shared_y)) < float(np.linalg.norm(true_y - base_y)), "aggregation_is_better_than_per_expert": float(np.linalg.norm(true_sum - shared_sum)) < float(np.linalg.norm(true_sum - np.tensordot(weights, base_y, axes=(0, 0))))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=MODEL)
    ap.add_argument("--parity", type=Path, default=ROOT / "receipts/headless/FLASH_FAST_COMPACT_L0_L7_PARITY.json")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--probes", type=int, default=32)
    ap.add_argument("--out", type=Path, default=ROOT / "receipts/headless/FLASH_ROUTE_CONDITIONED_OUTPUT_BASIS_L4.json")
    args = ap.parse_args()
    started = time.perf_counter_ns()
    parity = json.loads(args.parity.read_text())
    row = next(x for x in parity["comparisons"] if x.get("layer") == args.layer)
    experts = sorted(set(int(x) for x in row["dense_route_ids"]))
    name = f"model.language_model.layers.{args.layer}.mlp.experts.down_proj"
    locations = locs(args.root.resolve())
    path, _, total, shape, dtype = locations[name]
    mats = read(locations[name], experts)
    rng = np.random.default_rng(3807)
    probes = rng.standard_normal((args.probes, shape[2]), dtype=np.float32)
    basis = left_basis(mats, 64)
    rows = [evaluate(mats, basis, probes, rank) for rank in (8, 16, 32, 64)]
    for x in rows:
        x["organ"] = "down_proj"
    frontier = [x for x in rows if not any(o["active_bpw"] <= x["active_bpw"] and o["shared_function_rel_fro"] <= x["shared_function_rel_fro"] and (o["active_bpw"] < x["active_bpw"] or o["shared_function_rel_fro"] < x["shared_function_rel_fro"]) for o in rows)]
    doc = {"schema": "hawking.flash.route_conditioned_output_basis.v1", "status": "STAGE_B_NR_SCREEN", "artifact_kind": "NR", "model": "Qwen3.8-Flash-Next", "layer": args.layer, "source": {"root": str(args.root.resolve()), "parity_receipt": str(args.parity), "route_ids": experts, "tensor": name, "shape": list(shape), "dtype": dtype, "full_tensor_bytes": total, "selected_bytes": int(mats.nbytes), "shard": str(path), "shard_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, "method": "shared output/left basis U with per-expert group-Q4 latent coefficients; weighted routed sum is accumulated in latent space and expanded once", "rows": rows, "frontier": frontier, "claim_boundary": "bounded real-weight output-basis and latent-aggregation screen only; no native kernel, complete-token, capability, TPS, EBPW, or residency claim", "promotion_allowed": False, "next_gate": "only candidates with function/aggregation error below the per-expert Q4 control may receive fused native organ parity", "bench": {"state": "UNKNOWN", "measurement_state": "MEASURED_STAGE_B", "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "machine": "Apple host CPU; selected real route union", "rule": "S032 §3 -- NR screen only", "elapsed_ns": time.perf_counter_ns() - started}, "parity_binding": {"receipt_sha256": hashlib.sha256(args.parity.read_bytes()).hexdigest()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    nr = {"nr_version": "1.0.0", "nr_kind": "hawking.nos.noetic_representation", "artifact_kind": "NR", "semantic_provenance": {"parent_model": "Qwen3.8-Flash-Next", "source_weight_hash": doc["source"]["shard_sha256"]}, "representation": {"scope": f"layer-{args.layer} routed output basis", "candidates": rows, "frontier_candidates": frontier, "runtime_required": False}, "dependencies": [], "kernel_requirements": [{"requires": "latent_routed_accumulate_then_expand", "parameters": "shared output basis and coefficient group"}], "verifier": "native fused organ parity required", "seal": {"status": "UNSEALED_STAGE_B", "promotion_allowed": False}, "bench": {"state": "UNKNOWN", "rule": "NR hypothesis only"}}
    args.out.with_suffix(".nr.json").write_text(json.dumps(nr, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "layer": args.layer, "route_count": len(experts), "rows": rows, "frontier": frontier, "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
