#!/usr/bin/env python3
"""Stage-B Flash representation screen for route-conditioned expert sharing.

The layer-4 compact-chain parity receipt found a real failure boundary: a small
layer-3 numeric drift changed a later top-k route.  This probe therefore uses
the *actual* layer-4 dense route union and asks a narrower question than
"compress the bank": can those routed experts share an input-side basis while
retaining the function seen by the router/MLP?  It is an NR hypothesis only.

Weights are read by expert range, never by materialising the 512-expert bank.
The shared basis is fp16 and per-expert coefficients are group-Q4.  Errors are
reported both in weight space and on held-out random function probes.  A
per-expert Q4 baseline is measured at the same time.  No native execution,
complete-token, capability, or residency claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def locations(root: Path) -> dict[str, tuple[Path, int, int, tuple[int, ...], str]]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    out: dict[str, tuple[Path, int, int, tuple[int, ...], str]] = {}
    for shard in sorted(set(index["weight_map"].values())):
        path = root / shard
        with path.open("rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(hlen))
        for name, meta in header.items():
            if not isinstance(meta, dict) or "data_offsets" not in meta:
                continue
            begin, end = meta["data_offsets"]
            out[name] = (path, 8 + hlen + int(begin), int(end - begin), tuple(meta["shape"]), meta["dtype"])
    return out


def read_experts(loc, experts: list[int]) -> np.ndarray:
    path, offset, total, shape, dtype = loc
    if dtype != "BF16" or len(shape) != 3 or max(experts) >= shape[0]:
        raise ValueError(f"unsupported expert geometry: {shape} {dtype}")
    per = int(np.prod(shape[1:])) * 2
    raw = bytearray()
    with path.open("rb") as fh:
        for expert in experts:
            fh.seek(offset + expert * per)
            payload = fh.read(per)
            if len(payload) != per:
                raise IOError(f"short expert read: {path} expert={expert}")
            raw.extend(payload)
    vals = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (vals << 16).view("<f4").reshape(len(experts), shape[1], shape[2])


def q4(x: np.ndarray, group: int = 64) -> tuple[np.ndarray, float]:
    """Symmetric group Q4, with one fp16 scale per group."""
    flat = x.reshape(-1, x.shape[-1])
    groups = flat.shape[1] // group
    if groups * group != flat.shape[1]:
        raise ValueError((x.shape, group))
    g = flat.reshape(flat.shape[0], groups, group)
    qmax = 7.0
    scale = np.maximum(np.max(np.abs(g), axis=2, keepdims=True) / qmax, 1e-30)
    code = np.clip(np.rint(g / scale), -8, 7).astype(np.int8)
    return (code.astype(np.float32) * scale).reshape(x.shape), float(np.mean(np.abs(g - code * scale)))


def shared_right_basis(mats: np.ndarray, rank: int, seed: int = 3807) -> np.ndarray:
    """Randomized range finder over the expert-stacked row space."""
    stacked = mats.reshape(-1, mats.shape[-1]).astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((stacked.shape[1], rank + 8), dtype=np.float32)
    y = stacked @ omega
    q, _ = np.linalg.qr(y, mode="reduced")
    b = q.T @ stacked
    _, _, vt = np.linalg.svd(b, full_matrices=False)
    return vt[:rank].T.astype(np.float32, copy=False)


def evaluate(mats: np.ndarray, basis: np.ndarray, probes: np.ndarray, rank: int) -> dict:
    v = basis[:, :rank]
    # Store the shared basis as fp16, then quantize each expert's coefficient
    # matrix.  This prices the candidate instead of silently scoring fp32 math.
    vf = v.astype(np.float16).astype(np.float32)
    coeff = np.einsum("sen,nr->ser", mats, vf, optimize=True)
    group = min(64, rank)
    coeff_q, coeff_mae = q4(coeff, group)
    recon = np.einsum("ser,nr->sen", coeff_q, vf, optimize=True)
    baseline, baseline_mae = q4(mats, 64)
    x = probes
    true_y = np.einsum("pn,sen->sep", x, mats, optimize=True)
    # Project the probe through the shared input basis before applying each
    # expert's coefficient matrix.  (Using an unshared label here would sum
    # the probe and coefficient dimensions independently and fake a huge error.)
    projected_x = x @ vf
    shared_y = np.einsum("pr,ser->sep", projected_x, coeff_q, optimize=True)
    base_y = np.einsum("pn,sen->sep", x, baseline, optimize=True)
    denom_w = max(float(np.linalg.norm(mats)), 1e-30)
    denom_y = max(float(np.linalg.norm(true_y)), 1e-30)
    shared_w = float(np.linalg.norm(mats - recon) / denom_w)
    base_w = float(np.linalg.norm(mats - baseline) / denom_w)
    shared_f = float(np.linalg.norm(true_y - shared_y) / denom_y)
    base_f = float(np.linalg.norm(true_y - base_y) / denom_y)
    s, m, n = mats.shape
    bits_per_elem = (n * rank * 16 + s * m * rank * (4 + 16 / group)) / (s * m * n)
    return {
        "rank": rank,
        "coefficient_group": group,
        "active_bpw": bits_per_elem,
        "shared_weight_rel_fro": shared_w,
        "baseline_q4_weight_rel_fro": base_w,
        "shared_function_rel_fro": shared_f,
        "baseline_q4_function_rel_fro": base_f,
        "shared_beats_baseline_function": shared_f < base_f,
        "coefficient_mean_abs_error": coeff_mae,
        "baseline_q4_mean_abs_error": baseline_mae,
        "stored_bytes": int((n * rank * 2) + s * m * rank * (0.5 + 2 / group)),
        "dense_selected_bytes": int(mats.nbytes),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--parity", type=Path, default=ROOT / "receipts/headless/FLASH_FAST_COMPACT_L0_L7_PARITY.json")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--probes", type=int, default=96)
    ap.add_argument("--out", type=Path, default=ROOT / "receipts/headless/FLASH_ROUTE_CONDITIONED_SHARED_BASIS_L4.json")
    args = ap.parse_args()
    started = time.perf_counter_ns()
    parity = json.loads(args.parity.read_text())
    comparison = next(row for row in parity["comparisons"] if row.get("layer") == args.layer)
    experts = sorted(set(int(x) for x in comparison["dense_route_ids"]))
    loc = locations(args.root.resolve())
    organs = {
        "gate_up_proj": f"model.language_model.layers.{args.layer}.mlp.experts.gate_up_proj",
        "down_proj": f"model.language_model.layers.{args.layer}.mlp.experts.down_proj",
    }
    rng = np.random.default_rng(3807)
    rows = []
    source = {}
    for organ, name in organs.items():
        path, _, total, shape, dtype = loc[name]
        mats = read_experts(loc[name], experts)
        source[organ] = {"tensor": name, "shard": str(path), "dtype": dtype, "shape": list(shape), "full_tensor_bytes": total, "selected_bytes": int(mats.nbytes), "sha256": sha256(path)}
        probes = rng.standard_normal((args.probes, shape[2]), dtype=np.float32)
        max_rank = min(64, shape[2], mats.shape[1])
        basis = shared_right_basis(mats, max_rank)
        for rank in (8, 16, 32, 64):
            if rank <= max_rank:
                result = evaluate(mats, basis, probes, rank)
                rows.append({"organ": organ, **result})
        del mats, basis
    by_organ = {organ: [r for r in rows if r["organ"] == organ] for organ in organs}
    frontier = [r for r in rows if not any(
        o["organ"] == r["organ"] and o["active_bpw"] <= r["active_bpw"] and
        o["shared_function_rel_fro"] <= r["shared_function_rel_fro"] and
        (o["active_bpw"] < r["active_bpw"] or o["shared_function_rel_fro"] < r["shared_function_rel_fro"])
        for o in rows)]
    doc = {
        "schema": "hawking.flash.route_conditioned_shared_basis.v1",
        "status": "STAGE_B_NR_SCREEN",
        "artifact_kind": "NR",
        "model": "Qwen3.8-Flash-Next",
        "layer": args.layer,
        "source": {"root": str(args.root.resolve()), "parity_receipt": str(args.parity), "route_ids": experts, "route_source": "dense layer comparison", "source_tensors": source},
        "method": "shared input/right basis over the actual routed expert union; fp16 shared basis plus per-expert group-Q4 coefficients; held-out random function probes",
        "rows": rows,
        "frontier": frontier,
        "organ_summary": by_organ,
        "claim_boundary": "bounded representation hypothesis screen only; no native compact kernel, complete-token, capability, TPS, EBPW, or residency claim",
        "promotion_allowed": False,
        "next_gate": "if a route-conditioned row beats per-expert Q4 in function space, implement compact fused decode and exact organ parity; otherwise retain as negative science",
        "bench": {"state": "UNKNOWN", "measurement_state": "MEASURED_STAGE_B", "probes": args.probes, "elapsed_ns": time.perf_counter_ns() - started, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "machine": "Apple host CPU; source range reads", "rule": "S032 §3 -- representation screen timing is not a native execution claim", "cold_warm": "unspecified; filesystem cache state not controlled"},
        "parity_binding": {"receipt_sha256": hashlib.sha256(args.parity.read_bytes()).hexdigest(), "first_route_mismatch_layer": parity.get("divergence", {}).get("first_route_id_mismatch_layer")},
    }
    nr = {
        "nr_version": "1.0.0", "nr_kind": "hawking.nos.noetic_representation", "artifact_kind": "NR",
        "semantic_provenance": {"parent_model": "Qwen/Qwen3.8-Flash-Next", "parent_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915", "source_weight_hashes": {k: v["sha256"] for k, v in source.items()}},
        "representation": {"scope": f"layer-{args.layer} routed expert union {experts}", "candidates": rows, "frontier_candidates": frontier, "runtime_required": False},
        "dependencies": [], "kernel_requirements": [{"requires": "shared_basis_fused_decode", "parameters": "rank and coefficient group"}],
        "verifier": "tools/flash_route_conditioned_shared_basis.py; exact organ parity required before native promotion",
        "seal": {"status": "UNSEALED_STAGE_B", "promotion_allowed": False},
        "bench": {"state": "MEASURED_STAGE_B", "rule": "NR hypothesis only; no NX claim"},
    }
    nr_path = args.out.with_suffix(".nr.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    nr_path.write_text(json.dumps(nr, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "layer": args.layer, "route_count": len(experts), "rows": rows, "frontier": frontier, "out": str(args.out), "nr": str(nr_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
