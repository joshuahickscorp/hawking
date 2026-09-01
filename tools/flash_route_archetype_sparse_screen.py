#!/usr/bin/env python3
"""Screen route-conditioned expert archetypes with sparse residuals.

This is a bank-level Stage-B NR experiment.  It deliberately evaluates the
representation in function space on held-out probes, not only weight error,
and never constructs a native or complete-model claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np


MODEL = Path("/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc")
DEFAULT_PARITY = Path("receipts/headless/FLASH_FAST_COMPACT_L0_L7_PARITY.json")
DEFAULT_OUT = Path("receipts/headless/FLASH_ROUTE_ARCHETYPE_SPARSE_SCREEN_L4.json")


def tensor_locations(root: Path) -> dict[str, tuple[Path, int, tuple[int, ...], str, int]]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    out: dict[str, tuple[Path, int, tuple[int, ...], str, int]] = {}
    for shard in sorted(set(index["weight_map"].values())):
        path = root / shard
        with path.open("rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        for name, meta in header.items():
            if isinstance(meta, dict) and "data_offsets" in meta:
                begin, end = meta["data_offsets"]
                out[name] = (
                    path,
                    8 + header_len + int(begin),
                    tuple(int(x) for x in meta["shape"]),
                    str(meta["dtype"]),
                    int(end - begin),
                )
    return out


def read_experts(loc, experts: list[int]) -> np.ndarray:
    path, offset, shape, dtype, _ = loc
    if dtype != "BF16" or len(shape) != 3:
        raise ValueError(f"expected BF16 expert tensor, got {shape}/{dtype}")
    per_expert = int(np.prod(shape[1:])) * 2
    payload = bytearray()
    with path.open("rb") as fh:
        for expert in experts:
            if expert < 0 or expert >= shape[0]:
                raise ValueError(f"expert {expert} outside {shape[0]}")
            fh.seek(offset + expert * per_expert)
            chunk = fh.read(per_expert)
            if len(chunk) != per_expert:
                raise IOError(f"short read for expert {expert} from {path}")
            payload.extend(chunk)
    values = np.frombuffer(payload, dtype="<u2").astype(np.uint32)
    return (values << 16).view("<f4").reshape(len(experts), shape[1], shape[2])


def q4_function_error(mats: np.ndarray, probes: np.ndarray) -> float:
    flat = mats.reshape(mats.shape[0], mats.shape[1], -1)
    groups = flat.reshape(flat.shape[0], flat.shape[1], flat.shape[2] // 64, 64)
    scale = np.maximum(np.max(np.abs(groups), axis=3, keepdims=True) / 7.0, 1e-30)
    codes = np.clip(np.rint(groups / scale), -8, 7).astype(np.int8)
    quant = (codes.astype(np.float32) * scale).reshape(flat.shape)
    truth = np.einsum("eoi,pi->epo", mats, probes, optimize=True)
    approx = np.einsum("eoi,pi->epo", quant, probes, optimize=True)
    return float(np.linalg.norm(truth - approx) / max(float(np.linalg.norm(truth)), 1e-30))


def archetype_indices(mats: np.ndarray, count: int) -> list[int]:
    count = min(count, mats.shape[0])
    vectors = mats.reshape(mats.shape[0], -1).astype(np.float64)
    norms = np.einsum("ij,ij->i", vectors, vectors)
    chosen = [0]
    nearest = norms + norms[0] - 2.0 * (vectors @ vectors[0])
    for _ in range(1, count):
        pick = int(np.argmax(nearest))
        chosen.append(pick)
        distance = norms + norms[pick] - 2.0 * (vectors @ vectors[pick])
        nearest = np.minimum(nearest, distance)
    return chosen


def evaluate(mats: np.ndarray, probes: np.ndarray, count: int, residual_fraction: float, q4_error: float) -> dict:
    chosen = archetype_indices(mats, count)
    vectors = mats.reshape(mats.shape[0], -1).astype(np.float32)
    archetypes = vectors[chosen].astype(np.float16).astype(np.float32)
    distances = ((vectors[:, None, :] - archetypes[None, :, :]) ** 2).mean(axis=2)
    assignment = np.argmin(distances, axis=1)
    base = archetypes[assignment]
    residual = vectors - base
    total = residual.shape[1]
    keep = max(1, int(round(total * residual_fraction))) if residual_fraction else 0
    reconstructed = base.copy()
    nonzero = 0
    if keep:
        for row in range(residual.shape[0]):
            indices = np.argpartition(np.abs(residual[row]), -keep)[-keep:]
            values = residual[row, indices].astype(np.float16).astype(np.float32)
            reconstructed[row, indices] += values
            nonzero += int(indices.size)
    truth = np.einsum("eoi,pi->epo", vectors.reshape(mats.shape), probes, optimize=True)
    approx = np.einsum("eoi,pi->epo", reconstructed.reshape(mats.shape), probes, optimize=True)
    function_error = float(np.linalg.norm(truth - approx) / max(float(np.linalg.norm(truth)), 1e-30))
    weight_error = float(np.linalg.norm(vectors - reconstructed) / max(float(np.linalg.norm(vectors)), 1e-30))
    # Archetype BF16 values plus FP16 residual values and 32-bit element indexes.
    stored_bytes = count * total * 2 + nonzero * (2 + 4) + mats.shape[0] * 2
    active_bpw = stored_bytes * 8.0 / (mats.shape[0] * total)
    return {
        "archetype_count": count,
        "archetype_source_indices": [int(chosen[i]) for i in range(len(chosen))],
        "assignment": [int(x) for x in assignment],
        "residual_fraction": residual_fraction,
        "residual_elements": nonzero,
        "active_bpw": active_bpw,
        "weight_rel_fro": weight_error,
        "function_rel_fro": function_error,
        "baseline_q4_function_rel_fro": q4_error,
        "beats_q4_function": function_error < q4_error,
        "native_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=MODEL)
    parser.add_argument("--parity", type=Path, default=DEFAULT_PARITY)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--probes", type=int, default=32)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    index_path = args.root.resolve() / "model.safetensors.index.json"
    index_sha256 = hashlib.sha256(index_path.read_bytes()).hexdigest()
    parity = json.loads(args.parity.read_text())
    rows = parity.get("comparisons")
    if not isinstance(rows, list):
        rows = [
            item
            for group in parity.get("groups", [])
            for item in group.get("layers", [])
            if isinstance(item, dict)
        ]
    row = next(item for item in rows if int(item["layer"]) == args.layer)
    experts = [
        int(x)
        for x in (
            row.get("dense_route_ids")
            or row.get("route_ids_expected")
            or row.get("route_ids_observed")
        )
    ]
    if len(experts) != 10 or len(set(experts)) != 10:
        raise ValueError("expected ten distinct dense route IDs")
    locations = tensor_locations(args.root.resolve())
    rng = np.random.default_rng(3808)
    reports = []
    for tensor_suffix in ("gate_up_proj", "down_proj"):
        name = f"model.language_model.layers.{args.layer}.mlp.experts.{tensor_suffix}"
        loc = locations[name]
        mats = read_experts(loc, experts)
        probes = rng.standard_normal((args.probes, mats.shape[2]), dtype=np.float32)
        q4_error = q4_function_error(mats, probes)
        for count in (1, 2, 4, 8):
            for fraction in (0.0, 0.001, 0.005, 0.01, 0.02):
                reports.append(
                    {
                        "tensor": tensor_suffix,
                        "shape": list(mats.shape),
                        "full_tensor_bytes": int(mats.nbytes),
                        **evaluate(mats, probes, count, fraction, q4_error),
                    }
                )
    frontier = [
        item
        for item in reports
        if not any(
            other["active_bpw"] <= item["active_bpw"]
            and other["function_rel_fro"] <= item["function_rel_fro"]
            and (
                other["active_bpw"] < item["active_bpw"]
                or other["function_rel_fro"] < item["function_rel_fro"]
            )
            for other in reports
        )
    ]
    doc = {
        "schema": "hawking.flash.route_archetype_sparse_screen.v1",
        "status": "STAGE_B_NR_SCREEN",
        "artifact_kind": "NR",
        "model": "Qwen3.8-Flash-Next",
        "layer": args.layer,
        "source": {
            "root": str(args.root.resolve()),
            "parity_receipt": str(args.parity),
            "route_ids": experts,
            "route_count": len(experts),
            "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
            "index_sha256": index_sha256,
        },
        "method": "route-conditioned nearest expert archetypes with BF16 archetypes and FP16+u32 sparse residual element records; held-out random probes are evaluated in function space",
        "rows": reports,
        "frontier": frontier,
        "doctor_funnel": {
            "stage": "B",
            "baseline": "per-expert group-Q4 function error on the same probes",
            "next": "native fused organ only for a frontier row that beats Q4 function error and has credible active bytes",
        },
        "claim_boundary": "bounded real-weight route-conditioned archetype/sparse-residual screen only; no native kernel, complete-token, capability, TPS, EBPW, or residency claim",
        "promotion_allowed": False,
        "bench": {
            "state": "UNKNOWN",
            "measurement_state": "MEASURED_STAGE_B",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recorded_by": "tools/flash_route_archetype_sparse_screen.py",
            "machine": "Apple host CPU; selected real route union",
            "rule": "S032 §3 -- NR screen only",
            "elapsed_ns": time.perf_counter_ns() - started,
        },
    }
    doc["seal_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    nr = {
        "nr_version": "1.0.0",
        "nr_kind": "hawking.nos.noetic_representation",
        "artifact_kind": "NR",
        "semantic_provenance": {
            "parent_model": "Qwen3.8-Flash-Next",
            "source_index_sha256": index_sha256,
        },
        "representation": {"scope": f"layer-{args.layer} route-conditioned archetype sparse residual", "frontier_candidates": frontier, "runtime_required": False},
        "kernel_requirements": ["CODEBOOK_LOOKUP", "SPARSE_RESIDUAL", "ROUTED_SELECT", "WEIGHTED_ACCUMULATE"],
        "verifier": "native organ parity required",
        "seal": {"status": "UNSEALED_STAGE_B", "promotion_allowed": False},
        "bench": {"state": "UNKNOWN", "rule": "NR hypothesis only"},
    }
    args.out.with_suffix(".nr.json").write_text(json.dumps(nr, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "layer": args.layer, "route_count": len(experts), "rows": len(reports), "frontier": len(frontier), "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
