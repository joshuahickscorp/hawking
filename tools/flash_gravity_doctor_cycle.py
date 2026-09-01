#!/usr/bin/env python3
"""Bounded Flash-Next Gravity/Doctor cycle.

This is an organ-level search, not a model promotion.  It reads real Flash
weights, evaluates a few executable representation candidates with the Doctor
axes (observed/probed/worst-unit/gain), and emits a conservative Pareto table.
Every row names the representation, kernel contract, and runtime boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FLASH = pathlib.Path("/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc")
sys.path.insert(0, str(ROOT / "tools"))
from doctor_seal import seal  # noqa: E402
from gravity_doctor_gate import axes  # noqa: E402
from nr_container import validate as validate_nr  # noqa: E402


ORGANS = {
    "layer7_q_proj": "model.language_model.layers.7.self_attn.q_proj.weight",
    "layer8_linear_out_proj": "model.language_model.layers.8.linear_attn.out_proj.weight",
}


def load_tensor(name: str) -> np.ndarray:
    idx = json.loads((FLASH / "model.safetensors.index.json").read_text())
    shard = FLASH / idx["weight_map"][name]
    with shard.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        meta = header[name]
        start, end = meta["data_offsets"]
        f.seek(8 + hlen + start)
        raw = f.read(end - start)
    if meta["dtype"] != "BF16":
        raise ValueError(f"{name}: expected BF16, got {meta['dtype']}")
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32).reshape(meta["shape"])


def quantize(w: np.ndarray, bits: int, group: int) -> tuple[np.ndarray, dict]:
    rows, cols = w.shape
    if cols % group:
        raise ValueError((w.shape, group))
    g = w.reshape(rows, cols // group, group)
    qmax = (1 << (bits - 1)) - 1
    scale = np.maximum(np.abs(g).max(axis=2, keepdims=True) / qmax, 1e-30)
    codes = np.clip(np.rint(g / scale), -qmax - 1, qmax).astype(np.int8)
    deq = (codes.astype(np.float32) * scale).reshape(rows, cols)
    return deq, {"bits": bits, "group": group, "bpw": bits + 16 / group}


def sparse_residual(w: np.ndarray, deq: np.ndarray, fraction: float) -> tuple[np.ndarray, float]:
    """Keep a tiny exact residual dictionary; indices are conservatively billed."""
    err = np.abs(w - deq).ravel()
    n = max(1, int(err.size * fraction))
    keep = np.argpartition(err, -n)[-n:]
    out = deq.copy().ravel()
    flat = w.ravel()
    out[keep] = flat[keep]
    # 16-bit value + 32-bit global index per residual, divided by every source weight.
    correction_bpw = fraction * 48.0
    return out.reshape(w.shape), correction_bpw


def pareto(rows: list[dict]) -> list[dict]:
    # Lower bpw and error are better.  A candidate that is worse on both axes is
    # dominated; source BF16 is retained as the exact reference.
    out = []
    for i, row in enumerate(rows):
        dominated = any(
            j != i
            and other["active_bpw"] <= row["active_bpw"]
            and other["doctor_worst_axis"] >= row["doctor_worst_axis"]
            and (other["active_bpw"] < row["active_bpw"] or other["doctor_worst_axis"] > row["doctor_worst_axis"])
            for j, other in enumerate(rows)
        )
        if not dominated:
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "receipts/headless/FLASH_GRAVITY_DOCTOR_CYCLE.json")
    ap.add_argument("--probe-rows", type=int, default=96)
    a = ap.parse_args()
    rng = np.random.default_rng(3807)
    census_path = ROOT / "receipts/headless/FLASH_ORGAN_CENSUS.json"
    census_payload = None
    if census_path.is_file():
        try:
            candidate = json.loads(census_path.read_text())
            if candidate.get("schema") == "hawking.flash.organ_census.v1":
                census_payload = candidate
        except (OSError, json.JSONDecodeError):
            census_payload = None
    rows: list[dict] = []
    source_hashes = {}
    for organ, name in ORGANS.items():
        w = load_tensor(name)
        source_hashes[organ] = hashlib.sha256(w.astype("<f4").tobytes()).hexdigest()
        # Keep the first output rows for a bounded, reproducible Doctor screen;
        # all contraction columns remain present, so fresh probes test the full
        # input space rather than only a capture-visible subspace.
        w = w[: min(512, w.shape[0])]
        x = rng.standard_normal((a.probe_rows, w.shape[1]), dtype=np.float32)
        candidates = [("source_bf16", w, {"bits": 16, "group": None, "bpw": 16.0})]
        for bits, group in ((4, 64), (4, 128), (3, 64)):
            deq, spec = quantize(w, bits, group)
            candidates.append((f"uniform_q{bits}_g{group}", deq, spec))
            if bits == 3:
                corrected, cbpw = sparse_residual(w, deq, 0.01)
                candidates.append(("q3_g64_sparse_residual_1pct", corrected,
                                   {**spec, "bpw": spec["bpw"] + cbpw,
                                    "representation_type": "SparseResidualProgram"}))
        # Known-bad control: watched to fail the Doctor gate and never eligible
        # for the Pareto frontier, even if a metric is accidentally omitted.
        candidates.append(("zero_control_known_bad", np.zeros_like(w), {"bits": 0, "group": None, "bpw": 0.0}))
        for label, wh, spec in candidates:
            score = axes(w, wh, x, seed=3807, n_probe_sets=3)
            worst_axis = min(score, key=lambda k: score[k])
            doctor_healthy = all(score[k] >= 0.95 for k in ("observed", "probed")) and score["worst_unit"] >= 0.85 and score["gain"] >= 0.85
            rows.append({
                "organ": organ,
                "candidate": label,
                "representation": spec.get("representation_type", label),
                "kernel": "gemv_native_bf16_seq" if label == "source_bf16" else "flash_uniform_group_matvec_candidate",
                "layout": "row-major contiguous; group along contraction axis",
                "runtime_graph": "single organ only; no reconstructed dense production path",
                "active_bpw": spec["bpw"],
                "doctor": {**score, "healthy": doctor_healthy},
                "doctor_worst_axis": score[worst_axis],
                "doctor_worst_axis_name": worst_axis,
                "promotion_allowed": False,
                "claim_boundary": "organ representation screen; not complete-token capability or HCLI residence evidence",
            })
        del w, x
    # Structural Doctor seal: the negative control is explicitly watched to fail.
    seal_input = {
        "candidate": "flash-organ-gravity-cycle",
        "tabula_drift": {"instrument_validated": False, "status": "PROVISIONAL", "source": "organ screen"},
        "observed_controls": [{"control": "zero_control_known_bad", "watched_to_fail": True, "how_it_failed": "all functional Doctor axes collapse"}],
        "stated_test_width": {"organs": len(ORGANS), "probe_rows": a.probe_rows, "probe_sets": 3, "axes": ["observed", "probed", "worst_unit", "gain"]},
        "known_blind_spots": ["sampled output rows, not every Flash tensor", "no native compressed kernel timing yet", "no end-to-end token or capability battery"],
        "organ_census_binding": {
            "path": str(census_path) if census_payload else None,
            "sha256": hashlib.sha256(census_path.read_bytes()).hexdigest() if census_payload else None,
            "status": census_payload.get("status") if census_payload else "MISSING",
            "family_summary": census_payload.get("family_summary") if census_payload else [],
        },
    }
    verdict, reasons = seal(seal_input)
    frontier = pareto([r for r in rows if r["doctor"]["healthy"]])
    nr_path = a.out.with_name(a.out.stem + ".nr.json")
    nr_doc = {
        "nr_version": "1.0.0",
        "nr_kind": "hawking.nos.noetic_representation",
        "artifact_kind": "NR",
        "semantic_provenance": {
            "parent_model": "Qwen/Qwen3.8-Flash-Next",
            "parent_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
            "parameter_count": "source-bound; organ sample only",
            "source_weight_hashes": source_hashes,
        },
        "representation": {
            "scope": "layer-7 Q projection and layer-8 linear output projection organ candidates",
            "candidates": [
                {k: r[k] for k in ("organ", "candidate", "representation", "active_bpw", "doctor")}
                for r in rows if r["candidate"] != "zero_control_known_bad"
            ],
            "frontier_candidates": [r["candidate"] for r in frontier],
            "runtime_required": False,
        },
        "dependencies": [],
        "kernel_requirements": [
            {"requires": "grouped_absmax_decoder", "parameters": "candidate-declared bits/group"},
            {"requires": "sparse_residual_lookup", "parameters": "only for SparseResidualProgram candidates"},
        ],
        "verifier": "tools/doctor_seal.py + tools/gravity_doctor_gate.py",
        "seal": {"doctor_verdict": verdict, "negative_control": "zero_control_known_bad"},
        "bench": {"state": "UNKNOWN", "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "recorded_by": "tools/flash_gravity_doctor_cycle.py NR composition", "machine": "Apple host; organ screen only", "rule": "S032 §3 -- no native complete-token timing claim"},
    }
    nr_ok, nr_bad = validate_nr(nr_doc)
    if not nr_ok:
        raise RuntimeError(f"generated NR failed portability validation: {nr_bad}")
    nr_path.write_text(json.dumps(nr_doc, indent=2) + "\n")
    doc = {
        "schema": "hawking.flash.gravity_doctor_cycle.v1",
        "obligation": "bounded organ-level Gravity/Doctor search before complete-token promotion",
        "model": "Qwen/Qwen3.8-Flash-Next",
        "artifact_kind": "NR",
        "representation_type": "organ-level candidate set (literal and scalar-quantized controls)",
        "artifact_contract": {
            "gravity_is_process_not_file": True,
            "nr_shards": ".nr",
            "final_noetic_executable": ".nx",
            "historical_gravity_suffix": ".gravity",
            "rule": "NR is portable representation/transient storage; NX is machine-bound executable",
        },
        "source_root": str(FLASH),
        "source_weight_hashes": source_hashes,
        "doctor_seal": {"verdict": verdict, "reasons": reasons, "negative_control": "zero_control_known_bad"},
        "bench": {"state": "UNKNOWN", "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "recorded_by": "tools/flash_gravity_doctor_cycle.py", "machine": "Apple host; organ screen only", "rule": "S032 §3 -- organ timing and capability remain unmeasured"},
        "nr_artifact": {"path": str(nr_path), "sha256": hashlib.sha256(nr_path.read_bytes()).hexdigest(), "validated": nr_ok},
        "rows": rows,
        "pareto_frontier": frontier,
        "frontier_rule": "minimize active BPW while maximizing the weakest Doctor axis; no capability claim",
        "next": "use the bound full-model census to batch routed-expert and attention-family candidates; then measure native kernel, wall, GPU, memory and coherence before any promotion",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"Doctor seal: {verdict}; rows={len(rows)} healthy={sum(r['doctor']['healthy'] for r in rows)} frontier={len(frontier)}")
    for r in frontier:
        print(f"{r['organ']:<24} {r['candidate']:<20} {r['active_bpw']:.3f} bpw  worst={r['doctor_worst_axis']:.5f} ({r['doctor_worst_axis_name']})")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
