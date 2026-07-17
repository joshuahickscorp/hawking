#!/usr/bin/env python3.12
"""Gravity Forge tournament on the REAL GPT-OSS-120B experts.

Runs the materially-distinct Forge families (transform_pq, shared_expert_grammar,
repairability_shaped) against the naive baselines (naive_rvq, low_rank) at matched sub-bit
whole-artifact BPW, on the actual on-disk dequantized experts, and records for each cell:

  * achieved whole_artifact_bpw (EVERYTHING counted), base/doctor/overhead split;
  * F1  weight-space relative Frobenius error (PROXY, diagnostic only);
  * F1.5 output-space divergence of the reference MoE block forward (proto-F2, synthetic acts).

It is bounded (a few experts / configs) so it completes synchronously and checkpoints each cell,
exactly like the naive run. It writes a sealed FORGE_FRONTIER.json. It DOES NOT enable Gravity,
emit an Event-Horizon Telegram, or authorize any escape - all of that stays gated.

Honest boundary re-stated in the output: no cell here is a capability-parity claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gravity_forge as gf          # noqa: E402
import gptoss_moe_runtime as rt     # noqa: E402

RUN_SCHEMA = "hawking.gravity_forge.run.v1"
OUT = Path("reports/condense/gravity_forge")
CELLS = OUT / "cells"


def _rel(w, r):
    return gf._rel_error(w, r)


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def _cfgs_for(rows: int, cols: int) -> list[dict[str, Any]]:
    """Config grid per family aimed at ~0.5 and ~0.8 whole-artifact BPW on a [rows,cols] expert.
    On large experts codebooks amortize to ~0, so index-rate ~= stages*log2(k)/dim dominates."""
    return [
        # naive baselines (the sealed negative), at matched rate
        {"family": "naive_rvq", "rate": 0.5, "kw": dict(dim=32, k=256, stages=2)},   # 2*8/32=0.5
        {"family": "naive_rvq", "rate": 0.8, "kw": dict(dim=16, k=64, stages=2)},    # 2*6/16=0.75
        {"family": "low_rank", "rate": 0.5, "kw": dict(rank=max(1, int(0.5 * rows * cols / (16 * (rows + cols)))))},
        {"family": "low_rank", "rate": 0.8, "kw": dict(rank=max(1, int(0.8 * rows * cols / (16 * (rows + cols)))))},
        # Forge advanced families
        {"family": "transform_pq", "rate": 0.5, "kw": dict(dim=32, subspaces=2, k=256)},  # 2*8/32=0.5
        {"family": "transform_pq", "rate": 0.8, "kw": dict(dim=16, subspaces=2, k=64)},    # 2*6/16=0.75
        {"family": "repairability_shaped", "rate": 0.8,
         "kw": dict(base_dim=32, base_k=64, corr_rank=8, sparse_rows=16)},
        {"family": "repairability_shaped", "rate": 0.5,
         "kw": dict(base_dim=32, base_k=16, corr_rank=4, sparse_rows=8)},
        # ternary latent factorization lives in the DEEP sub-bit regime (cheap per rank), so it is
        # reported at its natural ranks rather than matched to the 0.5/0.8 grid.
        {"family": "ternary_factor", "rate": 0.02, "kw": dict(rank=16)},
        {"family": "ternary_factor", "rate": 0.07, "kw": dict(rank=64)},
    ]


def _pack(family: str, w: np.ndarray, kw: dict[str, Any]) -> gf.PackedArtifact:
    fn = {"naive_rvq": gf.pack_naive_rvq, "low_rank": gf.pack_low_rank,
          "transform_pq": gf.pack_transform_pq,
          "repairability_shaped": gf.pack_repairability_shaped,
          "ternary_factor": gf.pack_ternary_factor}[family]
    return fn(w, **kw)


def _write_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Seal + persist one tournament cell (resumable checkpoint)."""
    cell["cell_sha256"] = _sha({k: v for k, v in cell.items() if k != "cell_sha256"})
    tag = "_".join(str(cell.get(k)) for k in ("family", "kind", "block", "organ", "target_bpw") if k in cell)
    (CELLS / f"{tag}.json").write_text(json.dumps(cell, indent=2, sort_keys=True, default=str))
    return cell


def run(*, blocks: list[int], experts: list[int], organs: list[str],
        cluster_size: int = 4, do_output_div: bool = True) -> dict[str, Any]:
    reader = rt.ProvenanceReader()
    sample = reader.by_name.get("block.0.mlp.gate.weight")
    if sample is None or not Path(sample["shard_path"]).exists():
        raise RuntimeError("120B source shards absent; cannot run Forge on real experts")
    CELLS.mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, Any]] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ---- per-expert single-tensor families ----
    for b in blocks:
        for e in experts:
            ex = rt.load_expert(reader, b, e)
            for organ in organs:
                w = ex[organ].astype(np.float32)
                rows, cols = w.shape
                for cfg in _cfgs_for(rows, cols):
                    t0 = time.time()
                    art = _pack(cfg["family"], w, cfg["kw"])
                    err = _rel(w, art.recon)
                    cell = {
                        "kind": "single_tensor", "block": b, "expert": e, "organ": organ,
                        "shape": [rows, cols], "family": cfg["family"], "target_bpw": cfg["rate"],
                        "whole_artifact_bpw": round(art.whole_artifact_bpw, 4),
                        "base_bpw": round(art.base_bpw, 4), "doctor_bpw": round(art.doctor_bpw, 4),
                        "overhead_bpw": round(art.overhead_bpw, 4),
                        "f1_weight_rel_error": round(err, 5), "f1_verdict": gf._verdict(err),
                        "physical_bytes": art.physical_bytes, "config": art.config,
                        "seconds": round(time.time() - t0, 2), "signal": "weight_proxy",
                    }
                    cells.append(_write_cell(cell))

    # ---- shared_expert_grammar: amortize across a cluster of experts (the MoE lever) ----
    for b in blocks:
        organ = "mlp1"
        cluster = [rt.load_expert(reader, b, e)[organ].astype(np.float32) for e in range(cluster_size)]
        for target, kw in [(0.5, dict(dim=32, k=256, stages=2, corr_rank=0)),
                           (0.8, dict(dim=16, k=64, stages=2, corr_rank=4))]:
            t0 = time.time()
            art = gf.pack_shared_grammar(cluster, **kw)
            err = _rel(np.stack(cluster), art.recon)
            cells.append(_write_cell({
                "kind": "expert_cluster", "block": b, "organ": organ,
                "n_experts": cluster_size, "family": "shared_expert_grammar", "target_bpw": target,
                "whole_artifact_bpw": round(art.whole_artifact_bpw, 4),
                "base_bpw": round(art.base_bpw, 4), "doctor_bpw": round(art.doctor_bpw, 4),
                "overhead_bpw": round(art.overhead_bpw, 4),
                "f1_weight_rel_error": round(err, 5), "f1_verdict": gf._verdict(err),
                "physical_bytes": art.physical_bytes, "config": art.config,
                "seconds": round(time.time() - t0, 2), "signal": "weight_proxy",
            }))

    # ---- F1.5 output-space divergence: pack exactly the ROUTED experts, compare block output ----
    output_div = None
    if do_output_div:
        b = blocks[0]
        pack_fn = lambda w: gf.pack_transform_pq(w, dim=16, subspaces=2, k=64).recon  # ~0.75 bpw
        output_div = gf.output_divergence(reader, b, pack_fn, n_inputs=8)
        output_div["family"] = "transform_pq"
        output_div["target_bpw"] = 0.8

    doc = {
        "schema": RUN_SCHEMA, "parent": "120B", "hf_or_source_id": "openai/gpt-oss-120b",
        "started_at": started, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "blocks": blocks, "experts": experts, "organs": organs, "cluster_size": cluster_size,
        "device": gf._device().type,
        "evidence_level": "F1 (weight proxy) + F1.5 (output proxy)",
        "is_deployable_artifact": False, "authorizes_escape": False, "is_event_horizon": False,
        "capability_parity": False,
        "note": "Materially-distinct Forge families on real GPT-OSS-120B experts. Weight error is a "
                "PROXY; output divergence uses synthetic activations until the tokenizer+runtime land. "
                "No cell authorizes a Gravity escape or an Event-Horizon seal.",
        "cells": cells, "output_divergence": output_div,
    }
    doc["run_sha256"] = _sha({k: v for k, v in doc.items() if k != "run_sha256"})
    (OUT / "FORGE_FRONTIER.json").write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gravity Forge tournament on real 120B experts.")
    ap.add_argument("--blocks", default="0,35")
    ap.add_argument("--experts", default="0")
    ap.add_argument("--organs", default="mlp1,mlp2")
    ap.add_argument("--cluster-size", type=int, default=4)
    ap.add_argument("--no-output-div", action="store_true")
    args = ap.parse_args(argv)
    doc = run(blocks=[int(b) for b in args.blocks.split(",")],
              experts=[int(e) for e in args.experts.split(",")],
              organs=args.organs.split(","), cluster_size=args.cluster_size,
              do_output_div=not args.no_output_div)
    # compact human summary
    print(f"device={doc['device']}  cells={len(doc['cells'])}")
    print(f"{'family':<22}{'target':>7}{'whole_bpw':>11}{'base':>7}{'doctor':>8}{'weight_err':>12}  verdict")
    for c in doc["cells"]:
        tag = c.get("organ", c.get("kind", ""))
        print(f"{c['family']:<22}{c['target_bpw']:>7}{c['whole_artifact_bpw']:>11}"
              f"{c['base_bpw']:>7}{c['doctor_bpw']:>8}{c['f1_weight_rel_error']:>12}  "
              f"{c['f1_verdict']}  [{tag}]")
    if doc["output_divergence"]:
        od = doc["output_divergence"]
        print(f"\noutput-divergence (proto-F2, {od['family']} @ {od['target_bpw']} bpw): "
              f"mean_rel={od['mean_output_rel_div']} max_rel={od['max_output_rel_div']} "
              f"experts_exercised={od['n_experts_exercised']} (capability_parity={od['capability_parity']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
