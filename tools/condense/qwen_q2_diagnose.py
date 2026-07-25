#!/usr/bin/env python3.12
"""Isolate which Qwen expert organ caused the first <=0.77-BPW frontier failure."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

import qwen_correction_wave as C
import qwen_real_forward as Q

ROOT = Path(__file__).resolve().parents[2]
GF = ROOT / "reports/condense/general_frontier"


def _atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _metrics(ref: np.ndarray, got: np.ndarray) -> dict[str, float]:
    a, b = ref.astype(np.float64), got.astype(np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    cosine = float((a @ b) / max(na * nb, 1e-30))
    rel = float(np.linalg.norm(a - b) / max(na, 1e-30))
    gain = float((a @ b) / max(float(b @ b), 1e-30))
    gain_rel = float(np.linalg.norm(a - gain * b) / max(na, 1e-30))
    return {"relative_error": round(rel, 8), "cosine": round(cosine, 8),
            "optimal_scalar_gain": round(gain, 8),
            "gain_corrected_relative_error": round(gain_rel, 8)}


def run(meta_dir: Path, cache_dir: Path, q2_path: Path, output: Path, state: Path,
        candidate: str) -> int:
    q2 = json.loads(q2_path.read_text())
    token_id = int(q2["bounded_token_id"])
    expert_id = int(q2["first_gravity_candidate"]["real_selected_expert"])
    fwd = Q.from_remote(meta_dir, cache_dir=cache_dir)
    x0 = fwd.reader.bf16_rows("model.embed_tokens.weight", [token_id])
    x = x0 + fwd._attention(0, x0)
    h = Q.rmsnorm(x, fwd.reader.bf16("model.layers.0.post_attention_layernorm.weight"), fwd.g.eps)[0]
    source = fwd._load_expert(0, expert_id)

    if candidate not in C.CANDIDATES:
        raise ValueError(f"unknown candidate {candidate}")
    mapping = C.CANDIDATES[candidate]
    packed: dict[str, np.ndarray] = {}
    class_bpw: dict[str, float] = {}
    for cls in C.CLASSES:
        spec = mapping[cls]
        row = C.forge_pack(spec["family"], source[cls], seed=expert_id, params=spec["params"])
        packed[cls] = row["recon"]
        class_bpw[cls] = round(float(row["whole_bpw"]), 8)

    def output_for(use: set[str]) -> np.ndarray:
        gate = packed["gate"] if "gate" in use else source["gate"]
        up = packed["up"] if "up" in use else source["up"]
        down = packed["down"] if "down" in use else source["down"]
        return down @ Q.swiglu(gate @ h, up @ h)

    ref = output_for(set())
    variants = {
        "gate_only": {"gate"}, "up_only": {"up"}, "down_only": {"down"},
        "gate_up": {"gate", "up"}, "all_three": {"gate", "up", "down"},
    }
    metrics = {name: _metrics(ref, output_for(classes)) for name, classes in variants.items()}
    singles = {c: metrics[f"{c}_only"]["relative_error"] for c in C.CLASSES}
    dominant = max(singles, key=singles.get)
    gain_improvement = (metrics["all_three"]["relative_error"]
                        - metrics["all_three"]["gain_corrected_relative_error"])
    if dominant in ("gate", "up"):
        next_move = ("Qwen refutes the direct 120B robustness ordering on this activation: move budget "
                     "from down islands into activation-aware gate/up PQ, holding <=0.77 BPW.")
    else:
        next_move = ("Qwen confirms the 120B down-projection sensitivity ordering: retain protected "
                     "down islands and add a billed gain/residual correction, holding <=0.77 BPW.")
    if gain_improvement > 0.1:
        next_move += " A billed output-gain calibration is admitted because it removes material error."

    receipt: dict[str, Any] = {
        "schema": "hawking.qwen3_235b.q2_organ_diagnosis.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent_q2_sha256": q2["sha256"], "revision": Q.DEFAULT_REVISION,
        "real_layer": 0, "real_expert": expert_id, "real_token_id": token_id,
        "candidate": candidate, "class_bpw": class_bpw,
        "combined_expert_projection_bpw": round(sum(class_bpw.values()) / 3.0, 8),
        "diagnostic_metrics": metrics, "dominant_failure_organ": dominant,
        "single_organ_relative_error": singles, "next_frontier_move": next_move,
        "transport": fwd.reader.telemetry_json(),
        "claim_scope": "one real routed expert activation; diagnostic only, not capability parity",
    }
    receipt["sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic(output, receipt)
    _atomic(state, {
        "schema": "hawking.qwen3_235b.gravity_state.v1",
        "status": "Q2_DIAGNOSED_NEXT_CANDIDATE_READY", "final": False,
        "updated_at": receipt["generated_at"], "q2_receipt": str(q2_path),
        "diagnosis_receipt": str(output), "diagnosis_sha256": receipt["sha256"],
        "dominant_failure_organ": dominant, "next_frontier_move": next_move,
        "target_whole_artifact_bpw_ceiling": C.TARGET_WHOLE_BPW,
    })
    print(json.dumps({"dominant_failure_organ": dominant, "metrics": metrics,
                      "next_frontier_move": next_move, "transport": receipt["transport"]}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--q2", type=Path, default=GF / "QWEN3_235B_Q2_RECEIPT.json")
    ap.add_argument("--output", type=Path, default=GF / "QWEN3_235B_Q2_ORGAN_DIAGNOSIS.json")
    ap.add_argument("--state", type=Path, default=GF / "QWEN3_235B_GRAVITY_STATE.json")
    ap.add_argument("--candidate", default="T3_qwen_organ_alloc")
    args = ap.parse_args(argv)
    return run(args.meta_dir, args.cache_dir, args.q2, args.output, args.state, args.candidate)


if __name__ == "__main__":
    raise SystemExit(main())
