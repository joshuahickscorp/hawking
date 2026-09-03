"""doctor6 treat — execute a prescription, re-measure, seal under the ceiling."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
    collect_expert_activations,
    holdout_split,
    organ_activations,
)
from lab.operators.doctor6.billing import project_complete_bpw, seal_with_ceiling
from lab.operators.doctor6.coherence import LAYER_TARGET_COS, predict_composition
from lab.operators.doctor6.prescribe import (
    SAMPLE_SEED,
    _split_seed,
    map_in_stable_order,
    resolve_workers,
)
from lab.operators.doctor6.rungs import apply_rung, quant_binary, score_vs_incumbent
from lab.operators.one_bit_ceiling import CeilingViolation
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map


def _treat_one_organ(
    org: dict[str, Any],
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    by_le: dict[tuple[int, int], Any],
    device: str,
    qat_steps: int,
    qat_lr: float,
    target_cos: float,
) -> dict[str, Any]:
    name = org["tensor_name"]
    layer, expert, component = int(org["layer"]), int(org["expert"]), org["component"]
    chain = list(org.get("chain") or ["incumbent_binary"])
    print(f"  treat {name} chain={'+'.join(chain)}", flush=True)
    W = load_tensor(model_dir, weight_map, name).astype(np.float32, copy=False)
    X_all = organ_activations(
        layer=layer,
        expert=expert,
        component=component,
        X_hidden=by_le[(layer, expert)],
        model_dir=model_dir,
        weight_map=weight_map,
    )
    X_fit, X_hold = holdout_split(X_all, seed=_split_seed(layer, expert, component))
    W_inc, _ = quant_binary(W)

    # Apply last kept rung in the chain (each rung is full transform from W).
    # Prefer the deepest non-incumbent rung.
    apply_name = chain[-1]
    if apply_name == "incumbent_binary":
        W_hat, nbytes = W_inc, int(org.get("payload_bytes") or 0)
        if nbytes <= 0:
            _, nbytes = quant_binary(W)
        meta = {"codec": "binary_g128"}
    else:
        result = apply_rung(
            apply_name,
            W,
            X_fit,
            organ_key=name,
            sensitivity=float(org.get("sensitivity") or 0.5),
            seed=0xD0C70A,
            device=device,
            qat_steps=qat_steps,
            qat_lr=float(qat_lr),
            prefer_budget=True,
        )
        W_hat, nbytes, meta = result.W_hat, result.payload_bytes, result.meta

    score = score_vs_incumbent(
        W=W, X_hold=X_hold, W_hat=W_hat, W_incumbent=W_inc
    )
    return {
        "tensor_name": name,
        "layer": layer,
        "expert": expert,
        "component": component,
        "band": org.get("band"),
        "chain": chain,
        "applied_rung": apply_name,
        "predicted_cosine": org.get("prescribed_cosine"),
        "actual_cosine": score["output_cosine"],
        "incumbent_cosine": score["incumbent_cosine"],
        "surplus_over_incumbent": score["surplus_over_incumbent"],
        "payload_bytes": int(nbytes),
        "component_bpw": 8.0 * nbytes / max(W.size, 1),
        "meta": meta,
        "clears_target": bool(score["output_cosine"] >= target_cos),
    }


def treat(
    *,
    prescription_path: Path,
    model_dir: Path | None = None,
    capture: Path | None = None,
    device: str = "cpu",
    qat_steps: int = 200,
    qat_lr: float | None = None,
    out_path: Path | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    prescription_path = Path(prescription_path)
    rx = json.loads(prescription_path.read_text())
    if rx.get("status") == "REFUSED":
        return {
            "schema": "hawking.doctor6.treatment.v1",
            "status": "REFUSED",
            "reason": "prescription was REFUSED; treat will not execute",
            "prescription": str(prescription_path),
            "refusal": rx.get("refusal"),
        }

    model_dir = Path(model_dir or rx["inputs"]["model_dir"])
    capture_run = Path(capture or rx["inputs"]["capture"])
    target_bpw = float(rx["objective"]["target_bpw"])
    target_cos = float(rx["objective"].get("target_organ_cos", LAYER_TARGET_COS))
    if qat_lr is None:
        qat_lr = float(rx.get("inputs", {}).get("qat_lr", 1e-3))
    escape_receipt = (rx.get("ceiling") or {}).get("escape_receipt")

    print(f"[treat] loading capture {capture_run}", flush=True)
    wanted_keys = {
        (int(o["layer"]), int(o["expert"]))
        for o in (rx.get("organs") or [])
    }
    inputs = rx.get("inputs") or {}
    sample = rx.get("sample") or {}
    max_rows = inputs.get("max_rows_per_expert")
    if max_rows is None:
        max_rows = sample.get("max_rows_per_expert")
    row_seed = int(inputs.get("row_sample_seed") or sample.get("row_sample_seed") or SAMPLE_SEED)
    by_le, _ = collect_expert_activations(
        capture_run,
        wanted_keys=wanted_keys or None,
        max_rows_per_expert=int(max_rows) if max_rows else None,
        row_sample_seed=row_seed,
    )
    weight_map = load_weight_map(model_dir)
    organs = list(rx.get("organs") or [])
    workers_n = resolve_workers(workers, n_items=len(organs) or 1)
    print(f"[treat] {len(organs)} organs (workers={workers_n})", flush=True)

    def _one(i: int, org: dict[str, Any]) -> dict[str, Any]:
        del i
        return _treat_one_organ(
            org,
            model_dir=model_dir,
            weight_map=weight_map,
            by_le=by_le,
            device=device,
            qat_steps=qat_steps,
            qat_lr=float(qat_lr),
            target_cos=target_cos,
        )

    rows_out = map_in_stable_order(_one, organs, workers=workers_n)

    mean_payload = float(np.mean([r["payload_bytes"] for r in rows_out])) if rows_out else 0.0
    bill = project_complete_bpw(mean_expert_payload_bytes=mean_payload)
    try:
        ceiling = seal_with_ceiling(
            bill,
            target_bpw=target_bpw,
            note="doctor6.treat",
            escape_receipt=escape_receipt,
        )
        sealed = True
    except CeilingViolation as exc:
        ceiling = {"enforcer_called": True, "legal": False, "error": str(exc)}
        sealed = False

    coherence = predict_composition(
        [
            {
                "layer": r["layer"],
                "component": r["component"],
                "output_cosine": r["actual_cosine"],
            }
            for r in rows_out
        ]
    )

    status = "SEALED" if sealed and coherence.pass_gate else (
        "REFUSED_CEILING" if not sealed else "REFUSED_COHERENCE"
    )
    report = {
        "schema": "hawking.doctor6.treatment.v1",
        "status": status,
        "prescription": str(prescription_path),
        "elapsed_seconds": time.perf_counter() - t0,
        "billing": bill,
        "ceiling": ceiling,
        "coherence_screen": coherence.as_dict(),
        "predicted_vs_actual": [
            {
                "tensor_name": r["tensor_name"],
                "layer": r["layer"],
                "component": r["component"],
                "predicted_cosine": r["predicted_cosine"],
                "actual_cosine": r["actual_cosine"],
                "delta": (
                    None
                    if r["predicted_cosine"] is None
                    else float(r["actual_cosine"] - r["predicted_cosine"])
                ),
                "chain": r["chain"],
            }
            for r in rows_out
        ],
        "organs": rows_out,
        "claim_boundary": {
            "no_full_model_repack": True,
            "sample_only": True,
            "sealed_means_ceiling_and_coherence_passed_on_sample": True,
        },
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        print(f"[treat] wrote {out_path}", flush=True)
    return report
