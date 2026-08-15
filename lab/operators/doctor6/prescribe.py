"""doctor6 prescribe — diagnose, compose, predict, refuse or emit."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
    COMPONENTS,
    HOLD_FRAC,
    SEED,
    collect_expert_activations,
    holdout_split,
    organ_activations,
)
from lab.operators.doctor6.atomic import enforce_atomic_chains
from lab.operators.doctor6.billing import (
    maybe_run_gravity_allocator,
    project_complete_bpw,
    seal_with_ceiling,
)
from lab.operators.doctor6.capture_gate import (
    DEFAULT_MIN_ROWS_PER_ORGAN,
    check_never_routed_census,
    check_per_organ_rows,
)
from lab.operators.doctor6.coherence import (
    COMPOSITION_FLOOR,
    LAYER_TARGET_COS,
    N_LAYERS,
    predict_composition,
)
from lab.operators.doctor6.compose import compose_organ_chain, compose_status_summary
from lab.operators.doctor6.rungs import quant_binary, list_rung_status
from lab.operators.mixed_precision_alloc import BPW, allocate_from_holdout
from lab.operators.one_bit_ceiling import CeilingViolation
from lab.operators.q30_activation_aware_family_probe import load_capture
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map

MAIN_RECORDS = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records"
)
DEFAULT_MODEL = MAIN_RECORDS / "runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_CAPTURE = (
    MAIN_RECORDS
    / "ascension-sandbox/physical/qwen30/quality-diagnostics"
    / "source-bf16-capture-v1/full-run"
)
SAMPLE_SEED = 0xD0C70A
BANDS = ((0, 11), (12, 23), (24, 35), (36, 47))
LE_PER_BAND = 4


@dataclass
class OrganRef:
    layer: int
    expert: int
    component: str
    band: int
    n_routed: int

    @property
    def name(self) -> str:
        return (
            f"model.layers.{self.layer}.mlp.experts.{self.expert}."
            f"{self.component}.weight"
        )


def _comp_seed(component: str) -> int:
    return int(hashlib.sha256(component.encode()).hexdigest()[:8], 16)


def _split_seed(layer: int, expert: int, component: str) -> int:
    return SEED ^ (layer * 9176) ^ (expert * 1009) ^ (_comp_seed(component) & 0xFFFF)


def deterministic_sample(
    by_layer_expert: dict[tuple[int, int], np.ndarray],
    *,
    le_per_band: int = LE_PER_BAND,
    min_tokens: int = DEFAULT_MIN_ROWS_PER_ORGAN,
) -> list[OrganRef]:
    rng = np.random.default_rng(SAMPLE_SEED)
    organs: list[OrganRef] = []
    for band_i, (lo, hi) in enumerate(BANDS):
        eligible = [
            (L, e, int(X.shape[0]))
            for (L, e), X in by_layer_expert.items()
            if lo <= L <= hi and X.shape[0] >= min_tokens
        ]
        eligible.sort(key=lambda t: (-t[2], t[0], t[1]))
        pool = eligible[: max(le_per_band * 8, le_per_band)]
        if len(pool) < le_per_band:
            raise RuntimeError(
                f"band {band_i} ({lo}-{hi}) has only {len(pool)} eligible LE pairs "
                f"(need {le_per_band} with >= {min_tokens} rows)"
            )
        idx = sorted(int(i) for i in rng.choice(len(pool), size=le_per_band, replace=False))
        for i in idx:
            L, e, n = pool[i]
            for component in COMPONENTS:
                organs.append(
                    OrganRef(
                        layer=int(L),
                        expert=int(e),
                        component=component,
                        band=band_i,
                        n_routed=int(n),
                    )
                )
    return organs


def _jsonable_organ(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k not in ("W_hat", "W_incumbent")}
    return out


def prescribe(
    *,
    model_id: str = "qwen3-coder-30b-a3b",
    model_dir: Path | None = None,
    capture: Path | None = None,
    target_bpw: float = 1.0,
    target_cos: float = LAYER_TARGET_COS,
    le_per_band: int = LE_PER_BAND,
    min_rows: int = DEFAULT_MIN_ROWS_PER_ORGAN,
    device: str = "cpu",
    qat_steps: int = 200,
    qat_lr: float = 1e-3,
    out_path: Path | None = None,
    force_over_ceiling: bool = False,
) -> dict[str, Any]:
    """Run the full prescribe pipeline and return a machine-readable prescription."""
    t0 = time.perf_counter()
    model_dir = Path(model_dir or DEFAULT_MODEL)
    capture_run = Path(capture or DEFAULT_CAPTURE)

    # --- load capture ---
    print(f"[prescribe] loading capture {capture_run}", flush=True)
    cap = load_capture(capture_run)
    by_le, act_prov = collect_expert_activations(capture_run, cap)

    # Capture starvation gate on all LE pairs that will be considered.
    row_counts = {
        f"L{L}_E{e}": int(X.shape[0]) for (L, e), X in by_le.items()
    }
    # Gate on the *sampled* organs' LE pairs after sampling eligibility uses min_rows.
    # First refuse if the capture overall cannot support the floor for any LE.
    if not by_le:
        return {
            "schema": "hawking.doctor6.prescription.v1",
            "status": "REFUSED",
            "refusal": {
                "kind": "empty_capture",
                "reason": "capture has no per-expert activations",
            },
        }

    # Census token count if available.
    n_tokens_census = int(act_prov.get("n_tokens", act_prov.get("total_tokens", 0)) or 0)
    if n_tokens_census == 0:
        # approximate from unique rows sum (upper bound not exact)
        n_tokens_census = int(sum(int(X.shape[0]) for X in by_le.values()))
    census_gate = check_never_routed_census(n_tokens_census)

    organs = deterministic_sample(by_le, le_per_band=le_per_band, min_tokens=min_rows)
    sample_row_counts = {o.name: o.n_routed for o in organs}
    cap_gate = check_per_organ_rows(sample_row_counts, floor=min_rows)
    if not cap_gate.ok:
        return {
            "schema": "hawking.doctor6.prescription.v1",
            "status": "REFUSED",
            "refusal": {
                "kind": "capture_starvation",
                "reason": cap_gate.reason,
                "gate": cap_gate.as_dict(),
            },
            "elapsed_seconds": time.perf_counter() - t0,
        }

    weight_map = load_weight_map(model_dir)
    print(f"[prescribe] sample {len(organs)} organs; composing per-organ chains", flush=True)

    # Sensitivity from incumbent binary fit cosine (train-free).
    prepared: list[dict[str, Any]] = []
    for org in organs:
        W = load_tensor(model_dir, weight_map, org.name).astype(np.float32, copy=False)
        X_all = organ_activations(
            layer=org.layer,
            expert=org.expert,
            component=org.component,
            X_hidden=by_le[(org.layer, org.expert)],
            model_dir=model_dir,
            weight_map=weight_map,
        )
        X_fit, X_hold = holdout_split(
            X_all, seed=_split_seed(org.layer, org.expert, org.component)
        )
        rec, _ = quant_binary(W)
        from lab.operators.ascension_dual_gravity_worker import _mean_row_cosine

        cos_fit = float(_mean_row_cosine(X_fit @ W.T, X_fit @ rec.T))
        prepared.append(
            {
                "org": org,
                "W": W,
                "X_fit": X_fit,
                "X_hold": X_hold,
                "sens_raw": 1.0 - cos_fit,
            }
        )
    sens_raw = [p["sens_raw"] for p in prepared]
    order = np.argsort(sens_raw)
    sens_rank = np.zeros(len(sens_raw), dtype=np.float64)
    for rank_i, idx in enumerate(order):
        sens_rank[idx] = rank_i / max(len(sens_raw) - 1, 1)

    organ_results: list[dict[str, Any]] = []
    for i, p in enumerate(prepared):
        org: OrganRef = p["org"]
        print(
            f"  [{i+1}/{len(prepared)}] {org.name} sens={sens_rank[i]:.2f}",
            flush=True,
        )
        result = compose_organ_chain(
            W=p["W"],
            X_fit=p["X_fit"],
            X_hold=p["X_hold"],
            organ_key=org.name,
            component=org.component,
            sensitivity=float(sens_rank[i]),
            seed=SAMPLE_SEED,
            device=device,
            qat_steps=qat_steps,
            qat_lr=qat_lr,
            target_cos=target_cos,
            measure_all=False,
        )
        result.update(
            {
                "tensor_name": org.name,
                "layer": org.layer,
                "expert": org.expert,
                "component": org.component,
                "band": org.band,
                "n_routed": org.n_routed,
                "n_params": int(p["W"].size),
                "n_fit": int(p["X_fit"].shape[0]),
                "n_hold": int(p["X_hold"].shape[0]),
            }
        )
        organ_results.append(result)
        print(
            f"    chain={'+'.join(result['chain'])} "
            f"cos={result['prescribed_cosine']:.4f} "
            f"(inc={result['incumbent_cosine']:.4f}) "
            f"dropped={len(result['dropped_rungs'])}",
            flush=True,
        )

    # Atomic triple enforcement (last stage that can create partials).
    atomic = enforce_atomic_chains(organ_results)

    # Re-score is not re-run after chain union (would re-measure); record that
    # atomic union may have extended some chains for seal admissibility.
    # Predicted fidelity uses the per-organ prescribed cosines (pre-union measure).
    organ_cos_rows = [
        {
            "layer": r["layer"],
            "expert": r["expert"],
            "component": r["component"],
            "output_cosine": r["prescribed_cosine"],
        }
        for r in organ_results
    ]
    coherence = predict_composition(organ_cos_rows, n_layers=N_LAYERS)

    mean_payload = float(np.mean([r["payload_bytes"] for r in organ_results]))
    bill = project_complete_bpw(mean_expert_payload_bytes=mean_payload)
    if force_over_ceiling:
        # Acceptance: over-ceiling prescription must fail closed.
        bill = project_complete_bpw(
            mean_expert_payload_bytes=mean_payload * 50.0,
            doctor_bytes=10**9,
        )

    # Per-organ mixed-precision water-fill. Always invoked (not gated on the
    # complete-artifact ceiling binding). Sensitivity is activation-holdout
    # output cosine — never the weight-space grid_quant relRMS proxy.
    waterfill = allocate_from_holdout(
        [
            {
                "name": r["tensor_name"],
                "elems": int(r.get("n_params") or 0) or 1,
                "holdout_cosine": float(r["prescribed_cosine"]),
                "component": r["component"],
                "layer": r["layer"],
                "band": r.get("band"),
            }
            for r in organ_results
        ],
        bits_set=(1, 2, 3, 4),
        target_bpw=float(target_bpw),
        layer_target=float(target_cos),
    )
    for r in organ_results:
        bits = int(waterfill["allocation"].get(r["tensor_name"], 1))
        r["allocated_bits"] = bits
        r["allocated_eff_bpw"] = float(BPW.get(bits, BPW[1]))
    gravity_alloc = maybe_run_gravity_allocator(
        bill,
        target_bpw=target_bpw,
        organ_sensitivities={
            "expert_down": 0.75,
            "expert_gate": 0.45,
            "expert_up": 0.50,
        },
    )
    alloc_report = dict(waterfill)
    alloc_report["gravity_allocator_gated"] = gravity_alloc

    ceiling_report: dict[str, Any]
    try:
        ceiling_report = seal_with_ceiling(bill, target_bpw=target_bpw, note="doctor6.prescribe")
        ceiling_ok = True
    except CeilingViolation as exc:
        ceiling_report = {
            "enforcer_called": True,
            "legal": False,
            "error": str(exc),
            "escape_receipt": None,
            "escape_applied": False,
        }
        ceiling_ok = False

    compose_summary = compose_status_summary(organ_results)

    # Pareto front: (chain-effort, bpw, cosine) across organs' measurements.
    pareto = _pareto_front(organ_results)

    # Named deficits
    deficits: list[dict[str, Any]] = []
    if not coherence.pass_gate:
        deficits.append(
            {
                "kind": "coherence_composition",
                "detail": coherence.refusal_reason,
                "per_band_deficit": coherence.per_band_deficit,
            }
        )
    if not ceiling_ok:
        deficits.append(
            {
                "kind": "ceiling_violation",
                "detail": ceiling_report.get("error"),
            }
        )
    short_organs = [r for r in organ_results if not r["clears_target"]]
    if short_organs:
        deficits.append(
            {
                "kind": "organ_fidelity_shortfall",
                "n_short": len(short_organs),
                "mean_deficit": float(
                    np.mean([r["deficit_vs_target"] for r in short_organs])
                ),
                "by_component": {
                    c: float(
                        np.mean(
                            [
                                r["prescribed_cosine"]
                                for r in organ_results
                                if r["component"] == c
                            ]
                        )
                    )
                    for c in COMPONENTS
                },
                "by_band": {
                    f"band_{b}": float(
                        np.mean(
                            [
                                r["prescribed_cosine"]
                                for r in organ_results
                                if r["band"] == b
                            ]
                        )
                    )
                    for b in range(4)
                },
            }
        )
    if not atomic["seal_admissible"]:
        deficits.append(
            {
                "kind": "partial_expert_triples",
                "n_partial": atomic["n_partial"],
                "partials": atomic["partials"],
            }
        )

    # Refuse if coherence fails OR ceiling fails. Organ shortfall is reported but
    # prescription still emitted with status DEGRADED so treat can attempt recovery;
    # hard refuse only on coherence/ceiling/atomicity that would waste 77 minutes.
    hard_refuse = (not coherence.pass_gate) or (not ceiling_ok)
    if hard_refuse:
        status = "REFUSED"
    elif deficits:
        status = "DEGRADED"
    else:
        status = "OK"

    prescription = {
        "schema": "hawking.doctor6.prescription.v1",
        "status": status,
        "model_id": model_id,
        "recorded_at_unix": time.time(),
        "elapsed_seconds": time.perf_counter() - t0,
        "objective": {
            "target_bpw": float(target_bpw),
            "target_organ_cos": float(target_cos),
            "composition_floor": COMPOSITION_FLOOR,
            "n_layers": N_LAYERS,
            "layer_target_cos": LAYER_TARGET_COS,
        },
        "inputs": {
            "model_dir": str(model_dir),
            "capture": str(capture_run),
            "device": device,
            "qat_steps": qat_steps,
            "qat_lr": qat_lr,
            "min_rows_floor": min_rows,
            "holdout": {"HOLD_FRAC": HOLD_FRAC, "SEED": SEED},
            "activation_provenance": {
                k: act_prov[k]
                for k in act_prov
                if k not in ("hit_counts",) and _json_safe(act_prov[k])
            },
        },
        "capture_gate": cap_gate.as_dict(),
        "never_routed_census_gate": census_gate,
        "sample": {
            "n_organs": len(organs),
            "le_per_band": le_per_band,
            "bands": [list(b) for b in BANDS],
            "components": list(COMPONENTS),
            "organs": [
                {
                    "tensor_name": o.name,
                    "layer": o.layer,
                    "expert": o.expert,
                    "component": o.component,
                    "band": o.band,
                    "n_routed": o.n_routed,
                }
                for o in organs
            ],
        },
        "diagnose": {
            "mean_incumbent_cosine": compose_summary["mean_incumbent_cosine"],
            "mean_prescribed_cosine": compose_summary["mean_prescribed_cosine"],
            "n_clear_target": compose_summary["n_clear_target"],
            "by_component_prescribed": {
                c: float(
                    np.mean(
                        [r["prescribed_cosine"] for r in organ_results if r["component"] == c]
                    )
                )
                for c in COMPONENTS
            },
            "by_band_prescribed": {
                f"band_{b}": float(
                    np.mean(
                        [r["prescribed_cosine"] for r in organ_results if r["band"] == b]
                    )
                )
                for b in range(4)
            },
        },
        "compose": compose_summary,
        "atomic_triples": atomic,
        "coherence_screen": coherence.as_dict(),
        "billing": bill,
        "ceiling": ceiling_report,
        "allocator": alloc_report,
        "pareto_front": pareto,
        "rung_entrypoints": list_rung_status(),
        "deficits": deficits,
        "organs": [_jsonable_organ(r) for r in organ_results],
        "predicted_vs_actual": [
            {
                "tensor_name": r["tensor_name"],
                "layer": r["layer"],
                "component": r["component"],
                "band": r["band"],
                "predicted_cosine": r["prescribed_cosine"],
                "actual_cosine": r["prescribed_cosine"],
                "note": (
                    "prescribe measures holdout cosine of the chosen chain; "
                    "treat re-measures after execution (same number if deterministic)"
                ),
                "incumbent_cosine": r["incumbent_cosine"],
                "chain": r["chain"],
                "allocated_bits": r.get("allocated_bits"),
                "allocated_eff_bpw": r.get("allocated_eff_bpw"),
            }
            for r in organ_results
        ],
        "claim_boundary": {
            "organ_sample_not_full_model": True,
            "coherence_screen_is_necessary_not_sufficient": True,
            "no_full_model_repack": True,
            "scoring_vs_incumbent_not_null": True,
            "complete_bpw_projected_from_sample_mean": True,
        },
    }

    if hard_refuse:
        prescription["refusal"] = {
            "kind": "hard_gate",
            "reasons": [d["kind"] for d in deficits if d["kind"] in (
                "coherence_composition",
                "ceiling_violation",
            )],
            "detail": deficits,
        }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(prescription, indent=2, sort_keys=True, default=str) + "\n")
        tmp.replace(out_path)
        print(f"[prescribe] wrote {out_path}", flush=True)

    return prescription


def _json_safe(v: Any) -> bool:
    try:
        json.dumps(v)
        return True
    except Exception:
        return False


def _pareto_front(organ_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Joint (chain_len, bpw, cosine) points — cheaper may also be better."""
    pts = []
    for r in organ_results:
        for m in r.get("measurements") or []:
            if m.get("rung") == "incumbent_binary":
                continue
            pts.append(
                {
                    "organ": r["tensor_name"],
                    "rung": m["rung"],
                    "chain_depth": RUNG_DEPTH.get(m["rung"], 0),
                    "component_bpw": m.get("component_bpw"),
                    "output_cosine": m.get("output_cosine"),
                    "kept": m.get("kept"),
                }
            )
    # Keep only kept points and filter dominated (higher bpw, lower cos).
    kept = [p for p in pts if p.get("kept")]
    front = []
    for p in kept:
        dominated = False
        for q in kept:
            if q is p:
                continue
            if (
                (q["component_bpw"] or 0) <= (p["component_bpw"] or 0)
                and (q["output_cosine"] or 0) >= (p["output_cosine"] or 0)
                and (
                    (q["component_bpw"] or 0) < (p["component_bpw"] or 0)
                    or (q["output_cosine"] or 0) > (p["output_cosine"] or 0)
                )
            ):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda p: (p["component_bpw"] or 0, -(p["output_cosine"] or 0)))
    return front[:64]


RUNG_DEPTH = {
    "l0_calib": 1,
    "l1_awq": 2,
    "l1_rotation": 3,
    "l2_mixed_prec": 4,
    "l3_outlier_residual": 5,
    "l4_block_qat": 6,
}
