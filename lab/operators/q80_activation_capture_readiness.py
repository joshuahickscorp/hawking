#!/usr/bin/env python3
"""Q80 all-layer activation capture readiness and design authority.

Reports the cheapest honest path to real per-layer routed activations for
surplus-first gravity packing. Does not start a server, take a Metal lease,
or fit families. Negative results are first-class.

Campaign knowledge applied (not rediscovered):
- Select on surplus-over-null, not weight cosine.
- Report null first; the null is the instrument.
- Coverage is mandatory; L0-only cannot be coherent.
- complete_physical_bpw <= 1.5; current Q80 artifact ~1.133 LOW_FIDELITY.
- Direct execution: no dense expand at token time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lab.receipts import seal

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_HAWKING = Path("/Users/scammermike/Downloads/hawking")

DEFAULT_OUT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen80"
    / "quality-diagnostics/all-layer-activation-v1"
)

# Durable main-tree authorities (worktrees often omit multi-GB artifacts).
MAIN_Q80 = MAIN_HAWKING / "workspace/campaign/records/ascension-sandbox/physical/qwen80"
MAIN_ACQ = MAIN_HAWKING / "workspace/campaign/records/ascension-sandbox/physical/qwen80-acquisition"

MANIFEST = MAIN_Q80 / "complete-gravity/QWEN80_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
ADMISSION = MAIN_Q80 / "complete-gravity/QWEN80_COMPLETE_BINARY_GRAVITY_ADMISSION_CURRENT.json"
SOURCE_AUDIT = MAIN_ACQ / "QWEN80_SOURCE_BODY_AUDIT_CANDIDATE.json"
GQA_GAP = MAIN_Q80 / "complete-runtime/QWEN80_MULTI_LAYER_GQA_ENCODE_GAP_20260809T210000Z.json"
SCHEDULE = MAIN_Q80 / "complete-runtime/QWEN80_48_LAYER_EXECUTION_SCHEDULE_AUTHORITY_20260809T192559Z.json"
MULTI_LAYER_CAPTURE = (
    MAIN_Q80
    / "complete-runtime"
    / "QWEN80_MULTI_LAYER_fe88c4c3-dbcf-4a11-bd18-3d0211a53224_outer"
    / "inner"
    / "receipt.json"
)
L0_L1_CAPTURE = (
    MAIN_Q80
    / "complete-runtime"
    / "QWEN80_L0_L1_STRICT_HOST_OUTER_CAPTURE_20260809T115059Z"
    / "inner"
    / "receipt.json"
)

SCHEMA = "hawking.ascension.qwen80_all_layer_activation_capture_readiness.v1"
DESIGN_SCHEMA = "hawking.ascension.qwen80_all_layer_activation_capture_design.v1"

# Architecture from sealed config / schedule (measured, not assumed).
QWEN80_LAYERS = 48
QWEN80_HIDDEN = 2048
QWEN80_EXPERTS = 512
QWEN80_TOP_K = 10
GQA_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
DELTANET_PREFIX_READY = 3  # L0..L2 same-runtime Metal encode ready


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _bind(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"missing authority: {path}")
        return {"path": str(path), "present": False}
    doc = _read_json(path) or {}
    return {
        "path": str(path.resolve()),
        "present": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "document_seal_sha256": doc.get("seal_sha256"),
        "schema": doc.get("schema"),
        "status": doc.get("status"),
    }


def assess() -> dict[str, Any]:
    """Build the readiness assessment from sealed authorities on disk."""

    manifest_bind = _bind(MANIFEST)
    admission_bind = _bind(ADMISSION)
    audit_bind = _bind(SOURCE_AUDIT)
    gqa_bind = _bind(GQA_GAP)
    schedule_bind = _bind(SCHEDULE)
    multi_bind = _bind(MULTI_LAYER_CAPTURE, required=False)
    l0l1_bind = _bind(L0_L1_CAPTURE, required=False)

    manifest = _read_json(MANIFEST) or {}
    schedule = _read_json(SCHEDULE) or {}
    gqa = _read_json(GQA_GAP) or {}
    multi = _read_json(MULTI_LAYER_CAPTURE) or {}
    l0l1 = _read_json(L0_L1_CAPTURE) or {}

    ledger = manifest.get("complete_physical_bpw_ledger") or {}
    complete_bpw = ledger.get("complete_physical_bpw")
    tensor_count = len(manifest.get("tensors") or [])
    aggregate = schedule.get("aggregate") or {}

    gqa_ready = int(aggregate.get("same_runtime_gqa_encode_ready_layer_count") or 0)
    deltanet_ready = int(
        aggregate.get("same_runtime_deltanet_encode_ready_layer_count") or 0
    )
    multi_layers = (
        (multi.get("fresh_same_runtime_execution") or {}).get("layer_count")
        if multi
        else None
    )

    # Existing captures are component parity, not broad activation rows.
    existing_captures_usable_for_fit = False
    existing_reason = (
        "L0/L1 and multi-layer captures are single-token component parity receipts "
        "(second residual sha + max_abs_error). They do not retain per-token "
        "router-input f32 hiddens or multi-prompt route membership needed for "
        "null/surplus scoring and activation_weighted_svd fit."
    )

    options = [
        {
            "id": "metal_multi_layer_all_48",
            "name": "Metal multi-layer same-runtime all 48 layers",
            "cost_rank": 1,
            "honest": True,
            "ready_now": False,
            "covers_layers": list(range(QWEN80_LAYERS)) if gqa_ready == 12 else list(range(DELTANET_PREFIX_READY)),
            "blocked_by": [
                "same-runtime full-layer GQA encode for layers 3,7,...,47 "
                f"(ready_count={gqa_ready}/12)",
                "broad multi-probe activation capture binary (does not exist for Q80)",
                "owner-run Metal capture under gate profile / serialized lease",
            ],
            "once_ready": (
                "Extend multi-layer host to layer_count=48, add stratified "
                "hidden subsample writer (Q30 pattern), run broad prompt set."
            ),
            "matches_q30_pattern": True,
        },
        {
            "id": "metal_deltanet_prefix_only_l0_l2",
            "name": "Metal multi-layer DeltaNet prefix L0..L2 only",
            "cost_rank": 2,
            "honest": True,
            "ready_now": True,
            "covers_layers": list(range(DELTANET_PREFIX_READY)),
            "blocked_by": [],
            "once_ready": (
                "Could measure per-layer nulls on 3/48 layers after a broad "
                "multi-token capture binary is written. Cannot pack for "
                "coherence (coverage failure class of the Q30 L0-only mistake)."
            ),
            "coherence_eligible": False,
            "coverage_percent_upper_bound": round(100.0 * DELTANET_PREFIX_READY / QWEN80_LAYERS, 2),
            "matches_q30_pattern": False,
            "note": "Ready for instrument calibration only; refuse as packing input.",
        },
        {
            "id": "cpu_packed_hybrid_all_48",
            "name": "CPU packed hybrid sequential forward (admitted artifact)",
            "cost_rank": 3,
            "honest": True,
            "ready_now": False,
            "covers_layers": list(range(QWEN80_LAYERS)),
            "blocked_by": [
                "full-layer GQA CPU oracle integrated into sequential multi-token chain "
                "(isolated two-token GQA CPU oracle exists; multi-layer encode wires DeltaNet only)",
                "multi-token sequential DeltaNet+GQA state across 48 layers for capture",
                "broad capture writer emitting stratified router-input hiddens",
            ],
            "once_ready": (
                "No Metal lease required. Bound by admitted compact tensors only "
                "(no dense expand). Still multi-hour for ~4k tokens on CPU."
            ),
            "matches_q30_pattern": False,
        },
        {
            "id": "source_bf16_streamed_teacher",
            "name": "Streamed BF16 source teacher forward",
            "cost_rank": 4,
            "honest": True,
            "ready_now": False,
            "covers_layers": list(range(QWEN80_LAYERS)),
            "blocked_by": [
                "no Q80 streamed source teacher / layer-streamed BF16 oracle path exists",
                "full Qwen3-Next hybrid (DeltaNet+GQA+MoE) source forward must be built",
                "148G source residency / disk-offload schedule",
            ],
            "once_ready": "Authority-teacher activations; still not a packing substitute without coverage.",
            "matches_q30_pattern": False,
        },
        {
            "id": "reuse_l0_l1_component_captures",
            "name": "Reuse existing L0/L1/multi-layer component captures",
            "cost_rank": 0,
            "honest": False,
            "ready_now": True,
            "covers_layers": [],
            "blocked_by": [existing_reason],
            "coherence_eligible": False,
            "refused_reason": (
                "Would repeat the Q30 coverage failure (and worse: single-token "
                "component parity has no fit matrix at all)."
            ),
        },
    ]

    missing_exact = [
        {
            "id": "gqa_full_layer_same_runtime_encode",
            "description": gqa.get("exact_missing_input")
            or (
                "same-runtime full-layer encode for the GQA mixer, with caller-owned "
                "gqa_key_cache/gqa_value_cache state slots and rollback buffers"
            ),
            "authority": gqa_bind,
            "blocks": ["metal_multi_layer_all_48", "any complete 48-layer device chain"],
            "status": gqa.get("status")
            or "EARNED_NEGATIVE_GQA_FULL_LAYER_SAME_RUNTIME_ENCODE_ABSENT",
            "gqa_layers": list(GQA_LAYERS),
            "ready_layer_count": gqa_ready,
        },
        {
            "id": "broad_activation_capture_binary",
            "description": (
                "Q80 analogue of ascension_qwen30_broad_activation_all_layer_route_capture: "
                "multi-probe sequential forward that writes full route membership for every "
                "token at every layer and stratified raw f32 router-input hiddens "
                f"(default 1024 tokens/layer × {QWEN80_HIDDEN} f32)."
            ),
            "blocks": ["null-first scoring", "surplus-first fit", "coverage-honest repack"],
            "status": "ABSENT",
            "q30_reference": (
                "crates/hawking-core/examples/"
                "ascension_qwen30_broad_activation_all_layer_route_capture.rs"
            ),
        },
        {
            "id": "multi_token_sequential_state_for_capture",
            "description": (
                "Existing multi-layer same-runtime path is a single source-token "
                "component parity capture. Broad activation needs sequential state "
                "across prompt tokens (DeltaNet recurrent + GQA KV) for each probe."
            ),
            "blocks": ["any multi-token activation capture on the current component path"],
            "status": "ABSENT_ON_COMPONENT_PATH",
        },
    ]

    cheapest_honest = next(
        (
            o
            for o in sorted(options, key=lambda r: r["cost_rank"])
            if o.get("honest") and o.get("ready_now") and o.get("coherence_eligible") is not False
        ),
        None,
    )
    # Prefer coherence-eligible ready path; none exists.
    if cheapest_honest is None:
        cheapest_honest_now = next(
            (o for o in sorted(options, key=lambda r: r["cost_rank"]) if o.get("honest") and o.get("ready_now")),
            None,
        )
    else:
        cheapest_honest_now = cheapest_honest

    coherence_path = next(
        (o for o in sorted(options, key=lambda r: r["cost_rank"]) if o.get("honest") and o.get("id") == "metal_multi_layer_all_48"),
        options[0],
    )

    design = {
        "schema": DESIGN_SCHEMA,
        "status": "DESIGNED_NOT_EXECUTABLE_ALL_LAYER_CAPTURE_BLOCKED",
        "recorded_at": _utc_now(),
        "model": {
            "id": "Qwen3-Coder-Next-80B",
            "key": "qwen80",
            "layers": QWEN80_LAYERS,
            "hidden": QWEN80_HIDDEN,
            "num_experts": QWEN80_EXPERTS,
            "top_k": QWEN80_TOP_K,
            "gqa_layers": list(GQA_LAYERS),
            "deltanet_layers": QWEN80_LAYERS - len(GQA_LAYERS),
        },
        "baseline_artifact": {
            "manifest": manifest_bind,
            "admission": admission_bind,
            "source_audit": audit_bind,
            "complete_physical_bpw": complete_bpw,
            "tensor_count": tensor_count,
            "status": manifest.get("status"),
            "ceiling_bpw": 1.5,
            "under_ceiling": bool(complete_bpw is not None and float(complete_bpw) <= 1.5),
            "low_fidelity_unqualified": True,
        },
        "runtime_surface": {
            "schedule": schedule_bind,
            "gqa_encode_gap": gqa_bind,
            "same_runtime_deltanet_encode_ready_layer_count": deltanet_ready,
            "same_runtime_gqa_encode_ready_layer_count": gqa_ready,
            "device_multi_layer_earned_prefix": {
                "layers": "L0..L2",
                "layer_count": DELTANET_PREFIX_READY,
                "multi_layer_capture": multi_bind,
                "observed_layer_count_on_latest_capture": multi_layers,
                "status": multi.get("status") if multi else None,
            },
            "l0_l1_capture": l0l1_bind,
            "l0_l1_status": l0l1.get("status") if l0l1 else None,
            "existing_captures_usable_for_activation_fit": existing_captures_usable_for_fit,
            "existing_captures_reason": existing_reason,
        },
        "options": options,
        "missing_exact": missing_exact,
        "cheapest_honest_path_now": {
            "option_id": (cheapest_honest_now or {}).get("id"),
            "covers_layers": (cheapest_honest_now or {}).get("covers_layers"),
            "coherence_eligible": (cheapest_honest_now or {}).get("coherence_eligible", False),
            "note": (
                "The only ready honest path covers 3/48 DeltaNet layers after a "
                "new multi-token capture binary is written. It is instrument "
                "calibration only — packing it would repeat the Q30 coverage failure."
            ),
        },
        "cheapest_honest_path_for_coherence": {
            "option_id": coherence_path.get("id"),
            "ready_now": False,
            "primary_blocker": missing_exact[0],
            "secondary_blockers": missing_exact[1:],
            "q30_pattern": (
                "ascension_qwen30_broad_activation_all_layer_route_capture + "
                "q30_activation_null_first_report + "
                "ascension_qwen30_activation_weighted_svd_repack"
            ),
            "bounded_storage_plan": {
                "full_route_membership_all_tokens_all_layers": True,
                "raw_hidden_strategy": "stratified_token_subsample",
                "default_max_hidden_tokens_per_layer": 1024,
                "hidden_bytes_at_default": 48 * 1024 * QWEN80_HIDDEN * 4,
                "note": (
                    "Same strategy as Q30 all-layer capture: full routes (tiny) + "
                    "stratified raw hiddens so Gram/fit still has holdout rows."
                ),
            },
        },
        "selection_policy_when_capture_lands": {
            "primary": "surplus_over_null",
            "secondary": "weight_cosine",
            "weight_cosine_role": "distribution_local_guard_only",
            "family": "activation_weighted_svd_low_rank_q",
            "component_bpw_ceiling": 1.5,
            "complete_physical_bpw_ceiling": 1.5,
            "require_all_layer_capture": True,
            "refuse_layer0_only": True,
            "null_reported_first": True,
            "coverage_receipt_mandatory": True,
            "direct_execution_no_dense_expand_at_token_time": True,
        },
        "claim_boundary": {
            "all_layer_activation_capture_not_produced": True,
            "no_per_layer_nulls_measured_this_lane": True,
            "no_family_fit_or_repack_performed": True,
            "no_coherence_claim": True,
            "no_server_started": True,
            "no_exclusive_gpu_lease": True,
            "no_tps_benchmark": True,
            "negative_result_is_valid_deliverable": True,
            "fitting_on_layer0_or_component_captures_is_refused": True,
        },
        "verdict": "ALL_LAYER_ACTIVATION_CAPTURE_NOT_YET_POSSIBLE",
        "verdict_detail": (
            f"Q80 device multi-layer same-runtime currently reaches {DELTANET_PREFIX_READY} "
            f"of {QWEN80_LAYERS} layers (DeltaNet L0..L2). GQA full-layer same-runtime encode "
            f"ready_count={gqa_ready}/12. Existing L0/L1/multi-layer receipts are single-token "
            "component parity, not broad activation matrices. No Q80 streamed source teacher "
            "exists. Therefore per-layer nulls, surplus-first family tables, and a "
            "coverage-honest repack cannot be produced without inventing activations — "
            "which this lane refuses."
        ),
    }
    return design


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    design = assess()
    design_path = out_dir / "CAPTURE_DESIGN.json"
    design_path.write_text(
        json.dumps(design, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    readiness = seal(
        {
            "schema": SCHEMA,
            "status": design["verdict"],
            "recorded_at": _utc_now(),
            "design_path": str(design_path),
            "design_sha256": _sha256_file(design_path),
            "verdict": design["verdict"],
            "verdict_detail": design["verdict_detail"],
            "missing_exact_ids": [m["id"] for m in design["missing_exact"]],
            "cheapest_honest_path_now": design["cheapest_honest_path_now"],
            "cheapest_honest_path_for_coherence": {
                "option_id": design["cheapest_honest_path_for_coherence"]["option_id"],
                "ready_now": False,
                "primary_blocker_id": design["missing_exact"][0]["id"],
            },
            "baseline_complete_physical_bpw": design["baseline_artifact"]["complete_physical_bpw"],
            "claim_boundary": design["claim_boundary"],
        }
    )
    readiness_path = out_dir / "CAPTURE_READINESS.json"
    readiness_path.write_text(
        json.dumps(readiness, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    gqa_status = str((_read_json(GQA_GAP) or {}).get("status"))
    status_md = f"""# Q80 all-layer activation capture — readiness

**Verdict:** `{design["verdict"]}`

## Headline (null first, families later)

Per-layer nulls: **not measured** — no all-layer (or any multi-token) activation
matrix exists for Q80. This is not a null-trap finding; it is a **missing instrument** finding.

## Baseline artifact (admitted, low fidelity)

| field | value |
|---|---|
| complete_physical_bpw | **{design["baseline_artifact"]["complete_physical_bpw"]}** |
| ceiling | 1.5 |
| tensor_count | {design["baseline_artifact"]["tensor_count"]} |
| status | `{design["baseline_artifact"]["status"]}` |

## Why all-layer capture is blocked

1. **GQA full-layer same-runtime encode absent** (ready 0/12). Layers {list(GQA_LAYERS)} refuse at CPU preflight before lease. Authority: `{GQA_GAP.name}` status `{gqa_status}`.
2. **Device multi-layer earned prefix is L0..L2 only** (3/48 DeltaNet).
3. **Existing L0/L1/multi-layer captures are single-token component parity** — second residual sha + max_abs_error only; no router-input f32 rows for fit.
4. **No Q80 streamed BF16 source teacher** and no multi-token CPU hybrid capture chain.

## Cheapest honest path

| goal | path | ready now? |
|---|---|---|
| instrument calibration on 3 layers | Metal multi-token L0..L2 after new capture binary | path exists; binary absent |
| **coherence packing** | Metal all-48 after GQA encode + broad capture binary | **no** |

Fitting on L0-only or on component captures is **refused** (Q30 coverage failure class).

## What lands when GQA encode is ready

1. Owner runs Metal broad all-layer capture (see `RUN_CAPTURE.command.txt`).
2. `python3 -m lab.operators.q80_activation_null_first_report` — per-layer nulls **before** any family row.
3. `python3 -m lab.operators.ascension_qwen80_activation_weighted_svd_repack` — surplus-first under 1.5 BPW with coverage receipt.
4. Owner admits and serves on a **new** port; text generation required before any coherence claim.

## Claim boundary

- No server started, no exclusive GPU lease, no TPS.
- No family table invented.
- Negative result is the valid deliverable for this open question.
"""
    (out_dir / "STATUS.md").write_text(status_md, encoding="utf-8")

    print(
        json.dumps(
            {
                "verdict": design["verdict"],
                "readiness_path": str(readiness_path),
                "design_path": str(design_path),
                "missing_exact_ids": [m["id"] for m in design["missing_exact"]],
                "complete_physical_bpw": design["baseline_artifact"]["complete_physical_bpw"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
