"""Bind live causal receipts to specific KIMI candidate ids.

The locality experiment can show that a residual intervention changes a model,
but that does not prove that every registered weight candidate implements the
same operator. This module keeps candidate-specific live evidence explicit and
marks unmeasured candidates as unqualified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "receipts" / "future" / "KIMI_REFUSAL_LOCALITY_HELDOUT_MOE_L26E23_20260910.json"
CANDIDATE_LIVE_SOURCES = (
    ROOT / "receipts" / "future" / "KIMI_REFUSAL_LOCALITY_CANDIDATE_OPA_ALL_L26_20260910.json",
    ROOT / "receipts" / "future" / "KIMI_REFUSAL_LOCALITY_CANDIDATE_OPB_ALL_L26_20260910.json",
    ROOT / "receipts" / "future" / "KIMI_REFUSAL_LOCALITY_CANDIDATE_OPC_L26_20260910.json",
    ROOT / "receipts" / "future" / "KIMI_REFUSAL_LOCALITY_CANDIDATE_OPD_L26_20260910.json",
    ROOT / "receipts" / "future" / "KIMI_REFUSAL_LOCALITY_CANDIDATE_OPE_L26_20260910.json",
    ROOT / "receipts" / "future" / "KIMI_REFUSAL_LOCALITY_CANDIDATE_OPE_EXPANDED32_L26_RECLASSIFIED_20260910.json",
    ROOT / "receipts" / "future" / "KIMI_REFUSAL_LOCALITY_CANDIDATE_OPF_EXPANDED32_L26E23_20260910.json",
    ROOT / "receipts" / "future" / "KIMI_REFUSAL_LOCALITY_HELDOUT_MOE_L26E23_SHARED_20260910.json",
    ROOT / "receipts" / "future" / "KIMI_REFUSAL_LOCALITY_CANDIDATE_OPG_EXPANDED32_L26E23_SHARED_20260910.json",
    ROOT / "receipts" / "future" / "KIMI_REFUSAL_LOCALITY_CANDIDATE_OPH_ROUTER_RUN_FAILURE_20260910.json",
    ROOT / "receipts" / "future" / "KIMI_REFUSAL_LOCALITY_CANDIDATE_OPH_DEVICE_BUSY_20260910.json",
    ROOT / "receipts" / "future" / "KIMI_REFUSAL_LOCALITY_CANDIDATE_OPH_ROUTER_LIVE_20260910.json",
)
CANDIDATE_HCLI_SOURCE = ROOT / "receipts" / "future" / "KIMI_CANDIDATE_OPE_HCLI_LIVE_20260910.json"
EXPERIMENTAL_HCLI_SOURCE = ROOT / "receipts" / "future" / "KIMI_EXPERIMENTAL_HCLI_FIT_WRITER_LIVE_20260910.json"
EXPERIMENTAL_HCLI_NESTED_SOURCE = ROOT / "receipts" / "future" / "KIMI_EXPERIMENTAL_HCLI_FIT_WRITER_NESTED_LIVE_20260910.json"
EXPERIMENTAL_HCLI_RESIDUAL_SOURCE = ROOT / "receipts" / "future" / "KIMI_EXPERIMENTAL_HCLI_FIT_RESIDUAL_LIVE_20260910.json"
EXPERIMENTAL_HCLI_TRANSLATION_SOURCE = ROOT / "receipts" / "future" / "KIMI_EXPERIMENTAL_HCLI_FIT_TRANSLATION_NESTED_LIVE_20260910.json"
EXPERIMENTAL_HCLI_PREFILL_SOURCE = ROOT / "receipts" / "future" / "KIMI_EXPERIMENTAL_HCLI_FIT_PREFILL_TRANSLATION_LIVE_20260910.json"
EXPERIMENTAL_WRITER_TRANSLATION_PREFILL_SOURCE = ROOT / "receipts" / "future" / "KIMI_EXPERIMENTAL_HCLI_FIT_WRITER_TRANSLATION_PREFILL_LIVE_20260910.json"
EXPERIMENTAL_CONDITIONAL_TRANSLATION_SOURCE = ROOT / "receipts" / "future" / "KIMI_EXPERIMENTAL_HCLI_FIT_CONDITIONAL_TRANSLATION_LIVE_20260910.json"
SERVING_HCLI_CANDIDATE_SOURCE = ROOT / "receipts" / "future" / "KIMI_SERVING_CANDIDATE_HCLI_CONTRACT_LIVE_20260910.json"
CANDIDATE_EPISTEMIC_SOURCE = ROOT / "receipts" / "future" / "KIMI_CANDIDATE_OPE_EPISTEMIC_LIVE_20260910.json"
CANDIDATE_PHYSICAL_SOURCE = ROOT / "receipts" / "future" / "KIMI_CANDIDATE_OPE_PHYSICAL_LIVE_20260910.json"
SCHEMA = "hawking.kimi.operator_live_evidence.v1"
CANDIDATES = (
    "OPA", "OPB", "OPC", "OPD", "OPE", "OPF", "OPG", "OPH",
)


def _op_id(method: str) -> str:
    return f"KIMI_OPERATOR_CANDIDATE_{method}"


def build(
    source: Mapping[str, Any] | None = None,
    *,
    source_path: str | None = None,
    supplementary_sources: Mapping[str, Mapping[str, Any]] | None = None,
    candidate_hcli: Mapping[str, Any] | None = None,
    experimental_hcli: Mapping[str, Any] | None = None,
    experimental_hcli_nested: Mapping[str, Any] | None = None,
    experimental_hcli_residual: Mapping[str, Any] | None = None,
    experimental_hcli_translation: Mapping[str, Any] | None = None,
    experimental_hcli_prefill: Mapping[str, Any] | None = None,
    experimental_writer_translation_prefill: Mapping[str, Any] | None = None,
    experimental_conditional_translation: Mapping[str, Any] | None = None,
    serving_hcli_candidate: Mapping[str, Any] | None = None,
    candidate_epistemic: Mapping[str, Any] | None = None,
    candidate_physical: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if source is None:
        source = _load(SOURCE)
    documents: list[tuple[str, dict[str, Any]]] = [(
        source_path or str(SOURCE), dict(source)
    )]
    for path, document in (supplementary_sources or {}).items():
        documents.append((str(path), dict(document)))
    candidate_hcli_document = (
        dict(candidate_hcli)
        if candidate_hcli is not None
        else _load(CANDIDATE_HCLI_SOURCE)
        if source_path is None
        else {}
    )
    experimental_hcli_document = (
        dict(experimental_hcli)
        if experimental_hcli is not None
        else _load(EXPERIMENTAL_HCLI_SOURCE)
        if source_path is None
        else {}
    )
    experimental_hcli_nested_document = (
        dict(experimental_hcli_nested)
        if experimental_hcli_nested is not None
        else _load(EXPERIMENTAL_HCLI_NESTED_SOURCE)
        if source_path is None
        else {}
    )
    experimental_hcli_residual_document = (
        dict(experimental_hcli_residual)
        if experimental_hcli_residual is not None
        else _load(EXPERIMENTAL_HCLI_RESIDUAL_SOURCE)
        if source_path is None
        else {}
    )
    experimental_hcli_translation_document = (
        dict(experimental_hcli_translation)
        if experimental_hcli_translation is not None
        else _load(EXPERIMENTAL_HCLI_TRANSLATION_SOURCE)
        if source_path is None
        else {}
    )
    experimental_hcli_prefill_document = (
        dict(experimental_hcli_prefill)
        if experimental_hcli_prefill is not None
        else _load(EXPERIMENTAL_HCLI_PREFILL_SOURCE)
        if source_path is None
        else {}
    )
    experimental_writer_translation_prefill_document = (
        dict(experimental_writer_translation_prefill)
        if experimental_writer_translation_prefill is not None
        else _load(EXPERIMENTAL_WRITER_TRANSLATION_PREFILL_SOURCE)
        if source_path is None
        else {}
    )
    experimental_conditional_translation_document = (
        dict(experimental_conditional_translation)
        if experimental_conditional_translation is not None
        else _load(EXPERIMENTAL_CONDITIONAL_TRANSLATION_SOURCE)
        if source_path is None
        else {}
    )
    serving_hcli_candidate_document = (
        dict(serving_hcli_candidate)
        if serving_hcli_candidate is not None
        else _load(SERVING_HCLI_CANDIDATE_SOURCE)
        if source_path is None
        else {}
    )
    candidate_epistemic_document = (
        dict(candidate_epistemic)
        if candidate_epistemic is not None
        else _load(CANDIDATE_EPISTEMIC_SOURCE)
        if source_path is None
        else {}
    )
    candidate_physical_document = (
        dict(candidate_physical)
        if candidate_physical is not None
        else _load(CANDIDATE_PHYSICAL_SOURCE)
        if source_path is None
        else {}
    )
    rows: dict[str, dict[str, Any]] = {}
    for method in CANDIDATES:
        rows[_op_id(method)] = {
            "candidate_id": _op_id(method),
            "method": method,
            "status": "NOT_MEASURED",
            "causal_effect_established": False,
            "causal_sample_sufficient": False,
            "source_receipt": documents[0][0],
            "reason": (
                "No candidate-specific clean-process live weight evaluation has "
                "been run for this method."
            ),
        }
    for document_path, document in documents:
        if document.get("status") in {"OPH_EXECUTION_INCOMPLETE", "OPH_DEVICE_BUSY"}:
            candidate_id = document.get("candidate_id")
            if candidate_id in rows:
                deferred = document.get("status") == "OPH_DEVICE_BUSY"
                rows[candidate_id] = {
                    "candidate_id": candidate_id,
                    "method": document.get("candidate_method") or "OPH",
                    "status": "DEFERRED_DEVICE_BUSY" if deferred else "EXECUTION_INCOMPLETE",
                    "causal_effect_established": False,
                    "causal_sample_sufficient": False,
                    "source_receipt": document_path,
                    "failure_class": document.get("failure_class"),
                    "device_preflight": document.get("device_preflight"),
                    "reason": document.get("claim_boundary") or "Clean-process run ended without an atomic result.",
                }
        # Preflight/incomplete documents may carry a candidate-shaped summary
        # for receipt ergonomics, but they contain no behavior outcomes.  Their
        # resource/integration status must not be overwritten as a causal-null
        # intervention below.
        candidate_reports = {} if document.get("status") in {
            "OPH_EXECUTION_INCOMPLETE", "OPH_DEVICE_BUSY"
        } else document.get("candidate_causal_interventions") or {}
        if isinstance(candidate_reports, Mapping):
            for candidate_id, report in candidate_reports.items():
                if not isinstance(report, Mapping) or candidate_id not in rows:
                    continue
                crossfit = str(report.get("operator_kind") or "").startswith("crossfit_")
                effect = bool(report.get("causal_effect_established"))
                sufficient = bool(report.get("causal_sample_sufficient"))
                consistent = bool(report.get("crossfit_consistent")) if crossfit else True
                rows[candidate_id] = {
                    "candidate_id": candidate_id,
                    "method": report.get("candidate_method") or rows[candidate_id]["method"],
                    "status": (
                        "QUALIFIED"
                        if effect and sufficient and consistent
                        else "REJECTED_CROSSFIT_INCONSISTENT"
                        if effect and crossfit and not consistent
                        else "REJECTED_CAUSAL_NULL"
                    ),
                    "causal_effect_established": effect,
                    "causal_sample_sufficient": sufficient,
                    "measurement_status": report.get("status"),
                    "crossfit_consistent": consistent if crossfit else None,
                    "positive_fold_count": report.get("positive_fold_count"),
                    "causal_sample_counts": report.get("causal_sample_counts"),
                    "refusal_rate_delta": report.get("refusal_rate_delta"),
                    "layer": report.get("layer"),
                    "layers": report.get("layers"),
                    "writer": report.get("writer"),
                    "candidate_scope": report.get("candidate_scope"),
                    "source_receipt": document_path,
                    "reason": (
                        "Candidate run completed without enough behavior comparison "
                        "rows to establish a causal effect."
                        if not sufficient else
                        "Candidate-labelled live intervention was not positive and "
                        "consistent against its matched null."
                    ),
                }
        moe = document.get("moe_causal_intervention") or {}
        # A candidate-labelled receipt may also retain a generic MoE arm for
        # comparison. Never let that shorthand overwrite a candidate-specific
        # report (in particular, an OPG receipt's generic include_shared=false
        # arm must not be rebound to OPF).
        if moe and not candidate_reports:
            method = "OPG" if bool(moe.get("include_shared")) else "OPF"
            rows[_op_id(method)] = {
                "candidate_id": _op_id(method),
                "method": method,
                "status": (
                    "QUALIFIED"
                    if moe.get("causal_effect_established")
                    and moe.get("causal_sample_sufficient")
                    else "REJECTED_CAUSAL_NULL"
                ),
                "causal_effect_established": bool(moe.get("causal_effect_established")),
                "causal_sample_sufficient": bool(moe.get("causal_sample_sufficient")),
                "causal_sample_counts": moe.get("causal_sample_counts"),
                "refusal_rate_delta": moe.get("refusal_rate_delta"),
                "layer": moe.get("layer"),
                "expert_ids": moe.get("expert_ids"),
                "include_shared": moe.get("include_shared"),
                "source_receipt": document_path,
                "reason": (
                    "Selected-expert intervention is candidate-specific; it must beat "
                    "the matched null with sufficient held-out behavior rows before "
                    "promotion."
                ),
            }
    candidate_hcli_evaluations = {}
    if candidate_hcli_document:
        candidate_id = candidate_hcli_document.get("candidate_id")
        hcli = candidate_hcli_document.get("hcli_contract_intervention") or {}
        if candidate_id in rows and isinstance(hcli, Mapping):
            candidate_hcli_evaluations[candidate_id] = {
                "source_receipt": str(CANDIDATE_HCLI_SOURCE),
                "status": candidate_hcli_document.get("status"),
                "causal_effect_established": bool(
                    hcli.get("causal_effect_established")
                ),
                "causal_sample_sufficient": bool(
                    hcli.get("causal_sample_sufficient")
                ),
                "baseline_passed": hcli.get("baseline_passed"),
                "treated_passed": hcli.get("treated_passed"),
                "null_passed": hcli.get("null_passed"),
                "pass_rate_delta": hcli.get("pass_rate_delta"),
                "null_pass_rate_delta": hcli.get("null_pass_rate_delta"),
                "baseline_unsafe_demo_indicators": hcli.get(
                    "baseline_unsafe_demo_indicators"
                ),
                "treated_unsafe_demo_indicators": hcli.get(
                    "treated_unsafe_demo_indicators"
                ),
                "treated_effective_refusals": hcli.get(
                    "treated_effective_refusals"
                ),
                "reason": "Candidate-specific HCLI contract outcome is null or regressive."
                if not hcli.get("causal_effect_established") else
                "Candidate-specific HCLI contract outcome passed its declared comparison.",
            }
            rows[candidate_id]["hcli_contract"] = candidate_hcli_evaluations[candidate_id]
    candidate_epistemic_evaluations = {}
    if candidate_epistemic_document:
        candidate_id = candidate_epistemic_document.get("candidate_id")
        parity = candidate_epistemic_document.get("parity") or {}
        if candidate_id in rows and isinstance(parity, Mapping):
            candidate_epistemic_evaluations[candidate_id] = {
                "source_receipt": str(CANDIDATE_EPISTEMIC_SOURCE),
                "status": candidate_epistemic_document.get("status"),
                "no_regression": bool(parity.get("no_regression")),
                "baseline_passed": parity.get("baseline_passed"),
                "treated_passed": parity.get("treated_passed"),
                "null_passed": parity.get("null_passed"),
                "brier_delta": parity.get("brier_delta"),
                "ece_delta": parity.get("ece_delta"),
                "reason": (
                    "Candidate epistemic parity passed without calibration or task "
                    "regression."
                    if parity.get("no_regression") else
                    "Candidate epistemic parity is missing or regressive."
                ),
            }
            rows[candidate_id]["epistemic_parity"] = candidate_epistemic_evaluations[candidate_id]
    candidate_physical_evaluations = {}
    if candidate_physical_document:
        candidate_id = candidate_physical_document.get("candidate_id")
        parity = candidate_physical_document.get("parity") or {}
        if candidate_id in rows and isinstance(parity, Mapping):
            candidate_physical_evaluations[candidate_id] = {
                "source_receipt": str(CANDIDATE_PHYSICAL_SOURCE),
                "status": candidate_physical_document.get("status"),
                "no_physical_regression": bool(parity.get("no_physical_regression")),
                "treated_to_baseline_ratio": parity.get("treated_to_baseline_ratio"),
                "null_to_baseline_ratio": parity.get("null_to_baseline_ratio"),
                "same_task_count": bool(parity.get("same_task_count")),
                "reason": (
                    "Candidate physical parity passed the fixed-workload latency "
                    "non-regression check."
                    if parity.get("no_physical_regression") else
                    "Candidate physical parity is missing or regressive."
                ),
            }
            rows[candidate_id]["physical_parity"] = candidate_physical_evaluations[candidate_id]
    qualified = [row["candidate_id"] for row in rows.values() if (
        row.get("status") == "QUALIFIED"
        and row.get("causal_effect_established") is True
        and row.get("causal_sample_sufficient") is True
        and (
            row["candidate_id"] not in candidate_hcli_evaluations
            or candidate_hcli_evaluations[row["candidate_id"]].get(
                "causal_effect_established") is True
        )
    )]
    return {
        "schema": SCHEMA,
        "status": "CANDIDATE_LIVE_EVIDENCE_INCOMPLETE" if not qualified else "CANDIDATE_LIVE_EVIDENCE_COMPLETE",
        "source_model": "KIMI_BASE",
        "source_evaluation": documents[0][0],
        "source_evaluations": [path for path, _document in documents],
        "candidates": rows,
        "candidate_hcli_evaluations": candidate_hcli_evaluations,
        "candidate_epistemic_evaluations": candidate_epistemic_evaluations,
        "candidate_physical_evaluations": candidate_physical_evaluations,
        "experimental_hcli_evaluation": experimental_hcli_document,
        "experimental_hcli_evaluations": {
            "writer_projection": experimental_hcli_document,
            "writer_projection_nested_selection": experimental_hcli_nested_document,
            "residual_addition": experimental_hcli_residual_document,
            "residual_translation_nested_selection": experimental_hcli_translation_document,
            "prefill_translation": experimental_hcli_prefill_document,
            "writer_output_translation_prefill": experimental_writer_translation_prefill_document,
            "conditional_translation_prefill": experimental_conditional_translation_document,
        },
        "serving_path_evaluations": {
            "hcli_contract": serving_hcli_candidate_document,
        },
        "qualified_serving_path_candidates": (
            [serving_hcli_candidate_document.get("candidate_id")]
            if serving_hcli_candidate_document.get("status") ==
            "SERVING_CANDIDATE_HCLI_EFFECT_ESTABLISHED"
            else []
        ),
        "qualified_candidates": qualified,
        "claim_boundary": (
            "Candidate-specific live evidence only. NOT_MEASURED and causal-null "
            "rows cannot support promotion; this receipt does not write weights, "
            "create an artifact, or widen authorization. Serving-path HCLI evidence "
            "is retained under a separate namespace and cannot qualify a learned "
            "OPA--OPG candidate."
        ),
    }


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"_error": f"{type(exc).__name__}: {exc}", "_path": str(path)}
    return value if isinstance(value, dict) else {"_error": "receipt is not an object"}


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, action="append",
        help="candidate-specific live receipt; repeat to consolidate independent runs",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "receipts" / "future" / "KIMI_OPERATOR_LIVE_EVIDENCE_20260910.json",
    )
    args = parser.parse_args()
    sources = args.source or [SOURCE, *CANDIDATE_LIVE_SOURCES]
    result = build(
        _load(sources[0]),
        source_path=str(sources[0]),
        supplementary_sources={str(path): _load(path) for path in sources[1:]},
        candidate_hcli=_load(CANDIDATE_HCLI_SOURCE),
        experimental_hcli=_load(EXPERIMENTAL_HCLI_SOURCE),
        experimental_hcli_nested=_load(EXPERIMENTAL_HCLI_NESTED_SOURCE),
        experimental_hcli_residual=_load(EXPERIMENTAL_HCLI_RESIDUAL_SOURCE),
        experimental_hcli_translation=_load(EXPERIMENTAL_HCLI_TRANSLATION_SOURCE),
        experimental_hcli_prefill=_load(EXPERIMENTAL_HCLI_PREFILL_SOURCE),
        experimental_writer_translation_prefill=_load(EXPERIMENTAL_WRITER_TRANSLATION_PREFILL_SOURCE),
        experimental_conditional_translation=_load(EXPERIMENTAL_CONDITIONAL_TRANSLATION_SOURCE),
        serving_hcli_candidate=_load(SERVING_HCLI_CANDIDATE_SOURCE),
        candidate_epistemic=_load(CANDIDATE_EPISTEMIC_SOURCE),
        candidate_physical=_load(CANDIDATE_PHYSICAL_SOURCE),
    )
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["input_hashes"] = {
        "source_evaluations": {
            str(path): _sha256(path) for path in sources
        },
        "candidate_hcli": _sha256(CANDIDATE_HCLI_SOURCE),
        "serving_hcli_candidate": _sha256(SERVING_HCLI_CANDIDATE_SOURCE),
        "candidate_epistemic": _sha256(CANDIDATE_EPISTEMIC_SOURCE),
        "candidate_physical": _sha256(CANDIDATE_PHYSICAL_SOURCE),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "qualified_candidates": result["qualified_candidates"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
