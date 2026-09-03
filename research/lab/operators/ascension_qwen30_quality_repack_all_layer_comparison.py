"""Fail-closed, candidate-local comparison of one HQ30GR2 all-layer trace.

This operator consumes the sealed one-shot all-layer current-trace diagnostic
and describes only what that diagnostic actually retained: exact full-logit
vector identity via hashes, bounded top-k changes, route-digest changes, and
the typed L0/E0 sparse interception witness.  It deliberately has no source
full-model logit oracle and therefore cannot call a divergence an improvement,
semantic coherence, HCLI behaviour, or a performance result.

It is CPU-only.  It neither loads a model nor starts a Metal runtime, server,
watcher, HCLI client, or benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates"
    / "gate-up-residual-v1"
)
DEFAULT_DIAGNOSTIC = (
    DEFAULT_CANDIDATE_ROOT
    / "all-layer-current-trace-diagnostic-runs/lease.E6Nqqa72/outer-capture/inner/receipt.json"
)
DEFAULT_OUTER = (
    DEFAULT_CANDIDATE_ROOT
    / "all-layer-current-trace-diagnostic-runs/lease.E6Nqqa72/outer-capture/outer-terminal-receipt.json"
)
DEFAULT_CHAIN_CURRENT = DEFAULT_CANDIDATE_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CURRENT_HCLI_L0_E0_CHAIN_EFFECT.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_CANDIDATE_ROOT / "all-layer-current-trace-comparison/receipts"

SCHEMA = "hawking.ascension.qwen30_hq30gr2_all_layer_current_trace_comparison.v1"
STATUS = "EARNED_CANDIDATE_LOCAL_ALL_LAYER_DIVERGENCE_UNQUALIFIED_NON_PROMOTABLE"

INNER_SCHEMA = "hawking.ascension.qwen30_quality_repack_all_layer_current_trace_diagnostic.v1"
INNER_STATUS = "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_TYPED_HQ30GR2_ALL_LAYER_CURRENT_TRACE_UNQUALIFIED"
OUTER_SCHEMA = "hawking.ascension.qwen30_hq30gr2_all_layer_current_trace_outer_launcher.v1"
OUTER_STATUS = "CAPTURED_QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_DIAGNOSTIC_UNQUALIFIED"
CHAIN_CURRENT_SCHEMA = "hawking.ascension.qwen30_quality_repack_current_hcli_l0_e0_chain_current.v1"
CHAIN_CURRENT_STATUS = "CURRENT_QWEN30_HQ30GR2_CURRENT_HCLI_L0_E0_CHAIN_RECEIPT_SELECTED"
CHAIN_SCHEMA = "hawking.ascension.qwen30_quality_repack_current_hcli_l0_e0_chain.v1"
CHAIN_STATUS = "EARNED_CURRENT_CAPTURED_L0_E0_CHAIN_IMPROVEMENT_UNQUALIFIED"

TOP_K = 8
LAYER_COUNT = 48
EXPERTS_PER_LAYER = 8
PREFIX_TOKENS = 369
TARGET_POSITION = 337


class AllLayerComparisonError(RuntimeError):
    """A candidate-local comparison cannot safely bind its evidence."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _evidence(path: Path) -> dict[str, Any]:
    try:
        stat_result = path.stat()
    except OSError as exc:
        raise AllLayerComparisonError(f"cannot stat evidence {path}: {exc}") from exc
    return {
        "path": str(path.resolve()),
        "bytes": stat_result.st_size,
        "sha256": _sha256_file(path),
    }


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sealed(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        checked = verify(value, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise AllLayerComparisonError(f"{label} is absent or has an invalid seal: {exc}") from exc
    if not isinstance(checked, Mapping):
        raise AllLayerComparisonError(f"{label} is not an object")
    return dict(checked)


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AllLayerComparisonError(f"{label} must be an object")
    return dict(value)


def _text(value: object, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise AllLayerComparisonError(f"{label} must be a non-empty string")
    if sha256 and (len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
        raise AllLayerComparisonError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AllLayerComparisonError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise AllLayerComparisonError(f"{label} must be a boolean")
    return value


def _reference(document: Mapping[str, Any], path: Path) -> dict[str, Any]:
    return {**_evidence(path), "seal_sha256": _text(document.get("seal_sha256"), label="receipt seal", sha256=True)}


def _assert_schema_status(document: Mapping[str, Any], *, schema: str, status: str, label: str) -> None:
    if document.get("schema") != schema:
        raise AllLayerComparisonError(f"{label} schema is not {schema}")
    if document.get("status") != status:
        raise AllLayerComparisonError(f"{label} status is not {status}")


def _top_k(value: object, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != TOP_K:
        raise AllLayerComparisonError(f"{label} must contain exactly {TOP_K} entries")
    result: list[dict[str, Any]] = []
    previous = math.inf
    seen: set[int] = set()
    for index, entry in enumerate(value, start=1):
        row = _object(entry, label=f"{label}[{index}]")
        token_id = _integer(row.get("token_id"), label=f"{label}[{index}] token ID")
        logit = row.get("logit")
        if isinstance(logit, bool) or not isinstance(logit, (int, float)) or not math.isfinite(float(logit)):
            raise AllLayerComparisonError(f"{label}[{index}] logit must be finite")
        logit_float = float(logit)
        if logit_float > previous:
            raise AllLayerComparisonError(f"{label} is not in descending logit order")
        previous = logit_float
        bits = _integer(row.get("logit_bits"), label=f"{label}[{index}] logit bits")
        if bits > 0xFFFFFFFF:
            raise AllLayerComparisonError(f"{label}[{index}] logit bits exceed F32")
        if token_id in seen:
            raise AllLayerComparisonError(f"{label} repeats token ID {token_id}")
        seen.add(token_id)
        result.append({"token_id": token_id, "logit": logit_float, "logit_bits": bits})
    return result


def _logits(value: object, *, label: str) -> dict[str, Any]:
    row = _object(value, label=label)
    vocab_rows = _integer(row.get("vocab_rows"), label=f"{label} vocab rows", minimum=1)
    full_hash = _text(row.get("full_f32le_sha256"), label=f"{label} full F32LE SHA-256", sha256=True)
    return {"vocab_rows": vocab_rows, "full_f32le_sha256": full_hash, "top_k": _top_k(row.get("top_k"), label=f"{label} top-k")}


def _route_step(value: object, *, label: str) -> dict[str, Any]:
    row = _object(value, label=label)
    ids = row.get("l0_expert_ids")
    if not isinstance(ids, list) or len(ids) != EXPERTS_PER_LAYER:
        raise AllLayerComparisonError(f"{label} must retain exactly {EXPERTS_PER_LAYER} L0 experts")
    l0_ids = [_integer(item, label=f"{label} L0 expert") for item in ids]
    if len(set(l0_ids)) != EXPERTS_PER_LAYER:
        raise AllLayerComparisonError(f"{label} repeats an L0 expert")
    return {
        "position": _integer(row.get("position"), label=f"{label} position"),
        "input_token_id": _integer(row.get("input_token_id"), label=f"{label} input token"),
        "sampled_token_id": _integer(row.get("sampled_token_id"), label=f"{label} sampled token"),
        "route_ids_u32le_sha256": _text(row.get("route_ids_u32le_sha256"), label=f"{label} route digest", sha256=True),
        "l0_expert0_selected": _boolean(row.get("l0_expert0_selected"), label=f"{label} L0/E0 selection"),
        "l0_expert_ids": l0_ids,
        "all_layers_route_captured": _integer(row.get("all_layers_route_captured"), label=f"{label} layer captures"),
    }


def _prefix(value: object, *, label: str) -> dict[str, Any]:
    row = _object(value, label=label)
    result = {
        "exact_prefix_token_forwards": _integer(row.get("exact_prefix_token_forwards"), label=f"{label} prefix forwards"),
        "all_layer_route_captures": _integer(row.get("all_layer_route_captures"), label=f"{label} route captures"),
        "layers_per_forward": _integer(row.get("layers_per_forward"), label=f"{label} layers per forward"),
        "route_trace_sha256": _text(row.get("route_trace_sha256"), label=f"{label} trace digest", sha256=True),
        "l0_expert0_selected_positions": row.get("l0_expert0_selected_positions"),
        "target_position_step": _route_step(row.get("target_position_step"), label=f"{label} target step"),
        "final_prefix_step": _route_step(row.get("final_prefix_step"), label=f"{label} final prefix step"),
        "final_logits": _logits(row.get("final_logits"), label=f"{label} final logits"),
    }
    positions = result["l0_expert0_selected_positions"]
    if not isinstance(positions, list) or any(isinstance(position, bool) or not isinstance(position, int) for position in positions):
        raise AllLayerComparisonError(f"{label} L0/E0 positions must be an integer list")
    if result["exact_prefix_token_forwards"] != PREFIX_TOKENS:
        raise AllLayerComparisonError(f"{label} prefix forward count is not {PREFIX_TOKENS}")
    if result["layers_per_forward"] != LAYER_COUNT or result["all_layer_route_captures"] != PREFIX_TOKENS * LAYER_COUNT:
        raise AllLayerComparisonError(f"{label} does not prove {PREFIX_TOKENS} x {LAYER_COUNT} route capture")
    if result["target_position_step"]["position"] != TARGET_POSITION:
        raise AllLayerComparisonError(f"{label} does not retain target position {TARGET_POSITION}")
    if result["target_position_step"]["l0_expert0_selected"] is not True or TARGET_POSITION not in positions:
        raise AllLayerComparisonError(f"{label} does not prove L0/E0 route reach at {TARGET_POSITION}")
    return result


def _continuation(value: object, *, label: str) -> dict[str, Any]:
    row = _object(value, label=label)
    if _integer(row.get("additional_forwards"), label=f"{label} additional forwards") != 1:
        raise AllLayerComparisonError(f"{label} must retain exactly one forced continuation")
    return {
        "step": _route_step(row.get("step"), label=f"{label} step"),
        "final_logits": _logits(row.get("final_logits"), label=f"{label} final logits"),
    }


def _top_k_comparison(control: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    control_by_token = {row["token_id"]: (rank, row) for rank, row in enumerate(control, start=1)}
    candidate_by_token = {row["token_id"]: (rank, row) for rank, row in enumerate(candidate, start=1)}
    common = sorted(set(control_by_token) & set(candidate_by_token))
    shared = []
    for token_id in common:
        control_rank, control_row = control_by_token[token_id]
        candidate_rank, candidate_row = candidate_by_token[token_id]
        shared.append(
            {
                "token_id": token_id,
                "control_rank": control_rank,
                "candidate_rank": candidate_rank,
                "control_logit": control_row["logit"],
                "candidate_logit": candidate_row["logit"],
                "candidate_minus_control_logit": candidate_row["logit"] - control_row["logit"],
            }
        )
    return {
        "k": TOP_K,
        "control_top1_token_id": control[0]["token_id"],
        "candidate_top1_token_id": candidate[0]["token_id"],
        "top1_token_equal": control[0]["token_id"] == candidate[0]["token_id"],
        "shared_token_count": len(common),
        "union_token_count": len(set(control_by_token) | set(candidate_by_token)),
        "jaccard": len(common) / len(set(control_by_token) | set(candidate_by_token)),
        "control_only_token_ids": sorted(set(control_by_token) - set(candidate_by_token)),
        "candidate_only_token_ids": sorted(set(candidate_by_token) - set(control_by_token)),
        "shared_tokens_with_ranks_and_unreferenced_logit_deltas": shared,
        "ranking_or_logit_direction_is_not_a_source_or_semantic_quality_claim": True,
    }


def _full_logit_comparison(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    if control["vocab_rows"] != candidate["vocab_rows"]:
        raise AllLayerComparisonError("control and candidate final logits have different vocab row counts")
    equal = control["full_f32le_sha256"] == candidate["full_f32le_sha256"]
    return {
        "vocab_rows": control["vocab_rows"],
        "control_full_f32le_sha256": control["full_f32le_sha256"],
        "candidate_full_f32le_sha256": candidate["full_f32le_sha256"],
        "exact_f32_vector_equal_by_hash": equal,
        "numeric_distance_not_available_from_hash_only_witness": True,
        "source_final_logit_oracle_not_present": True,
    }


def _route_comparison(control: Mapping[str, Any], candidate: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    for field in ("position", "input_token_id"):
        if control[field] != candidate[field]:
            raise AllLayerComparisonError(f"{label} control/candidate {field} differs; comparison is not matched")
    if control["all_layers_route_captured"] != LAYER_COUNT or candidate["all_layers_route_captured"] != LAYER_COUNT:
        raise AllLayerComparisonError(f"{label} does not retain all {LAYER_COUNT} routes")
    if control["l0_expert0_selected"] != candidate["l0_expert0_selected"]:
        raise AllLayerComparisonError(f"{label} control/candidate L0/E0 selection differs")
    return {
        "label": label,
        "position": control["position"],
        "input_token_id": control["input_token_id"],
        "control_sampled_token_id": control["sampled_token_id"],
        "candidate_sampled_token_id": candidate["sampled_token_id"],
        "control_route_ids_u32le_sha256": control["route_ids_u32le_sha256"],
        "candidate_route_ids_u32le_sha256": candidate["route_ids_u32le_sha256"],
        "exact_serialized_all_layer_route_vector_equal_by_hash": control["route_ids_u32le_sha256"] == candidate["route_ids_u32le_sha256"],
        "l0_expert_ids_equal": control["l0_expert_ids"] == candidate["l0_expert_ids"],
        "l0_expert_ids": control["l0_expert_ids"] if control["l0_expert_ids"] == candidate["l0_expert_ids"] else None,
        "l0_expert0_selected": control["l0_expert0_selected"],
        "digest_only_witness_does_not_identify_changed_layer_count_or_ids": True,
    }


def _chain_current(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    current = _sealed(path, label="L0/E0 chain current pointer")
    _assert_schema_status(current, schema=CHAIN_CURRENT_SCHEMA, status=CHAIN_CURRENT_STATUS, label="L0/E0 chain current pointer")
    chain_ref = _object(current.get("chain_receipt"), label="L0/E0 chain current receipt reference")
    chain_path = Path(_text(chain_ref.get("path"), label="L0/E0 chain receipt path"))
    chain = _sealed(chain_path, label="L0/E0 chain receipt")
    _assert_schema_status(chain, schema=CHAIN_SCHEMA, status=CHAIN_STATUS, label="L0/E0 chain receipt")
    if chain.get("seal_sha256") != chain_ref.get("seal_sha256"):
        raise AllLayerComparisonError("L0/E0 chain current pointer does not bind its receipt seal")
    assessment = _object(chain.get("assessment"), label="L0/E0 chain assessment")
    if assessment.get("all_three_selected_positions_complete_mlp_down_output_improved_vs_source") is not True:
        raise AllLayerComparisonError("L0/E0 chain receipt does not prove all three local source improvements")
    return current, chain


def build_comparison(
    *,
    diagnostic_path: Path,
    outer_path: Path,
    chain_current_path: Path,
) -> dict[str, Any]:
    """Validate existing evidence and return a sealed, unqualified comparison."""
    inner = _sealed(diagnostic_path, label="all-layer inner diagnostic receipt")
    _assert_schema_status(inner, schema=INNER_SCHEMA, status=INNER_STATUS, label="all-layer inner diagnostic receipt")
    outer = _sealed(outer_path, label="all-layer outer terminal receipt")
    _assert_schema_status(outer, schema=OUTER_SCHEMA, status=OUTER_STATUS, label="all-layer outer terminal receipt")
    outer_inner = _object(outer.get("inner_probe_capture"), label="outer inner capture")
    outer_inner_ref = _object(outer_inner.get("receipt"), label="outer inner receipt reference")
    if Path(_text(outer_inner_ref.get("path"), label="outer inner receipt path")).resolve() != diagnostic_path.resolve():
        raise AllLayerComparisonError("outer terminal receipt does not bind the supplied inner diagnostic path")
    if _text(outer_inner_ref.get("sha256"), label="outer inner receipt SHA-256", sha256=True) != _sha256_file(diagnostic_path):
        raise AllLayerComparisonError("outer terminal receipt does not bind the supplied inner diagnostic bytes")
    if outer_inner.get("binding_valid") is not True or outer_inner.get("metal_performed") is not True:
        raise AllLayerComparisonError("outer terminal receipt does not validate actual inner Metal capture")

    policy = _object(inner.get("metal_execution_policy"), label="inner Metal execution policy")
    for field in (
        "timing_or_benchmarking_allowed",
        "hcli_or_server_allowed",
        "tps_or_tg_claim_allowed",
        "coherence_claim_allowed",
        "capability_claim_allowed",
        "tournament_claim_allowed",
    ):
        if policy.get(field) is not False:
            raise AllLayerComparisonError(f"inner policy unexpectedly allows {field}")

    witnesses = _object(inner.get("structural_witnesses"), label="inner structural witnesses")
    control_prefix = _prefix(witnesses.get("control_scalar_path"), label="control scalar prefix")
    candidate_prefix = _prefix(witnesses.get("candidate_typed_hq30gr2_path"), label="candidate HQ30GR2 prefix")
    control_continuation = _continuation(witnesses.get("control_forced_continuation"), label="control forced continuation")
    candidate_continuation = _continuation(witnesses.get("candidate_forced_continuation"), label="candidate forced continuation")

    execution = _object(inner.get("exact_trace_execution"), label="exact trace execution")
    if execution.get("source_template_token_count") != PREFIX_TOKENS:
        raise AllLayerComparisonError("inner execution did not bind the expected source-template token count")
    forced = _object(execution.get("forced_continuation"), label="forced continuation")
    forced_token = _integer(forced.get("forced_token_id"), label="forced continuation token")
    if control_continuation["step"]["input_token_id"] != forced_token or candidate_continuation["step"]["input_token_id"] != forced_token:
        raise AllLayerComparisonError("forced continuation input token differs from the sealed common token")

    target = _route_comparison(control_prefix["target_position_step"], candidate_prefix["target_position_step"], label="L0/E0 target step")
    if target["position"] != TARGET_POSITION or target["l0_expert0_selected"] is not True:
        raise AllLayerComparisonError("candidate did not retain the exact L0/E0 target route")
    final_prefix_route = _route_comparison(control_prefix["final_prefix_step"], candidate_prefix["final_prefix_step"], label="final prefix step")
    continuation_route = _route_comparison(control_continuation["step"], candidate_continuation["step"], label="forced continuation step")

    interception = _object(witnesses.get("typed_l0_e0_sparse_interception"), label="typed L0/E0 interception")
    device_encodes = _integer(interception.get("device_sparse_gate_up_encodes"), label="typed sparse device encodes")
    matching_routes = _integer(interception.get("matching_l0_e0_route_selections"), label="typed sparse matching routes")
    if device_encodes != matching_routes or device_encodes != 1:
        raise AllLayerComparisonError("typed sparse interception count is not exactly the one observed L0/E0 route")
    if interception.get("direct_fallback_for_sparse_residual_forbidden") is not True:
        raise AllLayerComparisonError("typed sparse interception did not forbid direct fallback")
    if interception.get("scalar_control_topology_for_all_unchanged_organs") is not True:
        raise AllLayerComparisonError("candidate does not preserve scalar topology for unchanged organs")

    chain_current, chain = _chain_current(chain_current_path)
    chain_assessment = _object(chain.get("assessment"), label="L0/E0 chain assessment")
    local_selected = _object(chain_assessment.get("actual_route_reach"), label="L0/E0 chain route reach")
    selected_positions = _object(local_selected.get("selected_positions"), label="L0/E0 chain selected positions")
    if selected_positions.get("literal_hawking") != TARGET_POSITION:
        raise AllLayerComparisonError("local L0/E0 chain does not bind literal_hawking target position")

    prefix_logit = _full_logit_comparison(control_prefix["final_logits"], candidate_prefix["final_logits"])
    continuation_logit = _full_logit_comparison(control_continuation["final_logits"], candidate_continuation["final_logits"])
    prefix_top_k = _top_k_comparison(control_prefix["final_logits"]["top_k"], candidate_prefix["final_logits"]["top_k"])
    continuation_top_k = _top_k_comparison(control_continuation["final_logits"]["top_k"], candidate_continuation["final_logits"]["top_k"])

    candidate_binding = _object(inner.get("artifact_binding"), label="candidate artifact binding")
    candidate_manifest_seal = _text(candidate_binding.get("candidate_manifest_seal_sha256"), label="candidate manifest seal", sha256=True)
    control_runtime_seal = _text(candidate_binding.get("control_runtime_receipt_seal_sha256"), label="control runtime seal", sha256=True)

    comparison = {
        "schema": SCHEMA,
        "status": STATUS,
        "recorded_at": _utc_now(),
        "evidence": {
            "inner_all_layer_diagnostic": _reference(inner, diagnostic_path),
            "outer_terminal": _reference(outer, outer_path),
            "l0_e0_chain_current": _reference(chain_current, chain_current_path),
            "l0_e0_chain_receipt": _reference(chain, Path(_text(_object(chain_current.get("chain_receipt"), label="chain current reference").get("path"), label="chain receipt path"))),
        },
        "binding": {
            "candidate_manifest_seal_sha256": candidate_manifest_seal,
            "candidate_admission_receipt_seal_sha256": _text(candidate_binding.get("candidate_admission_receipt_seal_sha256"), label="candidate admission seal", sha256=True),
            "control_manifest_seal_sha256": _text(candidate_binding.get("control_manifest_seal_sha256"), label="control manifest seal", sha256=True),
            "control_runtime_receipt_seal_sha256": control_runtime_seal,
            "probe_id": execution.get("probe_id"),
            "source_template_token_count": PREFIX_TOKENS,
            "source_template_token_ids_u32le_sha256": _text(execution.get("source_template_token_ids_u32le_sha256"), label="source template token hash", sha256=True),
            "forced_identical_continuation_token_id": forced_token,
        },
        "observed_candidate_local_effect": {
            "candidate_mutates_only": interception.get("selected_residual_organs"),
            "literal_hawking_l0_e0_selected_at_position": TARGET_POSITION,
            "control_and_candidate_l0_expert_ids_at_target_equal": target["l0_expert_ids_equal"],
            "typed_sparse_device_encodes": device_encodes,
            "matching_candidate_l0_e0_route_selections": matching_routes,
            "direct_fallback_for_sparse_residual_forbidden": True,
            "all_unchanged_organs_retained_scalar_control_topology": True,
            "prior_cpu_local_chain_source_improvement": {
                "all_three_actual_e0_selected_positions_complete_mlp_down_output_improved_vs_source": True,
                "scope": "local L0/E0 gate/up-to-down contribution only; not an all-layer or semantic result",
                "probe_positions": selected_positions,
            },
            "causal_limit": "The all-layer receipt does not retain pre/post L0 hidden vectors or a source full-model final-logit oracle, so it cannot quantify the residual's causal magnitude/direction at later layers.",
        },
        "divergence": {
            "final_logit_vectors": {
                "exact_prefix": prefix_logit,
                "forced_shared_continuation": continuation_logit,
            },
            "bounded_top_k": {
                "exact_prefix": prefix_top_k,
                "forced_shared_continuation": continuation_top_k,
            },
            "route_trace": {
                "exact_prefix_aggregate": {
                    "control_route_trace_sha256": control_prefix["route_trace_sha256"],
                    "candidate_route_trace_sha256": candidate_prefix["route_trace_sha256"],
                    "exact_serialized_369_step_trace_equal_by_hash": control_prefix["route_trace_sha256"] == candidate_prefix["route_trace_sha256"],
                    "digest_only_witness_does_not_identify_changed_positions_layers_or_experts": True,
                },
                "target_l0_e0_step": target,
                "final_prefix_step": final_prefix_route,
                "forced_shared_continuation_step": continuation_route,
            },
        },
        "classification": {
            "candidate_runtime_promotion": "NOT_ELIGIBLE",
            "hcli_coherence_or_semantic_promotion": "NOT_ELIGIBLE",
            "tps_tg_capability_or_tournament_promotion": "NOT_ELIGIBLE",
            "allowed_next_scope": "BROADER_DIAGNOSTIC_ONLY",
            "reason": "One sealed literal_hawking trace proves exact candidate execution, one typed L0/E0 interception, and observable control/candidate divergence. It supplies neither a source full-model final-logit oracle nor semantic/coherence evaluation, and the bounded top-k witnesses cannot establish improvement.",
            "smallest_next_discriminator": {
                "name": "single-source-bound-three-way-final-logit-distance-on-the-existing-forced-current-trace",
                "contract": "Using the identical sealed 369-token literal_hawking prefix and forced token, execute/reference the source BF16 model only as an oracle and compare scalar-control versus HQ30GR2 final logits to that source final-logit vector. Predeclare an error metric and require candidate error < scalar error. This remains a one-trace numerical discriminator, not coherence or HCLI evidence.",
                "why_smallest": "It tests whether the observed candidate/control logit divergence moves toward the source at the exact all-layer boundary without changing server/runtime state, running generation, or expanding to public HCLI probes.",
            },
        },
        "claim_boundary": {
            "cpu_only_comparison_of_existing_sealed_records": True,
            "does_not_execute_or_modify_candidate_runtime_or_artifact": True,
            "does_not_touch_live_server_watcher_adapter_or_hcli": True,
            "does_not_use_metal_or_take_gpu_lease": True,
            "does_not_claim_source_directed_improvement_from_final_logits": True,
            "does_not_claim_semantic_coherence_hcli_tps_tg_capability_or_tournament": True,
            "does_not_schedule_or_authorize_a_followup_run": True,
        },
    }
    return seal(comparison)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-receipt", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--outer-terminal-receipt", type=Path, default=DEFAULT_OUTER)
    parser.add_argument("--chain-current", type=Path, default=DEFAULT_CHAIN_CURRENT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_comparison(
            diagnostic_path=args.diagnostic_receipt,
            outer_path=args.outer_terminal_receipt,
            chain_current_path=args.chain_current,
        )
        if args.output is None:
            candidate_seal = result["binding"]["candidate_manifest_seal_sha256"]
            inner_seal = result["evidence"]["inner_all_layer_diagnostic"]["seal_sha256"]
            output = DEFAULT_OUTPUT_ROOT / f"QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_COMPARISON_{candidate_seal}_{inner_seal}.json"
        else:
            output = args.output
        _atomic_json(output, result)
    except AllLayerComparisonError as exc:
        print(f"Q30 HQ30GR2 all-layer comparison refused: {exc}")
        return 2
    print(json.dumps({"output": str(output.resolve()), "status": result["status"], "seal_sha256": result["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
