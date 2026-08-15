"""Seal the narrow candidate-only HQ30GR2 runtime integration handoff.

This is intentionally an integration *contract*, not an adapter or a runtime
launcher.  It pins the isolated quality candidate's frozen selections and the
two CPU-only proofs required for its two changed gate/up organs.  A later Q30
runtime/optimizer owner must re-check this contract before it can add HQ30GR2
support to a full model.  The handoff never selects the candidate globally,
does not touch a baseline/current pointer, and has no Metal/runtime authority.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators import ascension_qwen_complete_binary_admission as shared
from lab.operators.ascension_qwen30_quality_repack import ARTIFACT_PREFIX
from lab.operators.ascension_qwen30_quality_repack_packed_matvec_parity import (
    CURRENT_SCHEMA as PACKED_CURRENT_SCHEMA,
    RECEIPT_SCHEMA as PACKED_RECEIPT_SCHEMA,
    RESULT_STATUS as PACKED_RESULT_STATUS,
    _matvec_current_path,
    _matvec_receipts_root,
    _validate_evidence,
)
from lab.operators.ascension_qwen30_quality_repack_scalar_parity import (
    ScalarParityError,
    ScalarParityTarget,
    _file_binding,
    _read_sealed,
    _require_int,
    _require_list,
    _require_mapping,
    _require_sha256,
    _require_string,
    _same_path,
    _verify_binding,
)
from lab.receipts import seal


REPO_ROOT = Path(__file__).resolve().parents[2]
HANDOFF_SCHEMA = "hawking.ascension.qwen30_quality_repack_hq30gr2_runtime_integration_handoff.v1"
HANDOFF_STATUS = "EARNED_CANDIDATE_ONLY_HQ30GR2_RUNTIME_INTEGRATION_HANDOFF_NOT_RUNTIME_QUALIFIED"
HANDOFF_VERSION = "v1-production-py312"
SELECTED_ORGANS = (
    "model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.layers.0.mlp.experts.0.up_proj.weight",
)

DEFAULT_TARGET = ScalarParityTarget(
    root=REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates/gate-up-residual-v1",
    baseline_root=REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen30/complete-gravity",
)


class IntegrationHandoffError(ScalarParityError):
    """No runtime consumer may accept a mixed or overclaimed handoff."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fail(message: str) -> IntegrationHandoffError:
    return IntegrationHandoffError(message)


def _handoff_path(target: ScalarParityTarget, evidence: Mapping[str, Any]) -> Path:
    manifest = _require_mapping(evidence.get("candidate_manifest"), "handoff candidate manifest")
    manifest_seal = _require_sha256(manifest.get("seal_sha256"), "handoff candidate manifest seal")
    return (
        target.root
        / "runtime-integration"
        / f"{ARTIFACT_PREFIX}_HQ30GR2_RUNTIME_INTEGRATION_HANDOFF_{HANDOFF_VERSION}_{manifest_seal}.json"
    )


def _validate_packed_current(target: ScalarParityTarget, evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the final CPU-only packed-matvec selection without trusting it."""

    pointer_path = _matvec_current_path(target)
    pointer, pointer_meta = _read_sealed(pointer_path, "quality packed matvec current pointer")
    if (
        pointer.get("schema") != PACKED_CURRENT_SCHEMA
        or pointer.get("status") != "CURRENT_QWEN30_QUALITY_REPACK_CPU_PACKED_MATVEC_PARITY_RECEIPT_SELECTED"
        or pointer.get("candidate_root") != str(target.root.resolve())
    ):
        raise _fail("packed matvec current pointer is not selected for this isolated candidate")
    manifest, manifest_meta = _read_sealed(target.manifest_path, "handoff candidate manifest")
    _verify_binding(
        pointer.get("candidate_manifest"),
        target.manifest_path,
        manifest,
        manifest_meta,
        "packed matvec current manifest binding",
    )
    if _require_mapping(pointer.get("candidate_manifest"), "packed current manifest") != _require_mapping(
        evidence.get("candidate_manifest"), "handoff evidence manifest"
    ):
        raise _fail("packed matvec pointer manifest differs from current candidate authority")
    receipt_binding = _require_mapping(pointer.get("packed_matvec_parity_receipt"), "packed matvec current receipt")
    receipt_path = Path(_require_string(receipt_binding.get("path"), "packed matvec receipt path"))
    if not receipt_path.is_absolute() or receipt_path.parent.resolve(strict=False) != _matvec_receipts_root(target).resolve():
        raise _fail("packed matvec current receipt leaves the candidate receipt root")
    receipt, receipt_meta = _read_sealed(receipt_path, "quality packed matvec parity receipt")
    if (
        receipt.get("schema") != PACKED_RECEIPT_SCHEMA
        or receipt.get("status") != PACKED_RESULT_STATUS
        or receipt.get("candidate_root") != str(target.root.resolve())
    ):
        raise _fail("packed matvec receipt is not the required CPU-only candidate proof")
    _verify_binding(receipt_binding, receipt_path, receipt, receipt_meta, "packed matvec current receipt binding")
    for field in (
        "candidate_native_admission_receipt",
        "candidate_manifest",
        "candidate_source_binding_snapshot",
        "immutable_source_revalidation",
        "scalar_parity_receipt",
    ):
        if _require_mapping(receipt.get(field), f"packed receipt {field}") != _require_mapping(
            evidence.get(field), f"handoff evidence {field}"
        ):
            raise _fail(f"packed matvec receipt {field} differs from current authority")
    if receipt.get("selected_organs") != evidence.get("pair_bindings"):
        raise _fail("packed matvec receipt selected organs differ from the sealed candidate pair")
    native = _require_mapping(receipt.get("native_cpu_direct_packed_matvec_probe"), "packed matvec native probe")
    boundary = _require_mapping(native.get("claim_boundary"), "packed matvec native boundary")
    expected_boundary = {
        "cpu_only": True,
        "metal_not_opened": True,
        "direct_packed_matvec_operator_only": True,
        "not_a_full_qwen_layer_decoder_generation_hcli_or_tps_result": True,
        "not_a_capability_tg_agent_os_or_tournament_qualification": True,
        "later_candidate_full_model_integration_requires_fresh_layer_model_and_runtime_gates": True,
    }
    if any(boundary.get(key) is not value for key, value in expected_boundary.items()):
        raise _fail("packed matvec proof does not retain its CPU-only claim boundary")
    contract = _require_mapping(receipt.get("integration_contract"), "packed matvec integration contract")
    if (
        contract.get("candidate_scalar_adapter_must_bind_this_exact_receipt_before_using_hq30gr2") is not True
        or contract.get("no_direct_hq30g1b1_fallback_for_selected_organs") is not True
        or contract.get("candidate_manifest_current_pointer_source_snapshot_and_baseline_control_must_be_rechecked_at_integration")
        is not True
    ):
        raise _fail("packed matvec proof does not require the strict HQ30GR2 integration checks")
    isolation = _require_mapping(receipt.get("isolation"), "packed matvec isolation")
    if (
        isolation.get("candidate_root_only") is not True
        or isolation.get("baseline_runtime_server_tournament_and_current_pointers_untouched") is not True
        or isolation.get("metal_and_full_candidate_runtime_not_started") is not True
    ):
        raise _fail("packed matvec proof does not retain candidate-only isolation")
    return {
        "packed_matvec_parity_current_pointer": _file_binding(pointer_path, pointer, pointer_meta),
        "packed_matvec_parity_receipt": _file_binding(receipt_path, receipt, receipt_meta),
    }


def _row_by_name(rows: Sequence[object], name: str) -> Mapping[str, Any]:
    matches = [
        _require_mapping(row, "candidate manifest tensor row")
        for row in rows
        if isinstance(row, Mapping) and row.get("tensor_name") == name
    ]
    if len(matches) != 1:
        raise _fail(f"candidate manifest must contain exactly one selected organ: {name}")
    return matches[0]


def _selected_hq30gr2_contract(target: ScalarParityTarget, evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract only the two sealed residual organs and their rollback route."""

    manifest, manifest_meta = _read_sealed(target.manifest_path, "handoff candidate manifest")
    _verify_binding(
        evidence.get("candidate_manifest"),
        target.manifest_path,
        manifest,
        manifest_meta,
        "handoff candidate manifest binding",
    )
    rows = _require_list(manifest.get("tensors"), "handoff candidate tensor catalog")
    pairs = _require_list(evidence.get("pair_bindings"), "handoff pair bindings")
    pair_by_name = {
        _require_string(_require_mapping(pair, "handoff pair binding").get("organ"), "handoff pair organ"): _require_mapping(
            pair, "handoff pair binding"
        )
        for pair in pairs
    }
    if tuple(pair_by_name) != SELECTED_ORGANS:
        raise _fail("handoff pair order/selection is not the exact sealed gate/up pair")
    selected: list[dict[str, Any]] = []
    for name in SELECTED_ORGANS:
        row = _row_by_name(rows, name)
        pair = pair_by_name[name]
        candidate = _require_mapping(pair.get("candidate"), f"handoff {name} candidate payload")
        control = _require_mapping(pair.get("admitted_scalar_control"), f"handoff {name} control payload")
        if (
            row.get("artifact_path") != candidate.get("path")
            or row.get("artifact_sha256") != candidate.get("sha256")
            or row.get("artifact_bytes") != candidate.get("bytes")
        ):
            raise _fail(f"candidate payload differs from sealed pair binding: {name}")
        mutation = _require_mapping(row.get("candidate_mutation"), f"handoff {name} mutation")
        if mutation.get("changed_from_admitted_control") is not True:
            raise _fail(f"selected organ is not explicitly marked as a candidate mutation: {name}")
        rollback = _require_mapping(mutation.get("baseline_rollback"), f"handoff {name} rollback")
        if (
            rollback.get("baseline_artifact_path") != control.get("path")
            or rollback.get("baseline_artifact_sha256") != control.get("sha256")
            or rollback.get("baseline_artifact_bytes") != control.get("bytes")
            or rollback.get("rollback_action") != "use the separately admitted baseline tensor; this candidate never overwrites it"
        ):
            raise _fail(f"selected organ rollback does not bind the admitted control: {name}")
        layout = _require_mapping(row.get("layout"), f"handoff {name} layout")
        residual = _require_mapping(layout.get("residual"), f"handoff {name} residual layout")
        if (
            layout.get("family") != "binary_sign_scale_sparse_fp16_residual"
            or layout.get("magic") != "HQ30GR2\x00"
            or _require_int(layout.get("version"), f"handoff {name} wrapper version", positive=True) != 1
            or _require_int(residual.get("selected_count"), f"handoff {name} residual count", positive=True)
            != _require_int(pair.get("residual_count"), f"handoff {name} paired residual count", positive=True)
            or residual.get("index_dtype") != "uint32_little_endian"
            or residual.get("value_dtype") != "float16_little_endian"
        ):
            raise _fail(f"selected organ lacks the exact HQ30GR2 v1 grammar: {name}")
        base_layout = _require_mapping(layout.get("base_layout"), f"handoff {name} base layout")
        if base_layout.get("magic") != "HQ30G1B1" or _require_int(base_layout.get("version"), f"handoff {name} base version", positive=True) != 1:
            raise _fail(f"selected organ has no direct HQ30G1B1 base grammar: {name}")
        discriminator = _require_mapping(mutation.get("source_to_packed_discriminator"), f"handoff {name} source discriminator")
        if discriminator.get("payload_sha256") != candidate.get("sha256"):
            raise _fail(f"source-to-packed discriminator does not bind candidate payload: {name}")
        selected.append(
            {
                "organ": name,
                "shape": list(_require_list(row.get("shape"), f"handoff {name} shape")),
                "candidate_payload": dict(candidate),
                "admitted_scalar_control_payload": dict(control),
                "hq30gr2": {
                    "magic": layout["magic"],
                    "version": layout["version"],
                    "base_layout": dict(base_layout),
                    "base_payload_bytes": layout["base_payload_bytes"],
                    "header_and_shape_bytes": layout["header_and_shape_bytes"],
                    "residual": dict(residual),
                },
                "source_to_packed_discriminator": dict(discriminator),
                "rollback_target": dict(rollback),
            }
        )
    return selected


def _immutable_body(
    target: ScalarParityTarget,
    evidence: Mapping[str, Any],
    packed: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The exact non-timestamped contract used for immutable reuse checks."""

    return {
        "schema": HANDOFF_SCHEMA,
        "status": HANDOFF_STATUS,
        "candidate_root": str(target.root.resolve()),
        "candidate_admission_current_pointer": dict(_require_mapping(evidence.get("candidate_current_pointer"), "handoff admission current")),
        "candidate_native_admission_receipt": dict(_require_mapping(evidence.get("candidate_native_admission_receipt"), "handoff admission receipt")),
        "candidate_manifest": dict(_require_mapping(evidence.get("candidate_manifest"), "handoff manifest")),
        "candidate_source_binding_snapshot": dict(_require_mapping(evidence.get("candidate_source_binding_snapshot"), "handoff source snapshot")),
        "immutable_source_revalidation": dict(_require_mapping(evidence.get("immutable_source_revalidation"), "handoff source revalidation")),
        "scalar_parity_current_pointer": dict(_require_mapping(evidence.get("scalar_parity_current_pointer"), "handoff scalar current")),
        "scalar_parity_receipt": dict(_require_mapping(evidence.get("scalar_parity_receipt"), "handoff scalar receipt")),
        "packed_matvec_parity_current_pointer": dict(_require_mapping(packed.get("packed_matvec_parity_current_pointer"), "handoff packed current")),
        "packed_matvec_parity_receipt": dict(_require_mapping(packed.get("packed_matvec_parity_receipt"), "handoff packed receipt")),
        "admitted_scalar_control": {
            "manifest": dict(_require_mapping(evidence.get("admitted_control_manifest"), "handoff control manifest")),
            "admission_receipt": dict(_require_mapping(evidence.get("admitted_control_admission_receipt"), "handoff control admission")),
        },
        "selected_hq30gr2_organs": [dict(item) for item in selected],
        "hq30gr2_decoder_requirements": {
            "selected_organs_only": list(SELECTED_ORGANS),
            "unselected_tensor_grammar": "HQ30G1B1_v1_direct_packed_control_operator",
            "selected_tensor_grammar": "HQ30GR2_v1_embedded_HQ30G1B1_base_plus_sorted_unique_uint32_little_endian_sparse_float16_little_endian_residual",
            "must_bind_each_candidate_payload_and_embedded_control_to_this_handoff": True,
            "must_verify_source_to_packed_discriminator_before_selected_organ_use": True,
            "must_execute_direct_packed_base_plus_sparse_residual_matvec_without_dense_weight_materialization": True,
            "must_refuse_direct_hq30g1b1_decode_for_hq30gr2_selected_organs": True,
            "must_refuse_hq30gr2_decode_for_plain_hq30g1b1_controls": True,
            "must_not_strip_or_ignore_residual_indices_or_values": True,
            "must_reject_any_unknown_magic_or_version": True,
        },
        "full_model_parity_gates_before_any_candidate_promotion": [
            "revalidate_candidate_manifest_all_frozen_current_pointers_source_snapshot_immutable_source_revalidation_and_baseline_control_at_integration",
            "re-run_selected_gate_and_up_cpu_reference_parity_and_device_parity_with_exact_hq30gr2_fallback_refusal",
            "re-run_gate_up_swiglu_and_affected_moe_expert_layer_parity_for_early_middle_late_and_sensitive_layers",
            "re-run_complete_48_layer_native_decoder_final_norm_lm_head_sampler_and_multi_token_autoregression",
            "re-run_fresh_hcli_context_kv_restart_agent_os_and_clean_complete_token_tps_gates",
            "keep_candidate_separate_from_the_admitted_baseline_runtime_server_tournament_and_current_pointers_until_every_required_gate_passes",
        ],
        "rollback_target": {
            "action": "use the separately admitted Qwen30 complete-gravity control; do not overwrite or repoint it",
            "baseline_manifest": dict(_require_mapping(evidence.get("admitted_control_manifest"), "handoff rollback manifest")),
            "baseline_admission_receipt": dict(_require_mapping(evidence.get("admitted_control_admission_receipt"), "handoff rollback admission")),
            "candidate_replacement_forbidden_until_all_fresh_full_model_gates_pass": True,
        },
        "isolation": {
            "candidate_root_only": True,
            "baseline_runtime_server_tournament_and_current_pointers_untouched": True,
            "candidate_metal_and_full_runtime_not_started": True,
            "this_handoff_cannot_promote_runtime_server_tournament_or_tg": True,
        },
        "claim_boundary": {
            "candidate_native_admission_and_cpu_adapter_proofs_bound": True,
            "not_a_full_qwen_layer_decoder_generation_hcli_or_tps_result": True,
            "not_a_capability_tg_agent_os_or_tournament_qualification": True,
        },
    }


def _validate_existing_handoff(
    path: Path,
    expected_body: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, metadata = _read_sealed(path, "existing HQ30GR2 runtime integration handoff")
    stable = dict(document)
    stable.pop("recorded_at", None)
    stable.pop("seal_sha256", None)
    if stable != dict(expected_body):
        raise _fail("existing HQ30GR2 runtime integration handoff differs from the frozen candidate authority")
    return document, metadata


def run_once(target: ScalarParityTarget) -> dict[str, Any]:
    """Create/reuse one immutable handoff; never creates a mutable selector."""

    try:
        evidence = _validate_evidence(target)
    except ScalarParityError as exc:
        raise _fail(str(exc)) from exc
    packed = _validate_packed_current(target, evidence)
    selected = _selected_hq30gr2_contract(target, evidence)
    body = _immutable_body(target, evidence, packed, selected)
    path = _handoff_path(target, evidence)
    if path.exists():
        document, metadata = _validate_existing_handoff(path, body)
        return {
            "status": HANDOFF_STATUS,
            "handoff_path": str(path),
            "handoff_document_sha256": metadata["document_sha256"],
            "handoff_seal_sha256": document["seal_sha256"],
            "reused": True,
        }
    document = seal({**body, "recorded_at": _utc_now()})
    try:
        shared._write_immutable_json(path, document, "HQ30GR2 runtime integration handoff")
    except shared.CompleteBinaryAdmissionError as exc:
        raise _fail(str(exc)) from exc
    verified, metadata = _validate_existing_handoff(path, body)
    return {
        "status": HANDOFF_STATUS,
        "handoff_path": str(path),
        "handoff_document_sha256": metadata["document_sha256"],
        "handoff_seal_sha256": verified["seal_sha256"],
        "reused": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("once",), nargs="?", default="once")
    parser.add_argument("--root", type=Path, default=DEFAULT_TARGET.root)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_TARGET.baseline_root)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = ScalarParityTarget(root=args.root.expanduser().resolve(), baseline_root=args.baseline_root.expanduser().resolve())
    try:
        result = run_once(target)
    except IntegrationHandoffError as exc:
        print(f"BLOCKED_HQ30GR2_RUNTIME_INTEGRATION_HANDOFF: {exc}")
        return 2
    import json

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
