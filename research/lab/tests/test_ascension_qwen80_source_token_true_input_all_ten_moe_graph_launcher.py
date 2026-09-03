from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from lab.operators import ascension_qwen80_source_token_true_input_all_ten_moe_graph_launcher as launcher
from lab.operators import ascension_qwen80_source_token_all_ten_true_moe_outer_preflight as outer_preflight


def _sha(index: int) -> str:
    return f"{index:064x}"


def _evidence(name: str, index: int) -> dict[str, object]:
    return {
        "path": f"/tmp/{name}.json",
        "present": True,
        "bytes": index + 1,
        "sha256": _sha(index),
    }


def _context() -> launcher.PreflightContext:
    route_ids = tuple(range(10))
    return launcher.PreflightContext(
        outer_preflight={},
        outer_preflight_evidence=_evidence("outer-preflight", 1),
        outer_preflight_seal_sha256=_sha(2),
        manifest=_evidence("manifest", 3),
        admission_current=_evidence("admission-current", 4),
        source_authority=_evidence("source-authority", 5),
        first_residual=_evidence("first-residual", 6),
        typed_bridge=_evidence("typed-bridge", 7),
        fixed_suffix=_evidence("fixed-suffix", 8),
        manifest_seal_sha256=_sha(9),
        admission_pointer_seal_sha256=_sha(10),
        admission_receipt_seal_sha256=_sha(11),
        source_authority_seal_sha256=_sha(12),
        first_residual_seal_sha256=_sha(13),
        first_residual_output_sha256=_sha(14),
        typed_bridge_seal_sha256=_sha(15),
        route_ids=route_ids,
        route_weights=tuple([0.1] * launcher.TOP_K),
        probe_binary=_evidence("probe", 16),
        shader_source=_evidence("shader", 17),
        metal_registry={**_evidence("metal-registry", 18), "registered": True, "registry_append_required": False},
    )


def _child_preflight(context: launcher.PreflightContext) -> dict[str, object]:
    prefix = launcher._child_sealed_binding(context.first_residual, context.first_residual_seal_sha256)
    prefix["output_sha256"] = context.first_residual_output_sha256
    return {
        "schema": launcher.EXPECTED_CHILD_SCHEMA,
        "status": launcher.EXPECTED_CHILD_PREFLIGHT_STATUS,
        "mode": "preflight",
        "outer_preflight_binding": launcher._child_sealed_binding(
            context.outer_preflight_evidence, context.outer_preflight_seal_sha256
        ),
        "source_token_route_authority_binding": {
            **launcher._child_sealed_binding(context.source_authority, context.source_authority_seal_sha256),
            "route_ids": list(context.route_ids),
            "normalized_weights": list(context.route_weights),
        },
        "typed_bridge_binding": launcher._child_sealed_binding(
            context.typed_bridge, context.typed_bridge_seal_sha256
        ),
        "first_residual_antecedent": prefix,
        "fixed_suffix_contract_binding": launcher._child_fixed_binding(context.fixed_suffix),
        "same_command_graph_contract": {
            "source_token_id": launcher.SOURCE_TOKEN_ID,
            "zero_l0_state_required": True,
            "prefix_dispatches": launcher.PREFIX_DISPATCHES,
            "suffix_dispatches": launcher.SUFFIX_DISPATCHES,
            "total_dispatches": launcher.TOTAL_DISPATCHES,
            "route_guard_required": True,
            "all_ten_route_shared_routed_sum_second_residual_readbacks_required": True,
        },
        "claim_boundary": {
            "metal_device_or_dispatch_performed": False,
            "lease_issued": False,
            "legacy_fixture_router_or_plan_accepted": False,
            "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": True,
        },
    }


def _launch_context(context: launcher.PreflightContext) -> launcher.LaunchContext:
    return launcher.LaunchContext(
        proof=launcher.ProofContext(
            preflight=context,
            proof={},
            proof_evidence=_evidence("proof", 70),
            proof_seal_sha256=_sha(71),
        ),
        lease_receipt=_evidence("lease", 72),
        lease_seal_sha256=_sha(73),
        lease_id=_sha(74),
    )


def _valid_metal_inner_receipt(
    context: launcher.PreflightContext,
    launch_context: launcher.LaunchContext,
    outer_launch_authority: dict[str, object],
) -> dict[str, object]:
    outer_seal = str(outer_launch_authority["seal_sha256"])
    outer_binding = launcher._child_sealed_binding(outer_launch_authority, outer_seal)
    prefix = launcher._child_sealed_binding(
        context.first_residual, context.first_residual_seal_sha256
    )
    prefix["output_sha256"] = context.first_residual_output_sha256
    typed = launcher._child_sealed_binding(context.typed_bridge, context.typed_bridge_seal_sha256)
    typed.update(
        {
            "schema": outer_preflight.SOURCE_TYPED_BRIDGE_SCHEMA,
            "status": outer_preflight.SOURCE_TYPED_BRIDGE_STATUS,
        }
    )
    lease_binding = launcher._child_sealed_binding(
        launch_context.lease_receipt, launch_context.lease_seal_sha256
    )
    lease_binding["lease_id"] = launch_context.lease_id
    witnesses = [
        {
            "wave_index": index,
            "expert_id": context.route_ids[index],
            "normalized_weight": context.route_weights[index],
            "elements": 2_048,
            "max_abs_error": 0.0,
            "output_sha256": _sha(index + 100),
        }
        for index in range(launcher.TOP_K)
    ]
    return {
        "schema": launcher.EXPECTED_CHILD_SCHEMA,
        "status": launcher.EXPECTED_CHILD_METAL_STATUS,
        "mode": "metal",
        "metal_device_or_dispatch_performed": True,
        "component_only": True,
        "complete_layer_or_token_performed": False,
        "artifact_binding": {
            "manifest_document_sha256": context.manifest["sha256"],
            "manifest_seal_sha256": context.manifest_seal_sha256,
            "admission_pointer_seal_sha256": context.admission_pointer_seal_sha256,
            "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
            "layer": 0,
            "linear_state_slot": 0,
        },
        "outer_preflight_binding": launcher._child_sealed_binding(
            context.outer_preflight_evidence, context.outer_preflight_seal_sha256
        ),
        "outer_launch_authority_binding": outer_binding,
        "source_token_route_authority_binding": {
            **launcher._child_sealed_binding(
                context.source_authority, context.source_authority_seal_sha256
            ),
            "route_ids": list(context.route_ids),
            "normalized_weights": list(context.route_weights),
        },
        "typed_bridge_binding": typed,
        "first_residual_antecedent": prefix,
        "fixed_suffix_contract_binding": launcher._child_fixed_binding(context.fixed_suffix),
        "same_command_graph": {
            "source_token_id": launcher.SOURCE_TOKEN_ID,
            "same_command_graph_required": True,
            "same_command_graph_retained": True,
            "prefix_dispatches": launcher.PREFIX_DISPATCHES,
            "suffix_dispatches": launcher.SUFFIX_DISPATCHES,
            "total_dispatches": launcher.TOTAL_DISPATCHES,
            "command_buffer_fenced_once_after_prefix_and_suffix": True,
            "first_residual_matches_sealed_prefix_antecedent": True,
        },
        "route_guard_readback": {
            "value": 1,
            "passed": True,
            "observed_ids": list(context.route_ids),
            "expected_ids": list(context.route_ids),
            "observed_weights": list(context.route_weights),
            "expected_weights": list(context.route_weights),
        },
        "readback_parity": {
            "postnorm_max_abs_error": 0.0,
            "postnorm": {"max_abs_error": 0.0},
            "router_logits_max_abs_error": 0.0,
            "router_logits": {"max_abs_error": 0.0},
            "all_ten_route_witnesses": witnesses,
            "shared_expert_max_abs_error": 0.0,
            "shared_expert": {"max_abs_error": 0.0},
            "routed_sum_max_abs_error": 0.0,
            "routed_sum": {"max_abs_error": 0.0},
            "second_residual_max_abs_error": 0.0,
            "second_residual": {"max_abs_error": 0.0},
        },
        "metal_execution_policy": {
            "strict_math_required": True,
            "timing_or_benchmarking_allowed": False,
            "complete_layer_or_token_allowed": False,
            "tps_or_tg_claim_allowed": False,
            "lease_binding": lease_binding,
            "outer_launch_authority_binding": outer_binding,
        },
        "durable_capture": {
            "receipt_written_last_is_completion_marker": True,
            "outer_reaped_capture_required": True,
            "replay_guarded": True,
            "outer_reaper_binding": {
                "lease_id": launch_context.lease_id,
                "outer_launch_authority": outer_binding,
            },
        },
        "claim_boundary": {
            "source_token_l0_true_moe_component_only": True,
            "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": True,
            "no_watcher_or_server_started": True,
        },
    }


@pytest.mark.parametrize("legacy_argument", ["--router-receipt", "--router-outer-receipt", "--route-plan", "--fixed-abi-contract"])
def test_legacy_provenance_arguments_are_refused_before_argument_parsing(
    legacy_argument: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code = launcher.main([f"{legacy_argument}=/tmp/legacy.json"])

    assert code == 2
    assert legacy_argument in capsys.readouterr().out


def test_child_preflight_refuses_typed_authority_mismatch() -> None:
    context = _context()
    document = _child_preflight(context)
    bad = deepcopy(document)
    bad["typed_bridge_binding"]["seal_sha256"] = _sha(99)  # type: ignore[index]

    with pytest.raises(launcher.SourceTokenTrueInputAllTenMoeLauncherError, match="typed bridge"):
        launcher._validate_child_preflight_document(bad, context)


def test_child_preflight_accepts_distinct_source_token_contract() -> None:
    context = _context()
    document = _child_preflight(context)

    launcher._validate_child_preflight_document(document, context)
    assert document["fixed_suffix_contract_binding"]["schema"] == outer_preflight.FIXED_ABI_SCHEMA  # type: ignore[index]


def test_outer_launch_authority_binds_reaper_proof_probe_and_planned_capture() -> None:
    preflight = _context()
    proof = launcher.ProofContext(
        preflight=preflight,
        proof={},
        proof_evidence=_evidence("child-preflight-proof", 30),
        proof_seal_sha256=_sha(31),
    )
    launch_context = launcher.LaunchContext(
        proof=proof,
        lease_receipt=_evidence("lease", 32),
        lease_seal_sha256=_sha(33),
        lease_id=_sha(35),
    )
    config = launcher.LaunchConfig(
        base=launcher.BaseInputs(
            manifest=Path("/tmp/manifest.json"),
            admission_current=Path("/tmp/admission.json"),
            source_token_route_authority=Path("/tmp/authority.json"),
            first_residual_receipt=Path("/tmp/prefix.json"),
            typed_bridge_receipt=Path("/tmp/bridge.json"),
            fixed_suffix_contract=Path("/tmp/fixed.json"),
        ),
        probe_bin=Path("/tmp/ascension_qwen80_source_token_all_ten_true_moe_graph_device"),
        preflight_proof=Path("/tmp/proof.json"),
        lease_receipt=Path("/tmp/lease.json"),
        capture_dir=Path("/tmp/capture"),
        workers=2,
        timeout_seconds=60.0,
    )

    document = launcher._outer_launch_authority_document(
        config, launch_context, identity=_sha(34), capture=Path("/tmp/capture")
    )

    assert document["schema"] == launcher.OUTER_LAUNCH_AUTHORITY_SCHEMA
    assert document["lease_receipt"] == launch_context.lease_receipt
    assert document["lease_receipt_seal_sha256"] == launch_context.lease_seal_sha256
    assert document["outer_preflight"] == preflight.outer_preflight_evidence
    assert document["outer_preflight_seal_sha256"] == preflight.outer_preflight_seal_sha256
    assert document["preflight_proof"] == {
        **proof.proof_evidence,
        "seal_sha256": proof.proof_seal_sha256,
    }
    source = document["source_binding"]
    assert source["probe_binary"] == preflight.probe_binary
    assert source["preflight_proof"] == {
        **proof.proof_evidence,
        "seal_sha256": proof.proof_seal_sha256,
    }
    assert source["child_preflight_proof_binding"] == launcher._child_sealed_binding(
        proof.proof_evidence, proof.proof_seal_sha256
    )
    assert document["lease_id"] == _sha(35)
    assert document["planned_capture"]["inner_capture_dir"] == "/tmp/capture/inner"
    assert document["planned_capture"]["workers"] == 2
    assert document["planned_outer_capture_dir"] == "/tmp/capture"
    assert document["planned_inner_capture_dir"] == "/tmp/capture/inner"
    assert document["workers"] == 2
    assert document["execution_policy"]["quiet_qwen80_device_lease"] is True
    assert document["lifecycle"]["outer_reaped_capture_required"] is True
    assert document["watcher_coordination"]["watcher_hold_must_remain_active"] is True
    assert document["outer_reaper"]["outer_reaps_child_before_terminal_receipt"] is True


def test_preflight_lineage_allows_only_a_historical_current_pointer_reseal() -> None:
    base = {
        "manifest": _evidence("manifest", 40),
        "admission_current": _evidence("admission-current", 41),
        "admission_receipt": _evidence("admission-receipt", 42),
        "source_token_route_authority": _evidence("authority", 43),
        "first_residual_receipt": _evidence("prefix", 44),
        "typed_bridge_receipt": _evidence("bridge", 45),
        "fixed_suffix_contract": _evidence("fixed", 46),
        "manifest_seal_sha256": _sha(47),
        "admission_pointer_seal_sha256": _sha(48),
        "admission_receipt_seal_sha256": _sha(49),
        "source_token_route_authority_seal_sha256": _sha(50),
        "first_residual_receipt_seal_sha256": _sha(51),
        "typed_bridge_receipt_seal_sha256": _sha(52),
    }
    resealed = deepcopy(base)
    resealed["admission_current"]["sha256"] = _sha(53)  # type: ignore[index]
    resealed["admission_pointer_seal_sha256"] = _sha(54)

    assert launcher._same_source_binding(base, resealed)
    resealed["admission_receipt_seal_sha256"] = _sha(55)
    assert not launcher._same_source_binding(base, resealed)


def test_outer_inner_success_schema_requires_top_level_authority_and_exact_parity_mirrors() -> None:
    context = _context()
    launch_context = _launch_context(context)
    outer_launch_authority = {**_evidence("outer-launch-authority", 80), "seal_sha256": _sha(81)}
    receipt = _valid_metal_inner_receipt(context, launch_context, outer_launch_authority)

    launcher._validate_inner_receipt(receipt, launch_context, outer_launch_authority)

    missing_authority = deepcopy(receipt)
    del missing_authority["outer_launch_authority_binding"]
    with pytest.raises(launcher.SourceTokenTrueInputAllTenMoeLauncherError, match="outer launch authority"):
        launcher._validate_inner_receipt(missing_authority, launch_context, outer_launch_authority)

    mismatched_scalar = deepcopy(receipt)
    mismatched_scalar["readback_parity"]["routed_sum_max_abs_error"] = 0.25  # type: ignore[index]
    with pytest.raises(launcher.SourceTokenTrueInputAllTenMoeLauncherError, match="does not exactly match"):
        launcher._validate_inner_receipt(mismatched_scalar, launch_context, outer_launch_authority)
