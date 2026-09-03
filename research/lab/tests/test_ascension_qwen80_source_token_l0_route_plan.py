from __future__ import annotations

from copy import deepcopy

import pytest

from lab.operators import ascension_qwen80_source_token_l0_route_plan as route_plan


def _sha(index: int) -> str:
    return f"{index:064x}"


def _evidence(label: str, index: int) -> dict[str, object]:
    return {"path": f"/tmp/{label}.json", "present": True, "bytes": index + 1, "sha256": _sha(index)}


def _projection(name: str, shape: list[int], index: int) -> dict[str, object]:
    return {
        "tensor_name": name,
        "shape": shape,
        "elements": shape[0] * shape[1],
        "artifact_path": f"/tmp/{name}.{index}.bin",
        "artifact_bytes": 32,
        "artifact_sha256": _sha(100 + index),
        "source_dtype": "BF16",
        "source_shard": "model-00001-of-00040.safetensors",
        "source_shard_sha256": _sha(700),
        "layout": {
            "magic": "HQ30G1B1",
            "version": 1,
            "group_size": 128,
            "scale_dtype": "float16",
            "sign_bit_order": "little",
        },
        "payload_opened_by_this_plan": False,
    }


def _fixture() -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object], str, str]:
    manifest = _evidence("manifest", 1)
    admission = _evidence("admission-current", 2)
    immutable_admission = _evidence("admission", 3)
    prefix = _evidence("prefix", 4)
    old = _evidence("old-plan", 5)
    prefix_seal = _sha(6)
    admission_seal = _sha(7)
    prefix_document: dict[str, object] = {
        "first_residual_output": {
            "layer": 0,
            "linear_state_slot": 0,
            "elements": 2048,
            "same_command_graph_required": True,
            "sha256": _sha(8),
        },
        "source_binding": {"cpu_baseline_receipt": _evidence("cpu-baseline", 9)},
    }
    ids = list(range(10))
    weights = [0.1] * 10
    waves: list[dict[str, object]] = []
    for index, expert in enumerate(ids):
        stem = f"model.layers.0.mlp.experts.{expert}"
        waves.append(
            {
                "wave_index": index,
                "layer": 0,
                "expert_id": expert,
                "normalized_weight": weights[index],
                "route_execution_status": "NOT_EXECUTED_SOURCE_TOKEN_BOUND_PLAN_ONLY",
                "route_delta_materialized": False,
                "route_weight_applied": False,
                "gate": _projection(f"{stem}.gate_proj.weight", [512, 2048], index * 3),
                "up": _projection(f"{stem}.up_proj.weight", [512, 2048], index * 3 + 1),
                "down": _projection(f"{stem}.down_proj.weight", [2048, 512], index * 3 + 2),
            }
        )
    plan: dict[str, object] = {
        "schema": route_plan.SOURCE_PLAN_SCHEMA,
        "status": route_plan.SOURCE_PLAN_STATUS,
        "layer": 0,
        "source_input_provenance": {
            "source_token_id": 1,
            "same_input_state_identity_required": True,
            "prefix_outer_receipt": prefix,
            "prefix_outer_receipt_seal_sha256": prefix_seal,
            "strict_metal_prefix_first_residual_sha256": _sha(8),
            "cpu_baseline_receipt": prefix_document["source_binding"]["cpu_baseline_receipt"],  # type: ignore[index]
            "input_hidden_f32le_sha256": _sha(10),
            "cpu_first_residual_f32le_sha256": _sha(11),
            "zero_conv_state_f32le_sha256": _sha(12),
            "zero_recurrent_state_f32le_sha256": _sha(13),
        },
        "source_token_router_evidence": {
            "derived_from_direct_packed_source_token_l0_cpu_oracle": True,
            "router_component_only": True,
            "post_attention_normalized_hidden_f32le_sha256": _sha(14),
            "router_logits_f32le_sha256": _sha(15),
            "source_stable_route_ids": ids,
            "source_stable_normalized_weights": weights,
        },
        "manifest_descriptor_inventory": {
            "inventory_document_sha256": manifest["sha256"],
            "manifest_schema": route_plan.MANIFEST_SCHEMA,
            "manifest_seal_sha256": _sha(16),
            "resolved_route_tensor_count": 30,
            "payload_opened_by_this_plan": False,
        },
        "deterministic_waves": waves,
        "rawls_real_all_ten_provenance_gate": {
            "all_ten_source_bindings_complete": True,
            "expected_layer": 0,
            "route_order": ids,
            "normalized_weights": weights,
            "execution_receipt_required_for_each_wave": True,
            "rejects_tensor_substitution": True,
            "rejects_route_reorder": True,
            "rejects_duplicate_experts": True,
            "rejects_missing_tensor_or_weight": True,
        },
        "route_execution_performed": False,
        "route_combine_performed": False,
        "shared_expert_performed": False,
        "residual_combine_performed": False,
        "metal_device_or_dispatch_performed": False,
        "model_execution_performed": False,
        "hcli_execution_performed": False,
        "tps_or_tg_measurement_performed": False,
        "complete_layer_or_decoder_claim_earned": False,
    }
    material: dict[str, object] = {
        "schema": route_plan.MATERIAL_SCHEMA,
        "status": route_plan.MATERIAL_STATUS,
        "source_binding": {
            "manifest": manifest,
            "admission_current": admission,
            "admission_receipt": immutable_admission,
            "first_residual_outer_receipt": prefix,
            "historical_fixture_route_plan": old,
            "manifest_seal_sha256": _sha(16),
            "admission_receipt_seal_sha256": admission_seal,
            "source_audit_seal_sha256": _sha(17),
            "source_revision": "a" * 40,
        },
        "source_token_plan": plan,
        "fixture_divergence": {
            "old_route_plan": old,
            "conclusion": "the fixture-derived route plan is prohibited from driving the source-token true-MoE graph",
        },
        "artifact_scan": {
            "complete_artifact_admission_performed_once": True,
            "catalog_reused_for_embedding_mixer_router_and_all_thirty_descriptors": True,
            "raw_bf16_or_safetensors_opened": False,
        },
        "claim_boundary": {
            "cpu_discriminator_and_descriptor_plan_only": True,
            "metal_device_or_dispatch_performed": False,
            "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": True,
        },
    }
    return material, manifest, admission, immutable_admission, prefix, old, prefix_seal, admission_seal, prefix_document


def _validate(material: dict[str, object]) -> dict[str, object]:
    material, manifest, admission, immutable_admission, prefix, old, prefix_seal, admission_seal, prefix_document = _fixture()
    return route_plan._validate_material(
        material,
        manifest=manifest,
        manifest_seal=_sha(16),
        admission=admission,
        admission_receipt=immutable_admission,
        admission_receipt_seal=admission_seal,
        prefix=prefix,
        prefix_seal=prefix_seal,
        prefix_document=prefix_document,
        old_plan=old,
    )


def test_material_accepts_exact_source_token_all_ten_plan() -> None:
    plan = _validate({})
    assert plan["schema"] == route_plan.SOURCE_PLAN_SCHEMA
    assert len(plan["deterministic_waves"]) == 10


def test_material_refuses_duplicate_projection_payload() -> None:
    material, manifest, admission, immutable_admission, prefix, old, prefix_seal, admission_seal, prefix_document = _fixture()
    waves = material["source_token_plan"]["deterministic_waves"]  # type: ignore[index]
    waves[1]["gate"]["artifact_sha256"] = waves[0]["gate"]["artifact_sha256"]  # type: ignore[index]
    with pytest.raises(route_plan.SourceTokenRoutePlanError, match="reuses a projection"):
        route_plan._validate_material(
            material,
            manifest=manifest,
            manifest_seal=_sha(16),
            admission=admission,
            admission_receipt=immutable_admission,
            admission_receipt_seal=admission_seal,
            prefix=prefix,
            prefix_seal=prefix_seal,
            prefix_document=prefix_document,
            old_plan=old,
        )


def test_material_accepts_historical_pointer_reseal_with_stable_immutable_admission() -> None:
    material, manifest, admission, immutable_admission, prefix, old, prefix_seal, admission_seal, prefix_document = _fixture()
    material["source_binding"]["admission_current"]["sha256"] = _sha(99)  # type: ignore[index]
    admission_current = deepcopy(admission)
    admission_current["sha256"] = _sha(98)
    route_plan._validate_material(
        material,
        manifest=manifest,
        manifest_seal=_sha(16),
        admission=admission_current,
        admission_receipt=immutable_admission,
        admission_receipt_seal=admission_seal,
        prefix=prefix,
        prefix_seal=prefix_seal,
        prefix_document=prefix_document,
        old_plan=old,
    )


def test_material_refuses_true_immutable_admission_drift() -> None:
    material, manifest, admission, immutable_admission, prefix, old, prefix_seal, admission_seal, prefix_document = _fixture()
    with pytest.raises(route_plan.SourceTokenRoutePlanError, match="admission receipt"):
        route_plan._validate_material(
            material,
            manifest=manifest,
            manifest_seal=_sha(16),
            admission=admission,
            admission_receipt={**immutable_admission, "sha256": _sha(99)},
            admission_receipt_seal=admission_seal,
            prefix=prefix,
            prefix_seal=prefix_seal,
            prefix_document=prefix_document,
            old_plan=old,
        )
