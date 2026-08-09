from __future__ import annotations

from lab.operators import ascension_qwen80_source_token_all_ten_true_moe_bridge as bridge


def test_route_weight_comparison_accepts_f32_material_against_f64_authority() -> None:
    authority = [0.291953444480896] + [0.07867183950212267] * 9
    material = [0.29195344] + [0.07867184] * 9
    assert bridge._same_route_weights(material, authority)


def test_route_weight_comparison_refuses_identity_or_numerical_drift() -> None:
    expected = [0.1] * 10
    assert not bridge._same_route_weights(expected[:-1], expected)
    observed = expected.copy()
    observed[0] = 0.10001
    assert not bridge._same_route_weights(observed, expected)


def test_material_validation_uses_authority_document_for_plan_and_evidence_for_file_identity() -> None:
    digest = "a" * 64
    manifest = {"path": "/manifest", "present": True, "bytes": 1, "sha256": digest}
    admission = {"path": "/admission", "present": True, "bytes": 2, "sha256": digest}
    authority_evidence = {"path": "/authority", "present": True, "bytes": 3, "sha256": digest}
    prefix = {"path": "/prefix", "present": True, "bytes": 4, "sha256": digest}
    ids = list(range(10))
    weights = [0.1] * 10
    material = {
        "schema": bridge.MATERIAL_SCHEMA,
        "status": bridge.MATERIAL_STATUS,
        "source_binding": {
            "manifest": manifest,
            "admission_current": admission,
            "admission_receipt": admission,
            "source_token_route_authority": authority_evidence,
            "first_residual_receipt": prefix,
            "manifest_seal_sha256": digest,
            "admission_receipt_seal_sha256": digest,
            "source_token_route_authority_seal_sha256": digest,
            "first_residual_receipt_seal_sha256": digest,
            "source_audit_seal_sha256": digest,
        },
        "typed_bridge": {
            "layer": 0,
            "source_token_id": 1,
            "route_count": 10,
            "first_residual_elements": 2048,
            "same_command_graph_required": True,
            "first_residual_receipt_seal_sha256": digest,
            "source_token_route_authority_seal_sha256": digest,
            "first_residual_output_sha256": digest,
            "compact_section_sha256": {
                name: digest
                for name in ("gate_scales", "gate_signs", "up_scales", "up_signs", "down_scales", "down_signs")
            },
        },
        "route_authority": {
            "ids": ids,
            "normalized_weights": weights,
            "wave_count": 10,
            "all_thirty_wave_payloads_use_admission_verified_immutable_snapshots": True,
        },
        "artifact_scan": {
            "complete_artifact_admission_performed_once": True,
            "catalog_reused_for_source_token_all_ten_bridge": True,
            "raw_bf16_or_safetensors_opened": False,
        },
        "claim_boundary": {
            "cpu_source_token_bridge_material_only": True,
            "metal_device_or_dispatch_performed": False,
            "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": True,
        },
    }
    authority_document = {
        "source_token_plan": {
            "source_token_router_evidence": {
                "source_stable_route_ids": ids,
                "source_stable_normalized_weights": weights,
            }
        }
    }

    parsed_bridge, parsed_route = bridge._validate_material(
        material,
        manifest=manifest,
        manifest_seal=digest,
        admission=admission,
        admission_receipt=admission,
        admission_receipt_seal=digest,
        source_authority_evidence=authority_evidence,
        source_authority_document=authority_document,
        source_authority_seal=digest,
        prefix=prefix,
        prefix_seal=digest,
    )

    assert parsed_bridge["source_token_id"] == 1
    assert parsed_route["ids"] == ids
