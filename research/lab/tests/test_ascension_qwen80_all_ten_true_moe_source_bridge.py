from __future__ import annotations

import pytest

from lab.operators import ascension_qwen80_all_ten_true_moe_source_bridge as bridge


def _sha(ch: str) -> str:
    return ch * 64


def _evidence(tag: str) -> dict[str, object]:
    return {"path": f"/tmp/{tag}.json", "present": True, "bytes": 1, "sha256": _sha(tag[0])}


def _material() -> dict[str, object]:
    manifest = _evidence("manifest")
    admission = _evidence("admission")
    router = _evidence("router")
    router_outer = _evidence("router-outer")
    route_plan = _evidence("route-plan")
    first = _evidence("first")
    return {
        "schema": bridge.MATERIAL_SCHEMA,
        "status": bridge.MATERIAL_STATUS,
        "source_binding": {
            "manifest": manifest,
            "admission_current": admission,
            "router_receipt": router,
            "router_outer_receipt": router_outer,
            "route_plan": route_plan,
            "first_residual_receipt": first,
            "first_residual_cpu_baseline": _evidence("baseline"),
            "manifest_seal_sha256": _sha("m"),
            "admission_pointer_seal_sha256": _sha("c"),
            "admission_receipt_seal_sha256": _sha("a"),
            "router_outer_receipt_seal_sha256": _sha("o"),
        },
        "typed_bridge": {
            "layer": 0,
            "route_count": 10,
            "first_residual_elements": 2048,
            "same_command_graph_required": True,
            "first_residual_output_sha256": _sha("f"),
            "first_residual_receipt_seal_sha256": _sha("s"),
            "compact_section_sha256": {
                "gate_scales": _sha("1"),
                "gate_signs": _sha("2"),
                "up_scales": _sha("3"),
                "up_signs": _sha("4"),
                "down_scales": _sha("5"),
                "down_signs": _sha("6"),
            },
        },
        "route_authority": {
            "ids": list(range(10)),
            "normalized_weights": [0.1] * 10,
            "wave_count": 10,
            "all_thirty_wave_payloads_use_admission_verified_immutable_snapshots": True,
        },
        "artifact_scan": {
            "complete_artifact_admission_performed_once": True,
            "catalog_reused_for_all_ten_source_bridge": True,
            "raw_bf16_or_safetensors_opened": False,
        },
        "claim_boundary": {
            "metal_device_or_dispatch_performed": False,
            "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": True,
            "material_requires_separate_sealed_component_lease_and_outer_reaped_capture": True,
        },
    }


def test_material_validator_accepts_exact_cpu_only_bridge() -> None:
    material = _material()
    source = material["source_binding"]
    assert isinstance(source, dict)
    typed = bridge._validate_material(
        material,
        manifest=source["manifest"],
        manifest_seal=_sha("m"),
        admission=source["admission_current"],
        admission_pointer_seal=_sha("p"),
        admission_receipt_seal=_sha("a"),
        router=source["router_receipt"],
        router_outer=source["router_outer_receipt"],
        router_outer_seal=_sha("o"),
        route_plan=source["route_plan"],
        route_ids=tuple(range(10)),
        route_weights=(0.1,) * 10,
        first_residual=source["first_residual_receipt"],
        first_residual_seal=_sha("s"),
        first_residual_output_sha256=_sha("f"),
    )
    assert typed["route_count"] == 10


def test_material_validator_refuses_device_claim_or_missing_section() -> None:
    material = _material()
    source = material["source_binding"]
    assert isinstance(source, dict)
    material["claim_boundary"]["metal_device_or_dispatch_performed"] = True  # type: ignore[index]
    with pytest.raises(bridge.SourceBridgeSealError, match="claim boundary"):
        bridge._validate_material(
            material,
            manifest=source["manifest"],
            manifest_seal=_sha("m"),
            admission=source["admission_current"],
            admission_pointer_seal=_sha("p"),
            admission_receipt_seal=_sha("a"),
            router=source["router_receipt"],
            router_outer=source["router_outer_receipt"],
            router_outer_seal=_sha("o"),
            route_plan=source["route_plan"],
            route_ids=tuple(range(10)),
            route_weights=(0.1,) * 10,
            first_residual=source["first_residual_receipt"],
            first_residual_seal=_sha("s"),
            first_residual_output_sha256=_sha("f"),
        )


def test_material_validator_accepts_historical_pointer_reseal_with_stable_admission() -> None:
    material = _material()
    source = material["source_binding"]
    assert isinstance(source, dict)
    historical_admission = source["admission_current"]
    assert isinstance(historical_admission, dict)
    historical_admission["sha256"] = _sha("d")
    historical_admission["bytes"] = 99
    source["admission_pointer_seal_sha256"] = _sha("e")
    bridge._validate_material(
        material,
        manifest=source["manifest"],
        manifest_seal=_sha("m"),
        admission=_evidence("admission"),
        admission_pointer_seal=_sha("f"),
        admission_receipt_seal=_sha("a"),
        router=source["router_receipt"],
        router_outer=source["router_outer_receipt"],
        router_outer_seal=_sha("o"),
        route_plan=source["route_plan"],
        route_ids=tuple(range(10)),
        route_weights=(0.1,) * 10,
        first_residual=source["first_residual_receipt"],
        first_residual_seal=_sha("s"),
        first_residual_output_sha256=_sha("f"),
    )
