"""CPU-only tests for the future Qwen80 48-layer outer launch preparation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from lab.operators import ascension_qwen80_48_layer_command_graph_outer_launcher as launcher
from lab.receipts import seal, verify


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _plan(plan_sha: str) -> dict[str, object]:
    del plan_sha  # The raw SHA is deliberately external to this raw plan body.
    layers = []
    deltanet_slots = []
    gqa_slots = []
    for layer in range(launcher.LAYERS):
        if layer % 4 == 3:
            slot = layer // 4
            domain = "gqa_kv"
            gqa_slots.append({"slot": slot, "layer": layer, "domain": domain})
            mixer = "gqa"
        else:
            slot = layer // 4 * 3 + layer % 4
            domain = "delta_net_conv_and_recurrent"
            deltanet_slots.append({"slot": slot, "layer": layer, "domain": domain})
            mixer = "delta_net"
        layers.append(
            {
                "layer": layer,
                "mixer": mixer,
                "state_slot": {
                    "slot": slot,
                    "layer": layer,
                    "domain": domain,
                    "state_materialized_by_this_plan": False,
                },
            }
        )
    return {
        "schema": launcher.PAYLOAD_PLAN_SCHEMA,
        "status": launcher.PAYLOAD_PLAN_STATUS,
        "source_authority": {
            "model_id": launcher.MODEL_ID,
            "model_key": launcher.MODEL_KEY,
            "source_repository": launcher.SOURCE_REPOSITORY,
            "source_revision": launcher.SOURCE_REVISION,
            "source_config_sha256": launcher.SOURCE_CONFIG_SHA256,
            "descriptor_inventory_seal_sha256": launcher.MANIFEST_SEAL,
            "descriptor_inventory_tensor_count": launcher.TENSOR_COUNT,
            "descriptor_inventory_document_sha256": _sha("inventory-document"),
            "source_config_authority_document_sha256": _sha("config-document"),
            "source_config_authority_seal_sha256": _sha("config-seal"),
        },
        "geometry": {
            "layer_count": launcher.LAYERS,
            "hidden_size": launcher.HIDDEN,
            "experts": launcher.EXPERTS,
            "top_k": launcher.TOP_K,
            "vocab_size": launcher.VOCAB,
            "tokenizer_vocab_size": launcher.TOKENIZER_VOCAB,
            "reserved_lm_head_tail_rows": launcher.TAIL_ROWS,
        },
        "resolved_tensor_binding_count": launcher.TENSOR_COUNT,
        "all_48_layers_scheduled": True,
        "all_descriptors_source_artifact_bound": True,
        "embedding": {"tensor_name": "model.embed_tokens.weight", "shape": [launcher.VOCAB, launcher.HIDDEN]},
        "layers": layers,
        "deltanet_state_slots": deltanet_slots,
        "gqa_state_slots": gqa_slots,
        "terminal_head": {
            "final_norm": {"tensor_name": "model.norm.weight", "shape": [launcher.HIDDEN]},
            "lm_head": {"tensor_name": "lm_head.weight", "shape": [launcher.VOCAB, launcher.HIDDEN]},
            "all_row_lm_head_rows": launcher.VOCAB,
            "tokenizer_addressable_rows": launcher.TOKENIZER_VOCAB,
            "reserved_tail_rows": launcher.TAIL_ROWS,
            "execution_order": [
                "final_rmsnorm",
                "all_row_lm_head",
                "reserved_tail_mask",
                "deterministic_sample",
                "tokenizer_feedback",
            ],
        },
        "claim_boundary": {
            "assembly_authority_only": True,
            "decoder_readiness_report": False,
            "artifact_payload_open_or_scan_performed": False,
            "metal_device_or_dispatch_performed": False,
            "runtime_watcher_registry_server_or_hcli_changed": False,
            "model_execution_performed": False,
            "token_generation_or_feedback_performed": False,
            "tps_or_tg_measured": False,
            "execution_status": "PREPARED_NOT_EXECUTED",
        },
    }


def _binding(plan: dict[str, object], plan_sha: str) -> dict[str, object]:
    source = plan["source_authority"]
    assert isinstance(source, dict)
    return {
        "payload_schedule_plan_schema": launcher.PAYLOAD_PLAN_SCHEMA,
        "payload_schedule_plan_status": launcher.PAYLOAD_PLAN_STATUS,
        "payload_schedule_plan_sha256": plan_sha,
        "model_id": launcher.MODEL_ID,
        "model_key": launcher.MODEL_KEY,
        "source_repository": launcher.SOURCE_REPOSITORY,
        "source_revision": launcher.SOURCE_REVISION,
        "manifest_seal_sha256": launcher.MANIFEST_SEAL,
        "admission_receipt_seal_sha256": launcher.ADMISSION_RECEIPT_SEAL,
        "descriptor_inventory_document_sha256": source["descriptor_inventory_document_sha256"],
        "source_config_authority_document_sha256": source["source_config_authority_document_sha256"],
    }


def _decoder(plan: dict[str, object], plan_sha: str) -> dict[str, object]:
    return seal(
        {
            "schema": launcher.DECODER_SCHEMA,
            "status": launcher.DECODER_STATUS,
            "payload_schedule_binding": _binding(plan, plan_sha),
            "complete_decoder_readiness_earned": True,
            "real_gravity_server_launch_precondition_satisfied": True,
            "input_schema_valid": True,
            "source_artifact_binding_valid": True,
            "exact_48_layer_schedule_valid": True,
            "full_command_graph_device_parity_valid": True,
            "missing_operator_classes_or_layers": [],
        }
    )


def _layers(plan: dict[str, object], plan_sha: str, input_sha: str) -> dict[str, object]:
    rows = []
    for layer in range(launcher.LAYERS):
        rows.append(
            {
                "layer": layer,
                "mixer": "gqa" if layer % 4 == 3 else "delta_net",
                "source_bound": True,
                "artifact_bound": True,
                "full_path": True,
                "device_parity_passed": True,
                "fixture_only": False,
                "component_only": False,
                "synthetic_input": False,
                "fallback_used": False,
                "same_input_provenance_sha256": input_sha,
                "device_parity_receipt_seal_sha256": _sha(f"layer-{layer}"),
            }
        )
    return seal(
        {
            "schema": launcher.LAYER_SCHEMA,
            "status": launcher.LAYER_STATUS,
            "payload_schedule_binding": _binding(plan, plan_sha),
            "source_bound": True,
            "artifact_bound": True,
            "full_48_layer_device_parity_earned": True,
            "same_input_provenance_retained": True,
            "fixture_only": False,
            "component_only": False,
            "synthetic_input": False,
            "fallback_used": False,
            "shadow_model_used": False,
            "same_input_provenance_sha256": input_sha,
            "layers": rows,
        }
    )


def _slot(layer: int, slot: int, domain: str, plan_sha: str) -> dict[str, object]:
    return {
        "slot": slot,
        "layer": layer,
        "domain": domain,
        "device_allocated": True,
        "state_read_write_parity_passed": True,
        "rollback_parity_passed": True,
        "same_plan_bound": True,
        "fixture_only": False,
        "fallback_used": False,
        "payload_schedule_plan_sha256": plan_sha,
        "state_receipt_seal_sha256": _sha(f"state-{domain}-{slot}"),
    }


def _state(plan: dict[str, object], plan_sha: str) -> dict[str, object]:
    deltanet_slots = [
        _slot(slot // 3 * 4 + slot % 3, slot, "delta_net_conv_and_recurrent", plan_sha)
        for slot in range(launcher.DELTANET_LAYERS)
    ]
    gqa_slots = [_slot(slot * 4 + 3, slot, "gqa_kv", plan_sha) for slot in range(launcher.GQA_LAYERS)]
    return seal(
        {
            "schema": launcher.STATE_SCHEMA,
            "status": launcher.STATE_STATUS,
            "payload_schedule_binding": _binding(plan, plan_sha),
            "source_bound": True,
            "artifact_bound": True,
            "real_device_resident_state": True,
            "all_36_deltanet_state_slots_bound": True,
            "all_12_gqa_kv_slots_bound": True,
            "rollback_parity_passed": True,
            "no_cross_session_state_or_kv_leakage": True,
            "fixture_only": False,
            "component_only": False,
            "synthetic_input": False,
            "fallback_used": False,
            "deltanet_slots": deltanet_slots,
            "gqa_slots": gqa_slots,
        }
    )


def _terminal(plan: dict[str, object], plan_sha: str, input_sha: str) -> dict[str, object]:
    return seal(
        {
            "schema": launcher.TERMINAL_SCHEMA,
            "status": launcher.TERMINAL_STATUS,
            "payload_schedule_binding": _binding(plan, plan_sha),
            "source_bound": True,
            "artifact_bound": True,
            "post_48_hidden_device_parity_passed": True,
            "final_rmsnorm_device_parity_passed": True,
            "lm_head_all_rows_device_parity_passed": True,
            "reserved_tail_mask_applied_before_sample": True,
            "deterministic_sample_and_feedback_executed": True,
            "sampled_token_is_tokenizer_addressable": True,
            "fixture_only": False,
            "component_only": False,
            "synthetic_input": False,
            "fallback_used": False,
            "shadow_model_used": False,
            "post_48_hidden_shape": [launcher.HIDDEN],
            "lm_head_rows": launcher.VOCAB,
            "tokenizer_addressable_rows": launcher.TOKENIZER_VOCAB,
            "reserved_tail_rows": launcher.TAIL_ROWS,
            "same_input_provenance_sha256": input_sha,
        }
    )


def _evidence(plan_sha: str = _sha("static-plan")) -> dict[str, object]:
    plan = _plan(plan_sha)
    input_sha = _sha("real-input")
    return {
        "schema": launcher.INPUT_SCHEMA,
        "payload_schedule_plan": plan,
        "payload_schedule_plan_sha256": plan_sha,
        "decoder_readiness": _decoder(plan, plan_sha),
        "layer_parity_receipt": _layers(plan, plan_sha, input_sha),
        "state_rollback_receipt": _state(plan, plan_sha),
        "terminal_receipt": _terminal(plan, plan_sha, input_sha),
    }


def _condition(report: dict[str, object], name: str) -> dict[str, object]:
    return next(item for item in report["conditions"] if item["name"] == name)  # type: ignore[index,return-value]


def _reseal(document: dict[str, object]) -> dict[str, object]:
    body = deepcopy(document)
    body.pop("seal_sha256", None)
    return seal(body)


def test_current_component_evidence_is_hard_refused_without_child_or_device() -> None:
    report = launcher.assess_outer_launch(launcher.current_component_evidence())
    assert report["status"] == launcher.REFUSED_STATUS
    assert report["future_child_launch_eligible"] is False
    assert report["current_component_evidence_hard_refused"] is True
    assert report["claim_boundary"]["outer_launcher_started_no_child"] is True  # type: ignore[index]
    assert report["claim_boundary"]["outer_launcher_created_no_metal_context_or_dispatch"] is True  # type: ignore[index]
    assert _condition(report, "sealed_full_48_layer_device_parity")["satisfied"] is False


def test_complete_sealed_full_graph_evidence_only_prepares_a_future_child() -> None:
    report = launcher.assess_outer_launch(_evidence())
    assert report["status"] == launcher.PREPARED_STATUS
    assert report["future_child_launch_eligible"] is True
    assert report["separate_controlled_child_required_if_future_eligible"] is True
    assert report["blockers"] == []
    assert all(condition["satisfied"] for condition in report["conditions"])  # type: ignore[index]
    assert report["claim_boundary"]["outer_launcher_started_no_process"] is True  # type: ignore[index]
    assert report["claim_boundary"]["outer_launcher_bound_no_port"] is True  # type: ignore[index]


def test_partial_component_or_fixture_layer_evidence_is_refused() -> None:
    evidence = _evidence()
    layers = deepcopy(evidence["layer_parity_receipt"])
    layers["layers"][0]["component_only"] = True  # type: ignore[index]
    evidence["layer_parity_receipt"] = _reseal(layers)
    report = launcher.assess_outer_launch(evidence)
    assert report["future_child_launch_eligible"] is False
    condition = _condition(report, "sealed_full_48_layer_device_parity")
    assert condition["satisfied"] is False
    assert any("component_only" in blocker for blocker in condition["blockers"])  # type: ignore[index]


def test_schedule_or_raw_plan_sha_drift_is_refused() -> None:
    evidence = _evidence()
    evidence["payload_schedule_plan"]["layers"][3]["mixer"] = "delta_net"  # type: ignore[index]
    report = launcher.assess_outer_launch(evidence)
    assert report["future_child_launch_eligible"] is False
    assert _condition(report, "raw_static_payload_schedule_authority")["satisfied"] is False

    evidence = _evidence()
    terminal = deepcopy(evidence["terminal_receipt"])
    terminal["payload_schedule_binding"]["payload_schedule_plan_sha256"] = _sha("other-plan")  # type: ignore[index]
    evidence["terminal_receipt"] = _reseal(terminal)
    report = launcher.assess_outer_launch(evidence)
    assert report["future_child_launch_eligible"] is False
    assert _condition(report, "same_plan_same_input_cross_receipt_provenance")["satisfied"] is False


def test_state_and_terminal_path_requirements_are_not_optional() -> None:
    evidence = _evidence()
    state = deepcopy(evidence["state_rollback_receipt"])
    state["gqa_slots"][11]["rollback_parity_passed"] = False  # type: ignore[index]
    evidence["state_rollback_receipt"] = _reseal(state)
    terminal = deepcopy(evidence["terminal_receipt"])
    terminal["reserved_tail_mask_applied_before_sample"] = False
    evidence["terminal_receipt"] = _reseal(terminal)
    report = launcher.assess_outer_launch(evidence)
    assert report["future_child_launch_eligible"] is False
    assert _condition(report, "sealed_all_state_slots_and_rollback")["satisfied"] is False
    assert _condition(report, "sealed_terminal_head_full_token_path")["satisfied"] is False


def test_cli_writes_new_sealed_preparation_and_refuses_replay(tmp_path: Path) -> None:
    evidence = _evidence()
    plan_path = tmp_path / "payload-plan.json"
    plan_path.write_text(json.dumps(evidence["payload_schedule_plan"], sort_keys=True), encoding="utf-8")
    plan_sha = launcher._file_sha256(plan_path)
    evidence = _evidence(plan_sha)
    plan_path.write_text(json.dumps(evidence["payload_schedule_plan"], sort_keys=True), encoding="utf-8")
    plan_sha = launcher._file_sha256(plan_path)
    assert plan_sha == evidence["payload_schedule_plan_sha256"]

    paths: dict[str, Path] = {}
    for argument, key in (
        ("decoder", "decoder_readiness"),
        ("layers", "layer_parity_receipt"),
        ("state", "state_rollback_receipt"),
        ("terminal", "terminal_receipt"),
    ):
        path = tmp_path / f"{argument}.json"
        path.write_text(json.dumps(evidence[key], sort_keys=True), encoding="utf-8")
        paths[argument] = path
    output = tmp_path / "outer-preparation.json"
    arguments = [
        "--payload-schedule-plan",
        str(plan_path),
        "--payload-schedule-plan-sha256",
        plan_sha,
        "--decoder-readiness",
        str(paths["decoder"]),
        "--layer-parity-receipt",
        str(paths["layers"]),
        "--state-rollback-receipt",
        str(paths["state"]),
        "--terminal-receipt",
        str(paths["terminal"]),
        "--out",
        str(output),
    ]
    assert launcher.main(arguments) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    verify(report, label="outer preparation")
    assert report["status"] == launcher.PREPARED_STATUS
    assert launcher.main(arguments) == 2


def test_module_has_no_process_or_device_launcher_dependency() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "Popen(" not in source
    assert "os.system(" not in source
