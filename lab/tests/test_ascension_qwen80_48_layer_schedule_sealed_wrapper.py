from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lab.receipts import verify
from lab.operators import ascension_qwen80_48_layer_schedule_sealed_wrapper as wrapper


def _source_authority() -> dict[str, object]:
    return {
        "model_key": wrapper.MODEL_KEY,
        "model_id": wrapper.MODEL_ID,
        "source_repository": wrapper.SOURCE_REPOSITORY,
        "source_revision": wrapper.SOURCE_REVISION,
        "source_config_sha256": wrapper.SOURCE_CONFIG_SHA256,
        "descriptor_inventory_document_sha256": wrapper.MANIFEST_DOCUMENT_SHA256,
        "descriptor_inventory_seal_sha256": wrapper.MANIFEST_SEAL_SHA256,
        "descriptor_inventory_schema": "hawking.ascension.qwen80_complete_binary_gravity.v1",
        "source_config_authority_document_sha256": wrapper.SOURCE_CONFIG_AUTHORITY_DOCUMENT_SHA256,
        "source_config_authority_seal_sha256": wrapper.SOURCE_CONFIG_AUTHORITY_SEAL_SHA256,
        "source_config_authority_schema": "hawking.ascension.source_admission_candidate.v1",
    }


def _raw_schedule() -> dict[str, object]:
    layers: list[dict[str, object]] = []
    delta_slots: list[dict[str, object]] = []
    gqa_slots: list[dict[str, object]] = []
    for layer in range(48):
        mixer = "gqa" if layer % 4 == 3 else "delta_net"
        slots = gqa_slots if mixer == "gqa" else delta_slots
        state = {
            "layer": layer,
            "slot": len(slots),
            "domain": "gqa_kv" if mixer == "gqa" else "delta_net_conv_and_recurrent",
            "state_materialized_by_this_plan": False,
        }
        slots.append(state)
        layers.append({"layer": layer, "mixer": mixer, "state_slot": state})
    return {
        "schema": wrapper.RAW_SCHEMA,
        "status": wrapper.RAW_STATUS,
        "all_48_layers_scheduled": True,
        "all_descriptors_source_artifact_bound": True,
        "resolved_tensor_binding_count": 74391,
        "source_authority": _source_authority(),
        "layers": layers,
        "deltanet_state_slots": delta_slots,
        "gqa_state_slots": gqa_slots,
        "full_command_graph_order": [
            "embedding",
            *[f"layer_{layer}" for layer in range(48)],
            "final_rmsnorm",
            "all_row_lm_head",
            "reserved_tail_mask",
            "deterministic_sample",
            "tokenizer_feedback",
        ],
        "claim_boundary": {
            "artifact_payload_open_or_scan_performed": False,
            "metal_device_or_dispatch_performed": False,
            "runtime_watcher_registry_server_or_hcli_changed": False,
            "model_execution_performed": False,
            "token_generation_or_feedback_performed": False,
            "tps_or_tg_measured": False,
        },
    }


def _write_raw(path: Path, document: dict[str, object]) -> bytes:
    raw = json.dumps(document, sort_keys=True, indent=2).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _bind_fixture_as_canonical(monkeypatch: pytest.MonkeyPatch, path: Path, raw: bytes) -> None:
    monkeypatch.setattr(wrapper, "CANONICAL_RAW_SCHEDULE_FILENAME", path.name)
    monkeypatch.setattr(wrapper, "CANONICAL_RAW_SCHEDULE_BYTES", len(raw))
    monkeypatch.setattr(wrapper, "CANONICAL_RAW_SCHEDULE_SHA256", hashlib.sha256(raw).hexdigest())


def test_wrapper_binds_raw_bytes_source_identities_and_36_12_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path = tmp_path / wrapper.CANONICAL_RAW_SCHEDULE_FILENAME
    raw = _write_raw(raw_path, _raw_schedule())
    _bind_fixture_as_canonical(monkeypatch, raw_path, raw)
    output = tmp_path / "wrapper.json"

    body = wrapper.build_wrapper(raw_path.resolve())
    written = wrapper.write_new(output.resolve(), body)
    sealed = json.loads(written.read_text(encoding="utf-8"))

    verify(sealed, label="schedule wrapper")
    assert sealed["schema"] == wrapper.WRAPPER_SCHEMA
    assert sealed["status"] == wrapper.WRAPPER_STATUS
    assert sealed["raw_schedule_authority"]["path"] == str(raw_path.resolve())
    assert sealed["raw_schedule_authority"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert sealed["raw_schedule_authority"]["raw_schedule_seal_sha256"] is None
    assert sealed["schedule_facts"]["delta_net_layer_count"] == 36
    assert sealed["schedule_facts"]["gqa_layer_count"] == 12
    assert sealed["schedule_facts"]["layer_1"] == {
        "layer": 1,
        "mixer": "delta_net",
        "state_slot": 1,
        "state_domain": "delta_net_conv_and_recurrent",
    }
    assert sealed["claim_boundary"]["raw_schedule_rewritten_or_resealed"] is False
    assert sealed["claim_boundary"]["metal_device_or_dispatch_performed"] is False


def test_wrapper_refuses_schedule_or_source_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drifted = _raw_schedule()
    drifted["layers"][0]["mixer"] = "gqa"  # type: ignore[index]
    drifted["layers"][0]["state_slot"]["domain"] = "gqa_kv"  # type: ignore[index]
    raw_path = tmp_path / wrapper.CANONICAL_RAW_SCHEDULE_FILENAME
    raw = _write_raw(raw_path, drifted)
    _bind_fixture_as_canonical(monkeypatch, raw_path, raw)
    with pytest.raises(wrapper.ScheduleWrapperError, match="36 DeltaNet"):
        wrapper.build_wrapper(raw_path.resolve())

    source_drifted = _raw_schedule()
    source_drifted["source_authority"]["source_revision"] = "0" * 40  # type: ignore[index]
    source_path = tmp_path / wrapper.CANONICAL_RAW_SCHEDULE_FILENAME
    raw = _write_raw(source_path, source_drifted)
    _bind_fixture_as_canonical(monkeypatch, source_path, raw)
    with pytest.raises(wrapper.ScheduleWrapperError, match="source_revision"):
        wrapper.build_wrapper(source_path.resolve())


def test_wrapper_refuses_raw_reseal_and_output_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_document = _raw_schedule()
    raw_document["seal_sha256"] = "f" * 64
    raw_path = tmp_path / wrapper.CANONICAL_RAW_SCHEDULE_FILENAME
    raw = _write_raw(raw_path, raw_document)
    _bind_fixture_as_canonical(monkeypatch, raw_path, raw)
    with pytest.raises(wrapper.ScheduleWrapperError, match="remain unsealed"):
        wrapper.build_wrapper(raw_path.resolve())

    valid_path = tmp_path / wrapper.CANONICAL_RAW_SCHEDULE_FILENAME
    raw = _write_raw(valid_path, _raw_schedule())
    _bind_fixture_as_canonical(monkeypatch, valid_path, raw)
    body = wrapper.build_wrapper(valid_path.resolve())
    output = tmp_path / "existing.json"
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        wrapper.write_new(output.resolve(), body)


def test_wrapper_refuses_a_lookalike_path_even_when_the_bytes_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / wrapper.CANONICAL_RAW_SCHEDULE_FILENAME
    raw = _write_raw(canonical, _raw_schedule())
    _bind_fixture_as_canonical(monkeypatch, canonical, raw)

    lookalike = tmp_path / "QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY_COPY.json"
    lookalike.write_bytes(raw)
    with pytest.raises(wrapper.ScheduleWrapperError, match="raw schedule filename"):
        wrapper.build_wrapper(lookalike.resolve())
