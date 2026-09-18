from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.odyssey import flash_source_boundary_preflight as preflight


def _seal(document: dict) -> dict:
    result = dict(document)
    result["seal_sha256"] = preflight.gate._compact_utf8_sorted_seal(result)
    return result


def _reference(path: Path) -> None:
    path.write_text(json.dumps(_seal({
        "schema": preflight.gate.EXTERNAL_REFERENCE_SCHEMA,
        "status": preflight.gate.EXTERNAL_REFERENCE_STATUS,
        "source": {
            "model": preflight.gate.REPO_ID,
            "pinned_revision": preflight.gate.PINNED_REVISION,
            "ple_inclusive": True,
        },
        "provider": {"execution_schedule": preflight.REFERENCE_SCHEDULE},
        "prompt_token_ids": list(preflight.gate.PROMPT_IDS),
        "generated_token_ids": [271, 248045],
    })), encoding="utf-8")


def _native(path: Path) -> None:
    path.write_text(json.dumps(_seal({
        "schema": preflight.NATIVE_SCHEMA,
        "status": preflight.NATIVE_STATUS,
        "token_ids": [*preflight.gate.PROMPT_IDS, 2972, 96125],
        "execution": {"target": {"layer": 4}},
    })), encoding="utf-8")


def test_unsigned_preflight_is_fail_closed_and_names_ple_seam(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    native = tmp_path / "native.json"
    _reference(reference)
    _native(native)
    document = preflight.build_preflight(
        reference=reference,
        model_root=tmp_path,
        native_control=native,
        owner_authorization=None,
    )
    assert document["status"] == "WITHHELD_ADMISSION_AUTHORITY_AND_BOUNDARY_PAYLOADS"
    assert document["runtime_actions"] == {
        "model_loaded": False,
        "gpu_or_metal_started": False,
        "daemon_restarted": False,
        "kimi_loaded": False,
    }
    assert document["capture_contract"]["seams"][3]["stage"] == "ple_additive_output"
    assert document["capture_contract"]["layer0_trace"] == preflight._required_layer0_trace()
    assert document["capture_contract"]["layer0_trace"][1] == {
        "ordinal": 1,
        "layer": 0,
        "stage": "attention_hyper_connection_down_projection",
        "shape": [1, 1, 320],
    }
    assert document["capture_contract"]["layer0_trace_contract_version"] == 5
    assert document["capture_contract"]["layer4_cache"] == preflight._required_layer4_cache()
    assert preflight._shape_elements(
        document["capture_contract"]["layer4_cache"]["pre_attention"][1]["shape"]
    ) == 786_432
    assert document["seal_sha256"] == preflight.gate._compact_utf8_sorted_seal(
        {key: value for key, value in document.items() if key != "seal_sha256"}
    )


def test_native_frontier_rejects_shifted_token_zero(tmp_path: Path) -> None:
    native = tmp_path / "native.json"
    _native(native)
    document = json.loads(native.read_text(encoding="utf-8"))
    document["token_ids"][0] = 1
    native.write_text(json.dumps(_seal({key: value for key, value in document.items() if key != "seal_sha256"})), encoding="utf-8")
    with pytest.raises(ValueError, match="layer-4/token-0"):
        preflight._native_frontier(native, preflight.gate.PROMPT_IDS[0])


def test_write_new_refuses_clobber(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_text("owned", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        preflight._write_new(target, {"schema": "test"})
