"""Focused CPU-only checks for the HQ30GR2 sparse-component preparation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lab.operators import ascension_qwen30_quality_repack_sparse_gate_up_component_prepare as prepare


def _documents(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    hidden_path = tmp_path / prepare.TARGET_HIDDEN_RELATIVE
    hidden_path.parent.mkdir(parents=True, exist_ok=True)
    hidden_path.write_bytes(b"\0" * prepare.TARGET_HIDDEN_BYTES)
    hidden_sha256 = hashlib.sha256(hidden_path.read_bytes()).hexdigest()
    hidden = {"relative_path": prepare.TARGET_HIDDEN_RELATIVE, "sha256": hidden_sha256}
    route = {
        "binding": {"capture_output_root": str(tmp_path.resolve())},
        "probe_summary": [
            {
                "probe_id": prepare.TARGET_PROBE,
                "source_template_token_count": prepare.TARGET_TOKEN_COUNT,
                "l0_expert0_selected_positions": [prepare.TARGET_POSITION],
                "hidden_payloads": [hidden],
            }
        ],
    }
    preparation = {
        "planned_bounded_input": {
            "probe_id": prepare.TARGET_PROBE,
            "source_template_token_count": prepare.TARGET_TOKEN_COUNT,
            "l0_e0_selected_position": prepare.TARGET_POSITION,
            "l0_e0_router_input_hidden": hidden,
        }
    }
    return route, preparation


def test_component_input_binds_the_exact_device_produced_e0_vector(tmp_path: Path) -> None:
    route, preparation = _documents(tmp_path)
    selected = prepare._component_input(route, preparation)
    assert selected["probe_id"] == "literal_hawking"
    assert selected["l0_e0_selected_position"] == 337
    assert selected["device_produced_router_input_f32le"]["bytes"] == 8192


def test_component_input_refuses_a_non_selected_or_escaping_vector(tmp_path: Path) -> None:
    route, preparation = _documents(tmp_path)
    preparation["planned_bounded_input"]["l0_e0_selected_position"] = 338
    with pytest.raises(prepare.ComponentPreparationError, match="literal_hawking"):
        prepare._component_input(route, preparation)

    route, preparation = _documents(tmp_path)
    preparation["planned_bounded_input"]["l0_e0_router_input_hidden"]["relative_path"] = "../escape.f32le"
    with pytest.raises(prepare.ComponentPreparationError, match="literal_hawking"):
        prepare._component_input(route, preparation)
