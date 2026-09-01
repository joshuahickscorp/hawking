from __future__ import annotations

from copy import deepcopy

from hcli.agentos.fpga_preboard import (
    _attach_architecture_atlas,
    _map,
)
from tools.accelerator.architecture_atlas import build_atlas


def test_qwen27_atlas_projection_reaches_hwir_without_board_claim():
    atlas = build_atlas()
    stationary = next(row for row in atlas["entries"] if row["behavior_id"] == "stationary_representation")
    blocked = deepcopy(stationary)
    blocked["behavior_id"] = "blocked_stationary_probe"
    blocked["status"] = "BLOCKED"
    atlas["entries"] = [stationary, blocked]

    model_map = _attach_architecture_atlas(_map("qwen27"), "qwen27", atlas)
    projection = model_map["architecture_repatriation"]

    assert projection["status"] == "PROJECTED"
    assert projection["selected_behavior_ids"] == ["stationary_representation"]
    assert projection["selected_primitives"] == [stationary["hawking_primitive"]]
    assert model_map["hwir"]["architecture_repatriation"] == projection
    assert model_map["hwir"]["fingerprint"]
    assert model_map["device_genome"]["physical_board_present"] is False
    assert "hardware timing" in projection["claim_boundary"]


def test_flash_next_uses_flash_alias_and_missing_atlas_is_honest():
    atlas = build_atlas()
    flash = next(row for row in atlas["entries"] if row["behavior_id"] == "local_state_machine")
    atlas["entries"] = [flash]

    model_map = _attach_architecture_atlas(_map("flash-next"), "flash-next", atlas)
    assert model_map["architecture_repatriation"]["selected_behavior_ids"] == ["local_state_machine"]

    absent = _attach_architecture_atlas(_map("flash-next"), "flash-next", None)
    assert absent["architecture_repatriation"]["status"] == "ABSENT"
    assert absent["hwir"]["architecture_repatriation"]["status"] == "ABSENT"
    assert absent["device_genome"]["physical_board_present"] is False
