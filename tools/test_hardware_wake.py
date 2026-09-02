"""Every BLOCKED_HARDWARE gate carries a machine-readable wake condition."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.roadmap.hardware import WAKE_CONDITIONS, blocked_hardware_wakes

REPO = Path(__file__).resolve().parents[1]
GRAPH = REPO / "civilization" / "CAPABILITY_GRAPH.json"


def _graph():
    return json.loads(GRAPH.read_text())


def test_all_13_blocked_hardware_gates_carry_a_nonempty_wake_condition():
    doc = _graph()
    blocked = [g for g in doc["gates"].values() if g["status"] == "BLOCKED_HARDWARE"]
    assert len(blocked) == 13, (
        f"expected 13 BLOCKED_HARDWARE gates, got {len(blocked)}: "
        + ",".join(sorted(g["id"] for g in blocked))
    )
    wakes = blocked_hardware_wakes(doc["gates"])
    assert len(wakes) == 13
    seen = []
    for gid, wake in wakes:
        assert wake.strip(), f"{gid} empty wake_condition"
        assert wake == wake.upper() and "_" in wake, f"{gid} wake {wake!r} is not a machine id"
        assert wake in WAKE_CONDITIONS, f"{gid} wake {wake!r} not in WAKE_CONDITIONS"
        assert WAKE_CONDITIONS[wake].strip(), f"{wake} has an empty description"
        seen.append((gid, wake))
    # The 13: twelve U50 rungs + HMF device-visible trust.
    by_wake: dict[str, list[str]] = {}
    for gid, wake in seen:
        by_wake.setdefault(wake, []).append(gid)
    assert "U50_PRESENT" in by_wake
    assert "HMF_PRESENT" in by_wake
    assert len(by_wake["U50_PRESENT"]) == 12
    assert by_wake["HMF_PRESENT"] == ["HMF_DEVICE_VISIBLE_TRUST"]


def test_wake_conditions_catalog_is_machine_readable():
    required = {
        "U50_PRESENT",
        "DGX_PRESENT",
        "NEW_M_SERIES_PRESENT",
        "HMF_PRESENT",
        "EGPU_PRESENT",
    }
    assert required <= set(WAKE_CONDITIONS)
    for name, desc in WAKE_CONDITIONS.items():
        assert name == name.upper() and name.endswith("_PRESENT"), name
        assert isinstance(desc, str) and len(desc) > 20, name
        assert "TODO" not in desc.upper()
        assert "TBD" not in desc.upper()


def test_empty_wake_condition_is_refused():
    gates = {
        "U50_DMA_HBM": {
            "id": "U50_DMA_HBM",
            "status": "BLOCKED_HARDWARE",
            "wake_condition": "",
        }
    }
    with pytest.raises(ValueError, match="empty wake_condition"):
        blocked_hardware_wakes(gates)
    gates["U50_DMA_HBM"]["wake_condition"] = "NOT_A_PROBE"
    with pytest.raises(ValueError, match="not a known hardware id"):
        blocked_hardware_wakes(gates)


def test_wake_ids_match_the_committed_graph_rows():
    """Pin the 13 so a silent drop cannot hide behind a count."""
    doc = _graph()
    expected = {
        "U50_PURCHASE_ACCEPTANCE": "U50_PRESENT",
        "U50_SAFE_COOLING": "U50_PRESENT",
        "U50_DEVICE_PROFILE": "U50_PRESENT",
        "U50_DMA_HBM": "U50_PRESENT",
        "U50_FIRST_NATIVE_ENGINE": "U50_PRESENT",
        "U50_MIXED_APPLE_FPGA_GRAPH": "U50_PRESENT",
        "U50_34_TO_40": "U50_PRESENT",
        "U50_40_TO_50": "U50_PRESENT",
        "U50_50_TO_60": "U50_PRESENT",
        "U50_60_TO_70": "U50_PRESENT",
        "U50_70_TO_80": "U50_PRESENT",
        "U50_80_TO_90": "U50_PRESENT",
        "HMF_DEVICE_VISIBLE_TRUST": "HMF_PRESENT",
    }
    got = {g: w for g, w in blocked_hardware_wakes(doc["gates"])}
    assert got == expected
