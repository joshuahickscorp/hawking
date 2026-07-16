from __future__ import annotations

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
MODULE_PATH = CONDENSE / "tq_receipt_contract.py"
SPEC = importlib.util.spec_from_file_location("tq_receipt_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_valid_device_payload_passes() -> None:
    assert MODULE.validate_payload(MODULE._valid_payload()) == []


def test_recipe_layout_parity_and_traffic_drift_fail() -> None:
    payload = MODULE._valid_payload()
    payload["recipe"]["metadata"] = "expanded"
    payload["parity"]["mismatches"] = 1
    payload["logical_traffic"]["compressed_runtime_total"] += 1
    errors = MODULE.validate_payload(payload)
    assert "recipe does not match runtime_path" in errors
    assert "parity requires zero mismatches" in errors
    assert "compressed_runtime_total does not equal traffic components" in errors


def test_microbenchmark_cannot_change_default_and_requires_physical_counters() -> None:
    payload = MODULE._valid_payload()
    payload["default_change_requested"] = True
    payload["physical_counters"]["measured"] = False
    errors = MODULE.validate_payload(payload)
    assert "device microbenchmark cannot request a default change" in errors
    assert "physical counters must be measured" in errors


def test_negative_occupancy_is_rejected() -> None:
    payload = MODULE._valid_payload()
    payload["physical_counters"]["occupancy_percent"] = -1.0
    assert "occupancy_percent must be in [0,100]" in MODULE.validate_payload(payload)


def test_two_pass_bpw_requires_residual_weights_in_denominator() -> None:
    payload = MODULE._valid_payload()
    weights = payload["shape"]["weights"]
    payload["residual_probe"] = {
        "enabled": True,
        "tensor": {"weights": weights},
    }
    payload["logical_traffic"]["compressed_runtime_bpw"] = (
        payload["logical_traffic"]["compressed_runtime_total"] * 8
        / (2 * weights)
    )
    assert MODULE.validate_payload(payload) == []
    payload["logical_traffic"]["compressed_runtime_bpw"] *= 2
    assert any(
        "all decoded projection weights" in error
        for error in MODULE.validate_payload(payload)
    )
