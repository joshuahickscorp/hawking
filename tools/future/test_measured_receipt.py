"""A hardware number must be placeable in time.

write_receipt REFUSES hardware fields, so every GPU-measurement module in this
package hand-rolled its own json.dumps and inherited no bench block. The cost
surfaced the day /tmp/hawking-gpu-lane.lock was found wedged as a stale 0-byte
file: placing MLP_ALU_ROOFLINE, MLP_DECODE_CHEAPEN, DELTANET_WIDEN_AB and
AUX_U8_LUT against that contention window had to be done from GIT LANDING TIMES,
because not one of them recorded when it actually ran.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402


def test_provenance_carries_the_three_fields_an_audit_needs():
    p = c.measurement_provenance(lock_held=True, lane="unit")
    for field in c.REQUIRED_PROVENANCE:
        assert field in p
    assert p["measured_at"].endswith("Z")
    assert p["gpu_lane_lock_held"] is True


def test_hardware_number_without_provenance_raises(tmp_path):
    with pytest.raises(c.MeasurementProvenanceError):
        c.write_measured_receipt(tmp_path / "r.json", {"gpu_ns": 1234}, "unit")


def test_hardware_number_nested_in_a_list_is_also_caught(tmp_path):
    doc = {"arms": [{"name": "a"}, {"name": "b", "bandwidth_gbps": 370.9}]}
    with pytest.raises(c.MeasurementProvenanceError):
        c.write_measured_receipt(tmp_path / "r.json", doc, "unit")


def test_partial_provenance_names_what_is_missing(tmp_path):
    doc = {"gpu_ns": 1, "measurement_provenance": {"measured_at": "2026-01-01T00:00:00Z"}}
    with pytest.raises(c.MeasurementProvenanceError) as e:
        c.write_measured_receipt(tmp_path / "r.json", doc, "unit")
    assert "gpu_lane_lock_held" in str(e.value)


def test_empty_measured_at_is_refused_because_landing_time_is_a_proxy(tmp_path):
    doc = {
        "gpu_ns": 1,
        "measurement_provenance": {
            "measured_at": "", "gpu_lane_lock_held": False, "loadavg": None,
        },
    }
    with pytest.raises(c.MeasurementProvenanceError):
        c.write_measured_receipt(tmp_path / "r.json", doc, "unit")


def test_a_receipt_with_no_hardware_number_needs_no_provenance(tmp_path):
    out = c.write_measured_receipt(tmp_path / "r.json", {"note": "static"}, "unit")
    assert json.loads(out.read_text())["note"] == "static"


def test_stamped_hardware_number_is_written_and_sealed(tmp_path):
    doc = {"gpu_ns": 249750, "measurement_provenance": c.measurement_provenance(
        lock_held=True, loadavg="{ 1.00 1.00 1.00 }", lane="unit")}
    out = c.write_measured_receipt(tmp_path / "r.json", doc, "unit")
    got = json.loads(out.read_text())
    assert got["gpu_ns"] == 249750
    assert got["measurement_provenance"]["gpu_lane_lock_held"] is True
    assert len(got["seal_sha256"]) == 64


def test_the_sidecar_writer_still_refuses_hardware_entirely(tmp_path, monkeypatch):
    """The two writers have opposite jobs and neither may drift into the other."""
    monkeypatch.setattr(c, "RECEIPTS", tmp_path)
    with pytest.raises(c.HardwareClaimError):
        c.write_receipt("r.json", {"tps": 34.82}, "unit")
