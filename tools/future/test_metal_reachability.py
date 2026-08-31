"""Falsifying a blocker is only useful if the falsification cannot overreach.

The finding here is narrow on purpose: the HOST has a Metal GPU, therefore
"this host has no Metal-capable GPU" is false as a statement about the machine.
It says nothing about why one Rust process saw no device, and it does not
promise the blocked capture will now run. These tests keep it that narrow --
an overreaching correction would send someone to re-run a 256-row capture on
the strength of a claim this module never established.
"""
import json
import pathlib

import pytest

from tools.future import metal_reachability as mr
from tools.future import status_causality as sc
from tools.future._common import REPO, RECEIPTS


def test_a_present_device_falsifies_only_the_host_claim():
    v = mr.verdict({"system_default": "Apple M3 Ultra", "n_devices": 1})
    assert v["verdict"] == "FALSIFIED_AS_A_HOST_PROPERTY"
    assert "Apple M3 Ultra" in v["why"]
    disclaimed = " ".join(v["what_this_does_not_establish"])
    assert "will now succeed" in disclaimed, "must not promise the capture works"
    assert "cause" in disclaimed, "must not name a cause it did not establish"


def test_no_device_confirms_the_claim_rather_than_explaining_it_away():
    v = mr.verdict({"system_default": None, "n_devices": 0})
    assert v["verdict"] == "CONFIRMED"


def test_an_unavailable_probe_is_untested_never_confirmed():
    """Failing to ask is not the same as being told no."""
    v = mr.verdict(None, "swiftc is not on PATH")
    assert v["verdict"] == "UNTESTED"
    assert "swiftc" in v["why"]


def test_the_probe_creates_no_command_queue():
    """Enumeration is a capability query. A queue would make it a GPU user."""
    assert "newCommandQueue" not in mr.PROBE_SWIFT
    assert "makeCommandQueue" not in mr.PROBE_SWIFT
    assert "MTLCreateSystemDefaultDevice" in mr.PROBE_SWIFT


def test_receipt_claims_no_measurement_and_no_authority():
    out = mr.build()
    doc = json.loads(out.read_text())
    assert doc["is_a_measurement"] is False
    assert doc["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["seal_sha256"]


def test_the_claim_sites_are_real_files_that_carry_the_claim():
    """A correction has to be able to find every place the sentence lives."""
    rows = mr.claim_sites()
    assert rows, "no claim sites recorded"
    for row in rows:
        if row["present"]:
            assert (REPO / row["path"]).is_file()
    assert any(r["carries_claim"] for r in rows), "the claim was not found anywhere"


def test_the_hardcoded_boundary_status_is_recorded_as_a_negative_finding():
    """flash_meta_teacher_trace stamps BLOCKED_NO_METAL_GPU on ANY init error.

    That means the status alone never proves the GPU was the problem, and the
    receipt has to say so or it invites the same mistake again.
    """
    doc = json.loads((RECEIPTS / mr.RECEIPT).read_text())
    findings = " ".join(doc["negative_findings"])
    assert "ANY" in findings and "BLOCKED_NO_METAL_GPU" in findings


def test_the_rust_probe_uses_the_version_the_runtime_resolves():
    """Testing a different crate version would prove nothing about the runtime."""
    version = mr._runtime_metal_crate_version()
    assert version, "Cargo.lock must resolve a metal crate version"
    doc = json.loads((RECEIPTS / mr.RECEIPT).read_text())
    observed = doc.get("observed_runtime_binding")
    if observed:
        assert observed["metal_crate_version"] == version


def test_the_rust_probe_also_creates_no_command_queue():
    """It compiles a shader, which exercises the compiler service, not the GPU."""
    assert "new_command_queue" not in mr.PROBE_RUST_MAIN
    assert "Device::system_default()" in mr.PROBE_RUST_MAIN
    assert "new_library_with_source" in mr.PROBE_RUST_MAIN
    for dispatched in ("dispatch_thread", "commit()", "new_compute_command_encoder"):
        assert dispatched not in mr.PROBE_RUST_MAIN


def test_shader_compilation_is_only_claimed_when_it_was_exercised():
    """UNKNOWN must mean not-run, never a guess in either direction."""
    from tools.future import hardware_doctor as hwd

    doc = json.loads((RECEIPTS / mr.RECEIPT).read_text())
    observed = (doc.get("observed_runtime_binding") or {}).get("runtime_source_compile")
    state = hwd.metal_state()["runtime_source_compilation"]
    assert state in {"AVAILABLE", "UNKNOWN"}
    assert (state == "AVAILABLE") == (observed == "OK")
    if state == "AVAILABLE":
        note = doc["verdict"]["shader_compilation"]
        assert "not a wall" in note["why_it_matters"]
        assert "no command queue" in note["still_not_a_measurement"]


def test_the_rust_probe_builds_out_of_tree():
    """A campaign is running. Nothing here may touch the repo's build state."""
    import inspect

    src = inspect.getsource(mr.probe_rust)
    assert "CARGO_TARGET_DIR" in src, "the probe must not share the repo target dir"
    assert "--offline" in src, "a probe that reaches the network is not a local probe"
    assert "TemporaryDirectory" in src


def test_runtime_binding_verdict_is_reported_separately():
    """Host and binding are different claims and must not be merged into one."""
    doc = json.loads((RECEIPTS / mr.RECEIPT).read_text())
    v = doc["verdict"]
    assert "runtime_binding" in v
    assert v["runtime_binding"]["verdict"] in {
        "FALSIFIED_AS_A_HOST_PROPERTY", "CONFIRMED", "UNTESTED"
    }


# ---------------------------------------------------------------------------
# G007 consumer: verdict records the five causality fields.
# ---------------------------------------------------------------------------


def test_verdict_records_the_five_causality_fields():
    """A coverage number no test defends will drift back to zero."""
    r = mr.verdict({"system_default": "Apple M3 Ultra", "n_devices": 1})
    assert mr.records_five_fields(r)
    src = pathlib.Path(mr.__file__).read_text()
    assert "sc.emit(" in src
    assert r["verdict"] == "FALSIFIED_AS_A_HOST_PROPERTY"
    assert r["probe_performed"]
    assert "MTLCreateSystemDefaultDevice" in r["probe_performed"]
    assert r["direct_observation"] != r["verdict"]
    assert r["direct_observation"] != r["interpretation"]
    assert "system_default=" in r["direct_observation"]
    assert r["causality_verdict"] in {sc.SUPPORTED, sc.OVERREACHING, sc.UNTESTED}


def test_unsupplied_observation_records_untested_not_a_restatement():
    result = {"claim": mr.CLAIM, "verdict": "UNTESTED", "why": "probe skipped"}
    rec = mr.record_verdict_causality(
        result, probe_performed="", direct_observation=""
    )
    assert rec["verdict"] == sc.UNTESTED
    assert rec["direct_observation"] in ("", None)
    assert rec["direct_observation"] != "UNTESTED"
    assert "UNTESTED" not in str(rec["direct_observation"] or "")
    assert result["verdict"] == "UNTESTED"
    assert rec["interpretation"] != rec["direct_observation"]


def test_overreaching_does_not_override_metal_verdict(monkeypatch):
    def overreach(status, **kwargs):
        return {
            "probe_performed": kwargs.get("probe_performed") or "p",
            "direct_observation": kwargs.get("direct_observation") or "o",
            "interpretation": kwargs.get("interpretation") or status,
            "confidence": {
                "level": "LOW",
                "about": "a",
                "would_raise": "b",
                "would_lower": "c",
            },
            "alternatives": [
                {
                    "hypothetical": "h",
                    "consistent_with_observation": True,
                    "consistent_with_claim": False,
                }
            ],
            "verdict": sc.OVERREACHING,
            "falsifier": "f",
            "probe_kind": sc.PROBE_ENUMERATION,
            "claim_kind": sc.CLAIM_HOST_HARDWARE_ABSENCE,
        }

    monkeypatch.setattr(mr.sc, "emit", overreach)
    r = mr.verdict({"system_default": "Apple M3 Ultra", "n_devices": 1})
    assert r["verdict"] == "FALSIFIED_AS_A_HOST_PROPERTY"
    assert r["causality_verdict"] == sc.OVERREACHING


def test_unavailable_probe_observation_is_not_a_status_restatement():
    r = mr.verdict(None, "swiftc is not on PATH")
    assert r["verdict"] == "UNTESTED"
    assert mr.records_five_fields(r)
    assert r["direct_observation"] != "UNTESTED"
    assert "probe_ran=False" in r["direct_observation"]
    assert "swiftc" in r["direct_observation"]


def test_coverage_receipt_names_metal_reachability_as_recording():
    path = RECEIPTS / "STATUS_CAUSALITY_COVERAGE.json"
    doc = json.loads(path.read_text())
    assert "metal_reachability" in doc["recording_five_fields"]
    assert doc["n_gates"] == 18
