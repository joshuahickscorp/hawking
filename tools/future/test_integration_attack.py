"""The attacker must itself be watched failing.

An adversarial checker nobody has seen fire is exactly the defect it hunts, so
every attack here is exercised against a deliberately broken artifact.
"""
import json

import pytest

from tools.future import integration_attack as ia
from tools.future._common import HardwareClaimError, RECEIPTS, write_receipt


def test_runs_and_emits_sealed_receipt():
    doc = ia.run()
    assert doc["schema"] == "hawking.future.integration_attack.v1"
    assert set(doc["attacks_run"]) == set(ia.ATTACKS)
    assert doc["verdict"] in {"CLEAN", "P0_PRESENT", "P1_PRESENT"}


def test_common_refuses_a_hardware_claim_at_the_source():
    # The first line of defence: write_receipt must refuse before anything lands.
    with pytest.raises(HardwareClaimError):
        write_receipt("_probe_should_not_exist.json", {"tps": 51.2}, "test")
    assert not (RECEIPTS / "_probe_should_not_exist.json").exists()


def test_hardware_claim_attack_fires(tmp_path, monkeypatch):
    bad = {"bench": {"state": "UNKNOWN", "gpu_authority": False},
           "results": [{"token_ns": 41000}], "seal_sha256": "x"}
    p = RECEIPTS / "_attack_probe_hw.json"
    p.write_text(json.dumps(bad))
    try:
        kinds = {f["kind"] for f in ia.attack_hardware_claims()}
        assert "hardware_claim_without_hardware" in kinds
    finally:
        p.unlink()


def test_bench_override_attack_fires():
    bad = {"bench": {"state": "PROTECTED", "gpu_authority": True}, "seal_sha256": "x"}
    p = RECEIPTS / "_attack_probe_bench.json"
    p.write_text(json.dumps(bad))
    try:
        kinds = {f["kind"] for f in ia.attack_hardware_claims()}
        assert "bench_state_overridden" in kinds
        assert "gpu_authority_claimed" in kinds
    finally:
        p.unlink()


def test_seal_mismatch_attack_fires():
    p = RECEIPTS / "_attack_probe_seal.json"
    p.write_text(json.dumps({"a": 1, "seal_sha256": "deadbeef"}))
    try:
        kinds = {f["kind"] for f in ia.attack_seals()}
        assert "seal_mismatch" in kinds
    finally:
        p.unlink()


def test_forbidden_promotion_status_attack_fires():
    p = RECEIPTS / "_attack_probe_status.json"
    p.write_text(json.dumps({"status": "PROTECTED_PASS", "seal_sha256": "x"}, indent=1))
    try:
        kinds = {f["kind"] for f in ia.attack_forbidden_status()}
        assert "forbidden_promotion_status" in kinds
    finally:
        p.unlink()


def test_placeholder_attack_fires():
    p = ia.FUTURE / "_attack_probe_stub.py"
    p.write_text("def load_bearing():\n    raise NotImplementedError\n")
    try:
        kinds = {f["kind"] for f in ia.attack_placeholders()}
        assert "placeholder_in_module" in kinds
        assert "module_without_test" in {f["kind"] for f in ia.attack_missing_tests()}
    finally:
        p.unlink()


def test_skip_attack_fires():
    p = ia.FUTURE / "test__attack_probe_skip.py"
    p.write_text("import pytest\n\n@pytest.mark.skip\ndef test_nothing():\n    assert True\n")
    try:
        f = ia.attack_skipped_tests()
        assert any(x["kind"] == "test_skip" and x["severity"] == "P0" for x in f)
    finally:
        p.unlink()


def test_missing_negative_control_attack_fires():
    p = ia.FUTURE / "test__attack_probe_soft.py"
    p.write_text("def test_happy():\n    assert 1 + 1 == 2\n")
    try:
        kinds = {f["kind"] for f in ia.attack_missing_negative_controls()}
        assert "test_without_negative_control" in kinds
    finally:
        p.unlink()
