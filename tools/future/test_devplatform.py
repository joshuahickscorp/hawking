"""Tests for the developer-platform substrate.

A guard nobody has watched fail is not a guard: the compatibility suite
must refuse a profile/backend pair that requires a feature the provider
does not declare, and name that clause.
"""
from __future__ import annotations

import json

import pytest

from tools.future import devplatform as dp
from tools.future._common import HARDWARE_FIELDS, HardwareClaimError, RECEIPTS


def test_build_emits_sealed_receipt():
    out = dp.selftest()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "DEVELOPER_PLATFORM.json"
    assert doc["schema"] == "hawking.future.devplatform.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    report = dp.receipt_validate(doc)
    assert report["ok"], report["failures"]
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    for field in HARDWARE_FIELDS:
        assert field not in doc


def test_ir_roundtrip_is_byte_stable():
    doc = dp.make_ir("model_profile", dp.specimen_profile("qwen3.8-27b-mlx-4bit"))
    first, second, decoded = dp.roundtrip_ir(doc)
    assert first == second
    assert decoded["schema"] == dp.IR_SCHEMA
    assert decoded["ir_version"] == 1
    assert decoded["kind"] == "model_profile"
    again = dp.encode_ir(decoded)
    assert again == first


def test_ir_v0_migrates_to_v1_then_roundtrips():
    v0 = {
        "type": "workunit",
        "payload": {"id": "w0", "role": "code", "description": "migrate me"},
    }
    migrated = dp.migrate_ir(v0)
    assert migrated["ir_version"] == 1
    assert migrated["kind"] == "workunit"
    assert migrated["schema"] == dp.IR_SCHEMA
    assert migrated["payload"]["id"] == "w0"
    first, second, _ = dp.roundtrip_ir(migrated)
    assert first == second


def test_ir_unknown_version_refuses():
    with pytest.raises(dp.IrError) as exc:
        dp.migrate_ir({"ir_version": 99, "kind": "document", "payload": {}})
    assert "newer" in str(exc.value)


def test_ir_strips_wall_clock_from_hashed_content():
    doc = dp.make_ir(
        "document",
        {"n": 1, "recorded_at": "1999-01-01T00:00:00Z", "nested": {"generated_at": 1}},
    )
    encoded = dp.encode_ir(doc)
    assert b"recorded_at" not in encoded
    assert b"generated_at" not in encoded
    assert b'"n":1' in encoded


def test_receipt_validate_accepts_selftest_receipt():
    path = dp.build(selftest_results=dp.run_selftest_checks())
    doc = dp.receipt_read(path)
    report = dp.receipt_validate(doc)
    assert report["ok"], report["failures"]
    assert "receipt.seal.matches" in report["passed"]
    assert "receipt.bench.state.UNKNOWN" in report["passed"]


def test_receipt_validate_rejects_broken_seal():
    path = dp.build(selftest_results=dp.run_selftest_checks())
    doc = dp.receipt_read(path)
    doc["purpose"] = "tampered"
    report = dp.receipt_validate(doc)
    assert report["ok"] is False
    clauses = {row["clause"] for row in report["failures"]}
    assert "receipt.seal.matches" in clauses


def test_receipt_validate_rejects_numeric_hardware_field():
    doc = {
        "schema": "hawking.future.devplatform.v1",
        "seal_sha256": "dead",
        "bench": {
            "state": "UNKNOWN",
            "measurement_state": "STATIC_ONLY",
            "gpu_authority": False,
        },
        "tps": 12.0,
    }
    doc["seal_sha256"] = dp.receipt_seal_hex(doc)
    report = dp.receipt_validate(doc)
    assert report["ok"] is False
    clauses = {row["clause"] for row in report["failures"]}
    assert "receipt.hardware_fields.null_or_absent" in clauses


def test_receipt_write_does_not_swallow_hardware_claim():
    target = RECEIPTS / "SHOULD_NOT_EXIST_DEVPLATFORM.json"
    if target.exists():
        target.unlink()
    with pytest.raises(HardwareClaimError):
        dp.receipt_write(
            "SHOULD_NOT_EXIST_DEVPLATFORM.json",
            {"schema": "x", "tps": 1.0},
            "tools/future/test_devplatform.py",
        )
    assert not target.exists()


def test_model_profile_validates_known_specimens():
    for specimen_id in ("qwen3.8-27b-sealed-3.14", "qwen3.8-27b-mlx-4bit", "flash"):
        profile = dp.specimen_profile(specimen_id)
        report = dp.validate_model_profile(profile)
        assert report["ok"], (specimen_id, report["failures"])
        assert profile["schema"] == dp.MODEL_PROFILE_SCHEMA
        assert profile["version"] == 1
        for field in dp.MODEL_REQUIRED_FIELDS:
            assert field in profile


def test_model_profile_rejects_missing_required_and_catalog_drift():
    report = dp.validate_model_profile({"specimen_id": "nope"})
    assert report["ok"] is False
    assert any("missing required field" in item for item in report["failures"])

    drifted = dp.specimen_profile("qwen3.8-27b-sealed-3.14")
    drifted["family"] = "not-qwen"
    report = dp.validate_model_profile(drifted)
    assert report["ok"] is False
    assert any("does not match catalog" in item for item in report["failures"])


def test_machine_profile_matches_this_host():
    profile = dp.this_machine_profile()
    report = dp.validate_machine_profile(profile, against_live=True)
    assert report["ok"], report["failures"]
    assert profile["os"] == "Darwin"
    assert profile["arch"] == "arm64"
    assert profile["host_kind"] == "apple_silicon"
    assert profile["gpu_authority"] is False
    assert profile["measurement_state"] == "STATIC_ONLY"
    for field in HARDWARE_FIELDS:
        assert field not in profile

    other = dict(profile)
    other["arch"] = "x86_64"
    other["host_kind"] = "other"
    report = dp.validate_machine_profile(other, against_live=True)
    assert report["ok"] is False


def test_backend_contract_metal_present_others_not():
    contract = dp.metal_provider_contract()
    report = dp.validate_backend_contract(contract)
    assert report["ok"], report["failures"]
    assert contract["backend_id"] == "METAL"
    assert contract["available"] is True
    assert contract["availability"]["METAL"]["available"] is True
    for name in ("FPGA", "CUDA", "ANE"):
        assert contract["availability"][name]["available"] is False
    for method in dp.REQUIRED_BACKEND_METHODS:
        assert method in contract["required_methods"]
    assert "hbm_resident" not in contract["declared_features"]
    assert contract["not_an_fpga_backend"] is True
    assert contract["gpu_authority"] is False


def test_workunit_roundtrip_matches_hcli_content_hash():
    from hcli.workunit import WorkUnit, content_identity

    raw = {
        "id": "wu-1",
        "role": "code",
        "description": "round-trip the HCLI shape",
        "dependencies": ["wu-0"],
        "verifier": "pytest",
        "preferred_backend": "METAL",
        "provider": "mlx",
    }
    normalized = dp.workunit_normalize(raw)
    assert normalized["schema"] == dp.WORKUNIT_SCHEMA
    assert normalized["version"] == 1
    wu = WorkUnit.from_dict(normalized)
    assert content_identity(wu) == normalized["content_hash"]
    assert dp.workunit_content_hash(normalized) == content_identity(wu)
    report = dp.workunit_validate(normalized)
    assert report["ok"], report["failures"]

    ir = dp.workunit_to_ir(raw)
    first, second, decoded = dp.roundtrip_ir(ir)
    assert first == second
    assert decoded["kind"] == "workunit"
    assert decoded["payload"]["id"] == "wu-1"


def test_compat_accepts_matching_profile_backend_pair():
    model, machine, backend = dp._compatible_pair()
    wu = {
        "id": "ok",
        "role": "code",
        "description": "compatible",
        "preferred_backend": "mlx",
    }
    report = dp.check_compatibility(
        model=model, machine=machine, backend=backend, workunit=wu
    )
    assert report["ok"] is True, report["failures"]
    passed = {row["clause"] for row in report["passed"]}
    assert dp.CLAUSE_FEATURE_DECLARED in passed
    assert dp.CLAUSE_BACKEND_AVAILABLE in passed


def test_compat_rejects_undeclared_feature_and_names_the_clause():
    """NEGATIVE CONTROL: the refusal must actually fire and name the clause."""
    model, machine, backend = dp._incompatible_pair()
    assert "hbm_resident" in model["required_features"]
    assert "hbm_resident" not in backend["declared_features"]
    report = dp.check_compatibility(model=model, machine=machine, backend=backend)
    assert report["ok"] is False
    clauses = [row["clause"] for row in report["failures"]]
    assert dp.CLAUSE_FEATURE_DECLARED in clauses
    named = [row for row in report["failures"] if row["clause"] == dp.CLAUSE_FEATURE_DECLARED]
    assert named
    assert "hbm_resident" in named[0]["detail"]


def test_compat_rejects_fpga_backend_the_metal_provider_does_not_declare():
    model = dp.make_model_profile(
        "negative-control-fpga",
        family="qwen3.8",
        architecture="Qwen3.8",
        required_backend="FPGA",
        required_features=["chat_template_kwargs"],
    )
    report = dp.check_compatibility(
        model=model,
        machine=dp.this_machine_profile(),
        backend=dp.metal_provider_contract(),
    )
    assert report["ok"] is False
    clauses = {row["clause"] for row in report["failures"]}
    assert dp.CLAUSE_BACKEND_DECLARED in clauses
    assert dp.CLAUSE_BACKEND_AVAILABLE in clauses
    named = [
        row for row in report["failures"] if row["clause"] == dp.CLAUSE_BACKEND_DECLARED
    ]
    assert "FPGA" in named[0]["detail"]


def test_compat_rejects_metal_on_non_apple_host():
    model = dp.specimen_profile("qwen3.8-27b-mlx-4bit")
    machine = dp.this_machine_profile()
    machine["os"] = "Linux"
    machine["arch"] = "x86_64"
    machine["host_kind"] = "other"
    report = dp.check_compatibility(
        model=model, machine=machine, backend=dp.metal_provider_contract()
    )
    assert report["ok"] is False
    clauses = {row["clause"] for row in report["failures"]}
    assert dp.CLAUSE_MACHINE_CAN_RUN in clauses
