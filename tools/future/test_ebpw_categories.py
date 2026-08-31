"""Negative controls for the EBPW category validator.

A guard nobody has watched fail is not a guard.
"""
from __future__ import annotations

import json

import pytest

from tools.future import ebpw_categories as ec
from tools.future._common import RECEIPTS


def test_build_emits_sealed_receipt():
    out = ec.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "EBPW_CATEGORY_VALIDATOR.json"
    assert doc["schema"] == "hawking.future.ebpw_categories.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["selftest"]["meta_only_0_88_refused"] is True
    assert doc["selftest"]["cross_category_arithmetic_raises"] is True
    assert doc["selftest"]["dense_rematerializing_nx_rejected"] is True
    assert doc["selftest"]["honest_flash_meta_green"] is True
    assert doc["selftest"]["synthetic_promotion_variant_refused"] is True


def test_five_categories_are_distinct_types():
    values = [
        ec.ProspectiveMetaBpw(0.88),
        ec.SerializedMetaBytes(1024),
        ec.CompileTimeNrBytes(2048),
        ec.NxRuntimeBytes(4096),
        ec.CompletePhysicalEbpw(2.4),
    ]
    types = [type(v) for v in values]
    assert len(set(types)) == 5
    assert [v.category for v in values] == list(ec.CATEGORY_TYPES)


def test_same_category_arithmetic_is_ok():
    a = ec.SerializedMetaBytes(10)
    b = ec.SerializedMetaBytes(3)
    assert (a + b).value == 13
    assert (a - b).value == 7
    assert type(a + b) is ec.SerializedMetaBytes


def test_cross_category_arithmetic_raises():
    """NEGATIVE CONTROL: mixed-category ops must actually fire."""
    meta = ec.ProspectiveMetaBpw(0.88)
    phys = ec.CompletePhysicalEbpw(0.88)
    with pytest.raises(ec.CategoryError, match="not interchangeable"):
        _ = meta + phys
    with pytest.raises(ec.CategoryError, match="not interchangeable"):
        _ = meta - phys
    with pytest.raises(ec.CategoryError, match="not interchangeable"):
        _ = meta * phys
    with pytest.raises(ec.CategoryError, match="not interchangeable"):
        _ = meta / phys
    with pytest.raises(ec.CategoryError):
        _ = meta + 0.88
    with pytest.raises(ec.CategoryError):
        _ = 0.88 + meta
    with pytest.raises(ec.CategoryError, match="cannot compare"):
        _ = meta == phys
    with pytest.raises(ec.CategoryError):
        _ = float(meta)
    with pytest.raises(ec.CategoryError):
        _ = int(phys)
    nr = ec.CompileTimeNrBytes(16)
    nx = ec.NxRuntimeBytes(16)
    with pytest.raises(ec.CategoryError):
        _ = nr + nx
    with pytest.raises(ec.CategoryError):
        _ = ec.SerializedMetaBytes(16) + phys


def test_cannot_coerce_across_categories():
    with pytest.raises(ec.CategoryError, match="cannot coerce"):
        ec._coerce(
            ec.CompletePhysicalEbpw,
            ec.ProspectiveMetaBpw(0.88),
            "launder",
        )


def test_can_promote_refuses_meta_only_088():
    """NEGATIVE CONTROL: prospective_meta_bpw 0.88 and nothing else never promotes."""
    ok, reason = ec.can_promote({"prospective_meta_bpw": 0.88})
    assert ok is False
    assert "never promotes alone" in reason
    assert "byte ledger" in reason
    assert "capability-preserving runtime" in reason
    assert "complete_physical_ebpw" in reason

    ok2, reason2 = ec.can_promote(
        ec.PromotionLedger(
            prospective_meta_bpw=ec.ProspectiveMetaBpw(0.88, evidence="synthetic")
        )
    )
    assert ok2 is False
    assert "never promotes alone" in reason2


def test_can_promote_refuses_meta_with_caveat_and_flag():
    """A flag or caveat around a sub-1 budget is still not a promotion."""
    ok, reason = ec.can_promote(
        {
            "prospective_meta_bpw": 0.88,
            "promotion_allowed": True,
            "promotion_caveat": "budget only",
            "force_promote": True,
        }
    )
    assert ok is False
    assert "never promotes alone" in reason
    result = ec.validate(
        {
            "prospective_meta_bpw": 0.88,
            "promotion_allowed": True,
            "promotion_caveat": "budget only",
        }
    )
    assert result["verdict"] == "REFUSED"
    assert result["can_promote"] is False


def test_dense_rematerializing_nx_is_rejected():
    """NEGATIVE CONTROL: production decompress-then-ordinary-kernels is refused."""
    nx = {
        "schema": "hawking.flash.nx_genome.v1",
        "status": "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION",
        "path_kind": ec.PRODUCTION,
        "execution_path": {
            "decompresses_to_dense_weight_tensor": True,
            "runs_ordinary_kernels": True,
        },
    }
    remat = ec.judge_dense_rematerialization(nx)
    assert remat.ok is False
    assert "dense weight tensor" in remat.reason
    result = ec.validate(nx)
    assert result["verdict"] == "REFUSED"
    ok, reason = ec.can_promote(nx)
    assert ok is False
    assert "dense" in reason.lower() or "rematerial" in reason.lower()


def test_verification_may_reconstruct_but_cannot_promote():
    nx = {
        "path_kind": ec.VERIFICATION,
        "execution_path": {
            "decompresses_to_dense_weight_tensor": True,
            "runs_ordinary_kernels": True,
        },
    }
    remat = ec.judge_dense_rematerialization(nx)
    assert remat.ok is True
    assert remat.path_kind == ec.VERIFICATION
    result = ec.validate(nx)
    assert result["verdict"] == "GREEN"
    ok, reason = ec.can_promote(nx)
    assert ok is False
    assert "verification" in reason


def test_honest_flash_meta_is_green_and_does_not_promote():
    result = ec.validate_honest_flash_meta()
    assert result["verdict"] == "GREEN"
    assert result["can_promote"] is False
    assert result["quantities"]["prospective_meta_bpw"]["value"] is not None
    assert result["quantities"]["prospective_meta_bpw"]["value"] < 1.0
    assert result["quantities"]["complete_physical_ebpw"]["value"] is None
    assert result["quantities"]["serialized_meta_bytes"]["value"] is None
    doc, _via = ec.load_named_receipt(
        "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json"
    )
    if doc is not None:
        assert doc["status"] == "PROSPECTIVE_META_ONLY"
        assert doc["measurement_state"]["serialized_artifact"] == "NOT_BUILT"
        assert doc["measurement_state"]["physical_loader"] == "NOT_BUILT"
        assert doc["measurement_state"]["native_kernel"] == "NOT_BUILT"
        assert doc["measurement_state"]["physical_ebpw"] == "NULL_BY_RULE"
        assert doc["measurement_state"]["promotion_allowed"] is False


def test_synthetic_promotion_variant_is_refused():
    """NEGATIVE CONTROL: the honest receipt, mutated to claim promotion, is refused."""
    synthetic = json.loads(json.dumps(ec.HONEST_FLASH_META_MINIMAL))
    synthetic["promotion_allowed"] = True
    synthetic["measurement_state"]["promotion_allowed"] = True
    synthetic["measurement_state"]["physical_ebpw"] = 0.88
    synthetic["metric"]["physical_ebpw"] = 0.88
    result = ec.validate(synthetic)
    assert result["verdict"] == "REFUSED"
    assert result["can_promote"] is False
    assert any("physical EBPW" in r for r in result["reasons"])


def test_active_bytes_fields_stay_separate():
    acc = ec.ActiveBytesAccounting(
        total_artifact_bytes=10_000,
        resident_bytes=4_000,
        active_bytes_per_token=800,
        actual_read_bytes_per_token=700,
        transient_bytes=200,
    )
    with pytest.raises(ec.CategoryError, match="collapsed"):
        acc.collapsed_total()
    with pytest.raises(ec.CategoryError):
        _ = acc + acc
    with pytest.raises(ec.CategoryError):
        _ = acc + ec.SerializedMetaBytes(10)
    dumped = acc.as_dict()
    assert dumped["total_artifact_bytes"] == 10_000
    assert dumped["resident_bytes"] == 4_000
    assert dumped["active_bytes_per_token"] == 800
    assert dumped["actual_read_bytes_per_token"] == 700
    assert dumped["transient_bytes"] == 200


def test_positive_combinator_opens_only_when_every_predicate_holds():
    """The gate can open. A constant-False can_promote would still pass the refusals."""
    full = ec.PromotionLedger(
        complete_physical_ebpw=ec.CompletePhysicalEbpw(
            2.4, evidence="synthetic combinator control (not a measurement)"
        ),
        executable_byte_ledger={
            "self_contained": True,
            "for_this_executable": True,
            "complete_storage_bytes": 4096,
        },
        capability_preserving_runtime=True,
        physical_measurement_authority=ec.PROTECTED,
        bench_state="PROTECTED",
        measurement_state=ec.PROTECTED,
        path_kind=ec.PRODUCTION,
        dense_rematerialization=False,
        consumes_representation_directly=True,
    )
    ok, reason = ec.can_promote(full)
    assert ok is True, reason
    assert "all promotion predicates held" in reason

    missing_runtime = ec.PromotionLedger(
        complete_physical_ebpw=full.complete_physical_ebpw,
        executable_byte_ledger=full.executable_byte_ledger,
        capability_preserving_runtime=False,
        physical_measurement_authority=ec.PROTECTED,
        bench_state="PROTECTED",
        measurement_state=ec.PROTECTED,
        path_kind=ec.PRODUCTION,
        dense_rematerialization=False,
        consumes_representation_directly=True,
    )
    ok2, reason2 = ec.can_promote(missing_runtime)
    assert ok2 is False
    assert "capability-preserving runtime" in reason2


def test_diagnostic_relative_never_promotes():
    ledger = ec.PromotionLedger(
        complete_physical_ebpw=ec.CompletePhysicalEbpw(0.9, evidence="diagnostic"),
        executable_byte_ledger={
            "self_contained": True,
            "for_this_executable": True,
            "complete_storage_bytes": 4096,
        },
        capability_preserving_runtime=True,
        physical_measurement_authority=ec.DIAGNOSTIC,
        bench_state="PROTECTED",
        measurement_state=ec.DIAGNOSTIC,
        path_kind=ec.PRODUCTION,
        dense_rematerialization=False,
        consumes_representation_directly=True,
    )
    ok, reason = ec.can_promote(ledger)
    assert ok is False
    assert "DIAGNOSTIC_RELATIVE" in reason


def test_inventory_reports_five_quantities_per_receipt():
    rows = [ec.inventory_path(p) for p in ec.INVENTORY_PATHS]
    present = [r for r in rows if r["present"]]
    assert present, "expected at least FLASH_EBPW_BUDGET / QWEN80 / namespace receipt"
    for row in present:
        assert set(row["quantities"]) == set(ec.CATEGORY_TYPES)
        assert row["can_promote"] is False
        assert row["verdict"] in {"GREEN", "REFUSED"}


def test_selftest_watches_the_guards_fail():
    controls = ec.selftest()
    assert controls["meta_only_0_88_refused"] is True
    assert controls["cross_category_arithmetic_raises"] is True
    assert controls["dense_rematerializing_nx_rejected"] is True
    assert controls["honest_flash_meta_green"] is True
    assert controls["synthetic_promotion_variant_refused"] is True
    assert controls["verification_may_reconstruct"] is True
    assert controls["verification_cannot_promote"] is True
    assert controls["positive_combinator_opens"] is True
    assert controls["positive_combinator_is_not_a_measurement"] is True
