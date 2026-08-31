"""Token-roof decomposition: every loss named, no GPU-inefficiency bucket."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import token_roof_decomposition as trd
from tools.future._common import RECEIPTS


def test_every_transition_has_measured_loss_and_source():
    doc = trd.build()
    ids = [t["id"] for t in doc["transitions"]]
    assert ids == list(trd.TRANSITIONS)
    assert [s["id"] for s in doc["stages"]] == list(trd.STAGES)
    for row in doc["transitions"]:
        assert row["measured"] is True
        assert row["source_receipt"], row["id"]
        assert row["source_field"], row["id"]
        assert isinstance(row["loss_ms"], float)
        assert row["loss_ms"] >= 0.0
        assert row["loss_name"]
        assert row["loss_name"] not in trd.FORBIDDEN_BUCKETS
        assert "dispatches" in row
        assert "encoders" in row
        assert "command_buffers" in row
        assert "waits" in row
        assert "host_ms" in row
        assert "gpu_ms" in row
        assert "bytes_per_token" in row
        assert "useful_bytes_per_token" in row


def test_bytes_and_useful_bytes_are_different_columns():
    doc = trd.build()
    b = doc["bytes"]
    assert b["bytes_per_token"] != b["useful_bytes_per_token"]
    assert b["useful_bytes_per_token"] == b["bytes_per_token"] - b["auxiliary_bytes"]
    assert b["broadcast_aux_on_critical_path"] is False
    slower = b["removing_0p535_gb_made_things_slower"]
    assert slower["bytes_removed"] == 534773760
    assert slower["native_gb_s"] < slower["incumbent_gb_s"]
    for row in doc["transitions"] + doc["stages"]:
        assert row["bytes_per_token"] != row["useful_bytes_per_token"], row["id"]
        assert row["bytes_per_token"] == b["bytes_per_token"]
        assert row["useful_bytes_per_token"] == b["useful_bytes_per_token"]


def _loss_names(doc: dict) -> set[str]:
    names = {t["loss_name"] for t in doc["transitions"]}
    for t in doc["transitions"]:
        names.update(c["name"] for c in t.get("components") or [])
    names.update(doc.get("reconciliation", {}).get("gpu", {}).get("parts_ms", {}))
    names.update(doc.get("unrelated_losses_kept_apart", {}))
    return names


def test_no_gpu_inefficiency_bucket_in_receipt():
    doc = trd.build()
    trd.assert_no_forbidden_bucket(doc)
    assert doc["forbidden_bucket_present"] is False
    assert doc["forbidden_bucket"] == "GPU_INEFFICIENCY"
    names = _loss_names(doc)
    assert not any(trd._is_forbidden_bucket(n) for n in names)
    kept = set(doc["unrelated_losses_kept_apart"])
    assert kept == set(trd.UNRELATED_LOSSES)


def test_cannot_create_gpu_inefficiency_bucket():
    with pytest.raises(trd.ForbiddenLossBucket, match="GPU INEFFICIENCY"):
        trd.named_loss(
            name="GPU_INEFFICIENCY",
            ms=12.0,
            gb_s=200.0,
            source_receipt="receipts/future/MLP_ALU_ROOFLINE.json",
        )
    with pytest.raises(trd.ForbiddenLossBucket, match="GPU INEFFICIENCY"):
        trd.named_loss(
            name="gpu inefficiency",
            ms=12.0,
            source_receipt="receipts/future/MLP_ALU_ROOFLINE.json",
        )
    with pytest.raises(trd.ForbiddenLossBucket):
        trd.combine_losses(
            [
                {"name": "decode_arithmetic", "ms": 4.79},
                {"name": "addressing", "ms": 4.57},
                {"name": "deltanet_state_to_consume_stall", "ms": 0.182},
                {"name": "host_ceremony", "ms": 0.989},
            ],
            bucket="GPU_INEFFICIENCY",
        )
    with pytest.raises(trd.ForbiddenLossBucket):
        trd.combine_losses(
            [
                {"name": "decode_arithmetic", "ms": 4.79},
                {"name": "addressing", "ms": 4.57},
                {"name": "deltanet_state_to_consume_stall", "ms": 0.182},
                {"name": "host_ceremony", "ms": 0.989},
            ],
            bucket="token_tax",
        )


def test_unattributed_is_reported_with_size():
    doc = trd.build()
    u = doc["unattributed"]
    assert u["name"] == "UNATTRIBUTED"
    assert isinstance(u["ms"], float)
    recon = doc["reconciliation"]["gpu"]
    assert recon["unattributed_name"] == "UNATTRIBUTED"
    assert recon["unattributed_ms"] == pytest.approx(u["ms"])
    assert "UNATTRIBUTED" in recon["parts_ms"]
    t4 = next(t for t in doc["transitions"] if t["id"] == "REAL_DECODE_TO_COMPLETE_TOKEN")
    names = [c["name"] for c in t4["components"]]
    assert "UNATTRIBUTED" in names
    assert "deltanet_state_to_consume_stall" in names
    assert "host_ceremony" in names


def test_raises_when_parts_do_not_reconcile():
    with pytest.raises(trd.UnreconciledDecomposition, match="do not add up"):
        trd.reconcile(
            26.3026,
            {
                "clean_roof": 14.0,
                "addressing": 4.0,
                "geometry": 1.0,
                "real_decode": 4.0,
                "deltanet_state_to_consume_stall": 0.182,
                "UNATTRIBUTED": 0.0,
            },
        )
    with pytest.raises(trd.UnreconciledDecomposition, match="missing"):
        trd.reconcile(10.0, {"UNATTRIBUTED": 10.0})
    with pytest.raises(trd.UnreconciledDecomposition, match="UNATTRIBUTED must be present"):
        trd.reconcile(
            10.0,
            {"clean_roof": 10.0},
            required=("clean_roof",),
        )
    with pytest.raises(trd.EmptyGpuSample):
        trd.reconcile(0.0, {"UNATTRIBUTED": 0.0})
    closed = trd.reconcile(
        10.0,
        {"a": 6.0, "UNATTRIBUTED": 4.0},
        required=("a", "UNATTRIBUTED"),
    )
    assert closed["unattributed_ms"] == pytest.approx(4.0)
    assert closed["within_tolerance"] is True


def test_703_caveat_wherever_it_appears():
    doc = trd.build()
    trd.assert_703_qualified(doc)
    roof = doc["clean_kernel_roof"]
    assert roof["no_input_vector_load"] is True
    assert roof["usable_as_production_streaming_roof"] is False
    assert roof["guaranteed_production_bandwidth"] is False
    assert roof["kind"] == "MEASURED_CLEAN_KERNEL_ROOF"
    assert "NEVER guaranteed production bandwidth" in roof["clean_kernel_roof_caveat"]
    assert "input-vector load" in roof["clean_kernel_roof_caveat"]
    assert roof["statistic"] == "max"
    assert roof["median_gb_s"] == pytest.approx(699.5736545106142)
    assert roof["max_gb_s"] == pytest.approx(703.6072736347875)
    assert roof["campaign_label_gb_s"] == pytest.approx(703.5)
    clean = next(s for s in doc["stages"] if s["id"] == "CLEAN_ROOF")
    assert clean["no_input_vector_load"] is True
    assert clean["loads_activation"] is False
    t1 = next(t for t in doc["transitions"] if t["id"] == "CLEAN_ROOF_TO_ADDRESSING")
    assert t1["no_input_vector_load"] is True
    blob = json.dumps(doc)
    assert "NEVER guaranteed production bandwidth" in blob or "never as guaranteed" in blob.lower()
    with pytest.raises(trd.UnqualifiedCleanRoof):
        trd.assert_703_qualified({"gb_s": 703.5, "note": "the roof"})


def test_the_roof_rung_carries_the_same_caveat():
    """NOT the digits. The rung's value has moved three times - 66.54, 65.15,
    66.13 - and every move was accounting in the budget, not a change to this
    rung's standing. What must hold is that it reads its value from the budget
    and carries the no-input-vector-load caveat whatever that value is."""
    doc = trd.build()
    rung = doc["causal_budget_roof_on_todays_bytes"]
    import json as _json
    from tools.future import causal_budget_71 as _cb
    _ladder = _json.loads((_cb.RECEIPT).read_text())["ladder"]
    _live = next(
        r["tps"] for r in _ladder
        if r["rung"] == "every organ at the clean GEMV roof 703.5 GB/s"
    )
    assert rung["quoted_value"] == pytest.approx(_live)
    assert rung["no_input_vector_load"] is True
    assert rung["usable_as_production_streaming_roof"] is False
    assert rung["guaranteed_production_bandwidth"] is False
    assert "input-vector load" in rung["clean_kernel_roof_caveat"]
    assert rung["rests_on_roof_id"] == "q4_single_gemv_addr_13p6gb_max"
    assert rung["qualification"] == "NOT_QUALIFIED"
    assert rung["source_receipt"] == "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json"


def test_host_and_gpu_are_separated():
    doc = trd.build()
    split = doc["host_vs_gpu"]
    assert split["host_ms"] == pytest.approx(0.9894, abs=5e-4)
    assert split["gpu_ms"] == pytest.approx(26.3026, abs=5e-4)
    assert split["host_ms"] != split["gpu_ms"]
    t4 = next(t for t in doc["transitions"] if t["id"] == "REAL_DECODE_TO_COMPLETE_TOKEN")
    assert t4["host_ms"] == pytest.approx(split["host_ms"])
    host = next(c for c in t4["components"] if c["name"] == "host_ceremony")
    assert host["gpu_ms"] == pytest.approx(0.0)
    assert host["host_ms"] == pytest.approx(split["host_ms"])
    stall = next(c for c in t4["components"] if c["name"] == "deltanet_state_to_consume_stall")
    assert stall["host_ms"] == pytest.approx(0.0)
    assert stall["gpu_ms"] > 0
    decode = next(t for t in doc["transitions"] if t["id"] == "GEOMETRY_TO_REAL_DECODE")
    assert decode["host_ms"] == pytest.approx(0.0)
    assert decode["gpu_ms"] == pytest.approx(decode["loss_ms"])


def test_four_unrelated_losses_stay_apart():
    doc = trd.build()
    kept = doc["unrelated_losses_kept_apart"]
    assert kept["decode_arithmetic"]["transition"] == "GEOMETRY_TO_REAL_DECODE"
    assert kept["addressing"]["transition"] == "CLEAN_ROOF_TO_ADDRESSING"
    assert kept["deltanet_state_to_consume_stall"]["transition"] == "REAL_DECODE_TO_COMPLETE_TOKEN"
    assert kept["host_ceremony"]["transition"] == "REAL_DECODE_TO_COMPLETE_TOKEN"
    sources = {row["source_receipt"] for row in kept.values()}
    assert len(sources) == 4
    t3 = next(t for t in doc["transitions"] if t["id"] == "GEOMETRY_TO_REAL_DECODE")
    assert t3["loss_name"] == "decode_arithmetic"
    t1 = next(t for t in doc["transitions"] if t["id"] == "CLEAN_ROOF_TO_ADDRESSING")
    assert t1["loss_name"] == "catalog_topology_mixed_organs"
    stall_ms = kept["deltanet_state_to_consume_stall"]["ms"]
    decode_ms = kept["decode_arithmetic"]["ms"]
    addr_ms = kept["addressing"]["ms"]
    host_ms = kept["host_ceremony"]["ms"]
    assert stall_ms == pytest.approx(0.1821, abs=5e-4)
    assert decode_ms != addr_ms
    assert decode_ms != stall_ms
    assert host_ms != stall_ms


def test_gpu_reconciliation_closes_on_the_complete_token():
    doc = trd.build()
    recon = doc["reconciliation"]["gpu"]
    assert recon["within_tolerance"] is True
    whole = recon["whole_ms"]
    assert whole == pytest.approx(26.302583, abs=1e-6)
    parts = recon["parts_ms"]
    assert sum(parts.values()) == pytest.approx(whole, abs=1e-6)
    assert parts["clean_roof"] + parts["addressing"] + parts["geometry"] + parts[
        "real_decode"
    ] + parts["deltanet_state_to_consume_stall"] + parts["UNATTRIBUTED"] == pytest.approx(whole)
    wall = doc["reconciliation"]["wall"]
    assert wall["within_tolerance"] is True
    assert wall["whole_ms"] == pytest.approx(whole + wall["host_ceremony_ms"])


def test_transition_without_source_is_refused():
    with pytest.raises(trd.UnsourcedTransition):
        trd.named_loss(name="addressing", ms=1.0, source_receipt="")
    with pytest.raises(trd.TokenRoofError):
        trd.transition(
            ident="CLEAN_ROOF_TO_ADDRESSING",
            from_stage="CLEAN_ROOF",
            to_stage="ADDRESSING",
            loss_name="catalog_topology_mixed_organs",
            loss_ms=1.0,
            loss_gb_s=1.0,
            from_gb_s=703.5,
            to_gb_s=530.7,
            from_ms=14.0,
            to_ms=15.0,
            source_receipt="receipts/future/CATALOG_ADDRESSING.json",
            source_field="x",
            dispatches={},
            encoders={},
            command_buffers={},
            waits={},
            host_ms=0.0,
            gpu_ms=1.0,
            bytes_per_token=100,
            useful_bytes_per_token=100,
            native_measurement={},
            caveat=trd.CLEAN_KERNEL_ROOF_CAVEAT,
        )


def test_record_writes_sealed_receipt(tmp_path: Path):
    dest = tmp_path / "RESIDENT_TOKEN_ROOF_DECOMPOSITION.json"
    path = trd.record(path=dest)
    assert path == dest
    doc = json.loads(dest.read_text())
    assert doc["schema"] == trd.SCHEMA
    assert "seal_sha256" in doc
    assert doc["no_input_vector_load"] is True
    assert doc["unattributed"]["name"] == "UNATTRIBUTED"
    assert doc["gpu_authority"] is False
    assert doc["took_gpu_lease"] is False


def test_committed_receipt_if_present():
    path = RECEIPTS / trd.RECEIPT
    if not path.is_file():
        pytest.skip("receipt not recorded yet")
    doc = json.loads(path.read_text())
    assert doc["schema"] == trd.SCHEMA
    assert doc["no_input_vector_load"] is True
    assert doc["guaranteed_production_bandwidth"] is False
    ids = [t["id"] for t in doc["transitions"]]
    assert ids == list(trd.TRANSITIONS)
    for row in doc["transitions"]:
        assert row["source_receipt"]
        assert row["bytes_per_token"] != row["useful_bytes_per_token"]
        assert "dispatches" in row
        assert "encoders" in row
        assert "command_buffers" in row
        assert "waits" in row
        assert "host_ms" in row
        assert "gpu_ms" in row
    assert doc["unattributed"]["name"] == "UNATTRIBUTED"
    assert isinstance(doc["unattributed"]["ms"], float)
    names = _loss_names(doc)
    assert not any(trd._is_forbidden_bucket(n) for n in names)
    assert doc["causal_budget_66p54"]["quoted_value"] == pytest.approx(66.54)
    assert doc["causal_budget_66p54"]["no_input_vector_load"] is True
    trd.assert_703_qualified(doc)
    parts = doc["reconciliation"]["gpu"]["parts_ms"]
    whole = doc["reconciliation"]["gpu"]["whole_ms"]
    assert sum(parts.values()) == pytest.approx(whole, abs=1e-6)
