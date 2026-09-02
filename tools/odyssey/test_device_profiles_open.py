"""Specimen-open economics and the Tier-1 warm-set live on device_profiles.

Call sites: economics_from_genome must invoke specimen_open_economics and
warm_set_policy; those must invoke specimen_open.load_receipt and use
modellake.TIER1_BUDGET. An import is not a call site.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

from tools.odyssey import device_profiles as dp
from tools.odyssey import modellake as ml
from tools.odyssey import specimen_open as so


def test_economics_from_genome_calls_open_and_warm_set():
    names = dp.economics_from_genome.__code__.co_names
    assert "specimen_open_economics" in names
    assert "warm_set_policy" in names


def test_specimen_open_economics_calls_load_receipt():
    assert "load_receipt" in dp.specimen_open_economics.__code__.co_names


def test_warm_set_policy_uses_tier1_budget():
    assert "TIER1_BUDGET" in dp.warm_set_policy.__code__.co_names
    assert "SSD_STAGE" in dp.warm_set_policy.__code__.co_names


def test_missing_receipt_is_blocked_not_guessed(monkeypatch):
    monkeypatch.setattr(so, "load_receipt", lambda path=None: None)
    out = dp.specimen_open_economics()
    assert out["status"] == "BLOCKED"
    assert out["evidence_tier"] == "STATIC"
    assert "refusing to invent" in out["reason"]
    assert out["called"] == "specimen_open.load_receipt"


def test_overlay_keeps_hardware_measured_on_the_inputs():
    rec = {
        "schema": "hawking.odyssey.fast_specimen_open.v1",
        "sequential_rate": {
            "median_cold_gb_s": 0.2,
            "evidence_tier": "HARDWARE_MEASURED",
            "volume": "/Volumes/corpdrive (APFS over USB)",
        },
        "per_specimen": [{
            "id": "Qwen--Qwen3-0.6B@c1899de289a0",
            "file_bytes": 1503300328,
            "metadata_only": {
                "cold_s": 0.01, "warm_s": 0.001, "cache_hit_s": 0.0001,
                "bytes_read_cold": 35560, "touched_weight_bytes": False,
            },
            "first_usable_tensor": {"smallest_cold": {"seconds": 0.002}},
            "full_shards": {"cold_s": 8.0, "warm_s": 0.2, "cold_gb_s": 0.19},
            "before_after": {"metadata_common_path": {"speedup_cold_vs_full": 800}},
        }],
        "bottleneck": {"named": "full sequential read of weight bytes on USB."},
    }
    out = dp.specimen_open_economics(receipt=rec)
    assert out["status"] == "OK"
    assert out["evidence_tier"] == "COST_MODEL"
    assert out["sequential_evidence_tier"] == "HARDWARE_MEASURED"
    assert out["called"] == "specimen_open.load_receipt"
    row = out["specimens"][0]
    assert row["id"].startswith("Qwen--Qwen3-0.6B")
    assert row["metadata_touched_weight_bytes"] is False
    assert row["inputs_evidence_tier"] == "HARDWARE_MEASURED"
    assert "HARDWARE_MEASURED" in out["note"] and "COST_MODEL" in out["note"]


def test_warm_set_refuses_to_become_a_second_archive(tmp_path):
    cands = [
        {"id": "a--x@1", "bytes": 80 * 2**30},
        {"id": "b--x@2", "bytes": 80 * 2**30},
        {"id": "c--x@3", "bytes": 80 * 2**30},
    ]
    d = dp.warm_set_policy(cands, budget=140 * 2**30, max_n=2, stage_dir=tmp_path)
    assert d["copied"] is False
    assert d["copied_the_lake"] is False
    assert d["installed"] is False
    assert d["evidence_tier"] == "COST_MODEL"
    assert len(d["selected"]) == 1
    assert d["selected"][0]["id"] == "a--x@1"
    assert any("would exceed" in (r.get("refused") or "") for r in d["refused"])


def test_warm_set_caps_at_two_even_when_budget_allows(tmp_path):
    cands = [{"id": f"s{i}", "bytes": 10} for i in range(5)]
    d = dp.warm_set_policy(cands, budget=10_000, max_n=2, stage_dir=tmp_path)
    assert len(d["selected"]) == 2
    assert any("at most 2" in (r.get("refused") or "") for r in d["refused"])


def test_warm_set_uses_the_real_tier1_budget_constant(tmp_path):
    d = dp.warm_set_policy([{"id": "tiny", "bytes": 1}], stage_dir=tmp_path)
    assert d["budget_bytes"] == ml.TIER1_BUDGET
    assert d["selected"][0]["id"] == "tiny"


def test_a_body_larger_than_the_budget_is_refused(tmp_path):
    d = dp.warm_set_policy(
        [{"id": "the-lake", "bytes": 4_350_000_000_000}],
        budget=ml.TIER1_BUDGET,
        stage_dir=tmp_path,
    )
    assert d["selected"] == []
    assert "exceeds tier1 budget" in d["refused"][0]["refused"]


def test_warm_set_does_not_copy_or_delete():
    src = inspect.getsource(dp.warm_set_policy)
    assert "shutil" not in src
    assert "subprocess" not in src
    assert "rmtree" not in src
    assert "os.rename" not in src
    assert '"copied": False' in src


def test_warm_set_ignores_config_only_stage_stubs(tmp_path):
    stub = tmp_path / "kimi-vl-a3b"
    stub.mkdir()
    (stub / "config.json").write_text("{}")
    real = tmp_path / "Qwen--Qwen3-0.6B@c1899de289a0"
    real.mkdir()
    header = json.dumps({"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (real / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"\x00\x00\x00\x00")
    d = dp.warm_set_policy(stage_dir=tmp_path)
    assert [s["id"] for s in d["selected"]] == ["Qwen--Qwen3-0.6B@c1899de289a0"]


def test_warm_set_admits_the_two_measured_specimens_and_refuses_the_lake(tmp_path):
    """Qwen 1.4 GiB + Falcon 14 GiB fit the 140 GiB / 2-body bench. 360 GB does not."""
    d = dp.warm_set_policy(
        [
            {"id": "Qwen--Qwen3-0.6B@c1899de289a0", "bytes": 1503300328},
            {"id": "tiiuae--Falcon-H1-7B-Instruct@41e72f27effb", "bytes": 15171394304},
            {"id": "Qwen--Qwen3.8-Flash-Next@34567a4712bc", "bytes": 360_000_000_000},
        ],
        budget=ml.TIER1_BUDGET,
        max_n=2,
        stage_dir=tmp_path,
    )
    ids = {s["id"] for s in d["selected"]}
    assert ids == {
        "Qwen--Qwen3-0.6B@c1899de289a0",
        "tiiuae--Falcon-H1-7B-Instruct@41e72f27effb",
    }
    assert d["copied"] is False
    assert any("exceeds tier1 budget" in (r.get("refused") or "") for r in d["refused"])


def test_economics_from_genome_carries_the_overlay():
    g = {
        "memory_bytes": 96 * 2**30,
        "genome_digest": "test",
        "domains": {
            "uma_0": {"kind": "UMA", "present": True},
            "storage": {"kind": "STORAGE", "present": True, "mounts": []},
        },
    }
    econ = dp.economics_from_genome(g, profile="INTERACTIVE")
    assert econ["evidence_tier"] == "COST_MODEL"
    assert "specimen_open" in econ
    assert "warm_set" in econ
    assert econ["warm_set"]["copied"] is False
    assert econ["warm_set"]["installed"] is False
    assert econ["fpga_present"] is False
