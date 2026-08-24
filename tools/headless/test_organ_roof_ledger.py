"""N027 three roofs + ORGAN_ROOF_LEDGER: never collapse, rank by recoverable ns."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from organ_roof_ledger import (  # noqa: E402
    ABSENT,
    CITED,
    DERIVED,
    MEASURED,
    ORGANS,
    RECEIPT,
    ROOF,
    SCHEMA,
    occupancy_factor,
    roofs_for_organ,
)

KINDS = {MEASURED, DERIVED, ABSENT, CITED}
RECEIPT_DOC = None
PEAK = 819.0
SUSTAINED = 778.8


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        assert RECEIPT.is_file(), (
            f"missing {RECEIPT} — run python3 tools/headless/organ_roof_ledger.py"
        )
        RECEIPT_DOC = json.loads(RECEIPT.read_text())
    return RECEIPT_DOC


def _walk_qty(obj, path=""):
    if isinstance(obj, dict):
        if "kind" in obj and obj["kind"] in KINDS and "command" in obj and "unit" in obj:
            yield path, obj
        for k, v in obj.items():
            yield from _walk_qty(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_qty(v, f"{path}[{i}]")


def test_receipt_schema_and_discipline():
    doc = receipt()
    assert RECEIPT.is_file()
    assert doc["schema"] == SCHEMA
    assert "N027" in doc["obligation"]
    assert "S022" in doc["obligation"]
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_mutate_parent"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["did_not_rederive_dram_roof"] is True
    assert doc["dense_w_materialized"] == 0
    assert doc["gpu_confirm"]["ran"] is False
    assert set(doc["organs_named"]) == set(ORGANS)
    assert doc["ranking_quantity"] == "recoverable_token_ns"
    assert "pct_BW" in doc["not_the_ranking_quantity"]
    assert "10+ GiB" in doc["occupancy"]["note"] or "27B" in doc["occupancy"]["note"]


def test_three_roofs_are_separate_and_never_collapsed():
    doc = receipt()
    roofs = doc["three_roofs"]
    assert roofs["never_collapsed"] is True
    assert roofs["not_a_single_number"] is True
    assert roofs["distinct"] is True
    theo = roofs["DEVICE_THEORETICAL"]
    meas = roofs["DEVICE_MEASURED_SUSTAINED"]
    reach = roofs["MODEL_REACHABLE"]
    assert theo["kind"] == CITED
    assert meas["kind"] == CITED
    assert reach["kind"] == DERIVED
    assert theo["value"] == PEAK
    assert meas["value"] == SUSTAINED
    assert reach["value"] is not None
    assert theo["value"] != meas["value"]
    assert reach["value"] != theo["value"]
    assert reach["value"] != meas["value"]
    assert 100.0 < reach["value"] < PEAK
    # Sealed 778.8 is copied, not the raw 778.758 sweep median rounded here.
    roof = json.loads(ROOF.read_text())
    sealed = roof["anchor_roof"]["correction"]["new_roof_gb_s"]
    assert meas["value"] == sealed
    raw = roof["answer"]["highest_dram_read_gb_s"]
    assert abs(raw - SUSTAINED) > 0.01  # would be a collapse if we used raw as the metric
    assert "not re-derived" in (meas.get("note") or "").lower() or "Copied" in (meas.get("note") or "")
    # No singular collapsed field.
    assert "the_roof" not in doc
    assert "the_roof_gb_s" not in doc
    assert "single_roof" not in doc


def test_model_reachable_method_is_stated():
    doc = receipt()
    m = doc["model_reachable_method"]
    assert "arithmetic intensity" in m["statement"].lower() or "AI" in m["formula"]
    assert "occupancy" in m["statement"].lower()
    assert "1-CB" in m["statement"] or "command buffer" in m["dependency_structure"].lower()
    assert "recoverable_token_ns" in m["formula"]
    assert m["saturating_threadgroups"] == 240
    assert m["gpu_cores"] == 60
    assert "sampling" in m["why_not_pct_bw"].lower()
    assert doc["s022"]["section_1"]
    assert doc["s022"]["section_11"]
    assert doc["s022"]["section_12"]


def test_each_organ_has_required_fields():
    doc = receipt()
    organs = doc["organs"]
    assert set(organs) == set(ORGANS)
    for name in ORGANS:
        o = organs[name]
        assert o["bytes"]["traffic_bytes"]["value"] > 0, name
        assert o["bytes"]["weight_read"]["kind"] in KINDS
        assert o["arithmetic_intensity"]["kind"] in KINDS
        assert o["bound_class"] in ("memory", "compute", "launch")
        assert o["measured_bw"]["weight_gb_s"]["kind"] in KINDS
        assert o["measured_compute"]["kind"] in KINDS
        assert o["dispatch_sync"]["n_dispatches"]["value"] >= 1
        assert o["dispatch_sync"]["intra_cb_synchronization_ns"]["value"] == 0.0
        assert o["dispatch_sync"]["per_dispatch_gpu_ns"]["kind"] == ABSENT
        assert o["occupancy"]["hardware_counter"]["kind"] == ABSENT
        assert o["occupancy"]["launch_geometry_factor"]["kind"] == DERIVED
        assert 0 < o["occupancy"]["launch_geometry_factor"]["value"] <= 1.0
        assert o["complete_ns"]["gpu_ns"]["value"] > 0
        assert o["complete_ns"]["complete_wall_ns"]["kind"] == ABSENT
        lim = o["limits"]
        for key in (
            "theoretical_gb_s",
            "measured_device_gb_s",
            "reachable_gb_s",
            "theoretical_ns",
            "measured_device_ns",
            "reachable_ns",
        ):
            assert lim[key]["value"] is not None, (name, key)
            assert lim[key]["kind"] == DERIVED
        # Device theoretical (819) is never the organ reachable roof.
        assert lim["theoretical_gb_s"]["value"] != lim["reachable_gb_s"]["value"], name
        rec = o["recoverable_token_ns"]
        assert rec["kind"] == DERIVED
        assert rec["value"] >= 0
        assert o["dense_w_materialized"] == 0


def test_ranked_by_recoverable_token_ns_not_pct_bw():
    doc = receipt()
    ranked = doc["ranked_by_recoverable_token_ns"]
    names = [r["organ"] for r in ranked]
    assert names[0] == doc["largest_recoverable_organ"]
    assert set(names) == set(ORGANS)
    vals = [r["recoverable_token_ns"] for r in ranked]
    assert vals == sorted(vals, reverse=True)
    assert ranked[0]["organ"] == "mlp_gate_up"
    assert names[-1] in ("embedding", "sampling")
    # %BW ranking is recorded so we can prove we rejected it.
    pct = doc["ranked_by_pct_of_measured_bw_is_not_the_ranking"]["rows"]
    pct_names = [r["organ"] for r in pct]
    assert pct_names != names
    assert pct_names[0] in ("sampling", "embedding")
    assert "mlp_gate_up" in names[:3]
    assert "778.8" in doc["one_line"] or "778.8" in doc["answer"]
    assert "recoverable" in doc["one_line"].lower() or "token_ns" in doc["one_line"]
    rec_idx = {name: i for i, name in enumerate(names)}
    assert rec_idx["mlp_gate_up"] == 0
    assert rec_idx["mlp_down"] < rec_idx["sampling"]
    assert rec_idx["deltanet"] < rec_idx["embedding"]
    # N025 ranked deltanet above mlp_down by % of the 778.8 gap. Recoverable
    # token_ns does not have to preserve that order.
    assert rec_idx["mlp_down"] <= rec_idx["deltanet"] + 1


def test_recoverable_ns_matches_measured_minus_reachable():
    doc = receipt()
    for name, o in doc["organs"].items():
        meas = o["complete_ns"]["gpu_ns"]["value"]
        reach = o["limits"]["reachable_ns"]["value"]
        rec = o["recoverable_token_ns"]["value"]
        assert abs(rec - max(0.0, meas - reach)) < 1e-6, name


def test_memory_bound_high_occupancy_can_reach_sustained_not_theoretical():
    """A saturated memory-bound GEMV's reachable BW is 778.8, not 819."""
    r = roofs_for_organ(
        traffic_bytes=3_574_857_728,
        flops=22_824_222_720,
        occupancy=1.0,
        peak_gb_s=PEAK,
        sustained_gb_s=SUSTAINED,
        compute_gflops=8979.0,
    )
    assert r["bound_class"] == "memory"
    assert abs(r["organ_measured_device_gb_s"] - SUSTAINED) < 1e-9
    assert abs(r["organ_theoretical_gb_s"] - PEAK) < 1e-9
    assert abs(r["organ_reachable_gb_s"] - SUSTAINED) < 1e-9
    assert r["organ_theoretical_gb_s"] != r["organ_reachable_gb_s"]


def test_compute_bound_roof_is_below_dram():
    """High AI cannot claim the DRAM roof. MODEL_REACHABLE ≠ 778.8."""
    r = roofs_for_organ(
        traffic_bytes=1_000_000,
        flops=50_000_000,  # 50 FLOP/byte >> ridge 11.5
        occupancy=1.0,
        peak_gb_s=PEAK,
        sustained_gb_s=SUSTAINED,
        compute_gflops=8979.0,
    )
    assert r["bound_class"] == "compute"
    assert r["organ_reachable_gb_s"] < SUSTAINED
    assert r["organ_reachable_gb_s"] == r["compute_limited_gb_s"]


def test_occupancy_starved_cannot_claim_dram_roof():
    r = roofs_for_organ(
        traffic_bytes=993_284,
        flops=0.0,
        occupancy=occupancy_factor(1),
        peak_gb_s=PEAK,
        sustained_gb_s=SUSTAINED,
        compute_gflops=8979.0,
    )
    assert r["bound_class"] == "launch"
    assert r["organ_reachable_gb_s"] < 10.0
    assert r["organ_reachable_gb_s"] < SUSTAINED / 10


def test_absent_has_physical_reason_never_a_number():
    doc = receipt()
    n_absent = 0
    for path, q in _walk_qty(doc):
        if q["kind"] == ABSENT:
            n_absent += 1
            assert q["value"] is None, path
            assert q.get("absent_reason"), path
        else:
            assert q["value"] is not None, path
            assert q.get("absent_reason") is None, path
            assert q.get("command"), path
    assert n_absent >= 8
    # Hardware occupancy and per-dispatch timestamps stay ABSENT.
    sample = doc["organs"]["sampling"]
    assert sample["occupancy"]["hardware_counter"]["kind"] == ABSENT
    assert sample["dispatch_sync"]["per_dispatch_gpu_ns"]["kind"] == ABSENT
    assert sample["complete_ns"]["complete_wall_ns"]["kind"] == ABSENT
    reason = sample["occupancy"]["hardware_counter"]["absent_reason"].lower()
    assert "counter" in reason or "timestamp" in reason


def test_inputs_are_cited_not_remeasured():
    doc = receipt()
    reused = doc["inputs_reused_not_remeasured"]
    for name in (
        "ORGAN_BANDWIDTH.json",
        "DISPATCH_LEDGER.json",
        "GPU_LEDGER.json",
        "BANDWIDTH_ROOF.json",
    ):
        assert name in reused
    assert doc["gpu_confirm"]["optional"] is True
    law = doc["causal_benchmark_law"]
    assert "three" in law["sentinel"].lower()
    assert "pctBW" in law["bad_control"].replace("-", "") or "%BW" in law["bad_control"]


def test_token_model_reachable_is_a_third_number():
    doc = receipt()
    token = doc["token"]
    reach_ns = token["model_reachable_ns"]["value"]
    recov = token["recoverable_token_ns_sum"]["value"]
    gpu = token["production_gpu_ns"]["value"]
    assert reach_ns > 0
    assert recov > 0
    assert abs((reach_ns + recov) - gpu) / gpu < 0.02
    mr = doc["three_roofs"]["MODEL_REACHABLE"]["value"]
    assert mr > token["production_achieved_gb_s"]["value"]
    assert mr < SUSTAINED
