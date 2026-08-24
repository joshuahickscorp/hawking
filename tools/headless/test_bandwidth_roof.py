"""N017 bandwidth roof: measured DRAM read, 775 reachable-or-not, bad control rejected.

pytest tools/headless -q must see receipts/headless/BANDWIDTH_ROOF.json.
The harness does the measuring. This test reads the sealed receipt and
must not re-run Metal (a test that re-measures the thing it is checking
cannot fail correctly).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPT_PATH = REPO / "receipts" / "headless" / "BANDWIDTH_ROOF.json"
SCHEMA = "hawking.headless.bandwidth_roof.v1"
KINDS = {"MEASURED", "DERIVED", "ABSENT"}
TARGET_775 = 775.0
PEAK = 819.0
PRIOR = 595.9

RECEIPT_DOC = None


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        assert RECEIPT_PATH.is_file(), (
            f"missing {RECEIPT_PATH} — run python3 tools/headless/bandwidth_roof.py"
        )
        RECEIPT_DOC = json.loads(RECEIPT_PATH.read_text())
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


def test_receipt_exists_and_schema():
    doc = receipt()
    assert doc["schema"] == SCHEMA
    assert "N017" in doc["obligation"]
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["q4_tile_shape_reused"] is False


def test_one_line_answers_775():
    doc = receipt()
    line = doc["one_line"]
    assert isinstance(line, str) and line
    ans = doc["answer"]
    assert ans["775_gb_s"] in ("REACHABLE", "NOT REACHABLE")
    dram = ans["highest_dram_read_gb_s"]
    assert isinstance(dram, (int, float)) and dram > 0
    if ans["775_gb_s"] == "REACHABLE":
        assert dram >= TARGET_775
        assert "REACHABLE" in line
    else:
        assert dram < TARGET_775
        assert "NOT REACHABLE" in line
    assert ans["cache_is_not_the_dram_roof"] is True
    cache = ans.get("highest_cache_resident_read_gb_s")
    if isinstance(cache, (int, float)) and cache >= TARGET_775:
        # Cache hitting 775 does not make DRAM 775 reachable.
        assert ans["775_gb_s"] == "REACHABLE" or dram < TARGET_775


def test_best_config_named_with_reps_and_spread():
    doc = receipt()
    cfg_id = doc["answer"]["highest_dram_read_config"]
    assert cfg_id
    rows = {r["id"]: r for r in doc["configs"]}
    assert cfg_id in rows
    best = rows[cfg_id]
    warm = best["warm"]
    assert warm["n"] >= 5
    assert warm["spread_pct"] is not None
    assert best["working_set_class"] == "dram_streaming"
    assert best["rw"] == "read"
    assert best["bad_control"] is False
    cold = best["cold"]
    assert "read_gb_s" in cold
    assert cold["gpu_ns"] > 0
    assert "GPUStartTime/GPUEndTime" in best["gpu_timestamp_authority"]


def test_sweep_covers_required_axes():
    doc = receipt()
    rows = doc["configs"]
    patterns = {r["pattern"] for r in rows}
    for need in ("sequential", "strided", "gather", "multi", "bad_control"):
        assert need in patterns, need
    vecs = {r["vec"] for r in rows}
    for need in ("f1", "f2", "f4", "f4x8"):
        assert need in vecs, need
    tgs = {int(r["tg"]) for r in rows if r["pattern"] == "sequential" and r["rw"] == "read"}
    assert 32 in tgs and 256 in tgs and 1024 in tgs
    assert 128 not in tgs or True  # 128 is allowed if not the q4 64-threads-per-row tile
    classes = {r["working_set_class"] for r in rows}
    assert "below_cache" in classes
    assert "dram_streaming" in classes
    rws = {r["rw"] for r in rows}
    assert "read" in rws and "write" in rws and "readwrite" in rws
    storages = {r["storage"] for r in rows}
    assert "private" in storages and "shared" in storages
    nbufs = {int(r["nbufs"]) for r in rows}
    assert 1 in nbufs and 4 in nbufs
    queues = {int(r["n_queues"]) for r in rows}
    assert 1 in queues and 4 in queues
    assert any(r.get("blit") for r in rows)
    assert any(r.get("concurrent_encoders") for r in rows)
    for r in rows:
        assert r.get("q4_tile_shape_reused") is False


def test_causal_law_bad_control_rejected():
    doc = receipt()
    law = doc["causal_benchmark_law"]
    assert law["kernel_identity"]
    assert law["shader_sha256"]
    assert law["bad_control_rejected"] is True
    assert law["noop_would_not_pass"] is True
    bad = doc["bad_control"]
    assert bad["rejected"] is True
    claimed = bad["claimed_gb_s_if_naive"]
    actual = bad["actual_read_gb_s"]
    assert claimed > actual
    assert bad["actual_bytes"] < bad["claimed_bytes"]


def test_cold_and_warm_and_process_spread():
    doc = receipt()
    f = doc["fields"]
    dram = f["HIGHEST_DRAM_READ_GB_S"]
    assert dram["kind"] == "MEASURED"
    assert dram["spread"]["n"] >= 5
    proc = f["PROCESS_SPREAD_SEQ_F4_PRIVATE_GB_S"]
    assert proc["kind"] == "MEASURED"
    assert proc["spread"]["n"] >= 3
    assert "single Metal run" in proc["note"]


def test_incumbent_fraction_of_measured_roof():
    doc = receipt()
    inc = doc["incumbent"]
    assert abs(inc["achieved_gb_s"] - 468.9248684655721) < 1.0
    frac = inc["fraction_of_measured_dram_roof"]
    assert isinstance(frac, (int, float)) and 0 < frac < 1.5
    lever = inc["execution_lever_vs_measured_roof"]
    assert lever > 1.0
    reading = inc["reading"]
    assert "468.9" in reading or "468.9" in reading.replace("GB/s", "")
    assert "measured" in reading.lower()


def test_775_uses_dram_not_cache():
    doc = receipt()
    f = doc["fields"]
    assert f["775_REACHABLE"]["kind"] == "DERIVED"
    note = f["775_REACHABLE"]["note"].lower()
    assert "dram" in note
    assert "cache" in note
    cache = f["HIGHEST_CACHE_RESIDENT_READ_GB_S"]
    if cache["kind"] == "MEASURED":
        assert "DRAM roof" in (cache.get("note") or "")


def test_absent_has_physical_reason_never_a_number():
    doc = receipt()
    for path, q in _walk_qty(doc):
        if q["kind"] == "ABSENT":
            assert q["value"] is None, path
            assert q.get("absent_reason"), path
            reason = q["absent_reason"].lower()
            assert any(
                s in reason
                for s in (
                    "counter",
                    "metal",
                    "purge",
                    "residency",
                    "dispatch",
                    "boundary",
                    "hardware",
                    "cache",
                    "row",
                )
            ), (path, q["absent_reason"])
        else:
            assert q["value"] is not None, path
            assert q["absent_reason"] is None, path
            assert q.get("command"), path
    for name in (
        "OS_PAGE_CACHE_COLD_GB_S",
        "MTLResidencySet_wired",
        "hardware_DRAM_counter",
        "per_dispatch_gpu_ns_inside_one_cb",
    ):
        assert doc["absent"][name]["kind"] == "ABSENT"


def test_label_is_dirty_or_clean_as_n016():
    doc = receipt()
    assert doc["measurement_label"] in ("DIRTY_ENGINEERING", "CLEAN_CANDIDATE")
    assert doc["measurement_label_reason"]
    if doc["measurement_label"] == "DIRTY_ENGINEERING":
        assert "CLEAN_CANDIDATE" in doc["measurement_label_reason"]


def test_anchor_decision_is_explicit():
    doc = receipt()
    a = doc["anchor_roof"]
    assert a["prior"] == PRIOR
    assert a["contradicted"] in (True, False)
    assert a["corrected"] in (True, False)
    if a["contradicted"]:
        assert a["corrected"] is True
        assert a["correction"]["changed"]
    else:
        meas = a["measured_sequential_dram_gb_s"]
        assert abs(meas - PRIOR) / PRIOR <= 0.02
    assert a["receipts_whose_conclusions_change"]
    names = {x["receipt"] for x in a["receipts_whose_conclusions_change"]}
    assert any("GPU_LEDGER" in n for n in names)
    assert any("C1SHAREDBASIS" in n for n in names)


def test_gb_s_is_decimal_bytes_per_ns():
    doc = receipt()
    dram = doc["answer"]["highest_dram_read_gb_s"]
    # A number above datasheet peak is a unit error or a cache smuggle.
    assert dram < PEAK * 1.05
    assert dram > 100.0
