"""GPU ledger per token: ACTIVE/DRAM bytes first-class, ABSENT discipline, cold/warm.

pytest tools/headless -q must see receipts/headless/GPU_LEDGER.json for the
q4 incumbent. The harness does not load a second 27B.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from gpu_ledger import (  # noqa: E402
    ABSENT,
    DERIVED,
    MEASURED,
    RECEIPT,
    REQUIRED_STAGES,
    SCHEMA,
    build,
    write_receipt,
)

RECEIPT_DOC = None
RECEIPT_PATH = REPO / "receipts" / "headless" / "GPU_LEDGER.json"
KINDS = {MEASURED, DERIVED, ABSENT}


def receipt() -> dict:
    """READ the sealed measurement. Build only if there is none to read.

    This previously called build() + write_receipt() unconditionally, so every
    `pytest tools/headless` run RE-MEASURED the GPU and OVERWROTE the receipt.
    Three things were wrong with that:

      * it destroyed a sealed measurement as a side effect of running tests;
      * the test then asserted against numbers it had just produced, so it could
        never fail on a regression in the RECORDED measurement, only on current
        machine state; and
      * it failed spuriously under contention -- with other GPU lanes resident,
        `did_not_load_second_27b` flipped to False and took the suite red on a
        fact about the machine rather than about the code.

    Measuring is the harness's job. Checking what was measured is this test's job.
    """
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        if RECEIPT_PATH.is_file():
            RECEIPT_DOC = json.loads(RECEIPT_PATH.read_text())
        else:
            RECEIPT_DOC = build()
            write_receipt(RECEIPT_DOC)
    return RECEIPT_DOC


def _walk_qty(obj, path=""):
    """Yield (path, qty-dict) for every labelled quantity."""
    if isinstance(obj, dict):
        if (
            "kind" in obj
            and obj["kind"] in KINDS
            and "command" in obj
            and "unit" in obj
        ):
            yield path, obj
        for k, v in obj.items():
            yield from _walk_qty(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_qty(v, f"{path}[{i}]")


def test_harness_writes_receipt():
    doc = receipt()
    assert RECEIPT.is_file(), f"missing {RECEIPT} — run python3 tools/headless/gpu_ledger.py"
    disk = json.loads(RECEIPT.read_text())
    assert disk["schema"] == SCHEMA
    assert doc["schema"] == SCHEMA


def test_q4_incumbent_and_no_second_27b():
    doc = receipt()
    assert "qwen38-gravity-uniform-q4-v1" in doc["incumbent"]["artifact"]
    assert doc["did_not_load_second_27b"] is True
    assert doc["occupancy"]["loaded_a_second_27b"] is False


def test_active_and_dram_are_first_class_and_ranked_above_stored():
    doc = receipt()
    for key in ("ACTIVE_BYTES_PER_TOKEN", "DRAM_BYTES_PER_TOKEN"):
        assert key in doc, f"{key} must be a first-class field"
        q = doc[key]
        assert q["kind"] in (MEASURED, DERIVED), q
        assert isinstance(q["value"], (int, float)) and q["value"] > 0
        assert q["command"]
        assert q["absent_reason"] is None
    assert doc["rank_above_stored_size"] == [
        "ACTIVE_BYTES_PER_TOKEN",
        "DRAM_BYTES_PER_TOKEN",
    ]
    stored = doc["fields"]["STORED_BYTES"]["value"]
    active = doc["ACTIVE_BYTES_PER_TOKEN"]["value"]
    dram = doc["DRAM_BYTES_PER_TOKEN"]["value"]
    # Active is the streamed working set, not the embed table.
    assert active < stored
    # DRAM includes activations + state on top of the weight stream.
    assert dram >= active
    why = doc["doctor_objective"]["why"]
    assert "1 EBPW" in why and "5 EBPW" in why
    assert "unified-memory" in why


def test_doctor_objective_is_useful_function_over_active_bytes_times_token_ns():
    doc = receipt()
    d = doc["doctor_objective"]
    assert "useful_function / (ACTIVE_BYTES_PER_TOKEN × TOKEN_NS)" in d["formula"]
    assert d["rank_active_and_dram_above_stored_size"] is True
    assert d["ACTIVE_BYTES_PER_TOKEN"] == doc["ACTIVE_BYTES_PER_TOKEN"]["value"]
    assert d["TOKEN_NS"] == doc["fields"]["COMPLETE_TOKEN_WALL_NS"]["value"]
    assert d["denominator_byte_ns"] == d["ACTIVE_BYTES_PER_TOKEN"] * d["TOKEN_NS"]
    assert d["ranking_quantity"] == 1.0 / d["denominator_byte_ns"]


def test_cold_and_warm_separately_with_spread():
    doc = receipt()
    cold = doc["cold"]["gpu_ns"]
    warm = doc["warm"]["gpu_ns"]
    assert cold["kind"] == MEASURED
    assert warm["kind"] == MEASURED
    assert cold["spread"]["n"] == 3
    assert warm["spread"]["n"] == 3
    assert cold["spread"]["spread_pct"] is not None
    assert warm["spread"]["spread_pct"] is not None
    # Graph-cold first step is slower than warm steady decode.
    assert cold["value"] > warm["value"]
    # A single Metal run is not accepted as the headline.
    assert doc["warm"]["n_process_runs"] >= 3
    assert doc["fields"]["OS_PAGE_CACHE_COLD_GPU_NS"]["kind"] == ABSENT
    assert doc["fields"]["OS_PAGE_CACHE_COLD_GPU_NS"]["value"] is None


def test_dispatches_964_and_one_command_buffer():
    doc = receipt()
    assert doc["fields"]["dispatch_count"]["value"] == 964
    assert doc["fields"]["dispatch_count"]["kind"] == MEASURED
    assert doc["fields"]["command_buffer_count"]["value"] == 1
    assert doc["fields"]["command_buffer_count"]["kind"] == MEASURED
    assert doc["anchors_reproduced"] == {
        "dispatches_per_token": 964,
        "command_buffers_per_token": 1,
        "reproduced": True,
    }
    assert doc["production_shape"]["dispatches_per_token"] == 964
    assert doc["production_shape"]["command_buffers_per_token"] == 1


def test_required_gpu_fields_present_and_labelled():
    doc = receipt()
    f = doc["fields"]
    for name in (
        "GPU_NS",
        "GPU_BUSY_NS",
        "GPU_IDLE_GAPS_NS",
        "DRAM_READ_BYTES",
        "DRAM_WRITE_BYTES",
        "dispatch_count",
        "command_buffer_count",
        "encoder_count",
        "synchronization_count",
        "host_wait_ns",
        "active_threadgroups",
        "occupancy_estimate",
        "SIMD_utilization",
        "register_pressure",
        "threadgroup_memory_bytes",
        "ACTIVE_BYTES_PER_TOKEN",
        "DRAM_BYTES_PER_TOKEN",
    ):
        assert name in f, name
        assert f[name]["kind"] in KINDS, (name, f[name])
        assert f[name]["command"], name


def test_unmeasurable_is_absent_with_physical_reason_never_zero():
    doc = receipt()
    absent = []
    for path, q in _walk_qty(doc):
        if q["kind"] == ABSENT:
            absent.append(path)
            assert q["value"] is None, f"{path} ABSENT must not carry a value, got {q['value']!r}"
            assert q["value"] != 0
            assert q.get("absent_reason"), f"{path} ABSENT needs a physical reason"
            # The reason has to be physical, not "not implemented".
            reason = q["absent_reason"].lower()
            assert any(
                s in reason
                for s in (
                    "counter",
                    "metal",
                    "dispatch",
                    "purge",
                    "router",
                    "dense",
                    "architecture",
                    "register",
                    "occupancy",
                    "nsight",
                    "boundary",
                )
            ), (path, q["absent_reason"])
        else:
            assert q["value"] is not None, path
            assert q["absent_reason"] is None, path
            assert q.get("command"), path
    # The contractually unmeasurable ones must actually be ABSENT.
    f = doc["fields"]
    for name in (
        "GPU_IDLE_GAPS_NS",
        "DRAM_READ_BYTES",
        "DRAM_WRITE_BYTES",
        "SIMD_utilization",
        "register_pressure",
        "OS_PAGE_CACHE_COLD_GPU_NS",
        "hardware_occupancy_counter",
        "per_dispatch_gpu_ns",
    ):
        assert f[name]["kind"] == ABSENT, name
    assert doc["stages"]["routing"]["kind"] == ABSENT
    assert doc["stages"]["routing"]["live_ns"] is None
    assert "dense" in doc["stages"]["routing"]["absent_reason"].lower()
    assert len(absent) >= 8


def test_per_stage_split_covers_requested_names():
    doc = receipt()
    stages = doc["stages"]
    assert tuple(k for k in REQUIRED_STAGES) == REQUIRED_STAGES
    for name in REQUIRED_STAGES:
        assert name in stages, name
    # Present GPU stages partition production GPU (routing excluded).
    c = doc["stages_closure"]
    assert c["routing_in_sum"] is False
    present = sum(
        stages[n]["live_ns"]
        for n in ("representation_decode", "operator", "activation", "kv_state", "sampling")
    )
    # embed is in the closure but not a requested stage
    present_plus_embed = present + c["embed_ns_prior"] * c["scale"]
    assert abs(present_plus_embed - c["live_gpu_ns"]) < 2.0
    # Operator (weight addressing + mixer compute) is the majority.
    assert stages["operator"]["live_ns"] > stages["representation_decode"]["live_ns"]
    assert stages["operator"]["kind"] == DERIVED
    assert stages["representation_decode"]["kind"] == DERIVED


def test_q80_idle_anchor_is_contradicted_not_copied():
    doc = receipt()
    q = doc["q80_anchor"]
    assert q["anchor"]["pct_of_700_GBs"] == 0.79
    assert q["anchor"]["gpu_idle_pct"] == 51.0
    assert q["verdict"] == "CONTRADICTED_FOR_THIS_INCUMBENT"
    inc = q["q4_incumbent"]
    assert inc["pct_of_700_GBs"] > 10.0  # not 0.79
    assert inc["gpu_as_fraction_of_wall"] > 0.90
    assert inc["queue_wait_as_pct_of_wall"] < 10.0
    assert "false" in q["reading"].lower() or "CONTRADICT" in q["verdict"]


def test_every_labelled_number_has_a_command():
    doc = receipt()
    n = 0
    for path, q in _walk_qty(doc):
        n += 1
        assert q["kind"] in KINDS, path
        assert isinstance(q["command"], str) and q["command"], path
    assert n >= 20


def test_gpu_timestamps_are_not_cpu_wait_proxies():
    doc = receipt()
    assert "GPUStartTime" in doc["gpu_timestamp_authority"]
    assert "never a CPU-wait proxy" in doc["gpu_timestamp_authority"]
    assert doc["fields"]["GPU_NS"]["kind"] == MEASURED
    probe = doc["metal_probe"]
    sets = [cs["name"] for cs in probe["counterSets"]]
    assert sets == ["timestamp"]
    assert probe["supportsCounterSampling"]["atDispatchBoundary"] is False
