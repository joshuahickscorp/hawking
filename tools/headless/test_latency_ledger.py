"""N028 LATENCY_LEDGER: INTERACTIVE user-visible latency, measured.

`python3 -m pytest tools/headless -q` must see
receipts/headless/LATENCY_LEDGER.json covering TTFT, TPOT, prefill,
inter-token, p50/p95/p99 token latency, admission/cold/warm start,
scheduling, context-compile, sampler — each MEASURED or ABSENT — as a
per-stage breakdown with a named largest contributor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from latency_ledger import (  # noqa: E402
    ABSENT,
    CPL_LEDGER,
    CPL_REUSE,
    MEASURED,
    PARENT,
    RECEIPT,
    REQUIRED_STAGES,
    SCHEMA,
    argmax_f32,
    build,
    percentile,
    stats_ms,
    validate_receipt,
    write_receipt,
)

RECEIPT_DOC = None
KINDS = {MEASURED, ABSENT, "DERIVED"}


def receipt() -> dict:
    """Read the sealed ledger. Build only if there is none to read.

    Measuring is the harness's job. Checking what was measured is this test's.
    """
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        if RECEIPT.is_file():
            RECEIPT_DOC = json.loads(RECEIPT.read_text(encoding="utf-8"))
            if RECEIPT_DOC.get("schema") == SCHEMA and not validate_receipt(RECEIPT_DOC):
                return RECEIPT_DOC
        RECEIPT_DOC = build()
        write_receipt(RECEIPT_DOC)
    return RECEIPT_DOC


def test_argmax_matches_hawking_core_kernels():
    xs = [1.0, 5.0, 2.0, 0.0]
    assert argmax_f32(xs) == 1
    assert argmax_f32([-3.0, -1.0, -2.0]) == 1


def test_percentile_and_stats_have_spread():
    xs = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(xs, 50) == 30.0
    assert percentile(xs, 0) == 10.0
    assert percentile(xs, 100) == 50.0
    fig = stats_ms(xs)
    assert fig["n"] == 5
    assert fig["median"] == 30.0
    assert fig["p50"] == 30.0
    assert fig["samples"] == xs
    assert abs(fig["spread_pct"] - 100.0 * 40.0 / 30.0) < 1e-6


def test_harness_writes_receipt_schema_and_filename():
    doc = receipt()
    assert RECEIPT.is_file(), f"missing {RECEIPT} — run python3 tools/headless/latency_ledger.py"
    assert RECEIPT.name == "LATENCY_LEDGER.json"
    assert RECEIPT.resolve().parts[-2] == "headless"
    assert "ascent-2026-08-16" not in str(RECEIPT)
    assert "campaign" not in str(RECEIPT)
    disk = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert disk["schema"] == SCHEMA
    assert doc["schema"] == SCHEMA
    assert doc["gate"] == "LATENCY_LEDGER"
    fails = validate_receipt(doc)
    assert not fails, fails


def test_interactive_profile_and_baseline_has_no_targets():
    doc = receipt()
    assert doc["profile"]["name"] == "INTERACTIVE"
    assert doc["profile"]["s022_section"] == 5
    assert doc["profile"]["concurrency"] == 1
    base = doc["baseline"]
    assert base["s022_section"] == 43
    assert base["targets"] is None
    assert base["targets_invented"] is False
    assert "target_ms" not in doc
    assert "slo" not in {k.lower() for k in doc}


def test_required_stages_measured_or_absent_with_breakdown():
    doc = receipt()
    stages = doc["stages"]
    for name in REQUIRED_STAGES:
        assert name in stages, name
        st = stages[name]
        assert st["status"] in {MEASURED, ABSENT}, (name, st["status"])
        assert "absolute_ms" in st
        assert "pct_of_total" in st
        assert st.get("owner")
        assert "blocking_dep" in st
        assert "avoidable" in st
        assert "cacheable" in st
        assert "fusible" in st
        assert st.get("command"), name
        if st["status"] == MEASURED:
            assert isinstance(st["absolute_ms"], (int, float)), name
            spread = st["spread"]
            assert spread["n"] >= 1, name
            assert spread["samples"], name
            assert spread["median"] is not None, name
            assert spread.get("spread_pct") is not None or spread["n"] == 1, name
        else:
            assert st.get("reason"), name
            assert st["absolute_ms"] is None
    ids = [row["id"] for row in doc["breakdown"]]
    assert ids == list(REQUIRED_STAGES)


def test_user_visible_vector_covers_acceptance_names():
    doc = receipt()
    vec = doc["user_visible_vector"]
    for key in (
        "ttft_ms",
        "tpot_ms",
        "prefill_ms",
        "inter_token_ms",
        "token_latency_p50_ms",
        "token_latency_p95_ms",
        "token_latency_p99_ms",
        "admission_ms",
        "cold_start_ms",
        "warm_start_ms",
        "scheduling_ms",
        "context_compile_ms",
        "sampler_ms",
    ):
        assert key in vec, key
        assert vec[key] is None or isinstance(vec[key], (int, float)), (key, vec[key])
    # INTERACTIVE decode-path numbers must actually be present.
    for key in (
        "ttft_ms",
        "tpot_ms",
        "prefill_ms",
        "inter_token_ms",
        "token_latency_p50_ms",
        "token_latency_p95_ms",
        "token_latency_p99_ms",
        "sampler_ms",
        "admission_ms",
        "scheduling_ms",
        "context_compile_ms",
    ):
        assert isinstance(vec[key], (int, float)) and vec[key] > 0, key
    p50 = vec["token_latency_p50_ms"]
    p95 = vec["token_latency_p95_ms"]
    p99 = vec["token_latency_p99_ms"]
    assert p50 <= p95 <= p99
    # p50 of complete-token wall ≈ TPOT (same distribution).
    assert abs(p50 - vec["tpot_ms"]) / vec["tpot_ms"] < 0.05
    assert abs(vec["inter_token_ms"] - vec["tpot_ms"]) < 1e-9


def test_control_plane_reused_from_n022_not_rederived():
    doc = receipt()
    assert CPL_LEDGER.is_file()
    cpl = json.loads(CPL_LEDGER.read_text(encoding="utf-8"))
    by = {r["id"]: r for r in cpl["measurements"]}
    for stage_id, mid in CPL_REUSE.items():
        st = doc["stages"][stage_id]
        assert st["status"] == MEASURED, stage_id
        assert st["source"].endswith("CONTROL_PLANE_LATENCY_LEDGER.json")
        assert st["extra"]["n022_id"] == mid
        assert st["absolute_ms"] == by[mid]["warm_ms"]["median"]
        assert st["extra"]["cold_ms"]["median"] == by[mid]["cold_ms"]["median"]
        assert st["extra"]["warm_ms"]["samples"] == by[mid]["warm_ms"]["samples"]


def test_largest_contributor_named_and_is_prefill():
    doc = receipt()
    top = doc["largest_contributor"]
    assert top["id"] == "prefill"
    assert isinstance(top["ms"], (int, float)) and top["ms"] > 0
    assert top["owner"] == "GPU"
    assert top["pct_of_interactive_first_token"] > 50
    prefill = doc["stages"]["prefill"]
    assert prefill["status"] == MEASURED
    assert prefill["owner"] == "GPU"
    assert prefill["cacheable"] is True
    assert prefill["fusible"] is True
    # Prefill dominates the short-prompt first-token composition.
    assert prefill["pct_of_total"] > 90


def test_noop_adversary_rejects_trivial_path():
    doc = receipt()
    adv = doc["noop_adversary"]
    assert "python3 -c pass" in adv["noop"]
    floor = adv["floor_ms"]["median"]
    assert isinstance(floor, (int, float)) and floor > 0
    by = {c["id"]: c for c in adv["comparisons"]}
    # Prefill ~1s is not an interpreter spawn.
    assert by["prefill"]["would_noop_post_same_number"] is False
    assert by["ttft"]["would_noop_post_same_number"] is False
    # GPU token wall is a Metal CB. Even when its milliseconds sit near the
    # interpreter spawn floor, the command is hybrid_greedy, not `pass`.
    assert by["tpot"]["would_noop_post_same_number"] is False
    assert "python3 -c pass" not in " ".join(str(x) for x in by["tpot"]["command"])
    assert "complete-wall" in " ".join(str(x) for x in by["tpot"]["command"])
    # In-process sampler is not a process spawn.
    assert by["sampler"]["would_noop_post_same_number"] is False
    cpu = doc["cpu_sampler_this_run"]["greedy_ms"]["median"]
    assert cpu < floor, (cpu, floor)


def test_verified_work_per_wall_is_the_correlate():
    doc = receipt()
    v = doc["verified_work_per_wall"]
    assert v["s022_section"] == 42
    assert "verified" in v["ranking_quantity"]
    c1 = v["interactive_c1"]
    assert isinstance(c1["verified_work_per_wall_s"], (int, float))
    assert c1["verified_work_per_wall_s"] > 0
    assert c1["concurrency"] == 1
    assert "tok/s" in v["not_the_ranking_quantity"] or "TTFT" in v["not_the_ranking_quantity"]


def test_did_not_load_second_27b_or_mutate_parent():
    doc = receipt()
    assert doc["did_not_load_second_27b"] is True
    assert doc["occupancy"]["loaded_a_second_27b"] is False
    assert doc["did_not_mutate_sealed_parent"] is True
    assert doc["did_not_write_ascent_or_campaign"] is True
    assert doc["did_not_write_under_models"] is True
    before = doc["parent_identity_before"]
    after = doc["parent_identity_after"]
    if before.get("catalog_present"):
        assert before["catalog_ino"] == after["catalog_ino"]
        assert before["catalog_mtime_ns"] == after["catalog_mtime_ns"]
        assert Path(before["path"]).resolve() == PARENT.resolve()


def test_sampler_is_the_decode_path_argmax():
    doc = receipt()
    st = doc["stages"]["sampler"]
    assert st["status"] == MEASURED
    extra = st["extra"]
    cpu = extra.get("cpu_reference") or doc["cpu_sampler_this_run"]
    assert cpu["vocab"] == 248320
    assert cpu["picked"] == 248320 // 2
    assert cpu["greedy_ms"]["n"] >= 5
    assert "sample_argmax_f32" in (extra.get("kernel") or st.get("note") or "")
    assert st["owner"] in {"GPU", "CPU"}
    assert st["fusible"] is True
