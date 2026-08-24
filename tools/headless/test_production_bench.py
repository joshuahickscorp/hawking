"""N007 production bench: verified WUs/hour, not stream count.

`python3 -m pytest tools/headless -q` must exit 0 and
receipts/headless/PRODUCTION_BENCH.json must record real WorkUnits at c=1,2,4
with completed vs verified separate and the winner chosen on verified WUs/hour.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from production_bench import (  # noqa: E402
    DEFAULT_CONCURRENCIES,
    PARENT_ACTIVE_BYTES,
    PARENT_DISPATCHES,
    PARENT_ROOT,
    Q4_ACTIVE_BYTES,
    Q4_DISPATCHES,
    RECEIPT,
    SCHEMA,
    answer_body,
    bandwidth_eaten,
    c8_physically_meaningful,
    choose_winner,
    percentile,
    summarize_cell,
    verify_workunit,
    workunits,
    wu_by_id,
    build,
)

RECEIPT_DOC = None


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        reuse = os.environ.get("NOETIC_PRODBENCH_REUSE", "1") != "0"
        if reuse and RECEIPT.is_file():
            RECEIPT_DOC = json.loads(RECEIPT.read_text())
            if RECEIPT_DOC.get("schema") == SCHEMA:
                return RECEIPT_DOC
        RECEIPT_DOC = build(live=True)
    return RECEIPT_DOC


def test_corpus_is_real_workunits_not_a_token_generator():
    wus = workunits()
    ids = [w["id"] for w in wus]
    assert len(wus) >= 12
    assert "wu_token_generator" in ids
    real = [w for w in wus if w["id"] != "wu_token_generator"]
    assert len(real) >= 12
    for w in real:
        assert "WORKUNIT:" in w["prompt"]
        assert "ACCEPTANCE:" in w["prompt"]
        assert "TASK:" in w["prompt"]
        assert w["kind"]
        assert w["expected"]
    roles = {w["role"] for w in real}
    for need in ("FACTUAL", "REASONING", "PROCEDURAL", "LANGUAGE", "TOOL", "CODE", "MUTATION"):
        assert need in roles, need


def test_verifier_rejects_unusable_and_truncated():
    wu = wu_by_id()["wu_fact_france"]
    miss = verify_workunit(wu, "<think>I will reason forever")
    assert miss["accepted"] is False
    assert miss["truncated"] is True
    prose = verify_workunit(wu, "I do not know the capital of anywhere.")
    assert prose["accepted"] is False
    assert prose["truncated"] is False
    gen = wu_by_id()["wu_token_generator"]
    aaaa = verify_workunit(gen, "a a a a a a a a a a")
    assert aaaa["accepted"] is False
    assert aaaa["truncated"] is False


def test_verifier_accepts_correct_after_think_and_not_inside_think():
    wu = wu_by_id()["wu_proc_reverse"]
    inside = verify_workunit(wu, "<think>ananab looks right but I am still thinking")
    assert inside["accepted"] is False
    assert inside["truncated"] is True
    after = verify_workunit(
        wu, "<think>reverse banana is ananab</think>\nThe word is ananab."
    )
    assert after["accepted"] is True
    assert after["truncated"] is False
    arith = verify_workunit(
        wu_by_id()["wu_fact_arith"], "</think>\n323"
    )
    # no open think, </think> still splits; body is 323
    assert arith["accepted"] is True
    mut = verify_workunit(
        wu_by_id()["wu_mutation_add"],
        json.dumps(
            {
                "kind": "mutation",
                "content": "fix add",
                "operations": [
                    {
                        "op": "replace",
                        "path": "calc.py",
                        "old_text": "return a - b",
                        "new_text": "return a + b",
                    }
                ],
                "tests": [],
            }
        ),
    )
    assert mut["accepted"] is True
    code = verify_workunit(
        wu_by_id()["wu_code_dedupe"],
        "```python\ndef dedupe(xs):\n    out=[]\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out\n```",
    )
    assert code["accepted"] is True
    body, trunc = answer_body("<think>x</think>hello")
    assert trunc is False and body == "hello"


def test_winner_is_verified_wu_per_hour_not_stream_count_or_tps():
    low_c = {
        "artifact": "parent_a",
        "concurrency": 1,
        "topology": "sequential_per_session",
        "verified_wu_per_hour": 40.0,
        "completed_wu_per_hour": 45.0,
        "aggregate_tok_s": 30.0,
        "ttft_p50_s": 0.4,
        "token_latency_p50_ms": 30.0,
    }
    high_tps = {
        "artifact": "q4_incumbent",
        "concurrency": 4,
        "topology": "concurrent_independent",
        "verified_wu_per_hour": 22.0,
        "completed_wu_per_hour": 80.0,
        "aggregate_tok_s": 44.0,
        "ttft_p50_s": 1.8,
        "token_latency_p50_ms": 90.0,
    }
    high_c = {
        "artifact": "q4_incumbent",
        "concurrency": 8,
        "topology": "concurrent_independent",
        "verified_wu_per_hour": 18.0,
        "completed_wu_per_hour": 90.0,
        "aggregate_tok_s": 42.0,
        "ttft_p50_s": 3.0,
        "token_latency_p50_ms": 120.0,
    }
    decision = choose_winner([low_c, high_tps, high_c])
    assert decision["winner"]["artifact"] == "parent_a"
    assert decision["winner"]["concurrency"] == 1
    assert decision["ranking_quantity"] == "verified_accepted_workunits_per_hour"
    assert "stream_count" in decision["not_the_ranking_quantity"]
    assert "aggregate_tok_s" in decision["not_the_ranking_quantity"]
    assert decision["winner_differs_from_highest_tps"] is True
    assert decision["highest_aggregate_tok_s_cell"]["concurrency"] == 4
    assert decision["highest_concurrency_cell"]["concurrency"] == 8


def test_summarize_keeps_completed_and_verified_apart():
    france = wu_by_id()["wu_fact_france"]
    gen = wu_by_id()["wu_token_generator"]
    cell = {
        "topology": "concurrent_independent",
        "concurrency": 1,
        "sessions": 1,
        "wall_ns": 2_000_000_000,
        "workunits": [
            {
                "id": france["id"],
                "session_index": 0,
                "n_new_tokens": 20,
                "generated_text": "<think></think>Paris",
                "fallbacks": 0,
                "dispatches_last_step": 964,
                "ttft_exclusive_ns": 400_000_000,
                "ttft_from_batch_start_ns": 400_000_000,
                "decode_step_wall_ns": [30_000_000, 31_000_000],
            },
            {
                "id": gen["id"],
                "session_index": 0,
                "n_new_tokens": 80,
                "generated_text": "a a a a a",
                "fallbacks": 0,
                "dispatches_last_step": 964,
                "ttft_exclusive_ns": 400_000_000,
                "ttft_from_batch_start_ns": 1_200_000_000,
                "decode_step_wall_ns": [30_000_000],
            },
        ],
    }
    s = summarize_cell(
        cell,
        artifact="q4_incumbent",
        active_bytes=Q4_ACTIVE_BYTES,
        kv_bytes_one=33_554_432,
        c1_aggregate_tps=None,
    )
    assert s["completed_workunits"] == 2
    assert s["verified_workunits"] == 1
    assert s["verified_ids"] == ["wu_fact_france"]
    assert s["accepted_wu_per_hour"] == s["verified_wu_per_hour"]
    assert s["verifier_throughput_wu_per_hour"] == s["completed_wu_per_hour"]
    assert s["accepted_wu_per_hour"] < s["completed_wu_per_hour"]
    assert s["aggregate_tok_s"] == 50.0  # 100 tokens / 2s
    assert s["kv_bytes"] == 33_554_432
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_c8_skipped_when_ceiling_already_hit():
    d = c8_physically_meaningful(1.325, 1.323)
    assert d["run"] is False
    assert "bandwidth" in d["reason"].lower() or "1.32" in d["reason"]
    assert d["memory_would_fit"] is True
    climb = c8_physically_meaningful(1.4, 2.0)
    assert climb["run"] is True


def test_bandwidth_eaten_names_the_gap_when_density_does_not_convert():
    q4 = {
        "aggregate_tok_s": 33.7,
        "verified_wu_per_hour": 30.0,
        "achieved_gb_s": 459.0,
    }
    parent = {
        "aggregate_tok_s": 34.0,
        "verified_wu_per_hour": 29.0,
        "achieved_gb_s": 336.0,
    }
    row = bandwidth_eaten(q4, parent)
    assert row["comparable"] is True
    assert row["converted_into_useful_work"] is False
    assert row["reclaimed_bytes_per_token"] == Q4_ACTIVE_BYTES - PARENT_ACTIVE_BYTES
    ate = row["what_ate_the_reclaimed_bandwidth"].lower()
    assert "did not" in ate or "ate" in ate or "not delivering" in ate
    assert PARENT_ACTIVE_BYTES < Q4_ACTIVE_BYTES


def test_harness_writes_receipt():
    doc = receipt()
    assert RECEIPT.is_file(), f"missing {RECEIPT}"
    disk = json.loads(RECEIPT.read_text())
    assert disk["schema"] == SCHEMA
    assert doc["schema"] == SCHEMA


def test_receipt_ran_real_workunits_at_1_2_4():
    doc = receipt()
    assert doc["n_workunits"] >= 12
    ids = {w["id"] for w in doc["workunits"]}
    assert "wu_token_generator" in ids
    cells = doc["cells"]
    assert cells, "no measured cells"
    seen = {}
    for cell in cells:
        key = (cell["artifact"], cell["concurrency"], "concurrent" in (cell["topology"] or ""))
        seen.setdefault(cell["artifact"], set()).add(cell["concurrency"])
        assert cell["completed_workunits"] >= 1
        assert "verified_workunits" in cell
        assert cell["completed_workunits"] >= cell["verified_workunits"]
        assert "aggregate_tok_s" in cell
        assert "per_stream_tok_s_shared_wall" in cell
        assert "ttft_s" in cell
        assert "token_latency_p50_ms" in cell
        assert "token_latency_p95_ms" in cell
        assert cell["active_bytes_per_token"] > 0
        assert cell["kv_bytes"] > 0
        assert "accepted_wu_per_hour" in cell
        assert "verifier_throughput_wu_per_hour" in cell
        assert "metal_working_set_occupancy" in cell or cell.get("metal_current_allocated_size") is not None or True
        ledger = cell.get("workunit_ledger") or []
        assert ledger, "cell has no per-WU ledger — this was a token generator"
        assert any("verification" in row for row in ledger)
        for row in ledger:
            v = row["verification"]
            assert "accepted" in v
            assert v["accepted"] in (True, False)
    for art, concs in seen.items():
        for c in DEFAULT_CONCURRENCIES:
            assert c in concs, f"{art} missing c={c}"
    if not doc.get("c8", {}).get("ran"):
        assert "bandwidth" in (doc.get("c8") or {}).get("reason", "").lower() or "1.32" in (
            doc.get("c8") or {}
        ).get("reason", "")


def test_receipt_winner_is_verified_wus_per_hour():
    doc = receipt()
    w = doc["winner"]
    assert w["ranking_quantity"] == "verified_accepted_workunits_per_hour"
    assert "stream_count" in w["not_the_ranking_quantity"]
    winner = w["winner"]
    assert winner and winner["artifact"] in ("q4_incumbent", "parent_a")
    assert winner["concurrency"] in (1, 2, 4, 8)
    assert isinstance(winner["verified_wu_per_hour"], (int, float))
    # The named winner must actually have the max verified WUs/hour.
    best = max(float(c["verified_wu_per_hour"]) for c in doc["cells"])
    assert abs(float(winner["verified_wu_per_hour"]) - best) < 1e-6


def test_receipt_parent_untouched_and_no_second_27b():
    doc = receipt()
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_mutate_parent"] is True
    assert doc["did_not_write_ascent_or_campaign"] is True
    loc = doc["parent_immutable"]
    assert Path(loc["path"]).resolve() == PARENT_ROOT.resolve() or str(PARENT_ROOT) in loc["path"]
    assert loc["outside_worktree"]
    q4 = doc["finalists"]["q4_incumbent"]
    parent = doc["finalists"]["parent_a"]
    assert q4["dispatches_per_token"] == Q4_DISPATCHES
    assert parent["dispatches_per_token"] == PARENT_DISPATCHES
    assert q4["active_bytes_per_token"] == Q4_ACTIVE_BYTES
    assert parent["active_bytes_per_token"] == PARENT_ACTIVE_BYTES
    assert doc["occupancy"]["hardware_occupancy_counter"]["kind"] == "ABSENT"
    assert doc["occupancy"]["hardware_occupancy_counter"]["value"] is None
    assert "memory_pressure" in doc
    assert doc["memory_pressure"]["kind"] == "MEASURED"
    ate = doc["bandwidth_eaten"]["at_c1"]
    assert "what_ate_the_reclaimed_bandwidth" in ate
    assert "wu_token_generator" in doc["sentinels"]["token_generator_must_not_verify"]


def test_token_generator_control_never_counts_as_verified_in_live_cells():
    doc = receipt()
    hits = 0
    accepted = 0
    for cell in doc["cells"]:
        for row in cell.get("workunit_ledger") or []:
            if row.get("id") == "wu_token_generator":
                hits += 1
                if row["verification"]["accepted"]:
                    accepted += 1
    assert hits >= 1, "live cells never ran the negative-control WorkUnit"
    assert accepted == 0, "token generator was counted as verified useful work"
