"""ONE resident Noetic body, many isolated sessions.

`python3 -m pytest tools/headless -q` must exit 0 and
receipts/headless/NOETIC_MULTISESSION.json must record
NOETIC_MULTISESSION_SHARED_BODY=PASS with resident bytes proving one body
(not N copies), isolated per-session state, and aggregate throughput across
at least two scheduling topologies.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from noetic_multisession import (  # noqa: E402
    RECEIPT,
    SCHEMA,
    expected_n_copies_bytes,
    expected_shared_resident_bytes,
    one_body_not_n_copies,
    simulate_topologies,
    workspace_bytes,
    build,
)

RECEIPT_DOC = None


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        reuse = os.environ.get("NOETIC_MULTISESSION_REUSE", "1") != "0"
        if reuse and RECEIPT.is_file():
            RECEIPT_DOC = json.loads(RECEIPT.read_text())
            if RECEIPT_DOC.get("schema") == SCHEMA:
                return RECEIPT_DOC
        RECEIPT_DOC = build(live=True)
    return RECEIPT_DOC


def test_workspace_kv_is_the_seq_len_term():
    a = workspace_bytes(256)
    b = workspace_bytes(512)
    assert a["activation_bytes"] == b["activation_bytes"]
    assert a["deltanet_state_bytes"] == b["deltanet_state_bytes"]
    assert b["gqa_kv_bytes"] == a["gqa_kv_bytes"] * 2
    assert b["total_bytes"] - a["total_bytes"] == b["gqa_kv_bytes"] - a["gqa_kv_bytes"]
    assert a["total_bytes"] < 2 * 1024 * 1024 * 1024
    assert a["gqa_kv_bytes"] > 0


def test_one_body_formula_rejects_four_copies():
    body = 10_000_000_000
    ws = 250_000_000
    one = expected_shared_resident_bytes(body, ws, 4)
    copies = expected_n_copies_bytes(body, ws, 4)
    assert one < 12_000_000_000
    assert copies > 40_000_000_000
    assert one_body_not_n_copies(one, body, ws, 4)
    assert one_body_not_n_copies(one + 300_000_000, body, ws, 4)
    assert not one_body_not_n_copies(copies, body, ws, 4)
    assert not one_body_not_n_copies(4 * body, body, ws, 4)


def test_round_robin_clusters_ttft_sequential_piles_it():
    sim = simulate_topologies(4, prefill=8, decode=8, step_s=0.03)
    seq = sim["sequential_per_session"]["ttft_s"]
    rr = sim["round_robin_token"]["ttft_s"]
    assert seq[-1] > seq[0] * 3, "sequential later sessions wait for earlier ones"
    assert max(rr) - min(rr) < seq[-1] - seq[0]
    assert max(rr) < seq[-1]
    # Same total GPU work ⇒ same aggregate tok/s in the equal-step CPU model.
    assert abs(
        sim["sequential_per_session"]["aggregate_tps"]
        - sim["round_robin_token"]["aggregate_tps"]
    ) < 1e-9
    # Round-robin token latency includes the other sessions' steps.
    rr_lat = sim["round_robin_token"]["token_latency_s"][0][0]
    seq_lat = sim["sequential_per_session"]["token_latency_s"][0][0]
    assert rr_lat > seq_lat
    assert sim["sequential_per_session"]["policy"] == "throughput_background"
    assert sim["round_robin_token"]["policy"] == "latency_fair_foreground"


def test_harness_writes_receipt():
    doc = receipt()
    assert RECEIPT.is_file(), f"missing {RECEIPT}"
    on_disk = json.loads(RECEIPT.read_text())
    assert on_disk["schema"] == SCHEMA
    assert doc["schema"] == SCHEMA
    assert "NOETIC_MULTISESSION_SHARED_BODY" in doc


def test_shared_body_is_pass_with_resident_bytes():
    doc = receipt()
    assert doc["NOETIC_MULTISESSION_SHARED_BODY"] == "PASS", doc.get("judge")
    proof = doc["proof_one_body"]
    assert proof["one_body_not_n_copies"] is True
    c4 = proof.get("rss_c4_bytes") or proof.get("metal_c4_bytes")
    c1 = proof.get("rss_c1_bytes") or proof.get("metal_c1_bytes")
    copies = proof["predicted_four_copies_c4_bytes"]
    one = proof["predicted_one_body_c4_bytes"]
    assert c4 and c1 and copies and one
    assert c4 < copies / 2, f"c4={c4} looks like N copies (predicted {copies})"
    assert c4 / c1 < 2.0, f"c4/c1={c4/c1:.3f} — four copies would be ~4x"
    assert one < copies / 2
    live = doc["live"]
    assert live["weights_ptr_shared"] is True
    assert live["did_not_load_second_27b"] is True
    assert live["weight_loads"] == 1
    assert live["process_count"] == 1
    assert live["attached_sessions"] >= 4


def test_sessions_are_isolated():
    doc = receipt()
    iso = doc["isolation"]
    assert iso["isolated"] is True
    assert iso["kv_pointers_distinct"] is True
    assert iso["continuation_matches_control"] is True
    assert iso["other_session_did_not_mutate_session0_state"] is True
    assert iso["reset_a_does_not_reset_b"] is True
    # Short greedy samples of chat prompts often share a `<think>` prefix.
    # Isolation is pointer split + continuation, not token inequality.
    ids = iso["buffer_identities"]
    assert len(ids) >= 3
    ptrs = []
    for ident in ids:
        for key in (
            "gqa_key_ptr",
            "gqa_value_ptr",
            "conv_state_ptr",
            "rec_state_ptr",
            "sampled_ptr",
            "logits_ptr",
        ):
            ptrs.append(ident[key])
            assert ident[key] != 0
    assert len(set(ptrs)) == len(ptrs)


def test_two_topologies_measured_at_1_2_4():
    doc = receipt()
    measured = doc["measurements"]
    assert "sequential_per_session" in measured
    assert "round_robin_token" in measured
    for name in ("sequential_per_session", "round_robin_token"):
        by_c = measured[name]
        for c in ("1", "2", "4"):
            row = by_c[c]
            assert row["sessions"] == int(c)
            assert isinstance(row["aggregate_tps"], (int, float)) and row["aggregate_tps"] > 0
            assert row["per_stream_tps_exclusive"]
            assert len(row["per_stream_tps_exclusive"]) == int(c)
            assert row["ttft_ms"]
            assert len(row["ttft_ms"]) == int(c)
            assert row["token_latency_p50_ms"] is not None
            assert row["token_latency_p95_ms"] is not None
            assert row["token_latency_p95_ms"] >= row["token_latency_p50_ms"]
    kv = doc["kv_bytes"]
    peak = doc["peak_unified_memory_bytes"]
    assert isinstance(kv, int) and kv > 0
    assert isinstance(peak, int) and peak > 0
    seq_c1 = measured["sequential_per_session"]["1"]["aggregate_tps"]
    for name in ("sequential_per_session", "round_robin_token"):
        for c in ("2", "4"):
            scaling = measured[name][c]["aggregate_tps"] / seq_c1
            # A shared-body serial step cannot be 4x; if it is, the clock is wrong.
            assert scaling < 3.5, f"{name} c={c} scaling {scaling:.3f} looks invented"


def test_policies_and_microbatch_are_named():
    doc = receipt()
    pol = doc["policy"]
    assert pol["sequential_per_session"].startswith("throughput_background")
    assert pol["round_robin_token"].startswith("latency_fair_foreground")
    assert "operator_microbatch" in pol
    mb = doc.get("operator_microbatch") or []
    assert mb, "operator microbatch is supported and must be measured"
    sessions = {row["sessions"] for row in mb}
    assert {1, 2, 4} <= sessions


def test_did_not_touch_forbidden_trees():
    doc = receipt()
    assert doc["did_not_write_ascent_or_campaign"] is True
    assert doc["did_not_load_second_27b"] is True
    # The receipt itself must live under receipts/headless, not the forbidden trees.
    assert RECEIPT.resolve().parts[-2] == "headless"
    assert "ascent-2026-08-16" not in str(RECEIPT)
    assert "campaign" not in str(RECEIPT)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("done")
