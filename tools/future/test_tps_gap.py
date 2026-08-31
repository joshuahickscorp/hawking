"""TPS gap: UNATTRIBUTED stays unattributed, WOULD raises, 34.0 is not complete.

A validator nobody has watched reject is a validator that will silently
drift into fiction. Every judgement here has a negative: hardware keys
raise, a residue is not folded, a WOULD without a dirty pair is refused,
missing clocks refuse rather than skip, and recoverability is UNKNOWN
when the historical quantity cannot be recovered — never REGRESSION.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tools.future import tps_gap as G
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    HardwareClaimError,
    _assert_no_hardware_claims,
    write_receipt,
)


def _reply(**over: object) -> dict:
    base = G.synthetic_reply(
        prefill_ns=400_000_000,
        decode_ns=1_100_000_000,
        generation_ns=1_500_000_050,
        request_ns=1_520_000_000,
        generated_tokens=40,
        decode_steps=39,
        prompt_tokens=13,
        complete_ns=1_540_000_000,
    )
    base.update(over)
    return base


def test_build_seals_receipt_without_hardware_keys():
    out = G.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == G.RECEIPT
    assert doc["schema"] == G.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["self_timing"]["evidence_class"] == G.DIRTY
    assert "not PROTECTED_ABSOLUTE" in doc["self_timing"]["what_this_block_is_not"]
    assert doc["self_timing"]["this_process_did_not_run_the_resident"] is True
    assert doc["self_timing"]["numbers_decide_nothing"] is True
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]["fails_closed"]
    assert doc["workunit"]["species"] == "PROFILE_HOST_CEREMONY"
    assert doc["workunit"]["does_not_claim_would_improve"] is True
    assert doc["workunit"]["gpu_authority"] is False
    _assert_no_hardware_claims(doc)

    def walk(node: object, path: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                    raise AssertionError(f"{here} = {v!r} is a hardware field")
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)


def test_writing_a_hardware_named_field_raises():
    """NEGATIVE CONTROL: write_receipt must actually refuse tps / wall_ns."""
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "_TPS_GAP_HARDWARE_PROBE.json",
            {"schema": "probe", "tps": 34.0},
            "tools/future/test_tps_gap.py",
        )
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "_TPS_GAP_HARDWARE_PROBE.json",
            {"schema": "probe", "wall_ns": 123},
            "tools/future/test_tps_gap.py",
        )
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "_TPS_GAP_HARDWARE_PROBE.json",
            {"schema": "probe", "nested": {"gpu_ns": 1}},
            "tools/future/test_tps_gap.py",
        )
    with pytest.raises(HardwareClaimError):
        G.assert_timing_key_legal("tps")
    with pytest.raises(HardwareClaimError):
        G.assert_timing_key_legal("wall_ns")


def test_missing_clock_is_a_refusal_not_a_skip():
    """NEGATIVE CONTROL: absent input fails closed. A skip that fires is a P0."""
    with pytest.raises(G.GapRefuse, match="absent"):
        G.extract_clocks(None)
    with pytest.raises(G.GapRefuse, match="missing required clock"):
        G.extract_clocks({"prefill_wall_ns": 1, "decode_wall_ns": 1})
    with pytest.raises(G.GapRefuse, match="not a duration"):
        G.extract_clocks(_reply(prefill_wall_ns="fast"))
    with pytest.raises(G.GapRefuse, match="negative"):
        G.extract_clocks(_reply(decode_wall_ns=-1))
    # Zero is a claim the clock existed and read zero. That is allowed.
    clocks = G.extract_clocks(_reply(prefill_wall_ns=0))
    assert clocks["prefill_ns"] == 0


def test_residue_is_unattributed_never_folded():
    """NEGATIVE CONTROL: leftover is UNATTRIBUTED, not prefill, not decode."""
    # 50 ns of generation leftover, 20_000_000 ns of request ceremony.
    reply = _reply(generation_wall_ns=1_500_000_050)
    decomp = G.decompose_reply(reply)
    un = decomp["unattributed"]
    assert un["label"] == G.UNATTRIBUTED
    assert un["ns"] == 50
    assert un["folded_into_prefill"] is False
    assert un["folded_into_decode"] is False
    prefill = next(b for b in decomp["buckets"] if b["id"] == "prefill")
    decode = next(b for b in decomp["buckets"] if b["id"] == "decode")
    assert prefill["ns"] == 400_000_000
    assert decode["ns"] == 1_100_000_000
    assert prefill["ns"] + decode["ns"] + un["ns"] == decomp["clocks"]["generation_ns"]
    with pytest.raises(G.GapRefuse, match="UNATTRIBUTED"):
        G.refuse_fold_into_named_bucket(G.UNATTRIBUTED, "prefill")
    with pytest.raises(G.GapRefuse, match="UNATTRIBUTED"):
        G.refuse_fold_into_named_bucket(G.UNATTRIBUTED, "decode")


def test_unattributed_dominating_refuses_a_winner():
    """NEGATIVE CONTROL: a leftover bigger than prefill/decode is not renamed."""
    reply = G.synthetic_reply(
        prefill_ns=10,
        decode_ns=10,
        generation_ns=10_000_000,
        request_ns=10_000_000,
        generated_tokens=2,
        decode_steps=1,
        prompt_tokens=1,
    )
    decomp = G.decompose_reply(reply)
    assert decomp["unattributed"]["ns"] == 10_000_000 - 20
    ranking = G.tallest_in_decomposition(decomp)
    assert ranking["named"] is False
    assert ranking["winner"] is None
    assert "UNATTRIBUTED" in ranking["reason"]
    assert ranking["does_not_claim_would_improve"] is True


def test_prefill_is_not_always_the_tallest():
    """NEGATIVE CONTROL: the 'prefill dominates' judgement can return false."""
    # Long decode, tiny prompt: decode wins the mass.
    reply = G.synthetic_reply(
        prefill_ns=50_000_000,
        decode_ns=5_000_000_000,
        generation_ns=5_050_000_000,
        request_ns=5_051_000_000,
        generated_tokens=200,
        decode_steps=199,
        prompt_tokens=2,
    )
    decomp = G.decompose_reply(reply)
    ranking = G.tallest_in_decomposition(decomp)
    assert ranking["named"] is True
    assert ranking["winner"] == "decode"
    assert ranking["winner"] != "prefill"
    shares = decomp["shares_of_request"]
    assert shares["decode"] is not None and shares["prefill"] is not None
    assert shares["decode"] > shares["prefill"]


def test_prefill_can_lead_on_a_short_generation():
    reply = G.synthetic_reply(
        prefill_ns=1_200_000_000,
        decode_ns=700_000_000,
        generation_ns=1_900_000_000,
        request_ns=1_900_000_000,
        generated_tokens=24,
        decode_steps=23,
        prompt_tokens=34,
    )
    ranking = G.tallest_in_decomposition(G.decompose_reply(reply))
    assert ranking["named"] is True
    assert ranking["winner"] == "prefill"
    assert ranking["does_not_claim_would_improve"] is True
    assert ranking["status"] == G.HYPOTHESIS
    assert ranking["removable"] == G.UNKNOWN


def test_would_improve_without_dirty_pair_raises():
    """NEGATIVE CONTROL: a plan is not a speedup."""
    with pytest.raises(G.WouldImproveRefuse, match="no measurement"):
        G.refuse_would_improve("batched prefill")
    with pytest.raises(G.WouldImproveRefuse, match="not SELF_MEASURED_DIRTY"):
        G.refuse_would_improve(
            "drop inner reset",
            dirty_measurement={"evidence_class": "STATIC_ONLY"},
        )
    with pytest.raises(G.WouldImproveRefuse, match="without a paired"):
        G.refuse_would_improve(
            "drop inner reset",
            dirty_measurement={"evidence_class": G.DIRTY},
        )
    admitted = G.refuse_would_improve(
        "drop inner reset",
        dirty_measurement={
            "evidence_class": G.DIRTY,
            "observed_before_and_after": True,
        },
    )
    assert admitted["would_improve"] is False
    assert admitted["does_not_promote"] is True
    assert admitted["status"] == G.HYPOTHESIS


def test_receipt_contains_no_would_improve_claim():
    doc = json.loads(G.build().read_text())
    assert doc["workunit"]["does_not_claim_would_improve"] is True
    assert doc["tallest"]["does_not_claim_would_improve"] is True
    assert doc["workunit"].get("would_improve") is not True
    assert "would_improve" not in doc["tallest"]


def test_first_token_off_by_one_can_fail():
    """NEGATIVE CONTROL: accounting reports a miss instead of forcing n_new-1."""
    ok = G.first_token_accounting(_reply())
    assert ok["off_by_one"] is True
    assert ok["status"] == "MATCHES_SOURCE"
    bad = G.first_token_accounting(_reply(decode_steps=40))
    assert bad["off_by_one"] is False
    assert bad["status"] == "DOES_NOT_MATCH_SOURCE"
    missing = G.first_token_accounting(_reply(generated_tokens="lots"))
    assert missing["status"] == G.UNKNOWN
    assert missing["off_by_one"] is None


def test_recoverability_is_unknown_when_qualification_absent():
    """NEGATIVE CONTROL: missing 34.0 conditions are UNKNOWN, not a regression."""
    verdict = G.recoverability_verdict(
        {"recovered": False, "reason": "absent"},
        [],
    )
    assert verdict["verdict"] == G.UNKNOWN
    assert verdict["assumed_regression"] is False
    assert "not inferred as a regression" in verdict["reason"]
    assert "REGRESSION" not in verdict["verdict"]


def test_recoverability_is_unknown_on_denominator_mismatch():
    hist = {
        "recovered": True,
        "quantity": "inverse_median_gpu_ns_per_step",
        "recorded_single_stream_tokens_per_second": 34.1488,
    }
    diffs = G.condition_diff({}, hist)
    quantity = next(d for d in diffs if d["axis"] == "quantity")
    assert quantity["differs"] is True
    verdict = G.recoverability_verdict(hist, diffs)
    assert verdict["verdict"] == G.UNKNOWN
    assert verdict["assumed_regression"] is False
    assert verdict["assumed_number_was_wrong"] is False
    assert "complete" in verdict["reason"].lower()
    assert "REGRESSION" not in json.dumps(verdict)


def test_historical_34_quantity_is_inverse_gpu_step_not_complete():
    how, qual = G.load_authority(G.QUAL_REL)
    recovered = G.recover_historical_34(qual, how)
    if not recovered["recovered"]:
        # Sparse-missing is not a skip: the function already returned UNKNOWN.
        assert recovered["quantity"] == G.UNKNOWN
        assert recovered["recovered"] is False
        return
    assert recovered["quantity"] == "inverse_median_gpu_ns_per_step"
    assert recovered["not_complete_tps"] is True
    assert recovered["matches_sealed_34"] is True
    assert recovered["binary"].endswith("ascension_qwen38_hybrid_greedy")
    assert 24 in (recovered.get("n_new_tokens_observed") or [])
    assert recovered.get("fusion_graph") in {"UNFUSED_BASELINE_964", G.UNKNOWN}
    assert recovered.get("prompt_tokens_inferred_from_step_count") == 34 or recovered.get(
        "prompt_tokens_inferred_from_step_count"
    ) in {None, 34}


def test_live_probe_without_clocks_is_insufficient_not_invented():
    how, probe = G.load_authority(G.PROBE_REL)
    live = G.recover_live_probe(probe, how)
    if not live["recovered"]:
        assert live["has_four_clocks"] is False
        return
    assert live["has_four_clocks"] is False
    assert live["lane_brief_citation"]["in_probe_receipt"] is False
    assert live["lane_brief_citation"]["this_module_did_not_remeasure"] is True
    assert live["lane_brief_citation"]["status"] == G.HYPOTHESIS
    assert live["r2_generated_tokens"] == 40


def test_wall_ns_alias_is_read_but_not_written():
    clocks = G.extract_clocks(_reply())
    assert clocks["request_ns"] == 1_520_000_000
    assert clocks["aliases_used"][G.CLOCK_REQUEST] == "wall_ns"
    doc = json.loads(G.build().read_text())

    def keys_of(node: object) -> set[str]:
        out: set[str] = set()
        if isinstance(node, dict):
            out.update(node)
            for v in node.values():
                out |= keys_of(v)
        elif isinstance(node, list):
            for v in node:
                out |= keys_of(v)
        return out

    named = keys_of(doc)
    for banned in HARDWARE_FIELDS:
        # The string may appear in prose; the *key* must not carry a number.
        if banned in named:
            # Allowed only if every instance is non-numeric (null / str / dict).
            def values_for(node: object, key: str, acc: list) -> None:
                if isinstance(node, dict):
                    if key in node:
                        acc.append(node[key])
                    for v in node.values():
                        values_for(v, key, acc)
                elif isinstance(node, list):
                    for v in node:
                        values_for(v, key, acc)

            found: list = []
            values_for(doc, banned, found)
            for v in found:
                assert not isinstance(v, (int, float)) or isinstance(v, bool)


def test_negative_generation_residue_is_clock_inconsistent():
    reply = G.synthetic_reply(
        prefill_ns=100,
        decode_ns=100,
        generation_ns=50,
        request_ns=50,
        generated_tokens=2,
        decode_steps=1,
        prompt_tokens=1,
    )
    decomp = G.decompose_reply(reply)
    assert decomp["clock_state"] == "CLOCK_INCONSISTENT"
    assert decomp["unattributed"]["label"] == G.UNATTRIBUTED
    assert decomp["unattributed"]["folded_into_prefill"] is False


def test_workunit_is_cpu_analysis_and_not_a_campaign():
    unit = G.emit_workunit(
        {"named": True, "winner": "prefill", "removable": G.UNKNOWN, "does_not_claim_would_improve": True},
        {"verdict": G.UNKNOWN},
    )
    assert unit["resource_class"] == "CPU_ANALYSIS"
    assert unit["gpu_authority"] is False
    assert unit["species"] == "PROFILE_HOST_CEREMONY"
    assert unit["status"] == G.HYPOTHESIS
    assert "stop" in unit["stop_condition"].lower()
    assert unit["does_not_claim_would_improve"] is True


def test_module_parses_and_has_no_skip_or_stub():
    src = Path(G.__file__).read_text()
    ast.parse(src)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            fn = node.exc.func
            name = fn.id if isinstance(fn, ast.Name) else (
                fn.attr if isinstance(fn, ast.Attribute) else ""
            )
            assert name != "NotImplementedError"
        if isinstance(node, ast.Pass):
            # A bare pass in this module is a stub. Allow none.
            raise AssertionError(f"pass at line {node.lineno}")
    assert "TODO" not in src
    test_tree = ast.parse(Path(__file__).read_text())
    for node in ast.walk(test_tree):
        if isinstance(node, ast.Attribute) and node.attr in {"skip", "xfail"}:
            if isinstance(node.value, ast.Name) and node.value.id == "pytest":
                raise AssertionError(f"pytest.{node.attr} at line {node.lineno}")


def test_build_records_unknown_live_reply():
    doc = json.loads(G.build().read_text())
    live = doc["question_1_complete_minus_decode"]["live_resident_reply"]
    assert live["status"] == G.UNKNOWN
    assert live["this_module_did_not_run_the_resident"] is True
    rec = doc["question_2_is_34_recoverable"]["recoverability"]
    assert rec["verdict"] == G.UNKNOWN
    assert rec["assumed_regression"] is False
