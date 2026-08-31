"""Decode civilization: objective, rollback, verification WHERE vs WHAT."""
from __future__ import annotations

import json

import pytest

from tools.future import decode_civilization as dc
from tools.future._common import HARDWARE_FIELDS, RECEIPTS


def test_build_emits_sealed_receipt():
    out = dc.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "DECODE_CIVILIZATION.json"
    assert doc["schema"] == dc.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["selftest"]["ok"] is True
    assert doc["objective"]["includes_rollback"] is True
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]


def test_selftest_passes():
    report = dc.selftest()
    assert report["ok"] is True
    names = {c["name"] for c in report["checks"]}
    assert "objective_rejects_high_draft_low_accept" in names
    assert "sampled_what_is_refused" in names


def test_baseline_complete_token_cost_is_one():
    assert dc.accepted_complete_token_cost(dc.BASELINE_PLAN) == pytest.approx(1.0)


def test_leviathan_yield_matches_recovered_extremes():
    assert dc.expected_accepted_per_pass(0.0, 4) == 1.0
    assert dc.expected_accepted_per_pass(1.0, 4) == 5.0
    assert dc.expected_accepted_per_pass(0.5, 1) == pytest.approx(1.5)
    assert dc.expected_accepted_per_pass(1.0, 4, yield_includes_bonus=False) == 4.0
    assert dc.expected_accepted_per_pass(0.0, 4, yield_includes_bonus=False) == 1.0


def test_negative_control_high_draft_throughput_cannot_win():
    """High draft throughput + poor α must score WORSE than slower high-α.

    A guard nobody has watched fail is not a guard: raw_draft_throughput_units
    must prefer the fast plan, while the objective prefers the slow one.
    """
    fast = dc.HIGH_DRAFT_LOW_ACCEPT
    slow = dc.SLOW_DRAFT_HIGH_ACCEPT
    fast_raw = dc.raw_draft_throughput_units(fast)
    slow_raw = dc.raw_draft_throughput_units(slow)
    fast_cost = dc.accepted_complete_token_cost(fast)
    slow_cost = dc.accepted_complete_token_cost(slow)
    assert fast_raw > slow_raw, (
        f"fixture broken: fast raw throughput {fast_raw} must exceed slow {slow_raw}"
    )
    assert fast.alpha < slow.alpha
    assert fast.gamma > slow.gamma
    assert fast_cost > slow_cost, (
        f"raw throughput won the objective: fast={fast_cost} slow={slow_cost}"
    )
    assert slow_cost < 1.0
    assert fast_cost > 1.0


def test_g057_high_acceptance_still_loses():
    """The recorded trap: 87% accept at draft/verify=0.75 is a slowdown."""
    g057 = dc.accepted_complete_token_cost(dc.G057_HIGH_ACCEPT_SLOWDOWN)
    base = dc.accepted_complete_token_cost(dc.BASELINE_PLAN)
    assert dc.G057_HIGH_ACCEPT_SLOWDOWN.alpha == pytest.approx(0.87)
    assert g057 > base
    # G038 k=1 "pays if draft/verify < alpha" is TRUE here (0.75 < 0.87).
    # That is the trap: k=1 economics do not rescue a K=4 sequential draft.
    assert dc.k1_pays(dc.G057_ACCEPTANCE, dc.G057_DRAFT_OVER_VERIFY, 1.0)


def test_rollback_is_inside_the_objective():
    with_rb = dc.HIGH_DRAFT_LOW_ACCEPT
    without = dc.DecodePlan(
        name="no_rollback",
        gamma=with_rb.gamma,
        alpha=with_rb.alpha,
        draft_cost=with_rb.draft_cost,
        verify_cost=with_rb.verify_cost,
        rollback_cost=0.0,
        ceremony_cost=with_rb.ceremony_cost,
        verification_placement=with_rb.verification_placement,
        draft_kind=with_rb.draft_kind,
    )
    c_with = dc.accepted_complete_token_cost(with_rb)
    c_without = dc.accepted_complete_token_cost(without)
    assert c_with > c_without
    parts = dc.cycle_cost_units(with_rb)
    assert parts["rollback_leg"] > 0.0
    assert parts["p_rollback"] == pytest.approx(1.0 - (with_rb.alpha ** with_rb.gamma))


def test_perfect_accept_pays_no_rollback():
    plan = dc.DecodePlan(
        name="perfect",
        gamma=4,
        alpha=1.0,
        draft_cost=0.2,
        verify_cost=1.0,
        rollback_cost=0.5,
        verification_placement="fused",
    )
    parts = dc.cycle_cost_units(plan)
    assert parts["rollback_leg"] == 0.0
    assert parts["p_rollback"] == 0.0


def test_fused_ceremony_reduction_is_allowed_and_helps():
    host = dc.SLOW_DRAFT_HIGH_ACCEPT
    fused = dc.DecodePlan(
        name="fused",
        gamma=host.gamma,
        alpha=host.alpha,
        draft_cost=host.draft_cost,
        verify_cost=host.verify_cost,
        rollback_cost=host.rollback_cost,
        ceremony_cost=0.0,
        verification_placement="fused",
        draft_kind=host.draft_kind,
    )
    graph = dc.place_verification("fused")
    assert graph["schema"] == dc.PHYSICAL_GRAPH_SCHEMA
    assert graph["qualification"] == "PLAN_ONLY"
    assert graph["computation"][0]["predicate"] == dc.PREDICATE_EXACT
    assert graph["correctness_preserving"] is True
    assert dc.accepted_complete_token_cost(fused) < dc.accepted_complete_token_cost(host)


def test_negative_control_sampled_verification_is_rejected():
    """Refusal must actually fire — sampled changes WHAT is checked."""
    with pytest.raises(dc.VerificationCorrectnessError, match="WHAT_NOT_WHERE"):
        dc.place_verification("sampled")
    with pytest.raises(dc.VerificationCorrectnessError, match="WHAT_NOT_WHERE"):
        dc.place_verification("sparse")
    with pytest.raises(dc.VerificationCorrectnessError, match="WHAT_NOT_WHERE"):
        dc.accepted_complete_token_cost(
            dc.DecodePlan(
                name="sampled_cheat",
                gamma=8,
                alpha=0.95,
                draft_cost=0.05,
                verification_placement="sampled",
            )
        )


def test_digest_of_same_predicate_is_admitted():
    graph = dc.place_verification("digest")
    assert graph["admitted"] is True
    assert graph["representation"]["digest_commits_to_same_predicate"] is True
    assert graph["computation"][0]["what_is_checked"] == dc.PREDICATE_EXACT


def test_digest_of_weaker_predicate_is_rejected():
    with pytest.raises(dc.VerificationCorrectnessError, match="WHAT_NOT_WHERE"):
        dc.place_verification("digest", predicate="first_token_only")
    with pytest.raises(dc.VerificationCorrectnessError, match="WHAT_NOT_WHERE"):
        dc.place_verification("device_side", predicate="sketched_topk")


def test_tokenizer_inflation_can_make_a_smaller_vocab_lose():
    """G036: inflation is wall-time one-for-one; 11% byte save does not repay 1.19x tokens."""
    shrunk = dc.DecodePlan(
        name="required_hot",
        gamma=0,
        verify_cost=1.0,
        token_inflation=1.1912,
        lm_head_scale=0.013,
        lm_head_share=dc.TOKENIZER_GRAVITY_PAYLOAD_SHARE,
        verification_placement="host_exact",
    )
    assert dc.accepted_complete_token_cost(shrunk) > dc.accepted_complete_token_cost(
        dc.BASELINE_PLAN
    )


def test_kv_compression_is_bytes_class_not_capability():
    model = dc.kv_compression_model()
    assert model["asymmetric_kv_is_not_a_result"] is True
    assert {row["capability"] for row in model["ladder"]} == {"ABSENT"}
    bf16 = next(r for r in model["ladder"] if r["scheme"] == "bf16_bf16")
    q4 = next(r for r in model["ladder"] if r["scheme"] == "q4_q4")
    assert q4["bytes_per_token"] * 4 == bf16["bytes_per_token"]


def test_hybrid_state_crossover_is_the_cited_1152():
    st = dc.qwen38_state_bytes()
    assert st["kv_bytes_per_token_bf16"] == 65536
    assert st["recurrent_bytes_bf16"] == 75497472
    assert st["crossover_tokens_bf16"] == pytest.approx(1152.0)
    assert st["recurrent_grows_with_context"] is False
    assert st["kv_grows_with_context"] is True


def test_receipt_contains_no_hardware_numbers():
    out = dc.build()
    doc = json.loads(out.read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else key
                if key in HARDWARE_FIELDS:
                    assert not isinstance(value, (int, float)), here
                walk(value, here)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(doc)


def test_speculative_interfaces_are_named():
    ifaces = dc.speculative_interfaces()
    for name in (
        "mtp",
        "speculative_decoding",
        "draft_verify",
        "multi_token_microdecoder",
        "state_rollback",
        "acceptance_accounting",
    ):
        assert name in ifaces
    assert ifaces["draft_verify"]["verify_rule"] == dc.PREDICATE_EXACT
    assert ifaces["state_rollback"]["in_objective"] is True
    assert ifaces["acceptance_accounting"]["this_sidecar_does_not_emit_tps"] is True


def test_qwen27_budget_absence_is_recorded():
    findings = dc.negative_findings()
    qwen27 = next(f for f in findings if f["looked_for"].endswith("QWEN27_TOKEN_NS_BUDGET.json"))
    # Environment-coupled: this file is uncommitted, so it is invisible from a
    # sparse lane worktree and visible from the primary one. Its presence is a
    # fact about the checkout, not about this module -- assert the module COPES
    # either way rather than pinning the environment it was written in.
    assert isinstance(qwen27["found"], bool)
    assert qwen27["used_instead"] == dc.FLASH_BUDGET_REL
