"""Oracle-first bounds for the three unjudged DeltaNet families.

Load-bearing refusals:

1. A family with a missing input RAISES; it is not skipped.
2. The oracle is scored on held-out prompt ids (the judge's code:14, code:15).
3. A train figure cannot be reported as held-out.
4. An absent family is a refuse, not a pass.
5. Synthetic X is refused (NNS-001).
6. Milliseconds are cited at per-stream rates; state is activation.
7. The bound is OPPORTUNITY_BOUND_ON_PERFECT_SUCCESS, not a speedup.
8. Capability is UNMEASURED.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.future import deltanet_multistep as dnm
from tools.future import deltanet_program_oracles as dpo
from tools.future import deltanet_state_function as dsf
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


@pytest.fixture(scope="module")
def built_receipt():
    rc = dpo.main(["--build"])
    assert rc == 0
    path = RECEIPTS / dpo.RECEIPT
    assert path.is_file()
    return json.loads(path.read_text())


def _tiny_seq(*, split: str, prompt_id: str, n_tokens: int = 24, seed: int = 38):
    x = dnm.make_fixture_hidden(n_tokens, hidden=dsf.HIDDEN, seed=seed)
    return {
        "prompt_id": prompt_id,
        "layer": dpo.JUDGE_LAYER,
        "split": split,
        "n_tokens": n_tokens,
        "x": x,
        "synthetic": False,
        "held_out_unit": "prompt_id",
    }


# ---------------------------------------------------------------------------
# Missing input raises. An absent family is not a pass.
# ---------------------------------------------------------------------------


def test_missing_corpus_raises_rather_than_skipping_a_family(tmp_path):
    """A family with a missing input raises; it is not omitted and is not a pass."""
    missing = tmp_path / "no_such_mlp_teacher_corpus"
    with pytest.raises(dpo.CorpusUnavailable, match="REFUSED") as caught:
        dpo.load_hold_sequences(payload=missing)
    assert str(missing) in str(caught.value)
    assert caught.value.missing_paths
    assert str(missing) in caught.value.missing_paths

    with pytest.raises(dpo.CorpusUnavailable, match="REFUSED"):
        dpo.evaluate_family("generated_coefficients", payload=missing)
    with pytest.raises(dpo.CorpusUnavailable, match="REFUSED"):
        dpo.evaluate_family("learned_recurrence", payload=missing)
    with pytest.raises(dpo.CorpusUnavailable, match="REFUSED"):
        dpo.evaluate_family("conditional_recurrence", payload=missing)


def test_an_absent_family_is_a_refuse_not_a_pass():
    with pytest.raises(dpo.FamilyAbsentRefuse, match="absent"):
        dpo.require_all_families({"families": {}})
    with pytest.raises(dpo.FamilyAbsentRefuse, match="absent"):
        dpo.require_all_families(
            {
                "families": {
                    "generated_coefficients": {
                        "verdict": dpo.ORACLE_NEGATIVE,
                        "oracle_bound": {},
                    }
                }
            }
        )
    with pytest.raises(dpo.FamilyAbsentRefuse, match="oracle_bound"):
        dpo.require_all_families(
            {
                "families": {
                    name: {"verdict": dpo.ORACLE_NEGATIVE}
                    for name in dpo.REQUIRED_FAMILIES
                }
            }
        )


def test_unknown_family_raises():
    with pytest.raises(dpo.OracleRefuse, match="unknown family"):
        dpo.evaluate_family("fused_update_and_consume")


def test_empty_hold_is_missing_input_not_a_skip():
    train = [_tiny_seq(split="train", prompt_id="code:00")]
    with pytest.raises(dpo.FamilyInputMissing, match="skipped"):
        dpo.evaluate_family(
            "generated_coefficients",
            hold=[],
            train=train,
        )


# ---------------------------------------------------------------------------
# Held-out by prompt_id. Same two prompts the judge used.
# ---------------------------------------------------------------------------


def test_train_figure_cannot_be_reported_as_held_out():
    train = _tiny_seq(split="train", prompt_id="code:00")
    with pytest.raises(dpo.HeldOutRefuse, match="held-out"):
        dpo.assert_held_out_sequences([train])

    hold = _tiny_seq(split="hold", prompt_id="code:00")
    with pytest.raises(dpo.HeldOutRefuse, match="train id"):
        dpo.assert_held_out_sequences([hold], train_ids=["code:00"])


def test_oracle_is_scored_on_held_out_prompt_ids(built_receipt):
    """The scored holdout is the judge's code:14, code:15, not train tokens."""
    inp = built_receipt["input"]
    assert inp["report_as"] == "held_out"
    assert inp["held_out_unit"] == "prompt_id"
    assert inp["prompt_ids"] == list(dpo.JUDGE_HOLD_PROMPT_IDS)
    assert inp["kind"] == dpo.JUDGE_KIND
    assert inp["layer"] == 38
    train_used = set(inp["train_prompt_ids_used_for_generator"])
    assert not (train_used & set(dpo.JUDGE_HOLD_PROMPT_IDS))
    for name in dpo.REQUIRED_FAMILIES:
        scored = built_receipt["families"][name]["scored_on"]
        assert scored["split"] == "hold"
        assert scored["held_out_unit"] == "prompt_id"
        assert set(scored["prompt_ids"]) == set(dpo.JUDGE_HOLD_PROMPT_IDS)
        assert "code:00" not in scored["prompt_ids"]


def test_judge_holdout_rejects_a_different_prompt_set():
    seqs = [
        _tiny_seq(split="hold", prompt_id="code:16"),
        _tiny_seq(split="hold", prompt_id="code:17"),
    ]
    with pytest.raises(dpo.HeldOutRefuse, match="code:14"):
        dpo.assert_judge_holdout(seqs)


def test_synthetic_row_is_refused():
    seq = _tiny_seq(split="hold", prompt_id="code:14")
    seq["synthetic"] = True
    with pytest.raises(dpo.CorpusUnavailable, match="NNS-001"):
        dpo.assert_held_out_sequences(
            [seq], allowed_ids=dpo.JUDGE_HOLD_PROMPT_IDS
        )


# ---------------------------------------------------------------------------
# Billing. Per-stream rates. Bound is not a speedup.
# ---------------------------------------------------------------------------


def test_state_and_weights_are_billed_at_different_stream_rates():
    rates = dpo.stream_rates()
    assert rates["weight_codes"] == pytest.approx(0.547282)
    assert rates["broadcast_aux"] == pytest.approx(0.0)
    assert rates["activation"] == pytest.approx(2.906132)
    w = dpo.cited_ms(1_000_000_000, "weight_codes")
    a = dpo.cited_ms(1_000_000_000, "activation")
    assert a > w * 4
    with pytest.raises(dpo.OracleRefuse, match="organ average"):
        dpo.cited_ms(1, "organ_average")


def test_opportunity_bound_is_labelled_not_a_speedup():
    bound = dpo.opportunity_bound(
        removed_weight_codes=dsf.QKVZ_ACTIVE_TARGET,
        removed_activation=0,
        added_weight_codes=4_548_560,
        added_activation=0,
        residual=1.813,
    )
    assert bound["kind"] == dpo.BOUND_KIND
    assert bound["kind"] == "OPPORTUNITY_BOUND_ON_PERFECT_SUCCESS"
    assert bound["not_a_speedup"] is True
    assert bound["ms_are_cited_not_measured"] is True
    assert bound["capability"] == dpo.UNMEASURED
    assert bound["state_billed_as"] == "activation"
    assert bound["clears_residual_gap"] is False
    assert bound["opportunity_bound_ms"] == pytest.approx(
        bound["cited_ms_removed"]["total"] - bound["cited_ms_added"]["total"]
    )
    # qkvz-only removal cannot close 1.813 ms even with a free generator.
    assert bound["opportunity_bound_ms"] < 1.813


def test_generator_byte_model_at_rank_256_reconciles_to_the_recorded_candidate():
    model = dpo.generator_byte_model(256)
    named = dpo.load_named_candidate("generated_transition_coefficients")
    assert model["bytes_added"]["total"] == named["bytes_added"]["total"]
    assert model["bytes_removed"]["total"] == named["bytes_removed"]["total"]
    assert model["bytes_added"]["total"] == 4_548_560
    assert model["bytes_removed"]["catalog_weights"] == 2_139_096_960


def test_conditional_bills_all_rules_and_the_router_is_free():
    one = dpo.conditional_rule_bytes(1)
    many = dpo.conditional_rule_bytes(8)
    assert many["bytes_added"]["generator"] == 8 * one["bytes_per_rule"]
    assert many["router_bytes"] == 0
    assert many["router_cited_ms"] == 0.0
    assert many["n_regimes"] == 8


def test_learned_byte_model_scales_from_the_named_d_state():
    named = dpo.learned_byte_model(16)
    recorded = dpo.load_named_candidate("learned_recurrence")
    assert named["bytes_added"]["total"] == recorded["bytes_added"]["total"]
    double = dpo.learned_byte_model(32)
    assert double["bytes_added"]["state"] == 2 * named["bytes_added"]["state"]
    # The recurrent state is added as ACTIVATION, not weight codes.
    bound = dpo.bound_from_added_removed(
        double["bytes_added"], double["bytes_removed"], residual=1.813
    )
    assert bound["bytes"]["added"]["activation"] == double["bytes_added"]["state"]
    assert bound["state_billed_as"] == "activation"


# ---------------------------------------------------------------------------
# Verdicts. All three families. Capability UNMEASURED.
# ---------------------------------------------------------------------------


def test_all_three_families_carry_a_verdict_and_a_bound(built_receipt):
    dpo.require_all_families(built_receipt)
    assert set(built_receipt["families"]) == set(dpo.REQUIRED_FAMILIES)
    for name in dpo.REQUIRED_FAMILIES:
        row = built_receipt["families"][name]
        assert row["verdict"] in dpo.VERDICTS
        assert row["verdict"] != dpo.BLOCKED
        bound = row["oracle_bound"]
        assert bound["kind"] == dpo.BOUND_KIND
        assert "opportunity_bound_ms" in bound
        assert "bytes" in bound
        assert bound["ms_are_cited_not_measured"] is True
        assert row["capability"] == dpo.UNMEASURED
        assert bound["capability"] == dpo.UNMEASURED
        assert "cited_ms_removed" in bound
        assert "cited_ms_added" in bound


def test_capability_is_unmeasured(built_receipt):
    assert built_receipt["bar"]["capability"] == dpo.UNMEASURED
    for name in dpo.REQUIRED_FAMILIES:
        assert built_receipt["families"][name]["capability"] == dpo.UNMEASURED
    assert any("UNMEASURED" in x for x in built_receipt["what_this_does_not_prove"])


def test_hardware_fields_are_refused(built_receipt):
    _assert_no_hardware_claims(built_receipt)
    for key in HARDWARE_FIELDS:
        blob = json.dumps(built_receipt)
        # Keys may appear in prose; numeric hardware fields must not.
        if key in built_receipt:
            assert built_receipt[key] in (None, "UNKNOWN", False)


def test_prior_is_not_neutral(built_receipt):
    prior = built_receipt["prior"]
    assert prior["not_neutral"] is True
    assert prior["already_judged"]["smaller_state"]["verdict"] == "MEASURED_NEGATIVE"
    assert prior["already_judged"]["structured_transitions"]["verdict"] == "MEASURED_NEGATIVE"


def test_generated_coefficients_reports_transition_reconstruction(built_receipt):
    row = built_receipt["families"]["generated_coefficients"]
    recon = row["reconstruction"]
    assert recon["of"].startswith("transition")
    assert recon["bar"] == dpo.TRANSITION_BAR
    assert recon["named_rank"] == 256
    named = recon["named_rank_hold_interpolating"]
    assert named["coefficient_relative_l2"] > dpo.TRANSITION_BAR
    assert named["clears_transition_bar"] is False
    bound = row["oracle_bound"]
    assert bound["clears_residual_gap"] is False
    assert row["verdict"] == dpo.ORACLE_NEGATIVE


def test_learned_recurrence_reports_the_budget_it_actually_needed(built_receipt):
    row = built_receipt["families"]["learned_recurrence"]
    budget = row["budget_actually_needed"]
    assert budget["d_state_for_store_bar"] > budget["named_d_state"]
    assert budget["named_d_state"] == 16
    recon = row["reconstruction"]
    assert recon["named_clears_transition_bar"] is False
    assert recon["named_one_step_state_relative_l2_max"] > dpo.TRANSITION_BAR
    assert row["verdict"] == dpo.ORACLE_NEGATIVE
    assert row["oracle_bound"]["bytes"]["added"]["activation"] > 0


def test_conditional_recurrence_does_not_reconstruct_under_perfect_routing(built_receipt):
    row = built_receipt["families"]["conditional_recurrence"]
    recon = row["reconstruction"]
    assert recon["clears_bar"] is False
    assert recon["best_one_step_state_relative_l2_max"] > dpo.TRANSITION_BAR
    points = row["operating_points"]
    assert {int(p["n_regimes"]) for p in points} >= {4, 16, 64}
    for p in points:
        assert p["byte_model"]["router_bytes"] == 0
        assert p["byte_model"]["bytes_added"]["generator"] == (
            p["n_regimes"] * p["byte_model"]["bytes_per_rule"]
        )
    assert row["verdict"] == dpo.ORACLE_NEGATIVE


def test_the_three_are_closable_as_oracle_negative(built_receipt):
    answers = built_receipt["answers"]
    assert answers["closable_as_oracle_negative"] is True
    assert answers["any_permits_investigation"] is False
    assert answers["verdicts"] == {
        "generated_coefficients": dpo.ORACLE_NEGATIVE,
        "learned_recurrence": dpo.ORACLE_NEGATIVE,
        "conditional_recurrence": dpo.ORACLE_NEGATIVE,
    }


def test_residual_gap_is_cited_from_economics_not_frozen(built_receipt):
    from tools.future import deltanet_state_machine_economics as sme

    live = sme.price()["gap_to_71_residual_after_everything_on_record_ms"]
    assert built_receipt["bar"]["residual_gap_ms_cited"] == pytest.approx(live)
    for name in dpo.REQUIRED_FAMILIES:
        gap = built_receipt["families"][name]["oracle_bound"]["residual_gap_ms_cited"]
        assert gap == pytest.approx(live)


def test_decide_verdict_requires_both_reconstruction_and_residual():
    # Only perfect removal of ALL DeltaNet codes AND state clears the residual.
    paying = dpo.opportunity_bound(
        removed_weight_codes=int(dsf.DELTANET_ACTIVE_TARGET),
        removed_activation=int(dsf.REC_STATE_RESIDENT) + int(dsf.CONV_STATE_RESIDENT),
        added_weight_codes=1,
        added_activation=1,
        residual=1.813,
    )
    assert paying["clears_residual_gap"] is True
    assert dpo.decide_verdict(reconstruction_clears=True, bound=paying) == (
        dpo.ORACLE_PERMITS_INVESTIGATION
    )
    assert dpo.decide_verdict(reconstruction_clears=False, bound=paying) == (
        dpo.ORACLE_NEGATIVE
    )
    poor = dpo.opportunity_bound(
        removed_weight_codes=1,
        removed_activation=0,
        added_weight_codes=0,
        added_activation=0,
        residual=1.813,
    )
    assert dpo.decide_verdict(reconstruction_clears=True, bound=poor) == (
        dpo.ORACLE_NEGATIVE
    )


def test_receipt_schema_and_seal(built_receipt):
    assert built_receipt["schema"] == dpo.SCHEMA
    assert built_receipt["gpu_authority"] is False
    assert built_receipt["evidence_class"] == dpo.EVIDENCE_CLASS
    assert built_receipt["claim_boundary"]
    assert "seal_sha256" in built_receipt
    assert len(built_receipt["seal_sha256"]) == 64
    assert Path(RECEIPTS / dpo.RECEIPT).is_file()
