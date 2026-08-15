"""TG-ladder + Self-TG gauntlet harness scaffold tests.

No live Qwen benchmarks. Asserts metric separation, TG3 review rule, and the
Self-TG loop inventory. Generalizes DSV4F complete-token profile + metrics_sep.
"""

from __future__ import annotations

import pytest

from lab.operators.ascension_parity_ladder import ModelFamily, RungStatus
from lab.operators.ascension_tg_gauntlet import (
    SEPARATED_METRIC_NAMES,
    SELF_TG_LOOP_STEPS,
    TG3_REVIEW_THRESHOLD,
    TG3_TARGET_TPS,
    TG_LADDER,
    TG_RUNG_REQUIREMENTS,
    CompleteTokenProfiler,
    SeparatedMetric,
    TgGauntletHarness,
    TgRungObservation,
    assert_no_blended_tps,
    empty_separated_scoreboard,
    evaluate_tg_rung,
    scaffold_complete_token_profile,
    verify_gauntlet_receipt,
)
from lab.receipts import verify


def test_tg_ladder_targets_match_bible_section_10() -> None:
    expected = {
        "TG32": 31.25,
        "TG20": 50.0,
        "TG16": 62.5,
        "TG12": 83.3,
        "TG10": 100.0,
        "TG8": 125.0,
        "TG5": 200.0,
        "TG4": 250.0,
        "TG3": 333.0,
        "TG2": 500.0,
        "TG1": 1000.0,
    }
    assert {r.id: r.target_tps for r in TG_LADDER} == expected
    assert TG3_REVIEW_THRESHOLD == "TG3"
    assert TG3_TARGET_TPS == 333.0
    tg3 = next(r for r in TG_LADDER if r.id == "TG3")
    assert "review" in tg3.note.lower()


def test_self_tg_loop_steps_match_bible() -> None:
    assert SELF_TG_LOOP_STEPS[0] == "profile_own_complete_token"
    assert "rank_bottleneck" in SELF_TG_LOOP_STEPS
    assert "propose_three_materially_different_mechanisms" in SELF_TG_LOOP_STEPS
    assert "implement_in_isolated_worktree" in SELF_TG_LOOP_STEPS
    assert "protected_parity" in SELF_TG_LOOP_STEPS
    assert "protected_capability" in SELF_TG_LOOP_STEPS
    assert "clean_benchmark" in SELF_TG_LOOP_STEPS
    assert "adversarial_review" in SELF_TG_LOOP_STEPS
    assert SELF_TG_LOOP_STEPS[-1] == "continue"
    assert len(SELF_TG_LOOP_STEPS) == 12


def test_tg_rung_requirements_include_fallback_zero_and_real_gpu() -> None:
    assert "fallback_eq_0" in TG_RUNG_REQUIREMENTS
    assert "real_gpu_dispatch" in TG_RUNG_REQUIREMENTS
    assert "stable_p99" in TG_RUNG_REQUIREMENTS
    assert "complete_token_timing" in TG_RUNG_REQUIREMENTS
    assert "batch_1_base_runtime" in TG_RUNG_REQUIREMENTS


def test_separated_metrics_are_complete_and_not_blendable() -> None:
    assert SeparatedMetric.BASE_TRUE_TPS.value in SEPARATED_METRIC_NAMES
    assert SeparatedMetric.BLOCK_EXECUTED_TPS.value in SEPARATED_METRIC_NAMES
    assert SeparatedMetric.ACCELERATED_ACCEPTED_TPS.value in SEPARATED_METRIC_NAMES
    assert SeparatedMetric.PREFILL_TPS.value in SEPARATED_METRIC_NAMES
    assert SeparatedMetric.TTFT.value in SEPARATED_METRIC_NAMES
    assert SeparatedMetric.HCLI_TOOL_AUGMENTED_THROUGHPUT.value in SEPARATED_METRIC_NAMES
    board = empty_separated_scoreboard(reason="test")
    verify(board, label="scoreboard")
    assert board["blended_tps_forbidden"] is True
    assert board["average_of_scoreboards_forbidden"] is True
    assert board["metrics"]["BASE_TRUE_TPS"]["value"] is None
    assert board["metrics"]["BASE_TRUE_TPS"]["status"] == RungStatus.METRIC_WITHHELD.value
    assert_no_blended_tps(board)
    with pytest.raises(ValueError, match="blended"):
        assert_no_blended_tps({**board, "mean_tps": 42.0})


def test_evaluate_tg_rung_withholds_without_base_true_tps() -> None:
    rung = TG_LADDER[0]  # TG32
    obs = TgRungObservation(
        base_true_tps=None,
        fallback_count=0,
        gpu_dispatches=10,
        p99_stable=True,
        same_model=True,
        same_capability_tier=True,
        clean_benchmark=True,
        prompt_dependent_coherent=True,
        complete_token_timing=True,
        batch_1_base_runtime=True,
    )
    assert evaluate_tg_rung(rung, obs) == RungStatus.BASE_TRUE_TPS_WITHHELD


def test_evaluate_tg_rung_rejects_fallback_and_missing_gpu() -> None:
    rung = TG_LADDER[0]
    base = dict(
        base_true_tps=100.0,
        fallback_count=0,
        gpu_dispatches=10,
        p99_stable=True,
        same_model=True,
        same_capability_tier=True,
        clean_benchmark=True,
        prompt_dependent_coherent=True,
        complete_token_timing=True,
        batch_1_base_runtime=True,
    )
    assert (
        evaluate_tg_rung(rung, TgRungObservation(**{**base, "fallback_count": 1}))
        == RungStatus.REJECT_FALLBACK_NONZERO
    )
    assert (
        evaluate_tg_rung(rung, TgRungObservation(**{**base, "gpu_dispatches": 0}))
        == RungStatus.REJECT_NO_REAL_GPU_DISPATCH
    )
    assert (
        evaluate_tg_rung(rung, TgRungObservation(**{**base, "base_true_tps": 10.0}))
        == RungStatus.REJECT_CAPABILITY
    )


def test_evaluate_tg3_emits_review_required_not_final_pass() -> None:
    tg3 = next(r for r in TG_LADDER if r.id == "TG3")
    obs = TgRungObservation(
        base_true_tps=400.0,
        fallback_count=0,
        gpu_dispatches=100,
        p99_stable=True,
        same_model=True,
        same_capability_tier=True,
        clean_benchmark=True,
        prompt_dependent_coherent=True,
        complete_token_timing=True,
        batch_1_base_runtime=True,
    )
    assert evaluate_tg_rung(tg3, obs) == RungStatus.TG3_REVIEW_REQUIRED
    # TG3 is not the final target even when TPS clears
    assert tg3.target_tps == 333.0
    tg1 = next(r for r in TG_LADDER if r.id == "TG1")
    assert evaluate_tg_rung(tg1, obs) == RungStatus.REJECT_CAPABILITY  # 400 < 1000
    obs_tg1 = TgRungObservation(
        base_true_tps=1000.0,
        fallback_count=0,
        gpu_dispatches=100,
        p99_stable=True,
        same_model=True,
        same_capability_tier=True,
        clean_benchmark=True,
        prompt_dependent_coherent=True,
        complete_token_timing=True,
        batch_1_base_runtime=True,
    )
    assert evaluate_tg_rung(tg1, obs_tg1) == RungStatus.PASS_FULL_STACK


def test_complete_token_profiler_has_all_family_stages_and_zero_other() -> None:
    profile = scaffold_complete_token_profile(ModelFamily.QWEN3_MOE)
    verify(profile, label="complete token profile")
    stages = {row["stage"] for row in profile["stage_metrics"]}
    from lab.operators.ascension_parity_ladder import QWEN3_MOE_STAGES

    assert stages == set(QWEN3_MOE_STAGES)
    assert profile["timing_accounting"]["unexplained_other_wall_elapsed_ms"] == 0.0
    assert profile["timing_accounting"]["status"] == "PASS_ALL_TIME_EXPLICITLY_NAMED"
    assert profile["claim_boundary"]["base_true_tps"] is False


def test_complete_token_profiler_rejects_unknown_stage() -> None:
    p = CompleteTokenProfiler(
        family="QWEN3_MOE",
        phase="decode",
        token_ordinal=0,
        position=0,
        stages=("embedding", "runtime_bookkeeping"),
    )
    with pytest.raises(KeyError):
        p.mark_executed("not_a_stage", cpu_wall_elapsed_ms=1.0)


def test_gauntlet_receipt_for_30b_and_80b_is_scaffold_only() -> None:
    for family in (ModelFamily.QWEN3_MOE, ModelFamily.QWEN3_NEXT):
        h = TgGauntletHarness(family=family, weights_present=False)
        receipt = h.gauntlet_receipt()
        verify_gauntlet_receipt(receipt)
        assert receipt["status"] == RungStatus.PASS_SCAFFOLD_CONTRACT.value
        assert receipt["honesty"]["live_benchmark"] is False
        assert receipt["honesty"]["base_true_tps_claimed"] is False
        assert receipt["tg3_rule"]["is_final_target"] is False
        assert "emit_TG3_REVIEW_REQUIRED" in receipt["tg3_rule"]["action"]
        assert len(receipt["rung_receipts"]) == len(TG_LADDER)
        loop = receipt["self_tg_loop"]
        verify(loop, label="self-tg loop")
        assert loop["models_cannot_promote_themselves"] is True
        for rung_doc in receipt["rung_receipts"]:
            verify(rung_doc, label=rung_doc["rung"]["id"])
            assert rung_doc["status"] == RungStatus.REJECT_WEIGHTS_ABSENT.value
        assert_no_blended_tps(receipt["separated_scoreboard"])


def test_qwen3_next_profile_includes_deltanet_stages() -> None:
    profile = scaffold_complete_token_profile(ModelFamily.QWEN3_NEXT)
    stages = {row["stage"] for row in profile["stage_metrics"]}
    assert "gated_deltanet_state" in stages
    assert "deltanet_update" in stages
    assert "hybrid_schedule_slot" in stages
    assert "state_memory_accounting" in stages
