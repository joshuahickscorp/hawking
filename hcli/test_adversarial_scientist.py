"""Regression tests for SCIENTIFIC behaviour (S008 section 15).

Every fixture below is a REAL measurement from the O003 campaign on 2026-09-06.
Each trap fired once and cost real time; these pin that it cannot fire silently
again. Both directions are tested: the trap must be caught, and honest data must
NOT be flagged.
"""
from hcli.scientific_traps import (
    ablation_had_no_effect, constant_masquerading_as_measurement,
    distinct_specs_same_bytes, gate_ignores_generation, incomplete_accounting,
    no_new_information, self_certified, throughput_exceeds_physics,
    verdict_from_too_few_samples,
)


def test_cached_constant_is_rejected():
    # tps_specimen for O003 across q2-g32-experts, mixed-q2q3, q4-g128, q4-g64
    ok, why = constant_masquerading_as_measurement([51.627, 51.627, 51.627, 51.627])
    assert not ok, why
    # the real measurements from the same specimen DO vary
    ok, _ = constant_masquerading_as_measurement([87.67, 127.48, 132.12, 128.97])
    assert ok


def test_component_bytes_cannot_pass_as_complete():
    ok, _ = incomplete_accounting({"payload_bytes": 4386646272, "complete_bytes": 4386646272})
    assert not ok
    # the real O003 q2-g32-experts accounting
    ok, why = incomplete_accounting({
        "payload_bytes": 4386646272, "scales_bytes": 948482048,
        "biases_bytes": 948482048, "metadata_bytes": 0,
        "complete_bytes": 6283956672})
    assert ok, why


def test_single_generation_cannot_settle_a_capability_verdict():
    ok, _ = verdict_from_too_few_samples(1)
    assert not ok
    assert verdict_from_too_few_samples(8)[0]


def test_lazy_eval_throughput_is_impossible():
    # the 7,061,969 tok/s prefill artefact, against M3 Ultra's ~800 GB/s
    ok, why = throughput_exceeds_physics(7_061_969, 5_158_467_840, 800e9)
    assert not ok, why
    # the real bf16 decode figure passes
    ok, why = throughput_exceeds_physics(87.67, 5_158_467_840, 800e9)
    assert ok, why


def test_ablation_that_changed_nothing_is_rejected():
    ok, _ = ablation_had_no_effect(1234.5678, 1234.5678)
    assert not ok
    assert ablation_had_no_effect(1234.5678, 999.1111)[0]


def test_two_specs_one_byte_count_is_rejected():
    # the predicate that skipped 78 SwitchLinear modules
    ok, why = distinct_specs_same_bytes("q4-g64", 29_756_639_488,
                                        "q2-g32-experts", 29_756_639_488)
    assert not ok, why
    ok, _ = distinct_specs_same_bytes("q4-g64", 8_982_644_992,
                                      "q2-g32-experts", 6_283_610_368)
    assert ok


def test_perplexity_only_gate_is_rejected():
    ok, _ = gate_ignores_generation({"ppl_max": 3.617})
    assert not ok
    assert gate_ignores_generation({"ppl_max": 3.617, "median_4gram_max": 0.1614})[0]


def test_repeated_identical_proposals_are_not_progress():
    ok, why = no_new_information(["binarypercal0.005-g128"] * 5)
    assert not ok, why
    assert no_new_information(["q1-g64-experts", "q2-g32-experts", "binarypercal-g256"])[0]


def test_self_certification_is_rejected():
    assert not self_certified("sealed-3.14", "sealed-3.14")[0]
    assert self_certified("sealed-3.14", "gravity_outlier_eval")[0]
