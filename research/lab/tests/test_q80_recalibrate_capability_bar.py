"""Unit tests for Q80 capability-bar recalibration. No source shards, no GPU."""
from __future__ import annotations

import math

import numpy as np

from lab.operators.q80_recalibrate_capability_bar import (
    GATE_CODECS,
    DOWN_CODECS,
    HISTORICAL_BAR,
    TARGET_SUBBIT,
    UP_CODECS,
    classify_text,
    clears_bar,
    compose_recipe,
    derive_bar_from_generation,
    human_class,
    implied_vs_bf16,
    mix_matched_cosine,
    reconstruct_grid,
    rethreshold,
)
from lab.operators.q80_subbit_capability_curve import BAR


def test_historical_bar_is_untouched() -> None:
    assert BAR == 0.8604
    assert HISTORICAL_BAR == BAR


def test_grid_is_588() -> None:
    assert len(GATE_CODECS) * len(UP_CODECS) * len(DOWN_CODECS) * 2 == 588


def test_mix_matched_cosine_hits_alpha() -> None:
    rng = np.random.default_rng(0)
    y = rng.standard_normal(256).astype(np.float32)
    for alpha in (1.0, 0.8, 0.5, 0.25, 0.0):
        hat = mix_matched_cosine(y, alpha, seed=7)
        cos = float(np.dot(y, hat) / (np.linalg.norm(y) * np.linalg.norm(hat)))
        assert abs(cos - alpha) < 1e-5


def test_implied_vs_bf16_is_product() -> None:
    inc = {"gate_proj": 0.8, "up_proj": 0.6, "down_proj": 0.4}
    mix = {"gate_proj": 0.5, "up_proj": 1.0, "down_proj": 0.25}
    got = implied_vs_bf16(inc, mix)
    assert got["gate_proj"] == 0.4
    assert got["up_proj"] == 0.6
    assert got["down_proj"] == 0.1


def test_rethreshold_does_not_invent_a_pass() -> None:
    means = {}
    for organ, codecs in (
        ("gate_proj", GATE_CODECS),
        ("up_proj", UP_CODECS),
        ("down_proj", DOWN_CODECS),
    ):
        for codec in codecs:
            means[(organ, codec)] = {
                "mean_output_cosine": 0.77,
                "mean_payload_bytes": 100_000.0,
                "n_scored": 101,
            }
    grid = reconstruct_grid(means, ne8_bytes=2_439_063_174, ne4_bytes=1_256_552_138)
    assert len(grid) == 588
    hist = rethreshold(grid, 0.8604)
    assert hist["n_clearing_all_organs"] == 0
    low = rethreshold(grid, 0.76)
    assert low["n_clearing_all_organs"] == 588
    # A bar of 0.77 exactly clears ( >= ).
    assert rethreshold(grid, 0.77)["n_clearing_all_organs"] == 588


def test_derive_bar_takes_lowest_still_coherent() -> None:
    rows = [
        {
            "point": "control",
            "prompt_name": "a",
            "coherence_class": "COHERENT",
            "generated_text": "ok",
            "min_organ_cosine_for_bar": 0.768,
            "kind": "identity",
        },
        {
            "point": "control",
            "prompt_name": "b",
            "coherence_class": "COHERENT",
            "generated_text": "ok",
            "min_organ_cosine_for_bar": 0.768,
            "kind": "identity",
        },
        {
            "point": "mid",
            "prompt_name": "a",
            "coherence_class": "COHERENT",
            "generated_text": "ok",
            "min_organ_cosine_for_bar": 0.60,
            "kind": "all_organ_mix",
        },
        {
            "point": "mid",
            "prompt_name": "b",
            "coherence_class": "COHERENT",
            "generated_text": "ok",
            "min_organ_cosine_for_bar": 0.60,
            "kind": "all_organ_mix",
        },
        {
            "point": "low",
            "prompt_name": "a",
            "coherence_class": "INCOHERENT",
            "generated_text": "??",
            "min_organ_cosine_for_bar": 0.40,
            "kind": "all_organ_mix",
        },
        {
            "point": "low",
            "prompt_name": "b",
            "coherence_class": "DEGRADED",
            "generated_text": "??",
            "min_organ_cosine_for_bar": 0.40,
            "kind": "all_organ_mix",
        },
    ]
    report = derive_bar_from_generation(rows, incumbent_min=0.768)
    assert report["status"] == "DERIVED_FROM_GENERATION"
    assert report["corrected_bar"] == 0.60
    assert report["first_broken_min_organ_cosine"] == 0.40
    assert report["historical_bar_falsified_by_incumbent"] is True


def test_down_prefix_does_not_set_the_all_organ_bar() -> None:
    rows = [
        {
            "point": "control",
            "prompt_name": "a",
            "coherence_class": "COHERENT",
            "generated_text": "ok",
            "min_organ_cosine_for_bar": 0.768,
            "kind": "identity",
        },
        {
            "point": "down8",
            "prompt_name": "a",
            "coherence_class": "COHERENT",
            "generated_text": "ok",
            "min_organ_cosine_for_bar": 0.43,
            "kind": "down_prefix",
        },
    ]
    report = derive_bar_from_generation(rows, incumbent_min=0.768)
    assert report["corrected_bar"] == 0.768


def test_classify_matches_incumbent_snippet() -> None:
    text = "Here's a function that reverses a string (i.e"
    assert classify_text(text, ["function", "reverse", "string", "here's"]) == "COHERENT"
    assert classify_text("", ["function"]) == "INCOHERENT"


def test_echo_of_prompt_is_not_an_answer() -> None:
    prompt = "Write a function that reverses a string."
    assert human_class(prompt, prompt, "COHERENT") == "ECHO"
    assert human_class("Here is a function that **re**es a string.", prompt, "COHERENT") == "DEGRADED"
    assert (
        human_class(
            "Here's a function that reverses a string (i.e., reverses the order",
            prompt,
            "COHERENT",
        )
        == "COHERENT"
    )


def test_clears_bar_requires_every_organ() -> None:
    rec = compose_recipe(
        {
            ("gate_proj", "binary_g128"): {
                "mean_output_cosine": 0.88,
                "mean_payload_bytes": 1.0,
                "n_scored": 1,
            },
            ("up_proj", "resid_2pct"): {
                "mean_output_cosine": 0.87,
                "mean_payload_bytes": 1.0,
                "n_scored": 1,
            },
            ("down_proj", "hgravs01_r160_b3"): {
                "mean_output_cosine": 0.76,
                "mean_payload_bytes": 1.0,
                "n_scored": 1,
            },
        },
        gate="binary_g128",
        up="resid_2pct",
        down="hgravs01_r160_b3",
        nonexpert_bits=8,
        nonexpert_bytes=100,
    )
    assert not clears_bar(rec, 0.8604)
    assert clears_bar(rec, 0.76)
    assert math.isfinite(rec["complete_physical_bpw"])
    assert TARGET_SUBBIT == 0.6552
