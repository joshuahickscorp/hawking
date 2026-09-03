"""The two instruments that decide whether KV reuse is worth building on.

Both answer questions that were previously settled by wall clock, which cannot
separate "reuse worked" from "the prompt was shorter", and cannot separate
"attention grows with position" from "the host is doing more work per step".

The shapes here are SYNTHETIC and known in advance. An instrument that cannot
recover a shape it was handed cannot be trusted on a real trace.
"""
from __future__ import annotations

import pytest

from hcli.prefill_profile import attribute, bucket_profile
from hcli.prefix_probe import PrefixProbe, divergence_reason, longest_common_prefix


# --------------------------------------------------------------------------
# Prefix: what the BUILDER preserved
# --------------------------------------------------------------------------


def test_a_pure_append_is_fully_reusable():
    probe = PrefixProbe()
    assert probe.observe("g", "SYSTEM\nTOOLS\nOBS1") is None, "first turn has no prior"
    t = probe.observe("g", "SYSTEM\nTOOLS\nOBS1\nOBS2", prompt_tokens=100)
    assert t.reason_for_prefix_divergence == "pure_append"
    assert t.longest_common_prefix_chars == len("SYSTEM\nTOOLS\nOBS1")
    assert t.reusable_fraction == pytest.approx(17 / 22, rel=1e-3)


def test_a_rewritten_header_destroys_everything_after_it():
    """The architecture failure this exists to catch.

    One variable byte near the front makes every later token unreusable, no
    matter how much of the content is unchanged.
    """
    # Realistic scale on purpose: the bands are fractions of the prompt, and a
    # 24-character fixture cannot exercise a "leading 10 percent" band at all.
    body = "SYSTEM LAWS\n" + ("TOOLS AND OBSERVATIONS\n" * 200)
    probe = PrefixProbe()
    probe.observe("g", "TIME=1\n" + body)
    t = probe.observe("g", "TIME=2\n" + body + "OBS_NEW\n")
    assert t.longest_common_prefix_chars == len("TIME="), "one byte, everything after it lost"
    assert t.reusable_fraction < 0.01
    assert "leading_10_percent" in t.reason_for_prefix_divergence


def test_a_shrinking_prompt_is_named_as_such():
    probe = PrefixProbe()
    probe.observe("g", "SYSTEM\nOBS1\nOBS2\nOBS3")
    t = probe.observe("g", "SYSTEM\nOBS3")
    assert t.reason_for_prefix_divergence == "prompt_shrank_content_dropped_or_compacted"


def test_reusable_and_realized_are_different_numbers():
    """A gap between them is a resident problem, not an architecture one."""
    probe = PrefixProbe()
    probe.observe("g", "A" * 100)
    t = probe.observe(
        "g", "A" * 100 + "B" * 100, prompt_tokens=200, prefix_reused_tokens=0
    )
    assert t.reusable_fraction == pytest.approx(0.5)
    assert t.realized_reuse_fraction == pytest.approx(0.0)


def test_unmeasured_realized_reuse_is_not_reported_as_zero():
    """`None` and `0.0` mean opposite things and must not be conflated."""
    probe = PrefixProbe()
    probe.observe("g", "A" * 10)
    probe.observe("g", "A" * 20, prompt_tokens=20)
    summary = probe.summary()
    assert summary["realized_reuse_fraction_mean"] is None
    assert "NOT zero" in summary["realized_note"]


def test_separate_conversations_are_never_compared():
    probe = PrefixProbe()
    probe.observe("goal-a", "AAAA")
    assert probe.observe("goal-b", "BBBB") is None, "a first turn is a first turn"


def test_longest_common_prefix_is_exact():
    assert longest_common_prefix("abc", "abd") == 2
    assert longest_common_prefix("", "abc") == 0
    assert longest_common_prefix("abc", "abc") == 3


def test_divergence_reason_covers_the_identical_case():
    assert divergence_reason("abc", "abc", 3) == "identical_prompt"


# --------------------------------------------------------------------------
# Prefill: what the SHAPE implicates
# --------------------------------------------------------------------------


def _trace(walls, gpus=None):
    n = len(walls)
    return {
        "wall_ns": walls,
        "gpu_ns": gpus if gpus is not None else [w * 0.9 for w in walls],
        "encode_ns": [10] * n,
        "submit_ns": [10] * n,
        "wait_ns": [0] * n,
        "dispatches": [40] * n,
    }


def test_a_rising_curve_implicates_attention():
    walls = [1000 + 20 * i for i in range(800)]  # per-step cost grows with position
    prof = bucket_profile(_trace(walls), prefill_steps=800)
    verdict = attribute(prof)
    assert verdict["shape"] == "RISING_WITH_POSITION"
    assert "attention" in verdict["implicates"]
    assert "TRACE_DISPATCH" in verdict["next_discriminator"]


def test_a_flat_curve_says_prefill_is_linear():
    walls = [1000] * 800
    verdict = attribute(bucket_profile(_trace(walls), prefill_steps=800))
    assert verdict["shape"] == "FLAT_WITH_POSITION"
    assert "LINEAR" in verdict["implicates"]


def test_host_bound_prefill_is_called_out_separately_from_shape():
    """encode+submit dominating means no kernel change helps."""
    n = 400
    trace = {
        "wall_ns": [1000] * n,
        "gpu_ns": [100] * n,
        "encode_ns": [450] * n,
        "submit_ns": [400] * n,
        "wait_ns": [0] * n,
        "dispatches": [40] * n,
    }
    verdict = attribute(bucket_profile(trace, prefill_steps=n))
    assert verdict["host_bound"] is True
    assert verdict["host_control_share_of_wall"] > 0.35
    assert "dispatches per step" in verdict["host_note"]


def test_a_gpu_bound_flat_curve_is_not_called_host_bound():
    """Negative control for the host verdict."""
    n = 400
    trace = {
        "wall_ns": [1000] * n,
        "gpu_ns": [900] * n,
        "encode_ns": [20] * n,
        "submit_ns": [20] * n,
        "wait_ns": [0] * n,
        "dispatches": [40] * n,
    }
    verdict = attribute(bucket_profile(trace, prefill_steps=n))
    assert verdict["host_bound"] is False


def test_the_profile_never_counts_decode_steps_as_prefill():
    """With KV reuse the trace is shorter than the prompt. Bound by steps."""
    walls = list(range(1, 101))
    prof = bucket_profile(_trace(walls), prefill_steps=40)
    assert prof["prefill_steps"] == 40
    assert prof["totals"]["wall_ns"] == sum(range(1, 41))


def test_attribution_states_what_it_cannot_settle():
    verdict = attribute(bucket_profile(_trace([1000] * 100), prefill_steps=100))
    assert "only a per-kernel trace" in verdict["cannot_settle"]


def _profile(steps, dispatches):
    return {
        "prefill_steps": steps,
        "totals": {
            "dispatches": dispatches,
            "wall_ns": 169347647376,
            "gpu_ns": 165731042702,
            "encode_ns": 1802064129,
            "submit_ns": 12737347,
        },
        "buckets": [
            {"wall_ns_mean": 24419275.0, "gpu_ns_mean": 23606531.0},
            {"wall_ns_mean": 53134069.0, "gpu_ns_mean": 52290791.0},
        ],
    }


def test_the_verdict_says_prompt_tokens_are_paid_for_at_decode_prices():
    """The structural reading the shape alone never gave.

    Measured on one live call: 4,467 prompt tokens stepped one at a time for
    2,590,860 dispatches, 190s of wall at 98% GPU. Nothing in the old verdict
    said the loop was the problem rather than a kernel.
    """
    v = attribute(_profile(4467, 2590860), layers=64)
    assert v["dispatches_per_step"] == 580.0
    assert v["dispatches_per_layer_per_step"] == 9.06
    assert "one at a time" in v["structural"]
    assert "decode prices" in v["structural"]


def test_a_tight_per_layer_count_says_BATCH_and_a_fat_one_says_FUSE():
    """The lever must discriminate, or it is a slogan and not a diagnosis."""
    tight = attribute(_profile(4467, 2590860), layers=64)
    assert "BATCH THE PREFILL" in tight["lever"]

    # Same dispatch total, a body with eight layers: 72.5 per layer is fat, and
    # fusing adjacent kernels pays before restructuring the loop.
    fat = attribute(_profile(4467, 2590860), layers=8)
    assert "CUT DISPATCHES PER LAYER" in fat["lever"]


def test_the_per_layer_ratio_is_withheld_when_the_body_did_not_report_layers():
    """580 is damning at 8 layers and unremarkable at 64. Do not guess."""
    v = attribute(_profile(4467, 2590860))
    assert v["dispatches_per_step"] == 580.0
    assert "dispatches_per_layer_per_step" not in v
    assert "lever" not in v
