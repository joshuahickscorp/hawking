"""The correction must be exact, and must not be believable by coincidence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import per_generated_token_inflation as inf


def test_factor_is_passes_over_generated():
    assert inf.FORWARD_PASSES == inf.PROMPT_TOKENS + inf.DECODE_STEPS == 139
    assert abs(inf.factor() - 139 / 128) < 1e-12


def test_dispatch_check_is_exact_to_the_last_digit():
    """964 x 139/128 reproduces the reported 1046.84375 with no residual."""
    assert inf.TRUE_DISPATCHES_UNFUSED * inf.factor() == inf.REPORTED_DISPATCHES
    assert abs(inf.correct(inf.REPORTED_DISPATCHES) - 964.0) < 1e-9


def test_byte_check_agrees_with_an_independent_census():
    """The catalog sum was derived without reference to the resident."""
    c = next(x for x in inf.checks() if x["field"] == "active_bytes_per_token")
    assert abs(c["residual_ppm"]) < 20.0, c["residual_ppm"]
    # Not exact: ~69 KB separates what the resident loads from what the catalog
    # lists, which is a real small delta and not rounding. It must stay visible
    # rather than be asserted away.
    delta = c["corrected"] - inf.CATALOG_ACTIVE_BYTES_PER_PASS
    assert 0 < abs(delta) < 100_000, delta
    assert c["residual_bytes"] == delta


def test_correction_restores_the_71_roof():
    a = inf.corrected_anchors()
    assert 71.0 < a["roof_tps_at_clean_gemv_703_5"] < 71.5
    # And the inflated figure would have given a materially wrong roof.
    wrong = 703.5 / (inf.REPORTED_ACTIVE_BYTES / 1e9)
    assert wrong < 66.0, "the inflated anchor must be visibly worse, not close"


def test_the_reversal_is_recorded_not_quietly_dropped():
    d = inf.build()["what_this_reverses"]
    assert "65.58" in d["claim"]
    assert d["made_in"] and d["what_caught_it"]


def test_decode_block_is_named_as_the_safe_field():
    doc = inf.build()
    assert any("metrics.decode" in f for f in doc["unaffected_fields"])
    assert "active_bytes_per_token" in doc["affected_fields"]
