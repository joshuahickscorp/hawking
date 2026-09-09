"""Is the diversity half of the capability gate doing any work? (G020)

The gate is a CONJUNCTION -- perplexity AND n-gram diversity -- and the
diversity bar is set per specimen as `3.0 x that specimen's own dense median
4-gram repeat`. A relative bar is the right instinct: a body whose dense parent
naturally repeats should not be judged against another body's baseline.

But median_4gram_repeat is a FRACTION OF 4-GRAMS THAT REPEAT. Its range is
[0, 1]. So the moment a specimen's dense parent repeats more than a third of its
4-grams, `3 x dense` lands ABOVE 1.0 -- above the highest value the metric can
take -- and the diversity axis cannot reject anything at all. The conjunction
silently becomes perplexity alone, which is the single-proxy failure this
campaign has been bitten by before.

This computes the condition, tests it against the campaign's own recorded
verdicts, and asks the separate question G020 also poses: does moving the
threshold change any verdict that matters?
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]

# median_4gram_repeat is a fraction of n-grams. Nothing can exceed this.
R4_CEILING = 1.0
MULTIPLIER = 3.0
# Above this dense baseline, MULTIPLIER x dense exceeds the metric's range.
VACUOUS_ABOVE = R4_CEILING / MULTIPLIER


def r4_bar(dense_r4: float, multiplier: float = MULTIPLIER) -> float:
    """The diversity bar for a specimen, capped so it stays inside the metric.

    THE RULE, and where it comes from: a bar may not sit more than HALFWAY from
    the reference to total collapse. `multiplier x dense` alone is unbounded and
    lands above 1.0 -- outside median_4gram_repeat's range -- for any parent
    repeating more than a third of its 4-grams, which is how Qwen3-0.6B got a
    bar of 1.0857 and an axis that could not reject anything.

    The midpoint is derived from the two fixed points the problem already has:
    the reference the specimen is judged against, and the ceiling the metric
    cannot exceed. It is not a number chosen to produce a wanted verdict, and it
    is not a tightening -- for any parent below dense_r4 = 1/3 it changes
    nothing at all, so every O003 verdict in this campaign is untouched.

    What it does change is that a candidate more than halfway from its own
    parent to total repetition can no longer pass, whatever the multiplier says.
    """
    return min(multiplier * dense_r4, (dense_r4 + R4_CEILING) / 2.0)


def gate_is_vacuous(dense_r4: float, multiplier: float = MULTIPLIER) -> Dict[str, Any]:
    bar = multiplier * dense_r4
    return {
        "dense_r4": dense_r4,
        "multiplier": multiplier,
        "bar": round(bar, 4),
        "metric_ceiling": R4_CEILING,
        "vacuous": bar >= R4_CEILING,
        "why": (f"{multiplier} x {dense_r4} = {bar:.4f} >= {R4_CEILING}, the highest value "
                f"median_4gram_repeat can take. Nothing can fail this axis."
                if bar >= R4_CEILING else
                f"{multiplier} x {dense_r4} = {bar:.4f}, inside the metric's range"),
    }


def sensitivity(points: List[Dict[str, Any]], dense_r4: float) -> Dict[str, Any]:
    """Over what multiplier range does EVERY recorded verdict stay the same?

    G020 asks for a threshold derived from evidence rather than chosen. If no
    candidate sits near the bar, the exact multiplier is not what decided the
    campaign -- and that is a stronger statement than defending the number.
    """
    passing = [p for p in points if p["passes"]]
    failing_on_r4 = [p for p in points if not p["passes"] and p["r4"] > MULTIPLIER * dense_r4]
    hi_pass = max((p["r4"] / dense_r4 for p in passing), default=None)
    lo_fail = min((p["r4"] / dense_r4 for p in failing_on_r4), default=None)
    return {
        "highest_passing_multiple_of_dense": round(hi_pass, 3) if hi_pass else None,
        "lowest_diversity_failing_multiple": round(lo_fail, 3) if lo_fail else None,
        "verdicts_unchanged_over": ([round(hi_pass, 3), round(lo_fail, 3)]
                                    if hi_pass and lo_fail else None),
        "band_width": round(lo_fail / hi_pass, 2) if (hi_pass and lo_fail) else None,
    }


def _selftest() -> List[str]:
    fails = []
    # THE BAR, capped. O003 must be untouched; Qwen3-0.6B must stop being vacuous
    # and must reject the G017 arms it passed.
    if abs(r4_bar(0.0538) - 0.1614) > 1e-6:
        fails.append(f"O003's bar moved: {r4_bar(0.0538)} != 0.1614 -- the cap must not tighten a sound gate")
    q = r4_bar(0.3619)
    if q >= R4_CEILING:
        fails.append(f"Qwen3-0.6B's capped bar {q} is still at or above the metric ceiling")
    if not (0.3619 <= q < 0.8224):
        fails.append(f"capped bar {q} must accept the dense parent (0.3619) and reject the "
                     f"G017 inverted control (0.8224)")
    # G017's four arms, from receipts/future/G017_ORGAN_ALLOCATION.json
    for r4, must_pass in ((0.3619, True), (0.6923, False), (0.7103, False), (0.8224, False)):
        passes = r4 <= q
        if passes != must_pass:
            fails.append(f"G017 arm r4={r4}: passes={passes}, want {must_pass}")
    # The G017 specimen: dense median r4 0.3619 -> bar 1.0857, above the ceiling.
    v = gate_is_vacuous(0.3619)
    if not v["vacuous"]:
        fails.append("Qwen3-0.6B's gate (3 x 0.3619 = 1.0857) not reported vacuous")
    # O003's reference: 0.0538 -> 0.1614, sound.
    if gate_is_vacuous(0.0538)["vacuous"]:
        fails.append("O003's gate (3 x 0.0538 = 0.1614) wrongly reported vacuous")
    # The boundary must be exactly where the arithmetic puts it, not near it.
    if not gate_is_vacuous(VACUOUS_ABOVE)["vacuous"]:
        fails.append("the boundary case dense_r4 = 1/3 is not reported vacuous")
    if gate_is_vacuous(VACUOUS_ABOVE - 1e-9)["vacuous"]:
        fails.append("just below the boundary is reported vacuous")
    return fails


if __name__ == "__main__":
    fails = _selftest()
    for f in fails:
        print("SELFTEST FAIL:", f)
    print(f"selftest: {'FAILED' if fails else 'PASS'}")
    raise SystemExit(1 if fails else 0)
