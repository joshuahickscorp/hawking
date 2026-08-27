"""The per-simdgroup footprint of a lane permutation.

ACCELERATOR_LANE_ORDER_COSTS claimed a permutation leaves "the set of addresses
the simdgroup touches at each instant exactly identical". That is true over 64
lanes and FALSE over a 32-lane simdgroup, which is the coalescing unit. These
pin the arithmetic so the corrected claim cannot drift back.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gravity_native as G  # noqa: E402

TPR, SIMD = 64, 32


def _groups(mult, lanes):
    return sorted((l * mult) % TPR for l in range(lanes))


def _span_and_fragments(g):
    span = (max(g) - min(g) + 1) * (G.GROUP // 2)   # a group is GROUP/2 packed bytes
    frags = 1 + sum(1 for i in range(len(g) - 1) if g[i + 1] - g[i] > 1)
    return span, frags


def test_the_whole_threadgroup_footprint_IS_identical():
    # The half of the earlier claim that is TRUE, kept so the correction cannot
    # over-generalise into "a permutation changes the footprint" full stop.
    for mult in (1, 3, 33, 37, 63):
        assert _groups(mult, TPR) == list(range(TPR))


def test_the_SIMDGROUP_footprint_is_NOT_identical():
    # The half that is FALSE, and the reason the earlier receipt's mechanism
    # reasoning did not hold: only the identity keeps a simdgroup contiguous.
    identity_span, identity_frags = _span_and_fragments(_groups(1, SIMD))
    assert (identity_span, identity_frags) == (1024, 1)
    for mult in (3, 33, 37, 63):
        span, _ = _span_and_fragments(_groups(mult, SIMD))
        assert span == 2048, (mult, span)


def test_fragmentation_and_span_are_SEPARATE_axes():
    # THE DISCRIMINATOR. M=63 doubles the span with only 2 fragments while M=33
    # doubles it with 32 -- a 16x range in fragmentation at one span. If the two
    # moved together the sweep could not tell them apart, and a test asserting
    # only the span would hide that.
    frags = {m: _span_and_fragments(_groups(m, SIMD))[1] for m in (3, 33, 37, 63)}
    assert frags[63] == 2 and frags[33] == 32
    assert max(frags.values()) >= 10 * min(frags.values())


@pytest.mark.parametrize("mult", [3, 33, 37, 63])
def test_every_multiplier_here_is_a_real_permutation(mult):
    assert _groups(mult, TPR) == list(range(TPR))
    assert [(l * mult) % TPR for l in range(TPR)] != list(range(TPR))


def test_mean_circular_displacement_CANNOT_BE_VARIED_this_way():
    # Pinned because it refuted a pre-registered question before it was run: the
    # displacement-vs-disorder framing is ill-posed for multiplicative maps, and
    # a sweep reporting "no displacement trend" would have been a check whose
    # answer was fixed in advance.
    def disp(m):
        return sum(min((l * m - l) % TPR, (l - l * m) % TPR) for l in range(TPR)) / TPR
    for mult in (3, 17, 33, 37, 63):
        assert disp(mult) == pytest.approx(16.0), mult
    assert disp(1) == 0.0
