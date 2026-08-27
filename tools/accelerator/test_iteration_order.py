"""Per-iteration request order, with every loop-wide property pinned.

Twelve axes have died against this kernel's floor. This is the first probe in
the family where the per-lane group SET, the per-simdgroup loop-wide set, its
SPAN, its FRAGMENT COUNT and every stride in it are ALL identical between the
arms -- a blocked partition moved the set and a permutation moved the
simdgroup's span, so neither could isolate the ordering.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gravity_native as G  # noqa: E402

ROWS, TPR, TG, SIMD = 17408, 64, 128, 32


def _cnt(lane, groups):
    return (groups - lane + TPR - 1) // TPR


def _rot(lane, groups):
    n = _cnt(lane, groups)
    return [lane + ((i + lane) % n) * TPR for i in range(n)]


def _shipped(lane, groups):
    return list(range(lane, groups, TPR))


@pytest.mark.parametrize("groups", [80, 256])
def test_the_per_lane_SET_is_identical_lane_by_lane(groups):
    # THE SINGLE-VARIABLE CONDITION, and it is stronger than the blocked probe's:
    # not the same count, the same GROUPS, so nothing about which addresses a
    # lane touches has moved at all.
    for lane in range(TPR):
        assert sorted(_rot(lane, groups)) == _shipped(lane, groups), lane


@pytest.mark.parametrize("groups", [80, 256])
def test_the_ORDER_differs(groups):
    # Anti-vacuity: a rotation that silently reduced to the shipped order would
    # TIE, and the tie would read as a finding.
    assert any(_rot(l, groups) != _shipped(l, groups) for l in range(TPR))


@pytest.mark.parametrize("groups", [80, 256])
def test_SPAN_AND_FRAGMENTS_ARE_PINNED_which_is_what_the_earlier_probes_could_not_do(groups):
    # The properties ACCELERATOR_THE_SIMDGROUP_SPAN_DOUBLES and
    # ACCELERATOR_SPAN_IS_REFUTED_COALESCING_IS_NOT excluded, held EXACTLY here
    # rather than approximately, because they follow from the per-lane sets.
    def loopwide(f):
        g = sorted({x for l in range(SIMD) for x in f(l, groups)})
        span = (max(g) - min(g) + 1) * (G.GROUP // 2)
        frags = 1 + sum(1 for i in range(len(g) - 1) if g[i + 1] - g[i] > 1)
        return len(g), span, frags
    assert loopwide(_rot) == loopwide(_shipped)


@pytest.mark.parametrize("groups,expect", [(80, 17), (256, 32)])
def test_only_the_PER_ITERATION_request_moves(groups, expect):
    # THE VARIABLE. What a coalescer sees at one instant, which is the one thing
    # a loop-wide description cannot express.
    def runs_at(f, k):
        g = sorted(x[k] for l in range(SIMD) if len(x := f(l, groups)) > k)
        return 1 + sum(1 for i in range(len(g) - 1) if g[i + 1] - g[i] > 1)
    assert runs_at(_shipped, 0) == 1                  # one contiguous request
    assert runs_at(_rot, 0) == expect                 # shattered, and by how much


def test_MORE_GROUPS_PER_LANE_SHATTERS_HARDER_which_is_the_second_prediction():
    # P2's arithmetic: the ladder exists before any timing, so a measured
    # increase can be checked against a structural one rather than asserted.
    def runs_at(groups):
        g = sorted(_rot(l, groups)[0] for l in range(SIMD))
        return 1 + sum(1 for i in range(len(g) - 1) if g[i + 1] - g[i] > 1)
    assert runs_at(256) > runs_at(80) > 1


def test_the_CONTROL_keeps_the_shipped_order_and_pays_the_arithmetic():
    ctl = G.source_operand_probe(ROWS, 5120, "rot0", TPR, TG)
    rot = G.source_operand_probe(ROWS, 5120, "rotlane", TPR, TG)
    assert "((_i + 0u) % _n)" in ctl                   # rotation offset ZERO
    assert "((_i + lane) % _n)" in rot
    assert "uint _n = (80u - lane + 64u - 1u) / 64u;" in ctl   # same runtime modulo
    assert ctl != rot


def test_an_unknown_rot_offset_is_REFUSED():
    with pytest.raises(ValueError, match="rot0"):
        G.source_operand_probe(ROWS, 5120, "rot7", TPR, TG)


def test_a_drifted_template_RAISES_rather_than_returning_the_baseline():
    original = G.NATIVE_MATVEC_TPR
    try:
        G.NATIVE_MATVEC_TPR = original.replace(
            "for (uint g = lane; g < %(GROUPS)du; g += %(TPR)du) {", "for (uint g = lane;;) {")
        with pytest.raises(AssertionError):
            G.source_operand_probe(ROWS, 5120, "rotlane", TPR, TG)
    finally:
        G.NATIVE_MATVEC_TPR = original


@pytest.mark.parametrize("probe", ["rotlane", "rot0"])
def test_the_kernel_ACTUALLY_REORDERS_and_stays_correct(probe):
    """Both arms MUST be correct -- a reordering permutes the same terms.

    The suite otherwise pins a PYTHON reimplementation of the index formula, a
    mistake the previous block watched a mutation survive, so this runs the
    generated Metal: a wrong index breaks the cover and the answer moves far
    more than a reassociation does.
    """
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    rows, cols = 8, 5120
    rng = np.random.default_rng(31)
    packed = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
    scale = rng.random((rows, cols // G.GROUP)).astype(np.float16) + 0.5
    x = rng.standard_normal(cols).astype(np.float32)
    lo, hi = (packed & 0x0F).astype(np.int32), (packed >> 4).astype(np.int32)
    w = np.empty((rows, cols), dtype=np.float64)
    w[:, 0::2], w[:, 1::2] = lo - G.BOUND, hi - G.BOUND
    w *= np.repeat(scale.astype(np.float64), G.GROUP, axis=1)
    oracle = w @ x.astype(np.float64)
    k = mx.fast.metal_kernel(
        name=f"rot_{probe}", input_names=["packed", "scales", "x"], output_names=["out"],
        source=G.source_operand_probe(rows, cols, probe, TPR, TG), ensure_row_contiguous=True)
    (o,) = k(inputs=[mx.array(packed), mx.array(scale), mx.array(x)],
             grid=(rows * TPR, 1, 1), threadgroup=(TG, 1, 1),
             output_shapes=[(rows,)], output_dtypes=[mx.float32])
    got = np.array(o, dtype=np.float64)
    assert np.linalg.norm(got - oracle) / np.linalg.norm(oracle) < 1e-5
