"""IS THE CHARGED WIDTH THE ROW, OR SIXTY-FOUR LANES OF HARDWARE?

ACCELERATOR_THE_WIDTH_IS_THE_ROW located the charged unit at 64 lanes and named
its own remainder: at tpr=64 THE ROW, THE THREADS-PER-ROW AND A 2-SIMDGROUP PAIR
ARE THE SAME 64 LANES AND ARE NOT SEPARATED. Halving tpr separates them, because
at tpr=32 a row is 32 lanes -- exactly ONE simdgroup -- and a 64-lane hardware
width would span TWO ROWS sitting at unrelated base addresses.

  H_ROW    the unit is the lanes cooperating on one row, whatever their count.
           Then the first departure at tpr=32 (k=16, two runs per row) must show
           the same step tpr=64 showed at its own first departure (k=32).
  H_64LANE the unit is a fixed 64-lane hardware width. At tpr=32 that width
           always spans two disjoint rows, so the CONTROL already pays the step
           and k=16 is only a second departure -- worth a few points at most.

The two predicted bands do not overlap, which is the whole reason for the shape.

AND THE SHAPE ITSELF NEEDED CORRECTING FIRST: that receipt's `next` names
cols=32768 at tpr=32 as holding groups-per-lane fixed at four. It gives SIXTEEN.
The tests below pin the arithmetic so the wrong shape cannot come back.
"""
import pytest

import gravity_native as G

ROWS = 17408
REFERENCE = (16384, 64)  # the tpr=64 shape the previous block measured
PROBE = (8192, 32)       # the tpr=32 shape that holds per-lane work identical
SIMD = 32


def groups(cols):
    return cols // G.GROUP


def groups_per_lane(cols, tpr):
    return groups(cols) // tpr


def cnt(lane, cols, tpr):
    return (groups(cols) - lane + tpr - 1) // tpr


def blk(lane, k, cols, tpr):
    n = cnt(lane, cols, tpr)
    return [lane + ((i + lane // k) % n) * tpr for i in range(n)]


def strided(lane, cols, tpr):
    return list(range(lane, groups(cols), tpr))


def runs_over(lanes, k, cols, tpr):
    g = [blk(l, k, cols, tpr)[0] for l in lanes]
    return 1 + sum(1 for i in range(1, len(g)) if g[i] != g[i - 1] + 1)


# ------------------------------------------------- the shape correction


def test_the_SHAPE_THE_PREVIOUS_RECEIPT_NAMED_IS_THE_WRONG_ONE():
    """cols=32768 at tpr=32 gives SIXTEEN groups per lane, four times the
    reference's per-lane work -- it would have moved the very axis the sentence
    was written to hold fixed. cols=8192 is the shape that holds it."""
    assert groups_per_lane(32768, 32) == 16
    assert groups_per_lane(*PROBE) == 4
    assert groups_per_lane(*REFERENCE) == 4


def test_the_PER_LANE_WORK_IS_IDENTICAL_IN_BOTH_SHAPES():
    """This is what makes the two ratios comparable at all: same groups per lane,
    same elements per lane, only the lane count and the row width move."""
    for cols, tpr in (REFERENCE, PROBE):
        assert {cnt(l, cols, tpr) for l in range(tpr)} == {4}, (cols, tpr)
        assert 4 * G.GROUP == 256


def test_a_ROW_IS_ONE_SIMDGROUP_AT_TPR32_AND_TWO_AT_TPR64():
    """The fact that makes this experiment discriminating rather than a repeat."""
    assert PROBE[1] // SIMD == 1
    assert REFERENCE[1] // SIMD == 2


# ------------------------------------------------- the arms


@pytest.mark.parametrize("cols,tpr", [REFERENCE, PROBE])
@pytest.mark.parametrize("k", [1, 2, 4, 8, 16, 32, 64])
def test_the_ROW_RUN_COUNT_IS_TPR_OVER_K(cols, tpr, k):
    if k > tpr:
        # NOT a skip. This program's own rule is that a sweep must not record a
        # case it declined to run as if it had nothing to say: a block wider than
        # the row floors to zero for every lane, so it IS the control, and
        # asserting that is strictly more than skipping would have said.
        assert runs_over(range(tpr), k, cols, tpr) == 1
        assert all(blk(l, k, cols, tpr) == strided(l, cols, tpr) for l in range(tpr))
        return
    assert runs_over(range(tpr), k, cols, tpr) == tpr // k


def test_the_FIRST_DEPARTURE_IS_K_EQUALS_HALF_THE_ROW():
    """Two runs per row, the minimal non-identity order at each width -- and it
    is a DIFFERENT k at each, which is exactly what the two hypotheses disagree
    about."""
    assert runs_over(range(32), 16, *PROBE) == 2
    assert runs_over(range(64), 32, *REFERENCE) == 2


@pytest.mark.parametrize("k", [1, 2, 4, 8, 16, 32])
def test_the_per_lane_SET_is_identical_LANE_BY_LANE_at_tpr32(k):
    cols, tpr = PROBE
    for lane in range(tpr):
        assert sorted(blk(lane, k, cols, tpr)) == strided(lane, cols, tpr), lane


@pytest.mark.parametrize("k,expected", [(1, True), (2, True), (4, True),
                                        (8, True), (16, True), (32, False)])
def test_the_ORDER_differs_below_the_row_width_and_NOT_at_it(k, expected):
    cols, tpr = PROBE
    differs = any(blk(l, k, cols, tpr) != strided(l, cols, tpr) for l in range(tpr))
    assert differs is expected


def test_BOTH_DIVISORS_FLOOR_TO_ZERO_AT_TPR32_SO_THE_CONTROL_IS_UNAMBIGUOUS():
    """rotblk32 and rotblk64 compute the SAME order at tpr=32 because lane/32 and
    lane/64 are both zero for every lane. Pinned because a control that silently
    became an arm would TIE, and the tie would read as a finding."""
    cols, tpr = PROBE
    assert all(blk(l, 32, cols, tpr) == blk(l, 64, cols, tpr) for l in range(tpr))
    assert all(blk(l, 32, cols, tpr) == strided(l, cols, tpr) for l in range(tpr))


def test_THE_COMPARISON_IS_NOT_DEGENERATE():
    """Anti-vacuity for the sweep: the control and the first-departure arm must
    differ in run count, or the ratio measures nothing."""
    assert runs_over(range(32), 32, *PROBE) == 1
    assert runs_over(range(32), 16, *PROBE) == 2


# ------------------------------------------------- the emitted source


@pytest.mark.parametrize("k", [1, 2, 4, 8, 16, 32])
def test_THE_EMITTED_DIVISOR_IS_THE_REQUESTED_ONE(k):
    """Carried from the previous block, where this caught what nothing else
    could: every rotation is a BIJECTION, so a kernel emitting the wrong divisor
    still computes the right answer and no oracle can see it."""
    src = G.source_operand_probe(ROWS, PROBE[0], f"rotblk{k}", PROBE[1], 128)
    assert f"(lane / {k}u)" in src


def test_a_THREADGROUP_THAT_IS_NOT_A_WHOLE_NUMBER_OF_ROWS_IS_REFUSED_AT_TPR32():
    with pytest.raises(ValueError, match="whole number of rows"):
        G.source_operand_probe(ROWS, PROBE[0], "rotblk16", PROBE[1], 48)


def test_DRIFT_the_probe_raises_if_the_loop_it_rewrites_is_gone(monkeypatch):
    monkeypatch.setattr(G, "NATIVE_MATVEC_TPR",
                        G.NATIVE_MATVEC_TPR.replace("for (uint g = lane;", "for (uint g = 0u;"))
    with pytest.raises(AssertionError):
        G.source_operand_probe(ROWS, PROBE[0], "rotblk16", PROBE[1], 128)


# ------------------------------------------------- and it EXECUTES


def test_THE_ARMS_AGREE_WITH_A_FLOAT64_ORACLE_AT_TPR32():
    """The generated Metal, not this file's Python. A reordering must agree with
    the oracle and with itself, and the control -- summing in the shipped
    association order -- is asserted to be no worse than the reordered arm."""
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    rows, cols, tpr = 256, 4096, 32
    rng = np.random.default_rng(11)
    w = (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)
    x = rng.standard_normal(cols).astype(np.float32)
    packed, scale = G.pack_q4_g64(w)
    oracle = G.dequantize(packed, scale, cols).astype(np.float64) @ x.astype(np.float64)
    n = np.linalg.norm(oracle)
    dp, ds, dx = mx.array(packed), mx.array(scale), mx.array(x)
    err = {}
    for probe in ("rotblk32", "rotblk16"):
        src = G.source_operand_probe(rows, cols, probe, tpr, 128)
        k = mx.fast.metal_kernel(name=f"riu_{probe}", input_names=["packed", "scales", "x"],
                                 output_names=["out"], source=src, ensure_row_contiguous=True)
        (o,) = k(inputs=[dp, ds, dx], grid=(rows * tpr, 1, 1), threadgroup=(128, 1, 1),
                 output_shapes=[(rows,)], output_dtypes=[mx.float32])
        v = np.array(o, dtype=np.float64)
        err[probe] = float(np.linalg.norm(v - oracle) / n)
        assert err[probe] < 1e-5, (probe, err[probe])
        assert len(np.unique(v)) > rows // 2, "degenerate output"
    # THE OBVIOUS ASSERTION HERE WOULD REST ON AN ACCIDENT AND IS NOT MADE.
    # ACCELERATOR_THE_WIDTH_IS_THE_ROW asserted the control must read the LOWEST
    # rel_err "since it sums in the shipped association order" -- that reasoning
    # was never sound, because one association order has no claim to less
    # rounding error than another, and at tpr=32 it is measurably false: the
    # sweep's control reads 2.31162e-07 against k1's 2.29271e-07 in both runs,
    # and across four shapes the winner changes. What IS a property of a
    # reordering, rather than of a shape, is that the sum reassociates -- so the
    # arms must DIFFER, which is independent evidence the reorder landed at all.
    assert err["rotblk32"] != err["rotblk16"], (
        "the two arms returned bit-identical error, so the reorder did not land")
