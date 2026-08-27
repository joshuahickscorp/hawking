"""Varying the per-simdgroup span WITHOUT a permutation.

ACCELERATOR_THE_SIMDGROUP_SPAN_DOUBLES left SPAN as the surviving candidate for
the lane-order cost and said in its own words that it could not test it, because
"a permutation cannot" vary the span independently. A BLOCKED PARTITION can:
it gives every lane a contiguous run of exactly the count the strided loop gives
it, so the per-lane element count is identical lane by lane while the addresses
move. These pin what the arms hold fixed and what they move.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gravity_native as G  # noqa: E402

ROWS, COLS, TPR, SIMD = 17408, 5120, 64, 32
GROUPS = COLS // G.GROUP
Q, REM = GROUPS // TPR, GROUPS % TPR


def _cnt(lane):
    return (GROUPS - lane + TPR - 1) // TPR


def _off(lane):
    return (Q + 1) * lane if lane < REM else (Q + 1) * REM + Q * (lane - REM)


def _blocked(lane):
    return list(range(_off(lane), _off(lane) + _cnt(lane)))


def _strided(lane):
    return list(range(lane, GROUPS, TPR))


def test_blocked_is_an_EXACT_COVER():
    # Anti-vacuity, first half: a partition that dropped or repeated a group
    # would be WRONG, and this arm's whole point is that it is correct.
    assert sorted(g for l in range(TPR) for g in _blocked(l)) == list(range(GROUPS))


def test_the_per_lane_COUNT_is_identical_lane_by_lane():
    # THE SINGLE-VARIABLE CONDITION. Not "the same multiset" -- the same count
    # for the same lane, so no lane does more or less work than it did.
    assert [len(_blocked(l)) for l in range(TPR)] == [len(_strided(l)) for l in range(TPR)]


def test_blocked_is_NOT_the_strided_assignment():
    # Anti-vacuity, second half: an assignment that silently reduced to the
    # shipped one would TIE and the tie would read as a finding.
    assert any(_blocked(l) != _strided(l) for l in range(TPR))


def test_the_span_really_does_shrink_and_the_totals_do_not():
    # What the arm was built to move, and what it was built to hold.
    def summarise(f, sg):
        g = sorted({x for l in range(sg * SIMD, (sg + 1) * SIMD) for x in f(l)})
        span = (max(g) - min(g) + 1) * (G.GROUP // 2)
        frags = 1 + sum(1 for i in range(len(g) - 1) if g[i + 1] - g[i] > 1)
        return len(g), span, frags
    assert summarise(_strided, 0) == (48, 2560, 2)
    assert summarise(_blocked, 0) == (48, 1536, 1)          # 40% narrower, defragmented
    assert summarise(_strided, 1) == summarise(_blocked, 1) == (32, 1024, 1)


def test_only_the_SHIPPED_assignment_coalesces_per_ITERATION():
    # THE DISCRIMINATOR, and the reason the span result came out the way it did.
    # Span is a property of the addresses a simdgroup touches OVER THE WHOLE LOOP;
    # this is what it asks for AT ONE INSTANT, which is what a coalescer sees.
    def runs_at(f, k):
        g = sorted(x[k] for l in range(SIMD) if len(x := f(l)) > k)
        return 1 + sum(1 for i in range(len(g) - 1) if g[i + 1] - g[i] > 1)
    assert runs_at(_strided, 0) == 1                        # one contiguous request
    assert runs_at(_blocked, 0) > 10                        # shattered into runs
    assert runs_at(lambda l: [(l * 37) % TPR + k * TPR
                              for k in range(_cnt((l * 37) % TPR))], 0) > 10


def test_THE_TWO_AXES_ARE_IN_TENSION_which_is_why_this_is_not_a_clean_sweep():
    # Blocked IMPROVED the span and DESTROYED per-iteration coalescing. No
    # assignment maximises both, so they are not independent axes and this block
    # traded one for the other rather than isolating either.
    def runs_at(f, k):
        g = sorted(x[k] for l in range(SIMD) if len(x := f(l)) > k)
        return 1 + sum(1 for i in range(len(g) - 1) if g[i + 1] - g[i] > 1)
    strided_span = max(_strided(l)[0] for l in range(SIMD)) - min(_strided(l)[0] for l in range(SIMD))
    blocked_span = max(_blocked(l)[0] for l in range(SIMD)) - min(_blocked(l)[0] for l in range(SIMD))
    assert blocked_span > strided_span      # blocked spreads iteration 0 wider
    assert runs_at(_blocked, 0) > runs_at(_strided, 0)


@pytest.mark.parametrize("probe", ["blocked", "blockedctl"])
def test_the_probe_sources_differ_from_the_baseline_and_each_other(probe):
    base = G.source_tpr(ROWS, COLS, TPR, TG := 128)
    src = G.source_operand_probe(ROWS, COLS, probe, TPR, TG)
    assert src != base
    assert src != G.source_operand_probe(
        ROWS, COLS, "blockedctl" if probe == "blocked" else "blocked", TPR, TG)


def test_the_CONTROL_keeps_the_shipped_loop():
    # blockedctl exists to pay blocked's arithmetic at the shipped addresses. If
    # it ever stopped carrying the shipped loop it would no longer control for it.
    d = {"GROUPS": GROUPS, "TPR": TPR}
    ctl = G.source_operand_probe(ROWS, COLS, "blockedctl", TPR, 128)
    assert "for (uint g = lane; g < %(GROUPS)du; g += %(TPR)du) {" % d in ctl
    # IT MUST CONSUME THEM, NOT MERELY DECLARE THEM. Asserting `"_off" in ctl`
    # passed with the zero-multiply deleted -- the declaration survives, and a
    # declaration the compiler is free to fold away is a control that has stopped
    # paying the arithmetic it exists to pay, silently turning the comparison back
    # into addresses PLUS arithmetic. Whether it actually folds is a question about
    # GENERATED CODE and xcrun metal is ABSENT, so the source-level consume is the
    # strongest check available here and is asserted rather than assumed.
    assert "acc += 0.0f * (float)(_off + _cnt);" in ctl
    assert "for (uint g = _off;" not in ctl


def test_a_drifted_template_RAISES_rather_than_returning_the_baseline():
    # Patch what source_tpr actually reads. Patching NATIVE_MATVEC_TPR_BODY does
    # NOT work -- source_tpr builds from the pre-concatenated NATIVE_MATVEC_TPR --
    # and a drift test aimed at the wrong constant passes while proving nothing,
    # which is the mutation-that-never-landed shape three times over in this session.
    original = G.NATIVE_MATVEC_TPR
    try:
        G.NATIVE_MATVEC_TPR = original.replace(
            "for (uint g = lane; g < %(GROUPS)du; g += %(TPR)du) {", "for (uint g = lane;;) {")
        with pytest.raises(AssertionError):
            G.source_operand_probe(ROWS, COLS, "blocked", TPR, 128)
    finally:
        G.NATIVE_MATVEC_TPR = original


def test_the_kernel_ACTUALLY_COMPUTES_the_blocked_assignment():
    """The tests above verify a PYTHON REIMPLEMENTATION of the offset formula.

    That is not the kernel. A mutation that changed the emitted _off expression
    to `lane` left all of them passing -- the suite was pinning arithmetic it
    owned rather than arithmetic it shipped, which is this program's
    check-that-cannot-fail in a new place, found by watching the mutation survive.
    A wrong offset stops the assignment being an exact cover, so RUNNING it is
    what ties the formula to the thing that executes.
    """
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    rows, cols, tpr, tg = 8, COLS, TPR, 128
    rng = np.random.default_rng(30)
    packed = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
    scale = rng.random((rows, cols // G.GROUP)).astype(np.float16) + 0.5
    x = rng.standard_normal(cols).astype(np.float32)
    lo, hi = (packed & 0x0F).astype(np.int32), (packed >> 4).astype(np.int32)
    w = np.empty((rows, cols), dtype=np.float64)
    w[:, 0::2], w[:, 1::2] = lo - G.BOUND, hi - G.BOUND
    w *= np.repeat(scale.astype(np.float64), G.GROUP, axis=1)
    oracle = w @ x.astype(np.float64)

    dp, ds, dx = mx.array(packed), mx.array(scale), mx.array(x)
    for probe in ("blocked", "blockedctl"):
        k = mx.fast.metal_kernel(
            name=f"blk_{probe}", input_names=["packed", "scales", "x"],
            output_names=["out"], source=G.source_operand_probe(rows, cols, probe, tpr, tg),
            ensure_row_contiguous=True)
        (o,) = k(inputs=[dp, ds, dx], grid=(rows * tpr, 1, 1), threadgroup=(tg, 1, 1),
                 output_shapes=[(rows,)], output_dtypes=[mx.float32])
        got = np.array(o, dtype=np.float64)
        rel = np.linalg.norm(got - oracle) / np.linalg.norm(oracle)
        assert rel < 1e-5, (probe, rel)
