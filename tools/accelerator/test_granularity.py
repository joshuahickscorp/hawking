"""The granularity of the shattering, at a fixed address set.

ACCELERATOR_THE_INSTANT_IS_THE_COST confirmed that the per-instant request
pattern costs and named its remainder: the axis separating a coalescer width
from a transaction count is the GRANULARITY. Rotating by BLOCKS of k adjacent
lanes sweeps the run count geometrically while the per-lane group set stays
invariant, and k=32 is the discriminator -- a NON-IDENTITY order whose every
simdgroup iteration is still one contiguous run.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gravity_native as G  # noqa: E402

ROWS, COLS, TPR, TG, SIMD = 17408, 16384, 64, 128, 32
GROUPS = COLS // G.GROUP
N = GROUPS // TPR
KS = (1, 2, 4, 8, 16, 32, 64)


def _blk(lane, k):
    return [lane + ((i + lane // k) % N) * TPR for i in range(N)]


def _shipped(lane):
    return list(range(lane, GROUPS, TPR))


@pytest.mark.parametrize("k", KS)
def test_the_per_lane_SET_is_invariant_at_every_k(k):
    # THE SINGLE-VARIABLE CONDITION, asserted at every rung rather than at one.
    for lane in range(TPR):
        assert sorted(_blk(lane, k)) == _shipped(lane), (k, lane)


@pytest.mark.parametrize("k", KS)
def test_the_order_differs_EXCEPT_at_the_control(k):
    # Anti-vacuity in both directions: every rung must reorder something, and
    # rotblk64 must NOT -- lane/64 is 0 for every lane, so the control computes
    # the shipped order while paying the identical division and modulo. An arm
    # that silently reduced to the control would tie and read as a finding.
    assert any(_blk(l, k) != _shipped(l) for l in range(TPR)) == (k < 64)


def test_the_LADDER_is_geometric_and_computed_before_any_timing():
    def runs(k, width):
        g = sorted(l + ((l // k) % N) * TPR for l in range(width))
        return 1 + sum(1 for i in range(len(g) - 1) if g[i + 1] - g[i] > 1)
    assert [runs(k, SIMD) for k in KS] == [32, 16, 8, 4, 2, 1, 1]
    assert [runs(k, TPR) for k in KS] == [64, 32, 16, 8, 4, 2, 1]


def test_k32_IS_THE_DISCRIMINATOR_perfect_simdgroup_contiguity_off_the_shipped_order():
    """The arm the block turns on: it must be BOTH perfectly coalesced per
    simdgroup AND a different order, or it separates nothing."""
    def runs(k, width):
        g = sorted(l + ((l // k) % N) * TPR for l in range(width))
        return 1 + sum(1 for i in range(len(g) - 1) if g[i + 1] - g[i] > 1)
    assert runs(32, SIMD) == 1                      # every simdgroup request contiguous
    assert runs(32, TPR) == 2                       # the THREADGROUP's is not
    assert any(_blk(l, 32) != _shipped(l) for l in range(TPR))
    # and it is simdgroup 1 alone that moves -- simdgroup 0 is bit-for-bit the
    # shipped order, which is what makes the arm a minimal departure.
    assert all(_blk(l, 32) == _shipped(l) for l in range(SIMD))
    assert all(_blk(l, 32) != _shipped(l) for l in range(SIMD, TPR))


def test_a_NON_POWER_OF_TWO_block_is_REFUSED():
    # Lanes must divide evenly into blocks or the run count is not 32/k and the
    # ladder's own arithmetic stops describing the arm.
    with pytest.raises(ValueError, match="POWER OF TWO"):
        G.source_operand_probe(ROWS, COLS, "rotblk3", TPR, TG)


@pytest.mark.parametrize("k", KS)
def test_the_emitted_source_carries_the_block_divisor(k):
    src = G.source_operand_probe(ROWS, COLS, f"rotblk{k}", TPR, TG)
    assert f"((_i + (lane / {k}u)) % _n)" in src
    assert src != G.source_tpr(ROWS, COLS, TPR, TG)


def test_a_drifted_template_RAISES_rather_than_returning_the_baseline():
    original = G.NATIVE_MATVEC_TPR
    try:
        G.NATIVE_MATVEC_TPR = original.replace(
            "for (uint g = lane; g < %(GROUPS)du; g += %(TPR)du) {", "for (uint g = lane;;) {")
        with pytest.raises(AssertionError):
            G.source_operand_probe(ROWS, COLS, "rotblk8", TPR, TG)
    finally:
        G.NATIVE_MATVEC_TPR = original


@pytest.mark.parametrize("k", [1, 32, 64])
def test_the_kernel_ACTUALLY_REORDERS_and_stays_correct(k):
    """Runs the generated Metal. The tests above pin a Python reimplementation
    of the index formula, and the previous block watched a mutation survive a
    suite that pinned only its own Python."""
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    rows, cols = 8, COLS
    rng = np.random.default_rng(32)
    packed = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
    scale = rng.random((rows, cols // G.GROUP)).astype(np.float16) + 0.5
    x = rng.standard_normal(cols).astype(np.float32)
    lo, hi = (packed & 0x0F).astype(np.int32), (packed >> 4).astype(np.int32)
    w = np.empty((rows, cols), dtype=np.float64)
    w[:, 0::2], w[:, 1::2] = lo - G.BOUND, hi - G.BOUND
    w *= np.repeat(scale.astype(np.float64), G.GROUP, axis=1)
    oracle = w @ x.astype(np.float64)
    kern = mx.fast.metal_kernel(
        name=f"blk_{k}", input_names=["packed", "scales", "x"], output_names=["out"],
        source=G.source_operand_probe(rows, cols, f"rotblk{k}", TPR, TG),
        ensure_row_contiguous=True)
    (o,) = kern(inputs=[mx.array(packed), mx.array(scale), mx.array(x)],
                grid=(rows * TPR, 1, 1), threadgroup=(TG, 1, 1),
                output_shapes=[(rows,)], output_dtypes=[mx.float32])
    got = np.array(o, dtype=np.float64)
    assert np.linalg.norm(got - oracle) / np.linalg.norm(oracle) < 1e-5
