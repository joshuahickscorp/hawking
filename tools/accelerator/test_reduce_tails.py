"""The cross-lane reduction tail, as four arms that differ in nothing else.

ACCELERATOR_TWO_MORE_LEVERS_DIE eliminated six levers and named ONE structural
feature no arm had varied -- the serial tail where one lane sums TPR slots while
TPR-1 idle. These pin that the arms are comparable (same body), that the three
real ones are correct, and that the deletion control is WRONG, which is the only
thing standing between the comparison and vacuity.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gravity_native as G  # noqa: E402

mx = pytest.importorskip("mlx.core")

ROWS, COLS, TPR, TG = 128, 4096, 64, 128
REAL = ("serial", "simd", "tree")


def _sources():
    return {a: G.source_reduce(ROWS, COLS, a, TPR, TG) for a in G.REDUCE_TAILS}


def test_the_split_is_byte_identical_to_the_shipped_kernel():
    # The shipped constant is now a composition. If the split drifts, every arm
    # below is measuring something other than the kernel that executes.
    assert G.NATIVE_MATVEC_TPR == G.NATIVE_MATVEC_TPR_BODY + G.REDUCE_TAILS["serial"]


def test_the_arms_differ_only_in_the_tail():
    # This is what makes a timing difference attributable to the reduction: the
    # ELEMENT COUNT is identical because the body is byte-identical.
    body = G.NATIVE_MATVEC_TPR_BODY % {
        "PACKED_COLS": COLS // 2, "GROUPS": COLS // G.GROUP, "GROUP": G.GROUP,
        "BOUND": G.BOUND, "TPR": TPR, "TG": TG}
    for arm, src in _sources().items():
        assert src.startswith(body), arm


def test_four_distinct_kernels():
    assert len(set(_sources().values())) == 4


def _run(arm, dp, ds, dx):
    k = mx.fast.metal_kernel(
        name=f"t_reduce_{arm}", input_names=["packed", "scales", "x"],
        output_names=["out"], source=G.source_reduce(ROWS, COLS, arm, TPR, TG),
        ensure_row_contiguous=True)
    (o,) = k(inputs=[dp, ds, dx], grid=(ROWS * TPR, 1, 1), threadgroup=(TG, 1, 1),
             output_shapes=[(ROWS,)], output_dtypes=[mx.float32])
    mx.eval(o)
    return np.array(o, dtype=np.float64)


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((ROWS, COLS)) * 0.02).astype(np.float32)
    x = rng.standard_normal(COLS).astype(np.float32)
    packed, scale = G.pack_q4_g64(w)
    oracle = G.dequantize(packed, scale, COLS).astype(np.float64) @ x.astype(np.float64)
    return (mx.array(packed), mx.array(scale), mx.array(x),
            oracle, float(np.linalg.norm(oracle)))


@pytest.mark.parametrize("arm", REAL)
def test_real_tails_are_correct(arm, case):
    dp, ds, dx, oracle, n = case
    assert np.linalg.norm(_run(arm, dp, ds, dx) - oracle) / n < 1e-5


def test_the_deletion_control_is_wrong(case):
    # ANTI-VACUITY. "none" removes the reduction entirely, so out[row] holds one
    # lane's partial. If this ever matched, the reduction would be computing
    # nothing and every comparison against this control would prove nothing.
    dp, ds, dx, oracle, n = case
    assert np.linalg.norm(_run("none", dp, ds, dx) - oracle) / n > 0.1


def test_the_deletion_control_stores_from_every_lane(case):
    # NOT `if (lane == 0u)`: a predicated store lets the compiler sink the whole
    # loop into the branch, so TPR-1 lanes would do NO WORK and the control would
    # measure dead-code elimination instead of the tail's cost.
    assert "if (lane" not in G.REDUCE_TAILS["none"]
    assert G.REDUCE_TAILS["none"].strip() == "out[row] = acc;"


def test_simd_refuses_a_partial_simdgroup():
    # simd_sum reduces exactly SIMD_WIDTH lanes; a row spanning a partial
    # simdgroup would silently drop the lanes outside it.
    G.source_reduce(ROWS, COLS, "simd", 32, 128)
    with pytest.raises(ValueError, match="simd_sum reduces exactly"):
        G.source_reduce(ROWS, COLS, "simd", 16, 128)


def test_unknown_reduction_is_refused():
    with pytest.raises(ValueError, match="unknown reduction"):
        G.source_reduce(ROWS, COLS, "median", TPR, TG)
