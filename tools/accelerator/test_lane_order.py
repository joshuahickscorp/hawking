"""Which lane touches which group, at identical footprint and identical work.

ACCELERATOR_THE_ENTRY_COST_IS_FOOTPRINT measured that footprint costs and said
its mechanism was unnamed. A lane permutation is the one variable that leaves
the address SET untouched and moves only the ASSIGNMENT.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gravity_native as G  # noqa: E402

mx = pytest.importorskip("mlx.core")
ROWS, COLS, TPR, TG = 128, 4096, 64, 128


def _src(p):
    return G.source_operand_probe(ROWS, COLS, p, TPR, TG)


def test_the_permutation_is_a_BIJECTION_over_every_lane():
    # THE LOAD-BEARING CHECK OF THIS FAMILY. Every other probe here is wrong by
    # construction and its anti-vacuity control is "it must be wrong". A
    # permutation must be CORRECT, so what stands in that control's place is
    # this: if the map were not a bijection a group would be visited twice or
    # skipped, the footprint would move, and the arm would measure something
    # other than order while still looking plausible.
    for mult in (1, 37):
        assert sorted((l * mult) % TPR for l in range(TPR)) == list(range(TPR))
    assert [(l * 37) % TPR for l in range(TPR)] != list(range(TPR))


def test_an_even_multiplier_is_refused():
    with pytest.raises(ValueError, match="ODD multiplier"):
        G.source_operand_probe(ROWS, COLS, "_lane36", TPR, TG)


def test_the_arms_differ_and_differ_only_in_the_loop_init():
    a, b = _src("_lane1"), _src("_lane37")
    assert a != b
    da = [l for l in a.splitlines() if l not in b.splitlines()]
    assert len(da) == 1 and "for (uint g" in da[0], da


def test_the_identity_arm_is_the_control_not_the_shipped_kernel():
    # _lane1 carries the SAME multiply and modulo as _lane37, so it prices the
    # ORDER alone. Comparing against the shipped kernel would confound the order
    # with two extra instructions -- pinned because the two look interchangeable.
    assert _src("_lane1") != G.source_tpr(ROWS, COLS, TPR, TG)
    assert "(lane * 1u)" in _src("_lane1")


def test_a_missing_loop_raises_rather_than_returning_the_base():
    with pytest.raises(AssertionError, match="not in the source to reorder"):
        G.source_operand_probe(ROWS, COLS, "redonly_lane37", TPR, TG)


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((ROWS, COLS)) * 0.02).astype(np.float32)
    x = rng.standard_normal(COLS).astype(np.float32)
    packed, scale = G.pack_q4_g64(w)
    oracle = G.dequantize(packed, scale, COLS).astype(np.float64) @ x.astype(np.float64)
    return (mx.array(packed), mx.array(scale), mx.array(x),
            oracle, float(np.linalg.norm(oracle)))


def _run(src, dp, ds, dx, name):
    k = mx.fast.metal_kernel(
        name=name, input_names=["packed", "scales", "x"], output_names=["out"],
        source=src, ensure_row_contiguous=True)
    (o,) = k(inputs=[dp, ds, dx], grid=(ROWS * TPR, 1, 1), threadgroup=(TG, 1, 1),
             output_shapes=[(ROWS,)], output_dtypes=[mx.float32])
    mx.eval(o)
    return np.array(o, dtype=np.float64)


@pytest.mark.parametrize("probe", ["_lane1", "_lane37"])
def test_a_reordered_kernel_is_still_CORRECT(probe, case):
    # The inverse of every other probe in this family: a bijection changes only
    # which lane accumulates which group, so the answer must survive.
    dp, ds, dx, oracle, n = case
    assert np.linalg.norm(_run(_src(probe), dp, ds, dx, f"lo_{probe}") - oracle) / n < 1e-5
