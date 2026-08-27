"""Separating SUBMISSION from GRID LAUNCH.

ACCELERATOR_THE_FLOOR_IS_NOT_THE_ELEMENT_EITHER fitted a fixed cost of 0.2450 ms
= 86.1% of an isolated matvec and named its own boundary: every arm held the grid
at 1.11M threads, so submission and grid launch were not separated. The trivial
probe separates them by keeping the OUTPUT fixed while the grid moves 64x.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gravity_native as G  # noqa: E402

mx = pytest.importorskip("mlx.core")

ROWS, COLS, TG = 128, 4096, 128
TPRS = (1, 2, 8, 64)


def test_the_grid_moves_and_the_output_does_not():
    # This is the whole design: same 128 outputs at every grid from 128 threads
    # to 8192. If the output moved too, the sweep would confound size with grid.
    for t in TPRS:
        assert G.source_trivial(ROWS, COLS, t, TG)
    assert len({G.source_trivial(ROWS, COLS, t, TG) for t in TPRS}) == len(TPRS)


def test_every_buffer_is_referenced():
    # MLX binds the signature from the named inputs, so an unreferenced buffer
    # would make this a DIFFERENT dispatch from the arms it is compared against --
    # the one thing that would make the comparison meaningless.
    src = G.source_trivial(ROWS, COLS, 64, TG)
    for buf in ("packed", "scales", "x"):
        assert buf in src, buf


def test_illegal_geometry_is_refused():
    G.source_trivial(ROWS, COLS, 64, TG)
    with pytest.raises(ValueError, match="whole number of rows"):
        G.source_trivial(ROWS, COLS, 48, TG)
    with pytest.raises(ValueError, match="not a multiple of threadgroup"):
        G.source_trivial(100, COLS, 1, TG)


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((ROWS, COLS)) * 0.02).astype(np.float32)
    x = rng.standard_normal(COLS).astype(np.float32)
    packed, scale = G.pack_q4_g64(w)
    oracle = G.dequantize(packed, scale, COLS).astype(np.float64) @ x.astype(np.float64)
    return (mx.array(packed), mx.array(scale), mx.array(x),
            oracle, float(np.linalg.norm(oracle)))


def _run(tpr, dp, ds, dx):
    k = mx.fast.metal_kernel(
        name=f"tgp_{tpr}", input_names=["packed", "scales", "x"], output_names=["out"],
        source=G.source_trivial(ROWS, COLS, tpr, TG), ensure_row_contiguous=True)
    (o,) = k(inputs=[dp, ds, dx], grid=(ROWS * tpr, 1, 1), threadgroup=(TG, 1, 1),
             output_shapes=[(ROWS,)], output_dtypes=[mx.float32])
    mx.eval(o)
    return np.array(o, dtype=np.float64)


@pytest.mark.parametrize("tpr", TPRS)
def test_the_probe_is_WRONG(tpr, case):
    # ANTI-VACUITY: it must not be the matvec, or the comparison proves nothing.
    dp, ds, dx, oracle, n = case
    assert np.linalg.norm(_run(tpr, dp, ds, dx) - oracle) / n > 0.1


@pytest.mark.parametrize("tpr", TPRS)
def test_the_probe_is_NON_DEGENERATE(tpr, case):
    # Wrong is not enough: a kernel storing a CONSTANT is also wrong and would
    # time an empty dispatch rather than a cheap one.
    dp, ds, dx, _, _ = case
    got = _run(tpr, dp, ds, dx)
    assert np.count_nonzero(got) == ROWS
    assert len(np.unique(got)) > 8


def test_the_grid_does_not_change_the_answer(case):
    # The arms must differ ONLY in grid. If the answer moved with tpr, a timing
    # difference could be a different computation rather than a different launch.
    dp, ds, dx, _, _ = case
    ref = _run(TPRS[0], dp, ds, dx)
    for t in TPRS[1:]:
        assert np.array_equal(_run(t, dp, ds, dx), ref)
