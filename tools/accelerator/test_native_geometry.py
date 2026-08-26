"""The resident's own threads-per-row geometry, and the refusals that keep it legal."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
import gravity_native as G  # noqa: E402

mx = pytest.importorskip("mlx.core")

# COLS is 4096 ON PURPOSE, not for coverage: at GROUPS = COLS/64 the kernel gives
# each of the TPR lanes GROUPS/TPR groups, so below COLS=4096 most lanes get NO
# work and write their slot instantly. The reducing lane then finishes LAST and
# the race window is closed -- the barrier-stripped control was measured SILENT
# at 256x256, 0 of 8 runs, and a test written there would have reported the
# barrier unnecessary. This is the window mechanism ACCELERATOR_BARRIER_WINDOW
# measured, met again in a new kernel, and the shape is pinned so it cannot rot.
ROWS, COLS, TPR, TG = 256, 4096, 64, 128


def _run(src, name, packed, scale, x, grid):
    k = mx.fast.metal_kernel(name=name, input_names=["packed", "scales", "x"],
                             output_names=["out"], source=src, ensure_row_contiguous=True)
    (o,) = k(inputs=[mx.array(packed), mx.array(scale), mx.array(x)],
             grid=(grid, 1, 1), threadgroup=(TG, 1, 1),
             output_shapes=[(ROWS,)], output_dtypes=[mx.float32])
    mx.eval(o)
    return np.array(o, dtype=np.float64)


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(3)
    w = (rng.standard_normal((ROWS, COLS)) * 0.02).astype(np.float32)
    x = rng.standard_normal(COLS).astype(np.float32)
    packed, scale = G.pack_q4_g64(w)
    oracle = G.dequantize(packed, scale, COLS).astype(np.float64) @ x.astype(np.float64)
    return packed, scale, x, oracle


def test_tpr64_matches_the_same_oracle_as_one_thread_per_row(case):
    packed, scale, x, oracle = case
    n = np.linalg.norm(oracle)
    one = _run(G.sources(ROWS, COLS)["native"], "tg_t1", packed, scale, x, ROWS)
    many = _run(G.source_tpr(ROWS, COLS, TPR, TG), "tg_t64", packed, scale, x, ROWS * TPR)
    assert np.linalg.norm(one - oracle) / n < 1e-5
    assert np.linalg.norm(many - oracle) / n < 1e-5


def test_THE_BARRIER_IS_LOAD_BEARING(case):
    """Without this the correctness test above proves nothing about the barrier.
    kernel_forge.barrier_control_prior calls this shape LOUD -- nothing is
    upstream of the barrier, so the threadgroup is unsynchronized when it
    reaches the conflicting access and the race window is wide."""
    packed, scale, x, oracle = case
    n = np.linalg.norm(oracle)
    bad = G.source_tpr(ROWS, COLS, TPR, TG).replace(
        "threadgroup_barrier(mem_flags::mem_threadgroup);", "")
    fired = sum(
        np.linalg.norm(_run(bad, f"tg_nb{i}", packed, scale, x, ROWS * TPR) - oracle) / n > 1e-4
        for i in range(8))
    assert fired > 0, (
        "the barrier-stripped control never fired in 8 runs, so this sweep "
        "establishes nothing about the barrier -- not that it is unnecessary")


@pytest.mark.parametrize("rows,tpr,tg,why", [
    (ROWS, 64, 96, "whole number of rows"),
    (3, 64, 128, "padded"),
])
def test_an_ILLEGAL_GEOMETRY_IS_REFUSED_rather_than_racing(rows, tpr, tg, why):
    """A padded grid means threads that take the early return skip the barrier,
    which is undefined -- refusing beats emitting a kernel that usually works."""
    with pytest.raises(ValueError, match=why):
        G.source_tpr(rows, COLS, tpr, tg)


def test_the_two_geometries_are_DIFFERENT_KERNELS():
    """Pinned so a future edit cannot collapse them and leave the receipt's
    tpr1-vs-tpr64 comparison reading as one kernel measured twice."""
    assert G.source_tpr(ROWS, COLS, TPR, TG) != G.sources(ROWS, COLS)["native"]
    assert "thread_position_in_threadgroup" in G.source_tpr(ROWS, COLS, TPR, TG)
    assert "thread_position_in_threadgroup" not in G.sources(ROWS, COLS)["native"]
