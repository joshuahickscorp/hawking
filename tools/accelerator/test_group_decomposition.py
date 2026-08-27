"""Decomposing the group-loop term into the reduction and everything else.

ACCELERATOR_THE_FLOOR_SPLITS_THREE_WAYS left the group loop as the only part of
an isolated matvec that is neither submission, nor grid, nor elements, and named
the scale loads as the suspect. These pin the probes that decompose it, including
the one that exists ONLY because the obvious arm carried a second variable.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gravity_native as G  # noqa: E402

mx = pytest.importorskip("mlx.core")

ROWS, COLS, TPR, TG = 128, 4096, 64, 128
PROBES = ("redonly", "thin2_noreduce", "thin2_nobarrier")


def _src(p):
    return G.source_operand_probe(ROWS, COLS, p, TPR, TG)


def test_each_probe_removes_exactly_what_it_names():
    thin2 = _src("thin2")
    assert "for (uint g" in thin2 and "threadgroup_barrier" in thin2
    # redonly: no group loop at all, barrier kept.
    assert "for (uint g" not in _src("redonly")
    assert "threadgroup_barrier" in _src("redonly")
    # noreduce and nobarrier: loop kept, barrier gone.
    for p in ("thin2_noreduce", "thin2_nobarrier"):
        assert "for (uint g" in _src(p), p
        assert "threadgroup_barrier" not in _src(p), p


def test_nobarrier_holds_the_store_traffic_that_noreduce_changes():
    # THE REASON THE nobarrier ARM EXISTS. _noreduce drops the reduction AND
    # replaces one predicated store per row with an unconditional store from every
    # lane -- a SECOND VARIABLE, so it cannot price the barrier. _nobarrier keeps
    # the predicated store and still writes the threadgroup slot from every lane,
    # so the loop cannot be sunk into the branch.
    assert "if (lane == 0u) out[row]" in _src("thin2_nobarrier")
    assert "if (lane" not in _src("thin2_noreduce").split("part[")[-1]
    for p in ("thin2", "thin2_nobarrier"):
        assert "part[lid] = acc;" in _src(p), p


def test_every_probe_is_a_distinct_kernel():
    srcs = {p: _src(p) for p in PROBES}
    srcs["thin2"] = _src("thin2")
    srcs["trivial"] = G.source_trivial(ROWS, COLS, TPR, TG)
    assert len(set(srcs.values())) == len(srcs)


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


@pytest.mark.parametrize("probe", PROBES)
def test_every_probe_is_WRONG(probe, case):
    # ANTI-VACUITY: a probe that matched the oracle removed nothing.
    dp, ds, dx, oracle, n = case
    assert np.linalg.norm(_run(_src(probe), dp, ds, dx, f"tgd_{probe}") - oracle) / n > 0.1


@pytest.mark.parametrize("probe", PROBES)
def test_every_probe_is_NON_DEGENERATE(probe, case):
    # Wrong is not enough. redonly in particular must not reduce to a per-lane
    # CONSTANT, which would sum to the same value in every row and time a
    # degenerate kernel; its accumulator depends on the ROW as well as the lane.
    dp, ds, dx, _, _ = case
    got = _run(_src(probe), dp, ds, dx, f"tgn_{probe}")
    assert len(np.unique(got)) > ROWS * 0.9, probe


def test_the_baseline_is_still_correct(case):
    dp, ds, dx, oracle, n = case
    got = _run(G.source_tpr(ROWS, COLS, TPR, TG), dp, ds, dx, "tgd_base")
    assert np.linalg.norm(got - oracle) / n < 1e-5


def test_a_missing_tail_refuses_rather_than_silently_returning_the_base(monkeypatch):
    # If the tail template drifts, the replacement would no-op and the probe would
    # BE its base -- a tie that reads as a finding rather than a raise.
    G.source_operand_probe(ROWS, COLS, "thin2_nobarrier", TPR, TG)
    monkeypatch.setitem(G.REDUCE_TAILS, "serial", "\n// not in the source\n")
    with pytest.raises(AssertionError, match="serial tail is not in this source"):
        G.source_operand_probe(ROWS, COLS, "thin2_nobarrier", TPR, TG)


def test_an_unknown_probe_is_still_refused():
    with pytest.raises(ValueError, match="unknown operand probe"):
        G.source_operand_probe(ROWS, COLS, "redonly_nobarrier", TPR, TG)
