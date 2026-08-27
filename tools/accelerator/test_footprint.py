"""Address footprint at fixed work.

ACCELERATOR_ENTERING_THE_LOOP_COSTS_MORE_THAN_RUNNING_IT named cold cache lines
as the only unexamined candidate for the loop-entry cost. `_localN` is that
probe: it confines every address to N groups and changes NOTHING else.
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


def test_only_the_addresses_move():
    # THE SINGLE-VARIABLE CLAIM, asserted rather than believed: the loop bound,
    # the iteration count, the element count and the reduction must be identical
    # to the base, so a timing difference is the footprint and nothing else.
    base, local = _src("thin2"), _src("thin2_local2")
    assert base != local
    for line in base.splitlines():
        t = line.strip()
        if "c0 =" in t or "float s =" in t:
            continue
        assert t in local, t
    assert local.count("for (uint") == base.count("for (uint")
    assert "(g % 2u)" in local and "(g % 2u)" not in base


def test_the_index_still_varies_so_the_body_cannot_be_hoisted():
    # A fully loop-invariant body would let the compiler run one iteration and
    # the arm would measure hoisting, not footprint.
    assert "(g % 2u) * 64u" in _src("thin2_local2")


def test_a_non_divisor_is_refused():
    with pytest.raises(ValueError, match="DIVISOR"):
        G.source_operand_probe(ROWS, COLS, "thin2_local7", TPR, TG)


def test_a_drifted_template_raises_rather_than_returning_the_base():
    # A no-op replacement would make the probe BE its base -- a tie reading as a
    # finding, which is this program's most-repeated instrument failure.
    with pytest.raises(AssertionError, match="not in the source to confine"):
        G.source_operand_probe(ROWS, COLS, "redonly_local2", TPR, TG)


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


@pytest.mark.parametrize("probe", ["thin2_local1", "thin2_local2", "thin64_local2"])
def test_every_confined_probe_is_WRONG_and_NON_DEGENERATE(probe, case):
    # ANTI-VACUITY. A confined probe reads the WRONG weights by construction, so
    # it is not a candidate kernel; and wrong is not enough, because a folded
    # loop is also wrong.
    dp, ds, dx, oracle, n = case
    got = _run(_src(probe), dp, ds, dx, f"fpt_{probe}")
    assert np.linalg.norm(got - oracle) / n > 0.1
    assert len(np.unique(got)) > ROWS * 0.9


def test_the_widest_confinement_IS_the_baseline(case):
    # local at the full group count must reproduce the shipped kernel EXACTLY,
    # or the family does not contain what executes and its differences are not
    # measuring confinement.
    groups = COLS // G.GROUP
    dp, ds, dx, oracle, n = case
    got = _run(_src(f"thin64_local{groups}"), dp, ds, dx, "fpt_widest")
    assert np.linalg.norm(got - oracle) / n < 1e-5
