"""Which per-element operation sets the floor -- and whether any does.

ACCELERATOR_THE_FLOOR_IS_THE_ELEMENT_NOT_THE_BYTE named the floor "one x read and
one fused multiply-add per weight". Seven levers died against it, but no arm had
ever removed the x READ, and none had removed the ELEMENT WORK ITSELF at a fixed
grid. These pin the probes that do both, and pin that every one of them is WRONG
-- which is the only thing standing between the comparison and vacuity.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gravity_native as G  # noqa: E402

mx = pytest.importorskip("mlx.core")

ROWS, COLS, TPR, TG = 128, 4096, 64, 128
PROBES = ("xreuse", "nomul", "noxread", "thin2", "thin16")


def test_the_widest_thin_IS_the_shipped_kernel():
    # THE ANCHOR. thin64 keeps every element, so it must be the kernel itself byte
    # for byte -- otherwise the sweep measures a family that excludes what runs.
    assert (G.source_operand_probe(ROWS, COLS, "thin64", TPR, TG)
            == G.source_tpr(ROWS, COLS, TPR, TG))


def test_every_probe_is_a_distinct_kernel():
    srcs = {p: G.source_operand_probe(ROWS, COLS, p, TPR, TG) for p in PROBES}
    srcs["baseline"] = G.source_tpr(ROWS, COLS, TPR, TG)
    assert len(set(srcs.values())) == len(srcs)


def test_the_x_read_count_is_what_the_probes_claim():
    # The whole block rests on these counts, so they are asserted rather than read
    # off the template by eye.
    n = lambda p: G.source_operand_probe(ROWS, COLS, p, TPR, TG).count("x[c0")
    assert G.source_tpr(ROWS, COLS, TPR, TG).count("x[c0") == 2
    assert n("xreuse") == 1
    assert n("nomul") == 2      # the MULTIPLY is removed, not the reads
    assert n("noxread") == 0


def test_thin_refuses_a_count_it_cannot_emit():
    G.source_operand_probe(ROWS, COLS, "thin2", TPR, TG)
    for bad in ("thin3", "thin128", "thin0"):
        with pytest.raises(ValueError, match="EVEN element count"):
            G.source_operand_probe(ROWS, COLS, bad, TPR, TG)


def test_unknown_probe_is_refused():
    with pytest.raises(ValueError, match="unknown operand probe"):
        G.source_operand_probe(ROWS, COLS, "freelunch", TPR, TG)


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


def test_the_baseline_is_still_correct(case):
    dp, ds, dx, oracle, n = case
    got = _run(G.source_tpr(ROWS, COLS, TPR, TG), dp, ds, dx, "top_base")
    assert np.linalg.norm(got - oracle) / n < 1e-5


@pytest.mark.parametrize("probe", PROBES)
def test_every_probe_is_WRONG(probe, case):
    # ANTI-VACUITY. A probe that matched did not remove what it claims to remove,
    # and every timing comparison against it would prove nothing.
    dp, ds, dx, oracle, n = case
    got = _run(G.source_operand_probe(ROWS, COLS, probe, TPR, TG),
               dp, ds, dx, f"top_{probe}")
    assert np.linalg.norm(got - oracle) / n > 0.1


@pytest.mark.parametrize("probe", PROBES)
def test_every_probe_is_NON_DEGENERATE(probe, case):
    # Wrong is not enough: a probe whose loop folded to nothing would also be
    # wrong, and would time an empty kernel rather than a cheaper one.
    dp, ds, dx, _, _ = case
    got = _run(G.source_operand_probe(ROWS, COLS, probe, TPR, TG),
               dp, ds, dx, f"tnd_{probe}")
    assert np.count_nonzero(got) == ROWS
    assert float(np.std(got)) > 0.0


def test_a_probe_that_cannot_find_the_body_refuses(monkeypatch):
    # If the templates drift apart the replacement would silently no-op and the
    # probe would BE the baseline -- a tie that reads as a finding.
    monkeypatch.setitem(G.UNPACK_BODIES, "byte", "        // not in the source\n")
    with pytest.raises(AssertionError, match="templates have drifted apart"):
        G.source_operand_probe(ROWS, COLS, "noxread", TPR, TG)
