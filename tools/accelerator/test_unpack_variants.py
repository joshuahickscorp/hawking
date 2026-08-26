"""Three unpacks at one geometry. The variants exist to be MEASURED against each
other, so what the tests must pin is that they are genuinely different kernels
computing the SAME answer -- a variant that silently fell back to the byte body
would report a speedup of exactly 1.000 and read as a finding."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
import gravity_native as G  # noqa: E402

mx = pytest.importorskip("mlx.core")

ROWS, COLS, TPR, TG = 256, 4096, 64, 128
VARIANTS = ("byte", "word", "lut", "ilp4")


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(11)
    w = (rng.standard_normal((ROWS, COLS)) * 0.02).astype(np.float32)
    x = rng.standard_normal(COLS).astype(np.float32)
    packed, scale = G.pack_q4_g64(w)
    oracle = G.dequantize(packed, scale, COLS).astype(np.float64) @ x.astype(np.float64)
    return packed, scale, x, oracle


def _run(u, packed, scale, x):
    k = mx.fast.metal_kernel(
        name=f"tuv_{u}", input_names=["packed", "scales", "x"], output_names=["out"],
        source=G.source_unpack(ROWS, COLS, u, TPR, TG), ensure_row_contiguous=True)
    (o,) = k(inputs=[mx.array(packed), mx.array(scale), mx.array(x)],
             grid=(ROWS * TPR, 1, 1), threadgroup=(TG, 1, 1),
             output_shapes=[(ROWS,)], output_dtypes=[mx.float32])
    mx.eval(o)
    return np.array(o, dtype=np.float64)


@pytest.mark.parametrize("u", VARIANTS)
def test_every_unpack_computes_the_same_answer(u, case):
    packed, scale, x, oracle = case
    assert np.linalg.norm(_run(u, packed, scale, x) - oracle) / np.linalg.norm(oracle) < 1e-5


def test_x_STAGING_computes_the_same_answer_and_REFUSES_an_oversized_allocation(case):
    """x staged in threadgroup memory is the operand-reuse arm. The refusal is
    the load-bearing half: Metal allows 32,768 bytes and exceeding it fails
    twenty lines inside MLX's generated header with no mention of the cause."""
    packed, scale, x, oracle = case
    src = G.source_stage_x(ROWS, COLS, "byte", TPR, TG)
    assert f"threadgroup float xs[{COLS}]" in src
    assert " x[c0 + k" not in src, "a device read of x survived the rewrite"
    k = mx.fast.metal_kernel(
        name="tuv_stage", input_names=["packed", "scales", "x"], output_names=["out"],
        source=src, ensure_row_contiguous=True)
    (o,) = k(inputs=[mx.array(packed), mx.array(scale), mx.array(x)],
             grid=(ROWS * TPR, 1, 1), threadgroup=(TG, 1, 1),
             output_shapes=[(ROWS,)], output_dtypes=[mx.float32])
    mx.eval(o)
    got = np.array(o, dtype=np.float64)
    assert np.linalg.norm(got - oracle) / np.linalg.norm(oracle) < 1e-5
    with pytest.raises(ValueError, match="exceeds Metal"):
        G.source_stage_x(ROWS, 8192, "byte", TPR, 1024)


def test_ilp4_breaks_the_chain_WITHOUT_changing_the_element_count():
    """Four accumulators must change ASSOCIATION ORDER and nothing else. If the
    step or the loads moved too, a timing difference could not be attributed."""
    serial = G.source_unpack(ROWS, COLS, "byte", TPR, TG)
    ilp = G.source_unpack(ROWS, COLS, "ilp4", TPR, TG)
    assert "float a0 = 0.0f" in ilp and "float a0" not in serial
    assert "acc += (a0 + a1) + (a2 + a3);" in ilp
    assert ilp.index("acc += (a0 + a1)") < ilp.index("threadgroup float part[")
    # 4 bytes per iteration at 4x the step is the SAME bytes and elements.
    assert "k += 8" in ilp and ilp.count("packed[pbase") == 4
    assert "k += 2" in serial and serial.count("packed[pbase") == 1


def test_the_three_are_ACTUALLY_DIFFERENT_KERNELS():
    """Without this a variant that fell through to the byte body would time at
    exactly 1.000x and be reported as 'no difference' rather than 'not built'."""
    src = {u: G.source_unpack(ROWS, COLS, u, TPR, TG) for u in VARIANTS}
    assert len(set(src.values())) == len(VARIANTS)
    assert "device uint*" in src["word"] and "device uint*" not in src["byte"]
    assert "threadgroup float2 lut" in src["lut"] and "lut[" not in src["byte"]
    assert "k += 8" in src["word"] and "k += 2" in src["byte"]


def test_the_alignment_guard_CANNOT_FIRE_FOR_A_LEGAL_SHAPE_and_says_so():
    """FOUND BY WRITING THE TEST, not by reading the code. A 4-byte read off a
    2-byte boundary is undefined, so source_unpack guards it -- but the guard is
    UNREACHABLE for any shape the kernel accepts: GROUP is 64, so a legal cols is
    a multiple of 64, so cols/2 is a multiple of 32 and therefore of 4, always.

    The guard is KEPT because GROUP is a module constant somebody may lower and
    an unreachable guard costs nothing, while a missing one is a fault. What is
    NOT kept is the pretence that it is load-bearing: this test pins BOTH that
    every legal shape passes it AND that it fires on the illegal stride, so the
    day GROUP changes the second assertion is the one that still means something."""
    for extra in (0, 64, 128, 1024):           # every legal shape: never refused
        G.source_unpack(ROWS, COLS + extra, "word", TPR, TG)
    assert all((c % G.GROUP) or ((c // 2) % 4 == 0) for c in range(64, 8192, 64)), (
        "a legal cols with an unaligned stride exists; the guard is reachable "
        "after all and this test's own premise is wrong")
    with pytest.raises(ValueError, match="4-byte-aligned"):
        G.source_unpack(ROWS, COLS + 2, "word", TPR, TG)   # illegal cols, guard fires


def test_an_unknown_unpack_is_refused_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown unpack"):
        G.source_unpack(ROWS, COLS, "simd_shuffle_magic", TPR, TG)


def test_the_no_weights_TIMING_CONTROL_is_not_a_no_op(case):
    """The floor probe's control must still read x and still do the FMAs. If the
    compiler folded the loop the control would measure nothing and the 'weight
    bytes are free' reading would be an artefact of an empty kernel."""
    packed, scale, x, _ = case
    src = G.source_unpack(ROWS, COLS, "byte", TPR, TG).replace(
        "        uchar byte = packed[pbase + (c0 + k) / 2u];\n"
        f"        float w0 = (float)((int)(byte & 0x0F) - {G.BOUND}) * s;\n"
        f"        float w1 = (float)((int)(byte >> 4)   - {G.BOUND}) * s;\n"
        "        acc += w0 * x[c0 + k] + w1 * x[c0 + k + 1u];",
        "        acc += s * x[c0 + k] + s * x[c0 + k + 1u];")
    assert "packed[pbase" not in src, "the control still reads the packed array"
    k = mx.fast.metal_kernel(
        name="tuv_nofetch", input_names=["packed", "scales", "x"], output_names=["out"],
        source=src, ensure_row_contiguous=True)
    (o,) = k(inputs=[mx.array(packed), mx.array(scale), mx.array(x)],
             grid=(ROWS * TPR, 1, 1), threadgroup=(TG, 1, 1),
             output_shapes=[(ROWS,)], output_dtypes=[mx.float32])
    mx.eval(o)
    v = np.array(o, dtype=np.float64)
    assert np.count_nonzero(v) > ROWS * 0.9, "the control folded away to zeros"
