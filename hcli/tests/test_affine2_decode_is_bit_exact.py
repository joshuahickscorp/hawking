"""S006 17-19: the cost was the interpreter, not the algorithm -- and the fix
must produce THE SAME NUMBERS, not close ones.

decode_mixed_payload's CODEC_AFFINE branch ran one Python loop body per
ELEMENT. At Qwen3.8-27B's [17408, 5120] that is 89.1M interpreter iterations
for a single tensor. pack_hq30uq4() in the same file already packs this exact
layout with whole-array ops, so the vectorised idiom was present; this path
just did not use it.

This repo has a standing rule that a speedup which changes numerics is not a
speedup (tools/accelerator/gravity_native.py:751 holds its own equivalent
rewrite to 8.3e-17 relative, "not a tolerance"). Here the two forms are integer
gathers and one float32 multiply-add, so the bar is exact: identical bit
patterns, compared as uint32.
"""
import numpy as np
import pytest

from tools.qwen38_sub15_pack import CODEC_NAMES, PackError, check_codec_known


def _scalar_reference(codes, scales, biases, rows, cols):
    """The pre-vectorisation loop, verbatim, kept as the oracle."""
    out = np.empty((rows, cols), dtype=np.float32)
    gpr = cols // 32
    for row in range(rows):
        for col in range(cols):
            group = row * gpr + col // 32
            element = row * cols + col
            bit0 = element * 2
            byte = int(codes[bit0 >> 3])
            q = (byte >> (bit0 & 7)) & 3
            out[row, col] = float(q) * scales[group] + biases[group]
    return out


def _vectorised(codes, scales, biases, rows, cols):
    gpr = cols // 32
    n = rows * cols
    bit0 = np.arange(n, dtype=np.int64) * 2
    q = (codes[bit0 >> 3] >> (bit0 & 7).astype(np.uint8)) & 3
    col_ix = np.arange(n, dtype=np.int64) % cols
    row_ix = np.arange(n, dtype=np.int64) // cols
    group = row_ix * gpr + col_ix // 32
    return (q.astype(np.float32) * scales[group] + biases[group]).reshape(rows, cols)


def _case(rows, cols, seed):
    rng = np.random.default_rng(seed)
    groups = rows * (cols // 32)
    n = rows * cols
    return (rng.integers(0, 256, size=(n * 2 + 7) // 8, dtype=np.uint8),
            rng.normal(size=groups).astype(np.float32),
            rng.normal(size=groups).astype(np.float32))


@pytest.mark.parametrize("rows,cols", [(4, 32), (8, 64), (16, 128), (3, 96), (1, 32)])
def test_vectorised_decode_is_bit_identical_to_the_loop(rows, cols):
    codes, scales, biases = _case(rows, cols, seed=rows * 1000 + cols)
    a = _scalar_reference(codes, scales, biases, rows, cols)
    b = _vectorised(codes, scales, biases, rows, cols)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), (
        f"{rows}x{cols}: vectorised decode differs from the scalar oracle. "
        "A speedup that changes numbers is not a speedup.")


def test_the_oracle_can_actually_fail():
    """Negative control: perturb one code byte and the comparison must break,
    otherwise the equality above proves nothing."""
    rows, cols = 8, 64
    codes, scales, biases = _case(rows, cols, seed=1)
    a = _scalar_reference(codes, scales, biases, rows, cols)
    codes2 = codes.copy()
    codes2[0] ^= 0xFF
    b = _vectorised(codes2, scales, biases, rows, cols)
    assert not np.array_equal(a.view(np.uint32), b.view(np.uint32))


def test_an_unnamed_codec_is_refused_before_the_decode():
    """It used to decode 89M elements and THEN raise KeyError on the name."""
    for known in sorted(CODEC_NAMES):
        check_codec_known(known)
    with pytest.raises(PackError) as e:
        check_codec_known(5)
    assert "before decode" in str(e.value)
