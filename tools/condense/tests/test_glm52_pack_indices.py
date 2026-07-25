"""Index bit-packing: exact parity with the reference decode, at bounded cost.

The packed index stream is the whole artifact minus the codebook, so unpacking it is on
every load path.  The old decode paid 145x the payload in uint64 temporaries; these tests
pin both halves of the fix -- the answer must not move, and the cost must stay bounded.
"""
from __future__ import annotations

import os
import sys
import tracemalloc

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import glm52_pack as pack  # noqa: E402

# Real R0 geometry: gate/up [2048,6144] at D=8 is rows*nchunk = 2048*768 indices at 7 bits.
R0_COUNT = 2048 * 768
R0_BITS = 7


def _reference_unpack(raw: bytes, count: int, bits: int) -> np.ndarray:
    """The pre-fix decode, kept verbatim as the parity oracle."""
    unpacked = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[: count * bits]
    grid = unpacked.reshape(count, bits).astype(np.uint64)
    weights = (np.uint64(1) << np.arange(bits - 1, -1, -1, dtype=np.uint64))
    return (grid * weights).sum(axis=1)


@pytest.mark.parametrize("bits", list(range(1, 17)))
@pytest.mark.parametrize("count", [1, 7, 8, 9, 63, 100, 255, 1000, 4097])
def test_parity_with_reference(bits, count):
    rng = np.random.default_rng(bits * 10007 + count)
    values = rng.integers(0, 1 << bits, size=count, dtype=np.uint64)
    raw = pack.pack_indices(values, bits)
    got = pack.unpack_indices(raw, count, bits)
    assert np.array_equal(got.astype(np.uint64), _reference_unpack(raw, count, bits))


@pytest.mark.parametrize("bits", list(range(1, 17)))
def test_round_trip_is_exact(bits):
    rng = np.random.default_rng(bits)
    for count in (1, 8, 9, 8191):
        values = rng.integers(0, 1 << bits, size=count, dtype=np.uint64)
        got = pack.unpack_indices(pack.pack_indices(values, bits), count, bits)
        assert np.array_equal(got.astype(np.uint64), values)


@pytest.mark.parametrize("bits", list(range(1, 17)))
def test_extremal_values_survive(bits):
    """Saturated and zero indices are where an off-by-one shift shows up."""
    values = np.array([0, (1 << bits) - 1] * 37 + [0], dtype=np.uint64)
    got = pack.unpack_indices(pack.pack_indices(values, bits), values.size, bits)
    assert np.array_equal(got.astype(np.uint64), values)


@pytest.mark.parametrize("bits,expected", [(1, np.uint8), (7, np.uint8), (8, np.uint8),
                                           (9, np.uint16), (16, np.uint16)])
def test_output_dtype_is_narrow(bits, expected):
    raw = pack.pack_indices(np.zeros(64, dtype=np.uint64), bits)
    assert pack.unpack_indices(raw, 64, bits).dtype == expected


def test_bits_over_sixteen_refused():
    with pytest.raises(ValueError):
        pack.unpack_indices(b"\x00" * 64, 8, 17)


def test_peak_memory_bounded_at_r0():
    """MEASURED at the production index count: temporaries must stay near the payload."""
    rng = np.random.default_rng(0)
    values = rng.integers(0, 1 << R0_BITS, size=R0_COUNT, dtype=np.uint64)
    raw = pack.pack_indices(values, R0_BITS)
    assert len(raw) == 1376256

    pack.unpack_indices(raw, R0_COUNT, R0_BITS)  # warm any numpy first-call allocation
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = pack.unpack_indices(raw, R0_COUNT, R0_BITS)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert out.dtype == np.uint8
    assert peak <= 4 * len(raw), f"peak {peak} B is {peak / len(raw):.1f}x the payload"


def test_no_uint64_grid_allocated():
    """The 145x blowup was one array; assert no allocation can hold a per-bit uint64 grid."""
    rng = np.random.default_rng(1)
    count = 1 << 16
    raw = pack.pack_indices(rng.integers(0, 128, size=count, dtype=np.uint64), 7)
    tracemalloc.start()
    tracemalloc.reset_peak()
    pack.unpack_indices(raw, count, 7)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < count * 7 * 8, f"peak {peak} B admits a uint64 bit grid"


def test_deserialize_round_trip_on_a_real_pq_artifact():
    """End to end: the indices a packed tensor decodes to are the ones it was packed with."""
    import gravity_forge as forge

    rng = np.random.default_rng(0)
    weights = rng.standard_normal((256, 128)).astype(np.float32)
    artifact = forge.pack_product_quant(weights, dim=8, subspaces=1, k=128, seed=0, iters=4)
    codes = pack.deserialize(pack.serialize(artifact))
    assert np.array_equal(codes["indices"], artifact.config["pq_codes"]["indices"])
    # torch consumers index with this array, and torch reads uint8 as a boolean mask
    assert codes["indices"].dtype == np.int64
