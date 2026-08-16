"""Compact residual codec: same selection/math, cheaper storage."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lab.operators.ascension_dual_gravity_worker import (
    GROUP_BINARY,
    _binary_codec,
    _parse_container,
    _residual_codec,
)
from lab.operators.doctor6.rungs import quant_residual, quant_residual_compact
from lab.operators.residual_compact_codec import (
    MAGIC_RESIDUAL_COMPACT,
    _best_rice_k,
    _pack_rice,
    _unpack_rice,
    decode_residual_compact,
    encode_residual_compact,
    select_outlier_indices,
)


def _shaped(seed: int = 7, shape=(64, 96)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape, dtype=np.float32)


def test_quant_residual_signature_still_uses_the_48bit_incumbent() -> None:
    values = _shaped()
    rec, nbytes = quant_residual(values, outlier_ratio=0.02)
    incumbent = _residual_codec(values, outlier_ratio=0.02, group_size=GROUP_BINARY)
    np.testing.assert_array_equal(rec, incumbent.reconstruction.astype(np.float32))
    assert nbytes == len(incumbent.payload)


def test_selection_matches_incumbent_top_k_by_abs_residual() -> None:
    values = _shaped(11, (32, 40))
    incumbent = _residual_codec(values, outlier_ratio=0.03, group_size=GROUP_BINARY)
    binary = _binary_codec(values, group_size=GROUP_BINARY).reconstruction.reshape(-1)
    residual = values.reshape(-1) - binary
    changed = np.flatnonzero(incumbent.reconstruction.reshape(-1) != binary)
    selected, count = select_outlier_indices(residual, 0.03)
    assert count == max(1, int(math.ceil(values.size * 0.03)))
    np.testing.assert_array_equal(np.sort(changed.astype(np.uint32)), selected)


@pytest.mark.parametrize("index_mode", ("rice", "group_local", "bitmap"))
def test_fp16_compact_reconstruction_matches_incumbent(index_mode: str) -> None:
    values = _shaped(19, (48, 80))
    incumbent = _residual_codec(values, outlier_ratio=0.02, group_size=GROUP_BINARY)
    compact = encode_residual_compact(
        values, outlier_ratio=0.02, index_mode=index_mode, value_bits=16
    )
    np.testing.assert_array_equal(compact.reconstruction, incumbent.reconstruction)
    decoded = decode_residual_compact(compact.payload)
    np.testing.assert_array_equal(decoded, compact.reconstruction)
    header, body = _parse_container(compact.payload, expected_magic=MAGIC_RESIDUAL_COMPACT)
    assert header["schema"] == compact.metadata["schema"]
    assert header["index_mode"] == index_mode
    assert len(body) == (
        header["scale_bytes"]
        + header["sign_bytes"]
        + header["index_bytes"]
        + header["residual_scale_bytes"]
        + header["residual_bytes"]
    )
    if index_mode != "bitmap":
        # Bitmap loses at ~2% on a small tensor; rice / group-local must not.
        assert len(compact.payload) < len(incumbent.payload)


def test_rice_pack_roundtrip_and_k_choice() -> None:
    diffs = np.asarray([1, 2, 3, 8, 50, 64, 200, 1], dtype=np.uint64)
    k = _best_rice_k(diffs)
    packed = _pack_rice(diffs, k)
    unpacked = _unpack_rice(packed, diffs.size, k)
    np.testing.assert_array_equal(unpacked.astype(np.uint64), diffs)
    empty = _pack_rice(np.zeros(0, dtype=np.uint64), 3)
    assert empty == b""
    np.testing.assert_array_equal(_unpack_rice(b"", 0, 3), np.zeros(0, dtype=np.uint32))


def test_single_outlier_rice_encodes() -> None:
    values = np.zeros((8, 8), dtype=np.float32)
    values[0, 3] = 4.0
    compact = encode_residual_compact(
        values, outlier_ratio=0.01, index_mode="rice", value_bits=16
    )
    assert compact.metadata["outlier_count"] == 1
    decoded = decode_residual_compact(compact.payload)
    np.testing.assert_array_equal(decoded, compact.reconstruction)


@pytest.mark.parametrize("value_bits,value_scale", ((8, "absmax"), (4, "absmax"), (1, "mean_abs"), (1, "rms")))
def test_value_quantized_decode_matches_encode(value_bits: int, value_scale: str) -> None:
    values = _shaped(23, (40, 64))
    compact = encode_residual_compact(
        values,
        outlier_ratio=0.025,
        index_mode="rice",
        value_bits=value_bits,
        value_scale=value_scale,
    )
    decoded = decode_residual_compact(compact.payload)
    np.testing.assert_allclose(decoded, compact.reconstruction, rtol=0, atol=0)
    assert compact.metadata["value_bits"] == value_bits
    # Positions stay the incumbent top-k; only the stored value may change.
    incumbent = _residual_codec(values, outlier_ratio=0.025, group_size=GROUP_BINARY)
    binary = _binary_codec(values, group_size=GROUP_BINARY).reconstruction.reshape(-1)
    inc_pos = np.flatnonzero(incumbent.reconstruction.reshape(-1) != binary)
    new_pos = np.flatnonzero(compact.reconstruction.reshape(-1) != binary)
    np.testing.assert_array_equal(np.sort(inc_pos), np.sort(new_pos))


def test_quant_residual_compact_default_is_rice_one_bit() -> None:
    values = _shaped(9, (64, 128))
    rec, nbytes = quant_residual_compact(values, outlier_ratio=0.02)
    compact = encode_residual_compact(
        values, outlier_ratio=0.02, index_mode="rice", value_bits=1, value_scale="rms"
    )
    np.testing.assert_array_equal(rec, compact.reconstruction)
    assert nbytes == len(compact.payload)
    assert compact.metadata["value_bits"] == 1
    assert compact.metadata["value_scale"] == "rms"


def test_quant_residual_compact_wrapper_and_rice_q4_beats_legacy_size() -> None:
    values = _shaped(3, (128, 256))
    rec, nbytes = quant_residual_compact(
        values, outlier_ratio=0.02, index_mode="rice", value_bits=4
    )
    legacy_rec, legacy_n = quant_residual(values, outlier_ratio=0.02)
    assert rec.shape == values.shape
    assert nbytes < legacy_n
    count = max(1, int(math.ceil(values.size * 0.02)))
    binary_n = len(_binary_codec(values, group_size=GROUP_BINARY).payload)
    bits_per_outlier = 8.0 * (nbytes - binary_n) / count
    # Tiny tensors pay header overhead; still far below the 48-bit incumbent.
    legacy_bpo = 8.0 * (legacy_n - binary_n) / count
    assert bits_per_outlier < 20.0
    assert bits_per_outlier < legacy_bpo / 2.0


def test_group_local_and_bitmap_payloads_decode() -> None:
    values = _shaped(5, (96, 96))
    for mode in ("group_local", "bitmap"):
        compact = encode_residual_compact(
            values, outlier_ratio=0.015, index_mode=mode, value_bits=6
        )
        np.testing.assert_array_equal(
            decode_residual_compact(compact.payload), compact.reconstruction
        )
