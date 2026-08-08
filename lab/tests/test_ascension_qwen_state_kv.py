"""Focused coverage for the bounded source-bound Qwen state codec lane."""
from __future__ import annotations

import math

import numpy as np

from lab.operators.ascension_qwen_state_kv import (
    HEADER_BYTES,
    codec_storage_bytes,
    codec_suite,
    deterministic_component_input,
    growing_kv_ledger,
    recurrent_state_ledger,
    serialize_codec,
)


def test_state_codecs_materialize_and_reconstruct_a_non_aligned_component_array() -> None:
    values = np.linspace(-1.0, 1.0, 513, dtype=np.float32).reshape(27, 19)
    codecs = codec_suite(values)

    assert [codec.name for codec in codecs] == [
        "fp16_reference",
        "q8_group64",
        "q4_group64",
        "protected_residual_q4_group64_top1pct_fp16",
    ]
    for codec in codecs:
        payload = serialize_codec(codec)
        assert codec.reconstruction.shape == values.shape
        assert np.isfinite(codec.reconstruction).all()
        assert len(payload) == codec_storage_bytes(codec.name, elements=values.size)
        assert len(payload) == HEADER_BYTES + len(codec.body)

    q4 = codecs[2]
    protected = codecs[3]
    assert protected.residual_count == math.ceil(values.size * 0.01)
    assert np.linalg.norm(values - protected.reconstruction) <= np.linalg.norm(values - q4.reconstruction)


def test_deterministic_component_inputs_are_non_linguistic_and_reproducible() -> None:
    first = deterministic_component_input(8, 32, label="unit-state")
    second = deterministic_component_input(8, 32, label="unit-state")
    distinct = deterministic_component_input(8, 32, label="other-state")

    assert first.shape == (8, 32)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, distinct)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)


def test_qwen30_exact_kv_geometry_ledger_counts_per_layer_artifact_headers() -> None:
    ledger = growing_kv_ledger(
        layer_count=48,
        key_value_heads=4,
        head_dim=128,
        session_tokens=8,
    )

    assert ledger["values_per_layer_per_token"] == 1024
    assert ledger["values_per_layer_session"] == 8192
    assert ledger["values_per_session"] == 393216
    assert ledger["codecs"]["fp16_reference"]["bytes_per_session"] == 48 * (HEADER_BYTES + 8192 * 2)
    assert (
        ledger["codecs"]["q4_group64"]["bytes_per_session"]
        < ledger["codecs"]["q8_group64"]["bytes_per_session"]
        < ledger["codecs"]["fp16_reference"]["bytes_per_session"]
    )


def test_qwen80_recurrent_geometry_is_fixed_per_session_not_a_growing_kv_cache() -> None:
    ledger = recurrent_state_ledger(
        layer_count=36,
        heads=32,
        key_dim=128,
        value_dim=128,
        session_tokens=8,
    )

    assert ledger["values_per_layer"] == 32 * 128 * 128
    assert ledger["values_per_session_resident_state"] == 36 * 32 * 128 * 128
    for codec in ledger["codecs"].values():
        assert codec["growth_bytes_per_additional_token"] == 0
        assert codec["bytes_per_session_resident_state"] > 0
