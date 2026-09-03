"""Focused coverage for the bounded source-bound Qwen state codec lane."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lab.operators.ascension_qwen_state_kv import (
    CONTEXT_REGIMES,
    HEADER_BYTES,
    StateKVError,
    _artifact_descriptor,
    _write_recovery_manifest,
    codec_storage_bytes,
    codec_suite,
    deserialize_codec,
    deterministic_component_input,
    growing_kv_ledger,
    recurrent_state_ledger,
    serialize_codec,
    verify_recovery_manifest,
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


def test_every_state_codec_can_be_reopened_and_rehydrated_from_its_binary_payload() -> None:
    values = np.linspace(-0.9, 0.7, 257, dtype=np.float32).reshape(1, 257)

    for codec in codec_suite(values):
        rehydrated, parsed = deserialize_codec(serialize_codec(codec), shape=values.shape)
        assert parsed["codec"] == codec.name
        assert np.array_equal(rehydrated, codec.reconstruction)


def test_sealed_recovery_manifest_binds_a_durable_component_artifact(tmp_path) -> None:
    values = np.arange(96, dtype=np.float32).reshape(3, 32) / 10.0
    codec = codec_suite(values)[2]
    payload = serialize_codec(codec)
    artifact = tmp_path / "q4.hkv"
    artifact.write_bytes(payload)
    manifest = tmp_path / "manifest.json"
    _write_recovery_manifest(
        path=manifest,
        model="unit",
        component="state",
        context_tokens=8,
        codec_name=codec.name,
        artifact=_artifact_descriptor(artifact, payload),
        source_values=values,
        rehydrated_values=codec.reconstruction,
    )

    verified = verify_recovery_manifest(manifest)

    assert verified["status"] == "SEALED_DURABLE_ARTIFACT_REOPENED_AND_REHYDRATED"
    assert verified["codec"] == codec.name
    assert CONTEXT_REGIMES == (8, 32, 128)

    artifact.write_bytes(payload + b"tamper")
    with pytest.raises(StateKVError, match="byte/hash binding failed"):
        verify_recovery_manifest(manifest)
