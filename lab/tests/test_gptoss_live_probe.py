from __future__ import annotations

import numpy as np

from lab.operators.gptoss_live_probe import (
    DEFAULT_MANIFEST,
    apply_gate,
    decode_mxfp4_groups_bf16,
    live_probe,
)


def test_mxfp4_nibbles_decode_in_serialized_order() -> None:
    blocks = np.array([[[0x21, 0x43]]], dtype=np.uint8)
    scales = np.array([[127]], dtype=np.uint8)
    bits = decode_mxfp4_groups_bf16(blocks, scales)
    values = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    np.testing.assert_array_equal(values, np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32))


def test_gptoss_gate_is_interleaved_clamped_and_finite() -> None:
    values = np.array([[-8.0, -8.0, 2.0, 3.0]], dtype=np.float32)
    actual = apply_gate(values)
    assert actual.shape == (1, 2)
    assert np.isfinite(actual).all()
    assert actual[0, 0] > 0.0
    assert actual[0, 1] > actual[0, 0]


def test_live_source_probe_when_admitted() -> None:
    result = live_probe(DEFAULT_MANIFEST, block=0)
    if result["status"] == "SOURCE_ABSENT":
        return
    assert result["status"] == "PASS_BOUNDED_SOURCE_EXPERT_WAVE"
    assert result["selected_experts"] == [83, 86, 103, 73]
    assert result["output_sha256"] == "717629f67010b3c0529242948c66ac8c95d51b6a0f26df2016f6317be116d971"
    assert result["output_finite"] is True
