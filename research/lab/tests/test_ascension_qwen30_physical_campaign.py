"""Regression coverage for physical Qwen30 Gravity candidate packing."""
from __future__ import annotations

import numpy as np

from lab.operators.ascension_qwen30_physical_campaign import GROUP_SIZE, _pack_factor
from lab.operators.ascension_qwen30_complete_gravity import _pack_binary, _payload_bytes


def test_pack_factor_handles_zero_groups_without_nonfinite_codes() -> None:
    values = np.zeros((2, GROUP_SIZE), dtype=np.float32)
    metadata, payload, restored = _pack_factor(values, bits=3)

    assert metadata["groups"] == 2
    assert metadata["code_bytes"] > 0
    assert len(payload) == metadata["code_bytes"] + metadata["scale_bytes"]
    assert np.isfinite(restored).all()
    assert np.array_equal(restored, values)


def test_complete_binary_bills_retained_fixed_group_tail_bits() -> None:
    """A partial final group is physically stored as a full direct group."""

    values = np.arange(100, dtype=np.float32)
    payload, metrics, restored = _pack_binary(values, [100])

    # 32-byte header + one dimension + one FP16 scale + 128 retained sign
    # bits.  Billing only the 100 mathematical sign bits would undercount the
    # actual artifact and make nonaligned tensors fail mid-campaign.
    assert len(payload) == _payload_bytes([100]) == 54
    assert restored.shape == values.shape
    assert metrics["finite"] is True
