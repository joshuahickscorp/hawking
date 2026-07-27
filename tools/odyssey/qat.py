#!/usr/bin/env python3.12
"""QAT simulator interface + honestly-named fake quantizer.

This module never measures a parent model. The FakeUniformQuantizer is labelled
fake in its class name, its schema, and every result it returns. Calling code
must not re-label its outputs as measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

SCHEMA = "hawking.odyssey.qat_sim.interface.v1"
FAKE_SCHEMA = "hawking.odyssey.qat_sim.fake_uniform.v1"


@runtime_checkable
class Quantizer(Protocol):
    """Interface only. Implementations must declare whether they are fake."""

    name: str
    is_fake: bool

    def fake_quantize(self, weights: np.ndarray, *, bits: int = 4) -> "FakeQuantResult":
        ...


@dataclass(frozen=True)
class FakeQuantResult:
    """Result of a fake quantize step. kind is always 'fake_simulation' for fakes."""

    kind: str
    quantizer_name: str
    bits: int
    shape: list[int]
    scale: float
    reconstruction_mse: float
    dequantized: np.ndarray
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": FAKE_SCHEMA if self.kind == "fake_simulation" else SCHEMA,
            "kind": self.kind,
            "quantizer_name": self.quantizer_name,
            "is_measurement": False if self.kind == "fake_simulation" else None,
            "bits": self.bits,
            "shape": self.shape,
            "scale": self.scale,
            "reconstruction_mse": self.reconstruction_mse,
            "note": self.note,
            "honesty": (
                "This is a FAKE quantizer result on a fixture tensor. "
                "It is not a QAT measurement of any parent model."
            ),
        }


class FakeUniformQuantizer:
    """Honestly named fake: symmetric uniform fake-quant on small tensors.

    Not a real QAT trainer. Not a measurement of Math-Preserve or any substrate.
    """

    name = "FakeUniformQuantizer"
    is_fake = True

    def fake_quantize(self, weights: np.ndarray, *, bits: int = 4) -> FakeQuantResult:
        w = np.asarray(weights, dtype=np.float32)
        if bits < 2 or bits > 16:
            raise ValueError(f"bits must be in [2,16], got {bits}")
        qmax = (1 << (bits - 1)) - 1
        scale = float(np.max(np.abs(w)) / qmax) if qmax else 1.0
        if scale == 0.0:
            scale = 1.0
        q = np.clip(np.round(w / scale), -qmax - 1, qmax)
        deq = (q * scale).astype(np.float32)
        mse = float(np.mean((w - deq) ** 2))
        return FakeQuantResult(
            kind="fake_simulation",
            quantizer_name=self.name,
            bits=bits,
            shape=list(w.shape),
            scale=scale,
            reconstruction_mse=mse,
            dequantized=deq,
            note=(
                "FIXTURE fake-quant operator simulation. "
                "Do not report reconstruction_mse as a parent-model measurement."
            ),
        )


def get_default_fake_quantizer() -> FakeUniformQuantizer:
    return FakeUniformQuantizer()


def simulate_qat_step(
    weights: np.ndarray | None = None,
    *,
    bits: int = 4,
    quantizer: Quantizer | None = None,
) -> dict[str, Any]:
    """Run one fake QAT operator step. Always returns kind=fake_simulation for the default."""
    q = quantizer or get_default_fake_quantizer()
    if weights is None:
        rng = np.random.RandomState(0)
        weights = rng.randn(32, 32).astype(np.float32)
    result = q.fake_quantize(np.asarray(weights, dtype=np.float32), bits=bits)
    out = result.as_dict()
    out["interface_schema"] = SCHEMA
    out["quantizer_is_fake"] = bool(getattr(q, "is_fake", False))
    if out["quantizer_is_fake"]:
        out["is_measurement"] = False
    return out
