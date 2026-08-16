#!/usr/bin/env python3
"""Numpy/CPU oracle for the Q80 mixed-decode kernels.

Grades execution against what the packed artifact encodes, never the BF16
parent. Dense W is built only as a *parity* reconstruction of the codec, and
the matvec oracle is the same register-style consumption the kernels use:

  gate  : binary_group serial matvec
  up    : binary_group serial matvec + rice_q1 residual in index order
  down  : y = L @ (R @ x) of packed 3-bit factors (no dense W)

This module is the pack-lane / kernel-lane shared format authority for the
bodies. Pack must emit HGRAVB01 / HGRAVR02 / HGRAVS01; kernels consume the
bodies after the Gravity container header.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from lab.operators.ascension_dual_gravity_worker import (
    GROUP_BINARY,
    GROUP_UNIFORM,
    _binary_codec,
    _uniform_codec,
)
from lab.operators.residual_compact_codec import encode_residual_compact


def binary_group_matvec(W: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    codec = _binary_codec(W, group_size=GROUP_BINARY)
    y = np.asarray(codec.reconstruction, dtype=np.float32) @ np.asarray(x, dtype=np.float32)
    return y.astype(np.float32), dict(codec.metadata)


def binary_rice_q1_matvec(
    W: np.ndarray, x: np.ndarray, *, outlier_ratio: float = 0.02
) -> tuple[np.ndarray, dict[str, Any]]:
    codec = encode_residual_compact(
        W,
        outlier_ratio=outlier_ratio,
        group_size=GROUP_BINARY,
        index_mode="rice",
        value_bits=1,
        value_scale="rms",
    )
    y = np.asarray(codec.reconstruction, dtype=np.float32) @ np.asarray(x, dtype=np.float32)
    return y.astype(np.float32), dict(codec.metadata)


def hgravs01_two_stage_matvec(
    L: np.ndarray, R: np.ndarray, x: np.ndarray, *, bits: int = 3
) -> tuple[np.ndarray, dict[str, Any]]:
    left = _uniform_codec(L, bits=bits, group_size=GROUP_UNIFORM)
    right = _uniform_codec(R, bits=bits, group_size=GROUP_UNIFORM)
    mid = np.asarray(right.reconstruction, dtype=np.float32) @ np.asarray(x, dtype=np.float32)
    y = np.asarray(left.reconstruction, dtype=np.float32) @ mid
    return y.astype(np.float32), {
        "left": dict(left.metadata),
        "right": dict(right.metadata),
        "execution": "two_stage_L_R_x",
        "dense_W_formed": False,
    }


def _self_check() -> None:
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 128), dtype=np.float32)
    x = rng.standard_normal(128, dtype=np.float32)
    y_bin, _ = binary_group_matvec(w, x)
    y_up, meta = binary_rice_q1_matvec(w, x, outlier_ratio=0.02)
    assert y_bin.shape == (8,) and np.isfinite(y_bin).all()
    assert y_up.shape == (8,) and np.isfinite(y_up).all()
    assert meta["index_mode"] == "rice" and int(meta["value_bits"]) == 1
    L = rng.standard_normal((16, 8), dtype=np.float32)
    R = rng.standard_normal((8, 32), dtype=np.float32)
    x2 = rng.standard_normal(32, dtype=np.float32)
    y_dn, dn = hgravs01_two_stage_matvec(L, R, x2, bits=3)
    assert y_dn.shape == (16,) and np.isfinite(y_dn).all()
    assert dn["dense_W_formed"] is False
    _ = math.prod


if __name__ == "__main__":
    _self_check()
    print("q80_mixed_decode_oracle self-check passed")
