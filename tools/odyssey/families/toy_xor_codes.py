"""Toy representation family: XOR-scrambled group codes.

Exists to prove a NEW family can register end to end without editing
tools/odyssey/noetic_compiler.py or tools/future/complete_ebpw.py.

Not a research candidate. FUNCTIONAL_SIM on a synthetic micro-site.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tools.odyssey.noetic_compiler import (
    STREAM_BROADCAST_AUX,
    STREAM_WEIGHT_CODES,
    FamilySpec,
    register_family,
)

FAMILY_ID = "toy_xor_codes"
KEY = 0xA5
GROUP = 8
THIS = Path(__file__).resolve()
REPO = THIS.parents[3]


def pack(W: np.ndarray, group: int = GROUP, key: int = KEY) -> dict[str, Any]:
    W = np.asarray(W, dtype=np.float32)
    rows, cols = int(W.shape[0]), int(W.shape[1])
    if cols % group != 0:
        raise ValueError(f"cols {cols} not divisible by group {group}")
    gpr = cols // group
    g = W.reshape(rows * gpr, group)
    amax = np.max(np.abs(g), axis=1)
    scales = np.maximum(amax, np.float32(1e-8)).astype(np.float16)
    den = scales.astype(np.float32)[:, None]
    q = np.rint(g / den * 127.0).clip(-127, 127).astype(np.int16)
    stored = ((q + 128).astype(np.uint8) ^ np.uint8(key))
    return {
        "rows": rows,
        "cols": cols,
        "group": group,
        "key": int(key),
        "scales": scales.tobytes(),
        "codes": np.ascontiguousarray(stored).tobytes(),
    }


def _decoded_groups(payload: Mapping[str, Any]) -> tuple[np.ndarray, int, int, int]:
    rows = int(payload["rows"])
    cols = int(payload["cols"])
    group = int(payload["group"])
    key = int(payload["key"])
    gpr = cols // group
    scales = np.frombuffer(payload["scales"], dtype="<f2").astype(np.float32).reshape(rows, gpr)
    stored = np.frombuffer(payload["codes"], dtype=np.uint8).reshape(rows, gpr, group)
    q = (stored ^ np.uint8(key)).astype(np.int16) - 128
    values = q.astype(np.float32) * (scales[:, :, None] / np.float32(127.0))
    return values, rows, cols, group


def execute(payload: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    """Fused decode+FMA. Does not write a (rows × cols) W."""
    values, rows, cols, group = _decoded_groups(payload)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size != cols:
        raise ValueError("x length != cols")
    gpr = cols // group
    xg = x.reshape(gpr, group)
    return (values * xg[None, :, :]).sum(axis=(1, 2)).astype(np.float32)


def reconstruct(payload: Mapping[str, Any]) -> np.ndarray:
    values, rows, cols, _group = _decoded_groups(payload)
    return values.reshape(rows, cols)


def demo_payload() -> dict[str, Any]:
    rng = np.random.RandomState(7)
    W = rng.randn(4, 8).astype(np.float32)
    return pack(W)


def bill_parts(payload: Mapping[str, Any]) -> dict[str, list]:
    return {
        "regions": [],
        "generators": [],
        "metadata": [
            {
                "name": "toy_xor_scales",
                "bytes": len(payload["scales"]),
                "stream_class": STREAM_BROADCAST_AUX,
            }
        ],
        "tables": [
            {
                "name": "xor_key",
                "bytes": 1,
                "stream_class": STREAM_BROADCAST_AUX,
            }
        ],
        "residuals": [],
        "runtime_auxiliaries": [],
        "representation": [
            {
                "name": "toy_xor_codes",
                "bytes": len(payload["codes"]),
                "stream_class": STREAM_WEIGHT_CODES,
            }
        ],
        "model_specific_code": [
            {
                "name": "xor_decoder_stub",
                "bytes": 32,
                "stream_class": STREAM_BROADCAST_AUX,
            }
        ],
    }


register_family(
    FamilySpec(
        family_id=FAMILY_ID,
        ir_kind="toy_xor_codes",
        source_path="tools/odyssey/families/toy_xor_codes.py",
        invoked_symbols=("pack", "execute", "reconstruct"),
        executes=True,
        backend="INTERPRETER",
        backend_kernel=None,
        evidence_tier="FUNCTIONAL_SIM",
        kernel_requirements=({"requires": "toy_xor_decoder", "group": GROUP},),
        test_rel="tools/odyssey/test_noetic_representation_chain.py",
        pack=pack,
        execute=execute,
        reconstruct=reconstruct,
        demo_payload=demo_payload,
        bill_parts=bill_parts,
    )
)
