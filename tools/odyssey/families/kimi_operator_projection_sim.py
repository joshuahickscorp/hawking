"""Internal Gravity representation simulator for a KIMI output-space operator.

This is the bridge shape, not a KIMI artifact. It proves that an operator can
enter the existing Noetic Representation chain (IR, serialization, execution,
accounting, verification, lowering) without editing the representation
compiler. Noetic is a Gravity substrate, not a peer science. Live KIMI
promotion remains owned by the separate fail-closed operator gate.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.odyssey.noetic_compiler import (
    STREAM_BROADCAST_AUX,
    STREAM_WEIGHT_CODES,
    FamilySpec,
    register_family,
)

FAMILY_ID = "kimi_operator_projection_sim"


def pack(W: np.ndarray, direction: np.ndarray, strength: float = 0.5) -> dict[str, Any]:
    W = np.asarray(W, dtype=np.float32)
    d = np.asarray(direction, dtype=np.float32).reshape(-1)
    if W.ndim != 2 or W.shape[0] != d.size:
        raise ValueError("direction width must match operator output width")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        raise ValueError("operator direction has zero norm")
    d = d / norm
    projected = W - np.float32(strength) * np.outer(d, d @ W)
    return {
        "rows": int(W.shape[0]),
        "cols": int(W.shape[1]),
        "strength": float(strength),
        "direction": np.ascontiguousarray(d).tobytes(),
        "weights": np.ascontiguousarray(projected).tobytes(),
    }


def reconstruct(payload: Mapping[str, Any]) -> np.ndarray:
    return np.frombuffer(payload["weights"], dtype="<f4").reshape(
        int(payload["rows"]), int(payload["cols"])
    )


def execute(payload: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    values = reconstruct(payload)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size != values.shape[1]:
        raise ValueError("x length != operator input width")
    return (values @ x).astype(np.float32)


def demo_payload() -> dict[str, Any]:
    rng = np.random.RandomState(23)
    return pack(rng.randn(4, 8).astype(np.float32), rng.randn(4).astype(np.float32))


def bill_parts(payload: Mapping[str, Any]) -> dict[str, list]:
    return {
        "regions": [],
        "generators": [],
        "metadata": [{
            "name": "kimi_operator_direction",
            "bytes": len(payload["direction"]),
            "stream_class": STREAM_BROADCAST_AUX,
        }],
        "tables": [],
        "residuals": [],
        "runtime_auxiliaries": [],
        "representation": [{
            "name": "kimi_operator_projected_weights",
            "bytes": len(payload["weights"]),
            "stream_class": STREAM_WEIGHT_CODES,
        }],
        "model_specific_code": [{
            "name": "kimi_operator_projection_interpreter_stub",
            "bytes": 64,
            "stream_class": STREAM_BROADCAST_AUX,
        }],
    }


register_family(FamilySpec(
    family_id=FAMILY_ID,
    ir_kind="kimi_operator_projection",
    source_path="tools/odyssey/families/kimi_operator_projection_sim.py",
    invoked_symbols=("pack", "execute", "reconstruct"),
    executes=True,
    backend="INTERPRETER",
    backend_kernel=None,
    evidence_tier="FUNCTIONAL_SIM",
    roadmap_overlap=("functional replacement",),
    kernel_requirements=({"requires": "kimi_operator_projection_interpreter"},),
    test_rel="tools/future/test_kimi_noetic_bridge.py",
    pack=pack,
    execute=execute,
    reconstruct=reconstruct,
    demo_payload=demo_payload,
    bill_parts=bill_parts,
))
