#!/usr/bin/env python3
"""Symbolically test a simultaneous invariant quotient of HC streams.

E04 is intentionally a cheap Gravity falsifier.  It uses the already
canonical Flash HyperConnection formulas, not a model load or a guessed state
bank, to ask whether a stream-axis quotient ``z = P h`` with rank ``r < m``
could preserve every declared consumer.  The input-dependent read mix has
independent sigmoid coefficients, and the combine path writes each residual
stream directly; together they provide a full-rank observable map.

This receipt is a symbolic/algebraic closure result only.  It does not claim a
numeric source-state replay, EBPW reduction, direct compact runtime, or model
capability result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "hawking.odyssey.hyperconnection_closure_screen.v1"
DEFAULT_OUT = Path("receipts/headless/FLASH_HYPERCONNECTION_SYMBOLIC_CLOSURE_SCREEN.json")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(document: dict[str, Any]) -> str:
    body = dict(document)
    body.pop("seal_sha256", None)
    return _sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def screen_stream_closure(*, streams: int = 4, gate_probe: float = 1.0) -> dict[str, Any]:
    """Solve the small exact coefficient system for the declared HC consumers."""
    started_ns = time.perf_counter_ns()
    streams = int(streams)
    gate_probe = float(gate_probe)
    if streams < 2:
        raise ValueError("HC closure requires at least two streams")
    if not math.isfinite(gate_probe) or gate_probe == 0.0:
        raise ValueError("gate_probe must be finite and non-zero")
    sigma_zero = _sigmoid(0.0)
    sigma_probe = _sigmoid(gate_probe)
    identity = np.eye(streams, dtype=np.float64)
    ones = np.ones((streams, streams), dtype=np.float64)

    # For read_mix, choose one symbolic gate-logit probe per stream: the
    # selected stream has gate sigmoid(gate_probe), all others sigmoid(0).
    # The resulting coefficient rows are exactly the possible stream weights
    # of the input-dependent readout, including the source 1/m mean factor.
    read_mix = (sigma_zero * ones + (sigma_probe - sigma_zero) * identity) / streams
    # combine emits residual[id] directly for every stream, so its Jacobian in
    # the stream axis contains an identity consumer before block injection.
    combine_residual = identity
    joint = np.concatenate([read_mix, combine_residual], axis=0)
    singular_values = np.linalg.svd(joint, compute_uv=False)
    tolerance = max(joint.shape) * np.finfo(np.float64).eps * max(float(singular_values[0]), 1.0)
    joint_rank = int(np.linalg.matrix_rank(joint, tol=tolerance))
    nullity = int(streams - joint_rank)
    read_rank = int(np.linalg.matrix_rank(read_mix, tol=tolerance))
    read_determinant = float(np.linalg.det(read_mix))
    return {
        "status": "SYMBOLIC_FULL_JOINT_RANK__QUOTIENT_REJECTED" if nullity == 0 else "SYMBOLIC_NONTRIVIAL_QUOTIENT_SURVIVES",
        "geometry": {
            "source_streams_m": streams,
            "tested_quotient_condition": "r < m",
            "gate_probe": gate_probe,
        },
        "consumer_system": {
            "read_mix_formula": "y[h] = (1/m) * sum_s sigmoid(g_s[h]) * normalized[s,h]",
            "combine_formula": "output[s,h] = residual[s,h] + block_output[h] * 2*sigmoid(block_logits[s]/divisor)",
            "read_mix_probe_matrix": read_mix.tolist(),
            "combine_residual_stream_jacobian": combine_residual.tolist(),
            "joint_observable_matrix": joint.tolist(),
            "input_dependent_generators_included": ["read_mix sigmoid gate coefficients", "combine block-logit stream gates"],
        },
        "rank_solution": {
            "read_mix_rank": read_rank,
            "joint_observable_rank": joint_rank,
            "joint_observable_nullity": nullity,
            "singular_values": singular_values.tolist(),
            "numeric_tolerance": tolerance,
            "read_mix_determinant": read_determinant,
            "nontrivial_r_less_than_m_possible": bool(nullity > 0),
            "falsifier": (
                "The direct residual-stream identity consumer forces full stream-axis rank; no nontrivial simultaneous invariant quotient can preserve the declared consumers."
                if nullity == 0
                else "A nontrivial null space remains in this symbolic consumer set; require independent source-state validation before claiming closure."
            ),
        },
        "source_bindings": {
            "formula_owner": "crates/hawking-core/shaders/qwen_next.metal",
            "read_mix_kernel": "qwen_next_hyperconnection_read_mix",
            "combine_kernel": "qwen_next_hyperconnection_combine",
            "rust_dispatch_owner": "crates/hawking-core/src/kernels/mod.rs",
            "model_load": False,
            "source_state_replay": "NOT_PERFORMED",
        },
        "direct_execution": {
            "status": "NOT_APPLICABLE_SYMBOLIC_ONLY",
            "compact_quotient_candidate": "WITHHELD_AFTER_FULL_RANK_FALSIFIER",
        },
        "claim_boundary": (
            "This is a symbolic HC stream-axis closure falsifier using the canonical read-mix and combine formulas. "
            "It establishes full joint observable rank for the declared stream consumers at the algebraic probe, "
            "but it does not establish numeric source-state parity, a model representation, EBPW, physical latency, "
            "capability, or promotion."
        ),
        "elapsed_ns": int(time.perf_counter_ns() - started_ns),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--gate-probe", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    started = datetime.now(timezone.utc)
    result = screen_stream_closure(streams=args.streams, gate_probe=args.gate_probe)
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": started.isoformat(),
        **result,
    }
    document["seal_sha256"] = _canonical_sha256(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": document["status"],
                "streams": args.streams,
                "joint_rank": result["rank_solution"]["joint_observable_rank"],
                "nullity": result["rank_solution"]["joint_observable_nullity"],
                "elapsed_ns": result["elapsed_ns"],
                "output": str(args.output),
                "seal": document["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
