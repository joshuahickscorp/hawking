#!/usr/bin/env python3
"""One fixed source-bound Flash router representation screen.

This is the bounded E03 owner from the post-audit Gravity frontier.  It uses
the sealed layer-0 MLP input and route control, evaluates one deterministic
rowwise-INT4 router candidate, and bills the candidate's scales, metadata, and
shared decoder support.  It is a source-output discriminator, not a router
replacement, complete NR, native runtime, capability, or TPS claim.

The candidate is deliberately not swept.  A route-ID mismatch is a clean
negative for this representation at this source seam; it must not be repaired
by silently changing the source route or by using a second candidate from the
same family.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.flash_router_sensitivity_map import read_bf16
from tools.odyssey.stacked_expert_lowrank_sparse_repair import (
    load_source_activation_bridge,
)

SCHEMA = "hawking.odyssey.flash_router_source_output_screen.v1"
SOURCE_BRIDGE_SCHEMA = "hawking.flash_source_moe_bridge.v1"
DEFAULT_SPEC = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
)
DEFAULT_BRIDGE = ROOT / "receipts/headless/FLASH_NOETIC_LAYER0_SOURCE_MOE_BRIDGE.json"
DEFAULT_OUT = ROOT / "receipts/headless/FLASH_ROUTER_SOURCE_OUTPUT_SCREEN.json"
ROUTER_TENSOR = "model.language_model.layers.0.mlp.gate.weight"
TOP_K = 10
METADATA_BYTES = 256
DECODER_SUPPORT_BYTES = 65536
ROUTE_WEIGHT_TOLERANCE = 2e-3


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(document: dict[str, Any]) -> str:
    body = dict(document)
    body.pop("seal_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 values to the BF16 values actually stored."""
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return (rounded.astype(np.uint32) << 16).view(np.float32)


def _pack_signed_int4(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int16)
    if values.ndim != 2 or values.shape[1] % 2:
        raise ValueError("signed INT4 packing requires an even row width")
    nibbles = (values & 0xF).astype(np.uint8)
    return nibbles[:, 0::2] | (nibbles[:, 1::2] << np.uint8(4))


def rowwise_int4(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic signed-INT4 rows, BF16 scales, and reconstruction."""
    matrix = np.asarray(weights, dtype=np.float32)
    if (
        matrix.ndim != 2
        or not matrix.size
        or matrix.shape[1] % 2
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("router weights must be a finite non-empty 2D matrix")
    scales = np.max(np.abs(matrix), axis=1) / np.float32(7.0)
    scales = _bf16(np.maximum(scales, np.float32(1e-12)))
    quantized = np.clip(np.rint(matrix / scales[:, None]), -8, 7).astype(np.int8)
    reconstructed = quantized.astype(np.float32) * scales[:, None]
    return _pack_signed_int4(quantized), scales, reconstructed


def _bf16_storage_bytes(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.float32)
    return ((values.view(np.uint32) >> np.uint32(16)).astype("<u2")).tobytes()


def stable_topk(values: np.ndarray, k: int = TOP_K) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if k <= 0 or k > values.size:
        raise ValueError("top-k must be within the vector size")
    return np.argsort(-values, kind="stable")[:k]


def stable_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("router logits must be finite")
    shifted = values - np.max(values)
    result = np.exp(shifted, dtype=np.float32)
    result /= np.sum(result, dtype=np.float32)
    return result


def _vector_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError("router outputs have different shapes")
    error = reference - candidate
    ref_norm = max(float(np.linalg.norm(reference)), 1e-30)
    cand_norm = float(np.linalg.norm(candidate))
    return {
        "elements": int(reference.size),
        "relative_l2": float(np.linalg.norm(error) / ref_norm),
        "rmse": float(np.sqrt(np.mean(np.square(error), dtype=np.float64))),
        "max_abs_error": float(np.max(np.abs(error), initial=0.0)),
        "cosine": (
            float(np.dot(reference, candidate) / (ref_norm * cand_norm))
            if cand_norm > 0.0
            else None
        ),
        "finite": bool(np.isfinite(candidate).all()),
        "reference_sha256": _sha256_bytes(reference.astype("<f4", copy=False).tobytes()),
        "candidate_sha256": _sha256_bytes(candidate.astype("<f4", copy=False).tobytes()),
    }


def _router_weights(logits: np.ndarray, ids: np.ndarray) -> np.ndarray:
    probabilities = stable_softmax(logits)
    selected = probabilities[ids].astype(np.float32, copy=True)
    selected /= np.sum(selected, dtype=np.float32)
    return selected


def _source_bridge(path: Path, *, hidden: int) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    bridge_path = path.expanduser().resolve()
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    if not isinstance(bridge, dict) or bridge.get("schema") != SOURCE_BRIDGE_SCHEMA:
        raise ValueError("source bridge schema is not accepted")
    if bridge.get("status") != "PASSED":
        raise ValueError("source bridge is not passed")
    selection = bridge.get("route_selection")
    if not isinstance(selection, dict):
        raise ValueError("source bridge has no route selection")
    raw_ids = selection.get("expert_ids")
    raw_weights = selection.get("selected_weights")
    if not isinstance(raw_ids, list) or not isinstance(raw_weights, list):
        raise ValueError("source bridge route selection is malformed")
    ids = [int(value) for value in raw_ids]
    weights = [float(value) for value in raw_weights]
    if len(ids) != TOP_K or len(weights) != TOP_K or len(set(ids)) != TOP_K:
        raise ValueError("source bridge must contain exactly ten unique route IDs")
    if not all(0 <= value < 512 for value in ids) or not all(math.isfinite(v) for v in weights):
        raise ValueError("source bridge route control is non-finite or out of range")
    if abs(sum(weights) - 1.0) > 1e-5:
        raise ValueError("source bridge route weights do not sum to one")

    # This call validates the source receipt, activation hash, and exact
    # source-layer binding before the router is allowed to teach the screen.
    activation, activation_meta = load_source_activation_bridge(
        bridge_path,
        expected_columns=hidden,
        expert=ids[0],
    )
    return activation, bridge, {
        "bridge_path": str(bridge_path),
        "bridge_sha256": _sha256_bytes(bridge_path.read_bytes()),
        "source_route_ids": ids,
        "source_route_weights": weights,
        "router_logits_sha256": str((selection.get("router_logits") or {}).get("sha256") or ""),
        "activation": activation_meta,
    }


def accounting(*, rows: int, width: int) -> dict[str, Any]:
    if rows <= 0 or width <= 0:
        raise ValueError("router geometry must be positive")
    source_bf16_bytes = int(rows * width * 2)
    int4_payload_bytes = int(rows * ((width + 1) // 2))
    row_scale_bytes = int(rows * 2)  # stored BF16, one exact scale per row
    complete_bytes = int(
        int4_payload_bytes + row_scale_bytes + METADATA_BYTES + DECODER_SUPPORT_BYTES
    )
    denominator = int(rows * width)
    return {
        "rows": int(rows),
        "width": int(width),
        "logical_weights": denominator,
        "source_bf16_bytes": source_bf16_bytes,
        "int4_payload_bytes": int4_payload_bytes,
        "row_scale_bf16_bytes": row_scale_bytes,
        "metadata_bytes": METADATA_BYTES,
        "decoder_support_bytes": DECODER_SUPPORT_BYTES,
        "complete_candidate_bytes": complete_bytes,
        "active_candidate_bytes_excluding_shared_support": int(
            int4_payload_bytes + row_scale_bytes + METADATA_BYTES
        ),
        "complete_scoped_ebpw": complete_bytes * 8.0 / denominator,
        "active_scoped_ebpw_excluding_shared_support": (
            int4_payload_bytes + row_scale_bytes + METADATA_BYTES
        )
        * 8.0
        / denominator,
        "byte_reduction_percent_vs_source_bf16": (
            100.0 * (source_bf16_bytes - complete_bytes) / source_bf16_bytes
        ),
        "accounting_boundary": (
            "router scope only; shared decoder support is billed here and must not be "
            "recounted as learned weights in a whole-body NR"
        ),
    }


def screen_arrays(
    source_router: np.ndarray,
    activation: np.ndarray,
    source_ids: list[int],
    source_weights: list[float],
    *,
    seed: int = 1337,
) -> dict[str, Any]:
    """Evaluate the fixed candidate and a same-budget column-permutation null."""
    source_router = np.asarray(source_router, dtype=np.float32)
    activation = np.asarray(activation, dtype=np.float32).reshape(-1)
    if source_router.ndim != 2 or source_router.shape[1] != activation.size:
        raise ValueError("router and activation geometry mismatch")
    if len(source_ids) != TOP_K or len(source_weights) != TOP_K:
        raise ValueError("source route control must contain ten entries")
    quantized, scales, candidate_router = rowwise_int4(source_router)
    source_logits = source_router @ activation
    candidate_logits = candidate_router @ activation
    candidate_ids = stable_topk(candidate_logits)
    source_ids_array = np.asarray(source_ids, dtype=np.int64)
    recomputed_source_ids = stable_topk(source_logits)
    source_probability_weights = _router_weights(source_logits, recomputed_source_ids)
    candidate_probability_weights = _router_weights(candidate_logits, candidate_ids)
    candidate_weight_on_source_ids = np.asarray(
        [stable_softmax(candidate_logits)[int(i)] for i in source_ids_array],
        dtype=np.float32,
    )
    candidate_weight_on_source_ids /= np.sum(candidate_weight_on_source_ids, dtype=np.float32)

    rng = np.random.default_rng(int(seed))
    column_permutation = rng.permutation(source_router.shape[1])
    null_router = candidate_router[:, column_permutation]
    null_logits = null_router @ activation
    null_ids = stable_topk(null_logits)

    source_set = set(int(i) for i in recomputed_source_ids)
    candidate_set = set(int(i) for i in candidate_ids)
    null_set = set(int(i) for i in null_ids)
    source_weights_from_bridge = np.asarray(source_weights, dtype=np.float32)
    source_weights_from_bridge /= np.sum(source_weights_from_bridge, dtype=np.float32)
    candidate_route_weight_error = _vector_metrics(
        source_weights_from_bridge,
        candidate_weight_on_source_ids,
    )
    route_identity_pass = bool(
        recomputed_source_ids.tolist() == [int(i) for i in source_ids]
        and candidate_ids.tolist() == [int(i) for i in source_ids]
        and candidate_route_weight_error["max_abs_error"] <= ROUTE_WEIGHT_TOLERANCE
    )
    bill = accounting(rows=int(source_router.shape[0]), width=int(source_router.shape[1]))

    def topk_margin(logits: np.ndarray) -> float | None:
        ordered = np.sort(np.asarray(logits, dtype=np.float32))
        if ordered.size <= TOP_K:
            return None
        return float(ordered[-TOP_K] - ordered[-(TOP_K + 1)])

    return {
        "candidate": {
            "family": "rowwise_symmetric_int4_router",
            "quantized_dtype": "INT4_PACKED_SIGNED",
            "scale_dtype": "BF16",
            "scale_policy": "one positive max-abs / 7 scale per router row, stored BF16",
            "seed": int(seed),
            "weight_payload_sha256": _sha256_bytes(quantized.tobytes()),
            "scale_payload_sha256": _sha256_bytes(_bf16_storage_bytes(scales)),
            "source_output": {
                "logits": _vector_metrics(source_logits, candidate_logits),
                "source_ids": recomputed_source_ids.tolist(),
                "candidate_ids": candidate_ids.tolist(),
                "candidate_top10_symmetric_difference": int(len(source_set ^ candidate_set)),
                "candidate_top10_order_positions_changed": int(
                    sum(a != b for a, b in zip(recomputed_source_ids, candidate_ids))
                ),
                "candidate_selected_weights": candidate_probability_weights.tolist(),
                "candidate_weights_aligned_to_source_ids": candidate_weight_on_source_ids.tolist(),
                "source_weights_aligned_to_source_ids": source_weights_from_bridge.tolist(),
                "route_weight_error": candidate_route_weight_error,
                "route_identity_pass": route_identity_pass,
                "route_weight_tolerance_max_abs": ROUTE_WEIGHT_TOLERANCE,
                "source_top10_top11_margin": topk_margin(source_logits),
                "candidate_top10_top11_margin": topk_margin(candidate_logits),
            },
        },
        "matched_null": {
            "family": "same_budget_random_column_address_null",
            "seed": int(seed),
            "column_permutation_sha256": _sha256_bytes(
                np.asarray(column_permutation, dtype="<u4").tobytes()
            ),
            "source_output": {
                "logits": _vector_metrics(source_logits, null_logits),
                "null_ids": null_ids.tolist(),
                "null_top10_symmetric_difference": int(len(source_set ^ null_set)),
                "null_route_identity": null_ids.tolist() == [int(i) for i in source_ids],
            },
            "interpretation": "diagnostic address control only; not a candidate representation",
        },
        "accounting": bill,
        "source_control": {
            "source_router_ids_recomputed": recomputed_source_ids.tolist(),
            "source_router_ids_bridge": [int(i) for i in source_ids],
            "source_ids_match_bridge": recomputed_source_ids.tolist() == [int(i) for i in source_ids],
            "source_route_weights_bridge": source_weights_from_bridge.tolist(),
            "source_route_weight_sum": float(np.sum(source_weights_from_bridge, dtype=np.float32)),
            "source_logits_not_persisted": True,
        },
    }


def run(
    *,
    root: Path = DEFAULT_SPEC,
    bridge_path: Path = DEFAULT_BRIDGE,
    out: Path = DEFAULT_OUT,
    seed: int = 1337,
) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    router, shard, source_payload_bytes_f32 = read_bf16(root.resolve(), ROUTER_TENSOR)
    if router.shape != (512, 2560):
        raise ValueError(f"unexpected Flash router shape: {router.shape}")
    activation, bridge, bridge_meta = _source_bridge(bridge_path, hidden=int(router.shape[1]))
    result = screen_arrays(
        router,
        activation,
        bridge_meta["source_route_ids"],
        bridge_meta["source_route_weights"],
        seed=seed,
    )
    source_ids_match = bool(result["source_control"]["source_ids_match_bridge"])
    candidate = result["candidate"]["source_output"]
    if not source_ids_match:
        status = "SOURCE_CONTROL_RECOMPUTE_MISMATCH__WITHHELD"
    elif candidate["route_identity_pass"]:
        status = "FIXED_INT4_ROUTER_SOURCE_GATE_SURVIVES__REPLICATION_REQUIRED"
    else:
        status = "FIXED_INT4_ROUTER_ROUTE_GATE_REJECTED"
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "artifact_kind": "NR_DIAGNOSTIC",
        "model": "Qwen3.8-Flash-Next",
        "source": {
            "root": str(root.resolve()),
            "revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
            "router_tensor": ROUTER_TENSOR,
            "router_shape": [int(v) for v in router.shape],
            "router_dtype": "BF16",
            "router_source_bf16_bytes": int(router.size * 2),
            "router_payload_bytes_read_after_decode": int(source_payload_bytes_f32),
            "router_shard": shard,
            "router_shard_sha256": _sha256_bytes(Path(shard).read_bytes()),
            "source_activation_bridge": bridge_meta,
            "activation_sha256": str(bridge["mlp_input"]["sha256"]),
            "route_selection_semantics": "softmax(router_logits, float32), stable top-10, normalize selected probabilities",
        },
        "candidate": result["candidate"],
        "matched_null": result["matched_null"],
        "accounting": result["accounting"],
        "gate": {
            "source_control_recomputed_ids_match_bridge": source_ids_match,
            "fixed_candidate_route_identity_pass": bool(candidate["route_identity_pass"]),
            "source_output_gate": "route IDs exact and selected-weight max-abs error <= 0.002",
            "tail_non_router_organs": "WITHHELD; no compatible source-output control was loaded",
        },
        "claim_boundary": (
            "one source-bound layer-0 router candidate only; no tail representation, whole-body "
            "complete EBPW, direct execution, capability, TPS, or promotion claim"
        ),
        "promotion_allowed": False,
        "next_gate": (
            "retain the source router at high fidelity after this fixed INT4 rejection; "
            "reopen only with a materially different router representation or a new disjoint "
            "source-input bank, while keeping tail organs independently gated"
        ),
        "bench": {
            "measurement_state": "MEASURED_SOURCE_OUTPUT_DISCRIMINATOR",
            "elapsed_ns": int(time.perf_counter_ns() - started_ns),
            "machine": "Apple host CPU; bounded source BF16 router and sealed source MLP input",
            "source_model_loaded": False,
            "native_execution": False,
            "complete_token": False,
        },
    }
    document["seal_sha256"] = _canonical_sha256(document)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "complete_scoped_ebpw": result["accounting"]["complete_scoped_ebpw"],
                "source_ids": result["source_control"]["source_router_ids_bridge"],
                "candidate_ids": candidate["candidate_ids"],
                "top10_symmetric_difference": candidate["candidate_top10_symmetric_difference"],
                "source_logit_relative_l2": candidate["logits"]["relative_l2"],
                "elapsed_ns": document["bench"]["elapsed_ns"],
                "out": str(out),
            },
            indent=2,
        )
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    run(root=args.root, bridge_path=args.bridge, out=args.out, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
