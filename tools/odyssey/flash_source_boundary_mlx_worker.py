#!/usr/bin/env python3
"""Isolated MLX research worker for the predeclared Flash token-0 boundary.

This worker deliberately reuses the already captured provider's exact forward
function and intercepts its evaluated seams.  It is not independently
authoritative: the Hawking wrapper must first admit the existing V2 teacher and
must verify/package every payload emitted here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hawking.persist import atomic_write_bytes, atomic_write_json  # noqa: E402
from tools.odyssey import flash_external_greedy_reference_mlx_worker as source  # noqa: E402
from tools.odyssey import flash_source_boundary_preflight as contract  # noqa: E402


SCHEMA = "hawking.flash.source_boundary_mlx_worker.v1"
STATUS = "CAPTURED_EXTERNAL_SOURCE_LAYERS_0_THROUGH_4_TOKEN_0"


class _BoundaryComplete(Exception):
    """Internal control-flow marker: the requested layer-4 output is sealed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _capture_array(
    out_dir: Path,
    mx: Any,
    np: Any,
    *,
    ordinal: int,
    layer: int | None,
    stage: str,
    value: Any,
    file_prefix: str = "seam",
    host_tile: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    original_dtype = str(value.dtype)
    mx.eval(value)
    array = np.asarray(value.astype(mx.float32), dtype=np.float32)
    if host_tile is not None:
        array = np.tile(array, host_tile)
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"source boundary seam is non-finite: {stage}")
    raw = array.astype("<f4", copy=False).tobytes(order="C")
    payload_path = out_dir / f"{file_prefix}-{ordinal:02d}-{stage}.f32"
    atomic_write_bytes(payload_path, raw)
    return {
        "ordinal": ordinal,
        "layer": layer,
        "stage": stage,
        "original_dtype": original_dtype,
        "shape": list(array.shape),
        "payload": {
            "path": str(payload_path),
            "sha256": _sha256(raw),
            "dtype": "F32_LE",
            "elements": int(array.size),
            "bytes": len(raw),
        },
    }


_LAYER4_CACHE_ROLES = ("conv_state", "recurrent_state")
_IMPLICIT_ZERO_INITIALIZATION = "IMPLICIT_SOURCE_ZERO_STATE_MATERIALIZED_BY_CAPTURE"
_POST_ATTENTION_INITIALIZATION = "PROVIDER_MATERIALIZED_AFTER_ATTENTION"


def _implicit_linear_cache_values(
    mx: Any,
    cache_entry: Any,
    attention_input: Any,
    linear_attention: Any,
) -> list[tuple[str, Any]]:
    """Materialize only the provider-defined meaning of an empty GDN cache.

    Qwen4-Exp delegates its linear-attention implementation to
    ``Qwen3_5GatedDeltaNet``.  On its first token that implementation treats
    ``cache[0] is None`` as a zero convolution window and ``cache[1] is None``
    as a zero recurrent state.  This helper expresses those exact source
    formulas for observation; it does not mutate the provider cache or invent
    a native representation.
    """
    state = getattr(cache_entry, "state", None)
    if not isinstance(state, (list, tuple)) or len(state) != len(_LAYER4_CACHE_ROLES):
        raise ValueError("layer-4 source cache does not expose the expected two-slot GDN state")
    if any(value is not None for value in state):
        raise ValueError("implicit layer-4 source cache materialization requires an empty provider cache")
    shape = getattr(attention_input, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 3:
        raise ValueError("layer-4 attention input does not have [batch, token, hidden] shape")
    batch = int(shape[0])
    if batch <= 0:
        raise ValueError("layer-4 attention input batch is empty")
    try:
        conv_kernel = int(linear_attention.conv_kernel_size)
        conv_dim = int(linear_attention.conv_dim)
        value_heads = int(linear_attention.num_v_heads)
        value_dim = int(linear_attention.head_v_dim)
        key_dim = int(linear_attention.head_k_dim)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("layer-4 source linear-attention layout is incomplete") from exc
    if min(conv_kernel, conv_dim, value_heads, value_dim, key_dim) <= 0 or conv_kernel < 2:
        raise ValueError("layer-4 source linear-attention layout is invalid")
    return [
        (
            "conv_state",
            mx.zeros(
                (batch, conv_kernel - 1, conv_dim),
                dtype=attention_input.dtype,
            ),
        ),
        (
            "recurrent_state",
            mx.zeros(
                (batch, value_heads, value_dim, key_dim),
                dtype=mx.float32,
            ),
        ),
    ]


def _capture_layer4_linear_cache(
    out_dir: Path,
    mx: Any,
    np: Any,
    *,
    cache_entry: Any,
    boundary: str,
    attention_input: Any,
    linear_attention: Any,
) -> list[dict[str, Any]]:
    """Capture the closed two-slot Qwen GDN cache at one named boundary."""
    state = getattr(cache_entry, "state", None)
    if not isinstance(state, (list, tuple)) or len(state) != len(_LAYER4_CACHE_ROLES):
        raise ValueError("layer-4 source cache does not expose the expected two-slot GDN state")
    if all(value is None for value in state):
        if boundary != "pre-attention":
            raise ValueError("post-attention layer-4 source cache remained unmaterialized")
        entries = _implicit_linear_cache_values(mx, cache_entry, attention_input, linear_attention)
        initialization = _IMPLICIT_ZERO_INITIALIZATION
        cache_path_prefix = "implicit_source_cache"
    else:
        if boundary != "post-attention" or any(value is None for value in state):
            raise ValueError(f"layer-4 {boundary} source cache is only partially materialized")
        entries = list(zip(_LAYER4_CACHE_ROLES, state, strict=True))
        initialization = _POST_ATTENTION_INITIALIZATION
        cache_path_prefix = "provider_cache"

    records: list[dict[str, Any]] = []
    for index, (state_role, value) in enumerate(entries):
        if value is None or not hasattr(value, "shape"):
            raise ValueError(f"layer-4 {boundary} {state_role} cache is absent")
        original_dtype = str(value.dtype)
        mx.eval(value)
        array = np.asarray(value.astype(mx.float32), dtype=np.float32)
        if not bool(np.isfinite(array).all()):
            raise ValueError(f"layer-4 {boundary} {state_role} cache contains a non-finite value")
        raw = array.astype("<f4", copy=False).tobytes(order="C")
        path = out_dir / f"cache-layer4-{boundary}-{index:02d}.f32"
        atomic_write_bytes(path, raw)
        records.append({
            "state_role": state_role,
            "path_in_provider_cache": f"{cache_path_prefix}[{index}]",
            "boundary": boundary,
            "initialization": initialization,
            "original_dtype": original_dtype,
            "shape": list(array.shape),
            "payload": {
                "path": str(path),
                "sha256": _sha256(raw),
                "dtype": "F32_LE",
                "elements": int(array.size),
                "bytes": len(raw),
            },
        })
    return records


def run(
    root: Path,
    out_dir: Path,
    *,
    token_id: int,
    reference_sha256: str,
    authorization_sha256: str,
    preflight_sha256: str,
) -> dict[str, Any]:
    import mlx.core as mx
    import numpy as np
    from mlx import nn
    from mlx.utils import tree_flatten
    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.models.qwen4_exp.language import _QWEN4_BATCH_INVARIANT_FORWARD
    from mlx_vlm.utils import load_model

    root = root.resolve(strict=True)
    out_dir = out_dir.resolve(strict=True)
    manifest_path = out_dir / "source-boundary-manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite source boundary manifest: {manifest_path}")
    started = time.time_ns()
    progress: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING_EXTERNAL_SOURCE_BOUNDARY_CAPTURE",
        "root": str(root),
        "token_index": 0,
        "token_id": token_id,
        "reference_contract_sha256": reference_sha256,
        "owner_authorization_sha256": authorization_sha256,
        "preflight_sha256": preflight_sha256,
        "started_unix_ns": started,
        "claim_boundary": "in-flight external provider capture; no Hawking source-parity verdict",
    }
    atomic_write_json(manifest_path, progress)

    model = load_model(root, lazy=True, strict=True)
    language_model = model.language_model
    ple_storage = source._patch_row_addressed_ple(language_model, root, out_dir, mx, np)
    expert_store = source._patch_raw_source_experts(language_model, root, mx, np)
    cache = make_prompt_cache(language_model)
    if [index for index, layer in enumerate(language_model.model.layers) if hasattr(layer, "ple")] != [1]:
        raise ValueError("source PLE layer identity differs from the predeclared boundary contract")

    seams: list[dict[str, Any]] = []
    layer0_trace: list[dict[str, Any]] = []
    cache_records: dict[str, list[dict[str, Any]]] = {}
    route: dict[str, Any] | None = None
    original_eval_stage = source._eval_stage
    trace_expected = contract._required_layer0_trace()
    invariant = _QWEN4_BATCH_INVARIANT_FORWARD

    def capture_seam(
        mx_arg: Any,
        tree_flatten_arg: Any,
        outputs: Any,
        cache_arg: list[Any],
        **kwargs: Any,
    ) -> None:
        nonlocal route
        original_eval_stage(mx_arg, tree_flatten_arg, outputs, cache_arg, **kwargs)
        layer = kwargs["layer_index"]
        stage = kwargs["stage"]
        expected = contract._required_seams()

        def save(
            ordinal: int,
            value: Any,
            *,
            host_tile: tuple[int, ...] | None = None,
        ) -> None:
            declaration = expected[ordinal]
            record = _capture_array(
                out_dir,
                mx_arg,
                np,
                ordinal=ordinal,
                layer=declaration["layer"],
                stage=declaration["stage"],
                value=value,
                host_tile=host_tile,
            )
            if record["shape"] != declaration["shape"]:
                raise ValueError(
                    f"source seam shape differs at ordinal {ordinal}: "
                    f"{record['shape']} != {declaration['shape']}"
                )
            seams.append(record)

        def save_layer0_trace(ordinal: int, value: Any) -> None:
            declaration = trace_expected[ordinal]
            record = _capture_array(
                out_dir,
                mx_arg,
                np,
                ordinal=ordinal,
                layer=declaration["layer"],
                stage=declaration["stage"],
                value=value,
                file_prefix="layer0-trace",
            )
            if record["shape"] != declaration["shape"]:
                raise ValueError(
                    f"source layer-0 trace shape differs at ordinal {ordinal}: "
                    f"{record['shape']} != {declaration['shape']}"
                )
            layer0_trace.append(record)

        def capture_layer0_attention_precision_trace(values: Any) -> None:
            if not isinstance(values, tuple) or len(values) != 3:
                raise ValueError("source layer-0 attention HyperConnection did not return its declared triple")
            actual_mixed, hyper_input, injection_weights = values
            module = language_model.model.layers[0].attn_hyper_connection
            normed = module.hc_norm(hyper_input)
            down_projection = invariant._linear(module.input_mix_weight_down, normed)
            low_rank_activation = nn.silu(down_projection / module.hc_count)
            up_projection = invariant._linear(module.input_mix_weight_up, low_rank_activation)
            mix_weights = mx.sigmoid(up_projection)
            mix_streams = mix_weights.reshape(
                *mix_weights.shape[:-1], module.hc_count, module.hidden_size
            )
            normalized_streams = normed.reshape(
                *normed.shape[:-1], module.hc_count, module.hidden_size
            )
            recomputed_mixed = mx.mean(mix_streams * normalized_streams, axis=-2)
            mx_arg.eval(actual_mixed, recomputed_mixed)
            actual_f32 = np.asarray(actual_mixed.astype(mx_arg.float32), dtype=np.float32)
            recomputed_f32 = np.asarray(recomputed_mixed.astype(mx_arg.float32), dtype=np.float32)
            if actual_f32.shape != recomputed_f32.shape or not np.array_equal(actual_f32, recomputed_f32):
                raise ValueError(
                    "source layer-0 attention HyperConnection precision trace did not reproduce the provider mixed output"
                )
            save_layer0_trace(0, normed)
            save_layer0_trace(1, down_projection)
            save_layer0_trace(2, low_rank_activation)
            save_layer0_trace(3, up_projection)
            save_layer0_trace(4, mix_weights)
            save_layer0_trace(5, actual_mixed)
            save_layer0_trace(6, injection_weights)

        if layer == 0 and stage == "attention_hyper_connection":
            capture_layer0_attention_precision_trace(outputs)
        elif layer == 0 and stage == "attention":
            save_layer0_trace(7, outputs)
        elif layer == 0 and stage == "attention_injection":
            save_layer0_trace(8, outputs)

        if layer is None and stage == "embedding":
            # The embedding seam is the source hidden row repeated across HC
            # streams. The stage above has already evaluated ``outputs`` and
            # cleared MLX's cache; asking Metal to evaluate a second lazy tile
            # here can rebuild the large embedding graph and hit the macOS
            # command-buffer watchdog. Repeat the captured F32 row on host;
            # this is value-identical to the provider's tile and does not alter
            # the provider forward below, which still executes its real MLX
            # tile for downstream math.
            save(
                0,
                outputs,
                host_tile=(1, 1, language_model.model.args.hc_count),
            )
        elif layer == 0 and stage == "mlp_injection":
            save(1, outputs)
            save(2, outputs)
        elif layer == 1 and stage == "ple":
            save(3, outputs)
        elif layer in (1, 2, 3) and stage == "mlp_injection":
            save({1: 4, 2: 5, 3: 6}[layer], outputs)
        elif layer == 4 and stage == "attention_hyper_connection":
            save(7, outputs[0])
            cache_records["pre_attention"] = _capture_layer4_linear_cache(
                out_dir,
                mx_arg,
                np,
                cache_entry=cache_arg[4],
                boundary="pre-attention",
                attention_input=outputs[0],
                linear_attention=language_model.model.layers[4].linear_attn,
            )
        elif layer == 4 and stage == "attention":
            save(8, outputs)
            cache_records["post_attention"] = _capture_layer4_linear_cache(
                out_dir,
                mx_arg,
                np,
                cache_entry=cache_arg[4],
                boundary="post-attention",
                attention_input=outputs,
                linear_attention=language_model.model.layers[4].linear_attn,
            )
        elif layer == 4 and stage == "attention_injection":
            save(9, outputs)
        elif layer == 4 and stage == "mlp_hyper_connection":
            mixed = outputs[0]
            save(10, mixed)
            feed_forward = language_model.model.layers[4].mlp
            gates = mx.softmax(invariant._linear(feed_forward.gate, mixed), axis=-1, precise=True)
            indices = mx.argpartition(gates, kth=-feed_forward.top_k, axis=-1)[..., -feed_forward.top_k:]
            weights = mx.take_along_axis(gates, indices, axis=-1)
            weights = weights / weights.sum(axis=-1, keepdims=True)
            mx.eval(indices, weights)
            raw_ids = np.asarray(indices, dtype=np.int32).reshape(-1)
            raw_weights = np.asarray(weights.astype(mx.float32), dtype=np.float32).reshape(-1)
            descending = np.argsort(-raw_weights, kind="stable")
            ordered_ids = raw_ids[descending]
            ordered_weights = raw_weights[descending]
            ids_raw = ordered_ids.astype("<i4", copy=False).tobytes(order="C")
            weights_raw = ordered_weights.astype("<f4", copy=False).tobytes(order="C")
            ids_path = out_dir / "seam-11-router-top10-ids.i32"
            weights_path = out_dir / "seam-11-router-top10-weights.f32"
            atomic_write_bytes(ids_path, ids_raw)
            atomic_write_bytes(weights_path, weights_raw)
            route = {
                "ordinal": 11,
                "layer": 4,
                "stage": "router_top10_ids_and_weights",
                "shape": [10],
                "canonical_order": "descending normalized weight; stable provider-slot tie break",
                "expert_ids": [int(value) for value in ordered_ids],
                "weights": [float(value) for value in ordered_weights],
                "ids_payload": {
                    "path": str(ids_path), "sha256": _sha256(ids_raw), "dtype": "I32_LE",
                    "elements": int(ordered_ids.size), "bytes": len(ids_raw),
                },
                "weights_payload": {
                    "path": str(weights_path), "sha256": _sha256(weights_raw), "dtype": "F32_LE",
                    "elements": int(ordered_weights.size), "bytes": len(weights_raw),
                },
            }
        elif layer == 4 and stage == "mlp_route_and_experts":
            save(12, outputs)
        elif layer == 4 and stage == "mlp_injection":
            save(13, outputs)
            raise _BoundaryComplete

    source._eval_stage = capture_seam
    inputs = mx.array([[token_id]], dtype=mx.int64)
    try:
        source._checkpointed_qwen4_exp_logits(
            language_model,
            expert_store,
            inputs,
            cache,
            mx,
            tree_flatten,
            generation_index=0,
            progress=progress,
            capture_path=manifest_path,
        )
    except _BoundaryComplete:
        pass
    finally:
        source._eval_stage = original_eval_stage

    if [record["ordinal"] for record in seams] != [ordinal for ordinal in range(14) if ordinal != 11]:
        raise ValueError("source boundary capture did not cover every declared continuous seam")
    if [record["ordinal"] for record in layer0_trace] != list(range(len(trace_expected))):
        raise ValueError("source boundary capture did not cover every declared layer-0 trace payload")
    if route is None or len(route["expert_ids"]) != 10:
        raise ValueError("source boundary capture omitted the exact layer-4 top-10 route")
    if set(cache_records) != {"pre_attention", "post_attention"}:
        raise ValueError("source boundary capture omitted a declared layer-4 cache boundary")
    document = {
        **progress,
        "status": STATUS,
        "execution_schedule": source.EXECUTION_SCHEDULE,
        "token_index": 0,
        "token_id": token_id,
        "initial_cache": "empty",
        "layers_executed": [0, 1, 2, 3, 4],
        "ple_layer_ids": [1],
        "layer0_trace_contract_version": 5,
        "layer0_trace": layer0_trace,
        "seams": sorted([*seams, route], key=lambda record: record["ordinal"]),
        "layer4_cache": cache_records,
        "ple_storage_manifest": ple_storage,
        "expert_source_access": expert_store.stats(),
        "finished_unix_ns": time.time_ns(),
        "elapsed_ns": time.time_ns() - started,
        "max_resident_set_size_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "model_loaded": True,
        "full_body_executed": False,
        "lm_head_executed": False,
        "claim_boundary": (
            "External MLX-VLM research payloads only. Hawking must independently verify and package them; "
            "this is not source parity, native correctness, complete inference, TPS, capability, or promotion."
        ),
    }
    atomic_write_json(manifest_path, document)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--token-id", type=int, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--preflight-sha256", required=True)
    args = parser.parse_args(argv)
    if args.token_id < 0:
        raise ValueError("token-id must be non-negative")
    for label, value in (
        ("reference", args.reference_sha256),
        ("authorization", args.authorization_sha256),
        ("preflight", args.preflight_sha256),
    ):
        contract.gate._require_sha256(value, f"{label} sha256")
    result = run(
        args.root,
        args.out_dir,
        token_id=args.token_id,
        reference_sha256=args.reference_sha256,
        authorization_sha256=args.authorization_sha256,
        preflight_sha256=args.preflight_sha256,
    )
    print(json.dumps({"status": result["status"], "elapsed_ns": result["elapsed_ns"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
