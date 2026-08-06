#!/usr/bin/env python3
"""Emit one bounded P0 CPU-Torch Gate-route calibration shard (v2).

This is intentionally separate from the v1 logit-only shard.  It reuses the
immutable P0 Gate-input trace and full-stream admission checks, but retains no
raw activation or source-weight payload.  The four bounded targets are the
source CPU Torch F32 Gate logits, post-``F.softplus(...).sqrt()`` scores,
fixed-``tid2eid`` selected weights, and selected IDs.  It is an unsealed
diagnostic target, never a runtime, TPS, capability, or receipt claim.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import stat
import struct
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


SCHEMA = "hawking.gravity.deepseek_v4.p0_gate_torch_f32_route_calibration_shard.v2"
STATUS = "UNSEALED_QUALIFIED_SOURCE_CPU_TORCH_F32_GATE_ROUTE_TARGET_NON_RECEIPT"
V1_CALIBRATION_PRODUCER = "dsv4f_gate_torch_f32_calibration_shard.py"
SERIALIZED_SHARD_HARD_MAX_BYTES = 32 * 1024
F32 = struct.Struct("<f")
U16 = struct.Struct("<H")


class RouteCalibrationError(ValueError):
    """The bounded source CPU route calibration cannot be safely emitted."""


def _load_v1_validation_authority() -> ModuleType:
    """Load the existing strict trace/artifact/source validator by exact path."""

    script = Path(__file__).resolve()
    candidate = script.parent / V1_CALIBRATION_PRODUCER
    try:
        metadata = os.lstat(candidate)
    except OSError as exc:
        raise RouteCalibrationError(f"cannot stat v1 calibration validation authority: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RouteCalibrationError("v1 calibration validation authority must be a regular non-symlink file")
    spec = importlib.util.spec_from_file_location(
        "_hawking_dsv4f_gate_torch_f32_calibration_v1", candidate
    )
    if spec is None or spec.loader is None:
        raise RouteCalibrationError("cannot load v1 calibration validation authority")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(spec.name, None)
        raise RouteCalibrationError(f"cannot import v1 calibration validation authority: {exc}") from exc
    # Canonical JSON and hash helpers live on the source-probe module returned
    # by `_load_source_probe`; the v1 calibration producer exports only the
    # two bridge functions consumed below.
    for name in ("_load_source_probe", "_source_tensor_public_binding"):
        if not callable(getattr(module, name, None)):
            raise RouteCalibrationError(f"v1 calibration validation authority lacks {name}")
    return module


def _script_binding(probe: ModuleType) -> dict[str, Any]:
    path = probe._absolute_existing_regular(Path(__file__).resolve(), "v2 route calibration producer")
    return {"path": str(path), "sha256": probe._sha256_file(path), "bytes": path.stat().st_size}


def _f32_le(values: Sequence[float], expected_count: int, label: str) -> bytes:
    if len(values) != expected_count:
        raise RouteCalibrationError(f"{label} must contain exactly {expected_count} F32 values")
    encoded = bytearray()
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise RouteCalibrationError(f"{label} F32 value {index} is non-finite")
        encoded.extend(F32.pack(float(value)))
    return bytes(encoded)


def _u16_le(values: Sequence[int], expected_count: int, upper_bound: int, label: str) -> bytes:
    if len(values) != expected_count:
        raise RouteCalibrationError(f"{label} must contain exactly {expected_count} IDs")
    encoded = bytearray()
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < upper_bound:
            raise RouteCalibrationError(f"{label} ID {index} is outside the routed-expert range")
        encoded.extend(U16.pack(value))
    return bytes(encoded)


def _target(
    *,
    name: str,
    dtype: str,
    shape: Sequence[int],
    encoded: bytes,
    encoding: str,
    probe: ModuleType,
) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": dtype,
        "shape": list(shape),
        "element_count": math.prod(shape),
        "byte_order": "little_endian",
        "byte_count": len(encoded),
        "encoding": encoding,
        "sha256": probe._sha256(encoded),
        "data": encoded.hex(),
    }


def _torch_source_route(
    probe: ModuleType,
    gate_bytes: bytes,
    input_bytes: bytes,
    selected_ids: Sequence[int],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Execute the exact source-shaped post-linear Gate route on CPU Torch."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RouteCalibrationError("PyTorch is required but will not be installed by this producer") from exc

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise RouteCalibrationError(f"cannot enforce one PyTorch inter-op CPU thread: {exc}") from exc
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RouteCalibrationError("PyTorch did not accept the required one-thread CPU configuration")

    input_words = torch.frombuffer(bytearray(input_bytes), dtype=torch.uint16).clone()
    gate_words = torch.frombuffer(bytearray(gate_bytes), dtype=torch.uint16).clone()
    if (
        input_words.numel() != probe.HIDDEN_SIZE
        or gate_words.numel() != probe.ROUTED_EXPERTS * probe.HIDDEN_SIZE
    ):
        raise RouteCalibrationError("source Gate storage geometry differs from the bound route contract")
    input_tensor = input_words.view(torch.bfloat16).to(dtype=torch.float32, device="cpu")
    gate_tensor = gate_words.view(torch.bfloat16).reshape(
        probe.ROUTED_EXPERTS, probe.HIDDEN_SIZE
    ).to(dtype=torch.float32, device="cpu")
    ids_tensor = torch.tensor(list(selected_ids), dtype=torch.int64, device="cpu").reshape(
        1, probe.ACTIVATED_EXPERTS
    )
    if (
        input_tensor.device.type != "cpu"
        or gate_tensor.device.type != "cpu"
        or ids_tensor.device.type != "cpu"
        or input_tensor.ndim != 1
        or gate_tensor.ndim != 2
        or ids_tensor.shape != (1, probe.ACTIVATED_EXPERTS)
    ):
        raise RouteCalibrationError("Torch route inputs are not the bound CPU P0 shapes")

    # This mirrors Gate.forward post-linear semantics: a one-row tensor
    # preserves the source gather(dim=1) and normalization dimensions.
    with torch.no_grad():
        logits_matrix = functional.linear(
            input_tensor.unsqueeze(0).float(), gate_tensor.float()
        ).contiguous()
        original_scores_matrix = functional.softplus(logits_matrix).sqrt().contiguous()
        selected_scores = original_scores_matrix.gather(1, ids_tensor)
        selected_sum = selected_scores.sum(dim=-1, keepdim=True, dtype=torch.float32)
        if not bool(torch.isfinite(selected_sum).all()) or bool((selected_sum <= 0).any()):
            raise RouteCalibrationError("source Gate selected-score normalization is invalid")
        selected_weights_matrix = (selected_scores / selected_sum) * float(probe.ROUTE_SCALE)

    tensors = {
        "logits": logits_matrix.reshape(-1),
        "original_scores": original_scores_matrix.reshape(-1),
        "selected_weights": selected_weights_matrix.reshape(-1),
    }
    values: dict[str, list[float]] = {}
    for name, tensor in tensors.items():
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise RouteCalibrationError(f"source Gate {name} is not a CPU F32 tensor")
        result = [float(value) for value in tensor.tolist()]
        if not all(math.isfinite(value) for value in result):
            raise RouteCalibrationError(f"source Gate {name} contains a non-finite value")
        values[name] = result

    return {
        "logits": _f32_le(values["logits"], probe.ROUTED_EXPERTS, "source Gate logits"),
        "original_scores": _f32_le(
            values["original_scores"], probe.ROUTED_EXPERTS, "source Gate original scores"
        ),
        "selected_weights": _f32_le(
            values["selected_weights"], probe.ACTIVATED_EXPERTS, "source Gate selected weights"
        ),
        "selected_ids": _u16_le(
            selected_ids, probe.ACTIVATED_EXPERTS, probe.ROUTED_EXPERTS, "source Gate selected"
        ),
    }, {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "torch_git_version": getattr(torch.version, "git_version", None),
        "execution_device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_cuda_invoked": False,
        "input_dtype": "BF16 decoded to F32",
        "input_shape": [1, probe.HIDDEN_SIZE],
        "weight_dtype": "BF16 decoded to F32",
        "weight_shape": [probe.ROUTED_EXPERTS, probe.HIDDEN_SIZE],
        "logits_dtype": "F32",
        "logits_shape": [1, probe.ROUTED_EXPERTS],
        "original_scores_dtype": "F32",
        "original_scores_shape": [1, probe.ROUTED_EXPERTS],
        "selected_weights_dtype": "F32",
        "selected_weights_shape": [1, probe.ACTIVATED_EXPERTS],
        "selected_ids_dtype": "I64",
        "selected_ids_shape": [1, probe.ACTIVATED_EXPERTS],
        "torch_f_linear_contract": "torch.nn.functional.linear(x.float(), weight.float()) on CPU with bound P0 input reshaped to [1, 4096]",
        "post_linear_route_contract": "original_scores = torch.nn.functional.softplus(logits).sqrt(); indices = tid2eid[input_ids]; weights = original_scores.gather(1, indices); weights /= weights.sum(dim=-1, keepdim=True); weights *= route_scale",
        "route_scale": float(probe.ROUTE_SCALE),
    }


def build_route_calibration_shard(
    trace_path: Path,
    artifact_path: Path,
    *,
    v1: ModuleType | None = None,
    probe: ModuleType | None = None,
    validation_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    v1 = v1 or _load_v1_validation_authority()
    if probe is None:
        probe, validation_authority = v1._load_source_probe()
    if validation_authority is None:
        raise RouteCalibrationError("source-probe validation authority is missing")
    trace_path = probe._absolute_existing_regular(trace_path, "--trace")
    artifact_root = probe._absolute_existing_directory(artifact_path, "--artifact")
    trace_document, trace_raw = probe._read_json(trace_path, "P0 Gate-input trace")
    trace = probe._validate_trace(trace_document, trace_raw)
    manifest, artifact = probe._validate_manifest(artifact_root, trace["artifact"])
    source = probe._validate_model_source(artifact_root, manifest, trace["source"])
    gate_bytes, source_ids, source_tensors = probe._verify_gate_and_route_bindings(
        artifact_root, manifest, trace
    )
    encoded, runtime = _torch_source_route(probe, gate_bytes, trace["payload_bytes"], source_ids)
    targets = {
        "torch_logits_f32_le": _target(
            name="p0_gate_logits_torch_f32_le",
            dtype="F32",
            shape=[probe.ROUTED_EXPERTS],
            encoded=encoded["logits"],
            encoding="lowercase_hex_raw_f32_le",
            probe=probe,
        ),
        "original_scores_f32_le": _target(
            name="p0_gate_original_scores_torch_f32_le",
            dtype="F32",
            shape=[probe.ROUTED_EXPERTS],
            encoded=encoded["original_scores"],
            encoding="lowercase_hex_raw_f32_le",
            probe=probe,
        ),
        "selected_weights_f32_le": _target(
            name="p0_gate_selected_weights_torch_f32_le",
            dtype="F32",
            shape=[probe.ACTIVATED_EXPERTS],
            encoded=encoded["selected_weights"],
            encoding="lowercase_hex_raw_f32_le",
            probe=probe,
        ),
        "selected_expert_ids_u16_le": _target(
            name="p0_gate_selected_expert_ids_tid2eid_u16_le",
            dtype="U16",
            shape=[probe.ACTIVATED_EXPERTS],
            encoded=encoded["selected_ids"],
            encoding="lowercase_hex_raw_u16_le",
            probe=probe,
        ),
    }
    raw_bytes = sum(target["byte_count"] for target in targets.values())
    if raw_bytes != 2 * probe.ROUTED_EXPERTS * F32.size + probe.ACTIVATED_EXPERTS * (F32.size + U16.size):
        raise RouteCalibrationError("route calibration raw-target byte accounting is inconsistent")

    shard: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "unsealed": True,
        "is_receipt": False,
        "receipt_promoted": False,
        "measurement_scope": {
            "source_cpu_torch_gate_route_target": True,
            "direct_upstream_gate_route_path": True,
            "numeric_parity_v2_1": "not_evaluated",
            "runtime": "not_evaluated",
            "gpu": "not_used",
            "hcli": "not_evaluated",
            "token_generation": "not_evaluated",
            "base_true_tps": "not_evaluated",
            "claim_boundary": "One bounded source-CPU Torch Gate-route target for a P0 post-completion diagnostic comparison. It is not a full upstream runtime execution, Numeric Parity V2.1 admission, runtime/GPU/TPS/capability result, receipt, or promotion.",
        },
        "trace_binding": {
            "immutable_existing_trace": True,
            "path": str(trace_path),
            "file_sha256": trace["trace_file_sha256"],
            "schema": probe.TRACE_SCHEMA,
            "status": probe.TRACE_STATUS,
            "layer": trace["trace_binding"]["layer"],
            "token_id": trace["trace_binding"]["token_id"],
            "token_position": trace["trace_binding"]["token_position"],
            "p0_gate_input_sha256": trace["payload_sha256"],
            "raw_gate_input_payload_sha256": trace["payload_sha256"],
            "p0_gate_input_geometry": {
                "dtype": "BF16",
                "shape": [probe.HIDDEN_SIZE],
                "bytes": probe.BF16_ROW_BYTES,
            },
            "recorded_metal_gate_logits_f32_le_sha256": trace["gate_output_sha256"],
            "trace_producer_executable_binding": trace["trace_executable"],
            "raw_p0_input_copied_into_this_shard": False,
        },
        "artifact_binding": artifact,
        "source_binding": source,
        "source_tensor_bindings": v1._source_tensor_public_binding(source_tensors),
        "source_probe_validation_authority": validation_authority,
        "source_cpu_torch_binding": runtime,
        "route_target_contract": {
            "source_model_operator": "Gate.forward",
            "score_function": "sqrtsoftplus",
            "post_linear_operator_order": "F.softplus(scores).sqrt() -> original_scores -> tid2eid[input_ids] -> original_scores.gather(1, indices) -> divide selected sum -> multiply route_scale",
            "selection_method": "source tid2eid[token_id] fixed row",
            "layer": 0,
            "token_id": probe.TOKEN_ID,
            "token_position": 0,
            "route_scale": float(probe.ROUTE_SCALE),
            "top_k": probe.ACTIVATED_EXPERTS,
        },
        "raw_targets": targets,
        "storage_policy": {
            "raw_payload_count": 4,
            "raw_payload_actual_bytes": raw_bytes,
            "raw_payload_hard_max_bytes": raw_bytes,
            "raw_source_weight_payloads": 0,
            "raw_input_payloads": 0,
            "raw_other_activation_payloads": 0,
            "raw_route_payloads": 1,
            "raw_route_weight_payloads": 1,
            "raw_fp64_authority_payloads": 0,
            "serialized_shard_hard_max_bytes": SERIALIZED_SHARD_HARD_MAX_BYTES,
            "only_allowed_raw_payloads": "Torch F32 logits[256], Torch F32 original_scores[256], Torch F32 selected_weights[6], and tid2eid U16 selected_ids[6], all little-endian",
            "retained_values": "four bounded direct source-CPU Gate-route targets plus source/artifact/trace/tensor hashes and runtime bindings; no raw input or weights",
        },
        "producer": _script_binding(probe),
    }
    serialized_without_self_hash = probe._canonical_json(shard)
    if len(serialized_without_self_hash) > SERIALIZED_SHARD_HARD_MAX_BYTES:
        raise RouteCalibrationError("bounded v2 route calibration shard exceeds its 32 KiB serialized hard maximum")
    shard["canonical_sha256"] = probe._sha256(serialized_without_self_hash)
    if len(probe._canonical_json(shard)) > SERIALIZED_SHARD_HARD_MAX_BYTES:
        raise RouteCalibrationError("v2 route calibration shard with canonical hash exceeds its 32 KiB serialized hard maximum")
    return shard


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="absolute immutable P0 Gate-input trace JSON")
    parser.add_argument("--artifact", type=Path, required=True, help="absolute full Gravity artifact directory")
    parser.add_argument("--out", type=Path, required=True, help="absolute new v2 route calibration shard JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    v1: ModuleType | None = None
    probe: ModuleType | None = None
    try:
        v1 = _load_v1_validation_authority()
        probe, validation_authority = v1._load_source_probe()
        out = probe._absolute_new_output(args.out)
        shard = build_route_calibration_shard(
            args.trace, args.artifact, v1=v1, probe=probe, validation_authority=validation_authority
        )
        probe._publish_new_canonical_json(out, shard)
    except RouteCalibrationError as exc:
        print(f"dsv4f_gate_torch_f32_route_calibration_shard: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        if probe is not None and isinstance(exc, probe.ProbeError):
            print(f"dsv4f_gate_torch_f32_route_calibration_shard: {exc}", file=sys.stderr)
            return 2
        raise
    print(
        json.dumps(
            {
                "status": shard["status"],
                "out": str(out),
                "canonical_sha256": shard["canonical_sha256"],
                "torch_logits_sha256": shard["raw_targets"]["torch_logits_f32_le"]["sha256"],
                "original_scores_sha256": shard["raw_targets"]["original_scores_f32_le"]["sha256"],
                "selected_weights_sha256": shard["raw_targets"]["selected_weights_f32_le"]["sha256"],
                "selected_ids_sha256": shard["raw_targets"]["selected_expert_ids_u16_le"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
