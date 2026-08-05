#!/usr/bin/env python3
"""Emit one bounded P0 Torch Gate-logit calibration shard.

This producer deliberately reuses the strict trace/artifact/source validation
from ``dsv4f_gate_torch_flinear_probe.py``.  It references the immutable P0
Gate-input trace but does not copy that input.  Its only retained raw value is
one CPU Torch ``F.linear`` result: 256 little-endian binary32 logits (1024
bytes).  The independent serial binary64 authority is represented only by a
digest and aggregate metrics; its values are never serialized.

This is an unsealed, bounded calibration trace shard, not Numeric Parity
V2.1, a runtime/GPU/TPS result, a receipt, or a promotion.
"""
from __future__ import annotations

import argparse
import array
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


CALIBRATION_SCHEMA = "hawking.gravity.deepseek_v4.p0_gate_torch_f32_calibration_shard.v1"
CALIBRATION_STATUS = "UNSEALED_QUALIFIED_SOURCE_CPU_TORCH_F32_GATE_TARGET_NON_RECEIPT"
SOURCE_PROBE_FILENAME = "dsv4f_gate_torch_flinear_probe.py"
F32 = struct.Struct("<f")
F64 = struct.Struct("<d")
U32 = struct.Struct("<I")
SERIALIZED_SHARD_HARD_MAX_BYTES = 32 * 1024


class CalibrationError(ValueError):
    """The bounded calibration target cannot be safely produced."""


def _load_source_probe() -> tuple[ModuleType, dict[str, Any]]:
    """Load the local source-probe validation authority and bind its bytes."""

    script_path = Path(__file__)
    if not script_path.is_absolute():
        script_path = Path.cwd() / script_path
    candidate = script_path.parent / SOURCE_PROBE_FILENAME
    if not candidate.is_absolute():
        raise CalibrationError("source-probe path is not absolute")
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise CalibrationError(f"cannot stat source-probe validation authority: {exc}") from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise CalibrationError("source-probe validation authority must be a regular non-symlink file")
    module_name = "_hawking_dsv4f_gate_torch_flinear_probe_validation"
    specification = importlib.util.spec_from_file_location(module_name, candidate)
    if specification is None or specification.loader is None:
        raise CalibrationError("cannot load source-probe validation authority")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise CalibrationError(f"cannot import source-probe validation authority: {exc}") from exc

    required = {
        "TRACE_SCHEMA": "hawking.gravity.deepseek_v4.p7_layer0_position0_gate_input_trace.v1",
        "TRACE_STATUS": "UNSEALED_POST_COMPLETION_BOS_FFN_NORM_GATE_INPUT_TRACE_NON_RECEIPT",
        "MANIFEST_SCHEMA": "hawking.gravity.deepseek_v4.full_stream.v1",
        "MANIFEST_STATUS": "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY",
        "GATE_NAME": "layers.0.ffn.gate.weight",
        "TID2EID_NAME": "layers.0.ffn.gate.tid2eid",
        "HIDDEN_SIZE": 4096,
        "ROUTED_EXPERTS": 256,
        "TOKEN_ID": 0,
    }
    for name, expected in required.items():
        if getattr(module, name, None) != expected:
            raise CalibrationError(f"source-probe validation authority has incompatible {name}")
    for name in (
        "ProbeError",
        "_absolute_existing_regular",
        "_absolute_existing_directory",
        "_absolute_new_output",
        "_read_json",
        "_validate_trace",
        "_validate_manifest",
        "_validate_model_source",
        "_verify_gate_and_route_bindings",
        "_decode_bf16_row",
        "_canonical_json",
        "_publish_new_canonical_json",
        "_sha256",
        "_sha256_file",
    ):
        if not callable(getattr(module, name, None)):
            raise CalibrationError(f"source-probe validation authority lacks {name}")
    return module, {
        "path": str(candidate),
        "sha256": module._sha256_file(candidate),
        "bytes": metadata.st_size,
        "validation_reuse": "strict trace/artifact/source/tensor validation helpers imported directly",
    }


def _calibration_script_binding(probe: ModuleType) -> dict[str, Any]:
    path = Path(__file__).resolve()
    checked = probe._absolute_existing_regular(path, "calibration producer")
    return {
        "path": str(checked),
        "sha256": probe._sha256_file(checked),
        "bytes": checked.stat().st_size,
    }


def _f32_values_to_le(values: Sequence[float], expected_count: int) -> bytes:
    if len(values) != expected_count:
        raise CalibrationError(f"Torch Gate output must contain exactly {expected_count} logits")
    encoded = bytearray()
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise CalibrationError(f"Torch Gate output {index} is non-finite")
        encoded.extend(F32.pack(float(value)))
    return bytes(encoded)


def _torch_source_logits(
    probe: ModuleType, gate_bytes: bytes, input_bytes: bytes
) -> tuple[bytes, dict[str, Any]]:
    """Run the exact upstream-shaped, 1-D CPU Torch Gate call."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise CalibrationError(
            "PyTorch is required but not importable; this producer will not install packages"
        ) from exc

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise CalibrationError(f"cannot enforce one PyTorch inter-op CPU thread: {exc}") from exc
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise CalibrationError("PyTorch did not accept the required one-thread CPU configuration")

    input_words = torch.frombuffer(bytearray(input_bytes), dtype=torch.uint16).clone()
    gate_words = torch.frombuffer(bytearray(gate_bytes), dtype=torch.uint16).clone()
    if (
        input_words.numel() != probe.HIDDEN_SIZE
        or gate_words.numel() != probe.ROUTED_EXPERTS * probe.HIDDEN_SIZE
    ):
        raise CalibrationError("Torch BF16 storage geometry differs from the bound source Gate")
    input_tensor = input_words.view(torch.bfloat16).to(dtype=torch.float32, device="cpu")
    gate_tensor = gate_words.view(torch.bfloat16).reshape(
        probe.ROUTED_EXPERTS, probe.HIDDEN_SIZE
    ).to(dtype=torch.float32, device="cpu")
    if input_tensor.device.type != "cpu" or gate_tensor.device.type != "cpu":
        raise CalibrationError("Torch calibration unexpectedly selected a non-CPU device")
    if input_tensor.ndim != 1 or gate_tensor.ndim != 2:
        raise CalibrationError("Torch calibration tensors do not have the bound 1-D/2-D shapes")
    with torch.no_grad():
        logits_tensor = functional.linear(input_tensor.float(), gate_tensor.float()).contiguous()
    if logits_tensor.device.type != "cpu" or logits_tensor.ndim != 1:
        raise CalibrationError("Torch F.linear did not return the expected 1-D CPU output")
    logits = [float(value) for value in logits_tensor.tolist()]
    raw_logits = _f32_values_to_le(logits, probe.ROUTED_EXPERTS)
    if len(raw_logits) != probe.ROUTED_EXPERTS * F32.size:
        raise CalibrationError("Torch F.linear payload length is not exactly 1024 bytes")
    return raw_logits, {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "torch_git_version": getattr(torch.version, "git_version", None),
        "execution_device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_cuda_invoked": False,
        "input_dtype": "BF16 decoded to F32",
        "input_shape": [probe.HIDDEN_SIZE],
        "weight_dtype": "BF16 decoded to F32",
        "weight_shape": [probe.ROUTED_EXPERTS, probe.HIDDEN_SIZE],
        "output_dtype": "F32",
        "output_shape": [probe.ROUTED_EXPERTS],
        "torch_f_linear_contract": "torch.nn.functional.linear(x.float(), weight.float()) on CPU with the bound 1D P0 input",
    }


def _bf16_to_f64(word: int) -> float:
    """BF16 is exactly representable when expanded through binary32 to binary64."""

    return F32.unpack(U32.pack(word << 16))[0]


def _serial_fp64_authority(probe: ModuleType, gate_bytes: bytes, input_bytes: bytes) -> list[float]:
    """Independently recompute a fixed-order, non-FMA binary64 Gate authority."""

    if len(gate_bytes) != probe.ROUTED_EXPERTS * probe.BF16_ROW_BYTES:
        raise CalibrationError("source Gate bytes are not exactly BF16[256,4096]")
    input_f32 = probe._decode_bf16_row(input_bytes, "trace input for serial FP64 authority")
    input_f64 = [float(value) for value in input_f32]
    words = array.array("H")
    words.frombytes(gate_bytes)
    if sys.byteorder != "little":
        words.byteswap()
    if len(words) != probe.ROUTED_EXPERTS * probe.HIDDEN_SIZE:
        raise CalibrationError("source Gate BF16 word count is invalid")

    result: list[float] = []
    for row in range(probe.ROUTED_EXPERTS):
        accumulator = 0.0
        start = row * probe.HIDDEN_SIZE
        for column, activation in enumerate(input_f64):
            weight = _bf16_to_f64(int(words[start + column]))
            if not math.isfinite(weight):
                raise CalibrationError("source Gate contains a non-finite BF16 weight")
            # Separate Python binary64 multiply and add; no FMA or vector reduction.
            accumulator = accumulator + (activation * weight)
        if not math.isfinite(accumulator):
            raise CalibrationError("serial FP64 Gate authority produced a non-finite logit")
        result.append(accumulator)
    return result


def _fp64_values_sha256(probe: ModuleType, values: Sequence[float]) -> str:
    if len(values) != probe.ROUTED_EXPERTS:
        raise CalibrationError("serial FP64 authority length is invalid")
    encoded = bytearray()
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise CalibrationError(f"serial FP64 authority value {index} is non-finite")
        encoded.extend(F64.pack(value))
    return probe._sha256(bytes(encoded))


def _torch_f32_vs_fp64_metrics(torch_f32_le: bytes, fp64_values: Sequence[float]) -> dict[str, Any]:
    if len(torch_f32_le) % F32.size or len(torch_f32_le) // F32.size != len(fp64_values):
        raise CalibrationError("Torch/FP64 metric vector geometry differs")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    squared_error_terms: list[float] = []
    squared_reference_terms: list[float] = []
    for index in range(len(fp64_values)):
        torch_value = F32.unpack_from(torch_f32_le, index * F32.size)[0]
        reference = fp64_values[index]
        if not math.isfinite(torch_value) or not math.isfinite(reference):
            raise CalibrationError("Torch/FP64 metric value is non-finite")
        difference = float(torch_value) - reference
        absolute = abs(difference)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, absolute / max(abs(reference), 1.0e-12))
        squared_error_terms.append(difference * difference)
        squared_reference_terms.append(reference * reference)
    numerator = math.fsum(squared_error_terms)
    denominator = math.fsum(squared_reference_terms)
    relative_l2 = math.sqrt(numerator) / math.sqrt(denominator) if denominator else (
        0.0 if numerator == 0.0 else math.inf
    )
    if not math.isfinite(relative_l2):
        raise CalibrationError("Torch-vs-FP64 relative L2 is not finite")
    return {
        "elements": len(fp64_values),
        "max_abs": maximum_absolute,
        "relative_l2": relative_l2,
        "max_relative_reference_floor_1e-12": maximum_relative,
        "max_meaningful_relative_reference_floor_1e-12": maximum_relative,
    }


def _source_tensor_public_binding(source_tensors: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only source tensor identity, geometry, and chunk hashes—not values/routes."""

    output: dict[str, Any] = {}
    for key in ("gate", "tid2eid"):
        binding = source_tensors.get(key)
        if not isinstance(binding, Mapping):
            raise CalibrationError(f"source validation lacks {key} binding")
        output[key] = {
            "name": binding.get("name"),
            "dtype": binding.get("dtype"),
            "shape": binding.get("shape"),
            "bytes": binding.get("bytes"),
            "logical_tensor_sha256": binding.get("logical_tensor_sha256"),
            "verified_chunk_count": binding.get("verified_chunk_count"),
            "verified_chunk_bytes": binding.get("verified_chunk_bytes"),
            "verified_chunk_sha256": binding.get("verified_chunk_sha256"),
        }
    return output


def build_calibration_shard(
    trace_path: Path,
    artifact_path: Path,
    *,
    probe: ModuleType | None = None,
    validation_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if probe is None:
        probe, loaded_validation_authority = _load_source_probe()
        validation_authority = loaded_validation_authority
    if validation_authority is None:
        raise CalibrationError("source-probe validation binding is missing")
    trace_path = probe._absolute_existing_regular(trace_path, "--trace")
    artifact_root = probe._absolute_existing_directory(artifact_path, "--artifact")
    trace_document, trace_raw = probe._read_json(trace_path, "P0 Gate-input trace")
    trace = probe._validate_trace(trace_document, trace_raw)
    manifest, artifact = probe._validate_manifest(
        artifact_root, trace["artifact"]
    )
    source = probe._validate_model_source(artifact_root, manifest, trace["source"])
    gate_bytes, _source_ids, source_tensors = probe._verify_gate_and_route_bindings(
        artifact_root, manifest, trace
    )

    torch_logits_f32_le, cpu_runtime = _torch_source_logits(
        probe, gate_bytes, trace["payload_bytes"]
    )
    fp64_authority = _serial_fp64_authority(probe, gate_bytes, trace["payload_bytes"])
    torch_payload_sha256 = probe._sha256(torch_logits_f32_le)
    fp64_sha256 = _fp64_values_sha256(probe, fp64_authority)
    fp64_metrics = _torch_f32_vs_fp64_metrics(torch_logits_f32_le, fp64_authority)

    shard: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "status": CALIBRATION_STATUS,
        "unsealed": True,
        "is_receipt": False,
        "receipt_promoted": False,
        "measurement_scope": {
            "source_cpu_torch_gate_target": True,
            "source_cpu_serial_fp64_authority": True,
            "numeric_parity_v2_1": "not_evaluated",
            "runtime": "not_evaluated",
            "gpu": "not_used",
            "hcli": "not_evaluated",
            "token_generation": "not_evaluated",
            "base_true_tps": "not_evaluated",
            "claim_boundary": "One bounded source-CPU Gate calibration target for isolated candidate scoring. It is not a Numeric Parity V2.1 admission, runtime/GPU/TPS/capability result, receipt, or promotion.",
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
        "source_tensor_bindings": _source_tensor_public_binding(source_tensors),
        "source_tensor_binding": _source_tensor_public_binding(source_tensors)["gate"],
        "source_probe_validation_authority": validation_authority,
        "source_cpu_torch_binding": cpu_runtime,
        "raw_f32_le": {
            "name": "p0_gate_logits_torch_f32_le",
            "dtype": "F32",
            "shape": [probe.ROUTED_EXPERTS],
            "element_count": probe.ROUTED_EXPERTS,
            "byte_order": "little_endian",
            "byte_count": probe.ROUTED_EXPERTS * F32.size,
            "encoding": "lowercase_hex_raw_f32_le",
            "sha256": torch_payload_sha256,
            "data": torch_logits_f32_le.hex(),
        },
        "serial_fp64_authority": {
            "definition": "Independent row-major fixed-order binary64 reduction: BF16 input and weights are decoded exactly through binary32; for each row and columns 0..4095 evaluate accumulator = accumulator + (activation * weight) as separate Python binary64 operations. No FMA, vector reduction, Torch reduction, or raw FP64 serialization is used.",
            "dtype": "F64",
            "shape": [probe.ROUTED_EXPERTS],
            "element_count": probe.ROUTED_EXPERTS,
            "byte_order": "little_endian",
            "byte_count_if_materialized": probe.ROUTED_EXPERTS * F64.size,
            "logical_f64_le_sha256": fp64_sha256,
            "raw_values_retained": False,
            "torch_f32_vs_fp64_metrics": fp64_metrics,
        },
        "storage_policy": {
            "raw_payload_count": 1,
            "raw_payload_actual_bytes": probe.ROUTED_EXPERTS * F32.size,
            "raw_payload_hard_max_bytes": probe.ROUTED_EXPERTS * F32.size,
            "raw_source_weight_payloads": 0,
            "raw_input_payloads": 0,
            "raw_route_payloads": 0,
            "raw_route_weight_payloads": 0,
            "raw_fp64_authority_payloads": 0,
            "serialized_shard_hard_max_bytes": SERIALIZED_SHARD_HARD_MAX_BYTES,
            "only_allowed_raw_payload": "p0_gate_logits_torch_f32_le F32[256] little-endian, 1024 bytes",
            "retained_values": "one bounded Torch F32 logit target plus provenance hashes, tensor identities, runtime bindings, serial-FP64 digest, and aggregate metrics",
        },
        "producer": _calibration_script_binding(probe),
    }
    serialized_without_self_hash = probe._canonical_json(shard)
    if len(serialized_without_self_hash) > SERIALIZED_SHARD_HARD_MAX_BYTES:
        raise CalibrationError(
            "bounded calibration shard exceeds its 32 KiB serialized hard maximum"
        )
    shard["canonical_sha256"] = probe._sha256(serialized_without_self_hash)
    if len(probe._canonical_json(shard)) > SERIALIZED_SHARD_HARD_MAX_BYTES:
        raise CalibrationError(
            "bounded calibration shard with canonical hash exceeds its 32 KiB serialized hard maximum"
        )
    return shard


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="absolute immutable P0 Gate-input trace JSON")
    parser.add_argument("--artifact", type=Path, required=True, help="absolute full Gravity artifact directory")
    parser.add_argument("--out", type=Path, required=True, help="absolute new calibration shard JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    probe: ModuleType | None = None
    try:
        # Validate this before expensive source reconstruction or Torch import.
        probe, _validation_authority = _load_source_probe()
        out = probe._absolute_new_output(args.out)
        shard = build_calibration_shard(
            args.trace,
            args.artifact,
            probe=probe,
            validation_authority=_validation_authority,
        )
        probe._publish_new_canonical_json(out, shard)
    except CalibrationError as exc:
        print(f"dsv4f_gate_torch_f32_calibration_shard: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        if probe is not None and isinstance(exc, probe.ProbeError):
            print(f"dsv4f_gate_torch_f32_calibration_shard: {exc}", file=sys.stderr)
            return 2
        raise
    print(
        json.dumps(
            {
                "status": shard["status"],
                "out": str(out),
                "canonical_sha256": shard["canonical_sha256"],
                "torch_f32_logits_sha256": shard["raw_f32_le"]["sha256"],
                "serial_fp64_authority_sha256": shard["serial_fp64_authority"]["logical_f64_le_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
