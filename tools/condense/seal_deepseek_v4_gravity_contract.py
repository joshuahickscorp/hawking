#!/usr/bin/env python3.12
"""Seal a source-bound, non-executable contract for the streamed DSV4F body.

This deliberately does *not* alter the immutable full-stream manifest.  That
manifest is content-addressed and its current status correctly says that no
43-layer runtime is registered.  The output is a sidecar contract bound to
that manifest, intended to keep the exact source representation, execution
grammar, state requirements, and protected transplant surfaces available while
the native runtime is being built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from lab.receipts import seal, verify  # noqa: E402


FULL_SCHEMA = "hawking.gravity.deepseek_v4.full_stream.v1"
FULL_STATUS = "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY"
BASELINE_FILES = (
    "DSV4F_CHILD_BASELINE.json",
    "DSV4F_RUNTIME_PROFILE.json",
    "DSV4F_ROUTE_PROFILE.json",
    "DSV4F_LATENT_BRIDGE_CONTRACT.json",
    "DSV4F_TRANSPLANT_POINTS.json",
    "DSV4F_100TPS_SCOREBOARD.json",
    "DSV4F_KERNEL_REGISTRY.json",
    "DSV4F_ROOFLINE.json",
)


class ContractError(ValueError):
    """A source or frozen-evidence invariant did not hold."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"{label} must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain a JSON object")
    return value


def product(shape: object) -> int:
    if not isinstance(shape, list) or not shape:
        raise ContractError(f"invalid tensor shape: {shape!r}")
    answer = 1
    for value in shape:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContractError(f"invalid tensor dimension: {value!r}")
        answer *= value
    return answer


def layer_name(name: str) -> int | None:
    if not name.startswith("layers."):
        return None
    part = name.split(".", 2)[1]
    try:
        return int(part)
    except ValueError:
        return None


def logical_matrix_elements(name: str, tensor: Mapping[str, Any]) -> int:
    elements = product(tensor["shape"])
    return elements * 2 if tensor.get("dtype") == "I8" and name.endswith(".weight") else elements


def tensor_bucket(name: str) -> str:
    if name.startswith("embed."):
        return "embedding"
    if name.startswith("head.") or name.startswith("norm."):
        return "lm_head_and_final_norm"
    if name.startswith("mtp."):
        return "mtp_draft_only_excluded_from_base_decode"
    if ".ffn.experts." in name:
        return "routed_experts"
    if ".ffn.shared_experts." in name:
        return "shared_expert"
    if ".ffn.gate." in name:
        return "router"
    if ".hc_" in name:
        return "mhc"
    if ".attn.indexer." in name:
        return "compressed_attention_indexer"
    if ".attn.compressor." in name:
        return "compressed_attention_compressor"
    if ".attn." in name:
        return "attention"
    return "other_source_parameter"


def pair_statistics(tensors: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    bytes_by_kind: Counter[str] = Counter()
    logical_by_kind: Counter[str] = Counter()
    scale_tensor_names: set[str] = set()
    for name, tensor in tensors.items():
        if not name.endswith(".weight"):
            continue
        scale_name = f"{name[:-7]}.scale"
        scale = tensors.get(scale_name)
        if not isinstance(scale, Mapping):
            continue
        dtype = tensor.get("dtype")
        if (dtype, scale.get("dtype")) == ("F8_E4M3", "F8_E8M0"):
            kind = "native_fp8_e4m3fn_plus_e8m0"
        elif (dtype, scale.get("dtype")) == ("I8", "F8_E8M0"):
            kind = "native_fp4_e2m1fn_x2_plus_e8m0"
        else:
            continue
        weight_bytes = tensor.get("bytes")
        scale_bytes = scale.get("bytes")
        if not isinstance(weight_bytes, int) or not isinstance(scale_bytes, int):
            raise ContractError(f"missing bytes for native pair {name}")
        counts[kind] += 1
        bytes_by_kind[kind] += weight_bytes + scale_bytes
        logical_by_kind[kind] += logical_matrix_elements(name, tensor)
        scale_tensor_names.add(scale_name)

    output: dict[str, Any] = {}
    for kind in sorted(counts):
        logical = logical_by_kind[kind]
        output[kind] = {
            "native_pair_count": counts[kind],
            "physical_weight_plus_scale_bytes": bytes_by_kind[kind],
            "logical_weight_elements_excluding_scale_elements": logical,
            "bits_per_logical_weight_including_scale_overhead": bytes_by_kind[kind] * 8 / logical,
        }
    return {"by_representation": output, "paired_scale_tensor_names": sorted(scale_tensor_names)}


def matrix_fma_flops(name: str, tensor: Mapping[str, Any]) -> int:
    shape = tensor.get("shape")
    if not name.endswith(".weight") or not isinstance(shape, list) or len(shape) != 2:
        return 0
    return 2 * logical_matrix_elements(name, tensor)


def decode_matrix_work(tensors: Mapping[str, Mapping[str, Any]], layer_count: int) -> dict[str, Any]:
    """Exact selected-matrix FMA count for one source base decode path.

    It counts source linear matrices (not softmax/norm/nonlinear/integer work).
    Each routed expert has the same geometry, so expert zero is a valid shape
    representative while top-6 count is taken from the pinned config.
    """

    per_layer: dict[str, Any] = {}
    total = 0
    for layer in range(layer_count):
        prefix = f"layers.{layer}."
        dense = 0
        selected_routed = 0
        shared = 0
        for name, tensor in tensors.items():
            if not name.startswith(prefix):
                continue
            flops = matrix_fma_flops(name, tensor)
            if not flops:
                continue
            if ".ffn.experts." in name:
                if ".experts.0." in name:
                    selected_routed += flops
            elif ".ffn.shared_experts." in name:
                shared += flops
            else:
                dense += flops
        routed_top6 = selected_routed * 6
        per_layer[str(layer)] = {
            "non_routed_linear_fma_flops": dense,
            "one_routed_expert_linear_fma_flops": selected_routed,
            "top6_routed_expert_linear_fma_flops": routed_top6,
            "shared_expert_linear_fma_flops": shared,
            "selected_linear_fma_flops": dense + routed_top6 + shared,
        }
        total += dense + routed_top6 + shared

    head = matrix_fma_flops("head.weight", tensors["head.weight"])
    return {
        "per_layer": per_layer,
        "body_selected_linear_fma_flops": total,
        "lm_head_linear_fma_flops": head,
        "total_selected_linear_fma_flops_excluding_mhc_and_attention": total + head,
        "convention": "one multiply-plus-add is two floating-point operations; this is a source-matrix algebraic count, not a GPU hardware counter",
    }


def attention_work(config: Mapping[str, Any], context: int) -> dict[str, Any]:
    ratios = config["compress_ratios"]
    heads = config["n_heads"]
    head_dim = config["head_dim"]
    window = config["window_size"]
    index_topk = config["index_topk"]
    total = 0
    per_layer: dict[str, Any] = {}
    for layer, ratio in enumerate(ratios):
        if ratio == 0:
            compressed = 0
            mode = "sliding_window_only"
        elif ratio == 4:
            compressed = min(index_topk, context // ratio)
            mode = "indexer_topk"
        else:
            compressed = context // ratio
            mode = "compressed_all_positions"
        slots = window + compressed
        # QK dot plus weighted-V accumulation: 4 FLOPs per head-dim per slot.
        flops = 4 * heads * head_dim * slots
        per_layer[str(layer)] = {
            "compress_ratio": ratio,
            "selection_mode": mode,
            "decode_candidate_slots": slots,
            "algebraic_qk_plus_weighted_value_fma_flops": flops,
        }
        total += flops
    return {
        "context_tokens": context,
        "per_layer": per_layer,
        "total_sparse_attention_fma_flops": total,
        "excludes": ["softmax", "indexer scoring", "rotary", "QAT", "cache traffic", "kernel overhead"],
    }


def source_hashes(artifact: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    assets = manifest["source"]["metadata_assets"]
    output: dict[str, Any] = {}
    for key in ("config.json", "inference/config.json", "inference/model.py", "inference/kernel.py"):
        declared = assets.get(key)
        if not isinstance(declared, Mapping):
            raise ContractError(f"missing declared source asset {key}")
        path = artifact / "metadata" / key
        observed = sha256_file(path)
        if observed != declared.get("sha256"):
            raise ContractError(f"source asset hash mismatch for {key}")
        output[key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}
    return output


def baseline_bindings(baseline_dir: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for filename in BASELINE_FILES:
        path = baseline_dir / filename
        payload = read_json(path, filename)
        verify(payload, label=str(path))
        output[filename] = {
            "path": str(path),
            "schema": payload.get("schema"),
            "status": payload.get("status"),
            "seal_sha256": payload.get("seal_sha256"),
            "file_sha256": sha256_file(path),
        }
    return output


def tokenizer_template_binding(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = read_json(path, "tokenizer/template admission receipt")
    verify(payload, label=str(path))
    if payload.get("schema") != "hawking.gravity.deepseek_v4.tokenizer_template_admission.v1":
        raise ContractError("tokenizer/template receipt has an unexpected schema")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ContractError("tokenizer/template receipt lacks source binding")
    if source.get("repository") != manifest["source"]["repository"] or source.get("revision") != manifest["source"]["revision"]:
        raise ContractError("tokenizer/template receipt source identity differs from the full stream")
    return {
        "path": str(path),
        "schema": payload["schema"],
        "status": payload.get("status"),
        "seal_sha256": payload.get("seal_sha256"),
        "file_sha256": sha256_file(path),
        "chat_template_policy": "No template is admitted unless this receipt source-binds it; fallback role markers are not equivalent to a source template.",
    }


def build_contract(artifact: Path, baseline_dir: Path, tokenizer_receipt: Path) -> dict[str, Any]:
    manifest_path = artifact / "manifest.json"
    manifest = read_json(manifest_path, "full stream manifest")
    verify(manifest, label=str(manifest_path))
    if manifest.get("schema") != FULL_SCHEMA or manifest.get("status") != FULL_STATUS:
        raise ContractError("full stream schema/status is not the sealed runtime-pending contract")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, Mapping) or len(tensors) != 69_187:
        raise ContractError("full stream tensor mapping is incomplete")
    if not all(isinstance(name, str) and isinstance(tensor, Mapping) for name, tensor in tensors.items()):
        raise ContractError("full stream tensor mapping has an invalid member")

    config_path = artifact / "metadata" / "config.json"
    config = read_json(config_path, "source config")
    inferred = read_json(artifact / "metadata" / "inference" / "config.json", "inference config")
    layer_count = config.get("num_hidden_layers")
    if layer_count != 43 or inferred.get("n_layers") != layer_count:
        raise ContractError("source layer count does not bind the expected 43-layer body")
    if config.get("num_experts_per_tok") != 6 or config.get("n_routed_experts") != 256:
        raise ContractError("source MoE route geometry does not bind the expected top-6 / 256 experts")

    dtype_counts: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    bucket_bytes: Counter[str] = Counter()
    logical_parameters_excluding_paired_scales = 0
    native_pairs = pair_statistics(tensors)
    paired_scales = set(native_pairs.pop("paired_scale_tensor_names"))
    for name, tensor in tensors.items():
        dtype = tensor.get("dtype")
        size = tensor.get("bytes")
        if not isinstance(dtype, str) or not isinstance(size, int):
            raise ContractError(f"invalid tensor representation: {name}")
        dtype_counts[dtype] += 1
        dtype_bytes[dtype] += size
        bucket_bytes[tensor_bucket(name)] += size
        if name not in paired_scales:
            logical_parameters_excluding_paired_scales += logical_matrix_elements(name, tensor)

    total_tensor_bytes = manifest["artifact"]["total_tensor_bytes"]
    if sum(dtype_bytes.values()) != total_tensor_bytes:
        raise ContractError("dtype byte accounting does not reconcile with the manifest")
    matrices = decode_matrix_work(tensors, layer_count)
    attention_contexts = {str(ctx): attention_work(inferred, ctx) for ctx in (2048, 8192, 32768)}
    for value in attention_contexts.values():
        value["total_selected_linear_plus_attention_fma_flops"] = (
            matrices["total_selected_linear_fma_flops_excluding_mhc_and_attention"]
            + value["total_sparse_attention_fma_flops"]
        )

    baseline = baseline_bindings(baseline_dir)
    tokenizer = tokenizer_template_binding(tokenizer_receipt, manifest)
    static = read_json(
        Path(baseline["DSV4F_CHILD_BASELINE.json"]["path"]).parent.parent
        / "static-expert-residency-receipt-v2.json",
        "static residency receipt",
    )
    verify(static, label="static residency receipt")
    active = static["static_active_byte_summary"]

    return seal(
        {
            "schema": "hawking.gravity.deepseek_v4.runtime_contract_sidecar.v1",
            "status": "SEALED_SOURCE_BOUND_RUNTIME_CONTRACT_NOT_EXECUTION_RECEIPT",
            "artifact_binding": {
                "path": str(artifact),
                "manifest_file_sha256": sha256_file(manifest_path),
                "manifest_seal_sha256": manifest["seal_sha256"],
                "schema": manifest["schema"],
                "status": manifest["status"],
                "source_parent_persisted": manifest["source"]["source_parent_persisted"],
            },
            "source": {
                "repository": manifest["source"]["repository"],
                "revision": manifest["source"]["revision"],
                "verified_metadata_assets": source_hashes(artifact, manifest),
                "tokenizer_template_admission": tokenizer,
            },
            "representation": {
                "tensor_count": len(tensors),
                "tensor_bytes": total_tensor_bytes,
                "by_dtype": {
                    key: {"tensor_count": dtype_counts[key], "bytes": dtype_bytes[key]}
                    for key in sorted(dtype_counts)
                },
                "bytes_by_operator_family": dict(sorted(bucket_bytes.items())),
                "native_quantized_pairs": native_pairs["by_representation"],
                "all_tensor_storage_bits_per_logical_parameter_excluding_paired_scale_elements": (
                    total_tensor_bytes * 8 / logical_parameters_excluding_paired_scales
                ),
                "definition": "The all-tensor aggregate includes model tensors and quantization-scale storage in the numerator; the denominator expands packed FP4 weights and excludes paired scale elements. It is a storage representation statistic, not an accuracy or runtime claim.",
            },
            "source_kernel_grammar": {
                "base_decode": [
                    "exact_tokenizer_and_declared_template_contract",
                    "BF16_embedding_lookup_then_hc_mult_4_expand",
                    "for_each_of_43_base_layers: mhc_attn_pre -> attn_norm -> q_and_kv_branches -> compressed_or_window_sparse_attention -> output_projection -> mhc_attn_post",
                    "for_each_of_43_base_layers: mhc_ffn_pre -> ffn_norm -> gate -> top6_routed_fp4_experts_plus_shared_fp8_expert -> route_weighted_combine -> mhc_ffn_post",
                    "mhc_head -> final_rms_norm -> BF16_lm_head -> top_k_or_sampling -> minimal_readback -> HCLI_stream",
                ],
                "conditional_attention": {
                    "ratio_0": "128-token sliding-window sparse attention",
                    "ratio_4": "window plus learned indexer top-k compressed positions",
                    "ratio_128": "window plus compressed positions",
                    "source_compress_ratios": inferred["compress_ratios"],
                },
                "quantized_linear_contract": "source model.py linear(): BF16 activation -> dynamic E4M3FN plus UE8M0 act_quant -> native FP8 or packed FP4 weight/scale GEMM",
                "base_gate_excludes_mtp": True,
                "source_hash_bound": True,
            },
            "static_operation_contract": {
                "selected_linear_matrix_work": matrices,
                "decode_contexts": attention_contexts,
                "mhc_and_non_linear_work": "source-declared but not added to the FMA totals; it requires source-faithful GPU implementation and measured counters",
                "integer_bit_work": "QAT format encode/decode, packed FP4 nibble decode, top-k/indexing, and sampling are required source operations; no device hardware counters have been collected for a 43-layer token",
            },
            "state_and_residency_contract": {
                "model_weight_resident_bytes": "NOT_MEASURED_NO_REGISTERED_43_LAYER_RUNTIME",
                "physical_active_bytes_per_token": active["physical_active_bytes_per_token"],
                "static_selected_body_weight_logical_bytes_per_decode_token": active[
                    "body_selected_weight_logical_bytes_per_decode_token"
                ],
                "static_selected_body_weight_interpretation": active[
                    "body_selected_weight_logical_bytes_interpretation"
                ],
                "dense_lm_head_logical_source_bytes_without_residency": active[
                    "dense_lm_head_logical_source_bytes_per_decode_token_without_residency"
                ],
                "kv_state_format": "source Attention stores BF16 KV cache; non-rope dimensions are source-QAT simulated and compressed paths add compressor/indexer state. Exact native runtime allocation/traffic remains unmeasured.",
                "expert_cache_policy": "No native runtime cache policy is promoted by this sidecar. A subsequent storage probe or registered runtime must establish actual hits, cold latency, prefetch accuracy, and physical bytes.",
            },
            "dependency_and_synchronization_contract": {
                "source_dependency_depth": "43 sequential base blocks, each with two mHC residual phases; q and KV branches may be scheduled concurrently only after source-parity validation",
                "command_buffers_per_token": "NOT_MEASURED_NO_NATIVE_43_LAYER_RUNTIME",
                "cpu_visible_waits_per_token": "NOT_MEASURED_NO_NATIVE_43_LAYER_RUNTIME",
                "synchronization_cost": "NOT_MEASURED_NO_NATIVE_43_LAYER_RUNTIME",
                "no_empty_commit_promotion": True,
            },
            "protected_organs_and_future_bridges": {
                "protected": [
                    "exact tokenizer/template alignment",
                    "mHC state and Sinkhorn parameters",
                    "attention compressor/indexer/KV state",
                    "router logits/top6 route IDs/weights/margins",
                    "routed and shared expert execution boundaries",
                    "final hidden state and lm_head logits",
                ],
                "bridge_locations": {
                    "early": 0,
                    "middle": 21,
                    "late": 42,
                    "adapter_policy": "future adapters are reversible sidecars; direct donor-weight transplantation is not assumed",
                },
                "baseline_bridge_contract": baseline["DSV4F_LATENT_BRIDGE_CONTRACT.json"],
                "baseline_transplant_points": baseline["DSV4F_TRANSPLANT_POINTS.json"],
            },
            "capability_and_runtime_receipts": {
                "full_stream_reverify": baseline["DSV4F_CHILD_BASELINE.json"]["status"],
                "frozen_baseline": baseline["DSV4F_CHILD_BASELINE.json"],
                "kernel_registry": baseline["DSV4F_KERNEL_REGISTRY.json"],
                "runtime_profile": baseline["DSV4F_RUNTIME_PROFILE.json"],
                "tps_scoreboard": baseline["DSV4F_100TPS_SCOREBOARD.json"],
            },
            "claim_boundary": {
                "full_43_layer_runtime": False,
                "source_forward_parity": False,
                "numeric_parity_v2_1": False,
                "base_true_tps": False,
                "hcli_full_child_endpoint": False,
                "this_is_a_static_source_bound_contract": True,
            },
            "promotion_requirements": [
                "registered 43-layer engine consuming the admitted stream without parent safetensors",
                "source/CPU and Metal Numeric Parity V2.1 through complete token and continuation",
                "real GPU per-stage profiler with no unexplained other bucket above 2 percent",
                "actual expert-cache/residency counters and command topology",
                "eligible 8K BASE_TRUE_TPS trial protocol",
            ],
        }
    )


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(value) + b"\n"
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-template-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact_dir.resolve(strict=True)
    baseline_dir = args.baseline_dir.resolve(strict=True)
    tokenizer_receipt = args.tokenizer_template_receipt.resolve(strict=True)
    output = build_contract(artifact, baseline_dir, tokenizer_receipt)
    verify(output, label="generated DSV4F runtime contract")
    atomic_write(args.out.resolve(), output)
    print(json.dumps({"out": str(args.out.resolve()), "seal_sha256": output["seal_sha256"], "status": output["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
