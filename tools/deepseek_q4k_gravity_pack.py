#!/usr/bin/env python3
"""Pack a DeepSeek-V2-Lite GGUF into the canonical Gravity MLA/MoE ABI.

The source K/Q quant bytes are copied verbatim.  Rank-3 expert tensors are
split into one canonical rank-2 tensor per expert so the CPU Gravity contract
can route without understanding GGUF's expert-plane spelling.  This is a
container/adapter gate, not a compression claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from gguf import GGUFReader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.condense import artifact_client


SCHEMA = "hawking.tg.deepseek_source_quant_gravity_pack.v1"
Q4_K, Q5_0, Q6_K, Q8_0 = 12, 6, 14, 8
F32 = 0


def scalar(reader: GGUFReader, name: str) -> int | float:
    field = reader.fields[name]
    raw = field.parts[-1]
    if hasattr(raw, "item"):
        return raw.item()
    return raw


def string_field(reader: GGUFReader, name: str) -> str:
    field = reader.fields[name]
    raw = field.parts[-1]
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, list):
        return bytes(int(v) for v in raw).decode("utf-8")
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw).decode("utf-8")
    raise ValueError(f"{name}: expected GGUF string payload, got {type(raw).__name__}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_codec(tensor_type: int) -> tuple[str, int, int]:
    table = {
        Q4_K: ("ggml.q4_k", 256, 144),
        Q5_0: ("ggml.q5_0", 32, 22),
        Q6_K: ("ggml.q6_k", 256, 210),
        Q8_0: ("ggml.q8_0", 32, 34),
        F32: ("native.f32", 1, 4),
    }
    try:
        return table[int(tensor_type)]
    except KeyError as exc:
        raise ValueError(f"unsupported DeepSeek source GGML dtype {tensor_type}") from exc


def runtime_shape(shape: list[int]) -> list[int]:
    if len(shape) == 1:
        return shape
    if len(shape) == 2:
        return [shape[1], shape[0]]
    raise ValueError(f"expected rank-1/2 shape after expert split, got {shape}")


def arch(reader: GGUFReader) -> dict:
    prefix = string_field(reader, "general.architecture")
    if prefix != "deepseek2":
        raise ValueError(f"expected deepseek2 GGUF, got {prefix!r}")
    hidden = int(scalar(reader, "deepseek2.embedding_length"))
    key_length = int(scalar(reader, "deepseek2.attention.key_length"))
    rope_dim = int(scalar(reader, "deepseek2.rope.dimension_count"))
    return {
        "model_type": "deepseek2",
        "num_hidden_layers": int(scalar(reader, "deepseek2.block_count")),
        "hidden_size": hidden,
        "num_attention_heads": int(scalar(reader, "deepseek2.attention.head_count")),
        "num_key_value_heads": int(scalar(reader, "deepseek2.attention.head_count_kv")),
        "q_lora_rank": None,
        "kv_lora_rank": int(scalar(reader, "deepseek2.attention.kv_lora_rank")),
        "qk_nope_head_dim": key_length - rope_dim,
        "qk_rope_head_dim": rope_dim,
        "v_head_dim": int(scalar(reader, "deepseek2.attention.value_length")),
        "intermediate_size": int(scalar(reader, "deepseek2.feed_forward_length")),
        "moe_intermediate_size": int(scalar(reader, "deepseek2.expert_feed_forward_length")),
        "n_routed_experts": int(scalar(reader, "deepseek2.expert_count")),
        "n_shared_experts": int(scalar(reader, "deepseek2.expert_shared_count")),
        "num_experts_per_tok": int(scalar(reader, "deepseek2.expert_used_count")),
        "first_k_dense_replace": int(scalar(reader, "deepseek2.leading_dense_block_count")),
        # DeepSeek-V2-Lite-Chat's published config is the non-grouped V2
        # router.  These are explicit header fields, not runtime guesses.
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "greedy",
        "scoring_func": "softmax",
        "norm_topk_prob": False,
        "routed_scaling_factor": float(scalar(reader, "deepseek2.expert_weights_scale")),
        "vocab_size": int(scalar(reader, "deepseek2.vocab_size")),
        "rope_theta": float(scalar(reader, "deepseek2.rope.freq_base")),
        "rms_norm_eps": float(scalar(reader, "deepseek2.attention.layer_norm_rms_epsilon")),
        "max_position_embeddings": int(scalar(reader, "deepseek2.context_length")),
        "rope_scaling": {
            "type": "yarn",
            "factor": float(scalar(reader, "deepseek2.rope.scaling.factor")),
            "original_max_position_embeddings": int(
                scalar(reader, "deepseek2.rope.scaling.original_context_length")
            ),
            "beta_fast": 32,
            "beta_slow": 1,
            "yarn_log_multiplier": float(
                scalar(reader, "deepseek2.rope.scaling.yarn_log_multiplier")
            ),
        },
    }


def canonical_name(source_name: str) -> str | None:
    if source_name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if source_name == "output.weight":
        return "lm_head.weight"
    if source_name == "output_norm.weight":
        return "model.norm.weight"
    if not source_name.startswith("blk."):
        return None
    layer, suffix = source_name.split(".", 2)[1:]
    direct = {
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attn_kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
        "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
        "attn_kv_b.weight": "self_attn.kv_b_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
        "ffn_gate_inp.weight": "mlp.gate.weight",
        "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj.weight",
        "ffn_up_shexp.weight": "mlp.shared_experts.up_proj.weight",
        "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.weight",
    }
    target = direct.get(suffix)
    if target:
        return f"model.layers.{layer}.{target}"
    return None


def expert_name(source_name: str, expert: int) -> str | None:
    if not source_name.startswith("blk."):
        return None
    layer, suffix = source_name.split(".", 2)[1:]
    table = {
        "ffn_gate_exps.weight": "gate_proj.weight",
        "ffn_up_exps.weight": "up_proj.weight",
        "ffn_down_exps.weight": "down_proj.weight",
    }
    target = table.get(suffix)
    return (
        f"model.layers.{layer}.mlp.experts.{expert}.{target}"
        if target is not None
        else None
    )


def descriptor_and_blob(tensor, *, expert: int | None = None) -> tuple[dict, bytes]:
    source_shape = [int(v) for v in tensor.shape]
    codec, block_elems, block_bytes = source_codec(int(tensor.tensor_type))
    if expert is None:
        if len(source_shape) > 2:
            raise ValueError(f"{tensor.name}: rank-{len(source_shape)} tensor needs expert split")
        shape = runtime_shape(source_shape)
        blob = tensor.data.tobytes(order="C")
    else:
        if len(source_shape) != 3 or expert >= source_shape[2]:
            raise ValueError(f"{tensor.name}: invalid expert plane {expert}")
        shape = runtime_shape(source_shape[:2])
        blob = tensor.data[expert].tobytes(order="C")
    elements = 1
    for dim in shape:
        elements *= dim
    expected = (elements // block_elems) * block_bytes
    if len(blob) != expected:
        raise ValueError(
            f"{tensor.name}: raw bytes {len(blob)} != {expected} for {shape} {codec}"
        )
    return (
        {
            "name": "",
            "shape": shape,
            "codec": codec,
            "elements": elements,
            "bpw": len(blob) * 8 / max(1, elements),
            "terminal_state": "SOURCE_QUANT_BYTES_COPIED",
            "source_tensor": tensor.name,
            **({"source_expert": expert} if expert is not None else {}),
        },
        blob,
    )


def build(source: Path, output: Path | None, *, write: bool) -> dict:
    reader = GGUFReader(str(source), mode="r")
    source_hash = sha256(source)
    architecture = arch(reader)
    payloads: list[tuple[dict, bytes]] = []
    skipped: list[str] = []
    mapped = 0
    complete_bytes = compressed_bytes = 0
    complete_elements = compressed_elements = 0
    for tensor in reader.tensors:
        target = canonical_name(tensor.name)
        if target is not None:
            descriptor, blob = descriptor_and_blob(tensor)
            descriptor["name"] = target
            payloads.append((descriptor, blob))
            mapped += 1
            complete_bytes += len(blob)
            complete_elements += descriptor["elements"]
            if descriptor["codec"].startswith("ggml."):
                compressed_bytes += len(blob)
                compressed_elements += descriptor["elements"]
            continue
        if tensor.name.endswith("_exps.weight"):
            source_shape = [int(v) for v in tensor.shape]
            if len(source_shape) != 3:
                raise ValueError(f"{tensor.name}: expert tensor is not rank-3")
            for expert in range(source_shape[2]):
                target = expert_name(tensor.name, expert)
                if target is None:
                    skipped.append(tensor.name)
                    break
                descriptor, blob = descriptor_and_blob(tensor, expert=expert)
                descriptor["name"] = target
                payloads.append((descriptor, blob))
                mapped += 1
                complete_bytes += len(blob)
                complete_elements += descriptor["elements"]
                if descriptor["codec"].startswith("ggml."):
                    compressed_bytes += len(blob)
                    compressed_elements += descriptor["elements"]
            continue
        skipped.append(tensor.name)

    if not payloads:
        raise ValueError("source had no executable DeepSeek tensors")
    names = [descriptor["name"] for descriptor, _ in payloads]
    if len(names) != len(set(names)):
        raise ValueError("canonical DeepSeek tensor map contains duplicates")
    compression = {
        "codec": "ggml-source-q4_k-q5_0-q6_k-q8_0",
        "source_quantization_preserved": True,
        "packed_bpw": compressed_bytes * 8 / max(1, compressed_elements),
        "complete_bpw": complete_bytes * 8 / max(1, complete_elements),
        "lossy_fit": False,
    }
    plan = {
        "schema": SCHEMA,
        "status": "SOURCE_COVERAGE_READY_NOT_WRITTEN" if not write else "WRITTEN",
        "source": str(source.resolve()),
        "source_sha256": source_hash,
        "architecture": architecture,
        "mapped_tensors": mapped,
        "skipped_source_tensors": skipped,
        "payload_bytes": complete_bytes,
        "compressed_bytes": compressed_bytes,
        "complete_bpw": compression["complete_bpw"],
        "packed_bpw": compression["packed_bpw"],
        "lossy_fit": False,
        "runtime": "GravityDeepSeek CPU MLA/MoE raw GGML quant ABI",
    }
    if not write:
        return plan
    if output is None:
        raise ValueError("--output is required with --write")
    output.parent.mkdir(parents=True, exist_ok=True)
    telemetry: dict = {}
    artifact_client.write_shard(
        output,
        payloads,
        model={"family": "deepseek2", "source_gguf_sha256": source_hash},
        architecture=architecture,
        tokenizer={"kind": "embedded-gguf", "source_sha256": source_hash},
        compression=compression,
        telemetry=telemetry,
    )
    verification = artifact_client.verify(output)
    if not verification.get("ok"):
        raise RuntimeError(f"written artifact failed integrity: {verification}")
    plan.update({"output": str(output.resolve()), "verification": verification, "telemetry": telemetry})
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = build(args.source.resolve(), args.output.resolve() if args.output else None, write=args.write)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
