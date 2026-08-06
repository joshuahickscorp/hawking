#!/usr/bin/env python3
"""Pack a Llama GGUF into a source-preserving executable Gravity shard.

This is intentionally a container/runtime gate, not a compression claim.  Q4_K
and Q6_K payloads are copied byte-for-byte and the resident Gravity adapter
dispatches the same b9430 source kernels used by the strict GGUF lane.  Norms
remain native.f32.  A lossy representation must not be mixed into this gate.
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


SCHEMA = "hawking.tg.llama_source_quant_gravity_pack.v1"
Q4_K = 12
Q5_K = 13
Q6_K = 14
Q5_0 = 6
Q8_0 = 8
F32 = 0
F16 = 1
BF16 = 30


def scalar(reader: GGUFReader, name: str) -> int | float:
    field = reader.fields[name]
    return field.parts[-1].item()


def string_field(reader: GGUFReader, name: str) -> str:
    """Read a GGUF string field from the reader's byte-valued tail."""
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


def mapped_name(name: str) -> str | None:
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output.weight":
        return "lm_head.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    if not name.startswith("blk."):
        return None
    layer, suffix = name.split(".", 2)[1:]
    table = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_q.bias": "self_attn.q_proj.bias",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_k.bias": "self_attn.k_proj.bias",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_v.bias": "self_attn.v_proj.bias",
        "attn_output.weight": "self_attn.o_proj.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }
    target = table.get(suffix)
    return f"model.layers.{layer}.{target}" if target else None


def runtime_shape(tensor) -> list[int]:
    """GGUF dimensions are reversed; the memmap's logical row is last dim."""
    shape = [int(v) for v in tensor.shape]
    if len(shape) == 1:
        return shape
    if len(shape) != 2:
        raise ValueError(f"unsupported tensor rank {len(shape)} for {tensor.name}")
    return [shape[-1], shape[-2]]


def architecture(reader: GGUFReader, rope_layout_override: str | None = None) -> dict:
    gguf_arch = string_field(reader, "general.architecture")
    # Mistral GGUFs use the same dense Llama-family tensor namespace and
    # executable packed grammar, but keep their own metadata prefix.  Do not
    # normalize arbitrary architectures here: the resident runtime has an
    # explicit `mistral` branch and every other architecture remains rejected.
    if gguf_arch not in {"llama", "mistral", "qwen2"}:
        raise ValueError(
            f"source architecture {gguf_arch!r} is not covered by the dense Llama-family packer"
        )
    prefix = gguf_arch
    hidden = int(scalar(reader, f"{prefix}.embedding_length"))
    heads = int(scalar(reader, f"{prefix}.attention.head_count"))
    factors = next((tensor for tensor in reader.tensors if tensor.name == "rope_freqs.weight"), None)
    if factors is not None:
        rope_freq_factors = [float(v) for v in factors.data.tolist()]
    else:
        rope_freq_factors = None
    token_embd = next((tensor for tensor in reader.tensors if tensor.name == "token_embd.weight"), None)
    if token_embd is None:
        raise ValueError("source has no token_embd.weight tensor")
    vocab_size = runtime_shape(token_embd)[0]
    return {
        "model_type": prefix,
        "num_hidden_layers": int(scalar(reader, f"{prefix}.block_count")),
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": int(scalar(reader, f"{prefix}.attention.head_count_kv")),
        "head_dim": int(
            scalar(reader, f"{prefix}.attention.key_length")
            if f"{prefix}.attention.key_length" in reader.fields
            else hidden // heads
        ),
        "vocab_size": vocab_size,
        "rope_theta": float(scalar(reader, f"{prefix}.rope.freq_base")),
        "rms_norm_eps": float(scalar(reader, f"{prefix}.attention.layer_norm_rms_epsilon")),
        # Llama/Mistral GGUFs use adjacent-pair (interleaved) RoPE. Qwen2's
        # NeoX convention is the historical split-half default, so omit the
        # field for that family rather than silently changing its layout.
        **({"rope_layout": rope_layout_override or "interleaved"} if (rope_layout_override or prefix in {"llama", "mistral"}) else {}),
        "rope_freq_factors": rope_freq_factors,
    }


def codec_and_geometry(tensor, shape: list[int]) -> tuple[str, int]:
    """Return the executable raw-quant grammar and exact GGML block geometry."""
    dtype = int(tensor.tensor_type)
    if dtype == Q4_K:
        codec, block_bytes, block_elements = "ggml.q4_k", 144, 256
    elif dtype == Q5_K:
        codec, block_bytes, block_elements = "ggml.q5_k", 176, 256
    elif dtype == Q6_K:
        codec, block_bytes, block_elements = "ggml.q6_k", 210, 256
    elif dtype == Q5_0:
        codec, block_bytes, block_elements = "ggml.q5_0", 22, 32
    elif dtype == Q8_0:
        codec, block_bytes, block_elements = "ggml.q8_0", 34, 32
    elif dtype == F32:
        return "native.f32", 0
    elif dtype == F16:
        return "native.f16", 0
    elif dtype == BF16:
        return "native.bf16", 0
    else:
        raise ValueError(f"{tensor.name}: unsupported source GGML dtype {dtype}")
    if len(shape) != 2 or shape[1] % block_elements:
        raise ValueError(
            f"{tensor.name}: {codec} shape must be [rows, cols%{block_elements}==0], got {shape}"
        )
    expected = shape[0] * (shape[1] // block_elements) * block_bytes
    if int(tensor.data.nbytes) != expected:
        raise ValueError(f"{tensor.name}: raw bytes {tensor.data.nbytes} != expected {expected}")
    return codec, block_bytes


def build(
    source: Path,
    output: Path | None,
    *,
    write: bool,
    rope_layout_override: str | None = None,
    tokenizer_json: Path | None = None,
) -> dict:
    reader = GGUFReader(str(source), mode="r")
    source_hash = sha256(source)
    arch = architecture(reader, rope_layout_override=rope_layout_override)
    payloads: list[tuple[dict, bytes]] = []
    mapped = 0
    skipped: list[str] = []
    compressed_bytes = compressed_elements = 0
    complete_bytes = complete_elements = 0
    for tensor in reader.tensors:
        target = mapped_name(tensor.name)
        if target is None:
            skipped.append(tensor.name)
            continue
        shape = runtime_shape(tensor)
        codec, _ = codec_and_geometry(tensor, shape)
        blob = tensor.data.tobytes(order="C")
        elements = 1
        for dim in shape:
            elements *= dim
        descriptor = {
            "name": target,
            "shape": shape,
            "codec": codec,
            "elements": elements,
            "bpw": len(blob) * 8 / max(1, elements),
            "terminal_state": "SOURCE_QUANT_BYTES_COPIED",
            "source_tensor": tensor.name,
        }
        payloads.append((descriptor, blob))
        mapped += 1
        complete_bytes += len(blob)
        complete_elements += elements
        if not codec.startswith("native."):
            compressed_bytes += len(blob)
            compressed_elements += elements

    if not payloads:
        raise ValueError("source had no executable Llama tensors")
    names = [d["name"] for d, _ in payloads]
    if len(names) != len(set(names)):
        raise ValueError("mapped source tensors contain duplicate canonical names")
    compression = {
        "codec": "ggml-source-q4_k-q5_k-q5_0-q6_k-q8_0",
        "source_quantization_preserved": True,
        "packed_bpw": compressed_bytes * 8 / max(1, compressed_elements),
        "complete_bpw": complete_bytes * 8 / max(1, complete_elements),
        "lossy_fit": False,
    }
    tokenizer = {"kind": "embedded-gguf", "source_sha256": source_hash}
    if tokenizer_json is not None:
        if not tokenizer_json.is_file():
            raise ValueError(f"tokenizer JSON is not a file: {tokenizer_json}")
        tokenizer = {
            "kind": "tokenizer-json",
            "dir": str(tokenizer_json.parent.resolve()),
            "source": tokenizer_json.name,
            "sha256": sha256(tokenizer_json),
        }
    plan = {
        "schema": SCHEMA,
        "status": "SOURCE_COVERAGE_READY_NOT_WRITTEN" if not write else "WRITTEN",
        "source": str(source.resolve()),
        "source_sha256": source_hash,
        "architecture": arch,
        "mapped_tensors": mapped,
        "skipped_source_tensors": skipped,
        "payload_bytes": complete_bytes,
        "compressed_bytes": compressed_bytes,
        "complete_bpw": compression["complete_bpw"],
        "packed_bpw": compression["packed_bpw"],
        "lossy_fit": False,
        "runtime": (
            "GravityLlama direct raw-quant CPU grammar for Q4_K/Q5_0/Q6_K/Q8_0; "
            "GPU direct packed grammar admits Q4_K/Q5_0/Q6_K/Q8_0 and native tensors; "
            "a model-specific Numeric Parity V2.1 receipt remains required before any promotion"
        ),
        "tokenizer": tokenizer,
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
        model={"family": arch["model_type"], "source_gguf_sha256": source_hash},
        architecture=arch,
        tokenizer=tokenizer,
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
    parser.add_argument(
        "--rope-layout",
        choices=("interleaved", "split_half"),
        help="override the source layout for a diagnostic alternate artifact",
    )
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        help="immutable tokenizer.json staged beside the artifact and hash-bound in its header",
    )
    args = parser.parse_args()
    result = build(
        args.source.resolve(),
        args.output.resolve() if args.output else None,
        write=args.write,
        rope_layout_override=args.rope_layout,
        tokenizer_json=args.tokenizer_json.resolve() if args.tokenizer_json else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
