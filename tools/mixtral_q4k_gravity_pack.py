#!/usr/bin/env python3
"""Pack a Mixtral GGUF into the executable Gravity MoE container.

The source GGUF is the authority and is never dequantized here.  Fused expert
planes are split into the exact per-expert grammar already consumed by the
resident Metal Mixtral engine.  The resulting artifact is therefore a real
Gravity representation (seekable, hash-bound, and directly executable), not a
renamed GGUF or a dense reconstruction.  This first pass is source-preserving;
lossy expert codebooks are a separate, capability-gated representation.
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


SCHEMA = "hawking.tg.mixtral_source_quant_gravity_pack.v1"
F32, F16, BF16 = 0, 1, 30
Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 = 11, 12, 13, 14, 8


def _plain(value):
    if hasattr(value, "item") and getattr(value, "size", 1) == 1:
        return value.item()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8")
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def field_value(field):
    """Decode one python-gguf field without losing tokenizer arrays."""
    kind = field.types[0].name
    if kind == "ARRAY":
        subtype = field.types[1].name
        count = int(_plain(field.parts[4]))
        cursor = 5
        out = []
        for _ in range(count):
            if subtype == "STRING":
                # Each array string is encoded as length followed by bytes.
                length = int(_plain(field.parts[cursor]))
                cursor += 1
                raw = field.parts[cursor]
                cursor += 1
                raw = raw.tolist() if hasattr(raw, "tolist") else raw
                if isinstance(raw, int):
                    raw = [raw]
                if not isinstance(raw, (bytes, bytearray)):
                    raw = bytes(int(v) for v in raw)
                if len(raw) != length:
                    raise ValueError(f"array string length {len(raw)} != {length}")
                out.append(bytes(raw).decode("utf-8"))
            else:
                out.append(_plain(field.parts[cursor]))
                cursor += 1
        return out
    if kind == "STRING":
        value = field.parts[-1].tolist() if hasattr(field.parts[-1], "tolist") else field.parts[-1]
        if isinstance(value, int):
            value = [value]
    else:
        value = _plain(field.parts[-1])
    if kind == "STRING" and not isinstance(value, str):
        value = bytes(int(v) for v in value).decode("utf-8")
    if kind == "BOOL":
        return bool(value)
    return value


def source_metadata(reader: GGUFReader) -> dict:
    # Keep the exact embedded tokenizer and all function-changing GGUF
    # metadata.  The header is small compared with a 28 GB model body and
    # permits a Gravity-only load after the source body is evicted.
    return {name: field_value(field) for name, field in reader.fields.items()
            if not name.startswith("GGUF.")}


def scalar(metadata: dict, name: str) -> int | float:
    value = metadata[name]
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected scalar, got {type(value).__name__}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_codec(dtype: int) -> tuple[str, int, int]:
    table = {
        F32: ("native.f32", 1, 4),
        F16: ("native.f16", 1, 2),
        BF16: ("native.bf16", 1, 2),
        Q3_K: ("ggml.q3_k", 256, 110),
        Q4_K: ("ggml.q4_k", 256, 144),
        Q5_K: ("ggml.q5_k", 256, 176),
        Q6_K: ("ggml.q6_k", 256, 210),
        Q8_0: ("ggml.q8_0", 32, 34),
    }
    try:
        return table[int(dtype)]
    except KeyError as exc:
        raise ValueError(f"unsupported Mixtral GGML dtype {dtype}") from exc


def runtime_shape(source_shape: list[int]) -> list[int]:
    if len(source_shape) == 1:
        return source_shape
    if len(source_shape) == 2:
        return [source_shape[1], source_shape[0]]
    raise ValueError(f"expected rank-1/2 source tensor, got rank {len(source_shape)}")


def descriptor_and_blob(tensor, *, expert: int | None = None) -> tuple[dict, bytes]:
    source_shape = [int(v) for v in tensor.shape]
    codec, block_elements, block_bytes = source_codec(int(tensor.tensor_type))
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
    if len(blob) != (elements // block_elements) * block_bytes:
        raise ValueError(
            f"{tensor.name}: bytes {len(blob)} do not match {shape} {codec}"
        )
    return ({
        "name": "",
        "shape": shape,
        "codec": codec,
        "elements": elements,
        "bpw": len(blob) * 8 / max(1, elements),
        "terminal_state": "SOURCE_QUANT_BYTES_COPIED",
        "source_tensor": tensor.name,
        **({"source_expert": expert} if expert is not None else {}),
    }, blob)


def expert_target(name: str, expert: int) -> str | None:
    if not name.startswith("blk."):
        return None
    layer, suffix = name.split(".", 2)[1:]
    suffix_map = {
        "ffn_gate_exps.weight": "ffn_gate",
        "ffn_up_exps.weight": "ffn_up",
        "ffn_down_exps.weight": "ffn_down",
    }
    stem = suffix_map.get(suffix)
    return f"blk.{layer}.{stem}.{expert}.weight" if stem else None


def build(source: Path, output: Path | None, *, write: bool) -> dict:
    reader = GGUFReader(str(source), mode="r")
    metadata = source_metadata(reader)
    if metadata.get("general.architecture") != "llama":
        raise ValueError(f"expected llama Mixtral GGUF, got {metadata.get('general.architecture')!r}")
    source_hash = sha256(source)
    payloads: list[tuple[dict, bytes]] = []
    skipped: list[str] = []
    mapped = 0
    complete_bytes = compressed_bytes = 0
    complete_elements = compressed_elements = 0
    for tensor in reader.tensors:
        if tensor.name.endswith("_exps.weight"):
            shape = [int(v) for v in tensor.shape]
            if len(shape) != 3:
                raise ValueError(f"{tensor.name}: expected rank-3 fused expert tensor")
            for expert in range(shape[2]):
                target = expert_target(tensor.name, expert)
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
        # Every non-expert tensor is kept under its original GGUF spelling so
        # the existing Mixtral adapter can consume the artifact without a
        # second canonical-name table.
        if tensor.name.startswith("blk.") or tensor.name in {
            "token_embd.weight", "output.weight", "output_norm.weight"
        }:
            descriptor, blob = descriptor_and_blob(tensor)
            descriptor["name"] = tensor.name
            payloads.append((descriptor, blob))
            mapped += 1
            complete_bytes += len(blob)
            complete_elements += descriptor["elements"]
            if descriptor["codec"].startswith("ggml."):
                compressed_bytes += len(blob)
                compressed_elements += descriptor["elements"]
        else:
            skipped.append(tensor.name)
    if not payloads:
        raise ValueError("source had no executable Mixtral tensors")
    names = [descriptor["name"] for descriptor, _ in payloads]
    if len(names) != len(set(names)):
        raise ValueError("Mixtral Gravity tensor map contains duplicate names")
    if not any(name == "blk.0.ffn_gate.0.weight" for name in names):
        raise ValueError("fused Mixtral experts were not expanded")

    compression = {
        "codec": "ggml-source-q3_k-q4_k-q5_k-q6_k-q8_0",
        "source_quantization_preserved": True,
        "packed_bpw": compressed_bytes * 8 / max(1, compressed_elements),
        "complete_bpw": complete_bytes * 8 / max(1, complete_elements),
        "lossy_fit": False,
    }
    arch = {
        "model_type": "mixtral",
        "source_gguf_architecture": "llama",
        "num_hidden_layers": int(scalar(metadata, "llama.block_count")),
        "hidden_size": int(scalar(metadata, "llama.embedding_length")),
        "num_attention_heads": int(scalar(metadata, "llama.attention.head_count")),
        "num_key_value_heads": int(scalar(metadata, "llama.attention.head_count_kv")),
        "head_dim": int(scalar(metadata, "llama.embedding_length")) // int(scalar(metadata, "llama.attention.head_count")),
        "intermediate_size": int(scalar(metadata, "llama.feed_forward_length")),
        "n_experts": int(scalar(metadata, "llama.expert_count")),
        "top_k": int(scalar(metadata, "llama.expert_used_count")),
        "vocab_size": int(scalar(metadata, "llama.vocab_size")),
        "rope_theta": float(scalar(metadata, "llama.rope.freq_base")),
        "rms_norm_eps": float(scalar(metadata, "llama.attention.layer_norm_rms_epsilon")),
        "max_position_embeddings": int(scalar(metadata, "llama.context_length")),
        "total_parameters": "46.7B",
        "active_parameters": "12.9B",
        "experts": 8,
        "active_experts_per_token": 2,
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
        "runtime": "Gravity Mixtral direct raw GGML expert-split grammar; no dense reconstruction",
        "metadata_keys": len(metadata),
        "tokenizer": {"kind": "embedded-gguf", "source_sha256": source_hash},
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
        model={
            "family": "mistral_mixtral_sparse_moe",
            "repo": "mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF",
            "revision": "92bb790b153033f0594934ee23bb0e12ba897f4e",
            "source_gguf_sha256": source_hash,
            "total_parameters": "46.7B",
            "active_parameters": "12.9B",
        },
        architecture=arch,
        tokenizer=plan["tokenizer"],
        compression=compression,
        gguf_metadata=metadata,
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
