#!/usr/bin/env python3.12
"""Fail-closed GGUF -> executable Llama residual-PQ `.gravity` packer.

This is intentionally not a production promotion command.  It writes only
when the caller explicitly acknowledges that a complete artifact still needs
continuation/logit capability evidence.  The output nevertheless carries all
runtime tensors and uses the actual direct-execution grammar, rather than a
storage-only proxy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, dequantize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.operators import gravity_forge as forge
from tools.condense import artifact_client

MAGIC = b"LLM52RPK"
HEADER_BYTES = 64
CODEC = "llama.residual-pq.v1"
PROJECTIONS = (
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
)


def scalar(reader: GGUFReader, name: str) -> int | float:
    field = reader.fields[name]
    return field.parts[-1].item()


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


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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
        "attn_q.bias": "self_attn.q_proj.bias",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.bias": "self_attn.k_proj.bias",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.bias": "self_attn.v_proj.bias",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }
    dst = table.get(suffix)
    return f"model.layers.{layer}.{dst}" if dst else None


def source_tensor(reader: GGUFReader, name: str) -> np.ndarray:
    tensor = next(t for t in reader.tensors if t.name == name)
    # `gguf.dequantize` is the source-quant authority; it also preserves
    # f32/f16 tensors. GGUF's visible dims are reversed for matrices.
    return np.ascontiguousarray(dequantize(tensor.data, tensor.tensor_type), dtype=np.float32)


def train_residual_pq(
    weight: np.ndarray, *, dim: int, stages: int, card: int, seed: int,
    iterations: int, batch_rows: int, reservoir_rows: int,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    """Fit additive stages without retaining a dense reconstructed matrix."""
    rows, cols = weight.shape
    if cols % dim:
        raise ValueError(f"{rows}x{cols} cannot use D={dim}")
    chunks = cols // dim
    vectors = np.ascontiguousarray(weight.reshape(-1, dim), dtype=np.float32)
    residual = vectors.copy()
    indices = np.empty((rows * chunks, stages), dtype=np.uint8)
    codebooks: list[np.ndarray] = []
    torch = forge._torch()
    device = forge._device()
    rng = np.random.default_rng(seed)
    take = min(reservoir_rows, residual.shape[0])
    # A deterministic reservoir bounds centre training memory while every
    # source vector is still assigned and billed in the finished artifact.
    sample = rng.choice(residual.shape[0], size=take, replace=False)
    for stage in range(stages):
        train = torch.from_numpy(np.ascontiguousarray(residual[sample])).to(device)
        centres = forge._kmeans(train, card, iters=iterations, seed=seed + stage)
        centre_cpu = centres.detach().cpu().numpy().astype(np.float32, copy=False)
        codebooks.append(np.ascontiguousarray(centre_cpu))
        for begin in range(0, residual.shape[0], batch_rows):
            end = min(begin + batch_rows, residual.shape[0])
            batch = torch.from_numpy(np.ascontiguousarray(residual[begin:end])).to(device)
            chosen = forge._assign(batch, centres)
            chosen_cpu = chosen.detach().cpu().numpy().astype(np.uint8, copy=False)
            indices[begin:end, stage] = chosen_cpu
            residual[begin:end] -= centre_cpu[chosen_cpu]
    error = float(np.linalg.norm(residual.ravel()) / max(np.linalg.norm(vectors.ravel()), 1e-30))
    return codebooks, indices.reshape(rows, chunks, stages), error


def serialize_residual(
    weight: np.ndarray, *, dim: int, stages: int, card: int, seed: int,
    iterations: int, batch_rows: int, reservoir_rows: int,
) -> tuple[bytes, dict]:
    rows, cols = weight.shape
    codebooks, indices, error = train_residual_pq(
        weight, dim=dim, stages=stages, card=card, seed=seed,
        iterations=iterations, batch_rows=batch_rows, reservoir_rows=reservoir_rows,
    )
    bits = math.ceil(math.log2(card))
    header = MAGIC + struct.pack(
        "<HHHHIIIIHBB", dim, stages, card, 0, rows, cols, cols // dim,
        seed, bits, 0, stages,
    )
    header = header.ljust(HEADER_BYTES, b"\0")
    tables = b"".join(np.ascontiguousarray(table, dtype=np.float16).tobytes() for table in codebooks)
    packed = artifact_client.pack_indices(indices.astype(np.uint32, copy=False), bits)
    blob = header + tables + packed
    return blob, {
        "layout": "fp16[stage][card][D] + msb-first index[row][chunk][stage]",
        "d": dim, "stages": stages, "cardinality": card, "bits": bits,
        "relative_weight_error": error, "active_bytes": len(blob),
        "executed_ops": rows * cols * stages * 2,
        "sequential_dependencies": stages,
        "residency": "device-resident codebooks and indices",
        "parity": "compact CPU/Metal grammar parity required; source capability unproven",
        "fallback": "none; artifact loading is fail-closed",
    }


def architecture(reader: GGUFReader) -> dict:
    gguf_arch = string_field(reader, "general.architecture")
    if gguf_arch not in {"llama", "mistral", "qwen2"}:
        raise ValueError(f"source architecture {gguf_arch!r} is not covered by residual-PQ")
    hidden = int(scalar(reader, f"{gguf_arch}.embedding_length"))
    heads = int(scalar(reader, f"{gguf_arch}.attention.head_count"))
    token_embd = next(t for t in reader.tensors if t.name == "token_embd.weight")
    return {
        "model_type": gguf_arch,
        "num_hidden_layers": int(scalar(reader, f"{gguf_arch}.block_count")),
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": int(scalar(reader, f"{gguf_arch}.attention.head_count_kv")),
        "head_dim": hidden // heads,
        # GGUF matrix metadata is stored in reverse order; the runtime row
        # count is the final visible dimension for token embeddings.
        "vocab_size": int(token_embd.shape[-1]),
        "rope_theta": float(scalar(reader, f"{gguf_arch}.rope.freq_base")),
        "rms_norm_eps": float(scalar(reader, f"{gguf_arch}.attention.layer_norm_rms_epsilon")),
        # The Llama GGUF family uses adjacent-pair (interleaved) RoPE.  The
        # source-preserving Q4/K packer carries this explicitly; the compact
        # candidate must do the same or its low-bit quality gate is measuring
        # a layout mismatch rather than the codec.
        **({"rope_layout": "interleaved"} if gguf_arch in {"llama", "mistral"} else {}),
    }


def expected_names(layers: int) -> set[str]:
    names = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
    for layer in range(layers):
        names.update({
            f"model.layers.{layer}.input_layernorm.weight",
            f"model.layers.{layer}.post_attention_layernorm.weight",
            f"model.layers.{layer}.self_attn.q_proj.bias",
            f"model.layers.{layer}.self_attn.k_proj.bias",
            f"model.layers.{layer}.self_attn.v_proj.bias",
        })
        for suffix in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                       "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                       "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
            names.add(f"model.layers.{layer}.{suffix}")
    return names


def build(args: argparse.Namespace) -> dict:
    reader = GGUFReader(args.source, mode="r")
    arch = architecture(reader)
    mapped = [(t.name, mapped_name(t.name)) for t in reader.tensors]
    mapped = [(src, dst) for src, dst in mapped if dst is not None]
    actual = {dst for _, dst in mapped}
    required = expected_names(arch["num_hidden_layers"])
    missing = sorted(required - actual)
    if missing:
        raise RuntimeError(f"source cannot cover executable Llama runtime: {missing}")
    plan = {
        "schema": "hawking.tg.llama_residual_pq_pack_plan.v1",
        "source": str(Path(args.source).resolve()), "source_sha256": source_sha256(Path(args.source)),
        "architecture": arch, "runtime_required_tensors": len(required),
        "source_mapped_tensors": len(actual), "coverage_complete": not missing,
        "geometry": {"D": args.dim, "stages": args.stages, "cardinality": args.cardinality,
                     "fit_iterations": args.iterations, "reservoir_rows": args.reservoir_rows},
        "status": "PLAN_ONLY_CAPABILITY_UNPROVEN",
    }
    if not args.write:
        return plan
    if not args.unsafe_proxy_only:
        raise RuntimeError("--write requires --unsafe-proxy-only: full continuation/logit capability has not been established")
    payloads, tensor_reports = [], []
    for source_name, target_name in mapped:
        weights = source_tensor(reader, source_name)
        is_projection = source_name in ("token_embd.weight", "output.weight") or source_name.endswith(PROJECTIONS)
        if is_projection:
            blob, grammar = serialize_residual(
                weights, dim=args.dim, stages=args.stages, card=args.cardinality,
                seed=args.seed, iterations=args.iterations, batch_rows=args.batch_rows,
                reservoir_rows=args.reservoir_rows,
            )
            codec = CODEC
        else:
            blob = np.ascontiguousarray(weights, dtype=np.float32).tobytes()
            grammar, codec = {"representation": "exact native f32", "active_bytes": len(blob)}, "native.f32"
        desc = {"name": target_name, "elements": int(weights.size), "shape": list(weights.shape),
                "codec": codec, "bpw": len(blob) * 8 / weights.size, "grammar": grammar}
        payloads.append((desc, blob))
        tensor_reports.append({"name": target_name, "codec": codec, "bytes": len(blob, ), "grammar": grammar})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    telemetry: dict = {}
    artifact_client.write_shard(
        output, payloads, model={"source_gguf_sha256": plan["source_sha256"], "family": "llama"},
        architecture=arch, tokenizer={"source": "GGUF embedded; bound by source hash"},
        compression={"codec": CODEC, "status": "PROXY_ONLY_NOT_CAPABILITY_PROMOTED"}, telemetry=telemetry,
    )
    verify = artifact_client.verify(output)
    if not verify.get("ok"):
        raise RuntimeError(f"written artifact failed integrity: {verify}")
    plan.update({"status": "COMPLETE_ARTIFACT_PROXY_ONLY", "output": str(output),
                 "verify": verify, "telemetry": telemetry, "tensors": tensor_reports})
    return plan


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("artifacts/llama-residual-pq.gravity"))
    p.add_argument("--dim", type=int, default=8)
    p.add_argument("--stages", type=int, default=4)
    p.add_argument("--cardinality", type=int, default=128)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--seed", type=int, default=0x52A1)
    p.add_argument("--batch-rows", type=int, default=65536)
    p.add_argument("--reservoir-rows", type=int, default=262144)
    p.add_argument("--write", action="store_true")
    p.add_argument("--unsafe-proxy-only", action="store_true")
    args = p.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
