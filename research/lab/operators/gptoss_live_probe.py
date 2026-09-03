#!/usr/bin/env python3
"""Bounded live-source probe for OpenAI GPT-OSS-120B.

This is deliberately an operator, not a second campaign controller. It binds
the retained provenance manifest to the pinned original safetensors shards,
reads only requested tensor ranges, decodes one expert at a time from MXFP4,
and executes one real router/top-4 expert wave with the GPT-OSS activation.

It does not claim full-model K0, Metal residency, or a TG rung.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "workspace/campaign/records/reports/condense/subbit_frontier/GRAVITY_120B_PROVENANCE.json"
HIDDEN = 2880
EXPERTS = 128
TOP_K = 4
ALPHA = np.float32(1.702)
LIMIT = np.float32(7.0)
FP4_VALUES = np.asarray(
    (
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ),
    dtype=np.float32,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    u16 = np.asarray(bits, dtype=np.uint16)
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def decode_mxfp4_groups_bf16(blocks: Any, scales: Any) -> np.ndarray:
    """Decode ``[..., groups, packed_bytes]`` to canonical BF16 bits."""
    packed = np.asarray(blocks, dtype=np.uint8)
    scale_bytes = np.asarray(scales, dtype=np.uint8)
    if packed.ndim < 2 or tuple(packed.shape[:-1]) != tuple(scale_bytes.shape):
        raise ValueError("blocks/scales shapes do not satisfy MXFP4 pairing")
    expanded = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.float32)
    expanded[..., 0::2] = FP4_VALUES[packed & np.uint8(0x0F)]
    expanded[..., 1::2] = FP4_VALUES[packed >> np.uint8(4)]
    exponents = scale_bytes.astype(np.int32) - 127
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        np.ldexp(expanded, exponents[..., None], out=expanded)
    reserved = exponents[..., None] == 128
    reserved_zero = reserved & (expanded == 0)
    reserved_nonzero = reserved & ~reserved_zero
    expanded[reserved_zero] = np.float32("nan")
    expanded[reserved_nonzero] = np.copysign(np.float32("inf"), expanded[reserved_nonzero])
    flat_shape = (*packed.shape[:-2], packed.shape[-2] * packed.shape[-1] * 2)
    bits = expanded.reshape(flat_shape).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2", copy=False)


class ProvenanceReader:
    """Read exactly one manifest-bound tensor byte range at a time."""

    def __init__(self, manifest_path: Path = DEFAULT_MANIFEST):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.by_name = {row["tensor"]: row for row in self.manifest["tensors"]}

    def raw(self, tensor_name: str) -> tuple[bytes, dict[str, Any]]:
        row = self.by_name[tensor_name]
        start, end = row["byte_range"]
        path = Path(row["shard_path"])
        with path.open("rb", buffering=0) as handle:
            handle.seek(start)
            blob = handle.read(end - start)
        if len(blob) != end - start:
            raise OSError(f"short read for {tensor_name}: {len(blob)} != {end - start}")
        return blob, row

    def bf16(self, tensor_name: str) -> np.ndarray:
        blob, row = self.raw(tensor_name)
        return _bf16_bits_to_f32(np.frombuffer(blob, dtype="<u2")).reshape(row["shape"])

    def u8(self, tensor_name: str) -> np.ndarray:
        blob, row = self.raw(tensor_name)
        return np.frombuffer(blob, dtype=np.uint8).reshape(row["shape"])


def load_router(reader: ProvenanceReader, block: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        reader.bf16(f"block.{block}.mlp.gate.weight"),
        reader.bf16(f"block.{block}.mlp.gate.bias"),
    )


def load_expert(reader: ProvenanceReader, block: int, expert: int) -> dict[str, np.ndarray]:
    if not 0 <= expert < EXPERTS:
        raise ValueError(f"expert out of range: {expert}")

    def projection(kind: str, out_rows: int) -> np.ndarray:
        blocks = reader.u8(f"block.{block}.mlp.{kind}.blocks")[expert]
        scales = reader.u8(f"block.{block}.mlp.{kind}.scales")[expert]
        bits = decode_mxfp4_groups_bf16(blocks, scales)
        return _bf16_bits_to_f32(bits).reshape(out_rows, HIDDEN)

    return {
        "mlp1": projection("mlp1_weight", 2 * HIDDEN),
        "mlp2": projection("mlp2_weight", HIDDEN),
        "mlp1_bias": reader.bf16(f"block.{block}.mlp.mlp1_bias")[expert],
        "mlp2_bias": reader.bf16(f"block.{block}.mlp.mlp2_bias")[expert],
    }


def apply_gate(gate_up: np.ndarray) -> np.ndarray:
    """Transformers/OpenAI GPT-OSS interleaved, clamped gated activation."""
    gate = np.minimum(gate_up[..., ::2], LIMIT)
    up = np.clip(gate_up[..., 1::2], -LIMIT, LIMIT)
    glu = gate * (np.float32(1.0) / (np.float32(1.0) + np.exp(-ALPHA * gate)))
    return (up + np.float32(1.0)) * glu


def expert_wave(reader: ProvenanceReader, block: int, x: np.ndarray) -> dict[str, Any]:
    router_w, router_b = load_router(reader, block)
    logits = router_w @ x + router_b
    selected = np.argsort(-logits)[:TOP_K]
    route_weights = logits[selected]
    route_weights = np.exp(route_weights - route_weights.max())
    route_weights /= route_weights.sum()
    output = np.zeros(HIDDEN, dtype=np.float32)
    expert_digests: list[dict[str, Any]] = []
    for expert, route_weight in zip(selected, route_weights):
        loaded = load_expert(reader, block, int(expert))
        hidden = loaded["mlp1"] @ x + loaded["mlp1_bias"]
        activated = apply_gate(hidden)
        contribution = loaded["mlp2"] @ activated + loaded["mlp2_bias"]
        output += np.float32(route_weight) * contribution
        expert_digests.append(
            {
                "expert": int(expert),
                "route_weight": float(route_weight),
                "mlp1_sha256": _sha256_bytes(loaded["mlp1"].tobytes()),
                "mlp2_sha256": _sha256_bytes(loaded["mlp2"].tobytes()),
            }
        )
    return {
        "selected_experts": selected.astype(int).tolist(),
        "experts": expert_digests,
        "output": output,
        "router_shape": list(router_w.shape),
    }


def live_probe(manifest_path: Path = DEFAULT_MANIFEST, block: int = 0) -> dict[str, Any]:
    reader = ProvenanceReader(manifest_path)
    missing = sorted(
        {str(row["shard_path"]) for row in reader.by_name.values() if not Path(row["shard_path"]).is_file()}
    )
    if missing:
        return {"status": "SOURCE_ABSENT", "missing": missing}
    rng = np.random.default_rng(0)
    x = rng.standard_normal(HIDDEN).astype(np.float32) * np.float32(0.02)
    started = time.perf_counter()
    result = expert_wave(reader, block, x)
    elapsed = time.perf_counter() - started
    output = result.pop("output")
    return {
        "schema": "hawking.gptoss.live_probe.v1",
        "status": "PASS_BOUNDED_SOURCE_EXPERT_WAVE",
        "manifest": str(manifest_path),
        "manifest_declared_sha256": reader.manifest.get("manifest_sha256"),
        "block": block,
        "source_shards": len(reader.manifest["shards"]),
        "tensor_count": len(reader.by_name),
        **result,
        "input_sha256": _sha256_bytes(x.tobytes()),
        "output_sha256": _sha256_bytes(output.tobytes()),
        "output_finite": bool(np.isfinite(output).all()),
        "output_rms": float(np.sqrt(np.mean(output * output))),
        "wall_seconds": elapsed,
        "execution": "CPU_REFERENCE",
        "claim_boundary": "one real router/top-4 expert wave; no full-model token, Metal, or TG claim",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--block", type=int, default=0)
    args = parser.parse_args(argv)
    result = live_probe(args.manifest, args.block)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
