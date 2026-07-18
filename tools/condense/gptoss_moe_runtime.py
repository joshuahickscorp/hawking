#!/usr/bin/env python3.12
"""GPT-OSS (120B) per-expert STR2 loader + CPU-reference MoE runtime — Gravity blocker #3.

Resolves the reassembly half of `gptoss-moe-str2-loader-missing`: it reads each MoE tensor
DIRECTLY from its source shard byte range (from the sealed provenance manifest), dequantizes
the MXFP4 (U8) expert projections to fp32 via the vendored `decode_mxfp4_groups_bf16`, and
runs the router -> top-k -> fused gate/up/down (SwiGLU) -> combine on CPU.

HONEST SCOPE. This is the loader + a numerically-runnable CPU REFERENCE, proven against the
real 61 GB source (see selftest). It is NOT yet:
  - the Apple-Silicon Metal fused-expert runtime (`apple_silicon_moe_runtime`), and
  - HF-numerical-parity validated (needs a reference forward + tokenizer eval).
Those two remain, and the capabilities probe must stay False for them until they land.

GPT-OSS 120B MoE geometry (from the provenance manifest, verified):
  gate.weight  BF16 [128, 2880]              router: 128 experts, hidden 2880
  mlp1_weight  MXFP4[128, 5760, 90, 16]      up/gate proj: hidden 2880 -> 5760 (=2*2880)
  mlp2_weight  MXFP4[128, 2880, 90, 16]      down proj:   2880 -> 2880
  mlp{1,2}_bias BF16                          per-expert biases
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import doctor_v5_gptoss_mxfp4 as mxfp4  # noqa: E402
from eco_common import read_json_safe  # noqa: E402

DEFAULT_MANIFEST = "reports/condense/subbit_frontier/GRAVITY_120B_PROVENANCE.json"


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """BF16 stored as uint16 (top 16 bits of fp32) -> fp32."""
    u16 = np.asarray(bits, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


class ProvenanceReader:
    """Reads a tensor straight from its source shard byte range (STR2 reassembly primitive)."""

    def __init__(self, manifest_path: str | os.PathLike[str] = DEFAULT_MANIFEST):
        self.manifest = read_json_safe(manifest_path)
        self.by_name = {t["tensor"]: t for t in self.manifest["tensors"]}

    def raw(self, tensor_name: str) -> tuple[bytes, dict[str, Any]]:
        t = self.by_name.get(tensor_name)
        if t is None:
            raise KeyError(f"tensor not in provenance manifest: {tensor_name}")
        start, end = t["byte_range"]
        with Path(t["shard_path"]).open("rb") as fh:
            fh.seek(start)
            blob = fh.read(end - start)
        if len(blob) != end - start:
            raise IOError(f"short read for {tensor_name}: {len(blob)} != {end - start}")
        return blob, t

    def bf16(self, tensor_name: str) -> np.ndarray:
        blob, t = self.raw(tensor_name)
        arr = _bf16_bits_to_f32(np.frombuffer(blob, dtype=np.uint16))
        return arr.reshape(t["shape"])

    def u8(self, tensor_name: str) -> np.ndarray:
        blob, t = self.raw(tensor_name)
        return np.frombuffer(blob, dtype=np.uint8).reshape(t["shape"])


def load_router(reader: ProvenanceReader, block: int) -> dict[str, np.ndarray]:
    """Router (gate) weights for a block, straight from source bytes (BF16 -> fp32)."""
    return {"weight": reader.bf16(f"block.{block}.mlp.gate.weight"),   # [128, 2880]
            "bias": reader.bf16(f"block.{block}.mlp.gate.bias")}       # [128]


def load_expert(reader: ProvenanceReader, block: int, expert: int) -> dict[str, np.ndarray]:
    """Load + dequantize ONE expert's projections from the source (MXFP4 U8 -> fp32).

    mlp1: [5760, 2880] (up/gate), mlp2: [2880, 2880] (down), plus BF16 biases.
    """
    def dequant(kind: str, out_rows: int) -> np.ndarray:
        blocks = reader.u8(f"block.{block}.mlp.{kind}.blocks")[expert]   # [out, 90, 16]
        scales = reader.u8(f"block.{block}.mlp.{kind}.scales")[expert]   # [out, 90]
        bits = mxfp4.decode_mxfp4_groups_bf16(blocks, scales)            # BF16 bits [out, 90*32]
        w = _bf16_bits_to_f32(np.asarray(bits))
        return w.reshape(out_rows, -1)

    mlp1 = dequant("mlp1_weight", 5760)   # [5760, 2880]
    mlp2 = dequant("mlp2_weight", 2880)   # [2880, 2880]
    b1 = reader.bf16(f"block.{block}.mlp.mlp1_bias")[expert]   # [5760]
    b2 = reader.bf16(f"block.{block}.mlp.mlp2_bias")[expert]   # [2880]
    return {"mlp1": mlp1, "mlp2": mlp2, "mlp1_bias": b1, "mlp2_bias": b2}


def _swiglu(h: np.ndarray) -> np.ndarray:
    """GPT-OSS gated activation: split into (gate, up), SiLU(gate) * up. Reference structure;
    the exact clamp/alpha constants must be reconciled against HF before parity claims."""
    gate, up = np.split(h, 2, axis=-1)
    return (gate * (1.0 / (1.0 + np.exp(-gate)))) * up


def moe_forward_reference(x: np.ndarray, router: dict[str, np.ndarray],
                          load_expert_fn, *, top_k: int = 4) -> np.ndarray:
    """CPU reference: router softmax -> top-k experts -> fused mlp1/SwiGLU/mlp2 -> weighted sum.
    x: [hidden]=2880. Returns [hidden]. Loads only the selected experts (bounded memory)."""
    logits = router["weight"] @ x + router["bias"]           # [128]
    idx = np.argsort(-logits)[:top_k]
    w = logits[idx]
    w = np.exp(w - w.max()); w = w / w.sum()                  # softmax over top-k
    out = np.zeros_like(x)
    for e, gate_w in zip(idx, w):
        ex = load_expert_fn(int(e))
        h = ex["mlp1"] @ x + ex["mlp1_bias"]                  # [5760]
        a = _swiglu(h)                                        # [2880]
        y = ex["mlp2"] @ a + ex["mlp2_bias"]                  # [2880]
        out += gate_w * y
    return out


def selftest(manifest_path: str = DEFAULT_MANIFEST, *, block: int = 0) -> dict[str, Any]:
    """Prove the loader against the REAL 120B source: load router + expert 0, dequant, run one
    forward. Skips gracefully (ok=None) if the source shards are not present."""
    reader = ProvenanceReader(manifest_path)
    sample = reader.by_name.get(f"block.{block}.mlp.gate.weight")
    if sample is None or not Path(sample["shard_path"]).exists():
        return {"ok": None, "reason": "120B source shards not present; loader not exercised"}

    router = load_router(reader, block)
    assert router["weight"].shape == (128, 2880), router["weight"].shape
    ex0 = load_expert(reader, block, 0)
    assert ex0["mlp1"].shape == (5760, 2880), ex0["mlp1"].shape
    assert ex0["mlp2"].shape == (2880, 2880), ex0["mlp2"].shape
    assert np.isfinite(ex0["mlp1"]).all() and np.isfinite(ex0["mlp2"]).all()

    rng = np.random.default_rng(0)
    x = rng.standard_normal(2880).astype(np.float32) * 0.02
    cache: dict[int, dict[str, np.ndarray]] = {}
    def loader(e: int):
        if e not in cache:
            cache[e] = load_expert(reader, block, e)
        return cache[e]
    y = moe_forward_reference(x, router, loader, top_k=4)
    assert y.shape == (2880,) and np.isfinite(y).all()
    return {"ok": True, "block": block, "router_shape": list(router["weight"].shape),
            "mlp1_shape": list(ex0["mlp1"].shape), "mlp2_shape": list(ex0["mlp2"].shape),
            "experts_loaded": len(cache), "forward_out_finite": True,
            "note": "CPU reference proven on real source; Metal runtime + HF parity remain"}


def main(argv: list[str] | None = None) -> int:
    import json
    ap = argparse.ArgumentParser(description="GPT-OSS 120B per-expert loader + CPU MoE reference.")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    print(json.dumps(selftest(args.manifest, block=args.block), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
