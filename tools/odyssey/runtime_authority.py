#!/usr/bin/env python3.12
"""Runtime reproduction: bit-identical CPU-authority logits, twice.

Full-model forward of the 92 GB Math-Preserve artifact is not feasible on a
96 GB machine already under load (~0.16 GB free observed). This module:

1. Loads a real native tensor from a Math-Preserve shard (artifact-bound).
2. Runs a fixed-prompt CPU-authority path (RMSNorm + fixed projection) twice
   and requires bit-identical float32 logits.
3. Additionally runs the documented functional-codec CPU authority twice
   (lab/operators/gravity_functional_codec.execute) for organ-level
   bit-identity.

What is NOT checked: full multi-layer GLM-5.2 forward, full vocab lm_head
logits (lm_head lives on a 3.8 GB first shard; not loadable under current free
RAM without risking the concurrent campaign), Metal paths.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tools.odyssey._paths import MATH_ARTIFACT, ROOT

SCHEMA = "hawking.odyssey.t0.runtime_authority.v1"
FIXED_PROMPT = "Odyssey T0 baseline reproduction prompt: 1+1="
PROMPT_SEED = 0x4F44595353455954  # "ODYSSEY T" as u64-ish seed material
PREFIX_STRUCT = "<8sIQ"
PREFIX_BYTES = struct.calcsize(PREFIX_STRUCT)


def _read_tensor_payload(shard_path: Path, tensor_name: str) -> tuple[dict, bytes]:
    with shard_path.open("rb") as fh:
        prefix = fh.read(PREFIX_BYTES)
        magic, _version, header_length = struct.unpack(PREFIX_STRUCT, prefix)
        if magic != b"GRAVITY\x00":
            raise ValueError(f"{shard_path.name}: not a .gravity shard")
        header = json.loads(fh.read(header_length))
        base = PREFIX_BYTES + header_length
        entry = next((t for t in header["tensors"] if t["name"] == tensor_name), None)
        if entry is None:
            raise KeyError(tensor_name)
        fh.seek(base + int(entry["offset"]))
        blob = fh.read(int(entry["bytes"]))
    if len(blob) != int(entry["bytes"]):
        raise ValueError(f"truncated tensor {tensor_name}")
    if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
        raise ValueError(f"tensor integrity failed: {tensor_name}")
    return entry, blob


def _bf16_to_f32(blob: bytes) -> np.ndarray:
    """Interpret little-endian bf16 payload as float32 via bit shift."""
    u16 = np.frombuffer(blob, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).copy()


def _fixed_hidden(dim: int, seed: int = PROMPT_SEED) -> np.ndarray:
    """Deterministic 'prompt' embedding: hash the fixed prompt into a vector."""
    h = hashlib.sha256(FIXED_PROMPT.encode("utf-8") + seed.to_bytes(8, "little")).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], "little"))
    return rng.randn(dim).astype(np.float32)


def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    var = np.mean(x * x)
    return (x * np.float32(1.0 / np.sqrt(var + eps))) * weight


def _project_to_logits(hidden: np.ndarray, seed: int = PROMPT_SEED, n_logits: int = 64) -> np.ndarray:
    """Deterministic fixed projection used as a single-layer authority stand-in.

    Not a claim of full-model lm_head. Produces a stable float32 logit vector
    whose bit-identity across two runs is the reproducibility property under test.
    """
    h = hashlib.sha256(b"odyssey-t0-proj" + seed.to_bytes(8, "little")).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], "little"))
    w = rng.randn(hidden.shape[-1], n_logits).astype(np.float32) * np.float32(0.02)
    return hidden @ w


def artifact_single_layer_authority(artifact: Path = MATH_ARTIFACT) -> dict[str, Any]:
    """Load model.norm.weight from Math-Preserve, RMSNorm + project, twice."""
    shard_name = "model-00282-of-00282.gravity"
    tensor_name = "model.norm.weight"
    shard_path = artifact / shard_name
    if not shard_path.is_file():
        return {
            "status": "FAIL",
            "reason": f"shard missing: {shard_path}",
            "checked": [],
            "skipped": ["full model forward (memory)", "lm_head (3.8GB shard under free-RAM pressure)"],
        }

    entry, blob = _read_tensor_payload(shard_path, tensor_name)
    weight = _bf16_to_f32(blob)
    dim = int(weight.shape[0])
    x = _fixed_hidden(dim)

    def once() -> np.ndarray:
        return _project_to_logits(_rmsnorm(x, weight))

    a = once()
    b = once()
    identical = bool(np.array_equal(a, b))
    # Re-read tensor from disk and recompute — catches non-determinism in IO path.
    _entry2, blob2 = _read_tensor_payload(shard_path, tensor_name)
    weight2 = _bf16_to_f32(blob2)
    c = _project_to_logits(_rmsnorm(x, weight2))
    identical_reread = bool(np.array_equal(a, c))

    return {
        "status": "PASS" if identical and identical_reread else "FAIL",
        "path": "artifact_native_tensor + rmsnorm + fixed_projection",
        "shard": shard_name,
        "tensor": tensor_name,
        "tensor_sha256": entry["sha256"],
        "prompt": FIXED_PROMPT,
        "logit_shape": list(a.shape),
        "logit_sha256_run1": hashlib.sha256(a.tobytes()).hexdigest(),
        "logit_sha256_run2": hashlib.sha256(b.tobytes()).hexdigest(),
        "bit_identical_two_runs": identical,
        "bit_identical_after_reread": identical_reread,
        "checked": [
            f"read {tensor_name} from {shard_name} (real Math-Preserve shard)",
            "tensor payload sha256 matches shard header",
            "RMSNorm(fixed_prompt_vector, norm.weight) @ fixed_projection twice",
            "bit-identical float32 logits across two runs and after re-read",
        ],
        "skipped": [
            "full multi-layer GLM-5.2 forward (92 GB weights; free RAM ~0.16 GB observed)",
            "full vocab lm_head logits (lm_head on model-00001, ~3.87 GB shard)",
            "Metal / GPU paths (forbidden for this lane)",
        ],
    }


def functional_codec_authority() -> dict[str, Any]:
    """Documented CPU authority: gravity_functional_codec.execute, twice."""
    # This is the authoritative recomposed module.  The prior import pointed
    # into the retired tools/condense tree, so T0's CPU authority could be
    # described in a receipt while no longer being importable.
    from lab.operators import gravity_functional_codec as codec

    # Minimal direct-map payload (hidden=0): left is [width, out_width] float16.
    width, out_width = 8, 4
    rng = np.random.RandomState(17)
    left = rng.randn(width, out_width).astype(np.float16)
    payload = {
        "codec": codec.CODEC_ID if hasattr(codec, "CODEC_ID") else "glm52.functional.moe.v1",
        "width": width,
        "hidden": 0,
        "out_width": out_width,
        "rank": 0,
        "activation": "none",
        "seed": 17,
        "scale": 1.0,
        "layer": 0,
        "left": left,
        "right": None,
    }
    # Feature-map path using the frozen CPU authority generator.
    left2 = rng.randn(4, out_width).astype(np.float16)
    payload_h = {
        "width": 8,
        "hidden": 4,
        "out_width": out_width,
        "rank": 0,
        "activation": codec.ACTIVATION_SILU,
        "seed": 17,
        "scale": 1.0,
        "layer": 0,
        "left": left2,
        "right": None,
    }
    x = rng.randn(2, 8).astype(np.float32)
    y1 = codec.execute(payload_h, x)
    y2 = codec.execute(payload_h, x)
    path = "gravity_functional_codec.execute (feature-map + silu)"
    # Also exercise direct-map once to keep both code paths warm in the receipt.
    _ = codec.execute(payload, rng.randn(1, width).astype(np.float32))

    identical = bool(np.array_equal(y1, y2))
    return {
        "status": "PASS" if identical else "FAIL",
        "path": path,
        "module": "lab/operators/gravity_functional_codec.py",
        "bit_identical_two_runs": identical,
        "output_sha256_run1": hashlib.sha256(np.ascontiguousarray(y1).tobytes()).hexdigest(),
        "output_sha256_run2": hashlib.sha256(np.ascontiguousarray(y2).tobytes()).hexdigest(),
        "output_shape": list(y1.shape),
        "checked": [
            "CPU authority execute() twice on identical inputs",
            "bit-identical outputs required",
        ],
        "skipped": ["full teacher MoE organ against live capsules (not required for T0 smoke)"],
    }


def verify_runtime() -> dict[str, Any]:
    artifact = artifact_single_layer_authority()
    functional = functional_codec_authority()
    ok = artifact.get("status") == "PASS" and functional.get("status") == "PASS"
    return {
        "schema": SCHEMA,
        "status": "PASS" if ok else "FAIL",
        "artifact_single_layer": artifact,
        "functional_codec_cpu_authority": functional,
        "what_was_checked": (
            list(artifact.get("checked") or []) + list(functional.get("checked") or [])
        ),
        "what_was_skipped": (
            list(artifact.get("skipped") or []) + list(functional.get("skipped") or [])
        ),
        "memory_note": (
            "Full-model CPU forward of Math-Preserve requires holding ~92 GB weights "
            "plus activations/KV on a 96 GB machine currently near capacity; not attempted."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    result = verify_runtime()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
