#!/usr/bin/env python3
"""Static low-rank curve for one expert in a stacked MoE safetensors tensor.

Qwen4-Exp stores routed weights as [expert, out, in].  This reads one selected
expert directly from the payload, avoiding an accidental materialization of the
entire 512-expert bank.  It is a byte/error discriminator only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.future.representational_anatomy import _read_header_ex

DEFAULT_SPEC = Path("/Volumes/corpdrive/hawking-modellake/specimens/"
                    "Qwen--Qwen3.8-Flash-Next@34567a4712bc")
DEFAULT_TENSOR = "model.language_model.layers.1.mlp.experts.gate_up_proj"


def _decode(raw: bytes, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").copy()
    raise ValueError(f"unsupported dtype {dtype}")


def load_expert(spec: Path, tensor: str, expert: int) -> tuple[np.ndarray, dict]:
    widths = {"BF16": 2, "F16": 2, "F32": 4}
    for path in sorted(spec.glob("*.safetensors")):
        header, header_len = _read_header_ex(str(path))
        if tensor not in header:
            continue
        e = header[tensor]
        dtype, shape = e["dtype"], [int(x) for x in e["shape"]]
        if dtype not in widths or len(shape) != 3:
            raise ValueError(f"expected stacked 3D float tensor, got {dtype} {shape}")
        n_experts, rows, cols = shape
        if not 0 <= expert < n_experts:
            raise ValueError(f"expert {expert} outside [0,{n_experts})")
        nbytes = rows * cols * widths[dtype]
        offset = 8 + header_len + int(e["data_offsets"][0]) + expert * nbytes
        with path.open("rb") as fh:
            fh.seek(offset)
            raw = fh.read(nbytes)
        if len(raw) != nbytes:
            raise ValueError("short expert payload read")
        return _decode(raw, dtype).reshape(rows, cols), {
            "source_file": str(path), "dtype": dtype, "stacked_shape": shape,
            "expert": expert, "payload_bytes_read": nbytes,
        }
    raise KeyError(f"tensor not found: {tensor}")


def curve(weight: np.ndarray, ranks: list[int]) -> list[dict]:
    if weight.ndim != 2:
        raise ValueError(f"expected matrix, got {weight.shape}")
    rows, cols = weight.shape
    _, singular, _ = np.linalg.svd(weight, full_matrices=False)
    energy = singular * singular
    total = float(energy.sum())
    out = []
    for rank in sorted(set(ranks)):
        if not 1 <= rank <= min(rows, cols):
            raise ValueError(f"rank {rank} outside matrix range")
        residual = max(0.0, 1.0 - float(energy[:rank].sum()) / max(total, 1e-30))
        factor_bytes = rank * (rows + cols) * 2  # BF16 U and V factors
        metadata_bytes = 128
        out.append({
            "rank": rank,
            "relative_frobenius_error": residual ** 0.5,
            "factor_bytes": factor_bytes,
            "metadata_bytes": metadata_bytes,
            "complete_scoped_ebpw": (factor_bytes + metadata_bytes) * 8.0 / (rows * cols),
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    p.add_argument("--tensor", default=DEFAULT_TENSOR)
    p.add_argument("--expert", type=int, default=0)
    p.add_argument("--ranks", default="64,128,256,512,707")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    ranks = [int(x) for x in a.ranks.split(",")]
    weight, source = load_expert(a.spec, a.tensor, a.expert)
    result = {
        "schema": "hawking.odyssey.stacked_expert_lowrank_screen.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "STATIC_RATE_DISTORTION_ONLY",
        "specimen": str(a.spec), "tensor": a.tensor, "source": source,
        "matrix_shape": list(weight.shape), "rows": curve(weight, ranks),
        "claim_boundary": "One selected source expert and BF16 factor accounting only. No residual, codebook, routing, activation/capability, direct execution, full-body EBPW, runtime, or TPS claim.",
        "next_gate": "Use only a held-out output/capability-aware routed-expert experiment to decide whether any static rank is worth protecting.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
