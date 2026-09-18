#!/usr/bin/env python3
"""Bounded CPU reference for directly consuming one KIMI PQ organ tensor.

This deliberately works on one tensor, not the model.  It uses the isolated
KIMI runtime's PyTorch/safetensors installation, reads one tensor from
KIMI_BASE, fits a small row-wise product quantizer, and executes the matvec by
codebook groups.  The execution path expands one group at a time and never
constructs a dense replacement matrix.  A dense matvec is used only as a
verification oracle after the direct result is produced.

The result is a direct-execution reference and an output-error measurement, not
an end-to-end KIMI capability result or a production kernel qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.future.complete_ebpw import (  # noqa: E402
    STREAM_BROADCAST_AUX,
    STREAM_WEIGHT_CODES,
    candidate_from_parts,
    cost,
)

DEFAULT_SPEC = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "moonshotai--Kimi-VL-A3B-Instruct@398eede0903c"
)
DEFAULT_TENSOR = "language_model.model.layers.10.mlp.experts.23.up_proj.weight"
SCHEMA = "hawking.future.kimi_noetic_pq_reference.v1"
DECODER_SUPPORT_BYTES = 65536
METADATA_BYTES = 4096


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tensor(spec: Path, tensor_name: str):
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment
        raise RuntimeError(
            "safetensors is required; run with /tmp/hawking-kimi-runtime/bin/python"
        ) from exc
    index = json.loads((spec / "model.safetensors.index.json").read_text())
    try:
        shard_name = str(index["weight_map"][tensor_name])
    except KeyError as exc:
        raise RuntimeError(f"tensor is absent from KIMI index: {tensor_name}") from exc
    shard = spec / shard_name
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(tensor_name)
    return tensor, shard


def _fit_pq(weight, d: int, steps: int, entries: int):
    import torch

    if weight.ndim != 2:
        raise ValueError(f"reference requires a matrix, got shape={tuple(weight.shape)}")
    rows, width = (int(x) for x in weight.shape)
    groups = (width + d - 1) // d
    padded_width = groups * d
    data = weight.to(dtype=torch.float32).contiguous()
    if padded_width != width:
        data = torch.nn.functional.pad(data, (0, padded_width - width))
    blocks = data.view(rows, groups, d)
    codebooks = []
    codes = []
    with torch.no_grad():
        for group in range(groups):
            vectors = blocks[:, group, :]
            # Deterministic spread initialization; this is a cheap reference
            # fit, not an optimized PQ training run.
            seed = torch.linspace(0, rows - 1, min(entries, rows), dtype=torch.long)
            if seed.numel() < entries:
                seed = seed.repeat((entries + seed.numel() - 1) // seed.numel())[:entries]
            codebook = vectors.index_select(0, seed).clone()
            code = None
            for _ in range(max(1, int(steps))):
                distance = ((vectors[:, None, :] - codebook[None, :, :]) ** 2).sum(dim=2)
                code = distance.argmin(dim=1)
                sums = torch.zeros_like(codebook)
                counts = torch.zeros(codebook.shape[0], dtype=torch.int64)
                sums.index_add_(0, code, vectors)
                counts.index_add_(0, code, torch.ones_like(code, dtype=torch.int64))
                codebook = sums / counts.clamp_min(1).to(dtype=sums.dtype).unsqueeze(1)
            assert code is not None
            # Store/consume codebooks at the declared BF16 representation.
            codebooks.append(codebook.to(dtype=torch.bfloat16).to(dtype=torch.float32))
            codes.append(code)
    return codebooks, codes, rows, width, groups


def _direct_mv(codebooks, codes, x, *, width: int, d: int):
    import torch

    rows = int(codes[0].shape[0])
    out = torch.zeros(rows, dtype=torch.float32)
    for group, (codebook, code) in enumerate(zip(codebooks, codes)):
        start = group * d
        stop = min(start + d, width)
        values = codebook.index_select(0, code)
        out += values[:, : stop - start].matmul(x[start:stop])
    return out


def _median_ms(fn, reps: int) -> tuple[float, list[float], Any]:
    values = []
    result = None
    for _ in range(max(1, reps)):
        started = time.perf_counter()
        result = fn()
        values.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(values), values, result


def run(spec: Path, tensor_name: str, d: int, steps: int, reps: int, entries: int = 256) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(20260910)
    weight, shard = _load_tensor(spec, tensor_name)
    codebooks, codes, rows, width, groups = _fit_pq(
        weight, d=d, steps=steps, entries=entries
    )
    x = torch.randn(width, dtype=torch.float32)

    # Warm the two paths before recording comparable CPU timings.
    direct = lambda: _direct_mv(codebooks, codes, x, width=width, d=d)
    dense = lambda: weight.to(dtype=torch.float32).matmul(x)
    direct_result = direct()
    oracle = dense()
    direct_ms, direct_samples, direct_result = _median_ms(direct, reps)
    dense_ms, dense_samples, oracle = _median_ms(dense, reps)

    error = direct_result - oracle
    source_params = int(weight.numel())
    index_bytes = rows * groups
    codebook_bytes = groups * entries * d * 2
    candidate = candidate_from_parts(
        family_id=f"KIMI_PQ_D{d}_{tensor_name}",
        parent_params=source_params,
        parts={
            "regions": [{"name": "pq_indices", "bytes": index_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "generators": [],
            "metadata": [{"name": "scoped_metadata", "bytes": METADATA_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "tables": [{"name": "bf16_codebooks", "bytes": codebook_bytes, "stream_class": STREAM_WEIGHT_CODES}],
            "residuals": [],
            "runtime_auxiliaries": [{"name": "direct_decoder_support", "bytes": DECODER_SUPPORT_BYTES, "stream_class": STREAM_BROADCAST_AUX}],
            "representation": [],
            "model_specific_code": [],
        },
        reconstructs_dense_parent=False,
        consumes_representation_directly=True,
    )
    accounting = cost(candidate)
    transient_bytes = rows * d * 4
    return {
        "schema": SCHEMA,
        "status": "DIRECT_REFERENCE_PASS",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "specimen": str(spec),
        "tensor": {
            "name": tensor_name,
            "shape": [rows, width],
            "dtype": str(weight.dtype),
            "source_shard": str(shard),
            "source_shard_sha256": _sha256(shard),
            "source_payload_bytes": int(weight.numel() * 2),
        },
        "representation": {
            "family": f"pq_d{d}_rowwise_reference",
            "groups": groups,
            "codebook_entries": entries,
            "codebook_dtype": "bf16",
            "fit_steps": max(1, int(steps)),
            "index_bytes": index_bytes,
            "codebook_bytes": codebook_bytes,
            "complete_accounting": accounting,
        },
        "direct_execution": {
            "verified": True,
            "dense_parent_materialized_by_direct_path": False,
            "transient_group_expansion_bytes": transient_bytes,
            "persistent_representation_bytes": accounting["executable_bytes"],
            "verification_oracle": "dense matvec only; not part of direct path",
            "production_kernel": "NOT_VALIDATED",
        },
        "error": {
            "max_abs": float(error.abs().max().item()),
            "rmse": float(torch.sqrt(torch.mean(error * error)).item()),
            "relative_l2": float(torch.linalg.vector_norm(error).item() / max(torch.linalg.vector_norm(oracle).item(), 1e-12)),
            "cosine": float(torch.nn.functional.cosine_similarity(direct_result[None, :], oracle[None, :]).item()),
        },
        "timing": {
            "runtime": "isolated KIMI Python runtime",
            "torch_threads": 1,
            "direct_median_ms": direct_ms,
            "direct_samples_ms": direct_samples,
            "dense_oracle_median_ms": dense_ms,
            "dense_oracle_samples_ms": dense_samples,
            "fit_excluded_from_matvec_timing": True,
        },
        "claim_boundary": (
            "One-tensor CPU reference only. This validates a no-full-dense direct-consumer "
            "mechanism and measures approximation error for a cheap fit; it is not a fitted "
            "full-organ representation, model capability receipt, end-to-end TPS result, "
            "production kernel qualification, or KIMI promotion."
        ),
        "next_discriminator": "Repeat across the three projection tensors with output-aware calibration and compare residual/protected-island alternatives before body-wide compilation.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--tensor", default=DEFAULT_TENSOR)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--entries", type=int, default=256)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.d <= 0 or args.d > 256 or args.reps <= 0 or args.entries <= 0:
        raise SystemExit("d must be in 1..256; entries and reps must be positive")
    result = run(args.spec, args.tensor, args.d, args.steps, args.reps, args.entries)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["python"] = platform.python_version()
    result["script"] = str(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "tensor": result["tensor"]["name"],
        "complete_ebpw": result["representation"]["complete_accounting"]["complete_ebpw"],
        "relative_l2": result["error"]["relative_l2"],
        "direct_median_ms": result["timing"]["direct_median_ms"],
        "dense_oracle_median_ms": result["timing"]["dense_oracle_median_ms"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
