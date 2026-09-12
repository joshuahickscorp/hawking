#!/usr/bin/env python3
"""Held-out product-quantization screen for sharded BF16 lookup tables.

This is a reusable Odyssey discriminator for a large lookup-table organ that
cannot reasonably be loaded as one dense matrix during early Gravity work.  It
range-reads deterministic train and held-out rows from every matching tensor,
learns only shared BF16 product codebooks from the train rows, and bills the
projected full-table codes, codebooks, metadata, and decoder support.

The result is deliberately *not* a serialized representation: it cannot claim
complete-model EBPW, lookup semantics, direct execution, TPS, or capability.
It answers the cheaper prior question: does a table have a held-out
rate--distortion slope worth escalating into a real representation?
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SPEC = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
)
DEFAULT_TENSOR_PATTERN = "ngram_embedding.shard_"
METADATA_BYTES = 128
DECODER_SUPPORT_BYTES = 65536
SCHEMA = "hawking.odyssey.sharded_lookup_pq_screen.v1"


@dataclass(frozen=True)
class ProductQuantizer:
    """A fitted shared-PQ family reusable by function-aware controls."""

    subdimension: int
    cardinality: int
    codebooks: tuple[np.ndarray, ...]

    @property
    def code_bits_per_subspace(self) -> int:
        return _require_power_of_two(self.cardinality, "PQ card")

    @property
    def subspaces_per_row(self) -> int:
        return len(self.codebooks)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 values to the BF16 codebook values billed on disk."""
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return (rounded.astype(np.uint32) << 16).view(np.float32)


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"short safetensors length prefix: {path}")
        header_bytes = struct.unpack("<Q", raw_len)[0]
        raw = handle.read(header_bytes)
    try:
        header = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors header: {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return header, 8 + int(header_bytes)


def _sample_positions(rows: int, per_partition: int) -> tuple[list[int], list[int]]:
    if rows < 2:
        raise ValueError("lookup tensor needs at least two rows for train/held-out sampling")
    requested = min(rows, max(2, int(per_partition) * 2))
    positions = np.linspace(0, rows - 1, num=requested, dtype=np.int64).tolist()
    positions = list(dict.fromkeys(int(value) for value in positions))
    train = positions[::2]
    heldout = positions[1::2]
    if not train or not heldout:
        raise ValueError("lookup sampling could not form train and held-out partitions")
    return train, heldout


def load_sharded_rows(
    spec: Path,
    *,
    tensor_pattern: str,
    rows_per_partition: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Range-read deterministic rows from matching 2D BF16 lookup tensors."""
    root = spec.expanduser().resolve()
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index has no weight_map")
    pattern = tensor_pattern.lower()
    names = sorted(name for name in weight_map if pattern in name.lower())
    if not names:
        raise ValueError(f"no tensors match {tensor_pattern!r}")

    headers: dict[Path, tuple[dict[str, Any], int]] = {}
    train_rows: list[np.ndarray] = []
    heldout_rows: list[np.ndarray] = []
    sampled = []
    total_rows = 0
    total_entries = 0
    width: int | None = None
    range_read_bytes = 0
    raw_digest = hashlib.sha256()
    for name in names:
        shard = root / str(weight_map[name])
        if shard not in headers:
            headers[shard] = _read_header(shard)
        header, data_start = headers[shard]
        entry = header.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"missing indexed tensor in shard header: {name}")
        if entry.get("dtype") != "BF16":
            raise ValueError(f"{name} must be BF16, got {entry.get('dtype')!r}")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not (
            isinstance(shape, list)
            and len(shape) == 2
            and all(isinstance(item, int) and item > 0 for item in shape)
            and isinstance(offsets, list)
            and len(offsets) == 2
        ):
            raise ValueError(f"{name} does not have a supported 2D lookup-table layout")
        rows, columns = int(shape[0]), int(shape[1])
        if width is None:
            width = columns
        elif width != columns:
            raise ValueError(f"lookup tensors have inconsistent widths: {width} vs {columns}")
        source_bytes = int(offsets[1]) - int(offsets[0])
        if source_bytes != rows * columns * 2:
            raise ValueError(f"{name} BF16 payload geometry is inconsistent")
        total_rows += rows
        total_entries += rows * columns
        train_ids, heldout_ids = _sample_positions(rows, rows_per_partition)
        row_bytes = columns * 2
        source_offset = data_start + int(offsets[0])
        with shard.open("rb") as handle:
            for partition, row_ids, destination in (
                ("train", train_ids, train_rows),
                ("heldout", heldout_ids, heldout_rows),
            ):
                for row_id in row_ids:
                    handle.seek(source_offset + row_id * row_bytes)
                    raw = handle.read(row_bytes)
                    if len(raw) != row_bytes:
                        raise ValueError(f"short range read for {name} row {row_id}")
                    range_read_bytes += len(raw)
                    raw_digest.update(name.encode("utf-8"))
                    raw_digest.update(partition.encode("ascii"))
                    raw_digest.update(int(row_id).to_bytes(8, "little", signed=False))
                    raw_digest.update(raw)
                    destination.append(
                        (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16)
                        .view("<f4")
                        .copy()
                    )
        sampled.append(
            {
                "tensor": name,
                "shard": str(shard),
                "shape": [rows, columns],
                "source_bytes": source_bytes,
                "train_row_ids": train_ids,
                "heldout_row_ids": heldout_ids,
            }
        )
    if width is None or not train_rows or not heldout_rows:
        raise ValueError("no usable sampled lookup rows")
    return (
        np.stack(train_rows).astype(np.float32, copy=False),
        np.stack(heldout_rows).astype(np.float32, copy=False),
        {
            "specimen": str(root),
            "index_path": str(index_path),
            "index_sha256": _sha256_bytes(index_path.read_bytes()),
            "tensor_pattern": tensor_pattern,
            "tensor_count": len(names),
            "total_rows": total_rows,
            "total_entries": total_entries,
            "width": width,
            "train_rows": len(train_rows),
            "heldout_rows": len(heldout_rows),
            "sample_sha256": raw_digest.hexdigest(),
            "range_read_bytes": range_read_bytes,
            "source_passes": 1,
            "sampled_tensors": sampled,
            "source_reads": "deterministic BF16 range reads only; no full lookup tensor is loaded",
        },
    )


def _require_power_of_two(value: int, label: str) -> int:
    value = int(value)
    if value <= 1 or value & (value - 1):
        raise ValueError(f"{label} must be a power of two greater than one")
    return value.bit_length() - 1


def _kmeans(values: np.ndarray, card: int, iterations: int, seed: int) -> np.ndarray:
    """Small deterministic Lloyd fit; codebooks are BF16-rounded before use."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < card:
        raise ValueError("PQ training population is smaller than the codebook card")
    rng = np.random.default_rng(seed)
    centers = values[rng.choice(values.shape[0], size=card, replace=False)].copy()
    sample_norm = np.sum(values * values, axis=1, keepdims=True)
    for _ in range(max(1, int(iterations))):
        distance = sample_norm + np.sum(centers * centers, axis=1)[None, :] - 2.0 * values @ centers.T
        labels = np.argmin(distance, axis=1)
        counts = np.bincount(labels, minlength=card)
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, values)
        replacement_order = np.argsort(np.min(distance, axis=1))[::-1]
        next_centers = centers.copy()
        replacement = 0
        for code in range(card):
            if counts[code]:
                next_centers[code] = sums[code] / counts[code]
            else:
                next_centers[code] = values[replacement_order[replacement]]
                replacement += 1
        next_centers = _bf16(next_centers)
        if np.array_equal(next_centers, centers):
            centers = next_centers
            break
        centers = next_centers
    return centers


def _encode_decode(values: np.ndarray, codebooks: list[np.ndarray], subdim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    restored = np.empty_like(values)
    for chunk, codebook in enumerate(codebooks):
        start = chunk * subdim
        stop = start + subdim
        segment = values[:, start:stop]
        distance = (
            np.sum(segment * segment, axis=1, keepdims=True)
            + np.sum(codebook * codebook, axis=1)[None, :]
            - 2.0 * segment @ codebook.T
        )
        restored[:, start:stop] = codebook[np.argmin(distance, axis=1)]
    return restored


def fit_product_quantizer(
    train: np.ndarray,
    *,
    subdimension: int,
    cardinality: int,
    iterations: int,
    seed: int,
) -> ProductQuantizer:
    """Fit one deterministic BF16 shared-PQ family without serializing a table.

    Function-aware architecture adapters reuse this exact construction instead
    of rebuilding a subtly different PQ implementation around every new
    specimen.  The returned object contains only shared codebooks; callers
    remain responsible for billing projected per-row codes and for making no
    materialized-representation claim unless they actually construct one.
    """
    train = np.asarray(train, dtype=np.float32)
    if train.ndim != 2 or train.shape[0] < 2:
        raise ValueError("PQ training population must be a non-empty 2D matrix")
    width = int(train.shape[1])
    subdimension = int(subdimension)
    cardinality = int(cardinality)
    if subdimension <= 0 or width % subdimension:
        raise ValueError(f"sub-dimension {subdimension} does not divide lookup width {width}")
    _require_power_of_two(cardinality, "PQ card")
    if cardinality > train.shape[0]:
        raise ValueError(
            f"PQ card {cardinality} exceeds {train.shape[0]} train rows; increase sampling"
        )
    codebooks = tuple(
        _kmeans(
            train[:, index * subdimension:(index + 1) * subdimension],
            cardinality,
            iterations,
            seed + 104729 * index + 1009 * cardinality + subdimension,
        )
        for index in range(width // subdimension)
    )
    return ProductQuantizer(
        subdimension=subdimension,
        cardinality=cardinality,
        codebooks=codebooks,
    )


def encode_decode_product_quantizer(values: np.ndarray, quantizer: ProductQuantizer) -> np.ndarray:
    """Reconstruct rows with a fitted shared-PQ family."""
    return _encode_decode(values, list(quantizer.codebooks), quantizer.subdimension)


def row_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int | bool | None]:
    errors = reference - candidate
    reference_norm = np.linalg.norm(reference, axis=1)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    error_norm = np.linalg.norm(errors, axis=1)
    nonzero_reference = reference_norm > 1e-12
    cosine_defined = nonzero_reference & (candidate_norm > 1e-12)
    row_relative = error_norm[nonzero_reference] / reference_norm[nonzero_reference]
    cosine = (
        np.sum(reference[cosine_defined] * candidate[cosine_defined], axis=1)
        / (reference_norm[cosine_defined] * candidate_norm[cosine_defined])
    )
    return {
        "rows": int(reference.shape[0]),
        "nonzero_reference_rows": int(np.count_nonzero(nonzero_reference)),
        "zero_reference_rows": int(reference.shape[0] - np.count_nonzero(nonzero_reference)),
        "cosine_defined_rows": int(np.count_nonzero(cosine_defined)),
        "mean_row_relative_l2": float(np.mean(row_relative)) if row_relative.size else None,
        "p95_row_relative_l2": float(np.percentile(row_relative, 95)) if row_relative.size else None,
        "max_row_relative_l2": float(np.max(row_relative)) if row_relative.size else None,
        "mean_row_cosine": float(np.mean(cosine)) if cosine.size else None,
        "min_row_cosine": float(np.min(cosine)) if cosine.size else None,
        "global_relative_l2": float(np.linalg.norm(errors) / max(float(np.linalg.norm(reference)), 1e-30)),
        "rmse": float(np.sqrt(np.mean(np.square(errors, dtype=np.float64)))),
        "finite": bool(np.isfinite(candidate).all()),
    }


# Keep the historical private spelling for older receipts/tests while giving
# function-aware architecture adapters one obvious reusable metric owner.
_metrics = row_metrics


def screen(
    train: np.ndarray,
    heldout: np.ndarray,
    *,
    total_rows: int,
    subdims: list[int],
    cards: list[int],
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return held-out PQ curves plus complete *table-scoped projected* billing."""
    train = np.asarray(train, dtype=np.float32)
    heldout = np.asarray(heldout, dtype=np.float32)
    if train.ndim != 2 or heldout.ndim != 2 or train.shape[1] != heldout.shape[1]:
        raise ValueError("train and held-out rows must be compatible matrices")
    if train.shape[0] < 2 or heldout.shape[0] < 1 or total_rows < train.shape[0] + heldout.shape[0]:
        raise ValueError("invalid sampled or projected lookup-table geometry")
    width = int(train.shape[1])
    total_entries = int(total_rows) * width
    candidates: list[dict[str, Any]] = []
    for subdim in sorted(set(int(value) for value in subdims)):
        if subdim <= 0 or width % subdim:
            raise ValueError(f"sub-dimension {subdim} does not divide lookup width {width}")
        chunks = width // subdim
        for card in sorted(set(int(value) for value in cards)):
            quantizer = fit_product_quantizer(
                train,
                subdimension=subdim,
                cardinality=card,
                iterations=iterations,
                seed=seed,
            )
            bits = quantizer.code_bits_per_subspace
            train_reconstructed = encode_decode_product_quantizer(train, quantizer)
            heldout_reconstructed = encode_decode_product_quantizer(heldout, quantizer)
            code_payload_bits = int(total_rows) * chunks * bits
            code_payload_bytes = (code_payload_bits + 7) // 8
            codebook_bytes = chunks * card * subdim * 2
            complete_projected_bytes = (
                code_payload_bytes + codebook_bytes + METADATA_BYTES + DECODER_SUPPORT_BYTES
            )
            row_code_bytes = (chunks * bits + 7) // 8
            candidates.append(
                {
                    "family": "shared_product_quantization",
                    "subdimension": subdim,
                    "subspaces_per_row": chunks,
                    "cardinality": card,
                    "code_bits_per_subspace": bits,
                    "payload_bits_per_weight": bits / subdim,
                    "codebook_dtype": "BF16",
                    "train_metrics": _metrics(train, train_reconstructed),
                    "heldout_metrics": _metrics(heldout, heldout_reconstructed),
                    "projected_full_table": {
                        "logical_rows": int(total_rows),
                        "logical_weights": total_entries,
                        "code_payload_bytes": code_payload_bytes,
                        "codebook_bytes": codebook_bytes,
                        "metadata_bytes": METADATA_BYTES,
                        "decoder_support_bytes": DECODER_SUPPORT_BYTES,
                        "complete_projected_bytes": complete_projected_bytes,
                        "projected_complete_table_ebpw": complete_projected_bytes * 8.0 / total_entries,
                        "row_code_bytes": row_code_bytes,
                        "active_bytes_per_lookup": "NOT_MEASURED__depends_on_codebook_residency_and_kernel",
                    },
                    "representation_materialized": False,
                    "direct_execution": False,
                    "lookup_semantics_qualified": False,
                }
            )
    return candidates


def _pareto(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    best_error = math.inf
    for candidate in sorted(
        candidates,
        key=lambda row: (
            float(row["projected_full_table"]["projected_complete_table_ebpw"]),
            float(row["heldout_metrics"]["global_relative_l2"]),
        ),
    ):
        error = float(candidate["heldout_metrics"]["global_relative_l2"])
        if error < best_error:
            result.append(candidate)
            best_error = error
    return result


def _target_summary(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Materialize the next-run decision instead of requiring receipt spelunking."""
    summary = []
    for target in (0.1, 0.25, 0.5, 0.75, 1.0):
        admissible = [
            candidate
            for candidate in candidates
            if candidate["projected_full_table"]["projected_complete_table_ebpw"] <= target
        ]
        best = min(
            admissible,
            key=lambda candidate: candidate["heldout_metrics"]["global_relative_l2"],
            default=None,
        )
        summary.append(
            {
                "target_projected_table_ebpw": target,
                "best_candidate": (
                    {
                        "subdimension": best["subdimension"],
                        "cardinality": best["cardinality"],
                        "projected_complete_table_ebpw": best["projected_full_table"][
                            "projected_complete_table_ebpw"
                        ],
                        "heldout_global_relative_l2": best["heldout_metrics"][
                            "global_relative_l2"
                        ],
                        "heldout_mean_row_cosine": best["heldout_metrics"]["mean_row_cosine"],
                    }
                    if best is not None
                    else None
                ),
                "status": "SAMPLED_STATIC_ONLY__NOT_A_REPRESENTATION_CLAIM",
            }
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--tensor-pattern", default=DEFAULT_TENSOR_PATTERN)
    parser.add_argument(
        "--rows-per-partition",
        type=int,
        default=8,
        help="deterministic train rows and held-out rows per matching tensor",
    )
    parser.add_argument("--subdimensions", default="8,16,32,40")
    parser.add_argument("--cards", default="8,16,32")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.rows_per_partition < 1 or args.iterations < 1:
        raise SystemExit("rows-per-partition and iterations must be positive")
    subdims = [int(value) for value in args.subdimensions.split(",") if value.strip()]
    cards = [int(value) for value in args.cards.split(",") if value.strip()]
    if not subdims or not cards:
        raise SystemExit("subdimensions and cards must be non-empty")
    started = datetime.now(timezone.utc)
    started_ns = time.perf_counter_ns()
    train, heldout, source = load_sharded_rows(
        args.spec,
        tensor_pattern=args.tensor_pattern,
        rows_per_partition=args.rows_per_partition,
    )
    candidates = screen(
        train,
        heldout,
        total_rows=int(source["total_rows"]),
        subdims=subdims,
        cards=cards,
        iterations=args.iterations,
        seed=args.seed,
    )
    document = {
        "schema": SCHEMA,
        "generated_at": started.isoformat(),
        "status": "SAMPLED_STATIC_RATE_DISTORTION_ONLY",
        "source": source,
        "representation": {
            "base": "global shared BF16 product codebooks with packed per-row subspace codes",
            "training": "deterministic Lloyd fit on the named train rows only",
            "evaluation": "disjoint deterministic held-out rows from every matching tensor",
            "billed": {
                "code_payload": "all projected table rows",
                "codebooks": "one shared BF16 codebook per subspace",
                "metadata_bytes": METADATA_BYTES,
                "decoder_support_bytes": DECODER_SUPPORT_BYTES,
            },
        },
        "candidates": candidates,
        "pareto": _pareto(candidates),
        "target_summary": _target_summary(candidates),
        "claim_boundary": (
            "This is a sampled static discriminator. Projected full-table bytes are an "
            "arithmetic rate screen, not a materialized lookup artifact or complete NR. "
            "It does not establish token-frequency coverage, collision/lookup semantics, "
            "direct execution, whole-model EBPW, capability, TPS, or promotion."
        ),
        "next_gate": (
            "Only a held-out Pareto survivor earns a source-bound n-gram lookup semantics "
            "control, serialized table construction, direct native lookup, and later "
            "whole-NR accounting."
        ),
        "elapsed_ns": time.perf_counter_ns() - started_ns,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": document["status"],
                "tensors": source["tensor_count"],
                "train_rows": source["train_rows"],
                "heldout_rows": source["heldout_rows"],
                "candidates": len(candidates),
                "pareto": len(document["pareto"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
