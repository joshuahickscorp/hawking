#!/usr/bin/env python3
"""Function-aware PQ-family discriminator for a source-bound Qwen4-Exp PLE control.

The generic sharded-table PQ screen measures row reconstruction.  This adapter
uses its exact deterministic codebook construction and then asks the useful
next question: how much does a completely billed PQ-family table perturb an
actual PLE injection at a sealed source input?  It remains a cheap screen:
codes are projected for the whole table but not materialized, and a single
source control is neither full lookup coverage nor an NR claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Direct script execution and package imports are both supported.
    from tools.odyssey.ple_access_trace import load_contract, load_layout, sha256_bytes, trace_token_segments
    from tools.odyssey.ple_source_output_control import (
        SCHEMA as SOURCE_CONTROL_SCHEMA,
        evaluate_ple_sequence,
        load_source_lookup_embeddings,
        load_source_support,
    )
    from tools.odyssey.sharded_lookup_pq_screen import (
        DECODER_SUPPORT_BYTES,
        METADATA_BYTES,
        ProductQuantizer,
        ResidualProductQuantizer,
        encode_decode_product_quantizer,
        encode_decode_residual_product_quantizer,
        fit_product_quantizer,
        fit_residual_product_quantizer,
        load_sharded_rows,
        row_metrics,
    )
except ModuleNotFoundError:  # pragma: no cover - direct receipt invocation.
    from ple_access_trace import load_contract, load_layout, sha256_bytes, trace_token_segments
    from ple_source_output_control import (
        SCHEMA as SOURCE_CONTROL_SCHEMA,
        evaluate_ple_sequence,
        load_source_lookup_embeddings,
        load_source_support,
    )
    from sharded_lookup_pq_screen import (
        DECODER_SUPPORT_BYTES,
        METADATA_BYTES,
        ProductQuantizer,
        ResidualProductQuantizer,
        encode_decode_product_quantizer,
        encode_decode_residual_product_quantizer,
        fit_product_quantizer,
        fit_residual_product_quantizer,
        load_sharded_rows,
        row_metrics,
    )


SCHEMA = "hawking.odyssey.ple_lookup_pq_output_screen.v1"
REFERENCE_ORACLE_SCHEMA = "hawking.odyssey.ple_reference_formula_oracle.v1"
DEFAULT_CONTROL = Path("receipts/headless/FLASH_PLE_LAYER1_BOS_SOURCE_OUTPUT_CONTROL.json")
DEFAULT_PARITY = Path("receipts/headless/FLASH_PLE_LAYER1_BOS_REFERENCE_FORMULA_PARITY.json")
DEFAULT_CANDIDATES = "32:4,32:8,16:8,8:4,8:8,8:16"
DEFAULT_REPAIR_COUNTS = "0,1,2,4,8,16,32,64,160"
CODEBOOK_SCOPE_GLOBAL = "global_shared"
CODEBOOK_SCOPE_SHARD_LOCAL = "shard_local"
SHARD_LOCAL_CODEBOOK_DIRECTORY_BYTES = 32
QUANTIZER_FAMILY_SINGLE_STAGE = "single_stage"
QUANTIZER_FAMILY_RESIDUAL_ADDITIVE = "residual_additive"
Quantizer = ProductQuantizer | ResidualProductQuantizer


def parse_candidates(raw: str) -> list[tuple[int, int]]:
    """Parse unique ``subdimension:cardinality`` candidate pairs."""
    result: list[tuple[int, int]] = []
    for field in raw.split(","):
        field = field.strip()
        if not field:
            continue
        pieces = field.split(":")
        if len(pieces) != 2:
            raise ValueError(f"candidate must use subdimension:cardinality, got {field!r}")
        candidate = (int(pieces[0]), int(pieces[1]))
        if candidate[0] <= 0 or candidate[1] <= 1:
            raise ValueError(f"candidate values must be positive, got {field!r}")
        if candidate not in result:
            result.append(candidate)
    if not result:
        raise ValueError("at least one PQ candidate is required")
    return result


def parse_repair_counts(raw: str) -> list[int]:
    """Parse unique fixed sparse-repair counts per lookup row."""
    result: list[int] = []
    for field in raw.split(","):
        field = field.strip()
        if not field:
            continue
        value = int(field)
        if value < 0:
            raise ValueError("repair counts must be non-negative")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("at least one repair count is required")
    return result


def validate_repair_counts_for_scope(codebook_scope: str, repair_counts: Sequence[int]) -> None:
    """Keep a hierarchy screen from silently becoming a rejected hybrid retry."""
    if codebook_scope not in (CODEBOOK_SCOPE_GLOBAL, CODEBOOK_SCOPE_SHARD_LOCAL):
        raise ValueError(f"unsupported PQ codebook scope: {codebook_scope!r}")
    if codebook_scope == CODEBOOK_SCOPE_SHARD_LOCAL and any(int(count) for count in repair_counts):
        raise ValueError(
            "shard-local PQ is a separate hierarchy screen and accepts only --repair-counts 0; "
            "do not combine it with the rejected fixed local-repair composition"
        )


def validate_quantizer_configuration(
    quantizer_family: str,
    *,
    residual_stages: int,
    repair_counts: Sequence[int],
) -> None:
    """Keep additive PQ distinct from copied-source sparse-repair screens."""
    if quantizer_family == QUANTIZER_FAMILY_SINGLE_STAGE:
        if residual_stages != 1:
            raise ValueError("single-stage PQ requires --residual-stages 1")
        return
    if quantizer_family != QUANTIZER_FAMILY_RESIDUAL_ADDITIVE:
        raise ValueError(f"unsupported PLE quantizer family: {quantizer_family!r}")
    if residual_stages < 2:
        raise ValueError("residual-additive PQ requires at least two stages")
    if any(int(count) for count in repair_counts):
        raise ValueError(
            "residual-additive PQ accepts only --repair-counts 0; copied source-BF16 repair "
            "would be a different hybrid representation and must be separately discriminated"
        )


def fit_declared_quantizer(
    train: np.ndarray,
    *,
    quantizer_family: str,
    residual_stages: int,
    subdimension: int,
    cardinality: int,
    iterations: int,
    seed: int,
) -> Quantizer:
    """Use the reusable generic PQ owner for the declared representation family."""
    if quantizer_family == QUANTIZER_FAMILY_SINGLE_STAGE:
        return fit_product_quantizer(
            train,
            subdimension=subdimension,
            cardinality=cardinality,
            iterations=iterations,
            seed=seed,
        )
    if quantizer_family == QUANTIZER_FAMILY_RESIDUAL_ADDITIVE:
        return fit_residual_product_quantizer(
            train,
            subdimension=subdimension,
            cardinality=cardinality,
            stages=residual_stages,
            iterations=iterations,
            seed=seed,
        )
    raise ValueError(f"unsupported PLE quantizer family: {quantizer_family!r}")


def encode_decode_declared_quantizer(values: np.ndarray, quantizer: Quantizer) -> np.ndarray:
    """Decode only the representation object selected by the candidate family."""
    if isinstance(quantizer, ProductQuantizer):
        return encode_decode_product_quantizer(values, quantizer)
    if isinstance(quantizer, ResidualProductQuantizer):
        return encode_decode_residual_product_quantizer(values, quantizer)
    raise TypeError(f"unsupported PLE quantizer object: {type(quantizer)!r}")


def _read_bound_f32(path: Path, record: dict[str, Any]) -> np.ndarray:
    raw = path.read_bytes()
    if (
        record.get("dtype") != "F32_LE"
        or record.get("bytes") != len(raw)
        or record.get("elements") != len(raw) // 4
        or record.get("sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError(f"F32 artifact disagrees with its receipt: {path}")
    return np.frombuffer(raw, dtype="<f4").copy()


def output_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | bool]:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError("PLE output vectors have incompatible shapes")
    delta = candidate - reference
    ref_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    return {
        "relative_l2": float(np.linalg.norm(delta) / max(ref_norm, 1e-30)),
        "rmse": float(math.sqrt(float(np.mean(delta * delta)))),
        "max_abs": float(np.max(np.abs(delta))),
        "cosine": float(np.dot(reference, candidate) / max(ref_norm * candidate_norm, 1e-30)),
        "finite": bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
    }


def projected_table_accounting(
    quantizer: Quantizer,
    *,
    total_rows: int,
    width: int,
    repair_count_per_row: int = 0,
) -> dict[str, int | float]:
    """Bill the same projected global PQ closure as the generic screen."""
    if width % quantizer.subdimension:
        raise ValueError("quantizer subdimension does not divide table width")
    subspaces = width // quantizer.subdimension
    if subspaces != quantizer.subspaces_per_row:
        raise ValueError("quantizer codebook count does not match table width")
    bits = quantizer.code_bits_per_subspace
    code_payload_bits = int(total_rows) * subspaces * bits
    code_payload_bytes = (code_payload_bits + 7) // 8
    codebook_bytes = sum(int(book.size) * 2 for book in quantizer.codebooks)
    repair_count_per_row = int(repair_count_per_row)
    if repair_count_per_row < 0 or repair_count_per_row > width:
        raise ValueError("repair count is outside the lookup-row width")
    repair_index_bits = max(1, math.ceil(math.log2(width)))
    repair_value_bits = 16  # Exact original source BF16 values at selected offsets.
    repair_payload_bits = int(total_rows) * repair_count_per_row * (
        repair_index_bits + repair_value_bits
    )
    repair_payload_bytes = (repair_payload_bits + 7) // 8
    repair_metadata_bytes = 16 if repair_count_per_row else 0
    complete_projected_bytes = (
        code_payload_bytes
        + codebook_bytes
        + METADATA_BYTES
        + DECODER_SUPPORT_BYTES
        + repair_payload_bytes
        + repair_metadata_bytes
    )
    return {
        "logical_rows": int(total_rows),
        "logical_weights": int(total_rows) * int(width),
        "subspaces_per_row": subspaces,
        "code_bits_per_subspace": bits,
        "quantizer_stages": int(getattr(quantizer, "stages", 1)),
        "code_payload_bytes": code_payload_bytes,
        "codebook_bytes": codebook_bytes,
        "metadata_bytes": METADATA_BYTES,
        "decoder_support_bytes": DECODER_SUPPORT_BYTES,
        "repair_count_per_row": repair_count_per_row,
        "repair_index_bits": repair_index_bits if repair_count_per_row else 0,
        "repair_value_bits": repair_value_bits if repair_count_per_row else 0,
        "repair_payload_bytes": repair_payload_bytes,
        "repair_metadata_bytes": repair_metadata_bytes,
        "complete_projected_bytes": complete_projected_bytes,
        "projected_complete_table_ebpw": complete_projected_bytes * 8.0 / (int(total_rows) * int(width)),
    }


def apply_exact_sparse_repair(
    source_rows: np.ndarray, base_rows: np.ndarray, *, repair_count_per_row: int
) -> np.ndarray:
    """Restore a fixed number of highest-error source BF16 values in every row.

    A hypothetical materialized table would store each selected offset plus its
    exact original BF16 value.  This function only evaluates that deterministic
    representation on the sealed active rows; global storage is billed as a
    projection and is never claimed to have been constructed here.
    """
    source_rows = np.asarray(source_rows, dtype=np.float32)
    repaired = np.asarray(base_rows, dtype=np.float32).copy()
    if source_rows.shape != repaired.shape or source_rows.ndim != 2:
        raise ValueError("source/base rows for sparse repair are incompatible")
    count = int(repair_count_per_row)
    if count < 0 or count > source_rows.shape[1]:
        raise ValueError("repair count is outside the lookup-row width")
    if count == 0:
        return repaired
    # Stable argsort makes ties deterministic and makes the later serialized
    # index convention reproducible.
    indices = np.argsort(-np.abs(source_rows - repaired), axis=1, kind="stable")[:, :count]
    for row, row_indices in enumerate(indices):
        repaired[row, row_indices] = source_rows[row, row_indices]
    return repaired


def sampled_row_shard_ordinals(
    source_sample: Mapping[str, Any],
    shards: Sequence[Any],
    *,
    partition: str,
) -> list[int]:
    """Return physical-shard ordinals in ``load_sharded_rows`` sample order.

    ``load_sharded_rows`` intentionally sorts tensor names rather than assuming
    that their textual order is their physical PLE order.  This adapter binds
    each sampled row back to the source-layout owner before fitting a local
    codebook, so a shard-local candidate cannot silently borrow another
    shard's training rows.
    """
    sampled = source_sample.get("sampled_tensors")
    if not isinstance(sampled, list) or not sampled:
        raise ValueError("source sample has no per-tensor partition metadata")
    by_tensor = {str(shard.tensor): shard for shard in shards}
    if len(by_tensor) != len(shards):
        raise ValueError("source PLE layout has duplicate tensor names")
    result: list[int] = []
    key = f"{partition}_row_ids"
    for item in sampled:
        if not isinstance(item, dict):
            raise ValueError("source sample tensor record is malformed")
        tensor = str(item.get("tensor", ""))
        shard = by_tensor.get(tensor)
        row_ids = item.get(key)
        if shard is None or not isinstance(row_ids, list) or not all(
            isinstance(row_id, int) and 0 <= row_id < int(shard.rows) for row_id in row_ids
        ):
            raise ValueError(f"source sample does not bind {partition!r} rows to a PLE shard")
        result.extend([int(shard.ordinal)] * len(row_ids))
    if not result:
        raise ValueError(f"source sample has no {partition!r} rows")
    return result


def partition_source_sample_rows_by_shard(
    values: np.ndarray,
    source_sample: Mapping[str, Any],
    shards: Sequence[Any],
    *,
    partition: str,
) -> dict[int, np.ndarray]:
    """Split sampled source rows by their exact physical PLE shard.

    The returned matrices retain the original range-read values.  No source
    table is materialized and no cross-shard sampling is permitted.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("sampled source rows must be a matrix")
    ordinals = sampled_row_shard_ordinals(source_sample, shards, partition=partition)
    if len(ordinals) != values.shape[0]:
        raise ValueError(
            f"{partition} sample count does not match its matrix: {len(ordinals)} vs {values.shape[0]}"
        )
    grouped: dict[int, list[np.ndarray]] = {}
    for row, ordinal in zip(values, ordinals, strict=True):
        grouped.setdefault(ordinal, []).append(row)
    expected = {int(shard.ordinal) for shard in shards}
    if set(grouped) != expected:
        raise ValueError("sampled source rows do not cover every physical PLE shard")
    return {
        ordinal: np.stack(rows).astype(np.float32, copy=False)
        for ordinal, rows in sorted(grouped.items())
    }


def fit_shard_local_product_quantizers(
    train: np.ndarray,
    source_sample: Mapping[str, Any],
    shards: Sequence[Any],
    *,
    quantizer_family: str,
    residual_stages: int,
    subdimension: int,
    cardinality: int,
    iterations: int,
    seed: int,
) -> dict[int, Quantizer]:
    """Fit one deterministic BF16 PQ family per physical PLE shard.

    This is deliberately a materially different representation family from a
    globally shared codebook.  Each codebook is still explicitly billed below;
    local fit is not a hidden free side channel.
    """
    by_shard = partition_source_sample_rows_by_shard(
        train, source_sample, shards, partition="train"
    )
    smallest_population = min(rows.shape[0] for rows in by_shard.values())
    if cardinality > smallest_population:
        raise ValueError(
            "shard-local PQ card exceeds the smallest shard-local train population: "
            f"card={cardinality} smallest={smallest_population}; increase --rows-per-partition"
        )
    return {
        ordinal: fit_declared_quantizer(
            rows,
            quantizer_family=quantizer_family,
            residual_stages=residual_stages,
            subdimension=subdimension,
            cardinality=cardinality,
            iterations=iterations,
            # Preserve candidate determinism but keep physically distinct
            # shards from receiving identical pseudo-random initialization.
            seed=seed + 65537 * ordinal,
        )
        for ordinal, rows in by_shard.items()
    }


def encode_decode_shard_local_product_quantizers(
    values: np.ndarray,
    shard_ordinals: Sequence[int],
    quantizers: Mapping[int, Quantizer],
) -> np.ndarray:
    """Reconstruct rows through their bound physical-shard codebook only."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(shard_ordinals) != values.shape[0]:
        raise ValueError("shard-local values and physical-shard ordinals are incompatible")
    result = np.empty_like(values)
    ordered_ordinals = np.asarray([int(ordinal) for ordinal in shard_ordinals], dtype=np.int64)
    for ordinal in sorted(set(int(value) for value in ordered_ordinals)):
        quantizer = quantizers.get(ordinal)
        if quantizer is None:
            raise ValueError(f"missing shard-local quantizer for physical shard {ordinal}")
        positions = np.flatnonzero(ordered_ordinals == ordinal)
        result[positions] = encode_decode_declared_quantizer(values[positions], quantizer)
    return result


def projected_shard_local_table_accounting(
    quantizers: Mapping[int, Quantizer],
    shard_rows: Mapping[int, int],
    *,
    width: int,
    codebook_directory_bytes_per_shard: int = SHARD_LOCAL_CODEBOOK_DIRECTORY_BYTES,
) -> dict[str, int | float | str]:
    """Bill all local codebooks, codes, directory metadata and decoder state."""
    if not quantizers or set(quantizers) != set(shard_rows):
        raise ValueError("shard-local codebooks must cover exactly the physical source shards")
    if codebook_directory_bytes_per_shard < 0:
        raise ValueError("codebook directory bytes must be non-negative")
    total_rows = 0
    code_payload_bits = 0
    codebook_bytes = 0
    subspaces_per_row: int | None = None
    code_bits_per_subspace: int | None = None
    for ordinal in sorted(quantizers):
        quantizer = quantizers[ordinal]
        rows = int(shard_rows[ordinal])
        if rows <= 0 or width % quantizer.subdimension:
            raise ValueError("shard-local table geometry is incompatible with its PQ family")
        subspaces = width // quantizer.subdimension
        if subspaces != quantizer.subspaces_per_row:
            raise ValueError("shard-local quantizer codebook count does not match table width")
        bits = quantizer.code_bits_per_subspace
        total_rows += rows
        code_payload_bits += rows * subspaces * bits
        codebook_bytes += sum(int(book.size) * 2 for book in quantizer.codebooks)
        if subspaces_per_row is None:
            subspaces_per_row = subspaces
            code_bits_per_subspace = bits
        elif subspaces_per_row != subspaces or code_bits_per_subspace != bits:
            raise ValueError("this screen requires a uniform PQ geometry across local shards")
    code_payload_bytes = (code_payload_bits + 7) // 8
    directory_bytes = len(quantizers) * int(codebook_directory_bytes_per_shard)
    complete_projected_bytes = (
        code_payload_bytes
        + codebook_bytes
        + directory_bytes
        + METADATA_BYTES
        + DECODER_SUPPORT_BYTES
    )
    logical_weights = total_rows * int(width)
    return {
        "codebook_scope": "physical_shard_local",
        "logical_rows": total_rows,
        "logical_weights": logical_weights,
        "physical_shard_count": len(quantizers),
        "subspaces_per_row": int(subspaces_per_row or 0),
        "code_bits_per_subspace": int(code_bits_per_subspace or 0),
        "quantizer_stages": int(getattr(next(iter(quantizers.values())), "stages", 1)),
        "code_payload_bytes": code_payload_bytes,
        "codebook_bytes": codebook_bytes,
        "codebook_directory_bytes_per_shard": int(codebook_directory_bytes_per_shard),
        "codebook_directory_bytes": directory_bytes,
        "metadata_bytes": METADATA_BYTES,
        "decoder_support_bytes": DECODER_SUPPORT_BYTES,
        "complete_projected_bytes": complete_projected_bytes,
        "projected_complete_table_ebpw": complete_projected_bytes * 8.0 / logical_weights,
    }


def active_shard_local_representation_accounting(
    active_shard_ordinals: Sequence[int], quantizers: Mapping[int, Quantizer]
) -> dict[str, int]:
    """Report the cold active-control bytes for a local-codebook candidate.

    This is not a token-rate claim: it records the codewords and all distinct
    local codebooks/directory entries touched by the sealed control.  Decoder
    support is separately billed in the persistent representation closure.
    """
    active = [int(ordinal) for ordinal in active_shard_ordinals]
    if not active:
        raise ValueError("active PLE control has no lookup rows")
    active_code_bits = 0
    for ordinal in active:
        quantizer = quantizers.get(ordinal)
        if quantizer is None:
            raise ValueError(f"active PLE control references missing codebook shard {ordinal}")
        active_code_bits += quantizer.subspaces_per_row * quantizer.code_bits_per_subspace
    unique_ordinals = sorted(set(active))
    active_codebook_bytes = sum(
        sum(int(book.size) * 2 for book in quantizers[ordinal].codebooks)
        for ordinal in unique_ordinals
    )
    active_directory_bytes = len(unique_ordinals) * SHARD_LOCAL_CODEBOOK_DIRECTORY_BYTES
    active_code_bytes = (active_code_bits + 7) // 8
    return {
        "active_code_bytes_for_this_control": active_code_bytes,
        "active_codebook_bytes_for_this_control": active_codebook_bytes,
        "active_codebook_directory_bytes_for_this_control": active_directory_bytes,
        "active_representation_bytes_for_this_control": (
            active_code_bytes + active_codebook_bytes + active_directory_bytes
        ),
    }


def target_summary(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist the next source-control decision at each density pressure point."""
    result = []
    # The lowest points are research horizons, not representations or
    # capability claims.  Persisting them makes the rate frontier retrievable
    # without a later model rediscovering which families were tested there.
    for target in (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
        eligible = [
            item
            for item in candidates
            if item["projected_full_table"]["projected_complete_table_ebpw"] <= target
        ]
        best = min(
            eligible,
            key=lambda item: item["ple_output_metrics"]["relative_l2"],
            default=None,
        )
        result.append(
            {
                "target_projected_table_ebpw": target,
                "best_candidate": (
                    {
                        "codebook_scope": best.get("codebook_scope", CODEBOOK_SCOPE_GLOBAL),
                        "quantizer_family": best.get("quantizer_family", QUANTIZER_FAMILY_SINGLE_STAGE),
                        "residual_stages": best.get("residual_stages", 1),
                        "subdimension": best["subdimension"],
                        "cardinality": best["cardinality"],
                        "repair_count_per_row": best["repair_count_per_row"],
                        "projected_complete_table_ebpw": best["projected_full_table"][
                            "projected_complete_table_ebpw"
                        ],
                        "active_row_relative_l2": best["active_row_metrics"]["global_relative_l2"],
                        "ple_output_relative_l2": best["ple_output_metrics"]["relative_l2"],
                        "ple_output_cosine": best["ple_output_metrics"]["cosine"],
                    }
                    if best is not None
                    else None
                ),
                "status": "ONE_SOURCE_PLE_OUTPUT_CONTROL_ONLY__NOT_A_REPRESENTATION_CLAIM",
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--reference-parity", type=Path, default=DEFAULT_PARITY)
    parser.add_argument("--rows-per-partition", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--codebook-scope",
        choices=(CODEBOOK_SCOPE_GLOBAL, CODEBOOK_SCOPE_SHARD_LOCAL),
        default=CODEBOOK_SCOPE_GLOBAL,
        help="whether one codebook family is shared globally or fitted per physical PLE shard",
    )
    parser.add_argument(
        "--quantizer-family",
        choices=(QUANTIZER_FAMILY_SINGLE_STAGE, QUANTIZER_FAMILY_RESIDUAL_ADDITIVE),
        default=QUANTIZER_FAMILY_SINGLE_STAGE,
        help="single-stage PQ or a greedily decoded additive residual-PQ family",
    )
    parser.add_argument(
        "--residual-stages",
        type=int,
        default=1,
        help="number of additive residual-PQ stages; must be one for single-stage PQ",
    )
    parser.add_argument(
        "--repair-counts",
        default=DEFAULT_REPAIR_COUNTS,
        help="fixed exact-BF16 sparse repairs per projected lookup row",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows_per_partition < 1 or args.iterations < 1:
        raise ValueError("rows-per-partition and iterations must be positive")
    candidate_specs = parse_candidates(args.candidates)
    repair_counts = parse_repair_counts(args.repair_counts)
    validate_repair_counts_for_scope(args.codebook_scope, repair_counts)
    validate_quantizer_configuration(
        args.quantizer_family,
        residual_stages=args.residual_stages,
        repair_counts=repair_counts,
    )
    if (
        args.quantizer_family == QUANTIZER_FAMILY_RESIDUAL_ADDITIVE
        and args.codebook_scope == CODEBOOK_SCOPE_SHARD_LOCAL
    ):
        raise ValueError(
            "the first residual-additive screen is global-codebook only; combine it with physical "
            "shard-local codebooks only after this distinct function family earns a standalone decision"
        )
    started_ns = time.perf_counter_ns()
    control_path = args.control.expanduser().resolve()
    control = json.loads(control_path.read_text(encoding="utf-8"))
    if (
        control.get("schema") != SOURCE_CONTROL_SCHEMA
        or control.get("status")
        != "SOURCE_BOUND_PLE_OUTPUT_CONTROL__NO_INDEPENDENT_PLE_OUTPUT_PARITY"
    ):
        raise ValueError("source PLE control does not have the expected bounded status")
    parity_path = args.reference_parity.expanduser().resolve()
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if (
        parity.get("schema") != REFERENCE_ORACLE_SCHEMA
        or parity.get("status") != "REFERENCE_FRAMEWORK_PLE_FORMULA_PARITY_PASS"
        or (parity.get("source_control") or {}).get("seal_sha256") != control.get("seal_sha256")
    ):
        raise ValueError("reference PLE parity does not bind the supplied source control")
    specimen = control.get("specimen")
    input_control = control.get("input_control")
    outputs = control.get("outputs")
    sequence = control.get("token_sequence")
    if not all(isinstance(value, dict) for value in (specimen, input_control, outputs, sequence)):
        raise ValueError("source PLE control lacks specimen/input/output bindings")
    token_ids = sequence.get("token_ids")
    if not isinstance(token_ids, list) or len(token_ids) != 1 or not all(isinstance(value, int) for value in token_ids):
        raise ValueError("function-aware PLE screen requires exactly one sealed token")
    spec = Path(str(specimen["root"])).expanduser().resolve()
    contract, _ = load_contract(spec, 0)
    config = json.loads((spec / "config.json").read_text(encoding="utf-8"))["text_config"]
    hidden_size = int(config["hidden_size"])
    hc_count = int(config["hc_count"])
    input_state = _read_bound_f32(Path(str(input_control["state"]["path"])), input_control["state"])
    source_injection = _read_bound_f32(
        Path(str(outputs["ple_injection"]["path"])), outputs["ple_injection"]
    )
    reference_injection = _read_bound_f32(
        Path(str(parity["reference_output"]["path"])), parity["reference_output"]
    )
    expected_width = hidden_size * hc_count
    if input_state.size != expected_width or source_injection.size != expected_width:
        raise ValueError("sealed PLE control geometry does not match source config")
    shards, _ = load_layout(spec, contract, int(config["split_ngram_parts"]))
    trace = trace_token_segments(contract, [token_ids])
    embeddings, ordered_lookup_records, active_lookup = load_source_lookup_embeddings(shards, trace, contract)
    active_rows = embeddings.reshape(-1, contract.head_dim)
    active_shard_ordinals = [int(record["shard_ordinal"]) for record in ordered_lookup_records]
    if len(active_shard_ordinals) != active_rows.shape[0]:
        raise ValueError("active source lookup records do not match the PLE embedding rows")
    support = load_source_support(spec, contract, hidden_size=hidden_size, hc_count=hc_count)
    train, heldout, source_sample = load_sharded_rows(
        spec,
        tensor_pattern="ngram_embedding.shard_",
        rows_per_partition=args.rows_per_partition,
    )
    if int(source_sample["total_rows"]) != contract.padded_vocab_size or int(source_sample["width"]) != contract.head_dim:
        raise ValueError("generic PQ source sample does not match the PLE address contract")
    shard_rows = {int(shard.ordinal): int(shard.rows) for shard in shards}
    if sum(shard_rows.values()) != int(source_sample["total_rows"]):
        raise ValueError("physical PLE shard layout does not match sampled table row accounting")
    baseline = output_metrics(reference_injection, source_injection)
    candidates: list[dict[str, Any]] = []
    if args.codebook_scope == CODEBOOK_SCOPE_GLOBAL:
        for subdimension, cardinality in candidate_specs:
            quantizer = fit_declared_quantizer(
                train,
                quantizer_family=args.quantizer_family,
                residual_stages=args.residual_stages,
                subdimension=subdimension,
                cardinality=cardinality,
                iterations=args.iterations,
                seed=args.seed,
            )
            active_base = encode_decode_declared_quantizer(active_rows, quantizer)
            heldout_base = encode_decode_declared_quantizer(heldout, quantizer)
            for repair_count in repair_counts:
                active_reconstructed = apply_exact_sparse_repair(
                    active_rows, active_base, repair_count_per_row=repair_count
                )
                # The source table is the object being encoded.  Selecting its
                # fixed residual offsets is therefore an encoder-time operation,
                # not a learned held-out prediction.  Apply the same fully billed
                # representation to held-out rows so the receipt does not report
                # a base-PQ error beside an active repaired output error.
                heldout_reconstructed = apply_exact_sparse_repair(
                    heldout, heldout_base, repair_count_per_row=repair_count
                )
                candidate_embeddings = active_reconstructed.reshape(1, -1)
                candidate_injection, _, _ = evaluate_ple_sequence(
                    input_state.reshape(1, -1),
                    candidate_embeddings,
                    support,
                    hidden_size=hidden_size,
                    hc_count=hc_count,
                    ngram_size=contract.ngram_size,
                    eps=float(config["rms_norm_eps"]),
                )
                projected = projected_table_accounting(
                    quantizer,
                    total_rows=int(source_sample["total_rows"]),
                    width=int(source_sample["width"]),
                    repair_count_per_row=repair_count,
                )
                projected["active_code_bytes_for_this_control"] = (
                    (
                        active_rows.shape[0]
                        * quantizer.subspaces_per_row
                        * quantizer.code_bits_per_subspace
                        + 7
                    )
                    // 8
                )
                projected["active_sparse_repair_bytes_for_this_control"] = (
                    (
                        active_rows.shape[0]
                        * repair_count
                        * (projected["repair_index_bits"] + projected["repair_value_bits"])
                        + 7
                    )
                    // 8
                )
                projected["active_representation_bytes_for_this_control"] = (
                    projected["active_code_bytes_for_this_control"]
                    + projected["active_sparse_repair_bytes_for_this_control"]
                )
                candidates.append(
                    {
                        "family": (
                            "shared_product_quantization_plus_exact_sparse_repair"
                            if args.quantizer_family == QUANTIZER_FAMILY_SINGLE_STAGE
                            else "global_residual_additive_product_quantization"
                        ),
                        "codebook_scope": CODEBOOK_SCOPE_GLOBAL,
                        "quantizer_family": args.quantizer_family,
                        "residual_stages": int(getattr(quantizer, "stages", 1)),
                        "subdimension": subdimension,
                        "cardinality": cardinality,
                        "code_bits_per_subspace": quantizer.code_bits_per_subspace,
                        "payload_bits_per_weight": quantizer.code_bits_per_subspace / subdimension,
                        "codebook_dtype": "BF16",
                        "repair_count_per_row": repair_count,
                        "repair_semantics": (
                            "fixed highest-error offsets with exact original BF16 values"
                            if args.quantizer_family == QUANTIZER_FAMILY_SINGLE_STAGE
                            else "none; copied source-BF16 repair is deliberately excluded"
                        ),
                        "projected_full_table": projected,
                        "heldout_row_metrics": row_metrics(heldout, heldout_reconstructed),
                        "active_row_metrics": row_metrics(active_rows, active_reconstructed),
                        "ple_output_metrics": output_metrics(reference_injection, candidate_injection),
                        "representation_materialized": False,
                        "direct_execution": False,
                        "lookup_semantics_qualified": False,
                    }
                )
    else:
        heldout_shard_ordinals = sampled_row_shard_ordinals(
            source_sample, shards, partition="heldout"
        )
        if len(heldout_shard_ordinals) != heldout.shape[0]:
            raise ValueError("held-out source shard map does not match sampled rows")
        for subdimension, cardinality in candidate_specs:
            quantizers = fit_shard_local_product_quantizers(
                train,
                source_sample,
                shards,
                quantizer_family=args.quantizer_family,
                residual_stages=args.residual_stages,
                subdimension=subdimension,
                cardinality=cardinality,
                iterations=args.iterations,
                seed=args.seed,
            )
            active_reconstructed = encode_decode_shard_local_product_quantizers(
                active_rows, active_shard_ordinals, quantizers
            )
            heldout_reconstructed = encode_decode_shard_local_product_quantizers(
                heldout, heldout_shard_ordinals, quantizers
            )
            candidate_embeddings = active_reconstructed.reshape(1, -1)
            candidate_injection, _, _ = evaluate_ple_sequence(
                input_state.reshape(1, -1),
                candidate_embeddings,
                support,
                hidden_size=hidden_size,
                hc_count=hc_count,
                ngram_size=contract.ngram_size,
                eps=float(config["rms_norm_eps"]),
            )
            projected = projected_shard_local_table_accounting(
                quantizers,
                shard_rows,
                width=int(source_sample["width"]),
            )
            projected.update(active_shard_local_representation_accounting(active_shard_ordinals, quantizers))
            candidates.append(
                {
                    "family": "physical_shard_local_product_quantization",
                    "codebook_scope": CODEBOOK_SCOPE_SHARD_LOCAL,
                    "quantizer_family": args.quantizer_family,
                    "residual_stages": int(getattr(next(iter(quantizers.values())), "stages", 1)),
                    "subdimension": subdimension,
                    "cardinality": cardinality,
                    "code_bits_per_subspace": next(iter(quantizers.values())).code_bits_per_subspace,
                    "payload_bits_per_weight": next(iter(quantizers.values())).code_bits_per_subspace
                    / subdimension,
                    "codebook_dtype": "BF16",
                    "physical_shard_count": len(quantizers),
                    "repair_count_per_row": 0,
                    "repair_semantics": "none; fixed sparse repair is deliberately excluded",
                    "projected_full_table": projected,
                    "heldout_row_metrics": row_metrics(heldout, heldout_reconstructed),
                    "active_row_metrics": row_metrics(active_rows, active_reconstructed),
                    "ple_output_metrics": output_metrics(reference_injection, candidate_injection),
                    "representation_materialized": False,
                    "direct_execution": False,
                    "lookup_semantics_qualified": False,
                    }
                )
    if args.codebook_scope == CODEBOOK_SCOPE_GLOBAL:
        if args.quantizer_family == QUANTIZER_FAMILY_SINGLE_STAGE:
            status = "SOURCE_PLE_OUTPUT_PQ_PLUS_SPARSE_REPAIR_RATE_DISTORTION_ONLY"
            representation = {
                "codebook_scope": CODEBOOK_SCOPE_GLOBAL,
                "quantizer_family": QUANTIZER_FAMILY_SINGLE_STAGE,
                "base": "global shared BF16 PQ codebooks with projected packed per-row codes plus optional fixed sparse repair",
                "training": "deterministic train-only rows from every source n-gram shard",
                "function_control": "one sealed source PLE BOS injection checked by the reference oracle",
                "billed": {
                    "code_payload": "all projected global lookup rows",
                    "codebooks": "shared BF16 codebooks",
                    "metadata_bytes": METADATA_BYTES,
                    "decoder_support_bytes": DECODER_SUPPORT_BYTES,
                    "sparse_repair": "fixed per-row source-BF16 values plus packed offsets are projected and billed",
                },
            }
            next_action = (
                "Reject uniformly shared PQ plus fixed local repair if its PLE-output curve remains dominated. "
                "Escalate only a surviving non-uniform/generative or protected-repair composition to additional "
                "disjoint source controls; do not treat active rows or projected codes as a materialized complete representation."
            )
        else:
            status = "SOURCE_PLE_OUTPUT_ADDITIVE_PQ_RATE_DISTORTION_ONLY"
            representation = {
                "codebook_scope": CODEBOOK_SCOPE_GLOBAL,
                "quantizer_family": QUANTIZER_FAMILY_RESIDUAL_ADDITIVE,
                "residual_stages": args.residual_stages,
                "base": "global shared BF16 product codebooks greedily decoded as additive residual stages with projected packed stage codes",
                "training": "deterministic train-only rows from every source n-gram shard; each stage fits the BF16-decoded residual from prior stages",
                "function_control": "one sealed source PLE BOS injection checked by the reference oracle",
                "billed": {
                    "code_payload": "all projected global lookup rows times every additive stage",
                    "codebooks": "all global BF16 codebook stages",
                    "metadata_bytes": METADATA_BYTES,
                    "decoder_support_bytes": DECODER_SUPPORT_BYTES,
                    "sparse_repair": "excluded; no copied source-BF16 residual values are present",
                },
            }
            next_action = (
                "Compare this standalone additive residual-PQ function family against the uniform-PQ Scar. "
                "Only a clear source-control survivor may advance to disjoint source controls; a failure rejects "
                "the fixed greedy additive construction, not jointly trained/additive, context-conditioned, generated, learned, or Nova PLE functions."
            )
    else:
        status = "SOURCE_PLE_OUTPUT_SHARD_LOCAL_PQ_RATE_DISTORTION_ONLY"
        representation = {
            "codebook_scope": CODEBOOK_SCOPE_SHARD_LOCAL,
            "quantizer_family": QUANTIZER_FAMILY_SINGLE_STAGE,
            "base": "one BF16 product-codebook family per exact physical PLE checkpoint shard with projected packed per-row codes",
            "training": "deterministic train-only rows remain within their bound physical PLE shard",
            "function_control": "one sealed source PLE BOS injection checked by the reference oracle",
            "billed": {
                "code_payload": "all projected lookup rows across every physical source shard",
                "codebooks": "one separately billed BF16 PQ codebook family per physical source shard",
                "codebook_directory": f"{SHARD_LOCAL_CODEBOOK_DIRECTORY_BYTES} bytes per physical source shard",
                "metadata_bytes": METADATA_BYTES,
                "decoder_support_bytes": DECODER_SUPPORT_BYTES,
                "sparse_repair": "excluded by design so this hierarchy is not a hidden retry of the fixed-repair negative",
            },
        }
        next_action = (
            "Compare this physically shard-local hierarchy against the uniform-PQ Scar. Only a clear source-control "
            "survivor may advance to disjoint source controls; a failure rejects this fixed shard-local PQ family, "
            "not state-conditioned, generated, learned, or Nova-trained PLE functions."
        )
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_control": {
            "path": str(control_path),
            "sha256": sha256_bytes(control_path.read_bytes()),
            "seal_sha256": control.get("seal_sha256"),
        },
        "reference_parity": {
            "path": str(parity_path),
            "sha256": sha256_bytes(parity_path.read_bytes()),
            "seal_sha256": parity.get("seal_sha256"),
            "baseline_source_formula_metrics": baseline,
        },
        "active_lookup": active_lookup,
        "global_training_sample": source_sample,
        "physical_shard_layout": [
            {"ordinal": int(shard.ordinal), "tensor": str(shard.tensor), "rows": int(shard.rows)}
            for shard in shards
        ],
        "representation": representation,
        "candidates": candidates,
        "target_summary": target_summary(candidates),
        "claim_boundary": (
            "This is a one-source-control function-aware PQ-family discriminator. It projects complete lookup-table "
            "code/codebook closure for its declared scope and measures the resulting PLE injection distortion for one sealed BOS input, but it "
            "does not materialize codes or residual records for the table, establish lookup coverage/semantics, execute a native "
            "compact path, prove broader PLE state behavior, close whole-NR EBPW, measure TPS, or preserve capability."
        ),
        "promotion_allowed": False,
        "next": next_action,
        "elapsed_ns": time.perf_counter_ns() - started_ns,
    }
    document["seal_sha256"] = sha256_bytes(json.dumps(document, sort_keys=True).encode("utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": document["status"],
                "candidates": len(candidates),
                "out": str(args.output),
                "seal": document["seal_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
