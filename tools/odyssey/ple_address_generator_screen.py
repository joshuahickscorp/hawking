#!/usr/bin/env python3
"""Screen compact address-conditioned generators for a source-bound PLE table.

This is deliberately not another uniform codec or a renamed PQ sweep.  The
candidate replaces an addressable lookup row with a small deterministic
program of its canonical combined PLE row address. The initial families are a
smooth Fourier coordinate program and a discontinuous additive-hash factor
program. The latter tests whether a shared address-indexed functional
dictionary can explain rows without storing per-row payloads. Their first job
is a cheap structural falsifier: does the source table exhibit enough
address-conditioned regularity for a source-independent generator to be worth
a learned/Nova follow-up? The same generated rows are evaluated through the
independently reference-checked PLE formula on one sealed source input.

The screen bills the generator itself for the entire table but never claims to
have materialized a runnable table, native compact kernel, broader PLE state
semantics, complete NR, TPS, or capability preservation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # Support package imports and direct receipt execution.
    from tools.odyssey.ple_access_trace import (
        LookupShard,
        load_contract,
        load_layout,
        sha256_bytes,
        trace_token_segments,
    )
    from tools.odyssey.ple_lookup_pq_output_screen import output_metrics
    from tools.odyssey.ple_source_output_control import (
        SCHEMA as SOURCE_CONTROL_SCHEMA,
        evaluate_ple_sequence,
        load_source_lookup_embeddings,
        load_source_support,
    )
    from tools.odyssey.sharded_lookup_pq_screen import load_sharded_rows, row_metrics
except ModuleNotFoundError:  # pragma: no cover - direct receipt invocation.
    from ple_access_trace import LookupShard, load_contract, load_layout, sha256_bytes, trace_token_segments
    from ple_lookup_pq_output_screen import output_metrics
    from ple_source_output_control import (
        SCHEMA as SOURCE_CONTROL_SCHEMA,
        evaluate_ple_sequence,
        load_source_lookup_embeddings,
        load_source_support,
    )
    from sharded_lookup_pq_screen import load_sharded_rows, row_metrics


SCHEMA = "hawking.odyssey.ple_address_generator_screen.v1"
REFERENCE_ORACLE_SCHEMA = "hawking.odyssey.ple_reference_formula_oracle.v1"
DEFAULT_CONTROL = Path("receipts/headless/FLASH_PLE_LAYER1_BOS_SOURCE_OUTPUT_CONTROL.json")
DEFAULT_PARITY = Path("receipts/headless/FLASH_PLE_LAYER1_BOS_REFERENCE_FORMULA_PARITY.json")
DEFAULT_FEATURE_COUNTS = "5,9,17,33,65,129"
DEFAULT_HASH_BUCKET_COUNTS = "8,16,32,64,128"
GENERATOR_METADATA_BYTES = 256
GENERATOR_RUNTIME_SUPPORT_BYTES = 4096
HASH_MIX_TERMS = (
    (0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F, 29),
    (0xD6E8FEB86659FD93, 0xA5A3564E27F3A6A1, 31),
    (0x94D049BB133111EB, 0xBF58476D1CE4E5B9, 27),
)


def parse_feature_counts(raw: str) -> list[int]:
    """Parse unique positive, explicit generator feature counts."""
    result: list[int] = []
    for field in raw.split(","):
        field = field.strip()
        if not field:
            continue
        value = int(field)
        if value < 1:
            raise ValueError("feature counts must be positive")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("at least one feature count is required")
    return result


def parse_hash_bucket_counts(raw: str) -> list[int]:
    """Parse unique positive bucket counts for the additive hash program."""
    result: list[int] = []
    for field in raw.split(","):
        field = field.strip()
        if not field:
            continue
        value = int(field)
        if value < 1:
            raise ValueError("hash bucket counts must be positive")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("at least one hash bucket count is required")
    return result


def round_bf16(values: np.ndarray) -> np.ndarray:
    """Round F32 coefficients to the BF16 values that the candidate bills."""
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return (rounded.astype(np.uint32) << 16).view(np.float32)


def address_feature_matrix(
    combined_rows: np.ndarray,
    *,
    shards: Sequence[LookupShard],
    feature_count: int,
) -> np.ndarray:
    """Build the deterministic address-only Fourier features used at runtime.

    The representation is a program, not a table: given an already computed
    PLE combined row address, it derives local-row and shard coordinates then
    generates the same feature sequence.  No source rows participate here.
    """
    feature_count = int(feature_count)
    if feature_count < 1:
        raise ValueError("feature count must be positive")
    rows = np.asarray(combined_rows, dtype=np.int64).reshape(-1)
    if not shards:
        raise ValueError("address generator needs at least one PLE shard")
    starts = np.asarray([shard.global_start for shard in shards], dtype=np.int64)
    ends = starts + np.asarray([shard.rows for shard in shards], dtype=np.int64)
    shard_index = np.searchsorted(starts, rows, side="right") - 1
    if np.any(shard_index < 0) or np.any(rows >= ends[np.maximum(shard_index, 0)]):
        raise ValueError("combined row is outside the resolved PLE layout")
    local = rows - starts[shard_index]
    local_denominator = np.maximum(ends[shard_index] - starts[shard_index] - 1, 1)
    local_coordinate = local.astype(np.float32) / local_denominator.astype(np.float32)
    shard_coordinate = shard_index.astype(np.float32) / max(len(shards) - 1, 1)
    result = np.empty((rows.size, feature_count), dtype=np.float32)
    result[:, 0] = 1.0
    for column in range(1, feature_count):
        family = (column - 1) % 6
        harmonic = 1 + (column - 1) // 6
        angle = np.float32(2.0 * math.pi * harmonic)
        if family == 0:
            result[:, column] = np.sin(angle * local_coordinate)
        elif family == 1:
            result[:, column] = np.cos(angle * local_coordinate)
        elif family == 2:
            result[:, column] = np.sin(angle * shard_coordinate)
        elif family == 3:
            result[:, column] = np.cos(angle * shard_coordinate)
        elif family == 4:
            result[:, column] = np.sin(angle * (local_coordinate + shard_coordinate))
        else:
            result[:, column] = np.cos(angle * (local_coordinate - shard_coordinate))
    return result


def additive_hash_feature_matrix(
    combined_rows: np.ndarray,
    *,
    bucket_count: int,
    terms: int,
) -> np.ndarray:
    """Build a deterministic discontinuous address-factor feature program.

    Each output row is ``bias + sum(term-specific bucket vector)``. A compact
    BF16 coefficient matrix is therefore a direct shared-function description
    for every address, not a stored lookup payload. The independently mixed
    maps are intentionally unlike the smooth Fourier prior already scarred on
    Flash.
    """
    rows = np.asarray(combined_rows, dtype=np.int64).reshape(-1)
    bucket_count = int(bucket_count)
    terms = int(terms)
    if bucket_count < 1:
        raise ValueError("hash bucket count must be positive")
    if terms < 1 or terms > len(HASH_MIX_TERMS):
        raise ValueError(f"hash terms must be in [1, {len(HASH_MIX_TERMS)}]")
    values = rows.astype(np.uint64, copy=False)
    result = np.zeros((rows.size, 1 + terms * bucket_count), dtype=np.float32)
    result[:, 0] = 1.0
    sample_indices = np.arange(rows.size)
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    for term, (multiply, add, shift) in enumerate(HASH_MIX_TERMS[:terms]):
        mixed = values.copy()
        mixed ^= mixed >> np.uint64(shift)
        mixed = (mixed * np.uint64(multiply) + np.uint64(add)) & mask
        mixed ^= mixed >> np.uint64(33 - term)
        buckets = np.remainder(mixed, np.uint64(bucket_count)).astype(np.int64)
        result[sample_indices, 1 + term * bucket_count + buckets] = 1.0
    return result


def fit_address_generator(
    features: np.ndarray, source_rows: np.ndarray, *, ridge_l2: float
) -> np.ndarray:
    """Fit and BF16-round a numerically stable compact address generator.

    High harmonic counts can create near-collinear coordinates under a bounded
    source sample.  A declared small ridge keeps this a function-prior screen
    rather than allowing an unstable least-squares fit to masquerade as a
    generated representation.
    """
    features = np.asarray(features, dtype=np.float32)
    source_rows = np.asarray(source_rows, dtype=np.float32)
    if features.ndim != 2 or source_rows.ndim != 2 or features.shape[0] != source_rows.shape[0]:
        raise ValueError("generator training features and rows are incompatible")
    ridge_l2 = float(ridge_l2)
    if not math.isfinite(ridge_l2) or ridge_l2 <= 0.0:
        raise ValueError("ridge L2 must be finite and positive")
    basis = features.astype(np.float64)
    target = source_rows.astype(np.float64)
    gram = basis.T @ basis
    gram.flat[:: gram.shape[0] + 1] += ridge_l2
    coefficients = np.linalg.solve(gram, basis.T @ target).astype(np.float32)
    return round_bf16(coefficients)


def generator_accounting(*, feature_count: int, total_rows: int, width: int) -> dict[str, int | float]:
    """Bill the complete static generator, not an imaginary generated table."""
    feature_count = int(feature_count)
    total_rows = int(total_rows)
    width = int(width)
    if feature_count < 1 or total_rows < 1 or width < 1:
        raise ValueError("generator accounting geometry must be positive")
    coefficient_bytes = feature_count * width * 2
    complete_projected_bytes = (
        coefficient_bytes + GENERATOR_METADATA_BYTES + GENERATOR_RUNTIME_SUPPORT_BYTES
    )
    return {
        "logical_rows": total_rows,
        "logical_weights": total_rows * width,
        "coefficient_shape": [feature_count, width],
        "coefficient_dtype": "BF16",
        "coefficient_bytes": coefficient_bytes,
        "metadata_bytes": GENERATOR_METADATA_BYTES,
        "runtime_support_bytes": GENERATOR_RUNTIME_SUPPORT_BYTES,
        "complete_projected_bytes": complete_projected_bytes,
        "projected_complete_table_ebpw": complete_projected_bytes * 8.0 / (total_rows * width),
        "active_representation_bytes_per_token": complete_projected_bytes,
    }


def sampled_combined_rows(
    source_sample: dict[str, Any], shards: Sequence[LookupShard], *, partition: str
) -> np.ndarray:
    """Recover the canonical combined address for each generic sampled row."""
    field = f"{partition}_row_ids"
    by_tensor = {shard.tensor: shard for shard in shards}
    addresses: list[int] = []
    sampled_tensors = source_sample.get("sampled_tensors")
    if not isinstance(sampled_tensors, list):
        raise ValueError("generic row sample lacks per-tensor bindings")
    for sample in sampled_tensors:
        if not isinstance(sample, dict):
            raise ValueError("generic row sample contains an invalid tensor binding")
        shard = by_tensor.get(sample.get("tensor"))
        row_ids = sample.get(field)
        if shard is None or not isinstance(row_ids, list):
            raise ValueError("generic row sample does not bind a PLE tensor and rows")
        for row_id in row_ids:
            row_id = int(row_id)
            if row_id < 0 or row_id >= shard.rows:
                raise ValueError("generic row sample contains an out-of-range local row")
            addresses.append(shard.global_start + row_id)
    return np.asarray(addresses, dtype=np.int64)


def target_summary(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Record the best generated-function candidate at standard rate pressure."""
    result = []
    for target in (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
        eligible = [
            candidate
            for candidate in candidates
            if candidate["projected_full_table"]["projected_complete_table_ebpw"] <= target
        ]
        best = min(
            eligible,
            key=lambda candidate: candidate["ple_output_metrics"]["relative_l2"],
            default=None,
        )
        result.append(
            {
                "target_projected_table_ebpw": target,
                "best_candidate": (
                    {
                        "family": best.get("family", "legacy_unspecified_generator"),
                        "feature_count": best["feature_count"],
                        "configuration": best.get("configuration", {}),
                        "projected_complete_table_ebpw": best["projected_full_table"][
                            "projected_complete_table_ebpw"
                        ],
                        "heldout_row_relative_l2": best["heldout_row_metrics"]["global_relative_l2"],
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--reference-parity", type=Path, default=DEFAULT_PARITY)
    parser.add_argument(
        "--family",
        choices=("fourier", "additive-hash", "both"),
        default="fourier",
        help="functional row-generator family to screen; default preserves the original Fourier control",
    )
    parser.add_argument("--rows-per-partition", type=int, default=8)
    parser.add_argument("--feature-counts", default=DEFAULT_FEATURE_COUNTS)
    parser.add_argument("--hash-bucket-counts", default=DEFAULT_HASH_BUCKET_COUNTS)
    parser.add_argument(
        "--hash-terms",
        type=int,
        default=len(HASH_MIX_TERMS),
        help="number of independently mixed address bucket terms for additive-hash candidates",
    )
    parser.add_argument(
        "--ridge-l2",
        type=float,
        default=1e-3,
        help="fixed positive ridge used to stabilize the address-generator fit",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows_per_partition < 1:
        raise ValueError("rows-per-partition must be positive")
    if not math.isfinite(args.ridge_l2) or args.ridge_l2 <= 0.0:
        raise ValueError("ridge-l2 must be finite and positive")
    if args.hash_terms < 1 or args.hash_terms > len(HASH_MIX_TERMS):
        raise ValueError(f"hash-terms must be in [1, {len(HASH_MIX_TERMS)}]")
    feature_counts = parse_feature_counts(args.feature_counts) if args.family in {"fourier", "both"} else []
    hash_bucket_counts = (
        parse_hash_bucket_counts(args.hash_bucket_counts)
        if args.family in {"additive-hash", "both"}
        else []
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
        raise ValueError("address-generator screen requires exactly one sealed token")
    spec = Path(str(specimen["root"])).expanduser().resolve()
    contract, _ = load_contract(spec, 0)
    config = json.loads((spec / "config.json").read_text(encoding="utf-8"))["text_config"]
    hidden_size = int(config["hidden_size"])
    hc_count = int(config["hc_count"])
    input_state = _read_bound_f32(Path(str(input_control["state"]["path"])), input_control["state"])
    source_injection = _read_bound_f32(Path(str(outputs["ple_injection"]["path"])), outputs["ple_injection"])
    reference_injection = _read_bound_f32(
        Path(str(parity["reference_output"]["path"])), parity["reference_output"]
    )
    expected_width = hidden_size * hc_count
    if input_state.size != expected_width or source_injection.size != expected_width:
        raise ValueError("sealed PLE control geometry does not match source config")
    shards, layout = load_layout(spec, contract, int(config["split_ngram_parts"]))
    trace = trace_token_segments(contract, [token_ids])
    embeddings, ordered_lookup_records, active_lookup = load_source_lookup_embeddings(shards, trace, contract)
    active_rows = embeddings.reshape(-1, contract.head_dim)
    active_addresses = np.asarray(
        [int(record["combined_row"]) for record in ordered_lookup_records], dtype=np.int64
    )
    support = load_source_support(spec, contract, hidden_size=hidden_size, hc_count=hc_count)
    train, heldout, source_sample = load_sharded_rows(
        spec,
        tensor_pattern="ngram_embedding.shard_",
        rows_per_partition=args.rows_per_partition,
    )
    if (
        int(source_sample["total_rows"]) != contract.padded_vocab_size
        or int(source_sample["width"]) != contract.head_dim
    ):
        raise ValueError("generic generator source sample does not match the PLE address contract")
    train_addresses = sampled_combined_rows(source_sample, shards, partition="train")
    heldout_addresses = sampled_combined_rows(source_sample, shards, partition="heldout")
    if train_addresses.size != train.shape[0] or heldout_addresses.size != heldout.shape[0]:
        raise ValueError("generic sampled rows and reconstructed PLE addresses disagree")
    baseline = output_metrics(reference_injection, source_injection)
    candidates: list[dict[str, Any]] = []

    def append_candidate(
        *,
        family: str,
        feature_program: str,
        train_features: np.ndarray,
        heldout_features: np.ndarray,
        active_features: np.ndarray,
        configuration: dict[str, int],
    ) -> None:
        """Fit, bill, and measure one source-independent row-function prior."""
        feature_count = int(train_features.shape[1])
        if heldout_features.shape[1] != feature_count or active_features.shape[1] != feature_count:
            raise ValueError("generator feature width drifted across train/held-out/active rows")
        coefficients = fit_address_generator(train_features, train, ridge_l2=args.ridge_l2)
        train_reconstructed = train_features @ coefficients
        heldout_reconstructed = heldout_features @ coefficients
        active_reconstructed = active_features @ coefficients
        candidate_injection, _, _ = evaluate_ple_sequence(
            input_state.reshape(1, -1),
            active_reconstructed.reshape(1, -1),
            support,
            hidden_size=hidden_size,
            hc_count=hc_count,
            ngram_size=contract.ngram_size,
            eps=float(config["rms_norm_eps"]),
        )
        projected = generator_accounting(
            feature_count=feature_count,
            total_rows=int(source_sample["total_rows"]),
            width=int(source_sample["width"]),
        )
        candidates.append(
            {
                "family": family,
                "feature_count": feature_count,
                "feature_program": feature_program,
                "configuration": configuration,
                **configuration,
                "ridge_l2": args.ridge_l2,
                "generator_coefficients": {
                    "shape": [feature_count, int(source_sample["width"])],
                    "dtype": "BF16",
                    "sha256": sha256_bytes(coefficients.astype("<f4", copy=False).tobytes()),
                },
                "projected_full_table": projected,
                "train_row_metrics": row_metrics(train, train_reconstructed),
                "heldout_row_metrics": row_metrics(heldout, heldout_reconstructed),
                "active_row_metrics": row_metrics(active_rows, active_reconstructed),
                "ple_output_metrics": output_metrics(reference_injection, candidate_injection),
                "representation_materialized": False,
                "direct_execution": False,
                "source_independent_when_materialized": True,
                "lookup_semantics_qualified": False,
            }
        )

    for feature_count in feature_counts:
        append_candidate(
            family="address_conditioned_fourier_row_generator",
            feature_program="bias plus deterministic local-row/shard Fourier coordinates",
            train_features=address_feature_matrix(
                train_addresses, shards=shards, feature_count=feature_count
            ),
            heldout_features=address_feature_matrix(
                heldout_addresses, shards=shards, feature_count=feature_count
            ),
            active_features=address_feature_matrix(
                active_addresses, shards=shards, feature_count=feature_count
            ),
            configuration={"fourier_feature_count": feature_count},
        )

    for bucket_count in hash_bucket_counts:
        append_candidate(
            family="address_conditioned_additive_hash_factor_generator",
            feature_program=(
                "bias plus independently mixed address buckets; each row is the sum of shared "
                "term-specific BF16 factor vectors"
            ),
            train_features=additive_hash_feature_matrix(
                train_addresses, bucket_count=bucket_count, terms=args.hash_terms
            ),
            heldout_features=additive_hash_feature_matrix(
                heldout_addresses, bucket_count=bucket_count, terms=args.hash_terms
            ),
            active_features=additive_hash_feature_matrix(
                active_addresses, bucket_count=bucket_count, terms=args.hash_terms
            ),
            configuration={"bucket_count": bucket_count, "hash_terms": args.hash_terms},
        )
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "SOURCE_PLE_OUTPUT_GENERATOR_RATE_DISTORTION_ONLY",
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
        "layout": {
            **layout,
            "feature_address_space": "canonical combined source PLE row index",
            "feature_program": "candidate-specific deterministic address program; never a retained source-row payload",
        },
        "active_lookup": active_lookup,
        "global_training_sample": source_sample,
        "representation": {
            "base": "address-conditioned BF16 function generator; no per-row code or residual table",
            "families_screened": {
                "fourier": "bias plus deterministic local-row/shard Fourier coordinates",
                "additive-hash": (
                    "bias plus independently mixed address buckets and shared term-specific BF16 factor vectors"
                ),
            },
            "requested_family": args.family,
            "training": "deterministic train-only source rows from every PLE shard",
            "fit": "closed-form ridge regression with an explicitly billed BF16 coefficient matrix",
            "ridge_l2": args.ridge_l2,
            "function_control": "one sealed source PLE BOS injection checked by the reference oracle",
            "billed": {
                "generator_coefficients": "all BF16 address-generator coefficients",
                "metadata_bytes": GENERATOR_METADATA_BYTES,
                "runtime_support_bytes": GENERATOR_RUNTIME_SUPPORT_BYTES,
                "source_table": "not retained by the projected generator representation",
            },
        },
        "candidates": candidates,
        "target_summary": target_summary(candidates),
        "claim_boundary": (
            "This is a one-source-control address-generator discriminator. It bills the complete generated-row "
            "program and measures one sealed PLE injection, but it does not establish all-address lookup semantics, "
            "a materialized artifact, native compact execution, broader PLE state behavior, whole-NR EBPW, TPS, or capability."
        ),
        "promotion_allowed": False,
        "next": (
            "Escalate only an address-generator survivor to disjoint source controls and then direct execution. "
            "If a declared functional family is dominated, retain that scoped result as a structural Scar and move "
            "to a materially different state-conditioned/generated or learned/Nova PLE function rather than retrying "
            "uniform codecs, a rescaled version of the same family, or fixed local-error repair."
        ),
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
