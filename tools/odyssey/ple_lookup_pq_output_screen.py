#!/usr/bin/env python3
"""Function-aware PQ discriminator for a source-bound Qwen4-Exp PLE control.

The generic sharded-table PQ screen measures row reconstruction.  This adapter
uses its exact deterministic codebook construction and then asks the useful
next question: how much does a globally billed shared-PQ table perturb an
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
from typing import Any

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
        encode_decode_product_quantizer,
        fit_product_quantizer,
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
        encode_decode_product_quantizer,
        fit_product_quantizer,
        load_sharded_rows,
        row_metrics,
    )


SCHEMA = "hawking.odyssey.ple_lookup_pq_output_screen.v1"
REFERENCE_ORACLE_SCHEMA = "hawking.odyssey.ple_reference_formula_oracle.v1"
DEFAULT_CONTROL = Path("receipts/headless/FLASH_PLE_LAYER1_BOS_SOURCE_OUTPUT_CONTROL.json")
DEFAULT_PARITY = Path("receipts/headless/FLASH_PLE_LAYER1_BOS_REFERENCE_FORMULA_PARITY.json")
DEFAULT_CANDIDATES = "32:4,32:8,16:8,8:4,8:8,8:16"
DEFAULT_REPAIR_COUNTS = "0,1,2,4,8,16,32,64,160"


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
    quantizer: ProductQuantizer,
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


def target_summary(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist the next source-control decision at each density pressure point."""
    result = []
    for target in (0.1, 0.25, 0.5, 0.75, 1.0):
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
    embeddings, _, active_lookup = load_source_lookup_embeddings(shards, trace, contract)
    active_rows = embeddings.reshape(-1, contract.head_dim)
    support = load_source_support(spec, contract, hidden_size=hidden_size, hc_count=hc_count)
    train, heldout, source_sample = load_sharded_rows(
        spec,
        tensor_pattern="ngram_embedding.shard_",
        rows_per_partition=args.rows_per_partition,
    )
    if int(source_sample["total_rows"]) != contract.padded_vocab_size or int(source_sample["width"]) != contract.head_dim:
        raise ValueError("generic PQ source sample does not match the PLE address contract")
    baseline = output_metrics(reference_injection, source_injection)
    candidates: list[dict[str, Any]] = []
    for subdimension, cardinality in candidate_specs:
        quantizer = fit_product_quantizer(
            train,
            subdimension=subdimension,
            cardinality=cardinality,
            iterations=args.iterations,
            seed=args.seed,
        )
        active_base = encode_decode_product_quantizer(active_rows, quantizer)
        heldout_base = encode_decode_product_quantizer(heldout, quantizer)
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
                (active_rows.shape[0] * quantizer.subspaces_per_row * quantizer.code_bits_per_subspace + 7)
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
                    "family": "shared_product_quantization_plus_exact_sparse_repair",
                    "subdimension": subdimension,
                    "cardinality": cardinality,
                    "code_bits_per_subspace": quantizer.code_bits_per_subspace,
                    "payload_bits_per_weight": quantizer.code_bits_per_subspace / subdimension,
                    "codebook_dtype": "BF16",
                    "repair_count_per_row": repair_count,
                    "repair_semantics": "fixed highest-error offsets with exact original BF16 values",
                    "projected_full_table": projected,
                    "heldout_row_metrics": row_metrics(heldout, heldout_reconstructed),
                    "active_row_metrics": row_metrics(active_rows, active_reconstructed),
                    "ple_output_metrics": output_metrics(reference_injection, candidate_injection),
                    "representation_materialized": False,
                    "direct_execution": False,
                    "lookup_semantics_qualified": False,
                }
            )
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "SOURCE_PLE_OUTPUT_PQ_PLUS_SPARSE_REPAIR_RATE_DISTORTION_ONLY",
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
        "representation": {
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
        },
        "candidates": candidates,
        "target_summary": target_summary(candidates),
        "claim_boundary": (
            "This is a one-source-control function-aware PQ discriminator. It projects global lookup-table "
            "code bytes and measures the resulting PLE injection distortion for one sealed BOS input, but it "
            "does not materialize codes or residual records for the table, establish lookup coverage/semantics, execute a native "
            "compact path, prove broader PLE state behavior, close whole-NR EBPW, measure TPS, or preserve capability."
        ),
        "promotion_allowed": False,
        "next": (
            "Reject uniformly shared PQ if its PLE-output curve remains dominated. Escalate only a surviving "
            "non-uniform/generative or protected-repair composition to additional disjoint source controls; "
            "do not treat active rows or projected codes as a materialized complete representation."
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
