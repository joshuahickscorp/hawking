#!/usr/bin/env python3.12
"""Payload validators for every receipt schema named by spec re-entry."""

from __future__ import annotations

import json
import math
import pathlib
import sys
from typing import Any

import appendix_contract
import physical_counter_attestation


CONTRACT_SCHEMA = "hawking.spec_receipt_contract.v1"
RUNTIME_PATHS = {"stored", "compact", "hashed", "computed"}
WORKLOADS = {"code", "prose", "tool_json"}
PROPOSERS = {"user_ngram", "suffix_array", "retrieval"}
SCHEMAS = {
    "hawking.spec_tq_batched_parity.v1",
    "hawking.spec_verifier_curve.v1",
    "hawking.spec_proposer_oracle.v1",
    "hawking.spec_cost_oracle.v1",
    "hawking.spec_parallel_head.v1",
    "hawking.spec_learned_draft.v1",
    "hawking.spec_tree_verify.v1",
    "hawking.spec_composition_gate.v1",
}


def requirements() -> dict:
    return {
        "schema": CONTRACT_SCHEMA,
        "outer_schema": appendix_contract.SCHEMA,
        "payload_schemas": {
            "hawking.spec_tq_batched_parity.v1": {
                "required": ["runtime_path", "model_sha256", "tokenizer_sha256", "kernel", "coverage", "target_identity", "batches"],
                "invariants": [
                    "B=1..8", "20 prompts/B", "256 tokens/prompt", "zero mismatch", "zero skipped",
                    "all-linear TQ GPU coverage", "TQ single-token reference", "TQ batch-major verifier",
                ],
            },
            "hawking.spec_verifier_curve.v1": {
                "required": ["runtime_path", "model_sha256", "tokenizer_sha256", "physical_counters_measured", "counter_sources", "batches"],
                "invariants": [
                    "B=1..8", "5 trials/B", "positive monotone total-forward curve",
                    "bound energy/GPU-time/byte counter sources",
                ],
            },
            "hawking.spec_proposer_oracle.v1": {
                "required": ["proposer", "workload", "prompts", "scored_tokens", "draft_lengths"],
                "invariants": ["held-out exact target", "K=2..7", "lookup and miss cost charged"],
            },
            "hawking.spec_cost_oracle.v1": {
                "required": ["workloads"],
                "invariants": ["all workload classes", "speedup LCB >=1.10 per class", "quality pass"],
            },
            "hawking.spec_parallel_head.v1": {
                "required": ["architecture", "placeholder_token_count", "workloads"],
                "invariants": ["zero placeholders", "target-bound training", "all workload classes"],
            },
            "hawking.spec_learned_draft.v1": {
                "required": ["architecture", "placeholder_token_count", "workloads"],
                "invariants": ["zero placeholders", "target-bound training", "all workload classes"],
            },
            "hawking.spec_tree_verify.v1": {
                "required": ["tree_width", "parity_mismatches", "rollback_tests_passed"],
                "invariants": ["zero mismatch", "exact KV rollback", "CPU fallback excluded from speed evidence"],
            },
            "hawking.spec_composition_gate.v1": {
                "required": ["target_runtime_path", "workloads"],
                "invariants": ["all workload classes", "speedup LCB >=1.10 per class", "quality and exactness pass"],
            },
        },
    }


def _nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _positive(value: Any) -> bool:
    return _nonnegative(value) and value > 0


def _batch_rows(payload: dict, errors: list[str]) -> list[dict]:
    rows = payload.get("batches")
    if not isinstance(rows, list) or len(rows) != 8:
        errors.append("batches must contain exactly B=1..8")
        return []
    if [row.get("b") for row in rows if isinstance(row, dict)] != list(range(1, 9)):
        errors.append("batches must be ordered B=1..8")
    return [row for row in rows if isinstance(row, dict)]


def _workload_rows(payload: dict, errors: list[str]) -> list[dict]:
    rows = payload.get("workloads")
    if not isinstance(rows, list):
        errors.append("workloads must be a list")
        return []
    present = {row.get("workload") for row in rows if isinstance(row, dict)}
    if len(rows) != 3 or present != WORKLOADS:
        errors.append("workloads must cover code, prose, and tool_json exactly")
    return [row for row in rows if isinstance(row, dict)]


def _runtime(payload: dict, errors: list[str], field: str = "runtime_path") -> None:
    if payload.get(field) not in RUNTIME_PATHS:
        errors.append(f"{field} must be stored|compact|hashed|computed")


def _validate_parity(payload: dict, errors: list[str]) -> None:
    _runtime(payload, errors)
    mode = payload.get("runtime_path")
    expected_kernel = {
        "stored": "strand_bitslice_gemm_small_stored",
        "compact": "strand_bitslice_gemm_small_compact",
        "hashed": "strand_bitslice_gemm_small_hashed",
        "computed": "strand_bitslice_gemm_small_computed",
    }.get(mode)
    if payload.get("kernel") != expected_kernel:
        errors.append("parity kernel does not match runtime_path")
    for field in ("model_sha256", "tokenizer_sha256"):
        digest = payload.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            errors.append(f"parity {field} is invalid")
    coverage = payload.get("coverage")
    if not isinstance(coverage, dict):
        errors.append("parity TQ coverage is missing")
    else:
        expected = coverage.get("expected_all_linear")
        if (
            not isinstance(expected, int)
            or expected <= 0
            or coverage.get("mapped") != expected
            or coverage.get("gpu_resident") != expected
        ):
            errors.append("parity requires all-linear TQ GPU coverage")
    if payload.get("target_identity") != {
        "reference": "tq_single_token_greedy",
        "verifier": "tq_batch_major_b1_b8",
        "greedy_tie_break": "canonical_qwen_argmax",
        "all_owned_projections_tq_native": True,
    }:
        errors.append("parity target/verifier identity is invalid")
    for row in _batch_rows(payload, errors):
        b = row.get("b", "?")
        if not isinstance(row.get("prompts"), int) or row["prompts"] < 20:
            errors.append(f"B={b} requires at least 20 prompts")
        if not isinstance(row.get("generated_tokens_per_prompt"), int) or row["generated_tokens_per_prompt"] < 256:
            errors.append(f"B={b} requires at least 256 generated tokens/prompt")
        if row.get("exact_token_match") is not True or row.get("mismatches") != 0:
            errors.append(f"B={b} must have exact token match and zero mismatches")
        if row.get("skipped") != 0:
            errors.append(f"B={b} must have zero skipped cases")
    if payload.get("default_change_requested") is not False:
        errors.append("parity evidence cannot request a default change")


def _validate_curve(payload: dict, errors: list[str]) -> None:
    _runtime(payload, errors)
    for field in ("model_sha256", "tokenizer_sha256"):
        digest = payload.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            errors.append(f"verifier curve {field} is invalid")
    if payload.get("physical_counters_measured") is not True:
        errors.append("verifier curve requires measured physical counters")
    sources = payload.get("counter_sources")
    if (
        not isinstance(sources, dict)
        or set(sources) != {"energy_source", "gpu_time_source", "bytes_source"}
        or any(not isinstance(value, str) or not value.strip() for value in sources.values())
    ):
        errors.append("verifier curve physical counter sources are invalid")
    attestation = payload.get("counter_attestation")
    if not isinstance(attestation, dict) or attestation.get("schema") != physical_counter_attestation.SCHEMA:
        errors.append("verifier curve physical counter attestation is missing")
    else:
        unstamped = dict(attestation)
        claimed = unstamped.pop("attestation_sha256", None)
        if claimed != physical_counter_attestation.canonical_sha256(unstamped):
            errors.append("verifier curve counter attestation self-hash mismatch")
    if payload.get("curve_transform") != "none-observed-raw-ratios":
        errors.append("verifier curve must preserve untransformed observed ratios")
    method = payload.get("curve_method")
    if (
        not isinstance(method, dict)
        or set(method) != {
            "warmups_per_batch", "independent_repeats_per_batch",
            "paired_interleaved_baseline", "monotone_transform_applied",
            "ucb_method", "confidence_level", "phase_markers_sha256",
        }
        or not isinstance(method.get("warmups_per_batch"), int)
        or method["warmups_per_batch"] < 3
        or not isinstance(method.get("independent_repeats_per_batch"), int)
        or method["independent_repeats_per_batch"] < 5
        or method.get("paired_interleaved_baseline") is not True
        or method.get("monotone_transform_applied") is not False
        or method.get("ucb_method") != "paired_bootstrap_95"
        or method.get("confidence_level") != 0.95
        or not isinstance(method.get("phase_markers_sha256"), str)
        or len(method["phase_markers_sha256"]) != 64
    ):
        errors.append("verifier curve rigorous measurement method is invalid")
    totals = []
    raw_totals = []
    for row in _batch_rows(payload, errors):
        b = row.get("b", "?")
        if not isinstance(row.get("trials"), int) or row["trials"] < 5:
            errors.append(f"B={b} requires at least five trials")
        median = row.get("median_ns")
        p95 = row.get("p95_ns")
        ucb = row.get("ucb_ns")
        if not (_positive(median) and _positive(p95) and _positive(ucb)):
            errors.append(f"B={b} timing values must be positive")
        elif not (p95 >= median and ucb >= median):
            errors.append(f"B={b} p95/ucb must not be below median")
        total = row.get("total_forward_equiv")
        raw_total = row.get("raw_total_forward_equiv")
        if not _positive(raw_total):
            errors.append(f"B={b} raw_total_forward_equiv must be positive")
        else:
            raw_totals.append(float(raw_total))
        if not _positive(total):
            errors.append(f"B={b} total_forward_equiv must be positive")
        else:
            totals.append(float(total))
        if not _positive(row.get("energy_j_total")):
            errors.append(f"B={b} energy_j_total must be positive")
        if not _positive(row.get("gpu_time_ns")):
            errors.append(f"B={b} gpu_time_ns must be positive")
        byte_map = row.get("bytes")
        if (
            not isinstance(byte_map, dict)
            or set(byte_map) != set(appendix_contract.BYTE_FIELDS)
            or any(not isinstance(byte_map[field], int) or isinstance(byte_map[field], bool)
                   or byte_map[field] < 0 for field in appendix_contract.BYTE_FIELDS)
            or sum(byte_map.values()) <= 0
        ):
            errors.append(f"B={b} physical byte counters are invalid")
    if len(totals) == 8:
        if totals[0] < 1.0:
            errors.append("B=1 total_forward_equiv must be at least one")
        if any(right < left for left, right in zip(totals, totals[1:])):
            errors.append("observed total_forward_equiv must be monotone without transformation")
    if len(totals) == 8 and len(raw_totals) == 8:
        if any(
            not math.isclose(raw, observed, rel_tol=1e-12, abs_tol=1e-12)
            for raw, observed in zip(raw_totals, totals)
        ):
            errors.append("total_forward_equiv differs from observed raw equivalence")
    if payload.get("default_change_requested") is not False:
        errors.append("verifier curve cannot request a default change")


def _validate_proposer(payload: dict, errors: list[str]) -> None:
    if payload.get("proposer") not in PROPOSERS:
        errors.append("proposer is invalid")
    if payload.get("workload") not in WORKLOADS:
        errors.append("workload is invalid")
    if not isinstance(payload.get("prompts"), int) or payload["prompts"] < 10:
        errors.append("proposer oracle requires at least 10 prompts")
    if not isinstance(payload.get("scored_tokens"), int) or payload["scored_tokens"] < 1024:
        errors.append("proposer oracle requires at least 1024 scored tokens")
    if payload.get("held_out_exact_target") is not True:
        errors.append("proposer oracle must use held-out exact target tokens")
    if payload.get("lookup_and_miss_cost_charged") is not True:
        errors.append("lookup and miss cost must be charged")
    rows = payload.get("draft_lengths")
    if not isinstance(rows, list) or [row.get("k") for row in rows if isinstance(row, dict)] != list(range(2, 8)):
        errors.append("draft_lengths must be ordered K=2..7")
        return
    for row in rows:
        if not all(_nonnegative(row.get(field)) for field in ("proposed_tokens", "accepted_tokens", "lookup_ns", "miss_ns")):
            errors.append(f"K={row.get('k')} proposer counts/costs must be non-negative")
        elif row["accepted_tokens"] > row["proposed_tokens"]:
            errors.append(f"K={row.get('k')} accepted tokens exceed proposed tokens")


def _validate_workload_gate(payload: dict, errors: list[str], *, require_exact: bool) -> None:
    for row in _workload_rows(payload, errors):
        workload = row.get("workload", "?")
        if not isinstance(row.get("prompts"), int) or row["prompts"] < 10:
            errors.append(f"{workload} requires at least 10 prompts")
        if not isinstance(row.get("scored_tokens"), int) or row["scored_tokens"] < 1024:
            errors.append(f"{workload} requires at least 1024 scored tokens")
        if not _positive(row.get("speedup_lcb")) or row["speedup_lcb"] < 1.10:
            errors.append(f"{workload} speedup_lcb must be at least 1.10")
        if row.get("quality_passed") is not True:
            errors.append(f"{workload} quality gate did not pass")
        if require_exact and row.get("exact_target_commit") is not True:
            errors.append(f"{workload} exact target commit not proven")


def _validate_learned(payload: dict, errors: list[str]) -> None:
    if not isinstance(payload.get("architecture"), str) or not payload["architecture"]:
        errors.append("architecture must be named")
    if payload.get("placeholder_token_count") != 0:
        errors.append("learned draft receipt must contain zero placeholder tokens")
    if payload.get("trained_on_served_target_distribution") is not True:
        errors.append("learned draft must be trained on served target distribution")
    rows = _workload_rows(payload, errors)
    for row in rows:
        if not _nonnegative(row.get("draft_latency_ns")):
            errors.append(f"{row.get('workload')} draft_latency_ns must be non-negative")
        if not _nonnegative(row.get("accepted_tokens")) or not _nonnegative(row.get("proposed_tokens")):
            errors.append(f"{row.get('workload')} token counts must be non-negative")
        elif row["accepted_tokens"] > row["proposed_tokens"]:
            errors.append(f"{row.get('workload')} accepted exceeds proposed")


def _validate_tree(payload: dict, errors: list[str]) -> None:
    if payload.get("tree_width") not in {2, 4, 8}:
        errors.append("tree_width must be 2, 4, or 8")
    if payload.get("parity_mismatches") != 0:
        errors.append("tree verification requires zero parity mismatches")
    if payload.get("rollback_tests_passed") is not True:
        errors.append("tree KV rollback tests must pass")
    if payload.get("cpu_fallback_used_as_speed_evidence") is not False:
        errors.append("CPU fallback cannot be tree speed evidence")
    if payload.get("longest_argmax_confirmed_prefix") is not True:
        errors.append("tree commit rule must be longest argmax-confirmed prefix")


def validate_payload(schema: str, payload: Any) -> list[str]:
    if schema not in SCHEMAS:
        return [f"unknown speculative receipt schema: {schema}"]
    if not isinstance(payload, dict):
        return ["experiment_payload must be an object"]
    errors: list[str] = []
    if schema == "hawking.spec_tq_batched_parity.v1":
        _validate_parity(payload, errors)
    elif schema == "hawking.spec_verifier_curve.v1":
        _validate_curve(payload, errors)
    elif schema == "hawking.spec_proposer_oracle.v1":
        _validate_proposer(payload, errors)
    elif schema == "hawking.spec_cost_oracle.v1":
        _validate_workload_gate(payload, errors, require_exact=True)
    elif schema in {"hawking.spec_parallel_head.v1", "hawking.spec_learned_draft.v1"}:
        _validate_learned(payload, errors)
    elif schema == "hawking.spec_tree_verify.v1":
        _validate_tree(payload, errors)
    elif schema == "hawking.spec_composition_gate.v1":
        _runtime(payload, errors, "target_runtime_path")
        _validate_workload_gate(payload, errors, require_exact=True)
    return errors


def validate_receipt(receipt: Any, *, known_cell_ids: set[str] | None = None) -> list[str]:
    errors = appendix_contract.validate_receipt(receipt, known_cell_ids=known_cell_ids)
    if not isinstance(receipt, dict) or receipt.get("status") != "complete":
        return errors
    schema = receipt.get("experiment_schema")
    if schema not in SCHEMAS:
        errors.append("complete speculative receipt uses an unknown experiment_schema")
        return errors
    errors.extend(validate_payload(schema, receipt.get("experiment_payload")))
    if schema in {"hawking.spec_tq_batched_parity.v1", "hawking.spec_verifier_curve.v1"}:
        attestation = receipt.get("counter_attestation")
        if not isinstance(attestation, dict):
            errors.append("physical speculative receipt does not preserve counter attestation")
        else:
            unstamped = dict(attestation)
            claimed = unstamped.pop("attestation_sha256", None)
            if claimed != physical_counter_attestation.canonical_sha256(unstamped):
                errors.append("physical speculative receipt counter attestation hash mismatch")
            if receipt.get("physical_counter_attestation_sha256") != claimed:
                errors.append("physical speculative receipt does not bind counter attestation")
        bundle_sha = receipt.get("physical_counter_bundle_sha256")
        if (
            not isinstance(bundle_sha, str) or len(bundle_sha) != 64
            or any(character not in "0123456789abcdef" for character in bundle_sha)
        ):
            errors.append("physical speculative receipt does not bind counter bundle")
    return errors


def _load(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _selftest() -> int:
    parity = {
        "runtime_path": "stored",
        "model_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "kernel": "strand_bitslice_gemm_small_stored",
        "coverage": {
            "expected_all_linear": 196,
            "mapped": 196,
            "gpu_resident": 196,
            "residual_gpu_resident": 0,
        },
        "target_identity": {
            "reference": "tq_single_token_greedy",
            "verifier": "tq_batch_major_b1_b8",
            "greedy_tie_break": "canonical_qwen_argmax",
            "all_owned_projections_tq_native": True,
        },
        "default_change_requested": False,
        "batches": [
            {"b": b, "prompts": 20, "generated_tokens_per_prompt": 256,
             "exact_token_match": True, "mismatches": 0, "skipped": 0}
            for b in range(1, 9)
        ],
    }
    assert validate_payload("hawking.spec_tq_batched_parity.v1", parity) == []
    parity["batches"][4]["mismatches"] = 1
    assert validate_payload("hawking.spec_tq_batched_parity.v1", parity)
    curve = {
        "runtime_path": "compact",
        "model_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "physical_counters_measured": True,
        "counter_attestation": physical_counter_attestation.stamp({
            "schema": physical_counter_attestation.SCHEMA,
        }),
        "curve_transform": "none-observed-raw-ratios",
        "curve_method": {
            "warmups_per_batch": 3,
            "independent_repeats_per_batch": 5,
            "paired_interleaved_baseline": True,
            "monotone_transform_applied": False,
            "ucb_method": "paired_bootstrap_95",
            "confidence_level": 0.95,
            "phase_markers_sha256": "c" * 64,
        },
        "counter_sources": {
            "energy_source": "powermetrics",
            "gpu_time_source": "metal_trace",
            "bytes_source": "gpu_counters",
        },
        "default_change_requested": False,
        "batches": [
            {"b": b, "trials": 5, "median_ns": b * 10, "p95_ns": b * 11,
             "ucb_ns": b * 12, "raw_total_forward_equiv": float(b),
             "total_forward_equiv": float(b), "energy_j_total": b / 10,
             "gpu_time_ns": b * 1000,
             "bytes": {field: (b * 100 if field == "weights" else 0)
                       for field in appendix_contract.BYTE_FIELDS}}
            for b in range(1, 9)
        ],
    }
    assert validate_payload("hawking.spec_verifier_curve.v1", curve) == []
    print("spec_receipt_contract.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--requirements"]:
        print(json.dumps(requirements(), indent=2, sort_keys=True))
        return 0
    if argv == ["--selftest"]:
        return _selftest()
    if len(argv) == 2 and argv[0] == "--validate":
        errors = validate_receipt(_load(pathlib.Path(argv[1])))
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
        return 0 if not errors else 1
    print("usage: spec_receipt_contract.py --requirements | --validate PATH | --selftest", file=sys.stderr)
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
