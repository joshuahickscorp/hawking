#!/usr/bin/env python3.12
"""Fail-closed acceptance gate for a recovered Temporal Gravity profiler run.

The profiler receipt intentionally preserves partial evidence even when one of
its gates fails.  This validator is the promotion boundary: it accepts only the
promoted resident, native-BF16 token-only path with exact same-process A/B
equivalence, complete physical/stage accounting, nonzero operation provenance,
and at least 95 percent CPU attribution on every measured token.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

RUN_SCHEMA = "hawking.gravity.per_token_cost_ledger_run.v2"
ACCEPTANCE_SCHEMA = "hawking.gravity.profiler_acceptance.v1"
DEFAULT_MIN_TOKENS = 16
DEFAULT_MIN_ATTRIBUTION = 0.95
MATH_PRESERVE_FIXED_ACTIVE_BYTES = 3_054_873_024
MATH_PRESERVE_EXPECTED_ROUTED_EXPERTS = 600
MATH_PRESERVE_EXPECTED_ROUTED_PROJECTIONS = 1_800
MATH_PRESERVE_PROJECTION_BYTES = {
    "r4": 409_604,
    "r0": 1_378_308,
    "native_bf16": 25_165_824,
}
MATH_PRESERVE_WHOLE_TOKEN_MIN_BYTES = 3_792_160_224
MATH_PRESERVE_WHOLE_TOKEN_LAYER_CONSTRAINED_MAX_BYTES = 45_570_216_852


def _at(value: Any, *path: str) -> Any:
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _is_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def evaluate(
    receipt: dict[str, Any],
    *,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    min_attribution: float = DEFAULT_MIN_ATTRIBUTION,
) -> dict[str, Any]:
    failures: list[str] = []
    checks: dict[str, bool] = {}

    def require(name: str, condition: bool, detail: str) -> None:
        checks[name] = bool(condition)
        if not condition:
            failures.append(f"{name}: {detail}")

    require(
        "run_schema",
        receipt.get("schema") == RUN_SCHEMA,
        f"expected {RUN_SCHEMA!r}, got {receipt.get('schema')!r}",
    )

    resolved = _at(receipt, "run_configuration", "resolved")
    required_flags = {
        "resident_state": True,
        "gpu_native_bf16_head": True,
        "full_logits_readback": False,
        "expert_wave": False,
        "cost_ledger": True,
        "tcb_trace": "off",
    }
    require(
        "promoted_resolved_path",
        isinstance(resolved, dict)
        and all(resolved.get(key) == expected for key, expected in required_flags.items()),
        f"resolved flags must equal {required_flags!r}, got {resolved!r}",
    )

    active_contract = receipt.get("math_preserve_active_byte_contract")
    require(
        "math_preserve_active_byte_contract",
        isinstance(active_contract, dict)
        and active_contract.get("fixed_active_bytes")
        == MATH_PRESERVE_FIXED_ACTIVE_BYTES
        and active_contract.get("expected_routed_experts")
        == MATH_PRESERVE_EXPECTED_ROUTED_EXPERTS
        and active_contract.get("expected_routed_projections")
        == MATH_PRESERVE_EXPECTED_ROUTED_PROJECTIONS
        and active_contract.get("r4_projection_bytes")
        == MATH_PRESERVE_PROJECTION_BYTES["r4"]
        and active_contract.get("r0_projection_bytes")
        == MATH_PRESERVE_PROJECTION_BYTES["r0"]
        and active_contract.get("native_bf16_projection_bytes")
        == MATH_PRESERVE_PROJECTION_BYTES["native_bf16"]
        and active_contract.get("whole_token_min_bytes")
        == MATH_PRESERVE_WHOLE_TOKEN_MIN_BYTES
        and active_contract.get("whole_token_layer_constrained_max_bytes")
        == MATH_PRESERVE_WHOLE_TOKEN_LAYER_CONSTRAINED_MAX_BYTES
        and active_contract.get("physical_dram_claim", "").startswith("none"),
        f"current Math-Preserve byte contract is absent or wrong: {active_contract!r}",
    )

    ab = _at(receipt, "same_process_exact_input_ab", "equivalent")
    required_ab = (
        "all_required_checks_passed",
        "same_token",
        "same_output_mode",
        "same_logits",
        "same_topk_diagnostics",
        "same_router_and_indexer_choices",
        "same_resident_wait_count",
        "same_physical_command_and_encoder_counts",
        "same_per_token_cache_delta",
        "ledger_command_count_matches_physical_trace",
    )
    require(
        "same_process_exact_ab",
        isinstance(ab, dict) and all(ab.get(key) is True for key in required_ab),
        f"all exact A/B checks must be true; got {ab!r}",
    )
    for arm in ("unprofiled", "profiled"):
        output = _at(receipt, "same_process_exact_input_ab", arm, "output")
        require(
            f"ab_{arm}_token_only",
            isinstance(output, dict)
            and output.get("mode") == "token_plus_topk_diagnostics"
            and output.get("full_logits_readback") is False
            and isinstance(output.get("token"), int),
            f"{arm} output is not the token-only contract: {output!r}",
        )

    coverage = receipt.get("coverage_gate")
    coverage_required = _at(coverage, "required_attributed_fraction")
    coverage_minimum = _at(coverage, "minimum_token_attributed_fraction")
    require(
        "coverage_contract",
        _is_number(coverage_required)
        and float(coverage_required) >= min_attribution
        and _is_number(coverage_minimum)
        and float(coverage_minimum) >= min_attribution
        and _at(coverage, "all_tokens_at_least_95_percent") is True,
        f"coverage gate must be >= {min_attribution:.3f}; got {coverage!r}",
    )
    require(
        "coverage_exact_stage_accounting",
        _at(coverage, "untagged_gpu_dispatches") == 0
        and _at(coverage, "no_untagged_gpu_dispatches") is True
        and _at(coverage, "stage_dispatch_count_mismatches") == 0
        and _at(coverage, "all_command_buffer_stage_counts_exact") is True,
        f"GPU stage coverage is incomplete: {coverage!r}",
    )
    require(
        "coverage_no_generic_orchestration",
        _at(coverage, "generic_orchestration_lines") == []
        and _at(coverage, "no_generic_orchestration_bucket") is True,
        f"generic orchestration/residual line remains: {coverage!r}",
    )
    require(
        "coverage_operations_provenance",
        _at(coverage, "operations_nonzero_every_token") is True,
        f"operation provenance gate failed: {coverage!r}",
    )

    tokens = receipt.get("tokens")
    aggregate = receipt.get("aggregate")
    token_count = _at(aggregate, "token_count")
    require(
        "sustained_token_count",
        isinstance(tokens, list)
        and len(tokens) >= min_tokens
        and token_count == len(tokens),
        f"need >= {min_tokens} tokens and exact aggregate count; "
        f"tokens={len(tokens) if isinstance(tokens, list) else None}, aggregate={token_count!r}",
    )
    require(
        "all_gpu_timestamps_present",
        _at(aggregate, "tokens_missing_gpu_timestamps") == 0,
        f"aggregate reports missing GPU timestamps: {_at(aggregate, 'tokens_missing_gpu_timestamps')!r}",
    )
    require(
        "aggregate_sample_counts",
        isinstance(token_count, int)
        and all(
            _at(aggregate, field, "n") == token_count
            for field in (
                "wall_us",
                "unattributed_us",
                "attributed_fraction",
                "profiler_overhead_us",
                "device_gpu_execution_us",
                "active_bytes_read",
                "bytes_moved_total",
            )
        ),
        f"aggregate percentile sample counts do not all equal {token_count!r}",
    )
    require(
        "declared_ledger_token_count",
        receipt.get("ledger_tokens") == token_count,
        f"top-level ledger_tokens={receipt.get('ledger_tokens')!r}, aggregate={token_count!r}",
    )

    token_failures: list[str] = []
    if isinstance(tokens, list):
        for token_index, token in enumerate(tokens, start=1):
            if not isinstance(token, dict):
                token_failures.append(f"token {token_index}: token entry is not an object")
                continue
            if token.get("decode_token_index") != token_index:
                token_failures.append(
                    f"token {token_index}: decode_token_index={token.get('decode_token_index')!r}"
                )
            ledger = _at(token, "ledger")
            if not isinstance(ledger, dict):
                token_failures.append(f"token {token_index}: missing ledger")
                continue
            output = token.get("output")
            if not (
                isinstance(output, dict)
                and output.get("mode") == "token_plus_topk_diagnostics"
                and output.get("full_logits_readback") is False
                and isinstance(output.get("token"), int)
            ):
                token_failures.append(f"token {token_index}: output is not token-only")

            attributed = ledger.get("attributed_fraction")
            if not (_is_number(attributed) and float(attributed) >= min_attribution):
                token_failures.append(
                    f"token {token_index}: attributed_fraction={attributed!r}"
                )
            if ledger.get("unattributed_name") != "unattributed":
                token_failures.append(
                    f"token {token_index}: unattributed line is not explicitly named"
                )
            buckets = ledger.get("buckets_us")
            if not isinstance(buckets, dict):
                token_failures.append(f"token {token_index}: missing exclusive buckets")
            elif any(
                "orchestration" in key or "residual_scoped" in key for key in buckets
            ):
                token_failures.append(
                    f"token {token_index}: generic orchestration/residual bucket present"
                )
            elif not all(_is_nonnegative_int(value) for value in buckets.values()):
                token_failures.append(
                    f"token {token_index}: bucket timings are not nonnegative integers"
                )
            else:
                attributed_us = ledger.get("attributed_us")
                unattributed_us = ledger.get("unattributed_us")
                wall_us = ledger.get("wall_us")
                signed_us = ledger.get("unattributed_signed_us")
                bucket_total = sum(buckets.values())
                expected_fraction = (
                    bucket_total / wall_us
                    if _is_nonnegative_int(wall_us) and wall_us > 0
                    else (1.0 if wall_us == 0 else None)
                )
                if not (
                    _is_nonnegative_int(attributed_us)
                    and _is_nonnegative_int(unattributed_us)
                    and _is_nonnegative_int(wall_us)
                    and isinstance(signed_us, int)
                    and not isinstance(signed_us, bool)
                    and signed_us >= 0
                    and attributed_us == bucket_total
                    and attributed_us + unattributed_us == wall_us
                    and signed_us == unattributed_us
                    and _is_number(attributed)
                    and expected_fraction is not None
                    and math.isclose(
                        float(attributed),
                        expected_fraction,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    token_failures.append(
                        f"token {token_index}: exclusive bucket arithmetic is inconsistent"
                    )

            bucket_sources = ledger.get("bucket_sources")
            if not (
                isinstance(bucket_sources, dict)
                and isinstance(buckets, dict)
                and set(buckets).issubset(bucket_sources)
            ):
                token_failures.append(
                    f"token {token_index}: one or more CPU buckets lack source provenance"
                )

            counters = ledger.get("counters")
            if not isinstance(counters, dict):
                token_failures.append(f"token {token_index}: missing counters")
                continue
            commands = counters.get("command_buffers_submitted")
            dispatches = counters.get("dispatches_encoded")
            synchronizations = counters.get("synchronization_points")
            if not all(
                isinstance(value, int) and value > 0
                for value in (commands, dispatches, synchronizations)
            ):
                token_failures.append(
                    f"token {token_index}: nonpositive command/dispatch/sync counts"
                )
            if not (
                isinstance(counters.get("operations"), int)
                and counters["operations"] > 0
                and isinstance(counters.get("source_modelled_fp_operations"), int)
                and counters["source_modelled_fp_operations"] > 0
            ):
                token_failures.append(f"token {token_index}: operation counters are zero")
            categories = counters.get("active_bytes_by_category")
            if not (
                isinstance(categories, dict)
                and all(_is_nonnegative_int(value) for value in categories.values())
                and sum(categories.values()) == counters.get("active_bytes_read")
            ):
                token_failures.append(
                    f"token {token_index}: active-byte categories do not partition the total"
                )
            representations = counters.get("routed_representations")
            rep_fields = {
                "r4": ("r4_projection_touches", "r4_active_bytes"),
                "r0": ("r0_projection_touches", "r0_active_bytes"),
                "native_bf16": (
                    "native_bf16_projection_touches",
                    "native_bf16_active_bytes",
                ),
            }
            if not isinstance(representations, dict):
                token_failures.append(
                    f"token {token_index}: routed representation evidence is missing"
                )
            else:
                rep_values = list(representations.values())
                valid_rep_values = all(_is_nonnegative_int(value) for value in rep_values)
                touches = {
                    label: representations.get(touch_field)
                    for label, (touch_field, _) in rep_fields.items()
                }
                observed_rep_bytes = {
                    label: representations.get(byte_field)
                    for label, (_, byte_field) in rep_fields.items()
                }
                expected_rep_bytes = {
                    label: (
                        touches[label] * MATH_PRESERVE_PROJECTION_BYTES[label]
                        if _is_nonnegative_int(touches[label])
                        else None
                    )
                    for label in rep_fields
                }
                total_projection_touches = (
                    sum(touches.values())
                    if all(_is_nonnegative_int(value) for value in touches.values())
                    else -1
                )
                expected_routed_bytes = (
                    sum(expected_rep_bytes.values())
                    if all(
                        _is_nonnegative_int(value)
                        for value in expected_rep_bytes.values()
                    )
                    else -1
                )
                expected_total_bytes = (
                    MATH_PRESERVE_FIXED_ACTIVE_BYTES + expected_routed_bytes
                    if expected_routed_bytes >= 0
                    else -1
                )
                fixed_observed = (
                    sum(
                        value
                        for key, value in categories.items()
                        if key != "routed_experts"
                    )
                    if isinstance(categories, dict)
                    and all(_is_nonnegative_int(value) for value in categories.values())
                    else -1
                )
                if not (
                    valid_rep_values
                    and representations.get("other_projection_touches") == 0
                    and representations.get("other_active_bytes") == 0
                    and total_projection_touches
                    == MATH_PRESERVE_EXPECTED_ROUTED_PROJECTIONS
                    and all(value % 3 == 0 for value in touches.values())
                    and sum(value // 3 for value in touches.values())
                    == MATH_PRESERVE_EXPECTED_ROUTED_EXPERTS
                    and observed_rep_bytes == expected_rep_bytes
                    and isinstance(categories, dict)
                    and categories.get("routed_experts") == expected_routed_bytes
                    and fixed_observed == MATH_PRESERVE_FIXED_ACTIVE_BYTES
                    and counters.get("active_bytes_read") == expected_total_bytes
                    and ledger.get("geometry_active_bytes") == expected_total_bytes
                    and MATH_PRESERVE_WHOLE_TOKEN_MIN_BYTES
                    <= expected_total_bytes
                    <= MATH_PRESERVE_WHOLE_TOKEN_LAYER_CONSTRAINED_MAX_BYTES
                    and _is_number(ledger.get("active_bytes_vs_geometry_fraction"))
                    and math.isclose(
                        float(ledger["active_bytes_vs_geometry_fraction"]),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    token_failures.append(
                        f"token {token_index}: Math-Preserve active bytes do not "
                        "match the exact route-conditioned representation formula"
                    )

            device = ledger.get("device")
            command_buffers = _at(device, "command_buffers")
            if not (
                isinstance(device, dict)
                and device.get("gpu_timestamps_missing") == 0
                and device.get("gpu_timestamps_observed") == commands
                and isinstance(command_buffers, list)
                and len(command_buffers) == commands
            ):
                token_failures.append(
                    f"token {token_index}: physical command/timestamp count mismatch"
                )
                continue
            physical_dispatches = 0
            for command_index, command in enumerate(command_buffers, start=1):
                if not isinstance(command, dict):
                    token_failures.append(
                        f"token {token_index} command {command_index}: not an object"
                    )
                    continue
                composition = command.get("stage_composition")
                valid_composition = (
                    isinstance(composition, list)
                    and bool(composition)
                    and all(
                        isinstance(stage, dict)
                        and isinstance(stage.get("stage"), str)
                        and stage.get("stage") != "untagged"
                        and isinstance(stage.get("dispatches"), int)
                        and not isinstance(stage.get("dispatches"), bool)
                        and stage.get("dispatches") > 0
                        for stage in composition
                    )
                )
                stage_total = (
                    sum(stage["dispatches"] for stage in composition)
                    if valid_composition
                    else -1
                )
                dispatches_in_buffer = command.get("dispatches_in_buffer")
                if _is_nonnegative_int(dispatches_in_buffer):
                    physical_dispatches += dispatches_in_buffer
                if not (
                    _is_nonnegative_int(dispatches_in_buffer)
                    and dispatches_in_buffer > 0
                    and valid_composition
                    and command.get("stage_dispatches_match_buffer") is True
                    and command.get("stage_dispatches_total")
                    == dispatches_in_buffer
                    == stage_total
                    and _is_number(command.get("gpu_start_s"))
                    and _is_number(command.get("gpu_end_s"))
                    and command["gpu_end_s"] >= command["gpu_start_s"]
                    and _is_nonnegative_int(command.get("gpu_execution_us"))
                ):
                    token_failures.append(
                        f"token {token_index} command {command_index}: "
                        "timestamps/stage composition do not exactly cover physical dispatches"
                    )
            if physical_dispatches != dispatches:
                token_failures.append(
                    f"token {token_index}: physical dispatch sum {physical_dispatches} "
                    f"!= counter {dispatches}"
                )

    require(
        "every_token_contract",
        not token_failures,
        "; ".join(token_failures[:20])
        + (f"; ... {len(token_failures) - 20} more" if len(token_failures) > 20 else ""),
    )

    return {
        "schema": ACCEPTANCE_SCHEMA,
        "accepted": not failures,
        "run_schema": receipt.get("schema"),
        "minimum_tokens_required": min_tokens,
        "minimum_attributed_fraction_required": min_attribution,
        "checks": checks,
        "failures": failures,
        "observed": {
            "tokens": len(tokens) if isinstance(tokens, list) else None,
            "minimum_token_attributed_fraction": coverage_minimum,
            "aggregate_tokens_missing_gpu_timestamps": _at(
                aggregate, "tokens_missing_gpu_timestamps"
            ),
        },
    }


def _synthetic_receipt() -> dict[str, Any]:
    routed_bytes = (
        MATH_PRESERVE_EXPECTED_ROUTED_PROJECTIONS
        * MATH_PRESERVE_PROJECTION_BYTES["r4"]
    )
    active_bytes = MATH_PRESERVE_FIXED_ACTIVE_BYTES + routed_bytes
    stage = {"stage": "routed_experts", "dispatches": 2}
    command = {
        "gpu_start_s": 1.0,
        "gpu_end_s": 1.0001,
        "gpu_execution_us": 100,
        "dispatches_in_buffer": 2,
        "stage_composition": [stage],
        "stage_dispatches_total": 2,
        "stage_dispatches_match_buffer": True,
    }
    ledger = {
        "wall_us": 1000,
        "attributed_us": 990,
        "unattributed_us": 10,
        "unattributed_signed_us": 10,
        "attributed_fraction": 0.99,
        "unattributed_name": "unattributed",
        "buckets_us": {"routed_experts": 990},
        "bucket_sources": {"routed_experts": "synthetic source"},
        "counters": {
            "command_buffers_submitted": 1,
            "dispatches_encoded": 2,
            "synchronization_points": 1,
            "operations": 10,
            "source_modelled_fp_operations": 8,
            "active_bytes_read": active_bytes,
            "active_bytes_by_category": {
                "attention": MATH_PRESERVE_FIXED_ACTIVE_BYTES,
                "routed_experts": routed_bytes,
            },
            "routed_representations": {
                "r4_projection_touches": MATH_PRESERVE_EXPECTED_ROUTED_PROJECTIONS,
                "r4_active_bytes": routed_bytes,
                "r0_projection_touches": 0,
                "r0_active_bytes": 0,
                "native_bf16_projection_touches": 0,
                "native_bf16_active_bytes": 0,
                "other_projection_touches": 0,
                "other_active_bytes": 0,
            },
        },
        "geometry_active_bytes": active_bytes,
        "active_bytes_vs_geometry_fraction": 1.0,
        "device": {
            "gpu_timestamps_observed": 1,
            "gpu_timestamps_missing": 0,
            "command_buffers": [command],
        },
    }
    output = {
        "mode": "token_plus_topk_diagnostics",
        "token": 7,
        "full_logits_readback": False,
        "topk_indices": [7],
        "topk_values": [1.0],
    }
    ab_checks = {
        key: True
        for key in (
            "all_required_checks_passed",
            "same_token",
            "same_output_mode",
            "same_logits",
            "same_topk_diagnostics",
            "same_router_and_indexer_choices",
            "same_resident_wait_count",
            "same_physical_command_and_encoder_counts",
            "same_per_token_cache_delta",
            "ledger_command_count_matches_physical_trace",
        )
    }
    return {
        "schema": RUN_SCHEMA,
        "math_preserve_active_byte_contract": {
            "fixed_active_bytes": MATH_PRESERVE_FIXED_ACTIVE_BYTES,
            "expected_routed_experts": MATH_PRESERVE_EXPECTED_ROUTED_EXPERTS,
            "expected_routed_projections": MATH_PRESERVE_EXPECTED_ROUTED_PROJECTIONS,
            "r4_projection_bytes": MATH_PRESERVE_PROJECTION_BYTES["r4"],
            "r0_projection_bytes": MATH_PRESERVE_PROJECTION_BYTES["r0"],
            "native_bf16_projection_bytes": MATH_PRESERVE_PROJECTION_BYTES[
                "native_bf16"
            ],
            "whole_token_min_bytes": MATH_PRESERVE_WHOLE_TOKEN_MIN_BYTES,
            "whole_token_layer_constrained_max_bytes": (
                MATH_PRESERVE_WHOLE_TOKEN_LAYER_CONSTRAINED_MAX_BYTES
            ),
            "physical_dram_claim": "none; synthetic fixture",
        },
        "run_configuration": {
            "resolved": {
                "resident_state": True,
                "gpu_native_bf16_head": True,
                "full_logits_readback": False,
                "expert_wave": False,
                "cost_ledger": True,
                "tcb_trace": "off",
            }
        },
        "same_process_exact_input_ab": {
            "unprofiled": {"output": copy.deepcopy(output)},
            "profiled": {"output": copy.deepcopy(output)},
            "equivalent": ab_checks,
        },
        "coverage_gate": {
            "required_attributed_fraction": 0.95,
            "minimum_token_attributed_fraction": 0.99,
            "all_tokens_at_least_95_percent": True,
            "untagged_gpu_dispatches": 0,
            "no_untagged_gpu_dispatches": True,
            "stage_dispatch_count_mismatches": 0,
            "all_command_buffer_stage_counts_exact": True,
            "generic_orchestration_lines": [],
            "no_generic_orchestration_bucket": True,
            "operations_nonzero_every_token": True,
        },
        "aggregate": {
            "token_count": 1,
            "tokens_missing_gpu_timestamps": 0,
            "wall_us": {"n": 1},
            "unattributed_us": {"n": 1},
            "attributed_fraction": {"n": 1},
            "profiler_overhead_us": {"n": 1},
            "device_gpu_execution_us": {"n": 1},
            "active_bytes_read": {"n": 1},
            "bytes_moved_total": {"n": 1},
        },
        "ledger_tokens": 1,
        "tokens": [
            {
                "decode_token_index": 1,
                "output": output,
                "ledger": ledger,
            }
        ],
    }


def selftest() -> int:
    good = _synthetic_receipt()
    accepted = evaluate(good, min_tokens=1)
    assert accepted["accepted"], accepted

    bad = copy.deepcopy(good)
    bad["tokens"][0]["ledger"]["device"]["command_buffers"][0][
        "stage_composition"
    ][0]["stage"] = "untagged"
    rejected = evaluate(bad, min_tokens=1)
    assert not rejected["accepted"], rejected
    assert not rejected["checks"]["every_token_contract"], rejected

    bad = copy.deepcopy(good)
    bad["same_process_exact_input_ab"]["equivalent"]["same_token"] = False
    rejected = evaluate(bad, min_tokens=1)
    assert not rejected["accepted"], rejected
    assert not rejected["checks"]["same_process_exact_ab"], rejected

    bad = copy.deepcopy(good)
    bad["tokens"][0]["ledger"]["attributed_us"] = 989
    rejected = evaluate(bad, min_tokens=1)
    assert not rejected["accepted"], rejected
    assert not rejected["checks"]["every_token_contract"], rejected

    bad = copy.deepcopy(good)
    bad["tokens"][0]["ledger"]["device"]["command_buffers"][0]["gpu_start_s"] = None
    rejected = evaluate(bad, min_tokens=1)
    assert not rejected["accepted"], rejected
    assert not rejected["checks"]["every_token_contract"], rejected

    bad = copy.deepcopy(good)
    bad["tokens"][0]["ledger"]["counters"]["routed_representations"][
        "r4_projection_touches"
    ] -= 1
    rejected = evaluate(bad, min_tokens=1)
    assert not rejected["accepted"], rejected
    assert not rejected["checks"]["every_token_contract"], rejected

    # Internally byte-consistent, but impossible under the per-layer census:
    # there are not eight native-bf16 routed experts available in every layer.
    bad = copy.deepcopy(good)
    ledger = bad["tokens"][0]["ledger"]
    reps = ledger["counters"]["routed_representations"]
    reps["r4_projection_touches"] = 0
    reps["r4_active_bytes"] = 0
    reps["native_bf16_projection_touches"] = (
        MATH_PRESERVE_EXPECTED_ROUTED_PROJECTIONS
    )
    impossible_routed_bytes = (
        MATH_PRESERVE_EXPECTED_ROUTED_PROJECTIONS
        * MATH_PRESERVE_PROJECTION_BYTES["native_bf16"]
    )
    reps["native_bf16_active_bytes"] = impossible_routed_bytes
    impossible_total_bytes = MATH_PRESERVE_FIXED_ACTIVE_BYTES + impossible_routed_bytes
    ledger["counters"]["active_bytes_by_category"]["routed_experts"] = (
        impossible_routed_bytes
    )
    ledger["counters"]["active_bytes_read"] = impossible_total_bytes
    ledger["geometry_active_bytes"] = impossible_total_bytes
    rejected = evaluate(bad, min_tokens=1)
    assert not rejected["accepted"], rejected
    assert not rejected["checks"]["every_token_contract"], rejected

    print(
        json.dumps(
            {
                "schema": ACCEPTANCE_SCHEMA,
                "selftest": "PASS",
                "accepts_complete_fixture": True,
                "rejects_untagged_dispatch": True,
                "rejects_ab_mismatch": True,
                "rejects_inconsistent_exclusive_arithmetic": True,
                "rejects_missing_physical_timestamp": True,
                "rejects_route_conditioned_active_byte_mismatch": True,
                "rejects_architecturally_impossible_representation_mix": True,
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed acceptance gate for a Temporal Gravity profiler receipt."
    )
    parser.add_argument("receipt", nargs="?", help="profiler run JSON")
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument(
        "--min-attribution", type=float, default=DEFAULT_MIN_ATTRIBUTION
    )
    parser.add_argument("--out", help="optional acceptance JSON path")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.receipt:
        parser.error("receipt is required unless --selftest is used")
    if args.min_tokens < 1:
        parser.error("--min-tokens must be at least 1")
    if not 0.0 < args.min_attribution <= 1.0:
        parser.error("--min-attribution must be in (0, 1]")

    receipt_path = Path(args.receipt)
    receipt = json.loads(receipt_path.read_text())
    result = evaluate(
        receipt,
        min_tokens=args.min_tokens,
        min_attribution=args.min_attribution,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
