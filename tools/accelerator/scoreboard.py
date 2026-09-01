"""Build the compact Accelerator scoreboard from sealed receipt files.

This is deliberately a small *derived view*, not a telemetry service.  It
normalizes fields that serious runs already record, preserves missing values as
``None`` (never zero), and exposes the same conservative promotion ordering as
the Physical Graph Compiler.  A scoreboard row is evidence about its source
receipt; it is not a new measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Optional

# The tool is both importable in the test suite and directly executable from a
# shell.  Add the repository root only for the latter so ``python
# tools/accelerator/scoreboard.py`` sees the same HCLI package as pytest.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hcli.physical_graph import score_physical_candidates


SCHEMA = "hawking.accelerator.scoreboard.v1"
DEFAULT_OUT = Path("receipts/headless/ACCELERATOR_SCOREBOARD.json")

# Keep the default set deliberately small.  Callers can pass any number of
# receipt paths or --glob patterns when a larger campaign view is useful.
DEFAULT_RECEIPTS = (
    "receipts/headless/FLASH_FAST_FUSED_L0_L47_V1/FAST_CHAIN_SUMMARY.json",
    "receipts/headless/FLASH_FAST_FUSED_L0_L47_V1/terminal.json",
    "receipts/headless/FLASH_FAST_TIMING_LINEAR_L0_V1/FAST_CHAIN_SUMMARY.json",
    "receipts/headless/FLASH_FAST_TIMING_LINEAR_L0_V1/group-0-0/receipt.json",
    "receipts/headless/FLASH_FUSED_TIMING_L3_SERIAL_DIRECT.json",
    "receipts/headless/FLASH_STATEFUL_COMPLETE_TOKEN_ACCEPTED.json",
    "receipts/headless/FLASH_STATEFUL_COMPLETE_SESSION_TIMING.json",
    "receipts/headless/FLASH_CHAIN_TIMING_DECOMPOSITION.json",
    "receipts/headless/HCLI_PROTECTED_ACCELERATOR_BENCHMARK_AFTER_FLASH.json",
    "receipts/headless/QWEN27_MLP_PROTECTED_AB_AFTER_FLASH.json",
    "receipts/headless/QWEN38_FUSION_PROTECTED_AB_AFTER_FLASH.json",
    "receipts/headless/APPLE_ANE_DEVICE_PROFILE.json",
    "receipts/headless/APPLE_ANE_ATLAS.json",
)


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _nested(mapping: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = mapping
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        if value is not None:
            return value
    return None


def _first_nested_value(mapping: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    """Return the first present value from a list of receipt-shaped paths."""

    for path in paths:
        value = _nested(mapping, path)
        if value is not None:
            return value
    return None


def _number(value: Any) -> Optional[int | float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_number(*values: Any) -> Optional[int | float]:
    """Return the first present numeric value, preserving measured zero."""

    for value in values:
        parsed = _number(value)
        if parsed is not None:
            return parsed
    return None


def _summary_number(value: Any) -> Optional[int | float]:
    """Read a scalar or an explicit receipt summary without manufacturing it.

    Protected HCLI receipts keep repeated observations as
    ``{"all": [...], "median": ...}``, while older receipts often emit a
    plain list.  A list is reduced to its median only when a caller has already
    selected one metric from a repeated observation set; layer timing arrays
    continue to use ``_sum_numbers`` below.
    """

    if isinstance(value, Mapping):
        for key in ("median", "p50", "value", "mean"):
            parsed = _number(value.get(key))
            if parsed is not None:
                return parsed
        for key in ("all", "samples", "values"):
            if key in value:
                return _summary_number(value[key])
        return None
    if isinstance(value, (list, tuple)):
        numbers = [_number(item) for item in value]
        numbers = [item for item in numbers if item is not None]
        return median(numbers) if numbers else None
    return _number(value)


def _first_summary(*values: Any) -> Optional[int | float]:
    for value in values:
        parsed = _summary_number(value)
        if parsed is not None:
            return parsed
    return None


def _measurement_summary(payload: Mapping[str, Any], *keys: str) -> Optional[int | float]:
    """Take the median of an explicitly named per-run measurement field.

    Warmup rows are not part of the steady-state campaign denominator.  No
    value is returned when the receipt has no matching field.
    """

    measurements = payload.get("measurements")
    if not isinstance(measurements, list):
        return None
    values: list[int | float] = []
    for row in measurements:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("phase") or "").lower() == "warmup":
            continue
        value = _first_summary(*(row.get(key) for key in keys))
        if value is not None:
            values.append(value)
    return median(values) if values else None


def _measurement_max(payload: Mapping[str, Any], *keys: str) -> Optional[int | float]:
    """Return the worst explicit per-run counter, preserving zero."""

    measurements = payload.get("measurements")
    if not isinstance(measurements, list):
        return None
    values: list[int | float] = []
    for row in measurements:
        if not isinstance(row, Mapping):
            continue
        value = _first_summary(*(row.get(key) for key in keys))
        if value is not None:
            values.append(value)
    return max(values) if values else None


def _arm_summary(payload: Mapping[str, Any], *keys: str) -> Optional[int | float]:
    """Summarize one metric across explicitly recorded diagnostic arms."""

    arms = payload.get("arms")
    if not isinstance(arms, list):
        return None
    values: list[int | float] = []
    for arm in arms:
        if not isinstance(arm, Mapping):
            continue
        live = arm.get("live") if isinstance(arm.get("live"), Mapping) else {}
        decode = live.get("decode_metrics") if isinstance(live.get("decode_metrics"), Mapping) else {}
        value = _first_summary(
            _first(arm, *keys), _first(live, *keys), _first(decode, *keys)
        )
        if value is not None:
            values.append(value)
    return median(values) if values else None


def _sum_numbers(value: Any) -> Optional[int | float]:
    if not isinstance(value, (list, tuple)):
        return _number(value)
    numbers = [_number(item) for item in value]
    numbers = [item for item in numbers if item is not None]
    return sum(numbers) if numbers else None


def _sum_layer_field(layer_rows: list[Any], *keys: str) -> Optional[int | float]:
    """Sum a timing/cost field over layer rows while preserving measured zero."""

    values = [
        _sum_numbers(_first(row, *keys))
        for row in layer_rows
        if isinstance(row, Mapping)
    ]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _latency_accounting(
    *,
    wall_ns_per_token: Optional[int | float],
    gpu_ns_per_token: Optional[int | float],
    wall_ns: Optional[int | float],
    gpu_ns: Optional[int | float],
    wall_minus_gpu_ns_per_token: Optional[int | float],
    wall_minus_gpu_ns: Optional[int | float],
    timing_phases: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe the largest *same-scope* latency component.

    This is accounting over fields already present in one receipt.  It does
    not turn a session or layer total into a token metric, and it never calls
    an unmeasured phase zero.  The wall-minus-GPU residual is derived only
    when both operands use the same scope; an explicit residual wins.
    """

    if wall_ns_per_token is not None:
        scope = "per_token"
        denominator_name = "wall_ns_per_token"
        denominator = wall_ns_per_token
        gpu = gpu_ns_per_token
        reported_residual = wall_minus_gpu_ns_per_token
    elif wall_ns is not None:
        scope = "run"
        denominator_name = "wall_ns"
        denominator = wall_ns
        gpu = gpu_ns
        reported_residual = wall_minus_gpu_ns
    else:
        scope = "unknown"
        denominator_name = None
        denominator = None
        gpu = None
        reported_residual = None

    derived_residual = None
    if denominator is not None and gpu is not None:
        derived_residual = denominator - gpu
    # A reported residual may belong to a narrower subphase than the selected
    # wall denominator.  Keep it as evidence, but use the denominator's own
    # partition for ranking whenever both operands are available.
    residual = derived_residual if derived_residual is not None else reported_residual
    if derived_residual is not None:
        residual_source = "derived_wall_minus_gpu"
    elif reported_residual is not None:
        residual_source = "measured"
    else:
        residual_source = None

    components: list[dict[str, Any]] = []
    if gpu is not None:
        components.append({
            "metric": "gpu_execution",
            "ns": gpu,
            "source": "measured",
        })
    if residual is not None:
        components.append({
            "metric": "host_or_unattributed",
            "ns": residual,
            "source": residual_source,
        })
    ranked_components = [
        component for component in components
        if isinstance(component.get("ns"), (int, float))
        and component["ns"] >= 0
    ]
    ranked_components.sort(key=lambda component: component["ns"], reverse=True)

    if denominator is not None and denominator > 0:
        for component in components:
            value = component.get("ns")
            component["fraction_of_denominator"] = (
                value / denominator
                if isinstance(value, (int, float)) and value >= 0
                else None
            )
    else:
        for component in components:
            component["fraction_of_denominator"] = None

    negative_component = any(
        isinstance(component.get("ns"), (int, float)) and component["ns"] < 0
        for component in components
    )
    if denominator is None:
        accounting_status = "UNAVAILABLE"
    elif negative_component:
        accounting_status = "INCONSISTENT"
    elif (
        reported_residual is not None
        and derived_residual is not None
        and not math.isclose(
            reported_residual,
            derived_residual,
            rel_tol=0.01,
            abs_tol=1.0,
        )
    ):
        accounting_status = "INCONSISTENT"
    elif gpu is None or residual is None:
        accounting_status = "INCOMPLETE"
    else:
        accounting_status = "ACCOUNTED"

    known_phases = [
        {"metric": name, "ns": value}
        for name, value in timing_phases.items()
        if value is not None
    ]
    known_phases.sort(key=lambda item: item["ns"], reverse=True)
    if denominator is not None and denominator > 0:
        for phase in known_phases:
            value = phase["ns"]
            phase["fraction_of_denominator"] = (
                value / denominator
                if isinstance(value, (int, float)) and value >= 0
                else None
            )
    else:
        for phase in known_phases:
            phase["fraction_of_denominator"] = None

    dominant = ranked_components[0] if ranked_components else None
    return {
        "scope": scope,
        "denominator_metric": denominator_name,
        "denominator_ns": denominator,
        "components": components,
        "accounting_status": accounting_status,
        "dominant_component": dominant["metric"] if dominant else None,
        "dominant_component_ns": dominant["ns"] if dominant else None,
        "dominant_component_fraction": (
            dominant.get("fraction_of_denominator") if dominant else None
        ),
        "reported_residual_ns": reported_residual,
        "reported_residual_consistency": (
            "not_reported"
            if reported_residual is None
            else "not_comparable"
            if derived_residual is None
            else "consistent"
            if math.isclose(
                reported_residual,
                derived_residual,
                rel_tol=0.01,
                abs_tol=1.0,
            )
            else "mismatch"
        ),
        "known_timing_phases": known_phases,
        "timing_phase_claim": (
            "Explicit source-receipt phase values; phases are not assumed additive"
        ),
    }


def _identity(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    for key in ("id", "name", "kind", "type", "schema"):
        if value.get(key) is not None:
            return value[key]
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _benchmark_class(payload: Mapping[str, Any]) -> str:
    benchmark = payload.get("benchmark")
    value = _first(payload, "benchmark_class", "experiment_class")
    if value is None and isinstance(benchmark, Mapping):
        value = _first(benchmark, "class", "benchmark_class")
    if value is None:
        bench = payload.get("bench")
        if isinstance(bench, Mapping):
            value = _first(bench, "class", "benchmark_class")
    return str(value or "UNKNOWN").upper()


def _explicit_development_phase(
    payload: Mapping[str, Any],
    timing: Mapping[str, Any],
    execution_timing: Mapping[str, Any],
    totals: Mapping[str, Any],
    name: str,
) -> Optional[int | float]:
    """Read one development phase only when the receipt names that phase."""

    return _first_number(
        _first(payload, name),
        _first(timing, name),
        _first(execution_timing, name),
        _first(totals, name),
    )


def _explicit_outcome(payload: Mapping[str, Any]) -> Optional[str]:
    """Return an explicitly recorded hypothesis/experiment outcome."""

    value = _first(
        payload,
        "hypothesis_status",
        "experiment_verdict",
        "experiment_outcome",
        "verdict",
    )
    for container_name in ("hypothesis", "experiment", "result"):
        container = payload.get(container_name)
        if value is None and isinstance(container, Mapping):
            value = _first(
                container,
                "status",
                "verdict",
                "outcome",
                "hypothesis_status",
                "experiment_verdict",
            )
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _explicit_strong_model_turns(payload: Mapping[str, Any]) -> Optional[int | float]:
    return _first_summary(
        _first(payload, "strong_model_turns"),
        _first_nested_value(
            payload,
            ("development", "strong_model_turns"),
            ("experiment", "strong_model_turns"),
        ),
    )


def normalize_receipt(path: Path, payload: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Normalize one receipt without manufacturing an absent measurement."""

    execution = payload.get("execution") if isinstance(payload.get("execution"), Mapping) else {}
    timing = payload.get("timing") if isinstance(payload.get("timing"), Mapping) else {}
    execution_timing = execution.get("timing") if isinstance(execution.get("timing"), Mapping) else {}
    bytes_block = payload.get("bytes") if isinstance(payload.get("bytes"), Mapping) else {}
    bench = payload.get("bench") if isinstance(payload.get("bench"), Mapping) else {}
    machine_snapshot = payload.get("machine_snapshot") if isinstance(payload.get("machine_snapshot"), Mapping) else {}
    totals = payload.get("totals") if isinstance(payload.get("totals"), Mapping) else {}
    layer_rows = payload.get("layers") if isinstance(payload.get("layers"), list) else []
    segment_rows = payload.get("segments") if isinstance(payload.get("segments"), list) else []
    timing_rows = layer_rows or segment_rows

    # Complete-token work is only taken from an explicit complete-token field
    # or an explicit per-token aggregate.  Oracle wall time and arbitrary
    # session totals are intentionally not relabelled as accepted-token work.
    complete_token_ns = _first_summary(
        _first(payload, "accepted_complete_token_ns", "complete_token_ns", "complete_useful_ns"),
        _first(execution, "accepted_complete_token_ns", "complete_token_ns", "complete_useful_ns"),
        _first_nested_value(payload, ("aggregate", "complete_wall_ns_per_token"),
                            ("aggregate", "complete_token_ns")),
        _measurement_summary(payload, "accepted_complete_token_ns", "complete_wall_ns_per_token", "complete_token_ns"),
    )
    wall_ns = _first_summary(
        _first(payload, "wall_ns", "elapsed_wall_ns", "complete_wall_ns"),
        _first(execution, "forward_wall_ns", "wall_ns", "elapsed_wall_ns"),
        _first_nested_value(payload, ("aggregate", "complete_wall_ns"),
                            ("totals", "elapsed_wall_ns")),
        _measurement_summary(payload, "complete_wall_ns", "wall_ns", "elapsed_wall_ns"),
        _arm_summary(payload, "complete_wall_ns", "wall_ns"),
    )
    wall_ns_per_token = _first_summary(
        _first(payload, "wall_ns_per_token", "complete_wall_ns_per_token"),
        _first(execution, "wall_ns_per_token", "complete_wall_ns_per_token"),
        _first_nested_value(payload, ("aggregate", "complete_wall_ns_per_token")),
        _measurement_summary(payload, "complete_wall_ns_per_token", "wall_ns_per_token"),
    )
    gpu_ns = _first_summary(
        _first(payload, "GPU_ns", "gpu_ns"),
        _first(execution, "forward_gpu_ns", "gpu_ns", "graph_gpu_ns"),
        _first_nested_value(payload, ("totals", "measured_gpu_ns"),
                            ("totals", "gpu_ns"), ("aggregate", "gpu_ns")),
        _measurement_summary(payload, "gpu_ns"),
        _arm_summary(payload, "GPU_ns", "gpu_ns"),
    )
    if gpu_ns is None:
        row_gpu_values = [
            _sum_numbers(_first(row, "GPU_ns", "gpu_ns", "graph_gpu_ns"))
            for row in timing_rows
            if isinstance(row, Mapping)
        ]
        row_gpu_values = [value for value in row_gpu_values if value is not None]
        if row_gpu_values:
            gpu_ns = sum(row_gpu_values)
    gpu_ns_per_token = _first_summary(
        _first(payload, "GPU_ns_per_token", "gpu_ns_per_token"),
        _first(execution, "GPU_ns_per_token", "gpu_ns_per_token"),
        _first_nested_value(payload, ("aggregate", "gpu_ns_per_token")),
        _measurement_summary(payload, "gpu_ns_per_token"),
        _arm_summary(payload, "GPU_ns_per_token", "gpu_ns_per_token"),
    )
    source_bytes = _first_summary(
        _first(payload, "source_bytes_touched", "source_payload_bytes_read", "source_weight_bytes_read"),
        _first(bytes_block, "source_bytes_touched", "source_payload_bytes_read", "source_weight_bytes_read"),
        _first(execution, "source_bytes_touched", "source_payload_bytes_read"),
        _first_nested_value(payload, ("totals", "source_payload_bytes_read"),
                            ("totals", "source_bytes_touched")),
    )
    if source_bytes is None:
        row_source_values = [
            _summary_number(_first(row, "source_bytes_touched", "source_bytes_read", "source_payload_bytes_read"))
            for row in timing_rows
            if isinstance(row, Mapping)
        ]
        row_source_values = [value for value in row_source_values if value is not None]
        if row_source_values:
            source_bytes = sum(row_source_values)

    capability = _first(payload, "capability_verified", "capability_passed")
    if capability is None and isinstance(payload.get("capability"), Mapping):
        capability = _first(payload["capability"], "verified", "passed")
    if capability is None and isinstance(payload.get("qualification"), Mapping):
        capability = _first(payload["qualification"], "capability_verified", "capability_passed")
    if not isinstance(capability, bool):
        capability = None

    fallback_count = _first_summary(
        _first(payload, "fallback_count"),
        _first(execution, "fallback_count"),
        _measurement_max(payload, "fallback_count", "fallbacks"),
    )
    if fallback_count is None:
        arm_counts = []
        arms = payload.get("arms")
        if isinstance(arms, list):
            for arm in arms:
                if not isinstance(arm, Mapping):
                    continue
                live = arm.get("live") if isinstance(arm.get("live"), Mapping) else {}
                value = _first_summary(_first(arm, "fallback_count", "fallbacks"),
                                       _first(live, "fallback_count", "fallbacks"))
                if value is not None:
                    arm_counts.append(value)
        if arm_counts:
            fallback_count = max(arm_counts)
    fallback = _first(payload, "fallback")
    if fallback is not None and not isinstance(fallback, bool):
        fallback = None

    # Keep the wall-time denominator inspectable.  These fields are copied
    # from a receipt's timing block (or summed from layer rows); the scoreboard
    # never estimates a missing phase from wall time.
    phase = {
        "root_canonicalize_ns": _first_number(
            _first(payload, "root_canonicalize_ns"), _first(timing, "root_canonicalize_ns"),
            _first(execution_timing, "root_canonicalize_ns")
        ),
        "manifest_ns": _first_number(
            _first(payload, "manifest_ns"), _first(timing, "manifest_ns"),
            _first(execution_timing, "manifest_ns")
        ),
        "config_ns": _first_number(
            _first(payload, "config_ns"), _first(timing, "config_ns"),
            _first(execution_timing, "config_ns")
        ),
        "index_context_ns": _first_number(
            _first(payload, "index_context_ns"), _first(timing, "index_context_ns"),
            _first(execution_timing, "index_context_ns")
        ),
        "input_load_ns": _first_number(
            _first(payload, "input_load_ns"), _first(timing, "input_load_ns"),
            _first(execution_timing, "input_load_ns")
        ),
        "state_read_ns": _first_number(
            _first(payload, "state_read_ns"), _first(timing, "state_read_ns"),
            _first(execution_timing, "state_read_ns")
        ),
        "cpu_readout_ns": _first_number(
            _first(payload, "cpu_readout_ns"), _first(timing, "cpu_readout_ns"),
            _first(execution_timing, "cpu_readout_ns")
        ),
        "device_upload_ns": _first_number(
            _first(payload, "device_upload_ns"), _first(timing, "device_upload_ns"),
            _first(execution_timing, "device_upload_ns")
        ),
        "source_load_ns": _first_summary(
            _first(payload, "source_load_ns"), _first(timing, "source_load_ns"),
            _first(execution_timing, "source_load_ns"),
            _first(totals, "source_load_ns"),
            _sum_layer_field(timing_rows, "source_load_ns"),
        ),
        "device_prepare_ns": _first_summary(
            _first(payload, "device_prepare_ns"), _first(timing, "device_prepare_ns"),
            _first(execution_timing, "device_prepare_ns"),
            _first(totals, "device_prepare_ns"),
            _sum_layer_field(timing_rows, "device_prepare_ns"),
        ),
        "graph_setup_ns": _first_summary(
            _first(payload, "graph_setup_ns"), _first(timing, "graph_setup_ns"),
            _first(execution_timing, "graph_setup_ns"),
            _first(totals, "graph_setup_ns"),
            _sum_layer_field(timing_rows, "graph_setup_ns"),
        ),
        "encode_ns": _first_summary(
            _first(payload, "encode_ns"), _first(timing, "encode_ns"),
            _first(execution_timing, "encode_ns", "encode_wall_ns"),
            _first(totals, "encode_ns"),
            _sum_layer_field(timing_rows, "encode_ns"),
        ),
        "command_submit_ns": _first_summary(
            _first(payload, "command_submit_ns"), _first(timing, "command_submit_ns"),
            _first(execution_timing, "command_submit_ns"), _first(totals, "command_submit_ns")
        ),
        "command_wait_ns": _first_summary(
            _first(payload, "command_wait_ns"), _first(timing, "command_wait_ns"),
            _first(execution_timing, "command_wait_ns"),
            _first(totals, "command_wait_ns"),
            _sum_layer_field(timing_rows, "command_wait_ns"),
        ),
        "gpu_execution_ns": _first_summary(
            _first(payload, "GPU_execution_ns", "gpu_execution_ns"),
            _first(timing, "GPU_execution_ns", "gpu_execution_ns", "gpu_ns"),
            _first(execution_timing, "GPU_execution_ns", "gpu_execution_ns"),
            _first(totals, "GPU_execution_ns", "gpu_execution_ns", "measured_gpu_ns"),
            _sum_layer_field(timing_rows, "GPU_execution_ns", "gpu_execution_ns", "gpu_ns"),
        ),
        "parity_ns": _first_summary(
            _first(payload, "parity_ns"), _first(timing, "parity_ns"),
            _first(execution_timing, "parity_ns"),
            _sum_layer_field(timing_rows, "parity_ns"),
        ),
        "state_write_ns": _first_summary(
            _first(payload, "state_write_ns"), _first(timing, "state_write_ns"),
            _first(execution_timing, "state_write_ns"),
            _first(totals, "state_write_ns"),
            _sum_layer_field(timing_rows, "state_write_ns"),
        ),
        "state_reload_ns": _first_summary(
            _first(payload, "state_reload_ns"), _first(timing, "state_reload_ns"),
            _first(execution_timing, "state_reload_ns"), _first(totals, "state_reload_ns"),
            _sum_layer_field(timing_rows, "state_reload_ns"),
        ),
        "receipt_write_ns": _first_summary(
            _first(payload, "receipt_write_ns"), _first(timing, "receipt_write_ns"),
            _first(execution_timing, "receipt_write_ns"),
            _first(totals, "receipt_write_ns"),
            _sum_layer_field(timing_rows, "receipt_write_ns"),
        ),
        "experiment_turnaround_ns": _first_summary(
            _first(payload, "experiment_turnaround_ns", "total_experiment_turnaround_ns"),
            _first(timing, "experiment_turnaround_ns", "total_experiment_turnaround_ns"),
            _first(execution_timing, "experiment_turnaround_ns", "total_experiment_turnaround_ns"),
            _first(totals, "experiment_turnaround_ns", "total_experiment_turnaround_ns"),
        ),
    }

    development_phases = {
        name: _explicit_development_phase(
            payload, timing, execution_timing, totals, name
        )
        for name in (
            "transform_ns",
            "compile_ns",
            "load_ns",
            "benchmark_ns",
            "verification_ns",
            "receipt_ns",
        )
    }
    # ``experiment_turnaround_ns`` is the historical spelling used by HCLI;
    # expose the document's canonical total without inventing a sum from
    # incomplete phase fields.
    development_phases["total_experiment_turnaround_ns"] = phase[
        "experiment_turnaround_ns"
    ]
    hypothesis_outcome = _explicit_outcome(payload)
    strong_model_turns = _explicit_strong_model_turns(payload)

    model = _first_nested_value(
        payload, ("model",), ("model_id",), ("resident_final", "model_id"),
        ("identity", "profile", "model_id"), ("base_identity", "model_id"),
        ("execution", "model_id"),
    )
    representation = _identity(_first_nested_value(
        payload, ("representation",), ("representation_schema",), ("nx",),
        ("resident_final", "representation"), ("identity", "profile", "representation"),
        ("base_identity", "representation"), ("execution", "representation"),
    ))
    backend = _first_nested_value(
        payload, ("backend",), ("provider",), ("device",),
        ("execution", "provider"), ("execution", "backend"), ("execution", "device"),
        ("resident_final", "backend"), ("resident_final", "provider"),
        ("identity", "profile", "backend"), ("identity", "profile", "provider"),
        ("base_identity", "backend"), ("base_identity", "provider"),
    )
    machine = _first_nested_value(
        payload, ("machine",), ("machine_genome",), ("bench", "machine"),
        ("machine_snapshot", "platform"), ("machine_snapshot", "machine"),
        ("execution", "device"), ("resident_final", "machine_genome"),
    )
    if machine is None:
        machine = _first(machine_snapshot, "platform", "machine")

    executable_id = _first_nested_value(
        payload, ("executable_id",), ("nx_id",), ("artifact_id",),
        ("binary_sha256",), ("binary_sha256_16",),
        ("resident_final", "binary_sha256_16"),
        ("identity", "profile", "binary_sha256_16"),
        ("base_identity", "binary_sha256_16"),
    )
    dispatches = _first_summary(
        _first(payload, "dispatches"),
        _first(execution, "dispatches", "total_dispatches", "layer_graph_dispatches"),
        _first_nested_value(payload, ("totals", "dispatches"), ("aggregate", "dispatches")),
        _measurement_summary(payload, "dispatches"),
        _arm_summary(payload, "dispatches"),
    )
    if dispatches is None:
        row_dispatch_values = [
            _summary_number(_first(row, "dispatches"))
            for row in timing_rows
            if isinstance(row, Mapping)
        ]
        row_dispatch_values = [value for value in row_dispatch_values if value is not None]
        if row_dispatch_values:
            dispatches = sum(row_dispatch_values)
    dispatches_per_token = _first_summary(
        _first(payload, "dispatches_per_token"),
        _first(execution, "dispatches_per_token"),
        _first_nested_value(payload, ("aggregate", "dispatches_per_token")),
        _measurement_summary(payload, "dispatches_per_token"),
        _arm_summary(payload, "dispatches_per_token"),
    )
    command_buffers = _first_summary(
        _first(payload, "command_buffers"),
        _first(execution, "command_buffers", "total_command_buffers"),
        _first_nested_value(payload, ("totals", "command_buffers"), ("aggregate", "command_buffers")),
        _measurement_summary(payload, "command_buffers"),
    )
    if command_buffers is None:
        row_command_buffer_values = [
            _summary_number(_first(row, "command_buffers"))
            for row in timing_rows
            if isinstance(row, Mapping)
        ]
        row_command_buffer_values = [value for value in row_command_buffer_values if value is not None]
        if row_command_buffer_values:
            command_buffers = sum(row_command_buffer_values)

    accepted_tps = _first_summary(
        _first(payload, "accepted_tps", "flash_tps"),
        _first(execution, "accepted_tps", "flash_tps"),
        _first_nested_value(payload, ("aggregate", "accepted_tps")),
        _measurement_summary(payload, "accepted_tps"),
    )
    complete_ebpw = _first_summary(
        _first(payload, "complete_ebpw", "complete_system_ebpw"),
        _first(execution, "complete_ebpw", "complete_system_ebpw"),
        _first_nested_value(payload, ("aggregate", "complete_ebpw"),
                            ("aggregate", "complete_system_ebpw")),
    )
    physical_ebpw = _first_summary(
        _first(payload, "physical_ebpw"),
        _first(execution, "physical_ebpw"),
        _first_nested_value(payload, ("resident_final", "physical_ebpw"),
                            ("base_identity", "physical_ebpw"),
                            ("resident_final", "representation", "physical_ebpw"),
                            ("base_identity", "representation", "physical_ebpw")),
    )
    total_nx_bytes = _first_summary(
        _first(payload, "total_nx_bytes", "nx_bytes", "total_executable_bytes"),
        _first(bytes_block, "total_nx_bytes", "nx_bytes", "total_executable_bytes"),
        _first(execution, "total_nx_bytes", "nx_bytes", "total_executable_bytes"),
    )
    source_active_bytes_per_token = _first_summary(
        _first(payload, "source_active_bytes_per_token"),
        _first(bytes_block, "source_active_bytes_per_token"),
        _first(execution, "source_active_bytes_per_token"),
    )
    active_weight_bytes_per_generated_token = _first_summary(
        _first(payload, "active_weight_bytes_per_generated_token"),
        _first(bytes_block, "active_weight_bytes_per_generated_token"),
        _first(execution, "active_weight_bytes_per_generated_token"),
        _first_nested_value(payload, ("aggregate", "active_weight_bytes_per_generated_token")),
        _measurement_summary(payload, "active_weight_bytes_per_generated_token"),
    )
    actual_read_bytes_per_token = _first_summary(
        _first(payload, "actual_read_bytes_per_token", "read_bytes_per_token"),
        _first(bytes_block, "actual_read_bytes_per_token", "read_bytes_per_token"),
        _first(execution, "actual_read_bytes_per_token", "read_bytes_per_token"),
    )
    transient_bytes_per_token = _first_summary(
        _first(payload, "transient_bytes_per_token"),
        _first(bytes_block, "transient_bytes_per_token"),
        _first(execution, "transient_bytes_per_token"),
    )
    wall_minus_gpu_ns_per_token = _first_summary(
        _first(payload, "wall_minus_gpu_ns_per_token"),
        _first(execution, "wall_minus_gpu_ns_per_token"),
        _first_nested_value(
            payload,
            ("aggregate", "wall_minus_gpu_ns_per_token"),
        ),
        _measurement_summary(payload, "wall_minus_gpu_ns_per_token"),
    )
    wall_minus_gpu_ns = _first_summary(
        _first(payload, "wall_minus_gpu_ns"),
        _first(execution, "wall_minus_gpu_ns"),
        _first_nested_value(payload, ("totals", "wall_minus_gpu_ns")),
        _measurement_summary(payload, "wall_minus_gpu_ns"),
    )
    native_wall_minus_gpu_ns = _first_summary(
        _first(payload, "native_wall_minus_gpu_ns"),
        _first(execution, "native_wall_minus_gpu_ns"),
        _first_nested_value(payload, ("aggregate", "native_wall_minus_gpu_ns")),
        _measurement_summary(payload, "native_wall_minus_gpu_ns"),
    )
    native_wall_minus_gpu_ns_per_token = _first_summary(
        _first(payload, "native_wall_minus_gpu_ns_per_token"),
        _first(execution, "native_wall_minus_gpu_ns_per_token"),
        _first_nested_value(
            payload,
            ("aggregate", "native_wall_minus_gpu_ns_per_token"),
        ),
        _measurement_summary(payload, "native_wall_minus_gpu_ns_per_token"),
    )

    row = {
        "receipt": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "receipt_sha256": _sha256(path),
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "model": model,
        "backend": backend,
        "representation": representation,
        "machine": machine,
        "benchmark_class": _benchmark_class(payload),
        "bench_state": bench.get("state") if isinstance(bench, Mapping) else None,
        "complete_token_ns": complete_token_ns,
        "accepted_complete_token_ns": _first_summary(_first(payload, "accepted_complete_token_ns")),
        "accepted_tps": accepted_tps,
        "complete_ebpw": complete_ebpw,
        "physical_ebpw": physical_ebpw,
        "wall_ns_per_token": wall_ns_per_token,
        "GPU_ns": gpu_ns,
        "gpu_ns_per_token": gpu_ns_per_token,
        "wall_ns": wall_ns,
        "source_bytes_touched": source_bytes,
        "total_nx_bytes": total_nx_bytes,
        "nx_bytes_touched": _first_summary(
            _first(payload, "nx_bytes_touched", "executable_bytes_touched"),
            _first(bytes_block, "nx_bytes_touched", "executable_bytes_touched"),
            _first(execution, "nx_bytes_touched", "executable_bytes_touched"),
        ),
        "active_bytes_per_token": _first_summary(
            _first(payload, "active_bytes_per_token"),
            _first(bytes_block, "active_bytes_per_token"),
            _first(execution, "active_bytes_per_token"),
            _first_nested_value(payload, ("aggregate", "active_bytes_per_token")),
            _measurement_summary(payload, "active_bytes_per_token"),
        ),
        "active_weight_bytes_per_generated_token": active_weight_bytes_per_generated_token,
        "active_bytes_scope": _first(
            payload, "active_bytes_scope"
        ) or _first(bytes_block, "active_bytes_scope") or _first(execution, "active_bytes_scope"),
        "source_active_bytes_per_token": source_active_bytes_per_token,
        "actual_read_bytes_per_token": actual_read_bytes_per_token,
        "transient_bytes_per_token": transient_bytes_per_token,
        "resident_bytes": _first_summary(
            _first(payload, "resident_bytes"), _first(bytes_block, "resident_bytes"),
            _first(execution, "resident_bytes", "resident_weight_bytes"),
            _first(payload, "resident_weight_bytes"),
            _first_nested_value(payload, ("resident_final", "hot_bytes")),
        ),
        "resident_weight_bytes": _first_summary(
            _first(payload, "resident_weight_bytes"),
            _first(bytes_block, "resident_weight_bytes"),
            _first(execution, "resident_weight_bytes"),
            _first_nested_value(payload, ("aggregate", "resident_weight_bytes")),
            _measurement_summary(payload, "resident_weight_bytes"),
        ),
        "workspace_resident_bytes": _first_summary(
            _first(payload, "workspace_resident_bytes"),
            _first(bytes_block, "workspace_resident_bytes"),
            _first(execution, "workspace_resident_bytes"),
            _first_nested_value(payload, ("aggregate", "workspace_resident_bytes")),
            _measurement_summary(payload, "workspace_resident_bytes"),
        ),
        "dispatches": dispatches,
        "dispatches_per_token": dispatches_per_token,
        "command_buffers": command_buffers,
        "synchronization_events": _first_summary(
            _first(payload, "synchronization_events", "sync_events"),
            _first(execution, "synchronization_count"),
            _first(totals, "synchronization_events", "sync_events"),
        ),
        "synchronization_ns": _first_summary(
            _first(payload, "synchronization_ns", "sync_ns"),
            _first(timing, "synchronization_ns", "sync_ns"),
            _first(execution_timing, "synchronization_ns", "sync_ns"),
            _first(totals, "synchronization_ns", "sync_ns"),
        ),
        "host_roundtrips": _first_summary(
            _first(payload, "host_roundtrips", "host_activation_roundtrips"),
            _first(execution, "host_roundtrips", "host_activation_roundtrips"),
        ),
        "cold_load_ns": _first_summary(
            _first(payload, "cold_load_ns"), _first(timing, "cold_load_ns"),
            _first(execution_timing, "cold_load_ns"), _first(totals, "cold_load_ns"),
        ),
        "warm_start_ns": _first_summary(
            _first(payload, "warm_start_ns"), _first(timing, "warm_start_ns"),
            _first(execution_timing, "warm_start_ns"), _first(totals, "warm_start_ns"),
        ),
        "experiment_turnaround_ns": phase["experiment_turnaround_ns"],
        "total_experiment_turnaround_ns": development_phases[
            "total_experiment_turnaround_ns"
        ],
        "development_phases": development_phases,
        "hypothesis_outcome": hypothesis_outcome,
        "strong_model_turns": strong_model_turns,
        "wall_minus_gpu_ns_per_token": wall_minus_gpu_ns_per_token,
        "wall_minus_gpu_ns": wall_minus_gpu_ns,
        "native_wall_minus_gpu_ns": native_wall_minus_gpu_ns,
        "native_wall_minus_gpu_ns_per_token": native_wall_minus_gpu_ns_per_token,
        "capability_verified": capability,
        "fallback_count": fallback_count,
        "fallback": fallback,
        "claim_boundary": payload.get("claim_boundary"),
        "timing_phases": phase,
        "executable_id": executable_id,
        "evidence_mode": _first(payload, "evidence_mode") or _benchmark_class(payload),
    }
    row["latency_accounting"] = _latency_accounting(
        wall_ns_per_token=wall_ns_per_token,
        gpu_ns_per_token=gpu_ns_per_token,
        wall_ns=wall_ns,
        gpu_ns=gpu_ns,
        wall_minus_gpu_ns_per_token=wall_minus_gpu_ns_per_token,
        wall_minus_gpu_ns=wall_minus_gpu_ns,
        timing_phases=phase,
    )
    # A nonzero fallback count is enough to make the row ineligible, but keep
    # the explicit bool visible when a receipt provided only the count.
    row["fallback_observed"] = bool(fallback) or bool(fallback_count and fallback_count > 0)
    return row


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return true when left is no worse on all metrics shared with right."""

    metrics = (
        "complete_ebpw", "complete_token_ns", "wall_ns_per_token", "gpu_ns_per_token", "wall_ns",
        "resident_bytes", "active_bytes_per_token", "source_bytes_touched",
        "dispatches_per_token", "dispatches", "synchronization_ns", "host_roundtrips",
    )
    shared = [(left.get(key), right.get(key)) for key in metrics
              if isinstance(left.get(key), (int, float)) and isinstance(right.get(key), (int, float))]
    return bool(shared) and all(a <= b for a, b in shared) and any(a < b for a, b in shared)


_QUALIFIED_CLASSES = {"PROTECTED_ABSOLUTE", "QUALIFIED_PROTECTED"}
_FAILED_OUTCOMES = {
    "FAIL",
    "FAILED",
    "FALSIFIED",
    "REJECT",
    "REJECTED",
    "NO_IMPROVEMENT",
}


def _is_qualified_row(row: Mapping[str, Any]) -> bool:
    """Mirror the receipt-level protected promotion gate conservatively."""

    complete_ns = row.get("complete_token_ns")
    return bool(
        row.get("benchmark_class") in _QUALIFIED_CLASSES
        and isinstance(complete_ns, (int, float))
        and complete_ns > 0
        and row.get("capability_verified") is True
        and row.get("fallback_count") == 0
    )


def _development_productivity(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive development-loop rates only from complete explicit coverage."""

    total_rows = len(rows)
    turnaround_values = [
        row["total_experiment_turnaround_ns"]
        for row in rows
        if isinstance(row.get("total_experiment_turnaround_ns"), (int, float))
        and row["total_experiment_turnaround_ns"] >= 0
    ]
    total_turnaround_ns = sum(turnaround_values) if turnaround_values else None
    turnaround_complete = bool(total_rows and len(turnaround_values) == total_rows)

    qualified_count = sum(1 for row in rows if _is_qualified_row(row))
    qualified_rates = None
    if turnaround_complete and total_turnaround_ns and qualified_count:
        qualified_rates = qualified_count * 3_600_000_000_000 / total_turnaround_ns

    outcome_values = [
        row["hypothesis_outcome"]
        for row in rows
        if isinstance(row.get("hypothesis_outcome"), str)
        and row["hypothesis_outcome"]
    ]
    failed_count = sum(value in _FAILED_OUTCOMES for value in outcome_values)
    outcome_complete = bool(total_rows and len(outcome_values) == total_rows)
    failed_rates = None
    if outcome_complete and turnaround_complete and total_turnaround_ns:
        failed_rates = failed_count * 3_600_000_000_000 / total_turnaround_ns

    strong_turn_values = [
        row["strong_model_turns"]
        for row in rows
        if isinstance(row.get("strong_model_turns"), (int, float))
        and row["strong_model_turns"] >= 0
    ]
    strong_turns = sum(strong_turn_values) if strong_turn_values else None
    qualified_with_turns = sum(
        1
        for row in rows
        if _is_qualified_row(row) and isinstance(row.get("strong_model_turns"), (int, float))
    )
    strong_per_qualified = None
    if qualified_count and qualified_with_turns == qualified_count and strong_turns is not None:
        strong_per_qualified = strong_turns / qualified_count

    return {
        "rows": total_rows,
        "qualified_experiment_count": qualified_count,
        "failed_hypothesis_count": failed_count if outcome_values else None,
        "strong_model_turns": strong_turns,
        "total_experiment_turnaround_ns": total_turnaround_ns,
        "turnaround_coverage": {
            "reported_rows": len(turnaround_values),
            "total_rows": total_rows,
            "complete": turnaround_complete,
        },
        "hypothesis_outcome_coverage": {
            "reported_rows": len(outcome_values),
            "total_rows": total_rows,
            "complete": outcome_complete,
        },
        "strong_model_turn_coverage": {
            "reported_rows": len(strong_turn_values),
            "total_rows": total_rows,
            "qualified_rows_with_turns": qualified_with_turns,
        },
        "qualified_experiments_per_hour": qualified_rates,
        "failed_hypotheses_per_hour": failed_rates,
        "strong_model_turns_per_qualified_experiment": strong_per_qualified,
        "rate_claim_boundary": (
            "Rates are emitted only when the required receipt fields cover the full input view; "
            "missing phase/outcome fields are not treated as zero."
        ),
    }


def build_scoreboard(paths: Iterable[Path], *, root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append({"receipt": str(path), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if not isinstance(payload, Mapping):
            skipped.append({"receipt": str(path), "reason": "top-level JSON is not an object"})
            continue
        rows.append(normalize_receipt(path, payload, root=root))

    frontier_ids = [
        row["receipt"] for row in rows
        if not any(other is not row and _dominates(other, row) for other in rows)
    ]
    candidates = []
    for row in rows:
        candidate = dict(row)
        candidate["id"] = row["receipt"]
        candidate["complete_useful_ns"] = row["complete_token_ns"]
        candidate["fallback_count"] = row["fallback_count"]
        candidates.append(candidate)
    plan_score = score_physical_candidates(candidates)
    return {
        "schema": SCHEMA,
        "status": "DERIVED_RECEIPT_VIEW",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # The view repeats timing-shaped fields from its inputs, so the corpus
        # requires an explicit machine-state block even though this file does
        # not perform a benchmark of its own.  UNKNOWN is intentional: a
        # derived summary cannot manufacture quiescence.
        "bench": {
            "state": "UNKNOWN",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recorded_by": "tools/accelerator/scoreboard.py derived view",
            "machine": "Apple M3 Ultra (derived from source receipts)",
            "quiescence": None,
            "rule": "S032 §3 -- if quiescence is unknown the state is UNKNOWN, not quiet",
            "provenance": "No benchmark executed by the scoreboard; source receipt states remain authoritative",
        },
        "objective": "identify measured complete useful work and physical cost denominators; never infer missing metrics",
        "rows": rows,
        "frontier_receipts": frontier_ids,
        "physical_plan_score": plan_score,
        "development_productivity": _development_productivity(rows),
        "skipped": skipped,
        "claim_boundary": "Derived from source receipts; this view adds no physical measurement and cannot promote an unqualified model.",
    }


def _expand_inputs(root: Path, receipts: list[str], globs: list[str]) -> list[Path]:
    values = receipts or list(DEFAULT_RECEIPTS)
    paths = [root / value if not Path(value).is_absolute() else Path(value) for value in values]
    for pattern in globs:
        paths.extend(sorted(root.glob(pattern)))
    return [path for path in paths if path.is_file()]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="*", help="receipt paths, relative to --root")
    parser.add_argument("--glob", action="append", default=[], help="additional root-relative glob")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    body = build_scoreboard(_expand_inputs(root, args.receipts, args.glob), root=root)
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2) + "\n")
    print(json.dumps({"out": str(out), "rows": len(body["rows"]), "frontier": len(body["frontier_receipts"]), "skipped": len(body["skipped"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "build_scoreboard", "normalize_receipt"]
