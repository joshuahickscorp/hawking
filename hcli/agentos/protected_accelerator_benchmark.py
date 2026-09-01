"""Run a provider-neutral protected resident benchmark.

The current sealed Hawking resident is one profile that can use this command,
but the benchmark contract is deliberately model-neutral.  A profile selects
the resident and a provider may expose the optional native metrics envelope;
the command records missing metrics explicitly instead of manufacturing them.

The benchmark owns one bounded resident window.  It waits for an enumerated
quiet machine, takes one persistent resident through warmup and measured
requests, stops it before the closing quiescence sample, and emits a receipt
that separates capability/execution evidence from promotion evidence.
"""
from __future__ import annotations

import fcntl
import json
import os
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from hcli.agentos.accelerator_regression import (
    _bench,
    _profile_path,
    _quiescence,
    _repo_root,
    _safe,
)
from hcli.agentos.benchmark_boundary import classify_window
from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.protected_accelerator_benchmark.v1"
DEFAULT_PROMPT = "Return exactly: HAWKING_OK"
DEFAULT_WARMUP_REQUESTS = 1
DEFAULT_MEASURE_REQUESTS = 5
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_READY_TIMEOUT_S = 6 * 3600.0
DEFAULT_INTERVAL_S = 30.0
DEFAULT_TIMEOUT_S = 180.0
LOCK_NAME = "protected-accelerator-bench.lock"


def _lock_path(repo: Path) -> Path:
    path = repo / ".hcli" / "locks" / LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _try_lock(repo: Path) -> Optional[Any]:
    handle = _lock_path(repo).open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def _release_lock(handle: Optional[Any]) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        handle.close()
    except OSError:
        pass


def _write(report: Mapping[str, Any], destination: Path) -> None:
    atomic_write_json(destination, dict(report))


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _integer(value: Any) -> Optional[int]:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _distribution(values: Iterable[Any]) -> Dict[str, Any]:
    numbers = sorted(number for number in (_number(value) for value in values) if number is not None)
    if not numbers:
        return {"n": 0, "min": None, "median": None, "max": None, "spread_pct": None, "all": []}
    median = statistics.median(numbers)
    spread = None if median == 0 else round((numbers[-1] - numbers[0]) / median * 100.0, 3)
    return {
        "n": len(numbers),
        "min": numbers[0],
        "median": median,
        "max": numbers[-1],
        "spread_pct": spread,
        "all": numbers,
    }


def _metric_map(raw: Any) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    value = raw.get("native_metrics")
    return value if isinstance(value, Mapping) else {}


def _request_record(
    raw: Any,
    *,
    elapsed_ns: int,
    index: int,
    phase: str,
) -> Dict[str, Any]:
    """Normalize one OpenAI-shaped provider response without model assumptions."""
    body = raw if isinstance(raw, Mapping) else {}
    hawking = body.get("hawking") if isinstance(body.get("hawking"), Mapping) else {}
    metrics = _metric_map(hawking)
    health = hawking.get("resident_health") if isinstance(hawking.get("resident_health"), Mapping) else {}
    generated = _integer(hawking.get("generated_tokens"))
    token_ids = hawking.get("new_token_ids")
    if isinstance(token_ids, list):
        token_ids = list(token_ids)
        if generated is None:
            generated = len(token_ids)
    else:
        token_ids = None

    wall_per = elapsed_ns / generated if generated and generated > 0 else None
    gpu_per = _number(metrics.get("gpu_ns_per_generated_token"))
    gpu_total = _number(metrics.get("gpu_ns"))
    if gpu_per is None and gpu_total is not None and generated and generated > 0:
        gpu_per = gpu_total / generated
        gpu_source = "derived_from_native_gpu_ns"
    elif gpu_per is not None:
        gpu_source = "native_gpu_ns_per_generated_token"
    else:
        gpu_source = None

    # Native ``wall_minus_gpu_ns`` is a generation-subphase residual.  The
    # complete wall denominator below includes the provider request round
    # trip, so it must not be placed directly in the complete per-token
    # accounting field.  Preserve both scopes and derive the complete residual
    # only from the complete wall and GPU operands.
    native_wall_minus = _number(metrics.get("wall_minus_gpu_ns"))
    native_wall_minus_per = _number(metrics.get("wall_minus_gpu_ns_per_generated_token"))
    native_wall_minus_source = (
        "native_wall_minus_gpu_ns_per_generated_token"
        if native_wall_minus_per is not None
        else None
    )
    if native_wall_minus_per is None and native_wall_minus is not None and generated and generated > 0:
        native_wall_minus_per = native_wall_minus / generated
        native_wall_minus_source = "derived_from_native_wall_minus_gpu_ns"

    wall_minus = None
    wall_minus_source = None
    if wall_per is not None and gpu_per is not None:
        wall_minus = wall_per - gpu_per
        wall_minus_source = "derived_complete_wall_minus_gpu"

    dispatches = _number(metrics.get("dispatches"))
    dispatches_per = _number(metrics.get("dispatches_per_generated_token"))
    if dispatches_per is None and dispatches is not None and generated and generated > 0:
        dispatches_per = dispatches / generated
        dispatch_source = "derived_from_native_dispatches"
    elif dispatches_per is not None:
        dispatch_source = "native_dispatches_per_generated_token"
    else:
        dispatch_source = None

    active_bytes = _number(metrics.get("active_bytes_per_token"))
    active_weight_bytes = _number(metrics.get("active_weight_bytes_per_generated_token"))
    if active_bytes is None and active_weight_bytes is not None:
        active_bytes = active_weight_bytes
        active_bytes_source = "native_active_weight_bytes_per_generated_token"
    elif active_bytes is not None:
        active_bytes_source = "native_active_bytes_per_token"
    else:
        active_bytes_source = None
    resident_weight_bytes = _number(metrics.get("resident_weight_bytes"))
    workspace_resident_bytes = _number(metrics.get("workspace_resident_bytes"))
    actual_read_bytes = _number(metrics.get("actual_read_bytes_per_token"))
    transient_bytes = _number(metrics.get("transient_bytes_per_token"))

    capability = metrics.get("capability")
    capability = _safe(capability) if capability is not None else None
    ids_match = token_ids is None or generated == len(token_ids)
    fallbacks = _integer(hawking.get("fallbacks"))
    capability_sanity = {
        "status": "PASS" if generated is not None and generated > 0 and ids_match and fallbacks == 0 else "FAIL",
        "generated_tokens_positive": generated is not None and generated > 0,
        "token_id_count_matches": ids_match,
        "zero_fallbacks": fallbacks == 0,
        "provider_capability_envelope": capability,
        "provider_capability_envelope_declared": isinstance(capability, Mapping),
    }
    return {
        "index": index,
        "phase": phase,
        "ok": bool(body) and not body.get("error"),
        "generated_tokens": generated,
        "new_token_ids": token_ids,
        "fallbacks": fallbacks,
        "complete_wall_ns": int(elapsed_ns),
        "complete_wall_ns_per_token": wall_per,
        "gpu_ns": gpu_total,
        "gpu_ns_per_token": gpu_per,
        "gpu_metric_source": gpu_source,
        "wall_minus_gpu_ns_per_token": wall_minus,
        "wall_minus_gpu_metric_source": wall_minus_source,
        "native_wall_minus_gpu_ns": native_wall_minus,
        "native_wall_minus_gpu_ns_per_token": native_wall_minus_per,
        "native_wall_minus_gpu_metric_source": native_wall_minus_source,
        "dispatches": dispatches,
        "dispatches_per_token": dispatches_per,
        "dispatch_metric_source": dispatch_source,
        "active_bytes_per_token": active_bytes,
        "active_bytes_source": active_bytes_source,
        "active_bytes_scope": metrics.get("active_bytes_scope"),
        "active_weight_bytes_per_generated_token": active_weight_bytes,
        "resident_weight_bytes": resident_weight_bytes,
        "workspace_resident_bytes": workspace_resident_bytes,
        "actual_read_bytes_per_token": actual_read_bytes,
        "transient_bytes_per_token": transient_bytes,
        "prompt_tokens": _integer(hawking.get("prompt_tokens")),
        "prefill_steps": _integer(metrics.get("prefill_steps")),
        "decode_steps": _integer(metrics.get("decode_steps")),
        "prefill": _safe(metrics.get("prefill")) if metrics.get("prefill") is not None else None,
        "decode": _safe(metrics.get("decode")) if metrics.get("decode") is not None else None,
        "kernel_genome": _safe(metrics.get("kernel_genome")) if metrics.get("kernel_genome") is not None else None,
        "native_metrics": _safe(metrics),
        "resident_health": _safe(health),
        "capability_sanity": capability_sanity,
        "error": _safe(body.get("error")) if body.get("error") is not None else None,
    }


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _aggregate(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    records = [row for row in rows if isinstance(row, Mapping)]
    scalar_fields = (
        "generated_tokens",
        "complete_wall_ns",
        "complete_wall_ns_per_token",
        "gpu_ns",
        "gpu_ns_per_token",
        "wall_minus_gpu_ns_per_token",
        "native_wall_minus_gpu_ns",
        "native_wall_minus_gpu_ns_per_token",
        "dispatches",
        "dispatches_per_token",
        "active_bytes_per_token",
        "active_weight_bytes_per_generated_token",
        "resident_weight_bytes",
        "workspace_resident_bytes",
        "actual_read_bytes_per_token",
        "transient_bytes_per_token",
        "prefill_steps",
        "decode_steps",
    )
    result: Dict[str, Any] = {
        field: _distribution(row.get(field) for row in records)
        for field in scalar_fields
    }
    result["prefill"] = [
        _safe(row.get("prefill"))
        for row in records
        if row.get("prefill") is not None
    ]
    result["decode"] = [
        _safe(row.get("decode"))
        for row in records
        if row.get("decode") is not None
    ]
    result["kernel_genomes"] = [
        _safe(row.get("kernel_genome"))
        for row in records
        if row.get("kernel_genome") is not None
    ]
    result["kernel_genome_exact_and_stable"] = bool(result["kernel_genomes"]) and len(
        {_stable_json(row) for row in result["kernel_genomes"]}
    ) == 1
    result["generated_token_ids"] = [
        _safe(row.get("new_token_ids"))
        for row in records
        if isinstance(row.get("new_token_ids"), list)
    ]
    result["generated_token_ids_exact_and_stable"] = bool(result["generated_token_ids"]) and len(
        {_stable_json(row) for row in result["generated_token_ids"]}
    ) == 1
    return result


def _required_metrics(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    records = [row for row in rows if isinstance(row, Mapping)]
    fields = {
        "complete_wall_ns_per_token": all(row.get("complete_wall_ns_per_token") is not None for row in records),
        "gpu_ns_per_token": all(row.get("gpu_ns_per_token") is not None for row in records),
        "wall_minus_gpu_ns_per_token": all(row.get("wall_minus_gpu_ns_per_token") is not None for row in records),
        "dispatches_per_token": all(row.get("dispatches_per_token") is not None for row in records),
        "prefill": all(row.get("prefill") is not None for row in records),
        "decode": all(row.get("decode") is not None for row in records),
        "kernel_genome": all(row.get("kernel_genome") is not None for row in records),
    }
    return {
        "all_required_metrics_present": bool(records) and all(fields.values()),
        "fields": fields,
        "optional_physical_fields": {
            field: any(row.get(field) is not None for row in records)
            for field in (
                "active_bytes_per_token",
                "resident_weight_bytes",
                "workspace_resident_bytes",
                "actual_read_bytes_per_token",
                "transient_bytes_per_token",
            )
        },
        "missing_for_generic_provider_is_explicit": True,
    }


def _health_summary(
    ready: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    final: Mapping[str, Any],
    restarts: int,
) -> Dict[str, Any]:
    records = [row for row in rows if isinstance(row, Mapping)]
    ready_health = ready.get("resident_health")
    ready_health = ready_health if isinstance(ready_health, Mapping) else {}
    ready_pid = ready_health.get("pid", ready.get("pid"))
    pids = {
        row.get("resident_health", {}).get("pid")
        for row in records
        if isinstance(row.get("resident_health"), Mapping)
        and row.get("resident_health", {}).get("pid") is not None
    }
    declared_opens = [
        row.get("resident_health", {}).get("model_open_count")
        for row in records
        if isinstance(row.get("resident_health"), Mapping)
        and row.get("resident_health", {}).get("model_open_count") is not None
    ]
    declared_uploads = [
        row.get("resident_health", {}).get("weight_upload_count")
        for row in records
        if isinstance(row.get("resident_health"), Mapping)
        and row.get("resident_health", {}).get("weight_upload_count") is not None
    ]
    return {
        "ready": _safe(ready),
        "final": _safe(final),
        "request_pids": sorted(pids),
        "ready_pid": ready_pid,
        "one_pid_reused": len(pids) == 1 and ready_pid in pids,
        "model_open_count_declared_once": bool(declared_opens) and all(value == 1 for value in declared_opens),
        "weight_upload_count_declared_once": bool(declared_uploads) and all(value == 1 for value in declared_uploads),
        "connector_restart_count": int(restarts),
        "no_connector_restart": int(restarts) == 0,
        "health_fields_are_optional_provider_telemetry": True,
    }


def run_protected_accelerator_benchmark(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    profile: Optional[str | os.PathLike[str]] = None,
    resident_binary: Optional[str | os.PathLike[str]] = None,
    fusion_env_overrides: Optional[Mapping[str, Any]] = None,
    prompt: str = DEFAULT_PROMPT,
    warmup_requests: int = DEFAULT_WARMUP_REQUESTS,
    measure_requests: int = DEFAULT_MEASURE_REQUESTS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S,
    interval_s: float = DEFAULT_INTERVAL_S,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Measure one resident in one protected window, without promoting it."""
    repo = _repo_root(repo_root)
    profile_path = _profile_path(repo, profile)
    destination = Path(emit).expanduser() if emit else repo / "receipts" / "headless" / "HCLI_PROTECTED_ACCELERATOR_BENCHMARK.json"
    if not destination.is_absolute():
        destination = repo / destination
    warmup = max(0, min(8, int(warmup_requests)))
    measured = max(1, min(32, int(measure_requests)))
    token_limit = max(1, min(512, int(max_new_tokens)))
    ready_timeout = max(0.1, float(ready_timeout_s))
    interval = max(0.1, min(60.0, float(interval_s)))
    request_timeout = max(0.1, float(timeout_s))
    requested_fusion_overrides = {
        str(key): str(value)
        for key, value in dict(fusion_env_overrides or {}).items()
    }
    started = time.time()
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "status": "WAITING_FOR_QUIESCENCE",
        "qualification": False,
        "NOT_FOR_PROMOTION": True,
        "promotion_allowed": False,
        "repo_root": str(repo),
        "profile_path": str(profile_path),
        "resident_binary_override": str(Path(resident_binary).expanduser().resolve()) if resident_binary else None,
        "started_at": started,
        "ready_timeout_s": ready_timeout,
        "interval_s": interval,
        "request_timeout_s": request_timeout,
        "request_contract": {
            "prompt": prompt,
            "warmup_requests": warmup,
            "measure_requests": measured,
            "max_new_tokens": token_limit,
            "same_prompt_for_all_requests": True,
            "thinking_disabled": True,
        },
        "experiment_contract": {
            "one_experiment_at_a_time": True,
            "one_persistent_resident": True,
            "profile_and_source_mutation": False,
            "machine_quiescence_before_and_after": True,
            "complete_wall_includes_request_round_trip": True,
            "gpu_metric_must_be_provider_declared_or_explicitly_derived": True,
            "complete_wall_minus_gpu_uses_complete_wall_scope": True,
            "native_subphase_residual_recorded_separately": True,
            "missing_metrics_never_become_zero": True,
            "promotion_requires_separate_capability_and_quality_gate": True,
            "fusion_env_overrides_are_child_only": True,
        },
        "fusion_env_overrides": dict(requested_fusion_overrides),
        "machine": {"platform": platform.platform(), "architecture": platform.machine()},
        "readiness_polls": [],
        "warmup": [],
        "measurements": [],
        "errors": [],
    }
    lock_handle: Optional[Any] = None
    connector: Any = None
    before: Optional[Mapping[str, Any]] = None
    ready_identity: Dict[str, Any] = {}
    final_identity: Dict[str, Any] = {}
    try:
        lock_deadline = time.time() + ready_timeout
        while time.time() < lock_deadline:
            lock_handle = _try_lock(repo)
            if lock_handle is not None:
                break
            poll = _quiescence()
            report["readiness_polls"].append({"kind": "lock", "sample": _safe(poll), "lock_busy": True})
            _write(report, destination)
            time.sleep(min(interval, max(0.1, lock_deadline - time.time())))
        if lock_handle is None:
            report["status"] = "WAITING_FOR_LOCK"
            report["error"] = {"type": "LockTimeout", "message": f"exclusive benchmark lock remained busy for {ready_timeout:.1f}s"}
            return report

        quiet_deadline = time.time() + ready_timeout
        while time.time() < quiet_deadline:
            sample = _quiescence()
            report["readiness_polls"].append({"kind": "machine", "sample": _safe(sample), "quiet": sample.get("quiet") is True})
            _write(report, destination)
            if sample.get("quiet") is True:
                before = sample
                break
            time.sleep(min(interval, max(0.1, quiet_deadline - time.time())))
        if before is None:
            report["status"] = "WAITING_FOR_QUIESCENCE"
            report["error"] = {"type": "QuiescenceTimeout", "message": f"machine did not become quiet for {ready_timeout:.1f}s"}
            return report

        from hcli.hawking_native import HawkingNativeConfig, HawkingNativeConnector

        config = HawkingNativeConfig.from_file(str(profile_path))
        if resident_binary:
            config = replace(
                config,
                mode="resident",
                resident_binary=str(Path(resident_binary).expanduser().resolve()),
            )
        if requested_fusion_overrides:
            effective_fusion_env = dict(config.fusion_env)
            effective_fusion_env.update(requested_fusion_overrides)
            # A deliberate one-control experiment is allowed to change the
            # profile's required value in the child only.  The profile file,
            # source specimen, and sealed default remain untouched.
            config = replace(
                config,
                fusion_env=effective_fusion_env,
                require_fusion_env=False,
            )
        # Trace is a child-environment request, not a profile/source mutation.
        runtime_env = dict(getattr(config, "runtime_env", {}) or {})
        runtime_env.setdefault("HAWKING_TRACE_DISPATCH", "1")
        config = replace(config, runtime_env=runtime_env)
        config.validate()
        if config.effective_mode() != "resident":
            raise ValueError("protected accelerator benchmark requires a resident profile or --resident-binary")
        report["identity"] = {
            "profile": _safe(config.identity()),
            "child_runtime_env_overrides": {"HAWKING_TRACE_DISPATCH": runtime_env.get("HAWKING_TRACE_DISPATCH")},
            "child_fusion_env_overrides": dict(requested_fusion_overrides),
        }
        connector = HawkingNativeConnector(config)
        connector.start(timeout=request_timeout)
        ready_identity = connector.identity()
        report["resident_ready"] = _safe(ready_identity)
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": token_limit,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for index in range(warmup):
            started_ns = time.perf_counter_ns()
            raw = connector.complete_payload(payload, timeout=request_timeout)
            elapsed_ns = time.perf_counter_ns() - started_ns
            report["warmup"].append(_request_record(raw, elapsed_ns=elapsed_ns, index=index + 1, phase="warmup"))
        for index in range(measured):
            started_ns = time.perf_counter_ns()
            try:
                raw = connector.complete_payload(payload, timeout=request_timeout)
            except Exception as exc:  # keep prior measured rows durable
                report["errors"].append({"phase": "measure", "index": index + 1, "type": type(exc).__name__, "message": str(exc)[:2000]})
                break
            elapsed_ns = time.perf_counter_ns() - started_ns
            report["measurements"].append(_request_record(raw, elapsed_ns=elapsed_ns, index=index + 1, phase="measure"))
        final_identity = connector.identity()
        report["resident_final"] = _safe(final_identity)
        report["connector_restart_count"] = int(getattr(connector, "restart_count", 0))
    except Exception as exc:  # noqa: BLE001 - persist the exact measurement boundary
        report["errors"].append({"phase": "setup", "type": type(exc).__name__, "message": str(exc)[:2000]})
    finally:
        if connector is not None:
            try:
                connector.stop()
            except Exception as exc:  # noqa: BLE001 - preserve closing failure
                report["errors"].append({"phase": "stop", "type": type(exc).__name__, "message": str(exc)[:1200]})
        after = _quiescence()
        report["machine_after"] = _safe(after)
        bench = _bench(
            dict(before) if isinstance(before, Mapping) else None,
            after,
            "Protected persistent-resident complete-token measurement; the resident is stopped before the closing sample.",
        )
        report["bench"] = bench
        report.update(classify_window(
            dict(before) if isinstance(before, Mapping) else None,
            after,
            bench,
            qualification=False,
            not_for_promotion=True,
        ))
        report["aggregate"] = _aggregate(report["measurements"])
        report["required_metrics"] = _required_metrics(report["measurements"])
        report["health"] = _health_summary(
            ready_identity,
            report["measurements"],
            final_identity,
            int(report.get("connector_restart_count", 0)),
        )
        fallbacks = [row.get("fallbacks") for row in report["measurements"]]
        report["checks"] = {
            "profile_or_resident_started": bool(ready_identity.get("resident_health", {}).get("pid")) if isinstance(ready_identity.get("resident_health"), Mapping) else bool(ready_identity.get("pid")),
            "all_requested_measurements_completed": len(report["measurements"]) == measured,
            "all_measurements_ok": bool(report["measurements"]) and all(row.get("ok") is True for row in report["measurements"]),
            "zero_fallbacks": bool(fallbacks) and all(value == 0 for value in fallbacks),
            "capability_sanity": bool(report["measurements"]) and all(row.get("capability_sanity", {}).get("status") == "PASS" for row in report["measurements"]),
            "required_physical_metrics_present": report["required_metrics"]["all_required_metrics_present"],
            "exact_kernel_genome_recorded": report["required_metrics"]["fields"]["kernel_genome"],
            "one_persistent_pid_when_declared": report["health"]["one_pid_reused"] or not report["health"]["request_pids"],
            "no_connector_restart": report["health"]["no_connector_restart"],
            "no_unqualified_promotion": report["NOT_FOR_PROMOTION"] is True and report["promotion_allowed"] is False,
        }
        run_valid = all(report["checks"].values())
        report["measurement_verdict"] = "ACCEPT" if run_valid and report.get("protected_window") is True else "INCONCLUSIVE"
        report["qualification"] = bool(run_valid and report.get("protected_window") is True)
        waiting_boundary = (
            not report["measurements"]
            and not report["errors"]
            and isinstance(report.get("error"), Mapping)
            and report["error"].get("type") in {"LockTimeout", "QuiescenceTimeout"}
        )
        if not waiting_boundary:
            report["status"] = "PASSED" if report["measurements"] and not report["errors"] else "FAILED"
        report["claim_boundary"] = (
            "This receipt is a protected provider-neutral execution baseline when all "
            "physical metrics and quiescence checks pass. It never promotes a "
            "resident, representation, or model; a separate capability/quality "
            "verifier must accept any candidate."
        )
        report["finished_at"] = time.time()
        report["elapsed_s"] = round(report["finished_at"] - started, 3)
        report["receipt_path"] = str(destination.resolve())
        _release_lock(lock_handle)
        atomic_write_json(destination, report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--profile")
    parser.add_argument("--resident-binary")
    parser.add_argument(
        "--fusion-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="child-only fusion environment override; repeat for multiple controls",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--warmup-requests", type=int, default=DEFAULT_WARMUP_REQUESTS)
    parser.add_argument("--measure-requests", type=int, default=DEFAULT_MEASURE_REQUESTS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--ready-timeout-s", type=float, default=DEFAULT_READY_TIMEOUT_S)
    parser.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    fusion_env_overrides: Dict[str, str] = {}
    for assignment in args.fusion_env:
        key, separator, value = str(assignment).partition("=")
        if not separator or not key:
            parser.error(f"--fusion-env requires KEY=VALUE, got {assignment!r}")
        fusion_env_overrides[key] = value
    report = run_protected_accelerator_benchmark(
        repo_root=args.repo_root,
        profile=args.profile,
        resident_binary=args.resident_binary,
        fusion_env_overrides=fusion_env_overrides,
        prompt=args.prompt,
        warmup_requests=args.warmup_requests,
        measure_requests=args.measure_requests,
        max_new_tokens=args.max_new_tokens,
        ready_timeout_s=args.ready_timeout_s,
        interval_s=args.interval_s,
        timeout_s=args.timeout_s,
        emit=args.emit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_MEASURE_REQUESTS",
    "DEFAULT_PROMPT",
    "DEFAULT_READY_TIMEOUT_S",
    "DEFAULT_WARMUP_REQUESTS",
    "SCHEMA",
    "run_protected_accelerator_benchmark",
]


if __name__ == "__main__":
    raise SystemExit(main())
