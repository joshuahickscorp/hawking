#!/usr/bin/env python3.12
"""Production-progress-calibrated ETA for the active Doctor V5 campaign.

The original acceleration envelope is intentionally based on an isolated real-
tensor canary.  This companion reads only safetensors headers, the active
cell's small encode logs, and its queue launch timestamp, so it can replace that
optimistic calibration with observed wall progress without opening tensor
payloads or changing the live campaign.  Queue launch time is authoritative:
setup, retries, and resumptions may never disappear from the elapsed interval.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
from typing import Any

import doctor_v5_post_120b as post_120b
import doctor_v5_stacked_admission as stacked


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (ROOT / "reports/condense/doctor_v5_ultra/staged_acceleration/"
          "production_calibrated_eta.json")
RESULTS = ROOT / "reports/condense/doctor_v5_ultra/results"
SCHEMA = "hawking.doctor_v5_production_eta.v2"
MAX_HEADER_BYTES = 64 * 1024 * 1024
APPENDIX_SECONDS = (86_400, 259_200)
DONE_RE = re.compile(r"^\[done \d+/\d+\]\s+(\S+)")
INPUT_BINDING_PATHS = {
    "plan": stacked.PLAN,
    "campaign": stacked.CAMPAIGN,
    "observer_state": stacked.OBSERVER_STATE,
}
THROUGH_120B_REQUIREMENTS = [
    "segment-specific GPT-OSS-120B numerical/runtime receipt",
    "measured GPT-OSS-120B RAM receipt",
    "controlled terminal wiring",
]
APPENDIX_REQUIREMENTS = [
    "segment-specific Appendix device parity/performance receipt",
    "owner-free release probes and physical counters",
]
CLAIM_LIMITS = {
    "unavailable-live-production-calibration": [
        "no completion date is issued without a conservative live calibration",
        "setup, retry, and resumption time remain inside the wall-clock anchor",
        "120B and Appendix remain unavailable without segment-specific receipts",
        "mechanical sensitivity is not an ETA and emits no calendar date",
    ],
    "blocked-live-production-calibration": [
        "the observed active-cell speedup remains provisional",
        "no completion date is issued while any admitted cell is blocked",
        "120B and Appendix remain unavailable without segment-specific receipts",
        "sub-120B evidence is never transferred to GPT-OSS-120B",
    ],
    "provisional-live-production-calibration": [
        "active cell is incomplete, so this remains provisional",
        "two-lane lower bound omits CPU contention",
        "120B and Appendix remain unavailable without segment-specific receipts",
        "mechanical sensitivity has no calendar dates and is not an ETA",
        "sub-120B evidence is never transferred to GPT-OSS-120B",
    ],
}
CALIBRATION_KEYS = {
    "cell_id", "cell_identity_sha256", "model_label", "branch_rate",
    "log_artifacts", "queue_started_at", "first_encode_started_at",
    "queue_attempts", "completed_weights", "total_two_dimensional_weights",
    "progress_fraction", "elapsed_seconds", "observed_weights_per_second",
    "projected_full_cell_seconds", "legacy_cell_seconds",
    "sub_120b_observed_speedup", "transferable_to_gpt_oss_120b",
}


class ProductionEtaError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 \
        and all(character in "0123456789abcdef" for character in value)


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_bound_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one small JSON input once and bind exactly those bytes."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionEtaError(f"unsafe ETA input: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_HEADER_BYTES:
            raise ProductionEtaError(f"unsafe ETA input: {path}")
        raw = bytearray()
        while len(raw) <= MAX_HEADER_BYTES:
            chunk = os.read(descriptor, min(1024 * 1024,
                                             MAX_HEADER_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns
        )
        if len(raw) > MAX_HEADER_BYTES or identity(before) != identity(after) \
                or len(raw) != after.st_size:
            raise ProductionEtaError(f"ETA input changed while reading: {path}")
    finally:
        os.close(descriptor)
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise ProductionEtaError(f"ETA input vanished after reading: {path}") from exc
    if stat.S_ISLNK(path_after.st_mode) \
            or (path_after.st_dev, path_after.st_ino) != (before.st_dev, before.st_ino):
        raise ProductionEtaError(f"ETA input path changed while reading: {path}")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionEtaError(f"invalid ETA input JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProductionEtaError(f"ETA input root is not an object: {path}")
    reference = {
        "path": _relative_path(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    return value, reference


def _identity_valid(value: dict[str, Any], field: str) -> bool:
    claimed = value.get(field)
    return _valid_sha256(claimed) and claimed == _hash_value({
        key: row for key, row in value.items() if key != field
    })


def _read_inputs() -> tuple[dict[str, Any], dict[str, Any],
                            dict[str, Any], dict[str, Any]]:
    plan, plan_reference = _read_bound_json(stacked.PLAN)
    campaign, campaign_reference = _read_bound_json(stacked.CAMPAIGN)
    observer, observer_reference = _read_bound_json(stacked.OBSERVER_STATE)
    if not _identity_valid(plan, "plan_sha256"):
        raise ProductionEtaError("campaign plan identity is invalid")
    if not _identity_valid(campaign, "campaign_sha256"):
        raise ProductionEtaError("campaign identity is invalid")
    if not _identity_valid(observer, "state_sha256"):
        raise ProductionEtaError("observer state identity is invalid")
    if campaign.get("plan_sha256") != plan.get("plan_sha256"):
        raise ProductionEtaError("campaign/plan identity differs")
    bindings = {
        "plan": plan_reference,
        "campaign": campaign_reference,
        "observer_state": observer_reference,
        "plan_sha256": plan["plan_sha256"],
        "campaign_sha256": campaign["campaign_sha256"],
        "observer_state_sha256": observer["state_sha256"],
    }
    return plan, campaign, observer, bindings


def _reference_shape(reference: Any) -> bool:
    return isinstance(reference, dict) \
        and set(reference) == {"path", "sha256", "bytes"} \
        and isinstance(reference.get("path"), str) and reference["path"] \
        and _valid_sha256(reference.get("sha256")) \
        and not isinstance(reference.get("bytes"), bool) \
        and isinstance(reference.get("bytes"), int) and reference["bytes"] >= 0


def _input_binding_errors(bindings: Any, *, verify_freshness: bool) -> list[str]:
    expected_keys = {
        "plan", "campaign", "observer_state", "plan_sha256",
        "campaign_sha256", "observer_state_sha256",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected_keys:
        return ["ETA input bindings are incomplete"]
    errors: list[str] = []
    for name in ("plan", "campaign", "observer_state"):
        if not _reference_shape(bindings.get(name)):
            errors.append(f"ETA input binding is invalid: {name}")
    if any(not _valid_sha256(bindings.get(name)) for name in (
            "plan_sha256", "campaign_sha256", "observer_state_sha256")):
        errors.append("ETA declared source identities are invalid")
    if errors or not verify_freshness:
        return errors
    expected_paths = {
        name: _relative_path(path) for name, path in INPUT_BINDING_PATHS.items()
    }
    for name, path in INPUT_BINDING_PATHS.items():
        if bindings[name].get("path") != expected_paths[name]:
            errors.append(f"ETA input binding targets a non-canonical path: {name}")
            continue
        try:
            value, current = _read_bound_json(path)
        except ProductionEtaError as exc:
            errors.append(str(exc))
            continue
        if current != bindings[name]:
            errors.append(f"ETA input binding is stale: {name}")
            continue
        identity_field = {
            "plan": "plan_sha256",
            "campaign": "campaign_sha256",
            "observer_state": "state_sha256",
        }[name]
        binding_field = {
            "plan": "plan_sha256",
            "campaign": "campaign_sha256",
            "observer_state": "observer_state_sha256",
        }[name]
        if value.get(identity_field) != bindings.get(binding_field) \
                or not _identity_valid(value, identity_field):
            errors.append(f"ETA input declared identity is stale: {name}")
    return errors


def _header(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProductionEtaError(f"unsafe safetensors shard: {path}")
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ProductionEtaError(f"truncated safetensors header: {path}")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= MAX_HEADER_BYTES:
            raise ProductionEtaError(f"safetensors header exceeds bound: {path}")
        payload = handle.read(length)
        if len(payload) != length:
            raise ProductionEtaError(f"truncated safetensors JSON header: {path}")
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionEtaError(f"invalid safetensors JSON header: {path}") from exc
    if not isinstance(value, dict):
        raise ProductionEtaError(f"safetensors header root is invalid: {path}")
    return value


def _two_dimensional_weights(path: Path) -> dict[str, int]:
    rows: dict[str, int] = {}
    for name, row in _header(path).items():
        if name == "__metadata__" or not isinstance(row, dict):
            continue
        shape = row.get("shape")
        if not isinstance(shape, list) or len(shape) != 2 \
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                       for value in shape):
            continue
        rows[name] = math.prod(shape)
    return rows


def _parse_log(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionEtaError(f"unsafe encode log: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_HEADER_BYTES:
            raise ProductionEtaError(f"unsafe encode log: {path}")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024,
                                            MAX_HEADER_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_HEADER_BYTES:
                raise ProductionEtaError(f"unsafe encode log: {path}")
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns
        )
        if identity(before) != identity(after):
            raise ProductionEtaError(f"encode log changed while reading: {path}")
    finally:
        os.close(descriptor)
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise ProductionEtaError(f"encode log vanished after reading: {path}") from exc
    if stat.S_ISLNK(path_after.st_mode) \
            or (path_after.st_dev, path_after.st_ino) != (before.st_dev, before.st_ino):
        raise ProductionEtaError(f"encode log path changed while reading: {path}")
    raw = b"".join(chunks)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProductionEtaError(f"encode log is not UTF-8: {path}") from exc
    started: dt.datetime | None = None
    source: Path | None = None
    attempt_count = 0
    names: list[str] = []
    for line in text.splitlines():
        if line.startswith("{"):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = None
            if isinstance(event, dict) and event.get("event") == "child_start":
                try:
                    candidate_started = dt.datetime.fromisoformat(event["at"])
                    argv = event["argv"]
                    candidate_source = Path(
                        argv[argv.index("--in") + 1]
                    ).resolve(strict=True)
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    raise ProductionEtaError(f"invalid child_start event: {path}") from exc
                if source is not None and candidate_source != source:
                    raise ProductionEtaError(
                        f"encode log changed source across attempts: {path}"
                    )
                started, source = candidate_started, candidate_source
                attempt_count += 1
                # Logs are append-only across resumptions. Only completion
                # events after the latest bound child_start describe progress
                # retained by the current attempt; prior attempts remain paid
                # for through the queue-level wall-clock anchor.
                names = []
        match = DONE_RE.match(line)
        if match:
            names.append(match.group(1))
    if started is None or source is None:
        raise ProductionEtaError(f"encode log lacks a bound child_start: {path}")
    if len(names) != len(set(names)):
        raise ProductionEtaError(f"encode log repeats a completed tensor: {path}")
    weights = _two_dimensional_weights(source)
    unknown = sorted(set(names) - set(weights))
    if unknown:
        raise ProductionEtaError(f"encode log names unknown tensors: {path}: {unknown[:3]}")
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "path": str(path.resolve()), "sha256": digest, "bytes": size,
        "source_path": str(source), "started_at": started,
        "attempt_count": attempt_count,
        "completed_tensor_count": len(names),
        "completed_weights": sum(weights[name] for name in names),
    }


def _date(now: dt.datetime, seconds: float) -> str:
    return (now + dt.timedelta(seconds=seconds)).isoformat(timespec="seconds")


def _calibration(plan: dict[str, Any], campaign: dict[str, Any],
                 now: dt.datetime) -> dict[str, Any]:
    if now.tzinfo is None:
        raise ProductionEtaError("calibration observation lacks a timezone")
    blocked_cells = sorted(
        row.get("cell_id") for row in campaign.get("cells", [])
        if isinstance(row, dict) and row.get("status") == "blocked-execution"
        and isinstance(row.get("cell_id"), str)
    )
    if blocked_cells:
        raise ProductionEtaError(
            "one or more admitted cells are blocked-execution: "
            + ", ".join(blocked_cells)
        )
    active = campaign.get("active_cells")
    if not isinstance(active, list) or len(active) != 1:
        raise ProductionEtaError("production ETA requires exactly one active cell")
    cell_id = active[0]
    cells = {row["cell_id"]: row for row in plan["cells"]}
    cell = cells.get(cell_id)
    if not isinstance(cell, dict) or cell.get("model_family") != "qwen2.5-dense":
        raise ProductionEtaError("active cell is not a calibrated Qwen dense cell")
    campaign_rows = {
        row.get("cell_id"): row for row in campaign.get("cells", [])
        if isinstance(row, dict) and isinstance(row.get("cell_id"), str)
    }
    campaign_row = campaign_rows.get(cell_id)
    if not isinstance(campaign_row, dict) or campaign_row.get("status") != "running" \
            or not isinstance(campaign_row.get("started_at"), str):
        raise ProductionEtaError("active cell lacks a bound queue launch timestamp")
    try:
        queue_started = dt.datetime.fromisoformat(campaign_row["started_at"])
    except ValueError as exc:
        raise ProductionEtaError("active queue launch timestamp is invalid") from exc
    if queue_started.tzinfo is None:
        raise ProductionEtaError("active queue launch timestamp lacks a timezone")
    log_root = RESULTS / cell_id / "strand_ladder" / "logs"
    logs = [_parse_log(path) for path in sorted(log_root.glob("encode-*.log"))]
    if not logs:
        raise ProductionEtaError("active cell has no encode logs")
    sources = sorted(
        path.resolve(strict=True)
        for path in (ROOT / cell["model_dir"]).resolve().glob("*.safetensors")
    )
    if not sources:
        raise ProductionEtaError("active model has no safetensors shards")
    log_sources = [Path(row["source_path"]).resolve() for row in logs]
    if len(log_sources) != len(set(log_sources)) \
            or not set(log_sources) <= set(sources):
        raise ProductionEtaError(
            "encode logs do not bind unique active-model source shards"
        )
    if any(row["started_at"].tzinfo is None or row["started_at"] > now
           for row in logs):
        raise ProductionEtaError("encode log child_start is future-dated or timezone-free")
    total_weights = sum(sum(_two_dimensional_weights(path).values()) for path in sources)
    completed_weights = sum(row["completed_weights"] for row in logs)
    earliest_encode = min(row["started_at"] for row in logs)
    if earliest_encode.tzinfo is None or queue_started > earliest_encode \
            or queue_started > now:
        raise ProductionEtaError("queue/log wall-clock ordering is invalid")
    elapsed = max(0.0, (now - queue_started).total_seconds())
    if completed_weights <= 0 or total_weights <= 0 \
            or completed_weights > total_weights or elapsed <= 0:
        raise ProductionEtaError("active production progress is insufficient")
    rate = completed_weights / elapsed
    projected = total_weights / rate
    _, _, keyed_rates, _ = post_120b._runtime_rates(campaign)
    key = f"{cell['branch']}@{cell['rate_id']}"
    legacy = keyed_rates[key] * cell["exact_stored_parameter_count"] / 1e9
    speedup = legacy / projected
    if not math.isfinite(speedup) or speedup <= 1:
        raise ProductionEtaError("production progress does not yet prove acceleration")
    public_logs = [{key: value for key, value in row.items() if key != "started_at"}
                   for row in logs]
    return {
        "cell_id": cell_id,
        "cell_identity_sha256": cell["cell_identity_sha256"],
        "model_label": cell["model_label"],
        "branch_rate": key,
        "log_artifacts": public_logs,
        "queue_started_at": queue_started.isoformat(timespec="seconds"),
        "first_encode_started_at": earliest_encode.isoformat(timespec="seconds"),
        "queue_attempts": campaign_row.get("attempts"),
        "completed_weights": completed_weights,
        "total_two_dimensional_weights": total_weights,
        "progress_fraction": completed_weights / total_weights,
        "elapsed_seconds": elapsed,
        "observed_weights_per_second": rate,
        "projected_full_cell_seconds": projected,
        "legacy_cell_seconds": legacy,
        "sub_120b_observed_speedup": speedup,
        "transferable_to_gpt_oss_120b": False,
    }


def _empty_sub_120b() -> dict[str, Any]:
    return {
        "available": False,
        "seconds_range": None,
        "days_range": None,
        "date_range": None,
    }


def _gated_segment(*, appendix: bool) -> dict[str, Any]:
    return {
        "available": False,
        "execution_ready": False,
        "segment_receipt_validated": False,
        "seconds_range": None,
        "days_range": None,
        "date_range": None,
        "requires": list(
            APPENDIX_REQUIREMENTS if appendix else THROUGH_120B_REQUIREMENTS
        ),
    }


def _mechanical_sensitivity(*, observed_sub_120b_speedup: float | None,
                            through_range: list[float] | None,
                            blockers: list[str] | None = None) -> dict[str, Any]:
    available = through_range is not None
    through = list(through_range) if through_range is not None else None
    full = None if through is None else [
        through[0] + APPENDIX_SECONDS[0],
        through[1] + APPENDIX_SECONDS[1],
    ]
    return {
        "available": available,
        "not_an_eta": True,
        "calendar_dates_emitted": False,
        "sub_120b_speedup_applied": 1.0,
        "observed_sub_120b_speedup_excluded": observed_sub_120b_speedup,
        "sub_120b_speedup_transferable_to_120b": False,
        "gpt_oss_120b_speedup": 1.0,
        "gpt_oss_120b_speedup_evidence": "none",
        "appendix_increment_days_assumption": [1, 3],
        "through_120b_seconds_range": through,
        "through_120b_days_range": (
            None if through is None else [value / 86_400 for value in through]
        ),
        "through_120b_plus_appendix_seconds_range": full,
        "through_120b_plus_appendix_days_range": (
            None if full is None else [value / 86_400 for value in full]
        ),
        "blockers": sorted(set(blockers or [])),
    }


def build(now: dt.datetime | None = None) -> dict[str, Any]:
    observation = (now or dt.datetime.now(dt.timezone.utc)).replace(microsecond=0)
    plan, campaign, _observer, bindings = _read_inputs()
    try:
        calibration = _calibration(plan, campaign, observation)
    except ProductionEtaError as exc:
        document = {
            "schema": SCHEMA,
            "created_at": observation.isoformat(timespec="seconds"),
            "status": "unavailable-live-production-calibration",
            "eta_scope": "sub-120b-only",
            "calibration_available": False,
            "eta_blocked": True,
            "input_bindings": bindings,
            "blockers": [str(exc)],
            "sub_120b": _empty_sub_120b(),
            "through_120b": _gated_segment(appendix=False),
            "through_120b_plus_appendix": _gated_segment(appendix=True),
            "mechanical_sensitivity": _mechanical_sensitivity(
                observed_sub_120b_speedup=None, through_range=None,
                blockers=[str(exc)],
            ),
            "claim_limits": CLAIM_LIMITS[
                "unavailable-live-production-calibration"
            ],
        }
        document["document_sha256"] = _hash_value(document)
        return document
    speedup = calibration["sub_120b_observed_speedup"]
    evidence = stacked._observed_tier_rss(plan)
    one = stacked.simulate(
        plan, campaign, margin_bytes=stacked.DYNAMIC_BASE_MARGIN_BYTES,
        speedup=speedup,
        include_unready_120b=False, max_lanes=1, evidence=evidence,
    )
    packed = stacked.simulate(
        plan, campaign, margin_bytes=stacked.DYNAMIC_BASE_MARGIN_BYTES,
        speedup=speedup,
        include_unready_120b=False, max_lanes=2, evidence=evidence,
    )
    representative = next(cell for cell in plan["cells"] if cell["model_label"] == "120B")
    hypothetical = dict(evidence)
    hypothetical["120B"] = {
        "peak_bytes": stacked._projected_residency(
            representative, stacked.DYNAMIC_BASE_MARGIN_BYTES
        ),
        "samples": 3,
    }
    through_one = stacked.simulate(
        plan, campaign, margin_bytes=stacked.DYNAMIC_BASE_MARGIN_BYTES,
        speedup=1.0,
        include_unready_120b=True, max_lanes=1,
        evidence=hypothetical,
    )
    through_two = stacked.simulate(
        plan, campaign, margin_bytes=stacked.DYNAMIC_BASE_MARGIN_BYTES,
        speedup=1.0,
        include_unready_120b=True, max_lanes=2,
        evidence=hypothetical,
    )
    sub_simulations = {
        "sub_120b_one_lane": one,
        "sub_120b_packed": packed,
    }
    mechanical_simulations = {
        "through_120b_one_lane": through_one,
        "through_120b_packed": through_two,
    }
    failed_sub = {name: row for name, row in sub_simulations.items()
                  if row.get("ok") is not True}
    failed_mechanical = {
        name: row for name, row in mechanical_simulations.items()
        if row.get("ok") is not True
    }
    if failed_sub:
        failed = {**failed_sub, **failed_mechanical}
        blockers = sorted({
            str(row.get("blocker", f"{name} simulation failed"))
            for name, row in failed.items()
        })
        document = {
            "schema": SCHEMA,
            "created_at": observation.isoformat(timespec="seconds"),
            "status": "blocked-live-production-calibration",
            "eta_scope": "sub-120b-only",
            "calibration_available": True,
            "calibration": calibration,
            "eta_blocked": True,
            "input_bindings": bindings,
            "blockers": blockers,
            "failed_simulations": sorted(failed),
            "sub_120b": _empty_sub_120b(),
            "through_120b": _gated_segment(appendix=False),
            "through_120b_plus_appendix": _gated_segment(appendix=True),
            "mechanical_sensitivity": _mechanical_sensitivity(
                observed_sub_120b_speedup=speedup, through_range=None,
                blockers=[
                    str(row.get("blocker", f"{name} simulation failed"))
                    for name, row in failed_mechanical.items()
                ],
            ),
            "claim_limits": CLAIM_LIMITS[
                "blocked-live-production-calibration"
            ],
        }
        document["document_sha256"] = _hash_value(document)
        return document
    sub = sorted((packed["sub_120b_seconds"], one["sub_120b_seconds"]))
    through = None if failed_mechanical else sorted((
        through_two["through_120b_seconds"],
        through_one["through_120b_seconds"],
    ))
    document = {
        "schema": SCHEMA,
        "created_at": observation.isoformat(timespec="seconds"),
        "status": "provisional-live-production-calibration",
        "eta_scope": "sub-120b-only",
        "calibration_available": True,
        "calibration": calibration,
        "eta_blocked": False,
        "input_bindings": bindings,
        "sub_120b": {
            "available": True,
            "seconds_range": sub,
            "days_range": [value / 86_400 for value in sub],
            "date_range": [_date(observation, value) for value in sub],
        },
        "through_120b": _gated_segment(appendix=False),
        "through_120b_plus_appendix": _gated_segment(appendix=True),
        "mechanical_sensitivity": _mechanical_sensitivity(
            observed_sub_120b_speedup=speedup, through_range=through,
            blockers=[
                str(row.get("blocker", f"{name} simulation failed"))
                for name, row in failed_mechanical.items()
            ],
        ),
        "claim_limits": CLAIM_LIMITS[
            "provisional-live-production-calibration"
        ],
    }
    document["document_sha256"] = _hash_value(document)
    return document


def _valid_pair(value: Any, *, positive: bool = True) -> bool:
    floor = 0 if positive else -math.inf
    return (
        isinstance(value, list) and len(value) == 2
        and all(not isinstance(row, bool)
                and isinstance(row, (int, float))
                and math.isfinite(float(row))
                and float(row) > floor for row in value)
        and value == sorted(value)
    )


def _gated_segment_errors(value: Any, *, appendix: bool) -> list[str]:
    expected_keys = {
        "available", "execution_ready", "segment_receipt_validated",
        "seconds_range", "days_range", "date_range", "requires",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        return ["gated segment shape is not closed-world"]
    expected_requirements = (
        APPENDIX_REQUIREMENTS if appendix else THROUGH_120B_REQUIREMENTS
    )
    if value.get("available") is not False \
            or value.get("execution_ready") is not False \
            or value.get("segment_receipt_validated") is not False \
            or any(value.get(field) is not None for field in (
                "seconds_range", "days_range", "date_range"
            )) \
            or value.get("requires") != expected_requirements:
        return ["unreceipted segment exposes ETA data"]
    return []


def _mechanical_errors(value: Any, *, calibration_available: bool) -> list[str]:
    expected_keys = {
        "available", "not_an_eta", "calendar_dates_emitted",
        "sub_120b_speedup_applied", "observed_sub_120b_speedup_excluded",
        "sub_120b_speedup_transferable_to_120b",
        "gpt_oss_120b_speedup", "gpt_oss_120b_speedup_evidence",
        "appendix_increment_days_assumption", "through_120b_seconds_range",
        "through_120b_days_range",
        "through_120b_plus_appendix_seconds_range",
        "through_120b_plus_appendix_days_range", "blockers",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        return ["mechanical sensitivity shape is not closed-world"]
    errors: list[str] = []
    blockers = value.get("blockers")
    if value.get("not_an_eta") is not True \
            or value.get("calendar_dates_emitted") is not False \
            or value.get("sub_120b_speedup_applied") != 1.0 \
            or value.get("sub_120b_speedup_transferable_to_120b") is not False \
            or value.get("gpt_oss_120b_speedup") != 1.0 \
            or value.get("gpt_oss_120b_speedup_evidence") != "none" \
            or value.get("appendix_increment_days_assumption") != [1, 3] \
            or not isinstance(blockers, list) \
            or any(not isinstance(row, str) or not row for row in blockers) \
            or blockers != sorted(set(blockers)):
        errors.append("mechanical sensitivity claim boundary is invalid")
    speedup = value.get("observed_sub_120b_speedup_excluded")
    if calibration_available:
        if isinstance(speedup, bool) or not isinstance(speedup, (int, float)) \
                or not math.isfinite(float(speedup)) or float(speedup) <= 1:
            errors.append("mechanical sensitivity lacks a valid sub-120B speedup")
    elif speedup is not None:
        errors.append("unavailable calibration exposes a mechanical speedup")
    pairs = [value.get(field) for field in (
        "through_120b_seconds_range", "through_120b_days_range",
        "through_120b_plus_appendix_seconds_range",
        "through_120b_plus_appendix_days_range",
    )]
    if value.get("available") is False:
        if any(pair is not None for pair in pairs):
            errors.append("unavailable mechanical sensitivity exposes arithmetic")
        return errors
    if value.get("available") is not True or not all(_valid_pair(pair) for pair in pairs):
        errors.append("mechanical sensitivity ranges are invalid")
        return errors
    through_s, through_d, full_s, full_d = pairs
    if not all(math.isclose(float(day), float(seconds) / 86_400,
                            rel_tol=1e-12, abs_tol=1e-12)
               for day, seconds in zip(through_d, through_s, strict=True)) \
            or not all(math.isclose(float(day), float(seconds) / 86_400,
                                    rel_tol=1e-12, abs_tol=1e-12)
                       for day, seconds in zip(full_d, full_s, strict=True)) \
            or not all(math.isclose(float(full), float(base) + increment,
                                    rel_tol=1e-12, abs_tol=1e-9)
                       for full, base, increment in zip(
                           full_s, through_s, APPENDIX_SECONDS, strict=True
                       )):
        errors.append("mechanical sensitivity arithmetic differs")
    if blockers:
        errors.append("available mechanical sensitivity retains blockers")
    return errors


def validate(document: Any, *, verify_freshness: bool = True) -> list[str]:
    if not isinstance(document, dict):
        return ["production ETA must be an object"]
    errors: list[str] = []
    if document.get("schema") != SCHEMA:
        errors.append("production ETA schema differs")
    unstamped = {key: value for key, value in document.items() if key != "document_sha256"}
    try:
        sealed = document.get("document_sha256") == _hash_value(unstamped)
    except (TypeError, ValueError):
        sealed = False
    if not sealed:
        errors.append("production ETA document hash differs")
    status = document.get("status")
    common_keys = {
        "schema", "created_at", "status", "eta_scope",
        "calibration_available", "eta_blocked", "input_bindings",
        "sub_120b", "through_120b", "through_120b_plus_appendix",
        "mechanical_sensitivity", "claim_limits", "document_sha256",
    }
    status_keys = {
        "unavailable-live-production-calibration": common_keys | {"blockers"},
        "blocked-live-production-calibration": common_keys | {
            "blockers", "calibration", "failed_simulations",
        },
        "provisional-live-production-calibration": common_keys | {"calibration"},
    }
    expected_top = status_keys.get(status)
    if expected_top is None:
        errors.append("production ETA status differs")
    elif set(document) != expected_top:
        errors.append("production ETA top-level shape is not closed-world")
    try:
        created = dt.datetime.fromisoformat(document.get("created_at", ""))
    except (TypeError, ValueError):
        created = None
    if created is None or created.tzinfo is None:
        errors.append("production ETA creation time is invalid")
    if document.get("eta_scope") != "sub-120b-only":
        errors.append("production ETA scope is not sub-120B-only")
    if status in CLAIM_LIMITS \
            and document.get("claim_limits") != CLAIM_LIMITS[status]:
        errors.append("production ETA claim limits differ")
    errors.extend(_input_binding_errors(
        document.get("input_bindings"), verify_freshness=verify_freshness
    ))
    calibration = document.get("calibration")
    if not isinstance(calibration, dict):
        calibration = {}
    if status != "unavailable-live-production-calibration":
        numeric_calibration = (
            "completed_weights", "total_two_dimensional_weights",
            "progress_fraction", "elapsed_seconds", "observed_weights_per_second",
            "projected_full_cell_seconds", "legacy_cell_seconds",
            "sub_120b_observed_speedup",
        )
        logs = calibration.get("log_artifacts")
        log_keys = {
            "path", "sha256", "bytes", "source_path", "attempt_count",
            "completed_tensor_count", "completed_weights",
        }
        logs_valid = isinstance(logs, list) and bool(logs) and all(
            isinstance(row, dict) and set(row) == log_keys
            and isinstance(row.get("path"), str) and bool(row.get("path"))
            and isinstance(row.get("source_path"), str) and bool(row.get("source_path"))
            and _valid_sha256(row.get("sha256"))
            and all(not isinstance(row.get(field), bool)
                    and isinstance(row.get(field), int) and row.get(field) >= 0
                    for field in (
                        "bytes", "attempt_count", "completed_tensor_count",
                        "completed_weights",
                    ))
            for row in (logs or [])
        )
        if set(calibration) != CALIBRATION_KEYS \
                or any(not isinstance(calibration.get(field), str)
                       or not calibration.get(field) for field in (
                           "cell_id", "model_label", "branch_rate",
                           "queue_started_at", "first_encode_started_at"
                       )) \
                or not _valid_sha256(calibration.get("cell_identity_sha256")) \
                or not logs_valid \
                or isinstance(calibration.get("queue_attempts"), bool) \
                or not isinstance(calibration.get("queue_attempts"), int) \
                or calibration.get("queue_attempts") < 1 \
                or any(isinstance(calibration.get(field), bool)
                       or not isinstance(calibration.get(field), (int, float))
                       or not math.isfinite(float(calibration.get(field)))
                       or float(calibration.get(field)) <= 0
                       for field in numeric_calibration):
            errors.append("production calibration shape is not closed-world")
        if verify_freshness and created is not None and created.tzinfo is not None:
            try:
                current_plan, current_campaign, _current_observer, current_bindings = (
                    _read_inputs()
                )
                recomputed = _calibration(current_plan, current_campaign, created)
            except ProductionEtaError as exc:
                errors.append(f"production calibration cannot be recomputed: {exc}")
            else:
                if current_bindings != document.get("input_bindings") \
                        or recomputed != calibration:
                    errors.append("production calibration differs from current bound inputs")
    speedup = calibration.get("sub_120b_observed_speedup")
    if status != "unavailable-live-production-calibration" \
            and (isinstance(speedup, bool)
                 or not isinstance(speedup, (int, float))
                 or not math.isfinite(float(speedup)) or float(speedup) <= 1
                 or calibration.get("transferable_to_gpt_oss_120b") is not False):
        errors.append("sub-120B production speedup boundary is not proven")
    sub_document = document.get("sub_120b")
    if not isinstance(sub_document, dict):
        sub_document = {}
    through_document = document.get("through_120b")
    appendix_document = document.get("through_120b_plus_appendix")
    errors.extend(_gated_segment_errors(through_document, appendix=False))
    errors.extend(_gated_segment_errors(appendix_document, appendix=True))
    calibration_available = document.get("calibration_available") is True
    errors.extend(_mechanical_errors(
        document.get("mechanical_sensitivity"),
        calibration_available=calibration_available,
    ))
    empty_sub = (
        isinstance(sub_document, dict)
        and set(sub_document) == {"available", "seconds_range", "days_range", "date_range"}
        and sub_document.get("available") is False
        and all(sub_document.get(field) is None
                for field in ("seconds_range", "days_range", "date_range"))
    )
    if status == "unavailable-live-production-calibration":
        blockers = document.get("blockers")
        if document.get("calibration_available") is not False \
                or document.get("eta_blocked") is not True \
                or not isinstance(blockers, list) or not blockers \
                or any(not isinstance(row, str) or not row for row in blockers) \
                or blockers != sorted(set(blockers)) \
                or not empty_sub:
            errors.append("unavailable production ETA contract is invalid")
    elif status == "blocked-live-production-calibration":
        blockers = document.get("blockers")
        failed = document.get("failed_simulations")
        if document.get("calibration_available") is not True \
                or document.get("eta_blocked") is not True \
                or not isinstance(blockers, list) or not blockers \
                or any(not isinstance(row, str) or not row for row in blockers) \
                or blockers != sorted(set(blockers)) \
                or not isinstance(failed, list) or not failed \
                or any(not isinstance(row, str) or not row for row in failed) \
                or failed != sorted(set(failed)) \
                or not empty_sub:
            errors.append("blocked production ETA contract is invalid")
    elif status == "provisional-live-production-calibration":
        sub = sub_document.get("seconds_range")
        if document.get("calibration_available") is not True \
                or document.get("eta_blocked") is not False \
                or not isinstance(sub_document, dict) \
                or set(sub_document) != {
                    "available", "seconds_range", "days_range", "date_range"
                } \
                or sub_document.get("available") is not True \
                or not _valid_pair(sub) \
                or not _valid_pair(sub_document.get("days_range")) \
                or not all(math.isclose(float(day), float(second) / 86_400,
                                        rel_tol=1e-12, abs_tol=1e-12)
                           for day, second in zip(
                               sub_document.get("days_range", []), sub, strict=True
                           )) \
                or created is None \
                or sub_document.get("date_range") != [
                    _date(created, float(second)) for second in sub
                ]:
            errors.append("sub-120B ETA range is invalid")
    return errors


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _selftest() -> int:
    created_at = "2026-07-14T20:00:00+00:00"
    sub = [86_400.0, 172_800.0]
    through = [259_200.0, 345_600.0]
    anchor = dt.datetime.fromisoformat(created_at)
    reference = {"path": "fixture.json", "sha256": "a" * 64, "bytes": 1}
    payload = {
        "schema": SCHEMA,
        "created_at": created_at,
        "status": "provisional-live-production-calibration",
        "eta_scope": "sub-120b-only",
        "calibration_available": True,
        "eta_blocked": False,
        "input_bindings": {
            "plan": reference, "campaign": reference,
            "observer_state": reference, "plan_sha256": "b" * 64,
            "campaign_sha256": "c" * 64, "observer_state_sha256": "d" * 64,
        },
        "calibration": {
            "cell_id": "fixture-cell", "cell_identity_sha256": "e" * 64,
            "model_label": "3B", "branch_rate": "codec_control@4",
            "log_artifacts": [{
                "path": "/fixture/encode.log", "sha256": "f" * 64,
                "bytes": 1, "source_path": "/fixture/model.safetensors",
                "attempt_count": 1, "completed_tensor_count": 1,
                "completed_weights": 1,
            }], "queue_started_at": created_at,
            "first_encode_started_at": created_at, "queue_attempts": 1,
            "completed_weights": 1, "total_two_dimensional_weights": 2,
            "progress_fraction": 0.5, "elapsed_seconds": 1.0,
            "observed_weights_per_second": 1.0,
            "projected_full_cell_seconds": 2.0, "legacy_cell_seconds": 4.0,
            "sub_120b_observed_speedup": 2.0,
            "transferable_to_gpt_oss_120b": False,
        },
        "sub_120b": {
            "available": True, "seconds_range": sub,
            "days_range": [value / 86_400 for value in sub],
            "date_range": [_date(anchor, value) for value in sub],
        },
        "through_120b": _gated_segment(appendix=False),
        "through_120b_plus_appendix": _gated_segment(appendix=True),
        "mechanical_sensitivity": _mechanical_sensitivity(
            observed_sub_120b_speedup=2.0, through_range=through,
        ),
        "claim_limits": CLAIM_LIMITS[
            "provisional-live-production-calibration"
        ],
    }
    payload["document_sha256"] = _hash_value(payload)
    assert validate(payload, verify_freshness=False) == []
    broken = json.loads(json.dumps(payload))
    broken["through_120b"]["date_range"] = ["2026-07-15", "2026-07-16"]
    broken["document_sha256"] = _hash_value(
        {key: value for key, value in broken.items() if key != "document_sha256"}
    )
    assert "unreceipted segment exposes ETA data" in validate(
        broken, verify_freshness=False
    )
    print("doctor_v5_production_eta.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--write", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.status and args.write is None:
        parser.error("choose --status, --write, or --selftest")
    document = build()
    errors = validate(document)
    if errors:
        raise ProductionEtaError("; ".join(errors))
    if args.write is not None:
        _atomic_json(args.write, document)
    else:
        print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
