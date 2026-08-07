"""After-proto monitoring + hardening façade.

Purpose:
  - gate governance actions behind a verified Proto-Frankenstein offload seal
  - monitor host pressure with hysteresis and map levels into control actions
  - evaluate source-only cleanup candidates without mutating the filesystem
  - emit completion-safe notification objects for supervisor logging/evidence

The module does not delete files, stop models, or run external daemons.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .garbage_ecosystem import AutoDeleteDecision, ObjectRecord, build_cleanup_receipt, evaluate_auto_delete
from .notifications import (
    AuthoritySource,
    NotificationBus,
    NotificationKind,
)
from .pressure_governor import (
    GovernorAction,
    GovernorThresholds,
    PressureGovernor,
    PressureLevel,
)
from .signals import HostSignals, collect_host_signals

PROTO_OFFLOAD_ENDPOINT = "PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED"
PROTO_FLOOR_GIB = 10.0
MONITOR_SCHEMA = "hawking.ascension.after_proto_monitor.v1"


def _to_tuple(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _normalize_iso8601(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace("Z", "+00:00")
    if not normalized:
        return None
    try:
        return datetime.fromisoformat(normalized).isoformat()
    except ValueError:
        return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


@dataclass(frozen=True)
class ProtoOffloadCheck:
    """Result of a sealed-Proto verification used as a hard gate."""

    receipt_path: str
    exists: bool
    endpoint: str | None
    schema: str | None
    donor_weights_retained: bool | None
    recorded_at: str | None
    reasons: tuple[str, ...]
    dry_run: bool | None
    allowed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "receipt_path": self.receipt_path,
            "exists": self.exists,
            "endpoint": self.endpoint,
            "schema": self.schema,
            "donor_weights_retained": self.donor_weights_retained,
            "recorded_at": self.recorded_at,
            "dry_run": self.dry_run,
            "reasons": list(self.reasons),
            "allowed": self.allowed,
        }


def validate_proto_offload_receipt(
    receipt_path: str | Path,
    *,
    required_endpoint: str = PROTO_OFFLOAD_ENDPOINT,
    require_not_dry_run: bool = True,
) -> ProtoOffloadCheck:
    """Validate a Proto-Frankenstein receipt without trusting any non-authoritative signal."""
    path = Path(receipt_path).expanduser()
    if not path.exists():
        return ProtoOffloadCheck(
            receipt_path=str(path),
            exists=False,
            endpoint=None,
            schema=None,
            donor_weights_retained=None,
            recorded_at=None,
            reasons=("proto receipt file missing",),
            dry_run=None,
            allowed=False,
        )

    reasons: list[str] = []
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return ProtoOffloadCheck(
            receipt_path=str(path),
            exists=True,
            endpoint=None,
            schema=None,
            donor_weights_retained=None,
            recorded_at=None,
            reasons=(f"proto receipt unreadable: {exc}",),
            dry_run=None,
            allowed=False,
        )

    if not isinstance(payload, Mapping):
        return ProtoOffloadCheck(
            receipt_path=str(path),
            exists=True,
            endpoint=None,
            schema=None,
            donor_weights_retained=None,
            recorded_at=None,
            reasons=("proto receipt must be a JSON object",),
            dry_run=None,
            allowed=False,
        )

    endpoint = payload.get("endpoint")
    if not isinstance(endpoint, str):
        endpoint = payload.get("terminal_endpoint")
    if not isinstance(endpoint, str):
        endpoint = None
    schema = payload.get("schema")
    recorded_at = _normalize_iso8601(payload.get("recorded_at"))
    dry_run = _coerce_bool(payload.get("dry_run"))
    runtime_storage = payload.get("runtime_storage")
    storage = (
        runtime_storage.get("storage", {})
        if isinstance(runtime_storage, Mapping)
        else {}
    )
    donor_weights_retained = storage.get("donor_weights_retained")

    if not isinstance(runtime_storage, Mapping):
        reasons.append("proto receipt missing runtime_storage")
    elif not isinstance(storage, Mapping):
        reasons.append("proto receipt missing runtime_storage.storage")

    if runtime_storage is not None and not isinstance(storage, Mapping):
        donor_weights_retained = None
    if donor_weights_retained is not None and not isinstance(donor_weights_retained, bool):
        reasons.append("proto receipt donor_weights_retained must be boolean")
    if donor_weights_retained is None and isinstance(storage, Mapping):
        reasons.append("proto receipt missing donor_weights_retained")

    if not isinstance(endpoint, str):
        reasons.append("proto receipt missing required endpoint")
    elif endpoint != required_endpoint:
        reasons.append(f"proto endpoint mismatch: {endpoint!r} != {required_endpoint!r}")

    if recorded_at is None:
        reasons.append("proto receipt missing recorded_at")

    if require_not_dry_run and dry_run is True:
        reasons.append("proto receipt marked dry_run")
    if require_not_dry_run and dry_run is None and "dry_run" in payload:
        reasons.append("proto receipt dry_run must be boolean")

    if donor_weights_retained is True:
        reasons.append("proto receipt indicates donor weights retained")

    return ProtoOffloadCheck(
        receipt_path=str(path),
        exists=True,
        endpoint=endpoint,
        schema=schema,
        donor_weights_retained=donor_weights_retained,
        recorded_at=recorded_at,
        reasons=_to_tuple(reasons),
        dry_run=dry_run if isinstance(dry_run, bool) else None,
        allowed=not reasons,
    )


@dataclass(frozen=True)
class AfterProtoMonitorResult:
    """Comprehensive, serializable monitor output."""

    timestamp_utc: str
    proto_check: ProtoOffloadCheck
    pressure_action: GovernorAction
    host_signals: HostSignals
    cleanup_receipt: Mapping[str, Any] | None = None
    cleanup_decisions: tuple[AutoDeleteDecision, ...] = field(default_factory=tuple)
    notifications_sent: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    notifications_refused: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    blockers: tuple[str, ...] = field(default_factory=tuple)
    can_advance_after_proto: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MONITOR_SCHEMA,
            "timestamp_utc": self.timestamp_utc,
            "proto_check": self.proto_check.as_dict(),
            "pressure": self.pressure_action.as_dict(),
            "host_signals": self.host_signals.as_dict(),
            "cleanup_receipt": dict(self.cleanup_receipt) if self.cleanup_receipt else None,
            "cleanup_decisions": [d.as_dict() for d in self.cleanup_decisions],
            "notifications": {
                "sent": list(self.notifications_sent),
                "refused": list(self.notifications_refused),
            },
            "blockers": list(self.blockers),
            "can_advance_after_proto": self.can_advance_after_proto,
        }

    def as_receipt(self) -> dict[str, Any]:
        body = self.as_dict()
        body["monitor_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return body


def monitor_after_proto(
    *,
    proto_receipt_path: str | Path,
    signals: HostSignals | None = None,
    governor: PressureGovernor | None = None,
    thresholds: GovernorThresholds | None = None,
    disk_path: str = "/",
    required_endpoint: str = PROTO_OFFLOAD_ENDPOINT,
    require_not_dry_run: bool = True,
    cleanup_records: Sequence[ObjectRecord] | None = None,
    cleanup_apply: bool = False,
    free_bytes_before: int | None = None,
    free_bytes_after: int | None = None,
    bus: NotificationBus | None = None,
    proto_event_authority: AuthoritySource | str = AuthoritySource.SEALED_RECEIPT,
) -> AfterProtoMonitorResult:
    """Run one hardening/monitoring pass.

    *No filesystem cleanup/deletion is executed by this function.*
    """
    governor = governor or PressureGovernor(thresholds=thresholds or GovernorThresholds())
    if signals is None:
        signals = collect_host_signals(disk_path=disk_path)

    invalid_proto_authority = False
    try:
        proto_authority = AuthoritySource(proto_event_authority)
    except ValueError:
        proto_authority = AuthoritySource.SEALED_RECEIPT
        invalid_proto_authority = True

    pressure_action = governor.step(signals)
    proto_check = validate_proto_offload_receipt(
        proto_receipt_path,
        required_endpoint=required_endpoint,
        require_not_dry_run=require_not_dry_run,
    )

    monitor_bus = bus or NotificationBus()
    blockers: list[str] = []

    # Hard floor signal: explicit blocker for after-proto workflows.
    free_disk_gib = pressure_action.sample.signals.get(
        "free_disk_gib", signals.free_disk_gib
    )
    if free_disk_gib < PROTO_FLOOR_GIB:
        blockers.append(
            f"host free_disk_gib below {PROTO_FLOOR_GIB:g} GiB floor"
        )

    # Pressure -> safety notifications (authority for this path is explicit).
    if pressure_action.level is PressureLevel.YELLOW:
        monitor_bus.publish_built(
            NotificationKind.MEMORY_DISK_PRESSURE,
            f"proto-stage yield required at {pressure_action.level.value}",
            authority=AuthoritySource.PRESSURE_GOVERNOR,
            details={"governor_action": pressure_action.as_dict()},
            evidence_paths=[str(proto_receipt_path)],
        )
    if pressure_action.level is PressureLevel.RED:
        monitor_bus.publish_built(
            NotificationKind.MEMORY_DISK_PRESSURE,
            f"proto-stage reclaim and model-stability actions: {pressure_action.level.value}",
            authority=AuthoritySource.PRESSURE_GOVERNOR,
            details={"governor_action": pressure_action.as_dict()},
            severity="warn",
            evidence_paths=[str(proto_receipt_path)],
        )
        blockers.append("storage pressure RED blocks advancement")
    if pressure_action.level is PressureLevel.CRITICAL:
        monitor_bus.publish_built(
            NotificationKind.MEMORY_DISK_PRESSURE,
            "proto-stage critical pressure",
            authority=AuthoritySource.PRESSURE_GOVERNOR,
            details={"governor_action": pressure_action.as_dict()},
            severity="critical",
            evidence_paths=[str(proto_receipt_path)],
        )
        monitor_bus.publish_built(
            NotificationKind.HUMAN_DECISION_REQUIRED,
            "proto-stage requires operator intervention under critical pressure",
            authority=AuthoritySource.SUPERVISOR,
            severity="critical",
            details={"pressure_action": pressure_action.as_dict()},
        )
        blockers.append("storage pressure CRITICAL blocks advancement")

    # Completion-shaped gate only when provenance is present and valid.
    if proto_check.allowed:
        monitor_bus.publish_built(
            NotificationKind.PROTO_SEALED,
            "proto-frankenstein receipt gate satisfied",
            authority=proto_authority,
            evidence_paths=[str(proto_receipt_path)],
        )
        if invalid_proto_authority:
            blockers.append("invalid proto event authority requested; defaulted to sealed receipt")
    else:
        blockers.append("proto offload seal not valid for after-proto actions")

    cleanup_decisions: tuple[AutoDeleteDecision, ...] = ()
    cleanup_receipt: Mapping[str, Any] | None = None
    if cleanup_records is not None:
        if not proto_check.allowed:
            blockers.append("cleanup blocked until proto offload seal is valid")
        elif pressure_action.level is PressureLevel.CRITICAL:
            blockers.append("cleanup blocked while CRITICAL pressure persists")
        else:
            decisions = []
            for record in cleanup_records:
                decisions.append(evaluate_auto_delete(record, apply=cleanup_apply))
            cleanup_decisions = tuple(decisions)
            free_before = (
                free_bytes_before
                if free_bytes_before is not None
                else signals.free_disk_bytes
            )
            auto_reclaimed = 0
            if cleanup_apply:
                for record, decision in zip(cleanup_records, cleanup_decisions):
                    if decision.would_delete:
                        auto_reclaimed += int(record.size_bytes)
            free_after = free_bytes_after if free_bytes_after is not None else (free_before + auto_reclaimed)
            cleanup_receipt = build_cleanup_receipt(
                cleanup_decisions,
                free_bytes_before=free_before,
                free_bytes_after=free_after,
                dry_run=not cleanup_apply,
            )
            if cleanup_apply and auto_reclaimed == 0:
                blockers.append("cleanup apply requested but no deletions would have been allowed")

    can_advance = (
        proto_check.allowed
        and pressure_action.level in (PressureLevel.GREEN, PressureLevel.YELLOW)
        and free_disk_gib >= PROTO_FLOOR_GIB
    )

    return AfterProtoMonitorResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        proto_check=proto_check,
        pressure_action=pressure_action,
        host_signals=signals,
        cleanup_receipt=cleanup_receipt,
        cleanup_decisions=cleanup_decisions,
        notifications_sent=tuple(n.as_dict() for n in monitor_bus.sent),
        notifications_refused=tuple(n.as_dict() for n in monitor_bus.refused),
        blockers=_to_tuple(blockers),
        can_advance_after_proto=can_advance,
    )
