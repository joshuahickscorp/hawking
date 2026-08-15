"""Ascension governance scaffolds — garbage ecosystem, pressure governor, notifications.

Future programme, gated on Proto-Frankenstein offload. These modules generalize
tonight's live operators without mutating them:

- ``reclaim_storage_keep_proto.py`` → ``garbage_ecosystem``
- disk/memory/GPU sampling + floor alerts → ``pressure_governor`` / ``signals``
- ``v0_notifier.py`` event surface → ``notifications``

No detached daemons are started from this package. Live reclaim and Telegram
notifier remain the sole production actors until Proto-Frankenstein is sealed
and offloaded.
"""

from __future__ import annotations

from .garbage_ecosystem import (
    AutoDeleteDecision,
    ObjectClass,
    ObjectRecord,
    classify_object,
    evaluate_auto_delete,
    never_auto_delete_reason,
)
from .notifications import (
    AuthoritySource,
    NotificationEvent,
    NotificationKind,
    build_notification,
    may_declare_completion,
)
from .pressure_governor import (
    GovernorAction,
    PressureLevel,
    PressureSample,
    PressureGovernor,
    evaluate_pressure,
)
from .after_proto_monitor import (
    MONITOR_SCHEMA,
    AfterProtoMonitorResult,
    ProtoOffloadCheck,
    PROTO_OFFLOAD_ENDPOINT,
    PROTO_FLOOR_GIB,
    monitor_after_proto,
    validate_proto_offload_receipt,
)
from .signals import HostSignals, collect_host_signals

__all__ = [
    "AutoDeleteDecision",
    "AuthoritySource",
    "GovernorAction",
    "HostSignals",
    "NotificationEvent",
    "NotificationKind",
    "ObjectClass",
    "ObjectRecord",
    "PressureGovernor",
    "PressureLevel",
    "PressureSample",
    "AfterProtoMonitorResult",
    "ProtoOffloadCheck",
    "monitor_after_proto",
    "validate_proto_offload_receipt",
    "PROTO_OFFLOAD_ENDPOINT",
    "PROTO_FLOOR_GIB",
    "MONITOR_SCHEMA",
    "build_notification",
    "classify_object",
    "collect_host_signals",
    "evaluate_auto_delete",
    "evaluate_pressure",
    "may_declare_completion",
    "never_auto_delete_reason",
]
