"""Pressure governor — GREEN / YELLOW / RED / CRITICAL (bible §27).

State machine driven by real host signals (see ``signals.py``). Tonight's
operators already proved the signal path:

- ``v0_notifier.free_gib`` / disk-floor alerts (``FLOOR_GIB = 25``)
- ``v0_notifier.gpu_pct`` via ``ioreg`` IOAccelerator Device Utilization
- ``lab.operators.glm52_grounding.parse_darwin_memory`` via vm_stat + sysctl
- ``lab.operators.bounded_cache.available_ram_bytes`` / ``free_disk_bytes``
- ``reclaim_storage_keep_proto`` free-space reporting after reclaim

This module does not start daemons, unload models, or delete files. It only
classifies pressure and recommends the action set the bible specifies so a
future supervisor can wire them after Proto-Frankenstein offload.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from .signals import HostSignals


class PressureLevel(str, enum.Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return {
            PressureLevel.GREEN: 0,
            PressureLevel.YELLOW: 1,
            PressureLevel.RED: 2,
            PressureLevel.CRITICAL: 3,
        }[self]


# Bible §27 action vocabulary per level.
_ACTIONS: dict[PressureLevel, tuple[str, ...]] = {
    PressureLevel.GREEN: (
        "full_campaign",
        "normal_downloads",
        "normal_residency",
        "normal_concurrency",
    ),
    PressureLevel.YELLOW: (
        "reduce_sessions",
        "batch_reviews",
        "pause_new_large_downloads",
        "shrink_kv_session_count",
    ),
    PressureLevel.RED: (
        "checkpoint_models",
        "unload_reviewer",
        "evict_leased_cache",
        "pause_heavy_benchmark",
        "preserve_active_receipt",
    ),
    PressureLevel.CRITICAL: (
        "unload_target_model",
        "stop_downloads",
        "preserve_rollback_and_evidence",
        "return_resources_to_user",
        "emit_urgent_report",
    ),
}


@dataclass(frozen=True)
class GovernorThresholds:
    """Escalation thresholds. Disk values in GiB; ratios in [0, 1].

    Defaults mirror tonight's operational floors (disk floor ~25 GiB warn band,
    GPU contention high when sustained near full device util).
    """

    # Disk free GiB
    disk_yellow_gib: float = 40.0
    disk_red_gib: float = 25.0
    disk_critical_gib: float = 12.0
    # Available RAM GiB
    ram_yellow_gib: float = 12.0
    ram_red_gib: float = 6.0
    ram_critical_gib: float = 2.5
    # Swap used GiB
    swap_yellow_gib: float = 2.0
    swap_red_gib: float = 8.0
    swap_critical_gib: float = 16.0
    # RAM pressure ratio (used/total)
    ram_ratio_yellow: float = 0.80
    ram_ratio_red: float = 0.90
    ram_ratio_critical: float = 0.96
    # GPU util %
    gpu_yellow_pct: int = 85
    gpu_red_pct: int = 95
    # Hysteresis: require this many consecutive samples before leaving a level
    # upward is immediate on worst signal; downward needs dwell samples.
    deescalate_dwell: int = 2


@dataclass(frozen=True)
class PressureSample:
    """One evaluation of signals → level contribution + reasons."""

    level: PressureLevel
    reasons: tuple[str, ...]
    signals: Mapping[str, float | int | bool | None]
    actions: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "level": self.level.value,
            "reasons": list(self.reasons),
            "signals": dict(self.signals),
            "actions": list(self.actions),
        }


@dataclass
class GovernorAction:
    """Recommended campaign response for the current level."""

    level: PressureLevel
    previous_level: PressureLevel
    actions: tuple[str, ...]
    reasons: tuple[str, ...]
    changed: bool
    sample: PressureSample

    def as_dict(self) -> dict:
        return {
            "level": self.level.value,
            "previous_level": self.previous_level.value,
            "actions": list(self.actions),
            "reasons": list(self.reasons),
            "changed": self.changed,
            "sample": self.sample.as_dict(),
        }


def _level_from_disk(free_gib: float, th: GovernorThresholds) -> tuple[PressureLevel, str]:
    if free_gib < th.disk_critical_gib:
        return PressureLevel.CRITICAL, f"disk free {free_gib:.1f} GiB < critical {th.disk_critical_gib}"
    if free_gib < th.disk_red_gib:
        return PressureLevel.RED, f"disk free {free_gib:.1f} GiB < red {th.disk_red_gib}"
    if free_gib < th.disk_yellow_gib:
        return PressureLevel.YELLOW, f"disk free {free_gib:.1f} GiB < yellow {th.disk_yellow_gib}"
    return PressureLevel.GREEN, f"disk free {free_gib:.1f} GiB ok"


def _level_from_ram(avail_gib: float, ratio: float, th: GovernorThresholds) -> tuple[PressureLevel, str]:
    if avail_gib < th.ram_critical_gib or ratio >= th.ram_ratio_critical:
        return (
            PressureLevel.CRITICAL,
            f"ram avail {avail_gib:.2f} GiB / pressure {ratio:.2%} critical",
        )
    if avail_gib < th.ram_red_gib or ratio >= th.ram_ratio_red:
        return PressureLevel.RED, f"ram avail {avail_gib:.2f} GiB / pressure {ratio:.2%} red"
    if avail_gib < th.ram_yellow_gib or ratio >= th.ram_ratio_yellow:
        return PressureLevel.YELLOW, f"ram avail {avail_gib:.2f} GiB / pressure {ratio:.2%} yellow"
    return PressureLevel.GREEN, f"ram avail {avail_gib:.2f} GiB ok"


def _level_from_swap(swap_used_gib: float, th: GovernorThresholds) -> tuple[PressureLevel, str]:
    if swap_used_gib >= th.swap_critical_gib:
        return PressureLevel.CRITICAL, f"swap used {swap_used_gib:.1f} GiB critical"
    if swap_used_gib >= th.swap_red_gib:
        return PressureLevel.RED, f"swap used {swap_used_gib:.1f} GiB red"
    if swap_used_gib >= th.swap_yellow_gib:
        return PressureLevel.YELLOW, f"swap used {swap_used_gib:.1f} GiB yellow"
    return PressureLevel.GREEN, f"swap used {swap_used_gib:.1f} GiB ok"


def _level_from_gpu(gpu_pct: Optional[int], th: GovernorThresholds) -> tuple[PressureLevel, str]:
    if gpu_pct is None:
        return PressureLevel.GREEN, "gpu util unknown (non-escalating)"
    if gpu_pct >= th.gpu_red_pct:
        return PressureLevel.RED, f"gpu util {gpu_pct}% red (contention)"
    if gpu_pct >= th.gpu_yellow_pct:
        return PressureLevel.YELLOW, f"gpu util {gpu_pct}% yellow"
    return PressureLevel.GREEN, f"gpu util {gpu_pct}% ok"


def _level_from_thermal(thermal: Optional[bool]) -> tuple[PressureLevel, str]:
    if thermal is True:
        return PressureLevel.RED, "thermal throttling active"
    if thermal is False:
        return PressureLevel.GREEN, "thermal ok"
    return PressureLevel.GREEN, "thermal unknown (non-escalating)"


def _level_from_foreground(fg: Optional[bool]) -> tuple[PressureLevel, str]:
    """Foreground user activity nudges to YELLOW (shrink concurrency), not RED."""
    if fg is True:
        return PressureLevel.YELLOW, "foreground user activity — yield resources"
    return PressureLevel.GREEN, "no foreground-user yield requested"


def evaluate_pressure(
    signals: HostSignals,
    *,
    thresholds: Optional[GovernorThresholds] = None,
) -> PressureSample:
    """Map a host snapshot to the worst-of pressure level (bible §27 respond-to list).

    Inputs covered:
      unified-memory pressure, swap, disk floor, thermal throttling,
      GPU contention, foreground user activity.
    """
    th = thresholds or GovernorThresholds()
    contribs: list[tuple[PressureLevel, str]] = [
        _level_from_disk(signals.free_disk_gib, th),
        _level_from_ram(signals.available_ram_gib, signals.ram_pressure_ratio, th),
        _level_from_swap(signals.swap_used_gib, th),
        _level_from_gpu(signals.gpu_util_pct, th),
        _level_from_thermal(signals.thermal_throttling),
        _level_from_foreground(signals.foreground_user_active),
    ]
    worst = max(contribs, key=lambda c: c[0].rank)
    level = worst[0]
    reasons = tuple(r for lv, r in contribs if lv.rank >= level.rank and lv is not PressureLevel.GREEN)
    if not reasons:
        reasons = ("all signals GREEN",)
    return PressureSample(
        level=level,
        reasons=reasons,
        signals={
            "free_disk_gib": round(signals.free_disk_gib, 3),
            "available_ram_gib": round(signals.available_ram_gib, 3),
            "ram_pressure_ratio": round(signals.ram_pressure_ratio, 4),
            "swap_used_gib": round(signals.swap_used_gib, 3),
            "gpu_util_pct": signals.gpu_util_pct,
            "thermal_throttling": signals.thermal_throttling,
            "foreground_user_active": signals.foreground_user_active,
        },
        actions=_ACTIONS[level],
    )


@dataclass
class PressureGovernor:
    """Stateful governor with upward-immediate / downward-dwell hysteresis.

    Prevents flapping around a threshold (same lesson as SpecGovernor in
    ``crates/hawking-speculate/src/governor.rs``).
    """

    thresholds: GovernorThresholds = field(default_factory=GovernorThresholds)
    level: PressureLevel = PressureLevel.GREEN
    _pending_level: Optional[PressureLevel] = field(default=None, repr=False)
    _dwell_count: int = field(default=0, repr=False)
    history: list[PressureLevel] = field(default_factory=list)

    def step(self, signals: HostSignals) -> GovernorAction:
        sample = evaluate_pressure(signals, thresholds=self.thresholds)
        previous = self.level
        proposed = sample.level

        if proposed.rank > self.level.rank:
            # Escalate immediately on worst signal.
            self.level = proposed
            self._pending_level = None
            self._dwell_count = 0
        elif proposed.rank < self.level.rank:
            # De-escalate only after consecutive lower samples.
            if self._pending_level == proposed:
                self._dwell_count += 1
            else:
                self._pending_level = proposed
                self._dwell_count = 1
            if self._dwell_count >= self.thresholds.deescalate_dwell:
                self.level = proposed
                self._pending_level = None
                self._dwell_count = 0
        else:
            self._pending_level = None
            self._dwell_count = 0

        self.history.append(self.level)
        return GovernorAction(
            level=self.level,
            previous_level=previous,
            actions=_ACTIONS[self.level],
            reasons=sample.reasons,
            changed=self.level is not previous,
            sample=sample,
        )

    def actions_for(self, level: Optional[PressureLevel] = None) -> Sequence[str]:
        return _ACTIONS[level or self.level]
