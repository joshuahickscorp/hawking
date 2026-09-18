"""Shared AgentOS mission/work-unit state vocabulary."""
from __future__ import annotations

from enum import StrEnum
from typing import Any


class AgentState(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING_RESOURCE = "WAITING_RESOURCE"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    BLOCKED = "BLOCKED"
    FAILED_RECOVERABLE = "FAILED_RECOVERABLE"
    FAILED_TERMINAL = "FAILED_TERMINAL"


def workunit_state(unit: Any) -> AgentState:
    status = str(getattr(unit, "status", "pending") or "pending").lower()
    if status == "ready":
        return AgentState.READY
    if status == "running":
        return AgentState.RUNNING
    if status == "completed":
        verification = getattr(unit, "verification", None)
        if isinstance(verification, dict) and verification.get("ok") is False:
            return AgentState.REFUTED
        return AgentState.VERIFIED
    if status == "failed":
        if bool(getattr(unit, "repair_exhausted", False)):
            return AgentState.FAILED_TERMINAL
        return AgentState.FAILED_RECOVERABLE
    if status == "pending":
        return AgentState.NOT_STARTED
    return AgentState.INCONCLUSIVE


def mission_state(
    phase: Any,
    *,
    has_ready: bool = False,
    has_running: bool = False,
    has_failed: bool = False,
) -> AgentState:
    value = str(phase or "idle").lower()
    if value in {"idle", "not_started"}:
        return AgentState.NOT_STARTED
    if value == "running":
        if has_running:
            return AgentState.RUNNING
        if has_ready:
            return AgentState.READY
        return AgentState.WAITING_RESOURCE
    if value == "completed":
        return AgentState.INCONCLUSIVE if has_failed else AgentState.VERIFIED
    if value == "no_progress":
        return AgentState.INCONCLUSIVE
    if value == "evacuated":
        # The supervisor took the body back for memory; the mission is intact
        # and the next worker resumes it. Not a failure of any kind.
        return AgentState.WAITING_RESOURCE
    if value == "cancelled":
        return AgentState.FAILED_RECOVERABLE
    if value == "failed":
        return AgentState.BLOCKED
    return AgentState.INCONCLUSIVE


__all__ = ["AgentState", "mission_state", "workunit_state"]
