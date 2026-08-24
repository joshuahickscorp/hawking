"""Obligation ledger and completion governor for HCLI AgentOS.

This layer sits ABOVE WorkUnit scheduling. It answers "is the mission's
overall goal satisfied?" (a small number of falsifiable propositions, each
with its own verify command). It does not answer "what task runs next" —
that is ``workunit.py`` + ``dag_store.py`` + ``scheduler.py``. ``goal.py``'s
``WorkUnitDAG`` is GoalCompiler IR, not this ledger.

An empty ledger is not a satisfied goal. It is a missing one
(``status == EMPTY_LEDGER``). Only ``VERIFIED`` satisfies an obligation, and
only when the checkbox and the status field agree. A parser bug that lets
``[x]`` and ``status: VERIFIED`` disagree is a real defect: ``is_goal_met``
is false unless both are set.

An obligation is never marked VERIFIED by a model's say-so. ``mark_verified``
requires a fresh ``VerifyResult`` produced by ``run_verify`` with
``passed=True``. Fabricated bools, strings, or hand-built ``VerifyResult``
values are rejected.

==========================================================================
How mission.py (once it exists) is expected to call this
==========================================================================

The mission loop must treat the ledger as the sole completion predicate.
Hold one ``Ledger`` (parsed from the mission's GOAL.md, or built via
``add``). Wiring is one call at the point the loop would otherwise report
success — the internal equivalent of a harness Stop-hook gate — and that
call must not be caught-and-ignored:

    from hcli.ledger import Ledger, GoalNotMetError, NO_PROGRESS_THRESHOLD
    from hcli.steering import SteeringQueue

    ledger = Ledger.parse(goal_md_path)   # or the in-memory Ledger the
                                          # mission already holds
    queue = SteeringQueue(workspace, session_id)

    # On each operator steer:
    event = queue.enqueue(text, kind=kind)  # knowledge|correction|constraint
    if event.kind == "constraint":
        queue.apply_constraint(event, ledger)
    # knowledge / correction: recorded on the queue only. Do not call
    # apply_constraint (it raises SteerKindError). They never change what
    # the ledger considers done.

    # After work on an obligation, verify empirically — never by say-so:
    result = ledger.run_verify(obligation_id)
    if result.passed:
        ledger.mark_verified(obligation_id, result)
    # mark_verified writes the verify-receipt sidecar and, when the
    # ledger has a path, persists GOAL.md in the same step so a restart
    # sees VERIFIED together with the evidence that authorised it.

    # Immediately before reporting mission success:
    ledger.assert_may_complete()

``assert_may_complete()`` returns cleanly only when every obligation is
VERIFIED (checkbox and status agreeing). It uses ``outcome()`` and raises
a specific exception per non-``GOAL_MET`` case so a caller can distinguish
"keep working" (``GoalNotMetError``) from "genuinely stuck, human input
needed" (``TerminalBlockerError``) from "the ledger itself is broken"
(``EnforcementFailureError``) from a safety ceiling (``SafetyDisarmError``).

``outcome()`` is the five-way governor matching ultragoal-stop.mjs:
``GOAL_MET``, ``GOAL_NOT_MET``, ``TERMINAL_BLOCKER``, ``SAFETY_DISARM``,
``ENFORCEMENT_FAILURE``. Only ``GOAL_MET`` is completion. ``TERMINAL_BLOCKER``
is explicitly not ``GOAL_MET``. ``SAFETY_DISARM`` is requested by the
caller via ``outcome(budget_exceeded=True)`` — this module does not track
wall-clock or turn count. The older ``status`` property is unchanged
(``GOAL_MET`` / ``GOAL_NOT_MET`` / ``EMPTY_LEDGER``) so existing callers
keep working; new code should use ``outcome()``.

No-progress uses two hashes (full ledger text, and only lines matching
``evidence:``) matching the hook. Stalled requires BOTH to match the
previous check. ``watchdog_tier()`` / ``watchdog_message()`` map the
escalating 4-tier ladder. Inspect ``consecutive_no_progress_count()``
and ``watchdog_tier()``. Response to hitting a tier belongs to mission.py.

A steer never marks anything VERIFIED. Only ``run_verify`` +
``mark_verified`` can. ``apply_constraint`` amends THIS ledger and, when
the ledger was parsed from a path, persists so a re-parse of the same
file sees the change.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .persist import atomic_write_text as _atomic_write_text


VALID_STATUSES = (
    "PENDING",
    "READY",
    "ACTIVE",
    "BLOCKED",
    "FAILED",
    "VERIFIED",
    "STALE",
)

GOAL_MET = "GOAL_MET"
GOAL_NOT_MET = "GOAL_NOT_MET"
EMPTY_LEDGER = "EMPTY_LEDGER"
TERMINAL_BLOCKER = "TERMINAL_BLOCKER"
SAFETY_DISARM = "SAFETY_DISARM"
ENFORCEMENT_FAILURE = "ENFORCEMENT_FAILURE"

# After this many consecutive identical run_verify fingerprints, the driver
# (future mission.py) must replan. This module only makes the count observable.
# The hook's ladder is 4 tiers; this threshold remains the historic "replan"
# cue (watchdog_tier() == 2 / L2 MANDATORY REPLAN).
NO_PROGRESS_THRESHOLD = 3
DEFAULT_VERIFY_TIMEOUT = 30.0

_VERIFY_TOKEN = object()

RECEIPT_SOURCE = "Ledger.run_verify"
_RECEIPT_DIRNAME = ".hcli"
_RECEIPT_SUBDIR = "verify-receipts"

_HEADER_RE = re.compile(
    r"^-\s+\[([ xX])\]\s+(G\d+)\s+[\u2014\u2013-]+\s+(.*)$"
)
_FIELD_RE = re.compile(
    r"^(\s+)(acceptance|verify|evidence)\s*:\s*(.*)$",
    re.IGNORECASE,
)
# Mirror ultragoal-stop.mjs TERMINAL_BLOCKER = /^[ \t]*terminal_blocker:[ \t]*(.+)$/im
_TERMINAL_BLOCKER_RE = re.compile(
    r"^[ \t]*terminal_blocker:[ \t]*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_EVIDENCE_LINE_RE = re.compile(r"evidence:", re.IGNORECASE)

# Exact strings from ~/.claude/hooks/ultragoal-stop.mjs WATCHDOG (lines 357-369).
WATCHDOG_MESSAGES = {
    0: "",
    1: (
        "WATCHDOG L1 — no material change since the last block. "
        "Reconsider the hypothesis before acting again."
    ),
    2: (
        "WATCHDOG L2 — MANDATORY REPLAN. Two blocks with no material change. "
        "Do not retry the same approach. State what you now believe is wrong, "
        "change the hypothesis or the method, and record the change in the ledger."
    ),
    3: (
        "WATCHDOG L3 — CHANGE STRATEGY OR ESCALATE. Three blocks with no material "
        "change. Escalate the verification tier, change abstraction level, or "
        "escalate this obligation per the capability governor. Persistence means "
        "refusing to abandon the goal, not refusing to abandon a failing strategy."
    ),
}
WATCHDOG_L4 = (
    "WATCHDOG L4 — ROOT CAUSE OR BLOCKER PROOF. Repeated no-progress blocks "
    "after escalation. Stop retrying. Either establish the root cause with "
    "evidence, or prove a terminal blocker per ULTRA CORE §11 and record it "
    "in the ledger as `terminal_blocker:` with every remaining obligation "
    "marked `status: BLOCKED`."
)


class GoalNotMetError(Exception):
    """Raised by ``Ledger.assert_may_complete`` when the goal is not met.

    Attributes:
        unverified: list of ``(obligation_id, status)`` for every obligation
            that is not both checkbox-checked and ``status == VERIFIED``.
            Empty when the ledger itself is empty (a missing goal).
        ledger_status: ``GOAL_NOT_MET`` (keep working the frontier).
    """

    def __init__(
        self,
        unverified: Optional[Iterable[Tuple[str, str]]] = None,
        ledger_status: str = GOAL_NOT_MET,
    ) -> None:
        self.unverified = list(unverified or [])
        self.ledger_status = ledger_status
        ids = ", ".join(
            f"{oid}={status}" for oid, status in self.unverified
        ) or "(none)"
        super().__init__(
            f"goal is not met ({ledger_status}); unverified: {ids}"
        )


class TerminalBlockerError(Exception):
    """Raised when the ledger records a terminal blocker. This is NOT GOAL_MET."""

    def __init__(
        self,
        blocker: str,
        unverified: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> None:
        self.blocker = blocker
        self.unverified = list(unverified or [])
        self.ledger_status = TERMINAL_BLOCKER
        super().__init__(
            f"TERMINAL_BLOCKER — blocked on: {blocker}. This is NOT GOAL_MET."
        )


class SafetyDisarmError(Exception):
    """Raised when the caller reports the safety budget is exhausted.

    This is NOT completion. Ledger does not track wall-clock or turn count;
    the driver passes ``budget_exceeded=True``.
    """

    def __init__(
        self,
        unverified: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> None:
        self.unverified = list(unverified or [])
        self.ledger_status = SAFETY_DISARM
        super().__init__(
            "SAFETY_DISARM — budget exceeded. This is NOT completion."
        )


class EnforcementFailureError(Exception):
    """Raised when the ledger is missing, unreadable, or empty."""

    def __init__(
        self,
        unverified: Optional[Iterable[Tuple[str, str]]] = None,
        detail: str = "",
    ) -> None:
        self.unverified = list(unverified or [])
        self.ledger_status = ENFORCEMENT_FAILURE
        super().__init__(
            detail
            or (
                "ENFORCEMENT_FAILURE — ledger missing, unreadable, or empty. "
                "An empty ledger is not a satisfied goal."
            )
        )


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of ``Ledger.run_verify``.

    Only instances produced by ``run_verify`` are fresh. Hand-built values
    (and bare bools/strings) are rejected by ``mark_verified``.
    """

    passed: bool
    output: str
    exit_code: int
    obligation_id: str = ""
    _token: object = field(default=None, repr=False, compare=False)

    @classmethod
    def _fresh(
        cls,
        *,
        passed: bool,
        output: str,
        exit_code: int,
        obligation_id: str = "",
    ) -> "VerifyResult":
        return cls(
            passed=passed,
            output=output,
            exit_code=exit_code,
            obligation_id=obligation_id,
            _token=_VERIFY_TOKEN,
        )

    def is_fresh(self) -> bool:
        return self._token is _VERIFY_TOKEN


@dataclass
class Obligation:
    id: str
    text: str
    status: str = "PENDING"
    tier: str = "V2"
    acceptance: str = ""
    verify_command: str = ""
    evidence: str = "(none yet)"
    risk: str = "high"
    checked: bool = False


def _split_meta(rest: str) -> Tuple[str, Dict[str, str]]:
    parts = [part.strip() for part in rest.split("|")]
    text_part = parts[0] if parts else ""
    meta: Dict[str, str] = {}
    for part in parts[1:]:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        meta[key.strip().lower()] = value.strip()
    return text_part, meta


def _parse_obligation_at(lines: List[str], index: int) -> Tuple[Obligation, int]:
    match = _HEADER_RE.match(lines[index])
    if match is None:
        raise ValueError(f"not an obligation header: {lines[index]!r}")
    checked = match.group(1).lower() == "x"
    oid = match.group(2)
    text_part, meta = _split_meta(match.group(3).strip())
    acceptance = ""
    verify = ""
    evidence = "(none yet)"
    current: Optional[str] = None
    index += 1
    while index < len(lines):
        nxt = lines[index]
        if _HEADER_RE.match(nxt):
            break
        field_match = _FIELD_RE.match(nxt)
        if field_match:
            current = field_match.group(2).lower()
            value = field_match.group(3)
            if current == "acceptance":
                acceptance = value
            elif current == "verify":
                verify = value
            elif current == "evidence":
                evidence = value
            index += 1
            continue
        if nxt.strip() == "":
            break
        if nxt.startswith((" ", "\t")) and current:
            extra = nxt.strip()
            if current == "acceptance":
                acceptance = f"{acceptance}\n{extra}" if acceptance else extra
            elif current == "verify":
                verify = f"{verify}\n{extra}" if verify else extra
            elif current == "evidence":
                evidence = f"{evidence}\n{extra}" if evidence else extra
            index += 1
            continue
        break
    status = meta.get("status", "PENDING").upper()
    return (
        Obligation(
            id=oid,
            text=text_part,
            status=status,
            tier=meta.get("tier", "V2"),
            acceptance=acceptance,
            verify_command=verify,
            evidence=evidence,
            risk=meta.get("risk", "high"),
            checked=checked,
        ),
        index,
    )


def _format_obligation(ob: Obligation) -> str:
    box = "x" if ob.checked else " "
    return (
        f"- [{box}] {ob.id} — {ob.text} | status: {ob.status} | "
        f"risk: {ob.risk} | tier: {ob.tier}\n"
        f"      acceptance: {ob.acceptance}\n"
        f"      verify: {ob.verify_command}\n"
        f"      evidence: {ob.evidence}"
    )


class Ledger:
    """Mission-level obligation ledger. See module docstring for mission.py."""

    def __init__(
        self,
        obligations: Optional[Iterable[Obligation]] = None,
    ) -> None:
        self._items: Dict[str, Obligation] = {}
        self._order: List[str] = []
        self._source: Optional[str] = None
        self._dirty: bool = False
        self._preamble: str = ""
        self._epilogue: str = ""
        self._path: Optional[Path] = None
        self.terminal_blocker: Optional[str] = None
        self._last_ledger_hash: Optional[str] = None
        self._last_evidence_hash: Optional[str] = None
        self._no_progress_count: int = 0
        self._enforcement_failure: bool = False
        self._memory_receipts: Dict[str, Dict[str, object]] = {}
        if obligations:
            for ob in obligations:
                self._put(ob, dirty=True)

    @classmethod
    def parse(cls, path: Union[str, Path]) -> "Ledger":
        path = Path(path)
        with open(path, "r", encoding="utf-8", newline="") as handle:
            text = handle.read()
        return cls.from_markdown(text, path=path)

    @classmethod
    def from_markdown(
        cls,
        text: str,
        *,
        path: Optional[Union[str, Path]] = None,
    ) -> "Ledger":
        ledger = cls()
        ledger._source = text
        ledger._path = Path(path) if path is not None else None
        blocker = _TERMINAL_BLOCKER_RE.search(text)
        if blocker:
            ledger.terminal_blocker = blocker.group(1).strip()
        lines = text.splitlines()
        first_header: Optional[int] = None
        last_end = 0
        index = 0
        while index < len(lines):
            if not _HEADER_RE.match(lines[index]):
                index += 1
                continue
            if first_header is None:
                first_header = index
            ob, index = _parse_obligation_at(lines, index)
            last_end = index
            if ob.status == "VERIFIED" and not ledger._receipt_is_fresh(ob):
                # On-disk VERIFIED is not evidence. Only a sidecar written by
                # run_verify + mark_verified keeps the status.
                #
                # STALE, not PENDING: both are unsatisfied, so a hand-forged
                # VERIFIED still cannot produce GOAL_MET, but STALE preserves
                # the difference between "never attempted" and "was verified,
                # evidence not available here". Rewriting it to PENDING throws
                # that away and makes a moved ledger indistinguishable from a
                # fresh one.
                ob.status = "STALE"
                ob.checked = False
            ledger._put(ob, dirty=False)
        if first_header is not None:
            preamble = "\n".join(lines[:first_header])
            ledger._preamble = (preamble + "\n") if preamble else ""
            epilogue = "\n".join(lines[last_end:])
            ledger._epilogue = (epilogue + "\n") if epilogue else ""
        else:
            ledger._preamble = text
        ledger._dirty = False
        return ledger

    def to_markdown(self) -> str:
        if not self._dirty and self._source is not None:
            return self._source
        return self._serialize()

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Write this ledger to disk. Defaults to the path it was parsed from.

        Receipts are flushed first, then GOAL.md. A restart that sees
        ``status: VERIFIED`` therefore also sees the sidecar that
        ``parse`` requires; a crash between the two leaves PENDING plus
        an unused receipt, never a forged VERIFIED.
        """
        dest = Path(path) if path is not None else self._path
        if dest is None:
            raise ValueError("Ledger.save requires a path")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        previous = self._path
        self._path = dest
        try:
            self._flush_memory_receipts()
            text = self.to_markdown()
            _atomic_write_text(dest, text)
        except Exception:
            self._path = previous
            raise
        self._source = text
        self._dirty = False
        return dest

    def apply_constraint(self, event: Any, queue: Any = None) -> None:
        """Amend THIS ledger from a constraint steer and persist if a path is known.

        Delegates the amendment to ``SteeringQueue`` so there is one
        mutation implementation. Re-parsing the original path then sees
        the change. knowledge/correction raise ``SteerKindError``.
        """
        from .steering import SteeringQueue, SteerKindError

        kind = getattr(event, "kind", None)
        if kind != "constraint":
            raise SteerKindError(
                f"apply_constraint requires kind='constraint', got {kind!r}"
            )
        if queue is not None:
            queue.apply_constraint(event, self)
        else:
            dummy = SteeringQueue.__new__(SteeringQueue)
            SteeringQueue._amend_ledger(dummy, event, self)
        if self._path is not None:
            self.save()

    def add(
        self,
        text: str,
        *,
        acceptance: str = "",
        verify_command: str = "",
        risk: str = "high",
        tier: str = "V2",
        obligation_id: Optional[str] = None,
        status: str = "PENDING",
    ) -> Obligation:
        status = (status or "PENDING").upper()
        if status == "VERIFIED":
            raise ValueError(
                "VERIFIED requires mark_verified with a passing VerifyResult"
            )
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status {status!r}")
        oid = obligation_id or self._next_id()
        if oid in self._items:
            raise ValueError(f"duplicate obligation id: {oid}")
        ob = Obligation(
            id=oid,
            text=text,
            status=status,
            tier=tier,
            acceptance=acceptance,
            verify_command=verify_command,
            evidence="(none yet)",
            risk=risk,
            checked=False,
        )
        self._put(ob, dirty=True)
        return ob

    def get(self, obligation_id: str) -> Obligation:
        try:
            return self._items[obligation_id]
        except KeyError:
            raise KeyError(f"unknown obligation {obligation_id}") from None

    def obligations(self) -> List[Obligation]:
        return [self._items[oid] for oid in self._order]

    def remove(self, obligation_id: str) -> Obligation:
        ob = self.get(obligation_id)
        del self._items[obligation_id]
        self._order.remove(obligation_id)
        self._dirty = True
        return ob

    def replace_text(self, obligation_id: str, text: str) -> Obligation:
        ob = self.get(obligation_id)
        ob.text = text
        self._dirty = True
        return ob

    def mark_status(self, obligation_id: str, status: str) -> Obligation:
        status = (status or "").upper()
        if status == "VERIFIED":
            raise ValueError(
                "VERIFIED requires mark_verified with a passing VerifyResult"
            )
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status {status!r}")
        ob = self.get(obligation_id)
        ob.status = status
        ob.checked = False
        self._dirty = True
        return ob

    def mark_verified(self, obligation_id: str, evidence: VerifyResult) -> None:
        if not isinstance(evidence, VerifyResult):
            raise TypeError(
                "mark_verified requires a VerifyResult from run_verify, "
                f"got {type(evidence).__name__}"
            )
        if not evidence.is_fresh():
            raise TypeError(
                "VerifyResult must be produced by Ledger.run_verify; "
                "fabricated results are rejected"
            )
        if not evidence.passed:
            raise ValueError("cannot mark VERIFIED from a failing VerifyResult")
        if evidence.obligation_id and evidence.obligation_id != obligation_id:
            raise ValueError(
                f"VerifyResult is for {evidence.obligation_id}, not {obligation_id}"
            )
        ob = self.get(obligation_id)
        previous_status = ob.status
        previous_checked = ob.checked
        previous_evidence = ob.evidence
        previous_dirty = self._dirty
        # Receipt first: parse() treats on-disk VERIFIED without a sidecar
        # as STALE. Evidence without a VERIFIED checkbox is a lost
        # verification, not a forged one.
        self._write_receipt(ob, evidence)
        ob.status = "VERIFIED"
        ob.checked = True
        if evidence.output.strip():
            ob.evidence = evidence.output
        elif not ob.evidence or ob.evidence == "(none yet)":
            ob.evidence = f"passed (exit {evidence.exit_code})"
        self._dirty = True
        if self._path is not None:
            try:
                self.save()
            except Exception:
                ob.status = previous_status
                ob.checked = previous_checked
                ob.evidence = previous_evidence
                self._dirty = previous_dirty
                raise

    def is_goal_met(self) -> bool:
        obs = self.obligations()
        if not obs:
            return False
        return all(self._is_satisfied(ob) for ob in obs)

    @property
    def status(self) -> str:
        if not self._order:
            return EMPTY_LEDGER
        if self.is_goal_met():
            return GOAL_MET
        return GOAL_NOT_MET

    def outcome(self, budget_exceeded: bool = False) -> str:
        """Five-way governor matching ultragoal-stop.mjs.

        Order matches the hook: empty/unreadable → ENFORCEMENT_FAILURE;
        every obligation VERIFIED (checkbox and status) → GOAL_MET;
        ``terminal_blocker:`` set AND every unresolved obligation is
        BLOCKED → TERMINAL_BLOCKER (this is NOT GOAL_MET);
        ``budget_exceeded=True`` → SAFETY_DISARM (caller-supplied; Ledger
        has no wall-clock or turn count);
        otherwise GOAL_NOT_MET.
        """
        if self._enforcement_failure:
            return ENFORCEMENT_FAILURE
        obs = self.obligations()
        if not obs:
            return ENFORCEMENT_FAILURE
        if self.is_goal_met():
            return GOAL_MET
        unresolved = [ob for ob in obs if not self._is_satisfied(ob)]
        blocker = (self.terminal_blocker or "").strip()
        if blocker and all(ob.status == "BLOCKED" for ob in unresolved):
            return TERMINAL_BLOCKER
        if budget_exceeded:
            return SAFETY_DISARM
        return GOAL_NOT_MET

    def unverified(self) -> List[Tuple[str, str]]:
        return [
            (ob.id, ob.status)
            for ob in self.obligations()
            if not self._is_satisfied(ob)
        ]

    def run_verify(
        self,
        obligation_id: str,
        timeout: Optional[float] = None,
    ) -> VerifyResult:
        if timeout is None:
            timeout = DEFAULT_VERIFY_TIMEOUT
        ob = self.get(obligation_id)
        cmd = (ob.verify_command or "").strip()
        if not cmd:
            result = VerifyResult._fresh(
                passed=False,
                output="verify command is missing or empty",
                exit_code=-1,
                obligation_id=obligation_id,
            )
            self._apply_verify_outcome(ob, result)
            return result
        from .verifier_pipeline import command_is_admissible

        admitted, refuse_reason = command_is_admissible(cmd)
        if not admitted:
            result = VerifyResult._fresh(
                passed=False,
                output=refuse_reason,
                exit_code=-1,
                obligation_id=obligation_id,
            )
            self._apply_verify_outcome(ob, result)
            return result
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            result = VerifyResult._fresh(
                passed=(proc.returncode == 0),
                output=output,
                exit_code=int(proc.returncode),
                obligation_id=obligation_id,
            )
        except subprocess.TimeoutExpired as exc:
            chunks = [f"TIMEOUT after {timeout}s"]
            if exc.stdout:
                stdout = (
                    exc.stdout
                    if isinstance(exc.stdout, str)
                    else exc.stdout.decode("utf-8", "replace")
                )
                chunks.append(stdout)
            if exc.stderr:
                stderr = (
                    exc.stderr
                    if isinstance(exc.stderr, str)
                    else exc.stderr.decode("utf-8", "replace")
                )
                chunks.append(stderr)
            result = VerifyResult._fresh(
                passed=False,
                output="\n".join(chunks),
                exit_code=-1,
                obligation_id=obligation_id,
            )
        except Exception as exc:
            result = VerifyResult._fresh(
                passed=False,
                output=f"{type(exc).__name__}: {exc}",
                exit_code=-1,
                obligation_id=obligation_id,
            )
        self._apply_verify_outcome(ob, result)
        return result

    def assert_may_complete(self, budget_exceeded: bool = False) -> None:
        result = self.outcome(budget_exceeded=budget_exceeded)
        if result == GOAL_MET:
            return
        unverified = self.unverified()
        if result == TERMINAL_BLOCKER:
            raise TerminalBlockerError(
                self.terminal_blocker or "",
                unverified=unverified,
            )
        if result == SAFETY_DISARM:
            raise SafetyDisarmError(unverified=unverified)
        if result == ENFORCEMENT_FAILURE:
            raise EnforcementFailureError(unverified=unverified)
        raise GoalNotMetError(unverified, ledger_status=GOAL_NOT_MET)

    def consecutive_no_progress_count(self) -> int:
        return self._no_progress_count

    def observe_progress(self) -> int:
        """Record current ledger+evidence hashes. Returns the stall count."""
        self._tick_watchdog()
        return self._no_progress_count

    def watchdog_tier(self) -> int:
        """0 = no stall, 1-4 mapping to the ultragoal-stop.mjs ladder."""
        n = self._no_progress_count
        if n <= 0:
            return 0
        return 4 if n >= 4 else n

    def watchdog_message(self) -> str:
        """Exact ultragoal-stop.mjs WATCHDOG instruction for the current tier."""
        tier = self.watchdog_tier()
        if tier >= 4:
            return WATCHDOG_L4
        return WATCHDOG_MESSAGES.get(tier, "")

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self):
        return iter(self.obligations())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ledger):
            return NotImplemented
        return self._snapshot() == other._snapshot()

    def _is_satisfied(self, ob: Obligation) -> bool:
        return ob.status == "VERIFIED" and ob.checked

    def _next_id(self) -> str:
        n = 1
        while f"G{n:03d}" in self._items:
            n += 1
        return f"G{n:03d}"

    def _put(self, ob: Obligation, *, dirty: bool) -> None:
        if ob.id in self._items:
            raise ValueError(f"duplicate obligation id: {ob.id}")
        self._items[ob.id] = ob
        self._order.append(ob.id)
        if dirty:
            self._dirty = True

    def _serialize(self) -> str:
        body = "\n".join(
            _format_obligation(ob) for ob in self.obligations()
        )
        preamble = self._preamble
        if self.terminal_blocker:
            combined = (preamble or "") + (self._epilogue or "")
            if not _TERMINAL_BLOCKER_RE.search(combined):
                line = f"terminal_blocker: {self.terminal_blocker}\n"
                if preamble:
                    if not preamble.endswith("\n"):
                        preamble += "\n"
                    preamble = preamble + line
                else:
                    preamble = line
        if preamble or self._epilogue:
            parts: List[str] = []
            if preamble:
                parts.append(preamble.rstrip("\n"))
            if body:
                parts.append(body)
            if self._epilogue.strip():
                parts.append(self._epilogue.strip("\n"))
            result = "\n".join(parts)
            if self._source is not None and self._source.endswith("\n"):
                if not result.endswith("\n"):
                    result += "\n"
            return result
        if body:
            return body + "\n"
        return ""

    def _snapshot(self) -> Tuple[object, ...]:
        return (
            self.terminal_blocker,
            tuple(
                (
                    ob.id,
                    ob.text,
                    ob.status,
                    ob.tier,
                    ob.acceptance,
                    ob.verify_command,
                    ob.evidence,
                    ob.risk,
                    ob.checked,
                )
                for ob in self.obligations()
            ),
        )

    def _evidence_text(self, text: str) -> str:
        return "\n".join(
            line for line in text.split("\n") if _EVIDENCE_LINE_RE.search(line)
        )

    def _tick_watchdog(self) -> None:
        text = self.to_markdown()
        ledger_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        evidence_hash = hashlib.sha256(
            self._evidence_text(text).encode("utf-8")
        ).hexdigest()
        stalled = (
            self._last_ledger_hash is not None
            and self._last_evidence_hash is not None
            and ledger_hash == self._last_ledger_hash
            and evidence_hash == self._last_evidence_hash
        )
        if stalled:
            self._no_progress_count += 1
        else:
            self._no_progress_count = 0
        self._last_ledger_hash = ledger_hash
        self._last_evidence_hash = evidence_hash

    def _apply_verify_outcome(
        self,
        ob: Obligation,
        result: VerifyResult,
    ) -> None:
        if result.output.strip():
            ob.evidence = result.output
        else:
            ob.evidence = f"exit {result.exit_code}"
        self._dirty = True
        self._tick_watchdog()

    def _receipt_path(self, obligation_id: str) -> Optional[Path]:
        if self._path is None:
            return None
        return (
            self._path.parent
            / _RECEIPT_DIRNAME
            / _RECEIPT_SUBDIR
            / f"{obligation_id}.json"
        )

    def _receipt_is_fresh(self, ob: Obligation) -> bool:
        record: Optional[Dict[str, object]] = None
        dest = self._receipt_path(ob.id)
        if dest is not None and dest.is_file():
            try:
                loaded = json.loads(dest.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                loaded = None
            if isinstance(loaded, dict):
                record = loaded
        if record is None:
            memory = self._memory_receipts.get(ob.id)
            if isinstance(memory, dict):
                record = memory
        if not record:
            return False
        if record.get("source") != RECEIPT_SOURCE:
            return False
        if record.get("passed") is not True:
            return False
        if record.get("obligation_id") != ob.id:
            return False
        if record.get("verify_command") != ob.verify_command:
            return False
        return True

    def _write_receipt(self, ob: Obligation, evidence: VerifyResult) -> None:
        record: Dict[str, object] = {
            "source": RECEIPT_SOURCE,
            "obligation_id": ob.id,
            "passed": True,
            "exit_code": evidence.exit_code,
            "verify_command": ob.verify_command,
            "output_sha256": hashlib.sha256(
                (evidence.output or "").encode("utf-8")
            ).hexdigest(),
        }
        self._memory_receipts[ob.id] = record
        dest = self._receipt_path(ob.id)
        if dest is None:
            return
        _atomic_write_text(
            dest,
            json.dumps(record, indent=2, sort_keys=True) + "\n",
        )

    def _flush_memory_receipts(self) -> None:
        """Write every in-memory receipt whose path is now known.

        Covers the pathless ``mark_verified`` then ``save(path)`` case:
        VERIFIED markdown must not reach disk without the sidecar.
        """
        for oid, record in list(self._memory_receipts.items()):
            dest = self._receipt_path(oid)
            if dest is None or not isinstance(record, dict):
                continue
            _atomic_write_text(
                dest,
                json.dumps(record, indent=2, sort_keys=True) + "\n",
            )
