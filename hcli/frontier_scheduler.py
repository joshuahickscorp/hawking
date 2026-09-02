"""The supervisor that owns frontier switching -- the other half of the
sovereign stall guard.

The sovereign loop (tools/future/hcli_sovereign.py) can already PARK
itself: it detects a deterministic stuck state or an unproductive streak,
records a ``wake_condition``, appends a ``stop`` to
receipts/future/HCLI_MISSION_KERNEL.json, and exits. That is correct and it
is half the law -- "one stuck frontier must never stall Hawking". The other
half was missing: nothing picked up the next frontier, so a parked SUB2
just meant an idle machine.

This module is that missing half, and it is a SCHEDULER, not a second
mission system: given a set of named frontiers it answers three questions
-- which are ELIGIBLE right now, which is highest expected value, and what
to do when the running one parks -- and it never executes science, never
decides a frontier's content, and writes nothing.

Two states look identical from outside (nothing appears to be happening)
and must never be conflated -- this distinction is the entire point:

  WAITING   legitimately blocked on something external: a GPU experiment
            running, a model mid-download, a specimen's wake condition
            unmet, a protected lease held. Do not disturb; every WAITING
            state carries the specific ``wake_condition`` that lifts it
            (enforced structurally, see ``FrontierState.__post_init__``).

  STUCK     not a per-frontier state at all -- it is what happens if a
            SCHEDULER keeps re-proposing the same non-runnable frontier
            while others are ELIGIBLE. ``select_next`` is built so this is
            unreachable: it always prefers an ELIGIBLE frontier over an
            idle report, so the failure this module exists to prevent
            (reporting "waiting for instructions" while runnable work
            exists) cannot occur as long as at least one probe returns
            ELIGIBLE.

A parked frontier is SLEEPING, never dead: it is re-eligible the moment a
probe reports its wake condition met. Nothing here decides that a
frontier is finished forever.

Each named frontier is read through one small probe function grounded in a
real producer already written by that subsystem's own tooling -- no probe
invents a number:

  SUB2             receipts/future/HCLI_MISSION_KERNEL.json (the sovereign
                   loop's own kernel + its "stops" list)
  HCLI_SELF        receipts/future/HCLI_CAPABILITY_REACHABILITY.json
                   (typed_tools_dead backlog -- this campaign's own
                   recurring defect class)
  RESIDENT_SPEED   receipts/future/RESIDENT_TPS_RECON.json (last lever
                   verdict + its age)
  ODYSSEY          workspace/campaign/odyssey/patients/ (queue depth on
                   disk, the same directory hcli/odyssey.py reads)
  MODELLAKE        receipts/future/MODELLAKE_SCHEDULER_VIEW.json (the
                   watcher's own live.active_jobs / live.stale sample)
  FORBIDDEN_FRUIT  receipts/future/FORBIDDEN_FRUIT_LAB_READINESS.json
                   (smallest_missing_piece -- not live yet, per the brief)

A probe whose receipt is missing or unreadable reports UNKNOWN rather than
raising, and ``snapshot()`` catches any probe exception individually --
one broken probe must never sink the other five.

Runnable two ways::

    python3 -m pytest hcli/test_frontier_scheduler.py -q
    python3 -m hcli.frontier_scheduler
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
RECEIPTS = REPO / "receipts" / "future"
ODYSSEY_PATIENTS = REPO / "workspace" / "campaign" / "odyssey" / "patients"

# -- frontier kinds ---------------------------------------------------------
# RUNNING is part of the vocabulary but no probe below emits it: none of
# these six subsystems exposes a cheap, file-only live-process signal today
# (resident_health.py's pid-liveness check is the pattern to reuse if one is
# ever wired in). It stays a first-class kind so select_next and tests can
# exercise "leave the running one alone" against an injected frontier.
RUNNING = "RUNNING"
ELIGIBLE = "ELIGIBLE"
WAITING = "WAITING"
UNKNOWN = "UNKNOWN"
_VALID_KINDS = (RUNNING, ELIGIBLE, WAITING, UNKNOWN)

FRONTIER_NAMES = (
    "SUB2", "HCLI_SELF", "RESIDENT_SPEED", "ODYSSEY", "MODELLAKE",
    "FORBIDDEN_FRUIT",
)


@dataclass(frozen=True)
class FrontierState:
    """One frontier's eligibility, as of one probe call.

    ``wake_condition`` is mandatory on every WAITING state -- this is the
    structural guarantee that a parked frontier is sleeping, never dead.
    """

    name: str
    kind: str
    expected_value: float
    wake_condition: Optional[str]
    evidence: tuple = ()
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"{self.name}: unknown frontier kind {self.kind!r}")
        if self.kind == WAITING and not self.wake_condition:
            raise ValueError(
                f"{self.name}: WAITING must carry a wake_condition (sleeping, "
                "never dead) -- a shrug is not a state"
            )

    def to_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind,
            "expected_value": self.expected_value,
            "wake_condition": self.wake_condition,
            "evidence": list(self.evidence), "detail": self.detail,
        }


@dataclass(frozen=True)
class SchedulerDecision:
    """The scheduler's answer: what to run next, or why nothing runs."""

    action: str  # "CONTINUE" | "RUN" | "PARK_ALL"
    frontier: Optional[str]
    reason: str
    wake_conditions: dict  # populated on PARK_ALL: name -> wake_condition
    snapshot: tuple  # every FrontierState considered, for the audit trail

    def to_dict(self) -> dict:
        return {
            "action": self.action, "frontier": self.frontier,
            "reason": self.reason, "wake_conditions": dict(self.wake_conditions),
            "snapshot": [s.to_dict() for s in self.snapshot],
        }


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _parse_ts(s: Any) -> Optional[float]:
    if not isinstance(s, str):
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


# -- probes ------------------------------------------------------------
# Each takes an optional pre-loaded ``data`` mapping so tests can supply a
# fixture directly with zero file I/O; calling one with no arguments reads
# the real repo state at its documented path.

def probe_sub2(data: Optional[Mapping] = None,
               path: Path = RECEIPTS / "HCLI_MISSION_KERNEL.json") -> FrontierState:
    k = data if data is not None else _read_json(path)
    if k is None:
        return FrontierState("SUB2", UNKNOWN, 0.0, None, evidence=(str(path),))
    stops = k.get("stops") or []
    n_iter = len(k.get("iterations") or [])
    label = k.get("frontier") or "SUB2"
    if stops and stops[-1].get("n") == n_iter:
        # Nothing has iterated since the loop parked itself: still asleep.
        stop = stops[-1]
        return FrontierState(
            "SUB2", WAITING, 0.0,
            wake_condition=stop.get("wake_condition") or "operator steer",
            evidence=(str(path),),
            detail=f"{label} parked: {stop.get('reason', stop.get('event', 'stopped'))}",
        )
    n_hyp = len(k.get("hypotheses") or [])
    return FrontierState("SUB2", ELIGIBLE, float(n_hyp), None,
                          evidence=(str(path),),
                          detail=f"{label}: {n_hyp} open hypotheses, {n_iter} iterations logged")


def probe_hcli_self(data: Optional[Mapping] = None,
                     path: Path = RECEIPTS / "HCLI_CAPABILITY_REACHABILITY.json") -> FrontierState:
    d = data if data is not None else _read_json(path)
    if d is None:
        return FrontierState("HCLI_SELF", UNKNOWN, 0.0, None, evidence=(str(path),))
    dead = int((d.get("counts") or {}).get("typed_tools_dead", 0))
    if dead <= 0:
        return FrontierState(
            "HCLI_SELF", WAITING, 0.0,
            wake_condition="a reachability sweep finds a new typed tool with zero call sites",
            evidence=(str(path),),
        )
    return FrontierState("HCLI_SELF", ELIGIBLE, float(dead), None,
                          evidence=(str(path),),
                          detail=f"{dead} typed tools registered with zero call sites")


def probe_resident_speed(data: Optional[Mapping] = None,
                          path: Path = RECEIPTS / "RESIDENT_TPS_RECON.json",
                          now: Optional[float] = None) -> FrontierState:
    d = data if data is not None else _read_json(path)
    if d is None:
        return FrontierState("RESIDENT_SPEED", UNKNOWN, 0.0, None, evidence=(str(path),))
    ts = _parse_ts(d.get("recorded_at"))
    age_h = max(0.0, ((now if now is not None else time.time()) - ts) / 3600.0) if ts else 0.0
    verdict = str(d.get("verdict", ""))
    if verdict.upper().startswith(("WON", "LANDED")) and age_h < 1.0:
        # A lever just landed -- give it an hour before proposing another
        # attempt on the same body rather than re-litigating a fresh win.
        return FrontierState(
            "RESIDENT_SPEED", WAITING, 0.0,
            wake_condition="an hour since the last landed TPS receipt, or a new lever proposal",
            evidence=(str(path),),
        )
    # ponytail: expected_value here is receipt staleness (hours since the
    # last recon), capped at 48h -- a real signal (nobody has revisited
    # this frontier), not a proxy for actual speedup headroom. Upgrade
    # path: score by the causal-budget gap once one is derivable per lever.
    return FrontierState("RESIDENT_SPEED", ELIGIBLE, min(age_h, 48.0), None,
                          evidence=(str(path),),
                          detail=f"last verdict {verdict[:80]!r}, {age_h:.1f}h old")


def probe_odyssey(data: Optional[Mapping] = None,
                   patients_dir: Path = ODYSSEY_PATIENTS) -> FrontierState:
    if data is not None:
        n = int(data.get("n_patients", 0))
        evidence = (str(patients_dir),)
    else:
        try:
            n = sum(1 for p in patients_dir.iterdir() if p.is_dir())
        except OSError:
            return FrontierState("ODYSSEY", UNKNOWN, 0.0, None, evidence=(str(patients_dir),))
        evidence = (str(patients_dir),)
    if n <= 0:
        return FrontierState(
            "ODYSSEY", WAITING, 0.0,
            wake_condition="a new candidate admitted to the odyssey patient queue",
            evidence=evidence,
        )
    return FrontierState("ODYSSEY", ELIGIBLE, float(n), None, evidence=evidence,
                          detail=f"{n} patient packet(s) queued")


def probe_modellake(data: Optional[Mapping] = None,
                     path: Path = RECEIPTS / "MODELLAKE_SCHEDULER_VIEW.json") -> FrontierState:
    d = data if data is not None else _read_json(path)
    if d is None:
        return FrontierState("MODELLAKE", UNKNOWN, 0.0, None, evidence=(str(path),))
    live = d.get("live") or {}
    active = live.get("active_jobs") or []
    stale = bool(live.get("stale", True))
    if active and not stale:
        return FrontierState(
            "MODELLAKE", WAITING, 0.0,
            wake_condition=f"active download(s) finish: {', '.join(active)}",
            evidence=(str(path),),
            detail="watcher's live sample is fresh and shows job(s) in flight",
        )
    return FrontierState("MODELLAKE", ELIGIBLE, float(len(active) or 1), None,
                          evidence=(str(path),),
                          detail="no fresh in-flight job to protect")


def probe_forbidden_fruit(data: Optional[Mapping] = None,
                           path: Path = RECEIPTS / "FORBIDDEN_FRUIT_LAB_READINESS.json") -> FrontierState:
    d = data if data is not None else _read_json(path)
    if d is None:
        return FrontierState("FORBIDDEN_FRUIT", UNKNOWN, 0.0, None, evidence=(str(path),))
    missing = d.get("smallest_missing_piece")
    if not missing:
        return FrontierState("FORBIDDEN_FRUIT", ELIGIBLE, 0.0, None, evidence=(str(path),))
    return FrontierState(
        "FORBIDDEN_FRUIT", WAITING, 0.0, wake_condition=str(missing),
        evidence=(str(path),), detail=str(d.get("verdict", "")),
    )


DEFAULT_PROBES: "dict[str, Callable[[], FrontierState]]" = {
    "SUB2": probe_sub2,
    "HCLI_SELF": probe_hcli_self,
    "RESIDENT_SPEED": probe_resident_speed,
    "ODYSSEY": probe_odyssey,
    "MODELLAKE": probe_modellake,
    "FORBIDDEN_FRUIT": probe_forbidden_fruit,
}


def snapshot(probes: Optional[Mapping[str, Callable[[], FrontierState]]] = None) -> tuple:
    """Read every frontier once. One broken probe reports UNKNOWN for
    itself and never prevents the other frontiers from being read -- this
    is the auditable record the brief asks for: call it, keep the result,
    argue from it instead of from memory."""
    probes = probes if probes is not None else DEFAULT_PROBES
    out = []
    for name, probe in probes.items():
        try:
            state = probe()
        except Exception as exc:  # noqa: BLE001 - a probe's own bug is data, not our crash
            state = FrontierState(name, UNKNOWN, 0.0, None, detail=f"probe raised: {exc!r}")
        out.append(state)
    return tuple(out)


def select_next(states: Sequence[FrontierState]) -> SchedulerDecision:
    """Which frontier runs next, given one snapshot.

    Preference order: a RUNNING frontier is left alone; otherwise the
    highest-``expected_value`` ELIGIBLE frontier launches; only if neither
    exists does it report PARK_ALL, and PARK_ALL always carries every
    frontier's wake_condition -- never a bare idle report.
    """
    states = tuple(states)
    running = [s for s in states if s.kind == RUNNING]
    if running:
        s = running[0]
        return SchedulerDecision("CONTINUE", s.name,
                                  f"{s.name} is RUNNING; a parked frontier is picked up "
                                  "only when nothing is currently running",
                                  {}, states)

    eligible = [s for s in states if s.kind == ELIGIBLE]
    if eligible:
        best = max(eligible, key=lambda s: s.expected_value)
        return SchedulerDecision(
            "RUN", best.name,
            f"{best.name} has the highest expected_value "
            f"({best.expected_value:g}) among {len(eligible)} eligible frontier(s)",
            {}, states,
        )

    # Nothing runnable. This must never be reported as a shrug: collect
    # every recorded wake_condition so the caller always gets the specific
    # next thing to watch for, not "waiting for instructions".
    wake = {s.name: s.wake_condition for s in states if s.wake_condition}
    unknown = [s.name for s in states if s.kind == UNKNOWN]
    reason = (
        f"no eligible or running frontier among {len(states)}; all sleeping "
        "with a recorded wake condition" if wake else
        "no eligible or running frontier and no readable wake condition -- "
        "the underlying probes are unreadable, not the frontiers idle"
    )
    if unknown and not wake:
        wake = {"UNKNOWN": f"probe(s) unreadable, re-run once fixed: {unknown}"}
    return SchedulerDecision("PARK_ALL", None, reason, wake, states)


def decide(probes: Optional[Mapping[str, Callable[[], FrontierState]]] = None) -> SchedulerDecision:
    """One-shot: read every frontier, then decide. The entry point a real
    caller wires in (see module docstring / out_of_scope)."""
    return select_next(snapshot(probes))


def main() -> int:
    print(json.dumps(decide().to_dict(), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
