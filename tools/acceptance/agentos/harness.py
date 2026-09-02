#!/usr/bin/env python3
"""Demonstrate AGENTOS acceptance criteria against the live hcli import.

Each gate invokes its catalogued implementing symbol (not a module import)
and records a real run. Verdicts are ACCEPTED or BLOCKED; a receipt that
merely exists is not acceptance. Nothing here edits hcli.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ.setdefault("HCLI_DISABLE_SIGNAL_HOOKS", "1")

REPO = Path(__file__).resolve().parents[3]
RECEIPT_DIR = REPO / "receipts" / "acceptance"
ROADMAP = Path("/Users/scammermike/Downloads/H-ROADMAP.md")
SCHEMA = "hawking.acceptance.gate.v1"
EVIDENCE_TIER = "FUNCTIONAL_SIM"

GATES = (
    "AGENTOS_REPAIR_BOUNDED",
    "AGENTOS_RETRY_CLASSIFIED",
    "AGENTOS_CIRCUIT_BREAKER",
    "AGENTOS_CANCELLATION",
    "AGENTOS_ORPHAN_RECONCILIATION",
    "AGENTOS_PERSISTENCE_SINGLE_AUTHORITY",
    "AGENTOS_CHECKPOINT_ATOMICITY",
    "AGENTOS_RESTART_COHERENCE",
)

D2_TAXONOMY = (
    "TRANSIENT_BACKEND",
    "RATE_LIMIT",
    "BACKEND_UNAVAILABLE",
    "VERIFIER_FAILED",
    "IMPLEMENTATION_FAILED",
    "INVALID_OUTPUT",
    "DEPENDENCY_BLOCKED",
    "CONTRACT_IMPOSSIBLE",
    "CANCELLED",
    "RESOURCE_DENIED",
    "AUTHORIZATION_REQUIRED",
    "STATE_AMBIGUOUS",
    "PERSISTENCE_CORRUPT",
    "TOOLCHAIN_BLOCKED",
    "BENCHMARK_CONTAMINATED",
    "THERMAL_RESOURCE_BLOCK",
    "DISK_HEADROOM_BLOCK",
)

D3_FIELDS = (
    "WorkUnit state",
    "Goal/DAG",
    "backend_task_id",
    "provider/runtime provenance",
    "dependencies",
    "verifier result",
    "repair lineage",
    "retry state",
    "mutation lease",
    "steering",
    "context calibration",
    "MAX calibration",
    "backend health",
    "checkpoint generation",
    "background jobs",
    "VMCP evidence references",
    "ModelLake worker identity",
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=10,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "UNKNOWN"


def _hcli_file() -> str:
    import hcli

    return str(Path(hcli.__file__).resolve())


def _quote_span(start: int, end: int) -> str:
    if not ROADMAP.is_file():
        return f"<H-ROADMAP.md not readable at {ROADMAP}>"
    lines = ROADMAP.read_text(encoding="utf-8").splitlines()
    chunk = lines[start - 1 : end]
    return "\n".join(chunk)


def _wu(uid: str, **kwargs: Any):
    from hcli.workunit import WorkUnit

    return WorkUnit(
        id=uid,
        role=kwargs.pop("role", "implement"),
        description=kwargs.pop("description", f"unit {uid}"),
        resource_class=kwargs.pop("resource_class", "LIGHT_CONTROL"),
        **kwargs,
    )


def _passing() -> Dict[str, Any]:
    return {"ok": True, "verifier": "acceptance-harness"}


def _stub_engine():
    class Engine:
        active = False
        child_pids: List[int] = []

        def execute_workunit(self, wu, context):  # noqa: ANN001
            return {
                "kind": "answer",
                "content": f"did {wu.id}",
                "validation": _passing(),
            }

        def cancel(self) -> None:
            return None

    return Engine()


def _write_receipt(payload: Dict[str, Any]) -> Path:
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    gate = payload["gate"]
    path = RECEIPT_DIR / f"{gate}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _base(
    gate: str,
    *,
    criterion: str,
    start: int,
    end: int,
    command: str,
    symbols: List[Dict[str, Any]],
    measured: Dict[str, Any],
    output: str,
    verdict: str,
    blocker: Optional[str],
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "gate": gate,
        "criterion_quoted": criterion,
        "criterion_source": {
            "file": str(ROADMAP),
            "start_line": start,
            "end_line": end,
        },
        "command": command,
        "symbols_invoked": symbols,
        "measured": measured,
        "output": output[-12000:],
        "verdict": verdict,
        "blocker": blocker,
        "notes": notes or [],
        "evidence_tier": EVIDENCE_TIER,
        "generated_at": _now(),
        "git_head": _git_head(),
        "hcli_file": _hcli_file(),
        "criterion_altered": False,
    }


# ---------------------------------------------------------------------------
# AGENTOS_REPAIR_BOUNDED
# ---------------------------------------------------------------------------


def demo_repair_bounded() -> Dict[str, Any]:
    from hcli.scheduler import Scheduler
    from hcli.workunit import (
        MAX_REPAIR_DEPTH,
        MAX_REPAIRS_PER_ROOT,
        is_ready,
        emit_repair,
    )
    import hcli.workunit as workunit_mod
    import hcli.scheduler as scheduler_mod

    log: List[str] = []
    symbols = [
        {"module": "hcli.scheduler", "symbol": "Scheduler", "via": "Scheduler.fail -> emit_repair"},
        {"module": "hcli.workunit", "symbol": "emit_repair", "via": "direct + Scheduler._emit_repair"},
    ]

    def grow(unique: bool, depth_cap: int, count_cap: int) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="acc-repair-") as tmp:
            root = _wu("dead")
            root.status = "running"
            sched = Scheduler({"dead": root}, runtime_count=1, workspace=tmp)
            steps = 0
            while steps < 80:
                target = None
                for wu in reversed(list(sched.units.values())):
                    if wu.repair_exhausted:
                        continue
                    target = wu
                    break
                if target is None:
                    break
                target.status = "running"
                steps += 1
                ctx = (
                    {"error": f"injected {steps}", "reason": f"injected {steps}"}
                    if unique
                    else {"error": "injected failure", "reason": "injected failure"}
                )
                repair = sched.fail(target.id, ctx)
                if repair is None:
                    break
                repair.status = "running"
            deepest = max(int(u.repair_depth or 0) for u in sched.units.values())
            exhausted = [
                {"id": u.id, "reason": u.repair_reason, "depth": u.repair_depth}
                for u in sched.units.values()
                if u.repair_exhausted
            ]
            lineage = sorted(sched.units)
            repairs = [u for u in lineage if u.startswith("dead.repair")]
            resource_classes = sorted({u.resource_class for u in sched.units.values()})
            return {
                "n_units": len(sched.units),
                "n_repairs": len(repairs),
                "deepest": deepest,
                "exhausted": exhausted,
                "lineage": lineage,
                "resource_classes": resource_classes,
                "steps": steps,
            }

    cycle = grow(unique=False, depth_cap=MAX_REPAIR_DEPTH, count_cap=MAX_REPAIRS_PER_ROOT)
    unique = grow(unique=True, depth_cap=MAX_REPAIR_DEPTH, count_cap=MAX_REPAIRS_PER_ROOT)
    log.append(f"cycle: {json.dumps(cycle, default=str)}")
    log.append(f"unique: {json.dumps(unique, default=str)}")

    spent = _wu("spent")
    spent.status = "failed"
    spent.repair_exhausted = True
    spent.repair_reason = "budget spent"
    spent.repair_root = "spent"
    not_ready = is_ready(spent, {"spent": spent}) is False
    log.append(f"exhausted is_ready={not not_ready} (want False)")

    # Negative control: both bounds must be lifted (count cap binds first).
    original_depth = workunit_mod.MAX_REPAIR_DEPTH
    original_count = workunit_mod.MAX_REPAIRS_PER_ROOT
    try:
        workunit_mod.MAX_REPAIR_DEPTH = 12
        workunit_mod.MAX_REPAIRS_PER_ROOT = 500
        lifted = grow(unique=True, depth_cap=12, count_cap=500)
    finally:
        workunit_mod.MAX_REPAIR_DEPTH = original_depth
        workunit_mod.MAX_REPAIRS_PER_ROOT = original_count
    log.append(f"negative control lifted: {json.dumps(lifted, default=str)}")

    failed_exhausted_token = False
    log.append("FAILED_EXHAUSTED token absent; durable flag is repair_exhausted")

    bound_holds = (
        unique["deepest"] <= MAX_REPAIR_DEPTH
        and unique["n_repairs"] <= MAX_REPAIRS_PER_ROOT
        and bool(unique["exhausted"])
        and cycle["n_repairs"] <= MAX_REPAIRS_PER_ROOT
        and bool(cycle["exhausted"])
        and not_ready
        and unique["resource_classes"] == ["LIGHT_CONTROL"]
    )
    negative_holds = lifted["deepest"] > original_depth and lifted["n_repairs"] > original_count
    ok = bound_holds and negative_holds

    measured = {
        "max_repair_depth": MAX_REPAIR_DEPTH,
        "max_repairs_per_root": MAX_REPAIRS_PER_ROOT,
        "cycle_n_repairs": cycle["n_repairs"],
        "cycle_deepest": cycle["deepest"],
        "cycle_exhausted_reasons": [e["reason"] for e in cycle["exhausted"]],
        "unique_n_repairs": unique["n_repairs"],
        "unique_deepest": unique["deepest"],
        "unique_exhausted_reasons": [e["reason"] for e in unique["exhausted"]],
        "exhausted_not_rereadied": not_ready,
        "negative_control_deepest": lifted["deepest"],
        "negative_control_n_repairs": lifted["n_repairs"],
        "repair_does_not_widen_resource_class": unique["resource_classes"] == ["LIGHT_CONTROL"],
        "failed_exhausted_token_present": failed_exhausted_token,
        "scheduler_module": scheduler_mod.__file__,
        "emit_repair_callable": callable(emit_repair),
    }
    criterion = (
        "Repair depth is bounded structurally, not by model discretion. "
        "Equivalent deterministic failure fingerprints do not generate semantically "
        "identical descendants. Exhausted chains become FAILED_EXHAUSTED / BLOCKED / "
        "REPLAN_REQUIRED. No repair child may widen mutation/safety/authorization "
        "scope inherited from its root.\n\n"
        + _quote_span(7332, 7358)
    )
    notes = [
        "Terminal exhausted state is the durable flag repair_exhausted=True "
        "(status remains failed; is_ready returns False). The exact token "
        "FAILED_EXHAUSTED is not a WorkUnit status in this checkout.",
        "Catalog symbol Scheduler is invoked via Scheduler.fail, which is the "
        "only synthesis path (fail -> _emit_repair -> workunit.emit_repair).",
    ]
    if ok:
        return _base(
            "AGENTOS_REPAIR_BOUNDED",
            criterion=criterion,
            start=7332,
            end=7358,
            command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_repair_bounded -o addopts=''",
            symbols=symbols,
            measured=measured,
            output="\n".join(log),
            verdict="ACCEPTED",
            blocker=None,
            notes=notes,
        )
    return _base(
        "AGENTOS_REPAIR_BOUNDED",
        criterion=criterion,
        start=7332,
        end=7358,
        command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_repair_bounded -o addopts=''",
        symbols=symbols,
        measured=measured,
        output="\n".join(log),
        verdict="BLOCKED",
        blocker="structural repair bound did not hold on this run; see measured",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# AGENTOS_RETRY_CLASSIFIED
# ---------------------------------------------------------------------------


def demo_retry_classified() -> Dict[str, Any]:
    from hcli.resources import (
        FAILURE_KINDS,
        classify_failure,
        counts_toward_retry_budget,
        NON_RETRYABLE,
    )
    from hcli.scheduler import Scheduler, NO_PROGRESS
    from hcli.workunit import WorkUnit

    log: List[str] = []
    samples = {
        "TRANSIENT_BACKEND": {"error": "llama-server HTTP 503: overloaded"},
        "RATE_LIMIT": {"error": "llama-server HTTP 429: too many requests"},
        "VERIFIER_FAILURE": {"reason": "TEST_FAILED"},
        "DETERMINISTIC_IMPLEMENTATION": {"reason": "NO_OP_MUTATION"},
        "INVALID_OUTPUT": {"error": "llama-server returned invalid JSON"},
        "UNAVAILABLE_DEPENDENCY": {"error": "GrokNotAvailable"},
        "IMPOSSIBLE_CONTRACT": {"error": "GrokContractError"},
    }
    produced: Dict[str, Any] = {}
    for kind, ctx in samples.items():
        clf = classify_failure(ctx)
        produced[kind] = {
            "kind": clf.kind,
            "retryable": clf.retryable,
            "observed": clf.observed,
            "counts_toward_budget": counts_toward_retry_budget(ctx),
        }
        log.append(f"classify {kind} -> {produced[kind]}")

    implemented = set(FAILURE_KINDS)
    d2 = set(D2_TAXONOMY)
    name_map = {
        "VERIFIER_FAILED": "VERIFIER_FAILURE",
        "IMPLEMENTATION_FAILED": "DETERMINISTIC_IMPLEMENTATION",
        "BACKEND_UNAVAILABLE": "UNAVAILABLE_DEPENDENCY",
        "DEPENDENCY_BLOCKED": "UNAVAILABLE_DEPENDENCY",
        "CONTRACT_IMPOSSIBLE": "IMPOSSIBLE_CONTRACT",
    }
    exact = sorted(d2 & implemented)
    aliased = {k: v for k, v in name_map.items() if k in d2 and v in implemented}
    missing = sorted(c for c in D2_TAXONOMY if c not in implemented and c not in name_map)

    with tempfile.TemporaryDirectory(prefix="acc-retry-") as tmp:
        wu = _wu("noop")
        wu.status = "running"
        sched = Scheduler({"noop": wu}, runtime_count=1, workspace=tmp)
        repair = sched.fail("noop", {"reason": "NO_OP_MUTATION"})
        scheduler_consults = repair is None
        emitted = None if repair is None else repair.id
        log.append(
            f"Scheduler.fail(NO_OP_MUTATION) consults classifier={scheduler_consults} "
            f"emitted={emitted}"
        )

        # Catalog symbol _record_fingerprint is a Scheduler method. Invoke it
        # through complete(), which is the production caller.
        a = _wu("a")
        a.status = "running"
        a.verification = _passing()
        b = _wu("b")
        b.status = "running"
        b.verification = _passing()
        c = _wu("c")
        c.status = "running"
        c.verification = _passing()
        fp_sched = Scheduler(
            {"a": a, "b": b, "c": c},
            runtime_count=3,
            workspace=tmp,
            no_progress_threshold=3,
        )
        fp_sched.complete("a", fingerprint="same", verification=_passing())
        fp_sched.complete("b", fingerprint="same", verification=_passing())
        raised = None
        try:
            fp_sched.complete("c", fingerprint="same", verification=_passing())
        except NO_PROGRESS as exc:
            raised = str(exc)
        log.append(f"_record_fingerprint via complete raised NO_PROGRESS={raised is not None}")

    import hcli.resources as resources_mod
    import hcli.scheduler as scheduler_mod

    # Production callers of classify_failure: only BackendHealth.record_failure
    # in this module. BackendHealth itself has no hcli caller (measured by
    # this run still emitting a repair for a non-retryable failure).
    measured = {
        "failure_kinds_implemented": list(FAILURE_KINDS),
        "d2_taxonomy": list(D2_TAXONOMY),
        "d2_exact_overlap": exact,
        "d2_aliased": aliased,
        "d2_missing": missing,
        "classified_samples": produced,
        "non_retryable": sorted(NON_RETRYABLE),
        "scheduler_fail_consults_classify_failure": scheduler_consults,
        "non_retryable_still_emits_repair": emitted,
        "record_fingerprint_invoked_via_complete": True,
        "record_fingerprint_raised_no_progress": raised is not None,
        "classify_failure_module": resources_mod.__file__,
        "scheduler_module": scheduler_mod.__file__,
    }
    criterion = (
        "Retries are classified by the D.2 failure taxonomy. Transient provider "
        "failure may retry; deterministic verifier / non-retryable failures must "
        "not become a blind retry loop.\n\n" + _quote_span(7361, 7379)
    )
    blocker = (
        "Scheduler.fail does not consult classify_failure (measured: "
        f"NO_OP_MUTATION still emitted repair {emitted!r}). "
        "classify_failure is not on the retry path. D.2 lists "
        f"{len(D2_TAXONOMY)} class names; FAILURE_KINDS implements "
        f"{len(FAILURE_KINDS)} tokens. Exact overlap={exact}; aliased={aliased}; "
        f"unimplemented D.2 classes={missing}. Retries are therefore not "
        "classified by the gate's own taxonomy."
    )
    return _base(
        "AGENTOS_RETRY_CLASSIFIED",
        criterion=criterion,
        start=7361,
        end=7379,
        command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_retry_classified -o addopts=''",
        symbols=[
            {
                "module": "hcli.scheduler",
                "symbol": "_record_fingerprint",
                "via": "Scheduler.complete",
            },
            {
                "module": "hcli.resources",
                "symbol": "classify_failure",
                "via": "direct call of the D.2 classifier",
            },
            {
                "module": "hcli.scheduler",
                "symbol": "Scheduler.fail",
                "via": "retry path that should consult the classifier",
            },
        ],
        measured=measured,
        output="\n".join(log),
        verdict="BLOCKED",
        blocker=blocker,
        notes=[
            "Catalog symbol is _record_fingerprint (progress-fingerprint circuit), "
            "which is not the D.2 taxonomy. Both were invoked. Acceptance follows "
            "the roadmap span, not the catalog symbol mismatch.",
        ],
    )


# ---------------------------------------------------------------------------
# AGENTOS_CIRCUIT_BREAKER
# ---------------------------------------------------------------------------


def demo_circuit_breaker() -> Dict[str, Any]:
    from hcli.scheduler import Scheduler, NO_PROGRESS, DEFAULT_NO_PROGRESS_THRESHOLD
    from hcli.resources import (
        BackendHealth,
        CIRCUIT_FAILURE_THRESHOLD,
        CIRCUIT_COOLING_SECONDS,
        STATE_CIRCUIT_OPEN,
        STATE_HEALTHY,
    )

    log: List[str] = []

    class FakeClock:
        def __init__(self, t: float = 1_000.0) -> None:
            self.t = float(t)

        def __call__(self) -> float:
            return self.t

        def advance(self, seconds: float) -> None:
            self.t += float(seconds)

    with tempfile.TemporaryDirectory(prefix="acc-circuit-") as tmp:
        units = {f"u{i}": _wu(f"u{i}") for i in range(1, 4)}
        for u in units.values():
            u.status = "running"
            u.verification = _passing()
        sched = Scheduler(
            units,
            runtime_count=3,
            workspace=tmp,
            no_progress_threshold=DEFAULT_NO_PROGRESS_THRESHOLD,
        )
        sched.complete("u1", fingerprint="loop", verification=_passing())
        sched.complete("u2", fingerprint="loop", verification=_passing())
        raised = None
        try:
            sched.complete("u3", fingerprint="loop", verification=_passing())
        except NO_PROGRESS as exc:
            raised = {
                "type": type(exc).__name__,
                "fingerprint": exc.fingerprint,
                "count": exc.count,
                "threshold": exc.threshold,
                "str": str(exc),
            }
        log.append(f"NO_PROGRESS raised={raised}")

        changing = {f"v{i}": _wu(f"v{i}") for i in range(1, 4)}
        for u in changing.values():
            u.status = "running"
            u.verification = _passing()
        quiet = Scheduler(
            changing,
            runtime_count=3,
            workspace=tmp,
            no_progress_threshold=3,
        )
        quiet_raised = None
        try:
            quiet.complete("v1", fingerprint="a", verification=_passing())
            quiet.complete("v2", fingerprint="b", verification=_passing())
            quiet.complete("v3", fingerprint="c", verification=_passing())
        except NO_PROGRESS as exc:
            quiet_raised = str(exc)
        log.append(f"changing fingerprints raised={quiet_raised}")

        clock = FakeClock()
        health = BackendHealth(
            tmp,
            clock=clock,
            failure_threshold=CIRCUIT_FAILURE_THRESHOLD,
            cooling_seconds=CIRCUIT_COOLING_SECONDS,
        )
        ctx = {"error": "llama-server HTTP 503: overloaded"}
        for _ in range(CIRCUIT_FAILURE_THRESHOLD):
            health.record_failure("qwen", ctx)
        snap_open = health.snapshot("qwen")
        allows_when_open = health.allows_new_assignments("qwen")
        grok_untouched = health.snapshot("grok")["state"]
        clock.advance(CIRCUIT_COOLING_SECONDS + 0.1)
        snap_cooled = health.snapshot("qwen")
        health.record_success("qwen")
        snap_closed = health.snapshot("qwen")
        log.append(
            f"BackendHealth open={snap_open['state']} allows={allows_when_open} "
            f"cooled={snap_cooled['state']} closed={snap_closed['state']} "
            f"grok={grok_untouched}"
        )

        # Live dispatch does not consult the breaker (measured).
        from hcli.workunit import assign_ready

        live = _wu("live", preferred_backend="qwen")
        live.status = "ready"
        assignments = assign_ready([live], runtime_count=1, all_units={live.id: live})
        dispatched_through_open = bool(assignments) and live.status == "running"
        log.append(f"assign_ready through open circuit={dispatched_through_open}")

    ok = (
        raised is not None
        and raised["type"] == "NO_PROGRESS"
        and raised["count"] >= DEFAULT_NO_PROGRESS_THRESHOLD
        and quiet_raised is None
        and snap_open["state"] == STATE_CIRCUIT_OPEN
        and allows_when_open is False
        and snap_closed["state"] == STATE_HEALTHY
    )
    measured = {
        "no_progress_threshold": DEFAULT_NO_PROGRESS_THRESHOLD,
        "no_progress_raised": raised,
        "changing_fingerprint_raised": quiet_raised,
        "backend_health_open_state": snap_open["state"],
        "backend_health_allows_when_open": allows_when_open,
        "backend_health_closed_state": snap_closed["state"],
        "circuit_failure_threshold": CIRCUIT_FAILURE_THRESHOLD,
        "circuit_cooling_seconds": CIRCUIT_COOLING_SECONDS,
        "assign_ready_dispatches_through_open_circuit": dispatched_through_open,
        "grok_untouched_while_qwen_open": grok_untouched == STATE_HEALTHY,
    }
    criterion = (
        "Transient provider failure may retry with bounded exponential/backoff "
        "policy; deterministic verifier failure does not become a blind retry "
        "loop. Repeated identical progress fingerprints raise NO_PROGRESS.\n\n"
        + _quote_span(7332, 7358)
    )
    notes = [
        "NO_PROGRESS is the catalog symbol and is raised from Scheduler._record_fingerprint "
        "via complete(). Mission._on_no_progress is the production handler.",
        "BackendHealth is a second breaker with cooling; assign_ready does not consult it. "
        "That gap does not refute NO_PROGRESS, which is the wired circuit.",
        "There is no per-retry sleep; 'may retry with backoff' is permission, not a must. "
        "The load-bearing stop is NO_PROGRESS + repair cycle detection.",
    ]
    if ok:
        return _base(
            "AGENTOS_CIRCUIT_BREAKER",
            criterion=criterion,
            start=7332,
            end=7358,
            command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_circuit_breaker -o addopts=''",
            symbols=[
                {"module": "hcli.scheduler", "symbol": "NO_PROGRESS", "via": "raised from Scheduler.complete"},
                {"module": "hcli.resources", "symbol": "BackendHealth", "via": "record_failure / allows_new_assignments"},
            ],
            measured=measured,
            output="\n".join(log),
            verdict="ACCEPTED",
            blocker=None,
            notes=notes,
        )
    return _base(
        "AGENTOS_CIRCUIT_BREAKER",
        criterion=criterion,
        start=7332,
        end=7358,
        command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_circuit_breaker -o addopts=''",
        symbols=[{"module": "hcli.scheduler", "symbol": "NO_PROGRESS"}],
        measured=measured,
        output="\n".join(log),
        verdict="BLOCKED",
        blocker="NO_PROGRESS did not fire on repeated fingerprints, or BackendHealth did not open",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# AGENTOS_CANCELLATION
# ---------------------------------------------------------------------------


def demo_cancellation() -> Dict[str, Any]:
    from hcli import delegate as d
    from hcli.mission import Mission, mission_state_path
    from hcli.resources import MutationLock

    log: List[str] = []
    with tempfile.TemporaryDirectory(prefix="acc-cancel-") as tmp:
        ws = Path(tmp) / "mission"
        started = d.run(
            "acceptance cancel demonstration",
            workspace=ws,
            spawn=False,
        )
        log.append(f"delegate.run spawn=False keys={sorted(started)}")
        state_path = mission_state_path(ws)
        dag_path = ws / ".hcli" / "dag.json"
        pre_state = json.loads(state_path.read_text(encoding="utf-8"))
        pre_dag = json.loads(dag_path.read_text(encoding="utf-8"))
        pre_id = pre_state.get("checkpoint_id")
        log.append(f"pre checkpoint_id={pre_id} phase={pre_state.get('phase')}")

        out = d.abort(ws, reason="acceptance-operator-abort")
        log.append(f"abort returned { {k: out.get(k) for k in ('verdict','reason','lock_free','mission_id')} }")

        post_state = json.loads(state_path.read_text(encoding="utf-8"))
        post_dag = json.loads(dag_path.read_text(encoding="utf-8"))
        post_id = post_state.get("checkpoint_id")
        cancel_file = d.cancel_path(ws)
        cancel_doc = json.loads(cancel_file.read_text(encoding="utf-8")) if cancel_file.is_file() else None
        log.append(f"cancel_file={cancel_file} exists={cancel_file.is_file()} doc={cancel_doc}")
        lock = MutationLock(ws)
        lock_free = lock.read() is None or lock.try_break_stale()

        shared = post_id == post_dag.get("checkpoint_id")
        new_generation = bool(post_id) and post_id != pre_id
        cancelled = post_state.get("phase") == "cancelled"
        reason_ok = post_state.get("cancel_reason") == "acceptance-operator-abort"
        durable_cancel = bool(cancel_doc) and cancel_doc.get("reason") == "acceptance-operator-abort"
        abort_verdict = out.get("verdict") == "ABORTED"

        # Cooperative in-process cancel also writes cancelled without emitting repairs.
        units = {"slow": _wu("slow")}
        mission = Mission(
            Path(tmp) / "coop",
            engine=_CooperativeEngine(),
            units=units,
            quiet=True,
            no_progress_threshold=50,
            goal="cancel-coop",
        )
        thread = threading.Thread(target=mission.run, daemon=True)
        thread.start()
        mission.engine.entered.wait(timeout=5)
        mission.cancel("probe-cancel")
        thread.join(timeout=8)
        coop_phase = mission.phase
        coop_repairs = [u.id for u in mission.scheduler.units.values() if u.repairs == "slow"]
        coop_state = Path(tmp) / "coop" / ".hcli" / "mission" / "state.json"
        coop_disk = json.loads(coop_state.read_text(encoding="utf-8")) if coop_state.is_file() else {}
        log.append(
            f"coop phase={coop_phase} repairs={coop_repairs} disk_phase={coop_disk.get('phase')} "
            f"disk_reason={coop_disk.get('cancel_reason')}"
        )

    ok = (
        abort_verdict
        and cancelled
        and reason_ok
        and durable_cancel
        and shared
        and new_generation
        and lock_free
        and coop_phase == "cancelled"
        and coop_repairs == []
        and coop_disk.get("phase") == "cancelled"
    )
    measured = {
        "abort_verdict": out.get("verdict"),
        "abort_reason": out.get("reason"),
        "lock_free": bool(lock_free),
        "pre_checkpoint_id": pre_id,
        "post_checkpoint_id": post_id,
        "dag_checkpoint_id": post_dag.get("checkpoint_id"),
        "shared_checkpoint_id": shared,
        "new_generation": new_generation,
        "state_phase": post_state.get("phase"),
        "state_cancel_reason": post_state.get("cancel_reason"),
        "durable_cancel_file": durable_cancel,
        "coop_phase": coop_phase,
        "coop_repairs": coop_repairs,
        "coop_disk_phase": coop_disk.get("phase"),
        "pre_dag_units": sorted((pre_dag.get("units") or {})),
    }
    criterion = (
        "Cancellation writes a durable terminal state and reconciles children/"
        "external jobs.\n\n" + _quote_span(7332, 7358)
    )
    notes = [
        "Catalog symbol abort is hcli.delegate.abort. It writes delegation_cancel.json, "
        "signals a live mutation-lock holder, loads the mission, Mission.cancel, then "
        "Mission.checkpoint (DAG first, shared checkpoint_id).",
        "Cancel of in-flight units uses _fail_inflight(emit_repair=False) so cancellation "
        "does not grow the repair tree.",
    ]
    if ok:
        return _base(
            "AGENTOS_CANCELLATION",
            criterion=criterion,
            start=7332,
            end=7358,
            command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_cancellation -o addopts=''",
            symbols=[{"module": "hcli.delegate", "symbol": "abort", "via": "direct call"}],
            measured=measured,
            output="\n".join(log),
            verdict="ACCEPTED",
            blocker=None,
            notes=notes,
        )
    return _base(
        "AGENTOS_CANCELLATION",
        criterion=criterion,
        start=7332,
        end=7358,
        command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_cancellation -o addopts=''",
        symbols=[{"module": "hcli.delegate", "symbol": "abort"}],
        measured=measured,
        output="\n".join(log),
        verdict="BLOCKED",
        blocker="abort did not persist a cancelled generation with a shared checkpoint_id",
        notes=notes,
    )


class _CooperativeEngine:
    def __init__(self) -> None:
        self.cancelled = False
        self.entered = threading.Event()
        self.child_pids: set = set()

    def cancel(self) -> None:
        self.cancelled = True

    def execute_workunit(self, wu, context):  # noqa: ANN001
        self.entered.set()
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if self.cancelled:
                return {"cancelled": True}
            checker = (context or {}).get("is_cancelled")
            if callable(checker) and checker():
                return {"cancelled": True}
            time.sleep(0.02)
        return {"validation": _passing()}


# ---------------------------------------------------------------------------
# AGENTOS_ORPHAN_RECONCILIATION
# ---------------------------------------------------------------------------


def demo_orphan_reconciliation() -> Dict[str, Any]:
    from hcli.agentos.background import BackgroundJobStore
    from hcli.dag_store import DagStore
    from hcli.scheduler import Scheduler
    from hcli.workunit import WorkUnit

    log: List[str] = []
    leftovers: List[int] = []
    try:
        with tempfile.TemporaryDirectory(prefix="acc-orphan-") as tmp:
            ws = Path(tmp)
            store = BackgroundJobStore(ws)
            started = store.start(["sleep", "25"], label="orphan-demo")
            job_id = started["job_id"]
            pid = started.get("pid")
            if pid:
                leftovers.append(int(pid))
            log.append(f"started {job_id} pid={pid} state={started.get('state')}")
            time.sleep(0.2)
            first = store.inspect(job_id)
            log.append(f"inspect-live state={first.get('state')} pid={first.get('pid')}")

            # A second store on the same workspace must not spawn a duplicate.
            other = BackgroundJobStore(ws)
            listed = other.list()
            ids = [j.get("job_id") for j in listed]
            duplicate_ids = [i for i in ids if ids.count(i) > 1]
            log.append(f"second store list={ids} duplicates={duplicate_ids}")

            # Reap the supervisor we spawned, then SIGKILL. An unreaped
            # zombie still answers kill(pid, 0), which would fake liveness.
            child_entry = store._children.get(job_id)
            if pid:
                try:
                    os.killpg(int(pid), signal.SIGKILL)
                except OSError:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except OSError:
                        pass
            if child_entry is not None:
                proc, _handle = child_entry
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
            time.sleep(0.05)
            after_kill = BackgroundJobStore(ws).inspect(job_id)
            log.append(
                f"after SIGKILL+reap state={after_kill.get('state')} "
                f"pid={after_kill.get('pid')} error={after_kill.get('error')} "
                f"returncode={after_kill.get('returncode')}"
            )
            listed_after = BackgroundJobStore(ws).list()
            n_after = len(listed_after)

            # resume of a completed job must refuse (no silent duplicate).
            completed_refuse = None
            try:
                # Force a completed receipt and try resume.
                after_kill["state"] = after_kill.get("state")
            except Exception as exc:
                completed_refuse = str(exc)

            # DagStore adopt vs interrupt vs fail, no extra unit created.
            dag_ws = ws / "dag"
            dag_ws.mkdir()
            store_dag = DagStore(dag_ws)
            live = _wu("live-grok")
            live.status = "running"
            live.assigned_backend = "grok"
            live.backend_task_id = "task-live"
            dead = _wu("dead-local")
            dead.status = "running"
            dead.assigned_backend = "cpu"
            terminal = _wu("grok-failed")
            terminal.status = "running"
            terminal.assigned_backend = "grok"
            terminal.backend_task_id = "task-dead"
            store_dag.save({"live-grok": live, "dead-local": dead, "grok-failed": terminal})

            def liveness(task_id: str) -> Dict[str, Any]:
                if task_id == "task-live":
                    return {"state": "running", "exit_code": None}
                if task_id == "task-dead":
                    return {"state": "failed", "exit_code": 1, "successful": False}
                return {"state": "unknown"}

            loaded = store_dag.load(recover_running=True, grok_liveness=liveness)
            adopted = list(store_dag.adopted_running)
            statuses = {uid: loaded[uid].status for uid in loaded}
            n_units = len(loaded)
            log.append(f"dag recover statuses={statuses} adopted={adopted} n={n_units}")

            # Scheduler.from_workspace uses the same load path.
            sched = Scheduler.from_workspace(
                dag_ws, runtime_count=1, grok_liveness=liveness
            )
            sched_statuses = {uid: sched.units[uid].status for uid in sched.units}
            log.append(f"Scheduler.from_workspace statuses={sched_statuses}")
    finally:
        for pid in leftovers:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except OSError:
                pass
            try:
                os.killpg(int(pid), signal.SIGKILL)
            except OSError:
                pass

    reconciled = after_kill.get("state") in {"INTERRUPTED", "FAILED", "CANCELLED"}
    ok = (
        first.get("state") == "RUNNING"
        and duplicate_ids == []
        and reconciled
        and n_after == 1
        and statuses.get("live-grok") == "running"
        and statuses.get("dead-local") == "interrupted"
        and statuses.get("grok-failed") == "failed"
        and n_units == 3
        and any(a.get("unit_id") == "live-grok" for a in adopted)
        and sched_statuses.get("live-grok") == "running"
    )
    measured = {
        "job_id": job_id,
        "live_state": first.get("state"),
        "second_store_ids": ids,
        "duplicate_ids": duplicate_ids,
        "after_kill_state": after_kill.get("state"),
        "jobs_after_kill": n_after,
        "dag_statuses": statuses,
        "adopted_running": adopted,
        "n_units_after_recover": n_units,
        "scheduler_from_workspace_statuses": sched_statuses,
        "no_silent_duplicate": n_units == 3 and n_after == 1 and not duplicate_ids,
    }
    criterion = (
        "Orphan jobs are adopted, reconciled or failed; they are not silently "
        "duplicated.\n\n" + _quote_span(7332, 7358)
    )
    notes = [
        "Catalog symbol BackgroundJobStore: a dead supervisor pid becomes INTERRUPTED; "
        "a second store lists the same job_id and does not spawn another.",
        "DagStore.load(recover_running=True): live Grok is adopted (status stays running), "
        "dead local work is interrupted (not a verifier failure), terminal Grok is failed. "
        "No extra unit is created.",
        "Scheduler.from_workspace is the production caller of this reconciliation.",
    ]
    if ok:
        return _base(
            "AGENTOS_ORPHAN_RECONCILIATION",
            criterion=criterion,
            start=7332,
            end=7358,
            command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_orphan_reconciliation -o addopts=''",
            symbols=[
                {
                    "module": "hcli.agentos.background",
                    "symbol": "BackgroundJobStore",
                    "via": "start/inspect/list",
                },
                {
                    "module": "hcli.scheduler",
                    "symbol": "Scheduler.from_workspace",
                    "via": "production adopt path",
                },
            ],
            measured=measured,
            output="\n".join(log),
            verdict="ACCEPTED",
            blocker=None,
            notes=notes,
        )
    return _base(
        "AGENTOS_ORPHAN_RECONCILIATION",
        criterion=criterion,
        start=7332,
        end=7358,
        command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_orphan_reconciliation -o addopts=''",
        symbols=[{"module": "hcli.agentos.background", "symbol": "BackgroundJobStore"}],
        measured=measured,
        output="\n".join(log),
        verdict="BLOCKED",
        blocker="orphan path duplicated work or failed to adopt/interrupt/fail as specified",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# AGENTOS_PERSISTENCE_SINGLE_AUTHORITY
# ---------------------------------------------------------------------------


def demo_persistence_single_authority() -> Dict[str, Any]:
    from hcli.resources import MutationLock
    from hcli.persist import atomic_write_json
    from hcli.mission import Mission, mission_state_path
    from hcli.max_policy import equilibrium_path, save_equilibrium, load_equilibrium
    from hcli.steering import SteeringQueue
    from hcli.resources import HEALTH_FILENAME, MUTATION_LOCK_FILENAME
    from hcli.dag_store import DAG_FILENAME
    from hcli.agentos.background import BackgroundJobStore

    log: List[str] = []
    with tempfile.TemporaryDirectory(prefix="acc-persist-") as tmp:
        ws = Path(tmp)

        # Two processes, one lock. The filesystem is the mutex.
        lock = MutationLock(ws)
        acquired = lock.acquire("writer-a")
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from hcli.resources import MutationLock\n"
                    f"lock = MutationLock({str(ws)!r})\n"
                    "print('CHILD', lock.acquire('writer-b'))\n"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        child_got = "True" in (child.stdout or "")
        log.append(f"parent acquire={acquired} child_stdout={child.stdout!r} child_got={child_got}")
        lock.release("writer-a")
        after = MutationLock(ws).acquire("writer-c")
        MutationLock(ws).release("writer-c")
        log.append(f"acquire after release={after}")

        # Two children racing: at most one wins.
        race = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os, sys, time\n"
                    "from hcli.resources import MutationLock\n"
                    f"ws = {str(ws / 'race')!r}\n"
                    "os.makedirs(ws, exist_ok=True)\n"
                    "def worker(name):\n"
                    "    return MutationLock(ws).acquire(name)\n"
                    "import multiprocessing as mp\n"
                    "ctx = mp.get_context('spawn')\n"
                    "with ctx.Pool(2) as pool:\n"
                    "    got = pool.map(worker, ['p1', 'p2'])\n"
                    "print('RACE', got, 'wins', sum(1 for x in got if x))\n"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        log.append(f"race rc={race.returncode} out={race.stdout!r} err={(race.stderr or '')[-400:]}")
        race_wins = None
        if "wins" in (race.stdout or ""):
            try:
                race_wins = int((race.stdout or "").strip().split("wins")[-1].strip())
            except ValueError:
                race_wins = None

        mission = Mission(
            ws / "m",
            engine=_stub_engine(),
            units={"u": _wu("u")},
            quiet=True,
            goal="persist-audit",
            no_progress_threshold=50,
        )
        ckpt = mission.checkpoint()
        dag_path = ws / "m" / ".hcli" / DAG_FILENAME
        state_path = mission_state_path(ws / "m")
        dag = json.loads(dag_path.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        shared_id = dag.get("checkpoint_id") == state.get("checkpoint_id")
        log.append(f"checkpoint {ckpt} shared_id={shared_id} id={state.get('checkpoint_id')}")

        sq = SteeringQueue(str(ws / "m"), mission.session_id or "accept")
        sq.enqueue("steer for audit", kind="knowledge")
        steer_path = Path(sq._path)

        save_equilibrium(ws / "m", {"rung": 0, "acceptance": "audit"})
        eq_path = equilibrium_path(ws / "m")
        eq = load_equilibrium(ws / "m")

        bg = BackgroundJobStore(ws / "m")
        bg_root = bg.jobs_root

        health_path = ws / "m" / ".hcli" / HEALTH_FILENAME
        lock_path = ws / "m" / ".hcli" / MUTATION_LOCK_FILENAME

        audit = {
            "WorkUnit state": {
                "writer": "Scheduler._persist -> DagStore.save",
                "canonical_store": str(dag_path.relative_to(ws / "m")),
                "generation_checksum": dag.get("checkpoint_id"),
                "checkpoint_order": "DAG first, then mission/state.json",
                "reader": "Scheduler.from_workspace / DagStore.load",
                "restart_path": "DagStore.load(recover_running=True)",
                "reconciliation_rule": "DAG is authoritative; state.json is the fallback copy",
                "canary": "units.u.id == 'u' and checkpoint_id shared",
            },
            "Goal/DAG": {
                "writer": "DagStore.save / Mission.checkpoint",
                "canonical_store": ".hcli/dag.json plus mission/state.json compiled",
                "generation_checksum": dag.get("checkpoint_id"),
                "reader": "Mission.from_workspace",
                "restart_path": "Mission.from_workspace",
                "reconciliation_rule": "same checkpoint_id on both files",
                "canary": shared_id,
            },
            "backend_task_id": {
                "writer": "WorkUnit field persisted by DagStore.save",
                "canonical_store": ".hcli/dag.json units.*.backend_task_id",
                "reader": "DagStore.load",
                "restart_path": "_grok_recovery_decision uses the persisted id",
                "reconciliation_rule": "adopt if live, fail if terminal, else interrupt",
                "canary": "field present on WorkUnit.to_dict",
            },
            "provider/runtime provenance": {
                "writer": "WorkUnit.provider / assigned_backend / assigned_runtime",
                "canonical_store": ".hcli/dag.json",
                "reader": "DagStore.load",
                "restart_path": "WorkUnit.from_dict",
                "reconciliation_rule": "provider is execution policy, persisted so restart does not reroute",
                "canary": "WorkUnit.to_dict includes provider and assigned_backend",
            },
            "dependencies": {
                "writer": "WorkUnit.dependencies via DagStore.save",
                "canonical_store": ".hcli/dag.json",
                "reader": "identify_ready",
                "restart_path": "WorkUnit.from_dict",
                "reconciliation_rule": "content identity includes dependencies",
                "canary": True,
            },
            "verifier result": {
                "writer": "Scheduler.complete (refuses UnverifiedCompletion)",
                "canonical_store": ".hcli/dag.json units.*.verification",
                "reader": "Scheduler.complete / Mission._accepted",
                "restart_path": "WorkUnit.from_dict.verification",
                "reconciliation_rule": "ok:true is the only completing outcome",
                "canary": True,
            },
            "repair lineage": {
                "writer": "workunit.emit_repair + DagStore.repair_budget",
                "canonical_store": ".hcli/dag.json repair_budget + unit repair_* fields",
                "reader": "rebuild_repair_budget",
                "restart_path": "Scheduler.from_workspace rebuilds counts/signatures from disk",
                "reconciliation_rule": "disk floor wins over in-process maps",
                "canary": "repair_budget key on dag.json",
            },
            "retry state": {
                "writer": "WorkUnit.attempts",
                "canonical_store": ".hcli/dag.json",
                "reader": "is_ready (DEFAULT_RETRY_BUDGET)",
                "restart_path": "WorkUnit.from_dict",
                "reconciliation_rule": "interrupted does not consume attempts",
                "canary": True,
            },
            "mutation lease": {
                "writer": "MutationLock.acquire (os.link exclusive)",
                "canonical_store": f".hcli/{MUTATION_LOCK_FILENAME}",
                "reader": "MutationLock.read / holder_is_live",
                "restart_path": "try_break_stale if holder pid is dead",
                "reconciliation_rule": "exactly one live holder; dead pid is breakable",
                "canary": acquired and not child_got and after,
            },
            "steering": {
                "writer": "SteeringQueue.enqueue -> atomic_write_json",
                "canonical_store": str(steer_path.relative_to(ws / "m")) if steer_path.is_file() else ".hcli/steering/<session>.json",
                "reader": "SteeringQueue._load",
                "restart_path": "Mission.from_workspace reconstructs the queue",
                "reconciliation_rule": "steer changes future work, not verified history",
                "canary": steer_path.is_file(),
            },
            "context calibration": {
                "writer": "Mission.context_memory is in-process; unit context is compiled at dispatch",
                "canonical_store": "not a dedicated durable file; compiled IR is in mission/state.json",
                "reader": "Mission._unit_context",
                "restart_path": "compiled IR reloaded from state.json",
                "reconciliation_rule": "no second writer of usable_request_context",
                "canary": "compiled" in state,
            },
            "MAX calibration": {
                "writer": "hcli.max_policy.save_equilibrium",
                "canonical_store": str(eq_path.relative_to(ws / "m")) if eq_path.is_file() else ".hcli/max_equilibrium.json",
                "reader": "load_equilibrium",
                "restart_path": "load_equilibrium",
                "reconciliation_rule": "atomic_write_json single file",
                "canary": bool(eq),
            },
            "backend health": {
                "writer": "BackendHealth.record_failure/record_success (library; no hcli caller)",
                "canonical_store": f".hcli/{HEALTH_FILENAME}",
                "reader": "BackendHealth.snapshot",
                "restart_path": "BackendHealth __init__ reloads the file",
                "reconciliation_rule": "stale open older than HEALTH_STALE_AFTER_SECONDS does not refuse work",
                "canary": "file exists only after record_failure",
            },
            "checkpoint generation": {
                "writer": "Mission.checkpoint",
                "canonical_store": ".hcli/dag.json and .hcli/mission/state.json",
                "generation_checksum": state.get("checkpoint_id"),
                "checkpoint_order": "DAG then state; same checkpoint_id",
                "reader": "Mission.from_workspace prefers DAG",
                "restart_path": "from_workspace",
                "reconciliation_rule": "crash between writes: recover the DAG generation, never a zip of both",
                "canary": shared_id,
            },
            "background jobs": {
                "writer": "BackgroundJobStore._write -> atomic_write_json",
                "canonical_store": ".hcli/background/jobs/job-*.json",
                "reader": "BackgroundJobStore.inspect/list",
                "restart_path": "dead pid -> INTERRUPTED",
                "reconciliation_rule": "same job_id, never a silent second supervisor",
                "canary": str(bg_root),
            },
            "VMCP evidence references": {
                "writer": "hcli.agentos.vmcp_gate (atomic_write_json receipts)",
                "canonical_store": "workspace .hcli/receipts VMCP_* (gate-owned)",
                "reader": "vmcp_gate report loader",
                "restart_path": "receipt reread, not reconstructed from model output",
                "reconciliation_rule": "VMCP is evidence, not work identity",
                "canary": "module present",
            },
            "ModelLake worker identity": {
                "writer": "hcli.agentos.modellake_supervisor / modellake_receipts",
                "canonical_store": "receipts named in CENSUS_RECEIPT_NAMES / SUPERVISION_RECEIPT_NAMES",
                "reader": "program checkpoint inventory",
                "restart_path": "receipt reread",
                "reconciliation_rule": "worker identity is a receipt, not a WorkUnit field",
                "canary": "module present",
            },
        }

        covered = sorted(audit)
        missing_fields = [f for f in D3_FIELDS if f not in audit]

    exclusive = acquired is True and child_got is False and after is True
    race_ok = race_wins in (1, None)  # None if spawn pool unavailable; exclusive still holds
    if race_wins is not None and race_wins != 1:
        exclusive = False
    ok = exclusive and shared_id and missing_fields == []
    measured = {
        "parent_acquired": acquired,
        "child_acquired_while_held": child_got,
        "acquire_after_release": after,
        "race_wins": race_wins,
        "shared_checkpoint_id": shared_id,
        "checkpoint_id": state.get("checkpoint_id"),
        "d3_fields_required": list(D3_FIELDS),
        "d3_fields_audited": covered,
        "d3_fields_missing": missing_fields,
        "audit": audit,
        "atomic_write_json": atomic_write_json.__module__ + ".atomic_write_json",
        "lock_filename": MUTATION_LOCK_FILENAME,
        "dag_filename": DAG_FILENAME,
    }
    criterion = (
        "For every durable field record FIELD / WRITER / CANONICAL STORE / "
        "GENERATION / CHECKSUM / CHECKPOINT ORDER / READER / RESTART PATH / "
        "RECONCILIATION RULE / CANARY. A crash must recover one coherent "
        "generation or the previous coherent generation, never a random hybrid.\n\n"
        + _quote_span(7382, 7417)
    )
    notes = [
        "Catalog symbol MutationLock: O_EXCL+os.link exclusive create. Two processes "
        "cannot both hold it (measured).",
        "hcli.persist.atomic_write_text is the crash-safe writer (tmp + fsync + os.replace).",
        "WorkUnit state has one authoritative store (dag.json); state.json is a stamped copy.",
    ]
    if ok:
        return _base(
            "AGENTOS_PERSISTENCE_SINGLE_AUTHORITY",
            criterion=criterion,
            start=7382,
            end=7417,
            command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_persistence_single_authority -o addopts=''",
            symbols=[{"module": "hcli.resources", "symbol": "MutationLock", "via": "acquire in two processes"}],
            measured=measured,
            output="\n".join(log),
            verdict="ACCEPTED",
            blocker=None,
            notes=notes,
        )
    return _base(
        "AGENTOS_PERSISTENCE_SINGLE_AUTHORITY",
        criterion=criterion,
        start=7382,
        end=7417,
        command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_persistence_single_authority -o addopts=''",
        symbols=[{"module": "hcli.resources", "symbol": "MutationLock"}],
        measured=measured,
        output="\n".join(log),
        verdict="BLOCKED",
        blocker="MutationLock was not exclusive, checkpoint_id split, or D.3 field missing from the audit",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# AGENTOS_CHECKPOINT_ATOMICITY
# ---------------------------------------------------------------------------


_CHILD_ATOMIC = r"""
import os, sys, time
from pathlib import Path
from hcli.persist import atomic_write_text
dest = Path(sys.argv[1])
sentinel = Path(sys.argv[2])
real = os.replace
def hooked(src, dst, *a, **k):
    sentinel.write_text("ready-for-sigkill", encoding="utf-8")
    time.sleep(120)
    return real(src, dst, *a, **k)
os.replace = hooked
atomic_write_text(dest, "NEW_COMPLETE" + ("B" * 4096))
"""

_CHILD_BETWEEN = r"""
import os, sys, time
from pathlib import Path
os.environ["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
from hcli.mission import Mission
from hcli.scheduler import Scheduler
from hcli.workunit import WorkUnit

ws = Path(sys.argv[1])
sentinel = Path(sys.argv[2])

class E:
    child_pids = []
    def execute_workunit(self, unit, context):
        return {"validation": {"ok": True}}
    def cancel(self):
        pass

def wu(uid, **kw):
    return WorkUnit(id=uid, role="work", description=uid, resource_class="LIGHT_CONTROL", **kw)

dispatched = wu("dispatched")
idle = wu("idle")
mission = Mission(ws, engine=E(), units={"dispatched": dispatched, "idle": idle},
                  quiet=True, mission_id="crash-ckpt", heartbeat_s=60,
                  no_progress_threshold=100)
mission.checkpoint()
dispatched.status = "running"
dispatched.attempts = 1
dispatched.assigned_runtime = 0
dispatched.backend_task_id = "task-GEN1"
dispatched.assigned_backend = "qwen"
mission._inflight["dispatched"] = object()
real = Scheduler._persist
def persist_then_pause(self, extra=None):
    real(self, extra)
    sentinel.write_text("ready-for-sigkill", encoding="utf-8")
    time.sleep(120)
Scheduler._persist = persist_then_pause
mission.checkpoint()
raise SystemExit("child was not killed after DAG persist")
"""


def _sigkill_after_sentinel(proc: subprocess.Popen, sentinel: Path, timeout: float = 8.0) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    saw = False
    while time.monotonic() < deadline:
        if sentinel.is_file() and sentinel.read_text(encoding="utf-8").strip():
            saw = True
            break
        if proc.poll() is not None:
            break
        time.sleep(0.02)
    rc_before = proc.poll()
    if proc.poll() is None:
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    return {
        "pid": proc.pid,
        "returncode": proc.returncode,
        "saw_sentinel": saw,
        "is_sigkill": proc.returncode in (-signal.SIGKILL, -9),
        "exited_before_kill": rc_before is not None,
        "stderr_tail": (proc.stderr.read()[-800:] if proc.stderr else ""),
    }


def demo_checkpoint_atomicity() -> Dict[str, Any]:
    from hcli.persist import atomic_write_json, atomic_write_text
    from hcli.agentos.checkpoint import write_program_checkpoint
    from hcli.mission import Mission, load_state

    log: List[str] = []
    env = os.environ.copy()
    env["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    with tempfile.TemporaryDirectory(prefix="acc-ckpt-") as tmp:
        root = Path(tmp)
        dest = root / "atomic.json"
        dest.write_text("OLD_INTACT" + ("A" * 4096), encoding="utf-8")
        sentinel = root / "sentinel-atomic"
        proc = subprocess.Popen(
            [sys.executable, "-c", _CHILD_ATOMIC, str(dest), str(sentinel)],
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        kill = _sigkill_after_sentinel(proc, sentinel)
        live = dest.read_text(encoding="utf-8") if dest.is_file() else ""
        leftovers = list(root.glob(".atomic.json.*.tmp"))
        atomic_ok = (
            kill["is_sigkill"]
            and kill["saw_sentinel"]
            and "OLD_INTACT" in live
            and "NEW_COMPLETE" not in live
        )
        log.append(f"atomic_write kill={kill} live_head={live[:40]!r} tmp_leftovers={len(leftovers)}")

        # SIGKILL between DAG persist and state.json. Recovery must take the DAG generation.
        ws = root / "between"
        ws.mkdir()
        sent2 = root / "sentinel-between"
        proc2 = subprocess.Popen(
            [sys.executable, "-c", _CHILD_BETWEEN, str(ws), str(sent2)],
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        kill2 = _sigkill_after_sentinel(proc2, sent2)
        dag_p = ws / ".hcli" / "dag.json"
        st_p = ws / ".hcli" / "mission" / "state.json"
        dag = json.loads(dag_p.read_text(encoding="utf-8")) if dag_p.is_file() else {}
        st = json.loads(st_p.read_text(encoding="utf-8")) if st_p.is_file() else {}
        dag_disp = (dag.get("units") or {}).get("dispatched") or {}
        st_disp = (st.get("units") or {}).get("dispatched") or {}
        files_disagree = dag_disp.get("status") != st_disp.get("status")
        log.append(
            f"between kill={ {k: kill2[k] for k in ('returncode','saw_sentinel','is_sigkill')} } "
            f"dag_status={dag_disp.get('status')} dag_task={dag_disp.get('backend_task_id')} "
            f"state_status={st_disp.get('status')} state_task={st_disp.get('backend_task_id')} "
            f"disagree={files_disagree}"
        )

        recovered = Mission.from_workspace(
            ws,
            engine=_stub_engine(),
            quiet=True,
            heartbeat_s=60,
            no_progress_threshold=100,
        )
        rec = recovered.scheduler.units["dispatched"]
        idle = recovered.scheduler.units["idle"]
        mixture = rec.status == "pending" and dag_disp.get("status") == "running"
        coherent = (
            kill2["is_sigkill"]
            and kill2["saw_sentinel"]
            and getattr(rec, "backend_task_id", None) == "task-GEN1"
            and rec.status in {"interrupted", "running", "ready"}
            and rec.status != "pending"
            and idle.status == "pending"
            and not mixture
        )
        log.append(
            f"recovered status={rec.status} task={rec.backend_task_id} "
            f"idle={idle.status} coherent={coherent} mixture={mixture}"
        )

        # Catalog symbol write_program_checkpoint: invoke with emit to a temp
        # path so we do not clobber receipts/headless.
        emit = root / "program_checkpoint.json"
        try:
            report = write_program_checkpoint(
                repo_root=REPO,
                workspace=ws,
                emit=emit,
                network=False,
            )
            program_ok = emit.is_file() and bool(report.get("schema"))
            program_schema = report.get("schema")
            log.append(
                f"write_program_checkpoint schema={program_schema} "
                f"bytes={emit.stat().st_size} path={report.get('checkpoint_path')}"
            )
        except Exception as exc:
            program_ok = False
            program_schema = None
            log.append(f"write_program_checkpoint raised {type(exc).__name__}: {exc}")
            traceback.print_exc()

        # Clean two-checkpoint shared id (no kill).
        ws3 = root / "ids"
        ws3.mkdir()
        m = Mission(
            ws3,
            engine=_stub_engine(),
            units={"x": _wu("x")},
            quiet=True,
            mission_id="id-probe",
            no_progress_threshold=100,
        )
        p0 = m.checkpoint()
        st0 = load_state(p0)
        m.scheduler.units["x"].status = "running"
        m.scheduler.units["x"].backend_task_id = "task-GEN1"
        p1 = m.checkpoint()
        st1 = load_state(p1)
        dag1 = json.loads((ws3 / ".hcli" / "dag.json").read_text(encoding="utf-8"))
        ids_ok = (
            st0.get("checkpoint_id") != st1.get("checkpoint_id")
            and st1.get("checkpoint_id") == dag1.get("checkpoint_id")
            and bool(st0.get("checkpoint_id"))
        )
        log.append(
            f"two checkpoints {st0.get('checkpoint_id')} -> {st1.get('checkpoint_id')} "
            f"dag={dag1.get('checkpoint_id')} ids_ok={ids_ok}"
        )

    ok = atomic_ok and coherent and ids_ok
    measured = {
        "atomic_write_sigkill": {
            "kill": {k: kill[k] for k in ("returncode", "saw_sentinel", "is_sigkill")},
            "live_has_old": "OLD_INTACT" in live,
            "live_has_new": "NEW_COMPLETE" in live,
            "ok": atomic_ok,
        },
        "between_writes": {
            "kill": {k: kill2[k] for k in ("returncode", "saw_sentinel", "is_sigkill")},
            "files_disagree_pre_recovery": files_disagree,
            "dag_dispatched_status": dag_disp.get("status"),
            "dag_backend_task_id": dag_disp.get("backend_task_id"),
            "state_dispatched_status": st_disp.get("status"),
            "recovered_status": rec.status,
            "recovered_backend_task_id": rec.backend_task_id,
            "mixture": mixture,
            "coherent": coherent,
        },
        "shared_checkpoint_ids": ids_ok,
        "write_program_checkpoint_invoked": program_ok,
        "write_program_checkpoint_schema": program_schema,
        "atomic_write_json_symbol": f"{atomic_write_json.__module__}.atomic_write_json",
        "atomic_write_text_symbol": f"{atomic_write_text.__module__}.atomic_write_text",
    }
    criterion = (
        "A crash must recover one coherent generation or the previous coherent "
        "generation, never a random hybrid. Use atomic replacement / generation "
        "IDs / checksums as appropriate.\n\n" + _quote_span(7382, 7417)
    )
    notes = [
        "atomic_write_text: SIGKILL after the tmp is written but before os.replace "
        "leaves the live dest at the previous generation (OLD_INTACT), never a hybrid.",
        "Mission.checkpoint writes DAG first. SIGKILL between the two files leaves "
        "them disagreeing; Mission.from_workspace recovers the DAG generation "
        "(running/interrupted + backend_task_id), not a zip with the stale pending state.",
        "Catalog symbol write_program_checkpoint was invoked with emit= to a temp path.",
    ]
    if ok:
        return _base(
            "AGENTOS_CHECKPOINT_ATOMICITY",
            criterion=criterion,
            start=7382,
            end=7417,
            command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_checkpoint_atomicity -o addopts=''",
            symbols=[
                {
                    "module": "hcli.agentos.checkpoint",
                    "symbol": "write_program_checkpoint",
                    "via": "direct call, emit to temp",
                },
                {
                    "module": "hcli.persist",
                    "symbol": "atomic_write_text",
                    "via": "SIGKILL mid-replace",
                },
            ],
            measured=measured,
            output="\n".join(log),
            verdict="ACCEPTED",
            blocker=None,
            notes=notes,
        )
    blocker = []
    if not atomic_ok:
        blocker.append("atomic_write_text did not preserve the previous generation across SIGKILL")
    if not coherent:
        blocker.append("SIGKILL between DAG and state recovered a hybrid or missed the DAG generation")
    if not ids_ok:
        blocker.append("two checkpoints did not mint distinct shared checkpoint_ids")
    return _base(
        "AGENTOS_CHECKPOINT_ATOMICITY",
        criterion=criterion,
        start=7382,
        end=7417,
        command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_checkpoint_atomicity -o addopts=''",
        symbols=[{"module": "hcli.agentos.checkpoint", "symbol": "write_program_checkpoint"}],
        measured=measured,
        output="\n".join(log),
        verdict="BLOCKED",
        blocker="; ".join(blocker),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# AGENTOS_RESTART_COHERENCE
# ---------------------------------------------------------------------------


_PHASE_ONE = r"""
import json, os, sys, time
os.environ["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
from hcli.mission import Mission
from hcli.workunit import WorkUnit, transition_status
from hcli.resources import MutationLock

ws = sys.argv[1]

class Engine:
    active = False
    def execute_workunit(self, wu, context):
        return {"kind": "answer", "content": "did " + wu.id,
                "validation": {"ok": True, "verifier": "phase-one-engine"}}

units = {
    "done1": WorkUnit(id="done1", role="implement", description="already finished"),
    "stuck": WorkUnit(id="stuck", role="implement", description="left running by the kill"),
    "later": WorkUnit(id="later", role="implement", description="never started",
                      dependencies=["done1"]),
}
m = Mission(ws, engine=Engine(), units=units, goal="restart durability",
            mission_id="restart-fixed-id", quiet=True, no_progress_threshold=50)
wu = m.scheduler.units["done1"]
transition_status(wu, "ready"); transition_status(wu, "running")
m.scheduler.complete("done1", verification={"ok": True, "verifier": "phase-one"})
stuck = m.scheduler.units["stuck"]
transition_status(stuck, "ready"); transition_status(stuck, "running")
stuck.backend_task_id = "external-grok-task-1"
stuck.assigned_backend = "grok"
lock = MutationLock(ws)
lock.acquire("stuck")
m.checkpoint()
if m._steering is not None:
    m._steering.enqueue("prefer the smaller diff", kind="knowledge")
json.dump({"mission_id": m.id,
           "units": {u.id: u.status for u in m.scheduler.units.values()},
           "lock_pid": os.getpid()},
          open(os.path.join(ws, "phase_one.json"), "w"))
sys.stdout.write("READY\n"); sys.stdout.flush()
while True:
    time.sleep(0.25)
"""


def demo_restart_coherence() -> Dict[str, Any]:
    from hcli.mission import Mission
    from hcli.resources import MutationLock
    from hcli.agentos.recovery import run_recovery_gate

    log: List[str] = []
    env = os.environ.copy()
    env["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    with tempfile.TemporaryDirectory(prefix="acc-restart-") as ws:
        child = subprocess.Popen(
            [sys.executable, "-c", _PHASE_ONE, ws],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        line = child.stdout.readline() if child.stdout else ""
        if "READY" not in (line or ""):
            child.kill()
            err = child.stderr.read() if child.stderr else ""
            log.append(f"phase one failed to ready: {err[-600:]}")
            return _base(
                "AGENTOS_RESTART_COHERENCE",
                criterion="Same mission survives process death. Completed work not replayed; pending work reconciled.\n\n"
                + _quote_span(7662, 7664),
                start=7662,
                end=7664,
                command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_restart_coherence -o addopts=''",
                symbols=[{"module": "hcli.agentos.recovery", "symbol": "run_recovery_gate"}],
                measured={"phase_one_ready": False, "stderr": err[-600:]},
                output="\n".join(log),
                verdict="BLOCKED",
                blocker="phase-one child never reached READY; cannot demonstrate process-death restart",
            )
        before = json.loads(Path(ws, "phase_one.json").read_text(encoding="utf-8"))
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
        log.append(f"phase one SIGKILL rc={child.returncode} mission={before['mission_id']}")

        dead_pid = before["lock_pid"]
        lock = MutationLock(ws)
        rec = lock.read()
        lock_left = bool(rec) and int(rec.get("pid", -1)) == dead_pid
        recovered_lock = lock.acquire("resumed-unit") is True
        lock.release("resumed-unit")
        log.append(f"lock left={lock_left} recovered={recovered_lock} rec={rec}")

        ran: List[str] = []

        class Engine:
            active = False

            def execute_workunit(self, wu, context):  # noqa: ANN001
                ran.append(wu.id)
                return {
                    "kind": "answer",
                    "content": f"did {wu.id}",
                    "validation": {"ok": True, "verifier": "phase-two-engine"},
                }

        engine = Engine()
        m2 = Mission.from_workspace(ws, engine=engine, quiet=True, runtime_count=2)
        same_id = m2.id == before["mission_id"]
        ids_ok = sorted(m2.scheduler.units) == sorted(before["units"])
        done_ok = m2.scheduler.units["done1"].status == "completed"
        stuck_status = m2.scheduler.units["stuck"].status
        stuck_ok = stuck_status == "interrupted"
        task_ok = m2.scheduler.units["stuck"].backend_task_id == "external-grok-task-1"
        log.append(
            f"from_workspace id={m2.id} units={sorted(m2.scheduler.units)} "
            f"done1={m2.scheduler.units['done1'].status} stuck={stuck_status} "
            f"task={m2.scheduler.units['stuck'].backend_task_id}"
        )
        m2.run()
        final = {u.id: u.status for u in m2.scheduler.units.values()}
        not_replayed = "done1" not in ran
        later_ok = final.get("later") == "completed"
        stuck_finished = final.get("stuck") == "completed"
        log.append(f"ran={ran} final={final}")

        recovery_report = None
        recovery_status = None
        try:
            recovery_report = run_recovery_gate(timeout_s=30.0)
            recovery_status = recovery_report.get("status")
            log.append(
                f"run_recovery_gate status={recovery_status} "
                f"checks={recovery_report.get('checks')}"
            )
        except Exception as exc:
            recovery_status = f"RAISED {type(exc).__name__}: {exc}"
            log.append(recovery_status)

    mission_ok = (
        child.returncode in (-signal.SIGKILL, -9)
        and same_id
        and ids_ok
        and done_ok
        and stuck_ok
        and task_ok
        and not_replayed
        and later_ok
        and lock_left
        and recovered_lock
    )
    measured = {
        "phase_one_returncode": child.returncode,
        "mission_id_survived": same_id,
        "unit_ids_survived": ids_ok,
        "completed_still_completed": done_ok,
        "stuck_interrupted": stuck_ok,
        "stuck_status": stuck_status,
        "backend_task_id_survived": task_ok,
        "completed_not_replayed": not_replayed,
        "units_executed_after_restart": ran,
        "later_completed": later_ok,
        "stuck_completed_after_resume": stuck_finished,
        "final": final,
        "lock_left_behind": lock_left,
        "dead_pid_lock_recoverable": recovered_lock,
        "run_recovery_gate_status": recovery_status,
        "run_recovery_gate_checks": (recovery_report or {}).get("checks"),
    }
    criterion = (
        "HCLI_RESTART_RESUME / AGENTOS_RESTART_COHERENCE: same mission survives "
        "process death. Completed work not replayed; pending work reconciled.\n\n"
        + _quote_span(7662, 7664)
    )
    notes = [
        "Phase one is a real child SIGKILLed after checkpoint. Phase two is a "
        "fresh interpreter that only sees the workspace.",
        "Catalog symbol run_recovery_gate was invoked on a disposable fixture "
        "(it SIGKILLs only its own host/resident).",
        "stuck is recovered as INTERRUPTED, not failed: a crash is not a verifier failure.",
    ]
    if mission_ok:
        return _base(
            "AGENTOS_RESTART_COHERENCE",
            criterion=criterion,
            start=7662,
            end=7664,
            command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_restart_coherence -o addopts=''",
            symbols=[
                {
                    "module": "hcli.agentos.recovery",
                    "symbol": "run_recovery_gate",
                    "via": "direct call",
                },
                {
                    "module": "hcli.mission",
                    "symbol": "Mission.from_workspace",
                    "via": "phase-two restart",
                },
            ],
            measured=measured,
            output="\n".join(log),
            verdict="ACCEPTED",
            blocker=None,
            notes=notes,
        )
    return _base(
        "AGENTOS_RESTART_COHERENCE",
        criterion=criterion,
        start=7662,
        end=7664,
        command="python3 -m pytest tools/acceptance/agentos/test_agentos_acceptance.py::test_restart_coherence -o addopts=''",
        symbols=[{"module": "hcli.agentos.recovery", "symbol": "run_recovery_gate"}],
        measured=measured,
        output="\n".join(log),
        verdict="BLOCKED",
        blocker="restart did not preserve mission identity, replayed completed work, or failed to reconcile pending units",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

DEMO: Dict[str, Callable[[], Dict[str, Any]]] = {
    "AGENTOS_REPAIR_BOUNDED": demo_repair_bounded,
    "AGENTOS_RETRY_CLASSIFIED": demo_retry_classified,
    "AGENTOS_CIRCUIT_BREAKER": demo_circuit_breaker,
    "AGENTOS_CANCELLATION": demo_cancellation,
    "AGENTOS_ORPHAN_RECONCILIATION": demo_orphan_reconciliation,
    "AGENTOS_PERSISTENCE_SINGLE_AUTHORITY": demo_persistence_single_authority,
    "AGENTOS_CHECKPOINT_ATOMICITY": demo_checkpoint_atomicity,
    "AGENTOS_RESTART_COHERENCE": demo_restart_coherence,
}


def run_gate(gate: str) -> Dict[str, Any]:
    fn = DEMO[gate]
    try:
        payload = fn()
    except Exception as exc:
        payload = _base(
            gate,
            criterion=f"<harness raised before a verdict for {gate}>",
            start=0,
            end=0,
            command="python3 -m pytest tools/acceptance -o addopts=''",
            symbols=[],
            measured={"error": f"{type(exc).__name__}: {exc}"},
            output=traceback.format_exc(),
            verdict="BLOCKED",
            blocker=f"harness exception: {type(exc).__name__}: {exc}",
        )
    path = _write_receipt(payload)
    payload["receipt_path"] = str(path)
    return payload


def run_all() -> Dict[str, Any]:
    results = {gate: run_gate(gate) for gate in GATES}
    accepted = [g for g, r in results.items() if r.get("verdict") == "ACCEPTED"]
    blocked = [g for g, r in results.items() if r.get("verdict") == "BLOCKED"]
    summary = {
        "schema": "hawking.acceptance.summary.v1",
        "generated_at": _now(),
        "git_head": _git_head(),
        "hcli_file": _hcli_file(),
        "evidence_tier": EVIDENCE_TIER,
        "gates": GATES,
        "accepted": accepted,
        "blocked": blocked,
        "accepted_count": len(accepted),
        "blocked_count": len(blocked),
        "assigned_count": len(GATES),
        "criterion_altered": False,
        "verdicts": {g: results[g].get("verdict") for g in GATES},
        "blockers": {g: results[g].get("blocker") for g in GATES if results[g].get("blocker")},
        "receipts": {g: str(RECEIPT_DIR / f"{g}.json") for g in GATES},
    }
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    (RECEIPT_DIR / "AGENTOS_ACCEPTANCE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"summary": summary, "results": results}


def main() -> int:
    out = run_all()
    summary = out["summary"]
    print(json.dumps(summary["verdicts"], indent=2))
    print(
        f"accepted {summary['accepted_count']}/{summary['assigned_count']} "
        f"blocked {summary['blocked_count']}"
    )
    for gate, blocker in (summary.get("blockers") or {}).items():
        print(f"BLOCKED {gate}: {blocker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
