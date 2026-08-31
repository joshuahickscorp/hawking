"""PROFILE HCLI AS SEARCH SPACE — rank the resident's own wall, then stop.

Hawking that cannot treat its own machinery as a candidate will keep paying
the same costs forever. This sidecar attributes process wall of HCLI-shaped
work (scheduler decision, tool routing, verifier, git query, spawn, I/O,
context construction) as SELF_MEASURED_DIRTY numbers that rank and prune.
They decide nothing, they do not promote, and they are not a GPU result.

A profiler that only reports is worthless. Every attributed cost carries
why it exists, what would remove it (information / call / copy / wait /
re-read), and the cheapest falsifier. The worked example is already on
disk: `git status` on this ~43GB tree took minutes and now takes ~0.2s
because `--no-optional-locks` skips the index refresh
(`tools/future/_common.git`). HCLI `git.status` still does not pass that
flag. This module recovers that shape; it does not re-run the lock-taking
arm, and it will not invoke `hcli` `git.status` on this tree.

Refuses: GPU lease, cargo, bare `git status`, ranking with no timings,
an "actionable" cost with no removal hypothesis, any hardware-named field.

Cannot establish: a live HCLI session's queue-wait or idle calendar time
(those need a ledger of ready_at/running_at), compile-wait (cargo is
forbidden), or that index-refresh is the proven cause of the historical
minutes (the before-arm recreates the lock incident).
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import ast
import inspect
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from tools.future._common import (
    GIT_TIMEOUT_S,
    HARDWARE_FIELDS,
    RECEIPTS,
    REPO,
    HardwareClaimError,
    git,
    write_receipt,
)

RECEIPT = "HCLI_SELF_PROFILE.json"
SCHEMA = "hawking.future.hcli_self_profile.v1"
RECORDED_BY = "tools/future/hcli_self_profile.py"
VERSION = 1

DEFAULT_REPEATS = 3
MIN_REPEATS = 2

COST_BUCKETS: tuple[str, ...] = (
    "scheduler_decision",
    "queue_wait",
    "subprocess_launch",
    "source_receipt_io",
    "context_construction",
    "tool_routing",
    "verifier_overhead",
    "git_query",
    "compile_wait",
    "process_idle",
)

# Analogs this process can honestly time. The other three are attributed as
# UNKNOWN with a reason: inventing a duration for them is the failure mode.
MEASURABLE: frozenset[str] = frozenset(
    {
        "scheduler_decision",
        "subprocess_launch",
        "source_receipt_io",
        "context_construction",
        "tool_routing",
        "verifier_overhead",
        "git_query",
    }
)

REMOVAL_KINDS: frozenset[str] = frozenset(
    {"information", "call", "copy", "wait", "re-read"}
)

HCLI_REL = {
    "scheduler": "hcli/scheduler.py",
    "tool_registry": "hcli/tool_registry.py",
    "verifier": "hcli/verifier_pipeline.py",
    "engine": "hcli/engine.py",
    "workunit": "hcli/workunit.py",
    "ledger": "hcli/ledger.py",
    "executors": "hcli/executors.py",
}

FORBIDDEN_HCLI_INVOKE = frozenset({"git.status"})

CLAIM_BOUNDARY = (
    "Static sidecar artifact. Process-wall attribution of this Python is "
    "SELF_MEASURED_DIRTY: it ranks and prunes and decides nothing. It is not "
    "PROTECTED_ABSOLUTE, not a GPU result, and not a hardware field."
)


class RankRefused(ValueError):
    """No timing data, so a ranking would be arbitrary."""


class ActionableRefused(ValueError):
    """A cost without why / removal / falsifier is not reportable as actionable."""


class CompileWaitForbidden(ValueError):
    """compile_wait requires cargo; this campaign will not run it."""


class LiveGitStatusForbidden(ValueError):
    """Bare git status / hcli git.status takes the index lock on this tree."""


class SourceUnavailable(ValueError):
    """A required source was unseen in this checkout, HEAD, and the editable install."""


# ---------------------------------------------------------------------------
# Guards. A guard nobody has watched fail is not a guard.
# ---------------------------------------------------------------------------


def assert_timing_field_legal(name: str) -> None:
    """Refuse hardware-named fields even when the value is a Python timing."""
    if name in HARDWARE_FIELDS:
        raise HardwareClaimError(
            f"{name}: sidecar has no GPU authority; hardware field names are "
            "illegal even for python wall. Put the number under self_timing "
            "as median_ms / samples_ms."
        )


def refuse_bare_git_status(argv: Iterable[str] | None = None) -> None:
    """The lock-taking arm is the incident. This profiler will not recreate it."""
    tokens = [str(x) for x in (argv or ("git", "status"))]
    has_status = "status" in tokens
    has_git = any(Path(t).name == "git" for t in tokens) or tokens[:1] == ["git"]
    has_flag = "--no-optional-locks" in tokens
    if has_git and has_status and not has_flag:
        raise LiveGitStatusForbidden(
            "refusing git status without --no-optional-locks: on this ~43GB "
            "tree it refreshes the index, takes .git/index.lock, has run for "
            "minutes, and a timeout SIGKILLs git while holding the lock"
        )


def refuse_hcli_git_status_invoke(name: str) -> None:
    if name in FORBIDDEN_HCLI_INVOKE:
        raise LiveGitStatusForbidden(
            f"refusing to invoke hcli tool {name!r}: argv is git status "
            "without --no-optional-locks (see recover_hcli_git_status)"
        )
    raise ValueError(f"this profiler does not invoke {name!r}")


def record_compile_wait_as_measured(value_ms: float) -> None:
    """CPU proxy of cargo is not compile wait. The refusal has to fire."""
    raise CompileWaitForbidden(
        f"compile_wait={value_ms!r} ms refused: cargo build is forbidden "
        "(shared target-dir workspace/ops/build/rust; live campaign). "
        "See tools/future/turnaround.py GPU_OR_BUILD_PHASES."
    )


def as_actionable(record: Mapping[str, Any]) -> dict[str, Any]:
    """A cost is actionable only with why, a typed removal, and a falsifier."""
    cost = record.get("cost")
    why = str(record.get("why_this_cost_exists") or "").strip()
    removal = record.get("removal")
    falsifier = str(record.get("cheapest_falsifier") or "").strip()
    missing: list[str] = []
    if not cost:
        missing.append("cost")
    if not why:
        missing.append("why_this_cost_exists")
    if not isinstance(removal, Mapping):
        missing.append("removal")
        kind = None
        what = ""
    else:
        kind = str(removal.get("kind") or "").strip()
        what = str(removal.get("what_would_remove_it") or "").strip()
        if kind not in REMOVAL_KINDS:
            missing.append("removal.kind")
        if not what:
            missing.append("removal.what_would_remove_it")
    if not falsifier:
        missing.append("cheapest_falsifier")
    if missing:
        raise ActionableRefused(
            f"cost {cost!r} is not actionable; missing {missing}"
        )
    return {
        "cost": cost,
        "why_this_cost_exists": why,
        "removal": {"kind": kind, "what_would_remove_it": what},
        "cheapest_falsifier": falsifier,
        "actionable": True,
        "status": "HYPOTHESIS",
        "does_not_decide": True,
        "worked_example": bool(record.get("worked_example")),
    }


def rank_attributed_costs(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank MEASURED self-timings by median_ms descending. No data → refuse.

    UNKNOWN / refused buckets are not ranked. A ranking of an empty measured
    set would be arbitrary, so it raises rather than inventing an order.
    """
    measured: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("state") != "MEASURED_SELF_DIRTY":
            continue
        timing = rec.get("self_timing")
        if not isinstance(timing, Mapping):
            continue
        median = timing.get("median_ms")
        if isinstance(median, bool) or not isinstance(median, (int, float)):
            continue
        for key in timing:
            assert_timing_field_legal(str(key))
        measured.append(
            {
                "cost": rec.get("cost"),
                "median_ms": float(median),
                "n": timing.get("n"),
                "rank_use": "SELF_MEASURED_DIRTY rank and prune; decide nothing",
            }
        )
    if not measured:
        raise RankRefused(
            "no MEASURED_SELF_DIRTY timings; refusing to rank arbitrarily"
        )
    measured.sort(key=lambda r: (-float(r["median_ms"]), str(r.get("cost") or "")))
    return [{**row, "rank": i} for i, row in enumerate(measured, start=1)]


# ---------------------------------------------------------------------------
# Source recovery. Sparse-missing is not project-absent.
# ---------------------------------------------------------------------------


def _hcli_source(rel: str) -> tuple[str, str]:
    disk = REPO / rel
    if disk.is_file():
        return disk.read_text(encoding="utf-8", errors="replace"), f"disk:{disk}"
    blob = git("show", f"HEAD:{rel}")
    if blob:
        return blob, f"git:HEAD:{rel}"
    try:
        import hcli as pkg
    except Exception as exc:
        raise SourceUnavailable(
            f"{rel} unseen on disk and HEAD:{rel} empty; hcli import failed "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    live = Path(pkg.__file__).resolve().parent / Path(rel).relative_to("hcli")
    if live.is_file():
        return live.read_text(encoding="utf-8", errors="replace"), f"editable:{live}"
    raise SourceUnavailable(
        f"{rel} unseen on disk, HEAD:{rel} empty, and editable install "
        f"has no {live}"
    )


def _list_argv_containing(fn: ast.AST, needle: str) -> list[str] | None:
    for sub in ast.walk(fn):
        if not isinstance(sub, ast.List):
            continue
        consts = [
            elt.value
            for elt in sub.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        ]
        if needle in consts:
            out: list[str] = []
            for elt in sub.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    out.append(elt.value)
                else:
                    out.append("<dyn>")
            return out
    return None


def recover_hcli_git_status() -> dict[str, Any]:
    """What HCLI actually invokes. Absence of the flag is the finding."""
    text, how = _hcli_source(HCLI_REL["tool_registry"])
    tree = ast.parse(text)
    argv: list[str] | None = None
    timeout_default: float | None = None
    has_flag = "--no-optional-locks" in text
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_run_readonly":
            pairs = list(zip(node.args.args, [None] * (len(node.args.args) - len(node.args.defaults)) + list(node.args.defaults)))
            pairs.extend(zip(node.args.kwonlyargs, node.args.kw_defaults))
            for arg, default in pairs:
                if arg.arg != "timeout" or default is None:
                    continue
                if isinstance(default, ast.Constant):
                    try:
                        timeout_default = float(default.value)
                    except (TypeError, ValueError):
                        timeout_default = None
        if isinstance(node, ast.FunctionDef) and node.name == "_git_status":
            argv = _list_argv_containing(node, "status")
    if argv is None:
        raise SourceUnavailable(
            "hcli/tool_registry.py parsed but _git_status argv was not recovered"
        )
    # Recording the lock-taking argv is the finding. Running it is the incident.
    return {
        "path": HCLI_REL["tool_registry"],
        "recovered_from": how,
        "function": "_git_status",
        "argv": argv,
        "timeout_s_default": timeout_default,
        "carries_no_optional_locks": "--no-optional-locks" in argv,
        "file_mentions_no_optional_locks": has_flag,
        "finding": (
            "HCLI git.status is git status --short --branch with a 30s "
            "timeout and without --no-optional-locks. Sidecar _common.git "
            "already passes the flag. The minutes-scale cost is HCLI's path, "
            "not the sidecar's."
            if "--no-optional-locks" not in argv
            else "HCLI git.status already carries --no-optional-locks; the "
            "worked example would then be historical, not live."
        ),
    }


def recover_sidecar_git() -> dict[str, Any]:
    from tools.future import _common as common

    src = inspect.getsource(common.git)
    flag = "--no-optional-locks" in src
    timeout = "--no-optional-locks" in src and "timeout" in src
    return {
        "path": "tools/future/_common.py",
        "function": "git",
        "carries_no_optional_locks": flag,
        "timeout_s": GIT_TIMEOUT_S,
        "timeout_present_in_source": timeout,
        "historical_claim": (
            "Seconds before a read-only git query is abandoned. The tree is "
            "~43GB and dirty, so git status can run for minutes; --no-optional-locks "
            "tells git not to take index.lock for a query that does not need it."
        ),
        "historical_claim_source": "tools/future/_common.py module comment on git()",
        "not_remeasured_without_flag": True,
        "why_not_remeasured": (
            "the no-flag arm takes the index lock and has stranded it; this "
            "profiler refuses that arm"
        ),
    }


def recover_scheduler_nonblocking() -> dict[str, Any]:
    text, how = _hcli_source(HCLI_REL["scheduler"])
    marker = "Return assignments without waiting"
    return {
        "path": HCLI_REL["scheduler"],
        "recovered_from": how,
        "dispatch_returns_without_waiting": marker in text,
        "persist_on_submit_and_dispatch": "def _persist(" in text,
        "note": (
            "queue_wait and process_idle are calendar gaps, not a function "
            "that blocks inside dispatch(). A duration for them needs a live "
            "ready_at/running_at ledger this sidecar does not have."
        ),
    }


# ---------------------------------------------------------------------------
# Contamination context. Numbers without it look cleaner than they are.
# ---------------------------------------------------------------------------


def contamination_context() -> dict[str, Any]:
    """What else is running. Probe failure is UNKNOWN, never QUIESCENT-by-default."""
    out: dict[str, Any] = {
        "status": "UNKNOWN",
        "reason": "contamination snapshot not yet taken",
        "live_campaign_declared": True,
        "live_campaign_note": (
            "a live Codex Accelerator campaign is running in this repo; "
            "SELF_MEASURED_DIRTY timings are contaminated by construction"
        ),
        "load_1m": None,
        "ncpu": os.cpu_count(),
        "competing_workloads": [],
        "contamination_class": "UNKNOWN",
    }
    try:
        load = os.getloadavg()
        out["load_1m"] = float(load[0])
        out["load_5m"] = float(load[1])
        out["load_15m"] = float(load[2])
    except OSError as exc:
        out["reason"] = f"getloadavg failed: {exc}"
        return out
    try:
        from tools.future import contamination as C
    except Exception as exc:
        out["reason"] = f"contamination import failed: {type(exc).__name__}: {exc}"
        out["status"] = "PARTIAL"
        return out
    try:
        snap = C.snapshot()
        klass = C.classify_contamination(snap)
    except Exception as exc:
        out["reason"] = f"contamination.snapshot failed: {type(exc).__name__}: {exc}"
        out["status"] = "PARTIAL"
        return out
    competing = snap.get("competing_workloads") or []
    names = []
    if isinstance(competing, list):
        for row in competing[:12]:
            if isinstance(row, Mapping):
                names.append(
                    {
                        "name": row.get("name"),
                        "pid": row.get("pid"),
                        "cpu_pct": row.get("cpu_pct"),
                        "rss_gib": row.get("rss_gib"),
                    }
                )
    out.update(
        {
            "status": "OK",
            "reason": None,
            "contamination_class": klass.get("contamination_class"),
            "contamination_reason": klass.get("contamination_reason"),
            "competing_workloads": names,
            "thermal_state": snap.get("thermal_state"),
            "memory_pressure_name": (
                (snap.get("memory_pressure") or {}).get("pressure_name")
                if isinstance(snap.get("memory_pressure"), Mapping)
                else None
            ),
            "did_not_take_gpu_lease": True,
        }
    )
    return out


# ---------------------------------------------------------------------------
# HCLI import. Missing is recorded; analogs that need it go UNKNOWN.
# ---------------------------------------------------------------------------


def _hcli_modules() -> tuple[dict[str, Any] | None, str]:
    try:
        from hcli import tool_registry as tr
        from hcli import verifier_pipeline as vp
        from hcli import workunit as wu
        from hcli.resources import ResourceLimits
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return (
        {
            "workunit": wu,
            "tool_registry": tr,
            "verifier_pipeline": vp,
            "ResourceLimits": ResourceLimits,
        },
        "import:hcli (editable or path; read-only)",
    )


def _summarize(samples: list[float]) -> dict[str, Any]:
    xs = sorted(float(s) for s in samples)
    med = statistics.median(xs)
    min_ms = round(xs[0], 3)
    max_ms = round(xs[-1], 3)
    out = {
        "n": len(xs),
        "median_ms": round(med, 3),
        "min_ms": min_ms,
        "max_ms": max_ms,
        "range_ms": round(max_ms - min_ms, 3),
        "samples_ms": [round(s, 3) for s in samples],
    }
    for key in out:
        assert_timing_field_legal(key)
    return out


def _unknown(cost: str, reason: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "cost": cost,
        "state": "UNKNOWN",
        "reason": reason,
        "self_timing": None,
        "not_a_hardware_measurement": True,
    }
    if extra:
        rec.update(dict(extra))
    return rec


def _measured(cost: str, summary: dict[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "cost": cost,
        "state": "MEASURED_SELF_DIRTY",
        "reason": None,
        "self_timing": {
            "evidence_class": "SELF_MEASURED_DIRTY",
            "not_protected_absolute": True,
            "not_a_hardware_measurement": True,
            "use": "rank and prune; decide nothing",
            **summary,
        },
    }
    if extra:
        rec.update(dict(extra))
    return rec


def _repeat(fn: Callable[[], Any], repeats: int) -> tuple[list[float], Any]:
    samples: list[float] = []
    last: Any = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples, last


# ---------------------------------------------------------------------------
# Analogs. Each times a real function or a declared analog of one.
# ---------------------------------------------------------------------------


def _time_git_query() -> dict[str, Any]:
    """Sidecar path only. The HCLI path is the lock-taking arm and is refused."""
    argv = ["git", "--no-optional-locks", "status", "--porcelain"]
    refuse_bare_git_status(argv)

    def once() -> dict[str, Any]:
        try:
            proc = subprocess.run(
                argv,
                cwd=str(REPO),
                capture_output=True,
                text=True,
                check=False,
                timeout=GIT_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return {"timed_out": True, "n_lines": None, "returncode": None}
        except OSError as exc:
            return {"timed_out": False, "error": str(exc), "n_lines": None, "returncode": None}
        return {
            "timed_out": False,
            "n_lines": len((proc.stdout or "").splitlines()),
            "returncode": proc.returncode,
        }

    return once()


def _time_subprocess_launch() -> None:
    proc = subprocess.run(
        [_sys.executable, "-c", "pass"],
        capture_output=True,
        check=False,
        timeout=15,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"python -c pass failed rc={proc.returncode}")


def _time_source_receipt_io() -> dict[str, Any]:
    n_read = 0
    bytes_read = 0
    if RECEIPTS.is_dir():
        for path in sorted(RECEIPTS.glob("*.json"))[:8]:
            data = path.read_bytes()
            n_read += 1
            bytes_read += len(data)
    payload = json.dumps(
        {"schema": "hawking.future.hcli_self_profile.probe.v1", "n": 200, "ids": list(range(200))},
        separators=(",", ":"),
        sort_keys=True,
    )
    fd, path = tempfile.mkstemp(prefix="hcli-self-profile-io-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
    finally:
        os.unlink(path)
    return {"receipts_read": n_read, "bytes_read": bytes_read}


def _time_context_construction() -> dict[str, Any]:
    # Analog of Engine._gather_evidence: read named sources into a blob.
    rels = (
        "tools/future/_common.py",
        "tools/future/turnaround.py",
        "tools/future/autonomy_run.py",
        "tools/future/orchestration.py",
        "tools/future/frontiers.py",
        "tools/future/git_lock_doctor.py",
    )
    total = 0
    seen = 0
    for rel in rels:
        path = REPO / rel
        if not path.is_file():
            continue
        total += len(path.read_text(encoding="utf-8", errors="replace"))
        seen += 1
    return {"files_read": seen, "chars": total, "asked": len(rels)}


def _time_scheduler_decision(mods: dict[str, Any]) -> dict[str, Any]:
    wu_mod = mods["workunit"]
    Limits = mods["ResourceLimits"]
    WorkUnit = wu_mod.WorkUnit
    units = {}
    for i in range(400):
        units[f"u{i}"] = WorkUnit(
            id=f"u{i}",
            role="science",
            description="self-profile analog",
            status="pending",
            resource_class="LIGHT_CONTROL" if i % 10 else "GPU_EXCLUSIVE",
            dependencies=[] if i % 7 else [f"u{i-1}"] if i else [],
        )
    limits = Limits(
        gpu_decode=0,
        gpu_decode_source="self-profile analog; not a live decode cap",
        gpu_exclusive=0,
        mutation=0,
        compile=0,
        static_analysis=32,
        light_control=32,
        cpu_heavy=8,
    )
    ready = wu_mod.identify_ready(units)
    assigned = wu_mod.assign_ready(
        ready, runtime_count=4, all_units=units, limits=limits
    )
    n_gpu_ready = sum(
        1 for u in ready if getattr(u, "resource_class", "") == "GPU_EXCLUSIVE"
    )
    return {
        "n_units": len(units),
        "n_ready": len(ready),
        "n_assigned": len(assigned),
        "n_gpu_exclusive_ready": n_gpu_ready,
        "n_unassigned": len(ready) - len(assigned),
        "analog": "hcli.workunit.identify_ready + assign_ready",
    }


def _time_tool_routing(mods: dict[str, Any]) -> dict[str, Any]:
    tr = mods["tool_registry"]
    # Repo root as a read-only context: dummy handlers only, git.status never invoked.
    ctx = tr.ToolContext(REPO, REPO, None, frozenset({tr.READ_ONLY}))
    registry = tr.ToolRegistry(ctx)
    schema = {"type": "object", "additionalProperties": True}

    def handler(_c: Any, _a: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    for i in range(40):
        registry.register(
            tr.ToolSpec(f"probe.{i}", "self-profile analog", schema, handler=handler)
        )
    hits = 0
    misses: list[str] = []
    for i in range(40):
        spec = registry.get(f"probe.{i}")
        if spec is None:
            misses.append(f"probe.{i}")
            continue
        err = tr.validate_input({}, spec.input_schema)
        if err is None:
            spec.handler(ctx, {})
            hits += 1
    unknown = registry.get("git.status")
    return {
        "n_registered": 40,
        "n_routed": hits,
        "n_misses": len(misses),
        "git.status_registered": unknown is not None,
        "analog": "ToolRegistry.get + validate_input + handler; git.status not invoked",
    }


def _time_verifier(mods: dict[str, Any]) -> dict[str, Any]:
    vp = mods["verifier_pipeline"]
    commands = (
        "true",
        ":",
        "exit 0",
        "python3 -c 'raise SystemExit(0)'",
        "pytest tools/future/test_common.py",
        "python3 tools/future/_common.py",
        "sh -c true",
        "git --no-optional-locks status --porcelain",
        "cargo test",
        "",
    )
    rejected = 0
    admitted = 0
    reasons: list[str] = []
    for cmd in commands:
        ok, why = vp.command_is_admissible(cmd)
        if ok:
            admitted += 1
        else:
            rejected += 1
            reasons.append(f"{cmd!r}:{why}")
    if rejected == 0:
        raise RuntimeError(
            "verifier admitted every probe including vacuous true/exit 0; "
            "the negative control did not fire"
        )
    return {
        "n": len(commands),
        "n_admitted": admitted,
        "n_rejected": rejected,
        "rejected_reasons": reasons,
        "analog": "hcli.verifier_pipeline.command_is_admissible",
    }


# ---------------------------------------------------------------------------
# Hypotheses. Status is HYPOTHESIS until a cause is tested.
# ---------------------------------------------------------------------------


def hypothesis_catalog() -> dict[str, dict[str, Any]]:
    """One hypothesis per cost bucket. as_actionable() is the gate."""
    return {
        "git_query": {
            "cost": "git_query",
            "worked_example": True,
            "why_this_cost_exists": (
                "git status refreshes the index and therefore WRITES "
                ".git/index.lock. On this ~43GB dirty tree that refresh took "
                "minutes. hcli.tool_registry._git_status invokes "
                "`git status --short --branch` via _run_readonly (timeout=30) "
                "WITHOUT --no-optional-locks. A 30s timeout SIGKILLs git while "
                "it holds the lock. tools/future/_common.git already passes the "
                "flag; HCLI does not."
            ),
            "removal": {
                "kind": "wait",
                "what_would_remove_it": (
                    "Pass --no-optional-locks on the HCLI git.status argv "
                    "(the sidecar path already does). That skips the index "
                    "refresh. The lock doctor is durability after a kill, not "
                    "the removal of the wait."
                ),
            },
            "cheapest_falsifier": (
                "Time `git --no-optional-locks status --porcelain` (this "
                "module does) and inspect GIT_TRACE2 for an index refresh. If "
                "the no-lock path still refreshes or still takes minutes, the "
                "flag is not the removal. Do not re-run bare `git status`: "
                "that recreates the lock incident."
            ),
        },
        "scheduler_decision": {
            "cost": "scheduler_decision",
            "why_this_cost_exists": (
                "Every dispatch scans the unit map (identify_ready) and admits "
                "under per-class caps (assign_ready). Scheduler.__init__ / "
                "submit / dispatch also _persist the DAG when a store is set."
            ),
            "removal": {
                "kind": "information",
                "what_would_remove_it": (
                    "Keep a ready-set incrementally instead of scanning all "
                    "units every dispatch, and debounce _persist to a dirty "
                    "flag. The scan is cheap next to git.status; persist is "
                    "the scheduler cost that actually copies."
                ),
            },
            "cheapest_falsifier": (
                "Count identify_ready median_ms against source_receipt_io "
                "median_ms on a 10k-unit DAG. If identify_ready dominates "
                "I/O, the incremental-set hypothesis is live; if I/O "
                "dominates, debounce persist instead."
            ),
        },
        "queue_wait": {
            "cost": "queue_wait",
            "why_this_cost_exists": (
                "A ready unit waits when its resource class is at cap "
                "(GPU_EXCLUSIVE=1, MUTATION=1, decode limit). dispatch() "
                "returns without waiting; the wait is calendar time until a "
                "later dispatch sees a free slot."
            ),
            "removal": {
                "kind": "wait",
                "what_would_remove_it": (
                    "Do not put CPU-class work on GPU_EXCLUSIVE. Raise the "
                    "cap only from measured occupancy, never from a wish. "
                    "Park blocked GPU work SLEEPING (frontiers.py already "
                    "does) so it does not occupy a ready slot."
                ),
            },
            "cheapest_falsifier": (
                "A live DAG ledger of ready_at minus running_at, split by "
                "resource_class. If LIGHT_CONTROL gaps are large while "
                "occupancy is below cap, the cap is not the wait."
            ),
        },
        "subprocess_launch": {
            "cost": "subprocess_launch",
            "why_this_cost_exists": (
                "Engine._run_contained_subprocess starts a new session for "
                "every contained pytest/script. Tool _run_readonly does the "
                "same for git and shell. Process spawn is paid per call, not "
                "amortized."
            ),
            "removal": {
                "kind": "call",
                "what_would_remove_it": (
                    "In-process pytest runner for tiny files already imported "
                    "in this interpreter, and reuse one git child for batched "
                    "read-only queries. Spawn stays for isolation of untrusted "
                    "commands."
                ),
            },
            "cheapest_falsifier": (
                "Compare python3 -c pass (this analog) to in-process exec of "
                "an empty function. If spawn is not in the ranked head, leave "
                "it: isolation is the point."
            ),
        },
        "source_receipt_io": {
            "cost": "source_receipt_io",
            "why_this_cost_exists": (
                "Scheduler._persist writes the DAG on submit/replan/dispatch. "
                "Ledger.save atomically rewrites markdown. Engine evidence "
                "gather re-reads files named in the prompt every turn."
            ),
            "removal": {
                "kind": "re-read",
                "what_would_remove_it": (
                    "Content-address the evidence blob (mtime+size or sha) "
                    "and skip a persist when the DAG hash is unchanged. "
                    "Receipts are already sealed; the cache key is the seal."
                ),
            },
            "cheapest_falsifier": (
                "Two consecutive dispatch() calls with no unit-state change: "
                "if _persist still writes bytes, debounce is live. If the "
                "second write is zero bytes, the cost is already gone."
            ),
        },
        "context_construction": {
            "cost": "context_construction",
            "why_this_cost_exists": (
                "Engine._gather_evidence walks path tokens in the prompt, "
                "reads each file under a char budget, and may expand nested "
                "references from instruction documents. Context is rebuilt "
                "per complete_text, not reused across turns."
            ),
            "removal": {
                "kind": "copy",
                "what_would_remove_it": (
                    "Treat sealed receipts as the authority and put only "
                    "paths + seals in context, fetching bodies on miss. "
                    "FT.CONTEXT.disk-authority already names this bug: a "
                    "context blob must not compete with a receipt."
                ),
            },
            "cheapest_falsifier": (
                "Two consecutive executes of the same prompt: if the second "
                "still re-reads every evidence file, the cache is missing. "
                "If file reads drop to zero, the copy is already gone."
            ),
        },
        "tool_routing": {
            "cost": "tool_routing",
            "why_this_cost_exists": (
                "ToolRegistry.invoke looks up a name, validates input, checks "
                "permissions, calls the handler, validates output. The dict "
                "lookup is cheap; the handler (often git.status or a network "
                "fetch) is the cost mis-attributed to routing."
            ),
            "removal": {
                "kind": "call",
                "what_would_remove_it": (
                    "Stop calling git.status on every turn. Route status "
                    "through the sidecar git() (already --no-optional-locks) "
                    "or skip untracked. Routing itself does not need a new "
                    "dispatcher."
                ),
            },
            "cheapest_falsifier": (
                "Time ToolRegistry.get+validate (this analog) against "
                "git_query. If routing median_ms is not in the ranked head, "
                "do not rewrite the dispatcher."
            ),
        },
        "verifier_overhead": {
            "cost": "verifier_overhead",
            "why_this_cost_exists": (
                "command_is_admissible tokenizes, splits on shell combinators, "
                "and AST-parses python -c bodies so `true` / `exit 0` / "
                "`cmd || true` cannot launder a pass. evaluate_python_test_file "
                "may spawn pytest."
            ),
            "removal": {
                "kind": "call",
                "what_would_remove_it": (
                    "Cache admissibility by command hash. Do not weaken the "
                    "vacuous-command refuse: a verifier nobody has watched "
                    "reject is a verifier that will silently drift."
                ),
            },
            "cheapest_falsifier": (
                "This module's negative control: `true` and `exit 0` must "
                "come back rejected. If they are admitted, the overhead is "
                "fiction and the verifier is the bug. If rejected and "
                "median_ms is in the tail, do not rewrite it."
            ),
        },
        "compile_wait": {
            "cost": "compile_wait",
            "why_this_cost_exists": (
                "cargo build against the shared target-dir "
                "workspace/ops/build/rust contends with the live campaign. "
                "tests_run may select runner=cargo. This sidecar must not "
                "join that queue."
            ),
            "removal": {
                "kind": "wait",
                "what_would_remove_it": (
                    "Per-experiment CARGO_TARGET_DIR plus a content-addressed "
                    "skip fingerprint (turnaround.py lever "
                    "target_isolation_plus_input_fingerprint). Do not "
                    "substitute release-fast numbers for protected ones."
                ),
            },
            "cheapest_falsifier": (
                "turnaround.py already refuses to time compile as a CPU "
                "proxy. A protected window that records compile_ns on an "
                "isolated target-dir either drops the wait (fingerprint hit) "
                "or does not (cache miss / lock contention remains)."
            ),
        },
        "process_idle": {
            "cost": "process_idle",
            "why_this_cost_exists": (
                "dispatch() returns without waiting. Campaign-level idle is "
                "the gap between missions, not a blocking call. "
                "autonomy_run.py treats an idle / awaiting-instructions event "
                "as an automatic trial failure and has no path that emits one."
            ),
            "removal": {
                "kind": "wait",
                "what_would_remove_it": (
                    "Refill the frontier (frontiers.next_work / refill) "
                    "instead of sitting in an empty dispatch loop. Park "
                    "hardware work SLEEPING and move to CPU work."
                ),
            },
            "cheapest_falsifier": (
                "A live session timeline containing an idle or "
                "awaiting-instructions event would refute 'HCLI does not "
                "idle-wait'. Scheduler source says dispatch returns without "
                "waiting; this sidecar has no live timeline, so the duration "
                "stays UNKNOWN."
            ),
        },
    }


def _attach_hypothesis(cost_rec: dict[str, Any], catalog: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    hypo = catalog.get(str(cost_rec.get("cost")))
    if hypo is None:
        cost_rec["actionable"] = False
        cost_rec["actionable_reason"] = "no hypothesis catalogued for this cost"
        return cost_rec
    try:
        action = as_actionable(hypo)
    except ActionableRefused as exc:
        cost_rec["actionable"] = False
        cost_rec["actionable_reason"] = str(exc)
        return cost_rec
    cost_rec["actionable"] = True
    cost_rec["hypothesis"] = action
    return cost_rec


# ---------------------------------------------------------------------------
# Profile.
# ---------------------------------------------------------------------------


def profile(*, repeats: int = DEFAULT_REPEATS) -> dict[str, Any]:
    if repeats < MIN_REPEATS:
        raise ValueError(
            f"repeats must be >= {MIN_REPEATS} (median with spread, never a single sample)"
        )

    catalog = hypothesis_catalog()
    for hypo in catalog.values():
        as_actionable(hypo)

    hcli_git = recover_hcli_git_status()
    sidecar_git = recover_sidecar_git()
    sched_shape = recover_scheduler_nonblocking()
    contamination = contamination_context()
    mods, hcli_how = _hcli_modules()

    attributed: list[dict[str, Any]] = []

    # git_query (sidecar path)
    git_samples: list[float] = []
    git_last: dict[str, Any] | None = None
    git_failed: str | None = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        git_last = _time_git_query()
        dt = (time.perf_counter() - t0) * 1000.0
        if git_last.get("timed_out"):
            git_failed = f"git --no-optional-locks status timed out after {GIT_TIMEOUT_S}s"
            break
        if git_last.get("error"):
            git_failed = f"git query OSError: {git_last['error']}"
            break
        git_samples.append(dt)
    if git_failed or len(git_samples) < MIN_REPEATS:
        attributed.append(
            _unknown(
                "git_query",
                git_failed or "git_query produced fewer than MIN_REPEATS samples",
                extra={"hcli_path": hcli_git, "sidecar_path": sidecar_git},
            )
        )
    else:
        attributed.append(
            _measured(
                "git_query",
                _summarize(git_samples),
                extra={
                    "measurement_kind": "git_no_optional_locks_status_porcelain",
                    "argv": ["git", "--no-optional-locks", "status", "--porcelain"],
                    "last": git_last,
                    "hcli_path": hcli_git,
                    "sidecar_path": sidecar_git,
                    "is_hcli_path": False,
                    "note": (
                        "this times the sidecar path (flag present). HCLI's "
                        "path is unmeasured because it is the lock-taking arm."
                    ),
                },
            )
        )

    # subprocess_launch
    try:
        samples, _ = _repeat(_time_subprocess_launch, repeats)
        attributed.append(
            _measured(
                "subprocess_launch",
                _summarize(samples),
                extra={"measurement_kind": "python3_-c_pass", "analog_of": "Engine._run_contained_subprocess"},
            )
        )
    except Exception as exc:
        attributed.append(_unknown("subprocess_launch", f"{type(exc).__name__}: {exc}"))

    # source_receipt_io
    try:
        samples, last = _repeat(_time_source_receipt_io, repeats)
        attributed.append(
            _measured(
                "source_receipt_io",
                _summarize(samples),
                extra={
                    "measurement_kind": "receipt_read_plus_tempfile_fsync",
                    "analog_of": "Scheduler._persist / Ledger.save",
                    "last": last,
                },
            )
        )
    except Exception as exc:
        attributed.append(_unknown("source_receipt_io", f"{type(exc).__name__}: {exc}"))

    # context_construction
    try:
        samples, last = _repeat(_time_context_construction, repeats)
        attributed.append(
            _measured(
                "context_construction",
                _summarize(samples),
                extra={
                    "measurement_kind": "read_named_sidecar_sources",
                    "analog_of": "Engine._gather_evidence",
                    "last": last,
                },
            )
        )
    except Exception as exc:
        attributed.append(_unknown("context_construction", f"{type(exc).__name__}: {exc}"))

    if mods is None:
        for cost in ("scheduler_decision", "tool_routing", "verifier_overhead"):
            attributed.append(
                _unknown(cost, f"hcli not importable: {hcli_how}")
            )
        structural_unassigned = None
        verifier_rejected = None
    else:
        try:
            samples, last = _repeat(lambda: _time_scheduler_decision(mods), repeats)
            attributed.append(
                _measured(
                    "scheduler_decision",
                    _summarize(samples),
                    extra={
                        "measurement_kind": "identify_ready_plus_assign_ready",
                        "hcli_import": hcli_how,
                        "last": last,
                    },
                )
            )
            structural_unassigned = (last or {}).get("n_unassigned")
        except Exception as extra_exc:
            attributed.append(
                _unknown("scheduler_decision", f"{type(extra_exc).__name__}: {extra_exc}")
            )
            structural_unassigned = None
        try:
            samples, last = _repeat(lambda: _time_tool_routing(mods), repeats)
            attributed.append(
                _measured(
                    "tool_routing",
                    _summarize(samples),
                    extra={
                        "measurement_kind": "ToolRegistry_get_validate_handler",
                        "hcli_import": hcli_how,
                        "last": last,
                    },
                )
            )
        except Exception as extra_exc:
            attributed.append(
                _unknown("tool_routing", f"{type(extra_exc).__name__}: {extra_exc}")
            )
        try:
            samples, last = _repeat(lambda: _time_verifier(mods), repeats)
            attributed.append(
                _measured(
                    "verifier_overhead",
                    _summarize(samples),
                    extra={
                        "measurement_kind": "command_is_admissible_corpus",
                        "hcli_import": hcli_how,
                        "last": last,
                    },
                )
            )
            verifier_rejected = (last or {}).get("n_rejected")
        except Exception as extra_exc:
            attributed.append(
                _unknown("verifier_overhead", f"{type(extra_exc).__name__}: {extra_exc}")
            )
            verifier_rejected = None

    attributed.append(
        _unknown(
            "compile_wait",
            "cargo build is forbidden on the shared target-dir; a CPU proxy "
            "is not compile wait (turnaround.py already refuses this)",
            extra={
                "cargo_forbidden": True,
                "shared_target_dir": "workspace/ops/build/rust",
                "lever_owner": "tools/future/turnaround.py",
                "lever_id": "target_isolation_plus_input_fingerprint",
            },
        )
    )
    attributed.append(
        _unknown(
            "queue_wait",
            "dispatch() returns without waiting; queue wait is calendar time "
            "a ready unit spends blocked on occupancy. No live ready_at/"
            "running_at ledger in this sidecar.",
            extra={
                "scheduler_nonblocking": sched_shape,
                "structural_unassigned_in_analog": structural_unassigned,
                "structural_note": (
                    "the scheduler analog set gpu_exclusive=0 so GPU_EXCLUSIVE "
                    "units remain unassigned. That is evidence the wait exists "
                    "as a structure, not a duration."
                ),
            },
        )
    )
    attributed.append(
        _unknown(
            "process_idle",
            "dispatch returns without waiting; autonomy_run has no idle-event "
            "path. Campaign idle is a session gap. No live timeline here.",
            extra={"scheduler_nonblocking": sched_shape},
        )
    )

    by_name = {rec["cost"]: rec for rec in attributed}
    ordered = [_attach_hypothesis(dict(by_name[name]), catalog) for name in COST_BUCKETS]

    try:
        ranked = rank_attributed_costs(ordered)
        rank_state = "RANKED_SELF_DIRTY"
        rank_reason = None
    except RankRefused as exc:
        ranked = []
        rank_state = "REFUSED"
        rank_reason = str(exc)

    actionable = []
    for rec in ordered:
        hypo = rec.get("hypothesis")
        if rec.get("actionable") and isinstance(hypo, Mapping):
            row = dict(hypo)
            row["cost_state"] = rec.get("state")
            if rec.get("state") == "MEASURED_SELF_DIRTY":
                timing = rec.get("self_timing") or {}
                row["median_ms"] = timing.get("median_ms")
            actionable.append(row)
    if ranked:
        order = {row["cost"]: row["rank"] for row in ranked}
        actionable.sort(
            key=lambda r: (
                order.get(r["cost"], 10_000),
                0 if r.get("worked_example") else 1,
                str(r["cost"]),
            )
        )

    worked = recover_worked_example(hcli_git, sidecar_git, by_name.get("git_query"))

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Treat HCLI as part of Hawking's search space: attribute process "
            "wall across the resident's own machinery and rank actionable "
            "removal hypotheses. SELF_MEASURED_DIRTY numbers rank and prune "
            "and decide nothing."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "self_timing": {
            "evidence_class": "SELF_MEASURED_DIRTY",
            "not_protected_absolute": True,
            "not_a_hardware_measurement": True,
            "use": "rank and prune; decide nothing",
            "numbers_decide_nothing": True,
            "contamination": contamination,
            "repeats": repeats,
            "field_names": (
                "median_ms/min_ms/max_ms/samples_ms under this block; never "
                + ", ".join(sorted(HARDWARE_FIELDS))
            ),
        },
        "attributed_costs": ordered,
        "rank_state": rank_state,
        "rank_reason": rank_reason,
        "ranked": ranked,
        "actionable_hypotheses": actionable,
        "worked_example": worked,
        "hcli_import": hcli_how,
        "verifier_negative_fired": (
            None if verifier_rejected is None else bool(verifier_rejected)
        ),
        "recovered_implementation": [
            "tools/future/_common.py git() --no-optional-locks + GIT_TIMEOUT_S",
            "tools/future/git_lock_doctor.py (HCLI git.status lock-taking inventory)",
            "tools/future/turnaround.py (CPU-side phase timings; compile stays UNKNOWN)",
            "tools/future/dirty_measure.py (SELF_MEASURED_DIRTY; rank refuses without data)",
            "tools/future/contamination.py (machine-state snapshot, not a benchmark)",
            "tools/future/autonomy_run.py (idle event is a trial failure)",
            "tools/future/orchestration.py BINDINGS / FT.HCLI_SELF.emit-workunits",
            "tools/future/frontiers.py HCLI_SELF items",
            "hcli/tool_registry.py _git_status / _run_readonly",
            "hcli/scheduler.py dispatch() non-blocking + _persist",
            "hcli/workunit.py identify_ready / assign_ready",
            "hcli/verifier_pipeline.py command_is_admissible",
            "hcli/engine.py _gather_evidence / _run_contained_subprocess",
        ],
        "gaps_closed": [
            "No tools/future/hcli_self_profile.py existed; HCLI was not a ranked search space",
            "Wall-clock attribution across the ten named buckets, with UNKNOWN where a duration would be fiction",
            "Worked example recovered from _common.git vs HCLI git.status argv, without re-running the lock-taking arm",
            "Actionable gate: a cost with no removal hypothesis cannot be reported as actionable",
            "Rank gate: no MEASURED_SELF_DIRTY timings refuses to rank",
            "Hardware-named fields refused even as python timings",
        ],
        "negative_findings": [
            "Did not invoke hcli git.status or run git status without --no-optional-locks",
            "Did not run cargo build / cargo test or time compile_wait",
            "Did not take a GPU lease or flock a bench lock",
            "Did not start a resident model process",
            "queue_wait and process_idle stay UNKNOWN: no live ready_at/running_at ledger",
            "Did not prove index-refresh is the cause of the historical minutes (before-arm refused)",
            "orchestration.BINDINGS was not updated (not in this lane's write set); naming FT.HCLI_SELF.emit-workunits is what the receipt informs, not proof invoke() routed it",
            "hcli/ is sparse-missing in this checkout; recovered via git show HEAD:hcli/... and the editable install",
            "Did not write hcli/, crates/, tools/accelerator/, or any Codex surface",
        ],
        "resident_callable": {
            "entry_point": "tools.future.hcli_self_profile.profile()",
            "workunit": (
                "one ANALYSIS unit; profile HCLI self costs as SELF_MEASURED_DIRTY; "
                "rank removal hypotheses; never promote"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.HCLI_SELF.emit-workunits",
            "fails_closed": (
                "bare git status and hcli git.status raise LiveGitStatusForbidden; "
                "compile_wait as measured raises CompileWaitForbidden; "
                "rank with no timings raises RankRefused; "
                "actionable without why/removal/falsifier raises ActionableRefused; "
                "hardware field names raise HardwareClaimError"
            ),
            "can_hcli_invoke": True,
            "orchestration_bound": False,
            "orchestration_bound_reason": (
                "this lane cannot write tools/future/orchestration.py; a later "
                "glue lane must add the binding before invoke() can route it"
            ),
        },
    }


def recover_worked_example(
    hcli_git: Mapping[str, Any],
    sidecar_git: Mapping[str, Any],
    git_cost: Mapping[str, Any] | None,
) -> dict[str, Any]:
    timing = (git_cost or {}).get("self_timing") if git_cost else None
    after_ms = timing.get("median_ms") if isinstance(timing, Mapping) else None
    hcli_open = not bool(hcli_git.get("carries_no_optional_locks"))
    sidecar_landed = bool(sidecar_git.get("carries_no_optional_locks"))
    return {
        "id": "H.GIT.no-optional-locks",
        "cost": "git_query",
        "status": "HYPOTHESIS",
        "causal_claim_verified": False,
        "sidecar_path": "LANDED" if sidecar_landed else "OPEN",
        "hcli_path": "OPEN" if hcli_open else "LANDED",
        "historical_before": {
            "claim": "git status on this ~43GB dirty tree took minutes",
            "source": sidecar_git.get("historical_claim_source"),
            "not_remeasured": True,
            "why_not_remeasured": sidecar_git.get("why_not_remeasured"),
            "not_a_measurement": True,
        },
        "observed_after": {
            "median_ms": after_ms,
            "path": "git --no-optional-locks status --porcelain",
            "state": (git_cost or {}).get("state"),
            "note": "~0.2s is the recovered operator figure; median_ms is this run",
        },
        "hcli_argv": hcli_git.get("argv"),
        "hcli_recovered_from": hcli_git.get("recovered_from"),
        "why_hcli_still_pays": hcli_git.get("finding"),
        "does_not_decide": True,
    }


def build(*, repeats: int = DEFAULT_REPEATS) -> Path:
    doc = profile(repeats=repeats)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest(*, repeats: int = DEFAULT_REPEATS) -> Path:
    return build(repeats=repeats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = ap.parse_args()
    out = build(repeats=args.repeats)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
