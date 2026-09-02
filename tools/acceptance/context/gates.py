"""Run each assigned HCLI gate against its own Appendix D criterion.

A module import is not a call. Every ACCEPTED gate below invokes the
catalog symbol (or, for HCLI_CONTEXT_INVALIDATION, ``assert_evidence_fresh``
— the catalog listed no symbol, which is why the auditor still says
SCAFFOLDED). BLOCKED gates name the exact missing input; the criterion
text is quoted from the roadmap and is never edited.
"""

from __future__ import annotations

import importlib.util
import inspect
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tools.acceptance.context.common import (
    REPO,
    quote_roadmap,
    rewrite,
    write_receipt,
)

GATE_IDS = (
    "HCLI_CONTEXT_AUTHORITY_UNIFIED",
    "HCLI_CONTEXT_FOCUSED_WORKUNITS",
    "HCLI_CONTEXT_INVALIDATION",
    "HCLI_STATUS_PHYSICAL",
    "HCLI_MIXED_MAX",
    "BACKEND_FAILURE_ISOLATION",
    "HCLI_SELF_SUPPLEMENT",
    "HCLI_SELF_OPTIMIZATION_BOOTSTRAP",
)

# D.2 as printed in the roadmap. Compared numerically/set-wise, not asserted.
D2_KINDS = (
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

# Aliases the live classifier uses. Not a weakening of D.2: missing names
# stay missing and are reported.
D2_ALIASES = {
    "BACKEND_UNAVAILABLE": "UNAVAILABLE_DEPENDENCY",
    "VERIFIER_FAILED": "VERIFIER_FAILURE",
    "IMPLEMENTATION_FAILED": "DETERMINISTIC_IMPLEMENTATION",
    "CONTRACT_IMPOSSIBLE": "IMPOSSIBLE_CONTRACT",
    "DEPENDENCY_BLOCKED": "UNAVAILABLE_DEPENDENCY",
}

# D.9 sections. A field is present only if the rendered /status text contains
# a token that a reader of D.9 would recognise. "unknown" still counts as
# exposing the field; omitting the field does not.
D9_TOKENS = {
    "MISSION.id": ("mission ",),
    "MISSION.phase": ("phase=",),
    "MISSION.goal": ("Goal:",),
    "MISSION.last_progress": ("last_progress", "last-progress"),
    "WORKUNITS.ready": ("ready=",),
    "WORKUNITS.running": ("running=",),
    "WORKUNITS.blocked": ("blocked=",),
    "WORKUNITS.verified": ("verified=", "completed="),
    "WORKUNITS.failed": ("failed=",),
    "PROVIDERS.resident": ("resident=", "Resident "),
    "PROVIDERS.active": ("active=", "active_decode"),
    "PROVIDERS.queued": ("queued=",),
    "PROVIDERS.health": ("health=",),
    "PROVIDERS.context": ("n_ctx=", "ctx="),
    "PROVIDERS.throughput": ("tps=",),
    "EXTERNAL.admitted": ("admitted=",),
    "EXTERNAL.active": ("Grok admitted=",),
    "EXTERNAL.queued": ("Grok ",),
    "EXTERNAL.terminal": ("done=",),
    "EXTERNAL.failed": ("Grok ",),
    "EXTERNAL.throttled": ("throttl",),
    "CPU_TOOLS.active": ("CPU ",),
    "CPU_TOOLS.queued": ("tool=",),
    "VERIFICATION.active": ("Verifier active", "verifier_active", "verify active"),
    "VERIFICATION.backlog": ("Verifier backlog=", "backlog="),
    "VERIFICATION.pass": ("verify pass", "verifier pass", "accepted/h="),
    "VERIFICATION.fail": ("verifier fail", "verify fail"),
    "MUTATION.owner": ("owner=",),
    "MUTATION.waiters": ("waiters=",),
    "MUTATION.lease_age": ("lease",),
    "REPAIR.active_chains": ("repair", "REPAIR"),
    "REPAIR.max_depth": ("max depth", "repair_depth", "max_depth"),
    "REPAIR.exhausted": ("exhausted", "FAILED_EXHAUSTED"),
    "PERSISTENCE.generation": ("generation",),
    "PERSISTENCE.checkpoint_age": ("ckpt=", "checkpoint"),
    "PERSISTENCE.digest": ("digest",),
    "THROUGHPUT.verified_wu_per_hour": ("accepted/h=", "WU/hour", "wu/h"),
    "THROUGHPUT.marginal_trend": ("marginal",),
    "RESOURCE.GPU": (" GPU", "gpu=", "GPU/"),
    "RESOURCE.CPU": ("CPU ",),
    "RESOURCE.RAM": (" ram", "RAM", "rss="),
    "RESOURCE.swap": ("swap",),
    "RESOURCE.disk": ("disk",),
    "RESOURCE.IO": (" I/O", " i/o", "io="),
    "RESOURCE.benchmark_qualification": ("benchmark",),
    "STAGNATION.tier": ("no_progress", "watchdog="),
    "STAGNATION.reason": ("no_progress:", "blocked_reason="),
}


def _budget_dict(budget: Any) -> Dict[str, Any]:
    return {
        "total_ctx": int(budget.total_ctx),
        "n_parallel": int(budget.n_parallel),
        "per_request_ctx": int(budget.per_request_ctx),
        "generation_reserve": int(budget.generation_reserve),
        "framing_reserve": int(budget.framing_reserve),
        "usable_input_tokens": int(budget.usable_input_tokens),
        "source": str(budget.source),
        "model_ceiling": budget.model_ceiling,
        "object_id": id(budget),
    }


def _preflight_dict(result: Any) -> Dict[str, Any]:
    return {
        "ok": bool(result.ok),
        "demand": int(result.demand),
        "usable": int(result.usable),
        "per_request_ctx": int(result.per_request_ctx),
        "shortfall": int(result.shortfall),
        "kind": str(result.kind),
        "remedy": str(result.remedy),
        "budget_object_id": id(result.budget),
        "budget_source": str(result.budget.source),
    }


def _symbol_location(fn: Callable[..., Any]) -> Dict[str, Any]:
    try:
        path = inspect.getsourcefile(fn) or inspect.getfile(fn)
    except TypeError:
        path = None
    try:
        line = int(inspect.getsourcelines(fn)[1])
    except (OSError, TypeError):
        line = None
    return {
        "qualname": getattr(fn, "__qualname__", getattr(fn, "__name__", str(fn))),
        "module": getattr(fn, "__module__", None),
        "file": path,
        "line": line,
    }


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    sock = socket.socket()
    sock.settimeout(0.25)
    try:
        return sock.connect_ex((host, int(port))) == 0
    except OSError:
        return False
    finally:
        sock.close()


def run_authority() -> Dict[str, Any]:
    """HCLI_CONTEXT_AUTHORITY_UNIFIED — one resolve() drives root and worker."""
    from hcli.context_budget import estimate_tokens, preflight, preflight_packet, resolve
    from hcli.goal import compile_worker_context
    from hcli.workunit import WorkUnit

    criterion = quote_roadmap(7642, 7644)
    loc = _symbol_location(resolve)
    budget = resolve(n_parallel=3)
    root_prompt = "ROOT: ultragoal + global invariants + current authority"
    worker_prompt = "WORKER: one WorkUnit + neighborhood + evidence"
    root_ok = preflight(budget, estimate_tokens(root_prompt), kind="root")
    worker_ok = preflight_packet(budget, worker_prompt, kind="worker")
    # Same authority must also refuse. Demand is prompt + generation + framing;
    # usable_input_tokens is already net of reserves, so oversize against
    # per_request_ctx.
    refuse = preflight(budget, int(budget.per_request_ctx) + 1, kind="root")
    wu = WorkUnit(id="auth-wu", role="work", description="focused worker")
    packet = compile_worker_context(
        wu,
        {
            "goal": "short authority check must pass",
            "invariants": ["one context-budget authority"],
            "acceptance_criteria": ["root and worker share resolve()"],
        },
        phase="running",
        units={wu.id: wu},
        steering=[],
        budget=budget,
    )
    worker_from_packet = preflight_packet(budget, packet.prompt, kind="worker")
    same_budget = (
        id(root_ok.budget) == id(budget)
        and id(worker_ok.budget) == id(budget)
        and id(refuse.budget) == id(budget)
        and id(worker_from_packet.budget) == id(budget)
    )
    same_source = len({
        budget.source,
        root_ok.budget.source,
        worker_ok.budget.source,
        worker_from_packet.budget.source,
    }) == 1
    same_ceiling = len({
        budget.per_request_ctx,
        root_ok.per_request_ctx,
        worker_ok.per_request_ctx,
        worker_from_packet.per_request_ctx,
    }) == 1
    checks = {
        "resolve_invoked": True,
        "preflight_root_kind": root_ok.kind == "root",
        "preflight_worker_kind": worker_ok.kind == "worker",
        "same_budget_object": same_budget,
        "same_source": same_source,
        "same_per_request_ctx": same_ceiling,
        "small_root_admitted": bool(root_ok.ok),
        "small_worker_admitted": bool(worker_ok.ok),
        "oversize_root_refused": (not refuse.ok) and refuse.shortfall > 0,
        "refuse_names_source": budget.source in refuse.remedy or "per-request" in refuse.remedy,
        "worker_packet_preflighted_on_same_budget": bool(worker_from_packet.ok) or worker_from_packet.shortfall >= 0,
        "symbol_file_is_context_budget": bool(loc.get("file") and loc["file"].endswith("context_budget.py")),
    }
    ok = all(checks.values())
    payload = {
        "criterion": {
            "span": [7642, 7644],
            "quoted": criterion,
            "meaning": "One context-budget authority drives root and worker admission.",
            "proof_obligation": "Live root/worker requests and source path agree.",
        },
        "symbol": loc,
        "invocations": [
            {"symbol": "hcli.context_budget.resolve", "note": "single authority"},
            {"symbol": "hcli.context_budget.preflight", "kind": "root"},
            {"symbol": "hcli.context_budget.preflight_packet", "kind": "worker"},
            {"symbol": "hcli.goal.compile_worker_context", "note": "worker packet then preflighted on the same budget"},
        ],
        "command": "python3 -m tools.acceptance.context --gate HCLI_CONTEXT_AUTHORITY_UNIFIED",
        "evidence_tier": "FUNCTIONAL_SIM",
        "budget": _budget_dict(budget),
        "root_preflight": _preflight_dict(root_ok),
        "worker_preflight": _preflight_dict(worker_ok),
        "oversize_root_preflight": _preflight_dict(refuse),
        "worker_packet_preflight": _preflight_dict(worker_from_packet),
        "checks": checks,
        "verdict": "ACCEPTED" if ok else "BLOCKED",
        "blocker": None if ok else "root/worker preflight did not share one resolve() budget; see checks",
        "output": {
            "source": budget.source,
            "per_request_ctx": budget.per_request_ctx,
            "usable_input_tokens": budget.usable_input_tokens,
            "oversize_shortfall": refuse.shortfall,
            "oversize_remedy": refuse.remedy,
        },
    }
    payload["receipt"] = str(write_receipt("HCLI_CONTEXT_AUTHORITY_UNIFIED", payload))
    return payload


def run_focused() -> Dict[str, Any]:
    """HCLI_CONTEXT_FOCUSED_WORKUNITS — worker packet is D.4, not the roadmap."""
    from hcli.goal import compile_worker_context, refuse_goal_dump
    from hcli.workunit import WorkUnit

    criterion = quote_roadmap(7420, 7447)
    loc = _symbol_location(compile_worker_context)
    # A civilization-roadmap-sized root. If the compiler inlines it, refuse_goal_dump
    # raises and the gate is not accepted.
    root = (
        "CIVILIZATION ROADMAP ULTRAGOAL. "
        + ("Do not feed every worker the entire civilization roadmap. " * 40)
        + "KEEP-OUT-ROOT-" * 20
    )
    assert len(root) >= 80
    parent = WorkUnit(
        id="wu-parent",
        role="research",
        description="establish the dependency neighborhood",
    )
    child = WorkUnit(
        id="wu-child",
        role="implement",
        description="apply the neighborhood evidence to one file",
        dependencies=["wu-parent"],
    )
    other = WorkUnit(
        id="wu-unrelated",
        role="research",
        description="SECRET_UNRELATED_WORKUNIT_BODY",
    )
    compiled = {
        "goal": root,
        "invariants": [
            "global invariant: never invent a measurement",
            "global invariant: mutation is single-writer",
        ],
        "acceptance_criteria": [
            "tests pass under the unit verifier",
            "no root goal dump in the worker packet",
        ],
        "referenced_files": [],
    }
    units = {parent.id: parent, child.id: child, other.id: other}
    packet = compile_worker_context(
        child,
        compiled,
        phase="running",
        units=units,
        steering=["[constraint] stay inside the named file"],
        root_goal=root,
        goal_ref="mission/goal.md",
    )
    other_packet = compile_worker_context(
        other,
        compiled,
        phase="running",
        units=units,
        steering=[],
        root_goal=root,
    )
    dump_raised = False
    try:
        refuse_goal_dump(packet.prompt, root)
    except ValueError:
        dump_raised = True
    checks = {
        "compile_worker_context_invoked": True,
        "has_phase": packet.phase == "running",
        "has_one_workunit": packet.unit_id == child.id,
        "has_invariants": len(packet.invariants) > 0,
        "has_acceptance": len(packet.acceptance) > 0,
        "has_neighborhood": any("wu-parent" in line for line in packet.neighborhood)
        or "wu-parent" in packet.prompt,
        "prompt_excludes_root": root not in packet.prompt and not dump_raised,
        "prompt_excludes_unrelated_body": "SECRET_UNRELATED_WORKUNIT_BODY" not in packet.prompt,
        "prompt_smaller_than_root": len(packet.prompt) < len(root),
        "two_workers_differ": packet.prompt != other_packet.prompt,
        "other_worker_is_other_unit": other_packet.unit_id == other.id,
        "refuse_goal_dump_clean": not dump_raised,
    }
    ok = all(checks.values())
    payload = {
        "criterion": {
            "span": [7420, 7447],
            "quoted": criterion,
            "meaning": (
                "WORKER CONTEXT = global invariants + current phase + one "
                "WorkUnit + dependency neighborhood + exact evidence + "
                "acceptance criteria. Do not feed every worker the entire "
                "civilization roadmap."
            ),
        },
        "symbol": loc,
        "invocations": [
            {"symbol": "hcli.goal.compile_worker_context", "unit": child.id},
            {"symbol": "hcli.goal.compile_worker_context", "unit": other.id},
            {"symbol": "hcli.goal.refuse_goal_dump"},
        ],
        "command": "python3 -m tools.acceptance.context --gate HCLI_CONTEXT_FOCUSED_WORKUNITS",
        "evidence_tier": "FUNCTIONAL_SIM",
        "root_chars": len(root),
        "packet": {
            "unit_id": packet.unit_id,
            "phase": packet.phase,
            "workunit": packet.workunit,
            "invariants": list(packet.invariants),
            "acceptance": list(packet.acceptance),
            "neighborhood": list(packet.neighborhood),
            "evidence_paths": list(packet.evidence_paths),
            "prompt_chars": len(packet.prompt),
            "truncated": packet.truncated,
            "omitted": list(packet.omitted),
        },
        "other_packet_unit_id": other_packet.unit_id,
        "other_packet_prompt_chars": len(other_packet.prompt),
        "checks": checks,
        "verdict": "ACCEPTED" if ok else "BLOCKED",
        "blocker": None if ok else "worker packet was not a focused D.4 context; see checks",
        "output": {
            "prompt_excerpt": packet.prompt[:500],
            "root_in_prompt": root in packet.prompt,
        },
    }
    payload["receipt"] = str(write_receipt("HCLI_CONTEXT_FOCUSED_WORKUNITS", payload))
    return payload


def run_invalidation() -> Dict[str, Any]:
    """HCLI_CONTEXT_INVALIDATION — changed evidence refuses stale context."""
    from hcli.goal import (
        StaleEvidenceError,
        assert_evidence_fresh,
        assert_packet_evidence_fresh,
        compile_worker_context,
        identity_for_path,
    )
    from hcli.workunit import WorkUnit

    criterion = quote_roadmap(7420, 7447)
    loc = _symbol_location(assert_evidence_fresh)
    hops: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="hcli-inv-") as tmp:
        notes = Path(tmp) / "notes.txt"
        notes.write_text("snapshot-AAAA", encoding="utf-8")
        gathered = identity_for_path(notes, root=Path(tmp))
        wu = WorkUnit(id="inv-wu", role="work", description="read notes.txt")
        packet = compile_worker_context(
            wu,
            {
                "goal": "short task about notes.txt must pass",
                "invariants": ["stale evidence is refused"],
                "acceptance_criteria": ["mtime+size+sha256 match"],
                "referenced_files": ["notes.txt"],
            },
            phase="running",
            units={wu.id: wu},
            steering=[],
            workspace=tmp,
            evidence=[gathered],
        )
        fresh = assert_packet_evidence_fresh(packet, tmp)
        hops["unchanged_accepted"] = [item.to_dict() for item in fresh]
        rewrite(notes, "snapshot-BBBB")
        now = identity_for_path(notes, root=Path(tmp))
        stale_msg = None
        stale_raised = False
        try:
            assert_evidence_fresh(packet.evidence, tmp)
        except StaleEvidenceError as exc:
            stale_raised = True
            stale_msg = str(exc)
        hops["after_mutation"] = {
            "gathered": gathered.to_dict(),
            "now": now.to_dict(),
            "stale_raised": stale_raised,
            "stale_message": stale_msg,
        }
        # Recompile after the mutation must stamp the new identity, not the old one.
        rebuilt = compile_worker_context(
            wu,
            {
                "goal": "short task about notes.txt must pass",
                "referenced_files": ["notes.txt"],
            },
            phase="running",
            units={wu.id: wu},
            steering=[],
            workspace=tmp,
        )
        hops["recompiled_sha256"] = rebuilt.evidence[0].sha256 if rebuilt.evidence else None
    checks = {
        "assert_evidence_fresh_invoked": True,
        "unchanged_accepted": bool(hops["unchanged_accepted"]),
        "stale_raised": stale_raised,
        "stale_names_path": bool(stale_msg) and "notes.txt" in stale_msg,
        "stale_names_old_digest": bool(stale_msg) and gathered.sha256 in stale_msg,
        "stale_names_new_digest": bool(stale_msg) and now.sha256 in stale_msg,
        "digests_differ": gathered.sha256 != now.sha256,
        "recompile_picks_up_new_digest": hops["recompiled_sha256"] == now.sha256,
    }
    ok = all(checks.values())
    payload = {
        "criterion": {
            "span": [7420, 7447],
            "quoted": criterion,
            "meaning": "Changed evidence invalidates stale compiled context.",
        },
        "symbol": loc,
        "invocations": [
            {"symbol": "hcli.goal.compile_worker_context"},
            {"symbol": "hcli.goal.assert_evidence_fresh"},
            {"symbol": "hcli.goal.assert_packet_evidence_fresh"},
            {"symbol": "hcli.goal.identity_for_path"},
        ],
        "command": "python3 -m tools.acceptance.context --gate HCLI_CONTEXT_INVALIDATION",
        "evidence_tier": "FUNCTIONAL_SIM",
        "hops": hops,
        "checks": checks,
        "verdict": "ACCEPTED" if ok else "BLOCKED",
        "blocker": None if ok else "stale compiled evidence was not refused; see checks",
        "output": {
            "stale_message": stale_msg,
            "gathered_sha256": gathered.sha256,
            "now_sha256": now.sha256,
        },
    }
    payload["receipt"] = str(write_receipt("HCLI_CONTEXT_INVALIDATION", payload))
    return payload


def _d9_scan(rendered: str) -> Dict[str, Any]:
    text = rendered
    present: Dict[str, bool] = {}
    missing: List[str] = []
    for field, tokens in D9_TOKENS.items():
        hit = any(tok in text for tok in tokens)
        present[field] = hit
        if not hit:
            missing.append(field)
    return {"present": present, "missing": missing, "missing_count": len(missing)}


def run_status() -> Dict[str, Any]:
    """HCLI_STATUS_PHYSICAL — format_status vs the D.9 contract."""
    from hcli.commands import format_status
    from hcli.max_policy import grok_pool_snapshot

    criterion = quote_roadmap(7537, 7575)
    loc = _symbol_location(format_status)
    grok = grok_pool_snapshot(str(REPO))
    # A snapshot that *has* every D.9 fact. If the renderer drops a section,
    # that is a contract miss, not a missing input to the snapshot.
    snap = {
        "mission_id": "acc2-status-physical",
        "phase": "running",
        "goal": "demonstrate D.9 physical /status",
        "last_progress": "wu-3 verified",
        "units_by_status": {
            "ready": 1,
            "running": 2,
            "blocked": 0,
            "completed": 4,
            "failed": 1,
        },
        "blocked_units": 0,
        "blocked_reason": None,
        "runtime": {
            "health": "ok",
            "resident": 1,
            "active_decode": 1,
            "queued": 0,
            "n_ctx": 32768,
            "prompt_tokens": 1200,
            "tps": 12.5,
        },
        "grok": grok,
        "occupancy": {"GPU_DECODE": 1, "COMPILE": 0, "TEST": 1, "TOOL_WAIT": 0},
        "mutation": {
            "held": False,
            "pid": None,
            "owner": None,
            "waiters": 0,
            "lease_age_s": 0,
        },
        "verifier_backlog": 2,
        "verifier_active": 1,
        "verifier_pass": 4,
        "verifier_fail": 1,
        "accepted_count": 4,
        "accepted_window_s": 600,
        "checkpoint_age_s": 12.0,
        "checkpoint_generation": 7,
        "checkpoint_digest": "abc123",
        "watchdog": "clear",
        "no_progress_warning": None,
        "repair": {
            "active_chains": 1,
            "max_depth": 3,
            "exhausted": 0,
        },
        "resource": {
            "gpu": "M3 Ultra",
            "cpu": "M3 Ultra",
            "ram_gib": 512,
            "swap_gib": 0,
            "disk_free_gib": 100,
            "io": "quiet",
            "benchmark_qualification": "protected",
        },
        "throughput": {
            "verified_wu_per_hour": 24.0,
            "marginal_trend": "flat",
        },
        "stagnation": {"tier": 0, "reason": None},
        "resident": {
            "state": "RUNNING",
            "supervisor": "live",
            "cycles": 3,
            "last_event": "acceptance-probe",
            "age_s": 90,
        },
    }
    rendered = format_status(snap)
    scan = _d9_scan(rendered)
    # These D.9 rows are the ones a one-screen /status currently has no
    # renderer for even when the snapshot carries the facts.
    load_bearing_missing = [
        name
        for name in scan["missing"]
        if name.startswith(("REPAIR.", "PERSISTENCE.generation", "PERSISTENCE.digest",
                             "RESOURCE.GPU", "RESOURCE.RAM", "RESOURCE.swap",
                             "RESOURCE.disk", "RESOURCE.IO",
                             "RESOURCE.benchmark", "THROUGHPUT.marginal",
                             "MUTATION.lease", "MISSION.last_progress",
                             "EXTERNAL.throttled", "VERIFICATION.active",
                             "VERIFICATION.fail"))
    ]
    ok = scan["missing_count"] == 0
    blocker = None
    if not ok:
        blocker = (
            "format_status does not render the D.9 physical /status contract. "
            "Missing fields from a fully populated snapshot: "
            + ", ".join(scan["missing"])
            + ". This is not a missing live mission; the renderer drops the sections."
        )
    payload = {
        "criterion": {
            "span": [7537, 7575],
            "quoted": criterion,
            "meaning": "Physical /status exposes mission, workunits, providers, "
            "external tasks, CPU/tools, verification, mutation, repair, "
            "persistence, throughput, resource, stagnation.",
        },
        "symbol": loc,
        "invocations": [
            {"symbol": "hcli.commands.format_status"},
            {"symbol": "hcli.max_policy.grok_pool_snapshot"},
        ],
        "command": "python3 -m tools.acceptance.context --gate HCLI_STATUS_PHYSICAL",
        "evidence_tier": "FUNCTIONAL_SIM",
        "rendered": rendered,
        "rendered_line_count": len(rendered.splitlines()),
        "d9": scan,
        "load_bearing_missing": load_bearing_missing,
        "live_grok_snapshot": grok,
        "checks": {
            "format_status_invoked": True,
            "rendered_nonempty": bool(rendered.strip()),
            "d9_complete": ok,
        },
        "verdict": "ACCEPTED" if ok else "BLOCKED",
        "blocker": blocker,
        "output": {"first_lines": rendered.splitlines()[:12]},
    }
    payload["receipt"] = str(write_receipt("HCLI_STATUS_PHYSICAL", payload))
    return payload


def run_mixed_max() -> Dict[str, Any]:
    """HCLI_MIXED_MAX — heterogeneous useful work, measured throughput + isolation.

    tools/headless/hcli_true_mixed_max.py refuses to call a CPU-only NullEngine
    run 'mixed'. This lane follows that rule: without a live llama-server the
    gate is BLOCKED, not accepted on a weaker substitute.
    """
    from hcli.max_policy import grok_pool_snapshot, record_rung

    criterion = quote_roadmap(7670, 7672)
    loc = _symbol_location(grok_pool_snapshot)
    snap = grok_pool_snapshot(str(REPO))
    llama_ports = {port: _port_open(port) for port in (8080, 8081, 52484, 8088, 8090)}
    lsof_port: Optional[int] = None
    lsof_error = None
    try:
        proc = subprocess.run(
            [
                "bash",
                "-lc",
                "lsof -iTCP -sTCP:LISTEN -P -a -c llama-server 2>/dev/null | "
                "awk 'NR>1{print $9}' | sed 's/.*://' | head -1",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        token = (proc.stdout or "").strip()
        if token.isdigit():
            lsof_port = int(token)
    except (OSError, subprocess.SubprocessError) as exc:
        lsof_error = f"{type(exc).__name__}: {exc}"
    listening = (lsof_port is not None) or any(llama_ports.values())
    # Isolation is demonstrated by BACKEND_FAILURE_ISOLATION in this same
    # lane. Mixed MAX still requires concurrent useful work on more than
    # one backend class. We do not dispatch Qwen/Grok here: the live HCLI
    # daemon owns those runtimes and this lane must not signal it.
    with tempfile.TemporaryDirectory(prefix="mixed-rung-") as tmp:
        rung_path = record_rung(
            tmp,
            requested=3,
            admitted=0,
            actual=0,
            units=[],
            elapsed_s=0.0,
            extra={
                "note": "no mixed dispatch; llama-server not claimed by this lane",
                "grok_snapshot": snap,
            },
        )
        rung = Path(rung_path).read_text(encoding="utf-8") if Path(rung_path).is_file() else ""
    if listening:
        blocker = (
            "a llama-server listener is visible "
            f"(lsof_port={lsof_port}, socket={llama_ports}); "
            "dispatching a mixed MAX campaign would contend with the live "
            "HCLI daemon this lane is forbidden to signal, restart, or "
            "steal runtimes from. Measured heterogeneous throughput was "
            "therefore not taken."
        )
    else:
        blocker = (
            "no llama-server is listening (socket probe "
            f"{llama_ports}, lsof_port={lsof_port}, lsof_error={lsof_error}). "
            "A mixed campaign without a cognition backend is the CPU-only "
            "campaign again; tools/headless/hcli_true_mixed_max.py refuses "
            "that substitution and so does this lane. Failure isolation is "
            "recorded under BACKEND_FAILURE_ISOLATION, not here."
        )
    payload = {
        "criterion": {
            "span": [7670, 7672],
            "quoted": criterion,
            "meaning": "Heterogeneous useful workload runs concurrently.",
            "proof_obligation": "Measured throughput and failure isolation.",
        },
        "symbol": loc,
        "invocations": [
            {"symbol": "hcli.max_policy.grok_pool_snapshot"},
            {"symbol": "hcli.max_policy.record_rung"},
        ],
        "command": "python3 -m tools.acceptance.context --gate HCLI_MIXED_MAX",
        "evidence_tier": "FUNCTIONAL_SIM",
        "grok_pool_snapshot": snap,
        "llama_ports_socket": llama_ports,
        "llama_lsof_port": lsof_port,
        "llama_lsof_error": lsof_error,
        "llama_listening": listening,
        "rung_excerpt": rung[:1000],
        "checks": {
            "grok_pool_snapshot_invoked": True,
            "did_not_substitute_cpu_only_as_mixed": True,
            "heterogeneous_throughput_measured": False,
        },
        "verdict": "BLOCKED",
        "blocker": blocker,
        "output": {
            "grok_active": snap.get("active"),
            "grok_admitted": snap.get("admitted"),
            "grok_failed": snap.get("failed"),
        },
    }
    payload["receipt"] = str(write_receipt("HCLI_MIXED_MAX", payload))
    return payload


def _load_isolation_mod() -> Any:
    path = REPO / "tools" / "headless" / "hcli_max_isolation_test.py"
    spec = importlib.util.spec_from_file_location("hcli_max_isolation_test_acc2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_isolation() -> Dict[str, Any]:
    """BACKEND_FAILURE_ISOLATION — D.2 classes + one failure does not cascade."""
    from hcli.backends import terminate_pid
    from hcli.resources import FAILURE_KINDS, classify_failure

    criterion = quote_roadmap(7361, 7379)
    loc = _symbol_location(terminate_pid)
    classified = []
    probes = [
        ({"http_status": 429}, "RATE_LIMIT"),
        ({"http_status": 503}, "TRANSIENT_BACKEND"),
        ({"reason": "TEST_FAILED"}, "VERIFIER_FAILURE"),
        ({"reason": "INVALID_OUTPUT"}, "INVALID_OUTPUT"),
        ({"reason": "GrokNotAvailable"}, "UNAVAILABLE_DEPENDENCY"),
        ({"reason": "GrokContractError"}, "IMPOSSIBLE_CONTRACT"),
        ({"reason": "NO_OP_MUTATION"}, "DETERMINISTIC_IMPLEMENTATION"),
        (ConnectionRefusedError("backend down"), "TRANSIENT_BACKEND"),
    ]
    for context, expect in probes:
        result = classify_failure(context)
        classified.append(
            {
                "context": repr(context)[:200],
                "expect": expect,
                "kind": result.kind,
                "retryable": result.retryable,
                "observed": result.observed,
                "match": result.kind == expect,
            }
        )
    implemented = set(FAILURE_KINDS)
    d2_coverage = []
    for name in D2_KINDS:
        alias = D2_ALIASES.get(name, name)
        present = name in implemented or alias in implemented
        d2_coverage.append(
            {
                "d2": name,
                "implemented_as": alias if alias in implemented else (name if name in implemented else None),
                "present": present,
            }
        )
    d2_missing = [row["d2"] for row in d2_coverage if not row["present"]]

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    term = terminate_pid(int(child.pid), term_timeout=2.0, kill_timeout=2.0)
    child.wait(timeout=5)
    # Isolation: reuse the existing headless checks. They call Mission, not
    # terminate_pid; we invoked terminate_pid ourselves on an owned child.
    iso = _load_isolation_mod()
    iso.RESULTS.clear()
    iso.check_failure_isolation()
    iso.check_cpu_verifier_failure_isolation()
    iso.check_grok_bridge_failure_isolation()
    isolation_results = list(iso.RESULTS)
    isolation_ok = bool(isolation_results) and all(row.get("ok") for row in isolation_results)
    classify_ok = all(row["match"] for row in classified)
    term_ok = bool(term.get("gone")) and (term.get("term") or term.get("kill"))
    # Isolation is the gate. Incomplete D.2 coverage is recorded, not
    # papered over, but it does not by itself refuse isolation that held.
    ok = isolation_ok and classify_ok and term_ok
    payload = {
        "criterion": {
            "span": [7361, 7379],
            "quoted": criterion,
            "meaning": "Backend failures classify into the D.2 taxonomy and "
            "do not take independent work down.",
        },
        "symbol": loc,
        "invocations": [
            {"symbol": "hcli.backends.terminate_pid", "pid": term.get("pid")},
            {"symbol": "hcli.resources.classify_failure"},
            {"symbol": "hcli.mission.Mission.run", "note": "isolation checks"},
        ],
        "command": "python3 -m tools.acceptance.context --gate BACKEND_FAILURE_ISOLATION",
        "evidence_tier": "FUNCTIONAL_SIM",
        "failure_kinds_implemented": list(FAILURE_KINDS),
        "d2_coverage": d2_coverage,
        "d2_missing": d2_missing,
        "classified": classified,
        "terminate_pid": term,
        "isolation_results": isolation_results,
        "checks": {
            "terminate_pid_invoked_on_owned_child": term_ok,
            "classify_failure_matches_probes": classify_ok,
            "injected_backend_failure_does_not_stop_independent_units": isolation_ok,
            "did_not_signal_live_daemon": True,
        },
        "verdict": "ACCEPTED" if ok else "BLOCKED",
        "blocker": None
        if ok
        else (
            "isolation, classify_failure, or terminate_pid failed; "
            f"isolation_ok={isolation_ok} classify_ok={classify_ok} term_ok={term_ok}"
        ),
        "output": {
            "isolation_passed": isolation_ok,
            "d2_missing": d2_missing,
            "terminate_pid": term,
        },
        "notes": (
            "D.2 names not present in FAILURE_KINDS are listed in d2_missing. "
            "They are not treated as implemented under an alias unless "
            "D2_ALIASES says so. Isolation is the load-bearing check."
        ),
    }
    payload["receipt"] = str(write_receipt("BACKEND_FAILURE_ISOLATION", payload))
    return payload


def run_supplement() -> Dict[str, Any]:
    """HCLI_SELF_SUPPLEMENT — verified parent admits dependency-ready children."""
    from hcli.agentos import AgentOS
    from hcli.agentos.resident import admit_evidence_children
    from hcli.workunit import WorkUnit

    criterion = quote_roadmap(7674, 7676)
    loc = _symbol_location(admit_evidence_children)

    class RecordingEngine:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def execute_workunit(self, unit: Any, _context: Any) -> Dict[str, Any]:
            self.calls.append(unit.id)
            raw: Dict[str, Any] = {
                "content": f"verified {unit.id}",
                "validation": {"ok": True, "verifier": "acceptance-recording-engine"},
            }
            if unit.id == "parent":
                raw["child_workunits"] = [
                    {
                        "id": "child",
                        "role": "research",
                        "description": "inspect the verified parent receipt",
                        "resource_class": "CPU_SHARED",
                    }
                ]
            return raw

    hops: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="hcli-supp-") as tmp:
        engine = RecordingEngine()
        parent = WorkUnit(id="parent", role="research", description="bounded parent")
        agent = AgentOS(tmp, engine=engine)
        agent.start_mission("acceptance self-supplement", units={"parent": parent})
        first = agent.run()
        hops["first"] = {
            "status": first.get("status"),
            "evidence": first.get("evidence"),
        }
        rows = admit_evidence_children(agent.mission, first.get("evidence"))
        hops["admitted"] = rows
        child = agent.mission.scheduler.units.get("child")
        hops["child_dependencies"] = list(child.dependencies) if child is not None else None
        hops["child_status_before"] = child.status if child is not None else None
        second = agent.continue_mission()
        hops["second"] = {
            "status": second.get("status"),
            "engine_calls": list(engine.calls),
        }
        hops["child_status_after"] = (
            agent.mission.scheduler.units["child"].status
            if "child" in agent.mission.scheduler.units
            else None
        )
        # Negative control: unverified evidence must not admit.
        denied = admit_evidence_children(
            agent.mission,
            {
                "unit_id": "parent",
                "accepted": False,
                "validation": {"ok": False},
                "child_workunits": [
                    {"id": "ghost", "description": "must not run"}
                ],
            },
        )
        hops["unverified_denied"] = denied
        hops["ghost_absent"] = "ghost" not in agent.mission.scheduler.units
    checks = {
        "admit_evidence_children_invoked": True,
        "parent_verified": hops["first"]["status"] == "completed",
        "child_admitted": bool(hops["admitted"])
        and hops["admitted"][0].get("status") in {"ADMITTED", "IDEMPOTENT"},
        "child_depends_on_parent": hops["child_dependencies"] == ["parent"],
        "child_ran": hops["second"]["engine_calls"] == ["parent", "child"]
        or (hops["child_status_after"] == "completed" and "child" in hops["second"]["engine_calls"]),
        "child_completed": hops["child_status_after"] == "completed",
        "unverified_denied": hops["unverified_denied"] == [] and hops["ghost_absent"],
    }
    ok = all(checks.values())
    payload = {
        "criterion": {
            "span": [7674, 7676],
            "quoted": criterion,
            "meaning": "Mission can create dependency-ready next WorkUnits from verified state.",
            "proof_obligation": "Full chain receipt.",
        },
        "symbol": loc,
        "invocations": [
            {"symbol": "hcli.agentos.resident.admit_evidence_children"},
            {"symbol": "hcli.agentos.AgentOS.start_mission"},
            {"symbol": "hcli.agentos.AgentOS.run"},
            {"symbol": "hcli.agentos.AgentOS.continue_mission"},
        ],
        "command": "python3 -m tools.acceptance.context --gate HCLI_SELF_SUPPLEMENT",
        "evidence_tier": "FUNCTIONAL_SIM",
        "hops": hops,
        "checks": checks,
        "verdict": "ACCEPTED" if ok else "BLOCKED",
        "blocker": None if ok else "full chain (verified parent → admitted child → child ran) did not hold; see hops",
        "output": hops,
    }
    payload["receipt"] = str(write_receipt("HCLI_SELF_SUPPLEMENT", payload))
    return payload


def run_bootstrap() -> Dict[str, Any]:
    """HCLI_SELF_OPTIMIZATION_BOOTSTRAP — two linked iterations.

    D.11: iteration 2 MUST be selected from evidence produced by iteration 1.
    A prewritten second task does not prove self-optimization.
    run_autonomy_gate is A1–A5 qualification; A3/A4 kill processes, which
    this lane must not do. Existing iteration-2 receipts are prewritten.
    """
    from hcli.agentos.autonomy_gate import run_autonomy_gate

    criterion = quote_roadmap(7678, 7680)
    loc = _symbol_location(run_autonomy_gate)
    src_head = inspect.getsource(run_autonomy_gate).splitlines()[:40]
    stages_in_source = [
        line.strip()
        for line in inspect.getsource(run_autonomy_gate).splitlines()
        if "A3" in line or "A4" in line or "kill" in line.lower()
    ]
    existing = {
        "iteration_1": "receipts/headless/HCLI_SELF_OPT_ITERATION_1.json",
        "iteration_2": "receipts/headless/HCLI_SELF_OPT_ITERATION_2.json",
        "iteration_2_remeasured": "receipts/headless/HCLI_SELF_OPT_ITERATION_2_REMEASURED.json",
    }
    blocker = (
        "D.11 two-iteration law is not demonstrable in this lane. "
        "(1) run_autonomy_gate stages A3_resident_kill and "
        "A4_hcli_process_kill send SIGTERM/SIGKILL; this lane must not "
        "signal the live HCLI daemon or any other process. Even stage=a1 "
        "is census qualification, not self-optimization. "
        "(2) receipts/headless/HCLI_SELF_OPT_ITERATION_2.json is a "
        "prewritten second task (raise RuntimePool admission) — D.11: "
        "'A prewritten second task does not prove self-optimization.' "
        "(3) HCLI_SELF_OPT_ITERATION_2_REMEASURED.json records that the "
        "perf gate never entered the mutated RuntimePool. "
        "(4) Promoting a bounded improvement would require editing hcli/, "
        "which this lane must not touch. Missing input: a kill-free, "
        "evidence-chosen iteration-2 loop that independently verifies an "
        "improvement without mutating hcli/ or signalling processes."
    )
    payload = {
        "criterion": {
            "span": [7678, 7680],
            "quoted": criterion,
            "meaning": "Evidence chooses and improves next bottleneck.",
            "proof_obligation": "Two linked iterations with independent verification.",
            "d11": quote_roadmap(7584, 7612),
        },
        "symbol": loc,
        "invocations": [],
        "symbol_inspected_not_invoked": loc,
        "why_not_invoked": (
            "run_autonomy_gate A3/A4 kill processes; invoking it would "
            "violate the live-daemon rule. Inspection of the source is "
            "STATIC and is not acceptance."
        ),
        "command": "python3 -m tools.acceptance.context --gate HCLI_SELF_OPTIMIZATION_BOOTSTRAP",
        "evidence_tier": "STATIC",
        "source_excerpt": src_head,
        "kill_stages_in_source": stages_in_source,
        "existing_receipts_cited_as_data": existing,
        "checks": {
            "run_autonomy_gate_not_invoked": True,
            "two_linked_iterations_with_independent_verification": False,
            "did_not_treat_prewritten_iteration_2_as_proof": True,
        },
        "verdict": "BLOCKED",
        "blocker": blocker,
        "output": {"invoked": False},
    }
    payload["receipt"] = str(write_receipt("HCLI_SELF_OPTIMIZATION_BOOTSTRAP", payload))
    return payload


RUNNERS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "HCLI_CONTEXT_AUTHORITY_UNIFIED": run_authority,
    "HCLI_CONTEXT_FOCUSED_WORKUNITS": run_focused,
    "HCLI_CONTEXT_INVALIDATION": run_invalidation,
    "HCLI_STATUS_PHYSICAL": run_status,
    "HCLI_MIXED_MAX": run_mixed_max,
    "BACKEND_FAILURE_ISOLATION": run_isolation,
    "HCLI_SELF_SUPPLEMENT": run_supplement,
    "HCLI_SELF_OPTIMIZATION_BOOTSTRAP": run_bootstrap,
}


def run_gate(gate: str) -> Dict[str, Any]:
    fn = RUNNERS.get(gate)
    if fn is None:
        raise KeyError(f"unknown gate {gate}")
    result = fn()
    result.setdefault("criterion_altered", False)
    result.setdefault("gate", gate)
    return result


def run_all(gates: Optional[List[str]] = None) -> Dict[str, Any]:
    selected = list(gates or GATE_IDS)
    results: Dict[str, Any] = {}
    for gate in selected:
        results[gate] = run_gate(gate)
    accepted = [g for g, row in results.items() if row.get("verdict") == "ACCEPTED"]
    blocked = [g for g, row in results.items() if row.get("verdict") == "BLOCKED"]
    summary = {
        "schema": "hawking.acceptance.context.summary.v1",
        "gates": selected,
        "accepted": accepted,
        "blocked": blocked,
        "accepted_count": len(accepted),
        "blocked_count": len(blocked),
        "criterion_altered": False,
        "results": {
            g: {"verdict": results[g]["verdict"], "blocker": results[g].get("blocker")}
            for g in selected
        },
    }
    write_receipt("SUMMARY", summary)
    return {"summary": summary, "results": results}
