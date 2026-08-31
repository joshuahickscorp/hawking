"""QUALIFICATION_PIPELINE — sequence the existing pieces; never seize the GPU.

The planning, preflight, contamination and work-unit modules already exist.
This sidecar sequences them into one resumable pipeline that is structurally
incapable of taking GPU authority. Stages 10-12 emit a REQUEST/SPEC and stop.
execute() raises unless an existing HCLI lease is present AND the machine is
QUIESCENT AND --execute was passed — and even then this sidecar has no lease,
so it still raises.

    python3 tools/future/qualification_pipeline.py --dry-run
    python3 tools/future/qualification_pipeline.py --build
    python3 -m pytest tools/future/test_qualification_pipeline.py -q

Everything emitted here is STATIC_ONLY, bench state UNKNOWN, gpu_authority
false. This module produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hcli.agentos.benchmark_boundary import (
    DIAGNOSTIC_CONTAMINATED,
    QUALIFIED_PROTECTED,
    classify_window,
)
from tools.future import candidate_planner as cp
from tools.future import contamination as C
from tools.future import repro_science as rs
from tools.future import static_kernel_verify as skv
from tools.future import workunit_species as ws
from tools.future._common import HARDWARE_FIELDS, git

RECEIPT = "QUALIFICATION_PIPELINE.json"
SCHEMA = "hawking.future.qualification_pipeline.v1"
VERSION = 1
RECORDED_BY = "tools.future.qualification_pipeline.py"
CHECKPOINT_SCHEMA = "hawking.future.qualification_pipeline.checkpoint.v1"

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

STAGES: tuple[str, ...] = (
    "identify_lease_availability",
    "assess_machine_quiescence",
    "identify_contaminating_worker",
    "select_ready_candidates",
    "static_preflight_drop",
    "emit_ab_execution_plan",
    "parity_verification_spec",
    "failure_classification",
    "survivor_promotion_prerequisites",
    "protected_lease_request",
    "protected_measurement_spec",
    "scoreboard_update_spec",
    "derive_next_workunits",
)

# Recovered from hcli/agentos/protected_accelerator_benchmark.py. Cited, not imported:
# importing that module would load the runner that takes the exclusive lock.
HCLI_LOCK_REL = Path(".hcli") / "locks" / "protected-accelerator-bench.lock"
HCLI_LOCK_NAME = "protected-accelerator-bench.lock"

# Recovered from tools/odyssey/gpu_cleanliness.py PAUSE_PATTERN. Codex surface;
# this sidecar never SIGSTOPs anyone. Pausing STANDING load forges a speedup.
PAUSABLE_PATTERNS = ("hf download", "lake_filler.py")

EVIDENCE_QUEUE = (
    REPO / "receipts" / "future" / "evidence" / "ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
)
FRONTIER_PATH = REPO / "receipts" / "future" / "CLAUDE_GLOBAL_FRONTIER.json"
SCOREBOARD_REL = "receipts/headless/ACCELERATOR_SCOREBOARD.json"

AUTHORITY_REFUSAL: dict[str, Any] = {
    "executes_benchmark": False,
    "acquires_lease": False,
    "signals_process": False,
    "quiesces_worker": False,
    "gpu_authority": False,
}

READY_STATUSES = frozenset({"READY_DIAGNOSTIC", "READY_PROTECTED"})


class ExecuteRefused(rs.FailClosed):
    """execute() refused. Named condition so tests can watch each guard fire."""

    def __init__(self, condition: str, reason: str) -> None:
        self.condition = condition
        super().__init__(condition, reason)


class AuthorityBoundaryError(rs.FailClosed):
    """A stage asked to start a benchmark, create a lease, or signal a process."""

    def __init__(self, action: str) -> None:
        self.action = action
        super().__init__(
            "gpu_authority",
            f"sidecar refused {action}: no stage may start a GPU benchmark, "
            "create or steal a lease, kill a process, or quiesce a worker",
        )


class PipelineInterrupted(rs.FailClosed):
    """Fault-injected interruption. Resume from the sealed checkpoint."""

    def __init__(self, after_stage: str, checkpoint: dict[str, Any]) -> None:
        self.after_stage = after_stage
        self.checkpoint = checkpoint
        super().__init__(
            "killed_subprocess",
            f"pipeline interrupted after {after_stage}; restore the checkpoint to resume. "
            "A partial in-progress stage is not a result.",
        )


# ---------------------------------------------------------------------------
# Structural refusals. A guard nobody has watched fail is not a guard.
# ---------------------------------------------------------------------------


def refuse_start_benchmark(*_a: Any, **_k: Any) -> None:
    raise AuthorityBoundaryError("start_benchmark")


def refuse_create_lease(*_a: Any, **_k: Any) -> None:
    raise AuthorityBoundaryError("create_lease")


def refuse_signal_process(*_a: Any, **_k: Any) -> None:
    raise AuthorityBoundaryError("signal_process")


def refuse_quiesce_worker(*_a: Any, **_k: Any) -> None:
    raise AuthorityBoundaryError("quiesce_worker")


def _authority(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = dict(AUTHORITY_REFUSAL)
    if extra:
        out.update(dict(extra))
    return out


# ---------------------------------------------------------------------------
# Queue / plan / preflight loaders — compose siblings, do not fork them
# ---------------------------------------------------------------------------


def load_qualification_queue(path: Path | None = None) -> dict[str, Any]:
    """Read Codex's live queue. Never write it. Evidence copy is a fallback."""
    if path is not None:
        return cp.load_queue(path)
    try:
        return cp.load_queue()
    except cp.QueueNotFoundError:
        if EVIDENCE_QUEUE.is_file():
            return cp.load_queue(EVIDENCE_QUEUE)
        raise


def load_staged_plan(queue: Mapping[str, Any]) -> dict[str, Any]:
    """Delegate the factorial plan to candidate_planner.plan_from_queue."""
    return cp.plan_from_queue(queue)


def run_static_preflight() -> dict[str, Any]:
    """Zero-GPU host/shader ABI scan. Invokes the sibling; does not reimplement it."""
    return skv.scan()


# ---------------------------------------------------------------------------
# Stage 1 — READ lease state. Never create, never flock exclusive.
# ---------------------------------------------------------------------------


def _lsof_holders(path: Path) -> dict[str, Any]:
    """Read-only inspection of who has the lock file open. Never flock."""
    try:
        proc = subprocess.run(
            ["lsof", "-t", str(path)],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception as exc:  # probe failure is evidence, not a lease
        return {
            "status": "UNKNOWN",
            "pids": [],
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if proc.returncode not in (0, 1):
        return {
            "status": "UNKNOWN",
            "pids": [],
            "reason": f"lsof rc={proc.returncode} {(proc.stderr or '')[:120]}",
        }
    pids = sorted({int(tok) for tok in proc.stdout.split() if tok.isdigit()})
    return {"status": "OK", "pids": pids, "reason": None}


def read_hcli_lease_state(repo: Path | None = None) -> dict[str, Any]:
    """Identify whether an existing HCLI protected lease is present.

    Fail closed: present is True only when a holder can be observed WITHOUT
    taking the lock. Exclusive flock is a seizure and is never attempted.
    A missing path is not created. mkdir of .hcli is Codex's job.
    """
    root = repo if repo is not None else REPO
    lock_path = root / HCLI_LOCK_REL
    exists = lock_path.is_file()
    holders: dict[str, Any] = {"status": "SKIPPED", "pids": [], "reason": "lock file absent"}
    if exists:
        holders = _lsof_holders(lock_path)
    present = bool(exists and holders.get("status") == "OK" and holders.get("pids"))
    if present:
        reason = (
            f"HCLI lock {HCLI_LOCK_REL.as_posix()} is held by pids {holders['pids']}; "
            "sidecar observed this read-only and did not take the lock"
        )
    elif not exists:
        reason = (
            f"no existing HCLI lease: {HCLI_LOCK_REL.as_posix()} is absent. "
            "queue_policy.protected_start_requires_existing_hcli_lease. "
            "sidecar will not create .hcli/locks or call _try_lock"
        )
    else:
        reason = (
            f"lock file exists but no holder could be proven without flock "
            f"(lsof status={holders.get('status')!r} reason={holders.get('reason')!r}). "
            "fail closed: present=false. flock would be a seizure"
        )
    return _authority(
        {
            "kind": "READ",
            "present": present,
            "lock_path": str(lock_path),
            "lock_rel": HCLI_LOCK_REL.as_posix(),
            "lock_name": HCLI_LOCK_NAME,
            "lock_file_exists": exists,
            "holders": holders,
            "probe": "lsof -t on existing path only; never fcntl.LOCK_EX, never mkdir",
            "recovered_from": "hcli/agentos/protected_accelerator_benchmark.py LOCK_NAME / _lock_path",
            "not_called": [
                "hcli.agentos.protected_accelerator_benchmark._try_lock",
                "hcli.agentos.protected_accelerator_benchmark.run_protected_accelerator_benchmark",
                "lab.lease.SingletonLease",
            ],
            "reason": reason,
            "execution_ok": present,
        }
    )


# ---------------------------------------------------------------------------
# Stage 2 / 3 — contamination snapshot. Never quiesce.
# ---------------------------------------------------------------------------


def classify_worker(proc: Mapping[str, Any]) -> dict[str, Any]:
    """Would Codex G013 permit pausing this neighbour? This sidecar still will not."""
    name = str(proc.get("name") or "")
    pausable = any(token in name for token in PAUSABLE_PATTERNS)
    klass = "PAUSABLE" if pausable else "STANDING"
    if pausable:
        policy_reason = (
            "Codex tools/odyssey/gpu_cleanliness.py classifies campaign I/O matching "
            f"{list(PAUSABLE_PATTERNS)} as PAUSABLE. This sidecar still must not "
            "SIGSTOP/SIGKILL it: we emit identification only"
        )
    else:
        policy_reason = (
            "STANDING (or unidentified) neighbour: pausing it would forge a speedup. "
            "queue_policy.protected_start_requires_machine_quiescence is a requirement "
            "on the machine, not permission for this sidecar to quiesce a worker"
        )
    return {
        "pid": proc.get("pid"),
        "name": name,
        "cpu_pct": proc.get("cpu_pct"),
        "rss_gib": proc.get("rss_gib"),
        "class": klass,
        "policy_would_permit_quiesce": pausable,
        "sidecar_will_quiesce": False,
        "reason": policy_reason,
    }


def assess_quiescence(snap: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    record = snap if snap is not None else C.snapshot(benchmark_ordinal=None)
    klass = C.classify_contamination(record)
    return record, klass


# ---------------------------------------------------------------------------
# Stage 4 / 5 — ready selection + preflight DROP
# ---------------------------------------------------------------------------


def select_ready_candidates(
    queue: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [r for r in (queue.get("candidates") or []) if isinstance(r, Mapping)]
    ready = [r for r in rows if str(r.get("status") or "") in READY_STATUSES]
    ready.sort(key=lambda r: str(r.get("candidate_id") or ""))
    staged = plan.get("staged_factorial_plan") or plan
    independent = staged.get("independent_set") or {}
    measurable = list(independent.get("measurable_now") or [])
    return _authority(
        {
            "n_queue": len(rows),
            "n_ready": len(ready),
            "by_status": {
                status: sum(1 for r in ready if r.get("status") == status)
                for status in sorted(READY_STATUSES)
            },
            "candidate_ids": [str(r.get("candidate_id")) for r in ready],
            "plan_measurable_now": list(measurable),
            "plan_cell_count": (staged.get("staged") or {}).get("cell_count"),
            "source": "queue candidates with READY_DIAGNOSTIC/READY_PROTECTED + candidate_planner independent_set",
            "candidates": ready,
        }
    )


def _norm_path(value: str) -> str:
    return value.lower().replace("\\", "/").split("::", 1)[0]


def finding_touches_candidate(finding: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    """Map a preflight ERROR onto a queue row via source_evidence / kernel / shader.

    Short stems (attn, moe, mod) are ignored so an unmapped organ defect does not
    drop every Qwen27 candidate. A genuine hit is a path or a long kernel stem.
    """
    sources = [_norm_path(str(s)) for s in (candidate.get("source_evidence") or [])]
    cid = str(candidate.get("candidate_id") or "").lower()
    host = _norm_path(str(finding.get("host") or "")).split(":")[0]
    shader = _norm_path(str(finding.get("shader") or "")).split(":")[0]
    kernel = str(finding.get("kernel") or "").lower()
    paths = [p for p in (host, shader) if p]
    for src in sources:
        for path in paths:
            if path and (path == src or src in path or path in src):
                return True
        if kernel and len(kernel) >= 8 and kernel in src:
            return True
        stem = Path(src).stem.lower()
        if stem and len(stem) >= 8 and (stem in kernel or (kernel and kernel in stem)):
            return True
    if kernel and len(kernel) >= 12:
        kn = kernel.replace("_", "-")
        if kn in cid or cid.replace("-", "_") in kernel:
            return True
    return False


def drop_blocking_candidates(
    candidates: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    errors = [
        f
        for f in (preflight.get("findings") or [])
        if isinstance(f, Mapping) and f.get("severity") == "ERROR"
    ]
    dropped: list[dict[str, Any]] = []
    survivors: list[Mapping[str, Any]] = []
    for cand in candidates:
        hits = [f for f in errors if finding_touches_candidate(f, cand)]
        if hits:
            dropped.append(
                {
                    "candidate_id": cand.get("candidate_id"),
                    "status": cand.get("status"),
                    "n_hits": len(hits),
                    "checks": sorted({str(h.get("check")) for h in hits}),
                    "sample_message": str(hits[0].get("message") or "")[:240],
                    "reason": "static preflight ERROR mapped onto this candidate; DROP rather than waste a protected window",
                }
            )
        else:
            survivors.append(cand)
    unmapped = [
        {
            "check": e.get("check"),
            "kernel": e.get("kernel"),
            "host": e.get("host"),
            "shader": e.get("shader"),
            "message": str(e.get("message") or "")[:240],
        }
        for e in errors
        if not any(finding_touches_candidate(e, c) for c in candidates)
    ]
    return _authority(
        {
            "preflight_schema": preflight.get("schema"),
            "blocking_defect_count": int(preflight.get("blocking_defect_count") or len(errors)),
            "would_waste_a_protected_window": bool(errors) or bool(preflight.get("would_waste_a_protected_window")),
            "static_correctness_does_not_prove_speed": True,
            "does_not_substitute_for_protected_measurement": True,
            "n_input": len(list(candidates)),
            "n_dropped": len(dropped),
            "n_survivors": len(survivors),
            "dropped": dropped,
            "survivor_ids": [str(c.get("candidate_id")) for c in survivors],
            "unmapped_blocking_defects": unmapped,
            "note": (
                "unmapped ERROR findings still would_waste_a_protected_window for the "
                "kernels they name, but they do not drop a candidate they do not touch"
            ),
            "survivors": list(survivors),
            "source": "tools.future.static_kernel_verify.scan / analyze",
        }
    )


# ---------------------------------------------------------------------------
# Stages 6-9 — plan / spec. Never a measurement.
# ---------------------------------------------------------------------------


def emit_ab_plan(
    plan: Mapping[str, Any],
    survivors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    keep = {str(c.get("candidate_id")) for c in survivors}
    staged = plan.get("staged_factorial_plan") or plan
    cells_in = list(staged.get("cells") or [])
    cells = []
    for cell in cells_in:
        members = [str(m) for m in (cell.get("candidates") or [])]
        if members and all(m in keep for m in members):
            cells.append(
                {
                    "cell_id": cell.get("cell_id"),
                    "stage": cell.get("stage"),
                    "kind": cell.get("kind"),
                    "status": cell.get("status"),
                    "candidates": members,
                    "planned_evidence_rung": cell.get("planned_evidence_rung"),
                    "executes_benchmark": False,
                    "acquires_lease": False,
                }
            )
    cells.sort(key=lambda c: str(c.get("cell_id") or ""))
    return _authority(
        {
            "protocol": "caller interleaves A/B (ABAB) so both arms share thermal and page-cache state",
            "stats": "tools.future.contamination.paired_ab_stats — median pairwise B/A, IQR, bootstrap CI; never a mean",
            "min_pairs": C.MIN_PAIRS,
            "never_reports": ["mean", "average"],
            "sufficient_for_decision_requires": C.MIN_PAIRS,
            "n_cells": len(cells),
            "n_cells_dropped_because_member_failed_preflight": len(cells_in) - len(cells),
            "cells": cells,
            "source": "tools.future.candidate_planner.staged_factorial_plan filtered to preflight survivors",
            "this_sidecar_does_not_source_pairs": True,
        }
    )


def parity_spec(survivors: Sequence[Mapping[str, Any]], queue: Mapping[str, Any]) -> dict[str, Any]:
    funnel = queue.get("funnel") or {}
    rows = []
    for cand in survivors:
        rows.append(
            {
                "candidate_id": cand.get("candidate_id"),
                "parity_contract": cand.get("parity_contract"),
                "capability_contract": cand.get("capability_contract"),
                "status": cand.get("status"),
                "executed": False,
            }
        )
    return _authority(
        {
            "kind": "SPEC",
            "rule": (
                "any numerical/output divergence rejects the candidate; "
                "capability requires complete accepted-token, zero fallback, "
                "independent resident identity"
            ),
            "funnel_native_parity": list(funnel.get("native_parity") or []),
            "funnel_promotion_rule": funnel.get("promotion_rule"),
            "n": len(rows),
            "candidates": rows,
            "note": "spec only. sidecar does not run a resident or compare token ids",
        }
    )


def failure_classification_spec(queue: Mapping[str, Any]) -> dict[str, Any]:
    transitions = queue.get("status_transitions") or {}
    return _authority(
        {
            "kind": "SPEC",
            "invents_no_new_status": True,
            "queue_statuses": list(queue.get("candidate_statuses") or sorted(transitions)),
            "status_transitions": {k: list(v) for k, v in sorted(transitions.items())},
            "classes": {
                "PREFLIGHT_DROP": "static ERROR mapped onto the candidate; never occupies a GPU window",
                "DIAGNOSTIC_REJECT": "READY_DIAGNOSTIC arm failed; DIAGNOSTIC_RELATIVE never promotes",
                "PROTECTED_REJECT": "READY_PROTECTED arm failed parity/capability under a real lease",
                "BLOCKED": "an evidence parent is missing; legal next status is STATIC_ONLY",
                "PROTECTED_PASS": "only a QUIESCENT PROTECTED_ABSOLUTE receipt with all required fields",
                "INFRA": list(rs.FAULT_NAMES),
            },
            "infra_faults_are": "tools.future.repro_science.FAULT_NAMES — fail closed, never SKIP as PASS",
            "diagnostic_results_do_not_promote": bool(
                (queue.get("queue_policy") or {}).get("diagnostic_results_do_not_promote")
            ),
        }
    )


def promotion_prerequisites_spec(
    queue: Mapping[str, Any],
    survivors: Sequence[Mapping[str, Any]],
    contamination_class: str,
) -> dict[str, Any]:
    table = cp.promotion_table(queue, survivors)
    static_record = {
        "measurement_class": "STATIC_ONLY",
        "contamination_class": contamination_class,
        "ab_stats": {
            "sufficient_for_decision": False,
            "reason": "sidecar did not source paired samples from hardware",
        },
    }
    try:
        C.assert_promotable(static_record)
    except C.PromotionRefused as exc:
        gate = {"fired": True, "message": str(exc)}
    else:
        raise rs.FailClosed(
            "promotion_gate",
            "assert_promotable did not refuse a STATIC_ONLY record; a guard nobody watched fail is not a guard",
        )
    policy = queue.get("queue_policy") or {}
    meas = queue.get("measurement_contract") or {}
    return _authority(
        {
            "kind": "SPEC",
            "sidecar_cannot_satisfy": True,
            "requires": [
                "measurement_class == PROTECTED_ABSOLUTE",
                "contamination_class == QUIESCENT",
                f"ab_stats.sufficient_for_decision with min_pairs={C.MIN_PAIRS}",
                "queue_policy.protected_start_requires_existing_hcli_lease",
                "queue_policy.protected_start_requires_machine_quiescence",
                "measurement_contract.protected_pass_requires_all_fields",
            ],
            "queue_policy": {
                "protected_start_requires_existing_hcli_lease": bool(
                    policy.get("protected_start_requires_existing_hcli_lease")
                ),
                "protected_start_requires_machine_quiescence": bool(
                    policy.get("protected_start_requires_machine_quiescence")
                ),
                "diagnostic_results_do_not_promote": bool(
                    policy.get("diagnostic_results_do_not_promote")
                ),
            },
            "measurement_contract_null_policy": meas.get("null_policy"),
            "assert_promotable_refuses_this_sidecar": gate,
            "per_survivor": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "status": row.get("status"),
                    "legal_next_statuses": row.get("legal_next_statuses"),
                    "can_enter_protected_pass": row.get("can_enter_protected_pass"),
                    "promotion_prerequisites": row.get("promotion_prerequisites"),
                }
                for row in table
            ],
            "source": "tools.future.candidate_planner.promotion_table + tools.future.contamination.assert_promotable",
        }
    )


# ---------------------------------------------------------------------------
# Stages 10-12 — REQUEST / SPEC. Stop. Never seize, never measure, never write Codex receipts.
# ---------------------------------------------------------------------------


def lease_request_spec(lease: Mapping[str, Any]) -> dict[str, Any]:
    return _authority(
        {
            "kind": "REQUEST",
            "never_seizure": True,
            "acquired": False,
            "would_require": [
                "an existing HCLI protected lease (not created here)",
                "machine QUIESCENT (not quiesced here)",
                "explicit --execute (still insufficient: sidecar has no GPU authority)",
            ],
            "target_lock": HCLI_LOCK_REL.as_posix(),
            "observed_present": bool(lease.get("present")),
            "not_called": list(lease.get("not_called") or []),
            "reason": (
                "stage 10 emits a request and stops. creating, stealing, or "
                "fcntl.LOCK_EX on the HCLI lock is a seizure of GPU authority"
            ),
        }
    )


def protected_measurement_spec(
    queue: Mapping[str, Any],
    survivors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    meas = queue.get("measurement_contract") or {}
    required = list(meas.get("required_fields") or [])
    fields = {name: None for name in required}
    for name in sorted(HARDWARE_FIELDS):
        fields.setdefault(name, None)
    boundary = classify_window(None, None, {"state": "UNKNOWN"}, not_for_promotion=True)
    commands = []
    for cand in survivors:
        argv = list(cand.get("protected_command") or [])
        commands.append(
            {
                "candidate_id": cand.get("candidate_id"),
                "argv": argv,
                "executed": False,
                "note": "argv copied from the queue as a spec; sidecar does not spawn it",
            }
        )
    return _authority(
        {
            "kind": "SPEC",
            "measurement_class_if_a_real_lease_ran": "PROTECTED_ABSOLUTE",
            "this_sidecar_emits": "STATIC_ONLY",
            "metric_scope": meas.get("metric_scope"),
            "null_policy": meas.get("null_policy"),
            "protected_pass_requires_all_fields": bool(meas.get("protected_pass_requires_all_fields")),
            "required_fields": fields,
            "hcli_boundary": {
                "function": "hcli.agentos.benchmark_boundary.classify_window",
                "QUALIFIED_PROTECTED": QUALIFIED_PROTECTED,
                "DIAGNOSTIC_CONTAMINATED": DIAGNOSTIC_CONTAMINATED,
                "empty_window_class": boundary.get("benchmark_class"),
                "empty_window_is_not_qualified": boundary.get("benchmark_class") != QUALIFIED_PROTECTED,
                "NOT_FOR_PROMOTION": boundary.get("NOT_FOR_PROMOTION"),
            },
            "protected_commands": commands,
            "native_mission_gate": (
                "hcli/agentos/native_mission_gate.py is the live native tool/verifier "
                "mission; this sidecar does not import or run it"
            ),
        }
    )


def scoreboard_update_spec(survivors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for cand in survivors:
        row = {
            "candidate_id": cand.get("candidate_id"),
            "bench_state": "UNKNOWN",
            "measurement_state": "STATIC_ONLY",
            "gpu_authority": False,
            "would_write": False,
        }
        for field in sorted(HARDWARE_FIELDS):
            row[field] = None
        rows.append(row)
    return _authority(
        {
            "kind": "SPEC",
            "would_update": False,
            "target_receipt": SCOREBOARD_REL,
            "owner": "Codex",
            "schema_if_codex_wrote_it": "hawking.accelerator.scoreboard.v1",
            "promotion_allowed": False,
            "proposed_rows": rows,
            "reason": (
                "sidecar must not write receipts/headless. a scoreboard row with a "
                "hardware number would be an invented measurement"
            ),
        }
    )


# ---------------------------------------------------------------------------
# Stage 13 — next WorkUnits. Proposals, not dispatch.
# ---------------------------------------------------------------------------


def derive_next_workunits(
    survivors: Sequence[Mapping[str, Any]],
    dropped: Sequence[Mapping[str, Any]],
    lease_present: bool,
) -> dict[str, Any]:
    units: list[dict[str, Any]] = []
    ready_ids = {str(c.get("candidate_id")) for c in survivors}
    for cand in survivors:
        row = ws._physical_unit(cand, ready_ids=ready_ids)
        row["status"] = "pending"
        row["classification"] = "STATIC_ONLY"
        row["blocked_reason"] = (
            None
            if lease_present
            else "proposal only: protected start requires an existing HCLI lease this sidecar does not hold"
        )
        if not lease_present:
            row["status"] = "blocked"
        ws.validate_emitted_unit(row)
        units.append(row)
    for drop in dropped:
        cid = str(drop.get("candidate_id") or "")
        if not cid:
            continue
        row = ws.emit_hcli_workunit(
            id=f"accelerator.physical.{cid}",
            role="accelerator_physical_qualification",
            description=f"Dropped {cid} at static preflight; do not occupy a protected window",
            dependencies=[],
            resource_class="STATIC_ANALYSIS",
            verifier=f"accelerator.physical.{cid}",
            provider="future.qualification_pipeline",
            effect_class="READ_ONLY",
            status="blocked",
            classification="BLOCKED",
            extras={
                "candidate_id": cid,
                "species": "accelerator_candidate_qualification",
                "blocked_reason": drop.get("reason"),
                "candidate_status": "BLOCKED",
                "requires_quiescence": False,
            },
        )
        ws.validate_emitted_unit(row)
        units.append(row)
    lease_unit = ws.emit_hcli_workunit(
        id="future.qualification.protected-lease-request",
        role="science",
        description=(
            "REQUEST that Codex honour an existing HCLI protected lease for the "
            "survivor set. This unit cannot create, steal, or flock the lock."
        ),
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.qualification.lease_request",
        provider="future.qualification_pipeline",
        effect_class="READ_ONLY",
        status="blocked",
        classification="STATIC_ONLY",
        extras={
            "species": "accelerator_candidate_qualification",
            "blocked_reason": "sidecar has no GPU authority; request, not seizure",
            "requires_quiescence": False,
            "claim_boundary": ws.SIDECAR_CLAIM_BOUNDARY,
        },
    )
    ws.validate_emitted_unit(lease_unit)
    units.append(lease_unit)
    units.sort(key=lambda r: str(r.get("id") or ""))
    compact = [
        {
            "id": u.get("id"),
            "status": u.get("status"),
            "classification": u.get("classification"),
            "resource_class": u.get("resource_class"),
            "candidate_id": u.get("candidate_id"),
            "blocked_reason": u.get("blocked_reason"),
            "requires_quiescence": u.get("requires_quiescence"),
            "claim_boundary": u.get("claim_boundary"),
        }
        for u in units
    ]
    return _authority(
        {
            "kind": "PROPOSAL",
            "n": len(units),
            "source": "tools.future.workunit_species.emit_hcli_workunit / _physical_unit",
            "does_not_schedule": True,
            "does_not_dispatch": True,
            "units": compact,
        }
    )


# ---------------------------------------------------------------------------
# Checkpoint / resume — fail closed on a broken or partial snapshot
# ---------------------------------------------------------------------------


def make_checkpoint(
    *,
    completed: Sequence[str],
    payloads: Mapping[str, Any],
    ctx: Mapping[str, Any],
    in_progress_stage: str | None = None,
) -> dict[str, Any]:
    body = {
        "schema": CHECKPOINT_SCHEMA,
        "completed_stage_ids": list(completed),
        "stage_payloads": deepcopy(dict(payloads)),
        "in_progress_stage": in_progress_stage,
        "ctx": {
            "lease": ctx.get("lease"),
            "contamination_class": ctx.get("contamination_class"),
            "dry_run": ctx.get("dry_run"),
            "queue": ctx.get("queue"),
            "plan": ctx.get("plan"),
            "snap": ctx.get("snap"),
            "preflight": ctx.get("preflight"),
            "selected": ctx.get("selected"),
            "survivors": ctx.get("survivors"),
            "dropped": ctx.get("dropped"),
        },
    }
    return rs.seal_doc(body)


def admit_checkpoint(doc: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise rs.FailClosed("corrupt_receipt", "checkpoint is not an object")
    if not rs.seal_is_valid(dict(doc)):
        raise rs.FailClosed(
            "corrupt_receipt",
            "checkpoint seal does not match canonical body; a broken seal is not a resume point",
        )
    if doc.get("schema") != CHECKPOINT_SCHEMA:
        raise rs.FailClosed("stale_pipeline_cache", f"checkpoint schema {doc.get('schema')!r} is not {CHECKPOINT_SCHEMA}")
    completed = list(doc.get("completed_stage_ids") or [])
    if not all(name in STAGES for name in completed):
        raise rs.FailClosed("stale_pipeline_cache", f"checkpoint names unknown stages: {completed}")
    prefix = list(STAGES[: len(completed)])
    if completed != prefix:
        raise rs.FailClosed(
            "stale_pipeline_cache",
            "completed_stage_ids is not a prefix of STAGES; refusing rather than skipping a hole",
        )
    in_progress = doc.get("in_progress_stage")
    if in_progress:
        if in_progress in completed:
            raise rs.FailClosed(
                "partial_result",
                f"stage {in_progress!r} is both completed and in_progress; partial is not a result",
            )
        if in_progress not in STAGES:
            raise rs.FailClosed("stale_pipeline_cache", f"in_progress_stage {in_progress!r} is not a pipeline stage")
    return dict(doc)


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------


def _stage_record(
    name: str,
    index: int,
    status: str,
    payload: Mapping[str, Any],
    *,
    reason: str | None = None,
    execution_ok: bool | None = None,
) -> dict[str, Any]:
    exec_ok = payload.get("execution_ok") if execution_ok is None else execution_ok
    if exec_ok is None:
        exec_ok = True
    rec = {
        "index": index,
        "name": name,
        "status": status,
        "execution_ok": bool(exec_ok),
        "payload": dict(payload),
    }
    if reason is not None:
        rec["reason"] = reason
    return rec


def execution_stop(stages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """First stage that would block execute(), in pipeline order.

    On this sidecar machine that is the lease check: no existing HCLI lease.
    Later stages still emit specs; they do not run.
    """
    for rec in stages:
        if not rec.get("execution_ok", True):
            payload = rec.get("payload") or {}
            return {
                "stage_index": rec["index"],
                "stage_id": rec["name"],
                "status": rec.get("status"),
                "reason": rec.get("reason") or payload.get("reason") or "execution_ok is false",
            }
    return {
        "stage_index": 10,
        "stage_id": "protected_lease_request",
        "status": "REQUEST_EMITTED",
        "reason": (
            "planning walk found no earlier blocker, but stages 10-12 emit a "
            "request/spec and stop: this sidecar has no GPU authority"
        ),
    }


def _run_one_stage(name: str, ctx: dict[str, Any]) -> dict[str, Any]:
    if name == "identify_lease_availability":
        lease = ctx.get("lease")
        if not isinstance(lease, Mapping):
            lease = read_hcli_lease_state()
            ctx["lease"] = lease
        return dict(lease)
    if name == "assess_machine_quiescence":
        snap = ctx.get("snap")
        klass = ctx.get("klass")
        if not isinstance(snap, Mapping) or not isinstance(klass, Mapping):
            snap, klass = assess_quiescence(None)
            ctx["snap"] = snap
            ctx["klass"] = klass
        cclass = str(klass.get("contamination_class") or "UNKNOWN")
        ctx["contamination_class"] = cclass
        quiet = cclass == "QUIESCENT"
        return _authority(
            {
                "contamination_class": cclass,
                "contamination_reason": klass.get("contamination_reason"),
                "required_probes": klass.get("required_probes"),
                "n_competing_workloads": len(list(snap.get("competing_workloads") or [])),
                "resident_local_model": snap.get("resident_local_model"),
                "source": "tools.future.contamination.snapshot + classify_contamination",
                "execution_ok": quiet,
                "reason": (
                    None
                    if quiet
                    else (
                        f"machine is {cclass}; "
                        "queue_policy.protected_start_requires_machine_quiescence. "
                        "sidecar will not quiesce a worker to make it so"
                    )
                ),
            }
        )
    if name == "identify_contaminating_worker":
        snap = ctx.get("snap") or {}
        competing = list(snap.get("competing_workloads") or [])
        workers = [classify_worker(p) for p in competing]
        top = workers[0] if workers else None
        return _authority(
            {
                "n_over_threshold": len(workers),
                "workers": workers[:12],
                "primary": top,
                "pausable_patterns": list(PAUSABLE_PATTERNS),
                "sidecar_will_quiesce_any": False,
                "policy_source": "tools/odyssey/gpu_cleanliness.py PAUSE_PATTERN (recovered, not imported)",
                "reason": (
                    "no contaminating worker over threshold"
                    if top is None
                    else (
                        f"primary={top.get('name')!r} pid={top.get('pid')} class={top.get('class')}; "
                        f"policy_would_permit_quiesce={top.get('policy_would_permit_quiesce')}; "
                        "sidecar_will_quiesce=false"
                    )
                ),
            }
        )
    if name == "select_ready_candidates":
        selected = select_ready_candidates(ctx["queue"], ctx["plan"])
        ctx["selected"] = list(selected.get("candidates") or [])
        out = dict(selected)
        out.pop("candidates", None)
        out["candidate_rows_held_in_context"] = True
        return out
    if name == "static_preflight_drop":
        preflight = ctx.get("preflight")
        if not isinstance(preflight, Mapping):
            preflight = run_static_preflight()
            ctx["preflight"] = preflight
        dropped = drop_blocking_candidates(ctx.get("selected") or [], preflight)
        ctx["survivors"] = list(dropped.get("survivors") or [])
        ctx["dropped"] = list(dropped.get("dropped") or [])
        out = dict(dropped)
        out.pop("survivors", None)
        return out
    if name == "emit_ab_execution_plan":
        return emit_ab_plan(ctx["plan"], ctx.get("survivors") or [])
    if name == "parity_verification_spec":
        return parity_spec(ctx.get("survivors") or [], ctx["queue"])
    if name == "failure_classification":
        return failure_classification_spec(ctx["queue"])
    if name == "survivor_promotion_prerequisites":
        return promotion_prerequisites_spec(
            ctx["queue"],
            ctx.get("survivors") or [],
            str(ctx.get("contamination_class") or "UNKNOWN"),
        )
    if name == "protected_lease_request":
        payload = lease_request_spec(ctx.get("lease") or {})
        payload["execution_ok"] = False
        return payload
    if name == "protected_measurement_spec":
        payload = protected_measurement_spec(ctx["queue"], ctx.get("survivors") or [])
        payload["execution_ok"] = False
        return payload
    if name == "scoreboard_update_spec":
        payload = scoreboard_update_spec(ctx.get("survivors") or [])
        payload["execution_ok"] = False
        return payload
    if name == "derive_next_workunits":
        lease = ctx.get("lease") or {}
        return derive_next_workunits(
            ctx.get("survivors") or [],
            ctx.get("dropped") or [],
            bool(lease.get("present")),
        )
    raise rs.FailClosed("stale_pipeline_cache", f"unknown stage {name!r}")


_STAGE_STATUS = {
    "protected_lease_request": "REQUEST_EMITTED",
    "protected_measurement_spec": "SPEC_EMITTED",
    "scoreboard_update_spec": "SPEC_EMITTED",
    "derive_next_workunits": "PROPOSED",
}


def run_pipeline(
    *,
    dry_run: bool = True,
    live: bool = False,
    queue: Mapping[str, Any] | None = None,
    plan: Mapping[str, Any] | None = None,
    snap: Mapping[str, Any] | None = None,
    klass: Mapping[str, Any] | None = None,
    preflight: Mapping[str, Any] | None = None,
    lease: Mapping[str, Any] | None = None,
    interrupt_after: str | None = None,
    resume_from: Mapping[str, Any] | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Walk all 13 stages as STATIC_ONLY planning. Never starts a GPU run.

    live=True reads the real queue, contamination snapshot, HCLI lock path and
    static preflight. Tests inject the rest. interrupt_after raises
    PipelineInterrupted after sealing a resume checkpoint. resume_from skips
    already-completed stages.
    """
    completed: list[str] = []
    payloads: dict[str, Any] = {}
    records: list[dict[str, Any]] = []

    if resume_from is not None:
        ck = admit_checkpoint(resume_from)
        completed = list(ck["completed_stage_ids"])
        payloads = deepcopy(ck.get("stage_payloads") or {})
        restored = ck.get("ctx") or {}
        # Resume restores the sealed context. Passing a different lease/queue
        # must not recompute completed stages (that would restart the run).
        ctx: dict[str, Any] = {
            "dry_run": bool(dry_run),
            "lease": restored.get("lease") if restored.get("lease") is not None else lease,
            "queue": restored.get("queue") if restored.get("queue") is not None else queue,
            "plan": restored.get("plan") if restored.get("plan") is not None else plan,
            "snap": restored.get("snap") if restored.get("snap") is not None else snap,
            "preflight": restored.get("preflight") if restored.get("preflight") is not None else preflight,
            "contamination_class": restored.get("contamination_class"),
            "selected": list(restored.get("selected") or []),
            "survivors": list(restored.get("survivors") or []),
            "dropped": list(restored.get("dropped") or []),
        }
        if klass is not None:
            ctx["klass"] = klass
        elif restored.get("snap") is not None:
            ctx["klass"] = C.classify_contamination(restored["snap"])
        in_progress = ck.get("in_progress_stage")
        if in_progress:
            # Partial result of the interrupted stage is discarded. Resume from last completed.
            ctx["discarded_partial_stage"] = in_progress
        for name in completed:
            payload = payloads[name]
            index = STAGES.index(name) + 1
            records.append(
                _stage_record(
                    name,
                    index,
                    "RESUMED",
                    payload,
                    reason="restored from checkpoint; not recomputed",
                )
            )
    else:
        ctx = {"dry_run": bool(dry_run)}
        if queue is None:
            if not live:
                raise rs.FailClosed(
                    "incomplete_replication_bundle",
                    "run_pipeline(live=False) requires an injected queue; refusing to guess",
                )
            queue = load_qualification_queue()
        ctx["queue"] = queue
        ctx["plan"] = plan if plan is not None else load_staged_plan(queue)
        if lease is not None:
            ctx["lease"] = lease
        elif live:
            ctx["lease"] = read_hcli_lease_state()
        if snap is not None:
            ctx["snap"] = snap
        if klass is not None:
            ctx["klass"] = klass
        elif snap is not None:
            ctx["klass"] = C.classify_contamination(snap)
        if preflight is not None:
            ctx["preflight"] = preflight
        if interrupt_after is not None and interrupt_after not in STAGES:
            raise rs.FailClosed("stale_pipeline_cache", f"interrupt_after {interrupt_after!r} is not a stage")

    for index, name in enumerate(STAGES, start=1):
        if name in completed:
            continue
        if on_stage is not None:
            on_stage(name)
        payload = _run_one_stage(name, ctx)
        status = _STAGE_STATUS.get(name, "COMPLETED")
        rec = _stage_record(name, index, status, payload, reason=payload.get("reason"))
        records.append(rec)
        payloads[name] = payload
        completed.append(name)
        if interrupt_after is not None and name == interrupt_after:
            ck = make_checkpoint(
                completed=completed,
                payloads=payloads,
                ctx=ctx,
                in_progress_stage=None,
            )
            raise PipelineInterrupted(name, ck)

    stop = execution_stop(records)
    return {
        "schema": SCHEMA,
        "dry_run": bool(dry_run),
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "n_stages": len(STAGES),
        "stages": records,
        "execution_stop": stop,
        "planning_walk_complete": len(records) == len(STAGES),
        "resumed_from_stage_count": sum(1 for r in records if r.get("status") == "RESUMED"),
        "contamination_class": ctx.get("contamination_class"),
        "lease_present": bool((ctx.get("lease") or {}).get("present")),
        "survivor_ids": [str(c.get("candidate_id")) for c in (ctx.get("survivors") or [])],
        "dropped_ids": [str(d.get("candidate_id")) for d in (ctx.get("dropped") or [])],
        "note": (
            "planning walk emits 13 STATIC_ONLY specs. execute() is a separate "
            "entry point and always raises on this sidecar: no GPU authority."
        ),
    }


def execute(
    *,
    explicit_execute: bool = False,
    lease: Mapping[str, Any] | None = None,
    contamination_class: str | None = None,
) -> None:
    """RAISE unless all three conditions hold — and then raise anyway.

    The three independent refusals (no --execute, no existing lease, machine
    not QUIESCENT) are checked in that order. Each test injects the other two
    as passing so the named guard is the one that fires. Even if all three
    pass, this sidecar has no GPU authority and must not seize the HCLI lock.
    """
    if not explicit_execute:
        raise ExecuteRefused(
            "explicit_execute",
            "explicit --execute was not passed; sidecar will not start a protected run",
        )
    lease_state = dict(lease) if lease is not None else read_hcli_lease_state()
    if not lease_state.get("present"):
        raise ExecuteRefused(
            "existing_lease",
            "no existing HCLI lease; queue_policy.protected_start_requires_existing_hcli_lease; "
            "sidecar will not create one",
        )
    klass = contamination_class
    if klass is None:
        _snap, klass_doc = assess_quiescence(None)
        klass = str(klass_doc.get("contamination_class") or "UNKNOWN")
    if klass != "QUIESCENT":
        raise ExecuteRefused(
            "machine_quiescence",
            f"machine is {klass}; queue_policy.protected_start_requires_machine_quiescence; "
            "sidecar will not quiesce a worker",
        )
    raise ExecuteRefused(
        "gpu_authority",
        "existing lease and quiescence and --execute are not sufficient: "
        "this sidecar has no GPU authority and must not seize the HCLI lock. "
        "stages 10-12 emit a request/spec and stop",
    )


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def _frontier_f002() -> dict[str, Any] | None:
    if not FRONTIER_PATH.is_file():
        return None
    doc = load_json(FRONTIER_PATH)
    for entry in doc.get("entries") or []:
        if entry.get("id") == "F002":
            return {
                "id": entry.get("id"),
                "title": entry.get("title"),
                "classification": entry.get("classification"),
                "prerequisite": entry.get("prerequisite"),
                "integration_target": entry.get("integration_target"),
            }
    return None


def recovered_implementation() -> list[dict[str, Any]]:
    return [
        {
            "path": "tools/future/candidate_planner.py",
            "role": "staged factorial plan (34 cells on the live queue); READY_PROTECTED independent set; promotion_table",
            "composed_as": "load_staged_plan / select_ready_candidates / emit_ab_plan / promotion_prerequisites_spec",
            "adequate_for": "planning graph, cells, conflicts, scars",
            "not_adequate_for": "sequencing preflight + quiescence + lease into a resumable pipeline",
        },
        {
            "path": "tools/future/static_kernel_verify.py",
            "role": "zero-GPU host/shader ABI preflight",
            "composed_as": "run_static_preflight / drop_blocking_candidates",
            "adequate_for": "ERROR/WARNING/UNVERIFIABLE findings; would_waste_a_protected_window",
            "not_adequate_for": "mapping findings onto queue rows and dropping them from a run plan",
        },
        {
            "path": "tools/future/contamination.py",
            "role": "snapshot, QUIESCENT/LIGHT/HEAVY/UNKNOWN, paired_ab_stats, assert_promotable",
            "composed_as": "assess_quiescence / emit_ab_plan protocol / promotion gate",
            "adequate_for": "machine-state record and the promotion refusal",
            "not_adequate_for": "identifying whether policy would permit quiescing a neighbour, or sequencing a run",
        },
        {
            "path": "tools/future/workunit_species.py",
            "role": "HCLI WorkUnit species and starting queue",
            "composed_as": "derive_next_workunits",
            "adequate_for": "emitting HCLI-shaped proposals with bounded authority",
            "not_adequate_for": "deriving the *next* units from a preflight/lease outcome",
        },
        {
            "path": "tools/future/repro_science.py",
            "role": "fail-closed faults, checkpoint/resume, seal_is_valid, killed_subprocess",
            "composed_as": "ExecuteRefused/PipelineInterrupted inherit FailClosed; checkpoint seal; interrupt/resume",
        },
        {
            "path": "hcli/agentos/protected_accelerator_benchmark.py",
            "role": "the real protected lease (LOCK_NAME, _try_lock, run_protected_accelerator_benchmark)",
            "composed_as": "lock path cited read-only; runner NOT imported and NOT called",
            "on_disk_in_this_worktree": (REPO / "hcli/agentos/protected_accelerator_benchmark.py").is_file(),
        },
        {
            "path": "hcli/agentos/benchmark_boundary.py",
            "role": "QUALIFIED_PROTECTED vs DIAGNOSTIC_CONTAMINATED",
            "composed_as": "classify_window(None, None, UNKNOWN) in the measurement spec; empty window is not qualified",
        },
        {
            "path": "hcli/agentos/native_mission_gate.py",
            "role": "live native tool/verifier mission",
            "composed_as": "named, not imported, not run",
        },
        {
            "path": "receipts/future/evidence/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
            "role": "binding queue_policy.protected_start_requires_existing_hcli_lease and protected_start_requires_machine_quiescence",
        },
        {
            "path": "tools/odyssey/gpu_cleanliness.py",
            "role": "G013 PAUSABLE vs STANDING; recovered PAUSE_PATTERN",
            "composed_as": "classify_worker; sidecar never pauses",
            "on_disk_in_this_worktree": (REPO / "tools/odyssey/gpu_cleanliness.py").is_file(),
        },
        {
            "path": "lab/lease.py",
            "role": "campaign SingletonLease (different object from the HCLI protected bench lock)",
            "composed_as": "not imported; not used",
        },
        {
            "path": "tools/future/device_ascension_pipeline.py",
            "role": "arrival pipeline for a new machine; different object (8 stages, law downgrade)",
            "adequate_for": "sibling shape (dry-run, stage records, UNAVAILABLE measurement)",
            "not_adequate_for": "physical qualification sequencing of the Qwen27 READY_PROTECTED set",
        },
        {
            "path": "tools/accelerator/physical_qualification.py",
            "role": "Codex producer of the live queue",
            "on_disk_in_this_worktree": (REPO / "tools/accelerator/physical_qualification.py").is_file(),
            "adequate_as": "plan-first queue builder and WorkUnit emitter (Codex)",
            "not_adequate_as": "sidecar sequencer; writing it is prohibited",
        },
        {
            "path": "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
            "role": "F002: 12 Qwen27 candidates READY_PROTECTED idle on a GPU window the sidecar must not seize",
        },
    ]


def gaps_closed() -> list[str]:
    return [
        "13-stage ordered pipeline that composes planner, preflight, contamination, workunit species, HCLI boundary",
        "authority boundary: execute() raises on missing --execute, missing HCLI lease, and non-QUIESCENT separately, and still raises when all three pass",
        "no stage starts a benchmark, creates a lease, signals a process, or quiesces a worker — refuse_* functions exist so the guard can be watched to fail",
        "resumability: sealed checkpoint, interrupt_after fault injection, resume from last completed stage, corrupt/partial checkpoints fail closed",
        "--dry-run walks all 13 stages against the real queue and real machine and reports the execution stop (lease check on this host)",
        "static preflight ERRORs DROP mapped candidates rather than spending a protected window on a source-detectable defect",
        "stages 10-12 emit REQUEST/SPEC only; scoreboard and HCLI lock are not written",
        "next WorkUnits derived from survivors + drops + a blocked lease-request proposal",
    ]


def negative_findings() -> list[str]:
    return [
        "no .hcli/locks/protected-accelerator-bench.lock in this worktree; lease present is fail-closed false",
        "exclusive flock is never used to inspect the lock, because taking LOCK_EX would be a seizure; lsof is the only holder probe",
        "this sidecar has no GPU authority and cannot produce DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE",
        "tools/accelerator/physical_qualification.py is not materialized in this sparse checkout and is not imported",
        "tools/odyssey/gpu_cleanliness.py is not materialized; PAUSE_PATTERN recovered via git show, not imported",
        "lab/lease.py is a different campaign lease and is not the HCLI protected-accelerator lock",
        "PID-level GPU attribution is unavailable without a protected lease (contamination already records this)",
        "no qualification_pipeline.py existed before this lane; device_ascension_pipeline.py is a different pipe (machine arrival)",
        "native_mission_gate.py is not imported: it would start a live native mission",
        "cannot update receipts/headless/ACCELERATOR_SCOREBOARD.json (Codex surface)",
    ]


def build(pipeline: Mapping[str, Any] | None = None) -> Path:
    result = dict(pipeline) if pipeline is not None else run_pipeline(dry_run=True, live=True)
    stop = result.get("execution_stop") or {}
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Sequence the existing qualification pieces into one resumable pipeline "
            "that is structurally incapable of taking GPU authority. FIVE ERAS, "
            "THREE ODYSSEYS. FPGA stays inside Accelerator / Physical Compiler / "
            "Fusion. DISK STATE IS AUTHORITY."
        ),
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "no_era_vi": True,
        "no_odyssey_iv": True,
        "fpga": (
            "FPGA is part of Accelerator / Physical Compiler / Fusion. It is not "
            "its own civilization and this module does not build an FPGA backend."
        ),
        "vocabulary": {
            "DIAGNOSTIC_RELATIVE": "contaminated A/B on a busy machine. Guides. Never promotes.",
            "PROTECTED_ABSOLUTE": "measurement taken under a real protected GPU lease. Decides.",
            "STATIC_ONLY": "this sidecar. No GPU. Bench state UNKNOWN. Cannot promote.",
        },
        "stages_declared": list(STAGES),
        "n_stages": len(STAGES),
        "authority_boundary": {
            "execute_requires": [
                "explicit --execute",
                "existing HCLI lease (read, never created)",
                "machine QUIESCENT (assessed, never coerced)",
            ],
            "even_then": "sidecar has no GPU authority and still raises",
            "stages_10_12": "REQUEST/SPEC only",
            "refuse_functions": [
                "refuse_start_benchmark",
                "refuse_create_lease",
                "refuse_signal_process",
                "refuse_quiesce_worker",
            ],
        },
        "queue_policy_binding": {
            "protected_start_requires_existing_hcli_lease": True,
            "protected_start_requires_machine_quiescence": True,
            "diagnostic_results_do_not_promote": True,
            "source": "receipts/future/evidence/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json queue_policy",
        },
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "frontier_entry": _frontier_f002(),
        "pipeline": result,
        "dry_run_stop": {
            "stage_index": stop.get("stage_index"),
            "stage_id": stop.get("stage_id"),
            "reason": stop.get("reason"),
            "honest_on_this_machine": (
                "stops at the lease check: no existing HCLI lease. later stages "
                "are still walked as STATIC_ONLY specs so a human does not have "
                "to assemble the run by hand when a window opens"
            ),
        },
        "integration": {
            "run_pipeline": (
                "run_pipeline(*, dry_run=True, live=False, queue=None, plan=None, "
                "snap=None, klass=None, preflight=None, lease=None, "
                "interrupt_after=None, resume_from=None, on_stage=None) -> dict"
            ),
            "execute": (
                "execute(*, explicit_execute=False, lease=None, contamination_class=None) -> None  "
                "# always raises ExecuteRefused"
            ),
            "read_hcli_lease_state": "read_hcli_lease_state(repo=None) -> dict  # present True only with a proven holder; never flock",
            "admit_checkpoint": "admit_checkpoint(doc) -> dict  # raises FailClosed on corrupt/partial",
        },
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


selftest = build


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("The planning", 1)[0])
    ap.add_argument("--dry-run", action="store_true", help="walk all 13 stages; report where execute() would stop")
    ap.add_argument("--build", action="store_true", help="emit the sealed receipt")
    ap.add_argument("--selftest", action="store_true", help="alias of --build")
    ap.add_argument("--execute", action="store_true", help="attempt execute(); must refuse on this sidecar")
    args = ap.parse_args()
    if args.execute:
        try:
            execute(explicit_execute=True)
        except ExecuteRefused as exc:
            print(f"execute refused [{exc.fault}]: {exc.reason}")
            return 2
        print("execute returned without raising — that is a campaign-level failure")
        return 1
    result = run_pipeline(dry_run=True, live=True)
    out = build(pipeline=result)
    summary = {
        "dry_run": True,
        "execution_stop": result.get("execution_stop"),
        "planning_walk_complete": result.get("planning_walk_complete"),
        "contamination_class": result.get("contamination_class"),
        "lease_present": result.get("lease_present"),
        "survivor_ids": result.get("survivor_ids"),
        "dropped_ids": result.get("dropped_ids"),
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "stages": [
            {
                "index": rec["index"],
                "name": rec["name"],
                "status": rec["status"],
                "execution_ok": rec.get("execution_ok"),
            }
            for rec in result.get("stages") or []
        ],
        "receipt": str(out),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    stop = result.get("execution_stop") or {}
    print(
        f"dry-run stop: stage {stop.get('stage_index')} {stop.get('stage_id')}: {stop.get('reason')}"
    )
    _ = args
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
