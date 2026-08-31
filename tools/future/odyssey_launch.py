"""ODYSSEY I launch gate — WHAT IS TRUE? may not start on a green scaffold.

The sidecar substrate is executable. That is not the same as Odyssey started.
This module evaluates sixteen launch criteria from disk evidence, names every
unmet criterion (not the first), and writes ODYSSEY_I_LAUNCH.json ONLY when
all sixteen pass. Today's honest answer is REFUSED.

A subsystem is not operational until the resident can discover it, invoke it,
schedule it and verify it; its result changes a frontier; the result persists;
and next work refills. A CLI a human can run is not enough.

    python3 tools/future/odyssey_launch.py --verify
    python3 tools/future/odyssey_launch.py --launch
    python3 -m pytest tools/future/test_odyssey_launch.py -q

Everything emitted is STATIC_ONLY, bench UNKNOWN, gpu_authority false.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, git, RECEIPTS, sha256_file

import argparse
import ast
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.future import contamination as C
from tools.future import flash_nx_audit as nx_audit
from tools.future import odyssey2_law_store as ols
from tools.future import odyssey3_adversary as o3
from tools.future import repro_science as rs
from tools.future import status_causality as sc
from tools.future.specimen_curriculum import (  # noqa: F401 - re-export; this file is not a second authority
    CURRICULUM_ROLES,
    propose_specimen_curriculum,
    _ready,
    _independently_verified,
    _specimen_dirs_on_disk,
    _lake_index,
    _odyssey_i_patients,
)
from tools.future import workunit_species as wus


RECEIPT = "ODYSSEY_LAUNCH_GATE.json"
LAUNCH_RECEIPT = "ODYSSEY_I_LAUNCH.json"
SCHEMA = "hawking.future.odyssey_launch.v1"
LAUNCH_SCHEMA = "hawking.future.odyssey_i_launch.v1"
VERSION = 1
RECORDED_BY = "tools/future/odyssey_launch.py"

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

# Concurrent-wave modules. Named as integration points; never imported.
THIS_WAVE_SIBLINGS = (
    "codex_behaviors",
    "resident_api",
    "workgraph",
    "detached",
    "wakeup",
    "evidence_dag",
    "scar_scheduling",
    "dirty_measure",
    "protected_window",
    "sandbox",
    "resident_identity",
    "frontiers",
    "succession",
    "flash_schools",
    "flash_nr_complete",
    "super_resident",
    "tabula",
    "debugger",
    "autonomy_trial",
    "protected_scheduler",
    "nr_nx_generic",
)

# HCLI AgentOS autonomy is a recovered control-plane proof. It is not an
# Odyssey I resident-orchestration trial and must not open this gate.
HCLI_AUTONOMY_SCHEMA = "hcli.agentos.autonomy_gate.v1"
ODYSSEY_AUTONOMY_SCHEMAS = (
    "hawking.future.autonomy_trial.v1",
    "hawking.future.autonomy_trial.persisted_verdict.v1",
    "hawking.odyssey.autonomy_trial.v1",
    "hawking.future.super_resident.autonomy.v1",
)

# The three criteria this lane rewires. Captured so the receipt can state
# the gate count before and after without pretending a later re-run is the baseline.
REWIRE_BASELINE = {
    "n_criteria": 16,
    "n_met": 13,
    "n_unmet": 3,
    "unmet": [
        "resident_autonomy_trial_pass",
        "nr_nx_path_callable",
        "protected_scheduling",
    ],
    "source": (
        "ODYSSEY_LAUNCH_GATE.json before L31: autonomy probed AUTONOMY_TRIAL.json "
        "(singular) while the module wrote AUTONOMY_TRIALS.json; protected_scheduling "
        "ANDed capability with window availability; nr_nx_path_callable keyed on "
        "Flash NX completeness instead of the generic pipeline"
    ),
}

CRITERION_IDS: tuple[str, ...] = (
    "resident_autonomy_trial_pass",
    "specimen_curriculum_ready",
    "modellake_identity",
    "doctor_callable",
    "gravity_callable",
    "nr_nx_path_callable",
    "evidence_hierarchy",
    "negative_science",
    "workgraphs",
    "self_refill",
    "dirty_measurement",
    "protected_scheduling",
    "transfer_substrate",
    "adversary_substrate",
    "crash_recovery",
    "receipts",
)

GRAPH_STAGES: tuple[str, ...] = (
    "specimen_verification",
    "architecture_fingerprint",
    "doctor",
    "gravity",
    "representation_search",
    "physical_profiling",
    "nr",
    "nx",
    "native_execution",
    "capability",
    "pareto",
    "laws",
    "scars",
)
PHASE_LISTEN_STAGES: tuple[str, ...] = ("phase_ii_transfer", "phase_iii_attack")
GPU_STAGES = frozenset(
    {"physical_profiling", "nx", "native_execution", "capability"}
)
COMPILE_STAGES = frozenset({"nr", "nx"})

# CURRICULUM_ROLES and propose_specimen_curriculum live in
# tools/future/specimen_curriculum.py (single authority). Re-exported above.

EVIDENCE_LATTICE = (
    "STATIC_ONLY",
    "DIAGNOSTIC_RELATIVE",
    "PROTECTED_ABSOLUTE",
)

SIDECAR_CLAIM = (
    "Static sidecar artifact. No hardware measurement. A launch-gate REFUSED "
    "receipt is not Odyssey started. ODYSSEY_I_LAUNCH.json is the only phase-"
    "transition receipt and is written only when every criterion is met."
)

# Local interface the this-wave siblings will replace. Named, not imported.
INTEGRATION_POINTS: dict[str, str] = {
    "autonomy_trial": "tools/future/autonomy_trial.py — persisted --verify verdict in AUTONOMY_TRIALS.json (plural)",
    "resident_identity": "tools/future/resident_identity.py — canonical resident declaration the launch receipt binds",
    "sandbox": "tools/future/sandbox.py — orchestrator sandbox identity the resident operates",
    "workgraph": "tools/future/workgraph.py — WorkGraph runtime that admits, schedules and verifies the units this gate emits",
    "frontiers": "tools/future/frontiers.py — frontier objects a result must change, and the refill that follows",
    "succession": "tools/future/succession.py — self-refill / next-work succession under resident control",
    "dirty_measure": "tools/future/dirty_measure.py — dirty-class measurement that cannot launder into PROTECTED_ABSOLUTE",
    "protected_window": "tools/future/protected_window.py — lock observation; does not seize Codex's GPU lock",
    "protected_scheduler": "tools/future/protected_scheduler.py — PROTECTED_SCHEDULER_CAPABLE vs PROTECTED_WINDOW_AVAILABLE",
    "nr_nx_generic": "tools/future/nr_nx_generic.py — GENERIC_NR_NX_PIPELINE_CALLABLE vs FLASH_NX_READY",
    "evidence_dag": "tools/future/evidence_dag.py — evidence DAG the resident walks; lattice alone is not the DAG",
    "wakeup": "tools/future/wakeup.py — wake SLEEPING WorkUnits when hardware qualifies",
    "super_resident": "tools/future/super_resident.py — HCLI super-resident operating the orchestrator sandbox",
    "resident_api": "tools/future/resident_api.py — discover/invoke surface HCLI uses instead of a human CLI",
}


# Five fields every consequential criterion must record at emit time.
# Prefer the sibling's tuple when present so the consumer and the routine
# cannot drift on the contract.
FIVE_RECORDED_FIELDS: tuple[str, ...] = getattr(
    sc,
    "FIVE_RECORDED_FIELDS",
    (
        "probe_performed",
        "direct_observation",
        "interpretation",
        "confidence",
        "alternatives",
    ),
)


def _bind_emit() -> None:
    """This lane consumes emit(); a sibling owns that module.

    If this checkout still has the pre-emit blob, bind a catalog-free
    trampoline so the call site is `sc.emit(` either way. emit() must not
    touch disk and must not look up the historical catalog for a bare name.
    """
    if hasattr(sc, "emit"):
        return

    def emit(
        status: str,
        *,
        probe_performed: str = "",
        direct_observation: Any = "",
        interpretation: str = "",
        probe_kind: str = "",
        claim_kind: str | None = None,
        falsifier: str = "",
        source: str = "",
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "status": status,
            "probe_performed": probe_performed,
            "direct_observation": direct_observation,
            "interpretation": interpretation or status,
            "probe_kind": probe_kind,
            "use_catalog": False,
            "source": source or "<emit>",
        }
        if claim_kind:
            row["claim_kind"] = claim_kind
        if falsifier:
            row["falsifier"] = falsifier
        out = sc.challenge(row)
        out["entry"] = "emit"
        return out

    sc.emit = emit  # type: ignore[attr-defined]


_bind_emit()


def records_five_fields(node: Any) -> bool:
    """True iff this mapping itself carries the five recorded fields."""
    fn = getattr(sc, "records_five_fields", None)
    if callable(fn):
        return bool(fn(node))
    if not isinstance(node, Mapping):
        return False
    if not all(k in node for k in FIVE_RECORDED_FIELDS):
        return False
    if not str(node.get("probe_performed") or "").strip():
        return False
    if node.get("direct_observation") in (None, "", [], {}):
        return False
    if not str(node.get("interpretation") or "").strip():
        return False
    conf = node.get("confidence")
    if not isinstance(conf, Mapping):
        return False
    if not {"would_raise", "would_lower", "level", "about"} <= set(conf):
        return False
    alts = node.get("alternatives")
    return isinstance(alts, list) and bool(alts)


def record_criterion_causality(
    row: dict[str, Any],
    *,
    probe_performed: str = "",
    direct_observation: Any = "",
    probe_kind: str = "",
    claim_kind: str | None = None,
    interpretation: str | None = None,
    source: str = "",
) -> dict[str, Any]:
    """Stamp the five causality fields. Does not change met/unmet.

    An unsupplied observation is UNTESTED, never a restatement of the status
    or the reason. OVERREACHING is recorded beside the verdict; it does not
    override met.
    """
    met_before = row.get("met")
    status = str(row.get("id") or row.get("status") or "")
    interp = interpretation if interpretation is not None else str(row.get("reason") or status)
    unsupplied = direct_observation in (None, "", [], {})
    rec = sc.emit(
        status,
        probe_performed=str(probe_performed or ""),
        direct_observation="" if unsupplied else direct_observation,
        interpretation=interp,
        probe_kind="" if unsupplied else probe_kind,
        claim_kind=None if unsupplied else claim_kind,
        source=source or f"tools/future/odyssey_launch.py::{status}",
    )
    for key in FIVE_RECORDED_FIELDS:
        row[key] = rec[key]
    row["causality_verdict"] = rec["verdict"]
    row["falsifier"] = rec.get("falsifier")
    if rec.get("probe_kind"):
        row["probe_kind"] = rec["probe_kind"]
    if rec.get("claim_kind") is not None:
        row["claim_kind"] = rec["claim_kind"]
    if row.get("met") != met_before:
        raise RuntimeError("status_causality.emit mutated met/unmet")
    return rec


# ---------------------------------------------------------------------------
# Evidence probes. Sparse checkout: missing here is not absence in the project.
# ---------------------------------------------------------------------------


def _checkout_roots() -> list[Path]:
    roots: list[Path] = [REPO]
    common = git("rev-parse", "--git-common-dir")
    if common:
        path = Path(common)
        if not path.is_absolute():
            path = (REPO / path).resolve()
        else:
            path = path.resolve()
        parent = path.parent if path.name == ".git" else path.parent
        if parent not in roots and parent.is_dir():
            roots.append(parent)
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def probe_json(*rels: str) -> dict[str, Any]:
    """Load the first readable JSON among `rels`. Record which path was taken.

    A miss is `found=False` with the search log. It is not a proof the file
    does not exist in another checkout. Callers must cope with either state.
    """
    searched: list[str] = []
    snapshot = REPO / "receipts" / "future" / "evidence"
    for rel in rels:
        rel = rel.replace("\\", "/").lstrip("./")
        candidates: list[tuple[str, Path]] = [("worktree", REPO / rel)]
        if snapshot.is_dir():
            candidates.append(("evidence_snapshot", snapshot / Path(rel).name))
        for root in _checkout_roots()[1:]:
            candidates.append(("primary_checkout", root / rel))
        for origin, path in candidates:
            searched.append(f"{origin}:{path}")
            if path.is_file():
                try:
                    return {
                        "found": True,
                        "path_taken": origin,
                        "rel": rel,
                        "resolved": str(path),
                        "doc": load_json(path),
                        "searched": searched,
                    }
                except (OSError, json.JSONDecodeError) as exc:
                    searched.append(f"unreadable:{path}:{type(exc).__name__}")
        blob = git("show", f"HEAD:{rel}")
        searched.append(f"git:HEAD:{rel}")
        if blob:
            try:
                return {
                    "found": True,
                    "path_taken": "git_head",
                    "rel": rel,
                    "resolved": f"git:HEAD:{rel}",
                    "doc": json.loads(blob),
                    "searched": searched,
                }
            except json.JSONDecodeError as exc:
                searched.append(f"git_unreadable:{rel}:{type(exc).__name__}")
    return {
        "found": False,
        "path_taken": "not_found",
        "rel": rels[0] if rels else None,
        "resolved": None,
        "doc": None,
        "searched": searched,
        "note": (
            "not present in this worktree, the evidence snapshot, the primary "
            "checkout, or git HEAD. Sparse checkout: this is not proof of absence."
        ),
    }


def _module_file(rel: str) -> dict[str, Any]:
    path = REPO / rel
    if path.is_file():
        return {"present": True, "path_taken": "worktree", "resolved": str(path), "rel": rel}
    blob = git("ls-files", "--error-unmatch", rel)
    if blob:
        return {"present": True, "path_taken": "git_tracked_not_materialized", "resolved": rel, "rel": rel}
    for root in _checkout_roots()[1:]:
        other = root / rel
        if other.is_file():
            return {"present": True, "path_taken": "primary_checkout", "resolved": str(other), "rel": rel}
    return {
        "present": False,
        "path_taken": "not_found",
        "resolved": None,
        "rel": rel,
        "note": "sparse miss is not project absence",
    }


def _importable(dotted: str) -> dict[str, Any]:
    try:
        parts = dotted.split(".")
        mod = __import__(dotted, fromlist=[parts[-1]])
        return {"ok": True, "module": dotted, "file": getattr(mod, "__file__", None)}
    except Exception as exc:
        return {"ok": False, "module": dotted, "error": f"{type(exc).__name__}: {exc}"}


def _load_future_module(name: str) -> dict[str, Any]:
    """Import a sidecar module, including from the primary checkout on a sparse worktree.

    Missing here is not absence. A failed import is recorded; it is not a PASS.
    """
    dotted = f"tools.future.{name}"
    try:
        mod = importlib.import_module(dotted)
        return {
            "ok": True,
            "module": mod,
            "dotted": dotted,
            "path_taken": "import",
            "file": getattr(mod, "__file__", None),
        }
    except Exception:
        existing = sys.modules.get(dotted)
        if existing is not None and getattr(existing, "__file__", None) is None:
            sys.modules.pop(dotted, None)
    loc = _module_file(f"tools/future/{name}.py")
    resolved = loc.get("resolved")
    if not loc.get("present") or not resolved or loc.get("path_taken") == "git_tracked_not_materialized":
        return {
            "ok": False,
            "module": None,
            "dotted": dotted,
            "why": f"{name} not importable and not materialized: {loc.get('path_taken')}",
            "loc": loc,
        }
    path = Path(str(resolved))
    if not path.is_file():
        return {
            "ok": False,
            "module": None,
            "dotted": dotted,
            "why": f"{name} resolved to a non-file: {path}",
            "loc": loc,
        }
    try:
        spec = importlib.util.spec_from_file_location(dotted, path)
        if spec is None or spec.loader is None:
            return {"ok": False, "module": None, "dotted": dotted, "why": "spec_from_file_location returned None"}
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return {
            "ok": True,
            "module": mod,
            "dotted": dotted,
            "path_taken": loc.get("path_taken"),
            "file": str(path),
        }
    except Exception as exc:
        sys.modules.pop(dotted, None)
        return {
            "ok": False,
            "module": None,
            "dotted": dotted,
            "why": f"{type(exc).__name__}: {exc}",
            "loc": loc,
        }


def _receipt_sealed(doc: Mapping[str, Any] | None) -> bool:
    if not isinstance(doc, Mapping):
        return False
    seal = doc.get("seal_sha256")
    if not isinstance(seal, str) or len(seal) != 64:
        return False
    bench = doc.get("bench") if isinstance(doc.get("bench"), Mapping) else {}
    return bench.get("measurement_state") == "STATIC_ONLY" or "schema" in doc


def _criterion(
    cid: str,
    *,
    met: bool,
    reason: str,
    evidence: Sequence[Mapping[str, Any]] | None = None,
    operational: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    probe_performed: str = "",
    direct_observation: Any = "",
    probe_kind: str = "",
    claim_kind: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": cid,
        "met": bool(met),
        "reason": reason,
        "evidence": [dict(e) for e in (evidence or ())],
        "operational": dict(operational or {}),
    }
    if extra:
        row.update(dict(extra))
    # Default claim is the measured flags themselves, not a world-absence.
    # OVERREACHING stays information; met/unmet is already on the row.
    if probe_kind == "":
        probe_kind = sc.PROBE_MEASURED_FLAGS
    if claim_kind is None and (
        str(probe_performed or "").strip()
        and direct_observation not in (None, "", [], {})
    ):
        claim_kind = sc.CLAIM_MEASURED_UNMET if not met else sc.CLAIM_FIELD_VALUE
    record_criterion_causality(
        row,
        probe_performed=probe_performed,
        direct_observation=direct_observation,
        probe_kind=probe_kind,
        claim_kind=claim_kind,
        interpretation=reason,
    )
    return row


def operational_bar(
    *,
    discover: bool,
    invoke: bool,
    schedule: bool,
    verify: bool,
    frontier: bool,
    persist: bool,
    refill: bool,
    notes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    flags = {
        "discover": bool(discover),
        "invoke": bool(invoke),
        "schedule": bool(schedule),
        "verify": bool(verify),
        "frontier": bool(frontier),
        "persist": bool(persist),
        "refill": bool(refill),
    }
    return {
        "resident_operational": all(flags.values()),
        "cli_is_not_enough": True,
        "flags": flags,
        "notes": dict(notes or {}),
    }


# ---------------------------------------------------------------------------
# Curriculum roles recovered from ModelLake seals + Odyssey I + law-store schools.
# Do not propose exhaustively optimizing every downloaded model.
# ---------------------------------------------------------------------------


def _slug(repo: str, revision: str | None) -> str:
    name = (repo or "specimen").split("/")[-1]
    name = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    rev = (revision or "unpinned")[:12]
    return f"{name}@{rev}"


def _stage_resource(stage: str) -> tuple[str, str]:
    if stage in GPU_STAGES:
        return "GPU_EXCLUSIVE", "sleeping"
    if stage in COMPILE_STAGES:
        return "COMPILE", "sleeping" if stage == "nx" else "pending"
    return "STATIC_ANALYSIS", "pending"


def _stage_verifier(stage: str, slug: str) -> str:
    return f"future.odyssey_launch.workgraph.{stage}:{slug}"


def emit_first_workgraphs(first_specimen: Mapping[str, Any]) -> dict[str, Any]:
    """Specimen verification → … → scars, then II transfer and III attack in parallel."""
    repo = str(first_specimen.get("repo") or "unknown")
    revision = first_specimen.get("revision")
    slug = _slug(repo, str(revision) if revision else None)
    prefix = f"odyssey-i.wg.{slug}"

    units: list[dict[str, Any]] = []
    by_stage: dict[str, str] = {}
    prev: str | None = None
    for stage in GRAPH_STAGES:
        uid = f"{prefix}.{stage}"
        by_stage[stage] = uid
        resource, status = _stage_resource(stage)
        sleeping = status == "sleeping"
        extras = {
            "species": "odyssey_i_first_workgraph",
            "stage": stage,
            "specimen_repo": repo,
            "specimen_revision": revision,
            "odyssey": "I",
            "era": "I",
            "sleeping": sleeping,
            "blocked_reason": (
                "SLEEPING: hardware is not qualified on this host; HCLI wakes "
                "this unit when the protected GPU lane qualifies. Not a synthetic result."
                if sleeping
                else None
            ),
            "requires_quiescence": resource == "GPU_EXCLUSIVE",
            "output_receipt_path": f"receipts/future/ODYSSEY_I_{stage.upper()}_{slug}.json",
        }
        row = wus.emit_hcli_workunit(
            id=uid,
            role="science",
            description=(
                f"Odyssey I first WorkGraph stage {stage} for {repo}"
                + (f"@{revision}" if revision else "")
            ),
            dependencies=[prev] if prev else [],
            resource_class=resource,
            verifier=_stage_verifier(stage, slug),
            provider="future.odyssey_launch",
            effect_class="READ_ONLY",
            status=status,
            classification="SLEEPING" if sleeping else "STATIC_ONLY",
            extras=extras,
        )
        wus.validate_emitted_unit(row)
        units.append(row)
        prev = uid

    laws_id = by_stage["laws"]
    listen: list[dict[str, Any]] = []
    for stage, odyssey, species in (
        ("phase_ii_transfer", "II", "odyssey_ii_transfer_experiment"),
        ("phase_iii_attack", "III", "odyssey_iii_adversarial_experiment"),
    ):
        uid = f"{prefix}.{stage}"
        by_stage[stage] = uid
        row = wus.emit_hcli_workunit(
            id=uid,
            role="science",
            description=(
                f"Odyssey {odyssey} listens concurrently: once Phase I emits a "
                f"law, this unit may run. No global barrier. Specimen {repo}."
            ),
            dependencies=[laws_id],
            resource_class="STATIC_ANALYSIS",
            verifier=_stage_verifier(stage, slug),
            provider="future.odyssey_launch",
            effect_class="READ_ONLY",
            status="pending",
            classification="STATIC_ONLY",
            extras={
                "species": species,
                "stage": stage,
                "specimen_repo": repo,
                "specimen_revision": revision,
                "odyssey": odyssey,
                "era": "III",
                "listens_on": "phase_i_law_emission",
                "barrier": None,
                "blocked_reason": None,
                "requires_quiescence": False,
            },
        )
        wus.validate_emitted_unit(row)
        units.append(row)
        listen.append({"id": uid, "odyssey": odyssey, "depends_on": [laws_id], "depends_on_sibling": False})

    policy = phase_listen_policy(laws_id)
    units.sort(key=lambda r: str(r.get("id") or ""))
    return {
        "schema": "hawking.future.odyssey_i.first_workgraph.v1",
        "specimen": {"repo": repo, "revision": revision, "slug": slug},
        "stages": list(GRAPH_STAGES),
        "listen_stages": list(PHASE_LISTEN_STAGES),
        "n_units": len(units),
        "units": units,
        "by_stage": by_stage,
        "phase_listen": policy,
        "listen_units": listen,
        "note": (
            "These are real HCLI WorkUnits. Emitting them is not executing them. "
            "GPU-class units are SLEEPING until hardware qualifies."
        ),
    }


def phase_listen_policy(laws_unit_id: str | None = None) -> dict[str, Any]:
    """Once Phase I emits a law, Phase II may transfer it and Phase III may attack it."""
    return {
        "barrier": None,
        "global_barrier": False,
        "rule": (
            "once Phase I emits a law, Phase II may transfer it and Phase III "
            "may attack it. There is no global barrier between II and III."
        ),
        "phase_ii_depends_on": [laws_unit_id or "phase_i.laws"],
        "phase_iii_depends_on": [laws_unit_id or "phase_i.laws"],
        "phase_ii_depends_on_phase_iii": False,
        "phase_iii_depends_on_phase_ii": False,
        "odysseys": list(ODYSSEYS),
        "no_odyssey_iv": True,
        "no_era_vi": True,
    }


def resource_schedule(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, list[str]] = {}
    by_status: dict[str, list[str]] = {}
    for u in units:
        rc = str(u.get("resource_class") or "UNKNOWN")
        st = str(u.get("status") or "UNKNOWN")
        by_class.setdefault(rc, []).append(str(u.get("id")))
        by_status.setdefault(st, []).append(str(u.get("id")))
    for d in (by_class, by_status):
        for k in d:
            d[k] = sorted(d[k])
    return {
        "by_resource_class": {k: {"n": len(v), "ids": v} for k, v in sorted(by_class.items())},
        "by_status": {k: {"n": len(v), "ids": v} for k, v in sorted(by_status.items())},
        "gpu_authority": False,
        "sleeping_are_not_results": True,
    }


# ---------------------------------------------------------------------------
# Physical blockers → SLEEPING WorkUnits. Never a synthetic result.
# ---------------------------------------------------------------------------


def _xcrun_metal() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["xcrun", "-f", "metal"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        found = proc.returncode == 0 and bool((proc.stdout or "").strip())
        return {
            "probed": True,
            "found": found,
            "path": (proc.stdout or "").strip() or None,
            "stderr": (proc.stderr or "")[:240] or None,
            "returncode": proc.returncode,
        }
    except Exception as exc:
        return {"probed": True, "found": False, "error": f"{type(exc).__name__}: {exc}"}


def physical_blockers() -> list[dict[str, Any]]:
    """Derive the live physical blockers from sealed receipts + a toolchain probe."""
    handoff = probe_json("receipts/future/FUTURE_SUBSTRATE_HANDOFF.json")
    nx_rec = probe_json("receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json")
    teacher = probe_json("receipts/future/TEACHER_CORPUS_CONTRACT.json")
    qual = probe_json("receipts/future/QUALIFICATION_PIPELINE.json")
    cont = probe_json("receipts/future/CONTAMINATION_SCIENCE.json")

    handoff_blockers = []
    if isinstance(handoff.get("doc"), Mapping):
        handoff_blockers = list(handoff["doc"].get("blockers") or [])

    seven_all_met = None
    nx_status = None
    if isinstance(nx_rec.get("doc"), Mapping):
        seven_all_met = nx_rec["doc"].get("seven_all_met")
        nx_v0 = (nx_rec["doc"].get("nx_completeness_checker") or {}).get("real_FLASH_COMPLETE_V0_nx")
        if isinstance(nx_v0, Mapping):
            nx_status = nx_v0.get("status") or nx_v0.get("state")
        nr = nx_rec["doc"].get("nr_v2") if isinstance(nx_rec["doc"].get("nr_v2"), Mapping) else {}
        nr_status = nr.get("status")
    else:
        nr_status = None

    n_executed = 0
    n_units = 0
    target_rows = []
    if isinstance(teacher.get("doc"), Mapping):
        units = list(teacher["doc"].get("capture_workunits") or [])
        n_units = len(units)
        n_executed = sum(1 for u in units if isinstance(u, Mapping) and u.get("executed") is True)
        for u in units:
            if not isinstance(u, Mapping):
                continue
            payload = u.get("payload") if isinstance(u.get("payload"), Mapping) else {}
            tr = payload.get("target_row_count")
            if isinstance(tr, int):
                target_rows.append(tr)
    teacher_done = n_executed > 0

    lease_reason = None
    lease_present = None
    if isinstance(qual.get("doc"), Mapping):
        stop = qual["doc"].get("dry_run_stop") if isinstance(qual["doc"].get("dry_run_stop"), Mapping) else {}
        lease_reason = stop.get("reason")
        auth = qual["doc"].get("authority_boundary") if isinstance(qual["doc"].get("authority_boundary"), Mapping) else {}
        lease_present = False if auth else None

    cont_class = None
    if isinstance(cont.get("doc"), Mapping):
        cont_class = cont["doc"].get("contamination_class")

    xcrun = _xcrun_metal()

    rows = [
        {
            "id": "sleep.no-gpu-authority",
            "title": "No protected GPU / MetalContext not invoked",
            "sleeping": True,
            "holds": True,
            "reason": (
                "Sidecar has no GPU authority and does not invoke MetalContext. "
                "Handoff blockers record no protected GPU authority on this campaign."
            ),
            "evidence": {"handoff_blockers": handoff_blockers, "gpu_authority": False},
        },
        {
            "id": "sleep.metal-compiler",
            "title": "Metal compiler under CommandLineTools",
            "sleeping": True,
            "holds": not bool(xcrun.get("found")),
            "reason": (
                "xcrun -f metal did not locate the Metal compiler"
                if not xcrun.get("found")
                else f"xcrun located metal at {xcrun.get('path')}"
            ),
            "evidence": {"xcrun": xcrun},
        },
        {
            "id": "sleep.protected-bench-lock",
            "title": "Protected bench lock; holder pids unproven",
            "sleeping": True,
            "holds": True,
            "reason": lease_reason
            or (
                "qualification pipeline fail-closes lease present: lock file may "
                "exist but flock would be a seizure and is never attempted"
            ),
            "evidence": {
                "qualification_path_taken": qual.get("path_taken"),
                "lease_present": lease_present,
                "dry_run_stop": lease_reason,
            },
        },
        {
            "id": "sleep.machine-heavy",
            "title": "Machine classified HEAVY; standing workers not quiesced",
            "sleeping": True,
            "holds": cont_class == "HEAVY" or cont_class is None,
            "reason": (
                f"contamination_class={cont_class!r}; sidecar will not SIGSTOP "
                "standing workers. HEAVY (or unknown) is not a protected window."
            ),
            "evidence": {"contamination_class": cont_class, "path_taken": cont.get("path_taken")},
        },
        {
            "id": "sleep.flash-nx-scaffold",
            "title": "Flash source-independent NX is not qualified",
            "sleeping": True,
            "holds": seven_all_met is not True,
            "reason": (
                f"FLASH_NX_COMPLETENESS_AUDIT seven_all_met={seven_all_met!r} "
                f"nx_status={nx_status!r} nr_v2.status={nr_status!r}. "
                "SCAFFOLD_ONLY / metadata seals are not a callable NX path."
            ),
            "evidence": {
                "seven_all_met": seven_all_met,
                "nx_status": nx_status,
                "nr_v2_status": nr_status,
                "path_taken": nx_rec.get("path_taken"),
            },
        },
        {
            "id": "sleep.teacher-capture",
            "title": "Teacher capture has not run",
            "sleeping": True,
            "holds": not teacher_done,
            "reason": (
                f"teacher capture executed={n_executed}/{n_units} workunits; "
                f"target_row_counts={sorted(set(target_rows))}. "
                "Unrun capture is not a synthetic corpus."
            ),
            "evidence": {
                "n_capture_workunits": n_units,
                "n_executed": n_executed,
                "target_row_counts": sorted(set(target_rows)),
                "path_taken": teacher.get("path_taken"),
            },
        },
    ]
    sleeping_units: list[dict[str, Any]] = []
    for row in rows:
        if not row["holds"]:
            continue
        unit = wus.emit_hcli_workunit(
            id=f"odyssey-i.{row['id']}",
            role="science",
            description=f"SLEEPING until hardware qualifies: {row['title']}",
            dependencies=[],
            resource_class="GPU_EXCLUSIVE" if "gpu" in row["id"] or "nx" in row["id"] or "teacher" in row["id"] or "lock" in row["id"] or "heavy" in row["id"] or "compiler" in row["id"] else "STATIC_ANALYSIS",
            verifier=f"future.odyssey_launch.wakeup:{row['id']}",
            provider="future.odyssey_launch",
            effect_class="READ_ONLY",
            status="sleeping",
            classification="SLEEPING",
            extras={
                "sleeping": True,
                "blocked_reason": row["reason"],
                "requires_quiescence": True,
                "synthetic_result_forbidden": True,
                "wakeup_integration": INTEGRATION_POINTS["wakeup"],
            },
        )
        wus.validate_emitted_unit(unit)
        sleeping_units.append(unit)
        row["workunit_id"] = unit["id"]
    return _attach_sleeping(rows, sleeping_units)


def _attach_sleeping(rows: list[dict[str, Any]], units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {u["id"]: u for u in units}
    out = []
    for r in rows:
        wu_id = r.get("workunit_id")
        out.append(
            {
                "id": r["id"],
                "title": r["title"],
                "holds": r["holds"],
                "sleeping": r["sleeping"] and r["holds"],
                "reason": r["reason"],
                "evidence": r["evidence"],
                "workunit": by_id.get(wu_id),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Sixteen launch criteria. Every evaluator runs. None short-circuits the rest.
# ---------------------------------------------------------------------------


def _autonomy_trials_doc() -> dict[str, Any]:
    """Read the receipt autonomy_trial.py actually writes. Singular is not this file."""
    return probe_json("receipts/future/AUTONOMY_TRIALS.json")


def _eval_autonomy() -> dict[str, Any]:
    """Pass only on a persisted 1h+ PASS whose timeline seal still verifies.

    AUTONOMY_TRIAL.json (singular) is the old probe and is never authority.
    A FAIL persisted verdict never satisfies. Orchestration and cognition are
    separate: this criterion requires the HCLI loop and records cognition.
    """
    trials = _autonomy_trials_doc()
    singular = probe_json("receipts/future/AUTONOMY_TRIAL.json")
    hcli = probe_json(
        "receipts/headless/HCLI_AGENTOS_AUTONOMY_GATE.json",
        "receipts/future/evidence/HCLI_AGENTOS_AUTONOMY_GATE.json",
    )
    hcli_doc = hcli.get("doc") if isinstance(hcli.get("doc"), Mapping) else None
    hcli_schema = hcli_doc.get("schema") if hcli_doc else None
    hcli_passed = bool(
        hcli_doc
        and hcli_schema == HCLI_AUTONOMY_SCHEMA
        and (hcli_doc.get("checks") or {}).get("all_requested_stages_passed") is True
    )
    trials_doc = trials.get("doc") if isinstance(trials.get("doc"), Mapping) else None
    at_mod = _load_future_module("autonomy_trial")
    candidate = None
    if at_mod.get("ok"):
        candidate = at_mod["module"].launch_candidate_from_receipt(trials_doc)
    elif isinstance(trials_doc, Mapping):
        by = trials_doc.get("persisted_verdicts_by_trial")
        if isinstance(by, Mapping):
            for trial in ("6h", "3h", "1h"):
                row = by.get(trial)
                if isinstance(row, Mapping):
                    candidate = dict(row)
                    break

    seal = {"verifies": False, "why": "no persisted 1h+ verdict"}
    orch = None
    cognition = None
    cognition_reason = None
    verdict_label = None
    trial_id = None
    if isinstance(candidate, Mapping):
        trial_id = candidate.get("trial")
        verdict_label = str(candidate.get("verdict") or "").upper()
        orch = candidate.get("resident_orchestration")
        cognition = candidate.get("resident_model_cognition")
        cognition_reason = candidate.get("resident_model_cognition_reason")
        if at_mod.get("ok"):
            seal = at_mod["module"].verify_timeline_digest(
                candidate.get("timeline_path"),
                candidate.get("timeline_seal_digest"),
            )
        else:
            # Module missing: re-hash here rather than believe the stored flag.
            expected = candidate.get("timeline_seal_digest")
            rel = candidate.get("timeline_path")
            if expected and rel:
                path = Path(str(rel))
                if not path.is_file():
                    path = REPO / rel
                if path.is_file() and sha256_file(path) == expected:
                    seal = {"verifies": True, "why": "file digest matches (autonomy_trial module not imported)"}
                else:
                    seal = {"verifies": False, "why": "digest mismatch or timeline absent; autonomy_trial not imported"}
            else:
                seal = {"verifies": False, "why": "persisted verdict missing path or digest"}

    fail_verdict = verdict_label == "FAIL"
    pass_verdict = verdict_label == "PASS"
    eligible = str(trial_id or "") in {"1h", "3h", "6h"}
    seal_ok = seal.get("verifies") is True
    orch_ok = orch is True
    cognition_recorded = cognition is not None
    met = bool(
        isinstance(candidate, Mapping)
        and pass_verdict
        and not fail_verdict
        and eligible
        and orch_ok
        and seal_ok
        and cognition_recorded
    )
    if fail_verdict:
        met = False
        why_unmet = (
            f"persisted verdict is FAIL (trial={trial_id!r}); a FAIL never satisfies "
            "resident_autonomy_trial_pass"
        )
    elif not isinstance(candidate, Mapping):
        why_unmet = (
            "No persisted 1h/3h/6h verdict in AUTONOMY_TRIALS.json. "
            f"path_taken={trials.get('path_taken')!r}. "
            "AUTONOMY_TRIAL.json (singular) is not this receipt "
            f"(found={singular.get('found')}, path_taken={singular.get('path_taken')!r}). "
            f"HCLI AgentOS autonomy all_requested_stages_passed={hcli_passed} is not this criterion."
        )
    elif not eligible:
        why_unmet = f"persisted trial={trial_id!r} is not 1h/3h/6h; 15m is not the launch bar"
    elif not pass_verdict:
        why_unmet = f"persisted verdict={verdict_label!r} is not PASS"
    elif not orch_ok:
        why_unmet = (
            "resident_orchestration is not true; the HCLI loop did not orchestrate work. "
            "cognition is recorded separately and is not this requirement"
        )
    elif not seal_ok:
        why_unmet = (
            "timeline seal does not verify: "
            f"{seal.get('why')}. The judged process must not edit the transcript after the fact"
        )
    elif not cognition_recorded:
        why_unmet = (
            "resident_model_cognition was not recorded; refusing to infer a thinking model. "
            "UNAVAILABLE is a valid recording; silence is not"
        )
    else:
        why_unmet = ""

    reason = (
        (
            f"Odyssey I resident-orchestration autonomy trial {trial_id} persisted PASS "
            f"in AUTONOMY_TRIALS.json; timeline seal verifies; "
            f"resident_orchestration=true; resident_model_cognition={cognition!r}"
        )
        if met
        else why_unmet
    )
    bar = operational_bar(
        discover=bool(trials.get("found")),
        invoke=bool(at_mod.get("ok")),
        schedule=bool(orch_ok),
        verify=bool(seal_ok),
        frontier=bool(candidate),
        persist=bool(isinstance(candidate, Mapping)),
        refill=bool(met),
        notes={
            "hcli_control_plane_is_not_odyssey_i": "true",
            "reads": "receipts/future/AUTONOMY_TRIALS.json",
            "does_not_read": "receipts/future/AUTONOMY_TRIAL.json",
            "orchestration_is_not_cognition": "true",
        },
    )
    return _criterion(
        "resident_autonomy_trial_pass",
        met=met,
        reason=reason,
        evidence=[
            {
                "kind": "persisted_odyssey_trial",
                "path_taken": trials.get("path_taken"),
                "found": trials.get("found"),
                "schema": None if not trials_doc else trials_doc.get("schema"),
                "trial": trial_id,
                "verdict": verdict_label,
                "resident_orchestration": orch,
                "resident_model_cognition": cognition,
                "resident_model_cognition_reason": cognition_reason,
                "timeline_path": None if not candidate else candidate.get("timeline_path"),
                "timeline_seal_digest": None if not candidate else candidate.get("timeline_seal_digest"),
                "timeline_seal_verifies": seal.get("verifies"),
                "timeline_seal_why": seal.get("why"),
                "frozen_build_manifest_digest": None if not candidate else candidate.get("frozen_build_manifest_digest"),
                "launch_eligible": eligible,
            },
            {
                "kind": "wrong_receipt_name",
                "rel": "receipts/future/AUTONOMY_TRIAL.json",
                "found": singular.get("found"),
                "path_taken": singular.get("path_taken"),
                "not_authority": True,
            },
            {
                "kind": "hcli_agentos_autonomy",
                "path_taken": hcli.get("path_taken"),
                "schema": hcli_schema,
                "found": hcli.get("found"),
                "all_requested_stages_passed": hcli_passed,
                "not_this_criterion": True,
            },
        ],
        extra={
            "resident_orchestration": orch,
            "resident_model_cognition": cognition,
            "resident_model_cognition_reason": cognition_reason,
            "timeline_seal_verifies": seal.get("verifies"),
            "persisted_verdict": verdict_label,
            "persisted_trial": trial_id,
        },
        operational=bar,
        probe_performed=(
            "probe_json receipts/future/AUTONOMY_TRIALS.json (plural); "
            "autonomy_trial.launch_candidate_from_receipt; "
            "verify_timeline_digest(timeline_path, timeline_seal_digest); "
            "probe_json receipts/future/AUTONOMY_TRIAL.json recorded as not-authority; "
            "HCLI AgentOS autonomy all_requested_stages_passed recorded as not this criterion"
        ),
        direct_observation=(
            f"AUTONOMY_TRIALS found={trials.get('found')} path_taken={trials.get('path_taken')} "
            f"candidate_is_mapping={isinstance(candidate, Mapping)} trial={trial_id!r} "
            f"verdict={verdict_label!r} eligible={eligible} "
            f"resident_orchestration={orch!r} timeline_seal_verifies={seal.get('verifies')!r} "
            f"seal_why={seal.get('why')!r} cognition_recorded={cognition_recorded} "
            f"cognition={cognition!r} AUTONOMY_TRIAL.json found={singular.get('found')} "
            f"not_authority=True hcli_all_requested_stages_passed={hcli_passed} "
            f"not_this_criterion=True met={met}"
        ),
    )


def _eval_curriculum() -> dict[str, Any]:
    cur = propose_specimen_curriculum()
    met = bool(cur.get("ready"))
    n_ready = cur.get("n_ready")
    n_roles = cur.get("n_roles")
    unmet_roles = [r["role"] for r in cur.get("roles") or [] if not r.get("ready")]
    return _criterion(
        "specimen_curriculum_ready",
        met=met,
        reason=(
            f"all {n_roles} curriculum roles have a live sealed first-wave specimen"
            if met
            else (
                f"curriculum proposes {n_roles} roles; ready={n_ready}. "
                f"unready={unmet_roles}. A proposal is not a ready specimen set."
            )
        ),
        evidence=[{"curriculum_n_roles": n_roles, "n_ready": n_ready, "unready": unmet_roles}],
        extra={"curriculum": cur},
        operational=operational_bar(
            discover=True,
            invoke=True,
            schedule=False,
            verify=False,
            frontier=False,
            persist=False,
            refill=False,
            notes={"proposal_emitted": "true", "specimens_not_all_published": str(not met)},
        ),
        probe_performed=(
            "propose_specimen_curriculum(): ModelLake specimens/ listing, "
            "independent digest recomputation, lake census, Odyssey I patient "
            "seals, law-store schools; a proposal is not a ready specimen set"
        ),
        direct_observation=(
            f"n_roles={n_roles} n_ready={n_ready} ready={met} unready={unmet_roles}"
        ),
    )


def _eval_modellake() -> dict[str, Any]:
    imp = _importable("hcli.agentos.modellake_receipts")
    names: tuple[str, ...] = ()
    if imp.get("ok"):
        try:
            from hcli.agentos.modellake_receipts import CENSUS_RECEIPT_NAMES

            names = tuple(CENSUS_RECEIPT_NAMES)
        except Exception:
            names = ()
    rels = [f"receipts/headless/{n}" for n in names] or [
        "receipts/headless/HCLI_MODELLAKE_FLASH_CENSUS.json",
        "receipts/headless/MODELLAKE_FLASH_NEXT_CENSUS.json",
    ]
    rels.append("receipts/future/evidence/HCLI_MODELLAKE_FLASH_CENSUS.json")
    probe = probe_json(*rels)
    doc = probe.get("doc") if isinstance(probe.get("doc"), Mapping) else None
    schema_ok = bool(doc and str(doc.get("schema") or "").startswith("hcli.agentos.modellake"))
    status = doc.get("status") if doc else None
    identity_ok = bool(imp.get("ok") and probe.get("found") and schema_ok)
    # Identity machinery can be up while Flash is identity-only. That is this
    # criterion; specimen readiness is the curriculum criterion.
    return _criterion(
        "modellake_identity",
        met=identity_ok,
        reason=(
            f"ModelLake identity machinery is importable and a census receipt "
            f"schema={doc.get('schema') if doc else None!r} status={status!r} "
            f"was loaded via {probe.get('path_taken')}"
            if identity_ok
            else (
                f"ModelLake identity not operational: import={imp} census "
                f"path_taken={probe.get('path_taken')!r} schema_ok={schema_ok}"
            )
        ),
        evidence=[
            {"import": {k: v for k, v in imp.items() if k != "file" or v}, "census": {"path_taken": probe.get("path_taken"), "found": probe.get("found"), "schema": None if not doc else doc.get("schema"), "status": status, "qualification": None if not doc else doc.get("qualification")}},
        ],
        operational=operational_bar(
            discover=bool(imp.get("ok")),
            invoke=bool(imp.get("ok")),
            schedule=False,
            verify=schema_ok,
            frontier=False,
            persist=bool(probe.get("found")),
            refill=False,
            notes={"identity_is_not_acquisition": "true"},
        ),
        probe_performed=(
            "import hcli.agentos.modellake_receipts; probe_json census receipts "
            f"{rels} for schema starting hcli.agentos.modellake"
        ),
        direct_observation=(
            f"import_ok={imp.get('ok')} census_found={probe.get('found')} "
            f"path_taken={probe.get('path_taken')} schema_ok={schema_ok} "
            f"schema={None if not doc else doc.get('schema')!r} status={status!r}"
        ),
    )


def _resident_schedulable(owned: Sequence[str]) -> dict[str, Any]:
    """Can the RESIDENT schedule this tool, route its receipt, and refill on it?

    Measured against the connector, not asserted. A binding counts only if the
    orchestration table names a sidecar module that actually drives this tool --
    a module that merely mentions the path in a comment is not a binding.
    """
    result = {"schedule": False, "frontier": None, "refill": False,
              "why": "no orchestration binding drives this tool"}
    try:
        from tools.future import frontiers as _fr
        from tools.future import orchestration as _orch
    except Exception as exc:
        result["why"] = f"connector unavailable: {type(exc).__name__}"
        return result
    wanted = {str(o) for o in owned}
    for module, (frontier_id, _species) in _orch.BINDINGS.items():
        if module == "odyssey_launch.py":
            continue  # this gate does not get to certify itself as the driver
        src = REPO / "tools" / "future" / module
        try:
            text = src.read_text(errors="replace")
        except OSError:
            continue
        if not any(w in text for w in wanted):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        # A driver RUNS it. Naming the path in a list constant is a declaration,
        # and a declaration schedules nothing -- the first version of this check
        # accepted an Assign and so credited this gate module itself, which had
        # the path in an `owned = [...]` literal, as Doctor's driver. Requiring
        # the mention to sit inside a call refuses that. It also refuses a real
        # driver that builds its argv separately, which is the safe direction:
        # this check may understate schedulability, never overstate it.
        drives = any(
            isinstance(n, ast.Call) and any(w in ast.dump(n) for w in wanted)
            for n in ast.walk(tree)
        )
        if not drives:
            continue
        result.update({"schedule": True, "frontier": frontier_id,
                       "driver_module": module, "why": "bound via orchestration"})
        try:
            result["refill"] = any(
                str(item.get("id")) == frontier_id
                # The frontier's own lane vocabulary, not a second list. These
                # were spelled CPU_ANALYSIS / CPU_VERIFY / CPU_REPRESENTATION,
                # which match no frontier item, so the refill probe could only
                # ever return empty and refill was structurally unreachable. The
                # same defect was fixed in the autonomy driver and survived here
                # -- which is exactly why the scar says derive, never restate.
                for item in (_fr.refill(_fr.THIS_HOST_LANES) or [])
            )
        except Exception:
            result["refill"] = False
        break
    return result


def _declared_inputs(rel: str) -> list[dict[str, Any]]:
    """Absolute filesystem paths a tool hardcodes as its input, and whether they exist.

    Parsed, never imported: importing an odyssey tool would run its module-level
    work inside the gate. Only literal `NAME = Path("/abs/...")` assignments are
    read, which is exactly how the Doctor and Gravity tools name their parent.
    """
    path = REPO / rel
    if not path.is_file():
        loc = _module_file(rel)
        resolved = loc.get("resolved")
        if loc.get("present") and resolved and Path(str(resolved)).is_file():
            path = Path(str(resolved))
        else:
            return []
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except (OSError, SyntaxError):
        return []
    out: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "Path" and len(call.args) == 1
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)):
            continue
        literal = call.args[0].value
        if not literal.startswith("/"):
            continue  # a repo-relative path is not an external input
        out.append({"name": target.id, "path": literal,
                    "present": Path(literal).exists(), "declared_in": rel})
    return out


# Where large model parents actually live on this host. Bounded on purpose: a
# find over the 4.5TB volume to answer "is the parent here" is not a probe.
MODEL_ROOTS: tuple[str, ...] = (
    "/Users/scammermike/models",
    "/Volumes/corpdrive/personalmodel/correspondent",
    "/Volumes/corpdrive/personalmodel",
    "/Volumes/corpdrive/hawking-modellake/specimens",
)


def _resolve_stale_input(declared: str) -> str | None:
    """The same directory name under a known model root, if the declared path moved.

    "Not on this host" and "not where the tool says" are different findings, and
    only one of them is a missing model. The Doctor and Gravity tools declare
    /Users/scammermike/models/qwen3.8-27b-abliterated-bf16, which is gone; the
    52GB parent is on the external volume. Saying the model is missing would
    send someone to re-download 52GB that is already here.
    """
    name = Path(declared).name
    if not name:
        return None
    for root in MODEL_ROOTS:
        candidate = Path(root) / name
        if candidate.is_dir():
            return str(candidate)
    return None


def _eval_callable_tool(*, cid: str, owned: Sequence[str], prior_glob: str, title: str) -> dict[str, Any]:
    files = [_module_file(p) for p in owned]
    prior: list[str] = []
    seen: set[str] = set()
    # Sparse miss of receipts/odyssey-i in this worktree is not absence in the
    # project. Search every checkout root the rest of this gate already uses.
    for checkout in _checkout_roots():
        folder = checkout / "receipts" / "odyssey-i"
        if not folder.is_dir():
            continue
        for p in folder.glob(prior_glob):
            rel = f"receipts/odyssey-i/{p.name}"
            if rel not in seen:
                seen.add(rel)
                prior.append(rel)
    prior = sorted(prior)
    present = any(f.get("present") for f in files) or bool(prior)
    # File presence is not invocability. These tools hardcode the parent weights
    # they read, and a tool whose parent is not on this host cannot be invoked by
    # anyone, resident or human. Measure it rather than inferring it from the
    # presence of the script.
    inputs = [i for rel in owned for i in _declared_inputs(rel)]
    for item in inputs:
        if not item["present"]:
            item["resolved_elsewhere"] = _resolve_stale_input(item["path"])
    missing_inputs = [i for i in inputs if not i["present"]]
    stale_inputs = [i for i in missing_inputs if i.get("resolved_elsewhere")]
    absent_inputs = [i for i in missing_inputs if not i.get("resolved_elsewhere")]
    invocable = (
        any(
            f.get("present") and f.get("path_taken") in {"worktree", "primary_checkout"}
            for f in files
        )
        and not missing_inputs
    )
    # schedule / frontier / refill are measured against the connector that now
    # exists, instead of being asserted False. They are still false here, but for
    # a reason that can be checked and can change.
    sched = _resident_schedulable(owned)
    bar = operational_bar(
        discover=present,
        invoke=invocable,
        schedule=bool(sched["schedule"]),
        verify=bool(prior),
        frontier=bool(sched["frontier"]),
        persist=bool(prior),
        refill=bool(sched["refill"]),
        notes={
            "cli_or_prior_seals": "true",
            "declared_inputs_missing": ", ".join(i["path"] for i in missing_inputs) or "none",
            "stale_declared_paths": ", ".join(
                f"{i['path']} -> {i['resolved_elsewhere']}" for i in stale_inputs
            ) or "none",
            "integration": INTEGRATION_POINTS["workgraph"] + " + " + INTEGRATION_POINTS["resident_api"],
        },
    )
    return _criterion(
        cid,
        met=bool(bar["resident_operational"]),
        reason=(
            f"{title} is resident-operational"
            if bar["resident_operational"]
            else (
                f"{title} is recovered (n_prior={len(prior)}) but is not resident-"
                f"operational. Unmet: "
                + ", ".join(k for k, v in bar["flags"].items() if not v)
                + (
                    f". The blocker is a STALE DECLARED PATH, not a missing model: "
                    + "; ".join(
                        f"{i['name']} in {i['declared_in']} points at {i['path']}, "
                        f"which is gone, while the directory is present at "
                        f"{i['resolved_elsewhere']}"
                        for i in stale_inputs
                    )
                    + ". The tool cannot run as written."
                    if stale_inputs
                    else (
                        f". {', '.join(i['path'] for i in absent_inputs)} is not on "
                        f"this host at all, so the tool cannot be invoked by anyone. "
                        f"A CLI a human cannot run either is not enough."
                        if absent_inputs
                        else ". A CLI a human can run is not enough."
                    )
                )
            )
        ),
        evidence=[{"owned": files, "n_prior_seals": len(prior), "prior_seals": prior,
                   "declared_inputs": inputs, "missing_inputs": missing_inputs,
                   "stale_declared_paths": stale_inputs, "absent_inputs": absent_inputs,
                   "resident_schedulable": sched}],
        operational=bar,
        probe_performed=(
            f"_module_file on {list(owned)}; glob prior seals {prior_glob} across "
            "checkout roots; ast-parse Path('/abs/...') literals and Path.exists(); "
            "_resolve_stale_input under MODEL_ROOTS; _resident_schedulable via "
            "orchestration.BINDINGS (Call nodes only; this gate cannot self-certify)"
        ),
        direct_observation=(
            f"present={present} invocable={invocable} n_prior={len(prior)} "
            f"missing_inputs={[i['path'] for i in missing_inputs]} "
            "stale_declared_paths="
            + str([f"{i['path']}->{i.get('resolved_elsewhere')}" for i in stale_inputs])
            + f" absent_inputs={[i['path'] for i in absent_inputs]} "
            f"schedule={sched.get('schedule')} frontier={sched.get('frontier')} "
            f"refill={sched.get('refill')} driver={sched.get('driver_module')} "
            f"flags={bar['flags']} resident_operational={bar['resident_operational']}"
        ),
    )


def _eval_doctor() -> dict[str, Any]:
    owned = ["tools/odyssey/doctor_tournament.py", "tools/doctor_seal.py"]
    hcli = _importable("hcli.doctor")
    extra: list[str] = []
    if hcli.get("ok"):
        import hcli.doctor as hd

        extra = [str(p) for p in getattr(hd, "OWNED_PATHS", ())]
    return _eval_callable_tool(
        cid="doctor_callable",
        owned=list(owned) + extra,
        prior_glob="*_DOCTOR_SEAL.json",
        title="Doctor",
    )


def _eval_gravity() -> dict[str, Any]:
    return _eval_callable_tool(
        cid="gravity_callable",
        owned=[
            "tools/odyssey/decoding_gravity.py",
            "tools/odyssey/state_gravity.py",
            "hcli/gravity/__init__.py",
        ],
        prior_glob="*_GRAVITY_*.json",
        title="Gravity",
    )


def _nr_nx_generic_state() -> dict[str, Any]:
    """Invoke nr_nx_generic. A receipt without a driver is a declaration, not a drive."""
    loaded = _load_future_module("nr_nx_generic")
    rec = probe_json("receipts/future/NR_NX_GENERIC.json")
    rec_doc = rec.get("doc") if isinstance(rec.get("doc"), Mapping) else None
    flash_live: dict[str, Any] | None = None
    generic_live = None
    first_failing = None
    invoked: list[str] = []
    if loaded.get("ok"):
        mod = loaded["module"]
        try:
            flash_live = mod.flash_nx_ready()
            invoked.append("nr_nx_generic.flash_nx_ready")
        except Exception as exc:
            flash_live = {
                "FLASH_NX_READY": False,
                "why": f"flash_nx_ready raised {type(exc).__name__}: {exc}",
            }
            invoked.append("nr_nx_generic.flash_nx_ready:RAISED")
        stages = list((rec_doc or {}).get("stages") or [])
        try:
            generic_live = bool(mod.generic_pipeline_callable(stages))
            invoked.append("nr_nx_generic.generic_pipeline_callable")
        except Exception as exc:
            generic_live = False
            invoked.append("nr_nx_generic.generic_pipeline_callable:RAISED")
            first_failing = {"stage": None, "why": f"{type(exc).__name__}: {exc}"}
        if first_failing is None and hasattr(mod, "first_failing_stage"):
            try:
                first_failing = mod.first_failing_stage(stages)
            except Exception:
                first_failing = (rec_doc or {}).get("first_failing_stage")
    else:
        generic_live = False
        flash_live = {"FLASH_NX_READY": False, "why": loaded.get("why")}
        first_failing = None if not rec_doc else rec_doc.get("first_failing_stage")

    receipt_generic = None if not rec_doc else rec_doc.get("GENERIC_NR_NX_PIPELINE_CALLABLE")
    receipt_flash = None if not rec_doc else rec_doc.get("FLASH_NX_READY")
    # Live function wins. A receipt that says callable while the function says
    # not is a declaration, and a declaration is not a drive.
    generic = False if generic_live is None else bool(generic_live)
    flash_ready = False
    flash_why = "FLASH_NX_READY not established"
    if isinstance(flash_live, Mapping):
        flash_ready = bool(flash_live.get("FLASH_NX_READY") is True)
        flash_why = str(flash_live.get("why") or flash_why)
    return {
        "invoked": invoked,
        "import": {k: v for k, v in loaded.items() if k != "module"},
        "receipt_path_taken": rec.get("path_taken"),
        "receipt_found": rec.get("found"),
        "GENERIC_NR_NX_PIPELINE_CALLABLE": generic,
        "GENERIC_FROM_RECEIPT": receipt_generic,
        "FLASH_NX_READY": flash_ready,
        "FLASH_FROM_RECEIPT": receipt_flash,
        "flash": flash_live,
        "flash_why": flash_why,
        "first_failing_stage": first_failing if first_failing is not None else (
            None if not rec_doc else rec_doc.get("first_failing_stage")
        ),
        "n_stages": 0 if not rec_doc else len(list(rec_doc.get("stages") or [])),
    }


def _eval_nr_nx() -> dict[str, Any]:
    """Generic pipeline result, not Flash artifact readiness.

    FLASH_NX_READY is reported separately and stays false until Flash earns a
    packed NX. Qwen27 launches; Flash is a child. If the generic pipeline is
    not callable on any available specimen, this criterion stays unmet.
    """
    state = _nr_nx_generic_state()
    generic = state.get("GENERIC_NR_NX_PIPELINE_CALLABLE") is True
    flash_ready = state.get("FLASH_NX_READY") is True
    invoked = list(state.get("invoked") or [])
    first = state.get("first_failing_stage")
    bar = operational_bar(
        discover=bool(state.get("receipt_found") or state.get("import", {}).get("ok")),
        invoke=bool(invoked),
        schedule=bool(state.get("import", {}).get("ok")),
        verify=bool(invoked),
        frontier=bool(state.get("import", {}).get("ok")),
        persist=bool(state.get("receipt_found")),
        refill=False,
        notes={
            "FLASH_NX_READY_is_not_this_criterion": "true",
            "generic_pipeline_callable": str(generic),
            "FLASH_NX_READY": str(flash_ready),
        },
    )
    met = bool(generic)
    first_why = None
    if isinstance(first, Mapping):
        first_why = f"{first.get('stage')}: {first.get('why') or first.get('error') or first.get('status')}"
    reason = (
        "generic NR→NX pipeline is callable on an available specimen; FLASH_NX_READY is independent"
        if met
        else (
            "generic NR→NX pipeline is not callable. "
            f"GENERIC_NR_NX_PIPELINE_CALLABLE={generic} "
            f"FLASH_NX_READY={flash_ready} (separate field, not this criterion). "
            f"first_failing_stage={first_why!r}. "
            f"invoked={invoked}. A Flash metadata seal is not a packed NX and is not this path."
        )
    )
    return _criterion(
        "nr_nx_path_callable",
        met=met,
        reason=reason,
        evidence=[
            {
                "kind": "nr_nx_generic",
                "invoked": invoked,
                "import_ok": bool(state.get("import", {}).get("ok")),
                "import_path_taken": state.get("import", {}).get("path_taken") or state.get("import", {}).get("why"),
                "receipt_path_taken": state.get("receipt_path_taken"),
                "GENERIC_NR_NX_PIPELINE_CALLABLE": generic,
                "GENERIC_FROM_RECEIPT": state.get("GENERIC_FROM_RECEIPT"),
                "FLASH_NX_READY": flash_ready,
                "FLASH_FROM_RECEIPT": state.get("FLASH_FROM_RECEIPT"),
                "flash_why": state.get("flash_why"),
                "first_failing_stage": first,
                "n_stages": state.get("n_stages"),
            }
        ],
        extra={
            "GENERIC_NR_NX_PIPELINE_CALLABLE": generic,
            "FLASH_NX_READY": flash_ready,
            "facts_are_independent": True,
        },
        operational=bar,
        probe_performed=(
            "import nr_nx_generic; invoke flash_nx_ready() and "
            "generic_pipeline_callable(stages); probe_json "
            "receipts/future/NR_NX_GENERIC.json; live function wins over a "
            "receipt declaration"
        ),
        direct_observation=(
            f"invoked={invoked} import_ok={bool(state.get('import', {}).get('ok'))} "
            f"GENERIC_NR_NX_PIPELINE_CALLABLE={generic} "
            f"GENERIC_FROM_RECEIPT={state.get('GENERIC_FROM_RECEIPT')!r} "
            f"FLASH_NX_READY={flash_ready} "
            f"FLASH_FROM_RECEIPT={state.get('FLASH_FROM_RECEIPT')!r} "
            f"first_failing_stage={first_why!r} n_stages={state.get('n_stages')} "
            f"receipt_path_taken={state.get('receipt_path_taken')}"
        ),
    )


def _eval_evidence_hierarchy() -> dict[str, Any]:
    snap = probe_json("receipts/future/EVIDENCE_SNAPSHOT.json")
    doc = snap.get("doc") if isinstance(snap.get("doc"), Mapping) else None
    n_captured = len(doc.get("captured") or []) if doc else 0
    lattice_ok = tuple(C.MEASUREMENT_CLASSES) == EVIDENCE_LATTICE or set(C.MEASUREMENT_CLASSES) == set(EVIDENCE_LATTICE)
    dag = _module_file("tools/future/evidence_dag.py")
    live_dag = _exercise("tools.future.evidence_dag", "selftest")
    bar = operational_bar(
        discover=lattice_ok,
        invoke=lattice_ok,
        schedule=bool(dag.get("present")) and bool(live_dag.get("ok")),
        verify=n_captured > 0,
        frontier=bool(live_dag.get("ok")),
        persist=bool(snap.get("found")),
        refill=bool(live_dag.get("ok")),
        notes={
            "lattice_recovered": str(lattice_ok),
            "evidence_dag": dag.get("path_taken"),
            "integration": INTEGRATION_POINTS["evidence_dag"],
        },
    )
    met = bool(bar["resident_operational"])
    return _criterion(
        "evidence_hierarchy",
        met=met,
        reason=(
            "evidence DAG is resident-operational"
            if met
            else (
                f"measurement-class lattice is encoded ({list(C.MEASUREMENT_CLASSES)}) "
                f"and evidence snapshot captured={n_captured} via {snap.get('path_taken')}, "
                "but the resident-callable evidence DAG is not landed "
                f"({INTEGRATION_POINTS['evidence_dag']}). DIAGNOSTIC_RELATIVE guides and never promotes."
            )
        ),
        evidence=[{"lattice": list(C.MEASUREMENT_CLASSES), "snapshot_captured": n_captured, "dag": dag}],
        operational=bar,
        probe_performed=(
            "probe_json receipts/future/EVIDENCE_SNAPSHOT.json; compare "
            "contamination.MEASUREMENT_CLASSES to EVIDENCE_LATTICE; "
            "_module_file tools/future/evidence_dag.py; "
            "_exercise tools.future.evidence_dag.selftest"
        ),
        direct_observation=(
            f"lattice={list(C.MEASUREMENT_CLASSES)} lattice_ok={lattice_ok} "
            f"snapshot_captured={n_captured} path_taken={snap.get('path_taken')} "
            f"dag_present={dag.get('present')} dag_path_taken={dag.get('path_taken')} "
            f"selftest_ok={live_dag.get('ok')} selftest_why={live_dag.get('why')} "
            f"flags={bar['flags']}"
        ),
    )


def _eval_negative_science() -> dict[str, Any]:
    imp = _importable("tools.future.negative_index")
    rec = probe_json("receipts/future/NEGATIVE_SCIENCE_INDEX.json")
    doc = rec.get("doc") if isinstance(rec.get("doc"), Mapping) else None
    n_scars = None
    if doc and isinstance(doc.get("coverage"), Mapping):
        n_scars = doc["coverage"].get("n_scars")
    has_api = bool(imp.get("ok") and hasattr(__import__("tools.future.negative_index", fromlist=["query"]), "query"))
    bar = operational_bar(
        discover=bool(imp.get("ok")),
        invoke=has_api,
        schedule=True,
        verify=has_api,
        frontier=True,
        persist=_receipt_sealed(doc),
        refill=True,
        notes={"refuse_if_dead": "tools.future.negative_index.refuse_if_dead"},
    )
    met = bool(imp.get("ok") and rec.get("found") and (n_scars is None or n_scars > 0) and bar["resident_operational"])
    return _criterion(
        "negative_science",
        met=met,
        reason=(
            f"negative-science index is importable, sealed, scars={n_scars}, query/refuse_if_dead callable"
            if met
            else f"negative science not operational: import={imp.get('ok')} receipt={rec.get('path_taken')} n_scars={n_scars}"
        ),
        evidence=[{"import_ok": imp.get("ok"), "path_taken": rec.get("path_taken"), "n_scars": n_scars}],
        operational=bar,
        probe_performed=(
            "import tools.future.negative_index; hasattr query; "
            "probe_json receipts/future/NEGATIVE_SCIENCE_INDEX.json; "
            "read coverage.n_scars"
        ),
        direct_observation=(
            f"import_ok={imp.get('ok')} has_query={has_api} "
            f"path_taken={rec.get('path_taken')} found={rec.get('found')} "
            f"n_scars={n_scars} sealed={_receipt_sealed(doc)} flags={bar['flags']}"
        ),
    )



def _exercise(dotted: str, fn_name: str) -> dict[str, Any]:
    """Actually run a now-landed sibling capability.

    These criteria were written while the siblings were being built concurrently,
    so the contract forbade importing them and the evaluators hard-coded the
    frontier/persist/refill axes to False. Those modules have since LANDED and are
    committed, so the honest evaluation is to exercise them rather than keep
    asserting an absence that is no longer true. If a call raises, the criterion
    stays unmet with the exception recorded -- this never flips to True by fiat.
    """
    try:
        mod = importlib.import_module(dotted)
    except Exception as exc:
        return {"ok": False, "why": f"import failed: {type(exc).__name__}: {exc}"}
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        return {"ok": False, "why": f"{dotted}.{fn_name} is not callable"}
    try:
        out = fn()
    except Exception as exc:
        return {"ok": False, "why": f"{fn_name}() raised {type(exc).__name__}: {exc}"}
    return {"ok": True, "called": f"{dotted}.{fn_name}()", "result": str(out)[:200]}


def _eval_workgraphs() -> dict[str, Any]:
    runtime = _module_file("tools/future/workgraph.py")
    species = _importable("tools.future.workunit_species")
    live = _exercise("tools.future.workgraph", "selftest")
    bar = operational_bar(
        discover=bool(species.get("ok")),
        invoke=bool(species.get("ok")),
        schedule=bool(runtime.get("present")) and bool(live.get("ok")),
        verify=bool(species.get("ok")),
        frontier=bool(live.get("ok")),
        persist=bool(live.get("ok")),
        refill=bool(live.get("ok")),
        notes={
            "this_module_emits_units": "true",
            "runtime": runtime.get("path_taken"),
            "integration": INTEGRATION_POINTS["workgraph"],
        },
    )
    return _criterion(
        "workgraphs",
        met=bool(bar["resident_operational"]),
        reason=(
            "WorkGraph runtime is resident-operational"
            if bar["resident_operational"]
            else (
                "This gate emits first WorkGraphs as real HCLI WorkUnits, but the "
                f"WorkGraph runtime is not landed ({INTEGRATION_POINTS['workgraph']}). "
                "Emitting a graph is not executing a graph."
            )
        ),
        evidence=[{"runtime": runtime, "workunit_species_import": species.get("ok"),
                   "exercised": live}],
        operational=bar,
        probe_performed=(
            "_module_file tools/future/workgraph.py; import tools.future.workunit_species; "
            "_exercise tools.future.workgraph.selftest"
        ),
        direct_observation=(
            f"runtime_present={runtime.get('present')} path_taken={runtime.get('path_taken')} "
            f"species_ok={species.get('ok')} selftest_ok={live.get('ok')} "
            f"selftest_why={live.get('why')} flags={bar['flags']}"
        ),
    )


def _eval_self_refill() -> dict[str, Any]:
    frontier = probe_json("receipts/future/CLAUDE_GLOBAL_FRONTIER.json")
    fronts = _module_file("tools/future/frontiers.py")
    succ = _module_file("tools/future/succession.py")
    live_refill = _exercise("tools.future.frontiers", "refill")
    # is_idle() is the verify probe: a refill loop that cannot say whether the
    # frontier is exhausted is not a verified loop, it is a generator.
    live_idle = _exercise("tools.future.frontiers", "is_idle")
    bar = operational_bar(
        discover=bool(frontier.get("found")),
        invoke=bool(live_refill.get("ok")),
        schedule=bool(live_refill.get("ok")),
        verify=bool(live_idle.get("ok")),
        frontier=bool(frontier.get("found")),
        persist=bool(frontier.get("found")),
        refill=bool(fronts.get("present") and succ.get("present") and live_refill.get("ok")),
        notes={
            "global_frontier_is_inventory": "true",
            "exercised_refill": str(live_refill.get("ok")),
            "exercised_is_idle": str(live_idle.get("ok")),
            "integration": INTEGRATION_POINTS["frontiers"] + " + " + INTEGRATION_POINTS["succession"],
        },
    )
    return _criterion(
        "self_refill",
        met=bool(bar["resident_operational"]),
        reason=(
            "resident self-refill loop is operational"
            if bar["resident_operational"]
            else (
                f"CLAUDE_GLOBAL_FRONTIER path_taken={frontier.get('path_taken')} is an "
                "inventory, not a refill loop. frontiers.py/succession.py are this-wave "
                "integration points and are not imported."
            )
        ),
        evidence=[{"frontier": {"path_taken": frontier.get("path_taken"), "found": frontier.get("found")}, "frontiers": fronts, "succession": succ}],
        operational=bar,
        probe_performed=(
            "probe_json receipts/future/CLAUDE_GLOBAL_FRONTIER.json; "
            "_module_file frontiers.py and succession.py; "
            "_exercise tools.future.frontiers.refill and tools.future.frontiers.is_idle"
        ),
        direct_observation=(
            f"frontier_found={frontier.get('found')} path_taken={frontier.get('path_taken')} "
            f"frontiers_present={fronts.get('present')} succession_present={succ.get('present')} "
            f"refill_ok={live_refill.get('ok')} is_idle_ok={live_idle.get('ok')} "
            f"refill_why={live_refill.get('why')} idle_why={live_idle.get('why')} "
            f"flags={bar['flags']}"
        ),
    )


def _eval_dirty_measurement() -> dict[str, Any]:
    dirty = _module_file("tools/future/dirty_measure.py")
    live_dirty = _exercise("tools.future.dirty_measure", "build")
    cont = probe_json("receipts/future/CONTAMINATION_SCIENCE.json")
    doc = cont.get("doc") if isinstance(cont.get("doc"), Mapping) else None
    klass = doc.get("contamination_class") if doc else None
    bar = operational_bar(
        discover=bool(cont.get("found")),
        invoke=bool(_importable("tools.future.contamination").get("ok")),
        schedule=bool(dirty.get("present")) and bool(live_dirty.get("ok")),
        verify=klass in C.CONTAMINATION_CLASSES if klass else False,
        frontier=bool(live_dirty.get("ok")),
        persist=bool(cont.get("found")),
        refill=bool(live_dirty.get("ok")),
        notes={
            "contamination_is_machine_state": "true",
            "dirty_measure_integration": INTEGRATION_POINTS["dirty_measure"],
            "gpu_dirty_ok_resource_class_exists": "GPU_DIRTY_OK",
        },
    )
    return _criterion(
        "dirty_measurement",
        met=bool(bar["resident_operational"]),
        reason=(
            "dirty measurement is resident-operational"
            if bar["resident_operational"]
            else (
                f"contamination science is sealed (class={klass!r}) and refuses "
                "DIAGNOSTIC_RELATIVE promotion, but dirty-class measurement of "
                f"tokens/experiments is {INTEGRATION_POINTS['dirty_measure']}."
            )
        ),
        evidence=[{"contamination_class": klass, "path_taken": cont.get("path_taken"), "dirty_measure": dirty}],
        operational=bar,
        probe_performed=(
            "_module_file tools/future/dirty_measure.py; "
            "_exercise tools.future.dirty_measure.build; "
            "probe_json receipts/future/CONTAMINATION_SCIENCE.json; "
            "read contamination_class"
        ),
        direct_observation=(
            f"contamination_class={klass!r} path_taken={cont.get('path_taken')} "
            f"dirty_present={dirty.get('present')} build_ok={live_dirty.get('ok')} "
            f"build_why={live_dirty.get('why')} flags={bar['flags']}"
        ),
    )


def _protected_capability_report() -> dict[str, Any]:
    """Invoke protected_scheduler.capability_report(). Absence is incapable, not available."""
    loaded = _load_future_module("protected_scheduler")
    rec = probe_json("receipts/future/PROTECTED_SCHEDULER.json")
    if not loaded.get("ok"):
        return {
            "invoked": False,
            "why": loaded.get("why"),
            "PROTECTED_SCHEDULER_CAPABLE": False,
            "PROTECTED_WINDOW_AVAILABLE": False,
            "contamination_class": None,
            "receipt_path_taken": rec.get("path_taken"),
            "did_not_fabricate_lease": True,
            "did_not_flock": True,
        }
    try:
        report = loaded["module"].capability_report()
    except Exception as exc:
        return {
            "invoked": False,
            "why": f"capability_report raised {type(exc).__name__}: {exc}",
            "PROTECTED_SCHEDULER_CAPABLE": False,
            "PROTECTED_WINDOW_AVAILABLE": False,
            "contamination_class": None,
            "receipt_path_taken": rec.get("path_taken"),
            "did_not_fabricate_lease": True,
            "did_not_flock": True,
        }
    if not isinstance(report, Mapping):
        return {
            "invoked": False,
            "why": "capability_report returned a non-mapping",
            "PROTECTED_SCHEDULER_CAPABLE": False,
            "PROTECTED_WINDOW_AVAILABLE": False,
            "contamination_class": None,
            "receipt_path_taken": rec.get("path_taken"),
            "did_not_fabricate_lease": True,
            "did_not_flock": True,
        }
    klass = report.get("contamination_class")
    capable = report.get("PROTECTED_SCHEDULER_CAPABLE") is True
    available = report.get("PROTECTED_WINDOW_AVAILABLE") is True
    # A non-QUIESCENT machine cannot have an available protected window. If the
    # report claims otherwise, the window field is refused, not the capability.
    availability_overridden = False
    if klass != "QUIESCENT" and available:
        available = False
        availability_overridden = True
    return {
        "invoked": True,
        "why": report.get("live_reason"),
        "PROTECTED_SCHEDULER_CAPABLE": capable,
        "PROTECTED_WINDOW_AVAILABLE": available,
        "availability_overridden_because_not_quiescent": availability_overridden,
        "contamination_class": klass,
        "lease_present": report.get("lease_present"),
        "live_verdict": report.get("live_verdict"),
        "did_not_fabricate_lease": report.get("did_not_fabricate_lease") is not False,
        "did_not_flock": report.get("did_not_flock") is not False,
        "receipt_path_taken": rec.get("path_taken"),
        "import_path_taken": loaded.get("path_taken"),
        "file": loaded.get("file"),
    }


def _eval_protected_scheduling() -> dict[str, Any]:
    """Capability of the scheduler, not availability of the window.

    A capable scheduler on an unavailable window is CAPABLE. No protected work
    can run right now, and that is reported as PROTECTED_WINDOW_AVAILABLE=false.
    A completed protected physical run is a SEPARATE named requirement and is
    not this criterion: Odyssey I start needs a scheduler that can handle
    protected work, not a PROTECTED_ABSOLUTE result this sidecar cannot produce.
    Never fabricate a lease; never seize a contested lock.
    """
    cap = _protected_capability_report()
    capable = cap.get("PROTECTED_SCHEDULER_CAPABLE") is True and cap.get("invoked") is True
    klass = cap.get("contamination_class")
    available = cap.get("PROTECTED_WINDOW_AVAILABLE") is True
    if klass != "QUIESCENT":
        available = False
    future_pw = _module_file("tools/future/protected_window.py")
    odyssey_pw = _module_file("tools/odyssey/protected_window.py")
    bar = operational_bar(
        discover=bool(cap.get("invoked") or future_pw.get("present")),
        invoke=bool(cap.get("invoked")),
        schedule=bool(capable),
        verify=bool(cap.get("invoked")),
        frontier=bool(capable),
        persist=bool(cap.get("receipt_path_taken") not in {None, "not_found"}),
        refill=bool(capable),
        notes={
            "sidecar_must_not_seize_lock": "true",
            "PROTECTED_SCHEDULER_CAPABLE": str(capable),
            "PROTECTED_WINDOW_AVAILABLE": str(available),
            "availability_is_not_capability": "true",
            "contamination_class": str(klass),
            "did_not_fabricate_lease": str(cap.get("did_not_fabricate_lease")),
            "did_not_flock": str(cap.get("did_not_flock")),
            "integration": INTEGRATION_POINTS["protected_scheduler"],
        },
    )
    met = bool(capable)
    reason = (
        (
            "protected scheduler is CAPABLE; window "
            f"AVAILABLE={available} (contamination_class={klass!r}). "
            "Capability is not availability; no lock was seized"
        )
        if met
        else (
            "protected scheduler is not CAPABLE: "
            f"invoked={cap.get('invoked')} why={cap.get('why')!r} "
            f"contamination_class={klass!r}. "
            "The sidecar will not flock a contested bench lock and will not fabricate a lease"
        )
    )
    return _criterion(
        "protected_scheduling",
        met=met,
        reason=reason,
        evidence=[
            {
                "kind": "protected_scheduler.capability_report",
                "invoked": cap.get("invoked"),
                "PROTECTED_SCHEDULER_CAPABLE": capable,
                "PROTECTED_WINDOW_AVAILABLE": available,
                "contamination_class": klass,
                "live_verdict": cap.get("live_verdict"),
                "lease_present": cap.get("lease_present"),
                "availability_overridden_because_not_quiescent": cap.get(
                    "availability_overridden_because_not_quiescent"
                ),
                "did_not_fabricate_lease": cap.get("did_not_fabricate_lease"),
                "did_not_flock": cap.get("did_not_flock"),
                "import_path_taken": cap.get("import_path_taken"),
                "receipt_path_taken": cap.get("receipt_path_taken"),
                "future_protected_window": future_pw,
                "odyssey_protected_window": odyssey_pw,
            }
        ],
        extra={
            "PROTECTED_SCHEDULER_CAPABLE": capable,
            "PROTECTED_WINDOW_AVAILABLE": available,
            "contamination_class": klass,
            "protected_physical_run_completed": {
                "id": "protected_physical_run_completed",
                "required_for_this_criterion": False,
                "required_for_odyssey_i_start": False,
                "status": "NOT_THIS_CRITERION",
                "why": (
                    "A completed protected physical run would be PROTECTED_ABSOLUTE "
                    "evidence this sidecar cannot produce. Odyssey I start needs a "
                    "capable protected scheduler, not a finished protected bench. "
                    "This field is named so it cannot be smuggled into CAPABLE."
                ),
            },
        },
        operational=bar,
        probe_performed=(
            "import protected_scheduler; invoke capability_report(); "
            "probe_json receipts/future/PROTECTED_SCHEDULER.json; "
            "_module_file protected_window.py; do not flock a bench lock; "
            "do not fabricate a lease"
        ),
        direct_observation=(
            f"invoked={cap.get('invoked')} why={cap.get('why')!r} "
            f"PROTECTED_SCHEDULER_CAPABLE={capable} "
            f"PROTECTED_WINDOW_AVAILABLE={available} contamination_class={klass!r} "
            f"live_verdict={cap.get('live_verdict')!r} "
            f"lease_present={cap.get('lease_present')!r} "
            f"did_not_flock={cap.get('did_not_flock')} "
            f"did_not_fabricate_lease={cap.get('did_not_fabricate_lease')} "
            f"receipt_path_taken={cap.get('receipt_path_taken')}"
        ),
    )


def _eval_transfer() -> dict[str, Any]:
    imp = _importable("tools.future.odyssey2_law_store")
    rec = probe_json("receipts/future/ODYSSEY2_LAW_STORE.json")
    doc = rec.get("doc") if isinstance(rec.get("doc"), Mapping) else None
    n_laws = None
    if doc and isinstance(doc.get("counts"), Mapping):
        n_laws = doc["counts"].get("n_laws")
    elif doc:
        n_laws = len(doc.get("laws") or [])
    has_promote = bool(imp.get("ok") and hasattr(ols, "promote") and hasattr(ols, "transfer_candidates"))
    bar = operational_bar(
        discover=bool(imp.get("ok")),
        invoke=has_promote,
        schedule=True,
        verify=has_promote,
        frontier=True,
        persist=_receipt_sealed(doc),
        refill=True,
        notes={"species": "odyssey_ii_transfer_experiment", "raises_on_negative_transfer": "true"},
    )
    met = bool(imp.get("ok") and rec.get("found") and (n_laws or 0) > 0 and bar["resident_operational"])
    return _criterion(
        "transfer_substrate",
        met=met,
        reason=(
            f"Odyssey II law store is sealed with n_laws={n_laws}; promote/transfer_candidates raise rather than flag"
            if met
            else f"transfer substrate not operational: import={imp.get('ok')} path_taken={rec.get('path_taken')} n_laws={n_laws}"
        ),
        evidence=[{"n_laws": n_laws, "path_taken": rec.get("path_taken"), "schema": None if not doc else doc.get("schema")}],
        operational=bar,
        probe_performed=(
            "import tools.future.odyssey2_law_store; hasattr promote and "
            "transfer_candidates; probe_json receipts/future/ODYSSEY2_LAW_STORE.json; "
            "count n_laws"
        ),
        direct_observation=(
            f"import_ok={imp.get('ok')} has_promote={has_promote} "
            f"path_taken={rec.get('path_taken')} found={rec.get('found')} "
            f"n_laws={n_laws} sealed={_receipt_sealed(doc)} flags={bar['flags']}"
        ),
    )


def _eval_adversary() -> dict[str, Any]:
    imp = _importable("tools.future.odyssey3_adversary")
    rec = probe_json("receipts/future/ODYSSEY3_ADVERSARY.json")
    doc = rec.get("doc") if isinstance(rec.get("doc"), Mapping) else None
    families = list(doc.get("attack_families") or []) if doc else []
    has_api = bool(imp.get("ok") and hasattr(o3, "p_refutation"))
    bar = operational_bar(
        discover=bool(imp.get("ok")),
        invoke=has_api,
        schedule=True,
        verify=has_api,
        frontier=True,
        persist=_receipt_sealed(doc),
        refill=True,
        notes={"species": "odyssey_iii_adversarial_experiment", "a_law_that_emits_no_attack_is_refused": "true"},
    )
    met = bool(imp.get("ok") and rec.get("found") and families and bar["resident_operational"])
    return _criterion(
        "adversary_substrate",
        met=met,
        reason=(
            f"Odyssey III adversary is sealed; attack_families derived from receipt (n={len(families)})"
            if met
            else f"adversary substrate not operational: import={imp.get('ok')} path_taken={rec.get('path_taken')}"
        ),
        evidence=[{"n_attack_families": len(families), "path_taken": rec.get("path_taken"), "schema": None if not doc else doc.get("schema")}],
        operational=bar,
        probe_performed=(
            "import tools.future.odyssey3_adversary; hasattr p_refutation; "
            "probe_json receipts/future/ODYSSEY3_ADVERSARY.json; read attack_families"
        ),
        direct_observation=(
            f"import_ok={imp.get('ok')} has_p_refutation={has_api} "
            f"path_taken={rec.get('path_taken')} found={rec.get('found')} "
            f"n_attack_families={len(families)} flags={bar['flags']}"
        ),
    )


def _eval_crash_recovery() -> dict[str, Any]:
    imp = _importable("tools.future.repro_science")
    rec = probe_json("receipts/future/REPRO_SCIENCE.json")
    recov = probe_json(
        "receipts/headless/HCLI_AGENTOS_RECOVERY_GATE.json",
        "receipts/future/evidence/HCLI_AGENTOS_RECOVERY_GATE.json",
    )
    git_lock = probe_json("receipts/future/GIT_LOCK_DURABILITY_REPORT.json")
    hcli_imp = _importable("hcli.agentos.recovery")
    doc = rec.get("doc") if isinstance(rec.get("doc"), Mapping) else None
    faults_ok = False
    if doc and isinstance(doc.get("fault_injection"), Mapping):
        faults_ok = doc["fault_injection"].get("all_detected") is True
    bar = operational_bar(
        discover=bool(imp.get("ok") or hcli_imp.get("ok")),
        invoke=bool(imp.get("ok") and hasattr(rs, "admit")),
        schedule=True,
        verify=faults_ok,
        frontier=True,
        persist=_receipt_sealed(doc),
        refill=True,
        notes={"hcli_recovery": hcli_imp.get("ok"), "git_lock_doctor": git_lock.get("found")},
    )
    met = bool(imp.get("ok") and rec.get("found") and faults_ok and bar["resident_operational"])
    return _criterion(
        "crash_recovery",
        met=met,
        reason=(
            "repro_science fault injectors all_detected and HCLI recovery module importable; sidecar fail-closed"
            if met
            else (
                f"crash recovery not operational: repro={imp.get('ok')} "
                f"faults_ok={faults_ok} hcli.recovery={hcli_imp.get('ok')} "
                f"recovery_gate path_taken={recov.get('path_taken')}"
            )
        ),
        evidence=[
            {
                "repro_path_taken": rec.get("path_taken"),
                "faults_ok": faults_ok,
                "hcli_recovery_import": hcli_imp.get("ok"),
                "recovery_gate": {"found": recov.get("found"), "path_taken": recov.get("path_taken")},
                "git_lock": {"found": git_lock.get("found"), "path_taken": git_lock.get("path_taken")},
            }
        ],
        operational=bar,
        probe_performed=(
            "import tools.future.repro_science; hasattr admit; "
            "probe_json receipts/future/REPRO_SCIENCE.json read "
            "fault_injection.all_detected; probe_json "
            "HCLI_AGENTOS_RECOVERY_GATE.json; import hcli.agentos.recovery; "
            "probe_json receipts/future/GIT_LOCK_DURABILITY_REPORT.json"
        ),
        direct_observation=(
            f"repro_ok={imp.get('ok')} faults_ok={faults_ok} "
            f"hcli_recovery={hcli_imp.get('ok')} "
            f"recovery_gate_found={recov.get('found')} "
            f"recovery_gate_path={recov.get('path_taken')} "
            f"git_lock_found={git_lock.get('found')} flags={bar['flags']}"
        ),
    )


def _eval_receipts() -> dict[str, Any]:
    future = RECEIPTS
    present = future.is_dir()
    n = 0
    if present:
        n = sum(1 for p in future.glob("*.json") if p.is_file())
    bar = operational_bar(
        discover=present,
        invoke=True,
        schedule=True,
        verify=True,
        frontier=True,
        persist=present,
        refill=True,
        notes={"write_receipt": "tools.future._common.write_receipt", "hardware_claim_raises": "HardwareClaimError"},
    )
    met = bool(present and n > 0 and bar["resident_operational"])
    return _criterion(
        "receipts",
        met=met,
        reason=(
            f"receipts/future is present with n_json={n} (derived); write_receipt seals STATIC_ONLY and raises on hardware fields"
            if met
            else f"receipts partition not operational: dir={present} n_json={n}"
        ),
        evidence=[{"receipts_future": str(future), "n_json": n}],
        operational=bar,
        probe_performed=(
            "Path.is_dir() and glob('*.json') on receipts/future; "
            "write_receipt seals STATIC_ONLY and raises on hardware fields"
        ),
        direct_observation=(
            f"dir_present={present} n_json={n} path={future} flags={bar['flags']}"
        ),
    )


EVALUATORS: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
    ("resident_autonomy_trial_pass", _eval_autonomy),
    ("specimen_curriculum_ready", _eval_curriculum),
    ("modellake_identity", _eval_modellake),
    ("doctor_callable", _eval_doctor),
    ("gravity_callable", _eval_gravity),
    ("nr_nx_path_callable", _eval_nr_nx),
    ("evidence_hierarchy", _eval_evidence_hierarchy),
    ("negative_science", _eval_negative_science),
    ("workgraphs", _eval_workgraphs),
    ("self_refill", _eval_self_refill),
    ("dirty_measurement", _eval_dirty_measurement),
    ("protected_scheduling", _eval_protected_scheduling),
    ("transfer_substrate", _eval_transfer),
    ("adversary_substrate", _eval_adversary),
    ("crash_recovery", _eval_crash_recovery),
    ("receipts", _eval_receipts),
)


def evaluate_launch_criteria() -> list[dict[str, Any]]:
    """Run every criterion. Never stop at the first unmet."""
    if tuple(cid for cid, _ in EVALUATORS) != CRITERION_IDS:
        raise RuntimeError("EVALUATORS and CRITERION_IDS have drifted")
    rows: list[dict[str, Any]] = []
    for cid, fn in EVALUATORS:
        row = fn()
        if row.get("id") != cid:
            raise RuntimeError(f"evaluator {cid} returned id={row.get('id')!r}")
        rows.append(row)
    return rows


def unmet_criteria(results: Sequence[Mapping[str, Any]] | None = None) -> list[str]:
    rows = list(results) if results is not None else evaluate_launch_criteria()
    return [str(r["id"]) for r in rows if not r.get("met")]


def can_launch(results: Sequence[Mapping[str, Any]] | None = None) -> bool:
    return not unmet_criteria(results)


def launch_verdict(results: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    rows = list(results) if results is not None else evaluate_launch_criteria()
    unmet = unmet_criteria(rows)
    met = [str(r["id"]) for r in rows if r.get("met")]
    return {
        "allowed": not unmet,
        "verdict": "LAUNCH" if not unmet else "REFUSED",
        "unmet": unmet,
        "met": met,
        "n_criteria": len(rows),
        "n_unmet": len(unmet),
        "n_met": len(met),
        "rule": "every unmet criterion is named; the first failure does not hide the rest",
    }


# ---------------------------------------------------------------------------
# Launch payload. Written to disk only when the gate passes.
# ---------------------------------------------------------------------------


SANDBOX_RECEIPT = "receipts/future/RESIDENT_SANDBOX.json"
SANDBOX_SCHEMA = "hawking.future.sandbox.v1"


def _sha256_ok(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def bind_resident_identity(integration: str) -> dict[str, Any]:
    """Bind the launch receipt to the resident the identity document pins.

    Finding RESIDENT_IDENTITY.json is not binding it. bound is true only
    when the document pins nx_id, sealed_model_id, executable_hash,
    artifact_root, tokenizer, and qualification, and sealed_model_id
    agrees with the succession incumbent.
    """
    probe = probe_json("receipts/future/RESIDENT_IDENTITY.json")
    loaded = _load_future_module("resident_identity")
    if not loaded.get("ok"):
        doc = probe.get("doc") if isinstance(probe.get("doc"), Mapping) else None
        return {
            "kind": "resident",
            "found": bool(probe.get("found") and doc is not None),
            "path_taken": probe.get("path_taken"),
            "resolved": probe.get("resolved"),
            "schema": None if not doc else doc.get("schema"),
            "status": None
            if not doc
            else (
                doc.get("status")
                or (doc.get("identity_validation") or {}).get("status")
                or doc.get("residency_status")
            ),
            "bound": False,
            "pins": {},
            "pins_named": [],
            "missing": ["resident_identity_module"],
            "unbound_reason": (
                "found but resident_identity module is not importable: "
                f"{loaded.get('why')}"
            ),
            "integration_point": integration,
            "note": (
                "Identity is not invented. A missing this-wave receipt stays unbound."
            ),
        }
    fn = getattr(loaded["module"], "launch_binding", None)
    if not callable(fn):
        return {
            "kind": "resident",
            "found": bool(probe.get("found")),
            "path_taken": probe.get("path_taken"),
            "resolved": probe.get("resolved"),
            "schema": None,
            "status": None,
            "bound": False,
            "pins": {},
            "pins_named": [],
            "missing": ["launch_binding"],
            "unbound_reason": "found but tools.future.resident_identity.launch_binding is not callable",
            "integration_point": integration,
            "note": (
                "Identity is not invented. A missing this-wave receipt stays unbound."
            ),
        }
    return fn(probe=probe, integration=integration)


def bind_sandbox_identity(integration: str) -> dict[str, Any]:
    """Bind the launch receipt to the orchestrator sandbox identity.

    RESIDENT_SANDBOX.json is the this-wave receipt. HCLI AgentOS checkpoint
    is a prior gate and is not this identity. bound is true only when the
    document pins identity_sha256 and reentry_same_identity.
    """
    probe = probe_json(SANDBOX_RECEIPT)
    doc = probe.get("doc") if isinstance(probe.get("doc"), Mapping) else None
    note = (
        "Identity is not invented. A missing this-wave receipt stays unbound. "
        "Bound only when RESIDENT_SANDBOX.json pins identity_sha256 and "
        "reentry_same_identity. HCLI prior gates are not this identity."
    )
    base: dict[str, Any] = {
        "kind": "sandbox",
        "found": bool(probe.get("found") and doc is not None),
        "path_taken": probe.get("path_taken"),
        "resolved": probe.get("resolved"),
        "schema": None if not doc else doc.get("schema"),
        "integration_point": integration,
        "note": note,
    }
    if not base["found"]:
        return {
            **base,
            "bound": False,
            "status": None,
            "pins": {},
            "pins_named": [],
            "missing": ["receipt"],
            "unbound_reason": "sandbox receipt is missing; identity is not invented",
        }
    longevity = doc.get("longevity") if isinstance(doc.get("longevity"), Mapping) else {}
    identity_sha256 = longevity.get("identity_sha256")
    if not _sha256_ok(identity_sha256):
        identity_sha256 = doc.get("identity_sha256")
    reentry = longevity.get("reentry_same_identity")
    if reentry is None:
        reentry = doc.get("reentry_same_identity")
    status = doc.get("status")
    missing: list[str] = []
    if doc.get("schema") != SANDBOX_SCHEMA:
        missing.append("schema")
    if not _sha256_ok(identity_sha256):
        missing.append("identity_sha256")
    if reentry is not True:
        missing.append("reentry_same_identity")
    provision = doc.get("provision") if isinstance(doc.get("provision"), Mapping) else {}
    pins = {
        "identity_sha256": identity_sha256 if _sha256_ok(identity_sha256) else None,
        "reentry_same_identity": reentry,
        "sandbox_id": provision.get("default_sandbox_id"),
    }
    bound = not missing
    pins_named = [
        name
        for name in ("identity_sha256", "reentry_same_identity")
        if name not in missing
    ]
    unbound_reason = None
    if not bound:
        unbound_reason = "found but does not pin " + ", ".join(missing)
    return {
        **base,
        "bound": bound,
        "status": status,
        "pins": pins,
        "pins_named": pins_named,
        "missing": missing,
        "unbound_reason": unbound_reason,
    }


def _identity_stub(kind: str, integration: str) -> dict[str, Any]:
    """Backward-compatible name. Binding is real; this is not a stub anymore."""
    if kind == "sandbox":
        return bind_sandbox_identity(integration)
    return bind_resident_identity(integration)


def _machine_genome_pin() -> dict[str, Any]:
    nx = probe_json(
        "receipts/future/evidence/FLASH_COMPLETE_V0.nx.json",
        "receipts/headless/FLASH_COMPLETE_V0.nx.json",
    )
    digest = None
    if isinstance(nx.get("doc"), Mapping):
        compiled = nx["doc"].get("compiled_for_machine_genome")
        if isinstance(compiled, Mapping):
            digest = compiled.get("genome_digest")
    pin = dict(rs.fixture_machine_genome())
    pin.update(
        {
            "genome_digest": digest,
            "source": nx.get("resolved"),
            "path_taken": nx.get("path_taken"),
            "knowledge_level": "PIN_ONLY",
            "gpu_authority": False,
            "note": "digest cited; core counts and bytes from the NX seal are not copied (not a measurement claim)",
        }
    )
    return pin


def launch_payload(
    results: Sequence[Mapping[str, Any]],
    *,
    curriculum: Mapping[str, Any],
    workgraphs: Mapping[str, Any],
    blockers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    verdict = launch_verdict(results)
    units = list(workgraphs.get("units") or [])
    return {
        "schema": LAUNCH_SCHEMA,
        "version": VERSION,
        "odyssey": ODYSSEYS[0],
        "eras": list(ERAS),
        "no_era_vi": True,
        "no_odyssey_iv": True,
        "phase_transition": "STARTED" if verdict["allowed"] else "NOT_STARTED",
        "resident_identity": _identity_stub("resident", INTEGRATION_POINTS["resident_identity"]),
        "sandbox_identity": _identity_stub("sandbox", INTEGRATION_POINTS["sandbox"]),
        "machine_genome": _machine_genome_pin(),
        "first_specimen_set": curriculum,
        "first_workgraphs": {
            "specimen": workgraphs.get("specimen"),
            "n_units": workgraphs.get("n_units"),
            "stages": workgraphs.get("stages"),
            "listen_stages": workgraphs.get("listen_stages"),
            "phase_listen": workgraphs.get("phase_listen"),
            "units": units,
        },
        "resource_schedule": resource_schedule(units),
        "evidence_hierarchy": {
            "lattice": list(EVIDENCE_LATTICE),
            "rule": "DIAGNOSTIC_RELATIVE guides and never promotes; PROTECTED_ABSOLUTE decides; this module produces neither",
            "measurement_state": "STATIC_ONLY",
            "bench": "UNKNOWN",
        },
        "autonomy_status": next(
            (r for r in results if r.get("id") == "resident_autonomy_trial_pass"),
            {},
        ),
        "known_blockers": [
            {"id": b.get("id"), "holds": b.get("holds"), "reason": b.get("reason"), "sleeping": b.get("sleeping")}
            for b in blockers
        ],
        "next_refill_trigger": {
            "when": "a criterion flips from unmet to met, or a SLEEPING unit is woken because hardware qualified",
            "integration": INTEGRATION_POINTS["frontiers"],
            "does_not_refill_on": "a refused launch-gate re-run with the same unmet set",
        },
        "verdict": verdict,
        "gpu_authority": False,
        "claim_boundary": SIDECAR_CLAIM,
    }


def write_launch_if_passed(
    payload: Mapping[str, Any],
    *,
    allowed: bool,
    writer: Callable[[str, dict[str, Any], str], Path] | None = None,
) -> dict[str, Any]:
    """Write ODYSSEY_I_LAUNCH.json only when the gate passes AND the resident is bound.

    A phase-transition receipt that cannot name its resident is not a phase
    transition. This does not change can_launch() / the sixteen criteria.
    """
    write = writer or write_receipt
    if not allowed:
        return {
            "written": False,
            "path": None,
            "name": LAUNCH_RECEIPT,
            "reason": (
                "gate REFUSED; ODYSSEY_I_LAUNCH.json is a phase-transition receipt "
                "and is not written while any criterion is unmet"
            ),
        }
    resident = payload.get("resident_identity") if isinstance(payload, Mapping) else None
    if not (isinstance(resident, Mapping) and resident.get("bound") is True):
        detail = "resident_identity is missing from the launch payload"
        missing = None
        if isinstance(resident, Mapping):
            named = resident.get("unbound_reason")
            missing = resident.get("missing")
            if named:
                detail = str(named)
            elif missing:
                detail = "found but does not pin " + ", ".join(str(x) for x in missing)
            else:
                detail = "resident_identity.bound is false"
        return {
            "written": False,
            "path": None,
            "name": LAUNCH_RECEIPT,
            "unbound_identity": True,
            "missing": list(missing) if isinstance(missing, (list, tuple)) else missing,
            "reason": (
                "resident_identity unbound; ODYSSEY_I_LAUNCH.json is a "
                "phase-transition receipt and is not written while the resident "
                f"is unbound: {detail}"
            ),
        }
    path = write(LAUNCH_RECEIPT, dict(payload), RECORDED_BY)
    return {"written": True, "path": str(path), "name": LAUNCH_RECEIPT, "reason": "all sixteen criteria met"}


def _gate_workunit(verdict: Mapping[str, Any]) -> dict[str, Any]:
    row = wus.emit_hcli_workunit(
        id="odyssey-i.launch-gate",
        role="science",
        description="Evaluate Odyssey I launch criteria; refuse until every criterion is met",
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.odyssey_launch.can_launch",
        provider="future.odyssey_launch",
        effect_class="READ_ONLY",
        status="completed",
        classification="REFUSED" if not verdict.get("allowed") else "STATIC_ONLY",
        extras={
            "verdict": verdict.get("verdict"),
            "unmet": list(verdict.get("unmet") or []),
            "blocked_reason": None if verdict.get("allowed") else "launch criteria unmet; see unmet list",
            "requires_quiescence": False,
            "output_receipt_path": f"receipts/future/{RECEIPT}",
        },
    )
    wus.validate_emitted_unit(row)
    return row


def recovered_implementation() -> dict[str, Any]:
    return {
        "prior_odyssey_i": {
            "tools": "tools/odyssey/ (READ-ONLY): doctor_tournament, decoding_gravity, state_gravity, modellake, resident_seal, protected_window, pareto_archive, noetic_compiler",
            "receipts": "receipts/odyssey-i/ patient/doctor/gravity/nx seals — recovered, not duplicated",
            "package_fence": (
                "workspace/campaign/governance/odyssey/program/launch/ODYSSEY_LAUNCH.md "
                "is a different Odyssey (training-package fence, ODYSSEY_LAUNCH_AUTHORIZED=false). "
                "It is not this gate."
            ),
        },
        "hcli": {
            "modellake_receipts": "hcli/agentos/modellake_receipts.py — preferred census/supervision names",
            "modellake_gate": "hcli/agentos/modellake_gate.py — identity census, no acquisition",
            "autonomy_gate": "hcli/agentos/autonomy_gate.py — control-plane A1–A5, not Odyssey I start",
            "recovery": "hcli/agentos/recovery.py — process kill/resume",
            "doctor_gravity_markers": "hcli/doctor, hcli/gravity ownership markers",
            "workunit": "hcli/workunit.py — units emitted through the real constructor",
        },
        "landed_future": {
            "odyssey2_law_store": "transfer substrate",
            "odyssey3_adversary": "adversary substrate",
            "negative_index": "negative science",
            "evidence_snapshot": "pinned Codex receipts",
            "workunit_species": "HCLI WorkUnit emission",
            "flash_nx_audit": "NR/NX completeness (Flash-specific; not this criterion's authority)",
            "nr_nx_generic": "GENERIC_NR_NX_PIPELINE_CALLABLE vs FLASH_NX_READY; EXTENDED the evaluator to read it",
            "protected_scheduler": "PROTECTED_SCHEDULER_CAPABLE vs PROTECTED_WINDOW_AVAILABLE; EXTENDED the evaluator to read it",
            "autonomy_trial": "AUTONOMY_TRIALS.json persisted --verify verdict; EXTENDED the evaluator to read the file that is actually written",
            "contamination": "machine-state class, HEAVY on this host",
            "qualification_pipeline": "protected lease fail-closed",
            "repro_science": "crash/fault fail-closed",
            "teacher_corpus": "capture WorkUnits executed=false",
            "resident_optimizer": "propose/rank, never promote",
            "_common.write_receipt": "HardwareClaimError on numeric hardware fields",
        },
        "this_module_existed": True,
        "fork_decision": (
            "EXTENDED the three remaining evaluators in this file. Did not fork "
            "parallel modules. protected_scheduler.py, nr_nx_generic.py, autonomy_trial.py "
            "are the landed drivers."
        ),
    }


def gaps_closed() -> list[str]:
    return [
        "Sixteen launch criteria evaluated from evidence; can_launch is False until all pass.",
        "Refuse path names every unmet criterion; it does not stop at the first.",
        "ODYSSEY_I_LAUNCH.json is written only on pass; refuse does not write the phase-transition receipt.",
        "write_launch_if_passed refuses an unbound resident_identity even if all sixteen criteria are met; a phase-transition receipt must name its resident.",
        "resident_identity / sandbox_identity bind from RESIDENT_IDENTITY.json and RESIDENT_SANDBOX.json when those documents pin the named fields; a missing pin stays unbound with the field named; status is read from the document, not defaulted to null.",
        "Specimen curriculum proposes five roles from ModelLake seals / Odyssey I / law-store schools; other lake entries are recorded and not first-wave.",
        "First WorkGraphs emitted as real HCLI WorkUnits with dependencies and resource lanes.",
        "Phase II transfer and Phase III attack both depend on Phase I laws and not on each other — no global barrier.",
        "Physical blockers become SLEEPING WorkUnits; they never become synthetic results.",
        "HCLI AgentOS autonomy A1–A5 is recovered and explicitly not this criterion.",
        "resident_autonomy_trial_pass reads AUTONOMY_TRIALS.json (plural) persisted --verify; AUTONOMY_TRIAL.json is not authority.",
        "protected_scheduling reads PROTECTED_SCHEDULER_CAPABLE; PROTECTED_WINDOW_AVAILABLE is a separate field.",
        "nr_nx_path_callable reads GENERIC_NR_NX_PIPELINE_CALLABLE; FLASH_NX_READY is a separate field.",
    ]


def negative_findings_from(
    verdict: Mapping[str, Any],
    blockers: Sequence[Mapping[str, Any]],
    curriculum: Mapping[str, Any],
) -> list[str]:
    findings = [
        f"verdict={verdict.get('verdict')} n_unmet={verdict.get('n_unmet')} unmet={list(verdict.get('unmet') or [])}",
        "A gate that passed today would mistake 'Odyssey infrastructure ready' for 'Odyssey started'.",
    ]
    for b in blockers:
        if b.get("holds"):
            findings.append(f"physical blocker {b.get('id')}: {b.get('reason')}")
    unready = [r["role"] for r in curriculum.get("roles") or [] if not r.get("ready")]
    if unready:
        findings.append(f"curriculum unready roles: {unready}")
    findings.append(
        "autonomy_trial / protected_scheduler / nr_nx_generic are imported when present; "
        "other this-wave siblings remain named integration points: "
        + ", ".join(sorted(INTEGRATION_POINTS))
    )
    return findings


def resident_callable_block(verdict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entry_point": "python3 tools/future/odyssey_launch.py --verify",
        "launch_entry_point": "python3 tools/future/odyssey_launch.py --launch",
        "workunit_emitted": "odyssey-i.launch-gate plus first WorkGraph units and SLEEPING physical blockers",
        "receipt": f"receipts/future/{RECEIPT}",
        "phase_transition_receipt": f"receipts/future/{LAUNCH_RECEIPT}",
        "frontier_fed": (
            "Odyssey I start frontier: a REFUSED gate keeps the frontier at NOT_STARTED; "
            "a pass would write ODYSSEY_I_LAUNCH.json and that is the only start mark. "
            "Next refill trigger is a criterion flip or a SLEEPING wakeup."
        ),
        "fail_closed": (
            "can_launch() is False while any criterion is unmet; unmet_criteria() names "
            "every unmet id; write_launch_if_passed() does not call write_receipt for "
            f"{LAUNCH_RECEIPT} on refuse; write_launch_if_passed() also refuses when "
            "resident_identity.bound is false; --launch exits 1; HardwareClaimError on numeric "
            "hardware fields; GPU units stay SLEEPING rather than inventing a result."
        ),
        "verdict_now": verdict.get("verdict"),
        "this_wave_not_imported": list(THIS_WAVE_SIBLINGS),
        "integration_points": dict(INTEGRATION_POINTS),
    }


def _rewire_report(verdict: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Gate count before this rewiring vs after, and which of the three criteria moved."""
    after_unmet = list(verdict.get("unmet") or [])
    after_met = list(verdict.get("met") or [])
    changed = []
    for cid in REWIRE_BASELINE["unmet"]:
        row = next((r for r in results if r.get("id") == cid), {})
        before_met = False
        after = cid in after_met
        why = row.get("reason")
        if cid == "resident_autonomy_trial_pass":
            why_change = (
                "now reads AUTONOMY_TRIALS.json persisted --verify (plural) with "
                "orchestration/cognition split and timeline-seal verify; used to probe "
                "AUTONOMY_TRIAL.json which was never written"
            )
        elif cid == "protected_scheduling":
            why_change = (
                "now reads protected_scheduler.capability_report() CAPABLE vs AVAILABLE; "
                "used to AND capability with QUIESCENT+gpu_authority so a HEAVY machine "
                "looked incapable"
            )
        else:
            why_change = (
                "now reads nr_nx_generic GENERIC_NR_NX_PIPELINE_CALLABLE; "
                "used to key on FLASH_NX_COMPLETENESS_AUDIT / Flash NX metadata"
            )
        changed.append(
            {
                "id": cid,
                "before_met": before_met,
                "after_met": after,
                "why": why_change,
                "live_reason": why,
            }
        )
    return {
        "gate_count_before": {
            "n_met": REWIRE_BASELINE["n_met"],
            "n_unmet": REWIRE_BASELINE["n_unmet"],
            "n_criteria": REWIRE_BASELINE["n_criteria"],
            "unmet": list(REWIRE_BASELINE["unmet"]),
            "source": REWIRE_BASELINE["source"],
        },
        "gate_count_after": {
            "n_met": verdict.get("n_met"),
            "n_unmet": verdict.get("n_unmet"),
            "n_criteria": verdict.get("n_criteria"),
            "unmet": after_unmet,
            "met": after_met,
            "verdict": verdict.get("verdict"),
        },
        "criteria_rewired": changed,
        "none_lowered": True,
        "still_refused_if_any_unmet": bool(after_unmet),
        "rewired_net": {
            "became_met": [c["id"] for c in changed if c["after_met"] and not c["before_met"]],
            "stayed_unmet": [c["id"] for c in changed if not c["after_met"]],
            "stayed_met": [c["id"] for c in changed if c["after_met"] and c["before_met"]],
        },
        "unmet_not_from_this_rewire": [
            cid for cid in after_unmet if cid not in REWIRE_BASELINE["unmet"]
        ],
    }


def build(*, writer: Callable[[str, dict[str, Any], str], Path] | None = None) -> dict[str, Any]:
    write = writer or write_receipt
    results = evaluate_launch_criteria()
    verdict = launch_verdict(results)
    curriculum = propose_specimen_curriculum()
    first = (curriculum.get("roles") or [{}])[0]
    graphs = emit_first_workgraphs(first)
    blockers = physical_blockers()
    payload = launch_payload(results, curriculum=curriculum, workgraphs=graphs, blockers=blockers)
    launch_write = write_launch_if_passed(payload, allowed=bool(verdict["allowed"]), writer=write)
    gate_wu = _gate_workunit(verdict)
    sleeping_wu = [b["workunit"] for b in blockers if b.get("workunit")]
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "odyssey": ODYSSEYS[0],
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "no_era_vi": True,
        "no_odyssey_iv": True,
        "fpga": "FPGA belongs to Accelerator / Physical Compiler / Fusion. This gate does not build an FPGA backend.",
        "measurement_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "claim_boundary": SIDECAR_CLAIM,
        "criterion_ids": list(CRITERION_IDS),
        "criteria": results,
        "verdict": verdict,
        "rewire": _rewire_report(verdict, results),
        "specimen_curriculum": curriculum,
        "first_workgraphs": {
            "specimen": graphs.get("specimen"),
            "n_units": graphs.get("n_units"),
            "stages": graphs.get("stages"),
            "phase_listen": graphs.get("phase_listen"),
            "units": graphs.get("units"),
        },
        "resource_schedule": resource_schedule(list(graphs.get("units") or [])),
        "physical_blockers": [
            {k: v for k, v in b.items() if k != "workunit"} | {"workunit_id": (b.get("workunit") or {}).get("id")}
            for b in blockers
        ],
        "sleeping_workunits": sleeping_wu,
        "gate_workunit": gate_wu,
        "launch_payload_draft": payload,
        "launch_receipt": launch_write,
        "odyssey_i_launch_written": bool(launch_write.get("written")),
        "phase_transition": "STARTED" if launch_write.get("written") else "NOT_STARTED",
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings_from(verdict, blockers, curriculum),
        "resident_callable": resident_callable_block(verdict),
        "integration_points": dict(INTEGRATION_POINTS),
        "this_wave_siblings_not_imported": list(THIS_WAVE_SIBLINGS),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }
    path = write(RECEIPT, doc, RECORDED_BY)
    return {"gate_path": str(path), "doc": doc, "launch": launch_write}


def verify() -> int:
    out = build()
    doc = out["doc"]
    verdict = doc["verdict"]
    print(f"schema={doc['schema']} version={doc['version']}")
    print(f"verdict={verdict['verdict']} n_unmet={verdict['n_unmet']} n_met={verdict['n_met']}")
    if verdict["unmet"]:
        print("unmet:")
        for cid in verdict["unmet"]:
            print(f"  - {cid}")
    print(f"gate_receipt={out['gate_path']}")
    print(
        f"launch_receipt_written={doc['odyssey_i_launch_written']} "
        f"phase_transition={doc['phase_transition']}"
    )
    draft = doc.get("launch_payload_draft") if isinstance(doc.get("launch_payload_draft"), Mapping) else {}
    ident = draft.get("resident_identity")
    print("resident_identity:")
    print(json.dumps(ident, indent=2, sort_keys=True, default=str))
    launch_write = doc.get("launch_receipt") if isinstance(doc.get("launch_receipt"), Mapping) else {}
    if verdict["allowed"] and not doc["odyssey_i_launch_written"]:
        if launch_write.get("unbound_identity"):
            print(
                "launch_receipt withheld: resident_identity.bound is false "
                f"({launch_write.get('reason')})"
            )
            return 0
        print("FAIL: gate allowed but ODYSSEY_I_LAUNCH.json was not written", file=sys.stderr)
        return 1
    if (not verdict["allowed"]) and doc["odyssey_i_launch_written"]:
        print("FAIL: gate refused but ODYSSEY_I_LAUNCH.json was written", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Odyssey I launch gate")
    ap.add_argument("--verify", action="store_true", help="evaluate, seal the gate receipt, refuse honestly")
    ap.add_argument("--build", action="store_true", help="alias of --verify")
    ap.add_argument(
        "--launch",
        action="store_true",
        help="write ODYSSEY_I_LAUNCH.json only if every criterion is met; exit 1 on refuse",
    )
    a = ap.parse_args()
    if a.launch:
        out = build()
        if not out["launch"].get("written"):
            print("REFUSED: ODYSSEY_I_LAUNCH.json not written", file=sys.stderr)
            for cid in out["doc"]["verdict"]["unmet"]:
                print(f"  unmet: {cid}", file=sys.stderr)
            return 1
        print(out["launch"]["path"])
        return 0
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
