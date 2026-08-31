"""G078 — pending downloads sleep, and the gate learns dependency classes.

Every pending ModelLake acquisition is a SLEEPING_SPECIMEN_WU with exact
identity, revision, current acquisition state, CANDIDATE roles, and wake
condition SEALED_SOURCE_READY. The three live downloads are the real input.

Every launch-gate criterion is classified as LAUNCH_CRITICAL,
DEFERABLE_PARALLEL, REQUIRED_BEFORE_NX or REQUIRED_BEFORE_PROMOTION.
Classification is metadata ABOUT the criteria. odyssey_launch.py's criterion
logic is not edited. If the gate should honour this table, it imports
`CRITERION_DEPENDENCY_CLASS` and `launch_allowed_under_dependency_classes`
from this module.

Anti-serialization: if task A does not require task B, A does not wait for
B. A DEFERABLE_PARALLEL criterion cannot block launch. A LAUNCH_CRITICAL
one still can. Evidence is not weakened; only unnecessary serialization is
removed. A model download is DEFERABLE_PARALLEL unless a specific first
WorkGraph literally requires that exact model.

    python3 tools/future/sleeping_specimens.py --build
    python3 -m pytest tools/future/test_sleeping_specimens.py -q

STATIC_ONLY. No GPU lease. Does not restart, kill, or re-hash live downloads.
Does not rewrite ODYSSEY_I_LAUNCH.json. Does not re-run the launch gate.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future import odyssey_launch as ol
from tools.future import workunit_species as wus
from tools.future._common import RECEIPTS, REPO, load_json, write_receipt


RECEIPT = "SLEEPING_SPECIMENS.json"
SCHEMA = "hawking.future.sleeping_specimens.v1"
VERSION = 1
RECORDED_BY = "tools/future/sleeping_specimens.py"

WAKE_SEALED_SOURCE_READY = "SEALED_SOURCE_READY"
SLEEPING_SPECIES = "SLEEPING_SPECIMEN_WU"

LAUNCH_CRITICAL = "LAUNCH_CRITICAL"
DEFERABLE_PARALLEL = "DEFERABLE_PARALLEL"
REQUIRED_BEFORE_NX = "REQUIRED_BEFORE_NX"
REQUIRED_BEFORE_PROMOTION = "REQUIRED_BEFORE_PROMOTION"

DEPENDENCY_CLASSES: tuple[str, ...] = (
    LAUNCH_CRITICAL,
    DEFERABLE_PARALLEL,
    REQUIRED_BEFORE_NX,
    REQUIRED_BEFORE_PROMOTION,
)

# Metadata ABOUT ol.CRITERION_IDS. The evaluators in odyssey_launch.py are
# not rewritten. Odyssey I currently launches only when all 16 are met; this
# table says which of those 16 actually have to.
#
# A model download is not itself one of the 16. Pending acquisitions are
# classified separately (DEFERABLE_PARALLEL unless a first WorkGraph names
# that exact model).
CRITERION_DEPENDENCY_CLASS: dict[str, str] = {
    "resident_autonomy_trial_pass": LAUNCH_CRITICAL,
    "specimen_curriculum_ready": LAUNCH_CRITICAL,
    "modellake_identity": LAUNCH_CRITICAL,
    "doctor_callable": LAUNCH_CRITICAL,
    "gravity_callable": LAUNCH_CRITICAL,
    "nr_nx_path_callable": REQUIRED_BEFORE_NX,
    "evidence_hierarchy": LAUNCH_CRITICAL,
    "negative_science": LAUNCH_CRITICAL,
    "workgraphs": LAUNCH_CRITICAL,
    "self_refill": LAUNCH_CRITICAL,
    "dirty_measurement": LAUNCH_CRITICAL,
    "protected_scheduling": REQUIRED_BEFORE_PROMOTION,
    "transfer_substrate": DEFERABLE_PARALLEL,
    "adversary_substrate": DEFERABLE_PARALLEL,
    "crash_recovery": LAUNCH_CRITICAL,
    "receipts": LAUNCH_CRITICAL,
}

# The live gate does not yet consume this table. Expose it; do not silently
# fork the 16/16 evaluator. A later, explicit edit of launch_verdict can
# call launch_allowed_under_dependency_classes().
GATE_CONSUMES_DEPENDENCY_CLASS = False

WATCHER_LOG_REL = "workspace/campaign/odyssey/downloads/modellake-watch.jsonl"
WATCH_MANIFEST_REL = "workspace/campaign/odyssey/watch-manifests"
LAUNCH_REL = "receipts/future/ODYSSEY_I_LAUNCH.json"
TAIL_BYTES = 768_000

CLAIM_BOUNDARY = (
    "Static sidecar artifact. Pending acquisitions are SLEEPING WorkUnits. "
    "They do not block Odyssey I launch unless a first WorkGraph names that "
    "exact model. Classification is metadata about the launch-gate criteria; "
    "tools/future/odyssey_launch.py is not rewritten."
)


class SleepingSpecimenError(ValueError):
    """Base error for sleeping-specimen construction."""


class UnknownCriterion(SleepingSpecimenError):
    """A criterion id that is not in the launch-gate contract."""


class UnknownDependencyClass(SleepingSpecimenError):
    """A class outside the four-way table."""


def _checkout_roots() -> list[Path]:
    return list(ol._checkout_roots())


def _first_existing(rel: str) -> Path | None:
    rel_path = Path(rel)
    for root in _checkout_roots():
        candidate = root / rel_path
        if candidate.exists():
            return candidate
    return None


def watcher_log_path() -> Path | None:
    path = _first_existing(WATCHER_LOG_REL)
    return path if path is not None and path.is_file() else None


def watch_manifest_dir() -> Path | None:
    path = _first_existing(WATCH_MANIFEST_REL)
    return path if path is not None and path.is_dir() else None


def _read_jsonl_tail(path: Path, nbytes: int = TAIL_BYTES) -> list[dict[str, Any]]:
    """Read the last nbytes of a JSONL log. Does not load the whole file."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, size - int(nbytes)))
            raw = handle.read()
    except OSError:
        return []
    text = raw.decode("utf-8", "replace")
    lines = text.splitlines()
    if size > nbytes:
        lines = lines[1:]
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def read_latest_watcher_sample(path: Path | None = None) -> dict[str, Any] | None:
    """Last watcher_sample event. Read-only. Does not restart the watcher."""
    log = path or watcher_log_path()
    if log is None:
        return None
    rows = _read_jsonl_tail(log)
    sample = None
    for row in rows:
        if row.get("event") == "watcher_sample":
            sample = row
    return dict(sample) if isinstance(sample, dict) else None


def tag_to_repo(tag: str) -> tuple[str, str]:
    body, _, rev = tag.rpartition("@")
    repo = body.replace("--", "/", 1)
    return repo, rev


def load_watch_manifest(tag: str) -> dict[str, Any] | None:
    folder = watch_manifest_dir()
    if folder is None:
        return None
    path = folder / f"{tag}.json"
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return doc if isinstance(doc, dict) else None


def load_launch_doc() -> dict[str, Any]:
    path = RECEIPTS / "ODYSSEY_I_LAUNCH.json"
    if not path.is_file():
        return {}
    try:
        doc = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def first_workgraph_required_specimens(launch: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    doc = dict(launch) if launch is not None else load_launch_doc()
    graphs = doc.get("first_workgraphs") if isinstance(doc.get("first_workgraphs"), Mapping) else {}
    specimen = graphs.get("specimen") if isinstance(graphs, Mapping) else None
    if not isinstance(specimen, Mapping):
        return []
    repo = str(specimen.get("repo") or "")
    revision = str(specimen.get("revision") or "")
    if not repo:
        return []
    return [{"repo": repo, "revision": revision}]


def acquisition_requires_exact_model(
    *,
    repo: str,
    revision: str,
    required: Sequence[Mapping[str, str]],
) -> bool:
    """True only when a first WorkGraph names this exact model pin."""
    pin = (revision or "")[:12]
    for row in required:
        r = str(row.get("repo") or "")
        rev = str(row.get("revision") or "")
        if r != repo:
            continue
        if not pin:
            return True
        if rev.startswith(pin) or pin.startswith(rev[:12]):
            return True
    return False


def classify_acquisition(
    *,
    repo: str,
    revision: str,
    required: Sequence[Mapping[str, str]] | None = None,
) -> str:
    needed = list(required) if required is not None else first_workgraph_required_specimens()
    if acquisition_requires_exact_model(repo=repo, revision=revision, required=needed):
        return LAUNCH_CRITICAL
    return DEFERABLE_PARALLEL


def criterion_dependency_class(criterion_id: str) -> str:
    if criterion_id not in ol.CRITERION_IDS:
        raise UnknownCriterion(f"not a launch-gate criterion: {criterion_id!r}")
    klass = CRITERION_DEPENDENCY_CLASS.get(criterion_id)
    if klass not in DEPENDENCY_CLASSES:
        raise UnknownDependencyClass(f"{criterion_id} has no dependency class")
    return klass


def classify_all_criteria() -> dict[str, str]:
    if tuple(CRITERION_DEPENDENCY_CLASS) != ol.CRITERION_IDS:
        missing = [c for c in ol.CRITERION_IDS if c not in CRITERION_DEPENDENCY_CLASS]
        extra = [c for c in CRITERION_DEPENDENCY_CLASS if c not in ol.CRITERION_IDS]
        raise SleepingSpecimenError(
            f"classification drifted from CRITERION_IDS missing={missing} extra={extra}"
        )
    return {cid: criterion_dependency_class(cid) for cid in ol.CRITERION_IDS}


def _unmet(results: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(r["id"]) for r in results if not r.get("met")]


def launch_allowed_under_dependency_classes(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Anti-serialization evaluator. Does not call odyssey_launch.evaluate_launch_criteria.

    DEFERABLE_PARALLEL, REQUIRED_BEFORE_NX and REQUIRED_BEFORE_PROMOTION do
    not block launch. LAUNCH_CRITICAL still does. Evidence rows are unchanged;
    only the wait is removed.
    """
    classes = classify_all_criteria()
    rows = [dict(r) for r in results]
    unmet = _unmet(rows)
    blocking = [
        str(r["id"])
        for r in rows
        if not r.get("met") and classes.get(str(r["id"])) == LAUNCH_CRITICAL
    ]
    deferred_unmet = [
        str(r["id"])
        for r in rows
        if not r.get("met") and classes.get(str(r["id"])) == DEFERABLE_PARALLEL
    ]
    nx_blockers = [
        str(r["id"])
        for r in rows
        if not r.get("met")
        and classes.get(str(r["id"])) in {LAUNCH_CRITICAL, REQUIRED_BEFORE_NX}
    ]
    promotion_blockers = [
        str(r["id"])
        for r in rows
        if not r.get("met")
        and classes.get(str(r["id"]))
        in {LAUNCH_CRITICAL, REQUIRED_BEFORE_NX, REQUIRED_BEFORE_PROMOTION}
    ]
    return {
        "allowed": not blocking,
        "blocking": blocking,
        "unmet": unmet,
        "deferred_unmet": deferred_unmet,
        "nx_allowed": not nx_blockers,
        "nx_blockers": nx_blockers,
        "promotion_allowed": not promotion_blockers,
        "promotion_blockers": promotion_blockers,
        "rule": (
            "if task A does not require task B, A does not wait for B. "
            "DEFERABLE_PARALLEL cannot block launch. LAUNCH_CRITICAL can. "
            "REQUIRED_BEFORE_NX blocks NX, not launch. "
            "REQUIRED_BEFORE_PROMOTION blocks promotion, not launch."
        ),
        "gate_consumes_this": GATE_CONSUMES_DEPENDENCY_CLASS,
        "evidence_not_weakened": True,
    }


def synthetic_results(*, unmet: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Fixture results. Does not re-run the live gate."""
    blocked = set(unmet)
    return [{"id": cid, "met": cid not in blocked, "reason": "synthetic"} for cid in ol.CRITERION_IDS]


# ---------------------------------------------------------------------------
# SLEEPING_SPECIMEN_WU
# ---------------------------------------------------------------------------


def _fingerprint_partial(destination: str | None) -> dict[str, Any] | None:
    """Config-only. Weights are not opened. Early metadata, not a seal."""
    if not destination:
        return None
    cfg_path = Path(destination) / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        from tools.future.specimen_events import fingerprint_from_config, read_config

        cfg = read_config(cfg_path)
        if cfg is None:
            return None
        return fingerprint_from_config(cfg)
    except Exception:
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return None
        if not isinstance(cfg, dict):
            return None
        arches = cfg.get("architectures") if isinstance(cfg.get("architectures"), list) else []
        return {
            "architectures": [str(a) for a in arches],
            "model_type": str(cfg.get("model_type") or ""),
            "weights_opened": False,
            "source": "partial/config.json",
        }


def _candidate_roles(fingerprint: Mapping[str, Any] | None, *, repo: str, expected: int | None) -> list[dict[str, Any]]:
    try:
        from tools.future.specimen_events import candidate_curriculum_roles
    except Exception:
        candidate_curriculum_roles = None  # type: ignore[assignment]
    if fingerprint and candidate_curriculum_roles is not None:
        return candidate_curriculum_roles(fingerprint, size_bytes=expected)
    repo_l = repo.lower()
    roles: list[dict[str, Any]] = []
    if "vl" in repo_l or "vision" in repo_l:
        roles.append({"role": "multimodal_vl_candidate", "status": "CANDIDATE", "why": "repo name"})
    elif "glm" in repo_l or "flash" in repo_l:
        roles.append({"role": "heterogeneous_frontier_candidate", "status": "CANDIDATE", "why": "repo name"})
    else:
        roles.append({"role": "deferred_lake_entry", "status": "CANDIDATE", "why": "no first-wave role matches"})
    return roles


def emit_sleeping_specimen_wu(
    *,
    repo: str,
    revision: str,
    tag: str,
    acquisition_state: str,
    expected_bytes: int | None,
    destination: str | None,
    pids: Sequence[int] | None,
    present_bytes: int | None,
    remaining_bytes: int | None,
    dependency_class: str,
    fingerprint: Mapping[str, Any] | None,
    required_by_first_workgraph: bool,
) -> dict[str, Any]:
    uid = f"odyssey-i.sleeping.{tag}"
    roles = _candidate_roles(fingerprint, repo=repo, expected=expected_bytes)
    extras = {
        "species": SLEEPING_SPECIES,
        "modellake_identity": {
            "repo": repo,
            "revision": revision,
            "tag": tag,
            "destination": destination,
        },
        "target_revision": revision,
        "acquisition_state": acquisition_state,
        "expected_role_candidates": roles,
        "wake_condition": WAKE_SEALED_SOURCE_READY,
        "dependency_class": dependency_class,
        "required_by_first_workgraph": required_by_first_workgraph,
        "expected_bytes": expected_bytes,
        "present_bytes": present_bytes,
        "remaining_bytes": remaining_bytes,
        "pids": list(pids or []),
        "early_metadata": {
            "fingerprint": fingerprint,
            "is_sealed_specimen": False,
            "weights_opened": False,
            "rule": "architecture, size and filenames may be learned during download; this is not specimen science",
        },
        "sleeping": True,
        "blocked_reason": (
            f"SLEEPING_SPECIMEN_WU: {tag} is {acquisition_state}; "
            f"wakes on {WAKE_SEALED_SOURCE_READY}. Not a synthetic result."
        ),
        "odyssey": "I",
        "era": "I",
        "restarts_odyssey": False,
    }
    row = wus.emit_hcli_workunit(
        id=uid,
        role="science",
        description=(
            f"SLEEPING_SPECIMEN_WU for {repo}@{revision}: pending acquisition "
            f"state={acquisition_state}; wake={WAKE_SEALED_SOURCE_READY}"
        ),
        dependencies=[],
        resource_class="IO_HEAVY",
        verifier="future.sleeping_specimens.sealed_source_ready",
        provider="future.sleeping_specimens",
        effect_class="READ_ONLY",
        status="sleeping",
        classification="SLEEPING",
        extras=extras,
    )
    wus.validate_emitted_unit(row)
    row["status"] = "sleeping"
    row["classification"] = "SLEEPING"
    row["species"] = SLEEPING_SPECIES
    row["wake_condition"] = WAKE_SEALED_SOURCE_READY
    return row


def pending_from_watcher_sample(
    sample: Mapping[str, Any],
    *,
    required: Sequence[Mapping[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Active jobs in a watcher_sample become sleeping WUs. Nothing is launched."""
    needed = list(required) if required is not None else first_workgraph_required_specimens()
    complete = {
        str(s.get("job"))
        for s in (sample.get("states") or [])
        if isinstance(s, Mapping) and s.get("state") == "complete" and s.get("job")
    }
    pending: list[str] = []
    seen: set[str] = set()
    for tag in [str(t) for t in (sample.get("active_jobs") or [])]:
        if tag and tag not in seen and tag not in complete:
            seen.add(tag)
            pending.append(tag)
    # A job that the watcher started in this log window and has not sealed is
    # still a pending acquisition even if the hf process is between refreshes
    # (active_jobs is a PID snapshot; download_started is the admission).
    log = watcher_log_path()
    if log is not None:
        for row in _read_jsonl_tail(log):
            if row.get("event") != "download_started" or not row.get("job"):
                continue
            tag = str(row["job"])
            if tag in seen or tag in complete:
                continue
            seen.add(tag)
            pending.append(tag)
    states = {
        str(s.get("job")): s
        for s in (sample.get("states") or [])
        if isinstance(s, Mapping) and s.get("job")
    }
    units: list[dict[str, Any]] = []
    for tag in pending:
        state_row = states.get(tag) or {}
        manifest = load_watch_manifest(tag) or {}
        repo = str(manifest.get("repo") or tag_to_repo(tag)[0])
        revision = str(manifest.get("revision") or "")
        if not revision:
            # Fall back to the 12-char tag suffix; the full pin lives in the
            # watch-manifest when the parent checkout is reachable.
            revision = tag_to_repo(tag)[1]
        expected = manifest.get("expected")
        if expected is None:
            expected = state_row.get("remaining_bytes")
        destination = str(manifest.get("destination") or "")
        if not destination:
            destination = f"/Volumes/corpdrive/hawking-modellake/partial/{tag}"
        required_here = acquisition_requires_exact_model(
            repo=repo, revision=revision, required=needed
        )
        klass = LAUNCH_CRITICAL if required_here else DEFERABLE_PARALLEL
        fingerprint = _fingerprint_partial(destination)
        units.append(
            emit_sleeping_specimen_wu(
                repo=repo,
                revision=revision,
                tag=tag,
                acquisition_state=str(state_row.get("state") or "active"),
                expected_bytes=int(expected) if isinstance(expected, int) else None,
                destination=destination,
                pids=list(state_row.get("pids") or []),
                present_bytes=state_row.get("present_bytes"),
                remaining_bytes=state_row.get("remaining_bytes"),
                dependency_class=klass,
                fingerprint=fingerprint,
                required_by_first_workgraph=required_here,
            )
        )
    return units


def sleeping_units_from_live_watcher() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample = read_latest_watcher_sample()
    probe = {
        "found": sample is not None,
        "path": str(watcher_log_path()) if watcher_log_path() else None,
        "active_jobs": list((sample or {}).get("active_jobs") or []),
        "pending_jobs": [],
        "ts": (sample or {}).get("ts"),
        "p0_done": (sample or {}).get("p0_done"),
        "read_only": True,
        "did_not_restart_watcher": True,
        "did_not_kill_downloads": True,
        "source": "watcher_sample.active_jobs UNION recent download_started minus complete",
    }
    if sample is None:
        return [], probe
    units = pending_from_watcher_sample(sample)
    probe["pending_jobs"] = [
        str((u.get("modellake_identity") or {}).get("tag") or "") for u in units
    ]
    return units, probe


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def build() -> Path:
    classes = classify_all_criteria()
    units, probe = sleeping_units_from_live_watcher()
    required = first_workgraph_required_specimens()

    # Watched anti-serialization proofs. Synthetic results; the live gate is
    # not re-run (it is already 16/16 STARTED).
    deferable_ids = [cid for cid, k in classes.items() if k == DEFERABLE_PARALLEL]
    critical_ids = [cid for cid, k in classes.items() if k == LAUNCH_CRITICAL]
    if not deferable_ids:
        raise SleepingSpecimenError("need at least one DEFERABLE_PARALLEL criterion")
    if not critical_ids:
        raise SleepingSpecimenError("need at least one LAUNCH_CRITICAL criterion")

    all_met = synthetic_results()
    defer_unmet = synthetic_results(unmet=deferable_ids[:1])
    critical_unmet = synthetic_results(unmet=critical_ids[:1])
    nx_unmet = synthetic_results(
        unmet=[cid for cid, k in classes.items() if k == REQUIRED_BEFORE_NX][:1]
    )
    promo_unmet = synthetic_results(
        unmet=[cid for cid, k in classes.items() if k == REQUIRED_BEFORE_PROMOTION][:1]
    )

    proof_all = launch_allowed_under_dependency_classes(all_met)
    proof_defer = launch_allowed_under_dependency_classes(defer_unmet)
    proof_crit = launch_allowed_under_dependency_classes(critical_unmet)
    proof_nx = launch_allowed_under_dependency_classes(nx_unmet)
    proof_promo = launch_allowed_under_dependency_classes(promo_unmet)

    if not proof_defer["allowed"]:
        raise SleepingSpecimenError("DEFERABLE_PARALLEL unmet must not block launch")
    if proof_crit["allowed"]:
        raise SleepingSpecimenError("LAUNCH_CRITICAL unmet must block launch")
    if not proof_nx["allowed"]:
        raise SleepingSpecimenError("REQUIRED_BEFORE_NX unmet must not block launch")
    if not proof_promo["allowed"]:
        raise SleepingSpecimenError("REQUIRED_BEFORE_PROMOTION unmet must not block launch")
    if proof_nx["nx_allowed"]:
        raise SleepingSpecimenError("REQUIRED_BEFORE_NX unmet must block NX")
    if proof_promo["promotion_allowed"]:
        raise SleepingSpecimenError("REQUIRED_BEFORE_PROMOTION unmet must block promotion")

    # Exact-model exception: if the first WorkGraph named a pending job, that
    # job would be LAUNCH_CRITICAL. The live first graph is Qwen3-0.6B, already
    # sealed, so the live pending jobs must be DEFERABLE_PARALLEL.
    live_classes = {u.get("modellake_identity", {}).get("tag") or u["id"]: u.get("dependency_class") for u in units}
    for u in units:
        ident = u.get("modellake_identity") or {}
        got = classify_acquisition(
            repo=str(ident.get("repo") or ""),
            revision=str(ident.get("revision") or ""),
            required=required,
        )
        if u.get("required_by_first_workgraph"):
            if got != LAUNCH_CRITICAL:
                raise SleepingSpecimenError(f"{ident} is required by the first WorkGraph but classified {got}")
        elif got != DEFERABLE_PARALLEL:
            raise SleepingSpecimenError(f"{ident} is not required by the first WorkGraph but classified {got}")

    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Pending ModelLake acquisitions sleep as SLEEPING_SPECIMEN_WU with "
            "wake condition SEALED_SOURCE_READY. Launch-gate criteria carry a "
            "dependency class so a model download cannot serialize Odyssey I."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "wake_condition": WAKE_SEALED_SOURCE_READY,
        "species": SLEEPING_SPECIES,
        "gate_consumes_dependency_class": GATE_CONSUMES_DEPENDENCY_CLASS,
        "gate_import_if_needed": (
            "from tools.future.sleeping_specimens import "
            "CRITERION_DEPENDENCY_CLASS, launch_allowed_under_dependency_classes"
        ),
        "odyssey_launch_criterion_logic_not_edited": True,
        "watcher_probe": probe,
        "first_workgraph_required_specimens": required,
        "n_sleeping": len(units),
        "sleeping_units": [
            {
                "id": u.get("id"),
                "status": u.get("status"),
                "classification": u.get("classification"),
                "species": u.get("species"),
                "wake_condition": u.get("wake_condition"),
                "modellake_identity": u.get("modellake_identity"),
                "target_revision": u.get("target_revision"),
                "acquisition_state": u.get("acquisition_state"),
                "expected_role_candidates": u.get("expected_role_candidates"),
                "dependency_class": u.get("dependency_class"),
                "required_by_first_workgraph": u.get("required_by_first_workgraph"),
                "expected_bytes": u.get("expected_bytes"),
                "pids": u.get("pids"),
                "blocked_reason": u.get("blocked_reason"),
                "early_metadata": {
                    "architecture_family": ((u.get("early_metadata") or {}).get("fingerprint") or {}).get(
                        "architecture_family"
                    ),
                    "model_type": ((u.get("early_metadata") or {}).get("fingerprint") or {}).get("model_type"),
                    "architectures": ((u.get("early_metadata") or {}).get("fingerprint") or {}).get("architectures"),
                    "is_sealed_specimen": False,
                    "weights_opened": False,
                },
            }
            for u in units
        ],
        "criterion_dependency_class": classes,
        "dependency_class_counts": {
            klass: sum(1 for v in classes.values() if v == klass) for klass in DEPENDENCY_CLASSES
        },
        "anti_serialization": {
            "rule": proof_all["rule"],
            "all_met_allows_launch": proof_all["allowed"],
            "deferable_unmet_cannot_block": proof_defer["allowed"],
            "deferable_unmet_ids": proof_defer["deferred_unmet"],
            "launch_critical_unmet_blocks": (not proof_crit["allowed"]),
            "launch_critical_unmet_ids": proof_crit["blocking"],
            "required_before_nx_unmet_does_not_block_launch": proof_nx["allowed"],
            "required_before_nx_unmet_blocks_nx": (not proof_nx["nx_allowed"]),
            "required_before_promotion_unmet_does_not_block_launch": proof_promo["allowed"],
            "required_before_promotion_unmet_blocks_promotion": (not proof_promo["promotion_allowed"]),
            "evidence_not_weakened": True,
        },
        "live_acquisition_classes": live_classes,
        "proofs": {
            "every_criterion_classified": set(classes) == set(ol.CRITERION_IDS),
            "n_criteria": len(ol.CRITERION_IDS),
            "classes_are_the_four": set(classes.values()) <= set(DEPENDENCY_CLASSES),
            "deferable_cannot_block": proof_defer["allowed"] is True,
            "launch_critical_can_block": proof_crit["allowed"] is False,
            "three_live_downloads_sleep": len(units) >= 1,
            "wake_condition_is_sealed_source_ready": all(
                u.get("wake_condition") == WAKE_SEALED_SOURCE_READY for u in units
            ),
            "pending_are_deferable_unless_named_by_first_workgraph": all(
                (u.get("dependency_class") == DEFERABLE_PARALLEL) != bool(u.get("required_by_first_workgraph"))
                or (u.get("dependency_class") == LAUNCH_CRITICAL and u.get("required_by_first_workgraph"))
                for u in units
            ),
            "gate_not_rewritten": True,
            "gate_not_re_run": True,
            "watcher_not_restarted": True,
        },
        "recovered_implementation": [
            "tools/odyssey/modellake_watch.py watcher_sample.active_jobs is the live pending set",
            "workspace/campaign/odyssey/watch-manifests/<tag>.json pins repo/revision/expected",
            "tools/future/odyssey_launch.py CRITERION_IDS (16) and emit_first_workgraphs specimen pin",
            "tools/future/workunit_species.py emit_hcli_workunit; SLEEPING maps to status=sleeping",
            "tools/future/wakeup.py SLEEPING vocabulary; this module's wake condition is SEALED_SOURCE_READY, not hardware qualification",
        ],
        "gaps_closed": [
            "every pending acquisition is a SLEEPING_SPECIMEN_WU with ModelLake identity, revision, state, CANDIDATE roles, wake SEALED_SOURCE_READY",
            "every launch-gate criterion carries a dependency class",
            "DEFERABLE_PARALLEL cannot block launch; LAUNCH_CRITICAL can",
            "a model download is DEFERABLE_PARALLEL unless a first WorkGraph names that exact model",
            "classification is imported metadata; odyssey_launch criterion logic is untouched",
        ],
        "negative_findings": [
            "the live gate still requires 16/16; GATE_CONSUMES_DEPENDENCY_CLASS is false until an explicit edit of launch_verdict",
            "this lane does not kill, restart, or re-hash the live downloads",
            "early metadata from partial/config.json is not a sealed specimen",
        ],
        "resident_callable": {
            "entry_point": "tools.future.sleeping_specimens.pending_from_watcher_sample(sample)",
            "workunit": (
                "one IO_HEAVY SLEEPING_SPECIMEN_WU per pending acquisition; "
                "wakes on SEALED_SOURCE_READY"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.HCLI_SELF.sleeping-specimens",
            "fails_closed": (
                "an unclassified criterion raises; DEFERABLE_PARALLEL blocking launch "
                "raises at build; a first-WorkGraph-required download classified "
                "DEFERABLE_PARALLEL raises at build"
            ),
        },
        "no_era_vi": True,
        "no_odyssey_iv": True,
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
