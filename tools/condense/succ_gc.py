#!/usr/bin/env python3.12
"""Evidence-closed retirement and safe GC planner with receipts (master goal 13).

Space is first-class, but evidence is not disposable. This module is the successor
control plane's fail-closed garbage collector. It has two independent duties:

  retire_experiment  a PENDING experiment may leave the active set ONLY on one of the
                     five section-13.1 grounds, and retirement PRESERVES the plan
                     identity, rationale, supporting observations, uncertainty, affected
                     region, and reopening criterion. Retirement is a labelling act on
                     the plan record; it deletes nothing.

  gc_plan / gc_apply a source or scratch object may be DELETED only when all nine
                     section-13.2 conditions hold, and never for the six protected
                     classes (unique result artifacts, receipts, negative evidence,
                     required reproduction checkpoints, source identities, active or
                     unresolved source windows). gc_plan is a pure dry run: it emits a
                     sealed allowlist, dependency proof, exact byte/path manifest, and a
                     receipt-to-be. gc_apply performs the bounded deletion only under an
                     explicit go, using descriptor-relative no-follow handling so no
                     symlink or traversal can escape the deletion root, then fsyncs the
                     parent directory, verifies the object is gone, and seals a receipt.

Non-interference: nothing here writes under the campaign namespace
(reports/condense/doctor_v5_ultra). Successor state lives under
reports/condense/event_horizon_successor/gc. selftest runs fully offline in a tempdir
with synthetic candidates and never touches real campaign data or live processes.
"""
from __future__ import annotations

import dataclasses
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, seal_field, sealed, hash_value, now_iso,
    atomic_write_json, read_json_safe, is_sha256, repo_root,
)

# -- schema registry --------------------------------------------------------------------
SCHEMA_RETIREMENT = "hawking.successor.retirement.v1"
SCHEMA_GC_PLAN = "hawking.successor.gc_plan.v1"
SCHEMA_GC_RECEIPT = "hawking.successor.gc_receipt.v1"

# Successor-only state namespace. Deliberately NOT under doctor_v5_ultra (campaign owned).
GC_DIR = "reports/condense/event_horizon_successor/gc"

# A dependent experiment is settled when it is terminal or explicitly evidence-closed.
TERMINAL_STATUSES = frozenset({"complete", "negative", "unsupported", "evidence_closed"})

# Section 13.1: the ONLY grounds on which a pending experiment may be retired.
RETIREMENT_GROUNDS = frozenset({
    "dominated_replicated",                 # replicated evidence shows it is dominated
    "physically_impossible_under_budget",   # infeasible under the resource budget
    "equal_or_stronger_with_stricter_proof",  # a >= program already has stricter proof
    "cannot_change_frontier_conservative",  # cannot move the frontier under conservative bounds
    "explicit_incompatibility_receipt",     # an explicit incompatibility receipt exists
})

# Fields a retirement record must carry forward from the retired plan, unchanged.
PRESERVED_FIELDS = (
    "plan_identity", "rationale", "supporting_observations",
    "uncertainty", "affected_region", "reopening_criterion",
)

# Section 13.2 hard exclusions: classes that are NEVER deletable, whatever the gates say.
NEVER_DELETE_CLASSES = frozenset({
    "result_artifact",         # unique result artifacts
    "receipt",                 # receipts of any kind
    "negative_evidence",       # negative evidence is evidence
    "reproduction_checkpoint", # required reproduction checkpoints
    "source_identity",         # the identity of a source
    "active_source_window",    # active or unresolved source windows
})

# The nine section-13.2 conditions, in evaluation order (label -> human reason on failure).
GC_CONDITIONS: tuple[tuple[str, str], ...] = (
    ("dependents_settled", "a dependent experiment is not terminal or evidence-closed"),
    ("reporter_outputs_sealed", "required reporter outputs are not sealed"),
    ("evidence_covered_by_checkpoints", "queue accepted checkpoints do not cover the evidence"),
    ("no_queued_successor_needs_bytes", "a queued successor program still needs the bytes"),
    ("packed_durable_verified", "packed artifacts and results are not durable and verified"),
    ("deletion_manifest_exact", "an exact deletion manifest cannot be formed"),
    ("reacquisition_path_present", "a required reacquisition or rehydration path is missing"),
    ("operator_policy_permits_class", "operator policy does not permit deleting this class"),
    ("post_deletion_verifiable", "post-deletion verification cannot be guaranteed"),
)


class GcError(EcoError):
    """Fail-closed error in the successor GC / retirement planner."""


@dataclasses.dataclass(frozen=True)
class Config:
    """Where deletions may occur and which object classes operator policy permits."""

    root: Path                       # every candidate path is resolved relative to this
    gc_dir: Path                     # successor-only receipts/plans namespace
    operator_permitted_classes: frozenset[str] = frozenset({"source_scratch"})

    @property
    def plans_dir(self) -> Path:
        return self.gc_dir / "plans"

    @property
    def receipts_dir(self) -> Path:
        return self.gc_dir / "receipts"


def default_config() -> Config:
    root = repo_root()
    return Config(root=root, gc_dir=root / GC_DIR)


# ======================================================================================
# 13.1  evidence-closed retirement
# ======================================================================================
def retire_experiment(exp: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Retire a PENDING experiment iff one section-13.1 ground is proven.

    Returns a sealed retirement record that preserves plan identity, rationale,
    supporting observations, uncertainty, affected region, and reopening criterion.
    Raises GcError (fail closed) with all reasons when the experiment may not be retired.
    Retirement deletes nothing; it is a labelling act on the plan record.
    """
    reasons = _retirement_refusals(exp, evidence)
    if reasons:
        raise GcError("retirement refused: " + "; ".join(reasons))

    ground = evidence["ground"]
    record = {
        "schema": SCHEMA_RETIREMENT,
        "retired_at": now_iso(),
        "ground": ground,
        "preserved": {field: exp[field] for field in PRESERVED_FIELDS},
        "evidence": _retirement_evidence_summary(ground, evidence),
        "prior_status": exp.get("status", "pending"),
        "reversible": True,
        "note": "retirement labels the plan; evidence and observations are preserved, "
                "and the reopening criterion remains the path back into the active set",
    }
    return seal_field(record, "retirement_sha256")


def _retirement_refusals(exp: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not isinstance(exp, dict):
        return ["experiment is not an object"]
    if not isinstance(evidence, dict):
        return ["evidence is not an object"]

    status = exp.get("status", "pending")
    if status != "pending":
        reasons.append(f"experiment is not pending (status={status!r}); only pending plans retire")
    for field in PRESERVED_FIELDS:
        if field not in exp or exp[field] in (None, "", [], {}):
            reasons.append(f"cannot preserve required field {field!r}: absent or empty")

    ground = evidence.get("ground")
    if ground not in RETIREMENT_GROUNDS:
        reasons.append(f"ground {ground!r} is not one of the five section-13.1 grounds")
        return reasons  # ground-specific checks below are meaningless without a valid ground

    reasons.extend(_ground_specific_refusals(ground, evidence))
    return reasons


def _ground_specific_refusals(ground: str, evidence: dict[str, Any]) -> list[str]:
    """Each ground demands its own concrete, checkable support. Fail closed."""
    out: list[str] = []
    if ground == "dominated_replicated":
        reps = evidence.get("replications")
        if not isinstance(reps, list) or len(reps) < 2:
            out.append("dominated ground needs >= 2 independent replications")
        if not evidence.get("dominating_program"):
            out.append("dominated ground needs a named dominating program")
    elif ground == "physically_impossible_under_budget":
        if not is_sha256(evidence.get("budget_receipt_sha256")):
            out.append("impossible ground needs a budget receipt sha256")
        need = evidence.get("required_resource")
        have = evidence.get("available_resource")
        if not (isinstance(need, (int, float)) and isinstance(have, (int, float))):
            out.append("impossible ground needs numeric required/available resource")
        elif need <= have:
            out.append(f"impossible ground contradicted: required {need} <= available {have}")
    elif ground == "equal_or_stronger_with_stricter_proof":
        if not evidence.get("competitor_identity"):
            out.append("equal-or-stronger ground needs a competitor plan identity")
        if not is_sha256(evidence.get("stricter_proof_sha256")):
            out.append("equal-or-stronger ground needs a stricter-proof sha256")
    elif ground == "cannot_change_frontier_conservative":
        if not is_sha256(evidence.get("conservative_bound_sha256")):
            out.append("cannot-change-frontier ground needs a conservative-bound receipt sha256")
        if evidence.get("bound_direction") not in ("upper", "lower", "both"):
            out.append("cannot-change-frontier ground needs bound_direction upper/lower/both")
    elif ground == "explicit_incompatibility_receipt":
        if not is_sha256(evidence.get("incompatibility_receipt_sha256")):
            out.append("incompatibility ground needs an incompatibility receipt sha256")
    return out


def _retirement_evidence_summary(ground: str, evidence: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "dominated_replicated": ("replications", "dominating_program"),
        "physically_impossible_under_budget": (
            "budget_receipt_sha256", "required_resource", "available_resource"),
        "equal_or_stronger_with_stricter_proof": (
            "competitor_identity", "stricter_proof_sha256"),
        "cannot_change_frontier_conservative": (
            "conservative_bound_sha256", "bound_direction"),
        "explicit_incompatibility_receipt": ("incompatibility_receipt_sha256",),
    }[ground]
    return {field: evidence.get(field) for field in keep}


# ======================================================================================
# 13.2  safe GC planning (dry run)
# ======================================================================================
def _path_is_confined(rel: str) -> bool:
    """A candidate path must be relative, non-empty, and free of traversal / roots."""
    if not isinstance(rel, str) or not rel:
        return False
    p = Path(rel)
    if p.is_absolute():
        return False
    return ".." not in p.parts and not any(part in ("", ".") for part in p.parts)


def evaluate_candidate(
    candidate: dict[str, Any],
    dependency_index: dict[str, Any],
    config: Config,
) -> dict[str, Any]:
    """Evaluate one candidate against the hard exclusions + nine conditions.

    Returns a verdict dict: object_id, eligible, reasons, conditions (label->bool),
    dependency_proof, and (when eligible) the exact manifest entry (path, bytes, sha256).
    Never raises on candidate content; malformed candidates simply refuse.
    """
    reasons: list[str] = []
    object_id = candidate.get("object_id") if isinstance(candidate, dict) else None
    if not isinstance(candidate, dict) or not isinstance(object_id, str) or not object_id:
        return {"object_id": object_id, "eligible": False,
                "reasons": ["candidate malformed or missing object_id"],
                "conditions": {}, "dependency_proof": {}}

    object_class = candidate.get("object_class")

    # -- hard exclusion: the six never-delete classes (checked before any gate) ----------
    if object_class in NEVER_DELETE_CLASSES:
        reasons.append(f"protected class {object_class!r} is never deletable")

    rel_path = candidate.get("path")
    if not _path_is_confined(rel_path):
        reasons.append(f"path {rel_path!r} is not a confined relative path")

    # -- condition C1: every dependent experiment terminal / evidence-closed -------------
    dependents = candidate.get("dependents", [])
    dep_proof: dict[str, Any] = {}
    c1 = True
    if not isinstance(dependents, list):
        c1 = False
        reasons.append("dependents is not a list")
    else:
        for dep in dependents:
            row = dependency_index.get(dep) if isinstance(dependency_index, dict) else None
            if not isinstance(row, dict):
                c1 = False
                dep_proof[str(dep)] = {"known": False}
                reasons.append(f"dependent {dep!r} not found in dependency index")
                continue
            status = row.get("status")
            closed = bool(row.get("evidence_closed"))
            settled = status in TERMINAL_STATUSES or closed
            dep_proof[str(dep)] = {"known": True, "status": status,
                                   "evidence_closed": closed, "settled": settled}
            if not settled:
                c1 = False
                reasons.append(f"dependent {dep!r} not settled (status={status!r})")

    # -- conditions C2..C8: declared, verifiable predicates on the candidate -------------
    c2 = candidate.get("reporter_outputs_sealed") is True
    c3 = candidate.get("evidence_covered_by_checkpoints") is True
    c4 = candidate.get("needed_by_queued_successor") is False
    c5 = candidate.get("packed_durable_verified") is True

    # C6: an exact deletion manifest can be formed (confined path + byte count + sha256).
    manifest_bytes = candidate.get("bytes")
    manifest_sha = candidate.get("sha256")
    c6 = (_path_is_confined(rel_path)
          and isinstance(manifest_bytes, int) and manifest_bytes >= 0
          and is_sha256(manifest_sha))

    # C7: a reacquisition / rehydration path exists where required.
    if candidate.get("reacquisition_required") is True:
        c7 = bool(candidate.get("reacquisition_path"))
    else:
        c7 = True

    c8 = object_class in config.operator_permitted_classes

    # C9: post-deletion verification is guaranteed by descriptor-relative re-stat at apply
    # time; guaranteeable exactly when the path is confined (so the re-stat is unambiguous).
    c9 = _path_is_confined(rel_path)

    conditions = {
        "dependents_settled": c1,
        "reporter_outputs_sealed": c2,
        "evidence_covered_by_checkpoints": c3,
        "no_queued_successor_needs_bytes": c4,
        "packed_durable_verified": c5,
        "deletion_manifest_exact": c6,
        "reacquisition_path_present": c7,
        "operator_policy_permits_class": c8,
        "post_deletion_verifiable": c9,
    }
    for label, human in GC_CONDITIONS:
        if not conditions[label]:
            reasons.append(f"condition {label} failed: {human}")

    eligible = (object_class not in NEVER_DELETE_CLASSES
                and all(conditions.values())
                and _path_is_confined(rel_path))

    verdict: dict[str, Any] = {
        "object_id": object_id,
        "object_class": object_class,
        "eligible": eligible,
        "reasons": reasons,
        "conditions": conditions,
        "dependency_proof": dep_proof,
    }
    if eligible:
        verdict["manifest_entry"] = {
            "object_id": object_id,
            "object_class": object_class,
            "path": rel_path,
            "bytes": manifest_bytes,
            "sha256": manifest_sha,
            "reacquisition_path": candidate.get("reacquisition_path"),
        }
    return verdict


def gc_plan(
    candidates: list[dict[str, Any]],
    dependency_index: dict[str, Any],
    *,
    config: Config | None = None,
) -> dict[str, Any]:
    """Produce a sealed dry-run GC plan: allowlist, refusals, exact manifest, receipt-to-be.

    This function is side-effect free (it does not write or delete). Callers persist the
    plan with write_plan and hand it to gc_apply.
    """
    config = config or default_config()
    if not isinstance(candidates, list):
        raise GcError("candidates must be a list")
    dependency_index = dependency_index if isinstance(dependency_index, dict) else {}

    allowlist: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    dependency_proofs: dict[str, Any] = {}

    for candidate in candidates:
        verdict = evaluate_candidate(candidate, dependency_index, config)
        oid = verdict["object_id"]
        if verdict.get("dependency_proof"):
            dependency_proofs[str(oid)] = verdict["dependency_proof"]
        if verdict["eligible"]:
            allowlist.append({"object_id": oid, "object_class": verdict["object_class"],
                              "conditions": verdict["conditions"]})
            manifest.append(verdict["manifest_entry"])
        else:
            refused.append({"object_id": oid, "object_class": verdict.get("object_class"),
                            "reasons": verdict["reasons"]})

    manifest.sort(key=lambda row: row["path"])
    total_bytes = sum(int(row["bytes"]) for row in manifest)
    manifest_sha256 = hash_value(manifest)

    receipt_to_be = {
        "schema": SCHEMA_GC_RECEIPT,
        "status": "unsealed_template",
        "root": str(config.root),
        "manifest_sha256": manifest_sha256,
        "object_count": len(manifest),
        "total_bytes": total_bytes,
        "note": "operator acknowledges this manifest_sha256, then gc_apply(go=True) "
                "deletes exactly these paths under descriptor-relative no-follow handling",
    }

    plan = {
        "schema": SCHEMA_GC_PLAN,
        "generated_at": now_iso(),
        "root": str(config.root),
        "operator_permitted_classes": sorted(config.operator_permitted_classes),
        "never_delete_classes": sorted(NEVER_DELETE_CLASSES),
        "conditions_checked": [label for label, _ in GC_CONDITIONS],
        "allowlist": allowlist,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "dependency_proof": dependency_proofs,
        "refused": refused,
        "total_bytes": total_bytes,
        "receipt_to_be": receipt_to_be,
    }
    return seal_field(plan, "plan_sha256")


def write_plan(plan: dict[str, Any], config: Config | None = None) -> Path:
    """Persist a sealed plan under the successor-only namespace. Fails closed if unsealed."""
    config = config or default_config()
    if not sealed(plan, "plan_sha256"):
        raise GcError("refusing to write an unsealed plan")
    path = config.plans_dir / f"gc_plan_{plan['plan_sha256'][:16]}.json"
    atomic_write_json(path, plan)
    return path


# ======================================================================================
# 13.2  bounded application (descriptor-relative, no-follow)
# ======================================================================================
def _fsync_dir_fd(dir_fd: int) -> None:
    os.fsync(dir_fd)


def _open_parent_nofollow(root: Path, rel: str) -> tuple[int, str]:
    """Walk `rel` component-by-component from `root`, opening each intermediate directory
    with O_NOFOLLOW so no symlinked component can redirect the walk. Returns the parent
    directory fd and the final leaf name. Caller must os.close the returned fd.
    """
    if not _path_is_confined(rel):
        raise GcError(f"unconfined deletion path refused: {rel!r}")
    parts = Path(rel).parts
    # The root is a trusted config value; follow symlinks to reach it, then never again.
    dir_fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0))
    try:
        for component in parts[:-1]:
            nxt = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=dir_fd,
            )
            os.close(dir_fd)
            dir_fd = nxt
    except OSError as exc:
        os.close(dir_fd)
        raise GcError(f"symlinked or missing directory component in {rel!r}: {exc}") from exc
    return dir_fd, parts[-1]


def _safe_delete(root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """Delete exactly one manifest entry, descriptor-relative, refusing symlinks.

    Opens the parent no-follow, opens the leaf O_NOFOLLOW (so a symlinked leaf raises),
    confirms it is a regular file of the manifested size, unlinks via the parent fd,
    fsyncs the parent directory, and verifies the leaf is gone.
    """
    rel = entry["path"]
    dir_fd, leaf = _open_parent_nofollow(root, rel)
    try:
        try:
            leaf_fd = os.open(
                leaf, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), dir_fd=dir_fd)
        except OSError as exc:
            raise GcError(f"leaf is a symlink or unopenable, refusing: {rel!r}: {exc}") from exc
        try:
            info = os.fstat(leaf_fd)
        finally:
            os.close(leaf_fd)
        if not stat.S_ISREG(info.st_mode):
            raise GcError(f"leaf is not a regular file, refusing: {rel!r}")
        manifested = entry.get("bytes")
        if isinstance(manifested, int) and manifested != info.st_size:
            raise GcError(
                f"size drift for {rel!r}: manifest {manifested} != on-disk {info.st_size}")

        os.unlink(leaf, dir_fd=dir_fd)
        _fsync_dir_fd(dir_fd)

        # post-deletion verification: the leaf must no longer resolve under the parent.
        gone = False
        try:
            os.stat(leaf, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            gone = True
        if not gone:
            raise GcError(f"post-deletion verification failed, leaf still present: {rel!r}")
        return {"object_id": entry["object_id"], "path": rel,
                "bytes": info.st_size, "deleted": True}
    finally:
        os.close(dir_fd)


def gc_apply(
    plan: dict[str, Any],
    receipt: dict[str, Any],
    *,
    go: bool = False,
    config: Config | None = None,
) -> dict[str, Any]:
    """Apply a sealed plan. Deletes ONLY on an explicit go, and ONLY the allowlisted
    manifest paths, using descriptor-relative no-follow deletion. Emits a sealed receipt.

    `receipt` is the operator's acknowledgement: it must acknowledge the plan's
    plan_sha256 and manifest_sha256 and carry operator_ack True.
    """
    config = config or default_config()
    if not sealed(plan, "plan_sha256"):
        raise GcError("plan self-seal invalid; refusing to apply")
    if plan.get("schema") != SCHEMA_GC_PLAN:
        raise GcError(f"not a gc plan: schema={plan.get('schema')!r}")
    if str(plan.get("root")) != str(config.root):
        raise GcError(f"plan root {plan.get('root')!r} != config root {config.root!r}")

    manifest = plan.get("manifest", [])
    if not isinstance(manifest, list):
        raise GcError("plan manifest is not a list")
    # the sealed plan's manifest must match its declared manifest_sha256 (self-consistency)
    if hash_value(manifest) != plan.get("manifest_sha256"):
        raise GcError("plan manifest_sha256 does not match its manifest")

    if not isinstance(receipt, dict):
        raise GcError("operator acknowledgement receipt is not an object")
    if receipt.get("acknowledges_plan_sha256") != plan["plan_sha256"]:
        raise GcError("acknowledgement does not reference this plan_sha256")
    if receipt.get("acknowledges_manifest_sha256") != plan["manifest_sha256"]:
        raise GcError("acknowledgement does not reference this manifest_sha256")
    if receipt.get("operator_ack") is not True:
        raise GcError("operator_ack is not True")

    if not go:
        return {"schema": SCHEMA_GC_RECEIPT, "applied": False, "go": False,
                "would_delete": len(manifest),
                "would_free_bytes": plan.get("total_bytes", 0),
                "note": "dry run: pass go=True to delete the allowlisted manifest"}

    deleted: list[dict[str, Any]] = []
    for entry in manifest:
        deleted.append(_safe_delete(config.root, entry))

    freed = sum(int(row["bytes"]) for row in deleted)
    # File name is derived from the plan sha (known before sealing), so receipt_path can
    # be part of the sealed body and the returned record stays self-consistent.
    receipt_path = config.receipts_dir / f"gc_receipt_{plan['plan_sha256'][:16]}.json"
    out = {
        "schema": SCHEMA_GC_RECEIPT,
        "applied": True,
        "go": True,
        "sealed_at": now_iso(),
        "root": str(config.root),
        "plan_sha256": plan["plan_sha256"],
        "manifest_sha256": plan["manifest_sha256"],
        "operator_ack": True,
        "deleted": deleted,
        "object_count": len(deleted),
        "freed_bytes": freed,
        "receipt_path": str(receipt_path),
    }
    out = seal_field(out, "receipt_sha256")
    atomic_write_json(receipt_path, out)
    # read-back validation of the emitted receipt
    back = read_json_safe(receipt_path)
    if not sealed(back, "receipt_sha256"):
        raise GcError("emitted receipt failed read-back self-seal")
    return out


# ======================================================================================
# selftest (fully offline, tempdir, synthetic fixtures)
# ======================================================================================
def _closed_candidate(path: str, *, sha: str, object_class: str = "source_scratch",
                      dependents: list[str] | None = None, size: int = 4096,
                      **overrides: Any) -> dict[str, Any]:
    base = {
        "object_id": path,
        "object_class": object_class,
        "path": path,
        "bytes": size,
        "sha256": sha,
        "dependents": dependents if dependents is not None else ["exp-a"],
        "reporter_outputs_sealed": True,
        "evidence_covered_by_checkpoints": True,
        "needed_by_queued_successor": False,
        "packed_durable_verified": True,
        "reacquisition_required": True,
        "reacquisition_path": "hf://acme/model@rev",
    }
    base.update(overrides)
    return base


def selftest() -> dict[str, Any]:
    import tempfile

    checks: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            raise GcError(f"selftest failed: {name}")
        checks.append(name)

    sha_a = "a" * 64
    sha_b = "b" * 64

    with tempfile.TemporaryDirectory(prefix="succ-gc-selftest-") as raw:
        root = Path(raw) / "root"
        (root / "scratch").mkdir(parents=True)
        config = Config(root=root, gc_dir=Path(raw) / "state",
                        operator_permitted_classes=frozenset({"source_scratch"}))

        # -- 13.1 retirement: a valid ground with preserved fields succeeds --------------
        exp = {
            "status": "pending",
            "plan_identity": "plan/qwen72b/1.44bpw",
            "rationale": "scheduling-prior only; no measured pass",
            "supporting_observations": ["obs-1", "obs-2"],
            "uncertainty": {"bpw": 0.2},
            "affected_region": "72B sub-2bpw",
            "reopening_criterion": "reopen if collapse floor descends below 1.2bpw measured",
        }
        rec = retire_experiment(exp, {
            "ground": "dominated_replicated",
            "replications": ["run-1", "run-2"],
            "dominating_program": "plan/qwen72b/4bpw",
        })
        check("retire-seals", sealed(rec, "retirement_sha256"))
        check("retire-preserves-identity", rec["preserved"]["plan_identity"] == exp["plan_identity"])
        check("retire-preserves-reopen",
              rec["preserved"]["reopening_criterion"] == exp["reopening_criterion"])

        # invalid ground is refused
        refused = False
        try:
            retire_experiment(exp, {"ground": "i_feel_like_it"})
        except GcError:
            refused = True
        check("retire-refuses-bad-ground", refused)

        # missing preserved field is refused (evidence is otherwise valid)
        refused = False
        try:
            retire_experiment({k: v for k, v in exp.items() if k != "reopening_criterion"},
                              {"ground": "explicit_incompatibility_receipt",
                               "incompatibility_receipt_sha256": sha_a})
        except GcError:
            refused = True
        check("retire-refuses-lost-evidence", refused)

        # -- 13.2 dependency index -------------------------------------------------------
        dep_index = {
            "exp-a": {"status": "complete", "evidence_closed": True},
            "exp-open": {"status": "running", "evidence_closed": False},
        }

        # non-terminal dependent -> refused, absent from allowlist and manifest
        open_file = root / "scratch" / "open.bin"
        open_file.write_bytes(b"o" * 4096)
        plan_open = gc_plan(
            [_closed_candidate("scratch/open.bin", sha=sha_a, dependents=["exp-open"])],
            dep_index, config=config)
        check("plan-refuses-nonterminal-dependent",
              plan_open["allowlist"] == [] and len(plan_open["refused"]) == 1)
        check("plan-nonterminal-file-survives", open_file.exists())

        # never-delete class -> refused even with every gate green
        plan_protected = gc_plan(
            [_closed_candidate("scratch/keep.bin", sha=sha_a, object_class="receipt")],
            dep_index, config=config)
        check("plan-refuses-never-delete-class", plan_protected["allowlist"] == [])
        check("plan-never-delete-reason",
              any("protected class" in r for r in plan_protected["refused"][0]["reasons"]))

        # fully evidence-closed candidate -> produces a plan with one allowlisted entry
        target = root / "scratch" / "dead.bin"
        target.write_bytes(b"d" * 4096)
        plan_ok = gc_plan(
            [_closed_candidate("scratch/dead.bin", sha=sha_a)], dep_index, config=config)
        check("plan-admits-closed-candidate", len(plan_ok["allowlist"]) == 1)
        check("plan-manifest-exact",
              plan_ok["manifest"][0]["path"] == "scratch/dead.bin"
              and plan_ok["manifest"][0]["bytes"] == 4096)
        check("plan-self-seals", sealed(plan_ok, "plan_sha256"))
        written = write_plan(plan_ok, config)
        check("plan-persisted-under-config-gc-dir",
              written.exists() and config.plans_dir in written.parents
              and "doctor_v5_ultra" not in str(written))

        # gc_apply dry run (no go) deletes nothing
        ack = {
            "acknowledges_plan_sha256": plan_ok["plan_sha256"],
            "acknowledges_manifest_sha256": plan_ok["manifest_sha256"],
            "operator_ack": True,
        }
        dry = gc_apply(plan_ok, ack, go=False, config=config)
        check("apply-dry-run-noop", dry["applied"] is False and target.exists())

        # gc_apply on go deletes exactly the allowlisted file and writes a sealed receipt
        applied = gc_apply(plan_ok, ack, go=True, config=config)
        check("apply-go-deletes-target", not target.exists())
        check("apply-go-object-count", applied["object_count"] == 1)
        check("apply-receipt-sealed", sealed(applied, "receipt_sha256"))
        check("apply-receipt-written", Path(applied["receipt_path"]).exists())
        # sibling file that was never in the manifest is untouched
        check("apply-does-not-overreach", open_file.exists())

        # wrong acknowledgement is refused
        refused = False
        try:
            gc_apply(plan_ok, {"acknowledges_plan_sha256": "z" * 64,
                               "acknowledges_manifest_sha256": plan_ok["manifest_sha256"],
                               "operator_ack": True}, go=True, config=config)
        except GcError:
            refused = True
        check("apply-refuses-wrong-ack", refused)

        # -- symlinked candidate: descriptor-relative no-follow refuses at apply ----------
        real = root / "scratch" / "real_source.bin"
        real.write_bytes(b"r" * 4096)
        link_dir = root / "scratch"
        (link_dir / "link.bin").symlink_to(real)
        # build a plan manifest entry that points at the symlink, re-seal it, and apply
        evil_manifest = [{
            "object_id": "scratch/link.bin", "object_class": "source_scratch",
            "path": "scratch/link.bin", "bytes": 4096, "sha256": sha_b,
            "reacquisition_path": "hf://acme/model@rev",
        }]
        evil_plan = seal_field({
            "schema": SCHEMA_GC_PLAN, "generated_at": now_iso(), "root": str(config.root),
            "operator_permitted_classes": ["source_scratch"],
            "never_delete_classes": sorted(NEVER_DELETE_CLASSES),
            "conditions_checked": [label for label, _ in GC_CONDITIONS],
            "allowlist": [{"object_id": "scratch/link.bin"}],
            "manifest": evil_manifest, "manifest_sha256": hash_value(evil_manifest),
            "dependency_proof": {}, "refused": [], "total_bytes": 4096,
            "receipt_to_be": {"schema": SCHEMA_GC_RECEIPT},
        }, "plan_sha256")
        evil_ack = {
            "acknowledges_plan_sha256": evil_plan["plan_sha256"],
            "acknowledges_manifest_sha256": evil_plan["manifest_sha256"],
            "operator_ack": True,
        }
        refused = False
        try:
            gc_apply(evil_plan, evil_ack, go=True, config=config)
        except GcError:
            refused = True
        check("apply-refuses-symlink-leaf", refused)
        check("apply-symlink-target-survives", real.exists())
        check("apply-symlink-untouched", (link_dir / "link.bin").is_symlink())

        # -- traversal path is refused outright ------------------------------------------
        refused = False
        try:
            _open_parent_nofollow(config.root, "../escape.bin")
        except GcError:
            refused = True
        check("apply-refuses-traversal", refused)

    return {"ok": True, "checks": checks, "check_count": len(checks)}


if __name__ == "__main__":
    print(json.dumps(selftest(), indent=2, sort_keys=True))
