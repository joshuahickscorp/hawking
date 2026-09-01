"""Emit the scoped architectural-effects ledger for Hawking Accelerator.

This ledger is deliberately narrower than a vendor survey.  It records what
was imported, what physical invariant Hawking extracted, where it is
implemented, and what would falsify the transfer.  An atlas entry is never a
physical law merely because it names more than one model or backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hcli.persist import atomic_write_json
from tools.accelerator.architecture_atlas import (
    EVIDENCE_CLASSES,
    PRIMITIVES,
    SCHEMA as ATLAS_SCHEMA,
    STATUSES,
    build_atlas,
    validate_atlas,
)
from tools.accelerator.physical_qualification import (
    CANDIDATE_SCOPE_TAGS,
    validate_queue,
)


SCHEMA = "hawking.accelerator.repatriation_effects.v1"
DEFAULT_OUT = Path("receipts/headless/ACCELERATOR_REPATRIATION_EFFECTS.json")
LEVELS = ("SPECIMEN_IMPLEMENTATION", "ACCELERATOR_PRIMITIVE", "PHYSICAL_LAW")
GENERICITY = ("NOT_APPLICABLE", "CANDIDATE_UNVERIFIED", "VERIFIED")
CANDIDATE_OUTCOMES = (
    "IMPLEMENTED_UNMEASURED",
    "REJECTED_PARITY",
    "REJECTED_PHYSICAL",
    "PHYSICAL_WIN_MODEL_LOCAL",
    "PHYSICAL_WIN_FAMILY",
    "GENERIC_CANDIDATE",
    "GENERIC_VERIFIED",
)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _reference_path(root: Path, raw: str) -> Path:
    reference = str(raw).split("::", 1)[0]
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def _implementation_paths(root: Path, evidence: Sequence[str]) -> list[str]:
    rows: list[str] = []
    for raw in evidence:
        path = _reference_path(root, raw)
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        if relative.startswith(("crates/", "hcli/", "tools/")):
            rows.append(str(raw))
    if rows:
        return list(dict.fromkeys(rows))
    return [str(raw) for raw in evidence[:1]]


def _effect_scope(entry: Mapping[str, Any]) -> dict[str, Any]:
    models = [str(value) for value in entry.get("applicable_models") or []]
    backends = [str(value) for value in entry.get("applicable_backends") or []]
    if len(models) > 1:
        scope_tags = ["ARCHITECTURE_FAMILY"]
    else:
        scope_tags = ["MODEL_LOCAL"]
    if len(backends) > 1:
        scope_tags.append("BACKEND_FAMILY")
    return {
        "level": "ACCELERATOR_PRIMITIVE",
        "model_scope": models,
        "backend_scope": backends,
        "organ_scope": [str(value) for value in entry.get("applicable_organs") or []],
        "candidate_scope_tags": scope_tags,
        "genericity": "CANDIDATE_UNVERIFIED",
        "transfer_evidence": [],
        "transfer_evidence_status": "PENDING_PROTECTED_CROSS_SCOPE_AB",
    }


def _measured_result(entry: Mapping[str, Any]) -> dict[str, Any]:
    evidence_class = str(entry.get("evidence_class") or "")
    receipt_paths = [
        str(raw)
        for raw in entry.get("source_evidence") or []
        if str(raw).startswith("receipts/")
    ]
    if evidence_class in {"HAWKING_MEASURED", "HAWKING_PROTECTED_VERIFIED"}:
        status = "SCOPED_RECEIPT_PRESENT"
    else:
        status = "NOT_MEASURED"
    return {
        "status": status,
        "evidence_class": evidence_class,
        "receipt_paths": receipt_paths,
        "metrics": {},
        "promotion": False,
        "claim_boundary": "any source or specimen measurement remains scoped; no cross-model or cross-backend law is promoted",
    }


def _candidate_outcome(row: Mapping[str, Any]) -> tuple[str, str]:
    """Project queue evidence into the repatriation outcome vocabulary.

    The queue remains the evidence authority.  Until a protected result is
    recorded, a source-backed candidate is explicitly unmeasured; no speed or
    transfer outcome is inferred from its mutation text.
    """
    tags = {str(value) for value in row.get("scope_tags") or []}
    explicit = str(row.get("repatriation_outcome") or row.get("outcome") or "")
    if explicit in CANDIDATE_OUTCOMES:
        return explicit, "explicit queue outcome supplied by the evidence authority"
    status = str(row.get("status") or "")
    if status in {"PROTECTED_PASS", "INTEGRATED"}:
        if "GENERIC_VERIFIED" in tags:
            return "GENERIC_VERIFIED", "protected result is already scoped as generic verified"
        if "GENERIC_CANDIDATE" in tags:
            return "PHYSICAL_WIN_FAMILY", "protected result establishes a cross-scope candidate win"
        if "ARCHITECTURE_FAMILY" in tags or "BACKEND_FAMILY" in tags:
            return "PHYSICAL_WIN_FAMILY", "protected result is family-scoped by the queue tags"
        return "PHYSICAL_WIN_MODEL_LOCAL", "protected result is model-local by the queue tags"
    if status == "PROTECTED_REJECT":
        evidence = " ".join(str(value) for value in row.get("evidence") or []).lower()
        if "parity" in evidence:
            return "REJECTED_PARITY", "protected rejection evidence names parity"
        return "REJECTED_PHYSICAL", "protected rejection has no parity marker; physical gate is the default"
    if "GENERIC_VERIFIED" in tags:
        return "GENERIC_VERIFIED", "queue scope tag is generic verified"
    if "GENERIC_CANDIDATE" in tags:
        return "GENERIC_CANDIDATE", "generic transfer remains unverified"
    return "IMPLEMENTED_UNMEASURED", "no protected result is present; implementation remains unmeasured"


def _effect_entry(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    evidence = [str(raw) for raw in entry.get("source_evidence") or []]
    return {
        "behavior_id": str(entry.get("behavior_id") or ""),
        "source_school": [str(raw) for raw in entry.get("source_architecture_ecosystem") or []],
        "source_behavior": str(entry.get("source_behavior") or ""),
        "physical_invariant": str(entry.get("architecture_independent_invariant") or ""),
        "hawking_primitive": str(entry.get("hawking_primitive") or ""),
        "implementations": _implementation_paths(root, evidence),
        "target_model": [str(raw) for raw in entry.get("applicable_models") or []],
        "target_backend": [str(raw) for raw in entry.get("applicable_backends") or []],
        "evidence": evidence,
        "measured_result": _measured_result(entry),
        "scope": _effect_scope(entry),
        "falsifier": str(entry.get("cheapest_falsifier") or ""),
        "atlas_status": str(entry.get("status") or ""),
        "atlas_evidence_class": str(entry.get("evidence_class") or ""),
    }


def build_effects(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve() if repo_root else REPO
    atlas_path = root / "receipts" / "headless" / "ACCELERATOR_ARCHITECTURE_ATLAS.json"
    queue_path = root / "receipts" / "headless" / "ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
    atlas = _load_json(atlas_path)
    queue = _load_json(queue_path)
    validate_atlas(atlas)
    validate_queue(queue)
    entries = atlas.get("entries")
    if not isinstance(entries, list):
        raise ValueError("architecture atlas entries are missing")
    candidates = queue.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("physical qualification candidates are missing")
    candidate_scope_bindings = [
        {
            "candidate_id": str(row.get("candidate_id") or ""),
            "model": str(row.get("model") or ""),
            "scope_tags": list(row.get("scope_tags") or []),
            "transfer_evidence": list(row.get("transfer_evidence") or []),
            "status": str(row.get("status") or ""),
            "outcome": _candidate_outcome(row)[0],
            "outcome_reason": _candidate_outcome(row)[1],
        }
        for row in candidates
        if isinstance(row, Mapping)
    ]
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "source_atlas": {
            "path": str(atlas_path),
            "schema": atlas.get("schema"),
            "fingerprint": atlas.get("fingerprint"),
        },
        "physical_queue": {
            "path": str(queue_path),
            "schema": queue.get("schema"),
            "fingerprint": queue.get("fingerprint"),
        },
        "levels": list(LEVELS),
        "candidate_scope_tags": sorted(CANDIDATE_SCOPE_TAGS),
        "transfer_policy": {
            "generic_candidate_requires": "explicit transfer evidence and a matched protected cross-scope A/B",
            "generic_verified_requires": "integrated evidence across the declared model/backend scope",
            "physical_law_requires": "repeated protected survival plus a named falsifier that has not fired",
            "current_physical_law_count": 0,
        },
        "entries": [_effect_entry(root, entry) for entry in entries if isinstance(entry, Mapping)],
        "candidate_scope_bindings": candidate_scope_bindings,
        "claim_boundary": "This ledger records scoped architectural effects and falsifiers. It does not turn source behavior, static accounting, or a specimen result into a generic physical law.",
    }
    body["fingerprint"] = _hash(body)
    return body


def validate_effects(document: Mapping[str, Any]) -> dict[str, Any]:
    if document.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")
    if document.get("levels") != list(LEVELS):
        raise ValueError("levels are stale")
    if document.get("candidate_scope_tags") != sorted(CANDIDATE_SCOPE_TAGS):
        raise ValueError("candidate scope taxonomy is stale")
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("effects entries are missing")
    ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("effects entries must be objects")
        required = (
            "behavior_id",
            "source_school",
            "source_behavior",
            "physical_invariant",
            "hawking_primitive",
            "implementations",
            "target_model",
            "target_backend",
            "evidence",
            "measured_result",
            "scope",
            "falsifier",
        )
        missing = [key for key in required if not entry.get(key)]
        if missing:
            raise ValueError(f"{entry.get('behavior_id') or '<entry>'} missing {missing}")
        behavior_id = str(entry["behavior_id"])
        if behavior_id in ids:
            raise ValueError(f"duplicate behavior_id {behavior_id!r}")
        ids.add(behavior_id)
        if str(entry.get("hawking_primitive")) not in PRIMITIVES:
            raise ValueError(f"{behavior_id}: unknown Hawking primitive")
        scope = entry["scope"]
        if not isinstance(scope, Mapping) or scope.get("level") not in LEVELS:
            raise ValueError(f"{behavior_id}: invalid scope level")
        tags = scope.get("candidate_scope_tags")
        if not isinstance(tags, list) or not tags or not set(tags).issubset(CANDIDATE_SCOPE_TAGS):
            raise ValueError(f"{behavior_id}: invalid candidate scope tags")
        if scope.get("genericity") not in GENERICITY:
            raise ValueError(f"{behavior_id}: invalid genericity")
        if scope.get("genericity") == "VERIFIED":
            raise ValueError(f"{behavior_id}: no generic law may be emitted by the planning ledger")
        measured = entry["measured_result"]
        if not isinstance(measured, Mapping) or measured.get("promotion") is not False:
            raise ValueError(f"{behavior_id}: measured result may not promote an effect")
    bindings = document.get("candidate_scope_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("candidate scope bindings are missing")
    candidate_ids: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise ValueError("candidate scope bindings must be objects")
        candidate_id = str(binding.get("candidate_id") or "")
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError(f"duplicate or empty candidate scope binding {candidate_id!r}")
        candidate_ids.add(candidate_id)
        if binding.get("outcome") not in CANDIDATE_OUTCOMES:
            raise ValueError(f"{candidate_id}: invalid repatriation outcome")
        tags = binding.get("scope_tags")
        if not isinstance(tags, list) or not tags or not set(tags).issubset(CANDIDATE_SCOPE_TAGS):
            raise ValueError(f"{candidate_id}: invalid candidate scope tags")
        if "GENERIC_CANDIDATE" in tags and not binding.get("transfer_evidence"):
            raise ValueError(f"{candidate_id}: generic candidate lacks transfer evidence")
        if "GENERIC_VERIFIED" in tags:
            raise ValueError(f"{candidate_id}: generic verification is not established")
    if document.get("transfer_policy", {}).get("current_physical_law_count") != 0:
        raise ValueError("planning ledger cannot claim a physical law")
    expected = _hash({key: value for key, value in document.items() if key != "fingerprint"})
    if document.get("fingerprint") != expected:
        raise ValueError("effects fingerprint does not match canonical body")
    return {
        "schema": "hawking.accelerator.repatriation_effects_validation.v1",
        "passed": True,
        "entries": len(entries),
        "candidate_scope_bindings": len(bindings),
        "physical_laws": 0,
        "claim_boundary": "effects validation is scoped provenance, not physical performance",
    }


def emit_effects(*, repo_root: str | Path | None = None, output: str | Path | None = None) -> Path:
    root = Path(repo_root).expanduser().resolve() if repo_root else REPO
    destination = Path(output).expanduser() if output else root / DEFAULT_OUT
    if not destination.is_absolute():
        destination = root / destination
    body = build_effects(repo_root=root)
    validate_effects(body)
    atomic_write_json(destination, body)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--emit", default=None)
    parser.add_argument("--validate", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.validate:
        result = validate_effects(_load_json(Path(args.validate).expanduser()))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    destination = emit_effects(repo_root=args.repo_root, output=args.emit)
    body = _load_json(destination)
    print(json.dumps({"status": "PASSED", "path": str(destination), "fingerprint": body["fingerprint"], "entries": len(body["entries"])}, sort_keys=True))
    return 0


__all__ = [
    "CANDIDATE_OUTCOMES",
    "DEFAULT_OUT",
    "SCHEMA",
    "build_effects",
    "emit_effects",
    "main",
    "validate_effects",
]


if __name__ == "__main__":
    raise SystemExit(main())
