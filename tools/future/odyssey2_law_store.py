"""ODYSSEY2_LAW_STORE — every law carries its evidence and its SCOPE.

Odyssey II asks WHAT DID HAWKING ALREADY LEARN? Transfer receipts already
exist; nothing held a law together with a sequential scope lattice, and a
model-local observation could be quoted as if it were generic. This module
is that store. Promotion without evidence raises ScopeViolation. Transfers
already in tools/foundry/NEGATIVE_TRANSFER_ATLAS.json raise
NegativeTransferError. Flash and Qwen27 are the first transfer school.

    python3 tools/future/odyssey2_law_store.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

from tools.future._common import REPO, git, load_json, write_receipt
from tools.verify import status_causality as sc

RECEIPT = "ODYSSEY2_LAW_STORE.json"
SCHEMA = "hawking.future.odyssey2_law_store.v1"

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
    """Consumer-side emit. Sibling owns the routine; this checkout may predate it."""
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
    fn = getattr(sc, "records_five_fields", None)
    if callable(fn):
        return bool(fn(node))
    if not isinstance(node, dict):
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
    if not isinstance(conf, dict):
        return False
    if not {"would_raise", "would_lower", "level", "about"} <= set(conf):
        return False
    alts = node.get("alternatives")
    return isinstance(alts, list) and bool(alts)


def record_law_store_causality(
    result: dict[str, Any],
    *,
    probe_performed: str = "",
    direct_observation: Any = "",
    interpretation: str | None = None,
    probe_kind: str = "",
    claim_kind: str | None = None,
) -> dict[str, Any]:
    """Stamp the five causality fields. Does not change schools.Flash.physical_status.

    An unsupplied observation is UNTESTED, never a restatement of the field.
    """
    schools = result.get("schools") if isinstance(result.get("schools"), dict) else {}
    flash = schools.get("Flash") if isinstance(schools.get("Flash"), dict) else {}
    field_before = flash.get("physical_status")
    status = str(field_before or "metadata_only_weights_not_present")
    unsupplied = direct_observation in (None, "", [], {})
    rec = sc.emit(
        status,
        probe_performed=str(probe_performed or ""),
        direct_observation="" if unsupplied else direct_observation,
        interpretation=interpretation if interpretation is not None else status,
        probe_kind="" if unsupplied else probe_kind,
        claim_kind=None if unsupplied else claim_kind,
        source="tools/future/odyssey2_law_store.py::build",
    )
    for key in FIVE_RECORDED_FIELDS:
        result[key] = rec[key]
    result["causality_verdict"] = rec["verdict"]
    result["falsifier"] = rec.get("falsifier")
    if rec.get("probe_kind"):
        result["probe_kind"] = rec["probe_kind"]
    if rec.get("claim_kind") is not None:
        result["claim_kind"] = rec["claim_kind"]
    schools_after = result.get("schools") if isinstance(result.get("schools"), dict) else {}
    flash_after = schools_after.get("Flash") if isinstance(schools_after.get("Flash"), dict) else {}
    if flash_after.get("physical_status") != field_before:
        raise RuntimeError("status_causality.emit mutated schools.Flash.physical_status")
    return rec

LAW_FIELDS = (
    "law_id",
    "statement",
    "source_model",
    "source_device",
    "architecture_family",
    "organ_class",
    "backend",
    "evidence_strength",
    "evidence_refs",
    "scope",
    "transfer_candidates",
    "transfer_confidence",
    "counterexample_requirement",
    "expected_saved_experiments",
    "actual_saved_experiments",
    "time_to_first_useful_executable_ns",
)

# Sequential lattice. Skipping a level is refused outright.
SCOPES = (
    "MODEL_LOCAL",
    "ARCHITECTURE_FAMILY",
    "BACKEND_FAMILY",
    "MACHINE_LOCAL",
    "GENERIC_CANDIDATE",
    "GENERIC_VERIFIED",
)

EVIDENCE_STRENGTHS = (
    "ANECDOTE",
    "STATIC",
    "DIAGNOSTIC_RELATIVE",
    "PROTECTED_ABSOLUTE",
    "REPRODUCED",
)

# Vocabulary reused from tools/accelerator/akb.py, not a parallel scheme.
# AKB: Simulated/Derived/Measured/Reproduced/ProtectedVerified.
# Odyssey II evidence_strength is the campaign's contamination/authority axis.
AKB_TO_STRENGTH = {
    "Simulated": "ANECDOTE",
    "Derived": "STATIC",
    "Measured": "DIAGNOSTIC_RELATIVE",
    "Reproduced": "REPRODUCED",
    "ProtectedVerified": "PROTECTED_ABSOLUTE",
}

# tools/headless/cross_model_laws.py used a Qwen-centric 4-level ladder.
# Map it onto this lattice without silently promoting.
LEGACY_LEVEL_TO_SCOPE = {
    "QWEN_SPECIFIC": "MODEL_LOCAL",
    "FAMILY_TRANSFERRED": "ARCHITECTURE_FAMILY",
    "ARCHITECTURE_GENERAL": "GENERIC_CANDIDATE",
    "MACHINE_GENERAL": "MACHINE_LOCAL",
}

SEED_RECEIPTS = (
    "receipts/headless/ODYSSEY_TRANSFER_PROVEN.json",
    "receipts/headless/ACCELERATOR_TRANSFER_VERIFIED.json",
    "receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json",
    "receipts/headless/QWEN_TRANSFER_REHEARSAL.json",
    "receipts/headless/DOCTOR_TRANSFER.json",
    "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json",
    "receipts/headless/CROSS_MODEL_LAWS.json",
    "receipts/headless/QWEN_TRANSFER_REPORT.json",
    "receipts/headless/ACCELERATOR_LAW_BASE.json",
    "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
)

# First real transfer school. Identities from
# receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json, not invented.
SCHOOLS = {
    "Qwen27": {
        "school": "Qwen27",
        "source_model": "Qwen3.8-27B",
        "architecture_family": "dense_hybrid_transformer",
        "aliases": (
            "qwen3.8-27b-abliterated",
            "Qwen3.8-27B",
            "Qwen3.8-27B sealed resident / NOETIC_PARENT_A",
            "NOETIC_PARENT_A",
        ),
        "role_in_map": "model_a",
        "map_label": "Qwen3.8-27B sealed resident / NOETIC_PARENT_A",
    },
    "Flash": {
        "school": "Flash",
        "source_model": "Qwen/Qwen3.8-Flash-Next",
        "architecture_family": "qwen4_exp",
        "aliases": (
            "Flash",
            "Qwen/Qwen3.8-Flash-Next",
            "Qwen3.8-Flash-Next",
            "qwen4_exp",
        ),
        "role_in_map": "model_b",
        "map_label": "Qwen/Qwen3.8-Flash-Next pinned ModelLake source",
        "physical_status": "metadata_only_weights_not_present",
    },
}

TRANSFER_LABEL_CONFIDENCE = {
    "DIRECT_TRANSFER": (0.70, "DIRECT_TRANSFER in QWEN38_ACCELERATOR_TRANSFER_MAP; claim_boundary requires re-earn of parity"),
    "TEST_ON_FLASH": (0.40, "TEST_ON_FLASH hypothesis in QWEN38_ACCELERATOR_TRANSFER_MAP"),
    "ARCHITECTURE_SPECIFIC": (0.20, "ARCHITECTURE_SPECIFIC in QWEN38_ACCELERATOR_TRANSFER_MAP; not a free transfer"),
}

# Map architecture-class strings used in the seed receipts.
MODEL_FAMILY = {
    "qwen3.8-27b-abliterated": "dense_hybrid_transformer",
    "Qwen3.8-27B": "dense_hybrid_transformer",
    "Qwen3.8-27B sealed resident / NOETIC_PARENT_A": "dense_hybrid_transformer",
    "Qwen/Qwen3-30B-A3B": "qwen3_moe",
    "Qwen3-30B-A3B": "qwen3_moe",
    "Qwen3-VL-30B-A3B-Instruct": "qwen3_moe_vl",
    "Qwen3-VL-30B-A3B": "qwen3_moe_vl",
    "Kimi-VL-A3B-Instruct": "kimi_vl_moe",
    "Kimi-VL-A3B": "kimi_vl_moe",
    "tiiuae/Falcon-H1-7B-Instruct": "falcon_h1",
    "Falcon-H1-7B": "falcon_h1",
    "Qwen/Qwen3.8-Flash-Next": "qwen4_exp",
    "Flash": "qwen4_exp",
}


class LawStoreError(ValueError):
    """Base error for the scoped law store."""


class ScopeViolation(LawStoreError):
    """Raised when promote() is asked to widen a law its evidence does not support.

    Never silently clamped. The caller must see the exception.
    """

    def __init__(
        self,
        message: str,
        *,
        law_id: str | None = None,
        from_scope: str | None = None,
        to_scope: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.law_id = law_id
        self.from_scope = from_scope
        self.to_scope = to_scope
        self.reason = reason or message


class NegativeTransferError(LawStoreError):
    """Raised when transfer_candidates() would propose a transfer the atlas already killed."""

    def __init__(
        self,
        message: str,
        *,
        law_id: str | None = None,
        atlas_key: str | None = None,
        target: str | None = None,
    ) -> None:
        super().__init__(message)
        self.law_id = law_id
        self.atlas_key = atlas_key
        self.target = target


@dataclass(frozen=True)
class Law:
    law_id: str
    statement: str
    source_model: str
    source_device: str
    architecture_family: str
    organ_class: str
    backend: str
    evidence_strength: str
    evidence_refs: tuple[str, ...]
    scope: str
    transfer_candidates: tuple[dict[str, Any], ...]
    transfer_confidence: dict[str, Any]
    counterexample_requirement: str
    expected_saved_experiments: int | None
    actual_saved_experiments: int | None
    time_to_first_useful_executable_ns: None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence_refs"] = list(self.evidence_refs)
        d["transfer_candidates"] = [dict(sorted(p.items())) for p in self.transfer_candidates]
        d["transfer_confidence"] = dict(sorted(self.transfer_confidence.items()))
        d["time_to_first_useful_executable_ns"] = None
        return {k: d[k] for k in LAW_FIELDS}


def validate_law(law: Law) -> Law:
    missing = [f for f in LAW_FIELDS if not hasattr(law, f)]
    if missing:
        raise LawStoreError(f"{law.law_id}: missing fields {missing}")
    if law.scope not in SCOPES:
        raise ScopeViolation(
            f"{law.law_id}: scope {law.scope!r} is not on the lattice",
            law_id=law.law_id,
            from_scope=law.scope,
            reason="unknown scope",
        )
    if law.evidence_strength not in EVIDENCE_STRENGTHS:
        raise LawStoreError(
            f"{law.law_id}: evidence_strength {law.evidence_strength!r} is not a known class"
        )
    value = law.transfer_confidence.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LawStoreError(f"{law.law_id}: transfer_confidence.value must be a number")
    if not 0.0 <= float(value) <= 1.0:
        raise LawStoreError(f"{law.law_id}: transfer_confidence {value} not in [0, 1]")
    if "basis" not in law.transfer_confidence:
        raise LawStoreError(f"{law.law_id}: transfer_confidence must record its basis")
    if law.time_to_first_useful_executable_ns is not None:
        raise LawStoreError(
            f"{law.law_id}: time_to_first_useful_executable_ns is null until a "
            f"protected measurement exists; sidecar must not invent one"
        )
    return law


# ---------------------------------------------------------------------------
# Repo JSON: sparse checkout means most receipts are in git but not on disk.
# ---------------------------------------------------------------------------

def load_repo_json(rel: str) -> dict[str, Any]:
    """Load JSON from the working tree if present, else `git show HEAD:rel`."""
    path = REPO / rel
    if path.is_file():
        return load_json(path)
    blob = git("show", f"HEAD:{rel}")
    if not blob:
        raise FileNotFoundError(rel)
    return json.loads(blob)


def try_load(rel: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return load_repo_json(rel), None
    except FileNotFoundError:
        return None, f"not in working tree or git HEAD: {rel}"
    except json.JSONDecodeError as e:
        return None, f"unreadable JSON at {rel}: {e}"


def _receipt_path(citation: str) -> str:
    return citation.split("#", 1)[0]


def _uniq(items: list[str] | tuple[str, ...] | None) -> list[str]:
    out: list[str] = []
    for x in items or ():
        if x is None:
            continue
        s = str(x)
        if s not in out:
            out.append(s)
    return out


def _norm(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")


def school_of_model(name: str) -> str | None:
    n = (name or "").lower()
    if not n:
        return None
    if "flash" in n or "qwen4_exp" in n:
        return "Flash"
    if "qwen3.8-27" in n or "qwen3.8-27b" in n or "noetic_parent_a" in n:
        return "Qwen27"
    if "abliterated" in n and "qwen3.8" in n:
        return "Qwen27"
    for school, meta in SCHOOLS.items():
        for alias in meta["aliases"]:
            if n == alias.lower() or alias.lower() in n:
                return school
    return None


def architecture_family_of(model: str) -> str:
    if model in MODEL_FAMILY:
        return MODEL_FAMILY[model]
    school = school_of_model(model)
    if school:
        return str(SCHOOLS[school]["architecture_family"])
    return "UNKNOWN"


def _confidence(
    strength: str,
    scope: str,
    *,
    n_models: int = 1,
    n_families: int = 1,
    map_label: str | None = None,
) -> dict[str, Any]:
    if map_label in TRANSFER_LABEL_CONFIDENCE:
        value, basis = TRANSFER_LABEL_CONFIDENCE[map_label]
        return {"value": value, "basis": basis}
    base = {
        "ANECDOTE": 0.10,
        "STATIC": 0.25,
        "DIAGNOSTIC_RELATIVE": 0.45,
        "PROTECTED_ABSOLUTE": 0.80,
        "REPRODUCED": 0.85,
    }[strength]
    cap = {
        "MODEL_LOCAL": 0.50,
        "ARCHITECTURE_FAMILY": 0.65,
        "BACKEND_FAMILY": 0.70,
        "MACHINE_LOCAL": 0.60,
        "GENERIC_CANDIDATE": 0.75,
        "GENERIC_VERIFIED": 0.95,
    }[scope]
    value = min(base, cap)
    if n_models < 2 and scope != "MACHINE_LOCAL":
        value = min(value, 0.50)
    if n_families < 2 and scope in {"GENERIC_CANDIDATE", "GENERIC_VERIFIED"}:
        value = min(value, 0.55)
    return {
        "value": round(float(value), 2),
        "basis": (
            f"evidence_strength={strength} scope={scope} "
            f"n_models={n_models} n_architecture_families={n_families}"
        ),
    }


# ---------------------------------------------------------------------------
# Scope lattice
# ---------------------------------------------------------------------------

def _scope_index(scope: str) -> int:
    if scope not in SCOPES:
        raise ScopeViolation(
            f"scope {scope!r} is not on the lattice {SCOPES}",
            from_scope=scope,
            reason="unknown scope",
        )
    return SCOPES.index(scope)


def _named(values: list[str]) -> list[str]:
    return [v for v in values if v and v not in {"UNKNOWN", "NONE", "null"}]


def _check_architecture_family(law: Law, evidence: dict[str, Any]) -> None:
    models = _uniq(evidence.get("models"))
    if len(models) < 2:
        raise ScopeViolation(
            f"{law.law_id}: ARCHITECTURE_FAMILY needs >=2 distinct models in that family; "
            f"got {models}",
            law_id=law.law_id,
            from_scope=law.scope,
            to_scope="ARCHITECTURE_FAMILY",
            reason="need_two_models_in_family",
        )
    families = _uniq(evidence.get("architecture_families"))
    if (
        families
        and law.architecture_family not in families
        and law.architecture_family not in {"UNKNOWN", ""}
    ):
        raise ScopeViolation(
            f"{law.law_id}: evidence families {families} do not include "
            f"law architecture_family {law.architecture_family!r}",
            law_id=law.law_id,
            from_scope=law.scope,
            to_scope="ARCHITECTURE_FAMILY",
            reason="family_mismatch",
        )


def _check_backend_family(law: Law, evidence: dict[str, Any]) -> None:
    backends = _named(_uniq(evidence.get("backends")))
    if len(backends) < 2:
        raise ScopeViolation(
            f"{law.law_id}: BACKEND_FAMILY needs >=2 backends; got {backends}",
            law_id=law.law_id,
            from_scope=law.scope,
            to_scope="BACKEND_FAMILY",
            reason="need_two_backends",
        )


def _check_machine_local(law: Law, evidence: dict[str, Any]) -> None:
    machines = _named(_uniq(evidence.get("machines") or evidence.get("devices")))
    if not machines:
        raise ScopeViolation(
            f"{law.law_id}: MACHINE_LOCAL needs a named machine/device; "
            f"UNKNOWN is not a machine identity",
            law_id=law.law_id,
            from_scope=law.scope,
            to_scope="MACHINE_LOCAL",
            reason="need_named_machine",
        )


def _check_generic_candidate(law: Law, evidence: dict[str, Any]) -> None:
    families = _named(_uniq(evidence.get("architecture_families")))
    if len(families) < 2:
        raise ScopeViolation(
            f"{law.law_id}: GENERIC_CANDIDATE needs >=2 architecture families; got {families}",
            law_id=law.law_id,
            from_scope=law.scope,
            to_scope="GENERIC_CANDIDATE",
            reason="need_two_architecture_families",
        )


def _check_generic_verified(law: Law, evidence: dict[str, Any]) -> None:
    strength = evidence.get("evidence_strength")
    if strength not in {"PROTECTED_ABSOLUTE", "REPRODUCED"}:
        raise ScopeViolation(
            f"{law.law_id}: GENERIC_VERIFIED needs PROTECTED_ABSOLUTE or REPRODUCED evidence; "
            f"got {strength!r}",
            law_id=law.law_id,
            from_scope=law.scope,
            to_scope="GENERIC_VERIFIED",
            reason="need_protected_or_reproduced",
        )
    families = _named(_uniq(evidence.get("architecture_families")))
    if len(families) < 2:
        raise ScopeViolation(
            f"{law.law_id}: GENERIC_VERIFIED needs PROTECTED_ABSOLUTE or REPRODUCED "
            f"evidence on >=2 architecture families; got {families}",
            law_id=law.law_id,
            from_scope=law.scope,
            to_scope="GENERIC_VERIFIED",
            reason="need_two_architecture_families",
        )
    if not evidence.get("counterexample_discharged"):
        raise ScopeViolation(
            f"{law.law_id}: GENERIC_VERIFIED needs the counterexample requirement discharged; "
            f"requirement={law.counterexample_requirement!r}",
            law_id=law.law_id,
            from_scope=law.scope,
            to_scope="GENERIC_VERIFIED",
            reason="counterexample_not_discharged",
        )


_PROMOTION_CHECKS = {
    "ARCHITECTURE_FAMILY": _check_architecture_family,
    "BACKEND_FAMILY": _check_backend_family,
    "MACHINE_LOCAL": _check_machine_local,
    "GENERIC_CANDIDATE": _check_generic_candidate,
    "GENERIC_VERIFIED": _check_generic_verified,
}


def promote(law: Law, target_scope: str, evidence: dict[str, Any]) -> Law:
    """Widen `law` by exactly one lattice step, or raise ScopeViolation.

    Evidence must actually support the jump. Skipping a level is refused.
    The original law is not mutated.
    """
    validate_law(law)
    if target_scope not in SCOPES:
        raise ScopeViolation(
            f"{law.law_id}: target scope {target_scope!r} is not on the lattice",
            law_id=law.law_id,
            from_scope=law.scope,
            to_scope=target_scope,
            reason="unknown scope",
        )
    current_i = _scope_index(law.scope)
    target_i = _scope_index(target_scope)
    if target_i <= current_i:
        raise ScopeViolation(
            f"{law.law_id}: {law.scope} -> {target_scope} is not a widening",
            law_id=law.law_id,
            from_scope=law.scope,
            to_scope=target_scope,
            reason="not_a_widening",
        )
    if target_i != current_i + 1:
        raise ScopeViolation(
            f"{law.law_id}: refusing level skip {law.scope} -> {target_scope}; "
            f"next legal step is {SCOPES[current_i + 1]}",
            law_id=law.law_id,
            from_scope=law.scope,
            to_scope=target_scope,
            reason="level_skip",
        )
    check = _PROMOTION_CHECKS[target_scope]
    check(law, evidence or {})
    refs = _uniq(list(law.evidence_refs) + list(evidence.get("evidence_refs") or []))
    strength = evidence.get("evidence_strength") or law.evidence_strength
    if strength not in EVIDENCE_STRENGTHS:
        strength = law.evidence_strength
    families = _uniq(evidence.get("architecture_families"))
    models = _uniq(evidence.get("models"))
    conf = _confidence(
        strength,
        target_scope,
        n_models=max(len(models), 1),
        n_families=max(len(families), 1),
    )
    return validate_law(
        replace(
            law,
            scope=target_scope,
            evidence_refs=tuple(refs),
            evidence_strength=strength,
            transfer_confidence=conf,
        )
    )


# ---------------------------------------------------------------------------
# Negative transfer atlas
# ---------------------------------------------------------------------------

_ATLAS_CACHE: dict[str, Any] | None = None
_ATLAS_ERROR: str | None = None


def negative_transfer_atlas() -> dict[str, Any]:
    global _ATLAS_CACHE, _ATLAS_ERROR
    if _ATLAS_CACHE is None and _ATLAS_ERROR is None:
        doc, err = try_load("tools/foundry/NEGATIVE_TRANSFER_ATLAS.json")
        if doc is None:
            _ATLAS_ERROR = err or "atlas missing"
            _ATLAS_CACHE = {"schema": None, "entries": {}}
        else:
            _ATLAS_CACHE = doc
            _ATLAS_ERROR = None
    return _ATLAS_CACHE or {"entries": {}}


def atlas_load_error() -> str | None:
    negative_transfer_atlas()
    return _ATLAS_ERROR


def is_failed_transfer_entry(entry: dict[str, Any]) -> bool:
    """True when the atlas records this lever as a failed transfer.

    LIVE / POSITIVE entries are not failures. Mixed entries (dead at depth,
    live on layer 0) count as failed for a GENERIC transfer of the lever.
    """
    verdict = str(entry.get("verdict") or "")
    v_up = verdict.upper()
    if "LIVE AND CONVERGENT" in v_up:
        return False
    if verdict.strip().upper().startswith("LIVE") and "DEAD" not in v_up:
        return False
    if "NEARLY EXHAUSTED" in v_up and "NOT CLOSED" in v_up and "DEAD" not in v_up:
        return False
    killed = str(entry.get("killed_by") or "")
    if killed.lower().startswith("nothing"):
        return False
    tokens = (
        "DEAD",
        "DOMINATED",
        "PINNED",
        "SUPERSEDED",
        "COLLAPSED",
        "NO-GO",
        "ABSENT",
        "INSIDE THE NOISE",
        "DEAD ON ARRIVAL",
    )
    if any(tok in v_up for tok in tokens):
        return True
    if "the 88-token calibration is dead" in verdict:
        return True
    if killed and not killed.lower().startswith("nothing"):
        return True
    return False


def failed_atlas_entries() -> dict[str, dict[str, Any]]:
    atlas = negative_transfer_atlas()
    entries = atlas.get("entries") or {}
    return {
        key: entries[key]
        for key in sorted(entries)
        if isinstance(entries[key], dict) and is_failed_transfer_entry(entries[key])
    }


def _law_blob(law: Law) -> str:
    parts = [
        law.law_id,
        law.statement,
        law.organ_class,
        " ".join(p.get("primitive") or "" for p in law.transfer_candidates),
    ]
    return _norm(" ".join(parts))


def match_failed_atlas(law: Law) -> tuple[str, dict[str, Any]] | None:
    blob = _law_blob(law)
    for key, entry in failed_atlas_entries().items():
        lever = str(entry.get("lever") or key)
        needles = [_norm(key), _norm(lever)]
        for n in needles:
            compact = n.replace("_", "")
            if len(compact) < 8:
                continue
            if n in blob or compact in blob.replace("_", ""):
                return key, entry
    return None


def _school_meta(target: str) -> dict[str, Any]:
    if target in SCHOOLS:
        return SCHOOLS[target]
    school = school_of_model(target)
    if school:
        return SCHOOLS[school]
    raise LawStoreError(f"unknown transfer target {target!r}; schools are {sorted(SCHOOLS)}")


def transfer_candidates(law: Law, target: str) -> list[dict[str, Any]]:
    """Propose where `law` might apply next.

    `target` is a school name (Flash, Qwen27) or a model alias.
    Raises NegativeTransferError if the atlas already recorded this transfer
    as failed. Raises, never returns a flagged proposal.
    """
    validate_law(law)
    if atlas_load_error():
        raise NegativeTransferError(
            f"negative transfer atlas unavailable ({atlas_load_error()}); refusing all transfers",
            law_id=law.law_id,
            target=target,
        )
    hit = match_failed_atlas(law)
    if hit is not None:
        key, entry = hit
        raise NegativeTransferError(
            f"{law.law_id}: refusing transfer of atlas-dead lever {key!r} "
            f"to {target}: {entry.get('verdict')}",
            law_id=law.law_id,
            atlas_key=key,
            target=target,
        )
    meta = _school_meta(target)
    target_school = meta["school"]
    src_school = school_of_model(law.source_model)
    if src_school == target_school:
        return []
    conf = law.transfer_confidence
    proposal = {
        "target_school": target_school,
        "target_model": meta["source_model"],
        "target_architecture_family": meta["architecture_family"],
        "confidence": conf["value"],
        "confidence_basis": conf["basis"],
        "counterexample_requirement": law.counterexample_requirement,
        "source_school": src_school,
        "source_model": law.source_model,
    }
    return [proposal]


def attach_school_candidates(law: Law) -> Law:
    """Fill transfer_candidates for the Flash <-> Qwen27 school.

    Atlas-dead laws keep an empty tuple; the refusal is the engine's job when
    transfer_candidates() is called, not a silent flag on the record.
    """
    src = school_of_model(law.source_model)
    targets: list[str] = []
    if src == "Qwen27":
        targets = ["Flash"]
    elif src == "Flash":
        targets = ["Qwen27"]
    else:
        targets = ["Flash", "Qwen27"]
    proposals: list[dict[str, Any]] = []
    for t in targets:
        try:
            proposals.extend(transfer_candidates(law, t))
        except NegativeTransferError:
            continue
    return replace(law, transfer_candidates=tuple(proposals))


# ---------------------------------------------------------------------------
# Seeding from real receipts. No invented laws.
# ---------------------------------------------------------------------------

def _law(
    *,
    law_id: str,
    statement: str,
    source_model: str,
    architecture_family: str,
    organ_class: str,
    evidence_strength: str,
    evidence_refs: list[str],
    scope: str,
    counterexample_requirement: str,
    source_device: str = "UNKNOWN",
    backend: str = "UNKNOWN",
    expected_saved_experiments: int | None = None,
    actual_saved_experiments: int | None = None,
    map_label: str | None = None,
    n_models: int = 1,
    n_families: int = 1,
) -> Law:
    conf = _confidence(
        evidence_strength,
        scope,
        n_models=n_models,
        n_families=n_families,
        map_label=map_label,
    )
    law = Law(
        law_id=law_id,
        statement=statement,
        source_model=source_model,
        source_device=source_device,
        architecture_family=architecture_family,
        organ_class=organ_class,
        backend=backend,
        evidence_strength=evidence_strength,
        evidence_refs=tuple(_uniq(evidence_refs)),
        scope=scope,
        transfer_candidates=(),
        transfer_confidence=conf,
        counterexample_requirement=counterexample_requirement,
        expected_saved_experiments=expected_saved_experiments,
        actual_saved_experiments=actual_saved_experiments,
        time_to_first_useful_executable_ns=None,
    )
    return validate_law(law)


def _seed_cross_model_laws(doc: dict[str, Any]) -> tuple[list[Law], list[str]]:
    notes: list[str] = []
    laws: list[Law] = []
    for raw in doc.get("laws") or []:
        lid = raw["id"]
        legacy = raw["level"]
        models = list(raw.get("measured_on_models") or [])
        families = _uniq(architecture_family_of(m) for m in models)
        refs = _uniq(_receipt_path(c) for c in (raw.get("evidence") or []))
        refs.append("receipts/headless/CROSS_MODEL_LAWS.json")
        # Honest placement on THIS lattice, not a copy of the Qwen-centric level.
        # Two architecture classes make a GENERIC_CANDIDATE, never GENERIC_VERIFIED:
        # none of these carry PROTECTED_ABSOLUTE evidence, and Falcon already
        # refused one promotion (LAW-HELDOUT-REAL-ACTIVATIONS).
        if lid == "LAW-FITTED-AFFINE-BEATS-RTN" and len(families) >= 2:
            scope = "GENERIC_CANDIDATE"
        elif lid == "LAW-PER-ORGAN-FLOOR" and len(families) >= 2:
            scope = "GENERIC_CANDIDATE"
        elif lid == "LAW-HELDOUT-REAL-ACTIVATIONS":
            # Falcon measurement refused the composition-ranking claim.
            # Two Qwen architecture classes observed the method; the strong
            # form is not generic. Stay below GENERIC_CANDIDATE.
            scope = "ARCHITECTURE_FAMILY"
            notes.append(
                "LAW-HELDOUT-REAL-ACTIVATIONS: Falcon-H1 refused the composition "
                "misranking; seeded at ARCHITECTURE_FAMILY, not GENERIC_CANDIDATE"
            )
        elif lid == "LAW-DEVICE-ROOFS":
            scope = "MACHINE_LOCAL"
        elif legacy == "QWEN_SPECIFIC" or len(models) < 2:
            scope = "MODEL_LOCAL"
        elif len(set(families)) >= 2:
            scope = "GENERIC_CANDIDATE"
        else:
            scope = "ARCHITECTURE_FAMILY"
        refuted = raw.get("refuted_on_models") or []
        counter = raw.get("why_not_higher") or "a new measurement that fails the statement"
        if refuted:
            counter = (
                f"already refuted on {refuted}; "
                f"any further transfer of the VALUE must re-measure. {counter}"
            )
        strength = "DIAGNOSTIC_RELATIVE"
        if lid == "LAW-DEVICE-ROOFS":
            # Machine property quoted from a measurement receipt; sidecar does
            # not re-measure. DIAGNOSTIC_RELATIVE, never PROTECTED_ABSOLUTE.
            strength = "DIAGNOSTIC_RELATIVE"
        laws.append(
            _law(
                law_id=lid,
                statement=raw["law"],
                source_model=models[0] if models else "UNKNOWN",
                architecture_family=architecture_family_of(models[0]) if models else "UNKNOWN",
                organ_class="cross_model",
                evidence_strength=strength,
                evidence_refs=refs,
                scope=scope,
                counterexample_requirement=counter,
                n_models=len(models),
                n_families=len(families),
            )
        )
        notes.append(
            f"{lid}: legacy_level={legacy} -> scope={scope} "
            f"models={models} architecture_families={families}"
        )
    return laws, notes


def _seed_odyssey_transfer_proven(doc: dict[str, Any]) -> tuple[list[Law], list[str]]:
    notes: list[str] = []
    delta = doc.get("delta") or {}
    matched = doc.get("matched_bits_comparison") or {}
    specimen = str(doc.get("specimen") or "")
    model = "Qwen/Qwen3-30B-A3B"
    refs = ["receipts/headless/ODYSSEY_TRANSFER_PROVEN.json"]
    n_tiers = matched.get("n_tiers_seeded_wins")
    notes.append(
        f"ODYSSEY_TRANSFER_PROVEN pass={doc.get('pass')} organ={doc.get('organ')} "
        f"layer={doc.get('layer')} n_tiers_seeded_wins={n_tiers} "
        f"evaluations_avoided={delta.get('evaluations_avoided')} specimen={specimen!r}"
    )
    # Accounting law: transfer actually paid negative evaluations under the
    # loose landing target. Numbers are evaluation counts, not hardware.
    actual = delta.get("evaluations_avoided")
    if not isinstance(actual, int):
        actual = None
        notes.append("ODYSSEY_TRANSFER_PROVEN delta.evaluations_avoided is not an int; actual=null")
    law = _law(
        law_id="LAW-COLD-CONTROL-BEAT-TRANSFER-SEED",
        statement=(
            "On Qwen3-30B-A3B moe_expert layer 2, under the pre-registered loose "
            "landing target (held-out rel_fro <= 1.15x of uniform q4 g64), the COLD "
            "arm landed in 2 evaluations and the ODYSSEY_TRANSFER seed landed in 10; "
            "evaluations_avoided=-8. The transfer signal is in the matched-bits "
            "comparison (seeded affine wins 3/3 tiers), not in search-length. Values "
            "did not save experiments; the method still discriminated at matched bits."
        ),
        source_model=model,
        architecture_family="qwen3_moe",
        organ_class="moe_expert",
        evidence_strength="DIAGNOSTIC_RELATIVE",
        evidence_refs=refs,
        scope="MODEL_LOCAL",
        counterexample_requirement=(
            "a pre-registered landing target on this organ where the transfer seed "
            "uses strictly fewer evaluations than cold AND lands at equal or better "
            "held-out rel_fro; until then the savings claim is refuted here"
        ),
        expected_saved_experiments=None,
        actual_saved_experiments=actual,
        n_models=1,
        n_families=1,
    )
    return [law], notes


def _seed_accelerator_transfer_verified(doc: dict[str, Any]) -> tuple[list[Law], list[str]]:
    notes: list[str] = []
    identities = doc.get("identities") or {}
    models = list((identities.get("model") or {}).get("specimens") or [])
    device = ((identities.get("device") or {}).get("name")) or "UNKNOWN"
    backend = "Metal" if (identities.get("device") or {}).get("api") == "Metal" else "UNKNOWN"
    families = _uniq(architecture_family_of(m) for m in models)
    refs = ["receipts/headless/ACCELERATOR_TRANSFER_VERIFIED.json"]
    result = doc.get("result") or {}
    layout = result.get("layout_survey") or {}
    notes.append(
        f"ACCELERATOR_TRANSFER_VERIFIED pass={doc.get('pass')} "
        f"knowledge_level={doc.get('knowledge_level')} specimens={models} "
        f"families={families} layout_keys={sorted(layout)}"
    )
    if not models:
        notes.append("ACCELERATOR_TRANSFER_VERIFIED carries no specimen list; no law seeded from identities")
        return [], notes
    # Kernel reuse is a claim about storage layout. Observed across four
    # specimens / three architecture classes. One backend (Metal). Not protected.
    kernel_law = _law(
        law_id="LAW-KERNEL-REUSE-FOLLOWS-STORAGE-LAYOUT",
        statement=(
            "Kernel reuse is a claim about storage layout, not only about GEMV shape "
            "or architecture-config similarity. Qwen3-30B-A3B and Qwen3-VL-30B-A3B "
            "share byte-identical MSL only AFTER a de-interleave and a transpose: "
            "the VL variant stores fused 3-D gate_up_proj, so the tensor as stored "
            "emits a different kernel. Kimi-VL uses the same per-expert 2-D convention "
            "but a different row size and shares no kernel. Falcon-H1 has no expert "
            "tensors at all."
        ),
        source_model=models[0],
        architecture_family=architecture_family_of(models[0]),
        organ_class="moe_expert_kernel",
        evidence_strength="DIAGNOSTIC_RELATIVE",
        evidence_refs=refs,
        scope="GENERIC_CANDIDATE" if len(families) >= 2 else "ARCHITECTURE_FAMILY",
        counterexample_requirement=(
            "two specimens with divergent on-disk expert layout whose stored "
            "tensors emit the same MSL without a pack-time transpose/de-interleave"
        ),
        source_device=device,
        backend=backend,
        n_models=len(models),
        n_families=len(families),
    )
    cosine_law = _law(
        law_id="LAW-REPRESENTATION-COSINE-ROBUST-TO-LAYOUT",
        statement=(
            "Representation fidelity under ws_rtn_q4_g64 transfers across the Qwen3 "
            "MoE / VL storage-layout split: weight cosine is 0.993952 on model #2 "
            "gate_proj and 0.993965 on the VL variant's canonical gate, and 0.993816 "
            "even in the on-disk orientation. Adequacy per tensor is robust to layout; "
            "only the kernel is layout-sensitive."
        ),
        source_model="Qwen3-30B-A3B",
        architecture_family="qwen3_moe",
        organ_class="moe_expert",
        evidence_strength="DIAGNOSTIC_RELATIVE",
        evidence_refs=refs,
        scope="ARCHITECTURE_FAMILY",
        counterexample_requirement=(
            "a layout transform of the same tensor under ws_rtn_q4_g64 whose weight "
            "cosine against the representation drops below the pair observed here"
        ),
        source_device=device,
        backend=backend,
        n_models=2,
        n_families=1,
    )
    falcon_law = _law(
        law_id="LAW-FALCON-H1-HAS-NO-EXPERT-TENSORS",
        statement=(
            "Falcon-H1-7B has no expert tensors (0 of 751). It is the zero that makes "
            "the other kernel-reuse numbers meaningful and is not a MoE transfer target."
        ),
        source_model="Falcon-H1-7B",
        architecture_family="falcon_h1",
        organ_class="moe_expert",
        evidence_strength="STATIC",
        evidence_refs=refs,
        scope="MODEL_LOCAL",
        counterexample_requirement="a Falcon-H1 checkpoint that stores expert tensors",
        source_device=device,
        backend=backend,
        n_models=1,
        n_families=1,
    )
    return [kernel_law, cosine_law, falcon_law], notes


def _seed_qwen38_transfer_map(doc: dict[str, Any]) -> tuple[list[Law], list[str]]:
    notes: list[str] = []
    matrix = doc.get("transfer_matrix") or []
    model_a = (doc.get("model_a") or {}).get("label") or SCHOOLS["Qwen27"]["map_label"]
    model_b = (doc.get("model_b") or {}).get("label") or SCHOOLS["Flash"]["map_label"]
    phys = (doc.get("model_b") or {}).get("physical_status")
    notes.append(
        f"QWEN38_ACCELERATOR_TRANSFER_MAP model_a={model_a!r} model_b={model_b!r} "
        f"physical_status={phys!r} n_primitives={len(matrix)} "
        f"claim_boundary={str(doc.get('claim_boundary') or '')[:160]}"
    )
    if phys == "metadata_only_weights_not_present":
        notes.append(
            "Flash weights are metadata_only_weights_not_present; Flash-sourced laws "
            "are STATIC hypotheses, not measurements"
        )
    laws: list[Law] = []
    for row in matrix:
        primitive = row["primitive"]
        slug = _norm(primitive).replace("__", "_").upper()
        reason = row.get("reason") or ""
        label_a = row.get("model_a")
        label_b = row.get("model_b")
        # Qwen27 as source, Flash as target.
        laws.append(
            _law(
                law_id=f"LAW-XFER-QWEN27-{slug}",
                statement=(
                    f"Primitive {primitive!r}: labelled {label_a} on Qwen27 "
                    f"({model_a}) and {label_b} on Flash ({model_b}). {reason} "
                    f"Transfer labels are hypotheses; every physical primitive must "
                    f"re-earn same-model parity, capability, and complete-token acceptance."
                ),
                source_model=SCHOOLS["Qwen27"]["source_model"],
                architecture_family=SCHOOLS["Qwen27"]["architecture_family"],
                organ_class=primitive,
                evidence_strength="STATIC",
                evidence_refs=["receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json"],
                scope="MODEL_LOCAL",
                counterexample_requirement=(
                    f"a same-source Flash run of {primitive!r} that fails parity, "
                    f"capability, or complete-token acceptance against the Qwen27 body"
                ),
                map_label=label_a if label_a in TRANSFER_LABEL_CONFIDENCE else None,
                n_models=1,
                n_families=1,
            )
        )
        # Flash as source only where the map itself labelled Flash DIRECT_TRANSFER.
        # TEST_ON_FLASH / ARCHITECTURE_SPECIFIC on model_b is not Flash-sourced evidence.
        if label_b == "DIRECT_TRANSFER":
            laws.append(
                _law(
                    law_id=f"LAW-XFER-FLASH-{slug}",
                    statement=(
                        f"Primitive {primitive!r} is labelled DIRECT_TRANSFER on Flash "
                        f"({model_b}) as well as on Qwen27. {reason} Flash weights were "
                        f"not present on disk (metadata_only); this is a sealed hypothesis, "
                        f"not a Flash measurement."
                    ),
                    source_model=SCHOOLS["Flash"]["source_model"],
                    architecture_family=SCHOOLS["Flash"]["architecture_family"],
                    organ_class=primitive,
                    evidence_strength="STATIC",
                    evidence_refs=["receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json"],
                    scope="MODEL_LOCAL",
                    counterexample_requirement=(
                        f"a Qwen27 execution of {primitive!r} that disagrees with the "
                        f"Flash-labelled DIRECT_TRANSFER once Flash weights exist"
                    ),
                    map_label=label_b,
                    n_models=1,
                    n_families=1,
                )
            )
        else:
            notes.append(
                f"primitive {primitive!r}: Flash label is {label_b}; not seeded as a "
                f"Flash-sourced law (no Flash measurement)"
            )
    return laws, notes


def _seed_rehearsal_and_report(
    rehearsal: dict[str, Any] | None,
    report: dict[str, Any] | None,
) -> tuple[list[Law], list[str]]:
    notes: list[str] = []
    laws: list[Law] = []
    if rehearsal is None:
        notes.append("QWEN_TRANSFER_REHEARSAL.json missing; no rehearsal laws")
    else:
        notes.append(
            f"QWEN_TRANSFER_REHEARSAL pass={rehearsal.get('pass')} "
            f"architecture_class={(rehearsal.get('plan') or {}).get('architecture_class')} "
            f"n_methods_inherited={(rehearsal.get('plan') or {}).get('n_methods_inherited')} "
            f"n_prior_failures={(rehearsal.get('plan') or {}).get('n_prior_failures_applied')} "
            f"input_audit.clean={(rehearsal.get('input_audit') or {}).get('clean')}"
        )
        laws.append(
            _law(
                law_id="LAW-REHEARSAL-NO-SMUGGLING",
                statement=str(rehearsal.get("law") or ""),
                source_model="Qwen/Qwen3-30B-A3B",
                architecture_family="qwen3_moe",
                organ_class="method",
                evidence_strength="STATIC",
                evidence_refs=["receipts/headless/QWEN_TRANSFER_REHEARSAL.json"],
                scope="MODEL_LOCAL",
                counterexample_requirement=(
                    "a rehearsal whose input_audit.clean is false or whose "
                    "n_forbidden_reads > 0; that rehearsal is not a transfer"
                ),
            )
        )
    if report is None:
        notes.append(
            "QWEN_TRANSFER_REPORT.json missing; meta-law 'values do not transfer, "
            "methods do' not seeded from the report itself"
        )
    else:
        notes.append(
            f"QWEN_TRANSFER_REPORT pass={report.get('pass')} n_entries={report.get('n_entries')} "
            f"n_methods={report.get('n_methods')} n_negatives={report.get('n_negatives')} "
            f"n_rejected={report.get('n_rejected')} parent={report.get('parent')}"
        )
        refs = ["receipts/headless/QWEN_TRANSFER_REPORT.json"]
        if rehearsal is not None:
            refs.append("receipts/headless/QWEN_TRANSFER_REHEARSAL.json")
        laws.append(
            _law(
                law_id="LAW-VALUES-DO-NOT-TRANSFER-METHODS-DO",
                statement=str(report.get("law") or ""),
                source_model=str(report.get("parent") or "qwen3.8-27b-abliterated"),
                architecture_family="dense_hybrid_transformer",
                organ_class="method",
                evidence_strength="STATIC",
                evidence_refs=refs,
                scope="ARCHITECTURE_FAMILY",
                counterexample_requirement=(
                    "a numeric floor, roof, or bpw copied onto a new model that then "
                    "beats a re-measurement of that value on the new model's own organ"
                ),
                n_models=2 if rehearsal is not None else 1,
                n_families=2 if rehearsal is not None else 1,
            )
        )
    return laws, notes


def _seed_doctor_transfer(doc: dict[str, Any]) -> tuple[list[Law], list[str]]:
    notes: list[str] = []
    pq = doc.get("prescription_quality") or {}
    n_organs = doc.get("n_organs")
    n_tech = doc.get("n_techniques_in_library")
    experiments = (pq.get("experiments_to_run") or {}).get("value")
    irrelevant = (pq.get("irrelevant_treatments_avoided") or {}).get("value")
    repeated = (pq.get("repeated_failures_avoided") or {}).get("value")
    notes.append(
        f"DOCTOR_TRANSFER pass={doc.get('pass')} n_techniques={n_tech} n_organs={n_organs} "
        f"experiments_to_run={experiments} irrelevant_avoided={irrelevant} "
        f"repeated_failures_avoided={repeated} all_KEEP={doc.get('all_techniques_still_KEEP')}"
    )
    expected = None
    actual = None
    if isinstance(n_tech, int) and isinstance(n_organs, int) and isinstance(experiments, int):
        expected = (n_tech * n_organs) - experiments
    if isinstance(irrelevant, int) and isinstance(repeated, int):
        actual = irrelevant + repeated
    specimen = (doc.get("specimen") or {}).get("repo") or "Qwen/Qwen3-30B-A3B"
    law = _law(
        law_id="LAW-QWEN-FAILURE-NEVER-PRUNES",
        statement=str(doc.get("pruning_law") or ""),
        source_model=specimen,
        architecture_family="qwen3_moe",
        organ_class="method",
        evidence_strength="STATIC",
        evidence_refs=["receipts/headless/DOCTOR_TRANSFER.json"],
        scope="MODEL_LOCAL",
        counterexample_requirement=(
            "a technique deleted from the library solely because it failed on a Qwen "
            "organ, with no reopening condition attached"
        ),
        expected_saved_experiments=expected,
        actual_saved_experiments=actual,
    )
    return [law], notes


def seed_store() -> tuple[list[Law], dict[str, Any]]:
    """Populate laws from the real receipts. Returns (laws, seed_report)."""
    found: dict[str, bool] = {}
    docs: dict[str, dict[str, Any] | None] = {}
    missing: list[str] = []
    notes: list[str] = []
    for rel in SEED_RECEIPTS:
        doc, err = try_load(rel)
        docs[rel] = doc
        found[rel] = doc is not None
        if err:
            missing.append(err)
            notes.append(err)

    laws: list[Law] = []

    def take(extra_laws: list[Law], extra_notes: list[str]) -> None:
        laws.extend(extra_laws)
        notes.extend(extra_notes)

    cml = docs.get("receipts/headless/CROSS_MODEL_LAWS.json")
    if cml:
        take(*_seed_cross_model_laws(cml))
    otp = docs.get("receipts/headless/ODYSSEY_TRANSFER_PROVEN.json")
    if otp:
        take(*_seed_odyssey_transfer_proven(otp))
    atv = docs.get("receipts/headless/ACCELERATOR_TRANSFER_VERIFIED.json")
    if atv:
        take(*_seed_accelerator_transfer_verified(atv))
    tmap = docs.get("receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json")
    if tmap:
        take(*_seed_qwen38_transfer_map(tmap))
    take(
        *_seed_rehearsal_and_report(
            docs.get("receipts/headless/QWEN_TRANSFER_REHEARSAL.json"),
            docs.get("receipts/headless/QWEN_TRANSFER_REPORT.json"),
        )
    )
    docx = docs.get("receipts/headless/DOCTOR_TRANSFER.json")
    if docx:
        take(*_seed_doctor_transfer(docx))

    atlas_rel = "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json"
    atlas = docs.get(atlas_rel)
    failed_keys: list[str] = []
    live_keys: list[str] = []
    if atlas:
        entries = atlas.get("entries") or {}
        for key in sorted(entries):
            if not isinstance(entries[key], dict):
                continue
            if is_failed_transfer_entry(entries[key]):
                failed_keys.append(key)
            else:
                live_keys.append(key)
        notes.append(
            f"NEGATIVE_TRANSFER_ATLAS {len(entries)} entries; "
            f"failed={failed_keys}; live_or_open={live_keys}"
        )
    else:
        notes.append("NEGATIVE_TRANSFER_ATLAS missing; transfer engine fail-closes")

    atlas_arch = docs.get("receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json")
    if atlas_arch is None:
        notes.append(
            "ACCELERATOR_ARCHITECTURE_ATLAS.json is not in git HEAD of this worktree; "
            "cannot seed entries[].transfer_confidence / evidence_class / "
            "architecture_independent_invariant from it. Vocabulary reused from "
            "ACCELERATOR_LAW_BASE.json (evidence_class) and "
            "QWEN38_ACCELERATOR_TRANSFER_MAP.json (transfer labels) instead."
        )
    else:
        notes.append(
            "ACCELERATOR_ARCHITECTURE_ATLAS.json WAS found; not expected on this HEAD. "
            "Inspected, not forked."
        )

    akb = docs.get("receipts/headless/ACCELERATOR_LAW_BASE.json")
    if akb:
        notes.append(
            f"ACCELERATOR_LAW_BASE.json present: n_entries={len(akb.get('entries') or [])} "
            f"evidence_classes={akb.get('evidence_classes')} — Codex-owned 11-axis store; "
            f"consumed for vocabulary, not ingested as Odyssey II laws"
        )

    # Attach Flash <-> Qwen27 proposals. Atlas-dead laws stay empty.
    attached: list[Law] = []
    refusals: list[str] = []
    for law in laws:
        if match_failed_atlas(law) is not None:
            key, entry = match_failed_atlas(law)  # type: ignore[misc]
            refusals.append(f"{law.law_id} matches atlas-dead {key}: {entry.get('verdict')}")
            attached.append(law)
            continue
        attached.append(attach_school_candidates(law))

    # Dedup by law_id, first writer wins, later evidence_refs merged.
    by_id: dict[str, Law] = {}
    for law in attached:
        if law.law_id in by_id:
            prev = by_id[law.law_id]
            by_id[law.law_id] = replace(
                prev,
                evidence_refs=tuple(_uniq(list(prev.evidence_refs) + list(law.evidence_refs))),
            )
            notes.append(f"merged duplicate law_id {law.law_id}")
        else:
            by_id[law.law_id] = law
    ordered = [by_id[k] for k in sorted(by_id)]

    sources = sorted({law.source_model for law in ordered})
    targets = sorted(
        {
            p.get("target_school")
            for law in ordered
            for p in law.transfer_candidates
            if p.get("target_school")
        }
    )
    report = {
        "seed_receipts_found": found,
        "seed_receipts_missing": missing,
        "notes": notes,
        "atlas_failed_keys": failed_keys,
        "atlas_live_or_open_keys": live_keys,
        "atlas_seed_refusals": refusals,
        "source_models": sources,
        "transfer_target_schools": targets,
        "n_laws": len(ordered),
    }
    return ordered, report


def accounting_summary(laws: list[Law]) -> dict[str, Any]:
    expected_vals = [n for n in (law.expected_saved_experiments for law in laws) if n is not None]
    actual_vals = [n for n in (law.actual_saved_experiments for law in laws) if n is not None]
    return {
        "n_laws": len(laws),
        "n_with_expected_saved_experiments": len(expected_vals),
        "n_with_actual_saved_experiments": len(actual_vals),
        "sum_expected_saved_experiments": sum(expected_vals) if expected_vals else None,
        "sum_actual_saved_experiments": sum(actual_vals) if actual_vals else None,
        "null_time_to_first_useful_executable_ns": all(
            law.time_to_first_useful_executable_ns is None for law in laws
        ),
        "note": (
            "Counts are experiment-evaluations from sealed receipts, not hardware. "
            "A negative actual means the transfer arm spent more evaluations than cold. "
            "Nulls stay null: a prescription is not an execution."
        ),
    }


def school_presence(laws: list[Law]) -> dict[str, Any]:
    as_source = {s: 0 for s in SCHOOLS}
    as_target = {s: 0 for s in SCHOOLS}
    for law in laws:
        src = school_of_model(law.source_model)
        if src in as_source:
            as_source[src] += 1
        for p in law.transfer_candidates:
            t = p.get("target_school")
            if t in as_target:
                as_target[t] += 1
    return {
        "as_source": as_source,
        "as_target": as_target,
        "both_schools_are_sources": all(as_source[s] > 0 for s in SCHOOLS),
        "both_schools_are_targets": all(as_target[s] > 0 for s in SCHOOLS),
    }


def recovered_implementation(seed_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "cross_model_laws": {
            "path": "tools/headless/cross_model_laws.py",
            "receipt": "receipts/headless/CROSS_MODEL_LAWS.json",
            "found": bool(seed_report["seed_receipts_found"].get("receipts/headless/CROSS_MODEL_LAWS.json")),
            "legacy_levels": [
                "QWEN_SPECIFIC",
                "FAMILY_TRANSFERRED",
                "ARCHITECTURE_GENERAL",
                "MACHINE_GENERAL",
            ],
            "legacy_mapping": LEGACY_LEVEL_TO_SCOPE,
            "gap": (
                "Qwen-centric 4-level ladder with validate-at-emit. No BACKEND_FAMILY, "
                "no sequential skip guard, no Flash/Qwen27 school, no expected/actual "
                "saved-experiment accounting. Consumed as seed, not forked."
            ),
        },
        "accelerator_knowledge_base": {
            "path": "tools/accelerator/akb.py",
            "receipt": "receipts/headless/ACCELERATOR_LAW_BASE.json",
            "found": bool(seed_report["seed_receipts_found"].get("receipts/headless/ACCELERATOR_LAW_BASE.json")),
            "akb_evidence_classes_to_odyssey_strength": AKB_TO_STRENGTH,
            "gap": (
                "11-axis applicability store, Codex-owned. evidence_class vocabulary "
                "reused. Entries were not ingested: they are kernel/dispatch laws, not "
                "Odyssey II transfer laws, and forking them would duplicate Codex."
            ),
        },
        "negative_transfer_atlas": {
            "path": "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
            "found": bool(seed_report["seed_receipts_found"].get("tools/foundry/NEGATIVE_TRANSFER_ATLAS.json")),
            "failed_keys": seed_report.get("atlas_failed_keys"),
            "live_or_open_keys": seed_report.get("atlas_live_or_open_keys"),
            "role": "mechanical refusal source for transfer_candidates()",
        },
        "architecture_atlas": {
            "path": "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json",
            "found": bool(
                seed_report["seed_receipts_found"].get(
                    "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json"
                )
            ),
            "gap": (
                "Named by F010/F012 as the shape to reuse "
                "(entries[].transfer_confidence, evidence_class, "
                "architecture_independent_invariant). Not present in git HEAD of this "
                "worktree. Cannot seed from it."
            ),
        },
        "transfer_receipts_consumed": {
            "ODYSSEY_TRANSFER_PROVEN": "receipts/headless/ODYSSEY_TRANSFER_PROVEN.json",
            "ACCELERATOR_TRANSFER_VERIFIED": "receipts/headless/ACCELERATOR_TRANSFER_VERIFIED.json",
            "QWEN38_ACCELERATOR_TRANSFER_MAP": "receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json",
            "QWEN_TRANSFER_REHEARSAL": "receipts/headless/QWEN_TRANSFER_REHEARSAL.json",
            "DOCTOR_TRANSFER": "receipts/headless/DOCTOR_TRANSFER.json",
            "QWEN_TRANSFER_REPORT": "receipts/headless/QWEN_TRANSFER_REPORT.json",
        },
        "related_not_ingested": [
            "tools/odyssey/admission_chain_scope.py — packer/admission scoping, not a law lattice",
            "tools/odyssey/known_failures.py — T0 claim-regression registry, Codex-owned",
            "receipts/headless/DENSE_SUBBIT_TRANSFER.json — GLM->dense Qwen3.8 NO-GO, not in the F010 seed list",
            "receipts/headless/ACCELERATOR_CLIFF_TRANSFER.json — occupancy-cliff INSTANCE, not a transfer school",
        ],
        "not_adequate_alone": (
            "cross_model_laws.py and akb.py already store laws. Neither holds the "
            "Odyssey II lattice, neither refuses MODEL_LOCAL -> GENERIC_VERIFIED as a "
            "level skip, neither seeds Flash <-> Qwen27. This module extends them by "
            "consuming their receipts; it does not replace them."
        ),
    }


def gaps_closed() -> list[str]:
    return [
        "Law record with the Odyssey II field set, including scope and evidence_strength.",
        "Sequential scope lattice MODEL_LOCAL < ARCHITECTURE_FAMILY < BACKEND_FAMILY < "
        "MACHINE_LOCAL < GENERIC_CANDIDATE < GENERIC_VERIFIED.",
        "promote() raises typed ScopeViolation on level skip, on single-model "
        "ARCHITECTURE_FAMILY, on single-backend BACKEND_FAMILY, and on GENERIC_VERIFIED "
        "without PROTECTED_ABSOLUTE/REPRODUCED plus a discharged counterexample.",
        "Seeding from the listed transfer receipts (via git show when sparse-checkout hides them).",
        "Flash <-> Qwen27 transfer school from QWEN38_ACCELERATOR_TRANSFER_MAP, both as source and as target.",
        "transfer_candidates() consults NEGATIVE_TRANSFER_ATLAS and raises NegativeTransferError.",
        "expected_saved_experiments vs actual_saved_experiments accounting (null until known; negative actual is legal).",
        "time_to_first_useful_executable_ns forced null: sidecar has no executable-timing authority.",
    ]


def build() -> Any:
    laws, seed_report = seed_store()
    presence = school_presence(laws)
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Odyssey II scoped law store: every law carries its evidence and its "
            "scope; promotion without evidence is refused; Flash <-> Qwen27 is the "
            "first transfer school."
        ),
        "odyssey": "II WHAT DID HAWKING ALREADY LEARN?",
        "lattice": list(SCOPES),
        "evidence_strengths": list(EVIDENCE_STRENGTHS),
        "promotion_rules": {
            "ARCHITECTURE_FAMILY": ">=2 distinct models in that architecture family",
            "BACKEND_FAMILY": ">=2 backends",
            "MACHINE_LOCAL": "named machine/device identity; UNKNOWN is not a machine",
            "GENERIC_CANDIDATE": ">=2 architecture families",
            "GENERIC_VERIFIED": (
                "PROTECTED_ABSOLUTE or REPRODUCED on >=2 architecture families AND "
                "a discharged counterexample requirement"
            ),
            "level_skip": "refused outright (ScopeViolation, never clamped)",
        },
        "schools": SCHOOLS,
        "school_presence": presence,
        "laws": [law.to_dict() for law in laws],
        "counts": {
            "n_laws": len(laws),
            "by_scope": {
                s: sum(1 for law in laws if law.scope == s) for s in SCOPES
            },
            "by_evidence_strength": {
                e: sum(1 for law in laws if law.evidence_strength == e)
                for e in EVIDENCE_STRENGTHS
            },
        },
        "accounting": accounting_summary(laws),
        "failed_transfers_indexed": seed_report.get("atlas_failed_keys") or [],
        "seed_report": seed_report,
        "recovered_implementation": recovered_implementation(seed_report),
        "gaps_closed": gaps_closed(),
        "negative_findings": _negative_findings(seed_report, presence),
    }
    flash_status = SCHOOLS["Flash"]["physical_status"]
    record_law_store_causality(
        doc,
        probe_performed=(
            "read SCHOOLS['Flash']['physical_status'] from the in-module catalog; "
            "no Path.exists, no hash of weight files"
        ),
        direct_observation=f"schools.Flash.physical_status={flash_status!r}",
        interpretation=(
            "the law-store catalog records this field; it is not a measurement "
            "of whether Flash weights exist on disk"
        ),
        probe_kind=sc.PROBE_METADATA,
        claim_kind=sc.CLAIM_FIELD_VALUE,
    )
    return write_receipt(RECEIPT, doc, "tools/future/odyssey2_law_store.py")


def selftest() -> Any:
    return build()


def _negative_findings(seed_report: dict[str, Any], presence: dict[str, Any]) -> list[str]:
    findings = list(seed_report.get("seed_receipts_missing") or [])
    if not presence.get("both_schools_are_sources"):
        findings.append(
            f"Flash/Qwen27 not both sources: {presence.get('as_source')}"
        )
    if not presence.get("both_schools_are_targets"):
        findings.append(
            f"Flash/Qwen27 not both targets: {presence.get('as_target')}"
        )
    findings.append(
        "No PROTECTED_ABSOLUTE or REPRODUCED evidence is available to this sidecar, "
        "so no law is seeded at GENERIC_VERIFIED. That is a correct UNKNOWN, not a skip."
    )
    findings.append(
        "BACKEND_FAMILY is uninhabited in the seed: every cited receipt is Metal-only. "
        "promote() still requires >=2 backends; we did not invent a CUDA/FPGA measurement."
    )
    findings.append(
        "time_to_first_useful_executable_ns is null on every law; a number here would "
        "be an executable-timing claim this lane cannot make."
    )
    if seed_report.get("atlas_seed_refusals"):
        findings.extend(f"seed refusal: {r}" for r in seed_report["atlas_seed_refusals"])
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print(selftest())
        return 0
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
