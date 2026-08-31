"""SCAR_SCHEDULING — a scar must cost something before work is admitted.

`negative_index.refuse_if_dead()` is the retrieval side: it cites a scar.
This module is the scheduling and propagation side. Rediscovery is no
longer free.

    admit(workunit)              query the index BEFORE the unit is scheduled
    propagate_failure(failure)   four systems at once, not a JSON write
    ingest_known_scars()         Codex handoff known_scars + receipts/seals

A refusal is a first-class outcome recorded against the unit, never an
exception a caller can swallow. Writing a failed JSON is not sufficient
propagation.

    python3 tools/future/scar_scheduling.py --selftest
    python3 tools/future/scar_scheduling.py --admit '{"model":"...","hypothesis_family":"..."}'
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, git, RECEIPTS

import argparse
import json
from pathlib import Path
from typing import Any

from tools.future import candidate_planner as cp
from tools.future import negative_index as ni
from tools.future import odyssey2_law_store as ols
from tools.future import odyssey3_adversary as o3


RECEIPT = "SCAR_SCHEDULING.json"
SCHEMA = "hawking.future.scar_scheduling.v1"
RECORDED_BY = "tools/future/scar_scheduling.py"
HANDOFF_REL = "CODEX_ACCELERATOR_HANDOFF.json"
O2_RECEIPT = "ODYSSEY2_LAW_STORE.json"

DECISION_ADMITTED = "ADMITTED"
DECISION_REFUSED = "REFUSED"

# Stated rule (c). A law that predicted success has its transfer_confidence
# halved, floored at CONFIDENCE_FLOOR. Zero would look like "there is no law".
CONFIDENCE_REDUCTION_FACTOR = 0.5
CONFIDENCE_FLOOR = 0.01
CONFIDENCE_REDUCTION_RULE = (
    "halve transfer_confidence.value (factor "
    f"{CONFIDENCE_REDUCTION_FACTOR}) when a scar refutes a predicted "
    f"success on a matching model/organ/hypothesis; floor {CONFIDENCE_FLOOR}"
)

# Odyssey II lattice is a WIDENING sequence. A refuting scar NARROWS toward
# MODEL_LOCAL. MACHINE_LOCAL is a machine-identity axis, not a model-organ
# claim, so a model-organ scar does not move it.
O2_NARROW = {
    "GENERIC_VERIFIED": "GENERIC_CANDIDATE",
    "GENERIC_CANDIDATE": "ARCHITECTURE_FAMILY",
    "BACKEND_FAMILY": "ARCHITECTURE_FAMILY",
    "ARCHITECTURE_FAMILY": "MODEL_LOCAL",
    "MACHINE_LOCAL": "MACHINE_LOCAL",
    "MODEL_LOCAL": "MODEL_LOCAL",
}

HYPOTHESIS_KEYS = (
    "hypothesis_family",
    "technique",
    "mechanism",
    "lever",
    "seed",
    "family",
)

WORKUNIT_ID = "future.scar-scheduling.admission-gate"

# Isolated four-way proof. Not a physical measurement. Names are deliberately
# off the live corpus so the proof does not depend on HEAD counts.
FOUR_WAY_PARENT = "fix-kronecker-parent"
FOUR_WAY_CHILD_A = "fix-kronecker-child-a"
FOUR_WAY_CHILD_B = "fix-kronecker-child-b"
FOUR_WAY_UNRELATED = "fix-unrelated-lm-head"
FOUR_WAY_MODEL = "fixture-parent-alpha"


def _clip(text: Any, n: int = 320) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, str):
        try:
            loaded = json.loads(obj)
        except json.JSONDecodeError:
            return {"hypothesis_family": obj}
        return dict(loaded) if isinstance(loaded, dict) else {"hypothesis_family": obj}
    if hasattr(obj, "to_dict"):
        d = obj.to_dict()
        return dict(d) if isinstance(d, dict) else {}
    out: dict[str, Any] = {}
    for k in (
        "id",
        "model",
        "organ",
        "organ_class",
        "representation",
        "machine",
        "hypothesis_family",
        "technique",
        "mechanism",
        "candidate_id",
        "status",
        "extras",
        "description",
        "role",
    ):
        if hasattr(obj, k):
            out[k] = getattr(obj, k)
    return out


def _pick(src: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        v = src.get(k)
        if v not in (None, "", [], {}):
            return v
    return None


def proposal_from_workunit(workunit: Any) -> dict[str, Any]:
    """Minimal local WorkUnit -> negative_index proposal. Swap point: resident_api."""
    raw = _as_dict(workunit)
    extras = raw.get("extras") if isinstance(raw.get("extras"), dict) else {}
    nested = raw.get("proposal") if isinstance(raw.get("proposal"), dict) else {}

    def pick(*keys: str) -> Any:
        for src in (raw, extras, nested):
            v = _pick(src, *keys)
            if v not in (None, "", [], {}):
                return v
        return None

    family = pick(*HYPOTHESIS_KEYS)
    return {
        "id": pick("id", "workunit_id", "candidate_id"),
        "model": pick("model", "source_model", "parent"),
        "organ": pick("organ", "organ_class"),
        "representation": pick("representation", "codec"),
        "machine": pick("machine", "source_device"),
        "hypothesis_family": family,
        "technique": pick("technique"),
        "mechanism": pick("mechanism", "failure_mechanism"),
        "lever": pick("lever"),
        "candidate_id": pick("candidate_id"),
        "predicted_success": pick("predicted_success"),
    }


def _checkout_roots() -> list[Path]:
    """This worktree, then the primary checkout (git common-dir parent)."""
    roots: list[Path] = [REPO]
    common = git("rev-parse", "--git-common-dir")
    if common:
        path = Path(common)
        if not path.is_absolute():
            path = (REPO / path).resolve()
        else:
            path = path.resolve()
        parent = path.parent if path.name == ".git" else path.parent
        if parent not in roots:
            roots.append(parent)
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def load_handoff() -> dict[str, Any]:
    """Recover CODEX_ACCELERATOR_HANDOFF.json. Sparse absence is not absence."""
    searched: list[str] = []
    for root in _checkout_roots():
        path = root / HANDOFF_REL
        searched.append(str(path))
        if path.is_file():
            try:
                doc = load_json(path)
            except (OSError, json.JSONDecodeError, ValueError) as e:
                return {
                    "present": None,
                    "source": "unreadable",
                    "path": str(path),
                    "error": str(e),
                    "known_scars": [],
                    "searched": searched,
                }
            if not isinstance(doc, dict):
                return {
                    "present": None,
                    "source": "unreadable",
                    "path": str(path),
                    "error": "root is not an object",
                    "known_scars": [],
                    "searched": searched,
                }
            scars = doc.get("known_scars") or []
            if not isinstance(scars, list):
                scars = []
            return {
                "present": True,
                "source": "disk",
                "path": str(path.relative_to(root) if path.is_relative_to(root) else path),
                "resolved": str(path),
                "schema": doc.get("schema"),
                "known_scars": [s for s in scars if isinstance(s, dict)],
                "searched": searched,
            }
    pinned = RECEIPTS / "evidence" / Path(HANDOFF_REL).name
    searched.append(str(pinned))
    if pinned.is_file():
        try:
            doc = load_json(pinned)
        except (OSError, json.JSONDecodeError, ValueError):
            doc = {}
        scars = doc.get("known_scars") or [] if isinstance(doc, dict) else []
        return {
            "present": True,
            "source": "pinned_snapshot",
            "path": str(pinned.relative_to(REPO)),
            "resolved": str(pinned),
            "known_scars": [s for s in scars if isinstance(s, dict)],
            "searched": searched,
        }
    blob = git("show", f"HEAD:{HANDOFF_REL}")
    searched.append(f"git:HEAD:{HANDOFF_REL}")
    if blob:
        try:
            doc = json.loads(blob)
        except json.JSONDecodeError as e:
            return {
                "present": None,
                "source": "git_unreadable",
                "error": str(e),
                "known_scars": [],
                "searched": searched,
            }
        scars = doc.get("known_scars") or [] if isinstance(doc, dict) else []
        return {
            "present": True,
            "source": "git",
            "path": f"HEAD:{HANDOFF_REL}",
            "known_scars": [s for s in scars if isinstance(s, dict)],
            "searched": searched,
        }
    return {
        "present": None,
        "source": "unresolved",
        "note": (
            "CODEX_ACCELERATOR_HANDOFF.json is not in this worktree, the primary "
            "checkout, the pinned evidence snapshot, or git HEAD. Sparse checkout "
            "is not evidence of absence."
        ),
        "known_scars": [],
        "searched": searched,
    }


def load_o2_laws() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load Odyssey II laws if the store is on disk. Missing is a path, not a fail."""
    path = RECEIPTS / O2_RECEIPT
    if not path.is_file():
        return [], {"present": None, "path": str(path.relative_to(REPO)), "n": 0}
    try:
        doc = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        return [], {"present": None, "path": str(path.relative_to(REPO)), "error": str(e), "n": 0}
    laws = doc.get("laws") or []
    if not isinstance(laws, list):
        laws = []
    out = [l for l in laws if isinstance(l, dict) and l.get("law_id")]
    return out, {"present": True, "path": str(path.relative_to(REPO)), "n": len(out)}


def demote_odyssey2_scope(scope: str) -> dict[str, Any]:
    """Narrow an Odyssey II scope by the stated rule. Never widens."""
    before = str(scope or "")
    after = O2_NARROW.get(before, before)
    moved = after != before
    return {
        "scope_before": before,
        "scope_after": after,
        "moved": moved,
        "direction": "DOWN" if moved else "NONE",
        "rule": (
            "Odyssey II lattice is widening; a refuting scar narrows toward "
            "MODEL_LOCAL. MACHINE_LOCAL is a machine-identity axis and does not "
            "move on a model-organ scar."
        ),
        "mapping": dict(sorted(O2_NARROW.items())),
    }


def four_way_fixture() -> dict[str, Any]:
    """Deterministic isolated inputs for the four-way proof. Synthetic, labelled."""
    law = {
        "law_id": "LAW-FIXTURE-KRONECKER-ON-GATE",
        "statement": (
            "kronecker factorization of the gate organ preserves capability "
            "across this architecture family"
        ),
        "source_model": FOUR_WAY_MODEL,
        "source_device": "UNKNOWN",
        "architecture_family": "fixture_family",
        "organ_class": "gate",
        "backend": "UNKNOWN",
        "evidence_strength": "STATIC",
        "evidence_refs": ["tools/future/scar_scheduling.py#four_way_fixture"],
        "scope": "ARCHITECTURE_FAMILY",
        "transfer_candidates": [],
        "transfer_confidence": {
            "value": 0.64,
            "basis": "fixture prior before scar; not a measurement",
        },
        "counterexample_requirement": (
            "a kronecker packing of gate whose capability collapses"
        ),
        "expected_saved_experiments": None,
        "actual_saved_experiments": None,
        "time_to_first_useful_executable_ns": None,
    }
    failure = {
        "model": FOUR_WAY_MODEL,
        "organ": "gate",
        "representation": "kronecker",
        "machine": "m3_ultra",
        "hypothesis_family": "kronecker",
        "mechanism": (
            "kronecker packing destroyed gate magnitude; capability collapsed"
        ),
        "candidate_id": FOUR_WAY_PARENT,
        "verdict": "REFUTED",
        "predicted_success": True,
        "source_path": "tools/future/scar_scheduling.py#four_way_fixture",
        "synthetic": True,
    }
    return {
        "law": law,
        "failure": failure,
        "declared_edges": [
            {"from": FOUR_WAY_PARENT, "to": FOUR_WAY_CHILD_A},
            {"from": FOUR_WAY_PARENT, "to": FOUR_WAY_CHILD_B},
        ],
        "unrelated_candidate_id": FOUR_WAY_UNRELATED,
        "dead_unit": {
            "id": "future.scar-scheduling.fixture.dead-kronecker-gate",
            "model": FOUR_WAY_MODEL,
            "organ": "gate",
            "representation": "kronecker",
            "machine": "m3_ultra",
            "hypothesis_family": "kronecker",
            "candidate_id": FOUR_WAY_CHILD_A,
        },
        "near_miss_unit": {
            "id": "future.scar-scheduling.fixture.near-miss-lmhead-q4",
            "model": FOUR_WAY_MODEL,
            "organ": "lm_head",
            "representation": "q4",
            "machine": "m3_ultra",
            "hypothesis_family": "kronecker",
            "candidate_id": FOUR_WAY_UNRELATED,
        },
        "confidence_before": 0.64,
        "confidence_after": 0.32,
        "scope_before": "ARCHITECTURE_FAMILY",
        "scope_after": "MODEL_LOCAL",
        "synthetic": True,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }


def emit_resident_workunit() -> dict[str, Any]:
    """Local WorkUnit shape. Swap: workunit_species.emit_hcli_workunit."""
    return {
        "id": WORKUNIT_ID,
        "role": "science",
        "description": (
            "Query the scar index before a unit is scheduled; refuse a known-dead "
            "hypothesis citing the scar; on a fresh failure store the scar, "
            "invalidate named dependents, reduce the confidence of any law that "
            "predicted success, narrow Odyssey II scope and emit an Odyssey III "
            "implication. Rediscovery is not free."
        ),
        "dependencies": [],
        "status": "pending",
        "assigned_runtime": None,
        "attempts": 0,
        "resource_class": "STATIC_ANALYSIS",
        "repairs": None,
        "failure_context": None,
        "preferred_backend": None,
        "assigned_backend": None,
        "backend_task_id": None,
        "verifier": "future.scar_scheduling.receipt_sealed",
        "effect_class": "READ_ONLY",
        "workspace": "repo-root",
        "verification": None,
        "repair_root": None,
        "repair_depth": 0,
        "repair_reason": None,
        "repair_exhausted": False,
        "ready_at": None,
        "running_at": None,
        "finished_at": None,
        "classification": "STATIC_ONLY",
        "provider": "future.scar_scheduling",
        "content_hash": None,
        "claim_boundary": (
            "WorkUnit is a proposal; receipts/future/SCAR_SCHEDULING.json remains "
            "authoritative. STATIC_ONLY / bench UNKNOWN. Cannot acquire a GPU "
            "lease or promote evidence class."
        ),
        "output_receipt_path": f"receipts/future/{RECEIPT}",
        "command": ["python3", "tools/future/scar_scheduling.py", "--selftest"],
        "frontier": "negative_science",
        "species": "scar_scheduling_admission",
        "requires_quiescence": False,
    }


def _summary_text(entry: dict[str, Any]) -> str:
    summary = entry.get("summary")
    if isinstance(summary, dict):
        return json.dumps(summary, sort_keys=True, default=str)
    return str(summary or "")


def _keys_from_known_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Derive index keys from a Codex known_scar object. No invented parents."""
    name = str(entry.get("name") or "unnamed")
    receipt = str(entry.get("receipt") or "")
    summary = _summary_text(entry)
    state = str(entry.get("state") or "")
    blob = " ".join(part for part in (name, summary, state, receipt) if part)
    models = ni.extract_models(blob)
    organs = ni.extract_organs(blob)
    family = ni.canon_family(name.replace("-", "_"))
    # Summary-driven family when the name is an event, not a technique.
    # Do not use a 'synthetic_*' slug: canon_family maps those onto
    # synthetic_activation (Gaussian-proxy scars), which is a different kill.
    low = blob.lower()
    if "no synthetic" in low and "teacher" in low:
        family = "no_invented_teacher_rows"
    elif "no blind cleanup" in low or ("stale" in low and "lock" in low):
        family = "blind_lock_cleanup"
    elif "does not prove speed" in low or "static correctness does not prove" in low:
        family = "static_preflight_proves_physical"
    elif "equivalence siblings" in low or "hard-invalidates" in low:
        family = "cross_model_equivalence_assumed"
    representation = ni.canon_representation(blob)
    machine = ni.canon_machine(blob)
    if machine == ni.UNRECORDED and "metal" in low:
        machine = "metal"
    reopen = ni.UNRECORDED
    if "no metal-capable gpu" in low or "0/256" in low:
        reopen = "Metal-capable GPU present and teacher capture rows > 0; no synthetic rows"
    if "rolled back" in low:
        reopen = "a head fusion whose complete-token path does not recompute RMSNorm per lm-head group"
    verdict = "REFUTED" if any(
        tok in low for tok in ("rolled back", "refuted", "dead", "does not prove", "no synthetic", "no blind")
    ) else (state or "RECORDED")
    # Resolved tooling findings with zero outstanding defects are memory, not a refuse.
    refuse = True
    if isinstance(entry.get("summary"), dict):
        sm = entry["summary"]
        if sm.get("outstanding_confirmed_defects") == 0 and sm.get("dead_paths") == 0:
            refuse = False
            verdict = "RESOLVED"
    if "zero outstanding" in low and "resolved" in low:
        refuse = False
        verdict = "RESOLVED"
    sleeping = bool("no metal-capable gpu" in low or "0/256" in low)
    if sleeping and "no synthetic" in low:
        # The dead hypothesis is synthesizing rows. The capture itself sleeps.
        refuse = True
    return {
        "original_id": name,
        "model": models[0] if models else ni.UNRECORDED,
        "models": models or [ni.UNRECORDED],
        "organ": organs[0] if organs else ni.UNRECORDED,
        "organs": organs or [ni.UNRECORDED],
        "representation": representation,
        "machine": machine,
        "hypothesis_family": family,
        "failure_mechanism": _clip(summary or state or name),
        "verdict": _clip(verdict),
        "refuse_eligible": refuse,
        "sleeping": sleeping,
        "reopen_condition": reopen,
        "claim_refuted": _clip(summary or name),
        "level": "MODEL_SPECIFIC",
        "source_path": receipt or f"{HANDOFF_REL}#{name}",
        "receipt": receipt or None,
        "receipt_seal_sha256": entry.get("receipt_seal_sha256"),
        "name": name,
    }


def _scar_from_keys(
    keys: dict[str, Any],
    *,
    origin: str,
    scar_id: str,
) -> ni.Scar:
    models = list(keys.get("models") or [keys.get("model") or ni.UNRECORDED])
    organs = list(keys.get("organs") or [keys.get("organ") or ni.UNRECORDED])
    return ni.Scar(
        scar_id=scar_id,
        source_path=str(keys.get("source_path") or origin),
        source_origin=origin,
        parse_status=ni.PARSED,
        model=str(keys.get("model") or ni.UNRECORDED),
        models=[str(m) for m in models],
        organ=str(keys.get("organ") or ni.UNRECORDED),
        organs=[str(o) for o in organs],
        representation=str(keys.get("representation") or ni.UNRECORDED),
        machine=str(keys.get("machine") or ni.UNRECORDED),
        hypothesis_family=str(keys.get("hypothesis_family") or ni.UNRECORDED),
        failure_mechanism=str(keys.get("failure_mechanism") or ni.UNRECORDED),
        verdict=str(keys.get("verdict") or "REFUTED"),
        refuse_eligible=bool(keys.get("refuse_eligible", True)),
        reopen_condition=str(keys.get("reopen_condition") or ni.UNRECORDED),
        claim_refuted=str(keys.get("claim_refuted") or ni.UNRECORDED),
        level=str(keys.get("level") or "MODEL_SPECIFIC"),
        original_id=str(keys.get("original_id") or scar_id),
    ).finalize()


def _same_known(scar: ni.Scar, entry: dict[str, Any]) -> bool:
    name = str(entry.get("name") or "")
    receipt = str(entry.get("receipt") or "")
    if not name:
        return False
    if scar.original_id == name:
        return True
    sid = scar.scar_id
    if sid.endswith("#" + name) or sid.endswith(":" + name) or sid.endswith("/" + name):
        return True
    if receipt and scar.source_path == receipt and name.replace("-", "_") in sid.replace("-", "_"):
        return True
    return False


def _retain_name(raw: str, canon: str) -> str:
    """Keep a caller-named token when the catalog has no alias for it."""
    if canon != ni.UNRECORDED:
        return canon
    slug = ni._slug(raw)
    return slug if slug else ni.UNRECORDED


def _canon_tuple(proposal: dict[str, Any]) -> tuple[str, str, str, str, str]:
    model = str(proposal.get("model") or "")
    organ = str(proposal.get("organ") or "")
    representation = str(proposal.get("representation") or "")
    family = str(proposal.get("hypothesis_family") or proposal.get("technique") or "")
    machine = str(proposal.get("machine") or "")
    return (
        _retain_name(model, ni.canon_model(model)),
        _retain_name(organ, ni.canon_organ(organ)),
        _retain_name(representation, ni.canon_representation(representation)),
        _retain_name(family, ni.canon_family(family)),
        _retain_name(machine, ni.canon_machine(machine)),
    )


def _law_predicted_success(law: dict[str, Any], failure: dict[str, Any]) -> bool:
    """True when this law claimed the failed hypothesis would hold.

    Organ must match when both sides name one. Model must match exactly, or
    the law's Odyssey II scope must be wider than MODEL_LOCAL (a family-level
    claim covers in-family parents). Hypothesis overlap is the family slug
    appearing in law_id / statement / organ_class.
    """
    family = ni.canon_family(
        str(
            failure.get("hypothesis_family")
            or failure.get("technique")
            or failure.get("mechanism")
            or ""
        )
    )
    if not family or family == ni.UNRECORDED:
        return False
    blob = ni._slug(
        " ".join(
            str(law.get(k) or "")
            for k in ("law_id", "statement", "organ_class", "hypothesis_family")
        )
    )
    if family not in blob and ni.canon_family(str(law.get("organ_class") or "")) != family:
        return False
    f_organ = ni.canon_organ(str(failure.get("organ") or ""))
    l_organ = ni.canon_organ(str(law.get("organ_class") or law.get("organ") or ""))
    if f_organ != ni.UNRECORDED and l_organ != ni.UNRECORDED and f_organ != l_organ:
        return False
    f_model = ni.canon_model(str(failure.get("model") or ""))
    l_model = ni.canon_model(str(law.get("source_model") or law.get("model") or ""))
    scope = str(law.get("scope") or "MODEL_LOCAL")
    wider = scope in {
        "ARCHITECTURE_FAMILY",
        "BACKEND_FAMILY",
        "GENERIC_CANDIDATE",
        "GENERIC_VERIFIED",
        "FAMILY_VERIFIED",
    }
    if f_model != ni.UNRECORDED and l_model != ni.UNRECORDED and f_model != l_model and not wider:
        return False
    return True


def _confidence_value(law: dict[str, Any]) -> float | None:
    tc = law.get("transfer_confidence")
    if isinstance(tc, dict):
        v = tc.get("value")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return None
    if isinstance(tc, (int, float)) and not isinstance(tc, bool):
        return float(tc)
    return None


def _apply_confidence_reduction(law: dict[str, Any], scar_id: str) -> dict[str, Any] | None:
    before = _confidence_value(law)
    if before is None:
        return None
    after = max(CONFIDENCE_FLOOR, round(before * CONFIDENCE_REDUCTION_FACTOR, 4))
    tc = law.get("transfer_confidence")
    basis_suffix = f"; {CONFIDENCE_REDUCTION_RULE}; scar={scar_id}; before={before}"
    if isinstance(tc, dict):
        law["transfer_confidence"] = {
            "value": after,
            "basis": str(tc.get("basis") or "") + basis_suffix,
        }
    else:
        law["transfer_confidence"] = after
    return {
        "law_id": law.get("law_id"),
        "before": before,
        "after": after,
        "factor": CONFIDENCE_REDUCTION_FACTOR,
        "floor": CONFIDENCE_FLOOR,
        "rule": CONFIDENCE_REDUCTION_RULE,
        "scar_id": scar_id,
    }


def _o3_law_from_failure(failure: dict[str, Any], scar_id: str) -> dict[str, Any]:
    model = str(failure.get("model") or "UNKNOWN")
    organ = str(failure.get("organ") or "UNKNOWN")
    family = str(failure.get("hypothesis_family") or failure.get("technique") or "UNKNOWN")
    statement = (
        f"{family} on {organ} of {model} was predicted to succeed; the scar "
        f"{scar_id} is the counterexample"
    )
    return {
        "law_id": f"LAW-FROM-SCAR-{ni._slug(scar_id)[:80]}",
        "statement": statement,
        "source_model": model,
        "source_device": str(failure.get("machine") or "UNKNOWN"),
        "architecture_family": "UNKNOWN",
        "organ_class": organ,
        "backend": "UNKNOWN",
        "evidence_strength": "ANECDOTE",
        "evidence_refs": [
            str(failure.get("source_path") or scar_id),
        ],
        "scope": "FAMILY_VERIFIED",
        "transfer_candidates": ["hostile_holdout"],
        "transfer_confidence": 0.40,
        "counterexample_requirement": str(
            failure.get("mechanism") or "this failure"
        ),
    }


class ScarScheduler:
    """Admission gate + four-way propagation. In-memory overlay on the index."""

    def __init__(
        self,
        *,
        include_corpus: bool = True,
        declared_edges: list[dict[str, str]] | None = None,
        laws: list[dict[str, Any]] | None = None,
    ) -> None:
        self.include_corpus = include_corpus
        self.declared_edges: list[dict[str, str]] = list(declared_edges or [])
        self.laws: list[dict[str, Any]] = [dict(l) for l in (laws or [])]
        self.extra_scars: list[ni.Scar] = []
        self.invalidated: dict[str, dict[str, Any]] = {}
        self.refusals: list[dict[str, Any]] = []
        self.admissions: list[dict[str, Any]] = []
        self.ingest_report: dict[str, Any] | None = None
        self._propagated: dict[str, dict[str, Any]] = {}
        self.confidence_reductions: list[dict[str, Any]] = []
        self.scope_updates: list[dict[str, Any]] = []
        self.o3_implications: list[dict[str, Any]] = []

    def pool(self) -> list[ni.Scar]:
        extra = list(self.extra_scars)
        if not self.include_corpus:
            return extra
        return extra + list(ni.ingest())

    def store_scar(self, keys: dict[str, Any], *, origin: str, scar_id: str) -> ni.Scar:
        scar = _scar_from_keys(keys, origin=origin, scar_id=scar_id)
        # Idempotent identity: same scar_id is a no-op replace.
        self.extra_scars = [s for s in self.extra_scars if s.scar_id != scar.scar_id]
        self.extra_scars.append(scar)
        self.extra_scars.sort(key=lambda s: s.scar_id)
        return scar

    def admit(self, workunit: Any) -> dict[str, Any]:
        """Refuse a known-dead hypothesis; admit a structurally different one.

        Never raises on a dead hypothesis. Index failure fail-closes as REFUSED.
        """
        raw = _as_dict(workunit)
        annotated = dict(raw)
        proposal = proposal_from_workunit(raw)
        unit_id = str(proposal.get("id") or proposal.get("candidate_id") or "")
        try:
            if unit_id and unit_id in self.invalidated:
                inv = self.invalidated[unit_id]
                refusal = {
                    "refused": True,
                    "reason": (
                        "dependent of a scarred candidate; hard-invalidated by "
                        f"{inv.get('scar_id')}"
                    ),
                    "scar_id": inv.get("scar_id"),
                    "source_path": inv.get("source_path"),
                    "invalidated_by": inv.get("failed_candidate_id"),
                    "hypothesis_family": proposal.get("hypothesis_family"),
                }
                return self._record_refusal(annotated, unit_id, proposal, refusal)
            pool = self.pool()
            family = proposal.get("hypothesis_family")
            if not family:
                outcome = {
                    "decision": DECISION_ADMITTED,
                    "workunit_id": unit_id,
                    "reason": (
                        "no hypothesis_family/technique/mechanism/lever; this is "
                        "not a scientific-hypothesis gate"
                    ),
                    "workunit": annotated,
                    "evidence_class": "STATIC_ONLY",
                    "bench_state": "UNKNOWN",
                }
                self.admissions.append(outcome)
                return outcome
            refusal = ni.refuse_if_dead(proposal, scars=pool)
        except Exception as e:  # fail closed
            refusal = {
                "refused": True,
                "reason": f"fail-closed: scar index unavailable ({type(e).__name__}: {e})",
                "scar_id": None,
                "source_path": None,
                "fail_closed": True,
            }
            return self._record_refusal(annotated, unit_id, proposal, refusal)

        if refusal is None:
            outcome = {
                "decision": DECISION_ADMITTED,
                "workunit_id": unit_id,
                "proposal": {
                    "model": proposal.get("model"),
                    "organ": proposal.get("organ"),
                    "representation": proposal.get("representation"),
                    "machine": proposal.get("machine"),
                    "hypothesis_family": proposal.get("hypothesis_family"),
                },
                "reason": "no refuse-eligible scar matches this hypothesis on this parent",
                "workunit": annotated,
                "evidence_class": "STATIC_ONLY",
                "bench_state": "UNKNOWN",
            }
            self.admissions.append(outcome)
            return outcome
        return self._record_refusal(annotated, unit_id, proposal, refusal)

    def _record_refusal(
        self,
        annotated: dict[str, Any],
        unit_id: str,
        proposal: dict[str, Any],
        refusal: dict[str, Any],
    ) -> dict[str, Any]:
        annotated["status"] = "blocked"
        annotated["classification"] = "SCAR_REFUSED"
        annotated["blocked_reason"] = (
            f"known-dead hypothesis; scar_id={refusal.get('scar_id')}; "
            f"{refusal.get('reason')}"
        )
        annotated["failure_context"] = {
            "decision": DECISION_REFUSED,
            "scar_id": refusal.get("scar_id"),
            "source_path": refusal.get("source_path"),
            "hypothesis_family": refusal.get("hypothesis_family")
            or proposal.get("hypothesis_family"),
            "reason": refusal.get("reason"),
        }
        outcome = {
            "decision": DECISION_REFUSED,
            "workunit_id": unit_id,
            "scar_id": refusal.get("scar_id"),
            "source_path": refusal.get("source_path"),
            "original_id": refusal.get("original_id"),
            "hypothesis_family": refusal.get("hypothesis_family")
            or proposal.get("hypothesis_family"),
            "model": refusal.get("model") or proposal.get("model"),
            "organ": refusal.get("organ") or proposal.get("organ"),
            "representation": proposal.get("representation"),
            "verdict": refusal.get("verdict"),
            "failure_mechanism": refusal.get("failure_mechanism"),
            "reopen_condition": refusal.get("reopen_condition"),
            "reason": refusal.get("reason"),
            "fail_closed": bool(refusal.get("fail_closed")),
            "workunit": annotated,
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
        }
        self.refusals.append(outcome)
        return outcome

    def invalidate_dependents(
        self,
        failed_candidate_id: str | None,
        *,
        scar_id: str,
        source_path: str,
        extra_named: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        ids: list[str] = []
        if failed_candidate_id and self.declared_edges:
            ids.extend(cp.descendants_of(failed_candidate_id, self.declared_edges))
        for name in extra_named or []:
            if name and name not in ids and name != failed_candidate_id:
                ids.append(name)
        named: list[dict[str, Any]] = []
        for ident in ids:
            rec = {
                "candidate_id": ident,
                "failed_candidate_id": failed_candidate_id,
                "scar_id": scar_id,
                "source_path": source_path,
                "reason": (
                    "rejection hard-invalidates declared descendants "
                    "(candidate_planner.lineage_scars / descendants_of)"
                ),
            }
            self.invalidated[ident] = rec
            named.append(rec)
        named.sort(key=lambda r: str(r["candidate_id"]))
        return named

    def propagate_failure(self, failure: Any) -> dict[str, Any]:
        """Four systems at once. Writing a JSON file is not this function.

        (a) store the scar with the six index keys
        (b) invalidate named dependents
        (c) reduce confidence of any law that predicted success
        (d) narrow Odyssey II scope and generate an Odyssey III implication
        """
        raw = _as_dict(failure)
        family = str(
            raw.get("hypothesis_family")
            or raw.get("technique")
            or raw.get("mechanism")
            or ""
        )
        model = str(raw.get("model") or "")
        organ = str(raw.get("organ") or "")
        representation = str(raw.get("representation") or "")
        machine = str(raw.get("machine") or "")
        mechanism = str(raw.get("mechanism") or raw.get("failure_mechanism") or family)
        candidate_id = str(raw.get("candidate_id") or "")
        retained = _canon_tuple({
            "model": model,
            "organ": organ,
            "representation": representation,
            "hypothesis_family": family,
            "machine": machine,
        })
        key = "|".join(retained)
        if key in self._propagated:
            out = dict(self._propagated[key])
            out["duplicate"] = True
            return out

        scar_id = (
            "scar_scheduling:fresh:"
            + ":".join(part if part != ni.UNRECORDED else "_" for part in retained)
        )
        keys = {
            "original_id": scar_id,
            "model": retained[0],
            "models": ni.extract_models(model) if model else [ni.UNRECORDED],
            "organ": retained[1],
            "organs": ni.extract_organs(organ) if organ else [ni.UNRECORDED],
            "representation": retained[2],
            "machine": retained[4],
            "hypothesis_family": retained[3],
            "failure_mechanism": _clip(mechanism),
            "verdict": str(raw.get("verdict") or "REFUTED"),
            "refuse_eligible": True,
            "reopen_condition": str(raw.get("reopen_condition") or ni.UNRECORDED),
            "claim_refuted": _clip(raw.get("claim_refuted") or mechanism),
            "level": "MODEL_SPECIFIC",
            "source_path": str(raw.get("source_path") or scar_id),
        }
        if keys["models"] == [ni.UNRECORDED] and keys["model"] != ni.UNRECORDED:
            keys["models"] = [keys["model"]]
        elif keys["model"] != ni.UNRECORDED and keys["model"] not in keys["models"]:
            keys["models"] = [keys["model"]] + [m for m in keys["models"] if m != ni.UNRECORDED]
        if keys["organs"] == [ni.UNRECORDED] and keys["organ"] != ni.UNRECORDED:
            keys["organs"] = [keys["organ"]]
        elif keys["organ"] != ni.UNRECORDED and keys["organ"] not in keys["organs"]:
            keys["organs"] = [keys["organ"]] + [o for o in keys["organs"] if o != ni.UNRECORDED]

        # (a)
        scar = self.store_scar(keys, origin="scar_scheduling.fresh_failure", scar_id=scar_id)
        scar_dict = scar.to_dict()

        # (b)
        extra_named = raw.get("dependent_candidates")
        if not isinstance(extra_named, list):
            extra_named = []
        invalidated = self.invalidate_dependents(
            candidate_id or None,
            scar_id=scar_id,
            source_path=str(keys["source_path"]),
            extra_named=[str(x) for x in extra_named],
        )

        # (c) + Odyssey II half of (d)
        reductions: list[dict[str, Any]] = []
        scope_updates: list[dict[str, Any]] = []
        for law in self.laws:
            if not _law_predicted_success(law, raw):
                continue
            red = _apply_confidence_reduction(law, scar_id)
            if red is not None:
                reductions.append(red)
                self.confidence_reductions.append(red)
            before_scope = str(law.get("scope") or "")
            demote = demote_odyssey2_scope(before_scope)
            law["scope"] = demote["scope_after"]
            update = {
                "law_id": law.get("law_id"),
                "scar_id": scar_id,
                **demote,
            }
            scope_updates.append(update)
            self.scope_updates.append(update)

        # (d) Odyssey III implication — always, from the failure itself
        implication: dict[str, Any]
        try:
            o3_law = _o3_law_from_failure(raw, scar_id)
            plan = o3.emit_for_law(o3_law)
            ranked = list(plan.get("ranked_attacks") or [])
            attack = ranked[0] if ranked else None
            if attack is None:
                raise o3.NoAttackError(f"{o3_law['law_id']}: no attack")
            applied = o3.apply_result(
                o3_law,
                attack,
                {
                    "verdict": "REFUTED",
                    "synthetic": True,
                    "reason": mechanism or "fresh failure",
                    "evidence_class": "STATIC_ONLY",
                    "bench_state": "UNKNOWN",
                },
            )
            implication = {
                "law_id": o3_law["law_id"],
                "attack_id": applied.get("attack_id"),
                "family": applied.get("family"),
                "verdict": applied.get("verdict"),
                "scope_before": applied.get("scope_before"),
                "scope_after": applied.get("scope_after"),
                "moved": applied.get("moved"),
                "direction": applied.get("direction"),
                "selected_attack_id": plan.get("selected_attack_id"),
                "n_attacks": plan.get("n_attacks"),
                "synthetic": True,
                "evidence_class": "STATIC_ONLY",
                "bench_state": "UNKNOWN",
            }
            if not implication["moved"]:
                raise o3.ScopeUnmovedError(
                    f"{o3_law['law_id']}: Odyssey III implication did not move scope"
                )
        except (o3.LawSchemaError, o3.NoAttackError, o3.ScopeUnmovedError, ValueError) as e:
            implication = {
                "error": str(e),
                "moved": False,
                "fail_closed": True,
                "evidence_class": "STATIC_ONLY",
                "bench_state": "UNKNOWN",
            }
        self.o3_implications.append(implication)

        out = {
            "scar_stored": True,
            "scar": {
                "scar_id": scar_dict["scar_id"],
                "model": scar_dict["model"],
                "organ": scar_dict["organ"],
                "representation": scar_dict["representation"],
                "machine": scar_dict["machine"],
                "hypothesis_family": scar_dict["hypothesis_family"],
                "failure_mechanism": scar_dict["failure_mechanism"],
                "source_path": scar_dict["source_path"],
                "refuse_eligible": scar_dict["refuse_eligible"],
                "keys_filled": scar_dict["keys_filled"],
                "verdict": scar_dict["verdict"],
            },
            "invalidated_candidates": invalidated,
            "invalidated_candidate_ids": [r["candidate_id"] for r in invalidated],
            "confidence_reductions": reductions,
            "odyssey2_scope_updates": scope_updates,
            "odyssey3_implication": implication,
            "duplicate": False,
            "synthetic": bool(raw.get("synthetic", False)),
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
        }
        self._propagated[key] = out
        return out

    def ingest_known_scars(self, handoff: dict[str, Any] | None = None) -> dict[str, Any]:
        loaded = handoff if handoff is not None else load_handoff()
        entries = list(loaded.get("known_scars") or [])
        corpus: list[ni.Scar] = []
        if self.include_corpus:
            try:
                corpus = list(ni.ingest())
            except Exception:
                corpus = []
        already = 0
        newly = 0
        records: list[dict[str, Any]] = []
        for entry in entries:
            name = str(entry.get("name") or "")
            present_in = None
            for existing in list(self.extra_scars) + corpus:
                if _same_known(existing, entry):
                    present_in = existing.scar_id
                    break
            keys = _keys_from_known_entry(entry)
            rec = {
                "name": name,
                "receipt": entry.get("receipt"),
                "receipt_seal_sha256": entry.get("receipt_seal_sha256"),
                "keys": {
                    "model": keys["model"],
                    "organ": keys["organ"],
                    "representation": keys["representation"],
                    "machine": keys["machine"],
                    "hypothesis_family": keys["hypothesis_family"],
                    "failure_mechanism": keys["failure_mechanism"],
                },
                "refuse_eligible": keys["refuse_eligible"],
                "sleeping": keys["sleeping"],
                "verdict": keys["verdict"],
            }
            if present_in:
                already += 1
                rec["status"] = "already_present"
                rec["existing_scar_id"] = present_in
            else:
                scar_id = f"scar_scheduling:codex:{name}"
                scar = self.store_scar(
                    keys, origin="CODEX_ACCELERATOR_HANDOFF.known_scars", scar_id=scar_id
                )
                newly += 1
                rec["status"] = "newly_added"
                rec["scar_id"] = scar.scar_id
            records.append(rec)
        records.sort(key=lambda r: str(r.get("name") or ""))
        report = {
            "handoff_source": loaded.get("source"),
            "handoff_path": loaded.get("path") or loaded.get("resolved"),
            "handoff_present": loaded.get("present"),
            "n_input": len(entries),
            "n_already_present": already,
            "n_newly_added": newly,
            "records": records,
            "note": loaded.get("note"),
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
        }
        self.ingest_report = report
        return report

    def experiments_prevented(self) -> dict[str, Any]:
        refused_ids = sorted({
            str(r.get("workunit_id") or r.get("scar_id") or "")
            for r in self.refusals
        })
        invalidated_ids = sorted(self.invalidated)
        unique = sorted({i for i in refused_ids + invalidated_ids if i})
        citations = []
        for r in self.refusals:
            citations.append(
                {
                    "kind": "refused_admission",
                    "workunit_id": r.get("workunit_id"),
                    "scar_id": r.get("scar_id"),
                    "hypothesis_family": r.get("hypothesis_family"),
                }
            )
        for ident in invalidated_ids:
            inv = self.invalidated[ident]
            citations.append(
                {
                    "kind": "invalidated_dependent",
                    "candidate_id": ident,
                    "scar_id": inv.get("scar_id"),
                    "failed_candidate_id": inv.get("failed_candidate_id"),
                }
            )
        citations.sort(key=lambda c: (str(c.get("kind")), str(c.get("workunit_id") or c.get("candidate_id") or "")))
        return {
            "n_refused_admissions": len(self.refusals),
            "n_invalidated_dependents": len(self.invalidated),
            "n_unique": len(unique),
            "n_total": len(self.refusals) + len(self.invalidated),
            "refused_workunit_ids": [r.get("workunit_id") for r in self.refusals],
            "invalidated_candidate_ids": invalidated_ids,
            "citations": citations,
            "rule": (
                "an experiment is prevented when admit() REFUSES it or when "
                "propagate_failure() names it as a hard-invalidated descendant"
            ),
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
        }


_SCHEDULER: ScarScheduler | None = None


def get_scheduler() -> ScarScheduler:
    global _SCHEDULER
    if _SCHEDULER is None:
        _SCHEDULER = ScarScheduler(include_corpus=True)
    return _SCHEDULER


def admit(workunit: Any, *, scheduler: ScarScheduler | None = None) -> dict[str, Any]:
    return (scheduler or get_scheduler()).admit(workunit)


def propagate_failure(failure: Any, *, scheduler: ScarScheduler | None = None) -> dict[str, Any]:
    return (scheduler or get_scheduler()).propagate_failure(failure)


def ingest_known_scars(*, scheduler: ScarScheduler | None = None) -> dict[str, Any]:
    return (scheduler or get_scheduler()).ingest_known_scars()


def experiments_prevented(*, scheduler: ScarScheduler | None = None) -> dict[str, Any]:
    return (scheduler or get_scheduler()).experiments_prevented()


def run_four_way_proof() -> dict[str, Any]:
    """Isolated four-way proof used by --selftest and the receipt."""
    fx = four_way_fixture()
    sch = ScarScheduler(
        include_corpus=False,
        declared_edges=list(fx["declared_edges"]),
        laws=[dict(fx["law"])],
    )
    prop = sch.propagate_failure(fx["failure"])
    dead = sch.admit(fx["dead_unit"])
    near = sch.admit(fx["near_miss_unit"])
    return {
        "propagation": prop,
        "dead_admit": dead,
        "near_miss_admit": near,
        "experiments_prevented": sch.experiments_prevented(),
        "fixture": {
            "model": FOUR_WAY_MODEL,
            "parent": FOUR_WAY_PARENT,
            "children": [FOUR_WAY_CHILD_A, FOUR_WAY_CHILD_B],
            "unrelated": FOUR_WAY_UNRELATED,
            "synthetic": True,
        },
        "scheduler_laws_after": [dict(l) for l in sch.laws],
    }


def recovered_implementation() -> list[dict[str, str]]:
    return [
        {
            "path": "tools/future/negative_index.py",
            "role": "keyed index + refuse_if_dead(); 51+ dead scars; retrieval only",
            "adequate": "no",
            "gap": "nothing called refuse_if_dead before a WorkUnit was scheduled; a fresh failure did not store, invalidate, demote, or attack",
        },
        {
            "path": "tools/future/candidate_planner.py",
            "role": "lineage_scars / descendants_of — rejection hard-invalidates declared descendants",
            "adequate": "no",
            "gap": "a table in a staged plan, not an admission gate a resident can invoke",
        },
        {
            "path": "tools/future/odyssey2_law_store.py",
            "role": "scope lattice + transfer_confidence; promote() widens only",
            "adequate": "no",
            "gap": "no demote(); a refuting scar could not reduce confidence or narrow scope",
        },
        {
            "path": "tools/future/odyssey3_adversary.py",
            "role": "emit_for_law / apply_result closed loop",
            "adequate": "no",
            "gap": "not wired to a scheduling failure; a JSON write was not an implication",
        },
        {
            "path": "tools/future/propagate.py",
            "role": "idempotent ingest-delta routing into seven consumers including the scar index",
            "adequate": "no",
            "gap": "consumes Codex deltas, does not admit WorkUnits or count experiments prevented",
        },
        {
            "path": "CODEX_ACCELERATOR_HANDOFF.json known_scars",
            "role": "six campaign scars with receipts/seals (training trace, not an archive)",
            "adequate": "no",
            "gap": "not ingested into the refuse path",
        },
    ]


def gaps_closed() -> list[str]:
    return [
        "admit(workunit) queries the scar index before schedule and REFUSES a known-dead hypothesis as a first-class outcome recorded on the unit.",
        "Four-way propagate_failure: (a) store six-key scar (b) name-invalidate dependents (c) stated confidence reduction (d) Odyssey II scope narrow + Odyssey III implication.",
        "Specificity: a scar on one organ/representation pair does not block a structurally different near-miss.",
        "Codex known_scars ingested with receipts and seals; already-present vs newly-added reported from data.",
        "experiments_prevented is counted from actual refusals and named invalidations, not from a file existing.",
        "Fail-closed: index exceptions become REFUSED, not silent ADMITTED.",
        "Idempotent re-propagation of the same hypothesis does not double-halve confidence.",
    ]


def negative_findings() -> list[str]:
    return [
        "CODEX_ACCELERATOR_HANDOFF.json is not in git HEAD of this worktree; it is recovered from the primary checkout when present. Sparse absence is not treated as zero scars.",
        "Odyssey II and Odyssey III use different scope ladders. Confidence reduction and O2 narrowing run on the O2 lattice; the O3 implication is a separately constructed law fed to apply_result.",
        "MACHINE_LOCAL O2 laws are not narrowed by a model-organ scar (different axis). That is a stated non-move, not a missed demotion.",
        "This lane produces STATIC_ONLY / bench UNKNOWN. It does not classify DIAGNOSTIC_RELATIVE vs PROTECTED_ABSOLUTE.",
        "claude-six-static-abi-findings is ingested as RESOLVED (zero outstanding defects) and is not refuse-eligible. A resolved tooling scar must not block new work.",
        "Flash teacher capture 0/256 is a hardware sleeper plus a refuse on synthetic rows; it is not a synthetic physical result.",
        "HCLI WorkUnit constructor is not imported (this-wave resident_api / workunit wiring is concurrent). Local field set is the swap point.",
    ]


def resident_callable_doc() -> dict[str, Any]:
    unit = emit_resident_workunit()
    return {
        "can_hcli_invoke": True,
        "entry_point": "python3 tools/future/scar_scheduling.py --selftest",
        "import": "from tools.future.scar_scheduling import admit, propagate_failure, ingest_known_scars",
        "admit_cli": "python3 tools/future/scar_scheduling.py --admit '{...}'",
        "workunit_emitted": {
            "id": unit["id"],
            "role": unit["role"],
            "resource_class": unit["resource_class"],
            "verifier": unit["verifier"],
            "effect_class": unit["effect_class"],
            "classification": unit["classification"],
            "command": unit["command"],
            "output_receipt_path": unit["output_receipt_path"],
            "species": unit["species"],
            "status": unit["status"],
        },
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier_fed": (
            "negative_science index overlay; Odyssey II scope lattice (narrowing); "
            "Odyssey III attack queue (implication); candidate invalidation set "
            "(descendants_of). Next admit() on a scarred hypothesis is REFUSED, "
            "so the frontier refills away from rediscovery."
        ),
        "fail_closed": (
            "admit() never raises on a dead hypothesis. If the scar index raises, "
            "the decision is REFUSED with fail_closed=true (work is not scheduled). "
            "An Odyssey III implication that does not move scope is recorded as "
            "fail_closed and selftest raises. Missing hypothesis_family is not a "
            "scientific refuse (plumbing units still admit). No GPU lease is taken; "
            "hardware-blocked capture is SLEEPING in the ingest record, never a "
            "synthetic measurement."
        ),
        "bounded_authority": [
            "query_negative_index",
            "propose_workunit",
            "write_sidecar_receipt",
            "read_receipts",
            "seed_law_store",
            "adversarially_attack_a_claimed_law",
        ],
        "forbidden_authority": [
            "acquire_gpu_lease",
            "promote_candidate",
            "claim_hardware_measurement",
            "mutate_codex_surface",
        ],
    }


def _probe_ingested(sch: ScarScheduler) -> dict[str, Any]:
    """Admit one unit per newly ingested refuse-eligible scar; record the path taken."""
    report = sch.ingest_report or {"records": []}
    probes: list[dict[str, Any]] = []
    for rec in report.get("records") or []:
        family = (rec.get("keys") or {}).get("hypothesis_family")
        name = rec.get("name")
        if not family:
            continue
        # Family-only: extra keys would filter unrecorded fields out of query().
        unit = {
            "id": f"future.scar-scheduling.ingest-probe.{name}",
            "hypothesis_family": family,
        }
        outcome = sch.admit(unit)
        probes.append(
            {
                "name": name,
                "hypothesis_family": family,
                "refuse_eligible": rec.get("refuse_eligible"),
                "decision": outcome.get("decision"),
                "scar_id": outcome.get("scar_id"),
            }
        )
    probes.sort(key=lambda p: str(p.get("name") or ""))
    return {"n": len(probes), "probes": probes}


def build(
    *,
    scheduler: ScarScheduler | None = None,
    four_way: dict[str, Any] | None = None,
    corpus_probe: dict[str, Any] | None = None,
) -> Path:
    sch = scheduler or ScarScheduler(include_corpus=True)
    ingest = sch.ingest_report or sch.ingest_known_scars()
    ingest_probes = _probe_ingested(sch)
    if four_way is None:
        four_way = run_four_way_proof()
    if corpus_probe is None:
        corpus_probe = _corpus_negative_control()
    o2_laws, o2_meta = load_o2_laws()
    ingest_prevented = sch.experiments_prevented()
    four_prevented = four_way.get("experiments_prevented") or {}
    corpus_prevented = (corpus_probe or {}).get("experiments_prevented") or {}
    prevented = {
        "n_refused_admissions": int(ingest_prevented.get("n_refused_admissions") or 0)
        + int(corpus_prevented.get("n_refused_admissions") or 0)
        + int(four_prevented.get("n_refused_admissions") or 0),
        "n_invalidated_dependents": int(ingest_prevented.get("n_invalidated_dependents") or 0)
        + int(four_prevented.get("n_invalidated_dependents") or 0),
        "n_total": int(ingest_prevented.get("n_total") or 0)
        + int(corpus_prevented.get("n_total") or 0)
        + int(four_prevented.get("n_total") or 0),
        "by_source": {
            "ingest_probes": ingest_prevented,
            "corpus_negative_control": corpus_prevented,
            "four_way_proof": four_prevented,
        },
        "rule": (
            "sum of this run's admit() REFUSED outcomes and named descendant "
            "invalidations across the ingest probes, the live corpus negative "
            "control, and the isolated four-way proof. Not a historical campaign "
            "total. Derived from the lists, not a fixed bound."
        ),
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Admission gate over the negative-science index, plus four-way "
            "propagation of a fresh failure. A scar now costs a schedule slot."
        ),
        "confidence_reduction_rule": CONFIDENCE_REDUCTION_RULE,
        "confidence_reduction_factor": CONFIDENCE_REDUCTION_FACTOR,
        "confidence_floor": CONFIDENCE_FLOOR,
        "odyssey2_narrow_mapping": [
            {"from": k, "to": O2_NARROW[k]} for k in ols.SCOPES if k in O2_NARROW
        ],
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "known_scars_ingest": ingest,
        "ingest_probes": ingest_probes,
        "corpus_negative_control": corpus_probe,
        "four_way_proof": {
            "scar_stored": (four_way.get("propagation") or {}).get("scar_stored"),
            "scar_keys": (four_way.get("propagation") or {}).get("scar"),
            "invalidated_candidate_ids": (four_way.get("propagation") or {}).get(
                "invalidated_candidate_ids"
            ),
            "confidence_reductions": (four_way.get("propagation") or {}).get(
                "confidence_reductions"
            ),
            "odyssey2_scope_updates": (four_way.get("propagation") or {}).get(
                "odyssey2_scope_updates"
            ),
            "odyssey3_implication": (four_way.get("propagation") or {}).get(
                "odyssey3_implication"
            ),
            "dead_admit": {
                "decision": (four_way.get("dead_admit") or {}).get("decision"),
                "scar_id": (four_way.get("dead_admit") or {}).get("scar_id"),
            },
            "near_miss_admit": {
                "decision": (four_way.get("near_miss_admit") or {}).get("decision"),
            },
            "experiments_prevented": four_prevented,
            "synthetic": True,
        },
        "experiments_prevented": prevented,
        "odyssey2_store": o2_meta,
        "n_overlay_scars": len(sch.extra_scars),
        "workunit": emit_resident_workunit(),
        "resident_callable": resident_callable_doc(),
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "how_to_use": {
            "admit": (
                "from tools.future.scar_scheduling import admit; "
                "outcome = admit({'model': ..., 'organ': ..., "
                "'representation': ..., 'hypothesis_family': ...})"
            ),
            "propagate_failure": (
                "from tools.future.scar_scheduling import propagate_failure; "
                "effects = propagate_failure({'model': ..., 'organ': ..., "
                "'representation': ..., 'machine': ..., 'hypothesis_family': ..., "
                "'mechanism': ..., 'candidate_id': ...})"
            ),
            "refusal_is_not_an_exception": (
                "outcome['decision'] is ADMITTED or REFUSED. REFUSED copies "
                "status=blocked classification=SCAR_REFUSED and failure_context "
                "onto the unit. Callers must not treat a return as success."
            ),
        },
    }
    # Silence unused o2_laws load (identity check that the store parsed).
    doc["odyssey2_store"]["n_laws_seen"] = len(o2_laws)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def _corpus_negative_control() -> dict[str, Any]:
    sch = ScarScheduler(include_corpus=True)
    dead_unit = {
        "id": "future.scar-scheduling.probe.cross-expert-gate",
        "model": "qwen3-235b-a22b",
        "organ": "gate",
        "hypothesis_family": "cross_expert_structure",
    }
    near_unit = {
        "id": "future.scar-scheduling.probe.cross-expert-lmhead-q4",
        "model": "qwen3-235b-a22b",
        "organ": "lm_head",
        "representation": "q4",
        "hypothesis_family": "cross_expert_structure",
    }
    other_family = {
        "id": "future.scar-scheduling.probe.hwir-node-types",
        "model": "qwen3-235b-a22b",
        "organ": "gate",
        "hypothesis_family": "hwir_node_types",
    }
    dead = sch.admit(dead_unit)
    near = sch.admit(near_unit)
    other = sch.admit(other_family)
    return {
        "dead": {
            "decision": dead.get("decision"),
            "scar_id": dead.get("scar_id"),
            "source_path": dead.get("source_path"),
            "hypothesis_family": dead.get("hypothesis_family"),
        },
        "near_miss_organ_rep": {"decision": near.get("decision")},
        "other_family": {"decision": other.get("decision")},
        "experiments_prevented": sch.experiments_prevented(),
        "note": (
            "live corpus probe, not a fixture. cross_expert_structure on "
            "qwen3-235b-a22b/gate must REFUSE; lm_head/q4 and hwir_node_types "
            "must ADMIT."
        ),
    }


def selftest() -> Path:
    probe = _corpus_negative_control()
    if (probe.get("dead") or {}).get("decision") != DECISION_REFUSED:
        raise RuntimeError(
            "selftest: admit() did not REFUSE the known-dead cross_expert_structure "
            f"hypothesis: {probe.get('dead')}"
        )
    if (probe.get("near_miss_organ_rep") or {}).get("decision") != DECISION_ADMITTED:
        raise RuntimeError(
            "selftest: admit() refused a structurally different organ/representation "
            f"near-miss: {probe.get('near_miss_organ_rep')}"
        )
    if (probe.get("other_family") or {}).get("decision") != DECISION_ADMITTED:
        raise RuntimeError(
            "selftest: admit() blanket-refused a different family: "
            f"{probe.get('other_family')}"
        )
    four = run_four_way_proof()
    prop = four["propagation"]
    if not prop.get("scar_stored"):
        raise RuntimeError("selftest: (a) scar was not stored")
    scar = prop.get("scar") or {}
    for field in (
        "model",
        "organ",
        "representation",
        "machine",
        "hypothesis_family",
        "failure_mechanism",
    ):
        if not scar.get(field) or scar.get(field) == ni.UNRECORDED:
            raise RuntimeError(f"selftest: (a) scar missing key {field}: {scar}")
    ids = list(prop.get("invalidated_candidate_ids") or [])
    if FOUR_WAY_CHILD_A not in ids or FOUR_WAY_CHILD_B not in ids:
        raise RuntimeError(f"selftest: (b) dependents not named: {ids}")
    if FOUR_WAY_UNRELATED in ids:
        raise RuntimeError(f"selftest: (b) unrelated candidate invalidated: {ids}")
    reds = list(prop.get("confidence_reductions") or [])
    if not reds:
        raise RuntimeError("selftest: (c) no law confidence reduced")
    if reds[0].get("before") != 0.64 or reds[0].get("after") != 0.32:
        raise RuntimeError(f"selftest: (c) stated rule not applied: {reds[0]}")
    scopes = list(prop.get("odyssey2_scope_updates") or [])
    if not scopes or not scopes[0].get("moved"):
        raise RuntimeError(f"selftest: (d) Odyssey II scope did not move: {scopes}")
    if scopes[0].get("scope_before") != "ARCHITECTURE_FAMILY":
        raise RuntimeError(f"selftest: (d) unexpected O2 before: {scopes[0]}")
    if scopes[0].get("scope_after") != "MODEL_LOCAL":
        raise RuntimeError(f"selftest: (d) unexpected O2 after: {scopes[0]}")
    impl = prop.get("odyssey3_implication") or {}
    if not impl.get("moved"):
        raise RuntimeError(f"selftest: (d) Odyssey III implication did not move: {impl}")
    if four["dead_admit"].get("decision") != DECISION_REFUSED:
        raise RuntimeError("selftest: after failure, dead unit still admitted")
    if four["near_miss_admit"].get("decision") != DECISION_ADMITTED:
        raise RuntimeError("selftest: after failure, near-miss was refused")
    sch = ScarScheduler(include_corpus=True)
    ingest = sch.ingest_known_scars()
    if ingest.get("n_input") and ingest["n_already_present"] + ingest["n_newly_added"] != ingest["n_input"]:
        raise RuntimeError(f"selftest: ingest counts do not partition: {ingest}")
    return build(scheduler=sch, four_way=four, corpus_probe=probe)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--admit", metavar="JSON")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--prevented", action="store_true")
    a = ap.parse_args()
    if a.admit is not None:
        outcome = admit(a.admit)
        print(json.dumps(outcome, indent=1, sort_keys=True, default=str))
        return 0 if outcome.get("decision") == DECISION_ADMITTED else 2
    if a.ingest:
        print(json.dumps(ingest_known_scars(), indent=1, sort_keys=True, default=str))
        return 0
    if a.prevented:
        print(json.dumps(experiments_prevented(), indent=1, sort_keys=True, default=str))
        return 0
    if a.selftest:
        print(selftest())
        return 0
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
