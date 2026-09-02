"""AUTONOMY SCARS — defects in the resident's own scheduling, kept as science.

Negative science covers representation hypotheses. It covered nothing about the
ORCHESTRATOR, and every defect in this campaign's autonomy loop was of that
second kind: the machinery existed and the evidence model around it was wrong.
Those are exactly the failures that survive a rewrite, because nothing records
them and the next scheduler schema reinvents the same mistake.

Each entry is a defect that actually fired, with the symptom that hid it. The
symptom matters more than the fix: all four looked healthy from outside.

This module is also the scoped Law/Scar registry HCLI consults. It extends
the existing scar records (these four, campaign scars, noetic entries, the
Odyssey II law store) rather than replacing negative_index. A scar bound to
one measured scope does not block an out-of-scope retry. A law retrieved
outside the scope it was measured in is refused, never silently generalized.

    python3 tools/future/autonomy_scars.py --build
    python3 -m pytest tools/future -q -o addopts="" -k 'scar or law'
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools.future._common import REPO, write_receipt
from tools.future import negative_index as ni
from tools.future import odyssey2_law_store as ols

RECEIPT = "AUTONOMY_SCARS.json"
SCHEMA = "hawking.future.autonomy_scars.v1"

SCARS: tuple[dict[str, Any], ...] = (
    {
        "id": "STATIC_LANE_TAXONOMY_DIVERGED_FROM_FRONTIER",
        "family": "scheduler_taxonomy",
        "verdict": "BURIED",
        "claim_refuted": (
            "that a scheduler may keep its own list of resource lanes alongside the "
            "frontier's"
        ),
        "what_happened": (
            "autonomy_run declared AVAILABLE_LANES = CPU_ANALYSIS, CPU_VERIFY, "
            "CPU_REPRESENTATION, DISK_IO. The frontier matches required_lanes <= "
            "available against CPU, ANALYSIS, REPRESENTATION, SIMULATION, ODYSSEY, "
            "TOOLING. No item required an invented name, so the subset test was false "
            "for all 31 NEXT_WORK items and next_work() and refill() returned an empty "
            "list on every call, in every run, from the day the driver was written."
        ),
        "why_it_hid": (
            "the loop was never short of work -- it queued capabilities and specimens "
            "directly -- so it looked busy and healthy, and the trial that scored it "
            "had already passed. Nothing reads as broken when a filter silently "
            "matches nothing."
        ),
        "cost": "the frontier's own work never ran once, while the driver documented "
                "itself as deriving work from the frontier",
        "law": "a scheduler derives its lane vocabulary from the authority; it never restates it",
        "reopen_condition": "never; a second lane list is the defect itself",
        "regression_test": "tools/future/test_autonomy_run.py::"
                           "test_the_driver_speaks_the_lane_vocabulary_the_frontier_actually_uses",
    },
    {
        "id": "SCAR_LOOKUP_ON_IMPLEMENTATION_NAMES",
        "family": "negative_science_keying",
        "verdict": "BURIED",
        "claim_refuted": "that a module name can be asked whether it is a dead hypothesis family",
        "what_happened": (
            "the loop consulted refuse_if_dead with hypothesis_family set to a python "
            "module name. The index keys on hypothesis, representation, organ, model, "
            "machine and mechanism, so the question was a category error and the only "
            "possible answer was no. 71 consultations, 0 refusals."
        ),
        "why_it_hid": (
            "consultations were counted and reported, so the loop appeared to be using "
            "negative science. A counter of questions asked is not a measure of "
            "questions answered."
        ),
        "cost": "the resident could not reject anything on evidence, and did not know it",
        "law": "negative science is keyed by hypothesis semantics, never by implementation identity",
        "reopen_condition": "never",
        "regression_test": "tools/future/test_autonomy_run.py::"
                           "test_proposal_space_is_the_fixed_taxonomy_not_the_set_of_dead_ideas",
    },
    {
        "id": "DECLARED_CAPABILITY_READ_AS_EXECUTED_CAPABILITY",
        "family": "self_certification",
        "verdict": "BURIED",
        "claim_refuted": "that naming a tool in source is evidence the tool was driven",
        "what_happened": (
            "the Odyssey gate's resident-schedulability probe accepted any AST Assign "
            "containing a tool's path as proof a module drives it. odyssey_launch names "
            "Doctor's scripts in an `owned = [...]` literal, so the gate certified "
            "ITSELF as Doctor's resident driver and scored schedule and frontier true."
        ),
        "why_it_hid": (
            "it made a criterion look closer to met, which is the direction nobody "
            "audits"
        ),
        "cost": "a launch criterion would have passed on the gate's own declaration",
        "law": "DECLARED CAPABILITY != EXECUTED CAPABILITY. Require invocation, a "
               "resulting receipt, and the link between them.",
        "reopen_condition": "never",
        "regression_test": "tools/future/test_odyssey_launch.py::"
                           "test_negative_control_the_gate_cannot_certify_itself_as_the_driver",
    },
    {
        "id": "STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM",
        "family": "evidence_compression",
        "verdict": "BURIED",
        "claim_refuted": (
            "that a failure status naming a subsystem is evidence that subsystem failed"
        ),
        "what_happened": (
            "flash_meta_teacher_trace writes status BLOCKED_NO_METAL_GPU for ANY "
            "dense_source_bf16_prefix_initialization error, and the claim_boundary text "
            "asserts the host has no Metal-capable GPU. That sentence was then carried "
            "across the campaign as a hardware fact, including into this sidecar's own "
            "autonomy driver. The host is an M3 Ultra; the device enumerates from "
            "Swift, from the exact metal crate the runtime uses, and shaders compile "
            "from source."
        ),
        "why_it_hid": (
            "the status was specific, plausible, and repeated. Specificity reads as "
            "evidence."
        ),
        "cost": (
            "gate 2 of the Flash meta funnel -- and every family behind it -- was "
            "classified as waiting on hardware that was present the whole time"
        ),
        "law": "STATUS LABELS ARE HYPOTHESES UNTIL THEIR CAUSAL CLAIM IS VERIFIED. "
               "A failure receipt records the exact underlying error and the probes "
               "that would separate the candidate causes.",
        "reopen_condition": (
            "the specific process context of the original failure is still "
            "unidentified; that diagnosis is open work, not a closed scar"
        ),
        "regression_test": "tools/future/test_metal_reachability.py::"
                           "test_the_hardcoded_boundary_status_is_recorded_as_a_negative_finding",
    },
)

SISTER_SYMPTOMS: tuple[str, ...] = (
    "\"model missing\" may be a stale hardcoded path -- three Odyssey tools pointed at "
    "a directory that had moved, while the 52GB parent sat on another volume",
    "\"no work\" may be a scheduler taxonomy mismatch",
    "\"Doctor driven\" may be self-certification",
    "\"no GPU\" may be error laundering",
    "\"retired specimen\" may mean historically retired, not scientifically unusable",
)


NOETIC_REL = "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json"
O2_REL = "receipts/future/ODYSSEY2_LAW_STORE.json"
CLAIM_REL = "receipts/future/CLAIM_SCOPE.json"
CAMPAIGN_REL = "receipts/future/CAMPAIGN_SCARS.json"
ATLAS_REL = "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json"
AUTONOMY_RECEIPT_REL = "receipts/future/AUTONOMY_SCARS.json"

LAW_FIELDS = (
    "identity",
    "claim",
    "evidence",
    "scope",
    "confidence",
    "falsifier",
    "transfer_tests",
    "machine_binding",
    "scar_ids",
    "source_path",
)
SCAR_FIELDS = (
    "identity",
    "failed_mechanism",
    "scope",
    "evidence",
    "reason",
    "reopen_if",
    "source_path",
)

# Lattice rank used only to refuse silent widening. MACHINE_LOCAL is an axis,
# not a promotion of MODEL_LOCAL; both sit at 0 so neither is "wider".
_LATTICE_RANK = {
    "MODEL_LOCAL": 0,
    "ORGAN_LOCAL": 0,
    "MACHINE_LOCAL": 0,
    "MODEL_SPECIFIC": 0,
    "ARCHITECTURE_FAMILY": 1,
    "FAMILY": 1,
    "BACKEND_FAMILY": 2,
    "GENERIC_CANDIDATE": 3,
    "GENERIC_VERIFIED": 4,
    "GENERAL_PHYSICAL": 5,
}

_NOT_MEASURED = re.compile(r"\bnot measured on\b\s*(.+)$", re.I | re.S)
_UNBOUND = frozenset({"", "unknown", "unrecorded", "none", "null", "n/a", "unmeasured"})


class OutOfScopeError(ValueError):
    """A law or scar was asked to apply outside the scope it was measured in."""

    def __init__(
        self,
        message: str,
        *,
        identity: str | None = None,
        flag: str = "OUT_OF_SCOPE",
    ) -> None:
        super().__init__(message)
        self.identity = identity
        self.flag = flag


def _load_json(rel: str) -> dict[str, Any]:
    """Disk first, then git show. Sparse absence is not absence."""
    return ols.load_repo_json(rel)


def _split_measured_and_excluded(text: str) -> tuple[str, str]:
    raw = text or ""
    m = _NOT_MEASURED.search(raw)
    if not m:
        return raw, ""
    return raw[: m.start()], m.group(1)


def _canon_models(text: str | Iterable[str] | None) -> tuple[str, ...]:
    if text is None:
        return ()
    chunks: list[str]
    if isinstance(text, str):
        chunks = [text]
    else:
        chunks = [str(x) for x in text]
    found: list[str] = []
    for chunk in chunks:
        for model in ni.extract_models(chunk):
            if model != ni.UNRECORDED and model not in found:
                found.append(model)
    return tuple(found)


def _canon_organs(text: str | Iterable[str] | None) -> tuple[str, ...]:
    if text is None:
        return ()
    if not isinstance(text, str):
        text = " ".join(str(x) for x in text)
    found = tuple(o for o in ni.extract_organs(text) if o != ni.UNRECORDED)
    if found:
        return found
    slug = ni._slug(text) if text else ""
    if slug and slug.lower() not in _UNBOUND:
        return (slug,)
    return ()


def _bound(text: str | None) -> str:
    s = (text or "").strip()
    if s.lower() in _UNBOUND:
        return ""
    return s


def _canon_machine(text: str | None) -> str:
    if not _bound(text):
        return ""
    slug = ni.canon_machine(str(text))
    if not slug or slug == ni.UNRECORDED or slug.lower() in _UNBOUND:
        return ""
    return slug


def _canon_representation(text: str | None) -> str:
    """Bind a real representation slug. Prose codec descriptions stay unbound."""
    if not _bound(text):
        return ""
    slug = ni.canon_representation(str(text))
    if not slug or slug == ni.UNRECORDED or slug.lower() in _UNBOUND:
        return ""
    if len(slug) > 32:
        return ""
    return slug


def _canon_family(text: str | None) -> str:
    if not (text or "").strip():
        return ""
    slug = ni.canon_family(str(text))
    return "" if slug == ni.UNRECORDED else slug


def _model_in(needle: str, haystack: tuple[str, ...]) -> bool:
    want = ni.canon_model(needle) if needle else ""
    if not want or want == ni.UNRECORDED:
        return False
    for item in haystack:
        if ni.canon_model(item) == want:
            return True
    return False


def _organ_in(needle: str, haystack: tuple[str, ...]) -> bool:
    want = ni.canon_organ(needle) if needle else ""
    if want and want != ni.UNRECORDED:
        canons = {ni.canon_organ(o) if o != ni.UNRECORDED else o for o in haystack}
        if want in haystack or want in canons:
            return True
    slug = ni._slug(needle)
    return bool(slug) and (slug in haystack or slug in {ni._slug(o) for o in haystack})


def _lattice_is_wider(candidate: str, measured: str) -> bool:
    if not candidate or not measured:
        return False
    c = _LATTICE_RANK.get(candidate)
    m = _LATTICE_RANK.get(measured)
    if c is None or m is None:
        return False
    return c > m


@dataclass(frozen=True)
class Scope:
    """Axes a measurement was taken in. Empty means unbound, not wildcard."""

    models: tuple[str, ...] = ()
    organs: tuple[str, ...] = ()
    machine: str = ""
    representation: str = ""
    architecture_family: str = ""
    backend: str = ""
    condition: str = ""
    lattice: str = ""
    excluded_models: tuple[str, ...] = ()
    model_tier: str = ""
    organ_tier: str = ""
    machine_tier: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": list(self.models),
            "organs": list(self.organs),
            "machine": self.machine,
            "representation": self.representation,
            "architecture_family": self.architecture_family,
            "backend": self.backend,
            "condition": self.condition,
            "lattice": self.lattice,
            "excluded_models": list(self.excluded_models),
            "model_tier": self.model_tier,
            "organ_tier": self.organ_tier,
            "machine_tier": self.machine_tier,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "Scope":
        if not raw:
            return cls()
        model_text = raw.get("models")
        if model_text is None:
            model_text = raw.get("model") or ""
        organ_text = raw.get("organs")
        if organ_text is None:
            organ_text = raw.get("organ") or ""
        measured_models, excluded_text = _split_measured_and_excluded(
            model_text if isinstance(model_text, str) else " ".join(str(x) for x in (model_text or ()))
        )
        excluded = tuple(
            dict.fromkeys(
                _canon_models(excluded_text) + _canon_models(raw.get("excluded_models"))
            )
        )
        models = _canon_models(measured_models)
        # Drop models that only appeared in a "NOT measured on" clause.
        if excluded_text:
            models = tuple(m for m in models if m not in excluded)
        lattice_raw = raw.get("lattice") or raw.get("level") or ""
        lattice = lattice_raw if isinstance(lattice_raw, str) else ""
        if lattice in {"model", "MODEL_LOCAL"}:
            lattice = "MODEL_LOCAL"
        if lattice not in _LATTICE_RANK:
            lattice = lattice if lattice else ""
        return cls(
            models=models,
            organs=_canon_organs(organ_text),
            machine=_canon_machine(str(raw.get("machine") or "")),
            representation=_canon_representation(
                str(raw.get("representation") or raw.get("codec") or "")
            ),
            architecture_family=_bound(str(raw.get("architecture_family") or "")),
            backend=_bound(str(raw.get("backend") or "")),
            condition=str(raw.get("condition") or raw.get("regime") or ""),
            lattice=lattice,
            excluded_models=excluded,
            model_tier=str(raw.get("model_tier") or ""),
            organ_tier=str(raw.get("organ_tier") or ""),
            machine_tier=str(raw.get("machine_tier") or ""),
        )


def scope_covers(measured: Scope, candidate: Scope) -> bool:
    """True iff `candidate` sits inside the measured scope.

    A bound axis on the scar/law must match. A candidate that omits a bound
    axis is out of scope: one local failure must not become a global ban.
    Unbound measured axes do not constrain. Explicit excluded_models refuse.
    """
    cand_models = candidate.models
    if measured.excluded_models and cand_models:
        if any(_model_in(m, measured.excluded_models) for m in cand_models):
            return False
    if measured.models:
        if not cand_models:
            return False
        if not any(_model_in(m, measured.models) for m in cand_models):
            return False
    if measured.organs:
        if not candidate.organs:
            return False
        if not any(_organ_in(o, measured.organs) for o in candidate.organs):
            return False
    if _bound(measured.machine):
        if not _bound(candidate.machine):
            return False
        if _canon_machine(candidate.machine) != _canon_machine(measured.machine):
            return False
    if _bound(measured.representation):
        if not _bound(candidate.representation):
            return False
        if _canon_representation(candidate.representation) != _canon_representation(
            measured.representation
        ):
            return False
    if _bound(measured.architecture_family) and _bound(candidate.architecture_family):
        if candidate.architecture_family != measured.architecture_family:
            return False
    if _bound(measured.backend) and _bound(candidate.backend):
        if candidate.backend.lower() != measured.backend.lower():
            return False
    if measured.condition and candidate.condition:
        if measured.condition.lower() not in candidate.condition.lower() and (
            candidate.condition.lower() not in measured.condition.lower()
        ):
            return False
    if _lattice_is_wider(candidate.lattice, measured.lattice):
        return False
    return True


def scope_from_candidate(candidate: Mapping[str, Any] | None) -> Scope:
    if not candidate:
        return Scope()
    nested = candidate.get("scope")
    if isinstance(nested, Scope):
        return nested
    if isinstance(nested, Mapping) and any(
        k in nested for k in ("models", "model", "organs", "organ", "machine", "lattice")
    ):
        base = Scope.from_dict(nested)
        # Overlay explicit candidate axes when present.
        model = candidate.get("model") or candidate.get("source_model")
        organ = candidate.get("organ") or candidate.get("organ_class")
        machine = candidate.get("machine") or candidate.get("source_device")
        lattice = candidate.get("lattice")
        return Scope(
            models=_canon_models(str(model)) if model else base.models,
            organs=_canon_organs(str(organ)) if organ else base.organs,
            machine=_canon_machine(str(machine)) if machine else base.machine,
            representation=_canon_representation(
                str(candidate.get("representation") or candidate.get("codec") or base.representation)
            )
            or base.representation,
            architecture_family=str(
                candidate.get("architecture_family") or base.architecture_family
            ),
            backend=str(candidate.get("backend") or base.backend),
            condition=str(candidate.get("condition") or base.condition),
            lattice=str(lattice or base.lattice),
            excluded_models=base.excluded_models,
            model_tier=base.model_tier,
            organ_tier=base.organ_tier,
            machine_tier=base.machine_tier,
        )
    model = candidate.get("model") or candidate.get("source_model") or ""
    organ = candidate.get("organ") or candidate.get("organ_class") or ""
    machine = candidate.get("machine") or candidate.get("source_device") or ""
    lattice = candidate.get("lattice") or ""
    if isinstance(lattice, dict):
        lattice = str(lattice.get("lattice") or "")
    return Scope(
        models=_canon_models(str(model)) if model else (),
        organs=_canon_organs(str(organ)) if organ else (),
        machine=_canon_machine(str(machine)),
        representation=_canon_representation(
            str(candidate.get("representation") or candidate.get("codec") or "")
        ),
        architecture_family=str(candidate.get("architecture_family") or ""),
        backend=str(candidate.get("backend") or ""),
        condition=str(candidate.get("condition") or ""),
        lattice=str(lattice or candidate.get("level") or ""),
    )


@dataclass(frozen=True)
class Law:
    identity: str
    claim: str
    evidence: tuple[str, ...]
    scope: Scope
    confidence: dict[str, Any]
    falsifier: str
    transfer_tests: tuple[str, ...]
    machine_binding: str
    scar_ids: tuple[str, ...]
    source_path: str
    experiment_identity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "claim": self.claim,
            "evidence": list(self.evidence),
            "scope": self.scope.to_dict(),
            "confidence": dict(self.confidence),
            "falsifier": self.falsifier,
            "transfer_tests": list(self.transfer_tests),
            "machine_binding": self.machine_binding,
            "scar_ids": list(self.scar_ids),
            "source_path": self.source_path,
            "experiment_identity": dict(self.experiment_identity),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Law":
        evidence = raw.get("evidence") or ()
        tests = raw.get("transfer_tests") or ()
        scars = raw.get("scar_ids") or ()
        conf = raw.get("confidence") or {}
        return cls(
            identity=str(raw.get("identity") or raw.get("law_id") or ""),
            claim=str(raw.get("claim") or raw.get("statement") or ""),
            evidence=tuple(str(x) for x in evidence),
            scope=raw["scope"] if isinstance(raw.get("scope"), Scope) else Scope.from_dict(raw.get("scope") if isinstance(raw.get("scope"), Mapping) else {}),
            confidence=dict(conf) if isinstance(conf, Mapping) else {"value": conf, "basis": "UNRECORDED"},
            falsifier=str(raw.get("falsifier") or ""),
            transfer_tests=tuple(str(x) for x in tests),
            machine_binding=str(raw.get("machine_binding") or ""),
            scar_ids=tuple(str(x) for x in scars),
            source_path=str(raw.get("source_path") or ""),
            experiment_identity=dict(raw.get("experiment_identity") or {}),
        )


@dataclass(frozen=True)
class Scar:
    identity: str
    failed_mechanism: str
    scope: Scope
    evidence: tuple[str, ...]
    reason: str
    reopen_if: str
    source_path: str
    hypothesis_family: str = ""
    verdict: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "scar_id": self.identity,
            "failed_mechanism": self.failed_mechanism,
            "hypothesis_family": self.hypothesis_family or self.failed_mechanism,
            "scope": self.scope.to_dict(),
            "evidence": list(self.evidence),
            "reason": self.reason,
            "claim_refuted": self.reason,
            "reopen_if": self.reopen_if,
            "reopen_condition": self.reopen_if,
            "source_path": self.source_path,
            "verdict": self.verdict,
            "parse_status": ni.PARSED,
            "refuse_eligible": True,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Scar":
        evidence = raw.get("evidence") or ()
        scope_raw = raw.get("scope")
        if isinstance(scope_raw, Scope):
            scope = scope_raw
        elif isinstance(scope_raw, Mapping):
            scope = Scope.from_dict(scope_raw)
        else:
            scope = Scope.from_dict(raw)
        return cls(
            identity=str(raw.get("identity") or raw.get("scar_id") or raw.get("id") or ""),
            failed_mechanism=str(
                raw.get("failed_mechanism") or raw.get("hypothesis_family") or raw.get("family") or ""
            ),
            scope=scope,
            evidence=tuple(str(x) for x in evidence),
            reason=str(raw.get("reason") or raw.get("claim_refuted") or ""),
            reopen_if=str(raw.get("reopen_if") or raw.get("reopen_condition") or ""),
            source_path=str(raw.get("source_path") or ""),
            hypothesis_family=str(
                raw.get("hypothesis_family") or raw.get("family") or raw.get("failed_mechanism") or ""
            ),
            verdict=str(raw.get("verdict") or ""),
        )


def _proposal_family(candidate: Mapping[str, Any]) -> str:
    for key in (
        "hypothesis_family",
        "technique",
        "mechanism",
        "failed_mechanism",
        "lever",
        "seed",
        "family",
    ):
        value = candidate.get(key)
        if value:
            return str(value)
    return ""


def _mechanism_matches(scar: Scar, candidate: Mapping[str, Any]) -> bool:
    family = _canon_family(_proposal_family(candidate))
    if not family:
        return False
    scar_fam = _canon_family(scar.hypothesis_family or scar.failed_mechanism)
    if not scar_fam:
        return False
    return family == scar_fam


def scar_blocks_candidate(
    scar: Scar | Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Cite `scar` only when the candidate is inside the scar's measured scope.

    Returns a refusal dict, or None. Out of scope is None, never a global ban.
    """
    record = scar if isinstance(scar, Scar) else Scar.from_dict(scar)
    if not _mechanism_matches(record, candidate):
        return None
    if not scope_covers(record.scope, scope_from_candidate(candidate)):
        return None
    return {
        "refused": True,
        "reason": (
            "known-dead hypothesis inside its measured scope; "
            "reopen only under reopen_if"
        ),
        "scar_id": record.identity,
        "failed_mechanism": record.failed_mechanism,
        "hypothesis_family": record.hypothesis_family or _canon_family(record.failed_mechanism),
        "scope": record.scope.to_dict(),
        "evidence": list(record.evidence),
        "reopen_if": record.reopen_if,
        "source_path": record.source_path,
        "verdict": record.verdict,
    }


def retrieve_law(
    identity: str,
    query: Mapping[str, Any],
    *,
    laws: Iterable[Law] | None = None,
    require_in_scope: bool = False,
) -> dict[str, Any]:
    """Fetch a law bound to the scope it was measured in.

    Out of scope is refused and flagged (usable=False). Never returned as if
    it applied. Set require_in_scope to raise OutOfScopeError instead.
    """
    pool = list(laws) if laws is not None else load_registry().laws
    found = next((law for law in pool if law.identity == identity), None)
    if found is None:
        result = {
            "found": False,
            "usable": False,
            "refused": True,
            "flag": "NOT_FOUND",
            "identity": identity,
            "law": None,
        }
        if require_in_scope:
            raise OutOfScopeError(f"law {identity!r} not in registry", identity=identity, flag="NOT_FOUND")
        return result
    query_scope = scope_from_candidate(query)
    if not scope_covers(found.scope, query_scope):
        result = {
            "found": True,
            "usable": False,
            "refused": True,
            "flag": "OUT_OF_SCOPE",
            "identity": found.identity,
            "reason": (
                f"law {found.identity} was measured in scope "
                f"{found.scope.to_dict()}; query {query_scope.to_dict()} "
                f"is outside that scope and must not be silently generalized"
            ),
            "law": found.to_dict(),
            "source_path": found.source_path,
        }
        if require_in_scope:
            raise OutOfScopeError(result["reason"], identity=found.identity)
        return result
    return {
        "found": True,
        "usable": True,
        "refused": False,
        "flag": None,
        "identity": found.identity,
        "law": found.to_dict(),
        "source_path": found.source_path,
    }


def scar_from_noetic_entry(entry: Mapping[str, Any], source_path: str = NOETIC_REL) -> Scar:
    scope_raw = entry.get("scope") if isinstance(entry.get("scope"), Mapping) else {}
    measured, excluded_text = _split_measured_and_excluded(str(scope_raw.get("model") or ""))
    models = _canon_models(measured)
    excluded = _canon_models(excluded_text)
    if excluded:
        models = tuple(m for m in models if m not in excluded)
    evidence_rows = entry.get("evidence") or []
    evidence: list[str] = []
    if isinstance(evidence_rows, list):
        for row in evidence_rows:
            if isinstance(row, Mapping) and row.get("path"):
                field = row.get("field") or ""
                evidence.append(f"{row['path']}#{field}" if field else str(row["path"]))
            elif isinstance(row, str):
                evidence.append(row)
    seed = str(entry.get("seed") or entry.get("id") or "")
    return Scar(
        identity=str(entry.get("id") or seed),
        failed_mechanism=seed or str(entry.get("claim_refuted") or ""),
        scope=Scope(
            models=models,
            organs=_canon_organs(str(scope_raw.get("organ") or "")),
            representation=_canon_representation(str(scope_raw.get("codec") or "")),
            condition=str(scope_raw.get("regime") or ""),
            lattice="MODEL_SPECIFIC",
            excluded_models=excluded,
        ),
        evidence=tuple(evidence),
        reason=str(entry.get("claim_refuted") or entry.get("kind_reasoning") or ""),
        reopen_if=str(entry.get("reopen_condition") or entry.get("reopen_if") or ""),
        source_path=source_path,
        hypothesis_family=_canon_family(seed) or seed,
        verdict=str(entry.get("kind") or entry.get("verdict") or ""),
    )


def law_from_odyssey2_record(raw: Mapping[str, Any], source_path: str = O2_REL) -> Law:
    refs = tuple(str(x) for x in (raw.get("evidence_refs") or ()))
    transfers = raw.get("transfer_candidates") or ()
    tests: list[str] = []
    if isinstance(transfers, list):
        for row in transfers:
            if isinstance(row, Mapping):
                target = row.get("target_school") or row.get("target_model") or ""
                basis = row.get("confidence_basis") or row.get("counterexample_requirement") or ""
                tests.append(f"{target}: {basis}".strip(": "))
            else:
                tests.append(str(row))
    falsifier = str(raw.get("counterexample_requirement") or raw.get("falsifier") or "")
    if falsifier and falsifier not in tests:
        tests.append(falsifier)
    conf = raw.get("transfer_confidence") if isinstance(raw.get("transfer_confidence"), Mapping) else {}
    lattice = str(raw.get("scope") or "MODEL_LOCAL")
    return Law(
        identity=str(raw.get("law_id") or raw.get("identity") or ""),
        claim=str(raw.get("statement") or raw.get("claim") or ""),
        evidence=refs,
        scope=Scope(
            models=_canon_models(str(raw.get("source_model") or "")),
            organs=_canon_organs(str(raw.get("organ_class") or "")),
            machine=_canon_machine(str(raw.get("source_device") or "")),
            architecture_family=_bound(str(raw.get("architecture_family") or "")),
            backend=_bound(str(raw.get("backend") or "")),
            lattice=lattice if lattice in _LATTICE_RANK else "MODEL_LOCAL",
        ),
        confidence=dict(conf) if conf else {"value": "UNRECORDED", "basis": "UNRECORDED on source receipt"},
        falsifier=falsifier,
        transfer_tests=tuple(tests),
        machine_binding=str(raw.get("source_device") or "UNRECORDED"),
        scar_ids=tuple(str(x) for x in (raw.get("scar_ids") or ())),
        source_path=source_path,
    )


def law_from_claim_scope_record(raw: Mapping[str, Any], source_path: str = CLAIM_REL) -> Law:
    scope_raw = raw.get("scope") if isinstance(raw.get("scope"), Mapping) else {}
    refs = tuple(str(x) for x in (raw.get("evidence_refs") or ()))
    identity_block = raw.get("experiment_identity") if isinstance(raw.get("experiment_identity"), Mapping) else {}
    tests = [
        str(row.get("why") or row.get("specimen") or row)
        for row in (raw.get("failed_transfers") or [])
        if row
    ]
    if raw.get("narrowing"):
        tests.append(str(raw["narrowing"]))
    scar_ids: list[str] = []
    blob = " ".join([str(raw.get("statement") or ""), str(raw.get("narrowing") or "")])
    if "FUNCTION-REPLACEMENT" in str(raw.get("law_id") or "") or "function replacement" in blob.lower():
        scar_ids.append("MLP_FUNCTION_REPLACEMENT_CLOSED")
    return Law(
        identity=str(raw.get("law_id") or raw.get("identity") or ""),
        claim=str(raw.get("statement") or raw.get("claim") or ""),
        evidence=refs,
        scope=Scope(
            models=_canon_models(str(raw.get("parent") or " ".join(raw.get("tested_specimens") or ()))),
            organs=_canon_organs(str(raw.get("organ") or "")),
            machine=_canon_machine(str(raw.get("machine") or "")),
            lattice="MODEL_LOCAL",
            condition=str(raw.get("scope_kind") or ""),
            model_tier=str(scope_raw.get("model") or ""),
            organ_tier=str(scope_raw.get("organ") or ""),
            machine_tier=str(scope_raw.get("machine") or ""),
        ),
        confidence={
            "value": "UNRECORDED",
            "basis": f"claim_scope.scope_kind={raw.get('scope_kind')!r}; cited, not remeasured",
        },
        falsifier=str(raw.get("narrowing") or raw.get("falsifier") or ""),
        transfer_tests=tuple(tests),
        machine_binding=str(raw.get("machine") or "UNRECORDED"),
        scar_ids=tuple(scar_ids),
        source_path=source_path,
        experiment_identity=dict(identity_block),
    )


def scar_from_autonomy_dict(raw: Mapping[str, Any], source_path: str = AUTONOMY_RECEIPT_REL) -> Scar:
    evidence = raw.get("evidence") or [raw.get("regression_test")]
    evidence_t = tuple(str(x) for x in evidence if x)
    scope_raw = raw.get("scope") if isinstance(raw.get("scope"), Mapping) else {
        "lattice": "GENERAL_PHYSICAL",
        "condition": "orchestrator",
    }
    return Scar(
        identity=str(raw.get("id") or raw.get("identity") or ""),
        failed_mechanism=str(raw.get("failed_mechanism") or raw.get("family") or ""),
        scope=Scope.from_dict(scope_raw) if scope_raw.get("models") or scope_raw.get("lattice") else Scope(
            lattice="GENERAL_PHYSICAL",
            condition=str(scope_raw.get("condition") or "orchestrator"),
        ),
        evidence=evidence_t,
        reason=str(raw.get("reason") or raw.get("why_it_hid") or raw.get("claim_refuted") or ""),
        reopen_if=str(raw.get("reopen_if") or raw.get("reopen_condition") or ""),
        source_path=source_path,
        hypothesis_family=_canon_family(str(raw.get("family") or raw.get("failed_mechanism") or "")),
        verdict=str(raw.get("verdict") or ""),
    )


def scar_from_campaign_dict(raw: Mapping[str, Any], source_path: str = CAMPAIGN_REL) -> Scar:
    evidence = [str(x) for x in (raw.get("source_receipts") or [])]
    if raw.get("regression_test"):
        evidence.append(str(raw["regression_test"]))
    return Scar(
        identity=str(raw.get("id") or ""),
        failed_mechanism=str(raw.get("generalized_class") or raw.get("hypothesis_family") or ""),
        scope=Scope(lattice="GENERAL_PHYSICAL", condition="campaign_process"),
        evidence=tuple(evidence),
        reason=str(raw.get("claim_refuted") or raw.get("wrongly_concluded") or ""),
        reopen_if=str(raw.get("reopen_condition") or raw.get("reopen_if") or ""),
        source_path=source_path,
        hypothesis_family=_canon_family(str(raw.get("hypothesis_family") or raw.get("id") or "")),
        verdict=str(raw.get("verdict") or ""),
    )


def scar_from_atlas_entry(key: str, entry: Mapping[str, Any], source_path: str = ATLAS_REL) -> Scar:
    parent = str(entry.get("parent") or "")
    evidence: list[str] = []
    ev = entry.get("evidence")
    if isinstance(ev, list):
        evidence.extend(str(x) for x in ev)
    elif isinstance(ev, Mapping):
        evidence.append(json.dumps(ev, sort_keys=True)[:240])
    elif ev:
        evidence.append(str(ev))
    return Scar(
        identity=str(key),
        failed_mechanism=str(entry.get("lever") or key),
        scope=Scope(
            models=_canon_models(parent),
            lattice="MODEL_SPECIFIC",
        ),
        evidence=tuple(evidence) or (source_path,),
        reason=str(entry.get("killed_by") or entry.get("verdict") or ""),
        reopen_if=str(entry.get("reopen_condition") or entry.get("reopen_if") or ""),
        source_path=source_path,
        hypothesis_family=_canon_family(str(key)),
        verdict=str(entry.get("verdict") or ""),
    )


def law_from_autonomy_scar(raw: Mapping[str, Any]) -> Law:
    sid = str(raw.get("id") or "")
    return Law(
        identity=f"LAW-AUTONOMY-{sid}",
        claim=str(raw.get("law") or ""),
        evidence=(str(raw.get("regression_test") or ""),),
        scope=Scope(lattice="GENERAL_PHYSICAL", condition="orchestrator"),
        confidence={"value": "UNRECORDED", "basis": "qualitative orchestrator law; not a numeric transfer"},
        falsifier=str(raw.get("reopen_condition") or raw.get("reopen_if") or ""),
        transfer_tests=(),
        machine_binding="UNRECORDED",
        scar_ids=(sid,) if sid else (),
        source_path=AUTONOMY_RECEIPT_REL,
    )


@dataclass
class Registry:
    laws: list[Law]
    scars: list[Scar]
    sources: tuple[str, ...]


_REGISTRY: Registry | None = None


def load_registry(*, force: bool = False) -> Registry:
    """Load laws and scars from named real receipts. Not fixtures."""
    global _REGISTRY
    if _REGISTRY is not None and not force:
        return _REGISTRY
    laws: list[Law] = []
    scars: list[Scar] = []
    sources: list[str] = []

    for raw in SCARS:
        scars.append(scar_from_autonomy_dict(raw, AUTONOMY_RECEIPT_REL))
        if raw.get("law"):
            laws.append(law_from_autonomy_scar(raw))
    sources.append(AUTONOMY_RECEIPT_REL)

    try:
        campaign = _load_json(CAMPAIGN_REL)
        for row in campaign.get("scars") or []:
            if isinstance(row, Mapping):
                scars.append(scar_from_campaign_dict(row, CAMPAIGN_REL))
        sources.append(CAMPAIGN_REL)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass

    try:
        noetic = _load_json(NOETIC_REL)
        for row in noetic.get("entries") or []:
            if isinstance(row, Mapping):
                scars.append(scar_from_noetic_entry(row, NOETIC_REL))
        sources.append(NOETIC_REL)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass

    try:
        atlas = _load_json(ATLAS_REL)
        entries = atlas.get("entries") or {}
        if isinstance(entries, Mapping):
            for key, row in entries.items():
                if isinstance(row, Mapping):
                    scars.append(scar_from_atlas_entry(str(key), row, ATLAS_REL))
        sources.append(ATLAS_REL)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass

    try:
        o2 = _load_json(O2_REL)
        for row in o2.get("laws") or []:
            if isinstance(row, Mapping):
                laws.append(law_from_odyssey2_record(row, O2_REL))
        sources.append(O2_REL)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass

    try:
        claims = _load_json(CLAIM_REL)
        for row in claims.get("laws") or []:
            if isinstance(row, Mapping):
                laws.append(law_from_claim_scope_record(row, CLAIM_REL))
        sources.append(CLAIM_REL)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass

    _REGISTRY = Registry(laws=laws, scars=scars, sources=tuple(sources))
    return _REGISTRY


def consult(
    candidate: Mapping[str, Any],
    *,
    scars: Iterable[Scar] | None = None,
    laws: Iterable[Law] | None = None,
) -> dict[str, Any]:
    """HCLI entry: scoped scar block and in-scope law retrieval.

    A matching-family scar outside the candidate's scope is recorded as
    out_of_scope_not_blocked, never as a refusal.
    """
    registry = None
    if scars is None or laws is None:
        registry = load_registry()
    pool_scars = list(scars) if scars is not None else registry.scars
    pool_laws = list(laws) if laws is not None else registry.laws
    blocked: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for scar in pool_scars:
        if not _mechanism_matches(scar, candidate):
            continue
        hit = scar_blocks_candidate(scar, candidate)
        if hit is not None:
            blocked.append(hit)
        else:
            skipped.append(
                {
                    "scar_id": scar.identity,
                    "source_path": scar.source_path,
                    "flag": "OUT_OF_SCOPE",
                    "scope": scar.scope.to_dict(),
                    "reason": "scar matches family but not measured scope; retry is not banned",
                }
            )
    usable_laws: list[dict[str, Any]] = []
    flagged_laws: list[dict[str, Any]] = []
    for law in pool_laws:
        result = retrieve_law(law.identity, candidate, laws=pool_laws)
        if result.get("usable"):
            usable_laws.append(result)
        elif result.get("flag") == "OUT_OF_SCOPE":
            flagged_laws.append(result)
    return {
        "blocked": bool(blocked),
        "blocked_by": blocked,
        "out_of_scope_not_blocked": skipped,
        "laws": usable_laws,
        "laws_out_of_scope": flagged_laws,
        "entry_point": "tools.future.autonomy_scars.consult",
        "evidence_class": "STATIC",
    }


def round_trip_scar(scar: Scar) -> Scar:
    return Scar.from_dict(scar.to_dict())


def round_trip_law(law: Law) -> Law:
    return Law.from_dict(law.to_dict())


def scars() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scar in SCARS:
        row = dict(scar)
        row.setdefault("failed_mechanism", row.get("family"))
        row.setdefault("reopen_if", row.get("reopen_condition"))
        row.setdefault("reason", row.get("why_it_hid") or row.get("claim_refuted"))
        row.setdefault("evidence", [row["regression_test"]] if row.get("regression_test") else [])
        row.setdefault(
            "scope",
            {"lattice": "GENERAL_PHYSICAL", "condition": "orchestrator"},
        )
        out.append(row)
    return out


def missing_regression_tests() -> list[dict[str, str]]:
    """A scar whose regression test does not exist is a story, not a guard."""
    out = []
    for scar in SCARS:
        rel, _, name = str(scar["regression_test"]).partition("::")
        path = REPO / rel
        present = path.is_file() and (not name or f"def {name}(" in path.read_text(errors="replace"))
        if not present:
            out.append({"id": scar["id"], "regression_test": scar["regression_test"]})
    return out


def build() -> Path:
    missing = missing_regression_tests()
    registry = load_registry()
    in_scope = consult(
        {
            "model": "qwen3-80b",
            "organ": "gate",
            "hypothesis_family": "cross_expert_structure",
        },
        scars=registry.scars,
        laws=registry.laws,
    )
    out_of_scope = consult(
        {
            "model": "deepseek-v4-flash",
            "organ": "gate",
            "hypothesis_family": "cross_expert_structure",
        },
        scars=registry.scars,
        laws=registry.laws,
    )
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Defects in the resident's own scheduling and evidence model, kept as "
            "negative science so the next scheduler schema does not reinvent them."
        ),
        "evidence_class": "STATIC_ONLY",
        "why_this_exists": (
            "negative science covered representation hypotheses and nothing about the "
            "orchestrator. Every autonomy defect this campaign found was of the second "
            "kind: the machinery existed and the evidence model around it was wrong."
        ),
        "n_scars": len(SCARS),
        "scars": scars(),
        "general_law": (
            "STATUS LABELS ARE HYPOTHESES UNTIL THEIR CAUSAL CLAIM IS VERIFIED."
        ),
        "sister_symptoms": list(SISTER_SYMPTOMS),
        "scars_without_a_regression_test": missing,
        "recovered_implementation": [
            "tools/future/negative_index.py keys scars by hypothesis semantics; these "
            "are orchestration defects and do not belong in that keyspace",
            "tools/future/odyssey2_law_store.py stores laws with a scope lattice; "
            "this registry retrieves them without silently widening",
            "tools/future/scar_reevaluator.py classifies reopenability; consult_candidate "
            "now asks autonomy_scars.consult before treating a scar as a ban",
        ],
        "gaps_closed": [
            "autonomy defects were fixed but never recorded as science",
            "a scoped scar can no longer ban an out-of-scope retry",
        ],
        "negative_findings": [
            "all four defects looked healthy from outside; none produced an error",
        ],
        "registry": {
            "n_laws": len(registry.laws),
            "n_scoped_scars": len(registry.scars),
            "sources": list(registry.sources),
            "consult": "tools.future.autonomy_scars.consult",
            "retrieve_law": "tools.future.autonomy_scars.retrieve_law",
            "over_generalization_guard": (
                "a scar scoped to one condition does not block an out-of-scope retry"
            ),
            "in_scope_cross_expert_qwen3_80b_blocked": in_scope["blocked"],
            "out_of_scope_cross_expert_dsv4f_blocked": out_of_scope["blocked"],
            "evidence_tier": "STATIC",
        },
        "resident_callable": {
            "entry_point": "tools.future.autonomy_scars.scars()",
            "consult": "tools.future.autonomy_scars.consult()",
            "workunit": "one CPU_ANALYSIS unit; consulted before a scheduler change",
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.HCLI_SELF.emit-workunits",
            "fails_closed": "a scar whose regression test is absent is reported, not hidden",
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/autonomy_scars.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    doc = json.loads(out.read_text())
    print(out)
    print(json.dumps({"n_scars": doc["n_scars"],
                      "without_regression_test": doc["scars_without_a_regression_test"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
