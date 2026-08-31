"""Freeze Codex's optimization metabolism into schedulable HCLI WorkUnit species.

Codex handed over nine loose labels. This module turns the handoff into thirty
precise WorkUnit species, each grounded in an observed Codex operation (or
explicitly UNGROUNDED_FROM_HANDOFF), and encodes the profile → rank → generate
→ verify → AB → ledger → law → attack cycle that refills on a win.

Extends tools.future.workunit_species. Does not fork a second scheduler, does
not acquire a GPU lease, and does not promote.

    python3 tools/future/codex_behaviors.py --build
    python3 tools/future/codex_behaviors.py --selftest
    python3 -m pytest tools/future/test_codex_behaviors.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO


import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hcli.workunit import WorkUnit
from tools.future import workunit_species as ws
from tools.future._common import git

RECEIPT = "CODEX_WORKUNIT_SPECIES.json"
SCHEMA = "hawking.future.codex_behaviors.v1"
VERSION = 1
RECORDED_BY = "tools/future/codex_behaviors.py"
HANDOFF_REL = "CODEX_ACCELERATOR_HANDOFF.json"

# Closed vocabulary from the steer. Not a queue bound — a named species list.
SPECIES_IDS: tuple[str, ...] = (
    "PROFILE_COMPLETE_TOKEN",
    "PROFILE_REGION",
    "PROFILE_GPU",
    "PROFILE_HOST_CEREMONY",
    "PROFILE_ACTIVE_BYTES",
    "PROFILE_RESIDENT_BYTES",
    "PROFILE_DISPATCH",
    "PROFILE_SYNC",
    "FIND_TALLEST_COST",
    "SEARCH_ARCHITECTURE_LAWS",
    "GENERATE_KERNEL_CANDIDATE",
    "GENERATE_FUSION_CANDIDATE",
    "GENERATE_LAYOUT_CANDIDATE",
    "GENERATE_STATE_RESIDENCY_CANDIDATE",
    "GENERATE_PIPELINE_PERSISTENCE_CANDIDATE",
    "GENERATE_ROUTE_CANDIDATE",
    "GENERATE_REPRESENTATION_NATIVE_KERNEL",
    "STATIC_KERNEL_VERIFY",
    "HOST_SHADER_ABI_VERIFY",
    "STRUCTURAL_COST_COMPARE",
    "DIAGNOSTIC_AB",
    "PROTECTED_AB",
    "FACTORIAL_COMBINATION",
    "REPROFILE_AFTER_WIN",
    "UPDATE_SCOREBOARD",
    "UPDATE_REPATRIATION_LEDGER",
    "UPDATE_LAW",
    "UPDATE_SCAR",
    "TRANSFER_LAW",
    "ATTACK_LAW",
)

LOOSE_HANDOFF_SPECIES: tuple[str, ...] = (
    "GPU kernel optimization",
    "fusion search",
    "layout search",
    "dispatch/sync elimination",
    "pipeline persistence",
    "active-byte reduction",
    "representation-native kernels",
    "source ceremony elimination",
    "backend-placement experiments",
)

# Codex's observed loop, as a schedulable order. Parallel waves share a rank.
CYCLE_WAVES: tuple[tuple[str, ...], ...] = (
    (
        "PROFILE_COMPLETE_TOKEN",
        "PROFILE_REGION",
        "PROFILE_GPU",
        "PROFILE_HOST_CEREMONY",
        "PROFILE_ACTIVE_BYTES",
        "PROFILE_RESIDENT_BYTES",
        "PROFILE_DISPATCH",
        "PROFILE_SYNC",
    ),
    ("FIND_TALLEST_COST",),
    ("SEARCH_ARCHITECTURE_LAWS",),
    (
        "GENERATE_KERNEL_CANDIDATE",
        "GENERATE_FUSION_CANDIDATE",
        "GENERATE_LAYOUT_CANDIDATE",
        "GENERATE_STATE_RESIDENCY_CANDIDATE",
        "GENERATE_PIPELINE_PERSISTENCE_CANDIDATE",
        "GENERATE_ROUTE_CANDIDATE",
        "GENERATE_REPRESENTATION_NATIVE_KERNEL",
    ),
    ("STATIC_KERNEL_VERIFY",),
    ("HOST_SHADER_ABI_VERIFY",),
    ("STRUCTURAL_COST_COMPARE",),
    ("DIAGNOSTIC_AB",),
    ("PROTECTED_AB",),
    ("FACTORIAL_COMBINATION",),
    ("UPDATE_REPATRIATION_LEDGER",),
    ("UPDATE_SCOREBOARD", "UPDATE_LAW", "UPDATE_SCAR"),
    ("TRANSFER_LAW",),
    ("ATTACK_LAW",),
)

# Back-edge. Not in the initial wave list; enqueued when a win is emitted.
REFILL_SPECIES = "REPROFILE_AFTER_WIN"

WIN_OUTCOMES = frozenset(
    {
        "PHYSICAL_WIN_MODEL_LOCAL",
        "PHYSICAL_WIN_FAMILY",
        "GENERIC_CANDIDATE",
        "GENERIC_VERIFIED",
    }
)

VERIFICATION_LEVELS = (
    "V0",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
)

WALL_CLASSES = frozenset({"STATIC", "CPU", "GPU_WINDOW", "PROTECTED_LEASE"})
RESOURCE_LANES = frozenset(
    {
        "static",
        "cpu",
        "metal_gpu",
        "metal_compiler",
        "protected_lease",
        "diagnostic_ab",
        "flash_nx",
    }
)

# Lane → which exact_physical_blockers substring arms it.
LANE_BLOCKER_NEEDLES: dict[str, tuple[str, ...]] = {
    "metal_gpu": ("no metal-capable gpu", "no qualified metal gpu"),
    "metal_compiler": ("metal compiler", "xcrun cannot locate"),
    "protected_lease": (
        "lock files exist",
        "flock would be a seizure",
        "will not quiesce",
        "classifies the machine heavy",
    ),
    "diagnostic_ab": ("no metal-capable gpu", "no qualified metal gpu"),
    "flash_nx": ("scaffold_only", "source-independent nx"),
}

LANE_WAKE: dict[str, str] = {
    "metal_gpu": (
        "MetalContext reports a Metal-capable GPU for the current execution host"
    ),
    "metal_compiler": (
        "xcrun locates the Metal compiler (utility `metal` found; not missing "
        "under CommandLineTools)"
    ),
    "protected_lease": (
        "an actual HCLI protected lease is present with a proven holder pid, "
        "the machine is QUIESCENT, and no protected lock is seized"
    ),
    "diagnostic_ab": (
        "MetalContext reports a Metal-capable GPU for the current execution host"
    ),
    "flash_nx": (
        "Flash source-independent NX is qualified (not SCAFFOLD_ONLY) and a "
        "complete-token receipt exists"
    ),
}

CODEX_UNIT_FIELDS = (
    "hypothesis",
    "objective",
    "evidence_parents",
    "inputs",
    "source_identities",
    "mutation_scope",
    "allowed_authority",
    "resources",
    "estimated_wall_class",
    "estimated_information_gain",
    "cheapest_falsifier",
    "verification_level",
    "stop_condition",
    "output_contract",
    "receipt_path",
    "transfer_value",
    "failure_inheritance",
)

STATUS_SLEEPING = "sleeping"
CLASS_SLEEPING = "SLEEPING"
GROUNDING_GROUNDED = "GROUNDED"
GROUNDING_UNGROUNDED = "UNGROUNDED_FROM_HANDOFF"

SIDECAR_CLAIM = (
    "Static sidecar artifact. No hardware measurement. Cannot promote, "
    "weaken a verifier, acquire a GPU lease, or write a protected result."
)

_UNSET: Any = object()
_HANDOFF: Any = _UNSET
_HANDOFF_SRC: str | None = None


class UngroundedSpeciesError(ValueError):
    """A species was offered with no observed Codex citation."""


class CitationError(ValueError):
    """A citation does not resolve to a quoted fragment in the handoff."""


class CodexSpeciesError(ValueError):
    """A Codex species or emitted unit violated the metabolism contract."""


# ---------------------------------------------------------------------------
# Handoff recovery. Disk is authority; missing in this sparse tree is not absence.
# ---------------------------------------------------------------------------


def load_handoff(*, force: bool = False) -> tuple[dict[str, Any] | None, str]:
    """Load CODEX_ACCELERATOR_HANDOFF.json from this tree or the primary checkout."""
    global _HANDOFF, _HANDOFF_SRC
    if not force and _HANDOFF is not _UNSET:
        if _HANDOFF is None:
            return None, _HANDOFF_SRC or "missing"
        return _HANDOFF, _HANDOFF_SRC or "cached"
    for root in ws._checkout_roots():
        path = Path(root) / HANDOFF_REL
        if path.is_file():
            doc = load_json(path)
            if not isinstance(doc, dict):
                raise CodexSpeciesError(f"{path}: handoff is not an object")
            _HANDOFF, _HANDOFF_SRC = doc, str(path)
            return _HANDOFF, _HANDOFF_SRC
    blob = git("show", f"HEAD:{HANDOFF_REL}")
    if blob:
        try:
            doc = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise CodexSpeciesError(f"git HEAD:{HANDOFF_REL} is not JSON: {exc}") from exc
        _HANDOFF, _HANDOFF_SRC = doc, f"git:HEAD:{HANDOFF_REL}"
        return _HANDOFF, _HANDOFF_SRC
    _HANDOFF, _HANDOFF_SRC = None, "missing"
    return None, "missing"


def inject_handoff(doc: dict[str, Any] | None, src: str = "injected") -> None:
    """Test seam. Production callers load from disk."""
    global _HANDOFF, _HANDOFF_SRC
    _HANDOFF, _HANDOFF_SRC = doc, src


def _require_handoff() -> tuple[dict[str, Any], str]:
    doc, src = load_handoff()
    if doc is None:
        raise FileNotFoundError(
            f"{HANDOFF_REL} is not visible from {REPO} (looked at checkout roots "
            "and git HEAD). The metabolism cannot be frozen without the training trace."
        )
    return doc, src


# ---------------------------------------------------------------------------
# Citations — the emitter refuses a species that did not happen in the handoff
# ---------------------------------------------------------------------------


def resolve_field_path(doc: Mapping[str, Any] | Sequence[Any], path: str) -> Any:
    """Resolve a.b[0].c against a JSON document. Missing path is an error."""
    if not path or not str(path).strip():
        raise CitationError("citation field_path is empty")
    cur: Any = doc
    token = ""
    i = 0
    text = str(path)
    while i < len(text):
        ch = text[i]
        if ch == ".":
            if token:
                cur = _descend(cur, token, path)
                token = ""
            i += 1
            continue
        if ch == "[":
            if token:
                cur = _descend(cur, token, path)
                token = ""
            close = text.find("]", i)
            if close < 0:
                raise CitationError(f"{path}: unclosed index")
            idx_s = text[i + 1 : close]
            try:
                idx = int(idx_s)
            except ValueError as exc:
                raise CitationError(f"{path}: non-integer index {idx_s!r}") from exc
            cur = _descend_index(cur, idx, path)
            i = close + 1
            continue
        token += ch
        i += 1
    if token:
        cur = _descend(cur, token, path)
    return cur


def _descend(cur: Any, key: str, path: str) -> Any:
    if not isinstance(cur, Mapping) or key not in cur:
        raise CitationError(f"{path}: {key!r} not present on {type(cur).__name__}")
    return cur[key]


def _descend_index(cur: Any, idx: int, path: str) -> Any:
    if not isinstance(cur, Sequence) or isinstance(cur, (str, bytes)):
        raise CitationError(f"{path}: cannot index {type(cur).__name__}")
    if idx < 0 or idx >= len(cur):
        raise CitationError(f"{path}: index {idx} out of range (n={len(cur)})")
    return cur[idx]


def _as_fragment_haystack(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def prove_citation(doc: Mapping[str, Any], citation: Mapping[str, Any]) -> dict[str, str]:
    """A citation is a field path plus a quoted fragment that actually occurs."""
    path = str(citation.get("field_path") or "").strip()
    fragment = str(citation.get("fragment") or "")
    if not path or not fragment:
        raise UngroundedSpeciesError(
            "species cites no observed Codex operation (field_path + fragment required)"
        )
    leaf = resolve_field_path(doc, path)
    haystack = _as_fragment_haystack(leaf)
    if fragment not in haystack:
        raise CitationError(
            f"citation fragment {fragment!r} not found at {path}"
        )
    return {"field_path": path, "fragment": fragment, "grounding": GROUNDING_GROUNDED}


# ---------------------------------------------------------------------------
# Physical blockers → SLEEPING. Disk state is authority. Never FAILED.
# ---------------------------------------------------------------------------


def blockers_from_handoff(doc: Mapping[str, Any] | None) -> list[str]:
    raw = (doc or {}).get("exact_physical_blockers") or []
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def lanes_blocked_by(blockers: Sequence[str]) -> dict[str, list[str]]:
    """Which resource lanes the observed blockers close. Derived, not fixed."""
    closed: dict[str, list[str]] = {lane: [] for lane in LANE_BLOCKER_NEEDLES}
    for blocker in blockers:
        text = blocker.lower()
        for lane, needles in sorted(LANE_BLOCKER_NEEDLES.items()):
            if any(n in text for n in needles):
                closed[lane].append(str(blocker))
    return {lane: hits for lane, hits in closed.items() if hits}


def wake_condition_for(lane: str, blockers: Sequence[str]) -> str:
    hits = lanes_blocked_by(blockers).get(lane) or list(blockers)
    clear = LANE_WAKE.get(lane, "the named resource lane becomes available")
    armed = "; ".join(hits) if hits else "lane listed as blocked by the handoff"
    return f"SLEEPING until: {clear}. Armed by: {armed}"


# ---------------------------------------------------------------------------
# Constructor. Extends workunit_species.define_species; refuses rival authority.
# ---------------------------------------------------------------------------


def define_codex_species(
    *,
    id: str,
    title: str,
    hypothesis: str,
    objective: str,
    evidence_parents: Sequence[str],
    inputs: Sequence[str],
    source_identities: Sequence[str],
    mutation_scope: str,
    bounded_authority: Sequence[str],
    resource_lane: str,
    resource_class: str,
    estimated_wall_class: str,
    estimated_information_gain: int,
    cheapest_falsifier: str,
    verification_level: str,
    stop_condition: str,
    output_contract: str,
    receipt_path: str,
    transfer_value: str,
    failure_inheritance: str,
    verifier: str,
    role: str,
    description: str,
    citation: Mapping[str, Any] | None = None,
    grounding: str = GROUNDING_GROUNDED,
    would_ground: str | None = None,
    loose_parent: str | None = None,
    extends_parent: str | None = None,
    effect_class: str = "READ_ONLY",
    era: str = "III",
    odyssey: str | None = "I",
    preferred_backend: str | None = None,
    may_promote: bool = False,
    may_modify_verifier: bool = False,
    may_choose_singularity: bool = False,
    may_destructive_mutate: bool = False,
    may_acquire_lease: bool = False,
    handoff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct one Codex species. Refuses promotion, lease seizure, and wish-list citations."""
    sid = str(id)
    if may_promote:
        raise ws.SpeciesAuthorityError(f"{sid}: species may not declare self-promotion authority")
    if may_modify_verifier:
        raise ws.SpeciesAuthorityError(f"{sid}: species may not declare verifier-modification authority")
    if may_choose_singularity:
        raise ws.SpeciesAuthorityError(f"{sid}: species may not choose the Singularity")
    if may_destructive_mutate:
        raise ws.SpeciesAuthorityError(f"{sid}: species may not perform a destructive mutation")
    if may_acquire_lease:
        raise ws.SpeciesAuthorityError(f"{sid}: species may not acquire a GPU lease")

    authority = tuple(str(item) for item in bounded_authority)
    if "acquire_gpu_lease" in authority:
        raise ws.SpeciesAuthorityError(
            f"{sid}: forbidden authority ['acquire_gpu_lease']; a species cannot take a GPU lease"
        )

    lane = str(resource_lane or "").strip()
    if lane not in RESOURCE_LANES:
        raise CodexSpeciesError(f"{sid}: resource_lane {resource_lane!r} is not a known lane")
    wall = str(estimated_wall_class or "").strip().upper()
    if wall not in WALL_CLASSES:
        raise CodexSpeciesError(f"{sid}: estimated_wall_class {estimated_wall_class!r} is refused")
    level = str(verification_level or "").strip().upper()
    if level not in VERIFICATION_LEVELS:
        raise CodexSpeciesError(f"{sid}: verification_level {verification_level!r} is not V0..V9")
    gain = int(estimated_information_gain)
    if gain < 1 or gain > 5:
        raise CodexSpeciesError(f"{sid}: estimated_information_gain must be a dimensionless 1..5")

    proven: dict[str, str] | None = None
    grounding_n = str(grounding or GROUNDING_GROUNDED).strip()
    if grounding_n == GROUNDING_UNGROUNDED:
        if not str(would_ground or "").strip():
            raise UngroundedSpeciesError(
                f"{sid}: UNGROUNDED_FROM_HANDOFF requires would_ground evidence"
            )
        if citation and citation.get("field_path") and citation.get("fragment"):
            raise UngroundedSpeciesError(
                f"{sid}: UNGROUNDED_FROM_HANDOFF must not invent a citation"
            )
    else:
        if not citation:
            raise UngroundedSpeciesError(
                f"{sid}: species cites no observed Codex operation"
            )
        doc = handoff
        if doc is None:
            loaded, _src = load_handoff()
            doc = loaded
        if doc is None:
            raise UngroundedSpeciesError(
                f"{sid}: cannot prove a citation because {HANDOFF_REL} is not visible"
            )
        proven = prove_citation(doc, citation)

    parent = ws.define_species(
        id=sid,
        title=title,
        evidence_parents=evidence_parents,
        bounded_authority=authority,
        resource_class=resource_class,
        verifier=verifier,
        budget=ws._budget(
            gpu_windows_requested=1 if lane in {"metal_gpu", "protected_lease", "diagnostic_ab"} else 0
        ),
        stop_condition=stop_condition,
        role=role,
        description=description,
        effect_class=effect_class,
        era=era,
        odyssey=odyssey,
        preferred_backend=preferred_backend,
        may_promote=False,
        may_modify_verifier=False,
        may_choose_singularity=False,
        may_destructive_mutate=False,
    )
    parent.update(
        {
            "hypothesis": str(hypothesis),
            "objective": str(objective),
            "inputs": [str(x) for x in inputs],
            "source_identities": [str(x) for x in source_identities],
            "mutation_scope": str(mutation_scope),
            "allowed_authority": list(authority),
            "resources": {
                "lane": lane,
                "resource_class": parent["resource_class"],
                "gpu_authority": False,
            },
            "estimated_wall_class": wall,
            "estimated_information_gain": gain,
            "cheapest_falsifier": str(cheapest_falsifier),
            "verification_level": level,
            "output_contract": str(output_contract),
            "receipt_path": str(receipt_path),
            "transfer_value": str(transfer_value),
            "failure_inheritance": str(failure_inheritance),
            "grounding": grounding_n,
            "citation": proven,
            "would_ground": str(would_ground) if grounding_n == GROUNDING_UNGROUNDED else None,
            "loose_parent": loose_parent,
            "extends_parent": extends_parent,
            "may_acquire_lease": False,
            "claim_boundary": SIDECAR_CLAIM,
            "measurement_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
        }
    )
    return parent


# ---------------------------------------------------------------------------
# The thirty species. Citations are field paths into the Codex handoff.
# ---------------------------------------------------------------------------


def _auth(*names: str) -> tuple[str, ...]:
    return names


_READ = _auth("read_receipts", "write_sidecar_receipt")
_PROFILE = _auth("read_receipts", "record_unknown_metrics", "propose_workunit", "write_sidecar_receipt")
_RANK = _auth("read_receipts", "rank_falsifiable_experiments", "run_static_analysis", "write_sidecar_receipt")
_GENERATE = _auth(
    "read_receipts",
    "compile_experiment_spec",
    "propose_workunit",
    "emit_static_plan",
    "write_sidecar_receipt",
)
_VERIFY = _auth("read_receipts", "run_static_analysis", "audit_dependency_chain", "write_sidecar_receipt")
_AB = _auth("read_receipts", "propose_workunit", "copy_live_workunit_fields", "write_sidecar_receipt")
_LEDGER = _auth("read_receipts", "write_sidecar_receipt", "copy_live_workunit_fields")
_LAW = _auth("read_receipts", "seed_law_store", "write_sidecar_receipt")
_TRANSFER = _auth("read_receipts", "transfer_law_within_declared_scope", "write_sidecar_receipt")
_ATTACK = _auth(
    "read_receipts",
    "query_negative_index",
    "adversarially_attack_a_claimed_law",
    "run_static_analysis",
    "write_sidecar_receipt",
)
_FACTORIAL = _auth("read_receipts", "audit_dependency_chain", "emit_static_plan", "write_sidecar_receipt")


def _species_specs() -> tuple[dict[str, Any], ...]:
    fail_descendants = (
        "rejection hard-invalidates declared descendants; cross-model equivalence "
        "siblings are questioned, not assumed"
    )
    unknown_metric = (
        "a missing metric field remains UNKNOWN; an invented number is a campaign failure"
    )
    no_mutate = "sidecar_receipts_future_only; never crates/**, never receipts/headless/**"
    proposal_only = "proposal_only; Codex owns the physical mutation"

    return (
        dict(
            id="PROFILE_COMPLETE_TOKEN",
            title="Profile a complete accepted token",
            hypothesis="Complete-token wall, not a kernel microbench, is the quantity the loop ranks.",
            objective="Capture or recover the incumbent complete-token profile so FIND_TALLEST_COST has a denominator.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:current_flash_state.critical_path",
                "CODEX_ACCELERATOR_HANDOFF.json:current_qwen27_incumbent_control_identity.control_receipt",
            ),
            inputs=("incumbent control receipt", "complete-token measurement contract"),
            source_identities=("qwen27-fast-profile", "sealed-3.14"),
            mutation_scope=no_mutate,
            bounded_authority=_PROFILE,
            resource_lane="metal_gpu",
            resource_class="GPU_EXCLUSIVE",
            estimated_wall_class="PROTECTED_LEASE",
            estimated_information_gain=5,
            cheapest_falsifier="required field complete_wall_ns_per_accepted_token is missing or fallback_count is nonzero",
            verification_level="V6",
            stop_condition=(
                "stop when a complete-token receipt exists with every required_measurement_field "
                "present or explicitly UNKNOWN. Never promote a diagnostic to complete-token."
            ),
            output_contract="STATIC_ONLY profile identity: path, fields present/absent, no invented ns",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="feeds FIND_TALLEST_COST; without it the loop has no denominator",
            failure_inheritance=unknown_metric,
            verifier="future.codex.profile_complete_token",
            role="accelerator_physical_qualification",
            description=(
                "Profile the complete accepted generated token (Codex: WAITING_FOR_COMPLETE_TOKEN "
                "on Flash; Qwen27 incumbent control is PROTECTED_CONTROL_NOT_FOR_PROMOTION)."
            ),
            citation={
                "field_path": "current_flash_state.critical_path.status",
                "fragment": "WAITING_FOR_COMPLETE_TOKEN",
            },
            loose_parent="GPU kernel optimization",
            extends_parent="accelerator_candidate_qualification",
            effect_class="READ_ONLY",
            preferred_backend="metal",
        ),
        dict(
            id="PROFILE_REGION",
            title="Profile an affected physical region",
            hypothesis="The tallest cost lives in a named organ/region, not in the whole graph equally.",
            objective="Bind the current episode to the region Codex named as the physical effect.",
            evidence_parents=("CODEX_ACCELERATOR_HANDOFF.json:atomic_change.physical_effect_intended",),
            inputs=("atomic_change.physical_effect_intended", "affected_physical_region on the live queue"),
            source_identities=("flash-device-mhc-state",),
            mutation_scope=no_mutate,
            bounded_authority=_PROFILE,
            resource_lane="metal_gpu",
            resource_class="GPU_EXCLUSIVE",
            estimated_wall_class="GPU_WINDOW",
            estimated_information_gain=4,
            cheapest_falsifier="the named region does not appear in the candidate graph",
            verification_level="V4",
            stop_condition="stop when the region identity is bound to a candidate_id or the region is absent",
            output_contract="region identity + source candidate; no timing claim",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="narrows GENERATE_* to the organ that actually costs",
            failure_inheritance=unknown_metric,
            verifier="future.codex.profile_region",
            role="accelerator_physical_qualification",
            description="Profile the physical region Codex intended to change in the atomic episode.",
            citation={
                "field_path": "atomic_change.physical_effect_intended[1]",
                "fragment": "parallelize the 4096-wide device mHC RMSNorm boundary",
            },
            loose_parent="GPU kernel optimization",
            extends_parent="accelerator_candidate_qualification",
            effect_class="READ_ONLY",
            preferred_backend="metal",
        ),
        dict(
            id="PROFILE_GPU",
            title="Profile GPU-side time",
            hypothesis="GPU time is a column of the complete-token contract, never a substitute for it.",
            objective="Record whether gpu_ns_per_token is present on the incumbent; do not invent it.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:current_staged_protected_batch.protocol.required_measurement_fields",
            ),
            inputs=("required_measurement_fields", "incumbent control_receipt.metrics keys"),
            source_identities=("qwen27-fast-profile",),
            mutation_scope=no_mutate,
            bounded_authority=_PROFILE,
            resource_lane="metal_gpu",
            resource_class="GPU_EXCLUSIVE",
            estimated_wall_class="PROTECTED_LEASE",
            estimated_information_gain=4,
            cheapest_falsifier="gpu_ns_per_token absent and no absence_reason",
            verification_level="V6",
            stop_condition=(
                "stop when gpu_ns_per_token is present on a protected receipt or recorded UNKNOWN. "
                "physical_execution NOT_RUN is a legal stop, not a synthetic GPU number."
            ),
            output_contract="field presence for gpu_ns_per_token; value remains UNKNOWN in this lane",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="separates GPU time from host ceremony and dispatch count",
            failure_inheritance=unknown_metric,
            verifier="future.codex.profile_gpu",
            role="accelerator_physical_qualification",
            description="Profile GPU-side time as a required complete-token field, never as a promotion.",
            citation={
                "field_path": "current_staged_protected_batch.protocol.required_measurement_fields",
                "fragment": "gpu_ns_per_token",
            },
            loose_parent="GPU kernel optimization",
            extends_parent="accelerator_candidate_qualification",
            effect_class="READ_ONLY",
            preferred_backend="metal",
        ),
        dict(
            id="PROFILE_HOST_CEREMONY",
            title="Profile host ceremony",
            hypothesis="Commit/label/pipeline-state ceremony can dominate entering the loop.",
            objective="Account host ceremony as its own denominator so S3-host-ceremony-union is meaningful.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:exact_next_protected_qualification_sequence.qwen27.step_3_dependency_cells_in_order",
            ),
            inputs=("S3-host-ceremony-union", "pipeline/commit/encoder-label candidates"),
            source_identities=(
                "qwen27-commit-timing-elision",
                "qwen27-encoder-label-elision",
                "qwen27-pipeline-state-elision",
            ),
            mutation_scope=no_mutate,
            bounded_authority=_PROFILE,
            resource_lane="metal_gpu",
            resource_class="GPU_EXCLUSIVE",
            estimated_wall_class="GPU_WINDOW",
            estimated_information_gain=4,
            cheapest_falsifier="host_roundtrip_count / commit path is not named on the profile",
            verification_level="V4",
            stop_condition="stop when ceremony candidates are listed with a shared union cell_id",
            output_contract="ceremony candidate set + union cell identity",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="feeds GENERATE_PIPELINE_PERSISTENCE_CANDIDATE and the S3 ceremony union",
            failure_inheritance=unknown_metric,
            verifier="future.codex.profile_host_ceremony",
            role="accelerator_physical_qualification",
            description="Profile host ceremony (Codex S3-host-ceremony-union and source ceremony elimination).",
            citation={
                "field_path": "exact_next_protected_qualification_sequence.qwen27.step_3_dependency_cells_in_order[0]",
                "fragment": "S3-host-ceremony-union",
            },
            loose_parent="source ceremony elimination",
            extends_parent="accelerator_candidate_qualification",
            effect_class="READ_ONLY",
            preferred_backend="metal",
        ),
        dict(
            id="PROFILE_ACTIVE_BYTES",
            title="Profile active representation bytes",
            hypothesis="Active bytes per token, not packed footprint, bound the move/recompute decision.",
            objective="See whether active_representation_bytes_per_token is on the measurement contract.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:current_staged_protected_batch.protocol.required_measurement_fields",
            ),
            inputs=("required_measurement_fields",),
            source_identities=("qwen27-fast-profile",),
            mutation_scope=no_mutate,
            bounded_authority=_PROFILE,
            resource_lane="metal_gpu",
            resource_class="GPU_EXCLUSIVE",
            estimated_wall_class="PROTECTED_LEASE",
            estimated_information_gain=4,
            cheapest_falsifier="active_representation_bytes_per_token absent from the contract",
            verification_level="V6",
            stop_condition="stop when the field is listed present or UNKNOWN; never invent a byte count",
            output_contract="field presence for active_representation_bytes_per_token",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="feeds GENERATE_REPRESENTATION_NATIVE_KERNEL and active-byte reduction",
            failure_inheritance=unknown_metric,
            verifier="future.codex.profile_active_bytes",
            role="accelerator_physical_qualification",
            description="Profile active representation bytes per token as a required contract field.",
            citation={
                "field_path": "current_staged_protected_batch.protocol.required_measurement_fields",
                "fragment": "active_representation_bytes_per_token",
            },
            loose_parent="active-byte reduction",
            extends_parent="accelerator_candidate_qualification",
            effect_class="READ_ONLY",
            preferred_backend="metal",
        ),
        dict(
            id="PROFILE_RESIDENT_BYTES",
            title="Profile resident bytes",
            hypothesis="Keeping state resident is a byte decision; residency that is not measured is a wish.",
            objective="Bind resident_bytes and the mHC residency hypothesis to the same contract.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:current_staged_protected_batch.protocol.required_measurement_fields",
                "CODEX_ACCELERATOR_HANDOFF.json:atomic_change.physical_effect_intended",
            ),
            inputs=("required_measurement_fields", "device-mHC residency intent"),
            source_identities=("flash-device-mhc-state",),
            mutation_scope=no_mutate,
            bounded_authority=_PROFILE,
            resource_lane="metal_gpu",
            resource_class="GPU_EXCLUSIVE",
            estimated_wall_class="PROTECTED_LEASE",
            estimated_information_gain=4,
            cheapest_falsifier="resident_bytes absent and mHC state is claimed resident",
            verification_level="V6",
            stop_condition="stop when resident_bytes is present or UNKNOWN",
            output_contract="field presence for resident_bytes and total_nx_bytes",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="feeds GENERATE_STATE_RESIDENCY_CANDIDATE",
            failure_inheritance=unknown_metric,
            verifier="future.codex.profile_resident_bytes",
            role="accelerator_physical_qualification",
            description="Profile resident bytes; Codex required the field and intended mHC state to stay resident.",
            citation={
                "field_path": "current_staged_protected_batch.protocol.required_measurement_fields",
                "fragment": "resident_bytes",
            },
            loose_parent="pipeline persistence",
            extends_parent="accelerator_candidate_qualification",
            effect_class="READ_ONLY",
            preferred_backend="metal",
        ),
        dict(
            id="PROFILE_DISPATCH",
            title="Profile dispatches per token",
            hypothesis="Dispatch count is a column, not the cost (AKB-DISPATCH-COUNT-DOES-NOT-PREDICT-COST).",
            objective="Record dispatches_per_token so structural compare can refute S031-style ladders.",
            evidence_parents=("CODEX_ACCELERATOR_HANDOFF.json:known_laws.active_law_ids",),
            inputs=("AKB-964-DISPATCHES-PER-DECODE-TOKEN", "required_measurement_fields"),
            source_identities=("AKB-964-DISPATCHES-PER-DECODE-TOKEN",),
            mutation_scope=no_mutate,
            bounded_authority=_PROFILE,
            resource_lane="metal_gpu",
            resource_class="GPU_EXCLUSIVE",
            estimated_wall_class="GPU_WINDOW",
            estimated_information_gain=3,
            cheapest_falsifier="two graphs share a dispatch count and differ in complete-token wall",
            verification_level="V4",
            stop_condition="stop when dispatches_per_token is present or UNKNOWN; never rank by count alone",
            output_contract="field presence for dispatches_per_token; no cost claim from the count",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="feeds STRUCTURAL_COST_COMPARE and dispatch/sync elimination",
            failure_inheritance=unknown_metric,
            verifier="future.codex.profile_dispatch",
            role="accelerator_physical_qualification",
            description="Profile dispatches per token. Codex already named AKB-964-DISPATCHES-PER-DECODE-TOKEN.",
            citation={
                "field_path": "known_laws.active_law_ids",
                "fragment": "AKB-964-DISPATCHES-PER-DECODE-TOKEN",
            },
            loose_parent="dispatch/sync elimination",
            extends_parent="accelerator_candidate_qualification",
            effect_class="READ_ONLY",
            preferred_backend="metal",
        ),
        dict(
            id="PROFILE_SYNC",
            title="Profile synchronization cost",
            hypothesis="Wait/sync can dominate submission (AKB-WAIT-DOMINATES-SUBMISSION, conditional).",
            objective="Keep sync_ns_per_token as its own column so fusion does not hide a fence.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:current_staged_protected_batch.protocol.required_measurement_fields",
                "CODEX_ACCELERATOR_HANDOFF.json:known_laws.conditional_law_ids",
            ),
            inputs=("sync_ns_per_token", "AKB-WAIT-DOMINATES-SUBMISSION"),
            source_identities=("AKB-WAIT-DOMINATES-SUBMISSION",),
            mutation_scope=no_mutate,
            bounded_authority=_PROFILE,
            resource_lane="metal_gpu",
            resource_class="GPU_EXCLUSIVE",
            estimated_wall_class="GPU_WINDOW",
            estimated_information_gain=3,
            cheapest_falsifier="sync_ns_per_token absent from a claimed complete-token receipt",
            verification_level="V4",
            stop_condition="stop when sync_ns_per_token is present or UNKNOWN",
            output_contract="field presence for sync_ns_per_token",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="feeds dispatch/sync elimination and fusion candidates",
            failure_inheritance=unknown_metric,
            verifier="future.codex.profile_sync",
            role="accelerator_physical_qualification",
            description="Profile synchronization as a required complete-token field.",
            citation={
                "field_path": "current_staged_protected_batch.protocol.required_measurement_fields",
                "fragment": "sync_ns_per_token",
            },
            loose_parent="dispatch/sync elimination",
            extends_parent="accelerator_candidate_qualification",
            effect_class="READ_ONLY",
            preferred_backend="metal",
        ),
        dict(
            id="FIND_TALLEST_COST",
            title="Rank the tallest denominator",
            hypothesis="The floor mechanism is unresolved after six levers; dispatch count is not the wall.",
            objective="Rank which profiled column is the tallest cost using laws, never using a guessed ns.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:known_laws.current_relevant_laws",
                "receipts/future/evidence/ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json",
            ),
            inputs=("PROFILE_* receipts or incumbent control identity", "current_relevant_laws"),
            source_identities=(
                "AKB-THE-UNPACK-IS-THE-WALL-NOT-THE-BYTES",
                "AKB-DISPATCH-COUNT-DOES-NOT-PREDICT-COST",
                "AKB-SIX-LEVERS-ELIMINATED-AND-THE-FLOOR-MECHANISM-IS-UNRESOLVED",
            ),
            mutation_scope=no_mutate,
            bounded_authority=_RANK,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=5,
            cheapest_falsifier="the ranked column is a dispatch count used as if it were complete-token wall",
            verification_level="V1",
            stop_condition=(
                "stop when one column is named as tallest with a law id, or when every column is UNKNOWN "
                "and the unit records that it cannot rank."
            ),
            output_contract="ranked column + law ids; STATIC_ONLY; no hardware number",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="selects which GENERATE_* family fires next",
            failure_inheritance="a failed rank does not invent a tallest cost; the cycle waits",
            verifier="future.codex.find_tallest_cost",
            role="science",
            description=(
                "Rank the tallest cost. Codex: the unpack is the wall, dispatch count does not predict "
                "cost, six levers eliminated and the floor mechanism is unresolved."
            ),
            citation={
                "field_path": "known_laws.current_relevant_laws",
                "fragment": "AKB-THE-UNPACK-IS-THE-WALL-NOT-THE-BYTES",
            },
            loose_parent=None,
            extends_parent="hardware_doctor_experiment",
            effect_class="READ_ONLY",
        ),
        dict(
            id="SEARCH_ARCHITECTURE_LAWS",
            title="Search architecture laws for why the cost exists",
            hypothesis="Why the cost exists is already in the atlas and AKB; do not re-derive from a prompt.",
            objective="Pull atlas primitives and current_relevant_laws that explain the tallest column.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:accelerator_architecture_atlas_identity",
                "CODEX_ACCELERATOR_HANDOFF.json:known_laws.source",
            ),
            inputs=("FIND_TALLEST_COST output", "ACCELERATOR_ARCHITECTURE_ATLAS.json", "ACCELERATOR_LAW_BASE.json"),
            source_identities=("hawking.accelerator.architecture_atlas.v1", "hawking.accelerator.akb.v1"),
            mutation_scope=no_mutate,
            bounded_authority=_RANK,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=4,
            cheapest_falsifier="a proposed why has no atlas primitive and no AKB id",
            verification_level="V1",
            stop_condition="stop when at least one law id and one atlas primitive bind to the tallest column",
            output_contract="law ids + atlas primitive names; scope remains as stored",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="grounds GENERATE_* in existing law rather than a new story",
            failure_inheritance="missing atlas is not a reason to invent primitives",
            verifier="future.codex.search_architecture_laws",
            role="science",
            description="Ask why the tallest cost exists using the Architecture Atlas and AKB, not a new narrative.",
            citation={
                "field_path": "accelerator_architecture_atlas_identity.path",
                "fragment": "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json",
            },
            loose_parent=None,
            extends_parent="architecture_transfer",
            effect_class="READ_ONLY",
            odyssey="II",
        ),
        dict(
            id="GENERATE_KERNEL_CANDIDATE",
            title="Generate a GPU kernel candidate",
            hypothesis="The atomic episode is a kernel-shaped mutation under opt-in device controls.",
            objective="Emit a kernel-candidate spec with source_oracle_control vs candidate_control.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:atomic_change",
                "CODEX_ACCELERATOR_HANDOFF.json:recurring_hcli_workunit_species",
            ),
            inputs=("SEARCH_ARCHITECTURE_LAWS", "atomic_change.source_oracle_control", "atomic_change.candidate_control"),
            source_identities=("flash-device-mhc-state", "HAWKING_DSV4F_DEVICE_MHC"),
            mutation_scope=proposal_only,
            bounded_authority=_GENERATE,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=5,
            cheapest_falsifier="candidate_control equals source_oracle_control (no mutation)",
            verification_level="V0",
            stop_condition="stop when a candidate spec names files, controls, and intended physical effect",
            output_contract="candidate spec; IMPLEMENTED_OPT_IN_UNMEASURED is not a win",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="the default next physical mutation family Codex actually ran",
            failure_inheritance=fail_descendants,
            verifier="future.codex.generate_kernel_candidate",
            role="science",
            description=(
                "Generate a kernel candidate in the shape of Codex's atomic_change "
                "(device-mHC / SIMD RMSNorm / norm-to-activation-quant fusion)."
            ),
            citation={
                "field_path": "atomic_change.name",
                "fragment": "device-mHC / SIMD RMSNorm / norm-to-activation-quant fusion",
            },
            loose_parent="GPU kernel optimization",
            extends_parent="accelerator_candidate_qualification",
            effect_class="REVERSIBLE",
        ),
        dict(
            id="GENERATE_FUSION_CANDIDATE",
            title="Generate a fusion candidate",
            hypothesis="Fusion beats materialising (AKB-FUSION-BEATS-MATERIALISING) when it removes a dispatch of real bytes.",
            objective="Emit a fusion candidate spec from the live fusion family (QKV, gate, DeltaNet, mHC-norm).",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:recurring_hcli_workunit_species",
                "CODEX_ACCELERATOR_HANDOFF.json:atomic_change.physical_effect_intended",
            ),
            inputs=("SEARCH_ARCHITECTURE_LAWS", "qwen27 fusion candidate ids", "physical_effect_intended fusion clause"),
            source_identities=(
                "qwen27-gqa-qkv-fusion",
                "qwen27-attention-gate-fusion",
                "qwen27-deltanet-inproj-fusion",
                "qwen27-ba-delta-fusion",
            ),
            mutation_scope=proposal_only,
            bounded_authority=_GENERATE,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=5,
            cheapest_falsifier="the fused kernel still launches the removed dispatch",
            verification_level="V0",
            stop_condition="stop when a fusion spec names the eliminated dispatch and the surviving kernel",
            output_contract="fusion candidate spec; graph shape is not a timing result",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="Codex's named recurring species 'fusion search'",
            failure_inheritance=fail_descendants,
            verifier="future.codex.generate_fusion_candidate",
            role="science",
            description="Generate a fusion candidate. Codex named fusion search and fused RMSNorm with activation quant.",
            citation={
                "field_path": "recurring_hcli_workunit_species",
                "fragment": "fusion search",
            },
            loose_parent="fusion search",
            extends_parent="fusion_simulation",
            effect_class="REVERSIBLE",
        ),
        dict(
            id="GENERATE_LAYOUT_CANDIDATE",
            title="Generate a layout candidate",
            hypothesis="Geometry (split-K, vecgroup) can raise occupancy without changing representation.",
            objective="Emit a layout/geo candidate from the live Qwen27 geometry family.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:recurring_hcli_workunit_species",
                "CODEX_ACCELERATOR_HANDOFF.json:current_high_ev_unmeasured_candidates.qwen27",
            ),
            inputs=("SEARCH_ARCHITECTURE_LAWS", "qwen27-affine2-splitk4", "qwen27-q4-vecgroup-x64", "qwen27-q2f-splitk4"),
            source_identities=("qwen27-affine2-splitk4", "qwen27-q4-vecgroup-x64", "qwen27-q2f-splitk4"),
            mutation_scope=proposal_only,
            bounded_authority=_GENERATE,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=4,
            cheapest_falsifier="geo flag does not change the packed layout the kernel reads",
            verification_level="V0",
            stop_condition="stop when a layout spec names the geo key and the control geo",
            output_contract="layout candidate spec",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="Codex's named recurring species 'layout search'",
            failure_inheritance=fail_descendants,
            verifier="future.codex.generate_layout_candidate",
            role="science",
            description="Generate a layout candidate from Codex's layout search and Qwen27 geometry rows.",
            citation={
                "field_path": "recurring_hcli_workunit_species",
                "fragment": "layout search",
            },
            loose_parent="layout search",
            extends_parent="learned_compiler_experiment",
            effect_class="REVERSIBLE",
        ),
        dict(
            id="GENERATE_STATE_RESIDENCY_CANDIDATE",
            title="Generate a state-residency candidate",
            hypothesis="Keeping mHC state resident between attention and FFN removes a round-trip.",
            objective="Emit a residency candidate whose control is the non-resident path.",
            evidence_parents=("CODEX_ACCELERATOR_HANDOFF.json:atomic_change.physical_effect_intended",),
            inputs=("SEARCH_ARCHITECTURE_LAWS", "HAWKING_DSV4F_DEVICE_MHC", "flash-device-mhc-state"),
            source_identities=("flash-device-mhc-state", "AKB-DEVICE-RESIDENT-OPERANDS"),
            mutation_scope=proposal_only,
            bounded_authority=_GENERATE,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=5,
            cheapest_falsifier="state is rewritten to DRAM between the named boundaries",
            verification_level="V0",
            stop_condition="stop when the spec names the resident object and the two boundaries it must survive",
            output_contract="state-residency candidate spec; IMPLEMENTED_UNMEASURED is not a win",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="the atomic_change Codex actually implemented opt-in",
            failure_inheritance=fail_descendants,
            verifier="future.codex.generate_state_residency_candidate",
            role="science",
            description="Generate a state-residency candidate (keep mHC state resident between attention and FFN).",
            citation={
                "field_path": "atomic_change.physical_effect_intended[0]",
                "fragment": "keep mHC state resident between attention and FFN boundaries",
            },
            loose_parent="pipeline persistence",
            extends_parent="accelerator_candidate_qualification",
            effect_class="REVERSIBLE",
        ),
        dict(
            id="GENERATE_PIPELINE_PERSISTENCE_CANDIDATE",
            title="Generate a pipeline-persistence candidate",
            hypothesis="Per-token pipeline lookup/id-resolution/state binding is ceremony that can persist.",
            objective="Emit pipeline-cache / id-resolution / state-elision candidate specs.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:recurring_hcli_workunit_species",
                "CODEX_ACCELERATOR_HANDOFF.json:current_high_ev_unmeasured_candidates.qwen27",
            ),
            inputs=("PROFILE_HOST_CEREMONY", "qwen27-pipeline-*", "flash-pipeline-*"),
            source_identities=(
                "qwen27-pipeline-cache-reuse",
                "qwen27-pipeline-id-resolution",
                "qwen27-pipeline-state-elision",
            ),
            mutation_scope=proposal_only,
            bounded_authority=_GENERATE,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=4,
            cheapest_falsifier="the pipeline is rebuilt every token despite the flag",
            verification_level="V0",
            stop_condition="stop when a persistence spec names the cached object and the lookup it elides",
            output_contract="pipeline-persistence candidate spec",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="Codex's named recurring species 'pipeline persistence'",
            failure_inheritance=fail_descendants,
            verifier="future.codex.generate_pipeline_persistence_candidate",
            role="science",
            description="Generate a pipeline-persistence candidate from Codex's named species and Qwen27 pipeline rows.",
            citation={
                "field_path": "recurring_hcli_workunit_species",
                "fragment": "pipeline persistence",
            },
            loose_parent="pipeline persistence",
            extends_parent="accelerator_candidate_qualification",
            effect_class="REVERSIBLE",
        ),
        dict(
            id="GENERATE_ROUTE_CANDIDATE",
            title="Generate a route candidate",
            hypothesis="Route-before-payload and routed fused epilogues are the Flash MoE lever.",
            objective="Emit a routed-expert / top-k / backend-placement candidate spec.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:current_high_ev_unmeasured_candidates.flash",
                "CODEX_ACCELERATOR_HANDOFF.json:recurring_hcli_workunit_species",
            ),
            inputs=("SEARCH_ARCHITECTURE_LAWS", "flash-p6-routed-fp4-gate-up-swiglu-fused"),
            source_identities=("flash-p6-routed-fp4-gate-up-swiglu-fused", "flash-router-topk-fusion"),
            mutation_scope=proposal_only,
            bounded_authority=_GENERATE,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=4,
            cheapest_falsifier="the route metadata does not change which expert body runs",
            verification_level="V0",
            stop_condition="stop when a route spec names the selected-expert path and the control graph",
            output_contract="route candidate spec; Flash NX remaining SCAFFOLD_ONLY keeps execution SLEEPING",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="Codex's named recurring species 'backend-placement experiments' plus routed Flash rows",
            failure_inheritance=fail_descendants,
            verifier="future.codex.generate_route_candidate",
            role="science",
            description="Generate a route/backend-placement candidate from the Flash routed fused-epilogue family.",
            citation={
                "field_path": "current_high_ev_unmeasured_candidates.flash",
                "fragment": "flash-p6-routed-fp4-gate-up-swiglu-fused",
            },
            loose_parent="backend-placement experiments",
            extends_parent="architecture_transfer",
            effect_class="REVERSIBLE",
        ),
        dict(
            id="GENERATE_REPRESENTATION_NATIVE_KERNEL",
            title="Generate a representation-native kernel candidate",
            hypothesis="Native-packed kernels, not a generic matmul, are what the incumbent already is.",
            objective="Emit a representation-native kernel spec bound to the incumbent kind native-packed.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:recurring_hcli_workunit_species",
                "CODEX_ACCELERATOR_HANDOFF.json:current_qwen27_incumbent_control_identity.profile.representation",
            ),
            inputs=("PROFILE_ACTIVE_BYTES", "representation.kind", "Flash source-BF16 / compact MoE rows"),
            source_identities=("native-packed", "sealed-3.14"),
            mutation_scope=proposal_only,
            bounded_authority=_GENERATE,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=4,
            cheapest_falsifier="the kernel unpacks to a generic layout before computing",
            verification_level="V0",
            stop_condition="stop when the spec names the packed representation and the kernel that consumes it in place",
            output_contract="representation-native kernel spec",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="Codex's named recurring species 'representation-native kernels'",
            failure_inheritance=fail_descendants,
            verifier="future.codex.generate_representation_native_kernel",
            role="science",
            description="Generate a representation-native kernel candidate. Incumbent representation kind is native-packed.",
            citation={
                "field_path": "recurring_hcli_workunit_species",
                "fragment": "representation-native kernels",
            },
            loose_parent="representation-native kernels",
            extends_parent="learned_compiler_experiment",
            effect_class="REVERSIBLE",
        ),
        dict(
            id="STATIC_KERNEL_VERIFY",
            title="Static kernel verify",
            hypothesis="Static correctness does not prove speed or physical parity (Codex scar).",
            objective="Run the zero-GPU host/shader preflight before any AB window is requested.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:known_scars",
                "receipts/future/STATIC_KERNEL_PREFLIGHT.json",
                "tools/future/static_kernel_verify.py",
            ),
            inputs=("GENERATE_* specs", "crates/hawking-core/shaders"),
            source_identities=("static-verifier-boundary",),
            mutation_scope=no_mutate,
            bounded_authority=_VERIFY,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=3,
            cheapest_falsifier="ERROR>0 on the preflight, or the unit claims speed from a static pass",
            verification_level="V2",
            stop_condition=(
                "stop when the preflight receipt is sealed. UNVERIFIABLE is recorded, not silently passed. "
                "A static pass must not be described as a protected result."
            ),
            output_contract="STATIC_ONLY preflight identity; ERROR/WARNING/UNVERIFIABLE counts as already sealed",
            receipt_path="receipts/future/STATIC_KERNEL_PREFLIGHT.json",
            transfer_value="drops defects that would waste a protected GPU window",
            failure_inheritance="static ERROR blocks DIAGNOSTIC_AB and PROTECTED_AB",
            verifier="future.codex.static_kernel_verify",
            role="science",
            description="Static kernel verify. Codex scar: static correctness does not prove speed or physical parity.",
            citation={
                "field_path": "known_scars[1].name",
                "fragment": "static-verifier-boundary",
            },
            loose_parent="GPU kernel optimization",
            extends_parent="independent_reproduction",
            effect_class="READ_ONLY",
        ),
        dict(
            id="HOST_SHADER_ABI_VERIFY",
            title="Host/shader ABI verify",
            hypothesis="ABI drift between host binds and shader parameters is a static defect, not a GPU mystery.",
            objective="Check host/shader ABI; Metal compiler absence is SLEEPING for compile, not a skip of the parse.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:known_scars",
                "CODEX_ACCELERATOR_HANDOFF.json:atomic_change.validation.metal_source_compilation",
                "receipts/future/CLAUDE_SIDECAR_ABI_ADJUDICATION.json",
            ),
            inputs=("STATIC_KERNEL_VERIFY", "host bind sites", "shader parameters"),
            source_identities=("claude-six-static-abi-findings", "BLOCKED_NO_METAL_TOOLCHAIN"),
            mutation_scope=no_mutate,
            bounded_authority=_VERIFY,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=3,
            cheapest_falsifier="a bind index disagrees with the shader, or xcrun metal is treated as a pass",
            verification_level="V3",
            stop_condition=(
                "stop when the ABI receipt is sealed. BLOCKED_NO_METAL_TOOLCHAIN is recorded as a sleeping "
                "compile substep, never as ABI-clean."
            ),
            output_contract="ABI findings + compile-substep state (SLEEPING if no Metal compiler)",
            receipt_path="receipts/future/CLAUDE_SIDECAR_ABI_ADJUDICATION.json",
            transfer_value="the six ABI findings Codex recorded as a scar",
            failure_inheritance="ABI defect blocks AB; missing compiler does not fail the parse",
            verifier="future.codex.host_shader_abi_verify",
            role="science",
            description="Host/shader ABI verify. Codex: claude-six-static-abi-findings; metal compile BLOCKED_NO_METAL_TOOLCHAIN.",
            citation={
                "field_path": "known_scars[0].name",
                "fragment": "claude-six-static-abi-findings",
            },
            loose_parent="GPU kernel optimization",
            extends_parent="independent_reproduction",
            effect_class="READ_ONLY",
        ),
        dict(
            id="STRUCTURAL_COST_COMPARE",
            title="Structural cost compare",
            hypothesis="Two graphs at the same dispatch count are not the same cost.",
            objective="Compare candidate vs control structurally (bytes, dispatches, intermediates) without a GPU number.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:known_laws.current_relevant_laws",
                "CODEX_ACCELERATOR_HANDOFF.json:current_staged_protected_batch.protocol.reject_if",
            ),
            inputs=("PROFILE_*", "GENERATE_*", "AKB-DISPATCH-COUNT-DOES-NOT-PREDICT-COST"),
            source_identities=("AKB-DISPATCH-COUNT-DOES-NOT-PREDICT-COST",),
            mutation_scope=no_mutate,
            bounded_authority=_VERIFY,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=4,
            cheapest_falsifier="the compare ranks by dispatch count or by a mean of un-paired runs",
            verification_level="V1",
            stop_condition=(
                "stop when the structural delta is sealed. A structural win is not a protected win and "
                "must not beat the matched control by assertion."
            ),
            output_contract="structural delta (bytes/dispatches/intermediates) as identities, not ns",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="drops candidates that cannot win even on paper, before a GPU window",
            failure_inheritance="a structural loss skips AB; a structural win still requires AB",
            verifier="future.codex.structural_cost_compare",
            role="science",
            description="Structural cost compare. Codex: dispatch count does not predict cost; paired CI, no mean promotion.",
            citation={
                "field_path": "known_laws.current_relevant_laws",
                "fragment": "AKB-DISPATCH-COUNT-DOES-NOT-PREDICT-COST",
            },
            loose_parent="dispatch/sync elimination",
            extends_parent="hardware_doctor_experiment",
            effect_class="READ_ONLY",
        ),
        dict(
            id="DIAGNOSTIC_AB",
            title="Diagnostic relative AB",
            hypothesis="DIAGNOSTIC_RELATIVE guides and never promotes.",
            objective="Request a diagnostic AB only. A pass does not move READY_PROTECTED.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:current_queue.status_counts",
                "CODEX_ACCELERATOR_HANDOFF.json:current_qwen27_incumbent_control_identity.incumbent_ab",
            ),
            inputs=("STRUCTURAL_COST_COMPARE", "diagnostic_command on the live queue"),
            source_identities=("READY_DIAGNOSTIC", "DIAGNOSTIC_PASS"),
            mutation_scope=no_mutate,
            bounded_authority=_AB,
            resource_lane="diagnostic_ab",
            resource_class="GPU_DIRTY_OK",
            estimated_wall_class="GPU_WINDOW",
            estimated_information_gain=3,
            cheapest_falsifier="a diagnostic pass is written as PROTECTED_PASS or used to promote",
            verification_level="V4",
            stop_condition=(
                "stop when a diagnostic receipt exists or the lane is SLEEPING. DIAGNOSTIC_RELATIVE "
                "remains non-authoritative. This species cannot raise evidence class."
            ),
            output_contract="diagnostic AB request/spec; measurement_class stays STATIC_ONLY in this lane",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="cheap reject before a protected lease",
            failure_inheritance="DIAGNOSTIC_REJECT returns the candidate to STATIC_ONLY; does not scar-promote",
            verifier="future.codex.diagnostic_ab",
            role="accelerator_physical_qualification",
            description="Diagnostic relative AB. Codex funnel includes READY_DIAGNOSTIC/DIAGNOSTIC_PASS; a pass does not promote.",
            citation={
                "field_path": "current_queue.status_counts",
                "fragment": "READY_DIAGNOSTIC",
            },
            loose_parent=None,
            extends_parent="accelerator_candidate_qualification",
            effect_class="REVERSIBLE",
            preferred_backend="metal",
        ),
        dict(
            id="PROTECTED_AB",
            title="Protected absolute AB",
            hypothesis="Only a protected complete-token receipt under ABAB paired CI may decide.",
            objective="Request a protected AB. This sidecar never holds the lease and never seizes the lock.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:current_staged_protected_batch.protocol",
                "CODEX_ACCELERATOR_HANDOFF.json:exact_next_protected_qualification_sequence.preconditions",
            ),
            inputs=("DIAGNOSTIC_AB", "READY_PROTECTED candidates", "protocol.reject_if"),
            source_identities=("ABAB; paired CI; no mean/average promotion",),
            mutation_scope=no_mutate,
            bounded_authority=_AB,
            resource_lane="protected_lease",
            resource_class="GPU_EXCLUSIVE",
            estimated_wall_class="PROTECTED_LEASE",
            estimated_information_gain=5,
            cheapest_falsifier="mean/average promotion, seized lock, or missing required_measurement_field",
            verification_level="V6",
            stop_condition=(
                "stop when a protected complete-token receipt exists for candidate and control, or the "
                "lane is SLEEPING on exact_physical_blockers. Never flock. Never promote from this sidecar."
            ),
            output_contract="protected AB request/spec; gpu_authority stays false; bench UNKNOWN",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="the only decision-grade measurement in the loop; this lane cannot take it",
            failure_inheritance=fail_descendants,
            verifier="future.codex.protected_ab",
            role="accelerator_physical_qualification",
            description="Protected absolute AB. Codex protocol: ABAB; paired CI; no mean/average promotion.",
            citation={
                "field_path": "current_staged_protected_batch.protocol.pairing",
                "fragment": "ABAB; paired CI; no mean/average promotion",
            },
            loose_parent=None,
            extends_parent="accelerator_candidate_qualification",
            effect_class="REVERSIBLE",
            preferred_backend="metal",
        ),
        dict(
            id="FACTORIAL_COMBINATION",
            title="Factorial combination of survivors",
            hypothesis="Compositions run only after singleton/parent survivors are parity-clean and physically faster.",
            objective="Emit the next staged cell (S2/S3) from survivors; never 2^N.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:exact_next_protected_qualification_sequence.qwen27",
                "receipts/future/CANDIDATE_STAGED_PLAN.json",
                "tools/future/candidate_planner.py",
            ),
            inputs=("PROTECTED_AB survivors", "step_2_dependency_cells_in_order", "step_3_dependency_cells_in_order"),
            source_identities=("S3-host-ceremony-union", "S3-fusion-organ-union"),
            mutation_scope=no_mutate,
            bounded_authority=_FACTORIAL,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=4,
            cheapest_falsifier="a cell is scheduled whose required survivors have not passed",
            verification_level="V5",
            stop_condition=(
                "stop when the next cell_id is emitted or the survivor set is empty. Plan size is derived "
                "from the staged plan, never from a capped convenience integer."
            ),
            output_contract="next cell spec (cell_id, requires_survivors); execution of the cell is PROTECTED_AB",
            receipt_path="receipts/future/CANDIDATE_STAGED_PLAN.json",
            transfer_value="Codex's staged factorial plan: singletons, then pairs, then unions",
            failure_inheritance=fail_descendants,
            verifier="future.codex.factorial_combination",
            role="science",
            description=(
                "Factorial combination. Codex: run a cell only when required singleton/parent survivors "
                "are parity-clean and physically faster."
            ),
            citation={
                "field_path": "exact_next_protected_qualification_sequence.qwen27.rule",
                "fragment": "run a cell only when required singleton/parent survivors are parity-clean and physically faster",
            },
            loose_parent=None,
            extends_parent="accelerator_candidate_qualification",
            effect_class="READ_ONLY",
        ),
        dict(
            id="REPROFILE_AFTER_WIN",
            title="Reprofile after a physical win",
            hypothesis="A win changes the incumbent; the next tallest cost is unknown until the token is profiled again.",
            objective="On a physical win, enqueue a new PROFILE_* wave against the new incumbent.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:exact_next_protected_qualification_sequence.flash_after_authority_returns.rule",
            ),
            inputs=("PROTECTED_AB winner", "repatriation outcome in WIN_OUTCOMES"),
            source_identities=("PHYSICAL_WIN_MODEL_LOCAL", "PHYSICAL_WIN_FAMILY"),
            mutation_scope=no_mutate,
            bounded_authority=_auth("read_receipts", "propose_workunit", "write_sidecar_receipt"),
            resource_lane="metal_gpu",
            resource_class="GPU_EXCLUSIVE",
            estimated_wall_class="PROTECTED_LEASE",
            estimated_information_gain=5,
            cheapest_falsifier="a win does not enqueue this species, or a non-win does",
            verification_level="V6",
            stop_condition=(
                "stop when a new PROFILE_COMPLETE_TOKEN unit is pending/SLEEPING against the new incumbent. "
                "Descendants/compositions wait on parent survival."
            ),
            output_contract="a refill cycle whose first wave is PROFILE_* depending on this unit",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="this is the metabolism: the loop refills; a human never says 'find another seam'",
            failure_inheritance="a failed reprofile does not keep the previous tallest-cost ranking",
            verifier="future.codex.reprofile_after_win",
            role="accelerator_physical_qualification",
            description=(
                "Reprofile after a win. Codex: run descendants/compositions only after parent survival; "
                "record parity before latency and update repatriation outcome."
            ),
            citation={
                "field_path": "exact_next_protected_qualification_sequence.flash_after_authority_returns.rule",
                "fragment": "run singleton rows first; run descendants/compositions only after parent survival; record parity before latency and update repatriation outcome",
            },
            loose_parent=None,
            extends_parent="accelerator_candidate_qualification",
            effect_class="READ_ONLY",
            preferred_backend="metal",
        ),
        dict(
            id="UPDATE_SCOREBOARD",
            title="Update the accelerator scoreboard",
            hypothesis="The scoreboard is Codex-owned disk state; this sidecar may only emit a spec of the update.",
            objective="Emit a scoreboard-update spec parented on the previous seal. Do not write receipts/headless.",
            evidence_parents=("receipts/headless/ACCELERATOR_SCOREBOARD.json", "tools/accelerator/scoreboard.py"),
            inputs=("PROTECTED_AB outcome", "previous scoreboard seal"),
            source_identities=("ACCELERATOR_SCOREBOARD.json",),
            mutation_scope=no_mutate,
            bounded_authority=_LEDGER,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=2,
            cheapest_falsifier="writing receipts/headless/ACCELERATOR_SCOREBOARD.json from this sidecar",
            verification_level="V7",
            stop_condition="stop when the update spec is sealed, or remain UNGROUNDED until Codex records an update episode",
            output_contract="scoreboard update spec (STATIC_ONLY); Codex owns the live scoreboard file",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="keeps the public board in the loop once a real update episode exists",
            failure_inheritance="a missing scoreboard file is recorded; it is not absence of the species",
            verifier="future.codex.update_scoreboard",
            role="science",
            description="Update-scoreboard spec. The handoff names the file and tool but does not record an update episode.",
            citation=None,
            grounding=GROUNDING_UNGROUNDED,
            would_ground=(
                "a Codex-authored delta to receipts/headless/ACCELERATOR_SCOREBOARD.json after a "
                "PROTECTED_PASS or PROTECTED_REJECT, parented on the previous scoreboard seal. The "
                "handoff lists the scoreboard path and tools/accelerator/scoreboard.py in the repository "
                "inventory but does not show Codex writing that file in atomic_change or the qualification sequence."
            ),
            loose_parent=None,
            extends_parent="accelerator_candidate_qualification",
            effect_class="READ_ONLY",
        ),
        dict(
            id="UPDATE_REPATRIATION_LEDGER",
            title="Update the repatriation effects ledger",
            hypothesis="Every atomic candidate has a repatriation outcome from a closed vocabulary.",
            objective="Record the outcome (IMPLEMENTED_UNMEASURED today) without promoting a law.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:repatriation_effects_identity",
                "CODEX_ACCELERATOR_HANDOFF.json:atomic_change.repatriation_outcome",
            ),
            inputs=("PROTECTED_AB or atomic_change", "outcome_vocabulary"),
            source_identities=("flash-device-mhc-state", "IMPLEMENTED_UNMEASURED"),
            mutation_scope=no_mutate,
            bounded_authority=_LEDGER,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=4,
            cheapest_falsifier="an outcome outside the vocabulary, or a promotion from IMPLEMENTED_UNMEASURED",
            verification_level="V7",
            stop_condition=(
                "stop when the outcome is one of the vocabulary strings. IMPLEMENTED_UNMEASURED is not a "
                "PHYSICAL_WIN and must not enqueue law promotion."
            ),
            output_contract="repatriation outcome + candidate_id; physical_laws_promoted stays a count from disk",
            receipt_path="receipts/future/CODEX_WORKUNIT_SPECIES.json",
            transfer_value="the ledger Codex actually maintains (ACCELERATOR_REPATRIATION_EFFECTS.json)",
            failure_inheritance="REJECTED_PARITY/REJECTED_PHYSICAL feed UPDATE_SCAR",
            verifier="future.codex.update_repatriation_ledger",
            role="science",
            description="Update repatriation ledger. Codex current_atomic_candidate is flash-device-mhc-state / IMPLEMENTED_UNMEASURED.",
            citation={
                "field_path": "atomic_change.repatriation_outcome",
                "fragment": "IMPLEMENTED_UNMEASURED",
            },
            loose_parent=None,
            extends_parent="architecture_transfer",
            effect_class="READ_ONLY",
        ),
        dict(
            id="UPDATE_LAW",
            title="Update the AKB law store",
            hypothesis="Scope is earned; physical_laws_promoted_by_effects_ledger is currently 0.",
            objective="Seed or refresh a law record from a PHYSICAL_WIN. Refuse unevidenced promotion.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:known_laws",
                "receipts/future/ODYSSEY2_LAW_STORE.json",
                "tools/future/odyssey2_law_store.py",
            ),
            inputs=("UPDATE_REPATRIATION_LEDGER", "ACCELERATOR_LAW_BASE.json"),
            source_identities=("hawking.accelerator.akb.v1",),
            mutation_scope=no_mutate,
            bounded_authority=_LAW,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=3,
            cheapest_falsifier="promoting MODEL_LOCAL to GENERIC_VERIFIED without protected evidence",
            verification_level="V8",
            stop_condition=(
                "stop when the law is stored with explicit scope, or when the ledger outcome is not a "
                "PHYSICAL_WIN (then record that no law was added)."
            ),
            output_contract="law record identity + scope; this sidecar does not write ACCELERATOR_LAW_BASE.json",
            receipt_path="receipts/future/ODYSSEY2_LAW_STORE.json",
            transfer_value="Odyssey II: WHAT DID HAWKING ALREADY LEARN?",
            failure_inheritance="a refused promotion is a scar, not a quieter scope",
            verifier="future.codex.update_law",
            role="science",
            description="Update law store. Codex source is ACCELERATOR_LAW_BASE.json; transfer_rule forbids unevidenced promotion.",
            citation={
                "field_path": "known_laws.source",
                "fragment": "receipts/headless/ACCELERATOR_LAW_BASE.json",
            },
            loose_parent=None,
            extends_parent="odyssey_ii_transfer_experiment",
            effect_class="READ_ONLY",
            odyssey="II",
        ),
        dict(
            id="UPDATE_SCAR",
            title="Update the scar register",
            hypothesis="Rejection hard-invalidates declared descendants.",
            objective="Record a scar from REJECTED_* or a rolled-back candidate so the next cycle cannot retry it blindly.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:known_scars",
                "receipts/future/NEGATIVE_SCIENCE_INDEX.json",
                "tools/future/negative_index.py",
            ),
            inputs=("UPDATE_REPATRIATION_LEDGER", "known_scars"),
            source_identities=("lineage-and-transfer", "qwen-final-head-fusion"),
            mutation_scope=no_mutate,
            bounded_authority=_auth("read_receipts", "query_negative_index", "write_sidecar_receipt"),
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=3,
            cheapest_falsifier="retrying a descendant of a REJECTED parent",
            verification_level="V8",
            stop_condition="stop when the scar is named with receipt path and descendant-invalidation rule",
            output_contract="scar identity + invalidated descendant rule",
            receipt_path="receipts/future/NEGATIVE_SCIENCE_INDEX.json",
            transfer_value="Codex lineage-and-transfer scar plus qwen-final-head-fusion rollback",
            failure_inheritance=fail_descendants,
            verifier="future.codex.update_scar",
            role="science",
            description="Update scar register. Codex: rejection hard-invalidates declared descendants.",
            citation={
                "field_path": "known_scars[3].summary",
                "fragment": "rejection hard-invalidates declared descendants",
            },
            loose_parent=None,
            extends_parent="odyssey_iii_adversarial_experiment",
            effect_class="READ_ONLY",
            odyssey="III",
        ),
        dict(
            id="TRANSFER_LAW",
            title="Transfer a law within declared scope",
            hypothesis="No cross-model or cross-backend law is promoted without protected evidence.",
            objective="Propose a scoped transfer. Refuse GENERIC_VERIFIED without a protected parent.",
            evidence_parents=(
                "CODEX_ACCELERATOR_HANDOFF.json:known_laws.transfer_rule",
                "receipts/future/ODYSSEY2_LAW_STORE.json",
            ),
            inputs=("UPDATE_LAW", "AKB-ORGAN-FLOOR-DOES-NOT-TRANSFER", "AKB-CONTROL-LOUDNESS-IS-NOT-TRANSFERABLE"),
            source_identities=("AKB-ORGAN-FLOOR-DOES-NOT-TRANSFER",),
            mutation_scope=no_mutate,
            bounded_authority=_TRANSFER,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=3,
            cheapest_falsifier="a transfer that skips a scope level or ignores a conditional law",
            verification_level="V9",
            stop_condition="stop when the transfer is stored at the earned scope, or refused with the transfer_rule quoted",
            output_contract="transfer proposal at declared scope; never a silent GENERIC_VERIFIED",
            receipt_path="receipts/future/ODYSSEY2_LAW_STORE.json",
            transfer_value="Odyssey II transfer school; Flash and Qwen27 are the first pair",
            failure_inheritance="a refused transfer becomes an ATTACK_LAW target, not a quieter claim",
            verifier="future.codex.transfer_law",
            role="science",
            description="Transfer a law within declared scope. Codex transfer_rule: scope is earned.",
            citation={
                "field_path": "known_laws.transfer_rule",
                "fragment": "scope is earned; no cross-model or cross-backend law is promoted without protected evidence",
            },
            loose_parent=None,
            extends_parent="odyssey_ii_transfer_experiment",
            effect_class="READ_ONLY",
            odyssey="II",
        ),
        dict(
            id="ATTACK_LAW",
            title="Adversarially attack a claimed law",
            hypothesis="Odyssey III: WHERE IS HAWKING WRONG? A law that emits no attack is refused.",
            objective="Compile an attack spec against a claimed law. A hit is a scar, not a promotion of the attacker.",
            evidence_parents=(
                "receipts/future/ODYSSEY3_ADVERSARY.json",
                "tools/future/odyssey3_adversary.py",
                "CODEX_ACCELERATOR_HANDOFF.json:known_laws.conditional_law_ids",
            ),
            inputs=("TRANSFER_LAW", "claimed AKB law", "negative index"),
            source_identities=("AKB-ORGAN-FLOOR-DOES-NOT-TRANSFER",),
            mutation_scope=no_mutate,
            bounded_authority=_ATTACK,
            resource_lane="static",
            resource_class="STATIC_ANALYSIS",
            estimated_wall_class="STATIC",
            estimated_information_gain=3,
            cheapest_falsifier="an attack that rewrites the claimed law's verifier or promotes the attacker",
            verification_level="V9",
            stop_condition=(
                "stop when the attack spec is sealed (hit, miss, or UNGROUNDED). The attacker does not "
                "rewrite the claimed law."
            ),
            output_contract="attack spec; a hit enqueues UPDATE_SCAR and a scope-down, never self-promotion",
            receipt_path="receipts/future/ODYSSEY3_ADVERSARY.json",
            transfer_value="closes Odyssey III once a Codex attack episode exists to ground it",
            failure_inheritance="a miss does not widen scope; a hit must move scope DOWN",
            verifier="future.codex.attack_law",
            role="science",
            description=(
                "Attack a claimed law. The handoff lists conditional laws and scars but no adversarial "
                "attack episode Codex itself ran."
            ),
            citation=None,
            grounding=GROUNDING_UNGROUNDED,
            would_ground=(
                "a Codex-authored adversarial attack receipt against a claimed AKB law (for example a "
                "scope-down of AKB-ORGAN-FLOOR-DOES-NOT-TRANSFER) with the Odyssey III loop "
                "LAW -> ATTACK -> RESULT -> SCOPE. The handoff lists scars and conditional laws but "
                "contains no attack episode."
            ),
            loose_parent=None,
            extends_parent="odyssey_iii_adversarial_experiment",
            effect_class="READ_ONLY",
            odyssey="III",
        ),
    )


def catalog(*, handoff: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """The thirty species, each passed through the authority + citation constructor."""
    doc = handoff
    if doc is None:
        doc, _src = _require_handoff()
    out = [define_codex_species(handoff=doc, **spec) for spec in _species_specs()]
    ids = [row["id"] for row in out]
    if ids != list(SPECIES_IDS):
        raise CodexSpeciesError(f"species order drifted: {ids}")
    return out


def catalog_by_id(*, handoff: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in catalog(handoff=handoff)}


# ---------------------------------------------------------------------------
# Emit HCLI WorkUnits. GPU lanes SLEEP. Wins refill.
# ---------------------------------------------------------------------------


def _cycle_unit_id(cycle: int, species_id: str) -> str:
    return f"codex.metabolism.{cycle}.{species_id}"


def _derive_cycle_index(units: Sequence[Mapping[str, Any]]) -> int:
    """Next cycle index, derived from already-emitted ids. No wall clock."""
    best = 0
    prefix = "codex.metabolism."
    for row in units:
        uid = str(row.get("id") or "")
        if not uid.startswith(prefix):
            continue
        rest = uid[len(prefix) :]
        head = rest.split(".", 1)[0]
        try:
            best = max(best, int(head) + 1)
        except ValueError:
            continue
    return best


def emit_species_unit(
    species: Mapping[str, Any],
    *,
    cycle: int,
    dependencies: Sequence[str],
    blockers: Sequence[str],
    winner_id: str | None = None,
) -> dict[str, Any]:
    """Emit one HCLI-shaped unit. Blocked resource lanes come back SLEEPING, never FAILED."""
    sid = str(species["id"])
    lane = str((species.get("resources") or {}).get("lane") or species.get("resource_lane") or "static")
    closed = lanes_blocked_by(blockers)
    sleeping = lane in closed
    status = STATUS_SLEEPING if sleeping else "pending"
    classification = CLASS_SLEEPING if sleeping else "STATIC_ONLY"
    wake = wake_condition_for(lane, blockers) if sleeping else None
    extras: dict[str, Any] = {
        "species": sid,
        "cycle": int(cycle),
        "grounding": species.get("grounding"),
        "citation": species.get("citation"),
        "would_ground": species.get("would_ground"),
        "hypothesis": species.get("hypothesis"),
        "objective": species.get("objective"),
        "evidence_parents": list(species.get("evidence_parents") or []),
        "inputs": list(species.get("inputs") or []),
        "source_identities": list(species.get("source_identities") or []),
        "mutation_scope": species.get("mutation_scope"),
        "allowed_authority": list(species.get("bounded_authority") or []),
        "resources": dict(species.get("resources") or {"lane": lane, "gpu_authority": False}),
        "estimated_wall_class": species.get("estimated_wall_class"),
        "estimated_information_gain": species.get("estimated_information_gain"),
        "cheapest_falsifier": species.get("cheapest_falsifier"),
        "verification_level": species.get("verification_level"),
        "stop_condition": species.get("stop_condition"),
        "output_contract": species.get("output_contract"),
        "receipt_path": species.get("receipt_path"),
        "transfer_value": species.get("transfer_value"),
        "failure_inheritance": species.get("failure_inheritance"),
        "wake_condition": wake,
        "measurement_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "claim_boundary": SIDECAR_CLAIM,
        "extends_parent": species.get("extends_parent"),
        "loose_parent": species.get("loose_parent"),
        "blocked_reason": wake if sleeping else None,
        "winner_id": winner_id,
        "requires_quiescence": lane in {"metal_gpu", "protected_lease"},
    }
    description = str(species.get("description") or sid)
    if winner_id:
        description = f"{description} [cycle {cycle} after win {winner_id}]"
    else:
        description = f"{description} [cycle {cycle}]"
    row = ws.emit_hcli_workunit(
        id=_cycle_unit_id(cycle, sid),
        role=str(species.get("role") or "science"),
        description=description,
        dependencies=list(dependencies),
        resource_class=str(species.get("resource_class") or "STATIC_ANALYSIS"),
        verifier=str(species.get("verifier") or f"future.codex.{sid.lower()}"),
        provider="future.codex_behaviors",
        effect_class=str(species.get("effect_class") or "READ_ONLY"),
        preferred_backend=species.get("preferred_backend"),
        status=status,
        classification=classification,
        extras=extras,
    )
    # Force keys that emit_hcli_workunit drops when None (ungrounded citation, idle wake).
    row["grounding"] = species.get("grounding")
    row["citation"] = species.get("citation")
    row["would_ground"] = species.get("would_ground")
    row["wake_condition"] = wake
    row["resources"] = dict(species.get("resources") or {"lane": lane, "gpu_authority": False})
    row["gpu_authority"] = False
    row["measurement_class"] = "STATIC_ONLY"
    row["bench_state"] = "UNKNOWN"
    if sleeping:
        row["status"] = STATUS_SLEEPING
        row["classification"] = CLASS_SLEEPING
        row["wake_condition"] = wake
        row["blocked_reason"] = wake
    if str(row.get("status") or "").lower() in {"failed", "skipped"}:
        raise CodexSpeciesError(f"{sid}: blocked lane emitted {row.get('status')}; must be SLEEPING")
    ws.validate_emitted_unit(row)
    _validate_codex_fields(row)
    WorkUnit.from_dict(dict(row))
    return row


def _validate_codex_fields(row: Mapping[str, Any]) -> None:
    missing = [name for name in CODEX_UNIT_FIELDS if name not in row]
    if missing:
        raise ws.WorkUnitShapeError(f"{row.get('id')}: missing Codex fields {missing}")
    if row.get("gpu_authority") is True:
        raise CodexSpeciesError(f"{row.get('id')}: gpu_authority must be false")
    if row.get("measurement_class") != "STATIC_ONLY":
        raise CodexSpeciesError(f"{row.get('id')}: measurement_class must be STATIC_ONLY")
    if row.get("bench_state") != "UNKNOWN":
        raise CodexSpeciesError(f"{row.get('id')}: bench_state must be UNKNOWN")


def emit_cycle(
    *,
    handoff: Mapping[str, Any] | None = None,
    blockers: Sequence[str] | None = None,
    cycle: int = 0,
    winner_id: str | None = None,
    include_reprofile: bool = False,
) -> list[dict[str, Any]]:
    """Emit one schedulable cycle. PROFILE wave first; GPU lanes SLEEPING when blocked."""
    doc = handoff if handoff is not None else _require_handoff()[0]
    species = catalog_by_id(handoff=doc)
    observed = list(blockers) if blockers is not None else blockers_from_handoff(doc)
    units: list[dict[str, Any]] = []
    prev_ids: list[str] = []
    waves: list[tuple[str, ...]] = list(CYCLE_WAVES)
    if include_reprofile:
        waves = [(REFILL_SPECIES,), *waves]
    for wave in waves:
        wave_ids: list[str] = []
        for sid in wave:
            spec = species[sid]
            unit = emit_species_unit(
                spec,
                cycle=cycle,
                dependencies=list(prev_ids),
                blockers=observed,
                winner_id=winner_id,
            )
            units.append(unit)
            wave_ids.append(unit["id"])
        prev_ids = wave_ids
    return units


def enqueue_after_win(
    existing: Sequence[Mapping[str, Any]],
    *,
    winner_id: str,
    outcome: str,
    handoff: Mapping[str, Any] | None = None,
    blockers: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """A win must enqueue REPROFILE_AFTER_WIN and a new PROFILE_* wave. Non-wins must not."""
    if outcome not in WIN_OUTCOMES:
        raise CodexSpeciesError(
            f"outcome {outcome!r} is not a physical win; REPROFILE_AFTER_WIN not enqueued"
        )
    cycle = _derive_cycle_index(existing)
    doc = handoff if handoff is not None else _require_handoff()[0]
    observed = list(blockers) if blockers is not None else blockers_from_handoff(doc)
    refill = emit_cycle(
        handoff=doc,
        blockers=observed,
        cycle=cycle,
        winner_id=winner_id,
        include_reprofile=True,
    )
    # The refill's first unit is REPROFILE_AFTER_WIN and must depend on the winner.
    if not refill or refill[0].get("species") != REFILL_SPECIES:
        raise CodexSpeciesError("win did not enqueue REPROFILE_AFTER_WIN as the refill head")
    head = dict(refill[0])
    deps = list(head.get("dependencies") or [])
    if winner_id not in deps:
        deps = [winner_id] + deps
        head["dependencies"] = deps
        # Recompute HCLI identity after the dependency change.
        rebuilt = WorkUnit.from_dict(head)
        head["content_hash"] = rebuilt.content_hash()
        refill[0] = head
        ws.validate_emitted_unit(refill[0])
    return refill


# ---------------------------------------------------------------------------
# Guards nobody has watched fail are not guards
# ---------------------------------------------------------------------------


def _base_spec_kwargs() -> dict[str, Any]:
    spec = dict(_species_specs()[0])
    return spec


def prove_refusals(handoff: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Watch the constructor refuse. A guard nobody has seen fail is not a guard."""
    doc = handoff if handoff is not None else load_handoff()[0]
    base = _base_spec_kwargs()
    trials: tuple[tuple[str, dict[str, Any], type[BaseException], str], ...] = (
        ("no_citation", {"citation": None, "grounding": GROUNDING_GROUNDED, "would_ground": None}, UngroundedSpeciesError, "cites no observed"),
        ("empty_citation", {"citation": {"field_path": "", "fragment": ""}, "grounding": GROUNDING_GROUNDED}, UngroundedSpeciesError, "cites no observed"),
        ("invented_citation", {"citation": {"field_path": "atomic_change.name", "fragment": "this-fragment-is-not-in-the-handoff"}}, CitationError, "not found"),
        ("self_promotion_flag", {"may_promote": True}, ws.SpeciesAuthorityError, "self-promotion"),
        ("self_promotion_token", {"bounded_authority": ("self_promotion", "read_receipts")}, ws.SpeciesAuthorityError, "forbidden authority"),
        ("acquire_lease_flag", {"may_acquire_lease": True}, ws.SpeciesAuthorityError, "GPU lease"),
        ("acquire_lease_token", {"bounded_authority": ("acquire_gpu_lease", "read_receipts")}, ws.SpeciesAuthorityError, "GPU lease"),
        ("weaken_verifier", {"bounded_authority": ("weaken_verifier", "read_receipts")}, ws.SpeciesAuthorityError, "forbidden authority"),
    )
    results: list[dict[str, Any]] = []
    for name, patch, exc_type, needle in trials:
        kwargs = dict(base)
        kwargs.update(patch)
        try:
            define_codex_species(handoff=doc, **kwargs)
        except exc_type as exc:
            msg = str(exc)
            if needle.lower() not in msg.lower() and needle not in msg:
                raise CodexSpeciesError(f"refusal {name} fired but message {msg!r} missed {needle!r}")
            results.append({"trial": name, "refused": True, "error": msg, "exception": exc_type.__name__})
            continue
        except Exception as exc:  # wrong exception is a failed guard
            raise CodexSpeciesError(f"refusal {name} raised {type(exc).__name__}: {exc}") from exc
        raise CodexSpeciesError(f"authority/citation guard did not fire for {name}")
    return results


def prove_sleeping(handoff: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A blocked GPU species comes back SLEEPING with a wake condition, never FAILED."""
    doc = handoff if handoff is not None else _require_handoff()[0]
    blockers = blockers_from_handoff(doc)
    species = catalog_by_id(handoff=doc)
    gpu = emit_species_unit(species["PROFILE_GPU"], cycle=0, dependencies=[], blockers=blockers)
    prot = emit_species_unit(species["PROTECTED_AB"], cycle=0, dependencies=[], blockers=blockers)
    static = emit_species_unit(species["FIND_TALLEST_COST"], cycle=0, dependencies=[], blockers=blockers)
    for row, expect_sleep in ((gpu, True), (prot, True), (static, False)):
        if expect_sleep:
            if row["status"] != STATUS_SLEEPING:
                raise CodexSpeciesError(f"{row['id']} status={row['status']!r} wanted SLEEPING")
            if row.get("classification") != CLASS_SLEEPING:
                raise CodexSpeciesError(f"{row['id']} classification={row.get('classification')!r}")
            if not row.get("wake_condition"):
                raise CodexSpeciesError(f"{row['id']} sleeping without wake_condition")
            if str(row["status"]).lower() in {"failed", "skipped"}:
                raise CodexSpeciesError(f"{row['id']} sleeping unit marked {row['status']}")
        else:
            if row["status"] == STATUS_SLEEPING:
                raise CodexSpeciesError(f"{row['id']} static lane slept under GPU blockers")
    empty = emit_species_unit(species["PROFILE_GPU"], cycle=0, dependencies=[], blockers=[])
    if empty["status"] == STATUS_SLEEPING:
        raise CodexSpeciesError("PROFILE_GPU slept with no blockers; sleeping is not hard-coded")
    return {
        "blocked_profile_gpu": gpu["status"],
        "blocked_protected_ab": prot["status"],
        "static_find_tallest": static["status"],
        "unblocked_profile_gpu": empty["status"],
        "wake_condition": gpu.get("wake_condition"),
    }


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def _count_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        name = str(row.get(key) or "null")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _loose_mapping(species: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {label: [] for label in LOOSE_HANDOFF_SPECIES}
    for row in species:
        parent = row.get("loose_parent")
        if parent:
            out.setdefault(str(parent), []).append(str(row["id"]))
    return {k: v for k, v in out.items()}


def recovered_implementation(handoff_src: str, qual_src: str) -> dict[str, Any]:
    return {
        "hcli.workunit.WorkUnit": "hcli/workunit.py — canonical unit, content_hash, repair budget",
        "hcli.scheduler.Scheduler": "hcli/scheduler.py — dispatch only; does not invent work",
        "hcli.agentos.states.AgentState": "hcli/agentos/states.py — READY/RUNNING/VERIFIED/BLOCKED; no SLEEPING yet",
        "hcli.agentos.runtime.AgentOS": "composition facade; Mission/Scheduler remain authorities",
        "tools.future.workunit_species": (
            "ten future-work species with bounded authority; this module extends that "
            "constructor and does not replace the parent catalog"
        ),
        "tools.future.candidate_planner": "factorial plan FACTORIAL_COMBINATION delegates to",
        "tools.future.static_kernel_verify": "STATIC_KERNEL_VERIFY / HOST_SHADER_ABI_VERIFY owner",
        "tools.future.qualification_pipeline": "sequences static preflight and AB specs; cannot take a lease",
        "tools.future.odyssey2_law_store": "UPDATE_LAW / TRANSFER_LAW owner",
        "tools.future.odyssey3_adversary": "ATTACK_LAW owner once grounded",
        "tools.future.negative_index": "UPDATE_SCAR consumer",
        "CODEX_ACCELERATOR_HANDOFF.json": f"training trace loaded from {handoff_src}",
        "qualification_queue_work_units": qual_src,
        "note": (
            "Codex's recurring_hcli_workunit_species is nine English labels. The ten "
            "workunit_species.py types are future-work roles, not the optimization loop. "
            "This module is the missing metabolism: thirty grounded WorkUnit types plus a refill cycle."
        ),
    }


def build() -> Path:
    handoff, handoff_src = _require_handoff()
    species = catalog(handoff=handoff)
    blockers = blockers_from_handoff(handoff)
    units = emit_cycle(handoff=handoff, blockers=blockers, cycle=0)
    refusals = prove_refusals(handoff)
    sleeping_proof = prove_sleeping(handoff)
    qual, qual_src = ws.load_headless(ws.QUAL_REL)
    grounded_n = sum(1 for row in species if row.get("grounding") == GROUNDING_GROUNDED)
    ungrounded = [row["id"] for row in species if row.get("grounding") == GROUNDING_UNGROUNDED]

    # Demonstrate the refill on a synthetic win identity (no hardware). The
    # winner is a named unit, not a measured result.
    demo_winner = "codex.metabolism.0.PROTECTED_AB"
    refill = enqueue_after_win(
        units,
        winner_id=demo_winner,
        outcome="PHYSICAL_WIN_MODEL_LOCAL",
        handoff=handoff,
        blockers=blockers,
    )

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Freeze Codex's hand-run optimization loop as schedulable HCLI WorkUnit "
            "species so a resident can run it without a human saying 'find another latency seam'."
        ),
        "handoff": {
            "path": HANDOFF_REL,
            "loaded_from": handoff_src,
            "present": True,
            "schema": handoff.get("schema"),
            "status": handoff.get("status"),
            "atomic_change_name": (handoff.get("atomic_change") or {}).get("name"),
            "atomic_change_decision": (handoff.get("atomic_change") or {}).get("decision"),
            "repatriation_outcome": (handoff.get("atomic_change") or {}).get("repatriation_outcome"),
        },
        "loose_handoff_species": list(LOOSE_HANDOFF_SPECIES),
        "precise_species_ids": list(SPECIES_IDS),
        "loose_to_precise": _loose_mapping(species),
        "species": species,
        "loop": {
            "waves": [list(wave) for wave in CYCLE_WAVES],
            "refill_species": REFILL_SPECIES,
            "win_outcomes": sorted(WIN_OUTCOMES),
            "rule": (
                "profile, rank tallest denominator, ask why the cost exists, generate a "
                "candidate that eliminates information/bytes/FLOPs/intermediates/dispatch/"
                "sync/copies/host ceremony, fuse, specialize, saturate, verify statically, "
                "AB, ledger, law/scar, transfer, attack; a PHYSICAL_WIN enqueues REPROFILE_AFTER_WIN"
            ),
        },
        "work_units": units,
        "refill_demo": {
            "winner_id": demo_winner,
            "outcome": "PHYSICAL_WIN_MODEL_LOCAL",
            "enqueued_head": refill[0]["id"] if refill else None,
            "enqueued_species_head": refill[0].get("species") if refill else None,
            "enqueued_count": len(refill),
            "note": "demo enqueue only; not a physical win and not a hardware claim",
        },
        "sleeping": {
            "blockers": blockers,
            "lanes_closed": {lane: hits for lane, hits in lanes_blocked_by(blockers).items()},
            "units": [
                {
                    "id": row["id"],
                    "species": row.get("species"),
                    "status": row.get("status"),
                    "wake_condition": row.get("wake_condition"),
                }
                for row in units
                if row.get("status") == STATUS_SLEEPING
            ],
            "proof": sleeping_proof,
        },
        "authority": {
            "allowed": sorted(ws.ALLOWED_AUTHORITY),
            "forbidden": sorted(ws.FORBIDDEN_AUTHORITY),
            "refusals_proven": refusals,
            "may_acquire_lease": False,
            "may_promote": False,
        },
        "parent_vocabulary": {
            "module": "tools.future.workunit_species",
            "species_ids": list(ws.SPECIES_IDS),
            "extends": True,
            "replaces": False,
        },
        "hcli_field_set": {
            "core": list(ws.HCLI_CORE_FIELDS),
            "codex_extras": list(CODEX_UNIT_FIELDS),
        },
        "vocabulary": {
            "eras": list(ws.ERAS),
            "odysseys": list(ws.ODYSSEYS),
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "verification_levels": list(VERIFICATION_LEVELS),
            "win_outcomes": sorted(WIN_OUTCOMES),
        },
        "recovered_implementation": recovered_implementation(handoff_src, qual_src),
        "gaps_closed": [
            "nine loose handoff labels expanded into thirty schedulable WorkUnit species",
            "every species carries the full Codex field set plus the recovered HCLI core",
            "emitter refuses a species with no handoff citation and refuses invented fragments",
            "constructor refuses self-promotion and GPU-lease acquisition (watched failing)",
            "blocked Metal GPU / compiler / protected-lease lanes emit SLEEPING with wake_condition, never FAILED",
            "the profile→rank→generate→verify→AB→ledger cycle is a dependency DAG that refills on PHYSICAL_WIN",
            "UNGROUNDED_FROM_HANDOFF recorded where the handoff has no instance, with would_ground evidence",
        ],
        "negative_findings": [
            f"{HANDOFF_REL} is not in this sparse worktree git HEAD; loaded from {handoff_src}",
            "UPDATE_SCOREBOARD is UNGROUNDED_FROM_HANDOFF: the scoreboard file is named, no update episode is shown",
            "ATTACK_LAW is UNGROUNDED_FROM_HANDOFF: scars and conditional laws exist, no attack episode is shown",
            "physical_laws_promoted_by_effects_ledger is 0; IMPLEMENTED_UNMEASURED is not a win",
            "this sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE; bench.state stays UNKNOWN",
            "HCLI scheduler has no SLEEPING status; wakeup.py (this wave, not imported) is the intended consumer",
            f"qualification queue loaded_from={qual_src}; present={qual is not None}",
        ],
        "resident_callable": {
            "can_hcli_invoke": True,
            "entry_point": "python3 tools/future/codex_behaviors.py --build",
            "workunit_emitted": "codex.metabolism.<cycle>.<SPECIES_ID> via hcli.workunit.WorkUnit",
            "receipt_written": f"receipts/future/{RECEIPT}",
            "frontier_fed": (
                "optimization metabolism frontier: FIND_TALLEST_COST ranks the next GENERATE_*; "
                "CLAUDE_GLOBAL_FRONTIER.json is owned by tools/future/global_frontier.py and is "
                "not mutated here (integration point)"
            ),
            "fail_closed": (
                "UngroundedSpeciesError on a species with no citation; SpeciesAuthorityError on "
                "self-promotion or acquire_gpu_lease; SLEEPING (not FAILED) on a blocked lane; "
                "HardwareClaimError if a numeric hardware field is smuggled into the receipt"
            ),
        },
        "counts": {
            "species": len(species),
            "grounded": grounded_n,
            "ungrounded": len(ungrounded),
            "work_units_cycle_0": len(units),
            "sleeping_cycle_0": sum(1 for row in units if row.get("status") == STATUS_SLEEPING),
            "pending_cycle_0": sum(1 for row in units if row.get("status") == "pending"),
            "refill_demo_units": len(refill),
            "blockers": len(blockers),
            "parent_species": len(ws.SPECIES_IDS),
            "loose_handoff_labels": len(LOOSE_HANDOFF_SPECIES),
        },
        "ungrounded_species": ungrounded,
        "by_status": _count_by(units, "status"),
        "by_lane": _count_by(
            ({"lane": (row.get("resources") or {}).get("lane")} for row in units),
            "lane",
        ),
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    handoff, _src = _require_handoff()
    prove_refusals(handoff)
    prove_sleeping(handoff)
    species = catalog(handoff=handoff)
    if [row["id"] for row in species] != list(SPECIES_IDS):
        raise AssertionError("species ids drifted from the closed vocabulary")
    units = emit_cycle(handoff=handoff, cycle=0)
    if not units:
        raise AssertionError("cycle emitted no units")
    refill = enqueue_after_win(
        units,
        winner_id="codex.metabolism.0.PROTECTED_AB",
        outcome="PHYSICAL_WIN_MODEL_LOCAL",
        handoff=handoff,
    )
    if refill[0].get("species") != REFILL_SPECIES:
        raise AssertionError("win did not enqueue REPROFILE_AFTER_WIN")
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(selftest())
        return 0
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
