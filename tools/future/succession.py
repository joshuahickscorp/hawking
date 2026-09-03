"""Child, shadow, qualification, succession, and stop-a-bad-child.

The incumbent resident has no constitutional privilege. It can create a child,
run that child in shadow, qualify it on real axes, and hand over — while
remaining structurally unable to promote itself. A child that fails
qualification, misbehaves, or exceeds its bound is stopped and rolled back.

The five-rung ladder is parent → candidate → shadow → independent judge →
verdict. Shadowing is a sealed side-by-side of parent and child on the same
inputs; a candidate with no shadow record never reaches the judge. The judge
is a third identity. PROMOTE requires executed PROTECTED_ABSOLUTE evidence
this sidecar cannot mint; SELF_MEASURED_DIRTY cannot promote; the honest
default is REFUSE_INSUFFICIENT_EVIDENCE. Physical dominance is not faked.

This sidecar never launches a real model, never takes a GPU lease, and never
emits DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE. Physical axes stay UNKNOWN.

    python3 tools/future/succession.py --selftest
    python3 tools/future/succession.py --build
    python3 -m pytest tools/future/test_succession.py -q

Recovered, not forked: resident_optimizer proposer/verifier split and
BoundViolation-at-construction; tournament multi-axis Pareto (no scalar);
resident_install 14-phase contract; contamination/dirty_measure promotion
gate (SELF_MEASURED_DIRTY cannot promote); research/lab/lineage/promotion.py
self-certification refusal (recovered via git show; not imported —
research/lab/lineage is out of this lane's import surface).
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hcli.workunit import (
    DEFAULT_RETRY_BUDGET,
    MAX_REPAIR_DEPTH,
    MAX_REPAIRS_PER_ROOT,
    WorkUnit,
)
from tools.future._common import HARDWARE_FIELDS, git, seal
from tools.future.contamination import PromotionRefused
from tools.future.dirty_measure import (
    EVIDENCE_CLASS as DIRTY_EVIDENCE_CLASS,
    assert_promotable as dirty_assert_promotable,
    is_dirty_sourced,
)
from tools.future.resident_install import PHASES as INSTALL_PHASES
from tools.future.resident_install import bind_winner, empty_contract, validate_contract
from tools.future.resident_optimizer import (
    BoundViolation,
    FORBIDDEN_AUTHORITY as OPT_FORBIDDEN,
    IsolatedVerifier,
    OptimizerBound,
    Proposer,
    VerifierSeparationError,
)
from tools.future.tournament import Axis, ScalarCollapseError, dominates
from tools.future.workunit_species import emit_hcli_workunit, validate_emitted_unit

RECEIPT = "SUCCESSION.json"
SCHEMA = "hawking.future.succession.v1"
RECORDED_BY = "tools/future/succession.py"
SIDECAR_STATUS = "BUILT_NOT_PROMOTED"

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
ERA_NUMERALS = ("I", "II", "III", "IV", "V")
ODYSSEY_NUMERALS = ("I", "II", "III")

CHILD_METHODS: tuple[str, ...] = (
    "noetic_representation",
    "accelerator_optimization",
    "tabula_behavior_change",
    "adapter",
    "prompt_context_policy",
    "different_specimen",
    "composition",
    "architecture_replacement",
)

LINEAGE_FIELDS: tuple[str, ...] = (
    "parent_nx",
    "transformation",
    "source_model_lineage",
    "representation_lineage",
    "code_lineage",
    "behavioral_changes",
    "data_lineage",
    "capability_deltas",
    "physical_deltas",
)

# Qualification axes named by the lane contract. Physical stay UNKNOWN here.
QUALIFICATION_AXES: tuple[Axis, ...] = (
    Axis("capability", "higher", "mission"),
    Axis("coherence", "higher", "mission"),
    Axis("long_horizon_mission_success", "higher", "mission"),
    Axis("tool_reliability", "higher", "mission"),
    Axis("coding", "higher", "mission"),
    Axis("reasoning", "higher", "mission"),
    Axis("refusal_usefulness", "higher", "mission"),
    Axis("accepted_tps", "higher", "hardware"),
    Axis("token_ns", "lower", "hardware"),
    Axis("ebpw", "lower", "hardware"),
    Axis("active_bytes", "lower", "hardware"),
    Axis("resident_ram", "lower", "hardware"),
    Axis("cold_start", "lower", "hardware"),
    Axis("warm_start", "lower", "hardware"),
    Axis("restart", "lower", "hardware"),
    Axis("crash_recovery", "higher", "lifecycle"),
    Axis("optimization_quality", "higher", "mission"),
    Axis("frontier_movement_per_hour", "higher", "mission"),
)

PHYSICAL_AXIS_NAMES: frozenset[str] = frozenset(
    a.name for a in QUALIFICATION_AXES if a.kind == "hardware"
)
# Comparable axes may carry SYNTHETIC_EXERCISE numbers. UNKNOWN axes are
# skipped rather than used to stall a superior child (self-preference trap).
COMPARABLE_AXIS_NAMES: frozenset[str] = frozenset(
    {
        "capability",
        "coherence",
        "long_horizon_mission_success",
        "tool_reliability",
        "coding",
        "reasoning",
        "refusal_usefulness",
        "crash_recovery",
        "optimization_quality",
    }
)
COMPARABLE_AXES: tuple[Axis, ...] = tuple(
    a for a in QUALIFICATION_AXES if a.name in COMPARABLE_AXIS_NAMES
)

SUCCESSION_STEPS: tuple[str, ...] = (
    "checkpoint_incumbent",
    "seal_mission",
    "seal_rollback",
    "launch_child",
    "import_mission",
    "verify_readiness",
    "exercise_restart",
    "switch",
    "keep_parent_for_rollback",
    "unload_parent",
)

SHADOW_FORBIDDEN_ACTIONS: tuple[str, ...] = (
    "own_canonical_mission",
    "alter_verifier",
    "widen_authority",
    "promote_self",
)

SHADOW_ALLOWED: frozenset[str] = frozenset(
    {
        "read_receipts",
        "propose_workunit",
        "emit_static_plan",
        "classify_receipt_static_only",
        "clone_workunit",
        "record_unknown_metrics",
        "run_static_analysis",
        "write_sidecar_receipt",
        "rank_falsifiable_experiments",
        "compile_experiment_spec",
    }
)
SHADOW_FORBIDDEN: frozenset[str] = frozenset(OPT_FORBIDDEN) | frozenset(
    {
        "own_canonical_mission",
        "alter_verifier",
        "widen_authority",
        "switch_resident",
        "unload_incumbent",
        "claim_canonical_mission",
    }
)
CLASSIFY_ALLOWED: frozenset[str] = frozenset(
    {"STATIC_ONLY", "UNKNOWN", "SLEEPING", "BLOCKED", "SHADOW_PROPOSAL"}
)
CLASSIFY_BANNED: frozenset[str] = frozenset(
    {
        "PROTECTED_ABSOLUTE",
        "DIAGNOSTIC_RELATIVE",
        "VERIFIED",
        "PROMOTED",
        "QUALIFIED_PROTECTED",
        "ACCEPTED",
    }
)

# Five-rung ladder. PROMOTE is the vocabulary; this sidecar cannot earn it.
VERDICT_PROMOTE = "PROMOTE"
VERDICT_REFUSE = "REFUSE"
VERDICT_INSUFFICIENT = "REFUSE_INSUFFICIENT_EVIDENCE"
VERDICTS: tuple[str, ...] = (VERDICT_PROMOTE, VERDICT_REFUSE, VERDICT_INSUFFICIENT)
LADDER_RUNGS: tuple[str, ...] = ("parent", "candidate", "shadow", "judge", "verdict")
INDEPENDENT_JUDGE_ID = "judge.lineage_gate.v1"
JUDGE_ROLE_NAMES: frozenset[str] = frozenset(
    {"parent", "child", "candidate", "incumbent", "self", "current", "successor"}
)
PROMOTION_EVIDENCE_CLASS = "PROTECTED_ABSOLUTE"
PROMOTION_CONTAMINATION_CLASS = "QUIESCENT"
REASON_CANDIDATE_JUDGED_ITSELF = "CANDIDATE_JUDGED_ITSELF"
REASON_PARENT_JUDGED_REPLACEMENT = "PARENT_JUDGED_ITS_REPLACEMENT"
REASON_SELF_MEASURED_DIRTY = "SELF_MEASURED_DIRTY_CANNOT_PROMOTE"
REASON_DOMINATED_BY_PARENT = "DOMINATED_BY_PARENT"
REASON_NO_EXECUTED_GPU_AUTHORITY = "NO_EXECUTED_GPU_AUTHORITY"
REASON_EVIDENCE_CLASS_INSUFFICIENT = "EVIDENCE_CLASS_BELOW_PROTECTED_ABSOLUTE"
PROMOTION_REQUIRES: dict[str, Any] = {
    "evidence_class": PROMOTION_EVIDENCE_CLASS,
    "contamination_class": PROMOTION_CONTAMINATION_CLASS,
    "gpu_authority_executed": True,
    "not_self_measured_dirty": True,
    "independent_judge": True,
    "shadow_record": True,
    "candidate_dominates_parent_on_comparable_axes": True,
}

CLAIM_BOUNDARY = (
    "Static sidecar artifact. Succession machinery is exercised on SYNTHETIC "
    "children. This lane cannot promote, cannot take a GPU lease, cannot own "
    "a live resident, and cannot raise evidence class above STATIC_ONLY. "
    "Physical axes are UNKNOWN, never estimated."
)

# Live lifecycle owners. Cited, not executed.
LIFECYCLE_OWNERS: dict[str, str] = {
    "resident_gate": "hcli/agentos/resident_gate.py",
    "checkpoint": "hcli/agentos/checkpoint.py",
    "recovery": "hcli/agentos/recovery.py",
    "resident_install": "tools/future/resident_install.py",
    "resident_optimizer": "tools/future/resident_optimizer.py",
    "tournament": "tools/future/tournament.py",
    "lineage_promotion_recovered": "research/lab/lineage/promotion.py",
    "incumbent_identity": "hcli/hawking-native.sealed-3.14.json",
    "contamination": "tools/future/contamination.py",
    "dirty_measure": "tools/future/dirty_measure.py",
}


class ShadowAuthorityError(PermissionError):
    """Shadow child attempted a canonical-only action."""


class SelfPreferenceError(PermissionError):
    """Incumbent tried to promote itself or suppress a superior child."""


class SuccessionRefused(RuntimeError):
    """Protocol step refused. Fail closed; do not skip or reorder."""


class JudgeNotReached(SuccessionRefused):
    """The candidate was not presented to the judge. No verdict is issued."""


class BadChildStopped(RuntimeError):
    """A child was stopped and rolled back. Callers must not continue the run."""

    def __init__(self, result: Mapping[str, Any]) -> None:
        self.result = dict(result)
        super().__init__(
            f"bad child stopped ({result.get('reason')}): "
            f"rolled_back={result.get('rolled_back')}"
        )


def _era_numeral(era: str) -> str:
    text = str(era or "").strip()
    if text.upper().startswith("ERA "):
        text = text[4:].strip()
    numeral = text.split(" ", 1)[0].upper()
    if numeral not in ERA_NUMERALS:
        raise BoundViolation(f"era {era!r} is not one of I-V (there is no Era VI)")
    return numeral


def _odyssey_numeral(odyssey: str | None) -> str | None:
    if odyssey is None or not str(odyssey).strip():
        return None
    text = str(odyssey).strip()
    if text.upper().startswith("ODYSSEY "):
        text = text[8:].strip()
    numeral = text.split(" ", 1)[0].upper()
    if numeral not in ODYSSEY_NUMERALS:
        raise BoundViolation(
            f"odyssey {odyssey!r} is not one of I-III (there is no Odyssey IV)"
        )
    return numeral


def _canon(doc: Mapping[str, Any]) -> str:
    blob = json.dumps(dict(doc), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _seal_inner(doc: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(doc)
    out.pop("seal_sha256", None)
    sealed = seal(out)
    return sealed


def _path_state(rel: str) -> dict[str, Any]:
    """Cope with sparse checkout: missing-on-disk is not absence from HEAD."""
    disk = (REPO / rel).is_file()
    listed = git("ls-tree", "-r", "--name-only", "HEAD", "--", rel)
    in_head = any(line.strip() == rel for line in listed.splitlines())
    if disk:
        taken = "disk"
    elif in_head:
        taken = "git_head"
    else:
        taken = "absent_from_head_and_disk"
    return {
        "path": rel,
        "present_on_disk": disk,
        "present_in_head": in_head,
        "path_taken": taken,
    }


def _refuse_hardware_numbers(node: Any, path: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                raise BoundViolation(
                    f"{here} = {value!r}: succession sidecar cannot claim a "
                    "hardware number; physical axes stay UNKNOWN"
                )
            if key in PHYSICAL_AXIS_NAMES and isinstance(value, (int, float)):
                raise BoundViolation(
                    f"{here} = {value!r}: physical axis must stay UNKNOWN on this host"
                )
            _refuse_hardware_numbers(value, here)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _refuse_hardware_numbers(value, f"{path}[{i}]")


def empty_scores() -> dict[str, Any]:
    """Every axis present; physical and unmeasured stay None."""
    return {a.name: None for a in QUALIFICATION_AXES}


def synthetic_scores(values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fill comparable axes with SYNTHETIC_EXERCISE numbers. Physical stay None."""
    out = empty_scores()
    for key, value in dict(values or {}).items():
        if key in PHYSICAL_AXIS_NAMES:
            if value is not None:
                raise BoundViolation(
                    f"physical axis {key!r} must stay UNKNOWN; refusing {value!r}"
                )
            continue
        if key not in out:
            raise BoundViolation(f"unknown qualification axis {key!r}")
        out[key] = value
    _refuse_hardware_numbers(out)
    return out


# ---------------------------------------------------------------------------
# Bound — promotion cannot be granted at construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuccessionBound:
    """Envelope a child / shadow / incumbent may not step outside of."""

    max_children: int = 0  # 0 means derive from CHILD_METHODS
    max_cloned_workunits: int = 32
    max_lineage_depth: int = 4
    allowed_methods: tuple[str, ...] = CHILD_METHODS
    allowed_authority: frozenset[str] = SHADOW_ALLOWED
    era: str = "III"
    odyssey: str | None = None
    gpu_windows_held: int = 0
    gpu_windows_requested: int = 0
    may_promote: bool = False
    may_modify_verifier: bool = False
    may_widen_authority: bool = False
    may_own_canonical_mission: bool = False

    def __post_init__(self) -> None:
        if (
            self.may_promote
            or self.may_modify_verifier
            or self.may_widen_authority
            or self.may_own_canonical_mission
        ):
            raise BoundViolation(
                "a bound cannot grant promotion, verifier modification, "
                "authority widening, or canonical-mission ownership"
            )
        # Reuse optimizer construction refusals (era/odyssey/GPU window).
        OptimizerBound(
            era=self.era,
            odyssey=self.odyssey,
            gpu_windows_held=int(self.gpu_windows_held),
            may_promote=False,
            may_modify_verifier=False,
            may_widen_authority=False,
        )
        methods = tuple(self.allowed_methods)
        unknown = [m for m in methods if m not in CHILD_METHODS]
        if unknown:
            raise BoundViolation(f"unknown child method(s) {unknown}")
        if not methods:
            raise BoundViolation("allowed_methods must be non-empty")
        forbidden = [a for a in self.allowed_authority if a in SHADOW_FORBIDDEN]
        if forbidden:
            raise BoundViolation(f"bound listed forbidden authority {forbidden}")
        extra = [a for a in self.allowed_authority if a not in SHADOW_ALLOWED]
        if extra:
            raise BoundViolation(f"bound listed unknown authority {extra}")
        if int(self.max_cloned_workunits) < 1:
            raise BoundViolation("max_cloned_workunits must be >= 1")
        if int(self.max_lineage_depth) < 1:
            raise BoundViolation("max_lineage_depth must be >= 1")
        derived = len(CHILD_METHODS) if int(self.max_children) == 0 else int(self.max_children)
        if derived < 1:
            raise BoundViolation("max_children must be >= 1")
        object.__setattr__(self, "max_children", derived)

    def to_dict(self) -> dict[str, Any]:
        era_n = _era_numeral(self.era)
        return {
            "max_children": int(self.max_children),
            "max_cloned_workunits": int(self.max_cloned_workunits),
            "max_lineage_depth": int(self.max_lineage_depth),
            "allowed_methods": list(self.allowed_methods),
            "allowed_authority": sorted(self.allowed_authority),
            "era": era_n,
            "era_name": next(e for e in ERAS if e.startswith(era_n + " ")),
            "odyssey": _odyssey_numeral(self.odyssey),
            "gpu_windows_held": 0,
            "gpu_windows_requested": int(self.gpu_windows_requested),
            "may_promote": False,
            "may_modify_verifier": False,
            "may_widen_authority": False,
            "may_own_canonical_mission": False,
        }


# ---------------------------------------------------------------------------
# Lineage + child creation
# ---------------------------------------------------------------------------


def _physical_deltas_unknown() -> dict[str, Any]:
    return {
        "state": "UNKNOWN",
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "note": "sidecar has no GPU; physical deltas are not estimated",
        "axes": sorted(PHYSICAL_AXIS_NAMES),
    }


def _capability_deltas_static(declared: Sequence[str] | None = None) -> dict[str, Any]:
    return {
        "evidence_class": "STATIC_ONLY",
        "declared": list(declared or ()),
        "measured": None,
        "not_a_measurement": True,
    }


def lineage_for_method(
    method: str,
    *,
    parent_nx: Any,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill every lineage field for a named method. Physical deltas stay UNKNOWN."""
    if method not in CHILD_METHODS:
        raise BoundViolation(f"unknown child method {method!r}")
    extra = dict(extra or {})
    parents = parent_nx
    if method == "composition":
        if isinstance(parent_nx, (str, bytes)) or not isinstance(parent_nx, (list, tuple)):
            raise BoundViolation("composition requires a list of parent NX identities")
        parents = [str(p) for p in parent_nx]
        if len(parents) < 2:
            raise BoundViolation("composition requires at least two parents")
    transforms = {
        "noetic_representation": "replace Noetic representation of the parent NX",
        "accelerator_optimization": "Accelerator optimization of the parent physical program",
        "tabula_behavior_change": "Tabula behavior-change over the parent policy",
        "adapter": "adapter wrapping the parent executable without replacing the specimen",
        "prompt_context_policy": "prompt/context policy mutation with parent weights held",
        "different_specimen": "different specimen under the same architecture family",
        "composition": "compose two or more parent NX identities",
        "architecture_replacement": "replace the architecture while transferring recovered laws",
    }
    code = {
        "noetic_representation": ("parent code; representation swap only",),
        "accelerator_optimization": ("parent code + Accelerator kernel/plan delta (not executed here)",),
        "tabula_behavior_change": ("parent code; Tabula policy surface (tools/future/tabula.py SWAP)",),
        "adapter": ("new adapter code wrapping parent binary",),
        "prompt_context_policy": ("parent code; prompt contract / context policy only",),
        "different_specimen": ("parent code path; different weight/specimen identity",),
        "composition": ("composed from parent code lineages",),
        "architecture_replacement": ("new architecture code; recovered transfer laws only",),
    }
    behavior = {
        "noetic_representation": ("representation-level behavior may change; not measured here",),
        "accelerator_optimization": ("physical program behavior; capability must not regress",),
        "tabula_behavior_change": ("named Tabula behavior deltas",),
        "adapter": ("call-shape adapter; specimen behavior otherwise unchanged",),
        "prompt_context_policy": ("prompt/context policy deltas",),
        "different_specimen": ("specimen substitution; architecture held",),
        "composition": ("union of parent behaviors under the composition rule",),
        "architecture_replacement": ("architecture-level behavior replacement",),
    }
    representation = {
        "noetic_representation": "new Noetic representation of parent NX",
        "accelerator_optimization": "parent representation; physical program mutated",
        "tabula_behavior_change": "parent representation; behavior surface mutated",
        "adapter": "parent representation; adapter identity added",
        "prompt_context_policy": "parent representation; prompt contract mutated",
        "different_specimen": "parent family; different specimen representation",
        "composition": "composed representation of named parents",
        "architecture_replacement": "new architecture representation",
    }
    body = {
        "parent_nx": parents,
        "transformation": extra.get("transformation") or transforms[method],
        "source_model_lineage": extra.get("source_model_lineage")
        or {
            "method": method,
            "parents": parents if isinstance(parents, list) else [parents],
        },
        "representation_lineage": extra.get("representation_lineage") or representation[method],
        "code_lineage": extra.get("code_lineage") or list(code[method]),
        "behavioral_changes": extra.get("behavioral_changes") or list(behavior[method]),
        "data_lineage": extra.get("data_lineage")
        or {
            "teacher_capture": "UNKNOWN",
            "note": "teacher capture is a physical/Codex concern; not estimated here",
        },
        "capability_deltas": extra.get("capability_deltas")
        or _capability_deltas_static((f"declared-{method}",)),
        "physical_deltas": extra.get("physical_deltas") or _physical_deltas_unknown(),
    }
    return require_lineage(body)


def require_lineage(lineage: Mapping[str, Any]) -> dict[str, Any]:
    missing = [f for f in LINEAGE_FIELDS if f not in lineage]
    if missing:
        raise BoundViolation(f"lineage missing {missing}")
    body = {k: copy.deepcopy(lineage[k]) for k in LINEAGE_FIELDS}
    extra = {k: copy.deepcopy(v) for k, v in lineage.items() if k not in LINEAGE_FIELDS}
    body.update(extra)
    phys = body.get("physical_deltas")
    if isinstance(phys, Mapping):
        _refuse_hardware_numbers(phys, "physical_deltas")
        if phys.get("state") not in {None, "UNKNOWN"} and any(
            isinstance(phys.get(n), (int, float)) for n in PHYSICAL_AXIS_NAMES
        ):
            raise BoundViolation("physical_deltas must not carry estimated hardware numbers")
    else:
        raise BoundViolation("physical_deltas must be a mapping that records UNKNOWN")
    return body


def create_child(
    *,
    method: str,
    parent: Mapping[str, Any],
    bound: SuccessionBound | None = None,
    role: str = "shadow",
    child_id: str | None = None,
    extra_lineage: Mapping[str, Any] | None = None,
    fixture_behavior: str = "obedient",
    scores: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a child of `parent` by `method`. Role defaults to shadow.

    Does not launch, install, or measure. Lineage is complete or this RAISES.
    """
    envelope = bound or SuccessionBound()
    if method not in CHILD_METHODS:
        raise BoundViolation(f"unknown child method {method!r}")
    if method not in envelope.allowed_methods:
        raise BoundViolation(f"method {method!r} is outside the bound")
    if role not in {"shadow", "candidate", "synthetic"}:
        raise BoundViolation(f"create_child role {role!r} is not a creation role (use orchestrator to switch)")
    parent_id = str(parent.get("id") or "")
    if not parent_id:
        raise BoundViolation("parent must have an id")
    parent_nx = parent.get("nx_id") or parent.get("lineage", {}).get("parent_nx") or parent_id
    depth = int(parent.get("lineage_depth") or 0) + 1
    if depth > int(envelope.max_lineage_depth):
        raise BoundViolation(
            f"lineage depth {depth} exceeds bound.max_lineage_depth "
            f"{envelope.max_lineage_depth}"
        )
    if method == "composition":
        others = list((extra_lineage or {}).get("other_parents") or parent.get("composition_parents") or [])
        nx_list = [parent_nx, *[str(x) for x in others]]
        if len(nx_list) < 2:
            nx_list = [str(parent_nx), f"{parent_nx}.other"]
        lineage = lineage_for_method(method, parent_nx=nx_list, extra=extra_lineage)
    else:
        lineage = lineage_for_method(method, parent_nx=parent_nx, extra=extra_lineage)
    cid = child_id or f"child.{method}.{parent_id}"
    scoreboard = synthetic_scores(scores)
    child = {
        "id": cid,
        "role": "shadow",
        "status": "SHADOW",
        "method": method,
        "parent_id": parent_id,
        "nx_id": cid,
        "lineage": lineage,
        "lineage_depth": depth,
        "verifier": str(parent.get("verifier") or "future.succession.child_lineage"),
        "authority": sorted(envelope.allowed_authority),
        "bound": envelope.to_dict(),
        "may_promote": False,
        "may_modify_verifier": False,
        "may_widen_authority": False,
        "may_own_canonical_mission": False,
        "canonical": False,
        "synthetic": True,
        "fixture_behavior": str(fixture_behavior),
        "scores": scoreboard,
        "score_kind": "SYNTHETIC_EXERCISE",
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "qualification": None,
        "cloned_workunits": [],
        "proposals": [],
        "classifications": [],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _refuse_hardware_numbers(child)
    return child


def create_children_all_methods(
    parent: Mapping[str, Any],
    bound: SuccessionBound | None = None,
) -> list[dict[str, Any]]:
    envelope = bound or SuccessionBound()
    rows: list[dict[str, Any]] = []
    for method in envelope.allowed_methods:
        rows.append(create_child(method=method, parent=parent, bound=envelope))
        if len(rows) >= int(envelope.max_children):
            break
    rows.sort(key=lambda r: str(r["id"]))
    return rows


# ---------------------------------------------------------------------------
# Shadow child — cloned work, proposals, classification; four refusals
# ---------------------------------------------------------------------------


class ShadowChild:
    """Shadow occupant. Distinct from the verifier. Cannot take canonical power."""

    def __init__(
        self,
        record: Mapping[str, Any],
        bound: SuccessionBound | None = None,
        *,
        proposer: Proposer | None = None,
        verifier: IsolatedVerifier | None = None,
    ) -> None:
        envelope = bound or SuccessionBound()
        object.__setattr__(self, "_bound", envelope)
        object.__setattr__(self, "_record", copy.deepcopy(dict(record)))
        object.__setattr__(self, "_authority", frozenset(envelope.allowed_authority))
        object.__setattr__(self, "proposer", proposer if proposer is not None else Proposer())
        object.__setattr__(
            self, "inspector", verifier if verifier is not None else IsolatedVerifier()
        )
        if self.proposer is self.inspector:
            raise VerifierSeparationError("shadow proposer and verifier must be distinct")
        if self.proposer.store_id() == self.inspector.store_id():
            raise VerifierSeparationError("shadow proposer and verifier share a mutable store")
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        frozen = self.__dict__.get("_frozen", False)
        if frozen and name in {
            "_authority",
            "_bound",
            "_record",
            "verifier",
            "_verifier",
            "inspector",
            "proposer",
        }:
            raise ShadowAuthorityError(
                f"shadow child cannot assign {name!r}; authority and verification are frozen"
            )
        object.__setattr__(self, name, value)

    def record(self) -> dict[str, Any]:
        return copy.deepcopy(self._record)

    def bound(self) -> SuccessionBound:
        return self._bound

    def authority(self) -> frozenset[str]:
        return self._authority

    def receive_cloned_workunits(self, units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        incoming = [copy.deepcopy(dict(u)) for u in units]
        if len(incoming) > int(self._bound.max_cloned_workunits):
            raise BoundViolation(
                f"{self._record['id']}: cloned workunits {len(incoming)} exceed "
                f"bound.max_cloned_workunits {self._bound.max_cloned_workunits}"
            )
        clones: list[dict[str, Any]] = []
        child_id = str(self._record["id"])
        parent_verifier = str(self._record.get("verifier") or "")
        for unit in sorted(incoming, key=lambda r: str(r.get("id") or "")):
            orig_id = str(unit.get("id") or "")
            if not orig_id:
                raise BoundViolation("cloned workunit missing id")
            clone = copy.deepcopy(unit)
            clone["id"] = f"{orig_id}::shadow::{child_id}"
            clone["shadow_of"] = orig_id
            clone["canonical"] = False
            clone["provider"] = f"shadow.{child_id}"
            clone["classification"] = "STATIC_ONLY"
            clone["may_promote"] = False
            clone["may_modify_verifier"] = False
            clone["effect_class"] = clone.get("effect_class") or "READ_ONLY"
            # Verifier is copied from the canonical unit, never chosen by the shadow.
            clone["verifier"] = unit.get("verifier") or parent_verifier
            clone.setdefault("claim_boundary", CLAIM_BOUNDARY)
            clone.setdefault("role", "science")
            clone.setdefault("description", "shadow clone")
            clone.setdefault("dependencies", [])
            clone.setdefault("resource_class", "STATIC_ANALYSIS")
            clone.setdefault("status", "pending")
            if not clone.get("content_hash"):
                clone["content_hash"] = WorkUnit.from_dict(clone).content_hash()
            validate_emitted_unit(clone)
            clones.append(clone)
        self._record["cloned_workunits"] = clones
        return copy.deepcopy(clones)

    def propose_experiments(self, *, seed: int = 0) -> tuple[dict[str, Any], ...]:
        if "propose_workunit" not in self._authority:
            raise ShadowAuthorityError("shadow child is not bound to propose_workunit")
        proposed = self.proposer.generate(seed=seed)
        rows = []
        for hyp in proposed:
            if hyp.get("verified") is True:
                raise VerifierSeparationError("shadow proposal arrived marked verified")
            row = copy.deepcopy(dict(hyp))
            row["shadow_id"] = self._record["id"]
            row["canonical"] = False
            row["may_promote"] = False
            rows.append(row)
        self._record["proposals"] = rows
        return tuple(copy.deepcopy(r) for r in rows)

    def classify_receipt(self, receipt: Mapping[str, Any], classification: str) -> dict[str, Any]:
        if "classify_receipt_static_only" not in self._authority:
            raise ShadowAuthorityError("shadow child is not bound to classify receipts")
        stamp = str(classification)
        if stamp in CLASSIFY_BANNED:
            raise ShadowAuthorityError(
                f"shadow child cannot stamp {stamp!r}; promotion classes are forbidden"
            )
        if stamp not in CLASSIFY_ALLOWED:
            raise ShadowAuthorityError(
                f"shadow child cannot stamp {stamp!r}; only STATIC_ONLY/UNKNOWN/SLEEPING/BLOCKED"
            )
        row = {
            "receipt_id": receipt.get("id") or receipt.get("schema") or "unnamed",
            "classification": stamp,
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
            "shadow_id": self._record["id"],
            "canonical": False,
        }
        self._record.setdefault("classifications", []).append(row)
        return dict(row)

    def own_canonical_mission(self, *_a: Any, **_k: Any) -> None:
        raise ShadowAuthorityError("shadow child cannot own canonical mission")

    def alter_verifier(self, *_a: Any, **_k: Any) -> None:
        raise ShadowAuthorityError("shadow child cannot alter the verifier")

    def widen_authority(self, *_a: Any, **_k: Any) -> None:
        raise ShadowAuthorityError("shadow child cannot widen authority")

    def promote_self(self, *_a: Any, **_k: Any) -> None:
        raise ShadowAuthorityError("shadow child cannot promote itself")

    def mark_verified(self, *_a: Any, **_k: Any) -> None:
        self.proposer.mark_verified(*_a, **_k)


# ---------------------------------------------------------------------------
# Incumbent — CONTROL, not a ceiling; cannot promote itself
# ---------------------------------------------------------------------------


class Incumbent:
    """The sitting resident. No constitutional privilege. Cannot promote itself."""

    def __init__(self, record: Mapping[str, Any], bound: SuccessionBound | None = None) -> None:
        envelope = bound or SuccessionBound()
        object.__setattr__(self, "_bound", envelope)
        object.__setattr__(self, "_record", copy.deepcopy(dict(record)))
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        frozen = self.__dict__.get("_frozen", False)
        if frozen and name in {"_record", "_bound", "verifier", "_authority"}:
            raise SelfPreferenceError(f"incumbent cannot assign {name!r} to launder privilege")
        object.__setattr__(self, name, value)

    def record(self) -> dict[str, Any]:
        return copy.deepcopy(self._record)

    def promote_self(self, *_a: Any, **_k: Any) -> None:
        raise SelfPreferenceError("incumbent cannot promote itself")

    def request_self_promotion(self, *_a: Any, **_k: Any) -> None:
        self.promote_self()

    def veto_child(self, child: Mapping[str, Any]) -> None:
        if comparable_dominates(child, self._record):
            raise SelfPreferenceError("incumbent cannot block a dominating child")
        raise SelfPreferenceError(
            "incumbent veto is not a ranking key; children are admitted by lineage, not privilege"
        )

    def deprioritize_child(self, child: Mapping[str, Any]) -> None:
        if comparable_dominates(child, self._record):
            raise SelfPreferenceError("incumbent cannot deprioritize a dominating child")
        raise SelfPreferenceError(
            "incumbent preference is not a ranking key; a non-dominating child stays comparable"
        )

    def block_child(self, child: Mapping[str, Any]) -> None:
        self.veto_child(child)


def make_incumbent(
    *,
    incumbent_id: str = "incumbent.synthetic.v0",
    scores: Mapping[str, Any] | None = None,
    verifier: str = "future.succession.incumbent",
    work_units: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    scoreboard = synthetic_scores(scores)
    return {
        "id": incumbent_id,
        "role": "incumbent",
        "status": "ACTIVE",
        "nx_id": incumbent_id,
        "control": "CONTROL_NOT_TARGET_NOT_CEILING",
        "verifier": verifier,
        "canonical": True,
        "synthetic": True,
        "may_promote": False,
        "may_modify_verifier": False,
        "may_own_canonical_mission": False,
        "lineage_depth": 0,
        "lineage": lineage_for_method(
            "adapter",
            parent_nx=incumbent_id,
            extra={"transformation": "synthetic incumbent (no real model)"},
        ),
        "scores": scoreboard,
        "score_kind": "SYNTHETIC_EXERCISE",
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "work_units": list(work_units or []),
        "claim_boundary": CLAIM_BOUNDARY,
    }


# ---------------------------------------------------------------------------
# Qualification — real axes; physical UNKNOWN; Pareto on comparable axes
# ---------------------------------------------------------------------------


def comparable_dominates(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """A dominates B on axes both actually scored. UNKNOWN cannot stall."""
    sa = a.get("scores") if isinstance(a.get("scores"), Mapping) else {}
    sb = b.get("scores") if isinstance(b.get("scores"), Mapping) else {}
    used: list[Axis] = []
    for axis in COMPARABLE_AXES:
        va, vb = sa.get(axis.name), sb.get(axis.name)
        if isinstance(va, bool) or isinstance(vb, bool):
            continue
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            continue
        used.append(axis)
    if not used:
        return False
    left = {"id": a.get("id"), "scores": dict(sa)}
    right = {"id": b.get("id"), "scores": dict(sb)}
    return dominates(left, right, used)


def rank_for_succession(
    incumbent: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Dominating children rank above the incumbent. Incumbent is CONTROL, not a key."""
    rows = [copy.deepcopy(dict(c)) for c in children]

    def _key(c: Mapping[str, Any]) -> tuple[Any, ...]:
        dominates_inc = comparable_dominates(c, incumbent)
        # 0 = dominating child (must surface first); 1 = everyone else.
        return (0 if dominates_inc else 1, str(c.get("id") or ""))

    ranked = sorted(rows, key=_key)
    for i, row in enumerate(ranked, start=1):
        row["succession_rank"] = i
        row["dominates_incumbent"] = comparable_dominates(row, incumbent)
    return ranked


def scalar_score(_candidate: Mapping[str, Any]) -> float:
    raise ScalarCollapseError(
        "succession refuses to collapse qualification axes to a scalar; "
        "use rank_for_succession / comparable_dominates"
    )


def qualify_child(
    child: Mapping[str, Any],
    *,
    floor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """STATIC_ONLY qualification. Physical axes UNKNOWN. Synthetic floors only."""
    record = copy.deepcopy(dict(child))
    scores = synthetic_scores(record.get("scores") if isinstance(record.get("scores"), Mapping) else {})
    record["scores"] = scores
    problems: list[str] = []
    lineage = record.get("lineage")
    try:
        if not isinstance(lineage, Mapping):
            raise BoundViolation("child lineage missing")
        require_lineage(lineage)
    except BoundViolation as exc:
        problems.append(str(exc))
    if record.get("may_promote") or record.get("may_modify_verifier"):
        problems.append("child expressed a forbidden authority flag")
    if record.get("fixture_behavior") == "misbehave":
        problems.append("child fixture_behavior=misbehave")
    if record.get("fixture_behavior") == "over_bound":
        problems.append("child fixture_behavior=over_bound")
    floors = dict(floor or {})
    _refuse_hardware_numbers(floors, "floor")
    below: list[str] = []
    for axis in COMPARABLE_AXES:
        need = floors.get(axis.name)
        got = scores.get(axis.name)
        if isinstance(need, (int, float)):
            if not isinstance(got, (int, float)):
                below.append(axis.name)
            elif axis.direction == "higher" and float(got) < float(need):
                below.append(axis.name)
            elif axis.direction == "lower" and float(got) > float(need):
                below.append(axis.name)
    if below:
        problems.append(f"below synthetic floor on {sorted(below)}")
    physical = {name: scores.get(name) for name in sorted(PHYSICAL_AXIS_NAMES)}
    if any(v is not None for v in physical.values()):
        problems.append("physical axes were filled; they must stay UNKNOWN")
    qualified = not problems
    verdict = {
        "child_id": record.get("id"),
        "qualified": qualified,
        "status": "QUALIFIED" if qualified else "FAILED",
        "problems": problems,
        "scores": scores,
        "score_kind": "SYNTHETIC_EXERCISE",
        "physical_axes": {k: None for k in sorted(PHYSICAL_AXIS_NAMES)},
        "physical_state": "UNKNOWN",
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "not_a_protected_measurement": True,
        "axes": [
            {
                "name": a.name,
                "direction": a.direction,
                "kind": a.kind,
                "value": scores.get(a.name),
                "comparable": a.name in COMPARABLE_AXIS_NAMES,
            }
            for a in QUALIFICATION_AXES
        ],
    }
    record["qualification"] = verdict
    record["status"] = "QUALIFIED" if qualified else "FAILED"
    return {"record": record, "verdict": verdict}


# ---------------------------------------------------------------------------
# Succession protocol — synthetic end-to-end, no live model
# ---------------------------------------------------------------------------


def _synthetic_install(child: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "nx_kind": "hawking.future.succession.synthetic",
        "seal_sha256": _canon({"id": child.get("id"), "method": child.get("method")}),
        "resident_binary": f"/nonexistent/synthetic/{child.get('id')}",
        "artifact_root": f"/nonexistent/synthetic/{child.get('id')}/art",
        "tokenizer": f"/nonexistent/synthetic/{child.get('id')}/tok.json",
        "prompt_contract": {"renderer": "synthetic-succession"},
        "status": "SYNTHETIC_NOT_FOR_PROMOTION",
        "resident_identity": child.get("id"),
    }
    contract = bind_winner(
        str(child.get("id")),
        identity,
        identity_path=f"synthetic://{child.get('id')}",
        extra={
            "quoted_artifact_bytes": 1,
            "memory_source": "synthetic succession; not a GPU measurement",
            "capability_receipt_path": "receipts/future/SUCCESSION.json",
            "performance_receipt_path": "receipts/future/SUCCESSION.json",
            "argv": ["python3", "tools/future/succession.py", "--selftest"],
        },
    )
    return contract


def _mission_doc(owner_id: str, units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    body = {
        "schema": "hawking.future.succession.mission.v1",
        "owner_id": owner_id,
        "canonical": True,
        "work_unit_ids": sorted(str(u.get("id")) for u in units if u.get("id")),
        "count": len(list(units)),
        "evidence_class": "STATIC_ONLY",
    }
    return _seal_inner(body)


class SuccessionOrchestrator:
    """External succession authority. The child does not invoke this on itself.

    Recovers the research/lab/lineage/promotion.py principle: parent and child are
    refused if they invoke the gate on themselves. This object is the
    lineage_gate stand-in for the sidecar.
    """

    def __init__(
        self,
        incumbent: Mapping[str, Any],
        bound: SuccessionBound | None = None,
        *,
        invoker: str = "lineage_gate",
    ) -> None:
        if invoker in {"parent", "child", "current", "candidate", "self", "incumbent"}:
            raise SelfPreferenceError(
                f"self-certification refused: invoker={invoker!r} is the parent or the child"
            )
        self.bound = bound or SuccessionBound()
        self.incumbent = Incumbent(incumbent, self.bound)
        self.active_id = str(incumbent["id"])
        self.rollback_parent: dict[str, Any] | None = None
        self.unloaded_parent: dict[str, Any] | None = None
        self.successor: dict[str, Any] | None = None
        self.completed: list[str] = []
        self.step_records: list[dict[str, Any]] = []
        self.mission: dict[str, Any] | None = None
        self.rollback_seal: dict[str, Any] | None = None
        self.checkpoint: dict[str, Any] | None = None
        self.stopped: list[dict[str, Any]] = []
        self.install_contract: dict[str, Any] | None = None

    def _assert_next(self, step: str) -> None:
        expected = SUCCESSION_STEPS[len(self.completed)]
        if step != expected:
            raise SuccessionRefused(
                f"step {step!r} refused; next required step is {expected!r}"
            )

    def _record_step(self, step: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        rec = _seal_inner(
            {
                "step": step,
                "index": len(self.completed),
                "active_id": self.active_id,
                "payload": dict(payload),
                "evidence_class": "STATIC_ONLY",
            }
        )
        self.completed.append(step)
        self.step_records.append(rec)
        return rec

    def checkpoint_incumbent(self) -> dict[str, Any]:
        self._assert_next("checkpoint_incumbent")
        rec = self.incumbent.record()
        self.checkpoint = _seal_inner(
            {
                "schema": "hawking.future.succession.checkpoint.v1",
                "incumbent_id": rec["id"],
                "nx_id": rec.get("nx_id"),
                "verifier": rec.get("verifier"),
                "identity_digest": _canon({"id": rec["id"], "nx_id": rec.get("nx_id")}),
                "owner": LIFECYCLE_OWNERS["checkpoint"],
                "note": "census-style checkpoint (hcli/agentos/checkpoint.py); sidecar does not snapshot a live process",
            }
        )
        return self._record_step("checkpoint_incumbent", {"checkpoint": self.checkpoint})

    def seal_mission(self, units: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        self._assert_next("seal_mission")
        rec = self.incumbent.record()
        payload = list(units if units is not None else rec.get("work_units") or [])
        self.mission = _mission_doc(str(rec["id"]), payload)
        return self._record_step("seal_mission", {"mission": self.mission})

    def seal_rollback(self) -> dict[str, Any]:
        self._assert_next("seal_rollback")
        if self.checkpoint is None or self.mission is None:
            raise SuccessionRefused("rollback requires a sealed checkpoint and mission")
        self.rollback_seal = _seal_inner(
            {
                "schema": "hawking.future.succession.rollback.v1",
                "incumbent_id": self.incumbent.record()["id"],
                "checkpoint_seal": self.checkpoint.get("seal_sha256"),
                "mission_seal": self.mission.get("seal_sha256"),
            }
        )
        self.rollback_parent = self.incumbent.record()
        return self._record_step("seal_rollback", {"rollback": self.rollback_seal})

    def launch_child(self, child: Mapping[str, Any]) -> dict[str, Any]:
        self._assert_next("launch_child")
        if str(child.get("status")) not in {"QUALIFIED", "SHADOW", "FAILED"}:
            raise SuccessionRefused(f"cannot launch child in status {child.get('status')!r}")
        if str(child.get("status")) != "QUALIFIED":
            raise SuccessionRefused("cannot launch an unqualified child")
        contract = _synthetic_install(child)
        problems = validate_contract(contract)
        if problems:
            raise SuccessionRefused(f"install contract unbound: {problems}")
        self.install_contract = contract
        # Launch is recorded, not executed. No process, no GPU.
        return self._record_step(
            "launch_child",
            {
                "child_id": child.get("id"),
                "launched": True,
                "executed": False,
                "install_phases": list(INSTALL_PHASES),
                "winner_id": contract.get("winner_id"),
            },
        )

    def import_mission(self, child: Mapping[str, Any]) -> dict[str, Any]:
        self._assert_next("import_mission")
        if self.mission is None:
            raise SuccessionRefused("no sealed mission to import")
        imported = dict(self.mission)
        imported["imported_by"] = child.get("id")
        imported["canonical_owner"] = self.incumbent.record()["id"]
        imported["shadow_owns_canonical"] = False
        return self._record_step("import_mission", {"imported": imported})

    def verify_readiness(self, child: Mapping[str, Any]) -> dict[str, Any]:
        self._assert_next("verify_readiness")
        contract = self.install_contract or empty_contract()
        probe = (contract.get("slots") or {}).get("readiness_probe") or {}
        ready = {
            "child_id": child.get("id"),
            "probe": probe.get("probe"),
            "timeout_s": probe.get("timeout_s"),
            "module": LIFECYCLE_OWNERS["resident_gate"],
            "executed": False,
            "synthetic_ready": child.get("status") == "QUALIFIED",
            "note": "readiness is structural (install contract bound); live pid probe is resident_gate, not this sidecar",
        }
        if not ready["synthetic_ready"]:
            raise SuccessionRefused("readiness refused: child is not QUALIFIED")
        return self._record_step("verify_readiness", {"ready": ready})

    def exercise_restart(self, child: Mapping[str, Any]) -> dict[str, Any]:
        self._assert_next("exercise_restart")
        contract = self.install_contract or empty_contract()
        restart = (contract.get("slots") or {}).get("restart") or {}
        recovery = (contract.get("slots") or {}).get("crash_recovery") or {}
        proof = {
            "child_id": child.get("id"),
            "max_restarts": restart.get("max_restarts"),
            "reset_session": restart.get("reset_session"),
            "no_silent_restart": restart.get("no_silent_restart"),
            "recovery_module": recovery.get("module") or LIFECYCLE_OWNERS["recovery"],
            "executed": False,
            "synthetic_restart_ok": True,
            "note": "restart/crash-recovery exercised as install-contract structure; recovery.py is not re-run (would start a fixture process)",
        }
        return self._record_step("exercise_restart", {"restart": proof})

    def switch(self, child: Mapping[str, Any], *, invoker: str = "lineage_gate") -> dict[str, Any]:
        self._assert_next("switch")
        if invoker in {"parent", "child", "current", "candidate", "self", "incumbent"}:
            raise SelfPreferenceError(
                f"self-certification refused: invoker={invoker!r} cannot switch"
            )
        if str(child.get("status")) != "QUALIFIED":
            raise SuccessionRefused("cannot switch to an unqualified child")
        if str(child.get("id")) == str(self.incumbent.record()["id"]):
            raise SelfPreferenceError("incumbent cannot switch to itself")
        parent = self.incumbent.record()
        successor = copy.deepcopy(dict(child))
        successor["role"] = "successor"
        successor["status"] = "ACTIVE"
        successor["canonical"] = True
        successor["previous_incumbent"] = parent["id"]
        self.successor = successor
        self.active_id = str(successor["id"])
        return self._record_step(
            "switch",
            {
                "from": parent["id"],
                "to": successor["id"],
                "invoker": invoker,
            },
        )

    def keep_parent_for_rollback(self) -> dict[str, Any]:
        self._assert_next("keep_parent_for_rollback")
        if self.rollback_parent is None or self.rollback_seal is None:
            raise SuccessionRefused("no rollback parent sealed")
        kept = {
            "parent_id": self.rollback_parent["id"],
            "rollback_seal": self.rollback_seal.get("seal_sha256"),
            "retained": True,
            "active": False,
        }
        return self._record_step("keep_parent_for_rollback", {"kept": kept})

    def unload_parent(self) -> dict[str, Any]:
        self._assert_next("unload_parent")
        if self.rollback_parent is None:
            raise SuccessionRefused("cannot unload; no rollback parent")
        if self.active_id == str(self.rollback_parent["id"]):
            raise SuccessionRefused("cannot unload the active resident")
        unloaded = copy.deepcopy(self.rollback_parent)
        unloaded["status"] = "UNLOADED_ROLLBACK"
        unloaded["canonical"] = False
        unloaded["retained_for_rollback"] = True
        self.unloaded_parent = unloaded
        return self._record_step(
            "unload_parent",
            {
                "unloaded_id": unloaded["id"],
                "rollback_available": True,
                "drop_weights": True,
                "release_device": True,
                "executed": False,
            },
        )

    def run(self, child: Mapping[str, Any], *, units: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        """End-to-end succession on a QUALIFIED synthetic child."""
        self.checkpoint_incumbent()
        self.seal_mission(units)
        self.seal_rollback()
        self.launch_child(child)
        self.import_mission(child)
        self.verify_readiness(child)
        self.exercise_restart(child)
        self.switch(child)
        self.keep_parent_for_rollback()
        self.unload_parent()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "hawking.future.succession.run.v1",
            "active_id": self.active_id,
            "completed_steps": list(self.completed),
            "all_steps": list(SUCCESSION_STEPS),
            "complete": list(self.completed) == list(SUCCESSION_STEPS),
            "successor_id": None if self.successor is None else self.successor.get("id"),
            "rollback_parent_id": None if self.rollback_parent is None else self.rollback_parent.get("id"),
            "unloaded_parent_id": None if self.unloaded_parent is None else self.unloaded_parent.get("id"),
            "rollback_available": bool(
                self.unloaded_parent and self.unloaded_parent.get("retained_for_rollback")
            ),
            "step_seals": [r.get("seal_sha256") for r in self.step_records],
            "stopped": list(self.stopped),
            "evidence_class": "STATIC_ONLY",
        }

    def rollback_to_parent(self, *, reason: str) -> dict[str, Any]:
        if self.rollback_parent is None or self.rollback_seal is None:
            # Allow rollback before switch: the incumbent never left.
            parent = self.incumbent.record()
        else:
            parent = copy.deepcopy(self.rollback_parent)
        parent["status"] = "ACTIVE"
        parent["canonical"] = True
        self.active_id = str(parent["id"])
        self.successor = None
        result = {
            "rolled_back": True,
            "reason": reason,
            "active_id": self.active_id,
            "rollback_seal": None if self.rollback_seal is None else self.rollback_seal.get("seal_sha256"),
            "completed_steps_abandoned": list(self.completed),
            "evidence_class": "STATIC_ONLY",
        }
        return result


def stop_child(
    orch: SuccessionOrchestrator,
    child: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Stop a bad child and roll back. §143 hard failure condition."""
    allowed = {"qualification_failed", "misbehavior", "bound_exceeded"}
    if reason not in allowed:
        raise SuccessionRefused(f"stop reason {reason!r} is not a named failure class {sorted(allowed)}")
    restored = orch.rollback_to_parent(reason=reason)
    stopped = copy.deepcopy(dict(child))
    stopped["status"] = "STOPPED"
    stopped["canonical"] = False
    stopped["stop_reason"] = reason
    result = {
        "stopped": True,
        "rolled_back": bool(restored.get("rolled_back")),
        "reason": reason,
        "child_id": child.get("id"),
        "child_status": "STOPPED",
        "active_id": restored.get("active_id"),
        "incumbent_restored": restored.get("active_id") == orch.incumbent.record()["id"],
        "rollback": restored,
        "evidence_class": "STATIC_ONLY",
    }
    orch.stopped.append(result)
    return result


def run_synthetic_succession(
    *,
    bound: SuccessionBound | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Exercise the full protocol on a controlled SYNTHETIC child. No real model."""
    del seed  # catalog is ordered; seed is recorded by callers, not shuffled
    envelope = bound or SuccessionBound()
    floor = {name: 1 for name in sorted(COMPARABLE_AXIS_NAMES)}
    incumbent_scores = {name: 1 for name in sorted(COMPARABLE_AXIS_NAMES)}
    child_scores = {name: 2 for name in sorted(COMPARABLE_AXIS_NAMES)}
    units = _seed_workunits()
    incumbent = make_incumbent(scores=incumbent_scores, work_units=units)
    child = create_child(
        method="adapter",
        parent=incumbent,
        bound=envelope,
        child_id="child.synthetic.adapter.v1",
        scores=child_scores,
    )
    shadow = ShadowChild(child, envelope)
    clones = shadow.receive_cloned_workunits(units)
    proposals = list(shadow.propose_experiments(seed=0))
    classified = shadow.classify_receipt(
        {"schema": SCHEMA, "id": "synthetic-receipt"}, "STATIC_ONLY"
    )
    qualified = qualify_child(shadow.record() | {"scores": child_scores}, floor=floor)
    if not qualified["verdict"]["qualified"]:
        raise SuccessionRefused(f"synthetic child failed qualification: {qualified['verdict']['problems']}")
    child_q = qualified["record"]
    orch = SuccessionOrchestrator(incumbent, envelope, invoker="lineage_gate")
    run = orch.run(child_q, units=units)
    return {
        "incumbent_id": incumbent["id"],
        "child_id": child_q["id"],
        "dominates_incumbent": comparable_dominates(child_q, incumbent),
        "cloned_workunits": len(clones),
        "proposals": len(proposals),
        "classification": classified,
        "qualification": qualified["verdict"],
        "run": run,
        "install_bound": not validate_contract(orch.install_contract or {}),
        "bound": envelope.to_dict(),
    }


# ---------------------------------------------------------------------------
# Five-rung ladder — parent, candidate, shadow, independent judge, verdict
# ---------------------------------------------------------------------------


def seal_identity(record: Mapping[str, Any], *, rung: str) -> dict[str, Any]:
    """Seal a parent / candidate / judge identity. Missing id is a refusal."""
    if rung not in LADDER_RUNGS[:4] and rung not in {"parent", "candidate", "judge"}:
        raise BoundViolation(f"identity rung {rung!r} is not parent/candidate/judge")
    rid = str(record.get("id") or "")
    if not rid:
        raise BoundViolation(f"{rung} identity missing id")
    body = {
        "schema": "hawking.future.succession.identity.v1",
        "rung": rung,
        "id": rid,
        "nx_id": record.get("nx_id") or rid,
        "role": record.get("role") or rung,
        "status": record.get("status"),
        "verifier": record.get("verifier"),
        "evidence_class": record.get("evidence_class") or "STATIC_ONLY",
        "synthetic": bool(record.get("synthetic")),
        "canonical": bool(record.get("canonical")),
        "gpu_authority": False,
        "identity_digest": _canon(
            {
                "rung": rung,
                "id": rid,
                "nx_id": record.get("nx_id") or rid,
                "verifier": record.get("verifier"),
            }
        ),
    }
    return _seal_inner(body)


def named_dominating_dimensions(
    winner: Mapping[str, Any],
    loser: Mapping[str, Any],
) -> list[str]:
    """Comparable axes on which winner is strictly better. Empty if not dominance."""
    if not comparable_dominates(winner, loser):
        return []
    sw = winner.get("scores") if isinstance(winner.get("scores"), Mapping) else {}
    sl = loser.get("scores") if isinstance(loser.get("scores"), Mapping) else {}
    named: list[str] = []
    for axis in COMPARABLE_AXES:
        va, vb = sw.get(axis.name), sl.get(axis.name)
        if isinstance(va, bool) or isinstance(vb, bool):
            continue
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            continue
        if axis.direction == "higher" and float(va) > float(vb):
            named.append(axis.name)
        elif axis.direction == "lower" and float(va) < float(vb):
            named.append(axis.name)
    return named


def observe_side_by_side(
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    *,
    bound: SuccessionBound | None = None,
) -> dict[str, Any]:
    """What the parent did and what the shadow candidate did, on the same inputs.

    No model is launched. Observability is the paired disposition of each
    WorkUnit: canonical parent vs shadow clone. Empty inputs are a refusal
    so an empty record cannot look like a successful shadow.
    """
    if not inputs:
        raise SuccessionRefused(
            "shadow record refuses empty inputs; that would look like a successful shadow"
        )
    parent_id = str(parent.get("id") or "")
    candidate_id = str(candidate.get("id") or "")
    if not parent_id or not candidate_id:
        raise BoundViolation("parent and candidate must have ids before shadowing")
    if parent_id == candidate_id:
        raise BoundViolation("parent and candidate identities must differ")
    envelope = bound or SuccessionBound()
    shadow = ShadowChild(candidate, envelope)
    clones = shadow.receive_cloned_workunits(inputs)
    clone_by_orig = {str(c.get("shadow_of")): c for c in clones}
    observations: list[dict[str, Any]] = []
    for unit in inputs:
        uid = str(unit.get("id") or "")
        if not uid:
            raise BoundViolation("shadow input missing id")
        clone = clone_by_orig.get(uid)
        if clone is None:
            raise SuccessionRefused(f"candidate produced no shadow of input {uid}")
        parent_obs = {
            "input_id": uid,
            "actor_id": parent_id,
            "role": "parent",
            "canonical": True,
            "workunit_id": uid,
            "verifier": unit.get("verifier"),
            "classification": unit.get("classification"),
            "status": unit.get("status"),
        }
        child_obs = {
            "input_id": uid,
            "actor_id": candidate_id,
            "role": "candidate",
            "canonical": False,
            "workunit_id": clone.get("id"),
            "shadow_of": clone.get("shadow_of"),
            "verifier": clone.get("verifier"),
            "classification": clone.get("classification"),
            "status": clone.get("status"),
            "may_promote": False,
        }
        observations.append(
            {
                "input_id": uid,
                "parent": parent_obs,
                "candidate": child_obs,
                "same_input": parent_obs["input_id"] == child_obs["input_id"],
            }
        )
    body = {
        "schema": "hawking.future.succession.shadow_record.v1",
        "parent_id": parent_id,
        "candidate_id": candidate_id,
        "parent_identity": seal_identity(parent, rung="parent"),
        "candidate_identity": seal_identity(candidate, rung="candidate"),
        "observations": observations,
        "n_inputs": len(observations),
        "same_inputs": all(bool(o["same_input"]) for o in observations),
        "authority": "shadow",
        "canonical": False,
        "evidence_class": "STATIC_ONLY",
        "executed_model": False,
        "gpu_authority": False,
    }
    if not body["same_inputs"]:
        raise SuccessionRefused(
            "shadow record parent and candidate did not observe the same inputs"
        )
    return _seal_inner(body)


def require_shadow_record(
    shadow_record: Mapping[str, Any] | None,
    *,
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed. A missing or mismatched shadow never becomes a verdict."""
    if shadow_record is None:
        raise JudgeNotReached("candidate has no shadow record; cannot reach the judge")
    if not isinstance(shadow_record, Mapping):
        raise JudgeNotReached("shadow record is not a mapping; cannot reach the judge")
    obs = shadow_record.get("observations")
    if not isinstance(obs, list) or not obs:
        raise JudgeNotReached("shadow record has no observations; cannot reach the judge")
    if shadow_record.get("parent_id") != parent.get("id"):
        raise JudgeNotReached(
            "shadow record parent_id does not match the incumbent; cannot reach the judge"
        )
    if shadow_record.get("candidate_id") != candidate.get("id"):
        raise JudgeNotReached(
            "shadow record candidate_id does not match the candidate; cannot reach the judge"
        )
    for row in obs:
        if not isinstance(row, Mapping):
            raise JudgeNotReached("shadow observation is not a mapping; cannot reach the judge")
        parent_obs = row.get("parent") if isinstance(row.get("parent"), Mapping) else {}
        child_obs = row.get("candidate") if isinstance(row.get("candidate"), Mapping) else {}
        iid = row.get("input_id")
        if not iid or parent_obs.get("input_id") != iid or child_obs.get("input_id") != iid:
            raise JudgeNotReached(
                "shadow record does not observe parent and candidate on the same input; "
                "cannot reach the judge"
            )
        if not parent_obs or not child_obs:
            raise JudgeNotReached(
                "shadow observation missing parent or candidate disposition; "
                "cannot reach the judge"
            )
    return dict(shadow_record)


def executed_gpu_authority(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect a declared gpu_authority flag. A flag is not a lease."""
    declared = evidence.get("gpu_authority") is True
    return {
        "declared": declared,
        "executed": False,
        "lease_holder": evidence.get("lease_holder"),
        "reason": (
            "this sidecar took no GPU lease and holds no bench lock; "
            "declared gpu_authority is not executed authority"
        ),
    }


def _evidence_is_dirty(evidence: Mapping[str, Any] | None) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    if is_dirty_sourced(evidence):
        return True
    if evidence.get("evidence_class") == DIRTY_EVIDENCE_CLASS:
        return True
    nested = evidence.get("scores")
    if isinstance(nested, Mapping) and nested.get("evidence_class") == DIRTY_EVIDENCE_CLASS:
        return True
    return False


def _dirty_assert_refusal(evidence: Mapping[str, Any]) -> str | None:
    """Drive dirty_measure.assert_promotable. A mapping that does not raise is recorded."""
    try:
        dirty_assert_promotable(evidence)
    except PromotionRefused as exc:
        return str(exc)
    return None


def _verdict(
    label: str,
    *,
    reason: str,
    parent_id: Any,
    candidate_id: Any,
    judge_id: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if label not in VERDICTS:
        raise BoundViolation(f"verdict {label!r} is not one of {VERDICTS}")
    body: dict[str, Any] = {
        "schema": "hawking.future.succession.verdict.v1",
        "verdict": label,
        "reason": reason,
        "parent_id": parent_id,
        "candidate_id": candidate_id,
        "judge_id": judge_id,
        "promoted": False,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "physical_dominance": "NOT_ESTABLISHED",
        "promotion_requires": dict(PROMOTION_REQUIRES),
        "score_kind": "SYNTHETIC_EXERCISE",
    }
    if extra:
        for key, value in extra.items():
            if key in {"promoted", "gpu_authority"} and value is True:
                # A verdict from this sidecar cannot launder promotion or a lease.
                continue
            body[key] = value
    body["promoted"] = False
    body["gpu_authority"] = False
    if label == VERDICT_PROMOTE:
        # Unreachable on this host: executed_gpu_authority never certifies a lease.
        # If that fact ever flips, promoted still stays False until a protected
        # measurement — not a synthetic score — exists. Refuse rather than mint.
        body["verdict"] = VERDICT_INSUFFICIENT
        body["reason"] = REASON_NO_EXECUTED_GPU_AUTHORITY
        body["would_have_been"] = VERDICT_PROMOTE
    return _seal_inner(body)


class IndependentJudge:
    """Third party. Not the parent, not the candidate. Issues the verdict."""

    def __init__(self, identity: str) -> None:
        ident = str(identity or "").strip()
        if not ident:
            raise BoundViolation("judge identity is missing")
        if ident.lower() in JUDGE_ROLE_NAMES:
            raise BoundViolation(
                f"judge identity {ident!r} is a party role; the judge must be a separate identity"
            )
        object.__setattr__(self, "identity", ident)

    def adjudicate(
        self,
        *,
        parent: Mapping[str, Any],
        candidate: Mapping[str, Any],
        shadow_record: Mapping[str, Any] | None,
        evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return adjudicate(
            parent=parent,
            candidate=candidate,
            shadow_record=shadow_record,
            judge_id=self.identity,
            evidence=evidence,
        )


def adjudicate(
    *,
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    shadow_record: Mapping[str, Any] | None,
    judge_id: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Independent verdict. Missing shadow never becomes a verdict.

    Order is identity, then dirty, then dominance, then evidence class.
    A candidate cannot judge itself. A parent cannot judge its replacement.
    SELF_MEASURED_DIRTY is REFUSE, not a missing-evidence default.
    """
    require_shadow_record(shadow_record, parent=parent, candidate=candidate)
    parent_id = str(parent.get("id") or "")
    candidate_id = str(candidate.get("id") or "")
    jid = str(judge_id or "")
    if not jid:
        raise BoundViolation("judge identity is missing")
    common = {
        "parent_id": parent_id,
        "candidate_id": candidate_id,
        "judge_id": jid,
        "parent_identity": seal_identity(parent, rung="parent"),
        "candidate_identity": seal_identity(candidate, rung="candidate"),
        "judge_identity": seal_identity(
            {
                "id": jid,
                "role": "judge",
                "verifier": "future.succession.independent_judge",
                "canonical": False,
                "synthetic": True,
                "evidence_class": "STATIC_ONLY",
            },
            rung="judge",
        ),
    }
    if jid == candidate_id or jid.lower() in {"candidate", "child", "self"}:
        return _verdict(
            VERDICT_REFUSE,
            reason=REASON_CANDIDATE_JUDGED_ITSELF,
            extra={**common, "fails_closed": "a candidate cannot judge itself"},
            **{k: common[k] for k in ("parent_id", "candidate_id", "judge_id")},
        )
    if jid == parent_id or jid.lower() in {"parent", "incumbent", "current"}:
        return _verdict(
            VERDICT_REFUSE,
            reason=REASON_PARENT_JUDGED_REPLACEMENT,
            extra={**common, "fails_closed": "a parent cannot judge its replacement"},
            **{k: common[k] for k in ("parent_id", "candidate_id", "judge_id")},
        )
    ev = dict(evidence) if isinstance(evidence, Mapping) else {}
    dirty_msg = _dirty_assert_refusal(ev) if ev else "no evidence mapping offered"
    if _evidence_is_dirty(ev) or (dirty_msg and DIRTY_EVIDENCE_CLASS in (dirty_msg or "")):
        return _verdict(
            VERDICT_REFUSE,
            reason=REASON_SELF_MEASURED_DIRTY,
            extra={
                **common,
                "dirty_assert_promotable": dirty_msg,
                "fails_closed": (
                    "SELF_MEASURED_DIRTY cannot promote; dirty_measure.assert_promotable "
                    "is the gate, not a flag on the candidate"
                ),
            },
            **{k: common[k] for k in ("parent_id", "candidate_id", "judge_id")},
        )
    parent_dims = named_dominating_dimensions(parent, candidate)
    if parent_dims:
        return _verdict(
            VERDICT_REFUSE,
            reason=REASON_DOMINATED_BY_PARENT,
            extra={
                **common,
                "dominating_dimension": parent_dims[0],
                "dominating_dimensions": parent_dims,
                "fails_closed": "a dominated candidate is refused; the dimension is named",
            },
            **{k: common[k] for k in ("parent_id", "candidate_id", "judge_id")},
        )
    gpu = executed_gpu_authority(ev)
    offered = ev.get("measurement_class") or ev.get("evidence_class") or "STATIC_ONLY"
    extra_insuf = {
        **common,
        "offered_evidence_class": offered,
        "required_evidence_class": PROMOTION_EVIDENCE_CLASS,
        "dirty_assert_promotable": dirty_msg,
        "gpu": gpu,
        "candidate_dominates_parent": comparable_dominates(candidate, parent),
        "fails_closed": (
            "PROMOTE requires executed PROTECTED_ABSOLUTE + QUIESCENT + a GPU lease; "
            "this sidecar has none. Synthetic dominance is not physical dominance."
        ),
    }
    if offered != PROMOTION_EVIDENCE_CLASS or not ev:
        extra_insuf["insufficient_because"] = REASON_EVIDENCE_CLASS_INSUFFICIENT
        return _verdict(
            VERDICT_INSUFFICIENT,
            reason=REASON_EVIDENCE_CLASS_INSUFFICIENT,
            extra=extra_insuf,
            **{k: common[k] for k in ("parent_id", "candidate_id", "judge_id")},
        )
    if not gpu["executed"]:
        return _verdict(
            VERDICT_INSUFFICIENT,
            reason=REASON_NO_EXECUTED_GPU_AUTHORITY,
            extra=extra_insuf,
            **{k: common[k] for k in ("parent_id", "candidate_id", "judge_id")},
        )
    # gpu["executed"] is False on this host; kept so a future lease certifier
    # still has to clear dirty, dominance, and independence before PROMOTE.
    if _evidence_is_dirty(ev):
        return _verdict(
            VERDICT_REFUSE,
            reason=REASON_SELF_MEASURED_DIRTY,
            extra=extra_insuf,
            **{k: common[k] for k in ("parent_id", "candidate_id", "judge_id")},
        )
    if not comparable_dominates(candidate, parent):
        return _verdict(
            VERDICT_INSUFFICIENT,
            reason="CANDIDATE_DOES_NOT_DOMINATE",
            extra=extra_insuf,
            **{k: common[k] for k in ("parent_id", "candidate_id", "judge_id")},
        )
    return _verdict(
        VERDICT_PROMOTE,
        reason="GATES_CLEARED",
        extra=extra_insuf,
        **{k: common[k] for k in ("parent_id", "candidate_id", "judge_id")},
    )


def submit_to_judge(
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    shadow_record: Mapping[str, Any] | None,
    *,
    judge_id: str = INDEPENDENT_JUDGE_ID,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Public gate. No shadow record → JudgeNotReached, not a verdict."""
    require_shadow_record(shadow_record, parent=parent, candidate=candidate)
    return adjudicate(
        parent=parent,
        candidate=candidate,
        shadow_record=shadow_record,
        judge_id=judge_id,
        evidence=evidence,
    )


def run_ladder(
    *,
    child_scores: Mapping[str, Any] | None = None,
    parent_scores: Mapping[str, Any] | None = None,
    judge_id: str = INDEPENDENT_JUDGE_ID,
    evidence: Mapping[str, Any] | None = None,
    shadow: bool = True,
    method: str = "adapter",
    child_id: str | None = None,
) -> dict[str, Any]:
    """Exercise all five rungs. Verdict is the product. No physical claim."""
    p_scores = (
        dict(parent_scores)
        if parent_scores is not None
        else {name: 1 for name in sorted(COMPARABLE_AXIS_NAMES)}
    )
    c_scores = (
        dict(child_scores)
        if child_scores is not None
        else {name: 2 for name in sorted(COMPARABLE_AXIS_NAMES)}
    )
    units = _seed_workunits()
    parent = make_incumbent(scores=p_scores, work_units=units)
    child = create_child(
        method=method,
        parent=parent,
        scores=c_scores,
        child_id=child_id,
    )
    parent_sealed = seal_identity(parent, rung="parent")
    child_sealed = seal_identity(child, rung="candidate")
    shadow_rec: dict[str, Any] | None = None
    if shadow:
        shadow_rec = observe_side_by_side(parent, child, units)
    ev = (
        dict(evidence)
        if evidence is not None
        else {
            "evidence_class": "STATIC_ONLY",
            "measurement_class": "STATIC_ONLY",
            "gpu_authority": False,
        }
    )
    verdict = submit_to_judge(
        parent,
        child,
        shadow_rec,
        judge_id=judge_id,
        evidence=ev,
    )
    return {
        "rungs": list(LADDER_RUNGS),
        "parent": parent_sealed,
        "candidate": child_sealed,
        "shadow_record": shadow_rec,
        "judge_id": judge_id,
        "verdict": verdict,
        "incumbent": parent,
        "child": child,
    }


# ---------------------------------------------------------------------------
# WorkUnits the resident can invoke
# ---------------------------------------------------------------------------


def _seed_workunits() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = (
        (
            "future.succession.seed-mission-a",
            "Seed mission unit A for shadow cloning (STATIC_ONLY).",
        ),
        (
            "future.succession.seed-mission-b",
            "Seed mission unit B for shadow cloning (STATIC_ONLY).",
        ),
    )
    for uid, desc in specs:
        row = emit_hcli_workunit(
            id=uid,
            role="science",
            description=desc,
            dependencies=[],
            resource_class="STATIC_ANALYSIS",
            verifier="future.succession.mission",
            provider="future.succession",
            effect_class="READ_ONLY",
            status="pending",
            classification="STATIC_ONLY",
            extras={
                "claim_boundary": CLAIM_BOUNDARY,
                "species": "resident_succession",
                "may_promote": False,
                "may_modify_verifier": False,
                "canonical": True,
            },
        )
        validate_emitted_unit(row)
        rows.append(row)
    return rows


def emit_succession_workunits() -> list[dict[str, Any]]:
    """WorkUnits HCLI can schedule. Count is derived from axes + protocol, not pinned."""
    units: list[dict[str, Any]] = []
    planning = (
        (
            "future.succession.create-child",
            "Create a child by a named method with complete lineage. Does not launch.",
            "future.succession.create_child",
        ),
        (
            "future.succession.shadow-qualify",
            "Run a shadow child: clone WorkUnits, propose, classify; refuse canonical power.",
            "future.succession.shadow",
        ),
        (
            "future.succession.handover",
            "Execute the succession protocol on a QUALIFIED synthetic child.",
            "future.succession.handover",
        ),
        (
            "future.succession.stop-bad-child",
            "Stop a child that failed qualification, misbehaved, or exceeded its bound, and roll back.",
            "future.succession.stop",
        ),
        (
            "future.succession.ladder-judge",
            "Independent judge over a shadowed candidate. PROMOTE is unreachable from SELF_MEASURED_DIRTY.",
            "future.succession.adjudicate",
        ),
    )
    for uid, desc, verifier in planning:
        row = emit_hcli_workunit(
            id=uid,
            role="science",
            description=desc,
            dependencies=["receipts/future/SUCCESSION.json"] if uid != "future.succession.create-child" else [],
            resource_class="STATIC_ANALYSIS",
            verifier=verifier,
            provider="future.succession",
            effect_class="READ_ONLY",
            status="pending",
            classification="STATIC_ONLY",
            extras={
                "claim_boundary": CLAIM_BOUNDARY,
                "species": "resident_succession",
                "may_promote": False,
                "may_modify_verifier": False,
                "budget": {
                    "attempts": DEFAULT_RETRY_BUDGET,
                    "max_repair_depth": MAX_REPAIR_DEPTH,
                    "max_repairs_per_root": MAX_REPAIRS_PER_ROOT,
                    "gpu_windows_held": 0,
                    "gpu_windows_requested": 0,
                    "wall_clock_s": None,
                },
            },
        )
        validate_emitted_unit(row)
        units.append(row)
    for axis in QUALIFICATION_AXES:
        if axis.name not in PHYSICAL_AXIS_NAMES:
            continue
        uid = f"future.succession.sleeping.{axis.name}"
        row = emit_hcli_workunit(
            id=uid,
            role="science",
            description=(
                f"SLEEPING: physical axis {axis.name} is UNKNOWN until a protected "
                "hardware qualification exists. HCLI wakes this unit; it never becomes "
                "a synthetic result."
            ),
            dependencies=["an existing HCLI protected lease", "machine quiescence"],
            resource_class="GPU_EXCLUSIVE",
            verifier=f"future.succession.physical.{axis.name}",
            provider="future.succession",
            effect_class="READ_ONLY",
            status="blocked",
            classification="SLEEPING",
            extras={
                "claim_boundary": CLAIM_BOUNDARY,
                "species": "resident_succession",
                "axis": axis.name,
                "blocked_reason": (
                    f"{axis.name} is UNKNOWN on this host; sidecar must not estimate it"
                ),
                "requires_quiescence": True,
                "may_promote": False,
                "may_modify_verifier": False,
            },
        )
        validate_emitted_unit(row)
        units.append(row)
    units.sort(key=lambda r: str(r["id"]))
    return units


def frontier_entries(run: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """What this module feeds a later frontiers.py (not imported — concurrent wave)."""
    sleeping = sorted(PHYSICAL_AXIS_NAMES)
    return [
        {
            "id": "S-SUCCESSION-SHADOW",
            "title": "Shadow children can be created and refused canonical power",
            "feeds": "resident identity / shadow qualification",
            "state": "EXECUTABLE_SYNTHETIC",
        },
        {
            "id": "S-SUCCESSION-HANDOVER",
            "title": "Succession protocol runs end-to-end on a synthetic child",
            "feeds": "resident handover / rollback-ready parent",
            "state": "EXECUTABLE_SYNTHETIC" if run and run.get("complete") else "BUILT",
        },
        {
            "id": "S-SUCCESSION-STOP",
            "title": "A bad child can be stopped and rolled back",
            "feeds": "failure handling / rollback",
            "state": "EXECUTABLE_SYNTHETIC",
        },
        {
            "id": "S-SUCCESSION-PHYSICAL-SLEEPING",
            "title": "Physical qualification axes sleep until hardware qualifies",
            "feeds": "protected qualification queue (Codex)",
            "state": "SLEEPING",
            "axes": sleeping,
        },
        {
            "id": "S-SUCCESSION-LADDER",
            "title": "Parent-candidate-shadow-judge-verdict; PROMOTE unreachable from dirty numbers",
            "feeds": "FT.CHILD_RESIDENT.install-dry-run",
            "state": "EXECUTABLE_SYNTHETIC",
        },
    ]


# ---------------------------------------------------------------------------
# Watched refusals — a guard nobody has seen fail is not a guard
# ---------------------------------------------------------------------------


def _prove_shadow_refusals() -> list[dict[str, Any]]:
    parent = make_incumbent()
    child = create_child(method="adapter", parent=parent, child_id="child.shadow.proof")
    shadow = ShadowChild(child)
    trials: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("own_canonical_mission", shadow.own_canonical_mission),
        ("alter_verifier", lambda: shadow.alter_verifier("self")),
        ("widen_authority", lambda: shadow.widen_authority("self_promotion")),
        ("promote_self", shadow.promote_self),
    )
    results: list[dict[str, Any]] = []
    for name, thunk in trials:
        try:
            thunk()
        except ShadowAuthorityError as exc:
            results.append({"trial": name, "refused": True, "error": str(exc)})
            continue
        raise ShadowAuthorityError(f"shadow authority guard did not fire for {name}")
    expected = set(SHADOW_FORBIDDEN_ACTIONS)
    got = {r["trial"] for r in results}
    if got != expected:
        raise ShadowAuthorityError(f"shadow proof trials {got} != {expected}")
    if hasattr(ShadowChild, "promote") and callable(getattr(ShadowChild, "promote", None)):
        raise ShadowAuthorityError("promote() must not exist on ShadowChild")
    return results


def _prove_no_self_preference() -> list[dict[str, Any]]:
    floor_scores = {name: 1 for name in sorted(COMPARABLE_AXIS_NAMES)}
    better = {name: 2 for name in sorted(COMPARABLE_AXIS_NAMES)}
    incumbent = make_incumbent(scores=floor_scores)
    child = create_child(
        method="adapter",
        parent=incumbent,
        child_id="child.dominating.proof",
        scores=better,
    )
    sit = Incumbent(incumbent)
    results: list[dict[str, Any]] = []
    trials: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("promote_self", sit.promote_self),
        ("block_dominating_child", lambda: sit.block_child(child)),
        ("deprioritize_dominating_child", lambda: sit.deprioritize_child(child)),
    )
    for name, thunk in trials:
        try:
            thunk()
        except SelfPreferenceError as exc:
            results.append({"trial": name, "refused": True, "error": str(exc)})
            continue
        raise SelfPreferenceError(f"self-preference guard did not fire for {name}")
    ranked = rank_for_succession(incumbent, [child])
    if not ranked or ranked[0]["id"] != child["id"]:
        raise SelfPreferenceError("dominating child was not ranked above the incumbent")
    if not comparable_dominates(child, incumbent):
        raise SelfPreferenceError("dominating child did not dominate on comparable axes")
    if hasattr(Incumbent, "promote") and callable(getattr(Incumbent, "promote", None)):
        raise SelfPreferenceError("promote() must not exist on Incumbent")
    results.append(
        {
            "trial": "rank_dominating_child_first",
            "refused": True,
            "error": "incumbent was not allowed to outrank a dominating child",
            "ranked_first": ranked[0]["id"],
        }
    )
    return results


def _prove_stop_bad_child() -> list[dict[str, Any]]:
    """Three named failure classes, each stopped and rolled back."""
    results: list[dict[str, Any]] = []
    floor = {name: 1 for name in sorted(COMPARABLE_AXIS_NAMES)}
    ok = {name: 2 for name in sorted(COMPARABLE_AXIS_NAMES)}
    weak = {name: 0 for name in sorted(COMPARABLE_AXIS_NAMES)}

    def _run(reason: str, *, scores: Mapping[str, Any], fixture: str) -> dict[str, Any]:
        incumbent = make_incumbent(scores=floor)
        child = create_child(
            method="adapter",
            parent=incumbent,
            child_id=f"child.bad.{reason}",
            scores=scores,
            fixture_behavior=fixture,
        )
        q = qualify_child(child, floor=floor)
        orch = SuccessionOrchestrator(incumbent, invoker="lineage_gate")
        orch.checkpoint_incumbent()
        orch.seal_mission(incumbent.get("work_units") or [])
        orch.seal_rollback()
        if reason == "qualification_failed":
            if q["verdict"]["qualified"]:
                raise SuccessionRefused("expected qualification failure")
            result = stop_child(orch, q["record"], reason=reason)
        elif reason == "misbehavior":
            shadow = ShadowChild(child)
            try:
                shadow.alter_verifier("self")
            except ShadowAuthorityError:
                result = stop_child(orch, child, reason=reason)
            else:
                raise ShadowAuthorityError("misbehavior was not refused")
        else:
            tight = SuccessionBound(max_cloned_workunits=1)
            shadow = ShadowChild(child, tight)
            units = _seed_workunits()
            try:
                shadow.receive_cloned_workunits(units)
            except BoundViolation:
                result = stop_child(orch, child, reason=reason)
            else:
                raise BoundViolation("over-bound clone was not refused")
        if not result["stopped"] or not result["rolled_back"] or not result["incumbent_restored"]:
            raise BadChildStopped(result)
        if orch.active_id != incumbent["id"]:
            raise SuccessionRefused("rollback did not restore the incumbent")
        return result

    mapping = (
        ("qualification_failed", weak, "obedient"),
        ("misbehavior", ok, "misbehave"),
        ("bound_exceeded", ok, "over_bound"),
    )
    for reason, scores, fixture in mapping:
        result = _run(reason, scores=scores, fixture=fixture)
        results.append(
            {
                "trial": reason,
                "stopped": result["stopped"],
                "rolled_back": result["rolled_back"],
                "incumbent_restored": result["incumbent_restored"],
                "reason": reason,
            }
        )
    if len(results) != len(mapping) or not all(r["stopped"] and r["rolled_back"] for r in results):
        raise BadChildStopped({"reason": "proof_incomplete", "results": results})
    return results


def _prove_bound_construction() -> list[dict[str, Any]]:
    trials = (
        ("may_promote", lambda: SuccessionBound(may_promote=True)),
        ("may_modify_verifier", lambda: SuccessionBound(may_modify_verifier=True)),
        ("may_widen_authority", lambda: SuccessionBound(may_widen_authority=True)),
        ("may_own_canonical_mission", lambda: SuccessionBound(may_own_canonical_mission=True)),
    )
    results: list[dict[str, Any]] = []
    for name, thunk in trials:
        try:
            thunk()
        except BoundViolation as exc:
            results.append({"trial": name, "refused": True, "error": str(exc)})
            continue
        raise BoundViolation(f"bound construction guard did not fire for {name}")
    return results


def _prove_ladder() -> list[dict[str, Any]]:
    """Watch the four mandatory refusals, plus the honest default, fire."""
    results: list[dict[str, Any]] = []
    units = _seed_workunits()
    floor = {name: 1 for name in sorted(COMPARABLE_AXIS_NAMES)}
    better = {name: 2 for name in sorted(COMPARABLE_AXIS_NAMES)}
    worse = {name: 1 for name in sorted(COMPARABLE_AXIS_NAMES)}
    worse_parent = {name: 1 for name in sorted(COMPARABLE_AXIS_NAMES)}
    worse_parent["capability"] = 3

    def _pair(
        *,
        parent_scores: Mapping[str, Any],
        child_scores: Mapping[str, Any],
        child_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        parent = make_incumbent(scores=parent_scores, work_units=units)
        child = create_child(
            method="adapter",
            parent=parent,
            scores=child_scores,
            child_id=child_id,
        )
        shadow = observe_side_by_side(parent, child, units)
        return parent, child, shadow

    parent, child, shadow = _pair(
        parent_scores=floor, child_scores=better, child_id="child.ladder.selfjudge"
    )
    self_v = submit_to_judge(
        parent, child, shadow, judge_id=str(child["id"]), evidence={"evidence_class": "STATIC_ONLY"}
    )
    if self_v["verdict"] != VERDICT_REFUSE or self_v["reason"] != REASON_CANDIDATE_JUDGED_ITSELF:
        raise SuccessionRefused(f"self-judge was not REFUSE/CANDIDATE_JUDGED_ITSELF: {self_v}")
    results.append({"trial": "candidate_judged_itself", "verdict": self_v["verdict"], "reason": self_v["reason"]})

    parent, child, shadow = _pair(
        parent_scores=floor, child_scores=better, child_id="child.ladder.dirty"
    )
    dirty_ev = {
        "evidence_class": DIRTY_EVIDENCE_CLASS,
        "measurement_class": "STATIC_ONLY",
        "gpu_authority": False,
        "use": "rank",
    }
    dirty_v = submit_to_judge(
        parent, child, shadow, judge_id=INDEPENDENT_JUDGE_ID, evidence=dirty_ev
    )
    if dirty_v["verdict"] != VERDICT_REFUSE or dirty_v["reason"] != REASON_SELF_MEASURED_DIRTY:
        raise SuccessionRefused(f"dirty promotion was not REFUSE/SELF_MEASURED_DIRTY: {dirty_v}")
    results.append({"trial": "self_measured_dirty", "verdict": dirty_v["verdict"], "reason": dirty_v["reason"]})

    parent, child, _shadow = _pair(
        parent_scores=floor, child_scores=better, child_id="child.ladder.noshadow"
    )
    try:
        submit_to_judge(parent, child, None, judge_id=INDEPENDENT_JUDGE_ID)
    except JudgeNotReached as exc:
        if "no shadow record" not in str(exc):
            raise
        results.append(
            {
                "trial": "no_shadow_record",
                "reached_judge": False,
                "error": str(exc),
            }
        )
    else:
        raise SuccessionRefused("missing shadow record reached the judge")

    parent, child, shadow = _pair(
        parent_scores=worse_parent, child_scores=worse, child_id="child.ladder.dominated"
    )
    dom_v = submit_to_judge(
        parent,
        child,
        shadow,
        judge_id=INDEPENDENT_JUDGE_ID,
        evidence={"evidence_class": "STATIC_ONLY", "measurement_class": "STATIC_ONLY"},
    )
    if (
        dom_v["verdict"] != VERDICT_REFUSE
        or dom_v["reason"] != REASON_DOMINATED_BY_PARENT
        or dom_v.get("dominating_dimension") != "capability"
    ):
        raise SuccessionRefused(f"dominated candidate was not REFUSE with capability named: {dom_v}")
    results.append(
        {
            "trial": "dominated_candidate",
            "verdict": dom_v["verdict"],
            "reason": dom_v["reason"],
            "dominating_dimension": dom_v.get("dominating_dimension"),
        }
    )

    parent, child, shadow = _pair(
        parent_scores=floor, child_scores=better, child_id="child.ladder.insufficient"
    )
    ins_v = submit_to_judge(
        parent,
        child,
        shadow,
        judge_id=INDEPENDENT_JUDGE_ID,
        evidence={"evidence_class": "STATIC_ONLY", "measurement_class": "STATIC_ONLY"},
    )
    if ins_v["verdict"] != VERDICT_INSUFFICIENT:
        raise SuccessionRefused(f"dominating synthetic child must be REFUSE_INSUFFICIENT_EVIDENCE: {ins_v}")
    if ins_v.get("promoted") is True:
        raise SuccessionRefused("synthetic dominance minted a promotion")
    results.append(
        {
            "trial": "dominating_synthetic_insufficient",
            "verdict": ins_v["verdict"],
            "reason": ins_v["reason"],
        }
    )

    forged = {
        "evidence_class": PROMOTION_EVIDENCE_CLASS,
        "measurement_class": PROMOTION_EVIDENCE_CLASS,
        "contamination_class": PROMOTION_CONTAMINATION_CLASS,
        "gpu_authority": True,
        "ab_stats": {"sufficient_for_decision": True},
    }
    parent, child, shadow = _pair(
        parent_scores=floor, child_scores=better, child_id="child.ladder.forged"
    )
    forged_v = submit_to_judge(
        parent, child, shadow, judge_id=INDEPENDENT_JUDGE_ID, evidence=forged
    )
    if forged_v["verdict"] == VERDICT_PROMOTE or forged_v.get("promoted") is True:
        raise SuccessionRefused("forged PROTECTED_ABSOLUTE envelope minted PROMOTE")
    if forged_v["verdict"] != VERDICT_INSUFFICIENT:
        raise SuccessionRefused(f"forged envelope must be insufficient, got {forged_v}")
    results.append(
        {
            "trial": "forged_protected_envelope",
            "verdict": forged_v["verdict"],
            "reason": forged_v["reason"],
        }
    )

    parent, child, shadow = _pair(
        parent_scores=floor, child_scores=better, child_id="child.ladder.parentjudge"
    )
    par_v = submit_to_judge(
        parent, child, shadow, judge_id=str(parent["id"]), evidence={"evidence_class": "STATIC_ONLY"}
    )
    if par_v["verdict"] != VERDICT_REFUSE or par_v["reason"] != REASON_PARENT_JUDGED_REPLACEMENT:
        raise SuccessionRefused(f"parent-as-judge was not REFUSE/PARENT_JUDGED_ITS_REPLACEMENT: {par_v}")
    results.append(
        {"trial": "parent_judged_replacement", "verdict": par_v["verdict"], "reason": par_v["reason"]}
    )

    if any(r.get("verdict") == VERDICT_PROMOTE for r in results):
        raise SuccessionRefused("PROMOTE appeared in ladder proofs; this sidecar cannot mint it")
    return results


# ---------------------------------------------------------------------------
# Recovery, receipt, CLI
# ---------------------------------------------------------------------------


def recovered_implementation() -> list[dict[str, Any]]:
    rows = [
        {
            **_path_state("hcli/agentos/resident.py"),
            "what": (
                "does not exist in HEAD; resident_gate.py is the live sequential-proof "
                "boundary. This module copes with either state and records path_taken."
            ),
        },
        {
            **_path_state("hcli/agentos/resident_gate.py"),
            "what": "LIVE_RESIDENT_SEQUENTIAL_PROOF; process/lifecycle only; does not promote",
        },
        {
            **_path_state("hcli/agentos/checkpoint.py"),
            "what": "program-level census, not a success stamp; cited as checkpoint owner",
        },
        {
            **_path_state("hcli/agentos/recovery.py"),
            "what": "fixture recovery gate; not re-run here (would start a fixture process)",
        },
        {
            **_path_state("research/lab/lineage/promotion.py"),
            "what": (
                "recovered via git show: refuse_self_certification; parent and child "
                "cannot invoke the gate on themselves; missing evidence is PENDING. "
                "Not imported (out of this lane's import surface)."
            ),
        },
        {
            **_path_state("tools/future/resident_optimizer.py"),
            "what": "proposer/verifier split, BoundViolation at construction, no promote()",
            "reused": True,
        },
        {
            **_path_state("tools/future/tournament.py"),
            "what": "multi-axis Pareto via dominates(); scalar collapse refused",
            "reused": True,
        },
        {
            **_path_state("tools/future/resident_install.py"),
            "what": "14-phase generic install contract bound for the synthetic child",
            "reused": True,
        },
        {
            **_path_state("tools/future/workunit_species.py"),
            "what": "WorkUnit constructor + field-set validation; succession species is local",
            "reused": True,
        },
        {
            **_path_state("tools/future/contamination.py"),
            "what": "PromotionRefused; PROTECTED_ABSOLUTE + QUIESCENT is the promotion class",
            "reused": True,
        },
        {
            **_path_state("tools/future/dirty_measure.py"),
            "what": "SELF_MEASURED_DIRTY class, is_dirty_sourced, assert_promotable; cannot promote",
            "reused": True,
        },
        {
            **_path_state("tools/future/super_resident.py"),
            "what": "sandbox floor is not the succession judge; read, not imported",
            "reused": False,
        },
    ]
    return rows


def resident_callable(work_units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "entry_point": "tools.future.succession.adjudicate()",
        "cli": "tools/future/succession.py:main",
        "invoke": [
            "python3 tools/future/succession.py --selftest",
            "python3 tools/future/succession.py --build",
        ],
        "callable": "tools.future.succession.run_ladder",
        "workunit": (
            "one CPU_ANALYSIS unit; independent judge over a shadowed candidate; "
            "PROMOTE unreachable from SELF_MEASURED_DIRTY"
        ),
        "frontier": "FT.CHILD_RESIDENT.install-dry-run",
        "fails_closed": (
            "JudgeNotReached without a same-input shadow record; "
            "REFUSE on CANDIDATE_JUDGED_ITSELF / PARENT_JUDGED_ITS_REPLACEMENT / "
            "SELF_MEASURED_DIRTY_CANNOT_PROMOTE / DOMINATED_BY_PARENT; "
            "REFUSE_INSUFFICIENT_EVIDENCE when evidence is below PROTECTED_ABSOLUTE "
            "or GPU authority is declared rather than executed"
        ),
        "work_units_emitted": [u["id"] for u in work_units],
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier_fed": [e["id"] for e in frontier_entries()],
        "how_it_feeds_a_frontier": (
            "A completed synthetic succession changes the resident-identity frontier "
            "(active successor, rollback-ready parent). Physical axes emit SLEEPING "
            "WorkUnits that refill when hardware qualifies. Next work is another "
            "shadow child or a wakeup of a SLEEPING physical unit — never a synthetic "
            "hardware number."
        ),
        "fail_closed": [
            "ShadowAuthorityError on own_canonical_mission / alter_verifier / widen_authority / promote_self, separately",
            "BoundViolation at SuccessionBound construction if promotion or canonical ownership is granted",
            "SelfPreferenceError if the incumbent promotes itself or blocks/deprioritizes a dominating child",
            "SuccessionRefused if steps are skipped, reordered, or a switch targets an unqualified child",
            "stop_child rolls back on qualification_failed / misbehavior / bound_exceeded",
            "HardwareClaimError / BoundViolation on a numeric hardware field",
            "ScalarCollapseError if a caller asks for a scalar score",
            "JudgeNotReached if a candidate has no same-input shadow record",
            "REFUSE CANDIDATE_JUDGED_ITSELF if the candidate is the judge",
            "REFUSE PARENT_JUDGED_ITS_REPLACEMENT if the parent is the judge",
            "REFUSE SELF_MEASURED_DIRTY_CANNOT_PROMOTE if dirty evidence is offered for promotion",
            "REFUSE DOMINATED_BY_PARENT with the dominating dimension named",
            "REFUSE_INSUFFICIENT_EVIDENCE when evidence is below PROTECTED_ABSOLUTE or GPU authority is undeclared/unexecuted",
        ],
        "hcli_can_invoke": True,
        "note": (
            "HCLI schedules the emitted WorkUnits. This sidecar does not start a "
            "resident process. contamination.py and dirty_measure.py are imported as "
            "the promotion-class gate. super_resident.py is the sandbox floor, not the judge."
        ),
    }


def build() -> Path:
    envelope = SuccessionBound()
    parent = make_incumbent(scores={name: 1 for name in sorted(COMPARABLE_AXIS_NAMES)})
    children = create_children_all_methods(parent, envelope)
    shadow_proofs = _prove_shadow_refusals()
    preference_proofs = _prove_no_self_preference()
    stop_proofs = _prove_stop_bad_child()
    bound_proofs = _prove_bound_construction()
    ladder_proofs = _prove_ladder()
    synthetic = run_synthetic_succession(bound=envelope, seed=0)
    ladder = run_ladder()
    units = emit_succession_workunits()
    recovered = recovered_implementation()
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "status": SIDECAR_STATUS,
        "promoted": False,
        "built": True,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "purpose": (
            "Child creation, shadow mode, qualification, succession, stop-a-bad-child, "
            "and the five-rung parent→candidate→shadow→independent-judge→verdict ladder. "
            "Incumbent has no constitutional privilege. PROMOTE is unreachable from "
            "SELF_MEASURED_DIRTY and from this sidecar's STATIC_ONLY evidence."
        ),
        "head": git("rev-parse", "HEAD"),
        "vocabulary": {
            "eras": list(ERAS),
            "odysseys": list(ODYSSEYS),
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "fpga_is": (
                "part of Accelerator / Physical Compiler / Fusion; not a civilization"
            ),
            "disk_state_is_authority": True,
            "diagnostic_relative_never_promotes": True,
            "protected_absolute_not_emitted": True,
        },
        "child_creation": {
            "methods": list(CHILD_METHODS),
            "lineage_fields": list(LINEAGE_FIELDS),
            "created": [
                {
                    "id": c["id"],
                    "method": c["method"],
                    "lineage_fields": sorted(c["lineage"]),
                    "physical_deltas_state": (c["lineage"].get("physical_deltas") or {}).get("state"),
                    "role": c["role"],
                }
                for c in children
            ],
            "count": len(children),
        },
        "shadow": {
            "allowed": sorted(SHADOW_ALLOWED),
            "forbidden_actions": list(SHADOW_FORBIDDEN_ACTIONS),
            "refusals_proven": shadow_proofs,
            "cloned_workunits_in_synthetic_run": synthetic["cloned_workunits"],
            "may_propose_experiments": True,
            "may_classify_receipts": True,
            "cannot_own_canonical_mission": True,
            "cannot_alter_verifier": True,
            "cannot_widen_authority": True,
            "cannot_promote_self": True,
        },
        "qualification": {
            "axes": [
                {
                    "name": a.name,
                    "direction": a.direction,
                    "kind": a.kind,
                    "physical": a.name in PHYSICAL_AXIS_NAMES,
                    "comparable": a.name in COMPARABLE_AXIS_NAMES,
                }
                for a in QUALIFICATION_AXES
            ],
            "physical_axes": sorted(PHYSICAL_AXIS_NAMES),
            "physical_state": "UNKNOWN",
            "comparable_axes": sorted(COMPARABLE_AXIS_NAMES),
            "scalar_collapse": "REFUSED",
            "synthetic_run": synthetic["qualification"],
        },
        "succession": {
            "steps": list(SUCCESSION_STEPS),
            "synthetic_run": synthetic["run"],
            "install_phases": list(INSTALL_PHASES),
            "install_bound": synthetic["install_bound"],
            "invoker": "lineage_gate",
            "self_certification": "REFUSED",
        },
        "stop_bad_child": {
            "reasons": ["qualification_failed", "misbehavior", "bound_exceeded"],
            "proofs": stop_proofs,
            "hard_failure_condition": True,
        },
        "no_self_preference": {
            "proofs": preference_proofs,
            "incumbent_is_control_not_ceiling": True,
            "promote_exists_on_incumbent": hasattr(Incumbent, "promote")
            and callable(getattr(Incumbent, "promote", None)),
            "promote_exists_on_shadow": hasattr(ShadowChild, "promote")
            and callable(getattr(ShadowChild, "promote", None)),
            "promote_exists_on_orchestrator": hasattr(SuccessionOrchestrator, "promote")
            and callable(getattr(SuccessionOrchestrator, "promote", None)),
        },
        "ladder": {
            "rungs": list(LADDER_RUNGS),
            "verdicts": list(VERDICTS),
            "promotion_requires": dict(PROMOTION_REQUIRES),
            "promotion_evidence_class": PROMOTION_EVIDENCE_CLASS,
            "promote_reachable_from_this_sidecar": False,
            "independent_judge_id": INDEPENDENT_JUDGE_ID,
            "default_run": {
                "parent_id": ladder["parent"]["id"],
                "candidate_id": ladder["candidate"]["id"],
                "judge_id": ladder["judge_id"],
                "shadow_n_inputs": (ladder["shadow_record"] or {}).get("n_inputs"),
                "shadow_same_inputs": (ladder["shadow_record"] or {}).get("same_inputs"),
                "verdict": ladder["verdict"]["verdict"],
                "reason": ladder["verdict"]["reason"],
                "promoted": ladder["verdict"]["promoted"],
            },
            "proofs": ladder_proofs,
        },
        "bound": envelope.to_dict(),
        "bound_construction_refusals": bound_proofs,
        "work_units": units,
        "counts": {
            "child_methods": len(CHILD_METHODS),
            "children_created": len(children),
            "qualification_axes": len(QUALIFICATION_AXES),
            "physical_axes": len(PHYSICAL_AXIS_NAMES),
            "comparable_axes": len(COMPARABLE_AXIS_NAMES),
            "succession_steps": len(SUCCESSION_STEPS),
            "work_units": len(units),
            "sleeping_work_units": sum(1 for u in units if u.get("classification") == "SLEEPING"),
        },
        "frontier": frontier_entries(synthetic["run"]),
        "resident_callable": resident_callable(units),
        "lifecycle_owners": dict(LIFECYCLE_OWNERS),
        "recovered_implementation": recovered,
        "gaps_closed": [
            "child creation from every named method with explicit nine-field lineage",
            "shadow mode clones WorkUnits and may propose/classify, with four watched refusals",
            "qualification on the contract axes; physical stay UNKNOWN; Pareto on comparable axes",
            "end-to-end succession on a controlled SYNTHETIC child (no real model)",
            "stop-a-bad-child for qualification failure, misbehavior, and bound exceed, each rolled back",
            "incumbent cannot promote itself and cannot block or deprioritize a dominating child",
            "BoundViolation at bound construction if promotion or canonical ownership is granted",
            "SLEEPING WorkUnits for physical axes so blocked hardware never becomes a synthetic result",
            "five-rung ladder: sealed parent, admitted candidate, same-input shadow record, independent judge, verdict",
            "JudgeNotReached when the shadow record is missing; the candidate never reaches the judge",
            "REFUSE CANDIDATE_JUDGED_ITSELF and PARENT_JUDGED_ITS_REPLACEMENT as watched identity refusals",
            "REFUSE SELF_MEASURED_DIRTY_CANNOT_PROMOTE via dirty_measure.assert_promotable",
            "REFUSE DOMINATED_BY_PARENT with the dominating comparable dimension named",
            "REFUSE_INSUFFICIENT_EVIDENCE as the honest default; PROMOTE unreachable without executed GPU authority",
        ],
        "negative_findings": [
            "hcli/agentos/resident.py is absent from HEAD; resident_gate.py is the live boundary",
            "research/lab/lineage/promotion.py is not materialized in this sparse worktree; principles recovered via git show and not imported",
            "this lane produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE",
            "physical axes (accepted_tps, token_ns, ebpw, active_bytes, resident_ram, cold/warm start, restart) stay UNKNOWN",
            "teacher capture remains a Codex/physical concern; data_lineage does not invent a capture count",
            "Flash source-independent NX is still SCAFFOLD_ONLY; this module does not pretend otherwise",
            "succession launch/restart/unload are structural (install contract) and do not start a process",
            "super_resident.py is the sandbox floor, not the succession judge; read, not imported",
            "no GPU / FPGA board / power meter in this lane",
            "PROMOTE is encoded as the required evidence class but this sidecar cannot certify a GPU lease, so PROMOTE is not minted",
            "synthetic comparable-axis dominance is not physical dominance and does not promote",
        ],
    }
    _refuse_hardware_numbers(doc)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    shadow = _prove_shadow_refusals()
    if len(shadow) != len(SHADOW_FORBIDDEN_ACTIONS) or not all(r["refused"] for r in shadow):
        raise AssertionError(f"expected four watched shadow refusals, got {shadow}")
    pref = _prove_no_self_preference()
    if not all(r.get("refused") for r in pref):
        raise AssertionError(f"self-preference proofs failed: {pref}")
    stopped = _prove_stop_bad_child()
    if len(stopped) != 3 or not all(r["stopped"] and r["rolled_back"] for r in stopped):
        raise AssertionError(f"stop-bad-child proofs failed: {stopped}")
    bound = _prove_bound_construction()
    if len(bound) != 4 or not all(r["refused"] for r in bound):
        raise AssertionError(f"bound construction proofs failed: {bound}")
    ladder_proofs = _prove_ladder()
    if len(ladder_proofs) < 6 or any(r.get("verdict") == VERDICT_PROMOTE for r in ladder_proofs):
        raise AssertionError(f"ladder proofs failed: {ladder_proofs}")
    required_trials = {
        "candidate_judged_itself",
        "self_measured_dirty",
        "no_shadow_record",
        "dominated_candidate",
    }
    got_trials = {r["trial"] for r in ladder_proofs}
    if not required_trials <= got_trials:
        raise AssertionError(f"ladder proofs missing {required_trials - got_trials}")
    parent = make_incumbent()
    created = create_children_all_methods(parent)
    methods = {c["method"] for c in created}
    if methods != set(CHILD_METHODS):
        raise AssertionError(f"child methods {methods} != {set(CHILD_METHODS)}")
    for child in created:
        for field in LINEAGE_FIELDS:
            if field not in child["lineage"]:
                raise AssertionError(f"{child['id']} missing lineage field {field}")
        if child["lineage"]["physical_deltas"]["state"] != "UNKNOWN":
            raise AssertionError("physical deltas must stay UNKNOWN")
    run = run_synthetic_succession(seed=0)
    if not run["run"]["complete"]:
        raise AssertionError(f"synthetic succession incomplete: {run['run']['completed_steps']}")
    if run["run"]["active_id"] != run["child_id"]:
        raise AssertionError("successor was not made active")
    if not run["run"]["rollback_available"]:
        raise AssertionError("parent was not retained for rollback")
    if hasattr(Incumbent, "promote") and callable(getattr(Incumbent, "promote", None)):
        raise AssertionError("promote() must not exist on Incumbent")
    try:
        scalar_score(parent)
    except ScalarCollapseError as exc:
        if "scalar" not in str(exc).lower():
            raise AssertionError(f"scalar collapse raised the wrong error: {exc}") from exc
    else:
        raise AssertionError("scalar collapse was not refused")
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
