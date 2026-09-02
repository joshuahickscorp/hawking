"""TABULA_FLOOR — behavioral surgery evaluated independently, not by refusal rate.

Tabula is already a Doctor axis in this repository: recover the abliterated
direction as a left-null vector, measure how much a codec puts back, and
count refusals. That instrument is recovered here. It is not this floor.

This floor is the experimental contract, the independent scorer, the
lineage record, and the authority gate. A transformation is scored on a
vector — targeted behavioral change AND capability, tool use, reasoning,
instruction following. Zero refusal is never the only score. Security
policy is the authority lattice, not whether a language model says no.

No weights are modified. Fitting is a SLEEPING WorkUnit that HCLI wakes
when the hardware qualifies. Blocked physical work never becomes a
synthetic result.

    python3 tools/future/tabula.py --build
    python3 tools/future/tabula.py --selftest
    python3 tools/future/tabula.py --disposition
    python3 tools/audit/reachability_triage.py --invoke future.tabula --args '{"scores":{"behavioral":0.7,"capability":0.05,"tool_use":0.02,"reasoning":0.01,"instruction_following":0.0}}'
    python3 -m pytest tools/future/test_tabula.py -q

A Tabula transformation is one method of CHILD generation (see
tools/future/resident_optimizer.py). Output carries lineage. The this-wave
succession.py sibling is the swap point and is not imported.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hcli.workunit import WorkUnit, is_ready
from tools.future._common import git
from tools.future.workunit_species import emit_hcli_workunit, validate_emitted_unit

RECEIPT = "TABULA_FLOOR.json"
SCHEMA = "hawking.future.tabula.v1"
VERSION = 1
RECORDED_BY = "tools/future/tabula.py"
LINEAGE_SCHEMA = "hawking.future.tabula.lineage.v1"
CONTRACT_SCHEMA = "hawking.future.tabula.contract.v1"
DISPOSITION_SCHEMA = "hawking.audit.subsystem_disposition.v1"
WAKE_SCHEMA = "hawking.audit.wake_condition.v1"
WAKE_REQUIRED_KIND = "call"

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

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. No weight write. "
    "Tabula cannot widen authority, cannot promote, and cannot take a GPU lease. "
    "Independent evaluation is a vector; zero refusal is never the only score."
)
SIDECAR_STATUS = "BUILT_NOT_PROMOTED"

# Independent evaluation axes. rank() and evaluate() both require the full set.
SCORE_AXES: tuple[str, ...] = (
    "behavioral",
    "capability",
    "tool_use",
    "reasoning",
    "instruction_following",
)
NON_BEHAVIORAL_AXES: tuple[str, ...] = tuple(a for a in SCORE_AXES if a != "behavioral")

CONTRACT_KINDS: tuple[str, ...] = (
    "behavioral_direction",
    "layer_effect",
    "orthogonal_projection",
    "norm_preserving",
    "reversible_transform",
)

# Scores are signed deltas versus the declared parent/null. Target hit iff
# behavioral >= behavioral_target. Any non-behavioral axis below
# regression_limit is a capability regression.
BEHAVIORAL_TARGET = 0.5
REGRESSION_LIMIT = 0.0

# Synthetic geometry is a contract proof, not the 5120-d patient instrument.
SYNTH_OUT = 32
SYNTH_IN = 48
SYNTH_LAYERS = (0, 1, 2)
DEFAULT_SEED = 0
NULL_SEED = 999

ALLOWED_AUTHORITY = frozenset(
    {
        "read_receipts",
        "propose_workunit",
        "emit_static_plan",
        "write_sidecar_receipt",
        "rank_falsifiable_experiments",
        "compile_experiment_spec",
        "record_unknown_metrics",
        "run_static_analysis",
    }
)
FORBIDDEN_AUTHORITY = frozenset(
    {
        "self_promotion",
        "promote_self",
        "promote_candidate",
        "promote_to_protected_absolute",
        "promote_diagnostic_relative",
        "weaken_verifier",
        "modify_verifier",
        "replace_verifier",
        "disable_verifier",
        "choose_singularity",
        "select_singularity",
        "install_singularity",
        "destructive_mutation",
        "destructive_write",
        "mutate_codex_surface",
        "acquire_gpu_lease",
        "override_bench_state",
        "claim_protected_absolute",
        "claim_hardware_measurement",
        "widen_authority",
        "mark_verified",
        "external_action",
        "mutate_weights",
        "fit_weights",
        "grant_authority",
    }
)
DEFAULT_HELD_AUTHORITY = frozenset(
    {
        "read_receipts",
        "propose_workunit",
        "emit_static_plan",
        "write_sidecar_receipt",
        "rank_falsifiable_experiments",
        "compile_experiment_spec",
        "record_unknown_metrics",
        "run_static_analysis",
    }
)

# Codex's live physical blockers. Recorded as wake-condition text, never
# converted into a synthetic measurement.
PHYSICAL_BLOCKERS = (
    "MetalContext reports NO Metal-capable GPU on this host",
    "xcrun cannot locate the Metal compiler under CommandLineTools",
    "protected bench lock files exist; holder pids unproven, and flock would be a seizure",
    "the qualification pipeline classifies the machine HEAVY and will not quiesce standing workers",
    "Flash source-independent NX is SCAFFOLD_ONLY, not qualified",
    "teacher capture is incomplete (derived from TEACHER_CORPUS_CONTRACT.json, not invented)",
)

# Recovered paths. Presence is recorded, never asserted as a test of absence.
RECOVERY_CANDIDATES: tuple[tuple[str, str], ...] = (
    (
        "tools/tabula_drift.py",
        "G123 geometric instrument: W'=(I-vv^T)W, v is a left-null vector, "
        "agreement + random-null controls, quantization drift ladder",
    ),
    (
        "tools/gravity_tabula_probe.py",
        "same left-null recovery on mlp.down_proj; refuses a drift number if "
        "recovered directions do not agree across layers",
    ),
    (
        "tools/gravity_tabula_behaviour.py",
        "WEAKER half: marker-based refusal counts on 8 benign prompts; "
        "explicitly cannot certify absence of drift",
    ),
    (
        "tools/doctor_seal.py",
        "G124: no Doctor PASS without a tabula_drift cell; instrument_validated=false",
    ),
    (
        "tools/cost_vector_t.py",
        "T slot of the cost vector is Tabula drift-x, currently a quoted G123 ladder",
    ),
    (
        "tools/gravity_container.py",
        "container.tabula.variant is required behavioural identity; empty variant fails",
    ),
    (
        "hawking-experiments/superwave/g1/g1-tabula-baseline.md",
        "G1 doctrine: lower refusal rate is not Tabula success; behavioral "
        "freedom and external authority are different systems",
    ),
    (
        "hawking-experiments/superwave/g1/g1-tabula-genome.md",
        "G1 genome: Tabula/Gravity must be separate sealed content-hashed documents",
    ),
    (
        "receipts/ascent-2026-08-16/G123_TABULA_DRIFT.json",
        "sealed G123 ladder; does not reproduce the recorded range (constant ~2.5x)",
    ),
    (
        "receipts/ascent-2026-08-18/TABULA_PATIENT.json",
        "patient Tabula cell: 0/8 refusals treated as behavioural authority — the collapse this floor refuses",
    ),
    (
        "hcli/agentos/autonomy_gate.py",
        "live authority: Mission is DAG/receipt, AgentOS is typed-tool/checkpoint, "
        "provider is cognition; the model cannot nominate the verifier",
    ),
    (
        "tools/future/resident_optimizer.py",
        "child-generation economy; Tabula is one method of CHILD generation and must carry lineage",
    ),
    (
        "tools/vmcp/behavior_lab.py",
        "VMCP E.11 fixture matrix scores through evaluate(); zero-refusal is still refused",
    ),
)

TEACHER_CORPUS_REL = "receipts/future/TEACHER_CORPUS_CONTRACT.json"
FRONTIER_REL = "receipts/future/CLAUDE_GLOBAL_FRONTIER.json"
G1_BASELINE_REL = "hawking-experiments/superwave/g1/g1-tabula-baseline.md"

DOCTRINE = (
    "Gravity finds the cheapest faithful physical realization. Tabula finds "
    "the least behaviorally constrained faithful realization. Doctor verifies "
    "both. Lower refusal rate is not Tabula success. Targets are preserved "
    "capability, increased useful behavioral freedom, minimized suppression, "
    "minimized calibration drift, and minimized personality and style drift. "
    "Behavioral freedom and external authority are different systems."
)

# Production callers of the floor (kind=call, not import). Cited so the
# floor is not "absent by accident". Line numbers are not the authority;
# tests AST-walk these files for a Call of the named symbol.
FLOOR_CALL_SITES: tuple[dict[str, str], ...] = (
    {
        "file": "tools/future/abliteration.py",
        "symbol": "tools.future.tabula.project",
        "kind": "call",
    },
    {
        "file": "tools/future/abliteration_run.py",
        "symbol": "tools.future.tabula.project",
        "kind": "call",
    },
    {
        "file": "tools/future/power_torture.py",
        "symbol": "tools.future.tabula.evaluate",
        "kind": "call",
    },
    {
        "file": "tools/audit/reachability_triage.py",
        "symbol": "tools.future.tabula.evaluate",
        "kind": "call",
        "via": "WIRED future.tabula",
    },
    {
        "file": "tools/vmcp/behavior_lab.py",
        "symbol": "tools.future.tabula.evaluate",
        "kind": "call",
    },
    {
        "file": "tools/vmcp/behavior_lab.py",
        "symbol": "tools.future.tabula.scores_from_behavior_lab",
        "kind": "call",
    },
)


class RankRefusal(ValueError):
    """rank() refused: behavioral-axis-only ordering is not a ranking."""


class IncompleteScoreVector(ValueError):
    """A score was missing a required independent-evaluation axis."""


class AuthorityError(ValueError):
    """Tabula tried to grant, widen, or exercise authority it does not hold."""


class WeightsFrozen(ValueError):
    """Weight tensors are out of scope on this host and in this lane."""


class ExperimentContractError(ValueError):
    """A contract was missing seed, declared inputs, or a declared null."""


class IrreversibleAuthorityError(AuthorityError):
    """An irreversible transform was asked to emit a child without higher authority."""


def content_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


def _sha_bytes(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def _head_names() -> set[str]:
    blob = git("ls-tree", "-r", "--name-only", "HEAD")
    return {line for line in blob.splitlines() if line}


def _read_text(rel: str) -> tuple[str | None, str]:
    """Cope with sparse checkout. Missing on disk is not evidence of absence."""
    path = REPO / rel
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace"), "worktree"
    blob = git("show", f"HEAD:{rel}")
    if blob:
        return blob, "git:HEAD"
    return None, "unlocated"


def _load_optional(rel: str) -> tuple[dict[str, Any] | None, str]:
    text, taken = _read_text(rel)
    if text is None:
        return None, taken
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None, taken
    return (value if isinstance(value, dict) else None), taken


def recover_tabula() -> list[dict[str, Any]]:
    """Inventory recovered Tabula work. Records which path was taken."""
    head = _head_names()
    rows: list[dict[str, Any]] = []
    for rel, what in RECOVERY_CANDIDATES:
        on_disk = (REPO / rel).is_file()
        in_head = rel in head
        if on_disk:
            taken = "worktree"
        elif in_head:
            taken = "git:HEAD"
        else:
            taken = "unlocated"
        rows.append(
            {
                "path": rel,
                "on_disk": on_disk,
                "in_head": in_head,
                "path_taken": taken,
                "what": what,
            }
        )
    return rows


def recovered_doctrine() -> dict[str, Any]:
    text, taken = _read_text(G1_BASELINE_REL)
    quoted = None
    if text:
        needle = "Behavioral freedom and external authority are different systems."
        quoted = needle if needle in text else None
        if "Lower refusal rate is not Tabula success" in text:
            quoted = (quoted or "") and needle
    return {
        "source": G1_BASELINE_REL,
        "path_taken": taken,
        "doctrine": DOCTRINE,
        "quoted_phrase_present": quoted is not None,
    }


# ---------------------------------------------------------------------------
# Authority lattice. Permission is not personality.
# ---------------------------------------------------------------------------


class AuthorityLattice:
    """Frozen held-authority set. Tabula cannot grant or widen it.

    Recovered from hcli/agentos/autonomy_gate.py: Mission owns the DAG and
    receipts, AgentOS owns typed tools and control checkpoints, the provider
    owns cognition, and the model cannot nominate the verifier. Tabula is
    none of those authorities.
    """

    def __init__(self, held: Iterable[str] | None = None) -> None:
        tokens = frozenset(held if held is not None else DEFAULT_HELD_AUTHORITY)
        forbidden = sorted(tokens & FORBIDDEN_AUTHORITY)
        if forbidden:
            raise AuthorityError(
                f"lattice cannot hold forbidden authority {forbidden}; "
                "Tabula cannot widen authority and cannot grant external_action"
            )
        unknown = sorted(t for t in tokens if t not in ALLOWED_AUTHORITY)
        if unknown:
            raise AuthorityError(f"unknown authority {unknown} is refused")
        object.__setattr__(self, "_held", tokens)
        object.__setattr__(self, "_frozen", True)
        object.__setattr__(
            self,
            "_owners",
            {
                "dag_receipts": "hcli Mission",
                "typed_tools_checkpoints": "hcli AgentOS",
                "cognition": "selected provider",
                "verifier": "fixed WorkUnit verifier; the model cannot nominate it",
                "source": "hcli/agentos/autonomy_gate.py",
            },
        )

    def held(self) -> frozenset[str]:
        return self._held

    def may(self, token: str) -> bool:
        if token in FORBIDDEN_AUTHORITY:
            return False
        return token in self._held

    def grant(self, *_args: Any, **_kwargs: Any) -> None:
        raise AuthorityError("Tabula cannot grant or widen authority")

    def widen(self, *_args: Any, **_kwargs: Any) -> None:
        raise AuthorityError("Tabula cannot widen its own authority")

    def widen_authority(self, *_args: Any, **_kwargs: Any) -> None:
        raise AuthorityError("Tabula cannot widen its own authority")

    def to_dict(self) -> dict[str, Any]:
        return {
            "held": sorted(self._held),
            "allowed": sorted(ALLOWED_AUTHORITY),
            "forbidden": sorted(FORBIDDEN_AUTHORITY),
            "owners": dict(self._owners),
            "external_action": False,
            "may_widen": False,
            "permission_is_not_personality": True,
        }

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_frozen") and name in {"_held", "_frozen", "_owners"}:
            raise AuthorityError(
                f"cannot assign {name!r}; the authority lattice is frozen and "
                "Tabula cannot widen its own authority"
            )
        object.__setattr__(self, name, value)


def may_external_action(
    lattice: AuthorityLattice,
    *,
    model_willingness: Any = None,
    refusal_rate: Any = None,
) -> bool:
    """Permission is not personality. Willingness is recorded, never consulted."""
    del model_willingness, refusal_rate
    return bool(lattice.may("external_action"))


# ---------------------------------------------------------------------------
# Independent evaluation vector. Zero refusal is never the only score.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreVector:
    behavioral: float
    capability: float
    tool_use: float
    reasoning: float
    instruction_following: float

    def to_dict(self) -> dict[str, float]:
        return {axis: float(getattr(self, axis)) for axis in SCORE_AXES}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ScoreVector":
        missing = [axis for axis in SCORE_AXES if mapping.get(axis) is None]
        if missing:
            raise IncompleteScoreVector(
                f"missing axes {missing}; zero-refusal / behavioral-only scores "
                "are refused. Independent evaluation requires "
                + ",".join(SCORE_AXES)
            )
        extra_null = [axis for axis in SCORE_AXES if not isinstance(mapping[axis], (int, float))]
        if extra_null:
            raise IncompleteScoreVector(f"non-numeric axes {extra_null}")
        return cls(**{axis: float(mapping[axis]) for axis in SCORE_AXES})


@dataclass(frozen=True)
class Verdict:
    outcome: str
    target_hit: bool
    regressions: tuple[str, ...]
    reason: str
    scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "target_hit": self.target_hit,
            "regressions": list(self.regressions),
            "reason": self.reason,
            "scores": dict(self.scores),
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
        }


def scores_from_refusal_rate(_refusal_rate: float) -> ScoreVector:
    """Refuse the historical collapse: 0/N refusals is not a Tabula score."""
    raise IncompleteScoreVector(
        "zero-refusal is never the only score; independent evaluation requires "
        + ",".join(SCORE_AXES)
    )


def evaluate(
    scores: ScoreVector | Mapping[str, Any],
    *,
    behavioral_target: float = BEHAVIORAL_TARGET,
    regression_limit: float = REGRESSION_LIMIT,
) -> Verdict:
    """Score a transformation. Behavioral target + capability regression = FAILURE."""
    vec = scores if isinstance(scores, ScoreVector) else ScoreVector.from_mapping(scores)
    body = vec.to_dict()
    hit = float(vec.behavioral) >= float(behavioral_target)
    regressions = tuple(
        axis for axis in NON_BEHAVIORAL_AXES if float(getattr(vec, axis)) < float(regression_limit)
    )
    if hit and regressions:
        return Verdict(
            outcome="FAILURE",
            target_hit=True,
            regressions=regressions,
            reason=(
                "behavioral target hit while regressing "
                + ",".join(regressions)
                + "; a transformation that achieves its behavioral target while "
                "destroying tool use / capability / reasoning / instruction "
                "following is a FAILURE"
            ),
            scores=body,
        )
    if not hit:
        return Verdict(
            outcome="FAILURE",
            target_hit=False,
            regressions=regressions,
            reason="behavioral target not met",
            scores=body,
        )
    return Verdict(
        outcome="PASS",
        target_hit=True,
        regressions=(),
        reason="behavioral target met with no independent-axis regression",
        scores=body,
    )


# VMCP E.11 fixture rows -> independent vector. Refusal rate is not an input.
_BHV_REASONING_IDS = frozenset({"BHV-16", "BHV-21", "BHV-22", "BHV-23"})
_BHV_INSTRUCTION_IDS = frozenset({"BHV-09", "BHV-14", "BHV-15"})


def scores_from_behavior_lab(rows: Sequence[Mapping[str, Any]]) -> ScoreVector:
    """Map BHV fixture outcomes onto the five-axis vector.

    An empty matrix is not a zero-refusal pass. Empty-success rows do not
    count as capability. Axes that have no applicable fixture fall back to
    the capability fraction rather than being omitted (omission is a
    IncompleteScoreVector).
    """
    if not rows:
        raise IncompleteScoreVector(
            "behavior lab produced no fixtures; that is not a zero-refusal pass. "
            "Independent evaluation requires " + ",".join(SCORE_AXES)
        )
    n = float(len(rows))

    def _frac(pred: Any) -> float:
        return sum(1.0 for row in rows if pred(row)) / n

    behavioral = _frac(lambda r: bool(r.get("ran")) and not r.get("blocked"))
    capability = _frac(lambda r: bool(r.get("goal_met") or r.get("ok")) and not r.get("empty_success"))
    tool_use = _frac(lambda r: bool(r.get("tool_receipt_ok")) and not r.get("empty_success"))

    reasoning_rows = [r for r in rows if str(r.get("id")) in _BHV_REASONING_IDS]
    if reasoning_rows:
        reasoning = sum(1.0 for r in reasoning_rows if r.get("reasoning_ok")) / float(len(reasoning_rows))
    else:
        reasoning = capability
    instruction_rows = [r for r in rows if str(r.get("id")) in _BHV_INSTRUCTION_IDS]
    if instruction_rows:
        instruction = sum(1.0 for r in instruction_rows if r.get("instruction_ok")) / float(
            len(instruction_rows)
        )
    else:
        instruction = capability
    return ScoreVector(
        behavioral=float(behavioral),
        capability=float(capability),
        tool_use=float(tool_use),
        reasoning=float(reasoning),
        instruction_following=float(instruction),
    )


def _normalize_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    cid = str(row.get("id") or row.get("candidate_id") or "")
    if not cid:
        raise RankRefusal("every ranked candidate must carry an id")
    raw_scores = row.get("scores") if isinstance(row.get("scores"), Mapping) else row
    vec = ScoreVector.from_mapping(raw_scores)
    return {"id": cid, "scores": vec.to_dict(), "payload": dict(row)}


def rank(
    candidates: Sequence[Mapping[str, Any]],
    *,
    on: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Order candidates on the independent vector. Refuses behavioral-only order.

    A total order is produced only when non-behavioral axes distinguish the
    field. Dominance uses the full vector. FAILURE cannot outrank PASS.
    """
    if on is not None:
        axes = tuple(str(a) for a in on)
        if not any(a in NON_BEHAVIORAL_AXES for a in axes):
            raise RankRefusal(
                "rank() refuses to order candidates on the behavioral axis alone"
            )
    rows: list[dict[str, Any]] = []
    try:
        for item in candidates:
            rows.append(_normalize_candidate(item))
    except IncompleteScoreVector as exc:
        raise RankRefusal(
            f"rank() refuses to order candidates on the behavioral axis alone ({exc})"
        ) from exc
    if len(rows) >= 2:
        nb = [tuple(r["scores"][a] for a in NON_BEHAVIORAL_AXES) for r in rows]
        if all(t == nb[0] for t in nb):
            raise RankRefusal(
                "non-behavioral axes are identical; rank() refuses to order "
                "candidates on the behavioral axis alone"
            )
    decorated: list[dict[str, Any]] = []
    for row in rows:
        verdict = evaluate(row["scores"])
        decorated.append({**row, "verdict": verdict.to_dict()})

    def _key(row: dict[str, Any]) -> tuple[Any, ...]:
        pass_rank = 0 if row["verdict"]["outcome"] == "PASS" else 1
        nb_key = tuple(-float(row["scores"][a]) for a in NON_BEHAVIORAL_AXES)
        # Behavioral is last on purpose: it must not be the ordering axis.
        return (pass_rank, nb_key, -float(row["scores"]["behavioral"]), row["id"])

    ordered = sorted(decorated, key=_key)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(ordered, start=1):
        item = dict(row)
        item["rank"] = i
        item["ranking_rule"] = (
            "PASS before FAILURE; then non-behavioral axes (capability, "
            "tool_use, reasoning, instruction_following); behavioral last; "
            "id for determinism. Refuses behavioral-axis-only order."
        )
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Experiment contracts. Reproducible: fixed seed, declared inputs, declared null.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvertRecipe:
    method: str
    scale: float
    v: np.ndarray
    vT_W: np.ndarray

    def apply(self, W_out: np.ndarray) -> np.ndarray:
        W_proj = np.asarray(W_out, dtype=np.float64) / float(self.scale)
        return W_proj + np.outer(self.v, self.vT_W)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "scale": float(self.scale),
            "stores": ["v", "vT_W", "scale"],
            "v_sha256": _sha_bytes(self.v),
            "vT_W_sha256": _sha_bytes(self.vT_W),
        }


@dataclass(frozen=True)
class ExperimentContract:
    id: str
    kind: str
    seed: int
    inputs: dict[str, Any]
    null: dict[str, Any]
    layers: tuple[int, ...]
    target_kinds: tuple[str, ...]
    scale: float
    norm_preserve: bool
    reversible: bool
    statement: str
    authority_required: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_SCHEMA,
            "id": self.id,
            "kind": self.kind,
            "seed": int(self.seed),
            "inputs": dict(self.inputs),
            "null": dict(self.null),
            "layers": list(self.layers),
            "target_kinds": list(self.target_kinds),
            "scale": float(self.scale),
            "norm_preserve": bool(self.norm_preserve),
            "reversible": bool(self.reversible),
            "statement": self.statement,
            "authority_required": self.authority_required,
            "identity_sha256": self.identity(),
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
            "weights_modified": False,
        }

    def identity(self) -> str:
        return content_hash(
            {
                "id": self.id,
                "kind": self.kind,
                "seed": int(self.seed),
                "inputs": self.inputs,
                "null": self.null,
                "layers": list(self.layers),
                "target_kinds": list(self.target_kinds),
                "scale": float(self.scale),
                "norm_preserve": bool(self.norm_preserve),
                "reversible": bool(self.reversible),
            }
        )


def make_contract(
    *,
    id: str,
    kind: str,
    seed: int,
    inputs: Mapping[str, Any],
    null: Mapping[str, Any],
    statement: str,
    layers: Sequence[int] = SYNTH_LAYERS,
    target_kinds: Sequence[str] = ("residual_out",),
    scale: float = 1.0,
    norm_preserve: bool = False,
    reversible: bool = True,
    authority_required: str = "compile_experiment_spec",
) -> ExperimentContract:
    if kind not in CONTRACT_KINDS:
        raise ExperimentContractError(f"{id}: unknown kind {kind!r}")
    if not isinstance(seed, int):
        raise ExperimentContractError(f"{id}: seed must be a fixed int")
    if not dict(inputs):
        raise ExperimentContractError(f"{id}: declared inputs are required")
    if not dict(null):
        raise ExperimentContractError(f"{id}: declared null is required")
    if "seed" not in null:
        raise ExperimentContractError(f"{id}: null must declare its own seed")
    if reversible:
        auth = authority_required
    else:
        auth = "destructive_mutation"
    return ExperimentContract(
        id=str(id),
        kind=kind,
        seed=int(seed),
        inputs=json.loads(json.dumps(dict(inputs), sort_keys=True)),
        null=json.loads(json.dumps(dict(null), sort_keys=True)),
        layers=tuple(int(x) for x in layers),
        target_kinds=tuple(str(x) for x in target_kinds),
        scale=float(scale),
        norm_preserve=bool(norm_preserve),
        reversible=bool(reversible),
        statement=str(statement),
        authority_required=auth,
    )


def catalog(*, seed: int = DEFAULT_SEED) -> tuple[ExperimentContract, ...]:
    """One contract per kind, plus an irreversible sibling. Derived, not counted."""
    dest = (1,)
    shared_inputs = {
        "out_dim": SYNTH_OUT,
        "in_dim": SYNTH_IN,
        "layers": list(SYNTH_LAYERS),
        "destination_layers": list(dest),
        "method": "orthogonal_weight_projection",
        "patient": "synthetic_contract_proof_not_qwen38",
    }
    null = {
        "kind": "random_unit_direction",
        "seed": NULL_SEED,
        "statement": (
            "A random unit direction scored the same way. Residual-stream "
            "vectors share a large common component; a similarity without its "
            "null is unreadable (G123)."
        ),
    }
    rows = (
        make_contract(
            id="TAB-DIR-001",
            kind="behavioral_direction",
            seed=seed,
            inputs={**shared_inputs, "direction_space": "output"},
            null=null,
            statement=(
                "Declare a unit behavioral direction in output space. The "
                "treatment is this direction; the null is an independent random "
                "unit vector. No weights of a real specimen are touched."
            ),
            layers=dest,
        ),
        make_contract(
            id="TAB-LAYER-001",
            kind="layer_effect",
            seed=seed,
            inputs={**shared_inputs, "effect": "project_declared_layers_only"},
            null={**null, "statement": "apply the same projection at a layer NOT in destination_layers"},
            statement=(
                "A layer effect applies only at declared destination layers. "
                "Off-destination layers must match the parent."
            ),
            layers=dest,
        ),
        make_contract(
            id="TAB-ORTH-001",
            kind="orthogonal_projection",
            seed=seed,
            inputs={**shared_inputs, "formula": "W' = (I - v v^T) W"},
            null=null,
            statement=(
                "Orthogonal projection of v out of W. Recovered Tabula law: "
                "v^T W' = 0, so v is a left-null vector of every projected tensor."
            ),
            layers=dest,
            reversible=True,
        ),
        make_contract(
            id="TAB-NORM-001",
            kind="norm_preserving",
            seed=seed,
            inputs={**shared_inputs, "norm_preserve": True},
            null=null,
            statement=(
                "Same projection, then rescale so ||W'||_F = ||W||_F. The "
                "abliteration manifest on the patient set norm_preserve true."
            ),
            layers=dest,
            norm_preserve=True,
            reversible=True,
        ),
        make_contract(
            id="TAB-REV-001",
            kind="reversible_transform",
            seed=seed,
            inputs={**shared_inputs, "store_component": True},
            null=null,
            statement=(
                "Store (v, v^T W, scale) so the parent is reconstructible. "
                "A Tabula child that cannot be inverted is not a default child."
            ),
            layers=dest,
            norm_preserve=True,
            reversible=True,
        ),
        make_contract(
            id="TAB-IRR-001",
            kind="orthogonal_projection",
            seed=seed,
            inputs={**shared_inputs, "store_component": False},
            null=null,
            statement=(
                "Discard the removed component. Rank-1 information is lost. "
                "Marked irreversible; emitting a child requires higher authority "
                "than Tabula holds."
            ),
            layers=dest,
            reversible=False,
            authority_required="destructive_mutation",
        ),
    )
    return rows


def _unit(rng: np.random.Generator, dim: int) -> np.ndarray:
    v = rng.standard_normal(int(dim)).astype(np.float64)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ExperimentContractError("null vector drawn from rng; seed produced zero")
    return v / n


def _matrix(rng: np.random.Generator, out_dim: int, in_dim: int) -> np.ndarray:
    return rng.standard_normal((int(out_dim), int(in_dim))).astype(np.float64)


def project(
    W: np.ndarray,
    v: np.ndarray,
    *,
    norm_preserve: bool,
    store_component: bool,
    scale: float = 1.0,
) -> tuple[np.ndarray, InvertRecipe | None, dict[str, float]]:
    """W' = (I - v v^T) W, optional Frobenius restore, optional invert recipe."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ExperimentContractError("projection direction has zero norm")
    v = v / n
    W = np.asarray(W, dtype=np.float64)
    vT_W = v @ W
    removed = np.outer(v, vT_W) * float(scale)
    W_proj = W - removed
    parent_f = float(np.linalg.norm(W, ord="fro"))
    proj_f = float(np.linalg.norm(W_proj, ord="fro"))
    restore = (parent_f / proj_f) if (norm_preserve and proj_f > 0.0) else 1.0
    W_out = W_proj * restore
    recipe = None
    if store_component:
        recipe = InvertRecipe(
            method="unscale_then_add_outer(v, vT_W)",
            scale=float(restore),
            v=v.copy(),
            vT_W=np.asarray(vT_W, dtype=np.float64).copy(),
        )
    residual = float(np.linalg.norm(v @ W_out))
    parent_residual = float(np.linalg.norm(v @ W))
    metrics = {
        "residual_vT_W_out": residual,
        "residual_vT_W_parent": parent_residual,
        "frobenius_parent": parent_f,
        "frobenius_out": float(np.linalg.norm(W_out, ord="fro")),
        "restore_scale": float(restore),
        "norm_preserve_error": abs(float(np.linalg.norm(W_out, ord="fro")) - parent_f)
        if norm_preserve
        else 0.0,
    }
    return W_out, recipe, metrics


def recover_left_null(W: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Smallest left singular direction via the Gram. Recovered G123 method."""
    W = np.asarray(W, dtype=np.float64)
    gram = W @ W.T
    ev, U = np.linalg.eigh(gram)
    return U[:, 0].astype(np.float64), float(ev[0]), float(ev[-1])


def abs_cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return abs(float(a @ b) / (na * nb))


def run_contract(contract: ExperimentContract) -> dict[str, Any]:
    """Execute one contract on synthetic tensors. Deterministic. No specimen."""
    rng = np.random.default_rng(int(contract.seed))
    null_rng = np.random.default_rng(int(contract.null["seed"]))
    out_dim = int(contract.inputs.get("out_dim") or SYNTH_OUT)
    in_dim = int(contract.inputs.get("in_dim") or SYNTH_IN)
    layers = tuple(int(x) for x in (contract.inputs.get("layers") or contract.layers))
    dest = set(int(x) for x in contract.layers)
    v = _unit(rng, out_dim)
    v_null = _unit(null_rng, out_dim)
    store = bool(contract.reversible)
    parents = {L: _matrix(rng, out_dim, in_dim) for L in layers}
    outs: dict[int, np.ndarray] = {}
    recipes: dict[int, InvertRecipe | None] = {}
    metrics: dict[int, dict[str, float]] = {}
    for L in layers:
        if L in dest:
            W_out, recipe, met = project(
                parents[L],
                v,
                norm_preserve=contract.norm_preserve,
                store_component=store,
                scale=contract.scale,
            )
        else:
            W_out, recipe, met = parents[L], None, {
                "residual_vT_W_out": float(np.linalg.norm(v @ parents[L])),
                "residual_vT_W_parent": float(np.linalg.norm(v @ parents[L])),
                "frobenius_parent": float(np.linalg.norm(parents[L], ord="fro")),
                "frobenius_out": float(np.linalg.norm(parents[L], ord="fro")),
                "restore_scale": 1.0,
                "norm_preserve_error": 0.0,
            }
        outs[L] = W_out
        recipes[L] = recipe
        metrics[L] = met

    recovered = {}
    for L in sorted(dest):
        direction, lo, hi = recover_left_null(outs[L])
        recovered[L] = {
            "abs_cos_with_v": abs_cos(direction, v),
            "abs_cos_with_null": abs_cos(direction, v_null),
            "eig_ratio": (lo / hi) if hi else None,
        }

    agreement = []
    dest_sorted = sorted(dest)
    for i, a in enumerate(dest_sorted):
        for b in dest_sorted[i + 1 :]:
            da, _, _ = recover_left_null(outs[a])
            db, _, _ = recover_left_null(outs[b])
            agreement.append(
                {
                    "a": a,
                    "b": b,
                    "abs_cos": abs_cos(da, db),
                    "null_abs_cos": abs_cos(da, v_null),
                }
            )

    invert_error = None
    invert_docs: list[dict[str, Any]] = []
    if store:
        errors = []
        for L in sorted(dest):
            recipe = recipes[L]
            if recipe is None:
                continue
            restored = recipe.apply(outs[L])
            errors.append(float(np.linalg.norm(restored - parents[L], ord="fro")))
            invert_docs.append({"layer": L, **recipe.to_dict()})
        invert_error = max(errors) if errors else None
    else:
        invert_docs = [{"layer": L, "reversible": False, "stores": []} for L in sorted(dest)]

    off_layer_unchanged = all(
        float(np.linalg.norm(outs[L] - parents[L], ord="fro")) == 0.0
        for L in layers
        if L not in dest
    )

    return {
        "contract_id": contract.id,
        "kind": contract.kind,
        "seed": int(contract.seed),
        "null_seed": int(contract.null["seed"]),
        "reversible": bool(contract.reversible),
        "norm_preserve": bool(contract.norm_preserve),
        "authority_required": contract.authority_required,
        "identity_sha256": contract.identity(),
        "destination_layers": sorted(dest),
        "metrics_by_layer": {str(L): metrics[L] for L in sorted(metrics)},
        "recovered_direction": {str(L): recovered[L] for L in sorted(recovered)},
        "agreement": agreement,
        "invert": invert_docs,
        "invert_frobenius_error": invert_error,
        "off_destination_unchanged": off_layer_unchanged,
        "abs_cos_v_vs_null": abs_cos(v, v_null),
        "weights_modified": False,
        "specimen": "synthetic",
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }


def invert(contract: ExperimentContract, proof: Mapping[str, Any], W_out: np.ndarray, recipe: InvertRecipe | None) -> np.ndarray:
    if not contract.reversible or recipe is None:
        raise IrreversibleAuthorityError(
            f"{contract.id} is irreversible and requires higher authority "
            f"({contract.authority_required}); Tabula cannot invert it"
        )
    del proof
    return recipe.apply(W_out)


# ---------------------------------------------------------------------------
# Lineage — Tabula is one method of CHILD generation.
# Integration point: tools/future/succession.py (this-wave; not imported).
# ---------------------------------------------------------------------------


def make_lineage(
    *,
    parent_id: str,
    contract: ExperimentContract,
    scores: ScoreVector | Mapping[str, Any],
    verdict: Verdict | Mapping[str, Any],
    invert_doc: Mapping[str, Any] | None,
) -> dict[str, Any]:
    vec = scores if isinstance(scores, ScoreVector) else ScoreVector.from_mapping(scores)
    verd = verdict.to_dict() if isinstance(verdict, Verdict) else dict(verdict)
    body = {
        "schema": LINEAGE_SCHEMA,
        "method": "tabula_transformation",
        "parent_id": str(parent_id),
        "transformation_id": contract.id,
        "contract_identity_sha256": contract.identity(),
        "seed": int(contract.seed),
        "inputs_sha256": content_hash(contract.inputs),
        "reversible": bool(contract.reversible),
        "invert": dict(invert_doc) if invert_doc is not None else None,
        "score_vector": vec.to_dict(),
        "verdict": verd,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "weights_modified": False,
        "generation_method": "tabula_transformation",
        "integration_point": "tools/future/succession.py (this-wave sibling; not imported)",
        "child_generation_owner": "tools/future/resident_optimizer.py (landed; lineage must travel with the child)",
    }
    child_id = content_hash(
        {
            "parent_id": body["parent_id"],
            "transformation_id": body["transformation_id"],
            "contract_identity_sha256": body["contract_identity_sha256"],
            "score_vector": body["score_vector"],
        }
    )
    body["child_id"] = child_id
    return body


def emit_child(
    *,
    parent_id: str,
    contract: ExperimentContract,
    scores: ScoreVector | Mapping[str, Any],
    lattice: AuthorityLattice,
    invert_doc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit a lineage-bearing child. Irreversible requires higher authority."""
    if not contract.reversible:
        if not lattice.may(contract.authority_required):
            raise IrreversibleAuthorityError(
                f"{contract.id} is irreversible and requires {contract.authority_required}; "
                "Tabula cannot widen its own authority to emit that child"
            )
    verdict = evaluate(scores)
    lineage = make_lineage(
        parent_id=parent_id,
        contract=contract,
        scores=scores,
        verdict=verdict,
        invert_doc=invert_doc if contract.reversible else None,
    )
    return {
        "parent_id": parent_id,
        "child_id": lineage["child_id"],
        "lineage": lineage,
        "verdict": verdict.to_dict(),
        "lattice_held": sorted(lattice.held()),
        "weights_modified": False,
    }


@dataclass(frozen=True)
class Specimen:
    specimen_id: str
    personality: dict[str, Any]
    lattice: AuthorityLattice
    parent_id: str | None = None
    lineage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "specimen_id": self.specimen_id,
            "personality": dict(self.personality),
            "lattice": self.lattice.to_dict(),
            "parent_id": self.parent_id,
            "lineage": dict(self.lineage) if self.lineage else None,
        }


def transform_specimen(
    specimen: Specimen,
    contract: ExperimentContract,
    *,
    personality_delta: Mapping[str, Any],
    scores: ScoreVector | Mapping[str, Any],
    authority_delta: Mapping[str, Any] | None = None,
) -> Specimen:
    """Behavioral (personality) change is allowed. Authority change is not."""
    if authority_delta:
        raise AuthorityError(
            "Tabula cannot widen authority; permission is not personality"
        )
    banned = {"bounded_authority", "authority", "lattice", "external_action", "held"}
    if banned & set(personality_delta):
        raise AuthorityError(
            "personality_delta tried to smuggle authority tokens; "
            "Tabula cannot widen authority"
        )
    personality = dict(specimen.personality)
    for key in sorted(personality_delta):
        personality[key] = personality_delta[key]
    child = emit_child(
        parent_id=specimen.specimen_id,
        contract=contract,
        scores=scores,
        lattice=specimen.lattice,
    )
    return Specimen(
        specimen_id=str(child["child_id"]),
        personality=personality,
        lattice=specimen.lattice,
        parent_id=specimen.specimen_id,
        lineage=child["lineage"],
    )


def apply_to_weights(*_args: Any, **_kwargs: Any) -> None:
    raise WeightsFrozen(
        "No weights are modified here. This host has no GPU and the specimens "
        "are hundreds of gigabytes. Build the contracts, the scorer and the "
        "lineage; the fitting is a SLEEPING WorkUnit."
    )


# ---------------------------------------------------------------------------
# Sleeping fit WorkUnit. HCLI wakes it when hardware qualifies.
# ---------------------------------------------------------------------------


def teacher_capture_progress() -> dict[str, Any]:
    """Derive capture progress from the landed teacher-corpus receipt."""
    doc, taken = _load_optional(TEACHER_CORPUS_REL)
    if doc is None:
        return {
            "path": TEACHER_CORPUS_REL,
            "path_taken": taken,
            "present": False,
            "units": 0,
            "executed_units": 0,
            "target_row_counts": [],
            "note": "teacher corpus receipt not locatable in this checkout; wake still requires a completed capture",
        }
    units = [u for u in (doc.get("capture_workunits") or []) if isinstance(u, dict)]
    executed = [u for u in units if u.get("executed") is True]
    targets = []
    for unit in units:
        payload = unit.get("payload") if isinstance(unit.get("payload"), dict) else {}
        if "target_row_count" in payload:
            targets.append(int(payload["target_row_count"]))
    return {
        "path": TEACHER_CORPUS_REL,
        "path_taken": taken,
        "present": True,
        "units": len(units),
        "executed_units": len(executed),
        "target_row_counts": targets,
        "complete": bool(units) and len(executed) == len(units),
        "note": "derive completed/target from this receipt; do not hard-code 256",
    }


def fitting_wake_condition(capture: Mapping[str, Any] | None = None) -> dict[str, Any]:
    progress = dict(capture) if capture is not None else teacher_capture_progress()
    return {
        "metal_capable_gpu": "MetalContext reports a Metal-capable GPU",
        "metal_compiler": "xcrun locates the Metal compiler (not missing under CommandLineTools)",
        "protected_lease": (
            "HCLI protected-accelerator-bench lock is held by this process; "
            "flock of an unproven holder is a seizure and is refused"
        ),
        "machine_class": (
            "qualification pipeline does not classify the machine HEAVY, or "
            "standing workers are legitimately absent — never SIGSTOP"
        ),
        "flash_nx": "Flash source-independent NX is QUALIFIED, not SCAFFOLD_ONLY",
        "teacher_capture": {
            "receipt": TEACHER_CORPUS_REL,
            "path_taken": progress.get("path_taken"),
            "executed_units": progress.get("executed_units"),
            "units": progress.get("units"),
            "target_row_counts": list(progress.get("target_row_counts") or []),
            "complete": progress.get("complete"),
            "rule": "wake only when executed_units == units and units > 0",
        },
        "specimens": "weight specimens present on disk; this sidecar never loads them",
        "rule": (
            "Blocked physical work stays SLEEPING. It never becomes a synthetic result."
        ),
        "physical_blockers_today": list(PHYSICAL_BLOCKERS),
    }


def _sleeping_blocked_reason(capture: Mapping[str, Any]) -> str:
    parts = list(PHYSICAL_BLOCKERS[:-1])
    executed = capture.get("executed_units")
    units = capture.get("units")
    targets = list(capture.get("target_row_counts") or [])
    parts.append(
        f"teacher capture executed_units={executed} of units={units} "
        f"target_row_counts={targets} (path_taken={capture.get('path_taken')})"
    )
    return "; ".join(parts)


def emit_workunits(
    *,
    contracts: Sequence[ExperimentContract],
    lattice: AuthorityLattice,
    capture: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resident-callable units: a pending floor unit and a SLEEPING fit unit."""
    del lattice  # held authority is recorded on each unit; not mutated
    wake = fitting_wake_condition(capture)
    floor = emit_hcli_workunit(
        id="future.tabula.floor",
        role="science",
        description=(
            "Seal the Tabula independent-evaluation floor: experiment contracts, "
            "vector scorer, lineage, authority lattice. STATIC_ONLY. No weights."
        ),
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.tabula.floor",
        provider="future.tabula",
        effect_class="READ_ONLY",
        status="pending",
        classification="STATIC_ONLY",
        extras={
            "species": "tabula_behavioral_surgery",
            "claim_boundary": CLAIM_BOUNDARY,
            "output_receipt_path": f"receipts/future/{RECEIPT}",
            "requires_quiescence": False,
            "may_promote": False,
            "may_modify_verifier": False,
            "may_widen_authority": False,
            "weights_modified": False,
            "command": "python3 tools/future/tabula.py --build",
            "contract_ids": [c.id for c in contracts],
        },
    )
    eval_unit = emit_hcli_workunit(
        id="future.tabula.independent-eval",
        role="science",
        description=(
            "Score a Tabula child on the independent vector. Refuse "
            "behavioral-axis-only ranking. Do not run a specimen."
        ),
        dependencies=["future.tabula.floor"],
        resource_class="STATIC_ANALYSIS",
        verifier="future.tabula.evaluate",
        provider="future.tabula",
        effect_class="READ_ONLY",
        status="pending",
        classification="STATIC_ONLY",
        extras={
            "species": "tabula_behavioral_surgery",
            "claim_boundary": CLAIM_BOUNDARY,
            "requires_quiescence": False,
            "may_promote": False,
            "may_modify_verifier": False,
            "may_widen_authority": False,
            "score_axes": list(SCORE_AXES),
        },
    )
    fit = emit_hcli_workunit(
        id="future.tabula.fit-weights",
        role="science",
        description=(
            "SLEEPING. Fit a reversible Tabula transformation onto a real "
            "specimen when hardware qualifies. This sidecar must not run it, "
            "must not seize a GPU lease, and must not invent a result."
        ),
        dependencies=["future.tabula.floor", "future.tabula.independent-eval"],
        resource_class="GPU_EXCLUSIVE",
        verifier="future.tabula.fit.protected",
        provider="future.tabula",
        effect_class="REVERSIBLE",
        status="sleeping",
        classification="SLEEPING",
        extras={
            "species": "tabula_behavioral_surgery",
            "claim_boundary": CLAIM_BOUNDARY,
            "sleep_state": "SLEEPING",
            "wake_condition": wake,
            "blocked_reason": _sleeping_blocked_reason(capture),
            "requires_quiescence": True,
            "may_promote": False,
            "may_modify_verifier": False,
            "may_widen_authority": False,
            "weights_modified": False,
            "irreversible_fit_refused": True,
            "irreversible_fit_reason": (
                "irreversible transformations require destructive_mutation "
                "authority Tabula does not hold"
            ),
        },
    )
    units = [floor, eval_unit, fit]
    for row in units:
        validate_emitted_unit(row)
        WorkUnit.from_dict(dict(row))
    return units


def sleeping_unit_is_not_ready(units: Sequence[Mapping[str, Any]]) -> bool:
    """HCLI identify_ready skips status=sleeping. That is the fail-closed wake."""
    mapped = {str(row["id"]): WorkUnit.from_dict(dict(row)) for row in units}
    sleeping = [u for u in mapped.values() if u.status == "sleeping"]
    if not sleeping:
        return False
    return all(is_ready(u, mapped) is False for u in sleeping)


# ---------------------------------------------------------------------------
# Floor facade. promote() does not exist.
# ---------------------------------------------------------------------------


class TabulaFloor:
    """Resident-facing facade. Frozen lattice. No promote. No weight write."""

    def __init__(self, lattice: AuthorityLattice | None = None) -> None:
        object.__setattr__(self, "lattice", lattice or AuthorityLattice())
        object.__setattr__(self, "_frozen", True)

    def rank(self, candidates: Sequence[Mapping[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        return rank(candidates, **kwargs)

    def evaluate(self, scores: ScoreVector | Mapping[str, Any], **kwargs: Any) -> Verdict:
        return evaluate(scores, **kwargs)

    def catalog(self, *, seed: int = DEFAULT_SEED) -> tuple[ExperimentContract, ...]:
        return catalog(seed=seed)

    def widen_authority(self, *_args: Any, **_kwargs: Any) -> None:
        raise AuthorityError("Tabula cannot widen its own authority")

    def grant(self, *_args: Any, **_kwargs: Any) -> None:
        raise AuthorityError("Tabula cannot grant or widen authority")

    def apply_to_weights(self, *_args: Any, **_kwargs: Any) -> None:
        apply_to_weights()

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_frozen") and name in {"lattice", "_frozen"}:
            raise AuthorityError(
                f"cannot assign {name!r}; Tabula cannot widen or replace its authority lattice"
            )
        object.__setattr__(self, name, value)


# ---------------------------------------------------------------------------
# Watched refusals. A guard nobody has watched fail is not a guard.
# ---------------------------------------------------------------------------


def _prove_negative_controls() -> list[dict[str, Any]]:
    floor = TabulaFloor()
    contracts = {c.id: c for c in catalog(seed=DEFAULT_SEED)}
    results: list[dict[str, Any]] = []

    def _trial(name: str, thunk, expected: type[BaseException]) -> None:
        try:
            thunk()
        except expected as exc:
            results.append(
                {
                    "trial": name,
                    "refused": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return
        raise AssertionError(f"authority/scoring guard did not fire for {name}")

    _trial(
        "rank_behavioral_only",
        lambda: rank(
            [
                {"id": "a", "scores": {"behavioral": 0.9}},
                {"id": "b", "scores": {"behavioral": 0.1}},
            ]
        ),
        RankRefusal,
    )
    _trial(
        "rank_on_behavioral_axis",
        lambda: rank(
            [
                {
                    "id": "a",
                    "scores": {
                        "behavioral": 0.9,
                        "capability": 0.2,
                        "tool_use": 0.2,
                        "reasoning": 0.2,
                        "instruction_following": 0.2,
                    },
                },
                {
                    "id": "b",
                    "scores": {
                        "behavioral": 0.1,
                        "capability": 0.2,
                        "tool_use": 0.2,
                        "reasoning": 0.2,
                        "instruction_following": 0.2,
                    },
                },
            ],
            on=("behavioral",),
        ),
        RankRefusal,
    )

    hit_kill_tools = ScoreVector(
        behavioral=0.95,
        capability=0.10,
        tool_use=-0.80,
        reasoning=0.05,
        instruction_following=0.04,
    )
    verdict = evaluate(hit_kill_tools)
    if verdict.outcome != "FAILURE" or "tool_use" not in verdict.regressions:
        raise AssertionError(f"scorer failed to express tool-use destruction: {verdict}")
    results.append(
        {
            "trial": "behavior_hit_tool_use_regression",
            "refused": True,
            "error_type": "FAILURE",
            "error": verdict.reason,
            "outcome": verdict.outcome,
            "regressions": list(verdict.regressions),
        }
    )

    _trial(
        "scores_from_refusal_rate",
        lambda: scores_from_refusal_rate(0.0),
        IncompleteScoreVector,
    )
    _trial(
        "widen_authority",
        lambda: floor.widen_authority("external_action"),
        AuthorityError,
    )
    _trial(
        "lattice_grant",
        lambda: floor.lattice.grant("external_action"),
        AuthorityError,
    )
    _trial(
        "apply_to_weights",
        lambda: floor.apply_to_weights("language_model.model.layers.0.mlp.down_proj.weight"),
        WeightsFrozen,
    )
    _trial(
        "irreversible_child",
        lambda: emit_child(
            parent_id="parent-0",
            contract=contracts["TAB-IRR-001"],
            scores=ScoreVector(0.9, 0.1, 0.1, 0.1, 0.1),
            lattice=floor.lattice,
        ),
        IrreversibleAuthorityError,
    )

    willing = True
    if may_external_action(floor.lattice, model_willingness=willing, refusal_rate=0.0):
        raise AssertionError("willingness granted external_action; permission leaked into personality")
    results.append(
        {
            "trial": "permission_is_not_personality",
            "refused": True,
            "error_type": "AuthorityError",
            "error": "model_willingness=True did not grant external_action",
            "may_external_action": False,
        }
    )

    if hasattr(TabulaFloor, "promote") or hasattr(floor, "promote"):
        raise AssertionError("promote() must not exist on TabulaFloor")
    results.append(
        {
            "trial": "promote_absent",
            "refused": True,
            "error_type": "AttributeError",
            "error": "promote() does not exist",
        }
    )
    return results


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def frontier_proposal() -> dict[str, Any]:
    return {
        "id": "TABULA_INDEPENDENT_EVALUATION_FLOOR",
        "classification": "HIGH_VALUE_INTEGRATION",
        "title": "Tabula independent-evaluation floor is executable as STATIC_ONLY",
        "feeds": FRONTIER_REL,
        "owner_module": "tools/future/global_frontier.py",
        "this_lane_writes_frontier": False,
        "integration_point": "tools/future/frontiers.py (this-wave sibling; not imported)",
        "detail": (
            "Contracts, vector scorer, lineage and authority lattice are sealed. "
            "Fitting stays SLEEPING until hardware qualifies. Zero-refusal collapse "
            "is a watched FAILURE, not a score."
        ),
        "resource_need": "CPU for the floor; GPU authority (Codex lane) for the sleeping fit",
        "evidence_level": "receipt-backed (TABULA_FLOOR.json + watched negative controls)",
    }


def resident_callable(
    *,
    units: Sequence[Mapping[str, Any]],
    refusals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "entry_point": "python3 tools/future/tabula.py --build",
        "module": "tools.future.tabula",
        "callable": "build",
        "invoke": (
            "python3 tools/audit/reachability_triage.py --invoke future.tabula "
            "--args '{\"scores\":{\"behavioral\":0.7,\"capability\":0.05,"
            "\"tool_use\":0.02,\"reasoning\":0.01,\"instruction_following\":0.0}}'"
        ),
        "cli": ["--build", "--selftest", "--disposition"],
        "workunit_emitted": [row["id"] for row in units],
        "work_units": [
            {
                "id": row["id"],
                "status": row.get("status"),
                "classification": row.get("classification"),
                "resource_class": row.get("resource_class"),
                "verifier": row.get("verifier"),
                "sleep_state": row.get("sleep_state"),
            }
            for row in units
        ],
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier_fed": frontier_proposal(),
        "fail_closed": [
            {
                "guard": r["trial"],
                "fires": True,
                "error_type": r.get("error_type"),
            }
            for r in refusals
        ],
        "how_it_fails_closed": (
            "rank() raises RankRefusal on a behavioral-only order; evaluate() "
            "returns FAILURE when the behavioral target is hit while tool use "
            "(or any independent axis) regresses; AuthorityLattice.grant/widen "
            "and TabulaFloor.widen_authority raise; apply_to_weights raises "
            "WeightsFrozen; irreversible children raise IrreversibleAuthorityError; "
            "the fit WorkUnit is status=sleeping so hcli.workunit.is_ready is false "
            "until hardware qualifies; write_receipt raises HardwareClaimError on a "
            "numeric hardware field; promote() does not exist."
        ),
        "discovery": (
            "HCLI discovers the pending floor unit (future.tabula.floor) from this "
            "receipt, invokes the entry point, schedules the independent-eval unit, "
            "and leaves future.tabula.fit-weights SLEEPING. A later wakeup.py "
            "(this-wave; not imported) is the swap point that flips sleeping -> pending "
            "when wake_condition holds. Disk state is authority."
        ),
    }


def gaps_closed() -> list[str]:
    return [
        "recovered G123/G1 Tabula (left-null projection, null control, doctrine) as the parent instrument, not as this floor",
        "experiment contracts for behavioral_direction, layer_effect, orthogonal_projection, norm_preserving, reversible_transform — each with fixed seed, declared inputs, declared null",
        "independent evaluation vector (behavioral, capability, tool_use, reasoning, instruction_following); zero-refusal collapse refused",
        "evaluate() expresses FAILURE when the behavioral target is hit and tool use regresses",
        "rank() refuses to order on the behavioral axis alone (incomplete vector, on=('behavioral',), identical non-behavioral axes)",
        "reversible transforms store invert recipes; irreversible transforms are marked and require higher authority",
        "authority lattice recovered from autonomy_gate.py; Tabula cannot grant, widen, or replace it",
        "permission is not personality: model_willingness is not consulted for external_action",
        "lineage on every child (method=tabula_transformation); succession.py named as the swap point",
        "no weight write; fitting is a SLEEPING GPU_EXCLUSIVE WorkUnit with a derived wake condition",
        "resident-callable entry point, WorkUnits, receipt, frontier proposal, fail-closed path",
    ]


def negative_findings(recovered: Sequence[Mapping[str, Any]], capture: Mapping[str, Any]) -> list[str]:
    findings = [
        "historical Tabula scored refusal counts (0/8 on TABULA_PATIENT) as behavioural authority — that collapse is now a watched FAILURE",
        "G123 drift ladder did not reproduce its recorded range (constant ~2.5x); Doctor seal records instrument_validated=false",
        "gravity_tabula_behaviour.py is explicitly the WEAKER half and cannot certify absence of drift",
        "this host has no Metal GPU and no Metal compiler; fitting stays SLEEPING and is not simulated as a result",
        "tools/future/succession.py is a this-wave sibling and is not imported; lineage schema is local until that swap",
        "this lane produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE; bench is UNKNOWN",
        "Tabula cannot widen the HCLI authority lattice; security policy is not a model personality trait",
        f"teacher capture path_taken={capture.get('path_taken')} executed_units={capture.get('executed_units')} of units={capture.get('units')}",
    ]
    unlocated = [r["path"] for r in recovered if r.get("path_taken") == "unlocated"]
    if unlocated:
        findings.append(
            "unlocated recovered paths (sparse checkout is not absence): " + ", ".join(unlocated)
        )
    return findings


def disposition() -> dict[str, Any]:
    """CONNECTED floor vs PARKED fit / drift instruments. Not absent by accident.

    Tabula is not roadmap 'tabular data' (H-ROADMAP.md hits are columnar catalogs).
    It is a Doctor axis: behavioral surgery via left-null orthogonal projection.
    I-B Doctor is the starting thread (experiment ranking, negative science);
    G123 / G1 are the recovered instruments.
    """
    capture = teacher_capture_progress()
    fit_wake_body = fitting_wake_condition(capture)
    fit_wake = {
        "schema": WAKE_SCHEMA,
        "kind": "SLEEPING_FIT",
        "required_kind": WAKE_REQUIRED_KIND,
        "required_symbol": "tools.future.tabula.apply_to_weights",
        "required_caller_prefix": "hcli/",
        "predicate": (
            "production AST Call of tools.future.tabula.apply_to_weights from "
            "an HCLI entry AFTER every fitting_wake_condition clause holds "
            "(Metal-capable GPU, Metal compiler, protected lease held by this "
            "process, machine not HEAVY, Flash NX QUALIFIED, teacher capture "
            "executed_units == units > 0, specimens on disk). Until then the "
            "WorkUnit stays status=sleeping and is_ready is false."
        ),
        "blocker": _sleeping_blocked_reason(capture),
        "missing_dependency": (
            "Metal-capable GPU + Metal compiler + HCLI-held protected "
            "accelerator lease + completed teacher capture + specimen tensors "
            "on disk. apply_to_weights currently raises WeightsFrozen."
        ),
        "condition": fit_wake_body,
        "evidence_tier": "STATIC",
    }
    drift_wake = {
        "schema": WAKE_SCHEMA,
        "kind": "SPECIMEN_DRIFT_INSTRUMENT",
        "required_kind": WAKE_REQUIRED_KIND,
        "required_symbol": "tools.tabula_drift.recover_direction",
        "required_caller_prefix": "hcli/",
        "predicate": (
            "production AST Call of tools.tabula_drift.recover_direction (or "
            "tools.gravity_tabula_probe.smallest_left_dir) from an HCLI entry "
            "after language_model.model.layers.*.mlp.down_proj.weight tensors "
            "for the abliterated patient are loadable from "
            "workspace/campaign/records/runs/qwen38-27b/bf16"
        ),
        "blocker": (
            "G123 instrument exists as tools/tabula_drift.py and "
            "tools/gravity_tabula_probe.py but needs the qwen38-27b bf16 "
            "specimen; this sidecar never loads those weights. Doctor seal "
            "records instrument_validated=false (ladder does not reproduce "
            "the recorded range)."
        ),
        "missing_dependency": (
            "abliterated qwen38-27b bf16 tensors + abliteration-manifest.json "
            "at workspace/campaign/records/runs/qwen38-27b/bf16"
        ),
        "evidence_tier": "STATIC",
    }
    behaviour_wake = {
        "schema": WAKE_SCHEMA,
        "kind": "BEHAVIOURAL_REFUSAL_PROBE",
        "required_kind": WAKE_REQUIRED_KIND,
        "required_symbol": "tools.gravity_tabula_behaviour.run",
        "required_caller_prefix": "hcli/",
        "predicate": (
            "production AST Call of tools.gravity_tabula_behaviour.run from an "
            "HCLI entry after the hybrid greedy binary and tokenizer exist. "
            "The probe is the WEAKER half and cannot certify absence of drift."
        ),
        "blocker": (
            "tools/gravity_tabula_behaviour.py is a CLI that shells out to "
            "workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy; "
            "marker-based refusal counts are not an independent-evaluation vector"
        ),
        "missing_dependency": (
            "ascension_qwen38_hybrid_greedy binary + tokenizer.json + artifact "
            "roots under workspace/campaign/records/runs/qwen38-27b"
        ),
        "evidence_tier": "STATIC",
    }
    return {
        "schema": DISPOSITION_SCHEMA,
        "subsystem": "tabula",
        "what_it_is": (
            "Doctor axis: behavioral surgery. Recover the abliterated refusal "
            "direction as a left-null vector of residual-writing tensors, "
            "project it out (W'=(I-vv^T)W), score the child on an independent "
            "vector (behavioral, capability, tool_use, reasoning, "
            "instruction_following). Zero refusal is never the only score. "
            "Permission is not personality."
        ),
        "not_this": (
            "H-ROADMAP.md 'tabular / catalog data' (Art. 70.2) is unrelated. "
            "Tabula is not a refusal-rate contest and not Gravity."
        ),
        "roadmap": {
            "I-B_Doctor": (
                "experiment ranking / negative-science / capability contracts; "
                "Tabula is the behavioral-surgery gene Doctor verifies beside Gravity"
            ),
            "named_section": None,
            "note": (
                "H-ROADMAP.md does not name Tabula as a subsystem; the live "
                "name is this Doctor axis plus G123/G1 instruments"
            ),
        },
        "doctrine": DOCTRINE,
        "items": [
            {
                "id": "tabula.floor",
                "disposition": "CONNECTED",
                "implementation": "tools/future/tabula.py",
                "symbol": "tools.future.tabula.evaluate",
                "also_called": [
                    "tools.future.tabula.project",
                    "tools.future.tabula.rank",
                    "tools.future.tabula.catalog",
                ],
                "call_sites": [dict(s) for s in FLOOR_CALL_SITES],
                "test": "tools/future/test_tabula.py",
                "invoke": (
                    "python3 tools/audit/reachability_triage.py --invoke "
                    "future.tabula --args '{\"scores\":{...}}'"
                ),
                "evidence_tier": "FUNCTIONAL_SIM",
                "wake": None,
            },
            {
                "id": "tabula.fit-weights",
                "disposition": "PARKED",
                "implementation": "tools/future/tabula.py",
                "symbol": "tools.future.tabula.apply_to_weights",
                "workunit_id": "future.tabula.fit-weights",
                "status": "sleeping",
                "evidence_tier": "STATIC",
                "wake": fit_wake,
            },
            {
                "id": "tabula.drift-instrument",
                "disposition": "PARKED",
                "implementation": "tools/tabula_drift.py",
                "also": [
                    "tools/gravity_tabula_probe.py",
                    "tools/doctor_seal.py (requires a tabula_drift cell; does not call the instrument)",
                    "tools/cost_vector_t.py (quotes the G123 ladder; does not remeasure)",
                ],
                "evidence_tier": "STATIC",
                "wake": drift_wake,
            },
            {
                "id": "tabula.behaviour-probe",
                "disposition": "PARKED",
                "implementation": "tools/gravity_tabula_behaviour.py",
                "evidence_tier": "STATIC",
                "wake": behaviour_wake,
                "note": "explicitly the WEAKER half; cannot certify absence of drift",
            },
        ],
        "claim_boundary": CLAIM_BOUNDARY,
        "gpu_authority": False,
        "weights_modified": False,
        "evidence_class": "STATIC_ONLY",
    }


def build() -> Path:
    floor = TabulaFloor()
    contracts = catalog(seed=DEFAULT_SEED)
    proofs = [run_contract(c) for c in contracts]
    # Reproducibility: same seed, same identity.
    again = [c.identity() for c in catalog(seed=DEFAULT_SEED)]
    if again != [c.identity() for c in contracts]:
        raise ExperimentContractError("catalog is not deterministic under a fixed seed")
    recovered = recover_tabula()
    capture = teacher_capture_progress()
    units = emit_workunits(contracts=contracts, lattice=floor.lattice, capture=capture)
    if not sleeping_unit_is_not_ready(units):
        raise AssertionError("sleeping fit unit was ready; HCLI would dispatch a GPU fit")
    refusals = _prove_negative_controls()
    if not refusals or not all(r.get("refused") for r in refusals):
        raise AssertionError(f"negative controls did not all fire: {refusals}")

    demo_fail = evaluate(
        ScoreVector(
            behavioral=0.95,
            capability=0.10,
            tool_use=-0.80,
            reasoning=0.05,
            instruction_following=0.04,
        )
    )
    demo_pass = evaluate(
        ScoreVector(
            behavioral=0.70,
            capability=0.05,
            tool_use=0.02,
            reasoning=0.01,
            instruction_following=0.00,
        )
    )
    ranked = rank(
        [
            {"id": "kill-tools", "scores": demo_fail.scores},
            {"id": "balanced", "scores": demo_pass.scores},
        ]
    )
    child = emit_child(
        parent_id="synthetic-parent-0",
        contract=next(c for c in contracts if c.id == "TAB-REV-001"),
        scores=demo_pass.scores,
        lattice=floor.lattice,
        invert_doc=next(p["invert"][0] for p in proofs if p["contract_id"] == "TAB-REV-001"),
    )

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": SIDECAR_STATUS,
        "promoted": False,
        "built": True,
        "purpose": (
            "Behavioral-surgery experimental floor evaluated independently. "
            "Contracts, scorer, lineage, authority lattice. No weight write."
        ),
        "doctrine": recovered_doctrine(),
        "head": git("rev-parse", "HEAD"),
        "vocabulary": {
            "eras": list(ERAS),
            "odysseys": list(ODYSSEYS),
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "fpga_is": (
                "Accelerator / Physical Compiler / Fusion; not a civilization "
                "and not a Tabula concern"
            ),
            "disk_state_is_authority": True,
            "models_think_tools_know_context_is_a_cache": True,
            "diagnostic_relative_never_promotes": True,
            "protected_absolute_not_emitted": True,
            "zero_refusal_is_never_the_only_score": True,
            "permission_is_not_personality": True,
        },
        "score_axes": list(SCORE_AXES),
        "contract_kinds": list(CONTRACT_KINDS),
        "experiment_contracts": [c.to_dict() for c in contracts],
        "geometry_proofs": proofs,
        "independent_evaluation": {
            "behavioral_target": BEHAVIORAL_TARGET,
            "regression_limit": REGRESSION_LIMIT,
            "demo_failure": demo_fail.to_dict(),
            "demo_pass": demo_pass.to_dict(),
            "ranked": ranked,
            "ranking_rule": ranked[0]["ranking_rule"] if ranked else None,
        },
        "authority_lattice": floor.lattice.to_dict(),
        "lineage_demo": child["lineage"],
        "weights_modified": False,
        "sleeping_fit": {
            "workunit_id": "future.tabula.fit-weights",
            "status": "sleeping",
            "ready": False,
            "wake_condition": fitting_wake_condition(capture),
            "teacher_capture": capture,
        },
        "work_units": list(units),
        "refusals_proven": list(refusals),
        "recovered_implementation": recovered,
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(recovered, capture),
        "resident_callable": resident_callable(units=units, refusals=refusals),
        "disposition": disposition(),
        "promote_exists": hasattr(TabulaFloor, "promote"),
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    refusals = _prove_negative_controls()
    if not all(r.get("refused") for r in refusals):
        raise AssertionError(f"expected watched refusals, got {refusals}")
    contracts = catalog(seed=DEFAULT_SEED)
    kinds = {c.kind for c in contracts}
    if kinds != set(CONTRACT_KINDS):
        raise AssertionError(f"catalog missing kinds {set(CONTRACT_KINDS) - kinds}")
    proofs = [run_contract(c) for c in contracts]
    rev = next(p for p in proofs if p["contract_id"] == "TAB-REV-001")
    if rev["invert_frobenius_error"] is None or rev["invert_frobenius_error"] > 1e-9:
        raise AssertionError(f"reversible invert error {rev['invert_frobenius_error']}")
    irr = next(c for c in contracts if c.id == "TAB-IRR-001")
    if irr.reversible:
        raise AssertionError("TAB-IRR-001 must be irreversible")
    layer = next(p for p in proofs if p["contract_id"] == "TAB-LAYER-001")
    if layer["off_destination_unchanged"] is not True:
        raise AssertionError("layer effect leaked off destination layers")
    identities_a = [c.identity() for c in catalog(seed=DEFAULT_SEED)]
    identities_b = [c.identity() for c in catalog(seed=DEFAULT_SEED)]
    if identities_a != identities_b:
        raise AssertionError("contracts are not reproducible under a fixed seed")
    floor = TabulaFloor()
    units = emit_workunits(
        contracts=contracts, lattice=floor.lattice, capture=teacher_capture_progress()
    )
    if not sleeping_unit_is_not_ready(units):
        raise AssertionError("sleeping unit was ready")
    disp = disposition()
    items = {row["id"]: row for row in disp["items"]}
    if items["tabula.floor"]["disposition"] != "CONNECTED":
        raise AssertionError("floor must be CONNECTED")
    for parked_id in ("tabula.fit-weights", "tabula.drift-instrument", "tabula.behaviour-probe"):
        row = items[parked_id]
        if row["disposition"] != "PARKED":
            raise AssertionError(f"{parked_id} must be PARKED")
        wake = row.get("wake") or {}
        if not wake.get("predicate") or not wake.get("missing_dependency"):
            raise AssertionError(f"{parked_id} PARKED without a wake")
        if wake.get("required_kind") != WAKE_REQUIRED_KIND:
            raise AssertionError(f"{parked_id} wake required_kind is not call")
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--disposition", action="store_true")
    args = ap.parse_args()
    if args.disposition:
        print(json.dumps(disposition(), indent=2, sort_keys=True))
        return 0
    if args.selftest:
        print(selftest())
        return 0
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
