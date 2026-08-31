#!/usr/bin/env python3
"""IMPROVEMENT_METABOLISM — the causal option substrate.

Hawking has agency, mutation, sandboxes, receipts and autonomy. What it did
not have is a machine-readable model of WHAT TO DO NEXT AND WHY. This module
is that model: a nested causal hypothesis tree per frontier, scientific
WorkUnit roles a scheduler can balance, terminal states with no limbo, and
an ingest path that every landed result is required to change.

Tenet: HAWKING IMPROVES BY COLLAPSING UNCERTAINTY FASTER THAN IT CREATES IT.
Law:  EVERY RESULT MUST CHANGE WHAT HAWKING DOES NEXT.

    python3 tools/future/improvement_metabolism.py --build
    python3 -m pytest tools/future/test_improvement_metabolism.py -q

x2 (tools/future/improvement_trial.py) imports this module. The public
surface is the names in ``__all__``. Do not grow it without a caller.

evidence_class STATIC_ONLY. No GPU lease. Numbers are cited from landed
receipts, never invented here.
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
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from tools.future._common import REPO, git, write_receipt, _assert_no_hardware_claims


RECEIPT = "IMPROVEMENT_METABOLISM.json"
SCHEMA = "hawking.future.improvement_metabolism.v1"
VERSION = 1
RECORDED_BY = "tools/future/improvement_metabolism.py"

TENET = (
    "HAWKING IMPROVES BY COLLAPSING UNCERTAINTY FASTER THAN IT CREATES IT"
)
LAW = "EVERY RESULT MUST CHANGE WHAT HAWKING DOES NEXT"

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. Every GB/s, ms, byte "
    "and occupancy figure is cited from a named landed receipt; this module "
    "does not re-measure and does not hold a GPU lease. MIXED is a first-class "
    "verdict: strong evidence for a branch with a pre-registered rule unsatisfied "
    "is not a pass and is not a fail."
)

# ---------------------------------------------------------------------------
# Vocabulary. Keep these strings stable: x2 imports the names, not the values
# as magic. Changing a token is a contract break.
# ---------------------------------------------------------------------------

PROBE = "PROBE"
FALSIFIER = "FALSIFIER"
ORACLE = "ORACLE"
MUTATION = "MUTATION"
REPLICATION = "REPLICATION"
QUALIFICATION = "QUALIFICATION"
ADVERSARY = "ADVERSARY"
ROLES: tuple[str, ...] = (
    PROBE,
    FALSIFIER,
    ORACLE,
    MUTATION,
    REPLICATION,
    QUALIFICATION,
    ADVERSARY,
)

KEEP = "KEEP"
ROLLBACK = "ROLLBACK"
PARK = "PARK"
SCAR = "SCAR"
TERMINALS: tuple[str, ...] = (KEEP, ROLLBACK, PARK, SCAR)

OPEN = "OPEN"
PROMOTED = "PROMOTED"
DEMOTED = "DEMOTED"
MIXED = "MIXED"
KILLED = "KILLED"
SETTLED = "SETTLED"
PARKED = "PARKED"
STATUSES: tuple[str, ...] = (
    OPEN,
    PROMOTED,
    DEMOTED,
    MIXED,
    KILLED,
    SETTLED,
    PARKED,
)

POLARITY_PROMOTE = "PROMOTE"
POLARITY_DEMOTE = "DEMOTE"
POLARITY_MIXED = "MIXED"
POLARITIES: tuple[str, ...] = (POLARITY_PROMOTE, POLARITY_DEMOTE, POLARITY_MIXED)

INGEST_CHANGED_NOTHING = "ingest_changed_nothing"

RESOURCES: tuple[str, ...] = (
    "CPU",
    "ANALYSIS",
    "SIMULATION",
    "REPRESENTATION",
    "TOOLING",
    "ODYSSEY",
    "GPU_PROTECTED",
)

# Source receipts the campaign tree is required to cite. Unseen is a defect.
REQUIRED_SOURCES: tuple[str, ...] = (
    "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json",
    "receipts/future/MLP_ALU_ROOFLINE.json",
    "receipts/future/MLP_FUNCTIONAL_RANK.json",
    "receipts/future/MLP_NONLINEAR_PROGRAM.json",
    "receipts/future/MLP_SHARED_PROGRAM.json",
    "receipts/future/DELTANET_GENERATED_TRANSITION.json",
    "receipts/future/DELTANET_MULTISTEP.json",
    "receipts/future/PATH_TO_71.json",
    "receipts/future/EXECUTABLE_ECONOMICS.json",
    "receipts/future/ODYSSEY_LAUNCH_GATE.json",
)

HARD_VERBS = frozenset(
    {
        "investigate",
        "explore",
        "examine",
        "research",
        "consider",
        "review",
        "analyze",
        "analyse",
        "study",
        "understand",
        "think",
        "inspect",
        "assess",
        "evaluate",
        "brainstorm",
        "ponder",
        "discuss",
        "debug",
        "hunt",
        "figure",
        "look",
        "contemplate",
        "speculate",
        "wonder",
    }
)
SOFT_VERBS = frozenset(
    {
        "check",
        "measure",
        "probe",
        "profile",
        "compare",
        "run",
        "test",
        "try",
        "verify",
        "confirm",
        "revisit",
        "reopen",
    }
)
_COMMAND_RE = re.compile(
    r"(?i)(python3?\b|\bcargo\b|\bpytest\b|\bxcrun\b|\.py\b|\.rs\b|"
    r"receipts/|tools/|crates/|--[\w-]+)"
)
_SPECIFIC_RE = re.compile(
    r"(?i)(\d|gb/s|gb_s|weight_bytes|fma|threadgroup|kernel|receipt|"
    r"matched pair|arm [ab]|layer\b|command buffer|falsif|"
    r"held-out|relative l2|monarch|butterfly)"
)
_FIRST_WORD = re.compile(r"[A-Za-z]+")


# ---------------------------------------------------------------------------
# Errors. A missing field is a refusal, never a default.
# ---------------------------------------------------------------------------


class NotAHypothesis(ValueError):
    """A node without an exact falsifier is not a hypothesis."""


class VerbExperiment(ValueError):
    """cheapest_decisive_experiment is a verb, not a command or probe."""


class ParkWithoutWake(ValueError):
    """PARK with no wake condition is limbo; the module refuses it."""


class ScarIncomplete(ValueError):
    """SCAR without scope or REOPEN_IF is not a scar."""


class MissingReceipt(FileNotFoundError):
    """A required campaign receipt is unseen in this checkout and in git."""


class UnknownRole(ValueError):
    """WorkUnit scientific role is not one of the seven."""


class UnknownTerminal(ValueError):
    """Experiment outcome is not KEEP / ROLLBACK / PARK / SCAR."""


class UnknownPolarity(ValueError):
    """Ingest polarity is not PROMOTE / DEMOTE / MIXED."""


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass
class Hypothesis:
    """One nested causal claim. Children nest; a flat queue is not this."""

    id: str
    title: str
    prior_confidence: float
    max_possible_gain_ms: float
    cheapest_decisive_experiment: str
    expected_runtime_s: float
    required_resource: str
    falsifier: str
    status: str = OPEN
    reopen_if: str | None = None
    wake_condition: str | None = None
    children: list[Hypothesis] = field(default_factory=list)
    parent_id: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    evidence_for: bool | None = None
    rule_satisfied: bool | None = None
    formal_verdict: str | None = None

    def walk(self) -> Iterator[Hypothesis]:
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, node_id: str) -> Hypothesis | None:
        for node in self.walk():
            if node.id == node_id:
                return node
        return None

    def ids(self) -> list[str]:
        return [n.id for n in self.walk()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "prior_confidence": self.prior_confidence,
            "max_possible_gain_ms": self.max_possible_gain_ms,
            "cheapest_decisive_experiment": self.cheapest_decisive_experiment,
            "expected_runtime_s": self.expected_runtime_s,
            "required_resource": self.required_resource,
            "falsifier": self.falsifier,
            "status": self.status,
            "reopen_if": self.reopen_if,
            "wake_condition": self.wake_condition,
            "parent_id": self.parent_id,
            "evidence": list(self.evidence),
            "notes": self.notes,
            "evidence_for": self.evidence_for,
            "rule_satisfied": self.rule_satisfied,
            "formal_verdict": self.formal_verdict,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class WorkUnit:
    """Scientific work, not an HCLI WorkUnit. Role is one of ROLES."""

    id: str
    role: str
    hypothesis_id: str
    frontier_id: str
    experiment: str
    required_resource: str
    expected_runtime_s: float
    status: str = "QUEUED"
    terminal: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IngestResult:
    """What a landed receipt did to a tree. ``changed is False`` is a defect."""

    changed: bool
    defect: str | None
    promoted: list[str]
    demoted: list[str]
    mixed: list[str]
    settled: list[str]
    descendants_updated: list[str]
    formal_verdict: str | None
    receipt_name: str
    schema: str | None
    notes: list[str]
    nuance: str | None = None

    @property
    def nothing_changed(self) -> bool:
        return not self.changed

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["nothing_changed"] = self.nothing_changed
        return d


@dataclass
class RoleBalance:
    """Scheduler view: 20 mutations and no falsifier is a visible defect."""

    counts: dict[str, int]
    n: int
    missing_roles: list[str]
    unbalanced: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Frontier:
    """Answers WHY IT EXISTS without a human in the loop."""

    id: str
    objective: str
    current_best: dict[str, Any]
    target: dict[str, Any]
    remaining_gap: dict[str, Any]
    causal_hypotheses: Hypothesis
    scars: list[dict[str, Any]]
    live_candidates: list[dict[str, Any]]
    biggest_unknown: str
    next_decisive_experiments: list[str]
    resource_requirements: list[str]

    def why_it_exists(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "current_best": self.current_best,
            "target": self.target,
            "remaining_gap": self.remaining_gap,
            "biggest_unknown": self.biggest_unknown,
            "live_candidates": list(self.live_candidates),
            "next_decisive_experiments": list(self.next_decisive_experiments),
            "resource_requirements": list(self.resource_requirements),
            "n_hypotheses": sum(1 for _ in self.causal_hypotheses.walk()),
            "n_scars": len(self.scars),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "current_best": self.current_best,
            "target": self.target,
            "remaining_gap": self.remaining_gap,
            "causal_hypotheses": self.causal_hypotheses.to_dict(),
            "scars": list(self.scars),
            "live_candidates": list(self.live_candidates),
            "biggest_unknown": self.biggest_unknown,
            "next_decisive_experiments": list(self.next_decisive_experiments),
            "resource_requirements": list(self.resource_requirements),
            "why_it_exists": self.why_it_exists(),
        }


@dataclass
class Metabolism:
    """The campaign's option substrate: frontiers + work + ingest log."""

    frontiers: dict[str, Frontier]
    work_units: list[WorkUnit] = field(default_factory=list)
    ingest_log: list[IngestResult] = field(default_factory=list)
    cited: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    def ingest(self, receipt: Mapping[str, Any] | str) -> IngestResult:
        result = ingest(self, receipt)
        self.ingest_log.append(result)
        return result

    def role_balance(self) -> RoleBalance:
        return role_balance(self.work_units)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frontiers": {k: v.to_dict() for k, v in self.frontiers.items()},
            "work_units": [w.to_dict() for w in self.work_units],
            "ingest_log": [r.to_dict() for r in self.ingest_log],
            "cited": self.cited,
            "sources": self.sources,
            "role_balance": self.role_balance().to_dict(),
        }


# ---------------------------------------------------------------------------
# Constructors. Invalid states raise; they are never stored.
# ---------------------------------------------------------------------------


def _strip(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _looks_specific(text: str) -> bool:
    return bool(_COMMAND_RE.search(text) or _SPECIFIC_RE.search(text))


def require_experiment(text: Any) -> str:
    """Refuse a verb. Accept an exact command or a specific probe description."""
    raw = _strip(text)
    if not raw:
        raise VerbExperiment(
            "cheapest_decisive_experiment is empty; a hypothesis needs an "
            "exact command or probe description, never a verb"
        )
    first_m = _FIRST_WORD.match(raw)
    first = first_m.group(0).lower() if first_m else ""
    if raw.lower() in HARD_VERBS or raw.lower() in SOFT_VERBS:
        raise VerbExperiment(
            f"{raw!r} is a verb, not a command or probe description"
        )
    if first in HARD_VERBS:
        raise VerbExperiment(
            f"{raw!r} starts with the verb {first!r}; give an exact command "
            "or a probe that names a kernel, receipt, flag, or measurement"
        )
    if first in SOFT_VERBS and not _looks_specific(raw):
        raise VerbExperiment(
            f"{raw!r} is a vague {first!r}, not an exact command or probe"
        )
    return raw


def require_falsifier(text: Any) -> str:
    raw = _strip(text)
    if not raw:
        raise NotAHypothesis(
            "a node without a falsifier is not a hypothesis; the module refuses it"
        )
    return raw


def hypothesis(
    *,
    id: str,
    title: str,
    prior_confidence: float,
    max_possible_gain_ms: float,
    cheapest_decisive_experiment: str,
    expected_runtime_s: float,
    required_resource: str,
    falsifier: str,
    status: str = OPEN,
    reopen_if: str | None = None,
    wake_condition: str | None = None,
    children: Sequence[Hypothesis] = (),
    notes: str = "",
    evidence_for: bool | None = None,
    rule_satisfied: bool | None = None,
    formal_verdict: str | None = None,
) -> Hypothesis:
    """Construct a hypothesis node. Missing falsifier / verb experiment refuse."""
    node_id = _strip(id)
    if not node_id:
        raise NotAHypothesis("hypothesis id is required")
    if not _strip(title):
        raise NotAHypothesis("hypothesis title is required")
    conf = float(prior_confidence)
    if not 0.0 <= conf <= 1.0:
        raise NotAHypothesis(f"prior_confidence {conf} is outside [0, 1]")
    gain = float(max_possible_gain_ms)
    if gain < 0.0:
        raise NotAHypothesis(f"max_possible_gain_ms {gain} is negative")
    runtime = float(expected_runtime_s)
    if runtime < 0.0:
        raise NotAHypothesis(f"expected_runtime_s {runtime} is negative")
    resource = _strip(required_resource)
    if not resource:
        raise NotAHypothesis("required_resource is empty")
    status_u = _strip(status).upper() or OPEN
    if status_u not in STATUSES:
        raise NotAHypothesis(f"unknown hypothesis status {status_u!r}")
    reopen = _strip(reopen_if) or None
    wake = _strip(wake_condition) or None
    if status_u == KILLED and not reopen:
        raise ScarIncomplete(
            f"killed hypothesis {node_id!r} requires REOPEN_IF"
        )
    if status_u == PARKED and not wake:
        raise ParkWithoutWake(
            f"parked hypothesis {node_id!r} requires a wake condition"
        )
    node = Hypothesis(
        id=node_id,
        title=_strip(title),
        prior_confidence=conf,
        max_possible_gain_ms=gain,
        cheapest_decisive_experiment=require_experiment(
            cheapest_decisive_experiment
        ),
        expected_runtime_s=runtime,
        required_resource=resource,
        falsifier=require_falsifier(falsifier),
        status=status_u,
        reopen_if=reopen,
        wake_condition=wake,
        notes=_strip(notes),
        evidence_for=evidence_for,
        rule_satisfied=rule_satisfied,
        formal_verdict=_strip(formal_verdict) or None,
    )
    for child in children:
        attach(node, child)
    return node


def attach(parent: Hypothesis, child: Hypothesis) -> Hypothesis:
    """Nest ``child`` under ``parent``. Ids must be unique in the subtree."""
    if child.id in parent.ids():
        raise NotAHypothesis(
            f"duplicate hypothesis id {child.id!r} under {parent.id!r}"
        )
    child.parent_id = parent.id
    parent.children.append(child)
    return child


def kill_hypothesis(node: Hypothesis, *, reopen_if: str, note: str = "") -> Hypothesis:
    """Mark a node KILLED. REOPEN_IF is mandatory."""
    reopen = _strip(reopen_if)
    if not reopen:
        raise ScarIncomplete(
            f"refusing to kill {node.id!r} without REOPEN_IF"
        )
    node.status = KILLED
    node.reopen_if = reopen
    if note:
        node.notes = (node.notes + " " + note).strip()
    return node


def terminal(kind: str, **kwargs: Any) -> dict[str, Any]:
    """KEEP / ROLLBACK / PARK / SCAR. PARK needs wake; SCAR needs scope+reopen."""
    k = _strip(kind).upper()
    if k not in TERMINALS:
        raise UnknownTerminal(
            f"{kind!r} is not a terminal state; every experiment ends "
            "KEEP, ROLLBACK, PARK or SCAR"
        )
    wake = _strip(kwargs.get("wake_condition"))
    scope = _strip(kwargs.get("scope"))
    reopen = _strip(kwargs.get("reopen_if"))
    if k == PARK and not wake:
        raise ParkWithoutWake("PARK requires a wake condition")
    if k == SCAR and (not scope or not reopen):
        raise ScarIncomplete("SCAR requires scope and REOPEN_IF")
    out = {
        "kind": k,
        "hypothesis_id": kwargs.get("hypothesis_id"),
        "note": _strip(kwargs.get("note")) or None,
    }
    if k == PARK:
        out["wake_condition"] = wake
    if k == SCAR:
        out["scope"] = scope
        out["reopen_if"] = reopen
    if k == ROLLBACK:
        out["restores"] = kwargs.get("restores")
    return out


def apply_terminal(node: Hypothesis, term: Mapping[str, Any]) -> Hypothesis:
    """Bind a terminal experiment outcome onto a hypothesis. No limbo."""
    kind = _strip(term.get("kind")).upper()
    # Re-validate so a hand-built dict cannot smuggle PARK/SCAR without fields.
    terminal(kind, **{k: v for k, v in term.items() if k != "kind"})
    if kind == PARK:
        node.status = PARKED
        node.wake_condition = _strip(term.get("wake_condition"))
    elif kind == SCAR:
        kill_hypothesis(
            node,
            reopen_if=_strip(term.get("reopen_if")),
            note=_strip(term.get("note")),
        )
    elif kind == KEEP:
        # KEEP the measurement. It does not settle the hypothesis as true.
        node.evidence.append({"terminal": KEEP, "note": term.get("note")})
    elif kind == ROLLBACK:
        node.evidence.append({"terminal": ROLLBACK, "note": term.get("note")})
    return node


def work_unit(
    *,
    id: str,
    role: str,
    hypothesis_id: str,
    frontier_id: str,
    experiment: str,
    required_resource: str,
    expected_runtime_s: float,
    status: str = "QUEUED",
    terminal: Mapping[str, Any] | None = None,
) -> WorkUnit:
    r = _strip(role).upper()
    if r not in ROLES:
        raise UnknownRole(
            f"{role!r} is not a scientific role; expected one of {ROLES}"
        )
    if not _strip(id):
        raise ValueError("work_unit id is required")
    if not _strip(hypothesis_id):
        raise ValueError("work_unit hypothesis_id is required")
    if not _strip(frontier_id):
        raise ValueError("work_unit frontier_id is required")
    resource = _strip(required_resource)
    if not resource:
        raise ValueError("work_unit required_resource is empty")
    term = dict(terminal) if terminal else None
    if term is not None:
        # Validate now so a bad PARK/SCAR cannot sit on the queue.
        kind = term.get("kind", "")
        rest = {k: v for k, v in term.items() if k != "kind"}
        term = globals()["terminal"](kind, **rest)
    return WorkUnit(
        id=_strip(id),
        role=r,
        hypothesis_id=_strip(hypothesis_id),
        frontier_id=_strip(frontier_id),
        experiment=require_experiment(experiment),
        required_resource=resource,
        expected_runtime_s=float(expected_runtime_s),
        status=_strip(status) or "QUEUED",
        terminal=term,
    )


def role_balance(units: Iterable[WorkUnit] | Iterable[Mapping[str, Any]]) -> RoleBalance:
    counts = {r: 0 for r in ROLES}
    n = 0
    for u in units:
        role = u.role if isinstance(u, WorkUnit) else _strip(u.get("role")).upper()
        if role in counts:
            counts[role] += 1
        n += 1
    missing = [r for r in ROLES if counts[r] == 0]
    mutations = counts[MUTATION]
    falsifiers = counts[FALSIFIER]
    unbalanced = mutations > 0 and falsifiers == 0
    if unbalanced:
        note = (
            f"{mutations} mutations queued and no falsifier; a scheduler "
            "that runs this queue creates uncertainty faster than it collapses it"
        )
    elif missing:
        note = "missing roles: " + ", ".join(missing)
    else:
        note = "all seven scientific roles are present"
    return RoleBalance(
        counts=counts,
        n=n,
        missing_roles=missing,
        unbalanced=unbalanced,
        note=note,
    )


def frontier(
    *,
    id: str,
    objective: str,
    current_best: Mapping[str, Any],
    target: Mapping[str, Any],
    remaining_gap: Mapping[str, Any],
    causal_hypotheses: Hypothesis,
    scars: Sequence[Mapping[str, Any]] = (),
    live_candidates: Sequence[Mapping[str, Any]] = (),
    biggest_unknown: str,
    next_decisive_experiments: Sequence[str] = (),
    resource_requirements: Sequence[str] = (),
) -> Frontier:
    if not _strip(id):
        raise ValueError("frontier id is required")
    obj = _strip(objective)
    if not obj:
        raise ValueError("frontier objective is required; it must answer WHY IT EXISTS")
    unk = _strip(biggest_unknown)
    if not unk:
        raise ValueError("frontier biggest_unknown is required")
    return Frontier(
        id=_strip(id),
        objective=obj,
        current_best=dict(current_best),
        target=dict(target),
        remaining_gap=dict(remaining_gap),
        causal_hypotheses=causal_hypotheses,
        scars=[dict(s) for s in scars],
        live_candidates=[dict(c) for c in live_candidates],
        biggest_unknown=unk,
        next_decisive_experiments=[_strip(x) for x in next_decisive_experiments if _strip(x)],
        resource_requirements=[_strip(x) for x in resource_requirements if _strip(x)],
    )


# ---------------------------------------------------------------------------
# Receipt loading. Sparse checkout is not evidence of absence.
# ---------------------------------------------------------------------------


def load_receipt(rel: str) -> dict[str, Any]:
    """Load a JSON object from disk or ``git show HEAD:<rel>``."""
    data, origin = load_receipt_origin(rel)
    if data is None:
        raise MissingReceipt(f"{rel} unseen ({origin})")
    return data


def load_receipt_origin(rel: str) -> tuple[dict[str, Any] | None, str]:
    rel = rel.lstrip("./")
    path = REPO / rel
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return None, f"unreadable:{path}:{exc}"
        if isinstance(data, dict):
            return data, f"disk:{path}"
        return None, f"not_object:{path}"
    blob = git("show", f"HEAD:{rel}")
    if blob:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError as exc:
            return None, f"git_unreadable:HEAD:{rel}:{exc}"
        if isinstance(data, dict):
            return data, f"git:HEAD:{rel}"
        return None, f"git_not_object:HEAD:{rel}"
    return None, "unseen_in_this_checkout"


# ---------------------------------------------------------------------------
# Confidence updates. Deterministic; MIXED rises but stays below a settle.
# ---------------------------------------------------------------------------


def _promote_conf(prior: float) -> float:
    return round(min(0.92, prior + (1.0 - prior) * 0.55), 4)


def _demote_conf(prior: float) -> float:
    return round(prior * 0.22, 4)


def _mixed_conf(prior: float) -> float:
    # Strong evidence, rule not satisfied: rise, but never into settle territory.
    return round(min(0.78, max(prior, 0.55) + 0.12), 4)


def _apply_polarity(
    node: Hypothesis,
    polarity: str,
    *,
    receipt_name: str,
    note: str,
    extra: Mapping[str, Any] | None = None,
) -> str:
    pol = _strip(polarity).upper()
    if pol not in POLARITIES:
        raise UnknownPolarity(f"{polarity!r} is not PROMOTE, DEMOTE or MIXED")
    before = node.prior_confidence
    extra = dict(extra or {})
    if pol == POLARITY_PROMOTE:
        node.status = PROMOTED
        node.prior_confidence = _promote_conf(before)
        node.evidence_for = True
        node.rule_satisfied = extra.get("rule_satisfied", True)
    elif pol == POLARITY_DEMOTE:
        node.status = DEMOTED
        node.prior_confidence = _demote_conf(before)
        node.evidence_for = False
        node.rule_satisfied = extra.get("rule_satisfied", False)
    else:
        node.status = MIXED
        node.prior_confidence = _mixed_conf(before)
        node.evidence_for = bool(extra.get("evidence_for", True))
        node.rule_satisfied = bool(extra.get("rule_satisfied", False))
        node.formal_verdict = extra.get("formal_verdict") or MIXED
        if extra.get("nuance"):
            node.notes = extra["nuance"]
    if extra.get("formal_verdict"):
        node.formal_verdict = extra["formal_verdict"]
    if extra.get("notes"):
        node.notes = (node.notes + " " + extra["notes"]).strip()
    node.evidence.append(
        {
            "receipt": receipt_name,
            "polarity": pol,
            "confidence_before": before,
            "confidence_after": node.prior_confidence,
            "note": note,
            "evidence_for": node.evidence_for,
            "rule_satisfied": node.rule_satisfied,
            "formal_verdict": node.formal_verdict,
        }
    )
    return node.id


def _already_applied(node: Hypothesis, receipt_name: str) -> bool:
    return any(e.get("receipt") == receipt_name for e in node.evidence)


def _propagate_demote(
    node: Hypothesis,
    receipt_name: str,
    *,
    skip: set[str],
) -> list[str]:
    updated: list[str] = []
    for child in node.children:
        if child.id in skip:
            updated.extend(_propagate_demote(child, receipt_name, skip=skip))
            continue
        if child.status in {KILLED, SETTLED, PARKED}:
            continue
        if child.status == OPEN or child.status == PROMOTED:
            _apply_polarity(
                child,
                POLARITY_DEMOTE,
                receipt_name=receipt_name,
                note=f"inherited demotion from {node.id}",
            )
            updated.append(child.id)
        updated.extend(_propagate_demote(child, receipt_name, skip=skip))
    return updated


# ---------------------------------------------------------------------------
# Ingest. A result that changes nothing is an explicit defect.
# ---------------------------------------------------------------------------


def _receipt_name_of(receipt: Mapping[str, Any] | str) -> tuple[str, dict[str, Any]]:
    if isinstance(receipt, str):
        rel = receipt if receipt.startswith("receipts/") else f"receipts/future/{receipt}"
        doc = load_receipt(rel)
        return Path(rel).name, doc
    name = str(
        receipt.get("recorded_by")
        or receipt.get("schema")
        or receipt.get("source")
        or "anonymous_receipt"
    )
    if isinstance(receipt.get("recorded_by"), str) and receipt["recorded_by"].endswith(".py"):
        # Prefer the conventional receipt filename when we can recover it.
        pass
    # Common case: schema hawking.future.mlp_alu_roofline.v1 -> MLP_ALU_ROOFLINE.json
    schema = str(receipt.get("schema") or "")
    mapped = {
        "hawking.future.mlp_alu_roofline.v1": "MLP_ALU_ROOFLINE.json",
        "hawking.future.causal_budget_71.v1": "RESIDENT_71TPS_CAUSAL_BUDGET.json",
        "hawking.future.mlp_functional_rank.v1": "MLP_FUNCTIONAL_RANK.json",
        "hawking.future.mlp_nonlinear_program.v1": "MLP_NONLINEAR_PROGRAM.json",
        "hawking.future.mlp_shared_program.v1": "MLP_SHARED_PROGRAM.json",
        "hawking.future.deltanet_generated_transition.v1": "DELTANET_GENERATED_TRANSITION.json",
        "hawking.future.deltanet_multistep.v1": "DELTANET_MULTISTEP.json",
        "hawking.future.path_to_71.v1": "PATH_TO_71.json",
        "hawking.future.executable_economics.v1": "EXECUTABLE_ECONOMICS.json",
        "hawking.future.odyssey_launch.v1": "ODYSSEY_LAUNCH_GATE.json",
    }
    return mapped.get(schema, Path(name).name), dict(receipt)


def _empty_ingest(name: str, schema: str | None, note: str) -> IngestResult:
    return IngestResult(
        changed=False,
        defect=INGEST_CHANGED_NOTHING,
        promoted=[],
        demoted=[],
        mixed=[],
        settled=[],
        descendants_updated=[],
        formal_verdict=None,
        receipt_name=name,
        schema=schema,
        notes=[note],
        nuance=None,
    )


def _patches_mlp_alu(doc: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    """Translate the landed ALU receipt into polarities. Do not re-judge."""
    mlp = doc["mlp"]
    judged = mlp["judgement"]
    occ = mlp["production"].get("occupancy") or {}
    decode = mlp.get("decode_tax") or {}
    to_lm = (decode.get("to_match_lm_head_497") or {})
    verdict = str(judged.get("verdict") or doc.get("verdict") or MIXED)
    why = str(judged.get("why_not_forced") or "")
    nuance = (
        "strong evidence for B, rule not satisfied: ARM A jumped "
        f"{judged.get('arm_a_over_production')}x from {judged.get('production_gb_s')} "
        f"to {judged.get('arm_a_gb_s')} GB/s on the same "
        f"{mlp['production']['weight_bytes']} bytes with loads surviving, but "
        "the pre-registered ALU_BOUND rule needed ARM B to scale sub-linearly "
        f"and ARM B time/byte was {judged.get('arm_b_time_over_byte')} "
        f"(linear={judged.get('arm_b_linear')}, sublinear={judged.get('arm_b_sublinear')}) "
        "because half K also halves FMAs. Formal verdict MIXED, not ALU_BOUND."
    )
    loads = judged.get("loads_survived") or {}
    patches = [
        {
            "node_id": "mlp.why_330",
            "polarity": POLARITY_MIXED,
            "note": nuance,
            "extra": {
                "evidence_for": True,
                "rule_satisfied": False,
                "formal_verdict": verdict,
                "nuance": nuance,
            },
        },
        {
            "node_id": "mlp.why_330.A",
            "polarity": POLARITY_DEMOTE,
            "note": (
                "ARM A kept production bytes and access pattern and jumped; "
                "a bandwidth ceiling on this kernel is not what 329.6 GB/s is."
            ),
        },
        {
            "node_id": "mlp.why_330.A1",
            "polarity": POLARITY_DEMOTE,
            "note": (
                f"same {mlp['production']['weight_bytes']} bytes, ARM A "
                f"{judged.get('arm_a_gb_s')} GB/s vs production "
                f"{judged.get('production_gb_s')}; raw DRAM did not cap the stripped arm."
            ),
        },
        {
            "node_id": "mlp.why_330.A2",
            "polarity": POLARITY_DEMOTE,
            "note": (
                "loads survived "
                f"(stripped {loads.get('stripped_gpu_ns')} ns vs zero-load "
                f"{loads.get('zero_load_gpu_ns')} ns); transaction inefficiency "
                "of the production access pattern is not the ceiling."
            ),
        },
        {
            "node_id": "mlp.why_330.A3",
            "polarity": POLARITY_DEMOTE,
            "note": (
                "occupancy ruled out: "
                f"{mlp.get('threads_per_threadgroup')} threads/threadgroup against "
                f"max {occ.get('max_total_threads_per_threadgroup')}, "
                f"{occ.get('threadgroups_per_core')} threadgroups/core; "
                f"occupancy_limited={judged.get('occupancy_limited')}."
            ),
        },
        {
            "node_id": "mlp.why_330.B",
            "polarity": POLARITY_MIXED,
            "note": (
                "strong evidence for the arithmetic ceiling, rule not satisfied. "
                + why
            ),
            "extra": {
                "evidence_for": True,
                "rule_satisfied": False,
                "formal_verdict": MIXED,
                "nuance": "strong evidence for B, rule not satisfied",
            },
        },
        {
            "node_id": "mlp.why_330.B1",
            "polarity": POLARITY_PROMOTE,
            "note": (
                "production decode costs "
                f"{decode.get('production_decode_fma_per_weight_byte')} "
                "dequant-FMA per weight-byte; 497.4 GB/s at the same issue rate "
                f"needs {to_lm.get('target_decode_fma_per_weight_byte')} "
                f"(cheapening {to_lm.get('required_decode_cheapening')})."
            ),
            "extra": {"rule_satisfied": False},
        },
    ]
    return patches, verdict, nuance


def _handler_for(schema: str | None, name: str) -> str | None:
    if schema == "hawking.future.mlp_alu_roofline.v1" or name == "MLP_ALU_ROOFLINE.json":
        return "mlp_alu"
    if schema == "hawking.future.mlp_nonlinear_program.v1" or name == "MLP_NONLINEAR_PROGRAM.json":
        return "mlp_nonlinear"
    return None


def ingest(
    target: Hypothesis | Frontier | Metabolism,
    receipt: Mapping[str, Any] | str,
    *,
    polarities: Sequence[Mapping[str, Any]] | None = None,
) -> IngestResult:
    """Apply a landed receipt to the affected subtree. Nothing-changed is explicit.

    ``polarities`` is the escape hatch x2 uses for a receipt this module has
    not registered. Each item is ``{node_id, polarity, note, extra?}``.
    """
    name, doc = _receipt_name_of(receipt)
    schema = doc.get("schema") if isinstance(doc.get("schema"), str) else None

    trees: list[Hypothesis] = []
    if isinstance(target, Hypothesis):
        trees = [target]
    elif isinstance(target, Frontier):
        trees = [target.causal_hypotheses]
    elif isinstance(target, Metabolism):
        trees = [f.causal_hypotheses for f in target.frontiers.values()]
    else:
        raise TypeError(f"ingest target {type(target)!r} is not a tree")

    patches: list[dict[str, Any]]
    verdict: str | None = None
    nuance: str | None = None
    if polarities is not None:
        patches = [dict(p) for p in polarities]
    else:
        kind = _handler_for(schema, name)
        if kind == "mlp_alu":
            patches, verdict, nuance = _patches_mlp_alu(doc)
        elif kind == "mlp_nonlinear":
            # Already seeded as scars at campaign(); a second ingest that
            # restates MEASURED_NEGATIVE without new node movement is a defect
            # unless a still-OPEN r-bottleneck remains. Seed kills them.
            patches = []
            for tree in trees:
                for node in tree.walk():
                    if node.status == OPEN and node.id.startswith("mlp.fn.r_bottleneck"):
                        patches.append(
                            {
                                "node_id": node.id,
                                "polarity": POLARITY_DEMOTE,
                                "note": "cheap round MEASURED_NEGATIVE; scar already named",
                            }
                        )
        else:
            patches = []

    promoted: list[str] = []
    demoted: list[str] = []
    mixed: list[str] = []
    descendants: list[str] = []
    notes: list[str] = []
    explicit: set[str] = {str(p.get("node_id")) for p in patches}

    any_change = False
    for patch in patches:
        node_id = str(patch.get("node_id") or "")
        pol = str(patch.get("polarity") or "")
        note = str(patch.get("note") or "")
        extra = patch.get("extra") if isinstance(patch.get("extra"), Mapping) else {}
        found = None
        tree_found = None
        for tree in trees:
            found = tree.find(node_id)
            if found is not None:
                tree_found = tree
                break
        if found is None:
            notes.append(f"no node {node_id}")
            continue
        if _already_applied(found, name) and found.status != OPEN:
            notes.append(f"{node_id} already ingested {name}")
            continue
        _apply_polarity(found, pol, receipt_name=name, note=note, extra=extra)
        any_change = True
        if pol.upper() == POLARITY_PROMOTE:
            promoted.append(node_id)
        elif pol.upper() == POLARITY_DEMOTE:
            demoted.append(node_id)
            if tree_found is not None:
                descendants.extend(
                    _propagate_demote(found, name, skip=explicit)
                )
        else:
            mixed.append(node_id)

    descendants = [d for d in descendants if d not in explicit]
    if descendants:
        any_change = True

    if not any_change:
        return _empty_ingest(
            name,
            schema,
            f"ingesting {name} changed nothing; that is a defect the caller can see",
        )

    # Refresh next-experiments on any frontier whose tree moved.
    if isinstance(target, Frontier):
        _refresh_frontier(target)
    elif isinstance(target, Metabolism):
        for fr in target.frontiers.values():
            _refresh_frontier(fr)

    return IngestResult(
        changed=True,
        defect=None,
        promoted=promoted,
        demoted=demoted,
        mixed=mixed,
        settled=[],
        descendants_updated=descendants,
        formal_verdict=verdict,
        receipt_name=name,
        schema=schema,
        notes=notes,
        nuance=nuance,
    )


def _refresh_frontier(fr: Frontier) -> None:
    live = []
    next_ex = []
    for node in fr.causal_hypotheses.walk():
        if node.status in {OPEN, MIXED, PROMOTED, PARKED}:
            live.append(
                {
                    "id": node.id,
                    "status": node.status,
                    "prior_confidence": node.prior_confidence,
                    "max_possible_gain_ms": node.max_possible_gain_ms,
                    "required_resource": node.required_resource,
                }
            )
            next_ex.append(node.cheapest_decisive_experiment)
    # Rank by gain * remaining uncertainty (1 - confidence).
    ranked = sorted(
        live,
        key=lambda r: r["max_possible_gain_ms"] * (1.0 - r["prior_confidence"]),
        reverse=True,
    )
    fr.live_candidates = ranked
    # Unique experiments, live order.
    seen: set[str] = set()
    ordered: list[str] = []
    by_id = {n.id: n for n in fr.causal_hypotheses.walk()}
    for row in ranked:
        exp = by_id[row["id"]].cheapest_decisive_experiment
        if exp not in seen:
            seen.add(exp)
            ordered.append(exp)
    fr.next_decisive_experiments = ordered[:6]


# ---------------------------------------------------------------------------
# Campaign seed — live receipts, not a fixture.
# ---------------------------------------------------------------------------


def _h(
    node_id: str,
    title: str,
    *,
    conf: float,
    gain: float,
    exp: str,
    runtime: float,
    resource: str,
    falsifier: str,
    children: Sequence[Hypothesis] = (),
    status: str = OPEN,
    reopen_if: str | None = None,
    notes: str = "",
) -> Hypothesis:
    return hypothesis(
        id=node_id,
        title=title,
        prior_confidence=conf,
        max_possible_gain_ms=gain,
        cheapest_decisive_experiment=exp,
        expected_runtime_s=runtime,
        required_resource=resource,
        falsifier=falsifier,
        children=children,
        status=status,
        reopen_if=reopen_if,
        notes=notes,
    )


def _mlp_why_330_tree(*, gain_ms: float, weight_bytes: int) -> Hypothesis:
    """The obvious instance: WHY IS THE MLP AT ~330 GB/s? Nodes nest."""
    a1 = _h(
        "mlp.why_330.A1",
        "A1 raw DRAM",
        conf=0.40,
        gain=gain_ms,
        exp=(
            "matched pair ARM A: production vs stripped-ALU on the same "
            f"weight_bytes={weight_bytes}, MTLCommandBuffer GPUStartTime/GPUEndTime; "
            "command python3 tools/future/mlp_alu_roofline.py --measure --record"
        ),
        runtime=180.0,
        resource="GPU_PROTECTED",
        falsifier=(
            "ARM A (same bytes, arithmetic stripped) jumps >= 1.25x over production "
            "with loads surviving -> raw DRAM is not the ceiling of this kernel"
        ),
    )
    a2 = _h(
        "mlp.why_330.A2",
        "A2 transaction inefficiency",
        conf=0.35,
        gain=gain_ms,
        exp=(
            "ARM A stripped vs zero-load floor vs ARM A half-K on the same "
            f"geo_tpr64 production access pattern, weight_bytes={weight_bytes}; "
            "python3 tools/future/mlp_alu_roofline.py --measure --record"
        ),
        runtime=180.0,
        resource="GPU_PROTECTED",
        falsifier=(
            "stripped time above the zero-load floor and dropping when bytes "
            "halve proves the loads survived; a transaction tax that vanished "
            "when arithmetic vanished is not a memory-transaction ceiling"
        ),
    )
    a3 = _h(
        "mlp.why_330.A3",
        "A3 cache behaviour",
        conf=0.30,
        gain=gain_ms,
        exp=(
            "python3 tools/future/mlp_alu_roofline.py --record; cite "
            "mlp.production.occupancy.threads_per_threadgroup vs "
            "max_total_threads_per_threadgroup and threadgroups_per_core on "
            "qwen_affine_q2_group32_matvec_geo_tpr64_tg128 from "
            "receipts/future/MLP_ALU_ROOFLINE.json"
        ),
        runtime=30.0,
        resource="GPU_PROTECTED",
        falsifier=(
            "threads_per_threadgroup well below max AND threadgroups_per_core >> 1 "
            "rules occupancy/cache-thrash-from-oversubscription out as the ceiling"
        ),
    )
    a = _h(
        "mlp.why_330.A",
        "A bandwidth ceiling",
        conf=0.42,
        gain=gain_ms,
        exp=(
            "matched pair ARM A (bytes identical, arithmetic stripped) on one "
            "representative layer; python3 tools/future/mlp_alu_roofline.py --measure --record"
        ),
        runtime=180.0,
        resource="GPU_PROTECTED",
        falsifier=(
            "ARM A stays within 1.12x of production -> memory-system bound; "
            "ARM A jumps >= 1.25x -> not a bandwidth ceiling of this kernel"
        ),
        children=(a1, a2, a3),
    )
    b1 = _h(
        "mlp.why_330.B1",
        "B1 unpack/decode",
        conf=0.38,
        gain=gain_ms,
        exp=(
            "python3 tools/future/mlp_alu_roofline.py --record; cite "
            "mlp.decode_tax.production_decode_fma_per_weight_byte (8 dequant FMA / "
            "6 weight bytes) against to_match_lm_head_497.target_decode_fma_per_weight_byte"
        ),
        runtime=10.0,
        resource="ANALYSIS",
        falsifier=(
            "if stripping decode+dequant (keeping the same loads) does not move "
            "GB/s, unpack/decode is not the lever; if it lands on the LM-head "
            "rate, decode FMA/byte is the quantified lever"
        ),
    )
    b2 = _h(
        "mlp.why_330.B2",
        "B2 conversion",
        conf=0.28,
        gain=gain_ms,
        exp=(
            "ARM A' keep production K and dequant-FMA, replace int-to-float with "
            "a bitcast sink, same weight_bytes; extend alu_roofline_organs.rs "
            "--arm-a conversion-stripped then python3 tools/future/mlp_alu_roofline.py --measure --record"
        ),
        runtime=180.0,
        resource="GPU_PROTECTED",
        falsifier=(
            "conversion-only strip that does not move time falsifies B2 as the "
            "dominant arithmetic; a jump comparable to full ARM A isolates it"
        ),
    )
    b3 = _h(
        "mlp.why_330.B3",
        "B3 accumulation",
        conf=0.28,
        gain=gain_ms,
        exp=(
            "ARM B': keep production weight_bytes and cut MAC-FMA issue in half "
            "without cutting K (skip every other MAC, retain dequant). Command: "
            "python3 tools/future/mlp_alu_roofline.py --measure --record after "
            "alu_roofline_organs.rs --arm-b ops-thinned --keep-k"
        ),
        runtime=180.0,
        resource="GPU_PROTECTED",
        falsifier=(
            "time tracks remaining FMA count at constant bytes -> accumulation/"
            "arithmetic ceiling; time stays at production -> accumulation is not it. "
            "The landed ARM B (half K) CANNOT close this: it halves FMAs with bytes"
        ),
    )
    b = _h(
        "mlp.why_330.B",
        "B arithmetic ceiling",
        conf=0.40,
        gain=gain_ms,
        exp=(
            "pre-registered matched pair: ARM A jump AND ARM B sub-linear. "
            "python3 tools/future/mlp_alu_roofline.py --measure --record. "
            "ALU_BOUND only if both conjuncts fire and loads survive"
        ),
        runtime=180.0,
        resource="GPU_PROTECTED",
        falsifier=(
            "ALU_BOUND iff ARM A >= 1.25x AND ARM B time/byte > 1.20 AND loads "
            "survived. ARM A jump with ARM B linear (half K also halves FMAs) "
            "is MIXED, not ALU_BOUND — the conjunction is the falsifier"
        ),
        children=(b1, b2, b3),
    )
    c1 = _h(
        "mlp.why_330.C1",
        "C1 instruction chain",
        conf=0.22,
        gain=gain_ms,
        exp=(
            "disassemble qwen_affine_q2_group32_matvec_geo_tpr64_tg128 inner loop "
            "and report dependent dequant-then-MAC chain length vs dual-issue "
            "slots; xcrun metal / air-lld if present, else PARK on compiler"
        ),
        runtime=60.0,
        resource="GPU_PROTECTED",
        falsifier=(
            "an inner-loop chain that dual-issues dequant with MAC at the measured "
            "issue rate falsifies a dependency ceiling; a serial dequant-MAC chain "
            "at 1.33 FMA/byte that matches the 1.51x ARM A jump supports it"
        ),
    )
    c2 = _h(
        "mlp.why_330.C2",
        "C2 register pressure",
        conf=0.22,
        gain=gain_ms,
        exp=(
            "python3 tools/future/mlp_alu_roofline.py --record; cite "
            "mlp.production.occupancy.registers_per_thread on "
            "qwen_affine_q2_group32_matvec_geo_tpr64_tg128 (currently null: "
            "toolchain does not expose it)"
        ),
        runtime=30.0,
        resource="GPU_PROTECTED",
        falsifier=(
            "registers_per_thread reported and occupancy_of_max_threads rising "
            "when arithmetic is stripped would support spill/pressure; a null "
            "register count cannot confirm C2 and cannot kill it"
        ),
    )
    c = _h(
        "mlp.why_330.C",
        "C dependency ceiling",
        conf=0.25,
        gain=gain_ms,
        exp=(
            "C1 chain length plus C2 registers_per_thread on the production "
            "geo_tpr64 body; both from the same pipeline-state dump as occupancy"
        ),
        runtime=60.0,
        resource="GPU_PROTECTED",
        falsifier=(
            "a reported register count AND an inner-loop DAG that still dual-issues "
            "at production issue rate falsifies a dependency ceiling independent "
            "of B1 decode tax"
        ),
        children=(c1, c2),
    )
    d = _h(
        "mlp.why_330.D",
        "D physical execution topology",
        conf=0.18,
        gain=gain_ms,
        exp=(
            "counterfactual dispatch topology: one MLP layer as 1 fused region "
            "vs production 3 dispatches already ran (MLP_REGION_FALSIFIER.json "
            "331.6 -> 332.2 GB/s). Remaining D probe: tile/SIMD-group mapping "
            "of geo_tpr64 vs LM-head Q4 on the same 83.56 MB working set, "
            "python3 tools/future/mlp_alu_roofline.py --measure --record"
        ),
        runtime=180.0,
        resource="GPU_PROTECTED",
        falsifier=(
            "region granularity is already GRANULARITY_REFUTED at 332.2 vs 331.6; "
            "a topology claim that does not beat that delta is the same scar. "
            "Reopen only for a different physical mapping (not buffer contiguity)"
        ),
    )
    return _h(
        "mlp.why_330",
        "WHY IS THE MLP AT ~330 GB/s?",
        conf=0.50,
        gain=gain_ms,
        exp=(
            "matched pair ARM A + ARM B on one representative layer of sealed-3.14 "
            "production affine2 geo_tpr64; python3 tools/future/mlp_alu_roofline.py "
            "--measure --record. Verdict is the pre-registered conjunction, not ARM A alone"
        ),
        runtime=180.0,
        resource="GPU_PROTECTED",
        falsifier=(
            "pre-registered: A jumps AND B sub-linear AND loads survived -> ALU_BOUND; "
            "A stays -> MEMORY_SYSTEM_BOUND; anything else (including A jumps, B linear "
            "because half K halves FMAs) -> MIXED. Do not force a binary"
        ),
        children=(a, b, c, d),
    )


def _fn_replacement_tree(*, gain_ms: float, families: Sequence[Mapping[str, Any]]) -> Hypothesis:
    kids: list[Hypothesis] = []
    for fam in families:
        name = str(fam["family"])
        l2 = fam.get("held_out_relative_l2_best")
        reopen = str(fam.get("reopen") or "")
        kids.append(
            _h(
                f"mlp.fn.r_bottleneck.{name}",
                f"r-bottleneck family {name}",
                conf=0.05,
                gain=gain_ms,
                exp=(
                    f"held-out relative L2 of {name} on the sealed-3.14 teacher "
                    "corpus, prompt_id split, billed through executable_economics.score; "
                    "python3 tools/future/mlp_nonlinear_program.py --build"
                ),
                runtime=120.0,
                resource="CPU",
                falsifier=(
                    "held-out relative L2 in the 0.9 band at any affordable size is "
                    f"the cheap kill (landed {l2})"
                ),
                status=KILLED,
                reopen_if=reopen,
                notes=str(fam.get("mechanism") or ""),
            )
        )
    live = _h(
        "mlp.fn.full_width_structured",
        "full-width structured operator (Monarch / butterfly / distilled, not an r-bottleneck)",
        conf=0.35,
        gain=gain_ms,
        exp=(
            "cheap round: one representative layer (38), Monarch or butterfly "
            "factorisation of F that is not rank-r, held-out relative L2 on the "
            "teacher corpus prompt_id split, bytes via executable_economics.score; "
            "python3 tools/future/mlp_nonlinear_program.py --build"
        ),
        runtime=600.0,
        resource="CPU",
        falsifier=(
            "held-out relative L2 in the 0.9 band at affordable size kills this "
            "instantiation (do not widen rank). Function replacement as a class "
            "stays open only if a different full-width algebra is proposed"
        ),
    )
    return _h(
        "mlp.fn.root",
        "MLP function replacement: stop storing q, represent F",
        conf=0.40,
        gain=gain_ms,
        exp=(
            "score function_replacement through executable_economics.score with "
            "bytes_added billed, then a held-out F replacement that is not an "
            "r-bottleneck; python3 tools/future/mlp_nonlinear_program.py --build"
        ),
        runtime=600.0,
        resource="CPU",
        falsifier=(
            "a candidate whose held-out relative L2 stays in the 0.9 band is "
            "MEASURED_NEGATIVE; dense rematerialization of W is REJECTED_DENSE_REMAT"
        ),
        children=tuple(kids) + (live,),
    )


def campaign(*, apply_landed: bool = True) -> Metabolism:
    """Populate the tree from the live campaign receipts. Not a fixture."""
    origins: dict[str, str] = {}
    docs: dict[str, dict[str, Any]] = {}
    for rel in REQUIRED_SOURCES:
        doc, origin = load_receipt_origin(rel)
        if doc is None:
            raise MissingReceipt(f"required {rel} unseen ({origin})")
        docs[rel] = doc
        origins[rel] = origin

    budget = docs["receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json"]
    alu = docs["receipts/future/MLP_ALU_ROOFLINE.json"]
    rank = docs["receipts/future/MLP_FUNCTIONAL_RANK.json"]
    nonlinear = docs["receipts/future/MLP_NONLINEAR_PROGRAM.json"]
    shared = docs["receipts/future/MLP_SHARED_PROGRAM.json"]
    dn_gen = docs["receipts/future/DELTANET_GENERATED_TRANSITION.json"]
    dn_ms = docs["receipts/future/DELTANET_MULTISTEP.json"]
    path = docs["receipts/future/PATH_TO_71.json"]
    econ = docs["receipts/future/EXECUTABLE_ECONOMICS.json"]
    gate = docs["receipts/future/ODYSSEY_LAUNCH_GATE.json"]

    organs = {o["organ"]: o for o in budget["measured_now"]["organs"]}
    mlp_organ = organs["mlp"]
    dn_organ = organs["deltanet"]
    ranked = {e["id"]: e for e in budget["experiments_ranked_by_gain"]}
    mlp_gain = float(ranked["reach_demonstrated_bandwidth_mlp"]["ms_saved"])
    dn_gain_demonstrated = float(ranked["reach_demonstrated_bandwidth_deltanet"]["ms_saved"])

    mlp_j = alu["mlp"]["judgement"]
    mlp_prod = alu["mlp"]["production"]
    mlp_decode = alu["mlp"]["decode_tax"]
    weight_bytes = int(mlp_prod["weight_bytes"])
    weight_mb = round(weight_bytes / 1_000_000.0, 2)

    dn_iso_gb_s = float(alu["deltanet"]["production"]["effective_gb_s"])
    dn_organ_gb = float(dn_organ["gb"])
    dn_organ_ms = float(dn_organ["ms"])
    dn_organ_gb_s = float(dn_organ["gb_s"])
    dn_counterfactual_ms = round(dn_organ_gb / dn_iso_gb_s * 1000.0, 3)
    dn_unexplained_ms = round(dn_organ_ms - dn_counterfactual_ms, 3)

    fn_gain = 0.0
    for cand in econ.get("candidates_ranked") or []:
        if cand.get("id") == "function_replacement":
            fn_gain = float(cand["predicted_ms_saved"])
            break
    if fn_gain <= 0.0:
        top = {t["id"]: t for t in econ.get("top_live_material") or []}
        if "function_replacement" in top:
            fn_gain = float(top["function_replacement"]["predicted_ms_saved"])

    aux_gain = {
        e["id"]: float(e["ms_saved"])
        for e in budget["experiments_ranked_by_gain"]
        if e["id"] in {"group_size_1024", "group_size_256", "quantize_aux_u8"}
    }

    cited = {
        "mlp_alu": {
            "source": "receipts/future/MLP_ALU_ROOFLINE.json",
            "production_gb_s": mlp_j["production_gb_s"],
            "arm_a_gb_s": mlp_j["arm_a_gb_s"],
            "arm_a_over_production": mlp_j["arm_a_over_production"],
            "weight_bytes": weight_bytes,
            "weight_mb": weight_mb,
            "arm_b_time_over_byte": mlp_j["arm_b_time_over_byte"],
            "arm_b_linear": mlp_j["arm_b_linear"],
            "arm_b_sublinear": mlp_j["arm_b_sublinear"],
            "loads_survived": (mlp_j.get("loads_survived") or {}).get("survived"),
            "occupancy_limited": mlp_j["occupancy_limited"],
            "threads_per_threadgroup": alu["mlp"]["threads_per_threadgroup"],
            "max_total_threads_per_threadgroup": (
                mlp_prod.get("occupancy") or {}
            ).get("max_total_threads_per_threadgroup"),
            "threadgroups_per_core": (mlp_prod.get("occupancy") or {}).get(
                "threadgroups_per_core"
            ),
            "production_decode_fma_per_weight_byte": mlp_decode.get(
                "production_decode_fma_per_weight_byte"
            ),
            "target_decode_fma_per_weight_byte_at_497": (
                mlp_decode.get("to_match_lm_head_497") or {}
            ).get("target_decode_fma_per_weight_byte"),
            "required_decode_cheapening": (
                mlp_decode.get("to_match_lm_head_497") or {}
            ).get("required_decode_cheapening"),
            "verdict": alu.get("verdict"),
            "mlp_verdict": alu["mlp"].get("verdict"),
            "why_not_forced": mlp_j.get("why_not_forced"),
            "lm_head_gb_s": alu.get("lm_head_gb_s"),
        },
        "causal_budget": {
            "source": "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json",
            "cited_token_ms": budget["measured_now"]["token_ms"],
            "cited_tps": budget["measured_now"]["tps"],
            "mlp_organ_ms": mlp_organ["ms"],
            "mlp_organ_gb_s": mlp_organ["gb_s"],
            "mlp_organ_gb": mlp_organ["gb"],
            "deltanet_organ_ms": dn_organ["ms"],
            "deltanet_organ_gb_s": dn_organ["gb_s"],
            "deltanet_organ_gb": dn_organ["gb"],
            "deltanet_dispatches": dn_organ["dispatches"],
            "mlp_ms_saved_at_497": mlp_gain,
            "deltanet_ms_saved_at_497": dn_gain_demonstrated,
            "demonstrated_regime_cited_tps": budget["the_two_numbers_that_matter"][
                "demonstrated_regime_tps"
            ],
            "roof_on_todays_bytes_cited_tps": budget["the_two_numbers_that_matter"][
                "roof_on_todays_bytes_tps"
            ],
        },
        "deltanet_isolation": {
            "source": [
                "receipts/future/MLP_ALU_ROOFLINE.json",
                "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json",
            ],
            "organ_gb_s": dn_organ_gb_s,
            "organ_ms": dn_organ_ms,
            "organ_gb": dn_organ_gb,
            "isolated_kernel_gb_s": dn_iso_gb_s,
            "isolated_kernel": alu["deltanet"].get("kernel"),
            "counterfactual_organ_ms_at_isolated_rate": dn_counterfactual_ms,
            "unexplained_ms": dn_unexplained_ms,
            "arithmetic": (
                f"{dn_organ_ms} ms at {dn_organ_gb_s} GB/s over {dn_organ_gb} GB; "
                f"isolated in_proj_qkvz {dn_iso_gb_s} GB/s would be "
                f"{dn_counterfactual_ms} ms; unexplained {dn_unexplained_ms} ms "
                "(campaign language: up to 4 ms)"
            ),
        },
        "path_to_71": {
            "source": "receipts/future/PATH_TO_71.json",
            "cited_token_ms": path["baseline"]["token_ms"],
            "cited_tps": path["baseline"]["tps"],
            "best_composed_path": path["gap_to_71"]["best_composed_path"],
            "best_composed_token_ms": path["gap_to_71"]["best_composed_token_ms"],
            "best_composed_cited_tps": path["gap_to_71"]["best_composed_tps"],
            "still_to_remove_ms": path["gap_to_71"]["still_to_remove_ms"],
            "target_token_ms": path["gap_to_71"]["target_token_ms"],
        },
        "function_replacement": {
            "source": [
                "receipts/future/MLP_NONLINEAR_PROGRAM.json",
                "receipts/future/MLP_SHARED_PROGRAM.json",
                "receipts/future/MLP_FUNCTIONAL_RANK.json",
                "receipts/future/EXECUTABLE_ECONOMICS.json",
            ],
            "n_nonlinear_families": len(nonlinear.get("families") or []),
            "families": list(nonlinear.get("families") or []),
            "n_survivors_nonlinear": nonlinear.get("n_survivors"),
            "n_survivors_shared": shared.get("n_survivors"),
            "shared_candidates_measured_negative": (
                shared.get("candidate_counts") or {}
            ).get("measured_negative"),
            "oracle_pca_r64_held_out_relative_l2": (
                (shared.get("oracle_output_pca") or [{}])[-1] or {}
            ).get("held_out_relative_l2"),
            "affordable_f16_rank_cap": rank.get("affordable_f16_rank_cap"),
            "uses_essentially_all_of_W": (
                (rank.get("answers") or {}).get("does_the_model_use_essentially_all_of_W") or {}
            ).get("status"),
            "predicted_ms_saved_if_F_holds": fn_gain,
            "next": nonlinear.get("next"),
        },
        "odyssey_gate": {
            "source": "receipts/future/ODYSSEY_LAUNCH_GATE.json",
            "n_criteria": (gate.get("verdict") or {}).get("n_criteria"),
            "n_met": (gate.get("verdict") or {}).get("n_met"),
            "n_unmet": (gate.get("verdict") or {}).get("n_unmet"),
            "met": (gate.get("verdict") or {}).get("met"),
            "unmet": (gate.get("verdict") or {}).get("unmet"),
            "verdict": (gate.get("verdict") or {}).get("verdict"),
            "phase_transition": gate.get("phase_transition"),
        },
        "deltanet_generated": {
            "source": "receipts/future/DELTANET_GENERATED_TRANSITION.json",
            "candidate_id": dn_gen.get("candidate_id"),
            "fit": dn_gen.get("fit"),
            "verdict": dn_gen.get("verdict"),
            "net_bytes": (dn_gen.get("bytes") or {}).get("net_bytes"),
            "predicted_ms_saved": (dn_gen.get("economics") or {}).get("predicted_ms_saved"),
        },
        "deltanet_multistep": {
            "source": "receipts/future/DELTANET_MULTISTEP.json",
            "generated_transition_landed": (
                (dn_ms.get("landed_candidates") or {}).get("landed")
            ),
            "one_step_only_admissible": (
                (dn_ms.get("answers") or {}).get("is_one_step_admissible") or {}
            ).get("one_step_only_admissible"),
            "argmax_is_not_parity": (
                (dn_ms.get("answers") or {}).get("is_argmax_parity") or {}
            ).get("argmax_is_not_parity"),
        },
    }

    # --- mlp execution frontier ---
    mlp_tree = _mlp_why_330_tree(gain_ms=mlp_gain, weight_bytes=weight_bytes)
    mlp_scars = []
    for lever in budget.get("refuted_levers") or []:
        mlp_scars.append(
            {
                "id": lever.get("id"),
                "verdict": lever.get("verdict"),
                "source": lever.get("source"),
                "scope": "mlp_executor",
                "reopen_if": (
                    "a different mechanism than the one this receipt killed; "
                    "retrying the same lever is the scar"
                ),
                "evidence": lever.get("evidence"),
            }
        )
    for name in alu.get("refuted_elsewhere") or []:
        if name not in {s["id"] for s in mlp_scars}:
            mlp_scars.append(
                {
                    "id": name,
                    "verdict": "REFUTED_ELSEWHERE",
                    "source": "receipts/future/MLP_ALU_ROOFLINE.json",
                    "scope": "mlp_executor",
                    "reopen_if": "a different physical mapping than the named scar",
                }
            )

    mlp_fr = frontier(
        id="mlp_execution",
        objective=(
            "Explain and close the MLP's ~330 GB/s. The organ is "
            f"{mlp_organ['ms']} ms at {mlp_organ['gb_s']} GB/s; one representative "
            f"layer measures production {mlp_j['production_gb_s']} GB/s on "
            f"{weight_mb} MB. The target is the LM head's demonstrated "
            f"{alu.get('lm_head_gb_s')} GB/s, worth {mlp_gain} ms if the organ "
            "reaches it. This frontier exists because that gap is the largest "
            "executor-shaped term in the causal budget."
        ),
        current_best={
            "cited_layer_production_gb_s": mlp_j["production_gb_s"],
            "cited_organ_gb_s": mlp_organ["gb_s"],
            "cited_organ_ms": mlp_organ["ms"],
            "source": [
                "receipts/future/MLP_ALU_ROOFLINE.json",
                "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json",
            ],
        },
        target={
            "cited_gb_s": alu.get("lm_head_gb_s"),
            "cited_ms_at_demonstrated": ranked["reach_demonstrated_bandwidth_mlp"][
                "target_ms_at_demonstrated"
            ],
            "source": "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json",
        },
        remaining_gap={
            "cited_ms": mlp_gain,
            "source": "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json",
            "id": "reach_demonstrated_bandwidth_mlp",
        },
        causal_hypotheses=mlp_tree,
        scars=mlp_scars,
        biggest_unknown=(
            "Whether cheaper decode (B1) can be isolated from conversion (B2) and "
            "accumulation (B3) without an ARM B that also cuts bytes — the landed "
            "conjunction is MIXED until a keep-K FMA-thinned arm exists. "
            "registers_per_thread is null so C2 is unmeasured."
        ),
        resource_requirements=["GPU_PROTECTED", "ANALYSIS"],
    )

    # --- DeltaNet execution ---
    dn_other = _h(
        "dn.unexplained.other_kernels",
        "kernels other than isolated in_proj_qkvz consume the organ milliseconds",
        conf=0.55,
        gain=dn_unexplained_ms,
        exp=(
            "census GPU ns of every DeltaNet dispatch class in the 337-dispatch "
            f"organ (cited {dn_organ['dispatches']} dispatches, {dn_organ_ms} ms, "
            f"{dn_organ_gb_s} GB/s) against isolated "
            f"{alu['deltanet'].get('kernel')} at {dn_iso_gb_s} GB/s; "
            "python3 tools/future/mlp_alu_roofline.py --record plus the organ "
            "workgraph in receipts/future/FLASH_ORGAN_WORKGRAPHS.json"
        ),
        runtime=120.0,
        resource="ANALYSIS",
        falsifier=(
            f"if the sum of isolated kernel times at their measured GB/s equals "
            f"{dn_organ_ms} ms, there is no unexplained organ cost; a residual "
            f"near {dn_unexplained_ms} ms means the largest named kernel is not "
            "the organ's cost"
        ),
    )
    dn_fusion = _h(
        "dn.unexplained.fusion_or_ba",
        "BA/delta / fusion ceremony, not the Q4 matvec, is the organ cost",
        conf=0.30,
        gain=dn_unexplained_ms,
        exp=(
            "compare organ 8.227 ms to isolated qkvz counterfactual "
            f"{dn_counterfactual_ms} ms; attribute the {dn_unexplained_ms} ms "
            "residual to BA/delta (receipts/future/BA_DELTA_AB.json, 0.1384 ms "
            "qualified) vs unnamed kernels. python3 tools/future/ba_delta_ab.py --build"
        ),
        runtime=60.0,
        resource="ANALYSIS",
        falsifier=(
            "BA_DELTA_AB qualified 0.1384 ms; that cannot cover a ~3 ms residual. "
            "A fusion/ceremony claim must name a dispatch class whose GPU ns sum "
            f"to the {dn_unexplained_ms} ms gap"
        ),
    )
    dn_tree = _h(
        "dn.unexplained",
        "DeltaNet organ 360 GB/s vs isolated largest kernel 600.9 GB/s",
        conf=0.50,
        gain=dn_unexplained_ms,
        exp=(
            "subtract isolated in_proj_qkvz time at 600.9 GB/s from the cited "
            "organ 8.227 ms / 2.961659904 GB / 360.0 GB/s; name the kernels that "
            f"make the {dn_unexplained_ms} ms residual"
        ),
        runtime=60.0,
        resource="ANALYSIS",
        falsifier=(
            "an organ-level accounting that sums to 8.227 ms with named kernels "
            "closes this; until then the largest kernel is not the cost"
        ),
        children=(dn_other, dn_fusion),
    )
    dn_fr = frontier(
        id="deltanet_execution",
        objective=(
            "The DeltaNet organ is "
            f"{dn_organ_gb_s} GB/s over {dn_organ_ms} ms. Its largest kernel "
            f"({alu['deltanet'].get('kernel')}) measures {dn_iso_gb_s} GB/s in "
            f"isolation. Arithmetic: {cited['deltanet_isolation']['arithmetic']}. "
            "This frontier exists because that residual is now the largest "
            "unexplained cost in the token."
        ),
        current_best={
            "cited_organ_gb_s": dn_organ_gb_s,
            "cited_organ_ms": dn_organ_ms,
            "cited_isolated_kernel_gb_s": dn_iso_gb_s,
            "source": cited["deltanet_isolation"]["source"],
        },
        target={
            "label": "organ time explained by named kernels",
            "cited_unexplained_ms": 0.0,
        },
        remaining_gap={
            "cited_unexplained_ms": dn_unexplained_ms,
            "cited_counterfactual_ms": dn_counterfactual_ms,
            "formula": "organ_ms - organ_gb / isolated_kernel_gb_s * 1000",
        },
        causal_hypotheses=dn_tree,
        scars=[
            {
                "id": "deltanet_q3",
                "verdict": "REFUTED",
                "source": "receipts/future/PATH_TO_71.json",
                "scope": "deltanet q/k/v/z bit-descent",
                "reopen_if": (
                    "a sensitivity measurement that licenses lower bits; "
                    "entropies 3.465-3.479 of 4 and any_supported=false is this scar"
                ),
            }
        ],
        live_candidates=[
            {
                "id": dn_gen.get("candidate_id"),
                "fit": dn_gen.get("fit"),
                "verdict": dn_gen.get("verdict"),
                "status": "NOT_LANDED",
                "source": "receipts/future/DELTANET_GENERATED_TRANSITION.json",
            }
        ],
        biggest_unknown=(
            f"Which of the {dn_organ['dispatches']} dispatches consume the "
            f"{dn_unexplained_ms} ms that isolated in_proj_qkvz does not."
        ),
        next_decisive_experiments=[dn_other.cheapest_decisive_experiment],
        resource_requirements=["ANALYSIS", "GPU_PROTECTED"],
    )

    # --- auxiliary byte levers ---
    aux_kids = []
    aux_specs = [
        (
            "group_size_1024",
            aux_gain.get("group_size_1024", 2.914),
            1.0027008,
            "refit HGRAVF01 at group_size=1024 and run a held-out reconstruction "
            "plus organ error on a real layer; python3 tools/future/mlp_auxiliary_information.py --build",
        ),
        (
            "group_size_256",
            aux_gain.get("group_size_256", 2.331),
            0.80216064,
            "refit HGRAVF01 at group_size=256; same capability screen as group_size_1024; "
            "python3 tools/future/mlp_auxiliary_information.py --build",
        ),
        (
            "quantize_aux_u8",
            aux_gain.get("quantize_aux_u8", 1.554),
            0.53477376,
            "u8 of per-group scale/bias (byte model exact) plus held-out organ error; "
            "python3 tools/future/mlp_auxiliary_information.py --build",
        ),
    ]
    for aid, ag, gb, exp in aux_specs:
        aux_kids.append(
            _h(
                f"mlp.aux.{aid}",
                aid,
                conf=0.45,
                gain=ag,
                exp=exp,
                runtime=300.0,
                resource="CPU",
                falsifier=(
                    "held-out reconstruction plus organ error on a real layer; "
                    "capability UNMEASURED until that screen runs. Overlapping "
                    "levers are not additive (PATH_TO_71 skipped aux_u8 against "
                    "group_size_1024)"
                ),
            )
        )
    aux_tree = _h(
        "mlp.aux.root",
        "auxiliary byte levers on the 1.07 GB scale/bias/header",
        conf=0.45,
        gain=max(aux_gain.values() or [0.0]),
        exp=aux_specs[0][3],
        runtime=300.0,
        resource="CPU",
        falsifier=(
            "a generate-gate failure at the claimed group size / u8 aux kills "
            "the lever; a byte model without a capability screen stays PROSPECTIVE"
        ),
        children=tuple(aux_kids),
    )
    aux_fr = frontier(
        id="aux_byte_levers",
        objective=(
            "The MLP auxiliary arrays are 1,069,605,696 bytes "
            "(receipts/future/MLP_AUXILIARY_INFORMATION.json). Open exact-byte "
            "levers: group_size_1024 "
            f"({aux_gain.get('group_size_1024')} ms, 1.0027008 GB), "
            f"group_size_256 ({aux_gain.get('group_size_256')} ms), "
            f"quantize_aux_u8 ({aux_gain.get('quantize_aux_u8')} ms). "
            "Capability is UNMEASURED. This frontier exists because these are "
            "the only remaining exact-byte moves on today's weights."
        ),
        current_best={
            "group_size": 64,
            "auxiliary_bytes": 1069605696,
            "source": "receipts/future/MLP_AUXILIARY_INFORMATION.json",
        },
        target={
            "id": "group_size_1024",
            "gb_saved": 1.0027008,
            "cited_ms_saved": aux_gain.get("group_size_1024"),
            "capability": "UNMEASURED",
        },
        remaining_gap={
            "cited_ms": aux_gain.get("group_size_1024"),
            "capability": "UNMEASURED",
            "overlaps": "aux_u8 overlaps group_size; PATH_TO_71 does not sum them",
        },
        causal_hypotheses=aux_tree,
        scars=[
            {
                "id": "drop_biases",
                "verdict": "MEASURED_NEGATIVE",
                "scope": "HGRAVF01 bias arrays of sealed-3.14",
                "reopen_if": "a packing that does not use the bias channel",
                "source": "receipts/future/MLP_AUXILIARY_INFORMATION.json",
            },
            {
                "id": "generate_scales_from_group_index",
                "verdict": "MEASURED_NEGATIVE",
                "scope": "HGRAVF01 scale arrays of sealed-3.14",
                "reopen_if": "a new source that is not this residual of S",
                "source": "receipts/future/MLP_AUXILIARY_INFORMATION.json",
            },
        ],
        biggest_unknown="capability of the exact-byte levers (no generate gate has been run)",
        resource_requirements=["CPU", "GPU_PROTECTED"],
    )

    # --- function replacement ---
    fam_scars = list(nonlinear.get("scars") or [])
    fn_tree = _fn_replacement_tree(gain_ms=fn_gain, families=fam_scars)
    fn_fr = frontier(
        id="mlp_function_replacement",
        objective=(
            "PATH_TO_71 best composed path reaches 42.36 cited TPS; 71 needs "
            f"another {path['gap_to_71']['still_to_remove_ms']} ms. Both organs "
            "are at their entropy floor, so the only remaining source of that "
            "magnitude is weights that need not exist independently. Six "
            "r-bottleneck families are MEASURED_NEGATIVE; the live reopen is a "
            "full-width structured operator. This frontier exists because "
            "function_replacement is the only economics candidate that is live "
            f"and material at {fn_gain} ms if F holds."
        ),
        current_best={
            "n_survivors_nonlinear": nonlinear.get("n_survivors"),
            "n_survivors_shared": shared.get("n_survivors"),
            "oracle_pca_r64_held_out_relative_l2": cited["function_replacement"][
                "oracle_pca_r64_held_out_relative_l2"
            ],
            "affordable_f16_rank_cap": rank.get("affordable_f16_rank_cap"),
        },
        target={
            "label": "replace F, do not code q better",
            "predicted_ms_saved_if_F_holds": fn_gain,
            "source": "receipts/future/EXECUTABLE_ECONOMICS.json",
        },
        remaining_gap={
            "cited_ms": path["gap_to_71"]["still_to_remove_ms"],
            "best_composed_cited_tps": path["gap_to_71"]["best_composed_tps"],
            "source": "receipts/future/PATH_TO_71.json",
        },
        causal_hypotheses=fn_tree,
        scars=[
            {
                "id": s.get("family"),
                "verdict": s.get("status"),
                "scope": s.get("object") or "sealed-3.14 MLP F",
                "reopen_if": s.get("reopen"),
                "held_out_relative_l2_best": s.get("held_out_relative_l2_best"),
                "mechanism": s.get("mechanism"),
                "source": "receipts/future/MLP_NONLINEAR_PROGRAM.json",
            }
            for s in fam_scars
        ],
        live_candidates=[
            {
                "id": "full_width_structured_nonlinear",
                "status": "OPEN",
                "not": list(nonlinear.get("families") or []),
            }
        ],
        biggest_unknown=(
            "a full-width structured nonlinear (Monarch / butterfly / distilled "
            "operator) that is not an r-dimensional bottleneck"
        ),
        resource_requirements=["CPU", "REPRESENTATION"],
    )

    # --- Odyssey gate ---
    v = gate.get("verdict") or {}
    unmet = list(v.get("unmet") or [])
    gate_kids = []
    for cid in unmet:
        gate_kids.append(
            _h(
                f"odyssey.unmet.{cid}",
                cid,
                conf=0.50,
                gain=0.0,
                exp=(
                    f"re-evaluate criterion {cid} from disk evidence; "
                    "python3 tools/future/odyssey_launch.py --verify"
                ),
                runtime=30.0,
                resource="ODYSSEY",
                falsifier=(
                    f"{cid} is MET only when the resident can discover, invoke, "
                    "schedule and verify it and the result persists; a CLI a "
                    "human can run is not enough"
                ),
            )
        )
    n_met = v.get("n_met")
    n_crit = v.get("n_criteria")
    gate_tree = _h(
        "odyssey.gate",
        f"Odyssey I launch gate {n_met}/{n_crit}",
        conf=0.80,
        gain=0.0,
        exp="python3 tools/future/odyssey_launch.py --verify",
        runtime=30.0,
        resource="ODYSSEY",
        falsifier=(
            "ODYSSEY_I_LAUNCH.json is written only when every criterion is met; "
            f"the sealed receipt is {v.get('verdict')} with unmet={unmet}"
        ),
        children=tuple(gate_kids),
    )
    odyssey_fr = frontier(
        id="odyssey_gate",
        objective=(
            f"Odyssey I (WHAT IS TRUE?) is {v.get('verdict')} at {n_met}/{n_crit} "
            f"cited from receipts/future/ODYSSEY_LAUNCH_GATE.json. Unmet: {unmet}. "
            "This frontier exists because a green scaffold is not Odyssey started."
        ),
        current_best={
            "n_met": n_met,
            "n_criteria": n_crit,
            "verdict": v.get("verdict"),
            "source": "receipts/future/ODYSSEY_LAUNCH_GATE.json",
        },
        target={"n_met": n_crit, "n_criteria": n_crit, "verdict": "LAUNCH"},
        remaining_gap={"unmet": unmet, "n_unmet": v.get("n_unmet")},
        causal_hypotheses=gate_tree,
        scars=[],
        biggest_unknown=(
            "the unmet criteria: " + ", ".join(unmet) if unmet else "none unmet"
        ),
        resource_requirements=["ODYSSEY", "TOOLING"],
    )

    metab = Metabolism(
        frontiers={
            "mlp_execution": mlp_fr,
            "deltanet_execution": dn_fr,
            "aux_byte_levers": aux_fr,
            "mlp_function_replacement": fn_fr,
            "odyssey_gate": odyssey_fr,
        },
        cited=cited,
        sources=origins,
    )

    # Scientific work queue: all seven roles, including a falsifier.
    metab.work_units = [
        work_unit(
            id="wu.mlp.arm_b_keep_k",
            role=FALSIFIER,
            hypothesis_id="mlp.why_330.B",
            frontier_id="mlp_execution",
            experiment=(
                "ARM B' keep-K FMA-thinned: production weight_bytes="
                f"{weight_bytes}, skip every other MAC, judge sub-linear time "
                "vs production; python3 tools/future/mlp_alu_roofline.py --measure --record"
            ),
            required_resource="GPU_PROTECTED",
            expected_runtime_s=180.0,
            terminal=terminal(
                PARK,
                wake_condition=(
                    "protected GPU lease held AND alu_roofline_organs example "
                    "binary present AND --arm-b ops-thinned --keep-k implemented"
                ),
                hypothesis_id="mlp.why_330.B",
            ),
        ),
        work_unit(
            id="wu.mlp.b2_conversion",
            role=PROBE,
            hypothesis_id="mlp.why_330.B2",
            frontier_id="mlp_execution",
            experiment=(
                "ARM A conversion-stripped (int-to-float bitcast sink, dequant-FMA "
                f"kept, weight_bytes={weight_bytes}); "
                "python3 tools/future/mlp_alu_roofline.py --measure --record"
            ),
            required_resource="GPU_PROTECTED",
            expected_runtime_s=180.0,
        ),
        work_unit(
            id="wu.mlp.c2_registers",
            role=PROBE,
            hypothesis_id="mlp.why_330.C2",
            frontier_id="mlp_execution",
            experiment=(
                "python3 tools/future/mlp_alu_roofline.py --record; cite "
                "mlp.production.occupancy.registers_per_thread on "
                "qwen_affine_q2_group32_matvec_geo_tpr64_tg128 (currently null)"
            ),
            required_resource="GPU_PROTECTED",
            expected_runtime_s=30.0,
        ),
        work_unit(
            id="wu.dn.kernel_census",
            role=PROBE,
            hypothesis_id="dn.unexplained.other_kernels",
            frontier_id="deltanet_execution",
            experiment=dn_other.cheapest_decisive_experiment,
            required_resource="ANALYSIS",
            expected_runtime_s=120.0,
        ),
        work_unit(
            id="wu.fn.monarch",
            role=MUTATION,
            hypothesis_id="mlp.fn.full_width_structured",
            frontier_id="mlp_function_replacement",
            experiment=(
                "Monarch / butterfly factorisation of F on teacher-corpus layer 38, "
                "held-out relative L2, bytes via executable_economics.score; "
                "python3 tools/future/mlp_nonlinear_program.py --build"
            ),
            required_resource="CPU",
            expected_runtime_s=600.0,
        ),
        work_unit(
            id="wu.fn.oracle_pca",
            role=ORACLE,
            hypothesis_id="mlp.fn.root",
            frontier_id="mlp_function_replacement",
            experiment=(
                "oracle PCA of F at rank 64 on the teacher corpus "
                f"(landed held-out relative L2 "
                f"{cited['function_replacement']['oracle_pca_r64_held_out_relative_l2']}); "
                "python3 tools/future/mlp_shared_program.py --build"
            ),
            required_resource="CPU",
            expected_runtime_s=120.0,
        ),
        work_unit(
            id="wu.mlp.alu_replication",
            role=REPLICATION,
            hypothesis_id="mlp.why_330",
            frontier_id="mlp_execution",
            experiment=(
                "repeat the matched pair on a second representative layer of "
                "sealed-3.14 with the same ARM A/B protocol; "
                "python3 tools/future/mlp_alu_roofline.py --measure --record"
            ),
            required_resource="GPU_PROTECTED",
            expected_runtime_s=180.0,
        ),
        work_unit(
            id="wu.aux.g1024_capability",
            role=QUALIFICATION,
            hypothesis_id="mlp.aux.group_size_1024",
            frontier_id="aux_byte_levers",
            experiment=aux_specs[0][3],
            required_resource="CPU",
            expected_runtime_s=300.0,
        ),
        work_unit(
            id="wu.mlp.mixed_adversary",
            role=ADVERSARY,
            hypothesis_id="mlp.why_330.B",
            frontier_id="mlp_execution",
            experiment=(
                "attack the collapse MIXED -> ALU_BOUND: assert the pre-registered "
                "rule (ARM A jump AND ARM B sub-linear) against the landed "
                f"arm_b_sublinear={mlp_j['arm_b_sublinear']} "
                f"arm_b_linear={mlp_j['arm_b_linear']}; a promotion that ignores "
                "ARM B is the adversary win. python3 tools/future/improvement_metabolism.py --build"
            ),
            required_resource="ANALYSIS",
            expected_runtime_s=5.0,
        ),
        work_unit(
            id="wu.odyssey.unmet",
            role=QUALIFICATION,
            hypothesis_id="odyssey.gate",
            frontier_id="odyssey_gate",
            experiment="python3 tools/future/odyssey_launch.py --verify",
            required_resource="ODYSSEY",
            expected_runtime_s=30.0,
        ),
    ]

    for fr in metab.frontiers.values():
        _refresh_frontier(fr)

    if apply_landed:
        metab.ingest(alu)

    return metab


def build(*, path: Path | None = None) -> Path:
    """Seal IMPROVEMENT_METABOLISM.json from the live campaign."""
    metab = campaign(apply_landed=True)
    mlp_fr = metab.frontiers["mlp_execution"]
    alu_log = [r for r in metab.ingest_log if r.receipt_name == "MLP_ALU_ROOFLINE.json"]
    alu_ingest = alu_log[-1].to_dict() if alu_log else None
    why = mlp_fr.causal_hypotheses.to_dict()
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "tenet": TENET,
        "law": LAW,
        "measurement_class": "STATIC_ONLY",
        "produces_diagnostic_relative": False,
        "produces_protected_absolute": False,
        "sources": metab.sources,
        "cited": metab.cited,
        "frontiers": {k: v.to_dict() for k, v in metab.frontiers.items()},
        "mlp_330_tree": why,
        "alu_ingest": alu_ingest,
        "role_balance": metab.role_balance().to_dict(),
        "work_units": [w.to_dict() for w in metab.work_units],
        "ingest_log": [r.to_dict() for r in metab.ingest_log],
        "formal_verdict": (alu_ingest or {}).get("formal_verdict"),
        "nuance": (alu_ingest or {}).get("nuance"),
        "what_the_alu_result_did": {
            "promoted": (alu_ingest or {}).get("promoted"),
            "demoted": (alu_ingest or {}).get("demoted"),
            "mixed": (alu_ingest or {}).get("mixed"),
            "formal_verdict": (alu_ingest or {}).get("formal_verdict"),
            "nuance": (alu_ingest or {}).get("nuance"),
            "not_a_pass": True,
            "not_a_fail": True,
            "collapsed_to_ALU_BOUND": False,
        },
        "no_era_vi": True,
        "no_odyssey_iv": True,
    }
    _assert_no_hardware_claims(doc)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        sealed = dict(doc)
        from tools.future._common import seal, bench_block

        sealed.setdefault("bench", bench_block(RECORDED_BY))
        sealed.setdefault("claim_boundary", CLAIM_BOUNDARY)
        seal(sealed)
        path.write_text(json.dumps(sealed, indent=1, sort_keys=True) + "\n")
        return path
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"no to_dict for {type(obj)!r}")


__all__ = (
    "ADVERSARY",
    "CLAIM_BOUNDARY",
    "DEMOTED",
    "FALSIFIER",
    "INGEST_CHANGED_NOTHING",
    "KEEP",
    "KILLED",
    "LAW",
    "MIXED",
    "MUTATION",
    "OPEN",
    "ORACLE",
    "PARK",
    "PARKED",
    "POLARITIES",
    "POLARITY_DEMOTE",
    "POLARITY_MIXED",
    "POLARITY_PROMOTE",
    "PROBE",
    "PROMOTED",
    "QUALIFICATION",
    "RECEIPT",
    "REPLICATION",
    "RESOURCES",
    "ROLES",
    "ROLLBACK",
    "SCAR",
    "SCHEMA",
    "SETTLED",
    "STATUSES",
    "TENET",
    "TERMINALS",
    "VERSION",
    "Frontier",
    "Hypothesis",
    "IngestResult",
    "Metabolism",
    "MissingReceipt",
    "NotAHypothesis",
    "ParkWithoutWake",
    "RoleBalance",
    "ScarIncomplete",
    "UnknownRole",
    "UnknownTerminal",
    "VerbExperiment",
    "WorkUnit",
    "apply_terminal",
    "attach",
    "build",
    "campaign",
    "frontier",
    "hypothesis",
    "ingest",
    "kill_hypothesis",
    "load_receipt",
    "role_balance",
    "terminal",
    "to_dict",
    "work_unit",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="improvement_metabolism")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        # Tiny in-process guard so a --selftest caller does not need pytest.
        try:
            hypothesis(
                id="x",
                title="t",
                prior_confidence=0.5,
                max_possible_gain_ms=1.0,
                cheapest_decisive_experiment="investigate",
                expected_runtime_s=1.0,
                required_resource="CPU",
                falsifier="f",
            )
        except VerbExperiment:
            pass
        else:
            raise SystemExit("selftest: verb experiment was not refused")
        print("selftest_ok")
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
