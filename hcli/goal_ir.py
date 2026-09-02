"""GoalIR: the typed schema every goal-compiler lane normalizes into.

``hcli.goal.GoalCompiler`` is an OBLIGATION EXTRACTOR: it tokenizes goal
prose straight into ``WorkUnit``s, one obligation per sentence/heading. That
is useful compiler IR for dispatch, but it throws away everything this
module exists to keep: who said it (provenance), whether it is the outcome
or just one way to reach it (OUTCOME vs METHOD), whether it is negotiable
(HARD_CONSTRAINT vs SOFT_PREFERENCE), and the exact original bytes it came
from.

This module is DATA AND NORMALIZATION ONLY:

* no prose parsing (that is the tokenizer lane -- it reads the directive and
  decides what ``GoalType``/``Provenance``/``statement`` a sentence becomes)
* no graph algorithms (that is the graph lane -- dedupe, contradiction
  detection, and multi-node traversal all require comparing many nodes,
  which is a semantic judgment this module refuses to make; the closest
  thing here, ``content_signature``, is a fingerprint the graph lane can use
  for that job, not a decision)
* no scheduler contact (``hcli.goal.WorkUnitDAG`` / ``hcli.mission`` /
  ``hcli.ledger`` are untouched here; the adapter lane compiles a
  ``GoalNode`` graph into WorkUnits, this module does not)

THE LOAD-BEARING SEPARATION: ``OBJECTIVE`` (an outcome, e.g. "reduce Odyssey
wall time") and ``SUGGESTED_METHOD`` (one way to get there, e.g. "cache
models on SSD") are different ``GoalType`` values on purpose. A method can be
tried, fail its verifier, and be superseded by another method while the
objective it served stays ``ACTIVE``. Folding a method into its objective's
``statement`` is exactly the mistake this schema exists to make impossible
by construction: there would be nothing left to supersede without also
discarding the outcome it was in service of.

PROVENANCE CANNOT BE SILENTLY PROMOTED. ``GoalNode`` is a frozen dataclass:
no field, including ``provenance``, can be reassigned after construction --
"upgrading" an inference means building a brand new node, and every
construction path (the constructor, ``dataclasses.replace``, ``from_dict``)
re-runs ``__post_init__``, which enforces that ``provenance=EXPLICIT_USER``
requires a genuine ``PASTE`` ``source_ref`` (see ``preserve_source``) and
that ``provenance=MODEL_INFERRED`` may never carry one. This stops the
*silent* case -- an inferred node quietly relabeled as user intent without
ever capturing user text. It does not stop a caller willing to fabricate a
``SourceRef`` outright; ``verify_source_refs`` closes that gap by checking
the referenced bytes actually exist and hash-match, but it requires I/O so
it is never run implicitly at construction time.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple, Type, Union

from .goal import identity_for_path
from .paste_cache import PasteCache, PasteNotFound
from .tool_registry import MUTATION_CLASSES

# A node holds a compact restatement, never a transcript -- the original
# text lives in a SourceRef's paste, retrievable in full for audit.
STATEMENT_MAX_CHARS = 500
MAX_ID_CHARS = 96
PRIORITY_MIN, PRIORITY_MAX = 0, 3

_ID_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


class GoalType(str, Enum):
    """Typed atoms. Trimmed to what V1 needs -- do not grow this ad hoc."""

    ULTRAGOAL = "ULTRAGOAL"
    OBJECTIVE = "OBJECTIVE"
    SUBOBJECTIVE = "SUBOBJECTIVE"
    SUCCESS_CRITERION = "SUCCESS_CRITERION"
    FAILURE_CRITERION = "FAILURE_CRITERION"
    HARD_CONSTRAINT = "HARD_CONSTRAINT"
    SOFT_PREFERENCE = "SOFT_PREFERENCE"
    AUTHORITY_GRANT = "AUTHORITY_GRANT"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    PROHIBITION = "PROHIBITION"
    EVIDENCE_REQUIREMENT = "EVIDENCE_REQUIREMENT"
    PRIORITY = "PRIORITY"
    DEPENDENCY = "DEPENDENCY"
    RESOURCE_REQUIREMENT = "RESOURCE_REQUIREMENT"
    TEMPORAL_CONSTRAINT = "TEMPORAL_CONSTRAINT"
    CONTINUATION_POLICY = "CONTINUATION_POLICY"
    STOP_CONDITION = "STOP_CONDITION"
    ANTI_GOAL = "ANTI_GOAL"
    SUGGESTED_METHOD = "SUGGESTED_METHOD"
    EXAMPLE = "EXAMPLE"
    HYPOTHESIS = "HYPOTHESIS"
    OPEN_QUESTION = "OPEN_QUESTION"
    FUTURE_OPTION = "FUTURE_OPTION"


class Provenance(str, Enum):
    """Mandatory on every node. See the module docstring for the promotion rule."""

    EXPLICIT_USER = "EXPLICIT_USER"
    DERIVED = "DERIVED"
    MODEL_INFERRED = "MODEL_INFERRED"
    DISK_DERIVED = "DISK_DERIVED"
    POLICY_DERIVED = "POLICY_DERIVED"


class Status(str, Enum):
    """Lifecycle. See ``_STATUS_TRANSITIONS`` for what moves where."""

    ACTIVE = "ACTIVE"
    SLEEPING = "SLEEPING"
    PARKED = "PARKED"
    COMPLETE = "COMPLETE"
    SUPERSEDED = "SUPERSEDED"
    BLOCKED = "BLOCKED"


class SourceKind(str, Enum):
    """What a ``SourceRef`` points at."""

    PASTE = "PASTE"       # verbatim directive text, stored via PasteCache
    DISK = "DISK"          # a file, identified via hcli.goal.identity_for_path
    POLICY = "POLICY"      # a standing policy/doctrine, not one utterance
    INFERENCE = "INFERENCE"  # no literal span; a model inferred this


def _coerce_enum(value: Any, enum_cls: Type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except ValueError:
        valid = ", ".join(m.value for m in enum_cls)
        raise ValueError(
            f"{field_name}={value!r} is not one of: {valid}"
        ) from None


@dataclass(frozen=True)
class SourceRef:
    """A pointer from a GoalNode back to the exact span it was derived from.

    Never a copy of the text itself -- ``ref`` names where the real bytes
    live (a ``PasteCache`` id, a file path, a policy name, or a free-form
    label for an ungrounded inference) and ``char_start``/``char_end`` (when
    meaningful) narrow it to a span within that source.
    """

    kind: SourceKind
    ref: str
    sha256: str = ""
    char_start: int = 0
    char_end: Optional[int] = None
    mission: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_enum(self.kind, SourceKind, "kind"))
        ref = str(self.ref).strip()
        if not ref:
            raise ValueError("SourceRef.ref must not be empty")
        object.__setattr__(self, "ref", ref)
        if self.char_start < 0:
            raise ValueError(f"SourceRef.char_start must be >= 0, got {self.char_start}")
        if self.char_end is not None and self.char_end < self.char_start:
            raise ValueError("SourceRef.char_end must be >= char_start")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "ref": self.ref,
            "sha256": self.sha256,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "mission": self.mission,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceRef":
        return cls(
            kind=data["kind"],
            ref=data["ref"],
            sha256=str(data.get("sha256") or ""),
            char_start=int(data.get("char_start") or 0),
            char_end=data.get("char_end"),
            mission=data.get("mission"),
        )


def preserve_source(
    cache: PasteCache,
    raw_text: str,
    *,
    char_start: int = 0,
    char_end: Optional[int] = None,
    mission: Optional[str] = None,
) -> SourceRef:
    """Store *raw_text* verbatim and return the one sanctioned PASTE ref to it.

    This is the only way to mint a PASTE ``SourceRef`` from live text: the
    bytes are written, hashed, and timestamped by ``PasteCache.store`` (raw
    bytes + sha256 + timestamp + mission -- exactly what source preservation
    requires) before anything can point at them. A ``SourceRef`` can name a
    span; it cannot invent one.
    """
    ref = cache.store(raw_text, mission=mission)
    end = len(raw_text) if char_end is None else char_end
    return SourceRef(
        kind=SourceKind.PASTE,
        ref=ref.id,
        sha256=ref.sha256,
        char_start=char_start,
        char_end=end,
        mission=mission,
    )


def preserve_disk_source(
    path: Union[str, Path], *, root: Optional[Union[str, Path]] = None
) -> SourceRef:
    """A DISK ``SourceRef``, identified the same way ``goal.py`` freshness-checks evidence."""
    ident = identity_for_path(path, root=root)
    return SourceRef(kind=SourceKind.DISK, ref=ident.path, sha256=ident.sha256)


class SourceIntegrityError(ValueError):
    """Raised when a SourceRef's claimed bytes do not match what is on disk."""


def verify_source_refs(
    node: "GoalNode", *, paste_cache: Optional[PasteCache] = None
) -> None:
    """Confirm every PASTE/DISK source_ref still points at matching bytes.

    Construction-time validation (``GoalNode.__post_init__``) only checks
    SHAPE -- that a ref of the right kind is present. It cannot check that
    the bytes are genuine without I/O, so it does not try. This is the
    explicit, I/O-doing check: call it before trusting a node's lineage for
    something consequential (audit, reinterpretation, a permission decision).
    POLICY and INFERENCE refs have nothing on disk to compare against and
    are skipped.
    """
    cache = paste_cache
    for sref in node.source_refs:
        if sref.kind is SourceKind.PASTE:
            if cache is None:
                cache = PasteCache()
            try:
                text = cache.get(sref.ref)
            except PasteNotFound as exc:
                raise SourceIntegrityError(
                    f"{node.id}: paste {sref.ref!r} no longer exists"
                ) from exc
            except ValueError as exc:
                # PasteCache.get() itself refuses a paste whose stored bytes
                # no longer match its own sidecar's sha256 -- still a
                # tampered/corrupt source from this node's point of view.
                raise SourceIntegrityError(
                    f"{node.id}: paste {sref.ref!r} sha256 mismatch: {exc}"
                ) from exc
            actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if sref.sha256 and actual != sref.sha256:
                raise SourceIntegrityError(
                    f"{node.id}: paste {sref.ref!r} sha256 mismatch "
                    f"(recorded {sref.sha256}, now {actual})"
                )
        elif sref.kind is SourceKind.DISK:
            try:
                ident = identity_for_path(sref.ref)
            except OSError as exc:
                raise SourceIntegrityError(
                    f"{node.id}: disk source {sref.ref!r} unreadable: {exc}"
                ) from exc
            if sref.sha256 and ident.sha256 != sref.sha256:
                raise SourceIntegrityError(
                    f"{node.id}: disk source {sref.ref!r} sha256 mismatch "
                    f"(recorded {sref.sha256}, now {ident.sha256})"
                )


def make_stable_id(goal_type: Union[GoalType, str], slug: str) -> str:
    """Derive a canonical id: ``TYPE_NORMALIZED_SLUG``.

    Identity must survive paraphrase, compaction, and a change of which
    model wrote the compiled goal. It is deliberately NOT a hash of the
    directive text -- ``hash("make odyssey faster pls")`` mints a fresh id
    for every rephrasing of the exact same goal and silently forks it. The
    caller (the tokenizer lane, which reads the prose) names the topic the
    way a person names a ticket; this function only normalizes that name
    into a stable, collision-legible format.

    What this CANNOT do: recognize, on its own, that two differently-worded
    slugs name the same real goal, or that a reused slug for what is
    actually a new goal is a coincidence rather than a restatement. Both are
    semantic judgments over more than one node -- the graph lane's job (see
    ``content_signature`` for a fingerprint it can use), not this function's.
    """
    gt = _coerce_enum(goal_type, GoalType, "goal_type")
    normalized = _SLUG_RE.sub("_", str(slug).strip()).strip("_").upper()
    if not normalized:
        raise ValueError(f"slug has no alphanumeric content: {slug!r}")
    stable_id = f"{gt.value}_{normalized}"
    if len(stable_id) > MAX_ID_CHARS:
        raise ValueError(
            f"stable id is {len(stable_id)} chars (max {MAX_ID_CHARS}); "
            f"shorten the slug: {stable_id!r}"
        )
    return stable_id


def _clean_tuple(value: Any) -> Tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(item).strip() for item in (value or ()) if str(item).strip()
        )
    )


@dataclass(frozen=True)
class GoalNode:
    """One typed, provenance-tagged atom of a compiled goal.

    Immutable by construction (frozen dataclass): a lifecycle change or any
    other update goes through ``dataclasses.replace``/``transition``, which
    produce a new node and re-run ``__post_init__``'s validation rather than
    mutating one in place. Kept COMPACT on purpose -- ``statement`` is a
    restatement, not a transcript; the transcript lives behind
    ``source_refs`` (see ``preserve_source``).
    """

    id: str
    type: GoalType
    statement: str
    provenance: Provenance
    confidence: float = 1.0
    priority: int = 2
    status: Status = Status.ACTIVE
    dependencies: Tuple[str, ...] = ()
    blockers: Tuple[str, ...] = ()
    success_criteria: Tuple[str, ...] = ()
    failure_criteria: Tuple[str, ...] = ()
    evidence_requirements: Tuple[str, ...] = ()
    authority_class: Optional[str] = None
    resources: Tuple[str, ...] = ()
    parent_ultragoal: Optional[str] = None
    related_frontiers: Tuple[str, ...] = ()
    reopen_condition: Optional[str] = None
    source_refs: Tuple[SourceRef, ...] = ()
    superseded_by: Optional[str] = None

    def __post_init__(self) -> None:
        node_id = str(self.id).strip()
        if not _ID_RE.match(node_id):
            raise ValueError(
                f"GoalNode.id must be canonical UPPER_SNAKE, got {self.id!r}; "
                "see make_stable_id()"
            )
        object.__setattr__(self, "id", node_id)

        object.__setattr__(self, "type", _coerce_enum(self.type, GoalType, "type"))
        object.__setattr__(
            self, "provenance", _coerce_enum(self.provenance, Provenance, "provenance")
        )
        object.__setattr__(self, "status", _coerce_enum(self.status, Status, "status"))

        statement = str(self.statement).strip()
        if not statement:
            raise ValueError(f"{node_id}: statement must not be empty")
        if len(statement) > STATEMENT_MAX_CHARS:
            raise ValueError(
                f"{node_id}: statement is {len(statement)} chars "
                f"(max {STATEMENT_MAX_CHARS}) -- a node holds a compact "
                "restatement, not a transcript; preserve the original via "
                "preserve_source() and cite it in source_refs"
            )
        object.__setattr__(self, "statement", statement)

        confidence = float(self.confidence)
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"{node_id}: confidence must be in [0,1], got {confidence!r}")
        object.__setattr__(self, "confidence", confidence)

        priority = int(self.priority)
        if not (PRIORITY_MIN <= priority <= PRIORITY_MAX):
            raise ValueError(
                f"{node_id}: priority must be in [{PRIORITY_MIN},{PRIORITY_MAX}], "
                f"got {priority!r}"
            )
        object.__setattr__(self, "priority", priority)

        for field_name in (
            "dependencies",
            "blockers",
            "success_criteria",
            "failure_criteria",
            "evidence_requirements",
            "resources",
            "related_frontiers",
        ):
            object.__setattr__(self, field_name, _clean_tuple(getattr(self, field_name)))

        object.__setattr__(
            self,
            "source_refs",
            tuple(
                item if isinstance(item, SourceRef) else SourceRef.from_dict(item)
                for item in (self.source_refs or ())
            ),
        )

        if self.authority_class is not None and self.authority_class not in MUTATION_CLASSES:
            raise ValueError(
                f"{node_id}: authority_class={self.authority_class!r} is not one "
                f"of hcli.tool_registry.MUTATION_CLASSES: {sorted(MUTATION_CLASSES)}"
            )

        if self.parent_ultragoal is not None:
            object.__setattr__(
                self, "parent_ultragoal", str(self.parent_ultragoal).strip() or None
            )
        if self.reopen_condition is not None:
            object.__setattr__(
                self, "reopen_condition", str(self.reopen_condition).strip() or None
            )
        if self.superseded_by is not None:
            superseded_by = str(self.superseded_by).strip() or None
            object.__setattr__(self, "superseded_by", superseded_by)
            if superseded_by and self.status is not Status.SUPERSEDED:
                raise ValueError(
                    f"{node_id}: superseded_by is set but status is "
                    f"{self.status.value}, not SUPERSEDED"
                )

        # ---- the promotion guard: see the module docstring -----------------
        kinds = {sref.kind for sref in self.source_refs}
        if self.provenance is Provenance.EXPLICIT_USER and SourceKind.PASTE not in kinds:
            raise ValueError(
                f"{node_id}: provenance=EXPLICIT_USER requires a PASTE "
                "source_ref (preserve_source()) -- an inference must never "
                "be recorded as if the user said it verbatim"
            )
        if self.provenance is Provenance.MODEL_INFERRED and SourceKind.PASTE in kinds:
            raise ValueError(
                f"{node_id}: provenance=MODEL_INFERRED must not carry a "
                "PASTE source_ref -- that would claim the inference is "
                "captured user text; use EXPLICIT_USER or DERIVED instead"
            )
        if self.provenance is Provenance.DERIVED and not self.source_refs:
            raise ValueError(f"{node_id}: provenance=DERIVED requires at least one source_ref")
        if self.provenance is Provenance.DISK_DERIVED and SourceKind.DISK not in kinds:
            raise ValueError(
                f"{node_id}: provenance=DISK_DERIVED requires a DISK source_ref"
            )
        if self.provenance is Provenance.POLICY_DERIVED and SourceKind.POLICY not in kinds:
            raise ValueError(
                f"{node_id}: provenance=POLICY_DERIVED requires a POLICY source_ref"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "statement": self.statement,
            "provenance": self.provenance.value,
            "confidence": self.confidence,
            "priority": self.priority,
            "status": self.status.value,
            "dependencies": list(self.dependencies),
            "blockers": list(self.blockers),
            "success_criteria": list(self.success_criteria),
            "failure_criteria": list(self.failure_criteria),
            "evidence_requirements": list(self.evidence_requirements),
            "authority_class": self.authority_class,
            "resources": list(self.resources),
            "parent_ultragoal": self.parent_ultragoal,
            "related_frontiers": list(self.related_frontiers),
            "reopen_condition": self.reopen_condition,
            "source_refs": [sref.to_dict() for sref in self.source_refs],
            "superseded_by": self.superseded_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalNode":
        return cls(
            id=data["id"],
            type=data["type"],
            statement=data["statement"],
            provenance=data["provenance"],
            confidence=data.get("confidence", 1.0),
            priority=data.get("priority", 2),
            status=data.get("status", Status.ACTIVE.value),
            dependencies=tuple(data.get("dependencies") or ()),
            blockers=tuple(data.get("blockers") or ()),
            success_criteria=tuple(data.get("success_criteria") or ()),
            failure_criteria=tuple(data.get("failure_criteria") or ()),
            evidence_requirements=tuple(data.get("evidence_requirements") or ()),
            authority_class=data.get("authority_class"),
            resources=tuple(data.get("resources") or ()),
            parent_ultragoal=data.get("parent_ultragoal"),
            related_frontiers=tuple(data.get("related_frontiers") or ()),
            reopen_condition=data.get("reopen_condition"),
            source_refs=tuple(data.get("source_refs") or ()),
            superseded_by=data.get("superseded_by"),
        )


def content_signature(node: GoalNode) -> str:
    """Cheap fingerprint of (type, statement, provenance), id/status ignored.

    A candidate signal for the graph lane's dedupe -- two nodes sharing this
    signature are very likely restatements of the same claim. This is NOT
    dedupe: deciding what to do when two DIFFERENT signatures are still the
    same real-world goal is a semantic judgment this module does not make.
    """
    payload = {
        "type": node.type.value,
        "statement": node.statement.strip().lower(),
        "provenance": node.provenance.value,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class InvalidGoalTransitionError(ValueError):
    """Raised when ``transition`` is asked to move a GoalNode illegally."""

    def __init__(self, node_id: str, from_status: Status, to_status: Status) -> None:
        self.node_id = node_id
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"invalid GoalNode transition {from_status.value!r} -> "
            f"{to_status.value!r} for {node_id}"
        )


# SUPERSEDED is terminal: a superseded node does not come back, a new node
# referencing it (parent_ultragoal / dependencies) replaces it. COMPLETE can
# return to ACTIVE because reopen_condition is meaningful for it (a
# regression reopens "make tests green"); it is not meaningful for a goal
# that was replaced by a different one entirely.
_STATUS_TRANSITIONS: Dict[Status, FrozenSet[Status]] = {
    Status.ACTIVE: frozenset(
        {Status.SLEEPING, Status.PARKED, Status.BLOCKED, Status.COMPLETE, Status.SUPERSEDED}
    ),
    Status.SLEEPING: frozenset({Status.ACTIVE, Status.SUPERSEDED}),
    Status.PARKED: frozenset({Status.ACTIVE, Status.SUPERSEDED}),
    Status.BLOCKED: frozenset({Status.ACTIVE, Status.PARKED, Status.SUPERSEDED}),
    Status.COMPLETE: frozenset({Status.ACTIVE}),
    Status.SUPERSEDED: frozenset(),
}


def can_transition(from_status: Status, to_status: Status) -> bool:
    return to_status in _STATUS_TRANSITIONS.get(from_status, frozenset())


def transition(
    node: GoalNode,
    to_status: Union[Status, str],
    *,
    superseded_by: Optional[str] = None,
    reopen_condition: Optional[str] = None,
) -> GoalNode:
    """Return a NEW GoalNode moved to *to_status*. Never mutates *node*."""
    target = _coerce_enum(to_status, Status, "to_status")
    if not can_transition(node.status, target):
        raise InvalidGoalTransitionError(node.id, node.status, target)
    changes: Dict[str, Any] = {"status": target}
    if target is Status.SUPERSEDED and superseded_by:
        changes["superseded_by"] = superseded_by
    if reopen_condition is not None:
        changes["reopen_condition"] = reopen_condition
    return replace(node, **changes)


__all__ = [
    "GoalType",
    "Provenance",
    "Status",
    "SourceKind",
    "SourceRef",
    "GoalNode",
    "SourceIntegrityError",
    "InvalidGoalTransitionError",
    "STATEMENT_MAX_CHARS",
    "MAX_ID_CHARS",
    "PRIORITY_MIN",
    "PRIORITY_MAX",
    "make_stable_id",
    "preserve_source",
    "preserve_disk_source",
    "verify_source_refs",
    "content_signature",
    "can_transition",
    "transition",
]
