"""Typed research bus. Epistemic class is load-bearing.

Parent, Child A and Child B exchange MEASURED_FACT, HYPOTHESIS,
NEGATIVE_SCIENCE, MECHANISM, COUNTEREXAMPLE, REQUEST, PATCH_SUMMARY,
PROFILE_DELTA, NEXT_BOTTLENECK. A MEASURED_FACT without a receipt
reference is refused. "This codec failed" is NEGATIVE_SCIENCE; "this
codec is impossible" requires evidence that establishes impossibility.
"""
from __future__ import annotations

from enum import Enum
import re
from typing import Any, Mapping

from lab.lineage.canon import (
    LineageValueError,
    require_mapping,
    require_nonempty_str,
    require_sha256,
    utc_now,
)
from lab.receipts import seal

SCHEMA = "hawking.lineage.research_bus.v1"
MESSAGE_SCHEMA = "hawking.lineage.research_message.v1"
GENERATION_SCOPES = frozenset({"GENERATION_BOUND", "STRUCTURALLY_TRANSFERABLE"})
_HEAD_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class BusRefusal(ValueError):
    """Message rejected before it enters the log."""


class MessageType(str, Enum):
    MEASURED_FACT = "MEASURED_FACT"
    HYPOTHESIS = "HYPOTHESIS"
    NEGATIVE_SCIENCE = "NEGATIVE_SCIENCE"
    MECHANISM = "MECHANISM"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"
    REQUEST = "REQUEST"
    PATCH_SUMMARY = "PATCH_SUMMARY"
    PROFILE_DELTA = "PROFILE_DELTA"
    NEXT_BOTTLENECK = "NEXT_BOTTLENECK"


class EpistemicClass(str, Enum):
    MEASURED = "MEASURED"
    HYPOTHESIS = "HYPOTHESIS"
    NEGATIVE = "NEGATIVE"
    IMPOSSIBLE = "IMPOSSIBLE"
    MECHANISM = "MECHANISM"
    REQUEST = "REQUEST"
    CLAIM = "CLAIM"


TYPE_TO_CLASSES: dict[MessageType, frozenset[EpistemicClass]] = {
    MessageType.MEASURED_FACT: frozenset({EpistemicClass.MEASURED}),
    MessageType.HYPOTHESIS: frozenset({EpistemicClass.HYPOTHESIS}),
    MessageType.NEGATIVE_SCIENCE: frozenset({EpistemicClass.NEGATIVE, EpistemicClass.IMPOSSIBLE}),
    MessageType.MECHANISM: frozenset({EpistemicClass.MECHANISM, EpistemicClass.HYPOTHESIS}),
    MessageType.COUNTEREXAMPLE: frozenset({EpistemicClass.MEASURED, EpistemicClass.NEGATIVE}),
    MessageType.REQUEST: frozenset({EpistemicClass.REQUEST}),
    MessageType.PATCH_SUMMARY: frozenset({EpistemicClass.CLAIM}),
    MessageType.PROFILE_DELTA: frozenset({EpistemicClass.MEASURED}),
    MessageType.NEXT_BOTTLENECK: frozenset({EpistemicClass.HYPOTHESIS, EpistemicClass.MEASURED}),
}

RECEIPT_REQUIRED: frozenset[tuple[MessageType, EpistemicClass]] = frozenset(
    {
        (MessageType.MEASURED_FACT, EpistemicClass.MEASURED),
        (MessageType.PROFILE_DELTA, EpistemicClass.MEASURED),
        (MessageType.COUNTEREXAMPLE, EpistemicClass.MEASURED),
        (MessageType.NEXT_BOTTLENECK, EpistemicClass.MEASURED),
        (MessageType.NEGATIVE_SCIENCE, EpistemicClass.IMPOSSIBLE),
    }
)


def _as_type(value: MessageType | str) -> MessageType:
    if isinstance(value, MessageType):
        return value
    try:
        return MessageType(str(value))
    except ValueError as exc:
        raise BusRefusal(
            f"unknown message type {value!r}; expected one of {[t.value for t in MessageType]}"
        ) from exc


def _as_class(value: EpistemicClass | str) -> EpistemicClass:
    if isinstance(value, EpistemicClass):
        return value
    try:
        return EpistemicClass(str(value))
    except ValueError as exc:
        raise BusRefusal(
            f"unknown epistemic_class {value!r}; expected one of {[c.value for c in EpistemicClass]}"
        ) from exc


def _receipt_ref(raw: Mapping[str, Any]) -> str:
    for key in ("receipt_ref", "receipt_reference", "measurement_ref", "measurement_reference"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _claims_impossibility(body: Mapping[str, Any]) -> bool:
    if body.get("impossible") is True:
        return True
    if body.get("establishes_impossibility") is True:
        return True
    scope = str(body.get("claim_scope") or "").strip().lower()
    if scope in {"impossible", "impossibility"}:
        return True
    return False


def _impossibility_established(body: Mapping[str, Any]) -> bool:
    if body.get("establishes_impossibility") is not True:
        return False
    resource = body.get("named_saturated_resource")
    if not isinstance(resource, str) or not resource.strip():
        return False
    return True


def validate_message(raw: Mapping[str, Any]) -> dict[str, Any]:
    data = require_mapping(raw, "research_message")
    msg_type = _as_type(data.get("type") or data.get("message_type"))
    epistemic = _as_class(data.get("epistemic_class"))
    allowed = TYPE_TO_CLASSES[msg_type]
    if epistemic not in allowed:
        raise BusRefusal(
            f"{msg_type.value} cannot carry epistemic_class {epistemic.value}; "
            f"allowed={sorted(c.value for c in allowed)}"
        )

    sender = require_nonempty_str(data.get("sender"), "sender")
    artifact_sha = require_sha256(data.get("artifact_sha"), "artifact_sha")
    runtime_sha = require_sha256(data.get("runtime_sha"), "runtime_sha")
    subsystem = require_nonempty_str(data.get("affected_subsystem"), "affected_subsystem")
    receipt = _receipt_ref(data)
    body = data.get("body")
    if body is None:
        body = {}
    if not isinstance(body, Mapping):
        raise BusRefusal("body must be a mapping when provided")
    body_d = dict(body)

    if msg_type is MessageType.MEASURED_FACT and not receipt:
        raise BusRefusal("MEASURED_FACT without a receipt reference is refused")
    if (msg_type, epistemic) in RECEIPT_REQUIRED and not receipt:
        raise BusRefusal(
            f"{msg_type.value} with epistemic_class {epistemic.value} requires a receipt reference"
        )

    claiming = _claims_impossibility(body_d) or epistemic is EpistemicClass.IMPOSSIBLE
    if claiming:
        if not _impossibility_established(body_d):
            raise BusRefusal(
                "impossibility claim refused: NEGATIVE_SCIENCE may report a failed "
                "mechanism with a receipt; IMPOSSIBLE requires "
                "establishes_impossibility=true and named_saturated_resource"
            )
        if epistemic is not EpistemicClass.IMPOSSIBLE:
            raise BusRefusal(
                "a message that claims impossibility must use epistemic_class=IMPOSSIBLE"
            )
        if msg_type is not MessageType.NEGATIVE_SCIENCE:
            raise BusRefusal("impossibility may only be typed NEGATIVE_SCIENCE")

    return {
        "schema": MESSAGE_SCHEMA,
        "type": msg_type.value,
        "sender": sender,
        "artifact_sha": artifact_sha,
        "runtime_sha": runtime_sha,
        "epistemic_class": epistemic.value,
        "receipt_ref": receipt,
        "affected_subsystem": subsystem,
        "body": body_d,
        "recorded_at": str(data.get("recorded_at") or utc_now()),
    }


def validate_generation_envelope(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Require the provenance needed to carry science across generations."""
    data = require_mapping(raw, "generation_envelope")
    generation = data.get("generation")
    if type(generation) is not int or generation < 0:
        raise BusRefusal("generation must be a non-negative integer")
    repo_head = str(data.get("repo_head") or "").strip().lower()
    if not _HEAD_RE.fullmatch(repo_head):
        raise BusRefusal("repo_head must be a 40- or 64-character lowercase hex digest")
    # A missing envelope field is a refusal, not an internal error. canon raises
    # LineageValueError, which is a sibling of BusRefusal rather than a subclass,
    # so a publisher catching refusals would crash instead of being turned away.
    try:
        facet = require_nonempty_str(data.get("facet"), "facet")
        invalidation = require_nonempty_str(
            data.get("invalidation_condition"), "invalidation_condition"
        )
    except LineageValueError as exc:
        raise BusRefusal(str(exc)) from exc
    scope = str(data.get("generation_scope") or "").strip().upper()
    if scope not in GENERATION_SCOPES:
        raise BusRefusal(
            f"generation_scope must be one of {sorted(GENERATION_SCOPES)}"
        )
    return {
        "generation": generation,
        "repo_head": repo_head,
        "facet": facet,
        "invalidation_condition": invalidation,
        "generation_scope": scope,
    }


class ResearchBus:
    """Append-only typed log. Invalid messages never enter."""

    def __init__(self) -> None:
        self._log: list[dict[str, Any]] = []

    def publish(self, message: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_message(message)
        sealed = seal(validated)
        self._log.append(sealed)
        return sealed

    def publish_generation_aware(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """Publish with the full cross-generation provenance envelope."""
        validated = validate_message(message)
        validated.update(validate_generation_envelope(message))
        sealed = seal(validated)
        self._log.append(sealed)
        return sealed

    def generation_view(
        self,
        *,
        generation: int,
        artifact_sha: str,
        runtime_sha: str,
    ) -> list[dict[str, Any]]:
        """Project immutable messages into a new generation without rewriting history."""
        require_sha256(artifact_sha, "artifact_sha")
        require_sha256(runtime_sha, "runtime_sha")
        rows: list[dict[str, Any]] = []
        for original in self._log:
            row = dict(original)
            structurally_transferable = (
                row.get("generation_scope") == "STRUCTURALLY_TRANSFERABLE"
            )
            same_identity = (
                row.get("generation") == generation
                and row.get("artifact_sha") == artifact_sha
                and row.get("runtime_sha") == runtime_sha
            )
            if same_identity:
                continuity_state = "CURRENT"
            elif structurally_transferable:
                continuity_state = "TRANSFERRED_STRUCTURAL"
            else:
                continuity_state = "STALE_PENDING_REVALIDATION"
            row["continuity_state"] = continuity_state
            rows.append(row)
        return rows

    def messages(
        self,
        *,
        type: MessageType | str | None = None,  # noqa: A002 — matches the bus field name
        sender: str | None = None,
        epistemic_class: EpistemicClass | str | None = None,
    ) -> list[dict[str, Any]]:
        rows = list(self._log)
        if type is not None:
            want = _as_type(type).value
            rows = [m for m in rows if m["type"] == want]
        if sender is not None:
            rows = [m for m in rows if m["sender"] == sender]
        if epistemic_class is not None:
            want_c = _as_class(epistemic_class).value
            rows = [m for m in rows if m["epistemic_class"] == want_c]
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "count": len(self._log),
            "messages": list(self._log),
        }
