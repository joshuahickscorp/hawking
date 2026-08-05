#!/usr/bin/env python3.12
"""Paired GLM/DSV4F functional-transfer evidence schema + membership manager.

Stage 2/3 scaffolding: formats, loaders, validators, disjoint membership.
Capture of real GLM trajectories is REQUIRES_GLM_RUNTIME (fail closed).
No fabricated teacher trajectories, activations, or benchmark numbers.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lab.operators.frankenstein_gates import (
    REQUIRES_GLM_RUNTIME,
    fail_closed,
    gate_record,
)
from lab.receipts import seal, verify


TRACE_SCHEMA = "hawking.frankenstein.paired_functional_trace.v1"
MEMBERSHIP_SCHEMA = "hawking.frankenstein.evidence_membership.v1"
CORPUS_INDEX_SCHEMA = "hawking.frankenstein.evidence_corpus_index.v1"

MEMBERSHIP_SPLITS: tuple[str, ...] = (
    "train",
    "calibration",
    "public_test",
    "hidden_test",
)

# Fields that a complete paired evidence item may carry.
TRACE_FIELD_SPEC: dict[str, dict[str, Any]] = {
    "example_id": {"type": "str", "required": True},
    "membership": {"type": "split", "required": True},
    "prompt_text": {"type": "str", "required": True},
    "decoded_spans": {"type": "list", "required": True},
    "method_family_choice": {"type": "str|null", "required": False},
    "decomposition": {"type": "list|null", "required": False},
    "proof_plan": {"type": "list|null", "required": False},
    "formal_actions": {"type": "list", "required": True},
    "tool_events": {"type": "list", "required": True},
    "repair_steps": {"type": "list", "required": True},
    "verification": {"type": "object|null", "required": False},
    "bounded_logits_top_k": {"type": "list|null", "required": False},
    "representative_hidden_states": {"type": "list|null", "required": False},
    "route_statistics": {"type": "object|null", "required": False},
    "sides": {"type": "object", "required": True},  # glm / dsv4f partials
}


class TraceFormatError(RuntimeError):
    """Schema / membership / validation failure."""


class CaptureGatedError(TraceFormatError):
    """Real capture refused — GLM runtime absent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _regular_file(path: Path, label: str) -> None:
    try:
        node = os.lstat(path)
    except OSError as exc:
        raise TraceFormatError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        raise TraceFormatError(f"{label} must be a regular non-symlink file")


# ---------------------------------------------------------------------------
# Decoded spans / actions (tokenizer-independent units)
# ---------------------------------------------------------------------------


def make_decoded_span(
    *,
    text: str,
    byte_start: int,
    byte_end: int,
    role: str = "content",
    side: str | None = None,
) -> dict[str, Any]:
    """A UTF-8 decoded span with exclusive byte range into the shared surface."""

    if byte_start < 0 or byte_end < byte_start:
        raise TraceFormatError(f"invalid byte range [{byte_start}, {byte_end})")
    encoded = text.encode("utf-8")
    if byte_end - byte_start != len(encoded) and text:
        # Allow empty spans; otherwise length should match when text is the span body.
        pass
    return {
        "text": text,
        "byte_start": int(byte_start),
        "byte_end": int(byte_end),
        "role": role,
        "side": side,
        # Explicitly no token_ids — aligners must not use them across tokenizers.
        "token_ids": None,
        "token_ids_forbidden_for_alignment": True,
    }


def make_formal_action(
    *,
    action_type: str,
    payload: Mapping[str, Any] | None = None,
    span_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "payload": dict(payload or {}),
        "span_ref": dict(span_ref) if span_ref else None,
    }


def make_tool_event(
    *,
    tool_name: str,
    args: Mapping[str, Any] | None = None,
    result_summary: str | None = None,
    ok: bool | None = None,
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "args": dict(args or {}),
        "result_summary": result_summary,
        "ok": ok,
    }


def make_route_statistics(
    *,
    expert_counts: Mapping[int, int] | None = None,
    top_k: int | None = None,
    load_balance_cv: float | None = None,
) -> dict[str, Any]:
    return {
        "expert_counts": {str(k): int(v) for k, v in (expert_counts or {}).items()},
        "top_k": top_k,
        "load_balance_cv": load_balance_cv,
        "note": "route stats are student-native; never copy GLM router weights",
    }


# ---------------------------------------------------------------------------
# Membership manager (disjoint train/calib/public/hidden)
# ---------------------------------------------------------------------------


@dataclass
class MembershipManager:
    """Disjoint split assignment; refuses double-booking an example_id."""

    assignments: dict[str, str] = field(default_factory=dict)

    def assign(self, example_id: str, split: str) -> None:
        if split not in MEMBERSHIP_SPLITS:
            raise TraceFormatError(
                f"unknown split {split!r}; permitted={list(MEMBERSHIP_SPLITS)}"
            )
        if not example_id or not str(example_id).strip():
            raise TraceFormatError("example_id must be non-empty")
        eid = str(example_id)
        existing = self.assignments.get(eid)
        if existing is not None and existing != split:
            raise TraceFormatError(
                f"example_id {eid!r} already in split {existing!r}; "
                f"refusing reassignment to {split!r} (membership must be disjoint)"
            )
        self.assignments[eid] = split

    def get(self, example_id: str) -> str | None:
        return self.assignments.get(str(example_id))

    def assert_disjoint(self) -> dict[str, list[str]]:
        buckets: dict[str, list[str]] = {s: [] for s in MEMBERSHIP_SPLITS}
        for eid, split in self.assignments.items():
            buckets[split].append(eid)
        # Disjoint by construction (one map); report sizes.
        return buckets

    def seal_document(self) -> dict[str, Any]:
        buckets = self.assert_disjoint()
        doc = {
            "schema": MEMBERSHIP_SCHEMA,
            "recorded_at": _utc_now(),
            "splits": MEMBERSHIP_SPLITS,
            "counts": {s: len(v) for s, v in buckets.items()},
            "assignments": dict(sorted(self.assignments.items())),
            "disjoint": True,
            "note": (
                "train/calibration/public_test/hidden_test must never share example_ids"
            ),
        }
        return seal(doc)

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "MembershipManager":
        if document.get("schema") != MEMBERSHIP_SCHEMA:
            raise TraceFormatError(
                f"membership schema mismatch: {document.get('schema')!r}"
            )
        verify(document, label="membership")
        mgr = cls()
        for eid, split in (document.get("assignments") or {}).items():
            mgr.assign(str(eid), str(split))
        return mgr


# ---------------------------------------------------------------------------
# Trace build / validate / load
# ---------------------------------------------------------------------------


def empty_side(side: str) -> dict[str, Any]:
    return {
        "side": side,
        "present": False,
        "decoded_spans": [],
        "method_family_choice": None,
        "decomposition": None,
        "proof_plan": None,
        "formal_actions": [],
        "tool_events": [],
        "repair_steps": [],
        "verification": None,
        "bounded_logits_top_k": None,
        "representative_hidden_states": None,
        "route_statistics": None,
        "capture_status": "ABSENT",
    }


def build_paired_trace(
    *,
    example_id: str,
    membership: str,
    prompt_text: str,
    dsv4f_side: Mapping[str, Any] | None = None,
    glm_side: Mapping[str, Any] | None = None,
    decoded_spans: Sequence[Mapping[str, Any]] | None = None,
    method_family_choice: str | None = None,
    decomposition: Sequence[Any] | None = None,
    proof_plan: Sequence[Any] | None = None,
    formal_actions: Sequence[Mapping[str, Any]] | None = None,
    tool_events: Sequence[Mapping[str, Any]] | None = None,
    repair_steps: Sequence[Mapping[str, Any]] | None = None,
    verification: Mapping[str, Any] | None = None,
    bounded_logits_top_k: Sequence[Mapping[str, Any]] | None = None,
    representative_hidden_states: Sequence[Mapping[str, Any]] | None = None,
    route_statistics: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a sealed paired-trace document (may be student-only until GLM lands)."""

    if membership not in MEMBERSHIP_SPLITS:
        raise TraceFormatError(f"invalid membership {membership!r}")
    if not example_id or not str(example_id).strip():
        raise TraceFormatError("example_id required")

    sides = {
        "dsv4f": dict(dsv4f_side) if dsv4f_side is not None else empty_side("dsv4f"),
        "glm": dict(glm_side) if glm_side is not None else empty_side("glm"),
    }
    for key in ("dsv4f", "glm"):
        sides[key].setdefault("side", key)

    document = {
        "schema": TRACE_SCHEMA,
        "recorded_at": _utc_now(),
        "example_id": str(example_id),
        "membership": membership,
        "prompt_text": prompt_text,
        "decoded_spans": [dict(s) for s in (decoded_spans or [])],
        "method_family_choice": method_family_choice,
        "decomposition": list(decomposition) if decomposition is not None else None,
        "proof_plan": list(proof_plan) if proof_plan is not None else None,
        "formal_actions": [dict(a) for a in (formal_actions or [])],
        "tool_events": [dict(t) for t in (tool_events or [])],
        "repair_steps": [dict(r) for r in (repair_steps or [])],
        "verification": dict(verification) if verification is not None else None,
        "bounded_logits_top_k": (
            [dict(x) for x in bounded_logits_top_k]
            if bounded_logits_top_k is not None
            else None
        ),
        "representative_hidden_states": (
            [dict(x) for x in representative_hidden_states]
            if representative_hidden_states is not None
            else None
        ),
        "route_statistics": (
            dict(route_statistics) if route_statistics is not None else None
        ),
        "sides": sides,
        "alignment_policy": {
            "align_on": ["decoded_spans", "byte_ranges", "formal_actions", "tool_events"],
            "never_align_on": ["token_ids", "incompatible_vocab_indices"],
        },
        "meta": dict(meta or {}),
        "fabricated": False,
    }
    validate_trace(document)
    return seal(document)


def validate_trace(document: Mapping[str, Any]) -> None:
    """Fail closed on schema / membership / token-id-alignment violations."""

    if not isinstance(document, Mapping):
        raise TraceFormatError("trace must be a mapping")
    schema = document.get("schema")
    if schema is not None and schema != TRACE_SCHEMA:
        raise TraceFormatError(f"trace schema mismatch: {schema!r}")
    for key in ("example_id", "membership", "prompt_text", "sides"):
        if key not in document:
            raise TraceFormatError(f"trace missing required field {key!r}")
    if document["membership"] not in MEMBERSHIP_SPLITS:
        raise TraceFormatError(f"bad membership {document['membership']!r}")
    sides = document["sides"]
    if not isinstance(sides, Mapping) or "dsv4f" not in sides or "glm" not in sides:
        raise TraceFormatError("sides must include dsv4f and glm")
    # Forbid token-id alignment payloads.
    for span in document.get("decoded_spans") or []:
        if not isinstance(span, Mapping):
            raise TraceFormatError("decoded_span must be mapping")
        if span.get("token_ids") not in (None, [], ()):
            if span.get("token_ids_forbidden_for_alignment") is not True:
                raise TraceFormatError(
                    "decoded_span carries token_ids without "
                    "token_ids_forbidden_for_alignment=true; refuse"
                )
    if document.get("fabricated") is True:
        raise TraceFormatError("refusing fabricated=true trace")


def load_trace(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    _regular_file(p, "trace")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("schema") != TRACE_SCHEMA:
        raise TraceFormatError(f"unexpected schema {doc.get('schema')!r}")
    verify(doc, label="paired trace")
    validate_trace(doc)
    return doc


def load_trace_corpus(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    return [load_trace(p) for p in paths]


def index_corpus(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a sealed corpus index; enforces membership disjointness."""

    mgr = MembershipManager()
    rows: list[dict[str, Any]] = []
    for tr in traces:
        validate_trace(tr)
        eid = str(tr["example_id"])
        split = str(tr["membership"])
        mgr.assign(eid, split)
        rows.append(
            {
                "example_id": eid,
                "membership": split,
                "seal_sha256": tr.get("seal_sha256"),
                "glm_present": bool((tr.get("sides") or {}).get("glm", {}).get("present")),
                "dsv4f_present": bool(
                    (tr.get("sides") or {}).get("dsv4f", {}).get("present")
                ),
            }
        )
    doc = {
        "schema": CORPUS_INDEX_SCHEMA,
        "recorded_at": _utc_now(),
        "n_traces": len(rows),
        "membership": mgr.seal_document(),
        "rows": rows,
        "fabricated": False,
    }
    return seal(doc)


# ---------------------------------------------------------------------------
# Capture interface (fail closed)
# ---------------------------------------------------------------------------


def capture_glm_trajectory(
    *,
    example_id: str,
    prompt_text: str,
    membership: str,
    glm_runtime: Any | None = None,
) -> dict[str, Any]:
    """Capture GLM side evidence — REQUIRES_GLM_RUNTIME, never faked."""

    if glm_runtime is None:
        closed = fail_closed(
            REQUIRES_GLM_RUNTIME,
            stage="2_paired_evidence",
            operation="capture_glm_trajectory",
        )
        closed["example_id"] = example_id
        closed["membership"] = membership
        closed["prompt_text_sha256"] = _sha256(prompt_text.encode("utf-8"))
        closed["gate_record"] = gate_record(REQUIRES_GLM_RUNTIME, open_=False)
        return closed
    # Real runtime path is not implemented in this scaffold; still fail closed
    # unless a future callable protocol is registered.
    if not callable(getattr(glm_runtime, "generate_trace", None)):
        closed = fail_closed(
            REQUIRES_GLM_RUNTIME,
            stage="2_paired_evidence",
            operation="capture_glm_trajectory",
        )
        closed["detail"] = "glm_runtime lacks generate_trace callable"
        return closed
    # Delegate only — do not invent fields here.
    return glm_runtime.generate_trace(
        example_id=example_id, prompt_text=prompt_text, membership=membership
    )


def capture_paired_evidence(
    *,
    example_id: str,
    prompt_text: str,
    membership: str,
    dsv4f_side: Mapping[str, Any] | None = None,
    glm_runtime: Any | None = None,
) -> dict[str, Any]:
    """Build student-side trace + attempt GLM capture (fail closed if absent)."""

    glm_capture = capture_glm_trajectory(
        example_id=example_id,
        prompt_text=prompt_text,
        membership=membership,
        glm_runtime=glm_runtime,
    )
    glm_present = glm_capture.get("status") != "FAIL_CLOSED" and bool(
        glm_capture.get("present")
    )
    glm_side = empty_side("glm")
    if glm_present:
        glm_side = {**glm_side, **glm_capture, "present": True, "capture_status": "OK"}
    else:
        glm_side["capture_status"] = "FAIL_CLOSED"
        glm_side["gate"] = glm_capture

    if dsv4f_side is None:
        dsv4f = empty_side("dsv4f")
        dsv4f["capture_status"] = "NOT_PROVIDED"
    else:
        dsv4f = {**empty_side("dsv4f"), **dict(dsv4f_side), "present": True}

    if not glm_present:
        # Honest partial trace: student may be present; GLM gated.
        return build_paired_trace(
            example_id=example_id,
            membership=membership,
            prompt_text=prompt_text,
            dsv4f_side=dsv4f,
            glm_side=glm_side,
            meta={
                "glm_capture": glm_capture,
                "complete_pair": False,
            },
        )
    return build_paired_trace(
        example_id=example_id,
        membership=membership,
        prompt_text=prompt_text,
        dsv4f_side=dsv4f,
        glm_side=glm_side,
        meta={"complete_pair": True},
    )
