#!/usr/bin/env python3.12
"""Tokenizer-independent alignment for paired GLM/DSV4F evidence.

Aligns on decoded spans, byte ranges, formal actions, and tool events.
Never matches incompatible token IDs directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class AlignerError(RuntimeError):
    """Alignment failed closed."""


@dataclass(frozen=True)
class SpanAlignment:
    left_index: int
    right_index: int
    score: float
    method: str
    left_byte_range: tuple[int, int]
    right_byte_range: tuple[int, int]


def _norm_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _span_bytes(span: Mapping[str, Any]) -> tuple[int, int]:
    return int(span.get("byte_start", 0)), int(span.get("byte_end", 0))


def assert_no_token_id_alignment(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> None:
    """Refuse any attempt to pair sequences of raw token IDs across tokenizers."""

    def _looks_like_token_ids(seq: Sequence[Mapping[str, Any]]) -> bool:
        if not seq:
            return False
        # A "token id sequence" payload would use keys like token_id without text.
        tokenish = 0
        for item in seq:
            if not isinstance(item, Mapping):
                continue
            if "token_id" in item and "text" not in item and "byte_start" not in item:
                tokenish += 1
            if item.get("align_by") == "token_id":
                raise AlignerError(
                    "align_by=token_id is forbidden across tokenizers"
                )
        return tokenish == len(seq) and len(seq) > 0

    if _looks_like_token_ids(left) or _looks_like_token_ids(right):
        raise AlignerError(
            "refusing token-ID sequence alignment; use decoded spans / byte ranges / "
            "formal actions / tool events"
        )


def align_decoded_spans(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Greedy many-to-one alignment by normalized text + overlapping byte roles.

    Score = 1.0 exact text match, 0.75 if one contains the other after normalize,
    0.0 otherwise.  Byte ranges recorded for provenance; not required equal
    (different tokenizers may emit different surface offsets into their own
    buffers — we align by text content, not shared buffer).
    """

    assert_no_token_id_alignment(left, right)
    used_right: set[int] = set()
    alignments: list[dict[str, Any]] = []
    for i, ls in enumerate(left):
        ltext = _norm_text(str(ls.get("text", "")))
        if not ltext:
            continue
        best_j = None
        best_score = 0.0
        for j, rs in enumerate(right):
            if j in used_right:
                continue
            rtext = _norm_text(str(rs.get("text", "")))
            if not rtext:
                continue
            if ltext == rtext:
                score = 1.0
            elif ltext in rtext or rtext in ltext:
                score = 0.75
            else:
                # Token-set Jaccard on whitespace tokens as soft signal.
                lt = set(ltext.split())
                rt = set(rtext.split())
                if not lt or not rt:
                    score = 0.0
                else:
                    score = len(lt & rt) / len(lt | rt)
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is not None and best_score >= min_score:
            used_right.add(best_j)
            lb = _span_bytes(ls)
            rb = _span_bytes(right[best_j])
            alignments.append(
                {
                    "left_index": i,
                    "right_index": best_j,
                    "score": float(best_score),
                    "method": "decoded_span_text",
                    "left_byte_range": [lb[0], lb[1]],
                    "right_byte_range": [rb[0], rb[1]],
                    "left_text": ls.get("text"),
                    "right_text": right[best_j].get("text"),
                }
            )
    return alignments


def align_byte_ranges(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    # Only meaningful when both sides index the *same* shared surface buffer.
    require_shared_surface: bool = True,
) -> list[dict[str, Any]]:
    """Align spans whose byte ranges are identical on a shared surface.

    If sides use independent buffers (typical cross-tokenizer case), this returns
    empty unless the caller sets require_shared_surface=False and ranges happen
    to coincide numerically (fixture use).
    """

    assert_no_token_id_alignment(left, right)
    if require_shared_surface:
        left_surface = {s.get("surface_id") for s in left if isinstance(s, Mapping)}
        right_surface = {s.get("surface_id") for s in right if isinstance(s, Mapping)}
        # If either side omits surface_id, we cannot prove a shared surface.
        if None in left_surface or None in right_surface:
            return []
        if left_surface != right_surface:
            return []

    right_by_range: dict[tuple[int, int], list[int]] = {}
    for j, rs in enumerate(right):
        key = _span_bytes(rs)
        right_by_range.setdefault(key, []).append(j)

    out: list[dict[str, Any]] = []
    for i, ls in enumerate(left):
        key = _span_bytes(ls)
        for j in right_by_range.get(key, []):
            out.append(
                {
                    "left_index": i,
                    "right_index": j,
                    "score": 1.0,
                    "method": "byte_range_exact",
                    "left_byte_range": [key[0], key[1]],
                    "right_byte_range": [key[0], key[1]],
                }
            )
    return out


def align_formal_actions(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Align formal/tool-plan actions by action_type + payload canonical keys."""

    def _key(action: Mapping[str, Any]) -> tuple[str, str]:
        at = str(action.get("action_type", ""))
        payload = action.get("payload") or {}
        # Stable payload fingerprint without token ids.
        if isinstance(payload, Mapping):
            items = sorted(
                (str(k), json_safe(v))
                for k, v in payload.items()
                if k not in {"token_id", "token_ids", "input_ids"}
            )
            payload_fp = repr(items)
        else:
            payload_fp = repr(payload)
        return at, payload_fp

    used: set[int] = set()
    out: list[dict[str, Any]] = []
    right_keys = [_key(a) for a in right]
    for i, la in enumerate(left):
        lk = _key(la)
        for j, rk in enumerate(right_keys):
            if j in used:
                continue
            if lk[0] and lk[0] == rk[0]:
                score = 1.0 if lk == rk else 0.6
                used.add(j)
                out.append(
                    {
                        "left_index": i,
                        "right_index": j,
                        "score": score,
                        "method": "formal_action",
                        "action_type": lk[0],
                        "exact_payload": lk == rk,
                    }
                )
                break
    return out


def align_tool_events(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Align tool events by tool_name (+ optional arg key overlap)."""

    used: set[int] = set()
    out: list[dict[str, Any]] = []
    for i, le in enumerate(left):
        lname = str(le.get("tool_name", ""))
        if not lname:
            continue
        largs = set((le.get("args") or {}).keys()) if isinstance(le.get("args"), Mapping) else set()
        best_j = None
        best_score = 0.0
        for j, re in enumerate(right):
            if j in used:
                continue
            rname = str(re.get("tool_name", ""))
            if lname != rname:
                continue
            rargs = (
                set((re.get("args") or {}).keys())
                if isinstance(re.get("args"), Mapping)
                else set()
            )
            if not largs and not rargs:
                score = 1.0
            elif not largs or not rargs:
                score = 0.7
            else:
                score = 0.7 + 0.3 * (len(largs & rargs) / len(largs | rargs))
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is not None:
            used.add(best_j)
            out.append(
                {
                    "left_index": i,
                    "right_index": best_j,
                    "score": float(best_score),
                    "method": "tool_event",
                    "tool_name": lname,
                }
            )
    return out


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return repr(value)


def align_paired_sides(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    """Full alignment report for two side dicts or two trace documents."""

    # Accept either a full trace or a side blob.
    if "sides" in left or "sides" in right:
        raise AlignerError("pass side dicts or span lists; use align_traces for full traces")

    l_spans = list(left.get("decoded_spans") or [])
    r_spans = list(right.get("decoded_spans") or [])
    l_actions = list(left.get("formal_actions") or [])
    r_actions = list(right.get("formal_actions") or [])
    l_tools = list(left.get("tool_events") or [])
    r_tools = list(right.get("tool_events") or [])

    assert_no_token_id_alignment(l_spans, r_spans)

    span_align = align_decoded_spans(l_spans, r_spans)
    byte_align = align_byte_ranges(l_spans, r_spans, require_shared_surface=True)
    action_align = align_formal_actions(l_actions, r_actions)
    tool_align = align_tool_events(l_tools, r_tools)

    return {
        "method_policy": {
            "allowed": [
                "decoded_spans",
                "byte_ranges",
                "formal_actions",
                "tool_events",
            ],
            "forbidden": ["token_ids", "vocab_index_match"],
        },
        "decoded_spans": span_align,
        "byte_ranges": byte_align,
        "formal_actions": action_align,
        "tool_events": tool_align,
        "summary": {
            "n_span_alignments": len(span_align),
            "n_byte_alignments": len(byte_align),
            "n_action_alignments": len(action_align),
            "n_tool_alignments": len(tool_align),
            "mean_span_score": (
                sum(a["score"] for a in span_align) / len(span_align)
                if span_align
                else None
            ),
        },
    }


def align_traces(
    left_trace: Mapping[str, Any],
    right_trace: Mapping[str, Any],
    *,
    left_side: str = "dsv4f",
    right_side: str = "glm",
) -> dict[str, Any]:
    """Align two paired traces by extracting named sides."""

    def _extract(tr: Mapping[str, Any], side: str) -> dict[str, Any]:
        sides = tr.get("sides") or {}
        blob = dict(sides.get(side) or {})
        # Fall back to top-level fields for fixture convenience.
        for key in (
            "decoded_spans",
            "formal_actions",
            "tool_events",
            "repair_steps",
        ):
            if not blob.get(key) and tr.get(key):
                blob[key] = tr[key]
        return blob

    report = align_paired_sides(_extract(left_trace, left_side), _extract(right_trace, right_side))
    report["left_example_id"] = left_trace.get("example_id")
    report["right_example_id"] = right_trace.get("example_id")
    report["left_side"] = left_side
    report["right_side"] = right_side
    return report
