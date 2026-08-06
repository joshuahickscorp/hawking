#!/usr/bin/env python3.12
"""Tokenizer-independent alignment for paired GLM/DSV4F evidence.

Aligns on decoded spans, UTF-8 byte ranges, formal actions, tool events, and
shared semantic anchors (claims, proof steps, subgoals, code AST regions,
tool/formal actions, answer spans).

Never matches incompatible token IDs directly.  Token→span maps are always
token-piece → UTF-8 byte range on a shared surface, never token-ID↔token-ID.
"""
from __future__ import annotations

import ast
import re
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


@dataclass(frozen=True)
class SharedSurface:
    """Shared UTF-8 text surface that both tokenizers decode into."""

    text: str
    surface_id: str = "shared"

    @property
    def raw(self) -> bytes:
        return self.text.encode("utf-8")

    @property
    def n_bytes(self) -> int:
        return len(self.raw)

    def slice_bytes(self, start: int, end: int) -> str:
        if start < 0 or end < start or end > self.n_bytes:
            raise AlignerError(f"byte range [{start}, {end}) out of surface")
        return self.raw[start:end].decode("utf-8")


@dataclass
class TokenByteSpan:
    """One tokenizer piece mapped onto the shared surface (never a foreign token id)."""

    token_index: int
    piece: str
    byte_start: int
    byte_end: int
    side: str  # "glm" | "dsv4f" | other
    # token_id is recorded only for provenance; never used for cross-side match.
    token_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_index": self.token_index,
            "piece": self.piece,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "side": self.side,
            "token_id": self.token_id,
            "align_by": "byte_span",  # explicit: never token_id
            "token_ids_forbidden_for_alignment": True,
        }


# Semantic anchor kinds required by PROTO_FRANKENSTEIN_V0 steer.
SEMANTIC_ANCHOR_KINDS: tuple[str, ...] = (
    "claim",
    "proof_step",
    "subgoal",
    "code_ast_region",
    "tool_action",
    "formal_action",
    "answer",
)


def _norm_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _span_bytes(span: Mapping[str, Any]) -> tuple[int, int]:
    return int(span.get("byte_start", 0)), int(span.get("byte_end", 0))


# ---------------------------------------------------------------------------
# Shared-surface UTF-8 spans + token-piece → byte-span maps
# ---------------------------------------------------------------------------


def utf8_byte_span(surface: SharedSurface | str, text: str) -> dict[str, Any]:
    """Locate `text` as a UTF-8 byte span inside the shared surface (first match)."""

    surf = surface if isinstance(surface, SharedSurface) else SharedSurface(text=surface)
    raw = surf.raw
    needle = text.encode("utf-8")
    if not needle:
        return {
            "text": text,
            "byte_start": 0,
            "byte_end": 0,
            "surface_id": surf.surface_id,
            "found": True,
        }
    idx = raw.find(needle)
    if idx < 0:
        return {
            "text": text,
            "byte_start": -1,
            "byte_end": -1,
            "surface_id": surf.surface_id,
            "found": False,
        }
    return {
        "text": text,
        "byte_start": idx,
        "byte_end": idx + len(needle),
        "surface_id": surf.surface_id,
        "found": True,
    }


def char_range_to_byte_span(
    surface: SharedSurface | str,
    char_start: int,
    char_end: int,
) -> dict[str, Any]:
    """Convert Python char indices to UTF-8 exclusive byte range on the surface."""

    surf = surface if isinstance(surface, SharedSurface) else SharedSurface(text=surface)
    if char_start < 0 or char_end < char_start or char_end > len(surf.text):
        raise AlignerError(
            f"char range [{char_start}, {char_end}) out of text len={len(surf.text)}"
        )
    prefix = surf.text[:char_start].encode("utf-8")
    body = surf.text[char_start:char_end].encode("utf-8")
    return {
        "byte_start": len(prefix),
        "byte_end": len(prefix) + len(body),
        "text": surf.text[char_start:char_end],
        "surface_id": surf.surface_id,
    }


def map_token_pieces_to_byte_spans(
    surface: SharedSurface | str,
    pieces: Sequence[str],
    *,
    side: str,
    token_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Map tokenizer *pieces* (decoded strings) onto shared UTF-8 byte spans.

    Greedy left-to-right consumption of the surface.  Empty / special pieces
    (e.g. ``''``, BOS markers that do not appear in surface) get zero-width
    spans at the current cursor.  Never aligns by token ID across sides.
    """

    surf = surface if isinstance(surface, SharedSurface) else SharedSurface(text=surface)
    raw = surf.raw
    cursor = 0
    out: list[dict[str, Any]] = []
    ids = list(token_ids) if token_ids is not None else [None] * len(pieces)
    if len(ids) != len(pieces):
        raise AlignerError("token_ids length must match pieces length")

    for i, piece in enumerate(pieces):
        tid = ids[i]
        if tid is not None:
            tid = int(tid)
        # Special / empty pieces: zero-width at cursor (still not used for ID match).
        if piece is None or piece == "":
            span = TokenByteSpan(
                token_index=i,
                piece="",
                byte_start=cursor,
                byte_end=cursor,
                side=side,
                token_id=tid,
            )
            out.append(span.as_dict())
            continue
        piece_s = str(piece)
        # HuggingFace / SentencePiece often uses 'Ġ' or '▁' for leading space.
        normalized = piece_s.replace("Ġ", " ").replace("▁", " ")
        needle = normalized.encode("utf-8")
        # Search from cursor; allow one soft skip over whitespace mismatch.
        pos = raw.find(needle, cursor)
        if pos < 0:
            # Try stripped form (piece may include only partial whitespace).
            stripped = normalized.lstrip()
            if stripped and stripped != normalized:
                needle2 = stripped.encode("utf-8")
                pos = raw.find(needle2, cursor)
                if pos >= 0:
                    needle = needle2
                    normalized = stripped
        if pos < 0:
            # Unmatched piece: zero-width mark at cursor (caller can score coverage).
            span = TokenByteSpan(
                token_index=i,
                piece=piece_s,
                byte_start=cursor,
                byte_end=cursor,
                side=side,
                token_id=tid,
            )
            d = span.as_dict()
            d["matched"] = False
            out.append(d)
            continue
        end = pos + len(needle)
        span = TokenByteSpan(
            token_index=i,
            piece=piece_s,
            byte_start=pos,
            byte_end=end,
            side=side,
            token_id=tid,
        )
        d = span.as_dict()
        d["matched"] = True
        out.append(d)
        cursor = end
    return out


def glm_token_to_span_map(
    surface: SharedSurface | str,
    pieces: Sequence[str],
    *,
    token_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """GLM tokenizer pieces → shared-surface byte spans."""

    return map_token_pieces_to_byte_spans(
        surface, pieces, side="glm", token_ids=token_ids
    )


def dsv4f_token_to_span_map(
    surface: SharedSurface | str,
    pieces: Sequence[str],
    *,
    token_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """DSV4F tokenizer pieces → shared-surface byte spans."""

    return map_token_pieces_to_byte_spans(
        surface, pieces, side="dsv4f", token_ids=token_ids
    )


def pair_token_maps_via_byte_overlap(
    glm_spans: Sequence[Mapping[str, Any]],
    dsv4f_spans: Sequence[Mapping[str, Any]],
    *,
    min_overlap_bytes: int = 1,
) -> list[dict[str, Any]]:
    """Pair GLM and DSV4F token pieces by overlapping UTF-8 byte ranges only.

    Explicitly refuses token_id equality as a pairing signal.
    """

    # Hard refuse if either side requests token_id alignment.
    for side_name, spans in (("glm", glm_spans), ("dsv4f", dsv4f_spans)):
        for s in spans:
            if s.get("align_by") == "token_id":
                raise AlignerError(
                    f"{side_name}: align_by=token_id is forbidden across tokenizers"
                )

    pairs: list[dict[str, Any]] = []
    for i, gs in enumerate(glm_spans):
        g0, g1 = int(gs.get("byte_start", 0)), int(gs.get("byte_end", 0))
        if g1 <= g0:
            continue
        best_j = None
        best_overlap = 0
        for j, ds in enumerate(dsv4f_spans):
            d0, d1 = int(ds.get("byte_start", 0)), int(ds.get("byte_end", 0))
            if d1 <= d0:
                continue
            overlap = max(0, min(g1, d1) - max(g0, d0))
            if overlap > best_overlap:
                best_overlap = overlap
                best_j = j
        if best_j is not None and best_overlap >= min_overlap_bytes:
            ds = dsv4f_spans[best_j]
            pairs.append(
                {
                    "glm_token_index": i,
                    "dsv4f_token_index": best_j,
                    "overlap_bytes": best_overlap,
                    "glm_byte_range": [g0, g1],
                    "dsv4f_byte_range": [
                        int(ds.get("byte_start", 0)),
                        int(ds.get("byte_end", 0)),
                    ],
                    "method": "byte_span_overlap",
                    # Document that token_ids are NOT the join key even if present.
                    "joined_by_token_id": False,
                }
            )
    return pairs


def pool_indices_for_byte_span(
    token_spans: Sequence[Mapping[str, Any]],
    byte_start: int,
    byte_end: int,
) -> list[int]:
    """Indices of token pieces that overlap [byte_start, byte_end) for activation pooling."""

    out: list[int] = []
    for i, s in enumerate(token_spans):
        s0, s1 = int(s.get("byte_start", 0)), int(s.get("byte_end", 0))
        if s1 <= s0:
            continue
        if max(0, min(byte_end, s1) - max(byte_start, s0)) > 0:
            out.append(int(s.get("token_index", i)))
    return out


# ---------------------------------------------------------------------------
# Shared semantic anchors
# ---------------------------------------------------------------------------


_CLAIM_RE = re.compile(
    r"(?im)^(?:claim|theorem|lemma|proposition|statement)\s*[:.]?\s*(.+)$"
)
_PROOF_STEP_RE = re.compile(
    r"(?im)^(?:proof(?:\s+step)?|step\s+\d+|tactic)\s*[:.]?\s*(.+)$"
)
_SUBGOAL_RE = re.compile(
    r"(?im)^(?:subgoal|goal|todo|remaining)\s*[:.]?\s*(.+)$"
)
_ANSWER_RE = re.compile(
    r"(?im)^(?:answer|final answer|result|thus|therefore)\s*[:.]?\s*(.+)$"
)
_TOOL_RE = re.compile(
    r"(?im)^(?:tool|action)\s*[=:]\s*(\S+)(.*)$"
)
_CODE_FENCE_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)


def _anchor(
    *,
    kind: str,
    text: str,
    byte_start: int,
    byte_end: int,
    surface_id: str,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if kind not in SEMANTIC_ANCHOR_KINDS:
        raise AlignerError(f"unknown semantic anchor kind {kind!r}")
    return {
        "kind": kind,
        "text": text,
        "byte_start": byte_start,
        "byte_end": byte_end,
        "surface_id": surface_id,
        "meta": dict(meta or {}),
        "token_ids": None,
        "token_ids_forbidden_for_alignment": True,
    }


def extract_semantic_anchors(
    surface: SharedSurface | str,
    *,
    extra_hints: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Extract shared semantic anchors from a surface text.

    Heuristic + optional explicit hints.  Code regions use Python ``ast`` when
    a fenced python block parses; otherwise the whole fence is one region.
    """

    surf = surface if isinstance(surface, SharedSurface) else SharedSurface(text=surface)
    text = surf.text
    raw = surf.raw
    anchors: list[dict[str, Any]] = []

    def _add_match(kind: str, match: re.Match[str], group: int = 1) -> None:
        # Map char span of whole match to bytes.
        c0, c1 = match.start(group), match.end(group)
        br = char_range_to_byte_span(surf, c0, c1)
        anchors.append(
            _anchor(
                kind=kind,
                text=br["text"],
                byte_start=br["byte_start"],
                byte_end=br["byte_end"],
                surface_id=surf.surface_id,
            )
        )

    for m in _CLAIM_RE.finditer(text):
        _add_match("claim", m)
    for m in _PROOF_STEP_RE.finditer(text):
        _add_match("proof_step", m)
    for m in _SUBGOAL_RE.finditer(text):
        _add_match("subgoal", m)
    for m in _ANSWER_RE.finditer(text):
        _add_match("answer", m)
    for m in _TOOL_RE.finditer(text):
        c0, c1 = m.start(0), m.end(0)
        br = char_range_to_byte_span(surf, c0, c1)
        anchors.append(
            _anchor(
                kind="tool_action",
                text=br["text"],
                byte_start=br["byte_start"],
                byte_end=br["byte_end"],
                surface_id=surf.surface_id,
                meta={"tool_name": m.group(1)},
            )
        )

    # Formal action cues (apply / exact / intro / rw …).
    formal_re = re.compile(
        r"(?im)\b((?:apply|exact|intro|rw|simp|refine|constructor|cases)\b[^\n]*)"
    )
    for m in formal_re.finditer(text):
        c0, c1 = m.start(1), m.end(1)
        br = char_range_to_byte_span(surf, c0, c1)
        anchors.append(
            _anchor(
                kind="formal_action",
                text=br["text"],
                byte_start=br["byte_start"],
                byte_end=br["byte_end"],
                surface_id=surf.surface_id,
            )
        )

    # Code fences → AST regions when possible.
    for m in _CODE_FENCE_RE.finditer(text):
        block = m.group(1)
        c0, c1 = m.start(1), m.end(1)
        br = char_range_to_byte_span(surf, c0, c1)
        regions: list[dict[str, Any]] = []
        try:
            tree = ast.parse(block)
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    # lineno is 1-based within block.
                    lines = block.splitlines(keepends=True)
                    start_line = max(int(node.lineno) - 1, 0)
                    end_line = int(getattr(node, "end_lineno", node.lineno) or node.lineno)
                    char_off = sum(len(lines[i]) for i in range(start_line))
                    char_end = sum(len(lines[i]) for i in range(end_line))
                    # Offset into full surface.
                    abs_c0 = c0 + char_off
                    abs_c1 = c0 + char_end
                    sub = char_range_to_byte_span(surf, abs_c0, abs_c1)
                    regions.append(
                        _anchor(
                            kind="code_ast_region",
                            text=sub["text"],
                            byte_start=sub["byte_start"],
                            byte_end=sub["byte_end"],
                            surface_id=surf.surface_id,
                            meta={
                                "node": type(node).__name__,
                                "name": getattr(node, "name", None),
                            },
                        )
                    )
        except SyntaxError:
            regions = []
        if regions:
            anchors.extend(regions)
        else:
            anchors.append(
                _anchor(
                    kind="code_ast_region",
                    text=br["text"],
                    byte_start=br["byte_start"],
                    byte_end=br["byte_end"],
                    surface_id=surf.surface_id,
                    meta={"node": "fence", "parsed": False},
                )
            )

    # Explicit hints (e.g. known answer string, claim list from corpus).
    if extra_hints:
        for kind, texts in extra_hints.items():
            if kind not in SEMANTIC_ANCHOR_KINDS:
                continue
            for t in texts:
                loc = utf8_byte_span(surf, str(t))
                if loc["found"]:
                    anchors.append(
                        _anchor(
                            kind=kind,
                            text=str(t),
                            byte_start=loc["byte_start"],
                            byte_end=loc["byte_end"],
                            surface_id=surf.surface_id,
                            meta={"from_hint": True},
                        )
                    )

    # Stable order by byte position then kind.
    anchors.sort(key=lambda a: (a["byte_start"], a["byte_end"], a["kind"]))
    return anchors


def align_semantic_anchors(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Align semantic anchors by kind + normalized text (never token ids)."""

    assert_no_token_id_alignment(left, right)
    used: set[int] = set()
    out: list[dict[str, Any]] = []
    for i, la in enumerate(left):
        lkind = str(la.get("kind", ""))
        ltext = _norm_text(str(la.get("text", "")))
        if not lkind or not ltext:
            continue
        best_j = None
        best_score = 0.0
        for j, ra in enumerate(right):
            if j in used:
                continue
            if str(ra.get("kind", "")) != lkind:
                continue
            rtext = _norm_text(str(ra.get("text", "")))
            if not rtext:
                continue
            if ltext == rtext:
                score = 1.0
            elif ltext in rtext or rtext in ltext:
                score = 0.8
            else:
                lt, rt = set(ltext.split()), set(rtext.split())
                score = (len(lt & rt) / len(lt | rt)) if lt and rt else 0.0
                score *= 0.9  # soft
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is not None and best_score >= min_score:
            used.add(best_j)
            lb = _span_bytes(la)
            rb = _span_bytes(right[best_j])
            out.append(
                {
                    "left_index": i,
                    "right_index": best_j,
                    "score": float(best_score),
                    "method": "semantic_anchor",
                    "kind": lkind,
                    "left_byte_range": [lb[0], lb[1]],
                    "right_byte_range": [rb[0], rb[1]],
                }
            )
    return out


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
    *,
    surface_text: str | None = None,
) -> dict[str, Any]:
    """Full alignment report for two side dicts or two trace documents.

    Optional ``surface_text`` enables shared-surface semantic anchors and
    token-piece→byte-span maps when sides provide ``token_pieces``.
    """

    # Accept either a full trace or a side blob.
    if "sides" in left or "sides" in right:
        raise AlignerError("pass side dicts or span lists; use align_traces for full traces")

    l_spans = list(left.get("decoded_spans") or [])
    r_spans = list(right.get("decoded_spans") or [])
    l_actions = list(left.get("formal_actions") or [])
    r_actions = list(right.get("formal_actions") or [])
    l_tools = list(left.get("tool_events") or [])
    r_tools = list(right.get("tool_events") or [])
    l_anchors = list(left.get("semantic_anchors") or [])
    r_anchors = list(right.get("semantic_anchors") or [])

    assert_no_token_id_alignment(l_spans, r_spans)

    span_align = align_decoded_spans(l_spans, r_spans)
    byte_align = align_byte_ranges(l_spans, r_spans, require_shared_surface=True)
    action_align = align_formal_actions(l_actions, r_actions)
    tool_align = align_tool_events(l_tools, r_tools)

    # Shared-surface path: extract anchors if surface given and sides omit them.
    token_pair_report: dict[str, Any] | None = None
    if surface_text is not None:
        surface = SharedSurface(text=surface_text, surface_id="shared")
        if not l_anchors:
            l_anchors = extract_semantic_anchors(surface)
        if not r_anchors:
            # Same surface → same anchors; still run aligner for API symmetry.
            r_anchors = extract_semantic_anchors(surface)
        l_pieces = list(left.get("token_pieces") or [])
        r_pieces = list(right.get("token_pieces") or [])
        if l_pieces and r_pieces:
            glm_map = glm_token_to_span_map(
                surface, l_pieces, token_ids=left.get("token_ids")
            )
            dsv_map = dsv4f_token_to_span_map(
                surface, r_pieces, token_ids=right.get("token_ids")
            )
            token_pair_report = {
                "glm_token_to_span": glm_map,
                "dsv4f_token_to_span": dsv_map,
                "pairs": pair_token_maps_via_byte_overlap(glm_map, dsv_map),
                "joined_by": "utf8_byte_span_overlap",
                "joined_by_token_id": False,
            }

    anchor_align = align_semantic_anchors(l_anchors, r_anchors)

    return {
        "method_policy": {
            "allowed": [
                "decoded_spans",
                "byte_ranges",
                "formal_actions",
                "tool_events",
                "semantic_anchors",
                "token_piece_to_byte_span",
            ],
            "forbidden": ["token_ids", "vocab_index_match", "token_id_to_token_id"],
        },
        "decoded_spans": span_align,
        "byte_ranges": byte_align,
        "formal_actions": action_align,
        "tool_events": tool_align,
        "semantic_anchors": anchor_align,
        "token_byte_maps": token_pair_report,
        "summary": {
            "n_span_alignments": len(span_align),
            "n_byte_alignments": len(byte_align),
            "n_action_alignments": len(action_align),
            "n_tool_alignments": len(tool_align),
            "n_anchor_alignments": len(anchor_align),
            "n_token_byte_pairs": (
                len(token_pair_report["pairs"]) if token_pair_report else 0
            ),
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
