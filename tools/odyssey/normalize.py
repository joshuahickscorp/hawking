"""Text normalization for content addressing and near-duplicate comparison."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(text: str) -> str:
    """Canonical form for exact-match and shingle extraction.

    NFC unicode, lower-case, collapse whitespace, strip edge whitespace.
    Punctuation is retained for exact content-address of the *body*; use
    :func:`normalize_for_shingles` for near-dup comparison.
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = s.lower()
    s = _WS.sub(" ", s).strip()
    return s


def normalize_for_shingles(text: str) -> str:
    """Stricter form used only for character-shingle near-dup: drop punctuation."""
    s = normalize_text(text)
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def extract_comparison_text(item: dict[str, Any]) -> str:
    """Pull the text a train/eval overlap check should compare.

    Preference order: explicit ``text``, ``prompt``, flattened chat ``messages``,
    then ``prompt_template`` with needle (haystack omitted — the needle is the
    identity of long-context tasks).
    """
    if not isinstance(item, dict):
        return normalize_text(str(item))
    if item.get("text"):
        return str(item["text"])
    if item.get("prompt"):
        return str(item["prompt"])
    messages = item.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for m in messages:
            if isinstance(m, dict) and m.get("content") is not None:
                parts.append(str(m["content"]))
            elif isinstance(m, str):
                parts.append(m)
        if parts:
            return "\n".join(parts)
    if item.get("prompt_template"):
        tmpl = str(item["prompt_template"])
        needle = item.get("needle") or ""
        # Do not expand the full haystack; compare on template + needle identity.
        return tmpl.replace("{haystack}", needle) if "{haystack}" in tmpl else tmpl
    # Last resort: join string-ish scalar fields (never nested dumps of expect lists alone).
    for key in ("id", "query", "input", "question"):
        if item.get(key) and isinstance(item[key], str):
            return str(item[key])
    return ""
