"""Exact and near-duplicate detection via content hashes and character shingles."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from tools.odyssey._paths import JACCARD_WITHIN_CORPUS_NEAR_DUP, SHINGLE_SIZE
from tools.odyssey.normalize import normalize_for_shingles, normalize_text


def content_sha256(text: str) -> str:
    """SHA-256 of UTF-8 normalized text (exact-match identity)."""
    body = normalize_text(text).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def char_shingles(text: str, size: int = SHINGLE_SIZE) -> frozenset[str]:
    s = normalize_for_shingles(text)
    if not s:
        return frozenset()
    if len(s) < size:
        return frozenset([s])
    return frozenset(s[i : i + size] for i in range(len(s) - size + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass(frozen=True)
class NearDupHit:
    other_index: int
    jaccard: float
    other_id: str | None = None


def find_near_duplicates(
    texts: list[str],
    *,
    threshold: float = JACCARD_WITHIN_CORPUS_NEAR_DUP,
    ids: list[str] | None = None,
    shingle_size: int = SHINGLE_SIZE,
) -> dict[int, list[NearDupHit]]:
    """Pairwise near-dup hits for indices i < j. O(n^2) — fine for fixtures and small sets."""
    shingles = [char_shingles(t, shingle_size) for t in texts]
    hits: dict[int, list[NearDupHit]] = {}
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            score = jaccard(shingles[i], shingles[j])
            if score >= threshold:
                oid = ids[j] if ids else None
                hits.setdefault(i, []).append(NearDupHit(j, score, oid))
                hits.setdefault(j, []).append(
                    NearDupHit(i, score, ids[i] if ids else None)
                )
    return hits


def exact_dedup_indices(texts: Iterable[str]) -> tuple[list[int], dict[str, list[int]]]:
    """Return keep-indices (first occurrence) and hash -> all indices map."""
    seen: dict[str, int] = {}
    groups: dict[str, list[int]] = {}
    keep: list[int] = []
    for i, t in enumerate(texts):
        h = content_sha256(t)
        groups.setdefault(h, []).append(i)
        if h not in seen:
            seen[h] = i
            keep.append(i)
    return keep, groups
