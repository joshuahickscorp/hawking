"""Train/eval contamination barrier.

A naming convention is not a barrier. This module loads evaluation text
(support-halo + T0 hidden memberships + public selection), builds exact and
near-duplicate indexes, and **rejects** any training item that overlaps.

The support-halo corpus is sealed; we verify its sha256 before use and never
modify it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.odyssey._paths import (
    EXPECTED_SUPPORT_HALO_CORPUS_SHA256,
    HIDDEN_COMMITMENT,
    HIDDEN_ITEMS,
    JACCARD_TRAIN_VS_EVAL,
    PUBLIC_SELECTION,
    ROOT,
    SHINGLE_SIZE,
    SUPPORT_HALO_CORPUS,
    SUPPORT_HALO_SEAL,
)
from tools.odyssey.dedup import char_shingles, content_sha256, jaccard
from tools.odyssey.normalize import extract_comparison_text


@dataclass
class EvalItem:
    source: str
    item_id: str
    text: str
    role: str  # eval | hidden | selection
    exact_hash: str
    shingles: frozenset[str]


@dataclass
class ContaminationHit:
    reason: str  # exact_match | near_duplicate
    eval_source: str
    eval_id: str
    jaccard: float | None = None
    train_hash: str | None = None
    eval_hash: str | None = None


@dataclass
class Barrier:
    """Mechanical train/eval barrier over a frozen eval index."""

    eval_items: list[EvalItem] = field(default_factory=list)
    exact_index: dict[str, list[EvalItem]] = field(default_factory=dict)
    corpus_sha256_ok: bool = False
    support_halo_sha256: str | None = None
    hidden_commitment_ok: bool | None = None
    sources_loaded: list[str] = field(default_factory=list)
    jaccard_threshold: float = JACCARD_TRAIN_VS_EVAL
    shingle_size: int = SHINGLE_SIZE

    def check(self, train_text: str) -> list[ContaminationHit]:
        hits: list[ContaminationHit] = []
        if not train_text or not train_text.strip():
            return hits
        th = content_sha256(train_text)
        for ev in self.exact_index.get(th, []):
            hits.append(
                ContaminationHit(
                    reason="exact_match",
                    eval_source=ev.source,
                    eval_id=ev.item_id,
                    jaccard=1.0,
                    train_hash=th,
                    eval_hash=ev.exact_hash,
                )
            )
        if hits:
            return hits  # exact is enough to reject; still report all exacts
        t_sh = char_shingles(train_text, self.shingle_size)
        for ev in self.eval_items:
            score = jaccard(t_sh, ev.shingles)
            if score >= self.jaccard_threshold:
                hits.append(
                    ContaminationHit(
                        reason="near_duplicate",
                        eval_source=ev.source,
                        eval_id=ev.item_id,
                        jaccard=score,
                        train_hash=th,
                        eval_hash=ev.exact_hash,
                    )
                )
        return hits

    def admits(self, train_text: str) -> bool:
        return not self.check(train_text)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not path.is_file():
        return items
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i}: {e}") from e
    return items


def _add_eval(
    barrier: Barrier,
    *,
    source: str,
    role: str,
    items: list[dict[str, Any]],
    id_key: str = "id",
) -> None:
    for obj in items:
        text = extract_comparison_text(obj)
        if not text.strip():
            continue
        eid = str(obj.get(id_key) or obj.get("item_id") or content_sha256(text)[:12])
        h = content_sha256(text)
        ev = EvalItem(
            source=source,
            item_id=eid,
            text=text,
            role=role,
            exact_hash=h,
            shingles=char_shingles(text, barrier.shingle_size),
        )
        barrier.eval_items.append(ev)
        barrier.exact_index.setdefault(h, []).append(ev)
    barrier.sources_loaded.append(source)


def verify_support_halo_seal(corpus_path: Path = SUPPORT_HALO_CORPUS) -> dict[str, Any]:
    """Verify sealed corpus hash; never mutates the file."""
    if not corpus_path.is_file():
        return {
            "ok": False,
            "reason": "corpus_missing",
            "path": str(corpus_path),
        }
    digest = _sha256_file(corpus_path)
    seal_expected = EXPECTED_SUPPORT_HALO_CORPUS_SHA256
    if SUPPORT_HALO_SEAL.is_file():
        seal = json.loads(SUPPORT_HALO_SEAL.read_text(encoding="utf-8"))
        seal_expected = seal.get("corpus_sha256", seal_expected)
    ok = digest == seal_expected == EXPECTED_SUPPORT_HALO_CORPUS_SHA256
    return {
        "ok": ok,
        "path": str(corpus_path.relative_to(ROOT)) if corpus_path.is_relative_to(ROOT) else str(corpus_path),
        "computed_sha256": digest,
        "expected_sha256": EXPECTED_SUPPORT_HALO_CORPUS_SHA256,
        "seal_sha256": seal_expected,
    }


def verify_hidden_commitment(
    items_path: Path = HIDDEN_ITEMS,
    commitment_path: Path = HIDDEN_COMMITMENT,
) -> dict[str, Any]:
    """Recompute T0-style commitment over canonical JSONL lines."""
    if not items_path.is_file() or not commitment_path.is_file():
        return {"ok": False, "reason": "hidden_memberships_missing"}
    lines = [ln for ln in items_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    objs = [json.loads(ln) for ln in lines]
    canonical = [
        json.dumps(o, sort_keys=True, separators=(",", ":")) for o in objs
    ]
    h = hashlib.sha256()
    for line in canonical:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    recomputed = h.hexdigest()
    committed = json.loads(commitment_path.read_text(encoding="utf-8"))
    match = recomputed == committed.get("commitment_sha256")
    return {
        "ok": match,
        "recomputed": recomputed,
        "committed": committed.get("commitment_sha256"),
        "n_hidden": len(objs),
    }


def build_barrier(
    *,
    include_support_halo: bool = True,
    include_hidden: bool = True,
    include_public_selection: bool = True,
    jaccard_threshold: float = JACCARD_TRAIN_VS_EVAL,
    require_support_halo_seal: bool = True,
) -> Barrier:
    barrier = Barrier(jaccard_threshold=jaccard_threshold)
    if include_support_halo:
        seal = verify_support_halo_seal()
        barrier.support_halo_sha256 = seal.get("computed_sha256")
        barrier.corpus_sha256_ok = bool(seal.get("ok"))
        if require_support_halo_seal and not barrier.corpus_sha256_ok:
            raise RuntimeError(
                f"support-halo corpus seal check failed: {json.dumps(seal)}"
            )
        items = _load_jsonl(SUPPORT_HALO_CORPUS)
        _add_eval(
            barrier,
            source="odyssey/evaluation/support_halo_corpus_v0.jsonl",
            role="eval",
            items=items,
        )
    if include_hidden:
        hc = verify_hidden_commitment()
        barrier.hidden_commitment_ok = hc.get("ok")
        if HIDDEN_ITEMS.is_file():
            items = _load_jsonl(HIDDEN_ITEMS)
            _add_eval(
                barrier,
                source="odyssey/evaluation/hidden/hidden_items.jsonl",
                role="hidden",
                items=items,
            )
    if include_public_selection and PUBLIC_SELECTION.is_file():
        items = _load_jsonl(PUBLIC_SELECTION)
        _add_eval(
            barrier,
            source="odyssey/t0/public_eval/selection_items.jsonl",
            role="selection",
            items=items,
        )
    return barrier


def barrier_rules_document(barrier: Barrier) -> dict[str, Any]:
    return {
        "schema": "hawking.odyssey.contamination_barrier.v1",
        "rules": [
            "training items are compared on extract_comparison_text (prompt/text/messages)",
            "exact match: sha256(normalize_text(text)) equals any eval item hash → REJECT",
            f"near-duplicate: character shingles size={barrier.shingle_size}, "
            f"Jaccard ≥ {barrier.jaccard_threshold} vs any eval item → REJECT",
            "evaluation sources: support-halo (sealed), T0 hidden memberships, public selection",
            "a naming convention is not a barrier; this index is mechanical",
            "support-halo corpus and seal files are read-only; mutation is a contract violation",
        ],
        "jaccard_threshold_train_vs_eval": barrier.jaccard_threshold,
        "shingle_size": barrier.shingle_size,
        "sources_loaded": list(barrier.sources_loaded),
        "n_eval_items_indexed": len(barrier.eval_items),
        "support_halo_corpus_sha256": barrier.support_halo_sha256,
        "support_halo_seal_ok": barrier.corpus_sha256_ok,
        "hidden_commitment_ok": barrier.hidden_commitment_ok,
        "expected_support_halo_sha256": EXPECTED_SUPPORT_HALO_CORPUS_SHA256,
    }
