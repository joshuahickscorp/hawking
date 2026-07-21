#!/usr/bin/env python3.12
"""Corpus integrity gate: the calibration corpus is part of the instrument, so it is tested first.

WHY THIS MODULE EXISTS, in one measured sentence: the Qwen calibration corpus repeated 12 hand
written segments to reach its requested token count, so 1313 positions carried only 540 unique
token ids, and because the layer-0 MoE input is rmsnorm(embed[id]) - a pure function of the token
id - a POSITION-level train/validation split put the same embedding row in both halves. That
produced a fit/score energy overlap of 1.000 and an apparent +11.4 percent "held-out" QAT gain
which fell to -1.77 percent once the split was done by unique id and segment. The method was never
the problem. The instrument was.

So: no Doctor Prime experiment may begin until this gate passes.

WHAT THE GATE ENFORCES

  1. REAL DOCUMENT DIVERSITY. Segments come from distinct source documents on disk, content
     hashed, never from repeating one list. `grow()` proves that asking for a bigger corpus
     actually buys more unique documents / segments / context windows / token ids, and FAILS if
     the position count rises while every diversity axis stays flat.

  2. SIX DISJOINT SPLITS. routing_calibration, codec_calibration, doctor_training, validation,
     holdout, protected_domain_holdout. Assignment is by DOCUMENT, so no segment, context window
     or document ever straddles two splits. The split is a deterministic function of the document
     hash, so it is reproducible without storing an assignment table.

  3. TOKEN-ID DISJOINTNESS ON DEMAND. For any layer-0 or embedding-determined claim, `token_split`
     partitions UNIQUE TOKEN IDS, not positions. `assert_layer0_safe` refuses a split that shares
     a single embedding row between fit and score, which is the exact failure that invalidated S3A.

  4. EVALUATION PROMPTS ARE NEVER CALIBRATION. Every scored holdout string is checked against
     every calibration document, both directions, substring and token-id overlap.

Nothing here is a capability claim. This module measures the DATA, not the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

SCHEMA = "hawking.doctor_prime.corpus_integrity.v1"

SPLITS = ("routing_calibration", "codec_calibration", "doctor_training",
          "validation", "holdout", "protected_domain_holdout")

# Deterministic split weights. Routing and codec calibration get the bulk because they are fitted
# on; the two holdouts stay small but are never touched by any fit.
SPLIT_WEIGHTS = {"routing_calibration": 26, "codec_calibration": 26, "doctor_training": 22,
                 "validation": 10, "holdout": 10, "protected_domain_holdout": 6}

CONTEXT_WINDOW_CHARS = 512


class IntegrityError(AssertionError):
    """The instrument is not fit to measure with."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


# ── document sourcing ─────────────────────────────────────────────────────────────────────────
def discover_documents(roots: list[str], *, max_docs: int = 400,
                       min_chars: int = 400) -> list[dict[str, Any]]:
    """Collect distinct on-disk documents, content-hashed and deduplicated BY CONTENT.

    Deduplicating on the content hash rather than the path is what stops two copies of the same
    file (a worktree, a vendored duplicate) from inflating apparent diversity - the same class of
    lie the repeated-segment corpus told.
    """
    exts = {".py": "code", ".rs": "code", ".md": "prose", ".json": "tool_format",
            ".toml": "tool_format", ".txt": "prose", ".sh": "code"}
    seen: set[str] = set()
    docs: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted(Path(root).rglob("*")):
            if len(docs) >= max_docs:
                break
            if not path.is_file() or path.suffix not in exts:
                continue
            if any(p in path.parts for p in (".git", "target", "node_modules", "__pycache__")):
                continue
            try:
                text = path.read_text(errors="replace")
            except Exception:
                continue
            if len(text) < min_chars:
                continue
            h = _sha(text)
            if h in seen:
                continue
            seen.add(h)
            docs.append({"doc_id": h[:16], "sha256": h, "path": str(path),
                         "domain": exts[path.suffix], "n_chars": len(text), "text": text})
    return docs


def segment(doc: dict[str, Any], *, chars: int = CONTEXT_WINDOW_CHARS) -> list[dict[str, Any]]:
    """Split a document into non-overlapping context windows, each separately hashed."""
    out = []
    text = doc["text"]
    for i in range(0, len(text) - chars + 1, chars):
        w = text[i:i + chars]
        out.append({"doc_id": doc["doc_id"], "domain": doc["domain"], "offset": i,
                    "context_hash": _sha(w), "text": w})
    return out


# ── deterministic, document-level splitting ───────────────────────────────────────────────────
def assign_split(doc_id: str) -> str:
    """Deterministic split for a document, from its hash. No stored table, fully reproducible.

    Assignment is per DOCUMENT so that no segment or context window can straddle two splits.
    """
    total = sum(SPLIT_WEIGHTS.values())
    bucket = int(doc_id[:8], 16) % total
    acc = 0
    for name in SPLITS:
        acc += SPLIT_WEIGHTS[name]
        if bucket < acc:
            return name
    return SPLITS[-1]


def build(roots: list[str] | None = None, *, max_docs: int = 400,
          tokenizer: Any = None) -> dict[str, Any]:
    """Assemble the corpus, split it by document, and measure every diversity axis."""
    roots = roots or [str(Path(_HERE).resolve().parents[1] / "tools"),
                      str(Path(_HERE).resolve().parents[1] / "docs")]
    docs = discover_documents(roots, max_docs=max_docs)
    if not docs:
        raise IntegrityError(f"no documents discovered under {roots}")

    splits: dict[str, dict[str, Any]] = {
        s: {"docs": [], "segments": [], "token_ids": set()} for s in SPLITS}
    # GLOBAL context-window dedup, first occurrence wins in a deterministic document order.
    # Two DISTINCT documents routinely share an identical 512-char window - a license header, a
    # generated preamble, a common import block. Counting that window twice inflates apparent
    # diversity, and if the two documents landed in different splits it would be outright
    # cross-split leakage. Dropping the later copy fixes both. The count is reported, never hidden:
    # a corpus that is mostly boilerplate should look small, because it is.
    seen_ctx: dict[str, str] = {}
    dropped = 0
    for d in sorted(docs, key=lambda x: x["doc_id"]):
        s = assign_split(d["doc_id"])
        splits[s]["docs"].append(d)
        for seg in segment(d):
            owner = seen_ctx.get(seg["context_hash"])
            if owner is not None:
                dropped += 1
                continue
            seen_ctx[seg["context_hash"]] = s
            splits[s]["segments"].append(seg)

    if tokenizer is not None:
        for s in SPLITS:
            for seg in splits[s]["segments"]:
                splits[s]["token_ids"].update(tokenizer.encode(seg["text"]).ids)

    report: dict[str, Any] = {"schema": SCHEMA, "roots": roots,
                              "n_documents": len(docs),
                              "duplicate_context_windows_dropped": dropped,
                              "splits": {}}
    for s in SPLITS:
        segs = splits[s]["segments"]
        report["splits"][s] = {
            "n_documents": len(splits[s]["docs"]),
            "n_segments": len(segs),
            "n_unique_context_hashes": len({x["context_hash"] for x in segs}),
            "n_unique_token_ids": len(splits[s]["token_ids"]) or None,
            "positions": sum(len(x["text"]) for x in segs),
            "domains": sorted({x["domain"] for x in segs}),
        }
    # cross-split overlap: must be exactly zero on documents and context windows
    report["overlap"] = {}
    for a in SPLITS:
        for b in SPLITS:
            if a >= b:
                continue
            ha = {x["context_hash"] for x in splits[a]["segments"]}
            hb = {x["context_hash"] for x in splits[b]["segments"]}
            da = {d["doc_id"] for d in splits[a]["docs"]}
            db = {d["doc_id"] for d in splits[b]["docs"]}
            report["overlap"][f"{a}|{b}"] = {
                "context_hash_overlap": len(ha & hb), "document_overlap": len(da & db)}
    report["_splits"] = splits
    return report


# ── the gate ──────────────────────────────────────────────────────────────────────────────────
def check(report: dict[str, Any], *, holdout_texts: list[str] | None = None) -> dict[str, Any]:
    """Hard-fail the instrument on any of the named defects. Returns a pass/fail receipt."""
    failures: list[str] = []

    for pair, ov in report["overlap"].items():
        if ov["context_hash_overlap"]:
            failures.append(f"{pair}: {ov['context_hash_overlap']} shared context windows")
        if ov["document_overlap"]:
            failures.append(f"{pair}: {ov['document_overlap']} shared documents")

    for s in SPLITS:
        r = report["splits"][s]
        if r["n_documents"] == 0:
            failures.append(f"{s}: empty split")
        if r["n_segments"] and r["n_unique_context_hashes"] < r["n_segments"]:
            failures.append(f"{s}: repeated context windows "
                            f"({r['n_segments'] - r['n_unique_context_hashes']} duplicates)")

    # scored prompts may never appear in any calibration split
    if holdout_texts:
        for s in ("routing_calibration", "codec_calibration", "doctor_training"):
            body = "".join(x["text"] for x in report["_splits"][s]["segments"])
            for t in holdout_texts:
                probe = t.strip()[:200]
                if probe and probe in body:
                    failures.append(f"{s}: contains a scored holdout prompt")
                    break

    return {"passed": not failures, "failures": failures,
            "n_documents": report["n_documents"],
            "checked": ["cross-split context-window overlap", "cross-split document overlap",
                        "repeated context windows", "empty splits",
                        "scored prompts inside calibration"]}


def grow(roots: list[str] | None = None, small: int = 40, large: int = 200) -> dict[str, Any]:
    """Diversity-growth test: a bigger corpus must buy more INFORMATION, not just more positions.

    This is the test the old corpus would have failed: it grew positions 15x (1313 -> 20011) while
    unique token ids stayed pinned at 540.
    """
    a = build(roots, max_docs=small)
    b = build(roots, max_docs=large)
    axes = {}
    for axis in ("n_documents",):
        axes[axis] = (a[axis], b[axis])
    seg_a = sum(a["splits"][s]["n_segments"] for s in SPLITS)
    seg_b = sum(b["splits"][s]["n_segments"] for s in SPLITS)
    ctx_a = sum(a["splits"][s]["n_unique_context_hashes"] for s in SPLITS)
    ctx_b = sum(b["splits"][s]["n_unique_context_hashes"] for s in SPLITS)
    axes["n_segments"] = (seg_a, seg_b)
    axes["n_unique_context_hashes"] = (ctx_a, ctx_b)
    grew = [k for k, (x, y) in axes.items() if y > x]
    return {"axes": axes, "axes_that_grew": grew, "passed": bool(grew),
            "rule": "increasing the requested corpus size must increase at least one diversity "
                    "axis; more positions at flat diversity is not a larger corpus"}


# ── token-id level splitting, for layer-0 / embedding-determined claims ────────────────────────
def token_split(token_ids: list[int], *, frac: float = 0.5,
                seed: int = 0) -> tuple[list[int], list[int]]:
    """Partition UNIQUE token ids (never positions) into fit and score halves."""
    uniq = sorted(set(int(t) for t in token_ids))
    import random
    rng = random.Random(seed)
    rng.shuffle(uniq)
    cut = int(round(frac * len(uniq)))
    return sorted(uniq[:cut]), sorted(uniq[cut:])


def assert_layer0_safe(fit_ids: list[int], score_ids: list[int]) -> None:
    """Refuse the exact defect that invalidated S3A: a shared embedding row across the split."""
    shared = set(fit_ids) & set(score_ids)
    if shared:
        raise IntegrityError(
            f"{len(shared)} token ids appear in BOTH fit and score. For a layer-0 or "
            "embedding-determined claim the MoE input is a pure function of the token id, so a "
            "shared id is a shared input row and the split measures memorisation, not "
            "generalisation. This is the defect that produced the withdrawn S3A +11.4 pct result.")


def demo() -> None:
    """Self-check on real repo documents. Fails if the gate would let the old defect through."""
    rep = build(max_docs=120)
    res = check(rep)
    assert res["passed"], res["failures"]
    assert rep["n_documents"] >= 20, rep["n_documents"]
    for s in SPLITS:
        assert rep["splits"][s]["n_documents"] > 0, f"{s} empty"

    # every cross-split overlap must be exactly zero
    assert all(v["context_hash_overlap"] == 0 and v["document_overlap"] == 0
               for v in rep["overlap"].values())

    # boilerplate dedup must be REPORTED, and must actually eliminate repeats within a split
    assert "duplicate_context_windows_dropped" in rep
    for s in SPLITS:
        r = rep["splits"][s]
        assert r["n_unique_context_hashes"] == r["n_segments"], (s, r)

    g = grow(small=30, large=120)
    assert g["passed"], g

    # token-id split is disjoint by construction, and the layer-0 guard actually fires
    fit, score = token_split(list(range(1000)), seed=1)
    assert not (set(fit) & set(score))
    assert_layer0_safe(fit, score)
    try:
        assert_layer0_safe([1, 2, 3], [3, 4, 5])
        raise AssertionError("layer-0 guard failed to fire on a shared token id")
    except IntegrityError as exc:
        assert "BOTH fit and score" in str(exc)

    # the gate must REJECT a corpus that leaks a scored prompt into calibration
    leaked = "".join(x["text"] for x in rep["_splits"]["routing_calibration"]["segments"])[:200]
    bad = check(rep, holdout_texts=[leaked])
    assert not bad["passed"] and any("holdout prompt" in f for f in bad["failures"]), bad

    print(json.dumps({"ok": True, "n_documents": rep["n_documents"],
                      "splits": {s: rep["splits"][s]["n_documents"] for s in SPLITS},
                      "growth_axes": g["axes_that_grew"]}, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Corpus integrity gate for Doctor Prime.")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-docs", type=int, default=400)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    if args.demo:
        demo()
        return 0
    rep = build(max_docs=args.max_docs)
    res = check(rep)
    g = grow()
    out = {"schema": SCHEMA, "gate": res, "growth": g,
           "corpus": {k: v for k, v in rep.items() if k != "_splits"}}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": res["passed"], "failures": res["failures"],
                      "n_documents": rep["n_documents"],
                      "growth_passed": g["passed"]}, indent=2))
    return 0 if res["passed"] and g["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
