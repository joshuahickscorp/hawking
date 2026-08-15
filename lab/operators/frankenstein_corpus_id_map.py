#!/usr/bin/env python3.12
"""Reconcile GLM × DSV4F sequence IDs against the frozen PROTO_FRANKENSTEIN_V0 corpus.

Canonical ladder corpora live under::

    workspace/campaign/evidence/models/frankenstein/corpus/
        PROTO_FRANKENSTEIN_V0_L0_CORPUS.jsonl   # 32 sequences
        PROTO_FRANKENSTEIN_V0_L1_CORPUS.jsonl   # 128 sequences

GLM teacher-forced capture freezes the same rows under ``FROZEN_CORPUS_L{0,1}.json``
with ``example_id`` values like ``pfv0:…``.

The first DSV4F fullseq capture lane instead baked a short synthetic prompt table
(``v0_math_01``, ``v0_code_01``, …) into ``gravity_deepseek_v4_fullseq_capture``.
Those IDs are **not** aliases of the frozen V0 rows — content hashes prove zero
overlap. This module:

1. Defines the **content key** used for cross-side identity:
   ``sha256(utf-8(prompt_or_surface_text))`` (matches GLM ``prompt_text_sha256``).
2. Proves GLM frozen ↔ PROTO identity by example_id **and** content key.
3. Proves legacy DSV4F synthetic IDs have **zero** content-key intersection.
4. Emits a sealed reconciliation receipt for the correspondence pipeline.
5. After DSV4F re-capture loads the frozen corpus, the same check proves
   identity by content hash (not merely label equality).

Honesty: never invents a mapping between unequal content. A label-only "map"
from ``v0_math_01`` → some ``pfv0:…`` is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = (
    REPO_ROOT / "workspace" / "campaign" / "evidence" / "models" / "frankenstein"
)
CORPUS_DIR = EVIDENCE_ROOT / "corpus"
CARTOGRAPHY_DIR = EVIDENCE_ROOT / "cartography"
DEFAULT_PROTO_L0 = CORPUS_DIR / "PROTO_FRANKENSTEIN_V0_L0_CORPUS.jsonl"
DEFAULT_PROTO_L1 = CORPUS_DIR / "PROTO_FRANKENSTEIN_V0_L1_CORPUS.jsonl"
DEFAULT_GLM_L0 = (
    EVIDENCE_ROOT
    / "teacher_forced"
    / "official_L0_stream_full_20260805T200728Z"
    / "FROZEN_CORPUS_L0.json"
)
DEFAULT_DSV4F_L0_TRACES = REPO_ROOT / "receipts" / "dsv4f_fullseq_capture_L0" / "traces"
DEFAULT_DSV4F_L1_TRACES = REPO_ROOT / "receipts" / "dsv4f_fullseq_capture_L1" / "traces"
DEFAULT_OUT = CARTOGRAPHY_DIR / "GLM_DSV4F_CORPUS_ID_RECONCILIATION.json"

SCHEMA = "hawking.frankenstein.corpus_id_reconciliation.v1"


class CorpusIdMapError(RuntimeError):
    """Reconciliation failed closed."""


def content_key(text: str) -> str:
    """Canonical sequence content identity: sha256 of UTF-8 surface/prompt text."""

    if not isinstance(text, str):
        raise CorpusIdMapError(f"content_key expects str, got {type(text).__name__}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    raw = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    encoded = raw.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        node = os.lstat(path)
        if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
            raise CorpusIdMapError(f"not a regular file: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def load_proto_corpus(path: Path | str) -> list[dict[str, Any]]:
    """Load PROTO_FRANKENSTEIN_V0 jsonl rows (canonical IDs + surface_text)."""

    p = Path(path)
    if not p.is_file():
        raise CorpusIdMapError(f"PROTO corpus missing: {p}")
    rows: list[dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusIdMapError(f"{p}:{lineno}: bad JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise CorpusIdMapError(f"{p}:{lineno}: row must be object")
            eid = row.get("example_id")
            text = row.get("surface_text") or row.get("prompt_text")
            if not eid or not isinstance(text, str) or not text:
                raise CorpusIdMapError(f"{p}:{lineno}: missing example_id/surface_text")
            ck = content_key(text)
            # Prefer recomputed surface hash for pairing; keep provenance content_hash.
            rows.append(
                {
                    "example_id": str(eid),
                    "membership": row.get("membership"),
                    "family": row.get("family") or row.get("method_family"),
                    "prompt_text": text,
                    "content_key": ck,
                    "source_content_hash": row.get("content_hash"),
                    "source": "proto_frankenstein_v0",
                    "path": str(p),
                }
            )
    if not rows:
        raise CorpusIdMapError(f"empty PROTO corpus: {p}")
    return rows


def load_glm_frozen_corpus(path: Path | str) -> list[dict[str, Any]]:
    """Load GLM FROZEN_CORPUS_L*.json sequences."""

    p = Path(path)
    if not p.is_file():
        raise CorpusIdMapError(f"GLM frozen corpus missing: {p}")
    with open(p, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict):
        raise CorpusIdMapError(f"GLM frozen root must be object: {p}")
    seqs = doc.get("sequences")
    if not isinstance(seqs, list) or not seqs:
        raise CorpusIdMapError(f"GLM frozen has no sequences: {p}")
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(seqs):
        if not isinstance(row, dict):
            raise CorpusIdMapError(f"GLM sequence {i} not an object")
        eid = row.get("example_id")
        text = row.get("prompt_text")
        if not eid or not isinstance(text, str) or not text:
            raise CorpusIdMapError(f"GLM sequence {i} missing example_id/prompt_text")
        ck = content_key(text)
        claimed = row.get("prompt_text_sha256")
        if claimed is not None and claimed != ck:
            raise CorpusIdMapError(
                f"GLM {eid}: prompt_text_sha256 mismatch "
                f"(claimed {claimed[:16]}… recomputed {ck[:16]}…)"
            )
        rows.append(
            {
                "example_id": str(eid),
                "membership": row.get("membership"),
                "family": row.get("domain") or row.get("family"),
                "prompt_text": text,
                "content_key": ck,
                "prompt_text_sha256": claimed or ck,
                "source": "glm_frozen_corpus",
                "path": str(p),
            }
        )
    return rows


def load_dsv4f_trace_corpus(traces_dir: Path | str) -> list[dict[str, Any]]:
    """Load example_id + prompt_text from DSV4F fullseq trace JSONs."""

    d = Path(traces_dir)
    if not d.is_dir():
        raise CorpusIdMapError(f"DSV4F traces dir missing: {d}")
    paths = sorted(d.glob("*.json"))
    if not paths:
        raise CorpusIdMapError(f"no DSV4F traces in {d}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
        if not isinstance(doc, dict):
            continue
        eid = doc.get("example_id") or path.stem
        text = doc.get("prompt_text")
        if not isinstance(text, str) or not text:
            # try decoded_spans
            spans = doc.get("decoded_spans") or []
            for span in spans:
                if isinstance(span, dict) and span.get("role") == "prompt":
                    text = span.get("text")
                    break
        if not isinstance(text, str) or not text:
            raise CorpusIdMapError(f"DSV4F trace {path.name} has no prompt_text")
        rows.append(
            {
                "example_id": str(eid),
                "membership": doc.get("membership"),
                "family": doc.get("method_family_choice") or doc.get("method_family"),
                "prompt_text": text,
                "content_key": content_key(text),
                "source": "dsv4f_fullseq_trace",
                "path": str(path),
            }
        )
    return rows


def _index_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        eid = str(row["example_id"])
        if eid in out:
            raise CorpusIdMapError(f"duplicate example_id {eid!r}")
        out[eid] = dict(row)
    return out


def _index_by_content_key(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row["content_key"]), []).append(dict(row))
    return out


def pair_by_content_key(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pair two corpora by content_key. Label-only matches do not count."""

    l_by_ck = _index_by_content_key(left)
    r_by_ck = _index_by_content_key(right)
    shared_keys = sorted(set(l_by_ck) & set(r_by_ck))
    pairs: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    for ck in shared_keys:
        lrows = l_by_ck[ck]
        rrows = r_by_ck[ck]
        if len(lrows) != 1 or len(rrows) != 1:
            collisions.append(
                {
                    "content_key": ck,
                    "left_ids": [r["example_id"] for r in lrows],
                    "right_ids": [r["example_id"] for r in rrows],
                }
            )
            continue
        l, r = lrows[0], rrows[0]
        pairs.append(
            {
                "content_key": ck,
                "left_example_id": l["example_id"],
                "right_example_id": r["example_id"],
                "ids_equal": l["example_id"] == r["example_id"],
                "left_membership": l.get("membership"),
                "right_membership": r.get("membership"),
                "prompt_preview": (l.get("prompt_text") or "")[:120],
            }
        )
    left_only = sorted(set(l_by_ck) - set(r_by_ck))
    right_only = sorted(set(r_by_ck) - set(l_by_ck))
    return {
        "n_left": len(left),
        "n_right": len(right),
        "n_shared_content_keys": len(shared_keys),
        "n_pairs": len(pairs),
        "n_collisions": len(collisions),
        "pairs": pairs,
        "collisions": collisions,
        "n_left_only_content_keys": len(left_only),
        "n_right_only_content_keys": len(right_only),
        "left_only_example_ids": [
            l_by_ck[ck][0]["example_id"] for ck in left_only if l_by_ck[ck]
        ],
        "right_only_example_ids": [
            r_by_ck[ck][0]["example_id"] for ck in right_only if r_by_ck[ck]
        ],
    }


def pair_by_example_id(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pair by example_id, then verify content_key equality per shared id."""

    l_by_id = _index_by_id(left)
    r_by_id = _index_by_id(right)
    shared = sorted(set(l_by_id) & set(r_by_id))
    content_ok: list[str] = []
    content_mismatch: list[dict[str, Any]] = []
    for eid in shared:
        l, r = l_by_id[eid], r_by_id[eid]
        if l["content_key"] == r["content_key"]:
            content_ok.append(eid)
        else:
            content_mismatch.append(
                {
                    "example_id": eid,
                    "left_content_key": l["content_key"],
                    "right_content_key": r["content_key"],
                }
            )
    return {
        "n_left": len(left),
        "n_right": len(right),
        "n_shared_ids": len(shared),
        "n_content_match": len(content_ok),
        "n_content_mismatch": len(content_mismatch),
        "shared_ids": shared,
        "content_match_ids": content_ok,
        "content_mismatches": content_mismatch,
        "left_only_ids": sorted(set(l_by_id) - set(r_by_id)),
        "right_only_ids": sorted(set(r_by_id) - set(l_by_id)),
        "identical_id_and_content": (
            len(shared) == len(left) == len(right)
            and len(content_ok) == len(shared)
            and not content_mismatch
        ),
    }


def build_normalization_map(
    *,
    proto: Sequence[Mapping[str, Any]],
    glm: Sequence[Mapping[str, Any]] | None = None,
    dsv4f: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build id normalization tables. Only content-proven pairs are mapped.

    Returns maps:
      - ``glm_to_canonical``: glm example_id → proto example_id (when content matches)
      - ``dsv4f_to_canonical``: dsv4f example_id → proto example_id (when content matches)
      - ``canonical_ids``: ordered proto example_ids
    """

    proto_by_ck = _index_by_content_key(proto)
    canonical_ids = [str(r["example_id"]) for r in proto]
    glm_map: dict[str, str] = {}
    dsv_map: dict[str, str] = {}
    glm_unmapped: list[str] = []
    dsv_unmapped: list[str] = []

    if glm is not None:
        for row in glm:
            ck = str(row["content_key"])
            hits = proto_by_ck.get(ck) or []
            if len(hits) == 1:
                glm_map[str(row["example_id"])] = str(hits[0]["example_id"])
            else:
                glm_unmapped.append(str(row["example_id"]))
    if dsv4f is not None:
        for row in dsv4f:
            ck = str(row["content_key"])
            hits = proto_by_ck.get(ck) or []
            if len(hits) == 1:
                dsv_map[str(row["example_id"])] = str(hits[0]["example_id"])
            else:
                dsv_unmapped.append(str(row["example_id"]))

    return {
        "canonical_source": "PROTO_FRANKENSTEIN_V0",
        "content_key_definition": "sha256(utf-8(prompt_or_surface_text))",
        "canonical_ids": canonical_ids,
        "glm_to_canonical": glm_map,
        "dsv4f_to_canonical": dsv_map,
        "glm_unmapped_example_ids": glm_unmapped,
        "dsv4f_unmapped_example_ids": dsv_unmapped,
        "note": (
            "Maps only include content-key proven pairs. "
            "Legacy DSV4F v0_math_*/v0_code_* synthetic prompts are deliberately "
            "unmapped — re-capture with --corpus-mode frozen to get pfv0:* IDs."
        ),
    }


def reconcile(
    *,
    proto_path: Path | str = DEFAULT_PROTO_L0,
    glm_path: Path | str | None = DEFAULT_GLM_L0,
    dsv4f_traces: Path | str | None = DEFAULT_DSV4F_L0_TRACES,
    ladder: str = "L0",
) -> dict[str, Any]:
    """Full GLM × PROTO × DSV4F reconciliation report (unsealed body)."""

    proto = load_proto_corpus(proto_path)
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "name": "GLM_DSV4F_CORPUS_ID_RECONCILIATION",
        "recorded_at": _utc_now(),
        "ladder": ladder,
        "content_key_definition": "sha256(utf-8(prompt_or_surface_text))",
        "canonical": {
            "path": str(proto_path),
            "n": len(proto),
            "example_ids": [r["example_id"] for r in proto],
            "id_scheme": "pfv0:*",
        },
        "fabricated": False,
    }

    glm_rows = None
    if glm_path is not None and Path(glm_path).is_file():
        glm_rows = load_glm_frozen_corpus(glm_path)
        id_pair = pair_by_example_id(proto, glm_rows)
        ck_pair = pair_by_content_key(proto, glm_rows)
        body["glm"] = {
            "path": str(glm_path),
            "n": len(glm_rows),
            "id_scheme": "pfv0:*",
            "by_example_id": id_pair,
            "by_content_key": {
                "n_shared_content_keys": ck_pair["n_shared_content_keys"],
                "n_pairs": ck_pair["n_pairs"],
                "n_collisions": ck_pair["n_collisions"],
                "all_pairs_ids_equal": all(p["ids_equal"] for p in ck_pair["pairs"]),
            },
            "identical_to_canonical": bool(id_pair["identical_id_and_content"]),
            "status": (
                "ALIGNED"
                if id_pair["identical_id_and_content"]
                else "MISALIGNED"
            ),
        }
    else:
        body["glm"] = {
            "path": str(glm_path) if glm_path else None,
            "status": "ABSENT",
            "identical_to_canonical": False,
        }

    dsv_rows = None
    if dsv4f_traces is not None and Path(dsv4f_traces).is_dir():
        dsv_rows = load_dsv4f_trace_corpus(dsv4f_traces)
        id_pair = pair_by_example_id(proto, dsv_rows)
        ck_pair = pair_by_content_key(proto, dsv_rows)
        # Sample legacy IDs for the receipt.
        sample_ids = [r["example_id"] for r in dsv_rows[:8]]
        body["dsv4f"] = {
            "path": str(dsv4f_traces),
            "n": len(dsv_rows),
            "id_scheme_observed": (
                "v0_*_synthetic"
                if any(str(r["example_id"]).startswith("v0_") for r in dsv_rows)
                else "mixed_or_pfv0"
            ),
            "sample_example_ids": sample_ids,
            "by_example_id": {
                "n_shared_ids": id_pair["n_shared_ids"],
                "n_content_match": id_pair["n_content_match"],
                "shared_ids": id_pair["shared_ids"][:16],
            },
            "by_content_key": {
                "n_shared_content_keys": ck_pair["n_shared_content_keys"],
                "n_pairs": ck_pair["n_pairs"],
                "pairs": ck_pair["pairs"][:16],
            },
            "identical_to_canonical": bool(
                id_pair["identical_id_and_content"] and ck_pair["n_pairs"] == len(proto)
            ),
            "status": (
                "ALIGNED"
                if (
                    id_pair["identical_id_and_content"]
                    and ck_pair["n_pairs"] == len(proto)
                )
                else (
                    "CONTENT_ALIGNED_LABEL_DRIFT"
                    if ck_pair["n_pairs"] == len(proto) == len(dsv_rows)
                    else "MISALIGNED_SYNTHETIC_OR_FOREIGN"
                )
            ),
            "blocker": (
                None
                if ck_pair["n_pairs"] > 0
                else (
                    "DSV4F fullseq capture used a hard-coded short synthetic prompt "
                    "table (v0_math_*/v0_code_*/…) that shares zero content_keys with "
                    "PROTO_FRANKENSTEIN_V0 / GLM frozen corpus. Fix: re-run "
                    "gravity_deepseek_v4_fullseq_capture with --corpus-mode frozen "
                    "(loads PROTO jsonl; emits pfv0:* example_ids)."
                )
            ),
        }
    else:
        body["dsv4f"] = {
            "path": str(dsv4f_traces) if dsv4f_traces else None,
            "status": "ABSENT",
            "identical_to_canonical": False,
        }

    body["normalization_map"] = build_normalization_map(
        proto=proto, glm=glm_rows, dsv4f=dsv_rows
    )

    glm_ok = bool((body.get("glm") or {}).get("identical_to_canonical"))
    dsv_ok = bool((body.get("dsv4f") or {}).get("identical_to_canonical"))
    dsv_content = (
        (body.get("dsv4f") or {}).get("by_content_key", {}).get("n_shared_content_keys")
        or 0
    )
    if glm_ok and dsv_ok:
        status = "FULLY_ALIGNED"
    elif glm_ok and dsv_content == 0:
        status = "GLM_ALIGNED_DSV4F_NEEDS_RECAPTURE"
    elif glm_ok:
        status = "GLM_ALIGNED_DSV4F_PARTIAL"
    else:
        status = "BLOCKED"
    body["status"] = status
    body["correspondence_ready"] = bool(glm_ok and dsv_ok)
    body["claim_boundary"] = {
        "identity_proven_by": "content_key=sha256(utf-8(text))",
        "label_only_mapping_refused": True,
        "glm_matches_proto": glm_ok,
        "dsv4f_matches_proto": dsv_ok,
        "legacy_dsv4f_synthetic_corpus": (
            (body.get("dsv4f") or {}).get("status")
            == "MISALIGNED_SYNTHETIC_OR_FOREIGN"
        ),
    }
    return body


def emit_reconciliation(
    *,
    proto_path: Path | str = DEFAULT_PROTO_L0,
    glm_path: Path | str | None = DEFAULT_GLM_L0,
    dsv4f_traces: Path | str | None = DEFAULT_DSV4F_L0_TRACES,
    ladder: str = "L0",
    out_path: Path | str | None = DEFAULT_OUT,
    write: bool = True,
) -> dict[str, Any]:
    """Seal + optionally write the reconciliation receipt."""

    body = reconcile(
        proto_path=proto_path,
        glm_path=glm_path,
        dsv4f_traces=dsv4f_traces,
        ladder=ladder,
    )
    doc = seal(body)
    verify(doc, label="corpus id reconciliation")
    if write and out_path is not None:
        path = Path(out_path)
        _atomic_write_json(path, doc)
        return {**doc, "_written_path": str(path)}
    return doc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prove GLM×DSV4F sequence identity via content hashes"
    )
    parser.add_argument("--ladder", default="L0", choices=("L0", "L1"))
    parser.add_argument("--proto", type=Path, default=None)
    parser.add_argument("--glm-frozen", type=Path, default=None)
    parser.add_argument("--dsv4f-traces", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.ladder == "L1":
        proto = args.proto or DEFAULT_PROTO_L1
        dsv = args.dsv4f_traces or DEFAULT_DSV4F_L1_TRACES
        glm = args.glm_frozen  # L1 GLM freeze may not exist yet
    else:
        proto = args.proto or DEFAULT_PROTO_L0
        dsv = args.dsv4f_traces or DEFAULT_DSV4F_L0_TRACES
        glm = args.glm_frozen or DEFAULT_GLM_L0

    doc = emit_reconciliation(
        proto_path=proto,
        glm_path=glm,
        dsv4f_traces=dsv,
        ladder=args.ladder,
        out_path=args.out,
        write=not args.no_write,
    )
    print(
        json.dumps(
            {
                "status": doc["status"],
                "correspondence_ready": doc["correspondence_ready"],
                "glm": (doc.get("glm") or {}).get("status"),
                "dsv4f": (doc.get("dsv4f") or {}).get("status"),
                "glm_identical": (doc.get("glm") or {}).get("identical_to_canonical"),
                "dsv4f_content_pairs": (doc.get("dsv4f") or {})
                .get("by_content_key", {})
                .get("n_pairs"),
                "normalization_glm_mapped": len(
                    (doc.get("normalization_map") or {}).get("glm_to_canonical") or {}
                ),
                "normalization_dsv4f_mapped": len(
                    (doc.get("normalization_map") or {}).get("dsv4f_to_canonical") or {}
                ),
                "seal_sha256": doc.get("seal_sha256"),
                "path": doc.get("_written_path"),
            },
            indent=2,
        )
    )
    return 0 if doc.get("glm", {}).get("identical_to_canonical") else 2


if __name__ == "__main__":
    raise SystemExit(main())
