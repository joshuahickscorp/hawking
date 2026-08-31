"""DERIVED_FRESHNESS — semantic vs byte freshness for sidecar artifacts.

Codex rewrites the physical qualification queue rapidly. Every rewrite changes
the file sha even when nothing semantic moved. This module distinguishes:

    FRESH                    byte sha and semantic fingerprint both match
    STALE_FINGERPRINT_ONLY   byte sha differs, meaning is identical
    STALE_SEMANTIC           meaning differs; regeneration REQUIRED
    UNKNOWN                  the derived artifact recorded no provenance

`--check` exits non-zero only on STALE_SEMANTIC. STALE_FINGERPRINT_ONLY is
reported and exits 0 — that is the point. `--refresh` invokes each stale
producer's own build entry point and never rewrites another module's receipt
itself.

    python3 tools/future/freshness.py --check
    python3 tools/future/freshness.py --report
    python3 tools/future/freshness.py --refresh
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO


import argparse
import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from tools.future._common import RECEIPTS, git, sha256_file


RECEIPT = "DERIVED_FRESHNESS.json"
SCHEMA = "hawking.future.freshness.v1"
VERSION = 1
RECORDED_BY = "tools/future/freshness.py"

PINNED_DIR = RECEIPTS / "evidence"
OWN_RECEIPT = RECEIPT

FRESH = "FRESH"
STALE_FINGERPRINT_ONLY = "STALE_FINGERPRINT_ONLY"
STALE_SEMANTIC = "STALE_SEMANTIC"
UNKNOWN = "UNKNOWN"

# Queue meaning. Ordering, whitespace, timestamps and zero-valued status
# buckets are not in this set. Candidate count is derived from the rows.
QUEUE_SEMANTIC_FIELDS: tuple[str, ...] = (
    "candidate_id",
    "status",
    "affected_physical_region",
    "exact_mutation",
    "dependencies",
    "blocked_reason",
)

COSMETIC_KEYS = frozenset(
    {
        "recorded_at",
        "recorded_at_utc",
        "source_mtime_epoch",
        "source_mtime_utc",
        "seal_sha256",
        "mtime",
        "timestamp",
        "fingerprint",
    }
)
ZERO_BUCKET_KEYS = frozenset({"by_status", "by_classification", "by_model"})

QUEUE_REL = "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
REPAT_REL = "receipts/headless/ACCELERATOR_REPATRIATION_QUEUE.json"
FLASH_NX_REL = "receipts/headless/FLASH_COMPLETE_V0.nx.json"
FLASH_NR_REL = "receipts/headless/FLASH_COMPLETE_V2.nr.json"
FLASH_NEXT_REL = "receipts/headless/FLASH_NEXT_MACHINE.nx.json"
FLASH_META_REL = "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json"
SCOREBOARD_REL = "receipts/headless/ACCELERATOR_SCOREBOARD.json"

RECOVERY_PROBES: tuple[tuple[str, str], ...] = (
    (
        "receipts/future/CANDIDATE_STAGED_PLAN.json",
        "already records input.queue_sha256 + input.queue_fingerprint + candidate_count + by_status",
    ),
    (
        "tools/future/candidate_planner.py",
        "assert_plan_dramatically_smaller no longer uses a capped n<=40 convenience integer",
    ),
    (
        "tools/future/evidence_snapshot.py",
        "pinned hash-manifest of Codex receipts; EVIDENCE_SNAPSHOT.json is the manifest",
    ),
    (
        "tools/future/codex_ingest.py",
        "durable sha256 cursor over receipts/headless; change detection is bytes, not mtime",
    ),
    (
        "receipts/future/QUALIFICATION_PIPELINE.json",
        "sequences planner/preflight; does not record source sha/fingerprint",
    ),
    (
        "receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json",
        "audits Flash NX completeness against the queue; no source sha recorded",
    ),
    (
        "receipts/future/HCLI_FUTURE_WORKUNITS.json",
        "live_sources.loaded_from is a path, not a sha",
    ),
    (
        "receipts/future/TOURNAMENT_READINESS.json",
        "reads NX identity documents; no source sha recorded",
    ),
    (
        "receipts/future/EVIDENCE_SNAPSHOT.json",
        "captured[].sha256 is per-file byte provenance",
    ),
    (
        "receipts/future/FUTURE_SUBSTRATE_HANDOFF.json",
        "41-system inventory; this lane extends it, does not replace it",
    ),
    (
        "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
        "live frontier; stale_entries is a different probe, not derived-artifact freshness",
    ),
    (
        "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
        "Codex live queue (untracked on many worktrees); disk state is authority",
    ),
    (
        "receipts/future/evidence/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
        "pinned snapshot copy used when live headless is not visible",
    ),
)


@dataclass(frozen=True)
class RegistryEntry:
    derived: str
    sources: tuple[str, ...]
    producer: str
    kind: str
    recorded_sha_path: str | None = None


REGISTRY: tuple[RegistryEntry, ...] = (
    RegistryEntry(
        derived="CANDIDATE_STAGED_PLAN.json",
        sources=(QUEUE_REL,),
        producer="tools.future.candidate_planner:build",
        kind="queue",
        recorded_sha_path="input.queue_sha256",
    ),
    RegistryEntry(
        derived="QUALIFICATION_PIPELINE.json",
        sources=(QUEUE_REL, "receipts/future/CANDIDATE_STAGED_PLAN.json"),
        producer="tools.future.qualification_pipeline:build",
        kind="generic",
    ),
    RegistryEntry(
        derived="FLASH_NX_COMPLETENESS_AUDIT.json",
        sources=(QUEUE_REL, FLASH_NX_REL, FLASH_NR_REL, FLASH_NEXT_REL, FLASH_META_REL),
        producer="tools.future.flash_nx_audit:build",
        kind="generic",
    ),
    RegistryEntry(
        derived="HCLI_FUTURE_WORKUNITS.json",
        sources=(QUEUE_REL, REPAT_REL),
        producer="tools.future.workunit_species:build",
        kind="generic",
    ),
    RegistryEntry(
        derived="TOURNAMENT_READINESS.json",
        sources=(FLASH_NX_REL, SCOREBOARD_REL),
        producer="tools.future.tournament:build",
        kind="generic",
    ),
    RegistryEntry(
        derived="EVIDENCE_SNAPSHOT.json",
        sources=(),
        producer="tools.future.evidence_snapshot:build",
        kind="manifest",
        recorded_sha_path="captured.sha256",
    ),
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _dot_get(doc: Mapping[str, Any] | None, path: str | None) -> Any:
    if not path or not isinstance(doc, Mapping):
        return None
    cur: Any = doc
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _stable(value: Any) -> Any:
    """Canonical form: dict keys sorted, list order preserved except as callers sort."""
    if isinstance(value, Mapping):
        return {str(k): _stable(value[k]) for k in sorted(value, key=lambda x: str(x))}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    if isinstance(value, tuple):
        return [_stable(v) for v in value]
    return value


def _cosmetic_key(key: str) -> bool:
    if key in COSMETIC_KEYS:
        return True
    kl = key.lower()
    if kl.endswith("_utc") and "mtime" in kl:
        return True
    return False


def _zeroish(value: Any) -> bool:
    return value == 0 or value == 0.0


def _strip_cosmetic(node: Any, parent_key: str | None = None) -> Any:
    if isinstance(node, Mapping):
        out: dict[str, Any] = {}
        for key in sorted(node, key=lambda x: str(x)):
            if _cosmetic_key(str(key)):
                continue
            value = node[key]
            if str(key) in ZERO_BUCKET_KEYS and isinstance(value, Mapping):
                value = {ik: iv for ik, iv in value.items() if not _zeroish(iv)}
            out[str(key)] = _strip_cosmetic(value, str(key))
        return out
    if isinstance(node, list):
        items = [_strip_cosmetic(v) for v in node]
        if items and all(isinstance(x, dict) and "candidate_id" in x for x in items):
            items = sorted(items, key=lambda x: str(x.get("candidate_id")))
        return items
    return node


def looks_like_queue(doc: Any) -> bool:
    """Derive queue-ness from the document. Never from a candidate count."""
    if not isinstance(doc, Mapping):
        return False
    schema = str(doc.get("schema") or "")
    cands = doc.get("candidates")
    if "qualification_queue" in schema and isinstance(cands, list):
        return True
    if isinstance(cands, list) and any(
        isinstance(row, Mapping) and "candidate_id" in row for row in cands
    ):
        return True
    return False


def queue_identity_rows(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Semantic tuples, one per candidate, sorted by id. Count comes from the list."""
    rows: list[dict[str, Any]] = []
    for cand in doc.get("candidates") or []:
        if not isinstance(cand, Mapping):
            continue
        cid = cand.get("candidate_id")
        if cid is None:
            continue
        deps = cand.get("dependencies") or []
        if not isinstance(deps, list):
            deps = [deps]
        rows.append(
            {
                "candidate_id": cid,
                "status": cand.get("status"),
                "affected_physical_region": cand.get("affected_physical_region"),
                "exact_mutation": _stable(cand.get("exact_mutation")),
                "dependencies": sorted((_stable(d) for d in deps), key=lambda x: _canonical_dumps(x)),
                "blocked_reason": cand.get("blocked_reason"),
            }
        )
    rows.sort(key=lambda r: str(r["candidate_id"]))
    return rows


def semantic_fingerprint(doc: Any) -> str:
    """Fingerprint over meaning, not bytes.

    For a qualification queue that is the set of
    (candidate_id, status, affected_physical_region, exact_mutation,
    dependencies, blocked_reason). Ordering, whitespace, timestamps and
    zero-valued status buckets do not change it.
    """
    if looks_like_queue(doc):
        payload = queue_identity_rows(doc)
    else:
        payload = _strip_cosmetic(doc)
    return _sha256_bytes(_canonical_dumps(payload).encode())


def candidate_diff(
    old_doc: Mapping[str, Any] | None, new_doc: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Exactly what changed: added / removed / status-changed candidate ids.

    Both sides must be present. An UNKNOWN (no historical document) must not
    report the entire current queue as "added".
    """
    have_old = old_doc is not None and looks_like_queue(old_doc)
    have_new = new_doc is not None and looks_like_queue(new_doc)
    old_rows = {
        str(r["candidate_id"]): r for r in (queue_identity_rows(old_doc) if have_old else [])
    }
    new_rows = {
        str(r["candidate_id"]): r for r in (queue_identity_rows(new_doc) if have_new else [])
    }
    old_ids = set(old_rows)
    new_ids = set(new_rows)
    if not (have_old and have_new):
        return {
            "added_candidate_ids": [],
            "removed_candidate_ids": [],
            "status_changed_candidate_ids": [],
            "semantic_field_changed_candidate_ids": [],
            "recorded_candidate_count": len(old_ids) if have_old else None,
            "current_candidate_count": len(new_ids) if have_new else None,
        }
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    status_changed = sorted(
        cid
        for cid in sorted(old_ids & new_ids)
        if old_rows[cid].get("status") != new_rows[cid].get("status")
    )
    field_changed = sorted(
        cid
        for cid in sorted(old_ids & new_ids)
        if old_rows[cid] != new_rows[cid] and cid not in status_changed
    )
    return {
        "added_candidate_ids": added,
        "removed_candidate_ids": removed,
        "status_changed_candidate_ids": status_changed,
        "semantic_field_changed_candidate_ids": field_changed,
        "recorded_candidate_count": len(old_ids) if old_doc is not None and looks_like_queue(old_doc) else None,
        "current_candidate_count": len(new_ids) if new_doc is not None and looks_like_queue(new_doc) else None,
    }


def classify_artifact(
    *,
    recorded_sha256: str | None,
    current_sha256: str | None,
    recorded_fp: str | None,
    current_fp: str | None,
    recorded_doc: Mapping[str, Any] | None = None,
    current_doc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Four-way classification of a derived artifact against one source."""
    diff = candidate_diff(recorded_doc, current_doc)
    provenance_recorded = recorded_sha256 is not None or recorded_fp is not None or recorded_doc is not None
    current_present = current_sha256 is not None or current_doc is not None
    if not provenance_recorded:
        status = UNKNOWN
        reason = "derived artifact records no provenance to compare"
    elif not current_present:
        status = UNKNOWN
        reason = "current source is not visible from this worktree or the pinned snapshot"
    elif recorded_sha256 is not None and current_sha256 is not None and recorded_sha256 == current_sha256:
        status = FRESH
        reason = "byte sha matches; semantic identity follows"
    else:
        rec_fp = recorded_fp if recorded_fp is not None else (
            semantic_fingerprint(recorded_doc) if recorded_doc is not None else None
        )
        cur_fp = current_fp if current_fp is not None else (
            semantic_fingerprint(current_doc) if current_doc is not None else None
        )
        if rec_fp is not None and cur_fp is not None and rec_fp == cur_fp:
            status = STALE_FINGERPRINT_ONLY
            reason = "byte sha differs; semantic fingerprint identical — cosmetic rewrite"
        elif rec_fp is None or cur_fp is None:
            status = STALE_SEMANTIC
            reason = (
                "byte sha differs and a semantic fingerprint could not be produced "
                "for both sides; fail closed, regeneration required"
            )
        else:
            status = STALE_SEMANTIC
            reason = "semantic fingerprint differs; regeneration required"
    return {
        "status": status,
        "reason": reason,
        "recorded_sha256": recorded_sha256,
        "current_sha256": current_sha256,
        "recorded_semantic_fingerprint": recorded_fp
        if recorded_fp is not None
        else (semantic_fingerprint(recorded_doc) if recorded_doc is not None else None),
        "current_semantic_fingerprint": current_fp
        if current_fp is not None
        else (semantic_fingerprint(current_doc) if current_doc is not None else None),
        "byte_match": bool(
            recorded_sha256 is not None
            and current_sha256 is not None
            and recorded_sha256 == current_sha256
        ),
        "semantic_match": bool(
            (recorded_fp if recorded_fp is not None else (
                semantic_fingerprint(recorded_doc) if recorded_doc is not None else None
            ))
            == (current_fp if current_fp is not None else (
                semantic_fingerprint(current_doc) if current_doc is not None else None
            ))
            and (
                recorded_fp is not None
                or recorded_doc is not None
            )
            and (
                current_fp is not None
                or current_doc is not None
            )
        ),
        **diff,
    }


def classify_documents(
    old_doc: Mapping[str, Any],
    new_doc: Mapping[str, Any],
    *,
    old_raw: bytes | None = None,
    new_raw: bytes | None = None,
) -> dict[str, Any]:
    """Classify two source documents against each other (tests + selftest)."""
    old_bytes = old_raw if old_raw is not None else _canonical_dumps(old_doc).encode()
    new_bytes = new_raw if new_raw is not None else _canonical_dumps(new_doc).encode()
    return classify_artifact(
        recorded_sha256=_sha256_bytes(old_bytes),
        current_sha256=_sha256_bytes(new_bytes),
        recorded_fp=semantic_fingerprint(old_doc),
        current_fp=semantic_fingerprint(new_doc),
        recorded_doc=old_doc,
        current_doc=new_doc,
    )


def exit_code_for(body: Mapping[str, Any]) -> int:
    """`--check` exits 1 iff any registered artifact is STALE_SEMANTIC."""
    for row in body.get("classifications") or []:
        if isinstance(row, Mapping) and row.get("status") == STALE_SEMANTIC:
            return 1
    return 0


def refresh_stale(
    classifications: Sequence[Mapping[str, Any]],
    *,
    invoke: Callable[[str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Invoke producer build() only for STALE_SEMANTIC rows."""
    fn = invoke or invoke_producer
    out: list[dict[str, Any]] = []
    for row in classifications:
        if row.get("status") != STALE_SEMANTIC:
            continue
        producer = row.get("producer")
        if not isinstance(producer, str) or not producer:
            out.append(
                {
                    "derived": row.get("derived"),
                    "ok": False,
                    "error": "registry row has no producer",
                }
            )
            continue
        if producer.startswith("tools.future.freshness:"):
            continue
        result = dict(fn(producer))
        result["derived"] = row.get("derived")
        out.append(result)
    return out


def invoke_producer(producer: str) -> dict[str, Any]:
    """Call another module's build entry point. Never write its receipt ourselves."""
    mod_name, sep, func_name = producer.partition(":")
    if not sep or not func_name:
        return {"ok": False, "producer": producer, "error": "producer must be module:callable"}
    try:
        module = importlib.import_module(mod_name)
        fn = getattr(module, func_name)
        result = fn()
    except Exception as exc:
        return {
            "ok": False,
            "producer": producer,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    return {
        "ok": True,
        "producer": producer,
        "result": str(result) if result is not None else None,
    }


# ---------------------------------------------------------------------------
# Source resolution. Missing is a recorded path, never a crash.
# ---------------------------------------------------------------------------


def _search_roots() -> list[tuple[str, Path]]:
    """this worktree, then the git-common primary checkout, then sibling worktrees.

    First unique path wins, so the primary Codex checkout is tagged git_common
    rather than other_worktree. Sibling lanes hold sparse git copies, not the
    live uncommitted frontier.
    """
    rows: list[tuple[str, Path]] = [("this_worktree", REPO)]
    common = git("rev-parse", "--git-common-dir")
    if common:
        p = Path(common)
        if not p.is_absolute():
            p = (REPO / p).resolve()
        else:
            p = p.resolve()
        parent = p.parent if p.name == ".git" else p
        rows.append(("git_common", parent))
    blob = git("worktree", "list", "--porcelain")
    for line in blob.splitlines():
        if line.startswith("worktree "):
            rows.append(("other_worktree", Path(line.split(" ", 1)[1])))
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for kind, root in rows:
        try:
            key = str(root.resolve()) if root.exists() else str(root)
        except OSError:
            key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append((kind, root))
    return out


def _evidence_source_for(kind: str) -> str:
    if kind == "pinned_snapshot":
        return "pinned_snapshot"
    return "live_headless"


def _load_maybe(path: Path) -> tuple[dict[str, Any] | None, bytes | None]:
    try:
        raw = path.read_bytes()
    except OSError:
        return None, None
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None, raw
    if isinstance(obj, dict):
        return obj, raw
    return None, raw


def resolve_copies(rel: str) -> list[dict[str, Any]]:
    """Every visible copy of `rel`. Copes with live-present or live-absent."""
    copies: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, root in _search_roots():
        path = root / rel
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if not is_file:
            continue
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        sha = sha256_file(path)
        copies.append(
            {
                "kind": kind,
                "path": resolved,
                "rel": rel,
                "sha256": sha,
                "evidence_source": _evidence_source_for(kind),
            }
        )
    pinned = PINNED_DIR / Path(rel).name
    try:
        pinned_is = pinned.is_file()
    except OSError:
        pinned_is = False
    if pinned_is:
        try:
            resolved = str(pinned.resolve())
        except OSError:
            resolved = str(pinned)
        if resolved not in seen:
            copies.append(
                {
                    "kind": "pinned_snapshot",
                    "path": resolved,
                    "rel": f"receipts/future/evidence/{Path(rel).name}",
                    "sha256": sha256_file(pinned),
                    "evidence_source": "pinned_snapshot",
                }
            )
    return copies


def choose_current(copies: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Prefer a live Codex copy so a semantic queue rewrite is not silent.

    Rank: other_worktree / git_common (live frontier) > this_worktree > pin.
    """
    if not copies:
        return None

    def rank(copy: Mapping[str, Any]) -> int:
        kind = copy.get("kind")
        # Live Codex frontier (git-common primary) first. Sibling worktrees
        # are sparse git checkouts and must not beat an uncommitted live file.
        if kind == "git_common":
            return 0
        if kind == "this_worktree":
            return 1
        if kind == "other_worktree":
            return 2
        return 3

    return dict(sorted(copies, key=lambda c: (rank(c), str(c.get("path") or "")))[0])


def _copy_by_sha(copies: Sequence[Mapping[str, Any]], sha: str | None) -> dict[str, Any] | None:
    if not sha:
        return None
    for copy in copies:
        if copy.get("sha256") == sha:
            return dict(copy)
    return None


# ---------------------------------------------------------------------------
# Registry assessment
# ---------------------------------------------------------------------------


def _load_derived(name: str) -> dict[str, Any] | None:
    path = RECEIPTS / name
    try:
        if not path.is_file():
            return None
        doc = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _status_rank(status: str) -> int:
    return {
        STALE_SEMANTIC: 3,
        UNKNOWN: 2,
        STALE_FINGERPRINT_ONLY: 1,
        FRESH: 0,
    }.get(status, 2)


def _aggregate_status(statuses: Iterable[str]) -> str:
    best = FRESH
    best_rank = -1
    n = 0
    for status in statuses:
        n += 1
        rank = _status_rank(status)
        if rank > best_rank:
            best = status
            best_rank = rank
    if n == 0:
        return UNKNOWN
    return best


def _assess_one_source(
    rel: str,
    *,
    recorded_sha: str | None,
    recorded_doc_hint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    copies = resolve_copies(rel)
    current = choose_current(copies)
    historical = _copy_by_sha(copies, recorded_sha)
    current_doc = None
    current_raw = None
    current_sha = None
    current_path = None
    evidence_source = None
    if current is not None:
        current_path = current["path"]
        current_sha = current["sha256"]
        evidence_source = current["evidence_source"]
        current_doc, current_raw = _load_maybe(Path(current_path))
        del current_raw
    recorded_doc = recorded_doc_hint
    if historical is not None:
        recorded_doc, _ = _load_maybe(Path(historical["path"]))
        if recorded_sha is None:
            recorded_sha = historical.get("sha256")
    elif recorded_sha is not None and current_sha == recorded_sha:
        recorded_doc = current_doc
    recorded_fp = semantic_fingerprint(recorded_doc) if recorded_doc is not None else None
    current_fp = semantic_fingerprint(current_doc) if current_doc is not None else None
    result = classify_artifact(
        recorded_sha256=recorded_sha,
        current_sha256=current_sha,
        recorded_fp=recorded_fp,
        current_fp=current_fp,
        recorded_doc=recorded_doc,
        current_doc=current_doc,
    )
    result.update(
        {
            "rel": rel,
            "evidence_source": evidence_source,
            "resolved_path": current_path,
            "historical_path": None if historical is None else historical.get("path"),
            "copies": [
                {
                    "kind": c["kind"],
                    "path": c["path"],
                    "sha256": c["sha256"],
                    "evidence_source": c["evidence_source"],
                }
                for c in copies
            ],
        }
    )
    return result


def assess_entry(entry: RegistryEntry, derived_doc: Mapping[str, Any] | None = None) -> dict[str, Any]:
    derived_doc = derived_doc if derived_doc is not None else _load_derived(entry.derived)
    if derived_doc is None:
        return {
            "derived": entry.derived,
            "status": UNKNOWN,
            "reason": "derived receipt is not readable in receipts/future/",
            "producer": entry.producer,
            "kind": entry.kind,
            "sources": [],
        }
    sources_out: list[dict[str, Any]] = []
    if entry.kind == "manifest":
        captured = derived_doc.get("captured") or []
        if not isinstance(captured, list):
            captured = []
        for row in captured:
            if not isinstance(row, Mapping):
                continue
            rel = str(row.get("source_path") or "")
            if not rel:
                continue
            recorded_sha = row.get("sha256") if isinstance(row.get("sha256"), str) else None
            snap_rel = row.get("snapshot_path")
            hint = None
            if isinstance(snap_rel, str):
                snap_path = REPO / snap_rel
                if snap_path.is_file():
                    hint, _ = _load_maybe(snap_path)
                    if recorded_sha is None:
                        recorded_sha = sha256_file(snap_path)
            sources_out.append(_assess_one_source(rel, recorded_sha=recorded_sha, recorded_doc_hint=hint))
    else:
        recorded_sha = None
        if entry.recorded_sha_path:
            value = _dot_get(derived_doc, entry.recorded_sha_path)
            if isinstance(value, str) and value:
                recorded_sha = value
        for rel in entry.sources:
            # Only the first source (typically the queue) inherits the recorded sha.
            sha = recorded_sha if rel == entry.sources[0] else None
            sources_out.append(_assess_one_source(rel, recorded_sha=sha))
    statuses = [s.get("status") or UNKNOWN for s in sources_out]
    status = _aggregate_status(statuses)
    added: list[str] = []
    removed: list[str] = []
    changed: list[str] = []
    evidence_sources = []
    for src in sources_out:
        for cid in src.get("added_candidate_ids") or []:
            if cid not in added:
                added.append(cid)
        for cid in src.get("removed_candidate_ids") or []:
            if cid not in removed:
                removed.append(cid)
        for cid in src.get("status_changed_candidate_ids") or []:
            if cid not in changed:
                changed.append(cid)
        if src.get("evidence_source"):
            evidence_sources.append(src["evidence_source"])
    reason = {
        FRESH: "all compared sources match in bytes and meaning",
        STALE_FINGERPRINT_ONLY: "at least one source is a cosmetic rewrite; meaning is unchanged",
        STALE_SEMANTIC: "at least one source changed in meaning; regeneration required",
        UNKNOWN: "no provenance recorded, or a current source is not visible",
    }[status]
    unique_es = []
    for es in evidence_sources:
        if es not in unique_es:
            unique_es.append(es)
    return {
        "derived": entry.derived,
        "status": status,
        "reason": reason,
        "producer": entry.producer,
        "kind": entry.kind,
        "evidence_source": (
            unique_es[0]
            if len(unique_es) == 1
            else ("live_headless" if "live_headless" in unique_es else (unique_es[0] if unique_es else None))
        ),
        "added_candidate_ids": added,
        "removed_candidate_ids": removed,
        "status_changed_candidate_ids": changed,
        "missing_at_capture": (
            list(derived_doc.get("missing") or [])
            if entry.kind == "manifest" and isinstance(derived_doc.get("missing"), list)
            else []
        ),
        "sources": sources_out,
    }


def list_unregistered(registry: Sequence[RegistryEntry] | None = None) -> list[str]:
    registered = {e.derived for e in (registry or REGISTRY)}
    registered.add(OWN_RECEIPT)
    names: list[str] = []
    try:
        entries = sorted(RECEIPTS.glob("*.json"))
    except OSError:
        entries = []
    for path in entries:
        name = path.name
        if name in registered:
            continue
        names.append(name)
    return names


def _probe_path(rel: str) -> dict[str, Any]:
    p = REPO / rel
    in_git = bool(git("ls-tree", "--name-only", "HEAD", rel).strip())
    try:
        on_disk = p.exists()
    except OSError:
        on_disk = False
    return {"path": rel, "on_disk": on_disk, "in_git": in_git}


def recovered_implementation() -> dict[str, Any]:
    rows = []
    for path, note in RECOVERY_PROBES:
        row = _probe_path(path)
        row["note"] = note
        rows.append(row)
    return {
        "already_existed": rows,
        "adequate_as": (
            "CANDIDATE_STAGED_PLAN.input records queue_sha256 (bytes) and "
            "queue_fingerprint (the queue document's own full-body hash). "
            "codex_ingest hashes file bytes. evidence_snapshot pins captured[].sha256. "
            "None of those distinguish a cosmetic rewrite from a semantic change."
        ),
        "not_adequate_as": (
            "queue.fingerprint is sha256 of the whole document minus the fingerprint "
            "field, so timestamps, zero-valued status buckets and candidate list order "
            "move it. No module classified STALE_FINGERPRINT_ONLY vs STALE_SEMANTIC, "
            "gated --check on meaning, or refreshed only the semantic-stale producers."
        ),
        "gap_closed_by_this_module": (
            "semantic fingerprint over the candidate identity set; four-way "
            "classification; --check/--refresh; a one-row registry."
        ),
    }


def gaps_closed() -> list[str]:
    return [
        "semantic fingerprint over (candidate_id, status, affected_physical_region, "
        "exact_mutation, dependencies, blocked_reason); count derived from the rows",
        "ordering, whitespace, timestamps and zero-valued status buckets do not move the fingerprint",
        "FRESH / STALE_FINGERPRINT_ONLY / STALE_SEMANTIC / UNKNOWN with added/removed/status-changed ids",
        "--check exits 1 on STALE_SEMANTIC and 0 on STALE_FINGERPRINT_ONLY",
        "--refresh invokes each producer's own build() and does not rewrite sibling receipts here",
        "registry of derived -> source(s) -> producer; unregistered receipts/future/*.json are findings",
        "source resolution copes with live-present, live-absent, and pinned-snapshot copies and records which",
    ]


def negative_findings(classifications: Sequence[Mapping[str, Any]], unregistered: Sequence[str]) -> list[str]:
    findings = [
        "QUALIFICATION_PIPELINE.json, FLASH_NX_COMPLETENESS_AUDIT.json, "
        "HCLI_FUTURE_WORKUNITS.json and TOURNAMENT_READINESS.json record no source "
        "sha256/semantic fingerprint, so they classify UNKNOWN until those producers "
        "start writing the CANDIDATE_STAGED_PLAN.input pattern",
        "queue.fingerprint on ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json is a "
        "full-document hash, not the 6-tuple semantic fingerprint used here",
        "this sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE; "
        "bench.state stays UNKNOWN",
        "no GPU / FPGA / power meter in this lane; no hardware quantity is asserted",
        "--refresh was not invoked by --check/--report; STALE_SEMANTIC artifacts stay "
        "on disk until a caller passes --refresh",
    ]
    unknown = [c.get("derived") for c in classifications if c.get("status") == UNKNOWN]
    if unknown:
        findings.append("UNKNOWN artifacts: " + ", ".join(str(x) for x in unknown))
    semantic = [c.get("derived") for c in classifications if c.get("status") == STALE_SEMANTIC]
    if semantic:
        findings.append("STALE_SEMANTIC artifacts (regeneration required): " + ", ".join(str(x) for x in semantic))
    if unregistered:
        findings.append(f"{len(unregistered)} receipts/future/*.json files are not in the freshness registry")
    live_queue = resolve_copies(QUEUE_REL)
    kinds = {c["kind"] for c in live_queue}
    if "this_worktree" not in kinds:
        findings.append(
            "this worktree has no receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json; "
            "resolution used other_worktree/git_common/pinned_snapshot when present "
            "(absence here is a checkout fact, not a claim that the file does not exist)"
        )
    return findings


def _counts(classifications: Sequence[Mapping[str, Any]], unregistered: Sequence[str]) -> dict[str, int]:
    by = {FRESH: 0, STALE_FINGERPRINT_ONLY: 0, STALE_SEMANTIC: 0, UNKNOWN: 0}
    for row in classifications:
        status = str(row.get("status") or UNKNOWN)
        by[status] = by.get(status, 0) + 1
    return {
        "registered": len(classifications),
        "unregistered": len(unregistered),
        **by,
    }


def _evidence_source_top(classifications: Sequence[Mapping[str, Any]]) -> str:
    used: list[str] = []
    for row in classifications:
        es = row.get("evidence_source")
        if isinstance(es, str) and es not in used:
            used.append(es)
        for src in row.get("sources") or []:
            if isinstance(src, Mapping):
                value = src.get("evidence_source")
                if isinstance(value, str) and value not in used:
                    used.append(value)
    if used == ["pinned_snapshot"]:
        return "pinned_snapshot"
    return "live_headless" if used else "pinned_snapshot"


def assess(*, registry: Sequence[RegistryEntry] | None = None) -> dict[str, Any]:
    entries = tuple(registry) if registry is not None else REGISTRY
    classifications = [assess_entry(entry) for entry in entries]
    unregistered = list_unregistered(entries)
    recovered = recovered_implementation()
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Distinguish a cosmetic Codex rewrite (byte sha moved, meaning did not) "
            "from a semantic change that requires regenerating derived sidecar "
            "artifacts. FIVE ERAS, THREE ODYSSEYS. FPGA stays inside Accelerator / "
            "Physical Compiler / Fusion. DISK STATE IS AUTHORITY."
        ),
        "vocabulary": {
            FRESH: "byte sha and semantic fingerprint both match",
            STALE_FINGERPRINT_ONLY: "byte sha differs, semantic fingerprint identical — do not regenerate",
            STALE_SEMANTIC: "semantic fingerprint differs — regeneration required",
            UNKNOWN: "derived artifact recorded no provenance to compare (a finding)",
            "UNREGISTERED": "a receipts/future/*.json file with no registry row",
        },
        "queue_semantic_fields": list(QUEUE_SEMANTIC_FIELDS),
        "eras": [
            "I Genesis of the Laboratory",
            "II Compounding Civilization",
            "III Autonomous Science Civilization",
            "IV Synthetic Machine Civilization",
            "V Released Hawking Civilization",
        ],
        "odysseys": [
            "I WHAT IS TRUE?",
            "II WHAT DID HAWKING ALREADY LEARN?",
            "III WHERE IS HAWKING WRONG?",
        ],
        "no_era_vi": True,
        "no_odyssey_iv": True,
        "fpga": (
            "FPGA is part of Accelerator / Physical Compiler / Fusion. It is not "
            "its own civilization and this module does not build an FPGA backend."
        ),
        "measurement_class": "STATIC_ONLY",
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "registry": [
            {
                "derived": e.derived,
                "sources": list(e.sources),
                "producer": e.producer,
                "kind": e.kind,
                "recorded_sha_path": e.recorded_sha_path,
            }
            for e in entries
        ],
        "classifications": classifications,
        "unregistered": unregistered,
        "counts": _counts(classifications, unregistered),
        "check": {
            "stale_semantic": [
                c["derived"] for c in classifications if c.get("status") == STALE_SEMANTIC
            ],
            "would_exit_nonzero": any(c.get("status") == STALE_SEMANTIC for c in classifications),
            "rule": "exit 1 iff any registered artifact is STALE_SEMANTIC; STALE_FINGERPRINT_ONLY exits 0",
        },
        "recovered_implementation": recovered,
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(classifications, unregistered),
        "evidence_source": _evidence_source_top(classifications),
    }


def build(*, refresh: bool = False, registry: Sequence[RegistryEntry] | None = None) -> Path:
    body = assess(registry=registry)
    if refresh:
        refreshed = refresh_stale(body.get("classifications") or [])
        body = assess(registry=registry)
        body["refresh"] = refreshed
    return write_receipt(RECEIPT, body, RECORDED_BY)


# ---------------------------------------------------------------------------
# Synthetic queue helpers (selftest + unit tests). No live I/O.
# ---------------------------------------------------------------------------


def synthetic_candidate(candidate_id: str, **kwargs: Any) -> dict[str, Any]:
    row = {
        "candidate_id": candidate_id,
        "model": "Qwen27",
        "status": "READY_PROTECTED",
        "affected_physical_region": f"region-{candidate_id}",
        "dependencies": [],
        "blocked_reason": None,
        "exact_mutation": {"child_fusion_env": {"HAWKING_FOO": "1"}},
    }
    row.update(kwargs)
    return row


def synthetic_queue(rows: Sequence[Mapping[str, Any]], **extra: Any) -> dict[str, Any]:
    counted: dict[str, int] = {
        name: 0
        for name in (
            "BLOCKED",
            "DIAGNOSTIC_PASS",
            "DIAGNOSTIC_REJECT",
            "INTEGRATED",
            "PROTECTED_PASS",
            "PROTECTED_REJECT",
            "READY_DIAGNOSTIC",
            "READY_PROTECTED",
            "STATIC_ONLY",
        )
    }
    for row in rows:
        status = str(row.get("status") or "")
        counted[status] = counted.get(status, 0) + 1
    doc: dict[str, Any] = {
        "schema": "hawking.accelerator.physical_qualification_queue.v1",
        "version": 1,
        "bench": {
            "state": "UNKNOWN",
            "recorded_at": "2026-08-29T00:00:00Z",
            "recorded_by": "synthetic",
        },
        "candidates": list(rows),
        "counts": {
            "candidates": len(rows),
            "by_status": counted,
        },
    }
    doc.update(extra)
    return doc


def selftest() -> int:
    """A guard nobody has watched fail is not a guard."""
    a = synthetic_candidate("alpha", status="READY_PROTECTED")
    b = synthetic_candidate("beta", status="BLOCKED", blocked_reason="nx")
    old = synthetic_queue([a, b])
    # Cosmetic: key order, whitespace, an added zero-valued status bucket.
    new_cosmetic = {
        "version": old["version"],
        "schema": old["schema"],
        "counts": {
            "by_status": {**old["counts"]["by_status"], "BRAND_NEW_ZERO": 0},
            "candidates": old["counts"]["candidates"],
        },
        "candidates": [
            {
                "blocked_reason": b.get("blocked_reason"),
                "exact_mutation": b.get("exact_mutation"),
                "dependencies": b.get("dependencies"),
                "affected_physical_region": b.get("affected_physical_region"),
                "status": b.get("status"),
                "model": b.get("model"),
                "candidate_id": b.get("candidate_id"),
            },
            {
                "candidate_id": a["candidate_id"],
                "model": a["model"],
                "status": a["status"],
                "affected_physical_region": a["affected_physical_region"],
                "dependencies": a["dependencies"],
                "blocked_reason": a["blocked_reason"],
                "exact_mutation": {"child_fusion_env": {"HAWKING_FOO": "1"}},
            },
        ],
        "bench": {"recorded_by": "synthetic", "recorded_at": "2099-01-01T00:00:00Z", "state": "UNKNOWN"},
    }
    raw_old = json.dumps(old, indent=2).encode()
    raw_new = json.dumps(new_cosmetic, indent=4, sort_keys=True).encode()
    if _sha256_bytes(raw_old) == _sha256_bytes(raw_new):
        raise AssertionError("cosmetic pair must differ in bytes")
    cosmetic = classify_documents(old, new_cosmetic, old_raw=raw_old, new_raw=raw_new)
    if cosmetic["status"] != STALE_FINGERPRINT_ONLY:
        raise AssertionError(f"cosmetic rewrite classified {cosmetic['status']}, want STALE_FINGERPRINT_ONLY")
    if exit_code_for({"classifications": [cosmetic]}) != 0:
        raise AssertionError("STALE_FINGERPRINT_ONLY must not fail --check")
    moved = json.loads(json.dumps(new_cosmetic))
    for row in moved["candidates"]:
        if row.get("candidate_id") == "alpha":
            row["status"] = "BLOCKED"
    raw_moved = json.dumps(moved, indent=4, sort_keys=True).encode()
    semantic = classify_documents(old, moved, old_raw=raw_old, new_raw=raw_moved)
    if semantic["status"] != STALE_SEMANTIC:
        raise AssertionError(f"status change classified {semantic['status']}, want STALE_SEMANTIC")
    if "alpha" not in semantic["status_changed_candidate_ids"]:
        raise AssertionError(f"status_changed missing alpha: {semantic}")
    if exit_code_for({"classifications": [semantic]}) != 1:
        raise AssertionError("STALE_SEMANTIC must fail --check")
    return 0


def _print_report(body: Mapping[str, Any]) -> None:
    counts = body.get("counts") or {}
    print(
        "freshness "
        f"FRESH={counts.get(FRESH, 0)} "
        f"STALE_FINGERPRINT_ONLY={counts.get(STALE_FINGERPRINT_ONLY, 0)} "
        f"STALE_SEMANTIC={counts.get(STALE_SEMANTIC, 0)} "
        f"UNKNOWN={counts.get(UNKNOWN, 0)} "
        f"UNREGISTERED={counts.get('unregistered', 0)}"
    )
    for row in body.get("classifications") or []:
        derived = row.get("derived")
        status = row.get("status")
        extra = ""
        added = row.get("added_candidate_ids") or []
        removed = row.get("removed_candidate_ids") or []
        changed = row.get("status_changed_candidate_ids") or []
        bits = []
        if added:
            bits.append("added=" + ",".join(added))
        if removed:
            bits.append("removed=" + ",".join(removed))
        if changed:
            bits.append("status_changed=" + ",".join(changed))
        if bits:
            extra = "  " + "; ".join(bits)
        print(f"  {derived}: {status}{extra}")
    unreg = body.get("unregistered") or []
    if unreg:
        print(f"  unregistered ({len(unreg)}): " + ", ".join(unreg[:12]) + ("…" if len(unreg) > 12 else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if any artifact is STALE_SEMANTIC")
    ap.add_argument("--report", action="store_true", help="write DERIVED_FRESHNESS.json and print a summary")
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="regenerate only STALE_SEMANTIC artifacts via each producer's build()",
    )
    ap.add_argument("--selftest", action="store_true", help="run the cosmetic-vs-semantic negative control")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        print("selftest ok: cosmetic -> STALE_FINGERPRINT_ONLY (check=0); status change -> STALE_SEMANTIC (check=1)")
        return 0
    path = build(refresh=args.refresh)
    doc = load_json(path)
    print(path)
    _print_report(doc)
    if args.check:
        return exit_code_for(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
