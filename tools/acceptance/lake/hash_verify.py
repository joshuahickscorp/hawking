"""MODELLAKE_HASH_VERIFIED — CALL reconcile (scratch) + live size/oid checks.

reconcile() on the live lake would promote with go=True. This lane must not
move anything, so reconcile is invoked only against a scratch tree. Live
verification is read-only: per-file size against watch-manifests, and
verify_only() (sha256 vs hub LFS oid) on the canonical small specimen.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from tools.acceptance.lake.common import (
    GATES,
    LAKE,
    PARTIAL,
    SPECIMENS,
    WORKTREE,
    lake_mounted,
    timed,
    write_receipt,
)
from tools.odyssey import modellake as ml
from tools.odyssey import modellake_promote as mp
from tools.odyssey import modellake_watch as mw
from tools.odyssey.modellake_lineage import (
    CANONICAL_REPO,
    CANONICAL_REVISION,
    CANONICAL_SPECIMEN,
    load_watch_manifest,
)

GATE = "MODELLAKE_HASH_VERIFIED"


def call_reconcile() -> dict[str, Any]:
    """Production call site of the catalog symbol."""
    return mw.reconcile()


def run_reconcile_on_scratch(root: Path) -> dict[str, Any]:
    """Invoke reconcile() against an isolated tree. Never the live lake."""
    model_root = root / "model_root"
    partial = model_root / "partial"
    specimens = model_root / "specimens"
    manifests = root / "manifests"
    partial.mkdir(parents=True)
    specimens.mkdir(parents=True)
    manifests.mkdir(parents=True)
    log = root / "watch.jsonl"
    log.write_text("", encoding="utf-8")

    tag = "acme--hash@deadbeefcafe"
    files = {"config.json": b"{}", "weights.bin": b"0" * 2048}
    src = partial / tag
    src.mkdir()
    for name, content in files.items():
        (src / name).write_bytes(content)
    (manifests / f"{tag}.json").write_text(
        json.dumps(
            {
                "repo": "acme/hash",
                "revision": "deadbeefcafe",
                "mode": "safe",
                "expected": sum(len(v) for v in files.values()),
                "files": list(files),
                "sizes": {n: len(c) for n, c in files.items()},
                "resolved_sha": "deadbeefcafe",
            }
        ),
        encoding="utf-8",
    )

    saved = {
        "MODEL_ROOT": mp.MODEL_ROOT,
        "PARTIAL_ROOT": mp.PARTIAL_ROOT,
        "SPECIMEN_ROOT": mp.SPECIMEN_ROOT,
        "MANIFEST_DIR": mp.MANIFEST_DIR,
        "mw_SPECIMEN_ROOT": mw.SPECIMEN_ROOT,
        "mw_MANIFEST_DIR": mw.MANIFEST_DIR,
        "mw_LOG": mw.LOG,
        "mw_P0": mw.P0,
        "mw_QUEUE": mw.QUEUE,
        "mw_notify": mw.notify,
    }
    try:
        mp.MODEL_ROOT = model_root
        mp.PARTIAL_ROOT = partial
        mp.SPECIMEN_ROOT = specimens
        mp.MANIFEST_DIR = manifests
        mw.SPECIMEN_ROOT = specimens
        mw.MANIFEST_DIR = manifests
        mw.LOG = log
        mw.P0 = []
        mw.QUEUE = []
        mw.notify = lambda *a, **k: None  # noqa: ARG005 — scratch run is silent
        result = call_reconcile()
        dest = specimens / tag
        result = dict(result)
        result["scratch"] = {
            "promoted_tag": tag,
            "destination_present": dest.is_dir(),
            "partial_gone": not src.is_dir(),
            "weights_bytes": (dest / "weights.bin").stat().st_size if dest.is_dir() else None,
            "live_lake_untouched": str(SPECIMENS) not in str(dest),
        }
        return result
    finally:
        mp.MODEL_ROOT = saved["MODEL_ROOT"]
        mp.PARTIAL_ROOT = saved["PARTIAL_ROOT"]
        mp.SPECIMEN_ROOT = saved["SPECIMEN_ROOT"]
        mp.MANIFEST_DIR = saved["MANIFEST_DIR"]
        mw.SPECIMEN_ROOT = saved["mw_SPECIMEN_ROOT"]
        mw.MANIFEST_DIR = saved["mw_MANIFEST_DIR"]
        mw.LOG = saved["mw_LOG"]
        mw.P0 = saved["mw_P0"]
        mw.QUEUE = saved["mw_QUEUE"]
        mw.notify = saved["mw_notify"]


def _file_sizes(root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not root.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            path = Path(dirpath) / name
            try:
                rel = path.relative_to(root).as_posix()
                out[rel] = path.stat().st_size
            except OSError:
                continue
    return out


def size_audit_specimen(slug: str) -> dict[str, Any]:
    """Same rule as modellake_promote._verify_dir / modellake_watch.complete.

    Declared files only, exact size, including dotfiles such as .gitattributes.
    Extra files (.cache, .DS_Store) are not incompleteness.
    """
    root = SPECIMENS / slug
    watch = load_watch_manifest(slug, git_root=str(WORKTREE))
    row: dict[str, Any] = {
        "slug": slug,
        "present": root.is_dir(),
        "watch_present": watch is not None,
    }
    if not watch:
        from tools.odyssey.modellake_lineage import load_json_file

        doc = load_json_file(LAKE / "manifests" / f"{slug}.json")
        observed = _file_sizes(root)
        row.update(
            {
                "n_files_on_disk": len(observed),
                "bytes_on_disk": sum(observed.values()),
                "check": "lake_manifest_no_per_file_list",
                "size_complete": None,
                "lake_manifest": {
                    "present": doc is not None,
                    "n_files": None if doc is None else doc.get("n_files"),
                    "bytes": None if doc is None else doc.get("bytes"),
                    "n_sha256_verified": None if doc is None else doc.get("n_sha256_verified"),
                    "n_size_only_verified": None if doc is None else doc.get("n_size_only_verified"),
                },
            }
        )
        return row

    expected_files = list(watch.get("files") or [])
    sizes = dict(watch.get("sizes") or {})
    missing, wrong = [], []
    total = 0
    for name in expected_files:
        path = root / name
        try:
            observed = path.stat().st_size
        except OSError:
            missing.append(name)
            continue
        total += observed
        expected = sizes.get(name)
        if expected is not None and observed != expected:
            wrong.append({"file": name, "on_disk": observed, "expected": expected})
    row.update(
        {
            "check": "watch_manifest_per_file_size",
            "n_declared": len(expected_files),
            "n_files_on_disk": len(expected_files) - len(missing),
            "bytes_on_disk": total,
            "missing": len(missing),
            "wrong_size": len(wrong),
            "first_missing": missing[:5],
            "first_wrong": wrong[:5],
            "size_complete": not missing and not wrong,
        }
    )
    return row


def live_size_audit(slugs: list[str]) -> list[dict[str, Any]]:
    return [size_audit_specimen(slug) for slug in slugs]


def call_verify_only(repo: str, rev: str, root: str | Path) -> dict[str, Any]:
    """sha256 vs hub LFS oid. Read-only on the specimen tree."""
    return ml.verify_only(repo, rev, root)


def run_hash_gate(
    *,
    live: bool = True,
    scratch_root: Optional[Path] = None,
    run_canonical_oid_hash: bool = True,
) -> dict[str, Any]:
    command = ["python3", "-m", "tools.acceptance.lake", "--gate", GATE]
    with timed() as clock:
        if scratch_root is None:
            import tempfile

            tmp = Path(tempfile.mkdtemp(prefix="acc4-hash-"))
        else:
            tmp = Path(scratch_root)
        reconcile_result = run_reconcile_on_scratch(tmp)
        scratch = reconcile_result.get("scratch") or {}

        slugs = sorted(p.name for p in SPECIMENS.iterdir() if p.is_dir() and not p.name.startswith(".")) if live and lake_mounted() else []
        size_rows = live_size_audit(slugs) if slugs else []
        size_complete = [r for r in size_rows if r.get("size_complete") is True]
        size_failed = [r for r in size_rows if r.get("size_complete") is False]
        size_unknown = [r for r in size_rows if r.get("size_complete") is None]

        oid: Optional[dict[str, Any]] = None
        oid_error: Optional[dict[str, str]] = None
        if live and lake_mounted() and run_canonical_oid_hash:
            root = SPECIMENS / CANONICAL_SPECIMEN
            try:
                oid = call_verify_only(CANONICAL_REPO, CANONICAL_REVISION, root)
            except Exception as exc:
                oid_error = {"type": type(exc).__name__, "message": str(exc)[:2000]}

        # Live cryptographic coverage: 1 canonical specimen, not 4.35 TB.
        n_oid_ok = int(bool(oid and oid.get("verified") is True))
        n_need_oid = 55 if live and slugs else 0
        missing_tb = None
        if slugs:
            hashed_bytes = 0
            if oid and oid.get("verified"):
                hashed_bytes = next(
                    (r["bytes_on_disk"] for r in size_rows if r["slug"] == CANONICAL_SPECIMEN),
                    0,
                )
            total_bytes = sum(r["bytes_on_disk"] for r in size_rows)
            missing_tb = round((total_bytes - hashed_bytes) / 1e12, 3)

        checks = {
            "reconcile_invoked": True,
            "reconcile_promoted_scratch": bool(scratch.get("destination_present"))
            and bool(scratch.get("partial_gone")),
            "scratch_is_not_the_live_lake": bool(scratch.get("live_lake_untouched")),
            "live_partial_is_empty": (
                lake_mounted()
                and not any(
                    p.is_dir() and not p.name.startswith(".")
                    for p in PARTIAL.iterdir()
                )
                if live and PARTIAL.is_dir()
                else None
            ),
            "all_sealed_size_complete": bool(size_rows)
            and not size_failed
            and not size_unknown,
            "canonical_oid_hash_verified": bool(oid and oid.get("verified") is True),
            "full_lake_oid_hash_complete": n_oid_ok >= n_need_oid and n_need_oid > 0,
            "no_lake_write": True,
        }

        # Criterion is oid-backed hash of sealed specimens, not size alone.
        # Size completeness of 55/55 plus one canonical oid-hash is not the bar.
        if checks["full_lake_oid_hash_complete"]:
            verdict = "ACCEPTED"
            blocker = None
        else:
            verdict = "BLOCKED"
            blocker = {
                "missing_input": (
                    f"oid-backed sha256 of {n_need_oid - n_oid_ok} sealed specimens "
                    f"({missing_tb} TB remaining after the canonical 0.6B hash)"
                ),
                "why": (
                    "verify_only() hashes every file against the hub LFS oid. "
                    "Re-hashing the 4.35 TB school is a multi-hour HDD pass; this "
                    "lane hashed the canonical Qwen3-0.6B specimen and size-checked "
                    "the rest. Size match is not sha256. Catalog symbol reconcile() "
                    "was invoked on a scratch tree because the live call promotes "
                    "with go=True, which this lane must not do."
                ),
                "canonical_specimen": CANONICAL_SPECIMEN,
                "canonical_verify_only": oid,
                "canonical_error": oid_error,
            }

        measured = {
            "reconcile_promoted": list(reconcile_result.get("promoted") or []),
            "reconcile_anomalies": reconcile_result.get("anomalies"),
            "scratch": scratch,
            "sealed_specimens": len(slugs),
            "size_complete": len(size_complete),
            "size_failed": len(size_failed),
            "size_unknown": len(size_unknown),
            "oid_hashed_specimens": n_oid_ok,
            "oid_hash_required": n_need_oid,
            "canonical_verify_only": oid,
            "canonical_error": oid_error,
            "remaining_unhashed_tb": missing_tb,
        }
        output = {
            "summary": (
                f"reconcile scratch promoted={scratch.get('destination_present')}; "
                f"size-complete {len(size_complete)}/{len(size_rows)}; "
                f"oid-hash {n_oid_ok}/{n_need_oid or 55}"
            ),
            "size_failures": size_failed[:10],
            "size_unknown": size_unknown[:10],
            "size_rows": [
                {
                    "slug": r["slug"],
                    "size_complete": r.get("size_complete"),
                    "check": r.get("check"),
                    "n_files_on_disk": r.get("n_files_on_disk"),
                    "bytes_on_disk": r.get("bytes_on_disk"),
                    "missing": r.get("missing"),
                    "wrong_size": r.get("wrong_size"),
                }
                for r in size_rows
            ],
        }
        return write_receipt(
            GATE,
            verdict=verdict,
            command=command,
            output=output,
            measured=measured,
            checks=checks,
            evidence_tier="STATIC",
            symbol_invoked=True,
            blocker=blocker,
            elapsed_s=clock.snap(),
            extra={"lake": str(LAKE), "implementing_symbol": GATES[GATE]["symbol"]},
        )


def materialize_watch_manifests(dest: Path) -> Path:
    """Read watch-manifests from git (sparse fallback). Does not touch the lake."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        [
            "git",
            "-C",
            str(WORKTREE),
            "archive",
            "HEAD",
            "workspace/campaign/odyssey/watch-manifests",
        ],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["tar", "-x", "-C", str(dest)],
        input=archive.stdout,
        check=True,
    )
    return dest / "workspace" / "campaign" / "odyssey" / "watch-manifests"
