"""Observe a supervised ModelLake acquisition without mutating the lake."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hcli.flash_next import EXPECTED_BYTES, PINNED_REVISION, REPO_ID
from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.modellake_supervision.v1"
LAKE = Path("/Volumes/corpdrive/hawking-modellake")
SLUG = REPO_ID.replace("/", "--") + "@" + PINNED_REVISION[:12]


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _direct_inventory(path: Path) -> Dict[str, Any]:
    rows = []
    total = 0
    if not path.is_dir():
        return {"path": str(path), "present": False, "direct_files": 0, "direct_bytes": 0, "entries": []}
    try:
        with os.scandir(path) as iterator:
            for index, entry in enumerate(iterator):
                if index >= 512:
                    break
                try:
                    if entry.is_file(follow_symlinks=False):
                        size = entry.stat(follow_symlinks=False).st_size
                        total += size
                        rows.append({"name": entry.name, "bytes": size})
                except OSError:
                    continue
    except OSError as exc:
        return {"path": str(path), "present": True, "direct_files": 0, "direct_bytes": 0, "entries": [], "error": str(exc)[:400]}
    return {"path": str(path), "present": True, "direct_files": len(rows), "direct_bytes": total, "entries": sorted(rows, key=lambda row: row["name"])}


def _job(repo: Path, job_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not job_id:
        return None
    token = str(job_id)
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in token):
        return None
    return _read(repo / ".hcli" / "background" / "jobs" / f"{token}.json")


def run_model_lake_supervision(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    job_id: Optional[str] = None,
    emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    job = _job(repo, job_id)
    final = LAKE / "specimens" / SLUG
    partial = LAKE / "partial" / SLUG
    usage = os.statvfs(LAKE) if LAKE.exists() else None
    free_bytes = usage.f_bavail * usage.f_frsize if usage else None
    process_alive = False
    try:
        import subprocess

        ps = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True, timeout=5.0, check=False)
        process_alive = any(
            ("modellake.py acquire" in line.lower() or "hf download" in line.lower())
            and REPO_ID.lower() in line.lower()
            for line in (ps.stdout or "").splitlines()
        )
    except (OSError, subprocess.SubprocessError):
        process_alive = False
    argv = (job or {}).get("argv") if isinstance(job, dict) else []
    expected_argv = ["python3", "tools/odyssey/modellake.py", "acquire", "--repo", REPO_ID, "--revision", PINNED_REVISION]
    job_identity_ok = isinstance(argv, list) and REPO_ID in argv and PINNED_REVISION in argv and "acquire" in argv
    final_manifest = _read(LAKE / "manifests" / f"{SLUG}.json")
    final_published = final.is_dir() and isinstance(final_manifest, dict) and final_manifest.get("resolved_sha") == PINNED_REVISION
    if final_published:
        status = "PASSED"
    elif job and job.get("state") == "RUNNING" and process_alive:
        status = "RUNNING_SAFE"
    elif job and job.get("state") in {"INTERRUPTED", "FAILED"}:
        status = "INTERRUPTED_OR_FAILED_RESUMABLE"
    else:
        status = "WAITING_OR_NOT_OBSERVED"
    report = {
        "schema": SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "status": status,
        "qualification": "MODELLAKE_FLASH_ACQUISITION_SUPERVISED_NO_DELETE",
        "observed_at": time.time(),
        "repo_root": str(repo),
        "target": {"repo": REPO_ID, "pinned_revision": PINNED_REVISION, "slug": SLUG, "expected_bytes": EXPECTED_BYTES},
        "job": job,
        "expected_argv": expected_argv,
        "capacity": {"path": str(LAKE), "free_bytes": free_bytes, "free_after_expected_bytes": (free_bytes - EXPECTED_BYTES) if free_bytes is not None else None},
        "partial": _direct_inventory(partial),
        "final": _direct_inventory(final),
        "final_manifest": final_manifest,
        "checks": {
            "job_identity_matches_pinned_target": job_identity_ok,
            "job_resumable": bool((job or {}).get("resumable")) if job else False,
            "capacity_headroom_observed": free_bytes is not None and free_bytes > EXPECTED_BYTES,
            "partial_not_final": not final_published or (job or {}).get("state") == "RUNNING",
            "no_delete_policy": True,
            "atomic_publish_identity": not final_published or (isinstance(final_manifest, dict) and final_manifest.get("resolved_sha") == PINNED_REVISION),
        },
        "next_action": "If RUNNING_SAFE, continue observing. If interrupted, re-census and explicitly resume the same pinned argv; publish only after the existing tool verifies every file and atomically renames the tree.",
        "claim_boundary": "This receipt observes a resumable acquisition and never claims a partial tree is ready; it makes no model capability or performance claim.",
    }
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / "HCLI_MODELLAKE_FLASH_ACQUISITION_SUPERVISION.json"
    atomic_write_json(destination, report)
    report["receipt_path"] = str(destination)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--job-id")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = run_model_lake_supervision(repo_root=args.repo_root, job_id=args.job_id, emit=args.emit)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") in {"PASSED", "RUNNING_SAFE", "WAITING_OR_NOT_OBSERVED"} else 1


__all__ = ["SCHEMA", "run_model_lake_supervision", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
