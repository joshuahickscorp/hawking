"""Safe ModelLake census and pinned Flash-Next manifest capture.

This command inventories only bounded, immediate directory children and fetches
metadata from the pinned Hugging Face revision.  It never downloads weights,
recursively hashes a 360 GB tree, deletes a stale partial, or treats a partial
directory as a verified specimen.  Acquisition remains an explicit, separately
governed action.
"""
from __future__ import annotations

import sys
from pathlib import Path as _CausalityPath
_CAUSALITY_ROOT = _CausalityPath(__file__).resolve().parents[2]
if str(_CAUSALITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_CAUSALITY_ROOT))
from tools.future import status_causality as sc

import sys

import hashlib
import json
import os
import platform
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from hcli.flash_next import EXPECTED_BYTES, EXPECTED_FILE_COUNT, PINNED_REVISION, REPO_ID
from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.modellake_census.v1"
LAKE = Path("/Volumes/corpdrive/hawking-modellake")
TIER2 = LAKE / "specimens"
PARTIAL = LAKE / "partial"
MANIFESTS = LAKE / "manifests"
STALE_AFTER_S = 24 * 60 * 60
MAX_CHILDREN = 512
MAX_HASH_BYTES = 32 * 1024 * 1024
API = "https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
PROTECTED_VOLUME_NAMES = (
    "legal-scans-2026-08-23.tar.zst",
    "substrate",
    "legal-scans-2026-08-23.README.txt",
)


FIVE_RECORDED_FIELDS: tuple[str, ...] = getattr(
    sc,
    "FIVE_RECORDED_FIELDS",
    (
        "probe_performed",
        "direct_observation",
        "interpretation",
        "confidence",
        "alternatives",
    ),
)


def _bind_emit() -> None:
    if hasattr(sc, "emit"):
        return

    def emit(
        status: str,
        *,
        probe_performed: str = "",
        direct_observation: Any = "",
        interpretation: str = "",
        probe_kind: str = "",
        claim_kind: str | None = None,
        falsifier: str = "",
        source: str = "",
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "status": status,
            "probe_performed": probe_performed,
            "direct_observation": direct_observation,
            "interpretation": interpretation or status,
            "probe_kind": probe_kind,
            "use_catalog": False,
            "source": source or "<emit>",
        }
        if claim_kind:
            row["claim_kind"] = claim_kind
        if falsifier:
            row["falsifier"] = falsifier
        out = sc.challenge(row)
        out["entry"] = "emit"
        return out

    sc.emit = emit  # type: ignore[attr-defined]


_bind_emit()


def records_five_fields(node: Any) -> bool:
    fn = getattr(sc, "records_five_fields", None)
    if callable(fn):
        return bool(fn(node))
    if not isinstance(node, dict):
        return False
    if not all(k in node for k in FIVE_RECORDED_FIELDS):
        return False
    if not str(node.get("probe_performed") or "").strip():
        return False
    if node.get("direct_observation") in (None, "", [], {}):
        return False
    if not str(node.get("interpretation") or "").strip():
        return False
    conf = node.get("confidence")
    if not isinstance(conf, dict):
        return False
    if not {"would_raise", "would_lower", "level", "about"} <= set(conf):
        return False
    alts = node.get("alternatives")
    return isinstance(alts, list) and bool(alts)


def _record_gate_causality(
    report: Dict[str, Any],
    *,
    probe_performed: str = "",
    direct_observation: Any = "",
    interpretation: str | None = None,
    probe_kind: str = "",
    claim_kind: str | None = None,
    source: str = "",
) -> dict[str, Any]:
    """Stamp the five causality fields. Does not change status/qualification/checks.

    An unsupplied observation is UNTESTED, never a restatement of PASSED/FAILED.
    OVERREACHING is recorded beside the verdict; it does not override it.
    """
    status_before = report.get("status")
    qual_before = report.get("qualification")
    checks_before = dict(report["checks"]) if isinstance(report.get("checks"), dict) else report.get("checks")
    status = str(report.get("status") or "")
    unsupplied = direct_observation in (None, "", [], {})
    rec = sc.emit(
        status,
        probe_performed=str(probe_performed or ""),
        direct_observation="" if unsupplied else direct_observation,
        interpretation=interpretation if interpretation is not None else status,
        probe_kind="" if unsupplied else probe_kind,
        claim_kind=None if unsupplied else claim_kind,
        source=source,
    )
    for key in FIVE_RECORDED_FIELDS:
        report[key] = rec[key]
    report["causality_verdict"] = rec["verdict"]
    report["falsifier"] = rec.get("falsifier")
    if rec.get("probe_kind"):
        report["probe_kind"] = rec["probe_kind"]
    if rec.get("claim_kind") is not None:
        report["claim_kind"] = rec["claim_kind"]
    checks_after = dict(report["checks"]) if isinstance(report.get("checks"), dict) else report.get("checks")
    if (
        report.get("status") != status_before
        or report.get("qualification") != qual_before
        or checks_after != checks_before
    ):
        raise RuntimeError("status_causality.emit mutated the gate verdict")
    return rec


def _sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _fetch_manifest(repo: str, revision: str, timeout_s: float) -> Dict[str, Any]:
    url = API.format(repo=repo, revision=revision)
    request = urllib.request.Request(url, headers={"User-Agent": "hawking-hcli/1"})
    with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_s))) as response:
        payload = json.load(response)
    resolved = payload.get("sha")
    if resolved != revision:
        raise ValueError(f"pinned revision mismatch: requested {revision}, resolved {resolved}")
    files = []
    for item in payload.get("siblings") or []:
        if not isinstance(item, Mapping):
            continue
        lfs = item.get("lfs") if isinstance(item.get("lfs"), Mapping) else {}
        files.append({
            "file": item.get("rfilename"),
            "size": item.get("size"),
            "remote_hash": lfs.get("sha256") or lfs.get("oid"),
            "blob_id": item.get("blobId"),
        })
    return {
        "repo": repo,
        "requested_revision": revision,
        "resolved_revision": resolved,
        "last_modified": payload.get("lastModified"),
        "file_count": len(files),
        "total_declared_bytes": sum(int(item.get("size") or 0) for item in files),
        "files": files,
        "source_url": url,
    }


def _bounded_size(path: Path) -> Dict[str, Any]:
    """Measure one child with a short timeout; never walk indefinitely."""
    if path.is_file():
        try:
            return {"bytes": path.stat().st_size, "status": "EXACT_FILE_STAT"}
        except OSError as exc:
            return {"bytes": None, "status": "STAT_FAILED", "error": str(exc)[:300]}
    if not path.is_dir():
        return {"bytes": None, "status": "NOT_A_REGULAR_PATH"}
    try:
        result = subprocess.run(
            ["du", "-sk", str(path)],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return {"bytes": int(result.stdout.split()[0]) * 1024, "status": "BOUNDED_DU"}
        return {"bytes": None, "status": "DU_FAILED", "returncode": result.returncode}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"bytes": None, "status": "DU_TIMEOUT_OR_FAILED", "error": type(exc).__name__}


def _children(root: Path, *, now: float) -> Dict[str, Any]:
    if not root.is_dir():
        return {"path": str(root), "present": False, "entries": [], "truncated": False}
    entries = []
    truncated = False
    try:
        with os.scandir(root) as iterator:
            for index, entry in enumerate(iterator):
                if index >= MAX_CHILDREN:
                    truncated = True
                    break
                path = Path(entry.path)
                try:
                    stat = entry.stat(follow_symlinks=False)
                    age_s = max(0.0, now - stat.st_mtime)
                    kind = "directory" if entry.is_dir(follow_symlinks=False) else "file" if entry.is_file(follow_symlinks=False) else "other"
                    size = _bounded_size(path)
                except OSError as exc:
                    age_s = None
                    kind = "unreadable"
                    size = {"bytes": None, "status": "STAT_FAILED", "error": str(exc)[:300]}
                entries.append({
                    "name": entry.name,
                    "path": str(path),
                    "kind": kind,
                    "age_s": round(age_s, 1) if age_s is not None else None,
                    "size": size,
                })
    except OSError as exc:
        return {"path": str(root), "present": True, "entries": [], "truncated": False, "error": str(exc)[:500]}
    entries.sort(key=lambda item: item["name"])
    return {"path": str(root), "present": True, "entries": entries, "truncated": truncated}


def _processes() -> Dict[str, Any]:
    needles = ("modellake", "huggingface", "hf download", "hf_transfer", "lake_filler", "autoadvance")
    try:
        result = subprocess.run(
            ["ps", "-Ao", "pid=,pcpu=,rss=,command="],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "UNKNOWN", "needles": list(needles), "error": type(exc).__name__, "matches": []}
    matches = []
    for line in (result.stdout or "").splitlines():
        lower = line.lower()
        if not any(needle in lower for needle in needles):
            continue
        fields = line.split(None, 3)
        matches.append({
            "pid": fields[0] if fields else None,
            "cpu_pct": fields[1] if len(fields) > 1 else None,
            "rss_kib": fields[2] if len(fields) > 2 else None,
            "command": fields[3] if len(fields) > 3 else line,
        })
    return {"status": "AVAILABLE" if result.returncode == 0 else "UNKNOWN", "needles": list(needles), "matches": matches}


def _manifest_inventory(root: Path) -> Dict[str, Any]:
    if not root.is_dir():
        return {"path": str(root), "present": False, "receipts": []}
    rows = []
    try:
        for entry in sorted(root.iterdir(), key=lambda item: item.name):
            if not entry.is_file() or entry.suffix.lower() != ".json":
                continue
            value = _read_json(entry)
            rows.append({
                "path": str(entry),
                "bytes": entry.stat().st_size,
                "valid_json_object": value is not None,
                "repo": value.get("repo") if value else None,
                "revision": value.get("revision") if value else None,
                "resolved_sha": value.get("resolved_sha") if value else None,
                "specimen_path": value.get("path") if value else None,
                "n_files": value.get("n_files") if value else None,
                "n_sha256_verified": value.get("n_sha256_verified") if value else None,
                "n_size_only_verified": value.get("n_size_only_verified") if value else None,
            })
    except OSError as exc:
        return {"path": str(root), "present": True, "receipts": [], "error": str(exc)[:500]}
    return {"path": str(root), "present": True, "receipts": rows}


def _local_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "bytes": None, "sha256": None, "hash_status": "ABSENT"}
    try:
        size = path.stat().st_size
    except OSError as exc:
        return {"path": str(path), "exists": False, "bytes": None, "sha256": None, "hash_status": "STAT_FAILED", "error": str(exc)[:300]}
    if size <= MAX_HASH_BYTES:
        return {"path": str(path), "exists": True, "bytes": size, "sha256": _sha256(path), "hash_status": "HASHED"}
    return {"path": str(path), "exists": True, "bytes": size, "sha256": None, "hash_status": "DEFERRED_LARGE_FILE"}


def _target_manifest_view(remote: Mapping[str, Any]) -> Dict[str, Any]:
    slug = REPO_ID.replace("/", "--") + "@" + PINNED_REVISION[:12]
    final_root = TIER2 / slug
    partial_root = PARTIAL / slug
    rows = []
    for item in remote.get("files") or []:
        filename = str(item.get("file") or "")
        final = _local_file(final_root / filename)
        partial = _local_file(partial_root / filename)
        local = final if final.get("exists") else {**partial, "path": final.get("path"), "source": "partial"}
        verified = bool(
            local.get("exists")
            and item.get("size") is not None
            and local.get("bytes") == item.get("size")
            and item.get("remote_hash") is not None
            and local.get("sha256") == item.get("remote_hash")
        )
        rows.append({
            "file": filename,
            "size": item.get("size"),
            "remote_hash": item.get("remote_hash"),
            "local": local,
            "partial": partial,
            "verified": verified,
        })
    return {
        "slug": slug,
        "final_root": str(final_root),
        "partial_root": str(partial_root),
        "final_present": final_root.is_dir(),
        "partial_present": partial_root.is_dir(),
        "files": rows,
        "verified_file_count": sum(1 for row in rows if row["verified"]),
        "whole_tree_verified": bool(rows) and all(row["verified"] for row in rows),
    }


def _stale_partials(partials: Mapping[str, Any], processes: Mapping[str, Any], now: float) -> list[Dict[str, Any]]:
    commands = " ".join(str(item.get("command") or "").lower() for item in processes.get("matches") or [])
    rows = []
    for entry in partials.get("entries") or []:
        age_s = entry.get("age_s")
        if entry.get("kind") != "directory" or not isinstance(age_s, (int, float)):
            continue
        active = any(token in commands for token in ("modellake", "huggingface", "hf download", "hf_transfer", "lake_filler", "autoadvance"))
        rows.append({
            "path": entry.get("path"),
            "age_s": age_s,
            "candidate_stale": age_s > STALE_AFTER_S and not active,
            "active_process_evidence": active,
            "action": "REPORT_ONLY_NO_DELETE",
        })
    return rows


def _write(report: Dict[str, Any], emit: Optional[str], repo_root: Path) -> None:
    destination = Path(emit).expanduser() if emit else repo_root / "receipts" / "headless" / "HCLI_MODELLAKE_FLASH_CENSUS.json"
    if not destination.is_absolute():
        destination = repo_root / destination
    report["receipt_path"] = str(destination.resolve())
    atomic_write_json(destination, report)



def causality_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    error = report.get("error")
    capacity = report.get("capacity") if isinstance(report.get("capacity"), dict) else {}
    remote = report.get("remote_manifest") if isinstance(report.get("remote_manifest"), dict) else {}
    target = report.get("flash_target_manifest") if isinstance(report.get("flash_target_manifest"), dict) else {}
    policy = report.get("acquisition_policy") if isinstance(report.get("acquisition_policy"), dict) else {}
    unmet = [name for name, value in checks.items() if value is not True]
    if not checks and not error:
        return {
            "probe_performed": "",
            "direct_observation": "",
            "interpretation": str(report.get("status") or ""),
            "probe_kind": "",
            "claim_kind": None,
        }
    status = str(report.get("status") or "")
    return {
        "probe_performed": (
            "ModelLake census: os.statvfs of lake root, bounded iterdir of specimens/ "
            "and partial/, manifest inventory, huggingface API fetch of the pinned "
            "Flash-Next revision; no weight download, no recursive 360GB hash"
        ),
        "direct_observation": (
            f"capacity.mounted_directory={capacity.get('mounted_directory')!r}; "
            f"capacity.free_bytes={capacity.get('free_bytes')!r}; "
            f"remote.resolved_revision={remote.get('resolved_revision')!r}; "
            f"remote.file_count={remote.get('file_count')!r}; "
            f"remote.total_declared_bytes={remote.get('total_declared_bytes')!r}; "
            f"target.whole_tree_verified={target.get('whole_tree_verified')!r}; "
            f"download_performed={policy.get('download_performed')!r}; "
            f"checks={{{', '.join(f'{k}={v!r}' for k, v in sorted(checks.items()))}}}; "
            f"unmet={unmet!r}; error={error!r}"
        ),
        "interpretation": (
            "pinned Flash identity matched the remote manifest and the lake was mounted "
            "with enough free bytes; no acquisition ran"
            if status == "PASSED"
            else f"modellake census checks unmet: {unmet or ['exception before checks']}"
        ),
        "probe_kind": sc.PROBE_MEASURED_FLAGS,
        "claim_kind": sc.CLAIM_FIELD_VALUE if status == "PASSED" else sc.CLAIM_MEASURED_UNMET,
    }


def record_modellake_causality(report: Dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    payload = kwargs or causality_payload(report)
    return _record_gate_causality(
        report,
        source="hcli/agentos/modellake_gate.py::run_modellake_census",
        **payload,
    )

def run_modellake_census(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Capture storage/process/manifest identity without acquiring weights."""
    root = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    started = time.time()
    now = time.time()
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "qualification": "MODELLAKE_CENSUS_PINNED_FLASH_IDENTITY_NO_ACQUISITION",
        "started_at": started,
        "repo_root": str(root),
        "source": {"repo": REPO_ID, "requested_revision": PINNED_REVISION},
        "acquisition_policy": {
            "download_performed": False,
            "human_confirmation_required": True,
            "resumable_command": ["hf", "download", REPO_ID, "--revision", PINNED_REVISION, "--local-dir", "<partial destination>"],
            "atomic_publish_required": True,
            "large_tree_recursive_hash": "DEFERRED_UNTIL_EXPLICIT_ACQUIRE",
        },
    }
    processes = _processes()
    try:
        usage = os.statvfs(LAKE) if LAKE.exists() else None
        capacity = {
            "path": str(LAKE),
            "mounted_directory": LAKE.is_dir(),
            "filesystem": "observed via statvfs" if usage else None,
            "capacity_bytes": (usage.f_blocks * usage.f_frsize) if usage else None,
            "free_bytes": (usage.f_bavail * usage.f_frsize) if usage else None,
            "available_for_unprivileged_bytes": (usage.f_bavail * usage.f_frsize) if usage else None,
        }
        specimens = _children(TIER2, now=now)
        partials = _children(PARTIAL, now=now)
        manifests = _manifest_inventory(MANIFESTS)
        remote = _fetch_manifest(REPO_ID, PINNED_REVISION, timeout_s)
        target = _target_manifest_view(remote)
        protected = []
        volume_root = Path("/Volumes/corpdrive")
        for name in PROTECTED_VOLUME_NAMES:
            path = volume_root / name
            protected.append({"name": name, "path": str(path), "exists": path.exists(), "action": "NEVER_TOUCH"})
        stale = _stale_partials(partials, processes, now)
        report.update({
            "capacity": capacity,
            "processes": processes,
            "specimens": specimens,
            "partials": partials,
            "stale_partial_candidates": stale,
            "verified_receipts": manifests,
            "protected_paths": protected,
            "remote_manifest": remote,
            "flash_target_manifest": target,
            "checks": {
                "model_lake_mounted": capacity["mounted_directory"] is True,
                "capacity_observed": capacity["free_bytes"] is not None,
                "enough_free_capacity_for_pinned_source": capacity["free_bytes"] is not None and capacity["free_bytes"] > EXPECTED_BYTES,
                "pinned_revision_resolved_exactly": remote.get("resolved_revision") == PINNED_REVISION,
                "remote_file_count_matches_pin": remote.get("file_count") == EXPECTED_FILE_COUNT,
                "remote_bytes_match_pin": remote.get("total_declared_bytes") == EXPECTED_BYTES,
                "target_not_published_as_verified": target["whole_tree_verified"] is False,
                "no_download_performed": True,
                "protected_paths_reported": len(protected) == len(PROTECTED_VOLUME_NAMES),
            },
        })
        report["status"] = "PASSED" if all(report["checks"].values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001 - persist network/storage boundary
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        report["processes"] = processes
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    payload = causality_payload(report)
    record_modellake_causality(report, **payload)
    _write(report, str(emit) if emit is not None else None, root)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--emit")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args(argv)
    report = run_modellake_census(repo_root=args.repo_root, emit=args.emit, timeout_s=args.timeout_s)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "causality_payload", "record_modellake_causality", "records_five_fields", "run_modellake_census"]
