#!/usr/bin/env python3.12
"""Substrate reproduction: real integrity check of the Math-Preserve artifact.

This is NOT a "manifest reads itself" check. Externally sealed expected hashes
(from the campaign substrate facts) are compared against freshly computed
sha256 of the on-disk index and allocation manifest. Each shard body is then
hashed and compared against the index's recorded shard_body_sha256 map.

The full 92 GB scan is incremental and resumable. Partial runs prove the
machinery; remaining work is estimated from measured throughput.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.odyssey._paths import (
    EXPECTED_BYTES,
    EXPECTED_DECISION_COUNT,
    EXPECTED_INDEX_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SHARD_COUNT,
    MATH_ARTIFACT,
    T0_STATE,
)

PREFIX_STRUCT = "<8sIQ"
PREFIX_BYTES = struct.calcsize(PREFIX_STRUCT)
PROGRESS_PATH = T0_STATE / "substrate_verify_progress.json"
SCHEMA = "hawking.odyssey.t0.substrate_verify.v1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _body_sha256(path: Path) -> str:
    """Hash only the tensor body (after the fixed prefix + JSON header)."""
    with path.open("rb") as fh:
        prefix = fh.read(PREFIX_BYTES)
        if len(prefix) != PREFIX_BYTES:
            raise ValueError(f"{path.name}: shorter than gravity prefix")
        magic, _version, header_length = struct.unpack(PREFIX_STRUCT, prefix)
        if magic != b"GRAVITY\x00":
            raise ValueError(f"{path.name}: not a .gravity shard")
        fh.seek(PREFIX_BYTES + header_length)
        h = hashlib.sha256()
        while True:
            chunk = fh.read(8 << 20)
            if not chunk:
                break
            h.update(chunk)
        return h.hexdigest()


def _load_progress() -> dict[str, Any]:
    if PROGRESS_PATH.is_file():
        return json.loads(PROGRESS_PATH.read_text())
    return {
        "schema": "hawking.odyssey.t0.substrate_progress.v1",
        "verified": {},
        "failed": {},
        "bytes_verified": 0,
        "seconds": 0.0,
    }


def _save_progress(progress: dict[str, Any]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(progress, indent=1, sort_keys=True) + "\n")


def static_checks(artifact: Path = MATH_ARTIFACT) -> dict[str, Any]:
    """Index/manifest hashes, shard and decision counts. Cheap; always run."""
    index_path = artifact / "model.gravity.index.json"
    manifest_path = artifact / "PROMETHEUS_MATH_ALLOCATION_MANIFEST.json"
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    if not artifact.is_dir():
        add("artifact_present", False, f"missing: {artifact}")
        return {"ok": False, "checks": checks, "artifact": str(artifact)}

    add("artifact_present", True, str(artifact))

    if not index_path.is_file():
        add("index_present", False, str(index_path))
        return {"ok": False, "checks": checks, "artifact": str(artifact)}
    add("index_present", True)

    index_hash = _sha256_file(index_path)
    add(
        "index_sha256",
        index_hash == EXPECTED_INDEX_SHA256,
        f"observed={index_hash} expected={EXPECTED_INDEX_SHA256}",
    )

    if not manifest_path.is_file():
        add("manifest_present", False, str(manifest_path))
    else:
        add("manifest_present", True)
        man_hash = _sha256_file(manifest_path)
        add(
            "manifest_sha256",
            man_hash == EXPECTED_MANIFEST_SHA256,
            f"observed={man_hash} expected={EXPECTED_MANIFEST_SHA256}",
        )

    index = json.loads(index_path.read_text())
    shards = list(index.get("shards") or [])
    add(
        "shard_count",
        len(shards) == EXPECTED_SHARD_COUNT and int(index.get("shard_count", -1)) == EXPECTED_SHARD_COUNT,
        f"observed={len(shards)} expected={EXPECTED_SHARD_COUNT}",
    )

    decision_count = int(index.get("tensor_count") or len(index.get("weight_map") or {}))
    add(
        "decision_count",
        decision_count == EXPECTED_DECISION_COUNT,
        f"observed={decision_count} expected={EXPECTED_DECISION_COUNT}",
    )

    body_map = index.get("shard_body_sha256") or {}
    add(
        "shard_body_map_complete",
        len(body_map) == EXPECTED_SHARD_COUNT,
        f"map_entries={len(body_map)}",
    )

    present = sum(1 for name in shards if (artifact / name).is_file())
    add(
        "shards_present_on_disk",
        present == EXPECTED_SHARD_COUNT,
        f"present={present}/{EXPECTED_SHARD_COUNT}",
    )

    total_bytes = sum((artifact / name).stat().st_size for name in shards if (artifact / name).is_file())
    # Exact sealed byte total is for the whole artifact directory content as recorded.
    # We report observed shard-file sum separately from the sealed fact.
    add(
        "shard_bytes_sum_recorded",
        True,
        f"shard_file_bytes={total_bytes} sealed_artifact_bytes={EXPECTED_BYTES}",
    )

    ok = all(c["ok"] for c in checks if c["check"] != "shard_bytes_sum_recorded")
    return {
        "ok": ok,
        "checks": checks,
        "artifact": str(artifact),
        "index_sha256": index_hash,
        "shards": shards,
        "shard_body_sha256": body_map,
        "decision_count": decision_count,
    }


def verify_shards(
    *,
    max_shards: int | None = None,
    max_bytes: int | None = None,
    resume: bool = True,
    artifact: Path = MATH_ARTIFACT,
) -> dict[str, Any]:
    """Hash shard bodies against the index map. Resumable.

    Limits (max_shards / max_bytes) let a smoke run prove the path without
    scanning all 92 GB when the machine is under load.
    """
    static = static_checks(artifact)
    if not static["ok"]:
        return {
            "schema": SCHEMA,
            "status": "FAIL",
            "static": static,
            "shard_verification": None,
        }

    progress = _load_progress() if resume else {
        "schema": "hawking.odyssey.t0.substrate_progress.v1",
        "verified": {},
        "failed": {},
        "bytes_verified": 0,
        "seconds": 0.0,
    }

    body_map: dict[str, str] = static["shard_body_sha256"]
    shards: list[str] = static["shards"]
    newly = 0
    bytes_this_run = 0
    t0 = time.perf_counter()

    for name in shards:
        if max_shards is not None and newly >= max_shards:
            break
        if max_bytes is not None and bytes_this_run >= max_bytes:
            break
        if name in progress["verified"] and progress["verified"][name] == body_map.get(name):
            continue

        path = artifact / name
        size = path.stat().st_size
        try:
            observed = _body_sha256(path)
        except Exception as exc:  # noqa: BLE001 — record and continue
            progress["failed"][name] = str(exc)
            _save_progress(progress)
            continue

        expected = body_map.get(name)
        if observed != expected:
            progress["failed"][name] = f"body_sha256 mismatch observed={observed} expected={expected}"
        else:
            progress["verified"][name] = observed
            progress["bytes_verified"] = int(progress.get("bytes_verified", 0)) + size
            newly += 1
            bytes_this_run += size
            progress["failed"].pop(name, None)
        _save_progress(progress)

    elapsed = time.perf_counter() - t0
    progress["seconds"] = float(progress.get("seconds", 0.0)) + elapsed
    _save_progress(progress)

    n_ok = len(progress["verified"])
    n_fail = len(progress["failed"])
    remaining = [s for s in shards if s not in progress["verified"]]
    remaining_bytes = sum((artifact / s).stat().st_size for s in remaining if (artifact / s).is_file())
    rate = bytes_this_run / elapsed if elapsed > 0 and bytes_this_run else None
    eta_s = (remaining_bytes / rate) if rate else None

    if n_fail:
        status = "FAIL"
    elif not remaining:
        status = "PASS"
    else:
        status = "PARTIAL"

    return {
        "schema": SCHEMA,
        "status": status,
        "static": {k: static[k] for k in ("ok", "checks", "artifact", "index_sha256", "decision_count")},
        "shard_verification": {
            "shards_verified": n_ok,
            "shards_failed": n_fail,
            "shards_remaining": len(remaining),
            "bytes_verified_total": progress["bytes_verified"],
            "bytes_this_run": bytes_this_run,
            "seconds_this_run": elapsed,
            "throughput_bytes_per_s": rate,
            "eta_seconds_remaining": eta_s,
            "failed": dict(progress["failed"]),
            "progress_path": str(PROGRESS_PATH),
            "note": (
                "Each verified shard body was re-hashed from disk and compared to "
                "model.gravity.index.json's shard_body_sha256 entry. Index and "
                "manifest files were re-hashed and compared to sealed campaign facts."
            ),
        },
        "what_was_checked": [
            "index file sha256 vs sealed EXPECTED_INDEX_SHA256",
            "allocation manifest sha256 vs sealed EXPECTED_MANIFEST_SHA256",
            "shard_count == 282",
            "tensor_count/decision_count == 59585",
            f"{n_ok} shard bodies re-hashed vs index map",
        ],
        "what_was_skipped": (
            []
            if not remaining
            else [
                f"{len(remaining)} shard bodies not yet hashed "
                f"(~{remaining_bytes} bytes remaining; resumable via progress file)"
            ]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--max-bytes", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    result = verify_shards(
        max_shards=args.max_shards,
        max_bytes=args.max_bytes,
        resume=not args.no_resume,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0 if result["status"] in ("PASS", "PARTIAL") and result["static"]["ok"] else 1


if __name__ == "__main__":
    # Allow `python3 tools/odyssey/substrate_verify.py` from repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
