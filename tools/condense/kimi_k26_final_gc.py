#!/usr/bin/env python3.12
"""Dependency-safe garbage collection for generated Kimi K2.6 campaign bloat."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


FLOOR_BYTES = 5 * 1024**3
REPO = Path(__file__).resolve().parents[2]
RUNTIME = Path.home() / "Library/Application Support/Hawking/KimiK26"
XET_LOG_ROOT = Path.home() / ".cache/huggingface/xet/logs"
LEDGER_NAME = "KIMI_K26_FINAL_GC_LEDGER.jsonl"
REPORT_NAME = "KIMI_K26_FINAL_STORAGE_REPORT.md"
PROTECTED_ROOTS = (
    Path.home() / ".cache/huggingface/hub/models--moonshotai--Kimi-K2.6",
    Path.home() / "Downloads/mop",
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z",
    )


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def seal(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
    return {**unsigned, "seal_sha256": hashlib.sha256(canonical(unsigned)).hexdigest()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_record(record: dict[str, Any]) -> dict[str, Any]:
    value = seal(record)
    line = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    for root in (REPO, RUNTIME):
        path = root / LEDGER_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    return value


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def is_protected(path: Path) -> bool:
    resolved = path.resolve(strict=True)
    return any(resolved == root.resolve(strict=True) or
               resolved.is_relative_to(root.resolve(strict=True)) for root in PROTECTED_ROOTS)


def active_mappings(path: Path) -> list[str]:
    result = subprocess.run(["/usr/sbin/lsof", "--", str(path)], text=True,
                            capture_output=True, check=False)
    return result.stdout.splitlines()[1:] if result.returncode == 0 else []


def candidate_logs() -> list[Path]:
    if not XET_LOG_ROOT.is_dir():
        return []
    candidates = []
    for path in sorted(XET_LOG_ROOT.glob("xet_20260721T*.log")):
        if not path.is_file() or path.is_symlink():
            continue
        with path.open("rb") as handle:
            marker = handle.read(min(path.stat().st_size, 32 * 1024 * 1024))
        if b"Kimi-K2.6" in marker or b"moonshotai" in marker:
            candidates.append(path)
    return candidates


def run(repo: Path, *, execute: bool) -> dict[str, Any]:
    before = shutil.disk_usage(Path.home()).free
    process_snapshot = subprocess.run(
        ["/bin/zsh", "-lc", "ps -axo pid,command | rg 'hf download|huggingface|xet' | rg -v 'rg '"],
        text=True, capture_output=True, check=False,
    ).stdout.splitlines()
    candidates = []
    for path in candidate_logs():
        if is_protected(path):
            raise RuntimeError(f"protected path entered GC candidate set: {path}")
        mappings = active_mappings(path)
        if mappings:
            raise RuntimeError(f"active mapping blocks GC: {path}: {mappings}")
        stat = path.stat()
        candidates.append({
            "path": str(path.resolve(strict=True)),
            "logical_bytes": stat.st_size,
            "allocated_bytes": stat.st_blocks * 512,
            "sha256": sha256_file(path),
            "reason": "ROTATED_KIMI_XET_TRANSPORT_LOG",
            "reproducibility_status": (
                "NO_SCIENTIFIC_DEPENDENCY; source manifest and payload evidence are sealed elsewhere"
            ),
            "active_mappings": mappings,
            "queued_dependency": False,
        })
    plan = append_record({
        "schema": "hawking.kimi_k26.final_gc.v1",
        "event": "GC_PLAN",
        "at": now(),
        "execute_requested": execute,
        "disk_floor_bytes": FLOOR_BYTES,
        "free_before_bytes": before,
        "active_hf_or_xet_processes": process_snapshot,
        "candidates": candidates,
        "planned_logical_bytes": sum(item["logical_bytes"] for item in candidates),
        "protected_roots": [str(path.resolve(strict=True)) for path in PROTECTED_ROOTS],
        "replacement_evidence": {
            "source_manifest": str(RUNTIME / "KIMI_K26_OFFICIAL_MANIFEST.json"),
            "source_verification": str(RUNTIME / "KIMI_K26_SOURCE_VERIFICATION.json"),
            "one_copy_receipt": str(RUNTIME / "KIMI_K26_ONE_COPY_RECEIPT.json"),
        },
    })
    deleted = []
    if execute:
        for item in candidates:
            path = Path(item["path"])
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"candidate changed after planning: {path}")
            if path.stat().st_size != item["logical_bytes"] or sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"candidate hash/size changed after planning: {path}")
            if active_mappings(path):
                raise RuntimeError(f"candidate became active after planning: {path}")
            path.unlink()
            deleted.append(item)
            append_record({
                "schema": "hawking.kimi_k26.final_gc.v1",
                "event": "GC_DELETE",
                "at": now(),
                "plan_seal_sha256": plan["seal_sha256"],
                **item,
                "deleted": not path.exists(),
                "recoverability": "not retained; reproducibility unaffected by transport-log deletion",
            })
    after = shutil.disk_usage(Path.home()).free
    summary = append_record({
        "schema": "hawking.kimi_k26.final_gc.v1",
        "event": "GC_COMPLETE" if execute else "GC_DRY_RUN_COMPLETE",
        "at": now(),
        "plan_seal_sha256": plan["seal_sha256"],
        "disk_floor_bytes": FLOOR_BYTES,
        "free_before_bytes": before,
        "free_after_bytes": after,
        "filesystem_free_delta_bytes": after - before,
        "deleted_logical_bytes": sum(item["logical_bytes"] for item in deleted),
        "deleted_allocated_bytes": sum(item["allocated_bytes"] for item in deleted),
        "deleted_count": len(deleted),
        "floor_green_after": after > FLOOR_BYTES,
        "preserved": [
            "sole moonshotai/Kimi-K2.6 resident source",
            "MOP",
            "credentials",
            "current best P1 payload",
            "sealed laws/manifests/reports",
            "all reusable nonlinear/oracle NPZ captures",
        ],
    })
    lines = [
        "# Kimi K2.6 Final Storage Report", "",
        f"- Completed: `{summary['at']}`",
        f"- Mode: `{'EXECUTE' if execute else 'DRY_RUN'}`",
        f"- Hard floor: `{FLOOR_BYTES}` bytes (`5 GiB`)",
        f"- Free before/after: `{before}` / `{after}` bytes",
        f"- Files deleted: `{len(deleted)}`",
        f"- Logical bytes reclaimed: `{summary['deleted_logical_bytes']}`",
        f"- Allocated bytes reclaimed by ledger: `{summary['deleted_allocated_bytes']}`",
        f"- Filesystem free-space delta: `{summary['filesystem_free_delta_bytes']}`",
        f"- Floor green after: `{summary['floor_green_after']}`", "",
        "## Deleted paths", "",
    ]
    if deleted:
        lines.extend(
            f"- `{item['path']}` — {item['logical_bytes']} bytes — `{item['sha256']}` — {item['reason']}"
            for item in deleted
        )
    else:
        lines.append("- None.")
    lines.extend(["", "## Preserved", ""])
    lines.extend(f"- {item}" for item in summary["preserved"])
    lines.extend(["", "## Evidence", "",
                  f"- Plan seal: `{plan['seal_sha256']}`",
                  f"- Completion seal: `{summary['seal_sha256']}`", ""])
    for root in (repo, RUNTIME):
        atomic_text(root / REPORT_NAME, "\n".join(lines))
    return {"plan": plan, "summary": summary, "deleted": deleted}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run(args.repo.resolve(strict=True), execute=args.execute)
    print(json.dumps({
        "status": "PASS", "event": result["summary"]["event"],
        "deleted_count": result["summary"]["deleted_count"],
        "deleted_logical_bytes": result["summary"]["deleted_logical_bytes"],
        "free_after_bytes": result["summary"]["free_after_bytes"],
        "seal_sha256": result["summary"]["seal_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
