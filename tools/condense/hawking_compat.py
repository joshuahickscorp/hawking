#!/usr/bin/env python3.12
"""hawking compat - the tiny stable compatibility surface for the sealed legacy capsule (Section 4/8).

Replaces the hundreds of retired doctor_v5 campaign modules that used to sit in the active tree. It
does not carry their source; it resolves the sealed, content-addressed
`hawking-legacy-runtime-capsule-v1`, verifies it, hydrates it into a path-confined temp dir, runs one
pinned legacy command, and records a compatibility receipt. Historical reproduction lives entirely in
the capsule (Section 19), not in the active kernel.

Design budget: this is the whole active compatibility implementation and stays far under the 2,000
LOC ceiling (data + one executor). No network is required once the capsule archive is cached.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

MANIFEST = Path("packs/hawking-legacy-runtime-capsule-v1.json")
RECEIPTS = Path("reports/condense/gravity_forge/condensation/compat_receipts")


def _manifest() -> dict:
    if not MANIFEST.exists():
        raise SystemExit("capsule manifest missing: packs/hawking-legacy-runtime-capsule-v1.json")
    return json.loads(MANIFEST.read_text())


def _archive_path(m: dict) -> Path:
    return Path(m["offline_cache"]) / "capsule.tar.gz"


def verify(m: dict | None = None) -> dict:
    """Verify the capsule archive against the sealed sha256 (offline, no network)."""
    m = m or _manifest()
    ap = _archive_path(m)
    if not ap.exists():
        return {"ok": False, "reason": f"capsule archive absent at {ap}; hydrate from source_commit "
                f"{m['source_commit'][:12]} via `git archive`"}
    got = hashlib.sha256(ap.read_bytes()).hexdigest()
    return {"ok": got == m["archive_sha256"], "archive": str(ap),
            "expected": m["archive_sha256"][:16], "got": got[:16]}


def hydrate(m: dict, dest: Path) -> Path:
    """Extract the capsule into a path-confined directory. Rejects unsafe (absolute / ..) members."""
    ap = _archive_path(m)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ap, "r:gz") as tar:
        for member in tar.getmembers():
            p = Path(member.name)
            if p.is_absolute() or ".." in p.parts:
                raise SystemExit(f"unsafe capsule member rejected: {member.name}")
        tar.extractall(dest)                                   # members validated path-confined above
    return dest


def _receipt(action: str, payload: dict) -> None:
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    doc = {"schema": "hawking.compat_receipt.v1", "action": action, **payload}
    doc["receipt_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in doc.items() if k != "receipt_sha256"},
                                                      sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    (RECEIPTS / f"{action}_{doc['receipt_sha256'][:12]}.json").write_text(json.dumps(doc, indent=2, sort_keys=True))


def list_ids(m: dict) -> list[str]:
    """Legacy command ids = capsuled Python modules with their own __main__, keyed by basename."""
    ids = []
    for e in m.get("contents", []):
        b = os.path.basename(e["path"])
        if b.endswith(".py") and "test" not in b:
            ids.append(b[:-3])
    return sorted(ids)


def run(legacy_id: str, extra: list[str]) -> int:
    m = _manifest()
    v = verify(m)
    if not v["ok"]:
        print(json.dumps({"verify": v}, indent=2))
        return 2
    entry = next((e for e in m["contents"] if os.path.basename(e["path"]) == f"{legacy_id}.py"), None)
    if entry is None:
        print(f"unknown legacy id {legacy_id!r}; try `hawking compat list`")
        return 2
    with tempfile.TemporaryDirectory(prefix="hawking-capsule-") as td:
        root = hydrate(m, Path(td))
        modpath = root / entry["path"]
        cmd = [sys.executable, str(modpath), *extra]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(root))
        _receipt("compat_run", {"legacy_id": legacy_id, "source_commit": m["source_commit"],
                                "archive_sha256": m["archive_sha256"], "returncode": proc.returncode,
                                "seconds": round(time.time() - t0, 2),
                                "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return proc.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hawking compat", description="Run pinned legacy commands from the sealed capsule.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    sub.add_parser("list")
    r = sub.add_parser("run"); r.add_argument("legacy_id"); r.add_argument("extra", nargs="*")
    args = ap.parse_args(argv)
    if args.cmd == "verify":
        print(json.dumps(verify(), indent=2)); return 0
    if args.cmd == "list":
        m = _manifest(); ids = list_ids(m)
        print(f"{len(ids)} legacy command ids in {m['capsule']} (source {m['source_commit'][:12]}):")
        for i in ids[:40]:
            print(f"  {i}")
        if len(ids) > 40:
            print(f"  ... and {len(ids) - 40} more")
        return 0
    if args.cmd == "run":
        return run(args.legacy_id, args.extra)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
