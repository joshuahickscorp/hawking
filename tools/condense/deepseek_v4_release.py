#!/usr/bin/env python3.12
"""Release the terminal DeepSeek-V4-Flash source.

The cascade refuted the independent-per-layer functional paradigm on DeepSeek, so the raw
159.6 GB fp4 source is terminal for this closure. The validated forward is the official
transformers modeling plus a name+dequant map, not the raw bytes, and the rehydration
receipt carries a git-lfs sha256 per shard, so the body is re-fetchable byte-exact. This
gates the release the same way the GLM body was: verify every preservation and safety
condition, then delete only the source root and seal a reclaimed-byte receipt.

    gate      read-only; print every gate, exit non-zero if any is red
    release   re-run the gate; refuse unless all green and --confirm; delete; seal receipt
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONDENSE = Path(__file__).resolve().parent
REPO_ROOT = CONDENSE.parents[1]
GB = 10 ** 9
GIB = 1024 ** 3

SUPPORT = Path(os.environ.get(
    "DEEPSEEK_V4_SUPPORT_ROOT",
    str(Path.home() / "Library/Application Support/Hawking/DeepSeekV4Flash")))
SOURCE = SUPPORT / "source"
META = SUPPORT / "meta"

REPO = "deepseek-ai/DeepSeek-V4-Flash"
REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"

REHYDRATION = REPO_ROOT / "DEEPSEEK_V4_FLASH_REHYDRATION_RECEIPT.json"
ADMISSION = REPO_ROOT / "DEEPSEEK_V4_FLASH_SOURCE_ADMISSION.json"
READINESS = REPO_ROOT / "DEEPSEEK_V4_FLASH_RELEASE_READINESS.json"
RECEIPT = REPO_ROOT / "DEEPSEEK_V4_FLASH_RELEASE_RECEIPT.json"

# Evidence that must outlive the source.
PRESERVED = [
    REPO_ROOT / "DEEPSEEK_V4_FLASH_CASCADE_DECISION.json",
    REPO_ROOT / "DEEPSEEK_V4_FLASH_CONTRACTION_VERDICT.json",
    REPO_ROOT / "DEEPSEEK_V4_BYTE_AUCTION.json",
    REPO_ROOT / "DEEPSEEK_V4_RUNTIME_ACCOUNTING.json",
    CONDENSE / "deepseek_v4_reference.py",
    CONDENSE / "deepseek_v4_moe.py",
    CONDENSE / "deepseek_v4_primitive_parity.py",
    CONDENSE / "deepseek_v4_cascade.py",
    REPO_ROOT / "reports/condense/deepseek_v4_flash/DEEPSEEK_V4_PRIMITIVE_PARITY.json",
]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _process_scan() -> dict:
    target = str(SOURCE)
    matches = []
    lsof = shutil.which("lsof")
    lsof_ok = False
    if lsof:
        try:
            out = subprocess.run([lsof, "--", target], capture_output=True, text=True,
                                 timeout=60)
            lines = [ln for ln in out.stdout.splitlines()[1:] if ln.strip()]
            lsof_ok = True
            matches += [("lsof", ln[:150]) for ln in lines]
        except Exception:  # noqa: BLE001
            pass
    try:
        out = subprocess.run(["ps", "-Axww", "-o", "pid=,command="],
                             capture_output=True, text=True, timeout=60)
        hits = [ln for ln in out.stdout.splitlines()
                if target in ln and "deepseek_v4_release" not in ln]
        matches += [("argv", ln.strip()[:150]) for ln in hits]
    except Exception:  # noqa: BLE001
        pass
    return {"clean": not matches, "any_probe_ran": lsof_ok, "matches": matches[:5]}


def gate() -> dict:
    gates = {}

    gates["g01_source_present"] = {
        "green": SOURCE.exists() and len(list(SOURCE.glob("*.safetensors"))) > 0,
        "detail": f"{len(list(SOURCE.glob('*.safetensors'))) if SOURCE.exists() else 0} shards"}

    if REHYDRATION.exists():
        rec = json.loads(REHYDRATION.read_text())
        n = len(rec.get("per_file_sha256", {}))
        gates["g02_rehydration_receipt"] = {
            "green": rec.get("revision") == REVISION and n >= 46,
            "detail": f"{n} shards carry a git-lfs sha256 at the pinned revision"}
    else:
        gates["g02_rehydration_receipt"] = {"green": False, "detail": "missing"}

    missing = [str(p) for p in PRESERVED if not p.exists()]
    gates["g03_evidence_preserved"] = {
        "green": not missing,
        "detail": f"{len(PRESERVED) - len(missing)}/{len(PRESERVED)} evidence artifacts present",
        "missing": missing}

    parity = REPO_ROOT / "reports/condense/deepseek_v4_flash/DEEPSEEK_V4_PRIMITIVE_PARITY.json"
    gates["g04_forward_validated"] = {
        "green": parity.exists() and json.loads(parity.read_text()).get("all_match", False),
        "detail": "primitive parity vs official transformers reference is green"}

    scan = _process_scan()
    gates["g05_no_process_maps_source"] = {
        "green": scan["clean"], "detail": "no live process opens or names the source",
        "scan": scan}

    gates["g06_isolation"] = {
        "green": (SOURCE.name == "source" and SUPPORT in SOURCE.parents
                  and REPO_ROOT not in SOURCE.parents
                  and Path.home() / "Downloads" / "mop" != SOURCE),
        "detail": "source root is isolated from repo and MOP"}

    all_green = all(g["green"] for g in gates.values())
    report = {"schema": "hawking.deepseek_v4.release_readiness.v1", "evaluated_at": _now(),
              "source_root": str(SOURCE), "repo": REPO, "revision": REVISION,
              "gates": gates, "all_green": all_green}
    READINESS.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def release(confirm: bool) -> dict:
    report = gate()
    if not report["all_green"]:
        red = [n for n, g in report["gates"].items() if not g["green"]]
        raise SystemExit(f"release refused: red gates {red}")
    if not confirm:
        raise SystemExit("release refused: pass --confirm")
    if not SOURCE.exists():
        raise SystemExit("source already released")

    before = sum(p.stat().st_size for p in SOURCE.rglob("*") if p.is_file())
    shards = len(list(SOURCE.glob("*.safetensors")))
    free_before = shutil.disk_usage(str(SUPPORT)).free
    shutil.rmtree(SOURCE)
    free_after = shutil.disk_usage(str(SUPPORT)).free

    receipt = {
        "schema": "hawking.deepseek_v4.release_receipt.v1", "released_at": _now(),
        "deleted_path": str(SOURCE), "deleted_exists_after": SOURCE.exists(),
        "shards_deleted": shards, "reclaimed_bytes": before,
        "reclaimed_gb": round(before / GB, 1), "reclaimed_gib": round(before / GIB, 1),
        "free_after_gib": round(free_after / GIB, 1),
        "free_delta_gb": round((free_after - free_before) / GB, 1),
        "rollback": f"re-fetch {REPO} @ {REVISION}, verify against the rehydration receipt",
        "readiness_seal": hashlib.sha256(READINESS.read_bytes()).hexdigest(),
        "meta_retained": META.exists(),
        "reason": "terminal: cascade refuted the independent-per-layer functional paradigm; "
                  "the validated forward is the official modeling plus a map, not the raw bytes",
        "preserved": [p.name for p in PRESERVED],
    }
    receipt["seal_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in receipt.items() if k != "seal_sha256"},
                   sort_keys=True).encode()).hexdigest()
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if command == "gate":
        report = gate()
        for name, g in report["gates"].items():
            print(f"  {'OK ' if g['green'] else 'RED'} {name}: {g['detail']}")
        print(f"\nall_green={report['all_green']}")
        raise SystemExit(0 if report["all_green"] else 1)
    if command == "release":
        print(json.dumps(release("--confirm" in sys.argv), indent=2))
    else:
        raise SystemExit(f"unknown command: {command}")
