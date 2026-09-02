#!/usr/bin/env python3
"""DISK_TRUTH (directive §2) — what is actually on disk, with identities.

Disk state is authority.  This records the identity of the things the rest of
the campaign will reason about, so a later claim can be checked against bytes
rather than against a memory of bytes.

Records:
  * HCLI source of truth and the installed build, each with a rolled tree hash,
    and whether they agree (a drifted install is a silent-wrong-version trap)
  * the test inventory, per file
  * receipt counts by directory, and how many HCLI receipts failed
  * the named genomes the directive asks for, present or ABSENT
  * live model/runtime processes at census time
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(os.path.expanduser("~/Downloads/hawking-copy"))
RECOVERY = Path(os.path.expanduser("~/Downloads/hawking"))


def sh(cmd: str) -> str:
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True).stdout.strip()


def tree_hash(root: Path, pattern: str = "*.py") -> dict:
    if not root.is_dir():
        return {"present": False, "path": str(root)}
    files = sorted(p for p in root.rglob(pattern) if "__pycache__" not in p.parts)
    roll = hashlib.sha256()
    per = {}
    for p in files:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rel = str(p.relative_to(root))
        per[rel] = {"sha256": h, "bytes": p.stat().st_size}
        roll.update(rel.encode())
        roll.update(h.encode())
    return {"present": True, "path": str(root), "file_count": len(files),
            "tree_sha256": roll.hexdigest(), "files": per}


def count_json(d: Path) -> dict:
    if not d.is_dir():
        return {"present": False, "path": str(d)}
    files = list(d.glob("*.json"))
    return {"present": True, "path": str(d), "count": len(files),
            "empty_files": sum(1 for f in files if f.stat().st_size == 0)}


def hcli_receipt_health(d: Path) -> dict:
    if not d.is_dir():
        return {"present": False}
    rows = []
    for f in sorted(d.glob("*.json")):
        try:
            j = json.loads(f.read_text())
        except Exception:
            rows.append({"file": f.name, "parse": "FAILED", "bytes": f.stat().st_size})
            continue
        rows.append({
            "file": f.name,
            "status": j.get("status"),
            "kind": j.get("kind"),
            "ops": len(j.get("operations") or []),
            "tests": len(j.get("tests") or []),
            "has_error_field": "error" in j,
            "runtimes": j.get("runtime_count"),
            "model": os.path.basename(str(j.get("model") or "")),
            "started": (j.get("timestamps") or {}).get("started_at"),
            "finished": (j.get("timestamps") or {}).get("finished_at"),
        })
    ok = [r for r in rows if r.get("status") == "completed"]
    bad = [r for r in rows if r.get("status") not in ("completed", None)]
    return {
        "present": True, "path": str(d), "total": len(rows),
        "completed": len(ok), "not_completed": len(bad),
        "failed_without_error_field": sum(
            1 for r in bad if not r.get("has_error_field")),
        "rows": rows,
    }


GENOME_TARGETS = {
    "MachineGenome": [
        REPO / "receipts/headless/MACHINE_GENOME.json",
        REPO / ".hcli-legacy/bootstrap-director-v6/worker-equilibrium.json",
        Path(os.path.expanduser("~/.config/hcli/machine_genome.json")),
    ],
    "RuntimeGenome": [REPO / "receipts/headless/RUNTIME_GENOME.json"],
    "RepresentationGenome": [REPO / "receipts/headless/REPRESENTATION_GENOME.json"],
    "KernelGenome": [REPO / "receipts/headless/KERNEL_GENOME.json"],
    "VerificationGenome": [REPO / "receipts/headless/VERIFICATION_GENOME.json"],
    "StorageGenome/ArtifactLedger": [REPO / "receipts/headless/ARTIFACT_LEDGER.json"],
    "PerformanceLedger": [REPO / "receipts/headless/PERFORMANCE_LEDGER.jsonl"],
    "NegativeScience": [
        REPO / ".hcli-legacy/bootstrap-director-v6/negative-science.jsonl",
        RECOVERY / "workspace/campaign/odyssey/NEGATIVE_SCIENCE.json",
    ],
}


def main() -> int:
    src = tree_hash(REPO / "hcli")
    installed_link = Path(os.path.expanduser("~/.local/share/hcli/current"))
    installed = tree_hash(installed_link / "hcli") if installed_link.exists() else {"present": False}

    drift = None
    if src.get("present") and installed.get("present"):
        only_src = sorted(set(src["files"]) - set(installed["files"]))
        only_inst = sorted(set(installed["files"]) - set(src["files"]))
        differing = sorted(f for f in set(src["files"]) & set(installed["files"])
                           if src["files"][f]["sha256"] != installed["files"][f]["sha256"])
        drift = {
            "identical": src["tree_sha256"] == installed["tree_sha256"],
            "only_in_source": only_src,
            "only_in_installed": only_inst,
            "differing_files": differing,
            "meaning": ("If not identical, `hcli` on PATH is NOT running the source in this "
                        "repo. Every claim about HCLI behaviour must then name which tree it "
                        "was measured against."),
        }

    genomes = {}
    for name, cands in GENOME_TARGETS.items():
        hit = next((c for c in cands if c.exists()), None)
        genomes[name] = ({"present": True, "path": str(hit), "bytes": hit.stat().st_size,
                          "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime(hit.stat().st_mtime))}
                         if hit else
                         {"present": False, "searched": [str(c) for c in cands]})

    doc = {
        "schema": "hawking.headless.disk_truth.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary_repo": {
            "path": str(REPO),
            "head": sh(f"git -C {REPO} rev-parse HEAD"),
            "branch": sh(f"git -C {REPO} rev-parse --abbrev-ref HEAD"),
            "commit_count": sh(f"git -C {REPO} rev-list --count HEAD"),
            "dirty_entries": len(sh(f"git -C {REPO} status --porcelain").splitlines()),
            "remote": sh(f"git -C {REPO} remote -v | head -1"),
        },
        "recovery_tree": {
            "path": str(RECOVERY),
            "head": sh(f"git -C {RECOVERY} rev-parse HEAD"),
            "branch": sh(f"git -C {RECOVERY} rev-parse --abbrev-ref HEAD"),
            "commit_count": sh(f"git -C {RECOVERY} rev-list --count HEAD"),
            "role": "read-only archaeology; never a mutation target",
        },
        "hcli": {
            "entrypoint": os.path.expanduser("~/.local/bin/hcli"),
            "entrypoint_body": (open(os.path.expanduser("~/.local/bin/hcli")).read()
                                if os.path.isfile(os.path.expanduser("~/.local/bin/hcli")) else None),
            "jhcli_present": os.path.exists(os.path.expanduser("~/.local/bin/jhcli")),
            "source_of_truth": {k: v for k, v in src.items() if k != "files"},
            "installed_build": {k: v for k, v in installed.items() if k != "files"},
            "installed_symlink_target": (os.path.realpath(installed_link)
                                         if installed_link.exists() else None),
            "source_vs_installed": drift,
            "source_files": src.get("files", {}),
        },
        "tests": {
            "hcli_package_tests": tree_hash(REPO / "hcli/tests", "test_*.py"),
            "headless_tools": tree_hash(REPO / "tools/headless"),
        },
        "receipts": {
            "hcli_project_receipts": hcli_receipt_health(REPO / ".hcli/receipts"),
            "headless_receipts": count_json(REPO / "receipts/headless"),
            "recovery_ascent_2026_08_18": count_json(RECOVERY / "receipts/ascent-2026-08-18"),
            "recovery_ascent_2026_08_16": count_json(RECOVERY / "receipts/ascent-2026-08-16"),
            "recovery_odyssey_i": count_json(RECOVERY / "receipts/odyssey-i"),
        },
        "director_state": {
            "worker_equilibrium": (
                json.loads((REPO / ".hcli-legacy/bootstrap-director-v6/worker-equilibrium.json").read_text())
                if (REPO / ".hcli-legacy/bootstrap-director-v6/worker-equilibrium.json").exists() else None),
            "negative_science_lines": len(
                (REPO / ".hcli-legacy/bootstrap-director-v6/negative-science.jsonl").read_text().splitlines())
            if (REPO / ".hcli-legacy/bootstrap-director-v6/negative-science.jsonl").exists() else 0,
            "epoch_run_dirs": len(list((REPO / ".hcli-legacy/bootstrap-director-v6/runs").glob("*")))
            if (REPO / ".hcli-legacy/bootstrap-director-v6/runs").is_dir() else 0,
        },
        "genomes": genomes,
        "live_processes_at_census": {
            "llama_server": sh("ps -eo pid,command | grep llama-server | grep -v grep"),
            "mlx": sh("ps -eo pid,command | grep -i mlx | grep -v grep"),
            "hcli": sh("ps -eo pid,command | grep -E '[h]cli' | grep -v grep"),
            "odyssey": sh("ps -eo pid,command | grep -E '[o]dyssey' | grep -v grep"),
        },
        "disk": sh("df -h /System/Volumes/Data | tail -1"),
    }

    out = REPO / "receipts/headless"
    out.mkdir(parents=True, exist_ok=True)
    (out / "DISK_TRUTH.json").write_text(json.dumps(doc, indent=1))

    h = doc["hcli"]
    r = doc["receipts"]["hcli_project_receipts"]
    print("=== DISK TRUTH ===")
    print(f"  repo HEAD            {doc['primary_repo']['head'][:12]} "
          f"({doc['primary_repo']['dirty_entries']} dirty entries)")
    print(f"  hcli source          {h['source_of_truth'].get('file_count')} files "
          f"tree={str(h['source_of_truth'].get('tree_sha256'))[:12]}")
    print(f"  hcli installed       {h['installed_build'].get('file_count')} files "
          f"tree={str(h['installed_build'].get('tree_sha256'))[:12]}")
    print(f"  source == installed  {drift['identical'] if drift else 'n/a'}")
    if drift and not drift["identical"]:
        print(f"    differing: {drift['differing_files'][:8]}")
    print(f"  jhcli present        {h['jhcli_present']}")
    if r.get("present"):
        print(f"  hcli receipts        {r['total']} total, {r['completed']} completed, "
              f"{r['not_completed']} not completed, "
              f"{r['failed_without_error_field']} failures with NO error field")
    print("  genomes:")
    for k, v in doc["genomes"].items():
        print(f"    {k:<26} {'PRESENT' if v['present'] else 'ABSENT'}"
              + (f"  {v['path']}" if v["present"] else ""))
    print(f"\n-> {out/'DISK_TRUTH.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
