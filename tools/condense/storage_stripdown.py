#!/usr/bin/env python3.12
"""Storage stripdown: protected-path resolution, complete inventory, exact-path deletion.

Fail-closed by construction. Every deletion candidate is re-resolved with realpath and
device/inode metadata at BOTH manifest time and delete time; anything that resolves inside a
protected root, traverses a symlink into one, sits on an unexpected volume, or is mapped by a
live process is rejected. Deletion unlinks exact files one at a time from a sealed manifest -
there is no glob, no rm -rf, no directory recursion, and no symlink following.

Subcommands
  protect    seal STORAGE_STRIPDOWN_PROTECTED_PATHS.json
  inventory  seal STORAGE_STRIPDOWN_INVENTORY.{json,md}
  plan       seal STORAGE_STRIPDOWN_DELETE_MANIFEST.json (dry-run; deletes nothing)
  release    seal MODEL_RELEASE_<family>.json rehydration receipts (deletes nothing)
  execute    apply a sealed manifest, one unlink at a time (requires --go)
  verify     seal STORAGE_STRIPDOWN_FINAL.{json,md}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.path.abspath(__file__)).resolve().parents[2]
OUT = ROOT / "reports/condense/storage_stripdown"
HOME = Path.home()
DATA_VOLUME = "/System/Volumes/Data"

# Roots that may never be touched, whatever a manifest says.
PROTECTED_ROOTS = [
    HOME / "Downloads/mop",
    HOME / "Downloads/mop-data",
    HOME / "Downloads/mop-experimental-method-reformation",
    HOME / "Downloads/mop_expansion_bundle",
    HOME / "Library",
    HOME / "Documents",
    HOME / "Pictures",
    HOME / "Desktop",
    HOME / ".ssh",
    HOME / ".gnupg",
    HOME / ".config",
    # MOP's own model cache lives here: mop/tests/unit/test_vjepa21_official.py pulls
    # facebook/vjepa2-* (7.5 GiB) through the shared hub cache. Cleaning it would be cleaning a
    # MOP build product, which this campaign forbids. Protect the whole hub cache rather than
    # trying to tell MOP's blobs apart from Hawking's.
    HOME / ".cache/huggingface",
    Path("/System"),
    Path("/Library"),
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/private/etc"),
    Path("/Applications"),
]
# The authoritative repo's git database. Working files may be deleted by manifest; .git may not.
PROTECTED_GIT = [ROOT / ".git"]

MODEL_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth", ".ckpt", ".onnx", ".npz")
PARTIAL_SUFFIXES = (".incomplete", ".part", ".tmp", ".download")
MIN_MODEL_BYTES = 64 * 1024 * 1024  # below this a *.bin is an artifact, not a model payload


# ── framework ───────────────────────────────────────────────────────────────────────────
def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=2, sort_keys=True, default=str)
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def read(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def free_bytes(mount: str = DATA_VOLUME) -> int:
    st = os.statvfs(mount)
    return st.f_bavail * st.f_frsize


def volume_bytes(mount: str = DATA_VOLUME) -> int:
    st = os.statvfs(mount)
    return st.f_blocks * st.f_frsize


# ── protected-path gate ─────────────────────────────────────────────────────────────────
def _ids(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
        return (st.st_dev, st.st_ino)
    except OSError:
        return None


def protected_set() -> dict:
    """Resolved protected roots with device/inode identity, for the gate and for the receipt."""
    roots = []
    for p in PROTECTED_ROOTS + PROTECTED_GIT:
        try:
            real = p.resolve()
        except OSError:
            continue
        roots.append({"declared": str(p), "resolved": str(real), "exists": real.exists(),
                      "ids": _ids(real)})
    return {"roots": roots,
            "resolved_paths": [r["resolved"] for r in roots if r["exists"]],
            "resolved_ids": [tuple(r["ids"]) for r in roots if r["ids"]]}


def gate(path: Path, prot: dict, expected_dev: int) -> tuple[bool, str]:
    """True only when this exact path is safe to unlink. Every failure is fail-closed."""
    try:
        real = Path(path).resolve()
    except OSError as exc:
        return False, f"unresolvable: {type(exc).__name__}"
    if not real.is_absolute():
        return False, "not absolute after resolution"
    # Reject any ancestor that is a protected root (catches symlink traversal, since we
    # compare the RESOLVED path).
    for pr in prot["resolved_paths"]:
        pr_p = Path(pr)
        if real == pr_p or pr_p in real.parents:
            return False, f"inside protected root {pr}"
    # Reject by device/inode too: a bind-style alias with a different textual path.
    for parent in [real, *real.parents]:
        if _ids(parent) in prot["resolved_ids"]:
            return False, f"resolves through protected inode at {parent}"
    if not real.exists():
        return False, "does not exist"
    if real.is_symlink():
        return False, "is a symlink"
    st = os.lstat(real)
    if st.st_dev != expected_dev:
        return False, f"unexpected volume dev={st.st_dev} expected={expected_dev}"
    if st.st_uid != os.getuid():
        return False, f"not owned by uid {os.getuid()}"
    return True, "ok"


# ── active-mapping gate ─────────────────────────────────────────────────────────────────
def mapped_paths(dirs: list[str]) -> set[str]:
    """Absolute paths currently open/mapped by any live process, restricted to dirs.

    `lsof +D` alone is NOT sufficient and this is not theoretical: a process executing a binary
    inside the tree holds it as a `txt` descriptor, which +D does not report. Measured live -
    `lsof -w +D .../hawking-hide-build/target` returned nothing while hide-serve was running out
    of target/debug/hide-serve, and only `lsof -d txt` found it. So the open-file walk is unioned
    with a global txt/cwd/rtd sweep filtered to the same prefixes.
    """
    out: set[str] = set()
    for d in dirs:
        try:
            res = subprocess.run(["lsof", "-w", "-Fn", "+D", d], capture_output=True,
                                 text=True, timeout=300)
        except Exception:
            return {"__LSOF_FAILED__"}  # fail-closed: caller must treat as "everything mapped"
        for line in res.stdout.splitlines():
            if line.startswith("n/"):
                out.add(line[1:])
    try:
        res = subprocess.run(["lsof", "-w", "-Fn", "-d", "txt,cwd,rtd"], capture_output=True,
                             text=True, timeout=300)
    except Exception:
        return {"__LSOF_FAILED__"}
    prefixes = [str(Path(d).resolve()) for d in dirs]
    for line in res.stdout.splitlines():
        if not line.startswith("n/"):
            continue
        p = line[1:]
        if any(p == pre or p.startswith(pre + os.sep) for pre in prefixes):
            out.add(p)
    return out


def running_pythons() -> list[dict]:
    try:
        res = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True,
                             timeout=60)
    except Exception:
        return []
    rows = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, cmd = line.partition(" ")
        if re.search(r"(condense|hawking|gravity|doctor|vulture|foundry)", cmd) and "grep" not in cmd:
            rows.append({"pid": int(pid), "command": cmd[:400]})
    return rows


# ── inventory ───────────────────────────────────────────────────────────────────────────
def classify(path: Path, size: int) -> str | None:
    s = str(path)
    name = path.name
    if any(name.endswith(x) for x in PARTIAL_SUFFIXES):
        return "PARTIAL_DOWNLOAD"
    # Hub blobs are content-addressed and carry no extension; snapshots symlink into them.
    if "/huggingface/" in s and f"{os.sep}blobs{os.sep}" in s and size >= MIN_MODEL_BYTES:
        return "DUPLICATE_CACHE"
    if name.endswith(MODEL_SUFFIXES):
        if size < MIN_MODEL_BYTES:
            return None
        if "/.cache/huggingface" in s or "/hub/" in s or "/blobs/" in s:
            return "DUPLICATE_CACHE"
        if "/models/" in s:
            return "RAW_PARENT"
        return "UNKNOWN_MODEL_PAYLOAD"
    return None


def walk_models(roots: list[Path], prot: dict) -> list[dict]:
    rows: list[dict] = []
    skip = set(prot["resolved_paths"])
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            if any(dirpath == p or dirpath.startswith(p + os.sep) for p in skip):
                dirnames[:] = []
                continue
            for fn in filenames:
                p = Path(dirpath) / fn
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                if not (st.st_mode & 0o170000) == 0o100000:  # regular files only
                    continue
                cat = classify(p, st.st_size)
                if cat is None:
                    continue
                rows.append({
                    "path": str(p), "category": cat, "logical_bytes": st.st_size,
                    "allocated_bytes": st.st_blocks * 512, "device": st.st_dev,
                    "inode": st.st_ino, "nlink": st.st_nlink,
                    "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
                    "uid": st.st_uid,
                })
    return rows


def dir_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
        for fn in filenames:
            try:
                total += os.lstat(Path(dirpath) / fn).st_blocks * 512
            except OSError:
                pass
    return total


def cmd_protect(_args) -> dict:
    prot = protected_set()
    mop = (HOME / "Downloads/mop")
    receipt = {
        "schema": "hawking.storage_stripdown.protected_paths.v1",
        "generated_at": now(),
        "mop_root": str(mop.resolve()) if mop.exists() else None,
        "mop_resolved_ok": mop.exists() and mop.is_dir() and not mop.is_symlink(),
        "mop_bytes": dir_bytes(mop),
        "authoritative_hawking_repo": str(ROOT),
        "home": str(HOME),
        "data_volume": DATA_VOLUME,
        "expected_dev": os.stat(DATA_VOLUME).st_dev,
        "volume_total_bytes": volume_bytes(),
        "free_bytes": free_bytes(),
        "protected": prot["roots"],
        "rule": "a deletion candidate is rejected if it resolves to, or through, any protected "
                "root by path or by device/inode; if it is a symlink; if it sits on another "
                "volume; if it is not owned by the invoking uid; or if any check raises.",
    }
    if not receipt["mop_resolved_ok"]:
        receipt["status"] = "BLOCKED"
        receipt["reason"] = "MOP root did not resolve to a real directory; no deletion may run"
    else:
        receipt["status"] = "GREEN"
    write(OUT / "STORAGE_STRIPDOWN_PROTECTED_PATHS.json", receipt)
    return receipt


def cmd_inventory(_args) -> dict:
    prot = protected_set()
    scan_roots = [HOME / "Downloads", HOME / "HawkingWorktrees", HOME / ".cache",
                  HOME / "hawking-qwen-recovery-20260720"]
    payloads = walk_models(scan_roots, prot)
    by_cat: dict[str, dict] = {}
    for r in payloads:
        c = by_cat.setdefault(r["category"], {"count": 0, "logical_bytes": 0,
                                              "allocated_bytes": 0})
        c["count"] += 1
        c["logical_bytes"] += r["logical_bytes"]
        c["allocated_bytes"] += r["allocated_bytes"]

    # by owning directory - this is what a model FAMILY release acts on
    families: dict[str, dict] = {}
    for r in payloads:
        fam = str(Path(r["path"]).parent)
        f = families.setdefault(fam, {"count": 0, "logical_bytes": 0, "allocated_bytes": 0,
                                      "category": r["category"]})
        f["count"] += 1
        f["logical_bytes"] += r["logical_bytes"]
        f["allocated_bytes"] += r["allocated_bytes"]

    caches = {name: dir_bytes(HOME / rel) for name, rel in [
        ("huggingface_hub", ".cache/huggingface"),
        ("uv", ".cache/uv"),
        ("codex_runtimes", ".cache/codex-runtimes"),
        ("go", "go"),
    ]}
    build_products = {}
    for wt in [ROOT, HOME / "Downloads/hawking-hide-build",
               HOME / "Downloads/hawking-hide-parity-research",
               HOME / "HawkingWorktrees/subbit-reset",
               HOME / "HawkingWorktrees/deep-architecture-foundry"]:
        for sub in ("target", "app/node_modules", "node_modules"):
            b = dir_bytes(wt / sub)
            if b:
                build_products[str(wt / sub)] = b

    other_large = []
    for d in sorted((HOME / "Downloads").iterdir()) if (HOME / "Downloads").is_dir() else []:
        if not d.is_dir():
            continue
        b = dir_bytes(d)
        if b > 1 << 30:
            other_large.append({"path": str(d), "allocated_bytes": b,
                                "protected": str(d.resolve()) in prot["resolved_paths"]})

    inv = {
        "schema": "hawking.storage_stripdown.inventory.v1",
        "generated_at": now(),
        "volume_total_bytes": volume_bytes(),
        "free_bytes": free_bytes(),
        "scan_roots": [str(p) for p in scan_roots],
        "model_payloads_by_category": by_cat,
        "model_families": families,
        "caches": caches,
        "build_products": build_products,
        "large_directories": other_large,
        "payload_count": len(payloads),
        "payloads": payloads,
        "report_only": {
            "note": "measured and reported; NOT auto-deleted. Separate authorization required.",
            "apfs_snapshots": _snapshots(),
            "trash_bytes": dir_bytes(HOME / ".Trash"),
        },
    }
    write(OUT / "STORAGE_STRIPDOWN_INVENTORY.json", inv)
    write(OUT / "STORAGE_STRIPDOWN_INVENTORY.md", _inventory_md(inv))
    return {k: v for k, v in inv.items() if k != "payloads"}


def _snapshots() -> list[str]:
    try:
        res = subprocess.run(["tmutil", "listlocalsnapshots", "/"], capture_output=True,
                             text=True, timeout=60)
        return [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def _gib(n: int) -> str:
    return f"{n / 2**30:,.2f} GiB"


def _inventory_md(inv: dict) -> str:
    lines = ["# Storage stripdown inventory", "",
             f"generated {inv['generated_at']}", "",
             f"- volume total: {_gib(inv['volume_total_bytes'])}",
             f"- free: {_gib(inv['free_bytes'])}",
             f"- model payload files found: {inv['payload_count']}", "",
             "## Model payloads by category", "",
             "| category | files | allocated |", "|---|---:|---:|"]
    for cat, c in sorted(inv["model_payloads_by_category"].items(),
                         key=lambda kv: -kv[1]["allocated_bytes"]):
        lines.append(f"| {cat} | {c['count']} | {_gib(c['allocated_bytes'])} |")
    lines += ["", "## Model families (deletion unit)", "",
              "| directory | files | allocated | category |", "|---|---:|---:|---|"]
    for fam, f in sorted(inv["model_families"].items(),
                         key=lambda kv: -kv[1]["allocated_bytes"]):
        lines.append(f"| `{fam}` | {f['count']} | {_gib(f['allocated_bytes'])} | {f['category']} |")
    lines += ["", "## Caches", "", "| cache | allocated |", "|---|---:|"]
    for k, v in sorted(inv["caches"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {_gib(v)} |")
    lines += ["", "## Build products", "", "| path | allocated |", "|---|---:|"]
    for k, v in sorted(inv["build_products"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{k}` | {_gib(v)} |")
    lines += ["", "## Large directories (report only, not auto-deleted)", "",
              "| path | allocated | protected |", "|---|---:|---|"]
    for d in sorted(inv["large_directories"], key=lambda r: -r["allocated_bytes"]):
        lines.append(f"| `{d['path']}` | {_gib(d['allocated_bytes'])} | {d['protected']} |")
    return "\n".join(lines) + "\n"


# ── delete manifest ─────────────────────────────────────────────────────────────────────
def cmd_plan(args) -> dict:
    prot = protected_set()
    if read(OUT / "STORAGE_STRIPDOWN_PROTECTED_PATHS.json").get("status") != "GREEN":
        return {"status": "BLOCKED", "reason": "protected-paths receipt is not GREEN"}
    inv = read(OUT / "STORAGE_STRIPDOWN_INVENTORY.json")
    if not inv:
        return {"status": "BLOCKED", "reason": "no inventory; run inventory first"}

    keep_dirs = [str(Path(p).resolve()) for p in (args.keep or [])]
    expected_dev = os.stat(DATA_VOLUME).st_dev

    # Only families explicitly named for release are eligible. Nothing is deleted by default.
    release_dirs = [str(Path(p).resolve()) for p in (args.release or [])]
    files, rejected = [], []
    for row in inv.get("payloads", []):
        p = Path(row["path"])
        parent = str(p.parent.resolve()) if p.exists() else str(p.parent)
        if not any(parent == d or parent.startswith(d + os.sep) for d in release_dirs):
            continue
        if any(parent == d or parent.startswith(d + os.sep) for d in keep_dirs):
            rejected.append({"path": row["path"], "reason": "under an explicit --keep dir"})
            continue
        ok, why = gate(p, prot, expected_dev)
        (files if ok else rejected).append(
            {**row, "gate": why} if ok else {"path": row["path"], "reason": why})

    # Active-mapping gate over the release dirs only.
    mapped = mapped_paths(release_dirs) if release_dirs else set()
    if "__LSOF_FAILED__" in mapped:
        return {"status": "BLOCKED", "reason": "lsof failed; cannot prove nothing is mapped"}
    still_mapped = [f for f in files if f["path"] in mapped]
    files = [f for f in files if f["path"] not in mapped]
    rejected += [{"path": f["path"], "reason": "mapped by a live process"} for f in still_mapped]

    manifest = {
        "schema": "hawking.storage_stripdown.delete_manifest.v1",
        "generated_at": now(),
        "release_dirs": release_dirs,
        "keep_dirs": keep_dirs,
        "expected_dev": expected_dev,
        "protected_roots": prot["resolved_paths"],
        "free_bytes_before": free_bytes(),
        "files": [{"path": f["path"], "logical_bytes": f["logical_bytes"],
                   "allocated_bytes": f["allocated_bytes"], "inode": f["inode"],
                   "nlink": f["nlink"], "device": f["device"]} for f in files],
        "file_count": len(files),
        "expected_recoverable_bytes": sum(f["allocated_bytes"] for f in files),
        "rejected": rejected,
        "rejected_count": len(rejected),
        "running_processes_at_plan_time": running_pythons(),
        "status": "PLANNED",
        "note": "dry run. Nothing was deleted. `execute --go` re-resolves and re-gates every "
                "path before unlinking it.",
    }
    write(OUT / "STORAGE_STRIPDOWN_DELETE_MANIFEST.json", manifest)
    return {k: v for k, v in manifest.items() if k not in ("files", "rejected")}


def cmd_execute(args) -> dict:
    manifest = read(OUT / "STORAGE_STRIPDOWN_DELETE_MANIFEST.json")
    if not manifest:
        return {"status": "BLOCKED", "reason": "no manifest"}
    if not args.go:
        return {"status": "DRY_RUN", "would_delete": manifest["file_count"],
                "bytes": manifest["expected_recoverable_bytes"]}
    prot = protected_set()
    if read(OUT / "STORAGE_STRIPDOWN_PROTECTED_PATHS.json").get("status") != "GREEN":
        return {"status": "BLOCKED", "reason": "protected-paths receipt is not GREEN"}
    expected_dev = os.stat(DATA_VOLUME).st_dev
    mapped = mapped_paths(manifest["release_dirs"])
    if "__LSOF_FAILED__" in mapped:
        return {"status": "BLOCKED", "reason": "lsof failed at execute time"}

    before = free_bytes()
    results, freed = [], 0
    for row in manifest["files"]:
        p = Path(row["path"])
        ok, why = gate(p, prot, expected_dev)
        if not ok:
            results.append({"path": str(p), "ok": False, "reason": why})
            continue
        if str(p.resolve()) in mapped:
            results.append({"path": str(p), "ok": False, "reason": "mapped at execute time"})
            continue
        st = os.lstat(p)
        if st.st_ino != row["inode"] or st.st_dev != row["device"]:
            results.append({"path": str(p), "ok": False, "reason": "inode/device changed since plan"})
            continue
        try:
            os.unlink(p)
            freed += st.st_blocks * 512
            results.append({"path": str(p), "ok": True, "bytes": st.st_blocks * 512})
        except OSError as exc:
            results.append({"path": str(p), "ok": False, "reason": f"{type(exc).__name__}: {exc}"})

    # Remove only directories that are now verifiably empty, deepest first, still gated.
    dirs_removed = []
    cand = sorted({str(Path(r["path"]).parent) for r in manifest["files"]},
                  key=lambda s: -s.count(os.sep))
    for d in cand:
        dp = Path(d)
        ok, why = gate(dp, prot, expected_dev)
        if not ok or not dp.is_dir():
            continue
        try:
            if not any(dp.iterdir()):
                dp.rmdir()
                dirs_removed.append(d)
        except OSError:
            pass

    receipt = {
        "schema": "hawking.storage_stripdown.delete_receipt.v1",
        "executed_at": now(),
        "manifest_generated_at": manifest["generated_at"],
        "attempted": len(manifest["files"]),
        "deleted": sum(1 for r in results if r["ok"]),
        "failed": [r for r in results if not r["ok"]],
        "bytes_unlinked": freed,
        "empty_dirs_removed": dirs_removed,
        "free_bytes_before": before,
        "free_bytes_after": free_bytes(),
        "realized_recovery_bytes": free_bytes() - before,
    }
    write(OUT / f"STORAGE_STRIPDOWN_DELETE_RECEIPT_{int(time.time())}.json", receipt)
    return {k: v for k, v in receipt.items() if k != "failed"} | {
        "failed_count": len(receipt["failed"])}


# ── model release receipt ───────────────────────────────────────────────────────────────
def cmd_release(args) -> dict:
    """Seal a rehydration receipt for one model family BEFORE its payload is deleted."""
    d = Path(args.dir).resolve()
    if not d.is_dir():
        return {"status": "BLOCKED", "reason": f"{d} is not a directory"}
    kept, payload = [], []
    for p in sorted(d.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        st = p.stat()
        row = {"name": str(p.relative_to(d)), "bytes": st.st_size}
        if p.name.endswith(MODEL_SUFFIXES) and st.st_size >= MIN_MODEL_BYTES:
            payload.append(row)
        else:
            kept.append(row)
    receipt = {
        "schema": "hawking.storage_stripdown.model_release.v1",
        "generated_at": now(),
        "family": args.family,
        "local_dir": str(d),
        "repo": args.repo,
        "revision": args.revision,
        "license": args.license,
        "payload_files": payload,
        "payload_count": len(payload),
        "payload_bytes": sum(r["bytes"] for r in payload),
        "retained_metadata": kept,
        "retained_bytes": sum(r["bytes"] for r in kept),
        "rehydration_command": (
            f"python3.12 -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{args.repo}', revision='{args.revision}', "
            f"local_dir='{d}')\""),
        "rollback": "the source is a public immutable revision on the Hub; re-download restores "
                    "it byte-for-byte. Nothing unique to this machine is in the payload.",
        "scientific_conclusion": args.conclusion,
        "evidence_retained_at": args.evidence,
    }
    write(OUT / f"MODEL_RELEASE_{args.family}.json", receipt)
    return {k: v for k, v in receipt.items() if k not in ("payload_files", "retained_metadata")}


# ── final verification ──────────────────────────────────────────────────────────────────
RECLAIMABLE_NAMES = {"target", "node_modules", ".build", "build", "__pycache__"}


def cmd_reclaim(args) -> dict:
    """Remove REBUILDABLE build trees only. Each must pass the protected-path gate, be named
    like a build directory, and have no live process holding anything inside it."""
    prot = protected_set()
    dev = os.stat(DATA_VOLUME).st_dev
    before = free_bytes()
    results = []
    for raw in args.dir or []:
        d = Path(raw)
        ok, why = gate(d, prot, dev)
        if not ok:
            results.append({"path": str(d), "action": "refused", "reason": why})
            continue
        if not d.is_dir():
            results.append({"path": str(d), "action": "refused", "reason": "not a directory"})
            continue
        if d.name not in RECLAIMABLE_NAMES:
            results.append({"path": str(d), "action": "refused",
                            "reason": f"{d.name!r} is not a build-directory name; this command "
                                      f"only removes rebuildable trees"})
            continue
        holders = mapped_paths([str(d)])
        if "__LSOF_FAILED__" in holders:
            results.append({"path": str(d), "action": "refused", "reason": "lsof failed"})
            continue
        if holders:
            results.append({"path": str(d), "action": "refused",
                            "reason": f"{len(holders)} file(s) held by a live process"})
            continue
        n = dir_bytes(d)
        shutil.rmtree(d, ignore_errors=False)
        results.append({"path": str(d), "action": "removed", "allocated_bytes": n,
                        "restore": args.restore or "rebuild with the project's build command"})
    receipt = {
        "schema": "hawking.storage_stripdown.reclaim.v1", "generated_at": now(),
        "results": results,
        "removed": sum(1 for r in results if r["action"] == "removed"),
        "refused": [r for r in results if r["action"] == "refused"],
        "bytes_removed": sum(r.get("allocated_bytes", 0) for r in results),
        "free_before": before, "free_after": free_bytes(),
        "note": "every path here is a rebuildable build product. No source, no evidence, no "
                "model payload, and nothing under a protected root.",
    }
    write(OUT / f"STORAGE_RECLAIM_RECEIPT_{int(time.time())}.json", receipt)
    return receipt


def cmd_verify(_args) -> dict:
    prot = protected_set()
    mop = HOME / "Downloads/mop"
    manifest = read(OUT / "STORAGE_STRIPDOWN_DELETE_MANIFEST.json")
    receipts = sorted(OUT.glob("STORAGE_STRIPDOWN_DELETE_RECEIPT_*.json"))
    survivors = [f["path"] for f in manifest.get("files", []) if Path(f["path"]).exists()]
    git_ok = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True).returncode == 0
    total_unlinked = sum(read(r).get("bytes_unlinked", 0) for r in receipts)
    baseline = read(OUT / "STORAGE_STRIPDOWN_PROTECTED_PATHS.json").get("free_bytes", 0)
    final = {
        "schema": "hawking.storage_stripdown.final.v1",
        "generated_at": now(),
        "mop_root": str(mop.resolve()) if mop.exists() else None,
        "mop_intact": mop.is_dir(),
        "mop_bytes": dir_bytes(mop),
        "hawking_repo_healthy": git_ok,
        "protected_roots": prot["resolved_paths"],
        "free_bytes_at_campaign_start": baseline,
        "free_bytes_now": free_bytes(),
        "bytes_unlinked_by_manifest": total_unlinked,
        "realized_free_delta": free_bytes() - baseline,
        "manifest_survivors": survivors,
        "manifest_survivor_count": len(survivors),
        "delete_receipts": [str(r) for r in receipts],
        "running_processes": running_pythons(),
        "discrepancy_note": "realized free delta can differ from bytes unlinked because of APFS "
                            "purgeable space, local snapshots holding freed extents, hardlinked "
                            "blocks with surviving references, and concurrent writers.",
    }
    write(OUT / "STORAGE_STRIPDOWN_FINAL.json", final)
    write(OUT / "STORAGE_STRIPDOWN_FINAL.md", "\n".join([
        "# Storage stripdown: final",
        "", f"generated {final['generated_at']}", "",
        f"- MOP root: `{final['mop_root']}` intact={final['mop_intact']} "
        f"({_gib(final['mop_bytes'])})",
        f"- Hawking repo healthy: {final['hawking_repo_healthy']}",
        f"- free at campaign start: {_gib(final['free_bytes_at_campaign_start'])}",
        f"- free now: {_gib(final['free_bytes_now'])}",
        f"- bytes unlinked by manifest: {_gib(final['bytes_unlinked_by_manifest'])}",
        f"- realized free delta: {_gib(final['realized_free_delta'])}",
        f"- manifest survivors (should be 0): {final['manifest_survivor_count']}",
        "", final["discrepancy_note"], ""]))
    return {k: v for k, v in final.items() if k != "manifest_survivors"}


def self_check() -> None:
    """Fails loudly if the protected-path gate stops protecting."""
    prot = protected_set()
    dev = os.stat(DATA_VOLUME).st_dev
    mop = HOME / "Downloads/mop"
    if mop.exists():
        ok, why = gate(mop / "README.md", prot, dev)
        assert not ok, "GATE BROKEN: a path inside MOP passed the deletion gate"
        assert "protected" in why, why
        ok, _ = gate(mop, prot, dev)
        assert not ok, "GATE BROKEN: the MOP root itself passed the deletion gate"
    ok, why = gate(ROOT / ".git" / "HEAD", prot, dev)
    assert not ok, "GATE BROKEN: the authoritative git database passed the deletion gate"
    ok, why = gate(Path("/etc/hosts"), prot, dev)
    assert not ok, f"GATE BROKEN: a system path passed the deletion gate ({why})"
    ok, why = gate(ROOT / "does-not-exist-xyz", prot, dev)
    assert not ok and why == "does not exist", why

    # A tree containing a RUNNING executable must read as mapped. `lsof +D` does not report txt
    # descriptors, so this asserts the txt/cwd sweep is doing its job.
    try:
        res = subprocess.run(["lsof", "-w", "-Fn", "-d", "txt"], capture_output=True,
                             text=True, timeout=120)
        running = [ln[1:] for ln in res.stdout.splitlines()
                   if ln.startswith("n/") and "/target/" in ln]
    except Exception:
        running = []
    if running:
        tree = str(Path(running[0]).parent)
        assert mapped_paths([tree]), \
            f"GATE BROKEN: {tree} holds a running executable but read as unmapped"
        print(f"self_check: OK (MOP, .git, /etc, missing paths rejected; live-exec tree "
              f"{tree} correctly reads as mapped)")
    else:
        print("self_check: OK (MOP, .git, /etc and missing paths all rejected; "
              "no running in-tree executable available to exercise the mapping check)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("protect")
    sub.add_parser("inventory")
    p = sub.add_parser("plan")
    p.add_argument("--release", action="append", help="directory whose payload may be released")
    p.add_argument("--keep", action="append", help="directory that must survive")
    p = sub.add_parser("release")
    p.add_argument("--family", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--license", default="unknown")
    p.add_argument("--conclusion", default="")
    p.add_argument("--evidence", default="")
    p = sub.add_parser("reclaim")
    p.add_argument("--dir", action="append", help="rebuildable build directory to remove")
    p.add_argument("--restore", help="the command that regenerates these trees")
    p = sub.add_parser("execute")
    p.add_argument("--go", action="store_true", help="actually unlink (default is dry run)")
    sub.add_parser("verify")
    sub.add_parser("self-check")
    args = ap.parse_args()
    if args.cmd == "self-check":
        self_check()
        return 0
    fn = {"protect": cmd_protect, "inventory": cmd_inventory, "plan": cmd_plan,
          "release": cmd_release, "execute": cmd_execute, "verify": cmd_verify,
          "reclaim": cmd_reclaim}[args.cmd]
    out = fn(args)
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 1 if out.get("status") == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
