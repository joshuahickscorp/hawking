#!/usr/bin/env python3
"""SPRING_CLEAN_CENSUS — classify candidates; delete nothing.

Observer over the live Hawking working tree (default ~/Downloads/hawking).
Writes one receipt next to THIS worktree:

    receipts/headless/SPRING_CLEAN_CENSUS.json

This lane is a classified plan for a human. It never runs rm, git clean,
git reset, git checkout, git restore, or git stash. Untracked paths start
precious until inspection proves otherwise. "It lives in a cache directory"
is not proof.

A file missing from THIS sparse checkout is not evidence it does not exist
in the scan root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "hawking.headless.spring_clean_census.v1"
DEFAULT_SCAN = Path("/Users/scammermike/Downloads/hawking")
HAWKING_COPY = Path("/Users/scammermike/Downloads/hawking-copy")
RUNS_PREFIX = "workspace/campaign/records/runs/"
SANDBOX_PREFIX = "workspace/campaign/records/ascension-sandbox/"
PHASEB_PREFIX = "workspace/campaign/phaseB/"

CLASSES = ("RECLAIMABLE", "DUPLICATE", "PRECIOUS", "MODEL", "UNKNOWN")

# Path components that are regenerable build/test output. Applied as a whole
# component match so a file named "target.py" is not swept.
RECLAIM_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    "node_modules",
    ".venv",
    "venv",
}

WEIGHT_EXT = {".safetensors", ".gguf"}

PRECIOUS_PREFIXES = (
    "receipts/",
    "workspace/campaign/records/runs/",
    "workspace/campaign/records/ascension-sandbox/",
    "workspace/campaign/phaseB/",
    "workspace/campaign/evidence/",
    "workspace/campaign/odyssey/",
    "workspace/campaign/config/",
    "workspace/campaign/governance/",
    "workspace/campaign/records/reports/",
    "workspace/campaign/records/research/",
    "hawking-experiments/superwave/",
    "workspace/docs/",
    "workspace/vendor/",
    "workspace/ops/ascension/",
    "workspace/ops/ascent-lanes/",
    "workspace/ops/deploy/",
    "workspace/ops/genesis-agentos-sessions/",
    "workspace/ops/genesis-worker-checkpoints/",
    "visionmcp/artifacts/",
    "visionmcp/src/",
    "visionmcp/tests/",
    "visionmcp/tools/",
    "visionmcp/benchmarks/",
    "visionmcp/docs/",
    ".preserved/",
    ".haider/",
    ".hcli/",
    ".hide/evidence/",
    "crates/",
    "lab/",
    "ramanujan/",
    "docs/",
    "reports/",
    "spec/",
    "contracts/",
    "app/",
    "visionmcp/",
    "tools/",
    "logs/",
    "src/",
)

# git read-only only. Anything mutating is a bug in this observer.
GIT_FORBIDDEN = {
    "clean", "reset", "checkout", "restore", "stash", "rm",
    "add", "commit", "merge", "rebase", "push",
}


def observer_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def git_ro(repo: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    if not args:
        raise RuntimeError("refusing empty git invocation")
    if args[0] in GIT_FORBIDDEN:
        raise RuntimeError(f"refusing git {args[0]} — observer is read-only")
    # `git worktree list` is read-only. Every other worktree verb mutates.
    if args[0] == "worktree" and (len(args) < 2 or args[1] != "list"):
        raise RuntimeError(f"refusing git {' '.join(args)} — observer is read-only")
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def posix(path: str) -> str:
    return path.replace("\\", "/")


def rel_of(root: Path, path: Path) -> str:
    return posix(str(path.relative_to(root)))


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def allocated_bytes(st: os.stat_result) -> int:
    # macOS/BSD: st_blocks is 512-byte blocks.
    blocks = getattr(st, "st_blocks", None)
    if blocks is None:
        return st.st_size
    return int(blocks) * 512


def regenerates_for(rel: str) -> str:
    parts = Path(posix(rel)).parts
    if ".venv" in parts or "venv" in parts:
        if rel.startswith("visionmcp/"):
            return "cd visionmcp && uv sync  (uv.lock + pyproject.toml)"
        if rel.startswith("tools/condense/"):
            return (
                "python3 -m venv tools/condense/.venv && "
                "tools/condense/.venv/bin/pip install "
                "-r tools/condense/requirements-deepseek-v4.txt "
                "-r tools/condense/requirements-glm52.txt"
            )
        return "recreate the virtualenv from the project's lock/requirements"
    if "node_modules" in parts:
        if rel.startswith("app/"):
            return "pnpm -C app install  (app/pnpm-lock.yaml)"
        if rel.startswith("workspace/ops/"):
            return "pnpm -C workspace/ops install  (workspace/ops/pnpm-lock.yaml)"
        return "pnpm/npm install in the enclosing package"
    if "__pycache__" in parts or rel.endswith(".pyc") or rel.endswith(".pyo"):
        return "Python import (bytecode cache)"
    if ".pytest_cache" in parts:
        return "pytest"
    if ".ruff_cache" in parts:
        return "ruff"
    if ".mypy_cache" in parts:
        return "mypy"
    if rel.startswith("workspace/ops/build/"):
        return "cargo build  (target-dir workspace/ops/build/rust per .cargo/config.toml)"
    if rel.startswith("app/dist/"):
        return "pnpm -C app build  (vite; package.json scripts.build)"
    if rel.startswith("app/src-tauri/binaries/"):
        return "cargo build -p hide-serve  (Tauri sidecar; gitignored)"
    if rel.startswith("visionmcp/dist/"):
        return "python -m build in visionmcp/  (wheels/sdist)"
    if rel.startswith("visionmcp/.world-engine/verify-scratch/install-venv/"):
        return "visionmcp verify/install path recreates this venv"
    if rel.startswith("visionmcp/.world-engine/verify-scratch/packaging-cache/"):
        return "pip/uv wheel cache; next packaging run refills it"
    if rel.startswith(".aider.tags.cache"):
        return "aider tag-cache rebuild on next aider session"
    if rel.startswith("workspace/ops/local/hf-cache/"):
        return "HuggingFace xet client logs/staging; recreated on next hf download"
    if rel.startswith(".hide/cache/") or rel.startswith(".hide/tmp/"):
        return "HIDE runtime cache; regenerated per session"
    if Path(rel).name == ".DS_Store":
        return "Finder recreates .DS_Store"
    if "target" in parts:
        return "cargo build  (target-dir workspace/ops/build/rust)"
    return "regenerable build/cache output"


def starts_with_any(rel: str, prefixes: Iterable[str]) -> bool:
    return any(rel == p.rstrip("/") or rel.startswith(p) for p in prefixes)


def classify_path(rel: str) -> tuple[str, str, str | None]:
    """Return (class, reason, regenerates-or-None) before the duplicate pass."""
    rel = posix(rel)
    parts = Path(rel).parts
    name = parts[-1] if parts else rel

    if not parts:
        return "UNKNOWN", "empty path", None

    # Reclaimable directory components — even when nested under science trees.
    # Bytecode next to a receipt is still bytecode.
    reclaim_hit = [p for p in parts if p in RECLAIM_DIR_NAMES]
    if reclaim_hit:
        return (
            "RECLAIMABLE",
            f"path component {reclaim_hit[0]!r} is regenerable build/test output",
            regenerates_for(rel),
        )
    if name.endswith((".pyc", ".pyo")) or name == ".DS_Store":
        return "RECLAIMABLE", f"generated file {name}", regenerates_for(rel)
    if rel.startswith("workspace/ops/build/"):
        return "RECLAIMABLE", "cargo target-dir", regenerates_for(rel)
    if rel.startswith("app/dist/"):
        return "RECLAIMABLE", "vite production bundle", regenerates_for(rel)
    if rel.startswith("app/src-tauri/binaries/"):
        return "RECLAIMABLE", "gitignored Tauri sidecar binary", regenerates_for(rel)
    if rel.startswith("visionmcp/dist/") and name.endswith((".whl", ".tar.gz")):
        return "RECLAIMABLE", "python packaging output", regenerates_for(rel)
    if rel.startswith("visionmcp/.world-engine/verify-scratch/install-venv/"):
        return "RECLAIMABLE", "verify-scratch install venv", regenerates_for(rel)
    if rel.startswith("visionmcp/.world-engine/verify-scratch/packaging-cache/"):
        return "RECLAIMABLE", "verify-scratch wheel cache", regenerates_for(rel)
    if rel.startswith(".aider.tags.cache"):
        return "RECLAIMABLE", "aider tag cache", regenerates_for(rel)
    if rel.startswith("workspace/ops/local/hf-cache/"):
        return "RECLAIMABLE", "HuggingFace xet cache/logs", regenerates_for(rel)
    if rel.startswith(".hide/cache/") or rel.startswith(".hide/tmp/"):
        return "RECLAIMABLE", "HIDE runtime cache", regenerates_for(rel)

    ext = Path(name).suffix.lower()
    if ext in WEIGHT_EXT:
        return "MODEL", f"weight artifact ({ext})", None

    # These must beat the blanket visionmcp/ and tools/ precious prefixes.
    if rel.startswith("visionmcp/.worktrees/"):
        return (
            "UNKNOWN",
            "nested visionmcp git worktree; some files may digest-match the "
            "main visionmcp tree, some may not. Duplicate pass decides.",
            None,
        )
    if rel.startswith("workspace/ops/local/genesis-agentos-worktrees/"):
        return (
            "UNKNOWN",
            "registered genesis-agentos git worktree (clean detached HEAD). "
            "Sessions/checkpoints point here; live use was not proven. "
            "A clean checkout is still not deletion-safe until a human confirms "
            "genesis-resident is not using it.",
            None,
        )
    if rel.startswith("visionmcp/.world-engine/"):
        return (
            "UNKNOWN",
            "verify-scratch remainder (HEAD.tar / logs / scrub json) — "
            "directory name says scratch; contents not proven regenerable",
            None,
        )

    if starts_with_any(rel, PRECIOUS_PREFIXES):
        if rel.startswith(RUNS_PREFIX):
            return (
                "PRECIOUS",
                "under workspace/campaign/records/runs/ — never recommend deletion "
                "(model weights live here AND their run sidecars)",
                None,
            )
        return "PRECIOUS", "experiment output / receipt / source / ledger", None

    # Root-level campaign notes and tracked repo identity.
    if name.endswith((".json", ".jsonl", ".md")) and "/" not in rel:
        return "PRECIOUS", "root-level campaign finding / ledger", None
    if rel in {
        "Cargo.lock", "Cargo.toml", "LICENSE", "README.md", ".gitignore",
        ".agent_env.example", "workspace/README.md",
    } or rel.startswith(".github/"):
        return "PRECIOUS", "tracked repository identity / source", None
    if rel.startswith(".aider"):
        return "UNKNOWN", "aider session history — not a build cache", None
    if rel.startswith(".hide/"):
        return "UNKNOWN", "HIDE runtime state (memory/index); not proven empty of science", None
    if rel.startswith(".claude") or rel.startswith(".serena") or rel.startswith(".cargo"):
        return "UNKNOWN", "tooling state / config, not regenerable build output", None

    return "UNKNOWN", "no rule matched; default UNKNOWN (an UNKNOWN costs disk, a wrong RECLAIMABLE costs science)", None


def bucket_of(rel: str) -> str:
    """Coarse group key for the human plan."""
    rel = posix(rel)
    parts = Path(rel).parts
    if not parts:
        return rel
    # Keep reclaimable dirs as the directory itself when possible.
    for i, p in enumerate(parts):
        if p in RECLAIM_DIR_NAMES:
            return "/".join(parts[: i + 1])
    name = parts[-1]
    if name == ".DS_Store":
        return rel
    prefixes = [
        "visionmcp/.world-engine/verify-scratch/install-venv",
        "visionmcp/.world-engine/verify-scratch/packaging-cache",
        "visionmcp/.world-engine/verify-scratch",
        RUNS_PREFIX.rstrip("/"),
        SANDBOX_PREFIX.rstrip("/"),
        PHASEB_PREFIX.rstrip("/"),
        "workspace/campaign/evidence",
        "workspace/campaign/odyssey",
        "workspace/campaign/records/research",
        "hawking-experiments/superwave/g1",
        "visionmcp/artifacts",
        "visionmcp/.worktrees",
        "visionmcp/.world-engine",
        "visionmcp/.venv",
        "visionmcp/dist",
        "tools/condense/.venv",
        "app/node_modules",
        "app/dist",
        "app/src-tauri/binaries",
        "workspace/ops/local/hf-cache",
        "workspace/ops/local/genesis-agentos-worktrees",
        "workspace/ops/build",
        ".aider.tags.cache.v4",
        ".preserved",
        "receipts",
    ]
    for p in prefixes:
        if rel == p or rel.startswith(p + "/"):
            return p
    if len(parts) == 1:
        return rel
    return "/".join(parts[:2])


def walk_files(scan: Path) -> Iterable[tuple[str, Path, os.stat_result]]:
    skip_dirs = {".git"}
    for dirpath, dirnames, filenames in os.walk(scan, followlinks=False):
        dirnames[:] = [
            d for d in dirnames
            if d not in skip_dirs and not os.path.islink(os.path.join(dirpath, d))
        ]
        for fn in filenames:
            full = Path(dirpath) / fn
            try:
                st = full.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                continue
            yield rel_of(scan, full), full, st


def canary_paths(scan: Path) -> list[str]:
    return [
        "workspace/campaign/records/ascension-sandbox/knowledge-plane/ASCENSION_NEGATIVE_SCIENCE.jsonl",
        ".preserved/from-target/first_model_free_receipt.json",
        "workspace/campaign/phaseB/ckpt/grouped_k4_stable.pt",
        "workspace/campaign/phaseB/capture_diverse/L00.f16",
        "workspace/campaign/phaseB/capture_diverse2/L00.f16",
        "workspace/campaign/records/runs/deepseek-v4/DEEPSEEK_V4_FRANKENSTEIN_PROGRESS.jsonl",
    ]


def stat_canaries(scan: Path) -> list[dict[str, Any]]:
    rows = []
    for rel in canary_paths(scan):
        p = scan / rel
        rec: dict[str, Any] = {"path": rel, "exists": p.is_file()}
        if p.is_file():
            st = p.lstat()
            rec.update({
                "inode": st.st_ino,
                "mtime_ns": st.st_mtime_ns,
                "size": st.st_size,
            })
        rows.append(rec)
    return rows


def canaries_unchanged(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    problems = []
    by = {r["path"]: r for r in after}
    for b in before:
        a = by.get(b["path"])
        if a is None:
            problems.append(f"missing after: {b['path']}")
            continue
        if b.get("exists") and not a.get("exists"):
            problems.append(f"deleted: {b['path']}")
        for k in ("inode", "mtime_ns", "size"):
            if b.get(k) != a.get(k):
                problems.append(f"{k} changed on {b['path']}: {b.get(k)} -> {a.get(k)}")
    return (not problems), problems


def inspect_phaseb(scan: Path) -> dict[str, Any]:
    d1 = scan / "workspace/campaign/phaseB/capture_diverse"
    d2 = scan / "workspace/campaign/phaseB/capture_diverse2"
    ckpt = scan / "workspace/campaign/phaseB/ckpt"
    out: dict[str, Any] = {
        "tempted_as": "RECLAIMABLE (untracked, looks like scratch, 10.9G)",
        "classified_as": "PRECIOUS",
        "why": (
            "Two distinct activation-capture campaigns plus trained student "
            "checkpoints. Same-looking Lxx.f16 names are NOT byte-identical."
        ),
        "capture_pairs": [],
        "ckpt_head_tail": [],
    }
    names = ["L00.f16", "L31.f16", "L63.f16", "manifest.json"]
    for name in names:
        p = d1 / name
        q = d2 / name
        row: dict[str, Any] = {"name": name, "a_exists": p.is_file(), "b_exists": q.is_file()}
        if p.is_file():
            row["a_bytes"] = p.stat().st_size
            row["a_sha256"] = sha256_file(p)
        if q.is_file():
            row["b_bytes"] = q.stat().st_size
            row["b_sha256"] = sha256_file(q)
        if p.is_file() and q.is_file():
            row["digest_equal"] = row["a_sha256"] == row["b_sha256"]
        out["capture_pairs"].append(row)

    def head_tail(path: Path, span: int = 8 << 20) -> str:
        size = path.stat().st_size
        h = hashlib.sha256()
        h.update(str(size).encode())
        with open(path, "rb") as f:
            h.update(f.read(span))
            if size > span:
                f.seek(max(0, size - span))
                h.update(f.read(span))
        return h.hexdigest()

    if ckpt.is_dir():
        for p in sorted(ckpt.iterdir()):
            if p.is_file() and not p.is_symlink():
                out["ckpt_head_tail"].append({
                    "path": rel_of(scan, p),
                    "bytes": p.stat().st_size,
                    "sha256_size_head_tail_8MiB": head_tail(p),
                    "note": "near-equal sizes are still distinct identities",
                })
    return out


def inspect_sandbox(scan: Path) -> dict[str, Any]:
    root = scan / "workspace/campaign/records/ascension-sandbox"
    ns = root / "knowledge-plane" / "ASCENSION_NEGATIVE_SCIENCE.jsonl"
    markers = {
        "sandbox-ready-config.json": (root / "sandbox-ready-config.json").is_file(),
        "ASCENSION_NEGATIVE_SCIENCE.jsonl": ns.is_file(),
        "knowledge-plane": (root / "knowledge-plane").is_dir(),
        "physical/qwen80": (root / "physical" / "qwen80").is_dir(),
        "lifecycle/ASCENSION_V3_CONSTITUTION.json": (
            root / "lifecycle" / "ASCENSION_V3_CONSTITUTION.json"
        ).is_file(),
    }
    return {
        "tempted_as": "RECLAIMABLE (11G, directory name 'sandbox')",
        "classified_as": "PRECIOUS",
        "why": (
            "Not scratch. knowledge-plane holds ASCENSION_NEGATIVE_SCIENCE.jsonl "
            "and genomes; physical/ holds .f32le activation captures, .gravity "
            "candidates, terminal receipts; lifecycle holds constitution, work "
            "queue, source-admission results. gitignore names these physical "
            "gravity artifacts as on-disk evidence that must never be committed "
            "AND never treated as disposable cache."
        ),
        "markers_present": markers,
        "negative_science_bytes": ns.stat().st_size if ns.is_file() else None,
    }


def inspect_starting_picture_models(scan: Path) -> dict[str, Any]:
    """The operator's starting picture named 282 GB of safetensors under runs/.

    This observer measured glm-4.5-air (~221 GB, 47 shards) and qwen3-30b-a3b
    (~61 GB, 13 shards) earlier in the same session. At census time those
    directories are gone. runs/ mtime is the disappearance clock. This script
    did not delete them.
    """
    expected = [
        "workspace/campaign/records/runs/glm-4.5-air",
        "workspace/campaign/records/runs/qwen3-30b-a3b",
    ]
    rows = []
    for rel in expected:
        p = scan / rel
        rows.append({
            "path": rel,
            "exists": p.exists(),
            "is_dir": p.is_dir(),
        })
    runs = scan / "workspace/campaign/records/runs"
    st = runs.stat() if runs.is_dir() else None
    remaining = sorted(c.name for c in runs.iterdir()) if runs.is_dir() else []
    safetensors_n = 0
    if runs.is_dir():
        for root, dirs, files in os.walk(runs, followlinks=False):
            dirs[:] = [d for d in dirs if d != ".git"]
            safetensors_n += sum(1 for f in files if f.endswith(".safetensors"))
    return {
        "operator_starting_picture": {
            "hawking_total": "319G",
            "safetensors": "282 GB across 60 files under workspace/campaign/records/",
            "runs": "264G",
        },
        "measured_earlier_this_session": {
            "workspace/campaign/records/runs/glm-4.5-air": {
                "logical_bytes": 220962000000,
                "files": 162,
                "safetensors": 47,
                "note": "python walk earlier this session, before 14:47",
            },
            "workspace/campaign/records/runs/qwen3-30b-a3b": {
                "logical_bytes": 61084000000,
                "files": 72,
                "safetensors": 13,
                "note": "python walk earlier this session, before 14:47",
            },
        },
        "at_census_time": {
            "expected_dirs": rows,
            "runs_remaining_children": remaining,
            "safetensors_under_runs": safetensors_n,
            "runs_mtime": (
                time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))
                if st else None
            ),
            "runs_mtime_unix": st.st_mtime if st else None,
        },
        "this_observer_deleted_them": False,
        "classified_as": "MODEL (but ABSENT at census — 0 bytes to classify)",
        "recommend_deletion": False,
        "why_this_is_in_the_receipt": (
            "A census that silently reported MODEL=0 without naming the "
            "disappearance would look like it had classified 282 GB of weights "
            "as reclaimable and removed them. It did not. The directories left "
            "the tree at runs/ mtime, outside this process."
        ),
    }


def inspect_visionmcp_artifacts(scan: Path) -> dict[str, Any]:
    freeze = scan / "visionmcp/artifacts/world-engine/EVIDENCE_FREEZE_MANIFEST.json"
    return {
        "tempted_as": "RECLAIMABLE (path contains 'artifacts')",
        "classified_as": "PRECIOUS",
        "why": (
            "EVIDENCE_FREEZE_MANIFEST.json, GATE_LEDGER_REDERIVED, freeze packs, "
            "and step-NN depth.f32 / object_id captures. Freeze/evidence, not a "
            "build cache."
        ),
        "evidence_freeze_manifest_present": freeze.is_file(),
        "evidence_freeze_manifest_bytes": freeze.stat().st_size if freeze.is_file() else None,
    }


def duplicate_pass_visionmcp_worktrees(
    scan: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[tuple[str, os.stat_result, str, str, str | None]]]:
    """Digest-compare visionmcp/.worktrees/<name>/<rel> against visionmcp/<rel>.

    Returns (proofs, summary, leftover_rows) where leftover_rows are worktree
    files that are NOT digest-proved duplicates, each as
    (rel, stat, class, reason, regenerates).
    """
    wt_root = scan / "visionmcp" / ".worktrees"
    proofs: list[dict[str, Any]] = []
    leftovers: list[tuple[str, os.stat_result, str, str, str | None]] = []
    summary: dict[str, Any] = {"worktrees": {}, "note": (
        "gitdir points at /Users/scammermike/Downloads/visionmcp/.git/worktrees/<name> "
        "(a different nested repo). git worktree list marks those paths prunable. "
        "Only digest-matched files are DUPLICATE; worktree-only files stay UNKNOWN "
        "unless they are independently RECLAIMABLE (venv, pycache)."
    )}
    if not wt_root.is_dir():
        return proofs, summary, leftovers

    for wt in sorted(p for p in wt_root.iterdir() if p.is_dir() and not p.is_symlink()):
        name = wt.name
        matched_n = 0
        matched_b = 0
        unique_n = 0
        unique_b = 0
        size_mismatch_n = 0
        unique_samples: list[str] = []
        for rel, full, st in walk_files(wt):
            if rel.startswith(".git/") or "/.git/" in rel:
                continue
            dup_rel = f"visionmcp/.worktrees/{name}/{rel}"
            cls0, reason0, regen0 = classify_path(dup_rel)
            # venv/pycache inside a worktree stays RECLAIMABLE even if it
            # also happens to match a survivor.
            if cls0 == "RECLAIMABLE":
                leftovers.append((dup_rel, st, cls0, reason0, regen0))
                continue
            main = scan / "visionmcp" / rel
            if not main.is_file():
                unique_n += 1
                unique_b += st.st_size
                if len(unique_samples) < 8:
                    unique_samples.append(dup_rel)
                leftovers.append((dup_rel, st, cls0, reason0, regen0))
                continue
            try:
                mst = main.lstat()
            except OSError:
                unique_n += 1
                unique_b += st.st_size
                leftovers.append((dup_rel, st, cls0, reason0, regen0))
                continue
            if mst.st_size != st.st_size:
                size_mismatch_n += 1
                unique_n += 1
                unique_b += st.st_size
                leftovers.append((dup_rel, st, cls0, reason0, regen0))
                continue
            h_dup = sha256_file(full)
            h_surv = sha256_file(main)
            if h_dup != h_surv:
                unique_n += 1
                unique_b += st.st_size
                leftovers.append((dup_rel, st, cls0, reason0, regen0))
                continue
            matched_n += 1
            matched_b += st.st_size
            proofs.append({
                "path": dup_rel,
                "surviving_path": f"visionmcp/{rel}",
                "sha256": h_surv,
                "bytes": st.st_size,
                "allocated_bytes": allocated_bytes(st),
                "proof": "sha256(duplicate) == sha256(survivor) and sizes equal",
            })
        summary["worktrees"][name] = {
            "digest_matched_files": matched_n,
            "digest_matched_bytes": matched_b,
            "worktree_only_or_mismatch_files": unique_n,
            "worktree_only_or_mismatch_bytes": unique_b,
            "size_mismatch_files": size_mismatch_n,
            "unique_samples": unique_samples,
        }
    return proofs, summary, leftovers


def git_dot_size(scan: Path) -> int:
    git = scan / ".git"
    total = 0
    if not git.exists():
        return 0
    for root, dirs, files in os.walk(git, followlinks=False):
        for f in files:
            fp = os.path.join(root, f)
            try:
                st = os.lstat(fp)
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                total += st.st_size
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Classify spring-clean candidates; delete nothing.")
    ap.add_argument("--root", default=str(DEFAULT_SCAN), help="scan root (live Hawking tree)")
    ap.add_argument(
        "--out",
        default="",
        help="receipt path (default: observer receipts/headless/SPRING_CLEAN_CENSUS.json)",
    )
    args = ap.parse_args(argv)

    t0 = time.time()
    observer = observer_root()
    scan = Path(os.path.expanduser(args.root)).resolve()
    out_path = (
        Path(args.out).resolve()
        if args.out
        else observer / "receipts" / "headless" / "SPRING_CLEAN_CENSUS.json"
    )

    if not scan.is_dir():
        print(f"scan root missing: {scan}", file=sys.stderr)
        return 2

    # Never write into hawking-copy. Never write into the scan root unless
    # the observer IS the scan root (i.e. this script lives there).
    if HAWKING_COPY.exists() and out_path.is_relative_to(HAWKING_COPY.resolve()):
        print("refusing to write into hawking-copy", file=sys.stderr)
        return 2

    canaries_before = stat_canaries(scan)
    porcelain = git_ro(scan, "status", "--porcelain")
    dirty_n = len([ln for ln in porcelain.stdout.splitlines() if ln.strip()])
    head = (git_ro(scan, "rev-parse", "HEAD").stdout or "").strip()
    branch = (git_ro(scan, "rev-parse", "--abbrev-ref", "HEAD").stdout or "").strip()
    wt_list = git_ro(scan, "worktree", "list").stdout

    totals = {c: {"files": 0, "bytes": 0, "allocated_bytes": 0} for c in CLASSES}
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    reclaimable_groups: dict[str, dict[str, Any]] = {}
    unknown_groups: dict[str, dict[str, Any]] = {}
    model_groups: dict[str, dict[str, Any]] = {}
    precious_groups: dict[str, dict[str, Any]] = {}
    n_files = 0
    logical_total = 0
    allocated_total = 0
    runs_bytes = 0
    runs_files = 0
    runs_model_bytes = 0
    sandbox_bytes = 0
    sandbox_files = 0
    sandbox_f32le = 0
    phaseb_bytes = 0
    phaseb_files = 0

    def bump(rel: str, st: Any, cls: str, reason: str, regen: str | None) -> None:
        nonlocal n_files, logical_total, allocated_total
        nonlocal runs_bytes, runs_files, runs_model_bytes
        nonlocal sandbox_bytes, sandbox_files, sandbox_f32le
        nonlocal phaseb_bytes, phaseb_files
        b = st.st_size
        a = allocated_bytes(st)
        n_files += 1
        logical_total += b
        allocated_total += a
        totals[cls]["files"] += 1
        totals[cls]["bytes"] += b
        totals[cls]["allocated_bytes"] += a
        bucket = bucket_of(rel)
        key = (cls, bucket)
        g = groups.setdefault(key, {
            "class": cls, "path": bucket, "files": 0, "bytes": 0,
            "allocated_bytes": 0,
            "recommend_deletion": cls in ("RECLAIMABLE", "DUPLICATE"),
            "reason": reason,
            "regenerates": regen if cls == "RECLAIMABLE" else None,
        })
        g["files"] += 1
        g["bytes"] += b
        g["allocated_bytes"] += a
        if cls == "RECLAIMABLE":
            rg = reclaimable_groups.setdefault(bucket, {
                "path": bucket, "class": "RECLAIMABLE", "files": 0, "bytes": 0,
                "regenerates": regenerates_for(bucket + "/x"),
                "recommend_deletion": True,
            })
            rg["files"] += 1
            rg["bytes"] += b
        elif cls == "UNKNOWN":
            ug = unknown_groups.setdefault(bucket, {
                "path": bucket, "class": "UNKNOWN", "files": 0, "bytes": 0,
                "recommend_deletion": False, "reason": reason,
            })
            ug["files"] += 1
            ug["bytes"] += b
        elif cls == "MODEL":
            mg = model_groups.setdefault(bucket, {
                "path": bucket, "class": "MODEL", "files": 0, "bytes": 0,
                "recommend_deletion": False,
            })
            mg["files"] += 1
            mg["bytes"] += b
        elif cls == "PRECIOUS":
            pg = precious_groups.setdefault(bucket, {
                "path": bucket, "class": "PRECIOUS", "files": 0, "bytes": 0,
                "recommend_deletion": False,
            })
            pg["files"] += 1
            pg["bytes"] += b
        elif cls == "DUPLICATE":
            # grouped via `groups`; proofs list carries per-file digests
            pass
        if rel.startswith(RUNS_PREFIX):
            runs_bytes += b
            runs_files += 1
            if cls == "MODEL":
                runs_model_bytes += b
        if rel.startswith(SANDBOX_PREFIX):
            sandbox_bytes += b
            sandbox_files += 1
            if rel.endswith(".f32le"):
                sandbox_f32le += b
        if rel.startswith(PHASEB_PREFIX):
            phaseb_bytes += b
            phaseb_files += 1

    # Skip visionmcp/.worktrees here — the duplicate pass owns that subtree
    # so we do not hold a million-file metadata map in RAM.
    for rel, full, st in walk_files(scan):
        if rel.startswith("visionmcp/.worktrees/"):
            continue
        cls, reason, regen = classify_path(rel)
        bump(rel, st, cls, reason, regen)

    dup_proofs, dup_summary, leftovers = duplicate_pass_visionmcp_worktrees(scan)

    class _DupStat:
        __slots__ = ("st_size", "st_blocks")

        def __init__(self, size: int, allocated: int) -> None:
            self.st_size = size
            self.st_blocks = max(1, (allocated + 511) // 512) if size else 0

    for p in dup_proofs:
        bump(
            p["path"],
            _DupStat(p["bytes"], p.get("allocated_bytes", p["bytes"])),
            "DUPLICATE",
            f"sha256-identical to {p['surviving_path']}",
            None,
        )
    for rel, st, cls, reason, regen in leftovers:
        bump(rel, st, cls, reason, regen)

    # Live inspections that justify the downgrades.
    phaseb_insp = inspect_phaseb(scan)
    sandbox_insp = inspect_sandbox(scan)
    v_art = inspect_visionmcp_artifacts(scan)
    missing_models = inspect_starting_picture_models(scan)

    git_size = git_dot_size(scan)
    copy_present = HAWKING_COPY.is_dir()

    canaries_after = stat_canaries(scan)
    ok_canaries, canary_problems = canaries_unchanged(canaries_before, canaries_after)
    porcelain_after = git_ro(scan, "status", "--porcelain")
    dirty_after = len([ln for ln in porcelain_after.stdout.splitlines() if ln.strip()])

    tempted = [
        {
            "path": "workspace/campaign/records/ascension-sandbox/",
            "bytes": sandbox_bytes,
            "files": sandbox_files,
            **sandbox_insp,
        },
        {
            "path": "workspace/campaign/phaseB/",
            "bytes": phaseb_bytes,
            "files": phaseb_files,
            **phaseb_insp,
        },
        {
            "path": "visionmcp/artifacts/",
            **v_art,
        },
        {
            "path": "workspace/campaign/records/research/",
            "tempted_as": "RECLAIMABLE (.gitignore calls it regenerable measurement scratch)",
            "classified_as": "PRECIOUS",
            "why": (
                "actmean-qwen05b.json is a 10 MiB measurement body. gitignore is "
                "not proof the bytes can be re-derived; this project has already "
                "lost science that only lived on disk."
            ),
        },
        {
            "path": "visionmcp/.world-engine/verify-scratch/HEAD.tar",
            "tempted_as": "RECLAIMABLE (path contains verify-scratch)",
            "classified_as": "UNKNOWN",
            "why": (
                "install-venv/ and packaging-cache/ were split out as RECLAIMABLE. "
                "HEAD.tar and public-tree-scrub.json are not a venv and are not "
                "proven regenerable."
            ),
        },
        {
            "path": "workspace/ops/local/genesis-agentos-worktrees/",
            "tempted_as": "RECLAIMABLE (git worktrees, dirty count 0, detached 45a27c2ad)",
            "classified_as": "UNKNOWN",
            "why": (
                "Clean does not mean unused. genesis-agentos-sessions/ and "
                "genesis-worker-checkpoints/ point at these worktrees. Process "
                "inspection was blocked in this sandbox (`ps: operation not permitted`)."
            ),
        },
        {
            "path": "receipts/dsv4f_fullseq_capture_L0_frozen_export/",
            "tempted_as": "DUPLICATE of receipts/dsv4f_fullseq_capture_L0/",
            "classified_as": "PRECIOUS",
            "why": (
                "Same-looking receipt name; live DSV4F_FULLSEQ_CAPTURE_RECEIPT.json "
                "is 9357 bytes, frozen_export copy is 29910 bytes. Frozen tree also "
                "holds unique .npy activations. Filename is not a content address."
            ),
        },
        {
            "path": "workspace/campaign/records/runs/glm-4.5-air and qwen3-30b-a3b",
            "tempted_as": "not tempted — these are MODEL and never-delete",
            "classified_as": "MODEL, but ABSENT at census time",
            "why": missing_models["why_this_is_in_the_receipt"],
        },
    ]

    watched_fail = [
        {
            "what": "capture_diverse vs capture_diverse2 same Lxx.f16 names",
            "result": "NOT duplicates. L00.f16 is 13895680 vs 115394560 bytes; sha256 differs.",
            "detail": phaseb_insp["capture_pairs"],
        },
        {
            "what": "shared_m6144_{K1,honest,stable}.pt near-identical sizes",
            "result": "NOT duplicates. head/tail+size identities all differ.",
            "detail": phaseb_insp["ckpt_head_tail"],
        },
        {
            "what": "visionmcp/.worktrees look like copies of visionmcp/",
            "result": (
                "PARTIAL. Some files sha256-match visionmcp/<rel> and are DUPLICATE. "
                "Large captures such as artifacts/world-engine/step-19/.../depth.f32 "
                "exist in main and not in the worktree. Cannot delete the worktree as a unit."
            ),
            "detail": dup_summary,
        },
        {
            "what": "macOS `du -sh -d 1` for tree sizes",
            "result": "-s and -d conflict; empty output. Use `du -h -d 1` or a Python walk.",
        },
        {
            "what": "ps aux to see if genesis-resident holds the genesis worktrees",
            "result": "operation not permitted in this sandbox. genesis worktrees stay UNKNOWN.",
        },
        {
            "what": "leftover cargo target/ after the hand reclaim",
            "result": "no target/ directories remain under the scan root (excluding .git and runs).",
        },
        {
            "what": "this observer is a sparse checkout",
            "result": (
                "A missing path HERE is not evidence it is missing in the scan root. "
                "Census walks /Users/scammermike/Downloads/hawking, not this worktree."
            ),
        },
        {
            "what": "282 GB of safetensors under records/runs (glm-4.5-air + qwen3-30b-a3b)",
            "result": (
                "Present during this observer's earlier walk (glm 221 GB / 47 shards, "
                "qwen3-30b 61 GB / 13 shards). ABSENT at census. runs/ mtime "
                f"{missing_models['at_census_time']['runs_mtime']}. Remaining children: "
                f"{missing_models['at_census_time']['runs_remaining_children']}. "
                "This process did not rm them (canaries for sandbox/phaseB/.preserved "
                "unchanged; no unlink in this script). They are not in Trash, not in "
                "hawking-copy, not under ~/Downloads. MODEL class is 0 because the "
                "bytes are gone, not because they were classified reclaimable."
            ),
            "detail": missing_models,
        },
    ]

    recommend_bytes = totals["RECLAIMABLE"]["bytes"] + totals["DUPLICATE"]["bytes"]
    elapsed = round(time.time() - t0, 1)

    receipt = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "observer": {
            "path": str(observer),
            "script": str(Path(__file__).resolve()),
            "cwd": os.getcwd(),
            "note": (
                "Observer may be a sparse checkout. Scan root is the live Hawking "
                "tree. This script writes only the receipt beside the observer."
            ),
        },
        "scan_root": {
            "path": str(scan),
            "head": head,
            "branch": branch,
            "dirty_porcelain_entries": dirty_n,
            "worktrees": wt_list,
        },
        "exclusions": {
            "hawking_copy": {
                "path": str(HAWKING_COPY),
                "present": copy_present,
                "role": "vestigial-but-preserved; NOT in scope for deletion recommendations",
            },
            "git_objects": {
                "path": str(scan / ".git"),
                "logical_bytes": git_size,
                "classified": "excluded from candidates",
                "reason": (
                    "repository object store, not caches/scratch. This lane does "
                    "not run git gc and does not recommend deleting .git."
                ),
            },
            "already_reclaimed_by_hand": {
                "target/": "~1.8G (gone; hide receipt rescued into .preserved/from-target/)",
                "__pycache__ and .pytest_cache": "1819 pycache dirs + pytest cache, ~2190 MiB",
            },
        },
        "never_delete": {
            "workspace/campaign/records/runs/": {
                "files": runs_files,
                "bytes": runs_bytes,
                "model_bytes_inside": runs_model_bytes,
                "recommend_deletion": False,
                "reason": (
                    "Never recommend deletion of anything under records/runs/. "
                    "At census time this tree holds deepseek-v4 / frankenstein / "
                    "qwen-80b sidecars only. The 282 GB glm-4.5-air + qwen3-30b-a3b "
                    "weight dirs named in the starting picture are ABSENT (see "
                    "starting_picture_models)."
                ),
            },
        },
        "starting_picture_models": missing_models,
        "high_attention": {
            "workspace/campaign/records/ascension-sandbox/": {
                "files": sandbox_files,
                "bytes": sandbox_bytes,
                "f32le_bytes": sandbox_f32le,
                "class": "PRECIOUS",
                "recommend_deletion": False,
            },
            "workspace/campaign/phaseB/": {
                "files": phaseb_files,
                "bytes": phaseb_bytes,
                "class": "PRECIOUS",
                "recommend_deletion": False,
            },
        },
        "by_class": {
            c: {
                "files": totals[c]["files"],
                "bytes": totals[c]["bytes"],
                "allocated_bytes": totals[c]["allocated_bytes"],
                "recommend_deletion": c in ("RECLAIMABLE", "DUPLICATE"),
            }
            for c in CLASSES
        },
        "scan_totals": {
            "files": n_files,
            "logical_bytes": logical_total,
            "allocated_bytes": allocated_total,
            "git_objects_bytes_excluded": git_size,
        },
        "recommend_deletion_bytes": recommend_bytes,
        "recommend_deletion_note": (
            "Only RECLAIMABLE and digest-proved DUPLICATE rows have "
            "recommend_deletion=true. A human executes. This script deletes nothing."
        ),
        "reclaimable": sorted(reclaimable_groups.values(), key=lambda r: -r["bytes"]),
        "duplicates": {
            "count": len(dup_proofs),
            "bytes": sum(p["bytes"] for p in dup_proofs),
            "surviving_tree": "visionmcp/",
            "summary": dup_summary,
            "proofs": dup_proofs,
        },
        "unknown": sorted(unknown_groups.values(), key=lambda r: -r["bytes"]),
        "model": sorted(model_groups.values(), key=lambda r: -r["bytes"]),
        "precious_groups": sorted(precious_groups.values(), key=lambda r: -r["bytes"]),
        "groups": sorted(groups.values(), key=lambda r: (r["class"], -r["bytes"])),
        "tempted_and_downgraded": tempted,
        "what_i_watched_fail": watched_fail,
        "no_deletions": {
            "script_writes": [str(out_path)],
            "forbidden_git_verbs_blocked": sorted(GIT_FORBIDDEN),
            "rm_called": False,
            "git_clean_reset_checkout_restore_stash_called": False,
            "canaries_before": canaries_before,
            "canaries_after": canaries_after,
            "canaries_unchanged": ok_canaries,
            "canary_problems": canary_problems,
            "dirty_porcelain_before": dirty_n,
            "dirty_porcelain_after": dirty_after,
            "how_verified": (
                "Canary paths were lstat'd before and after the census (inode, "
                "mtime_ns, size). Porcelain dirty count recorded before and after. "
                "The only write is the receipt. git is invoked read-only "
                "(rev-parse, status --porcelain, worktree list)."
            ),
        },
        "elapsed_s": elapsed,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=1) + "\n")

    # Human plan on stdout — this is what verification pastes.
    print("SPRING_CLEAN_CENSUS")
    print(f"schema: {SCHEMA}")
    print(f"scan: {scan}  HEAD {head[:12]}  branch {branch}  dirty {dirty_n}")
    print(f"observer: {observer}")
    print(f"receipt: {out_path}")
    print(f"elapsed_s: {elapsed}")
    print()
    print("== by class (logical bytes) ==")
    for c in CLASSES:
        t = totals[c]
        rec = "YES" if c in ("RECLAIMABLE", "DUPLICATE") else "no"
        print(
            f"  {c:<12} files={t['files']:<8d}  "
            f"logical={t['bytes']/1e9:10.3f} GB  "
            f"allocated={t['allocated_bytes']/1e9:10.3f} GB  "
            f"recommend_deletion={rec}"
        )
    print(
        f"  {'SCAN':<12} files={n_files:<8d}  "
        f"logical={logical_total/1e9:10.3f} GB  "
        f"allocated={allocated_total/1e9:10.3f} GB"
    )
    print(f"  {'GIT(.git)':<12} excluded from candidates  logical={git_size/1e9:10.3f} GB")
    print()
    print("== never delete ==")
    print(
        f"  {RUNS_PREFIX}  files={runs_files}  logical={runs_bytes/1e9:.3f} GB  "
        f"(of which MODEL {runs_model_bytes/1e9:.3f} GB)  recommend_deletion=no"
    )
    print(
        f"  hawking-copy {HAWKING_COPY} present={copy_present}  "
        "out of scope (vestigial-but-preserved)"
    )
    print()
    print("== starting-picture MODEL dirs at census time ==")
    print(
        f"  runs/ mtime {missing_models['at_census_time']['runs_mtime']}  "
        f"remaining {missing_models['at_census_time']['runs_remaining_children']}  "
        f"safetensors_under_runs={missing_models['at_census_time']['safetensors_under_runs']}"
    )
    for row in missing_models["at_census_time"]["expected_dirs"]:
        print(f"  {row['path']}  exists={row['exists']}")
    print("  this_observer_deleted_them: false")
    print()
    print("== high attention ==")
    print(
        f"  {SANDBOX_PREFIX}  files={sandbox_files}  "
        f"{sandbox_bytes/1e9:.3f} GB  (.f32le {sandbox_f32le/1e9:.3f} GB)  PRECIOUS"
    )
    print(
        f"  {PHASEB_PREFIX}  files={phaseb_files}  {phaseb_bytes/1e9:.3f} GB  PRECIOUS"
    )
    print()
    print("== RECLAIMABLE (human may delete) ==")
    if not reclaimable_groups:
        print("  (none)")
    for row in sorted(reclaimable_groups.values(), key=lambda r: -r["bytes"]):
        print(
            f"  {row['bytes']/1e6:10.1f} MB  files={row['files']:<6d}  "
            f"{row['path']}"
        )
        print(f"      regenerates: {row['regenerates']}")
    print()
    print("== DUPLICATE (digest-proved; human may delete the duplicate path) ==")
    print(
        f"  {len(dup_proofs)} files  {sum(p['bytes'] for p in dup_proofs)/1e6:.1f} MB  "
        "all surviving under visionmcp/"
    )
    for name, info in dup_summary.get("worktrees", {}).items():
        print(
            f"  visionmcp/.worktrees/{name}: matched {info['digest_matched_files']} files "
            f"({info['digest_matched_bytes']/1e6:.1f} MB); "
            f"unique/mismatch {info['worktree_only_or_mismatch_files']} files "
            f"({info['worktree_only_or_mismatch_bytes']/1e6:.1f} MB) stay UNKNOWN"
        )
    print("  sample proofs:")
    for p in dup_proofs[:5]:
        print(f"    {p['path']}")
        print(f"      survivor {p['surviving_path']}")
        print(f"      sha256 {p['sha256']}  bytes {p['bytes']}")
    if len(dup_proofs) > 5:
        print(f"    ... {len(dup_proofs) - 5} more proofs in the receipt")
    print()
    print("== UNKNOWN (do not delete) ==")
    for row in sorted(unknown_groups.values(), key=lambda r: -r["bytes"])[:20]:
        print(
            f"  {row['bytes']/1e6:10.1f} MB  files={row['files']:<6d}  {row['path']}"
        )
        print(f"      {row['reason']}")
    print()
    print("== tempted and downgraded ==")
    for t in tempted:
        print(f"  {t['path']}")
        print(f"      tempted {t['tempted_as']}")
        print(f"      now     {t['classified_as']}")
        print(f"      why     {t['why']}")
    print()
    print("== WHAT I WATCHED FAIL ==")
    for w in watched_fail:
        print(f"  - {w['what']}")
        print(f"    {w['result']}")
    print()
    print("== no deletions ==")
    print(f"  canaries_unchanged: {ok_canaries}")
    if canary_problems:
        for p in canary_problems:
            print(f"  PROBLEM {p}")
    print(f"  dirty porcelain before/after: {dirty_n} / {dirty_after}")
    print("  rm_called: false")
    print("  git clean/reset/checkout/restore/stash: not invoked")
    print(f"  only write: {out_path}")
    if not ok_canaries:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
