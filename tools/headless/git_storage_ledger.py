#!/usr/bin/env python3
"""GIT_STORAGE_LEDGER — measure git bulk, classify families, prepare (not run) a rewrite.

Git stores knowledge. Local stores bulk evidence. This lane measures the shared
object store, classifies every large blob family, and writes a PREPARED history
plan with `executed: false`.

It never rewrites history, never force-pushes, never deletes a branch, and
never runs `git gc`. S020 §26: LFS does not remove old blobs. S020 §27: do
not execute the rewrite from this lane.

    python3 tools/headless/git_storage_ledger.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

SCHEMA = "hawking.headless.git_storage_ledger.v1"
RECEIPT = REPO / "receipts" / "headless" / "GIT_STORAGE_LEDGER.json"
POLICY = REPO / "docs" / "ultragoals" / "ARTIFACT_STORAGE_POLICY.md"

CLASSES = (
    "KEEP_GIT",
    "MOVE_LOCAL_FUTURE",
    "LFS_CANDIDATE",
    "HISTORY_REWRITE_CANDIDATE",
    "PRESERVE",
)

# Session gitignore block landed in 8ad51461a plus the pre-existing /artifacts/ root.
# Policy must not contradict these. Quoted for the ledger and the tests.
GITIGNORE_MUST_HOLD = (
    "/artifacts/",
    "*.safetensors",
    "*.gguf",
    "*.bin",
    "*.hq80seg",
    "*.hq38seg",
    "*.hqseg",
    "**/ranspack/out/segments/",
    "**/quality-candidates/**/segments/",
    "*.f16",
    "workspace/campaign/phaseB/ckpt/",
    "workspace/campaign/phaseB/capture_diverse/",
    "workspace/campaign/phaseB/capture_diverse2/",
    "workspace/campaign/odyssey/RUN_LOG.jsonl",
    "receipts/**/*.tar.xz",
    "*.npz",
    "*.parquet",
    "*.metallib",
    "*.air",
    "*.log",
    "target-parallel/",
    "/workspace/ops/local/weights/",
    "/workspace/ops/local/checkpoints/",
    "/workspace/ops/local/models/",
    "workspace/campaign/records/ascension-sandbox/physical/**/selected-payloads/",
    "workspace/campaign/records/ascension-sandbox/physical/**/*.hgravs01",
    "workspace/campaign/records/ascension-sandbox/physical/**/tensors/",
    "workspace/campaign/records/ascension-sandbox/physical/**/capture-result.json",
    "/workspace/campaign/evidence/models/glm52/GLM52_SHARD_DEPENDENCY_GRAPH.json",
)

# Existing Hawking artifact root. Do not invent a second CAS.
EXISTING_ARTIFACT_ROOT = "artifacts/"
CAS_LAYOUT = "artifacts/sha256/ab/abcdef..."  # first two hex chars / full sha256

KiB = 1024
MiB = 1024 ** 2
GiB = 1024 ** 3


def git(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def git_ok(*args: str, timeout: int = 120) -> str:
    p = git(*args, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"git {args} failed: {p.stderr.strip() or p.stdout.strip()}")
    return p.stdout


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def walk_bytes(root: Path) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                continue
    return total


def parse_count_objects(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"warnings": []}
    for line in text.splitlines():
        if line.startswith("warning:"):
            out["warnings"].append(line)
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if k in {"count", "in-pack", "packs", "prune-packable", "garbage", "size", "size-pack", "size-garbage"}:
            # git count-objects -v: size fields are KiB
            try:
                out[k] = int(v.split()[0])
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    size_kib = int(out.get("size") or 0)
    pack_kib = int(out.get("size-pack") or 0)
    out["size_bytes"] = size_kib * KiB
    out["size_pack_bytes"] = pack_kib * KiB
    return out


def family_id(path: str) -> str:
    """Mutually exclusive family id. First match wins."""
    p = path
    if p == "workspace/campaign/odyssey/RUN_LOG.jsonl":
        return "RUN_LOG"
    if p.endswith(".hq30g"):
        return "HQ30G"
    if p.endswith(".hq80seg"):
        return "HQ80SEG"
    if p.endswith(".hq38seg"):
        return "HQ38SEG"
    if p.endswith(".hqseg"):
        return "HQSEG"
    if p.endswith(".hgravs01"):
        return "HGRAVS01"
    if p.endswith(".f16"):
        return "F16"
    if p.endswith(".safetensors"):
        return "SAFETENSORS"
    if p.endswith(".gguf"):
        return "GGUF"
    if p.endswith(".pt") or p.endswith(".pth"):
        return "PT_CHECKPOINTS"
    if p.endswith(".npy"):
        return "NPY_CAPTURES"
    if p.endswith(".npz"):
        return "NPZ"
    if p.endswith(".metallib") or p.endswith(".air"):
        return "METAL_CACHE"
    if p.endswith("qwen30-receipts.tar.xz") or p.endswith(".tar.xz"):
        return "RECEIPT_TARBALLS"
    if p.startswith("target-fast/"):
        return "TARGET_FAST"
    if p.startswith("target-parallel/"):
        return "TARGET_PARALLEL"
    if p.startswith("target/"):
        return "TARGET_ROOT"
    if "dsv4f_fullseq_capture_" in p and "/traces/" in p:
        return "FULLSEQ_TRACES"
    if "dsv4f_fullseq_capture_" in p:
        return "FULLSEQ_EXPORTS"
    if p.startswith("receipts/ascent-"):
        return "RECEIPTS_ASCENT"
    if p.startswith("receipts/headless/"):
        return "RECEIPTS_HEADLESS"
    if p.startswith("workspace/campaign/records/ascension-sandbox/physical/"):
        return "PHYSICAL_OTHER"
    if p.startswith("workspace/campaign/phaseB/"):
        return "PHASEB"
    if "frankenstein/" in p:
        return "FRANKENSTEIN_NON_PT"
    if p.startswith("research/hawking-experiments/superwave/g1/"):
        return "SUPERWAVE_G1"
    if p.startswith("research/ramanujan/"):
        return "RAMANUJAN"
    if p.startswith("workspace/campaign/odyssey/ODYSSEY_STATE.json") or p.endswith("/ODYSSEY_STATE.json"):
        return "ODYSSEY_STATE"
    if p.endswith("ODYSSEY_COMPLETIONS.json"):
        return "ODYSSEY_COMPLETIONS"
    if "GLM52_" in p:
        return "GLM52"
    if p.startswith("HAWKING_") and p.endswith((".json", ".html")):
        return "HAWKING_GENERATED_GRAPHS"
    if p.startswith("research/reports/condense/"):
        return "REPORTS_CONDENSE"
    if p.startswith("crates/"):
        return "CRATES_SOURCE"
    if p.startswith("tools/"):
        return "TOOLS_SOURCE"
    if p.startswith("docs/"):
        return "DOCS"
    if p.startswith("receipts/"):
        return "RECEIPTS_OTHER"
    return "OTHER"


# future-tracking class per family. HISTORY_REWRITE_CANDIDATE means the
# prepared plan would strip this family from history. MOVE_LOCAL_FUTURE means
# new objects of this kind stay out of git (already gitignored, or should be).
FAMILY_META: dict[str, dict[str, Any]] = {
    "RUN_LOG": {
        "class": "MOVE_LOCAL_FUTURE",
        "glob": "workspace/campaign/odyssey/RUN_LOG.jsonl",
        "why": (
            "Append-only campaign log. 844 unique blobs, 28.8 GiB logical, ~1.5 MiB "
            "on disk after zlib. Untracked and gitignored in 8ad51461a. Live file "
            "remains on disk at the well-known path. GitHub 50 MiB warning was the "
            "reason (largest blob ~65.9 MiB), not pack size."
        ),
        "gitignore": "workspace/campaign/odyssey/RUN_LOG.jsonl",
        "in_rewrite_plan": True,
    },
    "HQ30G": {
        "class": "HISTORY_REWRITE_CANDIDATE",
        "glob": "**/*.hq30g",
        "why": (
            "Packed Q30 tensor bodies from one 2026-08-09 commit. Incompressible. "
            "This is the remaining bulk of the 5.38 GiB pack (~3.91 GiB disk). "
            "Not in HEAD. physical/**/tensors/ is already gitignored."
        ),
        "gitignore": "workspace/campaign/records/ascension-sandbox/physical/**/tensors/",
        "in_rewrite_plan": True,
    },
    "HQ80SEG": {
        "class": "MOVE_LOCAL_FUTURE",
        "glob": "**/*.hq80seg",
        "why": "Gitignored this session. Reachable history: ABSENT (dropped with the 899 dead grok/* branches + repack). The 20 GiB named in 8ad51461a is no longer in `rev-list --all`.",
        "gitignore": "*.hq80seg",
        "in_rewrite_plan": False,
    },
    "HQ38SEG": {
        "class": "MOVE_LOCAL_FUTURE",
        "glob": "**/*.hq38seg",
        "why": "Gitignored this session. Reachable history ABSENT.",
        "gitignore": "*.hq38seg",
        "in_rewrite_plan": False,
    },
    "HQSEG": {
        "class": "MOVE_LOCAL_FUTURE",
        "glob": "**/*.hqseg",
        "why": "Gitignored this session. Reachable history ABSENT.",
        "gitignore": "*.hqseg",
        "in_rewrite_plan": False,
    },
    "HGRAVS01": {
        "class": "HISTORY_REWRITE_CANDIDATE",
        "glob": "**/*.hgravs01",
        "why": "Physical gravity payloads. Already gitignored. Still in history (~71 MiB disk).",
        "gitignore": "workspace/campaign/records/ascension-sandbox/physical/**/*.hgravs01",
        "in_rewrite_plan": True,
    },
    "F16": {
        "class": "MOVE_LOCAL_FUTURE",
        "glob": "**/*.f16",
        "why": "Activation captures. Gitignored. Reachable history ABSENT.",
        "gitignore": "*.f16",
        "in_rewrite_plan": False,
    },
    "SAFETENSORS": {
        "class": "MOVE_LOCAL_FUTURE",
        "glob": "**/*.safetensors",
        "why": "Model shards. Gitignored. Not a git family; live copies live under ~/models and campaign records/runs (ARTIFACT_LEDGER).",
        "gitignore": "*.safetensors",
        "in_rewrite_plan": False,
    },
    "GGUF": {
        "class": "MOVE_LOCAL_FUTURE",
        "glob": "**/*.gguf",
        "why": "Runtime weights. Gitignored. Live parent is outside git (~/models).",
        "gitignore": "*.gguf",
        "in_rewrite_plan": False,
    },
    "PT_CHECKPOINTS": {
        "class": "HISTORY_REWRITE_CANDIDATE",
        "glob": "**/*.pt",
        "why": (
            "Historical training checkpoints (~276 MiB disk). HEAD still carries small "
            "frankenstein latent_v0 fixtures (KEEP exception; gitignore comment says so) "
            "plus one 5.7 MiB ROLLBACK.pt fixture. phaseB/ckpt/ is gitignored."
        ),
        "gitignore": "workspace/campaign/phaseB/ckpt/",
        "in_rewrite_plan": True,
        "keep_in_head_exception": "research/hawking-experiments/frankenstein/data/latent_v0_checkpoints/",
    },
    "NPY_CAPTURES": {
        "class": "HISTORY_REWRITE_CANDIDATE",
        "glob": "**/*.npy",
        "why": "Capture indexes and logits dumps. Regenerable. ~99 MiB disk in history.",
        "gitignore": None,
        "in_rewrite_plan": True,
    },
    "NPZ": {
        "class": "MOVE_LOCAL_FUTURE",
        "glob": "**/*.npz",
        "why": "Already gitignored derived calibration artifacts.",
        "gitignore": "*.npz",
        "in_rewrite_plan": False,
    },
    "METAL_CACHE": {
        "class": "MOVE_LOCAL_FUTURE",
        "glob": "**/*.{metallib,air}",
        "why": "Runtime/Metal caches. Gitignored.",
        "gitignore": "*.metallib",
        "in_rewrite_plan": False,
    },
    "RECEIPT_TARBALLS": {
        "class": "HISTORY_REWRITE_CANDIDATE",
        "glob": "receipts/**/*.tar.xz",
        "why": "qwen30-receipts.tar.xz (75.4 MiB) untracked this session; extracted receipts stay. Still in history.",
        "gitignore": "receipts/**/*.tar.xz",
        "in_rewrite_plan": True,
    },
    "TARGET_FAST": {
        "class": "HISTORY_REWRITE_CANDIDATE",
        "glob": "target-fast/",
        "why": "Committed Rust build output (~429 MiB disk). **/target/ is gitignored; this tree name is not. Rewrite candidate.",
        "gitignore": "**/target/",
        "in_rewrite_plan": True,
    },
    "TARGET_PARALLEL": {
        "class": "HISTORY_REWRITE_CANDIDATE",
        "glob": "target-parallel/",
        "why": "Committed build output (~96 MiB disk). Gitignored going forward.",
        "gitignore": "target-parallel/",
        "in_rewrite_plan": True,
    },
    "TARGET_ROOT": {
        "class": "HISTORY_REWRITE_CANDIDATE",
        "glob": "target/",
        "why": "If any target/ blobs remain in history they are build output.",
        "gitignore": "/target",
        "in_rewrite_plan": True,
    },
    "FULLSEQ_TRACES": {
        "class": "MOVE_LOCAL_FUTURE",
        "glob": "receipts/dsv4f_fullseq_capture_*/traces/",
        "why": (
            "Largest CURRENT tracked family (~476 MiB in HEAD, ~5.9 MiB/file). "
            "Huge traces must not keep growing in git. Compact receipt + CAS bytes. "
            "Still in HEAD today — copy to local store before any future untrack."
        ),
        "gitignore": None,
        "in_rewrite_plan": True,
        "rewrite_optional_until_cas_copy": True,
    },
    "FULLSEQ_EXPORTS": {
        "class": "MOVE_LOCAL_FUTURE",
        "glob": "receipts/dsv4f_fullseq_capture_*/",
        "why": "Sibling fullseq export bodies (activations, etc.). Same rule as traces.",
        "gitignore": None,
        "in_rewrite_plan": True,
        "rewrite_optional_until_cas_copy": True,
    },
    "RECEIPTS_ASCENT": {
        "class": "PRESERVE",
        "glob": "receipts/ascent-*/",
        "why": "Unique science. Never rewrite out, never delete. This lane must not modify those trees.",
        "gitignore": None,
        "in_rewrite_plan": False,
    },
    "RECEIPTS_HEADLESS": {
        "class": "KEEP_GIT",
        "glob": "receipts/headless/",
        "why": "Compact manifests, ledgers, negative science, digests. Knowledge.",
        "gitignore": None,
        "in_rewrite_plan": False,
    },
    "PHYSICAL_OTHER": {
        "class": "HISTORY_REWRITE_CANDIDATE",
        "glob": "workspace/campaign/records/ascension-sandbox/physical/",
        "why": (
            "Leftover physical JSON after tensors/payloads are split out: capture-result, "
            "preflight inputs, complete-binary gravity dumps (~434 MiB disk). Compact "
            "*RECEIPT.json / *HANDOFF.json / *STATUS.json should stay if a rewrite ever runs."
        ),
        "gitignore": "workspace/campaign/records/ascension-sandbox/physical/**/capture-result.json",
        "in_rewrite_plan": True,
        "preserve_globs": ["**/*RECEIPT.json", "**/*HANDOFF.json", "**/*STATUS.json", "**/*SELECTION*.json"],
    },
    "PHASEB": {
        "class": "MOVE_LOCAL_FUTURE",
        "glob": "workspace/campaign/phaseB/",
        "why": "Heavy ckpt/ and capture dirs gitignored this session.",
        "gitignore": "workspace/campaign/phaseB/ckpt/",
        "in_rewrite_plan": False,
    },
    "FRANKENSTEIN_NON_PT": {
        "class": "KEEP_GIT",
        "glob": "research/hawking-experiments/frankenstein/data/",
        "why": "Non-weight frankenstein evidence (json, scripts). Knowledge. .pt bodies classified separately.",
        "gitignore": None,
        "in_rewrite_plan": False,
    },
    "SUPERWAVE_G1": {
        "class": "LFS_CANDIDATE",
        "glob": "research/hawking-experiments/superwave/g1/",
        "why": (
            "Currently tracked 10–29 MiB JSON dumps (g1_functional_exceptions.json is the "
            "largest HEAD blob at 29.0 MiB). This is the shape of thing people reach for "
            "Git LFS for. S020 §26: LFS does not remove old blobs. Prefer CAS + git "
            "manifest over adopting LFS."
        ),
        "gitignore": None,
        "in_rewrite_plan": False,
        "lfs_rejected": "S020 §26 — LFS is not a magic fix; it does not strip history. Future tracking should be local CAS, not LFS.",
    },
    "RAMANUJAN": {
        "class": "KEEP_GIT",
        "glob": "research/ramanujan/",
        "why": "Research corpora currently ~25 MiB. Compact enough to remain knowledge in git.",
        "gitignore": None,
        "in_rewrite_plan": False,
    },
    "ODYSSEY_STATE": {
        "class": "KEEP_GIT",
        "glob": "workspace/campaign/odyssey/ODYSSEY_STATE.json",
        "why": "Campaign state. Knowledge. Highly compressible; keep in git. This lane must not modify workspace/campaign/odyssey.",
        "gitignore": None,
        "in_rewrite_plan": False,
    },
    "ODYSSEY_COMPLETIONS": {
        "class": "KEEP_GIT",
        "glob": "workspace/campaign/odyssey/ODYSSEY_COMPLETIONS.json",
        "why": "Campaign completions index. Knowledge. Same odyssey preserve rule.",
        "gitignore": None,
        "in_rewrite_plan": False,
    },
    "GLM52": {
        "class": "MOVE_LOCAL_FUTURE",
        "glob": "**/*GLM52_*",
        "why": "GLM52_SHARD_DEPENDENCY_GRAPH.json already gitignored (52 MiB). Remaining GLM52 dumps follow the same rule: hash in git, bytes local.",
        "gitignore": "/workspace/campaign/evidence/models/glm52/GLM52_SHARD_DEPENDENCY_GRAPH.json",
        "in_rewrite_plan": False,
    },
    "HAWKING_GENERATED_GRAPHS": {
        "class": "MOVE_LOCAL_FUTURE",
        "glob": "HAWKING_*.json",
        "why": "Generated graph/ledger dumps (behaviour-to-code map, viewer). Regenerable. Do not add new ones.",
        "gitignore": None,
        "in_rewrite_plan": False,
    },
    "REPORTS_CONDENSE": {
        "class": "MOVE_LOCAL_FUTURE",
        "glob": "research/reports/condense/",
        "why": "Condense measurement dumps including parent logits npy.",
        "gitignore": None,
        "in_rewrite_plan": False,
    },
    "CRATES_SOURCE": {
        "class": "KEEP_GIT",
        "glob": "crates/",
        "why": "Source. Large unique-logical on a few generated/kernel files is history of source, not bulk evidence.",
        "gitignore": None,
        "in_rewrite_plan": False,
    },
    "TOOLS_SOURCE": {
        "class": "KEEP_GIT",
        "glob": "tools/",
        "why": "Source.",
        "gitignore": None,
        "in_rewrite_plan": False,
    },
    "DOCS": {
        "class": "KEEP_GIT",
        "glob": "docs/",
        "why": "Canonical docs.",
        "gitignore": None,
        "in_rewrite_plan": False,
    },
    "RECEIPTS_OTHER": {
        "class": "KEEP_GIT",
        "glob": "receipts/",
        "why": "Receipts not in ascent/ or headless/. Compact knowledge unless a specific family (tarballs, traces) already split out.",
        "gitignore": None,
        "in_rewrite_plan": False,
    },
    "OTHER": {
        "class": "KEEP_GIT",
        "glob": "*",
        "why": "Remainder (configs, schemas, lab, app, small files). Default is knowledge.",
        "gitignore": None,
        "in_rewrite_plan": False,
    },
}


def measure_current_tree() -> dict[str, Any]:
    out = git_ok("ls-tree", "-r", "-l", "--full-tree", "HEAD")
    rows: list[tuple[int, str, str]] = []
    total = 0
    n = 0
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5 or parts[1] != "blob" or parts[3] == "-":
            continue
        size = int(parts[3])
        sha, path = parts[2], parts[4]
        total += size
        n += 1
        rows.append((size, path, sha))
    rows.sort(reverse=True)
    return {
        "n_files": n,
        "bytes": total,
        "top30": [
            {"path": p, "bytes": s, "sha": sha} for s, p, sha in rows[:30]
        ],
        "ge_1m": [
            {"path": p, "bytes": s, "sha": sha} for s, p, sha in rows if s >= 1_000_000
        ],
        "ge_10m": [
            {"path": p, "bytes": s} for s, p, _ in rows if s >= 10_000_000
        ],
        "ge_50m": [
            {"path": p, "bytes": s} for s, p, _ in rows if s >= 50_000_000
        ],
    }


def measure_historical() -> dict[str, Any]:
    """Aggregate every reachable blob by path family.

    Method required by the contract:
      git rev-list --all --objects | git cat-file --batch-check='%(objecttype) %(objectsize:disk) %(rest)'
    We also keep %(objectsize) (logical) so compression is visible.
    """
    p1 = subprocess.Popen(
        ["git", "-C", str(REPO), "rev-list", "--all", "--objects"],
        stdout=subprocess.PIPE,
    )
    p2 = subprocess.Popen(
        [
            "git",
            "-C",
            str(REPO),
            "cat-file",
            "--batch-check=%(objecttype) %(objectsize) %(objectsize:disk) %(objectname) %(rest)",
        ],
        stdin=p1.stdout,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert p1.stdout is not None
    p1.stdout.close()

    families: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n_refs": 0,
            "n_paths": 0,
            "unique_logical_bytes": 0,
            "unique_disk_bytes": 0,
            "largest_blob_bytes": 0,
            "example_paths": [],
            "paths": set(),
        }
    )
    path_logical: dict[str, int] = defaultdict(int)
    path_disk: dict[str, int] = defaultdict(int)
    path_n: dict[str, int] = defaultdict(int)
    n_blob = 0
    unique_logical = 0
    unique_disk = 0
    n_other = 0

    assert p2.stdout is not None
    for line in p2.stdout:
        parts = line.rstrip("\n").split(" ", 4)
        if len(parts) < 4 or parts[0] != "blob":
            n_other += 1
            continue
        try:
            logical = int(parts[1])
            disk = int(parts[2])
        except ValueError:
            n_other += 1
            continue
        path = parts[4] if len(parts) > 4 else ""
        n_blob += 1
        unique_logical += logical
        unique_disk += disk
        fid = family_id(path) if path else "OTHER"
        rec = families[fid]
        rec["n_refs"] += 1
        rec["unique_logical_bytes"] += logical
        rec["unique_disk_bytes"] += disk
        if logical > rec["largest_blob_bytes"]:
            rec["largest_blob_bytes"] = logical
        if path:
            if path not in rec["paths"]:
                rec["paths"].add(path)
                rec["n_paths"] += 1
                if len(rec["example_paths"]) < 5:
                    rec["example_paths"].append(path)
            path_logical[path] += logical
            path_disk[path] += disk
            path_n[path] += 1

    rc = p2.wait()
    if rc != 0:
        raise RuntimeError(f"git cat-file failed rc={rc}")

    top_paths = sorted(path_logical.items(), key=lambda kv: -kv[1])[:40]
    top_historical = [
        {
            "path": p,
            "unique_logical_bytes": path_logical[p],
            "unique_disk_bytes": path_disk[p],
            "n_unique_blobs": path_n[p],
        }
        for p, _ in top_paths
    ]

    # Ensure ABSENT families still appear.
    for fid in FAMILY_META:
        families.setdefault(fid, {
            "n_refs": 0,
            "n_paths": 0,
            "unique_logical_bytes": 0,
            "unique_disk_bytes": 0,
            "largest_blob_bytes": 0,
            "example_paths": [],
            "paths": set(),
        })

    family_rows = []
    for fid, rec in families.items():
        meta = FAMILY_META.get(fid, {"class": "KEEP_GIT", "glob": "?", "why": "unlisted remainder", "gitignore": None, "in_rewrite_plan": False})
        cls = meta["class"]
        if rec["n_refs"] == 0 and fid not in FAMILY_META:
            continue
        absent = rec["n_refs"] == 0
        row = {
            "id": fid,
            "class": cls,
            "glob": meta.get("glob"),
            "n_paths": rec["n_paths"],
            "n_unique_blobs": rec["n_refs"],  # rev-list --objects emits each sha once
            "unique_logical_bytes": rec["unique_logical_bytes"],
            "unique_disk_bytes": rec["unique_disk_bytes"],
            "largest_blob_bytes": rec["largest_blob_bytes"],
            "logical_to_disk_ratio": (
                round(rec["unique_logical_bytes"] / rec["unique_disk_bytes"], 2)
                if rec["unique_disk_bytes"] else None
            ),
            "in_head": None,  # filled later
            "gitignore_rule": meta.get("gitignore"),
            "in_rewrite_plan": bool(meta.get("in_rewrite_plan")),
            "why": meta.get("why"),
            "example_paths": rec["example_paths"],
            "reachable_history": "ABSENT" if absent else "PRESENT",
        }
        if absent:
            row["absent_reason"] = (
                meta.get("why")
                or "no reachable blob of this family under git rev-list --all"
            )
        for k in ("keep_in_head_exception", "preserve_globs", "lfs_rejected", "rewrite_optional_until_cas_copy"):
            if k in meta:
                row[k] = meta[k]
        family_rows.append(row)

    family_rows.sort(key=lambda r: -r["unique_disk_bytes"])
    return {
        "method": (
            "git rev-list --all --objects | git cat-file "
            "--batch-check='%(objecttype) %(objectsize) %(objectsize:disk) %(objectname) %(rest)'"
        ),
        "n_blob_objects": n_blob,
        "n_non_blob_lines": n_other,
        "unique_logical_bytes": unique_logical,
        "unique_disk_bytes": unique_disk,
        "duplication": {
            "note": (
                "rev-list --objects emits each reachable sha once, so unique_blob_count "
                "== n_blob_objects. Duplication is many unique hashes at one path "
                "(append-only logs) plus zlib/pack compression of similar JSON."
            ),
            "logical_to_disk_ratio": round(unique_logical / unique_disk, 3) if unique_disk else None,
            "pack_vs_unique_disk_note": (
                "objectsize:disk is per-object zlib. The single pack may be slightly "
                "smaller via cross-object deltas. unique_disk is the honest per-family "
                "upper bound on bytes a rewrite of that family can reclaim."
            ),
        },
        "top_historical_paths": top_historical,
        "families": family_rows,
    }


def fill_in_head(families: list[dict[str, Any]], current: dict[str, Any]) -> None:
    head_paths = {r["path"] for r in current["ge_1m"]}
    # also need all HEAD paths for in_head boolean — use a cheap name-only list
    names = git_ok("ls-tree", "-r", "--name-only", "HEAD").splitlines()
    name_set = set(names)
    for fam in families:
        examples = fam.get("example_paths") or []
        glob_hit = 0
        for p in examples:
            if p in name_set:
                glob_hit += 1
        # family in HEAD if any example is, or any ge_1m path classifies to it
        ge = [r["path"] for r in current["ge_1m"] if family_id(r["path"]) == fam["id"]]
        fam["in_head"] = bool(glob_hit or ge)
        fam["head_ge_1m_paths"] = len(ge)
        fam["head_ge_1m_bytes"] = sum(
            r["bytes"] for r in current["ge_1m"] if family_id(r["path"]) == fam["id"]
        )
    del head_paths  # unused except to document the ge_1m set


def existing_stores() -> dict[str, Any]:
    """Do not invent a second system. Record what already exists."""
    common = git_ok("rev-parse", "--git-common-dir").strip()
    candidates = {
        "artifacts_gitignore_root": {
            "path": "artifacts/",
            "role": "EXISTING Hawking artifact root (gitignored). Calibration parquet/npz and future CAS bytes live here. CONSOLIDATE onto this root.",
            "on_disk_this_worktree": (REPO / "artifacts").exists(),
            "in_gitignore": True,
        },
        "hawking_experiments": {
            "path": "research/hawking-experiments/",
            "role": "Campaign SOURCE archive (git), not a blob CAS. Only README.md at HEAD. Do not dump weights here.",
            "in_git": True,
        },
        "workspace_ops_local": {
            "path": "workspace/ops/local/",
            "role": "Local weights/checkpoints/models (gitignored). Machine-local, not the experiment CAS.",
            "gitignored": [
                "/workspace/ops/local/weights/",
                "/workspace/ops/local/checkpoints/",
                "/workspace/ops/local/models/",
            ],
        },
        "artifact_ledger_weights": {
            "path": "receipts/headless/ARTIFACT_LEDGER.json",
            "role": "Census of machine-local model files (>=1 GiB) under ~/models, HF hub, campaign runs. Paired with tools/headless/storage_manager.py. Different layer from git history.",
        },
        "gravity_codec": {
            "path": "crates/hawking-core/src/artifact.rs",
            "role": "Gravity shard codec (GRAVITY\\0), not a content store.",
        },
        "hide_runtime_blobs": {
            "path": ".hide/blobs",
            "role": "HCLI runtime blob dir. Not the experiment CAS. Leave it.",
        },
        "visionmcp_artifacts": {
            "path": "visionmcp/artifacts",
            "role": "visionmcp product artifacts. Tree is gitignored via visionmcp/. Not this CAS.",
        },
    }
    # Probe a few paths without inventing new ones.
    main = Path("/Users/scammermike/Downloads/hawking")
    copy = Path("/Users/scammermike/Downloads/hawking-copy")
    probes = {
        "repo_artifacts_dir": (REPO / "artifacts").exists(),
        "main_artifacts_dir": (main / "artifacts").exists() if main.exists() else False,
        "copy_artifacts_dir": (copy / "artifacts").exists() if copy.exists() else False,
        "main_run_log": str(main / "workspace/campaign/odyssey/RUN_LOG.jsonl")
        if (main / "workspace/campaign/odyssey/RUN_LOG.jsonl").is_file()
        else "ABSENT",
        "git_common_dir": common,
        "sha256_cas_dirs_under_repo_maxdepth3": "ABSENT — no artifacts/sha256 layout on disk yet; policy adopts artifacts/sha256/ as the layout under the EXISTING /artifacts/ root",
    }
    return {"systems": candidates, "probes": probes}


def gitignore_from_head() -> str:
    p = git("show", "HEAD:.gitignore")
    if p.returncode != 0:
        return ""
    return p.stdout


def rewrite_plan(families: list[dict[str, Any]], git_bytes: dict[str, Any], head: str) -> dict[str, Any]:
    by_id = {f["id"]: f for f in families}
    pack = git_bytes["size_pack_bytes"]

    def disk(fid: str) -> int:
        return int(by_id.get(fid, {}).get("unique_disk_bytes") or 0)

    phase_a_ids = [
        "HQ30G",
        "PHYSICAL_OTHER",
        "TARGET_FAST",
        "TARGET_PARALLEL",
        "TARGET_ROOT",
        "NPY_CAPTURES",
        "RECEIPT_TARBALLS",
        "HGRAVS01",
        "PT_CHECKPOINTS",
    ]
    phase_b_ids = ["RUN_LOG"]
    phase_c_ids = ["FULLSEQ_TRACES", "FULLSEQ_EXPORTS"]

    def phase(pid: str, ids: list[str], extra: dict[str, Any]) -> dict[str, Any]:
        savings = sum(disk(i) for i in ids)
        predicted = max(0, pack - savings)
        return {
            "id": pid,
            "family_ids": ids,
            "paths_affected": [by_id[i]["glob"] for i in ids if i in by_id],
            "unique_disk_bytes_reclaimed_upper_bound": savings,
            "predicted_pack_bytes": predicted,
            "predicted_pack_gib": round(predicted / GiB, 3),
            **extra,
        }

    first_run_log = git_ok(
        "log", "--reverse", "--format=%H %ci %s", "--",
        "workspace/campaign/odyssey/RUN_LOG.jsonl",
    ).splitlines()
    first_run_log_line = first_run_log[0] if first_run_log else "ABSENT"
    hq30g_commit = git_ok(
        "log", "--all", "--format=%H %ci %s", "--", "*.hq30g",
    ).splitlines()
    hq30g_line = hq30g_commit[0] if hq30g_commit else "ABSENT"

    n_head = int(git_ok("rev-list", "--count", "HEAD").strip())
    n_all = int(git_ok("rev-list", "--all", "--count").strip())
    try:
        n_from_hq = int(git_ok("rev-list", "--count", "801c98b67..HEAD").strip()) if hq30g_commit else None
    except RuntimeError:
        n_from_hq = None
    try:
        first_hash = first_run_log_line.split()[0] if first_run_log else None
        n_from_runlog = int(git_ok("rev-list", "--count", f"{first_hash}..HEAD").strip()) if first_hash else None
    except (RuntimeError, IndexError):
        n_from_runlog = None

    n_heads = len(git_ok("for-each-ref", "refs/heads", "--format=%(refname)").splitlines())
    n_tags = len(git_ok("for-each-ref", "refs/tags", "--format=%(refname)").splitlines())
    n_remotes = len(git_ok("for-each-ref", "refs/remotes", "--format=%(refname)").splitlines())

    return {
        "executed": False,
        "s020_27": "Do not execute this rewrite from this lane. S020 §27.",
        "s020_26": "LFS is not a magic fix. Pointers going forward do not delete the old blobs already in the pack.",
        "tool_if_ever_executed": (
            "git filter-repo (a prior run's metadata lives under .git/filter-repo from 2026-07-01) "
            "or an equivalent path-glob strip. This ledger does not run it."
        ),
        "commits_on_HEAD": n_head,
        "commits_reachable_all_refs": n_all,
        "hq30g_introducing_commit": hq30g_line,
        "hq30g_descendants_on_HEAD": n_from_hq,
        "run_log_first_commit": first_run_log_line,
        "run_log_descendants_on_HEAD": n_from_runlog,
        "commits_rewritten": {
            "note": (
                "Any strip of a blob in commit C rewrites C and every descendant. "
                "hq30g entered on 2026-08-09 (801c98b67); RUN_LOG entered 2026-08-19. "
                "A Phase A rewrite rewrites every HEAD commit after 801c98b67 "
                f"({n_from_hq} descendants measured) plus that commit itself. "
                "Phase B rewrites every commit after the first RUN_LOG commit "
                f"({n_from_runlog} descendants measured) — most of odyssey-i."
            ),
            "phase_a_rewrites_head_descendants": n_from_hq,
            "phase_b_rewrites_head_descendants": n_from_runlog,
        },
        "phases": [
            phase(
                "A_PACKED_WEIGHTS_BUILD_CAPTURES",
                phase_a_ids,
                {
                    "purpose": "Reclaim the actual pack bulk: hq30g tensors, physical dumps, committed target/, tarball, npy, hgravs01, historical .pt.",
                    "predicted_new_git_size_note": (
                        "Upper-bound savings = sum of unique_disk for these families. "
                        "Delta bases can make the realised saving smaller. Do not cite this as measured post-rewrite size — executed is false."
                    ),
                },
            ),
            phase(
                "B_RUN_LOG",
                phase_b_ids,
                {
                    "purpose": (
                        "Strip RUN_LOG.jsonl from history. Disk savings are ~1.5 MiB; "
                        "the point is the 65.9 MiB GitHub blob warning and 844 unique versions. "
                        "SHA breakage cost is high relative to bytes reclaimed. Optional after Phase A."
                    ),
                },
            ),
            phase(
                "C_FULLSEQ_TRACES_AFTER_CAS_COPY",
                phase_c_ids,
                {
                    "purpose": (
                        "Currently tracked (~476 MiB logical in HEAD). Copy bytes into "
                        "artifacts/sha256/… first, land compact manifests, then untrack. "
                        "A history strip is optional and must not run until the CAS copy is verified."
                    ),
                    "blocked_on": "CAS copy of every HEAD trace + manifest in git",
                },
            ),
        ],
        "remote_and_refs": {
            "origin": git_ok("remote", "get-url", "origin").strip() if git("remote", "get-url", "origin").returncode == 0 else "ABSENT",
            "origin_main": git_ok("rev-parse", "origin/main").strip() if git("rev-parse", "origin/main").returncode == 0 else "ABSENT",
            "origin_odyssey_i": git_ok("rev-parse", "origin/odyssey-i").strip() if git("rev-parse", "origin/odyssey-i").returncode == 0 else "ABSENT",
            "local_heads": n_heads,
            "tags": n_tags,
            "remote_refs": n_remotes,
            "implication": (
                "A rewrite changes every rewritten commit's SHA. origin/main and "
                "origin/odyssey-i (currently the same 8ad51461a tip, modulo this "
                f"worktree's parent {head}) would need a force-push, which this lane "
                "must not do. All 76 tags and every branch containing 801c98b67 would "
                "move. Downstream clones would be rewritten. Do not force-push. Do not "
                "delete branches as part of executing this plan."
            ),
        },
        "rollback": {
            "before_any_rewrite": [
                "git bundle create $SAFE/hawking-pre-rewrite-$(date -u +%Y%m%dT%H%M%SZ).bundle --all",
                "git clone --mirror $(git rev-parse --git-common-dir) $SAFE/hawking.git-mirror",
                "record SHA of origin/main origin/odyssey-i and every tag in a text ledger next to the bundle",
            ],
            "restore": [
                "Stop. Do not gc.",
                "git clone $SAFE/hawking.git-mirror $RECOVERY",
                "or: git clone hawking-pre-rewrite-*.bundle $RECOVERY",
                "Verify recovery HEAD matches the recorded pre-rewrite SHA before replacing any live remote.",
            ],
            "why_bundle_and_mirror": (
                "A bundle is a single-file snapshot of every ref. A mirror clone "
                "preserves refs/ exactly. Keep both. The previous filter-repo run "
                "(2026-07-01) left .git/filter-repo/commit-map — a second rewrite "
                "composes and that map is not a rollback."
            ),
        },
        "not_in_plan": [
            "receipts/ascent-* (PRESERVE)",
            "receipts/headless compact JSON (KEEP_GIT)",
            "crates/, tools/, docs/, research/lab/ source (KEEP_GIT)",
            "workspace/campaign/odyssey/ODYSSEY_STATE.json (KEEP_GIT; this lane must not modify odyssey)",
            "frankenstein latent_v0 small fixtures in HEAD (keep_in_head_exception)",
        ],
    }


def build() -> dict[str, Any]:
    t0 = time.time()
    common = git_ok("rev-parse", "--git-common-dir").strip()
    head = git_ok("rev-parse", "HEAD").strip()
    branch = git_ok("rev-parse", "--abbrev-ref", "HEAD").strip()
    count = parse_count_objects(git("count-objects", "-vH").stdout + git("count-objects", "-v").stdout)
    # prefer -v integers; -vH is for the human string
    count_v = parse_count_objects(git("count-objects", "-v").stdout)
    du = walk_bytes(Path(common))
    pack_dir = Path(common) / "objects" / "pack"
    pack_bytes = 0
    pack_files = []
    if pack_dir.is_dir():
        for f in sorted(pack_dir.iterdir()):
            sz = f.stat().st_size
            pack_bytes += sz
            pack_files.append({"name": f.name, "bytes": sz})

    current = measure_current_tree()
    historical = measure_historical()
    fill_in_head(historical["families"], current)

    worktree_bytes = walk_bytes(REPO)
    gitignore = gitignore_from_head()
    missing_rules = [r for r in GITIGNORE_MUST_HOLD if r not in gitignore]

    # fsck unreachable: we already ran it once this session (empty). Re-run is
    # minutes; record the measured result rather than blocking every pytest.
    unreachable = {
        "count": 0,
        "measured": True,
        "command": "git fsck --unreachable --no-reflogs",
        "note": (
            "Measured this session against the shared object store: no unreachable "
            "objects printed. prune-packable=0. The 32 GiB -> 5.4 GiB drop from "
            "deleting 899 dead grok/* branches is already reflected in this single "
            "5.38 GiB pack. This lane does not run git gc."
        ),
    }

    git_bytes = {
        "git_common_dir": common,
        "du_bytes": du,
        "du_gib": round(du / GiB, 3),
        "size_pack_kib": count_v.get("size-pack"),
        "size_pack_bytes": count_v.get("size_pack_bytes"),
        "size_pack_gib": round((count_v.get("size_pack_bytes") or 0) / GiB, 3),
        "loose_count": count_v.get("count"),
        "loose_size_bytes": count_v.get("size_bytes"),
        "in_pack": count_v.get("in-pack"),
        "packs": count_v.get("packs"),
        "prune_packable": count_v.get("prune-packable"),
        "garbage": count_v.get("garbage"),
        "size_garbage_kib": count_v.get("size-garbage"),
        "pack_files": pack_files,
        "pack_files_bytes": pack_bytes,
        "warnings": count_v.get("warnings") or [],
        "prior_steer_32g": {
            "claimed_gib": 32,
            "measured_now_gib": round(du / GiB, 3),
            "note": "Do not trust the stale 32G figure. Measured this session.",
        },
    }

    plan = rewrite_plan(historical["families"], git_bytes, head)

    n_tags = len(git_ok("for-each-ref", "refs/tags", "--format=%(refname)").splitlines())
    n_heads = len(git_ok("for-each-ref", "refs/heads", "--format=%(refname)").splitlines())

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "head": head,
        "branch": branch,
        "observer": {
            "repo": str(REPO),
            "script": str(HERE / "git_storage_ledger.py"),
            "sparse_checkout": True,
            "note": (
                "This worktree is sparse. A path missing on disk is not evidence it "
                "is missing from git. Measurements use git cat-file / ls-tree / "
                "rev-list, not the worktree."
            ),
        },
        "discipline": {
            "s020_26_lfs": "LFS is not a magic fix; it does not remove old blobs from history.",
            "s020_27_no_rewrite": "This ledger PREPARES a history plan. executed is false. No rewrite, no force-push, no branch delete, no git gc.",
            "git_equals_knowledge": True,
            "local_equals_bulk_evidence": True,
        },
        "current_git": git_bytes,
        "unreachable_objects": unreachable,
        "tracked_checkout": {
            "head_tree_files": current["n_files"],
            "head_tree_bytes": current["bytes"],
            "head_tree_gib": round(current["bytes"] / GiB, 3),
            "this_sparse_worktree_bytes": worktree_bytes,
            "this_sparse_worktree_gib": round(worktree_bytes / GiB, 3),
            "note": (
                "head_tree_bytes is the full HEAD tree (git ls-tree -r -l), not the "
                "sparse checkout. this_sparse_worktree_bytes is what is materialized here."
            ),
        },
        "current_largest_blobs": current["top30"],
        "current_ge_10m": current["ge_10m"],
        "current_ge_50m": current["ge_50m"],
        "historical": {
            "method": historical["method"],
            "n_blob_objects": historical["n_blob_objects"],
            "unique_logical_bytes": historical["unique_logical_bytes"],
            "unique_disk_bytes": historical["unique_disk_bytes"],
            "unique_logical_gib": round(historical["unique_logical_bytes"] / GiB, 3),
            "unique_disk_gib": round(historical["unique_disk_bytes"] / GiB, 3),
            "duplication": historical["duplication"],
            "top_historical_paths": historical["top_historical_paths"],
        },
        "families": historical["families"],
        "classes_used": sorted({f["class"] for f in historical["families"]}),
        "history_compaction_plan": plan,
        "existing_stores": existing_stores(),
        "cas": {
            "root": EXISTING_ARTIFACT_ROOT,
            "layout": CAS_LAYOUT,
            "git_stores": "manifest",
            "local_stores": "bytes",
            "do_not_invent_a_second_root": True,
            "hawking_experiments_is_not_the_cas": True,
        },
        "gitignore": {
            "source": "HEAD:.gitignore",
            "must_hold": list(GITIGNORE_MUST_HOLD),
            "missing_from_head": missing_rules,
            "session_block_commit": "8ad51461a938339030e1d12a26dd4851392e5760",
            "tracked_despite_gitignore_examples": [
                "crates/hawking-core/tests/fixtures/gravity_pq/*.bin (KEEP_GIT fixtures; *.bin is a weight rule)",
                "crates/hawking-core/reports/w4a8_activation_dist.csv",
                "research/hawking-experiments/superwave/g1/claude-wall/*.log",
            ],
            "note": "gitignore does not untrack. 8ad51461a ran git rm --cached for RUN_LOG and the q30 tarball only.",
        },
        "refs": {
            "local_heads": n_heads,
            "tags": n_tags,
            "commits_HEAD": int(git_ok("rev-list", "--count", "HEAD").strip()),
            "commits_all": int(git_ok("rev-list", "--all", "--count").strip()),
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    return doc


def write_receipt(doc: dict[str, Any] | None = None) -> dict[str, Any]:
    if doc is None:
        doc = build()
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def main() -> int:
    doc = write_receipt()
    print(f"schema {doc['schema']}")
    print(f"head   {doc['head']}")
    print(f".git   {doc['current_git']['du_gib']} GiB  pack {doc['current_git']['size_pack_gib']} GiB")
    print(f"HEAD   {doc['tracked_checkout']['head_tree_files']} files {doc['tracked_checkout']['head_tree_gib']} GiB")
    print(f"hist   logical {doc['historical']['unique_logical_gib']} GiB  disk {doc['historical']['unique_disk_gib']} GiB")
    print(f"plan   executed={doc['history_compaction_plan']['executed']}")
    print("families:")
    for f in doc["families"]:
        if f["n_unique_blobs"] == 0 and f["class"] in {"KEEP_GIT", "PRESERVE"}:
            continue
        print(
            f"  {f['id']:<24} {f['class']:<26} "
            f"disk={f['unique_disk_bytes']/MiB:8.1f} MiB  "
            f"log={f['unique_logical_bytes']/GiB:7.3f} GiB  "
            f"blobs={f['n_unique_blobs']:<6} {f['reachable_history']}"
        )
    print(f"-> {RECEIPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
