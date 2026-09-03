#!/usr/bin/env python3
"""DIRTY_TREE_PRESERVATION — prove the pre-existing dirty tree is still on disk.

`~/Downloads/hawking-copy` carried hundreds of modified-but-uncommitted paths
BEFORE this campaign began: Metal shaders, Rust model code, campaign state,
receipts. That work is unbacked-up. Losing it is unrecoverable.

This script is an observer. It answers, from live disk state:

  1. What currently differs from HEAD, plus every untracked path.
  2. Which of those is this campaign's own work, and which is pre-existing.
  3. Whether any previously-dirty path has been reverted to match HEAD
     (that is the loss this gate exists to catch).
  4. Whether any tracked dirty file is shorter than its HEAD blob.
  5. What `git fsck --no-reflogs --lost-found` finds in the object store.
  6. Every live worktree and every grok/* branch, naming any worktree that
     holds uncommitted work (work sitting outside every branch).

WRITE SCOPE
-----------
Writes exactly one file, next to this script's repository root:

    receipts/headless/DIRTY_TREE_PRESERVATION.json

It never writes into the preservation target (the main worktree). It never
runs git add / checkout / restore / stash / clean / reset against any live
Hawking worktree. The revert demonstration happens in a throwaway scratch
clone under tempfile, then that clone is deleted.

SPARSE CHECKOUT
---------------
This observer may itself be a sparse checkout. A file missing HERE is not
evidence it does not exist. The preservation target is the MAIN worktree of
the shared git directory (hawking-copy), which holds the dirty science.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


GENERATED_DIR_NAMES = ("__pycache__", ".pytest_cache", "node_modules", "target")
GENERATED_WHY = (
    "These directories are reproducible build/test artifacts (Python bytecode, "
    "pytest cache, JS packages, Rust incremental output). They are not scientific "
    "work. gitignore already drops them from `git status`; we additionally skip "
    "any path that still leaks through with one of those names as a component."
)

# Campaign-produced work, identifiable by path. Everything else is presumed
# pre-existing and precious.
CAMPAIGN_PREFIXES = (
    "tools/headless/",
    "hcli/",
    "hcli/",
    "receipts/headless/",
    ".hcli/",
)

DISK_TRUTH_REL = Path("receipts/headless/DISK_TRUTH.json")
FALLBACK_MAIN = Path(os.path.expanduser("~/Downloads/hawking-copy"))

# `git fsck --lost-found` writes under .git/lost-found, which is the object
# store, not the working tree. Still: never run it against a scratch we don't
# own for the live repo's working tree, and never use --lost-found as a repair.
FSCK_TIMEOUT_SEC = 600


def git(repo: Path, *args: str, timeout: int | None = 120,
        check: bool = False, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
        input=input_text,
    )


def git_out(repo: Path, *args: str, timeout: int | None = 120) -> str:
    # Porcelain lines may begin with a space (XY = ' M'). Never .strip() the
    # whole stdout — that ate the leading space on `.gitignore` and made the
    # first dirty path look missing. Trailing newlines only.
    return git(repo, *args, timeout=timeout).stdout.rstrip("\n")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def posix(path: str) -> str:
    return path.replace("\\", "/")


def is_generated(path: str) -> bool:
    return any(part in GENERATED_DIR_NAMES for part in Path(posix(path)).parts)


def classify(path: str) -> str:
    p = posix(path)
    for prefix in CAMPAIGN_PREFIXES:
        stem = prefix.rstrip("/")
        if p == stem or p.startswith(prefix):
            return "campaign"
    return "pre-existing"


def observer_root() -> Path:
    # tools/headless/this.py -> repo root of THIS worktree (may be sparse).
    return Path(__file__).resolve().parent.parent.parent


def discover_main_worktree(start: Path) -> Path:
    """The main worktree is the one whose .git is a directory (not a gitfile)."""
    common = git_out(start, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if not common:
        common = git_out(start, "rev-parse", "--git-common-dir")
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = (start / common_path).resolve()
    # Conventional: <main>/.git is the common dir.
    if common_path.name == ".git" and common_path.is_dir():
        candidate = common_path.parent
        if (candidate / ".git").is_dir():
            return candidate
    parsed = parse_worktree_list(start)
    for wt in parsed:
        git_meta = Path(wt["path"]) / ".git"
        if git_meta.is_dir():
            return Path(wt["path"])
    if FALLBACK_MAIN.is_dir():
        return FALLBACK_MAIN
    return start


def parse_worktree_list(repo: Path) -> list[dict[str, Any]]:
    raw = git_out(repo, "worktree", "list", "--porcelain")
    rows: list[dict[str, Any]] = []
    cur: dict[str, Any] = {}
    for line in (raw.splitlines() + [""]):
        if not line.strip():
            if cur:
                rows.append(cur)
                cur = {}
            continue
        if line.startswith("worktree "):
            cur = {"path": line[len("worktree "):], "bare": False, "detached": False,
                   "prunable": None, "head": None, "branch": None}
        elif line.startswith("HEAD "):
            cur["head"] = line[len("HEAD "):]
        elif line.startswith("branch "):
            ref = line[len("branch "):]
            cur["branch"] = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref
        elif line == "bare":
            cur["bare"] = True
        elif line == "detached":
            cur["detached"] = True
        elif line.startswith("prunable"):
            cur["prunable"] = line
    return rows


def parse_porcelain(text: str) -> list[dict[str, str]]:
    """Parse `git status --porcelain=v1` lines into {xy, path, orig_path}."""
    rows = []
    for line in text.splitlines():
        if not line:
            continue
        xy = line[:2]
        rest = line[3:] if len(line) > 3 else ""
        orig = None
        path = rest
        if " -> " in rest and xy.strip() and xy[0] in "RCU":
            orig, path = rest.split(" -> ", 1)
        # git may wrap paths in quotes
        if len(path) >= 2 and path[0] == '"' and path[-1] == '"':
            path = bytes(path[1:-1], "utf-8").decode("unicode_escape")
        rows.append({"xy": xy, "path": path, "orig_path": orig or ""})
    return rows


def count_xy(rows: list[dict[str, str]]) -> dict[str, int]:
    """Split porcelain into modified / untracked / staged, matching git's XY.

    modified  = index or worktree differs from HEAD for a tracked path
                (any porcelain line that is not '??' and not '!!')
    untracked = '??'
    staged    = first column in AMDRC (index differs from HEAD)
    """
    modified = untracked = staged = ignored = 0
    xy_hist: Counter[str] = Counter()
    for r in rows:
        xy = r["xy"]
        xy_hist[xy] += 1
        x, y = xy[0], xy[1]
        if xy == "??":
            untracked += 1
        elif xy == "!!":
            ignored += 1
        else:
            modified += 1
            if x in "AMDRCU":
                staged += 1
    return {
        "modified": modified,
        "untracked": untracked,
        "staged": staged,
        "ignored": ignored,
        "xy": dict(xy_hist),
    }


def head_blob(repo: Path, path: str) -> tuple[str | None, int | None]:
    """Return (sha1, size) of HEAD:path, or (None, None) if it does not exist."""
    r = git(repo, "cat-file", "--batch-check=%(objectname) %(objectsize) %(objecttype)",
            input_text=f"HEAD:{path}\n")
    line = (r.stdout or "").strip()
    if not line or line.endswith(" missing") or " missing" in line:
        return None, None
    parts = line.split()
    if len(parts) < 2:
        return None, None
    try:
        return parts[0], int(parts[1])
    except ValueError:
        return None, None


def worktree_sha1(repo: Path, path: str) -> str | None:
    r = git(repo, "hash-object", "--", path)
    if r.returncode != 0:
        return None
    return (r.stdout or "").strip() or None


def sha256_file(path: Path, limit: int = 64 * 1024 * 1024) -> str | None:
    """sha256 of a file, refusing to slurp anything over `limit` bytes."""
    try:
        st = path.stat()
    except OSError:
        return None
    if st.st_size > limit:
        return f"omitted:file_too_large:{st.st_size}"
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def disk_size(root: Path, path: str) -> int | None:
    p = root / path
    try:
        if p.is_file() or p.is_symlink():
            return p.stat().st_size
    except OSError:
        return None
    return None


def reverted_against_baseline(repo: Path, baseline_paths: list[str]) -> list[dict[str, Any]]:
    """A dirty path that NO LONGER differs from HEAD has been reverted.

    That is exactly the loss this gate exists to catch. Untracked baseline
    paths that have vanished from disk are also loss. Untracked paths that
    still exist are preserved (there is no HEAD blob to revert *to*).
    """
    lost = []
    for path in baseline_paths:
        wt = repo / path
        sha_head, _head_sz = head_blob(repo, path)
        exists = wt.exists()
        if sha_head is None:
            # Untracked at baseline. Loss = it is gone from disk.
            if not exists:
                lost.append({
                    "path": path,
                    "reason": "untracked_path_vanished_from_disk",
                    "head_sha1": None,
                })
            continue
        if not exists:
            lost.append({
                "path": path,
                "reason": "tracked_path_missing_on_disk",
                "head_sha1": sha_head,
            })
            continue
        sha_wt = worktree_sha1(repo, path)
        if sha_wt is not None and sha_wt == sha_head:
            lost.append({
                "path": path,
                "reason": "matches_HEAD_again",
                "head_sha1": sha_head,
                "worktree_sha1": sha_wt,
            })
    return lost


def demonstrate_detector_can_fire() -> dict[str, Any]:
    """Construct a throwaway clone, dirty a file, revert it, watch the gate fire.

    An all-green report from a gate never seen to fail is not evidence.
    This is the evidence. Nothing in here touches a live Hawking worktree.
    """
    tmp = Path(tempfile.mkdtemp(prefix="dirty-tree-preservation-scratch-"))
    env_git = [
        "git", "-c", "user.name=dirty-tree-preservation",
        "-c", "user.email=probe@invalid",
        "-c", "commit.gpgsign=false",
        "-c", "init.defaultBranch=main",
    ]
    log: list[str] = []

    def g(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        r = subprocess.run(
            [*env_git, *args],
            cwd=str(tmp),
            capture_output=True,
            text=True,
            check=False,
        )
        if check and r.returncode != 0:
            raise RuntimeError(f"scratch git {' '.join(args)} failed: {r.stderr}")
        return r

    try:
        g("init")
        precious = tmp / "metal_shader.metal"
        precious.write_text("// science v1: do not lose this\nkernel void k() {}\n")
        other = tmp / "notes.txt"
        other.write_text("campaign state v1\n")
        g("add", "metal_shader.metal", "notes.txt")
        g("commit", "-m", "baseline HEAD")
        head = g("rev-parse", "HEAD").stdout.strip()

        # Uncommitted science, the thing that must not be reverted.
        precious.write_text("// science v2: uncommitted, unbacked-up\nkernel void k() { /* real work */ }\n")
        other.write_text("campaign state v2 — also uncommitted\n")
        # Truncation case: a third file, dirty then shortened below HEAD.
        truncated = tmp / "ledger.json"
        truncated.write_text('{"n": 1, "payload": "' + ("x" * 200) + '"}\n')
        g("add", "ledger.json")
        g("commit", "-m", "ledger at full size")
        truncated.write_text('{"n": 1}\n')  # shorter than HEAD, still dirty

        status_before = g("status", "--porcelain=v1", "-uall").stdout
        dirty_before = [r["path"] for r in parse_porcelain(status_before)]
        log.append(f"scratch HEAD={head[:12]}")
        log.append(f"dirty before revert: {dirty_before}")

        trunc_hits_before = []
        for p in dirty_before:
            sha_h, head_sz = head_blob(tmp, p)
            dsz = disk_size(tmp, p)
            if sha_h is not None and dsz is not None and head_sz is not None and dsz < head_sz:
                trunc_hits_before.append({
                    "path": p, "disk_bytes": dsz, "head_bytes": head_sz,
                })
        log.append(f"truncation hits before revert: {trunc_hits_before}")

        # THE LOSS. `git checkout --` puts the file back to HEAD and destroys
        # the uncommitted science. This is the action the live repo must never
        # suffer. We do it only inside this scratch clone.
        g("checkout", "--", "metal_shader.metal")
        status_after = g("status", "--porcelain=v1", "-uall").stdout
        dirty_after = [r["path"] for r in parse_porcelain(status_after)]
        lost = reverted_against_baseline(tmp, dirty_before)
        fired = any(x["path"] == "metal_shader.metal" for x in lost)
        trunc_still = []
        for p in dirty_after:
            sha_h, head_sz = head_blob(tmp, p)
            dsz = disk_size(tmp, p)
            if sha_h is not None and dsz is not None and head_sz is not None and dsz < head_sz:
                trunc_still.append({
                    "path": p, "disk_bytes": dsz, "head_bytes": head_sz,
                })

        # notes.txt must still be dirty — the detector is path-specific, not
        # a blanket "something changed".
        notes_preserved = "notes.txt" in dirty_after
        ledger_trunc_fired = any(x["path"] == "ledger.json" for x in trunc_still)

        return {
            "scratch_clone": str(tmp),
            "scratch_head": head,
            "action": "git checkout -- metal_shader.metal   # THE LOSS this gate exists to catch",
            "dirty_before": dirty_before,
            "dirty_after": dirty_after,
            "reverted_detected": lost,
            "revert_detector_fired": fired,
            "unrelated_dirty_file_still_present": notes_preserved,
            "truncation_before_revert": trunc_hits_before,
            "truncation_after_revert": trunc_still,
            "truncation_detector_fired": ledger_trunc_fired,
            "log": log,
            "proof": (
                "A file that was in the dirty snapshot and now hashes equal to "
                "HEAD is reported by name. metal_shader.metal was deliberately "
                "reverted; the detector named it. notes.txt was left dirty and "
                "was not named. ledger.json was shortened below its HEAD blob "
                "and the truncation detector named it with both sizes."
            ),
        }
    except Exception as e:
        return {
            "scratch_clone": str(tmp),
            "revert_detector_fired": False,
            "truncation_detector_fired": False,
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def inspect_entry(repo: Path, row: dict[str, str]) -> dict[str, Any]:
    path = row["path"]
    xy = row["xy"]
    rec: dict[str, Any] = {
        "path": path,
        "xy": xy,
        "orig_path": row.get("orig_path") or None,
        "classification": classify(path),
        "untracked": xy == "??",
        "staged": bool(xy) and xy[0] in "AMDRCU",
        "disk_bytes": disk_size(repo, path),
        "present_on_disk": (repo / path).exists(),
    }
    if xy == "??":
        rec["head_sha1"] = None
        rec["head_bytes"] = None
        rec["worktree_sha1"] = None
        rec["still_differs_from_head"] = True  # no HEAD blob; presence is preservation
        rec["truncated_vs_head"] = False
        return rec

    sha_head, head_sz = head_blob(repo, path)
    rec["head_sha1"] = sha_head
    rec["head_bytes"] = head_sz
    if rec["present_on_disk"] and (repo / path).is_file():
        rec["worktree_sha1"] = worktree_sha1(repo, path)
        rec["worktree_sha256"] = sha256_file(repo / path)
    else:
        rec["worktree_sha1"] = None
        rec["worktree_sha256"] = None

    if sha_head is None:
        # staged-add / never in HEAD
        rec["still_differs_from_head"] = rec["present_on_disk"]
        rec["truncated_vs_head"] = False
        return rec

    rec["still_differs_from_head"] = (
        rec["worktree_sha1"] is not None and rec["worktree_sha1"] != sha_head
    ) or (not rec["present_on_disk"])
    rec["truncated_vs_head"] = (
        rec["disk_bytes"] is not None and head_sz is not None
        and rec["disk_bytes"] < head_sz
    )
    rec["size_delta"] = (
        None if rec["disk_bytes"] is None or head_sz is None
        else rec["disk_bytes"] - head_sz
    )
    return rec


def prefix_histogram(paths: list[str]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for path in paths:
        cl = classify(path)
        if cl == "campaign":
            for pref in CAMPAIGN_PREFIXES:
                if path == pref.rstrip("/") or path.startswith(pref):
                    c[pref] += 1
                    break
            continue
        parts = posix(path).split("/")
        if path.startswith("workspace/"):
            key = "/".join(parts[:3]) + "/"
        elif path.startswith("crates/") or path.startswith("receipts/") or path.startswith("tools/"):
            key = "/".join(parts[:2]) + "/"
        else:
            key = parts[0] + ("/" if len(parts) > 1 else "")
        c[key] += 1
    return dict(c.most_common())


def load_disk_truth(main: Path) -> dict[str, Any] | None:
    p = main / DISK_TRUTH_REL
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def fsck_object_store(main: Path) -> dict[str, Any]:
    """Run the requested fsck. Summarise dangling commits/blobs; name commits.

    `--lost-found` copies dangling objects into .git/lost-found/. That is the
    git directory, not the working tree. We record whether the directory was
    created and how many files landed there.
    """
    lost_found = Path(git_out(main, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    if not lost_found.is_absolute():
        lost_found = (main / git_out(main, "rev-parse", "--git-common-dir")).resolve()
    lf_dir = lost_found / "lost-found"
    existed_before = lf_dir.exists()

    r = git(main, "fsck", "--no-reflogs", "--lost-found", timeout=FSCK_TIMEOUT_SEC)
    text = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
    dangling_commits: list[str] = []
    dangling_blobs: list[str] = []
    dangling_trees: list[str] = []
    other: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("dangling commit "):
            dangling_commits.append(s.split()[-1])
        elif s.startswith("dangling blob "):
            dangling_blobs.append(s.split()[-1])
        elif s.startswith("dangling tree "):
            dangling_trees.append(s.split()[-1])
        elif s:
            other.append(s)

    named = []
    for sha in dangling_commits[:50]:
        show = git(main, "log", "-1", "--format=%H%x09%ci%x09%s", sha)
        if show.returncode == 0 and show.stdout.strip():
            parts = show.stdout.strip().split("\t", 2)
            named.append({
                "sha": parts[0],
                "date": parts[1] if len(parts) > 1 else None,
                "subject": parts[2] if len(parts) > 2 else None,
            })
        else:
            named.append({"sha": sha, "date": None, "subject": None,
                          "note": (show.stderr or "").strip()[:200] or "no log"})

    lf_files = []
    if lf_dir.exists():
        for p in lf_dir.rglob("*"):
            if p.is_file():
                lf_files.append(str(p.relative_to(lf_dir)))

    return {
        "command": "git fsck --no-reflogs --lost-found",
        "returncode": r.returncode,
        "dangling_commit_count": len(dangling_commits),
        "dangling_blob_count": len(dangling_blobs),
        "dangling_tree_count": len(dangling_trees),
        "dangling_commits": dangling_commits,
        "dangling_commits_named": named,
        "other_lines": other[:100],
        "other_line_count": len(other),
        "lost_found": {
            "path": str(lf_dir),
            "existed_before": existed_before,
            "file_count": len(lf_files),
            "files_sample": lf_files[:30],
            "note": (
                "lost-found lives under the git directory, not the working tree. "
                "Dangling objects are not loss by themselves; a dangling commit "
                "not reachable from any branch is named above."
            ),
        },
        "stderr_tail": (r.stderr or "")[-1500:],
    }


def worktree_uncommitted(wt_path: Path) -> dict[str, Any]:
    st = git(wt_path, "status", "--porcelain=v1")
    rows = parse_porcelain(st.stdout or "")
    sparse_list = git(wt_path, "sparse-checkout", "list")
    sparse_lines = [ln for ln in (sparse_list.stdout or "").splitlines() if ln.strip()]
    git_meta = wt_path / ".git"
    return {
        "path": str(wt_path),
        "porcelain_count": len(rows),
        "has_uncommitted_work": len(rows) > 0,
        "uncommitted_not_on_any_branch": len(rows) > 0,
        "paths": [f"{r['xy']} {r['path']}" for r in rows],
        "git_is_dir": git_meta.is_dir(),
        "sparse_checkout_entries": len(sparse_lines),
        "sparse_checkout_sample": sparse_lines[:20],
        "head": git_out(wt_path, "rev-parse", "HEAD") or None,
        "branch": git_out(wt_path, "rev-parse", "--abbrev-ref", "HEAD") or None,
    }


def grok_branches(repo: Path, worktrees: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = git_out(repo, "for-each-ref",
                  "--format=%(refname:short)%09%(objectname)%09%(committerdate:iso-strict)%09%(contents:subject)",
                  "refs/heads/grok")
    wt_by_branch = {w.get("branch"): w for w in worktrees if w.get("branch")}
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 3)
        name = parts[0]
        sha = parts[1] if len(parts) > 1 else ""
        date = parts[2] if len(parts) > 2 else ""
        subject = parts[3] if len(parts) > 3 else ""
        wt = wt_by_branch.get(name)
        rows.append({
            "branch": name,
            "sha": sha,
            "date": date,
            "subject": subject,
            "has_worktree": bool(wt),
            "worktree": (wt or {}).get("path"),
            "worktree_has_uncommitted_work": bool((wt or {}).get("has_uncommitted_work")),
        })
    return rows


def skip_worktree_bits(repo: Path) -> dict[str, Any]:
    """Files marked skip-worktree / assume-unchanged can hide dirt from status."""
    raw = git_out(repo, "ls-files", "-v")
    special = []
    for line in raw.splitlines():
        if not line or line[0] == "H":
            continue
        tag = line[0]
        # H = cached, S = skip-worktree, s = assume+skip, M = unmerged,
        # ? = others, r = deleted, C = modification or add, etc.
        if tag in "Ssa":
            special.append({"tag": tag, "path": line[2:] if len(line) > 2 else line[1:].lstrip()})
    return {
        "count": len(special),
        "entries": special,
        "meaning": (
            "S/s/a in `git ls-files -v` hide working-tree differences from "
            "`git status`. A skip-worktree bit on a precious file would let a "
            "revert (or a sparse hole) look like a clean tree. Zero is the "
            "expected, good answer."
        ),
    }


def inspect_ignored_generated(repo: Path) -> dict[str, Any]:
    raw = git_out(repo, "status", "--porcelain=v1", "--ignored", "--untracked-files=normal")
    generated_hits = []
    ignored_top = []
    for line in raw.splitlines():
        if not line.startswith("!!"):
            continue
        path = line[3:]
        ignored_top.append(path)
        if is_generated(path):
            generated_hits.append(path)
    return {
        "ignored_collapsed_count": len(ignored_top),
        "generated_ignored_dirs": generated_hits,
        "generated_ignored_count": len(generated_hits),
        "ignored_top_sample": ignored_top[:40],
        "excluded_from_this_inventory": generated_hits,
        "why": GENERATED_WHY,
        "note": (
            "DISK_TRUTH counted `git status --porcelain` (non-ignored). This "
            "gate uses the same grain for the comparable count, plus `-uall` "
            "for the full untracked-file inventory. gitignored trees such as "
            ".hcli-legacy/, .claude/, *.log, weights/ are therefore outside the "
            "373-style dirty count — they were outside it at DISK_TRUTH time "
            "too. Generated dirs listed here are the ones we skip even if "
            "they leak through gitignore."
        ),
    }


def print_report(doc: dict[str, Any]) -> None:
    t = doc["preservation_target"]
    c = doc["counts"]
    print("=== DIRTY TREE PRESERVATION ===")
    print(f"observer worktree     {doc['observer']['path']}")
    print(f"observer sparse       {doc['observer']['sparse']}")
    print(f"preservation target   {t['path']}")
    print(f"HEAD                  {t['head']}")
    print(f"branch                {t['branch']}")
    print(f"target sparse         {t['sparse']}")
    print()
    print("## 1. ENUMERATE")
    print(f"porcelain (normal, comparable to DISK_TRUTH)   {c['porcelain_normal']}")
    print(f"porcelain (-uall, before exclusions)           {c['porcelain_uall_raw']}")
    print(f"porcelain (-uall, after generated exclusions)  {c['porcelain_uall_kept']}")
    print(f"modified (tracked, index or worktree vs HEAD)  {c['modified']}")
    print(f"untracked (??, after exclusions)               {c['untracked']}")
    print(f"staged (index vs HEAD)                         {c['staged']}")
    print(f"pre-existing (precious)                        {c['pre_existing']}")
    print(f"campaign (tools/headless|haider/hcli|receipts/headless|.hcli)  {c['campaign']}")
    print(f"xy histogram                                   {c['xy']}")
    print()
    print("excluded generated directories:")
    print(f"  names: {list(GENERATED_DIR_NAMES)}")
    print(f"  why:   {GENERATED_WHY}")
    print(f"  leaked-through-and-dropped: {c['excluded_generated_paths']}")
    ign = doc["ignored_generated"]
    print(f"  gitignored collapsed entries: {ign['ignored_collapsed_count']}")
    print(f"  of which generated dirs:      {ign['generated_ignored_count']} {ign['generated_ignored_dirs']}")
    print()
    print("untracked / dirty by prefix (kept inventory):")
    for k, v in doc["prefix_histogram"].items():
        print(f"  {v:5d}  {k}")
    print()
    print("## 2. CLASSIFY  (every kept path; full list also in the receipt)")
    print(f"{'CLASS':<14} {'XY':<4} {'still':<5} {'disk':>10} {'head':>10}  path")
    for rec in doc["inventory"]:
        still = "yes" if rec.get("still_differs_from_head") else "NO"
        disk = rec.get("disk_bytes")
        head = rec.get("head_bytes")
        d_s = "-" if disk is None else str(disk)
        h_s = "-" if head is None else str(head)
        print(f"{rec['classification']:<14} {rec['xy']:<4} {still:<5} {d_s:>10} {h_s:>10}  {rec['path']}")
    print()
    print("## 3. PROVE NOTHING WAS LOST  (reverted-file check)")
    rev = doc["reverted"]
    print(f"live snapshot tracked dirty paths                 {rev['live_tracked_dirty_count']}")
    print(f"  still on disk, sha1 != HEAD                     {rev['live_still_differs_count']}")
    print(f"  staged-add (no HEAD blob), still on disk        {rev['live_new_present_count']}")
    print(f"  missing on disk                                 {rev['live_missing_on_disk']}")
    print(f"  worktree sha1 == HEAD sha1 (REVERTED)           {rev['live_matches_head_count']}")
    if rev["live_reverted"]:
        print("REVERTED PATHS (LOSS):")
        for x in rev["live_reverted"]:
            print(f"  {x}")
    else:
        print("no live path from the dirty snapshot hashes equal to HEAD")
        print("the check that would have found one:")
        print("  for path in dirty_snapshot:")
        print("      if worktree_sha1(path) == sha1(HEAD:path):  # LOSS, name the path")
        print("  demonstrated on a scratch clone in ## WHAT I WATCHED FAIL")
    print(f"skip-worktree / assume-unchanged flags            {doc['skip_worktree']['count']}")
    if doc["skip_worktree"]["entries"]:
        for e in doc["skip_worktree"]["entries"]:
            print(f"  HIDDEN {e}")
    dt = doc.get("disk_truth_baseline") or {}
    print(f"DISK_TRUTH baseline dirty_entries                 {dt.get('dirty_entries')}  "
          f"(HEAD {str(dt.get('head') or '')[:12]} at {dt.get('generated_at')})")
    print("DISK_TRUTH stored a COUNT, not a path list, so a silent revert of a")
    print("path that was dirty then and matches HEAD now cannot be named from")
    print("that receipt. This receipt stores the full path list so the next")
    print("run of this gate CAN name a revert.")
    print(f"path-level baseline written this run              {c['porcelain_uall_kept']} paths")
    print()
    print("## 4. TRUNCATION  (present but shorter than HEAD; report, do not accuse)")
    trunc = doc["truncation"]
    print(f"tracked dirty files shorter than HEAD blob: {len(trunc)}")
    if trunc:
        for x in trunc:
            print(f"  {x['path']}  disk={x['disk_bytes']}  head={x['head_bytes']}  "
                  f"delta={x['disk_bytes']-x['head_bytes']}  class={x['classification']}")
    else:
        print("  none")
    print()
    print("## 5. GIT FSCK")
    fk = doc["fsck"]
    print(f"command              {fk['command']}")
    print(f"returncode           {fk['returncode']}")
    print(f"dangling commits     {fk['dangling_commit_count']}")
    print(f"dangling blobs       {fk['dangling_blob_count']}")
    print(f"dangling trees       {fk['dangling_tree_count']}")
    if fk["dangling_commits_named"]:
        print("dangling commits named:")
        for n in fk["dangling_commits_named"]:
            print(f"  {n.get('sha','')}  {n.get('date') or ''}  {n.get('subject') or n.get('note') or ''}")
    else:
        print("no dangling commits to name")
    print(f"lost-found files     {fk['lost_found']['file_count']}  (existed_before={fk['lost_found']['existed_before']})")
    print()
    print("## 6. WORKTREES AND grok/* BRANCHES")
    print("worktrees:")
    for w in doc["worktrees"]:
        flag = "UNCOMMITTED-NOT-ON-ANY-BRANCH" if w["has_uncommitted_work"] else "clean"
        print(f"  [{flag}]  {w['branch'] or '(detached)'}  {w['head'][:12] if w.get('head') else '?'}  {w['path']}")
        if w["has_uncommitted_work"]:
            show = w["paths"] if len(w["paths"]) <= 80 else w["paths"][:80] + [f"... ({len(w['paths'])} total)"]
            for p in show:
                print(f"      {p}")
    print("grok/* branches:")
    for b in doc["grok_branches"]:
        wt = "worktree+UNCOMMITTED" if b["worktree_has_uncommitted_work"] else (
            "worktree" if b["has_worktree"] else "no-worktree")
        print(f"  {b['sha'][:12]}  {wt:<22}  {b['branch']}  {b['subject']}")
    print()
    print("## WHAT I WATCHED FAIL")
    fail = doc["what_i_watched_fail"]
    print(f"scratch clone (deleted after): {fail.get('scratch_clone')}")
    if fail.get("error"):
        print(f"ERROR: {fail['error']}")
    print(f"action: {fail.get('action')}")
    print(f"dirty before: {fail.get('dirty_before')}")
    print(f"dirty after:  {fail.get('dirty_after')}")
    print(f"revert detector FIRED: {fail.get('revert_detector_fired')}")
    print(f"reverted_detected: {fail.get('reverted_detected')}")
    print(f"unrelated dirty file still present: {fail.get('unrelated_dirty_file_still_present')}")
    print(f"truncation detector FIRED: {fail.get('truncation_detector_fired')}")
    print(f"truncation after: {fail.get('truncation_after_revert')}")
    print(fail.get("proof") or "")
    print()
    print("## VERDICT")
    print(doc["verdict"])
    for line in doc["verdict_reasons"]:
        print(f"  - {line}")
    print()
    print(f"receipt: {doc['receipt_path']}")


def main() -> int:
    observer = observer_root()
    cwd = Path.cwd()
    # Prefer discovering from cwd (the invocation the contract specifies:
    # `python3 tools/headless/dirty_tree_preservation.py` from the repo root)
    # and fall back to the script location.
    start = cwd if (cwd / ".git").exists() else observer
    try:
        main_wt = discover_main_worktree(start)
    except Exception:
        main_wt = FALLBACK_MAIN if FALLBACK_MAIN.is_dir() else observer

    generated_at = now_utc()
    head = git_out(main_wt, "rev-parse", "HEAD")
    branch = git_out(main_wt, "rev-parse", "--abbrev-ref", "HEAD")
    commit_count = git_out(main_wt, "rev-list", "--count", "HEAD")
    observer_head = git_out(observer, "rev-parse", "HEAD")
    observer_branch = git_out(observer, "rev-parse", "--abbrev-ref", "HEAD")

    porcelain_normal = git_out(main_wt, "-c", "core.quotePath=false",
                               "status", "--porcelain=v1")
    porcelain_uall = git_out(main_wt, "-c", "core.quotePath=false",
                             "status", "--porcelain=v1", "-uall")
    rows_normal = parse_porcelain(porcelain_normal)
    rows_uall = parse_porcelain(porcelain_uall)

    excluded_generated = [r["path"] for r in rows_uall if is_generated(r["path"])]
    kept_rows = [r for r in rows_uall if not is_generated(r["path"])]

    # Snapshot the dirty tracked paths BEFORE the (expensive) per-file inspect
    # so the revert check has a stable baseline for this run.
    tracked_snapshot = [r["path"] for r in kept_rows if r["xy"] != "??"]
    untracked_snapshot = [r["path"] for r in kept_rows if r["xy"] == "??"]

    inventory = [inspect_entry(main_wt, r) for r in kept_rows]

    tracked_inv = [rec for rec in inventory if not rec["untracked"]]
    live_reverted = [rec for rec in tracked_inv
                     if rec.get("head_sha1")
                     and rec.get("present_on_disk")
                     and rec.get("worktree_sha1") == rec.get("head_sha1")]
    live_missing = [rec for rec in tracked_inv if not rec.get("present_on_disk")]
    live_still = [rec for rec in tracked_inv if rec.get("still_differs_from_head")]
    live_new_present = [rec for rec in tracked_inv
                        if rec.get("head_sha1") is None and rec.get("present_on_disk")]
    truncation = [rec for rec in inventory if rec.get("truncated_vs_head")]

    counts_kept = count_xy(kept_rows)
    pre_n = sum(1 for rec in inventory if rec["classification"] == "pre-existing")
    camp_n = sum(1 for rec in inventory if rec["classification"] == "campaign")

    disk_truth = load_disk_truth(main_wt)
    dt_baseline = None
    if disk_truth:
        pr = disk_truth.get("primary_repo") or {}
        dt_baseline = {
            "path": pr.get("path"),
            "head": pr.get("head"),
            "dirty_entries": pr.get("dirty_entries"),
            "generated_at": disk_truth.get("generated_at"),
            "note": (
                "Count only — no path list. Cannot name a silent revert against "
                "this baseline. This receipt IS a path-level baseline."
            ),
        }

    # The live revert check against THIS snapshot: a path in tracked_snapshot
    # whose worktree now equals HEAD. Independently re-run the function so the
    # receipt shows the same code path that fires in the scratch clone.
    live_lost = reverted_against_baseline(main_wt, tracked_snapshot)

    fail = demonstrate_detector_can_fire()

    wt_meta = parse_worktree_list(main_wt)
    worktrees = []
    for meta in wt_meta:
        info = worktree_uncommitted(Path(meta["path"]))
        info["listed_head"] = meta.get("head")
        info["listed_branch"] = meta.get("branch")
        info["detached"] = meta.get("detached")
        info["is_preservation_target"] = os.path.realpath(meta["path"]) == os.path.realpath(main_wt)
        worktrees.append(info)

    grok = grok_branches(main_wt, worktrees)
    fsck = fsck_object_store(main_wt)
    skip = skip_worktree_bits(main_wt)
    ignored = inspect_ignored_generated(main_wt)

    observer_sparse = bool(git_out(observer, "sparse-checkout", "list"))
    target_sparse = bool(git_out(main_wt, "sparse-checkout", "list"))

    dirty_worktrees = [w for w in worktrees if w["has_uncommitted_work"]]

    reasons = []
    verdict = "PRESERVED"
    if live_lost:
        verdict = "LOSS"
        reasons.append(f"{len(live_lost)} path(s) from the dirty snapshot now match HEAD or are missing")
    if live_missing:
        verdict = "LOSS"
        reasons.append(f"{len(live_missing)} tracked dirty path(s) missing on disk")
    if not fail.get("revert_detector_fired"):
        verdict = "INCONCLUSIVE"
        reasons.append("scratch-clone revert detector did NOT fire — the gate is unproven")
    if not fail.get("truncation_detector_fired"):
        # truncation proof failing does not mean live loss, but the gate is weaker
        reasons.append("scratch-clone truncation detector did not fire (gate weaker, not live loss)")
    if skip["count"]:
        reasons.append(f"{skip['count']} skip-worktree/assume-unchanged flags present — status may hide dirt")
    if not live_lost and not live_missing:
        reasons.append(
            f"all {len(tracked_snapshot)} tracked dirty paths in the snapshot still differ from HEAD "
            f"(or are staged-adds with no HEAD blob and still present)"
        )
        reasons.append(
            f"{pre_n} pre-existing + {camp_n} campaign paths remain in the dirty/untracked inventory"
        )
        reasons.append(
            "DISK_TRUTH had no path list, so historical silent reverts between "
            f"{dt_baseline.get('generated_at') if dt_baseline else '?'} and now cannot be named; "
            "this receipt closes that hole"
        )
    if dirty_worktrees:
        reasons.append(
            f"{len(dirty_worktrees)} live worktree(s) hold uncommitted work not on any branch: "
            + ", ".join(w["path"] for w in dirty_worktrees)
        )

    receipt_path = observer / "receipts" / "headless" / "DIRTY_TREE_PRESERVATION.json"
    # Never write the receipt into the preservation target. If this script is
    # somehow running from hawking-copy itself, refuse rather than add a file
    # to the precious tree.
    if os.path.realpath(receipt_path.parent.parent.parent) == os.path.realpath(main_wt) \
            and os.path.realpath(observer) == os.path.realpath(main_wt):
        raise SystemExit(
            "refusing to write DIRTY_TREE_PRESERVATION.json into the preservation "
            "target; run from the observer worktree"
        )

    doc: dict[str, Any] = {
        "schema": "hawking.headless.dirty_tree_preservation.v1",
        "generated_at": generated_at,
        "receipt_path": str(receipt_path),
        "observer": {
            "path": str(observer),
            "cwd": str(cwd),
            "head": observer_head,
            "branch": observer_branch,
            "sparse": observer_sparse,
            "note": (
                "This lane is an observer. A file missing here is not evidence "
                "it does not exist. Preservation is judged against the main worktree."
            ),
        },
        "preservation_target": {
            "path": str(main_wt),
            "head": head,
            "branch": branch,
            "commit_count": commit_count,
            "sparse": target_sparse,
            "remote": git_out(main_wt, "remote", "-v").splitlines()[0] if git_out(main_wt, "remote", "-v") else None,
        },
        "exclusions": {
            "generated_dir_names": list(GENERATED_DIR_NAMES),
            "why": GENERATED_WHY,
            "excluded_paths": excluded_generated,
        },
        "campaign_prefixes": list(CAMPAIGN_PREFIXES),
        "classification_rule": (
            "A path is 'campaign' iff it is under tools/headless/, "
            "hcli/, receipts/headless/, or .hcli/. Everything else "
            "is presumed pre-existing and precious."
        ),
        "counts": {
            "porcelain_normal": len(rows_normal),
            "porcelain_uall_raw": len(rows_uall),
            "porcelain_uall_kept": len(kept_rows),
            "modified": counts_kept["modified"],
            "untracked": counts_kept["untracked"],
            "staged": counts_kept["staged"],
            "pre_existing": pre_n,
            "campaign": camp_n,
            "excluded_generated_paths": len(excluded_generated),
            "xy": counts_kept["xy"],
        },
        "prefix_histogram": prefix_histogram([r["path"] for r in kept_rows]),
        "inventory": inventory,
        "reverted": {
            "live_tracked_dirty_count": len(tracked_snapshot),
            "live_untracked_count": len(untracked_snapshot),
            "live_still_differs_count": len(live_still),
            "live_new_present_count": len(live_new_present),
            "live_missing_on_disk": len(live_missing),
            "live_matches_head_count": len(live_reverted),
            "live_reverted": live_reverted,
            "live_missing": live_missing,
            "live_new_present": [rec["path"] for rec in live_new_present],
            "function_result_on_snapshot": live_lost,
            "check": (
                "for path in dirty_snapshot: "
                "if HEAD:path exists and sha1(worktree path)==sha1(HEAD:path): LOSS"
            ),
        },
        "truncation": [
            {
                "path": rec["path"],
                "xy": rec["xy"],
                "classification": rec["classification"],
                "disk_bytes": rec["disk_bytes"],
                "head_bytes": rec["head_bytes"],
            }
            for rec in truncation
        ],
        "disk_truth_baseline": dt_baseline,
        "skip_worktree": skip,
        "ignored_generated": ignored,
        "fsck": fsck,
        "worktrees": worktrees,
        "grok_branches": grok,
        "what_i_watched_fail": fail,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "method": {
            "status_commands": [
                "git -c core.quotePath=false status --porcelain=v1",
                "git -c core.quotePath=false status --porcelain=v1 -uall",
            ],
            "still_differs": "git hash-object -- PATH  vs  git cat-file of HEAD:PATH",
            "truncation": "st_size(worktree) < git cat-file -s HEAD:PATH",
            "fsck": "git fsck --no-reflogs --lost-found",
            "worktrees": "git worktree list --porcelain + git status --porcelain=v1 per worktree",
            "grok_branches": "git for-each-ref refs/heads/grok",
            "no_mutations_of_target": True,
        },
    }

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(doc, indent=1) + "\n")

    print_report(doc)
    if verdict == "LOSS":
        return 2
    if verdict == "INCONCLUSIVE":
        return 1
    if not fail.get("revert_detector_fired"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
