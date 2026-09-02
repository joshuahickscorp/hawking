"""Recoverability without a worktree.

A worktree buys ISOLATION from a second concurrent writer. That was never
the actual gap here: there is exactly one writer, and a live sovereign
daemon reads these files continuously while it runs. What was missing was
RECOVERABILITY -- a point you can always get back to without stopping the
daemon, checking anything out, or disturbing a single file it might be
mid-read on.

The technique: point ``GIT_INDEX_FILE`` at a throwaway file outside
``.git``, ``git add -A`` the whole tree into THAT index, ``git write-tree``,
then ``git commit-tree`` the result on top of HEAD. None of the three
touches the real index, HEAD, or a working-tree file -- ``git status``
immediately after a checkpoint is identical to ``git status`` immediately
before one (verified: see test_checkpoint.py). The resulting commit is
filed under ``refs/hcli-checkpoints/<id>``, a ref namespace no branch, tag,
or checkout ever reads, so it is fully recoverable (a ref keeps the object
database from ever garbage-collecting it) without being visible anywhere
else in the repo's history.

Ceiling, stated honestly:
  * untracked-but-``.gitignore``'d files are NOT captured -- ``git add -A``
    respects ``.gitignore`` exactly like any other ``git add``.
  * this is RECOVERABILITY, not ISOLATION. Two writers editing the same
    tree at the same moment can still stomp each other; a checkpoint only
    guarantees a way back to a prior state, not that nobody clobbers the
    current one first. Reach for a worktree (or a real second writer) when
    isolation, not recovery, is the actual requirement.
  * ``restore_checkpoint`` only ever writes into a caller-supplied NEW,
    empty directory outside ``repo_root`` -- never onto the live tree.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

REF_PREFIX = "refs/hcli-checkpoints/"
_MESSAGE_PREFIX = "hcli-checkpoint: "
_LABEL_RE = re.compile(r"[^A-Za-z0-9._-]+")


class CheckpointError(RuntimeError):
    """A checkpoint could not be created, listed, or restored."""


def _slug(label: Any) -> str:
    text = _LABEL_RE.sub("-", str(label or "").strip()).strip("-.")
    return text or "checkpoint"


def _run(
    repo_root: Path, *args: str, env: Optional[Dict[str, str]] = None, timeout: float = 60.0
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True, text=True, timeout=timeout, check=False, env=env,
    )


def _checkpoint_id_of(ref: str) -> str:
    return ref[len(REF_PREFIX):] if ref.startswith(REF_PREFIX) else ref


def checkpoint(repo_root: Union[str, Path], label: str) -> Dict[str, Any]:
    """Preserve the current working tree WITHOUT disturbing it.

    Never stashes, never resets, never checks anything out. Builds the
    snapshot through a temporary index so the real index, HEAD, and every
    working-tree file are left exactly as they were. See module docstring
    for the technique and its ceiling.
    """
    root = Path(repo_root).resolve()
    toplevel = _run(root, "rev-parse", "--show-toplevel")
    if toplevel.returncode != 0:
        raise CheckpointError(f"not a git repository: {toplevel.stderr.strip()}")
    if Path(toplevel.stdout.strip()).resolve() != root:
        raise CheckpointError(
            f"repo_root {root} is not the toplevel of its own git repo "
            f"({toplevel.stdout.strip()!r})"
        )

    head = _run(root, "rev-parse", "HEAD")
    parent_sha = head.stdout.strip() if head.returncode == 0 else None

    checkpoint_id = f"{int(time.time() * 1000)}-{_slug(label)}"
    ref = REF_PREFIX + checkpoint_id

    with tempfile.TemporaryDirectory(prefix="hcli-checkpoint-") as scratch:
        env = dict(os.environ)
        env["GIT_INDEX_FILE"] = str(Path(scratch) / "index")

        add = _run(root, "add", "-A", env=env)
        if add.returncode != 0:
            raise CheckpointError(f"git add -A into scratch index failed: {add.stderr.strip()}")

        write_tree = _run(root, "write-tree", env=env)
        if write_tree.returncode != 0:
            raise CheckpointError(f"git write-tree failed: {write_tree.stderr.strip()}")
        tree_sha = write_tree.stdout.strip()

    commit_args = ["commit-tree", tree_sha, "-m", f"{_MESSAGE_PREFIX}{label}"]
    if parent_sha:
        commit_args += ["-p", parent_sha]
    commit = _run(root, *commit_args)
    if commit.returncode != 0:
        raise CheckpointError(f"git commit-tree failed: {commit.stderr.strip()}")
    commit_sha = commit.stdout.strip()

    update_ref = _run(root, "update-ref", ref, commit_sha)
    if update_ref.returncode != 0:
        raise CheckpointError(f"git update-ref failed: {update_ref.stderr.strip()}")

    branch = _run(root, "rev-parse", "--abbrev-ref", "HEAD")
    return {
        "checkpoint_id": checkpoint_id,
        "ref": ref,
        "commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "parent_sha": parent_sha,
        "label": str(label or ""),
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "created_at": time.time(),
    }


def list_checkpoints(repo_root: Union[str, Path]) -> List[Dict[str, Any]]:
    """Every checkpoint recorded under refs/hcli-checkpoints/, newest first."""
    root = Path(repo_root).resolve()
    result = _run(
        root, "for-each-ref",
        "--format=%(refname)%00%(objectname)%00%(creatordate:iso-strict)%00%(subject)",
        REF_PREFIX,
    )
    if result.returncode != 0:
        raise CheckpointError(f"git for-each-ref failed: {result.stderr.strip()}")

    out: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\x00")
        if len(parts) != 4:
            continue
        refname, commit_sha, created_at, subject = parts
        label = subject[len(_MESSAGE_PREFIX):] if subject.startswith(_MESSAGE_PREFIX) else subject
        out.append({
            "checkpoint_id": _checkpoint_id_of(refname),
            "ref": refname,
            "commit_sha": commit_sha,
            "created_at": created_at,
            "label": label,
        })
    out.sort(key=lambda c: c["checkpoint_id"], reverse=True)
    return out


def restore_checkpoint(
    repo_root: Union[str, Path], checkpoint_id: str, into: Union[str, Path]
) -> Dict[str, Any]:
    """Materialise a checkpoint into a NEW, empty directory. Never over the
    live tree: refused if ``into`` is ``repo_root`` or anywhere inside it.
    """
    root = Path(repo_root).resolve()
    ref = str(checkpoint_id) if str(checkpoint_id).startswith(REF_PREFIX) else REF_PREFIX + str(checkpoint_id)

    resolved = _run(root, "rev-parse", "--verify", ref)
    if resolved.returncode != 0:
        raise CheckpointError(f"no such checkpoint: {checkpoint_id!r}")
    commit_sha = resolved.stdout.strip()

    dest = Path(into).resolve()
    if dest == root or root in dest.parents:
        raise CheckpointError(f"refusing to restore over the live working tree at {root}")
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise CheckpointError(f"restore destination {dest} is not empty")

    archive = subprocess.Popen(
        ["git", "-C", str(root), "archive", "--format=tar", commit_sha],
        stdout=subprocess.PIPE,
    )
    extract = subprocess.run(
        ["tar", "-xf", "-", "-C", str(dest)],
        stdin=archive.stdout, capture_output=True, text=True,
    )
    if archive.stdout is not None:
        archive.stdout.close()
    archive_rc = archive.wait()
    if archive_rc != 0 or extract.returncode != 0:
        raise CheckpointError(
            f"restore failed (git archive rc={archive_rc}, tar rc={extract.returncode}): "
            f"{extract.stderr.strip()}"
        )

    files_restored = sum(1 for p in dest.rglob("*") if p.is_file())
    return {
        "checkpoint_id": _checkpoint_id_of(ref),
        "ref": ref,
        "commit_sha": commit_sha,
        "into": str(dest),
        "files_restored": files_restored,
    }


__all__ = [
    "REF_PREFIX",
    "CheckpointError",
    "checkpoint",
    "list_checkpoints",
    "restore_checkpoint",
]
