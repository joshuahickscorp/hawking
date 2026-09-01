"""G117: a self-improving system must not erase its own forgotten work.

~/Downloads/hawking-copy is a 17 GB SEPARATE CLONE - not a worktree, so it is
invisible to `git worktree list` from the canonical repo and reads as an
unrelated folder. Its HEAD 2f670c24 is UNREACHABLE from canonical main, it
carries 517 dirty paths and 4061 untracked files across 55 branches, and any
routine "clean up stale directories" sweep would have taken all of it.

Before HCLI gains autonomous cleanup authority, deletion has to pass a gate. This
is that gate, and it answers four questions about a target before anything is
removed:

    UNREACHABLE COMMITS   does it hold commits no canonical ref can reach
    DIRTY PATHS           does it hold uncommitted or untracked work
    EXTERNAL CLONE        is it a separate .git, not a worktree of canonical
    PRESERVATION          is there a receipt saying the work was captured first

A target failing any of the first three without the fourth is REFUSED. That is
the whole contract: not "warn", not "log and continue" - refuse, and name which
check failed.

WHAT IS ALREADY PRESERVED, and why the 17 GB is still on disk. The clone's
commits are fetched to refs/preserved/hawking-copy-head, so the history is
recoverable from canonical even if the directory vanishes. The working files are
NOT preserved and the directory is therefore NOT deletable under this gate. That
is the correct state: nothing was deleted, and the gate says out loud why nothing
may be.

    python3 tools/future/source_safety.py --build
    python3 tools/future/source_safety.py --check ~/Downloads/hawking-copy
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/source_safety.py"
RECEIPT_NAME = "PRESERVED_RECOVERY_SOURCE.json"

# Registered recovery sources: places holding work that canonical does not.
REGISTERED = (
    {
        "id": "hawking-copy",
        "path": "~/Downloads/hawking-copy",
        "kind": "separate_clone",
        "preserved_ref": "refs/preserved/hawking-copy-head",
        "why_it_matters": (
            "a separate .git, not a worktree, so `git worktree list` from "
            "canonical does not show it and it reads as an unrelated folder"
        ),
    },
)

CHECKS = ("unreachable_commits", "dirty_paths", "external_clone", "preservation")


class DeletionRefused(RuntimeError):
    """The target holds work that canonical cannot recover."""


def _git(*args: str, cwd: Path | None = None) -> str:
    r = subprocess.run(["git", "--no-optional-locks", *args],
                       cwd=str(cwd or REPO), capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def _is_git(p: Path) -> bool:
    return (p / ".git").exists()


def inspect_target(path: str | Path) -> dict[str, Any]:
    """Four facts about a deletion target. No opinion, no action."""
    p = Path(path).expanduser()
    out: dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        out.update(is_git=False, head=None, dirty_paths=None, untracked=None,
                   branches=None, head_reachable_from_canonical=None,
                   head_object_in_canonical=None, external_clone=False)
        return out
    out["is_git"] = _is_git(p)
    if not out["is_git"]:
        out.update(head=None, dirty_paths=None, untracked=None, branches=None,
                   head_reachable_from_canonical=None,
                   head_object_in_canonical=None, external_clone=False)
        return out
    head = _git("rev-parse", "HEAD", cwd=p).strip() or None
    out["head"] = head
    out["dirty_paths"] = len(_git("status", "--porcelain", cwd=p).splitlines())
    out["untracked"] = len(
        _git("ls-files", "--others", "--exclude-standard", cwd=p).splitlines())
    out["branches"] = len(
        _git("for-each-ref", "refs/heads", "--format=%(refname)", cwd=p).splitlines())
    # A worktree of canonical shares canonical's object store; a separate clone
    # does not. Resolving the common dir is what tells them apart.
    common = _git("rev-parse", "--git-common-dir", cwd=p).strip()
    resolved = (p / common).resolve() if common and not common.startswith("/") \
        else Path(common).resolve() if common else None
    out["git_common_dir"] = str(resolved) if resolved else None
    out["external_clone"] = bool(
        resolved and resolved != (REPO / ".git").resolve())
    if head:
        r = subprocess.run(
            ["git", "--no-optional-locks", "merge-base", "--is-ancestor",
             head, "main"], cwd=str(REPO), capture_output=True)
        out["head_reachable_from_canonical"] = r.returncode == 0
        e = subprocess.run(["git", "--no-optional-locks", "cat-file", "-e", head],
                           cwd=str(REPO), capture_output=True)
        out["head_object_in_canonical"] = e.returncode == 0
    else:
        out["head_reachable_from_canonical"] = None
        out["head_object_in_canonical"] = None
    return out


def preservation_for(path: str | Path) -> dict[str, Any]:
    """Is this target's history recoverable from canonical without it?"""
    p = str(Path(path).expanduser())
    reg = next((r for r in REGISTERED
                if str(Path(r["path"]).expanduser()) == p), None)
    if reg is None:
        return {"registered": False, "preserved_ref": None,
                "history_recoverable": False,
                "working_files_preserved": False,
                "why": "not a registered recovery source; nothing claims to "
                       "have captured it"}
    ref = reg["preserved_ref"]
    present = bool(_git("rev-parse", "--verify", "--quiet", ref).strip())
    return {
        "registered": True,
        "preserved_ref": ref,
        "preserved_sha": _git("rev-parse", ref).strip() or None,
        "history_recoverable": present,
        # The commits are fetched; the UNCOMMITTED files are not. That is the
        # distinction that keeps this directory undeletable, and collapsing the
        # two would license exactly the deletion this gate exists to stop.
        "working_files_preserved": False,
        "why": (
            "commits are fetched to the preserved ref so history survives the "
            "directory. Uncommitted and untracked working files are NOT "
            "captured, so the directory is not deletable."
        ),
    }


def gate(path: str | Path) -> dict[str, Any]:
    """May this be deleted? Returns the decision and every check behind it."""
    t = inspect_target(path)
    pres = preservation_for(path)
    failures: list[dict[str, Any]] = []
    if not t["exists"]:
        return {"path": t["path"], "decision": "NOTHING_TO_DELETE",
                "target": t, "preservation": pres, "failures": []}
    if t.get("head") and t.get("head_reachable_from_canonical") is False:
        failures.append({
            "check": "unreachable_commits",
            "detail": f"HEAD {t['head'][:9]} is not an ancestor of main",
            "recoverable_anyway": bool(pres.get("history_recoverable")),
        })
    if (t.get("dirty_paths") or 0) or (t.get("untracked") or 0):
        failures.append({
            "check": "dirty_paths",
            "detail": f"{t.get('dirty_paths')} dirty, {t.get('untracked')} untracked",
            "recoverable_anyway": bool(pres.get("working_files_preserved")),
        })
    if t.get("external_clone"):
        failures.append({
            "check": "external_clone",
            "detail": "separate .git; invisible to `git worktree list` from "
                      "canonical",
            "recoverable_anyway": bool(pres.get("history_recoverable")),
        })
    blocking = [f for f in failures if not f["recoverable_anyway"]]
    return {
        "path": t["path"],
        "decision": "REFUSED" if blocking else ("ALLOWED" if not failures
                                                else "ALLOWED_PRESERVED"),
        "blocking_failures": [f["check"] for f in blocking],
        "failures": failures,
        "target": t,
        "preservation": pres,
    }


def require_safe_to_delete(path: str | Path) -> dict[str, Any]:
    """Raise unless the gate allows it. This is the callable HCLI must use."""
    g = gate(path)
    if g["decision"] == "REFUSED":
        raise DeletionRefused(
            f"{g['path']} holds work canonical cannot recover: "
            f"{g['blocking_failures']}. A self-improving system must not erase "
            "its own forgotten work - preserve it first, then delete."
        )
    return g


def build() -> dict[str, Any]:
    rows = []
    for r in REGISTERED:
        g = gate(r["path"])
        rows.append({**r, "gate": {k: g[k] for k in
                                   ("decision", "blocking_failures", "failures")},
                     "state": g["target"], "preservation": g["preservation"]})
    return {
        "obligation": "G117",
        "question": "may this be deleted, and what would be lost if it were?",
        "checks": list(CHECKS),
        "registered_sources": rows,
        "nothing_deleted": True,
        "the_gate_is_a_refusal_not_a_warning": (
            "require_safe_to_delete RAISES. A gate that logs and continues is "
            "not a gate, and autonomous cleanup authority is exactly the "
            "setting where nobody reads the log."
        ),
        "why_17_gb_is_still_on_disk": (
            "hawking-copy's COMMITS are recoverable from "
            "refs/preserved/hawking-copy-head, but its 517 dirty and 4061 "
            "untracked files are not captured anywhere. History preserved is "
            "not work preserved, and treating them as the same would license "
            "the deletion this gate exists to stop."
        ),
        "evidence_class": "DERIVED_FROM_GIT",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", metavar="PATH")
    a = ap.parse_args(argv)
    if a.check:
        print(json.dumps(gate(a.check), indent=1))
        return 0
    doc = build()
    if a.build:
        print(write_receipt(REPO / "receipts" / "future" / RECEIPT_NAME,
                            doc, RECORDED_BY))
        return 0
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
