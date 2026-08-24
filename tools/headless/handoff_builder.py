#!/usr/bin/env python3
"""Assemble HAWKING_HEADLESS_COMPLETION_V3 from live state only.

A later session with no memory of this one has to resume from the file this
writes. Every claim therefore carries an evidence path AND a reproducing
command. A claim lacking either is refused and never emitted. A fact that
cannot be derived is recorded as `unknown`, never as a plausible default.

This builder does not hand-write status. It reads:

  * git HEAD (always)
  * tools/headless on disk in this worktree
  * receipts/headless at HEAD (git objects; this worktree may be sparse)
  * the primary checkout behind `git rev-parse --git-common-dir`, for
    gitignored live state (.hcli, extra receipts, dirty HCLI source)
  * .hcli / .haider in this worktree, if present

Default emit path is a STAGING file, not the durable name. The cold-read
test promotes the staging file to
`.haider/HAWKING_HEADLESS_COMPLETION_V3.json` only after every real claim
is SUPPORTED and a negative control has been seen to fail.

Run:
    python3 tools/headless/handoff_builder.py
    python3 tools/headless/handoff_builder.py --refuse-only
    python3 tools/headless/handoff_builder.py --output /tmp/handoff.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCHEMA = "hawking.headless.completion.v3"
STAGING_REL = ".haider/HAWKING_HEADLESS_COMPLETION_V3.staging.json"
FINAL_REL = ".haider/HAWKING_HEADLESS_COMPLETION_V3.json"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


# ---------------------------------------------------------------------------
# Self-contained resolver inlined into every python3 -c reproducing command.
# Cold readers do not import this module; they re-run the command as stored.
# Search order for live bytes: this worktree disk, then primary checkout disk.
# git_bytes is HEAD only and never consults a dirty tree.
# ---------------------------------------------------------------------------
RESOLVER = r"""
import json, subprocess, sys
from pathlib import Path

def _primary():
    cd = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"], text=True
    ).strip()
    p = Path(cd)
    if not p.is_absolute():
        p = (Path(".").resolve() / p).resolve()
    else:
        p = p.resolve()
    return p.parent if p.name == ".git" else p.parent

def git_bytes(rel):
    r = subprocess.run(
        ["git", "show", f"HEAD:{rel}"], capture_output=True
    )
    if r.returncode != 0:
        sys.exit("not at HEAD:" + rel)
    return r.stdout

def git_text(rel):
    return git_bytes(rel).decode("utf-8")

def disk_bytes(rel):
    seen = set()
    for base in (Path(".").resolve(), _primary()):
        cand = (base / rel).resolve()
        if cand in seen:
            continue
        seen.add(cand)
        if cand.is_file():
            return cand.read_bytes()
    sys.exit("not on disk:" + rel)

def disk_text(rel):
    return disk_bytes(rel).decode("utf-8")

def git_exists(rel):
    r = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{rel}"],
        capture_output=True,
    )
    return r.returncode == 0
"""


class ClaimRefused(ValueError):
    """A claim is not admissible and must not be emitted."""


@dataclass
class Claim:
    id: str
    kind: str
    statement: str
    evidence_path: str
    reproducing_command: List[str]
    derived_from: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "statement": self.statement,
            "evidence_path": self.evidence_path,
            "reproducing_command": self.reproducing_command,
            "reproducing_command_pretty": shlex.join(self.reproducing_command),
            "derived_from": self.derived_from,
        }
        d.update(self.extra)
        return d


class IdGen:
    def __init__(self) -> None:
        self._n: Dict[str, int] = {}

    def next(self, prefix: str) -> str:
        self._n[prefix] = self._n.get(prefix, 0) + 1
        return f"{prefix}{self._n[prefix]:03d}"


def git(
    args: Sequence[str],
    cwd: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd or REPO),
        capture_output=True,
        text=True,
        check=check,
    )


def git_out(args: Sequence[str], cwd: Optional[Path] = None) -> str:
    r = git(args, cwd=cwd, check=False)
    if r.returncode != 0:
        return ""
    return (r.stdout or "").rstrip("\n")


def at_head(rel: str) -> bool:
    r = git(["cat-file", "-e", f"HEAD:{rel}"], check=False)
    return r.returncode == 0


def git_bytes(rel: str) -> bytes:
    r = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        cwd=str(REPO),
        capture_output=True,
        check=True,
    )
    return r.stdout


def git_text(rel: str) -> str:
    return git_bytes(rel).decode("utf-8")


def git_json(rel: str) -> Any:
    return json.loads(git_text(rel))


def ls_head(*prefixes: str) -> List[str]:
    args = ["ls-tree", "-r", "--name-only", "HEAD"]
    if prefixes:
        args.append("--")
        args.extend(prefixes)
    r = git(args, check=False)
    if r.returncode != 0:
        return []
    return [ln for ln in r.stdout.splitlines() if ln]


def primary_checkout(repo: Path = REPO) -> Optional[Path]:
    cd = git_out(["rev-parse", "--git-common-dir"], cwd=repo)
    if not cd:
        return None
    p = Path(cd)
    if not p.is_absolute():
        p = (repo / p).resolve()
    else:
        p = p.resolve()
    root = p.parent if p.name == ".git" else p.parent
    return root if root.is_dir() else None


def py_cmd(*body: str) -> List[str]:
    return ["python3", "-c", RESOLVER + "\n" + "\n".join(body)]


def py_git_json_eq(path: str, dotted: str, expected: Any) -> List[str]:
    parts = json.dumps(dotted.split("."))
    payload = json.dumps(expected)
    return py_cmd(
        f"d = json.loads(git_text({path!r}))",
        "cur = d",
        f"for part in {parts}:",
        "    cur = cur[int(part)] if isinstance(cur, list) else cur[part]",
        f"exp = json.loads({json.dumps(payload)})",
        "assert cur == exp, (cur, exp)",
    )


def py_disk_json_eq(path: str, dotted: str, expected: Any) -> List[str]:
    parts = json.dumps(dotted.split("."))
    payload = json.dumps(expected)
    return py_cmd(
        f"d = json.loads(disk_text({path!r}))",
        "cur = d",
        f"for part in {parts}:",
        "    cur = cur[int(part)] if isinstance(cur, list) else cur[part]",
        f"exp = json.loads({json.dumps(payload)})",
        "assert cur == exp, (cur, exp)",
    )


def py_git_text_contains(path: str, needle: str, present: bool = True) -> List[str]:
    return py_cmd(
        f"t = git_text({path!r})",
        f"needle = {needle!r}",
        f"ok = needle in t",
        f"assert ok is {present}, ('found' if ok else 'missing', needle)",
    )


def py_disk_text_contains(path: str, needle: str, present: bool = True) -> List[str]:
    return py_cmd(
        f"t = disk_text({path!r})",
        f"needle = {needle!r}",
        f"ok = needle in t",
        f"assert ok is {present}, ('found' if ok else 'missing', needle)",
    )


def py_git_jsonl_len(path: str, n: int) -> List[str]:
    return py_cmd(
        f"raw = git_text({path!r})",
        "rows = [json.loads(l) for l in raw.splitlines() if l.strip()]",
        "cur = len(rows)",
        f"exp = {int(n)}",
        "assert cur == exp, (cur, exp)",
    )


def admit(claim: Claim) -> Claim:
    """Refuse a claim that a cold reader could not check."""
    if not claim.id or not str(claim.id).strip():
        raise ClaimRefused("refusing claim with empty id")
    if not claim.kind or not str(claim.kind).strip():
        raise ClaimRefused(f"refusing claim {claim.id!r}: missing kind")
    if not claim.statement or not str(claim.statement).strip():
        raise ClaimRefused(f"refusing claim {claim.id!r}: missing statement")
    if not claim.evidence_path or not str(claim.evidence_path).strip():
        raise ClaimRefused(
            f"refusing claim {claim.id!r}: missing evidence_path "
            f"(statement was {claim.statement!r})"
        )
    cmd = claim.reproducing_command
    if not cmd:
        raise ClaimRefused(
            f"refusing claim {claim.id!r}: missing reproducing_command "
            f"(a claim with no command is not admissible)"
        )
    if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
        raise ClaimRefused(
            f"refusing claim {claim.id!r}: reproducing_command must be a list of strings"
        )
    if not any(part.strip() for part in cmd):
        raise ClaimRefused(
            f"refusing claim {claim.id!r}: reproducing_command is empty"
        )
    return claim


def demonstrate_refusal() -> Tuple[int, List[str]]:
    """Deliberately malform claims and show the refusal firing.

    Exit 0 only if every malformed claim is refused. A builder that admits
    a claim with no command is the failure mode this campaign exists to
    prevent.
    """
    ids = IdGen()
    malformed: List[Tuple[str, Claim]] = [
        (
            "no reproducing_command",
            Claim(
                id="MALFORMED_NO_COMMAND",
                kind="capability",
                statement="the scheduler is bounded",
                evidence_path="tools/headless/hcli_scheduler_test.py",
                reproducing_command=[],
            ),
        ),
        (
            "no evidence_path",
            Claim(
                id="MALFORMED_NO_EVIDENCE",
                kind="capability",
                statement="repair depth is capped at 3",
                evidence_path="",
                reproducing_command=["python3", "-c", "print('no')"],
            ),
        ),
        (
            "empty statement",
            Claim(
                id="MALFORMED_NO_STATEMENT",
                kind="fact",
                statement="",
                evidence_path="receipts/headless/DISK_TRUTH.json",
                reproducing_command=["python3", "-c", "print('no')"],
            ),
        ),
    ]
    lines: List[str] = []
    fired = 0
    for label, claim in malformed:
        try:
            admit(claim)
            msg = (
                f"VACUOUS: {label} was admitted as {claim.id} — "
                f"the builder is broken"
            )
            print(msg)
            lines.append(msg)
        except ClaimRefused as exc:
            msg = f"REFUSED ({label}): {exc}"
            print(msg)
            lines.append(msg)
            fired += 1
        # ids is unused; keep a next() so a later reader sees the generator
        # is live. The malformed ids are fixed strings on purpose.
        _ = ids
    return (0 if fired == len(malformed) else 1), lines


def dotted_get(doc: Any, dotted: str) -> Any:
    cur = doc
    for part in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


# ---------------------------------------------------------------------------
# Live-state discovery
# ---------------------------------------------------------------------------


def discover(repo: Path = REPO) -> Dict[str, Any]:
    head = git_out(["rev-parse", "HEAD"]) or "unknown"
    branch = git_out(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    porcelain = git_out(["status", "--porcelain"])
    dirty_entries = [ln for ln in porcelain.splitlines() if ln] if porcelain else []
    sparse_on = git_out(["config", "--get", "core.sparseCheckout"]) == "true"
    sparse_list = git_out(["sparse-checkout", "list"])
    sparse_roots = [ln for ln in sparse_list.splitlines() if ln] if sparse_list else []

    primary = primary_checkout(repo)
    primary_head = (
        git_out(["rev-parse", "HEAD"], cwd=primary) if primary is not None else "unknown"
    )
    same_head = (
        primary is not None and head != "unknown" and primary_head == head
    )

    headless_dir = repo / "tools" / "headless"
    harnesses = sorted(
        p.name
        for p in headless_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
    ) if headless_dir.is_dir() else []

    git_receipts = [
        p
        for p in ls_head("receipts/headless")
        if p.endswith(".json") or p.endswith(".jsonl")
    ]
    git_receipts_other = [
        p
        for p in ls_head("receipts/headless")
        if p not in git_receipts
    ]

    haider_here = (repo / "hcli").is_dir() or (repo / "tools" / "haider" / "hcli").is_dir()
    receipts_here = (repo / "receipts" / "headless").is_dir()
    hcli_here = (repo / ".hcli").is_dir()
    haider_dot_here = (repo / ".haider").is_dir()

    haider_tests: List[str] = []
    if headless_dir.is_dir():
        for p in sorted(headless_dir.glob("*_test.py")):
            text = p.read_text(encoding="utf-8")
            if "haider" in text and "sys.path.insert" in text:
                haider_tests.append(p.name)

    primary_extra_receipts: List[Dict[str, Any]] = []
    primary_extra_harnesses: List[str] = []
    primary_hcli: Dict[str, Any] = {"present": False}
    primary_goal: Dict[str, Any] = {"present": False}
    primary_dirty: List[str] = []
    if primary is not None:
        pr = primary / "receipts" / "headless"
        if pr.is_dir():
            git_base = {Path(p).name for p in git_receipts + git_receipts_other}
            for p in sorted(pr.iterdir()):
                if p.name.startswith("."):
                    continue
                if not p.is_file():
                    continue
                if p.name in git_base:
                    continue
                primary_extra_receipts.append(
                    {
                        "name": p.name,
                        "rel": f"receipts/headless/{p.name}",
                        "bytes": p.stat().st_size,
                        "sha256": _sha256(p),
                    }
                )
        ph = primary / "tools" / "headless"
        if ph.is_dir():
            here_names = set(harnesses)
            for p in sorted(ph.iterdir()):
                if p.is_file() and p.name not in here_names and p.name != "__pycache__":
                    if at_head(f"tools/headless/{p.name}"):
                        continue
                    # present on primary disk, not in this worktree listing
                    if p.name not in here_names and not at_head(f"tools/headless/{p.name}"):
                        primary_extra_harnesses.append(p.name)
        # dirty tracked files in the HCLI tree (live vs HEAD)
        diff = git_out(
            ["diff", "--name-only", "HEAD", "--", "hcli", "tools/haider/hcli"],
            cwd=primary,
        )
        primary_dirty = [ln for ln in diff.splitlines() if ln] if diff else []
        goal = primary / ".hcli" / "GOAL.md"
        if goal.is_file():
            text = goal.read_text(encoding="utf-8")
            primary_goal = {
                "present": True,
                "path": str(goal),
                "rel": ".hcli/GOAL.md",
                "bytes": goal.stat().st_size,
                "sha256": _sha256(goal),
                "text": text,
                "pending": len(re.findall(r"^- \[ \]", text, re.M)),
                "checked": len(re.findall(r"^- \[[xX]\]", text, re.M)),
            }
        hcli_dir = primary / ".hcli"
        if hcli_dir.is_dir():
            dag = hcli_dir / "dag.json"
            dag_doc: Any = None
            if dag.is_file():
                try:
                    dag_doc = json.loads(dag.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    dag_doc = None
            units = (dag_doc or {}).get("units") if isinstance(dag_doc, dict) else None
            status_counts: Dict[str, int] = {}
            if isinstance(units, dict):
                for u in units.values():
                    if isinstance(u, dict):
                        st = str(u.get("status") or "unknown")
                        status_counts[st] = status_counts.get(st, 0) + 1
            receipt_count = (
                len(list((hcli_dir / "receipts").glob("*")))
                if (hcli_dir / "receipts").is_dir()
                else 0
            )
            primary_hcli = {
                "present": True,
                "path": str(hcli_dir),
                "entries": sorted(p.name for p in hcli_dir.iterdir()),
                "receipt_count": receipt_count,
                "dag": None
                if not isinstance(dag_doc, dict)
                else {
                    "version": dag_doc.get("version", "unknown"),
                    "active_decode_limit": dag_doc.get(
                        "active_decode_limit", "unknown"
                    ),
                    "active_decode_limit_source": dag_doc.get(
                        "active_decode_limit_source", "unknown"
                    ),
                    "no_progress_threshold": dag_doc.get(
                        "no_progress_threshold", "unknown"
                    ),
                    "n_units": len(units) if isinstance(units, dict) else "unknown",
                    "status_counts": status_counts,
                    "unit_ids": sorted(units) if isinstance(units, dict) else [],
                },
            }

    return {
        "repo": str(repo),
        "head": head or "unknown",
        "branch": branch or "unknown",
        "dirty_this_worktree": dirty_entries,
        "sparse_checkout": sparse_on,
        "sparse_roots": sparse_roots,
        "primary_checkout": str(primary) if primary else "unknown",
        "primary_head": primary_head or "unknown",
        "primary_same_head": same_head,
        "primary_dirty_hcli": primary_dirty,
        "this_worktree": {
            "tools_haider_hcli": haider_here,
            "receipts_headless": receipts_here,
            "dot_hcli": hcli_here,
            "dot_haider": haider_dot_here,
        },
        "harnesses_on_disk": harnesses,
        "haider_dependent_tests": haider_tests,
        "git_receipts": git_receipts,
        "git_receipts_other": git_receipts_other,
        "primary_extra_receipts": primary_extra_receipts,
        "primary_extra_harnesses": primary_extra_harnesses,
        "primary_hcli": primary_hcli,
        "primary_goal": primary_goal,
    }


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Claim assembly — every field is read from a file or a git object
# ---------------------------------------------------------------------------


def _add(out: List[Claim], claim: Claim) -> None:
    out.append(admit(claim))


def build_claims(disc: Dict[str, Any], ids: IdGen) -> List[Claim]:
    out: List[Claim] = []
    _facts(out, disc, ids)
    _git_receipts(out, disc, ids)
    _source_constants_at_head(out, ids)
    _runnable_harnesses(out, disc, ids)
    _live_primary(out, disc, ids)
    _blocked(out, disc, ids)
    _unknown(out, disc, ids)
    _not_done(out, disc, ids)
    _watched_fail(out, disc, ids)
    return out


def _facts(out: List[Claim], disc: Dict[str, Any], ids: IdGen) -> None:
    head = disc["head"]
    _add(
        out,
        Claim(
            id=ids.next("F"),
            kind="fact",
            statement=f"git HEAD is {head}",
            evidence_path=".git",
            reproducing_command=py_cmd(
                "import subprocess",
                "cur = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()",
                f"exp = {head!r}",
                "assert cur == exp, (cur, exp)",
            ),
            derived_from={"how": "git rev-parse HEAD", "source": ".git"},
        ),
    )
    branch = disc["branch"]
    _add(
        out,
        Claim(
            id=ids.next("F"),
            kind="fact",
            statement=f"the current branch name is {branch}",
            evidence_path=".git",
            reproducing_command=py_cmd(
                "import subprocess",
                "cur = subprocess.check_output(['git','rev-parse','--abbrev-ref','HEAD'], text=True).strip()",
                f"exp = {branch!r}",
                "assert cur == exp, (cur, exp)",
            ),
            derived_from={"how": "git rev-parse --abbrev-ref HEAD", "source": ".git"},
        ),
    )
    dirty = disc["dirty_this_worktree"]
    _add(
        out,
        Claim(
            id=ids.next("F"),
            kind="fact",
            statement=(
                "this worktree has a clean git status"
                if not dirty
                else f"this worktree has {len(dirty)} dirty entries"
            ),
            evidence_path=".git",
            reproducing_command=py_cmd(
                "import subprocess",
                "raw = subprocess.check_output(['git','status','--porcelain'], text=True)",
                "cur = len([ln for ln in raw.splitlines() if ln])",
                f"exp = {len(dirty)}",
                "assert cur == exp, (cur, exp, raw)",
            ),
            derived_from={
                "how": "git status --porcelain",
                "source": ".git",
                "entries": dirty,
            },
        ),
    )
    sparse_on = bool(disc["sparse_checkout"])
    roots = list(disc["sparse_roots"])
    _add(
        out,
        Claim(
            id=ids.next("F"),
            kind="fact",
            statement=(
                f"sparse-checkout is {'on' if sparse_on else 'off'} with "
                f"{len(roots)} listed roots"
            ),
            evidence_path=".git",
            reproducing_command=py_cmd(
                "import subprocess",
                "on = subprocess.run(['git','config','--get','core.sparseCheckout'], capture_output=True, text=True)",
                f"assert (on.stdout.strip() == 'true') is {sparse_on}",
                # `git sparse-checkout list` EXITS NONZERO in a non-sparse
                # worktree ("fatal: this worktree is not sparse"), so
                # check_output raised and the claim failed for a reason that
                # was not about the claim. A handoff is checked in trees other
                # than the one that produced it; a probe must survive that.
                "r = subprocess.run(['git','sparse-checkout','list'], "
                "capture_output=True, text=True)",
                "cur = [ln for ln in r.stdout.splitlines() if ln] "
                "if r.returncode == 0 else []",
                f"exp = json.loads({json.dumps(json.dumps(roots))})",
                "assert cur == exp, (cur, exp)",
            ),
            derived_from={
                "how": "git config core.sparseCheckout + git sparse-checkout list",
                "source": ".git",
                "roots": roots,
            },
        ),
    )
    _add(
        out,
        Claim(
            id=ids.next("F"),
            kind="fact",
            statement=(
                "tools/haider/hcli is not on disk in this worktree"
                if not disc["this_worktree"]["tools_haider_hcli"]
                else "tools/haider/hcli is on disk in this worktree"
            ),
            evidence_path="tools/headless/handoff_builder.py",
            reproducing_command=py_cmd(
                "from pathlib import Path",
                "cur = Path('tools/haider/hcli').is_dir()",
                f"exp = {bool(disc['this_worktree']['tools_haider_hcli'])}",
                "assert cur is exp, (cur, exp)",
            ),
            derived_from={
                "how": "Path.is_dir",
                "source": "tools/haider/hcli",
                "note": (
                    "sparse-checkout omission is not evidence the path is "
                    "absent from git; see git ls-tree"
                ),
            },
        ),
    )
    _add(
        out,
        Claim(
            id=ids.next("F"),
            kind="fact",
            statement="tools/haider/hcli/scheduler.py exists at git HEAD",
            evidence_path="tools/haider/hcli/scheduler.py",
            reproducing_command=py_cmd(
                "assert git_exists('tools/haider/hcli/scheduler.py')",
            ),
            derived_from={
                "how": "git cat-file -e HEAD:tools/haider/hcli/scheduler.py",
                "source": "tools/haider/hcli/scheduler.py",
                "via": "git_head",
            },
        ),
    )
    harnesses = list(disc["harnesses_on_disk"])
    _add(
        out,
        Claim(
            id=ids.next("F"),
            kind="fact",
            statement=(
                f"this worktree materializes {len(harnesses)} files under "
                f"tools/headless"
            ),
            evidence_path="tools/headless",
            reproducing_command=py_cmd(
                "from pathlib import Path",
                "cur = sorted(p.name for p in Path('tools/headless').iterdir() "
                "if p.is_file() and not p.name.startswith('.'))",
                f"exp = json.loads({json.dumps(json.dumps(harnesses))})",
                "assert cur == exp, (cur, exp)",
            ),
            derived_from={
                "how": "listdir",
                "source": "tools/headless",
                "files": harnesses,
            },
        ),
    )
    n_receipts = len(disc["git_receipts"])
    _add(
        out,
        Claim(
            id=ids.next("F"),
            kind="fact",
            statement=(
                f"git HEAD tracks {n_receipts} JSON/JSONL files under "
                f"receipts/headless"
            ),
            evidence_path="receipts/headless",
            reproducing_command=py_cmd(
                "import subprocess",
                "raw = subprocess.check_output("
                "['git','ls-tree','-r','--name-only','HEAD','--','receipts/headless'],"
                " text=True)",
                "cur = len([ln for ln in raw.splitlines() "
                "if ln.endswith('.json') or ln.endswith('.jsonl')])",
                f"exp = {n_receipts}",
                "assert cur == exp, (cur, exp)",
            ),
            derived_from={
                "how": "git ls-tree -r --name-only HEAD receipts/headless",
                "source": "receipts/headless",
                "via": "git_head",
                "paths": disc["git_receipts"],
            },
        ),
    )
    receipts_here = bool(disc["this_worktree"]["receipts_headless"])
    _add(
        out,
        Claim(
            id=ids.next("F"),
            kind="fact",
            statement=(
                "receipts/headless is on disk in this worktree"
                if receipts_here
                else "receipts/headless is not on disk in this worktree "
                "(sparse checkout; objects remain at HEAD)"
            ),
            evidence_path="tools/headless/handoff_builder.py",
            reproducing_command=py_cmd(
                "from pathlib import Path",
                "cur = Path('receipts/headless').is_dir()",
                f"exp = {receipts_here}",
                "assert cur is exp, (cur, exp)",
            ),
            derived_from={
                "how": "Path.is_dir",
                "source": "receipts/headless",
            },
        ),
    )


def _headline(doc: Dict[str, Any]) -> Tuple[str, Any]:
    for key in (
        "verdict",
        "decision",
        "bar",
        "result",
        "headline",
        "conclusion",
        "obligation",
        "schema",
    ):
        if key in doc and doc[key] not in (None, "", [], {}):
            return key, doc[key]
    return "schema", doc.get("schema", "unknown")


def _git_receipts(out: List[Claim], disc: Dict[str, Any], ids: IdGen) -> None:
    for rel in disc["git_receipts"]:
        if rel.endswith(".jsonl"):
            raw = git_text(rel)
            rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
            _add(
                out,
                Claim(
                    id=ids.next("C"),
                    kind="capability",
                    statement=(
                        f"{rel} at git HEAD is a JSONL ledger with {len(rows)} rows"
                    ),
                    evidence_path=rel,
                    reproducing_command=py_git_jsonl_len(rel, len(rows)),
                    derived_from={
                        "how": "jsonl parse",
                        "source": rel,
                        "via": "git_head",
                    },
                ),
            )
            continue
        try:
            doc = git_json(rel)
        except (json.JSONDecodeError, subprocess.CalledProcessError):
            _add(
                out,
                Claim(
                    id=ids.next("U"),
                    kind="unknown",
                    statement=f"{rel} is at HEAD but did not parse as JSON",
                    evidence_path=rel,
                    reproducing_command=py_cmd(
                        f"t = git_text({rel!r})",
                        "try:",
                        "    json.loads(t)",
                        "    sys.exit('parsed but builder recorded unknown')",
                        "except json.JSONDecodeError:",
                        "    raise SystemExit(0)",
                    ),
                    derived_from={"how": "json.loads failed", "source": rel, "via": "git_head"},
                ),
            )
            continue
        if not isinstance(doc, dict):
            _add(
                out,
                Claim(
                    id=ids.next("F"),
                    kind="fact",
                    statement=f"{rel} at git HEAD is a JSON {type(doc).__name__}",
                    evidence_path=rel,
                    reproducing_command=py_cmd(
                        f"d = json.loads(git_text({rel!r}))",
                        f"assert isinstance(d, {type(doc).__name__})",
                    ),
                    derived_from={"how": "json.loads", "source": rel, "via": "git_head"},
                ),
            )
            continue
        schema = doc.get("schema", "unknown")
        key, value = _headline(doc)
        _add(
            out,
            Claim(
                id=ids.next("C"),
                kind="capability",
                statement=(
                    f"{rel} at git HEAD has schema {schema!r} and {key}={value!r}"
                ),
                evidence_path=rel,
                reproducing_command=py_cmd(
                    f"d = json.loads(git_text({rel!r}))",
                    f"assert d.get('schema') == {json.dumps(schema)}",
                    f"cur = d.get({key!r})",
                    f"exp = json.loads({json.dumps(json.dumps(value))})",
                    "assert cur == exp, (cur, exp)",
                ),
                derived_from={
                    "how": f"json field {key} plus schema",
                    "source": rel,
                    "via": "git_head",
                },
            ),
        )

    # Specific numeric / structural claims — values always taken from the file.
    specifics: List[Tuple[str, str, str]] = [
        (
            "receipts/headless/MACHINE_GENOME.json",
            "ACTIVE_DECODE_LIMIT",
            "MachineGenome at HEAD records ACTIVE_DECODE_LIMIT",
        ),
        (
            "receipts/headless/MACHINE_GENOME.json",
            "RESIDENT_RUNTIME_LIMIT",
            "MachineGenome at HEAD records RESIDENT_RUNTIME_LIMIT",
        ),
        (
            "receipts/headless/MACHINE_GENOME.json",
            "single_decoder_tps",
            "MachineGenome at HEAD records single_decoder_tps",
        ),
        (
            "receipts/headless/MACHINE_GENOME.json",
            "best_aggregate_tps",
            "MachineGenome at HEAD records best_aggregate_tps",
        ),
        (
            "receipts/headless/QWEN38_GRAVITY_NATIVE.json",
            "bar",
            "native Qwen3.8 gravity receipt at HEAD records bar",
        ),
        (
            "receipts/headless/QWEN38_GRAVITY_NATIVE.json",
            "decode.median_complete_wall_tps",
            "native Qwen3.8 gravity receipt at HEAD records median complete-wall tok/s",
        ),
        (
            "receipts/headless/QWEN38_GRAVITY_NATIVE.json",
            "three_part_bar.coherent_compiler_prose",
            "native Qwen3.8 gravity receipt at HEAD records coherent_compiler_prose",
        ),
        (
            "receipts/headless/QWEN_TUNE_SKIPPED.json",
            "decision",
            "Qwen small-tune receipt at HEAD records decision",
        ),
        (
            "receipts/headless/G003_ROOT_CAUSE.json",
            "verdict",
            "G003 root-cause receipt at HEAD records verdict",
        ),
        (
            "receipts/headless/GPU_ATTACK.json",
            "achieved_tps",
            "GPU_ATTACK receipt at HEAD records achieved_tps",
        ),
        (
            "receipts/headless/GPU_ATTACK.json",
            "achieved_k",
            "GPU_ATTACK receipt at HEAD records achieved_k",
        ),
        (
            "receipts/headless/GPU_ATTACK.json",
            "achieved_topology",
            "GPU_ATTACK receipt at HEAD records achieved_topology",
        ),
        (
            "receipts/headless/CAPABILITY_llamacpp-q5k.json",
            "overall.passed",
            "llamacpp-q5k capability suite at HEAD records overall.passed",
        ),
        (
            "receipts/headless/CAPABILITY_llamacpp-q5k.json",
            "overall.total",
            "llamacpp-q5k capability suite at HEAD records overall.total",
        ),
        (
            "receipts/headless/CAPABILITY_mlx-4bit.json",
            "overall.passed",
            "mlx-4bit capability suite at HEAD records overall.passed",
        ),
        (
            "receipts/headless/CAPABILITY_mlx-4bit.json",
            "overall.total",
            "mlx-4bit capability suite at HEAD records overall.total",
        ),
        (
            "receipts/headless/STRUCTURED_OUTPUT_PROBE.json",
            "summary.no_think.valid_json_rate",
            "structured-output probe at HEAD records no_think valid_json_rate",
        ),
        (
            "receipts/headless/STRUCTURED_OUTPUT_PROBE.json",
            "summary.schema.valid_json_rate",
            "structured-output probe at HEAD records schema valid_json_rate",
        ),
        (
            "receipts/headless/STORAGE_SELECTION.json",
            "policy",
            "storage selection receipt at HEAD records policy",
        ),
    ]
    for rel, dotted, prefix in specifics:
        if not at_head(rel):
            continue
        doc = git_json(rel)
        try:
            value = dotted_get(doc, dotted)
        except (KeyError, IndexError, TypeError):
            _add(
                out,
                Claim(
                    id=ids.next("U"),
                    kind="unknown",
                    statement=f"{rel} does not contain {dotted} at HEAD",
                    evidence_path=rel,
                    reproducing_command=py_cmd(
                        f"d = json.loads(git_text({rel!r}))",
                        "cur = d",
                        "missing = False",
                        "try:",
                        f"    for part in {json.dumps(dotted.split('.'))}:",
                        "        cur = cur[int(part)] if isinstance(cur, list) else cur[part]",
                        "except (KeyError, IndexError, TypeError):",
                        "    missing = True",
                        "assert missing, 'field unexpectedly present: ' + "
                        f"{dotted!r}",
                    ),
                    derived_from={
                        "how": "dotted lookup raised",
                        "source": rel,
                        "field": dotted,
                        "via": "git_head",
                    },
                ),
            )
            continue
        _add(
            out,
            Claim(
                id=ids.next("C"),
                kind="capability",
                statement=f"{prefix}={value!r}",
                evidence_path=rel,
                reproducing_command=py_git_json_eq(rel, dotted, value),
                derived_from={
                    "how": f"json field {dotted}",
                    "source": rel,
                    "via": "git_head",
                },
            ),
        )

    # PORT_MATRIX finding is a long string; assert a distinctive substring.
    port_rel = "receipts/headless/PORT_MATRIX.json"
    if at_head(port_rel):
        finding = git_json(port_rel).get("real_finding")
        if isinstance(finding, str) and finding:
            needle = "no standalone base '/goal' command"
            _add(
                out,
                Claim(
                    id=ids.next("C"),
                    kind="capability",
                    statement=(
                        "PORT_MATRIX at HEAD records that the Claude-side "
                        "control plane has no standalone base /goal command"
                    ),
                    evidence_path=port_rel,
                    reproducing_command=py_cmd(
                        f"d = json.loads(git_text({port_rel!r}))",
                        f"assert {needle!r} in str(d.get('real_finding'))",
                    ),
                    derived_from={
                        "how": "real_finding substring",
                        "source": port_rel,
                        "via": "git_head",
                    },
                ),
            )
            follow = git_json(port_rel).get("unresolved_follow_ups")
            if isinstance(follow, list):
                _add(
                    out,
                    Claim(
                        id=ids.next("N"),
                        kind="not_done",
                        statement=(
                            f"PORT_MATRIX at HEAD lists {len(follow)} unresolved follow-ups"
                        ),
                        evidence_path=port_rel,
                        reproducing_command=py_git_json_eq(
                            port_rel, "unresolved_follow_ups", follow
                        ),
                        derived_from={
                            "how": "json field unresolved_follow_ups",
                            "source": port_rel,
                            "via": "git_head",
                            "items": follow,
                        },
                    ),
                )

    vacuity = "receipts/headless/ACCEPTANCE_VACUITY.json"
    if at_head(vacuity):
        findings = git_json(vacuity).get("findings") or []
        _add(
            out,
            Claim(
                id=ids.next("C"),
                kind="capability",
                statement=(
                    f"ACCEPTANCE_VACUITY at HEAD records {len(findings)} findings"
                ),
                evidence_path=vacuity,
                reproducing_command=py_cmd(
                    f"d = json.loads(git_text({vacuity!r}))",
                    "cur = len(d.get('findings') or [])",
                    f"exp = {len(findings)}",
                    "assert cur == exp, (cur, exp)",
                ),
                derived_from={
                    "how": "len(findings)",
                    "source": vacuity,
                    "via": "git_head",
                },
            ),
        )

    backend = "receipts/headless/BACKEND_CAPABILITY.json"
    if at_head(backend):
        gap = git_json(backend).get("the_gap") or {}
        finding = gap.get("finding") if isinstance(gap, dict) else None
        if isinstance(finding, str) and finding:
            _add(
                out,
                Claim(
                    id=ids.next("C"),
                    kind="capability",
                    statement=f"BACKEND_CAPABILITY at HEAD records the_gap.finding={finding!r}",
                    evidence_path=backend,
                    reproducing_command=py_git_json_eq(
                        backend, "the_gap.finding", finding
                    ),
                    derived_from={
                        "how": "json field the_gap.finding",
                        "source": backend,
                        "via": "git_head",
                    },
                ),
            )


def _source_constants_at_head(out: List[Claim], ids: IdGen) -> None:
    """Constants committed at HEAD. Live dirty trees are a separate section."""
    consts: List[Tuple[str, str, str]] = [
        (
            "tools/haider/hcli/workunit.py",
            "DEFAULT_RETRY_BUDGET = 3",
            "HEAD workunit.py records DEFAULT_RETRY_BUDGET = 3",
        ),
        (
            "tools/haider/hcli/resources.py",
            "DEFAULT_DECODE_LIMIT = 1",
            "HEAD resources.py records DEFAULT_DECODE_LIMIT = 1",
        ),
        (
            "tools/haider/hcli/resources.py",
            "MEMORY_HEAVY_LIMIT = 2",
            "HEAD resources.py records MEMORY_HEAVY_LIMIT = 2",
        ),
        (
            "tools/haider/hcli/ledger.py",
            "NO_PROGRESS_THRESHOLD = 3",
            "HEAD ledger.py records NO_PROGRESS_THRESHOLD = 3",
        ),
        (
            "tools/haider/hcli/mutation.py",
            "MAX_MUTATION_OPERATIONS = 20",
            "HEAD mutation.py records MAX_MUTATION_OPERATIONS = 20",
        ),
    ]
    for rel, needle, statement in consts:
        if not at_head(rel):
            _add(
                out,
                Claim(
                    id=ids.next("U"),
                    kind="unknown",
                    statement=f"{rel} is not at git HEAD; {needle} cannot be derived",
                    evidence_path=rel,
                    reproducing_command=py_cmd(
                        f"assert git_exists({rel!r}) is False",
                    ),
                    derived_from={"how": "git cat-file -e", "source": rel},
                ),
            )
            continue
        _add(
            out,
            Claim(
                id=ids.next("C"),
                kind="capability",
                statement=statement,
                evidence_path=rel,
                reproducing_command=py_git_text_contains(rel, needle, True),
                derived_from={
                    "how": "source substring at HEAD",
                    "source": rel,
                    "via": "git_head",
                    "needle": needle,
                },
            ),
        )

    sched = "tools/haider/hcli/scheduler.py"
    if at_head(sched):
        _add(
            out,
            Claim(
                id=ids.next("N"),
                kind="not_done",
                statement=(
                    "HEAD tools/haider/hcli/scheduler.py does not define "
                    "MAX_REPAIR_DEPTH (the bound lives only in the dirty "
                    "primary working tree, if at all)"
                ),
                evidence_path=sched,
                reproducing_command=py_git_text_contains(
                    sched, "MAX_REPAIR_DEPTH", False
                ),
                derived_from={
                    "how": "source substring absent at HEAD",
                    "source": sched,
                    "via": "git_head",
                },
            ),
        )
        _add(
            out,
            Claim(
                id=ids.next("N"),
                kind="not_done",
                statement=(
                    "HEAD tools/haider/hcli/scheduler.py does not define "
                    "MAX_REPAIRS_PER_ROOT"
                ),
                evidence_path=sched,
                reproducing_command=py_git_text_contains(
                    sched, "MAX_REPAIRS_PER_ROOT", False
                ),
                derived_from={
                    "how": "source substring absent at HEAD",
                    "source": sched,
                    "via": "git_head",
                },
            ),
        )

    sm = "tools/headless/storage_manager.py"
    if (REPO / sm).is_file():
        # Import would execute argparse later; read the assignment from source.
        text = (REPO / sm).read_text(encoding="utf-8")
        m = re.search(
            r"^PROTECTED_CLASSES = \{([^}]+)\}", text, re.M | re.S
        )
        if m:
            names = sorted(re.findall(r'"([A-Z0-9_]+)"', m.group(1)))
            _add(
                out,
                Claim(
                    id=ids.next("C"),
                    kind="capability",
                    statement=(
                        "storage_manager.PROTECTED_CLASSES is "
                        f"{names} (a KEEP_LIST in prose is not a mechanism)"
                    ),
                    evidence_path=sm,
                    reproducing_command=py_cmd(
                        "import ast, pathlib, re",
                        f"t = pathlib.Path({sm!r}).read_text(encoding='utf-8')",
                        "m = re.search(r'^PROTECTED_CLASSES = \\{([^}]+)\\}', t, re.M|re.S)",
                        "assert m, 'PROTECTED_CLASSES assignment not found'",
                        "cur = sorted(re.findall(r'\"([A-Z0-9_]+)\"', m.group(1)))",
                        f"exp = json.loads({json.dumps(json.dumps(names))})",
                        "assert cur == exp, (cur, exp)",
                    ),
                    derived_from={
                        "how": "parse PROTECTED_CLASSES assignment",
                        "source": sm,
                        "via": "disk",
                    },
                ),
            )

    pl = "tools/headless/performance_ledger.py"
    if (REPO / pl).is_file():
        text = (REPO / pl).read_text(encoding="utf-8")
        m = re.search(
            r"REQUIRED_FOR_PROMOTION = \[([^\]]+)\]", text, re.S
        )
        if m:
            fields = re.findall(r'"([a-z_]+)"', m.group(1))
            _add(
                out,
                Claim(
                    id=ids.next("C"),
                    kind="capability",
                    statement=(
                        "performance_ledger.REQUIRED_FOR_PROMOTION is "
                        f"{fields}"
                    ),
                    evidence_path=pl,
                    reproducing_command=py_cmd(
                        "import pathlib, re",
                        f"t = pathlib.Path({pl!r}).read_text(encoding='utf-8')",
                        "m = re.search(r'REQUIRED_FOR_PROMOTION = \\[([^\\]]+)\\]', t, re.S)",
                        "assert m, 'REQUIRED_FOR_PROMOTION not found'",
                        "cur = re.findall(r'\"([a-z_]+)\"', m.group(1))",
                        f"exp = json.loads({json.dumps(json.dumps(fields))})",
                        "assert cur == exp, (cur, exp)",
                    ),
                    derived_from={
                        "how": "parse REQUIRED_FOR_PROMOTION assignment",
                        "source": pl,
                        "via": "disk",
                    },
                ),
            )


def _runnable_harnesses(out: List[Claim], disc: Dict[str, Any], ids: IdGen) -> None:
    """Tests that do not import tools/haider and so can run in this sparse tree."""
    runnable = [
        (
            "tools/headless/storage_manager_test.py",
            "protected artifacts are impossible to select for deletion, "
            "and the policy can still say yes to a genuine REDOWNLOADABLE",
        ),
        (
            "tools/headless/performance_ledger_test.py",
            "promotions without comparable measurements are forbidden, "
            "and a comparable measured candidate is allowed as ELIGIBLE",
        ),
    ]
    on_disk = set(disc["harnesses_on_disk"])
    for rel, statement in runnable:
        name = Path(rel).name
        if name not in on_disk:
            _add(
                out,
                Claim(
                    id=ids.next("U"),
                    kind="unknown",
                    statement=f"{rel} is not on disk in this worktree",
                    evidence_path="tools/headless",
                    reproducing_command=py_cmd(
                        "from pathlib import Path",
                        f"assert not Path({rel!r}).is_file()",
                    ),
                    derived_from={"how": "listdir miss", "source": rel},
                ),
            )
            continue
        _add(
            out,
            Claim(
                id=ids.next("C"),
                kind="capability",
                statement=statement,
                evidence_path=rel,
                reproducing_command=["python3", rel],
                derived_from={
                    "how": "harness file present on disk; command is the harness itself",
                    "source": rel,
                    "via": "disk",
                },
            ),
        )


def _live_primary(out: List[Claim], disc: Dict[str, Any], ids: IdGen) -> None:
    primary = disc.get("primary_checkout")
    if primary == "unknown" or not primary:
        _add(
            out,
            Claim(
                id=ids.next("U"),
                kind="unknown",
                statement=(
                    "primary checkout behind git-common-dir could not be derived"
                ),
                evidence_path=".git",
                reproducing_command=py_cmd(
                    "p = _primary()",
                    "assert not p.is_dir()",
                ),
                derived_from={"how": "git rev-parse --git-common-dir", "source": ".git"},
            ),
        )
        return

    _add(
        out,
        Claim(
            id=ids.next("F"),
            kind="fact",
            statement=f"git-common-dir's parent (primary checkout) is {primary}",
            evidence_path=".git",
            reproducing_command=py_cmd(
                "cur = str(_primary())",
                f"exp = {primary!r}",
                "assert cur == exp, (cur, exp)",
            ),
            derived_from={
                "how": "git rev-parse --git-common-dir",
                "source": ".git",
            },
        ),
    )

    dirty = list(disc.get("primary_dirty_hcli") or [])
    # Do not freeze the exact dirty-name list: the primary tree is a live
    # shared checkout and other writers append to `git diff --name-only`
    # between assemble and cold-read. Byte inequality of the load-bearing
    # files against THIS worktree's HEAD is the durable fact.
    for rel in (
        "tools/haider/hcli/scheduler.py",
        "tools/haider/hcli/workunit.py",
    ):
        live_p = Path(primary) / rel
        if not live_p.is_file() or not at_head(rel):
            continue
        live_b = live_p.read_bytes()
        head_b = git_bytes(rel)
        if live_b == head_b:
            continue
        _add(
            out,
            Claim(
                id=ids.next("F"),
                kind="fact",
                statement=(
                    f"live {rel} is not byte-identical to HEAD:{rel} "
                    f"(HEAD {len(head_b)} bytes, live {len(live_b)} bytes)"
                ),
                evidence_path=rel,
                reproducing_command=py_cmd(
                    f"live = disk_bytes({rel!r})",
                    f"head = git_bytes({rel!r})",
                    "assert live != head, 'live file now matches HEAD'",
                ),
                derived_from={
                    "how": "bytes(primary file) != git show HEAD",
                    "source": rel,
                    "via": "primary_checkout",
                    "head_bytes": len(head_b),
                    "live_bytes": len(live_b),
                    "dirty_snapshot_at_generation": dirty,
                },
            ),
        )

    live_sched = "tools/haider/hcli/scheduler.py"
    # Only emit live-constant claims if the primary file actually contains them.
    primary_path = Path(primary) / live_sched
    if primary_path.is_file():
        text = primary_path.read_text(encoding="utf-8")
        for needle, statement in (
            (
                "MAX_REPAIR_DEPTH = 3",
                "live primary scheduler.py records MAX_REPAIR_DEPTH = 3 "
                "(not at git HEAD)",
            ),
            (
                "MAX_REPAIRS_PER_ROOT = 6",
                "live primary scheduler.py records MAX_REPAIRS_PER_ROOT = 6 "
                "(not at git HEAD)",
            ),
        ):
            if needle in text:
                _add(
                    out,
                    Claim(
                        id=ids.next("C"),
                        kind="capability",
                        statement=statement,
                        evidence_path=live_sched,
                        reproducing_command=py_disk_text_contains(
                            live_sched, needle, True
                        ),
                        derived_from={
                            "how": "source substring on primary disk",
                            "source": live_sched,
                            "via": "primary_checkout",
                            "needle": needle,
                            "warning": (
                                "this is live uncommitted state; HEAD does not "
                                "carry the bound"
                            ),
                        },
                    ),
                )
            else:
                _add(
                    out,
                    Claim(
                        id=ids.next("N"),
                        kind="not_done",
                        statement=(
                            f"live primary {live_sched} does not contain {needle}"
                        ),
                        evidence_path=live_sched,
                        reproducing_command=py_disk_text_contains(
                            live_sched, needle, False
                        ),
                        derived_from={
                            "how": "source substring absent on primary disk",
                            "source": live_sched,
                            "via": "primary_checkout",
                        },
                    ),
                )

    repair_rel = "receipts/headless/REPAIR_HOMEOSTASIS.json"
    repair_path = Path(primary) / repair_rel
    if repair_path.is_file():
        doc = json.loads(repair_path.read_text(encoding="utf-8"))
        _add(
            out,
            Claim(
                id=ids.next("C"),
                kind="capability",
                statement=(
                    "live REPAIR_HOMEOSTASIS receipt records result="
                    f"{doc.get('result')!r} and max_repair_depth="
                    f"{doc.get('max_repair_depth')!r} "
                    "(not at git HEAD; proven by "
                    "tools/headless/hcli_repair_homeostasis_test.py on the "
                    "primary tree)"
                ),
                evidence_path=repair_rel,
                reproducing_command=py_cmd(
                    f"d = json.loads(disk_text({repair_rel!r}))",
                    f"assert d.get('result') == json.loads({json.dumps(json.dumps(doc.get('result')))})",
                    f"assert d.get('max_repair_depth') == json.loads({json.dumps(json.dumps(doc.get('max_repair_depth')))})",
                    "assert git_exists('receipts/headless/REPAIR_HOMEOSTASIS.json') is False",
                ),
                derived_from={
                    "how": "json fields result + max_repair_depth; also assert absent at HEAD",
                    "source": repair_rel,
                    "via": "primary_checkout",
                    "sha256": _sha256(repair_path),
                },
                extra={
                    "content_sha256": _sha256(repair_path),
                    "not_at_head": True,
                },
            ),
        )
        results = doc.get("results") or []
        per_root = None
        for row in results:
            if isinstance(row, dict) and "per-root COUNT cap" in str(row.get("name")):
                per_root = row
                break
        if per_root is not None:
            _add(
                out,
                Claim(
                    id=ids.next("C"),
                    kind="capability",
                    statement=(
                        "live REPAIR_HOMEOSTASIS receipt records a passing "
                        "per-root COUNT cap check: "
                        f"{per_root.get('detail')!r}"
                    ),
                    evidence_path=repair_rel,
                    reproducing_command=py_cmd(
                        f"d = json.loads(disk_text({repair_rel!r}))",
                        "hits = [r for r in (d.get('results') or []) "
                        "if isinstance(r, dict) and 'per-root COUNT cap' in str(r.get('name'))]",
                        "assert hits and hits[0].get('ok') is True, hits",
                    ),
                    derived_from={
                        "how": "results[].name contains 'per-root COUNT cap'",
                        "source": repair_rel,
                        "via": "primary_checkout",
                    },
                ),
            )

    p0_rel = "receipts/headless/P0_GATES.json"
    p0_path = Path(primary) / p0_rel
    if p0_path.is_file():
        doc = json.loads(p0_path.read_text(encoding="utf-8"))
        _add(
            out,
            Claim(
                id=ids.next("C"),
                kind="capability",
                statement=(
                    "live P0_GATES receipt records red="
                    f"{doc.get('red')!r} green={doc.get('green')!r} "
                    "(not at git HEAD)"
                ),
                evidence_path=p0_rel,
                reproducing_command=py_disk_json_eq(p0_rel, "red", doc.get("red")),
                derived_from={
                    "how": "json field red",
                    "source": p0_rel,
                    "via": "primary_checkout",
                    "green": doc.get("green"),
                },
                extra={"not_at_head": True, "content_sha256": _sha256(p0_path)},
            ),
        )

    extras = list(disc.get("primary_extra_receipts") or [])
    extra_names = [e["name"] for e in extras]
    _add(
        out,
        Claim(
            id=ids.next("F"),
            kind="fact",
            statement=(
                f"primary checkout has at least {len(extra_names)} "
                f"receipts/headless files that are not at git HEAD"
            ),
            evidence_path="receipts/headless",
            reproducing_command=py_cmd(
                "from pathlib import Path",
                f"names = json.loads({json.dumps(json.dumps(extra_names))})",
                "pr = _primary() / 'receipts' / 'headless'",
                "for n in names:",
                "    assert (pr / n).is_file(), n",
                "    assert git_exists('receipts/headless/' + n) is False, n",
            ),
            derived_from={
                "how": "listdir primary receipts/headless minus git ls-tree (subset)",
                "source": "receipts/headless",
                "via": "primary_checkout",
                "names": extra_names,
            },
        ),
    )

    extra_h = list(disc.get("primary_extra_harnesses") or [])
    _add(
        out,
        Claim(
            id=ids.next("N"),
            kind="not_done",
            statement=(
                f"{len(extra_h)} tools/headless files exist on the primary "
                f"checkout but not at git HEAD (subset check): {extra_h}"
            ),
            evidence_path="tools/headless",
            reproducing_command=py_cmd(
                "from pathlib import Path",
                f"names = json.loads({json.dumps(json.dumps(sorted(extra_h)))})",
                "ph = _primary() / 'tools' / 'headless'",
                "for n in names:",
                "    assert (ph / n).is_file(), n",
                "    assert git_exists('tools/headless/' + n) is False, n",
            ),
            derived_from={
                "how": "listdir primary tools/headless minus this worktree minus git (subset)",
                "source": "tools/headless",
                "via": "primary_checkout",
            },
        ),
    )

    goal = disc.get("primary_goal") or {}
    if goal.get("present"):
        _add(
            out,
            Claim(
                id=ids.next("F"),
                kind="fact",
                statement=(
                    "primary .hcli/GOAL.md has "
                    f"{goal.get('pending')} unchecked and "
                    f"{goal.get('checked')} checked obligation lines"
                ),
                evidence_path=".hcli/GOAL.md",
                reproducing_command=py_cmd(
                    "import re",
                    "t = disk_text('.hcli/GOAL.md')",
                    "pending = len(re.findall(r'^- \\[ \\]', t, re.M))",
                    "checked = len(re.findall(r'^- \\[[xX]\\]', t, re.M))",
                    f"assert pending == {int(goal.get('pending') or 0)}, pending",
                    f"assert checked == {int(goal.get('checked') or 0)}, checked",
                ),
                derived_from={
                    "how": "count markdown checkboxes",
                    "source": ".hcli/GOAL.md",
                    "via": "primary_checkout",
                    "sha256": goal.get("sha256"),
                },
                extra={"content_sha256": goal.get("sha256"), "not_at_head": True},
            ),
        )

    hcli = disc.get("primary_hcli") or {}
    dag = hcli.get("dag") if isinstance(hcli, dict) else None
    if isinstance(dag, dict) and dag.get("active_decode_limit") != "unknown":
        _add(
            out,
            Claim(
                id=ids.next("F"),
                kind="fact",
                statement=(
                    "primary .hcli/dag.json records active_decode_limit="
                    f"{dag.get('active_decode_limit')!r} from "
                    f"{dag.get('active_decode_limit_source')!r} with "
                    f"{dag.get('n_units')!r} units, statuses "
                    f"{dag.get('status_counts')!r}"
                ),
                evidence_path=".hcli/dag.json",
                reproducing_command=py_cmd(
                    "d = json.loads(disk_text('.hcli/dag.json'))",
                    f"assert d.get('active_decode_limit') == json.loads({json.dumps(json.dumps(dag.get('active_decode_limit')))})",
                    "units = d.get('units') or {}",
                    f"assert (len(units) if isinstance(units, dict) else None) == json.loads({json.dumps(json.dumps(dag.get('n_units')))})",
                ),
                derived_from={
                    "how": "json fields on primary .hcli/dag.json",
                    "source": ".hcli/dag.json",
                    "via": "primary_checkout",
                },
                extra={"not_at_head": True},
            ),
        )


def _blocked(out: List[Claim], disc: Dict[str, Any], ids: IdGen) -> None:
    tests = list(disc.get("haider_dependent_tests") or [])
    haider_here = bool(disc["this_worktree"]["tools_haider_hcli"])
    if tests and not haider_here:
        _add(
            out,
            Claim(
                id=ids.next("B"),
                kind="blocked",
                statement=(
                    "HCLI protected tests in this worktree cannot execute: "
                    "they import tools/haider, which is not materialized by "
                    f"the sparse checkout. Blocked tests: {tests}"
                ),
                evidence_path="tools/headless",
                reproducing_command=py_cmd(
                    "from pathlib import Path",
                    "assert Path('tools/haider/hcli').is_dir() is False",
                    f"names = json.loads({json.dumps(json.dumps(tests))})",
                    "for n in names:",
                    "    assert (Path('tools/headless') / n).is_file(), n",
                    "assert git_exists('tools/haider/hcli/scheduler.py')",
                ),
                derived_from={
                    "how": "scan *_test.py for sys.path.insert(...haider) plus Path.is_dir",
                    "source": "tools/headless",
                    "tests": tests,
                    "reason": "sparse checkout does not materialize tools/haider",
                },
                extra={"reason": "tools/haider not materialized in this worktree"},
            ),
        )

    # Native decode is fast and incoherent — a blocked production parent.
    rel = "receipts/headless/QWEN38_GRAVITY_NATIVE.json"
    if at_head(rel):
        bar = git_json(rel).get("bar")
        _add(
            out,
            Claim(
                id=ids.next("B"),
                kind="blocked",
                statement=(
                    "native Qwen3.8 Gravity parent is blocked as a production "
                    f"runtime: receipt bar={bar!r} (speed is not coherence)"
                ),
                evidence_path=rel,
                reproducing_command=py_git_json_eq(rel, "bar", bar),
                derived_from={
                    "how": "json field bar",
                    "source": rel,
                    "via": "git_head",
                    "reason": "coherence bar failed",
                },
                extra={"reason": f"bar={bar!r}"},
            ),
        )

    skip = "receipts/headless/QWEN_TUNE_SKIPPED.json"
    if at_head(skip):
        decision = git_json(skip).get("decision")
        _add(
            out,
            Claim(
                id=ids.next("B"),
                kind="blocked",
                statement=(
                    "the small Qwen tune is blocked/skipped with evidence: "
                    f"decision={decision!r}"
                ),
                evidence_path=skip,
                reproducing_command=py_git_json_eq(skip, "decision", decision),
                derived_from={
                    "how": "json field decision",
                    "source": skip,
                    "via": "git_head",
                },
                extra={"reason": f"decision={decision!r}"},
            ),
        )


def _unknown(out: List[Claim], disc: Dict[str, Any], ids: IdGen) -> None:
    # These two were emitted UNCONDITIONALLY and asserted an ABSENCE, so a
    # builder run in a sparse worktree wrote "this tree has no .hcli" into a
    # durable handoff, and the cold-read then refused it in the real repository
    # where .hcli plainly exists. A handoff is read somewhere else, later: a
    # claim that holds only in the directory that produced it is worse than no
    # claim, because its reproducing command makes it look checkable.
    # Absence is a property of a moment, so it is recorded only when observed,
    # and the command re-derives the observation rather than asserting it.
    hcli_dir = REPO / ".hcli"
    if not (hcli_dir / "GOAL.md").is_file():
        _add(
            out,
            Claim(
                id=ids.next("U"),
                kind="unknown",
                statement=(
                    "no .hcli/GOAL.md was observed in the producing worktree and "
                    "none at git HEAD; goal-ledger contents there are unknown"
                ),
                evidence_path="tools/headless/handoff_builder.py",
                reproducing_command=py_cmd(
                    "import json, sys",
                    "from pathlib import Path",
                    "present = Path(sys.argv[1] if len(sys.argv) > 1 else '.')"
                    ".joinpath('.hcli/GOAL.md').is_file()",
                    "print(json.dumps({'observed_absent_at_build': True, "
                    "'present_now': present}))",
                ),
                derived_from={
                    "how": "Path.is_file at build time; the command reports the "
                    "state now rather than asserting the state then",
                    "source": ".hcli/GOAL.md",
                    "note": (
                        "a primary checkout may still have a live GOAL.md; that "
                        "is recorded separately when discoverable"
                    ),
                },
            ),
        )
    if not hcli_dir.is_dir():
        _add(
            out,
            Claim(
                id=ids.next("U"),
                kind="unknown",
                statement=(
                    "no .hcli directory was observed in the producing worktree; "
                    "live HCLI runtime_pool, sessions and mutation lock there "
                    "are unknown"
                ),
                evidence_path="tools/headless/handoff_builder.py",
                reproducing_command=py_cmd(
                    "import json, sys",
                    "from pathlib import Path",
                    "present = Path(sys.argv[1] if len(sys.argv) > 1 else '.')"
                    ".joinpath('.hcli').is_dir()",
                    "print(json.dumps({'observed_absent_at_build': True, "
                    "'present_now': present}))",
                ),
                derived_from={"how": "Path.is_dir", "source": ".hcli"},
            ),
        )
    _add(
        out,
        Claim(
            id=ids.next("U"),
            kind="unknown",
            statement=(
                "live GPU / model-process occupancy was not probed by this "
                "builder (unknown; see MACHINE_GENOME / WRITER_AUTHORITY for "
                "the last *recorded* census, which is not live)"
            ),
            evidence_path="receipts/headless/MACHINE_GENOME.json",
            reproducing_command=py_cmd(
                "d = json.loads(git_text('receipts/headless/MACHINE_GENOME.json'))",
                "assert 'generated_at' in d",
                "assert d.get('schema') == 'hawking.headless.machine_genome.v1'",
            ),
            derived_from={
                "how": "builder did not run machine_probe.py; census is point-in-time",
                "source": "receipts/headless/MACHINE_GENOME.json",
                "via": "git_head",
            },
        ),
    )

    # DISK_TRUTH is a census, not live. Record genomes it said were absent,
    # then note whether those paths exist at HEAD *now*.
    dt = "receipts/headless/DISK_TRUTH.json"
    if at_head(dt):
        genomes = git_json(dt).get("genomes") or {}
        if isinstance(genomes, dict):
            for name, info in genomes.items():
                if not isinstance(info, dict):
                    continue
                present = info.get("present")
                if present is True:
                    continue
                searched = info.get("searched") or []
                rels = []
                for s in searched:
                    # take the repo-relative tail if it looks like one
                    marker = "receipts/headless/"
                    if marker in str(s):
                        rels.append(str(s)[str(s).index(marker) :])
                _add(
                    out,
                    Claim(
                        id=ids.next("U"),
                        kind="unknown",
                        statement=(
                            f"DISK_TRUTH census recorded {name} present=false "
                            f"at {git_json(dt).get('generated_at')!r}; "
                            "that census is not a live view"
                        ),
                        evidence_path=dt,
                        reproducing_command=py_cmd(
                            f"d = json.loads(git_text({dt!r}))",
                            f"g = (d.get('genomes') or {{}}).get({name!r}) or {{}}",
                            "assert g.get('present') is False, g",
                        ),
                        derived_from={
                            "how": f"genomes.{name}.present",
                            "source": dt,
                            "via": "git_head",
                            "searched": searched,
                        },
                    ),
                )


def _not_done(out: List[Claim], disc: Dict[str, Any], ids: IdGen) -> None:
    skip = "receipts/headless/QWEN_TUNE_SKIPPED.json"
    if at_head(skip):
        missing = git_json(skip).get("what_was_not_done")
        if isinstance(missing, str) and missing:
            _add(
                out,
                Claim(
                    id=ids.next("N"),
                    kind="not_done",
                    statement=f"Qwen small-tune what_was_not_done: {missing}",
                    evidence_path=skip,
                    reproducing_command=py_git_json_eq(
                        skip, "what_was_not_done", missing
                    ),
                    derived_from={
                        "how": "json field what_was_not_done",
                        "source": skip,
                        "via": "git_head",
                    },
                ),
            )
        revisit = git_json(skip).get("revisit_when") or []
        if isinstance(revisit, list) and revisit:
            _add(
                out,
                Claim(
                    id=ids.next("N"),
                    kind="not_done",
                    statement=f"Qwen small-tune revisit_when={revisit!r}",
                    evidence_path=skip,
                    reproducing_command=py_git_json_eq(
                        skip, "revisit_when", revisit
                    ),
                    derived_from={
                        "how": "json field revisit_when",
                        "source": skip,
                        "via": "git_head",
                    },
                ),
            )

    # Named harness from the campaign context pack that is not at HEAD.
    missing_test = "tools/headless/hcli_command_ingress_test.py"
    _add(
        out,
        Claim(
            id=ids.next("N"),
            kind="not_done",
            statement=(
                f"{missing_test} is not at git HEAD"
                + (
                    " (it exists on the primary checkout as uncommitted work)"
                    if missing_test.split("/")[-1]
                    in (disc.get("primary_extra_harnesses") or [])
                    else ""
                )
            ),
            evidence_path="tools/headless",
            reproducing_command=py_cmd(
                f"assert git_exists({missing_test!r}) is False",
            ),
            derived_from={
                "how": "git cat-file -e",
                "source": missing_test,
            },
        ),
    )

    repair_test = "tools/headless/hcli_repair_homeostasis_test.py"
    _add(
        out,
        Claim(
            id=ids.next("N"),
            kind="not_done",
            statement=(
                f"{repair_test} is not at git HEAD"
                + (
                    " (it exists on the primary checkout as uncommitted work)"
                    if Path(repair_test).name
                    in (disc.get("primary_extra_harnesses") or [])
                    else ""
                )
            ),
            evidence_path="tools/headless",
            reproducing_command=py_cmd(
                f"assert git_exists({repair_test!r}) is False",
            ),
            derived_from={
                "how": "git cat-file -e",
                "source": repair_test,
            },
        ),
    )


def _watched_fail(out: List[Claim], disc: Dict[str, Any], ids: IdGen) -> None:
    # Receipts that *are* the record of a failure, not a memory of one.
    qwen = "receipts/headless/QWEN38_GRAVITY_NATIVE.json"
    if at_head(qwen):
        reasons = dotted_get(git_json(qwen), "three_part_bar.coherence_reasons")
        _add(
            out,
            Claim(
                id=ids.next("W"),
                kind="watched_fail",
                statement=(
                    "native Qwen3.8 decode produced collapsed output: "
                    f"coherence_reasons={reasons!r}"
                ),
                evidence_path=qwen,
                reproducing_command=py_git_json_eq(
                    qwen, "three_part_bar.coherence_reasons", reasons
                ),
                derived_from={
                    "how": "three_part_bar.coherence_reasons",
                    "source": qwen,
                    "via": "git_head",
                },
            ),
        )

    g003 = "receipts/headless/G003_ROOT_CAUSE.json"
    if at_head(g003):
        v = git_json(g003).get("verdict")
        _add(
            out,
            Claim(
                id=ids.next("W"),
                kind="watched_fail",
                statement=f"G003 was root-caused as {v!r}",
                evidence_path=g003,
                reproducing_command=py_git_json_eq(g003, "verdict", v),
                derived_from={
                    "how": "json field verdict",
                    "source": g003,
                    "via": "git_head",
                },
            ),
        )

    vacuity = "receipts/headless/ACCEPTANCE_VACUITY.json"
    if at_head(vacuity):
        titles = [
            f.get("title")
            for f in (git_json(vacuity).get("findings") or [])
            if isinstance(f, dict)
        ]
        _add(
            out,
            Claim(
                id=ids.next("W"),
                kind="watched_fail",
                statement=(
                    "ACCEPTANCE_VACUITY recorded these finding titles: "
                    f"{titles!r}"
                ),
                evidence_path=vacuity,
                reproducing_command=py_cmd(
                    f"d = json.loads(git_text({vacuity!r}))",
                    "cur = [f.get('title') for f in (d.get('findings') or []) if isinstance(f, dict)]",
                    f"exp = json.loads({json.dumps(json.dumps(titles))})",
                    "assert cur == exp, (cur, exp)",
                ),
                derived_from={
                    "how": "findings[].title",
                    "source": vacuity,
                    "via": "git_head",
                },
            ),
        )

    writer = "receipts/headless/WRITER_AUTHORITY.json"
    if at_head(writer):
        corr = git_json(writer).get("corrections") or []
        _add(
            out,
            Claim(
                id=ids.next("W"),
                kind="watched_fail",
                statement=(
                    "WRITER_AUTHORITY records "
                    f"{len(corr)} corrections, including a misattributed kill "
                    "of an operator-owned llama-server on port 8080"
                ),
                evidence_path=writer,
                reproducing_command=py_cmd(
                    f"d = json.loads(git_text({writer!r}))",
                    "cur = len(d.get('corrections') or [])",
                    f"exp = {len(corr)}",
                    "assert cur == exp, (cur, exp)",
                    "blob = json.dumps(d.get('corrections') or [])",
                    "assert 'port 8080' in blob or 'port-8080' in blob",
                ),
                derived_from={
                    "how": "len(corrections) plus substring",
                    "source": writer,
                    "via": "git_head",
                },
            ),
        )

    # The repair-tree blow-up is recorded in the (uncommitted) harness docstring
    # and the live receipt's negative-control row.
    repair_test = "tools/headless/hcli_repair_homeostasis_test.py"
    primary = disc.get("primary_checkout")
    if (
        isinstance(primary, str)
        and primary != "unknown"
        and (Path(primary) / repair_test).is_file()
    ):
        _add(
            out,
            Claim(
                id=ids.next("W"),
                kind="watched_fail",
                statement=(
                    "hcli_repair_homeostasis_test.py records an observed "
                    "unbounded repair tree (grokfail.repair.1.repair.1...) "
                    "from a dead backend; the file is not at git HEAD"
                ),
                evidence_path=repair_test,
                reproducing_command=py_cmd(
                    f"t = disk_text({repair_test!r})",
                    "assert 'grokfail.repair.1' in t",
                    "assert 'kept going' in t",
                    f"assert git_exists({repair_test!r}) is False",
                ),
                derived_from={
                    "how": "docstring substrings on primary disk",
                    "source": repair_test,
                    "via": "primary_checkout",
                },
            ),
        )

    # The builder's own refusal is a watched failure of the *inadmissible claim*.
    _add(
        out,
        Claim(
            id=ids.next("W"),
            kind="watched_fail",
            statement=(
                "handoff_builder.py refuses a claim with no reproducing_command "
                "(demonstrated by --refuse-only; a validator that cannot reject "
                "validates nothing)"
            ),
            evidence_path="tools/headless/handoff_builder.py",
            reproducing_command=[
                "python3",
                "tools/headless/handoff_builder.py",
                "--refuse-only",
            ],
            derived_from={
                "how": "running this file with --refuse-only",
                "source": "tools/headless/handoff_builder.py",
            },
        ),
    )


def assemble(disc: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    disc = disc if disc is not None else discover()
    ids = IdGen()
    claims = build_claims(disc, ids)
    by_kind: Dict[str, int] = {}
    for c in claims:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    watched = [c.as_dict() for c in claims if c.kind == "watched_fail"]
    doc = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "tools/headless/handoff_builder.py",
        "repo": {
            "toplevel": disc["repo"],
            "head": disc["head"],
            "branch": disc["branch"],
            "dirty_this_worktree": disc["dirty_this_worktree"],
            "sparse_checkout": disc["sparse_checkout"],
            "sparse_roots": disc["sparse_roots"],
            "primary_checkout": disc["primary_checkout"],
            "primary_head": disc["primary_head"],
            "primary_same_head": disc["primary_same_head"],
            "this_worktree": disc["this_worktree"],
        },
        "sources": {
            "git_receipts_headless": disc["git_receipts"],
            "git_receipts_other": disc["git_receipts_other"],
            "harnesses_on_disk": disc["harnesses_on_disk"],
            "haider_dependent_tests": disc["haider_dependent_tests"],
            "primary_extra_receipts": disc["primary_extra_receipts"],
            "primary_extra_harnesses": disc["primary_extra_harnesses"],
            "primary_dirty_hcli": disc["primary_dirty_hcli"],
            "primary_hcli": {
                k: v
                for k, v in (disc.get("primary_hcli") or {}).items()
                if k != "path" or True
            },
            "primary_goal": {
                k: v
                for k, v in (disc.get("primary_goal") or {}).items()
                if k != "text"
            },
            "goal_ledger_this_worktree": "unknown",
        },
        "claims": [c.as_dict() for c in claims],
        "what_i_watched_fail": watched,
        "counts": {
            "claims": len(claims),
            "by_kind": by_kind,
        },
        "notes": {
            "unknown_means": (
                "the builder looked and could not derive the fact from a file"
            ),
            "blocked_means": (
                "the work is identified and currently cannot proceed, with a reason"
            ),
            "not_done_means": (
                "the work is identified as absent from the durable (HEAD) record"
            ),
            "live_uncommitted": (
                "claims whose derived_from.via is primary_checkout are live "
                "working-tree state, not git HEAD. A cold reader with only a "
                "clone of HEAD will see those commands go UNSUPPORTED — that "
                "is the correct signal, not a missing default."
            ),
        },
    }
    return doc


def _short_cmd(row: Dict[str, Any]) -> str:
    cmd = row.get("reproducing_command")
    if isinstance(cmd, list) and len(cmd) >= 2 and cmd[0] == "python3" and cmd[1] == "-c":
        return f"python3 -c <assert {row.get('evidence_path')}>"
    if isinstance(cmd, list):
        return shlex.join(cmd)
    return str(cmd or "")


def print_watched_fail(doc: Dict[str, Any]) -> None:
    print()
    print("## WHAT I WATCHED FAIL")
    rows = doc.get("what_i_watched_fail") or []
    if not rows:
        print("(none derived)")
        return
    for row in rows:
        print(f"- {row.get('id')}: {row.get('statement')}")
        print(f"    evidence: {row.get('evidence_path')}")
        print(f"    reproduce: {_short_cmd(row)}")


def write_staging(doc: Dict[str, Any], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, dest)
    return dest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="",
        help=(
            "write the handoff JSON here. Default is the staging path "
            f"{STAGING_REL} (never the durable {FINAL_REL} unless you pass it)"
        ),
    )
    parser.add_argument(
        "--refuse-only",
        action="store_true",
        help="demonstrate claim refusal and exit; do not write a handoff",
    )
    parser.add_argument(
        "--allow-final",
        action="store_true",
        help=(
            f"permit --output {FINAL_REL}. Without this flag the builder "
            "refuses to overwrite the durable name; the cold-read test is "
            "the thing that promotes staging -> final."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rc, lines = demonstrate_refusal()
    if args.refuse_only:
        return rc
    if rc != 0:
        print("refusal demo did not fire; refusing to emit a handoff", file=sys.stderr)
        return rc

    dest = Path(args.output) if args.output else REPO / STAGING_REL
    if not dest.is_absolute():
        dest = (REPO / dest).resolve()
    final = (REPO / FINAL_REL).resolve()
    if dest == final and not args.allow_final:
        print(
            f"refusing to write {FINAL_REL} directly; write staging and let "
            "handoff_cold_read_test.py promote it after every claim is SUPPORTED",
            file=sys.stderr,
        )
        return 2

    disc = discover()
    doc = assemble(disc)
    doc["refusal_demo"] = {
        "fired": True,
        "lines": lines,
        "reproduce": [
            "python3",
            "tools/headless/handoff_builder.py",
            "--refuse-only",
        ],
    }
    path = write_staging(doc, dest)
    print(f"wrote staging: {path}")
    by_kind = ((doc.get("counts") or {}).get("by_kind") or {})
    print(
        "counts: "
        + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items()))
        + f" total={doc.get('counts', {}).get('claims')}"
    )
    print_watched_fail(doc)
    print()
    print(
        "cold-read next: python3 tools/headless/handoff_cold_read_test.py "
        f"--handoff {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
