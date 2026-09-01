"""G021 audit: the canonical history is clean, and the exposure is two branches.

S016 recorded that main and odyssey-i were rewritten, verified tree-identical and
force-pushed. This re-derives that from DISK rather than believing the ledger,
and then asks the question the amendment deferred: what still carries agent
attribution, and how much of it can anyone else see?

    CANONICAL LINES, verified against disk
        main         3951 commits   0 tool identities   0 trailers
        odyssey-i    4113 commits   0 tool identities   0 trailers
        origin/main  3951 commits   0 tool identities   0 trailers
        local main == origin/main, so the push happened

    ZERO CLAUDE OR ANTHROPIC ATTRIBUTION ANYWHERE, in any ref, ever. The standing
    rule was already being followed. The 3 "generated with" matches are PROSE -
    "all 800 records regenerated with different hashes", "profile JSON
    regenerated with the new candidate ID" - and stripping those would destroy
    meaning rather than attribution.

    WHAT REMAINS         203 refs
        published            2   arc/300k-integration
                                 grok/wave0-integrate-20260814-210202
        local heads        188
        other refs          13   *-landing/candidate

    The 4 `Co-authored-by: aider (openai/qwen3.8-27b-abliterated)` trailers are on
    unmerged grok/* lane branches only. Neither published branch carries one;
    both carry 8 tool-identity authors apiece.

THE EXPOSURE IS TWO BRANCHES. Everything else is unpublished local scratch that
no one but this machine can read. Both published branches are UNMERGED - 232 and
235 commits ahead of main - so neither is a stale mirror of work already landed,
and deleting them would discard commits that exist nowhere else.

This module AUDITS. It does not rewrite and it does not push: a history rewrite of
a published branch is a force-push, and that is the user's call, not a tool's.

    python3 tools/future/attribution_audit.py --build
"""
from __future__ import annotations

import argparse
import json
import functools
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/attribution_audit.py"
RECEIPT_NAME = "ATTRIBUTION_AUDIT.json"

CANONICAL = ("main", "odyssey-i", "origin/main")
TOOL_IDENTITIES = ("swg", "builder", "Hawking Builder", "B-RT5 Independent Auditor")
TRAILER = re.compile(r"^Co-authored-by:", re.I | re.M)
# Line-anchored. A body that merely CONTAINS the words is prose, not a footer.
FOOTER = re.compile(r"^(\N{ROBOT FACE} )?Generated with", re.M)
CLAUDE = re.compile(r"claude|anthropic", re.I)


class AuditRefused(RuntimeError):
    """git did not answer, so no claim is made about what it would have said."""


def _git(*args: str) -> str:
    r = subprocess.run(["git", "--no-optional-locks", *args],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise AuditRefused(f"git {' '.join(args)} failed: {r.stderr.strip()[:200]}")
    return r.stdout


@functools.lru_cache(maxsize=None)
def _scan(ref: str) -> dict[str, int]:
    authors = _git("log", ref, "--format=%an").splitlines()
    bodies = _git("log", ref, "--format=%B")
    return {
        "commits": len(authors),
        "tool_identities": sum(1 for a in authors if a in TOOL_IDENTITIES),
        "trailers": len(TRAILER.findall(bodies)),
        "generated_with_footers": len(FOOTER.findall(bodies)),
    }


@functools.lru_cache(maxsize=1)
def canonical() -> dict[str, Any]:
    out = {ref: _scan(ref) for ref in CANONICAL}
    local = _git("rev-parse", "main").strip()
    remote = _git("rev-parse", "origin/main").strip()
    out["main_is_pushed"] = local == remote
    out["clean"] = all(
        v["tool_identities"] == 0 and v["trailers"] == 0
        and v["generated_with_footers"] == 0
        for k, v in out.items() if isinstance(v, dict)
    )
    return out


@functools.lru_cache(maxsize=1)
def no_claude_attribution_anywhere() -> dict[str, Any]:
    """The standing rule, checked across every ref rather than assumed."""
    ids = _git("log", "--all", "--format=%an <%ae>%n%cn <%ce>").splitlines()
    hits = sorted({i for i in ids if CLAUDE.search(i)})
    bodies = _git("log", "--all", "--format=%B")
    return {
        "claude_or_anthropic_identities": hits,
        "n_identities": len(hits),
        "generated_with_footers_anywhere": len(FOOTER.findall(bodies)),
        "verdict": "NONE" if not hits and not FOOTER.search(bodies) else "PRESENT",
        "prose_is_not_attribution": (
            "bodies containing the words 'regenerated with' match a naive grep "
            "and are prose about hashes and profile JSON. The footer pattern is "
            "LINE-ANCHORED so it cannot count them."
        ),
    }


def _published() -> set[str]:
    out = set()
    r = subprocess.run(["git", "--no-optional-locks", "ls-remote", "--heads", "origin"],
                       cwd=REPO, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise AuditRefused(
            "ls-remote failed, so PUBLISHED cannot be determined. Reporting "
            "every ref as unpublished would understate the exposure, so this "
            "refuses instead."
        )
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].startswith("refs/heads/"):
            out.add(parts[1][len("refs/heads/"):])
    return out


@functools.lru_cache(maxsize=1)
def _published_cached() -> frozenset[str]:
    return frozenset(_published())


@functools.lru_cache(maxsize=1)
def remaining() -> dict[str, Any]:
    """Walks every ref, so it is cached: 203 refs times a git log each is minutes,
    and nothing in one process changes between calls."""
    pub = _published_cached()
    local_heads = {
        r[len("refs/heads/"):]
        for r in _git("for-each-ref", "--format=%(refname)", "refs/heads/").splitlines()
    }
    dirty_pub, dirty_local, dirty_other = [], [], []
    for ref in _git("for-each-ref", "--format=%(refname)").splitlines():
        try:
            s = _scan(ref)
        except AuditRefused:
            continue
        if s["tool_identities"] == 0 and s["trailers"] == 0:
            continue
        row = {"ref": ref, **s}
        if ref.startswith("refs/heads/"):
            name = ref[len("refs/heads/"):]
            (dirty_pub if name in pub else dirty_local).append(row)
        elif ref.startswith("refs/remotes/origin/"):
            # A published branch with NO local head is invisible to the
            # refs/heads/ pass, and skipping it here UNDERCOUNTS the exposure -
            # which this audit did on its first run, reporting 1 where the
            # answer is 2. Only skip when a local head actually covers it.
            name = ref[len("refs/remotes/origin/"):]
            if name in local_heads or name == "HEAD":
                continue
            if name in pub:
                dirty_pub.append(row)
            else:
                dirty_other.append(row)
        else:
            dirty_other.append(row)
    for row in dirty_pub:
        name = (row["ref"][len("refs/heads/"):] if row["ref"].startswith("refs/heads/")
                else row["ref"][len("refs/remotes/origin/"):])
        row["merged_into_main"] = subprocess.run(
            ["git", "merge-base", "--is-ancestor", f"origin/{name}", "main"],
            cwd=REPO, capture_output=True).returncode == 0
        row["commits_ahead_of_main"] = len(
            _git("rev-list", f"main..origin/{name}").splitlines())
    return {
        "n_published_dirty": len(dirty_pub),
        "published_dirty": dirty_pub,
        "n_local_only_dirty": len(dirty_local),
        "n_other_refs_dirty": len(dirty_other),
        "reading": (
            f"{len(dirty_pub)} branches anyone else can read still carry agent "
            f"attribution. The other {len(dirty_local) + len(dirty_other)} refs "
            "are unpublished local scratch."
        ),
    }


def what_this_does_not_do() -> dict[str, Any]:
    return {
        "does_not_rewrite": True,
        "does_not_push": True,
        "why": (
            "rewriting a PUBLISHED branch is a force-push. Both remaining "
            "branches are UNMERGED - ahead of main - so they are not stale "
            "mirrors of landed work and deleting them would discard commits "
            "that exist nowhere else. Which of rewrite / delete / leave is "
            "correct is the user's call, and this obligation's own words are "
            "'Nothing may be lost'."
        ),
        "preconditions_for_any_rewrite": [
            "a bundle of every ref, verified with git bundle verify, taken first",
            "--prune-empty=never --prune-degenerate=never, because the first "
            "S016 attempt lost 970 commits to filter-repo's defaults",
            "tree identity checked per branch, not just commit counts",
        ],
    }


def build() -> dict[str, Any]:
    c = canonical()
    r = remaining()
    return {
        "obligation": "G021",
        "question": "what in this repository's history still carries agent attribution, and who can see it?",
        "verdict": (
            "CANONICAL_CLEAN_EXPOSURE_IS_"
            f"{r['n_published_dirty']}_PUBLISHED_BRANCHES"
        ),
        "canonical": c,
        "no_claude_attribution_anywhere": no_claude_attribution_anywhere(),
        "remaining": r,
        "what_this_does_not_do": what_this_does_not_do(),
        "evidence_class": "DERIVED_FROM_GIT",
        "method": (
            "every ref walked with git log; identities matched exactly against "
            "the four known tool names; trailers and footers matched "
            "LINE-ANCHORED so prose cannot inflate the count"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(REPO / "receipts" / "future" / RECEIPT_NAME,
                            doc, RECORDED_BY))
        return 0
    print(json.dumps({k: doc[k] for k in
                      ("verdict", "canonical", "no_claude_attribution_anywhere",
                       "remaining")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
