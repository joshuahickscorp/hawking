#!/usr/bin/env python3
"""`git commit` takes the index, and `git apply --3way` writes to it.

The legacy Odyssey driver was retired for ending each window with a bare
`git commit`, which sweeps whatever another author left staged into a commit
under the driver's message. Within the same session the integration authority
did the same thing to itself: `git apply --3way` stages its result, so a later
`git add <explicit paths> && git commit` carried an unrelated lane's changes
under the wrong message.

    fb4240dad "the resident supervisor is a loop, not a cycle"
      also carries tools/future/autonomy_run.py and test_autonomy_run.py,
      which are the m2 torture-wiring lane, not the supervisor.

Content and authorship are correct; the grouping is not. The commit is recorded
here rather than rewritten, because rewriting a mid-stack commit to fix a
message is a worse trade than saying what happened.

The preventive rule is one line: commit with an explicit pathspec, which
ignores the index entirely.

    git add <paths> && git commit -F <msg> -- <paths>
        the pathspec still limits the commit even when other paths are staged.
        `git add` is required first for UNTRACKED files: a pathspec commit only
        matches paths git already knows, and errors with
        "pathspec ... did not match any file(s) known to git" otherwise.

    git commit -F <msg>
        takes the index, whatever is in it.

    python3 tools/future/index_provenance.py --check
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, git  # noqa: E402, require_known_flags

RECEIPT = REPO / "receipts" / "future" / "INDEX_PROVENANCE.json"

STAGING_SIDE_EFFECTS = (
    "git apply --3way        — stages its result, including on clean application",
    "git apply --index       — stages by definition",
    "git stash pop           — can leave conflicted paths staged",
    "git cherry-pick / merge — stage on conflict",
)

KNOWN_MIXED = (
    {
        "commit": "fb4240dad",
        "message": "feat(future): the resident supervisor is a loop, not a cycle",
        "belongs": ["tools/future/resident_supervisor.py",
                    "tools/future/test_resident_supervisor.py",
                    "receipts/future/RESIDENT_SUPERVISOR.json"],
        "swept_in": ["tools/future/autonomy_run.py",
                     "tools/future/test_autonomy_run.py"],
        "swept_from": "the m2 torture-wiring lane, applied with --3way minutes earlier",
        "harm": "provenance only — content is correct and authorship is correct",
        "not_rewritten_because": "rewriting a mid-stack commit to correct a "
                                 "message costs more than recording it",
    },
)


def staged_now() -> list[str]:
    return [p for p in git("diff", "--cached", "--name-only").splitlines() if p]


def check() -> dict[str, Any]:
    staged = staged_now()
    return {
        "staged_paths": staged,
        "n_staged": len(staged),
        "safe_to_bare_commit": not staged,
        "advice": (
            "index is clean; a bare commit would take nothing unexpected"
            if not staged
            else "index is NOT clean. Use `git commit -- <paths>` so these are "
                 "not swept in, or stage deliberately."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "schema": "hawking.future.index_provenance.v1",
        "version": 1,
        "recorded_by": "tools/future/index_provenance.py",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "law": (
            "A bare `git commit` commits the INDEX, not a path list. Any command "
            "that stages as a side effect can therefore put someone else's work "
            "under your message. Commit with an explicit pathspec."
        ),
        "commands_that_stage_as_a_side_effect": list(STAGING_SIDE_EFFECTS),
        "safe_form": "git add <paths> && git commit -F <msgfile> -- <paths>",
        "pathspec_caveat": (
            "A pathspec commit only matches paths git already tracks. An "
            "untracked new file must be `git add`ed first, and the pathspec on "
            "the commit still protects it from sweeping anything else in."
        ),
        "unsafe_form": "git commit -F <msgfile>",
        "known_mixed_commits": list(KNOWN_MIXED),
        "same_defect_as": "receipts/future/LEGACY_ODYSSEY_DRIVER_RETIREMENT.json "
                          "— the driver was retired for exactly this, and the "
                          "integration authority then did it once itself",
        "current": check(),
        "claim_boundary": (
            "Records one known mixed commit and the mechanism. It does not scan "
            "history for others; a full audit would need per-commit review of "
            "which paths belong to which message, which is judgement, not a "
            "check."
        ),
    }


def record() -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n")
    return RECEIPT


if __name__ == "__main__":
    from _common import require_known_flags
    require_known_flags(["--build", "--check", "--record"])
    if "--check" in sys.argv:
        print(json.dumps(check(), indent=1))
    elif "--record" in sys.argv:
        print(f"wrote {record()}")
    else:
        print(json.dumps(build(), indent=1))
