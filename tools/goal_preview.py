#!/usr/bin/env python3
"""Show what the goal compiler will actually hand the model.

A goal file is not delivered verbatim. GoalCompiler splits it into SENTENCES and
promotes any sentence containing "do not", "must", "never", "only", "without"
and friends into an INVARIANT; the OBJECTIVE is chosen from what is left.

A carefully written, heavily-qualified goal therefore inverts. Measured: a goal
whose task was "make the whole-file branch report total_lines" reached the model
as

    OBJECTIVE: obligation=G003 Do not edit it.
    INVARIANTS: Do not create it. / Do not edit it. / Do not create any file.
    EVIDENCE_PATHS: (none)

with the exact source block, supplied precisely so the model would not have to
guess, shredded into sentences and dropped. The model then answered with an
empty operation, which is exactly what it had been told to do.

Run this before spending a model call on a goal. It costs milliseconds; the call
it replaces costs minutes.
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: goal_preview.py <goal-file>")
        return 2
    from hcli.goal import GoalCompiler

    text = pathlib.Path(sys.argv[1]).read_text()
    c = GoalCompiler()
    sentences = c._sentences(text)
    invariants, eligible = [], []
    for s in sentences:
        (invariants if any(m in s.lower() for m in c._INVARIANT_MARKERS)
         else eligible).append(s)

    print(f"sentences        {len(sentences)}")
    print(f"objective-eligible {len(eligible)}")
    print(f"invariants       {len(invariants)}")
    print()
    if eligible:
        print(f"OBJECTIVE will come from:\n  {eligible[0][:110]}")
    else:
        print("NO objective-eligible sentence. Every sentence contains an "
              "invariant marker, so the objective will be a negation.")
    print()
    if invariants:
        print("promoted to INVARIANTS (not the task):")
        for s in invariants[:8]:
            print(f"  - {s[:100]}")
    print()
    files = c._referenced_files(text)
    print(f"referenced files {files if files else '(none) -- no evidence will be inlined'}")
    print()
    bad = [m for m in c._INVARIANT_MARKERS if m in (eligible[0].lower() if eligible else "")]
    if bad:
        print(f"WARNING: the first sentence contains {bad}; it may be taken as "
              f"an invariant instead of the objective.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
