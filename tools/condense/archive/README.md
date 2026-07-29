# tools/condense/archive — emptied by lane F1

Controller bodies that previously lived here were either:

1. **Restored** to `tools/condense/<module>.py` because tests or live code still import them
   (relocation, not condensation), or
2. **Deleted** because the campaign engine supersedes them. Specs under
   `tools/condense/engine/specs/` keep the fixture, receipt, reproduction command, and reopen
   condition. Git history holds the code.

**Tag (prepared):** `pre-controller-retirement-20260728` → commit `53435e75f8e80f2b1351f5da0fd4dbea0449f567` (tag object `e523721bf7a3d4e7453bf3634aa94bc2b0618365`).

Restore any deleted module:

```
git checkout pre-controller-retirement-20260728 -- tools/condense/archive/<module>.py
# equivalent by commit if the annotated tag is not yet on this clone:
git checkout 53435e75f8e80f2b1351f5da0fd4dbea0449f567 -- tools/condense/archive/<module>.py
```

See `docs/ARCHIVE_INDEX_2.md` section F1 for the full inventory.
