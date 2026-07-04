# CONDENSE_DOCS_REVIEW.md · staged doc deletions for the grader · condense/run-20260703

Docs track is gate-blind: merge only, preserve all unique content, never auto-delete. Real deletion only for
regenerable or byte-identical duplicates, and only staged here for a human. House style: no em or en dashes
· middot separators.

## Staged deletions

NONE. The docs tree is already at its consolidation fixpoint:

- Four prior passes already pruned 280+ tracked markdown files into git history, each recorded in
  `docs/ARCHIVE_INDEX.md` with a restore command and an annotated tag (`pre-hawking-rename`,
  `pre-consolidation-2026-07-01`). Passes: 2026-06-20 (225 files), 2026-06-28 (49), the campaign logs (18),
  the hide-bible archive (20), and the superseded plans set (10).
- Zero byte-identical markdown duplicates exist in the owned tree (checked by sha over every non-archive
  `.md`). There is nothing regenerable-or-duplicate to stage.
- The 38 surviving `docs/plans/*.md` are the curated set. Merging any two would require judging which prose
  is redundant, i.e. content work, not a footprint-safe dedupe. Deferred: no merge is safe to do blind.

## Out-of-footprint content note (not a deletion, flagged for a content pass)

`docs/ARCHIVE_INDEX.md` records known-stale cross-references left as-is by the prior passes:
`BASELINES.md`, `FAILURES.md`, `WATCHLIST.md`, and `GO.md` cite
`docs/plans/studio_maximization_2026_06_27.md`, which was archived (retrieve with
`git show 52f64684^:docs/plans/studio_maximization_2026_06_27.md`); `docs/plans/hide_handoff_2026_06_28.md`
cites a `docs/hide-bible/MASTER_PLAN.md` path that never existed. These are dangling-reference bugs. Fixing
them removes no footprint (so it is out of scope for /goal condense) but would improve reviewability. If you
want it, it is a small, separate content commit, not a condensation.

## Verdict

Docs footprint delta this run: 0. Nothing to approve here. The doc consolidation the operator remembers
wanting was already executed across the four passes above.
