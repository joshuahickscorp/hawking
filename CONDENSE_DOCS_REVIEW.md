# CONDENSE_DOCS_REVIEW.md · staged doc deletions for the grader · condense/run-20260703

Docs track is gate-blind: merge only, preserve all unique content, never auto-delete. Real deletion only for
regenerable or byte-identical duplicates, and only staged here for a human. House style: no em or en dashes
· middot separators.

## Staged deletions

REVIEW ONLY, not auto-deleted:

- `docs/plans/studio_pinned_requirements.txt` is byte-identical to
  `scaffolding/requirements.freeze.txt`. Do not delete blindly: `tools/condense/preflight.py` currently
  points operators at the docs/plans path, while `receipts/README.md` points at the scaffolding path. A human
  can choose the canonical location and update references in a separate content commit.

The markdown docs tree is already at its consolidation fixpoint:

- Four prior passes already pruned 280+ tracked markdown files into git history, each recorded in
  `docs/ARCHIVE_INDEX.md` with a restore command and an annotated tag (`pre-hawking-rename`,
  `pre-consolidation-2026-07-01`). Passes: 2026-06-20 (225 files), 2026-06-28 (49), the campaign logs (18),
  the hide-bible archive (20), and the superseded plans set (10).
- Zero byte-identical markdown duplicates exist in the owned tree (checked by sha over every non-archive
  `.md`). There is no markdown deletion to stage.
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

## Merged-tool references (from the code-track consolidation)

The tool-file consolidation (52 -> 40 in tools/condense) renamed seven families to subcommand-tools. Doc
references updated (count-asserted) in the two ACTIVE canonical docs: `docs/plans/STUDIO_GO.md` (7 refs:
kv_frontier/kv_hybrid/expert_sensitivity/expert_cache_policy/subbit_measure/subbit_admm/doctor_registry ->
their `kv.py`/`expert.py`/`subbit.py`/`doctor.py` subcommand forms) and
`docs/plans/quintessential_engine_2026_06_29.md` (doctor_registry).

Continuation update: `docs/plans/parameter_sweep_pipeline.md` now references `sweep.py render` instead of
the deleted `sweep_render.py` renderer (2 refs). This is an active command-reference update, not a doc
deletion.

Still carrying stale PROSE mentions of old tool names (historical/archival, out of footprint scope, listed
for a future content pass, NOT a deletion): `doctor_maximization_plan.md`, `condense_master_plan_2026_06_22.md`,
`M1ULTRA_POTENTIAL_AUDIT.md`, `parameter_sweep_pipeline.md`, `hawking_capability_frontier_2026_06_28.md`,
`doctor_capability_and_speed_roadmap.md`, `condense_autopilot_2026_06_27.md`, `hawking_handoff_2026_06_28.md`.
None is a runnable command (0 broken command references), so nothing is functionally broken.

## Verdict

Docs footprint delta this run: 0 (no deletions). Content-accuracy: 10 active references updated. One
byte-identical non-md requirements pair is staged above for human review only. The doc consolidation the
operator remembers wanting was already executed across the four archive passes above.
