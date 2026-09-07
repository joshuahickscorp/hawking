# Measured frontier input — a checkpoint MISS costs 2.27x a cold prefill — 2026-09-05

SUPERSEDES the earlier framing of this file, which said a RESTORE was the expensive path.
That was wrong and is corrected here. The measurements were always right; the diagnosis was
not, and it was wrong twice before this. Read the numbers, not the earlier headline.

## THE MEASUREMENT

Three calls, ONE resident process. Calls 2 and 3 share their entire leading text with the
call before them; only the final instruction line differs.

    label            prefix_source        prompt  reused  fresh   fresh t/s   eff t/s     wall
    1 cold           cold                   3923       0   3923       65.2      65.2    66.34 s
    2 shared prefix  cold                   3924       0   3924       26.1      26.1   150.54 s
    3 append         checkpoint_restore     3927    3909     18       19.1    4170.5     1.07 s

Read the `prefix_source` column carefully.

    call 2 is NOT a restore. It reports COLD and reused=0. It MISSED the stored checkpoint
           and fell back to a cold prefill -- one that took 2.27x the first cold prefill of
           the same length.
    call 3 IS the restore, and it is the fast one: 3,909 of 3,927 positions reused, 1.07 s.

Control from the same session: with prompts built from a DISJOINT vocabulary, sharing no
prefix at all, a second and third call in one process ran 60.35 s and 60.32 s against a
66.28 s fresh-process control. A warm process is not slow. A process that MISSES a
checkpoint is.

## WHY THE MISS HAPPENS (read from the code, not measured)

`snapshot_boundary` in crates/hawking-core/examples/ascension_qwen38_resident.rs chooses
where to snapshot. On the first pass the boundary lands PAST the point where the next prompt
diverges, so the stored checkpoint is not a prefix of that prompt and cannot be used. The
miss then sets `checkpoint_missed = true`, which lowers the floor to 16 and re-snapshots
early -- and the call after that HITS. Miss, then hit, is exactly what the table shows.

## THE QUESTION FOR THIS MISSION

1. WHY does a MISS cost 2.27x a plain cold prefill? Both calls are cold. Both step ~3,923
   fresh tokens. 66.34 s versus 150.54 s. Something on the miss path does work a
   first-in-process cold call does not. Find it. This is the open question and it is worth
   more than the other two.

2. Can the FIRST snapshot be placed where it will actually be reusable, so the miss never
   happens? The retry already knows how -- it uses floor 16. If the first snapshot used a
   boundary likely to be a prefix of the next prompt, call 2 would hit instead of missing.

3. Does HCLI hit or miss in production? Its traces show prefix_reused_tokens of
   1,151 / 1,421 / 3,130 / 3,222, so it does reuse. Measure the MISS RATE across a real
   mission: every miss is costing 2.27x a cold call.

## CONSTRAINTS

- Report fresh and effective prefill SEPARATELY, always. Call 3 reads 4,170 tok/s of prompt
  while freshly computing 18 tokens; quoting that as a prefill rate would be a lie.
- Token identity is non-negotiable.
- Every file in receipts/sovereign/VERIFIER_MANIFEST.json is PROTECTED.
- Smallest executable change. Name a test that fails before and passes after.
