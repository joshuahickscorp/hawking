# Proposed CLAUDE.md addition — Kill Protocol rule

**Status:** APPLIED 2026-05-30 to the project `CLAUDE.md` (the dismantle
operating contract), as a new `## Kill Protocol rule` section between the
Evidence rule and the Memory-coexist rule, on explicit user direction. This file
is retained as the rationale/proposal record; the live rule is in `CLAUDE.md`.

## Where it goes

`CLAUDE.md`, as a new top-level rule section, placed after the **Evidence rule**
and before the **Memory-coexist rule** (it belongs with the other gate-discipline
rules: Halt, Evidence, Verification). It is the standing-rule companion to the
worked kill ledger in `plans/throughput_bible_2026_05_30.md` §8.3.1.

## Why

Five offline oracles have now killed levers (block-256, then L1.3/L1.4/L1.5/L2.2).
The oracles are rigorous, but they do not by themselves stop us from killing an
*idea* when only one *form* of it was tested — or from re-spawning the same dead
lever a third time. One of the four kills (L1.4) turned out to have tested the
naive data-free SVD when the standard fix is activation-aware SVD. The protocol
institutionalizes catching that, while still accepting genuine Type-1 deaths
without manufacturing false hope.

## Exact proposed text (copy into CLAUDE.md verbatim)

---

## Kill Protocol rule

Before marking any lever **NO-GO** (in `reports/dead_levers.md`, a
blocked-doc, or a closeout), the kill MUST record three things:

1. **Type-1 or Type-2.** *Type-1* = died on a measured property of
   reality that no implementation cleverness changes (a delta with
   higher variance than the original; an FFN active set that is
   ~half the neurons; no hardware gather on Apple Silicon).
   *Type-2* = died only in the **form** tested, where a different
   formulation attacks the same goal (data-free vs **data-aware**,
   gather vs **gather-free**, extracted-post-hoc vs **trained-for**).
2. **The reframe considered.** Name the specific alternative
   formulation explicitly — even if only to reject it.
3. **Why the reframe also dies, OR a pointer to its oracle.** A
   Type-2 reframe is "alive" **only** with a named, cheap
   (offline / CPU NumPy-scale) oracle that could kill it.

Hard rules:
- **Never resurrect on vibes.** No nameable kill-oracle → the lever
  stays dead.
- **Never re-test a recorded Type-1 kill.** Its death is a fact
  about reality, not about our effort.
- **Accept Type-1 deaths.** Do not manufacture a reframe to avoid a
  kill. The point is to catch the *rare* genuine Type-2, not to pad
  the ledger — "all of these are genuinely Type-1" is a correct and
  common answer.

The worked example (the four Phase-A kills, classified and
retro-filled) lives in `plans/throughput_bible_2026_05_30.md`
§8.3.1 "Kill Protocol".

---
