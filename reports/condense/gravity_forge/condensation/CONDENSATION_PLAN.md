# Stage B condensation plan (CLEAN SLATE Sections 11-27) - grounded in the measured census

Generated at Stage-B kickoff. This is the honest floor and descent path, measured from git-tracked
files (not estimated). **Stage B is BEGUN, not ended** - the descent below is a multi-week body with
per-checkpoint commit/tag/test/rollback; it is not, and must not be, faked in one pass (Section 24/26:
an honest 70k system beats a fake 49k system with hidden code).

## Measured floor (git-tracked only; `codebase_census.py` + `condense_reachability.py`)

| surface | LOC | disposition |
|---|---|---|
| Rust `hawking-*` engine | 123,987 | kernel (structural condensation, hardest) |
| Rust `hide-*` product | 32,712 | extract -> `hawking-hide-desktop` pack |
| Python tools/condense **reachable** from CLIs | 13,873 (33 mod) | kernel spine (already near-condensed) |
| Python tools/condense **orphaned** | 180,373 (279 mod, 93%) | -> `hawking-lab` pack (manifest SEALED) |
| tests | 55,373 | condense via property/model tests (Section 22) |
| documentation | 28,240 | -> `hawking-docs-archive`, keep ~8 canonical docs |
| third_party (vendor/strand) | 80,556 | audit-only; not kernel |
| generated / assets / fixtures | ~14k | stop-track regenerable; pack large fixtures |
| **total tracked** | **568,921** | |

**True CLI-reachable kernel floor: ~137,860** (124k Rust + 14k Python). The Python spine is already
small; the two big levers are (1) packing the 180k orphaned Python and (2) condensing the 124k Rust
engine. The 50-75k kernel target is reachable only if the Rust engine condenses substantially; a
124k Rust inference engine may honestly floor nearer 75-100k, which Section 26 explicitly permits.

## Descent checkpoints (Section 23), each = commit + tag + tests + rollback + receipt

- **175k** - non-code consolidation: seal `hawking-docs-archive` (28k), `hawking-lab` MOVE the 279
  orphaned modules (manifest already sealed), stop-track regenerable generated files. One CLI, one
  config. Gate: reachable kernel still green, no import breaks.
- **150k** - evidence core + controller unification (Section 16/17): collapse `doctor_v5_* / eco_* /
  succ_* / gravity_* / forge_*` into one data-driven controller; one evidence authority.
- **125k** - pack extraction + default-checkout reduction; extract `hawking-hide-desktop` (Rust hide
  crates + frontend).
- **100k** - architecture-primitive adapters (Section 19).
- **80k** - Gravity/Forge/Doctor condensation to their Section-20 budgets (2-4k / 4-8k / 4-7k).
- **65k** - product/kernel split; optional HIDE fully external.
- **50k** - final honest kernel attempt on the Rust engine; accept the honest floor if two
  materially different architectures fail structurally (Section 26 Escape Receipt).

## Why the descent is NOT executed in this kickoff (honesty)

1. Moving the 279 orphaned modules needs the full test suite green afterward; the suite has 8
   pre-existing `doctor_v5` import failures unrelated to Forge - those must be triaged first.
2. Some orphaned modules are the live/retired `doctor_v5` campaign referenced by launchd jobs
   (`com.hawking.doctorv5.telegram`, `...post120b`); moving them blindly would break those services.
   The campaign lifecycle must be resolved before the move.
3. The Rust engine (124k) condensation is real structural work (crate unification, dead-code proof)
   that each needs parity evidence (Section 19: "every consolidation requires parity evidence").

Next safe checkpoint: triage the 8 test-collection failures + resolve the doctor_v5 campaign
lifecycle, then execute the 175k checkpoint (docs-archive + lab pack MOVE) with a full green gate.
