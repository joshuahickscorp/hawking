# Lane map — 250k rebuild

Controller-owned. Every lane runs in its own worktree off a recorded HEAD, so no lane can
collide with another on disk. Collisions happen at merge, and this file is where they are
seen before they happen.

## In flight

| lane | owns | maps to | notes |
|---|---|---|---|
| `s2-lab` | as below | Core C | **engine kept, cutover rejected** — deleted 1,596 assertions and 77 entrypoints with 29 added back. `lab/` preserved at `scratchpad/s2lab/lab/` |
| `s2b-lab-cutover` | `tools/condense`, `tools/campaign` (minus frozen), `tools/foundry`, `tools/prometheus`, `tools/odyssey`, `odyssey/` — 81,368 LOC | Core C | same scope, gated by the capability manifest and per-assertion accounting |
| `s3-hide` | `crates/hide-backend`, `crates/hide-kernel`, `crates/hide-core`, `crates/hawking-context` — 83,430 LOC | Core D (C6+C7) | target 52,500 |
| `graph-repair` | `tools/graph/**` | instrument | dominators, co-change weight, imports, plus the confidence-weighting defect |
| `perfgate` | `tools/verify/perfgate.py` | instrument | the runnable 2% gate |
| `recomp-p5-tests-docs` | `docs/**`, root `*.md`, `tools/condense/tests/**` | Core F, partial | started under the prior arc's rules, before this campaign |
| `recomp-bridge2-take-5376` | `crates/hawking-adapters`, `-events`, `-index`, `-orch`, `-research` | Core E (C8) | **stalled**: 2h20m, zero commits, zero bytes of output. Not blocking; left running |
| `s4-surface` | `crates/hide-protocol`, `-fleet`, `-acp`, `-serve`, `app/` — 35,820 LOC | Core E (C4+C8) + Core D (C9) | target 24,500; also owns closing the 6 unearned generated files |

## Known merge collision

`s2-lab` and `recomp-p5-tests-docs` both claim `tools/condense/tests/**`.

They were launched under different rules — p5 predates this campaign and is reducing the
existing test tree in place, while s2-lab replaces the whole of `tools/condense` with a
clean-room `lab/` and deletes it. If s2-lab lands, p5's work on that path is moot by
construction.

**Resolution at merge: s2-lab wins for `tools/condense/**`. Take p5's `docs/**` work only.**
As of 14:15 `recomp-p5-tests-docs` has also produced nothing in 2h20m, so this collision may
resolve itself by p5 delivering nothing at all.

Second, smaller collision: `s4-surface` owns `control/GENERATED_REGISTRY.json`, and one of
the six unearned generated files it must register — `crates/hawking-adapters/generated/sdk_types.d.ts`
— sits in `recomp-bridge2`'s tree. If bridge2 ever lands, that one entry is re-checked.
Recorded here rather than discovered later, because the failure mode is taking both and
ending up with a `lab/` plus a surviving `tools/condense/tests/` proving the deleted thing.

## Not yet started

| work | maps to | why it is waiting |
|---|---|---|
| device + runtime | Core B (C2+C3) | last by design — it holds the protected performance and numerical contracts, and it is the one boundary where a wrong cut is expensive rather than merely wasteful |
| `app/`, `hide-protocol`, `hide-acp`, `hide-serve`, `hide-fleet` | Core D (C9) + Core E (C4) | gated on `s3-hide` settling the agent core's shape |
| tests rebuilt from the constitution | Core F | gated on the slices, since there is no point regenerating tests against code that is about to be replaced |
| topology collapse | all cores | measured surface, waiting only on lane conflicts: **55 directories hold two or fewer source files** (78 files between them), and **251 files are under 50 lines** totalling 4,917. Collapsing those is most of the distance from 131 directories to the T4 target of 60, and it touches every crate, so it can only run when the build lanes are clear |

## Rules for every lane

- Own paths only. Never `git add -A` — this branch is shared with other agents.
- Cut over and delete in the same commit. No forwarders, no shims, no old+new double system.
- `tools/loc/**`, `tools/verify/blackbox.py`, `tools/campaign/rung_gate.py` and
  `control/SEMANTIC_GRAPH_SCHEMA.json` are frozen instruments.
- A check that stops running is a failure, not a neutral.
