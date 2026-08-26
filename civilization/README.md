# Civilization control plane

Drives `HAWKING_CIVILIZATION_ASCENSION_V1`. Frozen plan:
`~/Downloads/HAWKING_SUPER_ROADMAP_FREEZE_V1_2026-08-25.md`
(sha256 `b5745e7534ac7acb3d0e6e5ecc1af7d889d0e3196d098a646be82e17dd43d047`).

This directory exists so the next checkpoint can be produced **without repeating
the originating prompt**. `ERA_I_CHECKPOINT_001.json` was hand-written, which is
exactly why 002 needed the prompt again.

## Two authorities, deliberately not merged

| | |
|---|---|
| **Obligation ledger** | `~/.claude/ultragoal/hawking-odyssey-maxx-ascension/GOAL.md` — G001–G063. Obligation status lives here and **only** here. |
| **Civilization ledger** | `ROADMAP_STATE.json` — generated. **Points at** the ledger and receipts; never copies their authority. |

Duplicating them would be the "duplicated authorities" defect the directive's
census exists to find. If you want to change an obligation's status, edit
`GOAL.md`; the ledger will follow on the next build.

## The loop

```bash
python3 civilization/build_state.py      # regenerate ROADMAP_STATE.json from disk
python3 civilization/validate.py         # refuse it if it inflates; exit 1 on violation
python3 -m pytest civilization/ -q       # 36 mutation tests on both validators
python3 civilization/checkpoint.py 2     # emit ERA_I_CHECKPOINT_002.json
```

`checkpoint.py N` needs `authored_00N.json` beside it. That file is the
**judgement half** and is not optional — a checkpoint of derived numbers alone
cannot say what became true:

```json
{
  "what_became_physically_true": ["... (cites a receipt, obligation id or tool path)"],
  "what_was_refuted":            ["..."],
  "what_changed_in_the_roadmap": ["..."]
}
```

Every claim must cite something on disk. The validator refuses an authored claim
matching no `*.json`, `G\d{3}`, `tools/…` or `receipts/…` — prose is not evidence.

## What is derived vs what is judgement

**Derived** (never retype these): obligation status and counts, evidence
categories, all percentages, receipt counts, the commit, the test count, running
lanes, acquisition workers, retraction markers, regressions.

**Judgement, written in the open** in `build_state.py`: `ERA_MAP` (an obligation
lands where its *evidence* lands, not where its title sounds like it belongs),
`EVIDENCE` per-category tables and notes, `DEPENDENCIES`, `NAMED_GATES`,
`GATE_BLOCKS`, `NEXT_DECISIVE_GATES`.

`resource_ownership` used to be judgement and carried the literal
`"4 hf download workers"`. It was wrong the moment the fill changed shape. It is
derived from `ps` now. **Any field a human can retype is a field that will lie
with confidence.**

## Why I-D reads 0.0% completion against 100% evidence

`completion_pct = min(evidence_pct, obligation_pct)`. The minimum on purpose.
I-D Accelerator has 9/9 evidence categories and 0/10 obligations VERIFIED. It is
the deepest-evidenced civilization in the program with every gate still open.
Both numbers are published side by side because either alone misleads — evidence
alone inflates, obligations alone erases 77 receipts.

The first build of this ledger printed 100% for I-D. That is the specific defect
`test_inflating_completion_to_evidence_coverage_is_REFUSED` now prevents.

## The validators refuse, and have been watched refusing

Every rule in `validate.py` and `checkpoint.py` has a mutation test that breaks
the ledger on purpose and requires a violation. Both suites also assert the *real*
ledger produces **zero** violations — without that, a validator that refused
everything would make every other test pass.

Rules that caught real defects on their first run against live data:

- `SUDO_PURGE_OR_96GiB_WORKING_SET` was recorded as "a repeatable cold read" —
  an unquantified blocker, which directive XII refuses. Now quantified.
- `running_lanes` reported **0** while three Claude workflow agents were mid-edit
  in this repo. The detector knew only about Grok. A census blind to the executor
  actually in use is not a census.
- The grok pid check used `tf in c` with `tf` possibly empty, which matches every
  process line — so a lane naming no task file passed silently whenever any grok
  process existed anywhere on the machine.

## Executors are not equally observable

`running_lanes` records `detection` per lane and the difference is never smoothed
over:

- **grok — definitive.** A live `grok` process holding the lane's `task.md`.
  Never the status file: on 2026-08-25 four lanes killed by an HTTP 402 all still
  read `done` with `exit_code` 1 and a 0-byte diff.
- **claude — heuristic.** A workflow agent transcript touched within
  `AGENT_QUIET_SECS`. There is no pid, and an agent can be legitimately quiet
  during one long tool call, so this can report a finished agent as alive.

## Era sovereignty

Era I is sovereign. Later-era work (Fusion, HMF, eGPU, Green Machine) is permitted
only when it is already running, consumes an otherwise idle resource, produces
infrastructure Era I needs, or resolves an uncertainty that changes Era-I design.
It **never** earns civilization completion. `ERA_MAP` carries those entries
outside `ERA_I`, and they are excluded from Era-I percentages by construction.

## Environment facts that have already cost time

- Default `python3` is **3.14.6 and has no mlx**. The mlx interpreter is
  `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`. The same test
  file is *5 failed* under one and *9 passed* under the other. Any receipt
  quoting a test count must name its interpreter.
- `/Volumes/corpdrive` is owned by the ModelLake fill. Reading a few KB of JSON is
  fine; staging gigabytes is not. **Check a resource objection against the actual
  byte count** — a 1,661-byte config was once avoided on bus-contention grounds,
  and it cost the only coverage that mattered.
