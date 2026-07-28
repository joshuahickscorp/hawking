# Independent HIDE YOU/CHAT/IDE integration audit

## Role

Read-only audit. Do not edit files. Determine the exact remaining production
integration work for HIDE under the current Hawking final-ascent directive.

## Inputs

Inspect current HEAD and recent relevant commits/receipts, especially:

- `HAWKING_RESUME_CHECKPOINT.md`
- `HAWKING_PARALLEL_STATUS.json` and `.md`
- `HAWKING_PARALLEL_LANE_OWNERSHIP.json`
- `HIDE_YOU_SURFACE_AUTHORITY.json`
- `HIDE_YOU_MEMORY_CONTROLS.json`
- `HIDE_YOU_SWARM_CONTRACT.json`
- HIDE capability/receipt/status files
- `crates/hide-*`, `crates/hawking-context`, `crates/hawking-serve`
- `app/src` CHAT/IDE surfaces
- relevant commits since 2026-07-26 and unintegrated Grok worktrees, but do not
  treat an uncommitted worktree as landed

The named `HIDE_YOU_PERSONAL_AI_EXTENSION.md` is absent from current HEAD and all
local Git history visible to the controller; state that limitation.

## Required assessment

For each required subsystem—live real-model turn, tokenizer-true Context OS,
reversible compaction, context-rot, prefix/KV reuse, six memories and
inspect/correct/forget/export, personal context graph, multimodal objects, deep
research, connector ABI and real connectors, swarm/fleet, projects/handoffs,
personal tools, automations, privacy/offline/ephemeral, shared CHAT/IDE session,
permissions/effect ledger/checkpoints, tool-safe speculation—classify:

- production and evidence-backed;
- implemented but integration/test incomplete;
- fixture/simulator/test-provider only;
- declared only;
- blocked on capable Hawking provider;
- absent.

Trace every production claim to code plus a non-skipped test/receipt. Identify
stale or contradictory status claims. Distinguish committed work from dirty
unintegrated worktrees.

Return a dependency-ordered worklist with bounded tasks, exact files/tests,
promotion gates, and which tasks can proceed before the capable provider exists.
Do not call any fixture, declared connector, or CPU test provider production.
Keep `HIDE_KERNEL_TURN=false`.
