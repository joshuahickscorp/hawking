# Claude → Codex Exodus

Deep semantic migration report for the Hawking repository. Generated
2026-09-10 from the user's home-level Claude state and the current Hawking
checkout. This is a migration record, not a new Hawking ontology.

## Result

The first pass preserves Claude as recoverable source evidence and moves the
high-value behavior to the smallest existing owners:

- durable goals, missions, steering, receipts, and continuation stay with HCLI /
  Hawking;
- project-wide Codex discovery is supplied by `AGENTS.md`;
- provider-neutral engineering behavior is expressed as a short adapter here,
  while `H-MANIFESTO.md`, Gravity documents, and HCLI runbooks remain canonical;
- Codex already supplies the overlapping domain skills, browser, computer-use,
  worktree, and parallel-task surfaces;
- Grok remains an adapter reachable through the existing HCLI bridge and wrapper;
  HCLI now exposes the useful wrapper control plane, including revise, verify,
  doctor, telemetry, and the optional V2 mission planner;
- no Claude source was rewritten, deleted, or copied wholesale.

Machine-readable authority: [`CLAUDE_CODEX_EXODUS.json`](CLAUDE_CODEX_EXODUS.json)
Operational handoff: [`docs/CLAUDE_CODEX_EXODUS_HCLI_HANDOFF.md`](docs/CLAUDE_CODEX_EXODUS_HCLI_HANDOFF.md)

## Inventory

Measured source surfaces under `/Users/scammermike`:

| Surface | Evidence | Initial finding |
|---|---|---|
| Global instructions | `.claude/CLAUDE.md` (189 lines) | Git attribution, delegation policy, scratch hygiene, verification discipline, and agent routing. |
| Goal contract | `.claude/ULTRA-CORE.md` (487 lines) | Goal immutability, evidence-gated completion, frontier selection, parallelism, failure/recovery, and no-soft-terminal rules. |
| Commands | `.claude/commands/` — 11 files | Grok consult/delegate/audit, steering, waves, Ultragoal, DND, and ergonomics. |
| Read-side agents | `.claude/agents/` — 6 files | Narrow grep/read/test/verify/research/synthesis roles; no write authority. |
| Hooks | `.claude/hooks/` — hook, backup, and test | Ultragoal stop/prompt enforcement; SuperAgent lifecycle hooks in settings. |
| Workflows | `.claude/workflows/` — 2 files | SuperAgent planning/execution with tested invariants. |
| Skills | `.claude/skills/` — 14 top-level directories / 379 files | 13 Cloudflare/Workers/research skills already exist in `.codex/skills`; one custom Grok orchestration skill is provider-specific. |
| Goal state | `.claude/ultragoal/` — 315 files, 3 active slots | Many historical ledgers; active intent is mapped below, not blindly duplicated. |
| SuperAgent state | `.claude/superagent/` — 3,399 files | Large generated/session corpus; behavior is reduced to evidence and HCLI ownership. |
| Plugins | `caveman`, `ponytail` caches | Low-value ergonomics; both have Codex-compatible metadata but are not needed for Hawking capability. |
| MCP | `.claude.json` project/user config has no configured server; Cloudflare marketplace has five definitions | No Hawking-critical Claude MCP dependency proven. A live Blender extension was observed separately and is recorded in `CLAUDE_ONLY_REMAINING`. |
| Project Claude config | `Downloads/hawking/.claude/settings.local.json`, `launch.json` | Local permissions/launch only; no project instruction or goal authority. |

The full file-level counts, source hashes, and path inventory are in the JSON
manifest. Historical sessions, caches, generated artifacts, and plugin payloads
remain in place and are not treated as canonical source.

## Goal preservation

### Active Claude goal intent

The three active session slots were:

1. `EPOCH II: turn discovery into machinery that makes the next discovery
   cheaper -- daemonic HCLI, industrial Odyssey, executable physical
   representations; one 27B resident; no workflows`
2. `Complete the Global Odyssey and synthesize the Noetic executable (NX)`
3. `Return to Odyssey execution: advance real organisms, not the harness`

These are not copied into competing Codex prompts. Their durable owners and
evidence already exist in Hawking:

- HCLI goal/mission state and the daemon own persistence, continuation, steering,
  work units, and receipts.
- `civilization/sovereign-goal.txt` carries the current sovereign objective and
  protected verification obligations.
- `docs/ULTRAGOAL_HANDOFF_2026-09-09.md`, `docs/HANDOFF.md`,
  `docs/RESIDENT_DAEMON.md`, and `docs/HCLI_DELEGATION.md` carry current
  recovery/operator context.
- Odyssey/ModelLake state and receipts remain the scientific campaign authority.

Historical Claude ledgers remain recoverable at their original paths. The
machine-readable map records the active slot paths and the exact goal strings.
No goal is marked verified by this migration; migration verification only proves
that intent has an available durable owner.

## Ported / mapped behavior

### Codex-facing

`AGENTS.md` is the new project instruction surface. It preserves the useful
behavior from Claude's global rules and Ultra-Core without importing Claude
syntax or competing with H-MANIFESTO. It covers disk authority, cheap
falsification, one-writer/many-reader parallelism, receipt discipline, worktree
safety, no AI attribution, and the Codex ↔ HCLI bridge.

### HCLI / Hawking-facing

The following already exist and are the canonical replacements rather than new
ports:

| Claude behavior | Current owner | Verification |
|---|---|---|
| Ultragoal / persistent objective | HCLI `/ultragoal`, `/mission`, goal bank, mission state, sovereign goal | HCLI command registry and mission/goal tests; `python3 -m hcli --help` passes. |
| Mid-goal steering | HCLI `/steer` and `hcli steer` with knowledge/correction/constraint kinds | `hcli/commands.py`, HCLI delegation runbook, goal/mission tests. |
| Evidence-gated completion | HCLI verifier/envelope/receipt path and protected Hawking gates | `docs/HCLI_DELEGATION.md`, `civilization/sovereign-goal.txt`, existing test suite. |
| Continuation / daemon | `hawkingd`, resident state, checkpoints, missions, HCLI memory | `docs/RESIDENT_DAEMON.md` and live `hawkingd` process observed. |
| Grok delegation/audit/consult | HCLI `/grok`, `hcli.grok_bridge`, and `/Users/scammermike/.claude-grok/bin/grok-run` | HCLI command registry and existing bridge tests; wrapper remains an adapter. |
| Grok revise/verify/doctor/telemetry/V2 mission control | HCLI `GrokBridge` plus `/grok revise|verify|doctor|telemetry|mission` | Focused bridge and command tests; every result carries the provider/HCLI acceptance boundary. |
| Receipts and recovery | HCLI `.hcli` state plus tracked `receipts/` | `hcli/commands.py`, receipt tests, current receipt corpus. |
| Parallel waves | HCLI WorkUnits plus controlled Grok worktrees | Existing HCLI scheduler/WorkUnit machinery; Claude `swgrok` remains historical adapter. |

### Command mapping

| Claude command | Native destination | Status |
|---|---|---|
| `/ask-grok` | HCLI `/grok consult` or `grok-run consult`; verify independently | MAPPED |
| `/delegate-grok` | HCLI `/grok delegate` or `grok-run delegate` in a linked worktree | MAPPED |
| `/grok-audit` | HCLI `/grok audit`; read-only independent audit plus synthesis | MAPPED |
| `/steer` | HCLI `/steer` / `hcli steer` | PORTED |
| `/ultragoal` | HCLI `/ultragoal`, `/mission`, goal bank, sovereign goal | PORTED |
| `/sw`, `/swg`, `/swgrok` | HCLI WorkUnits and controlled parallel lanes; one writer, disjoint scopes | MAPPED; no 1:1 command created |
| `/dnd`, `/dndo` | HCLI bounded missions/daemon limits and Codex automation surfaces | SUPERSEDED; Claude stop-cap toggle is not a Hawking capability |
| `/work-with-grok` | HCLI `/grok` plus the provider adapter | MAPPED |

### Agents and skills

The six `sa-*` agents are semantically preserved as narrow read-side roles:
Codex can use targeted tools or parallel tasks for grep, read, test, verify,
research, and synthesis. No agent files were copied because Codex's task surface
and HCLI WorkUnits already provide the needed execution boundary.

The 13 Cloudflare/Workers skills in `.claude/skills/` have corresponding
`.codex/skills/` entries and are `SUPERSEDED` by the current Codex versions. The
custom `grok-orchestration` skill is `MAPPED` to global Codex delegation policy,
HCLI's Grok bridge, and the existing wrapper; literal Claude skill syntax is not
needed.

### Hooks, permissions, and external tools

- The Ultragoal stop hook's important invariants are now represented by HCLI
  mission/goal verification and repository tests. Its Claude lifecycle protocol
  is `NEEDS_REDESIGN`, not blindly translated.
- The SuperAgent hook and plugin hooks are `SUPERSEDED` or low-value ergonomics;
  deterministic checks belong in tests/scripts, not model lifecycle prompts.
- Claude's project/user config did not expose a Hawking-critical MCP server.
  Cloudflare MCP definitions remain a manual-auth option, not a migrated
  dependency. Browser/research work uses Codex-native web/computer-use surfaces.
- The actual environment is already authorized for the trusted Hawking project
  and full local execution in Codex. No new authority was granted by Exodus.

### HCLI access expansion

The useful provider-specific machinery is reachable from HCLI without making
Grok the authority:

- `GrokBridge.revise()` resumes an existing Grok task only with a caller-supplied
  WRITE/VERIFY contract and the shared HCLI mutation lock.
- `GrokBridge.verify()` preserves Grok's structured receipt as advisory data and
  explicitly sets `hcli_acceptance=false`.
- `GrokBridge.doctor()` exposes the wrapper's own health probe; `telemetry()`
  reads an observed `telemetry.json` or records an explicit absence.
- `GrokBridge.mission()` and `/grok mission` reach `grok-mission` with exact
  mode/repo/path capture, workspace containment, dry planning, and a durable
  HCLI receipt. V2 scheduling remains an adapter because its own limitations
  document no proven live multi-node mission.
- The existing HCLI verifier, WorkUnit ledger, resource classes, and guarded
  landing remain the acceptance boundary. Grok's report, telemetry, or verify
  output cannot mark an HCLI unit complete on its own.

The handoff document gives the next Hawking chat the exact commands and rollout
gate. The migration deliberately does not copy Grok's sparse-root compiler,
learned router, or SuperAgent cache as a second HCLI scheduler/context system.

## Provider-independent operating canon

This is the compact behavior that should survive any model harness:

1. Recover current disk/runtime truth before planning.
2. Keep objectives durable in HCLI/Hawking, not in a provider transcript.
3. Select the smallest high-information frontier and falsify cheaply first.
4. Parallelize independent readers/researchers; control writers and landing.
5. Treat prose as a claim and receipts/tests/runtime observations as evidence.
6. Preserve rollback, worktree traceability, source hashes, and user changes.
7. When blocked, record the mechanism and reopen condition, then advance a
   non-confounding front.
8. Reduce representation without reducing capability; keep one owner per
   durable behavior.

`H-MANIFESTO.md`, Gravity documents, and HCLI runbooks remain the canonical
owners of the broader principles. This list exists only to make the migration
boundary explicit to Codex.

## Verification performed

- Source inventory completed without mutating `/Users/scammermike/.claude/`.
- Source hashes recorded in the JSON manifest for the global rules, Ultra-Core,
  settings, key commands, hook, Grok skill, sovereign goal, and H-MANIFESTO.
- Codex global config inspected: trusted Hawking project, native browser/
  computer-use plugins, HCLI/Grok delegation guidance, no configured Claude MCP
  dependency.
- HCLI command registry inspected for `/ultragoal`, `/mission`, `/steer`,
  `/grok`, and `/receipts`; `hcli --help` succeeds.
- `tools/harness/check_exodus.py` is the repeatable migration discriminator.
- A real HCLI acceptance sequence was attempted with protected paths and left
  durable evidence: mission `7f3df89094c3` failed at the unavailable default
  endpoint, mission `689fd52c0a55` reached the `:8080` service but got HTTP 405,
  mission `b108aae6a722` reached the resident on `:8011` but the protected
  command guard correctly refused `python3`, and mission `a68079b4a376` used a
  narrowed protected contract but was aborted after its two-minute bound with
  no verifier progress. A final read-only `jq`/`shasum` mission
  (`9888e3021b2c`) reached the resident but returned `INCONCLUSIVE` without
  promoting command artifacts. All five reported no protected-file mutation.
  The checker itself passes locally; HCLI acceptance remains open rather than
  overstated.
- The expanded HCLI Grok surface passes focused tests: `52 passed` across
  `hcli/tests/test_grok_bridge.py` and `hcli/tests/test_commands_grok.py`.
  This verifies adapter behavior and boundaries, not a live V2 multi-node run.
- The broader `hcli/tests` suite completed with `1362 passed, 7 failed, 7
  skipped`; the seven failures are in unrelated resident/runtime/ML/tool
  surfaces and none are in the touched Exodus adapter tests. The worktree had
  those broader changes before this migration, so they remain classified rather
  than altered here.
- A read-only provider health check through the new HCLI adapter returned exit
  0 with Grok `1.0.3`, but reported `You are not authenticated.` Live Grok
  execution therefore remains externally blocked until the user's provider
  authentication is configured; no credential was recorded or changed.

## Claude-only remaining

See [`CLAUDE_ONLY_REMAINING`](CLAUDE_ONLY_REMAINING). No high-value Hawking
dependency is currently proven Claude-only. The two residuals are low-priority
Claude lifecycle/UI ergonomics and an external Blender MCP extension whose
continued use has not been established.

## Next

Highest-value follow-up: give the operational handoff to a fresh Hawking chat,
have it run the checker and focused tests, then dry-plan and execute one bounded
Grok V2/HCLI mission with an independently accepted verifier. Capture both the
HCLI mission receipt and provider receipt, and compare the result against the
same task's old Claude workflow. If a lifecycle hook, Blender tool, or parallel
lane is genuinely required, migrate that specific capability with a focused
test rather than expanding the harness wholesale.
