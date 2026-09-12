# Hawking project instructions

These instructions are the Codex-facing adapter for Hawking. They do not create
a new Hawking subsystem and do not replace the existing ontology.

## Authority and orientation

- Read `H-MANIFESTO.md` and `CLAUDE-CODEX-EXODUS.md` before changing the harness.
- Treat current source, live process state, receipts, tests, and measured runtime
  observations as higher authority than roadmaps, prose claims, or old prompts.
- The durable owners are HCLI, Hawking, Gravity, AgentOS, Nova, Noetic, NX, and
  Odyssey. Do not introduce a parallel Codex goal, memory, receipt, or daemon
  system when an existing owner can carry the behavior.
- `civilization/sovereign-goal.txt`, HCLI goal/mission state, and the current
  repository receipts carry persistent objectives. Claude's historical
  `.claude/ultragoal/` files are migration evidence, not an authority to edit.

## Working loop

1. Recover truth from disk: `git status --short --branch`, relevant call sites,
   live state, and the smallest useful receipt/test set.
2. Identify the narrowest frontier that can change the outcome. Prefer a cheap
   falsifier before an expensive experiment.
3. Keep one authoritative mutation writer. Independent read/research lanes may
   run in parallel; writers need disjoint scopes or a controlled landing step.
4. Preserve unrelated user changes. Never use destructive reset/checkout or
   delete a worktree with uncommitted work.
5. A report is a claim. Mark work complete only with fresh, reproducible
   evidence: command output, test, runtime observation, receipt, diff, or render.
6. Record important results in the repository's existing receipt/mission paths;
   do not manufacture a second memory or evidence ontology.

## HCLI bridge

Use HCLI for durable or bounded work that should survive a model session:

- `python3 -m hcli --help`
- Before a resident-backed delegation, verify the actual OpenAI endpoint. On
  this machine the live HCLI resident surface was `http://127.0.0.1:8011/v1`
  while `:8080` was the Open WebUI surface and returned HTTP 405 to delegation
  POSTs. Set `HCLI_ENDPOINT` explicitly when needed:
  `HCLI_ENDPOINT=http://127.0.0.1:8011/v1/chat/completions python3 -m hcli run ...`
- `python3 -m hcli run --goal "..." --verify strict --root "$PWD"`
- `python3 -m hcli status <mission>` and `python3 -m hcli result <mission>`
- `python3 -m hcli steer <mission> "..." --kind knowledge|correction|constraint`
- HCLI's `/ultragoal`, `/mission`, `/steer`, `/grok`, and `/receipts` commands
  are the native interactive equivalents. The `/grok` adapter exposes
  `delegate`, `audit`, `consult`, `revise`, `verify`, `status`, `wait`,
  `report`, `cleanup`, `doctor`, `telemetry`, and the optional V2 `mission`
  planner.
- Start a new Grok V2 mission with `/grok mission path/to/mission.md BALANCED
  --dry` before considering a live launch. The adapter records the exact
  invocation in `.hcli/grok/mission-*.json`; V2 stdout and `/grok verify` are
  provider evidence, not HCLI acceptance. HCLI must independently inspect the
  workspace and run its verifier.

Use absolute paths or an explicit workspace when a verifier needs repository
files. A mission that cannot reach its verifier inputs is not verified.
HCLI's protected-command guard may also refuse an interpreter command such as
`python3` when the contract names protected paths; preserve that fail-closed
behavior and either use a suitable read-only verifier command or narrow the
contract's protected-path set without weakening the actual check.

## Delegation, research, and worktrees

- Use Codex's native parallel/task surfaces for independent reasoning when they
  are sufficient. Preserve the old read-only scout, verifier, research, and
  synthesis roles semantically; do not recreate agent proliferation for its own
  sake.
- HCLI's Grok bridge and the existing
  `/Users/scammermike/.claude-grok/bin/grok-run` wrapper are adapters, not the
  authority. `grok-mission` is also adapter-only and its multi-node live path
  remains unproven. Contracts must name scope, non-goals, tests, and evidence.
  Review every result independently; use `/grok revise` only under the shared
  mutation lock.
- Repository worktrees belong under `.worktrees/` and must be linked to the
  repository with Git. Land or classify their work before cleanup.
- Use Codex's native browser/web/computer-use surfaces for current research and
  UI verification. Preserve source hierarchy and citation discipline; do not
  carry forward old browser workarounds without evidence that they are needed.

## Safety and attribution

- Keep credentials, tokens, cookies, private keys, and secret environment values
  out of manifests, receipts, prompts, commits, and reports. Record only the
  variable or secure configuration path when necessary.
- Preserve the configured human Git identity. Do not add Codex, OpenAI, Claude,
  Grok, or other AI attribution to commits, PRs, or generated headers unless the
  user explicitly requests it.
- Do not weaken protected verifiers, authorization boundaries, resource guards,
  or negative controls to make a result pass. If a capability is unavailable,
  record the mechanism and reopen condition.

## Migration provenance

The Claude source remains under `/Users/scammermike/.claude/` and is intentionally
not rewritten by this migration. The complete initial mapping, source hashes,
verification evidence, and unresolved gaps are in:

- `CLAUDE-CODEX-EXODUS.md`
- `CLAUDE_CODEX_EXODUS.json`
- `CLAUDE_ONLY_REMAINING`
