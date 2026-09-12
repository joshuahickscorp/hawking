# Claude → Codex Exodus: HCLI handoff

This is the operational handoff for the next Hawking chat. The deep audit is
complete. Claude source remains unchanged and recoverable under
`/Users/scammermike/.claude/` and `/Users/scammermike/.claude-grok/`.

## What is now reachable from HCLI

The useful Grok control plane is available through the existing HCLI bridge and
interactive `/grok` command:

`delegate`, `audit`, `consult`, `revise`, `verify`, `status`, `wait`, `report`,
`cleanup`, `doctor`, `telemetry`, and the optional V2 `mission` planner.

The adapter lives in `hcli/grok_bridge.py`; command wiring is in
`hcli/commands.py`. It records exact provider invocations and preserves
provider output as evidence. Grok output, Grok `verify`, and Grok telemetry do
not mark an HCLI WorkUnit complete. HCLI's verifier, receipt, resource lock,
and guarded landing remain authoritative.

## Inherited truth

- Read `H-MANIFESTO.md`, `AGENTS.md`, and `CLAUDE-CODEX-EXODUS.md` first.
- `civilization/sovereign-goal.txt` and HCLI goal/mission state own durable
  objectives. Do not import the historical Claude ledgers as competing goals.
- The repository is dirty with user work. Preserve unrelated changes; do not
  reset, checkout, clean, or delete worktrees.
- The Grok V2 scheduler is reachable but not promoted to a second HCLI
  scheduler. Its live multi-node behavior remains unproven.
- Grok's sparse-root compiler, learned router, and SuperAgent cache/reducer
  remain provider-specific source evidence, not new Hawking subsystems.

## Required first pass

From the Hawking root:

```sh
cd /Users/scammermike/Downloads/hawking
python3 tools/harness/check_exodus.py --source-integrity
python3 -m hcli --help
python3 -m pytest hcli/tests/test_grok_bridge.py hcli/tests/test_commands_grok.py -q
```

Expected focused result: `52 passed`. If source-integrity fails, stop and
report the changed source path; do not rewrite the manifest hash casually.

## Safe Grok sequence

1. In the HCLI interactive surface, run `/grok doctor`. Preserve the exact
   result; it is a provider health observation, not a Hawking acceptance gate.
   At handoff time the installed wrapper was healthy (`grok 1.0.3`) but its
   read-only doctor reported `You are not authenticated.` Do not put a token in
   this repository; treat provider authentication as an external reopen
   condition.
2. Prepare a mission Markdown file inside the Hawking workspace. Its sections
   are `## node-id`, with optional `deps: node-a, node-b` and `kind:` lines.
3. Dry-plan before launching anything:

   ```text
   /grok mission path/to/mission.md BALANCED --dry
   ```

   Inspect the resulting `.hcli/grok/mission-*.json`, especially the exact
   command, workspace, mode, node plan, and limitations.
4. For a live run, provide an explicit HCLI verifier and use the shared
   mutation lock. Launch only after the dry plan is coherent:

   ```text
   /grok mission path/to/mission.md BALANCED
   ```

   Use `ECONOMY`, `FAST`, or `ULTRA` only when the contract and resource budget
   justify it. Do not infer success from the provider's final prose.
5. Inspect the HCLI mission receipt and any provider task receipt. If a task
   needs a provider follow-up, use `/grok revise <task-id> <contract-file>`;
   revisions are mutation-class work and must remain under the shared lock.
6. `/grok verify <task-id>` and `/grok telemetry <task-id>` are useful
   observations. Treat `hcli_acceptance=false`, absent telemetry, missing
   artifacts, or an unpromoted verifier as inconclusive.
7. Run the independent HCLI verifier, inspect the diff/receipt, and only then
   promote the WorkUnit or claim completion.

The resident-backed HCLI path previously observed on this machine was:

```sh
HCLI_ENDPOINT=http://127.0.0.1:8011/v1/chat/completions \
  python3 -m hcli run --goal "..." --verify strict --root "$PWD"
```

Verify the endpoint at runtime. `127.0.0.1:8080` was the Open WebUI surface
and returned HTTP 405 to delegation POSTs. HCLI's protected-command guard also
correctly refused some `python3` verifier contracts; narrow the protected scope
or choose an accepted read-only verifier without weakening the check.

## Acceptance gate for this migration

The next chat should leave all of the following observable:

- checker and source-integrity pass;
- focused adapter tests pass;
- one dry V2 plan receipt;
- one bounded live HCLI/Grok attempt, or a precise blocked receipt if the
  resident/verifier boundary prevents it;
- independent HCLI verification of any claimed repository mutation;
- no protected-file mutation and no fabricated success.

If live acceptance remains blocked, record the mechanism and reopen condition
in the Exodus report. Do not “fix” the result by disabling the protected
command guard, inventing telemetry, or promoting Grok prose to HCLI evidence.

## Files to leave as the durable record

- `AGENTS.md` — Codex-facing operating boundary.
- `CLAUDE-CODEX-EXODUS.md` — semantic report and limitations.
- `CLAUDE_CODEX_EXODUS.json` — machine-readable inventory, lineage, hashes,
  adapter status, and verification entries.
- `CLAUDE_ONLY_REMAINING` — explicit residual boundary.
- `tools/harness/check_exodus.py` — repeatable discriminator.
- `hcli/grok_bridge.py` and `hcli/commands.py` — HCLI access layer.
- `.hcli/grok/` — runtime receipts; keep secrets out of them.

Do not add a parallel goal, memory, scheduler, cache ontology, or daemon for
this handoff. If a future task proves a real missing capability—such as a
Blender tool or lifecycle invariant—add only a focused adapter with a focused
test and update the manifest.
