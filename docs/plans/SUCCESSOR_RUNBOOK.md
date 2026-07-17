# Successor condenser: operational + rollback runbook

The unattended adaptive condenser control plane (`tools/condense/succ_*.py`). Additive,
default-off, non-interfering: it never mutates the immutable legacy campaign and never
launches heavy work while that campaign runs. Target end state while the legacy campaign is
live is State B (master goal section 3): green code, transition armed and bound, controller
booted into `WAIT_OLD_RELEASE`, queue durable, no further chat turn required to cross the
signed release boundary.

House style: no em or en dashes, no middots.

## 1. What is genuinely implemented (executable)

| module | role | state |
|---|---|---|
| `succ_events` | append-only hash-chained event log + sequence + tamper detection | executable |
| `succ_state` | canonical BOOT..SEALED_PARENT state machine + journaled checkpoint + exact resume + split-brain refusal | executable |
| `succ_queue` | durable queue, status vocabulary, section 11.2 row schema, 72B/120B/671B rows | executable |
| `succ_admission` | SOURCE-BOUND adapter capability probe (runs the adapter `capabilities` subcommand, parses its real report, ANDs the section 5.4 requirements) | executable |
| `succ_transition` | one-use transition intent, bound to exact identities, tamper-tested, rollback; re-derives the gate from disk (no caller `all_pass`) | executable |
| `succ_watchdog` | fcntl singleton lease, fail-closed adoption (never on pid alone), launchd plist, read-only resource sampling | executable |
| `succ_telegram` | successor Telegram service: events, dedup cursor, bounded retry, heartbeat, redaction, injected+real sender | executable |
| `succ_doctor` | typed mechanism registry (13 lineages + controls), joint base/healing allocator (real MCKP DP, matches brute force), causal-control set; unwired treatment hooks are NOT selectable | executable |
| `succ_engine` | acquisition function, source-bound program materialize/validate, GATED lightweight dispatch, idempotent ingest | executable |
| `succ_gc` | evidence-closed retirement + safe GC (no-follow, receipts, never-delete classes) | executable |
| `succ_eta` | empirical per-segment runtime + conservative/expected/optimistic ETA; refuses one global constant; marks uncalibrated 120B/giant as no-ETA | executable |
| `succ_audit` | E0 signed live-state packets (read-only) | executable |
| `succ_cli` | the `successor` command surface wiring all of the above | executable |

Tests: `python3.12 -m pytest tools/condense/tests/test_succ_*.py tools/condense/tests/test_eco_*.py` (green).

## 2. Honest blocked / not-yet-real surfaces

- Heavy 72B/120B/giant EXECUTION: gated by non-interference (legacy running) and by adapter
  readiness. 72B `codec_control` is still a live legacy cell; the successor row is
  `waiting_old_release`. Doctor treatment EXECUTION beyond `method=none` is blocked because
  the qwen2.5-dense adapter reports `lora_kd`/`blockwise_qat`/`strand_hessian` unsupported
  (`succ_doctor` marks them non-selectable; `succ_admission` shows only `method=none` ready).
- 120B: its own adapter (`doctor-v5-strand-ladder-gpt-oss-moe`) is a `0.1-contract` whose
  `run` refuses (exit 78). The queue row is `waiting_adapter` with the real capability-report
  blockers (source->STR2 conversion, MoE STR2 loader, tokenizer, evaluator, native parity,
  disk-infeasible 183 GB, human review). This is derived from a live probe, not a file check.
- 671B (DeepSeek-V3): `waiting_source_authority` (exact revision unbound, deepseek-moe adapter
  not built, 1342 GB source vs ~176 GB free). Bounded-stream lifecycle designed, not wired.
- CI: the Rust `check` and `frontend` jobs were pre-existing red on `main` (rustfmt drift;
  `app/pnpm-workspace.yaml` missing a `packages:` key under pnpm 9). Fixes are on a separate
  branch; clippy/build/test green needs a heavy compile only remote CI can confirm.
- The operator supersession authorization is integrity + possession control (self-seal +
  restricted file), not cryptographic authenticity, unless a real detached signature is
  supplied. This is labeled everywhere it appears.

## 3. Operating commands (all real; JSON output)

```sh
CD=tools/condense
python3.12 $CD/succ_cli.py audit            # write signed E0 packets (read-only)
python3.12 $CD/succ_cli.py compile          # import legacy evidence, probe adapters, build queue, boot into WAIT_OLD_RELEASE
python3.12 $CD/succ_cli.py status           # controller + queue + transition + telegram
python3.12 $CD/succ_cli.py queue            # all rows
python3.12 $CD/succ_cli.py queue --model 72B
python3.12 $CD/succ_cli.py explain-next     # acquisition-selected next experiment + why
python3.12 $CD/succ_cli.py ping             # WAIT_OLD_RELEASE heartbeat (appends an event)
python3.12 $CD/succ_cli.py resume           # exact resume from the journaled checkpoint
python3.12 $CD/succ_cli.py verify           # event-chain + queue-seal integrity
python3.12 $CD/succ_cli.py drain            # safe drain
python3.12 $CD/succ_cli.py transition-status
python3.12 $CD/succ_cli.py arm-transition --intent <intent.json>
python3.12 $CD/succ_cli.py telegram status
python3.12 $CD/succ_cli.py telegram test --go   # sends a real successor test message (needs Keychain creds)
python3.12 $CD/succ_cli.py calibrate --model 72B --out <path>   # precompile the post-release calibration
python3.12 $CD/succ_cli.py watch --once          # one detached tick (heartbeat + gate re-check)
python3.12 $CD/succ_cli.py arm-template --out <path>            # emit the unsigned intent template
python3.12 $CD/succ_cli.py watch-plist --out <path>            # write the launchd plist (does NOT install)
```

## 4. Arming the transition (State B) - the operator's two steps

The agent builds and tests all the machinery and precompiles the artifacts, but it does NOT
self-sign or install an auto-activating system agent. Two operator steps complete arming:

1. `successor compile` boots the controller into `WAIT_OLD_RELEASE` and materializes the queue.
2. `successor arm-template --out intent_template.json` writes an UNSIGNED template bound to the
   exact current identities (legacy plan sha, successor commit/tree, `expected_terminal_count` =
   the full 320-cell cohort). **Operator step 1:** add an `operator_signature` (a real detached
   signature over an operator key is preferred; a permission-restricted authorization file + self
   seal is the fallback and is possession/integrity control, not cryptographic authenticity), then
   `succ_transition.make_intent(**make_intent_fields, operator_signature=<sig>)` to produce the real
   sealed intent, and `successor arm-transition --intent <intent.json>`.
3. `successor watch-plist --out watcher.plist` writes the LaunchAgent plist calling
   `successor watch --once --go --intent <intent.json>` every 300 s. **Operator step 2:**
   `launchctl load watcher.plist`. The watcher then heartbeats and re-checks the gate each tick and
   fires `execute_transition` automatically the moment the gate passes (all cells terminal, both
   group reports sealed, checkpoints accepted, quiescent, signature valid, intent verifies, not
   consumed). One-use is enforced by an `O_EXCL` atomic claim + an append-only consumed receipt.

Until both operator steps are done, crossing the release is NOT hands-off. This boundary is
deliberate: an auto-activating system agent and a supersession signature are the operator's to
authorize.

## 5. Rollback and recovery

- Crash mid-state: `successor resume` reloads the journaled checkpoint and refuses if the log head
  diverged (split-brain) or the checkpoint self-seal is broken. The event chain is re-verified.
- Bad activation: `succ_transition.rollback(campaign_root)` restores the prior manifest or explicit
  default-off. Activation is one-use; a consumed intent cannot re-fire.
- Corrupt queue: a tampered row fails its seal on load (`Queue.load` raises), so a corrupt queue is
  never silently trusted.
- GC safety: `succ_gc.gc_apply` deletes only allowlisted objects with a valid receipt, refuses
  symlinks/traversal, and re-verifies post-delete. Never-delete classes (results, receipts,
  negative evidence, source identities, active windows) are refused.

## 6. Non-interference proof

- Additive only: all new files are `succ_*` / `eco_*`; no existing tracked file is modified except
  the run-report wave entry. Successor state lives under `reports/condense/event_horizon_successor/`,
  never the campaign namespace.
- `succ_engine.dispatch_lightweight` refuses any non-lightweight (heavy) adapter subcommand.
- The activation gate, run against the live campaign, refuses (legacy running, no reports sealed, no
  signature). During the build the legacy 72B cell advanced shards untouched.
