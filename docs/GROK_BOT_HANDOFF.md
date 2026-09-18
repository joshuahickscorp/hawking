# Grok Bot ↔ Hawking handoff — 2026-09-12

This is the compact continuation for the external-worker bridge. It is an
operational handoff, not a new goal, memory store, scheduler, or science
authority.

## GROK BOT

Machine-side status: READY FOR USER-MEDIATED SETUP. The canonical Rust
command surface now provisions a fresh Git worktree and records a stage receipt:

    hawking agent stage grok <task> --objective "..."

When using the checkout directly:

    cargo run -p hawking -- agent stage grok <task> --objective "..."

The default staging convention is the sibling directory
hawking-grok-staging/<task>. The task packet is
hawking-grok-staging/packets/<task>.md and the receipt is
hawking-grok-staging/receipts/<task>.json. The worktree is based on the
explicitly recorded canonical HEAD (or the supplied source revision) and uses
the isolated worker/<task> branch. The command records the task id, source
repository, source commit, source branch, worktree, branch, creation time,
allowed scope, worker brief, packet, receipt, and dirty-at-creation state.

After the worker stops, observe the task from the canonical checkout:

    hawking agent result <task>

Add --json for a machine-readable result receipt. This observation checks the
registered worktree, branch, current commit, changed files, diffs, whitespace,
and optional .hawking-worker/result.json. Worker reports remain advisory.
Promotion is always false on this surface.

Before handing the task to the worker, generate the derived, task-scoped
Hawking view:

    hawking agent context grok <task>

This writes `hawking-grok-staging/packets/<task>.context.json` and
`<task>.context.md`. The view is generated from the current continuation, the
canonical Gravity registry, and the stage receipt; it binds the canonical/base
revision and SHA-256 digests, and grants no additional authority. It projects
only relevant passport summaries, current frontier facts, bounded Gravity
methods, controls, Laws, evidence handles, and stop conditions.

This is a Git/CLI boundary, not a kernel sandbox for a local process with
broader OS permissions. Keep local execution at ASK EVERY TIME and do not grant
production credentials.

Authentication and user steps still required: enable the officially supported
Grok Bot local-execution or connector path on the M3 Ultra, point it at the
printed task packet and worktree, and keep approvals at ASK EVERY TIME. Grok
Bot availability/authentication was not assumed or reverse-engineered here.
No SSH key, public listener, production credential, model switch, or daemon
control was created. If the supported Grok Bot path requires SSH, the user
must configure a dedicated staging-only key/entry through the official setup,
without public exposure or production credentials.

Qualification status: not run. No Grok Bot installation or authenticated
session was available to this tranche, and Codex does not wait for interactive
login. The recommended first harmless qualification is the passport-validation
task from the tranche: inspect the 12 current Flash Organ Passports and the
canonical Gravity registry, identify one deterministic validation gap, and
make at most a small staging-only test improvement. It must not alter
scientific conclusions or touch model loading, GPU, daemon, teacher
authorization, or Pulsar promotion. The prepared packet is
`docs/worker/GROK_BOT_FIRST_QUALIFICATION_TASK.md`.

Permissions: read/search/history; edit only the printed staging worktree and
recorded scope; compile; CPU-only focused tests; prepare a patch/commit; write
evidence. Forbidden by default: production deploy or release promotion,
canonical branch push/force-update/reset/clean/checkout, sudo or secrets,
daemon restart, Kimi/Flash/GPU/TPS/full replay, teacher admission, capability
certification, Gravity promotion, and invocation of another agent.

Rollback/removal: capture the result receipt and review the diff first. Then a
human may remove the exact task worktree with Git's worktree removal command
and delete only the exact worker/<task> branch if it is no longer needed.
Do not remove the canonical checkout, reset it, or use a broad recursive
cleanup. The staging root and its receipt/packet are outside the canonical
checkout and can be archived or removed as a task-specific operation after
review.

## FLASH

The current Flash source-correctness state is unchanged. The authoritative
receipt is receipts/future/FLASH_SOURCE_CORRECTNESS_FRONTIER_20260912.json:

- status: BLOCKED_MISSING_MACHINE_OWNER_AUTHORIZATION_FOR_CAPTURED_EXTERNAL_TEACHER;
- last bounded frontier: layer 4 / token 0;
- captured candidate [271, 248045] is not admitted and has no teacher authority;
- owner-key blocker: the detached machine-admin Ed25519 trust anchor and
  authorization are absent;
- release-candidate status: no release candidate is claimed by that receipt
  (the release_candidate field is null);
- next authorized science action: the machine owner provisions
  /Library/Application Support/Hawking/owner_ed25519_public_key and signs the
  exact existing V2 request; then the existing source-boundary transaction may
  run once at a safe runtime boundary and stop at the first exact difference.

No fake owner key, teacher signature, source/native transaction, full replay,
TPS run, capability-gate change, or Kimi restart was performed. Kimi remains
cold pending its separate authorization boundary. The protected GPU and daemon
state were not used by this bridge. The changed-source successor executable
closure is separately sealed at
`receipts/future/HAWKING_FLASH_SOURCE_BOUNDARY_EXECUTABLE_CLOSURE_PLE_I64_GUARD_20260912.json`;
it is static-only and does not authorize execution.

## KIMI

Kimi is cold, unloaded, and has zero fresh calls. No Kimi/MLX provider child,
GPU lease, runtime restart, or retirement claim exists. Preserve Kimi as a
rollback/history/transfer specimen only.

## UNIFICATION

Canonical owner touched: the Rust hawking CLI, with the external-worker module
at crates/hawking/src/external_worker.rs and its context projection command.
The durable worker brief, packet template, and first qualification packet live
under docs/worker/. The existing Python normal-Grok bridge was not called or
changed. No parallel agent framework, goal system, memory store, scheduler,
daemon, or promotion path was created.

## CODEX

Codex closeout complete. Remaining allowance intentionally preserved. Do not
resume Codex without explicit user authorization. Run the qualification only
when Grok Bot is actually available.
