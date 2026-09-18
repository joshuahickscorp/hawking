# Arsenal release and rollback manifest — 2026-09-15

## Current release boundary

The working tree contains the additive Arsenal contracts. The already-running
`hawkingd` remains the prior loaded process; it was not restarted during this
checkpoint. This preserves the live service and gives a controlled cutover
boundary.

## Staging promotion procedure

1. Run the focused Python and `hawking-web` tests recorded in the status report.
2. Run a disposable daemon fixture using the same API routes and verify the
   candidate manifest, mutation intent reconciliation, source scope proof, and
   bounded event replay.
3. Inspect the worktree diff and process/resource state.
4. Restart or replace the daemon only under the existing Hawking service owner,
   after checkpointing active Goals and confirming no protected Flash/Pulsar
   resource is involved.
5. Re-run `/health`, `/v1/models`, read-only Web/session reads, and the recovery
   probe before enabling new writes.

## Rollback

The safe rollback is a service-owner rollback to the prior known-good daemon
release while preserving `.hawking` intents and receipts. Do not replay an
`UNKNOWN_OUTCOME` mutation; reconcile its recorded hashes first. Keep the
candidate manifests and test receipts for diagnosis. No destructive rollback
command was executed in this dirty worktree, and no commit/push was performed.

## Protected boundaries

Flash/Pulsar artifacts, `/private/tmp/ar-build.log`, the protected Kimi native
temporary tree, provider credentials, and historical Claude sources are outside
this release operation.
