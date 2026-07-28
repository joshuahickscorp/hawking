# Integrate production writers for HIDE's six memory classes

## Goal

In an isolated worktree at current final-ascent HEAD, reconcile and land the
missing production writer wiring for HIDE's six committed memory classes. The
current repository claims lane L20 is integrated, but
`crates/hide-backend/src/classed_writers.rs` and its production call sites are
absent.

This task may use the preserved prior evidence at:

- `~/.claude-grok/tasks/memory-writers-20260726-215341/grok-report.md`
- `~/.claude-grok/tasks/memory-writers-20260726-215341/diff.patch`

Treat the old patch as a claim and source of candidate code, not as something to
apply blindly. Reconcile every hunk against current HEAD and reimplement where
the host has changed.

## Required behavior

- Working memory: write at real turn start/end and clear at the retention
  boundary.
- Episodic memory: mirror real durable turn/tool/edit/verdict events with
  provenance; enforce bounded per-session retention.
- Procedural memory: write only from successful real tool receipts, never model
  prose.
- Semantic-project memory: distill only from explicit successful procedural
  evidence under a deterministic rule.
- User memory: write only from an explicit user memory intent and the correct
  scope.
- Verification memory: write only from verifier receipts with evidence tier;
  model output must not mint this authority.

Use one production capability-mint site per class. Do not weaken type or
construction boundaries. Preserve inspect/correct/forget/export semantics,
scope-promotion audit, and existing migrations.

## Required tests

Add or reconcile deterministic non-skipped tests proving:

1. each real producer writes only its allowed class;
2. failed tools do not write procedural/semantic facts;
3. model turns cannot write user or verification memory;
4. write then compile retrieves episodic and procedural records with provenance;
5. working and episodic retention bounds hold;
6. explicit user writes and verifier writes round-trip;
7. forget/export and capability non-widening regressions remain green.

Run at least:

```bash
cargo test -p hide-backend --lib -- production_ classed_writers --test-threads=1
cargo test -p hawking-context --lib -- --test-threads=1
cargo test -p hide-backend --lib -- --test-threads=1
```

If an existing unrelated host test fails, prove it is pre-existing with the
smallest exact rerun; do not call the suite green.

Update `HIDE_MEMORY_CLASSES.json` only to reflect behavior proven by committed
code and non-skipped tests. Keep any controller-review field honest.

## Forbidden

- Do not touch model weights, MOP, source shards, teacher capsules, runtime/Metal
  hot paths, Fabric/Bridge, or `odyssey/launch/`.
- Do not modify `HIDE_KERNEL_TURN` or any authorization fence.
- Do not turn fixtures into production claims.
- Do not include `.serena` files.
- Do not push, merge, or clean another worktree.

## Required report

Commit intended files on the Grok branch. Report the prior-patch hunks accepted,
changed, or rejected; files changed; exact tests and counts; live producer
traces; remaining ceilings; and the next safe action.
