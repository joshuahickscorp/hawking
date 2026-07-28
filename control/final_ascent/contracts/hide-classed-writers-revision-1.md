# HIDE classed-writer revision 1: fail-safe working-memory lifetime

Revise the existing `final-ascent-hide-classed-writers-20260728-080907`
worktree in place. Preserve the six intended production files and
`HIDE_MEMORY_CLASSES.json`; do not include `.serena`,
`.pending-commit-hide-classed-writers`, `apply-pending-commit.sh`, or build
artifacts.

## Blocking defect

The candidate writes working memory before many fallible operations in both
`run_turn_core` and `run_turn_kernel`, but calls `end_working_turn` only on the
normal success return. Any early `?`, inference failure, compiler failure,
event-log failure, cancellation, or panic-unwind can leave turn-local memory
resident. That violates the required retention boundary.

## Required correction

1. Introduce a small RAII lifetime guard, or an equivalently exhaustive
   structure, so a working-memory row is cleared on every return path after it
   is seeded in both production turn implementations. Prefer a guard whose
   `Drop` calls `ClassedMemorySystem::end_turn` and which owns only the minimum
   `Arc<ClassedMemorySystem>` plus turn id.
2. Do not double-clear manually on the success path unless the guard is
   explicitly disarmed. The simplest acceptable design is to let the guard
   clear on scope exit.
3. Add deterministic non-skipped tests for both the core and kernel-relevant
   lifetime primitive. At minimum, prove:
   - normal scope exit clears the row;
   - an injected/forced early `Err` after seeding clears the row;
   - panic unwind clears the row when unwind is supported.
   A focused guard-unit test is acceptable when both production functions are
   visibly bound to the same guard constructor.
4. Update `HIDE_MEMORY_CLASSES.json` so it does not claim an explicit normal-only
   end hook. State and cite the fail-safe lifetime guard and its tests.
5. Run:

   ```bash
   cargo test -p hide-backend --lib -- production_ classed_writers --test-threads=1
   cargo test -p hawking-context --lib -- --test-threads=1
   ```

   Reuse the existing target directory. Do not run unrelated disk-heavy builds.

## Preserve

- One production capability-mint site per class.
- Episodic retention cap and durable-event mirror.
- Successful-receipt-only procedural writes.
- Explicit deterministic semantic-project distillation.
- Explicit-scope-only user writes.
- Verifier-receipt-only verification writes.
- Existing inspect/correct/forget/export and scope-promotion semantics.
- `HIDE_KERNEL_TURN=false`.

## Report

Report the exact guard design, production binding sites, added early-error and
unwind tests, test counts, six intended source/status files, remaining
ceilings, and a clean commit containing no helper metadata.
