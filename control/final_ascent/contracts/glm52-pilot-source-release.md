# Build a fail-closed release gate for the sealed five-shard GLM pilot window

Work in an isolated worktree at current campaign HEAD. Implement and test a
small release controller for the exact five BF16 pilot shard bodies under:

`/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity/pilot_source`

The Grok task must **not release or mutate the real pilot source**. It may use
temporary fake directories for tests. The supervising controller will run the
real read-only gate and, only after review, the confirmed release.

## Context and authority

The bounded promotion-valid pilot is sealed by:

- `GLM52_BASIS_PILOT_RECEIPT.json`
- `GLM52_BASIS_PILOT_CONTROLLER_RESEAL.json`
- `GLM52_BASIS_PILOT_REVISION_0_EVIDENCE.json`
- `HAWKING_FINAL_ASCENT_SOURCE_REHYDRATION_RECEIPT.json`

The measurement completed and the source bodies may now be evicted. The
content-addressed rehydration route and exact hashes must survive. Do not touch
teacher capsules, compact artifacts, MOP, HIDE, Odyssey, or any authorization
fence.

## Required tool

Add `tools/condense/glm52_pilot_source_release.py` with separate commands:

```text
gate
release --confirm RELEASE_EXACT_SEALED_FIVE_SHARD_PILOT
status
```

`gate` and `status` are read-only. `release` must rerun the complete gate in the
same process and refuse unless every gate is green and the confirmation phrase
matches exactly.

## Required gates

1. Resolve the exact pilot root without symlinks. It must equal the configured
   absolute root and be below the GLM52Gravity support root.
2. Load the final-ascent rehydration receipt. Its five named shards, byte sizes,
   and sha256 values are the **only** allowed deletion set.
3. Require exactly those five shard bodies to exist as regular non-symlink
   files. Refuse additional `model-*.safetensors` bodies rather than expanding
   the deletion set.
4. Full-hash all five bodies immediately before release and match the sealed
   values. A size-only check is insufficient.
5. Verify the controller reseal:
   - its bound measurement receipt hash matches the live receipt;
   - its revision-0 hash matches the live revision-0 evidence;
   - its reviewed current pilot code and test hashes match the live files;
   - `measurement_math_changed=false`;
   - `full_traversal_authorized=false`.
6. Verify the measured receipt says all five source hashes were verified, no
   Gaussian selection occurred, no parent traversal started, and all three
   authorization fences are false.
7. Require the current final-ascent status/contract surfaces to keep:
   `ODYSSEY_LAUNCH_AUTHORIZED=false`,
   `RAMANUJAN_RESEARCH_AUTHORIZED=false`, and `HIDE_KERNEL_TURN=false`.
8. Scan the live process tree with both `lsof` (when available) and a full argv
   scan. Any process other than this controller that opens, maps, or names the
   pilot root blocks release. If neither probe can establish safety, fail
   closed.
9. Prove path isolation: the five resolved targets cannot contain or equal the
   repo, teacher capsules, compact artifacts, MOP, the support root, or any
   directory. No target may be outside the exact pilot root.
10. Record exact free bytes and deletion bytes before release.

## Release behavior

- Delete only the five exact verified shard files, one explicit resolved path at
  a time. No glob deletion and no recursive removal.
- Retain `REHYDRATE_LEDGER.jsonl`, rehydration logs, `hf_home`, and the pilot
  directory itself.
- After deletion, prove the five bodies are absent and retained evidence still
  exists.
- Write `HAWKING_FINAL_ASCENT_PILOT_SOURCE_RELEASE_RECEIPT.json` atomically in
  the repository with:
  - gate result and per-gate evidence;
  - exact paths, sizes, hashes, and aggregate bytes;
  - before/after free bytes;
  - immutable rehydration repo/revision;
  - hashes of the measurement receipt, controller reseal, revision-0 evidence,
    and current code/tests;
  - all three false fences;
  - a canonical seal sha256 over the receipt excluding the seal field.
- If deletion is partial or receipt publication fails, report exact state and
  fail nonzero. Never claim rollback restored deleted bytes; rollback is
  rehydration by immutable hash.

## Tests

Add `tools/condense/tests/test_glm52_pilot_source_release.py` using only small
temporary fake files. Cover at least:

- green gate and confirmed exact deletion;
- wrong confirmation;
- missing, extra, size-mismatch, and hash-mismatch shard;
- symlinked root and symlinked shard refusal;
- path escape / protected-target refusal;
- stale measurement receipt or reseal binding;
- current-code hash mismatch;
- a simulated live consumer;
- no-process-probe fail-closed;
- release reruns the gate;
- retained ledger/log/cache survive;
- sealed receipt verifies and cannot be replayed as a second successful release.

Run the fake-only suite and `py_compile`. Commit only the tool and test. Exclude
`.serena` and helper scripts.

## Required report

Report files, test count, exact gates, the real command the supervising
controller should run next, remaining risks, and explicitly confirm the real
pilot source was not mutated.
