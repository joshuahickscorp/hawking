# Hawking storage reclamation

Date: 2026-09-15
Scope: local artifact audit and reversible off-SSD retention moves

## Guardrails

- The current Hawking source and receipts remain authoritative.
- The running `hawkingd` and `hawking-gravityd` processes remain untouched.
- `/private/tmp/ar-build.log` remains protected, whether or not its writer is
  currently open.
- Source, active campaign state, receipts, dirty worktrees, Flash/Pulsar
  artifacts, and native-runtime boundaries are not cleanup targets.
- The selected hero artifacts are moved to the mounted `corpdrive` retention
  area. The user then explicitly narrowed the policy: discard Kimi scratch and
  all non-hero TTS variants, keep only the selected heroes, and leave Flash
  Next in place. Hawking source, receipts, and active work are not cleanup
  targets.

## Baseline

The audit measured approximately 199 GiB available on the system volume and
approximately 205 GiB for the `~/Downloads/hawking` tree. `corpdrive` was
mounted with approximately 421 GiB available.

## Retention decisions

### Kimi

Selected hero:

- `/Users/scammermike/hawking-agents/kimi-vl-a3b-q8`

This is the exact local snapshot named by the current
`KIMI_P0_OPERATIONAL_BOUNDED_FREEZE` and related Hawking receipts. It is the
canonical P0 artifact. It will be relocated to external hero storage with a
compatibility symlink at the current path so existing receipts and resolvers
continue to resolve it.

Delete as reproducible scratch:

- `kimi-pq-d8k1024` — information-content experiment, not P0
- `kimi-vl-a3b-p1-mixed48` — unqualified P1 derivative
- `kimi-vl-a3b-p2-route4` — unqualified route derivative
- `kimi-vl-a3b-p1-mixed48.failed-metal-timeout-20260913` — failed, empty
  artifact directory

### TTS

There is no checkpoint directory literally named `collapsed`. The current
`tts-bench/hawking/FINDINGS.md` records that q4 g64 is the best-size-at-zero-
quality-cost result (8/8, WER 0.000, spoken 1.00), while the 2-bit families
collapse and score 0/8. Therefore keep only:

- `/Users/scammermike/tts-bench/ckpt/higgs-audio-v3-4b-q4`

Relocate that hero to external storage with a compatibility symlink at the
current checkpoint path. Every other checkpoint directory, including q8 and
the failed/experimental 2-bit, 3-bit, AWQ, descriptor, rotation, sensitivity,
and mixed variants, is reproducible scratch and is discarded. The TTS source,
scripts, findings, smoke outputs, and job records remain.

## Action log

The destination is:

`/Volumes/corpdrive/hawking-retained/storage-reclamation-20260915/`

On 2026-09-15, the four Kimi derivative directories were temporarily moved
there (about 51.1 GiB) and then explicitly discarded as scratch. The partial
TTS transfer was stopped before completion and its non-hero partial tree was
discarded. All 44 non-q4 TTS directories were deleted from the source
checkpoint root. The selected heroes now live under `heroes/`; their original
canonical paths are compatibility symlinks. Measured system-volume free space
is approximately 361 GiB after the model and non-Flash target cleanup.

The inward Hawking audit found a separate set of non-Flash generated Cargo
scratch directories under `target/` with no open handles. Only those exact
named outputs were reduced; all `flash-*` output and the live release/runtime
boundary remained protected.

## Deferred high-value candidates

These require a separate active-handle/reference check before moving:

- generated Hawking Cargo output under `workspace/ops/build/rust`;
- clean versus dirty worktree-local generated outputs;
- the corrupted Mathlib `.lake` build/dependency tree;
- completed Noetic artifacts whose current receipts still name their SSD
  paths;
- stale temporary directories other than the protected active build log.

The following generated non-Flash target directories are the approved inward
cleanup set (about 21 GiB total):

- `target/hawking-dull-pass-baseline`
- `target/hawking-dull-pass-attention-trace`
- `target/hawking-dull-pass-bridge`
- `target/hawking-attention-trace-compile`
- `target/hawking-diagnostic-core-rust-20260914`
- `target/task033`
- `target/source-boundary-ple-i64-guard-20260912`

That exact target cleanup set has now been deleted and verified; the remaining
top-level target directories are the named `flash-*` outputs.

The separate unreferenced Rust build caches `workspace/ops/build/rust/debug`
(about 77 GiB) and `workspace/ops/build/rust/release-fast` (about 7.5 GiB)
were deleted after their no-handle check. The live daemon's
`workspace/ops/build/rust/release` directory is retained.

Additional approved inward reductions are:

- local `artifacts/nx/Qwen--Qwen3-0.6B@c1899de289a0` and
  `artifacts/nx/tiiuae--Falcon-H1-7B-Instruct@41e72f27effb`, both marked
  `nr_runtime_referenced=false` and reproducible from the external ModelLake;
- the clean, historical `.worktrees/hawking-unification-implementation`
  worktree (about 22 GiB), removed through Git while retaining its branch and
  commit.

The dirty Flash/Kimi worktree under `/private/tmp/hawking-kimi-p1-native` is
not a cleanup target.

The live campaign tree is retained: the watcher owns `workspace/campaign/odyssey`
and current tools/receipts reference `records`, `phaseB`, and model evidence.
The remaining approved low-risk inward trims are the rebuildable Mathlib
`.lake/build` and `.lake/packages` outputs, plus the clean historical
non-Pulsar worktrees `hawking-collapse-diagnostic` and
`kimi-p0-blind-stream-session`; their Git branches remain authoritative after
worktree removal.

These internal reductions were verified on 2026-09-15. Local free space is
approximately 490 GiB. The resulting external retention root contains only the
two selected heroes; no temporary derivative-transfer directories remain.

## Retained artifact manifest

| Artifact | Decision | Reason |
|---|---|---|
| `hawking-agents/kimi-vl-a3b-q8` | hero, externalized | canonical Kimi P0; compatibility symlink retained |
| `tts-bench/ckpt/higgs-audio-v3-4b-q4` | hero, externalized | validated compact TTS; compatibility symlink retained |
| `corpdrive/.../heroes/kimi-p0` | retain on corpdrive | only selected Kimi |
| `corpdrive/.../heroes/tts-q4` | retain on corpdrive | only selected TTS |
| Hawking source, receipts, campaign state | retain on SSD | authoritative/active |
| Flash/Pulsar/native-runtime artifacts | retain | protected boundary |
| `/private/tmp/ar-build.log` | retain | explicitly protected active build log |
