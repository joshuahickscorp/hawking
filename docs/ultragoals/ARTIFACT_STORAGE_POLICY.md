# Artifact storage policy

Git = knowledge. Local = bulk evidence.

This is the durable rule for what the Hawking repository tracks, what it
must not, and where bytes live when they are too large to be knowledge.
Numbers and family classifications are measured in
`receipts/headless/GIT_STORAGE_LEDGER.json`. This document is the law;
the ledger is the census.

S020 §26: Git LFS is not a magic fix — installing LFS does not remove old
blobs from history. S020 §27: a history rewrite is PREPARED only
(`executed: false` in the ledger). This lane does not rewrite, force-push,
delete branches, or run `git gc`.

Do not invent a second artifact system. Consolidate.

---

## What Git tracks

Git is the knowledge plane. Track:

- Source (`crates/`, `tools/`, `research/lab/`, `app/`, `src/`, `hcli/`,
  `workspace/vendor/` absorbed tracks).
- Configs, schemas, generated **small** ABI/JSON surfaces that are the
  contract (adapter registry, protocol goldens).
- Canonical docs (`docs/`, including this file and `docs/ultragoals/`).
- Compact manifests, receipts, ledgers, digests under `receipts/headless/`
  and other compact `receipts/*.json` that a later reader needs to
  understand a decision.
- Negative science (the failure, not the 50 MiB dump that produced it).
  Compact negative science stays in git; the bulk that produced it does not.
- Compact campaign state that is knowledge:
  `workspace/campaign/odyssey/ODYSSEY_STATE.json` (this lane does not
  modify the odyssey tree; the rule is still: the state file is knowledge,
  the unbounded `RUN_LOG.jsonl` is not).
- Small test fixtures that already live in git (frankenstein
  `latent_v0_checkpoints/` `.pt` files, `crates/hawking-core/tests/fixtures/`
  including the already-tracked `*.bin` PQ fixtures). New weight-sized
  `.bin` files are not fixtures.
- `research/hawking-experiments/` as a **source archive** of paused campaigns
  (README + future compact notes), never as a blob dump.

If a file is a sentence a later worker must read, it belongs in git. If it
is a tensor, a capture plane, a log that grows without bound, or a
reproducible binary, it does not.

## What Git must not track

Git is not a bulk store. Do not add:

- Model weights and packed bodies: `*.safetensors`, `*.gguf`, `*.bin`
  (weights), `*.hq80seg`, `*.hq38seg`, `*.hqseg`, `*.hq30g`, `*.hgravs01`.
- Activation captures: `*.f16`, large `*.npy` capture indexes, residual
  planes under physical capture dirs.
- Training checkpoints: `workspace/campaign/phaseB/ckpt/`,
  `workspace/campaign/phaseB/capture_diverse/`,
  `workspace/campaign/phaseB/capture_diverse2/`. The small frankenstein
  fixtures stay; the heavy checkpoint directories do not.
- Repeated raw benchmark logs and unbounded run logs:
  `workspace/campaign/odyssey/RUN_LOG.jsonl`, `*.log`, `runs/`,
  `bench_out/`.
- Generated executables and compiler output: `/target`, `**/target/`,
  `target-parallel/`, `target-fast/` (the latter was committed once;
  never again), `*.metallib`, `*.air`.
- Runtime / Metal caches, Python/JS virtualenvs, `node_modules`,
  `__pycache__`, `/workspace/ops/build/`, `/workspace/ops/local/weights/`,
  `/workspace/ops/local/checkpoints/`, `/workspace/ops/local/models/`.
- Huge traces (`receipts/dsv4f_fullseq_capture_*/traces/` is the current
  offender in HEAD — ~476 MiB of ~5.9 MiB JSON files). Git keeps a compact
  receipt plus a content hash; the bytes go to the local store.
- Receipt archive tarballs: `receipts/**/*.tar.xz` (extracted receipts
  stay).
- `*.npz`, `*.parquet` calibration shards.
- The 52 MiB GLM52 shard graph:
  `/workspace/campaign/evidence/models/glm52/GLM52_SHARD_DEPENDENCY_GRAPH.json`.

Physical gravity payload bodies have always been local:

```
workspace/campaign/records/ascension-sandbox/physical/**/selected-payloads/
workspace/campaign/records/ascension-sandbox/physical/**/*.hgravs01
workspace/campaign/records/ascension-sandbox/physical/**/tensors/
workspace/campaign/records/ascension-sandbox/physical/**/capture-result.json
workspace/campaign/records/ascension-sandbox/physical/**/hidden/
workspace/campaign/records/ascension-sandbox/physical/**/x/
workspace/campaign/records/ascension-sandbox/physical/**/layer_meta/
workspace/campaign/records/ascension-sandbox/physical/**/residual/
workspace/campaign/records/ascension-sandbox/physical/**/checkpoint.json
workspace/campaign/records/ascension-sandbox/physical/**/source-bf16-capture-reservoir-*/
```

A later `git add -A` must not be able to commit a 5.9 GiB candidate.

## Reconciliation with `.gitignore`

The policy is the `.gitignore` already in HEAD, including the session
block landed in `8ad51461a` ("Stop tracking large local artifacts; keep
them on disk"). This document does not contradict those rules.

Required rules that must remain (also listed in the ledger under
`gitignore.must_hold`):

- `/artifacts/`
- `*.safetensors` `*.gguf` `*.bin`
- `*.hq80seg` `*.hq38seg` `*.hqseg`
- `**/ranspack/out/segments/` `**/quality-candidates/**/segments/`
- `*.f16`
- `workspace/campaign/phaseB/ckpt/`
- `workspace/campaign/phaseB/capture_diverse/`
- `workspace/campaign/phaseB/capture_diverse2/`
- `workspace/campaign/odyssey/RUN_LOG.jsonl`
- `receipts/**/*.tar.xz`
- `*.npz` `*.parquet` `*.metallib` `*.air` `*.log`
- `target-parallel/`
- `/workspace/ops/local/weights/` `/workspace/ops/local/checkpoints/` `/workspace/ops/local/models/`
- physical payload globs listed above
- `/workspace/campaign/evidence/models/glm52/GLM52_SHARD_DEPENDENCY_GRAPH.json`

`.gitignore` does not untrack. Files already in HEAD that match a rule
(PQ `*.bin` fixtures, a couple of `*.log` under superwave, a csv under
`crates/hawking-core/reports/`) stay until a later lane runs a careful
`git rm --cached` on the ones that are not KEEP_GIT fixtures. This lane
does not do that for anything except by documenting it.

Recommended **later** additions (not applied here; `.gitignore` is
outside this lane's write set):

- `*.hq30g` (defense in depth; `physical/**/tensors/` already covers the
  live path).
- `receipts/dsv4f_fullseq_capture_*/traces/` after those files are copied
  to the CAS and a compact manifest is in git.
- `target-fast/` (sibling of `target-parallel/`).

Do not remove `*.bin` from `.gitignore` to "save" the PQ fixtures. They
are already tracked; the rule prevents new weight `.bin` files.

## Existing stores — consolidate, do not multiply

| System | Role | This policy |
|---|---|---|
| `/artifacts/` (gitignored) | **EXISTING Hawking artifact root.** Calibration parquet/npz, learned heads, per-tensor configs. | **CAS root.** Layout below. Do not invent a parallel root or a second sha256 store. |
| `workspace/ops/local/` | Local weights/checkpoints/models, HF cache helper. | Machine-local parent weights. Not the experiment CAS. |
| `research/hawking-experiments/` | Campaign **source** archive (git). README only at HEAD. | Notes and compact pointers, never weights. |
| `receipts/headless/ARTIFACT_LEDGER.json` + `tools/headless/storage_manager.py` | Census/reclaim of ≥1 GiB files under `~/models`, HF hub, campaign `runs/`. | Different layer (disk weights, not git history). Keep. Never-delete classes still bind. |
| `crates/hawking-core/src/artifact.rs` | Gravity shard codec (`GRAVITY\0`). | A file format, not a store. |
| `.hide/blobs` | HCLI runtime blobs. | Leave it. |
| `visionmcp/artifacts` | Other product; `visionmcp/` is gitignored. | Leave it. |

There is no `artifacts/sha256/` tree on disk yet. That is not a second
system — it is the layout under the existing `/artifacts/` root.

## Content-addressed artifact store

The local store is content-addressed: bytes named by sha256, git holds the
manifest. Root: `artifacts/` (already gitignored).

Layout:

```
artifacts/sha256/ab/abcdef0123...   # first two hex chars / full lowercase sha256
```

Put **bytes** there. Put the **manifest** in git (usually
`receipts/headless/` or beside the producer). Git never stores the bytes.

Manifest fields (required):

| field | meaning |
|---|---|
| `artifact_id` | Stable id (`family:experiment:name` or a ULID). |
| `sha256` | Hex digest of the bytes. |
| `size` | Byte count. |
| `kind` | `weights` `trace` `log` `capture` `checkpoint` `tarball` `other`. |
| `producer` | Tool/commit that wrote it. |
| `experiment` | Experiment / obligation / organ id. |
| `created_at` | UTC ISO-8601. |
| `source_identity` | Parent model identity (path + sha/head-tail + size) and git HEAD. |
| `required-for-reproduction` | `true` if a later worker cannot replay without these bytes. |
| `disposable` | `true` if the ARTIFACT_LEDGER/storage_manager may consider it (still subject to never-delete classes). |
| `path` | Logical name (`artifacts/sha256/ab/abcdef…` and/or the well-known live path). |

CAS write rule: hash first, store at `artifacts/sha256/{sha[:2]}/{sha}`,
then write the manifest in git. Never the reverse. Identical bytes
collapse to one object.

Well-known live paths (for example
`workspace/campaign/odyssey/RUN_LOG.jsonl`) may remain as the append
location. A **sealed snapshot** of that file is what goes into the CAS.
Do not fork a second live log.

`required-for-reproduction=true` artifacts are never candidates for
`storage_manager.select`. That joins this policy to the existing
never-delete mechanism rather than replacing it.

## RUN_LOG compaction

`workspace/campaign/odyssey/RUN_LOG.jsonl` is a log, not a deliverable.
It is already gitignored. The live 66 MiB file on the main worktree is
untouched. 55,625 events. Schema `hawking.odyssey.run_log.v1`.

| plane | what |
|---|---|
| Git | Compact canonical summary: schema, experiment identity, event count, byte count, sha256 of the full log (or of a sealed snapshot), first/last event timestamps, verdict histogram, hash of `ODYSSEY_STATE.json`. |
| Local (ignored) | The raw JSONL at the well-known path. Optional sealed snapshot in `artifacts/sha256/…`. |

Do not put a second full copy under `research/hawking-experiments/` or
`receipts/`. Do not LFS the log (S020 §26). History still contains 844
unique blobs (~28.8 GiB logical, ~1.5 MiB zlib disk); stripping them is
Phase B of the **unexecuted** plan in the ledger, not a task for this
lane.

## History rewrite (prepared, not executed)

See `receipts/headless/GIT_STORAGE_LEDGER.json` →
`history_compaction_plan` (`executed: false`).

Measured this session, not the stale 32 GiB figure:

- Shared `.git` is ~5.4 GiB (`git count-objects -vH` pack 5.38 GiB).
- The 32 GiB → 5.4 GiB drop from deleting 899 dead `grok/*` branches is
  already in the object store (one pack, `prune-packable=0`, fsck
  unreachable empty). This lane does not `git gc`.
- `*.hq80seg` / `*.hq38seg` / `*.f16` are **ABSENT** from reachable
  history. The 20 GiB of segments named in `8ad51461a` is gone with those
  branches.
- Remaining pack bulk is `*.hq30g` (~3.91 GiB disk, one 2026-08-09
  commit), plus committed `target-fast/` / `target-parallel/`, physical
  JSON dumps, npy, hgravs01, the q30 tarball, historical `.pt`.

Phase A of the plan would strip those bulk families. The ledger's
upper-bound savings (sum of `unique_disk_bytes`) predict ~0.13 GiB
remaining; delta bases can pin objects and cut the realised saving, so
treat 0.4–1.5 GiB as the conservative band until a rewrite is actually
measured. Phase B strips `RUN_LOG` (~1.5 MiB disk, high SHA-breakage
cost). Phase C fullseq traces only after a CAS copy.

Any rewrite of `801c98b67` rewrites every descendant on HEAD (~1492
commits measured). It would move `origin/main`, `origin/odyssey-i`, 76
tags, and every containing branch, and would require a force-push. This
lane does not do that.

Rollback if a later lane is ever authorised: `git bundle create … --all`
and `git clone --mirror` of the common dir **before** any filter-repo
run. Restore from the bundle/mirror. The 2026-07-01
`.git/filter-repo/commit-map` is not a rollback.

## LFS

Do not adopt Git LFS as the storage architecture. S020 §26: LFS pointers
on new commits leave every historical blob in the pack. The ledger
classifies `research/hawking-experiments/superwave/g1/` as `LFS_CANDIDATE` because that is
the size of thing people reach for LFS for (largest current HEAD blob is
`g1_functional_exceptions.json` at 29.0 MiB). The action is still: local
CAS + git manifest, not LFS.

## Machine-local weights (different layer)

`tools/headless/storage_manager.py` and `ARTIFACT_LEDGER.json` govern
≥1 GiB files on this machine (`~/models`, HuggingFace hub, campaign
`runs/`). Never-delete classes (`KEEP_ACTIVE_PARENT`,
`KEEP_UNIQUE_SCIENCE`, `UNKNOWN_DO_NOT_DELETE`, …) still bind. This
policy does not authorise deleting the production Qwen parent or anything
under `receipts/`. `runs/` is gitignored — a KEEP_LIST in a receipt is
not a mechanism; `storage_manager.assert_deletable` is.

## ABSENT

Anything unmeasured is ABSENT with a reason. In this census:

- Reachable `*.hq80seg` `*.hq38seg` `*.hqseg` `*.f16` `*.safetensors`
  `*.gguf` blobs: ABSENT from `git rev-list --all`.
- `artifacts/sha256/` on disk: ABSENT (layout adopted, directory not
  created by this lane).
- Unreachable objects in the pack: measured empty this session.
- A second CAS root: must remain ABSENT.
