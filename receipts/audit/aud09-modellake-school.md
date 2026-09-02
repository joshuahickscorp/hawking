# aud09 — ModelLake specimen school

Discovery audit only. No implementation. No H-ROADMAP rewrite. No lake writes.

Machine-readable twin: `receipts/audit/aud09-modellake-school.json`.

Evidence labels used: `PHYSICALLY_MEASURED`, `STATIC_VERIFICATION`, `SOURCE_INSPECTION`, `INFERRED`. Nothing here is a GPU/ANE/FPGA measurement.

## Current lake truth (measured this session)

| Fact | Value | Tier |
|---|---|---|
| Root | `/Volumes/corpdrive/hawking-modellake` | PHYSICALLY_MEASURED |
| Specimen directories | **55** | PHYSICALLY_MEASURED (`os.listdir`) |
| Live lake manifests | **55** | PHYSICALLY_MEASURED |
| `partial/` | **does not exist** | PHYSICALLY_MEASURED |
| Regular-file bytes in `specimens/` | **4,352,666,459,500** (4.353 TB) | STATIC_VERIFICATION of catalog `st_size` sum; Flash and Qwen3-0.6B walked independently |
| `TIER2_BUDGET` | 3.5e12 (`tools/odyssey/modellake.py`) | SOURCE_INSPECTION |
| Overage | **852.666 GB** | STATIC_VERIFICATION |
| Architecture families | 39 (9 of them `UNKNOWN`) | STATIC_VERIFICATION of `index/catalog.json` built 2026-09-02T18:58:06Z |
| Catalog `seal_status` | 49 `SEALED`, 6 `UNSEALED` | STATIC_VERIFICATION |
| Lifecycle | 45 `CENSUSED`, 9 `READY_COLD`, 1 `SSD_STAGED` | STATIC_VERIFICATION |
| Live watcher | PID **4183** `modellake_watch.py --poll-secs 0.10` | PHYSICALLY_MEASURED (`pgrep`) |
| In-flight `hf download` | none observed | PHYSICALLY_MEASURED |
| Newest specimen dir mtime | Inkling-Small, 2026-09-02T08:38:53Z (~10 h before audit) | PHYSICALLY_MEASURED |

G010 at 08:00Z this morning still saw live downloaders and a Falcon3 complete-but-unpromoted tree. That is historical. Falcon3, Mistral-Small, Qwen3-14B, Qwen3-Coder and Inkling-Small are specimen directories now.

## Flash is here. The “144-file partial” sentence is false.

`PHYSICALLY_MEASURED` on `specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc`:

- 144 files, 131 safetensors
- 360,023,286,454 bytes — matches `hcli.flash_next.EXPECTED_BYTES`
- newest file 2026-08-27T04:04:03Z
- no corresponding `partial/` tree

H-ROADMAP.md:361 (file lives at `/Users/scammermike/Downloads/H-ROADMAP.md`, **not in this git tree**) still says:

> Flash ModelLake acquisition was last reported as a pinned resumable worker with a 144-file partial tree; atomic final publication was still pending.

Replace with:

> Flash-Next pinned body `Qwen/Qwen3.8-Flash-Next@34567a4712bc` is published under `specimens/` (144 files, 360,023,286,454 bytes, 131 safetensors). No partial tree. Catalog `seal_status=UNSEALED` only because `workspace/campaign/odyssey/watch-manifests/` lacks this slug’s size map; the lake manifest is present. Do not start or restart a Flash acquisition worker.

`hcli/flash_next.py:flash_next_profile` still emits `READY_FOR_EXPLICIT_ACQUIRE` and a zero transfer ledger. `tools/future/odyssey2_law_store.py` still sets `SCHOOLS['Flash']['physical_status'] = "metadata_only_weights_not_present"`. Both are stale. Call sites: `CommandHandler._cmd_flash_next` → `flash_next_profile`.

## Qwen27 remains the first control school — and it is not on the lake

This is a loud surprise.

`propose_specimen_curriculum` hard-codes `QWEN27_SPECIMEN = "qwen3.8-27b-abliterated-bf16@local"` and says in source that the Qwen27 parent “is not a ModelLake specimen and never was.”

`PHYSICALLY_MEASURED`:

- `/Volumes/corpdrive/personalmodel/correspondent/qwen3.8-27b-abliterated-bf16` — 32 files, 55,586,059,478 bytes (51.8 GiB)
- `/Users/scammermike/noetic/NOETIC_PARENT_A` — native parent artifact
- **zero** Qwen3.8-27B directory under ModelLake `specimens/`

Do **not** copy it onto the over-budget HDD. Do **not** erase it as the control school. Doctor / Gravity / Accelerator scars and the protected-baseline contract live on this body.

## Three meanings of “sealed” (do not inflate)

| Authority | Meaning | Count now |
|---|---|---|
| Directory presence | a dir exists under `specimens/` | **55** |
| `modellake_index._file_seals` | git watch-manifest `sizes` exist and match | **49 SEALED / 6 UNSEALED** |
| `modellake_events` detection | lake manifest `resolved_sha`+bytes, joined to `SPECIMEN_REGISTRY` | **8** sealed, **41** “complete_unsealed” (receipt 18:39:54Z) |

The six catalog-UNSEALED slugs (no git watch-manifest) are:

- `Qwen--Qwen3.8-Flash-Next@34567a4712bc`
- `Qwen--Qwen3-30B-A3B@ad44e777bcd1`
- `Qwen--Qwen3-VL-30B-A3B-Instruct@9c4b90e1e4ba`
- `deepseek-ai--DeepSeek-V4-Flash@60d8d70770c6`
- `moonshotai--Kimi-VL-A3B-Instruct@398eede0903c`
- `tiiuae--Falcon-H1-7B-Instruct@41e72f27effb`

The event consumer’s 8 sealed ids are those six **plus** Qwen3-0.6B and Mistral-Small. Catalog SEALED (49) is mostly event `complete_unsealed`. Opposite directions. Pick three booleans (`published`, `watch_seal`, `hash_seal`) and stop using one word.

Git watch-manifests: 55 files = 49 lake-matching + 6 orphans (`blt-7b`, `gemma-3-4b-it`, `Llama-4-Scout`, `personaplex-7b`, `stable-audio-open-1.0`, retired `GLM-4.5`). Retired lake manifest only: `manifests/retired/zai-org--GLM-4.5@cbb2c7cfb52f.json`. No retired body under `specimens/`.

## Hash verification is not a 55-specimen fact

`MODELLAKE_HASH_VERIFIED` (09:14Z, verdict BLOCKED): oid-hashed **1/55** (canonical Qwen3-0.6B, 10/10). Size-complete 49/55; the six without watch size maps were `size_unknown`. This lane did not re-hash 4.35 TB (`BLOCKED_AUTHORITY`).

Tension, flagged not averaged: `SPECIMEN_CURRICULUM.json` at 08:27Z claims Flash `n_sha256_verified=144`, `bytes_hashed=360023286454` via `specimen_verify.py`. That is a different producer than the acceptance gate. Do not report the lake as hash-sealed.

## HCLI visibility — callable, Flash-shaped, not a 55-school UI

Every row has a **call of the symbol**, not an import.

| Surface | State | Actual caller |
|---|---|---|
| `specimens.registry` tool | CALLABLE, TESTED (tmp_path + cheap live enum `n>=40`) | `hcli/tool_registry.py:_specimens_registry` → `specimens.registry()`; also `tools/sovereign/g009_reachability.py` FRONTIER |
| `/flash-next` | CALLABLE, stale acquire copy | `CommandHandler._cmd_flash_next` → `flash_next_profile` |
| `agentos_cli modellake-census` | CALLABLE, Flash-pinned | `run_modellake_census`. Latest HCLI census 2026-09-02T00:14:32Z already has `final_present=true` but `target_not_published_as_verified=true` because it will not hash 360 GB |
| `agentos_cli modellake-supervise` | CALLABLE, **Flash slug hardcoded** | `run_model_lake_supervision`. Supervision receipt: 144 files / expected bytes at `specimens/<Flash>` |
| status line `modellake=` | CALLABLE | `_modellake_status` tails the watcher jsonl (jobs/remaining), not the school |
| `/models` | CALLABLE, **not the lake** | `discover_models` walks `~/models`. That directory currently has no GGUF/MLX |
| `acquisition.propose` | CALLABLE | tool handler → `acquisition.propose()`. New acquires would hit `modellake.admit` over-budget refusal |

`hcli/specimens.py` docstring still says **47** sealed specimens. `hcli/test_specimens.py` still says 47 / ~3.4 TiB. Registry reads **git watch-manifests**, so the six lake-not-in-watch slugs come back `verified_complete=None`.

No HCLI slash command lists the 55-body school. The enumerator exists as a tool.

## Curriculum eligibility

First-wave roles (single authority `tools/future/specimen_curriculum.py:propose_specimen_curriculum`), receipt 08:27Z, **5/5 ready**, 51 lake extras `not_proposed`:

1. `very_small_dense_procedural_speed` → Qwen3-0.6B (also SSD-staged, 10 files under `~/noetic/stage/`)
2. `small_dense_alternate_architecture_transfer` → Falcon-H1 (body present, catalog UNSEALED)
3. `mid_size_dense_compiler` → Mistral-Small-3.1-24B
4. `qwen27_mature_physical` → authorized external tree, **not** lake
5. `flash_heterogeneous_frontier` → Flash body present, catalog UNSEALED

Callers: `odyssey_launch._eval_curriculum`, `tools.future.consolidated_run`. **Zero HCLI callers.** TESTED under `tools/future/test_specimen_curriculum.py`. Not INTEGRATED as a 39-family school.

§14.3 diversity roles **present** on the lake: dense decoder, MoE, hybrid recurrent/state-space, long-context, multimodal, extreme expert count, alternative tokenizer, very-low-bit published checkpoint, state-heavy architecture.

**Missing:** native MTP/speculation, structured sparsity, codebook/additive quantization, new Apple-friendly runtime specimen. Flash `config.json` is `qwen4_exp` with `vision_config` but catalog `role_primary` is still `dense decoder`.

Kimi-K3 is **1,560,998,984,390 bytes (~1.45 TiB, ~36% of the lake)** and is not a first-wave role. That is a surprise. Do not load it casually.

## Event integration

END_TO_END, measured:

- Watcher jsonl `2026-09-02T18:39:54Z` `modellake_events_run` `n_new_seal_specimens=8`
- Producer: `modellake_watch.emit_modellake_events_once` → `modellake_events.build` / `consume`
- `receipts/future/MODELLAKE_EVENTS.json` `recorded_at` matches; `is_this_wired=true`; 8 specimens × 6 triggers = 48 units

`receipts/future/MODELLAKE_SCHEDULER_VIEW.json` still says `seal_contract.is_this_wired=false`. That receipt is a stale reader. Do not treat it as the producer.

After the events run, the same watcher emitted `probe_boom_error` / `sweep exploded` at 18:47, 18:49, 18:58. Not diagnosed this lane.

## Storage destinations

| Role | Path | Now |
|---|---|---|
| TIER2_COLD | `.../specimens/` | 54 of 55 (Qwen3-0.6B catalogued TIER1_HOT because it is also staged) |
| TIER1_HOT | `~/noetic/stage` | Qwen3-0.6B complete (10 files); four stub-like dirs (`falcon-h1-7b`, `kimi-vl-a3b`, `qwen3-30b-a3b`, `qwen3-vl-30b-a3b`) |
| PARTIAL | `.../partial/` | absent |
| GIT_METADATA | lake `manifests/` + `index/`; git `watch-manifests/` | 55 lake manifests, catalog+by-slug index, 55 git watch files |
| Authorized external | `.../personalmodel/correspondent/qwen3.8-27b-abliterated-bf16` | Qwen27 control school |

Index law (LAYOUT.md, catalog `layout.roles.specimens.writable_by_index=false`): the index never writes, moves, deletes or compresses `specimens/`. Observed: a concurrent `python3 -m tools.odyssey.modellake_index build` (PID 78096, not this lane) rewrote `index/catalog.json` 18:53Z → 18:58Z. `anomalies.orphaned_watch` flipped from 6 ids to `[]` across that rebuild. Git watch-vs-lake set-diff is the stable fact.

Retention recommendation (`does_not_retire=true`, `operator_decision_only=true`): covering set ~143 GiB of family representatives; 14 ranked-redundant-bulk rows. That is **not** a delete list. Do not retire, move, or delete anything.

`modellake.admit` will refuse any new acquire (`used + nbytes > TIER2_BUDGET`). MODE-007 is capacity-blocked. Do not start downloads.

## Retained verified bytes per wall second

- **Historical, MEASURED** (not this lane): G010 08:00Z, 660 s window, 60.53 MB/s while Qwen3-14B and Qwen3-Coder were still growing.
- **Now, INFERRED**: ~0. No `partial/`, watcher `active_remaining_bytes=0`, `idle_rearm_wait` 1800 s, newest body 10 h stale. This lane did not run a two-point G010 rescan.

## Invalidated roadmap / producer assumptions

Full list with replacement sentences is in the JSON `invalidated_roadmap_assumptions`. The ones that must not keep standing:

1. **Flash 144-file partial, publication pending** (H-ROADMAP §3). False since at least 2026-08-27.
2. **“CURRENT TWO-QWEN SCHOOL” as the specimen school.** The lake is 39 families / 55 bodies. Qwen27 is not on it.
3. **`SCHOOLS['Flash'].physical_status = metadata_only_weights_not_present`.**
4. **`flash_next_profile` `READY_FOR_EXPLICIT_ACQUIRE`.**
5. **`hcli/specimens.py` / `test_specimens.py` “47 specimens / 3.4 TiB”.**
6. **Census check name `target_not_published_as_verified` read as “Flash is absent”.** The body is present; oid hash is not.
7. **Scheduler-view `is_this_wired=false` as current wiring truth.** The consumer ran today.
8. **`SPECIMEN_REGISTRY.json` n=54 (2026-08-31).** Query `index/catalog.json`.
9. **FLAS-001 “verify next completed shard” as an open download.** 131/131 safetensors are here.
10. **G010 live-downloader defects as current lake state.**
11. **First-wave five roles as the entire school.** 51 residents are now the deferred mass, not a footnote.

## What should replace two-Qwen / Flash-only schooling language

Keep Qwen27 as Textbook #1 / first control school. Keep Flash as the heterogeneous / Noetic executable school, now that the source body is published.

Name a **separate** ModelLake specimen school:

- Control pair (not the whole school): Qwen27 (external) + Flash (lake, watch-manifest missing).
- First-wave five roles: still useful; do not lower `n_roles`.
- Second-wave, only because the bodies are already here (proposal, not an implementation):
  - Qwen3-Coder-30B-A3B — MoE + code, watch-SEALED
  - DeepSeek-V4-Flash — extreme expert count, catalog UNSEALED
  - Qwen3-VL-30B-A3B — VL MoE, catalog UNSEALED
  - Kimi-K3 — 1.45 TiB multimodal; residency economics before any load
  - GLM-5.3-Flash — 306 GiB dense
  - RWKV7 — hybrid recurrent
  - BitNet 1.58 — published very-low-bit
  - Mamba3 — catalog UNKNOWN family; fills state-space
  - Dream or iLLaDA — diffusion-language novelty
  - Kimi-Linear-48B-A3B — linear attention MoE
- Do not pretend native MTP/speculation, structured sparsity, codebook quantization, or a new Apple-friendly runtime specimen are filled. They are not in catalog `roles`.

HCLI: keep `specimens.registry` as the enumerator; stop presenting `/flash-next` and `modellake-census` as if the lake were one Flash acquire. Do not make `/models` walk 4.35 TB.

## Constitution (held, not proposed)

Exactly five eras. Exactly three Odysseys. No Era VI. FPGA stays inside Accelerator/Fusion. Theia is one generalist bounty model. 0.7% civilizational coordinate not rewritten. North star unchanged.

## Rust

`ABSENT` as a ModelLake school. `git grep modellake` over `crates/` hits Flash kernel example names, not a specimen registry.

## Concurrent interference (flagged)

Another session ran `python3 -m tools.odyssey.modellake_index build` (PID 78096) during this audit. Catalog `built_at` moved. This lane did not start it, did not kill it, did not write the lake.

## Acceptance

- `receipts/audit/aud09-modellake-school.json` exists, valid JSON
- Every capability has a call site of the symbol or an explicit blocker
- No H-ROADMAP rewrite, no `hcli/` / `tools/` / `crates/` / `civilization/` edits, no specimen mutation
