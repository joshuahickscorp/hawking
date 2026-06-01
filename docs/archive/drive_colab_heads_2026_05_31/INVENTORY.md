# Google Drive — dismantle Colab export inventory (2026-05-31)

Provenance record of the **dismantle** Google Drive footprint, captured before
deletion. All content was Colab training output (Eagle / spec-decode heads +
frozen caches + manifests/eval), owned by `joshuahicksboba@gmail.com`. This
doc is the complete record of *what existed and how big it was*; the small
text records (manifests, leaderboards, `tau`/eval JSONs, training logs, the one
`.md` summary) are archived next to this file under the three subdirs. The
multi-GB `.safetensors`/`.npz` weights themselves were **not** archived locally
(too large for the Drive connector) — they were pulled via the Drive app and/or
deleted per the cleanup decision below.

## Totals

| Top-level folder | Created | Size | Files | Folders |
|---|---|---|---|---|
| `dismantle/` → `dismantle_export/` | 2026-05-27 | 17,993,135,309 B (**18.0 GB**) | 102 | 61 |
| `dismantle_final_push/` | 2026-05-29 pm | 48,726,865,321 B (**48.7 GB**) | 73 | 31 |
| `dismantle_headbank_corrected/` | 2026-05-29 am | 7,973,597,906 B (**8.0 GB**) | 17 | 6 |
| **Grand total** | | **74,693,598,536 B (≈ 74.7 GB / 69.6 GiB)** | **192** | **~98** |

These three folders were the **entire** dismantle footprint on the Drive
(confirmed by `title contains 'dismantle'` + `fullText contains 'dismantle'`
searches — the only other hit was an unrelated doc, `dustdevilexplication...`).

## Generation timeline (why there is overlap)

The three folders are successive export generations of the same headbank work,
not three independent datasets:

1. **`dismantle/`** (05-27/28) — original export: `headbank_500u_v2` +
   `maximal_spec_500u`.
2. **`dismantle_headbank_corrected/`** (05-29 am) — a "corrected" re-export of
   the headbank. Several heads are byte-size-identical to the originals
   (q3b 1,563,461,128 B; q1p5b 1,112,827,392 B; q05b 605,568,472 B) → largely
   duplicates `dismantle/`.
3. **`dismantle_final_push/`** (05-29 pm, newest) — the authoritative set:
   `best_heads` (curated sweep winners) + `sweep` (raw hyperparameter sweep) +
   `frozen_cache`. `best_heads` head sizes match selected `sweep` winners → the
   sweep is the raw material `best_heads` was distilled from.

## Local backup status (at capture time)

Only **one** of these heads existed on local disk:
`checkpoints/eagle5_final/q3b/head_final.safetensors` (~1.7 GB) +
`artifacts/eagle5/qwen3b_frozen.npz`. Every other head — **q7b, q1p5b, q05b,
dsv2**, the full sweep, the curated `best_heads`, and the frozen caches — existed
**only on Drive**. Re-creating any of them requires re-running Colab.

---

## Tree 1 — `dismantle/` → `dismantle_export/` (18.0 GB)

### `maximal_spec_500u/` — 11.22 GB
- `heads/q1p5/` — **8.36 GB** (7 `.safetensors`): the `01..03_q1p5_*` configs +
  4 `overeng_q1p5_*` curriculum/calib heads (each 1.11–1.41 GB).
- `artifacts/` — 2.86 GB: `q3b_ref_frozen.npz` (1.24 GB), `q1p5_frozen.npz`
  (0.93 GB), `q0p5_frozen.npz` (0.55 GB) + AWQ JSON.
- `eval/q1p5/` — 273 KB: 19 run subfolders, each `frontier.json` + `tau.json`.
- `metadata/` — 60 KB: leaderboard/summary/progress JSON + `maximal_spec_summary.md`.
- `mine/` — 0.44 MB: 3× `mine_manifest.json`.
- `runtime_profiles/` — 4.6 KB: 2 JSON.
- root: `manifest.json` (26.6 KB), `manifest_overengineer.json` (10.2 KB).

### `headbank_500u_v2/` — 6.77 GB
Each q-bucket = one head + AWQ + leaderboard/runtime/eval JSON.
- `q7b/` — 2.88 GB (head 2.85 GB; `awq_smoothing.json` ≈ 30 MB — the single
  largest non-weight file, **not** text-archived).
- `dsv2/` — 1.70 GB (head 1.69 GB). **Unique to this folder** — no dsv2 head
  exists in `final_push` or `headbank_corrected`.
- `q3b/` — 1.59 GB (head 1.56 GB).
- `q05b/` — 0.61 GB (head 0.61 GB).
- root: `export_manifest.json` (10.5 KB), `headbank_manifest.json` (4.2 KB).

## Tree 2 — `dismantle_final_push/` (48.7 GB)

### `sweep/` — 37.07 GB (64 files) — raw hyperparameter sweep
Per config: `tau.json` + `head_final.safetensors` + `latest.npz` + `log.jsonl`,
across `b1/b2 × e12/e14` configs per bucket.
- `q7b/` 19.82 GB · `q3b/` 8.60 GB · `q1p5b/` 5.77 GB · `q05b/` 2.87 GB
- Largest single files: q7b b2 heads 3.71 GB ×2, q7b b1 heads 2.94 GB ×2,
  q7b `latest.npz` 1.67 GB ×2.

### `best_heads/` — 6.76 GB (curated winners)
- `q7b/head_final.safetensors` 2.94 GB · `q3b/` 1.71 GB · `q1p5b/` 1.11 GB ·
  `q05b/` 0.66 GB + `manifest.json` (4.1 KB).
- These are byte-size-identical to selected `sweep` winners (q7b↔sweep b1,
  q3b↔sweep b2).

### `frozen_cache/` — 4.90 GB
- `frozen_gguf.npz` per bucket: q7b 2.03 GB · q3b 1.24 GB · q1p5b 0.93 GB ·
  q05b 0.54 GB.

## Tree 3 — `dismantle_headbank_corrected/` (8.0 GB)

`heads_corrected/` + `headbank_manifest.json` (4.8 KB). Each bucket =
`head_final.safetensors` + `latest.npz` + `tau_eval.json` + `log.jsonl`.
- `q7b/` 3.85 GiB (head 3.16 GB, npz 0.98 GB) · `q3b/` 1.75 GiB (head 1.56 GB) ·
  `q1p5b/` 1.20 GiB (head 1.11 GB) · `q05b/` 0.62 GiB (head 0.61 GB).

---

## Also found: 40.75 GB of loose training checkpoints in My Drive root

A root scan (`parentId = '0AMLONUeOjcquUk9PVA'`) found **113 loose training files
dumped directly in My Drive root** — 5 separate Eagle-5 head runs from 2026-05-26
(the day before the organized exports), never foldered:

| Kind | Count | Size |
|---|---|---|
| `step_*.npz` (periodic training state) | 93 | 32.94 GB |
| `head_final.safetensors` | 5 | 5.76 GB |
| `latest.npz` | 6 | 2.05 GB |
| `log.jsonl` + `tau.json` | 9 | negligible |
| **Total** | **113** | **40.75 GB** |

All disposable: `step_*.npz` are periodic optimizer/state checkpoints, and the 5
`head_final.safetensors` are byte-size matches for heads later organized into the
dismantle folders (superseded duplicates). Everything **else** loose in root is
personal/academic (course notes, assignments, CVs, affidavits, PDFs, a video) and
is preserved, as are the unrelated project folders (`TradingModels`, `demucs`,
`gravel`, `apartments`, `BHM_highlight_videos`, `Colab Notebooks`).

## Cleanup decision (2026-05-31)

Colab work paused for the foreseeable future. **Everything dismantle/training goes
from Drive (~115 GB); a single keeper tar is pulled local first.** Two notebooks
in this dir drive it:

1. **`consolidate_drive_dismantle.ipynb`** — tars the keep-set
   (`final_push/best_heads` + `final_push/frozen_cache` + the unique `dsv2` head,
   ≈ 13 GB) into one `.tar` in Drive with a sha256, for a one-file download. The
   q3b head was already local; `dsv2` is folded into the tar so it survives.
2. **`clear_drive_dismantle_and_root.ipynb`** — after the tar is downloaded +
   verified, trashes (via Drive-API `trashed=true`, recoverable ~30 days) the
   3 dismantle folders (74.7 GB) **and** the 113 loose root checkpoints (40.75 GB),
   including the 5 superseded root heads. Guarded by a `CONFIRM` flag + dry-run.

Net: ~115 GB reclaimed from Drive, ~13 GB keeper tar preserved locally, all
personal docs + other projects untouched.
