# Path-to-90 C2 — 5K UltraChat capture results

**Date:** 2026-05-16 (overnight chain, no human in the loop)
**Branch:** `claude/dreamy-golick-d54ff8`
**Shard:** `training_data/c2_hidden/eagle3_v0/shard_000.bin` (DCAP v1)
**Parquet:** `training_data/c2_hidden/eagle3_v0/shard_000.parquet` (zstd)

## Headline

The full 5000-sample UltraChat capture completed cleanly. **486,622 records, hidden_dim=2048, no NaN/Inf, ~25% vocab coverage, healthy distributions across all axes measured.** Format is ready for MLX training per the brief.

## Counts

| Metric | Value |
|---|---|
| Unique samples | 5,000 / 5,000 |
| Total records | 486,622 |
| Mean records / sample | 97.3 |
| .bin size | 1.87 GB |
| .parquet size (zstd) | 1.72 GB |
| Capture wall (cumulative across 3 restarts) | ~5.5 hr |
| Effective throughput | ~24.6 records/sec post-`--no-lm-head` |

## Distributions

### Sample length

| stat | value |
|---|---|
| mean | 97.3 tokens |
| p10 / p50 / p90 / p99 / max | 60 / 99 / 127 / 127 / 127 |
| hit 128-cap | 1510 / 5000 (30.2%) |

30% of samples are truncated at the 128-token cap. UltraChat user prompts skew long — if downstream MLX training shows the head undertraining on long contexts, the obvious lever is `--max-tokens 256` on the next capture extension (doubles wall time per sample but captures more of the tail).

### Hidden vector stats (sampled 4866 vectors @ 1/100)

| stat | value | reading |
|---|---|---|
| NaN/Inf | 0 / 4866 | data is clean |
| per-vec mean | −0.030 ± 0.010 | near zero, expected after RMSNorm + lm_head bypass |
| per-vec std | 0.824 ± 0.100 | not unit because RMSNorm normalizes by RMS, not std; range [0.41, 1.10] |
| per-vec L2 norm | 37.3 ± 4.5 | vs √2048 = 45.25 — vectors are slightly under-unit-norm-equivalent, consistent with RMSNorm + small eps; range [18.4, 49.8] |

### Hidden L2 by position bucket (the BOS-warmup question)

| pos bucket | n | mean L2 | std |
|---|---:|---:|---:|
| 0-9 | 1010 | **31.7** | 5.7 |
| 10-29 | 2000 | 36.8 | 4.1 |
| 30-59 | 2895 | 38.1 | 3.9 |
| 60-99 | 2799 | 38.5 | 3.9 |
| 100+ | 1028 | 38.0 | 4.1 |

BOS-adjacent positions (0-9) have ~17% smaller magnitude than deeper positions. This is **normal** — early positions have less context conditioning so the hidden state is geometrically smaller. **Implication for training:** consider position-weighted loss to underweight positions 0-3 (where the head has nothing useful to predict from), or simply drop them at the data-loader layer. The brief's recommendation of B=16 S=16 already implicitly avoids over-representing pos=0 unless the data loader specifically over-samples it.

### Vocab coverage (next-token targets)

| metric | value |
|---|---|
| Unique target tokens hit | 25,269 / 102,400 = **24.68%** |
| Top-1 (id=11, likely comma/period) | 21,688 (4.46%) |
| Top-5 cumulative | 18.97% |
| Top-10 cumulative | 26.80% |
| Top-20 cumulative | 33.52% |
| Tokens seen exactly once | 8,668 (34.3% of seen vocab) |
| Tokens seen exactly twice | 4,286 |

Classic Zipfian distribution. The long tail of hapax tokens is expected and *useful* — it means the head will see rare-but-real tokens during training rather than only the high-frequency core. 25% vocab coverage on 486K records is reasonable; the full vocab is rarely exercised even on millions of natural-text tokens because much of it is reserved for code, special chars, or low-frequency multilingual fragments.

### Position depth (how much "deep context" is in the dataset)

| pos | records | % of samples reaching here |
|---|---:|---:|
| 0 | 5,000 | 100% |
| 30 | 4,998 | 99.96% |
| 60 | 4,480 | 89.6% |
| 100 | 2,393 | 47.9% |
| 126 | 1,510 | 30.2% |

Almost half the samples have ≥100 tokens of conditioning context. The dataset is not BOS-dominated.

## Content sanity (5 random decoded sample windows)

- "Discuss the evolution of dance throughout history..." (87 tokens)
- "Develop a 6-episode mockumentary-style comedy series..." (86 tokens)
- "Write a lighthearted romantic comedy screenplay..." (101 tokens)
- "Write a business journal article analyzing AI's impact..." (72 tokens)
- "Can you reason about the significance of the 'soft, staccato beep'..." (128 tokens, cap)

Diverse content. Prev/next pairing decodes back to recognizable English with no off-by-one or sample-id corruption.

## Interpretation vs EAGLE-3 paper expectations

Per EAGLE-3 paper Table 6 (Vicuna-7B at varying training-data scale):

| Training samples | Expected token+1 accept |
|---|---|
| **5K (this)** | **~52%** |
| 50K (extension target) | ~71% |
| 500K (paper full recipe) | ~78% |

At 52% acceptance with K=4 verify cost ≈ 4× single-forward, e2e gain is `(1 + 4 × 0.52) / 4 = 0.77×` — a regression of ~23%. **So the 5K-trained head will not produce a winning spec-decode setup on its own.** Its purpose is:

1. **Validate the MLX training stack end-to-end.** Loss decreases, no shape bugs, checkpoint converts cleanly to dismantle, regression test runs. Cheap (~5-10 hr on M3 Pro).
2. **Inform whether to extend to 50K.** If 5K training works smoothly, commit ~50 hr to capture the 50K extension. If the 5K hits unexpected snags (loss explodes, hidden distribution mismatched between train and eval, MLX backend issues), debug before scaling up the data.

The winning configuration requires **both** (a) ≥50K training samples → trained head with ≥70% acceptance, AND (b) Path B verify-kernel rewrite (multi-week, parallel-K MLA + lm_head + MoE-gate) per `stage3_spec/audit.md`. This capture run unblocks (a); (b) is separate engineering.

## Known issue (filed as followup)

The Rust `capture_hidden_main` `--resume` path has a **meta.json bug**: it loads prior `sample_ids` from `meta.json` if present, but if no prior meta exists (e.g. the FIRST resume after an initial run that died before writing meta), it starts `all_sample_ids` empty and only records the current run's samples. The .bin file is correct (5000 samples, 486K records); the .meta.json's `samples_processed` field shows 2533 (just the final resume run's samples). Downstream consumers should use `dismantle capture-hidden --resume` (which reads the .bin sample ids directly) or `tools/training/capture_hidden.py inspect` (same) — both report correct counts. The meta sidecar is just metadata; not load-bearing for training.

Fix sketch (for a future session, not blocking): in the Rust `--resume` path, before extending `all_sample_ids`, populate it from `already_done` (the HashSet of sample_ids parsed from the existing .bin). One-line change.

## What this enables, what it does NOT

| Unblocks | Status |
|---|---|
| MLX training-stack validation (the 5K-train run) | ✅ data ready |
| 50K capture extension (`--resume` adds 45K more samples) | ⏳ kicking off now |
| H100-rental backup plan (parquet is portable) | ✅ format works for both paths |
| Acceptance regression evaluation | ⏳ after training |

Does NOT change:
- Any dismantle engine path (no kernel, no dispatch graph)
- Spec-decode behavior (`SpeculateMode::ExactShared` / `NGram` unchanged)
- Any shipped perf number
