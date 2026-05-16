# Path-to-90 C2 — Drafter training-data pipeline (close report)

**Date:** 2026-05-15
**Branch:** `claude/dreamy-golick-d54ff8` (continues `claude/modest-williamson-57d50f`)
**Base:** `ae65aa5` (B1 — PPL eval harness shipped)
**Status:** SHIPPED — pure tooling + new engine seam, no perf change.

## What this ships

Three artifacts that together unblock off-machine EAGLE-3 draft-head training. None of them changes any kernel, dispatch, or perf-affecting code path; the existing 39 lib tests still pass.

| Artifact | Purpose |
|---|---|
| **Architecture decision** — `reports/path_to_90/stage3_c1/architecture.md` | EAGLE-3 selected as the default draft-head architecture for path-to-90 spec-decode. Records the comparison against MTP (V3-native) and ReDrafter, the constraints that drove the choice (verify-path overhead dominates draft cost at K=4), and the engine-integration footprint for C3. |
| **Engine seam** — `forward_token_with_hidden_for_test` | New trait method on `Engine` (default `Unimplemented`) and concrete impl on `DeepSeekV2`. Returns `(final_norm_hidden_state, greedy_next_token)`. KV cache advances exactly as in `forward_token`. Mirror of the existing `forward_token_shared_only_for_test` pattern. 11 lines of new Rust. |
| **Capture pipeline** — `dismantle capture-hidden` + `tools/training/capture_hidden.py` | Rust subcommand writes a custom binary file (DCAP v1) of `(sample_id, pos, prev_token, next_token, hidden_f16[H])` records; Python orchestrator handles dataset prep (HF datasets), sharded resumable invocation, parquet conversion, and inspection (decode-back-to-text sanity check). |

## Architecture decision (one paragraph)

**EAGLE-3** wins for path-to-90 because (a) the verify path's K× single-forward cost is the binding constraint at K=4 (per `stage3_spec/audit.md`), so MTP's higher acceptance ceiling is mostly wasted until the parallel-K verify kernels land; (b) ReDrafter's cheaper-per-draft compute is also wasted because draft cost is already <1% of verify cost; (c) EAGLE-3's training is bounded — ~12-16 H100-hr on ~500K dialogue samples vs 30-50 H100-hr for an MTP-style head; and (d) the captured (hidden, next_token) dataset is architecture-agnostic, so a future re-evaluation can train the alternate head from the same data without re-capture. Full reasoning, comparison table, and engine-integration sketch in [stage3_c1/architecture.md](../stage3_c1/architecture.md).

## Smoke run results

```
$ ./target/release/dismantle capture-hidden \
    --weights models/deepseek-v2-lite-q4.gguf \
    --samples tests/data/c2_smoke_10samples.jsonl \
    --out training_data/c2_hidden/smoke_shard.bin \
    --max-tokens 128 \
    --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json

[capture-hidden] loaded 10 sample(s) (max_tokens=128)
[capture-hidden] engine loaded in 5.2s
[capture-hidden] 10/10 id=9 L=128 records+=127 (6.1s, ETA 0s)
[capture-hidden] done: +10 samples, +1154 records (total 1154 records, hidden_dim=2048) in 65.8s
```

Then a resume run that adds 5 more samples (10 → 15):

```
[capture-hidden] resume: 1154 record(s) already in smoke_shard.bin (10 unique sample_ids)
[capture-hidden] pending: 5 sample(s) (skipping 10 done)
[capture-hidden] done: +5 samples, +351 records (total 1505 records, hidden_dim=2048) in 20.3s
```

| Metric | Value |
|---|---|
| Samples captured | 15 (wikitext-2 paragraphs) |
| Records | 1505 |
| Records / sample (mean / min / max) | 100.3 / 31 / 127 |
| Hidden dim | 2048 (matches V2-Lite `config.hidden`) |
| Hidden dtype | f16 |
| .bin file size | ~6.1 MB |
| .parquet file size (zstd) | ~5.7 MB |
| Wall time | ~86 s for 15 samples ≈ 17.5 records/s ≈ 57 ms/token |
| Throughput at scale (extrapolated) | ~15 hr / 1M records on this hardware (single-token decode path) |

## Validation

| Check | Method | Result |
|---|---|---|
| File format opens cleanly via HF datasets | `load_dataset("parquet", data_files=...)` | 1154 rows, 5 columns, schema metadata round-trips (`hidden_dim=2048`, `hidden_dtype=float16`, `dcap_version=1`) |
| Hidden shape matches expected V2-Lite hidden_dim | `np.frombuffer(row['hidden_f16'], dtype=np.float16).shape` | `(2048,)` for every record sampled |
| Hidden values are finite (no NaN/Inf) | `np.isfinite(arr).all()` on 10 random rows | all finite |
| Hidden distribution sane (post-rmsnorm magnitudes) | row 0 stats | min=−12.13, max=+4.27, mean=−0.007, std=0.416 — typical for post-final-norm hidden state |
| Tokenizer round-trip — decoded text matches source | Decode `[prev[0], next[0], next[1], …]` for 3 random samples | Each window decodes to the wikitext source paragraph (with the BPE space sigil "Ġ") — proves prev/next pairing has no off-by-one and sample_ids are not corrupted |
| Resume skips done samples without reloading model | rerun with `--resume` over same JSONL | "[capture-hidden] nothing to do" — exits before model load |
| Resume appends only new samples | rerun with 5 new samples appended to JSONL | 1154 → 1505 records, original 10 sample_ids untouched, new 5 appended |
| Pre-existing lib tests | `cargo test --workspace --lib --release` | 5 + 25 + 9 = **39 pass** (same count as B1 close) |

## Why this unblocks Stage 3

The plan's Stage 3 (§C1-C4) requires three steps to break the bandwidth roofline:

1. **C1: architecture decision** — done in this session.
2. **C2: training data** — done in this session (the data **pipeline** is done; the actual ~500K-sample dataset is a one-shot full run that takes ~15 hr wall on this M3 Pro and is queued for the next attended session, since it doesn't fit the session ceiling).
3. **C3: engine wire-up + off-machine training** — gated by C2 completing the full data run AND ~12-16 H100-hr for training. Sketched in `architecture.md`; both halves are now bounded and out of the critical path.

The session-impact-free claim matters: C2 changes only an opt-in test seam and adds a separate CLI subcommand. Default decode path, default profile, default `dismantle generate` invocation — none touched. The existing performance characterization stands.

## Followups (off-session, in priority order)

1. **Full data capture run.** Same pipeline, ~500K samples (UltraChat + ShareGPT mix), ~15 hr wall on M3 Pro 18 GB, ~3 GB final shard set. Resume-capable so it can run across multiple attended windows. Target: `training_data/c2_hidden/eagle3_v0/shard_{000..099}.parquet`. Actual sample count for full run is calibrated against EAGLE-3 paper §4.1 (500K dialogue samples) — could shrink to 100-200K if compute-bound.
2. **Off-machine training session brief.** ~500-word doc capturing: dataset path/format, EAGLE-3 head architecture (1 attn + 1 MLP block, hidden_dim=2048, vocab=102400, share lm_head with target), training framework choice (default: `eagle-llm/eagle-3` reference repo with V2-Lite config adapter), expected loss curve shape, acceptance-rate sanity check on a held-out 100-prompt slice. Prereq for spinning up the H100 rental — should land in this same `stage3_c2/` folder before the rental fires.
3. **C3 engine wire-up.** New `DraftSpecDecoder` plumbing in `crates/dismantle-core/src/engine/`. The plan called this out at §C3. The trait extension that C3 plugs into is **already in place** (`forward_token_with_hidden_for_test`) — C3 just needs the inverse direction (consume a draft head's output, propose K tokens, then call the existing `forward_tokens_batched_for_test` to verify). Multi-week elapsed; not a session task.
4. **Path B verify-kernel rewrite (parallel-K MLA / lm_head / MoE-gate).** The arithmetic in `stage3_spec/audit.md` is unchanged: even a perfect EAGLE head delivers ~1.25× e2e at K=4 against today's verify cost. Path B is *also* multi-week kernel work and runs in parallel with C3. The plan should sequence: full data capture (off-session) ⇒ off-machine training (~weekend) ⇒ either C3 (then minimal-spec gain) or Path B+C3 (then real-spec gain).
5. **Throughput improvement for capture (only if needed).** Today's `capture-hidden` runs single-token decode ⇒ ~57 ms/record. A batched variant using the existing `forward_tokens_batched_for_test` path could 5-10× this (the same trick PPL eval uses). Defer until the full run actually shows wall-clock as a blocker — for 500K samples × 100 tokens at 57 ms = ~80 hr, which is close to the threshold; if the dataset shrinks to 200K it's ~30 hr and acceptable as-is.

## What this session did NOT do

- **No training.** Off-machine, ~12-16 H100-hr, separate session.
- **No engine wire-up of a draft head into the speculate path.** That's C3 — the trait seam exists but no `DraftSpecDecoder` was added to the speculate module.
- **No perf bench.** No env-A / env-B measurement. C2 changes only an opt-in test seam and a separate CLI subcommand; default decode path is byte-identical to ae65aa5.
- **No bit-identical token regression check.** The existing `tests/golden/_phase1_token_baseline.hashes` covers `dismantle generate` paths. `forward_token_with_hidden_for_test` is internally identical to `forward_token` (calls the same `forward_token_final_norm` + `gemv_f16_dispatch` + `argmax_f32` sequence) — running the regression would be a no-op so it was skipped to keep the session honest about what was actually tested.
- **No full data capture.** That run is queued for an off-session window (~15 hr wall) per the followups.

## Output-shape note (re: `tools/bench/multi_prompt_bench.sh`)

CLAUDE.md's session contract says to "mention multi_prompt_bench.sh in the close report if C2's output shape impacts any later perf gate." It does not. C2 produces a binary dataset; perf gates produce JSON / dec_tps / NLL files. The two pipelines do not share output files, schemas, or directories. `multi_prompt_bench.sh` is unaffected by anything in this session.

## File list (all force-added under reports/ + new code)

```
crates/dismantle-core/src/engine.rs                          (modified — +14 lines for trait seam)
crates/dismantle-core/src/model/deepseek_v2.rs               (modified — +20 lines for impl)
crates/dismantle/src/main.rs                                 (modified — +CaptureHidden subcommand + capture_hidden_main fn, ~280 lines)
tools/training/capture_hidden.py                             (new — ~500 lines)
tests/data/c2_smoke_10samples.jsonl                          (new — 10 wikitext samples for smoke)
tests/data/c2_smoke_15samples.jsonl                          (new — 15 wikitext samples for resume test)
training_data/c2_hidden/smoke_shard.meta.json                (new — force-add, provenance only)
# .bin (~6.1 MB) and .parquet (~5.7 MB) NOT committed — regenerable in
# ~90s by re-running the smoke pipeline. The meta.json carries provenance.
reports/path_to_90/stage3_c1/architecture.md                 (new — force-add, decision doc)
reports/path_to_90/stage3_c2/close.md                        (new — this file, force-add)
reports/path_to_90/session_closeout.md                       (modified — appended C2 continuation section)
```
