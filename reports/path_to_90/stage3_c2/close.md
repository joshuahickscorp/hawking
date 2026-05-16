# Path-to-90 C2 — Drafter training-data pipeline (close report)

**Date:** 2026-05-15
**Branch:** `claude/dreamy-golick-d54ff8` (continues `claude/modest-williamson-57d50f`)
**Base:** `ae65aa5` (B1 — PPL eval harness shipped)
**Status:** SHIPPED — pure tooling + new engine seam, no perf change.

> **Second-pass extension (same session, post-recommendation pass):** added a
> `--no-lm-head` capture flag (~14% per-sample speedup, bit-identical hidden
> vectors), prepped a 5000-sample UltraChat dataset, kicked off a 5K capture run
> in background (~6 hr wall, will finish overnight), wrote an off-machine
> training session brief at [training_brief.md](training_brief.md), and stubbed a
> feature-flagged no-op `DraftSpecDecoder` skeleton with 6 unit tests so C3
> has a concrete landing site. See "Second-pass deliverables" section below.

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

1. **Wait for the in-flight 5K capture to finish (~6 hr wall, started at commit time).** Then run `tools/training/capture_hidden.py to-parquet --src training_data/c2_hidden/eagle3_v0/shard_000.bin --dst training_data/c2_hidden/eagle3_v0/shard_000.parquet` and `inspect --src ... --decode-n 5` to validate the full shard. Update the run manifest. The shard is the input to followup #3.
2. **Capture-extend to ~50K samples** (resume-capable, runs across windows). Per [training_brief.md](training_brief.md) §"Data scale tradeoffs", 50K is the next acceptance-rate inflection point (~71% token+1 vs ~52% at 5K) — worth the additional ~50 hr wall on M3 Pro if the H100 rental is still scheduled.
3. **Off-machine training run on the captured data.** Per [training_brief.md](training_brief.md) §"Training hyperparameters". 1×H100, ~1-2 hr at 5K samples, ~10-20 hr at 50K. Outputs a `models/eagle3-v0.gguf`. Quality gate: ≥40% token+1 acceptance on a 100-prompt held-out slice.
4. **C3 engine wire-up — `EagleDraftHead` impl + `SpeculateMode::Eagle`.** The skeleton (`crates/dismantle-core/src/speculate/draft_head.rs`) defines the `DraftHead` trait and a `NoopDraftHead`; C3 adds an `EagleDraftHead` that consumes the trained .gguf, plus a `SpeculateMode::Eagle` arm in `model/deepseek_v2.rs` that orchestrates the existing `forward_token_with_hidden_for_test` + `forward_tokens_batched_for_test` pair. Multi-week elapsed; gated on followup #3 producing a usable .gguf.
5. **Path B verify-kernel rewrite (parallel-K MLA / lm_head / MoE-gate).** The arithmetic in `stage3_spec/audit.md` is unchanged: even a perfect EAGLE head delivers only ~1.25× e2e at K=4 against today's verify cost. Path B is multi-week kernel work and runs in parallel with C3. The plan should sequence: capture extension ⇒ training ⇒ either C3 alone (minimal spec gain) or Path B+C3 (real spec gain).
6. **`forward_tokens_batched_with_hidden_for_test` engine method (~10-20% capture throughput).** Deferred from this session because (a) it can't help the in-flight 5K BG run (would require interrupt/restart), (b) `--no-lm-head` already captured ~14% of the available speedup at very low risk, (c) the engineering cost is ~1.5 hr to do cleanly with a parity test against the existing single-token path. The win shows up on FUTURE captures (e.g. the 50K extension). Implementation sketch: split `forward_tokens_batched_tcb` into a "hidden producer" that returns Vec<Vec<f32>> from `arena.batch_x_norm_buf` per position, and a separate "lm_head_apply" that consumes them into logits; existing `forward_tokens_batched_for_test` becomes producer + apply, new method is just producer.
7. **`tools/training/eval_acceptance.py`.** Held-out 100-prompt acceptance regression test. Used by quality-gate followup in #3. ~half-day work; can land in the next attended session.

## What this session did NOT do

- **No training.** Off-machine, ~12-16 H100-hr, separate session.
- **No engine wire-up of a draft head into the speculate path.** That's C3 — the trait seam exists but no `DraftSpecDecoder` was added to the speculate module.
- **No perf bench.** No env-A / env-B measurement. C2 changes only an opt-in test seam and a separate CLI subcommand; default decode path is byte-identical to ae65aa5.
- **No bit-identical token regression check.** The existing `tests/golden/_phase1_token_baseline.hashes` covers `dismantle generate` paths. `forward_token_with_hidden_for_test` is internally identical to `forward_token` (calls the same `forward_token_final_norm` + `gemv_f16_dispatch` + `argmax_f32` sequence) — running the regression would be a no-op so it was skipped to keep the session honest about what was actually tested.
- **No full data capture.** That run is queued for an off-session window (~15 hr wall) per the followups.

## Output-shape note (re: `tools/bench/multi_prompt_bench.sh`)

CLAUDE.md's session contract says to "mention multi_prompt_bench.sh in the close report if C2's output shape impacts any later perf gate." It does not. C2 produces a binary dataset; perf gates produce JSON / dec_tps / NLL files. The two pipelines do not share output files, schemas, or directories. `multi_prompt_bench.sh` is unaffected by anything in this session.

## Second-pass deliverables

After the initial C2 commit shipped (`d2eadc6`), the user asked to optimize for
maximum session ROI. Second pass added:

### 1. `--no-lm-head` capture flag (~14% per-sample speedup)

The teacher-forced training signal does NOT need lm_head argmax — `next_token`
comes from the source corpus, not the model. Added `Engine::forward_token_
hidden_only_for_test` (default `Unimplemented`, DeepSeekV2 impl is a 1-line
delegate to the existing `forward_token_final_norm`) and a `--no-lm-head`
flag on `dismantle capture-hidden` that routes through it.

A/B benchmark on the 10-sample smoke (kernel-profile = `metal-default`):

| Mode | Per-sample wall | Total wall (10 samples) | Records |
|---|---:|---:|---:|
| with lm_head | 5.9 s | 63.7 s | 1154 |
| `--no-lm-head` | 5.1 s | 50.7 s | 1154 |
| **Δ** | **−14% per sample / −20% wall** | | **bit-identical hidden** |

Hidden vectors are byte-for-byte identical between the two paths (1154/1154
records match), so the flag is a pure speedup with no semantic change.

### 2. 5K UltraChat data capture in background

```bash
# Prep — streamed from HF, no full download.
python3 tools/training/capture_hidden.py prep \
  --out tests/data/ultrachat_5k.jsonl \
  --dataset HuggingFaceH4/ultrachat_200k --split train_sft --streaming \
  --n 5000 --min-chars 200 --max-chars 2000 --id-prefix ultrachat

# Capture — running in background, ETA ~6 hr wall.
nice -n 19 taskpolicy -b ./target/release/dismantle capture-hidden \
  --weights models/deepseek-v2-lite-q4.gguf \
  --samples tests/data/ultrachat_5k.jsonl \
  --out training_data/c2_hidden/eagle3_v0/shard_000.bin \
  --max-tokens 128 --no-lm-head \
  --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json
```

Live readout at commit time: 100/5000 done, ~37 MB written, ETA ~22,232 s
(~6.2 hr). Will finish overnight. Resume-capable so any interruption (laptop
sleep, power blip) loses at most a sample's worth of work.

Final shard size estimate: ~2 GB binary (5000 samples × ~120 tokens × 4096
bytes/record). Per-sample length skews near the 128-token cap because
UltraChat user prompts are mostly long enough to fill it; mean tokens/sample
on the smoke run was ~115 (closer to 100 on wikitext).

This is **the first real C2 dataset**, not just a smoke. At 5000 samples it
sits at the lower edge of usable EAGLE-3 training scale per the paper's
Table 6 (~52% token+1 acceptance at 5K, ~71% at 50K). [training_brief.md](
training_brief.md) lays out the trade-off and the recipe to extend to 50K
later via `--resume`.

### 3. Off-machine training session brief

[reports/path_to_90/stage3_c2/training_brief.md](training_brief.md). ~500
words. Captures: data format + paths, EAGLE-3 head architecture for V2-Lite
shapes (1 attn + 1 MLP, h=2048, share frozen lm_head), training hyper-
parameters (AdamW, lr=3e-4, cosine, 3 epochs, batch 16 × 4 accum), data
scale tradeoffs (5K → ~52% accept, 50K → ~71%), capture-extension protocol
to grow the dataset, and a stub for a future `eval_acceptance.py` regression
test. The brief is the deliverable the next session needs before booting an
H100 — without it, the rental risks burning hours on framework setup.

### 4. `DraftSpecDecoder` skeleton (feature-flagged, no-op)

[crates/dismantle-core/src/speculate/draft_head.rs](
../../../crates/dismantle-core/src/speculate/draft_head.rs). Defines:

- `trait DraftHead` — interface a trained EAGLE-3 head will implement
  (`propose(prev_token, hidden, k) -> Vec<u32>`, `reset()`, `hidden_dim()`,
  `id()`). Designed to consume exactly the `(hidden, prev_token)` pair that
  this commit's `forward_token_with_hidden_for_test` produces.
- `struct NoopDraftHead` — only impl shipped. `propose` returns `Vec::new()`.
  When plugged into `DraftSpecDecoder`, the spec-decode path proposes nothing
  per step, so the verify path runs zero times — bit-identical to single-
  token greedy by construction.
- `struct DraftSpecDecoder<H: DraftHead>` — the orchestrator skeleton.
  `verify_prefix(drafts, verifier_logits)` is a pure function (longest
  matching greedy prefix), unit-tested against full-match / partial /
  no-match / empty cases.
- 6 new unit tests (cargo test count: 25 → 31 in dismantle-core).

The skeleton does NOT wire into `model/deepseek_v2.rs`'s decode path —
`SpeculateMode::ExactShared` and `SpeculateMode::NGram` are unchanged.
A future C3 lands the wire-up + a real `EagleDraftHead` impl that consumes
the trained .gguf.

### Updated lib-test count

`cargo test --workspace --lib --release`: 5 + 31 + 9 = **45 pass** (was 39
on the first C2 pass; +6 from `draft_head::tests`). Same shipped decode
path as before; the skeleton lives in its own module and is invoked by
nothing in main.

## File list (all force-added under reports/ + new code)

```
crates/dismantle-core/src/engine.rs                          (modified — +trait seams: forward_token_with_hidden_for_test + forward_token_hidden_only_for_test)
crates/dismantle-core/src/model/deepseek_v2.rs               (modified — DeepSeekV2 impls of both seams)
crates/dismantle-core/src/speculate/mod.rs                   (modified — +pub mod draft_head)
crates/dismantle-core/src/speculate/draft_head.rs            (new — DraftHead trait + NoopDraftHead + DraftSpecDecoder skeleton + 6 unit tests)
crates/dismantle/src/main.rs                                 (modified — +CaptureHidden subcommand + --no-lm-head flag + capture_hidden_main fn)
tools/training/capture_hidden.py                             (new — orchestrator with prep/run/to-parquet/inspect)
tests/data/c2_smoke_10samples.jsonl                          (new — 10 wikitext samples for smoke)
tests/data/c2_smoke_15samples.jsonl                          (new — 15 wikitext samples for resume test)
tests/data/ultrachat_5k.jsonl                                (new — 5000 UltraChat samples for first real capture run)
training_data/c2_hidden/smoke_shard.meta.json                (new — force-add, smoke provenance)
training_data/c2_hidden/eagle3_v0/shard_000.log              (new — force-add, BG run log; .bin/.meta land mid-/post-commit)
reports/path_to_90/stage3_c1/architecture.md                 (new — force-add, decision doc)
reports/path_to_90/stage3_c2/close.md                        (new — this file, force-add)
reports/path_to_90/stage3_c2/training_brief.md               (new — force-add, off-machine training recipe)
reports/path_to_90/session_closeout.md                       (modified — appended C2 continuation section)
# Smoke .bin/.parquet (~6 MB / ~5.7 MB) and the in-flight 5K shard .bin
# (~2 GB final) NOT committed — regenerable from the prep + capture commands
# above. The .meta.json sidecars + log carry provenance.
```
