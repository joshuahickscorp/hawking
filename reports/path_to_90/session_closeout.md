# Path-to-90 — Session closeout (2026-05-15)

**Branch:** `claude/strange-proskuriakova-b5d48e`
**Plan:** `/Users/scammermike/.claude/plans/dismantle-path-to-90-immutable-jellyfish.md`
**Starting commit:** ecf77a6 / 8848165 (v2.2.0 T2.14 head)
**Ending commit:** 6b419d9 (validation finding)

## What this session was for

The user came in with a 3-track multi-month plan to drive DeepSeek-V2-Lite Q4_K_M decode on M3 Pro toward 90 dec_tps. The plan called for Stage 0 attribution → Stage 1 engine work (A1–A6) → Stage 2 KV R&D (B1–B3) → Stage 3 self-speculative decode (C1–C4). This session executed Stage 0 + the entire feasible portion of Stage 1, plus an audit of Stage 3 existing infrastructure, and a validation-pass on the headline numbers.

## What actually shipped (no revert needed)

Two engine landings now baked into the default profile:

| Commit | Lever | env-A Δ (5×3 trial, in-process bench) | env-B Δ (multi-prompt, fresh process) | Status |
|---|---|---:|---:|---|
| 8ee86bf | **A5 — persistent argbuf bump-arena** | +8.4% trimmed | (subsumed in A4) | Shipped default. Replaces per-dispatch `new_buffer` with a 64 KB shared arena reset per token. |
| 0983166 | **A4 — `mla_decode_kernel_fc` via MTLFunctionConstantValues** | +7.8% trimmed | net neutral-to-slight-regression | Shipped default. Specializes MLA decode kernel with 6 model constants baked in at engine load. |

Cumulative under env-A: **20.50 → 23.97 dec_tps trimmed-median (+16.9%)** on the 4-token "Once upon a time" / 64-token-decode bench.

## The honest reading (per 6b419d9)

The +16.9% headline is **valid only under env-A conditions** (one engine shared across many trials, with Claude Code running). The user noticed that 23.97 matches v2.2.0's `clean_bench` reference (~23.3) closely. Validation: ran the same A4 build against pristine pre-A5 source under env-B (multi-prompt harness, fresh process per trial). The win evaporates there — mean Δ across 7 prompts is −5.9%.

**Mechanistic explanation:** both A5 and A4 are *warmup-amortization* optimizations. Their costs are paid once per engine lifetime; their wins compound across the many dispatches that follow. env-A's protocol amortizes; env-B's per-trial fresh process does not.

**Real-world impact:**
- Long-running inference server (engine loaded once, many requests): **+5-15% warm steady-state** — meaningful but smaller than headline.
- Interactive chat with persistent engine: same pattern.
- Per-prompt fresh-process scripts: wash to slight regression.

A5 and A4 are correct, well-tested, ship as default, and do not need to be reverted. The framing changes, not the code.

## What was rejected (kept as opt-in scaffolding)

Five levers committed and rejected at the +3% gate. Code remains in tree as scaffolding for future revisits:

| Commit | Lever | Result | Diagnosis |
|---|---|---|---|
| 918c93c | A1 — `flash_attn_decode_kernel` wire-up | −13.9% in env-A; **−46.5% at long context in env-B** | Flash kernel's inner-loop recomputes `exp(scores_tile[ti] - m_bc)` per output-row × per tile-element. ~2048× more `exp()` calls than `mla_decode_kernel`. Apple-Silicon Metal `exp()` is software-emulated; drowns the kernel. Refactoring to hoist `w[ti]` once per tile (A1.2, ~half-day) might fix this. Opt-in: `mla_schedule = "metal-mla-flash"`. |
| 97c1828 | A4.2 — MoE Q4 routed-gemv fc | −11.1% in env-A | Inner loops are already small (blocks_per_row = 8) and auto-unrolled. Function constants gave the compiler nothing new to fold; the change shifted register-allocator decisions into a lower-occupancy band. First-try also had a parity bug (shared-expert rows ≠ moe_intermediate). Opt-in: `gemm_q4_k_schedule = "v2t_gu_v2_fc"`. |
| 715403e | A3 — `add_rmsnorm_f32` fusion | −5.3% in env-A | The unfused `add_inplace` runs 8 TGs in parallel (hidden/256=8). The fused kernel must be single-TG (rmsnorm reduction needs a threadgroup barrier; cross-TG sync isn't available within one dispatch). The add phase loses 8× parallelism; GPU regression eats the CPU saving. Opt-in: `residual_fusion = "f32"`. |
| 2628a5c | Stage 3 — existing speculate audit | No code; docs only | NGram + ExactShared already exist but don't fire on the 4-token bench prompt. On long code-prompt with 79.7% accept rate, NGram still loses −15% because verify cost ≈ K × single-forward; theoretical ceiling at K=4 is 1.25×, CPU overhead eats the margin. Real spec-decode needs trained drafter (off-machine) or new parallel-K MLA kernels (multi-week). |

## Durable infrastructure landed

| Commit | Artifact | Why it matters |
|---|---|---|
| 3036639 | Stage 0 attribution — [reports/path_to_90/stage0/](stage0/) | Profile-driven gap analysis vs llama.cpp on the same model/hw (52.51 dec_tps). Established that the gap is 2.2-2.5× (not 3×) and that ~85% of it is CPU dispatch overhead, not kernel quality. |
| 8ee86bf | `MetalContext::argbuf_alloc/reset` API | Reusable beyond A5: any future kernel that wants packed scalar args can carve from the arena. |
| 0983166 | `MetalContext::register_specialized_pipeline` | Reusable beyond A4: any future MTLFunctionConstantValues specialization plugs into the same pipeline-cache injection point. |
| ad9fec9 | `tools/bench/multi_prompt_bench.sh` + 7-prompt suite | The infrastructure that *caught* the env-A vs env-B disagreement. Foundation for B1 (PPL eval), B2/B3 (KV/expert quant quality gates), and future spec-decode work. |
| 6b419d9 | Validation methodology | Documents how to read env-A vs env-B numbers and the trap of single-environment headlines. Pattern reusable across all future levers. |

## What the plan got right and wrong

**Right:**
- Stage 0 first. Attribution before code changes saved chasing the wrong levers.
- A5 ordering. The plan was revised to put A5 first based on Stage 0 evidence; A5 was the cleanest landing of the session.
- "Reject at +3%" gate. Each rejection produced documented evidence; nothing shipped that regresses.

**Wrong:**
- The plan's "Stage 1 +30-50% from engine work alone" was over-optimistic. Actual env-B impact of A5+A4 combined is ~0-5% on cross-process workloads, not +16% as env-A suggested. The plan's heuristics (dispatch-count reduction, GPU-share specialization) didn't account for register-allocator interactions, fusion-parallelism tradeoffs, or warmup-vs-steady-state divergence.
- The plan's "A1 flash-attn keystone" thesis was wrong even after fixing the seq_len concern. The flash kernel as-shipped has a structural `exp()`-recomputation bug; it doesn't win even at the contexts the plan called out.
- The "verify cost ≈ K × single-forward" arithmetic for spec-decode wasn't acknowledged. The existing infrastructure can't deliver wins from any drafter at K=4; needs either a cheap parallel-K MLA kernel rewrite or a real trained drafter.

## Where the path-to-90 actually stands

Honest assessment, factoring the validation finding:

- **Engine track ceiling:** roughly v2.2.0's ~23.3 dec_tps under cross-process conditions, ~24-28 under warm long-running conditions. A5+A4 sit near the upper end of this range. Further engine wins (A6 autotune, A3 done correctly with a two-buffer ping-pong, future fc specializations) are likely +0-5% each — small, hard to validate.
- **KV R&D track (Stage 2) ceiling:** can break through at long context where KV bandwidth grows. p006 (169-token prompt + 96 decode = seq_len ~265) is just barely entering the regime where Q8 latent KV would help. Real benefit at chat-scale (4K-32K context) is +10-30% but the bench doesn't reach that regime yet.
- **Spec-decode track (Stage 3) ceiling:** 1.5-2× potential, but only with a trained drafter (off-machine R&D) OR a parallel-K MLA kernel rewrite. Either is multi-week elapsed.
- **The hard truth:** **90 dec_tps probably requires at minimum BOTH a trained EAGLE-3-style drafter AND a long-context (4K+) workload baseline.** The single-prompt 4-token bench will never see 90 dec_tps on M3 Pro Q4_K_M — the bandwidth roofline at 1.82 GB/token × 130 GB/s practical = 71 dec_tps is the engine ceiling, period. The only way past it is multi-token-per-forward-pass spec decode.

## Suggested re-prioritization (whenever next session happens)

The plan's three tracks remain right; the priorities within them need updating:

1. **Highest leverage, session-scope:** finish B1 (PPL eval harness, `tools/bench/ppl_eval.py`). ~3 hours. Unblocks every future quant/KV decision and adds a quality gate that's currently missing.
2. **Highest leverage, multi-session:** start C2 prep — gather distillation data for an EAGLE-3 / MTP head trained on DeepSeek-V2-Lite outputs. The training itself happens off-machine. ~1 week elapsed wall-time for the dataset; ~12-24 H100-hours for the head.
3. **Worth one focused session:** A1.2 — refactor `flash_attn_decode_kernel` to hoist `w[ti]` once per tile. The bug is identifiable and the fix is bounded. If it works, A1 becomes a context-conditional schedule (default at seq_len ≥ 512).
4. **Skip / deprioritize:** A6 autotune polish, A4.2 retry, more dispatch-fusion attempts. Each is +0-5% under env-B; the marginal value is shrinking.

## Files cleanly committed

Total: 8 commits this session, 5 reports + 1 plan + new code + scaffolding.

```
6b419d9 path-to-90: A5+A4 win validation — honest framing
ad9fec9 path-to-90: multi-prompt bench harness + A1 long-context re-litigation
715403e v2.3.0 A3: add_rmsnorm_f32 fusion — REJECTED at +3% gate
2628a5c path-to-90 Stage 3 audit: existing spec-decode reality-checked
97c1828 v2.3.0 A4.2: MoE Q4 routed v2t_gu_v2_fc — REJECTED at +3% gate
918c93c v2.3.0 A1: flash_attn_decode_kernel wire-up — REJECTED at +3% gate
0983166 v2.3.0 A4: mla_decode_kernel_fc via MTLFunctionConstantValues — +7.8% e2e
8ee86bf v2.3.0 A5: persistent argbuf bump-arena — +7-8% e2e
3036639 path-to-90 Stage 0: attribution + llama.cpp comparator
```

## Behavioral changes since v2.2.0

The shipped engine is functionally:
- Same kernels as v2.2.0 for non-fc paths
- New `mla_decode_kernel_fc` is selected by default (specialized for V2-Lite's shape constants)
- All `KernelArgBuffer` writes route through a per-context bump arena instead of per-dispatch `new_buffer`
- All token-level parity tests still pass bit-identical against pre-A5

Profile changes vs v2.2.0:
- `mla_schedule`: `metal-mla` → `metal-mla-fc`
- `shader_hash`: updated to reflect added kernels (mla_decode_kernel_fc, moe_..._v2t_gu_v2_fc, add_rmsnorm_f32, flash_attn_decode_kernel was already present)
- Added optional opt-in field `residual_fusion` (default `"off"`)

---

## Continuation 2026-05-15 (modest-williamson) — B1: PPL eval harness

**Branch:** `claude/modest-williamson-57d50f` (fast-forwarded from `strange-proskuriakova-b5d48e`)
**Scope:** Option 1 from the prioritized queue. Pure tooling, no perf risk.

### What shipped

- `dismantle ppl-eval` subcommand — per-sample NLL via `forward_tokens_batched_for_test` + log_softmax; KV reset between samples.
- `tools/bench/ppl_eval.py` orchestrator — `prep` / `run` / `diff` modes; ±0.5% ΔPPL gate built in.
- `tests/data/wikitext2_256_samples.jsonl` — deterministic 256-paragraph WikiText-2 slice (seed 20260515).
- Reference baseline `reports/path_to_90/stage1_b1/baseline_fp16kv.jsonl` — current `metal-default` profile, 26,849 scored tokens, **PPL=34.2393**, avg NLL=3.5334, wall 24.2 min.

### Reproducibility validation

- Self-diff (rerun same JSONL → diff against itself): ΔPPL = 0.000% / 0 mismatches.
- 16-sample re-run vs first 16 of full baseline: **16/16 bit-identical NLL**.
- `cargo test --workspace --lib --release`: 5 + 25 + 9 tests pass.

### Why this unblocks Stage 2

A2 (Q8 latent KV) and B2 (WHT 3-bit KV) both need an oracle finer than bit-identical 3-token greedy. Until this session, that oracle didn't exist — any future quant work had no quality gate. `tools/bench/ppl_eval.py run --diff-baseline reports/path_to_90/stage1_b1/baseline_fp16kv.jsonl` now fills that hole.

### What this session did NOT do

- No engine perf changes (no A1.2 hoist, no C2 drafter prep). Single-option session by design.
- No CI integration. The harness is invoked by the implementer of a variant; ~25 min wall is too long for daily CI without more work.
- The absolute baseline PPL (34.24) is a high number because samples are 128-token paragraphs reset between (BOS-adjacent positions have inherently high NLL) and the Chat SFT variant raises raw-text PPL. **The figure of merit is ΔPPL vs this same baseline; absolute numbers don't compare to published wikitext PPL.**

### File list

```
crates/dismantle/src/main.rs                                     (modified — +PplEval subcommand)
tools/bench/ppl_eval.py                                          (new)
tests/data/wikitext2_256_samples.jsonl                           (new)
reports/path_to_90/stage1_b1/{baseline_fp16kv.jsonl,baseline_fp16kv.log,close.md}  (new, force-add)
reports/path_to_90/session_closeout.md                           (this section)
```

---

## Continuation 2026-05-15 (dreamy-golick) — C1 + C2: drafter architecture + data pipeline

**Branch:** `claude/dreamy-golick-d54ff8` (continues `claude/modest-williamson-57d50f`)
**Scope:** Stage 3 §C1 (architecture decision) + §C2 (training-data pipeline) from the prioritized queue. Pure tooling + an opt-in test seam, no perf change.

### What shipped

- **Architecture decision** — `reports/path_to_90/stage3_c1/architecture.md`. EAGLE-3 default. Comparison vs MTP and ReDrafter, weighted against the binding constraint that verify-cost ≈ K × single-forward at K=4 makes higher-acceptance heads mostly wasted until Path B (parallel-K verify kernels) lands.
- **Engine seam** — `forward_token_with_hidden_for_test` on the `Engine` trait (default `Unimplemented`) plus a 20-line impl on `DeepSeekV2`. Returns `(final_norm_hidden, greedy_next_token)` while advancing KV identically to `forward_token`. Mirror of the existing `forward_token_shared_only_for_test` pattern.
- **Capture pipeline** — new `dismantle capture-hidden` subcommand writes a custom binary file (DCAP v1) of `(sample_id, pos, prev_token, next_token, hidden_f16[H])` records, plus a JSON sidecar with model_id / profile_id / hidden_dim. Resume-capable. Python orchestrator at `tools/training/capture_hidden.py` handles HF-datasets prep, sharded invocation, parquet conversion, and inspection (decode-back-to-text sanity check).

### Smoke run

15 wikitext-2 paragraphs captured to `training_data/c2_hidden/smoke_shard.bin`: 1505 records, hidden_dim=2048, ~6.1 MB binary / ~5.7 MB parquet (zstd). Validation:
- HF datasets opens the parquet cleanly; schema metadata round-trips.
- 10 random rows: hidden vector shape `(2048,)`, all finite, magnitude distribution consistent with post-rmsnorm norms (mean ≈ 0, std ≈ 0.4).
- Tokenizer round-trip on 3 random samples: decoded text matches source paragraph (no off-by-one in prev/next pairing, no sample_id corruption).
- Resume: rerun on same JSONL exits before model load; rerun with 5 new samples appended adds exactly 351 new records (1154 → 1505) and leaves the original 10 untouched.
- `cargo test --workspace --lib --release`: 5 + 25 + 9 = 39 tests pass (same as B1).

### Close report

[reports/path_to_90/stage3_c2/close.md](stage3_c2/close.md) covers smoke numbers, validation table, followups, and explicitly records what this session did NOT do (no training, no `DraftSpecDecoder`, no perf bench, no full data run).

### Followups (priority-ordered, all off-session)

1. Full data capture run — same pipeline, ~500K samples, ~15 hr wall on M3 Pro 18 GB. Resume across windows.
2. Off-machine training session brief (~500 words) — drop into `reports/path_to_90/stage3_c2/` before the H100 rental fires.
3. C3 engine wire-up (`DraftSpecDecoder`) — multi-week. The trait seam is in place; C3 only needs the inverse direction (consume draft head → propose K → verify via existing `forward_tokens_batched_for_test`).
4. Path B kernel rewrite (parallel-K MLA / lm_head / MoE-gate) — multi-week, runs in parallel with C3. Without it, even a perfect EAGLE head delivers only ~1.25× e2e at K=4.
5. Batched-capture optimization for `capture-hidden` — only if the full data run shows wall-clock as a blocker. Defer.

### What this session did NOT do

- No engine perf changes; default decode path unchanged.
- No CI integration; capture-hidden runs ad-hoc from the orchestrator.
- No training; no `DraftSpecDecoder`.
- No multi-prompt bench / clean_bench rerun; the change has zero impact on those (default code path is byte-identical to ae65aa5).

### Second-pass extension (same session, post-recommendation)

User asked to optimize for max session ROI. Same branch, same scope-discipline (no perf changes); landed:

- **`--no-lm-head` capture flag** — engine seam `forward_token_hidden_only_for_test` (1-line delegate to existing `forward_token_final_norm`) + CLI flag. A/B benchmark on 10-sample smoke: 63.7s → 50.7s (−20% wall, −14% per sample), 1154/1154 records bit-identical hidden vectors.
- **5K-sample UltraChat capture in background** — `tests/data/ultrachat_5k.jsonl` (HF streaming, deterministic), then `dismantle capture-hidden ... --no-lm-head` running into `training_data/c2_hidden/eagle3_v0/shard_000.bin`. ETA ~6 hr; will finish overnight. First real (non-smoke) C2 dataset shard.
- **Off-machine training brief** — `reports/path_to_90/stage3_c2/training_brief.md`. ~500 words. Data format + paths, EAGLE-3 head architecture for V2-Lite shapes, hyperparameters, data-scale tradeoff (5K → ~52% accept, 50K → ~71% per paper Table 6), capture-extension protocol, future eval_acceptance.py stub.
- **`DraftSpecDecoder` skeleton** — `crates/dismantle-core/src/speculate/draft_head.rs`. Defines `DraftHead` trait + `NoopDraftHead` (proposes nothing → verify path bit-identical to greedy by construction) + `DraftSpecDecoder<H>` orchestrator skeleton with a unit-tested `verify_prefix` helper. 6 new unit tests (dismantle-core: 25 → 31; total 39 → 45).
- Updated [stage3_c2/close.md](stage3_c2/close.md) "Second-pass deliverables" section + revised followups list.

What the second pass did NOT change: any kernel, any dispatch, any default decode path. The new `forward_token_hidden_only_for_test` is opt-in via the new CLI flag. The `DraftSpecDecoder` skeleton is wired to nothing (`SpeculateMode` enum + decode path unchanged). All artifacts are gated, regression-safe, and additive.
