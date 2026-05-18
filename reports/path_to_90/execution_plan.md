# Path-to-90 execution plan — 28-step batched sequence

**One linear sequence, no phases.** Each step is fully-specified with
files touched, success criterion, and the deep-research finding that
motivates it. Steps that block downstream work are marked **🔒 BLOCKING**.

Plan was rewritten 2026-05-18 after the deep-research pass refined the
stage-5 ceiling from 140–165 (V4 spec doc) to **95–125 sustained, peak
135**. See `reports/path_to_90/eagle4_deep_research.md` for the
synthesis and citations. The convergence doc
(`reports/path_to_90/eagle4_convergence.md`) is the contract surface
between dismantle and eagle4 and remains authoritative for that.

## Starting state (after commit `376ba2e`)

- `eagle4/` source committed in-tree
- `Eagle4Head::from_npz()` works — loads `eagle4/checkpoints/eagle4_v3/best.npz`
- NPZ parser at `crates/dismantle-core/src/util/npz.rs` (587 lines, no deps)
- 43 dismantle-core lib tests green
- `Eagle4Head::propose()` still `Err(Unimplemented)`
- Engine trait method `forward_token_eagle4_for_test` not yet added
- No CLI wire-up
- Capture process paused at ~85/100000 samples (resumable, but
  redundant — eagle4 has its own 75.84% baseline)

## The sequence

### Foundation (must land before any optimization)

#### 1. 🔒 **Profile baseline bandwidth efficiency**
- **What**: `Instruments → Metal System Trace` on `dismantle generate` running 64-token decode against V2-Lite Q4_K_M. Measure: dec_tps, GPU active %, memory bandwidth used (target: GB/s observed vs M3 Pro's 150 GB/s).
- **Files**: none (profiling only). Record results in `reports/path_to_90/stage0_profile.md`.
- **Success**: profile report with three numbers — observed dec_tps, observed bandwidth, efficiency %.
- **Why**: the entire stage-5 ceiling depends on whether dismantle is currently at 40% efficiency (3× headroom from MLX-pattern adoption) or 70% (efficient already, focus on spec decode).
- **Effort**: 30 min.

#### 2. **Decision gate — MLX patterns adoption**
- **What**: based on step 1's efficiency %, decide whether to do a "Stage 0.5" MLX-pattern adoption pass before any spec-decode work.
- **Decision rule**:
  - Efficiency ≥ 60%: skip — dismantle is already MLX-class. Proceed to step 3.
  - Efficiency 40–60%: defer — note as Class A item, revisit after stage 2 if Stage 2 results disappoint.
  - Efficiency < 40%: **mandatory** — full MLX-pattern audit of dismantle's hottest kernel paths (gemv_q4k, MoE expert pair matmul, MLA decode) against MLX kernel sources at `mlx-lm/.../models/deepseek_v2.py`. Estimated 1–2 weeks.
- **Files**: `reports/path_to_90/stage0_5_mlx_decision.md` documenting the decision.
- **Success**: a written decision with the efficiency number that drove it.
- **Effort**: 30 min for decision; 1–2 weeks if "yes" path is taken.

#### 3. **Engine::forward_token_eagle4_for_test in DeepSeekV2**
- **What**: implement the 5-input capture trait method per `eagle4_convergence.md § Required dismantle changes #3`. Modifies the layer loop in `forward_token_final_norm_maybe_read` to capture `h_low` (layer 2 output), `h_mid` (layer 13 output), `h_high` (layer 25 output), and call `ffn_shared_only(li=26, x_pre_mlp)` for `h_shared`.
- **Files**: `crates/dismantle-core/src/engine.rs` (trait method declaration), `crates/dismantle-core/src/model/deepseek_v2.rs` (impl, layer loop modification at line ~2770).
- **Success**: returns `Eagle4Inputs { prev_token, h_low: Vec<f32>, h_mid: Vec<f32>, h_high: Vec<f32>, h_shared: Vec<f32> }`. Unit test with synthetic 32-token sequence; capture-then-check h_high norm ≈ V2-Lite native equivalent at atol=1e-3 fp16.
- **Effort**: ~4 hr.

#### 4. **Add `--dump-logits` flag to `eagle4/eagle4.py eval`**
- **What**: extend eagle4's eval subcommand to emit per-record (token_logits, mask_logits, calib_logit, draft_hidden) as an NPZ at a user-given path. Needed for parity test diff.
- **Files**: `eagle4/eagle4.py` (~30 lines added to eval main).
- **Success**: `python eagle4/eagle4.py eval --ckpt eagle4/checkpoints/eagle4_v3/best.npz --frozen eagle4/v2lite_frozen.npz --parquet eagle4/data/v2lite_3layer_heldout/shard_00000.parquet --max-records 10 --dump-logits /tmp/ref.npz` writes the npz; `python -c "import numpy as np; z = np.load('/tmp/ref.npz'); print(list(z.keys()), z['token_logits'].shape)"` shows `[(token_logits, (10, 102400)), (mask_logits, (10, 26, 64)), (calib_logit, (10,)), (draft_hidden, (10, 2048))]`.
- **Effort**: ~1 hr.

#### 5. 🔒 **Eagle4Head::propose CPU forward**
- **What**: the 5-input fusion → 1 transformer block → residual gate → frozen LM head + mask + calib pipeline, CPU-only via dismantle's existing `gemv_f32` / `silu_mul` / `rmsnorm_f32` helpers. Initially fp32 throughout for parity validation; Metal acceleration is step 7.
- **Files**: `crates/dismantle-core/src/speculate/eagle4_head.rs` (replace `Unimplemented` with real body).
- **Success**: parity test (`crates/dismantle-core/tests/eagle4_parity.rs`) under `EAGLE4_PARITY_TEST=1` passes — token_argmax matches Python reference exactly per record, mask_logits at atol=1e-3 fp16, calib_logit at atol=1e-3 fp16, draft_hidden at atol=1e-3 fp16. 10-record fixture from held-out shard.
- **Why**: foundational — every downstream step assumes the head forward is correct. Block-ship before anything else.
- **Effort**: ~1 day.

#### 6. **Parity test full wire-up**
- **What**: replace the staged-for-followup section of `eagle4_parity.rs` (the Python-reference diff portion) with the actual subprocess invocation + diff implementation.
- **Files**: `crates/dismantle-core/tests/eagle4_parity.rs`.
- **Success**: `EAGLE4_PARITY_TEST=1 cargo test -p dismantle-core --test eagle4_parity -- --ignored` passes end-to-end with diff output printed via `--nocapture`.
- **Effort**: ~3 hr.

#### 7. **Metal-accelerated Eagle4Head forward**
- **What**: rewrite the head forward using dismantle's Metal kernels (gemv_q4_k variants for in_proj / block.attn.* / block.mlp.* if any quantized variants exist for these matrix shapes; else fp32 Metal gemv_f32_metal). Keep the CPU path as a fallback / parity reference.
- **Files**: `eagle4_head.rs`, possibly new kernel dispatchers if existing ones don't cover the shapes.
- **Success**: 60M-param head forward in <5 ms on M3 Pro at batch=1. Parity test still passes.
- **Effort**: ~1 day.

### Stage 1 — eagle4 head wired, no Path B

#### 8. **CLI wire-up: `--speculate eagle4`**
- **What**: add `SpeculateMode::Eagle4 { head_path: PathBuf, calib_threshold: f32 }` to `crates/dismantle-core/src/engine.rs:56`. Decode path in `model/deepseek_v2.rs` routes through: capture 5 inputs → `head.propose(inputs, K)` → if calib < threshold, autoregressive single step; else `forward_tokens_batched_for_test` on K candidates, accept longest matching greedy prefix, KV rollback on rejection.
- **Files**: `crates/dismantle-core/src/engine.rs`, `crates/dismantle-core/src/model/deepseek_v2.rs`, `crates/dismantle/src/main.rs` (CLI flag parsing).
- **Success**: `dismantle generate --speculate eagle4 --draft-head eagle4/checkpoints/eagle4_v3/best.npz` produces output.
- **Effort**: ~half day.

#### 9. 🔒 **Bit-identical greedy regression**
- **What**: against same prompt, `SpeculateMode::Eagle4` (any K) must produce output bit-identical to `SpeculateMode::Off` greedy at 64 tokens. The longest-matching-greedy-prefix acceptance rule guarantees this by construction; the test catches any wire-up bugs.
- **Files**: `crates/dismantle-core/tests/eagle4_decode_parity.rs` (new).
- **Success**: 4 test prompts × K∈{1,2,4,8}, all bit-identical at 64 tokens.
- **Effort**: ~2 hr.

#### 10. **STAGE 1 MEASUREMENT**
- **What**: bench `dismantle generate --speculate eagle4` against baseline. Record dec_tps under chain spec decode (no Path B yet) on M3 Pro 18 GB. Expected per deep research: **12–22 tok/s** — slight regression on MoE because each spec step costs K sequential verify forwards.
- **Files**: `reports/path_to_90/stage1_eagle4_chain.md` with bench numbers.
- **Success**: documented number, even if regression. **The regression is expected and confirms the MoE-spec-decode minefield** — without K-batched verify, spec decode loses to autoregressive on MoE.
- **Block-ship gate**: 18–24 tok/s ± with zero quality regression on Spec-Bench MT-Bench. If quality regresses, halt and debug.
- **Effort**: ~2 hr (bench + writeup).

### Routing recall fix (parallelizable with steps 12–16)

#### 11. 🔒 **Routing recall fine-tune**
- **What**: dedicated fine-tune pass on EAGLE-4 head with mask loss as primary objective. Current weights: token CE 1.0 + aux MSE 0.5 + mask BCE 0.3. New schedule: freeze token-CE + aux-MSE-trained block weights, fine-tune only mask_proj_in + mask_proj_out + a small fraction of the in_proj weights, with mask loss weight 5.0+. Target: ≥60% top-8 recall (MoE-SpeQ shows 90% is achievable with 4-bit quantized draft → 60% is conservative).
- **Files**: `eagle4/eagle4.py` (new `fine_tune_routing` subcommand or modify existing train), `eagle4/checkpoints/eagle4_v3/best_recall.npz` (new checkpoint).
- **Success**: `python eagle4/eagle4.py eval --ckpt eagle4/checkpoints/eagle4_v3/best_recall.npz ...` reports `mask_topk_mean_recall ≥ 0.60`. Token acceptance may drop 2–4 pp from 87.48% — acceptable trade.
- **Effort**: ~1 day (train + iterate). Runs on user's hardware in eagle4 venv, not on dismantle.

### Stage 2 — Path B parallel-K verify

The convergence doc and original brief together specify three kernels.
Implementation order is easiest first to validate the dispatch graph.

#### 12. **Path B kernel design pass — sync with eagle4 masked-verify intent**
- **What**: a single design doc covering all three Path B kernels (`gemv_q6_k_v3_kbatch`, `mla_decode_kernel_fc_kbatch`, `moe_block_batched_indexed_kbatch`) PLUS the masked-verify variant that consumes the 26×64 routing prediction. Path B + eagle4 masked-verify share the same dispatch primitive; designing them together avoids implementing the kernel twice.
- **Files**: extend `reports/path_to_90/path_b/design.md` with a § "Masked verify integration".
- **Success**: design doc has kernel signatures (`fn ...(x, routed_indices, predicted_mask, expert_weights, out)`), threadgroup memory budgets verified against M3 Pro's ~32 KB/core, dispatch flow diagram.
- **Effort**: ~half day.

#### 13. **`gemv_q6_k_v3_kbatch` (easiest — validates dispatch)**
- **What**: K-query batched variant of existing `gemv_q6_k_v3`. Grid `(vocab_rows / TG_ROWS, K)`; weight read shared across K threadgroup columns.
- **Files**: `crates/dismantle-core/shaders/parallel_k_lmhead.metal` (new), `crates/dismantle-core/src/kernels/parallel_k.rs` (replace `Unimplemented`), `crates/dismantle-core/tests/path_b_parity.rs` (un-`#[ignore]` the test).
- **Success**: K=4 parity vs K=4 sequential single-token GEMVs at atol=1e-3 fp16. Wall-clock ≤ 1.8× single-token decode.
- **Effort**: ~3–5 days.

#### 14. **`mla_decode_kernel_fc_kbatch` (hardest — threadgroup memory budget)**
- **What**: K queries against same KV cache; KV-cache read amortized across K. Function-constant specialize for `(n_heads, head_dim, K)` so compiler can fully unroll.
- **Files**: `crates/dismantle-core/shaders/parallel_k_attn.metal`, `parallel_k.rs`, parity test.
- **Risk**: threadgroup memory budget. Existing MLA uses most of TG SRAM. K=4 may force tile-size reduction — design pass (step 12) must verify before coding.
- **Effort**: ~5–7 days.

#### 15. **`moe_block_batched_indexed_kbatch_masked`**
- **What**: most algorithmically novel. K queries, each with top-6 routes that may overlap. Kernel batches K queries' expert calls, sharing weight reads when routes overlap. **Accepts `predicted_mask` from eagle4 head and uses it for async expert prefetch.** Ship no-overlap (K sequential expert calls in one CB) first to validate parity; add overlap optimization as second commit.
- **Files**: `crates/dismantle-core/shaders/parallel_k_moe_masked.metal`, kernel dispatcher, parity test (`crates/dismantle-core/tests/path_b_eagle4_parity.rs`).
- **Success**: K=4 masked-verify vs K=4 unmasked vs K=1 sequential at atol=1e-3 fp16. With v2-routing checkpoint (26% recall) the masked path should be ≥5% faster than unmasked due to async prefetch.
- **Effort**: ~5–7 days.

#### 16. **K-batched verify wire-up in deepseek_v2.rs**
- **What**: new method `forward_tokens_batched_parallel_k(tokens, positions, predicted_mask)` that routes through the three new kernels. Profile flag `verify_kernels = "parallel-k"` toggles.
- **Files**: `crates/dismantle-core/src/model/deepseek_v2.rs`, profile schema.
- **Success**: `dismantle generate --speculate eagle4 --verify-kernels parallel-k` runs; output bit-identical to sequential-verify path.
- **Effort**: ~half day.

#### 17. **STAGE 2 MEASUREMENT**
- **What**: bench with eagle4 chain + Path B parallel-K verify. Expected per deep research: **38–50 tok/s** (Mixtral EAGLE only hit 1.5× vs dense's 2.7×; ours is similarly constrained).
- **Files**: `reports/path_to_90/stage2_pathb.md`.
- **Block-ship gate**: ≥38 tok/s sustained, parity bit-identical at K=1.
- **Effort**: ~2 hr.

### Stage 3 — masked verify with prefetch

#### 18. **Mask-driven async expert prefetch**
- **What**: when masked-verify kernel sees a `predicted_mask` bit set, issue async prefetch of that expert's Q4 weight tiles into Apple Silicon's L2 / TG residency hint **before** the dispatch needs them. For experts not in the prediction set that turn out to fire (recall miss), fall back to on-demand load.
- **Files**: `parallel_k_moe_masked.metal` (prefetch hints via Metal residency API), kernel dispatcher.
- **Prerequisite**: step 11 routing recall ≥60%, ELSE this is a 5–10% win not a 15% win.
- **Success**: bench shows ≥5% improvement over step 16's unmasked baseline with `best_recall.npz` loaded.
- **Effort**: ~3 days.

#### 19. **STAGE 3 MEASUREMENT**
- **What**: bench. Expected: **55–75 tok/s** with `best_recall.npz`; 55–60 with original `best.npz`.
- **Files**: `reports/path_to_90/stage3_masked_verify.md`.
- **Effort**: ~2 hr.

### Stage 4 — DySpec-style dynamic tree decode

#### 20. **Tree decoding design — DySpec-style dynamic (not Sequoia fixed)**
- **What**: tree mask + propose_tree + per-token tree-shape calibration. Use eagle4's `calib_logit` to dynamically size the tree: high-calib positions get wider branching, low-calib get narrow. The Qwen3.6-A3B llama.cpp evidence shows fixed tree shapes hurt MoE because expert-union grows uncontrolled.
- **Files**: `reports/path_to_90/tree_decode/design.md` (already exists — extend with DySpec adaptation), new `crates/dismantle-core/src/speculate/tree.rs`.
- **Success**: design doc covers (a) tree-shape function `f(calib) → (depth, width)`, (b) tree attention mask construction, (c) verify-side accept/reject across tree branches.
- **Effort**: ~half day.

#### 21. **Tree decode implementation**
- **What**: per the design.
- **Files**: `crates/dismantle-core/src/speculate/tree.rs`, `crates/dismantle-core/src/model/deepseek_v2.rs` (route through tree verify), tests.
- **Success**: bench shows ≥1.4× over chain spec decode on Spec-Bench. Bit-identical greedy still holds.
- **Effort**: ~1 week.

#### 22. **STAGE 4 MEASUREMENT**
- **What**: bench. Expected: **70–95 tok/s** (MoE tree multiplier 1.4–1.8×, less than dense's 1.5–2× because expert-union grows with tree size — MoE-Spec Figure 2b: 127-token tree on OLMoE activates 54/64 experts).
- **Files**: `reports/path_to_90/stage4_tree.md`.
- **Effort**: ~2 hr.

### Stage 5 — hardware paths

In ROI order per deep research.

#### 23. **AMX draft head via direct cblas**
- **What**: implement Eagle4Head's GEMV operations against `Accelerate.framework`'s `cblas_sgemm` directly (NOT Core ML — Core ML adds 8× dispatch overhead on small ops per michaelstinkerings.org). Wire as a backend in `eagle4_head.rs` selectable via env var `EAGLE4_BACKEND=amx`.
- **Files**: `crates/dismantle-core/src/speculate/eagle4_head.rs`, new `crates/dismantle-core/src/util/amx.rs` (Accelerate bindings).
- **Success**: head forward ≤2 ms (vs ~5 ms on GPU per step 7's target). End-to-end +10–15% tok/s.
- **Effort**: ~3 days.

#### 24. **Per-head adaptive MLA KV quantization**
- **What**: keep MLA's "sink" latent dimensions (~first 32) at FP16; quantize the rest to Q4. Per llama.cpp Issue #21385, this recovers most of the quality vs flat Q4. Default is Q8 in latent space; Q4 gate-tested behind eval.
- **Files**: `crates/dismantle-core/src/metal/decode_arena.rs` (KV buffer layout), MLA kernel paths.
- **Success**: perplexity on wikitext2-256 within 1% of FP16 KV; bandwidth reduction measured; +5% tok/s.
- **Block-ship gate**: perplexity regression ≤1%, else don't ship.
- **Effort**: ~2 days.

#### 25. **Async verify-start**
- **What**: pipeline draft↔verify. Last draft step's hidden production overlaps first MoE verify layer's expert prefetch. Class B item but trivially landable.
- **Files**: `crates/dismantle-core/src/model/deepseek_v2.rs`, command-buffer fence rearrangement.
- **Success**: +5–8% tok/s on Spec-Bench.
- **Effort**: ~2 days.

#### 26. **ANE routing-logits offload (NOT verify)**
- **What**: V2-Lite's per-MoE-layer router (gate_logits = x @ gate_w) is small (`(1, 64)` output) and compute-bound. ANE can run it concurrent with Metal verify without contending on bandwidth. NOT for verify FFN (UMA contention kills the gain).
- **Files**: new `crates/dismantle-core/src/ane/router.rs`, Core ML model file generated from V2-Lite's router weights.
- **Success**: +5% tok/s. The contribution is capped — see deep research § Apple Silicon specifics.
- **Effort**: ~3 days.

#### 27. **Multi-queue Metal scheduling**
- **What**: separate command queue for draft vs verify so dispatch overlaps. Profile-first decision — only ship if both queues stay saturated without bandwidth contention.
- **Files**: `crates/dismantle-core/src/metal/mod.rs` (queue management).
- **Success**: +3–8% tok/s OR halt and don't ship if bandwidth-contended.
- **Effort**: ~2 days.

#### 28. **STAGE 5 MEASUREMENT — the headline number**
- **What**: full stack bench. Expected per deep research: **95–125 sustained, peak 135**.
- **Files**: `reports/path_to_90/stage5_final.md` — the project's headline.
- **Block-ship gate**: ≥95 sustained, peak ≥120 on code prompts.
- **Effort**: ~half day (multiple workloads, statistical CIs).

### Stretch (Class B post-eagle4-v4)

These are 6-month-horizon items. Do not start until Stage 5 ships.

#### 29. **SuffixDecoding/ngram-mod hybrid fallback**
- **What**: for code/repetitive prompts where ngram trees outperform learned drafters, route through `SpeculateMode::SuffixHybrid` that swaps the draft head based on prompt characteristics (predicted via simple n-gram entropy heuristic).
- **Why**: SuffixDecoding reports 5.3× on AgenticSQL with Llama-3.1-8B; 2.8× faster than EAGLE-2/3. Bimodal but worth ~10% on agentic/coding workloads on top of EAGLE-4.
- **Effort**: ~1 week.

#### 30. **Predict-routing-trace via Jakiro-style decoupling**
- **What**: eagle4's head currently predicts tokens; the routing mask is a side-output. Decouple: a small head predicts the next position's routing trace directly, used for prefetch BEFORE token prediction. Per arXiv 2502.06282.
- **Effort**: ~2 weeks.

#### 31. **Medusa-style multi-head stack**
- **What**: K parallel heads each predicting position +1, +2, ..., +K. Replaces eagle4's autoregressive multi-step rollout. On dense models Medusa adds 1.2–1.5×; on MoE expect 1.1–1.2× because Medusa doesn't help the MoE verify bottleneck.
- **Effort**: ~1 month (training-side mostly in eagle4).

#### 32. **Final paper / portfolio writeup**
- **What**: "Multi-axis speculative decoding on Apple Silicon: routing-aware EAGLE-4 + parallel-K MoE verify + DySpec tree decode + AMX/ANE hardware paths." Repro instructions, all benchmark CIs, ablations per stage.
- **Effort**: ~2 weeks of writing + 1 week of running ablation experiments.

## Operational notes

- Set `sudo sysctl iogpu.wired_limit_mb=14336` before benching — M3 Pro
  18 GB memory pressure can otherwise force swap that destroys
  throughput.
- All commits authored as Joshua Hicks via inline git identity per
  `CLAUDE.md`. No Co-Authored-By lines, no "Generated with" footers.
- Block-ship gates are the halt rule: below the lower bound on any
  stage measurement → halt, debug, re-plan. Don't paper over a
  regression.
- Each stage measurement is its own commit + report. The git log is
  the audit trail.

## Decisions outstanding (parking lot — punt to user)

1. **Cancel paused eagle3 capture?** It was at ~85/100000 samples when
   paused. Eagle4 already has the eagle3-baseline 75.84% acceptance on
   its own data. Finishing dismantle's capture would not add new info.
   **Recommend: cancel.** Frees ~4 days of overnight compute.
2. **Retire `tools/training/mlx_eagle/`?** eagle4 supersedes it
   entirely. Recommend: yes, in a cleanup commit after stage 1 measures.
3. **MLX-LM full port (step 2's "yes" path)?** Depends on step 1's
   profile. Decide after profiling.
4. **Q3 quantization sensitivity sweep?** Deep research says not worth
   it on V2-Lite. Skip unless we hit the bandwidth ceiling at stage 5
   and need additional headroom.

## Step-to-stage-to-target mapping

```
Foundation (steps 1–7)        →  enables all downstream
Stage 1 (steps 8–10)          →  12–22 tok/s (regression expected)
Recall fix (step 11)          →  prerequisite for stage 3
Stage 2 (steps 12–17)         →  38–50 tok/s — first real win
Stage 3 (steps 18–19)         →  55–75 tok/s
Stage 4 (steps 20–22)         →  70–95 tok/s
Stage 5 (steps 23–28)         →  95–125 tok/s sustained, 135 peak ← headline
Class B (steps 29–32)         →  130–160 sustained, 175 peak (6 months out)
```

The four headline numbers — Stage 2, Stage 4, Stage 5, Class B — are the
project's measurable milestones. Each is its own commit + report. Each
is a defensible portfolio claim if it ships.
