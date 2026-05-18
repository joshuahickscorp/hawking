# Path-to-90: Brief for All Remaining Work (Multi-Session Handoff)

**Audience:** future session(s) — could be me with a fresh context window, you,
or a collaborator. Each section is a self-contained prompt that bootstraps the
next phase of work from current state.

> **UPDATE 2026-05-18 — EAGLE-4 integration supersedes parts of this brief.**
> EAGLE-4 (`eagle4`) has shipped a trained head with **87.48%
> target-argmax acceptance** vs EAGLE-3's 75.84% (+11.64 pp). Its README
> names dismantle as its inference runtime. Read
> `reports/path_to_90/eagle4_convergence.md` BEFORE acting on Phase 2 or
> C3 below — that doc is now authoritative for the integration contract.
> Three changes to call out:
>
> 1. **Layer indices for multi-hidden capture are {2, 13, 25}**, not
>    {2, 14, 24} as Phase 2 §4 below originally said. Constants in
>    `crates/dismantle-core/src/speculate/eagle4_head.rs::cfg` are the
>    source of truth.
> 2. **Phase 2's dismantle-side capture extension is retired.** EAGLE-4
>    has its own MLX-native capture (`eagle4/capture.py`, 745 records/sec)
>    that produces the 4-hidden + routing parquets it trains on.
>    Dismantle owns *inference*; eagle4 owns *training*. The brief's
>    capture-hidden modification is no longer needed.
> 3. **C3 work changes shape.** Instead of building an EAGLE-3-style
>    `EagleDraftHead`, dismantle implements `Eagle4Head` (skeleton landed
>    this commit at `crates/dismantle-core/src/speculate/eagle4_head.rs`).
>    Forward pass = 5-input fusion → 1 transformer block → residual gate
>    → frozen LM head + mask + calib. Loads from eagle4's NPZ checkpoint.
>
> Path B (parallel-K verify kernels) and tree decoding remain valid and
> converge with eagle4's masked-verify need — see convergence doc for the
> combined kernel signature.

> **UPDATE 2026-05-18 (later) — execution plan now authoritative.**
> A deep-research pass refined the stage-5 ceiling from the V4 spec
> doc's 140–165 down to **95–125 sustained, peak 135**, surfaced the
> 17.78% routing mask recall as the biggest risk (vs MoE-SpeQ's 90%
> achieved on Mixtral), and validated that naive MoE spec-decode is
> documented as net-negative without K-batched + masked verify. The
> sequenced 28-step plan that incorporates all of this lives at:
>
> - `reports/path_to_90/execution_plan.md` — the 28-step batched
>   sequence. **Authoritative for ordering and block-ship gates.**
> - `reports/path_to_90/eagle4_deep_research.md` — the synthesis +
>   citations the plan derives from.
>
> Phase 2 / Phase 3 / Path B / C3 sections below remain in this doc
> as historical context for what the plan absorbs and supersedes; do
> NOT execute them directly. The plan re-orders, re-scopes, and adds
> measurement gates that the originals lack.

**Current state snapshot** (post-recovery, 2026-05-17 ~22:30 EDT — see
`reports/path_to_90/recovery_2026-05-17.md` for what changed and why):

- Branch: `claude/dreamy-golick-d54ff8` on `joshuahickscorp/dismantle`
- Worktree: `.claude/worktrees/dreamy-golick-d54ff8/` (recreated; previous
  one was deleted with its untracked training data in flight)
- Capture entrypoint: run `tools/training/launch_main_capture.sh` from
  the worktree root. It calls the committed binary against
  `tests/data/ultrachat_100k_union.jsonl` with `--max-samples 100000 --resume`,
  then launches `pipeline_loop.sh`. Both detach.
- TIER2 target: 100K samples via `--max-samples`. The union file itself
  contains 155K samples (55K + 100K disjoint sets); the cap is what
  bounds wall time to the brief's ROI plan.
- Legacy shards (`shard_v1_legacy_43k.bin`, `shard_v2_partial_dcap1.bin`):
  **gone** — they were untracked files in the deleted worktree and are
  not recoverable. Phase 3 ablation paths that referenced them are
  unreachable and have been removed below.
- Historical artifacts that survived in git: `shard_000.meta.json` and
  `shard_000.log` describe a prior 50K run that captured to ~14K samples;
  the .bin was never committed (gitignored, too large). Treat the meta
  as history, not a resumable state.

**Documents to read first** (in priority order):

1. `reports/path_to_90/stage3_c2/phase2_multi_hidden_design.md` — concrete Phase
   2 implementation spec
2. `reports/path_to_90/path_b/design.md` — Path B parallel-K verify kernel
   design
3. `reports/path_to_90/tree_decode/design.md` — tree decoding design
4. `reports/path_to_90/stage3_c1/architecture.md` — overall EAGLE-3 architecture
   decision
5. The deep-research conversation in Claude chat (anchors the EAGLE-3 paper
   recipe and where our work sits relative to SOTA)

---

## Phase 2: Multi-layer hidden capture + TTT + HASS top-K

**Trigger:** start when fresh (4-6 hr focused session). Don't wait for capture
to finish — Phase 2 changes are additive to the engine and don't affect the
running Phase 1 capture. When Phase 2 lands, restart capture with
Phase 1+2 combined; Phase 1's partial progress is preserved.

**Prompt:**

> Implement EAGLE-3 multi-layer hidden capture for dismantle, per
> `reports/path_to_90/stage3_c2/phase2_multi_hidden_design.md`. Goal: capture
> hidden states from layers {2, 14, 24} of DeepSeek-V2-Lite in a single
> forward pass, store in DCAP v2 binary format, train EAGLE-3-style head
> with multi-layer fusion.
>
> Steps in order:
>
> 1. Read the design note end-to-end. Verify your understanding of the layer
>    loop's buffer choreography (x_buf accumulates residuals; ffn_out_buf
>    is added at the START of the next layer's phase1, not at the end of
>    the current layer's encode). The capture blit must therefore be
>    `copy(x_buf → capture_buf) + add(capture_buf, ffn_out_buf)` so the
>    capture buffer holds the full post-layer-N output without disturbing
>    x_buf for the next layer.
>
> 2. Add to `crates/dismantle-core/src/metal/decode_arena.rs::DecodeArena`:
>    ```rust
>    pub multi_layer_capture_buf: Vec<PinnedBuffer>,   // len = N, each hidden×f32
>    pub multi_layer_capture_indices: Vec<usize>,       // which layer each slot captures
>    ```
>    Initialize as empty Vec by default. Add a method
>    `init_multi_layer_capture(&mut self, ctx, indices)` that allocates
>    `indices.len()` buffers and stores the layer indices.
>
> 3. Add field to `DeepSeekV2` struct: `capture_layer_indices: Option<Vec<usize>>`.
>    Set via `set_capture_layers(&mut self, layers: Vec<usize>)` (calls
>    arena.init_multi_layer_capture and stores indices).
>
> 4. Modify the layer loop in `forward_token_final_norm_maybe_read`
>    (`crates/dismantle-core/src/model/deepseek_v2.rs:2770`). Important:
>    `encode_layer` is a **closure** declared inside the for-loop at
>    line 2785 — `let encode_layer = |tcb: &mut TokenCommandBuffer<'_>|
>    -> Result<bool> { ... }`. It captures `li` from the enclosing scope
>    and returns whether the layer was a real layer (vs skipped). It is
>    invoked from two sites:
>      - **single-TCB branch** at line 2867: `encode_layer(global_tcb.as_mut().unwrap())?`
>      - **per-layer-fallback branch** at line 2871: `encode_layer(&mut tcb)?`
>
>    Insert the capture blit AFTER the `encode_layer(...)?` call in both
>    branches (or scope the Phase 2 work to single-TCB only and skip the
>    fallback — the fallback path is rarely-used; cleanest is to patch
>    only single-TCB first, validate, then patch fallback if needed).
>    The blit itself:
>    ```rust
>    if let Some(ref capture_idx) = self.capture_layer_indices {
>        if let Some(slot) = capture_idx.iter().position(|&x| x == li) {
>            let dst = &arena.multi_layer_capture_buf[slot];
>            let sz = (h * std::mem::size_of::<f32>()) as u64;
>            tcb.copy_buffer_bytes(&arena.x_buf, 0, dst, 0, sz)?;
>            crate::kernels::add_inplace_metal_tcb(tcb, dst, &arena.ffn_out_buf, h)?;
>        }
>    }
>    ```
>    This does NOT modify x_buf (next layer's phase1 add_inplace continues
>    to add ffn_out_buf into x_buf as designed).
>
> 5. After the global TCB commits (around line 2960), read back each
>    capture buffer into a Vec<f32>. Return them packed as
>    `Vec<Vec<f32>>` indexed by capture slot. Don't disturb the existing
>    final-norm read-back path.
>
> 6. Add to `crates/dismantle-core/src/engine.rs::Engine` trait:
>    ```rust
>    fn forward_token_multi_hidden_for_test(
>        &mut self,
>        _token: u32,
>        _pos: usize,
>        _capture_layers: &[usize],
>    ) -> Result<Vec<Vec<f32>>> {
>        Err(crate::Error::Unimplemented("forward_token_multi_hidden_for_test"))
>    }
>    ```
>    Default `Unimplemented`. DeepSeekV2 impl: set capture indices, run
>    forward_token_final_norm, read back from arena.multi_layer_capture_buf.
>
> 7. Update `crates/dismantle/src/main.rs` capture-hidden subcommand:
>    - Add `--capture-layers` flag (comma-separated list, default "2,14,24")
>    - When provided, route through `forward_token_multi_hidden_for_test`
>      and write DCAP v2 records
>
> 8. DCAP v2 binary format (in `crates/dismantle/src/main.rs` capture_hidden_main):
>    - Header (16 bytes):
>      ```
>      0..4   magic = b"DCAP"
>      4..8   version = 2
>      8..12  hidden_dim
>      12..14 n_hiddens_per_record (u16)
>      14..16 reserved (u16 = 0)
>      ```
>    - Record:
>      ```
>      u16 id_len + utf8 sample_id
>      u32 pos
>      u32 prev_token
>      u32 next_token
>      N × hidden_dim × 2 bytes  (N hiddens packed f16)
>      ```
>    - Bump VERSION constant from 1 to 2 in capture_hidden_main.
>
> 9. Update Python `tools/training/capture_hidden.py::_iter_records` to
>    handle both DCAP v1 (n_hiddens=1) and v2 (n_hiddens≥1). Return
>    hidden_bytes as a list per record.
>
> 10. Update `tools/training/mlx_eagle/data.py` to load N hiddens per
>     record. The batch dict gains a `target_hidden_layers` key of shape
>     (B, S, N, H). For backward compat, if loading a DCAP v1 file, set
>     N=1 and the existing single-hidden code path works.
>
> 11. Update `tools/training/mlx_eagle/model.py::EagleHead`:
>     - Add `n_hidden_layers: int` to `EagleHeadConfig` (default 1)
>     - When n_hidden_layers > 1, add a fusion projection
>       `nn.Linear(n_hidden_layers * hidden_dim, hidden_dim)` that
>       projects concatenated multi-layer hidden back to hidden_dim
>     - Drop into the existing forward path as the input feature
>     - Trainable params count rises ~+12.6M for 3 layers × 2048 hidden
>
> 12. Add training-time test (TTT) to train.py per EAGLE-3 §3.2 Fig. 6:
>     - For each batch, after forward, generate a simulated "next step"
>       by replacing the next position's target_hidden with the head's
>       own predicted draft_hidden (detached)
>     - Run a second forward with the substituted hidden, compute CE loss
>     - Sum the original CE + TTT CE losses (paper uses equal weight or
>       0.5 + 0.5)
>     - Implementation hint: do this with a diagonal attention mask in
>       the same forward pass for efficiency, or accept the 2x cost for
>       a simpler implementation
>     - Default to 2 TTT steps (paper uses 1-3)
>
> 13. Add HASS top-K=10 distillation loss (arXiv:2408.15766):
>     - Modify the loss function: instead of CE over full vocab, mask
>       logits to keep only the top-K (default 10) tokens of the target's
>       argmax distribution, then CE on those
>     - Add `--hass-topk N` CLI flag (default 10, 0 disables)
>     - Add it ADDITIVELY to the CE + TTT losses with weight 1.0
>
> 14. Sanity test before full capture:
>     - Run `dismantle capture-hidden --capture-layers 2,14,24` on the
>       10-sample wikitext smoke
>     - Verify DCAP v2 file structure (magic, version=2, n_hiddens=3,
>       record size = 14 + id_len + 3*4096 bytes)
>     - Verify hidden values: layer 24's capture is PRE-final-norm — the
>       blit lands `x_buf + ffn_out_buf` before the final RMSNorm runs.
>       `forward_token_with_hidden_for_test` returns POST-final-norm
>       `x_norm`, so a direct equality check will always fail. Two options:
>       (a) apply `final_norm` to the captured tensor before comparing,
>       or (b) add a `forward_token_with_pre_norm_for_test` trait method
>       that returns the pre-norm value and compare against that.
>     - Verify no NaN/Inf at layers 2, 14, 24
>     - Train a 100-step smoke on the 10-sample shard; verify loss is
>       finite and TTT + HASS components combine cleanly
>
> 15. Restart capture with `--capture-layers 2,14,24` against
>     `tests/data/ultrachat_100k_union.jsonl` (the committed-pipeline
>     input; the previously-referenced `ultrachat_100k_v2_dialogue.jsonl`
>     was an off-script file from a prior session and is gone). Use a new
>     shard path: `training_data/c2_hidden/eagle3_v0/shard_v3_multilayer.bin`.
>     Estimated wall: ~7 days (same as Phase 1 since per-token forward
>     dominates; the multi-layer blit is sub-millisecond per token).
>
> 16. Update `tools/training/advance_pipeline.sh` defaults to point at the
>     new shard + add `--multi-hidden` / `--ttt` / `--hass-topk` to the
>     S5/S11 training invocations.
>
> 17. Commit + push as `path-to-90 C2 Phase 2: multi-layer hidden + TTT + HASS`.
>
> Estimated effort: 4-6 hours focused (1 hr arena + trait, 2 hr layer-loop
> blit + read-back, 1 hr DCAP v2 + Python, 1 hr TTT + HASS, 30 min smoke
> + restart). Risk: medium — the layer loop modification needs to leave
> x_buf untouched so subsequent layers work correctly.

---

## Phase 3: After capture completes (Sunday-Monday)

**Trigger:** pipeline_loop fires S1 (stage marker `S1_PARQUET_DONE`)
automatically. Monitor `bqsycr8ww` will emit the notification.

**Prompt:**

> Pipeline auto-fired S1-S8. Validate each stage's outputs and the first
> real acceptance number:
>
> 1. Read `reports/path_to_90/stage3_c2/eval_55k.json` (or `eval_100k.json`
>    depending on which tier triggered). The headline is `accept_top1`.
>    Per deep research: expected 40-65% for our Phase 1 setup
>    (EAGLE-3-data + EAGLE-1-architecture). If Phase 2 was applied,
>    expected 65-80%.
>
> 2. Read `reports/path_to_90/stage3_c2/spec_stub_55k.json`. The headline
>    is `headline_metrics.speedup_vs_no_spec_K_verify`. Per the spec-decode
>    arithmetic in `stage3_spec/audit.md`, this should be >1.0 only if
>    accept_top1 > 75% (because verify cost is K× single-forward without
>    Path B). Expect <1.0 (regression) unless Path B kernels have also
>    landed.
>
> 3. Write `reports/path_to_90/stage3_c2/close.md` capturing:
>    - Final acceptance number(s)
>    - Training loss curve (`tools/training/mlx_eagle/ckpt_55k/train_log.json`)
>    - Honest comparison to paper expectations (per deep research)
>    - Whether Phase 2 was applied or deferred
>    - Followups (Path B kernels, tree decoding, C3 wire-up)
>
> 4. Commit + push. If accept_top1 < 40%, debug:
>    - Check the train log for loss divergence
>    - Check the eval harness for off-by-one in next-token alignment
>    - Compare to the historical 5K-shard baseline metrics noted in
>      `reports/path_to_90/session_closeout.md` (legacy ablation
>      checkpoints referenced in prior brief versions no longer exist;
>      see `recovery_2026-05-17.md`)

---

## Path B: Parallel-K verify kernels (multi-week)

**Trigger:** start any time; doesn't depend on capture or training. Pure
engine work. Multi-week scope (3-4 weeks elapsed). See
`reports/path_to_90/path_b/design.md`.

**Prompt:**

> Implement Path B per the design doc. Three kernels in order of
> implementation difficulty (easiest first to validate the dispatch graph):
>
> 1. `gemv_q6_k_v3_kbatch` (~3-5 days). Existing `gemv_q6_k_v3` is the
>    single-token version. The kbatch variant processes K queries
>    against the same Q6_K weight matrix, sharing the weight read across
>    K threadgroup columns. Grid: `(vocab_rows / TG_ROWS, K)`. Validate
>    against K=4 sequential single-token GEMVs at atol=1e-3 fp16
>    (existing parity gate).
>
>    Files: `crates/dismantle-core/shaders/parallel_k_lmhead.metal` (new),
>    `crates/dismantle-core/src/kernels/parallel_k.rs` (replace
>    Unimplemented body with real dispatch), `crates/dismantle-core/tests/
>    path_b_parity.rs` (un-#[ignore] the gemv_q6_k_v3_kbatch test).
>
>    Success metric: at K=4 with synthetic-100%-acceptance verify,
>    measured wall-clock per spec step ≤ 1.8× single-token decode
>    wall-clock. If > 2.5×, the K-batch dispatch is wrong.
>
> 2. `mla_decode_kernel_fc_kbatch` (~5-7 days). Same KV cache, K queries.
>    The KV-cache read is the dominant cost (most of decode time) and
>    K-sharing amortizes it. Threadgroup memory budget is the tightest
>    constraint — existing MLA uses most of available TG SRAM; K-batching
>    may force tile-size reduction. Function-constant specialize for
>    (n_heads, head_dim, K) so the compiler can fully unroll.
>
>    Files: `crates/dismantle-core/shaders/parallel_k_attn.metal` (new),
>    parallel_k.rs + tests as above.
>
> 3. `moe_block_batched_indexed_kbatch` (~5-7 days). Most algorithmically
>    novel: each query has top-k=6 routes, may overlap with other
>    queries' routes. The kernel batches K queries' expert calls,
>    sharing expert weight reads when routes overlap. Ship a
>    no-overlap version first (just K sequential expert calls in one
>    CB) to validate parity; add overlap optimization as a second
>    commit.
>
> 4. Engine wire-up in `crates/dismantle-core/src/model/deepseek_v2.rs`:
>    new method `forward_tokens_batched_parallel_k(tokens, positions)`
>    that routes through the three new kernels instead of the sequential
>    loop. Profile flag `verify_kernels = "parallel-k"` defaults to
>    "sequential"; the new path activates only with the flag.
>
> 5. Spec-decode integration: when spec-decode is on and the profile
>    selects parallel-k, route verify through the new path. The trained
>    EAGLE-3 head's draft proposals get verified at ~1.5× single-forward
>    cost instead of K× single-forward.
>
> 6. Autotune sweep: re-run `tools/bench/autotune_sweep.sh` for the new
>    kernels at context lengths {128, 512, 1K, 4K, 16K, 32K}.
>
> Total: ~3-4 weeks. Correctness gate at every step. Commit per-kernel
> with the kernel's parity test green.

---

## Tree decoding implementation

**Trigger:** after Phase 2 lands AND at least 1 trained head exists.
Tree decoding extends Path B (the MLA kernel needs an attention mask
argument). See `reports/path_to_90/tree_decode/design.md`.

**Prompt:**

> Implement EAGLE-3 tree decoding per the design doc. Builds on Path B's
> parallel-K kernels.
>
> 1. Engine module `crates/dismantle-core/src/speculate/tree.rs`:
>    - `pub struct TreeProposal { node_tokens, node_parents, node_paths, node_hidden }`
>    - `pub fn tree_attention_mask(parents: &[i32]) -> Vec<Vec<f32>>` —
>      builds the (N, N) mask where mask[i,j]=0 if j is ancestor of i,
>      else -inf
>    - `pub fn longest_matching_path(proposal, verifier_argmax) -> Vec<u32>`
>
> 2. Extend `Path B`'s MLA kernel with an optional `(K, K)` attention
>    mask argument. Add `mla_decode_kernel_fc_kbatch_masked`. The mask
>    is causal-equivalent for linear K-spec, tree-structured for tree
>    spec.
>
> 3. Extend `DraftHead` trait in
>    `crates/dismantle-core/src/speculate/draft_head.rs`:
>    ```rust
>    fn propose_tree(&mut self, prev_token: u32, hidden: &[f32], topology: &[usize])
>        -> Result<TreeProposal> { Err(Unimplemented("propose_tree")) }
>    ```
>    The Python-side EagleHead.propose_tree (already shipped) is the
>    reference algorithm. Implement on the future EagleDraftHead Rust
>    impl.
>
> 4. Calibration script `tools/training/mlx_eagle/calibrate_tree.py`:
>    sweeps topologies on a held-out set, picks the one maximizing
>    expected tokens-per-verify. Output:
>    `reports/path_to_90/tree_decode/topology.json`.
>
> 5. Initial topology: `[3, 2, 2, 1, 1]` = 21 nodes. Adjust based on
>    calibration.
>
> Total: ~3-4 weeks AFTER Path B + first trained head exist.

---

## C3 wire-up: EagleDraftHead + SpeculateMode::Eagle

**Trigger:** after a trained EAGLE-3 head exists (post-Phase 3 or
post-Phase 2 capture+train).

**Prompt:**

> Wire the trained EAGLE-3 head into dismantle's decode path.
>
> 1. Convert MLX checkpoint to GGUF. Write
>    `tools/training/mlx_eagle/convert_to_dismantle_head.py`:
>    - Reads `tools/training/mlx_eagle/ckpt_55k/latest.npz`
>    - Writes a `models/eagle3-v0.gguf` with tensor layout dismantle's
>      EagleDraftHead loader expects
>    - Format: GGUF v3 with custom tensor names
>      (`eagle.in_proj.weight`, `eagle.block.attn.{q,k,v,o}.weight`,
>      `eagle.block.mlp.{gate,up,down}.weight`, `eagle.final_norm.weight`)
>    - Header metadata: `eagle.hidden_dim=2048`, `eagle.n_hidden_layers=N`,
>      `eagle.target_model="DeepSeek-V2-Lite-Chat"`
>
> 2. Rust `EagleDraftHead` impl in
>    `crates/dismantle-core/src/speculate/draft_head.rs`:
>    ```rust
>    pub struct EagleDraftHead {
>        hidden_dim: usize, vocab_size: usize,
>        in_proj: GgufTensor, block_attn_q/k/v/o: GgufTensor,
>        block_mlp_gate/up/down: GgufTensor,
>        final_norm: GgufTensor,
>        // Frozen target lm_head shared via reference
>    }
>    impl DraftHead for EagleDraftHead {
>        fn propose(&mut self, prev_token, hidden, k) -> Result<Vec<u32>> {
>            // Run 1-layer transformer forward via dismantle's existing
>            // kernel infrastructure. Output top-k tokens of
>            // (head_output @ lm_head).
>        }
>    }
>    ```
>
> 3. Add `SpeculateMode::Eagle` to `EngineConfig`:
>    ```rust
>    pub enum SpeculateMode {
>        Off, ExactShared, NGram, Eagle,
>    }
>    ```
>
> 4. In `model/deepseek_v2.rs` decode path, when `SpeculateMode::Eagle`:
>    - Load `models/eagle3-v0.gguf` at engine init
>    - Per decode step: capture hidden via existing forward_token_with_hidden,
>      call head.propose(prev_token, hidden, K-1), verify via the existing
>      forward_tokens_batched_for_test path (or Path B's
>      forward_tokens_batched_parallel_k if profile flag set)
>    - Accept longest matching greedy prefix
>    - Roll back KV on rejection
>
> 5. CLI (NEW flags — design intent, not present in main.rs today):
>    `dismantle generate --speculate eagle --draft-head models/eagle3-v0.gguf`.
>    The `--draft-head` flag and the `eagle` value for `--speculate` are
>    both new and must be added during C3 wire-up.
>
> 6. Bit-identical greedy regression: with `SpeculateMode::Eagle` on
>    against any prompt, output must match `SpeculateMode::Off` greedy
>    output at 64 tokens (the spec-decode invariant).
>
> 7. Bench harness: `dismantle bench --suite decode --speculate eagle`
>    gives the headline number.
>
> Total: 1-2 weeks.

---

## EAGLE-4 research extensions (optional, multi-month)

Per the EAGLE-4 audit in the deep-research thread. If you commit to a
~2-3 month research project beyond the path-to-90 deliverable:

**Learned tree topology** (cleanest single-paper-worth innovation):
- New `tree_topology_predictor` head that takes (target_hidden, uncertainty)
  → predicts per-position branching factor
- Train jointly with the EAGLE-3 head
- Replace static topology with per-position-learned one at inference

**M-series-specific kernel fusion** (real systems contribution):
- Custom Metal kernel that fuses draft head's forward + verify dispatch
  into a single command buffer
- Eliminates the CPU round-trip between draft and verify
- Uniquely valuable on Apple Silicon where memory bandwidth is the
  binding constraint

**Cross-paper writeup angle**: "Multi-layer EAGLE-3 + learned topology +
M-series kernel fusion on DeepSeek-V2-Lite Q4_K_M, achieving X tps where
prior SOTA (llama.cpp spec-decode) was Y tps." Concrete benchmark,
reproducible code, novel technique. Strong portfolio piece.

---

## How to resume capture if it dies

Use the committed launcher (recreated post-recovery):

```bash
./tools/training/launch_main_capture.sh
```

It sanity-checks the build + weights + union file, then launches the
capture and pipeline_loop detached. Equivalent to:

```bash
nohup nice -n 19 taskpolicy -b ./target/release/dismantle capture-hidden \
  --weights models/deepseek-v2-lite-q4.gguf \
  --samples tests/data/ultrachat_100k_union.jsonl \
  --out training_data/c2_hidden/eagle3_v0/shard_000 \
  --max-tokens 128 --max-samples 100000 --no-lm-head --resume \
  --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
  >> training_data/c2_hidden/eagle3_v0/shard_000.log 2>&1 < /dev/null &
disown
```

Note `--out` takes a prefix (no `.bin`); the binary writes `<prefix>.bin`
and `<prefix>.meta.json`. The union file is `ultrachat_100k_union.jsonl`
(155K samples; `--max-samples 100000` caps to the ROI plan); the prior
brief's `ultrachat_100k_v2_dialogue.jsonl` was an off-script file from
a session whose worktree was deleted — gone, do not look for it.

## How to restart pipeline_loop

```bash
nohup bash tools/training/pipeline_loop.sh \
  > training_data/c2_hidden/eagle3_v0/pipeline/loop.log 2>&1 < /dev/null &
disown
```

## How to halt the pipeline

```bash
touch training_data/c2_hidden/eagle3_v0/pipeline/HALT
```
