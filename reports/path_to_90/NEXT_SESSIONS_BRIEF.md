# Path-to-90: Brief for All Remaining Work (Multi-Session Handoff)

**Audience:** future session(s) — could be me with a fresh context window, you,
or a collaborator. Each section is a self-contained prompt that bootstraps the
next phase of work from current state.

**Current state snapshot** (as of commit `b121d9b`, 2026-05-17 ~21:30 EDT):

- Branch: `claude/dreamy-golick-d54ff8` on `joshuahickscorp/dismantle`
- Capture: PID 99052 running Phase 1 data (full UltraChat dialogues, single
  hidden, DCAP v1). ETA Sunday May 24. Writes to
  `training_data/c2_hidden/eagle3_v0/shard_000.bin`.
- Pipeline loop: PID 99053 polling every 60s; will auto-fire S1-S8 (tier1
  training + eval + stub) when capture completes.
- TIER2 short-circuited to ALL_DONE since TARGET=TIER2_TARGET=100K.
- Monitors: `bz8je83i5` (20-min pings), `bqsycr8ww` (stage notifications + 6h
  heartbeats).
- Two preserved legacy shards: `shard_v1_legacy_43k.bin` (user-prompts only,
  43K samples, EAGLE-1 recipe) and `shard_v2_partial_dcap1.bin` (full dialogue,
  ~30 samples, DCAP v1) — for ablation comparison if needed.

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
> 4. Modify the layer loop in `forward_token_final_norm_maybe_read` (the
>    Wedge C path, around line 2770-2905). Right after `encode_layer(li,
>    tcb)` returns, if `li` is in `capture_layer_indices`:
>    ```rust
>    let slot = self.capture_layer_indices.iter().position(|&x| x == li);
>    if let Some(slot) = slot {
>        let dst = &arena.multi_layer_capture_buf[slot];
>        let sz = (h * std::mem::size_of::<f32>()) as u64;
>        tcb.copy_buffer_bytes(&arena.x_buf, 0, dst, 0, sz)?;
>        crate::kernels::add_inplace_metal_tcb(tcb, dst, &arena.ffn_out_buf, h)?;
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
>     - Verify hidden values: layer 24's output should be ≈ pre-final-norm
>       state, comparable to existing forward_token_with_hidden_for_test
>     - Verify no NaN/Inf at layers 2, 14, 24
>     - Train a 100-step smoke on the 10-sample shard; verify loss is
>       finite and TTT + HASS components combine cleanly
>
> 15. Restart capture with `--capture-layers 2,14,24` against the existing
>     `tests/data/ultrachat_100k_v2_dialogue.jsonl`. Use a new shard path:
>     `training_data/c2_hidden/eagle3_v0/shard_v3_multilayer.bin`. Estimated
>     wall: ~7 days (same as Phase 1 since per-token forward dominates;
>     the multi-layer blit is sub-millisecond per token).
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
>    - Compare to legacy shard's results (run eval on the legacy
>      ckpts at `continuous_a_lion_next/at_005000/`)

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
> 5. CLI: `dismantle generate --speculate eagle --draft-head models/eagle3-v0.gguf`
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

```bash
nohup ./target/release/dismantle capture-hidden \
  --weights models/deepseek-v2-lite-q4.gguf \
  --samples tests/data/ultrachat_100k_v2_dialogue.jsonl \
  --out training_data/c2_hidden/eagle3_v0/shard_000.bin \
  --max-tokens 128 --no-lm-head --resume \
  --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
  >> training_data/c2_hidden/eagle3_v0/shard_000.log 2>&1 < /dev/null &
disown
```

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
