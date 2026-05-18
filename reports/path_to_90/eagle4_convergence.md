# EAGLE-4 convergence — dismantle integration plan

**Status:** prep work. EAGLE-4 has shipped trained heads + offline metrics
under `eagle4/` (in-tree as of the eagle4 drop commit; formerly a
standalone private repo at `~/Downloads/eagle4`). Its README explicitly
names dismantle as the inference runtime: *"The runtime that turns
offline acceptance into wall-clock tps lives separately in dismantle —
this repo is the head architecture and training pipeline."* This doc
captures the integration contract.

## What EAGLE-4 actually is (post-training, current state)

- **Trained head**: 60M params, single transformer block + 5-input fusion.
- **v2-spec checkpoint** (`best.npz`): **87.48% target-argmax acceptance**
  on the held-out shard. EAGLE-3 baseline on the same data: **75.84%**.
  +11.64 pp absolute / +15% relative.
- **v2-routing checkpoint** (`best_routing.npz`): 84.16% acceptance,
  **26% mask top-8 recall**. Better for prefetch-heavy runtimes.
- **Architectural trick**: residual gate (`α`, init ≈ 0.05) on top of the
  baseline `post_norm(h_high)` path. At init, draft output ≈ V2-Lite's
  own argmax. Training only learns a small refinement.
- **Wall-clock tps is not yet measured.** That requires this integration.

## EAGLE-4 forward (the contract dismantle must implement)

Per-token inputs from a V2-Lite forward pass:

| Name        | Shape    | Source                                                |
|-------------|----------|-------------------------------------------------------|
| `prev_token`| int32    | the token whose successor we're predicting           |
| `h_low`     | fp16[2048] | output of decoder **layer 2**                       |
| `h_mid`     | fp16[2048] | output of decoder **layer 13**                      |
| `h_high`    | fp16[2048] | output of decoder **layer 25**                      |
| `h_shared`  | fp16[2048] | last MoE layer's shared-expert applied to *that layer's pre-MLP input* |

> **Note on layer indices**: 0-indexed. V2-Lite has 27 decoder layers
> (0..26); layer 0 is dense, layers 1..26 are MoE. EAGLE-4 picks low/mid/high
> at {2, 13, 25}. The dismantle brief previously listed {2, 14, 24} — that
> was a rough heuristic and should be **changed to match eagle4 exactly**.

Forward (verbatim from `eagle4.py:114-128`):

```python
prev_embed     = V2Lite_token_embd[prev_token]
x              = concat(prev_embed, h_low, h_mid, h_high, h_shared)    # 5×2048
x              = Linear(5H → H)(x)                                      # in_proj
x              = TransformerBlock(x, diagonal_attn_mask)                # 1 block
draft_hidden   = post_norm(h_high) + α · x                              # residual gate
token_logits   = draft_hidden @ V2Lite_lm_head                          # frozen LM head
mask_logits    = Linear(512 → 26·64)(SiLU(Linear(H → 512)(draft_hidden)))
calib_logit    = Linear(H → 1)(draft_hidden)                            # P(accept)
```

The transformer block is RMSNorm → MultiHeadAttention(H=2048, 16 heads, no
bias) → residual → RMSNorm → SwiGLU(H, 5632) → residual. Diagonal attn mask
makes positions independent — only required during *training*, where it
packs (B, S) batches; at inference each token is a single position so the
mask is trivial.

## Required dismantle changes

### 1. Generalize `DraftHead` trait (small, additive)

Current `DraftHead::propose(prev_token, hidden: &[f32], k)` takes a single
hidden. EAGLE-4 needs four. The clean fix is to pass a struct:

```rust
pub struct DraftInputs<'a> {
    pub prev_token: u32,
    pub hiddens: &'a [&'a [f32]],  // length matches n_hiddens()
}

pub struct DraftOutputs {
    pub tokens: Vec<u32>,                  // top-k candidates
    pub routing_mask: Option<Vec<u8>>,     // 26 × 64 packed (eagle4 only)
    pub calib: Option<f32>,                // P(accept) (eagle4 only)
}

pub trait DraftHead: Send + Sync {
    fn propose(&mut self, inputs: &DraftInputs, k: usize) -> Result<DraftOutputs>;
    fn hidden_dim(&self) -> usize;
    fn n_hiddens(&self) -> usize;       // 1 = eagle3-style, 4 = eagle4
    fn reset(&mut self) {}
    fn id(&self) -> &str;
}
```

Files touched:

- `crates/dismantle-core/src/speculate/draft_head.rs` — trait + structs.
  Keep `NoopDraftHead` adapter; bit-identical-greedy regression still holds.

### 2. New `Eagle4Head` impl

New file: `crates/dismantle-core/src/speculate/eagle4_head.rs`.

Loads an NPZ checkpoint produced by `eagle4.py` and runs the head forward.
NPZ format is a ZIP of `.npy` arrays — well-defined, no external dep
beyond what dismantle already has for tensor I/O. Key naming follows
`_flat_params` walker in `eagle4.py:143-151`:

```
in_proj.weight              (HIDDEN, 5*HIDDEN)         fp16/bf16
block.attn_norm             (HIDDEN,)                  fp32
block.attn.{q,k,v,o}_proj.weight  (HIDDEN, HIDDEN)     fp16/bf16
block.mlp_norm              (HIDDEN,)                  fp32
block.mlp.{gate,up}.weight  (5632, HIDDEN)             fp16/bf16
block.mlp.down.weight       (HIDDEN, 5632)             fp16/bf16
residual_gate               (1,)                       fp32
mask_proj_in.weight         (512, HIDDEN)              fp16/bf16
mask_proj_out.weight        (26*64, 512)               fp16/bf16
calib_proj.weight           (1, HIDDEN)                fp16/bf16
calib_proj.bias             (1,)                       fp16/bf16
```

Plus the frozen weights from `v2lite_frozen.npz`:

```
token_embd                  (HIDDEN, vocab=102400)     fp16    (transposed)
lm_head                     (HIDDEN, vocab=102400)     fp16    (transposed)
output_norm                 (HIDDEN,)                  fp32
```

The frozen weights are already in dismantle's loaded V2-Lite GGUF — no
need to load `v2lite_frozen.npz` separately at runtime. The CLI just needs
to confirm the same model is in use.

### 3. Expose 5-input capture from V2-Lite

New trait method on `Engine`:

```rust
fn forward_token_eagle4_for_test(
    &mut self,
    _token: u32,
    _pos: usize,
) -> Result<Eagle4Inputs> {
    Err(crate::Error::Unimplemented("forward_token_eagle4_for_test"))
}

pub struct Eagle4Inputs {
    pub prev_token: u32,        // same as input
    pub h_low: Vec<f32>,        // layer 2 output
    pub h_mid: Vec<f32>,        // layer 13 output
    pub h_high: Vec<f32>,       // layer 25 output
    pub h_shared: Vec<f32>,     // last MoE shared-expert
}
```

DeepSeekV2 impl: capture during the layer loop in
`forward_token_final_norm_maybe_read` (deepseek_v2.rs:2770-2905), same
choreography as the brief's Phase 2 multi-hidden design but with **layer
indices {2, 13, 25}** (not {2, 14, 24} as the brief originally said) and
an additional 4th slot for `h_shared`.

`h_shared` is the LAST MoE layer's shared-expert output applied to that
layer's pre-MLP input. dismantle already has `ffn_shared_only(li, x)` at
`crates/dismantle-core/src/model/deepseek_v2.rs:3811` — calling that with
`li = 26` (the last MoE layer index in V2-Lite's 27-layer config) and
`x = <layer 26's pre-MLP input>` produces exactly the vector eagle4
captures. The pre-MLP input is what's fed into the MoE block at layer 26;
that lives in `arena.x_norm` (post-attn-rmsnorm) at the start of the
layer's MLP phase. Capture via blit, same shape as the other three
hiddens.

### 4. CLI wire-up

New flag on `dismantle generate`:

```
dismantle generate \
  --speculate eagle4 \
  --draft-head eagle4/checkpoints/best.npz \
  --calib-threshold 0.5    # optional; below this, fall back to autoregressive
  ...
```

Add `SpeculateMode::Eagle4` to `engine.rs:56`. The decode path in
`model/deepseek_v2.rs` routes through:

1. Capture 5 inputs at the current step
2. `head.propose(inputs, K)` → up to K tokens + routing_mask + calib
3. If `calib < threshold`: skip verify (autoregressive single step)
4. Else: run `forward_tokens_batched_for_test` on the K candidates,
   accept longest matching greedy prefix
5. Future: masked-verify kernel uses the routing_mask to prefetch /
   skip experts on the verify pass

### 5. Bit-identical greedy regression

`SpeculateMode::Eagle4` against any prompt, K=4, against
`SpeculateMode::Off` greedy at 64 tokens: outputs must match exactly. The
spec-decode acceptance criterion (longest-matching-prefix vs verifier
greedy) preserves this by construction.

## What dismantle does **not** need to do

These were on the dismantle brief but become unnecessary now:

- **Phase 2 multi-hidden capture extension in `capture-hidden`** — EAGLE-4
  has its own MLX-native capture (`capture.py`, 745 records/sec, runs against
  V2-Lite in bf16 via mlx-lm, ~2 min for 100K records). Dismantle's
  Metal-side Q4-dequant capture-hidden at ~95 records/sec is much slower
  and doesn't produce the eagle4 format. The right call: **eagle4 owns
  capture, dismantle owns inference.**

- **`mlx_eagle/` head training stack in dismantle** — EAGLE-4 has already
  trained both eagle3 and eagle4 heads. Dismantle just loads the .npz.
  We can retire `tools/training/mlx_eagle/` in a follow-up.

- **The currently-running eagle3 capture (PID was 40658)** — produces
  single-hidden DCAP v1 data that dismantle's `mlx_eagle/` would train an
  EAGLE-3 head on. Eagle4 has already produced 75.84% accept on the same
  baseline. Our capture wouldn't beat that. **Decision pending**: cancel
  to free compute, or run for an independent reproducibility check.

## What dismantle **does** need to do (revised priority)

In order:

1. **Generalize `DraftHead` trait + add `Eagle4Head` skeleton with Unimplemented bodies.** Code-only, no Metal. cargo check passes. **(prep work, this commit)**
2. **NPZ loader for the head .npz**. Code-only, no Metal. Tests against
   a small synthetic npz. **(small, 1-2 hr)**
3. **Engine trait method `forward_token_eagle4_for_test`** with the 5-input bundle. **(small)**
4. **Implementation of `Eagle4Head::propose`** — the 5-input fusion → block → residual-gate forward. Initially CPU-only via existing `gemv_f32` kernels for parity validation against eagle4's Python forward. **(half day)**
5. **Metal-accelerated version** of the head forward path. The block's GEMV/GEMM ops can reuse dismantle's existing kernels. The head is small enough (~60M params) that even unoptimized Metal would dominate Python by 5-10×. **(half day)**
6. **CLI + decode path wire-up.** `--speculate eagle4` route. Bit-identical-greedy regression. **(half day)**
7. **Masked-verify kernel** — uses `mask_logits` to skip experts on verify pass. **This is where Path B converges with eagle4's runtime requirements.** See "Path B reconciliation" below. **(1-2 weeks)**

## Path B reconciliation

The dismantle brief plans **"Path B" — parallel-K verify kernels** (LM head,
attention, MoE block) so verify of K candidate tokens is one batched
dispatch rather than K sequential single-token forwards. Without Path B,
spec-decode shows `speedup < 1.0` because verify cost overwhelms acceptance
gain.

EAGLE-4's mask output adds an orthogonal lever: at each MoE layer, the
mask predicts which experts will fire. Verify can skip dequant+matmul for
masked-out experts.

The two combine. The kernel signature dismantle needs is:

```rust
fn moe_block_batched_indexed_kbatch_masked(
    x: &[f32; K * hidden],
    routed_indices: &[u32; K * top_k],   // per-token actual routes (from running router on K)
    predicted_mask: &[u8; K * 64],        // top-8 prediction per token, eagle4 output
    expert_weights: &ExpertCache,
    out: &mut [f32; K * hidden],
) -> Result<KernelStats>;
```

The dispatch: union the K candidates' routes. For each expert in the
union, dispatch *once* with all K queries that hit it (K-batching). For
experts NOT in the union but PRESENT in `predicted_mask`, prefetch
asynchronously. For experts NOT in `predicted_mask`: dispatch on the
fallback path with measured cost.

The "experts in predicted_mask but not in actual routes" cost is a metric:
**predicted mask precision**. The "experts in routes but NOT predicted":
those force fallback dispatch, the **predicted mask recall** metric. Eagle4
v2-spec ships at 17.78% recall, v2-routing at 26%. The runtime is robust
to either; just trades prefetch effectiveness.

Files:

- `crates/dismantle-core/shaders/parallel_k_moe_masked.metal` (new) — fused K-query expert kernel that consumes the union of routes
- `crates/dismantle-core/src/kernels/parallel_k.rs` — dispatch logic, prefetch orchestration
- `crates/dismantle-core/tests/path_b_eagle4_parity.rs` — K=4 masked vs K=4 unmasked vs K=1 sequential at atol=1e-3 fp16

This is the **single biggest convergence opportunity** between the two
projects. Before doing Path B independently, sync the kernel design with
eagle4-core's masked-verify intent.

## Beyond eagle4 (parking lot)

EAGLE-4 v3/v4 specs in `eagle4/V3.md` and `V4.md` exist but the
current trained head is at "v2-spec." When v3/v4 land, dismantle's
masked-verify kernel framework should absorb them additively:

- **v3 item 3 (multi-step draft)**: head predicts K positions in one
  forward pass. Dismantle's `head.propose` returns `tokens: Vec<u32>` of
  length K; no other change needed.
- **v4 item 7 (fused dequant+matmul+SiLU)**: a Metal kernel optimization
  in dismantle's MoE path; orthogonal to head architecture.
- **v4 item 12 (AMX dispatch for draft head)**: dismantle's `Eagle4Head`
  could route GEMV calls to AMX once that backend exists.

These don't change the integration contract — they're optimizations on
either side of it. The contract above (4-hidden capture + npz checkpoint
+ propose() API) is stable.

## Decision points outstanding

1. **Cancel the running eagle3 capture?** (PID 40658 was paused; resumable
   tonight per the recovery doc.) Eagle4 has trained a 75.84%-accept eagle3
   baseline on its own data. Our capture finishing wouldn't add new info.
   **Recommend cancel; reclaim ~4 days of overnight compute for something
   else.** User decides.

2. **Retire `tools/training/mlx_eagle/`?** Same logic — eagle4 owns
   training. Could leave for a session-end cleanup commit.

3. **NPZ loader: in-tree or new crate?** Lean in-tree (one file, ~200 lines
   of npy parsing). NPZ is just ZIP of NPY; no real dep needed.

## Order of work for the next session

1. Generalize `DraftHead` trait + scaffold `Eagle4Head` with Unimplemented
   bodies. (prep, this commit — done before user wakes up)
2. NPZ loader + a parity test that loads `best.npz` from eagle4 repo and
   compares one forward pass against eagle4's Python output. (~half day)
3. Engine trait method + DeepSeekV2 impl for 5-input capture. (~half day)
4. End-to-end smoke: `dismantle generate --speculate eagle4 --draft-head
   .../best.npz` produces output bit-identical to greedy when K=1, and
   non-trivially accelerated when K=4 + bench. (~half day)
5. Path B + masked-verify kernel design pass. (separate multi-day work)
