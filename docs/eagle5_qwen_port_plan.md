# Eagle5 / Eagle6 spec-decode port to qwen_dense.rs

**Status:** the trained Eagle6 heads from
`colab/finish_q3b_reconciliation.ipynb` are silently no-ops on
Qwen-3B / Qwen-1.5B today. The `--speculate eagle5` CLI flag and
`--eagle5-head <path>` arg parse correctly, populate
`EngineConfig::eagle5_head_path`, but Qwen's dense forward loop never
calls the Eagle head.

Discovered 2026-05-26 by the end-to-end bench in that session:
`--speculate eagle5` at K=2/4/8 produced **identical dec_tps to
baseline** with `draft_accepted=0 draft_rejected=0` — the head was
inventory in RAM, not running.

## Phase decomposition

The port is **three** phases, not one. The original plan elided Phase A.

### Phase A — Trained-head loader + Rust forward pass (2–4 days)
The runtime-side ML implementation of the Eagle6 head.

- **A.1 — Safetensors loader.** ✅ **LANDED 2026-05-27** in commit
  TBD. `Eagle5Head::load_from_safetensors` now reads the actual
  Colab-trained safetensors files and populates the `Trained`
  variant with all 14 runtime tensors (excludes the training-only
  `calib_proj.{weight,bias}`). Verified end-to-end against the real
  q3b head: `cargo test --release --test eagle5_trained_head_load
  trained_head_q3b_loads -- --ignored` PASS in 4.25s.
  Files touched:
  - `crates/dismantle-core/src/speculate/safetensors_io.rs` (new) —
    minimal safetensors reader, no new crate deps.
  - `crates/dismantle-core/src/speculate/eagle5.rs` — `Inner::Trained`
    variant expanded with full struct (in_proj + blocks + frozen
    refs); `load_from_safetensors` implemented.
  - `crates/dismantle-core/tests/eagle5_trained_head_load.rs` (new) —
    integration test against real heads (`#[ignore]` default).

- **A.2 — Eagle6 forward pass in Rust.** ❌ **NOT IMPLEMENTED.** The
  `propose()` dispatcher for `Inner::Trained` currently runs a
  simplified linear projection (lm_head @ token_embd[prev]) as a
  placeholder. Real forward is needed to get useful accept rates.
  Steps:
  1. RMSNorm helper (already exists in `kernels/` for verifier; reuse).
  2. `in_proj` matmul: concat `[prev_token_embd | residual_in |
     intermediate]` (3 × hidden) → hidden.
  3. Transformer block(s):
     - pre-attn RMSNorm with `attn_norm`
     - multi-head self-attention: Q/K/V proj → split heads
       (n_heads × head_dim), scaled dot-product, out_proj.
       Single-token forward; no KV cache needed for the head itself.
     - residual add
     - pre-mlp RMSNorm with `mlp_norm`
     - gated SiLU MLP: `down @ (silu(gate @ x) * (up @ x))`
     - residual add
     - Optionally chain `extra_blocks.{0..N-2}`
  4. Final RMSNorm with `_output_norm`
  5. LM head: `_lm_head @ hidden` → vocab logits, argmax → draft id.
  6. Auto-regressive: feed draft id back as `prev_token` for the
     next of K drafts.

  **Critical input dependency:** the head needs the verifier's
  residual stream AND intermediate stream **from the capture layer**
  (layer 32 for Qwen-3B, 22 for Qwen-1.5B). This means Phase B must
  expose those tensors from the verifier's forward path; currently
  qwen_dense.rs discards them after each layer's compute. See
  Phase B integration point §4.

  Reference impl: `colab/eagle5_train_pytorch.py` `Eagle5Head`
  module. Port it line-by-line; correctness comes from matching
  the PyTorch forward bit-for-bit on the same input.

  Effort: 1–2 days for an experienced Rust+ML operator. Validation
  gate: run the PyTorch head on a fixed input vector and compare
  Rust output element-wise with `atol=1e-3` (fp32 path), `atol=1e-2`
  (f16 path if you quantize the frozen refs).

- **A.3 — (Optional) Quantize the head.** Trainable params are ~96M
  for q3b (b1_wide, 1 block) and ~150M for q1p5 (b2_wide, 2 blocks).
  At fp32 that's 384 MB / 600 MB of head weights. Q4_K of the
  projections would cut to ~48 MB / ~75 MB. Not needed for v1
  correctness; ships as Phase A.3+ if the head forward turns out
  to be wall-clock-significant. Initial measurement first.

### Phase B — Dispatch wire-up in qwen_dense.rs (2–4 days)
The original "port" plan. Per-section file:line refs from the
2026-05-27 audit of `crates/dismantle-core/src/model/deepseek_v2.rs`:

1. **Eagle5 head load at construct time** — `deepseek_v2.rs:715-752`.
   Pattern:
   ```rust
   let eagle5_head: Option<Eagle5Head> = if config.speculate_mode == SpeculateMode::Eagle5 {
       match config.eagle5_head_path.as_deref() {
           Some(p) => Some(Eagle5Head::load_from_safetensors(p, hidden, vocab)?),
           None    => Some(Eagle5Head::mock(0xea91e5_u64, hidden, vocab)),
       }
   } else { None };
   ```
   Add field `eagle5_head: Option<Eagle5Head>` to `QwenDense` struct
   (`qwen_dense.rs:180`). Initialize in `QwenDense::load` / `::new`.

2. **Pre-flight gate** — `deepseek_v2.rs:1049-1068`. At top of
   `forward_token_greedy_tcb` (or wherever the qwen generate loop
   starts):
   ```rust
   if self.speculate_mode == SpeculateMode::Eagle5 {
       if sampling.temperature != 0.0 {
           return Err(Error::Model("eagle5 spec-decode requires temperature=0".into()));
       }
       if sampling.repetition_penalty != 1.0 {
           return Err(Error::Model("eagle5 spec-decode requires repetition_penalty=1.0".into()));
       }
       if self.eagle5_head.is_none() {
           return Err(Error::Model("eagle5 requested but no head loaded".into()));
       }
   }
   ```

3. **Verify-then-draft loop** — `deepseek_v2.rs:1437-1599`. The
   substantive 150-line block. Steps the qwen port must replicate:
   - After computing logits for the verifier's current position,
     branch on `speculate_mode == Eagle5`.
   - Call `self.eagle5_head.as_mut().unwrap().propose(last_id, K)`
     to get K draft ids (where K = `config.verify_window`, default 4).
   - Run target model on `[last_id, draft_0, draft_1, ..., draft_{K-1}]`
     in **one batched forward pass** via `forward_tokens_batched`
     (deepseek_v2.rs:2444-2479). qwen_dense.rs already has the
     primitives but no `forward_tokens_batched` method; you'll need
     to write the qwen-flavored variant (or refactor an existing
     batched-prefill path into something reusable).
   - Greedy argmax of each batched output position to compare against
     drafts (`deepseek_v2.rs:1549-1561`):
     ```rust
     let mut first_reject = 0;
     let mut correction = None;
     for k in 0..K {
         let pred = argmax_f32(&logits_batch[k]);
         if pred != draft_ids[k] {
             correction = Some(pred);
             break;
         }
         first_reject += 1;
     }
     let bonus_id = correction.unwrap_or_else(|| argmax_f32(&logits_batch[K]));
     ```
   - KV cache rewind: `self.kv.seq_len = draft_start + first_reject + 1`
     (deepseek_v2.rs:1534, 1564). Position-indexed cache makes this
     a single pointer reset; no per-position slicing needed.
   - Increment counters: `stats.draft_accepted += first_reject;
     stats.draft_rejected += K - first_reject;` (deepseek_v2.rs:1565-1566).

4. **Expose capture-layer hidden states for the head.** Phase A.2's
   real forward needs `residual_in` and `intermediate` from the
   verifier's capture layer (32 for Qwen-3B, 22 for Qwen-1.5B). In
   qwen_dense.rs the per-layer compute currently writes back to the
   single residual stream and discards intermediate. Plumbing:
   - Add an optional `capture_buf: Option<MetalBufferF16>` field to
     QwenDense that's sized `[hidden]` for residual + `[hidden]` for
     intermediate.
   - In the layer loop, after the chosen capture layer's output, copy
     residual + intermediate to capture_buf.
   - In the Eagle5 branch, pass capture_buf to `eagle5_head.propose`.
     Requires extending the `propose` signature to accept these
     tensors — currently it takes only `prev_token: u32, k: usize`.
     Update both Mock (ignore the tensors) and Trained (use them in
     the in_proj input).

5. **Counter aggregation** — `EngineConfig::draft_accepted` /
   `draft_rejected` already exist on the engine struct
   (`engine.rs:189-190`). Increment from inside the qwen forward loop,
   same pattern as DeepSeek.

### Phase C — Profile + tune + validate (1–2 days)
- End-to-end paired bench: `tools/bench/eagle5_paired_bench.sh` with
  trained head, K=4. Gate: `dec_tps(K=4) > dec_tps(K=0)` by ≥10%.
- Accept rate sanity: `draft_accepted / (draft_accepted + draft_rejected)`
  should fall in the 30–60% range (Colab simulation projected
  accept/verify=6.84 for q3b at K=∞; for K=4 expect ~3.5–4.5 accepted
  per verify, which is ~70% per-step accept rate).
- Profile: where does the head forward sit on the timeline? If it's
  >20% of decode time, schedule Phase A.3 (head quantization).
- Compare against Colab projection: 170 dec_tps was Colab's optimistic
  model. Real M3 Pro likely lands at 40–60 dec_tps with the head
  running, which is still 1.5–2.2× baseline.

## Validation gates (per phase)

| Gate | Phase | Command |
|---|---|---|
| Loader reads real q3b head | A.1 | `DISMANTLE_Q3B_HEAD=$HOME/Downloads/head_final.safetensors cargo test --release --test eagle5_trained_head_load -- --ignored` ✅ green 2026-05-27 |
| Rust head forward matches PyTorch within 1e-3 | A.2 | (test to write — feed a fixed input vector through both, compare logits) |
| Build clean | B | `cargo build --release --workspace` zero new errors |
| Existing lib tests pass | B | `cargo test --workspace --lib` |
| Speculate=off equivalence | B | regression: bit-identical greedy with `--speculate off` vs pre-port |
| Speculate=eagle5 mock-head engages | B | `draft_accepted + draft_rejected > 0` |
| Speculate=eagle5 trained-head accept rate sane | B/C | `draft_accepted / total > 0.3` |
| Paired bench positive delta | C | `tools/bench/eagle5_paired_bench.sh` K=4 ≥ K=0 + 10% |

## Files to touch (Phase B onwards)
- `crates/dismantle-core/src/model/qwen_dense.rs` — add `eagle5_head`
  field, pre-flight gate, verify-then-draft dispatch, capture-layer
  hidden state plumbing.
- `crates/dismantle-core/src/model/mod.rs` — no changes (dispatcher
  already routes by arch).
- `crates/dismantle/src/main.rs` — no changes (flag plumbing exists).
- New test: `crates/dismantle-core/tests/qwen_eagle5_speculate_smoke.rs` —
  end-to-end `--speculate eagle5` on small Qwen prompt, asserts
  `draft_accepted + draft_rejected > 0`.
- New test: `crates/dismantle-core/tests/eagle5_head_forward_parity.rs` —
  Phase A.2 numerical parity vs PyTorch reference.

## What's NOT in this plan
- Training improvements. The trained heads from the reconciliation
  notebook are ready inventory.
- Continuous batching / Track E. Separate effort.
- The qwen_dense AWQ Option B path (already shipped behind env flags).
  Spec-decode work shouldn't touch it.
- Multi-block-via-shared-blocks optimization. Q1p5's 2-block head
  uses `extra_blocks.0.*` namespace; the loader handles this but
  there's no shared-residual-stream optimization across blocks at
  runtime — each block does a full residual loop. That's fine for v1.

## Total effort estimate
- Phase A.1 (loader): done.
- Phase A.2 (head forward): 1–2 days.
- Phase B (dispatch wire-up): 2–4 days.
- Phase C (validate + tune): 1–2 days.
- **Total: 4–8 days from today to first measured Eagle5 lift on Mac.**
