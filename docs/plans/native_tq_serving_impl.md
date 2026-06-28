# Native `.tq` serving — implementation plan (the winning tps bench)

> The one unbuilt piece of "condense → run". Today `hawking generate` cannot serve a
> `.tq`; the decoder is test-only. Wiring it unlocks the RAM-cliff tps bench (at 32B,
> Hawking 2-bit fits 9.6 GB on a 19 GB Mac; llama Q4_K 20 GB swaps). Do this RECHARGED
> (needs GPU to test). Two stages: (A) cheap correctness via f16 fallback, (B) the
> native bitslice kernel for the actual speed win.

## What already exists (no need to build)
- **GPU bitslice decode kernel**: `crates/hawking-core/shaders/strand_bitslice.metal`
  (`*_decode` / `*_decode_computed`), gate-verified byte-identical to the CPU decoder.
- **CPU decoder**: `strand_quant::decode::decode_tensor_fixed(enc, cfg) -> Vec<i32>`
  (then `q12_to_f32`). The `.tq` archive reader is in `strand_quant` (tq_bake writes it,
  reads it back for its round-trip check — reuse that read path).
- **`WeightKind::F16`** in `crates/hawking-core/src/backend/mod.rs:113` — documented
  "dequantized-once fallback"; served by `kernels::gemv_f16_metal_buf_tcb` (the `n`/F16
  kernel in `backend/metal.rs`). And `MegakernelLayerWeightsF16` (qwen_dense.rs:551)
  shows f16 layer weights are already a supported shape.

## The seam
`QwenDense::load(weights, config)` at `crates/hawking-core/src/model/qwen_dense.rs:760`:
- linears (`attn_q/k/v/output`, `ffn_*`) come from `tensor_ref(&gguf, name)` → Q4_K mmap
  bytes, dispatched by the GEMV ladder on the GgmlType-derived `WeightKind`.
- embeddings/norms/lm_head come from `dequant_f16/_f32`.

## Stage A — f16 dequant-on-load (correctness first, ~½ day, low risk)
Goal: serve a condensed model (quality-correct) even if tps is f16-slow. Lets the 3-way
quality+tps bench RUN in Hawking.
1. Add `HAWKING_QWEN_TQ=<path.tq>` env (additive, default-off — keeps gates green).
2. In `load()`, if set: open the `.tq` (strand_quant reader), build a `name -> Vec<i32>`
   map, `q12_to_f32` → f16.
3. For each linear, if the `.tq` has that tensor: construct a `WeightKind::F16` weight
   from the f16 buffer instead of `tensor_ref` (the F16 layer-weight path already exists).
   Else fall back to the gguf tensor (so a partial `.tq` works).
4. Gate: parity is NOT expected (condensed ≠ Q4_K); instead assert it loads + generates
   coherent text, and that f16-served `.tq` ppl ≈ the transformers `ppl_bench` number.
5. Bench: `compare_sota.sh` can now point Hawking at the `.tq` → real quality + tps
   (tps will LOSE on a fitting 7B — f16 is 2× Q4_K bytes — that's expected; Stage B fixes it).

## Stage B — native bitslice GEMV (the actual speed win, ~2–4 days, higher risk)
Goal: serve `.tq` at its true low-bit footprint → the RAM-cliff tps win.
1. Add `WeightKind::Tq { cfg }` + carry the encoded `.tq` tensor bytes (not decoded).
2. New GEMV path: dispatch `strand_bitslice.metal` decode → into a fused (or staged)
   GEMV. Start STAGED (decode to a scratch f16 tile, then existing f16 GEMV) to de-risk;
   fuse later if decode-bandwidth-bound.
3. Deploy invariant: `in_features % 256 == 0` (block_len). Refuse/fall-back ragged
   tensors (tq_bake already flags these as non-strict).
4. Gate: bitslice-GEMV output byte-identical to CPU `decode_tensor_fixed` → matmul
   (the gate-tropical/identity tests already assert the kernel decode; extend to GEMV).
5. Bench the cliff: condense a 32B (owner-gated download) → TQ2 (~9.6 GB) → serve on the
   19 GB Mac while llama Q4_K (20 GB) swaps. This is the headline.

## Risk controls
- Everything additive + env-gated → default build/behavior unchanged → regression gates
  stay green. Never edit the Q4_K hot path.
- Re-run `tools/ci/regression_gate.sh` + the RWKV/qwen parity gates before/after.
- Worktree-isolate if doing Stage B in parallel with other work.
