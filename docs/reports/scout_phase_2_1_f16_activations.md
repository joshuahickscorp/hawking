# Phase 2.1 spec — f16 activations + f16/Q8 KV (Qwen dense) (scout, 2026-06-01)

> READ-ONLY scope. file:line verified against `paradigm/exec`. **Charge premise is partly wrong — corrected below.**

## ⚠️ PREMISE CORRECTION (load-bearing — verify before any work)
- **There is NO `--q8-kv` CLI flag and NO Q8/f16 KV for the Qwen *dense* path.** The only Q8-KV in tree is **MLA/DeepSeek-V2** (`mla_decode_q8kv_metal` `kernels/mod.rs:3573`, `kv_append_q8_0_f32` `:3692`, test `q8_kv_parity.rs` — all MLA shapes: kv_lora=512). `[[q8-kv-runtime-landed]]` is the **MLA** latent-KV, not Qwen. The Qwen dense decode is **100% f32**: arena buffers all `f32_bytes` (`dense_decode_arena.rs:110-137`), KV appended f32 via `memcpy_f32_off_tcb` (`qwen_dense.rs:4133/4141`), `mha_decode_f32` reads `device const float* k_cache/v_cache` (`mha.metal:37-38`). So Q8-KV has **NOT** banked anything on the dense target; the "distinguish new win from Q8-KV-already-banked" question resolves to: *nothing is banked here yet.*
- **silicon #15 (int4-kv, per-channel, cosine 0.998) is HELD/quality-gated**, NOT shipped, NOT wired for Qwen (`dead_levers.md` "Not kills"). There is no `--int4-kv`.

## Residual vs transient map (the f16-safety boundary)
The decode dataflow cleanly separates the **accumulating residual** from **transient activations**:
- **Residual stream = `x_buf`** (`dense_decode_arena.rs:116`). Accumulated in `add_rmsnorm_fused` (`common.metal:144`): `x[i] = x[i] + attn_out[i]` (Phase-1), 27 layers deep. **This is exactly the dead lever's failure path.**
- **Transient (consumed once, regenerated each layer):** `x_norm_buf` (GEMV x-input for q/k/v/gate/up), `attn_out_buf` (o_proj x-input), `ffn_act_buf` (ffn_down x-input), `q/k/v_buf`, `ffn_gate/up_buf`, `o_proj_out_buf`. These never accumulate across layers.

## KILL CROSS-CHECK (mandatory)
- **🪦 f16 residual stream (Phase Z-1) — Type-1 kill.** Quote: *"residual |x|≈5-10; f16 epsilon≈1e-3; error after 27 layers ≈0.27/element → corrupts logits."* Resurrection check names **bf16** (8-bit exponent matches f32 range), gated on a per-kernel BW-saving proof first. ⇒ **`x_buf` MUST stay f32 (or bf16). Storing the residual as f16 is DEAD — do not scope it.** My scoped forms below (a) keep `x_buf` f32 and only touch transient buffers, OR (b) feed f16 *into* the GEMV without changing the accumulator — **distinct from the kill.**
- **🪦 Q8-KV layer-differential precision — Type-1** (uniform routing). Distinct: I scope *uniform* f16/Q8 KV, not per-layer. Not adjacent.
- **🪦 Mixed-precision W4A8 default — HELD** (1.115×<1.20 gate; 20% bit-identical). **Highly adjacent:** the W4A8 path *already* quantizes the three transient activations to int8 (`quantize_f32_to_int8_per_block_tcb` `:1917`, `..._scaled_tcb` `:1972`; arena `x_norm_int8`/`attn_out_int8`/`ffn_act_int8` `:33-38`). f16-x is a *milder* form of the same idea (16-bit not 8-bit). W4A8's kill was **quality + sub-additive-with-predec**, not feasibility — f16-x is far less lossy, so the quality kill does not transfer, but its **perf ceiling warning does** (see caution). **Resurrection check for W4A8 requires logit-cosine, not bit-identity** — adopt that gate here.

## The predec-GEMV x-input precision (the charge's core question — answered)
Verified the dominant kernels read x as `device const float* x` and each thread loads **`xl[8]` floats/block**: `_pair` (`quant.metal:2226,2257-2258`), `_2r` (`quant.metal:2401-2402`). **x is read by all 32 lanes; weights are read 1 byte/lane/`pi`.** Per block-row (FFN 11008×2048, 8 blocks/row): weights = 8×144 = **1152 B**, scales(f16s) = 8×32 = 256 B, **x = 8×256×4 = 8192 B but it is the SAME 2048-float x reread by every one of ceil(11008/8)=1376 simdgroups.** x is small per-row but **re-streamed per output tile** → in aggregate x traffic ≈ rows/8 × cols × 4 B. Halving x to f16 cuts that term ~2×. **But the 0.2 ledger already measured these GEMVs at ~52% peak and the kill "Decode-kernel micro-opt A5/A6 — Type-1" concluded the stall is "scale-read / x-traffic + layout," addressed by f16-scales, NOT load width.** f16-x attacks the *x-traffic* sub-term the A4 profile named — **this is the one genuinely-live, in-kernel BW lever in 2.1**, but its ceiling is bounded by x's share of the 192→160 B/block budget (x is ~the third stream behind weights+scales).

## Ordered, measurement-gated attempts (cheapest/safest first)

**2.1-a — f16 KV cache + f16 `mha_decode` (the real long-context win).** [highest value, lowest residual-risk]
- Touch: add f16 `k_cache_buf`/`v_cache_buf` variants (`dense_decode_arena.rs:113-114`); a `memcpy_f32_to_f16_off` append (replacing `memcpy_f32_off_tcb` `:4133/4141`); a `mha_decode_f16kv` kernel (clone `mha.metal:34`, change buffers 2/3 to `device const half*`, dequant in the dot loops `:60-62,108-110`). Q is tiny — keep `q_buf` f32. Gate behind `DISMANTLE_QWEN_F16_KV=1` (no flag exists — add one; do NOT claim `--q8-kv`).
- **Why first:** halves KV traffic + KV footprint (1.2 GB→0.6 GB @ long ctx), and `mha_decode` is **2.92% short-ctx** so any per-block f16 dequant cost is negligible there while the BW win compounds at length.

**2.1-b — f16 x into the predec GEMV (in-kernel BW lever).** [bit-identity impossible; quality-gated]
- Touch: f16 `x_norm_buf` (+ `attn_out_buf`, `ffn_act_buf`); add `half` x-load variants of `_pair`/`_2r` (`quant.metal:2257/2401`: `xl[k]=(float)xh[...]`); the producer is `add_rmsnorm_fused` Phase-2 write `x_norm[i]=...` (`common.metal:171`) — emit f16 there. **`x_buf` (the `x[i]=x[i]+attn_out[i]` accumulator, Phase-1 `:158`) stays f32.**
- Gate behind `DISMANTLE_QWEN_F16_ACT=1`. Compose with f16-scales (disjoint streams).

**2.1-c — Q8 KV (only if 2.1-a's f16 quality fails at long ctx, or for deeper footprint).** Reuse the MLA Q8 machinery pattern (`kv_append_q8_0_f32` `:3692`) adapted to dense GQA shapes. More lossy than f16; the silicon #15 verdict says **per-channel int4** is the better scheme if you go sub-f16 — but that's HELD, needs its own PPL gate. Keep Q8 as a fallback, not the lead.

## EXACT gates (per plan §3 / CLAUDE.md)
- **Kernel parity:** `atol=1e-3` fp16 vs the f32 CPU/kernel reference (NEVER loosen). f16-KV dequant adds nothing past 1e-3 (matches the MLA Q8 test's `ATOL=5e-3` *being looser*; f16 is tighter than Q8). Add `mha_decode_f16kv_parity` (clone `tests/q8_kv_parity.rs` structure) + a `predec_pair_f16x_atol1e-3` test (clone `tests/q4k_predec_parity.rs`).
- **Quality (the W4A8-resurrection gate, since not bit-identical):** **logit-cosine ≥ ~0.999** AND PPL within ~+0.05 of Q4_K baseline on a code corpus (same rig W4A8/f16-scales use; confirm thresholds or get user sign-off — plan 1.2 pattern).
- **Token parity:** first-3 greedy IDs must still match the locked baseline (won't be bit-identical end-to-end → this *will* drift if f16 changes argmax; treat a drift as the quality gate, not a hard fail, per W4A8 precedent).
- **MANDATORY long-context parity case** (plan 2.1 explicit): run parity + quality at ≥2-4K ctx, not just 64-tok — this is where KV f16 both pays AND where error could compound.

## SHIP gate (paired delta)
- Paired A/B (`paired_lever.sh`), report **range + 95% CI + IQR**; CI must exclude 0 in the right direction AND clear the **~3% second-position bias** the 0.1 noise-floor found (`paradigm_execution_log.md:104`). Energy: paired **ΔJ/tok** (proxy ok; GEMVs/KV ≈ GPU-time share).
- **Short-context ceiling is honest-tiny:** 0.2 says activation/attention traffic is small short-ctx (attention 2.92%; trivial ops incl. the `memcpy` KV-append 0.83% are ~9% total). **Do not expect a short-ctx tps headline.** Ship gate on the **long-ctx paired tps + the energy ΔJ/tok + the footprint cut**, not the 64-tok number.

## BLUNT STRATEGIC CAUTION (orchestrator)
- **This is primarily a long-context + energy + footprint win, NOT a short-ctx tps win.** Realistic short-ctx tps ceiling ≈ low single-digit % (KV is 2.92% + memcpy 0.83%; f16-x trims a sub-term of the 86.7% GEMV already at 52% peak). The 1.55–1.63× llama gap is **not** closed here — that's the GEMV-efficiency track (scout_phase_2_1_gemv) + dispatch fusion (2.2). Per the 0.2 reorder, f16-KV/act is **step 3 of 4**, correctly behind GEMV-eff and fusion.
- **Biggest risk:** silent residual contamination. The dead Z-1 lever proves a single f16 accumulator corrupts logits after 27 layers. The boundary is **one line** — `common.metal:158` (`x[i]=x[i]+attn_out[i]`) must stay f32. Any refactor that makes `x_buf` f16 (e.g. "just make the whole arena f16") **re-triggers a recorded Type-1 kill** → forbidden without bf16 + a fresh per-kernel BW proof.
- **Second risk:** f16-x (2.1-b) is the W4A8 family in milder form; W4A8 was sub-additive with predec (`predec+w4a8 1.151× < predec 1.340×`). f16-x may likewise give back most of its BW win because the GEMV is already BW-balanced post-f16-scales. **Spike 2.1-a (f16-KV) first — clean long-ctx win, no GEMV interaction; defer 2.1-b until a microbench shows x-traffic is still a measurable GEMV sub-term after f16-scales.**
- **Cheap offline oracle if a form looks dead:** the existing logit-cosine/PPL rig (W4A8/f16-scales gate) on a long-ctx capture settles 2.1-b/2.1-c quality before any kernel ships; the `q4k_predec_f16s_bench.rs` GB/s readout settles whether f16-x moves the GEMV bus at all.

### Critical Files for Implementation
- /Users/scammermike/Downloads/dismantle/crates/dismantle-core/src/model/qwen_dense.rs (KV append :4133/:4141; `mha_decode_f32_tcb` :4153; `add_rmsnorm_fused` residual :4238; GEMV x-inputs `x_norm_buf` :3997/:4290, `attn_out_buf` :4195, `ffn_act_buf` :4424)
- /Users/scammermike/Downloads/dismantle/crates/dismantle-core/src/metal/dense_decode_arena.rs (buffer dtypes :110-137 — the f16/f32 boundary; `x_buf` :116 stays f32)
- /Users/scammermike/Downloads/dismantle/crates/dismantle-core/shaders/mha.metal (`mha_decode_f32` :34, k/v buffers :37-38, dot loops :60-62/:108-110 — clone for f16-KV)
- /Users/scammermike/Downloads/dismantle/crates/dismantle-core/shaders/quant.metal (`_pair` x-load :2257; `_2r` x-load :2401 — clone for f16-x) and /Users/scammermike/Downloads/dismantle/crates/dismantle-core/shaders/common.metal (`add_rmsnorm_fused` :144 — residual stays f32, x_norm write :171 emits f16)
- /Users/scammermike/Downloads/dismantle/crates/dismantle-core/tests/q8_kv_parity.rs + /Users/scammermike/Downloads/dismantle/crates/dismantle-core/tests/q4k_predec_parity.rs (clone for the f16-KV and f16-x atol=1e-3 parity gates)
