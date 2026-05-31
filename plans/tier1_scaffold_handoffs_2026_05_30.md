# Tier-1 lever scaffold handoffs — fill-in-the-body guides

Each lever below is reduced to "fill in the kernel body + bench" with exact
insertion points, reserved env-gate names, a parity template, and a ready
`paired_lever.sh` command. Insertion points are anchored to function names
(robust to line drift); line numbers are 2026-05-30 approximations.

**Shared facts (verified 2026-05-30):**
- Env-gate idiom — `crates/dismantle-core/src/kernels/mod.rs` ~1146:
  ```rust
  let use_x = { static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
      *E.get_or_init(|| std::env::var_os("DISMANTLE_QWEN_X").map(|v| v != "0").unwrap_or(false)) };
  ```
- Predec scale table builder — `kernels::predecode_q4_k_scale_table(&[u8]) -> Vec<f32>`
  (mod.rs ~1042): 16 f32/block (d*scale, dmin*min per sub-block) from the 144 B Q4_K block.
- Predec cache — `QwenDense::ensure_q4k_predec_cache` (qwen_dense.rs ~2407): `HashMap<usize /*tref.offset*/, PinnedBuffer>`, walks q/k/v/o/gate/up/down per layer. **Does NOT include the LM head.**
- Predec dispatch wrappers (mod.rs): `gemv_q4_k_v4_predec_pinned_tcb` (~1095, single row), `gemv_q4_k_v4_predec_pair_pinned_tcb` (~1201, gate+up fused).
- Parity template — `crates/dismantle-core/tests/q4k_predec_parity.rs`: `Lazy<MetalContext>`, `make_q4k_bytes`, `new_f32_buf`/`read_f32_buf`, bit-identical via `to_bits()`. Q3_K numeric template: `tests/v1_1_q3_k_parity.rs` (atol 1e-2).
- Bench harness — `tools/bench/paired_lever.sh --label L --env-a "G=0" --env-b "G=1"`.
- The §1 gate is the methodology check (`analyze_tcb_trace.py`); 0.1 (`mst_export.sh`/`mst_analyze.py`) is the un-distorted profiler that should **rank these** before writing XL bodies.

---

## 1.5 — LM head → predec  *(no new kernel; reuse existing predec; bit-identical)*

**Gate:** `DISMANTLE_QWEN_LMHEAD_PREDEC` (default off).
**Insertion:** LM-head dispatch in `forward_token_greedy_tcb` (qwen_dense.rs ~3867)
currently calls `gemv_q4_k_m_v3_8r_pinned_tcb(&mut tcb, lhq, 0, vocab*row_bytes, vocab, h, &arena.x_norm_buf, &arena.logits_buf)` inside the `DISMANTLE_QWEN_Q4K_LMHEAD` path (~3823–3877, guarded by `h % 256 == 0`).
**Steps:**
1. Build a predec scale table for the LM-head Q4_K weight. **Open item:** locate where the Q4K LM-head buffer (`lhq` / `lm_head_q4k_buf`) is filled from Q4_K bytes (grep `lm_head_q4k`); call `predecode_q4_k_scale_table` on those same bytes once, store as a new `Option<PinnedBuffer>` field (e.g. `lm_head_predec_scales`) next to the buffer.
2. At ~3867, behind the new gate, call `gemv_q4_k_v4_predec_pinned_tcb(&mut tcb, lhq, 0, vocab*row_bytes, &scales, 0, vocab, h, &arena.x_norm_buf, &arena.logits_buf)`.
**Parity:** bit-identical by construction (predec == v3_8r math). Run `paired_lever.sh` parity mode.
**Bench:** `tools/bench/paired_lever.sh --label lmhead_predec --env-a "DISMANTLE_QWEN_LMHEAD_PREDEC=0" --env-b "DISMANTLE_QWEN_LMHEAD_PREDEC=1"`. Expected +1–2% (LM head ~4–5% of decode).

---

## 1.1 — Prefill MMA port  *(highest-confidence: proven +10–20% on M>1; speeds TTFT)*

**Gate:** `DISMANTLE_QWEN_PREFILL_MMA` (default off).
**Source:** `silicon-builds/dismantle-q4k-mma/src/bin/bench.rs` lines 132–177 = `gemm_q4k_mma_nwide` (N-wide reuse: dequant W tile once per K-step, reuse across BN2=32 cols; 4 simdgroup acc matrices). Dequant subroutine `dq_w` at 40–56. **Port `gemm_q4k_mma_nwide`** (it amortizes dequant, the bandwidth-bound regime).
**Insertion (kernel):** add the ported kernel to `shaders/quant.metal` next to `gemm_q4_k_m_batched_v3w`.
**Insertion (dispatch):** clone `gemm_q4_k_m_batched_v3w_pinned_tcb` (mod.rs ~595) → `..._mma_pinned_tcb` pointing at the new kernel (mind simdgroup tile dims for grid/threadgroup sizing).
**Insertion (call):** `forward_tokens_batch_tcb` `batched_proj!` macro (qwen_dense.rs ~4120) and the verify path (~4307); behind the gate, route to the MMA wrapper when `batch > 1`.
**Parity:** bit-identical greedy on a long prompt (prefill path); clone `q4k_predec_parity.rs` at a batched shape (e.g. M=8, 2048×2048).
**Bench:** prefill TTFT — `paired_lever.sh --label prefill_mma --env-a "DISMANTLE_QWEN_BATCH_PREFILL=1 DISMANTLE_QWEN_PREFILL_MMA=0" --env-b "DISMANTLE_QWEN_BATCH_PREFILL=1 DISMANTLE_QWEN_PREFILL_MMA=1"` on a long prompt (measure prefill ms, not just decode_tps).

---

## 1.6 — Q3_K fast-GEMV  *(unlocks the byte-cut: −11% bytes, +4.7% PPL from a clean source)*

**Gate:** `DISMANTLE_QWEN_Q3K_PREDEC` (default off).
**State:** only inline-decode `gemm_q3_k_fused_v2` exists (quant.metal ~344; 110 B/block, 256 thr/TG, 8 rows/TG); dispatch `gemv_q3_k_pinned_tcb` (mod.rs ~1975). **No predec.**
**Steps:**
1. `predecode_q3_k_scale_table(&[u8]) -> Vec<f32>` in mod.rs — analogous to Q4_K but for the Q3_K 110 B block layout (6-bit scales, separate high-bit plane). This Rust half is CPU-unit-testable now against `dequant_into(GgmlType::Q3_K, ...)`.
2. `gemm_q3_k_v4_predec` kernel in quant.metal (read pre-decoded scales; same 2r ILP geometry as Q4_K predec).
3. `gemv_q3_k_v4_predec_pinned_tcb` wrapper + extend `ensure_q4k_predec_cache` (or a Q3_K sibling) to build Q3_K scale tables.
**Parity:** clone `v1_1_q3_k_parity.rs`, atol 1e-2 vs CPU Q3_K deq + CPU GEMV.
**Bench:** needs a Q3_K Qwen-3B GGUF; `paired_lever.sh --label q3k_predec --no-parity --env-a "...Q3K_PREDEC=0" --env-b "...Q3K_PREDEC=1"` (compare against Q4_K_M baseline tps).

---

## 1.2 — f16 predec scales  *(−17% predec bytes: 192→160 B/block; atol-trade)*

**Gate:** `DISMANTLE_QWEN_PREDEC_F16SCALES` (default off).
**Steps:**
1. `predecode_q4_k_scale_table_f16(&[u8]) -> Vec<half::f16>` in mod.rs (cast the existing f32 table to f16). CPU-unit-testable now.
2. `gemm_q4_k_v4_predec_f16s` kernel in quant.metal (read `half` scales, widen to float in-register).
3. In `ensure_q4k_predec_cache`, store f16 scales when the gate is on (half the cache RSS too).
**Parity:** atol 1e-3 fp16 + token-identical greedy (NOT bit-identical — f16 scale rounding). Clone `q4k_predec_parity.rs` but assert `abs(a-b) < 1e-3` instead of `to_bits()`.
**Bench:** `paired_lever.sh --label predec_f16s --no-parity --env-a "...PREDEC_F16SCALES=0" --env-b "...PREDEC_F16SCALES=1"` (gate via the Rust atol parity test).

---

## 1.7 — simdgroup-matrix decode  *(XL, the MLX-class headline — gate on 0.1 first)*

The decode GEMV at M=1 underfills the 8×8 simdgroup tile; fill it by processing
multiple output rows per tile (the decode analogue of 1.1's prefill MMA). This is
the path from ~41% → 60–80% of peak. **Do not start before 0.1 profiling** shows
the predec GEMV is FMA/occupancy-limited rather than purely bandwidth-limited —
the Bible says decode is bandwidth-bound, which would cap this lever. Schedule as
its own multi-session push. Reuse 1.1's `dq_w` + simdgroup intrinsics; the M=1
fill is the novel part.
