# Findings Summary — validated conclusions (autonomous campaign, 2026-06-21)

## TL;DR
- **Biggest win — the long-context moat is the SSM path.** RWKV-7 0.4B decode is FLAT (119 tps @8k) while Qwen-3B
  collapses to 8.6 (~**14×**). No KV wall. Strategic: long-context product → SSM; the KD/draft training feeds small instruct SSMs.
- **predec is the engine's verified headline decode lever** (−46.7% when toggled OFF = ~2×, bit-identical, already default-on).
- **The decode speed config ceiling is small + at the bench noise floor** (`--profile fast` ~+3–7%, noisy; per-component
  attribution unresolvable). Decode KERNELS are genuinely tapped (bible §3.0 A5/A6/A7/A10 + my Q6_K 1r/2r/4r A/B + parity 3/3).
- **Compression:** `F16_KV` −50% KV (clean, long-ctx). Per-channel int4-KV −75% (numerics PARITY-VALIDATED ✅) is built but
  unwired → **deferred** (the SSM moat is the better long-ctx answer, lowering its urgency).
- **Spec-decode = NO-GO for speed** (per-cycle overhead wall); lossless router committed (`73fc5b4`), default-OFF.
- **Infra built:** `tools/bench/ratios.sh` (warm-median tps/qual harness + interleaved `abi`), `docs/env_flags.md`,
  `docs/architecture.md`, the 5 `docs/campaign/` artifacts. Post-crash tree verified clean (cargo check ✅, parity 3/3 ✅).

**Exact next commands (attended):**
```
# 1) Re-measure --profile fast cleanly (interleaved, idle machine — beats the noise floor):
tools/bench/ratios.sh abi "" short 10
# 2) Wire per-channel int4-KV (roadmap #1; steps in docs/dead_levers.md:401), then gate:
#    cargo test -p hawking-core --test mha_decode_perchannel_int4kv_parity   # already PASSES
#    then real-model perplexity + long-ctx tps @8-16k before default-on.
# 3) Productionize the RWKV-7 (SSM) serving path for the long-context product.
```

Model under test: `models/qwen2.5-3b-instruct-q4_k_m.gguf` (Qwen2.5-3B-Instruct-Q4_K_M), release binary
`./target/release/hawking generate`, M-series. All tps = warm 5-trial median unless noted. `HAWKING_QWEN_USER_DRAFT=0`.

## Baseline (measured)
- **Short-ctx warm decode ≈ 40 tps** (default). Cold single-run ≈ 30 tps — that gap is **PSO shader-compile**, not decode.
- **Long-ctx KV wall (severe):** 40 (short) → **18.8** (~2.5k) → **8.6** (~8k) tps = **4.6× drop**. Transformer KV-read scaling.
- Footprint: 1.80 GiB weights (~4.8 bpw) + 0.28 GiB KV @4096 ctx.
- Kernel efficiency (from bible): hawking Q4_K GEMV ≈ **56% of peak BW**; llama.cpp ≈ 60%; MLX ≈ 70-80%.

## Validated wins (shippable)
- **predec (`HAWKING_QWEN_Q4K_PREDEC`, default-ON) is the headline decode win** — toggling it OFF = **−46.7%** (40.4→21.6
  tps, ~2×), bit-identical. Already on; the engine's core optimization (verified warm this run).
- **`--profile fast` = a SMALL warm gain (~+3–7%, noisy)**, 83–90% argmax-identity (mild quant trade). The fast config
  stably hits ~41.2 vs default's noisy 38–40. Speed-priority config; keep the bit-identical default for quality. (Earlier
  "+7.5% clean" was the high end of the noise; per-component attribution is BELOW the bench noise floor — see `test_matrix.md`.)
- **`F16_KV` = −50% KV footprint**, ~0% short-ctx, **+1.9% @2.5k (scales ~15% @16k)**, 88% argmax-identity. Clean
  long-ctx + footprint lever.

## Validated structural facts (verified against code)
- **Q6_K ffn_down row-blocking is already optimal at 2r** (default). Warm A/B: 2r=40.48 > 4r=40.04 > 1r=39.91.
  (Red-team claimed the 2r was "unreachable" — FALSE; it missed the `use_2r` override at `kernels/mod.rs:4258`.)
- **Per-channel int4-KV is BUILT + quality-validated but DEAD-CALLED.** Kernels `kv_int4_calib_max_tcb`,
  `kv_quant_int4_append_pc_tcb`, `mha_decode_flash_int4kv_pc_tcb` (`kernels/mod.rs:9143-9261`) exist + are registered
  but are NOT called in `qwen_dense.rs`. cosine ~0.998 on real K/V (`dead_levers.md:401`). −75% KV (3.5× @32K). → WIRING.
- **Spec-decode (EH free-market + trained EAGLE) is net-negative for speed** on this engine (per-cycle overhead wall:
  87% accept → still 0.91×). The lossless cost-aware router is committed (`73fc5b4`) but default-OFF.

## Architecture — the long-context moat (VALIDATED, decisive)
- **RWKV-7 0.4B decode is FLAT across context: 118.6 (short) → 110.6 (2.5k) → 119.4 (8k) tps** — constant ~6 MiB
  recurrent state, no KV cache. Qwen-3B over the same range: 40 → 18.8 → 8.6 (**−78%**). **At 8k, RWKV-7 is ~14× faster.**
  Flatness is the SSM property (no per-token KV-read growth) — the genuine long-context differentiator vs the transformer
  KV wall. **Strategic:** the long-context product should run the SSM path; the KD/draft training work feeds small instruct SSMs.
- **Decode-kernel speed micro-opt is genuinely tapped** (bible §3.0, re-confirmed by my Q6_K 1r/2r/4r A/B): vectorized
  unpack (A5), occupancy (A6), **simdgroup-matrix decode (A7 — M=1 underfills the MMA tile 7/8)**, access-order layout
  (A10, −16.8%) all Type-1 dead for batch=1 decode; the predec GEMV is at the Apple-GPU memory-model optimum. The ~1.6×
  gap is structural for decode. The bible's only remaining axes are **fewer bytes** (Q3_K/QTIP — both decode-slower) and
  **stateful** (the SSM moat above). simdgroup-matrix / MMA remains a LIVE lever for PREFILL (M>1), not decode.

## Method lessons (carry forward)
1. **Warm-median (≥5 trials) only** — single cold runs measure PSO-compile, not steady-state.
2. **Validate over a distribution** — one argmax-identical prompt is not evidence (the `--profile fast` 1-prompt "identical"
   was 83% over 12).
3. **Short- and long-ctx are different regimes** — KV-bandwidth levers are ~0% short, grow with depth.
4. **Synthetic uniform-random parity does NOT catch outlier-driven quant collapse** — gate KV-quant on REAL captured K/V.

See `roadmap.md` (ranked next work), `kill_ledger.md` (rejected), `test_matrix.md` (all runs),
`docs/plans/ratios_roadmap_2026_06_21.md` (the full hardened roadmap).
