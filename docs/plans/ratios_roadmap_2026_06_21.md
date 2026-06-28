# Ratios Roadmap — speed · compression · density (validated 2026-06-21)

Deliverable of the throughput-pivot campaign. Every lever below was measured **warm** on this
binary (Qwen2.5-3B-Q4_K_M, M-series), or marked DESIGN/FRONTIER where a measurement wasn't safe
to run unattended. The headline methodological fix: **single cold runs measure PSO shader-compile,
not steady-state decode** — all numbers here are warm-median (≥5 trials) unless noted.

## Executive summary
The engine is **near its realistic config ceiling.** The accessible levers are tapped or dead;
the only large remaining speed lever is the hard ~1.6× llama.cpp gap (MST-diff profiling).

- ✅ **One shippable speed win:** `--profile fast` = **+7.5% warm, 83% argmax-identity (mild quant trade)** → recommend as the *speed-priority* config (keep the bit-identical default for quality-priority).
- ✅ **One clean compression win:** `F16_KV` = **−50% KV footprint** + long-ctx speed (scales with context), mild quality.
- ❌ **Validated dead ends:** FFN_DOWN_Q4K (cold-artifact), int4-KV (slower + quality-collapse), Q6_K predec (likely-null).
- 🧭 **Real frontier:** the ~1.6× llama.cpp gap. **NOT MST-only** (see red-team below) — it is diffable against MLX's open-source qmv.

## ⚔️ RED-TEAM HARDENING (adversarial 8-agent workflow, 2026-06-21)
An adversarial red-team (4 research + 3 code-audit agents, 6/7 succeeded) attacked every conclusion. **Verdict:
the ENV/CONFIG ceiling (~+7.5%) genuinely holds — but "the engine is near its realistic ceiling" is NOT established.
Three concrete, in-tree, untested levers survive:**

1. **Q6_K ffn_down GEMV is UNCOALESCED** (speed — the largest miss). The default `gemm_q6_k_fused_v2_swiglu` (1r)
   reads `ql` at **stride-8** (16 unique addrs across 32 lanes = half the cache line/iteration), unlike the Q4_K path
   (32 contiguous bytes, `quant.metal:690`). Q6_K ffn_down = **20.4% of per-token decode bandwidth, the single largest
   GEMV**, and was NEVER A/B'd for achieved GB/s. llama.cpp runs Q6_K at 2 rows/simdgroup by design.
   **VERIFIED 2026-06-21 (kill-ledger — I checked the red-team myself): the red-team OVER-CLAIMED.** The 2r variant is
   **already DEFAULT-ON** — it read the wrapper's `const KERNEL` 1r string and missed the `use_2r` override at
   `kernels/mod.rs:4258` (1r/2r/4r all exist + are parity-tested: `tests/q6k_swiglu_{2r,4r}_parity.rs`). So the
   row-blocking axis is *already exploited at 2r*. Two things remained: (a) **FREE A/B RESULT (warm 5-trial median):
   2r(default)=40.48 > 4r=40.04 > 1r=39.91 → 2r is already optimal, no free win**; (b) the stride-8 ql-load is CONFIRMED
   but **shared by all variants** (bit-identical) → fixing needs a Q6_K *repack* (sidecar + new kernel; high-effort — the
   bible's A10 layout attempt hit −16.8%) = **DEPRIORITIZED**. **→ Lever #1 CLOSED:** row-blocking is tapped; the repack is
   low-EV given the FFN_DOWN_Q4K warm-null. The campaign's "kernels near-optimal" verdict was right for the row-blocking axis.
2. **Per-channel int4-KV is BUILT + quality-validated but DEAD-CALLED** (compression). The int4-KV NO-GO was the
   **per-ROW** scheme. A **per-CHANNEL** path (`kv_quant_int4_append_pc_tcb`, `mha_decode_flash_int4kv_pc_tcb`,
   `kv_int4_calib_max_tcb` — `kernels/mod.rs:9143-9215`) is registered + validated (cosine ~0.998 real K/V) but has
   NO flag + NO dispatch → my NO-GO test ran the broken per-row path. **−75% KV** (vs F16's −50%) at cosine 0.998;
   ~70% of the work is in-tree. → wire behind `HAWKING_QWEN_INT4_KV_PC` + a real-model perplexity gate.
3. **The 1.6× gap is DIFFABLE against MLX** (method). MLX is open-source + faster than llama.cpp; its qmv/qmm is a
   concrete reference (hawking ~56% peak, MLX ~70-80%) — cheaper than "hard MST-diff". → diff layout/register-blocking.

**Corrections to the roadmap below:**
- "Engine near ceiling" → only the **ENV-FLAG** ceiling is ~+7.5%; the kernel/wiring levers above live outside env space.
- F16_KV is the **weaker** compression option — per-channel int4-KV (−75%, cosine 0.998) **dominates** it if wired.
- int4-KV "NO-GO" is **per-ROW only** — does NOT close the per-channel scheme.
- Q6_K predec **confirmed null** (int8 scales are trivial to decode); DRAM-cost corrected to +6.7% (f16 table), not +30%.
- Secondary survivors: f16-ACTIVATIONS in GEMV (x never byte-cut; argmax-gate); GQA group-coalesced MHA (long-ctx, 8× KV-read cut);
  column-split GEMV for under-occupied k/v_proj (+17% SpQt, modest e2e); fused-epilogue default-on (small bounded).

**Revised priority: (1) measure → attack Q6_K ffn_down coalescing; (2) wire per-channel int4-KV; (3) MLX-diff the gap.**

## SPEED lane (warm tps; baseline ~40)
| Lever | Warm gain | Quality | Status | Why |
|---|---|---|---|---|
| `--profile fast` | **+7.5%** | 83% identity (mild) | ✅ SHIP (speed-priority) | vocab-prune + Q4K-LM-head + Q4K-FFN-down + predec + f16-scales; ~17% of prompts diverge at token level (quant trade, not collapse) |
| FFN_DOWN_Q4K alone | ~0% | argmax-identical | ❌ no warm gain | the cold "+29%" was PSO-compile, not decode |
| f16-scales alone | ~0% | identical here | — opt-in | no measurable effect on this binary |
| Q6_K predec (ffn_down) | likely-null | bit-identical if built | 📐 DESIGN, deprioritized | Q6_K `int8` scales already cheap to decode (unlike Q4_K 6-bit packed → +34%); ALU saving offset by table DRAM read |
| ~1.6× llama.cpp gap | up to ~1.6× | — | 🧭 FRONTIER (hard) | decode is kernel/bandwidth-bound; MST-diff is the only path |
| continuous-batch B=8 | aggregate only | — | off-goal | helps throughput, not single-stream latency |

## COMPRESSION lane (baseline 1.80 GiB weights ~4.8 bpw + 0.28 GiB KV @4096)
| Lever | Footprint | Speed | Quality | Status |
|---|---|---|---|---|
| `F16_KV` | **−50% KV** | +1.9% @2.5k → ~15% @16k | 88% identity (mild) | ✅ clean win — long-ctx / memory-constrained |
| `int4-KV` | −75% KV | **−5.7% (slower)** | **0% (collapse)** | ❌ NO-GO (dequant overhead + per-row collapse) |
| Trellis FFN `.tq` | ~30% weights | decode-SLOWER | TBD (needs full bake) | max-compression option; runtime-wired (`HAWKING_QWEN_TQ`); existing `.tq` is a 19 MB partial |
| W4A8 / AWQ | weights | — | quality-blocked | held (prior campaigns) |

Note: at long context decode itself falls to ~19 tps (vs ~40 short) — the KV-read wall. That is
exactly where F16_KV's value grows and where the SSM long-ctx moat (RWKV-7) matters.

## DENSITY lane (black-hole)
- 92.4k Rust LoC, 13.1k Metal, 4 crates, 41 deps.
- **Consolidation candidate:** `eagle5*` (~1.6k LoC, trained-EAGLE) — now dead, since spec-decode is
  conclusively net-negative for speed on this engine. CAUTION before removal: the prior dead-code
  audit found kernels are name-string-referenced, and the committed cost-aware router (`73fc5b4`)
  may reference spec infra. Needs a careful reference-audit + parity gate. **Design-only this campaign.**

## Method lessons (carry forward)
1. **Warm-median only** (≥5 trials). Cold single-runs measure shader-compile.
2. **Validate quality over many prompts** — one argmax-identical prompt is not evidence; divergence shows over a distribution.
3. **The long-context regime (≥2.5k) is the slow regime** (~19 tps) and the one where KV compression pays.

## Recommendations (ranked)
1. **SHIP** `--profile fast` as the documented *speed-priority* config (+7.5%, 83% identity = mild quant trade; keep the bit-identical default for quality-priority).
2. **SHIP** `F16_KV` for long-context / memory-constrained use (−50% KV; mild quality; growing speed at depth).
3. **FRONTIER**: the only remaining big speed lever is the ~1.6× llama.cpp gap → fund a focused MST-diff profiling pass (bible's path). Everything easier is tapped or dead.
4. **DENSITY**: a careful `eagle5*` reference-audit → parity-gated consolidation (~1.6k LoC).
5. **SSM moat**: the genuine long-context differentiator is the RWKV-7/Mamba-2 flat-decode path (no KV wall), not more Qwen-Q4 micro-opt. The KD/draft work feeds this, not spec.
