# Phase 2 — reality check

**Status:** captured 2026-04-30 after WB (full attention + LM-head weight pinning) shipped and benched.
**Purpose:** pin the findings that retire the `r_mlx ≥ 0.8` goal and reframe Phase 2 around the dual-path positioning (FlashDMoE-on-Apple-Silicon + dense Qwen2.5-3B).

## Headline finding

**Weight pinning (WB) delivered no measurable speedup** on the M3 Pro 18GB unified-memory architecture for DeepSeek-V2-Lite Q4_K_M:

| Metric | Pre-WB (W3.1) | Post-WB (1-trial) | Δ |
|---|---:|---:|---:|
| dec_tps | 0.4812 | 0.4755 | -1.2% (within 1-trial noise) |

The WB code is correct (137 pinned `metal::Buffer`s parity-attested at atol=1e-3, 7/7 PASS in `phase2_weight_pinning_parity.rs`; phase-1 regression floor 8/8 unchanged) — it just doesn't help on this hardware. **Apple Silicon's `MTLResourceOptions::StorageModeShared` makes `new_buffer_with_bytes` essentially free**: CPU and GPU share one physical memory pool; the "memcpy" was Metal Buffer metadata setup, not actual data movement. The `_phase2_speed_followups.md` doc's "~220 ms/token of memcpy" estimate was discrete-GPU intuition transplanted to unified memory; it's wrong.

## What this invalidates

**The wedge-ladder cumulative math.** The previous plan claimed:

```
0.481 × 1.0 (WB) × 15× (WA FlashDMoE) × 1.4 (...) × ... = 27 dec_tps → r_mlx ≈ 0.90
```

With WB at 1.0× (measured) and the followups doc's other estimates similarly suspect, recalibrated:

| Wedge | Old est. | Realistic est. | Reasoning |
|---|---:|---:|---|
| WB | 1.2-1.4× | **1.0×** | measured: unified memory makes memcpy elimination a no-op |
| WA FlashDMoE | 15-30× | **1.2-1.5×** | dispatch overhead is 700+ dispatches × ~1 ms = ~1 sec of 2.1 sec/token. MoE = 189 of 700 = 27% of dispatch budget. Eliminating ~95% of MoE dispatches saves ~25% of dispatch time = ~12% of decode |
| WC Q8 attention | 1.3-1.5× | **1.1-1.2×** | bandwidth wins are also unified-memory-discounted |
| WD Metal MHA | 2-3× | **1.5-2×** | real win — eliminates CPU bottleneck |
| WE Q8 KV | 1.2× | **1.05-1.1×** | same unified-memory logic |
| WF simdgroup matmul | 1.3× | **1.2-1.5×** | kernel-execution speedup is real |
| WG GPU sampling | 1.3× | **1.05-1.1×** | sampling is small fraction of decode |

**Cumulative realistic on DeepSeek-V2-Lite / M3 Pro:**
```
0.481 × 1.0 × 1.3 × 1.15 × 1.7 × 1.07 × 1.3 × 1.07 ≈ 1.5 dec_tps → r_mlx ≈ 0.05
```

**Achievable r_mlx ranges per scenario:**
- DeepSeek-V2-Lite Q4 on M3 Pro, all wedges: **0.05–0.10** ceiling
- Same model on M3 Max / Ultra: same *ratio* (both dismantle and MLX get the hardware boost) — absolute tok/s up but `r_mlx` unchanged
- DeepSeek-V2-Lite Q4 + graph-level forward-path rewrite: maybe **0.3–0.5** (months of research-level engineering)
- r_mlx ≥ 0.8: needs different model + different architecture + likely different research direction

**Conclusion: r_mlx ≥ 0.8 is structurally not achievable on M3 Pro for this model class without a multi-month rewrite. Retired as the Phase-2 goal.**

## Real per-token decode breakdown (estimated)

~2.1 sec/token total decode, of which:

| Component | Estimated cost | Notes |
|---|---:|---|
| Dispatch overhead | ~1 sec | 700+ dispatches × 1-3 ms each (commit + waitUntilCompleted). MoE = 189, attn = 5 × 27 = 135, norms = 4 × 27 = 108, gate = 27, LM head = 1, plus various |
| CPU MHA decode step | ~0.3 sec | 27 layers × ~10 ms; fp32 softmax-attention runs on host, KV cache is `Vec<f32>` |
| Kernel execution | ~0.5 sec | actual GPU compute time across ~700 dispatches |
| Other | ~0.3 sec | embed lookup, residuals, sampling, sync points |

**The followups doc's "dispatch overhead is the elephant" thesis is correct** — but at smaller magnitude than the doc estimated. Eliminating most MoE dispatches via FlashDMoE saves ~12% of decode, not 90%.

## Strategic pivot

**Goal retired:** *r_mlx ≥ 0.8 on DeepSeek-V2-Lite Q4_K_M.*

**Goal adopted:** ship `dismantle 0.1.0` as a **dual-path Apple Silicon LLM engine**:

1. **MoE path (DeepSeek-V2-Lite Q4_K_M):** technical claim is *"first FlashDMoE-style single-launch fused MoE on Apple Silicon."* The `moe_block_fused` kernel at `shaders/moe.metal:208` is currently a stub (empty `(void)id;`). Writing it for real is the unique differentiator. Realistic 1.2-1.5× over current dismantle MoE path.
2. **Dense path (Qwen2.5-3B-Instruct Q4_K_M):** pragmatic working product. Reuses the 8 existing parity-attested Metal kernels. Realistic target r_llama ≥ 0.7, r_mlx ≥ 0.5 on dense (where llama.cpp has ≥2 years of optimization advantage and we're aiming for "competitive enough to be useful").

Both ship under MIT.

**Public-facing claim:** *"First MoE inference engine on Apple Silicon with single-launch FlashDMoE-style expert dispatch. Also runs popular dense models (Qwen, Llama) at competitive speeds. MIT-licensed; all kernels parity-tested at atol=1e-3."*

## Workflow change

The repeated `>>>>` bash continuation-prompt friction during haul launches was a symptom of the haul approach being miscast as the dev loop. Fix:

- **Interactive cargo + bench loop for impl + measurement.** No detached long runs during code work.
- **Haul reserved for batch closeout** at Stage 3 (publication mechanism, not dev driver).
- **`tools/haul/launch_super_haul_2.sh` stays** as the single-command paste-safe launcher for that one closeout pass.

## What's preserved

- **WB code stays.** Parity-attested, correct, doesn't hurt anything. May pay off on different hardware (M3 Max with different bandwidth) or future Apple Silicon. Removing now would be premature.
- **8 existing parity-attested Metal kernels** (G1.1-G1.4 + H2.1-H2.4) — foundation for both paths.
- **`_phase2_token_baseline_50.hashes`** — locked correctness floor.
- **Phase 1 + Phase 2 spec, manifest, prior closeouts** — historical record.

## What's dropped

- **r_mlx ≥ 0.8 as the Phase-2 goal.**
- **The wedge ladder (WA-WG cumulative math) as a roadmap.** Replaced by two stages: write `moe_block_fused` + add Qwen dense path.
- **Per-wedge haul-driven validation cycles.** Replaced by interactive cargo+bench.
- **`_phase2_haul1_manifest.md` as a dev driver.** Manifest stays for Stage-3 closeout only.

## Next steps (per the approved plan at `~/.claude/plans/it-happened-again-clearly-federated-dawn.md`)

1. Stage 1: write `moe_block_fused` shader for real, incrementally (1-expert → top-K → shared). Wire into `ffn()`. Parity-attest. Bench. ~3-5 days.
2. Stage 2: add Qwen2.5-3B dense forward path. Reuses existing kernels. ~3-4 days.
3. Stage 3: closeout doc + README + optional demo. ~1 day.

Total: ~7-12 attended days to ship `dismantle 0.1.0`.
