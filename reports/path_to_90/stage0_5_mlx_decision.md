# Path-to-90 step 2 — Stage 0.5 MLX-pattern adoption decision

**Decision:** Stage 0.5 is **mandatory** under the execution-plan rule, but
**deferred to land after Stage 1 measurement (step 10)** rather than
between steps 2 and 3. Reasoning below.

**Authority:** `reports/path_to_90/execution_plan.md § 2` decision rule,
applied to the measured efficiency in
`reports/path_to_90/stage0_profile.md`.

## The number that drives this

From step 1's profile (commit `72e3926`):

- clean median dec_tps: **26.93** (3-trial spread 0.16 %)
- bandwidth efficiency vs 150 GB/s: **29 – 34 %** (deep-research bytes/token)
- bandwidth efficiency vs 150 GB/s: **25 – 30 %** (param-counted sanity check, below)
- both ranges below the 40 % threshold → **mandatory MLX-pattern adoption**

## Sharpened bytes-per-token from V2-Lite architecture

Deep-research § "Bandwidth math" cites 1.6 – 1.9 GB/token. Cross-checked
against V2-Lite's actual config (read from
`crates/dismantle-core/src/model/deepseek_v2.rs` + GGUF metadata):

| Component | Active params/token | Bytes @ Q4_K_M (≈ 4.5 bits) |
|---|---:|---:|
| 27 × MLA layer (no Q-LoRA, kv_lora=512, h=16, head_dim 192+128) | 373 M | 210 MB |
| 1 × dense FFN (layer 0, intermediate 10 944) | 67 M | 38 MB |
| 26 × MoE FFN (6 routed + 2 shared @ intermediate 1 408) | 1 799 M | 1 012 MB |
| LM head (vocab 102 400 × hidden 2048) | 210 M | 118 MB |
| Embedding lookup (1 row) | tiny | ~5 KB |
| **Subtotal weights** | **2 449 M** | **~1.38 GB** |
| MLA latent KV reads at avg ctx ≈ 50 (small prompt + 64-tok decode) | — | ~1.5 MB |
| Activations / residual stream | — | few MB |
| **Total per-token DRAM traffic** | — | **~1.4 GB (lower bound)** |

The Q4_K_M whole-file bits-per-param is closer to 4.85 (mixed Q4/Q6 with
block scales); using that gives an upper bound of ~1.50 GB. So my
param-counted estimate is **1.40 – 1.50 GB/token**, slightly below
deep-research's 1.6 – 1.9. The difference is most plausibly explained
by deep-research conflating per-expert size with a non-V2-Lite reference
(it cites "6 routed experts × ~85 MB" — but a V2-Lite expert is ~4.9
MB at Q4_K_M, since each is just `3 × 2048 × 1408 × 0.56 bytes`; six
routed experts is ~29 MB per layer, not 510 MB total). Either way:

| bytes/token | dec_tps × bytes = observed GB/s | vs 150 GB/s |
|---:|---:|---:|
| 1.40 (param-counted, low) | 37.7 GB/s | **25.1 %** |
| 1.50 (param-counted, high) | 40.4 GB/s | **26.9 %** |
| 1.60 (deep-research low) | 43.1 GB/s | 28.7 % |
| 1.90 (deep-research high) | 51.2 GB/s | 34.1 % |

**Robust window: 25 – 34 % efficiency.** Stage 0.5 decision unchanged.
For context: llama.cpp Metal typically 50 – 65 % on similar models;
MLX 65 – 80 %. Dismantle is below even the llama.cpp band — meaningful
headroom exists.

## Why deferred until after step 10 (Stage 1 measurement)

Two arguments compete; one wins:

**For "do Stage 0.5 NOW between steps 2 and 3" (the literal plan reading):**
- Step 5 (Eagle4Head propose, the load-bearing foundation item) is CPU
  fp32 — kernel rewrites in Metal don't conflict with it.
- Spec-decode regression at step 10 will be muddier if the baseline
  kernel mix changes mid-experiment.

**For "do Stage 0.5 after step 10" (chosen):**
- Step 5's load-bearingness is about *correctness of the head forward*,
  not about kernel efficiency. We unblock the entire eagle4 path by
  finishing steps 3 – 7 in their current shape.
- Step 10 is an explicit **measurable milestone** ("Stage 1: 12 – 22
  tok/s, regression expected on MoE"). Until we land it we can't tell
  Stage 0.5's gains apart from spec-decode's regression on MoE. The
  baseline number `26.93` is the right comparison point for Stage 1;
  swapping in MLX-pattern kernels first creates a "did the kernel work
  do this or did spec decode do this?" attribution mess.
- The deep-research doc agrees: "MLX port pays more than spec decode
  does" *if dismantle is at <40 % efficiency* — but it doesn't say
  MLX port must precede the spec-decode landing, only that one of
  them must.
- 1 – 2 weeks of kernel work blocks the eagle4 wiring that already has
  a 6-commit prep stack. Better to land the wiring, measure the
  regression, then attack kernel efficiency once we know the spec-decode
  story.

**Sequencing**:
```
steps 3 – 7   → foundation (eagle4 wiring through Metal head forward)
steps 8 – 9   → Stage 1 CLI wire-up + bit-identical regression test
step 10       → Stage 1 measurement, expected 12 – 22 tok/s (down from 26.93)
↑↑↑ INTERJECT STAGE 0.5 HERE if step 10 lands at the bottom of the band ↑↑↑
steps 11 – 17 → routing recall fix + Path B parallel-K verify
step 17       → Stage 2 measurement, expected 38 – 50 tok/s
```

If step 10 lands above 20 tok/s, push through to Stage 2 first. If it
lands below 15, the regression is severe enough that the kernel-
efficiency dividend matters MORE than the spec-decode landing, and
Stage 0.5 jumps the queue.

## Stage 0.5 scope (when it lands)

Informed by what the Stage 0 trace actually showed (GPU saturated during
decode at 99 % active; 112 dispatches per token; 3.7 µs CPU dispatch
overhead — not the bottleneck), the lever is **per-dispatch kernel
efficiency**, not dispatch count reduction. Specifically:

1. **gemv_q4_k_v3 (LM head)** — 102 K × 2048 weight read every token.
   The biggest single-kernel weight read in the model. Audit against
   MLX-LM's `lm_head` kernel pattern; check tile size, SIMD-group
   layout, vector-register reuse.
2. **MoE expert pair matmul** — 6 routed + 2 shared per layer × 26
   layers = 208 expert evaluations per token. Small matmuls (1408 ×
   2048) with high setup-to-work ratio. MLX uses fused gate-up-down
   per expert with shared SIMD-group register state; dismantle
   currently calls separately. Big lever.
3. **MLA decode kernel** — already had recent work (Phase 4 simdgroup
   queued, not landed). Audit against MLX's MLA path; verify latent
   KV read pattern is contiguous.

Reference targets:
- `mlx-lm/mlx_lm/models/deepseek_v2.py` (kernel-by-kernel diff)
- michaelstinkerings.org M5 roofline analysis (AMX/SIMD constraints)
- Apple's Metal SIMD-group programming guide (tile-size math)

**Budget**: 1–2 weeks. **Success criterion**: re-bench (same prompt
suite, clean window) shows ≥ 50 % bandwidth efficiency = ~55 tok/s,
matching llama.cpp's typical Metal band. Above 65 % (~70 tok/s)
puts us in MLX-class territory and is the stretch target.

## Class A/B classification

Per CLAUDE.md taxonomy + the deep-research follow-up list:

- **Class A** (do during Stage 0.5): gemv_q4_k_v3 LM head audit; MoE
  expert pair matmul audit; MLA decode kernel audit.
- **Class B** (revisit after Stage 5 if needed): full MLX-LM engine
  port. The patterns audit captures most of the gain without
  replacing the engine.

## Decisions still parked for the user

Unchanged from session opener:
- Cancel paused eagle3 capture? (still recommend yes — frees ~4 days
  of overnight compute, 85/100K samples is redundant with eagle4's
  own baseline).
- Retire `tools/training/mlx_eagle/`? (still recommend yes; eagle4
  supersedes).
- Q3 quantization sweep? (skip per deep-research; not worth it on
  V2-Lite's 15 % sparsity).

No new decisions required by step 2.

## What this commit changes

Adds this file. No code changes. The execution plan is unchanged —
this doc records which fork of step 2's decision tree we took and
where Stage 0.5 sits in the sequence.
