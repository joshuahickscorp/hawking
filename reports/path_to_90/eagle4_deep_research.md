# EAGLE-4 + dismantle ceiling — deep-research synthesis

Distilled from a multi-source deep-research pass (May 2026). Practitioner
data points and citations preserved for traceability. Treat as the
**reality check on the V4 spec doc's 140–165 tok/s target** — the
realistic stage-5 ceiling is lower, and the path to it has hard
prerequisites that the V4 doc waves at.

## Refined tok/s projection (replaces convergence doc's projections)

| Stage | Adds | Earlier projection | **Refined** |
|---|---|---|---|
| 0 | baseline today | 25 | 25–35 (MLX port could lift before any spec decode) |
| 1 | EAGLE-4 chain, no K-batched verify | 18–25 (regression risk) | **12–22** (regression worse than feared on MoE) |
| 2 | + Path B parallel-K verify | 45–55 | **38–50** (Mixtral EAGLE only hit 1.5× vs dense's 2.7×) |
| 3 | + masked verify prefetch | 55–65 | **55–75 IF routing recall fixed**; 55–60 at current 17.78% |
| 4 | + tree decode | 80–90 | **70–95** (MoE tree multiplier 1.4–1.8×, less than dense's 1.5–2×) |
| 5 | + AMX/ANE/multi-queue/Q4-KV | 120–140 | **95–125 sustained, 135 peak** |
| **V4 spec doc target** | — | 140–165 | **Implausible without ANE doing real verify FLOPs** |

**Hard hardware ceiling**: ~190–220 tok/s theoretical, ~140–160 tok/s
practically achievable. 150 GB/s ÷ minimum per-token weight read
(~0.8 GB shared experts + attention + LM head) → 188 tok/s upper bound.
Anything above ~140 sustained needs the ANE providing additional
bandwidth-free MAC throughput — and the published evidence says ANE can't
do that for the verify pass (UMA contention).

## Six-month ceiling (with Class B items)

130–160 sustained, plausibly 175 peak on cache-friendly code. Stacks:
Medusa multi-head + async verify + full router fine-tune + predict-
routing-trace. Above 180 needs bandwidth that doesn't exist on M3 Pro
150 GB/s.

## Key reality checks the deep research surfaced

### 1. Per-token independent acceptance ≠ sequential chain acceptance

Our 87.48% is `P(draft_top1 == target_top1)` measured independently per
held-out position. **It is NOT chain `P(accept)` after K positions.**

Citations:
- e2enetworks practitioner write-up on EAGLE-3 (closest analogue to our
  baseline): "average acceptance length remains around 4.5–5.0 tokens
  per draft-verify cycle" on Llama-3.3-70B w/ 70–80% acceptance.
- Meta's production EAGLE on Llama-4 Maverick (arXiv 2508.08192):
  TPC=2.75, "decodes at about 4 ms per token (B=1) on 8×H100".

Per-token argmax-match overstates chain acceptance by 1.3–1.7× because
real-text positional dependencies cascade: one mispredicted token kills
the chain because EAGLE feeds draft hidden states forward.

**Applying ρ ≈ 0.94 per-step decay to α₀=0.8748**: α₁=0.822, α₂=0.773,
α₃=0.726, α₄=0.683, α₅=0.642. Expected chain length:
- K=4: τ ≈ 3.2
- K=5: τ ≈ 3.6
- K=8: τ ≈ 4.5
- Tree decode 1.4–1.8× on MoE → **effective tau ≈ 6.5–8 on code, 5–6 on chat**.

### 2. MoE spec-decode is a documented minefield

Stage 1's "regression risk on MoE" is empirically validated by every
published source:

- **"Utility-Driven Speculative Decoding for MoEs" (arXiv 2506.20675)**:
  *"In this work, we show that speculation is not a feasible solution
  for MoEs"* in its naive form.
- **MoE-Spec (arXiv 2602.16052)** on OLMoE: 127-token tree activates 54
  of 64 experts (~85% coverage), approaching full-model bandwidth.
  Expert-budgeting fix: 10–30% over EAGLE-3.
- **Mixtral-8x7B EAGLE-1**: only 1.5× speedup vs dense's 2.7–3.5×.
- **Qwen3.6-A3B llama.cpp benchmark (April 2026)**: zero spec
  configurations achieve net speedup on this MoE despite 100% draft
  acceptance. Mean decode drops 3–12% across ngram-cache, ngram-mod,
  classic draft. Cause: MoE expert-union loading on every drafted token.

**V2-Lite at 2.4B active / 16B total (15% activation ratio)** is closer
to OLMoE than to Mixtral, suggesting routing-aware mask is **necessary,
not optional**.

### 3. The 17.78% routing mask recall is our biggest risk

Comparison:
- **EAGLE-4 v2-spec: 17.78% top-8 recall**
- **EAGLE-4 v2-routing: 26.35% top-8 recall** (trades 3pp acceptance)
- **MoE-SpeQ (arXiv 2511.14102) on Mixtral**: *"a 4-bit quantized draft
  model can predict the expert selection of its full-precision parent
  with over 90% accuracy."*

Implication: Stage 3 prefetch only helps if prefetched experts are
right. At 17.78% recall, prefetching 8 experts → ~1.4 hits; the other
6.6 are wasted bandwidth, and we still wait on 4–5 we didn't prefetch.
**Caps Stage 3's prefetch win at ~5–10% instead of the ~15% it should
be.** That's where Stage 5 loses ~15 tok/s vs the V4 doc projection.

**The fix is routing-prediction fine-tuning** — MoE-SpeQ's 90% with a
4-bit quantized draft suggests our 17.78% is a training-data or
loss-weighting issue, not a fundamental limit. Probable cause: mask
loss weight 0.3 vs token CE 1.0 + aux MSE 0.5; mask head only sees
gradient when it's already small. Fix: dedicated routing-prediction
fine-tune pass with mask loss as primary objective.

### 4. Apple Silicon specifics on M3 Pro 18 GB

**Bandwidth math** (the spine of everything):
- M3 Pro: 150 GB/s
- Per-token reads on V2-Lite Q4_K_M:
  - Shared experts (2 active) + attention + LM head + embeddings: 1.0–1.3 GB
  - 6 routed experts × ~85 MB: ~510 MB
  - MLA latent KV (93% reduction vs MHA per arXiv 2506.02523): 50–100 MB
  - **Total: 1.6–1.9 GB → theoretical ceiling 79–94 tok/s at 100% efficiency**
- llama.cpp Metal typically achieves 50–65% efficiency; MLX 65–80%.
- **Realistic unaccelerated single-stream ceiling on M3 Pro: 45–65 tok/s.**

Our 25 dec_tps baseline is below this. Either:
- a) Dismantle is running at <40% efficiency → MLX patterns could 2–3× it before spec decode
- b) Bandwidth estimate is wrong (KV is bigger, weights smaller, etc.)

**Action**: profile baseline efficiency via `Instruments → Metal System
Trace` BEFORE any spec-decode work. If <60% efficient, MLX port pays
more than spec decode does.

**AMX is the right home for the 60M EAGLE-4 draft head** (michaelstinkerings.org
"M5 GPU Roofline Analysis"):
- AMX peaks at 1,790 GFLOPS via direct `cblas_sgemm` at batch=1
- Through Core ML: 225 GFLOPS (8× slower)
- ANE: 19–35 GFLOPS for sizes that matter (1024–4096)

**Action**: AMX draft head via `Accelerate.framework` direct, NOT Core ML.

**ANE concurrency with GPU is largely a mirage for bandwidth-bound
work**: NPUMoE paper (arXiv 2604.18788) confirms UMA contention — ANE
and GPU share the 150 GB/s bus. For compute-bound ops (routing logits,
embedding lookups) ANE concurrency adds throughput; for bandwidth-bound
MoE forward it does not. ~25 µs Core ML per-prediction overhead.
**Caps ANE's Stage-5 contribution to 5–10%, not 20–30%.**

**Q4 KV on MLA is risky**:
- llama.cpp Issue #21385 reports q4_0 KV is lossless on GQA/MHA but
  needs per-head adaptive allocation.
- MLA's compressed 512-dim latent is already low-rank — further
  quantizing it amplifies per-dim error.
- **No published practitioner has shipped Q4 KV on V2-Lite or similar
  MLA models.**
- **Default to Q8 in latent space; Q4 only behind eval gate**.

### 5. Apple ReDrafter is not a useful anchor

ReDrafter's 2.3× speedup is **on M2 Ultra (819 GB/s) running dense
Vicuna**, not M3 Pro on MoE. Do not anchor planning on it.

## Free lunches we're not exploiting

In ROI order:

1. **MLX-LM engine port** (or MLX patterns ported into dismantle).
   yage.ai documents 3× MLX-vs-llama.cpp on Qwen3-Coder-30B-A3B on M4
   Pro (130 vs 43 tok/s). If dismantle is llama.cpp-class efficient,
   we leave a 2–3× on the table.
2. **Per-head adaptive MLA KV quantization** — FP16 sinks, Q4 rest.
   Reported +8% quality at same compression for standard MHA; applies
   analogously to MLA's compressed latent.
3. **ngram-mod / SuffixDecoding hybrid as fallback alongside EAGLE-4**
   for code/repetitive prompts. SuffixDecoding (NeurIPS 2025) reports
   *"5.3× over vanilla decoding"* on AgenticSQL, 2.8× faster than
   EAGLE-2/3, 1.9× faster than Token Recycling. Bimodal but worth
   ~10% on agentic/coding workloads on top of EAGLE-4.
4. **Direct-cblas AMX path** — already noted; 8× over Core ML for
   small ops.
5. **Async verify-start before draft completes** — Class B item, but
   trivial to land. Overlap last draft step's hidden production with
   first MoE verify layer's expert prefetch. Worth ~5–8%.

## Q3 / IQ3 quantization sensitivity on V2-Lite

V2-Lite's 2.4B active / 16B (15% sparsity) is a different beast than
dense 7B. Each expert sees only ~6/64 of tokens, so its weight matrices
are under-trained relative to a dense FFN. Quantization noise compounds
on sparse weights.

- DQ3_K_M paper (arXiv 2505.02390): on DeepSeek-R1, dynamic 3-bit
  matches Q4_K_M; standard Q3_K_M loses 1.5–3%.
- For V2-Lite specifically: **no published Q3 perplexity data exists**.
- Bartowski's GGUF page: IQ3_XXS at 6.96 GB vs Q4_K_M at 10.4 GB (33%
  smaller) — but *"These I-quants can also be used on CPU and Apple
  Metal, but will be slower than their K-quant equivalent."*

**Stick with Q4_K_M for the headline. IQ4_XS at 8.57 GB is the
smallest safe step down** (still K-quant family on Metal, ~17%
smaller, negligible quality loss).

**Action**: if router weights show outsized quantization sensitivity
(routing logits are tight — top-6 of 64 often with thin margins),
isolate the router weights as FP16 even when everything else goes Q3.

## Cross-comparison data points

Adjacent published numbers on M-series (no public M3 Pro 18 GB V2-Lite
Q4_K_M number anywhere — our 25 tok/s baseline is the most specific
data point that exists):

- **Llama 3.3 8B dense on M3 Pro 18 GB**: 45–60 tok/s
- **Qwen3-Coder-30B-A3B (3.3B active) Q4_K_M on M3 Pro 18 GB**: ~7.4
  tok/s (memory-pressured — 22.9 GB model spills to host, not clean)
- **Qwen3-30B-A3B 4-bit MLX on M4 Max** (~410 GB/s): 68+ tok/s; Q4_K_M
  GGUF same Mac: ~40 tok/s
- **Awni Hannun's DeepSeek-V2.5 (236B/21B active) 4-bit MLX**: 17.41
  tok/s decode, 70.3 GB peak (pipeline, 2-node)
- **Meta on Llama-4 Maverick** (production EAGLE, 8×H100): 4 ms/token,
  TPC=2.75
- **Red Hat / vLLM EAGLE-3 on Llama-3.3-70B**: 2.5× throughput at low
  request rates, 1.6× latency at B=4, break-even at B=32

## Operational caveats

- **M3 Pro 18 GB memory headroom is tight**: V2-Lite Q4_K_M 10.4 GB +
  EAGLE-4 head ~1 GB + KV at 4K context ~1–2 GB + system overhead ≈
  14–15 GB. Sustained pressure may trigger `iogpu.wired_limit_mb`
  swapping that destroys throughput. **Set explicitly: `sudo sysctl
  iogpu.wired_limit_mb=14336`**.
- The "140–165 tok/s V4 spec doc" projection assumed ANE doing
  meaningful verify FLOPs. Until ANE concurrency on MoE expert kernels
  is demonstrated (NPUMoE attempts, doesn't ship), treat as research
  bet not stage-5 expectation.
- Several recent papers cited (NPUMoE, MoE-SpeQ, MoE-Spec) carry 2026
  datestamps and may be pre-prints — treat absolute speedups as
  directional, not specs.

## Recommended block-ship gates

For each stage to "count" as shipped:

- **Stage 0 baseline**: profile says ≥55% bandwidth efficiency,
  Spec-Bench MT-Bench subset green
- **Stage 1**: 18–24 tok/s with zero quality regression
- **Stage 2**: ≥38 tok/s sustained, parity bit-identical at K=1 vs
  autoregressive
- **Stage 3**: routing recall ≥60% (NOT the current 17.78%) before
  enabling prefetch
- **Stage 4**: tree decode strictly dominates chain on Spec-Bench
- **Stage 5**: ≥95 tok/s sustained, peak ≥120 on code prompts

Below the lower bound on any stage = halt, debug, re-plan. The
operating-contract halt rule applies (CLAUDE.md § Halt rule).

## What changes in the execution plan

See `reports/path_to_90/execution_plan.md` for the 28-step sequence
that incorporates all of the above. Major reshuffles from the original
brief:

1. **Stage 0.5 (MLX port decision)** inserted after baseline profiling.
2. **Routing recall fix is a hard prerequisite for Stage 3** — moved
   before any prefetch implementation.
3. **AMX dispatch path** is direct `cblas`, not Core ML.
4. **ANE is for routing logits only**, never verify.
5. **Q4 MLA-KV gated behind eval**, defaults to Q8 latent.
6. **Tree decode is DySpec-style dynamic**, not Sequoia fixed.
7. **SuffixDecoding hybrid** added as Stage 5+ Class A item.
8. **Stage 5 measurement target is 95–125 sustained**, not 140–165.
   Anchor expectations to the refined number.
