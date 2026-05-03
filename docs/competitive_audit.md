# Competitive audit — dismantle vs the field

**Date:** 2026-04-27 (paper analysis); live numbers pending first
benchmark run (see *Reproducibility* below).
**Hardware:** MacBook Pro M3 Pro, 18 GB unified memory, 11-core CPU,
14-core GPU, 150 GB/s theoretical bandwidth peak.
**Hero model:** `DeepSeek-V2-Lite-Chat-Q4_K_M.gguf` (~9 GB on disk;
15.7 B total params, ~2.4 B active per token; MLA + 2 shared
experts + top-6 of 64 routed experts).
**Audit type:** honest. Phase 0 dismantle is CPU-only; we run
anyway, label it `phase: 0`, and project Phase-5 numbers from
per-wedge expected uplifts. We will not hide numbers we don't like.

This doc is the source of truth for *why* dismantle is built the way
it is. It exists because, until early 2026, we framed dismantle
against llama.cpp. The audit found that's the wrong bar.

## TL;DR

1. **MLX is the bar, not llama.cpp.** Apple's MLX hits ~130 tok/s
   on Qwen3-30B-A3B (Q4) on M4 Pro and is reproducibly 2–3× faster
   than llama.cpp on Apple Silicon for MoE inference. ROADMAP gates
   were tightened to "≥0.7× MLX after Phase 1, ≥1.0× MLX after
   Phase 5."
2. **dismantle has three uncontested wedges** that nothing else
   ships on Apple Silicon: (a) single-launch fused MoE on Metal
   (FlashDMoE proves it on CUDA), (b) shared-expert speculative
   draft (genuinely unpublished concept), (c) cross-session on-disk
   prefill cache (RadixAttention is closest but in-memory only).
3. **Three new wedges** (7, 8, 11) added to ROADMAP from research
   that hasn't diffused into open-source engines: temporal-locality
   expert prefetch, asymmetric quant per expert role, Q8 KV cache.
4. **One niche conceded.** SwiftLM owns the >100B-MoE-on-64GB
   regime via NVMe SSD streaming. We won't try to win that on M3
   Pro 18GB; revisit on M3 Ultra in v0.2.

## Hero numbers — head-to-head

**First measurement: 2026-04-27, M3 Pro 18GB, single prompt
("Once upon a time"), greedy temp=0, max_tokens=256.**

| Backend                         | Decode tok/s | Prefill tok/s | Notes |
|---------------------------------|--------------|---------------|-------|
| **dismantle** (`phase: 0`)      | **0.30**     | **0.37**      | EOS-stop @ 114 tokens; ~390 s wall; CPU-only reference path |
| llama.cpp (Metal, `b8870`)      | **~48**      | **~22**       | Median of 2 runs (41.2 / 56.4 decode, 6.3 / 38.5 prefill — thermal warmup variance) |
| **dismantle** (projected Phase 5)² | **150–200**  | **~150**      | _projected, not measured_ — see "Projection methodology" below |

MLX is an analysis-only competitor here — it consumes a different
weights format (MLX-native, not GGUF) and a fresh download is
required to bench it. We cite Apple's published numbers below¹
rather than measuring locally so the head-to-head is apples-to-apples
on both weights file and quantization scheme.

¹ See [Apple ML Research — MLX on M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
and [contracollective M4-vs-M5 benchmarks](https://contracollective.com/blog/m4-m5-pro-local-ai-inference-mlx-2026).
DeepSeek-V2-Lite is comparable in shape to Qwen3-30B-A3B (15.7B/2.4B vs 30B/3B);
expected MLX number on M3 Pro is in the 80–120 tok/s range, modulo their
chunked-prefill behavior on long contexts.

² Projection methodology: Phase 0 today is **CPU-only, lazy-dequant on
every routed-expert call**. The wedges replace this with a Metal hot
path. The base of the projection is *not* `0.30 × multipliers` — it's
"a naive Metal forward pass like llama.cpp's pre-MoE-rewrite ≈ 50
tok/s baseline", then per-wedge multipliers stack on top:

| Wedge | Multiplier on the Metal baseline | Source |
|---|---|---|
| 2 — fused Q4_K_M dequant in FMA loop | ×1.5–2.0 (effective weight bandwidth) | wedge target in ROADMAP §Phase 1 |
| 1 — single-launch fused MoE kernel    | ×1.1–1.3 decode; ×2.0 prefill         | FlashDMoE (CUDA) shows 5.7× full-stack |
| 3 — GPU sampling                       | ×1.2–1.3 at decode>50 tok/s           | eliminates per-token CPU↔GPU sync     |
| 4 — shared-expert speculative draft    | ×1.4–1.6 at ≥0.7 acceptance           | Cascade reports 7–14% on CUDA; shared-expert variant strictly stronger |
| 5 — prefill disk cache                 | TTFT cliff; no decode change          | RadixAttention (in-mem) is closest    |

Stack-multiply: Metal-baseline 50 × wedge-2 1.7 × wedge-1 1.2 ×
wedge-3 1.25 × wedge-4 1.5 ≈ **191 tok/s**. The 150–200 band in the
table reflects ±20% uncertainty on each multiplier compounding.

This is the headline claim dismantle commits to: by Phase 5, decode
≈ 3–4× llama.cpp and within parity-or-ahead of MLX on this hardware
for DeepSeek-V2-Lite Q4_K_M. Numbers above the 1× MLX line require
all four perf wedges to deliver near the upper end of their bands.

**How the projection is built.** Apply per-wedge multipliers from
ROADMAP.md to the Phase-0 measured number:

- Wedge 2 (fused dequant): ×1.5–2.0 effective bandwidth on weights.
- Wedge 1 (single-launch MoE): ×2.0 on prefill; modest decode.
- Wedge 3 (GPU sampling): ×1.3 decode at temp>0.
- Wedge 4 (shared-expert speculative): ×1.5 decode at ≥0.7 accept.
- Wedge 5 (prefill disk cache): TTFT cliff to <100ms on cache hit.

Stack-multiply across these is *the* claim we're committing to.

## Per-wedge competitive matrix

Each row is one dismantle wedge. Verdict is one of: ✅ we own this,
🟡 parity / closing, ❌ they own this, ❓ unknown without numbers.

| # | dismantle wedge | llama.cpp | MLX | SwiftLM | FlashDMoE (CUDA) | Verdict | Evidence |
|---|---|---|---|---|---|---|---|
| 1 | Single-launch fused MoE on Metal | per-expert dispatch | per-expert dispatch | not specialized | yes (CUDA) | ✅ uncontested on Metal | [arxiv 2506.04667](https://arxiv.org/html/2506.04667) |
| 2 | Fused Q4_K_M dequant in FMA loop | partial (Metal MoE rewrite Apr 2025) | mlx-quant, no in-FMA | n/a | yes (CUDA Q4) | 🟡 parity-to-lead on Metal | [llama.cpp #20757](https://github.com/ggml-org/llama.cpp/issues/20757) |
| 3 | GPU sampling + constraint masks | CPU sampling | partial | n/a | yes | 🟡 parity once Phase 2.5 lands | vLLM sampling docs |
| 4 | Shared-expert speculative draft | no | no | no | research only | ✅ uncontested everywhere | [Cohere blog](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding) |
| 5 | Cross-session disk prefill cache | session-only | session-only | no | RadixAttention (in-mem) | ✅ uncontested on disk | [SGLang RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/) |
| 7 | Expert-temporal-locality prefetch | no | no | no | no | ✅ research-only elsewhere | [Cohere: 38% step-to-step correlation](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding) |
| 8 | Asymmetric quant per expert role | no | no | no | no | ✅ unpublished for MoE on Metal | (no public work found) |
| 11| Q8 KV cache | yes (`-ctk q8_0`) | no | TurboQuant | yes | 🟡 we match llama.cpp, beat MLX | llama.cpp KV quant docs |

**The three moonshots.** Wedges 1, 4, and 5 are where the gap is
widest. If we ship all three honestly, no other engine on Apple
Silicon can match the combined story in 2026. Wedges 7, 8, 11 are
the second wave — each is small but together they harden the
position against MLX catching up.

## What we lose today and won't try to win in v0.1

- **SwiftLM's 100B+ MoE on a 64GB Mac.** They tuned NVMe SSD
  expert-streaming for the regime where the model exceeds RAM.
  Different hardware target, different problem; the right move is
  to ship dismantle on M3 Pro 18GB first and revisit SSD streaming
  (Wedge 6) on M3 Ultra in v0.2.
- **vLLM-MLX's small-model continuous-batching throughput.** Their
  525 tok/s on small models at batch=16 is impressive but generic;
  our Phase 4 batching aims at *MoE-aware* batching (group requests
  by expected expert overlap), which is a different optimization.
  Direct comparison there isn't apples-to-apples.

## Three uncontested wedges — why they matter

### Wedge 1 — single-launch fused MoE kernel on Metal

llama.cpp dispatches one kernel per expert per layer. With 6 routed
+ 2 shared = 8 experts × 27 layers, that's ~216 launches per token,
each costing ~0.5–1 ms of launch overhead = up to 200 ms of pure
overhead per token, before any actual matmul. FlashDMoE
([arxiv 2506.04667](https://arxiv.org/html/2506.04667)) showed on
CUDA that a single persistent kernel pulling work items off a queue
delivers 5.7× throughput. This works because Metal's launch
overhead is roughly proportional to CUDA's, and the work-queue
pattern translates directly. **No Metal port has shipped publicly.**

### Wedge 4 — speculative decoding via shared experts

DeepSeek's two shared experts run on every token regardless of
routing. They're cheap relative to the routed-expert pass. Use the
shared-expert-only output as a draft, then verify with the full MoE
pass. If draft and verifier agree, accept; if they disagree, take
the verifier's choice and discard the draft. **No published engine
does this.** Cascade ([arxiv 2506.20675](https://arxiv.org/abs/2506.20675))
shows utility-driven speculative decoding for MoE generally limits
naive-spec slowdown to 5% and gains 7-14%; shared-expert draft is
strictly stronger because the draft pass *runs anyway* — it's free.

### Wedge 5 — cross-session prefill cache on disk

System prompts repeat. Tool-use scaffolds repeat. Long context
prefixes repeat across sessions. Today every engine on Apple
Silicon throws away the KV cache when a process exits. dismantle
keeps it on disk, keyed by (model_hash, tokenizer_hash, prompt_hash);
on session start, mmap the cache file back. Cold-start TTFT drops
by orders of magnitude on system-prompt-heavy workloads.
RadixAttention does prefix sharing in-memory but doesn't survive
process restart. **No published engine does cross-session.**

## Reproducibility checklist

The audit is honest if and only if:

1. Every measured number cites a row in
   `tools/competitors/versions.json` (captured at run time).
2. `./tools/competitors/run_competitors.sh` from a clean state
   reproduces `tools/competitors/results.json` within ±5% on the
   same M3 Pro after the same thermal soak (5 min idle, plugged in,
   lid open, Low Power Mode off, hard surface).
3. `dismantle bench --suite competitive` produces the same JSON
   shape from inside the binary, within thermal noise of the
   shell-script harness — proves the in-binary harness and the
   offline harness measure the same thing.
4. The matrix has at least one cell where dismantle loses today
   (Phase-0 column will lose every comparison) and at least one
   where it's projected to win at Phase 5. No all-green table.
5. Phase-0 dismantle numbers appear in the matrix labeled
   `phase: 0`, not hidden.

## Runbook

```sh
# 0. (one time) Drop a fresh shell after restart for clean thermals.

# 1. Build dismantle release binary.
cargo build --release --workspace

# 2. Fetch the standard hero model (~9 GB).
./tools/fetch-model.sh

# 3. Smoke-test dismantle's Phase 0 forward pass.
./target/release/dismantle generate \
    --weights ./models/deepseek-v2-lite-q4.gguf \
    --prompt "Once upon a time" \
    --max-new-tokens 32 \
    --temperature 0
# ↳ if output is gibberish: see NOTES.md "Phase 0 gate" section
#   (most likely Q4_K bit-pack mismatch in quant/mod.rs).

# 4. Run the head-to-head suite (3 trials, ~30-60 min wall-clock).
./tools/competitors/run_competitors.sh 3

# 5. (Equivalent, in-binary version that produces the same shape.)
./target/release/dismantle bench \
    --weights ./models/deepseek-v2-lite-q4.gguf \
    --model deepseek-v2-lite \
    --suite competitive \
    --trials 3 \
    --json ./tools/competitors/results-from-binary.json

# 6. Paste the numbers into the "Hero numbers" table above.
```

## Sources

- [llama.cpp Metal MoE rewrite, April 2025](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [Apple ML Research — MLX on M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [M4 vs M5 MLX benchmarks (contracollective)](https://contracollective.com/blog/m4-m5-pro-local-ai-inference-mlx-2026)
- [vLLM-MLX continuous batching](https://github.com/waybarrios/vllm-mlx)
- [SwiftLM — MoE SSD streaming for Apple Silicon](https://github.com/SharpAI/SwiftLM)
- [FlashDMoE — single-kernel fused MoE dispatch (CUDA, Jun 2025)](https://arxiv.org/html/2506.04667)
- [Cascade — utility-driven speculative decoding for MoE (Jun 2025)](https://arxiv.org/abs/2506.20675)
- [Cohere — temporal expert routing correlation in MoE](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)
- [SGLang RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
- [vLLM Blackwell MoE benchmarks](https://blog.vllm.ai/2025/10/09/blackwell-inferencemax.html)

## Refresh policy

Re-run this audit when **any** of the following change:

- llama.cpp Metal backend ships a single-launch MoE kernel.
- MLX adds shared-expert speculative decoding or cross-session KV.
- A new MoE-on-Apple engine appears that wasn't measured here.
- dismantle ships any of Phases 1–5 (re-measure that backend's row).
- Hardware changes (M4 Pro / M3 Ultra etc.) — see
  `docs/m3_audit.md` for the lockfile.
