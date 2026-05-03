# dismantle roadmap

Public-facing phase plan. Every phase has a verification gate; if the
gate fails, the phase doesn't ship and the work continues.

This project was previously named *prism* and aimed at discrete
diffusion language models. The diffusion direction is dropped; the
project is now a Mac MoE engine.

---

## v0.1.0 — Apple Silicon MoE inference

### Phase 0 — Scaffold + first text  (week 1)

- Rust workspace, four crates, macOS CI green
- GGUF v3 reader; load DeepSeek-V2-Lite Q4_K_M
- Naive forward pass via MPSGraph reference path
- `dismantle generate --weights … --prompt …` produces coherent text
- `docs/m3_audit.md` written

**Gate:** model output matches `llama-cli` on the same prompt at
temp=0 (≤2-token drift).

### Phase 1 — Wedge 2: fused Q4_K_M dequant  (weeks 2–3)

- `quant.metal`: Q4_K_M dequant fused inside the FMA loop in
  threadgroup memory. DRAM only ships 4-bit weights.
- `moe.metal v1`: per-expert grouped GEMM uses the fused-dequant
  kernel.
- Replaces Phase 0's MPSGraph MoE path.

**Gate:** ≥1.5× decode tok/s vs. llama.cpp Metal AND ≥0.7× MLX
on DeepSeek-V2-Lite Q4_K_M (closing, not winning yet); correctness
atol=1e-3 on 50-prompt suite. See `docs/competitive_audit.md` for
why MLX is the new bar.

### Phase 2 — Wedge 1: single-launch MoE  (weeks 3–4) ✓ CLOSED

Shipped in v0.1.0. Three layered wedges landed:

- Batched expert GEMV (indexed in-place from fused GGUF tensors).
- No-pack indexed dispatch (route IDs index expert weights directly;
  no byte packing or expert copy per token).
- One-command-buffer MoE block (routed + shared kernels commit and
  wait once per MoE block).
- Strict single-kernel fused FlashMoE (`moe_block_fused_v2lite` +
  indexed variant) — correctness-attested at atol < 1e-3, opt-in via
  profile `moe_schedule: single-kernel`. Not the default: the
  workgroup-per-output-row design is decode-redundant (~90× slower
  than batched on M3 Pro). Two-stage redesign is a v0.2 wedge.

**Measured:** dec_tps 1.61 median (M3 Pro 18 GB, DeepSeek-V2-Lite
Q4_K_M, 3 trials × 64 tokens). 4.7× over the pre-layered baseline
(0.34 tok/s). Dense path (Qwen2.5-3B Q4_K_M): 1.28 dec_tps.

See `docs/v0.1.0_closeout.md` for full attestation.

### Phase 2.5 — Wedge 3: GPU sampling  (week 4)

- `sample.metal`: top-K, top-P, temperature, repetition penalty,
  JSON-mask all on GPU. Logits never leave the GPU.

**Gate:** ≥1.3× decode tok/s vs. Phase 2 at temp=0; no quality
regression.

### Phase 3 — Qwen3-MoE + MLA polish  (week 5)

- `model/qwen_moe.rs` — second architecture validates the kernel
  pack isn't DeepSeek-shaped.
- MLA kernel optimization (compressed-KV layout).
- Cross-architecture correctness suite.

### Phase 4 — `dismantle serve` + continuous batching  (week 6)

- axum server, OpenAI-compat endpoints, SSE streaming.
- Slot manager: prefill/decode interleaving; concurrent requests
  share MoE kernel launches.

**Gate:** throughput at batch=4 ≥ 2.5× batch=1.

### Phase 4.5 — Wedge 4: shared-expert speculative decoding  (week 7)

- Shared-expert-only draft + routed-expert verifier loop.
- Ablatable via `--no-speculate` for headline benchmarking.

**Gate:** ≥1.5× decode tok/s on DeepSeek-V2-Lite at acceptance rate
≥0.7; no quality regression.

### Phase 5 — Wedge 5: prefill cache + headline numbers  (weeks 8–9)

- mmap-backed cross-session KV cache.
- Bench suites: `decode-tps`, `prefill-tps`, `ttft`,
  `throughput-vs-batch`, `bandwidth-utilization`, `competitive`
  (head-to-head vs. llama.cpp; MLX cited from Apple's published
  numbers in `docs/competitive_audit.md`).
- Auto-generated `docs/benchmarks.md`; demo video; HN/r/LocalLLaMA
  post.

**Gate:** decode tok/s ≥1.0× MLX; ≥3× llama.cpp; TTFT on cached
prefix prompts <100ms (vs. ~1s cold).

### Phase 5.5 — post-launch wedges (no committed timeline)

Three wedges added after the 2026-04 competitive audit found three
research gaps that nobody has shipped on Apple Silicon yet. Each is
small enough to land independently; none gates v0.1.0.

#### Wedge 7 — expert-temporal-locality predictor

Cohere measured 38% step-to-step routing correlation on MoE decode
(vs. 11.8% random baseline). Pre-fetch the top-K experts the gate
is most likely to pick *next* while the current MoE block is still
computing. Foundation already has `moe::dispatch::build_work_queue`;
extend it to consume an `ExpertPredictor` that biases prefetch order.

**Competes against:** MLX (no such mechanism), llama.cpp (no such
mechanism), SwiftLM (SSD-streaming-only, no temporal predictor).
**Expected uplift:** 1.3–1.6× decode tok/s on memory-bound workloads.
**Code already in place:** `crates/dismantle-core/src/moe/dispatch.rs`
work-queue builder; trait wiring is the new work.

#### Wedge 8 — asymmetric quant per expert role

Routed experts are sparse (only ~10% activate per token) and
bandwidth-bound by their count. Shared experts are dense (always-on,
fire on every token) and accumulate quantization error every step.
Quantize asymmetrically: **Q4_K_M for routed, Q8_0 (or fp8) for
shared**. Cheapest big wedge to land — model-loader and dispatch
change, no kernel work.

**Competes against:** everyone (genuinely unpublished for MoE on
Metal). **Expected uplift:** 1.15–1.25× on quality-equalized
comparisons; better quality at the same speed.
**Code already in place:** full Q4_K / Q5_K / Q6_K / Q8_0 coverage in
`crates/dismantle-core/src/quant/mod.rs`.

#### Wedge 11 — Q8 KV cache (with calibration)

Halves KV cache memory and KV cache read bandwidth. The win on the
M3 Pro 18GB is bigger than tok/s — it *unlocks* longer contexts that
MLX and llama.cpp default-fp16 can't fit in unified memory at all.

**Competes against:** matches llama.cpp (which has it); beats MLX
(no quant KV). **Expected uplift:** 1.2× decode tok/s, plus 2× max
context length at the same memory budget.
**Code already in place:** `crates/dismantle-core/src/cache/mod.rs`
`KvCache` is plumbed; needs a calibration pass + a Q8 storage
backend.

#### Punted to v0.2

- **Wedge 6 (SSD-resident expert streaming).** SwiftLM owns this
  niche; their >100B-on-64GB regime is out of scope on M3 Pro 18GB.
  Revisit on M3 Ultra hardware in v0.2.

---

## v0.2.0 (post-launch, no committed timeline)

### v0.2 performance wedges (in priority order)

1. **Two-stage fused MoE** — eliminate redundant intermediate compute
   by persisting the intermediate vector to device memory between two
   dispatches in one command buffer. Stage 1: compute all expert
   intermediates without per-output-row redundancy. Stage 2: read
   intermediates, apply routing weights, accumulate output. Realistic
   target: ≥1.3× over `indexed-no-pack-one-cb`.
2. **Metal MLA decode** — replace the current CPU `mla_decode_step`
   with a Metal kernel. Compresses KV retrieval bandwidth.
3. **Layer-CB** — extend command buffering across layer boundaries
   (currently per-MoE-block). Reduces encoder/decoder synchronization
   overhead.
4. **Decode arena** — pool transient Metal buffer allocations across
   decode steps. Eliminates per-step buffer alloc/dealloc churn.

### v0.2 model / architecture work

- safetensors loader (HF native fp16/bf16).
- Mixtral 8x7B as a hero number on M3 Max / M3 Ultra.
- DeepSeek-V2 (full, 236B) on M3 Ultra 192 GB.
- **Wedge 6 — SSD-resident expert streaming** (compete with SwiftLM
  on the >100B-on-Mac story; needs NVMe regime + M3 Ultra).

### v0.2 bench harness

- llama.cpp competitor now passes `-ngl 99` (landed in v0.1.0).
  Head-to-head bench vs. llama.cpp Metal + MLX is a v0.2 milestone.

## v0.3.0 (no commitment)

- CUDA backend (cross-platform). Only revisit after v0.2 lands and
  there is external demand from non-Mac users.

---

## What dismantle will not do

- Compete with llama.cpp on dense models. dismantle is a MoE engine;
  dense paths exist only because some MoE models have dense layers.
- Train or fine-tune models.
- Ship a chat UI. We provide an OpenAI-compatible HTTP surface;
  bring your own client.
