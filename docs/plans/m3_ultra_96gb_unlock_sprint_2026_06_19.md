# Future Sprint - M3 Ultra 96GB Unlocks

Date: 2026-06-19

This is the expanded, intentionally overbuilt sprint plan for the future M3
Ultra 96GB machine. It assumes the current M3 Pro class machine has nearly
exhausted the obvious production work: batch-1 Q4_K transformer decode has
already been squeezed hard, many micro-optimizations have moved to the kill
ledger, and the remaining high-upside areas are architecture, serving systems,
speculative decoding, quantization, and model release infrastructure.

This plan is not a promise of speedups. Every base number must be rebenchmarked
on the new machine. The point is to identify what becomes newly possible or
newly worth trying when the system has far more memory, far more bandwidth, and
enough headroom to keep multiple models and caches resident.

## Hardware Baseline To Assume, Then Verify

Official Apple sources list M3 Ultra Mac Studio configurations with:

- M3 Ultra, configurable up to 32-core CPU and 80-core GPU
- 32-core Neural Engine
- 819 GB/s memory bandwidth
- 96GB unified memory configuration available

Sources checked:

- Apple Mac Studio technical specs: https://support.apple.com/en-us/122211
- Apple Mac Studio product page: https://www.apple.com/mac-studio/
- Apple M3 Ultra announcement: https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/
- MLX unified memory docs: https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html

Important caveat: the exact purchased config matters. The 96GB configuration can
exist with different CPU/GPU binning depending on purchase. Day 1 must record
actual CPU cores, GPU cores, OS version, Xcode/Metal version, thermal behavior,
and sustained bandwidth before interpreting results.

## Executive Thesis

The M3 Ultra does not just make the current project faster. It changes the
project type.

Current device:

- single-machine runtime research
- careful small-model experiments
- one expensive thing resident at a time
- short eval loops
- conservative long-context tests
- lots of "skip if memory is tight"

M3 Ultra 96GB:

- Hawking model foundry
- tuned/quantized model release pipeline
- multi-model speculative decoding
- long-context cache and state systems
- high-concurrency local serving
- full eval ledgers
- realistic distillation and QAT loops
- public model artifacts with real Apple Silicon benchmarks

The public goal becomes:

> I downloaded a Hawking model because it is smaller, faster, measured, and
> tuned for my Apple Silicon machine.

## What "Bleeding Edge" Means Here

For this project, bleeding edge is not "run the largest possible model slowly."
That is already a known Apple Silicon party trick. The sharper frontier is:

1. Highest useful tokens/sec per byte of model.
2. Highest useful tokens/sec per watt.
3. Long-running local agents with reusable prompt state.
4. Exact speculative decoding using resident draft and verifier models.
5. Low-bit tuned models that are still coherent and useful.
6. Apple-native release artifacts with reproducible eval and speed ledgers.
7. A serving engine that treats state, cache, and model choice as first-class
   runtime objects.

The M3 Ultra should be used to make Hawking a model/runtime product, not only a
faster local experiment.

## Non-Negotiable Rule

Do not compare raw M3 Pro TPS to raw M3 Ultra TPS and call that research.

Every result must be normalized by:

- model
- quant format
- prompt length
- output length
- concurrency
- batch size
- context/KV budget
- profile/env vars
- OS version
- thermal window
- memory pressure
- power mode
- background load
- median and P95/P99, not just best run

Use the M3 Pro numbers as historical reference only. The new machine creates a
new roofline.

## Day 0 Before Hardware Arrives

Build everything that does not require the larger machine:

| Item | Why |
|---|---|
| Hawking env alias layer | New public name without breaking old scripts. |
| Serving concurrency harness | Needed before any runtime claim. |
| Benchmark report generator | Avoids ad hoc conclusions. |
| Eval ledger schema | Required for model releases. |
| HQA/TQ inspect command design | Quant artifacts need provenance. |
| DraftSource trait | Spec decode should not be hardcoded to one draft. |
| Replay oracle | Spec decode needs cheap offline measurement. |
| RWKV state-fork test seam | Enables exact SSM speculation later. |
| Detached prefix KV store API | Can be unit-tested small before big cache runs. |
| Model-card template | Releases need repeatable documentation. |
| Hardware bring-up script | Day 1 should be boring and complete. |

Rule: the current M3 Pro should run a tiny version of every new system, even if
the ambitious settings are skipped.

## Day 1 Hardware Bring-Up

Do not start with new research. First measure the machine.

### Hardware Inventory

Collect:

- `sysctl -a` relevant CPU/GPU/memory facts
- `system_profiler SPHardwareDataType`
- macOS version
- Xcode version
- Rust version
- Metal shader compiler behavior
- available unified memory
- default GPU memory limit behavior for large ML processes
- SSD speed if model paging/loading matters

Output:

```text
reports/hardware/m3_ultra_96gb_inventory.json
docs/reports/m3_ultra_96gb_inventory.md
```

### Thermal And Power Baseline

Run a fixed 20-minute load:

- 5 minute warmup
- 10 minute sustained decode
- 5 minute cool-down observation

Record:

- frequency stability if available
- power draw if available
- SoC temperature if available
- sustained TPS drift
- fan/noise state if observable

Output:

```text
reports/hardware/m3_ultra_96gb_sustained.jsonl
```

### Runtime Smoke

Run exactly:

- Qwen/Qwen2.5 3B Q4_K_M batch-1
- Qwen/Qwen2.5 3B B=2/4/8/16 if supported
- RWKV7 0.4B batch-1
- RWKV7 0.4B B=2/4/8/16
- one long-context transformer prompt at 8k, 32k, 64k if model supports it
- one repeated shared-prefix chat workload
- one server queue/churn smoke

Output:

```text
reports/hardware/m3_ultra_96gb_bringup.jsonl
docs/reports/m3_ultra_96gb_bringup.md
```

## Day 1 Roofline

The first serious question is whether the current kernels scale with bandwidth
or hit a new occupancy/scheduling wall.

Measure:

- effective GB/s per hot Q4_K GEMV shape
- effective GB/s per batched GEMM shape
- prefill tokens/sec
- decode tokens/sec
- B-scaling efficiency
- readback bytes/token
- CPU encode overhead
- GPU busy vs wall
- command-buffer encode/commit timing

Derived metrics:

```text
single_stream_scale = m3_ultra_tps_b1 / m3_pro_tps_b1
aggregate_scale_b8 = m3_ultra_tps_b8 / m3_pro_tps_b8
effective_bw_ratio = m3_ultra_kernel_gbps / m3_pro_kernel_gbps
batch_efficiency_B = aggregate_tps_B / (single_tps * B)
```

Interpretation:

| Result | Meaning | Next action |
|---|---|---|
| B=1 scales near bandwidth | Current kernels are still good. | Prioritize model foundry and serving. |
| B=1 under-scales badly | New GPU shape exposes occupancy issue. | Build M3 Ultra-specific kernel lane. |
| B=8/16 scales well | Continuous batching is a public story. | Productize high-concurrency server. |
| B=16/32 regresses | Kernel geometry not ready. | Build high-B kernels before claims. |
| Prefill dominates TTFT | Chunked/prefix prefill first. | Build cache/chunk scheduler. |
| Decode dominates | Spec decode and quant first. | Build draft/verifier loops. |

## Sprint A - Rebenchmark Everything

This is not busywork. It prevents old intuitions from poisoning the new machine.

### A1. Core Generation Matrix

Models:

- Qwen2.5 0.5B, 1.5B, 3B
- RWKV7 0.4B
- one 7B dense model
- one MoE model already supported, if it fits comfortably
- any Hawking-tuned artifact available at the time

Settings:

- B=1/2/4/8/16/32 where supported
- prompt tokens: 128, 1k, 4k, 16k, 64k
- output tokens: 32, 128, 512
- greedy and sampled
- exact, fast, race, efficient profiles

Metrics:

- prompt tokens/sec
- decode tokens/sec
- TTFT P50/P95
- wall P50/P95
- aggregate TPS
- per-user TPS
- RSS
- GPU readback bytes
- J/token if available

### A2. Server Matrix

Workloads:

- shared system prompt chat
- coding agent loop
- long prompt burst
- mixed short/long prompts
- high-concurrency decode
- slot churn
- cancellation
- stop strings
- JSON mode

Required report:

```text
docs/reports/hawking_m3_ultra_serving_matrix.md
```

### A3. Quality Matrix

Every model speed result must have a quality context:

- PPL on fixed corpus
- argmax fixture trajectory
- instruction sanity set
- code sanity set
- JSON validity if claiming structured output
- known failure modes

Do not publish a speed number for a tuned/quantized model without the eval
ledger beside it.

## Sprint B - High-B Continuous Batching

The M3 Ultra's bigger GPU and memory bandwidth should make aggregate serving a
primary product axis. The current B=8 mindset is too small for the new machine.

### B1. Batch Geometry Sweep

Run B=1,2,4,8,16,24,32,48,64 if memory and slot data structures allow.

For each B:

- decode TPS
- prefill TPS
- batch efficiency
- readback overhead
- scheduler overhead
- slot churn correctness
- latency distribution

Stop only when:

- latency becomes unacceptable,
- kernel efficiency drops,
- memory budget is exceeded,
- or scheduling overhead dominates.

### B2. New High-B Kernels

Candidate kernels:

- Q4_K predec batched GEMM for B=16/32
- Q4_K LM-head token-only top-1 for B=16/32
- grouped Q/K/V projection for B>8
- batched FFN down with fused activation for high B
- RWKV grouped projection kernels for B=16/32
- persistent per-layer buffers for high-B decode

Gate:

- no B=1 regression
- no B=4/B=8 regression
- high-B clean A/B > 10 percent before default
- parity for greedy
- sampled path correctness if logits are materialized

### B3. Scheduler Upgrade

The scheduler should become an engine, not just a slot table.

Add:

- age-aware fair scheduling
- prefix-affinity scheduling
- length-bucketed prefill
- greedy-lane packing
- JSON/grammar lane packing
- sampled/full-logits lane separation
- cancellation-aware cleanup
- cache-hit priority
- max-latency guardrails

Metrics:

- queue wait P50/P95/P99
- starvation count
- cache-hit admissions
- greedy lane hit rate
- full-logits lane rate
- dropped/cancelled request cleanup time

## Sprint C - Long-Context Transformer Serving

96GB makes serious transformer KV experiments realistic.

### C1. Detached KV Block Store

Build the real version of the current system KV bank.

Features:

- token-prefix hash to resident KV span
- LRU and byte budget
- refcounts
- exact token verification
- copy into slot
- eviction metrics
- saved prefill token counter
- cold/warm TTFT comparison

Modes:

- system prompt bank
- project prompt bank
- file-context bank
- RAG document bank
- transcript prefix bank

Gate:

- greedy output identical to cold prefill
- warm shared-prefix TTFT win measured
- cache eviction cannot corrupt outputs

### C2. Paged KV / Blocked KV

Hawking does not need to become vLLM, but it does need to learn from vLLM.

Build a blocked KV abstraction:

- fixed-size token blocks
- block table per slot
- free list
- refcount
- hash for full blocks
- copy-on-write or branch support
- contiguous hot path where possible
- compact debug visualization

First target:

- Qwen dense models only
- greedy only
- no quantized KV at first

Later:

- f16 KV
- int4/per-channel KV
- prefix block sharing
- chunked prefill integration

### C3. Chunked Prefill

Long prefills must not monopolize decode.

Features:

- configurable chunk tokens
- yield between chunks
- prefill/decode interleaving
- prefix cache hit before chunking
- chunk-level cancellation
- TTFT fairness metrics

Gate:

- a 64k prompt does not starve short prompts
- output parity with unchunked prefill
- P95 latency improves in mixed workloads

### C4. Radix-Like Prompt State

Once detached KV exists, build a radix/prefix tree for prompt state:

- each edge is a token span
- leaf references KV/state blocks
- LRU by subtree
- prompt canonicalization helpers
- subtree hit metrics
- debug endpoint

This becomes the "Hawking State Store" for transformers.

Public language:

> Hawking reuses prompt state across multi-turn local agents instead of
> recomputing the same text.

## Sprint D - RWKV/SSM Persistent State

This is the biggest differentiated frontier. Transformers get cache reuse; RWKV
gets persistent state.

### D1. Stateful Sessions

Expose RWKV sessions as first-class objects:

- create session
- append user turn
- generate
- snapshot
- fork
- rollback
- delete

Native endpoints:

```text
POST /v1/hawking/sessions
POST /v1/hawking/sessions/{id}/append
POST /v1/hawking/sessions/{id}/generate
POST /v1/hawking/sessions/{id}/snapshot
POST /v1/hawking/sessions/{id}/fork
DELETE /v1/hawking/sessions/{id}
```

This should be native Hawking API first. OpenAI compatibility can stay stateless.

### D2. State Store

RWKV state is small enough to store aggressively:

- per-session state
- per-turn snapshot
- pre-tool-call snapshot
- branch snapshots
- compressed state archive
- disk persistence
- state hash/provenance

Gate:

- restoring a snapshot produces identical next-token trajectory
- fork/rollback is exact
- memory per session is constant with transcript length

### D3. Long-Running Agents

Use RWKV's O(1) state to support:

- day-long local coding session
- project memory
- tool call branches
- "try three plans, keep best" branching
- background summarization only as optional memory hygiene, not required for fit

Benchmarks:

- 1k, 16k, 64k, 256k equivalent transcript tokens
- memory stays flat
- latency stays flat except sampling/tool overhead
- quality sanity across long sessions

### D4. SSM Product Claim

Do not say "infinite context" casually. Say:

> Hawking RWKV sessions keep constant-size recurrent state, so runtime memory
> does not grow linearly with transcript length.

Then publish measured session traces.

## Sprint E - Speculative Decoding, Real Version

The current machine can build the seams. The M3 Ultra can make the full system
practical because target, draft, verifier scratch, and eval corpora can stay
resident.

### E1. DraftSource Trait

All draft types implement:

- reset
- observe accepted token
- propose K tokens
- score/confidence
- reject feedback
- stats

Draft sources:

- user n-gram
- suffix automaton
- 191M RWKV draft
- 50M/100M distilled RWKV draft
- low-bit HQA draft
- MTP heads
- EAGLE-style head for transformer
- replay oracle source for tests

### E2. Exact Verifier Contract

Never approximate acceptance in the first production version.

Rules:

- verifier emits the final accepted tokens
- rejected proposals do not mutate live state
- grammar constraints are applied before acceptance
- JSON mode remains valid
- temp=0 output equals no-spec output

### E3. RWKV State-Fork Speculation

This is the highest-upside custom path.

Mechanism:

1. clone live RWKV state to scratch
2. draft proposes K tokens
3. target verifies on scratch
4. accepted prefix commits scratch state
5. reject falls back to normal decode
6. governor adapts K and draft source

Because RWKV state is constant-size, fork/rollback is much cheaper and cleaner
than transformer KV branching.

### E4. Multi-Draft Residency

96GB enables multiple drafts resident at once:

- free n-gram first
- tiny draft for medium entropy
- larger draft for high-value long generations
- low-bit draft for cheap always-on mode
- target-only fallback

Governor inputs:

- entropy
- logit margin
- recent reject streak
- grammar mode
- prompt type
- latency target
- batch pressure
- draft health

### E5. Tree / Lattice Verification

Only after linear speculation is stable:

- draft proposes multiple branches
- verifier checks compact tree
- choose accepted path
- preserve exactness

Gate:

- accepted tokens/sec improves end-to-end
- branch overhead does not erase gains
- no batch fairness regression

### E6. Multi-Token Prediction Heads

Train heads for t+2/t+3/t+4:

- RWKV hidden/state to future-token logits
- verifier remains exact
- heads are proposal-only
- export with Hawking model artifact

This may be better than a separate draft model for small Hawking releases.

### E7. Spec Decode Metrics

Never publish acceptance alone.

Publish:

- accepted tokens per target forward
- draft TPS
- verifier TPS
- end-to-end TPS
- reject streak distribution
- governor disabled fraction
- quality parity result
- memory overhead
- batch interaction overhead

## Sprint F - Hawking Model Foundry

The M3 Ultra should make Hawking a release pipeline.

Pipeline:

1. ingest source model
2. run baseline eval
3. tune/distill
4. quantize
5. re-evaluate
6. serve benchmark
7. package artifact
8. publish model card and ledger

### F1. Model Families

Prioritize:

- RWKV7 0.4B as stateful flagship
- Qwen2.5 1.5B as fast dense transformer
- Qwen2.5 3B as quality/speed baseline
- one 7B dense model as upper local quality target
- one MoE only if serving path is good enough
- Mamba2 only after parity and kernels exist

### F2. Distillation Types

Use multiple distillation levels:

- teacher text generation
- chosen/rejected DPO pairs
- top-k logit KD
- full-vocab KL where feasible
- hidden-state matching
- draft-model distillation
- quantization recovery KD
- self-improvement loop with judge model

### F3. Local Teacher/Student Co-Residency

With 96GB, test:

- teacher 7B + student 0.4B
- teacher 7B + student 1.5B
- teacher 14B if practical + student 1.5B
- target + reward model + draft model

The goal is faster iteration, not necessarily training every model fully local.

### F4. Preference / Reward Pipeline

Build:

- teacher answer generator
- rejection sampler
- pair builder
- local reward model
- DPO/SimPO
- GRPO-style experimental loop if stable
- eval before/after each stage

Gate:

- instruction eval improves
- PPL does not collapse beyond threshold
- model remains fast and compressible

### F5. Nightly Foundry Automation

Nightly jobs:

- run one tuning stage
- export checkpoint
- evaluate
- quantize candidate
- benchmark
- update ledger
- promote or reject

Outputs:

```text
artifacts/hawking_foundry/runs/<run_id>/
  manifest.json
  train.log
  eval_ledger.jsonl
  bench_ledger.jsonl
  model_card_draft.md
  promote_or_reject.md
```

## Sprint G - Low-Bit And HQA

The black-hole ideology lives here: smallest useful model, fastest useful
runtime.

### G1. HQA Public Archive

Define Hawking Quant Archive only when the runtime path is real.

Contents:

- source model hash
- source license
- quant payload
- tensor map
- bit ledger
- sideinfo/outlier ledger
- bake command
- git revision
- eval hash
- compatible runtime version
- target hardware profile
- known limitations

Commands:

```bash
hawking quant bake ...
hawking quant inspect model.hqa
hawking quant verify model.hqa
hawking serve --weights model.hqa
```

### G2. Low-Bit Ladder

Run:

- FFN-only STRAND-2
- FFN-only STRAND-1
- time-mix-only STRAND-2
- time-mix-only STRAND-1
- all-layer mixed
- ternary-trained STRAND-1
- protected LM head
- outlier side-channel variants
- per-tensor sensitivity allocator

### G3. QAT With KD

96GB makes QAT less painful:

- all-layer QAT
- last-N-layer ablations
- top-k teacher logits cached once
- mixed precision per tensor
- teacher KD during recovery
- HQA export after each rung

Promotion gates:

- PPL threshold
- argmax fixture stability
- generation sanity
- benchmark improvement
- bpw ledger
- no runtime panic paths

### G4. Runtime Kernels

Do not ship low-bit archives that decode to f32 every token and lose speed.

Required:

- resident payload buffers
- resident sideinfo buffers
- bitslice GEMV
- row/block reductions
- high-B GEMM variant
- CPU reference parity
- GPU parity
- end-to-end RWKV/Qwen path

## Sprint H - Feature Richness

Speed is not enough. Hawking should feel like a full local model system.

### H1. Model Manager

Commands:

```bash
hawking models list
hawking models pull Hawking-RWKV7-G1-0.4B-HQA
hawking models inspect Hawking-RWKV7-G1-0.4B-HQA
hawking models bench Hawking-RWKV7-G1-0.4B-HQA
hawking models doctor Hawking-RWKV7-G1-0.4B-HQA
```

### H2. Profile Manager

Profiles:

- exact
- fast
- race
- efficient
- long-context
- agent
- batch
- low-power
- m3-pro
- m3-ultra

Auto-selection:

- detect hardware
- detect memory
- detect model
- choose profile
- print active levers
- write reproduction command

### H3. Native Hawking API

OpenAI compatibility stays, but Hawking needs native features:

- sessions
- state snapshots
- model inspect
- quant inspect
- benchmark endpoint
- cache stats
- prefix cache controls
- draft/spec stats
- structured output diagnostics
- hardware profile endpoint

### H4. Structured Output

Add:

- JSON object mode fully enforced
- JSON schema mode
- GBNF-like grammar
- token mask cache
- grammar-aware speculation
- grammar lane scheduler

Gate:

- valid JSON under streaming
- valid JSON with speculation
- valid JSON under batch churn

### H5. Embeddings And Rerank

Feature richness for local apps:

- real hidden-state embeddings
- RWKV state embeddings
- rerank endpoint
- pooling modes
- dimensions in model metadata
- batch embedding path

### H6. Adapter Registry

Support:

- LoRA/adapters where architecture supports it
- per-request adapter choice
- adapter scale
- hot load/unload
- adapter benchmark
- adapter model card

This is valuable for Hawking-tuned variants without duplicating whole models.

### H7. Observability

Add a first-class dashboard or report output:

- tokens/sec
- TTFT
- queue wait
- batch fill
- cache hits
- KV bytes
- state bytes
- draft acceptance
- energy
- active profiles
- model metadata

CLI:

```bash
hawking serve --metrics --trace-report reports/run.jsonl
hawking report reports/run.jsonl --html reports/run.html
```

## Sprint I - MLX / MPS / Metal Interop

Hawking is custom Metal-first, but the larger machine makes interop useful.

### I1. MLX As Training/Experiment Harness

Use MLX where it speeds experimentation:

- teacher/student tuning prototypes
- learned quantization experiments
- custom training loops
- distributed-style experiments later if multiple machines exist
- fast Python-side evals

Keep runtime critical path in Rust/Metal unless MLX clearly wins and can be
packaged cleanly.

### I2. MPSGraph/MPS Comparison

Occasionally compare against:

- MPSGraph transformer ops
- MLX kernels
- llama.cpp Metal
- native Hawking kernels

Purpose:

- find missed kernel techniques
- validate roofline assumptions
- avoid local maxima

### I3. Neural Engine

Do not assume ANE helps for the main Q4_K decode path. Prior work killed ANE for
that shape on the smaller machine. Revisit only for different tasks:

- small classifier/router
- reward model
- embeddings projection
- grammar/router side model
- image/audio side tasks

Gate:

- ANE path must not steal bandwidth from GPU hot decode
- clear end-to-end win

## Sprint J - Multi-Model Runtime

96GB makes a resident model router plausible.

### J1. Resident Set

Keep multiple models loaded:

- tiny draft
- fast assistant
- quality assistant
- embedding model
- reward/judge model
- target verifier

Router chooses by:

- task type
- latency budget
- quality budget
- context length
- structured output requirement
- current load

### J2. Cascades

Examples:

- small model answers easy prompts
- large model verifies or repairs
- draft model proposes, target verifies
- reward model ranks candidates
- embedding model retrieves context

### J3. Public Product

This becomes "Hawking Pack":

- a runtime profile
- several models
- a router config
- benchmark report
- intended workload

Examples:

- `Hawking-Code-Pack-M3Ultra`
- `Hawking-Agent-Pack-M3Ultra`
- `Hawking-LongContext-Pack-M3Ultra`

## Sprint K - Model Release Standard

Every Hawking model release must include:

- source model
- license
- tuning recipe
- dataset summary
- quant recipe
- eval ledger
- benchmark ledger
- hardware target
- exact command
- known failure modes
- checksum
- runtime minimum version

Model card sections:

```text
Model
Intended Use
Source
Tuning
Quantization
Evaluation
Speed
Memory
Energy
Serving Profile
Known Limits
Reproduce
Changelog
```

No model release without a ledger.

## Sprint L - Benchmark Infrastructure

Build a benchmark suite users can run.

Commands:

```bash
hawking bench suite --weights model.hqa --suite local-agent
hawking bench suite --weights model.hqa --suite long-context
hawking bench suite --weights model.hqa --suite throughput
hawking bench compare --a upstream.gguf --b hawking.hqa
```

Suites:

- single prompt latency
- multi-turn chat
- coding agent loop
- long context
- structured output
- high concurrency
- embeddings
- rerank
- spec decode
- prefix cache warm/cold

Outputs:

- JSONL
- Markdown
- HTML
- model-card snippet

## Sprint M - What May Become Newly Live

These were weak or dead on the smaller machine, but may deserve controlled
rechecks because the roofline and occupancy regime changed:

| Area | Why revisit | Gate |
|---|---|---|
| High-B Q4_K MMA | Bigger GPU may make different tile shapes win. | B=16/32 clean A/B. |
| Transformer int4 KV | More long-context workloads now realistic. | Quality + TTFT/memory win. |
| Full logits sampling kernels | Larger sampled workloads and model serving need it. | Sampled path end-to-end win. |
| Top-k/top-p GPU path | Avoid huge readback at high B. | Sampling benchmark. |
| Larger command buffers | Higher B/prefill may alter encode overhead. | CPU trace shows material overhead. |
| ANE side models | Main decode no, routers/reward maybe. | End-to-end not microbench. |
| MLX training loops | More memory makes local loops practical. | Faster iteration or better quality. |
| Multi-model residency | Previously too tight. | Router improves latency/quality trade. |

Do not resurrect dead levers casually. Each revisit needs a reason tied to the
new hardware regime.

## Sprint N - Decision Dashboard

Create a living dashboard:

```text
docs/reports/hawking_m3_ultra_dashboard.md
```

Sections:

- best single-stream model
- best aggregate serving model
- best quality/speed model
- best low-bit artifact
- best draft model
- best long-context setup
- best energy profile
- current blockers
- killed levers on M3 Ultra
- promoted defaults

This prevents the project from drowning in results.

## First 72 Hours Plan

Hour 0-4:

- inventory hardware
- install toolchain
- build release
- run smoke tests
- record exact environment

Hour 4-12:

- core Qwen/RWKV B matrix
- sustained thermal run
- first server matrix
- compare exact/fast/race profiles

Hour 12-24:

- long-context prefill/KV tests
- prefix cache warm/cold tests
- RWKV flatness tests
- high-B failure point discovery

Hour 24-48:

- begin high-B kernel analysis
- run first all-layer QAT feasibility smoke
- run teacher/student co-residency smoke
- run spec decode replay oracle at larger corpus size

Hour 48-72:

- write bring-up report
- update priorities
- choose first Hawking model release candidate
- choose first runtime feature to productize

## First 30 Days Plan

Week 1:

- hardware bring-up report
- serving matrix
- high-B kernel decision
- foundry pipeline dry-run

Week 2:

- detached KV store MVP
- RWKV state sessions MVP
- QAT/KD ladder first serious run
- HQA inspect prototype

Week 3:

- spec decode resident draft prototype
- model card + eval ledger template filled for first candidate
- structured output hardening
- model manager prototype

Week 4:

- first Hawking model release candidate
- first Hawking serving benchmark report
- rename compatibility layer
- public launch draft

## Success Criteria

The M3 Ultra sprint succeeds if it produces at least one result in each class:

1. Speed: a measured serving or decode improvement that is not just hardware
   scaling.
2. Model: a Hawking-tuned or Hawking-quantized artifact with eval ledger.
3. Runtime: a feature that changes what local serving can do, such as detached
   KV, state sessions, or resident spec decode.
4. Product: a user-facing command/model card/report that makes Hawking feel like
   something to download and run.
5. Research: a clear negative or positive result that guides future work.

## Final North Star

The strongest version of Hawking on M3 Ultra is not "Dismantle, but faster."

It is:

> A local Apple Silicon model foundry that ships tuned, compressed, stateful
> models with the runtime, cache system, and benchmark ledgers needed to make
> them obviously useful.

