# Bleeding-edge throughput, energy, and moat plan
*Authored 2026-06-05. Goal: turn dismantle from a generic llama.cpp competitor into a specialized Apple Silicon token factory that wins on tokens/sec, joules/token, latency, and workload-specific moats.*

## Why this exists

The latest head-to-head exposed the uncomfortable truth:

- llama.cpp already has continuous batching.
- dismantle's default single-stream path is not the fast profile.
- dismantle serve still pays full-logits costs for many requests that only need greedy token IDs.
- the current autotune profile chooses a few Q4_K kernels, but it does not tune runtime behavior.
- custom kernels exist, but the engine still emits a very large number of small dispatches per forward.

The answer is not to become "llama.cpp in Rust." That is a losing frame. The answer is to become more specialized and more workload-aware than llama.cpp can afford to be.

North-star thesis:

> Most production chat/code requests do not need full logits. They need the next token, low latency, low joules/token, and reuse of repeated state. dismantle should make the common path brutally cheap and keep the generic logits path as a fallback.

## Baseline facts to keep in view

From the 2026-06-05 bench and local probes:

| Path | Current observation |
|---|---:|
| llama.cpp single-stream decode | about 55 t/s |
| dismantle single-stream default from head-to-head | about 26 t/s |
| dismantle local default in-process short decode | about 31 t/s |
| dismantle local `--profile fast` short decode | about 43 t/s |
| llama-server B=8 aggregate | about 90 t/s |
| dismantle serve B=8 aggregate | about 51 t/s |

Important interpretation:

- `--kernel-profile` is not `--profile fast`. The kernel profile chooses kernel variants; the fast runtime levers are separate.
- `serve` enables several fast defaults, but the scheduler asks for full logits through `forward_multiseq_batched`, then samples on CPU.
- The multiseq path has a GPU Q4K full-vocab LM-head option, but it still materializes and reads back `B * vocab` logits.
- If Q4K LM head is not active, multiseq falls back to per-slot CPU full-vocab fp16 LM-head matmuls.
- A `gpu_prod` trace showed matmul-shaped kernels dominate GPU time. The default f16 LM head is a large tax; fast removes much of it, but the engine still has too many small dispatches.

## Metrics that matter

Every feature in this plan must report all relevant metrics, not just tokens/sec:

- `decode_tps`: completion tokens divided by decode time.
- `aggregate_tps`: total emitted tokens across concurrent slots divided by wall time.
- `ttft_ms`: time to first streamed token.
- `J/tok_GPU`: average GPU watts divided by decode tokens/sec.
- `J/tok_pkg`: average package watts divided by decode tokens/sec.
- `readback_bytes_per_token`: CPU-visible bytes read from GPU/shared buffers per emitted token.
- `dispatches_per_forward`: Metal dispatch count per model forward.
- `commits_per_token`: command-buffer fences per emitted token.
- `effective_tps`: user-visible tokens/sec including prefix-cache and accepted speculative tokens.
- `quality_delta`: only for non-bit-identical levers.
- `RSS_peak_mb`: peak memory footprint.

The target is not only "more t/s." A change that gives +5% t/s and -20% J/tok is a win. A change that gives +15% t/s but +40% J/tok is not automatically a win.

## Non-negotiable gates

1. Correctness before speed.
   - Bit-identical paths must pass golden token hashes.
   - Non-bit-identical quant/profile paths need a declared quality gate before they can ship.

2. Paired bench first, clean-room absolute second.
   - Paired A/B can run while the desktop app is alive.
   - Absolute claims require a clean-room run with other heavy GPU apps closed.

3. No more blind Colab training.
   - Learned speculation does not get cloud compute until an offline replay oracle proves it can beat cheap n-gram/user-history speculation.
   - Cloud is not a development environment for the core runtime. It is only a bounded execution backend for a few proven jobs after local optimization has identified the exact workload, dataset, budget, and stop criteria.

4. Fast lanes must be explicit.
   - Generic logits support remains, but the server must route greedy/token-only requests away from full logits.

5. Every optimization must state whether it improves:
   - raw single-stream speed,
   - aggregate serving speed,
   - energy efficiency,
   - latency,
   - memory footprint,
   - workload moat.

## Target table

Targets for Qwen2.5-3B-Instruct Q4_K_M on M3 Pro 18 GB:

| Stage | Single-stream | Serve B=1 | Serve B=8 aggregate | Energy target |
|---|---:|---:|---:|---|
| Current measured region | 26-43 t/s | about 20 t/s | about 51 t/s | baseline |
| Phase 1 target | 50-60 t/s | 40-55 t/s | 100-130 t/s | -20% J/tok |
| Phase 2 target | 60-75 t/s | 55-70 t/s | 130-170 t/s | -30% J/tok |
| Phase 3 stretch | 75-90 t/s | 65-85 t/s | 170-220 t/s | -40% J/tok |
| Moat workloads | not the key metric | not the key metric | 2-3x llama effective | lower total joules per task |

## Track 0 - Measurement and bench honesty

### 0.1 Add a full report-card bench

Create a new bench command or extend `tools/bench/llama_head_to_head.sh` to print these lanes:

- `dismantle-default-generate`
- `dismantle-fast-generate`
- `dismantle-serve-full-logits`
- `dismantle-serve-greedy-token-only` once implemented
- `llama-cli`
- `llama-server`

Required output:

- t/s
- J/tok GPU
- J/tok package
- wall seconds
- prompt tokens/sec
- readback bytes/token
- feature flags actually active
- profile file actually loaded
- model hash
- git SHA and local modification count

Acceptance gate:

- One command produces a shareable table.
- The table cannot confuse `--kernel-profile` with global `--profile fast`.
- The table prints whether full logits or token-only path was used.

### 0.2 Add runtime counters

Add per-request stats:

- `gpu_readback_bytes`
- `logits_materialized_rows`
- `logits_materialized_vocab`
- `token_only_path_used`
- `lm_head_path`: `cpu-f16`, `gpu-q4k-full`, `gpu-q4k-pruned`, `gpu-q4k-pruned-argmax`
- `dispatches_per_forward`
- `tcb_trace_mode`
- `q4k_predec_cache_active`
- `fast_profile_levers_active`

Acceptance gate:

- `/metrics` exposes aggregate counters.
- `generate` and `serve` can emit one JSON stats block for a request.

### 0.3 Energy harness

Extend the macmon sampling harness to produce:

- average GPU watts during decode
- average package watts during decode
- idle-adjusted GPU watts if possible
- J/tok for decode
- J/request including prefill
- J/user-visible-token with prefix/spec credits

Acceptance gate:

- The harness can compare two commands and report paired deltas.
- Output includes "higher t/s but worse J/tok" warnings.

## Track 1 - Greedy token-only serving

This is the first attack. It is concrete, local, and directly addresses the bad serve numbers.

### 1.1 Add an engine trait method for token-only multiseq

Add a new trait method:

```rust
fn forward_multiseq_greedy_tokens(
    &mut self,
    tokens: &[u32],
    positions: &[usize],
    regions: &[usize],
) -> Result<Vec<u32>>;
```

Default implementation:

- call existing `forward_multiseq_batched`
- CPU argmax/sample as today

QwenDense implementation:

- run multiseq stack
- append LM-head dispatch in same command buffer
- run GPU argmax
- read back only `B * sizeof(u32)` token IDs

Acceptance gate:

- For `temperature=0`, output tokens match current serve path exactly.
- For B=1,2,4,8, token IDs match solo generate for a fixed prompt set.
- Readback per decode step drops from `B * vocab * sizeof(f32)` to `B * sizeof(u32)` on the greedy lane.

### 1.2 Add batched GPU argmax

Current `sample_argmax_f32` returns one token. Add:

```text
sample_argmax_f32_batched
```

Inputs:

- logits buffer: `(B, vocab)` contiguous
- token buffer: `(B)` u32
- `vocab`
- `B`

Output:

- one argmax token per slot

Acceptance gate:

- CPU argmax parity for random logits and real LM-head logits.
- Tie-breaking matches existing CPU and single-token GPU argmax.

### 1.3 Pruned LM-head token-only path

For greedy serving, prefer:

```text
stack -> pruned Q4K LM head -> batched argmax -> remap -> token IDs
```

Do not materialize full-vocab logits unless the request needs sampling.

Modes:

- first-N prune: direct token identity
- corpus whitelist: apply remap
- no prune: full-vocab Q4K token-only fallback

Acceptance gate:

- First-N prune matches existing fast profile behavior.
- Corpus remap path has a dedicated parity test.
- `serve` B=1 should approach in-process fast decode.

Expected impact:

- B=1 serve: about 20 t/s -> 40-55 t/s
- B=8 aggregate: about 51 t/s -> 100-130 t/s
- J/tok: significant drop by removing CPU full-logit path and huge readbacks

### 1.4 Scheduler lane classification

At request admission, classify:

- `GreedyTokenOnly`: temperature 0, top_p irrelevant, repetition penalty 1, no logprobs requested
- `SampledLogits`: temperature > 0 or top_p/top_k needed
- `Logprobs`: caller explicitly asks for logprobs/full logits
- `Speculative`: eligible for draft+verify lane

Only `GreedyTokenOnly` enters the token-only multiseq path.

Acceptance gate:

- Non-greedy requests remain behavior-compatible.
- Greedy OpenAI `/v1/completions` and `/v1/chat/completions` use token-only path.

## Track 2 - Fast profile and real runtime autotune

### 2.1 Fix profile naming confusion

The CLI currently has:

- global `--profile fast`
- autotune subcommand `--profile m3-pro-18gb`
- `--kernel-profile path.json`

This is easy to misunderstand.

Plan:

- Rename autotune's hardware string in help/docs to `--hardware-profile`.
- Keep old alias for compatibility if needed.
- Make bench output print active runtime profile separately from kernel profile.

Acceptance gate:

- No more warning like `unknown --profile 'm3-pro-18gb'` for a normal autotune command.
- Bench report clearly distinguishes runtime levers from kernel JSON.

### 2.2 Add `--profile race` and `--profile efficient`

Profiles:

- `fast`: current quality-trade bundle.
- `race`: maximum t/s, allowed to use quality-trade levers after quality gate.
- `efficient`: minimize J/tok under a target t/s floor.
- `exact`: bit-identical conservative path.

Potential profile knobs:

- vocab prune size
- Q4K LM head
- Q4K FFN-down
- Q4K predec
- f16 scales
- f16 KV
- batch gather window
- token-only serving
- sidecar format

Acceptance gate:

- Each profile prints active levers.
- Each profile has a quality and J/tok statement.

### 2.3 Runtime autotune that actually tunes runtime

Current autotune only selects some kernel schedules. Replace or extend it with end-to-end candidates.

Autotune matrix:

- B: 1,2,4,8
- mode: generate, serve, token-only
- vocab prune: none, 16k, 24k, 32k, 48k
- LM head: f16, Q4K full, Q4K pruned
- ffn_down: native, Q4K
- scale table: f32 predec, f16 predec
- KV: f32, f16
- batch gather window: 0 ms, 2 ms, 5 ms, 10 ms
- kernel schedule per shape

Scoring:

```text
score = weighted_tps_gain - quality_penalty - jtok_penalty - rss_penalty - ttft_penalty
```

Acceptance gate:

- Autotune emits a profile that changes runtime behavior, not just kernel names.
- Profile records evidence: model hash, device, shader hash, candidate metrics, quality gate result.

## Track 3 - Kernel and dispatch reduction

The trace says the stack still emits too many small kernels. One command buffer is good, but it is not enough.

### 3.1 Dispatch count budget

Set hard targets:

| Stage | Dispatches per Qwen forward |
|---|---:|
| Current rough region | hundreds |
| Phase 1 budget | less than 350 |
| Phase 2 budget | less than 220 |
| Stretch | less than 150 |

Every fusion pass must report dispatch count and J/tok.

### 3.2 Batched embed + layer-0 norm fusion

Fuse:

```text
embed_lookup -> layer0_rmsnorm
```

For multiseq, do this for all B slots in one dispatch.

Acceptance gate:

- Bit-identical output for layer-0 normalized activation.
- Small but measurable dispatch reduction.

### 3.3 Q/K/V projection dispatch strategy

Options:

1. Concurrent command encoder group for Q, K, V projections.
2. Fused QKV projection where tensor layout allows.
3. Sidecar-packed QKV weights to support one custom dispatch.

Preferred route:

- start with concurrent encoder group because it is less invasive.
- sidecar QKV pack only after measured.

Acceptance gate:

- Token hashes unchanged.
- `gpu_prod` trace shows lower wall/GPU idle gaps or better throughput.

### 3.4 Bias + RoPE + KV scatter fusion

Current path has separate bias, RoPE, and KV append/scatter kernels.

Investigate fused kernels:

- `add_bias_rope_q`
- `add_bias_rope_k_scatter`
- `v_bias_scatter`

This is attractive because Qwen has Q/K/V bias and RoPE every layer.

Acceptance gate:

- Bit-identical Q/K after RoPE.
- Bit-identical KV cache contents.
- Fewer dispatches and lower J/tok.

### 3.5 Fused SwiGLU and FFN-down preparation

Current:

```text
gate -> up -> silu_mul -> ffn_down
```

Already has a fused gate+up predec pair. Next opportunities:

- fuse `silu_mul` with activation quant/prep for W4A8 or Q4K-sidecar path
- avoid writing large intermediate activation when immediately consumed
- explore tiled FFN-down that consumes activation tiles directly

Acceptance gate:

- Non-fused and fused logits agree within declared precision.
- J/tok improves, not just t/s.

### 3.6 Layer micrograph

Longer-term: express a Qwen layer as a tiny static graph:

```text
attn_norm
q/k/v
bias/rope/kv
mha
o_proj
resid+ffn_norm
gate/up/swiglu/down
resid+next_norm
```

Then generate a dispatch plan per profile:

- exact path
- fast path
- energy path
- multiseq token-only path

Acceptance gate:

- No dynamic per-token planning overhead.
- Plan selected once at load/autotune time.

## Track 4 - Custom sidecar format

GGUF compatibility gets users in. A custom sidecar makes dismantle fast.

### 4.1 Define `.dismantle` sidecar v1

Sidecar contains:

- reordered Q4_K blocks for Metal-friendly access
- predecoded scales, f32 or f16
- pruned LM-head Q4K
- optional corpus whitelist/remap
- optional Q4K FFN-down requant
- tensor offset table
- quality metadata
- source GGUF hash
- tokenizer hash
- shader/profile compatibility hash

Acceptance gate:

- Sidecar is optional.
- If sidecar is missing, GGUF path still works.
- If sidecar hash mismatches source GGUF, fail loudly.

### 4.2 Bake command

Add:

```sh
dismantle bake-sidecar \
  --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --out models/qwen2.5-3b-instruct-q4_k_m.dismantle \
  --profile race
```

Bake outputs:

- sidecar file
- quality report
- expected speed/energy profile
- active levers

Acceptance gate:

- One command builds sidecar.
- Load path automatically detects matching sidecar unless disabled.

### 4.3 Mixed quant tier map

Use per-layer/per-tensor decisions:

- keep sensitive tensors Q4/Q6
- move insensitive tensors to Q3 or narrower sidecar representation
- keep LM head pruned/Q4K unless full logits requested

Quality gates:

- token hash only for bit-identical paths
- logit cosine for non-identical quant
- small eval suite for chat/code
- top-1 agreement rate

Energy goal:

- fewer weight bytes per token
- lower memory bandwidth
- lower J/tok

## Track 5 - Prefix, cache, and session moats

This is where dismantle can beat llama in actual chat workloads even if raw GEMV remains close.

### 5.1 Prompt-prefix trie in serve

Maintain an in-memory trie of token prefixes for active and recent sessions.

Use it for:

- exact KV reuse
- shared system prompt prefill once
- admission grouping by common prefix
- cache hit metrics

Acceptance gate:

- Cache hit outputs are bit-identical to cold prefill.
- Metrics show `prefill_tokens_skipped`.

### 5.2 Cross-request system prompt bank

Most chat apps reuse a system prompt. Make this explicit:

```sh
dismantle serve --system-prefix-cache auto
```

Server detects repeated first N prompt tokens and pins their KV.

Acceptance gate:

- B=8 identical-system-prompt workload shows large TTFT and J/request drop.
- No stale KV after model/profile/tokenizer mismatch.

### 5.3 KV compression lane

Investigate:

- f16 KV
- Q8 KV
- per-head adaptive KV precision

Decision metric:

- long-context J/tok
- long-context t/s
- quality drift

Acceptance gate:

- Long-context parity/quality gate.
- J/tok lower at 1k, 2k, 4k context.

### 5.4 Cache-aware scheduler

Scheduler should consider:

- greedy token-only lane
- full-logits lane
- shared-prefix lane
- speculative lane
- latency budget
- energy budget

Policy:

- Do not mix full-logits requests with greedy token-only requests unless needed.
- Prefer filling a token-only batch for 2-5 ms if it improves J/tok materially.
- Avoid delaying latency-sensitive single requests beyond target TTFT.

Acceptance gate:

- Bench with `BATCH_SIZES=1 2 4 8`, plus mixed greedy/sampling workload.
- Report throughput and p50/p95 TTFT.

## Track 6 - Speculative decode reboot, without wasting cloud compute

The rule:

> No trained speculative head gets Colab/HF compute until offline replay proves it can beat cheap local draft sources.

Cloud policy:

- All Rust, Metal, scheduler, serving, sidecar, profiling, energy, prefix-cache, and runtime-autotune work happens locally on the target Apple Silicon machine.
- Cloud is considered only after the local machine has already optimized the core path and produced a narrow, reproducible job that cannot be done reasonably locally.
- A cloud job must have a fixed input dataset, command, hardware target, maximum budget, expected metric, and stop condition before it starts.
- Cloud results are advisory until replayed or validated locally on the M3 Pro, because the winning bottlenecks are Metal/runtime-specific.

Cloud-worthy only after local gates:

- learned speculative-head training, if the offline replay oracle beats n-gram/user-history baselines by the required margin
- larger quality/eval sweeps for non-bit-identical quant or sidecar profiles, after local smoke gates pass
- larger dataset preprocessing for corpus vocab pruning or quant-tier maps, if local preprocessing becomes the bottleneck
- optional long-run regression sweeps that do not depend on Metal timing

Not cloud-worthy:

- Metal kernel optimization
- command-buffer and dispatch-count work
- QwenDense runtime changes
- token-only serving
- J/tok measurement
- scheduler policy tuning
- local prefix/session cache behavior
- Apple Silicon sidecar layout validation

### 6.1 Establish cheap baselines

Draft sources:

- per-user n-gram
- session n-gram
- repo/code n-gram
- prompt-local suffix trie
- static common-token fallback

Measure:

- accepted tokens per verification forward
- rejection rate
- added overhead
- net t/s
- net J/tok

Acceptance gate:

- Keep only draft sources with positive paired speed and J/tok deltas.

### 6.2 Offline replay oracle

Before training anything, capture verifier traces:

- prompt tokens
- true next tokens
- residual snapshots only if cheap and locally captured
- layer chosen
- context metadata

Replay candidate draft policies offline.

A learned model may advance only if replay shows:

- acceptance at depth 1 beats n-gram baseline by a meaningful margin
- acceptance at depths 2-4 is high enough to amortize verify
- proposed tokens can be generated cheaper than a verifier forward
- expected J/tok improves

Suggested initial thresholds:

- depth-1 acceptance above 55% on target workload
- mean accepted tokens per verify cycle above 1.35
- no more than 5% overhead on cycles with zero acceptance
- projected net t/s gain above 20%
- projected J/tok reduction above 10%

If it misses those, do not train.

### 6.3 Spec governor

Spec decode should auto-disable when it is losing.

Per session, track:

- rolling acceptance
- rolling tokens per verify forward
- rolling J/tok proxy
- rejection bursts

Policy:

- enable for code/repetitive workloads
- disable for creative/random chat
- disable when acceptance drops below threshold
- re-enable after pattern stability returns

Acceptance gate:

- Spec never hurts more than a small configured bound.
- Bench prints accepted/rejected and net speed/J impact.

### 6.4 Learned head only after replay passes

If replay passes, train small and local-first:

- start with tiny linear/MLP head
- train on already-captured traces
- use CPU/MPS local first if possible
- Colab only for a frozen experiment with a hard budget and stop criteria

Cloud stop criteria:

- if validation acceptance does not beat n-gram baseline by checkpoint 1, stop
- if depth-2+ acceptance is poor, stop
- if proposal cost exceeds budget, stop

Acceptance gate:

- A trained head must beat n-gram in paired end-to-end generation, not just offline metrics.

## Track 7 - Energy-first features

Speed and energy are correlated but not identical. Add lanes that explicitly optimize J/tok.

### 7.1 Energy-aware batching

Batching can lower J/tok even when it slightly increases TTFT.

Scheduler knobs:

- `--energy-mode off|balanced|efficient`
- gather window
- max latency budget
- target batch occupancy

Policy:

- `off`: lowest latency
- `balanced`: wait 2-5 ms for batch fill
- `efficient`: wait longer within p95 TTFT budget

Acceptance gate:

- Report t/s, TTFT, J/tok.
- Efficient mode must lower J/tok without unacceptable p95 latency.

### 7.2 Avoid CPU/GPU ping-pong

Kill paths that read large buffers to CPU:

- full logits on greedy
- x_norm readbacks except diagnostics
- CPU LM-head fallback in serve
- per-token allocation of logits vectors

Acceptance gate:

- Greedy serving readback is token IDs only.
- Diagnostics can still opt into full readback.

### 7.3 Race-to-idle vs steady-state

Measure two strategies:

- maximum speed then idle
- energy-smoothed batching

For short requests, race-to-idle may win. For continuous serving, steady batching may win.

Acceptance gate:

- Bench harness reports both request-level joules and steady-state J/tok.

### 7.4 Memory residency and cache pressure

Existing residency hooks should become profile-aware.

Track:

- resident weight mmap
- resident sidecar
- resident decode arena
- optional nonresident diagnostic buffers

Acceptance gate:

- No RSS explosion.
- Residency improves p95 latency or J/tok in paired bench.

## Track 8 - Product/API moats in the single binary

### 8.1 Lane-aware OpenAI-compatible server

Expose:

- `/v1/completions`
- `/v1/chat/completions`
- `/metrics`
- `/healthz`

Internally route to:

- greedy token-only
- sampled logits
- speculative
- prefix-cache
- diagnostics

Acceptance gate:

- OpenAI-compatible clients still work.
- Metrics show lane mix and win sources.

### 8.2 Dismantle-native low-overhead endpoint

Add optional endpoint:

```text
POST /v1/dismantle/tokens
```

Purpose:

- token-only streaming
- no OpenAI JSON chunk bloat
- ideal for local apps and benchmarks

Acceptance gate:

- Same generated tokens as OpenAI path.
- Lower CPU overhead and lower wall time at high B.

### 8.3 Live self-report

Add:

```sh
dismantle serve --explain-performance
```

Startup prints:

- model
- active profile
- fast levers
- sidecar status
- expected lane support
- whether token-only path is available
- whether full logits will be expensive

This prevents future confusion.

## Track 9 - Competitive moats beyond speed

### 9.1 One-binary reproducibility

Make a single binary that can:

- fetch or verify model hash
- bake sidecar
- autotune
- serve
- benchmark
- print energy report

This is a moat because users can reproduce performance without a pile of scripts.

### 9.2 Exactness modes

Modes:

- `exact`: bit-identical conservative path
- `fast`: measured quality-trade path
- `race`: maximum throughput
- `efficient`: lowest J/tok

Each mode must print its contract.

### 9.3 Workload packs

Profiles for:

- code completion
- chat with shared system prompt
- batch summarization
- local agent loops
- JSON/tool-call style outputs

Each workload pack configures:

- scheduler
- prefix cache
- speculative source
- sampling lane
- energy mode

## Recommended build order

### Phase A - Stop doing unnecessary work

1. Add bench report-card lanes.
2. Add stats counters for readback/logits/token-only.
3. Implement `forward_multiseq_greedy_tokens`.
4. Add batched GPU argmax.
5. Route greedy serve to token-only path.

Expected result:

- serve B=1 moves toward in-process fast decode.
- B=8 aggregate should clear llama-server or get close.
- J/tok drops because full-logit readback and CPU sampling disappear.

### Phase B - Make fast/race/efficient profiles real

1. Fix profile naming.
2. Add `race` and `efficient`.
3. Extend autotune to runtime candidates.
4. Make bench print active runtime levers.

Expected result:

- no more accidental conservative benchmarks.
- users can choose speed vs exactness vs energy.

### Phase C - Reduce dispatch count

1. Embed+norm fusion.
2. Q/K/V dispatch strategy.
3. bias+rope+scatter fusion.
4. SwiGLU/FFN-down preparation fusion.
5. Layer micrograph planning.

Expected result:

- single-stream raw decode moves from "near llama" to "beating llama."
- J/tok improves through less overhead and fewer memory round trips.

### Phase D - Sidecar and quant moat

1. Define sidecar v1.
2. Bake command.
3. Pruned LM-head sidecar.
4. Mixed quant tier map.
5. Sidecar-aware autotune.

Expected result:

- faster cold and hot decode.
- lower bytes/token.
- model-specific advantage llama.cpp cannot match without abandoning generic GGUF behavior.

### Phase E - Prefix/spec/session moat

1. Prefix trie.
2. system prompt KV bank.
3. n-gram/user-history draft governor.
4. offline replay oracle.
5. learned head only if replay passes.

Expected result:

- effective throughput beats llama by large margins on real chat/code workloads.
- no more blind training spend.

## Immediate first issue to file

Title:

```text
Implement greedy token-only multiseq serving lane
```

Scope:

- add `Engine::forward_multiseq_greedy_tokens`
- implement QwenDense token-only path
- add batched GPU argmax kernel
- route `temperature=0` serve requests through token-only path
- add stats counters for readback bytes and lane used

Acceptance:

- B=1,2,4,8 parity against current full-logits serve
- readback drops to `B * 4` bytes per decode step
- no behavior change for non-greedy requests
- bench shows B=1 serve and B=8 aggregate deltas
- report J/tok delta

Expected win:

- biggest near-term serving improvement.
- strongest energy improvement.
- simplest explanation to users: "we do not compute or read logits unless you ask for them."

## Kill criteria

Kill or hold a lever if:

- it is not bit-identical but has no quality gate.
- it increases J/tok materially without a clear t/s or latency reason.
- it helps only a synthetic benchmark and hurts serve.
- it requires cloud training before local replay proves it.
- it adds a sidecar/profile that cannot be validated against source GGUF hash.
- it makes the generic fallback worse.

## Final thesis

dismantle can beat llama.cpp by specializing in three ways:

1. Token-only fast path for the common case.
2. Stateful workload reuse: prefix, session, and cheap speculation.
3. Custom sidecar/runtime profiles that minimize bytes moved per token.

The raw GEMV fight is hard. The unnecessary-work fight is winnable immediately. The workload-specific fight is where the moat lives.
