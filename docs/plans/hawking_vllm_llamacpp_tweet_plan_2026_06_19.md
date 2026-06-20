# Dismantle/Hawking Serving Plan - vLLM vs llama.cpp Long-Context Tweet

Date: 2026-06-19

Trigger:

> Why is llama.cpp so much slower than vLLM at long context multi-turn tasks,
> and at high concurrency?

This document is the one-shot response plan. It is not just a draft tweet. It is
the serving roadmap this tweet points at: public diagnosis, benchmark plan,
competitor pain radar, implementation sequence, and the Hawking release
direction.

Dismantle is the current production shell. Hawking is the future release
identity and ideology: Apple-first, local-first, small as possible, fast as
possible, and strong enough that people can say "I used Hawking for this" rather
than "I used a dependency buried inside a project."

## Short Answer

vLLM tends to beat llama.cpp on long-context multi-turn and high-concurrency
serving because vLLM was built around serving as a runtime problem: KV memory is
managed in blocks, requests are continuously batched, long prefills can be
chunked, prefix reuse is a first-class performance path, and scheduling is
cache-aware.

llama.cpp is extremely strong as a local single-user inference engine and has
real server features, but its center of gravity is different. In workloads with
many concurrent conversations, long shared prefixes, repeated agent loops, and
cache churn, the runtime architecture matters as much as kernel speed.

The Dismantle/Hawking answer is not to clone vLLM. It is to take the serving
lesson seriously for Apple Silicon:

- prompt state should be cached and shared, not rebuilt as disposable text;
- long prefills should not starve short requests;
- resident memory should be visible, budgeted, and Apple UMA-aware;
- RWKV/SSM state should become a first-class advantage because it does not grow
  like transformer KV;
- benchmarks should measure P95/P99, cache reuse, and multi-turn continuity, not
  only single-stream tokens/sec.

## Public Reply Draft

Use this only after we have measurements. Until then, it is a target claim, not
a claim to publish.

> vLLM wins many high-concurrency transformer serving cases because it treats KV
> cache, batching, and scheduling as the product, not as wrapper code around a
> model forward pass. llama.cpp is excellent local inference infrastructure, but
> long-context multi-turn workloads expose the cost of prompt rebuilds, cache
> churn, and less server-native scheduling.
>
> Dismantle is taking the vLLM lesson into an Apple-local direction: prefix/state
> reuse, chunked prefill, cache-aware admission, and measured P95 latency on
> real agent/chat workloads. Hawking is where that becomes a named runtime/model
> line: small tuned models, STRAND/HQA quantization, Metal-first serving, and
> RWKV/SSM state that stays compact across long sessions.

Shorter version:

> vLLM is faster here because serving speed is not just kernels. It is KV cache
> architecture, continuous batching, chunked prefill, and scheduler policy.
> Dismantle/Hawking should copy that lesson for Apple Silicon, then go narrower:
> local tuned models, visible UMA memory policy, and RWKV/SSM state reuse.

## Naming And Positioning

Use Dismantle in code, reports, and commits until the rename is deliberate.
Use Hawking as the future-facing model/runtime vision.

The naming line:

| Layer | Name now | Future identity | Role |
|---|---|---|---|
| Repository | Dismantle | Hawking possible | Current production shell. |
| Runtime research | Dismantle serving | Hawking runtime | Apple-first inference engine work. |
| Quantization | STRAND/HQA/TQ | Hawking quantized releases | Compression and tuned model packaging. |
| Model artifacts | Dismantle experimental | Hawking model line | Downloadable tuned/quantized targets. |
| Public notes | Dismantle research | Hawking papers/releases | Evidence trail and launch surface. |

The message should be:

> Dismantle is the lab. Hawking is the artifact that eventually leaves the lab.

## What vLLM Does Right

This is the part to respect. Do not flatten it into "vLLM has batching."

| vLLM idea | Why it matters | Dismantle/Hawking translation |
|---|---|---|
| Paged/block KV cache | Many dynamic sequences do not require huge contiguous KV allocations. | Detached KV block/span store for transformer models. |
| Continuous batching | Active requests share model forwards instead of each request owning the engine. | Make `BatchDriver` production-grade and measured. |
| Prefix caching | Shared system prompts and repeated prefixes skip redundant prefill. | Upgrade `SystemPromptKvBank` from routing hint to resident state store. |
| Chunked prefill | Long prompts are processed in chunks so they do not monopolize the engine. | Add prefill chunks with decode interleaving. |
| Cache-aware scheduling | Scheduler protects cache hits and avoids avoidable recompute. | Prefix-grouped admission with age/fairness score. |
| High-throughput serving metrics | The engine optimizes aggregate throughput and tail latency. | Benchmark TTFT P50/P95/P99, per-user t/s, aggregate t/s, and queue wait. |
| Disaggregated prefill/decode | Large deployments can scale prefill and decode independently. | Future only; local single-machine first. |

## Where llama.cpp Is Strong

Do not make the lazy argument. llama.cpp is not "bad serving."

llama.cpp has:

- broad model/format reach;
- GGUF as a major distribution standard;
- strong local CPU/GPU portability;
- Metal support;
- server mode;
- slots;
- prompt caching;
- speculative decoding;
- continuous batching and multi-user paths.

The opportunity is not "llama.cpp lacks features." The opportunity is:

- long-context multi-turn workloads are especially sensitive to cache/state
  invalidation and rebuild cost;
- local Apple machines need UMA-aware memory policy, not server-GPU assumptions;
- agent loops repeat large prefixes and small deltas constantly;
- tail latency matters more than ideal single-user throughput;
- RWKV/SSM state opens a different long-session path than transformer KV.

## Apple-First Wedge

The project should focus on Apple first because that is where we can be
measured, obsessive, and differentiated.

Apple-first does not mean small ambition. It means narrower truth:

- one hardware family;
- one memory model;
- one kernel target;
- one user community that wants local AI to feel native;
- one benchmark story we can repeat every day.

The Apple-specific thesis:

| Apple constraint | Product meaning | Hawking opportunity |
|---|---|---|
| UMA RAM is model memory and KV memory | "Offload" is not free; every cache byte competes with model/context. | Visible memory ledger and cache budget planner. |
| Metal performance differs by workload shape | Single-token decode, prefill, and spec decode need separate gates. | Metal-specific benchmark matrix. |
| Local users care about interactivity | P95 TTFT matters more than max batch throughput. | Tail-latency-first scheduler. |
| 16GB machines hit cliffs early | Need small models and tight quantization. | STRAND/HQA/Hawking tuned artifacts. |
| 96GB machines unlock real local labs | Large context, concurrent agents, and richer evals become practical. | "Hawking 96GB frontier" benchmark tier. |

## 96GB Tier

The future M3 Ultra 96GB machine should not just make development faster. It
should become the high-water Apple lab target.

Unlocks:

- larger transformer baselines for honest head-to-heads;
- long-context stress at 64k/128k without constant memory compromise;
- multi-agent/concurrency benches on one local machine;
- bigger KV cache experiments;
- more aggressive prefix stores;
- local model bake/quant/eval loops;
- side-by-side Dismantle vs llama.cpp vs MLX/MLC vs vLLM-compatible backends;
- faster revisions on tuned/quantized Hawking artifacts.

Do not compare raw numbers from 16GB M3 Pro to 96GB M3 Ultra as if they are the
same benchmark tier. Rebench everything with hardware labels.

## Competitor Pain Radar

We should deliberately mine public complaints and convert them into benchmark
cases. This turns social noise into product truth.

Use public APIs and public pages only. Do not scrape private content. Do not
store private user data. X/Twitter can be handled through manual saved links or
official API paths later; GitHub and Hugging Face are enough for the first pass.

Primary sources:

- llama.cpp GitHub issues/discussions;
- vLLM GitHub issues/discussions;
- vLLM-Metal related threads;
- Ollama issues;
- MLC/MLX issues;
- Transformers Apple/MPS issues;
- Hugging Face model discussions for quant/load/runtime pain;
- curated links from Reddit/HN/X only when they become reproducible.

Seed public issues already worth tracking:

| Engine | Pain | URL |
|---|---|---|
| llama.cpp | Prompt cache reprocessing with long/hybrid turns | https://github.com/ggml-org/llama.cpp/issues/19794 |
| llama.cpp | Multi-turn prompt-cache state drift | https://github.com/ggml-org/llama.cpp/issues/21681 |
| llama.cpp | Context truncation/rebuild pain | https://github.com/ggml-org/llama.cpp/issues/19838 |
| llama.cpp | Metal speculative decode regression | https://github.com/ggml-org/llama.cpp/issues/23752 |
| vLLM | Agent context vs KV cache mismatch | https://github.com/vllm-project/vllm/issues/37168 |
| vLLM | Context/sequence parallelism RFC | https://github.com/vllm-project/vllm/issues/22693 |
| vLLM | KV connector/capacity/concurrency issue | https://github.com/vllm-project/vllm/issues/42024 |
| vLLM | Apple/MPS support gap | https://github.com/vllm-project/vllm/issues/1441 |
| vLLM | Metal support thread | https://github.com/vllm-project/vllm/issues/19073 |

Pain classes:

| Pain class | User symptom | Benchmark shape | Hawking response |
|---|---|---|---|
| Long-context reprocessing | "Second turn feels like the whole prompt ran again." | Same long prefix, small delta, repeated turns. | Detached prefix KV/state store. |
| Cache invalidation opacity | "I expected a cache hit but got a full rebuild." | Prefix match/mismatch cases with reason codes. | Token-exact validation and miss metrics. |
| High-concurrency collapse | "Throughput/tail latency falls apart with users." | B=1/2/4/8/16 streams. | Continuous batching and admission policy. |
| Long prefill starvation | "One huge request blocks short chats." | One 64k prompt plus many short prompts. | Chunked prefill and fairness. |
| Metal backend gap | "Feature exists but is slow on Apple." | Metal-only spec/prefill/decode gates. | Apple-first kernels and defaults. |
| Spec decode net loss | "Drafting makes it slower." | Draft accept-rate and net t/s matrix. | Only enable spec when net-positive on Apple. |
| Memory cliff | "Why did context/concurrency shrink?" | KV resident bytes and eviction cases. | Memory ledger and budget planner. |
| Install/runtime mismatch | "Docs assume server GPU paths." | Fresh-machine setup test. | One-command Apple path. |

Pain radar deliverables:

- `tools/research/pain_radar.py`
- `docs/research/pain_radar/ledger.jsonl`
- `docs/research/pain_radar/clusters.md`
- `tools/bench/workloads/from_pain_radar.py`
- `docs/reports/apple_serving_pain_fixes.md`

Ledger schema:

```json
{
  "url": "https://github.com/...",
  "source": "github",
  "engine": "llama.cpp",
  "title": "Prompt cache reprocessing...",
  "status": "open|closed|not_planned|unknown",
  "created_at": "2026-...",
  "updated_at": "2026-...",
  "pain_class": "long_context_reprocessing",
  "hardware": "apple|server_gpu|cpu|unknown",
  "model": "qwen|llama|rwkv|unknown",
  "reproducible": "yes|maybe|no",
  "benchmark_candidate": true,
  "dismantle_feature": "detached_prefix_state",
  "notes": "..."
}
```

Scoring rubric:

| Signal | Weight | Why |
|---|---:|---|
| Apple relevance | 5 | This is the wedge. |
| Reproducibility | 5 | Pain must become a test. |
| Recency | 4 | Fresh issues reflect current product reality. |
| Reactions/comments | 3 | More users means stronger roadmap signal. |
| Runtime leverage | 5 | Prefer architecture wins over config wins. |
| Publishability | 4 | Good fixes become public research notes. |
| Model-artifact tie-in | 3 | Best runtime wins should connect to Hawking models. |

The loop:

1. Mine public pain.
2. Cluster pain into workload shapes.
3. Reproduce on Apple Silicon.
4. Record Dismantle baseline.
5. Implement the smallest fix that moves the metric.
6. Rebench.
7. Publish a note.
8. Promote durable wins into Hawking release claims.

## Current Codebase Starting Point

Already present:

- `BatchDriver` and slot scheduler;
- `--max-batch-size`;
- greedy token-only lane;
- prefix index for active slots;
- `SystemPromptKvBank` as a serve-lifetime routing hint;
- `--batch-policy prefix-grouped`;
- `--workload chat-shared-prompt`;
- `/metrics` counters for active slots, queued requests, readback bytes, prefix
  reuse;
- RWKV multiseq path and batched capture machinery;
- Apple/Metal codepath as the current focus after non-Apple backend-facing
  surfaces were cut.

Not enough yet:

- detached KV storage that survives without a live source slot;
- resident RWKV/SSM state snapshots keyed by transcript prefix;
- chunked prefill;
- stop-string handling in the batch scheduler;
- cancellation cleanup for dropped SSE clients;
- high-concurrency benchmark harness with P50/P95/P99;
- long-context churn tests;
- public report comparing current Dismantle, llama.cpp, and vLLM-style
  behavior.

## Benchmark First

Do not write the big runtime changes before a benchmark can show the pain.

Minimum benchmark harness:

- `tools/bench/serve_concurrency_matrix.py`
- engine adapters for Dismantle and llama.cpp first;
- vLLM adapter where a compatible backend/hardware path exists;
- transcript generator for shared-prefix agent sessions;
- JSONL result output;
- Markdown report generator;
- optional pain-radar workload import.

Core command target:

```bash
python tools/bench/serve_concurrency_matrix.py \
  --engine dismantle \
  --concurrency 1,2,4,8,16 \
  --workload shared-agent \
  --prompt-tokens 8192,32768,65536 \
  --decode-tokens 128 \
  --out docs/reports/serve_matrix/dismantle_current.jsonl
```

Required workloads:

| Workload | Shape | Why |
|---|---|---|
| Shared system prompt chat | One fixed system prompt, many user turns. | Prefix reuse. |
| Agent loop | Repeated repo/file preamble plus small task delta. | Real coding-assistant shape. |
| Long prompt burst | 8-32 requests with 8k-64k prompts. | Chunked prefill/fairness. |
| Mixed latency | One long request plus many short requests. | Starvation and tail latency. |
| High concurrency decode | B=1/2/4/8/16 active streams. | Aggregate throughput and degradation. |
| Slot churn | Users connect/disconnect mid-generation. | Cleanup and cache safety. |
| Cache miss taxonomy | Same prefix, near prefix, divergent prefix. | Reason-coded cache behavior. |
| Spec decode gate | Draft on/off across model sizes. | Net win or loss on Metal. |

Metrics:

- TTFT P50/P95/P99;
- per-request wall time;
- aggregate decoded tokens/sec;
- per-user decoded tokens/sec;
- prompt tokens recomputed vs reused;
- prefix cache hit rate;
- KV/state bytes resident;
- queue wait time;
- prefill chunk count;
- cancellation cleanup time;
- J/token where available;
- error rate;
- output parity for deterministic cases.

Minimum report table:

| Engine | Model | Hardware | Concurrency | Prompt len | Prefix reuse | TTFT P50 | TTFT P95 | Agg t/s | User t/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|

## Implementation Sprints

### S0 - Pain Radar

Goal:

Turn public complaints into a ranked benchmark backlog.

Deliverables:

- issue query list;
- normalized public pain ledger;
- manual link curation path;
- first 10 benchmark candidates;
- first 5 Apple-relevant reproductions.

Gate:

- at least five pain cases have benchmark configs and pass/fail metrics.

### S1 - Serving Measurement Harness

Goal:

Build the measurement instrument before the engine rewrite.

Deliverables:

- concurrency matrix runner;
- OpenAI-compatible HTTP client;
- SSE timing capture;
- engine launch configs;
- JSONL schema;
- Markdown summary.

Gate:

- current Dismantle and llama.cpp can be compared on one Apple machine.
- report includes P95 TTFT and aggregate t/s.

### S2 - Detached Prefix KV Store

Goal:

Make repeated prefixes resident runtime objects.

Design:

- token-prefix hash;
- exact token verification before reuse;
- resident KV span;
- refcount;
- LRU/TTL;
- byte budget;
- copy into new slot;
- metrics: hit, miss, evict, copied tokens, saved prefill tokens.

Gate:

- repeated identical system prompt reuses prefix after source slot is gone.
- greedy output matches cold prefill.

### S3 - Chunked Prefill

Goal:

Prevent long prompts from blocking short prompts.

Design:

- configurable prefill chunk size;
- yield between chunks;
- decode-ready slots can advance between prefill chunks;
- preserve exact KV/state;
- metrics for chunks, queue relief, and starvation.

Gate:

- one 64k request no longer starves short requests.
- output parity with unchunked prefill.

### S4 - Prefix-Aware Admission

Goal:

Make `prefix-grouped` a policy, not just a selector.

Design:

- longest common prefix grouping;
- same-length buckets;
- cache-hit priority;
- age score to avoid starvation;
- memory pressure penalty;
- cancellation-aware cleanup.

Gate:

- shared-agent workload improves prefix reuse without unacceptable P95 regression.

### S5 - Stop Strings, Cancellation, And Safety

Goal:

Continuous batching cannot be production-grade until client semantics are right.

Deliverables:

- OpenAI `stop` in batch scheduler;
- SSE disconnect releases slot;
- cancellation does not poison reusable state;
- metrics for dropped clients and cleanup.

Gate:

- cancellation and stop-string tests pass under concurrent load.

### S6 - RWKV/SSM State Reuse

Goal:

Build the part where Hawking can be more than a vLLM-shaped transformer server.

Design:

- persistent per-user RWKV state;
- transcript-prefix state snapshots;
- state fork/rollback for speculative decode;
- compact long-horizon agent sessions;
- state cache budget separate from transformer KV budget.

Gate:

- multi-turn RWKV session memory stays effectively constant with history length.
- state snapshot reuse beats re-prefill on repeated agent turns.

### S7 - Spec Decode On Apple

Goal:

Spec decode must be proven net-positive on Metal before it becomes a default
claim.

Design:

- user n-gram draft;
- small draft model;
- RWKV state-fork verifier;
- acceptance-rate logging;
- net tokens/sec gate;
- latency gate, not only throughput gate.

Gate:

- spec path is disabled by default unless it wins on the target Apple tier.

### S8 - Hawking Release Candidates

Goal:

Connect runtime wins to downloadable artifacts.

Deliverables:

- Hawking-tuned RWKV/SSM target;
- Hawking quantized transformer target;
- model cards with Apple-local TTFT, P95, memory, and energy where available;
- one reproducible model/runtime bundle;
- release note with measured claims only.

Gate:

- a user can download one bundle and reproduce the flagship Apple serving
  result without reading internal notes.

## Suggested File Tree

```text
tools/research/pain_radar.py
tools/bench/serve_concurrency_matrix.py
tools/bench/workloads/shared_agent.json
tools/bench/workloads/long_prompt_burst.json
tools/bench/workloads/mixed_latency.json
docs/research/pain_radar/ledger.jsonl
docs/research/pain_radar/clusters.md
docs/reports/serve_matrix/README.md
docs/reports/apple_serving_pain_fixes.md
```

## Implemented Offline Scaffolding

The non-compute pieces now exist and are safe to use without launching models:

| Surface | Path | Purpose |
|---|---|---|
| Pain radar tool | `tools/research/pain_radar.py` | Seed, add, refresh, and summarize public pain links. |
| Pain ledger | `docs/research/pain_radar/ledger.jsonl` | Public issue/link JSONL with pain class, score, and benchmark candidate flag. |
| Pain clusters | `docs/research/pain_radar/clusters.md` | Human-readable grouping by pain class. |
| Claim ledger | `docs/reports/apple_serving_pain_fixes.md` | Planned/reproduced/fixed table for public claims. |
| Matrix harness | `tools/bench/serve_concurrency_matrix.py` | OpenAI-compatible concurrency/TTFT/P95 harness; inert unless given a server or launch command. |
| Workload generator | `tools/bench/workloads/from_pain_radar.py` | Converts pain ledger rows into workload configs. |
| Workload configs | `tools/bench/workloads/*.json` | Shared-agent, long-prompt, mixed-latency, cache taxonomy, and spec gate shapes. |
| Engine templates | `tools/bench/engines/*.json` | Dismantle, llama.cpp, and vLLM OpenAI-compatible launch/base URL templates. |
| Report directory | `docs/reports/serve_matrix/README.md` | Run protocol and report requirements. |

## Claim Discipline

Safe claims before implementation:

- vLLM is architecturally optimized for high-throughput serving.
- Dismantle should adopt the useful serving lessons.
- Apple-local workloads deserve their own benchmark tier.
- RWKV/SSM state gives a possible long-session advantage.

Unsafe claims before measurements:

- "Hawking is faster than vLLM."
- "llama.cpp is slow."
- "Spec decode is a win on Metal."
- "96GB numbers imply 16GB user experience."
- "Concurrency wins are guaranteed by batching."

## Do Not Do First

- Do not build a generic distributed vLLM clone.
- Do not reintroduce non-Apple backend complexity before the Apple path is
  measured.
- Do not chase multi-machine architecture before local scheduling works.
- Do not ship a benchmark without P95/P99.
- Do not make speculative decoding a public headline without a net-win gate.
- Do not let pain radar become a pile of links; every useful issue must become
  a workload, a skipped reproduction, or a closed note.

## Immediate Next Work

When training is not occupying attention, start here:

1. Run a plan-only `shared_agent` matrix and inspect the request shape.
2. Choose one local model artifact and record the hardware tier.
3. Run Dismantle `shared_agent` at concurrency 1/2/4 on the current Apple machine.
4. Add the llama.cpp measured adapter run for the same workload.
5. Convert the highest-score pain radar rows into generated workloads.
6. Write the first Dismantle serving report with current limitations plainly
   stated.

## Done Means

This workstream is complete when Dismantle has a report that says:

- what vLLM does right;
- where llama.cpp is already good enough;
- where Dismantle is slower today;
- which pain cases we reproduced;
- which serving changes moved real numbers;
- which workloads Dismantle/Hawking should claim;
- which workloads it should not claim;
- which Hawking model/runtime artifact is the recommended target.

For the Dismantle phase, complete means the implementation landed here and the
numbers came from local Apple hardware. Hawking claims should be framed as the
release direction until the rename and artifacts are real.

The outcome should be a model/runtime release note, not just a reply.
