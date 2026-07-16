# Hawking handbook

This is the compact entry point for the engine, model surface, HTTP API, and
documentation map. Dated handoffs, completed incidents, and generated reports
live in Git history rather than an in-tree archive.

## Canonical documents

- [Operations and Doctor runtime](OPERATIONS.md)
- [Research, evidence, baselines, and killed levers](RESEARCH.md)
- [STRAND execution and promotion](STRAND.md)
- [Program and HIDE roadmap](plans/ROADMAP.md)
- [Environment flags](env_flags.md)
- [Doctor V5 specification](plans/DOCTOR_V5.md)
- [Doctor research and adversarial proof plan](plans/DOCTOR_V5_RESEARCH_PASSES.md)
- [Training ladder](plans/TRAINING_LADDER_V5.md)
- [Appendix](plans/APPENDIX.md) and [handoff](plans/APPENDIX_HANDOFF.md)
- [Event Horizon status](plans/hawking_event_horizon_status.md)
- [Speculative-decode re-entry](plans/spec_decode_reentry_appendix_2026_07_14.md)
- [Studio speculative readiness](plans/spec_decode_studio_readiness_2026_07_12.md)
- [TQ compute-for-memory](plans/tq_compute_for_memory_appendix_2026_07_14.md)

The bound Doctor and Appendix documents are scientific authorities. Summaries
elsewhere do not override their source, evidence, or promotion gates.

## Engine architecture

Hawking is a Rust workspace producing one CLI around a Metal-native inference
engine for Apple silicon.

```text
CLI / HTTP server
        |
model layer: architecture, tokenizer, GGUF, cache, speculation
        |
runtime: typed kernel dispatch, quantization, sampling, Metal / CPU fallback
```

- `crates/hawking` owns the CLI and subcommand dispatch.
- `crates/hawking-serve` owns HTTP and streaming.
- `crates/hawking-bench` owns benchmark suites.
- `crates/hawking-core` owns the engine, model families, GGUF loading, caches,
  speculative paths, quantization, sampling, Metal context, and kernels.

Dense and MoE models are first-class. Metadata selects a family-specific model
implementation; the model composes architecture-independent runtime kernels.
`HAWKING_FORCE_CPU=1` or `EngineConfig::force_cpu` selects pure-Rust fallback
where supported without changing the model layer.

Weights use one mmap and a no-copy Metal buffer with tensor offsets. Metal
sources are embedded in the binary and compiled through the device runtime.
`bench`, `generate`, and `serve` use the same `Engine` trait.

### Architecture invariants

1. Correctness gates precede performance gates.
2. New levers are default-off and cannot change the golden decode hash.
3. Every performance feature has a feature-disabled baseline.
4. Dense and MoE dispatch are explicit and independently tested.
5. The same engine behavior serves CLI, benchmark, and HTTP paths.
6. Shader or profile drift falls back safely rather than silently claiming a
   tuned result.
7. CLI compatibility survives internal consolidation.

### Request flow

`hawking generate` loads the engine, builds a generation request, runs prefill
and decode, and sends each emitted token to a sink.

`hawking serve` maps an HTTP request into the same generation request, streams
tokens as SSE, and releases or preserves cache state according to runtime
policy.

## Models and quant formats

Evidence labels:

- **Verified:** parity or quality gates cover the exercised path.
- **Runs:** loading and generation have worked, but the current environment has
  no complete quality gate.
- **Untested:** architecture support exists without end-to-end local evidence.

| Model family | Status | Notes |
|---|---|---|
| Qwen2.5 0.5B / 3B Q4_K_M | Verified | CPU/Metal or greedy/multisequence parity gates |
| Qwen2.5 1.5B / 7B Q4_K_M | Runs | shared dense path; no complete local quality gate |
| Llama 3.x, Mistral, Gemma 2, Phi-3 | Untested | architecture recognized |
| DeepSeek-V2-Lite Q4_K_M | Runs | MLA/MoE path; local integration gates may skip without weights |
| Mixtral 8x7B Q3_K_M | Runs, impractical on 18 GB | SSD-bandwidth limited |
| Qwen3-MoE | Untested | architecture recognized |

| Format | Status |
|---|---|
| Q4_K_M | verified tuned decode path |
| Q6_K | exercised for selected tensors; kernel parity covered |
| Q3_K_M | exercised on Mixtral; bandwidth constrained |
| Q8_0 / f16 | reference and fallback paths |
| Q2_K, Q5_K, IQ variants | not generally verified |

Verify a model with:

```sh
hawking doctor --weights <model.gguf>
hawking generate --weights <model.gguf> \
  --prompt "Hello" --max-new-tokens 64
```

New model gates should skip cleanly when weights are absent so CI remains
portable.

## HTTP API

Start the loopback server:

```sh
hawking serve \
  --weights models/deepseek-v2-lite-q4.gguf \
  --addr 127.0.0.1:8080
until curl -sf http://127.0.0.1:8080/healthz; do sleep 1; done
```

Endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | readiness/liveness |
| `GET /v1/models` | loaded model metadata |
| `POST /v1/chat/completions` | chat-template generation, JSON or SSE |
| `POST /v1/completions` | legacy prompt generation |
| `GET /metrics` | Prometheus text surface |

Chat fields are `model`, `messages`, `max_tokens`, `temperature`, `top_p`,
`seed`, and `stream`. The model is fixed at load time. A streaming response
emits `data: <json>` events and terminates with `data: [DONE]`.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 16,
    "temperature": 0,
    "stream": false
  }'
```

Bind to loopback unless an authenticated reverse proxy supplies the external
security boundary. Instruct-quality tests must use the chat endpoint so the
architecture-specific template is applied.

## Documentation policy

1. A living fact has one canonical home.
2. Generated evidence stays in ignored report directories.
3. Completed incidents stay discoverable through Git and machine receipts.
4. Dated handoffs are folded into a canonical document before merge.
5. Source-bound documents require explicit supersession and validator
   migration.

Removed paths and replacements are recorded in
[`markdown_redirects.json`](markdown_redirects.json).
