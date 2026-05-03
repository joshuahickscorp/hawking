# dismantle architecture

## One-line summary

dismantle is a Rust workspace that compiles to a single CLI binary
(`dismantle`) wrapping a Metal-native Mixture-of-Experts inference
engine for Apple Silicon. It loads GGUF model files, executes the
forward pass via custom Metal kernels, and exposes both an
OpenAI-compatible HTTP server and a benchmark harness.

## The three layers

```
┌────────────────────────────────────────────────────────────────┐
│  dismantle (single Rust binary, several crates)                │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Server  (crates/dismantle-serve)                         │  │
│  │ • /v1/chat/completions   (SSE streaming)                 │  │
│  │ • /v1/completions        (legacy, SSE)                   │  │
│  │ • /healthz, /metrics                                     │  │
│  │ • Continuous batching: prefill/decode interleaving so    │  │
│  │   concurrent requests share MoE kernel launches.         │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                    │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │ Model layer  (crates/dismantle-core/src/{moe, attn,      │  │
│  │                model, gguf, tokenizer, cache,            │  │
│  │                speculate})                               │  │
│  │ • Composes runtime kernels into a forward pass.          │  │
│  │ • Knows DeepSeek-V2-Lite + Qwen3-MoE shapes.             │  │
│  │ • Owns KV cache (in-mem + on-disk prefill cache).        │  │
│  │ • Owns the speculative shared-expert draft loop.         │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                    │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │ Runtime  (crates/dismantle-core/src/{metal, kernels,     │  │
│  │            quant, sample}, shaders/*.metal)              │  │
│  │ • Pure Metal glue. No model knowledge.                   │  │
│  │ • Owns the MTLDevice, command queue, shader cache.       │  │
│  │ • Exposes typed Rust APIs for each .metal kernel.        │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## Why this shape

**Three-layer separation** keeps each concern testable in isolation:

- The **runtime** is pure Metal — exercisable with synthetic
  tensors. Every kernel benchmark `dismantle bench` publishes is
  produced here, with the model and server layers disabled.
- The **model layer** is the glue between kernels and architectures.
  It manipulates the runtime through a small set of typed kernel
  APIs; it never opens an `MTLCommandBuffer` directly.
- The **server layer** is a thin axum surface over an `Engine`
  trait. The bench binary uses the same trait, no HTTP. There is one
  inference path; the server is decorative.

**No dense-fallback path in core.** dismantle is a MoE engine. Models
with dense layers (DeepSeek-V2-Lite's first transformer block)
execute through the same MoE kernel with a single-expert config.
There is no `if dense else moe` branch.

**MIT-licensed Rust, Metal source under the same.** Both Rust crates
and `.metal` source are MIT-licensed and live in the same workspace.
The `.metal` shaders are embedded into the binary at build time via
`include_str!` and compiled at runtime through `MTLDevice
newLibraryWithSource:`.

## Workspace layout

```
dismantle/
├── Cargo.toml                    # workspace root
├── crates/
│   ├── dismantle/                # umbrella binary (clap, dispatches subcommands)
│   ├── dismantle-core/           # library: kernels + model + runtime
│   │   ├── src/
│   │   │   ├── lib.rs            # exports + the `Engine` trait
│   │   │   ├── engine.rs         # Engine, EngineConfig, GenerateRequest, ...
│   │   │   ├── error.rs          # Error / Result
│   │   │   ├── metal/            # device, command queue, shader cache
│   │   │   ├── kernels/          # Rust host: dispatching .metal kernels
│   │   │   ├── moe/              # gate, dispatch, grouped GEMM, gather
│   │   │   ├── attn/             # MLA + standard MHA
│   │   │   ├── quant/            # Q4_K_M / Q5_K_M / Q8_0 + dequant
│   │   │   ├── sample/           # GPU top-K / top-P / temp / mask  (wedge 3)
│   │   │   ├── speculate/        # shared-expert draft path        (wedge 4)
│   │   │   ├── cache/            # KV cache + on-disk prefill cache (wedge 5)
│   │   │   ├── model/            # DeepSeek-V2 + Qwen-MoE forward
│   │   │   ├── gguf/             # GGUF v3 reader
│   │   │   └── tokenizer/        # wrapper over `tokenizers` crate
│   │   └── shaders/              # .metal source, embedded at build
│   ├── dismantle-serve/          # axum HTTP server
│   └── dismantle-bench/          # benchmark suites
├── docs/
│   ├── m3_audit.md               # hardware lockfile
│   ├── kernels.md                # one section per shader
│   └── benchmarks.md             # auto-generated from bench JSON
└── tests/                        # integration tests (correctness + golden)
```

## Invariants

1. **No dense-fallback path in core.** MoE is the only path. Dense
   layers go through the MoE kernel with a single-expert config.
2. **Every kernel feature ships with a `dismantle bench` mode that
   demonstrates the win against a feature-disabled baseline.** No
   wedge ships without a number; every README claim is reproducible
   from one command.
3. **Correctness gate before perf gate.** Each new kernel passes
   numerical equivalence (atol=1e-3 fp16) against an MPSGraph
   reference path before its perf claim is published.
4. **GGUF only in v0.1.** safetensors loaders are v0.2+.
5. **CLI surface stable from v0.1.0.** `dismantle serve / generate /
   bench / version` survive any internal refactor. Subcommand flags
   may grow; existing flags do not break.
6. **Server is decorative.** The same `Engine` runs `dismantle
   bench`, `dismantle generate`, and `dismantle serve`. Numbers
   produced under bench match numbers produced under serve at the
   same batch size.

## How requests flow

### `dismantle generate` (single-shot)

1. clap parses args; `dismantle-core::Engine::load(weights, config)`
   constructs the engine.
2. `Engine::generate(request, sink)` runs prefill + decode, calling
   `sink` once per emitted token.
3. main prints tokens to stdout as they arrive; ends with stats
   summary on stderr.

### `dismantle serve` → `POST /v1/chat/completions`

1. axum handler parses the request, builds a `GenerateRequest`.
2. The slot manager assigns an in-flight slot; if a continuous batch
   is already running, the request joins it on the next decode step.
3. Tokens stream back through SSE; on completion the slot is freed
   and the KV cache is recycled (or persisted to disk if the prompt
   prefix matches a prefill-cache hit policy).

### `dismantle bench --suite wax-vs-llama-cpp`

1. Loads the model into dismantle.
2. Spawns `llama-cli` (pinned commit) as a sibling process with the
   same model and same prompts.
3. Runs side-by-side; emits a JSON document with per-wedge ratios,
   measured GB/s, and notes on what each wedge contributes.

## Module responsibilities

| Module | Owns | Imports from |
|---|---|---|
| `dismantle-core::metal` | MTLDevice, command queue, shader cache | (none) |
| `dismantle-core::kernels` | Typed Rust APIs for each shader | `metal` |
| `dismantle-core::quant` | Q4/Q5/Q8 layouts + dequant | `metal`, `kernels` |
| `dismantle-core::sample` | On-GPU sampling | `metal`, `kernels` |
| `dismantle-core::moe` | Gate + dispatch + grouped GEMM + gather | `metal`, `kernels`, `quant` |
| `dismantle-core::attn` | MLA + MHA | `metal`, `kernels`, `quant` |
| `dismantle-core::cache` | KV cache, on-disk prefill cache | (pure logic) |
| `dismantle-core::speculate` | Shared-expert draft + verify | `moe`, `attn`, `cache` |
| `dismantle-core::gguf` | GGUF v3 reader | (mmap) |
| `dismantle-core::tokenizer` | Tokenize / detokenize | `tokenizers` crate |
| `dismantle-core::model` | Per-architecture forward passes | everything above |
| `dismantle-serve` | HTTP, SSE, continuous batching | `dismantle-core` |
| `dismantle-bench` | Benchmark suites | `dismantle-core` |
| `dismantle` (umbrella) | CLI dispatch | `dismantle-serve`, `dismantle-bench`, `dismantle-core` |

## Versioning

dismantle follows semver. The version is set in the workspace
`Cargo.toml` and surfaced through `dismantle version`, which also
reports the loaded model's identity (DeepSeek-V2-Lite / Qwen3-MoE /
etc).
