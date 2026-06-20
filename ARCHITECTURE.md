# hawking architecture

hawking is a Rust workspace compiling to a single CLI binary that wraps a Metal-native inference engine for Apple Silicon. It loads GGUF models, detects the architecture from metadata, executes the forward pass via custom Metal kernels, and exposes an OpenAI-compatible HTTP server and a benchmark harness.

## Three layers

```
┌────────────────────────────────────────────────────────────────┐
│  hawking (single Rust binary)                                │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Server  (crates/hawking-serve)                         │  │
│  │ • /v1/chat/completions   (SSE streaming)                 │  │
│  │ • /v1/completions        (legacy, SSE)                   │  │
│  │ • /healthz, /metrics                                     │  │
│  │ • Continuous batching: prefill/decode interleaving       │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                    │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │ Model layer  (crates/hawking-core/src/{moe, attn,      │  │
│  │                model, gguf, tokenizer, cache,            │  │
│  │                speculate})                               │  │
│  │ • Composes runtime kernels into a forward pass.          │  │
│  │ • Dense: Qwen2.5, Llama 3.x/Mistral, Gemma2, Phi-3.     │  │
│  │ • MoE:   DeepSeek-V2-Lite, Mixtral, Qwen3-MoE.          │  │
│  │ • Shared GGUF arch-config reader (arch_config.rs).       │  │
│  │ • KV cache (in-mem + on-disk prefill cache).             │  │
│  │ • n-gram + EAGLE speculative draft loops.                │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                    │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │ Runtime  (crates/hawking-core/src/{metal, kernels,     │  │
│  │            quant, sample}, shaders/*.metal)              │  │
│  │ • Pure Metal glue. No model knowledge.                   │  │
│  │ • Owns MTLDevice, command queue, shader cache.           │  │
│  │ • Typed Rust APIs for each .metal kernel.                │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## Design notes

**Three-layer separation** keeps each concern independently testable. The runtime can be exercised with synthetic tensors via `hawking bench` with no model loaded. Off-macOS (or with `HAWKING_FORCE_CPU=1` / `EngineConfig::force_cpu`), dispatch helpers fall through to pure-Rust CPU primitives in `kernels/mod.rs`; the model layer is unchanged.

**Dense and MoE are both first-class.** `model/mod.rs` dispatches on the GGUF-detected architecture: dense families run the tuned Q4_K GEMV decode core; MoE families run grouped-expert GEMM with memory-conscious dispatch. Dense blocks inside a MoE model (e.g. DeepSeek-V2-Lite layer 0) use the MoE kernel with a single-expert config.

**CPU parity.** `EngineConfig::force_cpu = true` loads with `metal_ctx = None`. Cross-checked on Qwen2.5-0.5B-Q4_K_M: 12/12 leading greedy token IDs identical vs Metal. Dense models only; MoE CPU decode is a separate follow-up.

**Zero-copy GGUF load.** One mmap + a no-copy `MTLBuffer` over the mapping + per-tensor offsets. Weights are never copied into a second buffer.

**The server is decorative.** `hawking bench`, `hawking generate`, and `hawking serve` all run the same `Engine` trait. Numbers under bench match numbers under serve at the same batch size.

**Shaders are embedded.** `.metal` sources are compiled into the binary via `include_str!` and compiled at runtime through `MTLDevice newLibraryWithSource:`. No `metallib` artifact; no `xcrun` needed to build.

## Workspace layout

```
hawking/
├── Cargo.toml                    # workspace root
├── crates/
│   ├── hawking/                # umbrella binary (clap, dispatches subcommands)
│   ├── hawking-core/           # library: kernels + model + runtime
│   │   ├── src/
│   │   │   ├── lib.rs            # exports + the Engine trait
│   │   │   ├── engine.rs         # Engine, EngineConfig, GenerateRequest, ...
│   │   │   ├── error.rs          # Error / Result
│   │   │   ├── metal/            # MTLDevice, command queue, shader cache
│   │   │   ├── kernels/          # Rust host: dispatching .metal kernels
│   │   │   ├── moe/              # gate, dispatch, grouped GEMM, gather
│   │   │   ├── attn/             # MLA + standard MHA
│   │   │   ├── quant/            # Q4_K_M / Q5_K_M / Q8_0 + dequant
│   │   │   ├── sample/           # GPU top-K / top-P / temp / mask
│   │   │   ├── speculate/        # n-gram + EAGLE draft path
│   │   │   ├── cache/            # KV cache + on-disk prefill cache
│   │   │   ├── model/            # per-family forward passes + arch_config
│   │   │   │                     #   dense: qwen_dense, llama, gemma2, phi3
│   │   │   │                     #   MoE:   deepseek_v2, mixtral, qwen_moe
│   │   │   ├── gguf/             # GGUF v3 reader
│   │   │   └── tokenizer/        # wrapper over tokenizers crate
│   │   └── shaders/              # .metal source, embedded at build
│   ├── hawking-serve/          # axum HTTP server
│   └── hawking-bench/          # benchmark suites
├── tools/
│   ├── bench/                    # shell bench harness + oracle scripts
│   ├── bisect/                   # automated perf-regression bisect
│   ├── headbank/                 # Eagle5 head staging tool
│   └── training/                 # corpus + Eagle5 training scripts
├── docs/                         # design docs, kernel notes
└── tests/                        # integration tests (correctness + golden)
```

## Module responsibilities

| Module | Owns | Imports from |
|---|---|---|
| `hawking-core::metal` | MTLDevice, command queue, shader cache | — |
| `hawking-core::kernels` | Typed Rust APIs for each shader | `metal` |
| `hawking-core::quant` | Q4/Q5/Q8 layouts + dequant | `metal`, `kernels` |
| `hawking-core::sample` | On-GPU sampling | `metal`, `kernels` |
| `hawking-core::moe` | Gate + dispatch + grouped GEMM + gather | `metal`, `kernels`, `quant` |
| `hawking-core::attn` | MLA + MHA | `metal`, `kernels`, `quant` |
| `hawking-core::cache` | KV cache, on-disk prefill cache | (pure logic) |
| `hawking-core::speculate` | n-gram + EAGLE draft + verify | `moe`, `attn`, `cache` |
| `hawking-core::gguf` | GGUF v3 reader | (mmap) |
| `hawking-core::tokenizer` | Tokenize / detokenize | `tokenizers` crate |
| `hawking-core::model` | Per-architecture forward passes | everything above |
| `hawking-serve` | HTTP, SSE, continuous batching | `hawking-core` |
| `hawking-bench` | Benchmark suites | `hawking-core` |
| `hawking` (umbrella) | CLI dispatch | `hawking-serve`, `hawking-bench`, `hawking-core` |

## Invariants

1. Dense and MoE are both first-class. `model/mod.rs` dispatches on GGUF-detected architecture.
2. Every kernel feature ships with a `hawking bench` mode that demonstrates the win against a feature-disabled baseline. No wedge ships without a number.
3. Correctness gate before perf gate. Each kernel passes numerical equivalence (atol=1e-3 fp16) against a reference before its perf claim is published.
4. GGUF only in v0.1; safetensors loaders are v0.2+.
5. CLI surface stable from v0.1.0. `hawking serve / generate / bench / version` survive any internal refactor.
6. The same `Engine` runs bench, generate, and serve. Numbers produced under bench match serve at equal batch size.
7. New levers are default-off. A new feature must not change the default golden decode hash.

## Request flow

### `hawking generate`
1. clap parses args; `Engine::load(weights, config)` constructs the engine.
2. `Engine::generate(request, sink)` runs prefill + decode, calling `sink` once per emitted token.
3. main prints tokens to stdout; stats summary on stderr.

### `hawking serve` → `POST /v1/chat/completions`
1. axum handler parses the request, builds a `GenerateRequest`.
2. Slot manager assigns an in-flight slot; if a continuous batch is running, the request joins on the next decode step.
3. Tokens stream via SSE; on completion the slot is freed and the KV cache recycled (or persisted if the prefix matches a prefill-cache policy).

## Versioning

hawking follows semver. The version is set in the workspace `Cargo.toml` and surfaced through `hawking version`, which also reports the loaded model's architecture.
