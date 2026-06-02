# dismantle architecture

## One-line summary

dismantle is a Rust workspace that compiles to a single CLI binary
(`dismantle`) wrapping a Metal-native inference engine for Apple
Silicon that runs **both dense and Mixture-of-Experts** transformers.
It loads GGUF model files, detects the architecture from metadata,
executes the forward pass via custom Metal kernels, and exposes both
an OpenAI-compatible HTTP server and a benchmark harness. The primary
tuned target is Qwen2.5-3B-Instruct Q4_K_M (dense).

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
│  │ • Dense: Qwen2.5, Llama 3.x/Mistral, Gemma2, Phi-3.      │  │
│  │ • MoE:   DeepSeek-V2-Lite, Mixtral, Qwen3-MoE.           │  │
│  │ • Shared GGUF arch-config reader (arch_config.rs).       │  │
│  │ • Owns KV cache (in-mem + on-disk prefill cache).        │  │
│  │ • Owns the n-gram + EAGLE speculative draft loops.       │  │
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

- The **runtime** is Metal on macOS — exercisable with synthetic
  tensors. Every kernel benchmark `dismantle bench` publishes is
  produced here, with the model and server layers disabled. Off-macOS
  (or when `DISMANTLE_FORCE_CPU=1` / `EngineConfig::force_cpu` is set),
  the `*_dispatch` helpers fall through to pure-Rust CPU primitives in
  `kernels/mod.rs` (scalar f32 `gemv`, `rmsnorm`, `rope_inplace`, …);
  the model layer is unchanged.
- The **model layer** is the glue between kernels and architectures.
  It manipulates the runtime through a small set of typed kernel
  APIs; it never opens an `MTLCommandBuffer` directly.
- The **server layer** is a thin axum surface over an `Engine`
  trait. The bench binary uses the same trait, no HTTP. There is one
  inference path; the server is decorative.

**CPU reach path (Phase 3.3).** `EngineConfig::force_cpu = true` (or env
`DISMANTLE_FORCE_CPU=1`) loads the engine with `metal_ctx = None`, exercising
the same pure-Rust path the engine takes off-macOS. Cross-checked on
Qwen2.5-0.5B-Q4_K_M: the asserted gate is first-3 greedy token IDs identical
(12/12 leading tokens observed identical). Scope is dense models; MoE CPU
decode is a separate follow-up. Off-macOS build verification requires a
non-macOS toolchain (unverified in the macOS CI sandbox).

**Dense and MoE are both first-class.** The model layer dispatches on
the GGUF-detected architecture (`model/mod.rs`) to a per-family
forward pass: dense families (Qwen2.5, Llama 3.x, Gemma2, Phi-3) run
the tuned Q4_K GEMV decode core; MoE families (DeepSeek-V2-Lite,
Mixtral, Qwen3-MoE) run grouped-expert GEMM with memory-conscious
dispatch. Within a MoE model, dense blocks (e.g. DeepSeek-V2-Lite's
first transformer block) still execute through the MoE kernel with a
single-expert config. The duplicated GGUF-metadata reads are factored
into a shared `ArchReader` (`model/arch_config.rs`); each family keeps
its own `*Config` + vocab + per-arch extras.

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
│   │   │   ├── model/            # per-family forward passes + arch_config (ArchReader)
│   │   │   │                     #   dense: qwen_dense, llama, gemma2, phi3
│   │   │   │                     #   MoE:   deepseek_v2, mixtral, qwen_moe
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

1. **Dense and MoE are both first-class.** `model/mod.rs` dispatches
   on the GGUF-detected architecture; dense families run the tuned
   Q4_K GEMV core, MoE families the grouped-expert GEMM. Dense blocks
   inside a MoE model use the MoE kernel with a single-expert config.
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
7. **New levers are default-off.** A new feature must not change the
   default golden decode hash. Opt-in via env var or `EngineConfig`
   field. Applies to `--profile fast` (f16-scales quality trade) and
   `DISMANTLE_FORCE_CPU` (CPU reach path, perf not the bar).

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
| `dismantle-core::speculate` | n-gram + EAGLE draft + verify | `moe`, `attn`, `cache` |
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
