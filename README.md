# dismantle

Pure-Rust inference engine for transformer language models on Apple Silicon.
Single binary. No Python at runtime. No llama.cpp dependency. Loads GGUF weights
via mmap and runs them through hand-rolled Metal compute kernels.

dismantle runs **both dense and Mixture-of-Experts** models through one Metal
runtime. The primary tuned target is **Qwen2.5-3B-Instruct Q4_K_M** (dense); the
loader detects the architecture from GGUF metadata and routes to the matching
forward pass.

## Supported families

Architecture is auto-detected from GGUF metadata (`model/mod.rs`):

| Family | Kind | Notes |
|---|---|---|
| **Qwen2.5** (`qwen2` / `qwen2.5`) | dense | primary tuned target; full fast-decode stack |
| Llama 3.x / Mistral (`llama3.x`, `mistral`) | dense | shares the dense fast-decode core |
| Gemma 2 (`gemma2`) | dense | soft-cap logits + per-arch scales |
| Phi-3 / 3.5 (`phi3`) | dense | sliding-window attention |
| DeepSeek-V2-Lite (`deepseek2-lite`) | MoE | 16B params, 2.4B active/token; MLA attention |
| Mixtral 8×7B (`llama`+MoE) | MoE | runs on 18 GB Macs via memory-conscious expert dispatch |
| Qwen3-MoE (`qwen3moe`) | MoE | grouped-expert GEMM path |

Pass any GGUF via `--weights`; unknown architectures error with the supported list.

## What's distinctive

- **Pure Rust + Metal** — single binary, no Python in the runtime, no C++ shim.
  Source-build with `cargo`. The `.metal` shaders are embedded via `include_str!`
  and compiled at runtime through `MTLDevice newLibraryWithSource:`.
- **Dense + MoE in one runtime** — a single Metal kernel set serves both; MoE
  models route through grouped-expert GEMM, dense models through the tuned Q4_K
  GEMV path.
- **Zero-copy GGUF load** — one mmap + a no-copy `MTLBuffer` over the mapping +
  per-tensor offsets. Weights are never copied into a second buffer: **−1.9 GB
  RSS, −324 ms TTFT**, bit-identical (silicon lever #13).
- **OpenAI-compatible HTTP API** — `dismantle serve` exposes `/v1/chat/completions`
  and `/v1/completions` with SSE streaming and structured error codes.
- **Reproducible methodology** — every perf claim is reproducible from the
  `tools/bench/*` harness in tree (paired A/B, statistical CIs, kernel-level
  timing, clean-room contamination control). The kill-ledger
  (`reports/dead_levers.md`) records every lever that was built, measured, and
  ruled out, with its evidence.

## The moat — where dismantle is genuinely strong

Three levers that hold up under clean measurement (the rest are recorded as dead
in `reports/dead_levers.md`):

1. **Prefix-cache reuse** — a default-on exact KV prefix cache. On warm,
   shared-prefix workloads (agent loops, repeated system prompts, multi-turn
   chat) it elides re-prefilling the shared head — up to **~84% of prefill** on a
   matching prefix. The on-disk variant persists across runs.
2. **Speculation on code** — a free n-gram draft (τ=1.43 on code) feeding a
   pruned-vocab Q4_K GPU verify; **+148% decode on repetitive code** generation,
   bit-identical to greedy. (The trained EAGLE draft head is NO-GO — see
   verdicts.)
3. **Low-RAM footprint** — the zero-copy loader keeps peak RSS near the model
   size, which is what lets a 3B run alongside other GPU/RAM-heavy work on an
   18 GB box.

## Measured performance (M3 Pro 18 GB, clean-room, 2026-05-31)

| model | quant | dec_tps | energy | notes |
|---|---|---:|---:|---|
| Qwen2.5-3B-Instruct | Q4_K_M | **~31** | 0.17 J/tok | clean-room anchor (Claude quit), greedy temp=0; range ~29–31 |
| DeepSeek-V2-Lite-Chat | Q4_K_M | ~17 | — | TRIALS=4 TOKENS=24 coexist, 95% CI [16.6, 18.0] |
| Mixtral-8×7B-Instruct | Q3_K_M | ~0.1 | — | functional, SSD-bandwidth-limited on 18 GB |

**Honest ceiling.** llama.cpp Metal lands around ~50 dec_tps on Qwen-3B-Q4_K_M on
the same hardware. dismantle's ~31 is the result of an exhausted decode-kernel
micro-opt track: the Q4_K predec decode GEMV is at the **Apple-GPU memory-model
optimum for batch-1** (bandwidth-bound at ~56% of peak; vectorized unpack,
access-order repack, and occupancy tuning all measured Type-1 dead). Further
dense-decode throughput requires **fewer weight bytes** or the **spec / stateful**
axes — not more kernel micro-optimization. The ~31 anchor is accepted as the
honest steady-state number; the moat levers above are where the real wins are.

## Verdict outcomes (sub-Q4 byte-cut axis — closed 2026-06-01)

The remaining "fewer bytes" bets were taken to decisive quality gates and closed:

- **imatrix mixed-precision (non-uniform Q4/Q3) — NO-GO (Type-1).** Uniform-Q3
  requant is worse than Q4_K on every quality axis (PPL 4.68 > 4.59, logit cos
  0.983 < 0.993); the weight-RMSE oracle already showed the true mixed split
  trails Q4_K on 7/7 tensor families.
- **QTIP 3-bit trellis — leaning NO-GO.** The measured weight-quality bracket is
  +0.44 to +1.37 bits short of Q4_K_M; the decisive Cornell-RelaxML codec run is
  deferred (`ALLOW_FRESH_QTIP_CODEC`), so it is not yet a *recorded* kill.
- **W4A8 mixed-precision decode — HELD** at 1.115× (below the 1.20× ship gate;
  needs AWQ-from-f16, not requant-from-Q4_K).

Net: no sub-Q4 byte-cut GO cleared a gate → no default config flips. Full
classification + resurrection oracles in `reports/dead_levers.md`.

## Requirements

- Apple Silicon Mac (M1, M2, M3, or M4)
  (Off-macOS: the pure-Rust CPU reference path compiles; see `DISMANTLE_FORCE_CPU` below.
  A non-macOS toolchain is required to verify the build — not available in the CI sandbox.)
- Rust stable
- ~4 GB free RAM for Qwen2.5-3B Q4_K_M (model + KV cache)
- ~12 GB free RAM for DeepSeek-V2-Lite Q4_K_M; ~14 GB + 16 GB disk for Mixtral 8×7B

## Build

```sh
git clone https://github.com/joshuahickscorp/dismantle.git
cd dismantle
cargo build --release --workspace
# Binary: target/release/dismantle
```

## Get a model

```sh
./tools/fetch-model.sh        # DeepSeek-V2-Lite Q4_K_M (~9.7 GB)
./tools/fetch-mixtral.sh      # Mixtral 8×7B Q3_K_M (~16 GB)
```

Or download any GGUF (e.g. `qwen2.5-3b-instruct-q4_k_m.gguf`) into `models/` and
pass it via `--weights`. The architecture is detected from metadata.

## Usage

**Check fit before loading:**

```sh
dismantle doctor --weights models/qwen2.5-3b-instruct-q4_k_m.gguf
```

**Pick the fastest kernels for your machine** (run once, takes 1–2 min):

```sh
dismantle autotune \
  --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --out profiles/my-mac.json
```

**Generate:**

```sh
dismantle generate \
  --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --kernel-profile profiles/my-mac.json \
  --prompt "Write a Rust function that reverses a linked list." \
  --max-new-tokens 256
```

**Serve as an OpenAI-compatible HTTP API:**

```sh
dismantle serve \
  --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --kernel-profile profiles/my-mac.json \
  --addr 127.0.0.1:8080
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Write a haiku about Metal kernels."}],
    "max_tokens": 64
  }'
```

`/v1/completions` (legacy) is also served; both stream via SSE. Errors return the
OpenAI `{error:{message,type,code}}` shape. See [docs/serve.md](docs/serve.md).

## Named lever bundles — `--profile fast`

The `--profile` flag (global; place before or after the subcommand) applies a
named set of opt-in levers in one shot. Only one bundle is currently defined:

| Profile | What it sets | Trade-off |
|---|---|---|
| `fast` | `DISMANTLE_QWEN_VOCAB_PRUNE=32000` + Q4K LM-head + Q4K FFN-down + predec + **f16-scales** | +7.4% paired dec_tps on Qwen2.5-3B-Q4_K_M (contaminated-but-paired; clean-room absolute queued); mild quality trade (f16 scale rounding; output is **not** bit-identical to the default) |

```sh
dismantle generate --profile fast \
  --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --prompt "Write a Rust function that reverses a linked list." \
  --max-new-tokens 256
```

Omitting `--profile` leaves every lever at its default: the decode output is
**bit-identical** across runs. Explicitly-set `DISMANTLE_QWEN_*` env vars always
take precedence over the bundle.

The `fast` bundle activates the f16-scales predec variant
(`DISMANTLE_QWEN_PREDEC_F16SCALES=1`): the pre-decoded Q4_K sub-block scale
tables are stored as f16 instead of f32 (160 B/block vs 192 B/block, ~17% less
scale traffic across ~89% of decode). This is the only lever in the bundle that
is not already default-on.

## CPU reach path — `DISMANTLE_FORCE_CPU=1` / `EngineConfig.force_cpu`

A pure-Rust CPU reference backend is wired and exercised in CI (Phase 3.3).
On macOS, set `DISMANTLE_FORCE_CPU=1` to force the engine to load with no Metal
context — the same state it enters off-macOS where Metal is absent:

```sh
DISMANTLE_FORCE_CPU=1 dismantle generate \
  --weights models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt "The capital of France is"
```

**Parity (macOS cross-check):** CPU `forward_token` vs Metal `forward_token_greedy_tcb`
on Qwen2.5-0.5B-Q4_K_M — the asserted **gate is the first-3 greedy token IDs
identical**; **12/12 leading tokens were observed identical** in CI (test
`cpu_backend_parity.rs`). The CPU path pre-dequantizes Q4_K to f32 and runs a
scalar `gemv_f32`; Metal runs the predec fused-FMA GEMV — both agree at the
fp16 floor (atol ≈ 1e-3). Perf is not the bar: CPU decode is ~100× slower than
Metal and that is expected.

**Off-macOS build:** the `metal`/`MTLDevice` deps are already macOS-gated in
`Cargo.toml`; the CPU primitives compile unconditionally. Off-macOS verification
requires a non-macOS Rust toolchain (`aarch64-unknown-linux-*` or similar) —
this was not verifiable in the macOS-only CI sandbox. The programmatic knob
is `EngineConfig::force_cpu = true`; the env var is the CLI equivalent.

MoE CPU decode (DeepSeek-V2-Lite, Mixtral) is **not** in scope for Phase 3.3;
`route_experts` returns `Ok(None)` off-macOS. Scope is dense models only.

## Reproduce the perf numbers

```sh
TRIALS=4 TOKENS=24 bash tools/bench/coexist_bench.sh        # paired median + 95% CI + IQR
```

For the contamination-controlled anchor, quit Claude first and run
`tools/bench/clean_room_batch.sh`. See [tools/bench/README.md](tools/bench/README.md)
for the standardized bench parameter conventions.

## Mixtral 8×7B support

Mixtral Q3_K_M is a secondary target. See [docs/mixtral.md](docs/mixtral.md) for
fetch + run instructions. Throughput is SSD-bandwidth-limited on 18 GB machines
(expert weights page-fault between layers); 32+ GB machines run faster because
more experts stay resident.

## Project status

**v0.2.x, active development.** The engine ships at its current honest
performance with a small, auditable codebase. The decode-kernel micro-opt track
is exhausted (ceiling proven); future throughput work is on the speculation and
stateful-reuse axes. See [ARCHITECTURE.md](ARCHITECTURE.md) for the internal map
and `reports/dead_levers.md` for the full lever ledger.

## License

MIT. See [LICENSE](LICENSE).
