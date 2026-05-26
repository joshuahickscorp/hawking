# Changelog

## Unreleased (post-v2.0.0)

### Performance — Qwen2.5-3B-Q4_K_M decode shipped at 26.6 dec_tps (2026-05-26)

Locked-config default-on baseline for Qwen-3B-Q4_K_M on M3 Pro 18 GB:
**~26.6 dec_tps median** (n=5 paired, 32-token greedy). Up from ~21 dec_tps
on May 23, ~17 dec_tps on May 20, and ~1.3 dec_tps in early baselines.
Gap to llama.cpp Metal (~50 dec_tps on the same hardware/model) closes
from 2.46× → **1.88×** — first sub-2× measurement.

Headline lever: pre-decoded Q4_K sub-block scale tables
(`DISMANTLE_QWEN_Q4K_PREDEC`, default-on, opt-out via `=0`). Lifts the
8 sub-block (scale, min) f32-pair decoding out of the Q4_K matvec hot
path at load time. RSS cost ~760 MB. Math is exactly equivalent — 100%
bit-identical at N=100 corpus greedy sweep.

Stacked stack (all default-on at locked Qwen-3B config):
- `DISMANTLE_QWEN_TCB=1` — Token Command Buffer single-commit decode
- `DISMANTLE_QWEN_VOCAB_PRUNE_CORPUS=32000` — 32K corpus-derived LM head
- `DISMANTLE_QWEN_Q4K_LMHEAD=1` — quantized LM head GEMV
- `DISMANTLE_QWEN_FFN_DOWN_Q4K=1` — opt-in ffn_down Q6_K→Q4_K requant
- `DISMANTLE_QWEN_Q4K_PREDEC=1` — pre-decoded sub-block scales (new default)

Opt-in held levers (not default; tradeoffs explicit):
- `DISMANTLE_QWEN_Q4K_FAST=1` — custom 160 B/block sub-block-contiguous
  layout. Quality 91% (vs predec's 100%). Use when the 760 MB predec
  RAM cost is unaffordable; takes +24.6% vs baseline at lower RAM.
- `DISMANTLE_QWEN_W4A8=1` — per-block int8 activation × Q4_K weight.
  Quality 20% (drift on long-form generation). Bit-identical gate
  blocks default-on; opt-in for tolerant workloads delivers +14.1%.
- `DISMANTLE_QWEN_BATCH_PREFILL=1` — B=8 batched prefill (2.1× prefill).

See `memory/qwen_dense_metal_pipeline.md` for the canonical locked
config and `memory/composition_decision_matrix_2026_05_26.md` for the
full quality × perf matrix.

## v2.0.0 — pending

**Goals of this release:** ship dismantle as a working pure-Rust + Metal MoE
inference engine for Apple Silicon, with Mixtral 8×7B support, reproducible
benchmarking infrastructure, and an honest performance baseline. Performance
is not yet competitive with llama.cpp on raw throughput; this release
establishes the architecture and methodology that future versions will build
on.

### Added

- **Mixtral 8×7B Q3_K_M support** — architecture detection, GGUF loader for
  Mixtral's split-expert tensor layout, llama.cpp-compatible SentencePiece
  tokenizer, mixed-quant routing (Q4_K_M attention projections + Q5_K LM
  head + Q8_0 K/V), and a Q3_K Metal GEMV kernel.
- **MoE expert offloading infrastructure** (`--max-routed-expert-ram-mb`):
  per-layer per-expert access tracking + `posix_madvise(MADV_DONTNEED)`
  eviction of cold expert pages when the routed-expert RAM budget is set.
  Bit-identical output (eviction is OS-level hint).
- **`dismantle bench-server`** — model-persistent JSON-line bench harness.
  Loads model once, accepts requests over stdin. Eliminates the model-reload
  tax for fast development iteration.
- **`dismantle bench-kernel`** — per-kernel micro-bench at production shapes.
  Used to validate "is this kernel actually faster?" at sub-second iteration
  speed without needing a full model load.
- **`tools/bench/bench_diff.sh`** — cross-commit statistical comparison.
  Reports whether a perf change is significant given trial-to-trial variance.
- **TCB-internal trace** (`DISMANTLE_TCB_TRACE=1`) — per-kernel timing inside
  the single command buffer per token. Reveals what's actually hot vs the
  external commit-level trace.
- **Single-command-buffer-per-token forward path** — drops dispatch commits
  per token from ~400 to ~0.125 by encoding all kernels into one Metal CB.
- **K-token DecodeArena** — supports up to 8 tokens of intermediate buffer
  state, enabling batched forward pass.
- **N-gram speculative decoding** (`--speculate ngram`) — opt-in. Note: at
  current architecture, batched verify is sequential-in-CB (not K-parallel
  compute), so spec doesn't deliver perf wins on natural English. See
  `reports/v2.0.0_phase1B_decision.md` for measurement details.
- **Per-shape Q4_K kernel autotune** — `dismantle autotune` measures multiple
  Q4_K kernel variants at production shapes and picks the best per shape.
- **Per-device memory limits** — `--memory-limit-mb` + profile `device_limits`
  block enforce a budget at engine load.
- **Statistical bench harness** — coexist_bench.sh now reports median,
  trimmed mean, 95% CIs, IQR, and high-spread warnings. Appends every run
  to `bench_results/bench_history.jsonl`.
- **`dismantle stats`** subcommand — reports per-expert access distribution.

### Changed

- **README** rewritten with honest measured performance numbers and
  reproduction instructions.
- **Kernel argument convention** — converted hot kernels from per-call
  `set_bytes` scalar args to `KernelArgBuffer` pattern (Metal argument
  buffers). Foundation for Metal `IndirectCommandBuffer` work.

### Removed

- **`AGENTS.md`** (was duplicate of `CLAUDE.md`).
- **`NOTES.md`, `ROADMAP.md`** (replaced by `prompts/v2.0.0_*.md` and
  `reports/v1.1.0_architecture_audit.md` which carry the current strategic
  direction).
- **Phase 5B.2 LM head top-K kernel** — built but never wired into the
  generate path; pure dead code removed.
- **Broken `v056_foundation_parity` test** — was for an evolved 9-arg
  DecodeArena signature; foundation now exercised by every Phase parity
  test downstream.
- **Profile candidate variants** beyond the validated `metal-default`
  (older v0.x experimental schedules deleted in v1.0 simplification).

### Performance baseline

- DeepSeek-V2-Lite Q4_K_M, M3 Pro 18 GB: **~17 dec_tps** (TRIALS=4
  TOKENS=24 coexist median, 95% CI [16.6, 18.0])
- Mixtral 8×7B Q3_K_M, M3 Pro 18 GB: **~0.1 dec_tps** (SSD-bandwidth-limited;
  on 32+ GB machines should be faster as expert weights stay RAM-resident)

### Known limitations

- llama.cpp Metal on identical hardware is roughly 3× faster on V2-Lite.
  dismantle prioritizes a small Rust codebase over matching every C++ kernel
  optimization. The gap is engineering work, not an architectural ceiling.
- Speculative decode infrastructure is correct but doesn't deliver e2e wins
  on the current architecture (batched verify is sequential-in-CB, not
  K-parallel compute). See decision report for the architectural detail.
- Mixtral on 18 GB is functional but slow due to SSD page-faults on cold
  expert weights. 32+ GB recommended for usable Mixtral throughput.

### Pre-v2.0 history

This is the first tagged dismantle release. Prior development (v0.x and
v1.x phase work) is preserved in git history, with key reports in
`reports/archive/`. The broad arc:

- **v0.x** — foundation, kernel autotune, MLA decode, decode arena
- **v1.0** — Mixtral correctness, n-gram spec decode infrastructure
- **v1.1** — single-CB-per-token, batched MLA, fp16 KV/x_norm opt-ins,
  methodology infrastructure
- **v2.0** — honest baseline ship after 1A measurement showed compounded
  opt-in features deliver +0.86% (within noise), establishing the
  architectural ceiling on the current approach.
