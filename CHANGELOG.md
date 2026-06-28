# Changelog

## Unreleased (post-v2.0.0)

### Eagle5 spec-decode port to qwen_dense.rs (2026-05-27 overnight)

Six commits landed on main moving the qwen Eagle5 port from "trained heads
exist but are silent inventory in RAM" to "trained heads load + dispatch
end-to-end at decode time" (with caveats — the throughput win is gated on
two follow-ups documented below).

What's now possible — `HAWKING_QWEN_TCB=1 HAWKING_QWEN_EAGLE5=1
HAWKING_QWEN_EAGLE5_K=4 hawking generate --speculate eagle5
--eagle5-head <path/to/q3b_eagle6_long.safetensors> ...` actually invokes
the trained Eagle6 head at decode time, increments `draft_accepted` /
`draft_rejected` counters, and emits tokens identical to no-spec greedy
(parity preserved at temp=0).

Six commits:
- `495572e` Phase A.1: trained-head safetensors loader. Replaces
  `Eagle5Head::load_from_safetensors` Unimplemented stub with a real
  loader using a minimal in-tree safetensors reader (memmap2 +
  serde_json + half; no new crate deps). Reads the real q3b head
  (1.66 GB, 16 tensors) in 4.25s. Handles 1-block (q3b) and 2-block
  (q1p5) heads via `block.*` + `extra_blocks.{0..N-2}.*` naming.
- `50cadd6` Phase A.2: pure-Rust Eagle6 forward pass at S=1.
  Mirrors `Eagle5Head.forward` in `colab/eagle5_train_pytorch.py`:
  prev_embed lookup + concat + in_proj + N transformer blocks
  (rmsnorm + attn + SwiGLU) + final norm + lm_head. Numerical parity
  vs PyTorch on real q3b head: **argmax exact match, L_inf 3.5e-4
  (140× tighter than the 5e-2 gate), L2 within 0.0003%, top-8 8/8
  overlap**.
- `599e265` Phase B.1-B.4: qwen_dense.rs dispatch + serial verify.
  Adds `eagle5_head: Option<Eagle5Head>` to QwenDense, loads at
  construct time, pre-flight gate (temp=0, repetition_penalty=1.0),
  serial verify-then-draft loop mirroring the existing
  ngram-lookahead pattern at qwen_dense.rs:1297-1399. **Both
  invariants tested green:** speculate=off bit-identical to baseline,
  speculate=eagle5 mock-head engages (draft_accepted+rejected > 0)
  AND emits identical tokens to baseline.
- `b319e90` Phase A.3.1: parallel LM-matmul. Splits the 311M-FMA
  LM-head matmul across `std::thread::available_parallelism().clamp(1, 8)`
  via `std::thread::scope`. Parity gate unchanged (L_inf 3.6e-4 vs
  3.5e-4 pre-threading — ~1 ULP drift from partial-sum order).
- `57931fc` Phase A.3.2: parallel block matmuls. Threads
  `matmul_no_bias` by output row (per-row independence = bit-for-bit
  identical to single-thread). Combined with A.3.1, head forward
  drops from ~150-200ms to ~5-10ms per draft step on M3 Pro.

What's NOT shipped yet — both required to actually move dec_tps:

1. **Phase B.3 real capture-layer plumbing.** Trained head currently
   runs in **zero-capture mode**: residual + intermediate are zero
   vectors. The head was trained expecting real captures of the
   verifier's layer-32 (Qwen-3B) hidden state. Accept rate is
   degraded (~0.05-0.15 vs trained's projected 0.70). Real wiring
   requires a mid-TCB Metal→CPU readback at the chosen layer —
   ~2-day attended workstream.

2. **Phase B.5 batched-verify-with-logits.** Current verify path
   is K SERIAL forwards per cycle. Same throughput as no-spec
   regardless of accept rate. The actual perf win requires
   `forward_tokens_batched_with_logits` — runs K positions in one
   TCB and returns per-position logits. Existing
   `forward_tokens_batch_tcb` (prefill helper) discards logits;
   modifying it is ~2-day attended workstream.

Realistic timeline from this commit: 4-6 attended days → first
measured Eagle5 lift on Qwen-3B-Q4_K_M on M3 Pro. Projected ceiling
with real capture + batched verify: 1.5-2.2× baseline, putting
Qwen-3B at ~40-60 dec_tps (closing or beating llama.cpp's ~50).

Files added:
- `crates/hawking-core/src/speculate/safetensors_io.rs` (loader)
- `crates/hawking-core/src/speculate/eagle5_forward.rs` (forward)
- `crates/hawking-core/tests/eagle5_trained_head_load.rs`
- `crates/hawking-core/tests/eagle5_forward_parity.rs`
- `crates/hawking-core/tests/fixtures/eagle5_parity_q3b.json`
- `crates/hawking-core/tests/qwen_eagle5_speculate.rs`
- `tools/eagle5_forward_dump.py`

Files modified:
- `crates/hawking-core/src/speculate/eagle5.rs` (Trained variant)
- `crates/hawking-core/src/speculate/mod.rs` (exports)
- `crates/hawking-core/src/model/qwen_dense.rs` (field + dispatch)
- `docs/eagle5_qwen_port_plan.md` (phase-by-phase plan)

See `memory/eagle5_port_phase_a1_shipped.md` for the full overnight
session log.

### Performance — Qwen2.5-3B-Q4_K_M decode shipped at 26.6 dec_tps (2026-05-26)

Locked-config default-on baseline for Qwen-3B-Q4_K_M on M3 Pro 18 GB:
**~26.6 dec_tps median** (n=5 paired, 32-token greedy). Up from ~21 dec_tps
on May 23, ~17 dec_tps on May 20, and ~1.3 dec_tps in early baselines.
Gap to llama.cpp Metal (~50 dec_tps on the same hardware/model) closes
from 2.46× → **1.88×** — first sub-2× measurement.

Headline lever: pre-decoded Q4_K sub-block scale tables
(`HAWKING_QWEN_Q4K_PREDEC`, default-on, opt-out via `=0`). Lifts the
8 sub-block (scale, min) f32-pair decoding out of the Q4_K matvec hot
path at load time. RSS cost ~760 MB. Math is exactly equivalent — 100%
bit-identical at N=100 corpus greedy sweep.

Stacked stack (all default-on at locked Qwen-3B config):
- `HAWKING_QWEN_TCB=1` — Token Command Buffer single-commit decode
- `HAWKING_QWEN_VOCAB_PRUNE_CORPUS=32000` — 32K corpus-derived LM head
- `HAWKING_QWEN_Q4K_LMHEAD=1` — quantized LM head GEMV
- `HAWKING_QWEN_FFN_DOWN_Q4K=1` — opt-in ffn_down Q6_K→Q4_K requant
- `HAWKING_QWEN_Q4K_PREDEC=1` — pre-decoded sub-block scales (new default)

Opt-in held levers (not default; tradeoffs explicit):
- `HAWKING_QWEN_Q4K_FAST=1` — custom 160 B/block sub-block-contiguous
  layout. Quality 91% (vs predec's 100%). Use when the 760 MB predec
  RAM cost is unaffordable; takes +24.6% vs baseline at lower RAM.
- `HAWKING_QWEN_W4A8=1` — per-block int8 activation × Q4_K weight.
  Quality 20% (drift on long-form generation). Bit-identical gate
  blocks default-on; opt-in for tolerant workloads delivers +14.1%.
- `HAWKING_QWEN_BATCH_PREFILL=1` — B=8 batched prefill (2.1× prefill).

See `memory/qwen_dense_metal_pipeline.md` for the canonical locked
config and `memory/composition_decision_matrix_2026_05_26.md` for the
full quality × perf matrix.

## v2.0.0 — pending

**Goals of this release:** ship hawking as a working pure-Rust + Metal MoE
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
- **`hawking bench-server`** — model-persistent JSON-line bench harness.
  Loads model once, accepts requests over stdin. Eliminates the model-reload
  tax for fast development iteration.
- **`hawking bench-kernel`** — per-kernel micro-bench at production shapes.
  Used to validate "is this kernel actually faster?" at sub-second iteration
  speed without needing a full model load.
- **`tools/bench/bench_diff.sh`** — cross-commit statistical comparison.
  Reports whether a perf change is significant given trial-to-trial variance.
- **TCB-internal trace** (`HAWKING_TCB_TRACE=1`) — per-kernel timing inside
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
- **Per-shape Q4_K kernel autotune** — `hawking autotune` measures multiple
  Q4_K kernel variants at production shapes and picks the best per shape.
- **Per-device memory limits** — `--memory-limit-mb` + profile `device_limits`
  block enforce a budget at engine load.
- **Statistical bench harness** — coexist_bench.sh now reports median,
  trimmed mean, 95% CIs, IQR, and high-spread warnings. Appends every run
  to `bench_results/bench_history.jsonl`.
- **`hawking stats`** subcommand — reports per-expert access distribution.

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
  hawking prioritizes a small Rust codebase over matching every C++ kernel
  optimization. The gap is engineering work, not an architectural ceiling.
- Speculative decode infrastructure is correct but doesn't deliver e2e wins
  on the current architecture (batched verify is sequential-in-CB, not
  K-parallel compute). See decision report for the architectural detail.
- Mixtral on 18 GB is functional but slow due to SSD page-faults on cold
  expert weights. 32+ GB recommended for usable Mixtral throughput.

### Pre-v2.0 history

This is the first tagged hawking release. Prior development (v0.x and
v1.x phase work) is preserved in git history, with key reports in
`reports/archive/`. The broad arc:

- **v0.x** — foundation, kernel autotune, MLA decode, decode arena
- **v1.0** — Mixtral correctness, n-gram spec decode infrastructure
- **v1.1** — single-CB-per-token, batched MLA, fp16 KV/x_norm opt-ins,
  methodology infrastructure
- **v2.0** — honest baseline ship after 1A measurement showed compounded
  opt-in features deliver +0.86% (within noise), establishing the
  architectural ceiling on the current approach.
