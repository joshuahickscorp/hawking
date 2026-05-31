# Next session — opening prompt for dismantle (paste this whole file)

> Self-contained continuation prompt. Branch `codex/maximal-spec-colab`. Commits authored
> `Joshua Hicks <joshuahicksboba@gmail.com>` **inline** (`git -c user.name=... -c user.email=...`),
> **NO AI attribution of any kind** (no `Co-Authored-By`, no "Generated with"). **Do NOT push** —
> 12 commits are already local-ahead of origin; the user pushes when they decide. Read `CLAUDE.md`
> (operating contract) and `plans/session_closeout_2026_05_31.md` (what just shipped) first.

## What dismantle is
From-scratch Rust + Metal single-stream inference engine for Qwen2.5-3B-Q4_K_M on an M3 Pro (18 GB).
Strategy: `plans/throughput_bible_2026_05_30.md` — read **§3.0** (status correction) and **§8** (the
System-Level Shift / moat). Decode is at its kernel ceiling: ~31 clean tps, the Apple-GPU
memory-model optimum. The decode-GEMV micro-opt track is **CLOSED** — do NOT propose decode-kernel
optimizations. Forward progress runs through **fewer bytes**, **spec**, or the **stateful moat**.

## The contamination rule (load-bearing for every bench)
A running Claude session inflates throughput ~4–5×. **Paired-relative deltas are valid in-session**
(contamination cancels); **absolute numbers are NOT** (GB/s, joules, raw tps). Any oracle/bench with
an *absolute* threshold must be run **clean** (user quits Claude, or runs the committed binary
standalone). CPU/NumPy oracles are immune (no GPU). Always report **spread, not just the mean**, and
label **estimate (proxy) vs measured (production)**.

## Current state
- Tree clean except 3 untracked junk files (`on_smoke.err`, `server_smoke.err`, `traces/` ~250 MB).
- `stash@{0}` = **P1 prefill-MMA partial** (2 MMA kernels drafted in `quant.metal`; no wrappers/parity).
- Prefix cache is **default-ON + bit-identical + bounded** as of `fc93ea0`. The §8 moat is the frontier.

## The prioritized queue — pick by what's available (clean room? GPU? CPU only?)

### Tier 1 — RECOMMENDED FIRST: L3.1 moat oracles (the frontier, CPU/NumPy, in-session-safe)
The durable differentiation axis (bible §8 Layer 3). **Run 3 kill-or-keep oracles before any body**,
in confidence order. Full spec: **`plans/handoff_l31_moat_oracles_2026_05_31.md`** (read it).
- **Oracle 1 (highest confidence — build first if it clears): draft tuning.** Extend
  `tools/bench/oracle_spec_accept.py`: user-warm-start n-gram/SAM τ vs the generic **1.43** baseline.
  GO if warm τ lifts materially toward 2.5 (τ≥1.8 is a real win — free + lossless, no regression risk).
- **Oracle 2 (medium): semantic cache.** Extend `oracle_prefix_cache.py`: incremental near-dup reuse
  over the exact tier. GO if ≥10 pts over exact-only @ ≥95% verify-confirm.
- **Oracle 3 (lowest — full-rank head likely kills it): vocab screen.** New `oracle_vocab_coverage.py`:
  norm-bound certificate fall-back rate. GO if a small hot-set gives ≥80% certified fast-path.
- Pure NumPy/CPU — **no GPU, not contaminated, runs parallel-safe.** Build `usage_capture` (mirror
  `attn_capture.rs`, parity-neutral, default-off) + the cleared lever's body ONLY after an oracle GOes.

### Tier 2 — tactical TTFT: finish P1 prefill-MMA (GPU source build, paired-safe)
Resume `stash@{0}` (`git stash pop`). Add the Rust wrappers (`kernels/mod.rs`, mirror the
`*_v3w_pinned_tcb` wrappers), wire behind `DISMANTLE_QWEN_Q4K_MMA=1` in the `batched_proj!` macro
(`qwen_dense.rs` ~4645), **parity-gate FIRST** (`q4k_batched_gemm_parity` + `p3_batched_prefill_parity`,
atol 1e-3 + token-identity — HALT if it fails, don't ship wrong math), then paired prefill_ms bench
(OFF→ON; contamination cancels). Honest scope: the engine prefills in **B≤8 windows** → target the
**N=8 figure (~+10–15% on the GEMM ≈ ~+13% TTFT)**, NOT the +20.5% headline (needs an N=512
window-enlargement, out of scope). GO→commit; NO-GO→Kill-Protocol entry in `reports/dead_levers.md`.
One source-editing agent at a time; use a git worktree only if running it parallel to another source
lane (watch the base-staleness + model-symlink pitfalls logged in the overnight manifest).

### Tier 3 — CLEAN-ROOM ONLY (need Claude quit; absolute metrics)
- **Q3 byte-cut free oracle:** re-run the existing `q3k_bytecut_bench`, read f32-predec-Q3 GB/s —
  ~50% peak = GO (build the f16-predec-Q3 128-B repack per `plans/cheaper_decode_q3_design_2026_05_31.md`),
  ~30% = NO-GO (hmask residual is the wall → QTIP is the only byte-cut). Forecasts the whole build.
- **~39-vs-~31 anchor reconciliation** (bible §3.0 open item) + **energy baseline**
  (`tools/bench/measure_joules.sh`, macmon ready). Both need a clean run.

## Discipline (CLAUDE.md + bible §8.3.1)
- **Kill Protocol:** every NO-GO records Type-1 vs Type-2 + the reframe considered + why it dies /
  its named cheap oracle, in `reports/dead_levers.md`. Never re-test a Type-1; never resurrect on vibes.
- **Don't re-litigate the kills:** lm_head SVD full-rank, EAGLE-3 head, L1.1 KV working-set, FFN
  block-256 sparsity, f16s-default-on, the decode-kernel micro-opt track.
- Verify decode-path / quality-trade changes **yourself** (re-run the bit-identical / parity gate)
  before committing — don't trust an agent's "gate passed" summary on a default-flip or corruption fix.
- One purpose per commit; commit only the files the lever touches; no sweep commits.

## Recommended first action
**Start Tier 1, Oracle 1 (draft tuning).** It's the frontier, the highest-confidence lever (E by
construction, zero regression risk), CPU/NumPy (no clean room needed), and decisive. If you'd rather
make a GPU/source push, Tier 2 (finish P1) is the tactical alternative — they don't conflict
(CPU oracle lane vs. GPU kernel lane) and can run in parallel.

## Pointers (verified present 2026-05-31)
- Closeout: `plans/session_closeout_2026_05_31.md` · L3.1 spec: `plans/handoff_l31_moat_oracles_2026_05_31.md`
- Designs: `plans/cheaper_decode_q3_design_2026_05_31.md`, `plans/stateful_moat_continuation_design_2026_05_31.md`
- Manifest/tracker: `plans/overnight_build_queue_2026_05_31.md` · Roadmap: `plans/roadmap_2026_05_30.md`
- Kills: `reports/dead_levers.md` · Oracles: `tools/bench/oracle_*.py` · Observer template:
  `crates/dismantle-core/src/stateful/attn_capture.rs`
