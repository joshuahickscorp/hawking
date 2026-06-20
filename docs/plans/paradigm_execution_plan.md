# Paradigm execution plan — orchestrator brief

> **You are the orchestrator session.** Your job: build and bench the feature
> set in [`paradigmshift.md`](../paradigmshift.md) (repo root), in the order
> below, one change at a time, each correctness-gated and paired-benched. You
> start cold — this document + `paradigmshift.md` + `CLAUDE.md` are your only
> context. Read all three before touching code.
>
> **Two north-star metrics, always:** (1) decode **tokens/sec ↑**, (2)
> **joules/token ↓**. Every step is judged against them.
>
> **Authored 2026-06-01.** Flag/path/test names below come from a code audit of
> that date — **before relying on any env flag, subcommand, or test name,
> confirm it still exists** (`rg <name>` / `--help`). If a name has drifted,
> find the current one; do not invent.

---

## 0. NON-NEGOTIABLES (from CLAUDE.md — violating these fails the task)

1. **Correctness before performance.** A source change to a kernel or decode
   path is **not done** until its parity test is green. A faster kernel that
   fails parity is a regression, not a win.
2. **Commits:** authored by Joshua Hicks via inline options, **local only**, no
   AI attribution, no "Generated with" footer:
   `git -c user.name='Joshua Hicks' -c user.email='joshuahicksboba@gmail.com' commit -m "..."`
   Never `git config` globally. Never push unless the user asks.
3. **One change at a time.** No mid-task scope creep. See a bug? Log it in the
   ledger for a separate pass; do not fix inline.
4. **Bench honesty:** you are a running Claude session → you inflate `dec_tps`
   4–5×. **You may NOT report absolute throughput.** Gate every decision on
   **paired A/B deltas** (contamination cancels). Queue absolute re-measures
   for a clean-room run the *user* triggers (Claude quit). Honest clean anchor:
   **~31 dec_tps, ~0.17 J/tok** on Qwen2.5-3B-Q4_K_M, M3 Pro.
5. **Kill Protocol before any NO-GO** (CLAUDE.md / bible §8.3.1): classify
   Type-1 (died on reality) vs Type-2 (died in the form tested), name the
   reframe, give the kill-oracle or pointer. Record in `reports/dead_levers.md`.
6. **Halt cleanly** if a gate fails, a precondition is missing, or the build
   breaks out of scope. Write a short closeout (root cause, what ran with
   hashes/commits, what unblocks it). A clean logged halt is a success. Do not
   peel-onion-fix underlying issues mid-pass.
7. **Trust nothing from a worktree agent.** If you spawn worktree/sub-agents for
   parallel work, **re-run the parity / bit-identity check yourself in the
   target branch** before merging or trusting "parity passed."
8. **Coexistence:** launch dismantle subprocesses `nice -n 19 taskpolicy -b
   ./target/release/dismantle …`. RSS over ~5 GB is a regression — stop and
   investigate.

### Git setup (do this once, first)
- `git status` first. The working tree may carry 2 **uncommitted Codex-owned**
  JSONs under `docs/archive/.../*.json` — **leave them; never `git add` or
  commit them.** If a checkout would conflict on them, `git stash push -- <those
  paths>` and restore after.
- Create a dedicated branch off `main` for this work (e.g. `paradigm/exec`).
  Commit **only** files each step prescribes — no sweep-commits.

### The per-step loop (run this for EVERY step below)
1. Confirm the precondition / dependency is met.
2. Make the one change (in a worktree if parallelizing — see §3).
3. `cargo build --release --workspace`. If it fails, **stop** — don't bench a
   broken build.
4. `cargo test --workspace --lib` (baseline: ~94 core / 9 serve / 5 bench lib
   tests must stay green).
5. The change's **own parity test** (see each step). Re-run in `main`/target
   branch yourself if a worktree agent produced it.
6. **Paired bench** (`tools/bench/paired_lever.sh` or `coexist_bench.sh`) — B
   (feature on) vs A (feature off), report the **range + 95% CI**, tag
   measured/proxy. Never the mean alone.
7. Apply the **ship gate** (per step). SHIP → commit. HOLD → leave behind the
   flag, record why. KILL → Kill Protocol + `dead_levers.md`.
8. Append a row to the ledger (§4). Queue an absolute re-measure if the lever
   shipped.

---

## 1. ORDER OF OPERATIONS

Phases are sequential. **Within** a phase, steps marked `[parallel-ok]` touch
disjoint files and may run in parallel worktrees; steps marked `[serial]` touch
the decode path (`qwen_dense.rs`, `quant.metal`, `mha.metal`) and **must** run
one at a time, each re-baselining on the previous.

Key files (verified 2026-06-01): decode path
`crates/dismantle-core/src/model/qwen_dense.rs` (`forward_token_greedy_tcb`
~:3819); GEMV kernels `crates/dismantle-core/shaders/quant.metal`
(`gemm_q4_k_v4_predec` :2162); attention `crates/dismantle-core/shaders/mha.metal`
(`mha_decode_f32` :34); sampling `crates/dismantle-core/shaders/sample.metal`
(`sample_argmax_f32` :48); engine trait `crates/dismantle-core/src/engine.rs:200`;
macOS gate `crates/dismantle-core/Cargo.toml:27`; model
`models/qwen2.5-3b-instruct-q4_k_m.gguf` (1.93 GB). Parity tests live in
`crates/dismantle-core/tests/*.rs`; golden hashes in `tests/golden/*.hashes`.

---

### PHASE 0 — Instrument & baseline (NO engine changes)
*You cannot prove a win without a reference. Do this first.*

**0.1 — Capture the paired baseline + queue clean-room absolute** `[J/tok][tps] [serial]`
- Goal: lock the current default config as the A-reference for all later paired benches; queue one clean-room absolute capture.
- Do: build release; run `tools/bench/paired_lever.sh` self-vs-self to confirm the harness + record noise floor; write a `CLEAN-ROOM TODO` block (the exact `tools/bench/clean_room_batch.sh` command) into the ledger for the user to run with Claude quit.
- Gate: harness runs, noise floor recorded. No ship decision here.

**0.2 — Profile the 31→50 gap** `[tps] [serial]`
- Goal: find where the gap actually is (dispatch overhead vs kernel BW vs attention) — this **sets the order of Phase 2**.
- Do: capture the per-token GPU timeline (`DISMANTLE_TCB_TRACE=gpu` + `tools/bench/analyze_tcb_trace.py`); if available, a Metal System Trace (Instruments). Run llama.cpp on the same model+prompt as a paired reference. Diff: dispatches/token, per-kernel µs, attention share, idle gaps.
- Output: a short "gap anatomy" note in the ledger + a recommended Phase-2 order. No code change.

**0.3 — Per-phase energy attribution** `[J/tok] [parallel-ok]` *(timebox: stop if it balloons; the existing `measure_joules.sh` is enough to proceed)*
- Goal: attribute J/tok to decode phases so energy wins are gateable.
- Touch: `tools/bench/measure_joules.sh` (+ a small sampler aligning powermetrics/IOReport to TCB phase boundaries).
- Gate: produces per-phase J/tok on the baseline. Tooling only — no engine parity needed.

**0.4 — Moat regression guards** `[moat] [parallel-ok]`
- Goal: lock the existing moats so Phases 1–4 can't silently break them.
- Do: add/confirm bit-identity tests for prefix-cache (`prefix_cache_parity.rs`) and spec-on-code (n-gram draft) via `dismantle batch-hash` golden hashes in `tests/golden/`.
- Gate: tests green and wired into `cargo test`.

---

### PHASE 1 — Bank the free wins (built levers + sampling)

**1.1 — GPU greedy sampling default (temp=0)** `[tps] [serial]`
- Goal: eliminate the per-token ~600 KB logit copy to CPU; use `sample_argmax_f32` for greedy.
- Touch: sampling path in `qwen_dense.rs` / `sample/mod.rs`; kernel `sample.metal:48`.
- Correctness gate: **bit-identical** — first 3 greedy tokens match baseline AND `dismantle batch-hash` byte-identical (b3sum) vs the CPU-sample path on a real model.
- Bench: paired tps; expect small but free.
- Ship gate: bit-identical + paired delta ≥ 0 (not a regression). If not bit-identical → it's a bug, fix or HOLD.

**1.2 — "fast" profile: stack the byte-cut levers** `[tps][J/tok] [serial]`
- Goal: make the both-metrics-optimal config a **named profile** (a kernel-profile JSON or `--profile fast`), NOT a silent global default — so the default decode stays bit-identical and the fast profile is opt-in + documented.
- Sub-steps (do each as its own commit, paired + quality-gated, in this order):
  1. **f16-scales** (`DISMANTLE_QWEN_PREDEC_F16SCALES`) — NOT bit-identical. Quality gate: logit-cosine ≥ ~0.999 and PPL within ~+0.05 of the Q4_K baseline (confirm the repo's existing quality-oracle thresholds; if none, propose and get user sign-off). Measured prior: +9.3% tps, −1.4% J/tok.
  2. **Q4_K LM head** (`DISMANTLE_QWEN_Q4K_LMHEAD`) — quality gate (requantizes).
  3. **Q4_K FFN-down** (`DISMANTLE_QWEN_FFN_DOWN_Q4K`) — quality gate. ⚠️ **Known interaction:** lowers spec-decode accept rate (~7→3 in prior tests). Measure the FFN-down × spec-on-code interaction; if it kills the spec moat, keep FFN-down OUT of the profile or make them mutually exclusive. Record the call.
  4. **vocab-prune** (`--vocab-prune-path`) — needs a whitelist; quality gate on coverage.
- Bench: paired, the full stack vs baseline, **and** each lever marginal.
- Ship gate: each lever clears its quality gate AND the stacked profile shows a clear paired tps gain with J/tok not regressed. Per "reach for more" — stack, don't stop at the first win.

---

### PHASE 2 — Throughput structural wins (order from 0.2; all `[serial]`)

> Attack in the order 0.2's gap-anatomy recommends. Default expected order below.
> Every step: parity `atol=1e-3` fp16 (reduction-reorder may add `rtol=1e-4`,
> never loosen `atol`); 3-token greedy parity; then paired bench. Re-baseline on
> the prior step.

**2.1 — f16 activations + f16/Q8 KV cache** `[tps][J/tok]`
- Goal: cut activation + attention traffic ~2× (we run f32 today). Q8 KV exists (`--q8-kv`); add f16 activations through the decode path.
- Touch: `qwen_dense.rs` activation buffers; KV append (~:4128); `mha_decode_f32`.
- Gate: parity green incl. a long-context case; paired tps + J/tok.
- Ship gate: clear paired gain, parity green, quality (logit cosine) green.

**2.2 — Reduce dispatch count / fuse kernels** `[tps]`
- Goal: cut the ~180 dispatches/token (the likely biggest chunk of the llama.cpp gap). Candidates from 0.2: fuse adjacent GEMVs, fold norms, larger threadgroup batches.
- Touch: `qwen_dense.rs` dispatch sequence; `quant.metal` fused kernels.
- Gate: parity bit-identical where the math is unchanged (fusion is a reorder, not a precision change) — golden hashes should hold; paired tps.
- Ship gate: clear paired gain, hashes/parity green.

**2.3 — Flash-style decode attention** `[tps]`
- Goal: replace the materialize-all-scores `mha_decode_f32` with an online-softmax flash-decode kernel that holds throughput at long context.
- Touch: `mha.metal`; attention dispatch in `qwen_dense.rs`.
- Gate: parity `atol=1e-3` + `rtol=1e-4` (reduction reorder); **long-context** paired bench (this is where it pays).
- Ship gate: parity green; paired gain at long context; no short-context regression.

---

### PHASE 3 — Backend seam (bit-identical refactor → portability foundation)

> This is the highest-leverage structural move and the riskiest refactor. It
> must change **zero** observable behavior on Metal. Do it incrementally (one
> kernel family behind the trait at a time), each step golden-hash-identical.

**3.1 — Introduce `trait Backend / Device / Buffer`** `[portability] [serial]`
- Goal: a compute-backend seam behind the existing kernel calls (Burn-shape: one user-facing bound, op-traits behind it). The Metal code becomes the first `impl Backend`.
- Touch: new `backend/` module; route `qwen_dense.rs` kernel calls through the trait.
- Gate (HARD): `tests/golden/*.hashes` **unchanged** (bit-identical refactor); all lib tests green; paired tps within noise of pre-refactor (no perf regression).
- Ship gate: bit-identical + no perf regression. If hashes move, it's not a pure refactor — STOP and find what changed.

**3.2 — Op scheduler + CPU fallback** `[portability] [serial]`
- Goal: per-op backend routing with CPU fallback for unsupported ops (the `ggml_backend_sched` lesson — what lets a partial backend ship).
- Gate: Metal path still bit-identical; an op forced to CPU fallback produces parity-correct output.

**3.3 — CPU backend** `[portability] [parallel-ok once 3.1 lands]`
- Goal: a `std::simd`/`gemm`-based CPU `impl Backend` so the engine **compiles and runs off-macOS** (lift the `cfg(target_os="macos")` hard gate, `Cargo.toml:27`).
- Gate: builds on a non-macOS target (CI or cross-check); CPU decode output parity vs Metal (`atol=1e-3`) on the 0.5B model (fast).
- Ship gate: runs end-to-end on CPU with correct output. (Perf is not the bar here — reach is.)

---

### PHASE 4 — High-ceiling bets (explicit gates + Kill Protocol)

**4.1 — QTIP-on-Metal spike (GATE before any format work)** `[tps][quality] [serial]`
- Goal: settle the load-bearing risk — does trellis decode stay **bandwidth-bound** on Apple's cache/simdgroup model? (All QTIP numbers are external-GPU; HYB is tuned to external-GPU's ~4 KB L1 / 32× duplication.)
- Do: port `3INST`/`HYB` trellis decode to a Metal microbench kernel; measure achieved % of peak BW at batch-1.
- **GATE:** if it lands compute-bound (à la the dead Q3_K kernel, ~24% peak) → **STOP**, record a Kill (Type-1 if the simdgroup model can't hide it; Type-2 with a named reframe if a different kernel layout might) in `dead_levers.md`, and skip 4.2–4.3. Only if it stays BW-bound (target >~60% peak) → proceed.

**4.2 — Custom on-disk format (DWA)** `[tps][J/tok] [serial, gated on 4.1]`
- Goal: bake the both-metrics-optimal layout as the zero-cost default + host the trellis codebook + pre-multiplied scales, page-aligned, mmap-ready (no load-time repack).
- Touch: new format reader alongside `gguf/`; loader; `metal/mod.rs` zero-copy binding.
- Gate: **bit-identical decode** vs the in-memory path it replaces (`batch-hash` b3sum).

**4.3 — 2–2.5-bit QTIP model** `[tps][J/tok][quality] [serial, gated on 4.1/4.2]`
- Goal: the actual byte-cut win — ~half the bytes at ≈ Q4_K_M quality.
- Gate: quality vs **our Q4_K_M baseline** (PPL + logit-cosine + N≥100 corpus, the repo's quality-oracle pattern); paired tps + J/tok.
- Ship gate: quality within the agreed envelope of Q4_K_M AND a clear paired tps/J win. Else Kill Protocol.

**4.4 — Cross-vendor GPU backend** `[portability] [parallel-ok once 3.x lands]`
- Goal: reach AMD/Intel/external-GPU/Android. Spike **CubeCL** (single-source `#[cube]`, incl. direct MSL) maturity first; if not ready, **WGPU+CPU** (Ratchet model).
- Gate: runs on a non-Apple GPU with parity-correct output. Perf will trail hand-tuned Metal — that's expected and acceptable (reach, not speed).

---

## 2. CONFLICTS TO HOLD IN MIND (don't optimize one metric into the other)
- **Portability vs Metal speed:** never replace the hand-tuned Metal hot path
  with a generic kernel. Keep Metal specialized **behind** the seam; generic
  backends are for *other* vendors + CPU fallback.
- **tps vs J/tok:** they align except (a) f16-scales (a both-win — already in
  1.2) and (b) race-to-idle vs run-cool (only if you ever expose a DVFS knob —
  out of scope unless 0.3 shows it's controllable). If a lever helps tps but
  hurts J/tok, report **both** and let the gate weigh them; do not silently
  trade one away.

---

## 3. BENCH & MEASUREMENT PROTOCOL (precise)
- **Gating = paired A/B only.** `tools/bench/paired_lever.sh` (or
  `coexist_bench.sh`): interleaved B-vs-A trials, report median **+ range + 95%
  CI + IQR**. A lever ships only if the paired delta's CI excludes 0 in the
  right direction.
- **Absolute tps/J = deferred.** You (Claude open) cannot produce them. For each
  shipped lever, append a `CLEAN-ROOM TODO:` line with the exact
  `clean_room_batch.sh` command for the user to run with Claude quit.
- **Energy:** `measure_joules.sh` (macmon/powermetrics; `J/tok = avg_W ·
  decode_s / tokens`). Report J/tok as a paired delta too.
- **Always:** report ranges not means; tag every number **measured** vs
  **proxy/estimate** and name what would make it measured.
- **Parity floors:** kernel `atol=1e-3` fp16 (never loosen); reduction-reorder
  may add `rtol=1e-4`; token parity = first 3 greedy IDs; bit-identity = b3sum
  byte-identical via `dismantle batch-hash` on a real model.

### Parallelism (you are an "orchestrator")
- You MAY spawn worktree/sub-agents for `[parallel-ok]` steps (disjoint files).
- You MUST serialize `[serial]` steps and **re-run their parity/bench yourself**
  in the target branch before trusting any agent's result.
- Never let two in-flight changes touch the same decode-path file.

---

## 4. THE RUNNING LEDGER (`reports/paradigm_execution_log.md`)
`reports/` is gitignored — this is your on-disk working log. One row per step:

| step | branch@commit | parity (test / hashes / atol) | paired Δtps (range, CI, meas/proxy) | Δ J/tok | gate verdict (Scross-vendor GPU stack/HOLD/KILL) | CLEAN-ROOM TODO | notes |

Keep a top section: current baseline reference, the 0.2 gap-anatomy, and the
list of queued clean-room absolutes. When a lever dies, also write it to
`reports/dead_levers.md` with the Kill Protocol classification.

---

## 5. DEFINITION OF DONE
- **Per step:** builds, lib tests green, its parity gate green, paired-benched
  with the verdict + ledger row, committed (if SHIP) as Joshua Hicks, local.
- **Per phase:** all steps resolved (Scross-vendor GPU stack/HOLD/KILL recorded); moats still
  bit-identical (0.4 guards green); a clean-room absolute queued for the phase.
- **Overall:** Phases 0–3 complete (instrumented, free wins banked, structural
  throughput attacked, backend seam + CPU backend landed so the engine runs
  off-macOS); Phase 4 either shipped or cleanly killed per protocol. Final
  closeout: what shipped (with paired deltas), what's queued for clean-room
  absolute, what was killed and why, and the next-session opening prompt.

---

## 6. FIRST ACTION (the first hour)
1. Read `paradigmshift.md`, this file, and `CLAUDE.md` end to end.
2. `git status`; protect the Codex JSONs; branch off `main` (`paradigm/exec`).
3. `cargo build --release --workspace` && `cargo test --workspace --lib` — confirm a green starting point. If red, **halt and report**.
4. Execute **Phase 0** (0.1 baseline harness → 0.2 gap profile → 0.3 energy attribution → 0.4 moat guards). The 0.2 output decides Phase 2's order.
5. Then proceed phase by phase, per-step loop (§0), updating the ledger.
6. Halt + closeout the moment a gate fails or a precondition is missing — do not improvise around it.
