# Production roadmap — dismantle Eagle4 to 100 tps

**Purpose:** autonomous-execution roadmap for a fresh agent session to
iterate against until Eagle4 dec_tps clears 100 on M3 Pro 18 GB.

**Starting state:** commit `d1d50fb` — Eagle4 mode = 13 dec_tps median
(AMX cblas for head + Wedge C single-TCB capture). Bit-identical greedy
vs Off preserved. 89.6 % draft acceptance.

**Target:** 95-125 dec_tps sustained per the deep-research ceiling
(`eagle4_deep_research.md`). Practical floor 95; aspirational peak 125.

**Constraints (hard):**
- Bit-identical greedy vs `SpeculateMode::Off` preserved at every
  commit. The `eagle4_decode_parity` test is the gate.
- All commits authored as `Joshua Hicks <joshuahicksboba@gmail.com>`
  via inline git identity per `CLAUDE.md`. No Co-Authored-By trailers,
  no "Generated with" footers.
- Single-purpose commits. Each commit lands one stage step.
- Halt rule (CLAUDE.md): below a gate's lower bound → halt and debug,
  do not paper over.

**Constraints (soft):**
- Compute steps (bench in clean window) need user to Cmd-Q Claude.app.
  Provide a kickoff script; mark when manual run is required.
- Working-tree diagnostic edits in `engine.rs`, `kernels/mod.rs`,
  `deepseek_v2.rs`, and `tests/ffn_shared_only_nonzero.rs` are
  user-owned and preserved across commits — selectively stage only
  roadmap-related changes.

---

## Calendar overview

```
WEEK 1 — Stage 2 (Path B kernels): 9.6 → 38-50 tps
WEEK 2 — Stage 3 (routing recall + masked prefetch): 50 → 55-75
WEEK 3 — Stage 4 (DySpec tree decode): 75 → 70-95
WEEK 4 — Stage 5 (hardware + ANE + Q4-KV): 95 → 95-125 sustained
PARALLEL throughout — Stage 0.5 partial work: Off baseline 27 → ~55 tps
```

Stage 0.5 cascades to every stage above. Even partial wins (one kernel
hot-path rewrite) lift Off baseline → Eagle4 follows by ~0.4× ratio.

---

## Iteration protocol (per commit)

Every commit MUST follow this cycle. No exceptions.

```
1. Pick the smallest next task from the active stage's task list.
2. Architectural change (Rust / Metal MSL / Python).
3. cargo build --release --workspace        (must compile clean)
4. cargo test -p dismantle-core --lib --release  (45 lib tests green)
5. cargo test -p dismantle-core --release --test eagle4_capture_smoke
   (still green)
6. SMOKE: ./target/release/dismantle generate --speculate eagle4 ...
   max-new-tokens=16, capture dec_tps + accept rate.
   If smoke regresses by >2× without explanation → halt, write
   blocked doc, do not commit.
7. PARITY: EAGLE4_PARITY_TEST=1 cargo test --test eagle4_decode_parity
   -- --ignored --nocapture   (bit-identical to Off MUST pass)
8. Selectively stage (skip user's diagnostic-edit hunks):
   git -c user.name='Joshua Hicks' -c user.email='joshuahicksboba@gmail.com' \
       add <only-roadmap-files>
9. Single-purpose commit with structured message (see template below).
10. After 3-5 architectural commits land in a stage, the user runs the
    stage's bench script in a clean window. The agent then reads the
    raw.json result and decides next steps.
```

### Commit message template

```
path-to-90 stage <N> step <K>: <one-line subject>

<2-3 paragraph what + why>

Smoke (3-5 runs, "The quick brown fox" 16 tokens):
  dec_tps=X.X / X.X / X.X — median X.X
  accept: N/M

Bit-identical greedy regression: PASSES (or FAIL details if blocking).

What this commit ships:
  - <list of file changes>

What's deferred to next commit:
  - <followup items>
```

### Halt rule application

Per CLAUDE.md § Halt rule, on a stage measurement falling below its
band's lower bound:

1. Write `reports/path_to_90/stage<N>_halt_<short>.md` with: root
   cause hypothesis, what investigations would unblock, parking lot.
2. Do NOT continue to the next stage.
3. Spawn a chip (`mcp__ccd_session__spawn_task`) for the investigation
   if it's off the critical path. If on critical path, debug in-place
   with the structured commit message + halt doc.
4. The roadmap continues to the next stage only after the halt clears.

---

## STAGE 2 — Path B kernels (~5-7 days, target 38-50 tps)

### Goal

K-batched verify infrastructure. One V2-Lite forward verifies K eagle4
candidates with weight reads amortized across K query columns.

### Math justification

Currently per emit at K=1: ~76 ms = 41 ms V2-Lite + 15 ms Eagle4Head
(AMX) + 10 ms lm_head argmax + ~5 ms misc.

At K=4 with weight-amortizing kernels:
- 4 Eagle4Head AMX (chained autoregressive) = 60 ms
- 1 V2-Lite forward with K-batched verify = ~50 ms (slightly more than
  K=1's 41 ms due to per-row K-column accumulation)
- Per K-cycle: ~110 ms; emits ~3.2 tokens at 87 % chain accept
- Per emit: ~34 ms = **29 tps** — bottom of Stage 2 band

At K=4 with chain acceptance improved to τ=4.0 (post-Stage-3 fine-tune):
- Per K-cycle: 110 ms; emits 4 tokens → ~28 ms/emit = **36 tps**

Stage 2's plan-band 38-50 assumes K=4-8 with at least one of these in
place. Aim for K=4 first; if τ holds, extend to K=8 in a later commit.

### Task breakdown (in order)

#### 2.1 — Design doc refresh
**File:** `reports/path_to_90/path_b/design.md` (extend existing)
**Deliverable:** kernel signatures finalized for the 3 K-batched gemvs.
Threadgroup memory budget verified against M3 Pro (~32 KB/core).
**Test gate:** none (paper).
**Effort:** half day.

Tasks:
- Document `gemv_q6_k_v3_kbatch` signature for LM head
  (102400×2048 Q6_K, K query cols). Layout for the K-column TG memory.
- Document `gemv_q4_k_m_v2_kbatch` for V2-Lite attn/MoE projections
  (covers gemv_q4_k variants currently in `kernels/mod.rs`).
- Document `mla_decode_kernel_fc_kbatch` — K queries, shared KV read.
  Threadgroup memory tight at K=4; verify ≤ 32 KB.

#### 2.2 — `gemv_q6_k_v3_kbatch` Metal kernel (easiest first)
**Files:**
- `crates/dismantle-core/shaders/gemv_q6_k_v3_kbatch.metal` (new MSL)
- `crates/dismantle-core/src/kernels/parallel_k.rs` (replace
  Unimplemented dispatcher)
- `crates/dismantle-core/tests/path_b_parity.rs` (un-`#[ignore]`)
**Test gate:** K=4 output bit-identical to 4 sequential K=1 gemvs
(atol=1e-3 fp16). Wall-clock ≤ 1.8× single-token decode.
**Effort:** 3-5 days.

Implementation notes:
- Grid: `(vocab_rows / TG_ROWS, K)`. Each threadgroup processes TG_ROWS
  output rows × K input columns.
- TG memory: one K×HIDDEN_DIM scratch tile for the K query vectors (8 KB
  for K=4, HIDDEN=2048). Plus weight tile.
- Weight is read once per TG, shared across all K outputs.

#### 2.3 — `gemv_q4_k_m_v2_kbatch` for attn projections
**Files:**
- `crates/dismantle-core/shaders/gemv_q4_k_m_v2_kbatch.metal` (new MSL)
- `crates/dismantle-core/src/kernels/parallel_k.rs` (add dispatcher)
- Path B parity test extended for Q4_K
**Test gate:** K=4 bit-identical at atol=1e-3 fp16.
**Effort:** 3-5 days (Q4_K block dequant is more involved than Q6_K).

#### 2.4 — `mla_decode_kernel_fc_kbatch` (hardest)
**Files:**
- `crates/dismantle-core/shaders/mla_decode_fc_kbatch.metal` (new MSL)
- `crates/dismantle-core/src/kernels/parallel_k.rs` (dispatcher)
- Parity test
**Risk:** TG memory budget. K=4 may force tile-size reduction. The
design pass in 2.1 must verify before coding.
**Test gate:** K=4 vs 4 K=1; atol=1e-3 fp16. Wall-clock ≤ 1.5× single-
token MLA decode.
**Effort:** 5-7 days.

#### 2.5 — `moe_block_batched_indexed_kbatch_masked` (or skip-masked
variant for Stage 2)
**Files:**
- `crates/dismantle-core/shaders/moe_block_kbatch.metal`
- `crates/dismantle-core/src/kernels/parallel_k.rs`
- Parity test
**Approach:** ship no-overlap K=4 (K sequential expert calls in one TCB)
first to validate parity; the masked-prefetch variant is Stage 3 (task 3.2).
**Test gate:** K=4 vs K=1 sequential; atol=1e-3 fp16.
**Effort:** 5-7 days.

#### 2.6 — `forward_tokens_batched_parallel_k` wire-up in DeepSeekV2
**Files:** `crates/dismantle-core/src/model/deepseek_v2.rs`
**Deliverable:** new method that routes through the K-batched kernels
when the profile selects `verify_kernels = "parallel-k"`.
**Test gate:** profile flag toggleable; bit-identical at K=1 vs the
existing sequential path.
**Effort:** half day.

#### 2.7 — Eagle4 chain spec decode at K=4
**File:** `crates/dismantle-core/src/model/deepseek_v2.rs` (the Eagle4
decode-loop branch in `generate`)
**Deliverable:**
- Eagle4 head proposes K candidates autoregressively, using
  `draft_hidden` from step i as the `h_high` input for step i+1.
  `h_low` / `h_mid` / `h_shared` stay constant (most recent verifier
  values).
- One `forward_tokens_batched_parallel_k(K=4)` call verifies all K.
- Longest-matching-prefix acceptance rule: accept tokens [0..j] where j
  is the first position where draft[j] != verifier_argmax[j]. Emit
  draft[0..j] + verifier_argmax[j] as the bonus correction. KV rollback
  to seq_len_base + j + 1.
**Test gate:** `eagle4_decode_parity` (extended to K∈{1,2,4}) passes
bit-identical at all K.
**Effort:** 1 day.

#### 2.8 — Stage 2 measurement
**Compute step.** User runs `tools/bench/stage2_measurement.sh` (new) in
clean window. Reports Off vs Eagle4 K=4 dec_tps + acceptance + chain τ.
**Gate:** ≥ 38 tps sustained median, bit-identical at K=1 vs Off.
**Halt rule:** if < 30, halt — likely candidate is mla_decode_kbatch
or chain acceptance collapse. Spawn debug session.

---

## STAGE 3 — Routing-recall fine-tune + masked prefetch (~3-5 days, target 55-75)

### Task breakdown

#### 3.1 — Routing-recall fine-tune in eagle4
**File:** `eagle4/eagle4.py` — new subcommand `fine_tune_routing` or
modify existing `train` with a routing-mass-loss weight ≥ 5.0.
**Deliverable:** `eagle4/checkpoints/eagle4_v3/best_recall.npz` with
mask top-8 recall ≥ 60 %. Acceptance may drop 2-4 pp from 87.48 % to
~83-85 % — acceptable trade.
**Compute step.** Runs in eagle4 venv on user's hardware. ~1 day train.
Independent of Stage 2 kernels; can run in parallel.
**Test gate:** `python eagle4/eagle4.py eval --ckpt best_recall.npz ...`
reports `mask_topk_mean_recall ≥ 0.60`.
**Effort:** 1 day (training + iteration).

#### 3.2 — `moe_block_batched_indexed_kbatch_masked` (masked prefetch)
**Files:**
- `crates/dismantle-core/shaders/moe_block_kbatch_masked.metal`
- `crates/dismantle-core/src/kernels/parallel_k.rs`
- `crates/dismantle-core/tests/path_b_eagle4_parity.rs` (new)
**Deliverable:** kernel accepts `predicted_mask` (26×64 packed bits)
from eagle4 head; issues async prefetch (Metal residency hints) of
mask-bit-set expert weight tiles BEFORE the dispatch needs them. For
prefetch misses, fall back to on-demand load.
**Prerequisite:** task 3.1 — needs recall ≥ 60 %.
**Test gate:** K=4 masked verify vs K=4 unmasked; atol=1e-3 fp16. With
`best_recall.npz` loaded, ≥ 5 % faster than unmasked baseline.
**Effort:** 3 days.

#### 3.3 — Wire predicted mask through decode loop
**File:** `crates/dismantle-core/src/model/deepseek_v2.rs`
**Deliverable:** Eagle4 decode loop passes eagle4 head's mask_logits
(26×64) through to `forward_tokens_batched_parallel_k` as the prefetch
hint. mask_logits computation REACTIVATED in eagle4 head Metal/AMX
forward (was skipped in commits `0d6a2a3` and `808d8db`).
**Effort:** half day.

#### 3.4 — Stage 3 measurement
**Compute step.** Same as Stage 2 but with `best_recall.npz` head + masked
verify enabled.
**Gate:** ≥ 55 tps sustained; ≥ 60 % mask recall in evidence/raw.json.

---

## STAGE 4 — DySpec dynamic tree decode (~5-7 days, target 70-95)

### Background

Per `eagle4_deep_research.md`: Qwen3.6-A3B llama.cpp evidence shows
fixed tree shapes hurt MoE because expert-union grows uncontrolled.
DySpec: per-token tree-shape calibration using eagle4's `calib_logit`
(P(accept)). High-calib positions: wider branching. Low-calib: narrow.

### Task breakdown

#### 4.1 — Tree decode design refresh
**File:** `reports/path_to_90/tree_decode/design.md` (extend existing
step 20 doc from commit `f946033`)
**Deliverable:** tree-shape function `f(calib_logit) → (depth, width)`,
tree attention mask construction, verify-side accept/reject across tree
branches.
**Effort:** half day.

#### 4.2 — Tree mask + tree attention kernel
**Files:**
- `crates/dismantle-core/shaders/tree_attention.metal`
- `crates/dismantle-core/src/kernels/tree.rs`
- Parity test
**Deliverable:** Tree attention kernel handles the tree mask (not
diagonal at S>1, encodes parent-child constraints).
**Test gate:** tree mask reduces to diagonal at K=1; tree mask at
K=2-4 produces bit-identical output to running parents-only attention.
**Effort:** 3 days.

#### 4.3 — `propose_tree` on Eagle4Head + tree wire-up
**Files:**
- `crates/dismantle-core/src/speculate/eagle4_head.rs`
- `crates/dismantle-core/src/speculate/tree.rs` (new)
- `crates/dismantle-core/src/model/deepseek_v2.rs`
**Deliverable:** eagle4 head proposes a tree (not chain) of candidates.
DeepSeekV2 verifies the tree in one batched forward using the tree
attention kernel.
**Test gate:** tree decode mode bit-identical to chain mode at tree
depth 1; produces strictly more emits/forward at tree depth 2-3.
**Effort:** 3-4 days.

#### 4.4 — Stage 4 measurement
**Compute step.** Bench with tree decode enabled.
**Gate:** ≥ 70 tps sustained; tree multiplier 1.4× over chain spec
decode on Spec-Bench.

---

## STAGE 5 — Hardware paths (~1 week, target 95-125)

In ROI order per deep research.

### 5.1 — AMX further (already partially done; commit `d1d50fb`)
Currently AMX is used for Eagle4Head's 6 gemvs. Extend to:
- V2-Lite's smaller projection gemvs (q_b_proj, kv_b_proj where
  matrix size fits AMX's sweet spot)
- Test gate: bit-identical to Metal path at atol=1e-3 fp16.
**Effort:** 2 days.
**Estimated gain:** +5-10 tps.

### 5.2 — Per-head adaptive MLA KV quantization
**Files:** `crates/dismantle-core/src/metal/decode_arena.rs`, MLA kernels
**Deliverable:** keep MLA "sink" latent dims (~first 32) at FP16,
quantize the rest to Q4. Per llama.cpp Issue #21385 — recovers most
quality vs flat Q4.
**Block-ship gate:** perplexity regression ≤ 1 % on wikitext2-256.
**Effort:** 2 days.
**Estimated gain:** +5 tps.

### 5.3 — Async verify-start
**Files:** `crates/dismantle-core/src/model/deepseek_v2.rs`
**Deliverable:** last eagle4 draft step's hidden production OVERLAPS
first V2-Lite verify layer's expert prefetch. Class B but trivial to
land.
**Test gate:** still bit-identical (overlap is just scheduling).
**Effort:** 2 days.
**Estimated gain:** +5-8 tps.

### 5.4 — ANE routing-logits offload (NOT verify)
**Files:** `crates/dismantle-core/src/ane/router.rs` (new), Core ML
model file for V2-Lite's gate_logits matrix.
**Deliverable:** V2-Lite's per-MoE-layer router (1, 64) output runs on
ANE concurrent with Metal verify (no UMA contention since it's compute-
bound not bandwidth-bound).
**Test gate:** bit-identical (same gate_logits at atol=1e-4).
**Cap:** 5 % per deep-research § Apple Silicon (UMA contention limits).
**Effort:** 3 days.
**Estimated gain:** +5 tps.

### 5.5 — Multi-queue Metal scheduling
**File:** `crates/dismantle-core/src/metal/mod.rs`
**Deliverable:** separate command queue for draft vs verify so dispatch
overlaps. Profile-first decision.
**Test gate:** bench shows ≥ 3 % improvement OR halt and don't ship.
**Effort:** 2 days.
**Estimated gain:** +3-8 tps.

### 5.6 — Stage 5 measurement (THE HEADLINE)
**Compute step.** Full stack bench: 4 prompt suites × 64 tokens × 3
trials each.
**Gate:** ≥ 95 sustained, peak ≥ 120 on code prompts.

---

## STAGE 0.5 — MLX-pattern adoption (PARALLEL TRACK, 1-2 weeks)

Decoupled from the K-batching / tree / hardware path. Lifts Off baseline
and cascades to Eagle4 by the head-tax ratio (currently ~0.5).

### Task breakdown

#### 0.5.1 — Profile each hot kernel
**Tool:** `tools/bench/stage0_capture.sh` (exists) + per-kernel
isolated benches.
**Deliverable:** breakdown of where each hot kernel spends its time.

Three kernels to audit (priority):
- `gemv_q4_k_v3` — LM head, biggest single weight read per token
- MoE expert pair matmul — 208 expert evals per token
- MLA decode kernel — already has v3 simdgroup work queued

#### 0.5.2 — gemv_q4_k_v3 rewrite against MLX-LM patterns
**Reference:** `mlx-lm/mlx_lm/models/deepseek_v2.py` LM head kernel.
**File:** `crates/dismantle-core/shaders/gemv_q4_k_v3_mlx.metal` (new)
**Test gate:** atol=1e-3 fp16 vs current `gemv_q4_k_v3`.
**Estimated gain:** Off baseline 27 → ~35-40 tps.

#### 0.5.3 — MoE expert pair matmul rewrite
**Reference:** MLX-LM's MoE forward (fused gate+up+down per expert with
shared SIMD-group register state). dismantle currently calls separately.
**Files:** `crates/dismantle-core/shaders/moe_expert_pair_mlx.metal`
**Test gate:** parity at atol=1e-3 fp16.
**Estimated gain:** Off baseline → +5-10 tps.

#### 0.5.4 — MLA decode kernel finalization (Phase 4 simdgroup queued)
Memory note `mla_phase4_queued.md` says Phase 4 rewrite passes parity
but bench was contention-confounded. Confirm in clean window and merge.
**Test gate:** existing parity tests; clean-window bench ≥ 5 % gain.

---

## Compute scripts to maintain

For each stage's bench, write/update a kickoff script the user runs in
a clean window:

| stage | script | exists? |
|---|---|---|
| Stage 1 remeasure | `tools/bench/stage1_remeasurement.sh` | yes |
| Stage 2 | `tools/bench/stage2_measurement.sh` | **TODO at task 2.7** |
| Stage 3 | `tools/bench/stage3_measurement.sh` | **TODO at task 3.3** |
| Stage 4 | `tools/bench/stage4_measurement.sh` | **TODO at task 4.4** |
| Stage 5 | `tools/bench/stage5_measurement.sh` | **TODO at task 5.6** |

Script template (mirror stage1_remeasurement.sh):
- Bail if Claude.app running
- Pause slm if active
- 3 prompts × 16-64 tokens (longer for later stages — more steady-state
  signal)
- Parse `dec_tps` from `[stats]` line
- macOS notification on completion
- Write `reports/path_to_90/_stage<N>_capture/raw.json` for the resuming
  agent

---

## Halt-recovery decision tree

When a stage measurement halts below band, the agent decides:

```
Stage 2 halts ≤ 30 tps:
  - Most likely: mla_decode_kbatch threadgroup budget bug
  - Chip: re-audit threadgroup memory at K=4
  - Or: chain acceptance collapsed (eagle4 head needs Stage 3 first)
  - Action: re-order — do Stage 3.1 routing recall fine-tune FIRST,
    then return to Stage 2 measurement

Stage 3 halts ≤ 50 tps:
  - Likely: mask recall < 60 % (chip task 3.1 didn't converge)
  - Action: more training epochs in eagle4 venv; document in halt doc

Stage 4 halts ≤ 70 tps:
  - Likely: tree decode hurts on MoE because expert-union grows (per
    deep research § 2). Tree multiplier ~1.4× not 1.8×.
  - Action: tighten DySpec's calib threshold (narrower trees); accept
    landing in lower band

Stage 5 halts ≤ 95 tps:
  - Hardware ceiling near. Verify bandwidth utilization (Instruments
    Metal System Trace) — if < 50 %, more Stage 0.5 work needed.
    If ≥ 70 %, we're memory-bound. Land what we have, document the
    ceiling in stage5_halt.md.
  - This is the most likely halt — 95 is the deep-research floor.
```

---

## Class A / Class B classification

**Class A** (do during the stage that owns them):
- Stage 2: all 4 K-batched kernels + chain spec decode + measurement
- Stage 3: fine-tune + masked verify + measurement
- Stage 4: tree decode + measurement
- Stage 5: AMX extension, Q4 KV, async verify, ANE, multi-queue, measurement
- Stage 0.5: 3 kernel rewrites + clean re-bench

**Class B** (revisit after Stage 5 if needed for the 6-month-horizon):
- SuffixDecoding hybrid (step 29 in original plan)
- Predict-routing-trace via Jakiro decoupling (step 30)
- Medusa-style multi-head stack (step 31)
- Final paper / portfolio writeup (step 32)

---

## Working-tree state preservation

Per CLAUDE.md and recent commits, the user has uncommitted diagnostic
edits in:
- `crates/dismantle-core/src/engine.rs`
- `crates/dismantle-core/src/kernels/mod.rs`
- `crates/dismantle-core/src/model/deepseek_v2.rs` (DBG_FORCE_NONPINNED +
  ffn_shared_only_for_test trait/impl)
- `crates/dismantle-core/tests/ffn_shared_only_nonzero.rs` (untracked)

The agent MUST preserve these across commits using the
strip-restore-restore pattern (remove user hunks, commit roadmap
changes, re-add user hunks). See commits `679c077`, `808d8db`,
`acca22d` for the pattern.

---

## What to commit at end of each architectural session

Always end a session with:

1. A session-closeout doc at
   `reports/path_to_90/session_closeout_<DATE>.md` consolidating:
   - commits landed
   - dec_tps progression
   - what's settled vs open
   - working-tree state at closeout
   - next session's first task
2. Compute-script update if a new bench needs to be run.

Per `CLAUDE.md` § Tone of artifacts — terse, audit-trail focused.

---

## What's NOT in scope

The following are deliberately deferred even though they touch the
codebase:

- **CPU `attention()` divergence chip** — the latent bug in CPU MLA
  attention vs GPU. Independent correctness fix; chip is queued. Does
  NOT affect Eagle4 dec_tps (the production path uses GPU forward).
  Pick up only when the spawned chip is acted on; not blocking.
- **MLX-LM full engine port** — alternative to Stage 0.5 partial. Only
  pursued if Stage 5 halts below 95 AND Stage 0.5 hasn't lifted Off
  baseline to ~50 tps. Different agent session.
- **Q3 / IQ3 quantization sweep** — deep-research § says skip on
  V2-Lite. Don't pursue unless Stage 5 hits a bandwidth ceiling
  needing additional headroom.

---

## First task for the new session

Read this doc end-to-end. Then start at **STAGE 2 task 2.1** (design
doc refresh for K-batched kernels). Effort ~half day. The session can
land tasks 2.1 + 2.2 (gemv_q6_k_v3_kbatch) in a single architectural
push, then request the user run an interim bench script if dec_tps is
worth checking before continuing to 2.3.

If the agent hits the halt rule at any stage, write the halt doc and
stop. Do NOT proceed to a later stage with an open halt above it —
the dependencies are real (K=4 chain decode at Stage 2 unblocks
Stage 3's masked prefetch which unblocks Stage 4's tree decode).
