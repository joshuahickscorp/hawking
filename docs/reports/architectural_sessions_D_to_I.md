# Additional architectural sessions D-I — copy-paste prompts

Adds 6 more workstreams to the path-to-50/75 effort, each launchable
in its own Claude Code session (or worktree agent) alongside the
already-running A/C agents and Session B (landed). Each prompt is
self-contained.

**Concurrency model:** these sessions are file-isolated (or use
existing branches), so multiple can run in parallel as worktree agents
without conflict. Cargo builds will compete for CPU — stagger or
serialize the actual `cargo build` step if CPU-bound.

**Pause/resume:** any long-running compute that honors
`artifacts/runs/PAUSE` can be halted between stages with
`bash tools/bench/pause_bench.sh` and resumed with
`bash tools/bench/resume_bench.sh`. Architectural code work (Rust
edits) doesn't need pause — it's filesystem-local and finishes fast.

Sessions in priority order:

| # | Workstream | Effort | Gain | Risk |
|---|---|---|---|---|
| D | MLA Phase 4 simdgroup resurrection | ~1 day | +1-3 tps | LOW |
| E | N-gram drafts characterization | ~30 min - 2 days | unknown (possibly +3-8) | VERY LOW |
| F | Operator fusion (RMSNorm + matmul) | 2-3 weeks | +3-7 tps | MEDIUM |
| G | Smaller draft head (5M params) | 1-2 weeks | +5-10 tps | MEDIUM-HIGH |
| H | Spec-decode K-tuning autopilot | 3-4 days | +0.5-2 tps | LOW (depends on B) |
| I | CPU+GPU pipelining audit | 2-3 days | +1-3 tps | LOW |

Recommended launch order:
1. **D + E now** — quick, additive, complete the matrix
2. **I after first bench matrix** — independent, easy to validate
3. **F or G after data shows where the actual bottleneck is** — biggest
   lifts but most expensive; pick based on bench numbers
4. **H after B's runtime cost reduction is measured** — K autotune
   only makes sense when draft cost is right

---

## SESSION D — MLA Phase 4 simdgroup resurrection

--- SESSION D PROMPT ---

You are reviving an already-written MLA Phase 4 simdgroup attention
rewrite that's been parked on a branch with bit-identical parity
proven. This session's job: get it on main, add it to the bench
matrix, confirm tps win. **Expected gain: +1-3 dec_tps** (attention is
2.4% of decode time per memory; phase 4 targets simdgroup utilization).

Working dir: `/Users/scammermike/Downloads/dismantle`.
M3 Pro 18 GB. Baseline ~27 dec_tps clean.

### Critical context

- Branch with the rewrite: `claude/mla-phase4-experiment` (per memory
  `mla_phase4_queued.md`). Confirm with `git branch -a | grep -i mla`.
- The rewrite passes **bit-identical 3-tok greedy parity** vs main per
  memory note. The hold-up was: bench numbers were
  contention-confounded; needed clean-window validation.
- Today's session B added a `STRICT_CLEAN_BENCH=0` flag to
  `tools/training/overnight_path_to_50_bench.sh` — paired deltas with
  Claude open are valid.

### Tasks

1. **Resurrect the branch** (~30 min)
   ```sh
   git fetch
   git checkout claude/mla-phase4-experiment -- crates/dismantle-core/src/attn/
   ```
   (Use `--` to pick only the attention code; preserve everything else
   from main, which has Session B's spec-decode cost reductions.)

2. **Build + run parity** (~10 min)
   ```sh
   cargo build --release -p dismantle
   cargo test -p dismantle-core --test integration_greedy_64 --release
   cargo test -p dismantle-core --test v1_1_phase4D_spec_exact_mode --release
   ```
   Both must be green. If parity breaks: cherry-pick only the attention
   kernel changes (not any shape/dispatch table edits).

3. **Add to bench matrix** (~20 min)
   Edit `tools/bench/path_to_50_matrix.sh` to add a new lever after
   the existing L0-L4:
   ```bash
   # L5: MLA Phase 4 simdgroup attention
   bench_lever "L5_mla_phase4" "MLA Phase 4 (simdgroup attn)" \
     "" # no extra args — opt-in via cargo feature or runtime env
   ```
   If MLA Phase 4 is gated by a feature flag or env var (check the
   branch's docs), thread that through.

4. **Run the matrix** (paired-delta mode is fine)
   ```sh
   STRICT_CLEAN_BENCH=0 bash tools/training/overnight_path_to_50_bench.sh
   ```
   Compare L5 vs L0. Target: **+1 dec_tps minimum** to justify ship.

### What NOT to do

- No git attribution to Claude.
- No autonomous commits.
- Don't take any non-attention changes from `claude/mla-phase4-experiment`
  — that branch is from before Session B's work; merging blindly would
  revert the spec-decode cost reduction.
- Don't enable MLA Phase 4 by default; opt-in flag only until benched
  on a second prompt suite.

### Done condition

- Parity tests green
- Bench matrix shows L5 ≥ L0 + 1 dec_tps
- Memory note: `mla_phase4_landed.md`

End of Session D prompt.

---

## SESSION E — N-gram drafts characterization

--- SESSION E PROMPT ---

You are characterizing the already-implemented `--speculate ngram`
mode. It exists at `crates/dismantle-core/src/speculate/ngram.rs` but
has never been benched in our pipeline. **Expected: possibly a free
+3-8 tps** because n-gram drafts have zero compute cost (just prompt
lookup) — even modest acceptance beats off-mode.

Working dir: `/Users/scammermike/Downloads/dismantle`.

### Tasks

1. **Quick smoke** (~10 min)
   ```sh
   ./target/release/dismantle generate \
     --weights models/deepseek-v2-lite-q4.gguf \
     --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
     --speculate ngram --verify-window 4 \
     --prompt "Once upon a time" \
     --max-new-tokens 32 --seed 0
   ```
   Confirm it generates without crashing and reports `draft_accepted` +
   `draft_rejected` counts.

2. **Per-prompt sweep** (~30 min)
   Test n-gram on 4-5 different prompt types — repetitive
   (story/code) where n-gram lookup hits, vs novel (Q&A) where it
   misses. Document acceptance rates per prompt.

3. **Add to bench matrix** (~10 min)
   Edit `tools/bench/path_to_50_matrix.sh`:
   ```bash
   # L6: N-gram drafts (verify-window 4)
   bench_lever "L6_ngram_K4" "N-gram drafts K=4" \
     --speculate ngram --verify-window 4
   ```
   Maybe also L6b with K=8.

4. **K-tuning** (~1-2 days, if first results are promising)
   N-gram window size matters. Sweep window=3..8 on representative
   prompts. Document the sweet spot.

### What NOT to do

- Don't run benches with Claude live unless `STRICT_CLEAN_BENCH=0`.
- Don't change the ngram implementation — characterize first.

### Done condition

- Bench matrix shows L6_ngram_K4 result for at least 3 prompt types
- Decision documented: ship ngram (and at what K) or shelve
- Memory note: `ngram_drafts_characterized.md`

End of Session E prompt.

---

## SESSION F — Operator fusion: RMSNorm + matmul

--- SESSION F PROMPT ---

You are fusing RMSNorm + matmul (the next op) into a single Metal
kernel. Per memory `per_kernel_time_breakdown.md`, **RMSNorm + add is
24% of decode time** — eliminating dispatch overhead + double memory
reads is +3-7 dec_tps. ~2-3 weeks of focused Metal shader work.

Working dir: `/Users/scammermike/Downloads/dismantle`.

### Tasks

1. **Identify call sites** (~1 day)
   `grep -rn "rmsnorm\|RmsNorm\|rms_norm" crates/dismantle-core/src/`
   to find every RMSNorm dispatch. The high-frequency ones are
   pre-attention and pre-FFN.

2. **Profile baseline** (~1 day)
   `tools/bench/analyze_tcb_trace.py` over a `--trace-dispatch` run
   to confirm where the 24% lives.

3. **Write fused kernel** (~1 week)
   New Metal shader `shaders/rmsnorm_matmul_fused.metal`. Inputs:
   activation + RMS weight + matmul weight (q4/q6/q8). Output: matmul
   result. Fuse the normalize-then-multiply sequence.

4. **Parity** (~2-3 days)
   Bit-identical greedy 3-token gate per
   `feedback_kernel_parity_gate.md`. Wire into existing parity test
   harness.

5. **Bench + tune** (~3-4 days)
   Add to matrix as `L7_rmsnorm_fused`. Tune simdgroup count, threadgroup
   size per shape. Get the +3-7 tps win.

### What NOT to do

- Don't fuse rmsnorm with residual_add only — the matmul is the win.
- Don't break the off-fused fallback; keep it as a dispatcher option.

### Done condition

- Parity green
- Bench shows ≥+3 dec_tps
- Memory note: `operator_fusion_rmsnorm_matmul.md`

End of Session F prompt.

---

## SESSION G — Smaller draft head (5M params)

--- SESSION G PROMPT ---

You are training a 5M-param draft head to replace eagle4's ~30M for
spec-decode. Smaller head = 6× lower draft cost; combined with Session
B's runtime cost reduction, the math flips spec-decode to clearly
net-positive. **Expected gain: +5-10 dec_tps** when wired.

Working dir: `/Users/scammermike/Downloads/dismantle`.

### Critical context

- Eagle5 v2/v3/v4 training all failed due to corpus shape mismatch +
  silent OOMs. Don't try to fix eagle5 — different architecture here.
- 5M params at d_model=2048: roughly 1 transformer block (2 attn
  layers worth). MLX-compatible. Or a simpler n-gram-aware MLP.

### Tasks

1. **Architecture choice** (~1-2 days)
   Two candidates:
   - **Single-block transformer head**: 1 attn layer + FFN, 5M params,
     trained on next-token prediction. Reuses eagle4's training
     infrastructure (sans the broken intermediate-channel-prediction).
   - **MLP head**: 2-layer MLP on residual_in[capture_layer=25].
     Simpler, faster, possibly less accurate.

   Pick based on:
   - eagle4's draft acceptance was 38-48% on factual/code prompts
   - target: 25-35% acceptance at 6× lower draft cost = net positive

2. **Training data** (~1 day)
   Use V2-Lite-Chat's outputs directly (no corpus needed). Generate
   ~10K-100K (prompt, next-token) pairs by running V2-Lite, then train
   the head on those. Avoids the broken intermediate-capture path.

3. **Train** (~5-10 hours compute)
   MLX or PyTorch. Pause-aware (honors artifacts/runs/PAUSE).

4. **Eval — τ-at-depth-4** (~30 min)
   Same eagle4 metric. Target: τ ≥ 2.5 (eagle4 was 3.57; we sacrifice
   some acceptance for lower cost). Plus depth-1 ≥ 65%.

5. **Quantize to int4** (~15 min)

6. **Wire into runtime + bench** (~2 days)
   Add `--speculate small-head --head-path <path>` mode. Bench in matrix
   as `L8_small_head_K4`.

### What NOT to do

- Don't train on the V2-Lite calibration corpus we just built — it has
  the shape issue. Generate fresh (prompt, next-token) data instead.
- Don't try to beat eagle4 on raw acceptance; beat it on net tps.

### Done condition

- τ-at-depth-4 ≥ 2.5
- L8 in bench matrix shows ≥+5 dec_tps over L0 baseline
- Memory note: `small_draft_head_landed.md`

End of Session G prompt.

---

## SESSION H — Spec-decode K-tuning autopilot

--- SESSION H PROMPT ---

After Session B reduces draft cost, the optimal verify-window K
changes. Build an autotune that sweeps K per prompt type and picks the
best. **Expected gain: +0.5-2 dec_tps** (incremental refinement).

Working dir: `/Users/scammermike/Downloads/dismantle`.

### Prerequisite

Session B's spec-decode runtime cost reduction must be **landed and
benched first**. Without that, this session is tuning the wrong knob.

### Tasks

1. **K-sweep harness** (~1 day)
   Script: `tools/training/spec_decode_k_autotune.py`. Sweep
   K=1,2,4,8,12,16 × 3-5 prompt categories (story, code, Q&A, math,
   dialog). Output: median tps per (prompt-category, K).

2. **Per-prompt K classifier** (~2 days)
   Simple heuristic classifier: classify prompt into category by token
   patterns, then route to best K. Or learn it: tiny logistic model
   on prompt features → K.

3. **Wire as runtime flag** (~1 day)
   `--speculate-k auto` uses the classifier. `--speculate-k 4` forces.

4. **Bench in matrix** (~1 day)
   L9_speculate_auto_k. Compare vs fixed-K baselines.

### Done condition

- L9 shows ≥+0.5 dec_tps over fixed K=4
- Per-prompt K decision logged for inspection
- Memory note: `spec_decode_k_autotuned.md`

End of Session H prompt.

---

## SESSION I — CPU+GPU pipelining audit

--- SESSION I PROMPT ---

You are auditing whether dismantle's sampling/tokenization overlaps GPU
forward. **Expected gain: +1-3 dec_tps** if there's an unexploited
overlap opportunity.

Working dir: `/Users/scammermike/Downloads/dismantle`.

### Tasks

1. **Trace** (~1 day)
   Use `DISMANTLE_TCB_TRACE=1` + Instruments.app (or `trace`) to record
   one full decode trace. Look for CPU-idle gaps during GPU work, or
   GPU-idle gaps during CPU work.

2. **Identify** (~1 day)
   Likely candidates:
   - Sampler (argmax/top-k/temperature) waits on GPU logits, then runs
     on CPU, then dispatches next forward. Could the sampler run
     concurrently with the *next* token's forward prefill?
   - Tokenizer (input encode + output decode) — usually pre-emptive
     but worth verifying.
   - KV cache write — usually GPU but check.

3. **Fix the biggest gap** (~3-5 days)
   The fix is usually: move CPU work into a thread that runs while GPU
   is busy. Coordinate via channels.

4. **Bench** (~30 min)
   Add as L10_cpu_gpu_pipelined to the matrix.

### Done condition

- Trace before/after shows reduced idle time
- L10 shows ≥+1 dec_tps
- Memory note: `cpu_gpu_pipelining_landed.md`

End of Session I prompt.

---

## Compute plan with these sessions

When Sessions A and C agents finish (already in flight), Session B is
done, and these new sessions land, the bench matrix has up to 11 levers:

| Lever | Source |
|---|---|
| L0 baseline | always |
| L1 vocab-prune | already done |
| L2 mixed-precision | Session A |
| L3 Q8 KV | Session C |
| L4 spec-decode (cost-reduced) | Session B |
| L5 MLA Phase 4 | Session D |
| L6 n-gram drafts | Session E |
| L7 rmsnorm-matmul fused | Session F |
| L8 small draft head | Session G |
| L9 K autotune | Session H |
| L10 CPU+GPU pipelined | Session I |
| STACK | all enabled at once |

Each lever's bench takes ~1.5 min. The full matrix is ~20-25 min total
compute. Pause-aware so a fast architectural change can interrupt mid-
matrix.
