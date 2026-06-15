# All parallel session prompts — one doc, copy-paste ready

Consolidates every parallel architectural workstream into a single
document. Each session is launchable in its own Claude Code session
(or as a worktree agent) and is file-isolated from the others.

Working dir for all sessions: `/Users/scammermike/Downloads/dismantle`.
M3 Pro 18 GB. DeepSeek-V2-Lite-Chat MoE engine.

**Measured paired-delta baseline (Claude live):** L0 ~22 dec_tps.
Confirmed deployable today: vocab-prune + tier_aggressive + ngram K=4 =
~+5 tps. Path-to-50 in clean numbers: ~32 today → ~40 with these
sessions landing → ~50 with the big-rock kernel work.

## Sessions: current status + new launches

| # | Workstream | Status | Effort | Gain | GPU | Parallel-safe? |
|---|---|---|---|---|---|---|
| A | Mixed-precision Path A wedge | 🟡 in worktree `agent-a13b89bf62bacca0e` | 1-2d | +1-3 tps | low | yes |
| B | Spec-decode runtime cost reduction | ❌ LANDED + REGRESSED (-17 tps) | needs revert | n/a | low | yes |
| C | Q8 KV full wiring | 🟡 kernels + parity only; runtime unwired | 2-3d | +2-5 tps | low | yes |
| D | MLA Phase 4 cherry-pick | new | 1d | +1-3 tps | minimal | yes |
| F | RMSNorm + matmul fusion | new | 2-3w | +3-7 tps | low | yes |
| G | Small draft head MVP | new | 1-2w | +5-10 tps (speculative) | **HIGH** | ❌ no (serial GPU) |
| I | CPU+GPU pipelining audit | new | 2-3d | +1-3 tps | low | yes |
| J | MoE GEMM kernel sketch | new | 1w prototype, 2-4w full | +3-15 tps | medium | yes |
| K | Session B revert + investigate | new | 2-3h | restores baseline | low | yes |

**Launch order recommendation:**
1. **Session K first** — fastest, fixes a regression in main's tree
2. **Session C** — completes the half-done Q8 KV (kernels exist, runtime unwired)
3. **Session D** — quick MLA cherry-pick
4. **Session J** — biggest potential, longest tail; start in parallel with the above
5. **Session F** — second biggest potential; queue after J makes progress
6. **Session I** — independent, can run alongside any
7. **Session G** — DEFER until dedicated GPU block (locks MPS for hours)
8. **Session A** — let the in-flight agent finish

## NOTE (2026-05-22 post-run correction)

**`tools/bench/microbench_levers.sh` does NOT exist** in this repo. The prompts below reference it for the lever-by-lever bench protocol. Substitute the paired-bench pattern that Sessions C/F/J confirmed works: 3-trial paired `dismantle generate` runs at 16/64/256 tok, with the lever's env-gate flipped on vs off. See `tools/bench/quick_bench.sh` as the closest existing helper.

Also: prefer to run sessions **SERIAL** on this M3 Pro 18 GB — concurrent cargo builds OOM-killed 5 of 6 parallel agents in the original run. One worktree at a time.

## Hard constraints across ALL sessions

- **No Claude git attribution.** User's global rule. No `Co-Authored-By`,
  no "Generated with Claude" footers.
- **No autonomous commits.** Diffs land in working tree; user reviews.
- **Max 2 concurrent cargo builds.** Each peaks ~2-3 GB on M3 Pro 18 GB.
- **Microbenches serialize.** Use `tools/bench/pause_bench.sh` /
  `tools/bench/resume_bench.sh` to coordinate.
- **No new MPS training while a microbench runs.**
- **Opt-in flags only.** Don't change defaults.

---

## SESSION K — Session B revert + investigation

--- SESSION K PROMPT ---

You are investigating a -17 to -20 dec_tps regression introduced by
Session B's "spec-decode runtime cost reduction" work. Decide:
revert, partial-revert, or fix-forward.

Working dir: `/Users/scammermike/Downloads/dismantle`.

### Context

Session B's writeup: `reports/spec_decode_runtime_cost_2026_05_22.md`.
Summary of B's changes (in main's working tree, uncommitted):
- `crates/dismantle-core/src/model/deepseek_v2.rs`:
  - `DecodeArena.max_batch_size` bumped 8→17 (target: K=16 single-TCB)
  - `ExactShared` verify calls `forward_tokens_batched(&batch_tokens, &batch_positions)`
    instead of K+1 serial `forward_token_argmax` calls
  - `forward_token_argmax` marked dead-code

### Measured (paired-delta, Claude live)

| Lever | dec_tps | Δ vs L0 (22.00) |
|---|---:|---:|
| L0 baseline | 22.00 | — |
| L4_spec_exact_K4 | 5.50 | -16.50 |
| L4_spec_exact_K16 | 1.80 | -20.20 |

The K=16 result (Session B's design target) is WORSE than K=4. That
rules out "wrong K" — the batched verify path itself is broken or
slower than serial.

### Tasks

1. **Bisect the regression** (~30 min)
   ```sh
   git stash push -- crates/dismantle-core/src/model/deepseek_v2.rs
   cargo build --release -p dismantle
   bash tools/bench/microbench_levers.sh   # baseline without B's changes
   git stash pop
   ```
   Compare L4 dec_tps with/without B's changes. Confirms the regression
   is in this file.

2. **Identify root cause** (~1 h)
   Likely candidates:
   - `forward_tokens_batched` may not actually batch — could be calling
     forward K times under the hood, with extra overhead per call
   - `DecodeArena.max_batch_size = 17` may have allocator side effects
     (bigger workspace pinned per decode, fewer arena reuses)
   - The "single TCB fast path" may not be hit because K=4 < 17 threshold

3. **Pick disposition** (~30 min)
   Three options, in order of preference:
   - **Full revert**: `git checkout HEAD -- crates/dismantle-core/src/model/deepseek_v2.rs`.
     Safe; restores serial verify path. Spec-decode goes back to
     ~previous performance (still net-negative, but not -20 tps).
   - **Partial revert**: Keep `max_batch_size=17` for future use, but
     revert the batched-verify call to serial `forward_token_argmax`.
     Restores measured performance; preserves the arena change.
   - **Fix forward**: Profile the actual batched-verify path with
     `DISMANTLE_TCB_TRACE=1`. Find where the extra cost comes from.
     Estimated 1-2 days of investigation.

4. **Re-run microbench** to confirm restored baseline.

### Done condition

- L4_spec_exact_K4 ≥ -2 tps vs L0 (acceptable spec-decode overhead at K=4)
- `reports/session_b_revert_2026_05_22.md` documenting the decision
- Memory note: update `spec_decode_runtime_NOT_broken_2026_05_22.md`
  with the regression finding

Estimated effort: **2-3 hours**.

End of Session K prompt.

---

## SESSION C-completion — Q8 KV full runtime wiring

--- SESSION C-completion PROMPT ---

Session C (prior session) landed Q8 KV kernels (`mla_decode_q8kv_metal`,
`kv_append_q8_0_f32_metal`) + parity tests. **Runtime wiring is
missing** — no CLI flag, no cache allocator integration, no read-path
routing. The kernels are dead code from the user's perspective.

Working dir: `/Users/scammermike/Downloads/dismantle`.

### Verification of what's present

```sh
ls crates/dismantle-core/tests/q8_kv_parity.rs       # parity tests
grep -n "mla_decode_q8kv_metal\|kv_append_q8" \
    crates/dismantle-core/src/kernels/mod.rs          # kernels
```

### What's missing

```sh
grep -rn "q8_kv\|Q8KV\|kv_cache_quant" \
    crates/dismantle/src/main.rs \
    crates/dismantle-core/src/engine.rs \
    crates/dismantle-core/src/cache/  # all return EMPTY
```

### Tasks (in order)

1. **Add `q8_kv: bool` field to `EngineConfig`** (~30 min)
   File: `crates/dismantle-core/src/engine.rs`. Pattern: copy
   `vocab_prune_path: Option<PathBuf>`. Default `false`.

2. **Add `--q8-kv` CLI flag** (~30 min)
   File: `crates/dismantle/src/main.rs`. Plumb through every subcommand
   that takes `EngineConfig` (generate, bench, etc.). Follow the
   `--vocab-prune-path` pattern.

3. **Cache allocator dispatch** (~3-4 h)
   File: `crates/dismantle-core/src/cache/mod.rs`. Add a `KVCacheVariant`
   enum (or similar). When `config.q8_kv == true`, allocate int8 K and
   V storage with per-block scale factors. Else: existing fp16 path.

4. **Write path** (~2 h)
   The decode loop writes new K/V each token. Route to
   `kv_append_q8_0_f32_metal` when cache variant is Q8. Per-block scale
   stored alongside.

5. **Read path: MLA decode** (~3-4 h)
   File: `crates/dismantle-core/src/attn/` or wherever MLA decode
   lives. When cache variant is Q8, dispatch to `mla_decode_q8kv_metal`.
   Wire per-block scales through.

6. **Parity test confirmation** (~30 min)
   `crates/dismantle-core/tests/q8_kv_parity.rs` already exists. Add a
   higher-level integration test that runs a full 64-token generation
   with `q8_kv=true` and confirms output within 1 ULP of fp16 baseline
   at the same seed.

7. **Microbench** (~10 min)
   `./tools/bench/microbench_levers.sh` will auto-detect `--q8-kv`
   flag via its probe. Should show L3_q8_kv as a new lever after
   rebuild.

### Done condition

- `--q8-kv` flag accepts on `dismantle generate`
- Integration parity test green
- Microbench shows L3_q8_kv with measurable Δ (positive or negative
  — both are useful data)
- Memory note: `q8_kv_runtime_landed.md`

Estimated effort: **2-3 days**.

End of Session C-completion prompt.

---

## SESSION D — MLA Phase 4 cherry-pick

--- SESSION D PROMPT ---

Resurrect the MLA Phase 4 simdgroup attention rewrite from
`claude/mla-phase4-experiment` branch. Memory `mla_phase4_queued.md`
notes bit-identical 3-tok greedy parity was proven; held pending
clean-window bench.

Working dir: `/Users/scammermike/Downloads/dismantle`.

### Tasks

1. **Verify branch presence** (~10 min)
   ```sh
   git fetch
   git branch -a | grep -i "mla\|phase4\|experiment"
   ```

2. **Cherry-pick attention kernel changes only** (~30 min)
   ```sh
   git checkout claude/mla-phase4-experiment -- crates/dismantle-core/src/attn/
   git checkout claude/mla-phase4-experiment -- crates/dismantle-core/shaders/attn.metal
   ```
   Do NOT cherry-pick anything outside `attn/`. The branch may have
   stale changes from before Session B; preserve main's current state
   for non-attention code.

3. **Cargo build + parity** (~10 min)
   ```sh
   cargo build --release -p dismantle-core
   cargo test --release -p dismantle-core --test integration_greedy_64
   cargo test --release -p dismantle-core --test v1_1_phase4D_spec_exact_mode
   ```
   Both must be green. If parity breaks: the cherry-pick included
   incompatible changes; selectively re-pick at file level.

4. **Microbench** (~5 min)
   ```sh
   ./tools/bench/microbench_levers.sh
   ```
   MLA Phase 4 may need an opt-in flag or feature; check the branch's
   docs. If no opt-in, the cherry-pick is the activation. Compare L0
   before/after.

5. **Decision** (~5 min)
   - IF +1 tps or more: keep, plan to commit
   - IF ≤ +0.5 tps: revert with `git checkout HEAD -- crates/dismantle-core/src/attn/`
     Document in memory why MLA Phase 4 didn't pay off

### Done condition

- Parity green
- Bench delta documented
- Memory note: `mla_phase4_resurrected.md`

Estimated effort: **1 day**.

End of Session D prompt.

---

## SESSION F — RMSNorm + matmul fusion

--- SESSION F PROMPT ---

Fuse RMSNorm + the next matmul into a single Metal kernel.
**RMSNorm + add is 24% of decode time** per
`memory/per_kernel_time_breakdown.md`. Eliminating per-op dispatch +
double memory reads is +3-7 dec_tps.

Working dir: `/Users/scammermike/Downloads/dismantle`.

### Tasks

1. **Identify call sites** (~1 day)
   ```sh
   grep -rn "rmsnorm\|RmsNorm\|rms_norm" crates/dismantle-core/src/
   ```
   High-frequency: pre-attention norm + pre-FFN norm. Decide which
   downstream matmul to fuse with each.

2. **Trace-dispatch profile** (~1 day)
   ```sh
   DISMANTLE_TCB_TRACE=1 ./target/release/dismantle generate \
       --weights models/deepseek-v2-lite-q4.gguf \
       --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
       --prompt "Once upon a time" --max-new-tokens 8 --seed 0
   ```
   Parse with `tools/bench/analyze_tcb_trace.py`. Confirm 24% figure
   for current build.

3. **Write fused shader** (~1 week)
   New file: `crates/dismantle-core/shaders/rmsnorm_matmul_fused.metal`.
   Inputs: activation, RMS weight, matmul weight (q4/q6/q8). Outputs:
   matmul result. Fuse normalize-then-multiply in shared threadgroup
   memory.

4. **Dispatch route** (~2 days)
   In `crates/dismantle-core/src/kernels/mod.rs`, add a fused-variant
   selector. Gate on a config flag for safety. When disabled, fall
   back to existing two-pass path.

5. **Parity** (~2-3 days)
   Bit-identical greedy 3-token gate (per memory
   `feedback_kernel_parity_gate.md`). Add a parity test
   `crates/dismantle-core/tests/rmsnorm_fused_parity.rs`.

6. **Bench + tune** (~3-4 days)
   Add to microbench as `L7_rmsnorm_fused`. Tune simdgroup count,
   threadgroup size per shape. Target +3 tps minimum.

### Done condition

- Parity green
- Microbench shows ≥+3 dec_tps
- Memory note: `rmsnorm_matmul_fusion_landed.md`

Estimated effort: **2-3 weeks**.

End of Session F prompt.

---

## SESSION I — CPU+GPU pipelining audit

--- SESSION I PROMPT ---

Audit whether sampling/tokenization overlap GPU forward. **Expected
+1-3 dec_tps if there's an unexploited overlap.**

Working dir: `/Users/scammermike/Downloads/dismantle`.

### Tasks

1. **Trace** (~1 day)
   `DISMANTLE_TCB_TRACE=1` + Instruments.app (System Trace template) on
   a 32-token generation. Look for CPU-idle gaps during GPU work, or
   GPU-idle gaps during CPU work.

2. **Identify** (~1 day)
   Likely candidates:
   - Sampler (argmax/top-k/temperature) waits on GPU logits, then runs
     CPU, then dispatches next forward. Could the sampler run
     concurrently with the *next* token's forward prefill?
   - KV cache write — usually GPU but check the synchronization
     primitives.
   - Output detokenizer — usually pre-emptive but worth verifying.

3. **Fix biggest gap** (~3-5 days)
   Common fix: move CPU work to a thread that runs while GPU is busy.
   Coordinate via channels.

4. **Bench** (~10 min)
   Add as `L10_cpu_gpu_pipelined` to microbench.

### Done condition

- Before/after trace shows reduced idle time
- L10 shows ≥+1 dec_tps
- Memory note: `cpu_gpu_pipelining_landed.md`

Estimated effort: **2-3 days**.

End of Session I prompt.

---

## SESSION J — MoE GEMM kernel sketch (the big rock)

--- SESSION J PROMPT ---

Start a custom Metal kernel for V2-Lite's MoE down projection. **MoE
GEMMs are 50.5% of decode time** per
`memory/per_kernel_time_breakdown.md`. Per
`memory/v230_t215_close.md`, q4_gu_v2 is 2.68× faster than q4_v2t —
suggests shader-level wins are accessible.

Working dir: `/Users/scammermike/Downloads/dismantle`.

### Tasks (sketch phase — ~1 week)

1. **Profile the dominant shape** (~1 day)
   What MoE down GEMM shape is most-frequent? Likely
   `(top_k=6) × (hidden=1408) × (intermediate=2048)` or similar. Use
   `DISMANTLE_TCB_TRACE=1` to confirm. The most-frequent shape is
   the optimization target.

2. **Benchmark current kernel at that shape** (~30 min)
   `./target/release/dismantle bench-kernel --shape "1408x2048" --variants all`
   Identify the fastest current variant + its bottleneck (memory
   bandwidth? simdgroup util? register pressure?).

3. **Design new shader** (~1-2 days)
   In `crates/dismantle-core/shaders/`. Start with q4_K_indexed_v3 or
   v4. Hypotheses to test:
   - Larger threadgroup (better memory coalescing)
   - Simdgroup matrix ops (Apple Silicon supports these on M3+)
   - Different expert dispatch granularity (per-expert vs per-batch)

4. **Wire into dispatcher** (~1 day)
   Add new variant name in `crates/dismantle-core/src/kernels/mod.rs`.
   Make it opt-in (env var or config flag).

5. **Parity + microbench** (~1 day)
   Bit-identical 3-token greedy gate. If parity fails: shader bug;
   tune until green.

6. **Decision** (~1 day)
   - IF +3 tps or more: continue with full Phase 2 (other shapes)
   - IF <+1 tps: park; pick a different bottleneck
   - IF -X tps: revert; document why the hypothesis failed

### Done condition (sketch phase)

- One new kernel landed, parity green
- Microbench shows real delta (positive or negative)
- `reports/moe_gemm_kernel_sketch_phase1.md` documenting findings
- Memory note: `moe_gemm_v3_sketch.md`

Full path-to-50: this kernel work is the single biggest lever; expect
to spend 2-4 weeks iterating after the sketch phase.

End of Session J prompt.

---

## SESSION G — Small draft head MVP (SERIAL — GPU-heavy)

--- SESSION G PROMPT (ONLY IN A DEDICATED GPU BLOCK) ---

⚠️ This session locks MPS for 3-10 hours of training. Do NOT launch
alongside other GPU work or active microbenches.

Train a 5M-param MLP draft head that replaces eagle4. Smaller head =
6× lower draft cost; combined with a future spec-decode runtime fix
(separate Session) makes spec-decode net-positive.

Working dir: `/Users/scammermike/Downloads/dismantle`.

### Tasks

1. **Architecture** (~3-4 h)
   `tools/training/small_head.py`. 2-layer MLP on residual_in[layer=25].
   Hidden 2048 → 4096 → 2048 → vocab. ~5M params with quantization.
   PyTorch (not MLX — MLX broke eagle5; PyTorch + MPS is more robust).

2. **Generate teacher data** (~1 h)
   Run V2-Lite as teacher: 10,000 prompts × 256 tokens → (prompt,
   next_token) pairs. Save as compressed numpy. Skip the corpus-build
   path that broke eagle5.

3. **Train** (~3-6 h GPU compute)
   Standard CE loss on next-token. Periodic checkpoints + early stop
   on plateau. Honors `artifacts/runs/PAUSE`.

4. **Eval** (~30 min)
   τ-at-depth-K acceptance rate using V2-Lite as verifier. Target:
   τ ≥ 2.5 at depth 4. Single-step accept ≥ 65%.

5. **Quantize to int4** (~15 min)

6. **Wire into spec-decode runtime** (~2-3 h)
   New mode `--speculate small-head --head-path checkpoints/small_head_q4.npz`.

7. **Microbench** (~10 min)
   `L8_small_head_K4`. Target +5 tps over L0.

### Done condition

- τ ≥ 2.5 at depth 4
- L8 microbench ≥ +5 dec_tps
- Memory note: `small_draft_head_landed.md`

Estimated effort: **1-2 weeks** with debug iterations.

End of Session G prompt.

---

## Launch logistics

For each session you want to run:

1. **New Claude Code session** at `/Users/scammermike/Downloads/dismantle`.
2. Copy everything between `--- SESSION X PROMPT ---` markers for the
   chosen workstream.
3. Paste as the first message.
4. Worktree agents are file-isolated. Cargo builds on different sessions
   compete for CPU/RAM — keep concurrent builds ≤ 2.

**Coordinating microbenches across sessions:**
- Before running `tools/bench/microbench_levers.sh`, check:
  `pgrep -fl "microbench_levers|dismantle generate"` should be empty.
- If another session is benching, run
  `bash tools/bench/pause_bench.sh` — it'll halt at the next trial
  boundary. Then resume after your own bench.

**Done condition for the whole effort:** path-to-50 hit on a clean
bench (Claude quit). Wrap memo at
`~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/path_to_50_complete_v2.md`.
