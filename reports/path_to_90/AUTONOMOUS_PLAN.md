# dismantle — Autonomous plan to 125+ dec_tps

**Purpose:** the *sole* reference document for autonomous agent sessions
executing the path to 125+ Eagle4 dec_tps on M3 Pro 18 GB DeepSeek-V2-Lite
Q4_K_M. Self-contained. Every other plan/closeout/research doc in
`reports/path_to_90/` was deleted on 2026-05-18 — this file folds in
everything still load-bearing. A fresh agent can read this end-to-end and
execute.

**Author intent (user):** "let me know, encourage me." The honest case:
**125+ is plausible; 95-110 is high-probability.** The K-batched substrate
landed in commits `08d4742..2fa02bc`; the wins materialize once the
forward pass actually dispatches them. Stage 0.5 MLX kernel rewrites are
the biggest unrealized headroom — bigger than spec-decode itself on
current data.

---

## 1. Mission + ceiling math

**Target:** ≥125 dec_tps sustained median on V2-Lite Q4_K_M, M3 Pro 18 GB,
Eagle4 K=4 chain spec decode, "The quick brown fox" + Spec-Bench code/MT
prompts, 64-token horizon, 3 trials.

**Hard hardware ceilings (M3 Pro 150 GB/s memory bandwidth):**

Per-token reads on V2-Lite Q4_K_M:
- Shared experts (2 always-on) + attention + LM head + embeddings: **1.0–1.3 GB**
- 6 routed experts × ~85 MB: **~510 MB**
- MLA latent KV (93% smaller than MHA): **50–100 MB**
- **Total: 1.6–1.9 GB/tok → theoretical ceiling 79–94 tok/s at 100% mem efficiency**

llama.cpp Metal achieves 50-65% efficiency; MLX 65-80%. **Realistic single-
stream unaccelerated ceiling on M3 Pro: 45-65 tok/s.** With K-batched
verify (Path B) amortizing weight reads across K=4 queries, effective
per-emit weight read drops to ~0.4-0.5 GB → 300+ tok/s arithmetic ceiling,
practical ceiling 140-160 tok/s (the rest is overhead, KV growth, accept-
rate variance, AMX/ANE overheads).

**Per-stage refined projections (from deleted `eagle4_deep_research.md`,
preserved here):**

| Stage | What it adds | Refined band |
|---|---|---|
| 0 | baseline today (Off mode) | 25-35 tok/s |
| 1 | EAGLE-4 chain, no K-batched verify | **12-22** (current state — regression risk realized) |
| 2 | + Path B K-batched verify (kernels built; wire-up = Phase A) | **38-50** |
| 0.5 | + MLX kernel pattern adoption (Off baseline → 45-55) | **propagates ~0.4× to Eagle4** |
| 3 | + masked prefetch (gated on routing recall ≥60%) | **55-75** |
| 4 | + DySpec tree decode (MoE multiplier 1.4-1.8×) | **70-95** |
| 5 | + AMX further / Q4-KV / async verify / ANE routing / multi-queue | **95-125 sustained, 135 peak on code** |
| 6-mo Class B | + Medusa multi-head + SuffixDecoding hybrid | 130-160 |

**140+ requires:**
- Stage 0.5 hitting top of band (Off ≥55 tok/s) — *bandwidth audit gates this*
- Stage 3 routing recall ≥60% (currently 17.78% — needs fine-tune that hasn't been run)
- Tree decode delivering 1.6× MoE multiplier (1.4× is conservative; depends on Class A 3.2 masked prefetch)
- All Stage 5 levers stacking additively (not all of them will)

**Above 160 needs ANE doing real verify FLOPs.** Published evidence
(NPUMoE arXiv 2604.18788) says ANE can't on MoE due to UMA bandwidth
contention. Treat ANE > 5-10% contribution as research bet, not roadmap.

**Hard physics ceiling: 188 tok/s** at 150 GB/s ÷ 0.8 GB minimum
per-token read. Stop planning above 160-180 sustained.

---

## 2. Starting state (commits landed 2026-05-18b session)

**Branch:** `claude/dreamy-golick-d54ff8`
**HEAD:** `80ea90b` (session closeout 2026-05-18b)
**Worktree:** `/Users/scammermike/Downloads/dismantle/.claude/worktrees/dreamy-golick-d54ff8`

**Commits landed (most recent first):**

```
80ea90b path-to-90 session closeout 2026-05-18b (now superseded by this file)
2fa02bc stage 2 step 2.6: forward_tokens_batched_parallel_k SCAFFOLD + verify_kernels profile flag
cade5a7 stage 2 step 2.5: moe_block_batched_indexed_kbatch_tcb (no-overlap K-batched MoE)
e2b945f stage 2 step 2.4: mla_decode_kernel_fc_kbatch (K-batched MLA, device-scratch scores)
b261b2d stage 2 step 2.3: gemv_q4_k_m_v2_kbatch (K-batched Q4_K_M for attn/MoE projs)
08d4742 stage 2 step 2.2: gemv_f16_lmhead_kbatch (K-batched fp16 lm_head)
adedd4c stage 2 step 2.1: Path B design refresh
```

**Available K-batched kernels (Rust API in `parallel_k.rs`, all parity-tested):**

```rust
// fp16 lm_head: (vocab × hidden) × (K × hidden) → (K × vocab)
gemv_f16_lmhead_kbatch_tcb(tcb, w_buf, rows, cols, x_buf, y_buf, k_batch)

// Q4_K_M attn projections + MoE expert weights (pinned-buffer variant)
gemv_q4_k_m_v2_kbatch_pinned_tcb(tcb, model_buf, w_offset, w_byte_size,
                                  rows, cols, x_buf, y_buf, k_batch)

// MLA decode with K-batched queries against shared KV cache
mla_decode_metal_kbatch(ctx, q_kbatch, c_kv, k_pe, kv_b_proj,
                        n_heads, qk_nope, qk_rope, v_head, kv_lora,
                        seq_len, scale, out_kbatch, k_batch)

// MoE: Rust-side TCB wrapper, K independent MoE forwards in one TCB
moe_block_batched_indexed_kbatch_tcb(tcb, ctx, model_buf, ...per-K slots...)
```

All four kernels accept `k_batch ∈ [1, 8]`. At K=1 each is bit-equivalent
to the existing K=1 kernel by construction.

**Profile flag:** `kernel_profile.selected.verify_kernels`
- `"sequential"` (default, on-disk): existing `forward_tokens_batched_tcb` path
- `"parallel-k"`: routes to scaffold `forward_tokens_batched_parallel_k`
  - K=1 → delegates to sequential (bit-identical)
  - K>1 → returns `Err(Unimplemented)` UNTIL Phase A1 lands the body

**Current dec_tps (single-trial contended, Claude.app open):**
- Off mode: ~27 dec_tps
- Eagle4 mode: ~9-10 dec_tps (CURRENTLY LOSING vs Off — exactly the
  Stage 1 regression the deep research predicted)

**Working tree state (uncommitted user diagnostic edits — MUST preserve):**
```
 M crates/dismantle-core/src/engine.rs           (ffn_shared_only_for_test trait method)
 M crates/dismantle-core/src/kernels/mod.rs      (DBG_Q4KV2_PINNED eprintln)
 M crates/dismantle-core/src/model/deepseek_v2.rs (ffn_shared_only_for_test impl)
?? crates/dismantle-core/tests/ffn_shared_only_nonzero.rs
?? models/, tests/data/*.jsonl, training_data/c2_hidden/* (data dirs)
```

---

## 3. Pitfalls log (lessons paid for in prior sessions)

Read this section before writing code. Every pitfall here cost real time.

1. **`set_bytes` vs `ArgbufRowsCols` slot/struct mismatch** —
   `crates/dismantle-core/shaders/quant.metal`'s `gemm_q4_k_m_fused_v2`
   (and v3/simdmat siblings) declare `constant ArgbufRowsCols& args
   [[buffer(3)]]` (8-byte struct) but the non-pinned dispatchers
   (`dispatch_q4_k_m_gemv_v2` at `kernels/mod.rs:2636`) bind two 4-byte
   `set_bytes` at slots 3+4. Silent UB. Production decode is unaffected
   (uses pinned-tcb path with proper `KernelArgBuffer.bind`). All 11
   tests in `v1k_q4kgemm_simdmat_parity.rs` fail because of this. If a
   new K-batched parity test compares against these non-pinned wrappers
   it will fail spuriously — use a pure-CPU reference (`dequant_into +
   gemv_f32`) instead. Chip queued for an attended fix.

2. **Profile `shader_hash` gates everything.** Adding ANY new MSL file
   (or modifying an existing one) changes the hash of
   `metal::all_shader_sources()`. The next `MetalContext::new()` call
   will fail with "kernel profile shader hash mismatch" because
   `profile.rs::validate_for_gguf` compares the stored hash against the
   live one. After any shader change:
   ```
   cargo build --release -p dismantle
   ./target/release/dismantle shader-hash
   # → copy the new 24-char hex into profiles/deepseek-v2-lite-q4.m3pro18.json's shader_hash field
   ```

3. **V2-Lite lm_head is fp16, not Q6_K.** The "gemv_q6_k_v3" name in
   older docs is aspirational for quantized-lm_head models that don't
   exist on V2-Lite. Production lm_head path is `gemv_f16_simdmat_tcb`
   or `gemv_f16_metal_buf_tcb` (`model/deepseek_v2.rs:2146-2154`); the
   K-batched analogue is `gemv_f16_lmhead_kbatch_tcb` (already shipped).

4. **MLA decode TG memory budget at K=4.** Naive extension of the K=1
   `mla_decode_kernel_fc` scores buffer to K=4 reaches 64 KB at
   seq_len=4096 — busts the 32 KB/core ceiling. Two solutions:
   (a) device-scratch scores buffer (what `mla_decode_kernel_fc_kbatch`
   ships — simpler, slightly more device-mem traffic),
   (b) flash-style online softmax (future optimization if measurements
   show the device-mem traffic matters).

5. **Bench contamination — Claude.app open inflates dec_tps 4-5×.**
   Per memory `bench_contamination.md`. NEVER trust a contended single-
   trial reading. Per-commit smokes during arch work are fine for
   "did I break it" sanity (regression check, not absolute number). REAL
   measurements need user to Cmd-Q Claude.app + optionally pause `slm`
   training if active. See Phase A3 / Phase B5 / Phase F1 below for
   user-action gates.

6. **User has uncommitted diagnostic edits — strip-restore pattern is
   mandatory for files they've touched.** Specifically:
   `engine.rs` (`ffn_shared_only_for_test` trait default),
   `kernels/mod.rs` (`DBG_Q4KV2_PINNED` eprintln),
   `model/deepseek_v2.rs` (`ffn_shared_only_for_test` impl),
   `tests/ffn_shared_only_nonzero.rs` (untracked).
   For any commit touching one of these files:
   ```
   cp <file> /tmp/<file>.bak                  # backup full working state
   git checkout HEAD -- <file>                # restore to HEAD
   # apply MY edits via Edit/Write tools
   git add <file>
   git commit -m "..."
   # restore USER's hunks via Edit tool (they're small + scoped)
   ```
   See commit `2fa02bc` for a working example. NEVER `git add -A` or
   `git commit -a` on these files.

7. **`reports/` is gitignored.** Any new doc in `reports/` needs
   `git add -f`. The closeout/plan workflow is force-add then commit.

8. **CPU `attention()` divergence chip** — there's a known bug in CPU
   MLA attention vs GPU forward (commit `46c71ef`'s halt context).
   Production decode uses GPU forward so it's NOT on the path-to-125
   critical path. Do NOT spend autonomous-session time on it. Chip
   spawned already; attended session decides.

9. **Memory headroom is tight.** V2-Lite Q4_K_M 10.4 GB + Eagle4 head
   ~1 GB + KV at 4K ctx ~1-2 GB + system ≈ 14-15 GB. M3 Pro 18 GB
   leaves narrow margin. **Before any sustained bench, set:**
   ```
   sudo sysctl iogpu.wired_limit_mb=14336
   ```
   Otherwise sustained pressure may trigger swap that destroys
   throughput unrecoverably mid-run.

10. **AMX path must be direct `cblas`, not Core ML.** AMX peaks at 1790
    GFLOPS direct via `Accelerate.framework cblas_sgemm`; through Core
    ML it's 225 GFLOPS (8× slower). The Eagle4 head's 6 gemvs already
    use direct AMX cblas (commit `d1d50fb`). Stage 5.1 extends this to
    V2-Lite's smaller projection gemvs.

11. **ANE for routing logits ONLY.** UMA bandwidth contention kills ANE
    benefit for verify-pass MoE expert kernels (per arXiv 2604.18788).
    Stage 5.4 caps at +5%. Don't try to put verify on ANE.

12. **Q4 MLA-KV is risky.** No published practitioner has shipped Q4 KV
    on V2-Lite or any MLA model. Default to FP16 sinks (first ~32 dims)
    + Q4 rest, gated on ≤1% PPL regression on wikitext-2-256. Stage 5.2.

---

## 3.5 Autonomy charter — full self-judgment within software

You operate without human review between commits. The user is not
watching. You decide. The only hard constraints:

**Inviolable rules** (breaking these = halt):
- Bit-identical greedy gate (`eagle4_decode_parity`) MUST pass at every
  commit. This is the safety net for everything else.
- Production K=1 path (`forward_tokens_batched_tcb`) MUST remain
  unchanged in behavior. Touch by addition only — new methods, new
  routes, but never alter the existing dispatch sequence.
- Commit identity is always `Joshua Hicks
  <joshuahicksboba@gmail.com>` via inline `git -c`. Never `git config`.
  Never Co-Authored-By or "Generated with" trailers.
- User's diagnostic edits (pitfall #6) preserved across every commit.

**Everything else is your judgment.** Specifically authorized:

- **Re-order phases** when data justifies. Example: Phase A measures
  Eagle4 K=4 at only 22 dec_tps, which the decision matrix says is
  underperforming. You decide to insert a quick bandwidth audit
  (Phase B's first step) BEFORE the full Phase B because the audit
  may reveal the bottleneck and let you target Phase B work more
  surgically. Document the re-order in the next commit message.

- **Skip phases** when data says they won't help. Example: Phase E
  (tree decode) returns ≤1.1× over chain in early prototype — the
  MoE tree-decode minefield bit. Skip the rest of Phase E, move
  straight to Phase F. Document why.

- **Insert new investigations** when something doesn't behave per
  plan. Example: a Phase B kernel rewrite passes parity but
  regresses wall-clock. Don't commit; run an isolated bench, decide
  if it's the kernel or the dispatch pattern, write the diagnosis as
  a commit message, ship a corrected version.

- **Update this plan inline** when an assumption is invalidated by
  data. Edit AUTONOMOUS_PLAN.md, add a `## PLAN AMENDMENT YYYY-MM-DD`
  section at top with what changed and why. The plan is a working
  document, not a contract.

- **Choose between two plausible paths** based on which has lower
  wire-up risk + faster validation cycle. Example: AMX extension
  (F1) vs async verify (F3) — both projected +5-8 tps. F3 is
  scheduling, no math change → ship F3 first because the bit-
  identical gate validates trivially.

- **Roll back commits within the current session** if the data says
  to. `git reset --hard HEAD~1` for the most recent commit is OK
  IF you haven't pushed and IF the bit-identical gate failed. Always
  document the rollback in the next commit message.

- **Skip dead ends without writing a halt doc.** If you tried 3
  variants of a kernel rewrite and all under-perform, that's not a
  halt — that's a documented "did not ship" in a closeout. Halts are
  for HARD blockers (build fails you can't fix, correctness gates
  fail you can't root-cause, RSS >5 GB persistent).

**The only thing you cannot do without the user:** run a clean-window
bench. Per pitfall #5, Claude.app contaminates dec_tps 4-5×. Even
"fully autonomous" cannot defeat physics; the GPU is shared. When
you hit a measurement gate (A3 / B5 / C / D4 / E4 / G1), write
the bench script, surface it for the user, and either:
(a) keep working on the next phase's architectural prep (the agent
    does NOT block on user; just queues the measurement),
(b) write a session closeout if you've exhausted parallel arch work
    and need the measurement before deciding the next step.

**Session length:** there is no formal cap. Continue while you have
useful arch work to ship. The natural stop is when you queue 2+
measurement gates that all need the user, OR you've spent >6h
without landing a commit. At that point write the closeout and stop.

## 4. Iteration protocol

Every commit follows this cycle. No exceptions.

```
1. Pick smallest next task from active Phase.
2. Architectural change (Rust / Metal MSL / Python).
3. cargo build --release --workspace                    (must compile clean)
4. cargo test -p dismantle-core --lib --release         (45 lib tests green)
5. cargo test -p dismantle-core --test path_b_parity    (K-batched parity green)
6. cargo test -p dismantle-core --test eagle4_capture_smoke
7. SMOKE: ./target/release/dismantle generate --speculate eagle4 ...
   --max-new-tokens 16 -temp 0; capture dec_tps + accept rate.
   If smoke regresses >2× without explanation → HALT.
8. PARITY: EAGLE4_PARITY_TEST=1 cargo test -p dismantle-core --release
   --test eagle4_decode_parity -- --ignored --nocapture
   (bit-identical Off vs Eagle4 MUST pass at every commit — load-bearing)
9. Selectively stage (skip user diagnostic-edit hunks per pitfall #6).
10. Single-purpose commit as Joshua Hicks via inline git identity:
    git -c user.name='Joshua Hicks' -c user.email='joshuahicksboba@gmail.com' \
        commit -m "path-to-125 <phase> step <K>: <subject>"
11. After 3-5 architectural commits land in a Phase, request user's
    clean-window bench. Read raw.json result, decide next.
```

**Commit message template:**

```
path-to-125 <phase> step <K>: <one-line subject>

<2-3 paragraph what + why>

Smoke (single trial, contended):
  dec_tps=X.X / draft_accepted=N/M

Bit-identical greedy regression: PASSES

What this commit ships:
  - <list of file changes>

What's deferred to next commit:
  - <followup>
```

**Never:**
- Co-Authored-By, "Generated with" trailers (per `~/.claude/CLAUDE.md`)
- `git config` to set identity globally
- `git add -A` / `git commit -a` when user diagnostic files modified
- `--no-verify` / `--no-gpg-sign` / force push
- Modify this plan mid-Phase (treat as authoritative scope per CLAUDE.md)

---

## 5. Autonomous decision matrix

When a measurement gate completes (Phase A3, B5, C2, D4, E2, F1), the
agent reads `raw.json` and decides without asking the user:

| measurement | gate | decision |
|---|---|---|
| Phase A3: K=4 chain Eagle4 ≥ 30 tok/s | floor=30, ceiling=50 | <20: HALT + write blocked doc; 20-30: ship to B but investigate spread; ≥30 proceed to B |
| Phase B5: Off mode ≥ 45 tok/s | floor=35, ceiling=55 | <35: roll back failing kernel(s) per A/B; 35-45 ship + continue; ≥45 proceed to C re-measure |
| Phase C re-measure: Eagle4 K=4 with B kernels | expect 80-110 | <60: investigate; ≥80 proceed to D |
| Phase D4: routing recall mid-train | floor=40%, target=60% | <40%: more epochs OR halt + Python-side blocked doc; ≥60% gate masked variant in E |
| Phase E (masked) measurement | expect ≥5% over D baseline | <2%: skip masked variant; ≥5% ship |
| Phase F (tree) measurement | expect ≥1.3× over chain | <1.1×: skip tree (MoE expert-union grew uncontrolled per deep research); 1.1-1.3 ship + document; ≥1.3 ship |
| Each Phase G hardware lever | A/B vs current | <3% gain: skip + document; ≥3% ship |
| Phase H headline | ≥95 floor, ≥125 target | <80: full investigation; 95-125 SUCCESS state; ≥125 SUCCESS+ |

**Halt conditions (write blocked doc + stop) — NARROW LIST. Most
"things didn't work" are NOT halts; they're documented dead-ends per
§3.5. Halt only when you genuinely cannot make further progress
without the user:**
- Bit-identical greedy gate fails AND you cannot root-cause within
  ~30 min of investigation
- Build fails on `cargo build --release --workspace` AND error is
  outside your edits (e.g., upstream dependency issue)
- Memory pressure (`top -l 1 | grep dismantle | awk` shows RSS >5 GB
  sustained for >5 minutes)
- User-required action gate hit (Cmd-Q Claude.app for bench) AND
  you've exhausted parallel arch work
- Total session time >6 hr without landing a commit (write closeout,
  hand off)

If a kernel rewrite under-performs: NOT A HALT. Document, don't
ship, move on. If a phase doesn't deliver predicted gains: NOT A
HALT. Document, ship what works, move on. If you discover an
infrastructure bug: NOT A HALT (usually). Fix it inline or chip it
via `mcp__ccd_session__spawn_task` and keep working.

**Halt template** (`reports/path_to_90/halt_<phase>_<short>.md`):
```
# Halt — <phase>_<short>
**Halted at:** <iso8601>
**Halted on:** <gate-id>
## Root cause
<one paragraph>
## What ran
<bullet list with hashes>
## What attended work unblocks
<one paragraph + concrete files to inspect>
## Followups
<list>
```

---

## 6. Phase breakdown

### Phase A — Wire up K-batched verify (kernels exist; call them)

**Estimated effort:** 4-8 hours of autonomous agent time
**Predicted gain:** Eagle4 mode from ~9 → 35-50 tok/s
**Risk:** medium-high (touches production forward pass; bit-identical
greedy is the safety net)

#### A1. forward_tokens_batched_parallel_k full body (Stage 2.6 body)

**File:** `crates/dismantle-core/src/model/deepseek_v2.rs`
- Mirror `forward_tokens_batched_tcb` (currently at line 2654) but
  dispatch the K-batched kernels instead of K sequential ones.
- Replace per-K rmsnorm + attention + MoE + lm_head with K-batched
  equivalents.

**Concrete kernel substitutions:**
- K sequential `encode_attention_phase1/2/3` calls → K-batched MLA
  via `mla_decode_metal_kbatch` (one dispatch for all K)
- K sequential `gemv_q4_k_m_v2_pinned_tcb` for q_b/kv_b/q_a/kv_a →
  one `gemv_q4_k_m_v2_kbatch_pinned_tcb` per projection
- K sequential `encode_moe_block_batched_indexed_tcb` → one
  `moe_block_batched_indexed_kbatch_tcb` (no-overlap baseline, K
  forwards in one TCB)
- K sequential `gemv_f16_simdmat_tcb` (lm_head) → one
  `gemv_f16_lmhead_kbatch_tcb`

**Test gates:**
- Compile clean, lib tests pass
- Toggle `kernel_profile.selected.verify_kernels = "parallel-k"`:
  - K=1 still bit-identical greedy (Off vs Eagle4)
  - K=2..4 bit-identical to current sequential-TCB path at same K
    (NEW UNIT TEST: `forward_tokens_batched_parallel_k_matches_tcb_at_k4`)
- Smoke at K=4 Eagle4 still emits valid text

**Strip-restore needed** (touches `deepseek_v2.rs` which has user
diagnostic edit). Follow pitfall #6 protocol.

**Effort:** 2-4 hours including debug. Largest single commit on the
critical path. Split as A1a/A1b if needed: A1a = MLA + lm_head
K-batched, MoE + Q4_K_M projs still sequential; A1b = MoE + projs
K-batched.

#### A2. Eagle4 chain spec decode at K=4 (Stage 2.7)

**File:** `crates/dismantle-core/src/model/deepseek_v2.rs` —
the Eagle4 decode-loop branch in `generate()`.

**Deliverable:**
- Eagle4 head proposes K=4 candidates autoregressively using
  `draft_hidden[step_i]` as `h_high` input for `step_i+1`. `h_low /
  h_mid / h_shared` stay constant (most recent verifier values).
- One `forward_tokens_batched_parallel_k(K=4)` call verifies all 4.
- Longest-matching-prefix accept rule: accept tokens [0..j] where j
  is the first position where `draft[j] != verifier_argmax[j]`. Emit
  `draft[0..j]` + `verifier_argmax[j]` as bonus correction. KV
  rollback to `seq_len_base + j + 1`.

**Test gates:**
- `eagle4_decode_parity` extended to K ∈ {1, 2, 4}, bit-identical at
  each K vs Off mode

**Effort:** 1-2 hours.

#### A3. Stage 2 measurement [USER ACTION REQUIRED]

**File to write:** `tools/bench/stage2_measurement.sh` (mirror
`stage1_remeasurement.sh`):
- Bail with clear message if `pgrep -i "Claude.app"` returns nonzero
- Optionally pause slm if `pgrep -i "slm"` returns nonzero
- 3 prompts × 16 tokens × 3 trials, both Off and Eagle4 K=4 modes
- Parse `dec_tps=X.X` from `[stats]` line
- Write `reports/path_to_90/_phase_A_capture/raw.json`
- `osascript -e 'display notification ...'` on completion

**[USER]** Quit Claude.app (Cmd-Q) then run the script. Script must
fail loudly if Claude is detected.

**Gate (per decision matrix above):**
- ≥30 dec_tps Eagle4 K=4 sustained median → proceed to Phase B
- 20-30 → ship to Phase B but log "Path B underperforming vs deep
  research prediction" in `reports/path_to_90/_phase_A_capture/notes.md`
- <20 → HALT, write `halt_phase_A.md`

---

### Phase B — Lift the Off baseline (MLX kernel patterns)

**Estimated effort:** 6-10 hours autonomous
**Predicted gain:** Off mode 27 → 45-55 tok/s
**Risk:** medium (kernel rewrites with parity tests; lower wire-up risk)
**Why it goes after Phase A:** Phase A proves the K-batching
infrastructure on current kernels. Phase B then 2-3× the underlying
forward, which propagates to Eagle4 via K-batched verify.

#### B1. Bandwidth efficiency audit

**Tool:** Instruments → Metal System Trace; or
`./target/release/dismantle generate --bench-mode` with internal
counters.

**Capture:** per-kernel wall-clock breakdown for one forward pass.
Specifically:
- Per-kernel time as % of forward
- Per-kernel bytes-read (model bytes touched) divided by wall
- Compare against M3 Pro 150 GB/s

**Output:** `reports/path_to_90/_phase_B_audit/efficiency.md` —
table of (kernel, wall_us, bytes_read, GB/s, % bandwidth efficiency).

**Decision:**
- Kernels at <50% efficiency are MLX-rewrite candidates (B2-B4)
- Kernels at >70% are bandwidth-saturated; no rewrite helps

#### B2. gemv_q4_k_v3 rewrite against MLX-LM patterns

**Reference:** `mlx-lm/mlx_lm/models/deepseek_v2.py` LM head kernel
(study via web fetch if needed; the pattern is documented MLX-pub).

**Files:**
- New: `crates/dismantle-core/shaders/gemv_q4_k_v3_mlx.metal`
- New dispatcher in `crates/dismantle-core/src/kernels/mod.rs`

**Test gate:** synthetic parity at `atol=1e-3 fp16` vs CPU reference
(`dequant_into(Q4_K) + gemv_f32`, the slot-correct path).

**A/B gate:** isolated bench shows ≥10% wall-clock improvement vs
existing `gemm_q4_k_m_fused_v2` on V2-Lite expert-projection shape
(rows=10944, cols=2048).

If A/B fails: keep existing kernel, document attempt, move on.

#### B3. MoE expert pair matmul (fused gate+up) rewrite

**Reference:** MLX-LM's MoE forward fuses gate+up+down per expert
with shared SIMD-group register state.

**Files:**
- New: `crates/dismantle-core/shaders/moe_expert_pair_mlx.metal`
- New dispatcher in `crates/dismantle-core/src/kernels/mod.rs`

**Test gate:** parity at `atol=1e-3 fp16` vs the existing
`moe_batched_gemm_q4_indexed_v2t_gu_v2_fc` kernel.

**A/B gate:** ≥5% wall improvement on V2-Lite MoE call pattern.

#### B4. MLA decode Phase 4 simdgroup finalization

Per memory note `mla_phase4_queued.md` (deleted but the fact preserved
here): Phase 4 MLA simdgroup rewrite exists on branch
`claude/mla-phase4-experiment`, passes parity, bench was contention-
confounded.

**Action:** cherry-pick the Phase 4 commit onto current branch,
re-run parity, A/B in clean window.

**A/B gate:** ≥5% wall improvement vs current `mla_decode_kernel_fc`
in a clean window.

#### B5. Off-mode re-measurement [USER ACTION REQUIRED]

**[USER]** Cmd-Q Claude.app, run
`tools/bench/stage2_measurement.sh --off-only` (extend the script).

**Gate (per matrix):**
- Off ≥45 sustained → proceed to Phase C re-measurement
- 35-45 → ship + continue to Phase C
- <35 → roll back any B kernel that underperformed (A/B already
  guards individual commits; this is the integration check)

---

### Phase C — Re-measure Eagle4 with Phase A+B combined

**Estimated effort:** 30 min (no new code, just measurement)
**Predicted gain:** Eagle4 K=4 with B kernels: 80-110 tok/s

#### C1. Combined measurement [USER ACTION REQUIRED]

Same script as A3 + B5, full Off vs Eagle4 K=4 sweep.

**Gate:**
- ≥80 → proceed to Phase D
- 60-80 → proceed to Phase D, document underperformance
- <60 → HALT + investigate (likely Phase B kernel regressions)

---

### Phase D — Routing recall fine-tune + masked prefetch

**Estimated effort:** 4-6 hours autonomous + 1 day Python training
(parallelizable with Phase E if memory allows)
**Predicted gain:** +10-25 tok/s once recall ≥60%
**Risk:** medium-high (training-time path; convergence not guaranteed)

#### D1. Routing-recall fine-tune in eagle4 venv

**File:** `eagle4/eagle4.py` — extend `train` with a
`fine_tune_routing` subcommand or modify existing `train` with
routing-mass-loss weight ≥5.0 (current default per memory is 0.3).

**Hypothesis:** mask head only sees gradient when it's already small.
Bumping routing-mass-loss weight should converge mask recall from
17.78% → 60%+.

**Procedure:**
- Load `eagle4/checkpoints/eagle4_v3/best.npz` as warm start
- Train with routing_mass_loss_weight=5.0, token_ce=1.0, aux_mse=0.5
- 2-3 epochs on existing shards in `training_data/c2_hidden/`
- Save to `eagle4/checkpoints/eagle4_v3/best_recall.npz`

**Mid-training poll:** every 30 min of train wall, eval mask recall
on held-out shard. Decision per matrix.

**Gate:** `python eagle4/eagle4.py eval --ckpt best_recall.npz ...`
reports `mask_topk_mean_recall ≥ 0.60`. Acceptance may drop 2-4pp
from 87.48% → 83-85%; acceptable trade per deep research.

#### D2. moe_block_batched_indexed_kbatch_masked kernel

**Files:**
- New: `crates/dismantle-core/shaders/moe_block_kbatch_masked.metal`
- New dispatcher in `crates/dismantle-core/src/kernels/parallel_k.rs`

**Signature** (extends the no-overlap kbatch):
```rust
moe_block_batched_indexed_kbatch_masked_tcb(
    ...,
    predicted_mask: &PinnedBuffer,  // (N_ROUTED=64) u8 bits
    ...,
)
```

**Deliverable:** kernel accepts predicted_mask from Eagle4 head;
issues `MTLResidencySet.addAllocation` (async prefetch hint) of
mask-bit-set expert weight tiles BEFORE the dispatch needs them.
Prefetch misses fall back to on-demand load (no correctness impact).

**Parity gate:** K=4 masked vs K=4 unmasked on same inputs → bit-
identical at `atol=1e-3 fp16`. The masked path differs only in
dispatch ORDER and prefetch hints; math is identical.

**Wall-clock gate:** with `best_recall.npz` loaded (recall ≥60%):
≥5% wall improvement vs unmasked baseline.

#### D3. Wire predicted mask through decode loop

**File:** `crates/dismantle-core/src/model/deepseek_v2.rs` — the K=4
chain decode path from A2.

**Deliverable:** Eagle4 decode loop passes eagle4 head's
`mask_logits` (26×64) through to `forward_tokens_batched_parallel_k`
as the prefetch hint. `mask_logits` computation REACTIVATED in
eagle4 head Metal/AMX forward (was skipped in commits `0d6a2a3` /
`808d8db` for current-recall-uselessness; recall ≥60% reactivates).

#### D4. Phase D measurement [USER ACTION REQUIRED]

Same script structure as A3.

**Gate:**
- ≥55 Eagle4 K=4 with masked + recall ≥60% → proceed to Phase E
- <55 with recall ≥60% → masked variant underperforms; document and
  skip to Phase E without it
- recall <60% → loop back to D1 with more training epochs

---

### Phase E — DySpec dynamic tree decode

**Estimated effort:** 8-12 hours autonomous
**Predicted gain:** 1.4-1.8× over chain at K=4 → +20-40 tok/s
**Risk:** highest (per deep research, MoE tree decode is a documented
minefield; expert-union grows uncontrolled; Qwen3.6-A3B published
zero net speedup across all tree configs)

#### E1. Tree decode design

**File:** `reports/path_to_90/_phase_E_design.md`

**Design points:**
- Tree shape function `f(calib_logit) → (depth, width)` driven by
  Eagle4's per-position confidence
- Tree attention mask construction (parent-child constraints, not
  diagonal)
- Verify-side accept/reject across tree branches

#### E2. Tree mask + tree attention kernel

**Files:**
- New: `crates/dismantle-core/shaders/tree_attention.metal`
- New: `crates/dismantle-core/src/kernels/tree.rs`
- Parity test in `tests/path_b_parity.rs`

**Test gates:**
- Tree mask reduces to diagonal at K=1 (bit-identical to chain)
- Tree mask at K=2-4 produces bit-identical output to parents-only
  attention at `atol=1e-3 fp16`

#### E3. propose_tree on Eagle4Head + wire-up

**Files:**
- `crates/dismantle-core/src/speculate/eagle4_head.rs`
- New: `crates/dismantle-core/src/speculate/tree.rs`
- `crates/dismantle-core/src/model/deepseek_v2.rs`

**Test gate:** tree mode bit-identical to chain at tree depth 1;
strictly more emits/forward at depth 2-3.

#### E4. Phase E measurement [USER ACTION REQUIRED]

**Gate:**
- ≥1.3× over chain on Spec-Bench code subset → SHIP
- 1.1-1.3 → SHIP behind config flag, default off
- <1.1 → MoE-tree-decode mine hit; skip the variant, lock the chain
  path as production. Document. Move to F.

---

### Phase F — Stage 5 hardware paths (in ROI order)

**Estimated effort:** 4-8 hours autonomous per lever
**Predicted gain (stacked at high end):** +25-40 tok/s
**Risk:** low per-lever (each A/B-gated independently)

Each lever is independent. A/B each one; ship if ≥3% e2e gain;
otherwise document and skip.

#### F1. AMX extend to V2-Lite smaller projection gemvs (~+5-10)
Eagle4 head already uses direct cblas AMX. Extend to V2-Lite
`q_b_proj`, `kv_b_proj` where matrix fits AMX's sweet spot (rows ≤
1024, cols ≤ 2048).

#### F2. Per-head adaptive MLA KV quantization — FP16 sinks + Q4 rest (~+5)
**Pre-gate:** PPL on wikitext-2 first 256 samples ≤1% regression vs
all-FP16. **Block-ship if PPL gate fails.**

#### F3. Async verify-start (~+5-8)
Overlap last Eagle4 draft step's hidden production with first
V2-Lite verify layer's expert prefetch. Still bit-identical (just
scheduling).

#### F4. ANE routing-logits offload (~+5, capped)
V2-Lite's per-MoE-layer router (1, 64) output on ANE concurrent with
Metal verify. UMA contention caps at +5-10% per deep research.
Bit-identical (same gate_logits at `atol=1e-4`).

#### F5. Multi-queue Metal scheduling (~+3-8)
Separate command queue for draft vs verify so dispatch overlaps.
Profile-first.

---

### Phase G — Headline measurement [USER ACTION REQUIRED]

#### G1. Full stack bench

**Script:** `tools/bench/phase_G_headline.sh`
- 4 prompt suites × 64 tokens × 3 trials each
- Off baseline, Eagle4 K=4, Eagle4 tree-decode (if shipped)
- `iogpu.wired_limit_mb=14336` set before run
- Reports per-suite + overall median + p10/p90

**Output:** `reports/path_to_90/_phase_G_headline/raw.json` +
`reports/path_to_90/_phase_G_headline/HEADLINE.md` summary.

**Success criteria:**
- ≥95 sustained median Eagle4 → roadmap floor met
- ≥125 sustained median Eagle4 → roadmap target met
- ≥120 peak on code prompts → bonus

---

## 7. Bench script template

`tools/bench/stage2_measurement.sh` (template; reuse pattern for each
measurement gate):

```bash
#!/usr/bin/env bash
set -euo pipefail

# Bail if Claude.app or another contender is running.
if pgrep -i "Claude" >/dev/null; then
  echo "ERROR: Claude is running. Cmd-Q it first." >&2
  exit 2
fi
if pgrep -f "slm" >/dev/null; then
  echo "WARN: slm process detected — pausing for 30s to let it idle"
  sleep 30
fi

# Memory headroom (one-time per boot; user may need sudo).
CUR=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "0")
if [[ "$CUR" -lt 14336 ]]; then
  echo "WARN: iogpu.wired_limit_mb=$CUR; recommend 14336 before sustained bench"
  echo "  sudo sysctl iogpu.wired_limit_mb=14336"
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WEIGHTS="$REPO_ROOT/models/deepseek-v2-lite-q4.gguf"
PROFILE="$REPO_ROOT/profiles/deepseek-v2-lite-q4.m3pro18.json"
FROZEN_NPZ="$REPO_ROOT/eagle4/v2lite_frozen.npz"
DRAFT_NPZ="$REPO_ROOT/eagle4/checkpoints/eagle4_v3/best.npz"
DISMANTLE="$REPO_ROOT/target/release/dismantle"
OUTDIR="$REPO_ROOT/reports/path_to_90/_phase_A_capture"
mkdir -p "$OUTDIR"

prompts=(
  "The quick brown fox"
  "Write a Python function to compute Fibonacci"
  "Summarize the plot of Hamlet"
)

trials=3
tokens=16

for mode in off eagle4; do
  for prompt in "${prompts[@]}"; do
    for t in $(seq 1 $trials); do
      if [[ "$mode" == "off" ]]; then
        OUT=$($DISMANTLE generate --weights "$WEIGHTS" \
              --kernel-profile "$PROFILE" --prompt "$prompt" \
              --max-new-tokens "$tokens" --temperature 0 2>&1)
      else
        OUT=$($DISMANTLE generate --weights "$WEIGHTS" \
              --kernel-profile "$PROFILE" --prompt "$prompt" \
              --max-new-tokens "$tokens" --temperature 0 \
              --speculate eagle4 --draft-head "$DRAFT_NPZ" \
              --eagle4-frozen "$FROZEN_NPZ" 2>&1)
      fi
      DEC_TPS=$(echo "$OUT" | grep -oE 'dec_tps=[0-9.]+' | cut -d= -f2)
      ACCEPT=$(echo "$OUT" | grep -oE 'draft_accepted=[0-9]+')
      echo "{\"mode\":\"$mode\",\"prompt\":\"$prompt\",\"trial\":$t,\"dec_tps\":$DEC_TPS,\"$ACCEPT\"}"
    done
  done
done | tee "$OUTDIR/raw.jsonl"

osascript -e 'display notification "Phase A bench complete" with title "dismantle"'
```

Customize per-phase: change `OUTDIR`, optionally toggle
`verify_kernels` in profile, optionally adjust tokens/trials.

---

## 8. User-action gates (explicit list)

The agent CANNOT proceed past these without you:

| Gate | What you do | Roughly how long |
|---|---|---|
| Before any sustained bench | `sudo sysctl iogpu.wired_limit_mb=14336` (one-time per boot) | 5 sec |
| Phase A3 measurement | Cmd-Q Claude.app; run `tools/bench/stage2_measurement.sh` | 10 min |
| Phase B5 measurement | Same script, `--off-only` flag | 5 min |
| Phase C measurement | Same script | 10 min |
| Phase D4 measurement | Same script + load `best_recall.npz` head | 10 min |
| Phase E4 measurement | Same script + tree-decode toggle | 10 min |
| Phase G headline | `tools/bench/phase_G_headline.sh` 4 suites × 64 tok × 3 trials | ~30 min |

**Optional but recommended:** if `slm` training process is running on
the machine, pause it during benches (`kill -STOP <pid>`, then
`kill -CONT <pid>` after). The bench script will warn but not block.

---

## 9. What this plan does NOT pursue

Deliberately out of scope:

- **CPU `attention()` divergence chip** — known bug, GPU forward
  unaffected, not on critical path
- **MLX-LM full engine port** — only if Phase B partial doesn't lift
  Off ≥45 tok/s; treat as Phase B fallback, separate session
- **Q3 / IQ3 quantization sweep** — deep research says skip on
  V2-Lite (15% activation ratio amplifies quant noise); only if
  Phase G falls short and Phase F has exhausted gains
- **Continuous batching / multi-request** — different model;
  needed only above 180 sustained
- **Medusa multi-head / SuffixDecoding hybrid** — Class B, 6-month
  horizon

---

## 10. First task for next session

Phase A1 — full `forward_tokens_batched_parallel_k` body.

**Concrete starting moves:**

1. Read `crates/dismantle-core/src/model/deepseek_v2.rs:2654` end to
   end (the existing `forward_tokens_batched_tcb`). Understand every
   dispatch it performs. Note shared-arena buffer usage
   (`arena.batch_x_norm_buf[ki]`).

2. Read the scaffold at `crates/dismantle-core/src/model/deepseek_v2.rs:2845`
   (`forward_tokens_batched_parallel_k`). It's where the body goes.

3. Read the 4 K-batched kernel signatures in
   `crates/dismantle-core/src/kernels/parallel_k.rs`.

4. Read this plan's section 6 Phase A1 and sections 4-5 (iteration
   protocol + decision matrix).

5. Before writing code: backup the file (`cp deepseek_v2.rs
   /tmp/deepseek_v2.rs.bak`), then restore to HEAD (`git checkout
   HEAD -- ...`) to remove user's diagnostic edits, then apply YOUR
   edits, then re-apply user's edits after commit. Pitfall #6.

6. Test at K=1 first (bit-identical greedy passes already; this is
   the canary). Then K=2 (NEW: must match sequential-TCB at K=2).
   Then K=4.

7. If correctness holds, commit + smoke. Then move to A2 (chain spec
   decode). Then A3 (bench script + USER ACTION).

If you hit a halt at any point: write `reports/path_to_90/halt_<phase>_<short>.md`
per section 5's template, and STOP. Do not proceed to a later phase
with an open halt above it.

---

## 11. Tone of artifacts

Per CLAUDE.md § Tone of artifacts: terse, audit-trail focused. No
prose padding. No "we should consider…". The audit trail is for the
next session to act on, not to read.

Closeout docs go at `reports/path_to_90/session_closeout_<YYYY-MM-DD>.md`,
one per session, listing commits landed + dec_tps progression + open
state.

---

**End of plan. Execute autonomously. Halt only on real blockers.**
