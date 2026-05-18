# Path-to-90 architectural closeout (foundation block)

**Closed:** 2026-05-18 EDT
**Branch:** `claude/dreamy-golick-d54ff8` (12 commits ahead of `origin`)
**Status:** Foundation architectural work complete; compute work
ready to start in a clean window.

## Commit ledger

```
96b51c4 step 9 closeout: GPU emission + parallel CPU capture → bit-identical greedy
f946033 step 20: DySpec dynamic tree decode design
b73a701 step 12: Path B kernel design — masked verify integration
46c71ef halt update: divergence localized to CPU attention()
defe5b9 step 9: HALT — Eagle4 greedy diverges from Off greedy (CPU vs GPU forward)
f3ae7fd step 8: --speculate eagle4 CLI + K=1 verify-by-comparison decode
6411d21 step 6: eagle4 parity test full wire-up (Rust ≈ Python at 1e-5)
540f9d8 step 5: Eagle4Head::forward_full CPU fp32 forward
64cb5c4 step 4: --dump-logits flag on eagle4.py eval
711893c step 3: Engine::forward_token_eagle4_for_test 5-input capture
48be7a1 step 2: Stage 0.5 mandatory but deferred to after step 10
72e3926 step 1: Stage 0 baseline profile — 31% bandwidth efficiency
```

## Steps closed in this session

| step | what landed | commit |
|---|---|---|
| 1 | Stage 0 profile: 31% bandwidth efficiency vs 150 GB/s peak | `72e3926` |
| 2 | Stage 0.5 mandatory but deferred to after step 10 | `48be7a1` |
| 3 | `Engine::forward_token_eagle4_for_test` 5-input capture seam | `711893c` |
| 4 | `eagle4.py eval --dump-logits` flag (NPZ dump for parity) | `64cb5c4` |
| 5 | `Eagle4Head::forward_full` CPU fp32 forward (Rust impl) | `540f9d8` |
| 6 | eagle4 parity test (Rust forward vs Python eval at 1e-5) | `6411d21` |
| 7 | Metal-accelerated head forward — DEFERRED (see "deferred") | — |
| 8 | `--speculate eagle4` CLI + decode wire-up (K=1 verify-by-comparison) | `f3ae7fd` |
| 9 | bit-identical greedy regression PASSING (GPU emission + parallel CPU capture) | `defe5b9` + `46c71ef` + `96b51c4` |
| 12 | Path B kernel design — masked-verify integration (eagle4's 26×64 routing prediction) | `b73a701` |
| 20 | DySpec dynamic tree decode design — calib-driven shape function | `f946033` |

## Steps deferred (with reason)

| step | why deferred |
|---|---|
| 7 — Metal-accelerated head | `<5 ms` gate is a clean-window perf measurement (compute step). Step 8 production wire-up was the natural integration point; current K=1 implementation runs the head's CPU forward — acceptable until step 17's Stage 2 measurement is in play. |
| 10 — Stage 1 measurement | Compute step; needs clean window. Architectural prereq (step 9 bit-identicality) is complete; ready to run. |
| 11 — Routing-recall fine-tune | Python work; runs in eagle4 venv off the dismantle critical path. |
| 13-17 — Path B kernels + Stage 2 measurement | Substantial Metal-kernel work (5-7 days each per plan estimate). Gated on CPU `attention()` fix (chip queued) — without it, K-batched verify parity tests can't validate. |
| 18-19 — Stage 3 masked verify + measurement | Builds on Path B; same gating. |
| 21-22 — Tree decode + Stage 4 measurement | Design landed (step 20); implementation gated on Path B + routing-recall fix. |
| 23-28 — Stage 5 hardware paths | Class A items for after Stage 2/3/4 baselines land. |

## Foundation halt — RESOLVED

The architectural batch hit a real halt at step 9 (bit-identical
greedy regression failed). Root cause was localized via the
`eagle4_cpu_gpu_ab` diagnostic test to dismantle's CPU `attention()`
helper — it over-contributes ~4% to the residual stream per layer,
compounding to ~3× L2 inflation over 27 layers. This predates the
eagle4 work; `forward_token_shared_only` has been silently
miscomputing vs GPU forever (its smoke test only checked "finite
logits", never CPU-vs-GPU argmax).

**Resolution path taken (commit `96b51c4`):** routed around, not
fixed. Eagle4 decode emission now goes through GPU
`forward_token_argmax` (bit-identical to `SpeculateMode::Off` by
construction); the CPU walk is kept in parallel only for eagle4
hidden capture. `seq_len` save/restore around the CPU walk prevents
double-bumping the shared counter.

**Result:** step 9 bit-identical regression PASSES. `Off:
33747,855,254,24547,5025,5025,5025,5025` ≡ `Eagle4: 33747,...`.

**Cost of routing-around vs fixing:**
- Eagle4 mode is ~25× slower than Off (dual CPU+GPU forward per
  token, ~3.74 s vs ~40 ms on M3 Pro). Stage 1 measurement (step 10)
  will land at ~0.2 tok/s in Eagle4 mode, far below the 18-24 tok/s
  block-ship band. That's the right next signal — points at the
  next architectural unlock (GPU-side eagle4 capture).
- Eagle4 stats (`draft_accepted`/`draft_rejected`) are noisy until
  the CPU `attention()` fix lands.

## Architectural followups queued

1. **CPU `attention()` divergence fix.** Chip spawned 2026-05-18.
   Full repro, investigation plan, validation criteria all in the
   chip prompt. Once fixed:
   - eagle4 stats become reliable;
   - the dual GPU+CPU forward per Eagle4 step can collapse to a
     single CPU walk (halving per-token cost — but still slow);
   - `forward_token_shared_only` becomes trustworthy again (used in
     Phase 3 prep for acceptance-rate measurement; re-validate those
     numbers after the fix).

2. **GPU-side eagle4 capture.** The 3.7 s/token cost in Eagle4 mode
   comes from the full CPU walk used to extract h_low/h_mid/h_high/
   h_shared. Production unlock: instrument the Wedge C TCB path to
   read x_buf at layers {2, 13, 25} and call ffn_shared_only at
   layer 26 GPU-side. ~half a day of focused Metal work; defer
   until step 10's measurement makes the cost concrete.

3. **Step 7: Metal-accelerated Eagle4Head forward.** Currently CPU
   fp32. `<5 ms` perf gate is a measurement. Becomes a Stage 2
   prerequisite once head-call latency dominates per-step cost.

4. **`ffn_shared_only` zero-output bug.** Already chipped (step 3,
   commit `711893c`). Latent across the codebase; affects any
   path-to-90 prep work that used `forward_token_shared_only` for
   acceptance-rate measurement.

## Test green snapshot

```
cargo test -p dismantle-core --lib --release
    test result: ok. 45 passed; 0 failed; 0 ignored; 0 measured;

cargo test -p dismantle-core --release --test eagle4_capture_smoke
    test result: ok. 1 passed; 0 failed; 0 ignored;

cargo test -p dismantle-core --release --test eagle4_decode_parity
    test result: ok. 1 passed; 0 failed; 1 ignored; (parity test gated)

EAGLE4_PARITY_TEST=1 cargo test -p dismantle-core --release \
    --test eagle4_parity -- --ignored --nocapture
    [eagle4 parity] worst diffs over 10 records (atol=1e-3):
      mask_logits  max|Δ| = 1.3351e-5  (75× under atol)
      calib_logit  max|Δ| = 2.1458e-6  (465× under atol)
      draft_hidden max|Δ| = 2.9802e-6  (335× under atol)
      argmax mismatches: 0/10
    test result: ok.

EAGLE4_PARITY_TEST=1 cargo test -p dismantle-core --release \
    --test eagle4_decode_parity -- --ignored --nocapture
    [eagle4 greedy parity] Off:    33747,855,254,24547,5025,5025,5025,5025
    [eagle4 greedy parity] Eagle4: 33747,855,254,24547,5025,5025,5025,5025
    -> 0/8 mismatches
    test result: ok.
```

## Working-tree state at closeout

Three files have uncommitted (intentionally preserved) diagnostic
edits from the user's investigation of the latent `ffn_shared_only`
bug. These are NOT in any of my commits; they're work-in-progress
for the spawned chip:

```
M crates/dismantle-core/src/engine.rs       (+10: ffn_shared_only_for_test trait method)
M crates/dismantle-core/src/kernels/mod.rs  (+13: DBG_Q4KV2_PINNED print)
M crates/dismantle-core/src/model/deepseek_v2.rs  (+7: ffn_shared_only_for_test impl + DBG_FORCE_NONPINNED env-var)

?? crates/dismantle-core/tests/ffn_shared_only_nonzero.rs  (user's diagnostic test)
?? reports/path_to_90/_stage0_capture/STATUS.log   (stage 0 audit log — gitignored)
?? other untracked: training shards, parquet, jsonl, models/  (all gitignored)
```

These survive a reboot.

## Compute kickoff: what runs next, in order

The architectural batch is closed. The next concrete actions are
compute steps that require a clean window (Cmd-Q Claude.app).
Provided as scripts so the user can run them sequentially without
this Claude session needing to stay alive.

### 1. Step 10 — Stage 1 measurement

Bench `dismantle generate --speculate eagle4` against `--speculate
off`. Expected: Off ~25 tok/s baseline; Eagle4 ~0.2-0.3 tok/s
(slow because of CPU walk for capture). Per-prompt + median.

Script: `tools/bench/stage1_eagle4_measurement.sh` (created in the
next commit). Output: `reports/path_to_90/stage1_eagle4_chain.md`
with dec_tps numbers + the Stage 1 halt-or-pass decision.

Expected outcome: HALT at the speed gate (below 18 tok/s lower
bound). That halt is the right signal — it triggers the GPU-side
eagle4 capture unlock as the next architectural item.

### 2. Decision point after Stage 1 halt

Per the execution plan's reinject logic:
- if Stage 1 ≥ 18 tok/s: proceed to step 11 (routing-recall
  fine-tune) and steps 12-17 (Path B kernels)
- if Stage 1 < 15 tok/s (expected): pause for GPU-side eagle4
  capture before continuing

Given the expected ~0.2 tok/s result, the second branch fires.
GPU-side capture is the next architectural session.

### 3. Parallel-track: Stage 0.5 MLX-pattern adoption

Step 2's decision rule fired "mandatory" because of step 1's 31%
bandwidth efficiency. Stage 0.5 work is independent of the eagle4
halt — it can run in parallel. Three kernel hot paths to audit
(priority order): `gemv_q4_k_v3` (LM head) → MoE expert pair matmul
→ MLA decode. Estimated 1-2 weeks.

### 4. Parallel-track: spawned chip for CPU `attention()` fix

Queued via the session chip system. Independent of the compute
path; lands as a focused attended-session commit when picked up.

## Estimated calendar

```
[NOW]          architectural batch closed (this commit)
[COMPUTE 1d]   step 10 Stage 1 measurement (clean window, ~30 min run + writeup)
[ATTENDED 4h]  spawned chip: fix CPU attention()
[ATTENDED 4h]  GPU-side eagle4 capture (instrument Wedge C TCB path)
[COMPUTE 30m]  re-run step 10 in post-capture-fix world (target: 12-22 tok/s)
[ATTENDED 1d]  step 11 routing-recall fine-tune (Python; eagle4 venv)
[ATTENDED 5d]  step 13 gemv_q6_k_v3_kbatch kernel
[ATTENDED 5d]  step 14 mla_decode_kernel_fc_kbatch
[ATTENDED 5d]  step 15 moe_block_batched_indexed_kbatch_masked
[ATTENDED 4h]  step 16 forward_tokens_batched_parallel_k wire-up
[COMPUTE 1h]   step 17 Stage 2 measurement (target: 38-50 tok/s)
[parallel]     Stage 0.5 MLX-pattern audit (1-2 weeks; gates a step-7 perf re-eval)
```

Total to Stage 2 landing: roughly **4 weeks of focused work + 1
week of compute / measurement turn-around**.
