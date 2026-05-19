# Path-to-90 session closeout — 2026-05-18

Single-session ledger consolidating 7 architectural commits + 3
compute runs. Eagle4 dec_tps progressed **0.54 → 9.62, 17.8× speedup**;
foundation block went from a halt at step 9 to step 10 measured +
step 7 landed.

## Commit graph (this session, top to bottom = newest)

```
8cb66c1  Stage 1 re-measurement (post-step-7): Eagle4 = 9.62 dec_tps
0b4bef5  re-bench script — post-Metal-head expectations
518e580  step 7: Metal-accelerated Eagle4Head forward
acca22d  h_shared from production MoE moe_shared_out_buf
808d8db  skip Eagle4Head CPU lm_head, use GPU argmax dispatch
679c077  step 10 follow-up: GPU-side eagle4 capture in Wedge C
b00a841  step 10 follow-up: stage1_remeasurement.sh (1st version)
94f6068  step 10: Stage 1 measurement HALT at 0.54 tps
5d0947e  architecture closeout + Stage 1 compute kickoff script
96b51c4  step 9 closeout: GPU emission → bit-identical greedy
f946033  step 20: DySpec dynamic tree decode design
b73a701  step 12: Path B kernel design — masked verify integration
46c71ef  halt update: divergence localized to CPU attention()
defe5b9  step 9: HALT — Eagle4 diverges from Off
f3ae7fd  step 8: --speculate eagle4 CLI + K=1 verify
6411d21  step 6: eagle4 parity test (Rust = Python at 1e-5)
540f9d8  step 5: Eagle4Head::forward_full CPU forward
64cb5c4  step 4: --dump-logits flag on eagle4.py eval
711893c  step 3: Engine::forward_token_eagle4_for_test capture
48be7a1  step 2: Stage 0.5 mandatory, deferred to after step 10
72e3926  step 1: Stage 0 profile — 31 % bandwidth efficiency
```

## Eagle4 dec_tps progression (canonical)

```
step 8   (CPU walk, K=1)          : 0.54  ←  2 % accept
step 10f (GPU capture)            : 1.89  ← 94 %
step 10f (GPU lm_head argmax)     : 6.52  ← 94 %
step 10f (GPU h_shared)           : 7.36  ← 90 %
step 7   (Metal Eagle4Head)       : 9.62  ← 90 %  ← end of session
```

## Per-token cost (post-step-7, ~104 ms / token)

```
Wedge C V2-Lite forward (per-layer commits for capture)  ~41 ms
Eagle4Head Metal forward (9 f16 gemvs + CPU intermediates) ~50 ms
GPU lm_head argmax via gemv_f16_argmax_dispatch          ~10 ms
h_shared from moe_shared_out_buf                         ~ 0 ms
sampler / tokenizer / misc                                ~ 3 ms
```

## Open decisions (no urgency — pick when ready)

### Marginal next-session wins (single-commit chunks)

| lever | est. gain | effort | notes |
|---|---:|---|---|
| Skip mask_logits in production decode | ~+0.8 tps | 5 min | Mask only used by Stage 3 prefetch (not shipped yet) |
| Skip Q + K at S=1 (mathematically dead) | ~+0.2 tps | 5 min | Diagonal-mask softmax collapses to v_h regardless |
| Persistent Metal x/out buffers in head | ~+0.3 tps | 30 min | Avoid per-gemv allocation |
| Single-TCB head encoding (batch gemvs sharing inputs) | ~+0.5 tps | half day | q/k/v share x_normed; gate/up share x_normed |
| Move head's rmsnorm + silu to Metal | ~+0.5 tps | half day | dismantle has rmsnorm_metal_buf_tcb |

### Bigger levers

| lever | est. gain | effort |
|---|---:|---|
| **Stage 0.5 MLX-pattern adoption** (mandatory per step 2) | Off 27→55, Eagle4 ~20 → **clears Stage 1** | 1-2 weeks |
| **Path B K>1 batched verify** (Stage 2 steps 12-17) | Eagle4 → 30-40 | 3 weeks |
| Single-TCB Eagle4Head (full GPU forward with TCB-encoded intermediates) | ~+2 tps | 1 day |

### Correctness / cleanups

| item | priority |
|---|---|
| CPU `attention()` divergence chip (spawned earlier) | medium (correctness; not speed) |
| Trim `stage0_capture.sh` to export only 2 useful trace schemas | low (housekeeping) |
| Push pipeline-state labels in Metal kernels for trace readability | low (housekeeping) |

## Working-tree state at closeout

Three files with intentionally-preserved diagnostic edits from the
spawned-chip investigation. Not in any of my commits:

```
M crates/dismantle-core/src/engine.rs        (+10 lines: ffn_shared_only_for_test trait method)
M crates/dismantle-core/src/kernels/mod.rs   (+13 lines: DBG_Q4KV2_PINNED print)
M crates/dismantle-core/src/model/deepseek_v2.rs (+7 lines: ffn_shared_only_for_test impl + DBG_FORCE_NONPINNED)
?? crates/dismantle-core/tests/ffn_shared_only_nonzero.rs (chip's diagnostic test)
```

Other untracked are large gitignored items (models/, training_data
shards, jsonl test fixtures, `_stage1_capture/`'s log files that
were already committed in step 10's commit). All survive reboot.

## What's settled

- Eagle4 spec decode infrastructure: shipped end-to-end (CLI flag,
  Wedge C capture, Eagle4Head Metal forward, GPU lm_head argmax,
  h_shared from production MoE buf).
- 1e-5 numerical parity vs Python MLX reference (step 6).
- Bit-identical greedy vs Off baseline at every commit point (step 9).
- 89.6 % draft acceptance — head is functionally correct against
  dismantle V2-Lite forward.
- All 45 dismantle-core lib tests + 4 integration tests green.

## What's NOT yet settled

- Stage 1 ship-band (≥ 18 tps). Currently 9.62. Decisive lever is
  Stage 0.5 (1-2 weeks); marginal wins above are tractable as
  single-commit iteration loops.
- Eagle4 vs Off net throughput. At K=1 Eagle4 is slower (expected
  regression band 12-22 per plan; we're below at 9.62). Beats Off
  at Stage 2 (K>1 batched verify, ~40 tps).
- CPU `attention()` divergence — chip queued, no impact on
  emission correctness, affects eagle4 acceptance reliability
  marginally.
