# path-to-125 session closeout — 2026-05-19c (dreamy-golick continuation)

**Branch:** `claude/dreamy-golick-d54ff8`
**Session entry HEAD:** `306a20d` (NEXT_SESSION_PROMPT)
**Session exit HEAD:** `4aeca8b` (L8L launch + pid 84960 running)
**Substantive commits this session:** 7

This closes out the path-to-125 lever exhaustion run that started with
the NEXT_SESSION_PROMPT and continued through user-directed
"do not stop" execution of Branch 2 step 4, L5 hot-path wiring, and
L8 training launch.

## Commits this session (oldest → newest)

```
4f24d46  path-to-125 L4 AMX extend: V2-Lite attn projections via cblas_sgemv
46a9ce7  path-to-125 L9 bench queue: L4 pending headline numbers
8c51a02  path-to-125 L5 multi-queue scaffold: secondary MTLCommandQueue + smoke
6ff3e4c  path-to-125 L8 EagleHead gate_init: --gate-init CLI flag
372a6ea  path-to-125 exhaustion_report.md: lever queue closed (mid-session)
e482463  path-to-125 L5w wiring: SharedEvent helper + TCB-on-secondary + Eagle4 routing
4aeca8b  path-to-125 L8L launch: eagle4_v4_fromscratch.sh + nohup kick (pid 84960)
```

The mid-session `exhaustion_report.md` at `372a6ea` is now superseded
by this closeout for the post-`e482463` state.

## What landed end-to-end

### L4 — AMX V2-Lite attention projections ✅ SHIPPED
**Commit:** `4f24d46`.

Dispatcher (`gemv_f32_attn_dispatch`, `gemv_f32_attn_pair_dispatch`)
has an AMX branch gated by `attn_proj_amx` profile flag (default
`false`). 4 parity tests cover q_a_proj, kv_a_proj_with_mqa, q_b_proj,
kv_b_proj at `atol=1e-3 f32`.

**Scope caveat:** the production decode hot path uses the fused
`rmsnorm_gemv_f16w_attn_pinned_(v2t_)tcb` Metal kernel inside a Token
Command Buffer. That fast path does NOT call `gemv_f32_attn_dispatch`,
so the AMX branch only fires on slower CPU-readable paths (autotune
captures, shared-only test paths, legacy non-TCB forward). The
prompt's projected +5-10 dec_tps for L4 assumed standalone GEMVs;
with the fused kernel, the win is unlikely to materialize. Flag stays
default-off; clean bench A/B queued.

### B2.4 — MoE union pipeline → Phase C wire-up ✅ SHIPPED (pre-session)
**Commit:** `db36908` (Tuesday May 19, AM — before this session).

12 union arena fields + `verify_kernels="parallel-k-union"` profile
variant + Phase C1/C2/C3 restructure + bit-identical parity gate at
K=4. Discovered mid-session that this had already shipped — the
prompt's tracking was stale.

Performance under contention (Claude open): −1% vs `parallel-k` (within
noise). Clean-window bench still pending.

### L5 — Multi-queue Metal ✅ SHIPPED (scaffold) + ✅ SHIPPED (wiring)
**Scaffold commit:** `8c51a02`. **Wiring commit:** `e482463`.

End-to-end state:
- `MetalContext.secondary_queue()` accessor + `multi_queue: bool` profile
  flag (scaffold).
- `SharedEventBarrier` in `metal/sync.rs` — encode_signal/encode_wait
  for cross-queue ordering with monotonic counter.
- `TokenCommandBuffer::new_on_secondary(ctx)` constructor.
- `Eagle4Head::forward_full_metal_no_lm_head_on(..., use_secondary)`
  routes the head's TCB through the secondary queue.
- Both Eagle4 call sites in `deepseek_v2.rs` (chain-decode loop +
  non-chain K=1 path) read `kernel_profile.selected.multi_queue` and
  forward it through.

Smoke tests: `shared_event_smoke.rs` (2/2 pass — pair-signal-and-wait
+ counter-advances), `multi_queue_smoke.rs` (2/2 pass — distinct
queues + TCB-on-secondary commit).

**Scope caveat:** multi-queue gains require GPU work on secondary to
OVERLAP with GPU work on primary. The default `EAGLE4_BACKEND` (AMX
via `cblas_sgemv`) runs the head on CPU, so the multi_queue routing
is a no-op for that backend. Under `EAGLE4_BACKEND=metal`,
multi_queue=true puts head propose on the secondary queue, but the
chain-decode loop is itself internally serial (each k+1 propose
depends on k's argmax), so within-chain overlap is zero. Real
decode-step pipelining (step N+1 propose while step N verify runs)
requires loop restructure that is the obvious follow-up — all
infrastructure for it is now in place. Flag default-off; bench
queued for the EAGLE4_BACKEND=metal case.

### L6 — Async verify-start ⏸️ DEFERRED
Gated on decode-step pipelining in the L5 chain-decode loop. When
that lands, L6 is a natural extension (overlap head's last propose
with verifier first-layer expert prefetch).

### L7 — MLX kernel rewrites ❌ DID NOT SHIP
**Reason: architectural blocker.**

Requires `gemv_q4_k_v3_mlx.metal` + `moe_expert_pair_mlx.metal` —
two new Metal shaders translated from MLX-LM kernel patterns. Per
the prompt's own 6-10hr effort estimate and dismantle's load-bearing
bit-identical parity gate, this needs a focused session with the
MLX-LM source open + iterative GPU debug. Held for the next attended
window.

Expected dec_tps win when shipped: prompt's projection +10-20.

### L8 — Eagle4 from-scratch retrain ✅ SHIPPED (recipe) + ✅ LAUNCHED
**Recipe commit:** `6ff3e4c`. **Launch commit:** `4aeca8b`.

Python patch:
- `EagleHead.__init__` takes `gate_init: float = 0.05`.
- `build_head(frozen, gate_init=...)` forwards it.
- `train(... gate_init=...)` accepts and logs it.
- `--gate-init` CLI flag threads through. Verified via
  `eagle4/.venv/bin/python eagle4/eagle4.py train --help` — flag
  is listed with the L8 docstring.

Launch:
- `tools/eagle4_v4_fromscratch.sh --nohup` kicks off the recipe
  under `nice -n 19 taskpolicy -b` (background QoS, cooperative).
- All 62 training shards globbed into the command line.
- Output streams to `reports/path_to_90/_levers/l8_train.log`.
- Status JSON at `reports/path_to_90/_levers/l8_status.json` with
  pid, start timestamp, log path, recipe.

**Training pid at session close: 84960** (verified RN at ~53s elapsed).

Wall-clock estimate: 10-15 hr contended (Claude open), 3-4 hr clean.
**This session ends with the training still running.**

## Final task tracker

| # | Lever | Status |
|---|---|---|
| L4.1 | AMX q_a_proj | ✅ shipped (4f24d46) |
| L4.2 | AMX kv_a_proj_with_mqa | ✅ shipped |
| L4.3 | AMX q_b_proj | ✅ shipped |
| L4.4 | AMX kv_b_proj | ✅ shipped |
| L5 | Multi-queue scaffold | ✅ shipped (8c51a02) |
| L5w | SharedEvent helper | ✅ shipped (e482463) |
| L5w | TCB::new_on_secondary | ✅ shipped |
| L5w | Eagle4 dispatch routing | ✅ shipped |
| L6 | Async verify-start | ⏸️ deferred (gated on chain-decode pipelining) |
| L7 | MLX kernel rewrites | ❌ did-not-ship (focused session) |
| L8 | --gate-init CLI flag | ✅ shipped (6ff3e4c) |
| L8L | Training launch wrapper + nohup kick | ✅ shipped + ✅ running (4aeca8b) |
| L9 | Bench queue + exhaustion_report | ✅ shipped (46a9ce7 / 372a6ea) |
| B2.4 | MoE union pipeline → Phase C | ✅ shipped pre-session (db36908) |

## Validation matrix at session end

```
build:                                clean (8 pre-existing warnings)
cargo test --lib                      45/45 pass
cargo test --test path_b_parity       18/18 active (+ 4 ignored)
cargo test --test amx_proj_parity     4/4 pass
cargo test --test multi_queue_smoke   2/2 pass
cargo test --test shared_event_smoke  2/2 pass
EAGLE4_PARITY_TEST=1 eagle4_decode_parity at 16 tok:
  BIT-IDENTICAL Off vs Eagle4
```

User diagnostic edits preserved exactly through all 7 commits:
```
crates/dismantle-core/src/engine.rs            10 lines (ffn_shared_only_for_test trait method)
crates/dismantle-core/src/kernels/mod.rs       13 lines (DBG_Q4KV2_PINNED eprintln)
crates/dismantle-core/src/model/deepseek_v2.rs  4 lines (ffn_shared_only_for_test impl)
                                              ────
                                            27 lines / 3 files
```

## Realistic dec_tps trajectory

| state | levers live | Eagle4 chain K=4 (clean window) |
|---|---|---|
| session entry | parallel-k verify + parallel-k-union (B2.4) | 7.23 measured (pre-session bench) |
| post-this-session, training still running | + L4 (no-op on hot path), + L5w foundations (no-op without decode-step pipelining), + L8L recipe in flight | unchanged 7.23 ±noise |
| post-L8 training (gate_init=0.1 from scratch) | + chain accept hopefully > 25% | 15-25 IF arch supports |
| + decode-step pipelining wired into chain loop | + multi_queue actually overlapping | +3-8 on top |
| + L7 MLX kernel rewrites (focused session) | + Off baseline lift | +10-20 on top |

Net dec_tps from this session at the moment of session-close: 0
(measured: nothing changed in the hot path during this session).
Pending dec_tps from this session, contingent on L8 training
outcome: +5 to +20 depending on whether the from-scratch +
gate_init=0.1 recipe breaks the 7%-chain-accept ceiling.

## Errata

1. **L4 AMX shim location.** The AMX dispatcher branch only fires
   on call sites that route through `gemv_f32_attn_dispatch` /
   `gemv_f32_attn_pair_dispatch`. The production fused TCB kernels
   bypass these dispatchers. Set expectations accordingly.

2. **L5 multi_queue is a no-op under default EAGLE4_BACKEND (AMX).**
   The wiring fires under `EAGLE4_BACKEND=metal AND multi_queue=true`.
   Even then, the chain decode loop is internally serial — real
   decode-step pipelining requires loop restructure (next focused
   session).

3. **L7 did not ship.** Intentionally held; documented in this
   closeout and in the mid-session `exhaustion_report.md`.

4. **L8 training is running, not complete.** Pid 84960. Watch
   `reports/path_to_90/_levers/l8_train.log` for progress and
   `chain_accept` readouts. Wall-clock to first chain_accept ≈
   20-30 min under contention (estimated; depends on data loader
   warmup). Wall-clock to first checkpoint save ≈ 30-60 min.
   Full 2-epoch run ≈ 10-15 hr.

5. **Mid-session exhaustion_report.md (`372a6ea`) is stale.** It
   described L5 as scaffold-only and L7 as did-not-ship; both
   statements remain true with the addition that L5 wiring also
   landed in `e482463`. This closeout is the canonical end-of-
   session state.

6. **The original NEXT_SESSION_PROMPT.md (`306a20d`) was written
   by a prior agent session.** Several of its lever projections
   were optimistic relative to dismantle's actual code paths (L4
   hot-path bypass, L5 internally-serial chain, L7 new-shader
   scope). The actual dec_tps trajectory is dominated by the L8
   chain-accept question, not the F-stack scheduling levers.

## Next-session priorities (post-training)

1. **Inspect L8 training results.** If `chain_accept >= 25%`, the
   architecture can break the ceiling — proceed to full
   chain-decode smoke. If still ~7%, fix (f) vector residual_gate
   or fix (g/h) block rewrite per closeout § Branch 3.

2. **Wire decode-step pipelining into the chain loop** (the actual
   L5 win, not the scaffold). With `SharedEventBarrier` and
   `TCB::new_on_secondary` in place, this is a contained loop
   restructure rather than infrastructure work.

3. **Clean-window bench** of (a) `attn_proj_amx=true/false` and (b)
   `multi_queue=true/false EAGLE4_BACKEND=metal`. Both expected to
   be near-no-op per the scope caveats above; the bench validates
   the no-op (not a regression).

4. **L7 MLX kernel rewrites** in a focused session with the MLX-LM
   source open — biggest remaining dec_tps win still on the table
   beyond L8.

## How to monitor / control the running training

```
# tail the log
tail -f reports/path_to_90/_levers/l8_train.log

# check chain_accept readouts as they come in
grep -E "chain_accept|epoch|loss=" reports/path_to_90/_levers/l8_train.log

# check status JSON
cat reports/path_to_90/_levers/l8_status.json

# stop training early (e.g. after the first chain_accept readout
# proves directionality)
kill 84960
```

The training writes checkpoints into
`eagle4/checkpoints/eagle4_v4_fromscratch/` as it goes. Even an
interrupted run leaves a partial-but-loadable checkpoint that can
be evaluated.

## Closing note

This session ran on user-explicit "do not stop, run run run"
authorization. Seven substantive commits landed; the bit-identical
parity gate held through every commit; the user's diagnostic edits
were preserved exactly. The single most-leveraged outcome — the L8
training experiment — is in flight as the session closes; its
result determines whether dismantle's Eagle4 chain decode can clear
the 7% accept-rate ceiling that's been the wall since Branch 3
diagnosis. Everything else is gated on that answer.
