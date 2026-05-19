# path-to-125 lever exhaustion report — 2026-05-19c

**Session entry:** `306a20d` (NEXT_SESSION_PROMPT)
**Session exit:**  this commit
**Branch:** `claude/dreamy-golick-d54ff8`

This report closes the lever queue defined in
`reports/path_to_90/NEXT_SESSION_PROMPT.md`. Per the prompt's
return-of-control rule, every §3 lever has either shipped, shipped
as a partial scaffold, or has a documented DID-NOT-SHIP entry below.

## Outcomes

| Lever | Status | Commit | dec_tps delta |
|---|---|---|---|
| L4 — AMX V2-Lite attn projections | **SHIPPED (scope-limited)** | `4f24d46` | likely ≈0 on hot path; see note |
| L9 bench-queue tracker | **SHIPPED** | `46a9ce7` | (tooling) |
| L5 — multi-queue Metal | **SHIPPED (scaffold)** | `8c51a02` | 0 by design (no hot-path routing yet) |
| L6 — async verify-start | **DEFERRED** | — | gated on L5 hot-path wiring |
| L7 — MLX kernel rewrites | **DID NOT SHIP** | — | new-shader work; focused session |
| L8 — head from-scratch retrain | **SHIPPED (recipe)** | `6ff3e4c` | training run pending (10-15 hr) |

## L4 — AMX V2-Lite attention projections (SHIPPED, scope-limited)

**Commit:** `4f24d46`.

The dispatcher (`gemv_f32_attn_dispatch`, `gemv_f32_attn_pair_dispatch`)
now has an AMX branch ahead of the Metal path, gated by the new
`attn_proj_amx` profile flag (default `false`). When the flag is on,
attention projections route through `cblas_sgemv` against the f32
weight slice already resident on the layer struct.

Parity gates:
- `tests/amx_proj_parity.rs` — 4 shapes covered (q_a_proj 1536×2048,
  kv_a_proj_with_mqa 576×2048, q_b_proj 3072×1536, kv_b_proj 4096×512).
  All pass at `atol = 1e-3` absolute.
- `eagle4_decode_parity` bit-identical greedy gate passes with the
  flag at its default (`false`).

**Scope limit — important for dec_tps expectations:**

The production decode hot path uses the FUSED rmsnorm + attn-GEMV
kernel `rmsnorm_gemv_f16w_attn_pinned_(v2t_)tcb` enqueued into a
Token Command Buffer. That fast path does NOT route through
`gemv_f32_attn_dispatch`, so this AMX shim only fires on slower
paths (autotune captures, shared-only test paths, the legacy
non-TCB forward). The path-to-125 prompt's +5-10 dec_tps projection
for L4 assumed the projections were standalone GEMVs; with the
fused-kernel hot path, the projected gain is unlikely to materialize.

Wiring AMX into the fused-kernel hot path would require breaking the
TCB chain (commit + CPU sync + AMX call + new TCB start) and likely
costs more than it saves. Left as a follow-up if a bench shows the
slower paths are material.

**Action when bench window opens:**
A/B `attn_proj_amx=false` vs `attn_proj_amx=true` per
`reports/path_to_90/_levers/l9_bench_queued.md`. Expected: no change.

## L5 — Multi-queue Metal scheduling (SHIPPED scaffold; hot-path deferred)

**Commit:** `8c51a02`.

Adds a secondary `MTLCommandQueue` to `MetalContext`, exposed via
`secondary_queue()`. Adds the `multi_queue: bool` profile flag
(default `false`). Smoke test (`multi_queue_smoke.rs`) proves the
two queues are distinct and both accept + complete an empty
command buffer.

**What is NOT in the scaffold:**
1. `MTLSharedEvent` Rust helper (signal/wait at layer boundary).
   The `metal` crate exposes the underlying ObjC type; integration
   is ~50 lines + a `TokenCommandBuffer` constructor that takes a
   wait-event handle.
2. `TokenCommandBuffer::new_on_secondary(ctx)` constructor.
3. Eagle4 chain-decode dispatch routing — head propose to
   secondary, verifier first-layer kv_append + MLA Phase A to
   primary, sync at the layer boundary.
4. Parity gate for the wired path under `multi_queue=true`.

These were not landed inline because each touches the hot path and
demands focused attention; rushing them risks breaking the
bit-identical greedy gate, which is the load-bearing safety net.
The scaffold makes the next session's work mechanical: the queue is
already there, the flag is already there, and the default path is
unchanged.

Expected dec_tps win once wired: prompt's projection +3-8.

## L6 — Async verify-start (DEFERRED per L5-skip clause)

The prompt explicitly states: "Only after L5 lands. F3 uses the
multi-queue infrastructure to overlap head's last propose step with
verifier's first-layer expert prefetch. If L5 doesn't ship
(regression), SKIP L6 and document."

L5 shipped the scaffold but not the hot-path routing. L6's design
overlays on top of that routing, so it has nothing to overlap
against in this session.

Next-session prerequisite: L5 wiring lands (item 1-3 from the L5
"what is NOT in the scaffold" list above), THEN L6 can be designed.

## L7 — MLX kernel rewrites (DID NOT SHIP)

**Reason: architectural blocker.**

L7 requires two new Metal shaders:
1. `gemv_q4_k_v3_mlx.metal` — Q4_K_M GEMV against MLX-LM's
   `deepseek_v2.py` LM head kernel pattern. New shader for the
   V2-Lite expert-projection shape (rows=10944, cols=2048).
2. `moe_expert_pair_mlx.metal` — fuse gate+up+down with shared
   SIMD-group register state, per MLX-LM's MoE forward.

Each requires:
- Source review of MLX-LM's kernel internals (not in this repo;
  needs cross-repo reading).
- Translation of MLX's `mx.fast.metal_kernel` patterns to dismantle's
  `.metal` shader format.
- Per-shape parity at `atol=1e-3 fp16` against the existing fused
  Q4_K_M kernel.
- shader_hash regen (Pitfall #2).
- A/B benchmarking on the actual shape (rows=10944 cols=2048).

The prompt's own 6-10hr effort estimate is realistic. Attempting this
inline alongside the other levers would have meant a half-implemented
shader at session close — exactly the failure mode the system prompt
warns against ("no half-finished implementations"). Held for a
focused session with the MLX-LM source open.

Expected dec_tps win when shipped: prompt's projection +10-20.

## L8 — Eagle4 from-scratch retrain (SHIPPED recipe; training pending)

**Commit:** `6ff3e4c`.

The Python side is patched:
- `EagleHead.__init__` accepts `gate_init: float = 0.05`.
- `build_head` forwards it.
- `train` accepts and logs it (when not resuming).
- `--gate-init` CLI flag wires it through.

The default (`0.05`) reproduces v3 behavior. The from-scratch
experiment is `--gate-init 0.1` WITHOUT `--resume`.

Full from-scratch training command:

```
python eagle4/eagle4.py train \
  --parquet training_data/c2_hidden/eagle4_v0/shard_*.parquet \
  --frozen eagle4/v2lite_frozen.npz \
  --ckpt-dir eagle4/checkpoints/eagle4_v4_fromscratch \
  --epochs 2 \
  --multi-step-k 4 \
  --multi-step-decay 0.7 \
  --chain-h-high \
  --target-warmup-steps 500 \
  --multi-step-aux-decay 0.3 \
  --gate-init 0.1
```

Wall-clock: ~10-15 hr contended, ~3-4 hr clean. Outside session
scope to run; the user launches it (or schedules it).

**Why L8 is the load-bearing lever for chain dec_tps:**

The closeout doc identified that Eagle4 chain decode is capped at
~7% accept rate by the gate-stays-near-0 dynamic. The architecture
CAN accept higher rates if trained with the gate non-trivially live
from step 0. From-scratch + larger gate_init is the lowest-risk
test of that hypothesis. If chain accept climbs past 25%, the path
to 25-35 dec_tps Eagle4 chain decode is open. If still ~7%, the
architecture needs structural fixes (fix (f) vector gate or
(g)/(h) block rewrite per closeout § Branch 3).

Expected dec_tps win when training succeeds: prompt's projection
+5-15 (IF architecture supports — the experiment validates that).

## L9 — Headline bench (queued)

**Commit:** `46a9ce7` (queue note); this commit (final fold-up).

`tools/bench/path_to_125_bench.sh` refuses to run with Claude open,
which it should. Each lever above has its own parity / smoke gate;
the headline dec_tps comparisons are queued in
`reports/path_to_90/_levers/l9_bench_queued.md` for the user to run
during a clean window.

Pending A/B sets:

| Lever | A | B |
|---|---|---|
| L4 | `attn_proj_amx=false` (current default) | `attn_proj_amx=true` |
| L5 | — | n/a until hot-path routing lands |

Run procedure: Cmd-Q Claude, `cd` into the dreamy-golick worktree,
`tools/bench/path_to_125_bench.sh`. Paste numbers back in the next
Claude session.

## Trajectory revision

The path-to-125 prompt projected post-everything-merged at +27-51
dec_tps (sum of L4-L8 mid-estimates) over the current chain decode
baseline of 7.23. The realistic post-this-session expectation is:

- L4 shipped but expected near-zero on hot path → no change.
- L5 scaffold only → no change.
- L7 not shipped → no change.
- L8 recipe shipped but training pending → no change until training.

Net dec_tps from this session: ~0.

Net structural progress: L5 secondary queue is in place (foundation
for the actual +3-8 win); L8 gate_init recipe is in place (gating
the architecture-CAN-or-CANNOT-do-chain-decode question). When
both are followed up, the chain accept ceiling test can run, and
multi-queue can be wired in parallel.

The single most-leveraged next step is launching the L8 training
run. Everything else is bottlenecked on the answer.

## Pitfall compliance

- **Pitfall #6** (user diagnostic edits): preserved exactly through
  all 4 commits. `git diff --stat HEAD` shows
  `engine.rs +10 / kernels/mod.rs +13 / deepseek_v2.rs +4` =
  +27 lines / 3 files, identical to session entry.
- **Pitfall #2** (shader_hash): no new Metal shaders landed this
  session, so no regen needed.
- **Pitfall #7** (`reports/` gitignored): all reports added with
  `git add -f`.

## Commits this session

```
4f24d46  path-to-125 L4 AMX extend: V2-Lite attn projections via cblas_sgemv
46a9ce7  path-to-125 L9 bench queue: L4 pending headline numbers
8c51a02  path-to-125 L5 multi-queue scaffold: secondary MTLCommandQueue + smoke
6ff3e4c  path-to-125 L8 EagleHead gate_init: --gate-init CLI flag
(this)   path-to-125 exhaustion_report.md: lever queue closed
```
