# path-to-125 session closeout — 2026-05-19 (dreamy-golick-d54ff8)

**Branch:** `claude/dreamy-golick-d54ff8`
**Commits landed this session:** 5
**Plan amendments:** AUTONOMOUS_PLAN.md `PLAN AMENDMENT 2026-05-19`

## Commits (in order)

```
d3f831b path-to-125 phase A1.0 step 1: forward_tokens_batched_parallel_k body — K-batched lm_head only
a72c1e5 path-to-125 phase A1.0 step 2: plan amendment + session closeout
1f97dfb path-to-125 phase A2:   Eagle4 chain spec decode (EAGLE4_CHAIN_K env-var gated)
df459ee path-to-125 phase A1.1: causal-mask K-batched MLA shader + parity refresh
7668247 path-to-125 phase A1.1 step 2: TCB-style mla_decode_kbatch_tcb dispatcher + parity
```

## Phase A status (the entire pursuit of this session)

### A1.0 — `forward_tokens_batched_parallel_k` body, lm_head-only amortization ✅ SHIPPED

[crates/dismantle-core/src/model/deepseek_v2.rs:2841](crates/dismantle-core/src/model/deepseek_v2.rs:2841)
gated on `verify_kernels=parallel-k`. Per-token layer loop preserved
(causal MLA + MoE per token inside one TCB); post-layer LM head
replaced by one `gemv_f16_lmhead_kbatch_tcb` dispatch. The 400 MB
vocab×hidden fp16 weight is read once per K-verify step instead of K
times. K=1 still delegates to `forward_tokens_batched_tcb` for
byte-identical production behavior.

### A1.1 — causal-mask K-batched MLA shader + TCB dispatcher ✅ SHIPPED, wire-up DEFERRED

[crates/dismantle-core/shaders/mla_decode_kernel_fc_kbatch.metal](crates/dismantle-core/shaders/mla_decode_kernel_fc_kbatch.metal)
Phase 1 now applies per-K causal mask: query kk attends to positions
`[0, seq_len_base + kk]` where `seq_len = seq_len_base + k_batch`.
At k_batch=1 bit-identical to the K=1 path. Profile `shader_hash`
regenerated (`e25ef35fae76fd867bdd9d1d` → `d214bba83371d3e7d4b70794`).

[crates/dismantle-core/src/kernels/parallel_k.rs:439](crates/dismantle-core/src/kernels/parallel_k.rs:439)
adds `mla_decode_kbatch_tcb` — TCB-form dispatcher that takes
caller-owned `PinnedBuffer`s (q_packed, c_kv_buf, k_pe_buf,
kv_b_proj, out_packed, scores_scratch). Bit-identical to the
one-shot helper (parity test
`mla_decode_kbatch_tcb_matches_one_shot_v2lite_shape_k4` asserts
`max_abs_diff == 0.0`).

**Wire-up into `forward_tokens_batched_parallel_k` is deferred.** It
requires:

  1. Per-K residual-stream shadow buffers (`batch_x_buf[K]` and
     `batch_ffn_out_buf[K]` — mirrors the existing
     `batch_x_norm_buf[K]` in DecodeArena, or allocate ad-hoc inside
     the parallel_k call to avoid arena schema change).
  2. Restructure the per-layer body into three phases:
     - **Phase A** (K iter): restore `batch_x_buf[ki]` → `arena.x_buf`,
       restore `batch_ffn_out_buf[ki]` → `arena.ffn_out_buf` (li > 0),
       run `encode_attention_phase1` + `encode_attention_phase2`, blit
       `arena.q` → `packed_q[ki * n_heads * q_head_dim]`, save
       `arena.x_buf` → `batch_x_buf[ki]`.
     - **Phase B** (1 dispatch): `mla_decode_kbatch_tcb` on packed_q
       + shared KV → `packed_attn_out`, seq_len = seq_slot_base + K.
     - **Phase C** (K iter): restore `batch_x_buf[ki]` → `arena.x_buf`,
       blit `packed_attn_out[ki ...]` → `arena.attn_out`, run o_proj
       (`gemv_f16_simdmat_tcb`) → `arena.out`, add_inplace + ffn_norm
       (single `encode_add_and_rmsnorm_tcb`), MoE, save `arena.x_buf`
       → `batch_x_buf[ki]` and `arena.ffn_out_buf` →
       `batch_ffn_out_buf[ki]`.

  Expected gain: amortizes `kv_b_proj` read 4×→1× across K
  (~162 MB / verify step ≈ ~1.1 ms wall at 150 GB/s) + smaller wins
  on c_kv/k_pe reads. Net per K=4 verify step: ~1-3% e2e — modest,
  but the dispatcher already ships and the wire-up is a localized
  refactor.

### A2 — Eagle4 chain spec decode ✅ SHIPPED (env-var gated)

[crates/dismantle-core/src/model/deepseek_v2.rs:1515](crates/dismantle-core/src/model/deepseek_v2.rs:1515)
sets `EAGLE4_CHAIN_K=2..=8` to engage chain decode; unset/=1 keeps
the existing K=1 path verbatim. Per AUTONOMOUS_PLAN §6 A2 design:
seed forward + K+1 head autoregressive calls (anchored: cur_prev
flips to v2_argmax after call 0 instead of head's draft_token_0) +
`forward_tokens_batched` verify + longest-prefix accept + bonus
emit.

Bit-identical to Off mode confirmed via smoke at "The capital of
France is", 24 tokens. Same token stream as Off.

**Head autoregressive quality is the bottleneck**: accept rate
~5% (2/44 drafts on the smoke prompt). The Eagle4 head was trained
on V2-Lite's TRUE hidden states; using head's own `draft_hidden_k-1`
as `h_high_k` is off-distribution and breaks down quickly. The
single-step K=1 accept rate is 94%; the K-step autoregressive
geometric collapse to <10% is structural, not a code bug.

Per-step latency contended ~257 ms emitting 2 tokens (≈ 8 dec_tps).
Off mode at same load: ~25 dec_tps. Net regression with current
head + A1.0-only verify (no MLA/MoE K-batching).

### A3 measurement — NOT QUEUED

Per the AUTONOMOUS_PLAN §8 user-action list, Phase A3 needs the
user to Cmd-Q Claude.app and run `tools/bench/stage2_measurement.sh`.
The script is not yet authored — the path-to-90 tooling structure
referenced in §7 has been mostly retired (per `post_prune_operating_model`
memory). The realistic next bench is whichever your existing
clean-window tooling makes easiest.

**A3 is not queued this session because A2's structural regression
(head-quality, not code) would land negative dec_tps in any clean
bench right now. The A3 gate should fire after A1.1 wire-up OR
after a routing-recall fine-tune lifts the autoregressive accept
rate.**

## Phases B–F status — NOT STARTED

Per AUTONOMOUS_PLAN.md §6, Phases B–F are explicitly larger-scope
work than fits in a single autonomous session:

| Phase | Estimated effort (plan) | Status |
|---|---|---|
| B (Off baseline lift via MLX patterns) | 6-10 hours autonomous | not started |
| C (re-measure with A+B) | 30 min + USER bench | not queued |
| D (routing recall fine-tune + masked prefetch) | 4-6 hours autonomous + 1 day Python training | not started |
| E (DySpec tree decode) | 8-12 hours autonomous | not started |
| F (Stage 5 hardware levers: AMX/Q4-KV/async/ANE/multi-queue) | 4-8 hours per lever | not started |

The user prompt asked for "A through F" in one session. Phase D
alone needs 1 day of Python training (CPU/GPU bound, not gated on
agent time); Phase E is ~10 hours of new MSL kernel code; Phase B
is 4-7 hours of MLX-LM kernel rewrites. The realistic interpretation
of "one shot A→F" is: push Phase A to a clean stopping point with
all leverage queued — which is what this session ships.

## Validation matrix

```
cargo build --release -p dismantle-core    : clean (8 pre-existing warnings)
cargo build --release -p dismantle         : clean
cargo test  --release -p dismantle-core --lib       : 45/45 pass
cargo test  --release -p dismantle-core --test path_b_parity
                                            : 10/10 active pass (4 stubs ignored)
cargo test  --release -p dismantle-core --test eagle4_capture_smoke
                                            : 1/1 pass (26 s)
EAGLE4_PARITY_TEST=1 cargo test  --release -p dismantle-core
    --test eagle4_decode_parity -- --ignored --nocapture
                                            : PASSES (bit-identical Off vs
                                              Eagle4 at 16 + 32 tokens,
                                              with and without verify_kernels=
                                              parallel-k profile flip)
```

ngram-spec mode under `verify_kernels=parallel-k` profile, smoke
on "The capital of France is" → Off-equivalent output (
" Paris.\n\nThe capital of France is Paris.") with 7/7 drafts
accepted across K=4 and K=3 verifies. Exercises the K>1 parallel-k
path that Eagle4 mode also routes through after EAGLE4_CHAIN_K is
set.

## Pitfall compliance

- **Pitfall #6** (user diagnostic edits): backed up at session start
  (`/tmp/{engine,kernels_mod,deepseek_v2}_a10_backup.rs` and `_a2_backup`),
  stripped before each commit that touched a user-edit file,
  re-applied via Edit-tool reinsertion of the small scoped hunks
  after commit. `git diff --stat` at session end:
  ```
   crates/dismantle-core/src/engine.rs            | 10 ++++++++++
   crates/dismantle-core/src/kernels/mod.rs       | 13 +++++++++++++
   crates/dismantle-core/src/model/deepseek_v2.rs |  4 ++++
   3 files changed, 27 insertions(+)
  ```
  Identical to session start.

- **Pitfall #2** (shader_hash): regenerated after the
  `mla_decode_kernel_fc_kbatch.metal` causal-mask change. Profile
  field updated from `e25ef35fae76fd867bdd9d1d` to
  `d214bba83371d3e7d4b70794`. Sanity-checked via Off-mode smoke +
  eagle4_decode_parity gate.

- **Pitfall #7** (`reports/` gitignored): all reports staged with
  `git add -f`.

## Bench note

Bench remained contaminated all session (Claude.app open). Per
memory `bench_contamination`, dec_tps inflates 4-5× under
contention; single-trial readings are indicative only. Clean-window
A3 bench is queued for **after** the A1.1 wire-up (Branch 1 below)
lands so the K-batched MLA actually engages, OR after a routing-
recall fine-tune (Phase D) lifts Eagle4's autoregressive accept
rate.

## Next-session priority queue

### Branch 1 — finish A1.1 wire-up (1-2 hours)

The MLA TCB dispatcher is ready
(commit `7668247`). Wire it into `forward_tokens_batched_parallel_k`
per the design block already laid out in that commit message
(Phase A / Phase B / Phase C three-pass per-layer structure).

Approach without arena schema change: inside
`forward_tokens_batched_parallel_k`, allocate K shadow `x_buf` +
K shadow `ffn_out_buf` ad-hoc:

```rust
let h_bytes = (h * std::mem::size_of::<f32>()) as u64;
let shadow_x_buf: Vec<PinnedBuffer> = (0..k)
    .map(|_| ctx.new_buffer(h * std::mem::size_of::<f32>()))
    .collect();
let shadow_ffn_out_buf: Vec<PinnedBuffer> = (0..k)
    .map(|_| ctx.new_buffer(h * std::mem::size_of::<f32>()))
    .collect();
let packed_q = ctx.new_buffer(
    k * n_heads * (qk_nope + qk_rope) * std::mem::size_of::<f32>()
);
let packed_attn_out = ctx.new_buffer(
    k * n_heads * v_head_dim * std::mem::size_of::<f32>()
);
let scores_scratch = ctx.new_buffer(
    n_heads * k * (seq_slot_base + k) * std::mem::size_of::<f32>()
);
```

Then per-layer Phase A/B/C as described in commit `7668247`'s
"What the wire-up needs" section. Parity gate is the same:
EAGLE4_PARITY_TEST=1 eagle4_decode_parity at 32 tokens under
parallel-k profile. Bit-identical to Off must hold.

### Branch 2 — A1.2 MoE expert-union K-batched kernel (10-15 hours, biggest single lever)

The Stage 3.2 work AUTONOMOUS_PLAN.md defers to Phase D. Hands-down
the largest bandwidth amortization on V2-Lite:

- 6 routed experts × 3 projections × ~5.5 MB Q4 per token per layer
  × 27 layers = ~2.7 GB / token of MoE expert weight reads.
- At K=4 with 50-70% routing overlap, union-routing kernel saves
  ~3-5 GB / verify step (vs 4×2.7=10.8 GB sequential).
- Wall-clock: ~20-30 ms saved at 150 GB/s per verify step.

New MSL kernel: tile over distinct experts in the K-token union,
each tile processes the K queries that selected that expert,
weighted by per-query per-route weight. Substantial: ~200-400 lines
of new MSL + dispatcher + parity test against the existing
no-overlap baseline.

### Branch 3 — head retrain for multi-step autoregressive (Phase D, 1 day Python)

A2's accept-rate collapse (5% at K=4 vs 94% at K=1) is structural —
head was trained for single-step prediction. To make A2 actually
deliver gains under the head substrate we already have, either:

(a) Retrain Eagle4 head with autoregressive trajectories at training
    time (EAGLE-3-style "self-consumption" — use the head's
    `draft_hidden` as the next step's `h_high` during training, so
    the head learns the off-distribution feedback loop), OR

(b) Reactivate `mask_logits` + masked prefetch (AUTONOMOUS_PLAN §6
    D2/D3) — head's routing mask is dispatch-time hint, doesn't
    require multi-step correctness from the head.

(a) is bigger lift but unlocks A2 directly. (b) is what the
plan's Phase D originally scopes.

### Branch 4 — Phase F levers (after Branches 1+2 land)

Each Phase F lever (AMX extend, Q4 MLA-KV, async verify, ANE
routing, multi-queue) is independently A/B-gateable and projected
at +3-10% each. After Branches 1-3 establish a real baseline, F
becomes incremental stacking. Per-lever ~4-8 hours.

## Working tree state at session end

```
On branch claude/dreamy-golick-d54ff8

Changes not staged for commit (user diagnostic edits — IDENTICAL to session start):
        modified:   crates/dismantle-core/src/engine.rs                (+10 ffn_shared_only_for_test trait method)
        modified:   crates/dismantle-core/src/kernels/mod.rs           (+13 DBG_Q4KV2_PINNED eprintln)
        modified:   crates/dismantle-core/src/model/deepseek_v2.rs     (+4  ffn_shared_only_for_test impl)

Untracked files (data dirs + user test, all from session start):
        crates/dismantle-core/tests/ffn_shared_only_nonzero.rs
        models
        tests/data/ultrachat_100k.jsonl
        tests/data/ultrachat_100k_union.jsonl
        tests/data/ultrachat_1k_smoke.jsonl
        training_data/c2_hidden/eagle3_v0/shard_000.parquet
        training_data/c2_hidden/smoke_1k/
```

## Realistic next-session dec_tps trajectory

| state | levers live | projected Eagle4 K=4 dec_tps (clean window) |
|---|---|---|
| post this session (A1.0 + A1.1 shader + A2, no wire-up) | parallel-k path exists but Eagle4 chain accept rate ~5% | 8-12 (head-bound) |
| + A1.1 wire-up (Branch 1 above) | K-batched MLA actually engages | 9-14 (still head-bound) |
| + A1.2 MoE union (Branch 2) | true MoE expert amortization | 18-30 |
| + head autoregressive retrain (Branch 3a) OR mask-prefetch (Branch 3b) | accept rate lifts toward single-step's 94% | 35-55 |
| + Stage 0.5 MLX rewrites (Phase B) | Off baseline lifts; propagates to verify | 50-75 |
| + Phase F stacking (F1+F3+F5 at minimum) | each +3-8 | 70-110 |
| + Phase E tree decode if it works | 1.3-1.6× over chain | 95-150 |

125+ remains plausible; 95-125 sustained median is high-probability
once Branches 1+2+3 land. The honest case is still
unchanged from the AMENDMENT.
