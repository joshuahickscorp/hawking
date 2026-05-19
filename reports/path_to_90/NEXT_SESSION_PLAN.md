# Next-session plan — path-to-125 (semi-autonomous)

**Purpose:** keep the agent moving with minimal user input. Pre-baked
defaults for every fork; user only answers if they want to override.
Critical questions listed below — agent reads §1, asks ONLY the
questions whose default doesn't apply (e.g., no bench tooling on the
machine → don't ask about bench cadence), then executes.

This document is authored by the agent at the end of 2026-05-19,
intended for the next-session agent to read end-to-end before doing
anything else. Supersedes `session_closeout_2026-05-19.md`'s "next
priority queue" section (still consult it for code-level design hints
on each branch).

---

## 0. Pre-read summary

Last session shipped Phase A: 6 commits on
`claude/dreamy-golick-d54ff8` head `5681b16`. Current state:

- A1.0 ✅ `forward_tokens_batched_parallel_k` body, lm_head-only
  amortization, gated on `verify_kernels=parallel-k`.
- A1.1 ✅ shader (causal mask) + TCB dispatcher (`mla_decode_kbatch_tcb`)
  ready and parity-tested. **Not yet called from parallel_k.**
- A2  ✅ Eagle4 chain spec decode, env-var gated (`EAGLE4_CHAIN_K=k`).
  Bit-identical to Off. Head autoregressive accept rate ~5% → net
  regression vs K=1 until Phase D retrain.
- A3 measurement: **NOT queued.** Wait for A1.1 wire-up + ideally
  A1.2 before bothering the user for a clean-window run.

Bench remained contaminated all of last session. Per `bench_contamination`
memory: Claude.app open inflates dec_tps 4-5×. **All single-trial
numbers in commit messages are indicative; no clean number was
captured.**

User diagnostic edits (`engine.rs` +10, `kernels/mod.rs` +13,
`deepseek_v2.rs` +4) are present and untouched at session start.
**Pitfall #6 strip-restore is mandatory** for any commit that
modifies those files.

---

## 1. Critical questions (ask only if needed)

The agent should ASK only the subset of these where the **default
answer would meaningfully change autonomous direction**. If the
default works, just proceed; don't burn a user turn.

### Q1 — A1.2 commitment (10-15 hour bet)

**Question:** "A1.2 (MoE expert-union K-batched kernel) is the
biggest single lever — ~3-5 GB / verify-step bandwidth saving — but
~10-15 hours of new MSL + dispatcher + parity work, mostly on the
critical path. Want me to attempt it as the primary work of this
session, or de-risk with Branch 1 first and pause for your read?"

**Default (no answer):** **De-risk with Branch 1 first.** Ship A1.1
wire-up (1-2 h), validate parity passes, then start A1.2 if there's
headroom. This locks in real production engagement of parallel_k
before betting a session on a single large kernel.

**Ask if:** the agent reaches a fork where A1.2 design has multiple
plausible paths (e.g., union vs masked vs gated-routing) and the
choice is non-obvious from existing code.

### Q2 — Eagle4 head retrain feasibility

**Question:** "A2's accept-rate collapse (94% K=1 → 5% K=4) is
structural — head was trained on V2-Lite's true hidden states, not
its own `draft_hidden`. Two unlocks: (a) retrain head with
EAGLE-3-style autoregressive self-consumption (1 day of Python
training on your machine), (b) skip multi-step head, reactivate
mask_logits + masked-prefetch path (no training needed, smaller
gain). Which does this session pursue?"

**Default (no answer):** **Pursue (b)** first — no Python training
needed, agent can build masked-prefetch entirely. (a) is the bigger
unlock long-term but blocks on you running training.

**Ask if:** the agent finishes Branches 1 + 2 + (b) and is deciding
whether to start scoping the (a) training pipeline. At that point
the answer to Q2 actually gates the next session, not this one.

### Q3 — Bench cadence

**Question:** "Clean-window bench needs you to Cmd-Q Claude.app and
optionally pause `slm`. When in this session should I queue an
A3-style bench (after Branch 1? after Branch 2? at session end)?
Or skip the bench entirely this session and just queue it for next?"

**Default (no answer):** **Surface bench script + ask once, AFTER
Branch 1 wire-up commits AND a smoke shows non-error operation.**
Don't ask twice. If user says "skip", continue and write closeout
without a clean number.

**Ask if:** Branch 1 lands successfully — the resulting clean dec_tps
is the first real data point that proves parallel_k engages. After
that, default is to ask once.

### Q4 — A2 fate (sticky)

**Question:** "A2 chain decode shipped behind `EAGLE4_CHAIN_K` env
var; default behavior unchanged. It currently regresses dec_tps due
to head-quality issues, not code. Leave it env-gated (safe — no
production impact, future-ready) or revert (clean main, can re-add
later)?"

**Default (no answer):** **Leave it env-gated.** Zero production
impact unless flag set. Cost is ~200 LoC sitting in the source. Net
positive — keeps the seam alive for when head retrain lands.

**Ask only if:** the agent finds A2's env-gating interferes with
some other work (which it shouldn't).

---

## 2. Work order (ordered, with skip/proceed criteria)

### Branch 1 — A1.1 full wire-up (1-2 h, do this first)

Wire `mla_decode_kbatch_tcb` (commit `7668247`) into
`forward_tokens_batched_parallel_k`. Per the design sketched in that
commit's message:

**Approach (no DecodeArena schema change):** allocate per-K shadow
buffers ad-hoc inside the function — `shadow_x_buf[K]`,
`shadow_ffn_out_buf[K]`, `packed_q`, `packed_attn_out`,
`scores_scratch`. Sized for the call's K and `seq_slot_base + K`.

**Per-layer restructure** (replaces the current per-token layer loop):

```rust
for li in 0..n_layers {
    // Phase A: K iterations of phase1 + phase2 with per-K state.
    for ki in 0..k {
        if li > 0 {
            tcb.copy_buffer_bytes(&shadow_x_buf[ki], 0, &arena.x_buf, 0, h_bytes)?;
            tcb.copy_buffer_bytes(&shadow_ffn_out_buf[ki], 0, &arena.ffn_out_buf, 0, h_bytes)?;
        }
        encode_attention_phase1_into_tcb(...)?;     // li=0 embed_lookup; li>0 add_inplace(x_buf, ffn_out_buf)
        encode_attention_phase2_tcb(...)?;          // q_b_proj + rope, writes arena.q
        // Pack q.
        tcb.copy_buffer_bytes(&arena.q, 0, &packed_q, ki as u64 * q_per_token_bytes, q_per_token_bytes)?;
        // Save residual state.
        tcb.copy_buffer_bytes(&arena.x_buf, 0, &shadow_x_buf[ki], 0, h_bytes)?;
    }

    // Phase B: K-batched MLA (single dispatch).
    let seq_len = seq_slot_base + k;
    parallel_k::mla_decode_kbatch_tcb(
        tcb, &packed_q,
        &self.mla_c_kv_gpu[li], &self.mla_k_pe_gpu[li],
        kv_b_proj_buf, &packed_attn_out, &scores_scratch,
        n_heads, qk_nope, qk_rope, v_head, kv_lora,
        seq_len, scale, k,
    )?;

    // Phase C: K iterations of o_proj + residual + ffn_norm + MoE.
    for ki in 0..k {
        tcb.copy_buffer_bytes(&shadow_x_buf[ki], 0, &arena.x_buf, 0, h_bytes)?;
        tcb.copy_buffer_bytes(&packed_attn_out, ki as u64 * attn_per_token_bytes,
                              &arena.attn_out, 0, attn_per_token_bytes)?;
        gemv_f16_simdmat_tcb(tcb, o_proj_buf, h, n_heads * v_head, &arena.attn_out, &arena.out)?;
        encode_add_and_rmsnorm_tcb(tcb, &arena.x_buf, &arena.out, ffn_norm_buf, eps, h, &arena.x_norm_buf)?;
        // MoE or dense — copy block verbatim from current A1.0 body.
        // ...
        tcb.copy_buffer_bytes(&arena.x_buf, 0, &shadow_x_buf[ki], 0, h_bytes)?;
        tcb.copy_buffer_bytes(&arena.ffn_out_buf, 0, &shadow_ffn_out_buf[ki], 0, h_bytes)?;
    }
}

// Final per-K: restore shadow x_buf + ffn_out_buf, add_inplace, final norm,
// blit to x_packed[ki * h] for the K-batched lm_head.
// ...
```

**Parity gate (load-bearing):**
```
EAGLE4_PARITY_TEST=1 DISMANTLE_EAGLE4_GREEDY_TOKENS=32 \
  cargo test --release -p dismantle-core \
  --test eagle4_decode_parity -- --ignored --nocapture
```
with `verify_kernels=parallel-k` in the on-disk profile. Off and
Eagle4 must emit identical 32-token streams. If it fails:
- Likely culprit: blit ordering inside the per-layer loop
  (shadow restore happens before phase1's add_inplace; verify Phase
  A's `if li > 0` branch correctly restores ffn_out_buf).
- Second-likely: `scale` constant — must match what
  `dispatch_mla_decode_and_o_proj` uses (`1.0 / sqrt(head_dim_q)`).
- Third: o_proj kernel mismatch — `dispatch_mla_decode_and_o_proj`
  may route to a different o_proj than `gemv_f16_simdmat_tcb`
  depending on `mla_use_fc` / `mla_use_flash`. Inspect and match.

**Smoke gate:** ngram-spec mode under parallel-k profile produces
Off-equivalent output (the test from last session).

**Skip-proceed criteria:**
- ✅ Parity passes → commit, proceed to Branch 2.
- ❌ Parity fails AND can't root-cause in 30 min → revert the Phase
  A/B/C restructure, keep A1.0 path in `forward_tokens_batched_parallel_k`.
  Write a halt doc under `reports/path_to_90/halt_a11_wireup.md`,
  proceed to Branch 2 anyway (A1.2 doesn't depend on A1.1 wire-up).

**Pitfall #6:** `deepseek_v2.rs` carries user diagnostic edits.
Backup → strip → edit → commit → restore. Same pattern as last
session's commits.

### Branch 2 — A1.2 MoE expert-union K-batched kernel (10-15 h)

Subject to Q1's answer. Default: start AFTER Branch 1 ships.

The big design hooks (read these before coding):

- Current no-overlap kernel: `crates/dismantle-core/src/kernels/parallel_k.rs:471`
  `moe_block_batched_indexed_kbatch_tcb`. Reads K route_id buffers,
  K route_weight buffers, calls `encode_moe_block_batched_indexed_tcb`
  K times. NO weight amortization.
- Existing K=1 indexed MoE kernel:
  `encode_moe_block_batched_indexed_tcb` (in `kernels/mod.rs`) — this
  is what gets called K times. Look at its shader path and the
  per-expert dispatch to understand the kernel surface area.

**New kernel shape:**

```
Inputs:
  K route_id buffers (n_routed, top_k) → built per query
  K route_weight buffers (n_routed, top_k) f32
  K input x buffers (hidden,) f32
  K output y buffers (hidden,) f32

Algorithm:
  1. CPU-side preprocessing: build a `union_experts: Vec<u32>` of
     distinct experts across the K queries. Build per-union-expert
     index lists: for each union expert e, which K queries selected
     it, with what weight.
  2. Single Metal dispatch per union expert (or batched-grouped):
     for each union expert e, GEMM e's weight against ALL queries
     that selected e, weighted by per-query per-route weight, into
     y[k] for each selecting k.

Threadgroup layout: one TG per (union_expert, hidden_block). Each
TG reads the expert's weight ONCE into TG-mem, then iterates over
the queries that selected it.

Parity gate: at K=1 must reduce to encode_moe_block_batched_indexed_tcb
bit-identical (single expert per query, no overlap, single
contribution).

Wall-clock A/B vs no-overlap baseline: ≥30% improvement at K=4 on
V2-Lite shape (with ~50-70% routing overlap typical).
```

**Skip-proceed criteria:**
- ✅ Parity passes AND ≥10% wall-clock improvement → commit, wire
  into parallel_k Phase C, run full parity + smoke.
- ❌ Parity passes BUT no wall-clock win (≤5%) → commit the kernel
  as available-but-not-default, document in commit message, move to
  Branch 3 / Phase B.
- ❌ Parity fails AND can't root-cause in 90 min → halt doc, move on.

### Branch 3 — Mask-prefetch path (Phase D 3.2, default Q2)

Subject to Q2. Default: skip head retrain (a) this session; build
masked-prefetch (b) instead.

Per AUTONOMOUS_PLAN.md §6 D2/D3:

- Reactivate `mask_logits` in Eagle4 head's Metal/AMX forward
  (currently skipped per commits `0d6a2a3` / `808d8db` for current-
  recall-uselessness; logged but not consumed).
- New shader/dispatcher variant of the K-batched MoE kernel that
  accepts a `(N_ROUTED=64) u8` predicted-mask bit buffer and issues
  `MTLResidencySet.addAllocation` (async prefetch hint) for
  mask-bit-set expert weights BEFORE the K-batched dispatch needs
  them. Prefetch misses fall back to on-demand load (no correctness
  impact).
- Wire predicted mask through Eagle4 chain decode's verify call.

**Pre-gate:** Eagle4 head's `mask_topk_mean_recall` must be ≥40%
(plan §6 D4 decision matrix; <40% = masked variant unhelpful, skip).
The head checkpoint `eagle4/checkpoints/eagle4_v3/best.npz` reports
recall via `python eagle4/eagle4.py eval`. If recall is the existing
17.78%, skip Branch 3 entirely until Phase D fine-tune runs.

**Skip-proceed criteria:**
- ✅ Recall ≥60% AND masked variant shows ≥5% wall-clock win → ship
  + wire as default behavior at K>=2.
- ✅ Recall ≥40% AND ≥3% wall-clock win → ship behind config flag.
- ❌ Recall <40% → skip, document in closeout that Branch 3 is
  blocked on Phase D fine-tune.

### Branch 4 — Phase F levers (after Branches 1-3, time-permitting)

Each lever independent, A/B-gateable, ~3-10% per stack. Per ROI
from AUTONOMOUS_PLAN.md §6 F:

| Lever | Est effort | Risk | Order pref |
|---|---|---|---|
| F3 async verify-start | 2-3 h | low (scheduling only) | 1 |
| F5 multi-queue Metal | 2-3 h | low | 2 |
| F1 AMX extend to V2-Lite projections | 4-6 h | medium | 3 |
| F4 ANE routing-logits offload | 3-4 h | medium | 4 |
| F2 Q4 MLA-KV | 3-4 h + PPL eval | high (PPL regression risk) | 5 |

Default: attempt F3 first (lowest risk, concrete projected win).
Each ships behind A/B gate; skip if <3% e2e.

---

## 3. When to ask vs when to push

**ASK** (use AskUserQuestion tool) only when:

1. A Q1-Q4 fork applies and the default doesn't fit the data the agent
   just discovered.
2. A1.2's design has a non-obvious sub-fork (e.g., union granularity
   choice that affects shader complexity by 2×+).
3. Branch 1 wire-up parity FAILS unexpectedly and the diagnosis points
   to a kernel/profile assumption the user might know better than the
   agent.
4. The clean-window bench gate is reached (per Q3 default, ask once
   after Branch 1 ships).

**DON'T ASK** for:

- Implementation detail choices (kernel layout, blit ordering, buffer
  sizing — agent decides).
- Whether to "keep going" — autonomy charter says continue per §5.
- Permission for risky-but-reversible operations (git resets within
  current session, env-var experiments, profile flag toggles —
  agent decides per `.claude/CLAUDE.md`'s autonomy norms).
- Confirmation of obvious next-step from this plan.

**DO** (without asking):

- Commit as Joshua Hicks via inline `git -c user.name=...
  user.email=...`. Never Co-Authored-By, never `git config`.
- Pitfall #6 strip-restore on every commit touching `engine.rs`,
  `kernels/mod.rs`, `deepseek_v2.rs`.
- Regen shader_hash + update profile when any .metal file changes
  (pitfall #2).
- Run lib tests + path_b_parity + eagle4_decode_parity (the
  bit-identical greedy gate) at every commit. Halt on parity
  failure per §5 of AUTONOMOUS_PLAN.md.
- Update this NEXT_SESSION_PLAN.md inline when data invalidates an
  assumption (e.g., Branch 1 parity fails for a reason this plan
  didn't anticipate — add a "PLAN AMENDMENT YYYY-MM-DD" section).

---

## 4. Halt / checkpoint criteria

**Halt and write closeout when ANY:**

1. Branch 1 lands + 2 measurement gates queued for user (per Q3).
2. ~6 hours elapsed without landing a commit.
3. Bit-identical parity gate fails AND can't root-cause in 30 min
   (per AUTONOMOUS_PLAN.md §5).
4. RSS >5 GB sustained for >5 min (memory pressure halt).
5. The user requests stop.
6. Branches 1, 2, and either 3 or first Phase F lever have all
   shipped (natural session-cap point — A1.1 wire-up + A1.2 +
   one more lever is a strong session).

**Checkpoint (commit + brief status, keep going)** at:

- End of each branch.
- Any +5 commits without a status update.
- Any pivot from plan default (so audit trail captures the reason).

**Closeout location:** `reports/path_to_90/session_closeout_YYYY-MM-DD.md`
following the template from
`reports/path_to_90/session_closeout_2026-05-19.md`.

---

## 5. Pitfall reminders (read before code)

1. **User diagnostic edits** in 3 files — strip-restore pattern
   (commit `d3f831b` for working example).
2. **shader_hash** — regen + update profile JSON after ANY .metal
   change. `./target/release/dismantle shader-hash` after `cargo
   build --release -p dismantle`.
3. **Build is two-step** — `cargo build --release -p dismantle-core`
   builds the lib only; the CLI binary needs `-p dismantle`
   separately. Both before running smokes.
4. **K-batched MLA causal convention** — `seq_len = seq_slot_base
   + k_batch`; query kk attends to `[0, seq_slot_base + kk]`. ALL
   K KVs must be appended BEFORE the K-batched MLA dispatch.
5. **dispatch_mla_decode_and_o_proj has TWO o_proj kernel paths**
   (`mla_use_fc` flag selects `mla_decode_and_o_proj_arena_fc_tcb`
   vs `mla_decode_and_o_proj_arena_tcb`). Both use
   `gemv_f16_simdmat_tcb` for o_proj at the end. Match this in
   Branch 1's Phase C.
6. **Off uses `gemv_f16_metal_pinned` for the post-commit lm_head**
   (line 2197 of deepseek_v2.rs); A1.0's K-batched lm_head uses
   `gemv_f16_lmhead_kbatch_tcb` (= simdmat semantics). Argmaxes
   match in practice but aren't bit-identical FP. The eagle4
   parity gate is the safety net.
7. **Bench is contaminated** until user Cmd-Qs Claude.app. No
   absolute dec_tps numbers from contended runs.

---

## 6. Reference index

- Last closeout: `reports/path_to_90/session_closeout_2026-05-19.md`
  — Branches 1-4 design hints + dec_tps trajectory.
- Main plan: `reports/path_to_90/AUTONOMOUS_PLAN.md` — §3.5 autonomy
  charter, §5 halt criteria, §6 phase-by-phase scope. **Includes
  PLAN AMENDMENT 2026-05-19** that supersedes original §6 Phase A
  estimates.
- Code-level pointers (commits): `d3f831b` (A1.0 body), `df459ee`
  (A1.1 shader), `7668247` (A1.1 dispatcher), `1f97dfb` (A2 chain
  decode).
- Profile: `profiles/deepseek-v2-lite-q4.m3pro18.json`. Toggle
  `selected.verify_kernels` to `"parallel-k"` to engage. Current
  shader_hash: `d214bba83371d3e7d4b70794`.

---

**End of plan. Read once, execute. Ask only Q1-Q4 if needed; default
otherwise. Halt only on §4 criteria.**
