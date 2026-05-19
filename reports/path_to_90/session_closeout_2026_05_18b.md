# Path-to-90 session closeout — 2026-05-18 (Stage 2.1 → 2.6 scaffold)

Autonomous execution session against
`reports/path_to_90/production_roadmap_to_100_tps.md`, starting at
Stage 2 task 2.1.

## Commits landed

| commit | step | summary |
|---|---|---|
| `adedd4c` | 2.1 | Path B design refresh — finalize K-batched kernel surfaces + TG-mem budget; identify lm_head naming reality (V2-Lite is f16 not Q6_K) |
| `08d4742` | 2.2 | `gemv_f16_lmhead_kbatch` — K-batched fp16 lm_head GEMV; 3 parity tests green |
| `b261b2d` | 2.3 | `gemv_q4_k_m_v2_kbatch` — K-batched Q4_K_M GEMV; 3 parity tests green (vs CPU dequant reference, sidesteps pre-existing v2-non-pinned dispatcher slot/struct bug) |
| `e2b945f` | 2.4 | `mla_decode_kernel_fc_kbatch` — K-batched MLA decode with device-scratch scores (TG-mem budget honored); 3 parity tests green |
| `cade5a7` | 2.5 | `moe_block_batched_indexed_kbatch_tcb` — no-overlap K-batched MoE Rust wrapper (per roadmap; masked-prefetch variant is Stage 3.2) |
| `2fa02bc` | 2.6 (scaffold) | `forward_tokens_batched_parallel_k` scaffold + `verify_kernels` profile flag; routes at flag toggle; K=1 delegates, K>1 Unimplemented until body lands |

All commits authored as `Joshua Hicks <joshuahicksboba@gmail.com>` via
inline git identity per `CLAUDE.md`. No Co-Authored-By trailers; no
`Generated with` footers.

## dec_tps progression (single-trial, contended)

Baseline (session closeout 2026-05-18 morning): 9.62 dec_tps median.

Per-commit smoke (single trial, Claude.app open → contended):

| commit | dec_tps | draft accept |
|---|---|---|
| 2.2 post-land | 9.75  | 15/16 |
| 2.3 post-land | 10.47 | 15/16 |
| 2.4 post-land | 10.59 | 15/16 |
| 2.5 post-land | 9.62  | 15/16 |
| 2.6 post-land | 9.29  | 15/16 |

All within trial noise of baseline. No regression. Note: no commits in
this session change the production hot path — the K-batched kernels
are not yet called from the forward pass; the Stage 2.6 scaffold only
flips behavior when the profile flag is opted into.

## Bit-identical greedy gate

`eagle4_decode_parity::eagle4_greedy_bit_identical_with_off` passes at
every commit. Off mode and Eagle4 mode emit token-identical sequences
on "The quick brown fox" at 16 tokens, temp=0 greedy.

## What's settled vs open

### Settled
- **Path B kernel substrate** — three of the four K-batched kernels
  (lm_head fp16, Q4_K_M attn/MoE projection, MLA decode) are landed
  with parity tests at K ∈ {1, 2, 4, 8}. The fourth (MoE block) is a
  Rust-side TCB-batching wrapper; the real K-amortizing kernel is
  Stage 3.2 (masked-prefetch variant).
- **Stage 2.6 entry seam** — `kernel_profile.selected.verify_kernels`
  field added (default `"sequential"`, opt-in `"parallel-k"`).
  `forward_tokens_batched_parallel_k` scaffold installed with K=1
  delegation; production decode untouched.
- **Pre-existing Q4_K dispatcher bug surfaced** — the non-pinned
  `dispatch_q4_k_m_gemv_v2` (kernels/mod.rs:2636) and its v3/simdmat
  siblings bind two 4-byte `set_bytes` at slots 3/4 while the shaders
  declare a single 8-byte `ArgbufRowsCols` struct at slot 3. All 11
  tests in `v1k_q4kgemm_simdmat_parity.rs` fail because of this.
  Production decode is unaffected (uses the pinned-tcb dispatcher with
  proper `KernelArgBuffer`). Chip spawned for an attended session to
  fix the shader signatures.

### Open / next-up
- **Stage 2.6 (body)** — full K>1 K-batched verify forward. ~500-line
  mirror of `forward_tokens_batched_tcb` (deepseek_v2.rs:2654) but
  dispatching the Path B kernels. Test gate: bit-identical at K=1 vs
  sequential (the scaffold already passes this; the body adds K>1
  parity vs the existing batched-TCB sequential path at K=2..4). This
  is the highest-impact remaining commit on the critical path.
- **Stage 2.7** — Eagle4 chain spec decode at K=4. Touches
  `deepseek_v2.rs::generate` Eagle4 branch; uses
  `forward_tokens_batched_parallel_k(K=4)` from 2.6. The
  longest-matching-prefix accept rule from the roadmap.
- **Stage 2.8 — measurement (USER-REQUIRED)** — needs you to Cmd-Q
  Claude.app to clear GPU contention. Bench script to write:
  `tools/bench/stage2_measurement.sh`. Gate: ≥ 38 dec_tps sustained
  median, bit-identical at K=1 vs Off.
- **Stages 3, 4, 5** — routing fine-tune + masked prefetch + tree
  decode + hardware paths. All depend on Stage 2 measurement gating
  cleanly through.

## Working-tree state at closeout

User diagnostic edits **preserved** across all 6 commits via the
strip-restore pattern (see commit `2fa02bc` for the most-recent
example):

```
 M crates/dismantle-core/src/engine.rs           (ffn_shared_only_for_test trait method)
 M crates/dismantle-core/src/kernels/mod.rs      (DBG_Q4KV2_PINNED eprintln)
 M crates/dismantle-core/src/model/deepseek_v2.rs (ffn_shared_only_for_test impl)
?? crates/dismantle-core/tests/ffn_shared_only_nonzero.rs (untracked test)
?? models/                                       (model files)
?? tests/data/ultrachat_*.jsonl                  (training data)
?? training_data/c2_hidden/                      (training data)
```

`git diff HEAD --` for each tracked file matches the corresponding
working-tree state at session start. No diagnostic edits were lost
or re-committed.

## Profile state

`profiles/deepseek-v2-lite-q4.m3pro18.json`:
- `shader_hash` rotated three times this session as new shader sources
  landed:
  - Start: `ee4a8635...` (Stage 1 baseline)
  - Post-2.2: `46584c9c...`
  - Post-2.3: `2622e334...`
  - Post-2.4: `e25ef35f...` (current)
- `selected.verify_kernels` field NOT yet present on disk; loads as
  `"sequential"` via `#[serde(default)]`. Next autotune run should
  regenerate the profile to make the field explicit.

## Next session's first task

**Stage 2.6 (body)** — full K>1 K-batched verify forward.

Concrete starting points:
1. Read `deepseek_v2.rs:2654` (`forward_tokens_batched_tcb`) end-to-end
   to map every dispatch it performs.
2. Mirror its structure in `forward_tokens_batched_parallel_k` but
   replace:
   - K sequential MLA dispatches → one `mla_decode_kernel_fc_kbatch`
   - K sequential `gemv_q4_k_m_v2_pinned_tcb` calls for attn projs →
     one `gemv_q4_k_m_v2_kbatch_pinned_tcb`
   - K sequential MoE forwards → one
     `moe_block_batched_indexed_kbatch_tcb`
   - K sequential lm_head GEMV → one `gemv_f16_lmhead_kbatch_tcb`
3. Verify bit-identical at K=2..4 vs current sequential-TCB path with
   `verify_kernels="sequential"`. Add a unit test that toggles the
   flag mid-run.
4. Smoke + bit-identical greedy at K=4 (eagle4_decode_parity at
   `--max-new-tokens 64`).

Effort estimate: 2–4 hours of careful work (per roadmap "half day").

If body proves too risky to ship in one commit, split as 2.6.body.a
(MLA + lm_head K-batching, leave MoE + Q4_K_M projs sequential) and
2.6.body.b (MoE + projs K-batching). Both halves must preserve
bit-identical greedy.

## Per `CLAUDE.md` § Tone of artifacts

Terse, audit-trail focused. No prose padding.
