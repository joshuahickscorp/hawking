# path-to-125 session closeout — 2026-05-19 (dreamy-golick-d54ff8)

**Branch:** `claude/dreamy-golick-d54ff8`
**Commits landed this session:** 1 (`d3f831b`)
**Plan amendments:** AUTONOMOUS_PLAN.md `PLAN AMENDMENT 2026-05-19`

## Commits

```
d3f831b path-to-125 phase A1.0 step 1: forward_tokens_batched_parallel_k
        body — K-batched lm_head only
```

## What shipped — A1.0 (lm_head-only K-batched verify)

`forward_tokens_batched_parallel_k` body in
[crates/dismantle-core/src/model/deepseek_v2.rs:2841](crates/dismantle-core/src/model/deepseek_v2.rs:2841).
Gated on `kernel_profile.selected.verify_kernels == "parallel-k"`
(default remains `"sequential"`, so no production path change without
explicit opt-in).

The body mirrors `forward_tokens_batched_tcb` for the per-token layer
loop (per-token attn + MoE inside a single TCB — preserves MLA
causal-seq_len semantics and bit-identical greedy parity), and adds
one in-TCB `gemv_f16_lmhead_kbatch_tcb` dispatch in place of K post-
commit `gemv_f16_metal_pinned` calls. The 400 MB vocab×hidden fp16
weight is read once per verify step instead of K times.

## What the §6 A1 plan got wrong (now documented in PLAN AMENDMENT)

Three load-bearing assumptions failed on inspection of the kernel
substrate that landed in commits `08d4742..2fa02bc`:

1. **`mla_decode_metal_kbatch` is non-causal across K.** One
   `seq_len` constant for all K queries. The bundled parity test
   compares against K independent calls all using the same
   `seq_len` — so it validates *non-causal* semantics. Wiring as-is
   leaks future KVs between draft positions and breaks bit-identical
   greedy parity.

2. **`moe_block_batched_indexed_kbatch_tcb` is a no-overlap
   baseline.** It delegates to K calls of
   `encode_moe_block_batched_indexed_tcb` inside one TCB. No
   expert-weight amortization vs the existing per-token MoE loop.

3. **Eagle4 mode in main is K=1 today.** [crates/dismantle-core/src/model/deepseek_v2.rs:1456](crates/dismantle-core/src/model/deepseek_v2.rs:1456)
   forward_token_argmax per emitted token + Eagle4 head + stats-only
   compare. Always emits v2_argmax. So Eagle4 mode does Off-mode
   compute PLUS Eagle4-head per token — exactly why dec_tps is 9-10
   vs Off's 27. `forward_tokens_batched_parallel_k` is never called
   by Eagle4 until A2 (chain decode) lands.

Net consequence: A1.0 ships correct, parity-safe infrastructure that
**Eagle4 mode does not yet exercise**. ngram-spec mode (which DOES
call `forward_tokens_batched` at K>1) confirms the K-batched lm_head
path produces argmax-equivalent output to Off mode.

## Validation

- `cargo build --release -p dismantle-core`: clean (warnings pre-existed).
- `cargo test --release -p dismantle-core --lib`: **45/45 pass**.
- `cargo test --release -p dismantle-core --test path_b_parity`:
  **9/9 active pass** (4 unrelated stubs ignored).
- `cargo test --release -p dismantle-core --test eagle4_capture_smoke`:
  **1/1 pass** (26 s).
- `EAGLE4_PARITY_TEST=1 DISMANTLE_EAGLE4_GREEDY_TOKENS=32` parity gate
  with `verify_kernels=parallel-k`: **PASSES** (Off and Eagle4 emit
  identical 32-token streams; Eagle4 mode K=1 today so this validates
  the K=1 delegate-to-tcb branch).
- ngram-spec smoke with `verify_kernels=parallel-k` (real K>1
  exercise):
  ```
  prompt="The capital of France is" (max_new_tokens=16)
  Off:     " Paris.\n\nThe capital of"  (dec_tps=25.30, contended)
  ngram-p: " Paris.\n\nThe capital of France is Paris."
           (dec_tps=26.44, contended, accept=7/7 over two K-batched
            verifies at K=4 and K=3)
  ```
  → argmax-equivalent to Off at K=4 verify. The kbatch lm_head
  (simdmat semantics) matches `gemv_f16_metal_pinned` argmax in
  practice on V2-Lite.

## Pitfall compliance

- Pitfall #6 (user diagnostic edits): backed up + stripped before
  commit (`/tmp/{engine,kernels_mod,deepseek_v2}_a10_backup.rs`),
  re-applied after commit. `git diff --stat` confirms the
  3-file/+27-line user diagnostic delta is restored.
- Pitfall #2 (shader_hash): N/A — no .metal changes this session.
- Pitfall #7 (`reports/` gitignored): closeout + plan amendment
  staged with `git add -f`.

## Bench note

Bench is contended (Claude.app open). Per memory `bench_contamination`
inflated dec_tps 4-5×. Single-trial dec_tps numbers here are
indicative only; the A1.0 expected gain (~+10-15% on lm_head time =
~+1-2 dec_tps clean) is well below contended-bench resolution. **No
clean-window bench was attempted this session.** Phase A3 measurement
gate per the plan still needs the user to Cmd-Q Claude.app + run
`tools/bench/stage2_measurement.sh` (script not yet authored; see
"What's next" below).

## What's next

Three branches of work, ordered by criticality:

### Branch 1 — wire up A2 chain decode so A1.0 actually engages

The Eagle4 loop at [crates/dismantle-core/src/model/deepseek_v2.rs:1456](crates/dismantle-core/src/model/deepseek_v2.rs:1456)
needs replacement with a chain spec decode loop modeled on the
ngram-spec path at [crates/dismantle-core/src/model/deepseek_v2.rs:1382](crates/dismantle-core/src/model/deepseek_v2.rs:1382).
Per the AMENDMENT, the cleanest design:

1. Outer loop: at each chain step, head proposes K=4 drafts
   autoregressively. Initial captures come from `forward_token_argmax(last_id, pos, capture=true)`
   run ONCE before the loop (seeds h_low/h_mid/h_high/h_shared).
2. Autoregressive propose: for k in 0..4, `head.forward_full_amx_no_lm_head(cur_prev, h_low, h_mid, cur_h_high, h_shared)`;
   set `cur_h_high = head_out.draft_hidden`, `cur_prev = draft_id`.
3. Verify: `forward_tokens_batched([last_id, drafts...], [pos, ...])`
   — routes through `forward_tokens_batched_parallel_k` when profile
   selects parallel-k.
4. Accept: longest matching prefix, bonus = correction or argmax(logits[K]).
5. KV rollback: `self.kv.seq_len = draft_start_seq + first_reject + 1`.
6. Emit drafts[..first_reject] + bonus.
7. Run `forward_token_argmax(bonus, new_pos, capture=true)` to
   advance KV by 1 AND capture h_* for the next chain step.

Pitfall: forward_token_argmax expects `pos` as the position OF the
token being processed. After verify, `pos` should be the position of
bonus (= old pos + first_reject + 1). KV is already at
draft_start_seq+first_reject+1; the post-verify capture forward
appends bonus's KV at that slot, advancing seq_len to
draft_start_seq+first_reject+2.

Parity risk: the kbatch lm_head produces argmaxes argmax-equivalent
to `gemv_f16_metal_pinned` in the ngram smoke. Bit-identical greedy
gate at 32 tokens with verify_kernels=parallel-k passes. There's no
adversarial token-stream test in the suite yet — if the gate fails
on a longer sequence, fall back to gating parallel_k entry on
"only run kbatch lm_head when K=2" or rework the lm_head kernel.

### Branch 2 — A1.1 (causal-mask K-batched MLA kernel)

Extend `crates/dismantle-core/shaders/mla_decode_kernel_fc_kbatch.metal`
Phase 1 (line 87) with:

```metal
const uint kk_seq_cap = seq_len - k_batch + kk + 1;
for (uint t = tid; t < seq_len; t += tg_size) {
    if (t >= kk_seq_cap) { s_kk[t] = -INFINITY; continue; }
    ...
}
```

Convention: `seq_len = seq_slot_base + k_batch`; query kk attends to
`[0..seq_slot_base + kk + 1)`. Dispatcher signature stays the same.
Refresh parity test to compare against K independent K=1 calls each
using `seq_len = seq_slot_base + kk + 1`. Regen shader_hash
(pitfall #2). Then `forward_tokens_batched_parallel_k` can replace
the per-token MLA dispatch with one K-batched call → biggest single
bandwidth amortization (KV cache reads + kv_b_proj weight).

### Branch 3 — A1.2 (MoE expert-union K-batched kernel)

Stage 3.2 work the original plan defers to Phase D. On inspection
that's where the bulk of MoE bandwidth amortization actually lives.
Larger MSL kernel: tile the K-set of distinct experts and run K
queries against each tile. Probably 200-400 lines of new MSL +
dispatcher + parity test. Bigger lift than A1.1.

## Tasks captured for the next session

1. Implement A2 chain decode loop in Eagle4 mode (replaces the
   K=1 propose-then-compare path).
2. Capture-after-verify pattern (one extra `forward_token_argmax`
   per chain step).
3. Run parity gate with parallel-k profile after A2 — this is the
   real test of whether kbatch lm_head argmax-matches at K=4 on
   non-trivial sequences.
4. If parity holds, request clean-window bench (Phase A3 user-
   action gate from the plan §8).
5. If parity holds AND clean bench shows Eagle4 K=4 ≥ 13 dec_tps
   (above the projected band floor), move to A1.1 causal-mask MLA.

## Open user-action gates

- None mandatory yet — A1.0 doesn't require a bench to validate
  correctness. Phase A3 bench is queued for AFTER A2 lands (when
  the parallel-k path actually engages in Eagle4 mode).

## Working tree state at session end

```
 M crates/dismantle-core/src/engine.rs           (user diagnostic edits — RESTORED)
 M crates/dismantle-core/src/kernels/mod.rs      (user diagnostic edits — RESTORED)
 M crates/dismantle-core/src/model/deepseek_v2.rs (user diagnostic edits — RESTORED)
?? crates/dismantle-core/tests/ffn_shared_only_nonzero.rs  (untracked, user)
?? models/, tests/data/*, training_data/*  (untracked data dirs)
```

User's 3-file/+27-line diagnostic delta is identical to session start.
Verified via `git diff --stat`.
