# Spec-decode verify kernel is NOT bit-identical to greedy — 2026-06-21

**Severity: foundational.** The "lossless / bit-identical speculative decode" guarantee the
project rests on has a real crack at near-tie logits, affecting BOTH the Event Horizon (EH)
path and the pre-existing OFF user-draft spec path.

## The finding
`forward_tokens_verify` (the batched verify path, `qwen_dense.rs` fast path ~:9193-9259,
GEMM `gemm_q4_k_m_batched_v3w_pinned_tcb`) returns a **batch-size-dependent argmax** that is
**not bit-identical** to the single-token greedy kernel `forward_token_greedy_tcb` (the
LM-head GEMV / predec path plain greedy uses). At a byte-identical KV state and the same
query token/position, the two kernels disagree on near-tie logits, and the batched path also
disagrees with itself at b=1 vs b=3.

Because spec-decode emits "accepted" draft tokens whenever they match the **batched verify**
argmax, an accepted token can differ from what plain greedy (single-token kernel) would emit.
The losslessness-by-construction invariant ("every emitted token is the target's greedy
argmax") silently fails.

## Evidence (instrumented traces + the property gate)
- New property gate `crates/hawking-core/tests/event_horizon_parity_prop.rs` (20 seeded
  prompts × length buckets) fails **6/20** EH-ON-vs-EH-OFF on prompts whose greedy output
  enters a short repeating cycle (the only case where the n-gram/suffix proposers fire).
- No-spec ground truth (`HAWKING_QWEN_USER_DRAFT=0`) proves NEITHER side is greedy:
  - prompt[19] pos42 tok30: batched verify → `30`; greedy kernel → `77` (= true greedy). **OFF is wrong.**
  - prompt[7]  pos79 tok22: b=1 verify → `14`; truth `22`. **ON is wrong** (different draft → b=1 at that boundary).
  - prompt[17] pos102 tok15: b=1 verify → `16`; truth `15`. **ON is wrong.**
- KV audit FALSIFIED the "stale KV / seq_len" hypothesis: `forward_tokens_batch_tcb:7511-7517`
  hard-asserts `seq_len == positions[0]`, so a mismatched seq_len would PANIC, not mis-emit.
  KV histories at the divergent positions were byte-identical on both paths. The seq_len
  trajectory is mechanically symmetric ON vs OFF.
- **First-hand confirmation:** routing EH-ON verify through the greedy kernel
  (`forward_token_greedy` per position) → property gate **19/20**, only prompt[19] remains —
  exactly the OFF-wrong prompt. This proves (a) greedy-kernel verify makes EH-ON truly
  lossless, and (b) the OFF reference is itself non-lossless. (Diagnostic reverted; it also
  breaks the verifier MockTarget unit tests.)

## Scope / production exposure (NEEDS A DECISION)
The defect is in the **shared verify kernel**, so it hits the existing OFF user-draft spec
path too — that path only passed the old single-prompt `user_draft_parity_e2e` test by luck
of phase (the canned prompt never hit a near-tie). **Open question: is the OFF user-draft
spec path default-ON in any shipped config?** If so, production decode is already
non-lossless at near-ties. (EH itself is default-OFF, so EH ships nothing yet.)

## Options (strategic — owner decision)
1. **Lossless-but-slow.** Verify via the greedy kernel (sequential `forward_token_greedy`
   per position). Truly bit-identical, but forgoes the batched-GEMM speedup → spec-decode
   stops being faster than greedy (defeats its purpose for throughput). Also needs: fix the
   gate reference to no-spec greedy (not EH-OFF), and update the verifier MockTarget.
2. **Fast-and-lossless (the real prize).** Make the batched verify GEMM numerically
   bit-identical to the single-token greedy GEMV (match accumulation order/precision / tie
   handling). Preserves the speedup. Hard Metal kernel work; first step is to characterize
   WHY they differ (reduction order? fp accumulation? predec scale path? b=1 degenerate tile?).
3. **Document near-losslessness.** Accept that argmax flips only at near-ties and retract the
   "bit-identical" claim. Cheapest, but abandons the core differentiator.

## Recommendation
- Make `event_horizon_parity_prop` (and a no-spec-greedy variant) a **permanent gate** — it
  caught what every prior test missed.
- Pursue option 2: characterize the batched-vs-single-token numeric gap (is b=1 a degenerate
  batched tile that's specifically wrong? if so, route b==1 → greedy kernel and keep batching
  for b≥2 — a cheap partial that may recover most losslessness while keeping the b≥2 speedup).
- HOLD the EH build-wave commit until the path is provably bit-identical to no-spec greedy.

## Status
- Build-wave otherwise solid: build clean, 79/79 speculate lib tests, propose-first bug fixed,
  parallel_draft + suffix_automaton + measurement harnesses present, nothing committed, flag OFF.
- **Commit HELD** on the losslessness resolution. GPU measurement studies (market bench, τ-sweep,
  replay-τ) also HELD — the throughput story depends on which option is chosen.
