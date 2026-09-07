# Measured frontier input — prefill vs prefix checkpoint — 2026-09-05

Everything below was measured in this worktree. No estimates.

## THE DEFECT

crates/hawking-core/src/model/qwen38_hybrid_decode.rs:

    pub fn qwen38_batched_prefill_allowed(
        batch_enabled: bool, reuse: usize, snapshot_at: Option<usize>,
    ) -> bool {
        batch_enabled && reuse == 0 && snapshot_at.is_none()
    }

Batched prefill is refused when a prefix checkpoint is RESTORED (reuse != 0) or TAKEN
(snapshot_at.is_some()). Batching defaults ON, so nothing logs that it was declined.

## THE EVIDENCE THAT IT BINDS EVERY REAL CALL

From .hcli/receipts/*.json, mission c4afe4c6, every model call:

    prefix_checkpoint_taken_at   1151 / 3222 / 3225      (always set)
    prefix_reused_tokens         0 / 1151 / 3222
    dispatches_per_step          916.0                   (every call, sequential route)
    dispatches_per_layer_per_step 14.31
    prefill_profile.totals.dispatches   2,966,924 for 3,239 prompt tokens

    3,231 prompt tokens / 115.8 s prefill = 27.9 fresh tok/s

## THE RATE THE SAME CODE REACHES WITHOUT A CHECKPOINT

    batched prefill, no checkpoint    74.8 fresh tok/s   (2.116x at 1032 tokens,
                                                          parity sha f36da4b00db5c5ed)
    in-tree gate                      batched dispatches < sequential / 4
                                      (crates/hawking-core/tests/
                                       qwen38_batched_prefill_constrained.rs)

## WHAT THIS COSTS

receipts/runtime/FRONTIER_INPUT_2026_09_05.md records one resident call at 547 s with
"roughly 400 s unaccounted INSIDE the call". It costed that call at 74.8 tok/s. HCLI's calls
run at 27.9 tok/s because they checkpoint. That is the missing 400 s.

Across 41 durable WorkUnit receipts: 21,527 s of resident call wall, 548,629 prompt tokens
against 76,622 completion tokens -- 7.2 prompt tokens paid for every generated token.

## THE QUESTION FOR THIS MISSION

Prefix reuse is proven (3,222 of 3,295 tokens reused; prefill 116 s -> 3.4 s).
Batched prefill is proven (>4x fewer dispatches, token-identical).
The admission rule makes them mutually exclusive. Is that restriction NECESSARY, or merely
conservative?

Two specific sub-questions, each independently answerable:

1. RESTORE (reuse != 0). After a checkpoint restore, the remaining SUFFIX is an ordinary
   cold prefill over positions [reuse, prompt_len). Why can the suffix not be batched?

2. SNAPSHOT (snapshot_at.is_some()). A snapshot must be taken at an exact position. The
   batched route advances a chunk of up to QWEN38_PREFILL_CHUNK (64, the MMA N tile) at a
   time. Can the prefill batch up to the chunk boundary at or below snapshot_at, step the
   remainder, take the snapshot, then resume batched?

## CONSTRAINTS ON ANY MUTATION

- Generated token identity is NON-NEGOTIABLE. The existing test asserts
  seq.new_tokens() == bat.new_tokens() and seq.stop_reason == bat.stop_reason.
- Decode must not regress.
- The dispatch gate must be SATISFIED, never relaxed:
  receipts/sovereign/G005_prefill_pipeline.json requires
  dispatch_gate_threshold_changed == false.
- hcli/test_qwen38_prefill_pipeline.py and every file in
  receipts/sovereign/VERIFIER_MANIFEST.json are PROTECTED. Landing refuses any proposal
  touching them.
- Smallest executable change. One operation. Name a test that fails before and passes after.

## THE INCUMBENT JUSTIFICATION, VERBATIM (refute this, or confirm it)

At the call site, qwen38_hybrid_decode.rs:

    // The batched path consumes the WHOLE prompt in one pass, so it can
    // honour neither `reuse` (which skips an already-computed prefix) nor
    // `snapshot_at` (which needs the recurrent state part-way through the
    // prefill, and that state is a running summary with no rewind). Taking
    // the batched path anyway would silently drop both — a faster wall and
    // a different answer. It is therefore selected only when neither is in
    // play, and prefix reuse keeps its own sequential route.

Note what this argues and what it does not. It argues that ONE batched call cannot stop
part-way. It does not argue that the prompt cannot be SPLIT. If `session.prefill_prompt`
appends to existing session state rather than requiring an empty session, then:

    restore(reuse) ; prefill_prompt(prompt[reuse..])
    prefill_prompt(prompt[..snapshot_at]) ; snapshot ; prefill_prompt(prompt[snapshot_at..])

both honour the constraint the comment defends while still batching. Establish whether
`prefill_prompt` appends at a nonzero position BEFORE proposing either. If it does not, that
is the real boundary and it should be recorded as such rather than worked around.

"A faster wall and a different answer" is the failure this must not produce. Token identity
against the sequential route is the acceptance test, not the wall.
