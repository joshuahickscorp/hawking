# CP6h -- the LM head IS wasted in prefill, and it is only 4.3%

2026-09-04 · frontier G019 · resident loaded · **hypothesis measured and set aside**

## The hypothesis

`encode_full_token` is embed -> layers -> **terminal**, and `encode_terminal` is
the final norm plus a full-vocab matvec plus argmax. Prefill drives one `step()`
per prompt token. So every prefill position computes logits over the entire
vocabulary and every one but the last is discarded.

That is structurally true -- it was read directly from the encoder, not inferred.
It is also orthogonal to chunking, so if it were a large share of the prefill
wall it would be a cheaper win than batching anything.

## Measured

11 alternating reps, first dropped, same session and quiet lane, noop control
666 ns.

```
  lm_head  ns  min  886,499   med  1,253,583   max  1,570,374
  token    ns  min 28,094,291 med 28,902,166   max 29,285,625

  lm_head share of the token wall: 4.3%  [3.0% - 5.6%]
  upper bound if computed ONCE per prompt:  1.045x
```

**4.3%.** The waste is real and the fix is easy, but the ceiling is 1.045x and
that bound is generous -- it assumes removing the head changes nothing else.

## Decision

**Not the frontier. Recorded, not built.**

CP6g's hybrid chunk measured 1.27-1.33x on the same layers. Spending the next
implementation on a 1.045x ceiling while a 1.3x sits unwired would be choosing
the smaller lever because it was noticed second.

Reopen condition: if the chunked prefill lands and the LM head's share rises --
which it will, since chunking shrinks the denominator and leaves the head
untouched -- 4.3% of a 1.3x-faster wall is a larger fraction, and computing
logits once per chunk becomes worth the change. It should be reconsidered then,
against the wall as it exists then.

This is the frontier rule working as intended: a plausible structural waste was
found by reading the encoder, measured before being believed, and cost one
harness to rule out.
