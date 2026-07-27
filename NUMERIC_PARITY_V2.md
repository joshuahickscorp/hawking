# Numeric Parity Contract V2

**Status: CANONICAL, user-authorized 2026-07-26.** Supersedes bit-identical-logits as the
cross-backend gate. Adopted *after* a lane failed the old gate, and deliberately not during
one — a weaker contract chosen mid-failure is how a quality gate erodes.

## What does NOT change

- **Byte-exact artifact semantics.** The sealed artifact and its receipts are untouched.
- **Deterministic same-backend execution.** The same backend, same seed, same input must
  produce the same output, every run. Bounded error is permitted *across* backends, never
  as an excuse for nondeterminism *within* one.

## What changes

For **cross-backend transcendental operations** — `exp`, and therefore silu, sigmoid,
softmax — the gate is **bounded numerical error**, not bit-identical floating-point
intermediates.

Rationale, established by measurement: Metal's `exp` and libm's `exp` differ in the last
1-2 ULP, and `precise::exp` does not close it. Every remaining Temporal Gravity milestone
requires moving attention, routing, the head and sampling to device, and a transcendental
appears in most of them. Holding bit-identity on intermediates would keep transcendentals on
the host permanently and cap the ladder.

**Do not keep a transcendental on the host merely to preserve libm-vs-Metal final-bit
identity.**

## What must remain EXACT — discrete agreement, no tolerance

- router top-k and **selected experts**
- **greedy emitted tokens**
- stop conditions
- structured-output and tool-call boundaries

These are decisions, not measurements. A one-ULP difference that changes an expert choice or
a token is not a rounding difference, it is a different model.

## The near-tie guard

Bounded error is only safe where the decision is not balanced on a knife edge.

When a **router margin** or a **final-token margin** falls below the frozen threshold, the
**canonical reference decision is invoked for that decision only** — not the whole token, not
the whole layer.

Design requirements:

- the threshold is **frozen and versioned**, calibrated from measured margin distributions,
  not guessed;
- guard invocations are **counted and reported**; a rate that climbs is a signal the bounded
  error is larger than assumed;
- the guard must be **cheap in the common case** — a margin comparison, not a second forward
  pass;
- the guard is itself deterministic.

## Priority order

1. persistent BF16 `lm_head` execution on device
2. eliminate CPU BF16→FP32 widening and full host logits
3. attribute and reduce the 10.46 GB/token overread toward the 2.58 GB geometry requirement
4. rebuild expert-wave as an **additive isolated path** under Parity V2
5. 234 → 78 expert synchronizations, then toward a token-level command graph
6. **preserve the existing resident path unchanged** until the new path passes regression and
   end-to-end parity
7. recompute the complete-token roofline after each promoted milestone

Item 6 is not boilerplate. The rejected expert-wave lane failed precisely because a flagged,
additive change silently altered an already-merged, already-parity-green default path.

## The promise, stated plainly

> Exact model behavior where decisions matter, bounded numerical equivalence where continuous
> arithmetic differs, and canonical fallback at unstable margins.

`BASE_TRUE_TPS` stays separate from acceleration. Sealed Math-Preserve unchanged. MOP untouched.
