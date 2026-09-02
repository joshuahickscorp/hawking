"""Acceptance evidence: does a gate's own criterion actually hold.

One package per gate cluster. A gate is ACCEPTED only when a real run
demonstrates its stated criterion -- a call site proves reachable, not works.

THE DEFINING-PROPERTY LAW
-------------------------

    EQUIVALENCE BETWEEN TWO EXECUTIONS OF THE SAME IMPLEMENTATION
    DOES NOT PROVE THE DEFINING PROPERTY.

A fault inside a function cancels on both sides of a comparison between two
paths through it. Such a test is real evidence about the switch it toggles and
no evidence at all about correctness.

So for any capability whose semantics matter, at least one verifier here must
establish the defining property through an INDEPENDENT source: an oracle, a
mathematical invariant, a reference implementation, a controlled reconstruction,
an adversarial mutation, or another independent semantic test. Transcribe the
property from the obligation, not from the code under test -- a test that asks
the implementation what it does and then asserts it does that passes for any
implementation, including a broken one.

Watch for, in review:

    the same implementation on both sides of an equality
    the same conversion path on both sides
    a round trip whose encoder and decoder share one bug
    a simulator checked against simulator-generated expectations
    a benchmark compared against its own derived arithmetic
    a verifier reading a claim produced by the same producer

Do NOT mechanically flag all self-comparison. Determinism, purity and
idempotence legitimately compare two calls, and that is the property under test.
The pathology is a SEMANTIC function with ONLY shape or self-comparison coverage
and NO independent defining-property assertion. This is a reasoning rule for
whoever writes the verifier, not a lint.

Mutation clauses are not ceremonial end-of-task checks. They are the probe of
whether a verifier can detect a false reality. Break the load-bearing line,
confirm the test FAILS, restore it, and grep your marker before you finish.

Earned, not asserted. Every one of these was found by running that probe against
code already described as correct:

    _top_eigh returned the WORST rank-k subspace under an inverted slice and all
      19 fast tests passed -- one asserted only shapes, the other compared the
      serial path to the threaded path through the same function.
    residual_factors and residual_factors_batch returned r raw rows of R instead
      of the projection U^T R, and 20 tests passed.
    VMCP_RECEIPT_LAW was BUILT -- wired, acceptance-receipted -- with no test
      citing it at all, until its E.4 field list was transcribed from the
      roadmap and asserted here.
"""
