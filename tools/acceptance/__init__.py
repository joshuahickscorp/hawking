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


CRITERION_KEYS = ("quoted", "quote")


def criterion_text(doc):
    """The criterion a receipt was judged against, in ANY of its four shapes.

    There are four ways this corpus records one field:

        criterion.quoted      29 receipts
        criterion.quote       11
        criterion_quoted      top-level, written by tools/acceptance/agentos
        criterion             a bare string, rare

    Use this rather than reaching into the document. A verifier that knows one
    shape silently mis-reads the rest, and that is not hypothetical: surveying
    for missing criteria with a single selector produced "28 of 68 receipts have
    none", then "8 of 48", before the correct answer of ZERO -- every gate
    verdict records its criterion. Three false alarms in a row, from the drift
    this function exists to absorb.

    Returns "" when there is genuinely no criterion, which is correct for
    summaries and for .gate/.run/.cycle sidecars: those are not verdicts.
    """
    if not isinstance(doc, dict):
        return ""
    c = doc.get("criterion")
    if isinstance(c, dict):
        for key in CRITERION_KEYS:
            if c.get(key):
                return str(c[key])
    elif isinstance(c, str) and c.strip():
        return c
    return str(doc.get("criterion_quoted") or "")


def evidence_count(doc):
    """How many checks a receipt shows, across every shape it may use.

    checks may be a list of dicts, a dict, or absent -- and when absent the
    producer may instead record `measured`/`comparison`. FLASH_DENSE_VS_NF_AB
    read as "ACCEPTED with zero checks" under a list-only reading when it had in
    fact verified four candidates.
    """
    if not isinstance(doc, dict):
        return 0
    checks = doc.get("checks")
    if isinstance(checks, (list, dict)) and checks:
        return len(checks)
    # Producers that record no `checks` still carry evidence, under several
    # names. Each of these was discovered by MISCOUNTING first:
    #   measured / comparison   FLASH_DENSE_VS_NF_AB, read as "zero checks"
    #   run                     ODYSSEY_I_DISCOVERY, a real model census
    #   numeric_comparisons     present but often legitimately empty
    for key in ("measured", "comparison", "run", "numeric_comparisons"):
        value = doc.get(key)
        if value:
            return len(value) if isinstance(value, (list, dict)) else 1
    return 0


def verdict_needs_evidence(doc):
    """Only a verdict that CLAIMS something needs evidence behind it.

    BLOCKED and NOT_RUN are claims that the gate did not run, and a receipt
    honestly reporting that correctly carries no checks. Demanding evidence from
    them manufactures findings: four of the five receipts that first looked
    evidence-less were BLOCKED, which is the system working.
    """
    return isinstance(doc, dict) and doc.get("verdict") in {"ACCEPTED", "PASS", "FAIL"}
