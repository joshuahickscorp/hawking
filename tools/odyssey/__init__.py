"""Odyssey machinery: real reproductions, contract closures, and data membership.

This package is deliberately independent of the launch fence. Training stages
still refuse without authorization; baseline reproduction, contract tests, data
inventory and the contamination barrier must all be runnable while the fence
stays false.

It also inventories on-disk corpora, checks declared membership against reality,
ingests raw corpora into content-addressed training sets, and mechanically
rejects train/eval contamination. It does not download data.
"""

__all__ = [
    "SCHEMA_INVENTORY",
    "SCHEMA_MEMBERSHIP",
    "SCHEMA_BARRIER",
    "PROBE_CLASS_STATIC_STREAMABLE",
    "PROBE_CLASS_EXECUTION_REQUIRES_RESIDENCY_OR_OFFLOAD",
    "PROBE_CLASSES",
    "ProbeClassRefused",
    "require_classification",
    "assert_execution_evidence",
]

SCHEMA_INVENTORY = "hawking.odyssey.data_inventory.v1"
SCHEMA_MEMBERSHIP = "hawking.odyssey.membership_record.v1"
SCHEMA_BARRIER = "hawking.odyssey.contamination_barrier.v1"

# ---------------------------------------------------------------------------
# Probe epistemic classification (2026-09-05 operator directive).
#
# "An oversized model cannot be called evaluated because its weight shards
# were independently inspected." Every Odyssey probe result -- everything
# returned by tools/odyssey/*.py, tools/flash_*.py, and the patient-runner
# stages -- must declare exactly one of the two classes below wherever its
# result is recorded. The two must never mix (a probe is one or the other,
# not a spectrum), and an undeclared probe REFUSES rather than defaulting.
# ---------------------------------------------------------------------------

PROBE_CLASS_STATIC_STREAMABLE = "STATIC_STREAMABLE"
PROBE_CLASS_EXECUTION_REQUIRES_RESIDENCY_OR_OFFLOAD = "EXECUTION_REQUIRES_RESIDENCY_OR_OFFLOAD"
PROBE_CLASSES = (PROBE_CLASS_STATIC_STREAMABLE, PROBE_CLASS_EXECUTION_REQUIRES_RESIDENCY_OR_OFFLOAD)


class ProbeClassRefused(RuntimeError):
    """A probe's epistemic class is missing, unknown, or insufficient.

    Raised in two situations, both refusals rather than a default:
      - a probe result has no `classification` field, or one outside
        PROBE_CLASSES (require_classification);
      - a capability claim cites only STATIC_STREAMABLE probes as evidence
        of an executed capability (assert_execution_evidence).
    """


def require_classification(result: dict) -> dict:
    """Validate `result["classification"]`; refuse if absent or unrecognized.

    The field name matches the convention already landed in
    tools/odyssey/specimen_open.py's census_specimen/census_lake (and its
    tests) -- one key for this concept across the package, not two.

    Call this at the point a probe result is recorded (written to a receipt,
    or returned from the probe function) so a probe that never classified
    itself fails loudly instead of being treated as either class by default.
    Returns `result` unchanged, so it composes as `return require_classification(rec)`.
    """
    cls = result.get("classification") if isinstance(result, dict) else None
    if cls not in PROBE_CLASSES:
        raise ProbeClassRefused(
            f"probe result declares classification={cls!r}; must be exactly one of "
            f"{PROBE_CLASSES}. An oversized model cannot be called evaluated because "
            "its weight shards were independently inspected -- classify this probe "
            "or leave it refusing."
        )
    return result


def assert_execution_evidence(probes, *, context: str = "capability claim") -> None:
    """Refuse a claim of executed capability backed only by static evidence.

    `probes` are the evidence entries the claim cites (e.g. a capability
    register's `capability_evidence.probes`). Each is validated with
    require_classification first -- an unclassified cited probe refuses the
    whole claim, not just itself. If every cited probe is STATIC_STREAMABLE,
    the claim is refused: independently inspecting weight shards is not the
    same epistemic act as running the model.
    """
    probes = list(probes)
    for p in probes:
        require_classification(p)
    if probes and all(p["classification"] == PROBE_CLASS_STATIC_STREAMABLE for p in probes):
        raise ProbeClassRefused(
            f"{context} cites only STATIC_STREAMABLE probes as evidence of an executed "
            "capability; independently inspecting weight shards is not evaluation. At "
            f"least one {PROBE_CLASS_EXECUTION_REQUIRES_RESIDENCY_OR_OFFLOAD} probe is required."
        )
