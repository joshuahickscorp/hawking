"""Five orthogonal axes per capability, and a friendly status DERIVED from them.

The previous vocabulary collapsed independent facts into one word, and the word
was stronger than the evidence. VERIFIED_INTEGRATED read as "this is verified and
integrated" for gates that had a caller and an acceptance receipt but NO TEST
CITING THEM AT ALL, and VERIFIED_BUILT read as verified for gates whose own
acceptance had never been run. The details were honest; the headline was not.

So nothing here is a headline. Each axis answers one question, they are emitted
separately, and any friendly name is derived from the combination -- which is why
`INTEGRATED_ACCEPTANCE_UNRUN` exists as a status: it is what the evidence
actually supports, and it cannot be mistaken for verification.
"""
from __future__ import annotations

from typing import Any

# --- the axes -------------------------------------------------------------

IMPLEMENTATION = ("ABSENT", "SCAFFOLDED", "WIRED", "INTEGRATED")
ACCEPTANCE = ("UNRUN", "PASS", "FAIL", "BLOCKED")
VERIFICATION = (
    "NONE",
    "TEST_PRESENT",
    "DEFINING_PROPERTY_PROVEN",
    "MUTATION_PROVEN",
    "INDEPENDENT_ORACLE",
)
EVIDENCE = (
    "STATIC",
    "FUNCTIONAL_SIM",
    "LOCAL_MEASURED",
    "REPRODUCED",
    "PROTECTED_VERIFIED",
    "HARDWARE_MEASURED",
)
INTEGRATION = ("ISOLATED", "REACHABLE", "IN_PRODUCTION_PATH")

EVIDENCE_TIER_MEANING = {
    "STATIC": (
        "THIS AUDIT established source-level evidence only: definitions, call "
        "sites and receipts read out of HEAD blobs. It does NOT mean Hawking has "
        "never measured anything physically -- the campaign has produced real "
        "wall-clock, throughput and bandwidth measurements, and they live in "
        "receipts under receipts/. It means the CAPABILITY GRAPH has not imported "
        "them as gate evidence, so no gate may currently claim a physical result."
    ),
    "FUNCTIONAL_SIM": "a real local run of the software, not of the target hardware",
    "LOCAL_MEASURED": "measured on this machine, this run, with a retained receipt",
    "REPRODUCED": "measured independently more than once, agreeing within a stated bound",
    "PROTECTED_VERIFIED": "measured under the protected window with contamination controlled",
    "HARDWARE_MEASURED": "measured on the physical device the claim is about",
}


def implementation_state(gate: dict[str, Any]) -> str:
    status = gate.get("status")
    if status in {"ABSENT", "DORMANT"}:
        return "ABSENT"
    if (gate.get("wired") or {}).get("value"):
        return "INTEGRATED" if (gate.get("accepted") or {}).get("value") else "WIRED"
    return "SCAFFOLDED"


def acceptance_state(gate: dict[str, Any]) -> str:
    accepted = gate.get("accepted") or {}
    if accepted.get("value"):
        return "PASS"
    for ev in accepted.get("evidence") or []:
        kind = str(ev.get("kind") or "")
        if "refused" in kind or "blocked" in kind.lower():
            return "BLOCKED"
        if "fail" in kind:
            return "FAIL"
    return "UNRUN"


def verification_state(gate: dict[str, Any]) -> str:
    """How strong the verifier is, not merely whether one exists.

    Deliberately conservative: a test that merely cites the gate is TEST_PRESENT
    and nothing more. Claiming DEFINING_PROPERTY_PROVEN requires a human or a
    generator to assert it, because "does this test assert the defining property"
    is a judgement about meaning that no AST pass can make. Silence reads as the
    weaker answer.
    """
    if not gate.get("tests"):
        return "NONE"
    return str(gate.get("verification_state") or "TEST_PRESENT")


def integration_state(gate: dict[str, Any]) -> str:
    callers = gate.get("runtime_caller") or []
    if not callers:
        return "ISOLATED"
    for c in callers:
        rel = str((c or {}).get("file") or "")
        if not rel.startswith("tools/acceptance/") and "test" not in rel:
            return "IN_PRODUCTION_PATH"
    return "REACHABLE"


def axes(gate: dict[str, Any]) -> dict[str, str]:
    return {
        "implementation_state": implementation_state(gate),
        "acceptance_state": acceptance_state(gate),
        "verification_state": verification_state(gate),
        "evidence_tier": str(gate.get("evidence_tier") or "STATIC"),
        "integration_state": integration_state(gate),
    }


def derived_status(gate: dict[str, Any]) -> str:
    """A name that cannot overstate the axes it came from."""
    status = gate.get("status")
    if status == "BLOCKED_HARDWARE":
        return "BLOCKED_HARDWARE"
    if status == "BLOCKED_EXTERNAL":
        return "BLOCKED_EXTERNAL"
    if status == "UNREACHABLE":
        return "UNREACHABLE_DEPENDENCIES"

    a = axes(gate)
    impl, acc, ver = a["implementation_state"], a["acceptance_state"], a["verification_state"]

    if impl == "ABSENT":
        return "ABSENT"
    if impl == "SCAFFOLDED":
        return "SCAFFOLDED_NO_CALLER"
    if impl == "WIRED":
        return "WIRED_ACCEPTANCE_UNRUN" if acc == "UNRUN" else f"WIRED_ACCEPTANCE_{acc}"
    # INTEGRATED: wired AND acceptance passed. The verifier decides the name.
    if ver == "NONE":
        return "INTEGRATED_UNVERIFIED"
    if ver == "TEST_PRESENT":
        return "INTEGRATED_TEST_PRESENT"
    return f"INTEGRATED_{ver}"
