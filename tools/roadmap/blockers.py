"""What blocks a capability, in categories that route work correctly.

The previous five classes forced unlike things together. Theia models that were
never trained, a training substrate nobody has built, and a VMCP capability
waiting on a Chrome install were all LONG_RUN_EVIDENCE_REQUIRED, purely because
each takes time. That misroutes an operator: HCLI should not spend a night
"gathering long-run evidence" for a model that does not exist, and installing a
browser extra is not an evidence problem at all.

UNKNOWN_RESEARCH is reserved for questions whose ANSWER is unknown. Absent code
is not unknown research; it is unwritten code, and calling it research excuses it.
"""
from __future__ import annotations

from typing import Any

CLASSES = (
    "SOFTWARE_CONNECTION_REMAINING",   # the parts exist; nothing CALLS them
    "VERIFIER_MISSING",                # wired and accepted; nothing PROVES it
    "SOFTWARE_BUILD_REQUIRED",         # the code does not exist yet and must be written
    "EXPERIMENTATION_REQUIRED",        # built and verified; its criterion has never been run
    "LONG_RUN_EVIDENCE_REQUIRED",      # a run that must occupy real wall time
    "EXTERNAL_ENVIRONMENT_REQUIRED",   # an absent package, browser, toolchain or account
    "PHYSICAL_HARDWARE_REQUIRED",      # silicon that is not here
    "DEFERRED_PROGRAM",                # a whole campaign nobody has started
    "UNKNOWN_RESEARCH",                # the answer itself is unknown
)

# Named programs that are not "evidence pending" but "nobody has begun this".
_DEFERRED_PREFIXES = ("THEIA_",)

# Blocker text that means an absent external environment rather than absent work.
_EXTERNAL_MARKERS = (
    "extra", "blender", "chrome", "browser", "install", "package", "toolchain",
    # visionmcp is a separate product this repo consults, not Hawking code, so
    # anything parked on it is an absent environment rather than absent work.
    "visionmcp",
)


def classify(gate: dict[str, Any]) -> tuple[str, str]:
    """(class, the exact thing that is missing). Derived from evidence, never assigned."""
    gid = str(gate.get("id") or "")
    status = gate.get("status")

    if gate.get("hardware_blocker") or status == "BLOCKED_HARDWARE":
        return "PHYSICAL_HARDWARE_REQUIRED", f"silicon absent; wakes on {gate.get('wake_condition')}"

    if status == "UNREACHABLE":
        deps = ", ".join(gate.get("dependencies") or []) or "unnamed dependencies"
        return "PHYSICAL_HARDWARE_REQUIRED", f"dependencies unsatisfied: {deps}"

    blocker = str(gate.get("software_blocker") or "")
    if status == "BLOCKED_EXTERNAL" or blocker:
        if gid.startswith(_DEFERRED_PREFIXES):
            return "DEFERRED_PROGRAM", blocker[:220] or "a program nobody has started"
        if any(m in blocker.lower() for m in _EXTERNAL_MARKERS):
            return "EXTERNAL_ENVIRONMENT_REQUIRED", blocker[:220]
        return "LONG_RUN_EVIDENCE_REQUIRED", blocker[:220]

    wired = bool((gate.get("wired") or {}).get("value"))
    accepted = bool((gate.get("accepted") or {}).get("value"))
    has_impl = bool(gate.get("code_refs"))
    has_test = bool(gate.get("tests"))

    if not has_impl:
        # No implementation at all. That is code nobody wrote, NOT a research
        # question -- calling it research would excuse it from ever being built.
        return "SOFTWARE_BUILD_REQUIRED", "no implementation exists; the code must be written"
    if not wired:
        return "SOFTWARE_CONNECTION_REMAINING", "no non-test call site reaches this capability"
    if not has_test:
        # NOT a connection problem. Nothing is disconnected -- the capability has
        # real callers and, for a BUILT gate, a passed acceptance. What is missing
        # is a test that CITES it. Filing that under the same name as "nothing
        # calls this" sent an operator hunting for a caller that already exists
        # three times over, which is precisely the unlike-things-forced-together
        # defect this module's docstring says the five old classes caused.
        return "VERIFIER_MISSING", "wired but nothing verifies it"
    if not accepted:
        return "EXPERIMENTATION_REQUIRED", "wired and verified; its acceptance criterion has never been run"
    return "", "already integrated"
