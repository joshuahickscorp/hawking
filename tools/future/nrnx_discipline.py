"""Enforce the NR/NX distinction S012 §102 makes mandatory.

    NR = MUTABLE, UNEXECUTED representation. The scientific working body.
    NX = FROZEN, TINY, STANDALONE, qualified EXECUTABLE.

A temporary executable derived from an NR for measurement is NOT NX. Calling a
mutable candidate NX overstates it; calling the frozen executable NR understates
it. Both errors corrupt the campaign's central claim, which is exactly why the
steer says "DO NOT CONFUSE NR AND NX AGAIN".

This is a lint, not a theorem: it flags phrasings that assert NX status for
something with no promotion evidence. It is deliberately narrow -- prose ABOUT
the distinction is not a violation.
"""
from __future__ import annotations

import re
from pathlib import Path

# Claims that assert a thing IS an NX. Discussion of what NX must be is fine.
_ASSERTS_NX = [
    re.compile(r"\bthis\s+is\s+(?:the\s+|our\s+)?NX\b", re.I),
    re.compile(r"\bproduced\s+(?:an?\s+)?NX\b", re.I),
    re.compile(r"\bshipped\s+(?:an?\s+)?NX\b", re.I),
    re.compile(r"\bNX\s+is\s+(?:complete|done|finished|ready)\b", re.I),
    re.compile(r"\bpromoted\s+to\s+NX\b", re.I),
]
# The ten conditions G030 requires before any NX claim is admissible.
NX_PROMOTION_CONDITIONS = (
    "standalone_execution", "complete_persistent_accounting", "no_hidden_dense_parent",
    "capability_qualification", "oiii_adversarial_qualification", "physical_qualification",
    "restart_recovery_qualification", "machine_runtime_contract",
    "independent_verification", "immutable_artifact_identity",
)


def violations(text: str, where: str = "") -> list[dict]:
    """Assertions of NX status. Empty list means no claim was made."""
    out = []
    for pat in _ASSERTS_NX:
        for m in pat.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            out.append({"where": where, "line": line, "match": m.group(0),
                        "why": "asserts NX status; NX requires all ten G030 conditions"})
    return out


def nx_claim_is_admissible(record: dict) -> tuple[bool, list[str]]:
    """An NX claim is admissible only with every promotion condition recorded true."""
    missing = [c for c in NX_PROMOTION_CONDITIONS if record.get(c) is not True]
    return (not missing), missing


def scan(paths) -> list[dict]:
    found = []
    for p in paths:
        p = Path(p)
        try:
            found.extend(violations(p.read_text(encoding="utf-8", errors="ignore"), str(p)))
        except Exception:
            continue
    return found
