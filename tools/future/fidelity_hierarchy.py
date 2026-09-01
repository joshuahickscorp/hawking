#!/usr/bin/env python3
"""G108: a candidate may not be killed at a bar stricter than its claim.

S030 §2 and S031. Five levels, cheapest and strictest-in-the-wrong-way first:

    REPRESENTATION_FIDELITY    do the stored numbers match
    LOCAL_FUNCTIONAL_FIDELITY  does the layer output match
    DOWNSTREAM_BEHAVIOR        do the logits or tokens match
    CAPABILITY                 can the model still do the task
    HCLI_MISSION_CAPABILITY    can it still operate this laboratory

The defect this exists to prevent is concrete and expensive. Byte elimination
licensed 27.7 MB - 0.28% of the token - against a COSINE 0.99 per-region bar on
36 sampled regions, and concluded "nothing is licensed to drop". Reaching 60 TPS
needs 1773 MB. That is a 64x shortfall produced by a FIDELITY bar answering a
CAPABILITY question.

The rule: a refusal is only valid at or below the level of the claim being made.
A candidate claiming CAPABILITY may not be refused for failing
REPRESENTATION_FIDELITY, because the model demonstrably tolerates local
distortion that strict local bars would reject.

    python3 tools/future/fidelity_hierarchy.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/fidelity_hierarchy.py"
RECEIPT_NAME = "FIDELITY_HIERARCHY.json"

# Ordered. Index is the strength of evidence about USEFULNESS, not about
# numerical agreement - a higher level says more about whether the model still
# does the job, and less about whether the bits match.
LEVELS = (
    "REPRESENTATION_FIDELITY",
    "LOCAL_FUNCTIONAL_FIDELITY",
    "DOWNSTREAM_BEHAVIOR",
    "CAPABILITY",
    "HCLI_MISSION_CAPABILITY",
)
RANK = {name: i for i, name in enumerate(LEVELS)}

WHAT_EACH_MEASURES = {
    "REPRESENTATION_FIDELITY": "the stored numbers against the parent's numbers",
    "LOCAL_FUNCTIONAL_FIDELITY": "one layer's output against the incumbent's",
    "DOWNSTREAM_BEHAVIOR": "logits or emitted tokens",
    "CAPABILITY": "whether the model still performs a task correctly",
    "HCLI_MISSION_CAPABILITY": "whether it can still run this laboratory",
}

WHAT_EACH_CANNOT_ESTABLISH = {
    "REPRESENTATION_FIDELITY":
        "anything about behaviour; the codes are 93.5% independent information "
        "and matching them is neither necessary nor sufficient for capability",
    "LOCAL_FUNCTIONAL_FIDELITY":
        "capability. Zeroing 40% of a tensor's output rows moves the hidden "
        "state by 0.0059 of cosine, so this level is nearly flat exactly where "
        "the interesting question lives",
    "DOWNSTREAM_BEHAVIOR":
        "capability on tasks not in the probe set, and nothing about long-"
        "horizon or tool-using behaviour",
    "CAPABILITY":
        "whether autonomous operation survives; a body can keep benchmark "
        "scores and lose structured tool use",
    "HCLI_MISSION_CAPABILITY":
        "general capability outside this laboratory's tasks",
}

# Every refutation this campaign relies on, labelled with the level it was
# ACTUALLY measured at. Nothing here is re-judged; the labels are read off the
# receipts' own methods.
REFUTATIONS = (
    {
        "id": "mlp_code_entropy_1p87",
        "claim_made": "the codes cannot be losslessly compressed much further",
        "measured_at": "REPRESENTATION_FIDELITY",
        "source": "receipts/future/MLP_CODE_INFORMATION.json",
        "still_binds": True,
        "why": "a lossless-coding claim measured by a coding bar. The level and "
               "the claim agree, so this refutation stands.",
    },
    {
        "id": "gate_up_mutual_information",
        "claim_made": "joint coding of gate and up beats marginal coding",
        "measured_at": "REPRESENTATION_FIDELITY",
        "source": "receipts/future/_G118_ROLE_PROBE_raw.json",
        "still_binds": True,
        "why": "again a coding claim answered by a coding measurement: mutual "
               "information 0.00059 bits per weight pair.",
    },
    {
        "id": "shared_linear_low_rank",
        "claim_made": "a shared low-rank basis can replace the MLP factors",
        "measured_at": "LOCAL_FUNCTIONAL_FIDELITY",
        "source": "negative_index: relative L2 0.9 including an oracle PCA",
        "still_binds": False,
        "why": "the claim was about REPLACING A FUNCTION, which is a capability "
               "claim, and it was refused at relative L2. An oracle at rank 64 "
               "also sat at 0.9, so the structural result is strong - but the "
               "BAR was local. Whether 0.9 relative L2 costs capability was "
               "never measured.",
    },
    {
        "id": "affine_group_256_1024",
        "claim_made": "larger quantization groups preserve the model",
        "measured_at": "LOCAL_FUNCTIONAL_FIDELITY",
        "source": "receipts/future/AUX_CAPABILITY_SCREEN.json",
        "still_binds": False,
        "why": "recorded CAPABILITY REFUTED, but the screen it failed is a "
               "local one. It also cuts aux bytes, which bill 0.000 ms/GB, so "
               "reopening it would buy size and not speed either way.",
    },
    {
        "id": "capability_information_map_allocation",
        "claim_made": "no region is licensed to drop bits",
        "measured_at": "LOCAL_FUNCTIONAL_FIDELITY",
        "source": "receipts/future/CAPABILITY_INFORMATION_MAP.json",
        "still_binds": False,
        "why": "THE CENTRAL DEFECT. Named 'capability' and gated on cosine 0.99 "
               "per region over 36 sampled regions. It licensed 27.7 MB, 0.28% "
               "of the token, against the 1773 MB that 60 TPS needs - a 64x "
               "shortfall produced by the bar, not by the model.",
    },
    {
        "id": "functional_role_gate_dominant",
        "claim_made": "the gate deserves literal storage and up/down do not",
        "measured_at": "LOCAL_FUNCTIONAL_FIDELITY",
        "source": "receipts/future/FUNCTIONAL_ROLE_PROBE.json",
        "still_binds": True,
        "why": "the claim was about relative sensitivity under matched damage, "
               "which is what was measured. Gate never exceeded 1.31x up. The "
               "level and the claim agree.",
    },
)


class HierarchyRefused(RuntimeError):
    """A level was named that is not in the hierarchy."""


def rank(level: str) -> int:
    if level not in RANK:
        raise HierarchyRefused(f"{level!r} is not one of {LEVELS}")
    return RANK[level]


def may_refuse(*, claim_level: str, measured_level: str) -> dict[str, Any]:
    """S030 §2: a refusal is valid only at or below the level of the claim."""
    c, m = rank(claim_level), rank(measured_level)
    ok = m >= c
    return {
        "claim_level": claim_level,
        "measured_level": measured_level,
        "refusal_is_valid": ok,
        "why": (
            f"the measurement is at or above the claim, so it can speak to it"
            if ok else
            f"{measured_level} is WEAKER evidence about usefulness than "
            f"{claim_level}. Refusing a {claim_level} claim on a "
            f"{measured_level} measurement is the 64x error that "
            "CAPABILITY_INFORMATION_MAP made."
        ),
        "what_the_measurement_cannot_establish":
            WHAT_EACH_CANNOT_ESTABLISH[measured_level],
    }


def labelled_refutations() -> list[dict[str, Any]]:
    out = []
    for r in REFUTATIONS:
        rank(r["measured_at"])
        out.append({**r, "measured_rank": RANK[r["measured_at"]]})
    return out


def reopenable() -> dict[str, Any]:
    """Which refutations do NOT bind a capability-level claim."""
    rows = labelled_refutations()
    loose = [r for r in rows if not r["still_binds"]]
    return {
        "n_refutations": len(rows),
        "n_still_binding": sum(1 for r in rows if r["still_binds"]),
        "n_not_binding_a_capability_claim": len(loose),
        "ids": [r["id"] for r in loose],
        "rule": (
            "a refutation binds a claim only if it was measured at or above the "
            "claim's level. These were measured below CAPABILITY, so they do "
            "not by themselves refuse a capability-level candidate."
        ),
        "this_is_not_permission_to_delete": (
            "nothing here says these approaches WORK. It says the evidence "
            "against them answers a different question than the one a "
            "capability claim asks. Reopening one requires the capability "
            "measurement that was never taken."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "obligation": "G108",
        "authority": "S030 §2, §40, §41; S031; S032 §8",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "levels": list(LEVELS),
        "what_each_measures": WHAT_EACH_MEASURES,
        "what_each_cannot_establish": WHAT_EACH_CANNOT_ESTABLISH,
        "refutations": labelled_refutations(),
        "reopenable": reopenable(),
        "the_defect_this_prevents": (
            "CAPABILITY_INFORMATION_MAP is named for capability and gated on "
            "cosine 0.99 per region. It licensed 27.7 MB against the 1773 MB "
            "that 60 TPS requires - 64x short - and concluded that nothing is "
            "licensed to drop. The conclusion follows from the bar, not from "
            "the model."
        ),
        "the_rule": (
            "a candidate may not be refused at a level stricter than its claim. "
            "Local fidelity becomes DIAGNOSTIC for capability claims rather "
            "than authority."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    doc = build()
    if a.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps({k: doc[k] for k in
                      ("levels", "reopenable", "the_defect_this_prevents")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
