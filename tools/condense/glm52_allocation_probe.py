#!/usr/bin/env python3
"""Can any legal asymmetric allocation give the expert path the rate it needs?

Section 6.5 authorizes spending bits unevenly across roles, and the pilot found an uneven
failure: dense blocks clear the trajectory floor under product quantization while sparse
MoE blocks are family-bound.  The obvious move is to take bits from what survives and give
them to what does not.

This settles whether that move exists, before anyone builds it.  The arithmetic is short
and it is exact, because the weight shares come from the official index and the ceiling
is the campaign's own law:

    complete = share_experts * rate_experts + share_other * rate_other + native_floor

Routed experts are 97.492 percent of GLM-5.2's weight.  Everything else together is 2.48
percent.  So the most generous possible allocation, one that spends literally nothing on
attention, embeddings, the shared expert and the head, still buys the expert path only

    (ceiling - native_floor) / share_experts

bits.  If the pilot's measured requirement sits above that, no allocation closes the gap
and the answer is a different representation, not a different split.

    solve       write GLM52_GENERATION_B_ALLOCATION_PROBE.json
    selftest
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

REPO = HERE.parent.parent
REPORTS = REPO / "reports/condense/glm52_generation_b"

# Element counts straight from the official index crossed with the tensor contract.
# Not estimates: these are the numbers the coverage audit produced.
TOTAL_ELEMENTS = 753_329_940_480
ROLE_ELEMENTS = {
    "routed_expert": 734_439_407_616,
    "attention": 13_036_552_192,
    "shared_expert": 2_868_903_936,
    "embeddings": 951_582_720,
    "lm_head": 951_582_720,
    "dense_mlp": 679_477_248,
    "mtp_projection": 75_497_472,
    "protected_native": 326_936_576,
}

NATIVE_ELEMENTS = ROLE_ELEMENTS["protected_native"]
NATIVE_RATE = 16.0
BPW_CEILING = 1.0
PRIMARY_CEILING = 0.75
# The floor a compressible role cannot go below and still be a representation: one index
# bit per weight is already the degenerate end of the ladder.
MIN_COMPRESSIBLE_RATE = 0.0


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False).stdout.strip()


def shares() -> dict[str, float]:
    return {role: count / TOTAL_ELEMENTS for role, count in ROLE_ELEMENTS.items()}


def native_floor() -> float:
    return NATIVE_ELEMENTS / TOTAL_ELEMENTS * NATIVE_RATE


def max_expert_rate(ceiling: float, *, other_rate: float = MIN_COMPRESSIBLE_RATE) -> float:
    """The most the expert path can be given, at a ceiling, after paying everyone else.

    `other_rate` at zero is not achievable, it is the bound: it assumes attention,
    embeddings, the head, the shared expert and the dense MLPs are all free.
    """
    share = shares()
    expert_share = share["routed_expert"]
    other_share = sum(value for role, value in share.items()
                      if role not in ("routed_expert", "protected_native"))
    budget = ceiling - native_floor() - other_share * other_rate
    return budget / expert_share


def measured_requirement() -> dict:
    """What the pilot said the expert path needs, taken from the sealed results."""
    path = REPORTS / "GLM52_GENERATION_B_PILOT_RESULTS.json"
    if not path.exists():
        return {"status": "PILOT_NOT_SEALED"}
    results = json.loads(path.read_text())
    rows = {}
    for window, block in results["windows"].items():
        verdict = block["verdict"]
        if verdict["verdict"] != "REPRESENTATION_FAMILY_BOUND":
            continue
        rows[window] = {
            "extrapolated_bpw_to_reach_floor": verdict["extrapolated_bpw_to_reach_floor"],
            "cosine_per_bit": verdict["cosine_per_bit"],
            "best_carry_out_cosine": verdict["best_carry_out_cosine"],
        }
    if not rows:
        return {"status": "NO_FAMILY_BOUND_WINDOWS"}
    needed = [row["extrapolated_bpw_to_reach_floor"] for row in rows.values()
              if row["extrapolated_bpw_to_reach_floor"]]
    return {"status": "FROM_SEALED_PILOT", "windows": rows,
            "min_required_bpw": min(needed) if needed else None,
            "max_required_bpw": max(needed) if needed else None}


def solve() -> int:
    share = shares()
    requirement = measured_requirement()

    scenarios = []
    for label, ceiling in (("primary_candidate", PRIMARY_CEILING),
                           ("one_bit_law", BPW_CEILING)):
        for other_rate, description in ((0.0, "everything else free, the absolute bound"),
                                        (0.25, "everything else at a quarter bit"),
                                        (0.50, "everything else at half a bit")):
            scenarios.append({
                "ceiling": ceiling, "ceiling_role": label,
                "non_expert_rate": other_rate, "assumption": description,
                "max_expert_rate": max_expert_rate(ceiling, other_rate=other_rate),
            })

    best = max(row["max_expert_rate"] for row in scenarios)
    required = requirement.get("min_required_bpw")
    closes = bool(required is not None and best >= required)

    payload = {
        "schema": "hawking.glm52.generation_b_allocation_probe.v1",
        "generated_at": now(), "git_commit": git_head(),
        "question": ("can section 6.5 asymmetric allocation give the expert path the rate "
                     "the pilot says it needs, without breaking the one-bit law"),
        "weight_shares": share,
        "native_floor_bpw": native_floor(),
        "scenarios": scenarios,
        "most_generous_expert_rate": best,
        "measured_requirement": requirement,
        "allocation_closes_the_gap": closes,
        "verdict": ("ALLOCATION_SUFFICIENT" if closes else "ALLOCATION_CANNOT_CLOSE"),
        "reading": (
            "routed experts are {:.3f} of the weight, so the budget cannot be moved to "
            "them: even spending nothing at all on attention, embeddings, the head, the "
            "shared expert and the dense MLPs leaves the expert path {:.4f} bits under the "
            "one-bit law, against a measured requirement of {} to {}. Asymmetric "
            "allocation is the right instinct and the wrong lever here: there is no other "
            "role large enough to donate from."
            .format(share["routed_expert"], best,
                    requirement.get("min_required_bpw"),
                    requirement.get("max_required_bpw"))
            if not closes else
            "allocation reaches the measured requirement and should be tried before any "
            "new representation is built"),
        "what_this_does_not_close": (
            "this bounds product quantization at a rate, not the expert path in general. A "
            "representation that reaches the trajectory floor at a lower rate is untested "
            "and is exactly what sections 6.1 to 6.3 propose."),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    target = REPORTS / "GLM52_GENERATION_B_ALLOCATION_PROBE.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({
        "wrote": str(target.relative_to(REPO)),
        "routed_expert_weight_share": round(share["routed_expert"], 5),
        "max_expert_rate_under_one_bit": round(best, 4),
        "measured_requirement_bpw": [requirement.get("min_required_bpw"),
                                     requirement.get("max_required_bpw")],
        "verdict": payload["verdict"],
    }, indent=2))
    return 0


def selftest() -> int:
    share = shares()
    assert abs(sum(ROLE_ELEMENTS.values()) - TOTAL_ELEMENTS) / TOTAL_ELEMENTS < 1e-6, \
        "role element counts do not sum to the model"
    assert abs(share["routed_expert"] - 0.97492) < 1e-4, share["routed_expert"]
    assert abs(native_floor() - 0.00694) < 1e-4, native_floor()

    # The bound must be monotone in the ceiling and in what everyone else is given.
    assert max_expert_rate(1.0) > max_expert_rate(0.75)
    assert max_expert_rate(1.0, other_rate=0.0) > max_expert_rate(1.0, other_rate=0.5)

    # And it must never exceed what the ceiling could buy if the model were all experts.
    assert max_expert_rate(1.0) < 1.0 / share["routed_expert"] + 1e-9

    # A role carrying 97 percent of the weight cannot be subsidised: taking every bit from
    # everything else moves the expert rate by less than three percent.
    generous = max_expert_rate(1.0, other_rate=0.0)
    stingy = max_expert_rate(1.0, other_rate=0.5)
    assert (generous - stingy) / generous < 0.03, (generous, stingy)

    print("glm52_allocation_probe selftest OK")
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "solve"
    raise SystemExit({"solve": solve, "selftest": selftest}[command]())
