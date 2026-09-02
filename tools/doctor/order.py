"""H-ROADMAP §9.1 Doctor technique order.

The order is the law. Doctor asks these questions in this sequence; a
later stage does not run in place of an earlier one. QUANTIZE is what
remains when the zeros have been asked and have failed — it is not a
shortcut around ELIMINATE / SHARE / GENERATE / ROUTE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Technique:
    index: int
    name: str
    family: str
    zeros: tuple[str, ...]
    question: str
    cheapest_experiment: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "family": self.family,
            "zeros": list(self.zeros),
            "question": self.question,
            "cheapest_experiment": self.cheapest_experiment,
        }


# Roadmap 9.1, copied as data. Do not reorder.
TECHNIQUE_ORDER: tuple[Technique, ...] = (
    Technique(
        0,
        "ELIMINATE",
        "DOC-ELIMINATION",
        ("ZERO_STORAGE",),
        "SHOULD THIS STRUCTURE EXIST?",
        "cite architecture for unused/duplicated/tied structures; do not load weights",
    ),
    Technique(
        1,
        "REPARAMETERIZE / coordinate shaping",
        "DOC-COORDINATES",
        ("ZERO_INDEPENDENT_INFORMATION",),
        "MUST IT EXIST IN THIS COORDINATE SYSTEM?",
        "outlier / incoherence probe is ABSENT on metadata; cite COORDINATE_TRANSFORM_PROBE if present",
    ),
    Technique(
        2,
        "SHARE",
        "DOC-SHARE",
        ("ZERO_STORAGE", "ZERO_INDEPENDENT_INFORMATION"),
        "MUST IT EXIST INDEPENDENTLY?",
        "cross-expert / cross-layer cosine from an existing Doctor screen receipt, never a new weight pass",
    ),
    Technique(
        3,
        "FACTORIZE",
        "DOC-FACTOR",
        ("ZERO_INDEPENDENT_INFORMATION",),
        "MUST IT EXIST INDEPENDENTLY AS A FULL-RANK BODY?",
        "singular-spectrum energy from an existing screen receipt; do not SVD a parent tensor here",
    ),
    Technique(
        4,
        "GENERATE",
        "DOC-GENERATE",
        ("ZERO_STORAGE", "ZERO_INDEPENDENT_INFORMATION"),
        "MUST IT BE STORED RATHER THAN GENERATED?",
        "config/receipt declaration of a generator; absence is FAIL not a download",
    ),
    Technique(
        5,
        "ROUTE",
        "DOC-ROUTE",
        ("ZERO_EXECUTION",),
        "MUST IT EXECUTE FOR EVERY TOKEN?",
        "num_experts vs experts_per_tok from config; routing is architecture evidence",
    ),
    Technique(
        6,
        "SENSITIVITY-AWARE INFORMATION ASSIGNMENT",
        "DOC-SENSITIVITY",
        (),
        "MUST IT RETAIN THE SAME PRECISION AS ITS NEIGHBORS?",
        "organ-local sensitivity from an existing receipt; Hessian is ABSENT",
    ),
    Technique(
        7,
        "HEAL",
        "DOC-HEALING",
        (),
        "CAN A CORRECTION RESTORE A CRUSHED ORGAN?",
        "query negative science (low-rank never heals) before proposing a healer",
    ),
    Technique(
        8,
        "QUANTIZE",
        "DOC-REPRESENTATION",
        (),
        "WHAT IS ITS BEST PHYSICAL REPRESENTATION?",
        "only after zeros have been asked; all-three-FAIL is ordinary quantization (9.2)",
    ),
    Technique(
        9,
        "NATIVE OPERATORS",
        "DOC-NATIVE",
        (),
        "WHAT NATIVE OPERATOR SHOULD EXECUTE IT?",
        "native_kernel_status from an existing receipt; PLAN_ONLY is not a kernel",
    ),
    Technique(
        10,
        "RUNTIME STATE",
        "DOC-STATE",
        ("ZERO_EXECUTION",),
        "MUST THE STATE BE MATERIALIZED ON THE CRITICAL PATH?",
        "DeltaNet / KV / recurrent fields from config; do not dump state tensors",
    ),
    Technique(
        11,
        "REMOVE COMPUTE",
        "DOC-CONDITIONAL",
        ("ZERO_EXECUTION",),
        "MUST IT EXECUTE FOR EVERY TOKEN?",
        "mixture-of-depths / activation sparsity: UNKNOWN unless a receipt says otherwise",
    ),
    Technique(
        12,
        "REDUCE DECODE FORWARDS",
        "DOC-DECODE",
        ("ZERO_EXECUTION",),
        "MUST EVERY DECODE STEP BE A FULL FORWARD?",
        "MTP / n-gram / speculative fields from config or FLASH_EBPW organ census",
    ),
    Technique(
        13,
        "DEVICE COMPILE",
        "DOC-DEVICE",
        (),
        "WHAT DEVICE PROFILE BEST RUNS THAT OPERATOR?",
        "STATIC only on this host; FPGA/U50/DGX/eGPU are models, never measurements",
    ),
    Technique(
        14,
        "VERIFY COMPLETE FUNCTION",
        "DOC-VERIFY",
        (),
        "DOES THE EXECUTABLE STILL DO THE JOB?",
        "capability contract from an existing receipt; this lane does not run one",
    ),
)


assert tuple(t.index for t in TECHNIQUE_ORDER) == tuple(range(15))
assert TECHNIQUE_ORDER[0].name == "ELIMINATE"
assert TECHNIQUE_ORDER[8].name == "QUANTIZE"
assert TECHNIQUE_ORDER[14].name == "VERIFY COMPLETE FUNCTION"


def walk_order() -> tuple[Technique, ...]:
    """The sequence Doctor must ask. Callers iterate this, never a shuffled copy."""
    return TECHNIQUE_ORDER
