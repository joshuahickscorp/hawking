#!/usr/bin/env python3
"""N052 — HARDEN DOCTOR: general TechniqueLibrary (S026 §5, §64, §75, §107; S028; CPU).

Merges the N043 registry (15) and N046 RECOMMENDED_ADDITIONS (24) into one
hardened, general Doctor TechniqueLibrary. Doctor is a general physician;
this library is its toolkit across the Odyssey curriculum, not Qwen-only.

Literature is a hypothesis, not authority (S026 §5). A Qwen-MLP failure is
not a prune (S028 / §64 / §107). PRUNE only superseded, strictly-dominated,
or Metal-infeasible techniques.

    python3 tools/headless/doctor_technique_library.py
    python3 -m pytest tools/headless/test_doctor_technique_library.py -q
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from doctor_technique_registry import (  # noqa: E402
    REQUIRED_TECHNIQUE_IDS,
    SCAR_BINARY,
    SCAR_LOWRANK,
    SCAR_SHARED_BASIS,
    SCAR_SPARSE,
    SCAR_TERNARY,
    citation_exists,
    git_head,
    is_hawking_receipt_path,
    load_json,
    now_iso,
    write_json,
)
from literature_frontier import N043_SEED  # noqa: E402

SCHEMA = "hawking.headless.doctor_technique_library.v1"
RECEIPT = REPO / "receipts" / "headless" / "DOCTOR_TECHNIQUE_LIBRARY.json"
DOCS = REPO / "docs" / "ultragoals" / "DOCTOR_TECHNIQUE_LIBRARY.md"
GENERATOR = "tools/headless/doctor_technique_library.py"
N043_RECEIPT = "receipts/headless/DOCTOR_TECHNIQUE_REGISTRY.json"
N046_RECEIPT = "receipts/headless/LITERATURE_FRONTIER.json"
OBLIGATION = (
    "N052 — HARDEN DOCTOR: general TechniqueLibrary (S026 §5, §64, §75, §107; "
    "S028; CPU). Merge N043 (15) + N046 RECOMMENDED_ADDITIONS (24) into one "
    "hardened general Doctor toolkit. Applicability is per architecture class. "
    "A Qwen-MLP failure is not a prune."
)

LITERATURE_STATUS = "HYPOTHESIS"
KEEP = "KEEP"
PRUNE = "PRUNE"
DECISIONS = frozenset({KEEP, PRUNE})
PRUNE_CLASSES = frozenset({"superseded", "strictly_dominated", "metal_infeasible"})

ARCH_CLASSES = (
    "dense_mlp",
    "moe",
    "attention_gqa",
    "recurrent_deltanet",
    "multimodal",
    "kv_state",
    "decoding",
    "tokenizer",
)
GRADES = frozenset({"PLAUSIBLE", "UNLIKELY", "UNKNOWN"})
QWEN_OUTCOMES = frozenset({"worked", "failed", "untested"})

DEFAULT_LICENSE = (
    "arXiv perpetual non-exclusive license to distribute "
    "(default arXiv license; not a grant to ship)"
)

# Hawking receipts cited as Qwen datapoints / scars.
R_N044 = "receipts/headless/COORDINATE_TRANSFORM_PROBE.json"
R_BYTES = "receipts/headless/BYTES_FRONTIER.json"
R_BINARY = "receipts/headless/BINARY_HEALING.json"
R_SHARED_K = "receipts/headless/SHARED_BASIS_KERNEL.json"
R_SHARED_C = "receipts/headless/SHARED_BASIS_COHERENT.json"
R_HYBRID = "receipts/headless/HYBRID_OPERATOR.json"
R_C1 = "receipts/headless/C1SHAREDBASIS_DESIGN.json"
R_C3 = "receipts/headless/C3LOWRANKSPARSE_DESIGN.json"
R_C4 = "receipts/headless/C4CODEBOOK_DESIGN.json"
R_C5 = "receipts/headless/C5STRUCTTRANSFORM_DESIGN.json"
R_PREFILL = "receipts/headless/PREFILL_KV.json"
R_NNS = "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json"
R_CENSUS = "receipts/headless/NOETIC_OPERATION_CENSUS.json"
R_FLOORS = "receipts/headless/ORGAN_DENSITY_FLOORS.json"
R_RECOMPOSE = "receipts/headless/WHOLE_MODEL_RECOMPOSE.json"
R_TOK = "receipts/headless/TOKENIZER_GRAVITY.json"
R_DN = "receipts/headless/DELTANET_ORGAN.json"
R_FRONTIERS = "receipts/headless/ORGAN_FRONTIERS.json"
R_ONEBIT = "receipts/headless/ONEBIT_FAMILIES.json"
R_TERNARY = "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json"
R_QCR = "receipts/headless/QWEN_COMPLETION_RECEIPT.json"

ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")

# N046 recommended name -> library id. Gloeckle MTP merges into N043 medusa_mtp.
N046_SLUG = {
    "HIGGS": "higgs",
    "Gloeckle MTP / Qwen-native MTP census": "gloeckle_mtp",
    "TEAL": "teal",
    "LayerSkip": "layerskip",
    "ButterflyQuant": "butterflyquant",
    "QTIP": "qtip",
    "Quamba2": "quamba2",
    "DFlash": "dflash",
    "OCTOPUS": "octopus",
    "OSCAR": "oscar",
    "TurboQuant": "turboquant",
    "ParoQuant": "paroquant",
    "ResQ": "resq",
    "Sparse Delta Memory": "sparse_delta_memory",
    "HyperQuant": "hyperquant",
    "EAGLE-3": "eagle3",
    "MagicDec": "magicdec",
    "OSTQuant": "ostquant",
    "Palu": "palu",
    "PolarQuant": "polarquant",
    "SuperBPE": "superbpe",
    "FlatQuant": "flatquant",
    "Mixture-of-Recursions": "mixture_of_recursions",
    "TransMLA": "transmla",
}

# Dedup: N043 medusa_mtp already carries Gloeckle MTP (arxiv 2404.19737).
MERGE_N046_INTO = {
    "Gloeckle MTP / Qwen-native MTP census": "medusa_mtp",
}

N043_SHORT_TO_N046 = {
    "Medusa/MTP": "Medusa",
}

# Phrases that mean "we pruned this because Qwen failed" — illegal (S028).
QWEN_PRUNE_PHRASES = (
    "failed on qwen",
    "fails on qwen",
    "failure on qwen",
    "died on qwen",
    "qwen-mlp failure",
    "qwen mlp failure",
    "failed on qwen's mlp",
    "failed on qwen3",
    "because it failed on qwen",
    "qwen result",
)


# ---------------------------------------------------------------------------
# matrix helpers
# ---------------------------------------------------------------------------


def cell(grade: str, reason: str) -> dict[str, str]:
    if grade not in GRADES:
        raise ValueError(grade)
    reason = reason.strip()
    if not reason or "\n" in reason:
        raise ValueError(f"reason must be a one-line non-empty string: {reason!r}")
    return {"grade": grade, "reason": reason}


def as_matrix(cells: dict[str, tuple[str, str]]) -> dict[str, dict[str, str]]:
    missing = [c for c in ARCH_CLASSES if c not in cells]
    extra = [c for c in cells if c not in ARCH_CLASSES]
    if missing or extra:
        raise ValueError(f"applicability cells missing={missing} extra={extra}")
    return {cls: cell(g, r) for cls, (g, r) in cells.items()}


def weight_ptq_matrix(dense_mlp: tuple[str, str], **override: tuple[str, str]) -> dict[str, dict[str, str]]:
    """Linear-weight codec / rotation / codebook / trit / binary."""
    cells: dict[str, tuple[str, str]] = {
        "dense_mlp": dense_mlp,
        "moe": (
            "UNKNOWN",
            "Expert Linear maps are the same algebra; a Qwen dense-MLP result does not transfer (S028/§64/§107).",
        ),
        "attention_gqa": (
            "UNKNOWN",
            "GQA Q/O GEMVs were not the Qwen MLP composition probe; keep as a hypothesis.",
        ),
        "recurrent_deltanet": (
            "UNKNOWN",
            "DeltaNet in_proj is a different organ (floor 3.26 EBPW) and was not this codec's Qwen test.",
        ),
        "multimodal": (
            "UNKNOWN",
            "No vision/audio organ is in the Qwen3.8 hybrid census.",
        ),
        "kv_state": (
            "UNLIKELY",
            "This is a weight codec, not a KV/state codec.",
        ),
        "decoding": (
            "UNLIKELY",
            "Does not change accepted-tokens-per-forward; at best it cheapens a draft body.",
        ),
        "tokenizer": (
            "UNLIKELY",
            "Does not change vocabulary topology or tokenizer bytes.",
        ),
    }
    cells.update(override)
    return as_matrix(cells)


def kv_matrix(
    gqa: tuple[str, str],
    kv_state: tuple[str, str] | None = None,
    **override: tuple[str, str],
) -> dict[str, dict[str, str]]:
    cells: dict[str, tuple[str, str]] = {
        "dense_mlp": (
            "UNLIKELY",
            "KV/state codec; does not compress MLP weights.",
        ),
        "moe": (
            "UNKNOWN",
            "MoE routing does not by itself change GQA KV algebra; untested on routed parents.",
        ),
        "attention_gqa": gqa,
        "recurrent_deltanet": (
            "UNLIKELY",
            "DeltaNet state is a recurrent summary, not a token KV cache (NOETIC_CANON law 14).",
        ),
        "multimodal": (
            "UNKNOWN",
            "Cross-modal KV untested; no multimodal organ on this hybrid.",
        ),
        "kv_state": kv_state or ("PLAUSIBLE", gqa[1]),
        "decoding": (
            "UNKNOWN",
            "Smaller session state can change long-context decode economics; unmeasured here.",
        ),
        "tokenizer": (
            "UNLIKELY",
            "Does not change vocabulary topology.",
        ),
    }
    cells.update(override)
    return as_matrix(cells)


def decode_matrix(decoding: tuple[str, str], **override: tuple[str, str]) -> dict[str, dict[str, str]]:
    cells: dict[str, tuple[str, str]] = {
        "dense_mlp": (
            "UNLIKELY",
            "Speculative/MTP changes passes-per-token, not the MLP byte wall.",
        ),
        "moe": (
            "UNKNOWN",
            "Shared-expert draft is the in-tree MoE analogue (hawking-speculate); untested as this paper.",
        ),
        "attention_gqa": (
            "UNKNOWN",
            "Verify still pays GQA; draft may skip or reuse it. Untested.",
        ),
        "recurrent_deltanet": (
            "UNKNOWN",
            "Draft/verify over a hybrid GQA+DeltaNet body is untested; DN state is not prefix-shareable.",
        ),
        "multimodal": (
            "UNKNOWN",
            "No multimodal decode organ in this census.",
        ),
        "kv_state": (
            "UNKNOWN",
            "Speculation needs dual committed/provisional KV (in-tree); byte win is not the paper's claim.",
        ),
        "decoding": decoding,
        "tokenizer": (
            "UNLIKELY",
            "Does not change vocabulary topology.",
        ),
    }
    cells.update(override)
    return as_matrix(cells)


def ssm_matrix(dn: tuple[str, str], **override: tuple[str, str]) -> dict[str, dict[str, str]]:
    cells: dict[str, tuple[str, str]] = {
        "dense_mlp": (
            "UNLIKELY",
            "SSM/DeltaNet codec; does not target SwiGLU MLP.",
        ),
        "moe": (
            "UNKNOWN",
            "Routed SSM experts are an Odyssey parent, not this dense hybrid.",
        ),
        "attention_gqa": (
            "UNLIKELY",
            "GQA is softmax attention, not a delta-rule recurrence.",
        ),
        "recurrent_deltanet": dn,
        "multimodal": (
            "UNKNOWN",
            "No multimodal SSM organ in this census.",
        ),
        "kv_state": (
            "UNKNOWN",
            "DN session state is the other side of the 0.015 in_proj-irreducible law; depends on the paper.",
        ),
        "decoding": (
            "UNLIKELY",
            "Does not change accepted-tokens-per-forward.",
        ),
        "tokenizer": (
            "UNLIKELY",
            "Does not change vocabulary topology.",
        ),
    }
    cells.update(override)
    return as_matrix(cells)


def tokenizer_matrix(tok: tuple[str, str], **override: tuple[str, str]) -> dict[str, dict[str, str]]:
    cells: dict[str, tuple[str, str]] = {
        "dense_mlp": (
            "UNLIKELY",
            "Tokenizer/vocab technique; does not change MLP codecs.",
        ),
        "moe": (
            "UNLIKELY",
            "Does not change expert routing.",
        ),
        "attention_gqa": (
            "UNKNOWN",
            "Shorter sequences shrink GQA KV growth; only if the tokenizer is actually swapped.",
        ),
        "recurrent_deltanet": (
            "UNLIKELY",
            "DeltaNet state does not grow with sequence (TOKENIZER_GRAVITY); vocab swap is not a DN codec.",
        ),
        "multimodal": (
            "UNKNOWN",
            "Multimodal tokenizers untested.",
        ),
        "kv_state": (
            "UNKNOWN",
            "Token inflation changes GQA KV positions; N045 showed inflation can erase a head cut.",
        ),
        "decoding": (
            "UNLIKELY",
            "Does not change speculative draft/verify.",
        ),
        "tokenizer": tok,
    }
    cells.update(override)
    return as_matrix(cells)


N044_MLP = (
    "N044 ROTATION_MOVES_BARRIER=false on Qwen3.8 MLP "
    f"({R_N044}): Hadamard and learned block-orthogonal did not move "
    "held-out composition; the 2.25 floor is coordinate-robust."
)
S028_KEEP = "S028/§64/§107: a Qwen-MLP failure is not global death."


def qwen(
    outcome: str,
    receipt: str | None,
    scope: str,
    *,
    related_not_the_same: str | None = None,
) -> dict[str, Any]:
    if outcome not in QWEN_OUTCOMES:
        raise ValueError(outcome)
    doc = {
        "outcome": outcome,
        "receipt": receipt,
        "scope": scope,
        "related_not_the_same": related_not_the_same,
    }
    return doc


def experiment(
    xid: str,
    summary: str,
    *,
    success: str | None = None,
    cpu_only: bool = True,
    loads_model: bool = False,
    touches_gpu: bool = False,
) -> dict[str, Any]:
    return {
        "id": xid,
        "summary": summary,
        "cpu_only": cpu_only,
        "loads_model": loads_model,
        "touches_gpu": touches_gpu,
        "no_second_27b": True,
        "real_x_not_gaussian": True,
        "success_criterion": success,
    }


def wrap_n043_experiment(raw: dict[str, Any] | None, fallback_id: str) -> dict[str, Any]:
    raw = raw or {}
    summary = raw.get("name") or raw.get("why_cheapest") or fallback_id
    why = raw.get("why_cheapest")
    if why and why not in summary:
        summary = f"{summary}: {why}"
    return experiment(
        raw.get("id") or fallback_id,
        summary,
        success=raw.get("success_criterion"),
        cpu_only=bool(raw.get("cpu_only", True)),
        loads_model=bool(raw.get("loads_model", False)),
        touches_gpu=bool(raw.get("touches_gpu", False)),
    )


def wrap_n046_experiment(tid: str, text: str) -> dict[str, Any]:
    return experiment(f"HX-{tid.upper().replace('_', '-')}", text)


# ---------------------------------------------------------------------------
# per-technique hardening (S028). Keys are library ids.
# ---------------------------------------------------------------------------


def _hardening() -> dict[str, dict[str, Any]]:
    """KEEP/PRUNE, applicability, Qwen datapoint. Reasons are citations, not vibes."""
    rot_qwen = qwen(
        "failed",
        R_N044,
        "Qwen3.8 MLP composition (layers 0 and 31 gate/up/down + SwiGLU); "
        "not a global-architecture verdict.",
        related_not_the_same="N044 tested Hadamard + PCA-orthogonal, not every paper's parameterization.",
    )
    return {
        # ----- N043 seed -----
        "spinquant": {
            "applicability": weight_ptq_matrix(
                ("UNLIKELY", N044_MLP + " " + S028_KEEP),
                kv_state=(
                    "UNKNOWN",
                    "Absorbed weight rotation is not a KV codec; OSCAR/TurboQuant are the KV-rotation family.",
                ),
            ),
            "qwen_datapoint": rot_qwen,
            "decision": KEEP,
            "decision_reason": (
                "N044 closed the Qwen-MLP rotation reopen; KEEP because GQA/DeltaNet/MoE "
                "absorption is UNKNOWN and a purpose-rejection is not global death (S026 §64, S028)."
            ),
        },
        "twla": {
            "applicability": weight_ptq_matrix(
                (
                    "UNLIKELY",
                    "Ternary 5-in-8 on this body is slower + argmax-flip "
                    f"({R_BYTES}, {R_TERNARY}); N044 did not move 1.58 in rotated coords. {S028_KEEP}",
                ),
            ),
            "qwen_datapoint": qwen(
                "failed",
                R_TERNARY,
                "Native ternary 5-in-8 g64 MLP flipped whole-model argmax; "
                "TWLA's Kronecker tri-modal + W1.58A4 activations were not in that measurement.",
                related_not_the_same="BYTES_FRONTIER packing is not TWLA's KOTMS coordinate system.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Related ternary packing negative + N044 coordinate-robust 1.58 on Qwen MLP. "
                "KEEP: A4 activations and Kronecker shaping remain untested on other organs/arch (S028)."
            ),
        },
        "cat_q": {
            "applicability": weight_ptq_matrix(
                (
                    "UNLIKELY",
                    "Binary nearby-island healing restored coherent generation 0/4 "
                    f"({R_BINARY}); ternary snap already flipped argmax. {S028_KEEP}",
                ),
            ),
            "qwen_datapoint": qwen(
                "failed",
                R_BINARY,
                "Nearby-weight island/sparse heals on the 1.25 binary body; "
                "CAT-Q's LM+ST ternary modulation was not run.",
                related_not_the_same="BINARY_HEALING islands are not CAT-Q softened ternarization.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Qwen nearby-heal analog died as a resident body. KEEP: LM+ST is a different "
                "calibrator, and MoE/other-arch Linears are UNKNOWN (S028)."
            ),
        },
        "ptqtp": {
            "applicability": weight_ptq_matrix(
                (
                    "UNLIKELY",
                    "Single trit-plane 5-in-8 is slower + argmax-flip; two planes were not run. "
                    f"{S028_KEEP}",
                ),
            ),
            "qwen_datapoint": qwen(
                "failed",
                R_BYTES,
                "Single-plane ternary packing on Qwen MLP. Dual structured trit-planes untested.",
                related_not_the_same="5-in-8 is not PTQTP's additive dual-plane form.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Single-plane packing negative is not a dual-plane refutation. KEEP and require "
                "native execution (S026 §17) before any Metal kernel."
            ),
        },
        "onebit": {
            "applicability": weight_ptq_matrix(
                (
                    "UNLIKELY",
                    "PTQ binary_g64 is faster than q2f and uniformly injured; died at "
                    f"coherent_generation ({R_BINARY}). Dead as a Qwen resident generator.",
                ),
                decoding=(
                    "PLAUSIBLE",
                    "S026 §63-64 / N049: the failed 1.25 body is a legal speculative-draft "
                    "candidate (already faster; hawking-speculate can verify).",
                ),
            ),
            "qwen_datapoint": qwen(
                "failed",
                R_BINARY,
                "Resident generator: mix_c emitted 16 copies of token 271; "
                "n_that_reached_coherent_generation=0. Token_ns moved toward the roof.",
                related_not_the_same="Paper SVID+QAT is not binary_g64 PTQ; draft-acceptance is untested (N049).",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Dead as Qwen final generator (BINARY_HEALING) but KEEP as speculative-draft "
                "candidate (N049 / S026 §63-64). A purpose-rejection is not global death."
            ),
        },
        "aqlm": {
            "applicability": weight_ptq_matrix(
                (
                    "UNLIKELY",
                    "C4 refused the Qwen3.8 gravity_pq LUT port (NOT_WORTH_BUILDING_THE_QWEN38_PORT). "
                    f"{S028_KEEP}",
                ),
            ),
            "qwen_datapoint": qwen(
                "failed",
                R_C4,
                "LUT+accumulate did not beat Q4 dequant+mul on this M3 Ultra for the Qwen3.8 port.",
                related_not_the_same="C4 is gravity_pq VQ, not AQLM's additive multi-codebook fit.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Qwen3.8 PQ port refused; KEEP as the codebook baseline. QTIP/HIGGS are the "
                "cheaper next codebook probes. MoE/other-arch UNKNOWN (S028)."
            ),
        },
        "vptq": {
            "applicability": weight_ptq_matrix(
                (
                    "UNLIKELY",
                    f"HYBRID_OPERATOR: weight residual never heals under 2.25 on Qwen MLP. {S028_KEEP}",
                ),
            ),
            "qwen_datapoint": qwen(
                "failed",
                R_HYBRID,
                "Distributed residual under 1.0 extra bpw did not restore held-out activations.",
                related_not_the_same="VPTQ outlier-channels are a residual cousin, not the exact paper.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Weight-residual heal is dead on Qwen MLP. KEEP: activation-subspace residuals "
                "(ResQ) and other-arch outlier splits remain UNKNOWN (S028)."
            ),
        },
        "caldera": {
            "applicability": weight_ptq_matrix(
                (
                    "UNLIKELY",
                    f"HYBRID_OPERATOR coherent_hybrid_beats_q2f=false; Q+LR never heals on Qwen MLP. {S028_KEEP}",
                ),
            ),
            "qwen_datapoint": qwen(
                "failed",
                R_HYBRID,
                "Binary + distributed low-rank under 2.25 and 27.55 ms: no. Kernel was CLEAR.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Q+LR is dead as a Qwen-MLP byte lever (natively measured). KEEP for other "
                "organs/arch: purpose-scoped, not globally dead (S028)."
            ),
        },
        "squeezellm": {
            "applicability": weight_ptq_matrix(
                (
                    "UNLIKELY",
                    "Sensitivity islands (early16/down/gate/sparse_05) reached coherent generation "
                    f"0/4 ({R_BINARY}). Tax = full q2f body. {S028_KEEP}",
                ),
            ),
            "qwen_datapoint": qwen(
                "failed",
                R_BINARY,
                "SqueezeLLM-style mixed-precision islands on the injured binary body.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Island-heal failed as a Qwen resident body. KEEP: mixed-precision sensitivity "
                "on GQA/DeltaNet/MoE experts is UNKNOWN (S028)."
            ),
        },
        "kivi": {
            "applicability": kv_matrix(
                (
                    "PLAUSIBLE",
                    "Asymmetric 2-bit KV (K per-channel, V per-token) matches the GQA cache "
                    f"that exceeds MODEL_BYTES at 32K×4 ({R_PREFILL}).",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_PREFILL,
                "No Hawking receipt runs KIVI on a captured GQA K/V block. PREFILL_KV sizes the gap.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Untested on this hybrid; the session-state gap is real. KEEP. Not a Qwen-MLP codec."
            ),
        },
        "minicache": {
            "applicability": kv_matrix(
                (
                    "PLAUSIBLE",
                    "16 GQA layers can depth-merge KV; cosine of layer-ℓ vs ℓ+4 is the cheap probe.",
                ),
                recurrent_deltanet=(
                    "UNLIKELY",
                    "DeltaNet recurrent state is a summary, not a cache; MiniCache does not transfer.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_PREFILL,
                "Depth-merge cosine on GQA K unmeasured. DN transfer is predicted dead by organ type.",
            ),
            "decision": KEEP,
            "decision_reason": "Cheap CPU cosine probe still open on the 16 GQA layers. KEEP.",
        },
        "h2o": {
            "applicability": kv_matrix(
                (
                    "PLAUSIBLE",
                    "Heavy-hitter eviction applies to GQA token cache; attention-mass histogram is cheap.",
                ),
                recurrent_deltanet=(
                    "UNLIKELY",
                    "Eviction does not apply to DeltaNet state (not a token cache).",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_PREFILL,
                "No attention-mass histogram on a Qwen GQA layer is on disk.",
            ),
            "decision": KEEP,
            "decision_reason": "Untested GQA-KV policy, not a Qwen-MLP failure. KEEP.",
        },
        "mixture_of_depths": {
            "applicability": as_matrix(
                {
                    "dense_mlp": (
                        "UNLIKELY",
                        "Skipping FLOPs without skipping weight reads does nothing on a load-bound decode.",
                    ),
                    "moe": (
                        "UNKNOWN",
                        "Token-level layer skip on a routed parent is an Odyssey architecture hypothesis.",
                    ),
                    "attention_gqa": (
                        "UNKNOWN",
                        "Conditional GQA skip still pays residual/norm unless ZERO_EXECUTION is fused.",
                    ),
                    "recurrent_deltanet": (
                        "UNKNOWN",
                        "Skipping a DN layer still leaves state updates unspecified.",
                    ),
                    "multimodal": (
                        "UNKNOWN",
                        "No multimodal organ in this census.",
                    ),
                    "kv_state": (
                        "UNLIKELY",
                        "Does not compress KV; may skip writing it if a layer is skipped.",
                    ),
                    "decoding": (
                        "UNKNOWN",
                        "Conditional depth is not speculative decoding; TEAL is the cheaper skip probe.",
                    ),
                    "tokenizer": (
                        "UNLIKELY",
                        "Does not change vocabulary topology.",
                    ),
                }
            ),
            "qwen_datapoint": qwen(
                "untested",
                "receipts/headless/DISPATCH_LEDGER.json",
                "No MoD router was trained. Graph is organ-bound, not dispatch-bound (DISPATCH_LEDGER).",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Architecture/conditional-compute hypothesis for Odyssey, not a Qwen-MLP prune. KEEP."
            ),
        },
        "prosparse": {
            "applicability": weight_ptq_matrix(
                (
                    "UNKNOWN",
                    "SwiGLU sparsity histogram is the cheap probe and is still unrun; "
                    "ReLU-fication is a training change, not PTQ.",
                ),
                decoding=(
                    "UNLIKELY",
                    "Activation sparsity is not speculative decoding.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_NNS,
                "NNS-029 is a related negative on uniform activation-sparsity as a path, "
                "not a ProSparse measurement. TEAL is the training-free alternative.",
                related_not_the_same="A SwiGLU histogram is not ProSparse training.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Untested as ProSparse. KEEP. TEAL is the cheaper training-free cousin; "
                "do not prune ProSparse for an NNS-029 related note (S028)."
            ),
        },
        "medusa_mtp": {
            "applicability": decode_matrix(
                (
                    "PLAUSIBLE",
                    "Extra heads / native MTP reduce passes-per-token — the axis that bypasses the MLP byte wall.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_CENSUS,
                "N046 census: parent has no native MTP/Medusa heads (expected ABSENT). "
                "Head-training and binary-as-draft acceptance are untested (N049).",
                related_not_the_same="Binary TARGET death (BINARY_HEALING) is not MTP acceptance.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Merged with N046 Gloeckle MTP census. Native heads ABSENT is a census, not a prune. "
                "KEEP as the decode-gravity toolkit entry; binary-as-draft remains a legal purpose (N049)."
            ),
            "merged_from_extra_arxiv": ["2404.19737"],
        },
        # ----- N046 recommended -----
        "higgs": {
            "applicability": weight_ptq_matrix(
                (
                    "UNLIKELY",
                    N044_MLP + " HIGGS's Hadamard arm is the N044 discriminator. MSE-grid remainder untested. "
                    + S028_KEEP,
                ),
            ),
            "qwen_datapoint": qwen(
                "failed",
                R_N044,
                "Hadamard incoherence on Qwen MLP. HIGGS MSE-optimal Gaussian grid was not the N044 codec.",
                related_not_the_same="N044 re-fit binary/ternary/q2f, not an MSE-optimal Gaussian grid.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Hadamard arm closed on Qwen MLP. KEEP: MSE-grid codebook and other-arch absorption "
                "are UNKNOWN (S028). Cheapest remaining is the grid, not another FWHT."
            ),
        },
        "teal": {
            "applicability": weight_ptq_matrix(
                (
                    "PLAUSIBLE",
                    "Training-free magnitude skip of SwiGLU hidden channels; mlp_gate_up is memory-class. "
                    "Histogram of |x| per channel is the cheap falsifier.",
                ),
                decoding=(
                    "UNLIKELY",
                    "Activation skip is not speculative decoding.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_BYTES,
                "CSR-sparse 2% residual was compute-bound (wrong sparsity). TEAL channel-magnitude skip unrun.",
                related_not_the_same="BYTES_FRONTIER unstructured CSR is not TEAL structured skip.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Not the CSR sparse already killed. KEEP. One activation histogram falsifies it. "
                "Fused skip required before any token_ns claim."
            ),
        },
        "layerskip": {
            "applicability": decode_matrix(
                (
                    "PLAUSIBLE",
                    "Self-speculative: early-exit or binary_g64 drafts, remaining/q2f verifies. "
                    "No extra draft model. Highest-EV decode probe after native-MTP census.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_BINARY,
                "Binary is a measured fast injured TARGET, not a measured DRAFT. LayerSkip τ untested (N049).",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Not Medusa (no extra heads). Maps onto injured binary + hawking-speculate. KEEP as N049."
            ),
        },
        "butterflyquant": {
            "applicability": weight_ptq_matrix(
                (
                    "UNLIKELY",
                    N044_MLP + " Butterfly is a richer rotation family than Hadamard; N044's learned "
                    "block-orthogonal is the overlapping discriminator. " + S028_KEEP,
                ),
            ),
            "qwen_datapoint": rot_qwen,
            "decision": KEEP,
            "decision_reason": (
                "N044 showed rotation coordinate-robust on Qwen MLP → dense_mlp UNLIKELY, but "
                "attention/other-arch UNKNOWN. KEEP, do not prune (S028 explicit example)."
            ),
        },
        "qtip": {
            "applicability": weight_ptq_matrix(
                (
                    "UNKNOWN",
                    "Trellis+incoherence is not the C4 VQ-PQ that was refused; strand-quant already has trellis.rs.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_C4,
                "C4 refused gravity_pq VQ, not QTIP trellis. CPU trellis on L31.gate_proj unrun.",
                related_not_the_same="AQLM/VQ dimension ceiling is QTIP's claim, not a Hawking measurement.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Distinct from AQLM. In-tree trellis/RHT makes this the cheapest codebook experiment "
                "C4 did not already refuse. KEEP."
            ),
        },
        "quamba2": {
            "applicability": ssm_matrix(
                (
                    "PLAUSIBLE",
                    "SSM-specific PTQ (cluster/sort recurrence, per-state-group B/C). "
                    "DeltaNet is 20.7% of params at 3.261 complete EBPW — largest non-MLP density lever.",
                ),
                kv_state=(
                    "UNKNOWN",
                    "State-group quant of B/C is the new part; leftover currently billed at 32 bpw.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_FRONTIERS,
                "DeltaNet in_proj floor is grouped_absmax q3; Quamba2 clustered quant unrun.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "No SSM/DeltaNet technique is in the N043 seed. KEEP. Not a Qwen-MLP prune."
            ),
        },
        "dflash": {
            "applicability": decode_matrix(
                (
                    "PLAUSIBLE",
                    "Block-diffusion drafter with a Qwen3.8-27B public artifact; verify+rollback is in-tree.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_QCR,
                "Public Qwen3.8-27B DFlash weights were not loaded here (GPU lock is not this lane).",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Specimen-specific decode hypothesis, not a Qwen-MLP codec. KEEP. Do not take the GPU."
            ),
        },
        "octopus": {
            "applicability": kv_matrix(
                (
                    "PLAUSIBLE",
                    "Octahedral triplet KV after rotation; GQA head_dim=256 fits triplets. "
                    "Best published 2-bit rotation KV as of N046.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_PREFILL,
                "No octahedral triplet MSE on a captured Qwen GQA K head.",
            ),
            "decision": KEEP,
            "decision_reason": "GQA-KV addition, not a Qwen-MLP prune. KEEP.",
        },
        "oscar": {
            "applicability": kv_matrix(
                (
                    "PLAUSIBLE",
                    "Attention-aligned covariance rotation for 2-bit KV; Hadamard was weak as a "
                    "weight bits lever (N044/C5) but KV may need a different rotation.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_N044,
                "N044 rotated MLP weights, not GQA K covariance. OSCAR's KV rotation unrun.",
                related_not_the_same="Weight Hadamard ≠ attention-aware KV rotation.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "2026 attention-aware KV rotation is not KIVI and not SpinQuant. KEEP. Cheap covariance probe."
            ),
        },
        "turboquant": {
            "applicability": kv_matrix(
                (
                    "PLAUSIBLE",
                    "Rotation + PolarQuant + 1-bit QJL residual; data-free ~3-bit KV. "
                    "GQA KV is the compressible cache; DN state is not.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_PREFILL,
                "No Hadamard-rotate + Polar/Lloyd-Max on a captured GQA K head.",
            ),
            "decision": KEEP,
            "decision_reason": "ICLR 2026 KV rotation codec is not KIVI. KEEP. PolarQuant is an arm, not a duplicate prune.",
        },
        "paroquant": {
            "applicability": weight_ptq_matrix(
                (
                    "UNLIKELY",
                    N044_MLP + " Pairwise Givens is a cheaper structured rotation, still a rotation. "
                    + S028_KEEP,
                ),
            ),
            "qwen_datapoint": rot_qwen,
            "decision": KEEP,
            "decision_reason": (
                "ICLR 2026 pairwise rotation is not SpinQuant. N044 closed Qwen-MLP rotation; "
                "KEEP for other organs/arch (S028)."
            ),
        },
        "resq": {
            "applicability": weight_ptq_matrix(
                (
                    "UNKNOWN",
                    "Activation-subspace mixed precision, not the weight-LR residual HYBRID_OPERATOR killed.",
                ),
                decoding=(
                    "UNLIKELY",
                    "W4A4 helps compute-bound prefills more than load-bound decode token_ns.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_HYBRID,
                "HYBRID killed WEIGHT low-rank residuals. ResQ PCA of real MLP activations unrun.",
                related_not_the_same="CALDERA/HYBRID is weight LR, not activation subspace.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Not CALDERA. Register so Doctor does not conflate activation-subspace mixed "
                "precision with the killed weight LR residual. KEEP (S028)."
            ),
        },
        "sparse_delta_memory": {
            "applicability": ssm_matrix(
                (
                    "PLAUSIBLE",
                    "Sparse reads/writes into explicit GDN memory; targets DN session state, "
                    "not the 0.015 in_proj-irreducible law's weight side.",
                ),
                kv_state=(
                    "PLAUSIBLE",
                    "Directly targets DN state bytes (156.9 MB at seq 256), the larger session term.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_FRONTIERS,
                "GDN state effective rank/sparsity on a real prefix unmeasured. Do not build a sparse kernel first.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "2026 GDN-state compression is not MiniCache/H2O. KEEP. Sparse addressing is a "
                "Metal hazard (N033) but that is a kernel gate, not a prune of the mechanism."
            ),
        },
        "hyperquant": {
            "applicability": weight_ptq_matrix(
                (
                    "UNKNOWN",
                    "Unified RHT+lattice+Rice for weights and KV; HIGGS/TurboQuant probes first.",
                ),
                kv_state=(
                    "PLAUSIBLE",
                    "Same pipeline claims KV as well as weights; Rice on attention is a Gravity cousin.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_NNS,
                "NNS-003: Gravity Rice families cannot carry Qwen3.8 ≤1.5. HyperQuant as a unified pipeline unrun.",
                related_not_the_same="Gravity Rice is not HyperQuant's lattice+bit-strip pipeline.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "2026 unified RHT+lattice+Rice is not AQLM/HIGGS/KIVI. KEEP as a composition of "
                "those probes, not a first GPU build. Related Gravity Rice is not a prune (S028)."
            ),
        },
        "eagle3": {
            "applicability": decode_matrix(
                (
                    "PLAUSIBLE",
                    "2025 production draft (multi-layer fused hidden states). Train-cost is real; "
                    "run after LayerSkip/binary-draft.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_CENSUS,
                "No EAGLE-3 draft net on this parent. Do not train until LayerSkip τ is measured.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "2025 successor of Medusa with a different draft. KEEP; do not train 27B first."
            ),
        },
        "magicdec": {
            "applicability": decode_matrix(
                (
                    "PLAUSIBLE",
                    "Scheduling policy: speculative decoding at large batch/long context. "
                    "No new kernel. Orthogonal to LayerSkip.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_PREFILL,
                "Accept τ at seq 4K vs 256 unmeasured. DN state is not prefix-shareable, which may break the premise.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "ICLR 2025 long-context speculative policy is not Medusa. KEEP as production decode-gravity."
            ),
        },
        "ostquant": {
            "applicability": weight_ptq_matrix(
                (
                    "UNLIKELY",
                    N044_MLP + " OST is SpinQuant + diagonal scale; N044 covers the rotation. "
                    + S028_KEEP,
                ),
            ),
            "qwen_datapoint": rot_qwen,
            "decision": KEEP,
            "decision_reason": (
                "ICLR 2025 orthogonal+scale is not in the N043 name list. Qwen-MLP rotation closed; "
                "KEEP as SpinQuant-adjacent for other organs (S028). Not a separate GPU fit."
            ),
        },
        "palu": {
            "applicability": kv_matrix(
                (
                    "PLAUSIBLE",
                    "Low-rank factorization of K/V projections; cache the latent. Distinct from "
                    "weight-LR (C1/C3 killed) and from H2O eviction.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_FRONTIERS,
                "ORGAN_FRONTIERS aa_rank_256 survived locally at 1.47 bpw but was NOT q4-equivalent. "
                "Palu is KV-side SVD, unrun.",
                related_not_the_same="Weight low-rank ≠ KV hidden-dim factorization.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "ICLR 2025 KV low-rank is not MiniCache/H2O/KIVI and not the weight-LR family C1/C3 killed. KEEP."
            ),
        },
        "polarquant": {
            "applicability": kv_matrix(
                (
                    "PLAUSIBLE",
                    "Polar-angle Lloyd-Max codebook (TurboQuant stage 1). Keep as a distinct "
                    "falsifying arm, not a second campaign.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_PREFILL,
                "Polar vs scalar-after-Hadamard on a GQA K head unrun; share TurboQuant's CPU MSE.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "AISTATS 2026 polar KV is not KIVI. Not superseded: it is the cheap arm of TurboQuant, "
                "not a Metal-infeasible or strictly-dominated dead end. KEEP (S028: do not prune a subset arm)."
            ),
        },
        "superbpe": {
            "applicability": tokenizer_matrix(
                (
                    "PLAUSIBLE",
                    "Two-stage BPE (subwords then superwords) is a VocabularyGenome option, "
                    "distinct from N045 ASCII-prune.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_TOK,
                "N045 measured ASCII-prune inflation, not SuperBPE. Do not retrain 27B.",
                related_not_the_same="ASCII-prune CONTROL is not a from-scratch SuperBPE tokenizer.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Tokenizer family is N045, not N043. KEEP as a vocab-genome hypothesis. "
                "N045 prune is the cheap experiment; SuperBPE is not that control."
            ),
        },
        "flatquant": {
            "applicability": weight_ptq_matrix(
                (
                    "UNLIKELY",
                    "C5 generated Kronecker REFUTED (energy 0.015); FlatQuant's learned affine is "
                    "different but W4A4 is below the current 2-bit question. N044-class. "
                    + S028_KEEP,
                ),
            ),
            "qwen_datapoint": qwen(
                "failed",
                R_C5,
                "Generated Kronecker as a bits lever died (C5). FlatQuant learned affine on Qwen MLP unrun; "
                "N044 rotations also did not move 2.25.",
                related_not_the_same="C5 generated-W Kronecker is not FlatQuant's learned activation affine.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Not in N043; distinct from SpinQuant (affine/Kronecker, not orthogonal). "
                "Qwen-MLP W4A4 is not the bottleneck — KEEP as a N044 footnote, not a prune (S028)."
            ),
        },
        "mixture_of_recursions": {
            "applicability": as_matrix(
                {
                    "dense_mlp": (
                        "UNKNOWN",
                        "Depth-tying of layer weights is a different axis than C1 coefficient-sharing.",
                    ),
                    "moe": (
                        "UNKNOWN",
                        "Recursion + routing is an Odyssey architecture, not a Qwen GPU WU.",
                    ),
                    "attention_gqa": (
                        "UNKNOWN",
                        "Optional KV sharing from first recursion is untested on this hybrid.",
                    ),
                    "recurrent_deltanet": (
                        "UNKNOWN",
                        "Recursion over GDN layers is a different parent.",
                    ),
                    "multimodal": (
                        "UNKNOWN",
                        "No multimodal organ in this census.",
                    ),
                    "kv_state": (
                        "UNKNOWN",
                        "First-recursion KV sharing could cut session state; untested.",
                    ),
                    "decoding": (
                        "UNLIKELY",
                        "Adaptive depth is not speculative decoding (LayerSkip is the cheap cousin).",
                    ),
                    "tokenizer": (
                        "UNLIKELY",
                        "Does not change vocabulary topology.",
                    ),
                }
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_C1,
                "C1 shared-basis died on coefficient fidelity, not on depth-tying. Layer-ℓ vs ℓ+1 weight cosine is the cheap check.",
                related_not_the_same="C1 codec sharing ≠ MoR depth tying.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "2025 MoD successor. KEEP as Odyssey/architecture. C1's Qwen sharing negative is "
                "a different mechanism — not a prune (S028)."
            ),
        },
        "transmla": {
            "applicability": kv_matrix(
                (
                    "PLAUSIBLE",
                    "GQA→MLA conversion shrinks the KV that PREFILL_KV showed can exceed MODEL_BYTES. "
                    "6B-token recovery is Odyssey-cost; SVD of concatenated K/V is cheap.",
                ),
                dense_mlp=(
                    "UNLIKELY",
                    "Attention latent conversion; does not compress MLP.",
                ),
            ),
            "qwen_datapoint": qwen(
                "untested",
                R_PREFILL,
                "SVD of concatenated GQA K/V latents on a 4K prefix unrun. Do not fine-tune 27B.",
            ),
            "decision": KEEP,
            "decision_reason": "GQA→MLA is not KIVI. KEEP as state-architecture; cheap SVD first.",
        },
        # ----- campaign mechanism (S028 explicit KEEP example) -----
        "shared_basis": {
            "applicability": as_matrix(
                {
                    "dense_mlp": (
                        "UNLIKELY",
                        "Competent fused kernel but dead <2.25 on Qwen MLP (K=2 died at held-out activation). "
                        f"{S028_KEEP}",
                    ),
                    "moe": (
                        "UNKNOWN",
                        "Cross-expert / MoE shared substrate is untested; NNS-004 Q80 pairwise cosine is a different parent. "
                        "S028 explicit: KEEP for MoE-expert / cross-layer sharing UNKNOWN.",
                    ),
                    "attention_gqa": (
                        "UNKNOWN",
                        "Sharing Q/O bases across GQA layers untested; C1 was MLP coefficients.",
                    ),
                    "recurrent_deltanet": (
                        "UNKNOWN",
                        "Sharing in_proj bases across DN layers untested.",
                    ),
                    "multimodal": (
                        "UNKNOWN",
                        "No multimodal organ in this census.",
                    ),
                    "kv_state": (
                        "UNLIKELY",
                        "Shared basis is a weight-coefficient codec, not a KV codec.",
                    ),
                    "decoding": (
                        "UNLIKELY",
                        "Does not change accepted-tokens-per-forward.",
                    ),
                    "tokenizer": (
                        "UNLIKELY",
                        "Does not change vocabulary topology.",
                    ),
                }
            ),
            "qwen_datapoint": qwen(
                "failed",
                R_SHARED_C,
                "K=2 coherent_shared_basis_beats_q2f=false. Kernel competent "
                f"({R_SHARED_K}): byte win translated to token_ns.",
            ),
            "decision": KEEP,
            "decision_reason": (
                "Competent kernel, dead <2.25 on Qwen MLP. KEEP for MoE-expert / cross-layer "
                "sharing UNKNOWN (S028 explicit example). Do not prune a purpose-scoped negative."
            ),
        },
    }


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def _load_n043() -> dict[str, Any]:
    return load_json(N043_RECEIPT)


def _load_n046() -> dict[str, Any]:
    return load_json(N046_RECEIPT)


def _n046_by_name(n046: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {t["name"]: t for t in n046.get("techniques") or []}


def _license(n046_row: dict[str, Any] | None, n043_entry: dict[str, Any] | None) -> str:
    if n046_row and n046_row.get("license"):
        return str(n046_row["license"])
    if n043_entry:
        note = (n043_entry.get("licensing_provenance") or {}).get("paper_license_note") or ""
        if note.strip():
            return note.strip()
    return DEFAULT_LICENSE


def _code_license(n046_row: dict[str, Any] | None, n043_entry: dict[str, Any] | None) -> str | None:
    if n043_entry:
        note = (n043_entry.get("licensing_provenance") or {}).get("code_license_note")
        if note:
            return note
    if n046_row and n046_row.get("code_url"):
        return f"code at {n046_row['code_url']} — check license; do not copy (S026 §6, §88)."
    return None


def _arxiv_date(n046_row: dict[str, Any] | None, n043_entry: dict[str, Any] | None) -> str:
    if n046_row and n046_row.get("arxiv_date"):
        return str(n046_row["arxiv_date"])
    if n043_entry:
        d = (n043_entry.get("source_paper") or {}).get("approx_date") or ""
        # Normalize "2024-05" and "2024-01 / 2024-04"
        return str(d)
    return ""


def _mechanism(n043_entry: dict[str, Any] | None, n046_row: dict[str, Any] | None) -> str:
    parts = []
    if n043_entry and n043_entry.get("claimed_mechanism"):
        parts.append(str(n043_entry["claimed_mechanism"]).strip())
    if n046_row and n046_row.get("mechanism"):
        m = str(n046_row["mechanism"]).strip()
        if m not in parts:
            parts.append(m)
    if not parts:
        raise ValueError("no mechanism")
    if len(parts) == 1:
        return parts[0]
    return parts[0] + " N046 restates: " + parts[1]


def _metal(n046_row: dict[str, Any] | None, fallback_class: str, fallback_note: str) -> dict[str, Any]:
    cls = (n046_row or {}).get("metal_feasibility") or fallback_class
    note = (n046_row or {}).get("metal_note") or fallback_note
    return {
        "class": cls,
        "note": note,
        "cuda_result_is_not_metal_result": True,
        "s026_89": "A paper CUDA result is not a Metal result.",
    }


def _base_entry(
    *,
    tid: str,
    name: str,
    sources: list[str],
    family: str,
    n043_entry: dict[str, Any] | None,
    n046_row: dict[str, Any] | None,
    merged_from: list[dict[str, Any]],
    hardening: dict[str, Any],
) -> dict[str, Any]:
    paper = (n043_entry or {}).get("source_paper") or {}
    arxiv = None
    extra_arxiv = list(hardening.get("merged_from_extra_arxiv") or [])
    if n046_row and n046_row.get("arxiv_id"):
        arxiv = n046_row["arxiv_id"]
    elif paper.get("arxiv"):
        arxiv = str(paper["arxiv"]).split("/")[0].strip()
        # medusa_mtp paper.arxiv is 2401.10774; MTP 2404.19737 lives in venue/note.
    if tid == "medusa_mtp":
        arxiv = arxiv or "2401.10774"
        if "2404.19737" not in extra_arxiv:
            extra_arxiv.append("2404.19737")
    if tid == "shared_basis":
        provenance = {
            "kind": "campaign_mechanism",
            "arxiv_id": None,
            "arxiv_date": "2026-08",
            "license": "campaign-internal (not a paper; Hawking receipts are the provenance)",
            "authors": "this campaign (C1 shared basis)",
            "url": None,
            "code_url": None,
            "code_license_note": None,
            "do_not_copy_third_party_code": True,
            "receipts": [R_SHARED_K, R_SHARED_C, R_C1],
        }
        family = family or "shared_basis / cross-layer coefficients"
        mechanism = (
            "Share a small set of binary/low-bit basis vectors across MLP layers "
            "and store per-layer coefficients. Fused K=2 kernel is competent; "
            "K=2 composition dies below q2f 2.25 on held-out activations."
        )
        metal = _metal(
            None,
            "NATIVE_KERNEL_EXISTS",
            "shared_binary_k2 fused kernel is competent (byte win translated to token_ns). "
            "Competence did not save K=2 composition.",
        )
        cheapest = experiment(
            "HX-SHARED-BASIS-MOE-UNKNOWN",
            "Do not rebuild K=2 on Qwen MLP. Cheap next: pairwise cosine of expert "
            "(or cross-layer) bases on a routed parent — MoE sharing is UNKNOWN (S028).",
            success="If expert bases are ~orthogonal, MoE sharing is as dead as Qwen MLP K=2; if aligned, reopen that purpose.",
        )
    else:
        if not arxiv or not ARXIV_RE.match(arxiv.split()[0] if arxiv else ""):
            # Allow "2401.10774" only.
            aid = (arxiv or "").split()[0]
            if not ARXIV_RE.match(aid):
                raise ValueError(f"{tid}: bad arxiv_id {arxiv!r}")
            arxiv = aid
        provenance = {
            "kind": "paper",
            "arxiv_id": arxiv,
            "arxiv_ids": [arxiv] + [a for a in extra_arxiv if a != arxiv],
            "arxiv_date": _arxiv_date(n046_row, n043_entry),
            "license": _license(n046_row, n043_entry),
            "authors": (n046_row or {}).get("authors")
            or paper.get("authors")
            or "",
            "title": paper.get("title") or (n046_row or {}).get("name") or name,
            "venue": (n046_row or {}).get("venue") or paper.get("venue"),
            "url": f"https://arxiv.org/abs/{arxiv}",
            "code_url": (n046_row or {}).get("code_url")
            or (n043_entry or {}).get("licensing_provenance", {}).get("code_url"),
            "code_license_note": _code_license(n046_row, n043_entry),
            "do_not_copy_third_party_code": True,
            "s026_88": "provenance preserved",
            "s026_6": "no blind implementation",
        }
        if n043_entry and n043_entry.get("claimed_mechanism"):
            mechanism = _mechanism(n043_entry, n046_row)
        elif n046_row:
            mechanism = n046_row["mechanism"]
        else:
            raise ValueError(f"{tid}: no mechanism")
        if n046_row:
            metal = _metal(n046_row, "UNPROVEN_ON_METAL", n046_row.get("metal_note") or "")
            cheapest = wrap_n046_experiment(tid, n046_row["cheapest_falsifying_experiment"])
        elif n043_entry:
            metal = _metal(
                None,
                "UNPROVEN_ON_METAL",
                "See N043 hawking_experiment_mapping; CUDA ≠ Metal (S026 §89).",
            )
            cheapest = wrap_n043_experiment(
                (n043_entry.get("current_verdict") or {}).get("cheapest_hawking_experiment"),
                f"HX-{tid.upper()}",
            )
        else:
            raise ValueError(f"{tid}: no n043/n046 source")
        # Overlay cheapest if N043 has a structured experiment and we also have n046.
        if n043_entry and n046_row:
            n043_exp = wrap_n043_experiment(
                (n043_entry.get("current_verdict") or {}).get("cheapest_hawking_experiment"),
                f"HX-{tid.upper()}",
            )
            cheapest = {
                **cheapest,
                "n043_experiment_id": n043_exp.get("id"),
                "n043_summary": n043_exp.get("summary"),
                "union_of_evidence": True,
            }

    ident_family = family
    if n043_entry:
        ident_family = (n043_entry.get("technique_identity") or {}).get("s026_family") or family
    elif n046_row:
        ident_family = n046_row.get("family") or family

    entry = {
        "id": tid,
        "name": name,
        "sources": sources,
        "merged_from": merged_from,
        "family": ident_family,
        "n043_seed": tid in REQUIRED_TECHNIQUE_IDS or name in N043_SEED,
        "literature_status": LITERATURE_STATUS,
        "not_authority": True,
        "mechanism": mechanism,
        "provenance": provenance,
        "applicability": hardening["applicability"],
        "qwen_datapoint": hardening["qwen_datapoint"],
        "metal_feasibility": metal,
        "cheapest_falsifying_experiment": cheapest,
        "decision": hardening["decision"],
        "decision_reason": hardening["decision_reason"],
        "s028": (
            "Failing on Qwen's MLP != dead. A candidate rejected for a purpose "
            "is not globally dead (S026 §64, §107)."
        ),
    }
    if n043_entry:
        entry["n043_verdict_status"] = (n043_entry.get("current_verdict") or {}).get("status")
        entry["n043_campaign_verdict"] = (n043_entry.get("current_verdict") or {}).get(
            "campaign_verdict"
        )
        recs = (n043_entry.get("current_verdict") or {}).get("hawking_receipts") or []
        entry["n043_hawking_receipts"] = recs
    if n046_row:
        entry["n046_rank_score"] = n046_row.get("rank_score")
        entry["n046_metal_feasibility"] = n046_row.get("metal_feasibility")
        entry["n046_hawking_organ"] = n046_row.get("hawking_organ")
        entry["codebase_citations"] = n046_row.get("codebase_citations") or []
    return entry


def _n046_row_for_n043(short_name: str, by_name: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if short_name in by_name:
        return by_name[short_name]
    mapped = N043_SHORT_TO_N046.get(short_name)
    if mapped and mapped in by_name:
        return by_name[mapped]
    # Medusa/MTP
    head = short_name.split("/")[0]
    if head in by_name:
        return by_name[head]
    return None


def build_techniques(n043: dict[str, Any], n046: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hard = _hardening()
    by_n046 = _n046_by_name(n046)
    recommended = list(n046.get("RECOMMENDED_ADDITIONS") or [])
    rec_names = [r["name"] for r in recommended]
    if len(rec_names) != 24:
        raise SystemExit(f"N046 RECOMMENDED_ADDITIONS has {len(rec_names)} rows, expected 24")

    techniques: list[dict[str, Any]] = []
    dedup: list[dict[str, Any]] = []

    n043_entries = list(n043.get("techniques") or [])
    n043_ids = [(t.get("technique_identity") or {}).get("id") for t in n043_entries]
    missing = [i for i in REQUIRED_TECHNIQUE_IDS if i not in n043_ids]
    if missing:
        raise SystemExit(f"N043 missing {missing}")

    for t in n043_entries:
        ident = t["technique_identity"]
        tid = ident["id"]
        if tid not in hard:
            raise SystemExit(f"N052 hardening missing N043 id {tid}")
        n046_row = _n046_row_for_n043(ident["short_name"], by_n046)
        merged_from: list[dict[str, Any]] = []
        sources = ["N043"]
        if tid == "medusa_mtp":
            gloeckle = by_n046.get("Gloeckle MTP / Qwen-native MTP census")
            if gloeckle:
                sources.append("N046")
                merged_from.append(
                    {
                        "n046_name": gloeckle["name"],
                        "id_would_have_been": "gloeckle_mtp",
                        "arxiv_id": gloeckle["arxiv_id"],
                        "reason": (
                            "N043 medusa_mtp already registers Medusa (2401.10774) and "
                            "Gloeckle MTP (2404.19737) as one S026 mechanism. N046's "
                            "recommended MTP census is the same paper; keep the union of evidence."
                        ),
                    }
                )
                dedup.append(
                    {
                        "from_n046_name": gloeckle["name"],
                        "from_id": "gloeckle_mtp",
                        "into": "medusa_mtp",
                        "arxiv_id": gloeckle["arxiv_id"],
                        "reason": merged_from[0]["reason"],
                    }
                )
                # Union: prefer N046 cheapest (census) which N043 already has as HX-MTP-...
        techniques.append(
            _base_entry(
                tid=tid,
                name=ident["short_name"],
                sources=sources,
                family=ident.get("s026_family") or "",
                n043_entry=t,
                n046_row=n046_row,
                merged_from=merged_from,
                hardening=hard[tid],
            )
        )

    seen_ids = {t["id"] for t in techniques}
    for rec in recommended:
        name = rec["name"]
        if name in MERGE_N046_INTO:
            continue  # already merged
        slug = N046_SLUG.get(name)
        if not slug:
            raise SystemExit(f"unslugged N046 recommended name {name!r}")
        if slug in seen_ids:
            raise SystemExit(f"duplicate id {slug} for {name}")
        if slug not in hard:
            raise SystemExit(f"N052 hardening missing N046 id {slug}")
        row = by_n046.get(name)
        if not row:
            raise SystemExit(f"N046 catalog missing recommended {name}")
        techniques.append(
            _base_entry(
                tid=slug,
                name=name,
                sources=["N046"],
                family=row.get("family") or rec.get("family") or "",
                n043_entry=None,
                n046_row=row,
                merged_from=[],
                hardening=hard[slug],
            )
        )
        seen_ids.add(slug)

    if "shared_basis" not in hard:
        raise SystemExit("shared_basis hardening missing")
    techniques.append(
        _base_entry(
            tid="shared_basis",
            name="shared_basis",
            sources=["campaign"],
            family="shared_basis / cross-layer coefficients",
            n043_entry=None,
            n046_row=None,
            merged_from=[],
            hardening=hard["shared_basis"],
        )
    )

    # Stable order: N043 seed ids, then N046 recommended (minus merged), then campaign.
    order = list(REQUIRED_TECHNIQUE_IDS)
    for rec in recommended:
        if rec["name"] in MERGE_N046_INTO:
            continue
        order.append(N046_SLUG[rec["name"]])
    order.append("shared_basis")
    by_id = {t["id"]: t for t in techniques}
    missing_ids = [i for i in order if i not in by_id]
    extra_ids = [i for i in by_id if i not in order]
    if missing_ids or extra_ids:
        raise SystemExit(f"order mismatch missing={missing_ids} extra={extra_ids}")
    return [by_id[i] for i in order], dedup


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _qwen_prune_reason(reason: str) -> bool:
    blob = reason.lower()
    return any(p in blob for p in QWEN_PRUNE_PHRASES)


def technique_errors(entry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    tid = entry.get("id", "<unknown>")
    for key in (
        "id",
        "name",
        "sources",
        "mechanism",
        "provenance",
        "applicability",
        "qwen_datapoint",
        "metal_feasibility",
        "cheapest_falsifying_experiment",
        "decision",
        "decision_reason",
    ):
        if key not in entry:
            errors.append(f"{tid}: missing {key}")
    if entry.get("literature_status") != LITERATURE_STATUS:
        errors.append(f"{tid}: literature_status must be {LITERATURE_STATUS}")
    if entry.get("not_authority") is not True:
        errors.append(f"{tid}: not_authority must be true")
    prov = entry.get("provenance") or {}
    if prov.get("kind") == "paper":
        aid = prov.get("arxiv_id")
        if not aid or not ARXIV_RE.match(str(aid)):
            errors.append(f"{tid}: provenance.arxiv_id required for papers")
        if not prov.get("arxiv_date"):
            errors.append(f"{tid}: provenance.arxiv_date required")
        if not prov.get("license"):
            errors.append(f"{tid}: provenance.license required")
        if prov.get("do_not_copy_third_party_code") is not True:
            errors.append(f"{tid}: do_not_copy_third_party_code must be true")
    elif prov.get("kind") == "campaign_mechanism":
        recs = prov.get("receipts") or []
        if not recs:
            errors.append(f"{tid}: campaign mechanism needs receipt provenance")
        for p in recs:
            if not is_hawking_receipt_path(p) or not citation_exists(p):
                errors.append(f"{tid}: campaign receipt missing {p}")
    else:
        errors.append(f"{tid}: provenance.kind must be paper or campaign_mechanism")

    app = entry.get("applicability") or {}
    for cls in ARCH_CLASSES:
        cell_d = app.get(cls)
        if not isinstance(cell_d, dict):
            errors.append(f"{tid}: applicability.{cls} missing")
            continue
        if cell_d.get("grade") not in GRADES:
            errors.append(f"{tid}: applicability.{cls}.grade illegal {cell_d.get('grade')!r}")
        reason = (cell_d.get("reason") or "").strip()
        if not reason or "\n" in reason:
            errors.append(f"{tid}: applicability.{cls}.reason must be a one-line reason")

    qd = entry.get("qwen_datapoint") or {}
    if qd.get("outcome") not in QWEN_OUTCOMES:
        errors.append(f"{tid}: qwen_datapoint.outcome illegal {qd.get('outcome')!r}")
    rec = qd.get("receipt")
    if qd.get("outcome") != "untested":
        if not rec or not is_hawking_receipt_path(rec):
            errors.append(f"{tid}: qwen_datapoint {qd.get('outcome')} needs a Hawking receipt")
        elif not citation_exists(rec):
            errors.append(f"{tid}: qwen_datapoint receipt does not exist: {rec}")
    elif rec:
        if not is_hawking_receipt_path(rec) or not citation_exists(rec):
            errors.append(f"{tid}: untested related receipt missing {rec}")

    metal = entry.get("metal_feasibility") or {}
    if not metal.get("class"):
        errors.append(f"{tid}: metal_feasibility.class required")
    if metal.get("cuda_result_is_not_metal_result") is not True:
        errors.append(f"{tid}: cuda_result_is_not_metal_result must be true")
    if not metal.get("note"):
        errors.append(f"{tid}: metal_feasibility.note required")

    exp = entry.get("cheapest_falsifying_experiment") or {}
    if not exp.get("id") or not exp.get("summary"):
        errors.append(f"{tid}: cheapest_falsifying_experiment needs id+summary")
    if exp.get("cpu_only") is not True or exp.get("touches_gpu") is not False:
        errors.append(f"{tid}: cheapest experiment must be CPU, no GPU")
    if exp.get("no_second_27b") is not True:
        errors.append(f"{tid}: cheapest experiment must not load a second 27B")

    decision = entry.get("decision")
    if decision not in DECISIONS:
        errors.append(f"{tid}: decision must be KEEP or PRUNE")
    reason = entry.get("decision_reason") or ""
    if not reason.strip():
        errors.append(f"{tid}: decision_reason required")
    if decision == PRUNE:
        if _qwen_prune_reason(reason):
            errors.append(
                f"{tid}: PRUNED for a reason that is merely a Qwen failure "
                f"(S028 forbids this): {reason!r}"
            )
        pcls = entry.get("prune_class")
        if pcls not in PRUNE_CLASSES:
            errors.append(
                f"{tid}: PRUNE requires prune_class in {sorted(PRUNE_CLASSES)} "
                "(superseded / strictly_dominated / metal_infeasible), not a Qwen result"
            )
        # Qwen-failed KEEP is required to retain a PLAUSIBLE or UNKNOWN cell.
    if decision == KEEP and qd.get("outcome") == "failed":
        grades = {c.get("grade") for c in app.values()}
        if grades <= {"UNLIKELY"}:
            errors.append(
                f"{tid}: KEEP after Qwen failure but every architecture class is UNLIKELY "
                "— that is a stealth prune (S028). Leave at least one UNKNOWN/PLAUSIBLE cell."
            )
    return errors


def library_errors(doc: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    techs = doc.get("techniques") or []
    ids = [t.get("id") for t in techs]
    dup = sorted({i for i in ids if i and ids.count(i) > 1})
    if dup:
        errors.append(f"duplicate technique ids: {dup}")
    missing_n043 = [i for i in REQUIRED_TECHNIQUE_IDS if i not in ids]
    if missing_n043:
        errors.append(f"missing N043 techniques: {missing_n043}")
    if "shared_basis" not in ids:
        errors.append("missing campaign technique shared_basis (S028 KEEP example)")
    rec_names = doc.get("n046_recommended_names") or []
    if len(rec_names) != 24:
        errors.append(f"n046_recommended_names has {len(rec_names)}, expected 24")
    covered = set()
    for t in techs:
        covered.add(t.get("name"))
        for m in t.get("merged_from") or []:
            covered.add(m.get("n046_name"))
    missing_rec = [n for n in rec_names if n not in covered and n.split("/")[0] not in covered]
    # Medusa/MTP covers "Gloeckle MTP / Qwen-native MTP census" via merged_from.
    if missing_rec:
        errors.append(f"N046 recommended names not in library: {missing_rec}")
    for t in techs:
        errors.extend(technique_errors(t))
    pruned = doc.get("pruned") or []
    for row in pruned:
        if not row.get("id") or not row.get("reason"):
            errors.append(f"pruned row missing id/reason: {row}")
        if _qwen_prune_reason(row.get("reason") or ""):
            errors.append(
                f"pruned {row.get('id')}: reason is merely a Qwen failure (S028 forbids this)"
            )
        if row.get("prune_class") not in PRUNE_CLASSES:
            errors.append(f"pruned {row.get('id')}: prune_class must be a viability class")
    keep_ids = {t["id"] for t in techs if t.get("decision") == KEEP}
    prune_ids = {t["id"] for t in techs if t.get("decision") == PRUNE}
    listed = {r.get("id") for r in pruned}
    if prune_ids != listed:
        errors.append(f"pruned list {sorted(listed)} != techniques with PRUNE {sorted(prune_ids)}")
    # S028 examples must be KEEP.
    for must in ("butterflyquant", "onebit", "shared_basis"):
        if must not in keep_ids:
            errors.append(f"{must} must be KEEP (S028 example)")
    bf = next((t for t in techs if t.get("id") == "butterflyquant"), None)
    if bf:
        if bf["applicability"]["dense_mlp"]["grade"] != "UNLIKELY":
            errors.append("butterflyquant dense_mlp must be UNLIKELY (N044)")
        if bf["applicability"]["attention_gqa"]["grade"] not in {"UNKNOWN", "PLAUSIBLE"}:
            errors.append("butterflyquant attention_gqa must not be closed by the Qwen MLP result")
    ob = next((t for t in techs if t.get("id") == "onebit"), None)
    if ob and ob["applicability"]["decoding"]["grade"] != "PLAUSIBLE":
        errors.append("onebit decoding must be PLAUSIBLE (N049 draft)")
    sb = next((t for t in techs if t.get("id") == "shared_basis"), None)
    if sb and sb["applicability"]["moe"]["grade"] != "UNKNOWN":
        errors.append("shared_basis moe must be UNKNOWN (S028)")
    return errors


def build() -> dict[str, Any]:
    n043 = _load_n043()
    n046 = _load_n046()
    techniques, dedup = build_techniques(n043, n046)
    pruned = [
        {
            "id": t["id"],
            "name": t["name"],
            "prune_class": t.get("prune_class"),
            "reason": t["decision_reason"],
        }
        for t in techniques
        if t["decision"] == PRUNE
    ]
    rec_names = [r["name"] for r in (n046.get("RECOMMENDED_ADDITIONS") or [])]
    n_keep = sum(1 for t in techniques if t["decision"] == KEEP)
    n_prune = sum(1 for t in techniques if t["decision"] == PRUNE)
    xref = n043.get("campaign_cross_references") or {}
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "generated_by": GENERATOR,
        "obligation": OBLIGATION,
        "docs": "docs/ultragoals/DOCTOR_TECHNIQUE_LIBRARY.md",
        "hand_authored": False,
        "literature_is": LITERATURE_STATUS,
        "literature_is_not_authority": True,
        "doctor_is": "GENERAL_PHYSICIAN",
        "not_qwen_only": True,
        "s026": ["§5", "§64", "§75", "§107", "§6", "§88", "§89"],
        "s028": (
            "Failing on Qwen's MLP != dead. e.g. ButterflyQuant: N044 rotation "
            "coordinate-robust on Qwen MLP → dense_mlp UNLIKELY, attention/other-arch "
            "UNKNOWN (keep). Binary: dead as Qwen final generator but KEEP as "
            "speculative-draft (N049). Shared basis: competent kernel, dead <2.25 on "
            "Qwen MLP, KEEP for MoE-expert / cross-layer sharing UNKNOWN."
        ),
        "did_not_load_a_model": True,
        "did_not_touch_gpu": True,
        "did_not_run_cargo_or_metal_benchmarks": True,
        "did_not_mutate_parent": True,
        "did_not_write_under_models": True,
        "did_not_modify_ascent_or_campaign": True,
        "architecture_classes": list(ARCH_CLASSES),
        "applicability_grades": sorted(GRADES),
        "qwen_outcomes": sorted(QWEN_OUTCOMES),
        "decisions": sorted(DECISIONS),
        "prune_classes": sorted(PRUNE_CLASSES),
        "n043_receipt": N043_RECEIPT,
        "n046_receipt": N046_RECEIPT,
        "n043_ids": list(REQUIRED_TECHNIQUE_IDS),
        "n043_seed_names": list(N043_SEED),
        "n046_recommended_names": rec_names,
        "n_n043": 15,
        "n_n046_recommended": 24,
        "n_deduplicated": len(dedup),
        "deduplicated": dedup,
        "n_campaign_mechanisms": 1,
        "techniques": techniques,
        "n_techniques": len(techniques),
        "n_keep": n_keep,
        "n_prune": n_prune,
        "pruned": pruned,
        "prune_list_reason": (
            "Empty: none of the merged N043∪N046-recommended techniques is superseded, "
            "strictly-dominated, or Metal-infeasible. Qwen-MLP negatives stay KEEP with "
            "dense_mlp UNLIKELY and other architecture classes UNKNOWN/PLAUSIBLE (S028). "
            "PolarQuant is KEEP as TurboQuant's cheap polar arm, not a prune. CompactifAI / "
            "T-MAC / ShadowKV were already excluded from N046 RECOMMENDED_ADDITIONS."
        ),
        "campaign_scars": {
            "shared_basis": {
                "verdict": xref.get("shared_basis", {}).get("verdict") or SCAR_SHARED_BASIS,
                "decision": KEEP,
                "receipts": [R_SHARED_K, R_SHARED_C],
            },
            "binary": {
                "verdict": xref.get("binary", {}).get("verdict") or SCAR_BINARY,
                "decision": KEEP,
                "as": "speculative-draft candidate (N049)",
                "receipts": [R_BYTES, R_BINARY],
            },
            "low_rank_residual": {
                "verdict": xref.get("low_rank_residual", {}).get("verdict") or SCAR_LOWRANK,
                "decision": KEEP,
                "receipts": [R_HYBRID],
                "note": "Dead on Qwen MLP as a byte lever; not a global prune (S028).",
            },
            "ternary": {
                "verdict": xref.get("ternary", {}).get("verdict") or SCAR_TERNARY,
                "decision": KEEP,
                "receipts": [R_BYTES, R_TERNARY],
            },
            "sparse_residual": {
                "verdict": xref.get("sparse_residual", {}).get("verdict") or SCAR_SPARSE,
                "decision": KEEP,
                "receipts": [R_BYTES],
            },
        },
        "one_line": (
            f"{len(techniques)} techniques in the general Doctor library "
            f"({n_keep} KEEP / {n_prune} PRUNE); N043 15 + N046 24 merged with "
            f"{len(dedup)} N043∩N046 dedup (Medusa/MTP) and shared_basis KEEP; "
            "no Qwen-MLP prune."
        ),
    }
    errors = library_errors(doc)
    if errors:
        raise SystemExit("technique library invalid:\n  " + "\n  ".join(errors))
    return doc


def write_receipt(doc: dict[str, Any] | None = None) -> dict[str, Any]:
    if doc is None:
        doc = build()
    write_json(RECEIPT, doc)
    return doc


def main() -> int:
    doc = write_receipt()
    print(f"schema {doc['schema']}")
    print(f"wrote  {RECEIPT.relative_to(REPO)}")
    print(
        f"n      {doc['n_techniques']} techniques, {doc['n_keep']} KEEP, "
        f"{doc['n_prune']} PRUNE, {doc['n_deduplicated']} dedup"
    )
    for t in doc["techniques"]:
        qd = t["qwen_datapoint"]["outcome"]
        print(f"  {t['id']:<24} {t['decision']:<5} qwen={qd:<8} src={','.join(t['sources'])}")
    if doc["pruned"]:
        print("pruned:")
        for p in doc["pruned"]:
            print(f"  {p['id']}: [{p['prune_class']}] {p['reason']}")
    else:
        print("pruned: (none) — " + doc["prune_list_reason"][:120])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
