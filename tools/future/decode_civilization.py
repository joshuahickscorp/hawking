"""DECODE CIVILIZATION — tokenizer, LM head, sampling, state, speculative/MTP.

Hawking optimization must not collapse to weight matrices. Tokenizer, LM head,
sampling, KV, recurrent state, and speculative decoding are physical organs
with real cost. This sidecar builds their analytical cost models and the
interfaces around them.

THE OBJECTIVE is `accepted_complete_token_cost(plan)` in relative cost units
(1.0 = one baseline target complete token). Raw draft throughput is a
diagnostic that can be won while the real thing gets worse. Rollback is inside
the objective, not an afterthought.

Verification is part of physics. Placement may move WHERE the same predicate
is checked (fused / device-side / digest-of-same-check). Placement may not
change WHAT is checked (sampled / sparse / weaker predicate). Ceremony
reduction is allowed; correctness weakening is not.

Analytical only. No GPU. Costs are structural quantities and relative classes,
never nanoseconds.

    python3 tools/future/decode_civilization.py --build
    python3 tools/future/decode_civilization.py --selftest
    python3 -m pytest tools/future/test_decode_civilization.py -q

Recovers, does not fork: tools/headless/decoding_gravity.py (Leviathan yield,
spec_cycle without rollback), tools/odyssey/decoding_gravity.py (87% accept at
0.91x), crates/hawking-speculate AccelCostLedger / DualKv / ExactTarget
verifier, receipts/headless/FLASH_TOKEN_NS_BUDGET.json organ shapes.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, git

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RECEIPT = "DECODE_CIVILIZATION.json"
SCHEMA = "hawking.future.decode_civilization.v1"
PHYSICAL_GRAPH_SCHEMA = "hcli.physical_graph.v1"
NOMENCLATURE = "HAWKING_NOMENCLATURE_V1"
GENERATOR = "tools/future/decode_civilization.py"

# Exact target check. Durable sinks in hawking-speculate only consume this.
PREDICATE_EXACT = "exact_target_argmax_prefix"

# Cited Qwen3.8-27B geometry (crates/hawking-core/src/model/qwen38_geometry.rs).
QWEN38_LAYERS = 64
QWEN38_DELTANET_LAYERS = 48
QWEN38_GQA_LAYERS = 16
QWEN38_HIDDEN = 5120
QWEN38_VOCAB = 248_320
QWEN38_GQA_KV_HEADS = 4
QWEN38_GQA_HEAD_DIM = 256
QWEN38_LINEAR_KEY_HEADS = 16
QWEN38_LINEAR_VALUE_HEADS = 48
QWEN38_LINEAR_KEY_HEAD_DIM = 128
QWEN38_LINEAR_VALUE_HEAD_DIM = 128

# Tokenizer-gravity organ (receipts/headless/TOKENIZER_GRAVITY.json). Vocab
# row count there is 248077; geometry is 248320. Both are cited, not unified.
TOKENIZER_GRAVITY_VOCAB_ROWS = 248_077
TOKENIZER_GRAVITY_EMBED_BYTES = 496_640_242
TOKENIZER_GRAVITY_LM_HEAD_BYTES = 496_640_242
TOKENIZER_GRAVITY_COMBINED_BYTES = 993_280_484
TOKENIZER_GRAVITY_PAYLOAD_SHARE = 0.1138
TOKENIZER_BYTE_ALPHABET = 256

# Parent MTP (tools/headless/decoding_gravity.py census).
QWEN38_MTP_TENSORS = 15
QWEN38_MTP_BF16_BYTES = 849_398_784
QWEN38_MTP_KV_BYTES_PER_TOKEN_FP16 = 4 * 256 * 2 * 2  # one extra GQA layer

# G057 published ratios. Dimensionless. Not a measurement this lane took.
G057_ACCEPTANCE = 0.87
G057_DRAFT_OVER_VERIFY = 0.75
G057_GAMMA = 4
G057_OBSERVED_RATIO_VS_BASELINE = 0.91  # 0.91x = slowdown at 87% accept

FLASH_BUDGET_REL = "receipts/headless/FLASH_TOKEN_NS_BUDGET.json"
QWEN27_BUDGET_REL = "receipts/headless/QWEN27_TOKEN_NS_BUDGET.json"


class VerificationCorrectnessError(ValueError):
    """Placement changed WHAT is checked rather than WHERE it is checked."""

    CODE = "WHAT_NOT_WHERE"


# ---------------------------------------------------------------------------
# Recover helpers
# ---------------------------------------------------------------------------


def load_repo_json(rel: str) -> dict[str, Any] | None:
    """Disk first; git HEAD if the sparse checkout did not materialize it."""
    path = REPO / rel
    if path.is_file():
        try:
            return load_json(path)
        except (OSError, json.JSONDecodeError):
            return None
    raw = git("show", f"HEAD:{rel}")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def path_in_head(rel: str) -> bool:
    listing = git("ls-tree", "-r", "--name-only", "HEAD", rel)
    return any(line == rel for line in listing.splitlines())


def path_exists_or_in_head(rel: str) -> bool:
    return (REPO / rel).exists() or path_in_head(rel)


# ---------------------------------------------------------------------------
# Cost classes (structural, never nanoseconds)
# ---------------------------------------------------------------------------


def cost_class(bytes_per_token: float | None, mixer_bytes: float) -> str:
    if bytes_per_token is None or mixer_bytes <= 0:
        return "UNKNOWN"
    ratio = bytes_per_token / mixer_bytes
    if ratio >= 0.25:
        return "DOMINANT"
    if ratio >= 0.05:
        return "MATERIAL"
    if ratio >= 0.001:
        return "SUBORDINATE"
    return "NEGLIGIBLE"


# ---------------------------------------------------------------------------
# Leviathan yield (recovered from tools/headless/decoding_gravity.py)
# ---------------------------------------------------------------------------


def expected_accepted_per_pass(
    alpha: float, gamma: int, *, yield_includes_bonus: bool = True
) -> float:
    """Expected accepted tokens per speculative cycle.

    Leviathan (bonus included): (1-α^{γ+1})/(1-α), with α=0 → 1, α=1 → γ+1.
    G057-style (drafted kept, floored at 1): (1-α^γ)/(1-α).
    At α=0 the target still emits one token after rejecting the whole draft.
    """
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    if gamma == 0:
        return 1.0
    if alpha >= 1.0:
        return float(gamma + 1) if yield_includes_bonus else float(gamma)
    if alpha <= 0.0:
        return 1.0
    if yield_includes_bonus:
        return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)
    kept = (1.0 - alpha ** gamma) / (1.0 - alpha)
    return max(1.0, kept)


def k1_pays(alpha: float, draft_cost: float, verify_cost: float) -> bool:
    """G038 k=1: draft/verify < alpha. High acceptance is not a speedup by itself."""
    if verify_cost <= 0:
        return False
    return (draft_cost / verify_cost) < alpha


# ---------------------------------------------------------------------------
# Decode plan and the objective
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodePlan:
    """Analytical speculative (or baseline) plan. Costs are relative units.

    1.0 = one baseline target complete token (mixer + head + state + sample).
    No field here is a hardware measurement.
    """

    name: str
    gamma: int = 0
    alpha: float = 0.0
    draft_cost: float = 0.0
    verify_cost: float = 1.0
    rollback_cost: float = 0.0
    token_inflation: float = 1.0
    lm_head_scale: float = 1.0
    lm_head_share: float = 0.0
    sampling_cost: float = 0.0
    tokenizer_cost: float = 0.0
    ceremony_cost: float = 0.0
    verification_placement: str = "host_exact"
    predicate: str = PREDICATE_EXACT
    yield_includes_bonus: bool = True
    draft_kind: str = "none"


def _effective_verify_cost(plan: DecodePlan) -> float:
    share = plan.lm_head_share
    if share <= 0.0:
        return plan.verify_cost
    if not 0.0 <= share <= 1.0:
        raise ValueError("lm_head_share must be in [0, 1]")
    return plan.verify_cost * ((1.0 - share) + share * plan.lm_head_scale)


def _validate_plan_numbers(plan: DecodePlan) -> None:
    if plan.gamma < 0:
        raise ValueError("gamma must be >= 0")
    if not 0.0 <= plan.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    for field in (
        "draft_cost",
        "verify_cost",
        "rollback_cost",
        "token_inflation",
        "lm_head_scale",
        "sampling_cost",
        "tokenizer_cost",
        "ceremony_cost",
    ):
        if getattr(plan, field) < 0:
            raise ValueError(f"{field} must be >= 0")


def cycle_cost_units(plan: DecodePlan) -> dict[str, float]:
    """Draft + verify + rollback + ceremony. Rollback is not optional."""
    _validate_plan_numbers(plan)
    place_verification(plan.verification_placement, predicate=plan.predicate)
    verify = _effective_verify_cost(plan)
    draft_leg = float(plan.gamma) * plan.draft_cost
    if plan.gamma == 0:
        p_rollback = 0.0
        rollback_leg = 0.0
    else:
        p_rollback = 1.0 - (plan.alpha ** plan.gamma)
        rollback_leg = p_rollback * plan.rollback_cost * float(plan.gamma)
    organ_tax = plan.tokenizer_cost + plan.sampling_cost
    cycle = draft_leg + verify + rollback_leg + plan.ceremony_cost + organ_tax
    accepted = expected_accepted_per_pass(
        plan.alpha, plan.gamma, yield_includes_bonus=plan.yield_includes_bonus
    )
    return {
        "draft_leg": draft_leg,
        "verify_leg": verify,
        "rollback_leg": rollback_leg,
        "ceremony_leg": plan.ceremony_cost,
        "organ_tax": organ_tax,
        "cycle": cycle,
        "p_rollback": p_rollback,
        "expected_accepted": accepted,
    }


def accepted_complete_token_cost(plan: DecodePlan) -> float:
    """THE OBJECTIVE. Lower is better. Relative units, not nanoseconds.

    cycle(draft + verify + rollback + ceremony + tokenizer/sample tax)
    divided by expected accepted tokens, then multiplied by tokenizer
    inflation. A plan that raises draft throughput while lowering acceptance
    scores worse. A high-acceptance plan whose draft is not cheap enough
    also scores worse than baseline — the G057 scar.
    """
    parts = cycle_cost_units(plan)
    accepted = parts["expected_accepted"]
    if accepted <= 0.0:
        raise ValueError("expected accepted tokens must be positive")
    return (parts["cycle"] / accepted) * plan.token_inflation


def raw_draft_throughput_units(plan: DecodePlan) -> float:
    """Diagnostic only. Inverse of draft_cost. Must not be the scoreboard.

    Analog of AccelCostLedger::draft_side_throughput_not_scoreboard.
    """
    if plan.gamma <= 0 or plan.draft_cost <= 0.0:
        return 0.0
    return 1.0 / plan.draft_cost


def cost_breakdown(plan: DecodePlan) -> dict[str, Any]:
    parts = cycle_cost_units(plan)
    return {
        "plan": plan.name,
        "draft_kind": plan.draft_kind,
        "gamma": plan.gamma,
        "alpha": plan.alpha,
        "verification_placement": plan.verification_placement,
        "predicate": plan.predicate,
        "yield_includes_bonus": plan.yield_includes_bonus,
        "legs": parts,
        "accepted_complete_token_cost": accepted_complete_token_cost(plan),
        "raw_draft_throughput_units": raw_draft_throughput_units(plan),
        "objective_beats_baseline": accepted_complete_token_cost(plan) < 1.0,
        "unit": "relative_cost_units",
        "unit_definition": (
            "1.0 = one baseline target complete token; not a nanosecond"
        ),
    }


# Named plans for the negative control and the G057 scar.
# FAST drafts 8 cheap tokens at α=0.20. SLOW drafts 2 dearer tokens at α=0.90.
HIGH_DRAFT_LOW_ACCEPT = DecodePlan(
    name="high_draft_throughput_poor_acceptance",
    gamma=8,
    alpha=0.20,
    draft_cost=0.08,
    verify_cost=1.0,
    rollback_cost=0.04,
    ceremony_cost=0.05,
    verification_placement="host_exact",
    draft_kind="speculative_decoding",
)
SLOW_DRAFT_HIGH_ACCEPT = DecodePlan(
    name="slower_draft_high_acceptance",
    gamma=2,
    alpha=0.90,
    draft_cost=0.45,
    verify_cost=1.0,
    rollback_cost=0.04,
    ceremony_cost=0.05,
    verification_placement="host_exact",
    draft_kind="speculative_decoding",
)
BASELINE_PLAN = DecodePlan(
    name="baseline_no_speculation",
    gamma=0,
    alpha=0.0,
    draft_cost=0.0,
    verify_cost=1.0,
    rollback_cost=0.0,
    ceremony_cost=0.0,
    verification_placement="host_exact",
    draft_kind="none",
)
# G057: 87% accept, draft/verify=0.75, K=4, still a slowdown. G057 counted
# drafted-kept (no bonus) which is the convention that produced 0.91x.
G057_HIGH_ACCEPT_SLOWDOWN = DecodePlan(
    name="g057_87pct_accept_still_slower",
    gamma=G057_GAMMA,
    alpha=G057_ACCEPTANCE,
    draft_cost=G057_DRAFT_OVER_VERIFY,
    verify_cost=1.0,
    rollback_cost=0.05,
    verification_placement="host_exact",
    yield_includes_bonus=False,
    draft_kind="self_speculative",
)


# ---------------------------------------------------------------------------
# Verification placement (PhysicalGraph vocabulary)
# ---------------------------------------------------------------------------

# Ceremony (WHERE) may shrink. The predicate (WHAT) may not.
_PLACEMENT_WHERE = {
    "host_exact": {
        "moves": "where",
        "device_placement": {"selected": "host", "candidates": ["host", "gpu"]},
        "synchronization": [
            {
                "kind": "runtime_boundary",
                "status": "host_roundtrip_of_argmax_ids",
                "ceremony": "full",
            }
        ],
        "representation": {
            "verify_evidence": "full_argmax_ids",
            "digest_commits_to_same_predicate": False,
        },
        "ceremony": "full",
        "correctness_preserving": True,
        "admitted": True,
    },
    "fused": {
        "moves": "where",
        "device_placement": {
            "selected": "gpu",
            "note": "same command buffer as draft; no extra submit",
        },
        "synchronization": [
            {
                "kind": "fused_into_draft_cb",
                "status": "no_host_roundtrip",
                "ceremony": "reduced",
            }
        ],
        "representation": {
            "verify_evidence": "full_argmax_ids",
            "digest_commits_to_same_predicate": False,
        },
        "ceremony": "reduced",
        "correctness_preserving": True,
        "admitted": True,
    },
    "device_side": {
        "moves": "where",
        "device_placement": {"selected": "gpu", "candidates": ["gpu"]},
        "synchronization": [
            {
                "kind": "device_local_compare",
                "status": "one_result_flag_to_host",
                "ceremony": "reduced",
            }
        ],
        "representation": {
            "verify_evidence": "full_argmax_ids_resident_on_device",
            "digest_commits_to_same_predicate": False,
        },
        "ceremony": "reduced",
        "correctness_preserving": True,
        "admitted": True,
    },
    "digest": {
        "moves": "where",
        "device_placement": {"selected": "gpu"},
        "synchronization": [
            {
                "kind": "device_digest_of_predicate",
                "status": "compact_evidence_of_the_same_check",
                "ceremony": "reduced",
            }
        ],
        "representation": {
            "verify_evidence": "digest_of_exact_argmax_prefix",
            "digest_commits_to_same_predicate": True,
        },
        "ceremony": "reduced",
        "correctness_preserving": True,
        "admitted": True,
        "note": (
            "A digest is a compact commitment to the SAME predicate, not a "
            "weaker check. Replacing exact prefix match with a sketch that "
            "can false-accept is a WHAT change and is refused."
        ),
    },
}

_PLACEMENT_WHAT = {
    "sampled": {
        "moves": "what",
        "correctness_preserving": False,
        "admitted": False,
        "refuse_reason": (
            "sampled verification checks a subset of draft positions; that "
            "changes WHAT is checked, not WHERE"
        ),
    },
    "sparse": {
        "moves": "what",
        "correctness_preserving": False,
        "admitted": False,
        "refuse_reason": (
            "sparse verification skips positions; that changes WHAT is "
            "checked, not WHERE"
        ),
    },
}

PLACEMENT_CATALOG: dict[str, dict[str, Any]] = {**_PLACEMENT_WHERE, **_PLACEMENT_WHAT}


def place_verification(
    kind: str, *, predicate: str = PREDICATE_EXACT
) -> dict[str, Any]:
    """Admit a WHERE placement of the exact predicate; refuse a WHAT change.

    Returns a PhysicalGraph-shaped plan (hcli.physical_graph.v1 field names).
    Raises VerificationCorrectnessError with CODE WHAT_NOT_WHERE on refusal.
    """
    if kind not in PLACEMENT_CATALOG:
        raise VerificationCorrectnessError(
            f"{VerificationCorrectnessError.CODE}: unknown verification "
            f"placement {kind!r}"
        )
    spec = PLACEMENT_CATALOG[kind]
    if predicate != PREDICATE_EXACT:
        raise VerificationCorrectnessError(
            f"{VerificationCorrectnessError.CODE}: predicate {predicate!r} "
            f"is not {PREDICATE_EXACT}; changing the predicate changes WHAT "
            f"is checked"
        )
    if not spec.get("admitted"):
        raise VerificationCorrectnessError(
            f"{VerificationCorrectnessError.CODE}: {spec.get('refuse_reason')}"
        )

    computation = [
        {
            "id": "target_verify",
            "kind": "computation",
            "present": True,
            "predicate": PREDICATE_EXACT,
            "placement": kind,
            "what_is_checked": PREDICATE_EXACT,
            "moves": "where",
        }
    ]
    data = [
        {
            "id": "verify_evidence",
            "kind": "tensor_group",
            "bytes": None,
            "source": spec.get("representation", {}).get("verify_evidence"),
        }
    ]
    graph = {
        "schema": PHYSICAL_GRAPH_SCHEMA,
        "nomenclature_version": NOMENCLATURE,
        "semantic_type": "PhysicalGraphPlan",
        "compiler_stage": "PhysicalGraphCompiler",
        "model_id": "decode_civilization.verification",
        "computation": computation,
        "data": data,
        "representation": spec.get("representation", {}),
        "memory": [
            {"tier": "hot", "role": "draft_and_verify_working_set", "status": "candidate"}
        ],
        "residency": {"weights": "unresolved", "state": "committed_plus_provisional"},
        "state": {
            "kv_cache": "dual_committed_provisional",
            "recurrent_state": "rolled_back_with_kv",
            "invariant": "COMMITTED_IS_SOURCE_OF_TRUTH",
        },
        "precision": {"weight": "unresolved", "activation": "unresolved"},
        "dependencies": [
            {"from": "draft", "to": "target_verify", "kind": "dataflow"},
            {"from": "target_verify", "to": "accept_or_rollback", "kind": "control"},
        ],
        "device_placement": spec.get("device_placement", {"selected": None}),
        "synchronization": spec.get("synchronization", []),
        "evidence": [
            {
                "claim": "ceremony_reduction_does_not_change_predicate",
                "predicate": PREDICATE_EXACT,
                "placement": kind,
            }
        ],
        "qualification": "PLAN_ONLY",
        "ceremony": spec.get("ceremony"),
        "correctness_preserving": True,
        "admitted": True,
    }
    body = {k: v for k, v in graph.items()}
    graph["fingerprint"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return graph


def verification_catalog_public() -> dict[str, Any]:
    """Catalog including the refused WHAT placements, for the receipt."""
    rows = []
    for kind in sorted(PLACEMENT_CATALOG):
        spec = PLACEMENT_CATALOG[kind]
        rows.append(
            {
                "kind": kind,
                "moves": spec["moves"],
                "admitted": bool(spec.get("admitted")),
                "correctness_preserving": bool(spec.get("correctness_preserving")),
                "ceremony": spec.get("ceremony"),
                "refuse_reason": spec.get("refuse_reason"),
                "what_is_checked_if_admitted": (
                    PREDICATE_EXACT if spec.get("admitted") else None
                ),
            }
        )
    return {
        "law": (
            "Reducing verification CEREMONY is allowed. Weakening CORRECTNESS "
            "is not. A placement that changes WHAT is checked, rather than "
            "WHERE, is refused with WHAT_NOT_WHERE."
        ),
        "predicate": PREDICATE_EXACT,
        "where_admitted": sorted(_PLACEMENT_WHERE),
        "what_refused": sorted(_PLACEMENT_WHAT),
        "placements": rows,
    }


# ---------------------------------------------------------------------------
# Shape packs (structural; FLASH budget + Qwen3.8 geometry)
# ---------------------------------------------------------------------------


def qwen38_state_bytes() -> dict[str, Any]:
    kv_elems = QWEN38_GQA_LAYERS * 2 * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM
    kv_bf16 = kv_elems * 2
    rec_elems = (
        QWEN38_DELTANET_LAYERS
        * QWEN38_LINEAR_VALUE_HEADS
        * QWEN38_LINEAR_VALUE_HEAD_DIM
        * QWEN38_LINEAR_KEY_HEAD_DIM
    )
    rec_bf16 = rec_elems * 2
    rec_f32 = rec_elems * 4
    conv_channels = (
        QWEN38_LINEAR_KEY_HEADS * QWEN38_LINEAR_KEY_HEAD_DIM * 2
        + QWEN38_LINEAR_VALUE_HEADS * QWEN38_LINEAR_VALUE_HEAD_DIM
    )
    # conv state is per DeltaNet layer; kernel=4. Byte class only.
    crossover = rec_bf16 / kv_bf16 if kv_bf16 else None
    return {
        "source": "crates/hawking-core/src/model/qwen38_geometry.rs + STATE_GRAVITY census",
        "gqa_layers": QWEN38_GQA_LAYERS,
        "deltanet_layers": QWEN38_DELTANET_LAYERS,
        "kv_elements_per_token": kv_elems,
        "kv_bytes_per_token_bf16": kv_bf16,
        "recurrent_elements": rec_elems,
        "recurrent_bytes_bf16": rec_bf16,
        "recurrent_bytes_f32": rec_f32,
        "recurrent_grows_with_context": False,
        "kv_grows_with_context": True,
        "crossover_tokens_bf16": crossover,
        "conv_channels": conv_channels,
        "mtp_kv_bytes_per_token_fp16": QWEN38_MTP_KV_BYTES_PER_TOKEN_FP16,
    }


def kv_compression_ladder(kv_bytes_bf16: int) -> list[dict[str, Any]]:
    """Bytes-only ladder (odyssey state_gravity). Not a capability claim."""
    rows = []
    for name, kbits, vbits in (
        ("bf16_bf16", 16, 16),
        ("q8_q8", 8, 8),
        ("q8k_q4v_asymmetric", 8, 4),
        ("q4_q4", 4, 4),
    ):
        per = kv_bytes_bf16 * ((kbits + vbits) / 32.0)
        rows.append(
            {
                "scheme": name,
                "k_bits": kbits,
                "v_bits": vbits,
                "bytes_per_token": int(per),
                "reduction_x": (kv_bytes_bf16 / per) if per else None,
                "capability": "ABSENT",
                "note": (
                    "byte class only; long-context recall is the gate and is "
                    "not claimed here"
                ),
            }
        )
    return rows


def recover_flash_organs() -> dict[str, Any]:
    doc = load_repo_json(FLASH_BUDGET_REL)
    if not doc:
        return {
            "present": False,
            "path": FLASH_BUDGET_REL,
            "organs": [],
            "absent_reason": "not on disk and not readable from HEAD",
        }
    organs = []
    for row in doc.get("organs") or []:
        organs.append(
            {
                "organ": row.get("organ"),
                "source_active_bytes_per_token": row.get(
                    "source_active_bytes_per_token"
                ),
                "source_flops_per_token": row.get("source_flops_per_token"),
                "status": row.get("status"),
            }
        )
    organs.sort(key=lambda r: str(r.get("organ") or ""))
    mixer = 0.0
    for r in organs:
        if r.get("organ") in {"deltanet", "routed_experts", "sparse_attention"}:
            mixer += float(r.get("source_active_bytes_per_token") or 0)
    classified = []
    for r in organs:
        item = dict(r)
        item["cost_class"] = cost_class(
            r.get("source_active_bytes_per_token"), mixer or 1.0
        )
        classified.append(item)
    contract = doc.get("target_contract") or {}
    return {
        "present": True,
        "path": FLASH_BUDGET_REL,
        "schema": doc.get("schema"),
        "source": "git HEAD or disk; structural bytes/flops only",
        "kernel_only_or_raw_draft_timing_is_not_acceptable": contract.get(
            "kernel_only_or_raw_draft_timing_is_not_acceptable"
        ),
        "timing_unit_cited": contract.get("timing_unit"),
        "mixer_bytes_for_classification": mixer,
        "organs": classified,
    }


# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------


def tokenizer_cost_model(flash: dict[str, Any]) -> dict[str, Any]:
    return {
        "organ": "tokenizer",
        "topology": "byte_level_bpe_gpt2_style",
        "not_sentencepiece": True,
        "byte_alphabet_rows_required": TOKENIZER_BYTE_ALPHABET,
        "vocab_rows_geometry": QWEN38_VOCAB,
        "vocab_rows_tokenizer_gravity": TOKENIZER_GRAVITY_VOCAB_ROWS,
        "coupled_organs": ["tokenizer", "vocabulary", "embedding", "lm_head"],
        "embedding_bytes": TOKENIZER_GRAVITY_EMBED_BYTES,
        "lm_head_bytes": TOKENIZER_GRAVITY_LM_HEAD_BYTES,
        "combined_bytes": TOKENIZER_GRAVITY_COMBINED_BYTES,
        "payload_share": TOKENIZER_GRAVITY_PAYLOAD_SHARE,
        "cost_class": "MATERIAL_IF_INFLATION",
        "law": (
            "Removed tokens do not vanish: their text is re-encoded from what "
            "survives, so the model emits MORE tokens. Inflation multiplies "
            "accepted_complete_token_cost one-for-one. A vocabulary that "
            "saves bytes can lose wall time. ASCII-condensing is a CONTROL, "
            "never production policy (S011 §10)."
        ),
        "genome_cited": {
            "REQUIRED": 289,
            "HOT": 2931,
            "WARM": 5421,
            "COLD": 239436,
            "source": "receipts/headless/TOKENIZER_GRAVITY.json",
        },
        "heldout_trap": (
            "required+hot+warm saves ~11% payload at 1.56% corpus inflation "
            "and 2.2x/3.1x on held-out language. COLD is a fact about the "
            "corpus, not about what production must say."
        ),
        "flash_budget_has_tokenizer_organ": any(
            o.get("organ") == "tokenizer" for o in flash.get("organs") or []
        ),
    }


def lm_head_cost_model(flash: dict[str, Any]) -> dict[str, Any]:
    flash_head = next(
        (o for o in flash.get("organs") or [] if o.get("organ") == "lm_head"),
        None,
    )
    return {
        "organ": "lm_head",
        "shape": {
            "vocab": QWEN38_VOCAB,
            "hidden": QWEN38_HIDDEN,
            "kind": "GEMV_vocab_x_hidden_per_token",
        },
        "bytes_qwen38_tokenizer_gravity": TOKENIZER_GRAVITY_LM_HEAD_BYTES,
        "flash_source_active_bytes_per_token": (flash_head or {}).get(
            "source_active_bytes_per_token"
        ),
        "flash_source_flops_per_token": (flash_head or {}).get(
            "source_flops_per_token"
        ),
        "cost_class": (flash_head or {}).get("cost_class") or "MATERIAL",
        "shared_with_mtp": True,
        "law": (
            "LM head is paid on every target verify. MTP shares it "
            "(mtp_use_dedicated_embeddings=false on the Qwen3.8 parent). "
            "Shrinking V scales this organ and scales tokenizer inflation "
            "the other way; the objective multiplies inflation."
        ),
    }


def sampling_cost_model() -> dict[str, Any]:
    v = QWEN38_VOCAB
    h = QWEN38_HIDDEN
    return {
        "organ": "sampling",
        "argmax_compares": v,
        "softmax_flops_class": "O(V)",
        "lm_head_flops_class": "O(V*H)",
        "compute_ratio_vs_lm_head": (v / (2.0 * v * h)) if v and h else None,
        "cost_class_compute": "NEGLIGIBLE",
        "host_logits_readback_bytes": v * 4,
        "cost_class_if_host_readback": "MATERIAL",
        "law": (
            "Sampling FLOPs are subordinate to the LM-head GEMV by ~H. Host "
            "readback of full logits is ceremony (V f32 per token) and is "
            "the lever: keep sampling device-side or fused with the head. "
            "Argmax at temp=0 is the ExactTarget predicate."
        ),
        "flash_budget_has_sampling_organ": False,
    }


def state_representation_model() -> dict[str, Any]:
    st = qwen38_state_bytes()
    return {
        "organ": "state",
        "hybrid": True,
        "full_attention_interval": 4,
        "kv": {
            "grows_with_context": True,
            "bytes_per_token_bf16": st["kv_bytes_per_token_bf16"],
            "elements_per_token": st["kv_elements_per_token"],
            "cost_class": "MATERIAL_AT_LONG_CONTEXT",
        },
        "recurrent": {
            "grows_with_context": False,
            "bytes_bf16": st["recurrent_bytes_bf16"],
            "bytes_f32": st["recurrent_bytes_f32"],
            "elements": st["recurrent_elements"],
            "cost_class": "MATERIAL_AT_SHORT_CONTEXT",
        },
        "crossover_tokens_bf16": st["crossover_tokens_bf16"],
        "law": (
            "Only 16/64 layers keep KV, so this body already carries 4x less "
            "KV than a pure-attention model of the same depth. Recurrent "
            "state is a flat ~72 MiB bf16 whatever the context. KV overtakes "
            "it at ~1152 tokens. Prefix reuse beats KV precision below that "
            "crossover (G037 ranking)."
        ),
        "speculative_dual": {
            "invariant": "COMMITTED_IS_SOURCE_OF_TRUTH",
            "committed": "target-verified positions only",
            "provisional": "draft extension; rollback restores committed bit-identically",
            "source": "crates/hawking-speculate/src/kv_dual.rs",
        },
    }


def kv_compression_model() -> dict[str, Any]:
    st = qwen38_state_bytes()
    return {
        "organ": "kv_compression",
        "ladder": kv_compression_ladder(st["kv_bytes_per_token_bf16"]),
        "methods_censused_n048": ["KIVI", "MiniCache", "H2O"],
        "capability_gate": (
            "long-context recall is the gate; byte savings that would lose "
            "recall are not a GO (tools/headless/state_gravity.py)"
        ),
        "cost_class": "BYTES_ONLY_CAPABILITY_ABSENT",
        "asymmetric_kv_is_not_a_result": True,
    }


def recurrent_compression_model(flash: dict[str, Any]) -> dict[str, Any]:
    st = qwen38_state_bytes()
    flash_rec = next(
        (
            o
            for o in flash.get("organs") or []
            if o.get("organ") == "recurrent_state"
        ),
        None,
    )
    return {
        "organ": "recurrent_state_compression",
        "qwen38_recurrent_elements": st["recurrent_elements"],
        "qwen38_recurrent_bytes_f32": st["recurrent_bytes_f32"],
        "flash_source_active_bytes_per_token": (flash_rec or {}).get(
            "source_active_bytes_per_token"
        ),
        "flash_source_flops_per_token": (flash_rec or {}).get(
            "source_flops_per_token"
        ),
        "layout_cited": "[head][ki][vi], vi innermost; 128x128 per head",
        "not_an_mlp": True,
        "cannot_be_tiled_like_affine2_gemv": True,
        "rollback": (
            "provisional rec state rolls back with DualKv; cost scales with "
            "draft length and is inside accepted_complete_token_cost"
        ),
        "cost_class": (flash_rec or {}).get("cost_class") or "MATERIAL",
        "source": [
            "tools/headless/deltanet_organ.py",
            "crates/hawking-core/src/model/qwen38_token_ns_ledger.rs theoretical_state_bytes",
        ],
    }


def speculative_interfaces() -> dict[str, Any]:
    return {
        "mtp": {
            "role": "draft",
            "parent_qwen38": {
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": False,
                "n_tensors": QWEN38_MTP_TENSORS,
                "bf16_bytes": QWEN38_MTP_BF16_BYTES,
                "shared_lm_head": True,
                "extra_layer_is_gqa_not_deltanet": True,
                "runtimes_drop_heads": True,
                "alpha_mtp": "ABSENT",
                "source": "tools/headless/decoding_gravity.py",
            },
            "flash_organ_present": True,
            "status": "PARENT_HAS_HEADS_RUNTIMES_DROP_THEM",
        },
        "speculative_decoding": {
            "role": "draft_then_verify",
            "yield": "Leviathan expected_accepted_per_pass",
            "objective": "accepted_complete_token_cost",
            "forbids_scoreboard": "raw_draft_throughput_units",
            "foreign_metal_prior": {
                "url": "https://github.com/ggml-org/llama.cpp/issues/23752",
                "finding": (
                    "MTP speculative decoding degraded throughput on Metal "
                    "at every n_max in that foreign Qwen3.5-9B report"
                ),
                "not_our_27b": True,
                "kind": "CITED",
            },
        },
        "draft_verify": {
            "verify_rule": PREDICATE_EXACT,
            "trait": "ExactTarget",
            "source": "crates/hawking-speculate/src/verifier.rs",
            "durable_sinks_take": "VerifiedTokenId only",
            "empty_draft_is_one_greedy_bonus": True,
        },
        "multi_token_microdecoder": {
            "kind": "same_model_K_batch",
            "does_not_change_token_unit": True,
            "acceptance_is_binding": True,
            "source": "receipts/ascent-2026-08-16/G091_MULTI_TOKEN.json",
            "cited_breakeven_acceptance": 0.6302411121263896,
            "note": (
                "G091: only ACCEPTED tokens count; inflating TPS by changing "
                "the token unit is forbidden. Hardware TPS ceiling is UNKNOWN "
                "in this sidecar."
            ),
        },
        "state_rollback": {
            "invariant": "COMMITTED_IS_SOURCE_OF_TRUTH",
            "source": "crates/hawking-speculate/src/kv_dual.rs",
            "on_reject": "restore provisional to committed, bit-identical",
            "on_accept": "rebase committed by the verified prefix only",
            "in_objective": True,
        },
        "acceptance_accounting": {
            "objective": "accepted_complete_token_cost",
            "analog": (
                "crates/hawking-speculate/src/metrics_sep.rs AccelCostLedger "
                "(draft+verify+rollback in the denominator)"
            ),
            "separated_from": [
                "BASE_TRUE_TPS",
                "BLOCK_EXECUTED_TPS",
                "PREFILL_TPS",
                "TTFT",
                "raw draft throughput",
            ],
            "this_sidecar_does_not_emit_tps": True,
        },
    }


# ---------------------------------------------------------------------------
# Worked examples (deterministic, relative units)
# ---------------------------------------------------------------------------


def worked_examples() -> dict[str, Any]:
    baseline = cost_breakdown(BASELINE_PLAN)
    fast = cost_breakdown(HIGH_DRAFT_LOW_ACCEPT)
    slow = cost_breakdown(SLOW_DRAFT_HIGH_ACCEPT)
    g057 = cost_breakdown(G057_HIGH_ACCEPT_SLOWDOWN)
    fused = cost_breakdown(
        DecodePlan(
            name="same_as_slow_but_fused_verify",
            gamma=SLOW_DRAFT_HIGH_ACCEPT.gamma,
            alpha=SLOW_DRAFT_HIGH_ACCEPT.alpha,
            draft_cost=SLOW_DRAFT_HIGH_ACCEPT.draft_cost,
            verify_cost=SLOW_DRAFT_HIGH_ACCEPT.verify_cost,
            rollback_cost=SLOW_DRAFT_HIGH_ACCEPT.rollback_cost,
            ceremony_cost=0.0,
            verification_placement="fused",
            draft_kind="speculative_decoding",
        )
    )
    inflating = cost_breakdown(
        DecodePlan(
            name="tokenizer_required_hot_inflation",
            gamma=0,
            verify_cost=1.0,
            token_inflation=1.1912,
            lm_head_scale=0.013,
            lm_head_share=TOKENIZER_GRAVITY_PAYLOAD_SHARE,
            verification_placement="host_exact",
            draft_kind="none",
        )
    )
    return {
        "baseline": baseline,
        "high_draft_low_accept": fast,
        "slow_draft_high_accept": slow,
        "g057_high_accept_slowdown": g057,
        "fused_ceremony_reduction": fused,
        "tokenizer_inflation_loses": inflating,
        "negative_control": {
            "raw_draft_throughput_fast_gt_slow": (
                fast["raw_draft_throughput_units"]
                > slow["raw_draft_throughput_units"]
            ),
            "objective_fast_worse_than_slow": (
                fast["accepted_complete_token_cost"]
                > slow["accepted_complete_token_cost"]
            ),
            "g057_worse_than_baseline": (
                g057["accepted_complete_token_cost"]
                > baseline["accepted_complete_token_cost"]
            ),
            "inflation_worse_than_baseline": (
                inflating["accepted_complete_token_cost"]
                > baseline["accepted_complete_token_cost"]
            ),
            "fused_better_or_equal_slow": (
                fused["accepted_complete_token_cost"]
                <= slow["accepted_complete_token_cost"]
            ),
        },
    }


# ---------------------------------------------------------------------------
# Recovery census / gaps
# ---------------------------------------------------------------------------


def recovered_implementation() -> list[dict[str, Any]]:
    rows = [
        {
            "path": "tools/headless/decoding_gravity.py",
            "role": "N049 MTP census + Leviathan spec_cycle (no rollback)",
            "adequate_for": "yield arithmetic and the binary-as-draft refusal",
            "gap": "pass billed in ns; rollback absent; no PhysicalGraph placement",
        },
        {
            "path": "tools/odyssey/decoding_gravity.py",
            "role": "G038: high acceptance is not a speedup; 87% accept at 0.91x",
            "adequate_for": "the recorded trap this objective must express",
            "gap": "k=1 only; writes receipts/headless; not a sidecar interface",
        },
        {
            "path": "tools/odyssey/state_gravity.py",
            "role": "G037 KV vs recurrent census + KV precision ladder",
            "adequate_for": "hybrid state shapes and crossover_tokens=1152",
            "gap": "bytes only, no speculative DualKv cost",
        },
        {
            "path": "tools/headless/state_gravity.py",
            "role": "N048 KIVI/MiniCache/H2O/DeltaNet redundancy census",
            "adequate_for": "capability gate on KV compression",
            "gap": "CPU census; not an accepted-token objective",
        },
        {
            "path": "tools/headless/tokenizer_gravity.py",
            "role": "G036 tokenizer+embed+LM-head as one organ; inflation trap",
            "adequate_for": "vocab topology and the inflation-vs-bytes law",
            "gap": "not wired into a speculative complete-token objective",
        },
        {
            "path": "tools/headless/deltanet_organ.py",
            "role": "N026 recurrent-state organ autopsy (128x128, not an MLP)",
            "adequate_for": "recurrent state as a first-class organ",
            "gap": "GPU organ; sidecar must not rerun it",
        },
        {
            "path": "crates/hawking-core/src/decode_family.rs",
            "role": "G023 shared kernel family names (Q80 / Qwen3.8 / DSV4F)",
            "adequate_for": "which Metal symbols the decode graph dispatches",
            "gap": "name-only switch; not a cost model",
        },
        {
            "path": "crates/hawking-speculate/src/metrics_sep.rs",
            "role": "AccelCostLedger: accepted TPS includes draft+verify+rollback",
            "adequate_for": "the accounting identity this objective analogizes",
            "gap": "hardware ns; sidecar restates it in relative units",
        },
        {
            "path": "crates/hawking-speculate/src/kv_dual.rs",
            "role": "COMMITTED_IS_SOURCE_OF_TRUTH DualKv rollback/rebase",
            "adequate_for": "state rollback interface",
            "gap": "CPU mirror of ids; GPU tensor restore is Codex",
        },
        {
            "path": "crates/hawking-speculate/src/verifier.rs",
            "role": "ExactTarget prefix verify; VerifiedTokenId promotion",
            "adequate_for": "WHAT is checked (exact argmax prefix)",
            "gap": "no placement of WHERE the check runs",
        },
        {
            "path": "hcli/physical_graph.py",
            "role": "PhysicalGraph field vocabulary (PLAN_ONLY)",
            "adequate_for": "where/representation/synchronization names",
            "gap": "no verification-placement organ",
        },
        {
            "path": FLASH_BUDGET_REL,
            "role": "Flash-Next organ bytes/flops budget, including mtp and lm_head",
            "adequate_for": "structural shapes this cost model classifies",
            "gap": "all actual_* timing fields null; WAITING_FOR_NATIVE_EXECUTION",
        },
        {
            "path": "receipts/ascent-2026-08-16/G057_SELF_SPECULATIVE.json",
            "role": "REFUTED ON COST: 87% accept, 0.91x, draft/verify=0.75",
            "adequate_for": "the slowdown-at-high-acceptance scar",
            "gap": "campaign receipt, not a reusable plan object",
        },
        {
            "path": "receipts/ascent-2026-08-16/G091_MULTI_TOKEN.json",
            "role": "multi-token microdecoder; acceptance is the binding term",
            "adequate_for": "token unit must not change; breakeven α cited",
            "gap": "hardware TPS numbers; sidecar does not copy them",
        },
    ]
    for row in rows:
        row["on_disk_or_head"] = path_exists_or_in_head(row["path"])
    rows.sort(key=lambda r: r["path"])
    return rows


def negative_findings() -> list[dict[str, Any]]:
    qwen27 = load_repo_json(QWEN27_BUDGET_REL)
    stateful = path_exists_or_in_head(
        "receipts/headless/FLASH_STATEFUL_TPS_GATE_V14.json"
    )
    prefix_hits = git(
        "ls-tree", "-r", "--name-only", "HEAD"
    )
    prefix_fed = sorted(
        p
        for p in prefix_hits.splitlines()
        if "FLASH_PREFIX_FED" in p and p.endswith(".f32")
    )
    return [
        {
            "looked_for": QWEN27_BUDGET_REL,
            "found": qwen27 is not None,
            "used_instead": FLASH_BUDGET_REL,
            "note": (
                "No QWEN27_TOKEN_NS_BUDGET.json in HEAD. Closest structural "
                "token budget is FLASH_TOKEN_NS_BUDGET.json (Flash-Next organ "
                "bytes/flops). Qwen3.8-27B shapes come from qwen38_geometry.rs "
                "plus TOKENIZER_GRAVITY / STATE_GRAVITY receipts."
            ),
        },
        {
            "looked_for": "receipts/headless/FLASH_STATEFUL_TPS_GATE_V14.json",
            "found": stateful,
            "note": "Not in HEAD. Cannot cite a Flash stateful TPS gate.",
        },
        {
            "looked_for": "FLASH_PREFIX_FED_LAYER*_STATE.f32",
            "found": bool(prefix_fed),
            "hits": prefix_fed,
            "note": (
                "No prefix-fed layer state blobs in HEAD. Recurrent-state "
                "shapes cited from geometry and DELTANET_ORGAN constants, "
                "not from captured f32 dumps."
            ),
        },
        {
            "looked_for": "native MTP forward on a packed Qwen3.8 artifact",
            "found": False,
            "note": (
                "decoding_gravity: parent has mtp.* heads; every production "
                "runtime drops them. alpha_mtp is ABSENT. Not re-measured."
            ),
        },
        {
            "looked_for": "GPU-authoritative accepted complete-token ns",
            "found": False,
            "note": (
                "This sidecar has no protected GPU lease. Every hardware "
                "timing field is UNKNOWN / omitted. STATIC_ONLY."
            ),
        },
    ]


def gaps_closed() -> list[str]:
    return [
        "accepted_complete_token_cost(plan) in relative units, with rollback in the numerator",
        "named negative control: high draft throughput + poor α scores worse than a slower high-α plan",
        "G057 scar expressed as a DecodePlan that loses to baseline at 87% accept",
        "verification placement in PhysicalGraph vocabulary with WHAT_NOT_WHERE refusal",
        "tokenizer inflation wired into the same objective as speculation (vocab shrink can still lose)",
        "sidecar receipt under receipts/future (existing gravity modules write receipts/headless, Codex-owned)",
        "cost classes over FLASH_TOKEN_NS_BUDGET organ bytes plus Qwen3.8 hybrid state shapes, no nanoseconds",
    ]


# ---------------------------------------------------------------------------
# Self-test (the guards that must be able to fail)
# ---------------------------------------------------------------------------


def selftest() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    base = accepted_complete_token_cost(BASELINE_PLAN)
    record("baseline_is_one", math.isclose(base, 1.0, rel_tol=0, abs_tol=1e-12), base)

    fast = accepted_complete_token_cost(HIGH_DRAFT_LOW_ACCEPT)
    slow = accepted_complete_token_cost(SLOW_DRAFT_HIGH_ACCEPT)
    fast_raw = raw_draft_throughput_units(HIGH_DRAFT_LOW_ACCEPT)
    slow_raw = raw_draft_throughput_units(SLOW_DRAFT_HIGH_ACCEPT)
    record(
        "raw_throughput_would_pick_the_fast_plan",
        fast_raw > slow_raw,
        {"fast_raw": fast_raw, "slow_raw": slow_raw},
    )
    record(
        "objective_rejects_high_draft_low_accept",
        fast > slow,
        {"fast_cost": fast, "slow_cost": slow},
    )

    g057 = accepted_complete_token_cost(G057_HIGH_ACCEPT_SLOWDOWN)
    record("g057_high_accept_still_loses_to_baseline", g057 > base, g057)

    with_rb = HIGH_DRAFT_LOW_ACCEPT
    without_rb = DecodePlan(
        **{**asdict(HIGH_DRAFT_LOW_ACCEPT), "rollback_cost": 0.0}
    )
    record(
        "rollback_raises_the_objective",
        accepted_complete_token_cost(with_rb) > accepted_complete_token_cost(without_rb),
        {
            "with": accepted_complete_token_cost(with_rb),
            "without": accepted_complete_token_cost(without_rb),
        },
    )

    fused_ok = False
    try:
        place_verification("fused")
        fused_ok = True
    except VerificationCorrectnessError:
        fused_ok = False
    record("fused_where_is_admitted", fused_ok)

    sampled_fired = False
    sampled_msg = ""
    try:
        place_verification("sampled")
    except VerificationCorrectnessError as exc:
        sampled_fired = VerificationCorrectnessError.CODE in str(exc)
        sampled_msg = str(exc)
    record("sampled_what_is_refused", sampled_fired, sampled_msg)

    sparse_fired = False
    try:
        place_verification("sparse")
    except VerificationCorrectnessError as exc:
        sparse_fired = VerificationCorrectnessError.CODE in str(exc)
    record("sparse_what_is_refused", sparse_fired)

    weak_digest_fired = False
    try:
        place_verification("digest", predicate="first_token_only")
    except VerificationCorrectnessError as exc:
        weak_digest_fired = VerificationCorrectnessError.CODE in str(exc)
    record("digest_of_weaker_predicate_is_refused", weak_digest_fired)

    digest_ok = False
    try:
        place_verification("digest", predicate=PREDICATE_EXACT)
        digest_ok = True
    except VerificationCorrectnessError:
        digest_ok = False
    record("digest_of_exact_predicate_is_admitted", digest_ok)

    sampled_plan_fired = False
    try:
        accepted_complete_token_cost(
            DecodePlan(
                name="illegal_sampled",
                gamma=4,
                alpha=0.9,
                draft_cost=0.2,
                verification_placement="sampled",
            )
        )
    except VerificationCorrectnessError as exc:
        sampled_plan_fired = VerificationCorrectnessError.CODE in str(exc)
    record("objective_refuses_sampled_plan", sampled_plan_fired)

    extremes = (
        math.isclose(expected_accepted_per_pass(0.0, 4), 1.0)
        and math.isclose(expected_accepted_per_pass(1.0, 4), 5.0)
        and math.isclose(expected_accepted_per_pass(0.5, 1), 1.5)
    )
    record("leviathan_extremes", extremes)

    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "checks": checks}


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def build() -> Path:
    report = selftest()
    if not report["ok"]:
        failed = [c["name"] for c in report["checks"] if not c["ok"]]
        raise RuntimeError(f"decode_civilization selftest failed: {failed}")

    flash = recover_flash_organs()
    examples = worked_examples()
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Analytical cost models and interfaces for tokenizer, LM head, "
            "sampling, KV, recurrent state, and speculative/MTP decoding. "
            "Objective is accepted complete-token COST in relative units, "
            "never raw draft throughput."
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "objective": {
            "name": "accepted_complete_token_cost",
            "unit": "relative_cost_units",
            "unit_definition": (
                "1.0 = one baseline target complete token (mixer+head+state+"
                "sample), no speculation. Not a nanosecond."
            ),
            "formula": (
                "(gamma*draft_cost + verify + p_rollback*rollback_cost*gamma "
                "+ ceremony + tokenizer_cost + sampling_cost) "
                "/ expected_accepted_per_pass(alpha, gamma) * token_inflation"
            ),
            "includes_rollback": True,
            "forbids": [
                "raw draft throughput as the scoreboard",
                "acceptance rate alone as a win",
                "kernel-only timing",
                "changing the token unit to inflate the metric",
            ],
            "yield": "Leviathan; G057 scar uses drafted-kept (no bonus)",
            "hardware_authority": False,
        },
        "shapes": {
            "qwen38_geometry": {
                "layers": QWEN38_LAYERS,
                "deltanet_layers": QWEN38_DELTANET_LAYERS,
                "gqa_layers": QWEN38_GQA_LAYERS,
                "hidden": QWEN38_HIDDEN,
                "vocab": QWEN38_VOCAB,
            },
            "qwen38_state": qwen38_state_bytes(),
            "flash_token_budget": flash,
        },
        "cost_models": {
            "tokenizer": tokenizer_cost_model(flash),
            "lm_head": lm_head_cost_model(flash),
            "sampling": sampling_cost_model(),
            "state_representation": state_representation_model(),
            "kv_compression": kv_compression_model(),
            "recurrent_state_compression": recurrent_compression_model(flash),
        },
        "speculative_interfaces": speculative_interfaces(),
        "verification_placement": verification_catalog_public(),
        "worked_examples": examples,
        "selftest": report,
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "integration": {
            "call": "accepted_complete_token_cost(plan: DecodePlan) -> float",
            "place": (
                "place_verification(kind: str, *, predicate: str = "
                f"{PREDICATE_EXACT!r}) -> PhysicalGraph dict"
            ),
            "diagnostic_not_scoreboard": "raw_draft_throughput_units(plan) -> float",
            "refusal": "VerificationCorrectnessError.CODE == WHAT_NOT_WHERE",
        },
    }
    return write_receipt(RECEIPT, doc, GENERATOR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        report = selftest()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
