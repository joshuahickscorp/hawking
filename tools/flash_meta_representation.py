#!/usr/bin/env python3
"""Emit the prospective Flash meta-representation budget.

This is deliberately *not* another physical quantizer.  Traditional BPW is
the byte cost of a deployed packed tensor.  The meta metric in this file is a
pre-registration of how much executable information a teacher-constrained
Flash successor is allowed to spend on each organ when weights are replaced by
learned programs, latent codes, and protected exact islands.

The distinction matters for the current campaign:

* ``meta_bpw`` is a model-function description budget, normalized by the
  source parameter denominator;
* ``physical_ebpw`` stays null until a serialized representation and loader
  exist;
* coherence is a hard admission gate, not something inferred from the budget.

The output is therefore useful now as a single bounded target for Doctor and
Accelerator work, without laundering a prospective sub-1 target into a
complete-model compression claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CENSUS = ROOT / "receipts/headless/FLASH_ORGAN_CENSUS.json"
DEFAULT_BUDGET = ROOT / "receipts/headless/FLASH_EBPW_BUDGET.json"
DEFAULT_BYTES = ROOT / "receipts/headless/FLASH_COMPLETE_V0.BYTE_LEDGER.json"
DEFAULT_OUT = ROOT / "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json"

SCHEMA = "hawking.flash.meta_representation.v1"
MODEL = "Qwen/Qwen3.8-Flash-Next"
PINNED_REVISION = "34567a4712bc9766c4449e2e98e4468bfa24d915"

# The ladder is a research sweep, not a queue expansion and not a claim that
# any target is executable.  The first successful target must be selected by
# held-out function/capability evidence, with the critical islands allowed to
# retain premium information while predictable bulk is reduced.
META_BPW_SEARCH_LADDER = (1.0, 0.9, 0.8871807728336929, 0.8, 0.7, 0.5, 0.35, 0.25)

# These are pre-registered *targets* for a trained successor.  They are not
# measurements of the current source tensors.  The two large families get a
# budget that leaves room for exact small-organ islands while remaining below
# one meta bit per source weight.
META_TARGETS: dict[str, dict[str, Any]] = {
    "routed_experts": {
        "meta_bpw": 0.88,
        "program": "expert-local latent code + shared tile decoder + route-margin repair",
        "ledger": {
            "expert_latent_symbols": 0.54,
            "shared_decoder_amortized": 0.11,
            "router_margin_guard": 0.14,
            "format_and_seed": 0.09,
        },
        "runtime_shape": "route -> latent decode -> fused gate/up/SwiGLU/down accumulation",
    },
    "ngram_embedding": {
        "meta_bpw": 0.70,
        "program": "frequency-tiered n-gram generator + hot exact islands + residual symbols",
        "ledger": {
            "tier_code": 0.38,
            "shared_embedding_generator_amortized": 0.14,
            "hot_key_residual": 0.12,
            "format_and_seed": 0.06,
        },
        "runtime_shape": "n-gram key -> resident generator lookup -> fused embedding add",
    },
    "linear_attention_hyperconnection": {
        "meta_bpw": 2.50,
        "program": "protected mixed-precision executable island",
        "ledger": {"protected_island": 2.50},
        "runtime_shape": "resident recurrent state and fused projection path",
    },
    "embedding_lm_head": {
        "meta_bpw": 3.50,
        "program": "higher-precision vocabulary and terminal-logit island",
        "ledger": {"protected_island": 3.50},
        "runtime_shape": "resident lookup / terminal projection",
    },
    "full_attention": {
        "meta_bpw": 3.00,
        "program": "KV-sensitive protected attention island",
        "ledger": {"protected_island": 3.00},
        "runtime_shape": "fused QKV/attention/KV-cache path",
    },
    "mlp_hyperconnection": {
        "meta_bpw": 2.50,
        "program": "protected HyperConnection coefficient island",
        "ledger": {"protected_island": 2.50},
        "runtime_shape": "fused HC read/write boundary",
    },
    "shared_expert": {
        "meta_bpw": 2.50,
        "program": "protected shared-expert island until direct generator parity",
        "ledger": {"protected_island": 2.50},
        "runtime_shape": "resident shared-expert path",
    },
    "other": {
        "meta_bpw": 2.50,
        "program": "protected miscellaneous island",
        "ledger": {"protected_island": 2.50},
        "runtime_shape": "existing native path",
    },
    "norm": {
        "meta_bpw": 16.00,
        "program": "exact norm island",
        "ledger": {"protected_island": 16.00},
        "runtime_shape": "exact normalization/state semantics",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _source_families(census: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = census.get("family_summary")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Flash census has no family_summary")
    total = _finite_nonnegative(
        census.get("source_parameter_bytes_indexed"),
        name="source_parameter_bytes_indexed",
    )
    if total <= 0.0:
        raise ValueError("Flash census source denominator is empty")
    result: list[dict[str, Any]] = []
    fraction_sum = 0.0
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("family_summary rows must be objects")
        family = str(row.get("family") or "")
        source_bytes = _finite_nonnegative(row.get("bytes"), name=f"{family}.bytes")
        if family not in META_TARGETS:
            raise ValueError(f"no pre-registered meta target for family {family!r}")
        fraction = source_bytes / total
        fraction_sum += fraction
        target = META_TARGETS[family]
        result.append(
            {
                "family": family,
                "source_bytes": int(source_bytes),
                "source_fraction": fraction,
                "meta_bpw_target": float(target["meta_bpw"]),
                "weighted_meta_bpw": fraction * float(target["meta_bpw"]),
                "program": target["program"],
                "ledger": dict(target["ledger"]),
                "runtime_shape": target["runtime_shape"],
            }
        )
    if abs(fraction_sum - 1.0) > 1e-6:
        raise ValueError(f"family fractions do not close: {fraction_sum}")
    return result


def _optional_binding(path: Path, *, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "present": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _search_ladder() -> dict[str, Any]:
    gates = (
        "held-out hidden/state error",
        "router top-K membership and order",
        "low-margin route stability",
        "routed-output and terminal-logit behavior",
        "short-horizon agreement and long-run no-collapse",
        "capability-sensitive failures",
    )
    return {
        "objective": "find the smallest useful detached executable program, not merely a sub-1 budget",
        "allocation_policy": (
            "heterogeneous: premium bits for router/state/KV/terminal islands, near-zero only for bulk "
            "that survives conditional prediction and held-out tests"
        ),
        "targets": [
            {
                "meta_bpw_target": target,
                "status": "NOT_MEASURED",
                "physical_ebpw": None,
                "candidate_kind": "heterogeneous_functional_budget",
                "required_gates": list(gates),
                "failure_diagnosis": "record the first failing gate and preserve the scar before lowering the target",
            }
            for target in META_BPW_SEARCH_LADDER
        ],
        "lowest_useful_target": None,
        "physical_ebpw": None,
    }


def build_receipt(*, census_path: Path, budget_path: Path, bytes_path: Path) -> dict[str, Any]:
    started = time.perf_counter_ns()
    census = _load(census_path)
    budget = _load(budget_path) if budget_path.is_file() else {}
    byte_ledger = _load(bytes_path) if bytes_path.is_file() else {}
    families = _source_families(census)
    meta_bpw = sum(float(row["weighted_meta_bpw"]) for row in families)
    if not math.isfinite(meta_bpw):
        raise ValueError("meta budget is not finite")

    source_bytes = int(census["source_parameter_bytes_indexed"])
    source_params = source_bytes // 2
    target_bytes_equivalent = meta_bpw * source_params / 8.0
    current_control = (byte_ledger.get("complete_exact_control") or {})
    current_fast = (byte_ledger.get("measured_fastpath_profile") or {})
    bounded = budget.get("bounded_kernel_parity") or {}
    bounded_representation = bounded.get("noetic_representation") or {}

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PROSPECTIVE_META_ONLY",
        "model": MODEL,
        "pinned_revision": PINNED_REVISION,
        "metric": {
            "name": "meta_bpw",
            "definition": "teacher-constrained executable information budget per source parameter",
            "unit": "bits per source weight (description budget, not serialized bytes)",
            "traditional_physical_metric": "physical_ebpw",
            "physical_ebpw": None,
            "prospective_target": meta_bpw,
            "below_one_target": bool(meta_bpw < 1.0),
            "headroom_to_one": 1.0 - meta_bpw,
            "target_bytes_equivalent": int(round(target_bytes_equivalent)),
            "source_parameter_denominator": source_params,
        },
        "source": {
            "census": _optional_binding(census_path, label="Flash organ census"),
            "ebpw_budget": _optional_binding(budget_path, label="existing Flash representation budget"),
            "byte_ledger": _optional_binding(bytes_path, label="exact Flash control byte ledger"),
            "source_parameter_bytes_indexed": source_bytes,
        },
        "family_budget": families,
        "budget_search": _search_ladder(),
        "meta_program": {
            "kind": "detached_functional_representation",
            "version": 1,
            "dense_weight_materialization": "forbidden",
            "expert_bank": {
                "kind": "expert_local_teacher_distilled_generator",
                "why_not_global_sharing": "existing Flash route-conditioned shared/output basis screens are unhealthy; share the decoder program, not a single expert basis",
                "per_expert_code": "latent symbols plus small learned residual codes",
                "direct_consumer": "fused generated-tile GEMV / route accumulation",
            },
            "ngram_bank": {
                "kind": "frequency_tiered_embedding_generator",
                "hot_exact_island": True,
                "cold_key_policy": "generator code with measured frequency-weighted residual",
                "direct_consumer": "lookup and embedding add without dense rematerialization",
            },
            "protected_islands": [
                "router logits/top-k/tie semantics",
                "DeltaNet recurrent state transition",
                "HyperConnection norms and gates",
                "KV-cache indexing and attention normalization",
                "terminal lm_head / tokenizer boundary",
            ],
        },
        "coherence_contract": {
            "admission": "all hard gates must pass; meta budget alone never admits a model",
            "teacher_distillation": {
                "required": True,
                "fit_and_holdout_split": True,
                "objective": "hidden-state, router-logit, routed-output, and terminal-logit distillation",
            },
            "router": {
                "topk_membership_match": 1.0,
                "topk_order_match": 1.0,
                "margin_guard": "preserve low-margin decisions or invoke an explicitly measured exact island",
            },
            "state": {
                "recurrent_state_semantics": "exact_or_source-approved",
                "kv_cache_semantics": "exact_or_source-approved",
                "fallback_count": 0,
            },
            "generation": {
                "short_horizon_token_agreement": 1.0,
                "long_horizon_no_collapse": True,
                "capability_suite": "must match the protected teacher contract",
            },
            "measured_now": False,
        },
        "accelerator_contract": {
            "dense_rematerialization": False,
            "resident_shared_program": True,
            "route_before_payload": True,
            "fused_boundaries": [
                "route -> expert code fetch -> generated tile -> gate/up/SwiGLU -> down accumulation",
                "n-gram key -> generated embedding -> residual add",
                "MoE output -> HyperConnection/residual write",
            ],
            "latency_admission": {
                "gpu_ns_per_token": "must not exceed matched physical control",
                "complete_wall_ns_per_accepted_token": "must not exceed matched physical control",
                "dispatches_per_token": "must be measured; generator decode cannot hide extra submissions",
                "synchronization_ns_per_token": "must be reported separately",
            },
            "measured_now": False,
        },
        "current_evidence": {
            "control_complete_ebpw": current_control.get("complete_ebpw"),
            "fastpath_complete_ebpw": current_fast.get("complete_ebpw"),
            "bounded_routed_q4_component_bpw": bounded_representation.get("effective_bits_per_value"),
            "bounded_routed_q4_status": bounded.get("status"),
            "route_shared_basis_status": "negative_existing_screen",
        },
        "measurement_state": {
            "meta_budget": "COMPUTED_FROM_CENSUS_AND_PREREGISTERED_TARGETS",
            "serialized_artifact": "NOT_BUILT",
            "physical_loader": "NOT_BUILT",
            "native_kernel": "NOT_BUILT",
            "complete_token": "NOT_MEASURED",
            "capability": "NOT_MEASURED",
            "physical_ebpw": "NULL_BY_RULE",
            "promotion_allowed": False,
        },
        "claim_boundary": (
            "This receipt proves only that the pre-registered functional meta-budget closes below 1.0 meta-BPW when weighted over the indexed Flash organ census. "
            "It does not prove a serialized artifact, physical EBPW, source parity, capability, accepted-token coherence, GPU latency, or residency."
        ),
        "next_gate": (
            "fit the expert-local generator and frequency-tiered n-gram generator against teacher traces; then run held-out router/terminal coherence before any physical packer or Metal kernel"
        ),
        "bench": {
            "state": "UNKNOWN",
            "measurement_state": "STATIC_META_BUDGET",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recorded_by": "tools/flash_meta_representation.py",
            "machine": "Apple host CPU; receipt/header metadata only",
            "rule": "S032 §3 -- meta budget is not physical EBPW",
            "elapsed_ns": time.perf_counter_ns() - started,
        },
    }
    receipt["seal_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET)
    parser.add_argument("--bytes", dest="bytes_path", type=Path, default=DEFAULT_BYTES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    receipt = build_receipt(
        census_path=args.census,
        budget_path=args.budget,
        bytes_path=args.bytes_path,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "meta_bpw": receipt["metric"]["prospective_target"],
                "below_one_target": receipt["metric"]["below_one_target"],
                "physical_ebpw": receipt["metric"]["physical_ebpw"],
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
