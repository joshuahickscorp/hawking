#!/usr/bin/env python3.12
"""FS6: what a functional GLM-5.2 actually costs, over every logical weight in the model.

A student that replaces one layer's MoE at 0.0104 bits per replaced weight is not a
0.0104-bit model.  The MoE is 97.9 percent of the logical weights, so the 2.1 percent it
does not touch sets a floor the functional codec cannot cross: attention, the indexer,
embeddings, the head, the dense layers and every norm still have to be stored.  At source
precision that residue alone is 0.3378 bits per weight, which is above the one-third rung.

Every number here is derived from ``GLM52_LOGICAL_WEIGHT_LEDGER.json``, whose extents were
read from the immutable shard headers, and from the measured artifact bytes of the fitted
students.  Nothing is a headline parameter count.

    run
    selftest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CONDENSE = Path(__file__).resolve().parent
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "GLM52_LOGICAL_WEIGHT_LEDGER.json"
OUT = ROOT / "reports" / "condense" / "glm52_generation_b"

HIDDEN = 6144
MOE_LAYERS = 76          # 75 sparse main layers plus the MTP layer
HEADER_BYTES = 68        # functional payload header plus the trailing layer id

# What the functional codec stands in for, per MoE layer, exactly.
PER_LAYER_REPLACED = (256 * 2048 * HIDDEN * 3) + (2048 * HIDDEN * 3) + (HIDDEN * 256 + 256)

# Rungs the Ascension plan requires an exact answer for.
RUNGS = (0.75, 0.5, 1.0 / 3.0)

# Candidates carried out of FS0/FS3, with their measured per-layer payload and the skill
# they earned at layer 38 on the disjoint score split.
CANDIDATES = {
    "student_h1024": {"stored": 1024 * HIDDEN, "skill_l38": 0.5318,
                      "expanded_per_layer": HIDDEN * 1024 * 4,
                      "note": "random-feature student, seeded projection, fp16 readout"},
    "linear_full": {"stored": HIDDEN * HIDDEN, "skill_l38": 0.5441,
                    "expanded_per_layer": 0,
                    "note": "dense upper control; no generated state to hold"},
    "structured_h1024_r256": {"stored": 1024 * 256 + 256 * HIDDEN, "skill_l38": 0.3921,
                              "expanded_per_layer": HIDDEN * 1024 * 4,
                              "note": "random features with a rank-256 factored readout"},
    "linear_rank256": {"stored": 256 * HIDDEN, "skill_l38": 0.5052,
                       "expanded_per_layer": HIDDEN * 256 * 4,
                       "note": "seeded projection to 256, then a readout; no nonlinearity"},
    "linear_rank64": {"stored": 64 * HIDDEN, "skill_l38": 0.4453,
                      "expanded_per_layer": HIDDEN * 64 * 4,
                      "note": "cheapest row that still clears the gate"},
}

# What is left when the MoE function is gone.  These are real tensors that must ship.
PROTECTED_VIEWS = ("attention", "indexer", "embeddings", "lm_head", "dense_mlp",
                   "normalization", "mtp_projection", "mtp_normalization", "mtp_head_norm")
PROTECTED_RATES = {"bf16_source": 16.0, "int8": 8.0, "int4": 4.0, "two_bit": 2.0,
                   "one_bit": 1.0}


def views() -> dict:
    return json.loads(LEDGER.read_text())


def protected_weights(ledger: dict) -> dict:
    categories = ledger["primary_categories"]
    return {name: categories[name]["logical_weights"] for name in PROTECTED_VIEWS
            if name in categories}


def run() -> dict:
    ledger = views()
    total = ledger["logical_weight_denominator"]
    protected = protected_weights(ledger)
    protected_total = sum(protected.values())
    replaced_total = PER_LAYER_REPLACED * MOE_LAYERS

    accounted = protected_total + replaced_total
    coverage = {
        "logical_weight_denominator": total,
        "functionally_replaced_weights": replaced_total,
        "protected_weights": protected_total,
        "accounted": accounted,
        "unaccounted": total - accounted,
        "replaced_fraction": replaced_total / total,
        "protected_fraction": protected_total / total,
    }

    floors = {name: protected_total * bits / total for name, bits in PROTECTED_RATES.items()}

    rows = []
    for label, spec in CANDIDATES.items():
        per_layer_bytes = HEADER_BYTES + spec["stored"] * 2
        artifact_bytes = per_layer_bytes * MOE_LAYERS
        functional_bpw = artifact_bytes * 8 / total
        expanded = spec["expanded_per_layer"] * MOE_LAYERS
        row = {
            "candidate": label,
            "note": spec["note"],
            "layer_38_skill": spec["skill_l38"],
            "per_layer_artifact_bytes": per_layer_bytes,
            "organ_local_bpw": per_layer_bytes * 8 / PER_LAYER_REPLACED,
            "all_layers_artifact_bytes": artifact_bytes,
            "functional_contribution_bpw": functional_bpw,
            "expanded_feature_map_bytes_all_layers": expanded,
            "resident_bytes_explicit": artifact_bytes + expanded,
            "resident_bytes_procedural": artifact_bytes,
            "complete_model_bpw": {
                name: functional_bpw + floor for name, floor in floors.items()},
            "complete_model_bytes": {
                name: artifact_bytes + int(protected_total * bits / 8)
                for name, bits in PROTECTED_RATES.items()},
        }
        row["rungs"] = {
            f"{rung:.6f}": {
                "reachable_at_bf16_protected":
                    row["complete_model_bpw"]["bf16_source"] <= rung,
                "cheapest_protected_rate_that_reaches": next(
                    (name for name, bits in sorted(PROTECTED_RATES.items(),
                                                   key=lambda item: -item[1])
                     if functional_bpw + protected_total * bits / total <= rung), None),
            } for rung in RUNGS}
        rows.append(row)

    rows.sort(key=lambda row: row["complete_model_bpw"]["bf16_source"])
    return {
        "schema": "hawking.glm52.functional_byte_auction.v1",
        "stage": "FS6",
        "source": "GLM52_LOGICAL_WEIGHT_LEDGER.json, header-derived extents",
        "moe_layers": MOE_LAYERS,
        "per_layer_replaced_weights": PER_LAYER_REPLACED,
        "coverage": coverage,
        "protected_organs": protected,
        "protected_floor_bpw": floors,
        "binding_fact": (
            "the protected residue is 2.11 percent of the logical weights and at source "
            "precision costs %.6f bits per weight on its own, which is above the "
            "one-third rung before a single expert byte is stored"
            % floors["bf16_source"]),
        "candidates": rows,
        "not_billed_here": [
            "Doctor state, if a diagnosis-matched Doctor is ever fitted",
            "runtime tables and per-layer metadata beyond the payload header",
            "KV/state traffic, which is a runtime cost and not an artifact cost",
        ],
    }


def selftest() -> int:
    ledger = views()
    total = ledger["logical_weight_denominator"]

    # The replaced and protected sets must partition the model without overlap or gap.
    protected = sum(protected_weights(ledger).values())
    replaced = PER_LAYER_REPLACED * MOE_LAYERS
    assert protected + replaced == total, (protected + replaced, total)

    # The router is billed as replaced, because the functional contract removes routing.
    assert PER_LAYER_REPLACED * MOE_LAYERS == (
        ledger["primary_categories"]["routed_expert"]["logical_weights"]
        + ledger["primary_categories"]["shared_expert"]["logical_weights"]
        + ledger["primary_categories"]["router"]["logical_weights"]
        + ledger["primary_categories"]["router_control"]["logical_weights"])

    result = run()
    # A free student still cannot reach one third with BF16 protected organs: that is the
    # arithmetic the whole auction exists to state.
    assert result["protected_floor_bpw"]["bf16_source"] > 1.0 / 3.0, \
        result["protected_floor_bpw"]["bf16_source"]
    # And every candidate must be under a bit once the protected organs are int8.
    for row in result["candidates"]:
        assert row["complete_model_bpw"]["int8"] < 1.0, row["candidate"]
        assert row["organ_local_bpw"] > row["functional_contribution_bpw"], row["candidate"]

    print(json.dumps({
        "selftest": "PASS",
        "protected_fraction": round(result["coverage"]["protected_fraction"], 6),
        "protected_floor_bpw_bf16": round(result["protected_floor_bpw"]["bf16_source"], 6),
        "cheapest_complete_bpw_at_int8": round(
            min(row["complete_model_bpw"]["int8"] for row in result["candidates"]), 6),
    }))
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if command == "selftest":
        raise SystemExit(selftest())
    if command == "run":
        payload = run()
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "GLM52_FUNCTIONAL_BYTE_AUCTION.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(json.dumps({k: v for k, v in payload.items() if k != "candidates"},
                         indent=2))
        for row in payload["candidates"]:
            print(f"{row['candidate']:24} organ {row['organ_local_bpw']:.6f}  "
                  f"functional {row['functional_contribution_bpw']:.6f}  "
                  f"complete@bf16 {row['complete_model_bpw']['bf16_source']:.6f}  "
                  f"@int8 {row['complete_model_bpw']['int8']:.6f}  "
                  f"@int4 {row['complete_model_bpw']['int4']:.6f}")
    else:
        raise SystemExit(f"unknown command: {command}")
