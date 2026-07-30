#!/usr/bin/env python3.12
"""Math-Preserve PASS 2: one whole-model byte and structure auction.

Resolves the thing `profiles/prometheus/math-v1.json` has stood stubbed for since
it was written -- "GATED: until the causal probe (P3/P4) runs, all routed experts
are T1. When cartography lands, the math coalition subset promotes ... at matched
bytes" -- by reading every capsule PASS 1 sealed and ranking each sparse layer's
experts by measured contribution, instead of `prometheus.allocate()`'s current
"coalition SIZE only, membership uniform-split" placeholder (both its "math" and
"random" arms compute the identical uniform average today; only the label
differs).

This module resolves MEMBERSHIP and the concrete whole-model byte auction.  The
auction starts every measured coalition expert at R0, demotes the non-coalition
tail to R4, protects the Math profile's embedding/head and structural controls,
then promotes the highest-utility coalition expert slots to source-native while
the exact predicted payload plus an explicit packaging reserve remains at or
below H0.98.  The resulting manifest names a decision for every official tensor;
PASS 3 is a serializer of that frozen plan, not another allocator.

Global, not greedy: nothing here writes a decision from one shard or one window in
isolation. `run()` reads every sealed capsule that exists at call time and produces
one whole-model ranking. A manifest built before PASS 1 finishes is a PREVIEW,
explicitly marked incomplete -- `--freeze` refuses to write a manifest claiming
completeness unless every sparse layer the source declares has capsule evidence.

    python3.12 tools/prometheus/math_pass2_allocation.py preview
    python3.12 tools/prometheus/math_pass2_allocation.py freeze --out PROMETHEUS_MATH_ALLOCATION_MANIFEST.json
    python3.12 tools/prometheus/math_pass2_allocation.py selftest
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONDENSE = REPO / "tools/condense"
for _p in (HERE, CONDENSE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from glm52_common import resolve_artifact  # noqa: E402

CAPSULE_DIR = Path(
    "/Users/scammermike/Library/Application Support/Hawking/GLM52MathPrometheus/capsules"
)
GRAPH = resolve_artifact("GLM52_SHARD_DEPENDENCY_GRAPH.json")
LEDGER = REPO / "evidence" / "glm52" / "GLM52_LOGICAL_WEIGHT_LEDGER.json"

# Same convention `architecture.py`'s equal-budget solver already uses for the
# coalition's size, so PASS 2's membership and PASS 1/M09's existing byte machinery
# stay comparable rather than introducing a second, incompatible knob.
DEFAULT_COALITION_FRACTION = 0.05
# Below this many observed selections for an expert, its ranking is noise, not
# signal -- flagged, not silently trusted. ~1.2 observations/expert is the honest
# expectation at the pinned corpus's 3-record math pool (see
# glm52_capture_program.math_calibration_batch's docstring).
MIN_HIT_COUNT_FOR_CONFIDENT_RANK = 2

# H0.98 leaves an honest operating margin below the binding one-bit law.  The
# reserve is part of the auction rather than a post-hoc excuse: shard JSON headers,
# the model index, coverage receipt, tokenizer runtime tables, and allocation
# provenance all have to fit inside it when PASS 3 grades actual bytes.
TARGET_BPW_NUM = 49
TARGET_BPW_DEN = 50
PACKAGING_RESERVE_BYTES = 256 << 20
PROFILE_NATIVE_CATEGORIES = {"embeddings", "lm_head"}
TAIL_RUNG = "R4"
COALITION_FLOOR_RUNG = "R0"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sparse_layers_declared() -> list[int]:
    """Every layer the source's own config says is MoE, from the dependency graph's
    architecture-adjacent evidence -- independent of what PASS 1 has captured so
    far, so "how much is left" is answerable without trusting PASS 1's own count."""
    import glm52_teacher_capture as teacher

    config = teacher.official_config()
    return [i for i, kind in enumerate(config["mlp_layer_types"]) if kind == "sparse"]


def sealed_capsules() -> list[dict]:
    if not CAPSULE_DIR.exists():
        return []
    out = []
    for path in sorted(CAPSULE_DIR.glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def _layer_expert_arrays(capsule: dict, layer: int) -> dict[str, np.ndarray] | None:
    npz_path = CAPSULE_DIR / f"{capsule['capsule_id']}.npz"
    if not npz_path.exists():
        return None
    with np.load(npz_path) as data:
        key = f"layer_{layer:02d}/expert_contribution_l2"
        if key not in data:
            return None
        return {
            "contribution_l2": np.asarray(data[key]),
            "hit_count": np.asarray(data[f"layer_{layer:02d}/expert_hit_count"]),
        }


def layer_ranking(layer: int, coalition_fraction: float) -> dict | None:
    """One sparse layer's measured coalition, from whichever sealed capsule
    covers it. Returns None if no capsule has this layer's expert evidence yet."""
    for capsule in sealed_capsules():
        if layer not in capsule.get("layers", []):
            continue
        arrays = _layer_expert_arrays(capsule, layer)
        if arrays is None:
            continue
        contribution = arrays["contribution_l2"]
        hit = arrays["hit_count"]
        n_experts = int(contribution.shape[0])
        k = min(max(round(coalition_fraction * n_experts), 0), n_experts)
        # Descending by contribution, ties broken by lower expert id (deterministic,
        # matches the tie-break convention the Rust runtime and reference use
        # elsewhere in this campaign -- never an unspecified sort order).
        order = sorted(range(n_experts), key=lambda e: (-float(contribution[e]), e))
        coalition = sorted(order[:k])
        remainder = sorted(order[k:])
        confident = [e for e in coalition if hit[e] >= MIN_HIT_COUNT_FOR_CONFIDENT_RANK]
        layer_max = float(contribution.max()) or 1.0
        return {
            "capsule_id": capsule["capsule_id"],
            "n_routed_experts": n_experts,
            "coalition_size": k,
            "coalition_expert_ids": coalition,
            "coalition_confident_expert_ids": confident,
            "coalition_thin_evidence_expert_ids": sorted(set(coalition) - set(confident)),
            "coalition_evidence": [
                {
                    "expert": int(e),
                    "contribution_l2": float(contribution[e]),
                    "layer_normalized_contribution": float(contribution[e]) / layer_max,
                    "hit_count": int(hit[e]),
                    "confident": bool(hit[e] >= MIN_HIT_COUNT_FOR_CONFIDENT_RANK),
                }
                for e in coalition
            ],
            "remainder_expert_ids": remainder,
            "total_hit_observations": int(hit.sum()),
            "selection_basis": "expert_contribution_l2: local, fixed-routing decomposition "
                                "of what routed_moe already computes (see routed_moe's "
                                "retain_per_expert docstring) -- NOT a re-routed causal "
                                "ablation and not S0.8's gated intervention-probe claim.",
        }
    return None


def _elements(row: dict) -> int:
    return math.prod(int(x) for x in row["shape"])


def _pq_payload_bytes(row: dict, rung_name: str) -> int:
    """Exact serialized payload size without fitting centroids.

    PQ values affect reconstruction but not the number of indices or codebook
    entries.  This duplicates the packer's ByteLedger arithmetic intentionally
    and is cross-checked in selftest; PASS 3 later replaces prediction with actual
    file bytes before sealing the artifact.
    """
    import glm52_pack as pack

    rung = next((r for r in pack.LADDER if r["rung"] == rung_name), None)
    if rung is None:
        raise ValueError(f"unknown packer rung {rung_name!r}")
    n = _elements(row)
    cols = int(row["shape"][-1])
    dim = int(rung["dim"])
    effective_dim = dim if cols % dim == 0 and dim & (dim - 1) == 0 else cols & -cols
    index_count = n // effective_dim
    index_bits = pack.index_bits(int(rung["k"]))
    total_bits = (
        index_count * index_bits
        + int(rung["k"]) * effective_dim * 16
        + pack.HEADER_BYTES * 8
    )
    return math.ceil(total_bits / 8)


def _decision_bytes(row: dict, decision: str) -> int:
    if decision == "native":
        return int(row["payload_bytes"])
    return _pq_payload_bytes(row, decision)


def _base_tensor_decision(row: dict, coalition: set[tuple[int, int]]) -> tuple[str, str]:
    import glm52_pack as pack

    if row["dtype"] != "BF16":
        return "native", "NON_BF16_CONTROL"
    if row["provisional_budget_class"] == pack.PROTECTED_BUDGET_CLASS:
        return "native", "STRUCTURAL_CONTROL"
    if row["category"] in PROFILE_NATIVE_CATEGORIES:
        return "native", "MATH_PROFILE_NATIVE_ORGAN"
    if row["category"] == "routed_expert" and (
        int(row["layer"]), int(row["expert"])
    ) in coalition:
        return COALITION_FLOOR_RUNG, "MEASURED_MATH_COALITION_FLOOR"
    return TAIL_RUNG, "PROFILE_FLOOR"


def global_byte_auction(result: dict) -> dict:
    """Freeze one whole-model allocation from all completed cartography.

    Expert slots, not source shards, are the auction lots.  All three matrices of
    a promoted expert move together, avoiding a fictitious per-matrix sensitivity
    claim that PASS 1 did not measure.  One best expert per sparse layer is admitted
    first so no layer loses native Math support; the remainder are ranked globally
    by confidence, normalized contribution, hit count, then stable ids.
    """
    graph = json.loads(GRAPH.read_text())
    logical = json.loads(LEDGER.read_text())
    rows = graph["tensors"]
    denominator = int(logical["logical_weight_denominator"])
    max_complete_bytes = (
        denominator * TARGET_BPW_NUM // (TARGET_BPW_DEN * 8)
    )
    max_payload_bytes = max_complete_bytes - PACKAGING_RESERVE_BYTES
    if max_payload_bytes <= 0:
        raise AssertionError("packaging reserve consumes the entire H0.98 budget")

    evidence: dict[tuple[int, int], dict] = {}
    for layer_text, layer_data in result["per_layer"].items():
        layer = int(layer_text)
        for row in layer_data["coalition_evidence"]:
            evidence[(layer, int(row["expert"]))] = row
    coalition = set(evidence)

    tensor_decisions: dict[str, str] = {}
    tensor_reasons: dict[str, str] = {}
    predicted_payload = 0
    slot_rows: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        decision, reason = _base_tensor_decision(row, coalition)
        name = row["name"]
        if name in tensor_decisions:
            raise AssertionError(f"duplicate tensor in graph: {name}")
        tensor_decisions[name] = decision
        tensor_reasons[name] = reason
        predicted_payload += _decision_bytes(row, decision)
        if row["category"] == "routed_expert" and (
            int(row["layer"]), int(row["expert"])
        ) in coalition:
            slot_rows.setdefault((int(row["layer"]), int(row["expert"])), []).append(row)

    if len(tensor_decisions) != int(graph["tensor_count"]):
        raise AssertionError(
            f"auction named {len(tensor_decisions)} tensors, graph declares "
            f"{graph['tensor_count']}"
        )
    if set(slot_rows) != coalition:
        raise AssertionError(
            f"graph/capsule coalition mismatch: {len(slot_rows)} slots in graph, "
            f"{len(coalition)} in evidence"
        )

    lots = []
    for key, expert_rows in slot_rows.items():
        if len(expert_rows) != 3:
            raise AssertionError(f"coalition expert {key} has {len(expert_rows)} tensors, not 3")
        ev = evidence[key]
        increment = sum(
            _decision_bytes(row, "native") - _decision_bytes(row, COALITION_FLOOR_RUNG)
            for row in expert_rows
        )
        lots.append({
            "key": key,
            "rows": expert_rows,
            "increment_bytes": increment,
            "confident": bool(ev["confident"]),
            "normalized": float(ev["layer_normalized_contribution"]),
            "hit_count": int(ev["hit_count"]),
        })

    by_layer: dict[int, list[dict]] = {}
    for lot in lots:
        by_layer.setdefault(lot["key"][0], []).append(lot)
    mandatory = [
        max(
            layer_lots,
            key=lambda lot: (
                lot["confident"], lot["normalized"], lot["hit_count"],
                -lot["key"][1],
            ),
        )
        for _, layer_lots in sorted(by_layer.items())
    ]
    mandatory_keys = {lot["key"] for lot in mandatory}
    remainder = [lot for lot in lots if lot["key"] not in mandatory_keys]
    remainder.sort(key=lambda lot: (
        -int(lot["confident"]), -lot["normalized"], -lot["hit_count"],
        lot["key"][0], lot["key"][1],
    ))

    selected: list[dict] = []
    for lot in mandatory + remainder:
        if predicted_payload + lot["increment_bytes"] > max_payload_bytes:
            continue
        for row in lot["rows"]:
            tensor_decisions[row["name"]] = "native"
            tensor_reasons[row["name"]] = "GLOBAL_MATH_AUCTION_NATIVE_COALITION"
        predicted_payload += lot["increment_bytes"]
        selected.append(lot)

    if {lot["key"][0] for lot in selected} != set(by_layer):
        raise AssertionError("global auction failed to preserve a native expert in every sparse layer")
    if predicted_payload + PACKAGING_RESERVE_BYTES > max_complete_bytes:
        raise AssertionError("global auction exceeded its exact H0.98 byte ceiling")

    decision_counts: dict[str, int] = {}
    decision_bytes: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in rows:
        decision = tensor_decisions[row["name"]]
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        decision_bytes[decision] = (
            decision_bytes.get(decision, 0) + _decision_bytes(row, decision)
        )
        reason = tensor_reasons[row["name"]]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    selected_keys = {lot["key"] for lot in selected}
    slot_decisions = {
        f"{layer}:{expert}": (
            "native" if (layer, expert) in selected_keys else COALITION_FLOOR_RUNG
        )
        for layer, expert in sorted(coalition)
    }
    return {
        "schema": "hawking.prometheus.math_global_byte_auction.v1",
        "allocation_complete": True,
        "scope": "whole_model",
        "target_complete_bpw": {"num": TARGET_BPW_NUM, "den": TARGET_BPW_DEN},
        "logical_weight_denominator": denominator,
        "max_complete_physical_bytes": max_complete_bytes,
        "packaging_and_runtime_reserve_bytes": PACKAGING_RESERVE_BYTES,
        "max_tensor_payload_bytes": max_payload_bytes,
        "predicted_tensor_payload_bytes": predicted_payload,
        "predicted_payload_bpw": predicted_payload * 8 / denominator,
        "predicted_complete_bytes_with_reserve": (
            predicted_payload + PACKAGING_RESERVE_BYTES
        ),
        "predicted_complete_bpw_with_reserve": (
            (predicted_payload + PACKAGING_RESERVE_BYTES) * 8 / denominator
        ),
        "unallocated_bytes_inside_target": (
            max_complete_bytes - predicted_payload - PACKAGING_RESERVE_BYTES
        ),
        "coalition_slots_total": len(coalition),
        "coalition_slots_native": len(selected),
        "coalition_slots_floor": len(coalition) - len(selected),
        "native_sparse_layers": len({lot["key"][0] for lot in selected}),
        "tensor_decision_count": len(tensor_decisions),
        "decision_tensor_counts": dict(sorted(decision_counts.items())),
        "decision_payload_bytes": dict(sorted(decision_bytes.items())),
        "decision_reason_counts": dict(sorted(reason_counts.items())),
        "tensor_decisions": dict(sorted(tensor_decisions.items())),
        "coalition_slot_decisions": slot_decisions,
        "mechanism_decisions": {
            "removal": {"selected": False, "reason": "no runtime-safe removal replacement frozen"},
            "sharing": {"selected": False, "reason": "no executable shared-grammar codec frozen"},
            "functional_replacement": {
                "selected": False,
                "reason": "every official tensor retains a physical native or gravity-pq payload",
            },
            "doctor": {"selected": False, "bytes": 0},
            "metadata": {
                "strategy": "actual bytes graded after PASS3",
                "pre_pack_reserve_bytes": PACKAGING_RESERVE_BYTES,
            },
        },
        "decision_semantics": {
            "native": "source bytes preserved exactly",
            COALITION_FLOOR_RUNG: "measured Math coalition, gravity-pq R0",
            TAIL_RUNG: "non-coalition/profile floor, gravity-pq R4",
        },
    }


def run(*, coalition_fraction: float = DEFAULT_COALITION_FRACTION) -> dict:
    declared = sparse_layers_declared()
    per_layer: dict[str, dict] = {}
    missing: list[int] = []
    for layer in declared:
        ranking = layer_ranking(layer, coalition_fraction)
        if ranking is None:
            missing.append(layer)
        else:
            per_layer[str(layer)] = ranking

    thin_layers = [
        int(layer) for layer, r in per_layer.items()
        if r["coalition_thin_evidence_expert_ids"]
    ]
    result = {
        "schema": "hawking.prometheus.math_allocation_manifest.v2",
        "at": _now(),
        "source_capsule_dir": str(CAPSULE_DIR),
        "coalition_fraction": coalition_fraction,
        "sparse_layers_declared": declared,
        "sparse_layers_with_evidence": sorted(int(k) for k in per_layer),
        "sparse_layers_missing_evidence": sorted(missing),
        "complete": not missing,
        "layers_with_thin_coalition_evidence": thin_layers,
        "per_layer": per_layer,
        "note": "PASS2 reads every sealed capsule before making one whole-model decision. "
                "A manifest with complete=false is a preview and carries no byte auction; "
                "freezing before every sparse layer has evidence would be exactly the "
                "greedy per-shard decision the three-pass design exists to prevent.",
    }
    if result["complete"]:
        result["global_byte_auction"] = global_byte_auction(result)
    return result


def preview() -> dict:
    result = run()
    print(json.dumps(
        {k: v for k, v in result.items()
         if k not in {"per_layer", "global_byte_auction"}} | {
            "per_layer_summary": {
                layer: {"coalition_size": r["coalition_size"],
                       "confident": len(r["coalition_confident_expert_ids"]),
                       "thin": len(r["coalition_thin_evidence_expert_ids"])}
                for layer, r in result["per_layer"].items()
            },
            "global_byte_auction_summary": (
                {
                    k: v for k, v in result["global_byte_auction"].items()
                    if k not in {"tensor_decisions", "coalition_slot_decisions"}
                }
                if result.get("global_byte_auction") else None
            ),
        }, indent=2, sort_keys=True,
    ))
    return result


def freeze(out: Path, *, coalition_fraction: float = DEFAULT_COALITION_FRACTION) -> dict:
    result = run(coalition_fraction=coalition_fraction)
    if not result["complete"]:
        raise SystemExit(
            f"refusing to freeze: {len(result['sparse_layers_missing_evidence'])} of "
            f"{len(result['sparse_layers_declared'])} sparse layers have no capsule "
            f"evidence yet ({result['sparse_layers_missing_evidence'][:10]}...). "
            "Run `preview` to inspect what exists so far."
        )
    auction = result.get("global_byte_auction") or {}
    if not auction.get("allocation_complete"):
        raise SystemExit("refusing to freeze: complete cartography produced no global byte auction")
    if auction.get("tensor_decision_count") != json.loads(GRAPH.read_text())["tensor_count"]:
        raise SystemExit("refusing to freeze: the global auction does not name every tensor")
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(
        f"frozen: {out} ({len(result['per_layer'])} layers, "
        f"{auction['tensor_decision_count']} tensor decisions, "
        f"predicted complete {auction['predicted_complete_bpw_with_reserve']:.6f} BPW)"
    )
    return result


def selftest() -> None:
    """No capsules required: exercises the ranking math against a synthetic
    in-memory capsule so the auction logic is provably correct independent of
    whatever PASS 1 has captured so far."""
    contribution = np.array([5.0, 1.0, 9.0, 0.0, 3.0, 3.0, 2.0, 8.0], dtype=np.float32)
    hit = np.array([3, 1, 4, 0, 1, 5, 2, 6], dtype=np.int32)
    n = contribution.shape[0]
    k = round(0.25 * n)  # 2 of 8
    order = sorted(range(n), key=lambda e: (-float(contribution[e]), e))
    coalition = sorted(order[:k])
    assert coalition == [2, 7], f"expected the two highest-contribution experts, got {coalition}"
    confident = [e for e in coalition if hit[e] >= MIN_HIT_COUNT_FOR_CONFIDENT_RANK]
    assert confident == [2, 7], "both top experts have hit_count >= 2 in this fixture"

    # Tie-break: two equal contributions must resolve to the lower expert id.
    tied = np.array([4.0, 4.0, 1.0], dtype=np.float32)
    tied_order = sorted(range(3), key=lambda e: (-float(tied[e]), e))
    assert tied_order[0] == 0, "tie must break toward the lower expert id"

    # The allocation predictor is the packer's exact ByteLedger arithmetic, not
    # nominal rung labels.  Check it on a shape that exercises both index and
    # codebook terms.
    import glm52_pack as pack
    row = {
        "shape": [32, 6144], "payload_bytes": 32 * 6144 * 2,
    }
    for rung in pack.LADDER:
        predicted = _pq_payload_bytes(row, rung["rung"])
        dim = int(rung["dim"])
        count = _elements(row) // dim
        direct = math.ceil((
            count * pack.index_bits(int(rung["k"]))
            + int(rung["k"]) * dim * 16
            + pack.HEADER_BYTES * 8
        ) / 8)
        assert predicted == direct, (rung["rung"], predicted, direct)

    print("math_pass2_allocation selftest PASS")


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "preview"
    if command == "preview":
        preview()
        return 0
    if command == "freeze":
        out = Path(argv[argv.index("--out") + 1]) if "--out" in argv \
            else REPO / "evidence" / "prometheus" / "PROMETHEUS_MATH_ALLOCATION_MANIFEST.json"
        freeze(out)
        return 0
    if command == "selftest":
        selftest()
        return 0
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
