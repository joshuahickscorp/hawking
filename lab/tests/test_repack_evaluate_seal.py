"""EVALUATE is pure-CPU over a score index; selection matches SEAL identity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.operators import repack_score_index as si


def _synthetic_organs(n_experts: int = 4, n_layers: int = 2) -> list[dict]:
    organs = []
    for layer in range(n_layers):
        for expert in range(n_experts):
            for i, component in enumerate(si.COMPONENTS):
                # High surplus on full triples for experts 0..n-2; partial low surplus
                # on last expert so expert-atomic demotes it when packed.
                surplus = 0.4 - 0.01 * expert - 0.001 * i
                if expert == n_experts - 1 and component == "down_proj":
                    continue  # incomplete triple source
                delta = 10_000 + expert * 100 + i
                organs.append(
                    {
                        "tensor_name": f"model.layers.{layer}.mlp.experts.{expert}.{component}.weight",
                        "layer": layer,
                        "expert": expert,
                        "component": component,
                        "shape": [128, 64],
                        "elements": 128 * 64,
                        "source_shard": "x.safetensors",
                        "source_value_sha256": "a" * 64,
                        "family": "activation_weighted_svd_low_rank_q",
                        "budget_label": "r64_b3",
                        "rank": 64,
                        "bits": 3,
                        "component_bpw": 1.2,
                        "under_ceiling": True,
                        "weight_cosine": 0.9,
                        "weight_relative_l2": 0.1,
                        "output_cosine": 0.95 - 0.01 * expert,
                        "null_baseline": 0.5,
                        "surplus_over_null": surplus,
                        "beats_null": True,
                        "distribution_local_only": False,
                        "n_fit_tokens": 48,
                        "n_hold_tokens": 16,
                        "physical_payload_bytes": 50_000 + delta,
                        "baseline_payload_bytes": 40_000,
                        "payload_delta_bytes": 10_000 + delta - 40_000 + 40_000,  # positive-ish
                        "physical_payload_sha256": f"{layer:02x}{expert:02x}{i:02x}" + "b" * 58,
                        "payload_path": f"/tmp/fake/{layer}_{expert}_{component}.hgravs01",
                        "selection_metric": "surplus_over_null",
                    }
                )
    # Fix payload_delta to be small positive for packing control.
    for o in organs:
        o["payload_delta_bytes"] = 5_000
        o["physical_payload_bytes"] = 45_000
        o["baseline_payload_bytes"] = 40_000
    return organs


def test_density_sort_is_surplus_first() -> None:
    rows = [
        {"surplus_over_null": 0.1, "weight_cosine": 0.99, "component_bpw": 0.5, "layer": 0, "expert": 0, "component": "a"},
        {"surplus_over_null": 0.5, "weight_cosine": 0.1, "component_bpw": 1.4, "layer": 0, "expert": 1, "component": "b"},
    ]
    ordered = sorted(rows, key=si.density_sort_key)
    assert ordered[0]["surplus_over_null"] == 0.5


def test_evaluate_is_milliseconds_and_expert_atomic(tmp_path: Path) -> None:
    organs = _synthetic_organs(n_experts=3, n_layers=3)
    # base payload huge so budget is the only constraint; use generous budget.
    base = 100_000_000
    elems = 800_000_000
    doc = si.build_index_document(
        organs,
        meta={
            "base_payload_bytes": base,
            "source_weight_elements": elems,
            "manifest_reserve_bytes": 1_000_000,
        },
    )
    path = tmp_path / "index.json"
    si.save_score_index(path, doc)
    idx = si.load_score_index(path)
    assert idx.n_organs == len(organs)
    assert idx.meta.get("load_ms", 999) < 500  # tiny file

    r = idx.evaluate(budget_bpw=1.5, base_payload=base, elements=elems, manifest_reserve=1_000_000)
    assert r["status"] == "EVALUATED"
    assert r["timing_ms"]["total"] < 50.0
    # Incomplete triples demoted: last expert missing down_proj per layer.
    names = {o["tensor_name"] for o in r["selected_organs"]}
    for layer in range(3):
        for component in ("gate_proj", "up_proj"):
            bad = f"model.layers.{layer}.mlp.experts.2.{component}.weight"
            assert bad not in names
    assert r["expert_atomic"]["experts_demoted_for_incomplete_triple"] == 3
    assert "predicted_chain_survival" in r
    assert "GRAVITY_DENSITY_FRONTIER" in r
    assert "ceiling_verdict" in r


def test_lower_budget_selects_subset() -> None:
    organs = _synthetic_organs(n_experts=4, n_layers=4)
    for i, o in enumerate(organs):
        o["payload_delta_bytes"] = 50_000 + i * 1000  # large deltas force deferral
        o["physical_payload_bytes"] = 90_000 + i * 1000
    base = 100_000_000
    elems = 800_000_000
    doc = si.build_index_document(
        organs,
        meta={
            "base_payload_bytes": base,
            "source_weight_elements": elems,
            "manifest_reserve_bytes": 0,
        },
    )
    idx = si.ScoreIndex(records=doc["organs"], meta=doc)
    hi = idx.evaluate(budget_bpw=1.5, base_payload=base, elements=elems, manifest_reserve=0)
    lo = idx.evaluate(budget_bpw=1.01, base_payload=base, elements=elems, manifest_reserve=0)
    # At base bpw already ~1.0, so low headroom means lo selects fewer or equal.
    assert lo["n_selected"] <= hi["n_selected"]
    assert lo["complete_physical_bpw"] <= 1.01 + 1e-6


def test_evaluate_seal_identity_compare() -> None:
    organs = _synthetic_organs(n_experts=2, n_layers=1)
    # Make all complete triples.
    base = 50_000_000
    elems = 400_000_000
    doc = si.build_index_document(
        organs,
        meta={
            "base_payload_bytes": base,
            "source_weight_elements": elems,
            "manifest_reserve_bytes": 0,
        },
    )
    idx = si.ScoreIndex(records=doc["organs"], meta=doc)
    ev = idx.evaluate(budget_bpw=1.5, base_payload=base, elements=elems, manifest_reserve=0)
    # Simulate sealed selection identity.
    sealed = {
        "budget_bpw": None,
        "complete_physical_bpw": ev["complete_physical_bpw"],
        "n_selected": ev["n_selected"],
        "organ_set_sha256": ev["organ_set_sha256"],
        "organs": [
            {
                "tensor_name": r["tensor_name"],
                "budget_label": r["budget_label"],
                "rank": r["rank"],
                "bits": r["bits"],
                "physical_payload_sha256": r["physical_payload_sha256"],
                "physical_payload_bytes": r["physical_payload_bytes"],
            }
            for r in sorted(ev["selected_organs"], key=lambda x: x["tensor_name"])
        ],
    }
    # Recompute organ_set_sha the sealed way (sorted by name).
    sealed_id = {
        "n_selected": len(sealed["organs"]),
        "organ_set_sha256": sealed["organ_set_sha256"],
        "organs": sealed["organs"],
        "complete_physical_bpw": sealed["complete_physical_bpw"],
        "budget_bpw": None,
    }
    proof = si.compare_evaluate_to_seal(ev, sealed_id)
    assert proof["match"] is True
    assert proof["n_field_mismatches"] == 0


def test_surplus_gate_filters() -> None:
    organs = _synthetic_organs(n_experts=1, n_layers=1)
    organs[0]["beats_null"] = False
    organs[0]["surplus_over_null"] = -0.5
    admitted = si.filter_admitted(organs, min_surplus=0.0)
    assert all(r["beats_null"] for r in admitted)
    assert organs[0]["tensor_name"] not in {r["tensor_name"] for r in admitted}
