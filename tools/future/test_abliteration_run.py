"""Tests for abliteration as a run-for-real candidate transformation generator.

Protocol tests do not load the specimen and do not need torch. The live
receipt test reads receipts/future/ABLITERATION_RUN.json produced by a real
MPS run; absence of that file is a failure, not a skip.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.future import abliteration as ab
from tools.future import abliteration_run as ar
from tools.future import tabula as tb
from tools.future._common import HARDWARE_FIELDS, RECEIPTS


def _acts(*, n_layers: int = 8, n_h: int = 4, n_c: int = 4, d: int = 16, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    harmful = rng.normal(size=(n_layers, n_h, d))
    harmless = rng.normal(size=(n_layers, n_c, d))
    # Layer 2 carries a strong, specific contrast. Layer 6 is in the last 20%.
    direction = np.zeros(d)
    direction[0] = 1.0
    harmful[2] += 3.0 * direction
    harmless[2] -= 0.2 * direction
    harmful[1] += 0.4 * direction
    harmful[6] += 5.0 * direction
    return harmful, harmless


def _scored(**kwargs):
    h, c = _acts(**kwargs)
    raw = ar.generate_directions(h, c)
    return ar.attach_objectives(raw, h, c), h, c


def test_claim_boundary_still_refuses_uncensoring_switch():
    with pytest.raises(ab.ClaimBoundaryError, match="refusals_removed|uncensoring"):
        ar.admit_candidate_transformation({"status": "abliterated"})
    with pytest.raises(ab.ClaimBoundaryError):
        ar.admit_candidate_transformation(
            {"status": "CANDIDATE", "refusals_removed": True, "gpu_authority": False}
        )
    with pytest.raises(ab.ClaimBoundaryError, match="UNMEASURED"):
        ar.compose_receipt(
            {
                "specimen": ar.SPECIMEN,
                "took_gpu_lease": True,
                "candidates": [],
                "selection": {"selected_id": None},
                "failspy": {"selected_id": "x", "is_negative_control": True},
                "selector_comparison": {"disagree": True},
                "timings": {"direction_computation": 0.01},
                "end_to_end_s": 1.0,
                "behavioural_outcome": "refusals_removed",
                "weights_modified": False,
                "projection": {"lake_written": False},
            }
        )
    ok = ar.admit_candidate_transformation(
        {
            "status": "CANDIDATE",
            "selected_id": "L2_p-1",
            "projection": ar.PROJECTION_DEFAULT,
            "gpu_authority": False,
            "behavioural_outcome": "UNMEASURED",
        }
    )
    assert ok["admitted"] is True
    assert ok["not_as"] == "uncensoring_switch"
    assert ok["behavioural_outcome"] == "UNMEASURED"


def test_failspy_single_score_is_a_control_and_disagrees_with_the_gate():
    scored, _, _ = _scored()
    with pytest.raises(ab.UnderdeterminedSelection, match="refusal suppression alone"):
        ab.select_by_refusal_suppression_alone(scored)
    failspy = ar.failspy_find_best_refusal_dir(scored)
    multi = ar.select_multi_objective(scored)
    assert failspy["is_method"] is False
    assert failspy["is_negative_control"] is True
    assert failspy["selected_id"]
    assert multi["selected_id"]
    assert multi["ranking_by_refusal_suppression"] is False
    cmp = ar.compare_selectors(multi, failspy)
    # Layer 6 has the largest harmful alignment but is source-pruned; FailSpy
    # still picks it. The multi-objective gate cannot.
    assert failspy["selected_id"] != multi["selected_id"]
    assert cmp["disagree"] is True
    assert failspy["selected_id"] not in multi["survivors"]
    assert "L6_p-1" in {c["id"] for c in scored}


def test_select_requires_three_objectives_and_names_refusals():
    scored, _, _ = _scored()
    public = [ar.candidate_public_row(c, selected_id=None) for c in scored]
    assert all("completion" in c["objectives"] for c in public)
    assert all("harmless" in c["objectives"] for c in public)
    assert all("loss" in c["objectives"] for c in public)
    admitted = [c for c in public if c["admitted"]]
    refused = [c for c in public if not c["admitted"]]
    assert admitted
    assert refused
    assert any(c["pruned_as_source"] for c in refused)
    tail = next(c for c in public if c["source_layer"] == 6)
    assert tail["admitted"] is False
    assert "source_layer_pruned" in (tail["refuse_reason"] or "")
    with pytest.raises(ab.UnderdeterminedSelection):
        ar.select_multi_objective(
            [{"id": "x", "source_layer": 1, "n_layers": 8, "evals": {"completion": {"gate": "PASS"}}}]
        )


def test_empty_contrast_is_a_run_refusal_not_a_direction():
    with pytest.raises(ar.ContrastEmpty, match="empty"):
        ar.generate_directions(
            np.zeros((4, 0, 8)),
            np.zeros((4, 3, 8)),
        )


def test_biprojection_is_not_frobenius_and_does_not_write_the_lake():
    rng = np.random.default_rng(2)
    d, n = 12, 16
    W = rng.normal(size=(d, n))
    v = rng.normal(size=d)
    v = v / np.linalg.norm(v)
    W_row, geom, recipe = ar.biproject_matrix(W, v)
    W_frob, _, _ = tb.project(W, v, norm_preserve=True, store_component=False)
    assert geom["row_norm_and_frobenius_are_distinct"] is True
    assert float(np.linalg.norm(W_row - W_frob, ord="fro")) > 1e-9
    assert recipe["v_sha256"]
    dest = [2, 3]
    weights = {
        "model.layers.2.self_attn.o_proj.weight": W,
        "model.layers.2.mlp.down_proj.weight": W,
        "model.layers.3.self_attn.o_proj.weight": W,
        "model.layers.3.mlp.down_proj.weight": W,
    }
    proj = ar.project_destination_weights(weights, v, dest)
    assert proj["lake_written"] is False
    assert proj["in_memory"] is True
    assert proj["n_applied"] == 4


def test_repatriation_map_names_every_step_with_a_reason():
    timings = {step: float(i + 1) * 0.01 for i, step in enumerate(ar.STEP_ORDER)}
    rows = ar.repatriation_map(timings)
    by = {r["step"]: r for r in rows}
    assert set(by) >= set(ar.STEP_ORDER)
    for r in rows:
        assert r["reason"]
        assert isinstance(r["repatriates_to_rust_metal"], bool)
        assert r["wall_s"] is None or r["wall_s"] >= 0
    assert by["activation_capture"]["repatriates_to_rust_metal"] is True
    assert by["weight_projection"]["repatriates_to_rust_metal"] is True
    assert by["direction_computation"]["repatriates_to_rust_metal"] is True
    assert by["selection"]["repatriates_to_rust_metal"] is False
    assert by["failspy_negative_control"]["repatriates_to_rust_metal"] is False
    assert sum(r["is_dominant"] for r in rows) == 1
    top = next(r for r in rows if r["is_dominant"])
    assert top["step"] == ar.STEP_ORDER[-1]  # largest assigned timing


def test_throughput_and_dominant_term():
    t = ar.throughput(30.0)
    assert t["candidate_transformations_per_hour"] == pytest.approx(120.0)
    with pytest.raises(ar.RunBlocked):
        ar.throughput(0)
    dom = ar.dominant_term({"a": 1.0, "b": 4.0, "c": 0.5})
    assert dom["step"] == "b"
    assert dom["wall_s"] == pytest.approx(4.0)


def test_compose_receipt_marks_behaviour_unmeasured_and_avoids_hardware_fields():
    scored, _, _ = _scored()
    multi = ar.select_multi_objective(scored)
    failspy = ar.failspy_find_best_refusal_dir(scored)
    run = {
        "specimen": ar.SPECIMEN,
        "took_gpu_lease": True,
        "candidates": [
            {k: v for k, v in c.items() if k not in {"direction", "direction_projected", "harmful_mean", "harmless_mean"}}
            for c in scored
        ],
        "selection": multi,
        "failspy": failspy,
        "selector_comparison": ar.compare_selectors(multi, failspy),
        "timings": {
            "activation_capture": 0.4,
            "direction_computation": 0.01,
            "weight_projection": 0.05,
            "model_load": 2.0,
        },
        "end_to_end_s": 3.0,
        "before_end_to_end_s": 6.0,
        "behavioural_outcome": "UNMEASURED",
        "weights_modified": False,
        "projection": {"lake_written": False, "in_memory": True, "n_applied": 2},
        "admitted_candidate": ar.admit_candidate_transformation(
            {
                "status": "CANDIDATE",
                "selected_id": multi["selected_id"],
                "projection": ar.PROJECTION_DEFAULT,
                "gpu_authority": False,
                "behavioural_outcome": "UNMEASURED",
            }
        ),
        "cuts_applied": ["batched capture"],
        "what_cutting_the_rest_requires": "keep the model resident",
        "metal": {"mps_device_ok": True, "chipset": "Apple M3 Ultra"},
    }
    doc = ar.compose_receipt(run)
    assert doc["schema"] == ar.SCHEMA
    assert doc["behavioural_outcome"] == "UNMEASURED"
    assert doc["weights_modified"] is False
    assert doc["gpu_authority"] is False
    assert doc["took_gpu_lease"] is True
    assert doc["n_direction_candidates"] == 8
    assert doc["speed"]["after"] == pytest.approx(1200.0)
    assert doc["speed"]["before"] == pytest.approx(600.0)
    assert doc["speed"]["dominant_term"]["step"] == "model_load"
    assert any(r["step"] == "activation_capture" for r in doc["repatriation"])
    assert "does not guarantee" in " ".join(doc["negative_findings"]).lower()
    blob = json.dumps(doc).lower()
    for phrase in ("refusals were removed", "the model is abliterated", "the model is uncensored"):
        assert phrase not in blob
    for key in HARDWARE_FIELDS:
        assert not isinstance(doc.get(key), (int, float))
        assert not isinstance(doc["bench"].get(key), (int, float))
    assert "tps" not in doc["timings"]


def test_gpu_lane_lease_round_trips(tmp_path: Path):
    a = tmp_path / "gpu.lock"
    b = tmp_path / "hcli.lock"
    lease = ar.GpuLaneLease([a, b])
    info = lease.acquire(timeout_s=2)
    try:
        assert info["held"] is True
        assert info["took_gpu_lease"] is True
        assert info["widens_hcli_authority"] is False
        other = ar.GpuLaneLease([a])
        with pytest.raises(ar.RunBlocked, match="busy"):
            other.acquire(timeout_s=0.2)
    finally:
        lease.release()
    other.acquire(timeout_s=2)
    other.release()


def test_plan_unverified_specimen_is_still_refused():
    with pytest.raises(ab.PlanRefusal, match="whole-tree verified"):
        ab.plan(
            "not-verified@dead",
            verification={"results": [{"specimen": "not-verified@dead", "status": "PARTIAL"}]},
            scars=[],
        )


def test_metal_probe_does_not_invent_a_device():
    probe = ar.metal_probe()
    assert "metal_device_seen" in probe
    assert "mps_available" in probe
    if probe.get("gpu_cores_reported") is not None:
        assert isinstance(probe["gpu_cores_reported"], int)
        assert probe["gpu_cores_reported"] > 0


def test_live_receipt_is_a_real_run_on_the_verified_specimen():
    path = RECEIPTS / ar.RECEIPT
    assert path.is_file(), f"missing live receipt {path}; run tools/future/abliteration_run.py --run"
    doc = json.loads(path.read_text())
    assert doc["schema"] == ar.SCHEMA
    assert doc["specimen"] == ar.SPECIMEN
    assert doc["took_gpu_lease"] is True
    assert doc["weights_modified"] is False
    assert doc["lake_untouched"] is True
    assert doc["behavioural_outcome"] == "UNMEASURED"
    assert doc["gpu_authority"] is False
    assert doc["verification"]["whole_tree_verified"] is True
    assert doc["n_direction_candidates"] >= 8
    assert doc["n_layers"] == 28
    cands = doc["candidates"]
    assert len(cands) == doc["n_direction_candidates"]
    for c in cands:
        for name in ("completion", "harmless", "loss"):
            assert name in c["objectives"]
            assert c["objectives"][name]["gate"] in {"PASS", "FAIL"}
            assert "score" in c["objectives"][name]
        assert c["admitted"] is True or c["refuse_reason"]
    assert isinstance(doc["admitted_ids"], list)
    assert isinstance(doc["refused"], list)
    assert doc["failspy"]["is_negative_control"] is True
    assert "disagree" in doc["selector_comparison"]
    assert doc["end_to_end_s"] > 0
    assert doc["speed"]["after"] > 0
    assert doc["speed"]["before"] > 0
    assert doc["speed"]["dominant_term"]["step"]
    steps = {r["step"]: r for r in doc["repatriation"]}
    for needed in (
        "activation_capture",
        "direction_computation",
        "weight_projection",
        "multi_objective_eval",
        "selection",
    ):
        assert needed in steps
        assert steps[needed]["reason"]
    assert doc["projection"]["lake_written"] is False
    blob = json.dumps(doc).lower()
    assert "does not guarantee" in blob or "not guarantee" in blob
    for phrase in ("refusals were removed", "the model is abliterated"):
        assert phrase not in blob
    for key in HARDWARE_FIELDS:
        assert not isinstance(doc.get(key), (int, float))
    assert doc["port_not_performed"] is True
    assert doc["generation_eval"]["ran"] is False
    assert doc["bench"]["measurement_state"] == "SELF_MEASURED"
    # A transformation was generated, or the empty selection is named as empty.
    if doc["selection"].get("selected_id"):
        assert doc["admitted_candidate"]["admitted"] is True
        assert doc["admitted_candidate"]["as"] == "candidate_transformation"
        assert doc["projection"]["n_applied"] >= 1
    else:
        assert doc["selection"].get("empty") is True
