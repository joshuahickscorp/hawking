"""Hardware Doctor: emit guards, ranking, scar refusal, sealed receipt."""
from __future__ import annotations

import json

import pytest

from tools.future import hardware_doctor as hd
from tools.future._common import HARDWARE_FIELDS, RECEIPTS


def _valid(**over):
    base = {
        "id": "HD-TEST",
        "axis": "tiling",
        "hypothesis": (
            "within-organ tensor-parallel tiles that keep weight shards resident "
            "reduce per-token transport versus restreaming the weight body"
        ),
        "target_organ": "mlp_gate_up_down",
        "predicted_effect": {
            "direction": "reduce_per_token_transport",
            "magnitude_class": "UNKNOWN",
        },
        "uncertainty": "device unselected; link parameters are scenario not measurement",
        "cheapest_simulator": "transport_link_simulator",
        "falsifier": (
            "resident-tile transport class is not smaller than weight-body restream "
            "on the organ-map link simulator"
        ),
        "expected_removed_cost": "per-token weight-body transfer",
        "prerequisite": "organ map transport_link_simulator (exists)",
        "refutation_probability": "HIGH",
    }
    base.update(over)
    return base


def test_build_emits_sealed_receipt():
    out = hd.build()
    assert out.parent == RECEIPTS
    assert out.name == "HARDWARE_DOCTOR.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == "hawking.future.hardware_doctor.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert "experiment_queue" in doc
    assert doc["entries"]
    assert doc["n_proposals"] == 12
    assert set(doc["axes_covered"]) == set(hd.AXES)
    for entry in doc["experiment_queue"]:
        assert entry.get("cheapest_falsifier")
        assert entry["predicted_effect"]["direction"]
        assert entry["predicted_effect"]["magnitude_class"] in hd.MAGNITUDE_CLASSES
        assert entry["cheapest_simulator"]["fidelity"] in hd.SIMULATORS
        assert entry["evidence_class"] == "STATIC_ONLY"
        assert entry["bench_state"] == "UNKNOWN"


def test_selftest_aliases_build():
    a = hd.selftest()
    b = hd.build()
    assert a.name == b.name == "HARDWARE_DOCTOR.json"


def test_organs_are_from_real_maps_not_imagined():
    bundle = hd.load_organs()
    names = {r["organ"] for r in bundle["organs"]}
    assert "mlp_gate_up_down" in names
    assert "expert_bank" in names
    assert "deltanet_state_and_input_projection" in names
    assert "deltanet_persistent_state" in names
    assert "imaginary_systolic_array" not in names
    assert bundle["maps"]["flash-next"]["physical_board_present"] is False
    assert bundle["maps"]["qwen27"]["provider_capabilities"]["cycle_simulation"] is False
    for rec in hd.catalog():
        assert rec["target_organ"] in names, rec["id"]


def test_every_catalog_proposal_emits():
    organs = hd.load_organs()
    scars = hd.load_scars()["scars"]
    for raw in hd.catalog():
        out = hd.emit(raw, organs=organs, scars=scars)
        for field in hd.REQUIRED_FIELDS:
            assert hd._present(out.get(field) if field != "cheapest_simulator" else out["cheapest_simulator"])
        assert out["cheapest_falsifier"] == out["falsifier"]


def test_ranking_prefers_cheap_high_refutation():
    organs = hd.load_organs()
    scars = hd.load_scars()["scars"]
    cheap_high = hd.emit(
        _valid(
            id="A-cheap-high",
            cheapest_simulator="static_hwir",
            refutation_probability="HIGH",
            target_organ="deltanet_state_and_input_projection",
            hypothesis="resident sequence-lifetime state stays off the transport link",
            falsifier="HWIR buffer lifetime already allows per-token state shipping",
        ),
        organs=organs,
        scars=scars,
    )
    expensive_high = hd.emit(
        _valid(
            id="B-rtl-high",
            cheapest_simulator="rtl_resource_estimate",
            refutation_probability="HIGH",
            axis="arithmetic_width",
            hypothesis="packed stored-width MAC reduces LUT glue versus unpack-to-wide",
            falsifier="resource envelope does not shrink at equal initiation-interval class",
        ),
        organs=organs,
        scars=scars,
    )
    cheap_low = hd.emit(
        _valid(
            id="C-cheap-low",
            cheapest_simulator="static_hwir",
            refutation_probability="LOW",
            target_organ="command_buffer_graph",
            axis="dfx_boundary",
            hypothesis="a DFX cut around P1 organs keeps P0 GEMV resident across a P1 swap",
            falsifier="the cut crosses a token-path HWIR dependency",
        ),
        organs=organs,
        scars=scars,
    )
    ranked = hd.rank_queue([expensive_high, cheap_low, cheap_high])
    assert [r["id"] for r in ranked] == ["A-cheap-high", "C-cheap-low", "B-rtl-high"]
    assert ranked[0]["rank"] == 1
    # information per unit cost: HIGH/1 = 3, LOW/1 = 1, HIGH/4 = 0.75
    ipc = [
        r["information_per_cost"]["refutation_weight"] / r["information_per_cost"]["simulator_cost"]
        for r in ranked
    ]
    assert ipc == sorted(ipc, reverse=True)


def test_emit_raises_on_missing_falsifier():
    raw = _valid()
    del raw["falsifier"]
    with pytest.raises(hd.MissingFieldError, match="falsifier"):
        hd.emit(raw)


def test_emit_raises_on_missing_cheapest_simulator():
    raw = _valid()
    del raw["cheapest_simulator"]
    with pytest.raises(hd.MissingFieldError, match="cheapest_simulator"):
        hd.emit(raw)


def test_emit_raises_on_empty_uncertainty():
    with pytest.raises(hd.MissingFieldError, match="uncertainty"):
        hd.emit(_valid(uncertainty=""))


def test_emit_raises_on_incomplete_predicted_effect():
    with pytest.raises(hd.MissingFieldError, match="predicted_effect"):
        hd.emit(_valid(predicted_effect={"direction": "reduce_per_token_transport"}))


def test_unknown_organ_is_refused():
    with pytest.raises(hd.UnknownOrganError, match="imaginary_systolic_array"):
        hd.emit(_valid(target_organ="imaginary_systolic_array"))


def test_scar_matching_proposal_is_refused():
    """Negative control: a proposal restating a dead mechanism must fire ScarRefusal.

    The scar list is injected rather than read from disk. Which corpus is present
    is an environment fact -- when NEGATIVE_SCIENCE_INDEX.json exists the doctor
    consults the real 51-scar corpus and its ids are real paths, not synthetic
    NS-0xx. Pinning an id would test the environment; injecting tests the guard.
    """
    injected = [
        {
            "id": "TEST-DEAD-001",
            "dead": True,
            "family": "hardware_axis",
            "mechanism": "serial bitstream expand on the per-token path",
            "needles": ["serial bitstream expand"],
            "experiment": "rice_q1 serial expand",
            "source": "injected by test",
        }
    ]
    with pytest.raises(hd.ScarRefusal, match="TEST-DEAD-001"):
        hd.emit(
            _valid(
                axis="bit_serial_vs_bit_parallel",
                hypothesis="Run rice_q1 serial bitstream expand on the per-token path",
                target_organ="mlp_gate_up_down",
                cheapest_simulator="rtl_resource_estimate",
                falsifier="bind-time expand is faster, which would not kill this",
            ),
            scars=injected,
        )


def test_real_scar_corpus_is_actually_loaded_and_non_trivial():
    """The guard above proves the mechanism; this proves it has real ammunition."""
    loaded = hd.load_scars()
    dead = [x for x in loaded["scars"] if x.get("dead")]
    assert dead, "no dead scars recovered; the refusal path would never fire in practice"
    assert loaded["source_used"], loaded
    for scar in dead:
        assert scar["id"], scar
        assert scar.get("mechanism") or scar.get("experiment"), scar


def test_second_scar_also_fires():
    injected = [
        {
            "id": "TEST-DEAD-002",
            "dead": True,
            "family": "hardware_axis",
            "mechanism": "persistent expert residency arena in HBM",
            "needles": ["persistent expert residency arena"],
            "experiment": "expert residency arena",
            "source": "injected by test",
        }
    ]
    with pytest.raises(hd.ScarRefusal, match="TEST-DEAD-002"):
        hd.emit(
            _valid(
                axis="hbm_mapping",
                target_organ="expert_bank",
                hypothesis="Keep an 8 GiB persistent expert residency arena to win Q80 wall time",
                cheapest_simulator="static_hwir",
                falsifier="first-touch misses still dominate",
            ),
            scars=injected,
        )


def test_live_catalog_is_not_scar_refused():
    scars = hd.load_scars()["scars"]
    for raw in hd.catalog():
        hits = hd.scar_hits(raw["hypothesis"], scars)
        assert hits == [], f"{raw['id']} unexpectedly matched {[h['id'] for h in hits]}"


def test_avoid_list_cites_hardware_scars():
    avoid = hd.avoid_list()
    assert avoid, "avoid list is empty; the doctor would propose into known-dead ground"
    dead_ids = {s["id"] for s in hd.load_scars()["scars"] if s.get("dead")}
    blob = json.dumps(avoid)
    # Every avoid entry must trace to a scar in the corpus that is actually loaded.
    assert any(i in blob for i in dead_ids), sorted(dead_ids)[:5]
    for item in avoid:
        assert item["negative_science"]
        assert item["predicts_not_certifies"] is True


def test_receipt_carries_recovered_and_gaps():
    doc = json.loads(hd.build().read_text())
    paths = {r["path"]: r for r in doc["recovered_implementation"]}
    assert paths["tools/headless/doctor_diagnosis.py"]["present"] is True
    assert paths["receipts/headless/FLASH_NEXT_FPGA_ORGAN_MAP.json"]["present"] is True
    assert paths["receipts/headless/QWEN27_FPGA_ORGAN_MAP.json"]["present"] is True
    atlas = paths["receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json"]
    # Environment-coupled: this file is uncommitted, so it is invisible from a
    # sparse lane worktree and visible from the primary one. Its presence is a
    # fact about the checkout, not about this module -- assert the module COPES
    # either way rather than pinning the environment it was written in.
    assert isinstance(atlas["present"], bool)
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["scar_query"]["n_dead"] > 0


def test_receipt_has_no_hardware_measurement_fields():
    doc = json.loads(hd.build().read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                    raise AssertionError(f"{here} = {v!r} is a hardware claim")
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)


def test_predicted_effect_rejects_fabricated_number():
    with pytest.raises(hd.HardwareDoctorError, match="fabricated number"):
        hd.emit(
            _valid(
                predicted_effect={
                    "direction": "reduce_per_token_transport",
                    "magnitude_class": "UNKNOWN",
                    "magnitude": 12.5,
                }
            )
        )
