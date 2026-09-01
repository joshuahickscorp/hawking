"""Roof-anchor tests. A validator nobody has watched refuse is decoration."""
from __future__ import annotations

import json

import pytest

from tools.future import roof_anchor as ra
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


def test_registry_every_roof_has_required_fields():
    ra.validate_registry()
    assert ra.ROOFS
    for roof in ra.ROOFS.values():
        for field in ra.REQUIRED_ROOF_FIELDS:
            assert field in roof, (roof["id"], field)
        measured = roof["what_was_measured"]
        for field in ra.REQUIRED_MEASURED_FIELDS:
            assert field in measured, (roof["id"], field)
        assert isinstance(roof["value_gb_s"], (int, float))
        assert roof["source_receipt"]
        assert roof["source_field"]
        assert roof["measured_or_published"] in {"measured", "published"}
        assert isinstance(roof["hops_from_origin"], int)
        assert roof["hops_from_origin"] >= 0
        assert isinstance(roof["hops_to_nearest_ceiling"], int)


def test_named_roofs_cover_the_obligation():
    ids = set(ra.ROOFS)
    assert "published_peak_819" in ids
    assert "q4_single_gemv_addr_13p6gb_max" in ids
    assert "g072_family_scoring_595p9" in ids
    assert "census_promoted_595p9" in ids
    assert "machine_genome_f32_triad_589p73" in ids
    assert "q4_catalog_addr_401" in ids
    assert "mlp_arm_a_stripped_497p4" in ids
    assert "lm_head_production_497p4" in ids
    assert "deltanet_arm_a_stripped_943p2" in ids
    assert ra.ROOFS["published_peak_819"]["value_gb_s"] == 819.0
    assert ra.ROOFS["published_peak_819"]["measured_or_published"] == "published"
    assert ra.ROOFS["q4_single_gemv_addr_13p6gb_max"]["campaign_label"] == 703.5
    assert ra.ROOFS["g072_family_scoring_595p9"]["value_gb_s"] == 595.9
    assert ra.ROOFS["machine_genome_f32_triad_589p73"]["value_gb_s"] == 589.73
    assert ra.ROOFS["mlp_arm_a_stripped_497p4"]["value_gb_s"] == 497.4
    assert ra.ROOFS["deltanet_arm_a_stripped_943p2"]["value_gb_s"] == 943.2
    assert pytest.approx(ra.ROOFS["q4_catalog_addr_401"]["value_gb_s"], rel=0, abs=0.05) == 530.7


def test_703p5_is_addr_probe_without_activation():
    roof = ra.ROOFS["q4_single_gemv_addr_13p6gb_max"]
    measured = roof["what_was_measured"]
    assert measured["activation_loaded"] is False
    assert measured["arithmetic_ran"] is False
    assert measured["dispatches"] == 1
    assert measured["bytes"] == ra.GEMV_PAYLOAD_BYTES
    assert "no input-vector load" in " ".join(roof["caveats"]).lower() or "NO input-vector load" in measured["note"]
    assert roof["usable_as_production_streaming_roof"] is False
    median = ra.ROOFS["q4_single_gemv_addr_13p6gb_median"]
    assert median["value_gb_s"] == pytest.approx(699.5736545106142)
    assert roof["value_gb_s"] == pytest.approx(703.6072736347875)


def test_497p4_loads_activation():
    arm = ra.ROOFS["mlp_arm_a_stripped_497p4"]
    head = ra.ROOFS["lm_head_production_497p4"]
    assert arm["what_was_measured"]["activation_loaded"] is True
    assert arm["what_was_measured"]["arithmetic_ran"] is False
    assert arm["what_was_measured"]["bytes"] == ra.MLP_ARM_A_BYTES
    assert arm["what_was_measured"]["dispatches"] == 3
    assert head["what_was_measured"]["activation_loaded"] is True
    assert head["what_was_measured"]["arithmetic_ran"] is True
    assert head["what_was_measured"]["dispatches"] == 2
    assert arm["usable_as_production_streaming_roof"] is True
    assert head["usable_as_production_streaming_roof"] is True


def test_catalog_530p7_is_still_addr_probe():
    roof = ra.ROOFS["q4_catalog_addr_401"]
    assert roof["what_was_measured"]["activation_loaded"] is False
    assert roof["what_was_measured"]["dispatches"] == 401
    assert roof["usable_as_production_streaming_roof"] is False


def test_943p2_exceeds_published_peak():
    roof = ra.ROOFS["deltanet_arm_a_stripped_943p2"]
    published = ra.ROOFS["published_peak_819"]["value_gb_s"]
    assert roof["value_gb_s"] > published
    assert roof["value_gb_s"] / published == pytest.approx(1.15, rel=0.01)
    assert roof["usable_as_production_streaming_roof"] is False
    assert any("819" in c for c in roof["caveats"])


def test_589p73_traces_to_f32_triad():
    trace = ra.TRACE_589P73
    assert trace["value_gb_s"] == 589.73
    assert trace["is_honestly_measured"] is True
    assert trace["is_scoring_reference_promoted"] is False
    assert "triad" in trace["measured_of"]
    assert "f32" in trace["measured_of"]
    assert trace["hops_to_atlas_ceiling"] == 2
    hops = {h["hop"]: h for h in trace["hops"]}
    assert hops[0]["receipt"] == ra.GENOME_REL
    assert hops[0]["field"] == "measured_bandwidth.median_gb_s"
    assert hops[1]["receipt"] == ra.ATLAS_REL
    assert hops[1]["field"] == "identities.machine.measured_dram_gbps"
    assert hops[2]["receipt"] == ra.ATLAS_REL
    assert hops[2]["field"] == "THE_CEILING.measured_roof_gb_s"
    assert hops[2]["ceiling_value"] == pytest.approx(59.69591069708626)
    genome = ra.ROOFS["machine_genome_f32_triad_589p73"]
    assert genome["hops_from_origin"] == 0
    assert genome["hops_to_nearest_ceiling"] == 2
    assert genome["what_was_measured"]["bytes"] == ra.TRIAD_BYTES_PER_REP
    assert "UNSTATED-ROOF" in trace["same_defect_class_as_595p9"]


def test_595p9_three_hops_from_family_scoring_reference():
    trace = ra.TRACE_595P9
    assert trace["value_gb_s"] == 595.9
    assert trace["is_scoring_reference_promoted"] is True
    assert trace["hops_to_machine_property"] == 3
    hops = {h["hop"]: h for h in trace["hops"]}
    assert hops[0]["receipt"] == ra.G072_REL
    assert hops[1]["receipt"] == ra.GENESIS_REL
    assert hops[2]["receipt"] == ra.CANON_REL
    assert hops[3]["receipt"] == ra.CENSUS_REL
    assert ra.ROOFS["g072_family_scoring_595p9"]["hops_to_nearest_ceiling"] == 3
    assert ra.ROOFS["census_promoted_595p9"]["hops_from_origin"] == 3
    assert ra.ROOFS["census_promoted_595p9"]["kind"] == "promoted"


def test_ceiling_without_roof_raises():
    with pytest.raises(ra.UnstatedRoof, match="595.9"):
        ra.compute_ceiling(active_bytes=ra.ACTIVE_BYTES_ATLAS)
    with pytest.raises(ra.UnstatedRoof, match="595.9"):
        ra.compute_ceiling(roof_id=None, active_bytes=ra.ACTIVE_BYTES_ATLAS)
    with pytest.raises(ra.UnstatedRoof, match="595.9"):
        ra.compute_ceiling(roof_id="", active_bytes=ra.ACTIVE_BYTES_ATLAS)
    with pytest.raises(ra.UnstatedRoof, match="595.9"):
        ra.compute_ceiling(roof_id="   ", active_bytes=ra.ACTIVE_BYTES_ATLAS)


def test_ceiling_with_positional_gbs_raises():
    """A raw GB/s number is how 595.9 became a machine property."""
    with pytest.raises(ra.UnstatedRoof, match="raw GB/s"):
        ra.compute_ceiling(703.5, ra.ACTIVE_BYTES_ATLAS)
    with pytest.raises(ra.UnstatedRoof, match="raw GB/s"):
        ra.compute_ceiling(589.73, active_bytes=ra.ACTIVE_BYTES_ATLAS)


def test_ceiling_unknown_roof_raises():
    with pytest.raises(ra.UnknownRoof):
        ra.compute_ceiling(roof_id="not_a_roof", active_bytes=ra.ACTIVE_BYTES_ATLAS)


def test_ceiling_with_named_roof_names_it():
    row = ra.compute_ceiling(
        roof_id="mlp_arm_a_stripped_497p4",
        active_bytes=ra.ACTIVE_BYTES_ATLAS,
    )
    assert row["roof_id"] == "mlp_arm_a_stripped_497p4"
    assert row["roof_source_receipt"] == ra.ALU_REL
    assert row["roof_source_field"] == "mlp.arm_a_stripped.effective_gb_s"
    assert row["roof_value_gb_s"] == 497.4
    assert row["raw_tps_ceiling"] == pytest.approx(497.4e9 / ra.ACTIVE_BYTES_ATLAS)
    assert row["would_improve_tps"] is None
    assert row["what_was_measured"]["activation_loaded"] is True
    assert row["hops_from_origin"] == 0


def test_atlas_ceiling_recomputes_against_named_589p73():
    row = ra.compute_ceiling(
        roof_id="machine_genome_f32_triad_589p73",
        active_bytes=ra.ACTIVE_BYTES_ATLAS,
    )
    assert row["raw_tps_ceiling"] == pytest.approx(59.69591069708626)
    assert row["roof_id"] == "machine_genome_f32_triad_589p73"


def test_audit_covers_the_minimum_four():
    by_id = ra.audit_by_id()
    for needed in ra.MINIMUM_AUDIT_IDS:
        assert needed in by_id, needed
    atlas = by_id["atlas_the_ceiling"]
    assert atlas["rests_on_roof_id"] == "machine_genome_f32_triad_589p73"
    assert atlas["roof_named_in_record"] is False
    assert atlas["defect"] == "unstated_roof"
    census = by_id["census_anchor_595p9"]
    assert census["rests_on_roof_id"] == "g072_family_scoring_595p9"
    assert census["roof_named_in_record"] is False
    assert census["defect"] == "unstated_roof"
    budget = by_id["causal_budget_roof_on_todays_bytes"]
    assert budget["rests_on_roof_id"] == "q4_single_gemv_addr_13p6gb_max"
    # NOT the digits. quoted_value is resolved from the source reference at
    # evaluation time, so asserting it here pinned a moment and broke when the
    # budget was corrected. What must hold is that this rung rests on the
    # addr-probe roof and is known to steer priorities.
    assert budget["quote_checked"] is True
    assert budget["steers_priorities"] is True
    assert budget["caveat"] == "no_input_vector_load"
    path = by_id["path_to_71_campaign_target"]
    assert path["rests_on_roof_id"] is None
    assert path["roof_named_in_record"] is False
    assert path["defect"] == "unstated_roof"
    composed = by_id["path_to_71_best_composed"]
    assert composed["source"]["artifact"] == ra.PATH_REL
    assert composed["source"]["field"] == "gap_to_71.best_composed_tps"
    assert isinstance(composed["source"]["tolerance"], float)
    live = ra._resolve_field(composed["source"]["artifact"], composed["source"]["field"])
    assert live is not ra._UNRESOLVABLE
    assert composed["quoted_value"] == live
    assert composed["quote_checked"] is True


def test_causal_budget_47p97_names_497p4():
    row = ra.audit_by_id()["causal_budget_demonstrated_regime"]
    assert row["rests_on_roof_id"] == "lm_head_production_497p4"
    assert row["roof_named_in_record"] is True
    assert row["defect"] is None


def test_recommended_anchor_is_497p4_defended():
    rec = ra.RECOMMENDED_ANCHOR
    assert rec["roof_id"] == "mlp_arm_a_stripped_497p4"
    assert rec["value_gb_s"] == 497.4
    assert rec["corroborated_by"] == "lm_head_production_497p4"
    assert rec["agrees_with_the_brief"] is True
    assert rec["disagreement"] is None
    against = rec["against"]
    assert set(against) >= {
        "published_peak_819",
        "q4_single_gemv_addr_13p6gb_max",
        "g072_family_scoring_595p9",
        "machine_genome_f32_triad_589p73",
        "q4_catalog_addr_401",
        "deltanet_arm_a_stripped_943p2",
    }
    assert "not measured" in against["published_peak_819"]["rejected_because"]
    assert "activation" in against["q4_single_gemv_addr_13p6gb_max"]["rejected_because"]
    assert "three hops" in against["g072_family_scoring_595p9"]["rejected_because"]
    assert "triad" in against["machine_genome_f32_triad_589p73"]["rejected_because"]
    assert "addr_probe" in against["q4_catalog_addr_401"]["rejected_because"]
    assert "1.15" in against["deltanet_arm_a_stripped_943p2"]["rejected_because"]
    assert rec["catalog_full_sibling_does_not_overturn"]["value_gb_s"] == pytest.approx(
        505.8100047843556
    )


def test_build_emits_sealed_receipt():
    out = ra.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "ROOF_ANCHOR.json"
    assert doc["schema"] == ra.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    _assert_no_hardware_claims(doc)
    assert doc["registry"]
    assert doc["trace_589p73"]["hops_to_atlas_ceiling"] == 2
    assert doc["trace_595p9"]["hops_to_machine_property"] == 3
    assert doc["recommended_anchor"]["roof_id"] == ra.RECOMMENDED_ANCHOR_ID
    audited = {row["id"] for row in doc["ceiling_audit"]}
    assert set(ra.MINIMUM_AUDIT_IDS) <= audited
    assert "atlas_the_ceiling" in doc["unstated_roofs_on_record"]
    assert "census_anchor_595p9" in doc["unstated_roofs_on_record"]
    assert "path_to_71_campaign_target" in doc["unstated_roofs_on_record"]
    assert "causal_budget_roof_on_todays_bytes" in doc["no_input_vector_load_flags"]
    assert doc["recompute"]["atlas_matches_recompute"] is True
    assert doc["source_verification"]["ok"] is True
    assert not doc["source_verification"]["mismatches"]


def test_receipt_never_numeric_hardware_fields():
    doc = json.loads(ra.build().read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS:
                    assert not isinstance(v, (int, float)) or isinstance(v, bool), here
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)


def test_get_roof_refuses_silence():
    with pytest.raises(ra.UnstatedRoof):
        ra.get_roof("")
    with pytest.raises(ra.UnknownRoof):
        ra.get_roof("589.73")
    roof = ra.get_roof("machine_genome_f32_triad_589p73")
    assert roof["source_field"] == "measured_bandwidth.median_gb_s"
