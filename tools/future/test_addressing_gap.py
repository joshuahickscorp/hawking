"""Addressing-gap tests. A validator nobody has watched refuse is decoration."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tools.future import addressing_gap as ag
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


def _spread(median, lo, hi):
    return {
        "median_gb_s": median,
        "min_gb_s": lo,
        "max_gb_s": hi,
        "median_ns": 1,
        "min_ns": 1,
        "max_ns": 1,
        "all_ns": [1, 1, 1],
    }


def _probe(*, label, payload, median, lo, hi, dispatches=1, kernel="k", topology="t"):
    return {
        "label": label,
        "payload_bytes": payload,
        "dispatches": dispatches,
        "kernel": kernel,
        "topology": topology,
        "spread": _spread(median, lo, hi),
    }


def _honest(*, catalog=True, reduced_null_catalog=False):
    catalog_obj = None if reduced_null_catalog else {
        "dispatches": 401,
        "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe",
        "label": "production_catalog_401_gemvs",
        "payload_bytes": ag.GEMV_PAYLOAD_BYTES,
        "topology": "production_shape_catalog",
        "spread": _spread(530.6544688491846, 509.27, 539.03),
        "note": "192 MLP + 144 DN + 64 GQA + 1 lm_head",
    }
    decode_obj = None if reduced_null_catalog else {
        "dispatches": 401,
        "payload_bytes": ag.GEMV_PAYLOAD_BYTES,
        "spread": _spread(454.92, 293.99, 513.0233342441745),
    }
    full_obj = None if reduced_null_catalog else {
        "dispatches": 401,
        "payload_bytes": ag.GEMV_PAYLOAD_BYTES,
        "spread": _spread(505.81, 499.69, 511.05),
    }
    return {
        "hardware": {"published_peak_gb_s": 819.0},
        "byte_count_adjudication": {"defended_bytes": ag.GEMV_PAYLOAD_BYTES},
        "contamination_note": "contended",
        "timing_label": "GPU_PROTECTED_CPU_CONTENDED",
        "clean_box": False,
        "q4_single_gemv_addr_probe": [
            _probe(label="64mib", payload=67107840, median=817.14, lo=664.4, hi=833.6),
            _probe(
                label="gemv_payload_13p612gb",
                payload=ag.GEMV_PAYLOAD_BYTES,
                median=699.5736545106142,
                lo=693.1508595217028,
                hi=703.6072736347875,
                kernel="qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe",
                topology="single_gemv",
            ),
        ],
        "q4_single_gemv_decode_probe": [
            _probe(
                label="gemv_payload_13p612gb",
                payload=ag.GEMV_PAYLOAD_BYTES,
                median=683.797,
                lo=681.6,
                hi=686.96,
            )
        ],
        "q4_single_gemv_full": [
            _probe(
                label="gemv_payload_13p612gb",
                payload=ag.GEMV_PAYLOAD_BYTES,
                median=666.681,
                lo=664.2,
                hi=668.8,
            )
        ],
        "q4_production_catalog_addr_probe": catalog_obj if catalog else None,
        "q4_production_catalog_decode_probe": decode_obj,
        "q4_production_catalog_full": full_obj,
        "q4_tiled_production_organ": [
            _probe(
                label="13p612gb_tiled_gate_addr",
                payload=13589381120,
                median=591.1317979446468,
                lo=581.7,
                hi=597.7,
                dispatches=287,
            )
        ],
        "unique_once_sweep": [
            _probe(
                label="gemv_payload_13p612gb",
                payload=ag.GEMV_PAYLOAD_BYTES,
                median=375.65,
                lo=370.0,
                hi=380.0,
            )
        ],
    }


def _atlas():
    return {
        "headline": {"active_weight_bytes_per_token": 9878901136},
        "THE_CEILING": {
            "sealed_raw_tps_recorded": 34.14,
            "effective_weight_bandwidth_gb_s": 337.26568478304,
            "measured_roof_gb_s": 589.73,
            "raw_tps_ceiling_at_100pct_of_roof": 59.69591069708626,
            "raw_tps_needed_for_50_accepted_at_the_30_of_43_floor": 71.66666666666667,
        },
        "claim_boundary": ["NOTHING WAS TIMED. derivation."],
        "identities": {"machine": {"receipt": "receipts/headless/MACHINE_GENOME.json"}},
    }


def _ascent():
    return {
        "prior_not_rederived": {"parent_active_bytes_per_token": 9878901136},
        "production_decode": {
            "before": {"achieved_gb_s": 356.671220723892, "id": "tpr64"},
            "active_bytes_per_token": 9878901136,
        },
        "isolated_gemv": {
            "shapes": [
                {
                    "label": "mlp.gate_proj",
                    "weight_payload_bytes": 27852800,
                    "arms": {
                        "tpr64": {"weight_gb_s_median": 197.015},
                        "qmvfast_addr_probe": {"weight_gb_s_median": 968.793},
                    },
                }
            ]
        },
    }


def _injected(**overrides):
    base = {
        ag.LANE_HONEST: None,
        ag.LANE_HONEST_REDUCED: None,
        ag.LANE_G044: None,
        ag.ASCENT_HONEST: _honest(),
        ag.ASCENT_HONEST_REDUCED: _honest(reduced_null_catalog=True),
        ag.ASCENT_G044: {"bandwidth_ceiling_gb_s": 594.3492381201206},
        ag.ATLAS_REL: _atlas(),
        ag.ASCENT_REL: _ascent(),
        ag.CENSUS_REL: {
            "artifact": {"anchors_not_rederived": {"measured_roof_GB_s": 595.9}}
        },
        ag.G072_REL: {
            "measured_roof_gb_s": 595.9,
            "roof_basis": "the roofline sweep in the SAME run peaks at 595.9",
        },
        ag.GENOME_REL: {
            "measured_bandwidth": {
                "median_gb_s": 589.73,
                "is_theoretical_roof": False,
            }
        },
        ag.LEDGER_REL: {
            "three_roofs": {
                "DEVICE_THEORETICAL": {"value": 819.0},
                "DEVICE_MEASURED_SUSTAINED": {"value": 778.8},
                "MODEL_REACHABLE": {"value": 729.6978633780673},
            }
        },
        ag.ORGAN_BW_REL: {"organ_attribution": {"largest_roof_gap_organ": "mlp_gate_up"}},
        ag.UNPACK_REL: {"THE_WALL": {"native_best_gb_s": 160.87}},
    }
    base.update(overrides)
    return base


def test_build_emits_sealed_receipt():
    out = ag.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "ADDRESSING_GAP.json"
    assert doc["schema"] == ag.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    _assert_no_hardware_claims(doc)
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]["fails_closed"]
    assert doc["tallest_removable"]["not_a_tps_gain"] is True
    assert doc["tallest_removable"]["what_would_remove_it"]["would_improve_tps"] is None
    assert doc["self_timing"]["class"] == "SELF_MEASURED_DIRTY"
    assert "GPU" in doc["self_timing"]["not"]


def test_receipt_never_numeric_hardware_fields():
    doc = json.loads(ag.build().read_text())

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


def test_unsourced_number_is_refused():
    row = ag.cite(703.5, source_receipt=None, json_path="claimed", statistic="median")
    assert row["status"] == ag.STATUS_REFUSED
    assert row["reason"] == "unsourced_number"
    assert row["value"] is None
    missing = ag.cite("x", source_receipt="r", json_path="claimed", statistic="median")
    assert missing["status"] == ag.STATUS_REFUSED
    assert missing["reason"] == "absent_or_non_numeric"


def test_claimed_703p5_is_not_the_median():
    analysis = ag.analyze(_injected())
    adj = analysis["lane_claim_adjudication"]["single_gemv_703p5_as_median"]
    assert adj["status"] == ag.STATUS_REFUSED
    assert adj["reason"] == "claimed_does_not_match_sourced_statistic"
    median = analysis["sourced_rungs"]["single_gemv_addr_median_13p6gb"]
    maximum = analysis["sourced_rungs"]["single_gemv_addr_max_13p6gb"]
    assert median["status"] == ag.STATUS_LOADED
    assert median["value"] == pytest.approx(699.5736545106142)
    assert maximum["value"] == pytest.approx(703.6072736347875)
    assert adj["min_adjudication"]["status"] == ag.STATUS_REFUSED
    assert adj["max_adjudication"]["status"] == ag.STATUS_REFUSED


def test_headless_honest_path_is_refused_when_absent():
    analysis = ag.analyze(_injected())
    named = analysis["path_adjudication"]["lane_named_honest_headless"]
    recovered = analysis["path_adjudication"]["recovered_honest_ascent"]
    assert named["status"] == ag.STATUS_REFUSED
    assert recovered["status"] == ag.STATUS_LOADED
    assert named["rel"] == ag.LANE_HONEST
    assert recovered["rel"] == ag.ASCENT_HONEST


def test_reduced_catalog_probe_null_is_refused():
    analysis = ag.analyze(_injected())
    reduced = analysis["sourced_rungs"]["reduced_catalog_addr"]
    assert reduced["status"] == ag.STATUS_REFUSED
    live = analysis["sourced_rungs"]["catalog_addr_median"]
    assert live["status"] == ag.STATUS_LOADED
    assert live["value"] == pytest.approx(530.6544688491846)


def test_catalog_530p7_and_atlas_337p3_match_sourced_statistics():
    analysis = ag.analyze(_injected())
    assert analysis["lane_claim_adjudication"]["catalog_addr_530p7"]["status"] == ag.STATUS_LOADED
    assert analysis["lane_claim_adjudication"]["atlas_effective_337p3"]["status"] == ag.STATUS_LOADED
    assert analysis["lane_claim_adjudication"]["published_819"]["status"] == ag.STATUS_LOADED
    decode = analysis["lane_claim_adjudication"]["catalog_decode_513_as_max"]
    assert decode["status"] == ag.STATUS_LOADED
    assert decode["median_of_same_probe"]["value"] == pytest.approx(454.92)


def test_t2_is_attributed_and_t4_is_unattributed():
    analysis = ag.analyze(_injected())
    t2, t4 = analysis["transitions"]
    assert t2["id"].startswith("T2_")
    assert t2["status"] == ag.STATUS_ATTRIBUTED
    assert t2["delta"]["comparable"] is True
    assert t2["delta_gb_s"] == pytest.approx(699.5736545106142 - 530.6544688491846)
    assert t4["status"] == ag.STATUS_UNATTRIBUTED
    assert t4["delta_gb_s"] is None
    assert t4["delta"]["comparable"] is False
    assert t4["mechanism"] == ag.STATUS_UNATTRIBUTED


def test_unattributed_remainder_is_not_distributed():
    total = 100.0
    pieces = [
        {
            "id": "known",
            "status": ag.STATUS_ATTRIBUTED,
            "delta_gb_s": 30.0,
            "mechanism": "catalog",
        },
        {
            "id": "guess",
            "status": ag.STATUS_UNATTRIBUTED,
            "delta_gb_s": 70.0,
            "mechanism": "would_close_the_sum",
        },
    ]
    closed = ag.close_loss_chain(total, pieces)
    assert closed["attributed_gb_s"] == pytest.approx(30.0)
    assert closed["unattributed_gb_s"] == pytest.approx(70.0)
    assert closed["unattributed_label"] == ag.STATUS_UNATTRIBUTED
    assert [p["id"] for p in closed["pieces"]] == ["known"]
    full = ag.analyze(_injected())
    chain = full["loss_chain_closure"]
    assert chain["unattributed_gb_s"] == pytest.approx(
        chain["total_delta_gb_s"] - chain["attributed_gb_s"]
    )
    assert chain["unattributed_gb_s"] > 0
    assert chain["attributed_gb_s"] == pytest.approx(699.5736545106142 - 530.6544688491846)


def test_ceiling_without_roof_raises():
    with pytest.raises(ag.UnstatedRoof):
        ag.ceiling_raw_tps(9878901136, None)
    with pytest.raises(ag.UnstatedRoof):
        ag.ceiling_raw_tps(
            9878901136,
            {"status": ag.STATUS_LOADED, "value": 589.73, "source_receipt": None},
        )


def test_every_ceiling_names_its_roof():
    analysis = ag.analyze(_injected())
    assert analysis["ceilings"]
    named = []
    for row in analysis["ceilings"]:
        assert "roof_name" in row
        if row["status"] == ag.STATUS_LOADED:
            assert row["roof_source"]
            assert row["raw_tps_ceiling"] is not None
            assert row["would_improve_tps"] is None
            named.append(row["roof_name"])
    assert "published_peak" in named
    assert "q4_single_gemv_addr_13p6gb_median" in named
    assert "machine_genome_f32_triad" in named
    assert "g072_multi_plane_scoring_reference" in named
    assert "census_anchor_595p9" in named
    atlas_row = next(
        r for r in analysis["ceilings"] if r.get("roof_name") == "machine_genome_f32_triad"
    )
    assert atlas_row["raw_tps_ceiling"] == pytest.approx(59.69591069708626)
    honest_row = next(
        r
        for r in analysis["ceilings"]
        if r.get("roof_name") == "q4_single_gemv_addr_13p6gb_median"
    )
    # 699.57 GB/s against 9.879 GB/token is still below 71.67 needed.
    assert honest_row["raw_tps_ceiling"] < 71.66666666666667
    published_row = next(
        r for r in analysis["ceilings"] if r.get("roof_name") == "published_peak"
    )
    assert published_row["raw_tps_ceiling"] > 71.66666666666667
    check = analysis["atlas_ceiling_check"]
    assert check["matches_recompute"] is True
    assert "MACHINE_GENOME" in check["roof_the_atlas_used"]


def test_tps_claim_without_falsifier_is_refused():
    with pytest.raises(ag.TpsClaimRefused):
        ag.tps_hypothesis("ICB would raise TPS", falsifier=None)
    with pytest.raises(ag.TpsClaimRefused):
        ag.tps_hypothesis("ICB would raise TPS", falsifier="  ")
    ok = ag.tps_hypothesis("ICB might recover catalog topology", falsifier="A/B overlap dies")
    assert ok["would_improve_tps"] is None
    assert ok["kind"] == ag.STATUS_HYPOTHESIS


def test_moe_gather_is_refused_on_this_genome():
    row = ag.refuse_moe_gather("qwen38-27b-hybrid-deltanet-gqa")
    assert row["status"] == ag.STATUS_REFUSED
    assert row["mechanism"] == "gather_for_route_selection"
    moe = ag.refuse_moe_gather("mixtral-8x7b-moe")
    assert moe["status"] == ag.STATUS_STRUCTURAL


def test_gb_s_above_published_peak_is_not_a_dram_roof():
    published = ag.cite(
        819.0,
        source_receipt="r",
        json_path="peak",
        statistic="datasheet",
    )
    cache = ag.cite(
        968.793,
        source_receipt="r",
        json_path="addr",
        statistic="median",
    )
    row = ag.refuse_dram_roof(cache, published)
    assert row["status"] == ag.STATUS_REFUSED
    assert row["as_dram_roof"] is False
    ok = ag.refuse_dram_roof(
        ag.cite(699.57, source_receipt="r", json_path="addr", statistic="median"),
        published,
    )
    assert ok["as_dram_roof"] is True
    analysis = ag.analyze(_injected())
    assert analysis["q2_organ_addr_probe_as_dram_roof"]["as_dram_roof"] is False


def test_absent_receipts_do_not_skip_they_refuse():
    empty = {rel: None for rel in _injected()}
    analysis = ag.analyze(empty)
    median = analysis["sourced_rungs"]["single_gemv_addr_median_13p6gb"]
    assert median["status"] == ag.STATUS_REFUSED
    t2, t4 = analysis["transitions"]
    assert t2["status"] != ag.STATUS_ATTRIBUTED
    assert t4["status"] == ag.STATUS_UNATTRIBUTED
    assert any(c["status"] == ag.STATUS_REFUSED for c in analysis["ceilings"])
    assert analysis["active_bytes_adjudication"]["not_three_of_the_named_recover_set"] is True


def test_active_bytes_three_receipt_claim_does_not_pass_on_two():
    analysis = ag.analyze(_injected())
    adj = analysis["active_bytes_adjudication"]
    assert adj["claimed"] == 9878901136
    assert set(adj["recover_receipts_that_carry_the_claimed_count"]) == {
        "ACCELERATOR_TOKEN_BYTES_ATLAS",
        "BANDWIDTH_ASCENT",
    }
    assert adj["not_three_of_the_named_recover_set"] is True
    assert adj["status"] == "TWO_OF_NAMED_RECOVER_SET"
    payload = adj["honest_roof_defended_gemv_payload_bytes"]
    assert payload["value"] == ag.GEMV_PAYLOAD_BYTES


def test_every_loaded_rung_carries_a_receipt():
    analysis = ag.analyze(_injected())
    for name, row in analysis["sourced_rungs"].items():
        if row.get("status") == ag.STATUS_LOADED:
            assert row.get("source_receipt"), name
            assert row.get("json_path"), name
            assert row.get("statistic"), name


def test_module_parses():
    src = Path(ag.__file__).read_text()
    ast.parse(src)
    assert "raise NotImplementedError" not in src
    assert "TODO" not in src
    assert "pytest.skip(" not in src


def test_tallest_removable_is_t2_not_the_cross_genome_drop():
    analysis = ag.analyze(_injected())
    tallest = analysis["tallest_removable"]
    assert tallest["status"] == ag.STATUS_ATTRIBUTED
    assert "catalog" in tallest["name"]
    assert tallest["what_would_remove_it"]["falsifier"]
    assert "ICB" in tallest["what_would_remove_it"]["text"] or "ICB" in tallest[
        "what_would_remove_it"
    ]["falsifier"]
    ranking = {r["id"]: r for r in analysis["removable_ranking"]}
    assert ranking["per_dispatch_rebinding_and_catalog_indirection"]["declared_not_executed"] is True
    assert ranking["gather_for_route_selection"]["class"] == ag.STATUS_REFUSED
    assert ranking["many_tensors_vs_one_gemv"]["class"] == ag.STATUS_STRUCTURAL
    sleeping = [u for u in analysis["next_workunits"] if u["state"] == "SLEEPING"]
    assert sleeping
    assert all("flock" not in u.get("wake", "").lower() or "never flocks" in u["wake"] for u in sleeping)
