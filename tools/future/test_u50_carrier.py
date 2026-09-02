"""U50-family variants and CarrierEnvelope. PREHARDWARE. No board on this host.

Mutation target for the refusal test is hwir.CARRIER_ENVELOPE_BINDING.
Set it False (ignore carrier limits); this file's brochure-refusal test
must then FAIL. Restore to True. Never leave the flag False in source.
"""
from __future__ import annotations

import pytest

from tools.future import hwir


REQUIRED = hwir.U50_VARIANT_REQUIRED_FIELDS


def test_generic_u50_class_envelope_is_not_rewritten():
    profile = hwir.synthetic_u50_class()
    assert profile.device_id == "synthetic-u50-class"
    assert profile.LUT == 872_000
    assert profile.DSP == 9_024
    assert profile.BRAM == 2_016
    assert profile.URAM == 960
    assert profile.hbm_channels == 32
    assert profile.hbm_capacity_bytes == 8 * 1024 ** 3
    assert profile.origin == "SYNTHETIC_U50_CLASS_DECLARED_NOT_A_BOARD"
    doc = profile.to_dict()
    hwir.assert_no_hardware_measured(doc)
    assert doc["evidence_tier"] == "STATIC"
    assert "bandwidth_gbps" not in doc
    assert doc["real_carrier"] == hwir.UNPINNED


def test_u50_family_variants_are_selectable():
    ids = hwir.list_u50_family_profiles()
    assert tuple(ids) == hwir.U50_FAMILY_VARIANT_IDS
    assert set(ids) == {"u50", "u50c", "u50dd", "u50lv"}
    u50 = hwir.select_device_profile("u50")
    assert u50.variant_id == "u50"
    assert u50.sku == "A-U50-P00G-PQ-G"
    assert u50.LUT == 872_000
    assert u50.DSP == 5_952
    assert u50.BRAM == 1_344
    assert u50.URAM == 640
    assert u50.hbm_channels == 32
    assert u50.hbm_capacity_bytes == 8 * 1024 ** 3
    assert u50.power_envelope_w == 75
    assert u50.cooling == "passive"
    assert u50.pcie_generation == 3
    assert u50.pcie_lanes == 16
    lv = hwir.select_device_profile("u50lv")
    assert lv.sku == "A-U50-P00G-LV-G"
    assert lv.fpga_part == "XCU50-FSVH2104-2LV-E"
    assert lv.pcie_generation == 3
    assert lv.pcie_lanes is None  # UNPINNED: Table 1 x16 vs VLOW x4
    dd = hwir.select_device_profile("u50dd")
    assert dd.sku == "A-U50DD-P00G-ES3-G"
    assert dd.DSP == 5_952
    c = hwir.select_device_profile("u50c")
    assert c.variant_id == "u50c"
    assert c.LUT == 0
    assert c.DSP == 0
    assert c.hbm_capacity_bytes == 0
    assert c.sku is None


def test_every_variant_field_is_sourced_or_unpinned():
    for vid in hwir.U50_FAMILY_VARIANT_IDS:
        profile = hwir.u50_family_profile(vid)
        hwir.assert_variant_provenance(profile)
        doc = profile.to_dict()
        hwir.assert_no_hardware_measured(doc)
        assert "HARDWARE_MEASURED" not in hwir.collect_evidence_tiers(doc)
        assert doc["hardware_measured"] is False
        assert "bandwidth_gbps" not in doc
        prov = doc["field_provenance"]
        for name in REQUIRED:
            assert name in prov, f"{vid} missing provenance for {name}"
            meta = prov[name]
            if meta["pinned"]:
                assert meta["value"] != hwir.UNPINNED
                assert meta["value"] is not None
                assert meta["document_class"] not in {"", hwir.UNPINNED, None}
                assert meta["citation"]
                assert meta["evidence_tier"] in hwir.EVIDENCE_TIERS
                assert meta["hardware_measured"] is False
            else:
                assert meta["value"] == hwir.UNPINNED
                assert meta["note"]
                assert meta["hardware_measured"] is False
            # No silent default: a pinned field must name a document class.
            assert "document_class" in meta


def test_u50c_is_unpinned_not_interpolated_from_u50_or_u55c():
    c = hwir.u50_family_profile("u50c")
    for name in REQUIRED:
        meta = c.field_provenance[name]
        assert meta["pinned"] is False
        assert meta["value"] == hwir.UNPINNED
        assert "U55C" in meta["note"]
        assert "interpolat" in meta["note"].lower()
    # Must not silently inherit the U50 8 GB / 5952 DSP brochure.
    assert c.DSP != 5_952
    assert c.hbm_capacity_bytes != 8 * 1024 ** 3


def test_real_carrier_is_unpinned():
    carrier = hwir.unpinned_real_carrier()
    doc = carrier.to_dict()
    hwir.assert_no_hardware_measured(doc)
    assert doc["real_carrier"] == hwir.UNPINNED
    assert "comma" in doc["real_carrier_note"].lower()
    assert carrier.example is False
    assert carrier.pcie_generation is None
    assert carrier.pcie_lanes is None
    assert carrier.sustained_power_w is None
    assert carrier.airflow_class is None
    assert carrier.mechanical_limit is None
    for meta in doc["field_provenance"].values():
        assert meta["pinned"] is False
        assert meta["value"] == hwir.UNPINNED
        assert meta["note"]
    # Constraining with the real unpinned carrier must not invent a tighter envelope.
    device = hwir.u50_family_profile("u50")
    out = carrier.constrain(device)
    assert out.LUT == device.LUT
    assert out.DSP == device.DSP
    assert out.host_device_bytes_per_modelled_cycle == device.host_device_bytes_per_modelled_cycle
    assert out.power_envelope_w == device.power_envelope_w


def test_low_power_few_lane_carrier_reduces_bandwidth_and_shrinks_plan():
    device = hwir.u50_family_profile("u50")
    full = hwir.example_full_airflow_server_slot()
    low = hwir.example_constrained_low_power_slot()
    kernel = hwir.canonical_qgemv_kernel()

    plan_full = hwir.admissible_plan(kernel, device, full)
    plan_low = hwir.admissible_plan(kernel, device, low)
    hwir.assert_no_hardware_measured(plan_full)
    hwir.assert_no_hardware_measured(plan_low)
    assert plan_full["ok"] is True
    assert plan_low["ok"] is True
    assert "bandwidth_gbps" not in plan_full
    assert "bandwidth_gbps" not in plan_low

    assert (
        plan_low["host_device_bytes_per_modelled_cycle"]
        < plan_full["host_device_bytes_per_modelled_cycle"]
    )
    assert plan_low["modelled_cycles_total"] > plan_full["modelled_cycles_total"]
    assert plan_low["resource_budget"]["DSP"] < plan_full["resource_budget"]["DSP"]
    assert plan_low["resource_budget"]["LUT"] < plan_full["resource_budget"]["LUT"]
    assert plan_low["power_envelope_w"] < plan_full["power_envelope_w"]
    assert plan_low["pcie_lanes"] < plan_full["pcie_lanes"]
    assert plan_low["fpga_rows"] <= plan_full["fpga_rows"]
    assert plan_low["thermal_mismatch"] is True
    assert plan_full["thermal_mismatch"] is False

    constrained = hwir.constrain_device_profile(device, low)
    unconstrained = hwir.constrain_device_profile(device, full)
    xfer_low = hwir.model_host_device_transfer(kernel, constrained)
    xfer_full = hwir.model_host_device_transfer(kernel, unconstrained)
    assert xfer_low["assumed_bytes_per_modelled_cycle"] < xfer_full["assumed_bytes_per_modelled_cycle"]
    assert xfer_low["modelled_cycles_total"] > xfer_full["modelled_cycles_total"]
    assert "bandwidth_gbps" not in xfer_low
    assert "bandwidth_gbps" not in xfer_full


def test_brochure_kernel_refused_under_constrained_carrier():
    """A kernel that fits the brochure envelope is REFUSED under a constrained carrier.

    Mutation target: hwir.CARRIER_ENVELOPE_BINDING. If constrain() ignores
    carrier limits, this test FAILS (the kernel still fits).
    """
    device = hwir.u50_family_profile("u50")
    full = hwir.example_full_airflow_server_slot()
    low = hwir.example_constrained_low_power_slot()
    kernel = hwir.brochure_fit_kernel()

    used = hwir.estimate_qgemv_resources(kernel)["used"]
    assert used["DSP"] == kernel.mac_lanes * kernel.tile_m == 4096
    assert used["DSP"] <= device.DSP
    hwir.fit_kernel_to_device(kernel, device)  # brochure admits it

    plan_full = hwir.admissible_plan(kernel, device, full)
    assert plan_full["ok"] is True
    hwir.fit_kernel_to_device(kernel, hwir.constrain_device_profile(device, full))

    constrained = hwir.constrain_device_profile(device, low)
    assert constrained.DSP < used["DSP"]
    assert constrained.LUT < device.LUT
    assert constrained.host_device_bytes_per_modelled_cycle < device.host_device_bytes_per_modelled_cycle
    with pytest.raises(hwir.ResourceOverBudget) as exc:
        hwir.fit_kernel_to_device(kernel, constrained)
    assert "DSP" in exc.value.overflow
    plan_low = hwir.admissible_plan(kernel, device, low)
    assert plan_low["ok"] is False
    assert plan_low["refused"] is True
    hwir.assert_no_hardware_measured(plan_low)
    # Preboard path also sees the reduced envelope.
    pre = hwir.run_qgemv_preboard(kernel, device, carrier=low)
    hwir.assert_no_hardware_measured(pre)
    assert pre["resource_fit"]["ok"] is False
    assert pre["real_carrier"] == hwir.UNPINNED


def test_carrier_cannot_upgrade_brochure_envelope():
    device = hwir.u50_family_profile("u50")
    full = hwir.example_full_airflow_server_slot()
    out = full.constrain(device)
    assert out.LUT <= device.LUT
    assert out.DSP <= device.DSP
    assert out.host_device_bytes_per_modelled_cycle <= device.host_device_bytes_per_modelled_cycle
    assert out.power_envelope_w <= device.power_envelope_w
    # Full slot is Gen4 x16; U50 brochure is Gen3 x16 class. Beat must not rise.
    assert out.host_device_bytes_per_modelled_cycle == device.host_device_bytes_per_modelled_cycle


def test_example_carriers_are_labeled_examples_not_the_real_one():
    full = hwir.example_full_airflow_server_slot()
    low = hwir.example_constrained_low_power_slot()
    real = hwir.unpinned_real_carrier()
    assert full.example is True
    assert low.example is True
    assert real.example is False
    assert "NOT" in full.note or "not" in full.note
    assert "comma" in real.note.lower()
    assert full.carrier_id != real.carrier_id
    assert low.carrier_id != real.carrier_id
    assert hwir.select_carrier_envelope("full").carrier_id == full.carrier_id
    assert hwir.select_carrier_envelope("constrained").carrier_id == low.carrier_id
    assert hwir.select_carrier_envelope("unpinned").origin == "REAL_CARRIER_UNPINNED"


def test_no_hardware_measured_on_variants_or_carriers():
    for vid in hwir.U50_FAMILY_VARIANT_IDS:
        hwir.assert_no_hardware_measured(hwir.u50_family_profile(vid).to_dict())
    for factory in (
        hwir.example_full_airflow_server_slot,
        hwir.example_constrained_low_power_slot,
        hwir.unpinned_real_carrier,
    ):
        hwir.assert_no_hardware_measured(factory().to_dict())
    device = hwir.u50_family_profile("u50")
    low = hwir.example_constrained_low_power_slot()
    hwir.assert_no_hardware_measured(low.constrain(device).to_dict())
    assert hwir.CARRIER_ENVELOPE_BINDING is True


# ---------------------------------------------------------------------------
# comma Tiny Chestnut — the real inbound carrier, pinned 2026-09-02.
# ---------------------------------------------------------------------------


def test_chestnut_current_firmware_is_a_severe_downgrade_vs_a_full_slot():
    u50dd = hwir.u50_family_profile("u50dd")
    full = hwir.constrain_device_profile(u50dd, hwir.example_full_airflow_server_slot())
    chestnut = hwir.constrain_device_profile(u50dd, hwir.chestnut_current_firmware())
    assert chestnut.host_device_bytes_per_modelled_cycle < full.host_device_bytes_per_modelled_cycle


def test_observed_completer_ceiling_outranks_the_lane_derivation():
    """~1.68 GB/s is a completer property: x4 measures the same as x2.

    So it cannot be derived from lane count and must cap independently.
    """
    u50dd = hwir.u50_family_profile("u50dd")
    wide = hwir.CarrierEnvelope(
        carrier_id="wide-but-capped",
        origin="TEST",
        pcie_generation=4,
        pcie_lanes=4,
        sustained_power_w=75,
        airflow_class="forced",
        mechanical_limit=None,
        note="lane width says fast, completer says otherwise",
        observed_payload_beat=2,
        observed_payload_bytes_per_s=1_680_000_000,
    )
    assert hwir.constrain_device_profile(u50dd, wide).host_device_bytes_per_modelled_cycle == 2


def test_unpinned_chestnut_mode_does_not_fail_open():
    """An unmeasured carrier must not license MORE optimism than the measured one."""
    u50dd = hwir.u50_family_profile("u50dd")
    current = hwir.constrain_device_profile(u50dd, hwir.chestnut_current_firmware())
    unpinned = hwir.constrain_device_profile(u50dd, hwir.chestnut_hawking_optimized())
    assert unpinned.host_device_bytes_per_modelled_cycle <= current.host_device_bytes_per_modelled_cycle


def test_chestnut_provides_no_airflow_for_a_passive_card():
    assert hwir.u50_family_profile("u50dd").cooling == "passive"
    for make in hwir.CHESTNUT_MODES.values():
        assert make().airflow_class == "none"


def test_residency_ratio_separates_viable_from_doomed():
    assert hwir.residency_ratio(2 * 1024 ** 3, 4 * 1024 ** 2) > 100.0
    assert hwir.residency_ratio(int(1.5 * 1024 ** 3), 1024 ** 3) < 2.0
    with pytest.raises(ValueError):
        hwir.residency_ratio(1024, 0)


def test_chestnut_figures_are_third_party_reported_not_measured():
    for make in hwir.CHESTNUT_MODES.values():
        c = make()
        hwir.assert_no_hardware_measured(c.to_dict())
        for prov in dict(c.field_provenance).values():
            assert dict(prov)["tier"] == hwir.CHESTNUT_THIRD_PARTY
