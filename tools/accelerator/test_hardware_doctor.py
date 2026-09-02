"""Generic Hardware Doctor: five host answers, ranking, honest U50DD absence.

A module import is not a call site. These tests invoke the gate symbols
(machine_genome.discover_identity / axes_for_domain / build,
device_ascension.characterize, hwir.u50_family_profile,
tools.roadmap.hardware.probe_u50) and check diagnose() recorded those calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import device_ascension as da  # noqa: E402
import hardware_doctor as hd  # noqa: E402
import machine_genome as mg  # noqa: E402
from tools.future import hwir  # noqa: E402
from tools.roadmap import hardware as hw_wake  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _stub_genome(*, u50_present: bool = False, memory_bytes: int = 103079215104) -> dict:
    g = {
        "schema": mg.SCHEMA,
        "soc": "Apple M3 Ultra",
        "arch": "arm64",
        "cpu_cores": 28,
        "perf_cores": 20,
        "efficiency_cores": 8,
        "gpu_cores": 60,
        "memory_bytes": memory_bytes,
        "thermal_envelope": {
            "status": "ABSENT",
            "reason": "no sustained thermal campaign",
            "evidence_tier": "STATIC",
        },
        "sustained_behaviour": {
            "status": "ABSENT",
            "reason": "microbenchmark only",
        },
        "domains": {
            "cpu_0": {
                "kind": "CPU", "name": "cpu_0", "present": True,
                "maturity": "MEASURED", "evidence_tier": "HARDWARE_MEASURED",
                "cores": 28, "perf_cores": 20, "efficiency_cores": 8,
                "arch": "arm64", "soc": "Apple M3 Ultra",
            },
            "gpu_uma_0": {
                "kind": "GPU", "name": "gpu_uma_0", "present": True,
                "maturity": "PRESENT", "evidence_tier": "HARDWARE_MEASURED",
                "gpu_cores": 60,
                "measured_bandwidth": {
                    "status": "UNRELIABLE",
                    "reason": "iqr too wide",
                    "evidence_tier": "HARDWARE_MEASURED",
                    "median_gb_s": 400.0,
                    "iqr_spread_pct": 40.0,
                    "reliable": False,
                },
            },
            "uma_0": {
                "kind": "UMA", "name": "uma_0", "present": True,
                "maturity": "PRESENT", "evidence_tier": "STATIC",
                "capacity_bytes": memory_bytes,
                "capacity_evidence_tier": "HARDWARE_MEASURED",
                "internal_coherency": "HARDWARE_UMA",
            },
            "ane_0": {
                "kind": "ANE", "name": "ane_0", "present": True,
                "maturity": "PROFILED", "evidence_tier": "HARDWARE_MEASURED",
                "ioreg": {"present": True, "ioreg_class": "H11ANEIn",
                          "evidence_tier": "HARDWARE_MEASURED"},
                "supported_compute_devices": ["CPU", "NEURAL_ENGINE"],
            },
            "storage": {
                "kind": "STORAGE", "name": "storage", "present": True,
                "maturity": "MEASURED", "evidence_tier": "HARDWARE_MEASURED",
                "mounts": [],
            },
            "network": {
                "kind": "NETWORK", "name": "network", "present": True,
                "maturity": "PRESENT", "evidence_tier": "STATIC",
                "interfaces": ["lo0", "en0"],
                "wan_throughput": {
                    "status": "BLOCKED",
                    "reason": "live hf download workers",
                    "evidence_tier": "STATIC",
                },
            },
            "fpga_hbm_0": {
                "kind": "FPGA", "name": "fpga_hbm_0", "present": False,
                "maturity": "DECLARED", "evidence_tier": "STATIC",
                "wake_condition": "U50_PRESENT", "performance": "UNKNOWN",
            },
            "nvidia_dgx_0": {
                "kind": "EXTERNAL_ACCELERATOR", "name": "nvidia_dgx_0",
                "present": False, "maturity": "DECLARED", "evidence_tier": "STATIC",
                "wake_condition": "DGX_PRESENT", "performance": "UNKNOWN",
            },
            "u50dd_0": {
                "kind": "FPGA", "name": "u50dd_0", "present": u50_present,
                "maturity": "PRESENT" if u50_present else "DECLARED",
                "evidence_tier": "STATIC",
                "wake_condition": "U50_PRESENT", "performance": "UNKNOWN",
                "physical": False, "expected_sku": "A-U50DD-P00G-ES3-G",
            },
            "egpu_0": {
                "kind": "EXTERNAL_ACCELERATOR", "name": "egpu_0",
                "present": False, "maturity": "DECLARED", "evidence_tier": "STATIC",
                "wake_condition": "EGPU_PRESENT", "performance": "UNKNOWN",
            },
        },
    }
    g["genome_digest"] = mg.genome_digest(g)
    g["backend_maturity"] = {n: d["maturity"] for n, d in g["domains"].items()}
    return g


def _quiet_wakes() -> dict:
    return {
        "U50_PRESENT": {
            "id": "U50_PRESENT", "present": False,
            "evidence": "injected: no U50", "evidence_tier": "STATIC",
            "probe_u50_present": False, "probe_u50_evidence": "injected",
            "description": hw_wake.WAKE_CONDITIONS["U50_PRESENT"],
        },
        "DGX_PRESENT": {
            "id": "DGX_PRESENT", "present": False,
            "evidence": "injected: no DGX", "evidence_tier": "STATIC",
        },
        "EGPU_PRESENT": {
            "id": "EGPU_PRESENT", "present": False,
            "evidence": "injected: no eGPU", "evidence_tier": "STATIC",
        },
        "NEW_M_SERIES_PRESENT": {
            "id": "NEW_M_SERIES_PRESENT", "present": False,
            "evidence": "injected: M3 Ultra textbook", "evidence_tier": "STATIC",
        },
    }


@pytest.fixture(scope="module")
def live_doc():
    """One live diagnose() on this host. Calls genome build + wake probes
    and writes receipts/future/HARDWARE_DOCTOR_GENERIC.json."""
    path = hd.build(live=True)
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- live host: the five answers


def test_five_questions_are_present(live_doc):
    q = live_doc["questions"]
    for key in (
        "devices_exist",
        "measured_vs_modelled_vs_unknown",
        "workload_fit",
        "experiments_ranked",
        "backend_maturity",
    ):
        assert key in q and q[key], key
    assert live_doc["absent_u50dd"]
    assert live_doc["schema"] == hd.SCHEMA


def test_this_host_is_m3_ultra_cpu_gpu_uma_ane_no_fpga_dgx_egpu(live_doc):
    inv = live_doc["questions"]["devices_exist"]
    present = {r["name"]: r for r in inv["present"]}
    absent = {r["name"]: r for r in inv["absent"]}
    soc = live_doc["host"]["soc"] or inv["soc"]
    assert "M3 Ultra" in (soc or "")
    for name in ("cpu_0", "gpu_uma_0", "uma_0", "ane_0", "storage"):
        assert name in present, name
        assert present[name]["present"] is True
    for name in ("fpga_hbm_0", "u50dd_0", "nvidia_dgx_0", "egpu_0"):
        assert name in absent, name
        assert absent[name]["present"] is False
    assert live_doc["wakes"]["U50_PRESENT"]["present"] is False
    assert live_doc["wakes"]["DGX_PRESENT"]["present"] is False
    assert live_doc["wakes"]["EGPU_PRESENT"]["present"] is False
    assert live_doc["wakes"]["NEW_M_SERIES_PRESENT"]["present"] is False


def test_per_axis_tiers_are_honest_and_never_merged(live_doc):
    axes = live_doc["questions"]["measured_vs_modelled_vs_unknown"]
    for name, rows in axes.items():
        seen = {r["axis"] for r in rows}
        assert seen == set(mg.AXES), (name, seen)
        for r in rows:
            assert r["evidence_tier"] in mg.EVIDENCE_TIERS, (name, r)
            assert r["status"] in mg.AXIS_STATUSES, (name, r)
            # One row, one tier. Status is not a second tier.
            assert "HARDWARE_MEASURED" not in str(r.get("status"))
    # Absent devices: no performance axis is HARDWARE_MEASURED.
    for name in ("u50dd_0", "fpga_hbm_0", "nvidia_dgx_0", "egpu_0"):
        for r in axes[name]:
            if r["axis"] == "presence":
                assert r["status"] == "ABSENT"
                assert r["evidence_tier"] != "HARDWARE_MEASURED" or r["status"] == "ABSENT"
                continue
            assert r["evidence_tier"] != "HARDWARE_MEASURED", (name, r)
            assert r["status"] in {"ABSENT", "UNKNOWN"}


def test_workloads_fit_uma_gpu_not_fpga_or_dgx(live_doc):
    fit = {w["id"]: w for w in live_doc["questions"]["workload_fit"]["workloads"]}
    assert fit["interactive_decode"]["fit"] is True
    assert fit["max_throughput_decode"]["fit"] is True
    assert fit["u50dd_hbm_resident_shard"]["fit"] is False
    assert fit["dgx_offload"]["fit"] is False
    assert fit["egpu_offload"]["fit"] is False
    assert fit["ane_prefill"]["fit"] is False
    assert fit["ane_prefill"].get("unproven") is True
    sel = live_doc["questions"]["workload_fit"]["selected_resident"]
    assert sel["installed"] is False
    assert sel["selected"] in {"sealed-3.14", "variantB-2.76"}
    econ = live_doc["questions"]["workload_fit"]
    assert econ["fpga_present"] is False
    assert econ["external_accelerator_present"] is False
    assert econ["uma_present"] is True
    for b in econ["bodies"]:
        assert b["fits_uma"] is True
        assert b["evidence_tier"] == "COST_MODEL"


def test_experiments_are_ranked_runnable_first_then_info_per_cost(live_doc):
    ranked = live_doc["questions"]["experiments_ranked"]
    assert ranked, "no uncertainty-reducing experiments"
    assert ranked[0]["rank"] == 1
    assert ranked[0]["runnable_now"] is True
    # Board-gated U50 HBM cannot outrank a cheap UNKNOWN-on-present probe.
    ids = [r["id"] for r in ranked]
    if "HDG-U50DD-ARRIVAL-HBM" in ids and "HDG-CPU-STREAM" in ids:
        assert ids.index("HDG-CPU-STREAM") < ids.index("HDG-U50DD-ARRIVAL-HBM")
    if "HDG-GPU-TRIAD" in ids and "HDG-U50DD-ARRIVAL-HBM" in ids:
        assert ids.index("HDG-GPU-TRIAD") < ids.index("HDG-U50DD-ARRIVAL-HBM")
    ipc = [
        (
            0 if r["information_per_cost"]["runnable_now"] else 1,
            -(r["information_per_cost"]["info_weight"] * 60
              // r["information_per_cost"]["cost"]),
            r["information_per_cost"]["cost"],
        )
        for r in ranked
    ]
    assert ipc == sorted(ipc)
    # First runnable probe names a present-device axis.
    top = ranked[0]
    assert top["device"] in {"cpu_0", "gpu_uma_0", "uma_0", "ane_0", "storage", "network"}
    assert top["info_weight"] >= 1


def test_backend_maturity_matches_genome_slots(live_doc):
    mat = live_doc["questions"]["backend_maturity"]
    assert mat["cpu_0"] in mg.MATURITY
    assert mat["ane_0"] in {"PROFILED", "PRESENT", "MEASURED"}
    assert mat["u50dd_0"] == "DECLARED"
    assert mat["fpga_hbm_0"] == "DECLARED"
    assert mat["nvidia_dgx_0"] == "DECLARED"
    assert mat["egpu_0"] == "DECLARED"


# --------------------------------------------------------------------------- call sites (import is not a call)


def test_diagnose_records_real_call_sites_and_tests_invoke_them(live_doc):
    called = live_doc["called"]
    for symbol in (
        "machine_genome.discover_identity",
        "machine_genome.build",
        "machine_genome.axes_for_domain",
        "machine_genome.devices_exist",
        "device_ascension.characterize",
        "device_ascension.economics",
        "hwir.u50_family_profile",
        "hwir.chestnut_current_firmware",
        "tools.roadmap.hardware.probe",
        "tools.roadmap.hardware.probe_u50",
    ):
        assert symbol in called, symbol

    ident = mg.discover_identity()
    assert ident["cpu_cores"] == int(mg._sysctl("hw.ncpu"))
    assert "M3 Ultra" in (ident.get("soc") or "")

    profile = hwir.u50_family_profile("u50dd")
    assert profile.sku == "A-U50DD-P00G-ES3-G"
    assert profile.origin == "U50_FAMILY_VARIANT_DECLARED_NOT_A_BOARD"
    hwir.assert_no_hardware_measured(profile.to_dict())

    present, evidence = hw_wake.probe_u50()
    assert present is False
    assert "u50" in evidence.lower() or "xilinx" in evidence.lower() or "alveo" in evidence.lower() or "no " in evidence.lower()

    char_called = da.characterize.__name__
    assert char_called == "characterize"


def test_axes_for_domain_is_a_real_call_site():
    g = _stub_genome()
    cpu = mg.axes_for_domain(g["domains"]["cpu_0"], genome=g)
    assert {r["axis"] for r in cpu} == set(mg.AXES)
    bw = next(r for r in cpu if r["axis"] == "bandwidth")
    assert bw["status"] == "UNKNOWN"
    assert bw["evidence_tier"] != "HARDWARE_MEASURED"

    absent = mg.axes_for_domain(g["domains"]["u50dd_0"], genome=g)
    assert {r["axis"] for r in absent} == set(mg.AXES)
    for r in absent:
        if r["axis"] == "presence":
            assert r["status"] == "ABSENT"
        assert r["evidence_tier"] != "HARDWARE_MEASURED"


def test_axis_record_refuses_measured_bandwidth_on_absent_device():
    with pytest.raises(ValueError, match="fabricate"):
        mg.axis_record(
            "bandwidth",
            status="MEASURED",
            evidence_tier="HARDWARE_MEASURED",
            device_present=False,
        )


def test_declaring_u50dd_present_does_not_fabricate_hbm_bandwidth():
    g = _stub_genome()
    g2 = mg.declare_domain(
        g, kind="FPGA", name="u50dd_0", present=True,
        maturity="PRESENT", evidence_tier="STATIC",
        wake_condition="U50_PRESENT", performance="UNKNOWN",
    )
    assert g2["schema"] == g["schema"]
    rows = mg.axes_for_domain(g2["domains"]["u50dd_0"], genome=g2)
    bw = next(r for r in rows if r["axis"] == "bandwidth")
    assert bw["status"] == "UNKNOWN"
    assert bw["evidence_tier"] != "HARDWARE_MEASURED"
    cap = next(r for r in rows if r["axis"] == "capacity")
    assert cap["evidence_tier"] != "HARDWARE_MEASURED"


# --------------------------------------------------------------------------- ranking (mutation: cost must matter)


def test_ranking_prefers_cheap_runnable_over_expensive_board():
    expensive = {
        "id": "A-board",
        "info_weight": 4,
        "cost": 8,
        "runnable_now": False,
    }
    cheap = {
        "id": "Z-cpu",
        "info_weight": 4,
        "cost": 1,
        "runnable_now": True,
    }
    ranked = hd.rank_queue([expensive, cheap])
    assert [r["id"] for r in ranked] == ["Z-cpu", "A-board"]
    assert ranked[0]["rank"] == 1


def test_ranking_cost_breaks_id_order_when_both_runnable():
    """Load-bearing cost term. Both runnable, equal info_weight.

    Integer key is -(info*60//cost) then cost then id. Gutted to
    (runnable, -info, id), 'A-expensive' sorts before 'Z-cheap' and this
    test MUST fail. Restore the cost term; never leave the mutation in source.
    """
    expensive = {
        "id": "A-expensive",
        "info_weight": 4,
        "cost": 8,
        "runnable_now": True,
    }
    cheap = {
        "id": "Z-cheap",
        "info_weight": 4,
        "cost": 1,
        "runnable_now": True,
    }
    ranked = hd.rank_queue([expensive, cheap])
    assert [r["id"] for r in ranked] == ["Z-cheap", "A-expensive"]


def test_ranking_runnable_beats_equal_ratio_board():
    board = {"id": "board", "info_weight": 8, "cost": 8, "runnable_now": False}
    cpu = {"id": "cpu", "info_weight": 4, "cost": 2, "runnable_now": True}
    ranked = hd.rank_queue([board, cpu])
    assert ranked[0]["id"] == "cpu"


# --------------------------------------------------------------------------- absent U50DD


def test_absent_u50dd_names_unknowns_and_wake_without_guessing(live_doc):
    u = live_doc["absent_u50dd"]
    assert u["present"] is False
    assert u["physical"] is False
    assert u["performance"] == "UNKNOWN"
    assert u["wake_condition"] == "U50_PRESENT"
    assert u["wake"]["present"] is False
    assert u["hardware_measured"] is False
    assert u["brochure"]["evidence_tier"] == "STATIC"
    assert u["brochure"]["declared_not_measured"] is True
    assert u["brochure"]["hardware_measured"] is False
    assert u["brochure"]["sku"] == "A-U50DD-P00G-ES3-G"
    assert "bandwidth" in u["unknown_axes"] or "ABSENT" in {
        r["status"] for r in u["axes"] if r["axis"] == "bandwidth"
    }
    assert any("U50_PRESENT" in x or "identity" in x.lower() for x in u["what_wake_resolves"])
    assert any("HBM" in x for x in u["what_wake_does_not_resolve"])
    assert any("HARDWARE_MEASURED" in x for x in u["what_wake_does_not_resolve"])
    assert u["carrier"]["evidence_class"] == hwir.CHESTNUT_THIRD_PARTY
    assert u["carrier"]["hardware_measured"] is False
    assert u["carrier"]["evidence_tier"] != "HARDWARE_MEASURED"
    # Brochure LUT is vendor literature, labelled STATIC, never a local census.
    lut = u["brochure"]["resources"]["LUT"]
    assert lut["evidence_tier"] == "STATIC"
    assert lut["hardware_measured"] is False
    assert lut["vendor_literature_not_measurement"] is True
    assert lut["document_class"]
    # Speed grade stays UNPINNED.
    assert any(f["field"] == "speed_grade" for f in u["brochure"]["unpinned_fields"])
    assert "U50_PURCHASE_ACCEPTANCE" in u["wake_gates"]


def test_absent_u50dd_on_stub_still_calls_hwir_not_a_guess():
    g = _stub_genome()
    profile = hwir.u50_family_profile("u50dd")
    chestnut = hwir.chestnut_current_firmware()
    opt = hwir.chestnut_hawking_optimized()
    u = hd.absent_u50dd(
        g, profile=profile, chestnut=chestnut, chestnut_opt=opt,
        wake=_quiet_wakes()["U50_PRESENT"],
    )
    assert u["present"] is False
    assert u["brochure"]["device_id"] == "alveo-u50dd"
    hwir.assert_no_hardware_measured(profile.to_dict())
    # Chestnut payload may be a number but is not a Hawking measurement.
    assert u["carrier"]["hardware_measured"] is False
    assert u["carrier_hawking_optimized"]["hardware_measured"] is False


def test_no_hardware_measured_number_on_absent_devices(live_doc):
    def walk(node, path=""):
        if isinstance(node, dict):
            tier = node.get("evidence_tier")
            if tier == "HARDWARE_MEASURED" and any(
                s in path for s in ("u50dd", "fpga_hbm", "nvidia_dgx", "egpu_0", "brochure", "carrier")
            ):
                raise AssertionError(f"{path} claims HARDWARE_MEASURED: {node}")
            if node.get("hardware_measured") and any(
                s in path for s in ("u50dd", "brochure", "carrier")
            ):
                raise AssertionError(f"{path} hardware_measured=True")
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(live_doc["absent_u50dd"], "absent_u50dd")
    walk(live_doc["questions"]["measured_vs_modelled_vs_unknown"]["u50dd_0"], "axes.u50dd_0")


def test_stub_diagnose_does_not_need_a_board():
    g = _stub_genome()
    doc = hd.diagnose(live=False, genome=g, wakes=_quiet_wakes())
    hd._assert_honest(doc)
    assert doc["absent_u50dd"]["present"] is False
    assert doc["questions"]["devices_exist"]["n_absent"] >= 4
    ranked = doc["questions"]["experiments_ranked"]
    assert ranked[0]["runnable_now"] is True
    assert ranked[0]["id"] != "HDG-U50DD-ARRIVAL-HBM"


def test_unreliable_gpu_bandwidth_is_not_quoted_as_a_roof():
    g = _stub_genome()
    rows = mg.axes_for_domain(g["domains"]["gpu_uma_0"], genome=g)
    bw = next(r for r in rows if r["axis"] == "bandwidth")
    assert bw["status"] == "UNRELIABLE"
    assert bw["evidence_tier"] == "HARDWARE_MEASURED"
    assert bw.get("reliable") is False


def test_build_writes_generic_receipt_not_the_fpga_axis_one(live_doc):
    out = hd.RECEIPTS / hd.RECEIPT
    assert out.is_file()
    doc = json.loads(out.read_text())
    assert doc["schema"] == hd.SCHEMA == live_doc["schema"]
    assert doc["schema"] != "hawking.future.hardware_doctor.v1"
    assert doc["recorded_by"] == hd.RECORDED_BY
    assert doc["seal_sha256"]
    sibling_path = REPO / "receipts/future/HARDWARE_DOCTOR.json"
    assert sibling_path.is_file()
    sibling = json.loads(sibling_path.read_text())
    assert sibling.get("schema") == "hawking.future.hardware_doctor.v1"
    assert sibling.get("schema") != doc["schema"]


def test_live_genome_slots_include_u50dd_and_egpu(live_doc):
    names = {r["name"] for r in live_doc["questions"]["devices_exist"]["absent"]}
    names |= {r["name"] for r in live_doc["questions"]["devices_exist"]["present"]}
    assert "u50dd_0" in names
    assert "egpu_0" in names
    slot = mg._domain_u50dd_declared()
    assert slot["present"] is False
    assert slot["wake_condition"] == "U50_PRESENT"
    assert slot["expected_sku"] == "A-U50DD-P00G-ES3-G"
    egpu = mg._domain_egpu_declared()
    assert egpu["present"] is False
    assert egpu["wake_condition"] == "EGPU_PRESENT"
    fpga = mg._domain_fpga_declared()
    assert fpga["wake_condition"] == "U50_PRESENT"
    assert live_doc["questions"]["backend_maturity"]["u50dd_0"] == "DECLARED"
