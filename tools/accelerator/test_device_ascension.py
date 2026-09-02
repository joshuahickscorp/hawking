"""Device Ascension pins. This module had NO tests, which is exactly why a dict class
attribute inside a @dataclass shipped broken and 46 unrelated tests still passed.

Also the live MachineGenome / arrival-cycle pins: this M3 Ultra is the first
real architecture profile. A fixture (the future-pipeline M4 Pro bag: 14 CPU /
20 GPU / 51539607552 bytes) must not satisfy these assertions.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/odyssey"))
import device_ascension as da  # noqa: E402
import machine_genome as mg  # noqa: E402
import device_profiles as dp  # noqa: E402

FIXTURE_M4_PRO = {
    "soc": "Apple M4 Pro",
    "cpu_cores": 14,
    "gpu_cores": 20,
    "memory_bytes": 51539607552,
}


def test_the_module_actually_imports_and_constructs():
    """The regression that got through: @dataclass rejects a mutable class attribute."""
    assert da.Ascension({"soc": "X"}) is not None


def test_a_profile_cannot_be_sealed_without_the_stages_it_depends_on():
    a = da.Ascension({"soc": "X"})
    a.record("sustained_qualification", {"passed": True})
    a.not_run("concurrency_sweep", "not run")
    assert a.seal("SUSTAINED", {}, "EXACT_MACHINE")["status"] == "SEALED"
    m = a.seal("MAX_THROUGHPUT", {}, "EXACT_MACHINE")
    assert m["status"] == "PROVISIONAL"
    assert "concurrency_sweep" in m["why"]


def test_sustained_evidence_is_required_for_a_production_seal():
    a = da.Ascension({"soc": "X"})
    a.not_run("sustained_qualification", "microbenchmark only")
    assert a.seal("SUSTAINED", {}, "EXACT_MACHINE")["status"] == "PROVISIONAL"


def test_failed_sustained_does_not_seal():
    a = da.Ascension({"soc": "X"})
    a.record("sustained_qualification", {"passed": False})
    assert a.seal("SUSTAINED", {}, "EXACT_MACHINE")["status"] == "PROVISIONAL"


def test_an_unnamed_profile_is_refused():
    a = da.Ascension({"soc": "X"})
    with pytest.raises(ValueError, match="not a named profile"):
        a.seal("FAST_MODE", {}, "EXACT_MACHINE")


def test_the_two_vocabularies_are_distinct():
    """§26 tuning scope and §80 knowledge level are different questions."""
    import receipt
    a = da.Ascension({"soc": "X"})
    a.record("sustained_qualification", {"passed": True})
    with pytest.raises(ValueError, match="tuning scope"):
        a.seal("SUSTAINED", {}, "INSTANCE")          # a §80 level, not a §26 scope
    assert "EXACT_MACHINE" not in receipt.KNOWLEDGE_LEVELS
    assert "INSTANCE" not in da.TUNING_SCOPES


def test_a_stage_outside_the_eleven_is_refused():
    a = da.Ascension({"soc": "X"})
    with pytest.raises(ValueError, match="not one of the eleven"):
        a.record("vibes", {})


# --------------------------------------------------------------------------- arrival cycle


@pytest.fixture(scope="module")
def live_cycle():
    """One live discover->...->invalidate run. Selection is not an install."""
    return da.run_cycle()


def test_ascension_cycle_runs_end_to_end_and_emits_a_selection(live_cycle):
    r = live_cycle
    assert r["cycle"] == list(da.CYCLE)
    assert r["installed"] is False
    sel = r["selection"]
    assert sel["installed"] is False
    assert sel["selected"] in {"sealed-3.14", "variantB-2.76"}
    assert sel["genome_digest"]
    assert r["promotion"]["installed"] is False
    assert r["promotion"]["status"] == "PROVISIONAL"
    # Same genome: the just-emitted selection still holds.
    assert r["invalidation"]["invalidated"] is False
    assert r["invalidation"]["recompile_required"] is False
    # ADP fingerprint was filled from the live genome; remaining stages NOT_RUN.
    assert r["adp"]["fingerprint"]["soc"]
    assert "fingerprint" in r["adp"]["stages_completed"]
    assert "adp_seal" in r["adp"]["stages_not_run"]
    # Call sites are named, not implied by an import.
    assert r["stages"]["discover"]["called"] == "machine_genome.discover_identity"
    assert r["stages"]["characterize"]["called"] == "machine_genome.build"
    assert r["stages"]["economics"]["called"] == "device_profiles.economics_from_genome"
    assert r["stages"]["select"]["called"] == "device_profiles.select_resident"
    assert r["invalidation"]["called"] == "genome_changed"


def test_a_genome_change_invalidates_a_prior_selection(live_cycle):
    """The recompile trigger. A selection bound to genome A must not survive B."""
    import copy
    import machine_genome as mg

    sel = live_cycle["stages"]["select"]
    genome = live_cycle["genome"]
    hold = da.invalidate(sel, genome)
    assert hold["invalidated"] is False

    mutated = copy.deepcopy(genome)
    mutated["gpu_cores"] = 40
    mutated["genome_digest"] = mg.genome_digest(mutated)
    broken = da.invalidate(sel, mutated)
    assert broken["invalidated"] is True
    assert broken["recompile_required"] is True
    assert broken["prior_genome_digest"] != broken["current_genome_digest"]
    # genome_changed is the function the mutation check patches.
    assert da.genome_changed(sel["decision"]["genome"], mutated) is True
    assert da.genome_changed(sel["decision"]["genome"], genome) is False


def test_cycle_does_not_install_a_resident(live_cycle):
    assert live_cycle["installed"] is False
    assert live_cycle["promotion"]["installed"] is False
    assert live_cycle["selection"]["installed"] is False
    assert "copied" not in (live_cycle["promotion"].get("note") or "").lower() \
        or "not an installation" in live_cycle["promotion"]["note"]


# --------------------------------------------------------------------------- live genome (this M3 Ultra)


@pytest.fixture(scope="module")
def live_genome(live_cycle):
    return live_cycle["genome"]


def test_live_genome_cpu_gpu_uma_are_this_m3_ultra_not_a_fixture(live_genome):
    g = live_genome
    sysctl_cpu = int(mg._sysctl("hw.ncpu"))
    sysctl_mem = int(mg._sysctl("hw.memsize"))
    sysctl_soc = mg._sysctl("machdep.cpu.brand_string")
    assert g["soc"] == sysctl_soc
    assert "M3 Ultra" in g["soc"]
    assert g["cpu_cores"] == sysctl_cpu
    assert g["memory_bytes"] == sysctl_mem
    assert isinstance(g["gpu_cores"], int) and g["gpu_cores"] > 0
    assert g["soc"] != FIXTURE_M4_PRO["soc"]
    assert g["cpu_cores"] != FIXTURE_M4_PRO["cpu_cores"]
    assert g["gpu_cores"] != FIXTURE_M4_PRO["gpu_cores"]
    assert g["memory_bytes"] != FIXTURE_M4_PRO["memory_bytes"]
    uma = g["domains"]["uma_0"]
    assert uma["present"] is True
    assert uma["capacity_bytes"] == sysctl_mem
    assert uma["internal_coherency"] == "HARDWARE_UMA"
    assert g["domains"]["cpu_0"]["cores"] == sysctl_cpu
    assert g["domains"]["gpu_uma_0"]["gpu_cores"] == g["gpu_cores"]


def test_live_genome_storage_includes_real_corpdrive_mount(live_genome):
    g = live_genome
    storage = g["domains"]["storage"]
    mounts = {m["mount"]: m for m in storage["mounts"]}
    assert "/Volumes/corpdrive" in mounts
    corp = mounts["/Volumes/corpdrive"]
    assert corp["exists"] is True
    cap = corp["capacity"]["capacity_bytes"]
    live = os.statvfs("/Volumes/corpdrive")
    assert cap == live.f_frsize * live.f_blocks
    assert cap > 1 << 40
    assert corp["capacity"]["evidence_tier"] == "HARDWARE_MEASURED"
    assert storage["corpdrive"]["wrote"] is False
    write = corp.get("write_probe") or {}
    assert write.get("status") == "BLOCKED"
    seq = corp.get("sequential") or {}
    assert seq.get("evidence_tier") in mg.EVIDENCE_TIERS or seq.get("status") == "BLOCKED"
    if seq.get("path"):
        assert str(seq["path"]).startswith("/Volumes/corpdrive")
        assert "/partial/" not in str(seq["path"])
    meta = corp.get("metadata") or {}
    assert meta.get("evidence_tier") == "HARDWARE_MEASURED" or meta.get("status") == "BLOCKED"
    rnd = corp.get("random") or {}
    assert rnd.get("evidence_tier") in mg.EVIDENCE_TIERS or rnd.get("status") == "BLOCKED"


def test_ane_comes_from_the_lab_receipt_not_invented_fields(live_genome):
    ane = live_genome["domains"]["ane_0"]
    assert ane["present"] is True
    assert ane["neural_engine_present"] is True
    assert ane["ioreg"]["present"] is True
    assert ane["ioreg"]["evidence_tier"] == "HARDWARE_MEASURED"
    plan = ane.get("mlcomputeplan") or {}
    ops = plan.get("operations") or []
    if ops:
        supported = ops[0].get("supported") or []
        assert "NEURAL_ENGINE" in supported or "CPU" in supported
    if ane.get("lab_receipt"):
        assert "FORBIDDEN_FRUIT" in ane["lab_receipt"] or "ANE_DEVICE_PROFILE" in ane["lab_receipt"]


def test_fpga_and_dgx_are_declared_absent_slots(live_genome):
    g = live_genome
    fpga = g["domains"]["fpga_hbm_0"]
    dgx = g["domains"]["nvidia_dgx_0"]
    assert fpga["kind"] == "FPGA"
    assert fpga["present"] is False
    assert fpga["maturity"] == "DECLARED"
    assert fpga["evidence_tier"] == "STATIC"
    assert dgx["kind"] == "EXTERNAL_ACCELERATOR"
    assert dgx["present"] is False
    assert dgx["maturity"] == "DECLARED"
    assert "measurement" in (dgx.get("note") or "").lower() or dgx["performance"] == "UNKNOWN"


def test_future_fpga_and_dgx_declare_without_a_schema_change(live_genome):
    g = live_genome
    schema = g["schema"]
    g2 = mg.declare_domain(
        g, kind="FPGA", name="u50_0", present=False,
        product="Alveo U50", evidence_tier="STATIC",
        note="declared slot; no U50 is attached. COST_MODEL, not a measurement.",
    )
    g3 = mg.declare_domain(
        g2, kind="EXTERNAL_ACCELERATOR", name="dgx_b200_0", present=False,
        product_family="NVIDIA_DGX", evidence_tier="STATIC",
    )
    assert g3["schema"] == schema == g["schema"]
    assert g3["domains"]["u50_0"]["kind"] == "FPGA"
    assert g3["domains"]["dgx_b200_0"]["kind"] == "EXTERNAL_ACCELERATOR"
    assert g3["domains"]["u50_0"]["present"] is False
    assert g3["domains"]["dgx_b200_0"]["present"] is False
    assert g3["genome_digest"] != g["genome_digest"]
    assert schema == "hawking.accelerator.machine_genome.v2"


def test_every_domain_carries_an_honest_evidence_tier(live_genome):
    for name, d in live_genome["domains"].items():
        assert d["evidence_tier"] in mg.EVIDENCE_TIERS, name
        assert d["maturity"] in mg.MATURITY, name
        assert d["kind"]


def test_bandwidth_is_absent_or_measured_never_fabricated(live_genome):
    bw = live_genome["measured_bandwidth"]
    if bw.get("status") in {"ABSENT", "BLOCKED", "UNRELIABLE"}:
        assert "reason" in bw
        assert "median_gb_s" not in bw or bw.get("status") == "UNRELIABLE"
    else:
        assert bw.get("evidence_tier") == "HARDWARE_MEASURED"
        assert bw.get("median_gb_s")


def test_digest_ignores_rate_jitter_but_sees_identity_change(live_genome):
    import copy
    a = copy.deepcopy(live_genome)
    b = copy.deepcopy(live_genome)
    storage = b["domains"]["storage"]
    for m in storage.get("mounts") or []:
        seq = m.get("sequential")
        if isinstance(seq, dict) and seq.get("cold_gb_s"):
            seq["cold_gb_s"] = seq["cold_gb_s"] * 1.5
    assert mg.genome_digest(a) == mg.genome_digest(b)
    b["gpu_cores"] = 40
    assert mg.genome_digest(a) != mg.genome_digest(b)


def test_discover_identity_is_a_real_call_site():
    ident = mg.discover_identity()
    assert ident["cpu_cores"] == int(mg._sysctl("hw.ncpu"))
    assert ident["evidence_tier"] == "STATIC"


def test_probe_storage_is_a_real_call_site():
    s = mg.probe_storage()
    assert s["kind"] == "STORAGE"
    assert any(m.get("mount") == "/Volumes/corpdrive" for m in s["mounts"])


def test_probe_ane_is_a_real_call_site():
    a = mg.probe_ane()
    assert a["kind"] == "ANE"
    assert a["present"] is True


# --------------------------------------------------------------------------- device-profile economics (call sites)


def _stub_genome(*, fpga_present: bool = False, dgx_present: bool = False,
                 memory_bytes: int = 103079215104) -> dict:
    g = {
        "schema": mg.SCHEMA,
        "soc": "Apple M3 Ultra",
        "arch": "arm64",
        "cpu_cores": 28,
        "perf_cores": 20,
        "efficiency_cores": 8,
        "gpu_cores": 60,
        "memory_bytes": memory_bytes,
        "domains": {
            "cpu_0": {"kind": "CPU", "name": "cpu_0", "present": True, "maturity": "MEASURED",
                      "evidence_tier": "HARDWARE_MEASURED"},
            "gpu_uma_0": {"kind": "GPU", "name": "gpu_uma_0", "present": True, "maturity": "PRESENT",
                          "evidence_tier": "HARDWARE_MEASURED", "gpu_cores": 60},
            "uma_0": {"kind": "UMA", "name": "uma_0", "present": True, "maturity": "PRESENT",
                      "evidence_tier": "STATIC", "capacity_bytes": memory_bytes},
            "ane_0": {"kind": "ANE", "name": "ane_0", "present": True, "maturity": "PROFILED",
                      "evidence_tier": "HARDWARE_MEASURED"},
            "storage": {"kind": "STORAGE", "name": "storage", "present": True, "maturity": "MEASURED",
                        "evidence_tier": "HARDWARE_MEASURED", "mounts": []},
            "fpga_hbm_0": {"kind": "FPGA", "name": "fpga_hbm_0", "present": fpga_present,
                           "maturity": "DECLARED", "evidence_tier": "STATIC"},
            "nvidia_dgx_0": {"kind": "EXTERNAL_ACCELERATOR", "name": "nvidia_dgx_0",
                             "present": dgx_present, "maturity": "DECLARED",
                             "evidence_tier": "STATIC"},
        },
    }
    g["genome_digest"] = mg.genome_digest(g)
    return g


def test_economics_from_genome_is_cost_model_over_measured_capacity():
    g = _stub_genome()
    econ = dp.economics_from_genome(g, profile="INTERACTIVE")
    assert econ["evidence_tier"] == "COST_MODEL"
    assert econ["uma_bytes"] == g["memory_bytes"]
    assert econ["uma_present"] is True
    assert econ["fpga_present"] is False
    assert econ["genome_digest"] == g["genome_digest"]
    ids = {b["id"] for b in econ["bodies"]}
    assert ids == {"sealed-3.14", "variantB-2.76"}
    for b in econ["bodies"]:
        assert b["fits_uma"] is True


def test_select_resident_is_a_decision_not_an_install():
    g = _stub_genome()
    econ = dp.economics_from_genome(g, profile="INTERACTIVE")
    decision = dp.select_resident(econ, profile="INTERACTIVE")
    assert decision["installed"] is False
    assert decision["selected"] == "sealed-3.14"
    maxx = dp.select_resident(econ, profile="MAXX")
    assert maxx["installed"] is False
    assert maxx["selected"] == "variantB-2.76"


def test_fpga_required_candidate_is_refused_until_the_domain_is_present():
    """A future FPGA resident slots in by flipping present=True. No schema change."""
    fpga_body = {
        "id": "fpga-kernel-resident",
        "resident_bytes": 1 << 20,
        "requires_domain_kind": "FPGA",
    }
    absent = _stub_genome(fpga_present=False)
    econ = dp.economics_from_genome(absent)
    refused = dp.select_resident(econ, candidates=[fpga_body], profile="INTERACTIVE")
    assert refused["selected"] is None
    assert any("FPGA" in (r.get("refused") or "") for r in refused["refused"])

    present = _stub_genome(fpga_present=True)
    assert present["schema"] == absent["schema"]
    econ2 = dp.economics_from_genome(present)
    admitted = dp.select_resident(econ2, candidates=[fpga_body], profile="INTERACTIVE")
    assert admitted["selected"] == "fpga-kernel-resident"
    assert admitted["installed"] is False


def test_dgx_required_candidate_is_refused_on_this_host():
    dgx_body = {
        "id": "dgx-resident",
        "resident_bytes": 1 << 20,
        "requires_domain_kind": "EXTERNAL_ACCELERATOR",
    }
    g = _stub_genome(dgx_present=False)
    econ = dp.economics_from_genome(g)
    decision = dp.select_resident(econ, candidates=[dgx_body], profile="MAXX")
    assert decision["selected"] is None
    assert any("EXTERNAL_ACCELERATOR" in (r.get("refused") or "") for r in decision["refused"])
