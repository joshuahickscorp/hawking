"""Pins for tools/future/fusion_sim.py.

The load-bearing negative control is
test_unknown_transport_yields_no_speedup_and_write_receipt_refuses_a_smuggled_number:
a topology whose critical path contains an UNKNOWN transport cost must not
emit a speedup figure, and write_receipt must actually refuse a document that
tries to smuggle one into a hardware field.
"""
from __future__ import annotations

import json

import pytest

from tools.future import fusion_sim as fs
from tools.future._common import RECEIPTS, HardwareClaimError, write_receipt


# --------------------------------------------------------------------------- helpers


def _present(name: str, kind: str) -> fs.Node:
    return fs.Node(id=name, kind=kind, present=True, missing_dependency=None)


def _absent(name: str, kind: str, missing: str) -> fs.Node:
    return fs.Node(id=name, kind=kind, present=False, missing_dependency=missing)


def _measured_link(a: str, b: str, bw: float, lat: float) -> fs.Link:
    # Mechanism tags: these citations prove the gate can open. They are not an
    # interconnect measurement of a board this machine does not have.
    return fs.Link(
        a, b,
        bandwidth=fs.measured(bw, fs.GENOME_RECEIPT, "mechanism_test_only.not_an_interconnect", "GB/s"),
        latency=fs.measured(lat, fs.GENOME_RECEIPT, "mechanism_test_only.not_an_interconnect", "s"),
    )


def _unknown_link(a: str, b: str) -> fs.Link:
    return fs.Link(a, b, bandwidth=fs.UNKNOWN_INTERCONNECT, latency=fs.UNKNOWN_INTERCONNECT)


def _measured_compute(seconds: float) -> fs.Cost:
    return fs.measured(seconds, fs.GENOME_RECEIPT, "mechanism_test_only.not_token_ns", "s")


# --------------------------------------------------------------------------- receipt / selftest


def test_build_emits_sealed_receipt():
    out = fs.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "FUSION_SIMULATION.json"
    assert doc["schema"] == "hawking.future.fusion_sim.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["default_simulation"]["timing_decidable"] is False
    assert doc["default_simulation"]["speedup"] is None
    assert "recovered_implementation" in doc
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["vocabulary"]["no_era_vi"] is True
    assert doc["vocabulary"]["no_odyssey_iv"] is True
    assert doc["vocabulary"]["fpga_is_not_a_civilization"] is True


def test_selftest_runs_and_emits_receipt():
    out = fs.selftest()
    assert out.name == "FUSION_SIMULATION.json"
    assert out.is_file()


# --------------------------------------------------------------------------- node model


def test_hawking_nodes_present_absent_and_missing_dependency():
    nodes = {n.id: n for n in fs.hawking_nodes()}
    assert set(nodes) == {"APPLE", "FPGA_U50", "CUDA_DGX", "EGPU", "MAC_ADDITIONAL"}
    assert nodes["APPLE"].present is True
    assert nodes["APPLE"].missing_dependency is None
    assert nodes["APPLE"].genome_receipt == fs.GENOME_RECEIPT
    for name in ("FPGA_U50", "CUDA_DGX", "EGPU", "MAC_ADDITIONAL"):
        assert nodes[name].present is False
        assert nodes[name].missing_dependency
    assert "U50" in nodes["FPGA_U50"].missing_dependency or "FPGA" in nodes["FPGA_U50"].missing_dependency
    assert "civilization" in nodes["FPGA_U50"].missing_dependency.lower()
    assert "NVIDIA" in nodes["CUDA_DGX"].missing_dependency or "CUDA" in nodes["CUDA_DGX"].missing_dependency
    assert "eGPU" in nodes["EGPU"].missing_dependency or "Thunderbolt" in nodes["EGPU"].missing_dependency
    assert "INSTANCE" in nodes["MAC_ADDITIONAL"].missing_dependency


def test_present_node_refuses_a_missing_dependency_and_absent_node_requires_one():
    with pytest.raises(fs.FusionSimError):
        fs.Node(id="X", kind="APPLE", present=True, missing_dependency="something")
    with pytest.raises(fs.FusionSimError):
        fs.Node(id="Y", kind="FPGA_U50", present=False, missing_dependency=None)


# --------------------------------------------------------------------------- cost honesty


def test_unknown_cost_cannot_carry_a_number():
    with pytest.raises(fs.HonestyError, match="unrepresentable"):
        fs.Cost(kind="UNKNOWN", value=12.0)
    with pytest.raises(fs.HonestyError, match="unrepresentable"):
        fs.Cost(kind="UNKNOWN", value=0)


def test_measured_cost_requires_a_receipts_path():
    with pytest.raises(fs.HonestyError, match="receipts/"):
        fs.Cost(kind="MEASURED", value=1.0, receipt="datasheet://u50")
    with pytest.raises(fs.HonestyError, match="numeric"):
        fs.Cost(kind="MEASURED", value=None, receipt=fs.GENOME_RECEIPT)


def test_emit_speedup_refuses_when_timing_is_not_decidable():
    with pytest.raises(fs.HonestyError, match="unrepresentable"):
        fs.emit_speedup(1.0, 2.0, timing_decidable=False)


# --------------------------------------------------------------------------- placement


def _star_unknown() -> fs.Topology:
    t = fs.Topology()
    t.add_node(_present("APPLE", "APPLE"))
    t.add_node(_absent("FPGA_U50", "FPGA_U50", "no board"))
    t.add_link(_unknown_link("APPLE", "FPGA_U50"))
    return t


def test_immutable_organ_replicates_but_absent_replicas_are_hypothetical():
    t = _star_unknown()
    objs = [fs.SemanticObject(
        "W", "IMMUTABLE_WEIGHTS", fs.Granularity.ORGAN, 1 << 20,
        home_hint="APPLE", consumers=("APPLE", "FPGA_U50"),
    )]
    p = fs.place_objects(t, objs)["W"]
    assert p.home == "APPLE"
    assert p.replicas == ("FPGA_U50",)
    assert p.real_replicas == ()
    assert p.hypothetical_replicas == ("FPGA_U50",)
    assert p.mutable is False


def test_immutable_tensor_is_not_eagerly_replicated():
    t = _star_unknown()
    objs = [fs.SemanticObject(
        "w_tile", "IMMUTABLE_WEIGHTS", fs.Granularity.TENSOR, 4096,
        home_hint="APPLE", consumers=("APPLE", "FPGA_U50"),
    )]
    p = fs.place_objects(t, objs)["w_tile"]
    assert p.replicas == ()
    assert "TENSOR" in p.reason


def test_kv_state_never_replicated():
    t = _star_unknown()
    objs = [fs.SemanticObject(
        "KV", "KV_STATE", fs.Granularity.LAYER_GROUP, 1 << 20,
        home_hint="APPLE", consumers=("APPLE", "FPGA_U50"),
    )]
    p = fs.place_objects(t, objs)["KV"]
    assert p.replicas == ()
    assert p.mutable is True
    assert "mutable" in p.reason


# --------------------------------------------------------------------------- move vs recompute


def test_move_vs_recompute_undecidable_when_transport_is_unknown():
    t = _star_unknown()
    q = fs.DependencyQuery(
        identity="act", home_domain="APPLE", need_domain="FPGA_U50",
        nbytes=1 << 16, memory_class="ACTIVATIONS",
        recompute_cost=_measured_compute(1e-6), recompute_legal=True,
    )
    plan = fs.plan_dependency(t, q)
    assert plan["action"] == "UNDECIDABLE"
    assert plan["choice_decidable"] is False
    actions = {o["action"] for o in plan["options"]}
    assert "MOVE_DATA" in actions
    assert "RECOMPUTE" in actions
    assert all(o["cost_s"] is None or not o["choice_decidable"] or o["action"] == "RECOMPUTE"
               for o in plan["options"] if o["action"] == "MOVE_DATA")
    move = next(o for o in plan["options"] if o["action"] == "MOVE_DATA")
    assert move["choice_decidable"] is False
    assert move["cost_s"] is None


def test_recompute_wins_when_both_costs_are_measured_and_recompute_is_cheaper():
    t = fs.Topology()
    t.add_node(_present("APPLE", "APPLE"))
    t.add_node(_present("MAC_B", "MAC"))
    t.add_link(_measured_link("APPLE", "MAC_B", bw=1.0, lat=1.0))
    q = fs.DependencyQuery(
        identity="act", home_domain="APPLE", need_domain="MAC_B",
        nbytes=1 << 30, memory_class="ACTIVATIONS",
        recompute_cost=_measured_compute(1e-6), recompute_legal=True,
    )
    plan = fs.plan_dependency(t, q)
    assert plan["choice_decidable"] is True
    assert plan["action"] == "RECOMPUTE"


def test_already_resident_is_structural_zero():
    t = fs.hawking_topology()
    q = fs.DependencyQuery(
        identity="kv", home_domain="APPLE", need_domain="APPLE",
        nbytes=1 << 20, memory_class="KV_STATE",
    )
    plan = fs.plan_dependency(t, q)
    assert plan["action"] == "ALREADY_RESIDENT"
    assert plan["choice_decidable"] is True


# --------------------------------------------------------------------------- THE NEGATIVE CONTROL


def test_unknown_transport_yields_no_speedup_and_write_receipt_refuses_a_smuggled_number():
    """A guard nobody has watched fail is not a guard.

    1. Critical path has UNKNOWN transport → timing_decidable False, speedup is
       not a number, emit_speedup refuses.
    2. write_receipt actually raises HardwareClaimError on a smuggled tps /
       bandwidth_gbps and does not leave a file behind.
    """
    t = fs.Topology()
    t.add_node(_present("APPLE", "APPLE"))
    t.add_node(_absent("FPGA_U50", "FPGA_U50", "physical U50 absent"))
    t.add_link(_unknown_link("APPLE", "FPGA_U50"))
    objects = [fs.SemanticObject(
        "act", "ACTIVATIONS", fs.Granularity.TENSOR, 1 << 16,
        home_hint="APPLE", consumers=("APPLE", "FPGA_U50"), recompute_legal=True,
    )]
    work = [
        fs.WorkItem("on_fpga", "COMPUTE", "FPGA_U50", 1 << 16, 1 << 16, (),
                    "ACTIVATIONS", "fpga_hwir",
                    compute_cost=_measured_compute(0.01), recompute_legal=True),
    ]
    result = fs.simulate(t, objects, work, baseline_time_s=1.0)
    assert result["timing_decidable"] is False
    assert result["speedup"] is None
    assert result["timing"] is None
    assert result["unknown_inputs_on_critical_path"]
    assert fs._numeric_speedup_keys(result) == []
    for key in fs.SPEEDUP_KEYS:
        value = result.get(key)
        assert not isinstance(value, (int, float))

    with pytest.raises(fs.HonestyError, match="unrepresentable"):
        fs.emit_speedup(0.01, 1.0, timing_decidable=result["timing_decidable"])

    smuggle_name = "FUSION_SIM_SMUGGLE_SHOULD_NOT_EXIST.json"
    target = RECEIPTS / smuggle_name
    if target.exists():
        target.unlink()
    with pytest.raises(HardwareClaimError, match="tps"):
        write_receipt(smuggle_name, {
            "schema": "hawking.future.fusion_sim.v1",
            "version": 1,
            "tps": 123.4,
        }, "test_fusion_sim.py")
    assert not target.exists(), "write_receipt must not leave a file after refusing tps"

    nested_name = "FUSION_SIM_SMUGGLE_NESTED_SHOULD_NOT_EXIST.json"
    nested = RECEIPTS / nested_name
    if nested.exists():
        nested.unlink()
    with pytest.raises(HardwareClaimError, match="bandwidth_gbps"):
        write_receipt(nested_name, {
            "schema": "hawking.future.fusion_sim.v1",
            "link": {"bandwidth_gbps": 64.0},
        }, "test_fusion_sim.py")
    assert not nested.exists()


def test_default_hawking_spread_is_undecidable_and_not_executable():
    sim = fs.simulate_default()
    assert sim["timing_decidable"] is False
    assert sim["speedup"] is None
    assert sim["executable"] is False
    assert sim["not_executable_reasons"]
    assert any("FPGA_U50" in r for r in sim["not_executable_reasons"])
    assert sim["honesty"]["produces_protected_absolute"] is False
    assert sim["honesty"]["produces_diagnostic_relative"] is False
    assert sim["collectives"][0]["algorithm"] == "UNDECIDABLE"
    assert sim["collectives"][0]["choice_decidable"] is False


def test_all_measured_inputs_can_open_the_gate():
    """Positive control: the refusal is not stuck closed."""
    t = fs.Topology()
    t.add_node(_present("APPLE", "APPLE"))
    t.add_node(_present("MAC_B", "MAC"))
    t.add_link(_measured_link("APPLE", "MAC_B", bw=10.0, lat=0.001))
    objects = [fs.SemanticObject(
        "act", "ACTIVATIONS", fs.Granularity.TENSOR, 1000,
        home_hint="APPLE", consumers=("APPLE", "MAC_B"),
        recompute_legal=False,
    )]
    work = [
        fs.WorkItem("here", "COMPUTE", "APPLE", 1000, 1000, (),
                    "ACTIVATIONS", "metal_f16", compute_cost=_measured_compute(0.02)),
        fs.WorkItem("there", "COMPUTE", "MAC_B", 1000, 1000, (),
                    "ACTIVATIONS", "metal_f16", compute_cost=_measured_compute(0.01)),
    ]
    result = fs.simulate(t, objects, work, baseline_time_s=1.0)
    assert result["timing_decidable"] is True
    assert result["timing"] is not None
    assert result["timing"]["total_time_s"] > 0
    assert result["timing"]["not_a_protected_measurement"] is True
    assert isinstance(result["speedup"], float)
    assert result["speedup"] == fs.emit_speedup(
        result["timing"]["total_time_s"], 1.0, timing_decidable=True
    )


# --------------------------------------------------------------------------- collectives, overlap, conversion


def test_collective_two_participants_is_direct_even_when_cost_unknown():
    t = _star_unknown()
    plan = fs.plan_collective(
        t, fs.CollectiveSpec(fs.CollectiveOp.BROADCAST, ("APPLE", "FPGA_U50"), 64)
    )
    assert plan["algorithm"] == "DIRECT"
    assert plan["choice_decidable"] is False
    assert plan["cost_s"] is None


def test_collective_three_participants_undecidable_with_unknown_costs():
    t = fs.hawking_topology()
    plan = fs.plan_collective(
        t, fs.CollectiveSpec(fs.CollectiveOp.ALLREDUCE, ("APPLE", "FPGA_U50", "CUDA_DGX"), 64)
    )
    assert plan["algorithm"] == "UNDECIDABLE"
    assert plan["choice_decidable"] is False
    assert plan["cost_s"] is None


def test_collective_three_measured_participants_picks_ring():
    """At p=3, fusion_planner's alpha/beta pair makes TREE never cheaper (d_alpha=0)."""
    t = fs.Topology()
    t.add_node(_present("A", "APPLE"))
    t.add_node(_present("B", "MAC"))
    t.add_node(_present("C", "MAC"))
    t.add_link(_measured_link("A", "B", bw=10.0, lat=0.001))
    t.add_link(_measured_link("B", "C", bw=10.0, lat=0.001))
    t.add_link(_measured_link("C", "A", bw=10.0, lat=0.001))
    plan = fs.plan_collective(
        t, fs.CollectiveSpec(fs.CollectiveOp.ALLREDUCE, ("A", "B", "C"), 1 << 20)
    )
    assert plan["choice_decidable"] is True
    assert plan["algorithm"] == "RING"
    assert isinstance(plan["cost_s"], float)
    assert plan["cost_s"] > 0


def test_overlap_is_structural_and_carries_no_hidden_time():
    work = fs.hawking_work()
    pairs = fs.overlap_structure(work)
    for p in pairs:
        assert p["hidden_time_s"] is None
        assert p["may_overlap"] is True


def test_representation_conversion_is_unknown_without_a_board():
    sim = fs.simulate_default()
    conv = sim["representation_conversion"]
    assert conv
    assert all(c["timing_decidable"] is False for c in conv)
    assert all(c["cost"]["kind"] == "UNKNOWN" for c in conv)


# --------------------------------------------------------------------------- CUDA seam


def test_lowering_is_backend_neutral_and_has_no_performance_number():
    lowered = fs.lower_fusion_op("SUBMIT", "cuda")
    assert lowered["performance"] is None
    assert lowered["timing_decidable"] is False
    assert "cuda:submit" == lowered["kernel_ref"]
    with pytest.raises(fs.FusionSimError):
        fs.lower_fusion_op("NOT_AN_OP", "cuda")


def test_differential_schema_forbids_local_cuda_on_apple():
    rec = {
        "workload_id": "toy",
        "metal_receipt": "receipts/headless/X.json",
        "cuda_receipt": "receipts/headless/Y.json",
        "machine_metal": {"kind": "APPLE"},
        "machine_cuda": {"kind": "APPLE", "soc": "Apple M3 Ultra"},
        "bench_state_metal": "UNKNOWN",
        "bench_state_cuda": "UNKNOWN",
        "measurement_class_metal": "STATIC_ONLY",
        "measurement_class_cuda": "STATIC_ONLY",
    }
    with pytest.raises(fs.HonestyError, match="Apple"):
        fs.validate_metal_cuda_differential(rec)
    rec["machine_cuda"] = {"kind": "CUDA_DGX"}
    rec["local_cuda_on_apple"] = True
    with pytest.raises(fs.HonestyError, match="FORBIDDEN"):
        fs.validate_metal_cuda_differential(rec)
    rec["local_cuda_on_apple"] = False
    rec["cuda_tps"] = 900.0
    with pytest.raises(fs.HonestyError, match="hardware number"):
        fs.validate_metal_cuda_differential(rec)


# --------------------------------------------------------------------------- fault / recovery


def test_vanish_apple_loses_mutable_kv_and_cannot_use_hypothetical_weight_replicas():
    sim = fs.simulate_default(vanish="APPLE")
    fault = sim["fault"]
    assert fault["outcome"] == "DEFINED"
    by_id = {o["identity"]: o for o in fault["objects"]}
    assert by_id["kv"]["outcome"] == "DATA_LOSS"
    assert by_id["weights"]["outcome"] == "DATA_LOSS"
    assert "HYPOTHETICAL" in by_id["weights"]["detail"]
    assert any(c["outcome"] == "ABORT_COLLECTIVE" for c in fault["collectives"])
    assert any(i["outcome"] == "ABORT" for i in fault["inflight"])


def test_vanish_already_absent_fpga_is_a_defined_noop():
    sim = fs.simulate_default(vanish="FPGA_U50")
    fault = sim["fault"]
    assert fault["outcome"] == "ALREADY_ABSENT"


def test_vanish_real_replica_recovers_immutable_from_survivor():
    t = fs.Topology()
    t.add_node(_present("APPLE", "APPLE"))
    t.add_node(_present("MAC_B", "MAC"))
    t.add_link(_unknown_link("APPLE", "MAC_B"))
    objects = [fs.SemanticObject(
        "W", "IMMUTABLE_WEIGHTS", fs.Granularity.ORGAN, 1 << 20,
        home_hint="APPLE", consumers=("APPLE", "MAC_B"),
    )]
    work = [fs.WorkItem("w", "COMPUTE", "MAC_B", 8, 8, (), "IMMUTABLE_WEIGHTS", "metal_f16")]
    placements = fs.place_objects(t, objects)
    assert placements["W"].real_replicas == ("MAC_B",)
    fault = fs.vanish_node(t, placements, objects, work, (), "APPLE")
    by_id = {o["identity"]: o for o in fault["objects"]}
    assert by_id["W"]["outcome"] == "RECOVERED_FROM_REPLICA"


def test_in_flight_transport_aborts_when_either_end_vanishes():
    t = fs.hawking_topology()
    objects = fs.hawking_objects()
    work = fs.hawking_work()
    placements = fs.place_objects(t, objects)
    fault = fs.vanish_node(
        t, placements, objects, work, fs.hawking_collectives(), "APPLE",
        inflight_edges=(("APPLE", "FPGA_U50", "activations"),),
    )
    assert fault["inflight"][0]["outcome"] == "ABORT"
