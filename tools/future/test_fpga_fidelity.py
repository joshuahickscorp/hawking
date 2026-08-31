"""Pins for the FPGA multi-fidelity ladder and the four adaptation clocks."""
import json

import pytest

from tools.future import fpga_fidelity as ff
from tools.future._common import HARDWARE_FIELDS, HardwareClaimError, RECEIPTS, write_receipt


def test_build_emits_sealed_receipt():
    out = ff.selftest()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "FPGA_MULTIFIDELITY.json"
    assert doc["schema"] == "hawking.future.fpga_fidelity.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert doc["unimplemented_levels"][0]["status"] == "UNIMPLEMENTED"
    assert doc["unimplemented_levels"][1]["missing_dependency"] == "no U50 board"


def test_ladder_order_and_availability():
    levels = [p.level for p in ff.LADDER]
    assert levels == [
        ff.FidelityLevel.ANALYTICAL,
        ff.FidelityLevel.FUNCTIONAL_SIM,
        ff.FidelityLevel.CYCLE_MODEL,
        ff.FidelityLevel.HW_EMULATION,
        ff.FidelityLevel.REAL_HARDWARE,
    ]
    assert [p.available for p in ff.LADDER] == [True, True, True, False, False]
    assert ff.LADDER[3].status == "UNIMPLEMENTED"
    assert ff.LADDER[4].status == "UNIMPLEMENTED"
    assert ff.LADDER[3].missing_dependency == "no emulation seat"
    assert ff.LADDER[4].missing_dependency == "no U50 board"


def test_graph_must_declare_adaptation_clock():
    with pytest.raises(TypeError):
        ff.StructuralGraph(
            nodes=(ff.Node(id="n", kind="MAC", width=1, tile=1, banking=1, resource_class="DSP"),),
            edges=(),
        )
    with pytest.raises(ff.GraphError, match="adaptation_clock"):
        ff.StructuralGraph(
            graph_id="bad_clock",
            adaptation_clock=4,
            nodes=(ff.Node(id="n", kind="MAC", width=1, tile=1, banking=1, resource_class="DSP"),),
            edges=(),
        )


def test_analytical_emits_structural_counts_not_seconds():
    report = ff.estimate(ff.FidelityLevel.ANALYTICAL, ff.mixed_graph())
    assert report["tag"] == "STRUCTURAL_COUNTS"
    assert report["evidence_class"] == "STATIC_ONLY"
    res = report["resources"]
    for key in ("dsp", "lut", "bram", "uram"):
        assert isinstance(res[key], int)
        assert res[key] >= 0
    assert isinstance(report["hbm_channel_demand"], int)
    assert isinstance(report["bytes_in_flight"], int)
    assert report["bytes_in_flight"] == 256 * 16
    assert report["ii_by_node"]["gemv0"] == 1
    assert report["ii_by_node"]["gather0"] == 4 * 8
    assert report["pipeline_depth_by_node"]["gemv0"] >= 8
    assert report["feasibility"] == "UNKNOWN"
    assert report["seconds"] is None
    assert "bandwidth_gbps" not in report
    assert "gpu_ns" not in report


def test_cycle_model_is_modelled_not_measured_and_refuses_seconds():
    report = ff.estimate(ff.FidelityLevel.CYCLE_MODEL, ff.mixed_graph())
    assert report["tag"] == "MODELLED_NOT_MEASURED"
    assert isinstance(report["modelled_cycles"], int)
    assert report["modelled_cycles"] > 0
    assert report["seconds"] is None
    assert report["clock_hz"] == "UNKNOWN"
    assert report["conversion_to_seconds"] == "REFUSED"
    with pytest.raises(ff.UnmeasuredConversionError, match="cannot be converted"):
        ff.modelled_cycles_to_seconds(report["modelled_cycles"], clock_hz=300_000_000)


def test_functional_sim_kills_cyclic_and_flags_dead_nodes():
    cyclic = ff.estimate(ff.FidelityLevel.FUNCTIONAL_SIM, ff.cyclic_graph())
    assert cyclic["functional_ok"] is False
    assert cyclic["errors"]

    dead = ff.estimate(ff.FidelityLevel.FUNCTIONAL_SIM, ff.dead_node_graph())
    assert dead["functional_ok"] is True
    assert "dead0" in dead["dead_nodes"]
    assert "live0" in dead["live_nodes"]
    assert dead["node_ranking"][0] == "live0"


def test_compare_records_disagreement_as_finding_and_agreement_as_weak():
    mixed = ff.compare(ff.FidelityLevel.ANALYTICAL, ff.FidelityLevel.CYCLE_MODEL, ff.mixed_graph())
    assert mixed["kind"] == "DISAGREEMENT"
    assert mixed["evidence_weight"] == "FINDING"
    assert mixed["agree"] is False
    assert mixed["top_a"] != mixed["top_b"]
    assert mixed["top_a"] == "gemv0"
    assert mixed["top_b"] == "gather0"
    assert "must not promote" in mixed["hypothesis"]

    single = ff.compare(ff.FidelityLevel.ANALYTICAL, ff.FidelityLevel.CYCLE_MODEL, ff.gemv_graph())
    assert single["kind"] == "AGREEMENT"
    assert single["evidence_weight"] == "WEAK"
    assert single["agree"] is True

    liveness = ff.compare(
        ff.FidelityLevel.ANALYTICAL, ff.FidelityLevel.FUNCTIONAL_SIM, ff.dead_node_graph()
    )
    assert liveness["kind"] == "DISAGREEMENT"
    assert liveness["top_a"] == "dead0"
    assert liveness["top_b"] == "live0"


def test_graph_ranking_reverses_between_analytical_and_cycle_model():
    graphs = [ff.gemv_graph(), ff.gather_graph()]
    analytical = ff.rank_graphs(ff.FidelityLevel.ANALYTICAL, graphs)
    cycle = ff.rank_graphs(ff.FidelityLevel.CYCLE_MODEL, graphs)
    assert analytical == ["gather_heavy", "gemv_heavy"]
    assert cycle == ["gemv_heavy", "gather_heavy"]
    assert analytical != cycle


def test_search_ladder_hard_kills_only_overflow_and_ill_formed():
    result = ff.search_ladder(
        [ff.gemv_graph(), ff.gather_graph(), ff.overflow_graph(), ff.cyclic_graph()]
    )
    killed = {row["graph_id"]: row["reason"] for row in result["killed"]}
    assert killed["envelope_overflow"] == "INFEASIBLE_ENVELOPE"
    assert killed["cyclic"] == "FUNCTIONAL_ILL_FORMED"
    assert "gemv_heavy" in result["survivors"]
    assert "gather_heavy" in result["survivors"]
    skipped_levels = {row["level"] for row in result["skipped_unimplemented"]}
    assert skipped_levels == {"HW_EMULATION", "REAL_HARDWARE"}
    assert result["highest_available_level"] == "CYCLE_MODEL"


def test_adaptation_clocks_carry_required_fields():
    assert [c["id"] for c in ff.CLOCK_SPECS] == [0, 1, 2, 3]
    for spec in ff.CLOCK_SPECS:
        for key in (
            "id",
            "name",
            "identity",
            "compatibility_predicate",
            "load_switch_cost_class",
            "module_cache_key_rule",
            "resource_footprint_rule",
        ):
            assert key in spec and spec[key] is not None, f"clock {spec['id']} missing {key}"
    assert ff.CLOCK_SPECS[3]["live_switch_legal"] is False

    mixed = ff.mixed_graph()
    state = ff.clock_state(mixed)
    assert state["declared_on_graph"] == 2
    assert state["module_cache_key"]
    assert isinstance(state["resource_footprint"]["dsp"], int)

    parent = ff.gemv_graph(clock=0)
    same_module = ff.StructuralGraph(
        graph_id="alt_schedule",
        adaptation_clock=0,
        issue_count=parent.issue_count,
        architecture_id=parent.architecture_id,
        schedule_id="other",
        nodes=parent.nodes,
        edges=parent.edges,
    )
    assert ff.clock_compatible(parent, same_module) is True
    assert ff.clock_compatible(parent, ff.gather_graph(clock=0)) is False
    arch = ff.gemv_graph(clock=3)
    assert ff.clock_compatible(arch, arch) is False


def _assert_provider_raises(level, dependency, graph):
    with pytest.raises(ff.ProviderUnavailable) as ei:
        result = ff.estimate(level, graph)
        pytest.fail(f"{level} returned {result!r} instead of raising")
    assert ei.value.level is level
    assert ei.value.missing_dependency == dependency
    assert dependency in str(ei.value)


def test_hw_emulation_provider_raises_rather_than_returning_a_number():
    graph = ff.gemv_graph()
    _assert_provider_raises(ff.FidelityLevel.HW_EMULATION, "no emulation seat", graph)
    with pytest.raises(ff.ProviderUnavailable):
        ff.get_provider(ff.FidelityLevel.HW_EMULATION).estimate(graph)
    with pytest.raises(ff.ProviderUnavailable):
        ff.compare(ff.FidelityLevel.ANALYTICAL, ff.FidelityLevel.HW_EMULATION, graph)


def test_real_hardware_provider_raises_rather_than_returning_a_number():
    graph = ff.gemv_graph()
    _assert_provider_raises(ff.FidelityLevel.REAL_HARDWARE, "no U50 board", graph)
    with pytest.raises(ff.ProviderUnavailable):
        ff.get_provider("REAL_HARDWARE").estimate(graph)


def test_write_receipt_rejects_cycle_model_output_in_a_hardware_field():
    report = ff.estimate(ff.FidelityLevel.CYCLE_MODEL, ff.gemv_graph())
    cycles = report["modelled_cycles"]
    assert isinstance(cycles, int) and cycles > 0

    target = RECEIPTS / "FPGA_FIDELITY_SHOULD_NOT_SEAL.json"
    with pytest.raises(HardwareClaimError, match="gpu_ns") as ei:
        write_receipt(
            "FPGA_FIDELITY_SHOULD_NOT_SEAL.json",
            {
                "schema": ff.SCHEMA,
                "version": 1,
                "gpu_ns": cycles,
            },
            "tools/future/test_fpga_fidelity.py",
        )
    assert "sidecar has no GPU authority" in str(ei.value)
    assert not target.exists()

    with pytest.raises(HardwareClaimError, match="token_ns"):
        write_receipt(
            "FPGA_FIDELITY_SHOULD_NOT_SEAL.json",
            {
                "schema": ff.SCHEMA,
                "version": 1,
                "cycle_model": {"token_ns": cycles, "tag": "MODELLED_NOT_MEASURED"},
            },
            "tools/future/test_fpga_fidelity.py",
        )
    assert not target.exists()


def test_sealed_receipt_contains_no_hardware_field_numbers():
    doc = json.loads(ff.build().read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else key
                if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                    raise AssertionError(f"{here} = {value!r} is a hardware field number")
                walk(value, here)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(doc)
    assert doc["cycle_model"]["seconds"] is None
    assert isinstance(doc["cycle_model"]["modelled_cycles"], int)
