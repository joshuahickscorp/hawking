"""Tests for the physical primitive library.

Negative control (must actually fail if the guard is removed):
  * same semantic program at UMA vs HBM -> different identities
  * lowering to the CUDA seam RAISES, it does not return a plan
"""
from __future__ import annotations

import json

import pytest

from tools.future import physical_primitives as pp
from tools.future._common import HARDWARE_FIELDS, RECEIPTS


def test_build_emits_sealed_receipt():
    out = pp.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "PHYSICAL_PRIMITIVES.json"
    assert doc["schema"] == "hawking.future.physical_primitives.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert "Static sidecar artifact" in doc["claim_boundary"]


def test_selftest_is_build():
    assert pp.selftest is pp.build or pp.selftest() == pp.build()


def test_seventeen_atlas_primitives_in_atlas_order():
    assert len(pp.ATLAS_PRIMITIVES) == 17
    assert len(set(pp.ATLAS_PRIMITIVES)) == 17
    assert pp.ATLAS_PRIMITIVES[0] == "PersistentPhysicalRegion"
    assert pp.ATLAS_PRIMITIVES[-1] == "MemoryTierIdentity"
    assert "LayoutTransform" in pp.ATLAS_PRIMITIVES
    for name in pp.ATLAS_PRIMITIVES:
        spec = pp.CONTRACTS[name]
        assert spec["in_atlas"] is True
        for field in pp.REQUIRED_CONTRACT_FIELDS:
            assert field in spec, f"{name} missing {field}"
        assert spec["invariant"]
        assert spec["cost_removed"]["mechanism"]
        assert spec["cheapest_falsifier"]
        assert spec["preconditions"]
        assert spec["organ_classes"]
        assert spec["legal_memory_tiers"]
        assert set(spec["legal_memory_tiers"]) <= set(pp.MEMORY_TIERS)


def test_layout_transform_kept_atlas_name():
    # Directive omitted LayoutTransform. Atlas named it. Atlas wins.
    assert "LayoutTransform" in pp.ATLAS_PRIMITIVES
    assert pp.CONTRACTS["LayoutTransform"]["behavior_ids"] == ("layout_algebra",)


def test_verification_region_is_not_an_atlas_primitive():
    assert "VerificationRegion" not in pp.ATLAS_PRIMITIVES
    assert pp.CONTRACTS["VerificationRegion"]["in_atlas"] is False
    rec = pp.reconciliation()
    assert rec["left_out"] == []
    assert rec["directive_only"]["VerificationRegion"]["in_atlas"] is False
    assert rec["directive_only"]["VerificationRegion"]["disposition"] == (
        "IMPLEMENTED_OUTSIDE_ATLAS_SEVENTEEN"
    )


def test_conditional_physical_program_merges_both_atlas_entries():
    spec = pp.CONTRACTS["ConditionalPhysicalProgram"]
    assert "static_dynamic_skeleton" in spec["behavior_ids"]
    assert "npu_regular_island" in spec["behavior_ids"]
    rec = pp.reconciliation()
    assert "npu_regular_island -> ConditionalPhysicalProgram" in rec["merged_into_another"]


def test_behavior_taxonomy_has_twenty_one():
    assert len(pp.BEHAVIOR_TAXONOMY) == 21


def test_memory_tier_identity_uma_vs_hbm_differ():
    """NEGATIVE CONTROL: same semantic program, different tiers, different identity."""
    uma = pp.instantiate(
        "StationaryRepresentation",
        memory_tier="UMA",
        semantic_program_id="flash-complete-v2",
        backend="METAL",
        organ_class="mlp",
    )
    hbm = pp.instantiate(
        "StationaryRepresentation",
        memory_tier="HBM",
        semantic_program_id="flash-complete-v2",
        backend="METAL",
        organ_class="mlp",
    )
    assert uma.semantic_program_id == hbm.semantic_program_id
    assert uma.primitive == hbm.primitive
    assert uma.backend == hbm.backend
    assert uma.organ_class == hbm.organ_class
    assert uma.identity != hbm.identity
    # The MemoryTierIdentity primitive exists specifically to make this true.
    rule = pp.physical_identity(
        semantic_program_id="flash-complete-v2",
        primitive="MemoryTierIdentity",
        memory_tier="UMA",
        backend="METAL",
    )
    other = pp.physical_identity(
        semantic_program_id="flash-complete-v2",
        primitive="MemoryTierIdentity",
        memory_tier="HBM",
        backend="METAL",
    )
    assert rule != other


def test_same_tier_same_identity_is_stable():
    a = pp.physical_identity(
        semantic_program_id="p",
        primitive="GraphReplay",
        memory_tier="UMA",
        backend="METAL",
    )
    b = pp.physical_identity(
        semantic_program_id="p",
        primitive="GraphReplay",
        memory_tier="UMA",
        backend="METAL",
    )
    assert a == b
    assert len(a) == 64


def test_backend_is_also_an_identity_component():
    metal = pp.physical_identity(
        semantic_program_id="p",
        primitive="MoveOrRecompute",
        memory_tier="UMA",
        backend="METAL",
    )
    # Instantiating a CUDA identity is planning. Lowering it is what must raise.
    cuda = pp.physical_identity(
        semantic_program_id="p",
        primitive="MoveOrRecompute",
        memory_tier="UMA",
        backend="CUDA",
    )
    assert metal != cuda


def test_illegal_tier_raises():
    # SpatialPipeline's atlas mapping is spatial/local; UMA is not a legal occupant.
    with pytest.raises(pp.IllegalMemoryTierError):
        pp.instantiate(
            "SpatialPipeline",
            memory_tier="UMA",
            semantic_program_id="p",
        )
    with pytest.raises(pp.IllegalMemoryTierError):
        pp.physical_identity(
            semantic_program_id="p",
            primitive="StationaryRepresentation",
            memory_tier="REMOTE",
            backend="METAL",
        )
    with pytest.raises(pp.IllegalMemoryTierError):
        pp.normalize_tier("VRAM")


def test_unknown_primitive_raises():
    with pytest.raises(pp.UnknownPrimitiveError):
        pp.contract("CudaGraph")


def test_cuda_lowering_raises():
    """NEGATIVE CONTROL: the CUDA seam must actually refuse, not return a plan."""
    graph = pp.lower_nr_to_physical_graph(memory_tier="UMA", backend="METAL")
    with pytest.raises(pp.BackendUnavailableError) as ei:
        pp.lower_physical_graph_to_backend(graph, "CUDA")
    msg = str(ei.value)
    assert "CUDA" in msg or "NVIDIA" in msg
    assert "Apple" in msg or "NVIDIA hardware" in msg
    assert pp.SEAMS["CUDA"]["availability"] == "UNAVAILABLE"
    assert pp.SEAMS["CUDA"]["missing_dependency"]
    # Convenience path must refuse too.
    with pytest.raises(pp.BackendUnavailableError):
        pp.lower_nr_to_backend(backend="CUDA", memory_tier="UMA")


def test_ane_lowering_raises():
    graph = pp.lower_nr_to_physical_graph()
    with pytest.raises(pp.BackendUnavailableError) as ei:
        pp.lower_physical_graph_to_backend(graph, "ANE")
    assert "MLProgram" in str(ei.value) or "MLComputePlan" in str(ei.value)
    assert pp.SEAMS["ANE"]["availability"] == "UNAVAILABLE"


def test_metal_lowering_returns_planned_plan_not_a_measurement():
    graph = pp.lower_nr_to_physical_graph(memory_tier="UMA", backend="METAL")
    plan = pp.lower_physical_graph_to_backend(graph, "METAL")
    assert plan["availability"] == "PLANNED"
    assert plan["qualification"] == "PLAN_ONLY"
    assert plan["measurement_state"] == "STATIC_ONLY"
    assert plan["gpu_authority"] is False
    assert plan["backend"] == "METAL"
    assert plan["primitive_realizations"]
    assert "tps" not in plan


def test_fpga_lowering_is_a_seam_not_a_civilization():
    graph = pp.lower_nr_to_physical_graph(memory_tier="HBM", backend="FPGA")
    plan = pp.lower_physical_graph_to_backend(graph, "FPGA")
    assert plan["availability"] == "PLANNED"
    assert plan["qualification"] == "PLAN_ONLY"
    assert "not a civilization" in plan["not_a_civilization"].lower() or (
        "not an FPGA backend" in plan["not_a_civilization"]
    )


def test_nr_lowering_emits_physical_graph_shape():
    graph = pp.lower_nr_to_physical_graph(memory_tier="UMA", backend="METAL")
    assert graph["schema"] == "hcli.physical_graph.v1"
    assert graph["qualification"] == "PLAN_ONLY"
    assert graph["execution_policy"]["memory_tier_is_executable_identity"] is True
    assert "routed_experts" in graph["nr_families"]
    assert any(n["primitive"] == "DirectRoutedAccumulate" for n in graph["computation"])
    assert any(n["primitive"] == "FusedDecodeCompute" for n in graph["computation"])
    assert any(n["primitive"] == "MemoryTierIdentity" for n in graph["computation"])
    assert graph["fingerprint"]
    # Same inputs, same fingerprint (no wall-clock in hashed content).
    again = pp.lower_nr_to_physical_graph(memory_tier="UMA", backend="METAL")
    assert again["fingerprint"] == graph["fingerprint"]
    other_tier = pp.lower_nr_to_physical_graph(memory_tier="HBM", backend="METAL")
    assert other_tier["fingerprint"] != graph["fingerprint"]


def test_sidecar_does_not_propagate_codex_hardware_numbers():
    # The loaded NR may be Codex's real receipt, which legitimately carries
    # measured values -- Codex HAS hardware authority. What must never happen is
    # this sidecar copying one of those numbers into its own receipt, because the
    # sidecar has no GPU and a quoted measurement would read as a claimed one.
    nr_blob = json.dumps(pp.load_nr())
    doc = json.loads(pp.build().read_text())
    out_blob = json.dumps(doc)
    for leaky in ("mean_lookup_ns", "parameter_count"):
        if leaky in nr_blob:
            assert leaky not in out_blob, f"{leaky} leaked from the source NR into our receipt"
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["gpu_authority"] is False


def test_receipt_reconciliation_covers_all_seventeen():
    doc = json.loads(pp.build().read_text())
    rec = doc["reconciliation"]
    assert rec["atlas_count"] == 17
    assert set(rec["per_primitive"]) == set(pp.ATLAS_PRIMITIVES)
    assert rec["left_out"] == []
    assert set(rec["implemented"]) == set(pp.ATLAS_PRIMITIVES)
    assert doc["identity_rule"]["negative_control"]["identities_differ"] is True
    assert doc["backend_lowering"]["cuda_raises"] is True
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["recovered_implementation"]["atlas"]["backend_neutral_primitives"] == list(
        pp.ATLAS_PRIMITIVES
    )


def _walk_hardware_numbers(node, path=""):
    hits = []
    if isinstance(node, dict):
        for k, v in node.items():
            here = f"{path}.{k}" if path else k
            if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                hits.append((here, v))
            hits.extend(_walk_hardware_numbers(v, here))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            hits.extend(_walk_hardware_numbers(v, f"{path}[{i}]"))
    return hits


def test_receipt_contains_no_hardware_numbers():
    doc = json.loads(pp.build().read_text())
    assert _walk_hardware_numbers(doc) == []


def test_unknown_backend_raises():
    with pytest.raises(pp.UnknownBackendError):
        pp.normalize_backend("TPU")


def test_organ_class_refusal():
    with pytest.raises(pp.PrimitiveError):
        pp.instantiate(
            "DirectRoutedAccumulate",
            memory_tier="UMA",
            semantic_program_id="p",
            organ_class="normalization",
        )
