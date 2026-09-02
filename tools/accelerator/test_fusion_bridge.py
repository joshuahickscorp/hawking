"""Fusion Bridge pins.

Required coverage:
  - two-domain plan (host GPU_UMA = Apple GPU/UMA, declared FPGA_HBM)
    constructs and validates
  - NEGATIVE: coherency assumed across a non-coherent link is REJECTED
  - every cost is labeled COST_MODEL; no path emits HARDWARE_MEASURED
  - generic layer has no vendor-keyed control flow
  - readback correctness is not device-compute visibility
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import fusion_bridge as fb  # noqa: E402
import fusion_planner as fp  # noqa: E402
import placement as pl  # noqa: E402
import semantic_transport as st  # noqa: E402
from fusion_isa import FusionOp  # noqa: E402
from humf import MemoryClass  # noqa: E402
from semantic_transport import (  # noqa: E402
    COST_MODEL,
    HARDWARE_MEASURED,
    CoherencyAssumption,
    ComputeVisibilityEvidence,
    CostLabelError,
    DomainKind,
    DomainVisibility,
    ExecutionDomain,
    OwnershipTransfer,
    PayloadSemantics,
    SemanticTransportEdge,
    SyncRequirement,
    cost_model,
    store_and_forward_cost,
)

ACCEL = Path(__file__).resolve().parent
GENERIC = (
    ACCEL / "semantic_transport.py",
    ACCEL / "placement.py",
    ACCEL / "fusion_bridge.py",
)
VENDOR_NEEDLES = ("apple", "xilinx", "nvidia", "cuda", "metal")


# ---------------------------------------------------------------- two-domain happy path


def test_two_domain_apple_gpu_plus_declared_fpga_hbm_validates():
    """Acceptance happy path. gpu_uma_0 is the Apple GPU/UMA domain;
    fpga_hbm_0 is a declared FPGA HBM domain that does not exist on this
    machine."""
    plan = fb.two_domain_plan()
    report = fb.validate(plan)
    assert report.ok, report.errors
    assert set(plan.domains) == {fb.HOST_GPU_UMA, fb.DECLARED_FPGA_HBM}
    assert plan.domains[fb.HOST_GPU_UMA].kind is DomainKind.GPU_UMA
    assert plan.domains[fb.HOST_GPU_UMA].physical is True
    assert plan.domains[fb.DECLARED_FPGA_HBM].kind is DomainKind.FPGA_HBM
    assert plan.domains[fb.DECLARED_FPGA_HBM].physical is False
    assert any(e.link_coherency is CoherencyAssumption.NONE for e in plan.edges)
    assert plan.qualification == "PLAN_ONLY"


def test_cli_two_domain_plan_exits_0():
    r = subprocess.run(
        [sys.executable, str(ACCEL / "fusion_bridge.py")],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr or r.stdout
    assert "ok" in r.stdout
    assert "cost_label=COST_MODEL" in r.stdout


# ---------------------------------------------------------------- NEGATIVE: coherency overclaim


def _overclaim_plan(assumption: CoherencyAssumption) -> fb.HeterogeneousPlan:
    good = fb.two_domain_plan()
    bad_edges = tuple(fb.overclaiming_edge(e, assumption) for e in good.edges)
    return fb.HeterogeneousPlan(
        domains=good.domains,
        placements=good.placements,
        edges=bad_edges,
        object_placements=good.object_placements,
        notes=good.notes,
    )


def test_noncoherent_link_rejects_coherency_assumption():
    """THE load-bearing test. Roadmap §16.1 / §16.2: a plan that assumes
    coherency across a link declared NONE must be refused. Mutation of
    _reject_coherency_overclaim must make this FAIL."""
    plan = _overclaim_plan(CoherencyAssumption.SOFTWARE_MANAGED)
    assert plan.edges[0].link_coherency is CoherencyAssumption.NONE
    assert plan.edges[0].coherency_assumption is CoherencyAssumption.SOFTWARE_MANAGED
    report = fb.validate(plan)
    assert report.ok is False
    assert "COHERENCY_OVERCLAIM" in report.codes()


def test_noncoherent_link_rejects_hardware_uma_assumption():
    plan = _overclaim_plan(CoherencyAssumption.HARDWARE_UMA)
    report = fb.validate(plan)
    assert report.ok is False
    assert "COHERENCY_OVERCLAIM" in report.codes()
    assert "HARDWARE_UMA_ACROSS_DOMAINS" in report.codes()


def test_compose_refuses_an_overclaiming_plan():
    good = fb.two_domain_plan()
    bad_edges = tuple(
        fb.overclaiming_edge(e, CoherencyAssumption.OWNERSHIP_VERSIONED)
        for e in good.edges
    )
    with pytest.raises(fb.FusionBridgeError, match="COHERENCY_OVERCLAIM"):
        fb.compose(good.domains, good.placements, bad_edges)


def test_weaker_assumption_on_a_stronger_link_is_legal():
    """Assuming NONE on a SOFTWARE_MANAGED link is conservative, not an
    overclaim. The check is one-sided."""
    good = fb.two_domain_plan()
    edge = good.edges[0]
    conservative = SemanticTransportEdge(
        source=edge.source, destination=edge.destination,
        payload_semantics=edge.payload_semantics,
        ownership_transfer=edge.ownership_transfer,
        sync_requirement=edge.sync_requirement,
        coherency_assumption=CoherencyAssumption.NONE,
        link_coherency=CoherencyAssumption.SOFTWARE_MANAGED,
        cost=edge.cost,
        in_transit_transforms=edge.in_transit_transforms,
        object_id=edge.object_id, organ_id=edge.organ_id,
    )
    plan = fb.compose(good.domains, good.placements, (conservative,),
                      object_placements=good.object_placements)
    assert fb.validate(plan).ok is True


def test_matching_none_none_is_legal():
    plan = fb.two_domain_plan()
    edge = plan.edges[0]
    assert edge.coherency_assumption is CoherencyAssumption.NONE
    assert edge.link_coherency is CoherencyAssumption.NONE
    assert edge.assumes_stronger_than_link is False


# ---------------------------------------------------------------- COST_MODEL / HARDWARE_MEASURED


def test_every_cost_on_the_demo_plan_is_cost_model():
    plan = fb.two_domain_plan()
    assert plan.total_cost().label == COST_MODEL
    for edge in plan.edges:
        assert edge.cost.label == COST_MODEL
    overlay = fb.overlay_physical_graph({"device_placement": {}, "synchronization": [],
                                         "dependencies": []}, plan)
    assert overlay["device_placement"]["fusion_bridge"]["cost"]["label"] == COST_MODEL


def test_cost_constructor_refuses_hardware_measured():
    with pytest.raises(CostLabelError, match="HARDWARE_MEASURED"):
        st.Cost(time_s=1.0, nbytes=1, bandwidth_gb_s=1.0, latency_s=0.0,
                label=HARDWARE_MEASURED)


def test_cost_model_helper_always_stamps_cost_model():
    c = cost_model(time_s=0.1, nbytes=8, bandwidth_gb_s=12.0, latency_s=1e-4)
    assert c.label == COST_MODEL
    c2 = store_and_forward_cost(nbytes=1 << 20, bandwidth_gb_s=12.0, latency_s=2.5e-4)
    assert c2.label == COST_MODEL
    assert c2.time_s == pytest.approx((1 << 20) / (12.0 * 1e9) + 2.5e-4)


def test_route_wrapper_does_not_promote_planner_measured_to_hardware_measured():
    """fusion_planner stamps MEASURED on all-physical hops. The bridge must
    still emit COST_MODEL, never HARDWARE_MEASURED."""
    t = fp.Topology()
    t.add_domain("A", physical=True)
    t.add_domain("B", physical=True)
    t.add_link("A", "B", bandwidth_gb_s=100.0, latency_s=0.0, physical=True)
    route = t.shortest_path("A", "B", 1024)
    assert route.cost_provenance == "MEASURED"
    cost = fb.cost_from_route(route)
    assert cost.label == COST_MODEL
    assert cost.label != HARDWARE_MEASURED
    assert cost.all_hops_physical is True


def test_no_assignment_of_hardware_measured_as_a_cost_label():
    """Static: generic modules never assign HARDWARE_MEASURED to a label
    field. The token may appear in a refusal path."""
    for path in GENERIC:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "label":
                if isinstance(node.value, ast.Name) and node.value.id == "HARDWARE_MEASURED":
                    pytest.fail(f"{path.name} assigns HARDWARE_MEASURED to label")
                if (isinstance(node.value, ast.Constant)
                        and node.value.value == HARDWARE_MEASURED):
                    pytest.fail(f"{path.name} assigns HARDWARE_MEASURED to label")


# ---------------------------------------------------------------- §16.1 readback vs compute visibility


def test_readback_is_not_device_compute_visibility():
    good = fb.two_domain_plan()
    poisoned = ExecutionDomain(
        name=fb.DECLARED_FPGA_HBM,
        kind=DomainKind.FPGA_HBM,
        physical=False,
        visibility=DomainVisibility(
            readback=True,
            device_compute=True,
            device_compute_evidence=ComputeVisibilityEvidence.READBACK,
        ),
        internal_coherency=CoherencyAssumption.SOFTWARE_MANAGED,
    )
    domains = dict(good.domains)
    domains[fb.DECLARED_FPGA_HBM] = poisoned
    plan = fb.HeterogeneousPlan(
        domains=domains, placements=good.placements, edges=good.edges,
    )
    report = fb.validate(plan)
    assert report.ok is False
    assert "READBACK_IS_NOT_COMPUTE_VISIBILITY" in report.codes()


def test_fpga_hbm_cannot_declare_internal_hardware_uma():
    with pytest.raises(st.SemanticTransportError, match="HARDWARE_UMA"):
        ExecutionDomain(
            name="x", kind=DomainKind.FPGA_HBM, physical=False,
            internal_coherency=CoherencyAssumption.HARDWARE_UMA,
        )


def test_cpu_and_npu_domains_are_expressible_without_a_new_branch():
    cpu = ExecutionDomain(
        name="cpu_0", kind=DomainKind.CPU, physical=True,
        visibility=DomainVisibility(
            readback=True, device_compute=True,
            device_compute_evidence=ComputeVisibilityEvidence.EXPLICIT_CONTRACT,
        ),
        internal_coherency=CoherencyAssumption.SOFTWARE_MANAGED,
    )
    npu = ExecutionDomain(
        name="npu_0", kind=DomainKind.NPU, physical=False,
        visibility=DomainVisibility(
            readback=True, device_compute=True,
            device_compute_evidence=ComputeVisibilityEvidence.EXPLICIT_CONTRACT,
        ),
        internal_coherency=CoherencyAssumption.KERNEL_BOUNDARY,
    )
    assert cpu.kind is DomainKind.CPU
    assert npu.kind is DomainKind.NPU


# ---------------------------------------------------------------- vendor neutrality of the generic layer


def _vendor_in(text: str) -> bool:
    low = text.lower()
    return any(v in low for v in VENDOR_NEEDLES)


def _vendor_control_flow(path: Path) -> list[str]:
    """Hits in *predicates* of If / IfExp / While / Match / Assert, not in
    bodies (a body that mentions fusion_isa MATERIALIZE is not a vendor
    branch)."""
    tree = ast.parse(path.read_text())
    hits: list[str] = []

    def scan(expr: ast.AST, lineno: int) -> None:
        for child in ast.walk(expr):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                if _vendor_in(child.value):
                    hits.append(f"{path.name}:{lineno}: string {child.value!r}")
            elif isinstance(child, ast.Name) and _vendor_in(child.id):
                hits.append(f"{path.name}:{lineno}: name {child.id}")
            elif isinstance(child, ast.Attribute) and _vendor_in(child.attr):
                hits.append(f"{path.name}:{lineno}: attr {child.attr}")

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            scan(node.test, node.lineno)
        elif isinstance(node, ast.IfExp):
            scan(node.test, node.lineno)
        elif isinstance(node, ast.While):
            scan(node.test, node.lineno)
        elif isinstance(node, ast.Assert):
            scan(node.test, node.lineno)
        elif isinstance(node, ast.Match):
            scan(node.subject, node.lineno)
            for case in node.cases:
                scan(case.pattern, case.pattern.lineno)
                if case.guard is not None:
                    scan(case.guard, case.pattern.lineno)
    return hits


def test_generic_layer_has_no_vendor_keyed_control_flow():
    hits = []
    for path in GENERIC:
        hits.extend(_vendor_control_flow(path))
    assert hits == []


# ---------------------------------------------------------------- connections to existing modules


def test_lifts_fusion_planner_object_home_placement_into_storage_nodes():
    topo = fb.declared_heterogeneous_topology()
    objects = [
        fp.SemanticObject(
            "W", MemoryClass.IMMUTABLE_WEIGHTS, fp.Granularity.ORGAN, 1 << 20,
            home_hint=fb.HOST_GPU_UMA,
            consumers=(fb.HOST_GPU_UMA, fb.DECLARED_FPGA_HBM),
        ),
        fp.SemanticObject(
            "KV", MemoryClass.KV_STATE, fp.Granularity.LAYER_GROUP, 64,
            home_hint=fb.HOST_GPU_UMA,
            consumers=(fb.HOST_GPU_UMA, fb.DECLARED_FPGA_HBM),
        ),
    ]
    lifted, raw = fb.lift_object_placements(topo, objects)
    by_id = {p.node_id: p for p in lifted}
    assert by_id["W"].node_kind is pl.NodeKind.STORAGE
    assert by_id["W"].domain == fb.HOST_GPU_UMA
    assert by_id["W"].replicas == (fb.DECLARED_FPGA_HBM,)
    assert by_id["KV"].replicas == ()  # mutable, fusion_planner never replicates
    assert {p.identity for p in raw} == {"W", "KV"}


def test_plan_to_timeline_emits_acquire_copy_fence_submit():
    plan = fb.two_domain_plan()
    tl = fb.plan_to_timeline(plan)
    ops = [c.op for c in tl.commands()]
    assert FusionOp.ACQUIRE_READ in ops
    assert FusionOp.COPY in ops
    assert FusionOp.FENCE in ops
    assert FusionOp.RELEASE in ops
    assert FusionOp.SUBMIT in ops


def test_overlay_physical_graph_fills_hcli_fields_without_editing_hcli():
    from hcli.physical_graph import SCHEMA, compile_physical_graph

    graph = compile_physical_graph({"model_id": "fusion-bridge-test", "organs": []})
    assert graph["schema"] == SCHEMA
    plan = fb.two_domain_plan()
    over = fb.overlay_physical_graph(graph, plan)
    assert over["device_placement"]["fusion_bridge"]["cost"]["label"] == COST_MODEL
    assert any(s.get("kind") == "semantic_transport" for s in over["synchronization"])
    assert any(d.get("kind") == "semantic_transport" for d in over["dependencies"])
    # original graph is a separate dict; overlay must not be required to
    # mutate it, and hcli/physical_graph.py is not imported for writing.
    assert "fusion_bridge" in over


def test_hwir_lowering_uses_the_existing_semantic_transport_edge_primitive():
    from tools.future import hwir

    assert hwir.PRIMITIVE_TO_NODE_KIND["SemanticTransportEdge"] == "dma-transport"
    plan = fb.two_domain_plan()
    graph = fb.lower_fpga_domain_to_hwir(plan)
    report = graph.validate()
    assert report.ok, report.errors
    dma = [n for n in graph.nodes if n.kind == "dma-transport"]
    assert dma, "expected a dma-transport node for the inbound edge"
    assert all(n.primitive == "SemanticTransportEdge" for n in dma)
    assert graph.device_budget.declared_not_measured is True


def test_domain_from_machine_genome_reads_capacity_and_does_not_measure():
    genome = {"schema": "hawking.accelerator.machine_genome.v1",
              "memory_bytes": 96 << 30, "gpu_cores": 40}
    dom = fb.domain_from_machine_genome(genome)
    assert dom.kind is DomainKind.GPU_UMA
    assert dom.capacity_bytes == 96 << 30
    assert dom.physical is True
    assert dom.visibility.device_compute_evidence is ComputeVisibilityEvidence.EXPLICIT_CONTRACT


def test_device_profiles_are_hints_not_domains():
    p = pl.Placement(
        node_id="decode", node_kind=pl.NodeKind.COMPUTATION,
        domain=fb.HOST_GPU_UMA, profile_hint="INTERACTIVE",
    )
    assert p.profile_hint == "INTERACTIVE"
    with pytest.raises(pl.PlacementError, match="profile_hint"):
        pl.Placement(
            node_id="decode", node_kind=pl.NodeKind.COMPUTATION,
            domain=fb.HOST_GPU_UMA, profile_hint="GPU",
        )


def test_odyssey_adapter_annotates_a_compiler_receipt():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "odyssey"))
    import fusion_bridge_adapter as adapter  # noqa: WPS433
    import physical_graph_compiler as pgc  # noqa: WPS433

    stub = {"transformation_dag": [{"stage": "source_graph"}], "pass": True}
    plan = fb.two_domain_plan()
    annotated = adapter.annotate_compiler_output(stub, plan)
    assert annotated["transformation_dag"][-1]["stage"] == "heterogeneous_fusion_bridge"
    assert annotated["transformation_dag"][-1]["cost_delta"]["label"] == COST_MODEL
    via_compiler = pgc.attach_heterogeneous_plan(stub, plan.to_dict())
    assert via_compiler["transformation_dag"][-1]["cost_delta"]["label"] == COST_MODEL
    assert adapter.plan_is_valid() is True


def test_resources_exceeding_declared_capacity_are_rejected():
    good = fb.two_domain_plan()
    tight = ExecutionDomain(
        name=fb.DECLARED_FPGA_HBM, kind=DomainKind.FPGA_HBM, physical=False,
        capacity_bytes=8,
        visibility=good.domains[fb.DECLARED_FPGA_HBM].visibility,
        internal_coherency=CoherencyAssumption.SOFTWARE_MANAGED,
    )
    domains = dict(good.domains)
    domains[fb.DECLARED_FPGA_HBM] = tight
    plan = fb.HeterogeneousPlan(
        domains=domains, placements=good.placements, edges=good.edges,
    )
    report = fb.validate(plan)
    assert report.ok is False
    assert "RESOURCES_EXCEED_CAPACITY" in report.codes()


def test_missing_capacity_is_not_treated_as_zero():
    """An undeclared FPGA_HBM capacity must not fail closed as 'empty'."""
    plan = fb.two_domain_plan(fpga_capacity=None)
    assert plan.domains[fb.DECLARED_FPGA_HBM].capacity_bytes is None
    assert fb.validate(plan).ok is True


def test_atlas_primitive_name_matches_hwir_and_architecture_atlas():
    import architecture_atlas as atlas
    from tools.future import hwir

    assert st.ATLAS_PRIMITIVE == "SemanticTransportEdge"
    assert st.ATLAS_PRIMITIVE in atlas.PRIMITIVES
    assert hwir.PRIMITIVE_TO_NODE_KIND[st.ATLAS_PRIMITIVE] == st.HWIR_NODE_KIND


# ---------------------------------------------------------------- 3-domain + ordering / ownership


def test_three_domain_plan_constructs_and_validates():
    """Acceptance: a heterogeneous plan across >= 3 domains constructs
    and validates. GPU_UMA and NPU are physical on this host; FPGA_HBM
    is declared (COST_MODEL, not a measurement)."""
    plan = fb.three_domain_plan()
    report = fb.validate(plan)
    assert report.ok, report.errors
    assert len(plan.domains) >= 3
    assert set(plan.domains) == {fb.HOST_GPU_UMA, fb.HOST_NPU, fb.DECLARED_FPGA_HBM}
    assert plan.domains[fb.HOST_GPU_UMA].kind is DomainKind.GPU_UMA
    assert plan.domains[fb.HOST_GPU_UMA].physical is True
    assert plan.domains[fb.HOST_NPU].kind is DomainKind.NPU
    assert plan.domains[fb.HOST_NPU].physical is True
    assert plan.domains[fb.DECLARED_FPGA_HBM].kind is DomainKind.FPGA_HBM
    assert plan.domains[fb.DECLARED_FPGA_HBM].physical is False
    assert all(e.link_coherency is CoherencyAssumption.NONE for e in plan.edges)
    assert all(e.coherency_assumption is CoherencyAssumption.NONE for e in plan.edges)
    assert all(e.sync_requirement is SyncRequirement.FENCE for e in plan.edges)
    assert plan.total_cost().label == COST_MODEL
    assert plan.qualification == "PLAN_ONLY"
    assert fb.missing_facets(plan) == []


def test_three_domain_plan_expresses_required_facets():
    plan = fb.three_domain_plan()
    facets = fb.plan_facets(plan)
    for name in fb.REQUIRED_FACETS:
        assert facets[name], f"facet {name} is empty"
    kinds = {p.node_kind for p in plan.placements}
    assert pl.NodeKind.COMPUTATION in kinds
    assert pl.NodeKind.STORAGE in kinds
    assert pl.NodeKind.STATE in kinds
    assert plan.schedule
    assert plan.fusions
    assert plan.fusions[0].physical_op == "gate_up_swiglu"
    assert plan.fusions[0].domain == fb.DECLARED_FPGA_HBM
    assert all(p.resources.bytes >= 0 for p in plan.placements)


def test_unguaranteed_ordering_is_refused():
    """THE load-bearing ordering test. A plan that assumes FENCE across a
    non-coherent link whose requested protocol is NONE must be refused.
    Mutation of reject_unguaranteed_ordering must make this FAIL."""
    plan = fb.plan_assuming_unguaranteed_ordering()
    edge = plan.edges[0]
    assert edge.link_coherency is CoherencyAssumption.NONE
    assert edge.sync_requirement is SyncRequirement.NONE
    assert edge.effective_ordering_assumption is st.OrderingGuarantee.FENCE
    assert edge.assumes_unguaranteed_ordering is True
    report = fb.validate(plan)
    assert report.ok is False
    assert "ORDERING_OVERCLAIM" in report.codes()


def test_compose_refuses_unguaranteed_ordering():
    good = fb.three_domain_plan()
    bad_edges = tuple(
        fb.overclaiming_ordering_edge(
            e, st.OrderingGuarantee.PRODUCER_CONSUMER,
            sync=SyncRequirement.FENCE,
        )
        for e in good.edges
    )
    with pytest.raises(fb.FusionBridgeError, match="ORDERING_OVERCLAIM"):
        fb.compose(
            good.domains, good.placements, bad_edges,
            object_placements=good.object_placements,
            schedule=good.schedule,
            fusions=good.fusions,
        )


def test_weaker_ordering_on_a_stronger_protocol_is_legal():
    """Assuming NONE on a FENCE protocol is conservative, not an overclaim."""
    good = fb.three_domain_plan()
    conservative = tuple(
        fb.overclaiming_ordering_edge(e, st.OrderingGuarantee.NONE)
        for e in good.edges
    )
    schedule = tuple(
        fb.ScheduleConstraint(
            predecessor=c.predecessor,
            successor=c.successor,
            assumed_ordering=st.OrderingGuarantee.NONE,
            via=c.via,
            reason="conservative",
        )
        for c in good.schedule
    )
    plan = fb.compose(
        good.domains, good.placements, conservative,
        object_placements=good.object_placements,
        schedule=schedule,
        fusions=good.fusions,
    )
    assert fb.validate(plan).ok is True


def test_unguaranteed_ownership_is_refused():
    """Exclusive TRANSFER with no happens-before on a NONE link is refused.
    Exact assumption: write-authority moved without an ordered handoff."""
    plan = fb.plan_assuming_unguaranteed_ownership()
    edge = plan.edges[0]
    assert edge.ownership_transfer is OwnershipTransfer.TRANSFER
    assert edge.sync_requirement is SyncRequirement.NONE
    assert edge.link_coherency is CoherencyAssumption.NONE
    assert edge.ownership_handoff_unguaranteed is True
    report = fb.validate(plan)
    assert report.ok is False
    assert "OWNERSHIP_OVERCLAIM" in report.codes()


def test_three_domain_overlay_carries_schedule_and_fusion():
    plan = fb.three_domain_plan()
    over = fb.overlay_physical_graph(
        {"device_placement": {}, "synchronization": [], "dependencies": []},
        plan,
    )
    assert any(s.get("kind") == "happens_before" for s in over["scheduling"])
    assert any(f.get("physical_op") == "gate_up_swiglu" for f in over["fusion"])
    assert any(
        s.get("provided_ordering") == "FENCE" for s in over["synchronization"]
        if s.get("kind") == "semantic_transport"
    )
    assert over["fusion_bridge"]["n_domains"] >= 3
    assert over["device_placement"]["fusion_bridge"]["cost"]["label"] == COST_MODEL


def test_hwir_lowering_still_works_on_three_domain_fpga_placements():
    from tools.future import hwir

    plan = fb.three_domain_plan()
    graph = fb.lower_fpga_domain_to_hwir(plan)
    report = graph.validate()
    assert report.ok, report.errors
    dma = [n for n in graph.nodes if n.kind == "dma-transport"]
    assert dma
    assert all(n.primitive == "SemanticTransportEdge" for n in dma)
    assert graph.device_budget.declared_not_measured is True


def test_ordering_gate_is_called_from_validate_not_just_imported():
    """A module import is not a call site. validate() must invoke
    reject_unguaranteed_ordering by name."""
    tree = ast.parse((ACCEL / "fusion_bridge.py").read_text())
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name == "reject_unguaranteed_ordering":
            calls.append(node.lineno)
    assert calls, "validate() never calls reject_unguaranteed_ordering"
