"""PhysicalGraph compiler: heterogeneous plan across >= 3 domains.

Connects tools/odyssey/physical_graph_compiler.py to the fusion-bridge
transport/placement/sync/ownership gates. Does not load checkpoint
weights. Does not edit hcli/.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ODYSSEY = Path(__file__).resolve().parent
ACCEL = REPO / "tools" / "accelerator"
for _p in (str(REPO), str(ACCEL), str(ODYSSEY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fusion_bridge as fb  # noqa: E402
import physical_graph_compiler as pgc  # noqa: E402
from semantic_transport import (  # noqa: E402
    COST_MODEL,
    CoherencyAssumption,
    OrderingGuarantee,
    OwnershipTransfer,
    SyncRequirement,
)


COMPILER = ODYSSEY / "physical_graph_compiler.py"
GENERIC = (
    ACCEL / "semantic_transport.py",
    ACCEL / "placement.py",
    ACCEL / "fusion_bridge.py",
)
VENDOR_NEEDLES = ("apple", "xilinx", "nvidia", "cuda", "metal")


def test_compiler_three_domain_plan_constructs_and_validates():
    compiled = pgc.compile_heterogeneous_physical_graph()
    assert compiled["pass"] is True
    assert compiled["n_domains"] >= 3
    assert compiled["evidence_tier"] == COST_MODEL
    assert compiled["qualification"] == "PLAN_ONLY"
    assert compiled["validation"]["ok"] is True
    assert compiled["fusion_bridge_cost"]["label"] == COST_MODEL
    domains = compiled["hardware"]
    assert domains[fb.HOST_GPU_UMA]["physical"] is True
    assert domains[fb.HOST_NPU]["physical"] is True
    assert domains[fb.DECLARED_FPGA_HBM]["physical"] is False
    assert domains[fb.DECLARED_FPGA_HBM]["evidence_tier"] == COST_MODEL


def test_compiler_expresses_all_required_facets():
    compiled = pgc.compile_heterogeneous_physical_graph()
    facets = compiled["facets"]
    for name in fb.REQUIRED_FACETS:
        assert facets[name], f"facet {name} missing or empty"
    assert compiled["scheduling"]
    assert compiled["fusion"]
    assert any(f["physical_op"] == "gate_up_swiglu" for f in compiled["fusion"])
    # backend binding is by DomainKind; CUDA is named, not instantiated
    backends = compiled["backends"]
    kinds = {row["domain_kind"] for row in backends.values()}
    assert "GPU_UMA" in kinds
    assert "NPU" in kinds
    assert "FPGA_HBM" in kinds
    assert all(row["backend_id"] != "CUDA" for row in backends.values())
    fpga = next(row for row in backends.values() if row["domain_kind"] == "FPGA_HBM")
    assert fpga["present"] is False
    assert fpga["evidence_tier"] != "HARDWARE_MEASURED"


def test_compiler_calls_ordering_gate_by_name():
    """A module import is not a call site. The compiler must invoke
    reject_unguaranteed_ordering, not merely import fusion_bridge."""
    tree = ast.parse(COMPILER.read_text())
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
    assert calls, (
        "physical_graph_compiler.py never calls reject_unguaranteed_ordering"
    )


def test_compiler_calls_three_domain_plan_and_validate():
    tree = ast.parse(COMPILER.read_text())
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            names.add(func.attr)
        elif isinstance(func, ast.Name):
            names.add(func.id)
    for required in (
        "three_domain_plan",
        "validate",
        "reject_unguaranteed_ordering",
        "kind_admits",
        "resources_fit",
        "enumerate_backends",
        "overlay_physical_graph",
        "compile_physical_graph",
        "missing_facets",
    ):
        assert required in names, f"compiler never calls {required}"


def test_compiler_refusal_cases_name_the_violated_assumption():
    cases = pgc.refusal_cases()
    by_id = {c["id"]: c for c in cases}
    assert set(by_id) >= {
        "unguaranteed_ordering",
        "unguaranteed_coherency",
        "unguaranteed_ownership",
    }
    for case in cases:
        assert case["ok"] is False
        assert case["refused"] is True
        assert case["violates"] in case["codes"]
        assert case["assumption"]

    ordering = by_id["unguaranteed_ordering"]
    assert ordering["violates"] == "ORDERING_OVERCLAIM"
    assert "FENCE" in ordering["assumption"]
    assert "NONE" in ordering["assumption"]

    coherency = by_id["unguaranteed_coherency"]
    assert coherency["violates"] == "COHERENCY_OVERCLAIM"
    assert "SOFTWARE_MANAGED" in coherency["assumption"]

    ownership = by_id["unguaranteed_ownership"]
    assert ownership["violates"] == "OWNERSHIP_OVERCLAIM"
    assert "TRANSFER" in ownership["assumption"]


def test_compiler_validate_refuses_ordering_overclaim():
    plan = fb.plan_assuming_unguaranteed_ordering()
    report = pgc.validate_heterogeneous_plan(plan)
    assert report["ok"] is False
    assert "ORDERING_OVERCLAIM" in report["validate_codes"]
    assert "ORDERING_OVERCLAIM" in report["direct_gate_codes"]


def test_compiler_validate_refuses_coherency_overclaim():
    good = fb.three_domain_plan()
    plan = fb.HeterogeneousPlan(
        domains=good.domains,
        placements=good.placements,
        edges=tuple(
            fb.overclaiming_edge(e, CoherencyAssumption.SOFTWARE_MANAGED)
            for e in good.edges
        ),
        object_placements=good.object_placements,
        schedule=good.schedule,
        fusions=good.fusions,
    )
    report = pgc.validate_heterogeneous_plan(plan)
    assert report["ok"] is False
    assert "COHERENCY_OVERCLAIM" in report["validate_codes"]


def test_compiler_validate_refuses_ownership_overclaim():
    plan = fb.plan_assuming_unguaranteed_ownership()
    report = pgc.validate_heterogeneous_plan(plan)
    assert report["ok"] is False
    assert "OWNERSHIP_OVERCLAIM" in report["validate_codes"]
    edge = plan.edges[0]
    assert edge.ownership_transfer is OwnershipTransfer.TRANSFER
    assert edge.sync_requirement is SyncRequirement.NONE
    assert edge.effective_ordering_assumption is OrderingGuarantee.NONE


def test_attach_heterogeneous_plan_still_pure_dict_rewrite():
    stub = {"transformation_dag": [{"stage": "source_graph"}], "pass": True}
    plan = fb.three_domain_plan()
    annotated = pgc.attach_heterogeneous_plan(stub, plan.to_dict())
    assert annotated["transformation_dag"][-1]["stage"] == "heterogeneous_fusion_bridge"
    assert annotated["transformation_dag"][-1]["cost_delta"]["label"] == COST_MODEL
    assert annotated["fusion_bridge"]["n_domains"] >= 3


def test_generic_layer_still_has_no_vendor_keyed_control_flow():
    """Re-pin the fusion-bridge law: generic files must not branch on a
    product name. Mirrors test_fusion_bridge.GENERIC scan."""

    def vendor_in(text: str) -> bool:
        low = text.lower()
        return any(v in low for v in VENDOR_NEEDLES)

    hits = []
    for path in GENERIC:
        tree = ast.parse(path.read_text())

        def scan(expr: ast.AST, lineno: int, filename: str) -> None:
            for child in ast.walk(expr):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    if vendor_in(child.value):
                        hits.append(f"{filename}:{lineno}: string {child.value!r}")
                elif isinstance(child, ast.Name) and vendor_in(child.id):
                    hits.append(f"{filename}:{lineno}: name {child.id}")
                elif isinstance(child, ast.Attribute) and vendor_in(child.attr):
                    hits.append(f"{filename}:{lineno}: attr {child.attr}")

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                scan(node.test, node.lineno, path.name)
            elif isinstance(node, ast.IfExp):
                scan(node.test, node.lineno, path.name)
            elif isinstance(node, ast.While):
                scan(node.test, node.lineno, path.name)
            elif isinstance(node, ast.Assert):
                scan(node.test, node.lineno, path.name)
            elif isinstance(node, ast.Match):
                scan(node.subject, node.lineno, path.name)
    assert hits == []
