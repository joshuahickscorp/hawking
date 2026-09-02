"""Odyssey adapter for the Fusion Bridge.

Connects, does not replace:

  hcli.physical_graph.compile_physical_graph
      Produces a PLAN_ONLY PhysicalGraph. This adapter copies it and fills
      device_placement / synchronization / dependencies from a
      HeterogeneousPlan. hcli/ is not edited.

  tools/odyssey/physical_graph_compiler.py
      Organ-collapse compiler (gate_up_swiglu, router top-k). That tool
      measures numerical equivalence of fused operators on real weights.
      annotate_compiler_output() adds a fusion-bridge DAG stage with a
      COST_MODEL cost_delta so the compiler receipt can carry a
      heterogeneous placement without pretending the interconnect was
      measured.

  tools/odyssey/device_profiles.py
      INTERACTIVE / MAXX are workload profiles (one-stream latency vs
      four-stream aggregate), not device domains. They are admitted as
      Placement.profile_hint values by tools/accelerator/placement.py.
      This adapter does not treat them as execution domains.

This file does not import physical_graph_compiler (numpy/safetensors,
checkpoint weights). The DAG annotator is a pure dict rewrite.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[2]
ACCEL = REPO / "tools" / "accelerator"
for _p in (str(REPO), str(ACCEL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fusion_bridge import (  # noqa: E402
    COST_MODEL,
    HeterogeneousPlan,
    compile_and_overlay,
    overlay_physical_graph,
    three_domain_plan,
    two_domain_plan,
    validate,
)


def annotate_compiler_output(
    compiler_output: Mapping[str, Any],
    plan: HeterogeneousPlan | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Add a fusion-bridge stage to a physical_graph_compiler receipt.

    cost_delta.label is COST_MODEL. This is not a hardware measurement
    and does not change the compiler's numerical-equivalence verdict.
    """
    if plan is None:
        plan = two_domain_plan()
    plan_dict = plan.to_dict() if isinstance(plan, HeterogeneousPlan) else dict(plan)
    out = dict(compiler_output)
    dag = list(out.get("transformation_dag") or [])
    dag.append({
        "stage": "heterogeneous_fusion_bridge",
        "input": "physical_operator_graph",
        "output": "typed transport edges and node placements across execution domains",
        "cost_delta": {
            "label": COST_MODEL,
            "meaning": "planner estimate; not a hardware measurement",
        },
        "evidence": "tools/accelerator/fusion_bridge.py",
    })
    out["transformation_dag"] = dag
    out["fusion_bridge"] = plan_dict
    return out


def physical_graph_with_bridge(
    architecture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """compile_physical_graph + overlay. Planning annotation only."""
    return compile_and_overlay(architecture)


def overlay(graph: Mapping[str, Any], plan: HeterogeneousPlan | None = None) -> dict[str, Any]:
    return overlay_physical_graph(graph, plan or two_domain_plan())


def plan_is_valid(plan: HeterogeneousPlan | None = None) -> bool:
    return validate(plan or two_domain_plan()).ok


def three_domain_is_valid(plan: HeterogeneousPlan | None = None) -> bool:
    """Call site: validate() on a >=3 domain plan, not an import."""
    return validate(plan or three_domain_plan()).ok
