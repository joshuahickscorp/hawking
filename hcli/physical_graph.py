"""Provider-neutral PhysicalGraph planning boundary.

Architecture recognition produces observations; this module turns those
observations into a serialisable placement/dataflow graph.  It does not claim
that a device executed the graph or that a proposed kernel is valid.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from hcli.nomenclature import NOMENCLATURE_VERSION


SCHEMA = "hcli.physical_graph.v1"


def _copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


@dataclass
class PhysicalGraph:
    """A plan for computation/data/representation placement."""

    model_id: str = "unknown"
    computation: List[Dict[str, Any]] = field(default_factory=list)
    data: List[Dict[str, Any]] = field(default_factory=list)
    representation: Dict[str, Any] = field(default_factory=dict)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    residency: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    precision: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    device_placement: Dict[str, Any] = field(default_factory=dict)
    synchronization: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    qualification: str = "PLAN_ONLY"
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema": SCHEMA,
            "nomenclature_version": NOMENCLATURE_VERSION,
            "semantic_type": "PhysicalGraphPlan",
            "compiler_stage": "PhysicalGraphCompiler",
            "model_id": self.model_id,
            "computation": _copy(self.computation),
            "data": _copy(self.data),
            "representation": _copy(self.representation),
            "memory": _copy(self.memory),
            "residency": _copy(self.residency),
            "state": _copy(self.state),
            "precision": _copy(self.precision),
            "dependencies": _copy(self.dependencies),
            "device_placement": _copy(self.device_placement),
            "synchronization": _copy(self.synchronization),
            "evidence": _copy(self.evidence),
            "qualification": self.qualification,
            "generated_at": self.generated_at,
        }
        body["fingerprint"] = hashlib.sha256(
            json.dumps({key: value for key, value in body.items() if key != "generated_at"}, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return body


def compile_physical_graph(
    architecture: Mapping[str, Any],
    *,
    provider: Optional[Mapping[str, Any]] = None,
    devices: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Compile a conservative graph from an ArchitectureRecognizer report."""
    arch = architecture.get("architecture") if isinstance(architecture.get("architecture"), Mapping) else {}
    organs = architecture.get("organs") if isinstance(architecture.get("organs"), list) else []
    model_id = str(architecture.get("model_id") or "unknown")
    computation = []
    data = []
    for organ in organs:
        if not isinstance(organ, Mapping):
            continue
        organ_id = str(organ.get("id") or "unknown")
        node = {
            "id": organ_id,
            "kind": "computation",
            "present": bool(organ.get("present")),
            "tensor_count": organ.get("tensor_count"),
            "confidence": organ.get("confidence"),
        }
        computation.append(node)
        data.append({
            "id": organ_id,
            "kind": "tensor_group",
            "bytes": None,
            "active_bytes_per_token": None,
            "source": "architecture metadata; size unresolved",
        })
    graph = PhysicalGraph(
        model_id=model_id,
        computation=computation,
        data=data,
        representation={
            "architecture": _copy(arch),
            "native_representation_verified": False,
            "gravity_candidates": [],
        },
        memory=[
            {"tier": "hot", "role": "active working set", "status": "candidate"},
            {"tier": "cold", "role": "canonical source", "status": "candidate"},
        ],
        residency={"weights": "unresolved", "state": "unresolved", "page_cache": "unresolved"},
        state={"kv_cache": "unresolved", "recurrent_state": "unresolved"},
        precision={"weight": "unresolved", "activation": "unresolved", "accumulator": "unresolved"},
        dependencies=[
            {"from": "embedding", "to": "attention_or_recurrent", "kind": "dataflow"},
            {"from": "attention_or_recurrent", "to": "output_head", "kind": "dataflow"},
        ],
        device_placement={"candidates": list(devices or ("cpu", "gpu", "fpga", "remote")), "selected": None},
        synchronization=[{"kind": "runtime_boundary", "status": "unresolved"}],
        evidence=list(architecture.get("evidence") or []),
    )
    result = graph.to_dict()
    if provider is not None:
        result["provider_context"] = _copy(provider)
    return result


__all__ = ["PhysicalGraph", "SCHEMA", "compile_physical_graph"]
