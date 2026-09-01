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
SCORING_SCHEMA = "hcli.physical_graph_scoring.v1"
REPATRIATION_SCHEMA = "hcli.physical_graph_repatriation.v1"
PROTECTED_BENCHMARK_CLASSES = {"PROTECTED_ABSOLUTE", "QUALIFIED_PROTECTED"}
DIAGNOSTIC_BENCHMARK_CLASSES = {"DIAGNOSTIC_RELATIVE", "DIAGNOSTIC_CONTAMINATED"}

# These are the small backend-neutral operations that Gravity/NR lowering may
# compose into an NX.  Keeping the vocabulary here makes a representation
# portable without pretending that every backend implements every primitive.
NR_PRIMITIVES = (
    "BASIS_PROJECT",
    "COEFFICIENT_APPLY",
    "SPARSE_RESIDUAL",
    "QUANT_PROJECT",
    "CODEBOOK_LOOKUP",
    "ROUTED_SELECT",
    "WEIGHTED_ACCUMULATE",
    "STATE_UPDATE",
)


def _copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _positive_int(value: Any) -> Optional[int]:
    """Parse a metric without turning missing/invalid evidence into zero."""

    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _same_target(value: Any, target: str) -> bool:
    """Match atlas target labels without pretending model names are exact IDs."""

    source = str(value).strip().lower()
    if source in {"all", "*"}:
        return True
    left = "".join(ch for ch in str(value).lower() if ch.isalnum())
    right = "".join(ch for ch in str(target).lower() if ch.isalnum())
    if source == "qwen27" and "qwen" in right and "27" in right:
        return True
    if source == "flash" and "flash" in right:
        return True
    return bool(left and right and (left == right or left in right or right in left))


def apply_architecture_atlas(
    graph: Mapping[str, Any],
    atlas: Mapping[str, Any],
    *,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Project the architecture atlas into a PhysicalGraph plan.

    This is a planning operation only.  ``MAPPED`` and ``DIAGNOSTIC`` entries
    become selectable hypotheses, not performance claims; the protected
    scoreboard remains the authority for promotion.  Keeping the projection
    here lets Metal, ANE, FPGA, CUDA, and CPU lowerings share one physical
    vocabulary without importing a vendor runtime into the graph compiler.
    """

    if not isinstance(atlas, Mapping) or not isinstance(atlas.get("entries"), list):
        raise ValueError("architecture atlas must provide an entries list")
    result = _copy(graph)
    model_id = str(result.get("model_id") or "unknown")
    target_backend = str(backend or "").lower()
    entries: List[Dict[str, Any]] = []
    for entry in atlas["entries"]:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("status") or "").upper() in {"REJECTED", "BLOCKED"}:
            continue
        models = entry.get("applicable_models") or []
        backends = entry.get("applicable_backends") or []
        model_ok = any(_same_target(model, model_id) for model in models)
        backend_ok = not target_backend or any(_same_target(item, target_backend) for item in backends)
        if model_ok and backend_ok:
            entries.append(dict(entry))
    entries.sort(key=lambda item: (-float(item.get("expected_value_score") or 0), str(item.get("behavior_id"))))
    primitives = sorted({str(entry["hawking_primitive"]) for entry in entries if entry.get("hawking_primitive")})
    behaviors = [str(entry["behavior_id"]) for entry in entries]
    representation = result.setdefault("representation", {})
    if not isinstance(representation, dict):
        representation = {}
        result["representation"] = representation
    representation["architecture_atlas"] = {
        "schema": atlas.get("schema"),
        "fingerprint": atlas.get("fingerprint"),
        "selected_behavior_ids": behaviors,
        "backend_neutral_primitives": primitives,
        "selection_is_planning_only": True,
    }
    if "LayoutTransform" in primitives:
        # Layout is a compiler object, not a kernel-name hint.  Keep the
        # logical, physical, tile, lane, and arithmetic mappings distinct so a
        # later lowering can choose them from OrganShape/NR/MachineGenome.
        representation["layout_algebra"] = {
            "schema": "hcli.physical_graph_layout_algebra.v1",
            "logical_tensor": "organ_semantics",
            "physical_memory": "representation_and_machine_genome_selected",
            "tile_mapping": "organ_shape_and_nr_shape",
            "vector_mapping": "backend_capability_and_alignment",
            "threadgroup_or_spatial_mapping": "machine_genome",
            "lane_mapping": "backend_lowering",
            "arithmetic_mapping": "representation_native_consumer",
            "selection_is_parameterized": True,
            "nominal_utilization_is_not_authority": True,
        }
    if "StationaryRepresentation" in primitives:
        representation["stationarity"] = {
            "schema": "hcli.physical_graph_stationarity.v1",
            "candidate_operands": [
                "weights",
                "basis",
                "codebook",
                "recurrent_state",
                "kv",
                "partial_reduction",
            ],
            "selection": "reuse_minus_residency_and_movement_cost",
            "status": "candidate_until_measured",
        }
    execution_policy = result.setdefault("execution_policy", {})
    if not isinstance(execution_policy, dict):
        execution_policy = {}
        result["execution_policy"] = execution_policy
    execution_policy["architecture_repatriation"] = {
        "schema": REPATRIATION_SCHEMA,
        "selected_behaviors": behaviors,
        "static_skeleton": "precomputed dependencies/ownership where dynamic slots permit",
        "dynamic_slots": ["token", "position", "route", "sampling", "variable_state"],
        "memory_tier_is_executable_identity": True,
        "stationarity_is_explicit": True,
        "move_or_recompute_is_explicit": "costed_dependency_query",
        "device_count_is_not_speed_authority": True,
        "measurement_authority": "protected complete useful wall time with capability gate",
        "no_source_product_port_claim": True,
    }
    memory = result.setdefault("memory", [])
    if isinstance(memory, list):
        memory.append({
            "tier": "atlas_declared",
            "role": "representation/state residency and movement are part of executable identity",
            "selected_behaviors": behaviors,
            "status": "candidate",
        })
    device_placement = result.setdefault("device_placement", {})
    if isinstance(device_placement, dict):
        device_placement["primitive_realizations"] = {
            str(entry["hawking_primitive"]): dict(entry.get("physical_graph_mapping") or {})
            for entry in entries
            if entry.get("hawking_primitive")
        }
    synchronization = result.setdefault("synchronization", [])
    if isinstance(synchronization, list):
        synchronization.append({
            "kind": "atlas_measurement_boundary",
            "status": "unresolved",
            "contract": "transfer, conversion, launch, synchronization, and residency costs must be measured",
        })
    evidence = result.setdefault("evidence", [])
    if isinstance(evidence, list):
        evidence.append({
            "kind": "architecture_atlas",
            "schema": atlas.get("schema"),
            "fingerprint": atlas.get("fingerprint"),
            "claim": "hypothesis projection only; not physical performance evidence",
        })
    result["architecture_repatriation"] = {
        "schema": REPATRIATION_SCHEMA,
        "atlas_fingerprint": atlas.get("fingerprint"),
        "model_id": model_id,
        "backend": backend,
        "selected_behavior_ids": behaviors,
        "selected_primitives": primitives,
        "promotion": "not_allowed_without_protected_receipt",
    }
    body_for_fingerprint = {key: value for key, value in result.items() if key not in {"fingerprint", "generated_at"}}
    result["fingerprint"] = hashlib.sha256(
        json.dumps(body_for_fingerprint, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return result


def score_physical_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    require_protected: bool = False,
) -> Dict[str, Any]:
    """Rank physical plans by measured complete useful work.

    This is deliberately a conservative compiler-side scorer, not a learned
    scheduler.  It can rank diagnostic candidates for search, but it never
    grants promotion unless a candidate has a measured complete-token value,
    an independent capability result, no fallback, and (when requested) a
    ``PROTECTED_ABSOLUTE`` benchmark class.  Nominal utilization is retained as
    metadata only and never enters the ordering tuple.
    """

    rows: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        complete_ns = _positive_int(
            candidate.get("accepted_complete_token_ns")
            or candidate.get("complete_token_ns")
            or candidate.get("complete_useful_ns")
        )
        resident_bytes = _positive_int(candidate.get("resident_bytes"))
        active_bytes = _positive_int(candidate.get("active_bytes_per_token"))
        transfer_ns = _positive_int(candidate.get("transfer_ns")) or 0
        synchronization_ns = _positive_int(
            candidate.get("synchronization_ns") or candidate.get("sync_ns")
        ) or 0
        dispatches = _positive_int(candidate.get("dispatches")) or 0
        benchmark = candidate.get("benchmark")
        benchmark_value = benchmark.get("class") if isinstance(benchmark, Mapping) else None
        benchmark_class = str(candidate.get("benchmark_class") or benchmark_value or "UNKNOWN").upper()
        capability_value = candidate.get("capability_verified")
        if capability_value is None:
            capability_value = candidate.get("capability_passed")
        if capability_value is None:
            capability_value = str(candidate.get("capability") or "").upper() in {
                "PASS",
                "PASSED",
                "ACCEPTED",
                "VERIFIED",
            }
        capability_verified = capability_value is True
        fallback_count = _positive_int(candidate.get("fallback_count")) or 0
        fallback = bool(candidate.get("fallback")) or fallback_count > 0
        reasons: List[str] = []
        if complete_ns is None:
            reasons.append("complete_useful_work_unmeasured")
        if not capability_verified:
            reasons.append("independent_capability_not_verified")
        if fallback:
            reasons.append("forbidden_or_unreported_fallback")
        protected_class = benchmark_class in PROTECTED_BENCHMARK_CLASSES
        if require_protected and not protected_class:
            reasons.append("protected_absolute_evidence_required")
        row = {
            "id": str(candidate.get("id") or candidate.get("name") or f"candidate-{index}"),
            "representation": candidate.get("representation"),
            "device": candidate.get("device") or candidate.get("backend"),
            "operation_grouping": candidate.get("operation_grouping") or candidate.get("grouping"),
            "transfer_synchronization_boundary": candidate.get(
                "transfer_synchronization_boundary"
            )
            or candidate.get("transfer_boundary"),
            "benchmark_class": benchmark_class,
            "protected_benchmark_class": protected_class,
            "measurement_state": candidate.get("measurement_state") or candidate.get("state"),
            "complete_useful_ns": complete_ns,
            "resident_bytes": resident_bytes,
            "active_bytes_per_token": active_bytes,
            "transfer_ns": transfer_ns,
            "synchronization_ns": synchronization_ns,
            "dispatches": dispatches,
            "capability_verified": capability_verified,
            "fallback_count": fallback_count,
            "diagnostic_only": not protected_class,
            "eligible_for_selection": not reasons,
            "ineligibility_reasons": reasons,
            # Nominal utilization is intentionally visible but not ranked.
            "nominal_utilization": candidate.get("nominal_utilization"),
        }
        if not reasons:
            # The ordering is lexicographic and deterministic.  Complete useful
            # work dominates every secondary metric; the rest only breaks ties.
            row["ordering_key"] = [
                complete_ns,
                resident_bytes if resident_bytes is not None else 2**63 - 1,
                active_bytes if active_bytes is not None else 2**63 - 1,
                transfer_ns,
                synchronization_ns,
                dispatches,
                index,
            ]
        rows.append(row)

    eligible = [row for row in rows if row["eligible_for_selection"]]
    eligible.sort(key=lambda row: row["ordering_key"])
    winner = eligible[0] if eligible else None
    promotion_allowed = bool(
        winner
        and winner["protected_benchmark_class"]
        and winner["capability_verified"]
        and winner["complete_useful_ns"] is not None
        and winner["fallback_count"] == 0
    )
    return {
        "schema": SCORING_SCHEMA,
        "objective": "minimize measured accepted complete useful work after capability gate",
        "dimensions": [
            "representation",
            "device",
            "operation_grouping",
            "transfer_synchronization_boundary",
        ],
        "nominal_utilization_is_not_authority": True,
        "require_protected": require_protected,
        "candidates": rows,
        "eligible_order": [row["id"] for row in eligible],
        "winner": winner["id"] if winner else None,
        "promotion_allowed": promotion_allowed,
        "promotion_rule": "protected measured complete useful work + independent capability + no fallback",
    }


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
    execution_policy: Dict[str, Any] = field(default_factory=dict)
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
            "execution_policy": _copy(self.execution_policy),
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
    provider: Optional[Mapping[str, Any] | Any] = None,
    devices: Optional[Iterable[str]] = None,
    architecture_atlas: Optional[Mapping[str, Any]] = None,
    backend: Optional[str] = None,
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
    provider_payload: Dict[str, Any] = {}
    if provider is not None:
        if hasattr(provider, "to_dict"):
            provider_payload = _copy(provider.to_dict())
        elif isinstance(provider, Mapping):
            provider_payload = _copy(provider)
        else:
            provider_payload = {"kind": type(provider).__name__}
    candidate_devices = list(devices or ("cpu", "gpu", "fpga", "remote"))
    if provider_payload.get("kind") == "ANEProvider" and "ane" not in candidate_devices:
        candidate_devices.insert(2, "ane")
    graph = PhysicalGraph(
        model_id=model_id,
        computation=computation,
        data=data,
        representation={
            "architecture": _copy(arch),
            "native_representation_verified": False,
            "gravity_candidates": [],
            "nr_primitives": list(NR_PRIMITIVES),
            "physical_scoring": {
                "schema": SCORING_SCHEMA,
                "dimensions": [
                    "representation",
                    "device",
                    "operation_grouping",
                    "transfer_synchronization_boundary",
                ],
                "objective": "measured complete useful work after capability gate",
                "nominal_utilization_is_not_authority": True,
            },
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
        device_placement={
            "candidates": candidate_devices,
            "selected": None,
            "ane": {
                "status": "CANDIDATE_ONLY",
                "selection_authority": "measured complete useful work",
            } if provider_payload.get("kind") == "ANEProvider" else None,
        },
        synchronization=[{"kind": "runtime_boundary", "status": "unresolved"}],
        evidence=list(architecture.get("evidence") or []),
        execution_policy={
            "process": "long_lived_executor",
            "source_index": "cache_by_immutable_source_seal",
            "pipeline_state": "compile_once_reuse",
            "scratch": "allocate_once_reuse",
            "layer_semantics": "instantiate_qualified_species_contract_with_layer_data",
            "state_handoff": {
                "fast_verified": "device_resident",
                "deep_verification": "optional_host_snapshot",
            },
            "verification": {
                "fast": "L0_device_state_plus_L1_fingerprint_checkpoint",
                "protected": "L0_plus_L1_plus_L2_sampled_probes",
                "debug": "L0_plus_L1_plus_L2_plus_L3_full_state",
                "divergence": "checkpoint_bisection_then_local_deep_probe",
            },
            "grouping": "architecture_species_and_attention_boundaries",
            "cache_key": ["source_seal", "organ_identity", "representation_schema", "parameters", "verifier_version"],
            "promotion_metric": "measured_complete_useful_work",
            "device_count_is_not_speed_authority": True,
            "selection_costs": [
                "transfer",
                "conversion",
                "residency",
                "launch",
                "synchronization",
                "interference",
            ],
        },
    )
    result = graph.to_dict()
    physical_candidates = architecture.get("physical_candidates")
    if isinstance(physical_candidates, list):
        result["physical_plan_score"] = score_physical_candidates(physical_candidates)
    if provider is not None:
        result["provider_context"] = provider_payload
    if architecture_atlas is not None:
        result = apply_architecture_atlas(result, architecture_atlas, backend=backend)
    return result


__all__ = [
    "DIAGNOSTIC_BENCHMARK_CLASSES",
    "NR_PRIMITIVES",
    "PhysicalGraph",
    "PROTECTED_BENCHMARK_CLASSES",
    "REPATRIATION_SCHEMA",
    "SCHEMA",
    "SCORING_SCHEMA",
    "apply_architecture_atlas",
    "compile_physical_graph",
    "score_physical_candidates",
]
