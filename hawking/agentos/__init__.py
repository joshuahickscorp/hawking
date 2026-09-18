"""Internal Hawking compatibility facade.

The public product surface is ``hawking`` / ``h`` / ``hawkingd``.  This
package bridges legacy Python imports while the remaining Flash experimental
legs physically live below it; it adds no public command, port, receipt owner,
or runtime selection path.  Retire this facade when those legs have been
promoted into their Hawking kernels or merged into a smaller existing owner.

Canonical implementation modules already live at ``hawking.*`` and are
re-exported lazily here only so ordinary Hawking imports retain their cold-path
behavior.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any


# Compatibility bridge only. In particular, local perception is Hawking
# perception re-exported here; it is not a nested runtime with an independent
# lifecycle or import cost.
_CORE_EXPORTS: dict[str, tuple[str, str]] = {
    "DagStore": ("hawking.dag_store", "DagStore"),
    "GoalCompiler": ("hawking.goal", "GoalCompiler"),
    "Ledger": ("hawking.ledger", "Ledger"),
    "Mission": ("hawking.mission", "Mission"),
    "MutationError": ("hawking.mutation", "MutationError"),
    "MutationLock": ("hawking.resources", "MutationLock"),
    "ResourceClass": ("hawking.resources", "ResourceClass"),
    "ResourceLimits": ("hawking.resources", "ResourceLimits"),
    "Scheduler": ("hawking.scheduler", "Scheduler"),
    "SteeringQueue": ("hawking.steering", "SteeringQueue"),
    "WorkUnit": ("hawking.workunit", "WorkUnit"),
    "WorkUnitExecutor": ("hawking.executors", "WorkUnitExecutor"),
    "WorkerPacket": ("hawking.goal", "WorkerPacket"),
    "command_is_admissible": ("hawking.verifier_pipeline", "command_is_admissible"),
    "compile_worker_context": ("hawking.goal", "compile_worker_context"),
    "Hawking": ("hawking.agentos.runtime", "Hawking"),
    "AgentOS": ("hawking.agentos.runtime", "AgentOS"),
    "GoalBank": ("hawking.goal_bank", "GoalBank"),
    "GoalBankError": ("hawking.goal_bank", "GoalBankError"),
    "KnowledgeError": ("hawking.knowledge", "KnowledgeError"),
    "KnowledgeStore": ("hawking.knowledge", "KnowledgeStore"),
    "BackgroundJob": ("hawking.background", "BackgroundJob"),
    "BackgroundJobStore": ("hawking.background", "BackgroundJobStore"),
    "ResidentConfig": ("hawking.resident", "ResidentConfig"),
    "ResidentBodyRegistry": ("hawking.resident", "ResidentBodyRegistry"),
    "ResidentDaemon": ("hawking.resident", "ResidentDaemon"),
    "ResidentStore": ("hawking.resident", "ResidentStore"),
    "ResidentSupervisor": ("hawking.resident", "ResidentSupervisor"),
    "admit_evidence_children": ("hawking.resident", "admit_evidence_children"),
    "memory_decision": ("hawking.resident", "memory_decision"),
    "resident_behavior": ("hawking.resident", "resident_behavior"),
    "start_resident": ("hawking.resident", "start_resident"),
    "AgentState": ("hawking.agentos.states", "AgentState"),
    "mission_state": ("hawking.agentos.states", "mission_state"),
    "workunit_state": ("hawking.agentos.states", "workunit_state"),
    "Capability": ("hawking.providers", "Capability"),
    "CapabilityContract": ("hawking.providers", "CapabilityContract"),
    "GenerationRequest": ("hawking.providers", "GenerationRequest"),
    "GenerationResponse": ("hawking.providers", "GenerationResponse"),
    "ModelProvider": ("hawking.providers", "ModelProvider"),
    "ProviderFailure": ("hawking.providers", "ProviderFailure"),
    "ProviderHealth": ("hawking.providers", "ProviderHealth"),
    "ProviderReceipt": ("hawking.providers", "ProviderReceipt"),
    "ResidentProfile": ("hawking.providers", "ResidentProfile"),
    "ResidentProvider": ("hawking.providers", "ResidentProvider"),
    "RolePolicy": ("hawking.providers", "RolePolicy"),
    "RoleRouter": ("hawking.providers", "RoleRouter"),
    "PhysicalGraph": ("hawking.physical_graph", "PhysicalGraph"),
    "DIAGNOSTIC_BENCHMARK_CLASSES": ("hawking.physical_graph", "DIAGNOSTIC_BENCHMARK_CLASSES"),
    "NR_PRIMITIVES": ("hawking.physical_graph", "NR_PRIMITIVES"),
    "PROTECTED_BENCHMARK_CLASSES": ("hawking.physical_graph", "PROTECTED_BENCHMARK_CLASSES"),
    "compile_physical_graph": ("hawking.physical_graph", "compile_physical_graph"),
    "score_physical_candidates": ("hawking.physical_graph", "score_physical_candidates"),
    "ANEProvider": ("hawking.ane_provider", "ANEProvider"),
    "NOMENCLATURE_VERSION": ("hawking.nomenclature", "NOMENCLATURE_VERSION"),
    "ResultEnvelope": ("hawking.result_envelope", "ResultEnvelope"),
    "build_result_envelope": ("hawking.result_envelope", "build_result_envelope"),
    "ToolContext": ("hawking.tool_registry", "ToolContext"),
    "ToolRegistry": ("hawking.tool_registry", "ToolRegistry"),
    "ToolResult": ("hawking.tool_registry", "ToolResult"),
    "ToolSpec": ("hawking.tool_registry", "ToolSpec"),
    "default_tool_registry": ("hawking.tool_registry", "default_tool_registry"),
    "observe": ("hawking.perception", "observe"),
    "classify_bytes": ("hawking.perception", "classify_bytes"),
    "capture": ("hawking.perception", "capture"),
    "probe": ("hawking.perception", "probe"),
    "profile": ("hawking.perception", "profile"),
    "report": ("hawking.perception", "report"),
    "run_matrix": ("hawking.behavior", "run_matrix"),
    "see": ("hawking.perception", "see"),
    "hold": ("hawking.perception", "hold"),
    "know": ("hawking.perception", "know"),
    "check": ("hawking.perception", "check"),
    "prove": ("hawking.perception", "prove"),
    "compact_surface": ("hawking.perception", "compact_surface"),
    "perception_disposition": ("hawking.perception", "disposition"),
}

__all__ = [
    "DagStore",
    "GoalCompiler",
    "Ledger",
    "Mission",
    "MutationError",
    "MutationLock",
    "ResourceClass",
    "ResourceLimits",
    "Scheduler",
    "SteeringQueue",
    "WorkUnit",
    "WorkUnitExecutor",
    "WorkerPacket",
    "command_is_admissible",
    "compile_worker_context",
    "Hawking",
    "AgentOS",
    "GoalBank",
    "GoalBankError",
    "KnowledgeError",
    "KnowledgeStore",
    "BackgroundJob",
    "BackgroundJobStore",
    "ResidentConfig",
    "ResidentBodyRegistry",
    "ResidentDaemon",
    "ResidentStore",
    "ResidentSupervisor",
    "admit_evidence_children",
    "memory_decision",
    "resident_behavior",
    "start_resident",
    "RECOVERY_GATE_SCHEMA",
    "run_recovery_gate",
    "RESEARCH_GATE_SCHEMA",
    "run_research_gate",
    "PERCEPTION_GATE_SCHEMA",
    "run_perception_gate",
    "NATIVE_GATE_SCHEMA",
    "run_native_gate",
    "RESIDENT_GATE_SCHEMA",
    "run_resident_gate",
    "NATIVE_MISSION_GATE_SCHEMA",
    "run_native_mission_gate",
    "AUTONOMY_GATE_SCHEMA",
    "run_autonomy_gate",
    "UNATTENDED_WINDOW_SCHEMA",
    "run_unattended_window",
    "ACCELERATOR_REGRESSION_SCHEMA",
    "run_accelerator_regression",
    "QWEN27_RUNTIME_IDENTITY_SCHEMA",
    "QWEN27_RUNTIME_DIFF_SCHEMA",
    "run_runtime_archaeology",
    "QWEN27_MLP_DIAGNOSTIC_SCHEMA",
    "run_qwen27_mlp_diagnostic_ab",
    "QWEN38_FUSION_AUDIT_SCHEMA",
    "run_qwen38_fusion_source_audit",
    "MODELLAKE_CENSUS_SCHEMA",
    "run_modellake_census",
    "FLASH_SCIENCE_SCHEMA",
    "run_flash_science_gate",
    "FLASH_EXECUTABLE_SCHEMA",
    "run_flash_executable_scaffold",
    "FLASH_TENSOR_PROBE_SCHEMA",
    "run_flash_tensor_probe",
    "FLASH_VOCABULARY_PREFILTER_SCHEMA",
    "run_flash_vocabulary_prefilter",
    "verify_flash_vocabulary_prefilter_receipt",
    "verify_flash_vocabulary_prefilter_receipt_path",
    "FLASH_REPRESENTATION_EXPERIMENT_SCHEMA",
    "run_flash_representation_experiment",
    "FLASH_TRANSFORM_PARITY_SCHEMA",
    "run_flash_transform_parity",
    "FLASH_LOADER_ROUNDTRIP_SCHEMA",
    "run_flash_loader_roundtrip",
    "FLASH_GRAPH_COMPONENT_SCHEMA",
    "run_flash_graph_component",
    "FLASH_COMPONENT_BODY_SCHEMA",
    "run_flash_component_body",
    "FLASH_MATRIX_COMPONENT_BODY_SCHEMA",
    "run_flash_matrix_component_body",
    "FLASH_ROUTER_GRAPH_SCHEMA",
    "run_flash_router_graph",
    "FLASH_ROUTER_SELECTION_SCHEMA",
    "run_flash_router_selection",
    "FLASH_ROUTER_REPRESENTATION_AB_SCHEMA",
    "run_flash_router_representation_ab",
    "FLASH_COMPONENT_CAMPAIGN_SCHEMA",
    "run_flash_component_campaign",
    "PREBOARD_SCHEMA",
    "run_preboard",
    "INITIAL_CHARGE_SCHEMA",
    "create_initial_charge",
    "TRANSFER_MAP_SCHEMA",
    "PRECEDENT_MAP_SCHEMA",
    "write_science_maps",
    "DENSE_NF_AB_SCHEMA",
    "evaluate_ab",
    "run_ab_scaffold",
    "FPGA_PREBOARD_SCHEMA",
    "run_fpga_preboard",
    "PROTECTED_BENCHMARK_WATCHER_SCHEMA",
    "run_protected_benchmark_watcher",
    "PROTECTED_ACCELERATOR_BENCHMARK_SCHEMA",
    "run_protected_accelerator_benchmark",
    "MODELLAKE_SUPERVISION_SCHEMA",
    "run_model_lake_supervision",
    "OVERNIGHT_HANDOFF_SCHEMA",
    "build_handoff",
    "AgentState",
    "mission_state",
    "workunit_state",
    "Capability",
    "CapabilityContract",
    "GenerationRequest",
    "GenerationResponse",
    "ModelProvider",
    "ProviderFailure",
    "ProviderHealth",
    "ProviderReceipt",
    "ResidentProfile",
    "ResidentProvider",
    "RolePolicy",
    "RoleRouter",
    "PhysicalGraph",
    "DIAGNOSTIC_BENCHMARK_CLASSES",
    "NR_PRIMITIVES",
    "PROTECTED_BENCHMARK_CLASSES",
    "compile_physical_graph",
    "score_physical_candidates",
    "ANEProvider",
    "NOMENCLATURE_VERSION",
    "ResultEnvelope",
    "build_result_envelope",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "default_tool_registry",
    "observe",
    "classify_bytes",
    "capture",
    "probe",
    "profile",
    "report",
    "run_matrix",
    "see",
    "hold",
    "know",
    "check",
    "prove",
    "compact_surface",
    "perception_disposition",
]


def __getattr__(name: str) -> Any:
    """Load executable operational gates lazily so ``python -m`` is clean."""
    spec = _CORE_EXPORTS.get(name)
    if spec is not None:
        module_name, symbol = spec
        value = getattr(import_module(module_name), symbol)
        globals()[name] = value
        return value
    if name in {"RECOVERY_GATE_SCHEMA", "run_recovery_gate"}:
        from hawking import recovery_gate

        return recovery_gate.SCHEMA if name == "RECOVERY_GATE_SCHEMA" else recovery_gate.run_recovery_gate
    if name in {"RESEARCH_GATE_SCHEMA", "run_research_gate"}:
        from hawking import research_gate

        return research_gate.SCHEMA if name == "RESEARCH_GATE_SCHEMA" else research_gate.run_research_gate
    if name in {"PERCEPTION_GATE_SCHEMA", "run_perception_gate"}:
        perception_gate = import_module("hawking.perception_gate")

        return perception_gate.SCHEMA if name == "PERCEPTION_GATE_SCHEMA" else perception_gate.run_perception_gate
    if name in {"NATIVE_GATE_SCHEMA", "run_native_gate"}:
        from hawking import native_gate

        return native_gate.SCHEMA if name == "NATIVE_GATE_SCHEMA" else native_gate.run_native_gate
    if name in {"RESIDENT_GATE_SCHEMA", "run_resident_gate"}:
        from hawking import resident_gate

        return resident_gate.SCHEMA if name == "RESIDENT_GATE_SCHEMA" else resident_gate.run_resident_gate
    if name in {"NATIVE_MISSION_GATE_SCHEMA", "run_native_mission_gate"}:
        from hawking import native_mission_gate

        return native_mission_gate.SCHEMA if name == "NATIVE_MISSION_GATE_SCHEMA" else native_mission_gate.run_native_mission_gate
    if name in {"AUTONOMY_GATE_SCHEMA", "run_autonomy_gate", "UNATTENDED_WINDOW_SCHEMA", "run_unattended_window"}:
        from hawking import autonomy_gate

        if name == "AUTONOMY_GATE_SCHEMA":
            return autonomy_gate.SCHEMA
        if name == "UNATTENDED_WINDOW_SCHEMA":
            return autonomy_gate.WINDOW_SCHEMA
        return autonomy_gate.run_autonomy_gate if name == "run_autonomy_gate" else autonomy_gate.run_unattended_window
    if name in {"ACCELERATOR_REGRESSION_SCHEMA", "run_accelerator_regression"}:
        from hawking.agentos import accelerator_regression

        return accelerator_regression.SCHEMA if name == "ACCELERATOR_REGRESSION_SCHEMA" else accelerator_regression.run_accelerator_regression
    if name in {"QWEN27_RUNTIME_IDENTITY_SCHEMA", "QWEN27_RUNTIME_DIFF_SCHEMA", "run_runtime_archaeology"}:
        from hawking.agentos import qwen27_runtime_identity

        if name == "QWEN27_RUNTIME_IDENTITY_SCHEMA":
            return qwen27_runtime_identity.IDENTITY_SCHEMA
        if name == "QWEN27_RUNTIME_DIFF_SCHEMA":
            return qwen27_runtime_identity.DIFF_SCHEMA
        return qwen27_runtime_identity.run_runtime_archaeology
    if name in {"QWEN27_MLP_DIAGNOSTIC_SCHEMA", "run_qwen27_mlp_diagnostic_ab"}:
        from hawking.agentos import qwen27_mlp_diagnostic

        return qwen27_mlp_diagnostic.SCHEMA if name == "QWEN27_MLP_DIAGNOSTIC_SCHEMA" else qwen27_mlp_diagnostic.run_qwen27_mlp_diagnostic_ab
    if name in {"QWEN38_FUSION_AUDIT_SCHEMA", "run_qwen38_fusion_source_audit"}:
        from hawking.agentos import qwen38_fusion_audit

        return qwen38_fusion_audit.SCHEMA if name == "QWEN38_FUSION_AUDIT_SCHEMA" else qwen38_fusion_audit.run_qwen38_fusion_source_audit
    if name in {"MODELLAKE_CENSUS_SCHEMA", "run_modellake_census"}:
        from hawking import modellake_gate

        return modellake_gate.SCHEMA if name == "MODELLAKE_CENSUS_SCHEMA" else modellake_gate.run_modellake_census
    if name in {"FLASH_SCIENCE_SCHEMA", "run_flash_science_gate"}:
        from hawking.agentos import flash_science

        return flash_science.SCHEMA if name == "FLASH_SCIENCE_SCHEMA" else flash_science.run_flash_science_gate
    if name in {"FLASH_EXECUTABLE_SCHEMA", "run_flash_executable_scaffold"}:
        from hawking.agentos import flash_executable

        return flash_executable.SCHEMA if name == "FLASH_EXECUTABLE_SCHEMA" else flash_executable.run_flash_executable_scaffold
    if name in {"FLASH_TENSOR_PROBE_SCHEMA", "run_flash_tensor_probe"}:
        from hawking.agentos import flash_tensor_probe

        return flash_tensor_probe.SCHEMA if name == "FLASH_TENSOR_PROBE_SCHEMA" else flash_tensor_probe.run_flash_tensor_probe
    if name in {
        "FLASH_VOCABULARY_PREFILTER_SCHEMA",
        "run_flash_vocabulary_prefilter",
        "verify_flash_vocabulary_prefilter_receipt",
        "verify_flash_vocabulary_prefilter_receipt_path",
    }:
        from hawking.agentos import flash_vocabulary_prefilter

        if name == "FLASH_VOCABULARY_PREFILTER_SCHEMA":
            return flash_vocabulary_prefilter.SCHEMA
        return getattr(flash_vocabulary_prefilter, name)
    if name in {"FLASH_REPRESENTATION_EXPERIMENT_SCHEMA", "run_flash_representation_experiment"}:
        from hawking.agentos import flash_representation_experiment

        return flash_representation_experiment.SCHEMA if name == "FLASH_REPRESENTATION_EXPERIMENT_SCHEMA" else flash_representation_experiment.run_flash_representation_experiment
    if name in {"FLASH_TRANSFORM_PARITY_SCHEMA", "run_flash_transform_parity"}:
        from hawking.agentos import flash_transform_parity

        return flash_transform_parity.SCHEMA if name == "FLASH_TRANSFORM_PARITY_SCHEMA" else flash_transform_parity.run_flash_transform_parity
    if name in {"FLASH_LOADER_ROUNDTRIP_SCHEMA", "run_flash_loader_roundtrip"}:
        from hawking.agentos import flash_loader_roundtrip

        return flash_loader_roundtrip.SCHEMA if name == "FLASH_LOADER_ROUNDTRIP_SCHEMA" else flash_loader_roundtrip.run_flash_loader_roundtrip
    if name in {"FLASH_GRAPH_COMPONENT_SCHEMA", "run_flash_graph_component"}:
        from hawking.agentos import flash_graph_component

        return flash_graph_component.SCHEMA if name == "FLASH_GRAPH_COMPONENT_SCHEMA" else flash_graph_component.run_flash_graph_component
    if name in {"FLASH_COMPONENT_BODY_SCHEMA", "run_flash_component_body"}:
        from hawking.agentos import flash_component_body

        return flash_component_body.SCHEMA if name == "FLASH_COMPONENT_BODY_SCHEMA" else flash_component_body.run_flash_component_body
    if name in {"FLASH_MATRIX_COMPONENT_BODY_SCHEMA", "run_flash_matrix_component_body"}:
        from hawking.agentos import flash_matrix_component_body

        return flash_matrix_component_body.SCHEMA if name == "FLASH_MATRIX_COMPONENT_BODY_SCHEMA" else flash_matrix_component_body.run_flash_matrix_component_body
    if name in {"FLASH_ROUTER_GRAPH_SCHEMA", "run_flash_router_graph"}:
        from hawking.agentos import flash_router_graph

        return flash_router_graph.SCHEMA if name == "FLASH_ROUTER_GRAPH_SCHEMA" else flash_router_graph.run_flash_router_graph
    if name in {"FLASH_ROUTER_SELECTION_SCHEMA", "run_flash_router_selection"}:
        from hawking.agentos import flash_router_selection

        return flash_router_selection.SCHEMA if name == "FLASH_ROUTER_SELECTION_SCHEMA" else flash_router_selection.run_flash_router_selection
    if name in {"FLASH_ROUTER_REPRESENTATION_AB_SCHEMA", "run_flash_router_representation_ab"}:
        from hawking.agentos import flash_router_representation_ab

        return flash_router_representation_ab.SCHEMA if name == "FLASH_ROUTER_REPRESENTATION_AB_SCHEMA" else flash_router_representation_ab.run_flash_router_representation_ab
    if name in {"FLASH_COMPONENT_CAMPAIGN_SCHEMA", "run_flash_component_campaign"}:
        from hawking.agentos import flash_component_campaign

        return flash_component_campaign.SCHEMA if name == "FLASH_COMPONENT_CAMPAIGN_SCHEMA" else flash_component_campaign.run_flash_component_campaign
    if name in {"PREBOARD_SCHEMA", "run_preboard"}:
        from hawking.agentos import preboard

        return preboard.SCHEMA if name == "PREBOARD_SCHEMA" else preboard.run_preboard
    if name in {"INITIAL_CHARGE_SCHEMA", "create_initial_charge"}:
        from hawking.agentos import charge

        return charge.SCHEMA if name == "INITIAL_CHARGE_SCHEMA" else charge.create_initial_charge
    if name in {"TRANSFER_MAP_SCHEMA", "PRECEDENT_MAP_SCHEMA", "write_science_maps"}:
        from hawking.agentos import science_maps

        if name == "TRANSFER_MAP_SCHEMA":
            return science_maps.TRANSFER_SCHEMA
        if name == "PRECEDENT_MAP_SCHEMA":
            return science_maps.PRECEDENT_SCHEMA
        return science_maps.write_science_maps
    if name in {"DENSE_NF_AB_SCHEMA", "evaluate_ab", "run_ab_scaffold"}:
        from hawking.agentos import representation_ab

        if name == "DENSE_NF_AB_SCHEMA":
            return representation_ab.SCHEMA
        if name == "evaluate_ab":
            return representation_ab.evaluate_ab
        return representation_ab.run_ab_scaffold
    if name in {"FPGA_PREBOARD_SCHEMA", "run_fpga_preboard"}:
        from hawking.agentos import fpga_preboard

        return fpga_preboard.SCHEMA if name == "FPGA_PREBOARD_SCHEMA" else fpga_preboard.run_fpga_preboard
    if name in {"PROTECTED_BENCHMARK_WATCHER_SCHEMA", "run_protected_benchmark_watcher"}:
        from hawking.agentos import protected_benchmark_watcher

        return protected_benchmark_watcher.SCHEMA if name == "PROTECTED_BENCHMARK_WATCHER_SCHEMA" else protected_benchmark_watcher.run_protected_benchmark_watcher
    if name in {"PROTECTED_ACCELERATOR_BENCHMARK_SCHEMA", "run_protected_accelerator_benchmark"}:
        from hawking.agentos import protected_accelerator_benchmark

        return protected_accelerator_benchmark.SCHEMA if name == "PROTECTED_ACCELERATOR_BENCHMARK_SCHEMA" else protected_accelerator_benchmark.run_protected_accelerator_benchmark
    if name in {"MODELLAKE_SUPERVISION_SCHEMA", "run_model_lake_supervision"}:
        from hawking.agentos import modellake_supervisor

        return modellake_supervisor.SCHEMA if name == "MODELLAKE_SUPERVISION_SCHEMA" else modellake_supervisor.run_model_lake_supervision
    if name in {"OVERNIGHT_HANDOFF_SCHEMA", "build_handoff"}:
        from hawking.agentos import handoff

        return handoff.SCHEMA if name == "OVERNIGHT_HANDOFF_SCHEMA" else handoff.build_handoff
    raise AttributeError(name)
