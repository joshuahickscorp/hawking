"""AgentOS: Goal, obligations, WorkUnit DAG, scheduler, repair,
mutation authority, verifier orchestration, mission persistence, steering.

Canonical implementation modules remain ``hcli.goal``, ``hcli.workunit``,
``hcli.scheduler``, ``hcli.mission``, ``hcli.ledger``, ``hcli.steering``,
``hcli.mutation``, ``hcli.verifier_pipeline``, ``hcli.dag_store``,
``hcli.executors``, ``hcli.resources``. Re-exported here so ownership is
an importable package, not a comment — the class objects are the same.
"""
from hcli.dag_store import DagStore
from hcli.executors import WorkUnitExecutor
from hcli.goal import GoalCompiler, WorkerPacket, compile_worker_context
from hcli.ledger import Ledger
from hcli.mission import Mission
from hcli.mutation import MutationError
from hcli.resources import MutationLock, ResourceClass, ResourceLimits
from hcli.scheduler import Scheduler
from hcli.steering import SteeringQueue
from hcli.verifier_pipeline import command_is_admissible
from hcli.workunit import WorkUnit
from hcli.agentos.runtime import AgentOS
from hcli.agentos.background import BackgroundJob, BackgroundJobStore
from hcli.goal_bank import GoalBank, GoalBankError
from hcli.knowledge import KnowledgeError, KnowledgeStore
from hcli.agentos.resident import (
    ResidentConfig,
    ResidentBodyRegistry,
    ResidentDaemon,
    ResidentStore,
    ResidentSupervisor,
    admit_evidence_children,
    memory_decision,
    resident_behavior,
    start_resident,
)
from hcli.agentos.states import AgentState, mission_state, workunit_state
from hcli.providers import (
    Capability,
    CapabilityContract,
    GenerationRequest,
    GenerationResponse,
    ModelProvider,
    ProviderFailure,
    ProviderHealth,
    ProviderReceipt,
    ResidentProfile,
    ResidentProvider,
    RolePolicy,
    RoleRouter,
)
from hcli.physical_graph import (
    DIAGNOSTIC_BENCHMARK_CLASSES,
    NR_PRIMITIVES,
    PhysicalGraph,
    PROTECTED_BENCHMARK_CLASSES,
    compile_physical_graph,
    score_physical_candidates,
)
from hcli.ane_provider import ANEProvider
from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.result_envelope import ResultEnvelope, build_result_envelope
from hcli.tool_registry import ToolContext, ToolRegistry, ToolResult, ToolSpec, default_tool_registry

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
    "VMCP_GATE_SCHEMA",
    "run_vmcp_gate",
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
]


def __getattr__(name: str):
    """Load executable operational gates lazily so ``python -m`` is clean."""
    if name in {"RECOVERY_GATE_SCHEMA", "run_recovery_gate"}:
        from hcli.agentos import recovery

        return recovery.SCHEMA if name == "RECOVERY_GATE_SCHEMA" else recovery.run_recovery_gate
    if name in {"RESEARCH_GATE_SCHEMA", "run_research_gate"}:
        from hcli.agentos import research

        return research.SCHEMA if name == "RESEARCH_GATE_SCHEMA" else research.run_research_gate
    if name in {"VMCP_GATE_SCHEMA", "run_vmcp_gate"}:
        from hcli.agentos import vmcp_gate

        return vmcp_gate.SCHEMA if name == "VMCP_GATE_SCHEMA" else vmcp_gate.run_vmcp_gate
    if name in {"NATIVE_GATE_SCHEMA", "run_native_gate"}:
        from hcli.agentos import native_gate

        return native_gate.SCHEMA if name == "NATIVE_GATE_SCHEMA" else native_gate.run_native_gate
    if name in {"RESIDENT_GATE_SCHEMA", "run_resident_gate"}:
        from hcli.agentos import resident_gate

        return resident_gate.SCHEMA if name == "RESIDENT_GATE_SCHEMA" else resident_gate.run_resident_gate
    if name in {"NATIVE_MISSION_GATE_SCHEMA", "run_native_mission_gate"}:
        from hcli.agentos import native_mission_gate

        return native_mission_gate.SCHEMA if name == "NATIVE_MISSION_GATE_SCHEMA" else native_mission_gate.run_native_mission_gate
    if name in {"AUTONOMY_GATE_SCHEMA", "run_autonomy_gate", "UNATTENDED_WINDOW_SCHEMA", "run_unattended_window"}:
        from hcli.agentos import autonomy_gate

        if name == "AUTONOMY_GATE_SCHEMA":
            return autonomy_gate.SCHEMA
        if name == "UNATTENDED_WINDOW_SCHEMA":
            return autonomy_gate.WINDOW_SCHEMA
        return autonomy_gate.run_autonomy_gate if name == "run_autonomy_gate" else autonomy_gate.run_unattended_window
    if name in {"ACCELERATOR_REGRESSION_SCHEMA", "run_accelerator_regression"}:
        from hcli.agentos import accelerator_regression

        return accelerator_regression.SCHEMA if name == "ACCELERATOR_REGRESSION_SCHEMA" else accelerator_regression.run_accelerator_regression
    if name in {"QWEN27_RUNTIME_IDENTITY_SCHEMA", "QWEN27_RUNTIME_DIFF_SCHEMA", "run_runtime_archaeology"}:
        from hcli.agentos import qwen27_runtime_identity

        if name == "QWEN27_RUNTIME_IDENTITY_SCHEMA":
            return qwen27_runtime_identity.IDENTITY_SCHEMA
        if name == "QWEN27_RUNTIME_DIFF_SCHEMA":
            return qwen27_runtime_identity.DIFF_SCHEMA
        return qwen27_runtime_identity.run_runtime_archaeology
    if name in {"QWEN27_MLP_DIAGNOSTIC_SCHEMA", "run_qwen27_mlp_diagnostic_ab"}:
        from hcli.agentos import qwen27_mlp_diagnostic

        return qwen27_mlp_diagnostic.SCHEMA if name == "QWEN27_MLP_DIAGNOSTIC_SCHEMA" else qwen27_mlp_diagnostic.run_qwen27_mlp_diagnostic_ab
    if name in {"QWEN38_FUSION_AUDIT_SCHEMA", "run_qwen38_fusion_source_audit"}:
        from hcli.agentos import qwen38_fusion_audit

        return qwen38_fusion_audit.SCHEMA if name == "QWEN38_FUSION_AUDIT_SCHEMA" else qwen38_fusion_audit.run_qwen38_fusion_source_audit
    if name in {"MODELLAKE_CENSUS_SCHEMA", "run_modellake_census"}:
        from hcli.agentos import modellake_gate

        return modellake_gate.SCHEMA if name == "MODELLAKE_CENSUS_SCHEMA" else modellake_gate.run_modellake_census
    if name in {"FLASH_SCIENCE_SCHEMA", "run_flash_science_gate"}:
        from hcli.agentos import flash_science

        return flash_science.SCHEMA if name == "FLASH_SCIENCE_SCHEMA" else flash_science.run_flash_science_gate
    if name in {"FLASH_EXECUTABLE_SCHEMA", "run_flash_executable_scaffold"}:
        from hcli.agentos import flash_executable

        return flash_executable.SCHEMA if name == "FLASH_EXECUTABLE_SCHEMA" else flash_executable.run_flash_executable_scaffold
    if name in {"FLASH_TENSOR_PROBE_SCHEMA", "run_flash_tensor_probe"}:
        from hcli.agentos import flash_tensor_probe

        return flash_tensor_probe.SCHEMA if name == "FLASH_TENSOR_PROBE_SCHEMA" else flash_tensor_probe.run_flash_tensor_probe
    if name in {"FLASH_REPRESENTATION_EXPERIMENT_SCHEMA", "run_flash_representation_experiment"}:
        from hcli.agentos import flash_representation_experiment

        return flash_representation_experiment.SCHEMA if name == "FLASH_REPRESENTATION_EXPERIMENT_SCHEMA" else flash_representation_experiment.run_flash_representation_experiment
    if name in {"FLASH_TRANSFORM_PARITY_SCHEMA", "run_flash_transform_parity"}:
        from hcli.agentos import flash_transform_parity

        return flash_transform_parity.SCHEMA if name == "FLASH_TRANSFORM_PARITY_SCHEMA" else flash_transform_parity.run_flash_transform_parity
    if name in {"FLASH_LOADER_ROUNDTRIP_SCHEMA", "run_flash_loader_roundtrip"}:
        from hcli.agentos import flash_loader_roundtrip

        return flash_loader_roundtrip.SCHEMA if name == "FLASH_LOADER_ROUNDTRIP_SCHEMA" else flash_loader_roundtrip.run_flash_loader_roundtrip
    if name in {"FLASH_GRAPH_COMPONENT_SCHEMA", "run_flash_graph_component"}:
        from hcli.agentos import flash_graph_component

        return flash_graph_component.SCHEMA if name == "FLASH_GRAPH_COMPONENT_SCHEMA" else flash_graph_component.run_flash_graph_component
    if name in {"FLASH_COMPONENT_BODY_SCHEMA", "run_flash_component_body"}:
        from hcli.agentos import flash_component_body

        return flash_component_body.SCHEMA if name == "FLASH_COMPONENT_BODY_SCHEMA" else flash_component_body.run_flash_component_body
    if name in {"FLASH_MATRIX_COMPONENT_BODY_SCHEMA", "run_flash_matrix_component_body"}:
        from hcli.agentos import flash_matrix_component_body

        return flash_matrix_component_body.SCHEMA if name == "FLASH_MATRIX_COMPONENT_BODY_SCHEMA" else flash_matrix_component_body.run_flash_matrix_component_body
    if name in {"FLASH_ROUTER_GRAPH_SCHEMA", "run_flash_router_graph"}:
        from hcli.agentos import flash_router_graph

        return flash_router_graph.SCHEMA if name == "FLASH_ROUTER_GRAPH_SCHEMA" else flash_router_graph.run_flash_router_graph
    if name in {"FLASH_ROUTER_SELECTION_SCHEMA", "run_flash_router_selection"}:
        from hcli.agentos import flash_router_selection

        return flash_router_selection.SCHEMA if name == "FLASH_ROUTER_SELECTION_SCHEMA" else flash_router_selection.run_flash_router_selection
    if name in {"FLASH_ROUTER_REPRESENTATION_AB_SCHEMA", "run_flash_router_representation_ab"}:
        from hcli.agentos import flash_router_representation_ab

        return flash_router_representation_ab.SCHEMA if name == "FLASH_ROUTER_REPRESENTATION_AB_SCHEMA" else flash_router_representation_ab.run_flash_router_representation_ab
    if name in {"FLASH_COMPONENT_CAMPAIGN_SCHEMA", "run_flash_component_campaign"}:
        from hcli.agentos import flash_component_campaign

        return flash_component_campaign.SCHEMA if name == "FLASH_COMPONENT_CAMPAIGN_SCHEMA" else flash_component_campaign.run_flash_component_campaign
    if name in {"PREBOARD_SCHEMA", "run_preboard"}:
        from hcli.agentos import preboard

        return preboard.SCHEMA if name == "PREBOARD_SCHEMA" else preboard.run_preboard
    if name in {"INITIAL_CHARGE_SCHEMA", "create_initial_charge"}:
        from hcli.agentos import charge

        return charge.SCHEMA if name == "INITIAL_CHARGE_SCHEMA" else charge.create_initial_charge
    if name in {"TRANSFER_MAP_SCHEMA", "PRECEDENT_MAP_SCHEMA", "write_science_maps"}:
        from hcli.agentos import science_maps

        if name == "TRANSFER_MAP_SCHEMA":
            return science_maps.TRANSFER_SCHEMA
        if name == "PRECEDENT_MAP_SCHEMA":
            return science_maps.PRECEDENT_SCHEMA
        return science_maps.write_science_maps
    if name in {"DENSE_NF_AB_SCHEMA", "evaluate_ab", "run_ab_scaffold"}:
        from hcli.agentos import representation_ab

        if name == "DENSE_NF_AB_SCHEMA":
            return representation_ab.SCHEMA
        if name == "evaluate_ab":
            return representation_ab.evaluate_ab
        return representation_ab.run_ab_scaffold
    if name in {"FPGA_PREBOARD_SCHEMA", "run_fpga_preboard"}:
        from hcli.agentos import fpga_preboard

        return fpga_preboard.SCHEMA if name == "FPGA_PREBOARD_SCHEMA" else fpga_preboard.run_fpga_preboard
    if name in {"PROTECTED_BENCHMARK_WATCHER_SCHEMA", "run_protected_benchmark_watcher"}:
        from hcli.agentos import protected_benchmark_watcher

        return protected_benchmark_watcher.SCHEMA if name == "PROTECTED_BENCHMARK_WATCHER_SCHEMA" else protected_benchmark_watcher.run_protected_benchmark_watcher
    if name in {"PROTECTED_ACCELERATOR_BENCHMARK_SCHEMA", "run_protected_accelerator_benchmark"}:
        from hcli.agentos import protected_accelerator_benchmark

        return protected_accelerator_benchmark.SCHEMA if name == "PROTECTED_ACCELERATOR_BENCHMARK_SCHEMA" else protected_accelerator_benchmark.run_protected_accelerator_benchmark
    if name in {"MODELLAKE_SUPERVISION_SCHEMA", "run_model_lake_supervision"}:
        from hcli.agentos import modellake_supervisor

        return modellake_supervisor.SCHEMA if name == "MODELLAKE_SUPERVISION_SCHEMA" else modellake_supervisor.run_model_lake_supervision
    if name in {"OVERNIGHT_HANDOFF_SCHEMA", "build_handoff"}:
        from hcli.agentos import handoff

        return handoff.SCHEMA if name == "OVERNIGHT_HANDOFF_SCHEMA" else handoff.build_handoff
    raise AttributeError(name)
