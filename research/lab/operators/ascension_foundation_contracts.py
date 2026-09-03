"""Exact V3 build-fabric, resource, and Agent-OS foundation contracts.

These lists make foundational requirements executable as data instead of vague
status booleans.  They are configuration contracts only: a future receipt must
show direct evidence for every named element before the lifecycle can certify
the corresponding V3 foundation state.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lab.receipts import seal


SCHEMA = "hawking.ascension.v3_foundation_contracts.v1"
FILENAME = "ASCENSION_V3_FOUNDATION_CONTRACTS.json"

GROK_LANE_CLASSES: tuple[str, ...] = (
    "architecture_reconnaissance",
    "paper_and_source_research",
    "family_adapters",
    "kernel_prototypes",
    "gravity_representation_prototypes",
    "qat_doctor_scaffolding",
    "hcli_wiring",
    "agent_os_subsystems",
    "retrieval_tool_gateways",
    "memory_and_skills",
    "storage_cleanup_tooling",
    "tests",
    "benchmarks",
    "fixtures",
    "documentation",
    "migration",
    "code_generation",
    "profiling_analysis",
    "adversarial_review",
)

GROK_LANE_CONTRACT_FIELDS: tuple[str, ...] = (
    "goal",
    "owned_files",
    "forbidden_files",
    "input_receipts",
    "acceptance_tests",
    "expected_outputs",
    "resource_class",
    "merge_dependencies",
    "rollback",
)

GROK_RESOURCE_CLASSES: tuple[str, ...] = (
    "GPU_HEAVY",
    "CPU_HEAVY",
    "NETWORK_HEAVY",
    "DISK_HEAVY",
    "LIGHT_CODE",
    "RESEARCH",
    "TEST",
    "DOCS",
)

GROK_SCHEDULER_RULES: tuple[str, ...] = (
    "one_exclusive_gpu_owner_by_default",
    "cpu_research_docs_may_overlap_gpu",
    "network_prefetch_overlaps_compute",
    "disk_compaction_avoids_source_critical_io",
    "tests_run_against_sealed_commits",
    "no_lane_competes_with_active_full_model_benchmark",
    "concurrent_gpu_requires_measured_aggregate_progress",
    "lanes_are_isolated_branch_worktree_no_shared_dirty_files",
    "no_lane_auto_merges",
    "controller_independently_reviews_and_integrates",
)

RESOURCE_TELEMETRY_FIELDS: tuple[str, ...] = (
    "cpu_utilization_and_frequency",
    "gpu_utilization",
    "gpu_counters",
    "memory_bandwidth",
    "memory_pressure",
    "swap_and_swap_growth",
    "disk_throughput_and_latency",
    "network_throughput_and_retries",
    "thermal_state",
    "power_energy",
    "process_ownership",
    "foreground_user_activity",
)

TASK_SCHEDULER_FIELDS: tuple[str, ...] = (
    "resource_class",
    "dependency",
    "expected_scientific_value",
    "critical_path_status",
    "preemptibility",
    "checkpoint_cost",
)

SCHEDULER_SELECTION_OBJECTIVES: tuple[str, ...] = (
    "critical_path_reduction",
    "information_gain",
    "expected_reuse",
    "marginal_progress_per_watt",
    "risk_of_invalidation",
)

PRESSURE_MODES: tuple[str, ...] = ("GREEN", "YELLOW", "RED", "CRITICAL")

ENERGY_EVIDENCE_OUTPUTS: tuple[str, ...] = (
    "ASCENSION_V3_RESOURCE_ATLAS",
    "ASCENSION_V3_POWER_PROFILES",
    "ASCENSION_V3_CONCURRENCY_FRONTIER",
    "ASCENSION_V3_VERIFIED_WORK_PER_JOULE",
)

AGENT_ROLES: tuple[str, ...] = (
    "campaign_director",
    "researcher",
    "kernel_engineer",
    "gravity_engineer",
    "runtime_engineer",
    "profiling_analyst",
    "test_author",
    "integrator",
    "storage_steward",
    "security_adversary",
    "documentation_curator",
    "replicator",
)

AGENT_SCHEDULER_CAPABILITIES: tuple[str, ...] = (
    "session_queues",
    "priorities",
    "budgets",
    "continuous_batching",
    "tool_wait_suspension",
    "checkpoint_resume",
    "prefix_groups",
    "kv_state_allocation",
    "fairness",
    "starvation",
    "worktree_ownership",
    "model_residency",
    "agent_count_frontier_by_verified_tasks_per_hour",
)

CONTEXT_COMPILER_INPUTS: tuple[str, ...] = (
    "current_goal",
    "relevant_source_slices",
    "architecture_graph",
    "active_hypotheses",
    "latest_receipts",
    "negative_science",
    "tool_outputs",
    "unresolved_decisions",
)

CONTEXT_COMPILER_PROPERTIES: tuple[str, ...] = (
    "content_addressed",
    "source_linked",
    "versioned",
    "cacheable",
    "reconstructable",
    "preserves_critical_constraints",
    "preserves_file_identities",
    "preserves_claim_status",
    "preserves_measurement_provenance",
)

KV_STATE_CAPABILITIES: tuple[str, ...] = (
    "paged_or_segmented_state",
    "prefix_sharing",
    "readonly_prefix_deduplication",
    "session_checkpoint",
    "cold_session_eviction",
    "rehydration",
    "context_compaction",
    "state_integrity_hashes",
    "model_revision_binding",
    "no_mutable_prompt_state_sharing",
)

RETRIEVAL_INDEXES: tuple[str, ...] = (
    "current_papers",
    "official_docs",
    "source_repositories",
    "hawking_symbols",
    "commits",
    "receipts",
    "kernel_genome",
    "representation_genome",
    "negative_science",
    "tools",
    "skills",
    "models",
    "datasets",
)

TOOL_GATEWAY_CAPABILITIES: tuple[str, ...] = (
    "credentials",
    "versions",
    "health",
    "schemas",
    "effect_permissions",
    "session_affinity",
    "timeouts",
    "rollback",
)

MEMORY_TIERS: tuple[str, ...] = (
    "L0_ACTIVE",
    "L1_SESSION",
    "L2_PROJECT",
    "L3_SKILLS",
    "L4_GRAVEYARD",
    "L5_ARCHIVE",
    "L6_KERNEL_GENOME",
    "L7_REPRESENTATION_GENOME",
)

AGENT_OS_PERFORMANCE_GATES: tuple[str, ...] = (
    "one_session_hcli_ge_95_percent_raw_decode",
    "two_four_six_eight_session_tests",
    "verified_tasks_per_hour_improves_with_useful_concurrency",
    "no_starvation",
    "tool_waits_release_inference_capacity",
    "session_checkpoint_restart",
    "endpoint_restart",
    "context_compaction",
    "kv_state_fits_manager_residency",
    "search_tool_retrieval_does_not_flood_context",
    "agent_os_overhead_measured_and_attributed",
    "no_unexplained_agent_os_latency_bucket_above_5_percent",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def foundation_contracts(*, bible_sha256: str | None = None) -> dict[str, Any]:
    return seal(
        {
            "schema": SCHEMA,
            "status": "CONTROLLER_CONFIGURATION_ONLY",
            "recorded_at": _utc_now(),
            "bible_sha256": bible_sha256,
            "grok_build_fabric": {
                "lane_classes": list(GROK_LANE_CLASSES),
                "lane_contract_fields": list(GROK_LANE_CONTRACT_FIELDS),
                "resource_classes": list(GROK_RESOURCE_CLASSES),
                "scheduler_rules": list(GROK_SCHEDULER_RULES),
            },
            "resource_governor": {
                "telemetry_fields": list(RESOURCE_TELEMETRY_FIELDS),
                "task_scheduler_fields": list(TASK_SCHEDULER_FIELDS),
                "selection_objectives": list(SCHEDULER_SELECTION_OBJECTIVES),
                "pressure_modes": list(PRESSURE_MODES),
                "energy_evidence_outputs": list(ENERGY_EVIDENCE_OUTPUTS),
            },
            "agent_os": {
                "roles": list(AGENT_ROLES),
                "scheduler_capabilities": list(AGENT_SCHEDULER_CAPABILITIES),
                "context_compiler_inputs": list(CONTEXT_COMPILER_INPUTS),
                "context_compiler_properties": list(CONTEXT_COMPILER_PROPERTIES),
                "kv_state_capabilities": list(KV_STATE_CAPABILITIES),
                "retrieval_indexes": list(RETRIEVAL_INDEXES),
                "tool_gateway_capabilities": list(TOOL_GATEWAY_CAPABILITIES),
                "memory_tiers": list(MEMORY_TIERS),
                "performance_gates": list(AGENT_OS_PERFORMANCE_GATES),
            },
            "claim_boundary": {
                "configuration_is_not_live_hcli_agent_os": True,
                "configuration_is_not_energy_measurement": True,
                "configuration_is_not_grok_lane_completion": True,
                "only_direct_receipts_can_certify_foundation_states": True,
            },
        }
    )


def write_foundation_contracts(root: str | Path, *, bible_sha256: str | None = None) -> dict[str, Any]:
    resolved = Path(root).expanduser().resolve()
    document = foundation_contracts(bible_sha256=bible_sha256)
    _atomic_json(resolved / FILENAME, document)
    return document


__all__ = [
    "AGENT_OS_PERFORMANCE_GATES",
    "AGENT_ROLES",
    "AGENT_SCHEDULER_CAPABILITIES",
    "CONTEXT_COMPILER_INPUTS",
    "CONTEXT_COMPILER_PROPERTIES",
    "ENERGY_EVIDENCE_OUTPUTS",
    "FILENAME",
    "GROK_LANE_CLASSES",
    "GROK_LANE_CONTRACT_FIELDS",
    "GROK_RESOURCE_CLASSES",
    "GROK_SCHEDULER_RULES",
    "KV_STATE_CAPABILITIES",
    "MEMORY_TIERS",
    "PRESSURE_MODES",
    "RESOURCE_TELEMETRY_FIELDS",
    "RETRIEVAL_INDEXES",
    "SCHEMA",
    "SCHEDULER_SELECTION_OBJECTIVES",
    "TASK_SCHEDULER_FIELDS",
    "TOOL_GATEWAY_CAPABILITIES",
    "foundation_contracts",
    "write_foundation_contracts",
]
