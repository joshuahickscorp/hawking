#!/usr/bin/env python3
"""Authority census: every duplicate live implementation, and which one survives.

Read-only over the repository. Writes one receipt:

    receipts/headless/AUTHORITY_CENSUS.json

This worktree is a sparse checkout. Working-tree grep is not evidence of
absence. Every path is resolved against HEAD via `git cat-file` / `git show`.
Line numbers are taken from the HEAD blob, not from a prior receipt.

Classification (five ways; UNKNOWN if the evidence is not enough):

  canonical_authority        the implementation that should survive
  compatibility_wrapper      thin adapter over a named canonical; kept only
                             while a named verified caller still imports it
  obsolete_implementation    live in the tree, no remaining production caller,
                             or superseded by the canonical
  test_only_implementation   exists to measure or stub; not a control-plane
  historical_implementation  sealed schema / receipt / empty stub / fossil
                             namespace. Do not rename or delete receipts.

A concept with TWO (or more) canonical_authority rows is the finding, not
the list length. Compatibility layers without a named verified caller are
marked removable.

  python3 tools/headless/authority_census.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "hawking.headless.authority_census.v1"

REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts" / "headless" / "AUTHORITY_CENSUS.json"

CONCEPTS = (
    "Goal",
    "mission",
    "WorkUnit",
    "DAG",
    "scheduler",
    "checkpoint",
    "mutation lock",
    "verifier",
    "backend registry",
    "context budget",
    "runtime registry",
    "receipt",
    "experiment",
    "model identity",
    "machine identity",
    "status",
    "retry policy",
)

CLASSES = (
    "canonical_authority",
    "compatibility_wrapper",
    "obsolete_implementation",
    "test_only_implementation",
    "historical_implementation",
    "UNKNOWN",
)

# Tight definition patterns used to hunt uncatalogued live implementations.
# Working-tree git grep is silent on sparse-unmaterialized files; these run
# against HEAD.
SCAN_PATTERNS: Dict[str, str] = {
    "Goal": r"^(class |pub struct |struct )(Goal|GoalCompiler|GoalRecord|GoalStore|GoalNotMet)\b",
    "mission": r"^(class |pub struct |struct )(Mission)\b",
    "WorkUnit": r"^(class |pub struct |struct )(WorkUnit|WorkUnitDAG|WorkUnitExecutor|_WorkItem)\b",
    "DAG": r"^(class |pub struct |struct )(WorkUnitDAG|DagStore|HcliDag|PlanDag|HcliNode)\b",
    "scheduler": r"^(class |pub struct |struct )(Scheduler|AgentScheduler|LaneScheduler)\b",
    "checkpoint": r"^(class |pub struct |struct )(Checkpoint|CheckpointStore|CheckpointRecord|AgentCheckpoint|CheckpointId|CheckpointMeta)\b",
    "mutation lock": r"^(class |pub struct |struct )(MutationLock|SingletonLease)\b",
    "verifier": r"^(class |pub struct |struct )(Verifier|VerifierPipeline|VerificationAuthority|VerificationGate|VerifierRegistry|OracleSuite)\b",
    "backend registry": r"^(class |pub struct |struct )(RuntimeBackend|BackendHealth|RoleRegistry|LlamaServerBackend|MlxServerBackend)\b",
    "context budget": r"^(class |pub struct |struct )(ContextBudget|TokenBudget|RegionBudget|ContextCompiler|PacketBudgetError)\b",
    "runtime registry": r"^(class |pub struct |struct )(RuntimePool|Runtime|ExperimentRuntime|ProgramRuntime)\b",
    "receipt": r"^(class |pub struct |struct )(Receipt|ReceiptAuthority|GateEvidence)\b",
    "experiment": r"^(class |pub struct |struct )(Experiment|ExperimentSpec|ExperimentRuntime)\b",
    "model identity": r"^(class |pub struct |struct )(ModelRegistry|ModelInfo|ModelSpec|ModelIdentity)\b",
    "machine identity": r"^(class |pub struct |struct )(MachineGenome|MachineProbe|MemGate)\b",
    "status": r"^(def format_status|class Status\b|pub struct StatusResponse)\b",
    "retry policy": r"^(class |pub struct |struct )(RetryPolicy|FailureClassification)\b",
}

# Paths whose matches are receipts, lockfiles, or generated noise — never
# an implementation.
SCAN_SKIP_PREFIXES = (
    "receipts/",
    "visionmcp/uv.lock",
    "tools/agentos/genesis_body/Cargo.lock",
)

# Catalog: one live implementation per row. `needle` is matched against the
# HEAD blob to recover the current line. `callers` are paths that must exist
# in HEAD if classification is compatibility_wrapper; an empty callers list
# on a wrapper means removable.
#
# plane:
#   hcli-py          hcli  (live campaign control plane)
#   hide-rs          hide-kernel / hide-backend / hide-protocol
#   hawking-orch     fleet admission
#   hawking-serve    decode-slot batching
#   hawking-context  token-budget packer
#   lab              science campaign engine
#   headless         measurement harnesses
#   ramanujan        math verifier scaffold
#   agentos          genesis / machine_state
#   haider-v0        tools/hcli/bootstrap/snapshots/haider.py bootstrap
#   hawking-speculate speculative-decode accept (name collision)
CatalogRow = Dict[str, Any]

CATALOG: List[CatalogRow] = [
    # ------------------------------------------------------------------ Goal
    {
        "id": "hcli.ledger.Ledger",
        "concept": "Goal",
        "path": "hcli/ledger.py",
        "needle": "class Ledger:",
        "symbol": "Ledger",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": (
            "Mission-level obligation ledger (GOAL.md). Answers 'is the overall "
            "goal satisfied?' via obligation VERIFIED. Identity space is G001-style "
            "ids, distinct from WorkUnit.id."
        ),
        "move": (
            "Keep. Do not merge with WorkUnitDAG or hide-backend GoalStore. "
            "Callers that need 'goal met' must go through Ledger.outcome / "
            "assert_may_complete."
        ),
        "callers": [
            "hcli/mission.py",
            "hcli/commands.py",
            "hcli/steering.py",
        ],
        "evidence": "receipts/headless/DAG_CONSOLIDATION_DECISION.json",
    },
    {
        "id": "hcli.goal.GoalCompiler",
        "concept": "Goal",
        "path": "hcli/goal.py",
        "needle": "class GoalCompiler:",
        "symbol": "GoalCompiler",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": (
            "Compiles natural-language goal text into implement/validate WorkUnits. "
            "A different identity space from Ledger obligations. Documented split, "
            "not an accidental duplicate."
        ),
        "move": (
            "Keep as compiler. Do not teach it to decide GOAL_MET. Engine, "
            "Controller, and Mission already call GoalCompiler().compile."
        ),
        "callers": [
            "hcli/engine.py",
            "hcli/controller.py",
            "hcli/mission.py",
        ],
        "evidence": "receipts/headless/DAG_CONSOLIDATION_DECISION.json",
    },
    {
        "id": "hide-protocol.Goal",
        "concept": "Goal",
        "path": "crates/hide-protocol/src/plan.rs",
        "needle": "pub struct Goal {",
        "symbol": "Goal",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": True,
        "survives": True,
        "role": (
            "HIDE protocol schema (bible sec 14): declaration, not execution. "
            "This crate does not run plans."
        ),
        "move": (
            "Keep as the HIDE wire/schema type. Do not import it into HCLI-py. "
            "A translation layer between GoalId and Ledger G-ids does not exist "
            "and must not be invented as a third Goal type."
        ),
        "callers": ["crates/hide-protocol/src/plan.rs"],
    },
    {
        "id": "hide-backend.GoalStore",
        "concept": "Goal",
        "path": "crates/hide-backend/src/services_goal.rs",
        "needle": "pub struct GoalStore;",
        "symbol": "GoalStore / GoalRecord / GoalVerdict",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": True,
        "survives": True,
        "role": (
            "Durable HIDE session goal over the KV `goals` namespace. One active "
            "goal per session. Evaluated model-free from verify.result evidence."
        ),
        "move": (
            "Keep as HIDE's durable goal. Do not replace HCLI Ledger with this, "
            "or this with HCLI Ledger. TWO real authorities across products."
        ),
        "callers": ["crates/hide-backend/src/services_goal.rs"],
    },
    {
        "id": "haider-v0.haider.py",
        "concept": "Goal",
        "path": "tools/hcli/bootstrap/snapshots/haider.py",
        "needle": "haider — HCLI-v0 bootstrap",
        "symbol": "haider.py (HCLI-v0 loop)",
        "classification": "obsolete_implementation",
        "plane": "haider-v0",
        "two_real": False,
        "survives": False,
        "role": "Gate-zero bootstrap: one llama-server, one scoped edit, one receipt.",
        "move": (
            "Leave in tree (historical bootstrap). Do not route live missions "
            "through it. Live control plane is hcli/."
        ),
        "callers": [],
    },
    # ------------------------------------------------------------------ mission
    {
        "id": "hcli.mission.Mission",
        "concept": "mission",
        "path": "hcli/mission.py",
        "needle": "class Mission:",
        "symbol": "Mission",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": (
            "HCLI run-loop: owns scheduler, heartbeat, cancel, adopted Grok "
            "tasks, phase, and the mission state.json checkpoint. Not the ledger "
            "and not the WorkUnit DAG."
        ),
        "move": "Keep. Lab ExperimentRuntime and hide-kernel AgentDriver stay in their planes.",
        "callers": [
            "hcli/controller.py",
            "hcli/app.py",
        ],
    },
    {
        "id": "hide-kernel.AgentDriver",
        "concept": "mission",
        "path": "crates/hide-kernel/src/machine.rs",
        "needle": "pub struct AgentDriver",
        "symbol": "AgentDriver",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": True,
        "survives": True,
        "role": (
            "HIDE FSM driver (bible ch.02 §4.4). Name collision: this file is "
            "machine.rs but it is the agent loop, not hardware identity."
        ),
        "move": "Keep as HIDE's run-loop. Do not rename for the HCLI-py Mission.",
        "callers": ["crates/hide-kernel/src/machine.rs"],
    },
    {
        "id": "lab.ExperimentRuntime",
        "concept": "mission",
        "path": "research/lab/runtime.py",
        "needle": "class ExperimentRuntime",
        "symbol": "ExperimentRuntime",
        "classification": "canonical_authority",
        "plane": "lab",
        "two_real": True,
        "survives": True,
        "role": "Lab science campaign run-loop (lease, checkpoint, operators, receipts).",
        "move": (
            "Keep. Name collision with 'runtime registry' is documented there. "
            "Do not drive HCLI WorkUnits through this."
        ),
        "callers": ["research/lab/runtime.py"],
    },
    {
        "id": "lab.hcli.self_evolution",
        "concept": "mission",
        "path": "research/lab/hcli/self_evolution.py",
        "needle": "class EvolutionLedger:",
        "symbol": "EvolutionLedger",
        "classification": "obsolete_implementation",
        "plane": "lab",
        "two_real": False,
        "survives": False,
        "role": "Experimental self-evolution ledger under research/lab/hcli, not the live HCLI Mission.",
        "move": "Leave. Not a migration target for hcli/mission.py.",
        "callers": [],
    },
    # ------------------------------------------------------------------ WorkUnit
    {
        "id": "hcli.workunit.WorkUnit",
        "concept": "WorkUnit",
        "path": "hcli/workunit.py",
        "needle": "class WorkUnit:",
        "symbol": "WorkUnit",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": False,
        "survives": True,
        "role": (
            "The WorkUnit dataclass plus identify_ready / transition_status / "
            "assign_ready / emit_repair. Shared identity space with the scheduler."
        ),
        "move": "Keep. All status writes go through transition_status.",
        "callers": [
            "hcli/scheduler.py",
            "hcli/goal.py",
            "hcli/dag_store.py",
            "hcli/mission.py",
        ],
        "evidence": "receipts/headless/DAG_CONSOLIDATION_DECISION.json",
    },
    {
        "id": "hcli.goal.WorkUnitDAG",
        "concept": "WorkUnit",
        "path": "hcli/goal.py",
        "needle": "class WorkUnitDAG:",
        "symbol": "WorkUnitDAG",
        "classification": "compatibility_wrapper",
        "plane": "hcli-py",
        "two_real": False,
        "survives": True,
        "role": (
            "GoalCompiler IR. Readiness is identify_ready; completions walk "
            "transition_status. Persistence uses DagStore. Not a second DAG."
        ),
        "move": (
            "Keep as compiler IR while GoalCompiler / engine.py / "
            "hcli_dag_consolidation_test.py call it. Do not restore the naive "
            "'all deps completed' loop or direct wu.status writes."
        ),
        "callers": [
            "hcli/goal.py",
            "tools/headless/hcli_dag_consolidation_test.py",
        ],
        "evidence": "receipts/headless/DAG_CONSOLIDATION_DECISION.json",
    },
    {
        "id": "hcli.executors.WorkUnitExecutor",
        "concept": "WorkUnit",
        "path": "hcli/executors.py",
        "needle": "class WorkUnitExecutor:",
        "symbol": "WorkUnitExecutor",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": False,
        "survives": True,
        "role": "Executes an already-admitted WorkUnit. Does not invent work or decide readiness.",
        "move": "Keep. Not a second WorkUnit type.",
        "callers": ["hcli/mission.py"],
    },
    {
        "id": "lab._WorkItem",
        "concept": "WorkUnit",
        "path": "research/lab/engine_support.py",
        "needle": "class _WorkItem:",
        "symbol": "_WorkItem",
        "classification": "canonical_authority",
        "plane": "lab",
        "two_real": False,
        "survives": True,
        "role": "Lab experiment step. Different identity space from HCLI WorkUnit.id.",
        "move": "Keep in lab. Do not alias to hcli.workunit.WorkUnit.",
        "callers": ["research/lab/engine_support.py"],
    },
    # ------------------------------------------------------------------ DAG
    {
        "id": "hcli.workunit.identify_ready",
        "concept": "DAG",
        "path": "hcli/workunit.py",
        "needle": "def identify_ready(",
        "symbol": "identify_ready / assign_ready / is_ready",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": (
            "Canonical HCLI DAG logic: repair-aware readiness, retry budget, "
            "resource-class admission, MUTATION lock in assign_ready."
        ),
        "move": "Keep. Scheduler.dispatch and WorkUnitDAG.get_ready_units both call this.",
        "callers": [
            "hcli/scheduler.py",
            "hcli/goal.py",
        ],
        "evidence": "receipts/headless/DAG_CONSOLIDATION_DECISION.json",
    },
    {
        "id": "hcli.dag_store.DagStore",
        "concept": "DAG",
        "path": "hcli/dag_store.py",
        "needle": "class DagStore:",
        "symbol": "DagStore",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": False,
        "survives": True,
        "role": "Durable atomic writer for the HCLI WorkUnit DAG (.hcli/dag.json).",
        "move": "Keep as the only on-disk DAG for HCLI-py.",
        "callers": [
            "hcli/scheduler.py",
            "hcli/goal.py",
        ],
    },
    {
        "id": "hide-kernel.PlanDag",
        "concept": "DAG",
        "path": "crates/hide-kernel/src/plan.rs",
        "needle": "pub struct PlanDag;",
        "symbol": "PlanDag",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": True,
        "survives": True,
        "role": (
            "HIDE plan DAG: ready_steps is 'pending + all deps completed'. No "
            "repair-aware readiness. Used by AgentDriver."
        ),
        "move": (
            "Keep as HIDE's plan DAG. Do not replace HCLI identify_ready with "
            "this, or this with identify_ready. TWO real DAG authorities."
        ),
        "callers": ["crates/hide-kernel/src/machine.rs"],
    },
    {
        "id": "hide-backend.HaiderDag",
        "concept": "DAG",
        "path": "crates/hide-backend/src/haider/dag.rs",
        "needle": "pub struct HaiderDag {",
        "symbol": "HaiderDag / HaiderNode",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": True,
        "survives": True,
        "role": (
            "Rust HAIDER parallel task DAG (disjoint write scopes + MemGate). "
            "Lives under a fossil namespace; it is not aider."
        ),
        "move": (
            "Keep. Do not rename the crate path in this campaign (haider is a "
            "fossil namespace, not architecture). Do not merge with HCLI-py DAG."
        ),
        "callers": [
            "crates/hide-backend/src/haider/lanes.rs",
            "crates/hide-backend/src/haider/mod.rs",
        ],
    },
    # ------------------------------------------------------------------ scheduler
    {
        "id": "hcli.scheduler.Scheduler",
        "concept": "scheduler",
        "path": "hcli/scheduler.py",
        "needle": "class Scheduler:",
        "symbol": "Scheduler",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": (
            "Dispatch existing WorkUnits. FIFO by ready_at. Does not invent work. "
            "complete() refuses without a passing verifier (UnverifiedCompletion)."
        ),
        "move": (
            "Keep. Delete the post-import MAX_REPAIR_* reassignment (merge "
            "shadow) on a later source lane; re-export workunit.py names only. "
            "_remaining_depth is dead in dispatch — do not reintroduce depth sort."
        ),
        "callers": [
            "hcli/mission.py",
            "hcli/controller.py",
        ],
        "evidence": "receipts/headless/HCLI_SCHEDULER_QUALITY.json",
    },
    {
        "id": "hcli.scheduler._remaining_depth",
        "concept": "scheduler",
        "path": "hcli/scheduler.py",
        "needle": "def _remaining_depth(",
        "symbol": "_remaining_depth",
        "classification": "obsolete_implementation",
        "plane": "hcli-py",
        "two_real": False,
        "survives": False,
        "role": (
            "Hop-count helper. Dispatch does not call it (ready_at FIFO). "
            "LOG and MAX_REPAIR_* are redefined after this helper — merge artifact."
        ),
        "move": (
            "Remove on a later source lane together with the duplicated LOG = "
            "and MAX_REPAIR_* assignments. Do not use it as a scheduling key."
        ),
        "callers": [],
        "evidence": "tools/headless/hcli_scheduler_quality.py",
    },
    {
        "id": "headless.remaining_depth",
        "concept": "scheduler",
        "path": "tools/headless/hcli_scheduler_quality.py",
        "needle": "def remaining_depth(",
        "symbol": "remaining_depth",
        "classification": "test_only_implementation",
        "plane": "headless",
        "two_real": False,
        "survives": True,
        "role": "Measurement copy used to refute depth-sort. Not a scheduler.",
        "move": "Keep as the harness that proved depth-sort does not help.",
        "callers": ["tools/headless/hcli_scheduler_quality.py"],
    },
    {
        "id": "lab.engine_support.Scheduler",
        "concept": "scheduler",
        "path": "research/lab/engine_support.py",
        "needle": "class Scheduler:",
        "symbol": "Scheduler",
        "classification": "canonical_authority",
        "plane": "lab",
        "two_real": True,
        "survives": True,
        "role": "Lab experiment-step scheduler over ExperimentSpec.steps.",
        "move": "Keep in lab. Same English name as HCLI Scheduler; different object.",
        "callers": ["research/lab/runtime.py"],
    },
    {
        "id": "hawking-orch.Scheduler",
        "concept": "scheduler",
        "path": "crates/hawking-orch/src/scheduler.rs",
        "needle": "pub struct Scheduler {",
        "symbol": "Scheduler",
        "classification": "canonical_authority",
        "plane": "hawking-orch",
        "two_real": True,
        "survives": True,
        "role": "Energy/thermal/RAM admission for model roles. Not WorkUnit dispatch.",
        "move": "Keep. Do not merge with HCLI-py Scheduler.",
        "callers": ["crates/hawking-orch/src/scheduler.rs"],
    },
    {
        "id": "hawking-serve.batch.Scheduler",
        "concept": "scheduler",
        "path": "crates/hawking-serve/src/batch/scheduler.rs",
        "needle": "pub struct Scheduler {",
        "symbol": "Scheduler",
        "classification": "canonical_authority",
        "plane": "hawking-serve",
        "two_real": True,
        "survives": True,
        "role": "Decode-slot batch packer (KV slots, greedy-first, prefix reuse).",
        "move": "Keep. This is inference batching, not campaign dispatch.",
        "callers": ["crates/hawking-serve/src/batch/scheduler.rs"],
    },
    {
        "id": "hide-backend.AgentScheduler",
        "concept": "scheduler",
        "path": "crates/hide-backend/src/agent_scheduler.rs",
        "needle": "pub struct AgentScheduler {",
        "symbol": "AgentScheduler",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": True,
        "survives": True,
        "role": "HIDE backend agent/job scheduler (policy + metrics + checkpoint refs).",
        "move": "Keep in hide-backend. Not HCLI-py Scheduler.",
        "callers": ["crates/hide-backend/src/agent_scheduler.rs"],
    },
    {
        "id": "hide-backend.LaneScheduler",
        "concept": "scheduler",
        "path": "crates/hide-backend/src/haider/lanes.rs",
        "needle": "pub struct LaneScheduler {",
        "symbol": "LaneScheduler",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": False,
        "survives": True,
        "role": "MemGate-controlled HAIDER lane scheduler (Architect/Implementer/Adversary).",
        "move": "Keep with HaiderDag. Not a second HCLI-py scheduler.",
        "callers": ["crates/hide-backend/src/haider/lanes.rs"],
    },
    # ------------------------------------------------------------------ checkpoint
    {
        "id": "hcli.dag_store.persist",
        "concept": "checkpoint",
        "path": "hcli/dag_store.py",
        "needle": "class DagStore:",
        "symbol": "DagStore (HCLI crash checkpoint)",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": (
            "HCLI has no Checkpoint class. Durable restart is DagStore + "
            "mission state.json + MutationLock record. Crash behaviour is "
            "proven by tools/headless/hcli_crash_checkpoint_test.py."
        ),
        "move": (
            "Do not grow a third Checkpoint type in HCLI-py. If a name is "
            "needed, it is this pair, not hide-backend CheckpointStore."
        ),
        "callers": [
            "hcli/scheduler.py",
            "hcli/mission.py",
        ],
        "evidence": "receipts/headless/HCLI_CRASH_CHECKPOINT.json",
    },
    {
        "id": "hide-kernel.AgentCheckpoint",
        "concept": "checkpoint",
        "path": "crates/hide-kernel/src/checkpoint.rs",
        "needle": "pub struct AgentCheckpoint {",
        "symbol": "AgentCheckpoint",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": True,
        "survives": True,
        "role": "HIDE event-log fold snapshot/restore/resume/fork (bible ch.02 §4.13).",
        "move": "Keep as HIDE's agent-state checkpoint. Not HCLI dag.json.",
        "callers": ["crates/hide-kernel/src/checkpoint.rs"],
    },
    {
        "id": "hide-backend.CheckpointStore",
        "concept": "checkpoint",
        "path": "crates/hide-backend/src/services_goal.rs",
        "needle": "pub struct CheckpointStore;",
        "symbol": "CheckpointStore / CheckpointRecord",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": True,
        "survives": True,
        "role": "Durable HIDE session checkpoint over KV `checkpoints` namespace.",
        "move": "Keep. Protocol schema is hide-protocol::Checkpoint; this is the store.",
        "callers": ["crates/hide-backend/src/services_goal.rs"],
    },
    {
        "id": "hide-protocol.Checkpoint",
        "concept": "checkpoint",
        "path": "crates/hide-protocol/src/model.rs",
        "needle": "pub struct Checkpoint {",
        "symbol": "Checkpoint",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": False,
        "survives": True,
        "role": "HIDE protocol schema for a named restorable boundary (capsule + vcs_ref).",
        "move": "Keep as schema. Execution is AgentCheckpoint / CheckpointStore.",
        "callers": ["crates/hide-protocol/src/model.rs"],
    },
    {
        "id": "lab.checkpoint.CheckpointStore",
        "concept": "checkpoint",
        "path": "research/lab/checkpoint.py",
        "needle": "class CheckpointStore:",
        "symbol": "CheckpointStore",
        "classification": "canonical_authority",
        "plane": "lab",
        "two_real": True,
        "survives": True,
        "role": "Lab campaign controller checkpoint + hash-chain event log.",
        "move": "Keep in lab. Same class name as hide-backend and frankenstein; different schema.",
        "callers": ["research/lab/runtime.py"],
    },
    {
        "id": "frankenstein.CheckpointStore",
        "concept": "checkpoint",
        "path": "research/hawking-experiments/frankenstein/operators/frankenstein_latent_v0.py",
        "needle": "class CheckpointStore:",
        "symbol": "CheckpointStore (.pt slots)",
        "classification": "canonical_authority",
        "plane": "lab",
        "two_real": False,
        "survives": True,
        "role": "Latent-v0 weight-slot store (slot.pt). Not a controller checkpoint.",
        "move": "Keep local to the operator. Rename would clarify; not required for HCLI.",
        "callers": ["research/hawking-experiments/frankenstein/operators/frankenstein_latent_v0.py"],
    },
    {
        "id": "hawking-context.CheckpointId",
        "concept": "checkpoint",
        "path": "crates/hawking-context/src/kv.rs",
        "needle": "pub struct CheckpointId",
        "symbol": "CheckpointId / CheckpointMeta / KvCheckpoint",
        "classification": "canonical_authority",
        "plane": "hawking-context",
        "two_real": False,
        "survives": True,
        "role": "KV-cache checkpoint (prefix reuse), not a mission/controller checkpoint.",
        "move": "Keep. Do not treat as HCLI crash checkpoint.",
        "callers": ["crates/hawking-context/src/lib.rs"],
    },
    {
        "id": "hide-fleet.CheckpointId",
        "concept": "checkpoint",
        "path": "crates/hide-fleet/src/fabric_failure.rs",
        "needle": "pub struct CheckpointId(pub String);",
        "symbol": "CheckpointId (fleet fabric_failure)",
        "classification": "historical_implementation",
        "plane": "hide-rs",
        "two_real": False,
        "survives": True,
        "role": "Newtype id in fabric_failure. Name collision with hawking-context::CheckpointId.",
        "move": "Keep. Do not treat as HCLI or HIDE session checkpoint.",
        "callers": ["crates/hide-fleet/src/fabric_failure.rs"],
    },
    {
        "id": "tools.worker_checkpoint",
        "concept": "checkpoint",
        "path": "tools/worker_checkpoint.py",
        "needle": "REQUIRED = [\"hypothesis\"",
        "symbol": "worker_checkpoint (G132 rebind)",
        "classification": "canonical_authority",
        "plane": "headless",
        "two_real": False,
        "survives": True,
        "role": "Gravity-worker rebind protocol. Measurements do not survive a parent change.",
        "move": "Keep. Not HCLI dag.json and not hide-kernel AgentCheckpoint.",
        "callers": ["tools/worker_checkpoint.py"],
    },
    {
        "id": "tools.llama_checkpoint_bisect",
        "concept": "checkpoint",
        "path": "tools/llama_checkpoint_bisect.py",
        "needle": "",
        "symbol": "llama_checkpoint_bisect (GGUF)",
        "classification": "historical_implementation",
        "plane": "headless",
        "two_real": False,
        "survives": True,
        "role": "GGUF/weight checkpoint bisect. English collision only.",
        "move": "Keep. Do not cite as a control-plane checkpoint.",
        "callers": [],
    },
    # ------------------------------------------------------------------ mutation lock
    {
        "id": "hcli.resources.MutationLock",
        "concept": "mutation lock",
        "path": "hcli/resources.py",
        "needle": "class MutationLock:",
        "symbol": "MutationLock",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": (
            "Crash-safe exclusive lock for MUTATION WorkUnits. Replaced the "
            "advisory JSON check-then-replace that two processes could both win."
        ),
        "move": "Keep. Scheduler.assign_ready and GrokBridge.delegate must keep using it.",
        "callers": [
            "hcli/scheduler.py",
            "hcli/workunit.py",
            "hcli/commands.py",
            "hcli/grok_bridge.py",
        ],
        "evidence": "receipts/headless/AGENTOS_SINGLE_WRITER.json",
    },
    {
        "id": "hcli.grok_bridge.mutation_lock_param",
        "concept": "mutation lock",
        "path": "hcli/grok_bridge.py",
        "needle": "def _as_mutation_lock(",
        "symbol": "_as_mutation_lock / delegate(mutation_lock=)",
        "classification": "compatibility_wrapper",
        "plane": "hcli-py",
        "two_real": False,
        "survives": True,
        "role": "Accepts a callable or context manager wrapping MutationLock for Grok delegate.",
        "move": "Keep while CommandHandler._grok_mutation_lock and Mission pass it.",
        "callers": [
            "hcli/commands.py",
            "hcli/grok_bridge.py",
        ],
    },
    {
        "id": "lab.lease.SingletonLease",
        "concept": "mutation lock",
        "path": "research/lab/lease.py",
        "needle": "class SingletonLease:",
        "symbol": "SingletonLease",
        "classification": "canonical_authority",
        "plane": "lab",
        "two_real": True,
        "survives": True,
        "role": "fcntl exclusive lease for the lab campaign controller.",
        "move": "Keep in lab. Do not replace MutationLock with this, or this with MutationLock.",
        "callers": ["research/lab/runtime.py"],
    },
    {
        "id": "lab.glm52.SingletonLease",
        "concept": "mutation lock",
        "path": "research/lab/operators/glm52_state.py",
        "needle": "class SingletonLease(_EngineSingletonLease):",
        "symbol": "glm52_state.SingletonLease",
        "classification": "compatibility_wrapper",
        "plane": "lab",
        "two_real": False,
        "survives": True,
        "role": "Subclass of lab.lease.SingletonLease for the GLM-52 operator.",
        "move": "Keep as a named subclass. Canonical lock remains lab.lease.SingletonLease.",
        "callers": ["research/lab/operators/glm52_state.py"],
    },
    {
        "id": "tools.gpu_lane_guard",
        "concept": "mutation lock",
        "path": "tools/gpu_lane_guard.py",
        "needle": "def guard(",
        "symbol": "gpu_lane_guard / gpu_lane_lock.sh",
        "classification": "canonical_authority",
        "plane": "headless",
        "two_real": False,
        "survives": True,
        "role": (
            "GPU timing-lane witness. Marks contended timings VOID rather than "
            "claiming exclusive ownership of the repo."
        ),
        "move": "Keep. Different job from MutationLock.",
        "callers": ["tools/gpu_lane_guard.py"],
    },
    # ------------------------------------------------------------------ verifier
    {
        "id": "hcli.verifier_pipeline",
        "concept": "verifier",
        "path": "hcli/verifier_pipeline.py",
        "needle": "def verify(",
        "symbol": "verifier_pipeline.verify / plan / execute",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": (
            "Verifier-first pipeline (plan → evidence → mechanical command). "
            "Defines its own Obligation dataclass (id, statement, angles) — not "
            "ledger.Obligation."
        ),
        "move": (
            "Keep as the HCLI pipeline. Rename its Obligation to PipelineClaim "
            "on a later source lane so the two types cannot be confused. "
            "command_is_admissible stays the vacuity gate."
        ),
        "callers": [
            "hcli/mission.py",
            "hcli/scheduler.py",
        ],
        "evidence": "receipts/headless/AGENTOS_VERIFIER_AUTHORITY.json",
    },
    {
        "id": "hcli.ledger.Obligation",
        "concept": "verifier",
        "path": "hcli/ledger.py",
        "needle": "class Obligation:",
        "symbol": "ledger.Obligation + Ledger.run_verify",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": (
            "Durable GOAL.md obligation with verify_command. VerifyResult is "
            "fresh only when produced by run_verify (anti-forgery)."
        ),
        "move": "Keep. Different type from verifier_pipeline.Obligation. TWO real authorities in one package.",
        "callers": ["hcli/mission.py"],
        "evidence": "receipts/headless/AGENTOS_VERIFIER_AUTHORITY.json",
    },
    {
        "id": "hcli.scheduler.verification_passed",
        "concept": "verifier",
        "path": "hcli/scheduler.py",
        "needle": "def verification_passed(",
        "symbol": "verification_passed / UnverifiedCompletion",
        "classification": "compatibility_wrapper",
        "plane": "hcli-py",
        "two_real": False,
        "survives": True,
        "role": "Thin predicate: outcome is a dict with ok is True. complete() raises otherwise.",
        "move": "Keep. Verified caller: Scheduler.complete.",
        "callers": ["hcli/scheduler.py"],
    },
    {
        "id": "lab.VerificationAuthority",
        "concept": "verifier",
        "path": "research/lab/verification_authority.py",
        "needle": "class VerificationAuthority:",
        "symbol": "VerificationAuthority",
        "classification": "canonical_authority",
        "plane": "lab",
        "two_real": True,
        "survives": True,
        "role": "Lab science certification: models emit candidates, only the controller certifies.",
        "move": "Keep. Not the HCLI WorkUnit verifier.",
        "callers": ["research/lab/verification_authority.py"],
    },
    {
        "id": "hide-kernel.VerificationGate",
        "concept": "verifier",
        "path": "crates/hide-kernel/src/verify.rs",
        "needle": "pub struct VerificationGate {",
        "symbol": "VerificationGate / OracleSuite",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": True,
        "survives": True,
        "role": "HIDE oracle gate. AgentDriver will not advance without it.",
        "move": "Keep as HIDE verify. Do not import into HCLI-py.",
        "callers": ["crates/hide-kernel/src/machine.rs"],
    },
    {
        "id": "ramanujan.VerifierRegistry",
        "concept": "verifier",
        "path": "research/ramanujan/scaffold/research/verifier/registry.py",
        "needle": "class VerifierRegistry:",
        "symbol": "VerifierRegistry",
        "classification": "canonical_authority",
        "plane": "ramanujan",
        "two_real": False,
        "survives": True,
        "role": "Math verifier backends (lean / sympy / exact numeric).",
        "move": "Keep in ramanujan. Not an HCLI WorkUnit verifier.",
        "callers": ["research/ramanujan/scaffold/research/verifier/registry.py"],
    },
    {
        "id": "hawking-speculate.Verifier",
        "concept": "verifier",
        "path": "crates/hawking-speculate/src/verifier.rs",
        "needle": "pub struct Verifier {",
        "symbol": "Verifier (speculative decode)",
        "classification": "historical_implementation",
        "plane": "hawking-speculate",
        "two_real": False,
        "survives": True,
        "role": (
            "Draft-token accept rule for speculative decode. English collision "
            "with WorkUnit verification; not a control-plane verifier."
        ),
        "move": "Keep. Do not cite as HCLI/HIDE goal verification.",
        "callers": ["crates/hawking-speculate/src/verifier.rs"],
    },
    # ------------------------------------------------------------------ backend registry
    {
        "id": "hcli.backends.RuntimeBackend",
        "concept": "backend registry",
        "path": "hcli/backends.py",
        "needle": "class RuntimeBackend(",
        "symbol": "RuntimeBackend + LlamaServerBackend + MlxServerBackend",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": (
            "HCLI inference-server registry. supports() is the capability table; "
            "MLX reports response_format/grammar false so the engine degrades "
            "instead of silently sending ignored fields."
        ),
        "move": "Keep. New backends implement RuntimeBackend; do not add a second registry dict.",
        "callers": [
            "hcli/runtime.py",
            "hcli/engine.py",
        ],
        "evidence": "receipts/headless/BACKEND_CAPABILITY.json",
    },
    {
        "id": "hcli.resources.BackendHealth",
        "concept": "backend registry",
        "path": "hcli/resources.py",
        "needle": "class BackendHealth:",
        "symbol": "BackendHealth",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": False,
        "survives": True,
        "role": "Durable per-backend circuit breaker. Not a backend registry.",
        "move": "Keep beside RuntimeBackend. Do not let it grow spawn/complete methods.",
        "callers": ["hcli/resources.py"],
    },
    {
        "id": "hawking-orch.RoleRegistry",
        "concept": "backend registry",
        "path": "crates/hawking-orch/src/registry.rs",
        "needle": "pub struct RoleRegistry {",
        "symbol": "RoleRegistry",
        "classification": "canonical_authority",
        "plane": "hawking-orch",
        "two_real": True,
        "survives": True,
        "role": "HIDE/Hawking model-role registry (roles.toml). Not llama vs MLX servers.",
        "move": "Keep. Different object from RuntimeBackend.",
        "callers": ["crates/hawking-orch/src/registry.rs"],
    },
    {
        "id": "haider-v0.allocate_port",
        "concept": "backend registry",
        "path": "tools/hcli/bootstrap/snapshots/haider.py",
        "needle": "def allocate_port() -> int:",
        "symbol": "allocate_port (HCLI-v0)",
        "classification": "obsolete_implementation",
        "plane": "haider-v0",
        "two_real": False,
        "survives": False,
        "role": "Duplicate of backends.allocate_port in the v0 bootstrap.",
        "move": "Leave the v0 file. Live callers use hcli/backends.py:allocate_port.",
        "callers": [],
    },
    # ------------------------------------------------------------------ context budget
    {
        "id": "hcli.context_budget.ContextBudget",
        "concept": "context budget",
        "path": "hcli/context_budget.py",
        "needle": "class ContextBudget:",
        "symbol": "ContextBudget / resolve / preflight / per_seq_context",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": (
            "HCLI per-request context authority. Divides allocation by slot "
            "count; preflight refuses an over-large root prompt before inference."
        ),
        "move": (
            "Keep. Remaining gap (HCLI_CONTEXT_AUTHORITY): resolve() without "
            "demand_tokens still yields 32768/3=11008; root ingress must pass "
            "demand (G006/G007). Do not reintroduce a 32768 literal in config.py."
        ),
        "callers": [
            "hcli/runtime.py",
            "hcli/backends.py",
            "hcli/config.py",
            "hcli/engine.py",
            "hcli/goal.py",
        ],
        "evidence": "receipts/headless/HCLI_CONTEXT_AUTHORITY.json",
    },
    {
        "id": "hcli.context.reexport",
        "concept": "context budget",
        "path": "hcli/context.py",
        "needle": "from .goal import WorkerPacket, compile_worker_context",
        "symbol": "context.py (re-export)",
        "classification": "compatibility_wrapper",
        "plane": "hcli-py",
        "two_real": False,
        "survives": False,
        "role": (
            "Re-exports WorkerPacket from goal.py so there is not a second "
            "packet shape. No production importer of hcli.context was found."
        ),
        "move": "Removable: no named verified caller of hcli.context (only context_budget).",
        "callers": [],
    },
    {
        "id": "hawking-context.TokenBudget",
        "concept": "context budget",
        "path": "crates/hawking-context/src/budget.rs",
        "needle": "pub struct TokenBudget {",
        "symbol": "TokenBudget / RegionBudget",
        "classification": "canonical_authority",
        "plane": "hawking-context",
        "two_real": True,
        "survives": True,
        "role": "HIDE shell-side context compiler (reservation-aware knapsack).",
        "move": "Keep as HIDE's packer. Do not replace HCLI context_budget.py with this.",
        "callers": [
            "crates/hawking-context/src/lib.rs",
            "crates/hawking-context/src/compiler.rs",
        ],
    },
    {
        "id": "hide-backend.haider.ContextBudget",
        "concept": "context budget",
        "path": "crates/hide-backend/src/haider/lanes.rs",
        "needle": "pub struct ContextBudget {",
        "symbol": "haider::lanes::ContextBudget",
        "classification": "compatibility_wrapper",
        "plane": "hide-rs",
        "two_real": False,
        "survives": True,
        "role": "Per-lane max_tokens newtype used by EvidencePacket. Not the HCLI authority.",
        "move": (
            "Keep as a field on EvidencePacket. Verified caller: "
            "crates/hide-backend/src/haider/lanes.rs EvidencePacket. "
            "Do not give it resolve/preflight logic."
        ),
        "callers": ["crates/hide-backend/src/haider/lanes.rs"],
    },
    {
        "id": "hawking-context.ContextCompiler",
        "concept": "context budget",
        "path": "crates/hawking-context/src/compiler.rs",
        "needle": "pub struct ContextCompiler {",
        "symbol": "ContextCompiler",
        "classification": "canonical_authority",
        "plane": "hawking-context",
        "two_real": False,
        "survives": True,
        "role": "Packs TokenBudget into a replayable manifest. Same plane as TokenBudget, not a second budget.",
        "move": "Keep with TokenBudget. Do not merge with HCLI context_budget.py.",
        "callers": ["crates/hawking-context/src/lib.rs"],
    },
    {
        "id": "hide-backend.hcli_bridge.MAX_CONTEXT_TOKENS",
        "concept": "context budget",
        "path": "crates/hide-backend/src/hcli_bridge.rs",
        "needle": "pub const MAX_CONTEXT_TOKENS",
        "symbol": "MAX_CONTEXT_TOKENS (transport bound)",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": False,
        "survives": True,
        "role": "JSONL transport cap (2e6), independent of the model window.",
        "move": "Keep as a transport bound. Not a substitute for context_budget.resolve.",
        "callers": ["crates/hide-backend/src/hcli_bridge.rs"],
    },
    # ------------------------------------------------------------------ runtime registry
    {
        "id": "hcli.runtime.RuntimePool",
        "concept": "runtime registry",
        "path": "hcli/runtime.py",
        "needle": "class RuntimePool:",
        "symbol": "RuntimePool / Runtime",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": (
            "HCLI owned llama/MLX process pool. Limits come from "
            "machine.resolve_runtime_limits (genome is a prior, not a constant)."
        ),
        "move": "Keep. Do not hand-edit MACHINE_GENOME.json to change slot counts.",
        "callers": [
            "hcli/controller.py",
            "hcli/mission.py",
        ],
        "evidence": "receipts/headless/RUNTIME_AUTHORITY.json",
    },
    {
        "id": "lab.runtime.ExperimentRuntime",
        "concept": "runtime registry",
        "path": "research/lab/runtime.py",
        "needle": "class ExperimentRuntime",
        "symbol": "ExperimentRuntime",
        "classification": "canonical_authority",
        "plane": "lab",
        "two_real": True,
        "survives": True,
        "role": "Lab experiment runner. Name collision with HCLI RuntimePool.",
        "move": "Keep in lab. Do not spawn llama-server from here.",
        "callers": ["research/lab/runtime.py"],
    },
    {
        "id": "tools.glm52_gravity.Runtime",
        "concept": "runtime registry",
        "path": "tools/glm52_gravity.py",
        "needle": "class Runtime:",
        "symbol": "Runtime (glm52_gravity local)",
        "classification": "obsolete_implementation",
        "plane": "headless",
        "two_real": False,
        "survives": False,
        "role": "Local helper in a gravity tool. Not RuntimePool.",
        "move": "Leave in the tool. Do not import from hcli.runtime.",
        "callers": [],
    },
    {
        "id": "headless.machine_probe.Runtime",
        "concept": "runtime registry",
        "path": "tools/headless/machine_probe.py",
        "needle": "class Runtime:",
        "symbol": "Runtime (probe-local)",
        "classification": "test_only_implementation",
        "plane": "headless",
        "two_real": False,
        "survives": True,
        "role": "Local helper inside the genome probe. Not RuntimePool.",
        "move": "Keep inside the probe. Do not import from hcli.runtime.",
        "callers": ["tools/headless/machine_probe.py"],
    },
    # ------------------------------------------------------------------ receipt
    {
        "id": "lab.receipts.Receipt",
        "concept": "receipt",
        "path": "research/lab/receipts.py",
        "needle": "class Receipt:",
        "symbol": "Receipt / ReceiptAuthority / GateEvidence / seal",
        "classification": "canonical_authority",
        "plane": "lab",
        "two_real": True,
        "survives": True,
        "role": "Lab sealed receipt family (hawking.lab.receipt.v1).",
        "move": (
            "Keep as the lab seal. Do not rewrite historical receipts. Do not "
            "force HCLI engine receipts onto this schema."
        ),
        "callers": [
            "research/lab/runtime.py",
            "research/lab/verification_authority.py",
            "research/lab/checkpoint.py",
        ],
    },
    {
        "id": "hcli.engine._write_receipt",
        "concept": "receipt",
        "path": "hcli/engine.py",
        "needle": "def _write_receipt(",
        "symbol": "Engine._write_receipt",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": "Per-goal mutation receipt under the workspace.",
        "move": (
            "Keep the schema and filenames (historical receipts cite them). "
            "Share the atomic-write primitive with grok_bridge and ledger on a "
            "later source lane; do not merge the three payload shapes."
        ),
        "callers": ["hcli/engine.py"],
    },
    {
        "id": "hcli.grok_bridge._write_receipt",
        "concept": "receipt",
        "path": "hcli/grok_bridge.py",
        "needle": "def _write_receipt(",
        "symbol": "GrokBridge._write_receipt",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": "Per-Grok-task receipt. Different payload from Engine.",
        "move": "Keep. Share atomic-write only.",
        "callers": ["hcli/grok_bridge.py"],
    },
    {
        "id": "hcli.ledger._write_receipt",
        "concept": "receipt",
        "path": "hcli/ledger.py",
        "needle": "def _write_receipt(self, ob: Obligation, evidence: VerifyResult) -> None:",
        "symbol": "Ledger._write_receipt",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": "Per-obligation verify receipt. Forgery of VERIFIED without this is demoted to STALE.",
        "move": "Keep. Required by AGENTOS_VERIFIER_AUTHORITY.",
        "callers": ["hcli/ledger.py"],
        "evidence": "receipts/headless/AGENTOS_VERIFIER_AUTHORITY.json",
    },
    {
        "id": "hide-backend.write_sealed_receipt",
        "concept": "receipt",
        "path": "crates/hide-backend/src/headless.rs",
        "needle": "pub fn write_sealed_receipt(",
        "symbol": "write_sealed_receipt",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": True,
        "survives": True,
        "role": "HIDE headless sealed receipt writer.",
        "move": "Keep for hide-headless / rust hcli. Not HCLI-py engine receipts.",
        "callers": [
            "crates/hide-backend/src/bin/hide-headless.rs",
            "crates/hide-backend/src/bin/hcli.rs",
        ],
    },
    {
        "id": "haider-v0.write_receipt",
        "concept": "receipt",
        "path": "tools/hcli/bootstrap/snapshots/haider.py",
        "needle": "def write_receipt(",
        "symbol": "haider.py write_receipt",
        "classification": "obsolete_implementation",
        "plane": "haider-v0",
        "two_real": False,
        "survives": False,
        "role": "HCLI-v0 receipt writer.",
        "move": "Leave. Historical .hcli-legacy/receipts filenames are evidence.",
        "callers": [],
    },
    # ------------------------------------------------------------------ experiment
    {
        "id": "lab.spec.ExperimentSpec",
        "concept": "experiment",
        "path": "research/lab/spec.py",
        "needle": "class ExperimentSpec:",
        "symbol": "ExperimentSpec (hawking.lab.experiment_spec.v1)",
        "classification": "canonical_authority",
        "plane": "lab",
        "two_real": True,
        "survives": True,
        "role": "Lab campaign spec. lab.runtime and lab.engine_support import this one.",
        "move": "Keep as the campaign spec. Accepts hawking.lab.experiment.v1 as a compatibility schema id.",
        "callers": [
            "research/lab/runtime.py",
            "research/lab/engine_support.py",
            "research/lab/__init__.py",
        ],
    },
    {
        "id": "lab.bench_harness.ExperimentSpec",
        "concept": "experiment",
        "path": "research/lab/bench_harness/spec.py",
        "needle": "class ExperimentSpec:",
        "symbol": "ExperimentSpec (hawking.lab.experiment.v1 stages)",
        "classification": "canonical_authority",
        "plane": "lab",
        "two_real": True,
        "survives": True,
        "role": (
            "Second ExperimentSpec: id + stages runner schema. Imported by "
            "tools/foundry/tests, not by lab.runtime."
        ),
        "move": (
            "TWO real authorities. Rename this class to HarnessSpec on a later "
            "source lane; keep the sealed schema id hawking.lab.experiment.v1."
        ),
        "callers": ["tools/foundry/tests/test_foundry_tables_lifecycle.py"],
    },
    {
        "id": "lab.science_registry",
        "concept": "experiment",
        "path": "research/lab/science_registry.py",
        "needle": "class OperatorRecord:",
        "symbol": "OperatorRegistry / OperatorRecord",
        "classification": "canonical_authority",
        "plane": "lab",
        "two_real": False,
        "survives": True,
        "role": "Catalog of lab.operators. Not an ExperimentSpec.",
        "move": "Keep. Runtime looks up operators here.",
        "callers": ["research/lab/runtime.py"],
    },
    {
        "id": "headless.runtime_experiment",
        "concept": "experiment",
        "path": "tools/headless/runtime_experiment.py",
        "needle": "One measured local-runtime experiment",
        "symbol": "runtime_experiment.py (harness)",
        "classification": "test_only_implementation",
        "plane": "headless",
        "two_real": False,
        "survives": True,
        "role": "Measurement harness writing RUNTIME_EXPERIMENT.json. No Experiment type.",
        "move": "Keep as a harness. Do not promote into lab.spec.",
        "callers": [],
    },
    # ------------------------------------------------------------------ model identity
    {
        "id": "hcli.models.ModelRegistry",
        "concept": "model identity",
        "path": "hcli/models.py",
        "needle": "class ModelRegistry:",
        "symbol": "ModelRegistry / ModelInfo / resolve_model",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": (
            "Local GGUF inventory and selection. Discovery never loads. Ambiguous "
            "discovery stays ambiguous."
        ),
        "move": "Keep as HCLI selection. Do not merge with the promotion registry.",
        "callers": [
            "hcli/cli.py",
            "tools/headless/hcli_command_driver.py",
        ],
    },
    {
        "id": "headless.model_registry",
        "concept": "model identity",
        "path": "tools/headless/model_registry.py",
        "needle": "Model Registry — the parent/child lineage",
        "symbol": "model_registry.py + receipts/headless/MODEL_REGISTRY.json",
        "classification": "canonical_authority",
        "plane": "headless",
        "two_real": True,
        "survives": True,
        "role": (
            "Parent/child promotion lineage. promote() reads PerformanceLedger + "
            "capability receipt + incumbent; a child cannot promote itself."
        ),
        "move": (
            "Keep as the promotion authority. Different job from ModelRegistry "
            "path selection. TWO real authorities under the English name 'model identity'."
        ),
        "callers": ["tools/headless/model_registry.py"],
        "evidence": "receipts/headless/MODEL_REGISTRY.json",
    },
    {
        "id": "hawking-comms.ModelIdentity",
        "concept": "model identity",
        "path": "crates/hawking-comms/src/level3.rs",
        "needle": "pub struct ModelIdentity {",
        "symbol": "ModelIdentity",
        "classification": "canonical_authority",
        "plane": "hawking-orch",
        "two_real": True,
        "survives": True,
        "role": "Wire-level model identity in hawking-comms. Not HCLI path selection and not the promotion registry.",
        "move": "Keep as the comms identity type. Do not merge with ModelRegistry or MODEL_REGISTRY.json.",
        "callers": ["crates/hawking-comms/src/level3.rs"],
    },
    {
        "id": "hawking-orch.ModelSpec",
        "concept": "model identity",
        "path": "crates/hawking-orch/src/registry.rs",
        "needle": "struct ModelSpec {",
        "symbol": "ModelSpec (roles.toml parse)",
        "classification": "compatibility_wrapper",
        "plane": "hawking-orch",
        "two_real": False,
        "survives": True,
        "role": "TOML parse shape for RoleRegistry. Not lab.operators.ModelSpec.",
        "move": "Keep private to registry.rs. Verified caller: RoleRegistry::from_roles_toml.",
        "callers": ["crates/hawking-orch/src/registry.rs"],
    },
    {
        "id": "hawking-serve.ModelInfo",
        "concept": "model identity",
        "path": "crates/hawking-serve/src/http.rs",
        "needle": "struct ModelInfo {",
        "symbol": "ModelInfo (HTTP)",
        "classification": "compatibility_wrapper",
        "plane": "hawking-serve",
        "two_real": False,
        "survives": True,
        "role": "HTTP-layer model info. Name collision with hcli.models.ModelInfo.",
        "move": "Keep in hawking-serve. Do not share a type with HCLI-py ModelInfo.",
        "callers": ["crates/hawking-serve/src/http.rs"],
    },
    {
        "id": "lab.operators.ModelSpec.copies",
        "concept": "model identity",
        "path": "research/lab/operators/ascension_qwen_state_kv.py",
        "needle": "class ModelSpec:",
        "symbol": "ModelSpec (copied across operators)",
        "classification": "UNKNOWN",
        "plane": "lab",
        "two_real": False,
        "survives": True,
        "role": (
            "class ModelSpec appears in at least four lab operators. Whether they "
            "are copies of one type or independent locals was not fully compared."
        ),
        "move": "Follow-up: diff the four ModelSpec bodies before deleting any.",
        "callers": [],
        "unknown_reason": (
            "Four research/lab/operators/* ModelSpec classes exist "
            "(ascension_dual_gravity_worker, ascension_physical_gatekeeper, "
            "ascension_qwen_scientific_optimizer, ascension_qwen_state_kv). "
            "This lane did not byte-compare them."
        ),
    },
    # ------------------------------------------------------------------ machine identity
    {
        "id": "headless.machine_probe",
        "concept": "machine identity",
        "path": "tools/headless/machine_probe.py",
        "needle": '"schema": "hawking.headless.machine_genome.v1"',
        "symbol": "machine_probe.py (producer)",
        "classification": "canonical_authority",
        "plane": "headless",
        "two_real": True,
        "survives": True,
        "role": (
            "Produces receipts/headless/MACHINE_GENOME.json. Promotion of a new "
            "knee is a re-run of this probe, not an editor."
        ),
        "move": "Keep as the sole producer of the live genome.",
        "callers": ["tools/headless/machine_probe.py"],
        "evidence": "receipts/headless/RUNTIME_AUTHORITY.json",
    },
    {
        "id": "hcli.machine.live_machine_identity",
        "concept": "machine identity",
        "path": "hcli/machine.py",
        "needle": "def live_machine_identity()",
        "symbol": "live_machine_identity / assess_genome_freshness / resolve_runtime_limits / MachineProbe / MemGate",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": False,
        "survives": True,
        "role": (
            "Consumer of the genome. Reads (1) env (2) ~/.config/hcli/machine_genome.json "
            "(3) receipts/headless/MACHINE_GENOME.json (4) worker-equilibrium.json. "
            "STALE genomes (wrong machine, wrong model, too old) are not used."
        ),
        "move": "Keep as the consumer. Do not let it grow a write() that bypasses the probe.",
        "callers": [
            "hcli/runtime.py",
            "hcli/resources.py",
        ],
        "evidence": "receipts/headless/RUNTIME_AUTHORITY.json",
    },
    {
        "id": "hcli.machine.MachineGenome",
        "concept": "machine identity",
        "path": "hcli/machine.py",
        "needle": "class MachineGenome:",
        "symbol": "MachineGenome (HCLI_HOME/machine-genome.json)",
        "classification": "obsolete_implementation",
        "plane": "hcli-py",
        "two_real": True,
        "survives": False,
        "role": (
            "Thin JSON blob at ~/.local/share/hcli/machine-genome.json via "
            "get_machine_genome_path(). Production limit resolution does not "
            "read this class. Only hcli_persistence_audit.py constructs it."
        ),
        "move": (
            "Remove or wrap so it reads the same files as resolve_runtime_limits. "
            "Leaving it is a second genome authority."
        ),
        "callers": ["tools/headless/hcli_persistence_audit.py"],
    },
    {
        "id": "agentos.machine_state",
        "concept": "machine identity",
        "path": "tools/agentos/machine_state.py",
        "needle": "MachineState snapshot",
        "symbol": "machine_state.py (ancestor of Machine Genome)",
        "classification": "canonical_authority",
        "plane": "agentos",
        "two_real": False,
        "survives": True,
        "role": "Load/RAM/disk/heavy-lane snapshot for genesis. Not the HCLI genome.",
        "move": "Keep for agentos. Do not feed it to resolve_runtime_limits.",
        "callers": ["tools/agentos/machine_state.py"],
    },
    {
        "id": "hide-kernel.machine.rs",
        "concept": "machine identity",
        "path": "crates/hide-kernel/src/machine.rs",
        "needle": "pub struct AgentDriver",
        "symbol": "machine.rs (AgentDriver — name collision)",
        "classification": "historical_implementation",
        "plane": "hide-rs",
        "two_real": False,
        "survives": True,
        "role": "File name says machine; contents are the HIDE agent FSM. Not hardware identity.",
        "move": "Keep. Do not cite as machine identity.",
        "callers": ["crates/hide-kernel/src/machine.rs"],
    },
    # ------------------------------------------------------------------ status
    {
        "id": "hcli.workunit.transition_status",
        "concept": "status",
        "path": "hcli/workunit.py",
        "needle": "def transition_status(",
        "symbol": "transition_status / WORKUNIT_STATUSES",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": "WorkUnit state machine (pending/ready/running/completed/failed/interrupted).",
        "move": "Keep. Direct wu.status writes are forbidden (DAG consolidation).",
        "callers": [
            "hcli/scheduler.py",
            "hcli/goal.py",
        ],
    },
    {
        "id": "hcli.ledger.outcome",
        "concept": "status",
        "path": "hcli/ledger.py",
        "needle": "class Ledger:",
        "symbol": "Ledger.outcome / Ledger.status",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": (
            "Mission completion: GOAL_MET / GOAL_NOT_MET / TERMINAL_BLOCKER / "
            "SAFETY_DISARM / ENFORCEMENT_FAILURE. .status remains the three-way "
            "compat property."
        ),
        "move": "Keep. Compatibility: .status still returns EMPTY_LEDGER/GOAL_MET/GOAL_NOT_MET.",
        "callers": [
            "hcli/commands.py",
            "tools/headless/hcli_agentos_ledger_test.py",
        ],
        "evidence": "receipts/headless/DAG_CONSOLIDATION_DECISION.json",
    },
    {
        "id": "hcli.commands.format_status",
        "concept": "status",
        "path": "hcli/commands.py",
        "needle": "def format_status(",
        "symbol": "format_status / enrich_status_snapshot",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": (
            "One-screen /status renderer. Unmeasured fields print as unknown, "
            "never as 0. HEAD has exactly one def format_status; the 'later "
            "shadows earlier' finding did not reproduce."
        ),
        "move": "Keep as the only human /status renderer. TUI.render_status is a transcript line, not this.",
        "callers": [
            "hcli/commands.py",
            "hcli/tests/test_status_completeness.py",
            "hcli/tests/test_status_truth.py",
        ],
        "evidence": "receipts/headless/HCLI_STATUS_OBSERVABILITY.json",
    },
    {
        "id": "hcli.controller.status",
        "concept": "status",
        "path": "hcli/controller.py",
        "needle": "def status(",
        "symbol": "Controller.status (snapshot)",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": False,
        "survives": True,
        "role": "Builds the snapshot format_status renders (qwen/grok/mutation/occupancy).",
        "move": "Keep as the data side of /status. format_status must not grow its own probes.",
        "callers": ["hcli/commands.py"],
    },
    {
        "id": "hcli.tui.render_status",
        "concept": "status",
        "path": "hcli/tui.py",
        "needle": "def render_status(",
        "symbol": "TUI.render_status",
        "classification": "obsolete_implementation",
        "plane": "hcli-py",
        "two_real": False,
        "survives": True,
        "role": "Prints TUI.status string (idle/working). Not /status.",
        "move": "Keep as a transcript chrome, or later call format_status. Do not treat as the operator status surface.",
        "callers": ["hcli/tui.py"],
    },
    {
        "id": "hcli.grok.parse_grok_status",
        "concept": "status",
        "path": "hcli/grok_bridge.py",
        "needle": "def parse_grok_status(",
        "symbol": "parse_grok_status / grok_succeeded",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": False,
        "survives": True,
        "role": "Grok-run task liveness/terminal state. Input to Mission adoption and P0-3.",
        "move": "Keep. Not WorkUnit.status and not Ledger.outcome.",
        "callers": [
            "hcli/grok_bridge.py",
            "hcli/scheduler.py",
        ],
    },
    {
        "id": "hide-backend.StatusResponse",
        "concept": "status",
        "path": "crates/hide-backend/src/hcli_bridge.rs",
        "needle": "pub struct StatusResponse {",
        "symbol": "StatusResponse",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": True,
        "survives": True,
        "role": "Rust HCLI JSONL status. A second product named HCLI.",
        "move": "Keep for crates/hide-backend/src/bin/hcli.rs. Do not make Python format_status emit this schema.",
        "callers": ["crates/hide-backend/src/bin/hcli.rs"],
    },
    {
        "id": "ramanujan.Status",
        "concept": "status",
        "path": "research/ramanujan/scaffold/research/prover.py",
        "needle": "class Status(",
        "symbol": "Status (ramanujan prover)",
        "classification": "canonical_authority",
        "plane": "ramanujan",
        "two_real": False,
        "survives": True,
        "role": "Ramanujan prover enum. Unrelated to HCLI /status.",
        "move": "Keep.",
        "callers": ["research/ramanujan/scaffold/research/prover.py"],
    },
    # ------------------------------------------------------------------ retry policy
    {
        "id": "hcli.workunit.DEFAULT_RETRY_BUDGET",
        "concept": "retry policy",
        "path": "hcli/workunit.py",
        "needle": "DEFAULT_RETRY_BUDGET = 3",
        "symbol": "DEFAULT_RETRY_BUDGET / MAX_REPAIR_DEPTH / MAX_REPAIRS_PER_ROOT",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": True,
        "survives": True,
        "role": "Durable retry/repair caps on the unit and DAG document.",
        "move": "Keep here. Scheduler must re-export, not reassign.",
        "callers": [
            "hcli/workunit.py",
            "hcli/scheduler.py",
        ],
    },
    {
        "id": "hcli.scheduler.MAX_REPAIR_shadow",
        "concept": "retry policy",
        "path": "hcli/scheduler.py",
        "needle": "MAX_REPAIR_DEPTH = 3",
        "symbol": "MAX_REPAIR_DEPTH = 3 (shadows the import)",
        "classification": "obsolete_implementation",
        "plane": "hcli-py",
        "two_real": True,
        "survives": False,
        "role": (
            "Imported from workunit.py at line 20, then reassigned at line 79 "
            "after a duplicated LOG =. Same values today; a drift would split "
            "the repair cap between emit_repair and `from scheduler import`."
        ),
        "move": "Delete the reassignment; keep a re-export comment only.",
        "callers": ["hcli/tests/test_scheduler_quality.py"],
    },
    {
        "id": "hcli.resources.classify_failure",
        "concept": "retry policy",
        "path": "hcli/resources.py",
        "needle": "def classify_failure(",
        "symbol": "classify_failure / counts_toward_retry_budget / NON_RETRYABLE / FailureClassification",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": False,
        "survives": True,
        "role": "Retryability of a failure. Non-retryable names do not burn DEFAULT_RETRY_BUDGET.",
        "move": "Keep as the retryability authority. emit_repair consults this.",
        "callers": [
            "hcli/workunit.py",
            "tools/headless/repair_disposition_table.py",
        ],
    },
    {
        "id": "hide-kernel.RetryPolicy",
        "concept": "retry policy",
        "path": "crates/hide-kernel/src/program_runtime_ast.rs",
        "needle": "pub struct RetryPolicy {",
        "symbol": "RetryPolicy",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": True,
        "survives": True,
        "role": "HIDE program-runtime AST retry policy. Not WorkUnit DEFAULT_RETRY_BUDGET.",
        "move": "Keep in hide-kernel. Do not import into HCLI-py workunit.py.",
        "callers": ["crates/hide-kernel/src/program_runtime_ast.rs"],
    },
    {
        "id": "hcli.backends.structured_output_attempts",
        "concept": "retry policy",
        "path": "hcli/backends.py",
        "needle": "DEFAULT_STRUCTURED_OUTPUT_ATTEMPTS = 3",
        "symbol": "DEFAULT_STRUCTURED_OUTPUT_ATTEMPTS",
        "classification": "canonical_authority",
        "plane": "hcli-py",
        "two_real": False,
        "survives": True,
        "role": "Bounded parse retries for structured output (especially MLX without grammar).",
        "move": "Keep. Different budget from WorkUnit attempts.",
        "callers": ["hcli/backends.py"],
        "evidence": "receipts/headless/BACKEND_CAPABILITY.json",
    },
    # rust HCLI as a second product (spans several concepts; recorded under mission too)
    {
        "id": "hide-backend.bin.hcli",
        "concept": "mission",
        "path": "crates/hide-backend/src/bin/hcli.rs",
        "needle": "HCLI — Hawking's headless local-model product surface",
        "symbol": "bin/hcli.rs (Rust HCLI)",
        "classification": "canonical_authority",
        "plane": "hide-rs",
        "two_real": True,
        "survives": True,
        "role": (
            "A second product named HCLI: JSONL shell over hide-backend. Not "
            "hcli. bin/haider.rs is an empty blob (git e69de29)."
        ),
        "move": (
            "Keep as the HIDE product surface. Do not point ~/.local/bin/hcli "
            "at this binary without an explicit cutover. TWO real HCLI authorities."
        ),
        "callers": ["crates/hide-backend/src/bin/hcli.rs"],
    },
    {
        "id": "hide-backend.bin.haider_empty",
        "concept": "mission",
        "path": "crates/hide-backend/src/bin/haider.rs",
        "needle": "",
        "symbol": "bin/haider.rs (empty)",
        "classification": "historical_implementation",
        "plane": "hide-rs",
        "two_real": False,
        "survives": True,
        "role": "Empty file (git blob e69de29). Fossil namespace, not a runner.",
        "move": "Do not delete in this campaign (historical name). Do not implement into it.",
        "callers": [],
    },
]


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=check,
    )


def git_head() -> str:
    try:
        return git("rev-parse", "HEAD").stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNKNOWN:{exc}"


def path_in_head(path: str) -> bool:
    r = git("cat-file", "-e", f"HEAD:{path}", check=False)
    return r.returncode == 0


def blob(path: str) -> Optional[str]:
    r = git("show", f"HEAD:{path}", check=False)
    if r.returncode != 0:
        return None
    return r.stdout


def find_line(text: str, needle: str) -> Optional[int]:
    if not needle:
        return None
    for i, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return i
    return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_callers(callers: Sequence[str]) -> List[Dict[str, Any]]:
    out = []
    for c in callers:
        out.append({"path": c, "in_head": path_in_head(c)})
    return out


def classify_scan_hit(concept: str, path: str, line: int, snippet: str) -> Optional[str]:
    """Return a catalog id if this scan hit is already catalogued.

    Match is by path + definition token against any catalog row (a type
    catalogued under WorkUnit must not reappear as UNKNOWN under DAG).
    """
    del concept, line  # token/path are the identity
    token_m = re.search(
        r"(?:class |pub struct |struct |def )([A-Za-z_][A-Za-z0-9_]*)",
        snippet,
    )
    token = token_m.group(1) if token_m else ""
    if token.endswith("Error") or token.endswith("Exception"):
        return f"skip-exception:{token}"
    for row in CATALOG:
        if row["path"] != path:
            continue
        needle = (row.get("needle") or "").strip()
        if needle and needle.split("(")[0].split("{")[0].strip() in snippet:
            return row["id"]
        if token and (token == row["id"].split(".")[-1] or token in row["symbol"]):
            return row["id"]
    return None


def scan_head() -> Dict[str, List[Dict[str, Any]]]:
    """Scan HEAD for definition lines.

    git grep -E is POSIX ERE: ``\\b`` is not a word boundary there, so a
    pattern with ``\\b`` matches nothing (watched-fail). ``-P`` is PCRE.
    """
    extras: Dict[str, List[Dict[str, Any]]] = {}
    for concept, pattern in SCAN_PATTERNS.items():
        r = git("grep", "-n", "-P", pattern, "HEAD", "--", "*.py", "*.rs", check=False)
        if r.returncode not in (0, 1) and not r.stdout:
            r = git("grep", "-n", "-E", pattern.replace(r"\b", ""), "HEAD", "--", "*.py", "*.rs", check=False)
        hits = []
        for raw in r.stdout.splitlines():
            # HEAD:path:line:text
            if not raw.startswith("HEAD:"):
                continue
            rest = raw[5:]
            m = re.match(r"^(.+):(\d+):(.*)$", rest)
            if not m:
                continue
            path, line_s, snippet = m.group(1), m.group(2), m.group(3)
            if any(path.startswith(p) for p in SCAN_SKIP_PREFIXES):
                continue
            if "/tests/" in path or path.endswith("_test.py") or "/examples/" in path:
                kind = "test_or_example"
            else:
                kind = "production"
            cat_id = classify_scan_hit(concept, path, int(line_s), snippet)
            rec = {
                "path": path,
                "line": int(line_s),
                "snippet": snippet.strip()[:200],
                "kind": kind,
                "catalog_id": cat_id,
            }
            if cat_id is None:
                hits.append(rec)
            # skip-exception:* is classified-as-noise, not an extra
        extras[concept] = hits
    return extras


def build_implementations() -> Tuple[List[Dict[str, Any]], List[str]]:
    problems: List[str] = []
    impls: List[Dict[str, Any]] = []
    for row in CATALOG:
        path = row["path"]
        exists = path_in_head(path)
        text = blob(path) if exists else None
        line = find_line(text, row["needle"]) if text is not None and row.get("needle") else None
        if not exists:
            problems.append(f"missing HEAD path {path} ({row['id']})")
            classification = "UNKNOWN"
            unknown_reason = f"path not in HEAD: {path}"
        elif row.get("needle") and line is None:
            # empty files (haider.rs) and files identified by path-only
            if text == "":
                line = None
                classification = row["classification"]
                unknown_reason = row.get("unknown_reason")
            else:
                classification = "UNKNOWN"
                unknown_reason = f"needle {row['needle']!r} not in {path}"
                problems.append(unknown_reason)
        else:
            classification = row["classification"]
            unknown_reason = row.get("unknown_reason")
        if classification not in CLASSES:
            problems.append(f"bad class {classification} on {row['id']}")
        caller_info = verify_callers(row.get("callers") or [])
        missing_callers = [c["path"] for c in caller_info if not c["in_head"]]
        if missing_callers and classification == "compatibility_wrapper":
            problems.append(f"wrapper {row['id']} names missing callers {missing_callers}")
        removable = (
            classification == "compatibility_wrapper" and not (row.get("callers") or [])
        )
        loc = {
            "id": row["id"],
            "concept": row["concept"],
            "path": path,
            "line": line,
            "file_line": f"{path}:{line}" if line else path,
            "symbol": row["symbol"],
            "classification": classification,
            "plane": row["plane"],
            "two_real": bool(row.get("two_real")),
            "survives": bool(row.get("survives")),
            "role": row["role"],
            "move": row["move"],
            "callers": caller_info,
            "callers_unverified": missing_callers,
            "wrapper_removable": removable,
            "evidence_receipt": row.get("evidence"),
            "in_head": exists,
            "bytes": len(text.encode("utf-8")) if text is not None else None,
            "unknown_reason": unknown_reason,
        }
        impls.append(loc)
    return impls, problems


def targets(impls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for i in impls:
        by[i["concept"]].append(i)
    out = []
    for concept in CONCEPTS:
        rows = by.get(concept, [])
        canonicals = [r for r in rows if r["classification"] == "canonical_authority"]
        wrappers = [r for r in rows if r["classification"] == "compatibility_wrapper"]
        # A live second definition that must not survive (merge shadow, dead
        # class) is still TWO authorities until it is removed.
        two = [r for r in rows if r.get("two_real")]
        removable = [r for r in wrappers if r.get("wrapper_removable")]
        justified = [
            r
            for r in wrappers
            if not r.get("wrapper_removable") and any(c.get("in_head") for c in r.get("callers") or [])
        ]
        out.append(
            {
                "concept": concept,
                "canonical_ids": [r["id"] for r in canonicals],
                "canonical_file_lines": [r["file_line"] for r in canonicals],
                "two_real_authorities": len(two) >= 2,
                "two_real_ids": [r["id"] for r in two],
                "wrappers_justified": [
                    {
                        "id": r["id"],
                        "file_line": r["file_line"],
                        "callers": [c["path"] for c in r["callers"] if c["in_head"]],
                    }
                    for r in justified
                ],
                "wrappers_removable": [
                    {"id": r["id"], "file_line": r["file_line"], "reason": r["move"]}
                    for r in removable
                ],
                "what_must_move": [
                    {"id": r["id"], "survives": r["survives"], "move": r["move"]}
                    for r in rows
                ],
                "unknown": [
                    {"id": r["id"], "reason": r.get("unknown_reason")}
                    for r in rows
                    if r["classification"] == "UNKNOWN"
                ],
            }
        )
    return out


def write_scope_check() -> Dict[str, Any]:
    r = git("status", "--porcelain", "--untracked-files=all", check=False)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    allowed = {
        "tools/headless/authority_census.py",
        "receipts/headless/AUTHORITY_CENSUS.json",
    }
    dirty = []
    for ln in lines:
        path = ln[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path not in allowed:
            dirty.append(ln)
    return {
        "write": [
            "tools/headless/authority_census.py",
            "receipts/headless/AUTHORITY_CENSUS.json",
        ],
        "deny": [
            "workspace",
            "crates",
            "visionmcp",
            "app",
            "lab",
            "hcli",
            "ramanujan",
            "receipts/ascent-2026-08-16",
            "receipts/ascent-2026-08-18",
        ],
        "git_status_porcelain": lines,
        "outside_write_scope": dirty,
        "clean_write_scope": not dirty,
        "how_verified": (
            "git status --porcelain --untracked-files=all after writing the "
            "receipt; every path other than the two WRITE targets is a violation. "
            "No rm / git mv / git clean / git checkout / git restore / git reset "
            "was invoked."
        ),
    }


WATCHED_FAIL = [
    {
        "id": 1,
        "what": "Working-tree git grep is silent on the live control plane",
        "detail": (
            "This worktree is a sparse checkout. `git grep class Goal -- '*.py'` "
            "returned nothing even though HEAD:hcli/goal.py exists. "
            "Every search in this census is `git grep … HEAD` / `git show HEAD:path`."
        ),
    },
    {
        "id": 2,
        "what": "A start-anchored `class Goal` regex misses GoalCompiler",
        "detail": (
            "GoalCompiler is the HCLI compiler authority. Word-boundary search "
            "for class Goal found only GoalNotMetError. Catalog needles are "
            "literal substrings resolved against the blob."
        ),
    },
    {
        "id": 3,
        "what": "format_status defined twice, later shadowing earlier — did not reproduce",
        "detail": (
            "HEAD hcli/commands.py has exactly one `def format_status` "
            "(line recovered live). TUI.render_status is a different function. "
            "The shadowing bug is not present at this HEAD; claiming it still "
            "disables a fix would be a wrong confident answer."
        ),
    },
    {
        "id": 4,
        "what": "_remaining_depth is not defined twice; the merge duplicate is next to it",
        "detail": (
            "scheduler.py defines _remaining_depth once. Immediately after it, "
            "LOG = logging.getLogger is assigned a second time and MAX_REPAIR_DEPTH "
            "/ MAX_REPAIRS_PER_ROOT are reassigned, shadowing the import from "
            "workunit.py. Dispatch does not call _remaining_depth (ready_at FIFO). "
            "The measurement copy lives in tools/headless/hcli_scheduler_quality.py."
        ),
    },
    {
        "id": 5,
        "what": "class MachineGenome is not the machine genome",
        "detail": (
            "The live genome is receipts/headless/MACHINE_GENOME.json produced by "
            "tools/headless/machine_probe.py and consumed by resolve_runtime_limits. "
            "class MachineGenome writes ~/.local/share/hcli/machine-genome.json and "
            "is constructed only by hcli_persistence_audit.py. Treating the class "
            "as canonical would split hardware identity."
        ),
    },
    {
        "id": 6,
        "what": "hide-kernel machine.rs is not machine identity",
        "detail": (
            "crates/hide-kernel/src/machine.rs is AgentDriver, the HIDE FSM. "
            "Citing it as a hardware-identity implementation is a name collision."
        ),
    },
    {
        "id": 7,
        "what": "hawking-speculate Verifier is not a WorkUnit verifier",
        "detail": (
            "crates/hawking-speculate/src/verifier.rs is the speculative-decode "
            "draft-token accept rule. Catalogued as historical_implementation "
            "under the verifier concept so a later reader does not merge it."
        ),
    },
    {
        "id": 8,
        "what": "Cannot `import hcli` in this worktree",
        "detail": (
            "tools/haider is not materialized. Classification does not execute "
            "HCLI; it reads HEAD blobs. A missing on-disk file is not evidence "
            "the symbol is gone."
        ),
    },
    {
        "id": 9,
        "what": "crates/hide-backend/src/bin/haider.rs is an empty blob",
        "detail": (
            "git cat-file shows e69de29 (empty). A second HCLI lives at "
            "crates/hide-backend/src/bin/hcli.rs, which is a real binary. "
            "haider is a fossil namespace, not a runner."
        ),
    },
    {
        "id": 10,
        "what": "hcli.context has no production importer",
        "detail": (
            "context.py re-exports WorkerPacket from goal.py. git grep of "
            "`from hcli.context` / `from .context import` over HEAD found no "
            "production caller. The wrapper is marked removable."
        ),
    },
    {
        "id": 11,
        "what": "Receipts cite hawking-copy paths",
        "detail": (
            "DISK_TRUTH / RUNTIME_AUTHORITY / others name "
            "/Users/scammermike/Downloads/hawking-copy/…. Those strings in "
            "historical receipts are evidence, not debt, and were not rewritten."
        ),
    },
    {
        "id": 12,
        "what": "Four lab ModelSpec classes were not byte-compared",
        "detail": (
            "Listed as UNKNOWN with a follow-up rather than a confident "
            "'copy-paste, delete three'. Wrong merge would drop operator-local fields."
        ),
    },
    {
        "id": 13,
        "what": "git grep -E with `\\b` matches nothing",
        "detail": (
            "POSIX ERE has no word boundary. An earlier scan used git grep -E "
            "and `\\b`, so SCAN HITS reported (none) while `class Scheduler` "
            "existed in two files. The census uses git grep -P (PCRE)."
        ),
    },
]


def build() -> Dict[str, Any]:
    impls, problems = build_implementations()
    extras = scan_head()
    extra_unknown: List[Dict[str, Any]] = []
    for concept, hits in extras.items():
        for hit in hits:
            if hit["kind"] == "test_or_example":
                continue
            extra_unknown.append(
                {
                    "concept": concept,
                    "path": hit["path"],
                    "line": hit["line"],
                    "file_line": f"{hit['path']}:{hit['line']}",
                    "snippet": hit["snippet"],
                    "reason": (
                        "Definition matched the scan pattern and is not in the "
                        "catalog. Follow-up; not classified by guess."
                    ),
                    "classification": "UNKNOWN",
                }
            )
    tgts = targets(impls)
    two = [t for t in tgts if t["two_real_authorities"]]
    unknown_rows = [i for i in impls if i["classification"] == "UNKNOWN"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    scope = write_scope_check()
    receipt = {
        "schema": SCHEMA,
        "generated_at": now,
        "git_head": git_head(),
        "repo": str(REPO),
        "method": (
            "HEAD-tree scan (sparse-safe git show/grep) + curated classification "
            "with live line recovery. Working-tree grep was not used as evidence "
            "of absence."
        ),
        "anti_goodhart": (
            "A 20% LOC reduction that leaves two Goal stores or two HCLI "
            "binaries unnamed is a failure. Optimise verified capability over "
            "line count. Ten science lanes are in flight against tools/headless "
            "and receipts/headless — this receipt is a plan, not a migration."
        ),
        "standing_facts": {
            "haider_is_a_fossil_namespace": True,
            "preserve_sealed_schema_ids": True,
            "preserve_historical_receipt_filenames": True,
            "untracked_work_is_real_work": True,
            "precious_corpora": [
                "workspace/campaign/records/ascension-sandbox/",
                "workspace/campaign/phaseB/",
            ],
        },
        "planes": {
            "hcli-py": "hcli — live campaign control plane (33 modules)",
            "hide-rs": "hide-kernel / hide-backend / hide-protocol, including rust bin/hcli.rs",
            "hawking-orch": "model-role admission",
            "hawking-serve": "decode-slot batching",
            "hawking-context": "HIDE context packer",
            "lab": "science campaign engine",
            "headless": "measurement harnesses",
            "ramanujan": "math verifier scaffold",
            "agentos": "genesis / machine_state",
            "haider-v0": "tools/hcli/bootstrap/snapshots/haider.py bootstrap",
            "hawking-speculate": "speculative-decode accept (name collision)",
        },
        "concepts": list(CONCEPTS),
        "implementations": impls,
        "targets": tgts,
        "two_real_authority_concepts": [t["concept"] for t in two],
        "unscanned_unknown": extra_unknown,
        "catalog_unknown": [
            {"id": i["id"], "reason": i.get("unknown_reason"), "file_line": i["file_line"]}
            for i in unknown_rows
        ],
        "self_check_problems": problems,
        "watched_fail": WATCHED_FAIL,
        "write_scope": scope,
        "counts": {
            "catalog_rows": len(impls),
            "canonical": sum(1 for i in impls if i["classification"] == "canonical_authority"),
            "wrappers": sum(1 for i in impls if i["classification"] == "compatibility_wrapper"),
            "obsolete": sum(1 for i in impls if i["classification"] == "obsolete_implementation"),
            "test_only": sum(1 for i in impls if i["classification"] == "test_only_implementation"),
            "historical": sum(1 for i in impls if i["classification"] == "historical_implementation"),
            "unknown": len(unknown_rows) + len(extra_unknown),
            "two_real_concepts": len(two),
            "scan_uncatalogued_production": len(extra_unknown),
        },
    }
    receipt["receipt_sha256_of_body_without_this_field"] = sha256_text(
        json.dumps({k: v for k, v in receipt.items() if k != "receipt_sha256_of_body_without_this_field"},
                   sort_keys=True, default=str)
    )
    return receipt


def format_report(r: Dict[str, Any]) -> str:
    a: List[str] = []
    a.append("=" * 78)
    a.append("AUTHORITY CENSUS")
    a.append(f"schema     {r['schema']}")
    a.append(f"git_head   {r['git_head']}")
    a.append(f"generated  {r['generated_at']}")
    a.append(f"receipt    {RECEIPT}")
    c = r["counts"]
    a.append(
        f"rows       catalog={c['catalog_rows']}  canonical={c['canonical']}  "
        f"wrappers={c['wrappers']}  obsolete={c['obsolete']}  "
        f"test_only={c['test_only']}  historical={c['historical']}  "
        f"UNKNOWN={c['unknown']}"
    )
    a.append(f"two-real   {c['two_real_concepts']} concepts: {', '.join(r['two_real_authority_concepts'])}")
    a.append("")
    a.append(r["anti_goodhart"])
    a.append("")
    a.append("## TWO REAL AUTHORITIES (the finding)")
    for t in r["targets"]:
        if not t["two_real_authorities"]:
            continue
        a.append(f"  {t['concept']}")
        for i in t["two_real_ids"]:
            row = next(x for x in r["implementations"] if x["id"] == i)
            a.append(f"    - {row['file_line']}  {row['symbol']}  plane={row['plane']}")
        a.append(f"    target: {t['canonical_ids']}")
    a.append("")
    a.append("## EVERY CONCEPT")
    for t in r["targets"]:
        a.append("")
        a.append(f"### {t['concept']}")
        a.append(f"    canonical: {', '.join(t['canonical_file_lines']) or '(none)'}")
        if t["two_real_authorities"]:
            a.append("    TWO_REAL_AUTHORITIES=yes")
        for w in t["wrappers_justified"]:
            a.append(
                f"    wrapper KEEP {w['file_line']}  callers={w['callers']}"
            )
        for w in t["wrappers_removable"]:
            a.append(f"    wrapper REMOVABLE {w['file_line']}  {w['reason']}")
        for u in t["unknown"]:
            a.append(f"    UNKNOWN {u['id']}: {u['reason']}")
        for impl in r["implementations"]:
            if impl["concept"] != t["concept"]:
                continue
            flag = "SURVIVES" if impl["survives"] else "does-not-survive"
            a.append(
                f"    [{impl['classification']}] {impl['file_line']}  "
                f"{impl['symbol']}  {flag}  plane={impl['plane']}"
            )
            a.append(f"        {impl['role']}")
            a.append(f"        MOVE: {impl['move']}")
    a.append("")
    a.append("## SCAN HITS NOT IN THE CATALOG (UNKNOWN, follow-up)")
    if not r["unscanned_unknown"]:
        a.append("  (none)")
    else:
        for u in r["unscanned_unknown"]:
            a.append(f"  {u['concept']:18s} {u['file_line']}")
            a.append(f"                     {u['snippet']}")
    a.append("")
    a.append("## WHAT I WATCHED FAIL")
    for w in r["watched_fail"]:
        a.append(f"  {w['id']}. {w['what']}")
        a.append(f"     {w['detail']}")
    a.append("")
    a.append("## WRITE SCOPE")
    ws = r["write_scope"]
    a.append(f"  WRITE {ws['write']}")
    a.append(f"  DENY  {ws['deny']}")
    a.append(f"  clean_write_scope={ws['clean_write_scope']}")
    a.append(f"  how: {ws['how_verified']}")
    if ws["outside_write_scope"]:
        a.append("  OUTSIDE WRITE SCOPE:")
        for ln in ws["outside_write_scope"]:
            a.append(f"    {ln}")
    if r["self_check_problems"]:
        a.append("")
        a.append("## SELF-CHECK PROBLEMS")
        for p in r["self_check_problems"]:
            a.append(f"  - {p}")
    a.append("=" * 78)
    return "\n".join(a) + "\n"


def main() -> int:
    receipt = build()
    # Recompute write-scope AFTER writing the receipt so git status sees it.
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(receipt, indent=2, default=str) + "\n"
    RECEIPT.write_text(text)
    receipt["write_scope"] = write_scope_check()
    text = json.dumps(receipt, indent=2, default=str) + "\n"
    RECEIPT.write_text(text)
    report = format_report(receipt)
    sys.stdout.write(report)
    print(f"wrote {RECEIPT} ({RECEIPT.stat().st_size} bytes)")
    problems = list(receipt["self_check_problems"])
    if receipt["write_scope"]["outside_write_scope"]:
        problems.append("git status shows paths outside write scope")
    # Every concept must have at least one canonical unless UNKNOWN-only.
    for t in receipt["targets"]:
        if not t["canonical_ids"] and not t["unknown"]:
            problems.append(f"concept {t['concept']} has no canonical_authority")
        if t["concept"] not in CONCEPTS:
            problems.append(f"unexpected concept {t['concept']}")
    if len(receipt["targets"]) != len(CONCEPTS):
        problems.append("target count != concept count")
    if problems:
        print("CENSUS SELF-CHECK FAILED:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
