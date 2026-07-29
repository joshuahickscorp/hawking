#!/usr/bin/env python3.12
"""One campaign and experiment engine (C1 / lane A1).

Authority surface for every lifecycle verb that historical campaigns used to
reimplement: precheck, measure, allocate, pack, seal, monitor, resume, report.

Campaign and model variation is declarative :class:`ExperimentSpec` data.
Controllers retire; specs, fixtures, receipts, and reproduction commands remain.
"""
from __future__ import annotations

from .checkpoint import CheckpointStore, HashChainLog
from .governor import ResourceGovernor, ResourceLimits, ResourceSample
from .lease import LeaseError, SingletonLease
from .receipts import Receipt, ReceiptStore, seal_receipt, verify_receipt
from .runtime import CampaignRuntime, RunResult, run_campaign
from .seal_integrity import (
    SealIntegrityError,
    inspect_launcher_node,
    preflight_must_not_use_subprocess,
    reject_resealed_substitution,
    seal_document,
    verify_document_seal,
)
from .scheduler import Scheduler, WorkItem, WorkStatus
from .spec import (
    CampaignPhase,
    ExperimentSpec,
    ReopenCondition,
    ResourceClass,
    SpecError,
    StepSpec,
    load_spec,
    load_spec_path,
    validate_spec,
)
from .state_machine import (
    IllegalTransition,
    Phase,
    StateMachine,
    Transition,
)

__all__ = [
    "CampaignPhase",
    "CampaignRuntime",
    "CheckpointStore",
    "ExperimentSpec",
    "HashChainLog",
    "IllegalTransition",
    "LeaseError",
    "Phase",
    "Receipt",
    "ReceiptStore",
    "ReopenCondition",
    "ResourceClass",
    "ResourceGovernor",
    "ResourceLimits",
    "ResourceSample",
    "RunResult",
    "Scheduler",
    "SealIntegrityError",
    "SingletonLease",
    "SpecError",
    "StateMachine",
    "StepSpec",
    "Transition",
    "WorkItem",
    "WorkStatus",
    "inspect_launcher_node",
    "load_spec",
    "load_spec_path",
    "preflight_must_not_use_subprocess",
    "reject_resealed_substitution",
    "run_campaign",
    "seal_document",
    "seal_receipt",
    "validate_spec",
    "verify_document_seal",
    "verify_receipt",
]

__version__ = "1.0.0"
AUTHORITY = "tools.condense.engine"
SCHEMA = "hawking.condense.campaign_engine.v1"
