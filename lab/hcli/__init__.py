"""HCLI Agent OS scaffolds — self-evolution, Option-C sandbox, residency modes.

Future programme (Bible §§23–25), gated on Proto-Frankenstein offload.
These modules formalize interfaces and pure state machines. They do not load
Qwen/Gravity weights and do not grant sandbox models promotion authority.
"""
from __future__ import annotations

from lab.hcli.option_c import (
    MANDATORY_REVIEW_CATEGORIES,
    CandidateReport,
    OptionCController,
    OptionCSandbox,
    ReviewReport,
    Role,
)
from lab.hcli.residency import (
    ResidencyMode,
    ResidencyRefusal,
    ResidencyStateMachine,
    Slot,
)
from lab.hcli.self_evolution import (
    PROPOSAL_KINDS,
    AdmissionPipeline,
    AdmissionStage,
    EvolutionLedger,
    Proposal,
    ProposalKind,
    SelfEvolutionEngine,
)

__all__ = [
    "AdmissionPipeline",
    "AdmissionStage",
    "CandidateReport",
    "EvolutionLedger",
    "MANDATORY_REVIEW_CATEGORIES",
    "OptionCController",
    "OptionCSandbox",
    "PROPOSAL_KINDS",
    "Proposal",
    "ProposalKind",
    "ResidencyMode",
    "ResidencyRefusal",
    "ResidencyStateMachine",
    "ReviewReport",
    "Role",
    "SelfEvolutionEngine",
    "Slot",
]
