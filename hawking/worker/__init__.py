"""Hawking worker scaffolds — self-evolution, Option-C sandbox, residency modes.

Future programme (Bible §§23–25), gated on Proto-Frankenstein offload.
These modules formalize interfaces and pure state machines. They do not load
Qwen/Gravity weights and do not grant sandbox models promotion authority.
"""
from __future__ import annotations

from importlib import import_module

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

_LAZY_ATTRS = {
    "MANDATORY_REVIEW_CATEGORIES": ("option_c", "MANDATORY_REVIEW_CATEGORIES"),
    "CandidateReport": ("option_c", "CandidateReport"),
    "OptionCController": ("option_c", "OptionCController"),
    "OptionCSandbox": ("option_c", "OptionCSandbox"),
    "ReviewReport": ("option_c", "ReviewReport"),
    "Role": ("option_c", "Role"),
    "ResidencyMode": ("residency", "ResidencyMode"),
    "ResidencyRefusal": ("residency", "ResidencyRefusal"),
    "ResidencyStateMachine": ("residency", "ResidencyStateMachine"),
    "Slot": ("residency", "Slot"),
    "PROPOSAL_KINDS": ("self_evolution", "PROPOSAL_KINDS"),
    "AdmissionPipeline": ("self_evolution", "AdmissionPipeline"),
    "AdmissionStage": ("self_evolution", "AdmissionStage"),
    "EvolutionLedger": ("self_evolution", "EvolutionLedger"),
    "Proposal": ("self_evolution", "Proposal"),
    "ProposalKind": ("self_evolution", "ProposalKind"),
    "SelfEvolutionEngine": ("self_evolution", "SelfEvolutionEngine"),
}


def __getattr__(name: str):
    """Load a worker leg only when its public symbol is actually requested."""
    spec = _LAZY_ATTRS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    value = getattr(import_module(f".{module_name}", __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
