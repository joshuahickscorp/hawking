"""Genesis lineage reproduction machinery.

Three named slots, an external promotion gate, checksummed state transfer,
a typed research bus, and the four-slot scheduler.
"""
from __future__ import annotations

from importlib import import_module

__all__ = [
    "ALL_CLAUSES",
    "CANDIDATE",
    "CLAUSE_ARTIFACT_IDENTITY",
    "CLAUSE_BENCHMARK_UNCHANGED",
    "CLAUSE_BPW_UP_TOKEN_DOWN",
    "CLAUSE_CAPABILITY",
    "CLAUSE_COMPLETE_TOKEN_MATERIAL",
    "CLAUSE_NO_NEW_SILENT_FALLBACK",
    "CLAUSE_PROTECTED_TESTS",
    "CLAUSE_REPRESENTATION_BPW",
    "CLAUSE_ROLLBACK_ARTIFACT",
    "CLAUSE_RUNTIME_GENOME",
    "CLAUSE_STATE_TRANSFER",
    "CLAUSE_TPS_UP_CAP_DOWN",
    "CURRENT",
    "DEFAULT_BENCHMARK_FINGERPRINT",
    "DEFAULT_CAPABILITY_CONTRACT",
    "GENESIS_BPW",
    "GENESIS_COMPLETE_TOKEN_NS",
    "HARD_CAP",
    "LAST_KNOWN_GOOD",
    "PREEMPTIBLE_SLOT",
    "SLOT_NAMES",
    "SLOT_ROLES",
    "TRANSFER_PAYLOAD_KEYS",
    "BusRefusal",
    "EpistemicClass",
    "FourSlotScheduler",
    "GenesisInstance",
    "Invoker",
    "LaunchResult",
    "LineageError",
    "LineageInvariantError",
    "LineageState",
    "MessageType",
    "ResearchBus",
    "SelfCertificationRefused",
    "SlotCapError",
    "SlotError",
    "TransferChecksumError",
    "TransferError",
    "accept_transfer",
    "clause_status",
    "evaluate_promotion",
    "make_qwen38_genesis",
    "pack_state",
    "parent_research_payload",
    "refuse_self_certification",
    "reproduce",
    "validate_message",
]

_LAZY_ATTRS = {
    "BusRefusal": ("bus", "BusRefusal"),
    "EpistemicClass": ("bus", "EpistemicClass"),
    "MessageType": ("bus", "MessageType"),
    "ResearchBus": ("bus", "ResearchBus"),
    "validate_message": ("bus", "validate_message"),
    "reproduce": ("cycle", "reproduce"),
    "HARD_CAP": ("four_slots", "HARD_CAP"),
    "PREEMPTIBLE_SLOT": ("four_slots", "PREEMPTIBLE_SLOT"),
    "SLOT_ROLES": ("four_slots", "SLOT_ROLES"),
    "FourSlotScheduler": ("four_slots", "FourSlotScheduler"),
    "SlotCapError": ("four_slots", "SlotCapError"),
    "SlotError": ("four_slots", "SlotError"),
    "DEFAULT_BENCHMARK_FINGERPRINT": ("identity", "DEFAULT_BENCHMARK_FINGERPRINT"),
    "DEFAULT_CAPABILITY_CONTRACT": ("identity", "DEFAULT_CAPABILITY_CONTRACT"),
    "GENESIS_BPW": ("identity", "GENESIS_BPW"),
    "GENESIS_COMPLETE_TOKEN_NS": ("identity", "GENESIS_COMPLETE_TOKEN_NS"),
    "GenesisInstance": ("identity", "GenesisInstance"),
    "Invoker": ("identity", "Invoker"),
    "make_qwen38_genesis": ("identity", "make_qwen38_genesis"),
    "ALL_CLAUSES": ("promotion", "ALL_CLAUSES"),
    "CLAUSE_ARTIFACT_IDENTITY": ("promotion", "CLAUSE_ARTIFACT_IDENTITY"),
    "CLAUSE_BENCHMARK_UNCHANGED": ("promotion", "CLAUSE_BENCHMARK_UNCHANGED"),
    "CLAUSE_BPW_UP_TOKEN_DOWN": ("promotion", "CLAUSE_BPW_UP_TOKEN_DOWN"),
    "CLAUSE_CAPABILITY": ("promotion", "CLAUSE_CAPABILITY"),
    "CLAUSE_COMPLETE_TOKEN_MATERIAL": ("promotion", "CLAUSE_COMPLETE_TOKEN_MATERIAL"),
    "CLAUSE_NO_NEW_SILENT_FALLBACK": ("promotion", "CLAUSE_NO_NEW_SILENT_FALLBACK"),
    "CLAUSE_PROTECTED_TESTS": ("promotion", "CLAUSE_PROTECTED_TESTS"),
    "CLAUSE_REPRESENTATION_BPW": ("promotion", "CLAUSE_REPRESENTATION_BPW"),
    "CLAUSE_ROLLBACK_ARTIFACT": ("promotion", "CLAUSE_ROLLBACK_ARTIFACT"),
    "CLAUSE_RUNTIME_GENOME": ("promotion", "CLAUSE_RUNTIME_GENOME"),
    "CLAUSE_STATE_TRANSFER": ("promotion", "CLAUSE_STATE_TRANSFER"),
    "CLAUSE_TPS_UP_CAP_DOWN": ("promotion", "CLAUSE_TPS_UP_CAP_DOWN"),
    "SelfCertificationRefused": ("promotion", "SelfCertificationRefused"),
    "clause_status": ("promotion", "clause_status"),
    "evaluate_promotion": ("promotion", "evaluate_promotion"),
    "refuse_self_certification": ("promotion", "refuse_self_certification"),
    "CANDIDATE": ("state", "CANDIDATE"),
    "CURRENT": ("state", "CURRENT"),
    "LAST_KNOWN_GOOD": ("state", "LAST_KNOWN_GOOD"),
    "SLOT_NAMES": ("state", "SLOT_NAMES"),
    "LaunchResult": ("state", "LaunchResult"),
    "LineageError": ("state", "LineageError"),
    "LineageInvariantError": ("state", "LineageInvariantError"),
    "LineageState": ("state", "LineageState"),
    "TRANSFER_PAYLOAD_KEYS": ("transfer", "TRANSFER_PAYLOAD_KEYS"),
    "TransferChecksumError": ("transfer", "TransferChecksumError"),
    "TransferError": ("transfer", "TransferError"),
    "accept_transfer": ("transfer", "accept_transfer"),
    "pack_state": ("transfer", "pack_state"),
    "parent_research_payload": ("transfer", "parent_research_payload"),
}


def __getattr__(name: str):
    """Load only the lineage leg whose public symbol was requested."""
    spec = _LAZY_ATTRS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    value = getattr(import_module(f".{module_name}", __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
