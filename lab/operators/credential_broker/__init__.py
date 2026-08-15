"""Reusable credential broker for official model-source acquisition.

Bible §7: models never receive the Hugging Face token. This package generalises
the discipline already proven by tonight's DeepSeek-V4 streamer, GLM streamer,
Kimi K3 source admission, and source-only reclaim paths into one reusable
surface any future model (Qwen 30B, Qwen 80B, …) can call without rebuilding
stream→verify→Gravity-transform→seal→evict from scratch.

This package is a **scaffold**: pure offline types, preflight gates, and a
lifecycle state machine. It does not perform live Hub downloads, does not hold
or transmit tokens over the network, and does not start Qwen acquisition.
Live transport remains a future, separately gated executor.
"""
from __future__ import annotations

from lab.operators.credential_broker.floor import (
    DEFAULT_PROTECTED_FLOOR_BYTES,
    FloorProof,
    FloorViolation,
    assert_disk_floor,
)
from lab.operators.credential_broker.lifecycle import (
    LifecycleError,
    LifecyclePhase,
    SourceLifecycle,
)
from lab.operators.credential_broker.preflight import (
    AcquisitionPreflight,
    PreflightError,
    validate_preflight,
)
from lab.operators.credential_broker.secrets import (
    CredentialBroker,
    CredentialBrokerError,
    TokenHandle,
)
from lab.operators.credential_broker.types import (
    ArchitectureClassification,
    FileEntry,
    HashInventory,
    ImmutableRevision,
    OfficialSource,
    RangeRequest,
    ScientificPurpose,
    StorageForecast,
)

__all__ = [
    "AcquisitionPreflight",
    "ArchitectureClassification",
    "CredentialBroker",
    "CredentialBrokerError",
    "DEFAULT_PROTECTED_FLOOR_BYTES",
    "FileEntry",
    "FloorProof",
    "FloorViolation",
    "HashInventory",
    "ImmutableRevision",
    "LifecycleError",
    "LifecyclePhase",
    "OfficialSource",
    "PreflightError",
    "RangeRequest",
    "ScientificPurpose",
    "SourceLifecycle",
    "StorageForecast",
    "TokenHandle",
    "assert_disk_floor",
    "validate_preflight",
]
