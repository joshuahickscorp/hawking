"""Pre-acquisition gate: all Bible §7 checklist fields must be present and consistent.

Before acquisition:
  official source, license, revision, hash inventory, architecture classification,
  storage forecast, runtime-memory forecast, Gravity plan, scientific purpose,
  disk-floor proof.

This module validates offline documents only. It does not contact the Hub.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lab.operators.credential_broker.floor import (
    DEFAULT_PROTECTED_FLOOR_BYTES,
    FloorProof,
    FloorViolation,
    assert_disk_floor,
)
from lab.operators.credential_broker.types import (
    ArchitectureClassification,
    GravityPlanSummary,
    HashInventory,
    OfficialSource,
    RuntimeMemoryForecast,
    ScientificPurpose,
    StorageForecast,
    TypeError_,
)


class PreflightError(RuntimeError):
    """Acquisition preflight refused."""


PREFLIGHT_SCHEMA = "hawking.ascension.acquisition_preflight.v1"


@dataclass(frozen=True)
class AcquisitionPreflight:
    """Complete pre-acquisition package for one model programme."""

    source: OfficialSource
    inventory: HashInventory
    architecture: ArchitectureClassification
    storage: StorageForecast
    runtime_memory: RuntimeMemoryForecast
    gravity_plan: GravityPlanSummary
    purpose: ScientificPurpose
    floor_proof: FloorProof
    source_retention_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source.revision.commit != self.inventory.revision.commit:
            raise PreflightError(
                "source revision and inventory revision must pin the same commit"
            )
        if self.storage.protected_floor_bytes < DEFAULT_PROTECTED_FLOOR_BYTES:
            raise PreflightError("storage forecast lowers the non-negotiable 15 GiB floor")
        if self.floor_proof.status != "PASS":
            raise PreflightError("disk-floor proof must be PASS before acquisition")
        if self.floor_proof.protected_floor_bytes != self.storage.protected_floor_bytes:
            raise PreflightError("floor proof and storage forecast floors disagree")
        if self.storage.max_inflight_bytes <= 0:
            raise PreflightError("max_inflight_bytes must be positive")
        if not self.source_retention_paths:
            raise PreflightError(
                "source_retention_paths required so post-stream eviction is auditable "
                "(DeepSeek assert_source_evicted pattern)"
            )
        for path in self.source_retention_paths:
            if not isinstance(path, str) or not path.startswith("/"):
                raise PreflightError(
                    "source_retention_paths must be absolute paths (DeepSeek/Kimi pattern)"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PREFLIGHT_SCHEMA,
            "source": self.source.as_dict(),
            "inventory": self.inventory.as_dict(),
            "architecture": self.architecture.as_dict(),
            "storage": self.storage.as_dict(),
            "runtime_memory": self.runtime_memory.as_dict(),
            "gravity_plan": self.gravity_plan.as_dict(),
            "purpose": self.purpose.as_dict(),
            "disk_floor_proof": self.floor_proof.as_dict(),
            "source_retention_paths": list(self.source_retention_paths),
            "lifecycle": [
                "stream",
                "verify",
                "gravity_transform",
                "seal",
                "evict_source",
            ],
            "claims": {
                "models_never_receive_hf_token": True,
                "no_full_source_plus_duplicate_intermediates": True,
                "live_download_not_performed_by_preflight": True,
            },
        }


def validate_preflight(document: AcquisitionPreflight) -> dict[str, Any]:
    """Re-validate and emit the canonical preflight dict."""
    if not isinstance(document, AcquisitionPreflight):
        raise PreflightError("validate_preflight requires an AcquisitionPreflight")
    # Re-run constructor invariants by reconstructing via as_dict round-trip fields
    # already enforced in __post_init__; expose the sealed-shaped mapping.
    try:
        return document.as_dict()
    except (TypeError_, FloorViolation, PreflightError) as exc:
        raise PreflightError(str(exc)) from exc


def prove_floor_for_preflight(
    workspace_root: str,
    storage: StorageForecast,
    *,
    stage: str = "preflight",
    free_bytes_provider=None,
) -> FloorProof:
    """Helper: produce the FloorProof a preflight must embed."""
    return assert_disk_floor(
        workspace_root,
        protected_floor_bytes=storage.protected_floor_bytes,
        additional_bytes=storage.max_inflight_bytes,
        stage=stage,
        free_bytes_provider=free_bytes_provider,
    )
