"""Public Core ML Neural Engine provider for the Physical Graph Compiler.

This module is deliberately an evidence adapter, not an ANE driver.  Execution
stays in Apple's public Core ML APIs (the Swift probe and compiled MLProgram
assets); the compiler may consider ANE only when a device profile, MLComputePlan
placement, and measured complete-work evidence are present.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = REPO / "receipts/headless/APPLE_ANE_DEVICE_PROFILE.json"
DEFAULT_ATLAS = REPO / "receipts/headless/APPLE_ANE_ATLAS.json"


@dataclass(frozen=True)
class ANEProvider:
    """Compiler-facing ANE capability/evidence view.

    ``available`` means Core ML can see an ANE device.  It does not mean a
    particular operation is supported or that ANE wins the token wall.
    """

    profile: Mapping[str, Any]
    atlas: Mapping[str, Any]
    name: str = "ANEProvider"

    @classmethod
    def from_receipts(
        cls,
        profile_path: Path | str = DEFAULT_PROFILE,
        atlas_path: Path | str = DEFAULT_ATLAS,
    ) -> "ANEProvider":
        def read(path: Path | str) -> Mapping[str, Any]:
            try:
                value = json.loads(Path(path).read_text())
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return {}
            return value if isinstance(value, Mapping) else {}

        return cls(read(profile_path), read(atlas_path))

    @property
    def available(self) -> bool:
        return bool(self.profile.get("neural_engine_present"))

    @property
    def plan_ready(self) -> bool:
        plan = self.profile.get("mlcomputeplan")
        return isinstance(plan, Mapping) and plan.get("status") == "PLANNED"

    @property
    def atlas_ready(self) -> bool:
        return self.atlas.get("status") == "MEASURED"

    def operation_evidence(self, operation: str) -> Mapping[str, Any]:
        plan = self.profile.get("mlcomputeplan")
        if isinstance(plan, Mapping):
            for row in plan.get("operations", []):
                if isinstance(row, Mapping) and row.get("operator") == operation:
                    return row
        for row in self.atlas.get("graphs", []) if isinstance(self.atlas.get("graphs"), list) else []:
            if isinstance(row, Mapping) and row.get("operation") == operation:
                return row
        return {}

    def placement(self, operation: str, shape: list[int]) -> dict[str, Any]:
        evidence = dict(self.operation_evidence(operation))
        supported = evidence.get("supported") if isinstance(evidence.get("supported"), list) else []
        preferred = evidence.get("preferred")
        ane_supported = "NEURAL_ENGINE" in supported or preferred == "NEURAL_ENGINE"
        return {
            "provider": self.name,
            "operation": operation,
            "shape": list(shape),
            "available": self.available,
            "plan_ready": self.plan_ready,
            "ane_supported": ane_supported,
            "preferred": preferred,
            "supported": supported,
            "eligible": self.available and self.plan_ready and ane_supported and self.atlas_ready,
            "evidence": evidence,
        }

    def score_candidate(
        self,
        *,
        operation: str,
        shape: list[int],
        complete_token_ns: Optional[int],
        transfer_ns: int = 0,
    ) -> dict[str, Any]:
        """Return a measured-work score; never rank nominal utilization."""

        placement = self.placement(operation, shape)
        measured = complete_token_ns is not None and complete_token_ns > 0
        total = (complete_token_ns or 0) + max(0, transfer_ns)
        placement.update({
            "measured_complete_work": measured,
            "complete_token_ns": complete_token_ns,
            "transfer_ns": transfer_ns,
            "eligible_for_promotion": placement["eligible"] and measured,
            "objective": "minimize measured complete useful work",
            "total_candidate_ns": total if measured else None,
        })
        return placement

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.name,
            "public_api_only": True,
            "available": self.available,
            "plan_ready": self.plan_ready,
            "atlas_ready": self.atlas_ready,
            "profile_schema": self.profile.get("schema"),
            "atlas_schema": self.atlas.get("schema"),
            "selection_authority": "measured complete useful work, not nominal device utilization",
            "private_interface_control": "forbidden",
        }


__all__ = ["ANEProvider", "DEFAULT_ATLAS", "DEFAULT_PROFILE"]
