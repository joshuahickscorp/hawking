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
from typing import Any, Iterable, Mapping, Optional

from hcli.physical_graph import (
    MEGA_KERNEL_SCHEMA,
    MEGA_KERNEL_STAGES,
    _decode_state_layout,
)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = REPO / "receipts/headless/APPLE_ANE_DEVICE_PROFILE.json"
DEFAULT_ATLAS = REPO / "receipts/headless/APPLE_ANE_ATLAS.json"
MODEL_CHARACTERIZATION_SCHEMA = "hcli.ane.model_characterization.v1"


def _positive_int(value: Any) -> Optional[int]:
    """Parse an optional measured nanosecond value without inventing zero."""

    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _shape(value: Any) -> list[int]:
    """Return a deterministic integer shape, or an empty unresolved shape."""

    if not isinstance(value, (list, tuple)):
        return []
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            return []
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            return []
        if parsed <= 0:
            return []
        result.append(parsed)
    return result


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

    def characterize_model(
        self,
        model_id: str,
        organs: Iterable[Mapping[str, Any]],
        *,
        source_seal: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build a model-specific ANE work card for the PhysicalGraph.

        This is deliberately a planning/evidence record, not a device driver.
        A model gets its own organ rows because an ANE result for one model or
        one shape cannot be promoted as a generic result for another.  Missing
        plan, atlas, or complete-work evidence remains explicit and keeps every
        row out of promotion.
        """

        model_name = str(model_id or "unknown")
        normalized: list[tuple[str, Mapping[str, Any]]] = []
        for index, organ in enumerate(organs):
            if not isinstance(organ, Mapping):
                continue
            organ_id = str(organ.get("id") or organ.get("name") or f"organ-{index}")
            normalized.append((organ_id, organ))
        normalized.sort(key=lambda item: item[0])

        rows: list[dict[str, Any]] = []
        for organ_id, organ in normalized:
            operation = str(
                organ.get("operation")
                or organ.get("operator")
                or organ.get("kind")
                or "unknown"
            )
            shape = _shape(organ.get("shape"))
            complete_token_ns = _positive_int(
                organ.get("complete_token_ns") or organ.get("complete_useful_ns")
            )
            transfer_ns = _positive_int(organ.get("transfer_ns")) or 0
            row = self.score_candidate(
                operation=operation,
                shape=shape,
                complete_token_ns=complete_token_ns,
                transfer_ns=transfer_ns,
            )
            row.update(
                {
                    "model_id": model_name,
                    "organ_id": organ_id,
                    "source_seal": source_seal,
                    "measurement_state": organ.get("measurement_state")
                    or ("MEASURED" if complete_token_ns is not None else "UNMEASURED"),
                    "shape_source": organ.get("shape_source") or "unreported",
                    "representation": organ.get("representation") or "model_declared",
                    "capability_verified": organ.get("capability_verified") is True,
                    "fallback": bool(organ.get("fallback")),
                    "benchmark_class": str(
                        organ.get("benchmark_class") or "UNKNOWN"
                    ).upper(),
                    "controls_required": [
                        "CPU",
                        "GPU",
                        "ANE",
                        "CPU_PLUS_ANE",
                        "CPU_PLUS_GPU",
                        "CONCURRENT_WHERE_JUSTIFIED",
                    ],
                }
            )
            # CPU/GPU/ANE are a matched per-model experiment, not three
            # independent facts that happen to appear in a generic machine
            # dossier.  Keep the CPU/GPU controls on every exact organ card so
            # a later ANE result cannot be compared to a different shape,
            # representation, source, or token-wall definition.
            #
            # Only the ANE column has public-API placement evidence at this
            # boundary.  CPU/GPU values remain explicitly unmeasured until
            # their own model-scoped runner emits complete-token timings.
            row["backend_matrix"] = {
                "cpu": self._control_row(
                    backend="CPU",
                    model_id=model_name,
                    organ_id=organ_id,
                    source_seal=source_seal,
                    operation=operation,
                    shape=shape,
                    representation=row["representation"],
                ),
                "gpu": self._control_row(
                    backend="GPU",
                    model_id=model_name,
                    organ_id=organ_id,
                    source_seal=source_seal,
                    operation=operation,
                    shape=shape,
                    representation=row["representation"],
                ),
                "ane": {
                    "backend": "ANE",
                    "model_id": model_name,
                    "organ_id": organ_id,
                    "source_seal": source_seal,
                    "operation": operation,
                    "shape": list(shape),
                    "representation": row["representation"],
                    "measurement_state": row["measurement_state"],
                    "placement_evidence": {
                        "available": row["available"],
                        "plan_ready": row["plan_ready"],
                        "ane_supported": row["ane_supported"],
                        "atlas_ready": self.atlas_ready,
                    },
                    "complete_token_ns": row["complete_token_ns"],
                    "transfer_ns": row["transfer_ns"],
                    "capability_verified": row["capability_verified"],
                    "fallback": row["fallback"],
                    "benchmark_class": row["benchmark_class"],
                    "promotion_eligible": False,
                },
            }
            # Recompute the gate after adding model-scoped evidence fields.
            row["eligible_for_promotion"] = bool(
                row["eligible_for_promotion"]
                and row["capability_verified"]
                and row["benchmark_class"] in {"PROTECTED_ABSOLUTE", "QUALIFIED_PROTECTED"}
                and not row["fallback"]
            )
            row["backend_matrix"]["ane"]["promotion_eligible"] = row[
                "eligible_for_promotion"
            ]
            rows.append(row)

        blocker_codes: list[str] = []
        if not self.available:
            blocker_codes.append("neural_engine_not_observed")
        if not self.plan_ready:
            blocker_codes.append("mlcomputeplan_not_ready")
        if not self.atlas_ready:
            blocker_codes.append("measured_ane_atlas_not_ready")
        if not rows:
            blocker_codes.append("model_organs_not_declared")
        if not any(row["eligible_for_promotion"] for row in rows):
            blocker_codes.append("no_model_specific_protected_complete_work")

        return {
            "schema": MODEL_CHARACTERIZATION_SCHEMA,
            "model_id": model_name,
            "source_seal": source_seal,
            "model_specific": True,
            "public_api_only": True,
            "evidence_class": "MODEL_SCOPED_PLAN_OR_MEASUREMENT",
            "device_observation": self.to_dict(),
            "organs": rows,
            "controls_required": [
                "CPU",
                "GPU",
                "ANE",
                "CPU_PLUS_ANE",
                "CPU_PLUS_GPU",
                "CONCURRENT_WHERE_JUSTIFIED",
            ],
            "mega_kernel": {
                "schema": MEGA_KERNEL_SCHEMA,
                "role": "shared_physical_execution_spine",
                "stages": list(MEGA_KERNEL_STAGES),
                "model_id": model_name,
                "backend_slots": ["cpu", "gpu", "ane", "remote"],
                "dynamic_slots": [
                    "token",
                    "position",
                    "route",
                    "representation",
                    "state_layout",
                    "sampling",
                ],
                "organ_rows_are_model_specific": True,
                "decode_state_layout": _decode_state_layout([
                    {
                        "organ_id": row["organ_id"],
                        "operation": row["operation"],
                        "state": row.get("state", "model_declared"),
                    }
                    for row in rows
                ]),
                "fusion_policy": "fuse_only_after_source_parity_and_complete_work_measurement",
                "selection_authority": "measured_complete_useful_work_after_capability_gate",
                "not_a_single_vendor_kernel_claim": True,
            },
            "promotion": {
                "status": "WITHHELD",
                "eligible_organs": [row["organ_id"] for row in rows if row["eligible_for_promotion"]],
                "blockers": sorted(set(blocker_codes)),
                "rule": "model-scoped protected plan + source-parity organ result + complete-token wall + no fallback",
            },
            "reopen_conditions": [
                "public Core ML authoring and compilation succeeds for this model or organ",
                "MLComputePlan placement is observed for the exact model-scoped operation",
                "matched CPU/GPU/ANE controls include transfer and synchronization",
                "source-parity and capability receipts are independently verified",
                "complete-token protected wall-time result exists",
            ],
        }

    @staticmethod
    def _control_row(
        *,
        backend: str,
        model_id: str,
        organ_id: str,
        source_seal: Optional[str],
        operation: str,
        shape: list[int],
        representation: Any,
    ) -> dict[str, Any]:
        """Build a matched CPU/GPU control card without inventing a result."""

        return {
            "backend": backend,
            "model_id": model_id,
            "organ_id": organ_id,
            "source_seal": source_seal,
            "operation": operation,
            "shape": list(shape),
            "representation": representation,
            "measurement_state": "UNMEASURED",
            "placement_evidence": "MODEL_SCOPED_RUN_REQUIRED",
            "complete_token_ns": None,
            "transfer_ns": None,
            "capability_verified": False,
            "fallback": None,
            "benchmark_class": "UNKNOWN",
            "promotion_eligible": False,
        }

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


__all__ = [
    "ANEProvider",
    "DEFAULT_ATLAS",
    "DEFAULT_PROFILE",
    "MEGA_KERNEL_SCHEMA",
    "MEGA_KERNEL_STAGES",
    "MODEL_CHARACTERIZATION_SCHEMA",
]
