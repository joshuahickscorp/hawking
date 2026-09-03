#!/usr/bin/env python3.12
"""Metadata-only, fail-closed Qwen Ascension preflight (30B / 80B lanes).

This operator accepts **caller-supplied** manifest and evidence mappings and
validates them locally. It never contacts the network, never imports Hugging
Face / Xet, never loads a model, never launches a download, and never creates
cache files.

Default decision is **BLOCKED**. ``download_permitted`` stays false until every
required gate is green in the supplied evidence and the proposed family is
explicitly listed in runtime-capability evidence.

This is admission / preflight scaffolding only â not permission to fetch model
bodies. Exact public model identities must come from the supplied manifest;
this module does not invent Hub IDs, revision pins, or config digests.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Mapping

from lab.receipts import seal

# ---------------------------------------------------------------------------
# Schemas / vocabulary
# ---------------------------------------------------------------------------

PREFLIGHT_SCHEMA = "hawking.ascension.qwen_metadata_preflight.v1"
PREFLIGHT_STATUS_BLOCKED = "BLOCKED"
PREFLIGHT_STATUS_ADMITTED = "ADMITTED_METADATA_ONLY"

# Gates that must all be present and green before download may be permitted.
REQUIRED_GATES: tuple[str, ...] = (
    "source_admission",
    "pinned_identity",
    "runtime_loader_forward_support",
    "resource_supervisor_green",
    "actual_artifact_receipt",
    "profiler_parity_capability",
    "controller_approval",
)

# Prospective lanes only â family keys already used by the parity harness.
# Source repository / revision / digests are never hard-coded here.
class QwenLane(str, Enum):
    """Distinct prospective bootstrap lanes (bible Â§8 / Â§9)."""

    QWEN_30B = "30B"
    QWEN_80B = "80B"


LANE_RECORDS: dict[str, dict[str, Any]] = {
    QwenLane.QWEN_30B.value: {
        "lane_id": "qwen_30b",
        "scale_label": "30B",
        "family_key": "QWEN3_MOE",
        "role": "executor",
        "notes": (
            "Prospective 30B-class executor lane. Identity fields must be "
            "supplied by the caller; this scaffold does not pin a Hub ID."
        ),
    },
    QwenLane.QWEN_80B.value: {
        "lane_id": "qwen_80b",
        "scale_label": "80B",
        "family_key": "QWEN3_NEXT",
        "role": "reviewer",
        "notes": (
            "Prospective 80B-class reviewer / hybrid lane. Distinct family from "
            "30B; identity fields must be supplied by the caller."
        ),
    },
}

FAMILY_TO_LANE: dict[str, str] = {
    rec["family_key"]: scale for scale, rec in LANE_RECORDS.items()
}

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GREEN_STATUSES = frozenset({"PASS", "GREEN", "ADMITTED", "APPROVED", "TRUE"})


class PreflightInputError(ValueError):
    """Supplied manifest or evidence is malformed (fail closed)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreflightInputError(f"{label} must be a mapping")
    return value


def _nonempty_str(value: object, label: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _gate_status(raw: object) -> str:
    """Normalise a gate status token; unknown / missing â ABSENT."""
    if raw is True:
        return "PASS"
    if raw is False or raw is None:
        return "ABSENT"
    if isinstance(raw, Mapping):
        for key in ("status", "verdict", "state", "result"):
            if key in raw:
                return _gate_status(raw[key])
        if raw.get("green") is True or raw.get("passed") is True:
            return "PASS"
        return "ABSENT"
    if isinstance(raw, str):
        token = raw.strip().upper()
        if not token:
            return "ABSENT"
        if token in _GREEN_STATUSES or token in {"OK", "YES", "1"}:
            return "PASS"
        if token in {"FAIL", "FAILED", "RED", "REJECT", "REJECTED", "DENIED", "BLOCKED"}:
            return "FAIL"
        if token in {"PENDING", "ABSENT", "MISSING", "UNKNOWN", "WITHHELD"}:
            return "ABSENT"
        return token
    return "ABSENT"


def _is_green(status: str) -> bool:
    return status == "PASS"


def _identity_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Extract source / revision / config digest / license / architecture.

    Values are taken only from the supplied manifest. Missing fields stay null
    so callers can see exactly what was not provided â nothing is invented.
    """
    source = (
        _nonempty_str(manifest.get("source"), "source")
        or _nonempty_str(manifest.get("repository"), "repository")
        or _nonempty_str(manifest.get("hf_id"), "hf_id")
        or _nonempty_str(manifest.get("source_id"), "source_id")
    )
    revision = (
        _nonempty_str(manifest.get("revision"), "revision")
        or _nonempty_str(manifest.get("commit"), "commit")
        or _nonempty_str(
            (manifest.get("pinned_revision") or {}).get("commit")
            if isinstance(manifest.get("pinned_revision"), Mapping)
            else None,
            "pinned_revision.commit",
        )
    )
    if revision is not None:
        revision = revision.lower()

    config_digest = (
        _nonempty_str(manifest.get("config_digest"), "config_digest")
        or _nonempty_str(manifest.get("config_sha256"), "config_sha256")
    )
    if config_digest is not None:
        config_digest = config_digest.lower()

    license_id = (
        _nonempty_str(manifest.get("license"), "license")
        or _nonempty_str(manifest.get("license_id"), "license_id")
    )

    architecture = manifest.get("architecture") or manifest.get("architecture_identity")
    if isinstance(architecture, Mapping):
        arch_identity: dict[str, Any] = {
            "model_type": _nonempty_str(architecture.get("model_type"), "model_type"),
            "architectures": None,
        }
        arches = architecture.get("architectures")
        if isinstance(arches, (list, tuple)) and arches:
            cleaned = [a for a in arches if isinstance(a, str) and a.strip()]
            arch_identity["architectures"] = cleaned or None
        elif isinstance(arches, str) and arches.strip():
            arch_identity["architectures"] = [arches.strip()]
        for key in ("attention", "routing", "notes"):
            if key in architecture:
                arch_identity[key] = architecture[key]
    elif isinstance(architecture, str) and architecture.strip():
        arch_identity = {"model_type": architecture.strip(), "architectures": None}
    else:
        model_type = _nonempty_str(manifest.get("model_type"), "model_type")
        arch_identity = {"model_type": model_type, "architectures": None}

    return {
        "source": source,
        "revision": revision,
        "config_digest": config_digest,
        "license": license_id,
        "architecture_identity": arch_identity,
    }


def _resolve_lane(manifest: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    """Return (scale_label, reasons) for 30B vs 80B distinction."""
    reasons: list[str] = []
    scale = (
        _nonempty_str(manifest.get("scale_label"), "scale_label")
        or _nonempty_str(manifest.get("lane_scale"), "lane_scale")
        or _nonempty_str(manifest.get("scale"), "scale")
    )
    family = (
        _nonempty_str(manifest.get("family_key"), "family_key")
        or _nonempty_str(manifest.get("family"), "family")
        or _nonempty_str(manifest.get("model_family"), "model_family")
    )
    lane_id = _nonempty_str(manifest.get("lane_id"), "lane_id")

    if scale is not None:
        normalised = scale.upper().replace(" ", "")
        if normalised in {"30B", "30B-CLASS", "30B-A3B"}:
            scale = QwenLane.QWEN_30B.value
        elif normalised in {"80B", "80B-CLASS", "80B-CLASS-REVIEWER"}:
            scale = QwenLane.QWEN_80B.value
        elif normalised in LANE_RECORDS:
            scale = normalised
        else:
            reasons.append(f"unrecognised scale_label {scale!r}; expected 30B or 80B")
            return None, reasons

    if scale is None and family is not None:
        family_upper = family.upper()
        if family_upper in FAMILY_TO_LANE:
            scale = FAMILY_TO_LANE[family_upper]
        else:
            reasons.append(
                f"family_key {family!r} is not a recognised Qwen bootstrap family "
                f"(expected one of {sorted(FAMILY_TO_LANE)})"
            )

    if scale is None and lane_id is not None:
        for rec_scale, rec in LANE_RECORDS.items():
            if lane_id == rec["lane_id"] or lane_id.upper() == rec_scale:
                scale = rec_scale
                break
        if scale is None:
            reasons.append(f"lane_id {lane_id!r} does not map to a 30B or 80B lane")

    if scale is None:
        reasons.append(
            "manifest must identify a 30B or 80B lane via scale_label, family_key, or lane_id"
        )
        return None, reasons

    # Cross-check family against lane when both are present.
    expected_family = LANE_RECORDS[scale]["family_key"]
    if family is not None and family.upper() != expected_family:
        reasons.append(
            f"family_key {family!r} does not match lane {scale} "
            f"(expected {expected_family})"
        )
        return None, reasons

    return scale, reasons


def _evaluate_gates(evidence: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Score every required gate from supplied evidence. Missing â not green."""
    gates_raw = evidence.get("gates")
    if gates_raw is None:
        gates_raw = evidence
    if not isinstance(gates_raw, Mapping):
        return (
            {name: {"status": "ABSENT", "green": False, "detail": "gates mapping missing"}
             for name in REQUIRED_GATES},
            ["evidence.gates must be a mapping of required gate receipts"],
        )

    report: dict[str, Any] = {}
    reasons: list[str] = []
    for name in REQUIRED_GATES:
        if name not in gates_raw:
            report[name] = {
                "status": "ABSENT",
                "green": False,
                "detail": "gate receipt not supplied",
            }
            reasons.append(f"required gate {name!r} is absent")
            continue
        status = _gate_status(gates_raw[name])
        green = _is_green(status)
        entry: dict[str, Any] = {
            "status": status,
            "green": green,
            "detail": "supplied" if green else f"gate status is {status}",
        }
        raw = gates_raw[name]
        if isinstance(raw, Mapping):
            for keep in ("receipt_id", "schema", "note", "principal"):
                if keep in raw:
                    entry[keep] = raw[keep]
        report[name] = entry
        if not green:
            reasons.append(f"required gate {name!r} is not green (status={status})")
    return report, reasons


def _runtime_family_support(
    evidence: Mapping[str, Any],
    *,
    family_key: str | None,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Family must be *explicitly* listed in runtime-capability evidence."""
    reasons: list[str] = []
    runtime = evidence.get("runtime_capability") or evidence.get("runtime_capabilities")
    if not isinstance(runtime, Mapping):
        reasons.append(
            "runtime-capability evidence mapping is required and must explicitly "
            "list supported model families"
        )
        return False, {"present": False, "supported_families": [], "family_supported": False}, reasons

    supported_raw = (
        runtime.get("supported_families")
        or runtime.get("supported_model_families")
        or runtime.get("families")
    )
    supported: list[str] = []
    if isinstance(supported_raw, (list, tuple)):
        supported = [str(x).strip().upper() for x in supported_raw if str(x).strip()]
    elif isinstance(supported_raw, str) and supported_raw.strip():
        supported = [supported_raw.strip().upper()]

    loader = runtime.get("loader_support") or runtime.get("loader")
    forward = runtime.get("forward_support") or runtime.get("forward")
    loader_ok = _is_green(_gate_status(loader)) if loader is not None else False
    forward_ok = _is_green(_gate_status(forward)) if forward is not None else False

    if family_key is None:
        reasons.append("cannot verify runtime family support without a resolved family_key")
        family_supported = False
    else:
        family_supported = family_key.upper() in supported
        if not family_supported:
            reasons.append(
                f"proposed family {family_key!r} is not explicitly listed in "
                f"runtime_capability.supported_families ({supported or 'empty'})"
            )

    # Loader/forward evidence may live under runtime_capability *or* as the
    # dedicated required gate; surface both without inventing capability.
    detail = {
        "present": True,
        "supported_families": supported,
        "family_key": family_key,
        "family_supported": family_supported,
        "loader_support_green": loader_ok,
        "forward_support_green": forward_ok,
    }
    return family_supported, detail, reasons


def _identity_complete(identity: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Pinned identity requires source + 40-char revision + config digest + license + arch."""
    reasons: list[str] = []
    if not identity.get("source"):
        reasons.append("identity.source is missing (must be supplied; never invented here)")
    revision = identity.get("revision")
    if not revision:
        reasons.append("identity.revision is missing")
    elif _COMMIT_RE.fullmatch(str(revision)) is None:
        reasons.append(
            f"identity.revision must be a 40-character lowercase hex commit, got {revision!r}"
        )
    digest = identity.get("config_digest")
    if not digest:
        reasons.append("identity.config_digest is missing")
    elif _SHA256_RE.fullmatch(str(digest)) is None:
        reasons.append(
            f"identity.config_digest must be a 64-character hex sha256, got {digest!r}"
        )
    if not identity.get("license"):
        reasons.append("identity.license is missing")
    arch = identity.get("architecture_identity") or {}
    if not isinstance(arch, Mapping) or not arch.get("model_type"):
        reasons.append("identity.architecture_identity.model_type is missing")
    return (len(reasons) == 0), reasons


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def required_gates() -> tuple[str, ...]:
    """Return the ordered fail-closed gate checklist."""
    return REQUIRED_GATES


def lane_record(scale_label: str) -> dict[str, Any]:
    """Return the scaffold lane record for ``30B`` or ``80B``."""
    key = scale_label.strip().upper()
    if key not in LANE_RECORDS:
        raise PreflightInputError(f"unknown Qwen lane scale_label {scale_label!r}")
    return dict(LANE_RECORDS[key])


def distinguish_lane(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a manifest as the 30B or 80B prospective lane (metadata only)."""
    manifest = _as_mapping(manifest, "manifest")
    scale, reasons = _resolve_lane(manifest)
    if scale is None:
        return {
            "resolved": False,
            "scale_label": None,
            "lane": None,
            "reasons": reasons,
        }
    rec = LANE_RECORDS[scale]
    return {
        "resolved": True,
        "scale_label": scale,
        "lane_id": rec["lane_id"],
        "family_key": rec["family_key"],
        "role": rec["role"],
        "lane": dict(rec),
        "reasons": reasons,
    }


def evaluate_qwen_preflight(
    manifest: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    *,
    claimed_download_permitted: bool | None = None,
) -> dict[str, Any]:
    """Evaluate metadata-only Qwen Ascension preflight.

    Parameters
    ----------
    manifest:
        Caller-supplied identity for one prospective lane (30B or 80B). Must
        not rely on this function to invent source / revision / digests.
    evidence:
        Caller-supplied gate receipts and runtime-capability mapping.
    claimed_download_permitted:
        Optional claim from the caller. If true while gates fail, the decision
        records an explicit rejection of that claim.

    Returns
    -------
    Sealed decision dict. Default status is BLOCKED with download_permitted=false.
    """
    reasons: list[str] = []
    manifest_map: Mapping[str, Any] = manifest if isinstance(manifest, Mapping) else {}
    evidence_map: Mapping[str, Any] = evidence if isinstance(evidence, Mapping) else {}

    if not isinstance(manifest, Mapping):
        reasons.append("manifest mapping is required; defaulting to blocked")
    if not isinstance(evidence, Mapping):
        reasons.append("evidence mapping is required; defaulting to blocked")

    # --- Lane distinction (30B vs 80B) ------------------------------------
    lane_info = distinguish_lane(manifest_map) if isinstance(manifest, Mapping) else {
        "resolved": False,
        "scale_label": None,
        "lane": None,
        "reasons": ["manifest mapping is required"],
    }
    reasons.extend(lane_info.get("reasons") or [])
    family_key = lane_info.get("family_key") if lane_info.get("resolved") else None

    # --- Identity capture (never invent) ----------------------------------
    identity = _identity_fields(manifest_map) if isinstance(manifest, Mapping) else {
        "source": None,
        "revision": None,
        "config_digest": None,
        "license": None,
        "architecture_identity": {"model_type": None, "architectures": None},
    }
    identity_ok, identity_reasons = _identity_complete(identity)
    if not identity_ok:
        reasons.extend(identity_reasons)

    # --- Required gates ---------------------------------------------------
    gate_report, gate_reasons = _evaluate_gates(evidence_map)
    reasons.extend(gate_reasons)
    all_gates_green = all(g["green"] for g in gate_report.values()) and len(gate_report) == len(
        REQUIRED_GATES
    )

    # --- Runtime family support -------------------------------------------
    family_ok, runtime_detail, runtime_reasons = _runtime_family_support(
        evidence_map, family_key=family_key
    )
    reasons.extend(runtime_reasons)

    # --- Decision (fail closed) -------------------------------------------
    admitted = (
        bool(lane_info.get("resolved"))
        and identity_ok
        and all_gates_green
        and family_ok
    )
    download_permitted = bool(admitted)
    status = PREFLIGHT_STATUS_ADMITTED if admitted else PREFLIGHT_STATUS_BLOCKED

    claim_rejected = False
    if claimed_download_permitted is True and not download_permitted:
        claim_rejected = True
        reasons.append(
            "rejecting claimed download_permitted=true: not every required gate is green"
        )

    # De-duplicate reasons while preserving order.
    seen: set[str] = set()
    ordered_reasons: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            ordered_reasons.append(reason)

    if not ordered_reasons and not admitted:
        ordered_reasons.append("blocked by default: insufficient evidence")

    body: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "status": status,
        "download_permitted": download_permitted,
        "claimed_download_permitted": claimed_download_permitted,
        "claimed_download_permitted_rejected": claim_rejected,
        "lane": {
            "resolved": bool(lane_info.get("resolved")),
            "scale_label": lane_info.get("scale_label"),
            "lane_id": lane_info.get("lane_id"),
            "family_key": family_key,
            "role": lane_info.get("role"),
            "record": lane_info.get("lane"),
        },
        "identity": identity,
        "identity_complete": identity_ok,
        "gates": gate_report,
        "required_gates": list(REQUIRED_GATES),
        "all_gates_green": all_gates_green,
        "runtime_capability": runtime_detail,
        "family_supported_by_runtime": family_ok,
        "reasons": ordered_reasons,
        "claim_boundary": {
            "metadata_only": True,
            "network_calls": False,
            "hub_or_xet_invoked": False,
            "model_loaded": False,
            "download_launched": False,
            "cache_files_created": False,
            "permission_to_fetch_model_bodies": False,
            "invented_public_model_details": False,
            "scaffold_not_live_acquisition": True,
        },
        "honesty": {
            "default_decision": PREFLIGHT_STATUS_BLOCKED,
            "fail_closed": True,
            "download_permitted_requires_every_gate": True,
            "lanes_distinguished": ["30B", "80B"],
        },
    }
    return seal(body)


def default_blocked_decision() -> dict[str, Any]:
    """Empty-input baseline: always BLOCKED, download_permitted=false."""
    return evaluate_qwen_preflight(None, None)


__all__ = [
    "FAMILY_TO_LANE",
    "LANE_RECORDS",
    "PREFLIGHT_SCHEMA",
    "PREFLIGHT_STATUS_ADMITTED",
    "PREFLIGHT_STATUS_BLOCKED",
    "PreflightInputError",
    "QwenLane",
    "REQUIRED_GATES",
    "default_blocked_decision",
    "distinguish_lane",
    "evaluate_qwen_preflight",
    "lane_record",
    "required_gates",
]
