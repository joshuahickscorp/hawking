#!/usr/bin/env python3
"""Issue one exact, file-only Qwen80 L0 state-handoff component lease.

The issuer deliberately has no Metal, process, watcher-control, benchmark, or
retry path.  It revalidates the normalized CPU preflight chain and the held
Qwen80 watcher record, then atomically creates one immutable lease document.
The separately authorized outer reaper is the only program that may consume
that lease for a child invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen80_source_token_l0_state_handoff_launcher as launcher
from lab.receipts import seal


LEASE_SCHEMA = launcher.FUTURE_LEASE_SCHEMA
LEASE_STATUS = launcher.FUTURE_LEASE_STATUS
WATCHER_HOLD_SCHEMA = "hawking.ascension.qwen80.watcher_gpu_coordination_hold.v1"
WATCHER_HOLD_STATUS = "HELD_QWEN80_RESPAWNED_STATE_CHILD_BEFORE_UNGUARDED_METAL_FIXTURE"
ISSUER_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_lease_issuer.v1"
ISSUER_STATUS = "ISSUED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_NON_TIMED_DEVICE_PARITY_LEASE_FILE_ONLY"


@dataclass(frozen=True)
class LeaseContext:
    """Exact inputs consumed by the one child outer-reaper invocation."""

    proof: launcher.PreflightContext
    lease: dict[str, Any]
    lease_evidence: dict[str, Any]
    lease_seal_sha256: str
    lease_id: str
    watcher_hold: dict[str, Any]
    watcher_hold_evidence: dict[str, Any]


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise launcher.SourceTokenL0StateHandoffLauncherError(f"{label} must be an object")
    return dict(value)


def _exact(value: object, expected: Mapping[str, Any], label: str) -> None:
    if _mapping(value, label) != dict(expected):
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            f"{label} drifted from exact evidence"
        )


def _watcher_hold(
    path: Path, *, context: launcher.PreflightContext
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a held watcher record without controlling or probing a watcher.

    This is intentionally an evidence check only.  The record is historical
    control-plane authority; no process scan is used as a substitute for it.
    """
    document = launcher._unsealed_json(path, "--watcher-hold")
    evidence = launcher._evidence(path, "--watcher-hold")
    if (
        document.get("schema") != WATCHER_HOLD_SCHEMA
        or document.get("status") != WATCHER_HOLD_STATUS
    ):
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "--watcher-hold schema/status does not prove the required Qwen80 hold"
        )
    source = _mapping(document.get("source_binding"), "watcher hold source_binding")
    handoff = context.child.handoff
    required = {
        "manifest_seal_sha256": handoff.manifest_seal_sha256,
        "source_body_audit_seal_sha256": handoff.source_audit_seal_sha256,
        "admission_receipt_seal_sha256": handoff.admission_receipt_seal_sha256,
        "source_revision": handoff.source_revision,
    }
    for field, expected in required.items():
        if source.get(field) != expected:
            raise launcher.SourceTokenL0StateHandoffLauncherError(
                f"watcher hold source_binding.{field} drifted"
            )
    if source.get("manifest_path") != str(launcher.MANIFEST_PATH.resolve(strict=True)):
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "watcher hold source_binding.manifest_path drifted"
        )
    preserved = _mapping(document.get("preserved"), "watcher hold preserved")
    if (
        preserved.get("runtime_watcher_parent") is not True
        or preserved.get("raw_bf16_or_mps_production_fallback") is not False
    ):
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "watcher hold does not preserve the required Qwen80 coordination boundary"
        )
    boundary = _mapping(document.get("claim_boundary"), "watcher hold claim_boundary")
    if (
        boundary.get("this_is_only_gpu_coordination") is not True
        or boundary.get("does_not_change_qwen80_runtime_qualification") is not True
        or boundary.get("does_not_establish_generation_hcli_tps_or_tournament_eligibility")
        is not True
    ):
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "watcher hold claim boundary drifted"
        )
    return document, evidence


def _lease_id(
    *,
    context: launcher.PreflightContext,
    watcher_hold_evidence: Mapping[str, Any],
    out: Path,
) -> str:
    """Derive a fresh lease ID from immutable proof, held watcher, and target."""
    payload = {
        "schema": LEASE_SCHEMA,
        "proof": launcher._binding(context.proof_evidence, context.proof_seal_sha256),
        "probe_binary": context.probe_binary,
        "watcher_hold": dict(watcher_hold_evidence),
        "out": str(out),
        "recorded_at": launcher._utc_now(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _lease_document(
    *,
    context: launcher.PreflightContext,
    issuance_child: launcher.ChildPreflightContext,
    watcher_hold_evidence: Mapping[str, Any],
    out: Path,
) -> dict[str, Any]:
    """Create the exact lease grammar accepted by the future Rust child."""
    handoff = context.child.handoff
    lease_id = _lease_id(
        context=context, watcher_hold_evidence=watcher_hold_evidence, out=out
    )
    return seal(
        {
            "schema": LEASE_SCHEMA,
            "status": LEASE_STATUS,
            "recorded_at": launcher._utc_now(),
            "lease_id": lease_id,
            "artifact_binding": {
                "manifest_document_sha256": handoff.manifest_evidence["sha256"],
                "manifest_seal_sha256": handoff.manifest_seal_sha256,
                "admission_receipt_seal_sha256": handoff.admission_receipt_seal_sha256,
            },
            "preflight_proof_binding": launcher._binding(
                context.proof_evidence, context.proof_seal_sha256
            ),
            "preflight_versioned_current_admission": _mapping(
                context.proof.get("versioned_current_admission"),
                "preflight versioned-current admission",
            ),
            "lease_issue_versioned_current_admission": launcher._versioned_current_observation(
                issuance_child, phase="lease_issuance"
            ),
            "outer_preflight": context.outer_preflight_evidence,
            "outer_preflight_seal_sha256": context.outer_preflight_seal_sha256,
            "l0_state_handoff_child_preflight": context.child.evidence,
            "l0_state_handoff_child_preflight_seal_sha256": context.child.seal_sha256,
            "baseline_l0_to_l1_handoff_authority": handoff.authority_evidence,
            "baseline_l0_to_l1_handoff_authority_seal_sha256": handoff.authority_seal_sha256,
            "handoff_contract": launcher._handoff_contract(),
            "probe_binary": context.probe_binary,
            "execution_policy": {
                "component": "qwen80_source_token_l0_state_handoff",
                "quiet_qwen80_device_lease": True,
                "strict_math": True,
                "timing_or_benchmarking_allowed": False,
                "l1_prefix_execution_allowed": False,
                "complete_layer_or_token_allowed": False,
                "tps_or_tg_claim_allowed": False,
            },
            "lifecycle": {
                "fresh_for_this_exact_launch": True,
                "outer_reaped_capture_required": True,
                "lease_released_after_first_terminal_child": True,
                "automatic_retry_prohibited": True,
                "replay_guarded": True,
            },
            "watcher_coordination": {
                "watcher_hold": dict(watcher_hold_evidence),
                "watcher_hold_must_remain_active": True,
                "watcher_restart_or_transition_authorized": False,
            },
            "prelaunch_claim": {
                "issuer_performed_a_fresh_read_only_prelaunch_check": True,
                "current_normalized_admission_pointer_revalidated": True,
                "qwen80_watcher_hold_remains_active": True,
                "no_server_hcli_tps_or_tournament_authorized": True,
            },
            "claim_boundary": {
                "l0_post_state_rollback_retained_output_pre_l1_component_only": True,
                "l1_binding_not_executed": True,
                "l1_prefix_executed": False,
                "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": True,
                "lease_issuance_is_file_and_cpu_only": True,
            },
        }
    )


def _validate_extra_lease_bindings(
    document: Mapping[str, Any],
    *,
    context: launcher.PreflightContext,
    watcher_hold_evidence: Mapping[str, Any],
) -> None:
    _exact(
        document.get("preflight_proof_binding"),
        launcher._binding(context.proof_evidence, context.proof_seal_sha256),
        "lease preflight proof binding",
    )
    _exact(
        _mapping(document.get("preflight_versioned_current_admission"), "lease preflight pointer"),
        _mapping(context.proof.get("versioned_current_admission"), "proof pointer"),
        "lease historical preflight pointer",
    )
    issuance_observation = _mapping(
        document.get("lease_issue_versioned_current_admission"), "lease issuance pointer"
    )
    # The pointer itself may be newly resealed; the existing launcher validates
    # that this historical observation still carries the exact immutable chain.
    launcher._validate_versioned_current_observation(
        issuance_observation,
        context.child,
        phase="lease_issuance",
        label="lease issuance versioned-current admission",
    )
    watcher = _mapping(document.get("watcher_coordination"), "lease watcher coordination")
    _exact(
        watcher.get("watcher_hold"), watcher_hold_evidence, "lease watcher hold evidence"
    )
    if (
        watcher.get("watcher_hold_must_remain_active") is not True
        or watcher.get("watcher_restart_or_transition_authorized") is not False
    ):
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "lease watcher coordination drifted"
        )


def validate_lease(
    *,
    lease_receipt: Path,
    preflight_proof: Path,
    child_preflight: Path,
    handoff_authority: Path,
    probe_bin: Path,
    watcher_hold: Path,
) -> LeaseContext:
    """Validate a lease and recheck all mutable evidence immediately before use."""
    context = launcher.validate_preflight_proof(
        proof_path=preflight_proof,
        child_preflight=child_preflight,
        handoff_authority=handoff_authority,
        probe_bin=probe_bin,
    )
    # Make a fresh normalized-current check.  A valid pointer reseal is okay;
    # a manifest/receipt substitution remains an immediate hard refusal.
    launcher.validate_child_preflight(
        child_preflight, handoff_authority=handoff_authority
    )
    watcher_document, watcher_evidence = _watcher_hold(watcher_hold, context=context)
    lease, lease_evidence, lease_seal, lease_id = launcher._bind_future_lease(
        lease_receipt, context
    )
    _validate_extra_lease_bindings(
        lease,
        context=context,
        watcher_hold_evidence=watcher_evidence,
    )
    return LeaseContext(
        proof=context,
        lease=lease,
        lease_evidence=lease_evidence,
        lease_seal_sha256=lease_seal,
        lease_id=lease_id,
        watcher_hold=watcher_document,
        watcher_hold_evidence=watcher_evidence,
    )


def issue_lease(
    *,
    preflight_proof: Path,
    child_preflight: Path,
    handoff_authority: Path,
    probe_bin: Path,
    watcher_hold: Path,
    out: Path,
) -> LeaseContext:
    """Issue one create-new lease after only CPU/file evidence checks."""
    if not out.is_absolute() or out.exists():
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "--out must be a new absolute lease path"
        )
    context = launcher.validate_preflight_proof(
        proof_path=preflight_proof,
        child_preflight=child_preflight,
        handoff_authority=handoff_authority,
        probe_bin=probe_bin,
    )
    issuance_child = launcher.validate_child_preflight(
        child_preflight, handoff_authority=handoff_authority
    )
    watcher_document, watcher_evidence = _watcher_hold(watcher_hold, context=context)
    document = _lease_document(
        context=context,
        issuance_child=issuance_child,
        watcher_hold_evidence=watcher_evidence,
        out=out,
    )
    launcher._write_new(out, document)
    return validate_lease(
        lease_receipt=out,
        preflight_proof=preflight_proof,
        child_preflight=child_preflight,
        handoff_authority=handoff_authority,
        probe_bin=probe_bin,
        watcher_hold=watcher_hold,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-proof", type=Path, required=True)
    parser.add_argument("--child-preflight", type=Path, default=launcher.CHILD_PREFLIGHT_PATH)
    parser.add_argument("--handoff-authority", type=Path, default=launcher.HANDOFF_AUTHORITY_PATH)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--watcher-hold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = issue_lease(
            preflight_proof=args.preflight_proof,
            child_preflight=args.child_preflight,
            handoff_authority=args.handoff_authority,
            probe_bin=args.probe_bin,
            watcher_hold=args.watcher_hold,
            out=args.out,
        )
    except (launcher.SourceTokenL0StateHandoffLauncherError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema": ISSUER_SCHEMA,
                    "status": "REFUSED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_LEASE_ISSUANCE",
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "schema": ISSUER_SCHEMA,
                "status": ISSUER_STATUS,
                "lease": result.lease_evidence,
                "lease_id": result.lease_id,
                "seal_sha256": result.lease_seal_sha256,
                "watcher_hold": result.watcher_hold_evidence,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
