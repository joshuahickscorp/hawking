#!/usr/bin/env python3
"""Issue one immutable source-token Qwen80 L0 true-MoE component lease.

This issuer is deliberately file/CPU-only.  It revalidates the already
sealed source-token outer/child CPU preflight against the current admitted
identity, binds the held Qwen80 watcher, and writes one create-new lease for
the outer launcher.  It never creates a Metal context or starts a child.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen80_source_token_true_input_all_ten_moe_graph_launcher as launcher
from lab.receipts import seal


def _lease_id(*, proof: launcher.ProofContext, out: Path) -> str:
    """Derive a fresh opaque ID from the exact proof and create-new target."""
    payload = {
        "schema": launcher.LEASE_SCHEMA,
        "proof": proof.proof_evidence["sha256"],
        "proof_seal": proof.proof_seal_sha256,
        "probe": proof.preflight.probe_binary["sha256"],
        "recorded_at": launcher._utc_now(),
        "out": str(out),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _document(
    *, proof: launcher.ProofContext, watcher_hold: dict[str, Any], out: Path
) -> dict[str, Any]:
    context = proof.preflight
    prefix = launcher._child_sealed_binding(
        context.first_residual, context.first_residual_seal_sha256
    )
    prefix["output_sha256"] = context.first_residual_output_sha256
    return seal(
        {
            "schema": launcher.LEASE_SCHEMA,
            "status": launcher.LEASE_STATUS,
            "recorded_at": launcher._utc_now(),
            "lease_id": _lease_id(proof=proof, out=out),
            "artifact_binding": {
                "manifest_document_sha256": context.manifest["sha256"],
                "manifest_seal_sha256": context.manifest_seal_sha256,
                "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
            },
            "outer_preflight_binding": launcher._child_sealed_binding(
                context.outer_preflight_evidence, context.outer_preflight_seal_sha256
            ),
            "source_token_route_authority_binding": launcher._child_sealed_binding(
                context.source_authority, context.source_authority_seal_sha256
            ),
            "typed_bridge_binding": launcher._child_sealed_binding(
                context.typed_bridge, context.typed_bridge_seal_sha256
            ),
            "first_residual_antecedent": prefix,
            "fixed_suffix_contract_binding": launcher._child_fixed_binding(context.fixed_suffix),
            "child_preflight_proof_binding": launcher._child_sealed_binding(
                proof.proof_evidence, proof.proof_seal_sha256
            ),
            "implementation_binding": {
                "source_token_id": launcher.SOURCE_TOKEN_ID,
                "prefix_dispatches": launcher.PREFIX_DISPATCHES,
                "suffix_dispatches": launcher.SUFFIX_DISPATCHES,
                "total_dispatches": launcher.TOTAL_DISPATCHES,
                "same_command_buffer_fence_required": True,
                "probe_binary": context.probe_binary,
                "registered_all_ten_shader_source": context.shader_source,
                "metal_registry": context.metal_registry,
            },
            "execution_policy": {
                "component": launcher.LEASE_COMPONENT,
                "quiet_qwen80_device_lease": True,
                "strict_math": True,
                "timing_or_benchmarking_allowed": False,
                "complete_layer_or_token_allowed": False,
                "tps_or_tg_claim_allowed": False,
            },
            "lifecycle": {
                "fresh_for_this_exact_launch": True,
                "outer_reaped_capture_required": True,
                "lease_released_after_first_terminal_child": True,
                "automatic_retry_prohibited": True,
            },
            "watcher_coordination": {
                "watcher_hold": watcher_hold,
                "watcher_hold_must_remain_active": True,
                "watcher_restart_or_transition_authorized": False,
            },
            "prelaunch_claim": {
                "issuer_performed_a_fresh_read_only_prelaunch_check": True,
                "qwen80_watcher_hold_remains_active": True,
                "no_server_hcli_tps_or_tournament_authorized": True,
            },
            "claim_boundary": {
                "source_token_l0_true_moe_component_only": True,
                "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": True,
            },
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission-current", type=Path, required=True)
    parser.add_argument("--source-token-route-authority", type=Path, required=True)
    parser.add_argument("--first-residual-receipt", type=Path, required=True)
    parser.add_argument("--typed-bridge-receipt", type=Path, required=True)
    parser.add_argument("--fixed-suffix-contract", type=Path, required=True)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--preflight-proof", type=Path, required=True)
    parser.add_argument("--watcher-hold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.out.is_absolute() or args.out.exists():
            raise launcher.SourceTokenTrueInputAllTenMoeLauncherError(
                "--out must be a new absolute path"
            )
        base = launcher.BaseInputs(
            manifest=args.manifest,
            admission_current=args.admission_current,
            source_token_route_authority=args.source_token_route_authority,
            first_residual_receipt=args.first_residual_receipt,
            typed_bridge_receipt=args.typed_bridge_receipt,
            fixed_suffix_contract=args.fixed_suffix_contract,
        )
        proof = launcher.validate_preflight_proof(
            proof_path=args.preflight_proof, base=base, probe_bin=args.probe_bin
        )
        watcher_hold = launcher._file_evidence(args.watcher_hold, "--watcher-hold")
        document = _document(proof=proof, watcher_hold=watcher_hold, out=args.out)
        launcher._write_new(args.out, document)
        evidence, seal_sha256, lease_id = launcher._bind_lease(args.out, proof)
    except launcher.SourceTokenTrueInputAllTenMoeLauncherError as exc:
        print(
            json.dumps(
                {
                    "status": "REFUSED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_LEASE_ISSUANCE",
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "ISSUED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_COMPONENT_ONLY_QUIET_METAL_LEASE",
                "lease": evidence,
                "lease_id": lease_id,
                "seal_sha256": seal_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
