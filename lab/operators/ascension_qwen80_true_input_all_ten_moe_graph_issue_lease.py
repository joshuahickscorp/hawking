#!/usr/bin/env python3
"""Issue one immutable Qwen80 L0 true-input all-ten MoE component lease.

The paired outer launcher owns process creation and reaping.  This issuer is
CPU/filesystem-only: it binds the exact registered probe, current admitted
artifact, sealed router/prefix/typed-bridge authorities, unsealed fixed ABI,
and the watcher coordination hold before writing one create-new lease.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen80_true_input_all_ten_moe_graph_launcher as launcher
from lab.receipts import seal


@dataclass(frozen=True)
class LeaseContext:
    probe: dict[str, Any]
    shader: dict[str, Any]
    watcher_hold: dict[str, Any]
    manifest: dict[str, Any]
    manifest_seal_sha256: str
    admission: dict[str, Any]
    admission_receipt_seal_sha256: str
    router: dict[str, Any]
    router_outer: dict[str, Any]
    router_outer_seal_sha256: str
    route_plan: dict[str, Any]
    first_residual: dict[str, Any]
    first_residual_seal_sha256: str
    first_residual_output_sha256: str
    typed_bridge: dict[str, Any]
    typed_bridge_seal_sha256: str
    fixed_abi: dict[str, Any]


def _context(args: argparse.Namespace) -> LeaseContext:
    probe = launcher._canonical_regular(args.probe_bin, "--probe-bin", executable=True)
    if probe.name != launcher.EXPECTED_PROBE_BASENAME:
        raise launcher.TrueInputAllTenMoeGraphLauncherError(
            f"--probe-bin must name {launcher.EXPECTED_PROBE_BASENAME}, got {probe.name!r}"
        )
    shader = launcher._canonical_regular(args.shader, "--shader")
    if shader.name != "qwen80_all_ten_routed_expert_wave.metal":
        raise launcher.TrueInputAllTenMoeGraphLauncherError(
            "--shader must be the isolated Qwen80 all-ten routed-expert source"
        )
    watcher_hold = launcher._canonical_regular(args.watcher_hold, "--watcher-hold")
    manifest, manifest_seal = launcher._bind_manifest(args.manifest)
    admission, _pointer_seal, admission_receipt_seal = launcher._bind_admission(
        args.admission_current, manifest, manifest_seal
    )
    router, router_outer, router_outer_seal = launcher._bind_router(
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission=admission,
        admission_receipt_seal=admission_receipt_seal,
        router_path=args.router_receipt,
        router_outer_path=args.router_outer_receipt,
    )
    route_plan = launcher._file_evidence(args.route_plan, "--route-plan")
    launcher._route_ids_and_weights(launcher._read_json(args.route_plan, "--route-plan"))
    first_residual, first_residual_seal, first_residual_output = launcher._bind_first_residual(
        args.first_residual_receipt,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission=admission,
        admission_receipt_seal=admission_receipt_seal,
    )
    typed_bridge, typed_bridge_seal = launcher._bind_typed_bridge(
        args.typed_bridge_receipt,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission=admission,
        admission_receipt_seal=admission_receipt_seal,
        route_plan=route_plan,
        first_residual=first_residual,
        first_residual_seal=first_residual_seal,
        first_residual_output_sha256=first_residual_output,
    )
    fixed_abi = launcher._bind_fixed_abi_contract(
        args.fixed_abi_contract,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_receipt_seal=admission_receipt_seal,
    )
    return LeaseContext(
        probe=launcher._file_evidence(probe, "--probe-bin", executable=True),
        shader=launcher._file_evidence(shader, "--shader"),
        watcher_hold=launcher._file_evidence(watcher_hold, "--watcher-hold"),
        manifest=manifest,
        manifest_seal_sha256=manifest_seal,
        admission=admission,
        admission_receipt_seal_sha256=admission_receipt_seal,
        router=router,
        router_outer=router_outer,
        router_outer_seal_sha256=router_outer_seal,
        route_plan=route_plan,
        first_residual=first_residual,
        first_residual_seal_sha256=first_residual_seal,
        first_residual_output_sha256=first_residual_output,
        typed_bridge=typed_bridge,
        typed_bridge_seal_sha256=typed_bridge_seal,
        fixed_abi=fixed_abi,
    )


def _lease_document(context: LeaseContext) -> dict[str, Any]:
    lease_material = {
        "component": launcher.LEASE_COMPONENT,
        "probe": context.probe["sha256"],
        "shader": context.shader["sha256"],
        "manifest": context.manifest["sha256"],
        "admission_receipt": context.admission_receipt_seal_sha256,
        "router": context.router["sha256"],
        "route_plan": context.route_plan["sha256"],
        "first_residual": context.first_residual["sha256"],
        "typed_bridge": context.typed_bridge["sha256"],
        "fixed_abi": context.fixed_abi["sha256"],
        "watcher_hold": context.watcher_hold["sha256"],
    }
    lease_id = hashlib.sha256(
        json.dumps(lease_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return seal(
        {
            "schema": launcher.LEASE_SCHEMA,
            "status": launcher.LEASE_STATUS,
            "recorded_at": launcher._utc_now(),
            "lease_id": lease_id,
            "lifecycle": {
                "fresh_for_this_exact_launch": True,
                "automatic_retry_prohibited": True,
                "outer_reaped_capture_required": True,
                "lease_released_after_first_terminal_child": True,
            },
            "execution_policy": {
                "component": launcher.LEASE_COMPONENT,
                "quiet_qwen80_device_lease": True,
                "strict_math": True,
                "timing_or_benchmarking_allowed": False,
                "complete_layer_or_token_allowed": False,
                "tps_or_tg_claim_allowed": False,
            },
            "artifact_binding": {
                "manifest_document_sha256": context.manifest["sha256"],
                "manifest_seal_sha256": context.manifest_seal_sha256,
                "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
            },
            "upstream_authority": {
                "router_receipt": context.router,
                "router_outer_receipt": {
                    **context.router_outer,
                    "seal_sha256": context.router_outer_seal_sha256,
                },
                "route_plan": context.route_plan,
                "first_residual_receipt": {
                    **context.first_residual,
                    "seal_sha256": context.first_residual_seal_sha256,
                    "output_sha256": context.first_residual_output_sha256,
                },
            },
            "typed_bridge_binding": {
                "path": context.typed_bridge["path"],
                "document_sha256": context.typed_bridge["sha256"],
                "schema": launcher.BRIDGE_SCHEMA,
                "status": launcher.BRIDGE_STATUS,
                "seal_sha256": context.typed_bridge_seal_sha256,
            },
            "fixed_abi_contract_binding": {
                "path": context.fixed_abi["path"],
                "document_sha256": context.fixed_abi["sha256"],
                "schema": launcher.FIXED_ABI_SCHEMA,
                "status": launcher.FIXED_ABI_STATUS,
            },
            "implementation_binding": {
                "probe_binary": context.probe,
                "registered_shader": context.shader,
                "prefix_dispatches": 9,
                "suffix_dispatches": 14,
                "total_dispatches": 23,
                "same_command_buffer_fence_required": True,
            },
            "watcher_coordination": {
                "hold": context.watcher_hold,
                "watcher_hold_must_remain_active": True,
                "watcher_restart_or_transition_authorized": False,
            },
            "claim_boundary": {
                "source_input_layer0_all_ten_true_moe_component_only": True,
                "not_a_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament": True,
            },
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--shader", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission-current", type=Path, required=True)
    parser.add_argument("--router-receipt", type=Path, required=True)
    parser.add_argument("--router-outer-receipt", type=Path, required=True)
    parser.add_argument("--route-plan", type=Path, required=True)
    parser.add_argument("--first-residual-receipt", type=Path, required=True)
    parser.add_argument("--typed-bridge-receipt", type=Path, required=True)
    parser.add_argument("--fixed-abi-contract", type=Path, required=True)
    parser.add_argument("--watcher-hold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.out.is_absolute() or args.out.exists():
            raise launcher.TrueInputAllTenMoeGraphLauncherError("--out must be a new absolute path")
        context = _context(args)
        launcher._atomic_json_new(args.out, _lease_document(context))
        lease_evidence, lease_seal = launcher._bind_lease(
            args.out,
            manifest=context.manifest,
            manifest_seal=context.manifest_seal_sha256,
            admission_receipt_seal=context.admission_receipt_seal_sha256,
            typed_bridge=context.typed_bridge,
            typed_bridge_seal=context.typed_bridge_seal_sha256,
            fixed_abi_contract=context.fixed_abi,
        )
    except launcher.TrueInputAllTenMoeGraphLauncherError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_LEASE_ISSUANCE", "error": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "status": "ISSUED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_COMPONENT_ONLY_QUIET_METAL_LEASE",
                "lease": lease_evidence,
                "seal_sha256": lease_seal,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
