#!/usr/bin/env python3
"""Issue one immutable, component-only Qwen80 MoE-combine Metal lease.

This is deliberately not a runtime or watcher control.  It binds the current
admitted complete artifact, sealed router capture, sealed CPU baseline, exact
probe/shader bytes, and the existing watcher coordination hold before sealing
one lease file.  The existing outer launcher then remains responsible for the
only child process, reaping, and terminal capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen80_moe_combine_probe_launcher as launcher
from lab.receipts import seal


def _context(args: argparse.Namespace) -> launcher.LaunchContext:
    """Reuse the launcher's strict prerequisite validators before issuance."""

    probe = launcher._canonical_regular(args.probe_bin, "--probe-bin", executable=True)
    if probe.name != launcher.EXPECTED_PROBE_BASENAME:
        raise launcher.MoeCombineProbeLauncherError(
            f"--probe-bin must name {launcher.EXPECTED_PROBE_BASENAME}, got {probe.name!r}"
        )
    manifest, manifest_seal = launcher._bind_manifest(args.manifest)
    admission, pointer_seal, admission_receipt_seal = launcher._bind_current_admission(
        args.admission_current,
        manifest,
        manifest_seal,
    )
    router, router_outer, router_outer_seal, historical_admission = (
        launcher._bind_upstream_router(
            manifest=manifest,
            manifest_seal_sha256=manifest_seal,
            admission_current=admission,
            admission_receipt_seal=admission_receipt_seal,
            router_receipt_path=args.router_receipt,
            router_outer_path=args.router_outer_receipt,
        )
    )
    provisional = launcher.LaunchContext(
        probe_binary=launcher._file_evidence(probe, "--probe-bin"),
        manifest=manifest,
        manifest_seal_sha256=manifest_seal,
        admission_current=admission,
        admission_pointer_seal_sha256=pointer_seal,
        admission_receipt_seal_sha256=admission_receipt_seal,
        router_receipt=router,
        router_outer_receipt=router_outer,
        router_outer_seal_sha256=router_outer_seal,
        router_outer_historical_admission_pointer=historical_admission,
        cpu_baseline_receipt={},
        cpu_baseline_seal_sha256="",
        cpu_inner_receipt={},
        lease_receipt={},
        lease_seal_sha256="",
    )
    baseline, baseline_seal, cpu_inner = launcher._bind_cpu_baseline(
        args.cpu_baseline_receipt,
        context_without_baseline=provisional,
    )
    return replace(
        provisional,
        cpu_baseline_receipt=baseline,
        cpu_baseline_seal_sha256=baseline_seal,
        cpu_inner_receipt=cpu_inner,
    )


def _lease_document(args: argparse.Namespace, context: launcher.LaunchContext) -> dict[str, Any]:
    shader = launcher._file_evidence(
        launcher._canonical_regular(args.shader, "--shader"), "--shader"
    )
    watcher_hold = launcher._file_evidence(
        launcher._canonical_regular(args.watcher_hold, "--watcher-hold"), "--watcher-hold"
    )
    source_top10 = {
        "router_receipt_path": context.router_receipt["path"],
        "router_receipt_sha256": context.router_receipt["sha256"],
        "router_outer_receipt_path": context.router_outer_receipt["path"],
        "router_outer_receipt_sha256": context.router_outer_receipt["sha256"],
        "router_outer_receipt_seal_sha256": context.router_outer_seal_sha256,
        "ids": list(launcher.SOURCE_TOP10_IDS),
    }
    lease_id_material = {
        "component": launcher.MOE_COMBINE_LEASE_COMPONENT,
        "manifest": context.manifest["sha256"],
        "admission_receipt": context.admission_receipt_seal_sha256,
        "router": context.router_receipt["sha256"],
        "baseline": context.cpu_baseline_receipt["sha256"],
        "probe": context.probe_binary["sha256"],
        "shader": shader["sha256"],
    }
    lease_id = hashlib.sha256(
        json.dumps(lease_id_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return seal(
        {
            "schema": launcher.MOE_COMBINE_LEASE_SCHEMA,
            "status": launcher.MOE_COMBINE_LEASE_STATUS,
            "recorded_at": launcher._utc_now(),
            "lease_id": lease_id,
            "model": {
                "id": "Qwen3-Coder-Next-80B",
                "key": "qwen80",
                "repository": "Qwen/Qwen3-Coder-Next",
                "revision": "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
            },
            "execution_policy": {
                "component": launcher.MOE_COMBINE_LEASE_COMPONENT,
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
            "source_top10_binding": source_top10,
            "cpu_baseline_binding": {
                "receipt_path": context.cpu_baseline_receipt["path"],
                "receipt_document_sha256": context.cpu_baseline_receipt["sha256"],
                "schema": launcher.CPU_BASELINE_WRAPPER_SCHEMA,
                "status": launcher.CPU_BASELINE_WRAPPER_STATUS,
                "seal_sha256": context.cpu_baseline_seal_sha256,
            },
            "implementation_binding": {
                "probe_binary": context.probe_binary,
                "shader": shader,
                "kernel_sequence": [
                    "qwen80_moe_wave_aggregate_second_residual_route_sum",
                    "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
                ],
            },
            "watcher_coordination": {
                "hold": watcher_hold,
                "watcher_hold_must_remain_active": True,
                "watcher_restart_or_transition_authorized": False,
            },
            "one_shot": {
                "automatic_retry_prohibited": True,
                "outer_reaped_capture_required": True,
                "lease_released_after_first_terminal_child": True,
            },
            "claim_boundary": {
                "materialized_source_top10_vectors_only": True,
                "not_ten_physical_expert_waves_or_complete_layer": True,
                "not_token_decoder_generation_hcli_tps_tg_or_tournament": True,
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
    parser.add_argument("--cpu-baseline-receipt", type=Path, required=True)
    parser.add_argument("--watcher-hold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.out.is_absolute():
            raise launcher.MoeCombineProbeLauncherError("--out must be an absolute path")
        if args.out.exists():
            raise launcher.MoeCombineProbeLauncherError(
                f"refusing to overwrite existing lease {args.out}"
            )
        context = _context(args)
        document = _lease_document(args, context)
        launcher._atomic_json_new(args.out, document)
        # Validate the exact persisted bytes through the same contract that
        # guards the outer launcher before reporting a usable lease.
        lease_evidence, lease_seal = launcher._bind_lease(
            args.out,
            context_without_lease=context,
        )
    except launcher.MoeCombineProbeLauncherError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_MOE_COMBINE_LEASE_ISSUANCE", "error": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "status": "ISSUED_QWEN80_MOE_COMBINE_COMPONENT_ONLY_QUIET_METAL_LEASE",
                "lease": lease_evidence,
                "seal_sha256": lease_seal,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
