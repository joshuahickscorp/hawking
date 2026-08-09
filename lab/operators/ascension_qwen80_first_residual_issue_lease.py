#!/usr/bin/env python3
"""Issue one immutable prefix-only Qwen80 first-residual Metal lease.

The lease binds the currently selected admitted artifact, the retained CPU
baseline, exact probe binary, and the watcher coordination hold.  It does not
start a process or change the watcher; the paired outer launcher owns exactly
one child and reaps it.
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

from lab.operators import ascension_qwen80_first_residual_bridge_launcher as launcher
from lab.receipts import seal


def _context(args: argparse.Namespace) -> dict[str, Any]:
    probe = launcher._canonical_regular(args.probe_bin, "--probe-bin", executable=True)
    if probe.name != launcher.EXPECTED_PROBE_BASENAME:
        raise launcher.FirstResidualBridgeLauncherError(
            f"--probe-bin must name {launcher.EXPECTED_PROBE_BASENAME}"
        )
    manifest, manifest_seal = launcher._bind_manifest(args.manifest)
    admission, _pointer_seal, admission_receipt_seal, _audit, _revision = launcher._bind_admission(
        args.admission_current, manifest, manifest_seal
    )
    baseline, _input, _output = launcher._bind_cpu_baseline(
        args.cpu_baseline_receipt,
        manifest=manifest,
        manifest_seal=manifest_seal,
        source_audit_seal=_audit,
        source_revision=_revision,
    )
    watcher_hold = launcher._file_evidence(
        launcher._canonical_regular(args.watcher_hold, "--watcher-hold"), "--watcher-hold"
    )
    return {
        "probe": launcher._file_evidence(probe, "--probe-bin", executable=True),
        "manifest": manifest,
        "manifest_seal": manifest_seal,
        "admission": admission,
        "admission_receipt_seal": admission_receipt_seal,
        "baseline": baseline,
        "watcher_hold": watcher_hold,
    }


def _lease_document(context: dict[str, Any]) -> dict[str, Any]:
    lease_id = hashlib.sha256(
        json.dumps(
            {
                "component": launcher.LEASE_COMPONENT,
                "manifest": context["manifest"]["sha256"],
                "admission": context["admission_receipt_seal"],
                "baseline": context["baseline"]["sha256"],
                "probe": context["probe"]["sha256"],
                "watcher_hold": context["watcher_hold"]["sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
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
                "manifest_document_sha256": context["manifest"]["sha256"],
                "manifest_seal_sha256": context["manifest_seal"],
                "admission_receipt_seal_sha256": context["admission_receipt_seal"],
            },
            "cpu_baseline_binding": {
                "receipt_path": context["baseline"]["path"],
                "receipt_document_sha256": context["baseline"]["sha256"],
                "schema": launcher.CPU_BASELINE_SCHEMA,
                "status": launcher.CPU_BASELINE_STATUS,
            },
            "implementation_binding": {
                "probe_binary": context["probe"],
                "prefix_dispatches": 9,
                "same_command_buffer_fence_required": True,
                "registered_qwen_next_prefix_kernels_only": True,
            },
            "watcher_coordination": {
                "watcher_gpu_hold": context["watcher_hold"],
                "watcher_hold_must_remain_active": True,
                "watcher_restart_or_transition_authorized": False,
            },
            "claim_boundary": {
                "source_input_l0_deltanet_prefix_component_only": True,
                "no_true_moe_suffix_or_complete_layer": True,
                "not_token_decoder_generation_hcli_tps_tg_or_tournament": True,
            },
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission-current", type=Path, required=True)
    parser.add_argument("--cpu-baseline-receipt", type=Path, required=True)
    parser.add_argument("--watcher-hold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.out.is_absolute() or args.out.exists():
            raise launcher.FirstResidualBridgeLauncherError(
                "--out must be a new absolute path"
            )
        context = _context(args)
        launcher._atomic_json_new(args.out, _lease_document(context))
        lease, lease_seal = launcher._bind_lease(
            args.out,
            manifest=context["manifest"],
            manifest_seal=context["manifest_seal"],
            admission_receipt_seal=context["admission_receipt_seal"],
            cpu_baseline=context["baseline"],
        )
    except launcher.FirstResidualBridgeLauncherError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_FIRST_RESIDUAL_LEASE_ISSUANCE", "error": str(exc)}))
        return 2
    print(json.dumps({"status": "ISSUED_QWEN80_FIRST_RESIDUAL_PREFIX_ONLY_QUIET_METAL_LEASE", "lease": lease, "seal_sha256": lease_seal}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
