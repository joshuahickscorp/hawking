#!/usr/bin/env python3
"""Seal release of one terminal Qwen80 source-token L0 handoff lease.

This is deliberately a file-only coordination endpoint.  It accepts either a
sealed pass or refusal from the one-shot outer reaper, binds it to the exact
sealed lease and release recommendation, then emits one create-new release
receipt.  It never invokes Metal, starts a process, or changes watcher state.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen80_source_token_l0_state_handoff_issue_lease as issuer
from lab.operators import ascension_qwen80_source_token_l0_state_handoff_launcher as launcher
from lab.operators import ascension_qwen80_source_token_l0_state_handoff_outer_capture as outer
from lab.receipts import seal


RELEASE_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_quiet_metal_lease_release.v1"
RELEASE_STATUS = (
    "RELEASED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_COMPONENT_QUIET_METAL_LEASE_"
    "AFTER_TERMINAL_CAPTURE"
)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise launcher.SourceTokenL0StateHandoffLauncherError(f"{label} must be an object")
    return dict(value)


def _exact(value: object, expected: Mapping[str, Any], label: str) -> None:
    if _mapping(value, label) != dict(expected):
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            f"{label} is not bound to the exact expected evidence"
        )


def _terminal_status(value: object) -> bool:
    return value == outer.CAPTURED_STATUS or (
        isinstance(value, str) and value.startswith(outer.REFUSED_PREFIX)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--outer-terminal", type=Path, required=True)
    parser.add_argument("--recommended-release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.out.is_absolute() or args.out.exists():
            raise launcher.SourceTokenL0StateHandoffLauncherError(
                "--out must be a new absolute path"
            )
        lease, lease_seal = launcher._sealed_json(args.lease, "--lease")
        lease_evidence = launcher._evidence(args.lease, "--lease")
        if (
            lease.get("schema") != issuer.LEASE_SCHEMA
            or lease.get("status") != issuer.LEASE_STATUS
        ):
            raise launcher.SourceTokenL0StateHandoffLauncherError(
                "--lease is not the source-token L0 handoff lease"
            )
        lease_id = lease.get("lease_id")
        if not launcher._is_sha(lease_id):
            raise launcher.SourceTokenL0StateHandoffLauncherError("--lease has an invalid lease_id")
        lease_binding = launcher._binding(lease_evidence, lease_seal)

        terminal, terminal_seal = launcher._sealed_json(
            args.outer_terminal, "--outer-terminal"
        )
        terminal_evidence = launcher._evidence(args.outer_terminal, "--outer-terminal")
        if terminal.get("schema") != outer.SCHEMA or not _terminal_status(terminal.get("status")):
            raise launcher.SourceTokenL0StateHandoffLauncherError(
                "--outer-terminal is not a terminal L0 handoff capture"
            )
        if terminal.get("lease_id") != lease_id:
            raise launcher.SourceTokenL0StateHandoffLauncherError(
                "outer terminal lease_id does not match --lease"
            )
        source_binding = _mapping(terminal.get("source_binding"), "outer source_binding")
        _exact(source_binding.get("lease_receipt"), lease_binding, "outer lease binding")

        recommendation, recommendation_seal = launcher._sealed_json(
            args.recommended_release, "--recommended-release"
        )
        recommendation_evidence = launcher._evidence(
            args.recommended_release, "--recommended-release"
        )
        if (
            recommendation.get("schema") != outer.RELEASE_CONTRACT_SCHEMA
            or recommendation.get("status") != outer.RELEASE_CONTRACT_STATUS
        ):
            raise launcher.SourceTokenL0StateHandoffLauncherError(
                "--recommended-release schema/status drifted"
            )
        expected_recommendation = launcher._binding(
            recommendation_evidence, recommendation_seal
        )
        _exact(
            terminal.get("recommended_release_contract"),
            expected_recommendation,
            "outer recommended-release binding",
        )
        expected_recommendation_lease = {**lease_binding, "lease_id": lease_id}
        _exact(
            recommendation.get("lease"),
            expected_recommendation_lease,
            "recommended-release lease binding",
        )
        if recommendation.get("outer_terminal_path") != str(
            args.outer_terminal.resolve(strict=True)
        ):
            raise launcher.SourceTokenL0StateHandoffLauncherError(
                "recommended-release outer terminal path drifted"
            )

        release = seal(
            {
                "schema": RELEASE_SCHEMA,
                "status": RELEASE_STATUS,
                "recorded_at": launcher._utc_now(),
                "lease": {**lease_binding, "lease_id": lease_id},
                "outer_terminal": {
                    **terminal_evidence,
                    "seal_sha256": terminal_seal,
                    "status": terminal.get("status"),
                },
                "recommended_release_contract": expected_recommendation,
                "coordination": {
                    "quiet_qwen80_component_lease_released": True,
                    "watcher_hold_remains_active": True,
                    "watcher_restart_or_transition_authorized": False,
                    "new_qwen80_gpu_work_requires_a_fresh_explicit_lease": True,
                    "automatic_retry_prohibited": True,
                },
                "claim_boundary": {
                    "release_is_gpu_coordination_only": True,
                    "does_not_promote_component_to_layer_token_decoder_hcli_tps_tg_or_tournament": True,
                    "outer_terminal_refusal_is_preserved": terminal.get("status").startswith(
                        outer.REFUSED_PREFIX
                    ),
                },
            }
        )
        launcher._write_new(args.out, release)
    except launcher.SourceTokenL0StateHandoffLauncherError as exc:
        print(
            json.dumps(
                {
                    "status": "REFUSED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_LEASE_RELEASE",
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": RELEASE_STATUS,
                "release": str(args.out),
                "seal_sha256": release["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
