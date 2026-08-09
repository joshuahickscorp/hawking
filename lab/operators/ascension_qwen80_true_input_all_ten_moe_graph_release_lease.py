#!/usr/bin/env python3
"""Seal release of one terminal Qwen80 true-input all-ten component lease."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen80_true_input_all_ten_moe_graph_launcher as launcher
from lab.receipts import seal


RELEASE_SCHEMA = "hawking.ascension.qwen80_true_input_all_ten_moe_graph_quiet_metal_lease_release.v1"
RELEASE_STATUS = (
    "RELEASED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_COMPONENT_QUIET_METAL_LEASE_"
    "AFTER_TERMINAL_CAPTURE"
)
TERMINAL_STATUSES = {
    "CAPTURED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_TERMINAL_COMPONENT_ONLY",
    "REFUSED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_CHILD_SPAWN_ERROR",
    "REFUSED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_CHILD_TIMEOUT",
    "REFUSED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_CHILD_SIGNAL",
    "REFUSED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_CHILD_NONZERO",
    "REFUSED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_ZERO_EXIT_WITHOUT_STRICT_INNER_RECEIPT",
}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise launcher.TrueInputAllTenMoeGraphLauncherError(f"{label} must be an object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--outer-terminal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.out.is_absolute() or args.out.exists():
            raise launcher.TrueInputAllTenMoeGraphLauncherError("--out must be a new absolute path")
        lease, lease_seal = launcher._sealed_json(args.lease, "--lease")
        lease_evidence = launcher._file_evidence(args.lease, "--lease")
        if lease.get("schema") != launcher.LEASE_SCHEMA or lease.get("status") != launcher.LEASE_STATUS:
            raise launcher.TrueInputAllTenMoeGraphLauncherError("--lease is not a Qwen80 all-ten component lease")
        outer, outer_seal = launcher._sealed_json(args.outer_terminal, "--outer-terminal")
        outer_evidence = launcher._file_evidence(args.outer_terminal, "--outer-terminal")
        if outer.get("schema") != launcher.SCHEMA or outer.get("status") not in TERMINAL_STATUSES:
            raise launcher.TrueInputAllTenMoeGraphLauncherError("--outer-terminal is not a terminal all-ten capture")
        source = _mapping(outer.get("source_binding"), "outer source_binding")
        bound_lease = _mapping(source.get("lease_receipt"), "outer lease receipt")
        if (
            bound_lease.get("path") != lease_evidence["path"]
            or bound_lease.get("sha256") != lease_evidence["sha256"]
            or source.get("lease_receipt_seal_sha256") != lease_seal
        ):
            raise launcher.TrueInputAllTenMoeGraphLauncherError(
                "outer terminal is not bound to the exact lease being released"
            )
        release = seal(
            {
                "schema": RELEASE_SCHEMA,
                "status": RELEASE_STATUS,
                "recorded_at": launcher._utc_now(),
                "lease": {**lease_evidence, "seal_sha256": lease_seal},
                "outer_terminal": {
                    **outer_evidence,
                    "seal_sha256": outer_seal,
                    "status": outer.get("status"),
                },
                "coordination": {
                    "quiet_qwen80_component_lease_released": True,
                    "watcher_hold_remains_active": True,
                    "watcher_restart_or_transition_authorized": False,
                    "new_qwen80_gpu_work_requires_a_fresh_explicit_lease": True,
                },
                "claim_boundary": {
                    "release_is_gpu_coordination_only": True,
                    "does_not_promote_component_to_layer_token_decoder_hcli_tps_tg_or_tournament": True,
                },
            }
        )
        launcher._atomic_json_new(args.out, release)
    except launcher.TrueInputAllTenMoeGraphLauncherError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_LEASE_RELEASE", "error": str(exc)}))
        return 2
    print(
        json.dumps(
            {"status": RELEASE_STATUS, "release": str(args.out), "seal_sha256": release["seal_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
