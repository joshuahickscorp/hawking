#!/usr/bin/env python3.12
"""Fail-closed autoresume for the committed controlled-swap successor."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import doctor_v5_controlled_swap_activation as activation


ROOT = Path(__file__).resolve().parents[2]
ENV_MARKER = "DOCTOR_V5_CONTROLLED_SWAP_MARKER"
ENV_MARKER_SHA256 = "DOCTOR_V5_CONTROLLED_SWAP_MARKER_SHA256"
ENV_POLICY = "DOCTOR_V5_CONTROLLED_SWAP_POLICY"
ENV_POLICY_SHA256 = "DOCTOR_V5_CONTROLLED_SWAP_POLICY_SHA256"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _valid_current_control_state(paths: activation.Paths) \
        -> tuple[dict[str, Any], dict[str, Any]]:
    control = activation._read_json(paths.control)
    state = activation._read_json(paths.state)
    if control.get("schema") != "hawking.doctor_v5_ultra_control.v1" \
            or not activation._sealed(control, "control_sha256"):
        raise activation.ActivationError("current control identity is invalid")
    if state.get("schema") != "hawking.doctor_v5_ultra_queue_state.v1" \
            or not activation._sealed(state, "state_sha256"):
        raise activation.ActivationError("current state identity is invalid")
    if control.get("plan_sha256") != state.get("plan_sha256"):
        raise activation.ActivationError("current control/state plans differ")
    return control, state


def run_once(*, paths: activation.Paths | None = None,
             runner: Callable[..., Any] = subprocess.run) -> int:
    paths = paths or activation.production_paths(ROOT)
    try:
        control, state = _valid_current_control_state(paths)
    except (activation.ActivationError, OSError, ValueError, KeyError):
        return 2
    mode = control.get("mode")
    clean_drained = (mode == "drain" and state.get("status") == "drained"
                     and state.get("active_cells") == []
                     and state.get("active_children") == {})
    if state.get("status") == "complete" or (mode != "run" and not clean_drained):
        return 0
    try:
        marker, packet = activation.validate_active_marker(paths=paths,
                                                            verify_service=True)
        if state.get("plan_sha256") != packet["snapshot"]["plan_sha256"]:
            raise activation.ActivationError("current state left the activated plan")
        policy = activation._read_json(paths.successor_policy)
        if policy.get("policy_sha256") != packet.get("policy_sha256") \
                or marker.get("successor_queue") != activation._artifact(
                    paths.successor_queue):
            raise activation.ActivationError("marker/policy/queue generation is mixed")
    except (activation.ActivationError, OSError, ValueError, KeyError, TypeError):
        return 2
    env = os.environ.copy()
    env[ENV_MARKER] = str(paths.active_marker.resolve())
    env[ENV_MARKER_SHA256] = marker["marker_sha256"]
    env[ENV_POLICY] = str(paths.successor_policy.resolve())
    env[ENV_POLICY_SHA256] = policy["policy_sha256"]
    try:
        result = runner([sys.executable, str(paths.successor_queue), "start"],
                        cwd=str(paths.root), env=env, check=False)
    except OSError:
        return 2
    return int(result.returncode)


def main() -> int:
    return run_once()


if __name__ == "__main__":
    raise SystemExit(main())
