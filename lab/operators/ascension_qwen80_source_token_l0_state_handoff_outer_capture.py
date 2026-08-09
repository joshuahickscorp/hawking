#!/usr/bin/env python3
"""One-shot, receipt-last outer reaper for the authorized Qwen80 L0 handoff.

This is deliberately the only Metal-mode caller in this small boundary.  It
uses the normalized launcher to reserve a fresh replay-guarded capture, starts
exactly one leased child, reaps it, stores bounded child streams, records the
inner receipt faithfully, and writes the outer terminal receipt last.  It
never retries, releases a lease, starts a watcher/server, or promotes the
component capture into L1 execution, a layer, a token, TPS, TG, or tournament
result.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen80_source_token_l0_state_handoff_issue_lease as issuer
from lab.operators import ascension_qwen80_source_token_l0_state_handoff_launcher as launcher
from lab.receipts import seal


SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_outer_capture.v1"
CAPTURED_STATUS = (
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_TERMINAL_"
    "PRE_L1_COMPONENT_ONLY"
)
REFUSED_PREFIX = "REFUSED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_"
RELEASE_CONTRACT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_state_handoff_"
    "recommended_lease_release_contract.v1"
)
RELEASE_CONTRACT_STATUS = (
    "RECOMMENDED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_"
    "LEASE_RELEASE_AFTER_OUTER_TERMINAL"
)

# ``active.json`` belongs to the normalized planning launcher.  Preserve it as
# the immutable reservation record and add a distinct running observation.
RUNNING_FILENAME = "outer-running.json"
CHILD_FILENAME = "child.json"
TERMINAL_FILENAME = launcher.TERMINAL_RECEIPT_FILENAME
OUTER_LAUNCH_AUTHORITY_FILENAME = launcher.OUTER_LAUNCH_AUTHORITY_FILENAME
INNER_CAPTURE_DIRNAME = launcher.INNER_CAPTURE_DIRNAME
OUTER_STDOUT_FILENAME = "outer-child.stdout.log"
OUTER_STDERR_FILENAME = "outer-child.stderr.log"
MAX_CHILD_STREAM_BYTES = 1_000_000


@dataclass(frozen=True)
class CaptureConfig:
    preflight_proof: Path
    child_preflight: Path
    handoff_authority: Path
    probe_bin: Path
    watcher_hold: Path
    lease_receipt: Path
    capture_dir: Path
    replay_guard_dir: Path
    recommended_release_out: Path
    workers: int = 1
    timeout_seconds: float = 7200.0


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise launcher.SourceTokenL0StateHandoffLauncherError(f"{label} must be an object")
    return dict(value)


def _exact(value: object, expected: Mapping[str, Any], label: str) -> None:
    if _mapping(value, label) != dict(expected):
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            f"{label} drifted from exact evidence"
        )


def _binding(evidence: Mapping[str, Any], seal_sha256: str) -> dict[str, Any]:
    return launcher._binding(evidence, seal_sha256)


def _child_binding(evidence: Mapping[str, Any], seal_sha256: str) -> dict[str, Any]:
    return {
        "path": evidence["path"],
        "document_sha256": evidence["sha256"],
        "seal_sha256": seal_sha256,
    }


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise launcher.SourceTokenL0StateHandoffLauncherError(f"{label} must be non-empty")
    return value


def _terminal(returncode: int | None, *, timed_out: bool, spawn_error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reaped": returncode is not None,
        "timed_out": timed_out,
        "returncode": returncode,
        "exit_code": returncode if isinstance(returncode, int) and returncode >= 0 else None,
        "signal": -returncode if isinstance(returncode, int) and returncode < 0 else None,
    }
    if spawn_error is not None:
        result["spawn_error"] = spawn_error
        result["reaped"] = False
    return result


def _terminate_group(child: subprocess.Popen[bytes]) -> int | None:
    if child.poll() is not None:
        return child.returncode
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        return child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        return child.wait(timeout=10)


def _bounded_evidence(path: Path, label: str) -> dict[str, Any]:
    evidence = launcher._evidence(path, label)
    if evidence["bytes"] > MAX_CHILD_STREAM_BYTES:
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            f"{label} exceeds the {MAX_CHILD_STREAM_BYTES}-byte stream limit"
        )
    return evidence


def _launch_identity(config: CaptureConfig, context: issuer.LeaseContext) -> str:
    payload = {
        "schema": SCHEMA,
        "proof": _binding(context.proof.proof_evidence, context.proof.proof_seal_sha256),
        "lease": _binding(context.lease_evidence, context.lease_seal_sha256),
        "lease_id": context.lease_id,
        "watcher_hold": context.watcher_hold_evidence,
        "probe_binary": context.proof.probe_binary,
        "capture_dir": str(config.capture_dir),
        "workers": config.workers,
    }
    return launcher._sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _validate_config(config: CaptureConfig) -> issuer.LeaseContext:
    if isinstance(config.workers, bool) or not isinstance(config.workers, int) or not 1 <= config.workers <= 4:
        raise launcher.SourceTokenL0StateHandoffLauncherError("--workers must be 1..4")
    if not 1.0 <= config.timeout_seconds <= 7200.0:
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "--timeout-seconds must be between 1 and 7200"
        )
    if (
        not config.capture_dir.is_absolute()
        or not config.replay_guard_dir.is_absolute()
        or not config.recommended_release_out.is_absolute()
    ):
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "capture, replay-guard, and recommended release paths must be absolute"
        )
    if config.capture_dir == config.capture_dir.parent or config.capture_dir == REPO_ROOT:
        raise launcher.SourceTokenL0StateHandoffLauncherError("--capture-dir is too broad")
    return issuer.validate_lease(
        lease_receipt=config.lease_receipt,
        preflight_proof=config.preflight_proof,
        child_preflight=config.child_preflight,
        handoff_authority=config.handoff_authority,
        probe_bin=config.probe_bin,
        watcher_hold=config.watcher_hold,
    )


def _child_command(
    config: CaptureConfig,
    context: issuer.LeaseContext,
    *,
    capture: Path,
) -> list[str]:
    return [
        str(context.proof.probe_binary["path"]),
        "--outer-preflight",
        str(context.proof.outer_preflight_evidence["path"]),
        "--mode",
        "metal",
        "--lease-receipt",
        str(context.lease_evidence["path"]),
        "--outer-launch-authority",
        str(capture / OUTER_LAUNCH_AUTHORITY_FILENAME),
        "--outer-capture-dir",
        str(capture),
        "--capture-dir",
        str(capture / INNER_CAPTURE_DIRNAME),
        "--workers",
        str(config.workers),
    ]


def _state_entry(
    value: object,
    expected: Mapping[str, Any],
    label: str,
    *,
    hash_field: str,
) -> str:
    state = _mapping(value, label)
    for field in ("allocation_id", "slot", "offset_bytes", "capacity_bytes"):
        if state.get(field) != expected.get(field):
            raise launcher.SourceTokenL0StateHandoffLauncherError(
                f"{label}.{field} drifted from the static state layout"
            )
    if not _is_sha(state.get(hash_field)):
        raise launcher.SourceTokenL0StateHandoffLauncherError(f"{label} hash drifted")
    return _nonempty_string(state.get("device_buffer_id"), f"{label}.device_buffer_id")


def _validate_inner_receipt(
    document: Mapping[str, Any],
    *,
    context: issuer.LeaseContext,
    outer_authority: Mapping[str, Any],
) -> None:
    """Accept only the sealed L0 post-state/pre-L1 binding result."""
    if (
        document.get("schema") != launcher.PRE_L1_CAPTURE_SCHEMA
        or document.get("status") != launcher.PRE_L1_CAPTURE_STATUS
        or document.get("mode") != "metal"
        or document.get("metal_device_or_dispatch_performed") is not True
        or document.get("component_only") is not True
        or document.get("l1_binding_not_executed") is not True
        or document.get("l1_prefix_dispatches") != 0
        or document.get("complete_layer_or_token_performed") is not False
    ):
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "inner capture schema/status/scope drifted"
        )
    preflight = context.proof
    _exact(
        document.get("outer_preflight_binding"),
        _child_binding(
            preflight.outer_preflight_evidence, preflight.outer_preflight_seal_sha256
        ),
        "inner outer preflight binding",
    )
    outer_seal = outer_authority.get("seal_sha256")
    if not _is_sha(outer_seal):
        raise launcher.SourceTokenL0StateHandoffLauncherError("outer launch authority seal is invalid")
    authority_evidence = launcher._evidence(
        Path(str(outer_authority["planned_outer_capture_dir"]))
        / OUTER_LAUNCH_AUTHORITY_FILENAME,
        "outer launch authority",
    )
    _exact(
        document.get("outer_launch_authority_binding"),
        _child_binding(authority_evidence, str(outer_seal)),
        "inner outer launch authority binding",
    )
    graph = _mapping(document.get("same_command_graph"), "inner same command graph")
    for field, expected in {
        "source_token_id": 1,
        "prefix_dispatches": 9,
        "suffix_dispatches": 14,
        "total_dispatches": 23,
        "same_command_graph_retained": True,
        "fenced_once_after_prefix_and_suffix": True,
    }.items():
        if graph.get(field) != expected:
            raise launcher.SourceTokenL0StateHandoffLauncherError(
                f"inner same command graph {field} drifted"
            )
    handoff = _mapping(document.get("l0_state_handoff"), "inner L0 state handoff")
    if (
        handoff.get("schema") != launcher.PRE_L1_CAPTURE_SCHEMA
        or handoff.get("status") != launcher.PRE_L1_CAPTURE_STATUS
        or handoff.get("source_token_id") != 1
        or handoff.get("same_command_graph_retained") is not True
        or handoff.get("l1_binding_not_executed") is not True
        or handoff.get("l1_prefix_dispatches") != 0
    ):
        raise launcher.SourceTokenL0StateHandoffLauncherError("inner L0 handoff scope drifted")
    retained = _mapping(
        handoff.get("retained_l0_second_residual"), "inner retained L0 output"
    )
    expected_second = _mapping(
        _mapping(
            context.proof.child.handoff.authority.get("consumed_component_capture"),
            "baseline consumed component",
        ).get("second_residual"),
        "baseline second residual",
    )
    if (
        retained.get("elements") != 2048
        or retained.get("bytes") != 8192
        or retained.get("f32le_sha256") != expected_second.get("f32le_sha256")
        or retained.get("retained_for_future_layer1_encode") is not True
    ):
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "inner retained L0 output identity drifted"
        )
    retained_id = _nonempty_string(
        retained.get("device_buffer_id"), "inner retained L0 device buffer"
    )
    layout = _mapping(
        context.proof.child.handoff.authority.get("static_state_layout_authority"),
        "baseline static state layout",
    )
    l0_layout = _mapping(layout.get("l0"), "baseline L0 static state layout")
    l1_layout = _mapping(layout.get("l1"), "baseline L1 static state layout")
    if layout.get("l0_and_l1_slots_verified_disjoint") is not True:
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "baseline state layout does not prove L0/L1 slot disjointness"
        )
    state = _mapping(handoff.get("l0_post_state_commit"), "inner L0 post-state")
    if state.get("layer") != 0 or state.get("linear_state_slot") != 0 or state.get("checkpoint_before_mutation") is not True:
        raise launcher.SourceTokenL0StateHandoffLauncherError("inner L0 post-state scope drifted")
    active_conv = _state_entry(
        state.get("active_conv"),
        _mapping(l0_layout.get("active_conv"), "baseline L0 active conv"),
        "inner L0 active conv",
        hash_field="post_state_f32le_sha256",
    )
    active_recurrent = _state_entry(
        state.get("active_recurrent"),
        _mapping(l0_layout.get("active_recurrent"), "baseline L0 active recurrent"),
        "inner L0 active recurrent",
        hash_field="post_state_f32le_sha256",
    )
    rollback_conv = _state_entry(
        state.get("rollback_conv"),
        _mapping(l0_layout.get("rollback_conv"), "baseline L0 rollback conv"),
        "inner L0 rollback conv",
        hash_field="checkpoint_f32le_sha256",
    )
    rollback_recurrent = _state_entry(
        state.get("rollback_recurrent"),
        _mapping(l0_layout.get("rollback_recurrent"), "baseline L0 rollback recurrent"),
        "inner L0 rollback recurrent",
        hash_field="checkpoint_f32le_sha256",
    )
    if len({active_conv, active_recurrent, rollback_conv, rollback_recurrent}) != 4:
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "inner L0 active/rollback state identities alias"
        )
    layer1 = _mapping(handoff.get("layer1_input_binding"), "inner L1 binding")
    if (
        layer1.get("layer") != 1
        or layer1.get("linear_state_slot") != 1
        or layer1.get("input_device_buffer_id") != retained_id
        or layer1.get("input_f32le_sha256") != retained.get("f32le_sha256")
        or layer1.get("same_command_graph_retained") is not True
        or layer1.get("l1_binding_executed") is not False
    ):
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "inner L1 input binding drifted or implies L1 execution"
        )
    l1_conv = _mapping(layer1.get("active_conv"), "inner L1 active conv")
    l1_recurrent = _mapping(layer1.get("active_recurrent"), "inner L1 active recurrent")
    for actual, expected, label in (
        (l1_conv, _mapping(l1_layout.get("active_conv"), "baseline L1 active conv"), "inner L1 active conv"),
        (l1_recurrent, _mapping(l1_layout.get("active_recurrent"), "baseline L1 active recurrent"), "inner L1 active recurrent"),
    ):
        for field in ("allocation_id", "slot", "offset_bytes", "capacity_bytes"):
            if actual.get(field) != expected.get(field):
                raise launcher.SourceTokenL0StateHandoffLauncherError(
                    f"{label}.{field} drifted from the static state layout"
                )
    l1_ids = {
        _nonempty_string(l1_conv.get("device_buffer_id"), "inner L1 active conv ID"),
        _nonempty_string(l1_recurrent.get("device_buffer_id"), "inner L1 active recurrent ID"),
    }
    if len(l1_ids) != 2 or retained_id in l1_ids:
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "inner L1 state allocations are aliased with each other or the retained L0 output"
        )
    policy = _mapping(document.get("metal_execution_policy"), "inner Metal policy")
    for field in (
        "strict_math_required",
    ):
        if policy.get(field) is not True:
            raise launcher.SourceTokenL0StateHandoffLauncherError(f"inner policy {field} drifted")
    for field in (
        "timing_or_benchmarking_allowed",
        "l1_prefix_execution_allowed",
        "complete_layer_or_token_allowed",
        "tps_or_tg_claim_allowed",
    ):
        if policy.get(field) is not False:
            raise launcher.SourceTokenL0StateHandoffLauncherError(f"inner policy {field} drifted")
    expected_lease = _child_binding(context.lease_evidence, context.lease_seal_sha256)
    expected_lease["lease_id"] = context.lease_id
    _exact(policy.get("lease_binding"), expected_lease, "inner lease binding")
    durable = _mapping(document.get("durable_capture"), "inner durable capture")
    if (
        durable.get("receipt_written_last_is_completion_marker") is not True
        or durable.get("outer_reaped_capture_required") is not True
        or durable.get("replay_guarded") is not True
    ):
        raise launcher.SourceTokenL0StateHandoffLauncherError("inner durable capture drifted")
    boundary = _mapping(document.get("claim_boundary"), "inner claim boundary")
    if (
        boundary.get("l0_post_state_rollback_retained_output_component_only") is not True
        or boundary.get("l1_binding_not_executed") is not True
        or boundary.get("may_not_satisfy_next_layer_execution_dependency") is not True
        or boundary.get("no_complete_layer_token_decoder_generation_server_hcli_tps_tg_or_tournament_claim")
        is not True
    ):
        raise launcher.SourceTokenL0StateHandoffLauncherError("inner claim boundary drifted")


def _inner_evidence(
    capture: Path,
    *,
    context: issuer.LeaseContext,
    outer_authority: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = capture / INNER_CAPTURE_DIRNAME / "receipt.json"
    if not receipt_path.is_file():
        return {"present": False, "binding_valid": False, "error": "inner receipt is absent"}
    try:
        receipt, receipt_seal = launcher._sealed_json(receipt_path, "inner receipt")
        evidence = launcher._evidence(receipt_path, "inner receipt")
        _validate_inner_receipt(receipt, context=context, outer_authority=outer_authority)
    except launcher.SourceTokenL0StateHandoffLauncherError as exc:
        return {"present": True, "binding_valid": False, "error": str(exc)}
    return {
        "present": True,
        "binding_valid": True,
        "receipt": _binding(evidence, receipt_seal),
        "schema": receipt.get("schema"),
        "status": receipt.get("status"),
    }


def _terminal_status(terminal: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    if terminal.get("spawn_error"):
        return f"{REFUSED_PREFIX}CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return f"{REFUSED_PREFIX}CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return f"{REFUSED_PREFIX}CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return f"{REFUSED_PREFIX}CHILD_NONZERO"
    if inner.get("binding_valid") is not True:
        return f"{REFUSED_PREFIX}ZERO_EXIT_WITHOUT_STRICT_PRE_L1_RECEIPT"
    return CAPTURED_STATUS


def _stream_evidence(path: Path, label: str) -> dict[str, Any]:
    """Retain stream evidence even for an overflow refusal.

    The caller records overflow as a terminal error, so a malicious/noisy child
    cannot prevent receipt-last reaping by making evidence inspection fail.
    """
    evidence = launcher._evidence(path, label)
    return {**evidence, "within_max_stream_bytes": evidence["bytes"] <= MAX_CHILD_STREAM_BYTES}


def _recommended_release_contract(
    *,
    config: CaptureConfig,
    context: issuer.LeaseContext,
    capture: Path,
    identity: str,
) -> dict[str, Any]:
    """Write only a future release contract, never a release itself."""
    return seal(
        {
            "schema": RELEASE_CONTRACT_SCHEMA,
            "status": RELEASE_CONTRACT_STATUS,
            "recorded_at": launcher._utc_now(),
            "lease": {
                **_binding(context.lease_evidence, context.lease_seal_sha256),
                "lease_id": context.lease_id,
            },
            "outer_terminal_path": str(capture / TERMINAL_FILENAME),
            "outer_terminal_must_be_sealed_and_terminal_before_release": True,
            "recommended_release_output_path": str(config.recommended_release_out),
            "launch_identity_sha256": identity,
            "coordination": {
                "actual_release_not_performed_by_outer_reaper": True,
                "new_qwen80_gpu_work_requires_a_fresh_explicit_lease": True,
                "watcher_hold_must_remain_active": True,
                "watcher_restart_or_transition_authorized": False,
                "automatic_retry_prohibited": True,
            },
            "claim_boundary": {
                "recommendation_is_gpu_coordination_only": True,
                "does_not_promote_pre_l1_component_to_layer_token_decoder_hcli_tps_tg_or_tournament": True,
            },
        }
    )


def _terminal_receipt(
    *,
    config: CaptureConfig,
    context: issuer.LeaseContext,
    capture: Path,
    identity: str,
    command: Sequence[str],
    child_pid: int | None,
    started_at: str,
    terminal: Mapping[str, Any],
    capture_error: str | None,
    outer_authority: Mapping[str, Any],
    release_contract_evidence: Mapping[str, Any],
    release_contract_seal_sha256: str,
) -> dict[str, Any]:
    inner = _inner_evidence(capture, context=context, outer_authority=outer_authority)
    try:
        terminal_child = launcher.validate_child_preflight(
            config.child_preflight, handoff_authority=config.handoff_authority
        )
        terminal_pointer: dict[str, Any] = launcher._versioned_current_observation(
            terminal_child, phase="terminal"
        )
        terminal_pointer_valid = True
    except launcher.SourceTokenL0StateHandoffLauncherError as exc:
        terminal_pointer = {"validation_error": str(exc)}
        terminal_pointer_valid = False
        if capture_error is None:
            capture_error = f"terminal normalized-current validation failed: {exc}"
    status = _terminal_status(terminal, inner)
    if capture_error is not None and status == CAPTURED_STATUS:
        status = f"{REFUSED_PREFIX}OUTER_CAPTURE_EVIDENCE_INVALID"
    if not terminal_pointer_valid and status == CAPTURED_STATUS:
        status = f"{REFUSED_PREFIX}TERMINAL_CURRENT_POINTER_INVALID"
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "recorded_at": launcher._utc_now(),
        "lease_id": context.lease_id,
        "launch_identity_sha256": identity,
        "one_shot": {
            "automatic_retry_disabled": True,
            "same_capture_dir_never_starts_a_second_child": True,
            "terminal_receipt_written_last": True,
            "lease_reuse_prohibited_after_terminal": True,
            "outer_reaped_child": terminal.get("reaped") is True,
        },
        "source_binding": {
            "preflight_proof": _binding(
                context.proof.proof_evidence, context.proof.proof_seal_sha256
            ),
            "outer_preflight": _binding(
                context.proof.outer_preflight_evidence,
                context.proof.outer_preflight_seal_sha256,
            ),
            "lease_receipt": _binding(context.lease_evidence, context.lease_seal_sha256),
            "watcher_hold": context.watcher_hold_evidence,
            "outer_launch_authority": _binding(
                launcher._evidence(
                    capture / OUTER_LAUNCH_AUTHORITY_FILENAME, "outer launch authority"
                ),
                str(outer_authority["seal_sha256"]),
            ),
            "handoff_contract": launcher._handoff_contract(),
            "probe_binary": context.proof.probe_binary,
        },
        "versioned_current_admission": {
            "historical_preflight": _mapping(
                context.proof.proof.get("versioned_current_admission"),
                "preflight versioned-current admission",
            ),
            "terminal": terminal_pointer,
            "terminal_current_pointer_valid": terminal_pointer_valid,
            "pointer_reseal_allowed_only_when_immutable_authority_is_exact": True,
        },
        "child": {
            "pid": child_pid,
            "started_at": started_at,
            "finished_at": launcher._utc_now(),
            "command": list(command),
            "terminal": dict(terminal),
        },
        "outer_capture": {
            "directory": str(capture),
            "stdout": _stream_evidence(capture / OUTER_STDOUT_FILENAME, "outer stdout"),
            "stderr": _stream_evidence(capture / OUTER_STDERR_FILENAME, "outer stderr"),
            "inner_capture_dir": str(capture / INNER_CAPTURE_DIRNAME),
        },
        "inner_probe_capture": inner,
        "recommended_release_contract": _binding(
            release_contract_evidence, release_contract_seal_sha256
        ),
        "release": {
            "actual_release_performed": False,
            "separate_recommended_release_output_path": str(config.recommended_release_out),
            "release_requires_this_terminal_receipt": True,
        },
        "claim_boundary": {
            "l0_post_state_rollback_retained_output_pre_l1_component_only": True,
            "l1_binding_not_executed": True,
            "l1_prefix_executed": False,
            "no_complete_layer_token_decoder_generation_server_hcli_tps_tg_or_tournament_claim": True,
            "watcher_or_server_transition_not_authorized": True,
        },
    }
    if capture_error is not None:
        payload["capture_error"] = capture_error
    return seal(payload)


def _replay(capture: Path, lease_id: str) -> dict[str, Any]:
    terminal_path = capture / TERMINAL_FILENAME
    if not terminal_path.is_file():
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "capture directory exists without a terminal receipt; no second child is allowed"
        )
    receipt, _ = launcher._sealed_json(terminal_path, "outer terminal receipt")
    if receipt.get("schema") != SCHEMA or receipt.get("lease_id") != lease_id:
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "capture directory belongs to another lease"
        )
    return receipt


def run_attempt(config: CaptureConfig) -> dict[str, Any]:
    """Run exactly one explicitly authorized child; never retry or release it."""
    context = _validate_config(config)
    if config.capture_dir.exists():
        return _replay(config.capture_dir, context.lease_id)
    if config.recommended_release_out.exists():
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "--recommended-release-out must be a new path"
        )
    identity = _launch_identity(config, context)
    # This existing normalized-launcher call creates the fresh capture,
    # replay guard, active plan, and exact Rust authority before any child.
    authority = launcher.prepare_future_one_shot(
        preflight_proof=config.preflight_proof,
        child_preflight=config.child_preflight,
        handoff_authority=config.handoff_authority,
        probe_bin=config.probe_bin,
        lease_receipt=config.lease_receipt,
        capture_dir=config.capture_dir,
        replay_guard_dir=config.replay_guard_dir,
        workers=config.workers,
    )
    capture = config.capture_dir.resolve(strict=True)
    if authority.get("lease_id") != context.lease_id:
        raise launcher.SourceTokenL0StateHandoffLauncherError(
            "normalized launcher authority lease ID drifted"
        )
    command = _child_command(config, context, capture=capture)
    started_at = launcher._utc_now()
    launcher._write_new(
        capture / RUNNING_FILENAME,
        seal(
            {
                "schema": SCHEMA,
                "status": "STARTED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_ONE_SHOT",
                "recorded_at": started_at,
                "lease_id": context.lease_id,
                "launch_identity_sha256": identity,
                "command": command,
                "claim_boundary": {
                    "pre_l1_component_only": True,
                    "automatic_retry_disabled": True,
                    "l1_prefix_executed": False,
                },
            }
        ),
    )
    child_pid: int | None = None
    capture_error: str | None = None
    with (capture / OUTER_STDOUT_FILENAME).open("xb") as stdout, (
        capture / OUTER_STDERR_FILENAME
    ).open("xb") as stderr:
        try:
            child = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            terminal = _terminal(None, timed_out=False, spawn_error=f"{type(exc).__name__}: {exc}")
        else:
            child_pid = child.pid
            try:
                launcher._write_new(
                    capture / CHILD_FILENAME,
                    seal(
                        {
                            "schema": SCHEMA,
                            "status": "RUNNING_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_ONE_SHOT",
                            "recorded_at": launcher._utc_now(),
                            "lease_id": context.lease_id,
                            "launch_identity_sha256": identity,
                            "pid": child_pid,
                            "parent_pid": os.getpid(),
                            "command": command,
                            "mode": "metal",
                            "strict_pre_l1_component_lease_required": True,
                        }
                    ),
                )
            except launcher.SourceTokenL0StateHandoffLauncherError as exc:
                capture_error = str(exc)
                terminal = _terminal(_terminate_group(child), timed_out=False)
            else:
                try:
                    terminal = _terminal(child.wait(timeout=config.timeout_seconds), timed_out=False)
                except subprocess.TimeoutExpired:
                    terminal = _terminal(_terminate_group(child), timed_out=True)
    try:
        stdout_evidence = _bounded_evidence(capture / OUTER_STDOUT_FILENAME, "outer stdout")
        stderr_evidence = _bounded_evidence(capture / OUTER_STDERR_FILENAME, "outer stderr")
    except launcher.SourceTokenL0StateHandoffLauncherError as exc:
        stdout_evidence = {"present": False, "error": str(exc)}
        stderr_evidence = {"present": False, "error": str(exc)}
        if capture_error is None:
            capture_error = str(exc)
    release_contract = _recommended_release_contract(
        config=config, context=context, capture=capture, identity=identity
    )
    # It is a recommendation/contract only.  The actual release output is a
    # separate later action and must bind the terminal receipt below.
    launcher._write_new(config.recommended_release_out, release_contract)
    release_evidence = launcher._evidence(
        config.recommended_release_out, "recommended release contract"
    )
    receipt = _terminal_receipt(
        config=config,
        context=context,
        capture=capture,
        identity=identity,
        command=command,
        child_pid=child_pid,
        started_at=started_at,
        terminal=terminal,
        capture_error=capture_error,
        outer_authority=authority,
        release_contract_evidence=release_evidence,
        release_contract_seal_sha256=str(release_contract["seal_sha256"]),
    )
    # Completion marker: do not create any outer-capture file after this.
    launcher._write_new(capture / TERMINAL_FILENAME, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-one-shot", action="store_true", required=True)
    parser.add_argument("--preflight-proof", type=Path, required=True)
    parser.add_argument("--child-preflight", type=Path, default=launcher.CHILD_PREFLIGHT_PATH)
    parser.add_argument("--handoff-authority", type=Path, default=launcher.HANDOFF_AUTHORITY_PATH)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--watcher-hold", type=Path, required=True)
    parser.add_argument("--lease-receipt", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--replay-guard-dir", type=Path, required=True)
    parser.add_argument("--recommended-release-out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run_attempt(
            CaptureConfig(
                preflight_proof=args.preflight_proof,
                child_preflight=args.child_preflight,
                handoff_authority=args.handoff_authority,
                probe_bin=args.probe_bin,
                watcher_hold=args.watcher_hold,
                lease_receipt=args.lease_receipt,
                capture_dir=args.capture_dir,
                replay_guard_dir=args.replay_guard_dir,
                recommended_release_out=args.recommended_release_out,
                workers=args.workers,
                timeout_seconds=args.timeout_seconds,
            )
        )
    except (launcher.SourceTokenL0StateHandoffLauncherError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"schema": SCHEMA, "status": f"{REFUSED_PREFIX}LAUNCHER", "error": str(exc)},
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt.get("status") == CAPTURED_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
