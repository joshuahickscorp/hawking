from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lab.operators import ascension_qwen80_source_token_l0_state_handoff_issue_lease as issuer
from lab.operators import ascension_qwen80_source_token_l0_state_handoff_launcher as launcher
from lab.operators import ascension_qwen80_source_token_l0_state_handoff_outer_capture as outer
from lab.receipts import seal


WATCHER_HOLD = (
    launcher.RUNTIME_DIR / "QWEN80_WATCHER_GPU_COORDINATION_HOLD_20260808T220751Z.json"
)


@pytest.fixture
def cpu_only_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / launcher.EXPECTED_PROBE_BASENAME
    path.write_text(
        """#!/usr/bin/env python3
import hashlib
import json
import sys

arguments = sys.argv[1:]
if arguments[arguments.index('--mode') + 1] != 'preflight':
    raise SystemExit(91)
outer_path = arguments[arguments.index('--outer-preflight') + 1]
raw = open(outer_path, 'rb').read()
outer = json.loads(raw)
print(json.dumps({
    'schema': 'hawking.ascension.qwen80_source_token_l0_state_handoff_capture.v1',
    'status': 'PREPARED_QWEN80_SOURCE_TOKEN_L0_POST_STATE_ROLLBACK_RETAINED_OUTPUT_L1_BINDING_NOT_EXECUTED_CHILD_NOT_LEASED_OR_EXECUTED',
    'mode': 'preflight',
    'outer_preflight_binding': {'path': outer_path, 'document_sha256': hashlib.sha256(raw).hexdigest(), 'seal_sha256': outer['seal_sha256']},
    'same_command_graph_contract': {'source_token_id': 1, 'prefix_dispatches': 9, 'suffix_dispatches': 14, 'total_dispatches': 23, 'l1_prefix_dispatches': 0, 'l1_binding_not_executed': True, 'retained_l0_second_residual_elements': 2048, 'retained_l0_second_residual_bytes': 8192, 'l0_slot': 0, 'l1_slot': 1},
    'claim_boundary': {'metal_device_or_dispatch_performed': False, 'lease_issued': False, 'l1_prefix_executed': False, 'complete_layer_or_token_performed': False, 'cannot_satisfy_next_layer_execution_dependency': True, 'no_decoder_generation_server_hcli_tps_tg_or_tournament_claim': True},
}, sort_keys=True))
""",
        encoding="utf-8",
    )
    path.chmod(0o700)
    monkeypatch.setattr(
        launcher,
        "EXPECTED_PROBE_SHA256",
        launcher._evidence(path, "test CPU-only probe", executable=True)["sha256"],
    )
    return path


def _prepared(cpu_only_probe: Path, tmp_path: Path) -> tuple[Path, Path]:
    preflight_capture = tmp_path / "preflight"
    launcher.run_preflight_only(
        child_preflight=launcher.CHILD_PREFLIGHT_PATH,
        handoff_authority=launcher.HANDOFF_AUTHORITY_PATH,
        probe_bin=cpu_only_probe,
        capture_dir=preflight_capture,
    )
    proof = preflight_capture / launcher.PREFLIGHT_PROOF_FILENAME
    lease = tmp_path / "lease.json"
    issuer.issue_lease(
        preflight_proof=proof,
        child_preflight=launcher.CHILD_PREFLIGHT_PATH,
        handoff_authority=launcher.HANDOFF_AUTHORITY_PATH,
        probe_bin=cpu_only_probe,
        watcher_hold=WATCHER_HOLD,
        out=lease,
    )
    return proof, lease


def _config(cpu_only_probe: Path, proof: Path, lease: Path, tmp_path: Path) -> outer.CaptureConfig:
    return outer.CaptureConfig(
        preflight_proof=proof,
        child_preflight=launcher.CHILD_PREFLIGHT_PATH,
        handoff_authority=launcher.HANDOFF_AUTHORITY_PATH,
        probe_bin=cpu_only_probe,
        watcher_hold=WATCHER_HOLD,
        lease_receipt=lease,
        capture_dir=tmp_path / "outer-capture",
        replay_guard_dir=tmp_path / "replay-guards",
        recommended_release_out=tmp_path / "recommended-release.json",
        workers=1,
        timeout_seconds=30.0,
    )


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fake_inner_receipt(
    *,
    capture: Path,
    context: issuer.LeaseContext,
    authority: dict[str, object],
) -> dict[str, object]:
    authority_path = capture / outer.OUTER_LAUNCH_AUTHORITY_FILENAME
    authority_evidence = launcher._evidence(authority_path, "test outer authority")
    authority_seal = str(authority["seal_sha256"])
    layout = context.proof.child.handoff.authority["static_state_layout_authority"]
    l0 = layout["l0"]
    l1 = layout["l1"]
    second = context.proof.child.handoff.authority["consumed_component_capture"]["second_residual"]
    retained_id = "retained-l0-output-device-buffer"
    l0_active_conv = "l0-active-conv-buffer"
    l0_active_recurrent = "l0-active-recurrent-buffer"
    l0_rollback_conv = "l0-rollback-conv-buffer"
    l0_rollback_recurrent = "l0-rollback-recurrent-buffer"
    l1_conv = "l1-active-conv-buffer"
    l1_recurrent = "l1-active-recurrent-buffer"

    def state(name: str, device_buffer_id: str, hash_field: str) -> dict[str, object]:
        return {
            **l0[name],
            "device_buffer_id": device_buffer_id,
            hash_field: _sha(f"{name}-{hash_field}"),
        }

    lease_binding = {
        "path": context.lease_evidence["path"],
        "document_sha256": context.lease_evidence["sha256"],
        "seal_sha256": context.lease_seal_sha256,
        "lease_id": context.lease_id,
    }
    document = {
        "schema": launcher.PRE_L1_CAPTURE_SCHEMA,
        "status": launcher.PRE_L1_CAPTURE_STATUS,
        "mode": "metal",
        "metal_device_or_dispatch_performed": True,
        "component_only": True,
        "l1_binding_not_executed": True,
        "l1_prefix_dispatches": 0,
        "complete_layer_or_token_performed": False,
        "outer_preflight_binding": {
            "path": context.proof.outer_preflight_evidence["path"],
            "document_sha256": context.proof.outer_preflight_evidence["sha256"],
            "seal_sha256": context.proof.outer_preflight_seal_sha256,
        },
        "outer_launch_authority_binding": {
            "path": authority_evidence["path"],
            "document_sha256": authority_evidence["sha256"],
            "seal_sha256": authority_seal,
        },
        "same_command_graph": {
            "source_token_id": 1,
            "prefix_dispatches": 9,
            "suffix_dispatches": 14,
            "total_dispatches": 23,
            "same_command_graph_retained": True,
            "fenced_once_after_prefix_and_suffix": True,
        },
        "l0_state_handoff": {
            "schema": launcher.PRE_L1_CAPTURE_SCHEMA,
            "status": launcher.PRE_L1_CAPTURE_STATUS,
            "source_token_id": 1,
            "same_command_graph_retained": True,
            "l1_binding_not_executed": True,
            "l1_prefix_dispatches": 0,
            "retained_l0_second_residual": {
                "elements": 2048,
                "bytes": 8192,
                "f32le_sha256": second["f32le_sha256"],
                "device_buffer_id": retained_id,
                "retained_for_future_layer1_encode": True,
            },
            "l0_post_state_commit": {
                "layer": 0,
                "linear_state_slot": 0,
                "checkpoint_before_mutation": True,
                "active_conv": state("active_conv", l0_active_conv, "post_state_f32le_sha256"),
                "active_recurrent": state("active_recurrent", l0_active_recurrent, "post_state_f32le_sha256"),
                "rollback_conv": state("rollback_conv", l0_rollback_conv, "checkpoint_f32le_sha256"),
                "rollback_recurrent": state("rollback_recurrent", l0_rollback_recurrent, "checkpoint_f32le_sha256"),
            },
            "layer1_input_binding": {
                "session_id": layout["session_id"],
                "layer": 1,
                "linear_state_slot": 1,
                "input_device_buffer_id": retained_id,
                "input_f32le_sha256": second["f32le_sha256"],
                "same_command_graph_retained": True,
                "l1_binding_executed": False,
                "active_conv": {
                    **l1["active_conv"],
                    "device_buffer_id": l1_conv,
                    "device_buffer_identity_sha256": _sha(l1_conv),
                },
                "active_recurrent": {
                    **l1["active_recurrent"],
                    "device_buffer_id": l1_recurrent,
                    "device_buffer_identity_sha256": _sha(l1_recurrent),
                },
            },
        },
        "metal_execution_policy": {
            "strict_math_required": True,
            "timing_or_benchmarking_allowed": False,
            "l1_prefix_execution_allowed": False,
            "complete_layer_or_token_allowed": False,
            "tps_or_tg_claim_allowed": False,
            "lease_binding": lease_binding,
        },
        "durable_capture": {
            "receipt_written_last_is_completion_marker": True,
            "outer_reaped_capture_required": True,
            "replay_guarded": True,
        },
        "claim_boundary": {
            "l0_post_state_rollback_retained_output_component_only": True,
            "l1_binding_not_executed": True,
            "may_not_satisfy_next_layer_execution_dependency": True,
            "no_complete_layer_token_decoder_generation_server_hcli_tps_tg_or_tournament_claim": True,
        },
    }
    return seal(document)


def test_fake_child_is_reaped_and_outer_terminal_is_receipt_last(
    cpu_only_probe: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proof, lease = _prepared(cpu_only_probe, tmp_path)
    config = _config(cpu_only_probe, proof, lease, tmp_path)
    context = issuer.validate_lease(
        lease_receipt=lease,
        preflight_proof=proof,
        child_preflight=launcher.CHILD_PREFLIGHT_PATH,
        handoff_authority=launcher.HANDOFF_AUTHORITY_PATH,
        probe_bin=cpu_only_probe,
        watcher_hold=WATCHER_HOLD,
    )
    calls: list[list[str]] = []

    class FakePopen:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            calls.append(command)
            self.pid = 12345
            self.returncode: int | None = None
            stdout = kwargs["stdout"]
            stdout.write(b'{"fake_child":"metal receipt persisted separately"}\n')  # type: ignore[union-attr]
            capture = Path(command[command.index("--outer-capture-dir") + 1])
            inner = Path(command[command.index("--capture-dir") + 1])
            inner.mkdir()
            authority = json.loads(
                (capture / outer.OUTER_LAUNCH_AUTHORITY_FILENAME).read_text(encoding="utf-8")
            )
            receipt = _fake_inner_receipt(
                capture=capture, context=context, authority=authority
            )
            (inner / "receipt.json").write_text(
                json.dumps(receipt, sort_keys=True), encoding="utf-8"
            )

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            self.returncode = 0
            return 0

    monkeypatch.setattr(outer.subprocess, "Popen", FakePopen)
    receipt = outer.run_attempt(config)

    assert receipt["status"] == outer.CAPTURED_STATUS
    assert receipt["one_shot"]["outer_reaped_child"] is True
    assert receipt["inner_probe_capture"]["binding_valid"] is True
    assert receipt["claim_boundary"]["l1_prefix_executed"] is False
    assert len(calls) == 1
    assert "--mode" in calls[0] and calls[0][calls[0].index("--mode") + 1] == "metal"
    assert (config.capture_dir / launcher.TERMINAL_RECEIPT_FILENAME).is_file()
    release = json.loads(config.recommended_release_out.read_text(encoding="utf-8"))
    assert release["status"] == outer.RELEASE_CONTRACT_STATUS
    assert release["coordination"]["actual_release_not_performed_by_outer_reaper"] is True

    # A repeated invocation returns the immutable terminal record and cannot
    # spawn a second fake child or create a second release recommendation.
    replay = outer.run_attempt(config)
    assert replay["seal_sha256"] == receipt["seal_sha256"]
    assert len(calls) == 1


def test_inner_l1_execution_claim_is_refused_without_retry(
    cpu_only_probe: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proof, lease = _prepared(cpu_only_probe, tmp_path)
    config = _config(cpu_only_probe, proof, lease, tmp_path)
    context = issuer.validate_lease(
        lease_receipt=lease,
        preflight_proof=proof,
        child_preflight=launcher.CHILD_PREFLIGHT_PATH,
        handoff_authority=launcher.HANDOFF_AUTHORITY_PATH,
        probe_bin=cpu_only_probe,
        watcher_hold=WATCHER_HOLD,
    )

    class FakePopen:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            self.pid = 12346
            self.returncode: int | None = None
            capture = Path(command[command.index("--outer-capture-dir") + 1])
            inner = Path(command[command.index("--capture-dir") + 1])
            inner.mkdir()
            authority = json.loads(
                (capture / outer.OUTER_LAUNCH_AUTHORITY_FILENAME).read_text(encoding="utf-8")
            )
            receipt = _fake_inner_receipt(
                capture=capture, context=context, authority=authority
            )
            receipt["l1_prefix_dispatches"] = 1
            receipt = seal({key: value for key, value in receipt.items() if key != "seal_sha256"})
            (inner / "receipt.json").write_text(
                json.dumps(receipt, sort_keys=True), encoding="utf-8"
            )

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            self.returncode = 0
            return 0

    monkeypatch.setattr(outer.subprocess, "Popen", FakePopen)
    receipt = outer.run_attempt(config)

    assert receipt["status"].startswith(outer.REFUSED_PREFIX)
    assert receipt["inner_probe_capture"]["binding_valid"] is False
    assert receipt["claim_boundary"]["l1_prefix_executed"] is False
    assert (config.capture_dir / launcher.TERMINAL_RECEIPT_FILENAME).is_file()
