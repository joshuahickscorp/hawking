from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.operators import ascension_qwen80_source_token_l0_state_handoff_launcher as launcher
from lab.receipts import seal


def _sha(index: int) -> str:
    return f"{index:064x}"


@pytest.fixture
def probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / launcher.EXPECTED_PROBE_BASENAME
    path.write_text(
        """#!/usr/bin/env python3
import hashlib
import json
import sys

arguments = sys.argv[1:]
outer_path = arguments[arguments.index("--outer-preflight") + 1]
with open(outer_path, "rb") as handle:
    outer_raw = handle.read()
outer = json.loads(outer_raw)
document = {
    "schema": "hawking.ascension.qwen80_source_token_l0_state_handoff_capture.v1",
    "status": "PREPARED_QWEN80_SOURCE_TOKEN_L0_POST_STATE_ROLLBACK_RETAINED_OUTPUT_L1_BINDING_NOT_EXECUTED_CHILD_NOT_LEASED_OR_EXECUTED",
    "mode": "preflight",
    "outer_preflight_binding": {
        "path": outer_path,
        "document_sha256": hashlib.sha256(outer_raw).hexdigest(),
        "seal_sha256": outer["seal_sha256"],
    },
    "same_command_graph_contract": {
        "source_token_id": 1,
        "prefix_dispatches": 9,
        "suffix_dispatches": 14,
        "total_dispatches": 23,
        "l1_prefix_dispatches": 0,
        "l1_binding_not_executed": True,
        "retained_l0_second_residual_elements": 2048,
        "retained_l0_second_residual_bytes": 8192,
        "l0_slot": 0,
        "l1_slot": 1,
    },
    "claim_boundary": {
        "metal_device_or_dispatch_performed": False,
        "lease_issued": False,
        "l1_prefix_executed": False,
        "complete_layer_or_token_performed": False,
        "cannot_satisfy_next_layer_execution_dependency": True,
        "no_decoder_generation_server_hcli_tps_tg_or_tournament_claim": True,
    },
}
print(json.dumps(document, sort_keys=True))
""",
        encoding="utf-8",
    )
    path.chmod(0o700)
    monkeypatch.setattr(
        launcher,
        "EXPECTED_PROBE_SHA256",
        launcher._evidence(path, "test probe", executable=True)["sha256"],
    )
    return path


def test_current_child_preflight_binds_exact_handoff_component_and_raw_evidence() -> None:
    context = launcher.validate_child_preflight()

    assert context.seal_sha256 == launcher.CHILD_PREFLIGHT_SEAL_SHA256
    assert context.evidence["sha256"] == launcher.CHILD_PREFLIGHT_RAW_SHA256
    assert context.handoff.authority_seal_sha256 == launcher.HANDOFF_AUTHORITY_SEAL_SHA256
    assert context.handoff.component_outer["status"] == launcher.COMPONENT_OUTER_STATUS
    assert context.handoff.component_inner["status"] == launcher.COMPONENT_INNER_STATUS


def test_cpu_preflight_writes_a_terminal_plan_without_child_or_device_activity(
    probe: Path, tmp_path: Path
) -> None:
    capture = tmp_path / "preflight-capture"
    proof = launcher.run_preflight_only(
        child_preflight=launcher.CHILD_PREFLIGHT_PATH,
        handoff_authority=launcher.HANDOFF_AUTHORITY_PATH,
        probe_bin=probe,
        capture_dir=capture,
    )

    assert proof["schema"] == launcher.PREFLIGHT_PROOF_SCHEMA
    assert proof["claim_boundary"]["device_child_spawned"] is False  # type: ignore[index]
    outer = json.loads((capture / launcher.OUTER_PREFLIGHT_FILENAME).read_text(encoding="utf-8"))
    assert outer["schema"] == launcher.OUTER_PREFLIGHT_SCHEMA
    assert outer["status"] == launcher.OUTER_PREFLIGHT_STATUS
    assert set(outer["source_binding"]) == {
        "manifest",
        "manifest_seal_sha256",
        "admission_current",
        "admission_pointer_seal_sha256",
        "admission_receipt",
        "admission_receipt_seal_sha256",
        "source_audit_seal_sha256",
        "source_revision",
        "source_all_ten_outer_preflight",
        "source_all_ten_outer_preflight_seal_sha256",
        "l0_state_handoff_child_preflight",
        "l0_state_handoff_child_preflight_seal_sha256",
        "baseline_l0_to_l1_handoff_authority",
        "baseline_l0_to_l1_handoff_authority_seal_sha256",
    }
    assert outer["handoff_contract"]["l1_binding_not_executed"] is True
    assert outer["handoff_contract"]["l1_prefix_dispatches"] == 0
    assert outer["handoff_contract"]["strict_claim_boundary"]["l1_executed"] is False
    versioned = outer["versioned_current_admission"]
    assert versioned["observed_pointer_evidence"] == outer["source_binding"]["admission_current"]
    assert versioned["observed_pointer_seal_sha256"] == outer["source_binding"]["admission_pointer_seal_sha256"]
    assert versioned["acceptance"]["manifest_or_receipt_substitution_accepted"] is False
    assert (capture / launcher.OUTER_PREFLIGHT_FILENAME).is_file()
    assert (capture / launcher.TERMINAL_PLAN_FILENAME).is_file()
    assert (capture / launcher.CHILD_PREFLIGHT_STDOUT_FILENAME).is_file()
    assert (capture / launcher.CHILD_PREFLIGHT_STDERR_FILENAME).read_bytes() == b""
    assert not (capture / launcher.TERMINAL_RECEIPT_FILENAME).exists()
    assert launcher.run_preflight_only(
        child_preflight=launcher.CHILD_PREFLIGHT_PATH,
        handoff_authority=launcher.HANDOFF_AUTHORITY_PATH,
        probe_bin=probe,
        capture_dir=capture,
    )["seal_sha256"] == proof["seal_sha256"]


def test_raw_child_drift_is_refused_before_any_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, "CHILD_PREFLIGHT_RAW_SHA256", _sha(1))

    with pytest.raises(launcher.SourceTokenL0StateHandoffLauncherError, match="raw evidence"):
        launcher.validate_child_preflight()


def test_probe_binary_sha_drift_is_refused_before_cpu_child_execution(
    probe: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(launcher, "EXPECTED_PROBE_SHA256", _sha(2))

    with pytest.raises(launcher.SourceTokenL0StateHandoffLauncherError, match="SHA-256 drifted"):
        launcher.run_preflight_only(
            child_preflight=launcher.CHILD_PREFLIGHT_PATH,
            handoff_authority=launcher.HANDOFF_AUTHORITY_PATH,
            probe_bin=probe,
            capture_dir=tmp_path / "must-not-exist",
        )


def test_versioned_current_accepts_only_a_pointer_reseal() -> None:
    child = launcher.validate_child_preflight()
    observed = launcher._source_binding(child)
    observed["admission_current"] = {
        "path": str(launcher.ADMISSION_CURRENT_PATH.resolve(strict=True)),
        "present": True,
        "bytes": 1,
        "sha256": _sha(3),
    }
    observed["admission_pointer_seal_sha256"] = _sha(4)

    launcher._validate_versioned_current_source_binding(
        observed, child, "test versioned-current source"
    )

    manifest_drift = dict(observed)
    manifest_drift["manifest"] = {**observed["manifest"], "sha256": _sha(5)}
    with pytest.raises(launcher.SourceTokenL0StateHandoffLauncherError, match="immutable authority manifest"):
        launcher._validate_versioned_current_source_binding(
            manifest_drift, child, "test manifest substitution"
        )

    receipt_drift = dict(observed)
    receipt_drift["admission_receipt"] = {
        **observed["admission_receipt"],
        "sha256": _sha(6),
    }
    with pytest.raises(
        launcher.SourceTokenL0StateHandoffLauncherError,
        match="immutable authority admission_receipt",
    ):
        launcher._validate_versioned_current_source_binding(
            receipt_drift, child, "test receipt substitution"
        )


def _preflight(probe: Path, tmp_path: Path) -> tuple[Path, dict[str, object]]:
    capture = tmp_path / "preflight"
    proof = launcher.run_preflight_only(
        child_preflight=launcher.CHILD_PREFLIGHT_PATH,
        handoff_authority=launcher.HANDOFF_AUTHORITY_PATH,
        probe_bin=probe,
        capture_dir=capture,
    )
    return capture / launcher.PREFLIGHT_PROOF_FILENAME, proof


def _lease(path: Path, proof_path: Path, proof: dict[str, object], probe: Path) -> None:
    child = launcher.validate_child_preflight()
    preflight_capture = proof_path.parent
    outer_path = preflight_capture / launcher.OUTER_PREFLIGHT_FILENAME
    outer, outer_seal = launcher._sealed_json(outer_path, "outer")
    document = seal(
        {
            "schema": launcher.FUTURE_LEASE_SCHEMA,
            "status": launcher.FUTURE_LEASE_STATUS,
            "lease_id": _sha(77),
            "preflight_proof_binding": {
                **launcher._evidence(proof_path, "proof"),
                "seal_sha256": proof["seal_sha256"],
            },
            "outer_preflight": launcher._evidence(outer_path, "outer"),
            "outer_preflight_seal_sha256": outer_seal,
            "l0_state_handoff_child_preflight": child.evidence,
            "l0_state_handoff_child_preflight_seal_sha256": child.seal_sha256,
            "baseline_l0_to_l1_handoff_authority": child.handoff.authority_evidence,
            "baseline_l0_to_l1_handoff_authority_seal_sha256": child.handoff.authority_seal_sha256,
            "handoff_contract": launcher._handoff_contract(),
            "probe_binary": launcher._evidence(probe, "probe", executable=True),
            "artifact_binding": {
                "manifest_document_sha256": child.handoff.manifest_evidence["sha256"],
                "manifest_seal_sha256": child.handoff.manifest_seal_sha256,
                "admission_receipt_seal_sha256": child.handoff.admission_receipt_seal_sha256,
            },
            "lifecycle": {
                "fresh_for_this_exact_launch": True,
                "outer_reaped_capture_required": True,
                "lease_released_after_first_terminal_child": True,
                "automatic_retry_prohibited": True,
                "replay_guarded": True,
            },
            "execution_policy": {
                "component": "qwen80_source_token_l0_state_handoff",
                "quiet_qwen80_device_lease": True,
                "strict_math": True,
                "timing_or_benchmarking_allowed": False,
                "l1_prefix_execution_allowed": False,
                "complete_layer_or_token_allowed": False,
                "tps_or_tg_claim_allowed": False,
            },
            "watcher_coordination": {
                "watcher_hold_must_remain_active": True,
                "watcher_restart_or_transition_authorized": False,
            },
        }
    )
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    path.chmod(0o600)


def test_future_plan_binds_probe_child_lease_and_refuses_replay(probe: Path, tmp_path: Path) -> None:
    proof_path, proof = _preflight(probe, tmp_path)
    lease_path = tmp_path / "lease.json"
    _lease(lease_path, proof_path, proof, probe)
    guard = tmp_path / "lease-replay-guards"
    authority = launcher.prepare_future_one_shot(
        preflight_proof=proof_path,
        child_preflight=launcher.CHILD_PREFLIGHT_PATH,
        handoff_authority=launcher.HANDOFF_AUTHORITY_PATH,
        probe_bin=probe,
        lease_receipt=lease_path,
        capture_dir=tmp_path / "future-capture",
        replay_guard_dir=guard,
    )

    assert authority["lease_id"] == _sha(77)
    assert authority["outer_reaper"]["terminal_receipt_written_last"] is True  # type: ignore[index]
    assert authority["watcher_coordination"]["watcher_hold_must_remain_active"] is True  # type: ignore[index]
    assert authority["execution_policy"]["l1_prefix_execution_allowed"] is False  # type: ignore[index]
    assert authority["launch_versioned_current_admission"]["phase"] == "launch"  # type: ignore[index]
    assert authority["terminal_versioned_current_recheck"]["terminal_pointer_raw_and_seal_evidence_required"] is True  # type: ignore[index]
    assert authority["claim_boundary"]["device_child_spawned"] is False  # type: ignore[index]
    assert authority["required_pre_l1_handoff_capture"]["l1_binding_not_executed"] is True  # type: ignore[index]
    assert authority["required_pre_l1_handoff_capture"]["l1_prefix_dispatches"] == 0  # type: ignore[index]
    assert authority["required_pre_l1_handoff_capture"]["may_not_satisfy_next_layer_execution_dependency"] is True  # type: ignore[index]
    with pytest.raises(launcher.SourceTokenL0StateHandoffLauncherError, match="refusing non-unique"):
        launcher.prepare_future_one_shot(
            preflight_proof=proof_path,
            child_preflight=launcher.CHILD_PREFLIGHT_PATH,
            handoff_authority=launcher.HANDOFF_AUTHORITY_PATH,
            probe_bin=probe,
            lease_receipt=lease_path,
            capture_dir=tmp_path / "future-capture-2",
            replay_guard_dir=guard,
        )


@pytest.mark.parametrize("argument", ["--execute-one-shot", "--metal", "--router-receipt=/tmp/legacy.json"])
def test_device_and_legacy_cli_arguments_are_refused(
    argument: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert launcher.main([argument]) == 2
    assert argument.split("=", 1)[0] in capsys.readouterr().out
