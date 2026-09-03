from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lab.operators import ascension_qwen80_source_token_l0_state_handoff_issue_lease as issuer
from lab.operators import ascension_qwen80_source_token_l0_state_handoff_launcher as launcher


WATCHER_HOLD = (
    launcher.RUNTIME_DIR / "QWEN80_WATCHER_GPU_COORDINATION_HOLD_20260808T220751Z.json"
)


@pytest.fixture
def cpu_only_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A file-only child preflight stand-in; it has no Metal-mode behavior."""
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


def _proof(cpu_only_probe: Path, tmp_path: Path) -> Path:
    capture = tmp_path / "preflight"
    launcher.run_preflight_only(
        child_preflight=launcher.CHILD_PREFLIGHT_PATH,
        handoff_authority=launcher.HANDOFF_AUTHORITY_PATH,
        probe_bin=cpu_only_probe,
        capture_dir=capture,
    )
    return capture / launcher.PREFLIGHT_PROOF_FILENAME


def test_issuer_creates_exact_fresh_lease_from_cpu_only_preflight(
    cpu_only_probe: Path, tmp_path: Path
) -> None:
    proof = _proof(cpu_only_probe, tmp_path)
    out = tmp_path / "handoff-lease.json"

    result = issuer.issue_lease(
        preflight_proof=proof,
        child_preflight=launcher.CHILD_PREFLIGHT_PATH,
        handoff_authority=launcher.HANDOFF_AUTHORITY_PATH,
        probe_bin=cpu_only_probe,
        watcher_hold=WATCHER_HOLD,
        out=out,
    )

    assert result.lease["schema"] == launcher.FUTURE_LEASE_SCHEMA
    assert result.lease["status"] == launcher.FUTURE_LEASE_STATUS
    assert result.lease["lifecycle"]["replay_guarded"] is True
    assert result.lease["execution_policy"]["l1_prefix_execution_allowed"] is False
    assert result.lease["watcher_coordination"]["watcher_hold"] == launcher._evidence(
        WATCHER_HOLD, "watcher hold"
    )
    assert result.lease["claim_boundary"]["lease_issuance_is_file_and_cpu_only"] is True
    assert out.is_file()


def test_watcher_hold_manifest_substitution_is_refused(
    cpu_only_probe: Path, tmp_path: Path
) -> None:
    proof = _proof(cpu_only_probe, tmp_path)
    context = launcher.validate_preflight_proof(
        proof_path=proof,
        child_preflight=launcher.CHILD_PREFLIGHT_PATH,
        handoff_authority=launcher.HANDOFF_AUTHORITY_PATH,
        probe_bin=cpu_only_probe,
    )
    drift = json.loads(WATCHER_HOLD.read_text(encoding="utf-8"))
    drift["source_binding"]["manifest_seal_sha256"] = hashlib.sha256(b"drift").hexdigest()
    drift_path = tmp_path / "watcher-drift.json"
    drift_path.write_text(json.dumps(drift, sort_keys=True), encoding="utf-8")
    drift_path.chmod(0o600)

    with pytest.raises(launcher.SourceTokenL0StateHandoffLauncherError, match="manifest_seal"):
        issuer._watcher_hold(drift_path, context=context)


def test_issuer_refuses_to_overwrite_a_lease_output(
    cpu_only_probe: Path, tmp_path: Path
) -> None:
    proof = _proof(cpu_only_probe, tmp_path)
    out = tmp_path / "already-exists.json"
    out.write_text("{}", encoding="utf-8")

    with pytest.raises(launcher.SourceTokenL0StateHandoffLauncherError, match="new absolute"):
        issuer.issue_lease(
            preflight_proof=proof,
            child_preflight=launcher.CHILD_PREFLIGHT_PATH,
            handoff_authority=launcher.HANDOFF_AUTHORITY_PATH,
            probe_bin=cpu_only_probe,
            watcher_hold=WATCHER_HOLD,
            out=out,
        )
