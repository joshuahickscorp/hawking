#!/usr/bin/env python3.12
"""Adversarial offline tests for the GLM-5.2 phone/operator CLI."""
from __future__ import annotations

import base64
import hashlib
import json
import pathlib
import sys

import pytest


TOOLS = pathlib.Path(__file__).resolve().parents[2]
CONDENSE = TOOLS / "condense"
for directory in (TOOLS, CONDENSE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import glm52_gravity as cli  # noqa: E402
import glm52_evidence_auth as gea  # noqa: E402
import glm52_grounding_auth as gga  # noqa: E402
import glm52_state as gs  # noqa: E402
import glm52_telegram as gt  # noqa: E402
from glm52_common import (  # noqa: E402
    atomic_json,
    canonical,
    read_sealed_json,
    seal,
    verify_sealed,
)


REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
CAMPAIGN = "glm52-phone-cli-test"
SHARD = "model-00001-of-00282.safetensors"
TENSOR = "model.layers.0.self_attn.q_proj.weight"
CHAT_ID = 100424242
CHAT_DIGEST = gs.telegram_chat_identity_digest(CHAT_ID)
KEY = b"offline-cli-test-hmac-key-at-least-32-bytes!!"
EVIDENCE_KEY = b"e" * 32
GROUNDING_KEY = b"g" * 32


class FakeKeychain:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def get(self, service):
        return self.values.get(service)

    def set(self, service, value):
        self.values[service] = value


def _keychain(
    key: bytes = KEY,
    chat_id: int = CHAT_ID,
    evidence_key: bytes = EVIDENCE_KEY,
    grounding_key: bytes | None = GROUNDING_KEY,
) -> FakeKeychain:
    values = {
        gt.CHAT_SERVICE: str(chat_id),
        gt.HMAC_SERVICE: base64.urlsafe_b64encode(key).decode("ascii"),
        gea.EVIDENCE_HMAC_SERVICE:
            base64.urlsafe_b64encode(evidence_key).decode("ascii"),
    }
    if grounding_key is not None:
        values[gga.GROUNDING_HMAC_SERVICE] = (
            base64.urlsafe_b64encode(grounding_key).decode("ascii")
        )
    return FakeKeychain(values)


def _source_shard():
    return {
        "path": SHARD,
        "logical_bytes": 4096,
        "xet_hash": "a" * 64,
        "lfs_sha256": "b" * 64,
    }


def _gates():
    def artifact_policy(path, schema):
        return {
            "path": path,
            "expected_seal_sha256": None,
            "expected_schema": schema,
            "allowed_statuses": ["PASS"],
            "validator_id": "campaign_artifact_v1",
            "validator_source_sha256": gs.EVIDENCE_VALIDATOR_SOURCE_SHA256[
                "campaign_artifact_v1"
            ],
            "require_producer_hmac": True,
        }

    def checklist_policy(name):
        return {
            "path": f"evidence/stop_conditions/{name}.json",
            "expected_seal_sha256": None,
            "expected_schema": gs.STOP_CONDITION_EVIDENCE_SCHEMA,
            "allowed_statuses": ["PASS"],
            "validator_id": "stop_condition_v1",
            "validator_source_sha256": gs.EVIDENCE_VALIDATOR_SOURCE_SHA256[
                "stop_condition_v1"
            ],
            "require_producer_hmac": True,
        }

    return {
        "ASSEMBLE_ARTIFACT": {
            "require_source_complete": True,
            "require_tensor_complete": True,
            "require_final_source_eviction": True,
            "require_telegram_delivery": True,
            "require_phone_status": False,
            "required_phone_status_path": None,
            "required_artifacts": {
                "source_manifest": artifact_policy(
                    "GLM52_OFFICIAL_MANIFEST.json", "test.source_manifest.v1"
                )
            },
            "required_checklist": {
                "source_complete": checklist_policy("source_complete")
            },
        },
        "COMPLETE": {
            "require_source_complete": True,
            "require_tensor_complete": True,
            "require_final_source_eviction": True,
            "require_telegram_delivery": True,
            "require_phone_status": True,
            "required_phone_status_path": "phone/GLM52_PHONE_STATUS.json",
            "required_artifacts": {
                "final": artifact_policy(
                    "GLM52_GRAVITY_FINAL.json", "test.gravity_final.v1"
                )
            },
            "required_checklist": {
                "all_stop_conditions": checklist_policy("all_stop_conditions")
            },
        },
    }


def _contract():
    return gs.make_expected_campaign_contract(
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
        expected_chat_identity_digest=CHAT_DIGEST,
        source_shards=[_source_shard()],
        expected_tensors=[TENSOR],
        window_schedule=[
            {
                "schedule_index": 0,
                "window_id": "window-0001",
                "source_shards": [SHARD],
                "carry_in_shards": [],
                "new_fetch_shards": [SHARD],
                "refetch_shards": [],
                "carry_out_shards": [],
                "evict_shards": [SHARD],
                "tensor_set": [TENSOR],
            }
        ],
        state_gates=_gates(),
        source_profile="SYNTHETIC_TEST_ONLY",
        created_at="2026-07-21T00:00:00Z",
    )


def _auth():
    return gs.TelegramAuthConfig(hmac_key=KEY, expected_chat_identity_digest=CHAT_DIGEST)


def _evidence_auth():
    return gs.EvidenceAuthConfig(
        hmac_key=EVIDENCE_KEY,
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
    )


def _delivery(controller, intent):
    claim_id = intent["claim_id"]
    message_id = int(hashlib.sha256(claim_id.encode()).hexdigest()[:12], 16) + 1
    response = {
        "ok": True,
        "result": {
            "message_id": message_id,
            "chat": {"id": CHAT_ID, "type": "private"},
            "text": intent["rendered_message"],
        },
    }
    return gs.make_telegram_delivery_receipt(
        intent,
        auth=controller.telegram_auth,
        bot_api_response=response,
        http_status=200,
        delivered_at="2026-07-21T00:00:01Z",
    )


def _setup(tmp_path, *, reach_freeze=False):
    controller_root = tmp_path / "controller"
    contract_path = tmp_path / "expected-contract.json"
    config_path = tmp_path / "cli-config.json"
    phone_dir = tmp_path / "phone"
    atomic_json(contract_path, _contract())
    config = cli.make_config(
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
        controller_root=controller_root,
        artifact_root=tmp_path,
        expected_contract_path=contract_path,
        phone_status_directory=phone_dir,
        expected_chat_identity_digest=CHAT_DIGEST,
        allow_synthetic_contract=True,
    )
    atomic_json(config_path, config)
    keychain = _keychain()
    controller = gs.Controller(
        controller_root,
        artifact_root=tmp_path,
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
        expected_contract=_contract(),
        telegram_auth=_auth(),
        evidence_auth=_evidence_auth(),
        allow_synthetic_contract=True,
    )
    with controller:
        boot_claim = "cli:test:boot:0001"
        intent = controller.prepare_transition("PRECHECK", claim_id=boot_claim)
        controller.boot(
            transition_intent=intent,
            telegram_delivery=_delivery(controller, intent),
        )
        if reach_freeze:
            states = [
                "CLOSE_KIMI",
                "RELEASE_KIMI_SOURCE",
                "ADMIT_GLM_SOURCE",
                "BUILD_MANIFEST",
                "BUILD_DEPENDENCY_GRAPH",
                "AUTOTUNE_XET",
                "BUILD_ADAPTER",
                "BUILD_REFERENCE",
                "BUILD_CORPUS",
                "PILOT_ORACLES",
                "FREEZE_PROGRAM",
            ]
            for number, to_state in enumerate(states, 1):
                claim = f"cli:test:transition:{number:04d}"
                intent = controller.prepare_transition(to_state, claim_id=claim)
                controller.transition(
                    to_state,
                    transition_intent=intent,
                    telegram_delivery=_delivery(controller, intent),
                )
    runtime = cli.load_runtime(config_path, keychain=keychain)
    return runtime, config_path, keychain


def test_contract_derived_config_uses_artifact_root_for_phone_status(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    contract_path = artifact_root / "expected-contract.json"
    atomic_json(contract_path, _contract())
    config = cli.make_config_from_contract(
        expected_contract_path=contract_path,
        controller_root=tmp_path / "controller",
        artifact_root=artifact_root,
        allow_synthetic_contract=True,
    )
    assert config["campaign_id"] == CAMPAIGN
    assert config["source_revision"] == REVISION
    assert config["telegram"]["expected_chat_identity_digest"] == CHAT_DIGEST
    assert config["expected_contract_path"] == str(contract_path)
    assert config["artifact_root"] == str(artifact_root)
    assert config["phone_status_directory"] == str(artifact_root / "phone")
    config_path = tmp_path / "controller-config.json"
    atomic_json(config_path, config)
    runtime = cli.load_runtime(config_path, keychain=_keychain())
    assert runtime.phone_json_path == artifact_root / "phone/GLM52_PHONE_STATUS.json"
    assert runtime.contract_phone_status_path == "phone/GLM52_PHONE_STATUS.json"


def test_contract_derived_config_refuses_synthetic_without_explicit_opt_in(
    tmp_path,
) -> None:
    contract_path = tmp_path / "expected-contract.json"
    atomic_json(contract_path, _contract())
    with pytest.raises(cli.CliError, match="synthetic contract requires explicit"):
        cli.make_config_from_contract(
            expected_contract_path=contract_path,
            controller_root=tmp_path / "controller",
            artifact_root=tmp_path,
        )


def test_status_is_deterministic_atomic_sealed_and_secret_free(tmp_path):
    runtime, config_path, keychain = _setup(tmp_path)
    first = cli.run_command(runtime, "status")
    json_before = runtime.phone_json_path.read_bytes()
    markdown_before = runtime.phone_markdown_path.read_bytes()
    second = cli.run_command(runtime, "status")

    assert first == second
    assert first["overall_status"] == "RED"
    assert first["controller"]["live_worker_lease_ok"] is False
    assert first["controller"]["state"] == "PRECHECK"
    assert first["operator_control"]["effective_action"] == cli.ACTION_RUN
    assert first["operator_control"]["event_count"] == 0
    assert runtime.phone_json_path.read_bytes() == json_before
    assert runtime.phone_markdown_path.read_bytes() == markdown_before
    assert read_sealed_json(runtime.phone_json_path) == first
    verify_sealed(first)
    assert cli.main(
        ["--config", str(config_path), "status"], keychain=keychain
    ) == 3

    worker = runtime.controller()
    with worker:
        live_first = cli.run_command(runtime, "status")
        live_second = cli.run_command(runtime, "status")
        assert live_first == live_second
        assert live_first["overall_status"] == "GREEN"
        assert live_first["controller"]["live_worker_lease_ok"] is True
        assert cli.main(
            ["--config", str(config_path), "status"], keychain=keychain
        ) == 0

    persisted = b"\n".join(
        path.read_bytes()
        for path in (
            config_path,
            runtime.expected_contract_path,
            runtime.phone_json_path,
            runtime.phone_markdown_path,
        )
    )
    assert KEY not in persisted
    assert base64.b64encode(KEY) not in persisted
    assert base64.urlsafe_b64encode(KEY) not in persisted
    assert EVIDENCE_KEY not in persisted
    assert base64.b64encode(EVIDENCE_KEY) not in persisted
    assert base64.urlsafe_b64encode(EVIDENCE_KEY) not in persisted
    assert GROUNDING_KEY not in persisted
    assert base64.b64encode(GROUNDING_KEY) not in persisted
    assert base64.urlsafe_b64encode(GROUNDING_KEY) not in persisted
    assert str(CHAT_ID).encode() not in persisted
    assert b"receipt_hmac_key_env" not in persisted


def test_config_and_key_are_required_and_tamper_is_rejected(tmp_path):
    runtime, config_path, keychain = _setup(tmp_path)
    with pytest.raises(cli.CliError, match="not configured"):
        cli.load_runtime(config_path, keychain=FakeKeychain())
    with pytest.raises(cli.CliError, match="invalid"):
        cli.load_runtime(
            config_path,
            keychain=FakeKeychain({
                gt.CHAT_SERVICE: str(CHAT_ID),
                gt.HMAC_SERVICE: "not base64!",
            }),
        )
    wrong_key_runtime = cli.load_runtime(
        config_path,
        keychain=_keychain(b"z" * 32),
    )
    # With no signed control event yet, a different key cannot be detected.  Once an
    # intent exists, replay is authenticated and the wrong key turns status red.
    cli.run_command(runtime, "stop")
    assert cli.build_phone_status(wrong_key_runtime)["overall_status"] == "RED"
    with pytest.raises(cli.CliError, match="controller HMAC authentication failed"):
        cli.run_command(wrong_key_runtime, "resume")
    wrong_evidence_runtime = cli.load_runtime(
        config_path,
        keychain=_keychain(evidence_key=b"x" * 32),
    )
    with pytest.raises(cli.CliError, match="invalid producer authentication"):
        cli.run_command(wrong_evidence_runtime, "status")

    config = json.loads(config_path.read_text())
    config["controller_epoch"] = "tampered-controller-epoch"
    atomic_json(config_path, config)
    with pytest.raises(cli.CliError, match="seal mismatch"):
        cli.load_runtime(config_path, keychain=keychain)


def test_grounding_provider_is_sealed_exact_and_runtime_exposes_grounder(tmp_path):
    runtime, config_path, keychain = _setup(tmp_path)
    config = read_sealed_json(config_path)
    assert config["schema"] == cli.CONFIG_SCHEMA
    assert config["grounding_auth"] == {
        "credential_provider": "macOS Keychain",
        "producer_hmac_key_service": gga.GROUNDING_HMAC_SERVICE,
        "keychain_account": gga.KEYCHAIN_ACCOUNT,
    }
    assert isinstance(runtime.grounding_auth, gga.ProducerAuthenticator)
    assert "redacted" in repr(runtime.grounding_auth)

    tampered = json.loads(config_path.read_text())
    tampered["grounding_auth"]["keychain_account"] = "attacker-selected-account"
    tampered.pop("seal_sha256")
    alternate_path = tmp_path / "tampered-grounding-config.json"
    atomic_json(alternate_path, seal(tampered))
    with pytest.raises(cli.CliError, match="fixed grounding Keychain service/account"):
        cli.load_runtime(alternate_path, keychain=keychain)


def test_grounding_credential_absent_or_malformed_fails_runtime_load(tmp_path):
    _, config_path, _ = _setup(tmp_path)
    with pytest.raises(cli.CliError, match="filesystem observation authentication is not configured"):
        cli.load_runtime(config_path, keychain=_keychain(grounding_key=None))

    malformed = _keychain()
    malformed.values[gga.GROUNDING_HMAC_SERVICE] = "not-base64!"
    with pytest.raises(cli.CliError, match="filesystem observation authentication is invalid"):
        cli.load_runtime(config_path, keychain=malformed)


@pytest.mark.parametrize(
    ("telegram_key", "evidence_key", "grounding_key"),
    (
        (b"r" * 32, b"e" * 32, b"r" * 32),
        (b"t" * 32, b"r" * 32, b"r" * 32),
        (b"r" * 32, b"r" * 32, b"g" * 32),
    ),
)
def test_all_authentication_role_keys_must_be_pairwise_distinct(
    tmp_path, telegram_key, evidence_key, grounding_key
):
    _, config_path, _ = _setup(tmp_path)
    with pytest.raises(cli.CliError, match="must be pairwise distinct"):
        cli.load_runtime(
            config_path,
            keychain=_keychain(
                key=telegram_key,
                evidence_key=evidence_key,
                grounding_key=grounding_key,
            ),
        )


def test_phone_output_is_contract_bound_and_existing_tamper_is_not_overwritten(tmp_path):
    runtime, config_path, keychain = _setup(tmp_path)
    cli.run_command(runtime, "status")
    phone = json.loads(runtime.phone_json_path.read_text())
    phone["overall_status"] = "GREEN_BUT_TAMPERED"
    atomic_json(runtime.phone_json_path, phone)
    tampered = runtime.phone_json_path.read_bytes()
    with pytest.raises(cli.CliError, match="refusing to overwrite invalid phone status"):
        cli.run_command(runtime, "status")
    assert runtime.phone_json_path.read_bytes() == tampered
    with pytest.raises(cli.CliError, match="refusing to overwrite invalid phone status"):
        cli.run_command(runtime, "stop")
    assert not runtime.control_log_path.exists()

    # A validly self-sealed config still cannot redirect the contract-bound phone file.
    config = json.loads(config_path.read_text())
    config["phone_status_directory"] = str(tmp_path / "redirected")
    config.pop("seal_sha256")
    atomic_json(config_path, seal(config))
    with pytest.raises(cli.CliError, match="does not match the COMPLETE contract path"):
        cli.load_runtime(config_path, keychain=keychain)


def test_validly_resealed_phone_tamper_without_producer_key_is_rejected(tmp_path):
    runtime, _, _ = _setup(tmp_path)
    original = cli.run_command(runtime, "status")
    body = {
        key: json.loads(json.dumps(value))
        for key, value in original.items()
        if key != "seal_sha256"
    }
    body["overall_status"] = "GREEN"
    body["status"] = "GREEN"
    atomic_json(runtime.phone_json_path, seal(body))
    tampered = runtime.phone_json_path.read_bytes()
    with pytest.raises(cli.CliError, match="invalid producer authentication"):
        cli.run_command(runtime, "status")
    assert runtime.phone_json_path.read_bytes() == tampered


def test_phone_json_commit_recovers_from_crash_between_json_and_markdown(
    tmp_path, monkeypatch
):
    runtime, _, _ = _setup(tmp_path)
    baseline = cli.run_command(runtime, "status")
    cli.run_command(runtime, "stop")
    markdown_before_crash = runtime.phone_markdown_path.read_bytes()

    real_atomic_text = cli.atomic_text

    def crash_after_json(_path, _text):
        raise OSError("injected crash after JSON commit")

    monkeypatch.setattr(cli, "atomic_text", crash_after_json)
    with pytest.raises(OSError, match="injected crash"):
        cli.run_command(runtime, "resume")
    committed = read_sealed_json(runtime.phone_json_path)
    assert committed["operator_control"]["requested_action"] == cli.ACTION_RUN
    assert runtime.phone_markdown_path.read_bytes() == markdown_before_crash

    monkeypatch.setattr(cli, "atomic_text", real_atomic_text)
    repaired = cli.run_command(runtime, "status")
    assert repaired == committed
    assert runtime.phone_markdown_path.read_text(encoding="utf-8") == (
        cli._render_phone_markdown(repaired)
    )
    assert repaired["seal_sha256"] != baseline["seal_sha256"]


def test_phone_recovers_when_crash_precedes_json_commit(tmp_path, monkeypatch):
    runtime, _, _ = _setup(tmp_path)
    baseline = cli.run_command(runtime, "status")
    baseline_bytes = runtime.phone_json_path.read_bytes()
    real_atomic_json = cli.atomic_json

    def crash_before_json(path, value):
        if path == runtime.phone_json_path:
            raise OSError("injected crash before JSON commit")
        return real_atomic_json(path, value)

    monkeypatch.setattr(cli, "atomic_json", crash_before_json)
    with pytest.raises(OSError, match="injected crash"):
        cli.run_command(runtime, "stop")
    assert runtime.phone_json_path.read_bytes() == baseline_bytes
    assert cli.OperatorControlJournal(runtime).replay()["requested_action"] == (
        cli.ACTION_STOP
    )

    monkeypatch.setattr(cli, "atomic_json", real_atomic_json)
    recovered = cli.run_command(runtime, "stop")
    assert recovered["changed"] is False
    assert recovered["idempotent"] is True
    assert recovered["phone_status"]["status"] == "RED"
    assert recovered["phone_status"]["controller"]["live_worker_lease_ok"] is False
    assert recovered["phone_status"]["seal_sha256"] != baseline["seal_sha256"]


def test_operator_transition_law_idempotence_and_safe_state_gate(tmp_path):
    precheck, _, _ = _setup(tmp_path / "precheck")
    with pytest.raises(cli.CliError, match="pause-after-window is legal only"):
        cli.run_command(precheck, "pause-after-window")

    first_stop = cli.run_command(precheck, "stop")
    repeated_stop = cli.run_command(precheck, "stop")
    resumed = cli.run_command(precheck, "resume")
    assert first_stop["changed"] is True
    assert repeated_stop["changed"] is False
    assert repeated_stop["idempotent"] is True
    assert repeated_stop["control_event_count"] == 1
    assert resumed["effective_action"] == cli.ACTION_RUN
    assert resumed["control_event_count"] == 2

    frozen, _, _ = _setup(tmp_path / "frozen", reach_freeze=True)
    paused = cli.run_command(frozen, "pause-after-window")
    paused_again = cli.run_command(frozen, "pause-after-window")
    stopped = cli.run_command(frozen, "stop")
    assert paused["effective_action"] == cli.ACTION_PAUSE
    assert paused_again["changed"] is False
    assert paused_again["control_event_count"] == 1
    assert stopped["effective_action"] == cli.ACTION_STOP
    with pytest.raises(cli.CliError, match="illegal operator transition"):
        cli.run_command(frozen, "pause-after-window")
    assert cli.run_command(frozen, "resume")["effective_action"] == cli.ACTION_RUN
    final = cli.run_command(frozen, "pause-after-window")
    assert final["control_event_count"] == 4
    assert cli.OperatorControlJournal(frozen).replay()["journal_ok"] is True


def test_worker_applies_requests_only_at_authenticated_safe_points(tmp_path):
    runtime, _, _ = _setup(tmp_path, reach_freeze=True)
    acknowledgements = cli.WorkerControlAcknowledgements(runtime)
    detached = runtime.controller()
    with pytest.raises(gs.StateError, match="singleton lease"):
        acknowledgements.poll(detached, safe_point="BETWEEN_RANGE_READS")

    worker = runtime.controller()
    with worker:
        requested = cli.run_command(runtime, "pause-after-window")
        assert requested["requested_action"] == cli.ACTION_PAUSE
        assert requested["applied_action"] == cli.ACTION_RUN
        assert requested["application_state"] == "REQUESTED_NOT_APPLIED"
        assert requested["phone_status"]["overall_status"] == "AMBER_CONTROL_PENDING"

        mid_window = acknowledgements.poll(
            worker, safe_point="BETWEEN_RANGE_READS"
        )
        assert mid_window["application_state"] == "REQUESTED_PENDING_SAFE_POINT"
        assert mid_window["worker_directive"] == "CONTINUE_CURRENT_WINDOW"
        assert mid_window["event_count"] == 0

        paused = acknowledgements.poll(
            worker, safe_point="AFTER_WINDOW_EVICTION"
        )
        assert paused["changed"] is True
        assert paused["applied_action"] == cli.ACTION_PAUSE
        assert paused["worker_directive"] == "PAUSE"
        repeat = acknowledgements.poll(
            worker, safe_point="AFTER_WINDOW_EVICTION"
        )
        assert repeat["changed"] is False
        assert repeat["event_count"] == 1

        synchronized = cli.run_command(runtime, "status")
        assert synchronized["overall_status"] == "GREEN"
        assert synchronized["operator_control"]["application_state"] == "IN_SYNC"

        resumed_request = cli.run_command(runtime, "resume")
        assert resumed_request["application_state"] == "REQUESTED_NOT_APPLIED"
        resumed = acknowledgements.poll(worker, safe_point="BEFORE_WINDOW_FETCH")
        assert resumed["applied_action"] == cli.ACTION_RUN
        assert resumed["worker_directive"] == "CONTINUE"

        cli.run_command(runtime, "stop")
        stopped = acknowledgements.poll(worker, safe_point="BETWEEN_RANGE_READS")
        assert stopped["applied_action"] == cli.ACTION_STOP
        assert stopped["worker_directive"] == "STOP"
        assert stopped["event_count"] == 3
        assert cli.run_command(runtime, "status")["overall_status"] == "GREEN"

    assert cli.run_command(runtime, "status")["overall_status"] == "RED"


def test_tampered_worker_acknowledgement_fails_phone_and_worker_closed(tmp_path):
    runtime, _, _ = _setup(tmp_path, reach_freeze=True)
    cli.run_command(runtime, "stop")
    worker = runtime.controller()
    acknowledgements = cli.WorkerControlAcknowledgements(runtime)
    with worker:
        acknowledgements.poll(worker, safe_point="BEFORE_HEAVY_COMPUTE")
    event = json.loads(runtime.control_applied_log_path.read_text())
    event["payload"]["action"] = cli.ACTION_RUN
    runtime.control_applied_log_path.write_bytes(canonical(event) + b"\n")

    status = cli.run_command(runtime, "status")
    assert status["overall_status"] == "RED"
    assert status["operator_control"]["applied"]["applied_action"] == cli.ACTION_STOP
    with worker:
        with pytest.raises(cli.CliError, match="event chain hash mismatch"):
            acknowledgements.poll(worker, safe_point="BETWEEN_RANGE_READS")


def test_tampered_controller_checkpoint_reports_red_and_blocks_mutation(tmp_path):
    runtime, config_path, keychain = _setup(tmp_path)
    checkpoint_path = runtime.controller_root / "GLM52_CONTROLLER_CHECKPOINT.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["state"] = "COMPLETE"
    atomic_json(checkpoint_path, checkpoint)  # retain the old seal deliberately

    status = cli.run_command(runtime, "status")
    assert status["overall_status"] == "RED"
    assert status["controller"]["state"] is None
    assert status["controller"]["checkpoint_seal_ok"] is False
    assert read_sealed_json(runtime.phone_json_path)["overall_status"] == "RED"
    with pytest.raises(cli.CliError, match="durable state is not green"):
        cli.run_command(runtime, "stop")
    assert cli.main(
        ["--config", str(config_path), "status"], keychain=keychain
    ) == 3


def test_malformed_or_tampered_control_journal_is_fail_closed(tmp_path):
    runtime, _, _ = _setup(tmp_path)
    cli.run_command(runtime, "stop")
    raw = runtime.control_log_path.read_bytes()
    event = json.loads(raw)
    event["payload"]["action"] = cli.ACTION_PAUSE
    runtime.control_log_path.write_bytes(canonical(event) + b"\n")

    status = cli.run_command(runtime, "status")
    assert status["overall_status"] == "RED"
    assert status["operator_control"]["journal_ok"] is False
    assert status["operator_control"]["effective_action"] == cli.ACTION_STOP
    with pytest.raises(cli.CliError, match="event chain hash mismatch"):
        cli.run_command(runtime, "resume")

    runtime.control_log_path.write_bytes(b'{"torn":true}')
    torn = cli.run_command(runtime, "status")
    assert torn["overall_status"] == "RED"
    assert "unsealed/torn JSONL tail" in torn["operator_control_error"]


def test_cli_errors_are_machine_readable_and_do_not_echo_secret(tmp_path, capsys):
    _, config_path, _ = _setup(tmp_path)
    secret = base64.urlsafe_b64encode(KEY).decode("ascii")
    code = cli.main(
        ["--config", str(config_path), "status"],
        keychain=FakeKeychain({
            gt.CHAT_SERVICE: str(CHAT_ID),
            gt.HMAC_SERVICE: "invalid!!!",
        }),
    )
    captured = capsys.readouterr()
    assert code == 2
    error = json.loads(captured.err)
    assert error["status"] == "ERROR"
    assert error["error"] == "CliError"
    assert secret not in captured.err
    assert KEY.decode("ascii") not in captured.err
