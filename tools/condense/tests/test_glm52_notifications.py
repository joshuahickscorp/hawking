#!/usr/bin/env python3.12
"""Adversarial tests for the controller-bound GLM-5.2 notification journal."""
from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import shutil
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest


CONDENSE = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for location in (CONDENSE, TESTS):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import glm52_notifications as gn  # noqa: E402
import test_glm52_state as state_tests  # noqa: E402
from glm52_common import canonical, seal, utc_now  # noqa: E402


BOT_ID = 987654321


class Roles:
    def __init__(self) -> None:
        self.producer = gn.NotificationProducerSigner(b"P" * 32)
        self.receipt = gn.NotificationReceiptSigner(
            b"R" * 32,
            expected_chat_identity_digest=state_tests.CHAT_DIGEST,
            expected_bot_identity_digest=gn.telegram_bot_identity_digest(BOT_ID),
            producer_verifier=self.producer.verifier,
        )
        self.operator = gn.NotificationReconciliationSigner(b"O" * 32)


@contextmanager
def _release_controller(tmp_path: Path):
    controller = state_tests._controller(tmp_path)
    controller.acquire()
    try:
        state_tests._boot(controller)
        state_tests._transition(controller, "CLOSE_KIMI", 1)
        state_tests._transition(controller, "RELEASE_KIMI_SOURCE", 2)
        yield controller
    finally:
        controller.close()


@contextmanager
def _fetch_controller(tmp_path: Path):
    controller = state_tests._controller(tmp_path)
    controller.acquire()
    try:
        state_tests._reach_freeze(controller)
        controller.declare_window(**state_tests._window_args())
        state_tests._transition(
            controller,
            "FETCH_WINDOW",
            20,
            payload={"window_id": "window-0001"},
        )
        yield controller
    finally:
        controller.close()


def _storage(tmp_path: Path) -> Path:
    parent = tmp_path / "notification-storage"
    parent.mkdir(mode=0o700)
    return parent / "notifications.jsonl"


def _journal(
    path: Path,
    controller,
    roles: Roles,
    *,
    fault_injector=None,
) -> gn.NotificationJournal:
    return gn.NotificationJournal(
        path.resolve(),
        controller=controller,
        producer_signer=roles.producer,
        receipt_verifier=roles.receipt.verifier,
        reconciliation_verifier=roles.operator.verifier,
        expected_bot_identity_digest=gn.telegram_bot_identity_digest(BOT_ID),
        fault_injector=fault_injector,
    )


def _bot_response(intent: dict[str, Any], *, message_id: int = 7) -> dict[str, Any]:
    return {
        "ok": True,
        "result": {
            "message_id": message_id,
            "from": {"id": BOT_ID, "is_bot": True, "first_name": "GLM status"},
            "chat": {"id": state_tests.CHAT_ID, "type": "private"},
            "text": intent["rendered_message"],
        },
    }


def _receipt(
    roles: Roles,
    intent: dict[str, Any],
    started: dict[str, Any],
    *,
    message_id: int = 7,
) -> dict[str, Any]:
    return roles.receipt.make_delivery_receipt(
        intent,
        started,
        bot_api_response=_bot_response(intent, message_id=message_id),
        http_status=200,
        delivered_at=utc_now(),
    )


def test_roles_are_ed25519_distinct_and_private_keys_do_not_enter_journal(tmp_path: Path) -> None:
    roles = Roles()
    assert len({roles.producer.key_id, roles.receipt.key_id, roles.operator.key_id}) == 3
    assert "PPPP" not in repr(roles.producer)
    with _release_controller(tmp_path) as controller:
        journal = _journal(_storage(tmp_path), controller, roles)
        assert journal.receipt_verifier.role == "receipt"
        assert journal.reconciliation_verifier.role == "reconciliation"
        assert not any(
            isinstance(value, gn.NotificationReceiptSigner)
            or isinstance(value, gn.NotificationReconciliationSigner)
            for value in vars(journal).values()
        )
    same = gn.NotificationVerifier("receipt", roles.producer.verifier.public_key)
    with _release_controller(tmp_path / "same") as controller:
        with pytest.raises(gn.NotificationError, match="distinct"):
            gn.NotificationJournal(
                _storage(tmp_path / "same").resolve(),
                controller=controller,
                producer_signer=roles.producer,
                receipt_verifier=same,
                reconciliation_verifier=roles.operator.verifier,
                expected_bot_identity_digest=gn.telegram_bot_identity_digest(BOT_ID),
            )


def test_prepare_has_no_caller_facts_status_anchor_or_timestamp(tmp_path: Path) -> None:
    roles = Roles()
    with _release_controller(tmp_path) as controller:
        journal = _journal(_storage(tmp_path), controller, roles)
        with pytest.raises(TypeError):
            journal.prepare(  # type: ignore[call-arg]
                event_kind=gn.KIMI_RELEASE,
                facts={"release_status": "forged"},
                canonical_status={},
                controller_anchor={},
            )
        intent = journal.prepare(event_kind=gn.KIMI_RELEASE)
        status = intent["canonical_status"]
        assert intent["facts"]["controller_state"] == "RELEASE_KIMI_SOURCE"
        assert status["state"] == "RELEASE_KIMI_SOURCE"
        assert status["source_coverage"]["total_logical_bytes"] == 5_000_000_000
        assert status["process"]["lease_held"] is True
        assert intent["controller_snapshot_sha256"] == status["controller_snapshot_sha256"]
        assert gn.validate_notification_intent(intent, roles.producer.verifier) == intent
        assert "source_coverage=" in intent["rendered_message"]
        assert "evidence_seals=" in intent["rendered_message"]


def test_unsupported_and_premature_milestones_fail_closed(tmp_path: Path) -> None:
    roles = Roles()
    with _release_controller(tmp_path) as controller:
        journal = _journal(_storage(tmp_path), controller, roles)
        with pytest.raises(gn.NotificationError, match="dedicated grounded evidence schema"):
            journal.prepare(event_kind=gn.FAULT)
        with pytest.raises(gn.NotificationError, match="unsupported in verified controller state"):
            journal.prepare(event_kind=gn.FINAL_RESULT)
        with pytest.raises(gn.NotificationError, match="crossing detection"):
            journal.prepare(event_kind=gn.SOURCE_COVERAGE_5_PERCENT)


def test_stream_start_and_all_coverage_crossings_come_from_window_ledger(
    tmp_path: Path,
) -> None:
    roles = Roles()
    with _fetch_controller(tmp_path) as controller:
        journal = _journal(_storage(tmp_path), controller, roles)
        with pytest.raises(gn.NotificationError, match="premature"):
            journal.prepare(event_kind=gn.SOURCE_STREAM_STARTED)
        controller.advance_window(
            "window-0001",
            "FETCHING",
            claim_id="notification:test:window:fetching",
            patch={"download_start": "2026-07-21T12:00:00Z"},
            source_coverage={state_tests.SHARD: "FETCHING"},
        )
        stream = journal.prepare(event_kind=gn.SOURCE_STREAM_STARTED)
        assert stream["facts"]["subject"]["phase"] == "FETCHING"
        assert stream["facts"]["subject"]["download_start"] == "2026-07-21T12:00:00Z"
        started = journal.start_delivery(stream)
        journal.commit(stream, _receipt(roles, stream, started))
        controller.advance_window(
            "window-0001",
            "FETCHED",
            claim_id="notification:test:window:fetched",
            patch={
                "download_end": "2026-07-21T12:01:00Z",
                "bytes_transferred": 5_000_000_100,
                "transfer_accounting": {
                    "new_fetch_network_bytes": 5_000_000_000,
                    "refetch_network_bytes": 0,
                    "protocol_overhead_bytes": 100,
                },
            },
            source_coverage={state_tests.SHARD: "FETCHED"},
        )
        state_tests._transition(
            controller,
            "VERIFY_WINDOW",
            21,
            payload={"window_id": "window-0001"},
        )
        controller.advance_window(
            "window-0001",
            "VERIFIED",
            claim_id="notification:test:window:verified",
            patch={
                "hash_verification": {
                    "status": "VERIFIED",
                    "verified_shards": [state_tests.SHARD],
                }
            },
            source_coverage={state_tests.SHARD: "HASH_VERIFIED"},
            tensor_coverage={state_tests.TENSOR: "SOURCE_VERIFIED"},
        )
        crossings = journal.prepare_coverage_crossings()
        assert [
            intent["facts"]["subject"]["threshold_percent"] for intent in crossings
        ] == list(range(5, 101, 5))
        assert journal.prepare_coverage_crossings() == []


def test_exact_prepared_send_started_committed_lifecycle_and_redacted_receipt(
    tmp_path: Path,
) -> None:
    roles = Roles()
    with _release_controller(tmp_path) as controller:
        journal = _journal(_storage(tmp_path), controller, roles)
        intent = journal.prepare(event_kind=gn.KIMI_RELEASE)
        assert journal.pending()[0]["safe_to_send"] is True
        with pytest.raises(gn.NotificationError, match="SEND_STARTED"):
            journal.commit(intent, {})
        started = journal.start_delivery(intent)
        assert started["kind"] == gn.SEND_STARTED and started["attempt"] == 1
        assert journal.pending()[0]["ambiguous"] is True
        receipt = _receipt(roles, intent, started)
        rendered_receipt = canonical(receipt)
        assert str(state_tests.CHAT_ID).encode() not in rendered_receipt
        assert b'"bot_api_response"' not in rendered_receipt
        committed = journal.commit(intent, receipt)
        assert committed["kind"] == gn.COMMITTED
        replay = journal.replay()
        assert replay.lifecycle[intent["seal_sha256"]] == gn.COMMITTED
        assert replay.latest_delivered_status == intent["canonical_status"]
        assert journal.pending() == []
        with pytest.raises(gn.NotificationError, match="already committed"):
            journal.commit(intent, receipt)


def test_receipt_binds_attempt_start_sequence_hash_and_both_telegram_identities(
    tmp_path: Path,
) -> None:
    roles = Roles()
    with _release_controller(tmp_path) as controller:
        journal = _journal(_storage(tmp_path), controller, roles)
        intent = journal.prepare(event_kind=gn.KIMI_RELEASE)
        started = journal.start_delivery(intent)
        wrong_chat = _bot_response(intent)
        wrong_chat["result"]["chat"]["id"] = -999
        with pytest.raises(gn.NotificationError, match="different configured chat"):
            roles.receipt.make_delivery_receipt(
                intent, started, bot_api_response=wrong_chat, http_status=200
            )
        wrong_bot = _bot_response(intent)
        wrong_bot["result"]["from"]["id"] = BOT_ID + 1
        with pytest.raises(gn.NotificationError, match="different bot"):
            roles.receipt.make_delivery_receipt(
                intent, started, bot_api_response=wrong_bot, http_status=200
            )
        receipt = _receipt(roles, intent, started)
        tampered = copy.deepcopy(receipt)
        tampered["send_started_seq"] += 1
        body = {
            key: value for key, value in tampered.items()
            if key not in {"receipt_signature", "seal_sha256"}
        }
        tampered["receipt_signature"] = roles.receipt._sign(
            gn.BOT_RECEIPT_AUTH_SCHEMA, body
        )
        tampered = seal(tampered)
        with pytest.raises(gn.NotificationError, match="exact send attempt"):
            journal.commit(intent, tampered)


def test_real_utc_calendar_and_timestamp_order_are_enforced(tmp_path: Path) -> None:
    roles = Roles()
    with pytest.raises(gn.NotificationError, match="real UTC"):
        gn._parse_utc("2026-02-30T12:00:00Z", "test")
    with _release_controller(tmp_path) as controller:
        journal = _journal(_storage(tmp_path), controller, roles)
        intent = journal.prepare(event_kind=gn.KIMI_RELEASE)
        started = journal.start_delivery(intent)
        with pytest.raises(gn.NotificationError, match="prepare/send/deliver order"):
            roles.receipt.make_delivery_receipt(
                intent,
                started,
                bot_api_response=_bot_response(intent),
                http_status=200,
                delivered_at="2020-01-01T00:00:00Z",
            )


def test_ambiguity_requires_separately_signed_exact_reconciliation(tmp_path: Path) -> None:
    roles = Roles()
    with _release_controller(tmp_path) as controller:
        journal = _journal(_storage(tmp_path), controller, roles)
        intent = journal.prepare(event_kind=gn.KIMI_RELEASE)
        journal.start_delivery(intent)
        with pytest.raises(gn.NotificationError, match="reconciliation"):
            journal.start_delivery(intent)
        journal.mark_ambiguous(
            intent,
            reason="sender lost the response after its durable send-start record",
        )
        challenge = journal.make_reconciliation_challenge(intent)
        authorization = roles.operator.authorize(
            challenge,
            authorization_id="operator-reconciliation-0001",
            reason="authenticated Bot history proves the exact message is absent",
        )
        tampered = copy.deepcopy(challenge)
        tampered["authorized_attempt"] += 1
        tampered = seal(tampered)
        with pytest.raises(gn.NotificationError):
            journal.authorize_replay(intent, tampered, authorization)
        authorized = journal.authorize_replay(intent, challenge, authorization)
        assert authorized["kind"] == gn.REPLAY_AUTHORIZED
        started = journal.start_delivery(intent)
        assert started["attempt"] == 2
        journal.commit(intent, _receipt(roles, intent, started, message_id=8))


def test_controller_checkpoint_binds_every_notification_head(tmp_path: Path) -> None:
    roles = Roles()
    with _release_controller(tmp_path) as controller:
        journal = _journal(_storage(tmp_path), controller, roles)
        genesis = controller.notification_journal_binding()
        assert genesis is not None and genesis["entry_count"] == 0
        intent = journal.prepare(event_kind=gn.KIMI_RELEASE)
        first = controller.notification_journal_binding()
        assert first["entry_count"] == 1
        assert first["previous_binding_sha256"] == genesis["seal_sha256"]
        started = journal.start_delivery(intent)
        second = controller.notification_journal_binding()
        assert second["entry_count"] == 2
        assert second["head_chain_sha256"] == started["chain_sha256"]
        checkpoint = controller.resume()
        assert checkpoint["notification_journal"] == second
        assert controller.events.verified_events()[-1]["kind"] == "NOTIFICATION_HEAD"


def test_head_and_journal_deletion_or_prefix_rollback_are_detected(tmp_path: Path) -> None:
    roles = Roles()
    with _release_controller(tmp_path) as controller:
        path = _storage(tmp_path)
        journal = _journal(path, controller, roles)
        genesis_head = journal.head_path.read_bytes()
        intent = journal.prepare(event_kind=gn.KIMI_RELEASE)
        prepared_journal = journal.path.read_bytes()
        journal.start_delivery(intent)
        journal.path.write_bytes(prepared_journal)
        journal.head_path.write_bytes(genesis_head)
        os.chmod(journal.path, 0o600)
        os.chmod(journal.head_path, 0o600)
        with pytest.raises(
            gn.NotificationError, match="rolled back|forked|truncation|non-current head"
        ):
            journal.replay()

    with _release_controller(tmp_path / "deleted") as controller:
        journal = _journal(_storage(tmp_path / "deleted"), controller, Roles())
        journal.head_path.unlink()
        with pytest.raises(gn.NotificationError, match="lacks its head"):
            journal.replay()


def test_parent_journal_lock_and_fifo_replacement_are_detected(tmp_path: Path) -> None:
    roles = Roles()
    with _release_controller(tmp_path) as controller:
        path = _storage(tmp_path)
        journal = _journal(path, controller, roles)
        original_parent = path.parent.with_name(path.parent.name + "-old")
        path.parent.rename(original_parent)
        path.parent.mkdir(mode=0o700)
        for name in (path.name, journal.head_path.name, journal.lock_path.name):
            shutil.copy2(original_parent / name, path.parent / name, follow_symlinks=False)
        with pytest.raises(gn.NotificationError, match="parent identity changed"):
            journal.replay()

    with _release_controller(tmp_path / "fifo") as controller:
        journal = _journal(_storage(tmp_path / "fifo"), controller, Roles())
        moved = journal.lock_path.with_name("old.lock")
        journal.lock_path.rename(moved)
        os.mkfifo(journal.lock_path, 0o600)
        assert stat.S_ISFIFO(journal.lock_path.lstat().st_mode)
        with pytest.raises(gn.NotificationError, match="regular file"):
            journal.replay()


@pytest.mark.parametrize(
    "fault_stage",
    ["after_journal_fsync", "after_head_fsync_before_controller_anchor"],
)
def test_exact_one_entry_crash_tail_is_recovered(
    tmp_path: Path, fault_stage: str
) -> None:
    roles = Roles()
    with _release_controller(tmp_path) as controller:
        path = _storage(tmp_path)
        armed = {"value": False}

        def fault(stage: str) -> None:
            if armed["value"] and stage == fault_stage:
                raise RuntimeError("simulated power loss")

        journal = _journal(path, controller, roles, fault_injector=fault)
        armed["value"] = True
        with pytest.raises(RuntimeError, match="power loss"):
            journal.prepare(event_kind=gn.KIMI_RELEASE)
        recovered = _journal(path, controller, roles)
        replay = recovered.replay()
        assert len(replay.entries) == 1
        assert next(iter(replay.lifecycle.values())) == gn.PREPARED
        assert controller.notification_journal_binding()["entry_count"] == 1


def test_resealed_entry_tamper_and_clean_tail_truncation_fail(tmp_path: Path) -> None:
    roles = Roles()
    with _release_controller(tmp_path) as controller:
        journal = _journal(_storage(tmp_path), controller, roles)
        intent = journal.prepare(event_kind=gn.KIMI_RELEASE)
        journal.start_delivery(intent)
        lines = journal.path.read_bytes().splitlines()
        entry = json.loads(lines[-1])
        entry["recorded_at"] = "2026-07-21T23:59:59Z"
        chain_body = gn._entry_chain_body(entry)
        entry["chain_sha256"] = gn._sha256(chain_body)
        signed = {
            key: value for key, value in entry.items()
            if key not in {"producer_signature", "seal_sha256"}
        }
        entry["producer_signature"] = roles.producer._sign(gn.ENTRY_AUTH_SCHEMA, signed)
        entry = seal(entry)
        journal.path.write_bytes(lines[0] + b"\n" + canonical(entry) + b"\n")
        os.chmod(journal.path, 0o600)
        with pytest.raises(gn.NotificationError, match="fork|head"):
            journal.replay()


def test_concurrent_duplicate_prepare_has_one_winner(tmp_path: Path) -> None:
    roles = Roles()
    with _release_controller(tmp_path) as controller:
        path = _storage(tmp_path)
        first = _journal(path, controller, roles)
        second = _journal(path, controller, roles)

        def attempt(journal: gn.NotificationJournal) -> str:
            try:
                journal.prepare(event_kind=gn.KIMI_RELEASE)
                return "prepared"
            except gn.NotificationError:
                return "rejected"

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(attempt, (first, second)))
        assert sorted(outcomes) == ["prepared", "rejected"]
        assert len(first.replay().entries) == 1


def _audit(path: Path, controller, roles: Roles) -> dict[str, Any]:
    return gn.audit_notification_outbox(
        path.resolve(),
        controller=controller,
        producer_verifier=roles.producer.verifier,
        receipt_verifier=roles.receipt.verifier,
        reconciliation_verifier=roles.operator.verifier,
        expected_bot_identity_digest=gn.telegram_bot_identity_digest(BOT_ID),
    )


def test_read_only_audit_replays_exact_lifecycle_without_private_keys(
    tmp_path: Path,
) -> None:
    roles = Roles()
    with _release_controller(tmp_path) as controller:
        path = _storage(tmp_path)
        journal = _journal(path, controller, roles)
        genesis = _audit(path, controller, roles)
        assert genesis["replay_status"] == "VERIFIED"
        assert genesis["entry_count"] == genesis["intent_count"] == 0
        assert genesis["unresolved_intent_count"] == 0
        assert genesis["audit_sha256"] == gn._sha256({
            key: value for key, value in genesis.items() if key != "audit_sha256"
        })

        intent = journal.prepare(event_kind=gn.KIMI_RELEASE)
        prepared = _audit(path, controller, roles)
        assert prepared["entry_count"] == prepared["intent_count"] == 1
        assert prepared["unresolved_intent_count"] == 1
        assert prepared["safe_to_send_intent_count"] == 1
        assert prepared["intents"][0]["lifecycle"] == gn.PREPARED

        started = journal.start_delivery(intent)
        sending = _audit(path, controller, roles)
        assert sending["ambiguous_intent_count"] == 1
        assert sending["intents"][0]["attempt"] == 1
        journal.commit(intent, _receipt(roles, intent, started))
        complete = _audit(path, controller, roles)
        assert complete["committed_intent_count"] == 1
        assert complete["unresolved_intent_count"] == 0
        assert complete["latest_delivered_status"] == intent["canonical_status"]
        assert complete["latest_delivered_status_sha256"] == gn._sha256(
            intent["canonical_status"]
        )


def test_read_only_audit_performs_no_write_repair_anchor_or_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles = Roles()
    with _release_controller(tmp_path) as controller:
        path = _storage(tmp_path)
        _journal(path, controller, roles)
        before = {
            "journal": path.read_bytes(),
            "head": path.with_name(path.name + ".head.json").read_bytes(),
            "checkpoint": controller.checkpoint_path.read_bytes(),
            "events": controller.events.path.read_bytes(),
        }

        def forbidden(*_args, **_kwargs):
            raise AssertionError("read-only audit attempted a mutating operation")

        monkeypatch.setattr(controller, "anchor_notification_journal_head", forbidden)
        monkeypatch.setattr(gn.os, "write", forbidden)
        monkeypatch.setattr(gn.os, "rename", forbidden)
        monkeypatch.setattr(gn.os, "fsync", forbidden)
        result = _audit(path, controller, roles)
        assert result["status"] == "PASS"
        assert path.read_bytes() == before["journal"]
        assert path.with_name(path.name + ".head.json").read_bytes() == before["head"]
        assert controller.checkpoint_path.read_bytes() == before["checkpoint"]
        assert controller.events.path.read_bytes() == before["events"]


@pytest.mark.parametrize(
    ("fault_stage", "message"),
    [
        ("after_journal_fsync", "crash-tail gap"),
        ("after_head_fsync_before_controller_anchor", "controller binding"),
    ],
)
def test_read_only_audit_rejects_crash_gaps_without_recovery(
    tmp_path: Path, fault_stage: str, message: str,
) -> None:
    roles = Roles()
    with _release_controller(tmp_path) as controller:
        path = _storage(tmp_path)
        armed = {"value": False}

        def fault(stage: str) -> None:
            if armed["value"] and stage == fault_stage:
                raise RuntimeError("simulated audit crash gap")

        journal = _journal(path, controller, roles, fault_injector=fault)
        armed["value"] = True
        with pytest.raises(RuntimeError, match="audit crash gap"):
            journal.prepare(event_kind=gn.KIMI_RELEASE)
        before = {
            "journal": journal.path.read_bytes(),
            "head": journal.head_path.read_bytes(),
            "checkpoint": controller.checkpoint_path.read_bytes(),
            "events": controller.events.path.read_bytes(),
        }
        with pytest.raises(gn.NotificationError, match=message):
            _audit(path, controller, roles)
        assert journal.path.read_bytes() == before["journal"]
        assert journal.head_path.read_bytes() == before["head"]
        assert controller.checkpoint_path.read_bytes() == before["checkpoint"]
        assert controller.events.path.read_bytes() == before["events"]


def test_read_only_audit_rejects_wrong_configuration_and_unheld_lease(
    tmp_path: Path,
) -> None:
    roles = Roles()
    path: Path
    controller = None
    with _release_controller(tmp_path) as active:
        controller = active
        path = _storage(tmp_path)
        _journal(path, active, roles)
        wrong = gn.NotificationProducerSigner(b"X" * 32)
        with pytest.raises(gn.NotificationError, match="configuration differs"):
            gn.audit_notification_outbox(
                path.resolve(),
                controller=active,
                producer_verifier=wrong.verifier,
                receipt_verifier=roles.receipt.verifier,
                reconciliation_verifier=roles.operator.verifier,
                expected_bot_identity_digest=gn.telegram_bot_identity_digest(BOT_ID),
            )
        with pytest.raises(gn.NotificationError, match="configuration differs"):
            gn.audit_notification_outbox(
                path.resolve(),
                controller=active,
                producer_verifier=roles.producer.verifier,
                receipt_verifier=roles.receipt.verifier,
                reconciliation_verifier=roles.operator.verifier,
                expected_bot_identity_digest="f" * 64,
            )
        with pytest.raises(TypeError):
            gn.audit_notification_outbox(  # type: ignore[call-arg]
                path.resolve(),
                controller=active,
                producer_verifier=roles.producer.verifier,
                receipt_verifier=roles.receipt.verifier,
                reconciliation_verifier=roles.operator.verifier,
                expected_bot_identity_digest=gn.telegram_bot_identity_digest(BOT_ID),
                replay_status="PASS",
            )
    assert controller is not None
    with pytest.raises(gn.NotificationError, match="lease"):
        _audit(path, controller, roles)


def test_read_only_audit_rejects_fifo_lock_and_controller_bound_head_rollback(
    tmp_path: Path,
) -> None:
    roles = Roles()
    with _release_controller(tmp_path) as controller:
        path = _storage(tmp_path)
        journal = _journal(path, controller, roles)
        genesis_head = journal.head_path.read_bytes()
        genesis_journal = journal.path.read_bytes()
        journal.prepare(event_kind=gn.KIMI_RELEASE)
        journal.head_path.write_bytes(genesis_head)
        journal.path.write_bytes(genesis_journal)
        os.chmod(journal.head_path, 0o600)
        os.chmod(journal.path, 0o600)
        with pytest.raises(gn.NotificationError, match="controller binding"):
            _audit(path, controller, roles)

    with _release_controller(tmp_path / "fifo-audit") as controller:
        path = _storage(tmp_path / "fifo-audit")
        journal = _journal(path, controller, Roles())
        journal.lock_path.rename(journal.lock_path.with_name("old.lock"))
        os.mkfifo(journal.lock_path, 0o600)
        with pytest.raises(gn.NotificationError, match="regular file"):
            _audit(path, controller, Roles())
