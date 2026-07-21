#!/usr/bin/env python3.12
"""Offline security tests for the GLM-5.2 Telegram module."""
from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import sys
import threading
from typing import Any, Mapping

import pytest


CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_telegram as gt  # noqa: E402
import glm52_state as gs  # noqa: E402
from glm52_common import canonical, seal  # noqa: E402


TOKEN = "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcd"
CHAT_ID = "123456789"
HMAC_RAW = b"H" * 32
HMAC_ENCODED = base64.urlsafe_b64encode(HMAC_RAW).decode("ascii")
DEDUPE = "d" * 64
SOURCE_REVISION = "a" * 40
CONTRACT_SHA = "b" * 64
CONTROLLER_EPOCH = "glm52-telegram-test-epoch"


class FakeKeychain:
    def __init__(self, values: Mapping[str, str] | None = None) -> None:
        self.values = dict(values or {})
        self.set_calls: list[tuple[str, str]] = []

    def get(self, service: str) -> str | None:
        return self.values.get(service)

    def set(self, service: str, value: str) -> None:
        self.set_calls.append((service, value))
        self.values[service] = value


class FakeTransport:
    def __init__(self, handler=None) -> None:
        self.handler = handler
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def call(self, token: str, method: str, payload: Mapping[str, Any]) -> gt.TelegramHTTPResponse:
        copied = dict(payload)
        self.calls.append((token, method, copied))
        if self.handler is not None:
            return self.handler(token, method, copied)
        raise AssertionError("unexpected fake Telegram call")


def _get_me(_token: str, method: str, _payload: dict) -> gt.TelegramHTTPResponse:
    assert method == "getMe"
    return gt.TelegramHTTPResponse(
        200,
        {"ok": True, "result": {"id": 55, "is_bot": True, "username": "glm52_bot"}},
    )


def _updates(chat_ids: list[int]) -> gt.TelegramHTTPResponse:
    result = []
    for offset, chat_id in enumerate(chat_ids):
        result.append({
            "update_id": 100 + offset,
            "message": {
                "message_id": 200 + offset,
                "from": {"id": chat_id, "is_bot": False},
                "chat": {"id": chat_id, "type": "private", "first_name": "secret-name"},
                "text": "hello",
            },
        })
    return gt.TelegramHTTPResponse(200, {"ok": True, "result": result})


def _configured_keychain() -> FakeKeychain:
    return FakeKeychain({
        gt.TOKEN_SERVICE: TOKEN,
        gt.CHAT_SERVICE: CHAT_ID,
        gt.HMAC_SERVICE: HMAC_ENCODED,
    })


def _status() -> dict[str, Any]:
    return {
        "state": "AUTOTUNE_XET",
        "source_coverage_percent": 12.5,
        "shards": {"fetched": 40, "verified": 39, "evicted": 20, "total": 282},
        "network_bytes": 123456,
        "throughput_bytes_per_second": 98765.5,
        "eta_seconds": 3600,
        "current": {"window": "W003", "layer": 12},
        "candidate_rates": ["0.98", "0.75", "0.50"],
        "best_metrics": {"cosine": 0.999, "top1": 1.0},
        "resources": {
            "disk_free_bytes": 600_000_000_000,
            "ram_available_bytes": 70_000_000_000,
            "swap_used_bytes": 0,
        },
        "process": {"pid": 1234, "lease_held": True, "lease_owner": "glm52-controller"},
    }


def _intent(
    *,
    dedupe_key: str = DEDUPE,
    to_state: str = "PRECHECK",
    claim_id: str = "claim-telegram-0001",
    metric_delta: float = 0.0,
    anchor_counter: int = 0,
) -> dict[str, Any]:
    event_kind = gs.TRANSITION_EVENT_KINDS[to_state]
    status = _status()
    status["state"] = to_state
    status["best_metrics"]["cosine"] += metric_delta
    checkpoint = {
        "event_count": anchor_counter,
        "event_head_hash": hashlib.sha256(f"events:{anchor_counter}".encode()).hexdigest(),
        "window_event_count": 0,
        "window_event_head_hash": hashlib.sha256(b"windows:0").hexdigest(),
        "checkpoint_seal_sha256": None,
    }
    anchor_body = {
        "schema": gs.CONTROLLER_ANCHOR_SCHEMA,
        "campaign_id": "glm52-test-campaign",
        "source_revision": SOURCE_REVISION,
        "controller_epoch": CONTROLLER_EPOCH,
        "expected_contract_sha256": CONTRACT_SHA,
        "from_state": None,
        "checkpoint": checkpoint,
    }
    anchor = {
        **anchor_body,
        "anchor_sha256": hashlib.sha256(canonical(anchor_body)).hexdigest(),
    }
    status_sha = hashlib.sha256(canonical({
        "schema": gs.CAMPAIGN_STATUS_SCHEMA,
        "status": status,
    })).hexdigest()
    rendered = gs.render_campaign_status_message(
        event_kind,
        dedupe_key,
        status,
        anchor,
        claim_id=claim_id,
        from_state=None,
        to_state=to_state,
    )
    sealed_intent = seal({
        "schema": gs.TRANSITION_INTENT_SCHEMA,
        "campaign_id": "glm52-test-campaign",
        "source_revision": SOURCE_REVISION,
        "controller_epoch": CONTROLLER_EPOCH,
        "expected_contract_sha256": CONTRACT_SHA,
        "event_kind": event_kind,
        "from_state": None,
        "to_state": to_state,
        "claim_id": claim_id,
        "requested_payload": {},
        "state_payload": {},
        "request_sha256": hashlib.sha256(canonical({"claim_id": claim_id})).hexdigest(),
        "dedupe_key": dedupe_key,
        "controller_anchor": anchor,
        "canonical_status": status,
        "canonical_status_sha256": status_sha,
        "rendered_message": rendered,
        "rendered_message_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "prepared_at": "2026-07-21T12:00:00Z",
    })
    auth = gs.TelegramAuthConfig(
        hmac_key=HMAC_RAW,
        expected_chat_identity_digest=gs.telegram_chat_identity_digest(CHAT_ID),
    )
    return {
        **sealed_intent,
        "controller_hmac_sha256": auth.authenticate({
            "schema": "hawking.glm52.state_transition_intent_auth.v1",
            "intent": sealed_intent,
        }),
    }


def _ledger(
    tmp_path: pathlib.Path,
    *,
    fault_injector=None,
) -> gt.TelegramDeliveryLedger:
    return gt.TelegramDeliveryLedger(
        tmp_path / "telegram-delivery.jsonl",
        fault_injector=fault_injector,
        clock=lambda: "2026-07-21T12:00:01Z",
    )


def test_services_are_unique_and_glm_specific() -> None:
    assert len(gt.KEYCHAIN_SERVICES) == len(set(gt.KEYCHAIN_SERVICES)) == 3
    assert all(service.startswith("com.hawking.glm52.gravity.telegram.") for service in gt.KEYCHAIN_SERVICES)
    assert gt.KEYCHAIN_ACCOUNT == "hawking-glm52-gravity"


def test_macos_keychain_writes_secret_only_through_stdin() -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(arguments, **kwargs):
        calls.append((list(arguments), dict(kwargs)))
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    store = gt.MacOSKeychain(runner=runner)
    store.set(gt.TOKEN_SERVICE, TOKEN)
    arguments, kwargs = calls[0]
    assert TOKEN not in arguments
    assert arguments[-1] == "-w"
    assert kwargs["input"] == TOKEN + "\n"
    assert "env" not in kwargs


def test_macos_keychain_errors_never_echo_secret() -> None:
    def runner(arguments, **_kwargs):
        return subprocess.CompletedProcess(arguments, 1, stdout="", stderr=f"bad {TOKEN}")

    with pytest.raises(gt.TelegramSecurityError) as caught:
        gt.MacOSKeychain(runner=runner).set(gt.TOKEN_SERVICE, TOKEN)
    assert TOKEN not in str(caught.value)
    assert TOKEN not in repr(caught.value)


@pytest.mark.parametrize("operation", ["get", "set"])
def test_macos_keychain_redacts_backend_exceptions(operation: str) -> None:
    def runner(_arguments, **_kwargs):
        raise RuntimeError(f"backend accidentally included {TOKEN}")

    store = gt.MacOSKeychain(runner=runner)
    with pytest.raises(gt.TelegramSecurityError) as caught:
        if operation == "get":
            store.get(gt.TOKEN_SERVICE)
        else:
            store.set(gt.TOKEN_SERVICE, TOKEN)
    assert TOKEN not in str(caught.value) + repr(caught.value)


def test_urllib_transport_redacts_opener_exception() -> None:
    def opener(request, **_kwargs):
        raise RuntimeError(request.full_url)

    transport = gt.UrllibTelegramTransport(opener=opener)
    with pytest.raises(gt.TelegramSecurityError) as caught:
        transport.call(TOKEN, "getMe", {})
    assert TOKEN not in str(caught.value) + repr(caught.value)


def test_hidden_token_configuration_validates_getme_before_storage() -> None:
    keychain = FakeKeychain()
    transport = FakeTransport(_get_me)
    prompts: list[str] = []

    def hidden(prompt: str) -> str:
        prompts.append(prompt)
        return TOKEN

    result = gt.configure_token(keychain, transport, hidden_prompt=hidden)
    assert prompts and "hidden" in prompts[0]
    assert transport.calls == [(TOKEN, "getMe", {})]
    assert keychain.set_calls == [(gt.TOKEN_SERVICE, TOKEN)]
    assert result["token_configured"] is True
    assert gt.SHA256_RE.fullmatch(result["bot_identity_digest"])
    rendered = json.dumps(result)
    assert TOKEN not in rendered and "glm52_bot" not in rendered


@pytest.mark.parametrize(
    "response",
    [
        gt.TelegramHTTPResponse(500, {"ok": True, "result": {}}),
        gt.TelegramHTTPResponse(200, {"ok": False, "description": TOKEN}),
        gt.TelegramHTTPResponse(200, {"ok": True, "result": {"id": 1, "is_bot": False, "username": "x"}}),
    ],
)
def test_invalid_getme_never_stores_or_leaks_token(response: gt.TelegramHTTPResponse) -> None:
    keychain = FakeKeychain()
    transport = FakeTransport(lambda *_args: response)
    with pytest.raises(gt.TelegramSecurityError) as caught:
        gt.configure_token(keychain, transport, hidden_prompt=lambda _prompt: TOKEN)
    assert keychain.set_calls == []
    assert TOKEN not in str(caught.value)


def test_injected_transport_exception_is_redacted() -> None:
    def handler(token: str, _method: str, _payload: dict) -> gt.TelegramHTTPResponse:
        raise RuntimeError(f"provider leaked {token}")

    with pytest.raises(gt.TelegramSecurityError) as caught:
        gt.configure_token(
            FakeKeychain(),
            FakeTransport(handler),
            hidden_prompt=lambda _prompt: TOKEN,
        )
    assert TOKEN not in str(caught.value) + repr(caught.value)


def test_discovery_stores_one_safe_human_private_chat_and_returns_only_digest() -> None:
    keychain = FakeKeychain({gt.TOKEN_SERVICE: TOKEN})

    def handler(_token: str, method: str, _payload: dict) -> gt.TelegramHTTPResponse:
        assert method == "getUpdates"
        response = _updates([int(CHAT_ID)])
        # Group, bot-authored, and mismatched-sender updates must be ignored.
        response.body["result"].extend([
            {"message": {"from": {"id": 9, "is_bot": False}, "chat": {"id": -9, "type": "group"}}},
            {"message": {"from": {"id": 8, "is_bot": True}, "chat": {"id": 8, "type": "private"}}},
            {"message": {"from": {"id": 7, "is_bot": False}, "chat": {"id": 6, "type": "private"}}},
        ])
        return response

    result = gt.discover_private_chat(keychain, FakeTransport(handler))
    assert keychain.set_calls == [(gt.CHAT_SERVICE, CHAT_ID)]
    assert result["private_chat_configured"] is True
    assert result["chat_identity_digest"] == gt.telegram_chat_identity_digest(CHAT_ID)
    rendered = json.dumps(result)
    assert CHAT_ID not in rendered and "secret-name" not in rendered


@pytest.mark.parametrize("chat_ids", [[], [111, 222]])
def test_discovery_refuses_zero_or_multiple_private_chats(chat_ids: list[int]) -> None:
    keychain = FakeKeychain({gt.TOKEN_SERVICE: TOKEN})
    transport = FakeTransport(lambda *_args: _updates(chat_ids))
    with pytest.raises(gt.TelegramSecurityError):
        gt.discover_private_chat(keychain, transport)
    assert keychain.set_calls == []


def test_hmac_key_is_generated_at_32_bytes_and_never_returned() -> None:
    keychain = FakeKeychain()
    result = gt.configure_hmac_key(keychain, random_bytes=lambda size: b"K" * size)
    assert keychain.set_calls[0][0] == gt.HMAC_SERVICE
    assert base64.urlsafe_b64decode(keychain.set_calls[0][1]) == b"K" * 32
    assert result["hmac_key_configured"] is True
    assert gt.SHA256_RE.fullmatch(result["hmac_key_identity_digest"])
    assert "S0tL" not in json.dumps(result)


def test_public_status_contains_only_booleans_and_digests() -> None:
    status = gt.credential_status(_configured_keychain())
    assert status["ready"] is True
    for value in status.values():
        assert isinstance(value, bool) or (isinstance(value, str) and gt.SHA256_RE.fullmatch(value))
    rendered = json.dumps(status)
    assert TOKEN not in rendered and CHAT_ID not in rendered and HMAC_ENCODED not in rendered


def test_malformed_public_status_is_fail_closed_without_secret_echo() -> None:
    keychain = FakeKeychain({
        gt.TOKEN_SERVICE: "bad-token-secret",
        gt.CHAT_SERVICE: "not-a-chat-secret",
        gt.HMAC_SERVICE: "bad-key-secret",
    })
    status = gt.credential_status(keychain)
    assert status == {
        "token_configured": False,
        "private_chat_configured": False,
        "hmac_key_configured": False,
        "ready": False,
    }
    rendered = json.dumps(status)
    assert "secret" not in rendered


def test_load_auth_is_nonserializable_and_repr_redacts() -> None:
    auth = gt.load_telegram_auth(_configured_keychain())
    assert auth.expected_chat_identity_digest == gt.telegram_chat_identity_digest(CHAT_ID)
    assert "HHHH" not in repr(auth)
    with pytest.raises(TypeError):
        json.dumps(auth)


def test_campaign_status_requires_every_field_and_rejects_inconsistent_counts() -> None:
    status = _status()
    normalized = gt.validate_campaign_status(status)
    assert set(normalized) == gt.STATUS_KEYS
    missing = dict(status)
    missing.pop("resources")
    with pytest.raises(gt.TelegramSecurityError, match="incomplete"):
        gt.validate_campaign_status(missing)
    inconsistent = _status()
    inconsistent["shards"] = {"fetched": 1, "verified": 2, "evicted": 0, "total": 282}
    with pytest.raises(gt.TelegramSecurityError, match="inconsistent"):
        gt.validate_campaign_status(inconsistent)


def test_campaign_status_rejects_non_string_metric_names_fail_closed() -> None:
    status = _status()
    status["best_metrics"] = {1: 0.999}
    with pytest.raises(gt.TelegramSecurityError, match="metric name"):
        gt.validate_campaign_status(status)


def test_message_binds_dedupe_and_all_required_campaign_fields() -> None:
    text = gt.compose_message("xet_autotune_result", DEDUPE, _status())
    required_fragments = (
        f"dedupe_key={DEDUPE}",
        "state=AUTOTUNE_XET",
        "source_coverage_percent=12.500000",
        "shards fetched=40 verified=39 evicted=20 total=282",
        "network_bytes=123456",
        "throughput_bytes_per_second=98765.500000",
        "eta_seconds=3600",
        "current_window=W003 current_layer=12",
        "candidate_rates=0.98,0.75,0.50",
        "best_metrics=",
        "disk_free_bytes=600000000000",
        "ram_available_bytes=70000000000",
        "swap_used_bytes=0",
        "pid=1234 lease_held=true lease_owner=glm52-controller",
    )
    assert all(fragment in text for fragment in required_fragments)


def _successful_sender() -> tuple[FakeTransport, list[str]]:
    sent_text: list[str] = []

    def handler(_token: str, method: str, payload: dict) -> gt.TelegramHTTPResponse:
        assert method == "sendMessage"
        sent_text.append(payload["text"])
        return gt.TelegramHTTPResponse(200, {
            "ok": True,
            "result": {
                "message_id": 77,
                "chat": {"id": int(CHAT_ID), "type": "private"},
                "text": payload["text"],
            },
        })

    return FakeTransport(handler), sent_text


def test_sender_returns_state_authenticated_receipt_only_on_exact_success(
    tmp_path: pathlib.Path,
) -> None:
    keychain = _configured_keychain()
    transport, sent_text = _successful_sender()
    intent = _intent()
    ledger = _ledger(tmp_path)
    receipt = gt.send_campaign_status(
        intent,
        ledger=ledger,
        keychain=keychain,
        transport=transport,
    )
    assert sent_text == [intent["rendered_message"]]
    assert receipt["status"] == "DELIVERED"
    assert receipt["dedupe_key"] == DEDUPE
    assert receipt["canonical_status_sha256"] == intent["canonical_status_sha256"]
    assert receipt["rendered_message_sha256"] == intent["rendered_message_sha256"]
    assert receipt["controller_anchor_sha256"] == intent["controller_anchor"]["anchor_sha256"]
    assert receipt["transition_intent_sha256"] == intent["seal_sha256"]
    assert receipt["message_id"] == 77
    assert receipt["chat_identity_digest"] == gt.telegram_chat_identity_digest(CHAT_ID)
    auth = gt.load_telegram_auth(keychain)
    body = {key: value for key, value in receipt.items() if key != "hmac_sha256"}
    assert auth.verify(body, receipt["hmac_sha256"])
    entries = ledger.verified_entries(auth)
    assert [entry["kind"] for entry in entries] == list(gt.NORMAL_DELIVERY_LIFECYCLE)
    assert entries[-1]["receipt"] == receipt
    assert entries[-1]["binding"]["rendered_message"] == intent["rendered_message"]
    rendered = json.dumps(receipt)
    assert TOKEN not in rendered and CHAT_ID not in rendered and HMAC_ENCODED not in rendered


def test_delivery_replay_returns_exact_receipt_without_network(tmp_path: pathlib.Path) -> None:
    intent = _intent()
    ledger = _ledger(tmp_path)
    first_transport, _sent = _successful_sender()
    first = gt.send_campaign_status(
        intent,
        ledger=ledger,
        keychain=_configured_keychain(),
        transport=first_transport,
    )
    replay_transport = FakeTransport()
    replayed = gt.send_campaign_status(
        intent,
        ledger=gt.TelegramDeliveryLedger(ledger.path),
        keychain=_configured_keychain(),
        transport=replay_transport,
    )
    assert replayed == first
    assert replay_transport.calls == []


def test_sender_rejects_fabricated_controller_intent_before_outbox_or_network(
    tmp_path: pathlib.Path,
) -> None:
    intent = dict(_intent())
    intent["controller_hmac_sha256"] = "0" * 64
    ledger = _ledger(tmp_path)
    transport = FakeTransport()
    with pytest.raises(gt.TelegramSecurityError, match="controller transition intent"):
        gt.send_campaign_status(
            intent,
            ledger=ledger,
            keychain=_configured_keychain(),
            transport=transport,
        )
    assert transport.calls == []
    assert not ledger.path.exists()


@pytest.mark.parametrize("changed", ["event", "message", "status", "anchor"])
def test_same_dedupe_rejects_changed_bound_intent_without_network(
    changed: str,
    tmp_path: pathlib.Path,
) -> None:
    ledger = _ledger(tmp_path)
    transport, _sent = _successful_sender()
    gt.send_campaign_status(
        _intent(),
        ledger=ledger,
        keychain=_configured_keychain(),
        transport=transport,
    )
    if changed == "event":
        altered = _intent(to_state="CLOSE_KIMI")
    elif changed == "message":
        altered = _intent(claim_id="claim-telegram-0002")
    elif changed == "status":
        altered = _intent(metric_delta=0.0001)
    else:
        altered = _intent(anchor_counter=1)
    replay_transport = FakeTransport()
    with pytest.raises(gt.TelegramSecurityError, match="different delivery intent"):
        gt.send_campaign_status(
            altered,
            ledger=gt.TelegramDeliveryLedger(ledger.path),
            keychain=_configured_keychain(),
            transport=replay_transport,
        )
    assert replay_transport.calls == []


def test_crash_after_prepared_fsync_replays_safe_unsent_outbox(
    tmp_path: pathlib.Path,
) -> None:
    def crash(phase: str) -> None:
        if phase == "after_outbox_prepared_fsync":
            raise RuntimeError("simulated crash")

    intent = _intent()
    path = tmp_path / "telegram-delivery.jsonl"
    untouched = FakeTransport()
    with pytest.raises(RuntimeError, match="simulated crash"):
        gt.send_campaign_status(
            intent,
            ledger=gt.TelegramDeliveryLedger(path, fault_injector=crash),
            keychain=_configured_keychain(),
            transport=untouched,
        )
    assert untouched.calls == []
    transport, _sent = _successful_sender()
    receipt = gt.send_campaign_status(
        intent,
        ledger=gt.TelegramDeliveryLedger(path),
        keychain=_configured_keychain(),
        transport=transport,
    )
    assert receipt["status"] == "DELIVERED"
    assert len(transport.calls) == 1


def test_crash_after_send_success_is_ambiguous_and_never_resends(
    tmp_path: pathlib.Path,
) -> None:
    def crash(phase: str) -> None:
        if phase == "after_send_success_before_receipt_append":
            raise RuntimeError("simulated post-send crash")

    path = tmp_path / "telegram-delivery.jsonl"
    transport, _sent = _successful_sender()
    with pytest.raises(RuntimeError, match="post-send crash"):
        gt.send_campaign_status(
            _intent(),
            ledger=gt.TelegramDeliveryLedger(path, fault_injector=crash),
            keychain=_configured_keychain(),
            transport=transport,
        )
    assert len(transport.calls) == 1
    replay_transport = FakeTransport()
    with pytest.raises(gt.TelegramSecurityError, match="ambiguous"):
        gt.send_campaign_status(
            _intent(),
            ledger=gt.TelegramDeliveryLedger(path),
            keychain=_configured_keychain(),
            transport=replay_transport,
        )
    assert replay_transport.calls == []
    auth = gt.load_telegram_auth(_configured_keychain())
    entries = gt.TelegramDeliveryLedger(path).verified_entries(auth)
    assert [entry["kind"] for entry in entries] == [
        "OUTBOX_PREPARED", "SEND_STARTED", "AMBIGUOUS_BLOCKED",
    ]
    block = entries[-1]["reconciliation"]
    assert block["attempt"] == 1
    assert block["send_started_seq"] == entries[-2]["seq"]
    assert block["send_started_chain_sha256"] == entries[-2]["chain_sha256"]
    assert entries[-1]["receipt"] is None


def test_hmac_authorized_duplicate_retry_is_bound_durable_and_one_attempt(
    tmp_path: pathlib.Path,
) -> None:
    def post_send_crash(phase: str) -> None:
        if phase == "after_send_success_before_receipt_append":
            raise RuntimeError("simulated first-attempt crash")

    intent = _intent()
    path = tmp_path / "telegram-delivery.jsonl"
    first_transport, _sent = _successful_sender()
    with pytest.raises(RuntimeError, match="first-attempt crash"):
        gt.send_campaign_status(
            intent,
            ledger=gt.TelegramDeliveryLedger(path, fault_injector=post_send_crash),
            keychain=_configured_keychain(),
            transport=first_transport,
        )
    assert len(first_transport.calls) == 1

    no_retry = FakeTransport()
    with pytest.raises(gt.TelegramSecurityError, match="durably blocked"):
        gt.send_campaign_status(
            intent,
            ledger=gt.TelegramDeliveryLedger(path),
            keychain=_configured_keychain(),
            transport=no_retry,
        )
    assert no_retry.calls == []

    auth = gt.load_telegram_auth(_configured_keychain())
    blocked_entries = gt.TelegramDeliveryLedger(path).verified_entries(auth)
    blocked = blocked_entries[-1]
    forged_block = dict(blocked)
    forged_block["chain_sha256"] = "0" * 64
    with pytest.raises(gt.TelegramSecurityError, match="authentication"):
        gt.make_reconciliation_authorization(
            intent,
            ambiguity_entry=forged_block,
            auth=auth,
            action="AUTHORIZE_DUPLICATE_RETRY",
            authorization_claim_id="retry-claim-forged",
            reason="This must not authorize a retry",
        )
    authorization = gt.make_reconciliation_authorization(
        intent,
        ambiguity_entry=blocked,
        auth=auth,
        action="AUTHORIZE_DUPLICATE_RETRY",
        authorization_claim_id="retry-claim-0001",
        reason="Operator accepts one possible duplicate after process crash",
        authorized_at="2026-07-21T12:00:03Z",
    )
    assert authorization["ambiguous_entry_seq"] == blocked["seq"]
    assert authorization["ambiguous_entry_chain_sha256"] == blocked["chain_sha256"]
    assert authorization["send_started_seq"] == blocked["reconciliation"]["send_started_seq"]
    assert authorization["send_started_chain_sha256"] == \
        blocked["reconciliation"]["send_started_chain_sha256"]

    tampered = dict(authorization)
    tampered["hmac_sha256"] = "0" * 64
    refused_transport = FakeTransport()
    with pytest.raises(gt.TelegramSecurityError, match="HMAC"):
        gt.send_campaign_status(
            intent,
            ledger=gt.TelegramDeliveryLedger(path),
            keychain=_configured_keychain(),
            transport=refused_transport,
            duplicate_retry_authorization=tampered,
        )
    assert refused_transport.calls == []

    def authorization_crash(phase: str) -> None:
        if phase == "after_duplicate_retry_authorized_fsync":
            raise RuntimeError("simulated authorization checkpoint crash")

    untouched_retry = FakeTransport()
    with pytest.raises(RuntimeError, match="authorization checkpoint crash"):
        gt.send_campaign_status(
            intent,
            ledger=gt.TelegramDeliveryLedger(path, fault_injector=authorization_crash),
            keychain=_configured_keychain(),
            transport=untouched_retry,
            duplicate_retry_authorization=authorization,
        )
    assert untouched_retry.calls == []

    retry_transport, _retry_text = _successful_sender()
    receipt = gt.send_campaign_status(
        intent,
        ledger=gt.TelegramDeliveryLedger(path),
        keychain=_configured_keychain(),
        transport=retry_transport,
    )
    assert receipt["status"] == "DELIVERED"
    assert len(retry_transport.calls) == 1
    entries = gt.TelegramDeliveryLedger(path).verified_entries(auth)
    assert [entry["kind"] for entry in entries] == [
        "OUTBOX_PREPARED",
        "SEND_STARTED",
        "AMBIGUOUS_BLOCKED",
        "DUPLICATE_RETRY_AUTHORIZED",
        "SEND_STARTED",
        "DELIVERY_RECEIPT",
    ]
    assert entries[3]["reconciliation"] == authorization
    assert entries[2]["reconciliation"]["send_started_chain_sha256"] == \
        entries[1]["chain_sha256"]

    replay_transport = FakeTransport()
    assert gt.send_campaign_status(
        intent,
        ledger=gt.TelegramDeliveryLedger(path),
        keychain=_configured_keychain(),
        transport=replay_transport,
        duplicate_retry_authorization=authorization,
    ) == receipt
    assert replay_transport.calls == []


def test_duplicate_retry_authorization_claim_is_consumed_only_once(
    tmp_path: pathlib.Path,
) -> None:
    def first_attempt_crash(phase: str) -> None:
        if phase == "after_send_success_before_receipt_append":
            raise RuntimeError("simulated first-attempt crash")

    intent = _intent()
    path = tmp_path / "telegram-delivery.jsonl"
    first_transport, _sent = _successful_sender()
    with pytest.raises(RuntimeError):
        gt.send_campaign_status(
            intent,
            ledger=gt.TelegramDeliveryLedger(path, fault_injector=first_attempt_crash),
            keychain=_configured_keychain(),
            transport=first_transport,
        )
    with pytest.raises(gt.TelegramSecurityError, match="durably blocked"):
        gt.send_campaign_status(
            intent,
            ledger=gt.TelegramDeliveryLedger(path),
            keychain=_configured_keychain(),
            transport=FakeTransport(),
        )

    auth = gt.load_telegram_auth(_configured_keychain())
    first_block = gt.TelegramDeliveryLedger(path).verified_entries(auth)[-1]
    first_authorization = gt.make_reconciliation_authorization(
        intent,
        ambiguity_entry=first_block,
        auth=auth,
        action="AUTHORIZE_DUPLICATE_RETRY",
        authorization_claim_id="one-use-retry-claim",
        reason="Authorize exactly one retry attempt",
        authorized_at="2026-07-21T12:00:03Z",
    )

    def pre_network_crash(phase: str) -> None:
        if phase == "before_network_send":
            raise RuntimeError("simulated retry crash")

    with pytest.raises(RuntimeError, match="retry crash"):
        gt.send_campaign_status(
            intent,
            ledger=gt.TelegramDeliveryLedger(path, fault_injector=pre_network_crash),
            keychain=_configured_keychain(),
            transport=FakeTransport(),
            duplicate_retry_authorization=first_authorization,
        )
    with pytest.raises(gt.TelegramSecurityError, match="durably blocked"):
        gt.send_campaign_status(
            intent,
            ledger=gt.TelegramDeliveryLedger(path),
            keychain=_configured_keychain(),
            transport=FakeTransport(),
        )

    second_block = gt.TelegramDeliveryLedger(path).verified_entries(auth)[-1]
    reused_claim = gt.make_reconciliation_authorization(
        intent,
        ambiguity_entry=second_block,
        auth=auth,
        action="AUTHORIZE_DUPLICATE_RETRY",
        authorization_claim_id="one-use-retry-claim",
        reason="A consumed claim must not authorize another attempt",
        authorized_at="2026-07-21T12:00:04Z",
    )
    refused_transport = FakeTransport()
    with pytest.raises(gt.TelegramSecurityError, match="already consumed"):
        gt.send_campaign_status(
            intent,
            ledger=gt.TelegramDeliveryLedger(path),
            keychain=_configured_keychain(),
            transport=refused_transport,
            duplicate_retry_authorization=reused_claim,
        )
    assert refused_transport.calls == []


def test_ambiguous_send_rejects_operator_claim_and_requires_exact_bot_receipt(
    tmp_path: pathlib.Path,
) -> None:
    def crash(phase: str) -> None:
        if phase == "after_send_success_before_receipt_append":
            raise RuntimeError("simulated post-send crash")

    intent = _intent()
    path = tmp_path / "telegram-delivery.jsonl"
    transport, _sent = _successful_sender()
    with pytest.raises(RuntimeError):
        gt.send_campaign_status(
            intent,
            ledger=gt.TelegramDeliveryLedger(path, fault_injector=crash),
            keychain=_configured_keychain(),
            transport=transport,
    )
    auth = gt.load_telegram_auth(_configured_keychain())
    # Even an HMAC-authenticated operator assertion is not delivery evidence.  The
    # controller/outbox accept only the exact successful Bot API v3 receipt.
    obsolete_body = {
        "schema": "hawking.glm52.telegram_operator_delivery_confirmation.v1",
        "status": "OPERATOR_CONFIRMED_DELIVERED",
        "transition_intent": intent,
        "message_id": 77,
        "confirmation_claim_id": "operator-confirmation-0001",
        "confirmed_at": "2026-07-21T12:00:02Z",
    }
    operator_confirmation = {
        **obsolete_body,
        "hmac_sha256": auth.authenticate(obsolete_body),
    }
    ledger = gt.TelegramDeliveryLedger(path)
    entries_before = ledger.verified_entries(auth)
    with pytest.raises(gt.TelegramSecurityError, match="binding/HMAC"):
        ledger.reconcile_ambiguous(intent, operator_confirmation, auth=auth)
    assert ledger.verified_entries(auth) == entries_before
    response = {
        "ok": True,
        "result": {
            "message_id": 77,
            "chat": {"id": int(CHAT_ID), "type": "private"},
            "text": intent["rendered_message"],
        },
    }
    receipt = gs.make_telegram_delivery_receipt(
        intent,
        auth=auth,
        bot_api_response=response,
        http_status=200,
        delivered_at="2026-07-21T12:00:02Z",
    )
    tampered = dict(receipt)
    tampered["hmac_sha256"] = "0" * 64
    with pytest.raises(gt.TelegramSecurityError, match="binding/HMAC"):
        ledger.reconcile_ambiguous(intent, tampered, auth=auth)
    reconciled = ledger.reconcile_ambiguous(intent, receipt, auth=auth)
    assert reconciled == receipt
    replay_transport = FakeTransport()
    assert gt.send_campaign_status(
        intent,
        ledger=ledger,
        keychain=_configured_keychain(),
        transport=replay_transport,
    ) == receipt
    assert replay_transport.calls == []


def test_crash_after_receipt_ledger_fsync_recovers_without_network(
    tmp_path: pathlib.Path,
) -> None:
    def crash(phase: str) -> None:
        if phase == "after_delivery_receipt_ledger_fsync_before_head":
            raise RuntimeError("simulated head-update crash")

    path = tmp_path / "telegram-delivery.jsonl"
    transport, _sent = _successful_sender()
    with pytest.raises(RuntimeError, match="head-update crash"):
        gt.send_campaign_status(
            _intent(),
            ledger=gt.TelegramDeliveryLedger(path, fault_injector=crash),
            keychain=_configured_keychain(),
            transport=transport,
        )
    assert len(transport.calls) == 1
    replay_transport = FakeTransport()
    receipt = gt.send_campaign_status(
        _intent(),
        ledger=gt.TelegramDeliveryLedger(path),
        keychain=_configured_keychain(),
        transport=replay_transport,
    )
    assert receipt["status"] == "DELIVERED"
    assert replay_transport.calls == []


def test_ledger_tamper_is_rejected_before_network(tmp_path: pathlib.Path) -> None:
    ledger = _ledger(tmp_path)
    transport, _sent = _successful_sender()
    gt.send_campaign_status(
        _intent(), ledger=ledger, keychain=_configured_keychain(), transport=transport
    )
    lines = ledger.path.read_bytes().splitlines()
    row = json.loads(lines[0])
    row["recorded_at"] = "tampered"
    lines[0] = canonical(row)
    ledger.path.write_bytes(b"\n".join(lines) + b"\n")
    replay_transport = FakeTransport()
    with pytest.raises(gt.TelegramSecurityError, match="hash chain|HMAC"):
        gt.send_campaign_status(
            _intent(),
            ledger=gt.TelegramDeliveryLedger(ledger.path),
            keychain=_configured_keychain(),
            transport=replay_transport,
        )
    assert replay_transport.calls == []


def test_noncanonical_ledger_reencoding_is_rejected_before_network(
    tmp_path: pathlib.Path,
) -> None:
    ledger = _ledger(tmp_path)
    transport, _sent = _successful_sender()
    gt.send_campaign_status(
        _intent(), ledger=ledger, keychain=_configured_keychain(), transport=transport
    )
    lines = ledger.path.read_bytes().splitlines()
    lines[0] = b" " + lines[0]
    ledger.path.write_bytes(b"\n".join(lines) + b"\n")
    replay_transport = FakeTransport()
    with pytest.raises(gt.TelegramSecurityError, match="canonical JSON"):
        gt.send_campaign_status(
            _intent(),
            ledger=gt.TelegramDeliveryLedger(ledger.path),
            keychain=_configured_keychain(),
            transport=replay_transport,
        )
    assert replay_transport.calls == []


def test_torn_jsonl_tail_recovers_only_from_authenticated_head_without_network(
    tmp_path: pathlib.Path,
) -> None:
    ledger = _ledger(tmp_path)
    transport, _sent = _successful_sender()
    gt.send_campaign_status(
        _intent(), ledger=ledger, keychain=_configured_keychain(), transport=transport
    )
    with ledger.path.open("ab") as handle:
        handle.write(b'{"schema":')
    replay_transport = FakeTransport()
    receipt = gt.send_campaign_status(
        _intent(),
        ledger=gt.TelegramDeliveryLedger(ledger.path),
        keychain=_configured_keychain(),
        transport=replay_transport,
    )
    assert receipt["status"] == "DELIVERED"
    recovered = ledger.path.read_bytes()
    assert recovered.endswith(b"\n")
    assert len(recovered.splitlines()) == len(gt.NORMAL_DELIVERY_LIFECYCLE)
    assert b'{"schema":' not in recovered.splitlines()[-1]
    assert replay_transport.calls == []


def test_unanchored_torn_first_record_is_refused(tmp_path: pathlib.Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.path.write_bytes(b'{"partial":')
    with pytest.raises(gt.TelegramSecurityError, match="authenticated recovery head"):
        gt.send_campaign_status(
            _intent(),
            ledger=ledger,
            keychain=_configured_keychain(),
            transport=FakeTransport(),
        )


def test_authenticated_head_detects_clean_tail_truncation(tmp_path: pathlib.Path) -> None:
    ledger = _ledger(tmp_path)
    transport, _sent = _successful_sender()
    gt.send_campaign_status(
        _intent(), ledger=ledger, keychain=_configured_keychain(), transport=transport
    )
    lines = ledger.path.read_bytes().splitlines()
    ledger.path.write_bytes(b"\n".join(lines[:-1]) + b"\n")
    replay_transport = FakeTransport()
    with pytest.raises(gt.TelegramSecurityError, match="clean-tail truncated"):
        gt.send_campaign_status(
            _intent(),
            ledger=gt.TelegramDeliveryLedger(ledger.path),
            keychain=_configured_keychain(),
            transport=replay_transport,
        )
    assert replay_transport.calls == []


def test_missing_authenticated_head_is_rejected_before_network(
    tmp_path: pathlib.Path,
) -> None:
    ledger = _ledger(tmp_path)
    transport, _sent = _successful_sender()
    gt.send_campaign_status(
        _intent(), ledger=ledger, keychain=_configured_keychain(), transport=transport
    )
    ledger.head_path.unlink()
    replay_transport = FakeTransport()
    with pytest.raises(gt.TelegramSecurityError, match="lacks its authenticated durable head"):
        gt.send_campaign_status(
            _intent(),
            ledger=gt.TelegramDeliveryLedger(ledger.path),
            keychain=_configured_keychain(),
            transport=replay_transport,
        )
    assert replay_transport.calls == []


@pytest.mark.parametrize("symlink_kind", ["parent", "ledger", "lock", "head"])
def test_ledger_refuses_every_symlink_surface_before_network(
    symlink_kind: str,
    tmp_path: pathlib.Path,
) -> None:
    safe_parent = tmp_path / "safe"
    safe_parent.mkdir()
    target = tmp_path / "target"
    target.write_bytes(b"")
    if symlink_kind == "parent":
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        safe_parent.rmdir()
        safe_parent.symlink_to(real_parent, target_is_directory=True)
    ledger = gt.TelegramDeliveryLedger(safe_parent / "delivery.jsonl")
    if symlink_kind == "ledger":
        ledger.path.symlink_to(target)
    elif symlink_kind == "lock":
        ledger.lock_path.symlink_to(target)
    elif symlink_kind == "head":
        ledger.head_path.symlink_to(target)
    transport = FakeTransport()
    with pytest.raises(gt.TelegramSecurityError, match="unsafe|regular|parent|head"):
        gt.send_campaign_status(
            _intent(), ledger=ledger, keychain=_configured_keychain(), transport=transport
        )
    assert transport.calls == []


def test_concurrent_same_dedupe_sends_once_and_replays_one_receipt(
    tmp_path: pathlib.Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def handler(_token: str, _method: str, payload: dict) -> gt.TelegramHTTPResponse:
        entered.set()
        assert release.wait(timeout=5)
        return gt.TelegramHTTPResponse(200, {
            "ok": True,
            "result": {
                "message_id": 77,
                "chat": {"id": int(CHAT_ID), "type": "private"},
                "text": payload["text"],
            },
        })

    path = tmp_path / "telegram-delivery.jsonl"
    transport = FakeTransport(handler)

    def deliver() -> dict[str, Any]:
        return gt.send_campaign_status(
            _intent(),
            ledger=gt.TelegramDeliveryLedger(path),
            keychain=_configured_keychain(),
            transport=transport,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(deliver)
        assert entered.wait(timeout=5)
        second = pool.submit(deliver)
        release.set()
        first_receipt = first.result(timeout=5)
        second_receipt = second.result(timeout=5)
    assert first_receipt == second_receipt
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "mutation",
    ["http", "ok", "message_id", "text", "dedupe", "chat", "chat_type"],
)
def test_sender_rejects_every_response_binding_failure_without_secret_leak(
    mutation: str,
    tmp_path: pathlib.Path,
) -> None:
    def handler(_token: str, _method: str, payload: dict) -> gt.TelegramHTTPResponse:
        body: dict[str, Any] = {
            "ok": True,
            "result": {
                "message_id": 77,
                "chat": {"id": int(CHAT_ID), "type": "private"},
                "text": payload["text"],
            },
        }
        status = 200
        if mutation == "http":
            status = 503
        elif mutation == "ok":
            body = {"ok": False, "description": TOKEN}
        elif mutation == "message_id":
            body["result"]["message_id"] = 0
        elif mutation == "text":
            body["result"]["text"] = "wrong"
        elif mutation == "dedupe":
            body["result"]["text"] = payload["text"].replace(DEDUPE, "e" * 64)
        elif mutation == "chat":
            body["result"]["chat"]["id"] = int(CHAT_ID) + 1
        elif mutation == "chat_type":
            body["result"]["chat"]["type"] = "group"
        return gt.TelegramHTTPResponse(status, body)

    with pytest.raises(gt.TelegramSecurityError) as caught:
        gt.send_campaign_status(
            _intent(),
            ledger=_ledger(tmp_path),
            keychain=_configured_keychain(),
            transport=FakeTransport(handler),
        )
    rendered = str(caught.value) + repr(caught.value)
    assert TOKEN not in rendered and CHAT_ID not in rendered and HMAC_ENCODED not in rendered


def test_sender_credentials_repr_is_redacted() -> None:
    credentials = gt._load_sender_credentials(_configured_keychain())
    rendered = repr(credentials)
    assert "<redacted>" in rendered
    assert TOKEN not in rendered and CHAT_ID not in rendered and HMAC_ENCODED not in rendered


def test_cli_has_no_secret_arguments_or_send_payload_surface() -> None:
    parser = gt.build_parser()
    help_text = parser.format_help()
    assert "--token" not in help_text
    assert "--chat" not in help_text
    assert "--hmac" not in help_text
    for command in ("status", "configure-token", "discover-private-chat", "configure-hmac-key"):
        parsed = parser.parse_args([command])
        assert parsed.command == command


def test_cli_rejects_mistaken_secret_argument_without_echo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        gt.main(
            ["configure-token", "--token", TOKEN],
            keychain=FakeKeychain(),
            transport=FakeTransport(),
        )
    captured = capsys.readouterr()
    assert TOKEN not in captured.out + captured.err


def test_cli_uses_only_injected_fakes(capsys: pytest.CaptureFixture[str]) -> None:
    keychain = FakeKeychain()
    assert gt.main(
        ["configure-token"],
        keychain=keychain,
        transport=FakeTransport(_get_me),
        hidden_prompt=lambda _prompt: TOKEN,
    ) == 0
    assert gt.main(
        ["configure-hmac-key"],
        keychain=keychain,
        transport=FakeTransport(),
        random_bytes=lambda size: b"Z" * size,
    ) == 0
    assert gt.main(["status"], keychain=keychain, transport=FakeTransport()) == 0
    output = capsys.readouterr().out
    assert TOKEN not in output and CHAT_ID not in output and HMAC_ENCODED not in output
