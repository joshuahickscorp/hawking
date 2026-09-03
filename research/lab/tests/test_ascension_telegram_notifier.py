"""Offline tests for the physical Ascension Telegram observer."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from lab.operators.ascension_telegram_notifier import (
    CHAT_SERVICE,
    EXACT_RUNTIME_REQUIRED_FACTS,
    EXACT_RUNTIME_SCHEMA,
    EXACT_RUNTIME_STATUS,
    EVENT_CATALOG,
    MODEL_IDS,
    QUALIFICATION_RECEIPT_CONTRACT,
    TOKEN_SERVICE,
    AscensionTelegramNotifier,
    NotifierPaths,
)
from lab.receipts import seal


TOKEN = "123456789:abcdefghijklmnopqrstuvwx"
CHAT = "123456789"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _worker(paths: NotifierPaths, key: str) -> None:
    model = paths.model(key)
    _write(
        model.worker_status,
        {
            "recorded_at": _now(),
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "heartbeat": 1,
            "phase": "ACTIVE_EXPERIMENT",
            "status": "ACTIVE",
        },
    )
    _write(model.runtime_status, {"recorded_at": _now(), "pid": os.getpid(), "heartbeat": 1, "phase": "WAITING_FOR_NATIVE_COMPLETE_TOKEN_RUNTIME"})
    _write(model.complete_status, {"recorded_at": _now(), "pid": os.getpid(), "heartbeat": 1, "phase": "PACKING"})


def _notifier(tmp_path: Path, sent: list[str]) -> tuple[AscensionTelegramNotifier, NotifierPaths]:
    physical = tmp_path / "physical"
    paths = NotifierPaths(physical, tmp_path / "notifications")
    _worker(paths, "qwen30")
    _worker(paths, "qwen80")
    _write(paths.dashboard, {"global_resources": {"free_disk_bytes": 500 * 1024**3, "minimum_reserved_free_bytes": 160 * 1024**3, "host": {"memory_pressure": "System-wide memory free percentage: 93%", "swap": "total = 0.00M used = 0.00M", "thermal": "No thermal warning"}}})

    def reader(service: str) -> str | None:
        return TOKEN if service == TOKEN_SERVICE else CHAT if service == CHAT_SERVICE else None

    def sender(_token: str, _chat: str, text: str) -> dict:
        sent.append(text)
        return {"bot_api_http_status": 200, "message_id": 42, "message_date": 123}

    return AscensionTelegramNotifier(paths=paths, credential_reader=reader, sender=sender, remote_validator=lambda _token: True), paths


def _tg10_receipt(paths: NotifierPaths, key: str, **override: object) -> Path:
    path = paths.model(key).root / "tg3" / f"{key.upper()}_TG10_OPERATIONAL_PASS.json"
    body: dict[str, object] = {
        "schema": "hawking.ascension.qwen_tg10_operational_pass.v1",
        "status": "PASS",
        "complete_bpw": 1.2,
        "median_base_true_tps": 101.0,
        "complete_native_model": True,
        "real_metal": True,
        "autoregressive_generation": True,
        "hcli_pass": True,
        "fallback_count": 0,
    }
    body.update(override)
    _write(path, seal(body))
    return path


def _exact_runtime_receipt(
    paths: NotifierPaths,
    key: str,
    *,
    runtime_override: dict[str, object] | None = None,
    document_override: dict[str, object] | None = None,
) -> Path:
    path = paths.model(key).runtime_receipt
    runtime: dict[str, object] = {
        **{field: True for field in EXACT_RUNTIME_REQUIRED_FACTS},
        "measured_token_count": 42,
        "timing_scope": "complete_model_token_loop",
    }
    if runtime_override:
        runtime.update(runtime_override)
    body: dict[str, object] = {
        "schema": EXACT_RUNTIME_SCHEMA,
        "status": EXACT_RUNTIME_STATUS,
        "binding": {
            "model_id": MODEL_IDS[key],
            "source_content_identity_sha256": "a" * 64,
            "source_revalidation_seal_sha256": "b" * 64,
            "complete_artifact_admission_seal_sha256": "c" * 64,
            "runtime_executable_sha256": "d" * 64,
        },
        "runtime": runtime,
    }
    if document_override:
        body.update(document_override)
    _write(path, seal(body))
    return path


def _runtime_supersession(paths: NotifierPaths, key: str, receipt: Path) -> Path:
    original = json.loads(receipt.read_text(encoding="utf-8"))
    raw = receipt.read_bytes()
    archive = (
        receipt.parent
        / "runtime-receipt-history"
        / f"{key.upper()}_EXACT_FULL_TOKEN_RUNTIME_RECEIPT_{original['seal_sha256']}.json"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(raw)
    archive_sha = hashlib.sha256(raw).hexdigest()
    executable = original["binding"]["runtime_executable_sha256"]
    body = {
        "schema": "hawking.ascension.physical_exact_full_token_runtime_supersession.v1",
        "status": "REVOKED_TEST_RUNTIME_DEFECT",
        "recorded_at": _now(),
        "binding": {
            "model_id": MODEL_IDS[key],
            "canonical_runtime_receipt_path": str(receipt),
            "superseded_runtime_receipt_seal_sha256": original["seal_sha256"],
            "defective_runtime_executable_sha256": executable,
            "archived_runtime_receipt_path": str(archive),
            "archived_runtime_receipt_document_sha256": archive_sha,
        },
        "revoked_runtime": {
            "canonical_receipt_path": str(receipt),
            "canonical_receipt_seal_sha256": original["seal_sha256"],
            "complete_manifest_seal_sha256": "e" * 64,
            "model_id": MODEL_IDS[key],
            "runtime_executable_sha256": executable,
        },
        "historical_pass_archive_path": str(archive),
        "historical_pass_archive_sha256": archive_sha,
        "defect": {"class": "TEST_RUNTIME_DEFECT"},
        "invalidates": {
            "canonical_native_runtime_pass": True,
            "all_old_full_token_prompt_and_profile_controls_bound_to_runtime_sha": True,
            "native_http_adapter_and_transport_handoff_bound_to_runtime_sha": True,
            "any_hcli_tps_tg_capability_or_tournament_consumer_of_that_sha": True,
        },
        "required_before_reissue": ["new executable", "fresh proof"],
        "consumer_contract": {
            "fail_closed_if_canonical_status_is_not_pass": True,
            "fail_closed_if_this_supersession_revokes_the_bound_receipt_seal_or_runtime_executable_sha256": True,
            "historical_archive_is_for_negative_science_only_not_a_gate_authority": True,
        },
        "claim_boundary": {"revocation_is_not_a_new_runtime_pass": True},
    }
    path = paths.model(key).runtime_supersession
    _write(path, seal(body))
    return path


def test_event_catalog_has_every_requested_model_and_global_milestone() -> None:
    required = {
        "operator_path_test",
        "qwen_complete_native_runtime",
        "qwen_first_coherent_generation",
        "qwen_capability_pass",
        "qwen_capability_fail",
        "qwen_tg10_operational_pass",
        "qwen_tg_rung",
        "qwen_tg3_qualified",
        "qwen_worker_stalled",
        "qwen_worker_recovered",
        "qwen_physical_representation_frontier_improved",
        "global_both_tg10_operational",
        "global_both_tg3_qualified",
        "global_tournament_automatically_launched",
        "global_tournament_complete",
        "global_human_decision_required",
        "global_resource_emergency",
        "global_detached_worker_failure",
        "global_campaign_resumed",
    }
    assert required.issubset(set(EVENT_CATALOG))
    assert "router_tps" in QUALIFICATION_RECEIPT_CONTRACT["explicitly_rejected_as_model_tps"]
    assert QUALIFICATION_RECEIPT_CONTRACT["native_runtime"]["filename"] == "QWEN{30,80}_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json"
    assert QUALIFICATION_RECEIPT_CONTRACT["native_runtime"]["schema"] == EXACT_RUNTIME_SCHEMA
    assert QUALIFICATION_RECEIPT_CONTRACT["native_runtime"]["status"] == EXACT_RUNTIME_STATUS
    assert QUALIFICATION_RECEIPT_CONTRACT["tg10"]["minimum_median_base_true_tps"] == 100.0
    assert QUALIFICATION_RECEIPT_CONTRACT["tg3"]["minimum_median_base_true_tps"] == 333.0


def test_operator_path_test_sends_once_and_persists_a_redacted_receipt(tmp_path: Path) -> None:
    sent: list[str] = []
    notifier, paths = _notifier(tmp_path, sent)

    first = notifier.test_delivery()
    second = notifier.test_delivery()

    assert first["sent"] is True
    assert second["sent"] is False
    assert len(sent) == 1
    assert "OPERATOR_PATH_TEST" in sent[0]
    receipt_path = Path(first["test_event"]["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    serialized = json.dumps(receipt, sort_keys=True)
    assert receipt["status"] == "DELIVERED"
    assert receipt["delivery"] == {"bot_api_http_status": 200, "message_date": 123, "message_id": 42}
    assert TOKEN not in serialized
    assert CHAT not in serialized
    assert paths.status.is_file()


def test_only_strict_complete_token_tg10_receipts_can_notify_tps(tmp_path: Path) -> None:
    sent: list[str] = []
    notifier, paths = _notifier(tmp_path, sent)
    notifier.arm()
    _tg10_receipt(
        paths,
        "qwen30",
        complete_bpw=1.2,
        router_tps=99999.0,
        median_base_true_tps=None,
        complete_native_model=False,
    )

    result = notifier.run_once()

    assert result["sent"] == 0
    assert sent == []


def test_qualifying_tg10_delivery_is_deduped(tmp_path: Path) -> None:
    sent: list[str] = []
    notifier, paths = _notifier(tmp_path, sent)
    _exact_runtime_receipt(paths, "qwen30")
    notifier.arm()
    _tg10_receipt(paths, "qwen30")

    first = notifier.run_once()
    second = notifier.run_once()

    assert first["sent"] == 1
    assert second["sent"] == 0
    assert len(sent) == 1
    assert "qwen_tg10_operational_pass" in sent[0]


def test_only_canonical_exact_full_token_runtime_receipt_notifies_once(tmp_path: Path) -> None:
    sent: list[str] = []
    notifier, paths = _notifier(tmp_path, sent)
    notifier.arm()
    receipt = _exact_runtime_receipt(paths, "qwen30")

    first = notifier.run_once()
    second = notifier.run_once()

    assert receipt.name == "QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json"
    assert first["sent"] == 1
    assert second["sent"] == 0
    assert len(sent) == 1
    assert "qwen_complete_native_runtime" in sent[0]
    assert "qwen30/complete-runtime/QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json" in sent[0]


def test_sealed_runtime_revocation_suppresses_old_runtime_tg_and_sends_one_correction(
    tmp_path: Path,
) -> None:
    sent: list[str] = []
    notifier, paths = _notifier(tmp_path, sent)
    notifier.arm()
    runtime = _exact_runtime_receipt(paths, "qwen30")
    first = notifier.run_once()
    assert first["sent"] == 1
    assert "qwen_complete_native_runtime" in sent[0]

    # The stale TG-shaped receipt arrives at the same time as an immutable
    # runtime correction. It must not emit a TPS notification, while the
    # correction itself is a single bounded operator event.
    _tg10_receipt(paths, "qwen30")
    supersession = _runtime_supersession(paths, "qwen30", runtime)
    second = notifier.run_once()
    third = notifier.run_once()

    assert supersession.name == "QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION.json"
    assert second["sent"] == 1
    assert third["sent"] == 0
    assert len(sent) == 2
    assert "qwen_native_runtime_revoked" in sent[1]
    assert "qwen_tg10_operational_pass" not in sent[1]
    status = json.loads(paths.status.read_text(encoding="utf-8"))
    assert status["workers"]["qwen30"]["runtime_authority_currently_eligible"] is False
    assert status["workers"]["qwen30"]["runtime_authority_state"] == "CURRENT_RUNTIME_REVOKED"


def test_qwen80_uses_its_own_canonical_exact_full_token_runtime_path(tmp_path: Path) -> None:
    sent: list[str] = []
    notifier, paths = _notifier(tmp_path, sent)
    notifier.arm()
    receipt = _exact_runtime_receipt(paths, "qwen80")

    result = notifier.run_once()

    assert receipt.name == "QWEN80_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json"
    assert result["sent"] == 1
    assert len(sent) == 1
    assert "QWEN80" in sent[0]


def test_legacy_or_incomplete_runtime_receipts_cannot_emit_native_runtime_event(tmp_path: Path) -> None:
    sent: list[str] = []
    notifier, paths = _notifier(tmp_path, sent)
    notifier.arm()
    legacy = paths.model("qwen30").root / "complete-runtime" / "QWEN30_COMPLETE_NATIVE_RUNTIME_RECEIPT.json"
    _write(
        legacy,
        seal(
            {
                "schema": EXACT_RUNTIME_SCHEMA,
                "status": EXACT_RUNTIME_STATUS,
                "runtime": {field: True for field in EXACT_RUNTIME_REQUIRED_FACTS},
            }
        ),
    )
    _exact_runtime_receipt(paths, "qwen30", runtime_override={"no_fallback": False})

    result = notifier.run_once()

    assert result["sent"] == 0
    assert sent == []


def test_invalid_credential_path_never_attempts_operator_test(tmp_path: Path) -> None:
    physical = tmp_path / "physical"
    paths = NotifierPaths(physical, tmp_path / "notifications")
    _worker(paths, "qwen30")
    _worker(paths, "qwen80")
    attempts: list[str] = []
    notifier = AscensionTelegramNotifier(
        paths=paths,
        credential_reader=lambda _service: None,
        sender=lambda *_args: attempts.append("sent") or {"bot_api_http_status": 200, "message_id": 1},
        remote_validator=lambda _token: True,
    )

    result = notifier.test_delivery()

    assert result["sent"] is False
    assert result["reason"] == "credentials_not_validated"
    assert attempts == []
