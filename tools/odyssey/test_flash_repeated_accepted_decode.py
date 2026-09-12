from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).with_name("flash_repeated_accepted_decode.py")
SPEC = importlib.util.spec_from_file_location("flash_repeated_accepted_decode", SOURCE)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_reference_contract_requires_exact_terminal_chain_and_state_census():
    good = {
        "model": runner.REPO_ID,
        "pinned_revision": runner.PINNED_REVISION,
        "status": "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE",
        "accepted_generation_tokens": 2,
        "terminal": {"reference_checks": [
            {"expected_token_id": 2972, "predicted_token_id": 2972, "accepted": True},
            {"expected_token_id": 75, "predicted_token_id": 75, "accepted": True},
        ]},
        "execution": {
            "process_boundary": "one native process",
            "source_reset_or_reprefill": False,
            "source_payload_bytes_read": 8192,
            "state_memory": {
                "total_persistent_bytes": 4096,
                "growth_bytes_per_additional_token": 512,
            },
        },
    }
    runner._verify_repeated(good, [2972, 75])

    bad = json.loads(json.dumps(good))
    bad["execution"]["source_reset_or_reprefill"] = True
    try:
        runner._verify_repeated(bad, [2972, 75])
    except ValueError as exc:
        assert "reset/re-prefill" in str(exc)
    else:
        raise AssertionError("resetting a stateful verification session must fail")

    wrong_source = json.loads(json.dumps(good))
    wrong_source["pinned_revision"] = "unsealed"
    try:
        runner._verify_repeated(wrong_source, [2972, 75])
    except ValueError as exc:
        assert "revision" in str(exc)
    else:
        raise AssertionError("a receipt from a different source revision must fail")

    wrong_model = json.loads(json.dumps(good))
    wrong_model["model"] = "unrelated/model"
    try:
        runner._verify_repeated(wrong_model, [2972, 75])
    except ValueError as exc:
        assert "model" in str(exc)
    else:
        raise AssertionError("a receipt from a different source model must fail")

    wrong_process_boundary = json.loads(json.dumps(good))
    wrong_process_boundary["execution"]["process_boundary"] = "multiple processes"
    try:
        runner._verify_repeated(wrong_process_boundary, [2972, 75])
    except ValueError as exc:
        assert "one native process" in str(exc)
    else:
        raise AssertionError("a multi-process receipt must not establish the stateful gate")


def test_dry_run_is_flash_specific_and_never_starts_gpu_work(capsys):
    assert runner.main(["--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "DRY_RUN"
    assert "flash_stateful_complete_token_session" in payload["phase_1"]
    assert payload["claim_boundary"].startswith("no GPU work")


def test_gpu_preflight_ignores_only_reaped_provider_children(monkeypatch):
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="Z    mlx_vlm.server --model kimi\n"
        ),
    )
    assert runner._hawking_gpu_is_busy() is False

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="S    mlx_vlm.server --model kimi\n"
        ),
    )
    assert runner._hawking_gpu_is_busy() is True
