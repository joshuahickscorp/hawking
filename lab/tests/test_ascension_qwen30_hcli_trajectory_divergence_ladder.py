"""Pure unit tests for the fail-closed historical HCLI trajectory ladder."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lab.operators.ascension_qwen30_hcli_trajectory_divergence_ladder import (
    DivergenceLadderError,
    _blocked_ladder,
    _context_event_by_session,
    _exact_option,
)


def test_exact_option_refuses_ambiguous_or_missing_flag() -> None:
    assert _exact_option(["hcli", "run", "--prompt", "hello"], "--prompt", label="fixture") == "hello"
    with pytest.raises(DivergenceLadderError, match="exactly one --prompt"):
        _exact_option(["hcli", "--prompt", "one", "--prompt", "two"], "--prompt", label="fixture")


def test_context_without_exact_prefix_is_not_replayable(tmp_path: Path) -> None:
    event = {
        "id": "evt_fixture",
        "seq": 3,
        "session_id": "session_fixture",
        "kind": "context.compiled",
        "payload": {
            "retained": 2,
            "used_tokens": 9,
            "meter": {"used_estimated": True},
        },
        "chain_hash": "a" * 64,
    }
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps(event) + "\n", encoding="utf-8")
    observed = _context_event_by_session(events, "session_fixture")
    assert observed["replayability"] == "BLOCKED_PREFIX_TEXT_AND_TOKEN_IDS_NOT_PERSISTED"
    assert observed["selected_prefix_text_or_token_ids_persisted"] is False
    assert observed["event_log_sha256"] == hashlib.sha256(events.read_bytes()).hexdigest()


def test_context_with_visible_text_does_not_falsely_count_as_compiled_prefix(tmp_path: Path) -> None:
    event = {
        "id": "evt_fixture",
        "seq": 3,
        "session_id": "session_fixture",
        "kind": "context.compiled",
        "payload": {
            "retained": 2,
            "used_tokens": 9,
            "meter": {"used_estimated": True},
            # The user prompt exists in a separate event, not inside this
            # compiled event.  It therefore cannot authorize a replay.
        },
        "chain_hash": "a" * 64,
    }
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps(event) + "\n", encoding="utf-8")
    observed = _context_event_by_session(events, "session_fixture")
    assert observed["selected_prefix_text_or_token_ids_persisted"] is False


def test_all_post_prefix_stages_stay_blocked() -> None:
    ladder = _blocked_ladder()
    assert [row["stage"] for row in ladder] == [
        "template_tokenizer",
        "embedding",
        "attention_router",
        "moe",
        "final_norm_lm_head",
        "logit_top_k",
    ]
    assert {row["status"] for row in ladder} == {
        "NOT_EXECUTED_FAIL_CLOSED_EXACT_HISTORICAL_COMPILED_PREFIX_UNAVAILABLE"
    }
