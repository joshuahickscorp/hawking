"""Control-plane error text must not steer the next worker prompt.

HCLI records failed results into workspace knowledge and injects them as
`prior_knowledge`. One of its own error strings contains `<think>`,
`reasoning_content` and `enable_thinking`. Pasting those tokens into the
next prompt makes the resident emit a reasoning-only reply, which the
engine rejects, which is recorded as another failure containing the same
tokens. Sanitise on write (the producer) and on read (`snapshot`) so
already-poisoned on-disk records cannot steer either.
"""
from __future__ import annotations

import json

from hcli.knowledge import KnowledgeStore, sanitise_knowledge

HCLI_THINKING_ERROR = (
    "llama-server ignored chat_template_kwargs.enable_thinking=false "
    "(reply contained a <think> block or reasoning_content). "
    "Start llama-server with --jinja so the reasoning policy "
    "can take effect."
)

SCIENTIFIC_REASON = "test_expert_rank failed: rank90 812 exceeds budget"

_STEERING_TOKENS = ("<think>", "reasoning_content", "enable_thinking")


def _blob(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _assert_not_steering(blob: str) -> None:
    lowered = blob.lower()
    for token in _STEERING_TOKENS:
        assert token not in lowered, f"steering token {token!r} leaked into prompt-facing knowledge"


def test_snapshot_neutralises_the_hcli_thinking_error(tmp_path):
    store = KnowledgeStore(tmp_path)
    recorded = store.record_result(
        "do the 76-char task",
        {"status": "failed", "reason": HCLI_THINKING_ERROR},
    )
    assert recorded is not None
    snapshot = store.snapshot()
    blob = _blob(snapshot)
    _assert_not_steering(blob)
    assert "llama-server" in blob
    assert "jinja" in blob
    assert "[think-tag]" in blob
    assert "[reasoning-field]" in blob
    assert "[thinking-flag]" in blob
    assert "[chat-template-kwargs]" in blob
    assert "[reasoning]" in blob


def test_snapshot_scrubs_poison_already_on_disk_without_rewriting_it(tmp_path):
    """The live index is generation 2127 of poisoned records; do not migrate it."""
    store = KnowledgeStore(tmp_path)
    poisoned = {
        "schema": "hcli.workspace_knowledge.v1",
        "generation": 2127,
        "updated_at": 0,
        "records": [
            {
                "id": "knowledge-deadbeef",
                "at": 1.0,
                "kind": "result_claim",
                "source": "hcli_result",
                "priority": 65,
                "verified": False,
                "fingerprint": "abc",
                "data": {
                    "goal": "do the task",
                    "status": "failed",
                    "reason": HCLI_THINKING_ERROR,
                },
            }
        ],
    }
    store.path.write_text(json.dumps(poisoned), encoding="utf-8")

    snapshot = store.snapshot()
    blob = _blob(snapshot)
    _assert_not_steering(blob)
    assert "llama-server" in blob
    assert "jinja" in blob
    assert snapshot["generation"] == 2127

    on_disk = json.loads(store.path.read_text(encoding="utf-8"))
    disk_reason = on_disk["records"][0]["data"]["reason"]
    assert disk_reason == HCLI_THINKING_ERROR
    assert "<think>" in disk_reason


def test_sanitise_is_idempotent():
    once = sanitise_knowledge(HCLI_THINKING_ERROR)
    twice = sanitise_knowledge(once)
    assert once == twice
    _assert_not_steering(once)

    record = {
        "data": {"reason": HCLI_THINKING_ERROR, "nested": ["<THINK>", "</think>"]},
        "kind": "result_claim",
    }
    assert sanitise_knowledge(sanitise_knowledge(record)) == sanitise_knowledge(record)


def test_sanitise_does_not_raise_on_odd_input():
    long_text = ("neutral " * 20000) + "<think>" + (" pad" * 20000)
    samples = (
        None,
        0,
        1,
        -7,
        3.14,
        True,
        False,
        "",
        b"bytes with <think> and reasoning_content",
        bytearray(b"enable_thinking"),
        {"a": {"b": ["<think>", 2, None]}},
        ["nested", {"reasoning_content": "<|im_start|>"}],
        ("tuple", "<think>"),
        long_text,
        object(),
        {"reasoning_content": "<think>"},
    )
    for sample in samples:
        sanitise_knowledge(sample)


def test_scientific_reason_passes_through_unchanged():
    assert sanitise_knowledge(SCIENTIFIC_REASON) == SCIENTIFIC_REASON
    store_payload = {"goal": "rank experts", "status": "failed", "reason": SCIENTIFIC_REASON}
    assert sanitise_knowledge(store_payload) == store_payload


def test_scientific_reason_survives_record_and_snapshot(tmp_path):
    store = KnowledgeStore(tmp_path)
    store.record_result(
        "rank experts",
        {"status": "failed", "reason": SCIENTIFIC_REASON},
    )
    blob = _blob(store.snapshot())
    assert SCIENTIFIC_REASON in blob
    _assert_not_steering(blob)


def test_similar_chat_control_tags_and_special_tokens_are_neutralised():
    text = (
        "<THINK>secret</THINK> "
        "<thinking>also</thinking> "
        "<redacted_thinking/> "
        "<|im_start|>system<|im_end|> "
        "hidden_reasoning chain_of_thought"
    )
    cleaned = sanitise_knowledge(text)
    lowered = cleaned.lower()
    assert "<think>" not in lowered
    assert "</think>" not in lowered
    assert "<thinking>" not in lowered
    assert "<|im_start|>" not in cleaned
    assert "<|im_end|>" not in cleaned
    assert "hidden_reasoning" not in lowered
    assert "chain_of_thought" not in lowered
    assert "[think-tag]" in cleaned
    assert "[special-token]" in cleaned
    assert "secret" in cleaned
    assert "also" in cleaned


def test_recall_scrubs_poison_already_on_disk(tmp_path):
    store = KnowledgeStore(tmp_path)
    poisoned = {
        "schema": "hcli.workspace_knowledge.v1",
        "generation": 2127,
        "updated_at": 0,
        "records": [
            {
                "id": "knowledge-deadbeef",
                "at": 1.0,
                "kind": "result_claim",
                "source": "hcli_result",
                "priority": 65,
                "verified": False,
                "fingerprint": "abc",
                "data": {
                    "goal": "do the task",
                    "status": "failed",
                    "reason": HCLI_THINKING_ERROR,
                },
            }
        ],
    }
    store.path.write_text(json.dumps(poisoned), encoding="utf-8")
    blob = _blob(store.recall("do the task"))
    _assert_not_steering(blob)
    assert "llama-server" in blob


def test_write_path_stores_sanitised_records(tmp_path):
    store = KnowledgeStore(tmp_path)
    store.record_result(
        "do the task",
        {"status": "failed", "reason": HCLI_THINKING_ERROR},
    )
    on_disk = json.loads(store.path.read_text(encoding="utf-8"))
    disk_blob = _blob(on_disk)
    _assert_not_steering(disk_blob)
    assert "llama-server" in disk_blob
