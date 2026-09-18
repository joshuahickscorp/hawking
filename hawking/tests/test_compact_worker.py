from pathlib import Path

import pytest

from hawking.compact_worker import (
    CompactPacket,
    CompactWorkerError,
    NoOpMutation,
    StaleAnchor,
    compile_mutation_intent,
    create_source_anchor,
    record_funnel_event,
    token_usage,
)


def _authority():
    return {"capabilities": ["repo.edit"], "mutation_lease": "LEASE-1"}


def test_source_anchor_created_and_compiles_exact_old_text(tmp_path: Path):
    source = tmp_path / "sample.py"
    source.write_text("one\ntwo\nthree\n", encoding="utf-8")
    anchor = create_source_anchor(tmp_path, source, start_line=2, end_line=2, symbol="two")

    assert anchor.anchor.startswith("SRC-")
    assert (tmp_path / ".hawking" / "anchors" / f"{anchor.anchor}.json").exists()
    proposal = compile_mutation_intent(
        {"s": "MUTATE", "anchor": anchor.anchor, "op": "replace", "body": "TWO\n", "tests": ["T-1"]},
        root=tmp_path, authority=_authority(), goal_id="G", workunit_id="W",
    )
    assert proposal.operations[0]["old_text"] == "two\n"
    assert proposal.operations[0]["new_text"] == "TWO\n"
    assert proposal.required_tests == ("T-1",)


def test_stale_anchor_rejected_locally(tmp_path: Path):
    source = tmp_path / "sample.py"
    source.write_text("one\ntwo\n", encoding="utf-8")
    anchor = create_source_anchor(tmp_path, source, start_line=2, end_line=2)
    source.write_text("one\nchanged\n", encoding="utf-8")
    with pytest.raises(StaleAnchor):
        compile_mutation_intent(
            {"s": "MUTATE", "anchor": anchor.anchor, "body": "new\n"},
            root=tmp_path, authority=_authority(),
        )


def test_no_op_rejected_locally(tmp_path: Path):
    source = tmp_path / "sample.py"
    source.write_text("same\n", encoding="utf-8")
    anchor = create_source_anchor(tmp_path, source, start_line=1, end_line=1)
    with pytest.raises(NoOpMutation):
        compile_mutation_intent(
            {"s": "MUTATE", "anchor": anchor.anchor, "body": "same\n"},
            root=tmp_path, authority=_authority(),
        )


def test_compact_provider_cannot_return_old_text(tmp_path: Path):
    source = tmp_path / "sample.py"
    source.write_text("same\n", encoding="utf-8")
    anchor = create_source_anchor(tmp_path, source, start_line=1, end_line=1)
    with pytest.raises(CompactWorkerError, match="COMPACT_PROVIDER_STATE_FORBIDDEN"):
        compile_mutation_intent(
            {"s": "MUTATE", "anchor": anchor.anchor, "body": "new\n", "old_text": "same\n"},
            root=tmp_path, authority=_authority(),
        )


def test_compact_packets_and_stage_usage_are_machine_small():
    packet = CompactPacket.from_mapping({"s": "DONE", "evidence": ["E-1"]})
    assert packet.to_dict() == {"schema": "hawking.compact_worker.v1", "s": "DONE", "evidence": ["E-1"]}
    usage = token_usage({"prompt_tokens": 10, "completion_tokens": 3, "cost_usd": 0.02}, stage="accepted_mutation", model="deepseek")
    assert usage == {"stage": "accepted_mutation", "model": "deepseek", "input_tokens": 10,
                     "cached_input_tokens": 0, "output_tokens": 3, "total_tokens": 13, "cost_usd": 0.02}


def test_compact_protocol_covers_read_plan_mutate_review_done_blocked():
    for status in ("READ", "PLAN", "MUTATE", "REVIEW", "DONE", "BLOCKED"):
        value = {"s": status}
        if status == "MUTATE":
            value.update({"anchor": "SRC-X", "body": "replacement"})
        assert CompactPacket.from_mapping(value).status == status


def test_stage_accounting_is_durable_and_bounded(tmp_path: Path):
    event = record_funnel_event(tmp_path, workunit_id="W", model="deepseek",
                                stage="anchor_valid", usage={"input_tokens": 4}, outcome="PASS")
    assert event["stage"] == "anchor_valid"
    stored = (tmp_path / ".hawking" / "telemetry" / "acceptance-funnel.json").read_text()
    assert '"workunit_id": "W"' in stored
    assert '"input_tokens": 4' in stored
