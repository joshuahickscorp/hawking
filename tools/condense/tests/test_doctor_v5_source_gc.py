from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools" / "condense" / "doctor_v5_source_gc.py"
SPEC = importlib.util.spec_from_file_location("doctor_v5_source_gc", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["doctor_v5_source_gc"] = MODULE
SPEC.loader.exec_module(MODULE)

MODEL = "qwen2-5-14b"


def _fixture(base: pathlib.Path, **kwargs):
    return MODULE._make_fixture(base, **kwargs)


def test_refuses_when_any_cell_non_terminal(tmp_path: pathlib.Path) -> None:
    config = _fixture(tmp_path, statuses=["complete", "complete", "pending"])
    outcome = MODULE.run_once(config)["outcomes"][MODEL]
    assert outcome["refused"] is True
    assert any("non-terminal" in reason for reason in outcome["reasons"])
    assert config.staging_map[MODEL].exists()
    assert not list(config.gc_dir.glob(f"{MODEL}_*.json"))


def test_refuses_running_and_blocked_statuses(tmp_path: pathlib.Path) -> None:
    for status in ("running", "blocked-execution", "blocked-dependency"):
        base = tmp_path / status
        config = _fixture(base, statuses=["complete", status])
        outcome = MODULE.run_once(config)["outcomes"][MODEL]
        assert outcome["refused"] is True
        assert config.staging_map[MODEL].exists()


def test_refuses_during_transaction(tmp_path: pathlib.Path) -> None:
    config = _fixture(tmp_path, transaction_phase="state-mutated")
    outcome = MODULE.run_once(config)["outcomes"][MODEL]
    assert outcome["refused"] is True
    assert any("transaction in flight" in reason for reason in outcome["reasons"])
    assert config.staging_map[MODEL].exists()


def test_transaction_phase_complete_allows(tmp_path: pathlib.Path) -> None:
    config = _fixture(tmp_path, transaction_phase="complete")
    outcome = MODULE.run_once(config)["outcomes"][MODEL]
    assert outcome["deleted"] is True


def test_refuses_120b_before_320_of_320(tmp_path: pathlib.Path) -> None:
    config = _fixture(tmp_path, model="gpt-oss-120b", campaign_total=8)
    config = dataclasses.replace(config, campaign_total_cells=320)
    outcome = MODULE.run_once(config)["outcomes"]["gpt-oss-120b"]
    assert outcome["refused"] is True
    assert any("!= 320" in reason for reason in outcome["reasons"])
    assert config.staging_map["gpt-oss-120b"].exists()


def test_refuses_120b_with_320_cells_one_open(tmp_path: pathlib.Path) -> None:
    config = _fixture(
        tmp_path,
        model="gpt-oss-120b",
        statuses=["complete", "complete", "complete", "pending"],
        campaign_total=320,
    )
    config = dataclasses.replace(config, campaign_total_cells=320)
    outcome = MODULE.run_once(config)["outcomes"]["gpt-oss-120b"]
    assert outcome["refused"] is True
    assert any("hard gate" in reason or "non-terminal" in reason
               for reason in outcome["reasons"])


def test_allows_120b_at_320_of_320(tmp_path: pathlib.Path) -> None:
    config = _fixture(tmp_path, model="gpt-oss-120b", campaign_total=320)
    config = dataclasses.replace(config, campaign_total_cells=320)
    outcome = MODULE.run_once(config)["outcomes"]["gpt-oss-120b"]
    assert outcome["deleted"] is True


def test_deletes_and_receipts_when_sealed(tmp_path: pathlib.Path) -> None:
    config = _fixture(tmp_path)
    directory = config.staging_map[MODEL]
    payload = directory / "shard-000.bin"
    expected_sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    outcome = MODULE.run_once(config)["outcomes"][MODEL]
    assert outcome["deleted"] is True
    assert not directory.exists()

    receipt = json.loads(pathlib.Path(outcome["receipt"]).read_text())
    assert receipt["schema"] == "hawking.operator_source_gc_receipt.v1"
    assert receipt["quality_claims_permitted"] is False
    assert "parent_source_cleanup=disabled_separate_operator_action_only" \
        in receipt["authority"]
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    assert MODULE._hash_value(body) == receipt["receipt_sha256"]
    deleted = receipt["deleted"][0]
    assert deleted["file_count"] == 2
    rows = {row["path"]: row for row in deleted["files"]}
    rel = str(payload.relative_to(config.root))
    assert rows[rel]["sha256"] == expected_sha
    assert deleted["total_bytes"] == sum(row["bytes"] for row in deleted["files"])

    # idempotent second pass: dir gone, no new receipt
    receipts_before = sorted(config.gc_dir.glob(f"{MODEL}_*.json"))
    second = MODULE.run_once(config)["outcomes"][MODEL]
    assert second == {"model": MODEL, "deleted": False, "skip": "dir gone"}
    assert sorted(config.gc_dir.glob(f"{MODEL}_*.json")) == receipts_before


def test_refuses_when_reporter_not_sealed(tmp_path: pathlib.Path) -> None:
    config = _fixture(tmp_path, reporter_ok=False)
    outcome = MODULE.run_once(config)["outcomes"][MODEL]
    assert outcome["refused"] is True
    assert any("reporter not sealed" in reason for reason in outcome["reasons"])


def test_refuses_when_last_reporter_sync_stale_or_bad(tmp_path: pathlib.Path) -> None:
    config = _fixture(tmp_path)
    state_path = config.queue_state_path
    state = json.loads(state_path.read_text())
    state["last_reporter_sync"] = {"ok": False, "at": "2026-07-16T12:00:00+00:00"}
    state_path.write_text(json.dumps(state))
    outcome = MODULE.run_once(config)["outcomes"][MODEL]
    assert outcome["refused"] is True

    state["last_reporter_sync"] = {"ok": True, "at": "2026-07-16T09:00:00+00:00"}
    state_path.write_text(json.dumps(state))
    outcome = MODULE.run_once(config)["outcomes"][MODEL]
    assert outcome["refused"] is True
    assert any("predates" in reason for reason in outcome["reasons"])


def test_refuses_active_cell_reference(tmp_path: pathlib.Path) -> None:
    config = _fixture(tmp_path)
    state_path = config.queue_state_path
    state = json.loads(state_path.read_text())
    state["active_cells"] = [f"{MODEL}__0bpw__codec-control"]
    state_path.write_text(json.dumps(state))
    outcome = MODULE.run_once(config)["outcomes"][MODEL]
    assert outcome["refused"] is True
    assert any("active cell" in reason for reason in outcome["reasons"])


def test_refuses_bad_state_schema(tmp_path: pathlib.Path) -> None:
    config = _fixture(tmp_path)
    state_path = config.queue_state_path
    state = json.loads(state_path.read_text())
    state["schema"] = "hawking.some_other_schema.v9"
    state_path.write_text(json.dumps(state))
    with pytest.raises(MODULE.GcError, match="schema mismatch"):
        MODULE.run_once(config)
    assert config.staging_map[MODEL].exists()


def test_ledger_self_hash_chain_verifies(tmp_path: pathlib.Path) -> None:
    config = _fixture(tmp_path)
    assert MODULE.run_once(config)["outcomes"][MODEL]["deleted"] is True
    directory = config.staging_map[MODEL]
    directory.mkdir(parents=True)
    (directory / "again.bin").write_bytes(b"b" * 128)
    assert MODULE.run_once(config)["outcomes"][MODEL]["deleted"] is True

    entries = MODULE.verify_ledger(config)
    assert len(entries) == 2
    assert entries[0]["prev_entry_sha256"] is None
    assert entries[1]["prev_entry_sha256"] == entries[0]["entry_sha256"]
    for entry in entries:
        body = {k: v for k, v in entry.items() if k != "entry_sha256"}
        assert MODULE._hash_value(body) == entry["entry_sha256"]

    # tamper detection: flip a byte in the first line
    raw = config.ledger_path.read_text().splitlines()
    tampered = json.loads(raw[0])
    tampered["total_bytes"] = 999999
    raw[0] = json.dumps(tampered, sort_keys=True, ensure_ascii=False)
    config.ledger_path.write_text("\n".join(raw) + "\n")
    with pytest.raises(MODULE.GcError, match="self-hash mismatch"):
        MODULE.verify_ledger(config)


def test_symlink_refusal(tmp_path: pathlib.Path) -> None:
    # symlink inside the staging dir
    config = _fixture(tmp_path / "inner")
    directory = config.staging_map[MODEL]
    (directory / "escape").symlink_to(tmp_path)
    with pytest.raises(MODULE.GcError, match="symlink refused"):
        MODULE.run_once(config)
    assert directory.exists()
    assert not list(config.gc_dir.glob(f"{MODEL}_*.json"))

    # staging dir itself is a symlink
    config = _fixture(tmp_path / "outer")
    directory = config.staging_map[MODEL]
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.bin").write_bytes(b"k")
    replacement = directory.parent / "link.partial"
    replacement.symlink_to(victim)
    config = dataclasses.replace(config, staging_map={MODEL: replacement})
    outcome = MODULE.run_once(config)["outcomes"][MODEL]
    assert outcome["refused"] is True
    assert any("symlink" in reason for reason in outcome["reasons"])
    assert (victim / "keep.bin").exists()


def test_head_tail_hash_scheme(tmp_path: pathlib.Path, monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "HEAD_TAIL_THRESHOLD", 256)
    monkeypatch.setattr(MODULE, "HEAD_TAIL_WINDOW", 64)
    small = tmp_path / "small.bin"
    small.write_bytes(b"s" * 200)
    row = MODULE._hash_regular_file(small)
    assert row["sha256"] == hashlib.sha256(b"s" * 200).hexdigest()
    big = tmp_path / "big.bin"
    payload = bytes(range(256)) * 4  # 1024 bytes
    big.write_bytes(payload)
    row = MODULE._hash_regular_file(big)
    expected = hashlib.sha256(payload[:64] + payload[-64:]).hexdigest()
    assert row["sha256_head_tail_64m"] == expected
    assert "sha256" not in row


def test_selftest_is_green() -> None:
    result = MODULE.selftest()
    assert result["ok"] is True
    assert "deletes-when-sealed" in result["checks"]
    assert "symlink-refusal" in result["checks"]
