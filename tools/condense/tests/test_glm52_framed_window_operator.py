from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path

import pytest

from lab.lease import FixtureHeavyLease, FixtureLeaseError
from lab.operators.glm52_framed_window_operator import (
    FIXTURE_PROTOCOL,
    FixtureFramedWindowOperator,
    FramedWindowOperatorError,
    LocalFixtureColdStorage,
    frame_stream,
)
from lab.operators.glm52_range_stream_executor import RangeExecutorError, read_frame, write_frame
from ramanujan.restream_guard import ACCOUNTING_COMPONENTS, ALIGNMENT_BYTES


def _round_up(value: int) -> int:
    return ((value + ALIGNMENT_BYTES - 1) // ALIGNMENT_BYTES) * ALIGNMENT_BYTES


def _accounting(*payloads: bytes, source_component_bytes: int | None = None) -> dict[str, int]:
    source_bytes = sum(_round_up(len(payload)) for payload in payloads)
    accounting = {component: 0 for component in ACCOUNTING_COMPONENTS}
    accounting["source_range_rounded_bytes"] = source_component_bytes if source_component_bytes is not None else source_bytes
    # The fixture one-pass manifest and its sealed terminal receipt are each
    # deliberately budgeted at the contract's conservative minimum.
    accounting["retained_artifact_bytes"] = ALIGNMENT_BYTES
    accounting["metadata_bytes"] = ALIGNMENT_BYTES
    accounting["resident_incremental_bytes"] = sum(accounting.values())
    return accounting


def _messages(
    *payloads: bytes,
    floor_bytes: int = 0,
    accounting: dict[str, int] | None = None,
):
    range_ids = [f"range-{number}" for number in range(len(payloads))]
    window_accounting = accounting or _accounting(*payloads)
    messages: list[tuple[dict[str, object], bytes]] = [
        (
            {
                "kind": "WINDOW",
                "protocol": FIXTURE_PROTOCOL,
                "fixture_only": True,
                "window_id": "small-real-window",
                "execution_order": 0,
                "range_count": len(payloads),
                "ordered_range_ids": range_ids,
                "schedule_seal_sha256": "a" * 64,
                "policy_seal_sha256": "b" * 64,
                "incremental_bytes": window_accounting["resident_incremental_bytes"],
                "protected_floor_bytes": floor_bytes,
                "incremental_accounting": window_accounting,
                "payload_bytes": 0,
            },
            b"",
        )
    ]
    for number, payload in enumerate(payloads):
        messages.append(
            (
                {
                    "kind": "RANGE",
                    "range_id": range_ids[number],
                    "shard": f"fixture-{number}.bin",
                    "start": 0,
                    "end": len(payload),
                    "payload_bytes": len(payload),
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                },
                payload,
            )
        )
    messages.append(({"kind": "END_WINDOW", "window_id": "small-real-window", "payload_bytes": 0}, b""))
    return messages


def _lease(tmp_path: Path, *, timeout: float = 30.0) -> FixtureHeavyLease:
    lease = FixtureHeavyLease(
        tmp_path / "fixture-heavy.lease",
        campaign_id="framed-operator-test",
        heartbeat_timeout_seconds=timeout,
    )
    lease.acquire(contention_label="CLEAN")
    return lease


def test_exact_frame_parser_rejects_truncation_and_non_range_payload() -> None:
    encoded = io.BytesIO()
    write_frame(encoded, {"kind": "RANGE", "payload_bytes": 3}, [b"abc"])
    encoded.seek(0)
    message = read_frame(encoded, max_payload_bytes=3)
    assert message.header["kind"] == "RANGE"
    assert message.payload == b"abc"

    malformed = io.BytesIO()
    write_frame(malformed, {"kind": "END_WINDOW"}, ())
    malformed.write(b"ignored")
    malformed.seek(0)
    assert read_frame(malformed).header["kind"] == "END_WINDOW"
    # The second parse encounters bytes that cannot form a full header, rather
    # than silently treating arbitrary suffix bytes as another message.
    with pytest.raises(RangeExecutorError, match="truncated"):
        read_frame(malformed)


def test_small_real_framed_dry_run_hashes_handoffs_evicts_and_resumes(tmp_path: Path) -> None:
    lease = _lease(tmp_path)
    workspace = tmp_path / "workspace"
    try:
        operator = FixtureFramedWindowOperator(
            workspace_root=workspace,
            cold_storage=LocalFixtureColdStorage(tmp_path / "cold"),
            lease=lease,
        )
        # These bytes come from a real local file write/read path, not a mocked
        # hash return; the operator performs its own staged-file hash check.
        small_real = b"real small fixture range\n" * 7
        receipt = operator.run(frame_stream(_messages(small_real, b"second-range")))
        assert receipt["status"] == "FIXTURE_ONLY_PASS_REMOTE_HASHED_EVICTED"
        assert receipt["fixture_only"] is True
        assert receipt["production_authority"] is False
        assert receipt["range_count"] == 2
        assert receipt["artifact_handoff"]["remote_sha256_verified"] is True
        assert receipt["eviction"]["source_ranges_retained_zero"] is True
        assert receipt["eviction"]["local_artifact_bytes_retained_zero"] is True
        assert "exact_cache_items_purged" not in receipt["eviction"]
        assert "exact_trash_items_purged" not in receipt["eviction"]
        assert receipt["cache_semantics"]["cache_created"] is False
        assert receipt["trash_semantics"]["trash_created"] is False
        assert not (workspace / ".staging" / "small-real-window").exists()
        assert list((tmp_path / "cold").iterdir())

        resumed = operator.run(frame_stream(_messages(small_real, b"second-range")))
        assert resumed == receipt

        mismatched = _messages(small_real, b"second-range")
        mismatched[0][0]["execution_order"] = 1
        with pytest.raises(FramedWindowOperatorError, match="does not bind"):
            operator.run(frame_stream(mismatched))
    finally:
        lease.release()


def test_fixture_operator_refuses_symlinked_resume_receipt(tmp_path: Path) -> None:
    lease = _lease(tmp_path)
    workspace = tmp_path / "workspace"
    try:
        operator = FixtureFramedWindowOperator(
            workspace_root=workspace,
            cold_storage=LocalFixtureColdStorage(tmp_path / "cold"),
            lease=lease,
        )
        operator.run(frame_stream(_messages(b"resume-symlink")))
        receipt = workspace / "receipts" / "small-real-window.json"
        moved = tmp_path / "moved-receipt.json"
        receipt.rename(moved)
        receipt.symlink_to(moved)
        with pytest.raises(FramedWindowOperatorError, match="non-symlink regular file"):
            operator.run(frame_stream(_messages(b"resume-symlink")))
    finally:
        lease.release()


def test_fixture_operator_refuses_trailing_bytes_and_same_shard_overlap(tmp_path: Path) -> None:
    lease = _lease(tmp_path)
    try:
        operator = FixtureFramedWindowOperator(
            workspace_root=tmp_path / "workspace",
            cold_storage=LocalFixtureColdStorage(tmp_path / "cold"),
            lease=lease,
        )
        trailing = frame_stream(_messages(b"one"))
        trailing.seek(0, io.SEEK_END)
        trailing.write(b"trailing")
        trailing.seek(0)
        with pytest.raises(FramedWindowOperatorError, match="trailing bytes"):
            operator.run(trailing)

        overlapping = _messages(b"first", b"second")
        overlapping[2][0]["shard"] = overlapping[1][0]["shard"]
        with pytest.raises(FramedWindowOperatorError, match="ordered and non-overlapping"):
            operator.run(frame_stream(overlapping))
    finally:
        lease.release()


def test_fixture_operator_checks_floor_during_range_write(tmp_path: Path) -> None:
    lease = _lease(tmp_path)
    calls = {"count": 0}

    def sampled_free(_path: Path) -> int:
        calls["count"] += 1
        return 1_000_000 if calls["count"] < 4 else 99

    workspace = tmp_path / "workspace"
    try:
        operator = FixtureFramedWindowOperator(
            workspace_root=workspace,
            cold_storage=LocalFixtureColdStorage(tmp_path / "cold"),
            lease=lease,
            disk_free_bytes=sampled_free,
        )
        with pytest.raises(FramedWindowOperatorError, match="protected floor crossed at after_range_write"):
            operator.run(frame_stream(_messages(b"floor-crossing", floor_bytes=100)))
        assert not (workspace / ".staging" / "small-real-window").exists()
    finally:
        lease.release()


def test_fixture_rollback_reports_failed_cold_cleanup_without_overclaim(tmp_path: Path) -> None:
    class DeleteFailureColdStore:
        def __init__(self, root: Path) -> None:
            self.delegate = LocalFixtureColdStorage(root)

        def put_verified(self, *, key: str, source: Path, sha256: str):
            return self.delegate.put_verified(key=key, source=source, sha256=sha256)

        def delete(self, *, key: str) -> None:
            raise OSError(f"injected delete failure for {key}")

    lease = _lease(tmp_path)
    workspace = tmp_path / "workspace"
    try:
        operator = FixtureFramedWindowOperator(
            workspace_root=workspace,
            cold_storage=DeleteFailureColdStore(tmp_path / "cold"),
            lease=lease,
            failure_stage="after_handoff",
        )
        with pytest.raises(FramedWindowOperatorError, match="after cold handoff"):
            operator.run(frame_stream(_messages(b"cleanup-failure")))
        rollback = json.loads((workspace / "rollback" / "small-real-window.json").read_text())
        assert rollback["status"] == "FIXTURE_ONLY_ROLLBACK_INCOMPLETE_FAIL_CLOSED"
        assert rollback["cleanup_complete"] is False
        assert rollback["local_cleanup_verified"] is True
        assert rollback["remote_cleanup_verified"] is False
        assert rollback["cleanup_errors"]
    finally:
        lease.release()


def test_failed_cold_upload_rolls_back_local_staging(tmp_path: Path) -> None:
    class FailingPutColdStore:
        def put_verified(self, *, key: str, source: Path, sha256: str):
            raise OSError("injected cold upload interruption")

        def delete(self, *, key: str) -> None:
            raise AssertionError("no remote object exists to delete after failed upload")

    lease = _lease(tmp_path)
    workspace = tmp_path / "workspace"
    try:
        operator = FixtureFramedWindowOperator(
            workspace_root=workspace,
            cold_storage=FailingPutColdStore(),
            lease=lease,
        )
        with pytest.raises(OSError, match="cold upload interruption"):
            operator.run(frame_stream(_messages(b"upload-failure")))
        assert not (workspace / ".staging" / "small-real-window").exists()
        rollback = json.loads((workspace / "rollback" / "small-real-window.json").read_text())
        assert rollback["cleanup_complete"] is True
        assert rollback["remote_cleanup_verified"] is True
    finally:
        lease.release()


def test_cold_handoff_requires_exact_remote_hash_and_evicts_attempt_object(tmp_path: Path) -> None:
    class WrongRemoteHashColdStore:
        def __init__(self, root: Path) -> None:
            self.delegate = LocalFixtureColdStorage(root)

        def put_verified(self, *, key: str, source: Path, sha256: str):
            handoff = self.delegate.put_verified(key=key, source=source, sha256=sha256)
            return {
                **handoff,
                "remote_sha256": "0" * 64,
                "remote_sha256_verified": False,
            }

        def delete(self, *, key: str) -> None:
            self.delegate.delete(key=key)

    lease = _lease(tmp_path)
    workspace = tmp_path / "workspace"
    cold = tmp_path / "cold"
    try:
        operator = FixtureFramedWindowOperator(
            workspace_root=workspace,
            cold_storage=WrongRemoteHashColdStore(cold),
            lease=lease,
        )
        with pytest.raises(FramedWindowOperatorError, match="exact remote artifact hash"):
            operator.run(frame_stream(_messages(b"remote-hash-mismatch")))
        assert not (workspace / ".staging" / "small-real-window").exists()
        assert not list(cold.glob("*.fixture")) if cold.exists() else True
        rollback = json.loads((workspace / "rollback" / "small-real-window.json").read_text())
        assert rollback["cleanup_complete"] is True
        assert rollback["remote_cleanup_verified"] is True
    finally:
        lease.release()


def test_interrupted_staging_is_evicted_before_restart_safe_replay(tmp_path: Path) -> None:
    lease = _lease(tmp_path)
    workspace = tmp_path / "workspace"
    interrupted = workspace / ".staging" / "small-real-window"
    interrupted.mkdir(parents=True)
    (interrupted / "partial-range.bin").write_bytes(b"interrupted-local-write")
    try:
        operator = FixtureFramedWindowOperator(
            workspace_root=workspace,
            cold_storage=LocalFixtureColdStorage(tmp_path / "cold"),
            lease=lease,
        )
        receipt = operator.run(frame_stream(_messages(b"restart-safe")))
        assert receipt["resume"]["recovered_partial_staging"] is True
        assert not interrupted.exists()
        assert receipt["eviction"]["source_ranges_retained_zero"] is True
        assert receipt["eviction"]["local_artifact_bytes_retained_zero"] is True
    finally:
        lease.release()


@pytest.mark.parametrize("stage", ["after_range", "after_pack", "after_handoff"])
def test_failure_injection_rolls_back_source_artifact_and_cold_handoff(tmp_path: Path, stage: str) -> None:
    lease = _lease(tmp_path)
    workspace = tmp_path / "workspace"
    cold = tmp_path / "cold"
    try:
        operator = FixtureFramedWindowOperator(
            workspace_root=workspace,
            cold_storage=LocalFixtureColdStorage(cold),
            lease=lease,
            failure_stage=stage,
        )
        with pytest.raises(FramedWindowOperatorError, match="injected fixture failure"):
            operator.run(frame_stream(_messages(b"failure-path")))
        assert not (workspace / ".staging" / "small-real-window").exists()
        assert (workspace / "rollback" / "small-real-window.json").is_file()
        assert not list(cold.glob("*.fixture")) if cold.exists() else True
    finally:
        lease.release()


def test_rollback_never_deletes_an_idempotent_preexisting_cold_object(tmp_path: Path) -> None:
    lease = _lease(tmp_path)
    workspace = tmp_path / "workspace"
    cold = tmp_path / "cold"
    try:
        good = FixtureFramedWindowOperator(
            workspace_root=workspace,
            cold_storage=LocalFixtureColdStorage(cold),
            lease=lease,
        )
        payload = b"preexisting-remote-object"
        receipt = good.run(frame_stream(_messages(payload)))
        cold_objects = list(cold.glob("*.fixture"))
        assert len(cold_objects) == 1
        remote_bytes = cold_objects[0].read_bytes()
        (workspace / "receipts" / "small-real-window.json").unlink()

        failing = FixtureFramedWindowOperator(
            workspace_root=workspace,
            cold_storage=LocalFixtureColdStorage(cold),
            lease=lease,
            failure_stage="after_handoff",
        )
        with pytest.raises(FramedWindowOperatorError, match="after cold handoff"):
            failing.run(frame_stream(_messages(payload)))
        assert cold_objects[0].read_bytes() == remote_bytes
        assert receipt["artifact_handoff"]["created_new_object"] is True
    finally:
        lease.release()


def test_fixture_operator_refuses_over_envelope_before_any_body(tmp_path: Path) -> None:
    lease = _lease(tmp_path)
    try:
        operator = FixtureFramedWindowOperator(
            workspace_root=tmp_path / "workspace",
            cold_storage=LocalFixtureColdStorage(tmp_path / "cold"),
            lease=lease,
        )
        over_envelope = _accounting(b"never-written")
        over_envelope["source_range_rounded_bytes"] = _round_up(90_000_000_000)
        over_envelope["retained_artifact_bytes"] = 0
        over_envelope["metadata_bytes"] = 0
        over_envelope["resident_incremental_bytes"] = sum(
            over_envelope[component] for component in ACCOUNTING_COMPONENTS
        )
        with pytest.raises(FramedWindowOperatorError, match="<=90-GB"):
            operator.run(frame_stream(_messages(b"never-written", accounting=over_envelope)))
        assert not (tmp_path / "workspace" / ".staging").exists()
    finally:
        lease.release()


def test_underdeclared_window_is_rejected_before_the_excess_range_is_staged(tmp_path: Path) -> None:
    lease = _lease(tmp_path)
    workspace = tmp_path / "workspace"
    try:
        operator = FixtureFramedWindowOperator(
            workspace_root=workspace,
            cold_storage=LocalFixtureColdStorage(tmp_path / "cold"),
            lease=lease,
        )
        # Each 16-KiB body conservatively charges a full 64-KiB range.  The
        # sender declares capacity for only one, so the second body must fail
        # before its source path is created.
        payload = b"x" * (ALIGNMENT_BYTES // 4)
        underdeclared = _accounting(payload, payload, source_component_bytes=ALIGNMENT_BYTES)
        with pytest.raises(FramedWindowOperatorError, match="cumulative 64-KiB source ranges"):
            operator.run(frame_stream(_messages(payload, payload, accounting=underdeclared)))
        assert not (workspace / ".staging" / "small-real-window").exists()
        assert (workspace / "rollback" / "small-real-window.json").is_file()
        assert not (tmp_path / "cold").exists()
    finally:
        lease.release()


def test_fixture_heavy_lease_is_exclusive_heartbeated_and_foreign_process_safe(tmp_path: Path) -> None:
    first = _lease(tmp_path, timeout=0.001)
    second = FixtureHeavyLease(tmp_path / "fixture-heavy.lease", campaign_id="framed-operator-test")
    try:
        with pytest.raises(FixtureLeaseError, match="already held"):
            second.acquire(contention_label="CLEAN")
        first.heartbeat()
        first.assert_clean()
        time.sleep(0.01)
        with pytest.raises(FixtureLeaseError, match="stale"):
            first.assert_clean()
    finally:
        first.release()

    recovered = _lease(tmp_path)
    try:
        assert recovered.receipt()["recovered_unlocked_stale_record"] is True
    finally:
        recovered.release()

    foreign = FixtureHeavyLease(tmp_path / "foreign.lease", campaign_id="framed-operator-test")
    with pytest.raises(FixtureLeaseError, match="foreign-process"):
        foreign.acquire(contention_label="CLEAN", foreign_processes=[{"pid": 123, "label": "other-gpu-owner"}])
    owner = foreign._lease.read_owner()
    assert owner is not None
    assert owner["fixture_only"] is True
    assert owner["contention_label"] == "CONTENDED"


def test_fixture_lease_refuses_clock_regression_sleep_jump_and_record_theft(tmp_path: Path) -> None:
    clock = {"wall": 1_000_000_000, "mono": 2_000_000_000}
    lease = FixtureHeavyLease(
        tmp_path / "clock.lease",
        campaign_id="clock-attacks",
        heartbeat_timeout_seconds=1.0,
        wall_clock_ns=lambda: clock["wall"],
        monotonic_clock_ns=lambda: clock["mono"],
    ).acquire(contention_label="CLEAN")
    try:
        receipt = lease.receipt()
        assert len(receipt["process_identity"]) == 64

        clock["wall"] -= 1
        with pytest.raises(FixtureLeaseError, match="clock regressed"):
            lease.assert_clean()
        clock["wall"] += 1
        lease.heartbeat()

        # A sleep/wake-style wall jump is stale even if a platform monotonic
        # clock pauses while the machine sleeps.
        clock["wall"] += 2_000_000_000
        with pytest.raises(FixtureLeaseError, match="stale"):
            lease.assert_clean()
        clock["wall"] -= 2_000_000_000
        lease.heartbeat()

        # Same-user replacement/tamper is detected against the in-memory
        # lease identity and seal before protected work continues.
        stolen = json.loads(lease.path.read_text(encoding="utf-8"))
        stolen["lease_id"] = "stolen"
        lease.path.write_text(json.dumps(stolen), encoding="utf-8")
        with pytest.raises(FixtureLeaseError, match="replaced or tampered"):
            lease.assert_clean()
    finally:
        lease.release()


def test_fixture_lease_heartbeat_contention_latches_non_clean_until_release(tmp_path: Path) -> None:
    """A discovered foreign process cannot be ignored after a caught error."""
    lease = _lease(tmp_path)
    try:
        with pytest.raises(FixtureLeaseError, match="contention"):
            lease.heartbeat(
                contention_label="CLEAN",
                foreign_processes=[{"pid": 9876, "label": "late-gpu-tenant"}],
            )
        with pytest.raises(FixtureLeaseError, match="not CLEAN"):
            lease.assert_clean()
        record = lease._lease.read_owner()
        assert record is not None
        assert record["contention_label"] == "CONTENDED"
        assert record["foreign_processes"] == [{"pid": 9876, "label": "late-gpu-tenant"}]
    finally:
        lease.release()


def test_fixture_lease_binds_locked_inode_and_refuses_fresh_unlocked_recovery(tmp_path: Path) -> None:
    clock = {"wall": 1_000_000_000, "mono": 2_000_000_000}
    path = tmp_path / "inode.lease"
    lease = FixtureHeavyLease(
        path,
        campaign_id="inode-attacks",
        heartbeat_timeout_seconds=1.0,
        wall_clock_ns=lambda: clock["wall"],
        monotonic_clock_ns=lambda: clock["mono"],
    ).acquire(contention_label="CLEAN")
    try:
        # Replace the pathname with byte-for-byte identical content. Content
        # seals alone cannot detect that flock still protects the old inode.
        replacement = tmp_path / "replacement.lease"
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
        with pytest.raises(FixtureLeaseError, match="replaced or tampered"):
            lease.assert_clean()
    finally:
        lease.release()

    fresh = FixtureHeavyLease(
        path,
        campaign_id="inode-attacks",
        heartbeat_timeout_seconds=1.0,
        wall_clock_ns=lambda: clock["wall"],
        monotonic_clock_ns=lambda: clock["mono"],
    )
    with pytest.raises(FixtureLeaseError, match="still fresh"):
        fresh.acquire(contention_label="CLEAN")

    clock["wall"] += 1_000_000_001
    recovered = fresh.acquire(contention_label="CLEAN")
    try:
        assert recovered.receipt()["recovered_unlocked_stale_record"] is True
    finally:
        recovered.release()
