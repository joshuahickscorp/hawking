from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lab.operators.glm52_common import canonical, seal
from lab.operators.glm52_range_stream_executor import (
    RangeExecutorError,
    _call_with_deadline,
    RECEIPT_SCHEMA,
    _RangeDeadlineChunks,
    rebuild_schedule_ranges,
    range_source_identity_sha256,
    assert_unified_floor,
    validate_receipt_directory,
    validate_window_receipt,
    write_frame,
    write_frame_before_deadline,
)
from lab.operators.glm52_restream_contract import build_contract


_RECEIPT_PRIVATE = Ed25519PrivateKey.generate()
_RECEIPT_PUBLIC = _RECEIPT_PRIVATE.public_key().public_bytes_raw()
_LAUNCH_NONCE_SHA256 = "c" * 64
_LEASE_IDENTITY_SHA256 = "d" * 64
_OPERATOR_SHA256 = "e" * 64


@pytest.fixture(scope="module")
def rebuilt():
    schedule, policy = build_contract(
        manifest_path="evidence/glm52/GLM52_OFFICIAL_MANIFEST.json",
        graph_path="evidence/glm52/GLM52_SHARD_DEPENDENCY_GRAPH.json",
    )
    return schedule, policy, rebuild_schedule_ranges(schedule)


def _receipt(schedule, policy, window):
    body = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS_SEALED_REMOTE_HASHED_EVICTED",
        "schedule_seal_sha256": schedule["seal_sha256"],
        "policy_seal_sha256": policy["seal_sha256"],
        "window_id": window["window_id"],
        "execution_order": window["execution_order"],
        "range_count": window["range_count"],
        "ordered_range_ids_sha256": window["ordered_range_ids_sha256"],
        "payload_bytes": sum(row["payload_bytes"] for row in window["ranges"]),
        "source_range_identity_sha256": range_source_identity_sha256(window),
        "predecessor_window_receipt_seal_sha256": None,
        "launch_nonce_sha256": _LAUNCH_NONCE_SHA256,
        "lease_identity_sha256": _LEASE_IDENTITY_SHA256,
        "operator_executable_sha256": _OPERATOR_SHA256,
        "payload_hash_chain_sha256": "b" * 64,
        "artifact_handoff": {
            "cold_upload_complete": True, "remote_sha256_verified": True, "remote_sha256": "a" * 64,
        },
        "eviction": {
            "source_ranges_retained_zero": True, "local_artifact_bytes_retained_zero": True,
            "temp_bytes_retained_zero": True, "exact_cache_items_purged": True,
            "exact_trash_items_purged": True, "free_byte_recovery_measured": True,
            "free_bytes_before_eviction": 100, "free_bytes_after_eviction": 120, "recovered_bytes": 20,
        },
    }
    return seal({
        **body,
        "attestation": {
            "algorithm": "Ed25519",
            "public_key_sha256": __import__("hashlib").sha256(_RECEIPT_PUBLIC).hexdigest(),
            "signature_ed25519_hex": _RECEIPT_PRIVATE.sign(canonical(body)).hex(),
        },
    })


def _validate(receipt, schedule, policy, window):
    return validate_window_receipt(
        receipt,
        schedule=schedule,
        policy=policy,
        window=window,
        operator_receipt_public_key_bytes=_RECEIPT_PUBLIC,
        predecessor_receipt_seal_sha256=None,
        launch_nonce_sha256=_LAUNCH_NONCE_SHA256,
        lease_identity_sha256=_LEASE_IDENTITY_SHA256,
        operator_executable_sha256=_OPERATOR_SHA256,
    )


def test_rebuild_binds_every_real_range_to_official_xet_identity(rebuilt) -> None:
    schedule, _policy, windows = rebuilt
    assert len(windows) == 81
    assert sum(len(window["ranges"]) for window in windows) == 59_585
    assert all(len(row["xet_hash"]) == 64 for window in windows for row in window["ranges"])
    assert [window["window_id"] for window in windows] == [window["window_id"] for window in schedule["windows"]]
    assert all(len(range_source_identity_sha256(window)) == 64 for window in windows)


def test_binary_frame_is_length_delimited_and_exact() -> None:
    target = io.BytesIO()
    assert write_frame(target, {"kind": "RANGE"}, [b"abc", b"de"]) == 5
    body = target.getvalue()
    header_bytes = int.from_bytes(body[:8], "big")
    assert body[8 + header_bytes:] == b"abcde"


def test_receipt_requires_remote_hash_eviction_and_exact_recovery(rebuilt) -> None:
    schedule, policy, windows = rebuilt
    receipt = _receipt(schedule, policy, windows[0])
    assert _validate(receipt, schedule, policy, windows[0]) == receipt
    broken = dict(receipt)
    broken.pop("seal_sha256")
    broken["eviction"] = {**broken["eviction"], "recovered_bytes": 19}
    with pytest.raises(RangeExecutorError, match="recovery arithmetic"):
        _validate(seal(broken), schedule, policy, windows[0])

    bad_chain = dict(receipt)
    bad_chain.pop("seal_sha256")
    bad_chain["payload_hash_chain_sha256"] = "not-a-hash"
    with pytest.raises(RangeExecutorError, match="payload hash chain"):
        _validate(seal(bad_chain), schedule, policy, windows[0])

    forged = dict(receipt)
    forged.pop("seal_sha256")
    forged["payload_hash_chain_sha256"] = "f" * 64
    with pytest.raises(RangeExecutorError, match="attestation verification"):
        _validate(seal(forged), schedule, policy, windows[0])


def test_mid_window_floor_guard_refuses_the_next_write_exactly() -> None:
    assert assert_unified_floor(
        Path("."), protected_floor_bytes=100, additional_bytes=20,
        observed_free_bytes=120, stage="fixture range",
    )["free_bytes"] == 120
    with pytest.raises(RangeExecutorError, match="crossed at fixture range"):
        assert_unified_floor(
            Path("."), protected_floor_bytes=100, additional_bytes=21,
            observed_free_bytes=120, stage="fixture range",
        )


def test_resume_receipt_directory_rejects_duplicate_or_symlinked_receipts(tmp_path: Path) -> None:
    windows = [
        {"execution_order": 0, "window_id": "W0"},
        {"execution_order": 1, "window_id": "W1"},
    ]
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    validate_receipt_directory(receipts, windows)

    # A second terminal-looking record cannot be silently ignored on resume.
    (receipts / "000_W0.copy.json").write_text("{}")
    with pytest.raises(RangeExecutorError, match="unexpected entry"):
        validate_receipt_directory(receipts, windows)
    (receipts / "000_W0.copy.json").unlink()

    target = tmp_path / "outside-receipt.json"
    target.write_text("{}")
    (receipts / "000_W0.json").symlink_to(target)
    with pytest.raises(RangeExecutorError, match="not a regular owned file"):
        validate_receipt_directory(receipts, windows)


def test_xet_range_deadline_cancels_a_stuck_iterator() -> None:
    class Stuck:
        def __init__(self) -> None:
            self.cancelled = threading.Event()

        def __iter__(self):
            self.cancelled.wait()
            return
            yield b"unreachable"

        def cancel(self) -> None:
            self.cancelled.set()

    source = Stuck()
    with pytest.raises(RangeExecutorError, match="range deadline expired: fixture-range"):
        list(_RangeDeadlineChunks(source, range_id="fixture-range", timeout_seconds=0.02))
    assert source.cancelled.wait(timeout=0.2)


def test_xet_range_setup_deadline_aborts_the_campaign_endpoint() -> None:
    entered = threading.Event()
    aborted = threading.Event()

    def stuck_setup() -> object:
        entered.set()
        aborted.wait()
        return object()

    with pytest.raises(RangeExecutorError, match="setup deadline expired: fixture-setup"):
        _call_with_deadline(
            stuck_setup,
            range_id="fixture-setup",
            timeout_seconds=0.02,
            on_timeout=aborted.set,
        )
    assert entered.is_set()
    assert aborted.wait(timeout=0.2)


def test_operator_pipe_backpressure_is_bounded_by_the_range_deadline() -> None:
    read_fd, write_fd = os.pipe()
    handle = os.fdopen(write_fd, "wb", buffering=0)
    try:
        deadline = time.monotonic() + 0.02
        with pytest.raises(RangeExecutorError, match="pipe deadline expired: fixture-pipe"):
            write_frame_before_deadline(
                handle,
                {"kind": "RANGE", "payload_bytes": 2 * 1024 * 1024},
                [b"x" * (2 * 1024 * 1024)],
                deadline=deadline,
                range_id="fixture-pipe",
            )
    finally:
        handle.close()
        os.close(read_fd)
