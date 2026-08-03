"""Fixture-only physical framed-window operator for the GLM range protocol.

This module intentionally stops before a parent restream: it accepts bounded
already-fetched frames, verifies every frame body, makes a deterministic
fixture pack, copies that pack through a cold-store interface, seals the
terminal receipt, and removes every local source/temporary/artifact byte.  It
is useful for exercising the exact lifecycle without fetching model weights,
retaining a source shard, or minting a production capability result.

The real range executor remains responsible for official Xet acquisition and
requires an owner-approved external operator.  This module refuses an enabled
parent-restream environment and emits ``FIXTURE_ONLY`` receipts only.
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Protocol

from lab.lease import FixtureHeavyLease, FixtureLeaseError
from lab.operators.glm52_common import atomic_bytes, atomic_json, canonical, read_sealed_json, seal, sha256_file
from lab.operators.glm52_range_stream_executor import FramedMessage, RangeExecutorError, read_frame
from ramanujan.restream_guard import ACCOUNTING_COMPONENTS, ALIGNMENT_BYTES


FIXTURE_PROTOCOL = "hawking.glm52.window_stream.framed.fixture.v2"
FIXTURE_RECEIPT_SCHEMA = "hawking.glm52.fixture_framed_window_receipt.v1"
ROLLBACK_RECEIPT_SCHEMA = "hawking.glm52.fixture_framed_window_rollback.v1"
MAX_INCREMENTAL_BYTES = 90_000_000_000
DEFAULT_MAX_FIXTURE_RANGE_BYTES = 16 * 1024 * 1024


class FramedWindowOperatorError(RuntimeError):
    """A physical fixture frame, lifecycle, or storage gate failed closed."""


class ColdStorage(Protocol):
    """Minimal cold-store boundary used by the fixture operator."""

    def put_verified(self, *, key: str, source: Path, sha256: str) -> dict[str, Any]: ...

    def delete(self, *, key: str) -> None: ...


@dataclass(frozen=True)
class LocalFixtureColdStorage:
    """A local test double for remote cold storage, never a production backend."""

    root: Path

    def _target(self, key: str) -> Path:
        if not key or Path(key).name != key:
            raise FramedWindowOperatorError("cold-store key must be a basename")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise FramedWindowOperatorError("fixture cold-store root may not be a symlink")
        return self.root / key

    def put_verified(self, *, key: str, source: Path, sha256: str) -> dict[str, Any]:
        target = self._target(key)
        if not source.is_file() or source.is_symlink():
            raise FramedWindowOperatorError("fixture cold-store source is not a regular local artifact")
        if sha256_file(source) != sha256:
            raise FramedWindowOperatorError("fixture artifact changed before cold handoff")
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise FramedWindowOperatorError("fixture cold-store target is not a regular file")
            if sha256_file(target) != sha256:
                raise FramedWindowOperatorError("fixture cold-store key is already bound to different bytes")
            return {
                "kind": "LOCAL_FIXTURE_COLD_STORE",
                "key": key,
                "remote_sha256": sha256,
                "remote_sha256_verified": True,
                "idempotent_existing_object": True,
                "created_new_object": False,
            }
        # Stream to a separate file before atomic publication. A medium model
        # artifact must never be duplicated into one unbounded Python bytes
        # object merely to reach cold storage.
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=self.root)
        temp = Path(temp_name)
        try:
            with os.fdopen(descriptor, "wb") as destination, source.open("rb") as origin:
                shutil.copyfileobj(origin, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            if sha256_file(temp) != sha256:
                raise FramedWindowOperatorError("fixture streamed cold handoff hash mismatch")
            os.replace(temp, target)
            directory_fd = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        remote_sha256 = sha256_file(target)
        if remote_sha256 != sha256:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
            raise FramedWindowOperatorError("fixture cold-store hash mismatch")
        return {
            "kind": "LOCAL_FIXTURE_COLD_STORE",
            "key": key,
            "remote_sha256": remote_sha256,
            "remote_sha256_verified": True,
            "created_new_object": True,
        }

    def delete(self, *, key: str) -> None:
        target = self._target(key)
        try:
            target.unlink()
        except FileNotFoundError:
            return


def _sha256_ids(values: list[str]) -> str:
    return hashlib.sha256(canonical(values)).hexdigest()


def _safe_window_id(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or Path(value).name != value:
        raise FramedWindowOperatorError("window_id must be a non-empty basename")
    return value


def _require_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise FramedWindowOperatorError(f"{label} fields differ; missing={missing!r} extra={extra!r}")


def _nonnegative(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FramedWindowOperatorError(f"{label} must be a non-negative integer")
    return value


def _round_up(value: int) -> int:
    return ((value + ALIGNMENT_BYTES - 1) // ALIGNMENT_BYTES) * ALIGNMENT_BYTES


def _allocated_tree_bytes(root: Path) -> int:
    """Count allocated regular-file bytes without following symlinks."""
    if not root.exists():
        return 0
    if root.is_symlink():
        raise FramedWindowOperatorError("fixture local path may not be a symlink")
    if root.is_file():
        return root.stat().st_blocks * 512
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise FramedWindowOperatorError("fixture staging tree may not contain symlinks")
        if path.is_file():
            total += path.stat().st_blocks * 512
    return total


@dataclass(frozen=True)
class _WindowHeader:
    window_id: str
    execution_order: int
    range_count: int
    ordered_range_ids: tuple[str, ...]
    schedule_seal_sha256: str
    policy_seal_sha256: str
    incremental_bytes: int
    protected_floor_bytes: int
    accounting: dict[str, int]


class FixtureFramedWindowOperator:
    """Consume a bounded framed window under a clean fixture-only lease."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        cold_storage: ColdStorage,
        lease: FixtureHeavyLease,
        max_incremental_bytes: int = MAX_INCREMENTAL_BYTES,
        max_fixture_range_bytes: int = DEFAULT_MAX_FIXTURE_RANGE_BYTES,
        failure_stage: str | None = None,
        disk_free_bytes: Callable[[Path], int] | None = None,
    ) -> None:
        if os.environ.get("HAWKING_PARENT_RESTREAM_AUTHORIZED") == "YES":
            raise FramedWindowOperatorError("fixture-only framed operator refuses parent-restream authorization")
        if isinstance(max_incremental_bytes, bool) or not isinstance(max_incremental_bytes, int) or not 0 < max_incremental_bytes <= MAX_INCREMENTAL_BYTES:
            raise FramedWindowOperatorError("fixture operator max_incremental_bytes must be in (0, 90000000000]")
        if isinstance(max_fixture_range_bytes, bool) or not isinstance(max_fixture_range_bytes, int) or max_fixture_range_bytes <= 0:
            raise FramedWindowOperatorError("fixture range byte limit must be positive")
        if failure_stage not in {None, "after_range", "after_pack", "after_handoff"}:
            raise FramedWindowOperatorError("unknown fixture failure injection stage")
        self.workspace_root = workspace_root
        self.cold_storage = cold_storage
        self.lease = lease
        self.max_incremental_bytes = max_incremental_bytes
        self.max_fixture_range_bytes = max_fixture_range_bytes
        self.failure_stage = failure_stage
        self._disk_free_bytes = disk_free_bytes or (lambda path: shutil.disk_usage(path).free)

    def _root(self) -> Path:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        if self.workspace_root.is_symlink():
            raise FramedWindowOperatorError("fixture workspace root may not be a symlink")
        return self.workspace_root.resolve()

    def _receipt_path(self, window_id: str) -> Path:
        return self._root() / "receipts" / f"{window_id}.json"

    def _rollback_path(self, window_id: str) -> Path:
        return self._root() / "rollback" / f"{window_id}.json"

    def _staging_path(self, window_id: str) -> Path:
        return self._root() / ".staging" / window_id

    def _free_bytes(self) -> int:
        value = self._disk_free_bytes(self._root())
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FramedWindowOperatorError("fixture disk-free sampler returned an invalid byte count")
        return value

    def _assert_floor(self, window: _WindowHeader, *, stage: str) -> int:
        free_bytes = self._free_bytes()
        if free_bytes < window.protected_floor_bytes:
            raise FramedWindowOperatorError(
                f"fixture unified protected floor crossed at {stage}: "
                f"free={free_bytes} floor={window.protected_floor_bytes}"
            )
        return free_bytes

    @staticmethod
    def _require_regular_single_link(path: Path, *, label: str) -> None:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise FramedWindowOperatorError(f"{label} cannot be inspected: {exc}") from exc
        if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1:
            raise FramedWindowOperatorError(f"{label} must be a non-symlink regular file with one link")

    def _read_window_header(self, frame: FramedMessage) -> _WindowHeader:
        header = frame.header
        _require_keys(
            header,
            {
                "kind", "protocol", "fixture_only", "window_id", "execution_order", "range_count",
                "ordered_range_ids", "schedule_seal_sha256", "policy_seal_sha256", "incremental_bytes",
                "protected_floor_bytes", "incremental_accounting", "payload_bytes",
            },
            label="WINDOW frame",
        )
        if header["kind"] != "WINDOW" or header["protocol"] != FIXTURE_PROTOCOL or header["fixture_only"] is not True:
            raise FramedWindowOperatorError("WINDOW frame is not the fixture-only framed protocol")
        if frame.payload:
            raise FramedWindowOperatorError("WINDOW frame may not carry a payload")
        window_id = _safe_window_id(header["window_id"])
        execution_order = _nonnegative(header["execution_order"], label="execution_order")
        range_count = _nonnegative(header["range_count"], label="range_count")
        raw_ids = header["ordered_range_ids"]
        if not isinstance(raw_ids, list) or len(raw_ids) != range_count or not raw_ids:
            raise FramedWindowOperatorError("WINDOW frame range ids do not match range_count")
        if any(not isinstance(value, str) or not value for value in raw_ids) or len(set(raw_ids)) != len(raw_ids):
            raise FramedWindowOperatorError("WINDOW frame range ids must be unique non-empty strings")
        schedule = header["schedule_seal_sha256"]
        policy = header["policy_seal_sha256"]
        if any(not isinstance(value, str) or len(value) != 64 for value in (schedule, policy)):
            raise FramedWindowOperatorError("WINDOW frame requires exact schedule and policy seals")
        incremental = _nonnegative(header["incremental_bytes"], label="incremental_bytes")
        floor = _nonnegative(header["protected_floor_bytes"], label="protected_floor_bytes")
        raw_accounting = header["incremental_accounting"]
        if not isinstance(raw_accounting, Mapping):
            raise FramedWindowOperatorError("WINDOW frame requires complete incremental_accounting")
        expected_accounting = set(ACCOUNTING_COMPONENTS) | {"resident_incremental_bytes"}
        if set(raw_accounting) != expected_accounting:
            raise FramedWindowOperatorError("WINDOW frame incremental_accounting has missing or extra components")
        accounting = {
            key: _nonnegative(raw_accounting[key], label=f"incremental_accounting.{key}")
            for key in expected_accounting
        }
        if any(accounting[key] % ALIGNMENT_BYTES for key in ACCOUNTING_COMPONENTS):
            raise FramedWindowOperatorError("WINDOW frame accounting components must be 64-KiB rounded")
        component_sum = sum(accounting[key] for key in ACCOUNTING_COMPONENTS)
        if accounting["resident_incremental_bytes"] != component_sum or incremental != component_sum:
            raise FramedWindowOperatorError("WINDOW frame incremental bytes do not equal the complete component sum")
        if incremental > self.max_incremental_bytes or incremental > MAX_INCREMENTAL_BYTES:
            raise FramedWindowOperatorError("WINDOW frame exceeds the <=90-GB incremental envelope")
        free_bytes = self._free_bytes()
        if free_bytes - floor < incremental:
            raise FramedWindowOperatorError("fixture unified-floor admission failed before any frame body")
        return _WindowHeader(
            window_id=window_id,
            execution_order=execution_order,
            range_count=range_count,
            ordered_range_ids=tuple(raw_ids),
            schedule_seal_sha256=schedule,
            policy_seal_sha256=policy,
            incremental_bytes=incremental,
            protected_floor_bytes=floor,
            accounting=accounting,
        )

    def _read_range(
        self,
        frame: FramedMessage,
        *,
        expected_id: str,
        staging: Path,
        window: _WindowHeader,
        charged_source_bytes: int,
        prior_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        header = frame.header
        _require_keys(
            header,
            {"kind", "range_id", "shard", "start", "end", "payload_bytes", "payload_sha256"},
            label="RANGE frame",
        )
        if header["kind"] != "RANGE" or header["range_id"] != expected_id:
            raise FramedWindowOperatorError("RANGE frame is out of the declared exact order")
        shard = header["shard"]
        if not isinstance(shard, str) or not shard or Path(shard).name != shard:
            raise FramedWindowOperatorError("RANGE frame shard must be a basename")
        start = _nonnegative(header["start"], label="range start")
        end = _nonnegative(header["end"], label="range end")
        if end <= start:
            raise FramedWindowOperatorError("RANGE frame interval is empty or backwards")
        for prior in prior_rows:
            if prior["shard"] == shard and start < prior["end"]:
                raise FramedWindowOperatorError(
                    "RANGE frames for one shard must be strictly ordered and non-overlapping"
                )
        payload_bytes = _nonnegative(header["payload_bytes"], label="range payload_bytes")
        if payload_bytes != end - start or payload_bytes != len(frame.payload):
            raise FramedWindowOperatorError("RANGE frame payload length differs from its exact half-open interval")
        if payload_bytes > self.max_fixture_range_bytes:
            raise FramedWindowOperatorError("RANGE frame exceeds fixture-only body bound")
        expected_sha = header["payload_sha256"]
        observed_sha = hashlib.sha256(frame.payload).hexdigest()
        if not isinstance(expected_sha, str) or len(expected_sha) != 64 or observed_sha != expected_sha:
            raise FramedWindowOperatorError("RANGE frame payload hash does not verify")
        source_charge = _round_up(payload_bytes)
        prospective_charge = charged_source_bytes + source_charge
        if prospective_charge > window.accounting["source_range_rounded_bytes"]:
            raise FramedWindowOperatorError("cumulative 64-KiB source ranges exceed the declared window source component")
        if prospective_charge > window.incremental_bytes:
            raise FramedWindowOperatorError("cumulative source ranges exceed the declared incremental window envelope")
        self._assert_floor(window, stage=f"before_range_{expected_id}")
        source_path = staging / f"range-{len(list(staging.glob('range-*.bin'))):06d}.bin"
        atomic_bytes(source_path, frame.payload)
        self._assert_floor(window, stage=f"after_range_write_{expected_id}")
        if sha256_file(source_path) != expected_sha:
            raise FramedWindowOperatorError("staged RANGE body hash does not verify")
        local_staging_bytes = _allocated_tree_bytes(staging)
        if local_staging_bytes > window.accounting["source_range_rounded_bytes"]:
            raise FramedWindowOperatorError("actual staged source bytes exceed the declared source component")
        if prospective_charge + local_staging_bytes > window.incremental_bytes:
            raise FramedWindowOperatorError("actual staged source bytes exceed the declared incremental envelope")
        row = {
            "range_id": expected_id,
            "shard": shard,
            "start": start,
            "end": end,
            "payload_bytes": payload_bytes,
            "payload_sha256": observed_sha,
            "charged_source_bytes": source_charge,
            "local_staging_bytes": local_staging_bytes,
        }
        # A fixture pack processes the range then immediately removes its only
        # local source copy.  No source body survives the end of this method.
        source_path.unlink()
        if source_path.exists():
            raise FramedWindowOperatorError("fixture source-range eviction failed")
        return row

    def _rollback(
        self,
        *,
        window_id: str,
        stage: str,
        detail: str,
        staging: Path,
        remote_key: str | None,
        terminal_receipt_path: Path | None = None,
    ) -> None:
        cleanup_errors: list[str] = []
        remote_cleanup_verified = remote_key is None
        if remote_key is not None:
            try:
                self.cold_storage.delete(key=remote_key)
            except Exception as exc:  # noqa: BLE001 - preserve original failure but record cleanup issue.
                cleanup_errors.append(f"cold rollback failed: {exc}")
            else:
                remote_cleanup_verified = True
        if staging.exists():
            try:
                shutil.rmtree(staging)
            except Exception as exc:  # noqa: BLE001
                cleanup_errors.append(f"local staging rollback failed: {exc}")
        if terminal_receipt_path is not None:
            try:
                terminal_receipt_path.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:  # noqa: BLE001
                cleanup_errors.append(f"terminal receipt rollback failed: {exc}")
        local_cleanup_verified = not staging.exists() and (
            terminal_receipt_path is None or not terminal_receipt_path.exists()
        )
        cleanup_complete = local_cleanup_verified and remote_cleanup_verified and not cleanup_errors
        if cleanup_errors:
            detail = f"{detail}; {'; '.join(cleanup_errors)}"
        atomic_json(
            self._rollback_path(window_id),
            seal({
                "schema": ROLLBACK_RECEIPT_SCHEMA,
                "status": (
                    "FIXTURE_ONLY_ROLLED_BACK_ALL_ATTEMPT_BYTES_EVICTED"
                    if cleanup_complete
                    else "FIXTURE_ONLY_ROLLBACK_INCOMPLETE_FAIL_CLOSED"
                ),
                "fixture_only": True,
                "production_authority": False,
                "window_id": window_id,
                "stage": stage,
                "detail": detail,
                "cleanup_complete": cleanup_complete,
                "local_cleanup_verified": local_cleanup_verified,
                "remote_cleanup_verified": remote_cleanup_verified,
                "cleanup_errors": cleanup_errors,
                "staging_absent_after_rollback": not staging.exists(),
            }),
        )

    def run(self, stream: BinaryIO) -> dict[str, Any]:
        """Execute one small-real/synthetic frame sequence and seal its cleanup.

        A pre-existing valid terminal receipt is the atomic resume point.  Any
        interrupted staging tree is removed before the complete window is
        replayed, so a resume never trusts or retains a partial source range.
        """
        try:
            self.lease.assert_clean()
        except FixtureLeaseError as exc:
            raise FramedWindowOperatorError(f"fixture heavy lease is unavailable: {exc}") from exc
        try:
            first = read_frame(stream, max_payload_bytes=self.max_fixture_range_bytes)
        except RangeExecutorError as exc:
            raise FramedWindowOperatorError(str(exc)) from exc
        window = self._read_window_header(first)
        receipt_path = self._receipt_path(window.window_id)
        if receipt_path.exists() or receipt_path.is_symlink():
            self._require_regular_single_link(receipt_path, label="atomic resume receipt")
            try:
                existing = read_sealed_json(receipt_path)
            except Exception as exc:  # noqa: BLE001
                raise FramedWindowOperatorError("atomic resume found an invalid terminal receipt") from exc
            required = {
                "schema": FIXTURE_RECEIPT_SCHEMA,
                "status": "FIXTURE_ONLY_PASS_REMOTE_HASHED_EVICTED",
                "fixture_only": True,
                "production_authority": False,
                "protocol": FIXTURE_PROTOCOL,
                "window_id": window.window_id,
                "execution_order": window.execution_order,
                "schedule_seal_sha256": window.schedule_seal_sha256,
                "policy_seal_sha256": window.policy_seal_sha256,
                "incremental_bytes": window.incremental_bytes,
                "protected_floor_bytes": window.protected_floor_bytes,
                "incremental_accounting": window.accounting,
                "range_count": window.range_count,
                "ordered_range_ids_sha256": _sha256_ids(list(window.ordered_range_ids)),
            }
            if any(existing.get(key) != value for key, value in required.items()):
                raise FramedWindowOperatorError("atomic resume receipt does not bind this exact fixture window")
            return existing

        staging = self._staging_path(window.window_id)
        recovered_partial_staging = staging.exists()
        if recovered_partial_staging:
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=False)
        rows: list[dict[str, Any]] = []
        remote_key: str | None = None
        charged_source_bytes = 0
        peak_local_staging_bytes = 0
        try:
            for expected_id in window.ordered_range_ids:
                self.lease.heartbeat()
                self._assert_floor(window, stage=f"before_read_{expected_id}")
                try:
                    frame = read_frame(stream, max_payload_bytes=self.max_fixture_range_bytes)
                except RangeExecutorError as exc:
                    raise FramedWindowOperatorError(str(exc)) from exc
                row = self._read_range(
                    frame,
                    expected_id=expected_id,
                    staging=staging,
                    window=window,
                    charged_source_bytes=charged_source_bytes,
                    prior_rows=rows,
                )
                rows.append(row)
                charged_source_bytes += row["charged_source_bytes"]
                # _read_range records this before the immediate source eviction.
                # Looking at the staging tree here would always (and wrongly)
                # observe zero.
                peak_local_staging_bytes = max(peak_local_staging_bytes, row["local_staging_bytes"])
                if self.failure_stage == "after_range":
                    raise FramedWindowOperatorError("injected fixture failure after verified source range")
            try:
                end = read_frame(stream, max_payload_bytes=self.max_fixture_range_bytes)
            except RangeExecutorError as exc:
                raise FramedWindowOperatorError(str(exc)) from exc
            _require_keys(end.header, {"kind", "window_id", "payload_bytes"}, label="END_WINDOW frame")
            if end.header["kind"] != "END_WINDOW" or end.header["window_id"] != window.window_id or end.payload:
                raise FramedWindowOperatorError("END_WINDOW does not close the active fixture window")
            trailing = stream.read(1)
            if trailing != b"":
                raise FramedWindowOperatorError("fixture frame stream contains trailing bytes after END_WINDOW")
            if len(rows) != window.range_count or _sha256_ids([row["range_id"] for row in rows]) != _sha256_ids(list(window.ordered_range_ids)):
                raise FramedWindowOperatorError("fixture frame sequence does not cover the declared exact range order")
            self.lease.heartbeat()
            self._assert_floor(window, stage="before_pack")
            artifact = staging / "fixture-one-pass-pack.gravity"
            artifact_body = canonical({
                "schema": "hawking.glm52.fixture_one_pass_pack.v1",
                "fixture_only": True,
                "production_authority": False,
                "window_id": window.window_id,
                "range_rows": rows,
                "total_payload_bytes": sum(row["payload_bytes"] for row in rows),
            })
            atomic_bytes(artifact, artifact_body)
            self._assert_floor(window, stage="after_pack")
            artifact_sha256 = sha256_file(artifact)
            artifact_allocated_bytes = _allocated_tree_bytes(staging)
            artifact_charge = _round_up(artifact.stat().st_size)
            if artifact_charge > window.accounting["retained_artifact_bytes"]:
                raise FramedWindowOperatorError("actual one-pass artifact exceeds the declared retained-artifact component")
            if artifact_allocated_bytes > window.accounting["retained_artifact_bytes"]:
                raise FramedWindowOperatorError("actual allocated one-pass artifact exceeds the declared retained-artifact component")
            if charged_source_bytes + artifact_charge + window.accounting["metadata_bytes"] > window.incremental_bytes:
                raise FramedWindowOperatorError("cumulative source plus artifact and metadata exceed the declared window envelope")
            peak_local_staging_bytes = max(peak_local_staging_bytes, artifact_allocated_bytes)
            if peak_local_staging_bytes > window.incremental_bytes:
                raise FramedWindowOperatorError("actual local staging peak exceeds the declared window envelope")
            if self.failure_stage == "after_pack":
                raise FramedWindowOperatorError("injected fixture failure after one-pass pack")
            proposed_remote_key = f"{window.window_id}-{artifact_sha256}.fixture"
            self._assert_floor(window, stage="before_cold_handoff")
            handoff = self.cold_storage.put_verified(key=proposed_remote_key, source=artifact, sha256=artifact_sha256)
            if handoff.get("created_new_object") not in {True, False}:
                raise FramedWindowOperatorError("cold storage did not report object creation ownership")
            # Rollback may delete only the object minted by this exact attempt.
            # An idempotent pre-existing object is never this attempt's property.
            remote_key = proposed_remote_key if handoff["created_new_object"] is True else None
            if (
                handoff.get("remote_sha256_verified") is not True
                or handoff.get("remote_sha256") != artifact_sha256
            ):
                raise FramedWindowOperatorError(
                    "cold storage did not attest the exact remote artifact hash"
                )
            self._assert_floor(window, stage="after_cold_handoff")
            if self.failure_stage == "after_handoff":
                raise FramedWindowOperatorError("injected fixture failure after cold handoff")
            free_before_eviction = self._assert_floor(window, stage="before_local_eviction")
            local_evicted_bytes = sum(row["local_staging_bytes"] for row in rows) + artifact_allocated_bytes
            artifact.unlink()
            shutil.rmtree(staging)
            if staging.exists():
                raise FramedWindowOperatorError("fixture staging eviction failed")
            free_after_eviction = self._assert_floor(window, stage="after_local_eviction")
            receipt = seal({
                "schema": FIXTURE_RECEIPT_SCHEMA,
                "status": "FIXTURE_ONLY_PASS_REMOTE_HASHED_EVICTED",
                "fixture_only": True,
                "production_authority": False,
                "protocol": FIXTURE_PROTOCOL,
                "window_id": window.window_id,
                "execution_order": window.execution_order,
                "schedule_seal_sha256": window.schedule_seal_sha256,
                "policy_seal_sha256": window.policy_seal_sha256,
                "incremental_bytes": window.incremental_bytes,
                "protected_floor_bytes": window.protected_floor_bytes,
                "incremental_accounting": window.accounting,
                "range_count": len(rows),
                "ordered_range_ids_sha256": _sha256_ids([row["range_id"] for row in rows]),
                "payload_bytes": sum(row["payload_bytes"] for row in rows),
                "range_rows": rows,
                "artifact_handoff": handoff,
                "eviction": {
                    "source_ranges_retained_zero": True,
                    "local_artifact_bytes_retained_zero": True,
                    "temp_bytes_retained_zero": True,
                    "free_byte_recovery_measured": True,
                    "free_bytes_before_eviction": free_before_eviction,
                    "free_bytes_after_eviction": free_after_eviction,
                    "free_bytes_delta": free_after_eviction - free_before_eviction,
                    "local_bytes_evicted": local_evicted_bytes,
                    "peak_local_staging_bytes": peak_local_staging_bytes,
                    "charged_source_bytes": charged_source_bytes,
                    "charged_artifact_bytes": artifact_charge,
                    "actual_allocated_artifact_bytes": artifact_allocated_bytes,
                },
                "cache_semantics": {
                    "cache_created": False,
                    "cache_not_created_or_retained": True,
                    "purge_claim": "NONE — fixture operator creates no cache",
                },
                "trash_semantics": {
                    "trash_created": False,
                    "trash_not_created_or_retained": True,
                    "purge_claim": "NONE — fixture operator never moves bytes to Trash",
                },
                "resume": {
                    "resume_unit": "sealed_fixture_window_terminal_receipt",
                    "recovered_partial_staging": recovered_partial_staging,
                },
                "lease": self.lease.receipt(),
            })
            self._assert_floor(window, stage="before_terminal_receipt")
            atomic_json(receipt_path, receipt)
            self._assert_floor(window, stage="after_terminal_receipt")
            receipt_allocated_bytes = _allocated_tree_bytes(receipt_path)
            if receipt_allocated_bytes > window.accounting["metadata_bytes"]:
                raise FramedWindowOperatorError("actual terminal receipt exceeds the declared metadata component")
            if receipt_allocated_bytes > window.incremental_bytes:
                raise FramedWindowOperatorError("actual terminal receipt exceeds the declared window envelope")
            return receipt
        except BaseException as exc:
            self._rollback(
                window_id=window.window_id,
                stage=self.failure_stage or "frame_or_lifecycle_gate",
                detail=str(exc),
                staging=staging,
                remote_key=remote_key,
                terminal_receipt_path=receipt_path,
            )
            raise


def frame_stream(messages: list[tuple[Mapping[str, Any], bytes]]) -> io.BytesIO:
    """Build a bounded fixture byte stream; used only by focused dry-run tests."""
    from lab.operators.glm52_range_stream_executor import write_frame

    output = io.BytesIO()
    for header, payload in messages:
        if header.get("kind") == "RANGE" and header.get("payload_bytes") != len(payload):
            raise FramedWindowOperatorError("fixture RANGE header length must match supplied bytes")
        write_frame(output, header, [payload] if payload else ())
    output.seek(0)
    return output
