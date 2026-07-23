#!/usr/bin/env python3.12
"""Two-phase exact cleanup for nonresumable HF 1.24 Kimi temp blobs.

``audit`` writes a sealed, confirmation-required inventory while holding the
download supervisor's exact lease.  ``execute`` accepts only that inventory's
seal, re-verifies every identity under the same lease, and unlinks only exact
``<manifest-sha256>.<8-lowerhex>.incomplete`` leaves by blob-directory fd.

There is no glob, recursive removal, final-blob removal, Xet removal, network
operation, or downloader launch in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from . import kimi_k26_download_supervisor as supervisor
    from . import kimi_k26_release_cycle as phase1
else:
    import kimi_k26_download_supervisor as supervisor  # type: ignore[no-redef]
    import kimi_k26_release_cycle as phase1  # type: ignore[no-redef]


AUDIT_SCHEMA = "hawking.kimi_k26.stale_download_cleanup.audit.v1"
RECEIPT_SCHEMA = supervisor._CLEANUP_RECEIPT_SCHEMA  # noqa: SLF001
PARTIAL_RECEIPT_SCHEMA = (
    "hawking.kimi_k26.stale_download_cleanup.partial_failure.v1"
)
_MAX_DOCUMENT_BYTES = 2 * 1024 * 1024


class StaleDownloadCleanupError(RuntimeError):
    """Raised before any unlink when cleanup authority is not exact."""


@dataclass(frozen=True)
class CleanupHooks:
    phase1: supervisor.Phase1Hooks
    process_auditor: supervisor.ProcessAuditor
    sampler: supervisor.ResourceSampler
    clock: supervisor.Clock

    @classmethod
    def live(cls) -> "CleanupHooks":
        return cls(
            phase1=supervisor.Phase1Hooks.live(),
            process_auditor=supervisor.DarwinExactProcessAuditor(),
            sampler=supervisor.SystemResourceSampler(),
            clock=supervisor.SystemClock(),
        )


def _fail(message: str) -> None:
    raise StaleDownloadCleanupError(message)


def _safe_id(value: str) -> str:
    if supervisor._INVOCATION_ID.fullmatch(value) is None:  # noqa: SLF001
        _fail("cleanup id must match [a-z0-9][a-z0-9._-]{0,63}")
    return value


def _audit_name(cleanup_id: str) -> str:
    return f"stale-download-cleanup-audit.{_safe_id(cleanup_id)}.json"


def _receipt_name(cleanup_id: str) -> str:
    return f"stale-download-cleanup-receipt.{_safe_id(cleanup_id)}.json"


def _active_children(entries: Sequence[dict[str, Any]]) -> dict[int, str]:
    active: dict[int, str] = {}
    for entry in entries:
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            continue
        pid = payload.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int):
            continue
        if entry.get("event") == "CHILD_STARTED":
            invocation_id = entry.get("invocation_id")
            if not isinstance(invocation_id, str):
                _fail("journal child start has no exact invocation id")
            if pid in active:
                _fail(f"journal starts an already-active child PID: {pid}")
            active[pid] = invocation_id
        elif entry.get("event") == "CHILD_EXITED":
            invocation_id = entry.get("invocation_id")
            if pid in active and active[pid] != invocation_id:
                _fail(f"journal child exit invocation disagrees for PID: {pid}")
            active.pop(pid, None)
    return active


def _require_no_unfinished_child(entries: Sequence[dict[str, Any]]) -> None:
    active = _active_children(entries)
    if active:
        rendered = ",".join(f"{pid}:{active[pid]}" for pid in sorted(active))
        _fail(f"cleanup requires a journal with no unfinished child: {rendered}")


def _validated_plan_and_process_audit(
    layout: phase1.SessionLayout,
    hooks: CleanupHooks,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = supervisor._validated_plan(  # noqa: SLF001
        layout,
        supplied_plan=None,
        hooks=hooks.phase1,
        manifest_path=phase1.OFFICIAL_MANIFEST,
        mop_root=phase1.MOP_ROOT,
        shared_xet=phase1.SHARED_HF_XET_ROOT,
    )
    process_audit = hooks.process_auditor.audit(layout, plan)
    phase1.verify_sealed_document(process_audit, label="cleanup native process audit")
    if process_audit.get("status") != (
        "PASS_SNAPSHOT_NO_EXISTING_EXACT_SESSION_CACHE_DOWNLOADER_"
        "BEST_EFFORT_WITH_RACE"
    ):
        _fail("native process audit did not grant stale cleanup")
    return plan, process_audit


def _prior_context(
    layout: phase1.SessionLayout, entries: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    prior = supervisor._latest_finished_context(layout, entries)  # noqa: SLF001
    if prior is None:
        _fail("stale cleanup requires a durable prior downloader invocation")
    return prior


def _read_document(
    layout: phase1.SessionLayout, name: str, *, label: str
) -> dict[str, Any]:
    raw = supervisor._read_evidence_leaf(  # noqa: SLF001
        layout, name, maximum_bytes=_MAX_DOCUMENT_BYTES
    )
    if raw is None:
        _fail(f"{label} is absent")
    value = phase1.strict_json_bytes(raw, label=label)
    phase1.verify_sealed_document(value, label=label)
    return value


def audit(
    layout: phase1.SessionLayout,
    *,
    cleanup_id: str,
    hooks: CleanupHooks | None = None,
) -> dict[str, Any]:
    """Inventory exact candidates and durably request explicit confirmation."""
    selected = hooks or CleanupHooks.live()
    cleanup_id = _safe_id(cleanup_id)
    previous_umask = os.umask(0o077)
    try:
        with supervisor._exclusive_lease(layout), supervisor.JournalWriter(layout) as journal:  # noqa: SLF001
            _require_no_unfinished_child(journal.entries)
            _plan, process_audit = _validated_plan_and_process_audit(layout, selected)
            prior = _prior_context(layout, journal.entries)
            after_prior = journal.entries[prior["index"] + 1 :]
            completed_audits = {
                entry["payload"].get("audit_event_seal_sha256")
                for entry in after_prior
                if entry.get("event") == supervisor._CLEANUP_COMPLETED_EVENT  # noqa: SLF001
                and isinstance(entry.get("payload"), dict)
            }
            unresolved = [
                entry
                for entry in after_prior
                if entry.get("event") == supervisor._CLEANUP_AUDIT_EVENT  # noqa: SLF001
                and entry.get("seal_sha256") not in completed_audits
            ]
            if unresolved:
                _fail(
                    "prior cleanup audit is unresolved; retry its exact confirmed "
                    "inventory instead of superseding provenance"
                )
            inventory = supervisor._scan_nonresumable_incomplete_files(  # noqa: SLF001
                layout, manifest_path=phase1.OFFICIAL_MANIFEST
            )
            sampled = selected.sampler.sample(layout)
            value = phase1.seal_document(
                {
                    "schema": AUDIT_SCHEMA,
                    "status": "AWAITING_EXPLICIT_INVENTORY_SEAL_CONFIRMATION",
                    "cleanup_id": cleanup_id,
                    "session": os.fspath(layout.session),
                    "blobs_root": os.fspath(layout.blobs),
                    "created_at_utc": selected.clock.utc_now(),
                    "created_monotonic_ns": selected.clock.monotonic_ns(),
                    "prior_invocation_id": prior["entry"]["invocation_id"],
                    "prior_invocation_status": prior["status"]["status"],
                    "prior_invocation_journal_head_sha256": prior["entry"][
                        "seal_sha256"
                    ],
                    "prior_status_seal_sha256": prior["status"]["seal_sha256"],
                    "supervisor_journal_head_before_audit": journal.head,
                    "manifest_verification_seal_sha256": inventory[
                        "manifest_verification_seal_sha256"
                    ],
                    "manifest_seal_sha256": inventory["manifest_seal_sha256"],
                    "native_process_audit": process_audit,
                    "inventory": inventory,
                    "confirmation": {
                        "required_inventory_seal_sha256": inventory["seal_sha256"],
                        "execute_requires_exact_match": True,
                    },
                    "space_at_audit": {
                        "free_disk_bytes": sampled.free_disk_bytes,
                        "session_allocated_bytes": sampled.session_allocated_bytes,
                    },
                    "deletion_performed": False,
                    "network_accessed": False,
                    "glob_used": False,
                    "recursive_removal_used": False,
                    "final_blob_deletion_authorized": False,
                    "xet_deletion_authorized": False,
                }
            )
            name = _audit_name(cleanup_id)
            supervisor._write_new_document(layout, name, value)  # noqa: SLF001
            journal.append(
                event=supervisor._CLEANUP_AUDIT_EVENT,  # noqa: SLF001
                invocation_id=cleanup_id,
                timestamp_utc=selected.clock.utc_now(),
                monotonic_ns=selected.clock.monotonic_ns(),
                payload={
                    "audit_name": name,
                    "audit_seal_sha256": value["seal_sha256"],
                    "prior_invocation_id": prior["entry"]["invocation_id"],
                    "prior_status_seal_sha256": prior["status"]["seal_sha256"],
                    "manifest_verification_seal_sha256": inventory[
                        "manifest_verification_seal_sha256"
                    ],
                    "manifest_seal_sha256": inventory["manifest_seal_sha256"],
                    "removed_inventory_seal_sha256": inventory["seal_sha256"],
                    "file_count": inventory["file_count"],
                    "logical_bytes": inventory["logical_bytes"],
                    "allocated_bytes": inventory["allocated_bytes"],
                },
            )
            return value
    finally:
        os.umask(previous_umask)


def _require_exact_audit(
    layout: phase1.SessionLayout,
    journal: supervisor.JournalWriter,
    *,
    cleanup_id: str,
    confirmation_inventory_seal: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    audit_value = _read_document(
        layout, _audit_name(cleanup_id), label="stale incomplete cleanup audit"
    )
    if audit_value.get("schema") != AUDIT_SCHEMA \
            or audit_value.get("status") != (
                "AWAITING_EXPLICIT_INVENTORY_SEAL_CONFIRMATION"
            ) \
            or audit_value.get("cleanup_id") != cleanup_id \
            or audit_value.get("session") != os.fspath(layout.session):
        _fail("cleanup audit identity or schema changed")
    inventory = audit_value.get("inventory")
    if not isinstance(inventory, dict):
        _fail("cleanup audit has no sealed inventory")
    phase1.verify_sealed_document(inventory, label="cleanup removal inventory")
    if confirmation_inventory_seal != inventory.get("seal_sha256"):
        _fail("explicit confirmation does not match the exact inventory seal")
    if audit_value.get("confirmation") != {
        "required_inventory_seal_sha256": inventory["seal_sha256"],
        "execute_requires_exact_match": True,
    }:
        _fail("cleanup audit confirmation contract changed")
    matches = [
        entry
        for entry in journal.entries
        if entry.get("event") == supervisor._CLEANUP_AUDIT_EVENT  # noqa: SLF001
        and entry.get("invocation_id") == cleanup_id
        and isinstance(entry.get("payload"), dict)
        and entry["payload"].get("audit_seal_sha256")
        == audit_value["seal_sha256"]
    ]
    if len(matches) != 1:
        _fail("cleanup audit has no unique supervisor journal event")
    audit_event = matches[0]
    payload = audit_event.get("payload")
    assert isinstance(payload, dict)
    prior = _prior_context(layout, journal.entries)
    exact_prior = {
        "prior_invocation_id": prior["entry"]["invocation_id"],
        "prior_invocation_status": prior["status"]["status"],
        "prior_invocation_journal_head_sha256": prior["entry"]["seal_sha256"],
        "prior_status_seal_sha256": prior["status"]["seal_sha256"],
    }
    for key, expected in exact_prior.items():
        if audit_value.get(key) != expected:
            _fail(f"cleanup audit {key} no longer matches the prior invocation")
    if audit_value.get("supervisor_journal_head_before_audit") != audit_event.get(
        "previous_entry_seal_sha256"
    ):
        _fail("cleanup audit does not bind the exact prior journal head")
    exact_event = {
        "audit_name": _audit_name(cleanup_id),
        "audit_seal_sha256": audit_value["seal_sha256"],
        "prior_invocation_id": prior["entry"]["invocation_id"],
        "prior_status_seal_sha256": prior["status"]["seal_sha256"],
        "manifest_verification_seal_sha256": inventory[
            "manifest_verification_seal_sha256"
        ],
        "manifest_seal_sha256": inventory["manifest_seal_sha256"],
        "removed_inventory_seal_sha256": inventory["seal_sha256"],
        "file_count": inventory["file_count"],
        "logical_bytes": inventory["logical_bytes"],
        "allocated_bytes": inventory["allocated_bytes"],
    }
    for key, expected in exact_event.items():
        if payload.get(key) != expected:
            _fail(f"cleanup audit journal {key} changed")
    _validated_inventory_rows(inventory, layout=layout)
    return audit_value, inventory, audit_event


def _metadata_matches(row: dict[str, Any], metadata: os.stat_result) -> bool:
    return all(
        row.get(key) == expected
        for key, expected in {
            "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
            "logical_bytes": int(metadata.st_size),
            "blocks": int(metadata.st_blocks),
            "allocated_bytes": int(metadata.st_blocks) * 512,
            "mtime_ns": int(metadata.st_mtime_ns),
            "ctime_ns": int(metadata.st_ctime_ns),
            "uid": int(metadata.st_uid),
            "gid": int(metadata.st_gid),
            "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
            "hard_links": int(metadata.st_nlink),
        }.items()
    ) and stat.S_ISREG(metadata.st_mode)


def _validated_inventory_rows(
    inventory: dict[str, Any], *, layout: phase1.SessionLayout
) -> list[dict[str, Any]]:
    if inventory.get("schema") != (
        "hawking.kimi_k26.download_supervisor."
        "nonresumable_incomplete_inventory.v1"
    ) or inventory.get("status") != "PASS_EXACT_INVENTORY":
        _fail("cleanup inventory schema or status changed")
    if inventory.get("session") != os.fspath(layout.session) \
            or inventory.get("blobs_root") != os.fspath(layout.blobs):
        _fail("cleanup inventory session/blob root changed")
    files = inventory.get("files")
    count = inventory.get("file_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0 \
            or not isinstance(files, list) or len(files) != count:
        _fail("cleanup inventory file rows are malformed")
    names: list[str] = []
    for row in files:
        if not isinstance(row, dict):
            _fail("cleanup inventory contains a non-object row")
        name = row.get("name")
        if not isinstance(name, str) \
                or supervisor._PROCESS_UNIQUE_INCOMPLETE.fullmatch(name) is None:  # noqa: SLF001
            _fail("cleanup inventory contains an unsafe filename")
        names.append(name)
    if names != sorted(names) or len(set(names)) != len(names):
        _fail("cleanup inventory filenames are not unique canonical order")
    if inventory.get("logical_bytes") != sum(
        int(row.get("logical_bytes", -1)) for row in files
    ) or inventory.get("allocated_bytes") != sum(
        int(row.get("allocated_bytes", -1)) for row in files
    ):
        _fail("cleanup inventory byte totals changed")
    if bool(files) and inventory.get("blobs_root_present") is not True:
        _fail("cleanup inventory has files without a blob directory")
    return files


def _row_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(phase1.canonical_json(row)).hexdigest()


def _audit_index(
    entries: Sequence[dict[str, Any]], audit_event: dict[str, Any]
) -> int:
    seal = audit_event["seal_sha256"]
    matches = [
        index for index, entry in enumerate(entries) if entry.get("seal_sha256") == seal
    ]
    if len(matches) != 1:
        _fail("cleanup audit journal position is not unique")
    return matches[0]


def _cleanup_lifecycle(
    journal: supervisor.JournalWriter,
    *,
    cleanup_id: str,
    audit_event: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    files = _validated_inventory_rows(inventory, layout=journal.layout)
    start_events: list[dict[str, Any]] = []
    progress_events: list[dict[str, Any]] = []
    partial_events: list[dict[str, Any]] = []
    completion_events: list[dict[str, Any]] = []
    allowed = {
        supervisor._CLEANUP_STARTED_EVENT,  # noqa: SLF001
        supervisor._CLEANUP_UNLINK_EVENT,  # noqa: SLF001
        supervisor._CLEANUP_PARTIAL_EVENT,  # noqa: SLF001
        supervisor._CLEANUP_COMPLETED_EVENT,  # noqa: SLF001
    }
    for entry in journal.entries[_audit_index(journal.entries, audit_event) + 1 :]:
        event = entry.get("event")
        payload = entry.get("payload")
        if event not in allowed or entry.get("invocation_id") != cleanup_id \
                or not isinstance(payload, dict):
            _fail("cleanup audit is followed by an unrelated journal event")
        if payload.get("audit_event_seal_sha256") != audit_event["seal_sha256"] \
                or payload.get("removed_inventory_seal_sha256") != inventory[
                    "seal_sha256"
                ]:
            _fail("cleanup lifecycle event is not bound to the exact audit inventory")
        if event == supervisor._CLEANUP_STARTED_EVENT:  # noqa: SLF001
            attempt = payload.get("attempt")
            if isinstance(attempt, bool) or not isinstance(attempt, int) \
                    or attempt != len(start_events) + 1:
                _fail("cleanup attempt journal sequence changed")
            if payload.get("completed_before_count") != len(progress_events):
                _fail("cleanup attempt start does not bind prior committed rows")
            start_events.append(entry)
        elif event == supervisor._CLEANUP_UNLINK_EVENT:  # noqa: SLF001
            row_index = payload.get("row_index")
            if isinstance(row_index, bool) or not isinstance(row_index, int) \
                    or row_index != len(progress_events) or row_index >= len(files):
                _fail("cleanup unlink progress is not the original inventory prefix")
            attempt = payload.get("attempt")
            if isinstance(attempt, bool) or not isinstance(attempt, int) \
                    or attempt < 1 or attempt > len(start_events):
                _fail("cleanup unlink progress has no anchored attempt")
            row = files[row_index]
            exact = {
                "row_sha256": _row_sha256(row),
                "name": row["name"],
                "device": row["device"],
                "inode": row["inode"],
                "logical_bytes": row["logical_bytes"],
                "allocated_bytes": row["allocated_bytes"],
                "blocks": row["blocks"],
                "mtime_ns": row["mtime_ns"],
            }
            for key, expected in exact.items():
                if payload.get(key) != expected:
                    _fail(f"cleanup unlink progress {key} changed")
            if payload.get("commit_status") not in {
                "DIRFD_UNLINK_BLOB_DIRECTORY_FSYNCED_AND_JOURNAL_COMMITTED",
                "ANCHORED_MISSING_ROW_RECONCILED_DIRECTORY_FSYNCED",
            }:
                _fail("cleanup unlink progress commit status changed")
            progress_events.append(entry)
        elif event == supervisor._CLEANUP_PARTIAL_EVENT:  # noqa: SLF001
            if payload.get("attempt") not in range(1, len(start_events) + 1):
                _fail("cleanup partial failure has no anchored attempt")
            partial_events.append(entry)
        else:
            completion_events.append(entry)
    if len(completion_events) > 1:
        _fail("cleanup audit has multiple completion events")
    if completion_events and journal.entries[-1] is not completion_events[0]:
        _fail("cleanup completion is not terminal in the journal")
    return {
        "starts": start_events,
        "progress": progress_events,
        "partials": partial_events,
        "completions": completion_events,
    }


def _append_cleanup_event(
    journal: supervisor.JournalWriter,
    *,
    event: str,
    cleanup_id: str,
    audit_event: dict[str, Any],
    inventory: dict[str, Any],
    clock: supervisor.Clock,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return journal.append(
        event=event,
        invocation_id=cleanup_id,
        timestamp_utc=clock.utc_now(),
        monotonic_ns=clock.monotonic_ns(),
        payload={
            "audit_event_seal_sha256": audit_event["seal_sha256"],
            "removed_inventory_seal_sha256": inventory["seal_sha256"],
            **payload,
        },
    )


def _append_progress(
    journal: supervisor.JournalWriter,
    *,
    cleanup_id: str,
    audit_event: dict[str, Any],
    inventory: dict[str, Any],
    clock: supervisor.Clock,
    attempt: int,
    row_index: int,
    commit_status: str,
) -> dict[str, Any]:
    row = _validated_inventory_rows(inventory, layout=journal.layout)[row_index]
    return _append_cleanup_event(
        journal,
        event=supervisor._CLEANUP_UNLINK_EVENT,  # noqa: SLF001
        cleanup_id=cleanup_id,
        audit_event=audit_event,
        inventory=inventory,
        clock=clock,
        payload={
            "attempt": attempt,
            "row_index": row_index,
            "row_sha256": _row_sha256(row),
            "name": row["name"],
            "manifest_sha256": row["manifest_sha256"],
            "device": row["device"],
            "inode": row["inode"],
            "logical_bytes": row["logical_bytes"],
            "allocated_bytes": row["allocated_bytes"],
            "blocks": row["blocks"],
            "mtime_ns": row["mtime_ns"],
            "commit_status": commit_status,
        },
    )


def _fsync_original_blob_directory(
    layout: phase1.SessionLayout, inventory: dict[str, Any]
) -> bool:
    if not inventory.get("blobs_root_present"):
        if inventory.get("file_count"):
            _fail("cleanup inventory claims files without a blob directory")
        return False
    descriptor = phase1._open_absolute_directory(layout.blobs)  # noqa: SLF001
    try:
        root = os.fstat(descriptor)
        if inventory.get("blobs_root_identity") != {
            "device": int(root.st_dev),
            "inode": int(root.st_ino),
            "uid": int(root.st_uid),
            "mode": f"{stat.S_IMODE(root.st_mode):04o}",
        }:
            _fail("blob directory identity changed after confirmation")
        os.fsync(descriptor)
        return True
    finally:
        os.close(descriptor)


def _reconcile_current_inventory(
    layout: phase1.SessionLayout,
    *,
    cleanup_id: str,
    journal: supervisor.JournalWriter,
    audit_event: dict[str, Any],
    inventory: dict[str, Any],
    clock: supervisor.Clock,
    permit_one_anchored_missing: bool,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    files = _validated_inventory_rows(inventory, layout=layout)
    lifecycle = _cleanup_lifecycle(
        journal,
        cleanup_id=cleanup_id,
        audit_event=audit_event,
        inventory=inventory,
    )
    if lifecycle["completions"]:
        _fail("cleanup audit is already complete")
    current = supervisor._scan_nonresumable_incomplete_files(  # noqa: SLF001
        layout, manifest_path=phase1.OFFICIAL_MANIFEST
    )
    for key in (
        "session",
        "blobs_root",
        "blobs_root_present",
        "blobs_root_identity",
        "manifest_verification_seal_sha256",
        "manifest_seal_sha256",
        "filename_contract",
    ):
        if current.get(key) != inventory.get(key):
            _fail(f"cleanup current inventory {key} changed after audit")
    current_rows = {
        row["name"]: row
        for row in _validated_inventory_rows(current, layout=layout)
    }
    original_names = {row["name"] for row in files}
    if not set(current_rows).issubset(original_names):
        _fail("cleanup current inventory contains a non-audited name")
    completed = len(lifecycle["progress"])
    for row in files[:completed]:
        if row["name"] in current_rows:
            _fail("a journal-committed cleanup row reappeared")
    missing: list[int] = []
    for index, row in enumerate(files[completed:], start=completed):
        current_row = current_rows.get(row["name"])
        if current_row is None:
            missing.append(index)
        elif phase1.canonical_json(current_row) != phase1.canonical_json(row):
            _fail(f"cleanup target changed after confirmation: {row['name']}")
    if missing:
        if not permit_one_anchored_missing or not lifecycle["starts"] \
                or missing != [completed]:
            _fail("cleanup inventory changed outside one anchored uncommitted unlink")
        _fsync_original_blob_directory(layout, inventory)
        prior_attempt = lifecycle["starts"][-1]["payload"]["attempt"]
        _append_progress(
            journal,
            cleanup_id=cleanup_id,
            audit_event=audit_event,
            inventory=inventory,
            clock=clock,
            attempt=prior_attempt,
            row_index=completed,
            commit_status=(
                "ANCHORED_MISSING_ROW_RECONCILED_DIRECTORY_FSYNCED"
            ),
        )
        lifecycle = _cleanup_lifecycle(
            journal,
            cleanup_id=cleanup_id,
            audit_event=audit_event,
            inventory=inventory,
        )
    return current, lifecycle


def _unlink_remaining_inventory(
    layout: phase1.SessionLayout,
    *,
    cleanup_id: str,
    journal: supervisor.JournalWriter,
    audit_event: dict[str, Any],
    inventory: dict[str, Any],
    clock: supervisor.Clock,
    attempt: int,
    completed_count: int,
) -> bool:
    files = _validated_inventory_rows(inventory, layout=layout)
    if not inventory.get("blobs_root_present"):
        if files:
            _fail("cleanup inventory claims files without a blob directory")
        return False
    descriptor = phase1._open_absolute_directory(layout.blobs)  # noqa: SLF001
    try:
        root = os.fstat(descriptor)
        if inventory.get("blobs_root_identity") != {
            "device": int(root.st_dev),
            "inode": int(root.st_ino),
            "uid": int(root.st_uid),
            "mode": f"{stat.S_IMODE(root.st_mode):04o}",
        }:
            _fail("blob directory identity changed after confirmation")
        for row in files[completed_count:]:
            named = os.stat(
                row["name"], dir_fd=descriptor, follow_symlinks=False
            )
            if not _metadata_matches(row, named):
                _fail(f"cleanup target changed after confirmation: {row['name']}")
        for row_index, row in enumerate(
            files[completed_count:], start=completed_count
        ):
            os.unlink(row["name"], dir_fd=descriptor)
            os.fsync(descriptor)
            try:
                os.stat(row["name"], dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                _fail(f"cleanup target still exists after exact unlink: {row['name']}")
            _append_progress(
                journal,
                cleanup_id=cleanup_id,
                audit_event=audit_event,
                inventory=inventory,
                clock=clock,
                attempt=attempt,
                row_index=row_index,
                commit_status=(
                    "DIRFD_UNLINK_BLOB_DIRECTORY_FSYNCED_AND_JOURNAL_COMMITTED"
                ),
            )
        os.fsync(descriptor)
        return True
    finally:
        os.close(descriptor)


def _progress_chain(lifecycle: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    seals = [entry["seal_sha256"] for entry in lifecycle["progress"]]
    return {
        "entry_seal_sha256s": seals,
        "chain_sha256": hashlib.sha256(
            phase1.canonical_json({"entry_seal_sha256s": seals})
        ).hexdigest(),
    }


def _write_partial_failure(
    layout: phase1.SessionLayout,
    *,
    cleanup_id: str,
    journal: supervisor.JournalWriter,
    audit_value: dict[str, Any],
    audit_event: dict[str, Any],
    inventory: dict[str, Any],
    clock: supervisor.Clock,
    attempt: int,
    fault: BaseException,
) -> dict[str, Any]:
    lifecycle = _cleanup_lifecycle(
        journal,
        cleanup_id=cleanup_id,
        audit_event=audit_event,
        inventory=inventory,
    )
    files = _validated_inventory_rows(inventory, layout=layout)
    completed = len(lifecycle["progress"])
    current = supervisor._scan_nonresumable_incomplete_files(  # noqa: SLF001
        layout, manifest_path=phase1.OFFICIAL_MANIFEST
    )
    progress_chain = _progress_chain(lifecycle)
    receipt = phase1.seal_document(
        {
            "schema": PARTIAL_RECEIPT_SCHEMA,
            "status": "PARTIAL_FAILURE_NO_DIRECT_16_RESUME_AUTHORITY",
            "cleanup_id": cleanup_id,
            "attempt": attempt,
            "session": os.fspath(layout.session),
            "prior_invocation_id": audit_value["prior_invocation_id"],
            "prior_invocation_journal_head_sha256": audit_value[
                "prior_invocation_journal_head_sha256"
            ],
            "prior_status_seal_sha256": audit_value["prior_status_seal_sha256"],
            "audit_document_seal_sha256": audit_value["seal_sha256"],
            "audit_event_seal_sha256": audit_event["seal_sha256"],
            "removed_inventory_seal_sha256": inventory["seal_sha256"],
            "original_file_count": inventory["file_count"],
            "original_logical_bytes": inventory["logical_bytes"],
            "original_allocated_bytes": inventory["allocated_bytes"],
            "committed_file_count": completed,
            "committed_logical_bytes": sum(
                row["logical_bytes"] for row in files[:completed]
            ),
            "committed_allocated_bytes": sum(
                row["allocated_bytes"] for row in files[:completed]
            ),
            "remaining_file_count": len(files) - completed,
            "post_failure_incomplete_count": current["file_count"],
            "post_failure_inventory_seal_sha256": current["seal_sha256"],
            "progress_entry_seal_sha256s": progress_chain[
                "entry_seal_sha256s"
            ],
            "progress_chain_sha256": progress_chain["chain_sha256"],
            "fault": f"{type(fault).__name__}: {fault}"[:2_000],
            "completed_at_utc": clock.utc_now(),
            "completed_monotonic_ns": clock.monotonic_ns(),
            "direct_16_resume_authorized": False,
            "network_accessed": False,
            "final_blob_deletion_performed": False,
            "xet_deletion_performed": False,
        }
    )
    name = (
        f"stale-download-cleanup-partial.{cleanup_id}."
        f"attempt-{attempt:03d}.json"
    )
    supervisor._write_new_document(layout, name, receipt)  # noqa: SLF001
    _append_cleanup_event(
        journal,
        event=supervisor._CLEANUP_PARTIAL_EVENT,  # noqa: SLF001
        cleanup_id=cleanup_id,
        audit_event=audit_event,
        inventory=inventory,
        clock=clock,
        payload={
            "attempt": attempt,
            "partial_receipt_name": name,
            "partial_receipt_seal_sha256": receipt["seal_sha256"],
            "committed_file_count": completed,
            "remaining_file_count": len(files) - completed,
            "progress_chain_sha256": progress_chain["chain_sha256"],
            "direct_16_resume_authorized": False,
        },
    )
    return receipt


def execute(
    layout: phase1.SessionLayout,
    *,
    cleanup_id: str,
    confirmation_inventory_seal: str,
    hooks: CleanupHooks | None = None,
) -> dict[str, Any]:
    """Revalidate a confirmed audit and unlink only its exact dirfd leaves."""
    selected = hooks or CleanupHooks.live()
    cleanup_id = _safe_id(cleanup_id)
    if supervisor._HEX64.fullmatch(confirmation_inventory_seal) is None:  # noqa: SLF001
        _fail("confirmation inventory seal must be lowercase SHA-256")
    previous_umask = os.umask(0o077)
    try:
        with supervisor._exclusive_lease(layout), supervisor.JournalWriter(layout) as journal:  # noqa: SLF001
            _require_no_unfinished_child(journal.entries)
            _plan, process_audit = _validated_plan_and_process_audit(layout, selected)
            audit_value, inventory, audit_event = _require_exact_audit(
                layout,
                journal,
                cleanup_id=cleanup_id,
                confirmation_inventory_seal=confirmation_inventory_seal,
            )
            _current, lifecycle = _reconcile_current_inventory(
                layout,
                cleanup_id=cleanup_id,
                journal=journal,
                audit_event=audit_event,
                inventory=inventory,
                clock=selected.clock,
                permit_one_anchored_missing=bool(
                    _cleanup_lifecycle(
                        journal,
                        cleanup_id=cleanup_id,
                        audit_event=audit_event,
                        inventory=inventory,
                    )["starts"]
                ),
            )
            attempt = len(lifecycle["starts"]) + 1
            attempt_before = selected.sampler.sample(layout)
            _append_cleanup_event(
                journal,
                event=supervisor._CLEANUP_STARTED_EVENT,  # noqa: SLF001
                cleanup_id=cleanup_id,
                audit_event=audit_event,
                inventory=inventory,
                clock=selected.clock,
                payload={
                    "attempt": attempt,
                    "completed_before_count": len(lifecycle["progress"]),
                    "remaining_before_count": (
                        inventory["file_count"] - len(lifecycle["progress"])
                    ),
                    "native_process_audit_seal_sha256": process_audit[
                        "seal_sha256"
                    ],
                },
            )
            try:
                blob_directory_fsynced = _unlink_remaining_inventory(
                    layout,
                    cleanup_id=cleanup_id,
                    journal=journal,
                    audit_event=audit_event,
                    inventory=inventory,
                    clock=selected.clock,
                    attempt=attempt,
                    completed_count=len(lifecycle["progress"]),
                )
                after_inventory = supervisor._scan_nonresumable_incomplete_files(  # noqa: SLF001
                    layout, manifest_path=phase1.OFFICIAL_MANIFEST
                )
                supervisor._assert_no_nonresumable_incomplete(after_inventory)  # noqa: SLF001
                lifecycle = _cleanup_lifecycle(
                    journal,
                    cleanup_id=cleanup_id,
                    audit_event=audit_event,
                    inventory=inventory,
                )
                if len(lifecycle["progress"]) != inventory["file_count"]:
                    _fail("not every original inventory row has a durable commit")
                after = selected.sampler.sample(layout)
            except BaseException as original_fault:
                evidence_notes: list[str] = []
                try:
                    _reconcile_current_inventory(
                        layout,
                        cleanup_id=cleanup_id,
                        journal=journal,
                        audit_event=audit_event,
                        inventory=inventory,
                        clock=selected.clock,
                        permit_one_anchored_missing=True,
                    )
                except BaseException as reconcile_fault:
                    evidence_notes.append(
                        f"partial reconciliation failed: {type(reconcile_fault).__name__}: "
                        f"{reconcile_fault}"
                    )
                try:
                    _write_partial_failure(
                        layout,
                        cleanup_id=cleanup_id,
                        journal=journal,
                        audit_value=audit_value,
                        audit_event=audit_event,
                        inventory=inventory,
                        clock=selected.clock,
                        attempt=attempt,
                        fault=original_fault,
                    )
                except BaseException as evidence_fault:
                    evidence_notes.append(
                        f"partial evidence failed: {type(evidence_fault).__name__}: "
                        f"{evidence_fault}"
                    )
                for note in evidence_notes:
                    with_context = getattr(original_fault, "add_note", None)
                    if callable(with_context):
                        with_context(note)
                raise
            prior_id = audit_value["prior_invocation_id"]
            progress_chain = _progress_chain(lifecycle)
            audit_space = audit_value.get("space_at_audit")
            if not isinstance(audit_space, dict):
                _fail("cleanup audit has no initial space measurement")
            audit_free = audit_space.get("free_disk_bytes")
            audit_allocated = audit_space.get("session_allocated_bytes")
            if isinstance(audit_free, bool) or not isinstance(audit_free, int) \
                    or isinstance(audit_allocated, bool) \
                    or not isinstance(audit_allocated, int):
                _fail("cleanup audit space measurement changed")
            receipt = phase1.seal_document(
                {
                    "schema": RECEIPT_SCHEMA,
                    "status": "PASS_EXACT_STALE_INCOMPLETE_CLEANUP",
                    "cleanup_id": cleanup_id,
                    "session": os.fspath(layout.session),
                    "blobs_root": os.fspath(layout.blobs),
                    "completed_at_utc": selected.clock.utc_now(),
                    "completed_monotonic_ns": selected.clock.monotonic_ns(),
                    "prior_invocation_id": prior_id,
                    "prior_invocation_status": audit_value[
                        "prior_invocation_status"
                    ],
                    "prior_invocation_journal_head_sha256": audit_value[
                        "prior_invocation_journal_head_sha256"
                    ],
                    "prior_status_seal_sha256": audit_value[
                        "prior_status_seal_sha256"
                    ],
                    "supervisor_journal_head_before_execute": journal.head,
                    "manifest_verification_seal_sha256": inventory[
                        "manifest_verification_seal_sha256"
                    ],
                    "manifest_seal_sha256": inventory["manifest_seal_sha256"],
                    "audit_document_seal_sha256": audit_value["seal_sha256"],
                    "audit_event_seal_sha256": audit_event["seal_sha256"],
                    "removed_inventory_seal_sha256": inventory["seal_sha256"],
                    "removed_file_count": inventory["file_count"],
                    "removed_logical_bytes": inventory["logical_bytes"],
                    "removed_allocated_bytes": inventory["allocated_bytes"],
                    "post_cleanup_inventory_seal_sha256": after_inventory[
                        "seal_sha256"
                    ],
                    "post_cleanup_incomplete_count": after_inventory["file_count"],
                    "cleanup_attempt_count": len(lifecycle["starts"]),
                    "progress_entry_seal_sha256s": progress_chain[
                        "entry_seal_sha256s"
                    ],
                    "progress_chain_sha256": progress_chain["chain_sha256"],
                    "all_original_inventory_rows_committed": True,
                    "supervisor_journal_head_before_final_receipt": journal.head,
                    "space_before": {
                        "free_disk_bytes": audit_free,
                        "session_allocated_bytes": audit_allocated,
                    },
                    "final_attempt_space_before": {
                        "free_disk_bytes": attempt_before.free_disk_bytes,
                        "session_allocated_bytes": (
                            attempt_before.session_allocated_bytes
                        ),
                    },
                    "space_after": {
                        "free_disk_bytes": after.free_disk_bytes,
                        "session_allocated_bytes": after.session_allocated_bytes,
                    },
                    "free_disk_delta_bytes": (
                        after.free_disk_bytes - audit_free
                    ),
                    "session_allocated_delta_bytes": (
                        after.session_allocated_bytes
                        - audit_allocated
                    ),
                    "native_process_audit": process_audit,
                    "exact_dirfd_unlinks_only": True,
                    "blob_directory_fsynced": blob_directory_fsynced,
                    "evidence_file_and_directory_fsynced": True,
                    "glob_used": False,
                    "recursive_removal_used": False,
                    "final_blob_deletion_performed": False,
                    "xet_deletion_performed": False,
                    "network_accessed": False,
                }
            )
            name = _receipt_name(cleanup_id)
            supervisor._write_new_document(layout, name, receipt)  # noqa: SLF001
            journal.append(
                event=supervisor._CLEANUP_COMPLETED_EVENT,  # noqa: SLF001
                invocation_id=cleanup_id,
                timestamp_utc=selected.clock.utc_now(),
                monotonic_ns=selected.clock.monotonic_ns(),
                payload={
                    "receipt_name": name,
                    "receipt_seal_sha256": receipt["seal_sha256"],
                    "prior_invocation_id": prior_id,
                    "prior_status_seal_sha256": audit_value[
                        "prior_status_seal_sha256"
                    ],
                    "audit_event_seal_sha256": audit_event["seal_sha256"],
                    "manifest_verification_seal_sha256": inventory[
                        "manifest_verification_seal_sha256"
                    ],
                    "manifest_seal_sha256": inventory["manifest_seal_sha256"],
                    "removed_inventory_seal_sha256": inventory["seal_sha256"],
                    "removed_file_count": inventory["file_count"],
                    "removed_logical_bytes": inventory["logical_bytes"],
                    "removed_allocated_bytes": inventory["allocated_bytes"],
                    "post_cleanup_incomplete_count": 0,
                    "cleanup_attempt_count": len(lifecycle["starts"]),
                    "progress_chain_sha256": progress_chain["chain_sha256"],
                    "all_original_inventory_rows_committed": True,
                },
            )
            return receipt
    finally:
        os.umask(previous_umask)


def _print_json(value: dict[str, Any]) -> None:
    sys.stdout.buffer.write(phase1.canonical_json(value) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "execute"))
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--cleanup-id", required=True)
    parser.add_argument("--confirm-inventory-seal")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        layout = phase1.layout_for(args.session)
        if args.command == "audit":
            if args.confirm_inventory_seal is not None:
                _fail("audit does not accept confirmation")
            value = audit(layout, cleanup_id=args.cleanup_id)
        else:
            if args.confirm_inventory_seal is None:
                _fail("execute requires --confirm-inventory-seal")
            value = execute(
                layout,
                cleanup_id=args.cleanup_id,
                confirmation_inventory_seal=args.confirm_inventory_seal,
            )
        _print_json(value)
        return 0
    except (
        StaleDownloadCleanupError,
        supervisor.DownloadSupervisorError,
        phase1.ReleaseCycleError,
        OSError,
    ) as exc:
        _print_json(
            phase1.seal_document(
                {
                    "schema": "hawking.kimi_k26.stale_download_cleanup.error.v1",
                    "status": "BLOCKED_NO_UNLINK_AUTHORITY",
                    "error": f"{type(exc).__name__}: {exc}"[:2_000],
                    "network_accessed": False,
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
