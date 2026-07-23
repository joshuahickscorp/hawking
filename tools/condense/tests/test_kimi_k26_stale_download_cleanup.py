from __future__ import annotations

import ast
import copy
import fcntl
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

CONDENSE = Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import kimi_k26_download_supervisor as supervisor  # noqa: E402
import kimi_k26_release_cycle as phase1  # noqa: E402
import kimi_k26_stale_download_cleanup as cleanup  # noqa: E402


def _private(path: Path) -> None:
    path.mkdir(mode=0o700, parents=False, exist_ok=False)


def _layout(tmp_path: Path) -> phase1.SessionLayout:
    parent = tmp_path / "sessions"
    _private(parent)
    layout = phase1.layout_for(parent / "case", parent=parent)
    _private(layout.session)
    for path in (
        layout.hub,
        layout.xet,
        layout.build,
        layout.recovery,
        layout.evidence,
    ):
        _private(path)
    _private(layout.tmp)
    _private(layout.hf_home)
    _private(layout.model_cache)
    _private(layout.blobs)
    return layout


def _environment(layout: phase1.SessionLayout, workers: int) -> dict[str, str]:
    return {
        "HF_HOME": os.fspath(layout.hf_home),
        "HF_HUB_CACHE": os.fspath(layout.hub),
        "HF_XET_CACHE": os.fspath(layout.xet),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "HF_HUB_OFFLINE": "0",
        "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
        "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": str(workers),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": os.fspath(layout.tmp / "pycache"),
        "PYTHONSAFEPATH": "1",
        "TEMP": os.fspath(layout.tmp),
        "TMP": os.fspath(layout.tmp),
        "TMPDIR": os.fspath(layout.tmp),
    }


def _argv(layout: phase1.SessionLayout, workers: int) -> list[str]:
    return [
        os.fspath(phase1.HF_CLI),
        "download",
        phase1.KIMI_REPO,
        "--revision",
        phase1.KIMI_REVISION,
        "--repo-type",
        "model",
        "--cache-dir",
        os.fspath(layout.hub),
        "--max-workers",
        str(workers),
    ]


def _plan(layout: phase1.SessionLayout) -> dict[str, Any]:
    runtime = phase1.seal_document(
        {"schema": "hawking.test.fake_runtime.v1", "status": "PASS"}
    )
    profiles = []
    for workers, profile_id in ((8, "PRIMARY_8"), (16, "CONDITIONAL_RESTART_16")):
        profiles.append(
            {
                "profile_id": profile_id,
                "command_argv": _argv(layout, workers),
                "environment": _environment(layout, workers),
            }
        )
    return phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.release_cycle.download_plan.v1",
            "status": "PLANNED_NOT_EXECUTED",
            "command_argv": _argv(layout, 8),
            "environment_mode": "REPLACE_NOT_MERGE",
            "environment": _environment(layout, 8),
            "transfer_runtime": runtime,
            "transfer_runtime_seal_sha256": runtime["seal_sha256"],
            "restart_profiles": profiles,
        }
    )


class FakePhase1:
    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan

    def build(self, _layout: phase1.SessionLayout, **_kwargs: Any) -> dict[str, Any]:
        return copy.deepcopy(self.plan)

    def verify(
        self, value: dict[str, Any], _layout: phase1.SessionLayout, **_kwargs: Any
    ) -> dict[str, Any]:
        phase1.verify_sealed_document(value)
        assert phase1.canonical_json(value) == phase1.canonical_json(self.plan)
        return value

    def runtime(self, value: dict[str, Any]) -> dict[str, Any]:
        assert value == self.plan["transfer_runtime"]
        phase1.verify_sealed_document(value)
        return value

    def hooks(self) -> supervisor.Phase1Hooks:
        return supervisor.Phase1Hooks(self.build, self.verify, self.runtime)


class FakeProcessAudit:
    def __init__(self, *, conflict: bool = False) -> None:
        self.conflict = conflict
        self.calls = 0

    def audit(
        self, _layout: phase1.SessionLayout, _plan: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls += 1
        if self.conflict:
            raise supervisor.DownloadSupervisorError("fake exact-cache conflict")
        return phase1.seal_document(
            {
                "schema": "hawking.kimi_k26.download_supervisor.process_audit.v1",
                "status": (
                    "PASS_SNAPSHOT_NO_EXISTING_EXACT_SESSION_CACHE_DOWNLOADER_"
                    "BEST_EFFORT_WITH_RACE"
                ),
                "method": "FAKE_STRUCTURED",
                "conflict_count": 0,
            }
        )


class FakeSampler:
    def __init__(self) -> None:
        self.calls = 0

    def sample(self, _layout: phase1.SessionLayout) -> supervisor.ResourceSnapshot:
        self.calls += 1
        return supervisor.ResourceSnapshot(
            free_disk_bytes=700_000_000_000 + self.calls * 4096,
            session_allocated_bytes=30_000_000_000 - self.calls * 4096,
        )


class FakeClock:
    def __init__(self) -> None:
        self.ns = 0

    def utc_now(self) -> str:
        self.ns += 1
        return f"2026-07-21T00:00:00.{self.ns:06d}Z"

    def monotonic_ns(self) -> int:
        self.ns += 1
        return self.ns

    def sleep(self, _seconds: float) -> None:
        raise AssertionError("cleanup never sleeps")


def _hooks(
    layout: phase1.SessionLayout, *, conflict: bool = False
) -> cleanup.CleanupHooks:
    return cleanup.CleanupHooks(
        phase1=FakePhase1(_plan(layout)).hooks(),
        process_auditor=FakeProcessAudit(conflict=conflict),
        sampler=FakeSampler(),
        clock=FakeClock(),
    )


def _seed_finished(
    layout: phase1.SessionLayout,
    *,
    invocation_id: str = "prior-run",
    outcome: str = "RESOURCE_GUARD_TERMINATED_RESUMABLE_CACHE_PRESERVED",
) -> dict[str, Any]:
    status_value = phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.download_supervisor.status.v1",
            "status": outcome,
            "invocation_id": invocation_id,
            "exit_code": -9,
        }
    )
    name = f"status.{invocation_id}.json"
    supervisor._write_new_document(layout, name, status_value)
    with supervisor.JournalWriter(layout) as journal:
        entry = journal.append(
            event="INVOCATION_FINISHED",
            invocation_id=invocation_id,
            timestamp_utc="2026-07-21T00:00:00.000000Z",
            monotonic_ns=1,
            payload={
                "pid": 50001,
                "status": outcome,
                "exit_code": -9,
                "status_path": os.fspath(layout.evidence / name),
                "status_seal_sha256": status_value["seal_sha256"],
            },
        )
    return entry


def _manifest_sha() -> str:
    value = json.loads(phase1.OFFICIAL_MANIFEST.read_text(encoding="utf-8"))
    return next(row["sha256"] for row in value["files"] if row["sha256"])


def _incomplete(layout: phase1.SessionLayout, nonce: str = "deadbeef") -> Path:
    path = layout.blobs / f"{_manifest_sha()}.{nonce}.incomplete"
    path.write_bytes(b"partial-xet-object" * 64)
    os.chmod(path, 0o600)
    return path


def test_two_phase_exact_cleanup_preserves_final_blob_and_xet(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    _seed_finished(layout)
    partial = _incomplete(layout)
    final = layout.blobs / _manifest_sha()
    final.write_bytes(b"already-final")
    os.chmod(final, 0o600)
    xet = layout.xet / "must-remain"
    xet.write_bytes(b"xet")
    os.chmod(xet, 0o600)

    audit_value = cleanup.audit(
        layout, cleanup_id="cleanup-one", hooks=_hooks(layout)
    )
    phase1.verify_sealed_document(audit_value, label="cleanup audit")
    assert audit_value["inventory"]["file_count"] == 1
    assert partial.exists()
    with pytest.raises(cleanup.StaleDownloadCleanupError, match="confirmation"):
        cleanup.execute(
            layout,
            cleanup_id="cleanup-one",
            confirmation_inventory_seal="0" * 64,
            hooks=_hooks(layout),
        )
    assert partial.exists()

    receipt = cleanup.execute(
        layout,
        cleanup_id="cleanup-one",
        confirmation_inventory_seal=audit_value["inventory"]["seal_sha256"],
        hooks=_hooks(layout),
    )
    phase1.verify_sealed_document(receipt, label="cleanup receipt")
    assert receipt["status"] == "PASS_EXACT_STALE_INCOMPLETE_CLEANUP"
    assert receipt["removed_file_count"] == 1
    assert receipt["post_cleanup_incomplete_count"] == 0
    assert receipt["free_disk_delta_bytes"] == 4096
    assert receipt["session_allocated_delta_bytes"] == -4096
    assert not partial.exists()
    assert final.read_bytes() == b"already-final"
    assert xet.read_bytes() == b"xet"
    journal = supervisor._verify_journal_bytes(
        (layout.evidence / supervisor._JOURNAL_NAME).read_bytes()
    )
    assert journal[-3]["event"] == supervisor._CLEANUP_STARTED_EVENT
    assert journal[-2]["event"] == supervisor._CLEANUP_UNLINK_EVENT
    assert journal[-1]["event"] == supervisor._CLEANUP_COMPLETED_EVENT
    for path in layout.evidence.iterdir():
        metadata = path.lstat()
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o600


def test_execute_refuses_inode_or_mtime_change_after_confirmation(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    _seed_finished(layout)
    partial = _incomplete(layout)
    audit_value = cleanup.audit(
        layout, cleanup_id="cleanup-mutated", hooks=_hooks(layout)
    )
    partial.write_bytes(partial.read_bytes() + b"changed")
    os.chmod(partial, 0o600)
    with pytest.raises(
        cleanup.StaleDownloadCleanupError, match="changed after confirmation"
    ):
        cleanup.execute(
            layout,
            cleanup_id="cleanup-mutated",
            confirmation_inventory_seal=audit_value["inventory"]["seal_sha256"],
            hooks=_hooks(layout),
        )
    assert partial.exists()


def test_fault_after_unlink_is_journaled_and_retry_receipt_binds_original_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    _seed_finished(layout)
    manifest = json.loads(phase1.OFFICIAL_MANIFEST.read_text(encoding="utf-8"))
    digests = [row["sha256"] for row in manifest["files"] if row["sha256"]][:2]
    partials: list[Path] = []
    for digest, nonce in zip(digests, ("11111111", "22222222"), strict=True):
        path = layout.blobs / f"{digest}.{nonce}.incomplete"
        path.write_bytes(b"partial" * 128)
        os.chmod(path, 0o600)
        partials.append(path)
    hooks = _hooks(layout)
    audit_value = cleanup.audit(layout, cleanup_id="crash-retry", hooks=hooks)
    real_unlink = cleanup.os.unlink
    calls = 0

    def unlink_then_fault(name: str, *, dir_fd: int) -> None:
        nonlocal calls
        calls += 1
        real_unlink(name, dir_fd=dir_fd)
        if calls == 2:
            raise OSError("injected fault after second exact unlink")

    with monkeypatch.context() as scoped:
        scoped.setattr(cleanup.os, "unlink", unlink_then_fault)
        with pytest.raises(OSError, match="injected fault"):
            cleanup.execute(
                layout,
                cleanup_id="crash-retry",
                confirmation_inventory_seal=audit_value["inventory"]["seal_sha256"],
                hooks=hooks,
            )
    assert not any(path.exists() for path in partials)
    partial_receipt = json.loads(
        (
            layout.evidence
            / "stale-download-cleanup-partial.crash-retry.attempt-001.json"
        ).read_text(encoding="utf-8")
    )
    assert partial_receipt["status"] == (
        "PARTIAL_FAILURE_NO_DIRECT_16_RESUME_AUTHORITY"
    )
    assert partial_receipt["committed_file_count"] == 2
    journal = supervisor._verify_journal_bytes(
        (layout.evidence / supervisor._JOURNAL_NAME).read_bytes()
    )
    assert journal[-1]["event"] == supervisor._CLEANUP_PARTIAL_EVENT

    receipt = cleanup.execute(
        layout,
        cleanup_id="crash-retry",
        confirmation_inventory_seal=audit_value["inventory"]["seal_sha256"],
        hooks=hooks,
    )
    assert receipt["removed_inventory_seal_sha256"] == audit_value["inventory"][
        "seal_sha256"
    ]
    assert receipt["removed_file_count"] == 2
    assert receipt["cleanup_attempt_count"] == 2
    assert receipt["all_original_inventory_rows_committed"] is True
    assert len(receipt["progress_entry_seal_sha256s"]) == 2


@pytest.mark.parametrize(
    "name",
    [
        "not-a-manifest-object.deadbeef.incomplete",
        f"{'a' * 64}.deadbeef.incomplete",
        f"{_manifest_sha()}.DEADBEEF.incomplete",
        f"{_manifest_sha()}.deadbee.incomplete",
    ],
)
def test_audit_rejects_unknown_or_noncanonical_incomplete_name(
    tmp_path: Path, name: str
) -> None:
    layout = _layout(tmp_path)
    _seed_finished(layout)
    path = layout.blobs / name
    path.write_bytes(b"unknown")
    os.chmod(path, 0o600)
    with pytest.raises(supervisor.DownloadSupervisorError, match="incomplete"):
        cleanup.audit(layout, cleanup_id="bad-name", hooks=_hooks(layout))
    assert path.exists()


def test_cleanup_requires_exact_lease_no_unfinished_child_and_clean_process_audit(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    _seed_finished(layout)
    _incomplete(layout)
    descriptor = os.open(
        layout.evidence / supervisor._LEASE_NAME, os.O_RDWR | os.O_CREAT, 0o600
    )
    os.chmod(layout.evidence / supervisor._LEASE_NAME, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(supervisor.DownloadSupervisorError, match="exclusive lease"):
            cleanup.audit(layout, cleanup_id="locked", hooks=_hooks(layout))
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    with pytest.raises(supervisor.DownloadSupervisorError, match="conflict"):
        cleanup.audit(
            layout, cleanup_id="process-conflict", hooks=_hooks(layout, conflict=True)
        )

    with supervisor.JournalWriter(layout) as journal:
        journal.append(
            event="CHILD_STARTED",
            invocation_id="unfinished",
            timestamp_utc="2026-07-21T00:00:01.000000Z",
            monotonic_ns=2,
            payload={"pid": 59999, "profile": "primary-8", "workers": 8},
        )
    with pytest.raises(cleanup.StaleDownloadCleanupError, match="unfinished child"):
        cleanup.audit(layout, cleanup_id="unfinished", hooks=_hooks(layout))


def test_cleanup_source_has_only_exact_dirfd_unlink_and_no_broad_removal() -> None:
    tree = ast.parse(Path(cleanup.__file__).read_text(encoding="utf-8"))
    forbidden = {"glob", "rglob", "rmtree", "remove", "rmdir", "removedirs"}
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden
    ]
    unlinks = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "unlink"
    ]
    assert len(unlinks) == 1
    assert isinstance(unlinks[0].func.value, ast.Name)
    assert unlinks[0].func.value.id == "os"
    assert {item.arg for item in unlinks[0].keywords} == {"dir_fd"}
