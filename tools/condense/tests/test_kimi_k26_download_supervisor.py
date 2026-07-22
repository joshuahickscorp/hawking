from __future__ import annotations

import ast
import copy
import fcntl
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

CONDENSE = Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import kimi_k26_download_supervisor as supervisor  # noqa: E402
import kimi_k26_release_cycle as phase1  # noqa: E402


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


def _plan(layout: phase1.SessionLayout, *, ramp: bool = True) -> dict[str, Any]:
    runtime = phase1.seal_document(
        {
            "schema": "hawking.test.fake_runtime.v1",
            "status": "PASS_FAKE_RUNTIME",
        }
    )
    profiles: list[dict[str, Any]] = [
        {
            "profile_id": "PRIMARY_8",
            "command_argv": _argv(layout, 8),
            "environment": _environment(layout, 8),
            "activation": "INITIAL_TARGET",
        }
    ]
    if ramp:
        profiles.append(
            {
                "profile_id": "CONDITIONAL_RESTART_16",
                "command_argv": _argv(layout, 16),
                "environment": _environment(layout, 16),
                "activation": "LIVE_SUPERVISOR_ONLY_AFTER_SUSTAINED_LOW_MEASURED_TRANSFER",
                "prior_transfer_process_must_be_fully_exited": True,
                "concurrent_with_primary_forbidden": True,
                "same_hub_xet_tmp_hf_home_required": True,
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


class FakeHooks:
    def __init__(self, plan: dict[str, Any], events: list[str] | None = None) -> None:
        self.plan = plan
        self.events = events if events is not None else []
        self.runtime_verifications = 0

    def build(self, _layout: phase1.SessionLayout, **_kwargs: object) -> dict[str, Any]:
        return copy.deepcopy(self.plan)

    def verify_plan(
        self,
        value: dict[str, Any],
        _layout: phase1.SessionLayout,
        **_kwargs: object,
    ) -> dict[str, Any]:
        phase1.verify_sealed_document(value, label="fake exact plan")
        if phase1.canonical_json(value) != phase1.canonical_json(self.plan):
            raise phase1.ReleaseCycleError("fake deterministic plan mismatch")
        return value

    def verify_runtime(self, value: dict[str, Any]) -> dict[str, Any]:
        phase1.verify_sealed_document(value, label="fake runtime")
        if value != self.plan["transfer_runtime"]:
            raise phase1.ReleaseCycleError("fake runtime mismatch")
        self.runtime_verifications += 1
        self.events.append("runtime-verified")
        return value

    def hooks(self) -> supervisor.Phase1Hooks:
        return supervisor.Phase1Hooks(
            build_plan=self.build,
            verify_plan=self.verify_plan,
            verify_runtime=self.verify_runtime,
        )


class FakeSampler:
    def __init__(self, samples: list[supervisor.ResourceSnapshot] | None = None) -> None:
        self.samples = list(samples or [])
        self.calls = 0

    def sample(self, _layout: phase1.SessionLayout) -> supervisor.ResourceSnapshot:
        self.calls += 1
        if self.samples:
            return self.samples.pop(0)
        return supervisor.ResourceSnapshot(
            free_disk_bytes=supervisor.PRESTART_FREE_DISK_BYTES + 10_000_000,
            session_allocated_bytes=4096,
        )


class FakeClock:
    def __init__(self) -> None:
        self.nanoseconds = 0
        self.sleeps: list[float] = []

    def utc_now(self) -> str:
        return f"2026-07-21T00:00:{self.nanoseconds / 1e9:09.6f}Z"

    def monotonic_ns(self) -> int:
        return self.nanoseconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.nanoseconds += int(seconds * 1_000_000_000)


class FakeNetwork:
    def __init__(self, increment: int = 100_000_000) -> None:
        self.counter = 1_000_000_000
        self.increment = increment
        self.captures = 0
        self.counter_calls = 0

    def capture_active_default(self) -> dict[str, Any]:
        self.captures += 1
        return phase1.seal_document(
            {
                "schema": "hawking.kimi_k26.download_supervisor.network_path.v1",
                "status": "ACTIVE_DEFAULT_ROUTE_CAPTURED",
                "interface": "en0",
                "gateway": "192.0.2.1",
                "media": "autoselect (10Gbase-T <full-duplex>)",
                "planned_10g_media_observed": True,
            }
        )

    def received_bytes(self, interface: str) -> int:
        assert interface == "en0"
        self.counter_calls += 1
        value = self.counter
        self.counter += self.increment
        return value


class SequenceNetwork(FakeNetwork):
    def __init__(self, values: list[int]) -> None:
        super().__init__(increment=0)
        self.values = list(values)

    def received_bytes(self, interface: str) -> int:
        assert interface == "en0"
        self.counter_calls += 1
        if not self.values:
            return 2_000_000_002
        return self.values.pop(0)


class FakeProcessAuditor:
    def __init__(
        self, events: list[str] | None = None, *, conflict: bool = False
    ) -> None:
        self.events = events if events is not None else []
        self.conflict = conflict
        self.calls = 0

    def audit(
        self, _layout: phase1.SessionLayout, _plan: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls += 1
        self.events.append("process-audited")
        if self.conflict:
            raise supervisor.DownloadSupervisorError(
                "existing manually launched Kimi downloader uses exact cache"
            )
        return phase1.seal_document(
            {
                "schema": "hawking.kimi_k26.download_supervisor.process_audit.v1",
                "status": (
                    "PASS_SNAPSHOT_NO_EXISTING_EXACT_SESSION_CACHE_DOWNLOADER_"
                    "BEST_EFFORT_WITH_RACE"
                ),
                "method": "FAKE_STRUCTURED_PROCESS_AUDIT",
                "conflict_count": 0,
            }
        )


class FakeProcess:
    def __init__(self, pid: int, *, exit_after_polls: int, stubborn: bool = False) -> None:
        self.pid = pid
        self.exit_after_polls = exit_after_polls
        self.stubborn = stubborn
        self.polls = 0
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        self.polls += 1
        if self.polls >= self.exit_after_polls:
            self.returncode = 0
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            if self.stubborn and self.kill_calls == 0:
                raise subprocess.TimeoutExpired("fake-hf", timeout)
            self.returncode = -15 if self.kill_calls == 0 else -9
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self.stubborn:
            self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


class FakePopenFactory:
    def __init__(
        self,
        processes: list[FakeProcess],
        events: list[str] | None = None,
    ) -> None:
        self.processes = list(processes)
        self.events = events if events is not None else []
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.returned: list[FakeProcess] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> FakeProcess:
        self.events.append("popen")
        current = os.umask(0o077)
        os.umask(current)
        assert current == 0o077
        if self.returned:
            assert self.returned[-1].poll() is not None
        process = self.processes.pop(0)
        self.returned.append(process)
        self.calls.append((list(argv), kwargs))
        os.write(kwargs["stdout"], b"fake stdout\n")
        os.write(kwargs["stderr"], b"fake stderr\n")
        return process


def _hooks(layout: phase1.SessionLayout, *, ramp: bool = True, events=None) -> FakeHooks:
    return FakeHooks(_plan(layout, ramp=ramp), events=events)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_preflight_and_status_start_no_process_or_network_and_write_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    hooks = _hooks(layout)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("process/network helper must not run")

    monkeypatch.setattr(supervisor.subprocess, "Popen", forbidden)
    monkeypatch.setattr(supervisor.subprocess, "run", forbidden)
    before = list(layout.evidence.iterdir())
    result = supervisor.preflight(
        layout, hooks=hooks.hooks(), sampler=FakeSampler()
    )
    idle = supervisor.status(layout)
    assert result["status"] == "PASS_READY_NO_LIVE_ACTION"
    assert result["process_started"] is False
    assert result["network_accessed"] is False
    assert result["filesystem_written"] is False
    assert idle["status"] == "IDLE_NO_INVOCATIONS"
    assert list(layout.evidence.iterdir()) == before


def test_preflight_rejects_resealed_plan_substitution_and_missing_tmp(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    hooks = _hooks(layout)
    tampered = copy.deepcopy(hooks.plan)
    tampered["command_argv"][-1] = "16"
    tampered = phase1.seal_document(tampered)
    with pytest.raises(phase1.ReleaseCycleError, match="deterministic plan"):
        supervisor.preflight(
            layout,
            supplied_plan=tampered,
            hooks=hooks.hooks(),
            sampler=FakeSampler(),
        )
    layout.tmp.rmdir()
    with pytest.raises(supervisor.DownloadSupervisorError, match="TMPDIR"):
        supervisor.preflight(
            layout, hooks=hooks.hooks(), sampler=FakeSampler()
        )


def test_preflight_enforces_exact_capacity_floor(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    hooks = _hooks(layout)
    sampler = FakeSampler(
        [
            supervisor.ResourceSnapshot(
                supervisor.PRESTART_FREE_DISK_BYTES - 1,
                0,
            )
        ]
    )
    with pytest.raises(
        supervisor.DownloadSupervisorError,
        match="remaining exact source/runtime/headroom",
    ):
        supervisor.preflight(layout, hooks=hooks.hooks(), sampler=sampler)
    assert supervisor.PROJECTED_CAPACITY_BYTES == 622_838_008_432
    assert (
        supervisor.PROJECTED_SOURCE_ALLOCATED_BYTES
        + supervisor.PROJECTED_RUNTIME_ALLOCATED_BYTES
        + supervisor.PROJECTED_HEADROOM_BYTES
        == supervisor.PRESTART_FREE_DISK_BYTES
    )


def test_resume_credits_more_than_initial_margin_of_existing_session_bytes(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    existing = 15_000_000_000  # greater than the observed ~14.2 GB pristine margin
    free = supervisor.PRESTART_FREE_DISK_BYTES - 1_000_000_000
    result = supervisor.preflight(
        layout,
        hooks=_hooks(layout).hooks(),
        sampler=FakeSampler([supervisor.ResourceSnapshot(free, existing)]),
    )
    capacity = result["capacity"]
    assert capacity["prestart_floor_pass"] is False
    assert capacity["prestart_floor_is_launch_gate"] is False
    assert capacity["capacity_mode"] == (
        "RESUME_CREDITS_EXISTING_CAPPED_SESSION_ALLOCATION"
    )
    assert capacity["remaining_projected_need_bytes"] == (
        supervisor.SESSION_ALLOCATION_CAP_BYTES
        - existing
        + supervisor.PROJECTED_HEADROOM_BYTES
    )
    assert capacity["remaining_projected_capacity_pass"] is True


def test_run_uses_only_exact_pinned_child_contract_and_seals_evidence(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    events: list[str] = []
    hooks = _hooks(layout, events=events)
    factory = FakePopenFactory([FakeProcess(41001, exit_after_polls=2)], events)
    clock = FakeClock()
    result = supervisor.run(
        layout,
        invocation_id="exact-run",
        hooks=hooks.hooks(),
        sampler=FakeSampler(),
        network_probe=FakeNetwork(),
        process_auditor=FakeProcessAuditor(events),
        clock=clock,
        popen_factory=factory,
    )
    assert result["status"] == "DOWNLOAD_COMMAND_EXITED_ZERO_SOURCE_VERIFICATION_REQUIRED"
    assert result["pid"] == 41001
    assert result["exit_code"] == 0
    assert result["final_workers"] == 8
    assert events[-3:] == ["process-audited", "runtime-verified", "popen"]
    assert len(factory.calls) == 1
    argv, kwargs = factory.calls[0]
    assert argv == _argv(layout, 8)
    assert kwargs["env"] == _environment(layout, 8)
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["shell"] is False
    assert kwargs["close_fds"] is True
    assert len(kwargs["pass_fds"]) == 1
    assert isinstance(kwargs["pass_fds"][0], int)
    assert kwargs["start_new_session"] is True
    assert kwargs["cwd"] == os.fspath(layout.session)
    assert "HF_TOKEN" not in kwargs["env"]
    assert "PATH" not in kwargs["env"]
    for path in layout.evidence.iterdir():
        metadata = path.lstat()
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_nlink == 1
    intent = _read_json(layout.evidence / "invocation.exact-run.json")
    started = _read_json(layout.evidence / "started.exact-run.primary-8.json")
    durable_status = _read_json(layout.evidence / "status.exact-run.json")
    phase1.verify_sealed_document(intent, label="intent")
    phase1.verify_sealed_document(started, label="started")
    phase1.verify_sealed_document(durable_status, label="status")
    assert started["pid"] == 41001
    assert started["argv_sha256"] == result["argv_sha256"]
    assert (layout.evidence / "stdout.exact-run.primary-8.log").read_text() == "fake stdout\n"
    assert (layout.evidence / "stderr.exact-run.primary-8.log").read_text() == "fake stderr\n"
    assert all(value <= 0.25 for value in clock.sleeps)


def test_resource_guard_terminates_then_kills_and_preserves_resume_cache(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    marker = layout.hub / "resumable.partial"
    marker.write_bytes(b"keep")
    os.chmod(marker, 0o600)
    process = FakeProcess(41002, exit_after_polls=999, stubborn=True)
    factory = FakePopenFactory([process])
    samples = [
        supervisor.ResourceSnapshot(supervisor.PRESTART_FREE_DISK_BYTES + 1, 4096),
        supervisor.ResourceSnapshot(supervisor.PRESTART_FREE_DISK_BYTES + 1, 4096),
        supervisor.ResourceSnapshot(supervisor.RUNTIME_FREE_DISK_FLOOR_BYTES - 1, 4096),
    ]
    result = supervisor.run(
        layout,
        invocation_id="guard-run",
        hooks=_hooks(layout).hooks(),
        sampler=FakeSampler(samples),
        network_probe=FakeNetwork(),
        process_auditor=FakeProcessAuditor(),
        clock=FakeClock(),
        popen_factory=factory,
    )
    assert result["status"] == "RESOURCE_GUARD_TERMINATED_RESUMABLE_CACHE_PRESERVED"
    assert result["resource_violation"] == "RUNTIME_FREE_DISK_FLOOR_VIOLATED"
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.returncode == -9
    assert marker.read_bytes() == b"keep"


def test_resource_guard_detects_external_consumption_of_remaining_capacity(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    process = FakeProcess(41010, exit_after_polls=999, stubborn=True)
    existing = 1_000_000_000
    remaining_need = (
        supervisor.SESSION_ALLOCATION_CAP_BYTES
        - existing
        + supervisor.PROJECTED_HEADROOM_BYTES
    )
    samples = [
        supervisor.ResourceSnapshot(supervisor.PRESTART_FREE_DISK_BYTES + 1, 0),
        supervisor.ResourceSnapshot(supervisor.PRESTART_FREE_DISK_BYTES + 1, 0),
        supervisor.ResourceSnapshot(remaining_need - 1, existing),
    ]
    result = supervisor.run(
        layout,
        invocation_id="external-capacity-run",
        hooks=_hooks(layout).hooks(),
        sampler=FakeSampler(samples),
        network_probe=FakeNetwork(),
        process_auditor=FakeProcessAuditor(),
        clock=FakeClock(),
        popen_factory=FakePopenFactory([process]),
    )
    assert result["resource_violation"] == "REMAINING_PROJECTED_CAPACITY_VIOLATED"
    assert process.returncode == -9


def test_post_start_sampler_fault_stops_child_and_closes_durable_lifecycle(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)

    class RaisingSampler(FakeSampler):
        def sample(self, target: phase1.SessionLayout) -> supervisor.ResourceSnapshot:
            if self.calls == 2:
                self.calls += 1
                raise RuntimeError("injected sampler fault after child start")
            return super().sample(target)

    process = FakeProcess(41009, exit_after_polls=999, stubborn=True)
    with pytest.raises(RuntimeError, match="injected sampler fault"):
        supervisor.run(
            layout,
            invocation_id="sampler-fault-run",
            hooks=_hooks(layout).hooks(),
            sampler=RaisingSampler(),
            network_probe=FakeNetwork(),
            process_auditor=FakeProcessAuditor(),
            clock=FakeClock(),
            popen_factory=FakePopenFactory([process]),
        )
    assert process.returncode == -9
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    replay = supervisor.status(layout)
    assert replay["unfinished_children"] == []
    durable = _read_json(layout.evidence / "status.sampler-fault-run.json")
    phase1.verify_sealed_document(durable, label="fault status")
    assert durable["status"] == "SUPERVISOR_FAULT_CHILD_STOPPED_FAIL_CLOSED"
    assert durable["exit_code"] == -9


def test_exclusive_lease_refuses_concurrent_run_before_popen(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    lease = layout.evidence / supervisor._LEASE_NAME
    descriptor = os.open(lease, os.O_RDWR | os.O_CREAT, 0o600)
    os.chmod(lease, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    factory = FakePopenFactory([FakeProcess(41003, exit_after_polls=1)])
    try:
        with pytest.raises(supervisor.DownloadSupervisorError, match="exclusive lease"):
            supervisor.run(
                layout,
                invocation_id="lease-run",
                hooks=_hooks(layout).hooks(),
                sampler=FakeSampler(),
                network_probe=FakeNetwork(),
                process_auditor=FakeProcessAuditor(),
                clock=FakeClock(),
                popen_factory=factory,
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert factory.calls == []


def test_sustained_low_measured_transfer_serially_restarts_at_16(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    primary = FakeProcess(41004, exit_after_polls=999)
    ramp = FakeProcess(41005, exit_after_polls=2)
    factory = FakePopenFactory([primary, ramp])
    policy = supervisor.SupervisorPolicy(
        monitor_interval_seconds=0.25,
        network_sample_interval_seconds=0.25,
        ramp_warmup_seconds=0,
        ramp_measurement_seconds=0.5,
        ramp_target_bytes_per_second=1_000_000_000,
        ramp_min_counter_samples=3,
        termination_grace_seconds=1,
    )
    result = supervisor.run(
        layout,
        invocation_id="ramp-run",
        hooks=_hooks(layout).hooks(),
        sampler=FakeSampler(),
        network_probe=FakeNetwork(increment=10_000_000),
        process_auditor=FakeProcessAuditor(),
        clock=FakeClock(),
        popen_factory=factory,
        policy=policy,
    )
    assert result["status"] == "DOWNLOAD_COMMAND_EXITED_ZERO_SOURCE_VERIFICATION_REQUIRED"
    assert result["ramp_evaluated"] is True
    assert result["ramp_performed"] is True
    assert result["final_workers"] == 16
    assert primary.terminate_calls == 1
    assert primary.poll() is not None
    assert len(factory.calls) == 2
    assert factory.calls[0][0] == _argv(layout, 8)
    assert factory.calls[1][0] == _argv(layout, 16)
    assert factory.calls[0][1]["env"]["HF_HUB_CACHE"] == factory.calls[1][1]["env"][
        "HF_HUB_CACHE"
    ]
    assert factory.calls[0][1]["env"]["HF_XET_CACHE"] == factory.calls[1][1]["env"][
        "HF_XET_CACHE"
    ]


def test_ramp_uses_only_post_warmup_measurement_window(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    primary = FakeProcess(41011, exit_after_polls=999)
    ramp = FakeProcess(41012, exit_after_polls=1)
    factory = FakePopenFactory([primary, ramp])
    # Warmup receives 2 GB rapidly.  The post-warmup 0.5-second window then
    # receives only two bytes.  Whole-invocation averaging would incorrectly
    # suppress the ramp; the bounded post-warmup window must authorize it.
    network = SequenceNetwork(
        [0, 1_000_000_000, 2_000_000_000, 2_000_000_001, 2_000_000_002]
    )
    policy = supervisor.SupervisorPolicy(
        monitor_interval_seconds=0.25,
        network_sample_interval_seconds=0.25,
        ramp_warmup_seconds=0.5,
        ramp_measurement_seconds=0.5,
        ramp_target_bytes_per_second=100_000_000,
        ramp_min_counter_samples=3,
        termination_grace_seconds=1,
    )
    result = supervisor.run(
        layout,
        invocation_id="post-warmup-run",
        hooks=_hooks(layout).hooks(),
        sampler=FakeSampler(),
        network_probe=network,
        process_auditor=FakeProcessAuditor(),
        clock=FakeClock(),
        popen_factory=factory,
        policy=policy,
    )
    assert result["ramp_performed"] is True
    assert result["final_workers"] == 16
    journal = (layout.evidence / supervisor._JOURNAL_NAME).read_bytes()
    entries = supervisor._verify_journal_bytes(journal)
    decision = next(entry for entry in entries if entry["event"] == "RAMP_16_EVALUATED")
    assert decision["payload"]["post_warmup_received_bytes"] == 2
    assert decision["payload"]["actual_post_warmup_measurement_seconds"] == 0.5


def test_fast_measured_transfer_does_not_ramp(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    factory = FakePopenFactory([FakeProcess(41006, exit_after_polls=5)])
    policy = supervisor.SupervisorPolicy(
        monitor_interval_seconds=0.25,
        network_sample_interval_seconds=0.25,
        ramp_warmup_seconds=0,
        ramp_measurement_seconds=0.5,
        ramp_target_bytes_per_second=1,
        ramp_min_counter_samples=3,
        termination_grace_seconds=1,
    )
    result = supervisor.run(
        layout,
        invocation_id="fast-run",
        hooks=_hooks(layout).hooks(),
        sampler=FakeSampler(),
        network_probe=FakeNetwork(increment=500_000_000),
        process_auditor=FakeProcessAuditor(),
        clock=FakeClock(),
        popen_factory=factory,
        policy=policy,
    )
    assert result["ramp_evaluated"] is True
    assert result["ramp_performed"] is False
    assert result["final_workers"] == 8
    assert len(factory.calls) == 1


def test_missing_phase1_ramp_profile_never_derives_an_unplanned_command(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    factory = FakePopenFactory([FakeProcess(41007, exit_after_polls=5)])
    policy = supervisor.SupervisorPolicy(
        monitor_interval_seconds=0.25,
        network_sample_interval_seconds=0.25,
        ramp_warmup_seconds=0,
        ramp_measurement_seconds=0.5,
        ramp_target_bytes_per_second=1_000_000_000,
        ramp_min_counter_samples=3,
        termination_grace_seconds=1,
    )
    result = supervisor.run(
        layout,
        invocation_id="no-authority-run",
        hooks=_hooks(layout, ramp=False).hooks(),
        sampler=FakeSampler(),
        network_probe=FakeNetwork(increment=1),
        process_auditor=FakeProcessAuditor(),
        clock=FakeClock(),
        popen_factory=factory,
        policy=policy,
    )
    assert result["ramp_evaluated"] is True
    assert result["ramp_performed"] is False
    assert len(factory.calls) == 1
    assert factory.calls[0][0] == _argv(layout, 8)


def test_status_replays_hash_chain_and_rejects_tamper(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    supervisor.run(
        layout,
        invocation_id="status-run",
        hooks=_hooks(layout).hooks(),
        sampler=FakeSampler(),
        network_probe=FakeNetwork(),
        process_auditor=FakeProcessAuditor(),
        clock=FakeClock(),
        popen_factory=FakePopenFactory([FakeProcess(41008, exit_after_polls=1)]),
    )
    result = supervisor.status(layout)
    assert result["status"] == "DURABLE_STATE_VERIFIED"
    assert result["journal_entries"] >= 5
    assert result["unfinished_children"] == []
    journal = layout.evidence / supervisor._JOURNAL_NAME
    raw = bytearray(journal.read_bytes())
    raw[10] ^= 1
    journal.write_bytes(raw)
    os.chmod(journal, 0o600)
    with pytest.raises((supervisor.DownloadSupervisorError, phase1.ReleaseCycleError)):
        supervisor.status(layout)


def test_journal_refuses_append_before_crossing_replay_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    monkeypatch.setattr(supervisor, "_MAX_JOURNAL_BYTES", 64)
    with supervisor.JournalWriter(layout) as journal:
        with pytest.raises(supervisor.DownloadSupervisorError, match="cross the replay"):
            journal.append(
                event="OVERSIZED",
                invocation_id="bound-test",
                timestamp_utc="2026-07-21T00:00:00.000000Z",
                monotonic_ns=0,
                payload={"large": "x" * 100},
            )
    assert (layout.evidence / supervisor._JOURNAL_NAME).read_bytes() == b""


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--parent", "/tmp/other"),
        ("--manifest", "/tmp/manifest.json"),
        ("--mop-root", "/tmp/mop"),
        ("--shared-xet", "/tmp/xet"),
    ],
)
def test_production_cli_rejects_authority_redefinition(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        supervisor._parser().parse_args(
            [
                "preflight",
                "--session",
                os.fspath(phase1.SESSION_PARENT / "fixed"),
                flag,
                value,
            ]
        )


def test_preexec_process_conflict_fails_before_runtime_or_popen(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    events: list[str] = []
    hooks = _hooks(layout, events=events)
    factory = FakePopenFactory([FakeProcess(41013, exit_after_polls=1)], events)
    with pytest.raises(supervisor.DownloadSupervisorError, match="manually launched"):
        supervisor.run(
            layout,
            invocation_id="manual-conflict-run",
            hooks=hooks.hooks(),
            sampler=FakeSampler(),
            network_probe=FakeNetwork(),
            process_auditor=FakeProcessAuditor(events, conflict=True),
            clock=FakeClock(),
            popen_factory=factory,
        )
    assert "process-audited" in events
    assert "popen" not in events
    assert factory.calls == []
    durable = _read_json(layout.evidence / "status.manual-conflict-run.json")
    assert durable["pid"] is None
    assert "manually launched" in durable["fault"]


def test_native_auditor_matches_exact_structured_tokens_not_substrings(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    auditor = object.__new__(supervisor.DarwinExactProcessAuditor)
    exact = _argv(layout, 8)
    assert auditor._uses_exact_cache(exact, _environment(layout, 8), layout) is True
    lookalike = list(exact)
    lookalike[lookalike.index("--cache-dir") + 1] = os.fspath(layout.hub) + "-other"
    unrelated_environment = dict(_environment(layout, 8))
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "TMPDIR"):
        unrelated_environment[key] += "-other"
    assert auditor._uses_exact_cache(lookalike, unrelated_environment, layout) is False


def test_module_has_no_source_release_primitives() -> None:
    source_path = Path(supervisor.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_attributes = {"unlink", "rename", "replace", "rmdir", "removedirs"}
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden_attributes
    ]
    assert calls == []
