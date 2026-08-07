"""Unit tests for the DeepSeek-V4 build-full resource supervisor.

All process/stat surfaces are injected.  No live network downloads.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import deepseek_v4_resource_supervisor as sup  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeProcess:
    def __init__(
        self,
        *,
        pid: int = 4242,
        exit_after_samples: int | None = None,
        exit_code: int = 0,
    ) -> None:
        self.pid = pid
        self._exit_after_samples = exit_after_samples
        self._exit_code = exit_code
        self._samples_seen = 0
        self._alive = True
        self.terminate_calls = 0
        self.kill_calls = 0

    def note_sample(self) -> None:
        self._samples_seen += 1
        if (
            self._exit_after_samples is not None
            and self._samples_seen >= self._exit_after_samples
        ):
            self._alive = False

    def poll(self) -> int | None:
        if self._alive:
            return None
        return self._exit_code

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._alive = False

    def kill(self) -> None:
        self.kill_calls += 1
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        self._alive = False
        return self._exit_code


class ScriptedProvider:
    """Yield predetermined samples; advances on each sample() call."""

    def __init__(self, samples: list[sup.ResourceSample]) -> None:
        if not samples:
            raise ValueError("ScriptedProvider needs at least one sample")
        self._samples = list(samples)
        self._index = 0
        self.calls: list[int | None] = []
        self.process: FakeProcess | None = None

    def sample(self, pid: int | None) -> sup.ResourceSample:
        self.calls.append(pid)
        if self.process is not None:
            self.process.note_sample()
        if self._index >= len(self._samples):
            sample = self._samples[-1]
        else:
            sample = self._samples[self._index]
            self._index += 1
        # Stamp pid for receipt fidelity when a child is live.
        if pid is not None and sample.pid is None:
            return sup.ResourceSample(
                monotonic_s=sample.monotonic_s,
                process_rss_bytes=sample.process_rss_bytes,
                cpu_percent=sample.cpu_percent,
                free_disk_bytes=sample.free_disk_bytes,
                swap_used_bytes=sample.swap_used_bytes,
                transfer_bytes=sample.transfer_bytes,
                pid=pid,
            )
        return sample


class FakeClock:
    def __init__(self, start: float = 0.0, step: float = 1.0) -> None:
        self._now = start
        self.step = step

    def __call__(self) -> float:
        return self._now

    def advance(self, delta: float | None = None) -> None:
        self._now += self.step if delta is None else delta


def _sample(
    *,
    t: float,
    rss: int = 100 * 1024**2,
    cpu: float = 40.0,
    free_disk: int = 100 * 1024**3,
    swap: int = 0,
    transfer: int = 0,
    pid: int | None = None,
) -> sup.ResourceSample:
    return sup.ResourceSample(
        monotonic_s=t,
        process_rss_bytes=rss,
        cpu_percent=cpu,
        free_disk_bytes=free_disk,
        swap_used_bytes=swap,
        transfer_bytes=transfer,
        pid=pid,
    )


def _config(tmp_path: Path, **overrides: Any) -> sup.SupervisorConfig:
    artifact = tmp_path / "artifact"
    workspace = tmp_path / "workspace"
    xet = tmp_path / "xet"
    artifact.mkdir()
    workspace.mkdir()
    xet.mkdir()
    kwargs: dict[str, Any] = {
        "artifact_dir": artifact,
        "workspace_root": workspace,
        "xet_root": xet,
        "protected_floor_bytes": 25 * 1024**3,  # invocation example, not a default
        "worker_ramp": (4, 8, 12),
        "rss_budget_bytes": sup.RSS_BUDGET_BYTES,
        "sample_interval_seconds": 0.1,
        "measure_window_seconds": 2.0,
        "min_throughput_gain": 0.05,
        "python_executable": "/usr/bin/python3",
        "gravity_cli": REPO_ROOT / "tools" / "condense" / "deepseek_v4_gravity.py",
        "dry_run": False,
        "terminate_grace_seconds": 0.0,
    }
    kwargs.update(overrides)
    return sup.SupervisorConfig(**kwargs)


# ---------------------------------------------------------------------------
# Config / argv
# ---------------------------------------------------------------------------


def test_default_worker_ramp_and_ceiling() -> None:
    assert sup.DEFAULT_WORKER_RAMP == (4, 8, 12, 16)
    assert sup.MAX_PARALLEL_WORKERS == 32
    assert sup.RSS_BUDGET_BYTES == 5 * 1024**3
    assert sup.normalize_worker_ramp("4,8,12,16") == (4, 8, 12, 16)


def test_worker_ramp_rejects_above_ceiling_and_non_ascending() -> None:
    with pytest.raises(sup.ResourceSupervisorError, match="ceiling"):
        sup.normalize_worker_ramp((4, 8, 40))
    with pytest.raises(sup.ResourceSupervisorError, match="ascending"):
        sup.normalize_worker_ramp((8, 4, 12))
    with pytest.raises(sup.ResourceSupervisorError, match="duplicates"):
        sup.normalize_worker_ramp((4, 4, 8))


def test_build_full_argv_uses_existing_cli_and_passthrough(tmp_path: Path) -> None:
    config = _config(tmp_path, range_bytes=4 * 1024**2)
    argv = sup.build_full_argv(config, parallel_workers=12)
    assert argv[0] == "/usr/bin/python3"
    assert argv[1].endswith("deepseek_v4_gravity.py")
    assert "build-full" in argv
    assert argv[argv.index("--parallel-workers") + 1] == "12"
    assert argv[argv.index("--protected-floor-bytes") + 1] == str(25 * 1024**3)
    assert argv[argv.index("--range-bytes") + 1] == str(4 * 1024**2)
    # Supervisor must not invent download APIs or alter range defaults silently.
    assert "--download" not in argv
    joined = " ".join(argv)
    assert "build-full" in joined


def test_build_full_argv_omits_range_when_unset(tmp_path: Path) -> None:
    config = _config(tmp_path, range_bytes=None)
    argv = sup.build_full_argv(config, parallel_workers=4)
    assert "--range-bytes" not in argv


def test_evaluate_sample_rss_disk_swap() -> None:
    policy = sup.SupervisorPolicy(
        rss_budget_bytes=5 * 1024**3,
        protected_floor_bytes=25 * 1024**3,
        allow_swap_growth=False,
        min_throughput_gain=0.05,
    )
    ok = _sample(t=1.0, rss=1 * 1024**3, free_disk=30 * 1024**3, swap=0)
    assert sup.evaluate_sample(ok, baseline_swap_used_bytes=0, policy=policy) == []

    rss = _sample(t=1.0, rss=6 * 1024**3, free_disk=30 * 1024**3, swap=0)
    assert "RSS_BUDGET_BREACH" in sup.evaluate_sample(
        rss, baseline_swap_used_bytes=0, policy=policy
    )

    disk = _sample(t=1.0, rss=1 * 1024**3, free_disk=10 * 1024**3, swap=0)
    assert "DISK_FLOOR_BREACH" in sup.evaluate_sample(
        disk, baseline_swap_used_bytes=0, policy=policy
    )

    swap = _sample(t=1.0, rss=1 * 1024**3, free_disk=30 * 1024**3, swap=100)
    assert "SWAP_GROWTH" in sup.evaluate_sample(
        swap, baseline_swap_used_bytes=0, policy=policy
    )


def test_parse_ps_and_swapusage() -> None:
    cpu, rss = sup.parse_ps_cpu_rss(" 12.5  1048576 ")
    assert cpu == 12.5
    assert rss == 1048576 * 1024
    used = sup.parse_swapusage("vm.swapusage: total = 2.00G used = 128.0M free = 1.87G")
    assert used == int(128.0 * 1024**2)


def test_material_throughput_gain() -> None:
    assert sup.material_throughput_gain(None, 100.0, min_gain=0.05)
    assert sup.material_throughput_gain(100.0, 110.0, min_gain=0.05)
    assert not sup.material_throughput_gain(100.0, 102.0, min_gain=0.05)


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def test_dry_run_receipt_without_child(tmp_path: Path) -> None:
    config = _config(tmp_path, dry_run=True, worker_ramp=(4, 8, 12, 16))
    launches: list[list[str]] = []

    def forbidden_launcher(argv: list[str], env: Any = None) -> FakeProcess:
        launches.append(list(argv))
        raise AssertionError("dry-run must not launch a child")

    receipt = sup.run_ramp(
        config,
        provider=ScriptedProvider([_sample(t=0.0)]),
        launcher=forbidden_launcher,  # type: ignore[arg-type]
    )
    assert launches == []
    assert receipt["schema"] == sup.RECEIPT_SCHEMA
    assert receipt["status"] == "DRY_RUN"
    assert receipt["recommended_workers"] == 4
    assert receipt["stop_reason"] == "DRY_RUN_NO_CHILD"
    assert len(receipt["candidates"]) == 4
    assert all(c["status"] == "PLANNED" for c in receipt["candidates"])
    assert receipt["policy"]["rss_enforcement"] == "sample_and_terminate_not_ulimit"
    assert receipt["policy"]["zero_cpu_idle_promised"] is False
    assert receipt["target"]["downloads_model_objects_directly"] is False
    assert receipt["policy"]["invocation_floor_example_bytes"] == 25 * 1024**3
    # Caller floor is preserved (25 GiB example used as the real flag value).
    assert receipt["policy"]["protected_floor_bytes"] == 25 * 1024**3

    out = tmp_path / "receipt.json"
    path = sup.write_receipt(out, receipt)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["recommended_workers"] == 4


# ---------------------------------------------------------------------------
# Live ramp with injected providers (no network)
# ---------------------------------------------------------------------------


def _window_sleeper(clock: FakeClock, window: float):
    """Advance the fake clock by a full measure window on each sleep.

    That yields exactly two samples per candidate: one at t0 and one after the
    window elapses (when the loop re-checks ``elapsed >= measure_window``).
    """

    def sleeper(_seconds: float) -> None:
        clock.advance(window)

    return sleeper


def test_ramp_stops_on_rss_budget_breach(tmp_path: Path) -> None:
    config = _config(tmp_path, worker_ramp=(4, 8), measure_window_seconds=10.0)
    # Preflight (pid=None) then one ok sample and an RSS breach before the window ends.
    samples = [
        _sample(t=0.0, transfer=0, free_disk=100 * 1024**3),  # preflight
        _sample(t=1.0, rss=1 * 1024**3, transfer=1_000_000, cpu=55.0),
        _sample(t=2.0, rss=6 * 1024**3, transfer=2_000_000, cpu=80.0),  # breach
    ]
    provider = ScriptedProvider(samples)
    process = FakeProcess(pid=99)
    provider.process = process
    launched: list[list[str]] = []

    def launcher(argv: list[str], env: Any = None) -> FakeProcess:
        launched.append(list(argv))
        return process

    clock = FakeClock(start=0.0)
    # Small step so the breach sample is seen before the measure window ends.
    sleeper = _window_sleeper(clock, 1.0)

    receipt = sup.run_ramp(
        config,
        provider=provider,
        launcher=launcher,  # type: ignore[arg-type]
        sleeper=sleeper,
        clock=clock,
    )
    assert len(launched) == 1
    assert launched[0][launched[0].index("--parallel-workers") + 1] == "4"
    assert receipt["status"] == "STOPPED"
    assert "RSS_BUDGET_BREACH" in receipt["stop_reason"]
    assert receipt["recommended_workers"] is None  # no prior safe step
    assert process.terminate_calls >= 1
    candidate = receipt["candidates"][0]
    assert candidate["status"] == "BUDGET_BREACH"
    assert any(s["process_rss_bytes"] > sup.RSS_BUDGET_BYTES for s in candidate["samples"])
    assert any(a["action"] == "FAIL_CLOSED" for a in candidate["actions"])


def test_ramp_stops_on_swap_growth_and_keeps_prior(tmp_path: Path) -> None:
    window = 2.0
    config = _config(
        tmp_path,
        worker_ramp=(4, 8, 12),
        measure_window_seconds=window,
        sample_interval_seconds=0.1,
        min_throughput_gain=0.01,
    )
    # Preflight swap=0; worker 4 measures cleanly; worker 8 grows swap on first sample.
    samples = [
        _sample(t=0.0, transfer=0, swap=0, free_disk=100 * 1024**3),  # preflight
        # candidate 4: start + end-of-window
        _sample(t=0.0, transfer=0, swap=0, cpu=30.0, free_disk=100 * 1024**3),
        _sample(t=2.0, transfer=3_000_000, swap=0, cpu=35.0, free_disk=100 * 1024**3),
        # candidate 8: swap growth immediately
        _sample(t=2.0, transfer=3_000_000, swap=50 * 1024**2, cpu=40.0, free_disk=100 * 1024**3),
    ]
    provider = ScriptedProvider(samples)
    processes: list[FakeProcess] = []

    def launcher(argv: list[str], env: Any = None) -> FakeProcess:
        proc = FakeProcess(pid=1000 + len(processes))
        processes.append(proc)
        provider.process = proc
        return proc

    clock = FakeClock(start=0.0)
    receipt = sup.run_ramp(
        config,
        provider=provider,
        launcher=launcher,  # type: ignore[arg-type]
        sleeper=_window_sleeper(clock, window),
        clock=clock,
    )
    assert receipt["status"] == "STOPPED"
    assert "SWAP_GROWTH" in receipt["stop_reason"]
    assert receipt["recommended_workers"] == 4
    assert len(receipt["candidates"]) == 2
    assert receipt["candidates"][0]["status"] == "MEASURED"
    assert receipt["candidates"][1]["status"] == "BUDGET_BREACH"


def test_ramp_stops_on_error_exit(tmp_path: Path) -> None:
    config = _config(tmp_path, worker_ramp=(4, 8), measure_window_seconds=30.0)
    samples = [
        _sample(t=0.0, free_disk=100 * 1024**3),  # preflight
        _sample(t=1.0, transfer=100, free_disk=100 * 1024**3),
    ]
    provider = ScriptedProvider(samples)
    process = FakeProcess(pid=7, exit_after_samples=1, exit_code=17)
    provider.process = process

    def launcher(argv: list[str], env: Any = None) -> FakeProcess:
        return process

    clock = FakeClock()
    receipt = sup.run_ramp(
        config,
        provider=provider,
        launcher=launcher,  # type: ignore[arg-type]
        sleeper=_window_sleeper(clock, 1.0),
        clock=clock,
    )
    assert receipt["status"] == "STOPPED"
    assert receipt["stop_reason"] == "ERROR_EXIT"
    assert receipt["candidates"][0]["exit_code"] == 17
    assert receipt["recommended_workers"] is None
    # The supervisor must not call ps on the child after poll() observes exit.
    assert provider.calls[-1] is None


def test_ramp_stops_when_throughput_gain_is_not_material(tmp_path: Path) -> None:
    window = 2.0
    config = _config(
        tmp_path,
        worker_ramp=(4, 8, 12),
        measure_window_seconds=window,
        min_throughput_gain=0.20,  # require 20% gain
    )
    # Preflight + two full windows with flat throughput (~1e6 B/s each).
    samples = [
        _sample(t=0.0, transfer=0, free_disk=100 * 1024**3),
        _sample(t=0.0, transfer=0, free_disk=100 * 1024**3, cpu=40.0),
        _sample(t=2.0, transfer=2_000_000, free_disk=100 * 1024**3, cpu=42.0),
        _sample(t=2.0, transfer=2_000_000, free_disk=100 * 1024**3, cpu=50.0),
        _sample(t=4.0, transfer=4_000_000, free_disk=100 * 1024**3, cpu=51.0),
    ]
    provider = ScriptedProvider(samples)
    processes: list[FakeProcess] = []

    def launcher(argv: list[str], env: Any = None) -> FakeProcess:
        proc = FakeProcess(pid=2000 + len(processes))
        processes.append(proc)
        provider.process = proc
        return proc

    clock = FakeClock(start=0.0)
    receipt = sup.run_ramp(
        config,
        provider=provider,
        launcher=launcher,  # type: ignore[arg-type]
        sleeper=_window_sleeper(clock, window),
        clock=clock,
    )
    assert receipt["status"] == "STOPPED"
    assert receipt["stop_reason"] == "NO_MATERIAL_THROUGHPUT_GAIN"
    assert receipt["recommended_workers"] == 4
    assert len(receipt["candidates"]) == 2
    # Observed utilization is surfaced, not promised-zero.
    assert receipt["candidates"][0]["observed_cpu_percent_mean"] is not None
    assert receipt["candidates"][0]["observed_cpu_percent_mean"] > 0
    assert receipt["policy"]["zero_cpu_idle_promised"] is False


def test_ramp_completes_with_gains_and_recommends_last(tmp_path: Path) -> None:
    window = 2.0
    config = _config(
        tmp_path,
        worker_ramp=(4, 8),
        measure_window_seconds=window,
        min_throughput_gain=0.05,
    )
    samples = [
        _sample(t=0.0, transfer=0, free_disk=100 * 1024**3),
        _sample(t=0.0, transfer=0, free_disk=100 * 1024**3, cpu=20.0),
        _sample(t=2.0, transfer=1_000_000, free_disk=100 * 1024**3, cpu=25.0),
        _sample(t=2.0, transfer=1_000_000, free_disk=100 * 1024**3, cpu=30.0),
        _sample(t=4.0, transfer=3_000_000, free_disk=100 * 1024**3, cpu=45.0),
    ]
    provider = ScriptedProvider(samples)
    processes: list[FakeProcess] = []

    def launcher(argv: list[str], env: Any = None) -> FakeProcess:
        proc = FakeProcess(pid=3000 + len(processes))
        processes.append(proc)
        provider.process = proc
        return proc

    clock = FakeClock()
    receipt = sup.run_ramp(
        config,
        provider=provider,
        launcher=launcher,  # type: ignore[arg-type]
        sleeper=_window_sleeper(clock, window),
        clock=clock,
    )
    assert receipt["status"] == "COMPLETED"
    assert receipt["recommended_workers"] == 8
    assert receipt["stop_reason"] == "RAMP_COMPLETE"
    assert len(receipt["candidates"]) == 2
    assert all(c["throughput_bytes_per_second"] is not None for c in receipt["candidates"])


def test_preflight_disk_floor_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path, protected_floor_bytes=50 * 1024**3, worker_ramp=(4,))
    provider = ScriptedProvider(
        [_sample(t=0.0, free_disk=10 * 1024**3)]  # below floor
    )

    def launcher(argv: list[str], env: Any = None) -> FakeProcess:
        raise AssertionError("must not launch when preflight fails")

    receipt = sup.run_ramp(
        config,
        provider=provider,
        launcher=launcher,  # type: ignore[arg-type]
    )
    assert receipt["status"] == "PREFLIGHT_FAILED"
    assert "DISK_FLOOR_BREACH" in receipt["stop_reason"]
    assert receipt["recommended_workers"] is None
    assert receipt["candidates"] == []


def test_cli_dry_run_main(tmp_path: Path) -> None:
    artifact = tmp_path / "a"
    workspace = tmp_path / "w"
    xet = tmp_path / "x"
    artifact.mkdir()
    workspace.mkdir()
    xet.mkdir()
    receipt_out = tmp_path / "out.json"
    code = sup.main(
        [
            "--artifact-dir",
            str(artifact),
            "--workspace-root",
            str(workspace),
            "--xet-root",
            str(xet),
            "--protected-floor-bytes",
            str(25 * 1024**3),
            "--receipt-out",
            str(receipt_out),
            "--worker-ramp",
            "4,8,12,16",
            "--dry-run",
        ]
    )
    assert code == 0
    body = json.loads(receipt_out.read_text(encoding="utf-8"))
    assert body["status"] == "DRY_RUN"
    assert body["recommended_workers"] == 4
    assert body["policy"]["rss_budget_bytes"] == 5 * 1024**3
    for candidate in body["candidates"]:
        assert "build-full" in candidate["argv"]
        assert "--parallel-workers" in candidate["argv"]


def test_directory_transfer_bytes_counts_regular_files(tmp_path: Path) -> None:
    root = tmp_path / "art"
    (root / "chunks").mkdir(parents=True)
    (root / "chunks" / "aa.bin").write_bytes(b"x" * 100)
    (root / "stream-journal.json").write_bytes(b"y" * 20)
    assert sup.directory_transfer_bytes(root) == 120
    assert sup.directory_transfer_bytes(tmp_path / "missing") == 0
