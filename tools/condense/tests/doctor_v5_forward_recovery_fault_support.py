"""Isolated crash/fault helpers for Doctor V5 forward-recovery tests.

This module is intentionally production-agnostic.  It never imports, stages,
applies, recovers, or rolls back the Doctor campaign.  Tests pass an isolated
fixture and an already imported recovery module explicitly.

The helpers cover four jobs:

* byte-exact snapshots of protected files and complete result/runtime trees;
* deterministic one-shot soft faults and real ``os._exit(137)`` crashes;
* boundary wrappers for the module fault hook, atomic writes, ``os.replace``,
  launchctl calls, and detached-start calls; and
* restart/idempotence and fixture-only path-tampering assertions.

All destructive path helpers require an explicit fixture root and refuse to
operate outside it.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Pattern, Sequence, TypeVar
from unittest import mock


SNAPSHOT_SCHEMA = "hawking.doctor_v5_forward_recovery_fault_snapshot.v1"

# The locks and transaction workspace are intentionally excluded.  Lock inode
# metadata and WAL files are expected to change while the protected live
# surface is being checked.
DEFAULT_LIVE_FILE_FIELDS = (
    "plan",
    "state",
    "campaign",
    "control",
    "registry",
    "pid_file",
    "launch_agent",
    "active_marker",
    "overlay",
    "predecessor_journal",
    "canonical_reentry_packet",
    "gc_authority",
    "accelerated_queue",
    "accelerated_autoresume",
)
DEFAULT_LIVE_TREE_FIELDS = ("runtime_specs", "results")
DEFAULT_TRANSACTION_FILE_FIELDS = ("packet", "journal")
DEFAULT_TRANSACTION_TREE_FIELDS = ("stage_root",)


class SnapshotError(RuntimeError):
    """A test surface could not be captured without following unsafe paths."""


class SnapshotRaceError(SnapshotError):
    """A file or directory changed while its snapshot was being captured."""


class InjectedFault(RuntimeError):
    """Deterministic catchable fault used by fixture-only tests."""


def _identity(row: os.stat_result) -> tuple[int, int, int, int, int]:
    return row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns


def _stable_digest(path: Path) -> tuple[str, int]:
    """Hash a real regular file without following a final-component symlink."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SnapshotError(f"cannot open snapshot file {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise SnapshotError(f"snapshot path is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(fd)
        if _identity(before) != _identity(after):
            raise SnapshotRaceError(f"snapshot file changed while reading: {path}")
        return digest.hexdigest(), size
    finally:
        os.close(fd)


def _node_row(path: Path) -> dict[str, Any]:
    """Capture a single lexical node without following a symlink."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {"kind": "absent"}
    except OSError as exc:
        raise SnapshotError(f"cannot inspect snapshot path {path}: {exc}") from exc

    mode = stat.S_IMODE(info.st_mode)
    if stat.S_ISLNK(info.st_mode):
        return {"kind": "symlink", "mode": mode, "target": os.readlink(path)}
    if stat.S_ISREG(info.st_mode):
        sha256, size = _stable_digest(path)
        return {"kind": "file", "mode": mode, "bytes": size, "sha256": sha256}
    if stat.S_ISDIR(info.st_mode):
        return {"kind": "directory", "mode": mode}
    return {
        "kind": "special",
        "mode": mode,
        "file_type": stat.S_IFMT(info.st_mode),
        "device": info.st_rdev,
    }


def snapshot_path(path: Path | str) -> dict[str, Any]:
    """Return a JSON-serializable snapshot of one lexical path."""
    return _node_row(Path(path))


def snapshot_tree(root: Path | str) -> dict[str, Any]:
    """Snapshot every node below ``root`` without traversing symlinks.

    Directory mtimes are used only as race detectors and are not placed in the
    result, so a transaction may replace a directory and still compare equal if
    its complete logical contents and permission modes are identical.
    """
    root = Path(root)
    root_row = _node_row(root)
    if root_row["kind"] != "directory":
        return {"root": root_row, "entries": {}}

    entries: dict[str, dict[str, Any]] = {}

    def visit(directory: Path, relative: Path) -> None:
        try:
            before = directory.lstat()
            children = sorted(os.scandir(directory), key=lambda row: row.name)
        except OSError as exc:
            raise SnapshotError(f"cannot enumerate snapshot directory {directory}: {exc}") from exc
        for child in children:
            child_path = directory / child.name
            child_relative = (relative / child.name).as_posix()
            row = _node_row(child_path)
            entries[child_relative] = row
            if row["kind"] == "directory":
                visit(child_path, relative / child.name)
        try:
            after = directory.lstat()
        except OSError as exc:
            raise SnapshotRaceError(f"snapshot directory disappeared: {directory}") from exc
        if _identity(before) != _identity(after):
            raise SnapshotRaceError(f"snapshot directory changed while reading: {directory}")

    visit(root, Path())
    return {"root": root_row, "entries": entries}


def snapshot_live_surface(
    paths: Any,
    *,
    file_fields: Iterable[str] = DEFAULT_LIVE_FILE_FIELDS,
    tree_fields: Iterable[str] = DEFAULT_LIVE_TREE_FIELDS,
    extra_files: Mapping[str, Path | str] | None = None,
    extra_trees: Mapping[str, Path | str] | None = None,
    include_transaction: bool = False,
) -> dict[str, Any]:
    """Capture the full protected live surface of an isolated ``Paths`` fixture."""
    selected_files = list(file_fields)
    selected_trees = list(tree_fields)
    if include_transaction:
        selected_files.extend(DEFAULT_TRANSACTION_FILE_FIELDS)
        selected_trees.extend(DEFAULT_TRANSACTION_TREE_FIELDS)

    files: dict[str, dict[str, Any]] = {}
    trees: dict[str, dict[str, Any]] = {}
    for field_name in selected_files:
        if not hasattr(paths, field_name):
            raise SnapshotError(f"fixture Paths has no file field: {field_name}")
        files[field_name] = snapshot_path(getattr(paths, field_name))
    for field_name in selected_trees:
        if not hasattr(paths, field_name):
            raise SnapshotError(f"fixture Paths has no tree field: {field_name}")
        trees[field_name] = snapshot_tree(getattr(paths, field_name))
    for name, path in (extra_files or {}).items():
        if name in files:
            raise SnapshotError(f"duplicate snapshot file role: {name}")
        files[name] = snapshot_path(path)
    for name, path in (extra_trees or {}).items():
        if name in trees:
            raise SnapshotError(f"duplicate snapshot tree role: {name}")
        trees[name] = snapshot_tree(path)
    return {"schema": SNAPSHOT_SCHEMA, "files": files, "trees": trees}


def _snapshot_lines(value: Mapping[str, Any]) -> list[str]:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").splitlines(True)


def diff_snapshots(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> str:
    """Return a unified logical-content diff for two surface snapshots."""
    return "".join(difflib.unified_diff(
        _snapshot_lines(expected),
        _snapshot_lines(observed),
        fromfile="expected-surface",
        tofile="observed-surface",
    ))


def assert_snapshot_equal(
    expected: Mapping[str, Any], observed: Mapping[str, Any], *, label: str = "surface"
) -> None:
    """Raise an informative assertion if two captured surfaces differ."""
    if expected != observed:
        raise AssertionError(f"{label} differs:\n{diff_snapshots(expected, observed)}")


def assert_live_surface_equal(
    paths: Any, expected: Mapping[str, Any], *, label: str = "live surface", **snapshot_args: Any
) -> dict[str, Any]:
    observed = snapshot_live_surface(paths, **snapshot_args)
    assert_snapshot_equal(expected, observed, label=label)
    return observed


FaultMatcher = int | str | Pattern[str] | Callable[[str, int], bool]


@dataclass
class FaultInjector:
    """A one-shot boundary injector.

    Integer targets are one-based boundary occurrence numbers.  Strings match
    exact labels; compiled regexes use ``search``; callables receive
    ``(label, occurrence)``.  ``mode='hard'`` performs an uncatchable-by-Python
    ``os._exit(exit_code)``; ``mode='sigkill'`` sends real SIGKILL.  Both modes
    must be used only in a child process.
    """

    target: FaultMatcher
    mode: str = "soft"
    exit_code: int = 137
    exception_factory: Callable[[str], BaseException] = InjectedFault
    events: list[str] = field(default_factory=list)
    fired: bool = False
    fired_label: str | None = None

    def _matches(self, label: str, occurrence: int) -> bool:
        if isinstance(self.target, bool):
            return False
        if isinstance(self.target, int):
            return occurrence == self.target
        if isinstance(self.target, str):
            return label == self.target
        if isinstance(self.target, re.Pattern):
            return self.target.search(label) is not None
        return bool(self.target(label, occurrence))

    def __call__(self, label: str) -> None:
        occurrence = len(self.events) + 1
        self.events.append(label)
        if self.fired or not self._matches(label, occurrence):
            return
        self.fired = True
        self.fired_label = label
        if self.mode == "hard":
            os._exit(self.exit_code)
        if self.mode == "sigkill":
            os.kill(os.getpid(), signal.SIGKILL)
            raise AssertionError("SIGKILL unexpectedly returned")
        if self.mode != "soft":
            raise ValueError(f"unknown fault mode: {self.mode}")
        raise self.exception_factory(label)

    def assert_fired(self) -> None:
        if not self.fired:
            raise AssertionError(
                f"fault target {self.target!r} was not reached; observed={self.events!r}"
            )


@dataclass
class BoundaryRecorder:
    """Record production and wrapper cutpoint labels without injecting."""

    events: list[str] = field(default_factory=list)

    def __call__(self, label: str) -> None:
        self.events.append(label)


def _basename(value: Any) -> str:
    try:
        return Path(os.fspath(value)).name or "."
    except TypeError:
        return repr(value)


class MutationBoundaryHarness:
    """Patch mutation boundaries on an imported recovery module.

    The harness is designed for isolated fixture children.  By default it
    stubs launchctl and detached start, so entering the harness cannot reload a
    real LaunchAgent or launch a real campaign owner.  Other subprocess calls
    retain their original behavior unless ``subprocess_delegate`` is supplied.
    """

    def __init__(
        self,
        module: Any,
        hook: Callable[[str], None],
        *,
        patch_module_fault: bool = True,
        patch_replace: bool = True,
        patch_atomic: bool = True,
        patch_launchctl: bool = True,
        detached_pid: int | None = 424242,
        launchctl_delegate: Callable[..., Any] | None = None,
        subprocess_delegate: Callable[..., Any] | None = None,
    ) -> None:
        self.module = module
        self.hook = hook
        self.patch_module_fault = patch_module_fault
        self.patch_replace = patch_replace
        self.patch_atomic = patch_atomic
        self.patch_launchctl = patch_launchctl
        self.detached_pid = detached_pid
        self.launchctl_delegate = launchctl_delegate
        self.subprocess_delegate = subprocess_delegate
        self.events: list[str] = []
        self._stack: ExitStack | None = None
        self._replace_count = 0
        self._atomic_count = 0
        self._launchctl_count = 0
        self._start_count = 0

    def _boundary(self, label: str) -> None:
        self.events.append(label)
        self.hook(label)

    def __enter__(self) -> "MutationBoundaryHarness":
        stack = ExitStack()
        self._stack = stack
        module = self.module

        # Save every delegate before patching.  ``module.os`` and
        # ``module.subprocess`` are process-global module objects, which is why
        # this harness belongs only inside a short-lived fixture child/context.
        original_replace = module.os.replace
        original_atomic = getattr(module, "_atomic_bytes", None)
        original_run = module.subprocess.run
        original_start = getattr(module, "_start_detached", None)

        if self.patch_module_fault and hasattr(module, "_fault"):
            production_hook = getattr(module, "_fault")

            def fault_hook(label: str) -> None:
                rendered = f"module:{label}"
                self.events.append(rendered)
                self.hook(rendered)
                # A production no-op remains a no-op.  If a test installed a
                # secondary recorder before this harness, retain it.
                if production_hook is not None:
                    production_hook(label)

            stack.enter_context(mock.patch.object(module, "_fault", fault_hook))

        if self.patch_replace:

            def replace(source: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
                self._replace_count += 1
                operation = (
                    f"{self._replace_count:04d}:"
                    f"{_basename(source)}->{_basename(target)}"
                )
                self._boundary(f"before:os.replace:{operation}")
                result = original_replace(source, target, *args, **kwargs)
                self._boundary(f"after:os.replace:{operation}")
                return result

            stack.enter_context(mock.patch.object(module.os, "replace", replace))

        if self.patch_atomic and original_atomic is not None:

            def atomic(path: Any, raw: bytes, *args: Any, **kwargs: Any) -> Any:
                self._atomic_count += 1
                operation = f"{self._atomic_count:04d}:{_basename(path)}"
                self._boundary(f"before:atomic-write:{operation}")
                result = original_atomic(path, raw, *args, **kwargs)
                self._boundary(f"after:atomic-write:{operation}")
                return result

            stack.enter_context(mock.patch.object(module, "_atomic_bytes", atomic))

        if self.patch_launchctl:

            def run(command: Any, *args: Any, **kwargs: Any) -> Any:
                argv = list(command) if isinstance(command, (list, tuple)) else [command]
                if argv and os.fspath(argv[0]) == "launchctl":
                    self._launchctl_count += 1
                    verb = os.fspath(argv[1]) if len(argv) > 1 else "unknown"
                    operation = f"{self._launchctl_count:04d}:{verb}"
                    self._boundary(f"before:launchctl:{operation}")
                    if self.launchctl_delegate is None:
                        result = subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
                    else:
                        result = self.launchctl_delegate(command, *args, **kwargs)
                    self._boundary(f"after:launchctl:{operation}")
                    return result
                if self.subprocess_delegate is not None:
                    return self.subprocess_delegate(command, *args, **kwargs)
                return original_run(command, *args, **kwargs)

            stack.enter_context(mock.patch.object(module.subprocess, "run", run))

        if self.detached_pid is not None and original_start is not None:

            def start(*args: Any, **kwargs: Any) -> int:
                self._start_count += 1
                operation = f"{self._start_count:04d}"
                self._boundary(f"before:detached-start:{operation}")
                pid = self.detached_pid
                self._boundary(f"after:detached-start:{operation}")
                return pid

            stack.enter_context(mock.patch.object(module, "_start_detached", start))

        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        assert self._stack is not None
        return self._stack.__exit__(exc_type, exc, tb)


def discover_boundaries(
    module: Any,
    mutate: Callable[[], Any],
    *,
    harness_kwargs: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Run one successful fixture mutation and return its ordered cutpoints."""
    recorder = BoundaryRecorder()
    with MutationBoundaryHarness(module, recorder, **dict(harness_kwargs or {})) as harness:
        mutate()
    if tuple(recorder.events) != tuple(harness.events):
        raise AssertionError("boundary recorder and harness event streams diverged")
    return tuple(harness.events)


@dataclass(frozen=True)
class ChildOutcome:
    pid: int
    raw_status: int
    exit_code: int | None
    signal: int | None
    elapsed_seconds: float
    diagnostic: str

    @property
    def exited(self) -> bool:
        return self.exit_code is not None

    @property
    def was_signaled(self) -> bool:
        return self.signal is not None


def run_forked(action: Callable[[], Any], *, timeout: float = 30.0) -> ChildOutcome:
    """Run ``action`` in a real child and return its wait status.

    A diagnostic pipe reports ordinary Python exceptions.  ``os._exit`` and
    SIGKILL intentionally bypass that path, accurately modeling abrupt process
    death and lock release.
    """
    if not hasattr(os, "fork"):
        raise RuntimeError("run_forked requires a POSIX fork implementation")
    read_fd, write_fd = os.pipe()
    started = time.monotonic()
    pid = os.fork()
    if pid == 0:  # pragma: no branch - executed only in the child
        os.close(read_fd)
        try:
            action()
        except BaseException:
            diagnostic = traceback.format_exc().encode("utf-8", errors="replace")
            try:
                # Stay comfortably below the smallest common POSIX pipe
                # capacity because the parent waits before draining this pipe.
                os.write(write_fd, diagnostic[-8 * 1024 :])
            finally:
                os.close(write_fd)
            os._exit(120)
        os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    raw_status: int | None = None
    try:
        while raw_status is None:
            observed_pid, observed_status = os.waitpid(pid, os.WNOHANG)
            if observed_pid == pid:
                raw_status = observed_status
                break
            if time.monotonic() - started > timeout:
                os.kill(pid, signal.SIGKILL)
                _, raw_status = os.waitpid(pid, 0)
                raise TimeoutError(f"fixture child {pid} exceeded {timeout:.3f}s")
            time.sleep(0.005)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(read_fd, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(read_fd)
    assert raw_status is not None
    exit_code = os.waitstatus_to_exitcode(raw_status)
    return ChildOutcome(
        pid=pid,
        raw_status=raw_status,
        exit_code=exit_code if exit_code >= 0 else None,
        signal=-exit_code if exit_code < 0 else None,
        elapsed_seconds=time.monotonic() - started,
        diagnostic=b"".join(chunks).decode("utf-8", errors="replace"),
    )


def run_subprocess(
    argv: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    """Run a separate-process fixture driver with captured diagnostics."""
    return subprocess.run(
        [os.fspath(value) for value in argv],
        cwd=os.fspath(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def assert_hard_exit(outcome: ChildOutcome, *, exit_code: int = 137) -> None:
    if outcome.exit_code != exit_code or outcome.signal is not None:
        raise AssertionError(
            f"expected os._exit({exit_code}), observed exit={outcome.exit_code} "
            f"signal={outcome.signal}; diagnostic={outcome.diagnostic}"
        )


def assert_sigkill(outcome: ChildOutcome) -> None:
    if outcome.signal != signal.SIGKILL or outcome.exit_code is not None:
        raise AssertionError(
            f"expected SIGKILL, observed exit={outcome.exit_code} "
            f"signal={outcome.signal}; diagnostic={outcome.diagnostic}"
        )


@dataclass(frozen=True)
class ReconciliationOutcome:
    first_result: Any
    second_result: Any
    first_snapshot: Mapping[str, Any]
    second_snapshot: Mapping[str, Any]


def assert_repeatable_reconciliation(
    action: Callable[[], Any],
    snapshotter: Callable[[], Mapping[str, Any]],
    *,
    result_normalizer: Callable[[Any], Any] | None = None,
    label: str = "reconciliation",
) -> ReconciliationOutcome:
    """Run restart recovery/rollback twice and require a stable second pass."""
    first_result = action()
    first_snapshot = snapshotter()
    second_result = action()
    second_snapshot = snapshotter()
    assert_snapshot_equal(first_snapshot, second_snapshot, label=f"{label} second pass")
    if result_normalizer is not None:
        first_normalized = result_normalizer(first_result)
        second_normalized = result_normalizer(second_result)
        if first_normalized != second_normalized:
            raise AssertionError(
                f"{label} result is not idempotent: "
                f"{first_normalized!r} != {second_normalized!r}"
            )
    return ReconciliationOutcome(
        first_result=first_result,
        second_result=second_result,
        first_snapshot=first_snapshot,
        second_snapshot=second_snapshot,
    )


@dataclass(frozen=True)
class CrashReconciliationOutcome:
    child: ChildOutcome
    reconciliation: ReconciliationOutcome


def hard_crash_then_reconcile(
    *,
    module: Any,
    cutpoint: FaultMatcher,
    mutate: Callable[[], Any],
    reconcile: Callable[[], Any],
    snapshotter: Callable[[], Mapping[str, Any]],
    harness_kwargs: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
    exit_code: int = 137,
    fault_mode: str = "hard",
    result_normalizer: Callable[[Any], Any] | None = None,
) -> CrashReconciliationOutcome:
    """Crash at one boundary, then prove next-process stable reconciliation."""

    def child_action() -> None:
        injector = FaultInjector(cutpoint, mode=fault_mode, exit_code=exit_code)
        with MutationBoundaryHarness(module, injector, **dict(harness_kwargs or {})):
            mutate()
        # The target must be reachable.  A non-fired injector makes the child
        # fail diagnostically instead of silently passing the crash case.
        injector.assert_fired()

    child = run_forked(child_action, timeout=timeout)
    if fault_mode == "hard":
        assert_hard_exit(child, exit_code=exit_code)
    elif fault_mode == "sigkill":
        assert_sigkill(child)
    else:
        raise ValueError("hard_crash_then_reconcile requires hard or sigkill mode")
    reconciliation = assert_repeatable_reconciliation(
        reconcile,
        snapshotter,
        result_normalizer=result_normalizer,
        label=f"cutpoint {cutpoint!r}",
    )
    return CrashReconciliationOutcome(child=child, reconciliation=reconciliation)


T = TypeVar("T")


def close_fixture(case: Any) -> None:
    close = getattr(case, "close", None)
    if callable(close):
        close()


def exercise_hard_crash_matrix(
    cutpoints: Iterable[FaultMatcher],
    *,
    make_case: Callable[[], T],
    module_for_case: Callable[[T], Any],
    mutate: Callable[[T], Any],
    reconcile: Callable[[T], Any],
    snapshotter: Callable[[T], Mapping[str, Any]],
    terminal_assertion: Callable[[T, FaultMatcher, ReconciliationOutcome], None] | None = None,
    harness_kwargs: Mapping[str, Any] | None = None,
    cleanup: Callable[[T], None] = close_fixture,
    timeout: float = 30.0,
    fault_mode: str = "hard",
) -> list[tuple[FaultMatcher, CrashReconciliationOutcome]]:
    """Exercise independent fixtures at every supplied hard-crash cutpoint."""
    results: list[tuple[FaultMatcher, CrashReconciliationOutcome]] = []
    for cutpoint in cutpoints:
        case = make_case()
        try:
            outcome = hard_crash_then_reconcile(
                module=module_for_case(case),
                cutpoint=cutpoint,
                mutate=lambda: mutate(case),
                reconcile=lambda: reconcile(case),
                snapshotter=lambda: snapshotter(case),
                harness_kwargs=harness_kwargs,
                timeout=timeout,
                fault_mode=fault_mode,
            )
            if terminal_assertion is not None:
                terminal_assertion(case, cutpoint, outcome.reconciliation)
            results.append((cutpoint, outcome))
        finally:
            cleanup(case)
    return results


def _confined_fixture_path(path: Path | str, fixture_root: Path | str) -> Path:
    """Confine a lexical mutation by its real parent, not its final target."""
    candidate = Path(path)
    root = Path(fixture_root).resolve(strict=True)
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise SnapshotError(f"fixture mutation parent is absent: {candidate.parent}") from exc
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise SnapshotError(f"fixture mutation escapes root {root}: {candidate}") from exc
    if candidate == root:
        raise SnapshotError("refusing to replace the fixture root")
    return candidate


def replace_with_symlink(
    path: Path | str, target: Path | str, *, fixture_root: Path | str
) -> Path:
    """Replace an empty directory/file/absent fixture node with a symlink."""
    candidate = _confined_fixture_path(path, fixture_root)
    try:
        info = candidate.lstat()
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
            candidate.rmdir()  # Deliberately refuses non-empty evidence trees.
        else:
            candidate.unlink()
    candidate.symlink_to(target)
    return candidate


def make_broken_symlink(path: Path | str, *, fixture_root: Path | str) -> Path:
    candidate = Path(path)
    missing = candidate.parent / f".{candidate.name}.missing-target"
    if missing.exists() or missing.is_symlink():
        raise SnapshotError(f"broken-link target unexpectedly exists: {missing}")
    return replace_with_symlink(candidate, missing, fixture_root=fixture_root)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def rewrite_sealed_json(
    path: Path | str,
    mutate: Callable[[dict[str, Any]], None],
    *,
    fixture_root: Path | str,
    seal_field: str | None = None,
) -> dict[str, Any]:
    """Fixture-only JSON/path attack, optionally recomputing an unkeyed seal."""
    candidate = _confined_fixture_path(path, fixture_root)
    raw = candidate.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise SnapshotError(f"JSON attack target is not an object: {candidate}")
    mutate(value)
    if seal_field is not None:
        unsigned = {key: item for key, item in value.items() if key != seal_field}
        value[seal_field] = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    candidate.write_bytes(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    return value


def overwrite_fixture_bytes(
    path: Path | str, payload: bytes, *, fixture_root: Path | str
) -> None:
    """Corrupt one fixture artifact while refusing any external target."""
    candidate = _confined_fixture_path(path, fixture_root)
    info = candidate.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SnapshotError(f"corruption target is not a real file: {candidate}")
    candidate.write_bytes(payload)


def assert_path_unchanged(path: Path | str, expected: Mapping[str, Any]) -> None:
    observed = snapshot_path(path)
    if observed != expected:
        raise AssertionError(
            f"external/sentinel path changed: {path}\n"
            + diff_snapshots(
                {"schema": SNAPSHOT_SCHEMA, "files": {"path": expected}, "trees": {}},
                {"schema": SNAPSHOT_SCHEMA, "files": {"path": observed}, "trees": {}},
            )
        )
