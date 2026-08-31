"""DETACHED WORK TRIAL — the resident must not idle while a subprocess runs.

The 1-hour autonomy incident (receipts/future/AUTONOMY_TIMELINE_1h.json) sat
in one I/O-bound verifier loop for 47 minutes after every decision had already
landed. Waiting on a subprocess while independent WorkUnits remain runnable is
an autonomy defect. This module demonstrates the positive on a REAL trial
timeline: a real detached OS process (pid + start-token + expected receipt),
independent WorkUnit progress the whole time it is in flight, a kqueue receipt
wakeup that is not a poll, a hard in-flight bound, and the three 1h-scar
guards evaluated against the timeline that actually happened.

    python3 tools/future/detached_trial.py --selftest
    python3 tools/future/detached_trial.py --build
    python3 tools/future/detached_trial.py --record
    python3 -m pytest tools/future/test_detached_trial.py -q

Does not fork tools/future/detached.py. Does not take a GPU lease.
Everything emitted is STATIC_ONLY, bench UNKNOWN, gpu_authority false.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import inspect
import json
import os
import queue
import select
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from hcli.resources import pid_is_alive
from hcli.workunit import WorkUnit
from tools.future import detached as d
from tools.future._common import RECEIPTS, REPO, git, sha256_file, write_receipt
from tools.future.wakeup import (
    COMPLETED,
    CONSUMER_KINDS,
    Watcher,
    classify_receipt_bytes,
    write_sealed,
)

RECEIPT = "DETACHED_WORK_TRIAL.json"
SCHEMA = "hawking.future.detached_trial.v1"
CHILD_SCHEMA = "hawking.future.detached_trial.child.v1"
INDEPENDENT_SCHEMA = "hawking.future.detached_trial.independent.v1"
RECORDED_BY = "tools/future/detached_trial.py"
VERSION = 1

# Campaign trial must outlast the 1h incident's decision cutoff (t_s 120)
# so "decisions did not all land in the first two minutes" is testable.
DEFAULT_CHILD_DURATION_S = 128.0
DEFAULT_INDEPENDENT_DURATION_S = 5.0
SAFE_IN_FLIGHT_BOUND = 2
IDLE_DEFECT_S = 10.0
TWO_MINUTES_S = 120.0
KQUEUE_TIMEOUT_S = 0.25
INSPECT_YIELD_S = 0.02
PID_WAIT_S = 2.0

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. "
    "Neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE. "
    "A trial timeline is process identity, disk landing, and wall seconds; "
    "it is not a bench."
)

DECISION_KINDS = frozenset(
    {
        "CHILD_LAUNCHED",
        "INDEPENDENT_STARTED",
        "INDEPENDENT_COMPLETED",
        "WORK_REFILLED",
        "CONCURRENCY_BOUND_REFUSED",
        "RECEIPT_WAKEUP",
        "NEXT_DECISION",
        "CHILD_TERMINAL",
    }
)


class TrialError(RuntimeError):
    """Operational failure of the trial harness. Never a silent pass."""


class ConcurrencyBoundError(TrialError):
    """Launch refused because the independent in-flight bound is saturated."""

    def __init__(self, bound: int, in_flight: int, unit_id: str) -> None:
        self.bound = int(bound)
        self.in_flight = int(in_flight)
        self.unit_id = str(unit_id)
        super().__init__(
            f"independent in-flight bound {self.bound} saturated "
            f"(live={self.in_flight}); refused launch of {self.unit_id!r}; "
            f"did not spawn"
        )


# ---------------------------------------------------------------------------
# Corpus: real files this checkout already has. Hashing them is real work.
# ---------------------------------------------------------------------------


def corpus_files() -> list[Path]:
    files: list[Path] = []
    future = REPO / "tools" / "future"
    receipts = REPO / "receipts" / "future"
    if future.is_dir():
        files.extend(sorted(p for p in future.glob("*.py") if p.is_file()))
    if receipts.is_dir():
        files.extend(sorted(p for p in receipts.glob("*.json") if p.is_file()))
    return files


def _slice_files(files: Sequence[Path], slice_i: int, slice_n: int) -> list[Path]:
    n = max(1, int(slice_n))
    i = int(slice_i) % n
    out = [p for idx, p in enumerate(files) if idx % n == i]
    return out or list(files)


def one_pass(files: Sequence[Path]) -> dict[str, Any]:
    """One real hashing + JSON-parse + receipt-classify pass. Not a sleep."""
    digest = __import__("hashlib").sha256()
    n_files = 0
    n_bytes = 0
    n_parsed = 0
    n_classified = 0
    for path in files:
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        digest.update(raw)
        n_files += 1
        n_bytes += len(raw)
        if path.suffix == ".json":
            try:
                json.loads(raw)
                n_parsed += 1
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                pass
            try:
                classify_receipt_bytes(raw)
                n_classified += 1
            except Exception:
                pass
    return {
        "n_files": n_files,
        "n_bytes": n_bytes,
        "n_parsed": n_parsed,
        "n_classified": n_classified,
        "digest_sha256": digest.hexdigest(),
    }


def child_work(
    receipt_path: str | Path,
    *,
    seconds: float,
    label: str = "child",
    slice_i: int = 0,
    slice_n: int = 1,
    schema: str = CHILD_SCHEMA,
) -> int:
    """Detached worker entry. Real CPU/IO on this repo's files, then a sealed receipt."""
    out = Path(receipt_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = _slice_files(corpus_files(), slice_i, slice_n)
    duration = max(0.05, float(seconds))
    t0 = time.monotonic()
    t0_unix = time.time()
    n_passes = 0
    total_bytes = 0
    last: dict[str, Any] = {
        "n_files": 0,
        "n_bytes": 0,
        "n_parsed": 0,
        "n_classified": 0,
        "digest_sha256": None,
    }
    deadline = t0 + duration
    while True:
        last = one_pass(files)
        n_passes += 1
        total_bytes += int(last.get("n_bytes") or 0)
        if time.monotonic() >= deadline:
            break
    elapsed = time.monotonic() - t0
    written_at = time.time()
    doc = {
        "schema": schema,
        "complete": True,
        "label": str(label),
        "pid": os.getpid(),
        "written_at": written_at,
        "started_at": t0_unix,
        "elapsed_s": elapsed,
        "duration_s_requested": duration,
        "n_passes": n_passes,
        "n_files_per_pass": last.get("n_files"),
        "n_parsed_per_pass": last.get("n_parsed"),
        "n_classified_per_pass": last.get("n_classified"),
        "bytes_read": total_bytes,
        "digest_sha256": last.get("digest_sha256"),
        "slice": {"i": int(slice_i), "n": int(slice_n)},
        "work": (
            "sha256 + json.loads + wakeup.classify_receipt_bytes over "
            "tools/future/*.py and receipts/future/*.json; not a sleep"
        ),
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
    }
    write_sealed(out, doc)
    return 0


# ---------------------------------------------------------------------------
# kqueue watcher. Harvest is a reaction to a vnode event, not a poll of the path.
# ---------------------------------------------------------------------------


def _kqueue_fflags() -> int:
    return int(
        select.KQ_NOTE_WRITE
        | select.KQ_NOTE_EXTEND
        | select.KQ_NOTE_ATTRIB
        | select.KQ_NOTE_LINK
        | select.KQ_NOTE_RENAME
        | select.KQ_NOTE_DELETE
    )


def kqueue_watch(path: Path, out: "queue.Queue[dict[str, Any]]", stop: threading.Event) -> None:
    """Block on EVFILT_VNODE for `path`'s directory. Do not test the file on timeout."""
    if not hasattr(select, "kqueue"):
        out.put({"kind": "kqueue_unavailable", "t_unix": time.time()})
        return
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    kq = select.kqueue()
    fd = os.open(str(directory), os.O_RDONLY)
    n_kevents = 0
    armed_at = time.time()
    try:
        ev = select.kevent(
            fd,
            filter=select.KQ_FILTER_VNODE,
            flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
            fflags=_kqueue_fflags(),
        )
        kq.control([ev], 0)
        while not stop.is_set():
            fired = kq.control(None, 1, KQUEUE_TIMEOUT_S)
            if not fired:
                continue
            n_kevents += 1
            t_kevent = time.time()
            fflags = int(getattr(fired[0], "fflags", 0) or 0)
            # Existence is inspected ONLY because a vnode event fired.
            if path.is_file():
                out.put(
                    {
                        "kind": "kqueue_vnode",
                        "t_unix": t_kevent,
                        "fflags": fflags,
                        "n_kevents": n_kevents,
                        "armed_at": armed_at,
                        "path": str(path),
                        "filter": "EVFILT_VNODE",
                    }
                )
                return
    except Exception as exc:  # noqa: BLE001 — the trial records the fault
        out.put(
            {
                "kind": "kqueue_error",
                "t_unix": time.time(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            kq.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Units launched through detached.py (never Popen/subprocess.run here)
# ---------------------------------------------------------------------------


def _worker_argv(
    receipt: Path,
    *,
    seconds: float,
    label: str,
    slice_i: int,
    slice_n: int,
) -> list[str]:
    return [
        _sys.executable,
        str(Path(__file__).resolve()),
        "--child-work",
        str(receipt),
        "--seconds",
        str(float(seconds)),
        "--label",
        str(label),
        "--slice",
        str(int(slice_i)),
        "--of",
        str(int(slice_n)),
    ]


def make_child_unit(receipt: Path, duration_s: float) -> dict[str, Any]:
    unit = {
        "id": "WU.DETACHED_TRIAL.child",
        "role": "science",
        "description": (
            "Real detached hashing/classify pass over this repo's future tools "
            "and receipts until the wall duration elapses, then a sealed receipt."
        ),
        "command": _worker_argv(
            receipt, seconds=duration_s, label="trial-child", slice_i=0, slice_n=1
        ),
        "resource_class": "LIGHT_CONTROL",
        "output_receipt_path": str(receipt),
        "verifier": "future.detached_trial.child_receipt",
        "classification": "STATIC_ONLY",
        "provider": "future.detached_trial",
        "effect_class": "READ_ONLY",
        "timeout_s": float(duration_s) + 60.0,
        "gpu_authority": False,
        "dependencies": [],
    }
    refuse = d.refuse_reason(unit["command"], resource_class=unit["resource_class"])
    if refuse:
        raise TrialError(f"child argv refused by detached.py: {refuse}")
    return unit


def make_independent_unit(
    workspace: Path,
    seq: int,
    *,
    duration_s: float,
    slice_n: int = 8,
) -> dict[str, Any]:
    uid = f"WU.DETACHED_TRIAL.ind.{seq:04d}"
    receipt = workspace / "results" / "ind" / f"{uid}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    unit = {
        "id": uid,
        "role": "science",
        "description": f"independent hashing slice {seq % slice_n}/{slice_n}",
        "command": _worker_argv(
            receipt,
            seconds=duration_s,
            label=uid,
            slice_i=seq % slice_n,
            slice_n=slice_n,
        ),
        "resource_class": "LIGHT_CONTROL",
        "output_receipt_path": str(receipt),
        "verifier": "future.detached_trial.independent_receipt",
        "classification": "STATIC_ONLY",
        "provider": "future.detached_trial",
        "effect_class": "READ_ONLY",
        "timeout_s": float(duration_s) + 30.0,
        "gpu_authority": False,
        "dependencies": [],
        "species": "independent_reproduction",
    }
    refuse = d.refuse_reason(unit["command"], resource_class=unit["resource_class"])
    if refuse:
        raise TrialError(f"independent argv refused by detached.py: {refuse}")
    return unit


def _wait_pid(supervisor: d.DetachedSupervisor, rec: Mapping[str, Any], timeout_s: float = PID_WAIT_S) -> dict[str, Any]:
    """Wait only for identity to be persisted. Never wait for the child to finish."""
    live = dict(rec)
    deadline = time.monotonic() + max(0.05, timeout_s)
    while time.monotonic() < deadline:
        if isinstance(live.get("pid"), int) and live.get("start_token"):
            live["identity_status"] = d.identity_status(live)
            return live
        try:
            live = supervisor.inspect(str(live["job_id"]))
        except FileNotFoundError:
            break
        time.sleep(0.02)
    live["identity_status"] = d.identity_status(live)
    return live


# ---------------------------------------------------------------------------
# Bounded independent launcher. The bound is the safety property.
# ---------------------------------------------------------------------------


class BoundedIndependentLauncher:
    """At most `bound` independent children. Exceeding the bound does not spawn."""

    def __init__(self, supervisor: d.DetachedSupervisor, bound: int) -> None:
        if int(bound) < 1:
            raise TrialError("concurrency bound must be >= 1")
        self.supervisor = supervisor
        self.bound = int(bound)
        self.live: dict[str, dict[str, Any]] = {}

    def n_live(self) -> int:
        n = 0
        for rec in self.live.values():
            pid = rec.get("pid")
            if isinstance(pid, int) and pid > 0 and pid_is_alive(pid):
                n += 1
            elif rec.get("terminal") is None and rec.get("state") in {"RUNNING", "STARTING"}:
                n += 1
        return n

    def launch(self, unit: Mapping[str, Any]) -> dict[str, Any]:
        n = self.n_live()
        if n >= self.bound:
            raise ConcurrencyBoundError(self.bound, n, str(unit.get("id") or ""))
        rec = self.supervisor.launch(unit)
        rec = _wait_pid(self.supervisor, rec)
        self.live[str(rec["job_id"])] = rec
        return rec

    def inspect_live(self) -> list[dict[str, Any]]:
        snaps: list[dict[str, Any]] = []
        finished: list[str] = []
        for job_id in list(self.live):
            try:
                snap = self.supervisor.inspect(job_id)
            except FileNotFoundError:
                snap = dict(self.live[job_id])
                snap["terminal"] = snap.get("terminal") or "unknown"
                snap["crash_reason"] = "supervision record vanished"
            self.live[job_id] = snap
            snaps.append(snap)
            if snap.get("terminal"):
                finished.append(job_id)
        for job_id in finished:
            self.live.pop(job_id, None)
        return snaps


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------


def run_real_trial(
    workspace: str | os.PathLike[str],
    *,
    child_duration_s: float = DEFAULT_CHILD_DURATION_S,
    independent_duration_s: float = DEFAULT_INDEPENDENT_DURATION_S,
    bound: int = SAFE_IN_FLIGHT_BOUND,
    exceed_attempts: int = 2,
    exceed_second_after_s: float = 30.0,
) -> dict[str, Any]:
    """Launch a real detached child and keep independent WorkUnits moving.

    Returns a timeline document. Does not write the campaign receipt.
    """
    root = Path(workspace).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    child_dir = root / "results" / "child"
    child_dir.mkdir(parents=True, exist_ok=True)
    child_receipt = child_dir / "child.json"
    (root / "results" / "ind").mkdir(parents=True, exist_ok=True)

    supervisor = d.DetachedSupervisor(root)
    watcher = Watcher(root / "wakeup_ledger.json", root=root)
    launcher = BoundedIndependentLauncher(supervisor, bound)
    consumer_hits: dict[str, list[str]] = {k: [] for k in CONSUMER_KINDS}

    def _bind(kind: str):
        def _fn(event: Any) -> None:
            consumer_hits[kind].append(getattr(event, "state", None) or str(event))

        return _fn

    watcher.register_consumer("verifier", _bind("verifier"))
    watcher.register_consumer("graph", _bind("graph"))
    watcher.register_consumer("frontier", _bind("frontier"))

    t0 = time.time()
    timeline: list[dict[str, Any]] = []
    seq = 0

    def emit(kind: str, **payload: Any) -> dict[str, Any]:
        nonlocal seq
        row = {
            "seq": seq,
            "t_s": time.time() - t0,
            "t_unix": time.time(),
            "kind": kind,
            "payload": payload,
        }
        seq += 1
        timeline.append(row)
        return row

    wakeup_q: "queue.Queue[dict[str, Any]]" = queue.Queue()
    stop = threading.Event()
    kq_thread = threading.Thread(
        target=kqueue_watch,
        args=(child_receipt, wakeup_q, stop),
        name="detached-trial-kqueue",
        daemon=True,
    )
    kq_thread.start()
    time.sleep(0.05)
    kqueue_armed_at = time.time()
    kqueue_armed_before_launch = kq_thread.is_alive() and not child_receipt.is_file()

    child_unit = make_child_unit(child_receipt, child_duration_s)
    watcher.register_expectation(
        unit_id=str(child_unit["id"]),
        path=child_receipt,
        dependents=("WU.DETACHED_TRIAL.next",),
        verifier="verifier",
        graph="graph",
        frontier="frontier",
        required_schema=CHILD_SCHEMA,
    )

    launch_return_t0 = time.monotonic()
    try:
        child_rec = supervisor.launch(child_unit)
    except d.UnsafeCommandError as exc:
        stop.set()
        raise TrialError(f"child launch refused: {exc.reason}") from exc
    launch_return_s = time.monotonic() - launch_return_t0
    child_rec = _wait_pid(supervisor, child_rec)
    terminal_at_return = child_rec.get("terminal")
    child_job = str(child_rec["job_id"])
    emit(
        "CHILD_LAUNCHED",
        job_id=child_job,
        unit_id=child_unit["id"],
        pid=child_rec.get("pid"),
        start_token=child_rec.get("start_token"),
        supervisor_pid=child_rec.get("supervisor_pid"),
        identity_status=child_rec.get("identity_status"),
        expected_receipt_path=child_rec.get("expected_receipt_path") or str(child_receipt),
        terminal_at_return=terminal_at_return,
        launch_return_s=launch_return_s,
        start_new_session=True,
        work_alive=bool(
            isinstance(child_rec.get("pid"), int) and pid_is_alive(int(child_rec["pid"]))
        ),
    )

    independent_started: list[dict[str, Any]] = []
    independent_completed: list[dict[str, Any]] = []
    refill_sets: list[list[str]] = []
    refusals: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    next_seq = 0
    n_exceed_shown = 0
    wakeup_record: Optional[dict[str, Any]] = None
    child_terminal_row: Optional[dict[str, Any]] = None
    inspect_saw_terminal_at: Optional[float] = None
    idle_s = 0.0
    max_idle_gap_s = 0.0
    last_sample_unix = time.time()
    last_progress_unix = time.time()
    n_idle_samples = 0

    def _child_running() -> tuple[bool, dict[str, Any]]:
        nonlocal inspect_saw_terminal_at, child_terminal_row
        try:
            snap = supervisor.inspect(child_job)
        except FileNotFoundError:
            snap = dict(child_rec)
            snap["terminal"] = snap.get("terminal") or "unknown"
        running = snap.get("terminal") is None
        if not running and inspect_saw_terminal_at is None:
            inspect_saw_terminal_at = time.time()
            if child_terminal_row is None:
                child_terminal_row = emit(
                    "CHILD_TERMINAL",
                    job_id=child_job,
                    terminal=snap.get("terminal"),
                    returncode=snap.get("returncode"),
                    pid=snap.get("pid"),
                    source="inspect_not_wakeup",
                )
        return running, snap

    def _launch_batch(n: int) -> list[str]:
        nonlocal next_seq, last_progress_unix
        launched_ids: list[str] = []
        for _ in range(max(0, n)):
            unit = make_independent_unit(
                root, next_seq, duration_s=independent_duration_s
            )
            next_seq += 1
            try:
                rec = launcher.launch(unit)
            except ConcurrencyBoundError as exc:
                refusals.append(
                    {
                        "unit_id": exc.unit_id,
                        "bound": exc.bound,
                        "in_flight": exc.in_flight,
                        "spawned": False,
                        "pid": None,
                        "t_s": time.time() - t0,
                    }
                )
                emit(
                    "CONCURRENCY_BOUND_REFUSED",
                    unit_id=exc.unit_id,
                    bound=exc.bound,
                    in_flight=exc.in_flight,
                    spawned=False,
                    pid=None,
                )
                break
            launched_ids.append(str(unit["id"]))
            seen_ids.add(str(unit["id"]))
            independent_started.append(
                {
                    "unit_id": unit["id"],
                    "job_id": rec.get("job_id"),
                    "pid": rec.get("pid"),
                    "start_token": rec.get("start_token"),
                    "t_s": time.time() - t0,
                }
            )
            emit(
                "INDEPENDENT_STARTED",
                unit_id=unit["id"],
                job_id=rec.get("job_id"),
                pid=rec.get("pid"),
                start_token=rec.get("start_token"),
                identity_status=rec.get("identity_status"),
                expected_receipt_path=rec.get("expected_receipt_path"),
            )
            last_progress_unix = time.time()
        if launched_ids:
            refill_sets.append(list(launched_ids))
            emit(
                "WORK_REFILLED",
                unit_ids=list(launched_ids),
                queue_depth=launcher.n_live(),
                novel_vs_prior=True,
            )
        return launched_ids

    # Fill the independent bound immediately. This is the first refill.
    _launch_batch(bound)

    hard_deadline = t0 + float(child_duration_s) + 90.0
    wakeup_grace_deadline: Optional[float] = None

    try:
        while True:
            now = time.time()
            if now > hard_deadline:
                raise TrialError(
                    f"trial exceeded hard deadline ({hard_deadline - t0:.1f}s) "
                    "without a kqueue wakeup plus drain"
                )

            progressed = False
            child_running, child_snap = _child_running()

            # Wakeup path: kqueue thread only. Main loop does not stat the child receipt.
            kev = None
            try:
                kev = wakeup_q.get_nowait()
            except queue.Empty:
                kev = None
            if kev is not None and wakeup_record is None:
                land_t = None
                written_at = None
                if child_receipt.is_file():
                    try:
                        body = json.loads(child_receipt.read_text(encoding="utf-8"))
                        written_at = body.get("written_at")
                    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                        body = None
                    try:
                        land_t = child_receipt.stat().st_mtime
                    except OSError:
                        land_t = None
                harvest_at = time.time()
                events = watcher.harvest()
                dispatched = [watcher.dispatch(ev).to_dict() for ev in events]
                react_at = time.time()
                kq_t = float(kev.get("t_unix") or harvest_at)
                latency = None
                if isinstance(written_at, (int, float)):
                    latency = kq_t - float(written_at)
                wakeup_record = {
                    "mechanism": "kqueue EVFILT_VNODE",
                    "kqueue_event": kev,
                    "kqueue_t_unix": kq_t,
                    "receipt_written_at": written_at,
                    "receipt_mtime": land_t,
                    "harvest_at": harvest_at,
                    "dispatch_at": react_at,
                    "kqueue_latency_s": latency,
                    "supervisor_reaction_latency_s": (
                        (react_at - float(written_at))
                        if isinstance(written_at, (int, float))
                        else None
                    ),
                    "n_harvest_events": len(events),
                    "dispatch": dispatched,
                    "consumers_hit": {k: list(v) for k, v in consumer_hits.items()},
                    "poll": False,
                    "wait_terminal": False,
                }
                emit(
                    "RECEIPT_WAKEUP",
                    **{k: v for k, v in wakeup_record.items() if k != "dispatch"},
                    event_states=[e.state for e in events],
                )
                emit(
                    "NEXT_DECISION",
                    action="ingest_detached_result_and_continue_independent_work",
                    child_job_id=child_job,
                    n_independent_live=launcher.n_live(),
                    wakeup_mechanism="kqueue EVFILT_VNODE",
                )
                progressed = True
                last_progress_unix = time.time()

            # Independent completions. inspect is supervisor-side, not a child wait.
            for snap in launcher.inspect_live():
                if not snap.get("terminal"):
                    continue
                progressed = True
                last_progress_unix = time.time()
                independent_completed.append(
                    {
                        "unit_id": snap.get("workunit_id") or snap.get("unit_id"),
                        "job_id": snap.get("job_id"),
                        "terminal": snap.get("terminal"),
                        "pid": snap.get("pid"),
                        "t_s": time.time() - t0,
                        "child_still_running": child_running,
                        "receipt_present": bool(
                            snap.get("expected_receipt_path")
                            and Path(str(snap["expected_receipt_path"])).is_file()
                        ),
                    }
                )
                emit(
                    "INDEPENDENT_COMPLETED",
                    unit_id=snap.get("workunit_id"),
                    job_id=snap.get("job_id"),
                    terminal=snap.get("terminal"),
                    pid=snap.get("pid"),
                    child_still_running=child_running,
                )

            n_live = launcher.n_live()
            if child_running and n_live < bound:
                launched = _launch_batch(bound - n_live)
                if launched:
                    progressed = True

            # Demonstrate the bound: while saturated, a further unit is refused.
            want_exceed = n_exceed_shown == 0 or (
                n_exceed_shown >= 1 and (time.time() - t0) >= float(exceed_second_after_s)
            )
            if (
                child_running
                and launcher.n_live() >= bound
                and n_exceed_shown < exceed_attempts
                and want_exceed
            ):
                extra = make_independent_unit(
                    root, next_seq, duration_s=independent_duration_s
                )
                next_seq += 1
                live_before = launcher.n_live()
                spawned = False
                pid = None
                try:
                    rec = launcher.launch(extra)
                    spawned = rec.get("pid") is not None
                    pid = rec.get("pid")
                    # Bound failed to hold. Record honesty; cancel the overflow.
                    if rec.get("job_id"):
                        try:
                            supervisor.cancel(str(rec["job_id"]))
                        except Exception:
                            pass
                        launcher.live.pop(str(rec["job_id"]), None)
                except ConcurrencyBoundError as exc:
                    spawned = False
                    pid = None
                    refusals.append(
                        {
                            "unit_id": extra["id"],
                            "bound": exc.bound,
                            "in_flight": exc.in_flight,
                            "spawned": False,
                            "pid": None,
                            "t_s": time.time() - t0,
                            "live_before": live_before,
                        }
                    )
                    emit(
                        "CONCURRENCY_BOUND_REFUSED",
                        unit_id=extra["id"],
                        bound=exc.bound,
                        in_flight=exc.in_flight,
                        spawned=False,
                        pid=None,
                        live_before=live_before,
                    )
                else:
                    refusals.append(
                        {
                            "unit_id": extra["id"],
                            "bound": bound,
                            "in_flight": live_before,
                            "spawned": spawned,
                            "pid": pid,
                            "t_s": time.time() - t0,
                            "bound_held": False,
                        }
                    )
                    emit(
                        "CONCURRENCY_BOUND_REFUSED",
                        unit_id=extra["id"],
                        bound=bound,
                        in_flight=live_before,
                        spawned=spawned,
                        pid=pid,
                        bound_held=False,
                    )
                n_exceed_shown += 1

            # Idle runnable seconds: child in flight, no independent work progressing.
            sample_unix = time.time()
            dt = sample_unix - last_sample_unix
            last_sample_unix = sample_unix
            runnable_existed = child_running  # generator is unlimited while child runs
            nothing_progressed = launcher.n_live() == 0 and not progressed
            if runnable_existed and nothing_progressed and dt > 0:
                idle_s += dt
                n_idle_samples += 1
                if dt > max_idle_gap_s:
                    max_idle_gap_s = dt
                gap = sample_unix - last_progress_unix
                if gap > max_idle_gap_s:
                    max_idle_gap_s = gap

            n_live = launcher.n_live()
            if wakeup_record is not None and not child_running and n_live == 0:
                break
            if wakeup_record is None and not child_running:
                if wakeup_grace_deadline is None:
                    wakeup_grace_deadline = time.time() + 8.0
                elif time.time() > wakeup_grace_deadline:
                    break
            if n_live >= bound or (not child_running and n_live > 0) or (
                wakeup_record is None and not child_running
            ):
                time.sleep(INSPECT_YIELD_S)
    finally:
        stop.set()
        kq_thread.join(timeout=1.0)
        try:
            supervisor.reap_all()
        except Exception:
            pass

    completed_during_wait = [
        row for row in independent_completed if row.get("child_still_running")
    ]

    decision_ts = [float(e["t_s"]) for e in timeline if e.get("kind") in DECISION_KINDS]
    elapsed_s = time.time() - t0
    child_final = child_snap if "child_snap" in locals() else child_rec
    try:
        child_final = supervisor.inspect(child_job)
    except Exception:
        child_final = dict(child_rec)

    harvested_completed = False
    if wakeup_record:
        states = []
        for item in wakeup_record.get("dispatch") or []:
            if isinstance(item, Mapping):
                states.append(item.get("state"))
        harvested_completed = COMPLETED in states or "COMPLETED" in states
        if not harvested_completed:
            # harvest events recorded on the timeline payload
            for event in timeline:
                if event.get("kind") == "RECEIPT_WAKEUP":
                    event_states = (event.get("payload") or {}).get("event_states") or []
                    harvested_completed = COMPLETED in event_states
    wakeup_ok = bool(
        wakeup_record
        and wakeup_record.get("mechanism", "").startswith("kqueue")
        and wakeup_record.get("poll") is False
        and (wakeup_record.get("n_harvest_events") or 0) >= 1
        and harvested_completed
    )
    bound_held = all(not r.get("spawned") for r in refusals) and n_exceed_shown >= 1
    overlap_ok = len(completed_during_wait) >= 1 and len(independent_started) >= 1
    real_child_ok = (
        isinstance(child_rec.get("pid"), int)
        and int(child_rec["pid"]) > 1
        and bool(child_rec.get("start_token"))
        and launch_return_s < 2.0
        and terminal_at_return is None
    )

    guards = evaluate_autonomy_1h_guards(
        timeline,
        elapsed_s=elapsed_s,
        refill_sets=refill_sets,
        refusals=refusals,
        specimen_ingests=0,
    )

    idle_defect = idle_s > IDLE_DEFECT_S
    verdict = "PASS"
    unmet: list[str] = []
    if not real_child_ok:
        unmet.append("real_detached_child")
    if not overlap_ok:
        unmet.append("independent_progress_during_wait")
    if idle_defect:
        unmet.append("idle_runnable_seconds")
    if not wakeup_ok:
        unmet.append("receipt_wakeup")
    if not bound_held:
        unmet.append("concurrency_bound")
    if not guards["passed"]:
        unmet.append("autonomy_1h_guards")
    if unmet:
        verdict = "FAIL"

    return {
        "verdict": verdict,
        "unmet": unmet,
        "elapsed_s": elapsed_s,
        "t0_unix": t0,
        "child_duration_s_requested": float(child_duration_s),
        "independent_duration_s_requested": float(independent_duration_s),
        "safe_in_flight_bound": int(bound),
        "idle_runnable_seconds": idle_s,
        "max_idle_gap_s": max_idle_gap_s,
        "n_idle_samples": n_idle_samples,
        "idle_is_autonomy_defect": idle_defect,
        "idle_defect_threshold_s": IDLE_DEFECT_S,
        "kqueue_armed_before_launch": kqueue_armed_before_launch,
        "kqueue_armed_at": kqueue_armed_at,
        "child": {
            "job_id": child_job,
            "unit_id": child_unit["id"],
            "pid": child_rec.get("pid"),
            "start_token": child_rec.get("start_token"),
            "supervisor_pid": child_rec.get("supervisor_pid"),
            "identity_status_at_launch": child_rec.get("identity_status"),
            "expected_receipt_path": child_rec.get("expected_receipt_path")
            or str(child_receipt),
            "launch_return_s": launch_return_s,
            "terminal": child_final.get("terminal"),
            "returncode": child_final.get("returncode"),
            "receipt_present": child_receipt.is_file(),
            "start_new_session": True,
            "argv0": child_unit["command"][0],
            "argv_script": child_unit["command"][1] if len(child_unit["command"]) > 1 else None,
        },
        "independent": {
            "n_started": len(independent_started),
            "n_completed": len(independent_completed),
            "n_completed_during_wait": len(completed_during_wait),
            "started": independent_started,
            "completed": independent_completed,
            "completed_during_wait_unit_ids": [
                r.get("unit_id") for r in completed_during_wait
            ],
        },
        "refills": {
            "n": len(refill_sets),
            "sets": refill_sets,
            "all_ids": sorted(seen_ids),
        },
        "concurrency": {
            "bound": int(bound),
            "n_exceed_attempts": n_exceed_shown,
            "refusals": refusals,
            "held": bound_held,
            "on_exceed": "refuse launch; persist CONCURRENCY_BOUND_REFUSED; spawn nothing",
        },
        "wakeup": wakeup_record,
        "inspect_saw_terminal_at": inspect_saw_terminal_at,
        "timeline": timeline,
        "decision_t_s": decision_ts,
        "autonomy_1h_guards": guards,
        "proofs": {
            "real_child": {
                "passed": real_child_ok,
                "pid": child_rec.get("pid"),
                "start_token": child_rec.get("start_token"),
                "launch_return_s": launch_return_s,
                "identity_status": child_rec.get("identity_status"),
            },
            "independent_progress_during_wait": {
                "passed": overlap_ok,
                "n_started": len(independent_started),
                "n_completed_during_wait": len(completed_during_wait),
            },
            "idle_runnable_seconds": {
                "passed": not idle_defect,
                "value": idle_s,
                "threshold_s": IDLE_DEFECT_S,
            },
            "receipt_wakeup": {
                "passed": wakeup_ok,
                "mechanism": (wakeup_record or {}).get("mechanism"),
                "kqueue_latency_s": (wakeup_record or {}).get("kqueue_latency_s"),
                "supervisor_reaction_latency_s": (wakeup_record or {}).get(
                    "supervisor_reaction_latency_s"
                ),
            },
            "concurrency_bound": {
                "passed": bound_held,
                "bound": int(bound),
                "n_refusals": len(refusals),
            },
            "autonomy_1h_guards": guards,
        },
        "consumer_hits": {k: list(v) for k, v in consumer_hits.items()},
        "workspace": str(root),
    }


def evaluate_autonomy_1h_guards(
    timeline: Sequence[Mapping[str, Any]],
    *,
    elapsed_s: float,
    refill_sets: Sequence[Sequence[str]],
    refusals: Sequence[Mapping[str, Any]],
    specimen_ingests: int,
) -> dict[str, Any]:
    """The three scars from AUTONOMY_TIMELINE_1h. Fail closed if any fires."""
    # 1. Deterministic rejection table: 222 idea_rejected, all in t_s 7-16, never
    #    recur. This trial must not replay that table. Concurrency refusals are a
    #    different class and must not collapse into one early identical burst.
    idea_rejected = [e for e in timeline if e.get("kind") == "idea_rejected"]
    rejection_ts = [float(e.get("t_s") or 0) for e in idea_rejected]
    if idea_rejected:
        span = max(rejection_ts) - min(rejection_ts) if rejection_ts else 0.0
        clustered = (
            min(rejection_ts) >= 7.0
            and max(rejection_ts) <= 16.0
            and elapsed_s > TWO_MINUTES_S
        )
        payloads = [
            json.dumps(e.get("payload") or {}, sort_keys=True, default=str)
            for e in idea_rejected
        ]
        identical = len(set(payloads)) == 1 and len(payloads) > 10
        g1_pass = not (clustered and identical)
        g1 = {
            "passed": g1_pass,
            "n_idea_rejected": len(idea_rejected),
            "span_s": span,
            "clustered_in_t7_16": clustered,
            "identical_payloads": identical,
            "reason": (
                "replayed a deterministic rejection table"
                if not g1_pass
                else "idea_rejected events are not the 1h t_s 7-16 table"
            ),
        }
    else:
        # Concurrency refusals: different unit ids, not confined to t 7-16.
        refuse_ids = [str(r.get("unit_id") or "") for r in refusals]
        refuse_ts = [float(r.get("t_s") or 0) for r in refusals]
        unique_ids = len(set(refuse_ids)) == len(refuse_ids) if refuse_ids else True
        if refuse_ts and elapsed_s > TWO_MINUTES_S:
            only_early = max(refuse_ts) <= 16.0 and min(refuse_ts) >= 7.0
        else:
            only_early = False
        g1_pass = unique_ids and not only_early
        g1 = {
            "passed": g1_pass,
            "n_idea_rejected": 0,
            "n_concurrency_refusals": len(refusals),
            "refusal_ids_unique": unique_ids,
            "refusals_only_in_t7_16": only_early,
            "reason": (
                "no idea_rejected table; concurrency refusals are per-id and not "
                "a t_s 7-16 replay"
                if g1_pass
                else "refusals collapsed into the 1h rejection-table shape"
            ),
        }

    # 2. Refills must return NOVEL ids, not the same set four times.
    if len(refill_sets) < 2:
        g2 = {
            "passed": False,
            "n_refills": len(refill_sets),
            "reason": "need at least two refills to prove novelty against the 1h scar",
        }
    else:
        seen: set[str] = set()
        novel_per: list[list[str]] = []
        identical_consecutive = 0
        prev: Optional[set[str]] = None
        for batch in refill_sets:
            ids = [str(x) for x in batch]
            idset = set(ids)
            novel_per.append(sorted(idset - seen))
            if prev is not None and idset == prev:
                identical_consecutive += 1
            prev = idset
            seen |= idset
        later_novel = all(bool(n) for n in novel_per[1:])
        same_as_1h = identical_consecutive == len(refill_sets) - 1
        g2_pass = later_novel and not same_as_1h
        g2 = {
            "passed": g2_pass,
            "n_refills": len(refill_sets),
            "n_unique_ids": len(seen),
            "identical_consecutive_refills": identical_consecutive,
            "later_refills_had_novel_ids": later_novel,
            "novel_ids_per_refill": novel_per,
            "reason": (
                "each refill after the first introduced ids not seen before"
                if g2_pass
                else "refills repeated the same id set (1h scar)"
            ),
        }

    # 3. Decisions did not all land in the first two minutes.
    decision_ts = [float(e.get("t_s") or 0) for e in timeline if e.get("kind") in DECISION_KINDS]
    n_after = sum(1 for t in decision_ts if t > TWO_MINUTES_S)
    last = max(decision_ts) if decision_ts else 0.0
    if elapsed_s <= TWO_MINUTES_S:
        g3_pass = False
        g3_reason = (
            f"elapsed_s={elapsed_s:.1f} <= 120; cannot prove decisions continued "
            "past the 1h incident's decision cutoff"
        )
    else:
        g3_pass = n_after >= 1 and last > TWO_MINUTES_S
        g3_reason = (
            f"{n_after} decision(s) after t_s 120; last_decision_t_s={last:.3f}"
            if g3_pass
            else "every decision landed before t_s 120 (1h scar)"
        )
    g3 = {
        "passed": g3_pass,
        "elapsed_s": elapsed_s,
        "n_decisions": len(decision_ts),
        "n_decisions_after_120": n_after,
        "last_decision_t_s": last,
        "first_decision_t_s": min(decision_ts) if decision_ts else None,
        "reason": g3_reason,
    }

    specimen_ok = int(specimen_ingests) == 0
    passed = bool(g1["passed"] and g2["passed"] and g3["passed"] and specimen_ok)
    return {
        "passed": passed,
        "rejection_table": g1,
        "refill_novelty": g2,
        "decisions_after_two_minutes": g3,
        "specimen_verification_ingests": int(specimen_ingests),
        "specimen_verification_not_looped": specimen_ok,
    }


def _source_does_not_wait_terminal() -> dict[str, Any]:
    src = Path(__file__).read_text(encoding="utf-8")
    run_src = inspect.getsource(run_real_trial)
    return {
        "run_real_trial_calls_wait_terminal": "wait_terminal(" in run_src,
        "run_real_trial_calls_subprocess_run": "subprocess.run" in run_src,
        "run_real_trial_calls_popen": "Popen(" in run_src,
        "module_imports_detached": "from tools.future import detached as d" in src
        or "from tools.future import detached" in src,
        "module_imports_wakeup": "from tools.future.wakeup import" in src,
    }


def emit_resident_workunit() -> dict[str, Any]:
    unit = WorkUnit(
        id="future.detached_trial.real-timeline",
        role="science",
        description=(
            "Launch a real detached OS process, keep independent WorkUnits "
            "moving for the whole wait, wake on the child's receipt via kqueue, "
            "and report idle runnable seconds plus the three AUTONOMY_TIMELINE_1h guards."
        ),
        resource_class="LIGHT_CONTROL",
        provider="future.detached_trial",
        verifier="future.detached_trial.real_timeline",
        effect_class="READ_ONLY",
        workspace="repo-root",
        classification="STATIC_ONLY",
        status="pending",
    )
    row = unit.to_dict()
    row.update(
        {
            "command": [_sys.executable, str(Path(__file__).resolve()), "--record"],
            "output_receipt_path": str(RECEIPTS / RECEIPT),
            "claim_boundary": CLAIM_BOUNDARY,
            "species": "independent_reproduction",
            "requires_quiescence": False,
            "may_promote": False,
            "may_modify_verifier": False,
            "gpu_authority": False,
            "timeout_s": DEFAULT_CHILD_DURATION_S + 120.0,
        }
    )
    WorkUnit.from_dict(dict(row))
    return row


RECOVERED_IMPLEMENTATION = [
    "tools/future/detached.py DetachedSupervisor.launch/inspect/cancel — spawn, pid+start_token identity, expected receipt; this trial does not reimplement them",
    "tools/future/wakeup.py Watcher.register_expectation/harvest/dispatch — receipt appearance is the completion event; poll/wait aliases stay refused",
    "tools/future/no_wait_scheduler.py — the overlapping-interval idea this trial measures on a longer real child",
    "tools/future/improvement_trial.py open_handle_wait — the 477s negative control that must FAIL; this lane is the positive",
    "receipts/future/AUTONOMY_TIMELINE_1h.json — the scar: 222 rejections in t_s 7-16, four identical refills, all decisions before t_s 120, then 47 minutes of one I/O loop",
    "hcli/agentos/background.py start_new_session — the session path detached.py already recovered",
    "select.kqueue EVFILT_VNODE — macOS filesystem wakeup; not a poll of the receipt path",
]

GAPS_CLOSED = [
    "a real detached child (pid + start_token + expected receipt) runs while the supervisor keeps launching independent WorkUnits",
    "idle runnable seconds are measured on the wall clock, not asserted by intent",
    "receipt wakeup is a kqueue vnode event that then harvests and dispatches; the main loop does not wait_terminal the child",
    "SAFE_IN_FLIGHT_BOUND is enforced; a launch above the bound is refused and does not spawn",
    "the three AUTONOMY_TIMELINE_1h guards are evaluated on this run's timeline and can FAIL it",
]

NEGATIVE_FINDINGS = [
    "autonomy_run.py still calls subprocess.run; this lane is not allowed to edit it",
    "kqueue is a macOS/BSD primitive; this host is macOS and the trial FAILs if kqueue cannot arm",
    "this process has no GPU lease; GPU work remains SLEEPING via detached.py's refuse_reason",
    "a 1h-scale idle tail cannot be reproduced in a 128s trial; the guards test the decision-cutoff and refill/rejection shapes, not a 47-minute wait",
]


def build(
    *,
    child_duration_s: float = DEFAULT_CHILD_DURATION_S,
    independent_duration_s: float = DEFAULT_INDEPENDENT_DURATION_S,
    bound: int = SAFE_IN_FLIGHT_BOUND,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="hawking-detached-trial-") as td:
        trial = run_real_trial(
            Path(td),
            child_duration_s=child_duration_s,
            independent_duration_s=independent_duration_s,
            bound=bound,
        )
    source = _source_does_not_wait_terminal()
    unit = emit_resident_workunit()
    proofs = trial["proofs"]
    proofs_all_passed = trial["verdict"] == "PASS" and all(
        (v.get("passed") if isinstance(v, dict) else True) for v in proofs.values()
    )
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Waiting on a subprocess no longer idles the resident. Detached "
            "launch, receipt wakeup, and independent-WorkUnit progress during a "
            "wait are demonstrated on a real trial timeline."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "module_sha256": sha256_file(Path(__file__)),
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "claim_boundary": CLAIM_BOUNDARY,
        "synthetic": False,
        "fixture": False,
        "verdict": trial["verdict"],
        "unmet": trial["unmet"],
        "elapsed_s": trial["elapsed_s"],
        "t0_unix": trial["t0_unix"],
        "idle_runnable_seconds": trial["idle_runnable_seconds"],
        "max_idle_gap_s": trial["max_idle_gap_s"],
        "idle_is_autonomy_defect": trial["idle_is_autonomy_defect"],
        "safe_in_flight_bound": trial["safe_in_flight_bound"],
        "child": trial["child"],
        "independent": {
            "n_started": trial["independent"]["n_started"],
            "n_completed": trial["independent"]["n_completed"],
            "n_completed_during_wait": trial["independent"]["n_completed_during_wait"],
            "completed_during_wait_unit_ids": trial["independent"][
                "completed_during_wait_unit_ids"
            ],
            "started": trial["independent"]["started"],
            "completed": trial["independent"]["completed"],
        },
        "refills": trial["refills"],
        "concurrency": trial["concurrency"],
        "wakeup": trial["wakeup"],
        "timeline": trial["timeline"],
        "decision_t_s": trial["decision_t_s"],
        "autonomy_1h_guards": trial["autonomy_1h_guards"],
        "proofs": proofs,
        "proofs_all_passed": proofs_all_passed,
        "source_guards": source,
        "kqueue_armed_before_launch": trial["kqueue_armed_before_launch"],
        "negative_control": {
            "receipt": "receipts/future/IMPROVEMENT_TRIAL.json",
            "control": "open_handle_wait",
            "must_fail": True,
            "repro_s": 477,
            "this_lane": "the positive: independent WorkUnits complete while the detached child is still running",
        },
        "scar": {
            "receipt": "receipts/future/AUTONOMY_TIMELINE_1h.json",
            "rejections_in_t7_16": 222,
            "identical_refills": 4,
            "decisions_before_t_s": 120,
            "idle_tail_s": 2864,
            "specimen_verification_ingests": 29,
        },
        "recovered_implementation": RECOVERED_IMPLEMENTATION,
        "gaps_closed": GAPS_CLOSED,
        "negative_findings": NEGATIVE_FINDINGS,
        "resident_callable": {
            "callable": True,
            "entry_point": (
                "python3 tools/future/detached_trial.py --selftest | --build | --record"
            ),
            "python_api": "tools.future.detached_trial.run_real_trial(workspace, ...)",
            "workunit": unit,
            "workunit_id": unit["id"],
            "receipt": f"receipts/future/{RECEIPT}",
            "receipt_schema": SCHEMA,
            "fail_closed": [
                "unsafe argv raises via detached.UnsafeCommandError; nothing is spawned",
                "independent launch above SAFE_IN_FLIGHT_BOUND is refused and does not spawn",
                "a missing kqueue wakeup FAILs the trial rather than treating inspect as completion",
                "the three AUTONOMY_TIMELINE_1h guards FAIL the trial if any fires",
                "idle runnable seconds above the defect threshold FAIL the trial",
            ],
        },
        "does_not_call_subprocess_run_for_work": True,
        "does_not_call_wait_terminal_on_child": not source["run_real_trial_calls_wait_terminal"],
        "consumes_detached_py": True,
        "does_not_fork_detached_py": True,
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def record() -> Path:
    return build()


def main(argv: Optional[Sequence[str]] = None) -> int:
    values = list(argv if argv is not None else _sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--child-work")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--label", default="child")
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args(values)
    if args.child_work:
        return child_work(
            Path(args.child_work),
            seconds=float(args.seconds if args.seconds is not None else 1.0),
            label=str(args.label),
            slice_i=int(args.slice),
            slice_n=int(args.of),
        )
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
