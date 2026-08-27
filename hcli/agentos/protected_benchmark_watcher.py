"""Bounded, non-destructive watcher for a protected Qwen benchmark window.

The watcher owns no global process and never kills anything.  It waits for a
machine-wide quiet window, treats ModelLake as an untouchable blocker, and
optionally pauses only clearly identified HCLI benchmark jobs for the duration
of the diagnostic.  Any resulting A/B remains ``NOT_FOR_PROMOTION`` until the
caller separately accepts the complete protected evidence.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from hcli.agentos.accelerator_regression import _profile_path, _quiescence, _repo_root
from hcli.agentos.background import BackgroundJobStore
from hcli.agentos.qwen27_mlp_diagnostic import run_qwen27_mlp_diagnostic_ab
from hcli.flash_next import REPO_ID
from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.qwen_protected_benchmark_watcher.v1"
DEFAULT_EMIT_NAME = "QWEN_PROTECTED_BENCH_READY.json"
DEFAULT_RESULT_NAME = "QWEN27_MLP_PROTECTED_AB.json"
MODEL_LAKE_MARKERS = ("modellake", "tools/odyssey/modellake.py", REPO_ID.lower())
PAUSABLE_MARKERS = (
    "accelerator-regression",
    "autonomy-gate",
    "unattended-window",
    "qwen27-mlp-ab",
    "qwen38-fusion-audit",
    "flash-executable",
)


def _safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _job_is_model_lake(job: Mapping[str, Any]) -> bool:
    label = str(job.get("label") or "").lower()
    argv = " ".join(str(item) for item in (job.get("argv") or [])).lower()
    return any(marker in label or marker in argv for marker in MODEL_LAKE_MARKERS)


def _job_is_pausable_hcli(job: Mapping[str, Any]) -> bool:
    if _job_is_model_lake(job):
        return False
    label = str(job.get("label") or "").lower()
    argv = " ".join(str(item) for item in (job.get("argv") or [])).lower()
    return "hcli" in label and any(marker in argv or marker in label for marker in PAUSABLE_MARKERS)


def _running_jobs(repo: Path) -> list[Dict[str, Any]]:
    try:
        store = BackgroundJobStore(repo, allowed_roots=(repo,))
        owner_pids = {os.getpid(), os.getppid()}
        return [
            row
            for row in store.list()
            if row.get("state") == "RUNNING" and row.get("pid") not in owner_pids
        ]
    except (OSError, ValueError, RuntimeError):
        return []


def _watcher_quiescence() -> Dict[str, Any]:
    """Sample machine load while excluding only this watcher process tree."""
    sample = _quiescence()
    if not isinstance(sample, Mapping):
        return {"quiet": None, "method": "invalid", "contenders": []}
    body = dict(sample)
    owner_pids = {os.getpid(), os.getppid()}
    contenders = [
        row for row in (body.get("contenders") or [])
        if not isinstance(row, Mapping) or row.get("pid") not in owner_pids
    ]
    removed = [
        row for row in (body.get("contenders") or [])
        if isinstance(row, Mapping) and row.get("pid") in owner_pids
    ]
    if removed:
        body["contenders"] = contenders
        body["n_contenders"] = len(contenders)
        body["quiet"] = body.get("error") is None and not contenders
        body["self_excluded_pids"] = sorted(owner_pids)
        body["self_excluded_contenders"] = _safe(removed)
    return body


def _process_identity(pid: Any) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["ps", "-o", "pid=,pgid=,comm=,command=", "-p", str(int(pid))],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"pid": pid, "status": "UNKNOWN", "error": f"{type(exc).__name__}: {exc}"}
    line = (result.stdout or "").strip()
    return {"pid": pid, "status": "FOUND" if line else "ABSENT", "ps": line[:1000]}


def _pause_jobs(jobs: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    paused: list[Dict[str, Any]] = []
    for job in jobs:
        pid = job.get("pid")
        identity = _process_identity(pid)
        row: Dict[str, Any] = {
            "job_id": job.get("job_id"),
            "label": job.get("label"),
            "pid": pid,
            "identity_before": identity,
            "paused": False,
            "restored": False,
        }
        # Refuse to signal a process whose identity cannot be inspected. A
        # missing diagnostic job is safer than a signal sent to a reused PID.
        if identity.get("status") != "FOUND" or str(job.get("job_id") or "") not in str(identity.get("ps") or ""):
            row["reason"] = "process identity unavailable; no signal sent"
            paused.append(row)
            continue
        try:
            os.killpg(int(pid), signal.SIGSTOP)
            row["paused"] = True
        except (OSError, ValueError) as exc:
            row["reason"] = f"pause failed: {type(exc).__name__}: {exc}"
        paused.append(row)
    return paused


def _restore_jobs(rows: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    restored: list[Dict[str, Any]] = []
    for item in rows:
        row = dict(item)
        if row.get("paused") and not row.get("restored"):
            try:
                os.killpg(int(row["pid"]), signal.SIGCONT)
                row["restored"] = True
            except (OSError, ValueError) as exc:
                row["restore_error"] = f"{type(exc).__name__}: {exc}"
        restored.append(row)
    return restored


def _classify_blockers(jobs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    blockers: list[Dict[str, Any]] = []
    pausable: list[Dict[str, Any]] = []
    for job in jobs:
        if _job_is_model_lake(job):
            blockers.append({"kind": "MODELLAKE_UNTOUCHABLE", "job": _safe(job)})
        elif _job_is_pausable_hcli(job):
            pausable.append(_safe(job))
        else:
            blockers.append({"kind": "OTHER_RUNNING_JOB", "job": _safe(job)})
    return {"blockers": blockers, "pausable_hcli_jobs": pausable}


def _lock_path(repo: Path) -> Path:
    path = repo / ".hcli" / "locks" / "qwen-protected-bench.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _try_lock(repo: Path) -> Optional[Any]:
    handle = _lock_path(repo).open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def _write_progress(destination: Path, report: Mapping[str, Any]) -> None:
    atomic_write_json(destination, dict(report))


def run_protected_benchmark_watcher(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    profile: Optional[str | os.PathLike[str]] = None,
    resident_binary: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    result_emit: Optional[str | os.PathLike[str]] = None,
    duration_s: float = 6 * 3600.0,
    interval_s: float = 60.0,
    once: bool = False,
    pause_known_jobs: bool = False,
    timeout_s: float = 180.0,
) -> Dict[str, Any]:
    repo = _repo_root(repo_root)
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / DEFAULT_EMIT_NAME
    result_path = Path(result_emit).expanduser().resolve() if result_emit else repo / "receipts" / "headless" / DEFAULT_RESULT_NAME
    started = time.time()
    deadline = started + max(0.1, float(duration_s))
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "WAITING_FOR_QUIESCENCE",
        "qualification": False,
        "NOT_FOR_PROMOTION": True,
        "repo_root": str(repo),
        "profile_path": str(_profile_path(repo, profile)),
        "resident_binary": str(Path(resident_binary).expanduser().resolve()) if resident_binary else None,
        "started_at": started,
        "deadline": deadline,
        "duration_s": max(0.1, float(duration_s)),
        "interval_s": max(0.1, min(60.0, float(interval_s))),
        "once": bool(once),
        "pause_known_jobs": bool(pause_known_jobs),
        "polls": [],
        "runs": [],
        "claim_boundary": "This watcher governs benchmark readiness only. The diagnostic result remains NOT_FOR_PROMOTION; no process outside explicitly identified HCLI benchmark jobs is signaled.",
    }
    lock_handle: Optional[Any] = None
    try:
        while time.time() < deadline:
            now = time.time()
            sample = _watcher_quiescence()
            jobs = _running_jobs(repo)
            job_classes = _classify_blockers(jobs)
            poll: Dict[str, Any] = {
                "at": now,
                "machine": sample,
                "running_jobs": _safe(jobs),
                "job_classes": job_classes,
                "quiet": sample.get("quiet") is True,
                "eligible_without_pause": (
                    sample.get("quiet") is True
                    and not job_classes["blockers"]
                    and (pause_known_jobs or not job_classes["pausable_hcli_jobs"])
                ),
            }
            report["polls"].append(poll)
            _write_progress(destination, report)
            if poll["eligible_without_pause"]:
                lock_handle = _try_lock(repo)
                if lock_handle is None:
                    poll["decision"] = "LOCK_BUSY"
                    if once:
                        break
                else:
                    paused: list[Dict[str, Any]] = []
                    try:
                        # Re-check after acquiring the lock.  ModelLake is
                        # never paused, even when pause_known_jobs is enabled.
                        locked_sample = _watcher_quiescence()
                        locked_jobs = _running_jobs(repo)
                        locked_classes = _classify_blockers(locked_jobs)
                        poll["locked_machine"] = locked_sample
                        poll["locked_job_classes"] = locked_classes
                        locked_eligible = (
                            locked_sample.get("quiet") is True
                            and not locked_classes["blockers"]
                            and (pause_known_jobs or not locked_classes["pausable_hcli_jobs"])
                        )
                        if not locked_eligible:
                            poll["decision"] = "RECHECK_FAILED"
                        else:
                            if pause_known_jobs:
                                paused = _pause_jobs(locked_classes["pausable_hcli_jobs"])
                            poll["paused_jobs"] = paused
                            result = run_qwen27_mlp_diagnostic_ab(
                                repo_root=repo,
                                profile=profile,
                                resident_binary=resident_binary,
                                emit=result_path,
                                timeout_s=timeout_s,
                            )
                            report["runs"].append({
                                "result_receipt": str(result_path),
                                "status": result.get("status"),
                                "benchmark_class": result.get("benchmark_class"),
                                "qualification": result.get("qualification"),
                                "NOT_FOR_PROMOTION": result.get("NOT_FOR_PROMOTION"),
                                "experiment_verdict": result.get("experiment_verdict"),
                            })
                            poll["decision"] = "DIAGNOSTIC_COMPLETED"
                            report["status"] = "COMPLETED" if result.get("status") == "PASSED" else "DIAGNOSTIC_FAILED"
                    finally:
                        if paused:
                            poll["restored_jobs"] = _restore_jobs(paused)
                        _write_progress(destination, report)
                    break
            if once:
                break
            remaining = max(0.0, deadline - time.time())
            if remaining <= 0:
                break
            time.sleep(min(report["interval_s"], remaining))
    except Exception as exc:  # noqa: BLE001 - preserve the wait boundary
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    finally:
        if lock_handle is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                lock_handle.close()
            except OSError:
                pass
    if report.get("status") == "WAITING_FOR_QUIESCENCE" and time.time() >= deadline:
        report["status"] = "WAITING_FOR_QUIESCENCE"
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    report["receipt_path"] = str(destination)
    report["result_receipt_path"] = str(result_path)
    report["last_poll"] = report["polls"][-1] if report["polls"] else None
    report["protected_result_classes"] = [run.get("benchmark_class") for run in report["runs"]]
    report["protected_result_is_not_promotion"] = all(
        run.get("NOT_FOR_PROMOTION") is True for run in report["runs"]
    )
    atomic_write_json(destination, report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--profile")
    parser.add_argument("--resident-binary")
    parser.add_argument("--emit")
    parser.add_argument("--result-emit")
    parser.add_argument("--duration-s", type=float, default=6 * 3600.0)
    parser.add_argument("--interval-s", type=float, default=60.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--pause-known-jobs", action="store_true")
    args = parser.parse_args(argv)
    report = run_protected_benchmark_watcher(
        repo_root=args.repo_root,
        profile=args.profile,
        resident_binary=args.resident_binary,
        emit=args.emit,
        result_emit=args.result_emit,
        duration_s=args.duration_s,
        interval_s=args.interval_s,
        once=args.once,
        pause_known_jobs=args.pause_known_jobs,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") in {"COMPLETED", "WAITING_FOR_QUIESCENCE"} else 1


__all__ = ["SCHEMA", "run_protected_benchmark_watcher", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
