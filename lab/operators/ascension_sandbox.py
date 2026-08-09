"""Detached, fail-closed control plane for the Ascension research sandbox.

This is deliberately a *research* sandbox controller, not a shortcut around
the Ascension Bible's production-manager gate.  It continuously records direct
evidence, runs a small policy-bounded foundation suite, and keeps a durable
heartbeat.  It never loads a model, downloads a body, deletes an artifact,
merges work, or promotes a candidate.

The controller is intended to be launched by the accompanying launchd job.  A
live process therefore has a concrete meaning: the sandbox control plane is
running and monitoring its own admission state.  ``option_c_live`` remains
false until the independently evidenced Qwen executor/reviewer and controller
gates actually pass.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from lab.execution_sandbox import (
    ExecutionSandboxPolicy,
    SandboxAction,
    SandboxPrincipal,
)
from lab.operators.qwen_ascension_preflight import default_blocked_decision
from lab.operators.sandbox_ready_preflight import CONFIG_SCHEMA, evaluate_sandbox_ready
from lab.receipts import SealIntegrityError, seal, verify


SCHEMA = "hawking.ascension.research_sandbox.v1"
EVENT_SCHEMA = "hawking.ascension.research_sandbox_event.v1"
MODE = "PRE_SANDBOX_RESEARCH"
GIB = 1024**3
QWEN30_BODY_RESERVATION_BYTES = 61_063_697_531
QWEN30_PACK_WORKING_RESERVATION_BYTES = 32 * GIB
MAX_EVENT_LOG_BYTES = 1 * 1024 * 1024
FOUNDATION_TEST_SELECTOR = "foundation"
FOUNDATION_TEST_FILES: tuple[str, ...] = (
    "lab/tests/test_execution_sandbox.py",
    "lab/tests/test_option_c.py",
    "lab/tests/test_sandbox_ready_preflight.py",
    "lab/tests/test_qwen_ascension_preflight.py",
    "lab/tests/test_ascension_contracts.py",
    "lab/tests/test_ascension_parity_ladder_scaffold.py",
    "lab/tests/test_ascension_lifecycle.py",
    "lab/tests/test_ascension_source_admission.py",
    "lab/tests/test_ascension_campaign.py",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "workspace" / "campaign" / "records" / "ascension-sandbox"
DEFAULT_BIBLE = REPO_ROOT.parent / "bible.md"
PROTO_ROOT = (
    REPO_ROOT
    / "workspace"
    / "campaign"
    / "evidence"
    / "models"
    / "frankenstein"
    / "proto-v0-seal-refresh"
)


class AscensionSandboxError(RuntimeError):
    """Invalid local controller setup or a singleton conflict."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _tail(text: str, limit: int = 6000) -> str:
    return text if len(text) <= limit else text[-limit:]


def _safe_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _ensure_directory(path: Path, mode: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    node = os.lstat(path)
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
        raise AscensionSandboxError(f"sandbox path must be a real directory: {path}")
    os.chmod(path, mode)


@dataclass(frozen=True)
class SandboxPaths:
    """Owned runtime locations for the detached controller."""

    root: Path
    executor_root: Path
    reviewer_root: Path
    receipts_root: Path
    logs_root: Path
    state_root: Path
    config_path: Path
    status_path: Path
    preflight_path: Path
    event_log_path: Path
    lock_path: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "SandboxPaths":
        resolved = Path(root).expanduser().resolve()
        return cls(
            root=resolved,
            executor_root=resolved / "executor-worktree",
            reviewer_root=resolved / "reviewer-readonly",
            receipts_root=resolved / "receipts",
            logs_root=resolved / "logs",
            state_root=resolved / "state",
            config_path=resolved / "sandbox-ready-config.json",
            status_path=resolved / "state" / "status.json",
            preflight_path=resolved / "receipts" / "sandbox-ready-preflight.json",
            event_log_path=resolved / "logs" / "events.jsonl",
            lock_path=resolved / "state" / "controller.lock",
        )


def build_preflight_config(
    paths: SandboxPaths,
    *,
    repo_root: str | Path = REPO_ROOT,
    bible_path: str | Path = DEFAULT_BIBLE,
) -> dict[str, Any]:
    """Build an honest, conservative preflight configuration.

    The two historical checkpoint directories are intentionally kept in the
    active-envelope check until an operator classifies and removes/offloads
    them.  The controller must not make those paths disappear itself.
    """

    repo = Path(repo_root).expanduser().resolve()
    bible = Path(bible_path).expanduser().resolve()
    proto = repo / "workspace" / "campaign" / "evidence" / "models" / "frankenstein" / "proto-v0-seal-refresh"
    cloud = proto / "cloud-package"
    return {
        "schema": CONFIG_SCHEMA,
        "generated_by": SCHEMA,
        "generated_at": _utc_now(),
        "proto": {
            "terminal_receipt": str(proto / "PROTO_FRANKENSTEIN_V0_TERMINAL_RECEIPT.json"),
            "artifact": str(proto / "PROTO_FRANKENSTEIN_V0_ARTIFACT.json"),
            "independent_verify": str(proto / "PROTO_FRANKENSTEIN_V0_INDEPENDENT_VERIFY.json"),
            "cloud_sealed": str(cloud / "PROTO_CLOUD_SEALED.json"),
            "cloud_manifest": str(cloud / "PROTO_V0_CLOUD_MANIFEST.json"),
            "restore_script": str(cloud / "restore_proto_frankenstein_v0.sh"),
            "active_storage_paths_must_be_absent": [
                str(repo / "workspace" / "campaign" / "evidence" / "models" / "frankenstein" / "latent_v0_checkpoints"),
                str(repo / "workspace" / "campaign" / "evidence" / "models" / "frankenstein" / "bridge_train_real" / "checkpoints"),
            ],
        },
        "sandbox": {
            "root": str(paths.root),
            "executor_worktree_root": str(paths.executor_root),
            "reviewer_readonly_root": str(paths.reviewer_root),
            "receipts_root": str(paths.receipts_root),
            "logs_root": str(paths.logs_root),
            "reviewer_enforcement": "filesystem_readonly",
            "allowed_test_selectors": [FOUNDATION_TEST_SELECTOR],
            "approved_download_ids": [],
        },
        "authority": {
            "required_files": [
                str(bible),
                str(repo / "lab" / "execution_sandbox.py"),
                str(repo / "lab" / "hcli" / "option_c.py"),
                str(repo / "lab" / "operators" / "sandbox_ready_preflight.py"),
            ],
        },
        "resources": {
            "disk_path": str(repo),
            "minimum_free_disk_bytes": 25 * GIB,
            "qwen30_body_reservation_bytes": QWEN30_BODY_RESERVATION_BYTES,
            "qwen30_pack_working_reservation_bytes": QWEN30_PACK_WORKING_RESERVATION_BYTES,
            "process_tree_rss_cap_bytes": 5 * GIB,
            "swap_growth_allowed": False,
        },
        "claim_boundary": {
            "configuration_is_not_model_admission": True,
            "configuration_is_not_cloud_upload_proof": True,
            "controller_never_deletes_active_envelope_paths": True,
            "approved_download_ids_must_stay_empty_until_controller_admission": True,
        },
    }


def bootstrap_layout(
    root: str | Path = DEFAULT_ROOT,
    *,
    repo_root: str | Path = REPO_ROOT,
    bible_path: str | Path = DEFAULT_BIBLE,
    force_config: bool = False,
) -> SandboxPaths:
    """Create the controller-owned layout without touching model artifacts."""

    paths = SandboxPaths.from_root(root)
    _ensure_directory(paths.root, 0o750)
    _ensure_directory(paths.executor_root, 0o750)
    _ensure_directory(paths.reviewer_root, 0o555)
    _ensure_directory(paths.receipts_root, 0o750)
    _ensure_directory(paths.logs_root, 0o750)
    _ensure_directory(paths.state_root, 0o750)
    if force_config or not paths.config_path.exists():
        _atomic_json(
            paths.config_path,
            build_preflight_config(paths, repo_root=repo_root, bible_path=bible_path),
        )
    return paths


def _sealed_evidence(path: Path) -> dict[str, Any]:
    payload = _safe_json(path)
    if payload is None:
        return {"path": str(path), "exists": path.exists(), "seal_valid": False, "reason": "missing_or_unreadable"}
    try:
        checked = verify(payload, label=str(path))
    except SealIntegrityError as exc:
        return {
            "path": str(path),
            "exists": True,
            "seal_valid": False,
            "reason": str(exc),
            "schema": payload.get("schema"),
        }
    return {
        "path": str(path),
        "exists": True,
        "seal_valid": True,
        "schema": checked.get("schema"),
        "status": checked.get("status"),
        "endpoint": checked.get("terminal_endpoint", checked.get("endpoint")),
        "seal_sha256": checked.get("seal_sha256"),
    }


def evidence_retractions(
    config: Mapping[str, Any], *, repo_root: str | Path = REPO_ROOT
) -> list[dict[str, Any]]:
    """Find sealed retractions that make otherwise well-formed evidence unusable.

    A valid seal only proves that a document was not altered.  It does not make
    the document admissible if a later sealed retraction says the underlying
    measurements were synthetic or otherwise invalid.  The search is bounded
    to the active terminal-receipt lineage and its known real-fix sibling; it
    never treats an arbitrary filename elsewhere in the repository as a veto.
    """

    proto = config.get("proto") if isinstance(config.get("proto"), Mapping) else {}
    terminal_value = proto.get("terminal_receipt")
    candidates: list[Path] = []
    if isinstance(terminal_value, str) and terminal_value:
        candidates.append(Path(terminal_value).expanduser().parent / "RETRACTED.json")
    candidates.append(
        Path(repo_root).expanduser().resolve()
        / "workspace"
        / "campaign"
        / "evidence"
        / "models"
        / "frankenstein"
        / "proto-v0-real-fix"
        / "RETRACTED.json"
    )
    seen: set[Path] = set()
    observed: list[dict[str, Any]] = []
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        payload = _safe_json(path)
        if payload is None:
            continue
        try:
            checked = verify(payload, label=str(path))
        except SealIntegrityError as exc:
            observed.append(
                {
                    "path": str(path),
                    "seal_valid": False,
                    "status": payload.get("status"),
                    "reason": str(exc),
                }
            )
            continue
        observed.append(
            {
                "path": str(path),
                "seal_valid": True,
                "schema": checked.get("schema"),
                "status": checked.get("status"),
                "must_not_gate": checked.get("must_not_gate") is True,
                "summary": checked.get("summary"),
                "seal_sha256": checked.get("seal_sha256"),
            }
        )
    return observed


def apply_evidence_retractions(
    preflight: Mapping[str, Any], retractions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Fail closed when a sealed, relevant receipt says it must not gate work."""

    blocking = [
        item
        for item in retractions
        if item.get("seal_valid") is True
        and item.get("status") == "RETRACTED"
        and item.get("must_not_gate") is True
    ]
    if not blocking:
        return dict(preflight)
    body = {key: value for key, value in preflight.items() if key != "seal_sha256"}
    existing = list(body.get("blockers") or [])
    for item in blocking:
        existing.append(
            "retracted Proto evidence must not gate admission: "
            f"{item.get('path')}"
        )
    body["status"] = "BLOCKED"
    body["sandbox_foundation_preflight_ready"] = False
    body["qwen30_body_admission_candidate"] = False
    body["proto_terminal_claimed"] = False
    body["blockers"] = list(dict.fromkeys(str(reason) for reason in existing if reason))
    body["evidence_retractions"] = [dict(item) for item in retractions]
    boundary = dict(body.get("claim_boundary") or {})
    boundary["retracted_evidence_is_not_admission_authority"] = True
    body["claim_boundary"] = boundary
    return seal(body)


def _evidence_inventory(
    config: Mapping[str, Any], *, retractions: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    proto = config.get("proto") if isinstance(config.get("proto"), Mapping) else {}
    paths = {
        "terminal": proto.get("terminal_receipt"),
        "artifact": proto.get("artifact"),
        "independent_verify": proto.get("independent_verify"),
        "cloud_sealed": proto.get("cloud_sealed"),
        "cloud_manifest": proto.get("cloud_manifest"),
    }
    inventory = {
        name: _sealed_evidence(Path(str(value)).expanduser())
        for name, value in paths.items()
        if isinstance(value, str) and value
    }
    inventory["retractions"] = [dict(item) for item in retractions]
    return inventory


def _run_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return evaluate_sandbox_ready(config)
    except Exception as exc:  # An unreadable config must become evidence, not a daemon crash.
        return seal(
            {
                "schema": "hawking.ascension.sandbox_ready_preflight.v1",
                "recorded_at": _utc_now(),
                "status": "BLOCKED",
                "sandbox_foundation_preflight_ready": False,
                "blockers": [f"preflight evaluator error: {exc}"],
                "claim_boundary": {
                    "preflight_not_terminal_certification": True,
                    "preflight_not_qwen_download_authorization": True,
                },
            }
        )


def _foundation_policy(paths: SandboxPaths) -> ExecutionSandboxPolicy:
    return ExecutionSandboxPolicy(
        owned_worktree_roots=(str(paths.executor_root),),
        sandbox_root=str(paths.root),
        allowed_test_selectors=frozenset({FOUNDATION_TEST_SELECTOR}),
        approved_download_ids=frozenset(),
    )


def run_foundation_suite(
    paths: SandboxPaths,
    *,
    repo_root: str | Path = REPO_ROOT,
    python_executable: str | Path = sys.executable,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """Run only the small, policy-approved foundation suite.

    The selector is checked as the sandbox-model principal before the protected
    controller starts the subprocess.  This makes the suite a real test of the
    policy surface while keeping it separate from model execution.
    """

    policy = _foundation_policy(paths)
    decision = policy.authorize(
        SandboxPrincipal.SANDBOX_MODEL,
        SandboxAction.RUN_ALLOWED_TESTS,
        target=paths.executor_root,
        context={"test_selector": FOUNDATION_TEST_SELECTOR},
    )
    if not decision.allowed:
        return {
            "status": "POLICY_BLOCKED",
            "recorded_at": _utc_now(),
            "policy": decision.to_dict(),
        }

    repo = Path(repo_root).expanduser().resolve()
    command = [
        str(python_executable),
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *FOUNDATION_TEST_FILES,
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(repo),
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        combined = (exc.stdout or "") + (exc.stderr or "")
        return {
            "status": "TIMEOUT",
            "recorded_at": _utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "timeout_seconds": timeout_seconds,
            "command": command,
            "policy": decision.to_dict(),
            "output_tail": _tail(combined),
        }
    output = (completed.stdout or "") + (completed.stderr or "")
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "recorded_at": _utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "exit_code": completed.returncode,
        "command": command,
        "policy": decision.to_dict(),
        "output_tail": _tail(output),
    }


def _v3_campaign_inventory(paths: SandboxPaths) -> dict[str, Any]:
    """Read the separate V3 lifecycle/supervisor without giving it authority.

    The research-sandbox daemon and the full campaign supervisor have separate
    singleton locks.  This read-only inventory makes their relationship visible
    in one heartbeat while avoiding concurrent lifecycle mutation.
    """

    lifecycle_root = paths.root / "lifecycle"
    documents = {
        "lifecycle_state": lifecycle_root / "ASCENSION_V3_STATE.json",
        "campaign_supervisor": lifecycle_root / "ASCENSION_V3_CAMPAIGN_SUPERVISOR.json",
        "execution_manifest": lifecycle_root / "ASCENSION_V3_EXECUTION_MANIFEST.json",
    }
    inventory: dict[str, Any] = {"root": str(lifecycle_root), "documents": {}}
    for key, path in documents.items():
        payload = _safe_json(path)
        if payload is None:
            inventory["documents"][key] = {"path": str(path), "status": "ABSENT", "seal_valid": False}
            continue
        try:
            verified = verify(payload, label=str(path))
        except SealIntegrityError as exc:
            inventory["documents"][key] = {
                "path": str(path),
                "status": "INVALID",
                "seal_valid": False,
                "reason": str(exc),
            }
            continue
        inventory["documents"][key] = {
            "path": str(path),
            "status": verified.get("state", verified.get("status", "PRESENT")),
            "seal_valid": True,
            "recorded_at": verified.get("recorded_at"),
            "seal_sha256": verified.get("seal_sha256"),
            "first_unmet_state": verified.get("first_unmet_state"),
        }
    lifecycle = _safe_json(documents["lifecycle_state"])
    supervisor = _safe_json(documents["campaign_supervisor"])
    inventory["first_unmet_state"] = lifecycle.get("first_unmet_state") if lifecycle else None
    inventory["tournament"] = lifecycle.get("tournament") if lifecycle else None
    inventory["supervisor_heartbeat"] = supervisor.get("heartbeat") if supervisor else None
    return inventory


def _production_gate_report(
    preflight: Mapping[str, Any], qwen: Mapping[str, Any], campaign: Mapping[str, Any]
) -> dict[str, Any]:
    preflight_ready = preflight.get("sandbox_foundation_preflight_ready") is True
    qwen_ready = qwen.get("status") == "ADMITTED" and qwen.get("download_permitted") is True
    blockers = list(preflight.get("blockers") or [])
    blockers.extend(
        [
            "Qwen3-Coder-30B executor has no source-bound, measured, controller-certified qualification",
            "Qwen3-Coder-Next reviewer has no source-bound, measured, controller-certified qualification",
            "manager tournament and alternate-manager offload have no independent evidence",
            "Option-C protected parity, held-out capability, and CLEAN benchmark adapters remain unearned",
        ]
    )
    first_unmet = campaign.get("first_unmet_state")
    if isinstance(first_unmet, str) and first_unmet:
        blockers.append(f"V3 lifecycle first unmet state: {first_unmet}")
    tournament = campaign.get("tournament")
    if isinstance(tournament, Mapping) and tournament.get("status"):
        blockers.append(f"protected tournament controller: {tournament.get('status')}")
    return {
        "research_sandbox_active": True,
        "option_c_live": False,
        "production_sandbox_active": False,
        "status": "BLOCKED_UNQUALIFIED_MANAGERS",
        "foundation_preflight_ready": preflight_ready,
        "qwen_metadata_preflight_admitted": qwen_ready,
        "blockers": list(dict.fromkeys(str(item) for item in blockers if item)),
        "claim_boundary": {
            "running_control_plane_is_not_option_c_live": True,
            "running_control_plane_is_not_production_sandbox_activation": True,
            "no_model_loads": True,
            "no_body_downloads": True,
            "no_deletions": True,
            "no_promotions": True,
        },
    }


def _append_event(paths: SandboxPaths, status: Mapping[str, Any]) -> None:
    event_path = paths.event_log_path
    if event_path.exists() and event_path.stat().st_size >= MAX_EVENT_LOG_BYTES:
        previous = event_path.with_name("events.previous.jsonl")
        os.replace(event_path, previous)
    event = {
        "schema": EVENT_SCHEMA,
        "recorded_at": status.get("recorded_at"),
        "heartbeat": status.get("heartbeat"),
        "state": status.get("state"),
        "mode": status.get("mode"),
        "foundation_preflight_status": (status.get("preflight") or {}).get("status"),
        "foundation_tests_status": (status.get("foundation_tests") or {}).get("status"),
    }
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()


def tick(
    root: str | Path = DEFAULT_ROOT,
    *,
    repo_root: str | Path = REPO_ROOT,
    bible_path: str | Path = DEFAULT_BIBLE,
    run_tests: bool = False,
    python_executable: str | Path = sys.executable,
) -> dict[str, Any]:
    """Emit one durable research-sandbox heartbeat and direct gate audit."""

    paths = bootstrap_layout(root, repo_root=repo_root, bible_path=bible_path)
    config = _safe_json(paths.config_path)
    if config is None:
        raise AscensionSandboxError(f"sandbox config unreadable: {paths.config_path}")
    previous = _safe_json(paths.status_path) or {}
    retractions = evidence_retractions(config, repo_root=repo_root)
    preflight = apply_evidence_retractions(_run_preflight(config), retractions)
    _atomic_json(paths.preflight_path, preflight)
    qwen = default_blocked_decision()
    campaign = _v3_campaign_inventory(paths)
    test_result = previous.get("foundation_tests")
    if run_tests:
        test_result = run_foundation_suite(
            paths,
            repo_root=repo_root,
            python_executable=python_executable,
        )
    if not isinstance(test_result, Mapping):
        test_result = {"status": "NOT_RUN", "reason": "awaiting first bounded foundation suite"}

    heartbeat = previous.get("heartbeat")
    heartbeat_count = int(heartbeat) + 1 if isinstance(heartbeat, int) else 1
    test_status = str(test_result.get("status") or "NOT_RUN")
    state = "RUNNING" if test_status in {"PASS", "NOT_RUN"} else "RUNNING_DEGRADED"
    status = seal(
        {
            "schema": SCHEMA,
            "recorded_at": _utc_now(),
            "state": state,
            "mode": MODE,
            "heartbeat": heartbeat_count,
            "process": {
                "pid": os.getpid(),
                "parent_pid": os.getppid(),
                "process_group": os.getpgrp(),
                "python_executable": str(python_executable),
            },
            "paths": {
                "root": str(paths.root),
                "config": str(paths.config_path),
                "status": str(paths.status_path),
                "preflight_receipt": str(paths.preflight_path),
                "events": str(paths.event_log_path),
            },
            "evidence_inventory": _evidence_inventory(config, retractions=retractions),
            "preflight": preflight,
            "qwen_metadata_preflight": qwen,
            "v3_campaign": campaign,
            "production_gate_report": _production_gate_report(preflight, qwen, campaign),
            "foundation_tests": dict(test_result),
            "resource_snapshot": {
                "disk_path": str(repo_root),
                "free_disk_bytes": shutil.disk_usage(Path(repo_root)).free,
                "minimum_free_disk_bytes": 25 * GIB,
            },
            "claim_boundary": {
                "active": True,
                "research_control_plane_only": True,
                "does_not_claim_option_c_live": True,
                "does_not_claim_production_sandbox_active": True,
                "does_not_load_models": True,
                "does_not_download_model_bodies": True,
                "does_not_delete_or_evict": True,
                "does_not_merge_or_promote": True,
            },
        }
    )
    _atomic_json(paths.status_path, status)
    _append_event(paths, status)
    return status


@contextlib.contextmanager
def _exclusive_watch_lock(paths: SandboxPaths) -> Iterator[None]:
    paths.lock_path.touch(exist_ok=True)
    with paths.lock_path.open("r+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AscensionSandboxError(
                f"another ascension sandbox controller already owns {paths.lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def watch(
    root: str | Path = DEFAULT_ROOT,
    *,
    repo_root: str | Path = REPO_ROOT,
    bible_path: str | Path = DEFAULT_BIBLE,
    interval_seconds: float = 30.0,
    python_executable: str | Path = sys.executable,
) -> int:
    """Keep the bounded research-control plane detached and restart-safe."""

    if not 5.0 <= float(interval_seconds) <= 3600.0:
        raise AscensionSandboxError("interval_seconds must be between 5 and 3600")
    paths = bootstrap_layout(root, repo_root=repo_root, bible_path=bible_path)
    stopping = False

    def request_stop(_signal: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    old_term = signal.signal(signal.SIGTERM, request_stop)
    old_int = signal.signal(signal.SIGINT, request_stop)
    try:
        with _exclusive_watch_lock(paths):
            first_tick = True
            while not stopping:
                status = tick(
                    root,
                    repo_root=repo_root,
                    bible_path=bible_path,
                    run_tests=first_tick,
                    python_executable=python_executable,
                )
                print(
                    json.dumps(
                        {
                            "recorded_at": status["recorded_at"],
                            "state": status["state"],
                            "heartbeat": status["heartbeat"],
                            "foundation_preflight": status["preflight"].get("status"),
                            "foundation_tests": status["foundation_tests"].get("status"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                first_tick = False
                deadline = time.monotonic() + float(interval_seconds)
                while not stopping and time.monotonic() < deadline:
                    time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(item: argparse.ArgumentParser) -> None:
        item.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        item.add_argument("--repo-root", type=Path, default=REPO_ROOT)
        item.add_argument("--bible", type=Path, default=DEFAULT_BIBLE)

    init = sub.add_parser("init", help="create only the sandbox-owned layout and config")
    common(init)
    init.add_argument("--force-config", action="store_true")

    once = sub.add_parser("tick", help="record one direct audit and heartbeat")
    common(once)
    once.add_argument("--run-tests", action="store_true")

    daemon = sub.add_parser("watch", help="run the detached research-control loop")
    common(daemon)
    daemon.add_argument("--interval-seconds", type=float, default=30.0)

    status = sub.add_parser("status", help="print the most recent heartbeat")
    status.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        paths = bootstrap_layout(
            args.root,
            repo_root=args.repo_root,
            bible_path=args.bible,
            force_config=args.force_config,
        )
        print(json.dumps({"root": str(paths.root), "config": str(paths.config_path)}, sort_keys=True))
        return 0
    if args.command == "tick":
        result = tick(
            args.root,
            repo_root=args.repo_root,
            bible_path=args.bible,
            run_tests=args.run_tests,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "watch":
        return watch(
            args.root,
            repo_root=args.repo_root,
            bible_path=args.bible,
            interval_seconds=args.interval_seconds,
        )
    if args.command == "status":
        paths = SandboxPaths.from_root(args.root)
        result = _safe_json(paths.status_path)
        if result is None:
            print(json.dumps({"state": "ABSENT", "status_path": str(paths.status_path)}))
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
