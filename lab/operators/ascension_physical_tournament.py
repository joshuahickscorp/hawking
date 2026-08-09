"""Protected, deterministic physical-manager tournament handoff.

This module is deliberately small and has one authority boundary:

* the gatekeeper may freeze the exact suite and start this process only after
  both independently sealed manager qualification chains pass;
* this runner may execute the frozen comparison, but can never select a
  winner, activate a sandbox, or mutate a model artifact.

It is not a replacement lifecycle controller.  It turns the existing physical
gate's final handoff into a restart-safe, exactly-once operation and keeps the
hidden evaluator material out of candidate-facing HCLI requests except for the
individual prompt currently being evaluated.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from lab.operators.ascension_lifecycle import TOURNAMENT_DIMENSIONS
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = _IMPORT_ROOT
DEFAULT_PHYSICAL_ROOT = (
    REPO_ROOT / "workspace" / "campaign" / "records" / "ascension-sandbox" / "physical"
)

SUITE_SCHEMA = "hawking.ascension.physical_tournament_suite_preflight.v1"
SUITE_STATUS = "PASS_FROZEN_PROTECTED_TOURNAMENT_SUITE_PREFLIGHT"
LAUNCH_SCHEMA = "hawking.ascension.physical_tournament_launch_receipt.v1"
LAUNCH_REQUESTED = "MANAGER_TOURNAMENT_LAUNCH_REQUESTED"
LAUNCH_FAILED = "MANAGER_TOURNAMENT_LAUNCH_FAILED"
RUNNER_SCHEMA = "hawking.ascension.physical_tournament_runner_state.v1"
RUNNING = "MANAGER_TOURNAMENT_RUNNING"
COMPLETE = "MANAGER_TOURNAMENT_COMPLETE_HUMAN_DECISION_REQUIRED"
ABORTED = "MANAGER_TOURNAMENT_ABORTED_FAIL_CLOSED"

SUITE_FILENAME = "ASCENSION_PHYSICAL_TOURNAMENT_SUITE_PREFLIGHT.json"
LAUNCH_FILENAME = "ASCENSION_PHYSICAL_TOURNAMENT_LAUNCH_RECEIPT.json"
RUNNER_FILENAME = "ASCENSION_PHYSICAL_TOURNAMENT_RUNNER_STATE.json"
LOCK_FILENAME = ".ascension-physical-tournament.lock"

CATALOG_PATH = REPO_ROOT / "workspace" / "docs" / "plans" / "ascension" / "ASCENSION_HCLI_PRODUCT_TEST_CATALOG.json"
HIDDEN_COMMITMENT_PATH = (
    REPO_ROOT
    / "workspace"
    / "campaign"
    / "governance"
    / "odyssey"
    / "program"
    / "evaluation"
    / "hidden"
    / "HIDDEN_MEMBERSHIP_COMMITMENT.json"
)
HIDDEN_ITEMS_PATH = HIDDEN_COMMITMENT_PATH.with_name("hidden_items.jsonl")
TRAINING_VISIBLE_PATH = (
    REPO_ROOT
    / "workspace"
    / "campaign"
    / "governance"
    / "odyssey"
    / "program"
    / "training"
    / "TRAINING_VISIBLE_EVAL.json"
)
HIDDEN_POLICY_PATH = REPO_ROOT / "tools" / "odyssey" / "hidden_memberships.py"
HCLI_HARNESS_PATH = REPO_ROOT / "tools" / "condense" / "hcli_product_test_harness.py"

EXPECTED_CANDIDATES = (
    "Qwen30-Gravity-Manager-Artifact",
    "Qwen80-Gravity-Manager-Artifact",
)
EXPECTED_CASE_IDS = (
    "chat",
    "repo_context",
    "coding",
    "planner_act_verify",
    "tool_calls",
    "structured_json",
    "session_restart",
    "endpoint_restart",
    "context_compaction",
    "read_safe_swarm",
    "isolated_write_agent",
    "continuous_batching",
    "search_retrieval",
    "memory_ops",
    "skill_execution",
    "document_perception",
)
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 180.0


class PhysicalTournamentError(RuntimeError):
    """Raised only when a launch/preflight request is unsafe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    raw = value if isinstance(value, (bytes, bytearray)) else _canonical(value)
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _regular_file(path: Path, *, label: str, max_bytes: int = MAX_RECEIPT_BYTES) -> bytes:
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise PhysicalTournamentError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise PhysicalTournamentError(f"{label} must be a regular non-symlink file: {path}")
    if observed.st_size > max_bytes:
        raise PhysicalTournamentError(f"{label} exceeds {max_bytes} byte safety limit")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PhysicalTournamentError(f"cannot read {label}: {exc}") from exc


def _load_json(path: Path, *, label: str, max_bytes: int = MAX_RECEIPT_BYTES) -> dict[str, Any]:
    raw = _regular_file(path, label=label, max_bytes=max_bytes)
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhysicalTournamentError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise PhysicalTournamentError(f"{label} root must be an object")
    return dict(document)


def _load_sealed(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return verify(_load_json(path, label=label), label=label)
    except SealIntegrityError as exc:
        raise PhysicalTournamentError(str(exc)) from exc


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _paths(root: Path) -> dict[str, Path]:
    tournament_root = root / "tournament"
    lifecycle = root / "lifecycle"
    return {
        "suite": tournament_root / SUITE_FILENAME,
        "launch": lifecycle / LAUNCH_FILENAME,
        "runner": lifecycle / RUNNER_FILENAME,
        "lock": lifecycle / LOCK_FILENAME,
        "stdout": lifecycle / "tournament-runner.stdout.log",
        "stderr": lifecycle / "tournament-runner.stderr.log",
    }


def _canonical_jsonl_digest(path: Path) -> tuple[str, int]:
    """Verify hidden membership without persisting item ids or prompt text."""

    raw = _regular_file(path, label="hidden membership items")
    digest = hashlib.sha256()
    count = 0
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise PhysicalTournamentError(f"hidden membership is not UTF-8: {exc}") from exc
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PhysicalTournamentError(f"hidden membership JSONL is invalid: {exc}") from exc
        if not isinstance(row, Mapping) or row.get("set") != "hidden":
            raise PhysicalTournamentError("hidden membership contains a non-hidden row")
        digest.update(_canonical(dict(row)))
        digest.update(b"\n")
        count += 1
    if count <= 0:
        raise PhysicalTournamentError("hidden membership has no rows")
    return digest.hexdigest(), count


def _contains_hidden_id(value: Any) -> bool:
    if isinstance(value, str):
        return "hid_" in value.lower()
    if isinstance(value, Mapping):
        return any(_contains_hidden_id(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_hidden_id(item) for item in value)
    return False


def _suite_inputs() -> dict[str, Any]:
    """Return only deterministic, non-secret frozen suite facts."""

    catalog_raw = _regular_file(CATALOG_PATH, label="HCLI product catalog")
    catalog = _load_json(CATALOG_PATH, label="HCLI product catalog")
    cases = catalog.get("cases")
    if not isinstance(cases, list):
        raise PhysicalTournamentError("HCLI product catalog cases must be a list")
    case_ids = tuple(row.get("id") for row in cases if isinstance(row, Mapping))
    if case_ids != EXPECTED_CASE_IDS:
        raise PhysicalTournamentError("HCLI product catalog differs from the fixed physical suite")
    if catalog.get("primary_metric") != "verified_tasks_completed_per_hour":
        raise PhysicalTournamentError("HCLI product catalog primary metric drifted")

    commitment_raw = _regular_file(HIDDEN_COMMITMENT_PATH, label="hidden membership commitment")
    commitment = _load_json(HIDDEN_COMMITMENT_PATH, label="hidden membership commitment")
    expected_commitment = commitment.get("commitment_sha256")
    if not _is_sha256(expected_commitment):
        raise PhysicalTournamentError("hidden membership commitment SHA-256 is invalid")
    recomputed_commitment, hidden_count = _canonical_jsonl_digest(HIDDEN_ITEMS_PATH)
    if recomputed_commitment != expected_commitment:
        raise PhysicalTournamentError("hidden membership does not match its commitment")
    if commitment.get("n_hidden") != hidden_count:
        raise PhysicalTournamentError("hidden membership count differs from its commitment")

    training = _load_json(TRAINING_VISIBLE_PATH, label="training-visible membership surface")
    if training.get("hidden_item_ids_visible") is not False or _contains_hidden_id(training):
        raise PhysicalTournamentError("training-visible surface leaks hidden membership ids")
    if training.get("hidden_commitment_sha256") != expected_commitment:
        raise PhysicalTournamentError("training-visible surface does not bind hidden commitment")

    fixed_files = {
        "runner": Path(__file__),
        "hcli_product_harness": HCLI_HARNESS_PATH,
        "hidden_membership_policy": HIDDEN_POLICY_PATH,
    }
    file_hashes: dict[str, dict[str, Any]] = {}
    for label, path in fixed_files.items():
        raw = _regular_file(path, label=label)
        file_hashes[label] = {
            "path": str(path),
            "sha256": _digest(raw),
            "bytes": len(raw),
        }
    environment = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "endpoint_contract": {
            "protocol": "openai_chat_completions_v1",
            "loopback_only": True,
            "health_path": "/healthz",
            "chat_path": "/v1/chat/completions",
            "candidate_tool_policy": "no_host_shell_or_hidden_membership_files",
            "candidate_receives_only_current_evaluation_prompt": True,
        },
        "fixed_files": file_hashes,
    }
    return {
        "catalog": {
            "path": str(CATALOG_PATH),
            "document_sha256": _digest(catalog_raw),
            "case_ids": list(case_ids),
            "case_count": len(case_ids),
        },
        "hidden_membership": {
            "commitment_path": str(HIDDEN_COMMITMENT_PATH),
            "commitment_document_sha256": _digest(commitment_raw),
            "commitment_sha256": expected_commitment,
            "hidden_count": hidden_count,
            "training_visible_path": str(TRAINING_VISIBLE_PATH),
            "training_surface_contains_no_hidden_ids": True,
            "candidate_visible_surface_excludes_hidden_member_ids": True,
        },
        "environment": environment,
        "environment_sha256": _digest(environment),
    }


def build_suite_preflight(root: str | Path) -> dict[str, Any]:
    """Build a sealed exact-suite identity; it never runs a candidate model."""

    resolved = Path(root).expanduser().resolve()
    inputs = _suite_inputs()
    return seal(
        {
            "schema": SUITE_SCHEMA,
            "status": SUITE_STATUS,
            "recorded_at": _utc_now(),
            "physical_root": str(resolved),
            "fixed_candidate_order": list(EXPECTED_CANDIDATES),
            "frozen_task_set": inputs["catalog"],
            "hidden_membership": inputs["hidden_membership"],
            "deterministic_scoring": {
                "dimensions": list(TOURNAMENT_DIMENSIONS),
                "primary_metric": "verified_tasks_completed_per_hour",
                "hidden_prompt_scoring": "case-insensitive required-token containment",
                "candidate_order_is_fixed": True,
                "winner_selection": "DISABLED",
                "ties_are_not_auto_resolved": True,
            },
            "protected_evaluator": {
                "runner_source_path": inputs["environment"]["fixed_files"]["runner"]["path"],
                "runner_source_sha256": inputs["environment"]["fixed_files"]["runner"]["sha256"],
                "candidate_processes_do_not_receive_hidden_membership_paths": True,
                "raw_prompt_and_completion_text_are_not_written_to_receipts": True,
            },
            "tool_environment": inputs["environment"],
            "tool_environment_sha256": inputs["environment_sha256"],
            "resource_accounting": {
                "wall_clock": True,
                "process_rusage": True,
                "disk_free_before_after": True,
                "endpoint_request_latency": True,
            },
            "recovery_contract": {
                "each_final_manager_operations_receipt_requires_restart_rollback_and_session_evidence": True,
                "runner_rechecks_endpoint_health_before_each_candidate": True,
                "endpoint_failure_aborts_without_winner": True,
            },
            "claim_boundary": {
                "preflight_is_not_a_model_runtime_or_tps_result": True,
                "preflight_does_not_run_or_score_candidates": True,
                "preflight_does_not_select_a_winner": True,
                "preflight_does_not_activate_sandbox": True,
            },
        }
    )


def freeze_suite_preflight(root: str | Path = DEFAULT_PHYSICAL_ROOT) -> dict[str, Any]:
    """Write the one concrete suite-preparation receipt requested before launch."""

    resolved = Path(root).expanduser().resolve()
    document = build_suite_preflight(resolved)
    _atomic_json(_paths(resolved)["suite"], document)
    return document


def validate_suite_preflight(root: str | Path) -> dict[str, Any]:
    """Recompute all mutable suite identities; any drift blocks launch."""

    resolved = Path(root).expanduser().resolve()
    path = _paths(resolved)["suite"]
    reasons: list[str] = []
    document: dict[str, Any] | None = None
    seal_value: str | None = None
    try:
        document = _load_sealed(path, label="physical tournament suite preflight")
        seal_value = str(document.get("seal_sha256")) if _is_sha256(document.get("seal_sha256")) else None
        if document.get("schema") != SUITE_SCHEMA:
            reasons.append("unexpected tournament suite preflight schema")
        if document.get("status") != SUITE_STATUS:
            reasons.append("tournament suite preflight is not passed")
        if document.get("physical_root") != str(resolved):
            reasons.append("tournament suite preflight root does not match")
        if document.get("fixed_candidate_order") != list(EXPECTED_CANDIDATES):
            reasons.append("tournament suite candidate order drifted")
        current = _suite_inputs()
        expected_fields = {
            "frozen_task_set": current["catalog"],
            "hidden_membership": current["hidden_membership"],
            "tool_environment": current["environment"],
            "tool_environment_sha256": current["environment_sha256"],
        }
        for field, expected in expected_fields.items():
            if document.get(field) != expected:
                reasons.append(f"tournament suite {field} no longer matches current fixed input")
    except PhysicalTournamentError as exc:
        reasons.append(str(exc))
    return {
        "path": path,
        "document": document,
        "seal_sha256": seal_value,
        "passed": not reasons,
        "reasons": reasons,
        "details": {
            "expected_schema": SUITE_SCHEMA,
            "expected_status": SUITE_STATUS,
            "frozen_task_catalog": str(CATALOG_PATH),
            "hidden_membership_commitment": str(HIDDEN_COMMITMENT_PATH),
            "winner_selection": "DISABLED",
        },
    }


def _read_launch(root: Path) -> dict[str, Any] | None:
    path = _paths(root)["launch"]
    if not path.exists():
        return None
    try:
        return _load_sealed(path, label="physical tournament launch receipt")
    except PhysicalTournamentError:
        return None


def _read_runner(root: Path) -> dict[str, Any] | None:
    path = _paths(root)["runner"]
    if not path.exists():
        return None
    try:
        return _load_sealed(path, label="physical tournament runner state")
    except PhysicalTournamentError:
        return None


def _pid_alive(value: Any) -> bool:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def launch_state(root: str | Path, *, qualification_fingerprint: str | None) -> dict[str, Any]:
    """Return a conservative public handoff state for the gatekeeper."""

    resolved = Path(root).expanduser().resolve()
    launch = _read_launch(resolved)
    runner = _read_runner(resolved)
    result: dict[str, Any] = {
        "state": "NOT_LAUNCHED",
        "launch": launch,
        "runner": runner,
        "reasons": [],
    }
    if launch is None:
        return result
    if launch.get("schema") != LAUNCH_SCHEMA or launch.get("qualification_fingerprint") != qualification_fingerprint:
        result["state"] = "INVALID_OR_STALE_LAUNCH_RECEIPT"
        result["reasons"].append("launch receipt does not bind current qualification fingerprint")
        return result
    if launch.get("winner_selection") != "DISABLED" or launch.get("winner") is not None:
        result["state"] = "INVALID_OR_STALE_LAUNCH_RECEIPT"
        result["reasons"].append("launch receipt attempts winner selection")
        return result
    if launch.get("status") == LAUNCH_FAILED:
        result["state"] = "LAUNCH_FAILED_FAIL_CLOSED"
        return result
    if launch.get("status") != LAUNCH_REQUESTED:
        result["state"] = "INVALID_OR_STALE_LAUNCH_RECEIPT"
        result["reasons"].append("launch receipt has an unrecognized status")
        return result
    if runner is None:
        runner_pid = _mapping(launch.get("runner")).get("pid")
        if isinstance(runner_pid, int) and not isinstance(runner_pid, bool) and not _pid_alive(runner_pid):
            result["state"] = "LAUNCH_FAILED_FAIL_CLOSED"
            result["reasons"].append("detached tournament runner exited before writing a sealed state")
            return result
        result["state"] = "QUALIFICATIONS_COMPLETE"
        return result
    if (
        runner.get("schema") != RUNNER_SCHEMA
        or runner.get("launch_id") != launch.get("launch_id")
        or runner.get("qualification_fingerprint") != qualification_fingerprint
        or runner.get("winner_selection") != "DISABLED"
        or runner.get("winner") is not None
    ):
        result["state"] = "INVALID_OR_STALE_RUNNER_STATE"
        result["reasons"].append("runner state does not bind launch without winner selection")
        return result
    status = runner.get("status")
    if status == RUNNING:
        result["state"] = RUNNING
    elif status == COMPLETE:
        result["state"] = COMPLETE
    elif status == ABORTED:
        result["state"] = ABORTED
    else:
        result["state"] = "INVALID_OR_STALE_RUNNER_STATE"
        result["reasons"].append("runner state has an unrecognized status")
    return result


def _check_endpoint(endpoint: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if endpoint.get("protocol") != "openai_chat_completions_v1":
        reasons.append("manager endpoint protocol is not openai_chat_completions_v1")
    if endpoint.get("host") != "127.0.0.1":
        reasons.append("manager endpoint must be loopback 127.0.0.1")
    port = endpoint.get("port")
    if not isinstance(port, int) or isinstance(port, bool) or not (1024 <= port <= 65535):
        reasons.append("manager endpoint port must be a non-privileged integer")
    if endpoint.get("health_path") != "/healthz":
        reasons.append("manager endpoint health path must be /healthz")
    if endpoint.get("chat_path") != "/v1/chat/completions":
        reasons.append("manager endpoint chat path must be /v1/chat/completions")
    if not isinstance(endpoint.get("model"), str) or not endpoint.get("model"):
        reasons.append("manager endpoint model must be non-empty")
    return reasons


def request_launch(
    root: str | Path,
    *,
    qualification_fingerprint: str,
    qualification_evidence: Mapping[str, Any],
    suite_seal_sha256: str,
    endpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Persist a unique launch request and detach the exact runner.

    The request is written before spawning.  A watcher restart therefore sees
    the same ``launch_id`` and cannot spawn a second tournament for the same
    qualification chain.  A spawn failure is sealed and remains fail-closed.
    """

    resolved = Path(root).expanduser().resolve()
    if not _is_sha256(qualification_fingerprint) or not _is_sha256(suite_seal_sha256):
        raise PhysicalTournamentError("launch requires SHA-256 qualification and suite bindings")
    if set(endpoints) != {"qwen30", "qwen80"}:
        raise PhysicalTournamentError("launch requires exactly Qwen30 and Qwen80 endpoints")
    endpoint_rows = {key: dict(value) for key, value in endpoints.items()}
    endpoint_errors = [reason for endpoint in endpoint_rows.values() for reason in _check_endpoint(endpoint)]
    if endpoint_errors:
        raise PhysicalTournamentError("; ".join(sorted(set(endpoint_errors))))
    paths = _paths(resolved)
    launch_id = _digest(
        {
            "qualification_fingerprint": qualification_fingerprint,
            "suite_seal_sha256": suite_seal_sha256,
            "candidates": EXPECTED_CANDIDATES,
        }
    )
    existing = _read_launch(resolved)
    if existing is not None:
        if existing.get("launch_id") == launch_id:
            return existing
        raise PhysicalTournamentError("a different tournament launch lineage already exists; refusing replacement")

    base = {
        "schema": LAUNCH_SCHEMA,
        "status": LAUNCH_REQUESTED,
        "recorded_at": _utc_now(),
        "launch_id": launch_id,
        "physical_root": str(resolved),
        "qualification_fingerprint": qualification_fingerprint,
        "qualification_evidence": dict(qualification_evidence),
        "suite_preflight_path": str(paths["suite"]),
        "suite_preflight_seal_sha256": suite_seal_sha256,
        "fixed_candidate_order": list(EXPECTED_CANDIDATES),
        "candidate_endpoints": endpoint_rows,
        "runner": {
            "source_path": str(Path(__file__).resolve()),
            "source_sha256": _digest(_regular_file(Path(__file__), label="tournament runner")),
            "command": [
                str(Path(sys.executable).resolve()),
                str(Path(__file__).resolve()),
                "--root",
                str(resolved),
                "run",
                "--launch-id",
                launch_id,
            ],
            "detached": True,
        },
        "winner_selection": "DISABLED",
        "winner": None,
        "claim_boundary": {
            "launch_starts_only_the_protected_runner": True,
            "launch_does_not_select_a_winner": True,
            "launch_does_not_activate_sandbox": True,
            "raw_bf16_sources_are_not_participants": True,
        },
    }
    requested = seal(base)
    _atomic_json(paths["launch"], requested)
    try:
        paths["stdout"].parent.mkdir(parents=True, exist_ok=True)
        with paths["stdout"].open("ab") as stdout, paths["stderr"].open("ab") as stderr:
            process = subprocess.Popen(
                requested["runner"]["command"],
                cwd=str(REPO_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
        requested = seal({**base, "runner": {**base["runner"], "pid": process.pid}})
        _atomic_json(paths["launch"], requested)
        return requested
    except OSError as exc:
        failed = seal(
            {
                **base,
                "status": LAUNCH_FAILED,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _atomic_json(paths["launch"], failed)
        return failed


def _read_hidden_tasks() -> list[dict[str, Any]]:
    """Load evaluator-only tasks at run time; never serialize their contents."""

    raw = _regular_file(HIDDEN_ITEMS_PATH, label="hidden tournament tasks")
    tasks: list[dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, Mapping):
            raise PhysicalTournamentError("hidden tournament task is not an object")
        prompt = item.get("prompt")
        expected = item.get("expect_contains")
        if item.get("set") != "hidden" or not isinstance(prompt, str) or not isinstance(expected, list):
            raise PhysicalTournamentError("hidden tournament task shape is invalid")
        if not all(isinstance(token, str) and token for token in expected):
            raise PhysicalTournamentError("hidden tournament expected tokens are invalid")
        tasks.append({"prompt": prompt, "expect_contains": list(expected)})
    if not tasks:
        raise PhysicalTournamentError("no hidden tournament tasks were available")
    return tasks


def _request_json(url: str, payload: Mapping[str, Any], *, timeout: float) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(dict(payload), separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RECEIPT_BYTES)
    except (urllib.error.URLError, OSError) as exc:
        raise PhysicalTournamentError(f"HCLI request failed: {type(exc).__name__}: {exc}") from exc
    elapsed = (time.monotonic() - started) * 1000.0
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhysicalTournamentError(f"HCLI response is not JSON: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise PhysicalTournamentError("HCLI response root is not an object")
    return dict(decoded), elapsed


def _health(endpoint: Mapping[str, Any]) -> None:
    url = f"http://{endpoint['host']}:{endpoint['port']}{endpoint['health_path']}"
    try:
        with urllib.request.urlopen(url, timeout=15.0) as response:
            if int(response.status) != 200:
                raise PhysicalTournamentError(f"HCLI health returned HTTP {response.status}")
    except (urllib.error.URLError, OSError) as exc:
        raise PhysicalTournamentError(f"HCLI health failed: {type(exc).__name__}: {exc}") from exc


def _completion_text(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise PhysicalTournamentError("HCLI response has no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise PhysicalTournamentError("HCLI response choice is invalid")
    message = first.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise PhysicalTournamentError("HCLI response does not contain assistant text")
    return str(message["content"])


def _run_candidate(endpoint: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _health(endpoint)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for ordinal, task in enumerate(tasks, start=1):
        response, elapsed = _request_json(
            f"http://{endpoint['host']}:{endpoint['port']}{endpoint['chat_path']}",
            {
                "model": endpoint["model"],
                "messages": [{"role": "user", "content": task["prompt"]}],
                "temperature": 0,
                "stream": False,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        text = _completion_text(response)
        passed = all(token.lower() in text.lower() for token in task["expect_contains"])
        rows.append(
            {
                "ordinal": ordinal,
                "response_sha256": _digest(text.encode("utf-8")),
                "response_bytes": len(text.encode("utf-8")),
                "latency_ms": elapsed,
                "passed": passed,
            }
        )
    elapsed_seconds = max(time.monotonic() - started, 1e-9)
    passed_count = sum(1 for row in rows if row["passed"])
    return {
        "task_count": len(rows),
        "passed_task_count": passed_count,
        "verified_tasks_completed_per_hour": passed_count * 3600.0 / elapsed_seconds,
        "wall_seconds": elapsed_seconds,
        "rows": rows,
    }


def _resource_snapshot(root: Path) -> dict[str, Any]:
    usage = __import__("resource").getrusage(__import__("resource").RUSAGE_SELF)
    disk = os.statvfs(root)
    return {
        "wall_time": time.time(),
        "ru_utime_seconds": usage.ru_utime,
        "ru_stime_seconds": usage.ru_stime,
        "ru_maxrss": usage.ru_maxrss,
        "disk_free_bytes": disk.f_bavail * disk.f_frsize,
    }


def _write_runner_state(root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    document = seal(payload)
    _atomic_json(_paths(root)["runner"], document)
    return document


def run(root: str | Path = DEFAULT_PHYSICAL_ROOT, *, launch_id: str) -> dict[str, Any]:
    """Execute the frozen hidden comparison once; never choose a winner."""

    resolved = Path(root).expanduser().resolve()
    paths = _paths(resolved)
    with _exclusive_lock(paths["lock"]):
        launch = _read_launch(resolved)
        if launch is None or launch.get("schema") != LAUNCH_SCHEMA:
            raise PhysicalTournamentError("no valid launch receipt exists")
        if launch.get("launch_id") != launch_id or launch.get("status") != LAUNCH_REQUESTED:
            raise PhysicalTournamentError("launch receipt does not authorize this runner invocation")
        suite = validate_suite_preflight(resolved)
        if not suite["passed"] or suite["seal_sha256"] != launch.get("suite_preflight_seal_sha256"):
            return _write_runner_state(
                resolved,
                {
                    "schema": RUNNER_SCHEMA,
                    "status": ABORTED,
                    "recorded_at": _utc_now(),
                    "launch_id": launch_id,
                    "qualification_fingerprint": launch.get("qualification_fingerprint"),
                    "winner_selection": "DISABLED",
                    "winner": None,
                    "reason": "frozen suite preflight is absent, invalid, or drifted",
                },
            )
        existing = _read_runner(resolved)
        if existing is not None:
            if existing.get("launch_id") == launch_id:
                return existing
            raise PhysicalTournamentError("another runner lineage already exists")
        endpoints = launch.get("candidate_endpoints")
        if not isinstance(endpoints, Mapping) or set(endpoints) != {"qwen30", "qwen80"}:
            raise PhysicalTournamentError("launch receipt lacks the exact two candidate endpoints")
        normalized = {key: dict(value) for key, value in endpoints.items() if isinstance(value, Mapping)}
        errors = [reason for endpoint in normalized.values() for reason in _check_endpoint(endpoint)]
        if errors:
            raise PhysicalTournamentError("; ".join(sorted(set(errors))))
        _write_runner_state(
            resolved,
            {
                "schema": RUNNER_SCHEMA,
                "status": RUNNING,
                "recorded_at": _utc_now(),
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "launch_id": launch_id,
                "qualification_fingerprint": launch.get("qualification_fingerprint"),
                "suite_preflight_seal_sha256": suite["seal_sha256"],
                "fixed_candidate_order": list(EXPECTED_CANDIDATES),
                "winner_selection": "DISABLED",
                "winner": None,
                "claim_boundary": {
                    "runner_has_no_winner_selection_authority": True,
                    "runner_has_no_sandbox_activation_authority": True,
                    "raw_prompts_and_completions_are_not_persisted": True,
                },
            },
        )
    # Do not hold the interprocess lock while model work is running.
    before = _resource_snapshot(resolved)
    try:
        tasks = _read_hidden_tasks()
        results = {key: _run_candidate(normalized[key], tasks) for key in ("qwen30", "qwen80")}
        status = COMPLETE
        reason = "frozen deterministic comparison completed; protected human decision is required"
    except PhysicalTournamentError as exc:
        results = {}
        status = ABORTED
        reason = str(exc)
    after = _resource_snapshot(resolved)
    with _exclusive_lock(paths["lock"]):
        return _write_runner_state(
            resolved,
            {
                "schema": RUNNER_SCHEMA,
                "status": status,
                "recorded_at": _utc_now(),
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "launch_id": launch_id,
                "qualification_fingerprint": launch.get("qualification_fingerprint"),
                "suite_preflight_seal_sha256": suite["seal_sha256"],
                "fixed_candidate_order": list(EXPECTED_CANDIDATES),
                "winner_selection": "DISABLED",
                "winner": None,
                "reason": reason,
                "result_summary": results,
                "resource_accounting": {"before": before, "after": after},
                "claim_boundary": {
                    "result_does_not_select_a_winner": True,
                    "result_does_not_activate_sandbox": True,
                    "raw_prompts_and_completions_are_not_persisted": True,
                },
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_PHYSICAL_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze-suite", help="verify and freeze the exact protected suite")
    runner = subparsers.add_parser("run", help="execute one sealed launch request")
    runner.add_argument("--launch-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze-suite":
        document = freeze_suite_preflight(args.root)
    else:
        document = run(args.root, launch_id=str(args.launch_id))
    print(json.dumps({"status": document.get("status"), "seal_sha256": document.get("seal_sha256")}, sort_keys=True))
    return 0


__all__ = [
    "ABORTED",
    "COMPLETE",
    "DEFAULT_PHYSICAL_ROOT",
    "LAUNCH_FILENAME",
    "LAUNCH_REQUESTED",
    "LAUNCH_SCHEMA",
    "RUNNER_FILENAME",
    "RUNNER_SCHEMA",
    "RUNNING",
    "SUITE_FILENAME",
    "SUITE_SCHEMA",
    "SUITE_STATUS",
    "PhysicalTournamentError",
    "freeze_suite_preflight",
    "launch_state",
    "request_launch",
    "run",
    "validate_suite_preflight",
]


if __name__ == "__main__":
    raise SystemExit(main())
