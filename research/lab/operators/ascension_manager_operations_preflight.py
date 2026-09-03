"""Fail-closed live manager-operations preflight for the two Qwen artifacts.

This is deliberately a *measurement runner*, not a lifecycle controller and
not a replacement tournament gate.  It has a narrow job:

* wait for a canonical, gatekeeper-valid exact runtime and measured-HCLI
  receipt for one admitted Qwen Gravity artifact;
* bind a real loopback endpoint plus the already-frozen tournament suite;
* acquire the endpoint's explicit quiet benchmark lease, then exercise actual
  1/2/4/8 logical sessions and the endpoint's opt-in management probes;
* retain a sealed, redacted **unqualified** preflight result.

The final ``PASS_FINAL_MANAGER_OPERATIONS`` receipt remains owned by the
physical gate contract and is intentionally never written here.  In
particular, this runner cannot certify capability, HCLI, TPS, TG3, residency,
or tournament qualification merely because an endpoint answers a request.

The endpoint contract is intentionally explicit.  A normal OpenAI-compatible
chat endpoint is necessary but not sufficient to measure manager operations:
it must expose source-bound raw-decode measurement, per-request session/KV
telemetry, and safe isolated control probes.  Until that contract is included
in the sealed HCLI receipt, the watcher waits rather than fabricating values.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from lab.operators import ascension_physical_gatekeeper as gatekeeper
from lab.operators import ascension_physical_tournament as physical_tournament
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHYSICAL_ROOT = (
    REPO_ROOT / "workspace" / "campaign" / "records" / "ascension-sandbox" / "physical"
)

STATUS_SCHEMA = "hawking.ascension.manager_operations_preflight_status.v1"
RESULT_SCHEMA = "hawking.ascension.manager_operations_preflight_result.v1"
RESULT_STATUS = "MEASURED_MANAGER_OPERATIONS_PREFLIGHT_UNQUALIFIED"
BLOCKED_RESULT_STATUS = "BLOCKED_MANAGER_OPERATIONS_PREFLIGHT_REAL_TEST_FAILURE"
ENDPOINT_SCHEMA = "hawking.ascension.manager_operations_endpoint.v1"

SESSION_COUNTS: tuple[int, ...] = (1, 2, 4, 8)
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
MAX_HTTP_BYTES = 4 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 180.0
HEALTH_TIMEOUT_SECONDS = 15.0
RESTART_TIMEOUT_SECONDS = 180.0
DEFAULT_IDLE_SECONDS = 45.0
SESSION_HEADER = "X-Hawking-Session-Id"
TELEMETRY_KEY = "hawking_manager_operations"
CONTROL_OPERATIONS: tuple[str, ...] = (
    "acquire_quiet_benchmark_lease",
    "release_quiet_benchmark_lease",
    "residency_probe",
    "tool_recovery_probe",
    "rollback_probe",
    "storage_rollback_probe",
    "endpoint_restart",
)


class ManagerOperationsPreflightError(RuntimeError):
    """The live preflight cannot safely continue."""


@dataclass(frozen=True)
class Readiness:
    """The immutable evidence and endpoint binding for one bounded attempt."""

    spec: gatekeeper.ModelSpec
    root: Path
    runtime_seal_sha256: str
    hcli_seal_sha256: str
    suite_seal_sha256: str
    endpoint: dict[str, Any]
    fingerprint: str

    @property
    def agent_root(self) -> Path:
        return self.root / self.spec.key / "agent-os"


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


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    observed = float(value)
    return observed if math.isfinite(observed) else None


def _positive(value: Any) -> float | None:
    observed = _finite(value)
    return observed if observed is not None and observed > 0.0 else None


def _regular_read(path: Path, *, label: str, max_bytes: int = MAX_RECEIPT_BYTES) -> bytes:
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise ManagerOperationsPreflightError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ManagerOperationsPreflightError(f"{label} must be a regular non-symlink file: {path}")
    if observed.st_size > max_bytes:
        raise ManagerOperationsPreflightError(f"{label} exceeds {max_bytes} byte safety limit")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ManagerOperationsPreflightError(f"cannot read {label}: {exc}") from exc


def _load_sealed(path: Path, *, label: str) -> dict[str, Any]:
    raw = _regular_read(path, label=label)
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManagerOperationsPreflightError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ManagerOperationsPreflightError(f"{label} root is not an object")
    try:
        return verify(document, label=label)
    except SealIntegrityError as exc:
        raise ManagerOperationsPreflightError(str(exc)) from exc


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
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


def _model_spec(key: str) -> gatekeeper.ModelSpec:
    for spec in gatekeeper.MODEL_SPECS:
        if spec.key == key:
            return spec
    raise ManagerOperationsPreflightError(f"unknown manager key: {key}")


def _paths(root: Path, spec: gatekeeper.ModelSpec) -> dict[str, Path]:
    agent_root = root / spec.key / "agent-os"
    return {
        "status": agent_root / f"{spec.prefix}_MANAGER_OPERATIONS_PREFLIGHT_STATUS.json",
        "active": agent_root / f"{spec.prefix}_MANAGER_OPERATIONS_PREFLIGHT_ACTIVE.json",
        "runs": agent_root / "preflight-runs",
        "lock": agent_root / ".manager-operations-preflight.lock",
        "stdout": agent_root / "manager-operations-preflight.stdout.log",
        "stderr": agent_root / "manager-operations-preflight.stderr.log",
    }


def _safe_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ManagerOperationsPreflightError(f"{label} must be an absolute URL path")
    if value.startswith("//") or "?" in value or "#" in value or "\\" in value:
        raise ManagerOperationsPreflightError(f"{label} contains an unsafe URL form")
    parts = [part for part in value.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        raise ManagerOperationsPreflightError(f"{label} contains path traversal")
    return value


def _endpoint_from_hcli(document: Mapping[str, Any], spec: gatekeeper.ModelSpec) -> dict[str, Any]:
    """Read the opt-in manager-operations endpoint contract from HCLI evidence.

    It lives under ``measurement.manager_operations_endpoint`` so its network
    and control authority are sealed together with the exact HCLI evidence.
    A generic endpoint URL, status file, or a hand-written local config is not
    accepted as a substitute.
    """

    measurement = _mapping(document.get("measurement"))
    endpoint = _mapping(measurement.get("manager_operations_endpoint"))
    if not endpoint:
        raise ManagerOperationsPreflightError(
            "sealed measured-HCLI receipt lacks measurement.manager_operations_endpoint"
        )
    if endpoint.get("schema") != ENDPOINT_SCHEMA:
        raise ManagerOperationsPreflightError("manager-operations endpoint has an unexpected schema")
    if endpoint.get("protocol") != "openai_chat_completions_v1":
        raise ManagerOperationsPreflightError("manager-operations endpoint must use openai_chat_completions_v1")
    if endpoint.get("host") != "127.0.0.1":
        raise ManagerOperationsPreflightError("manager-operations endpoint must be loopback 127.0.0.1")
    port = endpoint.get("port")
    if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535:
        raise ManagerOperationsPreflightError("manager-operations endpoint port must be non-privileged")
    if endpoint.get("model") != spec.gravity_artifact_id:
        raise ManagerOperationsPreflightError("manager-operations endpoint model does not bind this Gravity artifact")
    if endpoint.get("gravity_artifact_id") != spec.gravity_artifact_id:
        raise ManagerOperationsPreflightError("manager-operations endpoint Gravity artifact binding is wrong")
    health_path = _safe_path(endpoint.get("health_path"), label="endpoint.health_path")
    chat_path = _safe_path(endpoint.get("chat_path"), label="endpoint.chat_path")
    if health_path != "/healthz" or chat_path != "/v1/chat/completions":
        raise ManagerOperationsPreflightError("manager-operations endpoint must bind the frozen HCLI paths")

    operations = _mapping(endpoint.get("operations"))
    if operations.get("session_header") != SESSION_HEADER:
        raise ManagerOperationsPreflightError(
            f"manager-operations endpoint must require {SESSION_HEADER} session affinity"
        )
    if operations.get("response_telemetry_field") != TELEMETRY_KEY:
        raise ManagerOperationsPreflightError(
            f"manager-operations endpoint must expose {TELEMETRY_KEY} telemetry"
        )
    raw_decode_path = _safe_path(operations.get("raw_decode_path"), label="operations.raw_decode_path")
    control_path = _safe_path(operations.get("control_path"), label="operations.control_path")
    if raw_decode_path == chat_path or control_path == chat_path:
        raise ManagerOperationsPreflightError("manager-operations raw/control paths must be distinct from chat")
    listed_operations = operations.get("control_operations")
    if not isinstance(listed_operations, list) or set(listed_operations) != set(CONTROL_OPERATIONS):
        raise ManagerOperationsPreflightError(
            "manager-operations endpoint must opt in to the exact isolated control probe set"
        )
    if operations.get("control_probes_are_isolated_and_non_destructive") is not True:
        raise ManagerOperationsPreflightError(
            "manager-operations endpoint must declare isolated non-destructive control probes"
        )
    if operations.get("single_model_body_shared_across_sessions") is not True:
        raise ManagerOperationsPreflightError(
            "manager-operations endpoint must declare one shared model body"
        )
    if operations.get("no_host_shell_or_hidden_membership_access") is not True:
        raise ManagerOperationsPreflightError(
            "manager-operations endpoint does not bind the protected tool policy"
        )
    return {
        "schema": ENDPOINT_SCHEMA,
        "protocol": "openai_chat_completions_v1",
        "host": "127.0.0.1",
        "port": port,
        "model": spec.gravity_artifact_id,
        "gravity_artifact_id": spec.gravity_artifact_id,
        "health_path": health_path,
        "chat_path": chat_path,
        "operations": {
            "session_header": SESSION_HEADER,
            "response_telemetry_field": TELEMETRY_KEY,
            "raw_decode_path": raw_decode_path,
            "control_path": control_path,
            "control_operations": list(CONTROL_OPERATIONS),
            "control_probes_are_isolated_and_non_destructive": True,
            "single_model_body_shared_across_sessions": True,
            "no_host_shell_or_hidden_membership_access": True,
        },
    }


def _gate_requirement(gate: Mapping[str, Any], spec: gatekeeper.ModelSpec, name: str) -> dict[str, Any]:
    rows = gate.get("models")
    if not isinstance(rows, list):
        raise ManagerOperationsPreflightError("physical gate status has no model rows")
    for row in rows:
        value = _mapping(row)
        if value.get("key") != spec.key:
            continue
        requirement = _mapping(_mapping(value.get("requirements")).get(name))
        if not requirement:
            raise ManagerOperationsPreflightError(f"physical gate has no {name} requirement for {spec.key}")
        return requirement
    raise ManagerOperationsPreflightError(f"physical gate has no {spec.key} row")


def readiness(root: str | Path, spec: gatekeeper.ModelSpec) -> tuple[Readiness | None, list[str], dict[str, Any]]:
    """Return one immutable runnable binding or exact fail-closed reasons.

    ``build_gate_status`` is intentionally read-only.  Reusing it here means
    this runner cannot reinterpret an arbitrary runtime/HCLI receipt as
    canonical simply because it has a familiar filename.
    """

    resolved = Path(root).expanduser().resolve()
    reasons: list[str] = []
    details: dict[str, Any] = {"model": spec.key}
    try:
        gate = gatekeeper.build_gate_status(resolved)
        runtime_gate = _gate_requirement(gate, spec, "native_exact_full_token_runtime")
        hcli_gate = _gate_requirement(gate, spec, "measured_hcli")
        if runtime_gate.get("state") != "PASS":
            reasons.append("canonical exact native full-token runtime gate has not passed")
        if hcli_gate.get("state") != "PASS":
            reasons.append("canonical measured-HCLI gate has not passed")
        runtime_seal = runtime_gate.get("seal_sha256")
        hcli_seal = hcli_gate.get("seal_sha256")
        if not _is_sha256(runtime_seal):
            reasons.append("canonical runtime receipt seal is unavailable")
        if not _is_sha256(hcli_seal):
            reasons.append("canonical HCLI receipt seal is unavailable")
        suite = physical_tournament.validate_suite_preflight(resolved)
        if not suite.get("passed"):
            reasons.append("frozen protected tournament suite preflight has not passed")
        suite_seal = suite.get("seal_sha256")
        if not _is_sha256(suite_seal):
            reasons.append("frozen protected tournament suite seal is unavailable")
        details.update(
            {
                "runtime_gate": runtime_gate,
                "hcli_gate": hcli_gate,
                "suite": {
                    "path": str(suite.get("path")),
                    "passed": bool(suite.get("passed")),
                    "seal_sha256": suite_seal,
                    "reasons": list(suite.get("reasons") or []),
                },
            }
        )
        if reasons:
            return None, list(dict.fromkeys(reasons)), details
        paths = gatekeeper._paths(resolved, spec)
        runtime_authority = gatekeeper.runtime_receipt_supersession_state(
            spec,
            runtime_path=paths["runtime"],
            supersession_path=paths["runtime_supersession"],
        )
        details["runtime_authority"] = runtime_authority
        if runtime_authority.get("current_runtime_eligible") is not True:
            reasons.append(
                "canonical runtime is revoked, superseded, or no longer an eligible PASS "
                f"({runtime_authority.get('state')})"
            )
            reasons.extend(
                f"runtime supersession: {reason}"
                for reason in runtime_authority.get("reasons") or ()
            )
        runtime = _load_sealed(paths["runtime"], label=f"{spec.key} exact runtime receipt")
        hcli = _load_sealed(paths["hcli"], label=f"{spec.key} measured HCLI receipt")
        if runtime.get("seal_sha256") != runtime_seal:
            reasons.append("runtime receipt changed after canonical gate validation")
        if runtime.get("seal_sha256") != runtime_authority.get("canonical_runtime_receipt_seal_sha256"):
            reasons.append("runtime receipt changed after supersession authority validation")
        if hcli.get("seal_sha256") != hcli_seal:
            reasons.append("HCLI receipt changed after canonical gate validation")
        binding = _mapping(hcli.get("binding"))
        if binding.get("runtime_receipt_seal_sha256") != runtime_seal:
            reasons.append("HCLI receipt does not bind the canonical runtime receipt")
        endpoint: dict[str, Any] | None = None
        if not reasons:
            try:
                endpoint = _endpoint_from_hcli(hcli, spec)
            except ManagerOperationsPreflightError as exc:
                reasons.append(str(exc))
        if reasons or endpoint is None:
            return None, list(dict.fromkeys(reasons)), details
        fingerprint = _digest(
            {
                "schema": RESULT_SCHEMA,
                "model": spec.key,
                "gravity_artifact_id": spec.gravity_artifact_id,
                "runtime_receipt_seal_sha256": runtime_seal,
                "hcli_receipt_seal_sha256": hcli_seal,
                "tournament_suite_preflight_seal_sha256": suite_seal,
                "endpoint": endpoint,
            }
        )
        details.update(
            {
                "runtime_receipt_path": str(paths["runtime"]),
                "hcli_receipt_path": str(paths["hcli"]),
                "endpoint": endpoint,
                "fingerprint": fingerprint,
            }
        )
        return (
            Readiness(
                spec=spec,
                root=resolved,
                runtime_seal_sha256=str(runtime_seal),
                hcli_seal_sha256=str(hcli_seal),
                suite_seal_sha256=str(suite_seal),
                endpoint=endpoint,
                fingerprint=fingerprint,
            ),
            [],
            details,
        )
    except Exception as exc:
        reasons.append(f"readiness reconciliation failed: {type(exc).__name__}: {exc}")
        return None, list(dict.fromkeys(reasons)), details


def _url(endpoint: Mapping[str, Any], path: str) -> str:
    return f"http://127.0.0.1:{int(endpoint['port'])}{path}"


def _request_json(
    endpoint: Mapping[str, Any],
    path: str,
    payload: Mapping[str, Any] | None,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    method: str | None = None,
) -> tuple[dict[str, Any], float]:
    """Issue a bounded loopback request and retain no raw model text."""

    body = None if payload is None else _canonical(dict(payload))
    request_headers = {"Accept": "application/json", "Cache-Control": "no-store"}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request_headers.update(dict(headers or {}))
    request = urllib.request.Request(
        _url(endpoint, path),
        data=body,
        headers=request_headers,
        method=method or ("POST" if body is not None else "GET"),
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if int(response.status) != 200:
                raise ManagerOperationsPreflightError(f"endpoint returned HTTP {response.status} for {path}")
            raw = response.read(MAX_HTTP_BYTES + 1)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        raise ManagerOperationsPreflightError(
            f"loopback endpoint request failed for {path}: {type(exc).__name__}: {exc}"
        ) from exc
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if len(raw) > MAX_HTTP_BYTES:
        raise ManagerOperationsPreflightError(f"endpoint response exceeds {MAX_HTTP_BYTES} byte safety limit")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManagerOperationsPreflightError(f"endpoint response is not JSON for {path}: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise ManagerOperationsPreflightError(f"endpoint response root is not an object for {path}")
    return dict(decoded), elapsed_ms


def _health(binding: Readiness) -> dict[str, Any]:
    document, elapsed_ms = _request_json(
        binding.endpoint,
        str(binding.endpoint["health_path"]),
        None,
        timeout=HEALTH_TIMEOUT_SECONDS,
    )
    if document.get("ready") is not True:
        raise ManagerOperationsPreflightError("endpoint health is not ready")
    if document.get("model_alone") is not True:
        raise ManagerOperationsPreflightError("endpoint health does not prove model-alone execution")
    if document.get("fallback_count") != 0:
        raise ManagerOperationsPreflightError("endpoint health reports a fallback")
    if document.get("gravity_artifact_id") != binding.spec.gravity_artifact_id:
        raise ManagerOperationsPreflightError("endpoint health binds a different Gravity artifact")
    instance = document.get("server_instance_id")
    if not isinstance(instance, str) or not instance:
        raise ManagerOperationsPreflightError("endpoint health lacks a restart-observable server_instance_id")
    return {
        "server_instance_id": instance,
        "health_sha256": _digest(document),
        "latency_ms": elapsed_ms,
    }


def _require_control_response(
    document: Mapping[str, Any], binding: Readiness, *, operation: str
) -> dict[str, Any]:
    value = _mapping(document)
    if value.get("operation") != operation:
        raise ManagerOperationsPreflightError(f"control response did not acknowledge {operation}")
    if value.get("completed") is not True:
        raise ManagerOperationsPreflightError(f"control operation {operation} did not complete")
    if value.get("gravity_artifact_id") != binding.spec.gravity_artifact_id:
        raise ManagerOperationsPreflightError(f"control operation {operation} bound a different artifact")
    if value.get("runtime_receipt_seal_sha256") != binding.runtime_seal_sha256:
        raise ManagerOperationsPreflightError(f"control operation {operation} bound a different runtime receipt")
    if value.get("hcli_receipt_seal_sha256") != binding.hcli_seal_sha256:
        raise ManagerOperationsPreflightError(f"control operation {operation} bound a different HCLI receipt")
    if value.get("no_fallback") is not True:
        raise ManagerOperationsPreflightError(f"control operation {operation} reports fallback")
    if value.get("isolated_non_destructive") is not True:
        raise ManagerOperationsPreflightError(f"control operation {operation} is not isolated/non-destructive")
    return value


def _control(binding: Readiness, operation: str, *, test_id: str, extra: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], float]:
    if operation not in CONTROL_OPERATIONS:
        raise ManagerOperationsPreflightError(f"unsupported control operation: {operation}")
    payload = {
        "schema": ENDPOINT_SCHEMA,
        "operation": operation,
        "test_id": test_id,
        "model": binding.spec.gravity_artifact_id,
        "gravity_artifact_id": binding.spec.gravity_artifact_id,
        "runtime_receipt_seal_sha256": binding.runtime_seal_sha256,
        "hcli_receipt_seal_sha256": binding.hcli_seal_sha256,
        "tournament_suite_preflight_seal_sha256": binding.suite_seal_sha256,
        **dict(extra or {}),
    }
    document, elapsed_ms = _request_json(
        binding.endpoint,
        str(_mapping(binding.endpoint["operations"])["control_path"]),
        payload,
    )
    return _require_control_response(document, binding, operation=operation), elapsed_ms


def _acquire_quiet_lease(binding: Readiness, *, test_id: str) -> tuple[str, dict[str, Any]]:
    document, elapsed_ms = _control(binding, "acquire_quiet_benchmark_lease", test_id=test_id)
    lease_id = document.get("lease_id")
    if not isinstance(lease_id, str) or not lease_id:
        raise ManagerOperationsPreflightError("quiet benchmark lease lacks lease_id")
    if document.get("exclusive_gpu") is not True:
        raise ManagerOperationsPreflightError("quiet benchmark lease is not GPU exclusive")
    return lease_id, {"response_sha256": _digest(document), "latency_ms": elapsed_ms}


def _release_quiet_lease(binding: Readiness, *, test_id: str, lease_id: str) -> dict[str, Any]:
    document, elapsed_ms = _control(
        binding,
        "release_quiet_benchmark_lease",
        test_id=test_id,
        extra={"lease_id": lease_id},
    )
    if document.get("lease_id") != lease_id:
        raise ManagerOperationsPreflightError("quiet benchmark lease release bound a different lease")
    return {"response_sha256": _digest(document), "latency_ms": elapsed_ms}


def _run_raw_measurement(binding: Readiness, *, session_count: int, test_id: str, lease_id: str) -> dict[str, Any]:
    payload = {
        "model": binding.spec.gravity_artifact_id,
        "logical_sessions": session_count,
        "test_id": test_id,
        "lease_id": lease_id,
        "runtime_receipt_seal_sha256": binding.runtime_seal_sha256,
        "hcli_receipt_seal_sha256": binding.hcli_seal_sha256,
        "tournament_suite_preflight_seal_sha256": binding.suite_seal_sha256,
        "prompt": "Return the single word READY.",
        "max_tokens": 8,
        "measurement_scope": "complete_model_token_loop",
    }
    document, elapsed_ms = _request_json(
        binding.endpoint,
        str(_mapping(binding.endpoint["operations"])["raw_decode_path"]),
        payload,
    )
    measurement = _mapping(document.get("measurement"))
    if measurement.get("timing_scope") != "complete_model_token_loop":
        raise ManagerOperationsPreflightError("raw measurement is not a complete model token loop")
    for field in ("uses_exact_native_runtime", "full_token_execution", "model_alone", "no_fallback"):
        if measurement.get(field) is not True:
            raise ManagerOperationsPreflightError(f"raw measurement does not prove {field}")
    if measurement.get("runtime_receipt_seal_sha256") != binding.runtime_seal_sha256:
        raise ManagerOperationsPreflightError("raw measurement bound a different runtime receipt")
    tokens = _positive(measurement.get("measured_token_count"))
    elapsed_seconds = _positive(measurement.get("elapsed_seconds"))
    reported_tps = _positive(measurement.get("base_true_tokens_per_second"))
    if tokens is None or elapsed_seconds is None or reported_tps is None:
        raise ManagerOperationsPreflightError("raw measurement lacks positive token/time/TPS fields")
    computed_tps = tokens / elapsed_seconds
    if abs(computed_tps - reported_tps) / max(computed_tps, reported_tps) > 0.05:
        raise ManagerOperationsPreflightError("raw measurement reported TPS does not match its tokens/time")
    return {
        "logical_sessions": session_count,
        "base_true_tokens_per_second": reported_tps,
        "computed_tokens_per_second": computed_tps,
        "measured_token_count": tokens,
        "elapsed_seconds": elapsed_seconds,
        "request_latency_ms": elapsed_ms,
        "response_sha256": _digest(document),
    }


def _completion_and_telemetry(
    document: Mapping[str, Any], binding: Readiness, *, session_id: str, continuation: bool
) -> tuple[int, dict[str, Any], str]:
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ManagerOperationsPreflightError("HCLI response has no usable choice")
    message = _mapping(choices[0].get("message"))
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ManagerOperationsPreflightError("HCLI response has no non-empty assistant content")
    usage = _mapping(document.get("usage"))
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int) or isinstance(completion_tokens, bool) or completion_tokens <= 0:
        raise ManagerOperationsPreflightError("HCLI response lacks positive usage.completion_tokens")
    telemetry = _mapping(document.get(TELEMETRY_KEY))
    if telemetry.get("session_id") != session_id:
        raise ManagerOperationsPreflightError("HCLI telemetry does not bind the logical session")
    if telemetry.get("gravity_artifact_id") != binding.spec.gravity_artifact_id:
        raise ManagerOperationsPreflightError("HCLI telemetry bound a different Gravity artifact")
    if telemetry.get("weight_body_id") != binding.spec.gravity_artifact_id:
        raise ManagerOperationsPreflightError("HCLI telemetry did not use the shared Gravity body")
    if telemetry.get("weight_reuse_observed") is not True:
        raise ManagerOperationsPreflightError("HCLI telemetry did not observe cross-session weight reuse")
    if telemetry.get("no_fallback") is not True:
        raise ManagerOperationsPreflightError("HCLI telemetry reports a fallback")
    if continuation and telemetry.get("context_reused") is not True:
        raise ManagerOperationsPreflightError("HCLI continuation did not reuse context/KV state")
    for field in ("kv_state_bytes", "context_compile_latency_ms", "tool_wait_ms", "queue_wait_ms"):
        value = _finite(telemetry.get(field))
        if value is None or value < 0.0:
            raise ManagerOperationsPreflightError(f"HCLI telemetry lacks finite non-negative {field}")
    return completion_tokens, telemetry, _digest(content.encode("utf-8"))


def _run_logical_session(binding: Readiness, *, session_id: str, lease_id: str) -> dict[str, Any]:
    headers = {SESSION_HEADER: session_id}
    initial_payload = {
        "model": binding.spec.gravity_artifact_id,
        "messages": [{"role": "user", "content": f"Reply briefly with marker {session_id}."}],
        "temperature": 0,
        "stream": False,
        "user": session_id,
        "hawking_manager_operations": {
            "lease_id": lease_id,
            "session_id": session_id,
            "turn": "initial",
        },
    }
    started = time.monotonic()
    initial, initial_latency_ms = _request_json(
        binding.endpoint, str(binding.endpoint["chat_path"]), initial_payload, headers=headers
    )
    initial_tokens, initial_telemetry, initial_content_sha = _completion_and_telemetry(
        initial, binding, session_id=session_id, continuation=False
    )
    prior_content = _mapping(initial.get("choices", [{}])[0]).get("message", {}).get("content")
    if not isinstance(prior_content, str):  # defended by _completion_and_telemetry above.
        raise ManagerOperationsPreflightError("initial HCLI content disappeared during validation")
    continuation_payload = {
        "model": binding.spec.gravity_artifact_id,
        "messages": [
            {"role": "user", "content": f"Reply briefly with marker {session_id}."},
            {"role": "assistant", "content": prior_content},
            {"role": "user", "content": f"Continue the same session and acknowledge {session_id}."},
        ],
        "temperature": 0,
        "stream": False,
        "user": session_id,
        "hawking_manager_operations": {
            "lease_id": lease_id,
            "session_id": session_id,
            "turn": "continuation",
        },
    }
    continuation_response, continuation_latency_ms = _request_json(
        binding.endpoint, str(binding.endpoint["chat_path"]), continuation_payload, headers=headers
    )
    continuation_tokens, continuation_telemetry, continuation_content_sha = _completion_and_telemetry(
        continuation_response, binding, session_id=session_id, continuation=True
    )
    elapsed_ms = (time.monotonic() - started) * 1000.0
    return {
        "session_id_sha256": _digest(session_id),
        "request_count": 2,
        "completed_generated_tokens": initial_tokens + continuation_tokens,
        "wall_ms": elapsed_ms,
        "initial_request_latency_ms": initial_latency_ms,
        "continuation_request_latency_ms": continuation_latency_ms,
        "initial_response_content_sha256": initial_content_sha,
        "continuation_response_content_sha256": continuation_content_sha,
        "kv_state_bytes": _finite(continuation_telemetry["kv_state_bytes"]),
        "context_compile_latency_ms": _finite(continuation_telemetry["context_compile_latency_ms"]),
        "tool_wait_ms": _finite(continuation_telemetry["tool_wait_ms"]),
        "queue_wait_ms": _finite(continuation_telemetry["queue_wait_ms"]),
        "weight_body_id": continuation_telemetry["weight_body_id"],
        "weight_reuse_observed": continuation_telemetry["weight_reuse_observed"],
        "context_reused": continuation_telemetry["context_reused"],
        "initial_telemetry_sha256": _digest(initial_telemetry),
        "continuation_telemetry_sha256": _digest(continuation_telemetry),
    }


def _p99(values: Sequence[float]) -> float:
    if not values:
        raise ManagerOperationsPreflightError("cannot compute p99 of no values")
    ordered = sorted(float(value) for value in values)
    # A nearest-rank p99 is conservative for the intentionally small 1/2/4/8
    # session samples: it selects the slowest logical session where needed.
    index = min(len(ordered) - 1, max(0, math.ceil(0.99 * len(ordered)) - 1))
    return ordered[index]


def _run_session_batch(binding: Readiness, *, session_count: int, test_id: str, lease_id: str) -> dict[str, Any]:
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=session_count, thread_name_prefix="ascension-manager-ops") as pool:
        futures = {
            pool.submit(
                _run_logical_session,
                binding,
                session_id=f"{binding.spec.key}-{test_id}-s{ordinal}",
                lease_id=lease_id,
            ): ordinal
            for ordinal in range(1, session_count + 1)
        }
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(f"session {futures[future]}: {type(exc).__name__}: {exc}")
    wall_seconds = max(time.monotonic() - started, 1e-9)
    if errors:
        raise ManagerOperationsPreflightError("; ".join(sorted(errors)))
    if len(rows) != session_count:
        raise ManagerOperationsPreflightError("session batch returned an unexpected row count")
    if any(row.get("weight_body_id") != binding.spec.gravity_artifact_id for row in rows):
        raise ManagerOperationsPreflightError("session batch did not share the exact one Gravity body")
    if not all(row.get("weight_reuse_observed") is True for row in rows):
        raise ManagerOperationsPreflightError("session batch did not observe shared-weight reuse")
    generated = sum(int(row["completed_generated_tokens"]) for row in rows)
    hcli_tps = generated / wall_seconds
    return {
        "logical_sessions": session_count,
        "raw_model_tps": None,  # Bound to the separately measured raw row below.
        "hcli_tps": hcli_tps,
        "per_session_p99_ms": _p99([float(row["wall_ms"]) for row in rows]),
        "verified_tasks_per_hour": session_count * 3600.0 / wall_seconds,
        "kv_state_bytes": max(float(row["kv_state_bytes"]) for row in rows),
        "context_compile_latency_ms": max(float(row["context_compile_latency_ms"]) for row in rows),
        "tool_wait_ms": max(float(row["tool_wait_ms"]) for row in rows),
        "queue_wait_ms": max(float(row["queue_wait_ms"]) for row in rows),
        "weight_reuse_observed": True,
        "starvation_free": True,
        "completed_generated_tokens": generated,
        "wall_seconds": wall_seconds,
        "session_rows": rows,
    }


def _probe_operations(binding: Readiness, *, test_id: str, lease_id: str) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    residency, residency_ms = _control(
        binding,
        "residency_probe",
        test_id=test_id,
        extra={"lease_id": lease_id, "logical_sessions": list(SESSION_COUNTS)},
    )
    if residency.get("resident_model_body_count") != 1:
        raise ManagerOperationsPreflightError("residency probe did not retain exactly one model body")
    if residency.get("weight_body_id") != binding.spec.gravity_artifact_id:
        raise ManagerOperationsPreflightError("residency probe bound a different model body")
    if residency.get("logical_sessions") != list(SESSION_COUNTS):
        raise ManagerOperationsPreflightError("residency probe did not cover 1/2/4/8 sessions")
    rows["residency"] = {"response_sha256": _digest(residency), "latency_ms": residency_ms}

    tool, tool_ms = _control(
        binding,
        "tool_recovery_probe",
        test_id=test_id,
        extra={"lease_id": lease_id, "tool_name": "isolated_recoverable_echo", "simulate_failure": True},
    )
    if tool.get("tool_recovery_passed") is not True:
        raise ManagerOperationsPreflightError("tool recovery probe did not recover")
    if _finite(tool.get("tool_wait_ms")) is None:
        raise ManagerOperationsPreflightError("tool recovery probe lacks tool wait telemetry")
    rows["tool_recovery"] = {"response_sha256": _digest(tool), "latency_ms": tool_ms}

    rollback, rollback_ms = _control(
        binding,
        "rollback_probe",
        test_id=test_id,
        extra={"lease_id": lease_id, "isolated_scratch_namespace": f"manager-ops-{test_id}"},
    )
    if rollback.get("rollback_passed") is not True:
        raise ManagerOperationsPreflightError("rollback probe did not complete an isolated rollback")
    rows["rollback"] = {"response_sha256": _digest(rollback), "latency_ms": rollback_ms}

    storage, storage_ms = _control(
        binding,
        "storage_rollback_probe",
        test_id=test_id,
        extra={"lease_id": lease_id, "isolated_scratch_namespace": f"manager-ops-{test_id}"},
    )
    if storage.get("storage_rollback_passed") is not True:
        raise ManagerOperationsPreflightError("storage rollback probe did not complete")
    if not isinstance(storage.get("disk_free_delta_bytes"), int) or isinstance(storage.get("disk_free_delta_bytes"), bool):
        raise ManagerOperationsPreflightError("storage rollback probe lacks integer disk free delta")
    rows["storage_rollback"] = {"response_sha256": _digest(storage), "latency_ms": storage_ms}
    return rows


def _restart_endpoint(binding: Readiness, *, test_id: str) -> dict[str, Any]:
    before = _health(binding)
    response, request_latency_ms = _control(binding, "endpoint_restart", test_id=test_id)
    if response.get("restart_requested") is not True:
        raise ManagerOperationsPreflightError("endpoint restart control did not request a real restart")
    deadline = time.monotonic() + RESTART_TIMEOUT_SECONDS
    after: dict[str, Any] | None = None
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            candidate = _health(binding)
        except ManagerOperationsPreflightError as exc:
            last_error = str(exc)
            time.sleep(0.5)
            continue
        if candidate["server_instance_id"] != before["server_instance_id"]:
            after = candidate
            break
        last_error = "endpoint health returned the same server_instance_id after restart"
        time.sleep(0.5)
    if after is None:
        raise ManagerOperationsPreflightError(
            f"endpoint restart did not produce a new ready server instance: {last_error}"
        )
    return {
        "control_response_sha256": _digest(response),
        "request_latency_ms": request_latency_ms,
        "before_server_instance_id_sha256": _digest(before["server_instance_id"]),
        "after_server_instance_id_sha256": _digest(after["server_instance_id"]),
        "after_health_sha256": after["health_sha256"],
    }


def _result_path(binding: Readiness) -> Path:
    return _paths(binding.root, binding.spec)["runs"] / f"{binding.spec.prefix}_MANAGER_OPERATIONS_PREFLIGHT_{binding.fingerprint}.json"


def _existing_result(binding: Readiness) -> dict[str, Any] | None:
    path = _result_path(binding)
    if not path.exists():
        return None
    try:
        result = _load_sealed(path, label=f"{binding.spec.key} manager-operations preflight result")
    except ManagerOperationsPreflightError:
        return None
    if result.get("schema") != RESULT_SCHEMA or result.get("attempt_fingerprint") != binding.fingerprint:
        return None
    return result


def _status(
    root: Path,
    spec: gatekeeper.ModelSpec,
    phase: str,
    *,
    readiness_value: Readiness | None = None,
    reasons: Sequence[str] = (),
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = _paths(root, spec)
    previous_heartbeat = 0
    try:
        prior = json.loads(paths["status"].read_text(encoding="utf-8"))
        if isinstance(prior, Mapping):
            value = prior.get("heartbeat")
            previous_heartbeat = int(value) if isinstance(value, int) and not isinstance(value, bool) else 0
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    document = {
        "schema": STATUS_SCHEMA,
        "recorded_at": _utc_now(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "heartbeat": previous_heartbeat + 1,
        "model": spec.key,
        "gravity_artifact_id": spec.gravity_artifact_id,
        "phase": phase,
        "attempt_fingerprint": readiness_value.fingerprint if readiness_value else None,
        "reasons": list(dict.fromkeys(str(reason) for reason in reasons)),
        "details": dict(details or {}),
        "claim_boundary": {
            "not_a_runtime_hcli_tps_tg_capability_or_tournament_result": True,
            "does_not_write_final_manager_operations_receipt": True,
            "only_actual_loopback_endpoint_requests_can_create_a_preflight_result": True,
            "raw_prompt_or_completion_text_is_not_persisted": True,
        },
    }
    _atomic_json(paths["status"], document)
    return document


def _active_record(paths: Mapping[str, Path]) -> dict[str, Any] | None:
    try:
        value = json.loads(paths["active"].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _pid_alive(value: Any) -> bool:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return False
    try:
        os.kill(value, 0)
    except OSError:
        return False
    return True


def _write_active(paths: Mapping[str, Path], document: Mapping[str, Any]) -> None:
    _atomic_json(paths["active"], document)


def _launch_run(binding: Readiness) -> dict[str, Any]:
    paths = _paths(binding.root, binding.spec)
    command = [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "--physical-root",
        str(binding.root),
        "run",
        "--model",
        binding.spec.key,
        "--fingerprint",
        binding.fingerprint,
    ]
    paths["stdout"].parent.mkdir(parents=True, exist_ok=True)
    try:
        with paths["stdout"].open("ab") as stdout, paths["stderr"].open("ab") as stderr:
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
    except OSError as exc:
        raise ManagerOperationsPreflightError(f"cannot launch manager-operations preflight: {exc}") from exc
    active = {
        "schema": STATUS_SCHEMA,
        "phase": "RUNNING",
        "pid": process.pid,
        "ppid": os.getpid(),
        "started_at": _utc_now(),
        "model": binding.spec.key,
        "attempt_fingerprint": binding.fingerprint,
        "command": command,
        "stdout_path": str(paths["stdout"]),
        "stderr_path": str(paths["stderr"]),
        "claim_boundary": "spawned child is not a manager operations pass or qualification",
    }
    _write_active(paths, active)
    return active


def reconcile_once(root: str | Path, spec: gatekeeper.ModelSpec) -> dict[str, Any]:
    """One idempotent watcher reconciliation cycle; it never runs a model inline."""

    resolved = Path(root).expanduser().resolve()
    binding, reasons, details = readiness(resolved, spec)
    if binding is None:
        return _status(
            resolved,
            spec,
            "WAITING_FOR_CANONICAL_RUNTIME_HCLI_SUITE_AND_MANAGER_OPERATIONS_ENDPOINT",
            reasons=reasons,
            details=details,
        )
    paths = _paths(resolved, spec)
    with _exclusive_lock(paths["lock"]):
        prior = _existing_result(binding)
        if prior is not None:
            return _status(
                resolved,
                spec,
                "PREFLIGHT_RESULT_RETAINED_UNQUALIFIED",
                readiness_value=binding,
                details={
                    "result_path": str(_result_path(binding)),
                    "result_status": prior.get("status"),
                    "result_seal_sha256": prior.get("seal_sha256"),
                },
            )
        active = _active_record(paths)
        if isinstance(active, Mapping) and active.get("attempt_fingerprint") == binding.fingerprint:
            if active.get("phase") == "RUNNING" and _pid_alive(active.get("pid")):
                return _status(
                    resolved,
                    spec,
                    "PREFLIGHT_REAL_ENDPOINT_RUN_ACTIVE",
                    readiness_value=binding,
                    details={"active": dict(active)},
                )
            return _status(
                resolved,
                spec,
                "PREFLIGHT_PREVIOUS_RUN_TERMINATED_UNSEALED_AWAITING_BINDING_CHANGE_OR_OPERATOR_RETRY",
                readiness_value=binding,
                reasons=["previous bounded preflight child is no longer alive and emitted no sealed result"],
                details={"active": dict(active)},
            )
        # A serving endpoint is still verified by the child before it takes a
        # quiet lease.  The watcher itself does no model work.
        try:
            active = _launch_run(binding)
        except ManagerOperationsPreflightError as exc:
            return _status(
                resolved,
                spec,
                "PREFLIGHT_RUN_LAUNCH_FAILED",
                readiness_value=binding,
                reasons=[str(exc)],
            )
        return _status(
            resolved,
            spec,
            "PREFLIGHT_REAL_ENDPOINT_RUN_LAUNCHED",
            readiness_value=binding,
            details={"active": active},
        )


def _run_attempt(binding: Readiness) -> dict[str, Any]:
    """Execute one real, quiet, source-bound manager-operations test attempt."""

    paths = _paths(binding.root, binding.spec)
    test_id = f"{binding.spec.key}-{binding.fingerprint[:20]}"
    started = time.monotonic()
    health_before: dict[str, Any] | None = None
    lease_id: str | None = None
    release: dict[str, Any] | None = None
    measurements: list[dict[str, Any]] = []
    operation_probes: dict[str, Any] = {}
    restart: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        health_before = _health(binding)
        lease_id, lease = _acquire_quiet_lease(binding, test_id=test_id)
        for session_count in SESSION_COUNTS:
            raw = _run_raw_measurement(
                binding, session_count=session_count, test_id=test_id, lease_id=lease_id
            )
            session = _run_session_batch(
                binding, session_count=session_count, test_id=test_id, lease_id=lease_id
            )
            measurements.append({**session, "raw_model_tps": raw["base_true_tokens_per_second"], "raw_measurement": raw})
        operation_probes = _probe_operations(binding, test_id=test_id, lease_id=lease_id)
        release = _release_quiet_lease(binding, test_id=test_id, lease_id=lease_id)
        lease_id = None
        restart = _restart_endpoint(binding, test_id=test_id)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if lease_id is not None:
            try:
                release = _release_quiet_lease(binding, test_id=test_id, lease_id=lease_id)
            except Exception as exc:
                errors.append(f"lease release: {type(exc).__name__}: {exc}")
    elapsed_seconds = max(time.monotonic() - started, 1e-9)
    all_session_rows = len(measurements) == len(SESSION_COUNTS)
    all_operations = set(operation_probes) == {"residency", "tool_recovery", "rollback", "storage_rollback"}
    complete = not errors and all_session_rows and all_operations and restart is not None and release is not None
    return seal(
        {
            "schema": RESULT_SCHEMA,
            "status": RESULT_STATUS if complete else BLOCKED_RESULT_STATUS,
            "recorded_at": _utc_now(),
            "attempt_fingerprint": binding.fingerprint,
            "model": binding.spec.key,
            "gravity_artifact_id": binding.spec.gravity_artifact_id,
            "binding": {
                "runtime_receipt_seal_sha256": binding.runtime_seal_sha256,
                "hcli_receipt_seal_sha256": binding.hcli_seal_sha256,
                "tournament_suite_preflight_seal_sha256": binding.suite_seal_sha256,
                "manager_operations_endpoint": binding.endpoint,
            },
            "health_before": health_before,
            "quiet_benchmark_lease_release": release,
            "session_measurements": measurements,
            "operation_probes": operation_probes,
            "endpoint_restart": restart,
            "wall_seconds": elapsed_seconds,
            "errors": errors,
            "preflight_complete": complete,
            "required_logical_sessions": list(SESSION_COUNTS),
            "required_dimensions": {
                "raw_vs_hcli": True,
                "context_kv": True,
                "weight_reuse": True,
                "restart": True,
                "residency": True,
                "rollback": True,
                "storage_rollback": True,
                "tool_recovery": True,
                "quiet_exclusive_benchmark_lease": True,
            },
            "claim_boundary": {
                "this_is_an_actual_endpoint_preflight_not_a_final_manager_operations_receipt": True,
                "does_not_claim_hcli_tps_tg3_capability_or_tournament_qualification": True,
                "does_not_write_or_replace_final_manager_operations_receipt": True,
                "raw_model_tps_is_diagnostic_preflight_measurement_not_a_tg_receipt": True,
                "raw_prompt_and_completion_text_are_not_persisted": True,
                "control_probes_are_required_to_be_isolated_and_non_destructive": True,
            },
        }
    )


def _invalidate_result_if_runtime_changed(
    result: Mapping[str, Any], *, reasons: Sequence[str]
) -> dict[str, Any]:
    """Keep an in-flight preflight observational if its runtime is revoked.

    The preflight is intentionally unqualified, but a stale positive-looking
    result still must not be reusable after the native runtime authority has
    changed.  Its actual measurements remain recorded; only its disposition is
    made explicitly blocked before the first write.
    """

    body = {key: value for key, value in dict(result).items() if key != "seal_sha256"}
    errors = list(body.get("errors") or [])
    errors.extend(f"runtime authority changed before result seal: {reason}" for reason in reasons)
    body["status"] = BLOCKED_RESULT_STATUS
    body["errors"] = list(dict.fromkeys(errors))
    body["preflight_complete"] = False
    body["runtime_authority_at_finish"] = {
        "state": "REVOKED_OR_SUPERSEDED",
        "reasons": list(dict.fromkeys(reasons)),
    }
    return seal(body)


def run_attempt(root: str | Path, spec: gatekeeper.ModelSpec, *, fingerprint: str) -> dict[str, Any]:
    """Run a child attempt only if the current canonical binding is unchanged."""

    resolved = Path(root).expanduser().resolve()
    binding, reasons, _ = readiness(resolved, spec)
    paths = _paths(resolved, spec)
    if binding is None or binding.fingerprint != fingerprint:
        terminal = {
            "schema": STATUS_SCHEMA,
            "phase": "TERMINAL_BINDING_CHANGED_OR_UNREADY",
            "finished_at": _utc_now(),
            "pid": os.getpid(),
            "model": spec.key,
            "attempt_fingerprint": fingerprint,
            "reasons": reasons if binding is None else ["canonical binding fingerprint changed before run"],
            "claim_boundary": "no endpoint test was run and no qualification result was created",
        }
        _write_active(paths, terminal)
        return terminal
    # Claim the fingerprint briefly, then release the watcher lock while the
    # real endpoint is working.  Holding it across a long token/session run
    # would make the detached watcher look dead and would prevent it from
    # reporting a material endpoint failure.  The active record is the
    # idempotency claim; a second process with the same fingerprint refuses to
    # overlap it.
    with _exclusive_lock(paths["lock"]):
        existing = _existing_result(binding)
        if existing is not None:
            return existing
        active = _active_record(paths)
        if (
            isinstance(active, Mapping)
            and active.get("phase") == "RUNNING"
            and active.get("attempt_fingerprint") == binding.fingerprint
            and _pid_alive(active.get("pid"))
            and active.get("pid") != os.getpid()
        ):
            return dict(active)
        _write_active(
            paths,
            {
                "schema": STATUS_SCHEMA,
                "phase": "RUNNING",
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "started_at": _utc_now(),
                "model": spec.key,
                "attempt_fingerprint": binding.fingerprint,
                "claim_boundary": "active preflight run is not a qualification result",
            },
        )
    result = _run_attempt(binding)
    final_binding, final_reasons, _ = readiness(resolved, spec)
    if final_binding is None or final_binding.fingerprint != binding.fingerprint:
        result = _invalidate_result_if_runtime_changed(
            result,
            reasons=(
                final_reasons
                if final_binding is None
                else ["canonical binding fingerprint changed during preflight"]
            ),
        )
    with _exclusive_lock(paths["lock"]):
        # A same-fingerprint sealed result wins over an interrupted/restarted
        # child.  Never overwrite an immutable measurement attempt.
        existing = _existing_result(binding)
        if existing is not None:
            return existing
        _atomic_json(_result_path(binding), result)
        _write_active(
            paths,
            {
                "schema": STATUS_SCHEMA,
                "phase": "TERMINAL_SEALED_UNQUALIFIED_PREFLIGHT_RESULT",
                "finished_at": _utc_now(),
                "pid": os.getpid(),
                "model": spec.key,
                "attempt_fingerprint": binding.fingerprint,
                "result_path": str(_result_path(binding)),
                "result_status": result.get("status"),
                "result_seal_sha256": result.get("seal_sha256"),
                "claim_boundary": "sealed preflight does not satisfy the final manager operations gate",
            },
        )
        return result


_STOP = False


def _stop(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def watch(root: str | Path, *, idle_seconds: float) -> int:
    if idle_seconds <= 0.0:
        raise ManagerOperationsPreflightError("idle seconds must be positive")
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    resolved = Path(root).expanduser().resolve()
    while not _STOP:
        for spec in gatekeeper.MODEL_SPECS:
            try:
                reconcile_once(resolved, spec)
            except Exception as exc:
                _status(
                    resolved,
                    spec,
                    "PREFLIGHT_WATCHER_RECOVERABLE_CYCLE_FAILURE",
                    reasons=[f"{type(exc).__name__}: {exc}"],
                )
        deadline = time.monotonic() + idle_seconds
        while not _STOP and time.monotonic() < deadline:
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-root", type=Path, default=DEFAULT_PHYSICAL_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    once = commands.add_parser("once", help="reconcile detached preflight work once")
    once.add_argument("--model", choices=[spec.key for spec in gatekeeper.MODEL_SPECS])
    watcher = commands.add_parser("watch", help="run the detached idempotent preflight watcher")
    watcher.add_argument("--idle-seconds", type=float, default=DEFAULT_IDLE_SECONDS)
    run = commands.add_parser("run", help="execute one source-bound endpoint attempt")
    run.add_argument("--model", choices=[spec.key for spec in gatekeeper.MODEL_SPECS], required=True)
    run.add_argument("--fingerprint", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.physical_root).expanduser().resolve()
    if args.command == "watch":
        return watch(root, idle_seconds=float(args.idle_seconds))
    if args.command == "run":
        result = run_attempt(root, _model_spec(args.model), fingerprint=str(args.fingerprint))
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result.get("status") == RESULT_STATUS else 1
    specs = (_model_spec(args.model),) if args.model else gatekeeper.MODEL_SPECS
    rows = [reconcile_once(root, spec) for spec in specs]
    print(json.dumps(rows, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by wrapper/launchd.
    raise SystemExit(main())
