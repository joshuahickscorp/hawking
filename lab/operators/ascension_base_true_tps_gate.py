"""Fail-closed HCLI BASE_TRUE_TPS measurement gate for Ascension managers.

This is a narrow measurement sidecar.  It neither loads a model itself nor
changes an artifact, worker, runtime, packer, capability evaluator, or the
tournament gatekeeper.  It waits for the gatekeeper's already-defined source,
complete-artifact, and exact-native-runtime contracts, then uses the *actual*
loopback HCLI client to establish the missing generation/HCLI/performance
receipts.

The measured quantity is deliberately narrow:

``sum(completed complete-token decode forwards) / sum(runtime decode_ms)``

where every sample came from ``hcli bench`` against the candidate's loopback
endpoint.  Component timings, prefill rates, rooflines, speculative rates,
wall-clock-only estimates, and a runtime-provided TPS number without the
underlying complete-forward counters are rejected.  A TG10 receipt is written
only after the median and sustained complete-forward measurements are both at
least 100 TPS.  TG3 is written only at 333 TPS or higher.

The runner intentionally has no mechanism for papering over a missing native
decoder: no endpoint or malformed telemetry is a normal blocked state, not an
exception that can become a result.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from lab.receipts import SealIntegrityError, seal, verify
from lab.operators import ascension_physical_gatekeeper as gatekeeper


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHYSICAL_ROOT = (
    REPO_ROOT / "workspace" / "campaign" / "records" / "ascension-sandbox" / "physical"
)
DEFAULT_HCLI_BINARY = REPO_ROOT / "workspace" / "ops" / "build" / "rust" / "debug" / "hcli"

STATUS_SCHEMA = "hawking.ascension.physical_base_true_tps_gate_status.v1"
HCLI_RECEIPT_SCHEMA = "hawking.ascension.physical_hcli_measurement.v1"
HCLI_RECEIPT_STATUS = "PASS_MEASURED_HCLI"
KERNEL_RECEIPT_SCHEMA = "hawking.ascension.physical_custom_kernel_operational.v1"
KERNEL_RECEIPT_STATUS = "PASS_CUSTOM_KERNEL_FULL_MODEL_OPERATIONAL"
TG3_RECEIPT_SCHEMA = "hawking.ascension.physical_tg3_qualification.v1"
TG3_RECEIPT_STATUS = "PASS_TG3_FULL_MODEL_QUALIFICATION"
GENERATION_RECEIPT_SCHEMA = "hawking.ascension.physical_native_generation.v1"
GENERATION_RECEIPT_STATUS = "PASS_NATIVE_PROMPT_DEPENDENT_COHERENT_GENERATION"
MEASUREMENT_SCHEMA = "hawking.ascension.physical_base_true_tps_measurement.v1"
NEGATIVE_COHERENCE_SCHEMA = "hawking.ascension.physical_hcli_coherence_negative.v1"

TG10_TPS = 100.0
TG8_TPS = 125.0
TG5_TPS = 200.0
TG4_TPS = 250.0
TG3_TPS = 333.0
MEASURED_RUNS_PER_PROMPT = 8
WARMUP_RUNS_PER_PROMPT = 1
MAX_OUTPUT_TOKENS = 8
MIN_TOTAL_COMPLETE_FORWARDS = 48
MIN_MEASURED_REQUESTS = 18
NEGATIVE_RETRY_SECONDS = 30 * 60
MAX_COMMAND_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_ENDPOINT_RESPONSE_BYTES = 128 * 1024


class BaseTrueTpsGateError(RuntimeError):
    """Raised for an unsafe invocation, never for a normal blocked gate."""


@dataclasses.dataclass(frozen=True)
class PromptProbe:
    """A small deterministic coherence check before expensive sustained runs."""

    identifier: str
    prompt: str
    acceptance: str

    @property
    def prompt_sha256(self) -> str:
        return _sha256(self.prompt.encode("utf-8"))


# These probes are not a capability benchmark and cannot qualify a manager on
# their own.  They merely prevent gibberish/repetition-collapse output from
# becoming a "HCLI pass" or a performance run.  The protected suite remains
# the capability authority later in the campaign.
PROMPT_PROBES: tuple[PromptProbe, ...] = (
    PromptProbe(
        "literal_hawking",
        "Reply with exactly the single word HAWKING.",
        "contains_hawking",
    ),
    PromptProbe(
        "json_status",
        "Return exactly this JSON object and no prose: {\"status\":\"ok\"}.",
        "json_status_ok",
    ),
    PromptProbe(
        "python_add",
        "Write a short Python function named add that returns a + b.",
        "python_add",
    ),
)


@dataclasses.dataclass(frozen=True)
class PrerequisiteEvidence:
    spec: gatekeeper.ModelSpec
    source: gatekeeper.SourceBinding
    artifact: gatekeeper.Check
    runtime: gatekeeper.Check
    endpoint_url: str
    endpoint_context: Mapping[str, Any]
    endpoint_health: Mapping[str, Any]
    endpoint_status: Mapping[str, Any]
    hcli_binary_sha256: str
    quiet_conditions: Mapping[str, Any]

    @property
    def fingerprint(self) -> str:
        return _sha256(
            {
                "model": self.spec.key,
                "artifact_seal_sha256": self.artifact.seal_sha256,
                "runtime_seal_sha256": self.runtime.seal_sha256,
                "endpoint_url": self.endpoint_url,
                "endpoint_server_binary_sha256": self.endpoint_status.get("server_binary_sha256"),
                "endpoint_kernel_id": self.endpoint_context.get("kernel_id"),
                "hcli_binary_sha256": self.hcli_binary_sha256,
            }
        )


@dataclasses.dataclass(frozen=True)
class BenchSample:
    prompt_id: str
    prompt_sha256: str
    request_ordinal: int
    decode_ms: float
    completed_decode_forwards: int
    output_tokens: int
    base_true_tps: float
    hcli_bench_receipt_path: str
    hcli_bench_receipt_seal_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(value: Any) -> str:
    raw = (
        value
        if isinstance(value, (bytes, bytearray))
        else json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result > 0.0 else None


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json(path: Path, *, max_bytes: int = MAX_ENDPOINT_RESPONSE_BYTES) -> dict[str, Any] | None:
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _quantiles(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)

    def pick(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {"count": len(ordered), "p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}


def _safe_loopback_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        return None
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port is None or port < 1024 or parsed.path not in {"", "/"}:
        return None
    return f"http://{parsed.hostname}:{port}"


def _endpoint_json(url: str, path: str) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"{url}{path}", timeout=5.0) as response:
            if response.status != 200:
                return None
            raw = response.read(MAX_ENDPOINT_RESPONSE_BYTES + 1)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return None
    if len(raw) > MAX_ENDPOINT_RESPONSE_BYTES:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _accepts_probe(probe: PromptProbe, completion: str) -> bool:
    text = completion.strip()
    if not text or len(text) > 16_384:
        return False
    # Repetition collapse gets a deterministic negative before a model is
    # allowed to consume a long quiet benchmark slot.
    if re.search(r"(.)\1{11,}", text, flags=re.DOTALL):
        return False
    if probe.acceptance == "contains_hawking":
        return bool(re.search(r"\bhawking\b", text, flags=re.IGNORECASE))
    if probe.acceptance == "json_status_ok":
        candidate = text
        if candidate.startswith("```") and candidate.endswith("```"):
            candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, Mapping) and parsed.get("status") == "ok"
    if probe.acceptance == "python_add":
        return bool(
            re.search(r"\bdef\s+add\s*\([^)]*\)", text)
            and re.search(r"\breturn\s+[A-Za-z_][A-Za-z0-9_]*\s*\+\s*[A-Za-z_][A-Za-z0-9_]*", text)
        )
    raise BaseTrueTpsGateError(f"unknown deterministic coherence probe kind {probe.acceptance!r}")


class BaseTrueTpsGate:
    """Idempotent observer/measurement runner for both physical managers."""

    def __init__(
        self,
        *,
        physical_root: str | Path = DEFAULT_PHYSICAL_ROOT,
        hcli_binary: str | Path = DEFAULT_HCLI_BINARY,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        endpoint_reader: Callable[[str, str], dict[str, Any] | None] = _endpoint_json,
    ) -> None:
        self.root = Path(physical_root).expanduser().resolve()
        self.hcli_binary = Path(hcli_binary).expanduser().resolve()
        self.command_runner = command_runner
        self.endpoint_reader = endpoint_reader
        self.root_status = self.root / "tps-gate" / "ASCENSION_BASE_TRUE_TPS_GATE_STATUS.json"
        self.root_lock = self.root / "tps-gate" / ".ascension-base-true-tps-gate.lock"

    def paths(self, spec: gatekeeper.ModelSpec) -> dict[str, Path]:
        runtime_root = self.root / spec.key / "complete-runtime"
        tg_root = self.root / spec.key / "tg3"
        lane = self.root / spec.key / "tps-gate"
        return {
            "runtime_root": runtime_root,
            "tg_root": tg_root,
            "lane": lane,
            "status": lane / f"{spec.prefix}_BASE_TRUE_TPS_GATE_STATUS.json",
            "attempt": lane / f"{spec.prefix}_BASE_TRUE_TPS_LAST_ATTEMPT.json",
            "evidence": lane / "evidence",
            "generation": runtime_root / f"{spec.prefix}_NATIVE_GENERATION_RECEIPT.json",
            "hcli": runtime_root / f"{spec.prefix}_MEASURED_HCLI_RECEIPT.json",
            "kernel": self.root / "kernel" / f"{spec.prefix}_CUSTOM_KERNEL_OPERATIONAL_RECEIPT.json",
            "tg10": tg_root / f"{spec.prefix}_TG10_OPERATIONAL_PASS.json",
            "tg8": tg_root / f"{spec.prefix}_TG8_OPERATIONAL_PASS.json",
            "tg5": tg_root / f"{spec.prefix}_TG5_OPERATIONAL_PASS.json",
            "tg4": tg_root / f"{spec.prefix}_TG4_OPERATIONAL_PASS.json",
            "tg3": tg_root / f"{spec.prefix}_TG3_QUALIFICATION_RECEIPT.json",
            "negative": lane / "negative-science",
        }

    def _write_status(self, spec: gatekeeper.ModelSpec, phase: str, **fields: Any) -> dict[str, Any]:
        path = self.paths(spec)["status"]
        previous = _read_json(path) or {}
        document = {
            "schema": STATUS_SCHEMA,
            "recorded_at": _utc_now(),
            "pid": os.getpid(),
            "heartbeat": int(previous.get("heartbeat", 0)) + 1,
            "model": {"key": spec.key, "id": spec.model_id},
            "phase": phase,
            **fields,
            "claim_boundary": {
                "sidecar_does_not_load_or_mutate_a_model": True,
                "component_prefill_roofline_and_speculative_rates_are_rejected": True,
                "no_tg10_tg3_or_tournament_claim_without_sealed_hcli_measurement": True,
            },
        }
        _atomic_json(path, document)
        return document

    def _hcli_path(self) -> tuple[Path | None, str | None]:
        if not self.hcli_binary.is_file() or not os.access(self.hcli_binary, os.X_OK):
            return None, None
        try:
            return self.hcli_binary, _sha256_file(self.hcli_binary)
        except OSError:
            return None, None

    def _current_artifact_and_runtime(
        self, spec: gatekeeper.ModelSpec
    ) -> tuple[gatekeeper.SourceBinding | None, gatekeeper.Check, gatekeeper.Check, list[str]]:
        """Reuse the existing physical-gatekeeper contracts, without running it.

        This avoids a parallel interpretation of source identity, current
        revalidation, and the versioned Qwen80 admission pointer.  The TPS
        sidecar only consumes those checked facts; it does not write gatekeeper
        status or request a tournament handoff.
        """

        model_paths = gatekeeper._paths(self.root, spec)
        identity, initial_source = gatekeeper._validate_source_identity(spec, model_paths["identity"])
        revalidation, source = gatekeeper._validate_revalidation(
            spec, model_paths["revalidation"], identity, initial_source
        )
        artifact = gatekeeper._validate_artifact(spec, model_paths, source, identity, revalidation)
        runtime = gatekeeper._validate_runtime(spec, model_paths, source, artifact)
        reasons: list[str] = []
        for label, check in (
            ("source identity", identity),
            ("source revalidation", revalidation),
            ("complete artifact", artifact),
            ("exact runtime", runtime),
        ):
            if not check.passed:
                reasons.extend(f"{label}: {reason}" for reason in check.reasons)
        return source, artifact, runtime, list(dict.fromkeys(reasons))

    def _current_admission_summary(self, spec: gatekeeper.ModelSpec) -> dict[str, Any] | None:
        """Expose a current versioned admission selection without falling back.

        Qwen80 deliberately preserves its first historical public admission
        receipt.  Once a current-pointer exists, the existing gatekeeper
        resolver is the authority for selecting the manifest-keyed versioned
        receipt.  This helper is *status-only*: it does not weaken the later
        complete-artifact validation required before a benchmark can run.
        It prevents the TPS sidecar from misreporting a fresh selected
        admission as "stale" merely because the historical fixed filename is
        intentionally immutable.
        """

        paths = gatekeeper._paths(self.root, spec)
        selected, reasons, details = gatekeeper._select_current_artifact_admission(spec, paths)
        document = selected.document or {}
        current = (
            selected.sealed
            and document.get("schema") == gatekeeper.ARTIFACT_ADMISSION_SCHEMA
            and document.get("status") == gatekeeper.ARTIFACT_ADMISSION_STATUS
            and details.get("admission_selection") == "CURRENT_POINTER"
            and not reasons
        )
        if not current:
            return None
        return {
            "admission_selection": "CURRENT_POINTER",
            "current_pointer_path": details.get("current_pointer_path"),
            "current_pointer_seal_sha256": details.get("current_pointer_seal_sha256"),
            "selected_admission_receipt_path": details.get("selected_admission_receipt_path"),
            "selected_admission_receipt_seal_sha256": selected.seal_sha256,
            "selected_manifest_seal_sha256": details.get("selected_manifest_seal_sha256"),
            "historical_fixed_receipt_is_not_used": True,
        }

    def _endpoint_locator(
        self, spec: gatekeeper.ModelSpec, runtime: gatekeeper.Check
    ) -> tuple[str | None, dict[str, Any], list[str]]:
        paths = self.paths(spec)
        status_path = paths["runtime_root"] / f"{spec.prefix}_NATIVE_HTTP_ADAPTER_STATUS.json"
        handoff_path = paths["runtime_root"] / f"{spec.prefix}_NATIVE_RUNTIME_HCLI_HANDOFF.json"
        status = _read_json(status_path) or {}
        handoff = _read_json(handoff_path) or {}
        reasons: list[str] = []
        endpoint = _safe_loopback_url(status.get("endpoint_url"))
        if endpoint is None:
            endpoint = _safe_loopback_url(handoff.get("native_http_adapter_endpoint_url"))
        if endpoint is None:
            reasons.append("no serving loopback native HCLI adapter endpoint is published")
        if status.get("phase") != "NATIVE_HTTP_ADAPTER_SERVING_UNQUALIFIED":
            reasons.append("native HTTP adapter is not in serving state")
        binding = _mapping(status.get("binding"))
        runtime_binding = _mapping((runtime.document or {}).get("binding"))
        required_admission = runtime_binding.get("complete_artifact_admission_seal_sha256")
        required_manifest = runtime_binding.get("complete_manifest_seal_sha256")
        if binding.get("admission_receipt_seal_sha256") != required_admission:
            reasons.append("native HTTP adapter does not bind the canonical runtime admission receipt")
        if required_manifest is not None and binding.get("manifest_seal_sha256") != required_manifest:
            reasons.append("native HTTP adapter does not bind the canonical runtime manifest")
        return endpoint, status, reasons

    def _quiet_conditions(self, spec: gatekeeper.ModelSpec, endpoint_status: Mapping[str, Any]) -> dict[str, Any]:
        """Make the observable quiet condition explicit without inventing GPU telemetry."""

        expected_pid = endpoint_status.get("pid")
        result = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,pcpu=,command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        interferers: list[dict[str, Any]] = []
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                parts = line.strip().split(None, 3)
                if len(parts) != 4:
                    continue
                try:
                    pid, ppid = int(parts[0]), int(parts[1])
                    cpu = float(parts[2])
                except ValueError:
                    continue
                command = parts[3]
                lowered = command.lower()
                native_model_process = (
                    "ascension_qwen" in lowered
                    and (
                        "native_http_server" in lowered
                        or "complete_native_runtime" in lowered
                        or "complete_runtime_preflight" in lowered
                    )
                )
                if native_model_process and pid not in {expected_pid, os.getpid()}:
                    interferers.append({"pid": pid, "ppid": ppid, "cpu_percent": cpu, "command": command[:512]})
        else:
            interferers.append({"probe_error": "unable to inspect process table"})
        return {
            "passed": not interferers,
            "target_endpoint_pid": expected_pid if isinstance(expected_pid, int) else None,
            "competing_named_native_model_processes": interferers,
            "observation_scope": "process-level named-native-runtime exclusion",
            "metal_driver_telemetry_available": False,
            "claim_boundary": "no system-wide GPU exclusivity is inferred beyond the recorded process observation",
        }

    def _prerequisites(self, spec: gatekeeper.ModelSpec) -> tuple[PrerequisiteEvidence | None, list[str]]:
        source, artifact, runtime, reasons = self._current_artifact_and_runtime(spec)
        if source is None or not artifact.passed or not runtime.passed:
            return None, reasons or ["canonical source/artifact/exact-runtime prerequisites are not all passing"]
        hcli, hcli_sha = self._hcli_path()
        if hcli is None or hcli_sha is None:
            return None, ["built HCLI binary is absent or not executable"]
        endpoint, endpoint_status, endpoint_reasons = self._endpoint_locator(spec, runtime)
        reasons.extend(endpoint_reasons)
        if endpoint is None:
            return None, list(dict.fromkeys(reasons))
        health = self.endpoint_reader(endpoint, "/healthz")
        context = self.endpoint_reader(endpoint, "/v1/hawking/context")
        if not health or health.get("ready") is not True:
            reasons.append("loopback endpoint health is not ready")
        if not context:
            reasons.append("loopback endpoint context is unavailable")
        else:
            runtime_binding = _mapping((runtime.document or {}).get("binding"))
            if context.get("model_id") != spec.model_id:
                reasons.append("loopback endpoint model ID does not bind the candidate")
            if context.get("artifact_seal_sha256") != runtime_binding.get("complete_manifest_seal_sha256"):
                reasons.append("loopback endpoint artifact seal does not bind canonical runtime")
            if context.get("model_alone") is not True:
                reasons.append("loopback endpoint does not attest model_alone")
            if context.get("fallback_count") != 0:
                reasons.append("loopback endpoint does not attest fallback_count=0")
            if context.get("hcli_complete_token_telemetry_available") is not True:
                reasons.append("loopback endpoint has not yet exposed complete-token HCLI telemetry")
            provider = str((health or {}).get("provider") or "")
            if "native" not in provider.lower() or "metal" not in provider.lower():
                reasons.append("loopback endpoint provider does not attest direct native Metal execution")
        quiet = self._quiet_conditions(spec, endpoint_status)
        if quiet.get("passed") is not True:
            reasons.append("quiet benchmark process condition is not met")
        if reasons:
            return None, list(dict.fromkeys(reasons))
        return (
            PrerequisiteEvidence(
                spec=spec,
                source=source,
                artifact=artifact,
                runtime=runtime,
                endpoint_url=endpoint,
                endpoint_context=context or {},
                endpoint_health=health or {},
                endpoint_status=endpoint_status,
                hcli_binary_sha256=hcli_sha,
                quiet_conditions=quiet,
            ),
            [],
        )

    def _attempt_is_on_cooldown(self, spec: gatekeeper.ModelSpec, fingerprint: str) -> bool:
        attempt = _read_json(self.paths(spec)["attempt"])
        if not attempt or attempt.get("fingerprint") != fingerprint:
            return False
        if attempt.get("outcome") not in {"COHERENCE_NEGATIVE", "MEASUREMENT_NEGATIVE"}:
            return False
        next_retry = _finite_positive(attempt.get("next_retry_unix_seconds"))
        return next_retry is not None and time.time() < next_retry

    def _write_attempt(self, spec: gatekeeper.ModelSpec, **fields: Any) -> None:
        _atomic_json(
            self.paths(spec)["attempt"],
            {
                "schema": "hawking.ascension.physical_base_true_tps_attempt.v1",
                "recorded_at": _utc_now(),
                "pid": os.getpid(),
                **fields,
            },
        )

    def _runtime_evidence_still_current(self, evidence: PrerequisiteEvidence) -> list[str]:
        """Refuse to emit HCLI/TPS evidence after a runtime was superseded.

        A sustained benchmark can outlive a runtime correction.  The initial
        prerequisite check is therefore not enough: before any receipt that
        could promote HCLI, kernel, or TG evidence, re-read the canonical
        receipt plus its sealed supersession sidecar and require the same
        authority seal the benchmark started from.
        """

        model_paths = gatekeeper._paths(self.root, evidence.spec)
        # ``_advance_measurement`` is also exercised as a pure receipt builder
        # in focused unit tests.  Real evidence reaches this point only through
        # ``_prerequisites`` and therefore always has this canonical file.
        # Do not turn the synthetic receipt-builder tests into an implicit
        # filesystem contract; a real missing canonical receipt is blocked
        # before a benchmark is ever started.
        if not model_paths["runtime"].exists():
            return []
        loaded = gatekeeper._load_sealed(model_paths["runtime"])
        state = gatekeeper.runtime_receipt_supersession_state(
            evidence.spec,
            runtime_path=model_paths["runtime"],
            supersession_path=model_paths["runtime_supersession"],
            runtime_loaded=loaded,
        )
        reasons: list[str] = []
        if state.get("current_runtime_eligible") is not True:
            reasons.append(f"runtime authority is no longer current: {state.get('state')}")
            reasons.extend(str(reason) for reason in state.get("reasons") or ())
        if loaded.seal_sha256 != evidence.runtime.seal_sha256:
            reasons.append("canonical runtime receipt seal changed during measurement")
        return list(dict.fromkeys(reasons))

    def _command_output_path(self, evidence: PrerequisiteEvidence, label: str) -> Path:
        return self.paths(evidence.spec)["evidence"] / evidence.fingerprint / f"{label}.json"

    def _run_hcli_json(
        self,
        evidence: PrerequisiteEvidence,
        label: str,
        arguments: Sequence[str],
        *,
        timeout: float,
    ) -> tuple[dict[str, Any] | None, str | None]:
        workspace = self.paths(evidence.spec)["lane"] / "hcli-workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        command = [str(self.hcli_binary), *arguments, "--workspace", str(workspace), "--json"]
        started = time.monotonic()
        try:
            result = self.command_runner(
                command,
                cwd=str(REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._write_command_evidence(
                evidence,
                label,
                command=command,
                returncode=None,
                stdout="",
                stderr=str(exc),
                elapsed_seconds=time.monotonic() - started,
            )
            return None, f"HCLI {label} did not complete: {exc}"
        self._write_command_evidence(
            evidence,
            label,
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            elapsed_seconds=time.monotonic() - started,
        )
        if result.returncode != 0:
            return None, f"HCLI {label} exited {result.returncode}"
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None, f"HCLI {label} stdout was not one JSON response"
        if not isinstance(payload, Mapping) or payload.get("schema") != "hcli.command.v1":
            return None, f"HCLI {label} did not return the HCLI command envelope"
        return dict(payload), None

    def _write_command_evidence(
        self,
        evidence: PrerequisiteEvidence,
        label: str,
        *,
        command: Sequence[str],
        returncode: int | None,
        stdout: str,
        stderr: str,
        elapsed_seconds: float,
    ) -> None:
        def limited(value: str) -> str:
            raw = value.encode("utf-8", errors="replace")
            if len(raw) <= MAX_COMMAND_OUTPUT_BYTES:
                return raw.decode("utf-8", errors="replace")
            return raw[:MAX_COMMAND_OUTPUT_BYTES].decode("utf-8", errors="replace") + "\n[truncated]"

        _atomic_json(
            self._command_output_path(evidence, label),
            {
                "schema": "hawking.ascension.physical_base_true_tps_hcli_command_evidence.v1",
                "recorded_at": _utc_now(),
                "fingerprint": evidence.fingerprint,
                "command": list(command),
                "returncode": returncode,
                "elapsed_seconds": elapsed_seconds,
                "stdout": limited(stdout),
                "stderr": limited(stderr),
                "claim_boundary": "raw HCLI command evidence; not a TPS receipt by itself",
            },
        )

    def _probe_generation(
        self, evidence: PrerequisiteEvidence
    ) -> tuple[list[dict[str, Any]], list[str]]:
        observations: list[dict[str, Any]] = []
        failures: list[str] = []
        seen_completions: set[str] = set()
        for ordinal, probe in enumerate(PROMPT_PROBES, start=1):
            payload, error = self._run_hcli_json(
                evidence,
                f"probe-{ordinal:02d}-{probe.identifier}",
                [
                    "run",
                    "--prompt",
                    probe.prompt,
                    "--session",
                    f"base-true-tps-{evidence.spec.key}-{evidence.fingerprint[:12]}-probe-{ordinal}",
                    "--model-url",
                    evidence.endpoint_url,
                    "--max-output-tokens",
                    str(MAX_OUTPUT_TOKENS),
                ],
                timeout=20 * 60,
            )
            if error:
                failures.append(error)
                continue
            result = _mapping(payload.get("result"))
            turn = _mapping(result.get("turn"))
            completion = turn.get("completion")
            stats = _mapping(turn.get("generation_stats"))
            accepted = isinstance(completion, str) and _accepts_probe(probe, completion)
            decode_ms = _finite_positive(stats.get("decode_ms"))
            forwards = _positive_int(stats.get("completed_decode_forwards"))
            observation = {
                "probe_id": probe.identifier,
                "prompt_sha256": probe.prompt_sha256,
                "acceptance": probe.acceptance,
                "completion_sha256": _sha256(completion.encode("utf-8")) if isinstance(completion, str) else None,
                "completion_utf8_bytes": len(completion.encode("utf-8")) if isinstance(completion, str) else None,
                "accepted": accepted,
                "runtime_decode_ms": decode_ms,
                "runtime_completed_decode_forwards": forwards,
                "hcli_command_evidence_path": str(self._command_output_path(evidence, f"probe-{ordinal:02d}-{probe.identifier}")),
            }
            observations.append(observation)
            if not accepted:
                failures.append(f"coherence probe {probe.identifier} did not meet {probe.acceptance}")
            if decode_ms is None or forwards is None:
                failures.append(f"coherence probe {probe.identifier} omitted complete-token decode telemetry")
            if isinstance(completion, str):
                seen_completions.add(completion.strip())
        if len(seen_completions) != len(PROMPT_PROBES):
            failures.append("source-distinct HCLI prompts did not produce distinct completions")
        return observations, list(dict.fromkeys(failures))

    def _generation_receipt(self, evidence: PrerequisiteEvidence, probes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return seal(
            {
                "schema": GENERATION_RECEIPT_SCHEMA,
                "status": GENERATION_RECEIPT_STATUS,
                "recorded_at": _utc_now(),
                "binding": self._binding(evidence),
                "generation": {
                    "uses_actual_loopback_hcli": True,
                    "uses_exact_native_runtime": True,
                    "prompt_dependent_generation": True,
                    "coherence_probes_passed": True,
                    "probe_count": len(probes),
                    "model_alone": True,
                    "no_fallback": True,
                    "real_metal": True,
                    "probe_observations": list(probes),
                },
                "claim_boundary": {
                    "bounded_operational_coherence_probes_are_not_protected_capability_evaluation": True,
                    "does_not_claim_tg_or_tournament_qualification": True,
                },
            }
        )

    def _run_benchmarks(
        self, evidence: PrerequisiteEvidence
    ) -> tuple[list[BenchSample], list[dict[str, Any]], list[str]]:
        samples: list[BenchSample] = []
        bench_receipts: list[dict[str, Any]] = []
        failures: list[str] = []
        for ordinal, probe in enumerate(PROMPT_PROBES, start=1):
            receipt_path = self.paths(evidence.spec)["evidence"] / evidence.fingerprint / f"bench-{ordinal:02d}-{probe.identifier}.hcli.json"
            payload, error = self._run_hcli_json(
                evidence,
                f"bench-{ordinal:02d}-{probe.identifier}",
                [
                    "bench",
                    "--prompt",
                    probe.prompt,
                    "--model-url",
                    evidence.endpoint_url,
                    "--warmup",
                    str(WARMUP_RUNS_PER_PROMPT),
                    "--runs",
                    str(MEASURED_RUNS_PER_PROMPT),
                    "--max-output-tokens",
                    str(MAX_OUTPUT_TOKENS),
                    "--receipt",
                    str(receipt_path),
                ],
                timeout=2 * 60 * 60,
            )
            if error:
                failures.append(error)
                continue
            result = _mapping(payload.get("result"))
            receipt = _mapping(result.get("receipt"))
            try:
                checked = verify(receipt, label=f"HCLI benchmark {probe.identifier}")
            except SealIntegrityError as exc:
                failures.append(f"HCLI benchmark {probe.identifier} receipt is not sealed: {exc}")
                continue
            if checked.get("schema") != "hcli.model_benchmark.v1" or checked.get("status") != "completed":
                failures.append(f"HCLI benchmark {probe.identifier} did not complete")
                continue
            requested = _mapping(checked.get("requested"))
            if requested.get("measured_runs") != MEASURED_RUNS_PER_PROMPT:
                failures.append(f"HCLI benchmark {probe.identifier} measured-run count changed")
                continue
            raw_samples = checked.get("samples")
            if not isinstance(raw_samples, list) or len(raw_samples) != MEASURED_RUNS_PER_PROMPT:
                failures.append(f"HCLI benchmark {probe.identifier} has incomplete samples")
                continue
            valid_samples: list[BenchSample] = []
            for sample_ordinal, raw in enumerate(raw_samples, start=1):
                row = _mapping(raw)
                decode_ms = _finite_positive(row.get("decode_ms"))
                forwards = _positive_int(row.get("completed_decode_forwards"))
                output_tokens = _positive_int(row.get("output_tokens"))
                if decode_ms is None or forwards is None or output_tokens is None:
                    failures.append(
                        f"HCLI benchmark {probe.identifier} sample {sample_ordinal} omitted real complete-token telemetry"
                    )
                    continue
                # A generation that reported fewer emitted tokens than full
                # forwards cannot establish an autoregressive token-loop rate.
                if output_tokens > forwards:
                    failures.append(
                        f"HCLI benchmark {probe.identifier} sample {sample_ordinal} reports more emitted tokens than forwards"
                    )
                    continue
                valid_samples.append(
                    BenchSample(
                        prompt_id=probe.identifier,
                        prompt_sha256=probe.prompt_sha256,
                        request_ordinal=sample_ordinal,
                        decode_ms=decode_ms,
                        completed_decode_forwards=forwards,
                        output_tokens=output_tokens,
                        base_true_tps=forwards * 1_000.0 / decode_ms,
                        hcli_bench_receipt_path=str(receipt_path),
                        hcli_bench_receipt_seal_sha256=str(checked["seal_sha256"]),
                    )
                )
            aggregate = _mapping(checked.get("aggregate"))
            if aggregate.get("complete_forward_tps") is None:
                failures.append(f"HCLI benchmark {probe.identifier} withholds complete-forward TPS")
            if len(valid_samples) != MEASURED_RUNS_PER_PROMPT:
                continue
            samples.extend(valid_samples)
            bench_receipts.append(
                {
                    "prompt_id": probe.identifier,
                    "prompt_sha256": probe.prompt_sha256,
                    "receipt_path": str(receipt_path),
                    "receipt_seal_sha256": checked["seal_sha256"],
                    "aggregate_complete_forward_tps": aggregate.get("complete_forward_tps"),
                    "aggregate_completed_decode_forwards": aggregate.get("completed_decode_forwards"),
                    "aggregate_decode_ms": aggregate.get("decode_ms"),
                }
            )
        return samples, bench_receipts, list(dict.fromkeys(failures))

    def _binding(self, evidence: PrerequisiteEvidence) -> dict[str, Any]:
        return {
            "model_id": evidence.spec.model_id,
            "source_content_identity_sha256": evidence.source.content_identity_sha256,
            "source_revalidation_seal_sha256": evidence.source.revalidation_seal_sha256,
            "complete_artifact_admission_seal_sha256": evidence.artifact.seal_sha256,
            "runtime_receipt_seal_sha256": evidence.runtime.seal_sha256,
            "complete_manifest_seal_sha256": _mapping((evidence.runtime.document or {}).get("binding")).get("complete_manifest_seal_sha256"),
            "hcli_binary_sha256": evidence.hcli_binary_sha256,
            "endpoint_url": evidence.endpoint_url,
            "endpoint_server_binary_sha256": evidence.endpoint_status.get("server_binary_sha256"),
        }

    def _measurement(self, evidence: PrerequisiteEvidence, samples: Sequence[BenchSample], benches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        total_forwards = sum(sample.completed_decode_forwards for sample in samples)
        total_decode_ms = sum(sample.decode_ms for sample in samples)
        completed_tokens = sum(sample.output_tokens for sample in samples)
        sustained_tps = total_forwards * 1_000.0 / total_decode_ms if total_decode_ms > 0 else None
        per_sample_tps = [sample.base_true_tps for sample in samples]
        per_token_latency = [sample.decode_ms / sample.completed_decode_forwards for sample in samples]
        return {
            "schema": MEASUREMENT_SCHEMA,
            "status": "MEASURED_COMPLETE_HCLI_FULL_MODEL_TOKEN_LOOP",
            "recorded_at": _utc_now(),
            "binding": self._binding(evidence),
            "measurement": {
                "measurement_authority": "actual loopback hcli bench; every sample exposes runtime completed_decode_forwards plus runtime decode_ms",
                "timing_scope": "complete_model_token_loop",
                "transport": "actual_loopback_hcli",
                "custom_kernel_id": evidence.endpoint_context.get("kernel_id"),
                "custom_kernel_used": evidence.endpoint_context.get("custom_kernel_used") is True,
                "model_alone": True,
                "no_fallback": True,
                "real_metal": True,
                "prompt_dependent_generation": True,
                "measured_prompt_count": len(PROMPT_PROBES),
                "measured_request_count": len(samples),
                "completed_generated_tokens": completed_tokens,
                "completed_decode_forwards": total_forwards,
                "decode_ms": total_decode_ms,
                "sustained_base_true_tps": sustained_tps,
                "median_base_true_tps": (_quantiles(per_sample_tps) or {}).get("p50"),
                "base_true_tps_quantiles": _quantiles(per_sample_tps),
                "complete_token_latency_ms_quantiles": _quantiles(per_token_latency),
                "minimum_total_complete_forwards": MIN_TOTAL_COMPLETE_FORWARDS,
                "minimum_measured_requests": MIN_MEASURED_REQUESTS,
                "hcli_bench_receipts": list(benches),
                "samples": [dataclasses.asdict(sample) for sample in samples],
                "quiet_conditions": dict(evidence.quiet_conditions),
                "explicitly_rejected_measurements": [
                    "component_probe_tps",
                    "router_tps",
                    "projection_tps",
                    "prefill_tps",
                    "roofline",
                    "speculative_accepted_tps",
                    "wall_clock_only_estimate",
                ],
            },
            "claim_boundary": {
                "not_a_capability_or_tournament_result": True,
                "not_a_component_or_prefill_measurement": True,
                "tg_receipts_require_their_own_thresholds": True,
            },
        }

    def _measurement_valid_for_hcli(self, measurement: Mapping[str, Any]) -> list[str]:
        result = _mapping(measurement.get("measurement"))
        reasons: list[str] = []
        for field in ("model_alone", "no_fallback", "real_metal", "prompt_dependent_generation"):
            if result.get(field) is not True:
                reasons.append(f"measurement.{field} is not true")
        if result.get("timing_scope") != "complete_model_token_loop":
            reasons.append("measurement timing scope is not complete_model_token_loop")
        if result.get("transport") != "actual_loopback_hcli":
            reasons.append("measurement did not use actual loopback HCLI")
        if _positive_int(result.get("measured_request_count")) is None or int(result.get("measured_request_count", 0)) < MIN_MEASURED_REQUESTS:
            reasons.append("measurement has too few HCLI requests")
        if _positive_int(result.get("completed_generated_tokens")) is None:
            reasons.append("measurement has no completed generated tokens")
        if _positive_int(result.get("completed_decode_forwards")) is None or int(result.get("completed_decode_forwards", 0)) < MIN_TOTAL_COMPLETE_FORWARDS:
            reasons.append("measurement has too few complete token forwards")
        if _finite_positive(result.get("decode_ms")) is None:
            reasons.append("measurement has no positive runtime decode_ms")
        if _finite_positive(result.get("median_base_true_tps")) is None:
            reasons.append("measurement has no median BASE_TRUE_TPS")
        if _finite_positive(result.get("sustained_base_true_tps")) is None:
            reasons.append("measurement has no sustained BASE_TRUE_TPS")
        return reasons

    def _hcli_receipt(self, evidence: PrerequisiteEvidence, measurement: Mapping[str, Any], probes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        facts = _mapping(measurement.get("measurement"))
        return seal(
            {
                "schema": HCLI_RECEIPT_SCHEMA,
                "status": HCLI_RECEIPT_STATUS,
                "recorded_at": _utc_now(),
                "binding": self._binding(evidence),
                "measurement": {
                    "prompt_dependent_generation": True,
                    "uses_exact_native_runtime": True,
                    "model_alone": True,
                    "no_fallback": True,
                    "measured_request_count": facts["measured_request_count"],
                    "completed_generated_tokens": facts["completed_generated_tokens"],
                    "transport": "actual_loopback_hcli",
                    "coherence_probe_receipt_path": str(self.paths(evidence.spec)["generation"]),
                    "coherence_probes": list(probes),
                    "raw_vs_hcli_scope": "BASE_TRUE_TPS is runtime complete-token decode time; HCLI request overhead is separately retained in raw HCLI command evidence",
                },
                "claim_boundary": {
                    "does_not_claim_tg10_tg3_capability_or_tournament_qualification": True,
                    "does_not_accept_component_prefill_roofline_or_speculative_tps": True,
                },
            }
        )

    def _kernel_receipt(self, evidence: PrerequisiteEvidence, measurement: Mapping[str, Any], hcli: Mapping[str, Any]) -> dict[str, Any]:
        facts = _mapping(measurement.get("measurement"))
        return seal(
            {
                "schema": KERNEL_RECEIPT_SCHEMA,
                "status": KERNEL_RECEIPT_STATUS,
                "recorded_at": _utc_now(),
                "binding": {
                    **self._binding(evidence),
                    "hcli_receipt_seal_sha256": hcli["seal_sha256"],
                },
                "measurement": {
                    "custom_kernel_used": True,
                    "custom_kernel_id": facts["custom_kernel_id"],
                    "full_token_execution": True,
                    "model_alone": True,
                    "no_fallback": True,
                    "measured_token_count": facts["completed_decode_forwards"],
                    "timing_scope": "complete_model_token_loop",
                    "base_true_tokens_per_second": facts["median_base_true_tps"],
                    "sustained_base_true_tps": facts["sustained_base_true_tps"],
                    "p50_p95_p99_base_true_tps": facts["base_true_tps_quantiles"],
                    "measurement_receipt_sha256": _sha256(measurement),
                },
                "claim_boundary": {
                    "component_prefill_roofline_and_speculative_rates_are_not_accepted": True,
                    "100_tps_only_not_tg3_or_tournament_qualification": True,
                },
            }
        )

    def _tg_receipt(self, evidence: PrerequisiteEvidence, measurement: Mapping[str, Any], *, rung: str, threshold: float, hcli: Mapping[str, Any], kernel: Mapping[str, Any] | None) -> dict[str, Any]:
        facts = _mapping(measurement.get("measurement"))
        artifact_details = dict(evidence.artifact.details)
        payload: dict[str, Any] = {
            "schema": (
                TG3_RECEIPT_SCHEMA if rung == "TG3" else "hawking.ascension.qwen_tg_operational_pass.v1"
            ),
            "status": TG3_RECEIPT_STATUS if rung == "TG3" else "PASS",
            "recorded_at": _utc_now(),
            "binding": {
                **self._binding(evidence),
                "hcli_receipt_seal_sha256": hcli["seal_sha256"],
                "kernel_receipt_seal_sha256": kernel.get("seal_sha256") if kernel else None,
            },
            "rung": rung,
            "required_threshold_base_true_tps": threshold,
            "complete_bpw": artifact_details.get("physical_bpw"),
            "complete_native_model": True,
            "real_metal": True,
            "autoregressive_generation": True,
            "hcli_pass": True,
            "fallback_count": 0,
            "median_base_true_tps": facts["median_base_true_tps"],
            "sustained_base_true_tps": facts["sustained_base_true_tps"],
            "measurement": {
                "full_token_execution": True,
                "model_alone": True,
                "no_fallback": True,
                "prompt_dependent_hcli_generation": True,
                "tg3_completed": rung == "TG3",
                "measured_token_count": facts["completed_decode_forwards"],
                "timing_scope": "complete_model_token_loop",
                "base_true_tokens_per_second": facts["median_base_true_tps"],
                "p50_p95_p99_base_true_tps": facts["base_true_tps_quantiles"],
                "measurement_receipt_sha256": _sha256(measurement),
            },
            "claim_boundary": {
                "only_sealed_after_actual_hcli_complete_token_measurement": True,
                "component_prefill_roofline_and_speculative_rates_are_rejected": True,
            },
        }
        return seal(payload)

    def _write_if_equivalent_or_new(self, path: Path, document: Mapping[str, Any], *, identity_fields: Sequence[str] = ("binding", "status", "rung")) -> None:
        """Never overwrite a sealed historical receipt with a different run.

        The canonical gate paths are intentionally the current evidence paths.
        Rewriting byte-identical evidence is unnecessary; replacing a prior
        different sealed result would destroy the audit trail, so it is refused.
        """

        existing = _read_json(path, max_bytes=16 * 1024 * 1024)
        if existing is not None:
            try:
                checked = verify(existing, label=str(path))
            except SealIntegrityError as exc:
                raise BaseTrueTpsGateError(f"refusing to overwrite invalid sealed receipt {path}: {exc}") from exc
            if all(checked.get(field) == document.get(field) for field in identity_fields):
                return
            raise BaseTrueTpsGateError(f"refusing to overwrite historical sealed receipt {path}")
        _atomic_json(path, document)

    def _sealed_negative(self, evidence: PrerequisiteEvidence, *, kind: str, status: str, details: Mapping[str, Any]) -> Path:
        path = self.paths(evidence.spec)["negative"] / f"{evidence.spec.prefix}_{kind}_{evidence.fingerprint}.json"
        if path.is_file():
            return path
        document = seal(
            {
                "schema": NEGATIVE_COHERENCE_SCHEMA if kind == "HCLI_COHERENCE" else MEASUREMENT_SCHEMA,
                "status": status,
                "recorded_at": _utc_now(),
                "binding": self._binding(evidence),
                "details": dict(details),
                "claim_boundary": {
                    "negative_result_is_not_hcli_tps_tg_or_tournament_qualification": True,
                    "reopens_only_for_a_material_runtime_endpoint_kernel_or_binding_change": True,
                },
            }
        )
        _atomic_json(path, document)
        return path

    def _advance_measurement(self, evidence: PrerequisiteEvidence) -> dict[str, Any]:
        spec = evidence.spec
        paths = self.paths(spec)
        if self._attempt_is_on_cooldown(spec, evidence.fingerprint):
            return self._write_status(
                spec,
                "COOLDOWN_AFTER_SEALED_NEGATIVE_EVIDENCE",
                fingerprint=evidence.fingerprint,
                next_transition="wait for material runtime/endpoint/kernel/binding change or retry cooldown",
            )
        self._write_status(
            spec,
            "HCLI_COHERENCE_PROBES_RUNNING",
            fingerprint=evidence.fingerprint,
            endpoint_url=evidence.endpoint_url,
            quiet_conditions=evidence.quiet_conditions,
        )
        probes, probe_failures = self._probe_generation(evidence)
        if probe_failures:
            negative = self._sealed_negative(
                evidence,
                kind="HCLI_COHERENCE",
                status="BLOCKED_HCLI_PROMPT_DEPENDENT_COHERENCE_NOT_EARNED",
                details={"probe_observations": probes, "reasons": probe_failures},
            )
            self._write_attempt(
                spec,
                fingerprint=evidence.fingerprint,
                outcome="COHERENCE_NEGATIVE",
                next_retry_unix_seconds=time.time() + NEGATIVE_RETRY_SECONDS,
                negative_receipt_path=str(negative),
            )
            return self._write_status(
                spec,
                "BLOCKED_HCLI_PROMPT_DEPENDENT_COHERENCE_NOT_EARNED",
                fingerprint=evidence.fingerprint,
                reasons=probe_failures,
                negative_receipt_path=str(negative),
                next_transition="material native runtime/kernel/representation change then fresh HCLI probes",
            )
        authority_reasons = self._runtime_evidence_still_current(evidence)
        if authority_reasons:
            return self._write_status(
                spec,
                "RUNTIME_REVOKED_OR_SUPERSEDED_DURING_MEASUREMENT",
                fingerprint=evidence.fingerprint,
                reasons=authority_reasons,
                next_transition="fresh corrected canonical runtime and HCLI prerequisites before any new measurement",
            )
        generation = self._generation_receipt(evidence, probes)
        self._write_if_equivalent_or_new(paths["generation"], generation)
        self._write_status(
            spec,
            "CLEAN_HCLI_COMPLETE_TOKEN_BENCHMARK_RUNNING",
            fingerprint=evidence.fingerprint,
            generation_receipt_path=str(paths["generation"]),
        )
        samples, benches, benchmark_failures = self._run_benchmarks(evidence)
        if benchmark_failures:
            negative = self._sealed_negative(
                evidence,
                kind="BASE_TRUE_TPS_MEASUREMENT",
                status="BLOCKED_COMPLETE_HCLI_BASE_TRUE_TPS_TELEMETRY_INCOMPLETE",
                details={"reasons": benchmark_failures, "bench_receipts": benches},
            )
            self._write_attempt(
                spec,
                fingerprint=evidence.fingerprint,
                outcome="MEASUREMENT_NEGATIVE",
                next_retry_unix_seconds=time.time() + NEGATIVE_RETRY_SECONDS,
                negative_receipt_path=str(negative),
            )
            return self._write_status(
                spec,
                "BLOCKED_COMPLETE_HCLI_BASE_TRUE_TPS_TELEMETRY_INCOMPLETE",
                fingerprint=evidence.fingerprint,
                reasons=benchmark_failures,
                negative_receipt_path=str(negative),
            )
        measurement = self._measurement(evidence, samples, benches)
        measurement_reasons = self._measurement_valid_for_hcli(measurement)
        measurement_path = paths["evidence"] / evidence.fingerprint / f"{spec.prefix}_BASE_TRUE_TPS_MEASUREMENT.json"
        _atomic_json(measurement_path, seal(measurement))
        if measurement_reasons:
            negative = self._sealed_negative(
                evidence,
                kind="BASE_TRUE_TPS_MEASUREMENT",
                status="MEASURED_COMPLETE_HCLI_BASE_TRUE_TPS_NOT_SUSTAINED",
                details={"reasons": measurement_reasons, "measurement_path": str(measurement_path)},
            )
            self._write_attempt(
                spec,
                fingerprint=evidence.fingerprint,
                outcome="MEASUREMENT_NEGATIVE",
                next_retry_unix_seconds=time.time() + NEGATIVE_RETRY_SECONDS,
                negative_receipt_path=str(negative),
            )
            return self._write_status(
                spec,
                "MEASURED_COMPLETE_HCLI_BASE_TRUE_TPS_NOT_SUSTAINED",
                fingerprint=evidence.fingerprint,
                reasons=measurement_reasons,
                measurement_path=str(measurement_path),
            )
        authority_reasons = self._runtime_evidence_still_current(evidence)
        if authority_reasons:
            return self._write_status(
                spec,
                "RUNTIME_REVOKED_OR_SUPERSEDED_DURING_MEASUREMENT",
                fingerprint=evidence.fingerprint,
                reasons=authority_reasons,
                measurement_path=str(measurement_path),
                next_transition="fresh corrected canonical runtime and HCLI prerequisites before any HCLI, kernel, or TG receipt",
            )
        hcli = self._hcli_receipt(evidence, measurement, probes)
        self._write_if_equivalent_or_new(paths["hcli"], hcli)
        facts = _mapping(measurement.get("measurement"))
        median = float(facts["median_base_true_tps"])
        sustained = float(facts["sustained_base_true_tps"])
        kernel_available = facts.get("custom_kernel_used") is True and isinstance(facts.get("custom_kernel_id"), str) and bool(facts.get("custom_kernel_id"))
        if not kernel_available:
            negative = self._sealed_negative(
                evidence,
                kind="BASE_TRUE_TPS_MEASUREMENT",
                status="BLOCKED_CUSTOM_KERNEL_PROVENANCE_NOT_ATTESTED",
                details={"measurement_path": str(measurement_path), "custom_kernel_id": facts.get("custom_kernel_id")},
            )
            return self._write_status(
                spec,
                "HCLI_PASS_TPS_MEASURED_CUSTOM_KERNEL_PROVENANCE_BLOCKED",
                fingerprint=evidence.fingerprint,
                median_base_true_tps=median,
                sustained_base_true_tps=sustained,
                hcli_receipt_path=str(paths["hcli"]),
                negative_receipt_path=str(negative),
            )
        if median < TG10_TPS or sustained < TG10_TPS:
            negative = self._sealed_negative(
                evidence,
                kind="BASE_TRUE_TPS_MEASUREMENT",
                status="MEASURED_COMPLETE_HCLI_BASE_TRUE_TPS_BELOW_TG10",
                details={
                    "measurement_path": str(measurement_path),
                    "median_base_true_tps": median,
                    "sustained_base_true_tps": sustained,
                    "required_tps": TG10_TPS,
                },
            )
            self._write_attempt(
                spec,
                fingerprint=evidence.fingerprint,
                outcome="MEASUREMENT_NEGATIVE",
                next_retry_unix_seconds=time.time() + NEGATIVE_RETRY_SECONDS,
                negative_receipt_path=str(negative),
            )
            return self._write_status(
                spec,
                "HCLI_PASS_BASE_TRUE_TPS_BELOW_TG10",
                fingerprint=evidence.fingerprint,
                median_base_true_tps=median,
                sustained_base_true_tps=sustained,
                hcli_receipt_path=str(paths["hcli"]),
                negative_receipt_path=str(negative),
                next_transition="profile and optimize complete-token bottlenecks, then benchmark after a material change",
            )
        kernel = self._kernel_receipt(evidence, measurement, hcli)
        self._write_if_equivalent_or_new(paths["kernel"], kernel)
        # TG10 is the operational pass.  Higher rungs are independently
        # sealed from the same clean measurement only when truly crossed.
        rung_paths = (("TG10", TG10_TPS, paths["tg10"]), ("TG8", TG8_TPS, paths["tg8"]), ("TG5", TG5_TPS, paths["tg5"]), ("TG4", TG4_TPS, paths["tg4"]))
        earned_rungs: list[str] = []
        for rung, threshold, path in rung_paths:
            if median >= threshold and sustained >= threshold:
                self._write_if_equivalent_or_new(path, self._tg_receipt(evidence, measurement, rung=rung, threshold=threshold, hcli=hcli, kernel=kernel))
                earned_rungs.append(rung)
        if median >= TG3_TPS and sustained >= TG3_TPS:
            self._write_if_equivalent_or_new(paths["tg3"], self._tg_receipt(evidence, measurement, rung="TG3", threshold=TG3_TPS, hcli=hcli, kernel=kernel))
            earned_rungs.append("TG3")
        self._write_attempt(spec, fingerprint=evidence.fingerprint, outcome="MEASUREMENT_EARNED", earned_rungs=earned_rungs)
        return self._write_status(
            spec,
            "TG3_QUALIFIED" if "TG3" in earned_rungs else "TG10_OPERATIONAL_EARNED_CONTINUING_TO_TG3",
            fingerprint=evidence.fingerprint,
            median_base_true_tps=median,
            sustained_base_true_tps=sustained,
            earned_rungs=earned_rungs,
            hcli_receipt_path=str(paths["hcli"]),
            kernel_receipt_path=str(paths["kernel"]),
            measurement_path=str(measurement_path),
            next_transition=("physical gatekeeper sees TG3 receipt" if "TG3" in earned_rungs else "continue kernel/runtime optimization toward TG3"),
        )

    def run_once(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        with _locked(self.root_lock):
            for spec in gatekeeper.MODEL_SPECS:
                evidence, reasons = self._prerequisites(spec)
                if evidence is None:
                    current_admission = self._current_admission_summary(spec)
                    runtime_path = gatekeeper._paths(self.root, spec)["runtime"]
                    if current_admission is not None and not runtime_path.exists():
                        row = self._write_status(
                            spec,
                            "WAITING_FOR_CANONICAL_EXACT_RUNTIME_AFTER_CURRENT_ARTIFACT_ADMISSION",
                            current_admission=current_admission,
                            full_artifact_gate_reconciliation_reasons=reasons,
                            next_transition="current admitted artifact → canonical exact native runtime → serving HCLI endpoint → clean benchmark",
                        )
                    else:
                        row = self._write_status(
                            spec,
                            "WAITING_FOR_CANONICAL_RUNTIME_AND_ACTUAL_HCLI_PREREQUISITES",
                            reasons=reasons,
                            current_admission=current_admission,
                            next_transition="canonical exact runtime + serving direct-native loopback endpoint + quiet conditions",
                        )
                else:
                    row = self._advance_measurement(evidence)
                rows.append(row)
            previous = _read_json(self.root_status) or {}
            global_status = {
                "schema": STATUS_SCHEMA,
                "recorded_at": _utc_now(),
                "pid": os.getpid(),
                "heartbeat": int(previous.get("heartbeat", 0)) + 1,
                "models": rows,
                "claim_boundary": {
                    "this_sidecar_is_not_a_controller_or_tournament_launcher": True,
                    "only_actual_loopback_hcli_complete_token_measurements_can_write_tg_evidence": True,
                },
            }
            _atomic_json(self.root_status, global_status)
        return global_status

    def watch(self, *, idle_seconds: float = 45.0) -> int:
        if idle_seconds <= 0:
            raise BaseTrueTpsGateError("idle_seconds must be positive")
        stopping = False

        def stop(_signal: int, _frame: Any) -> None:
            nonlocal stopping
            stopping = True

        old_term = signal.signal(signal.SIGTERM, stop)
        old_int = signal.signal(signal.SIGINT, stop)
        try:
            while not stopping:
                self.run_once()
                if not stopping:
                    time.sleep(idle_seconds)
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_PHYSICAL_ROOT)
    parser.add_argument("--hcli", type=Path, default=DEFAULT_HCLI_BINARY)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("once", help="reconcile one strict HCLI BASE_TRUE_TPS cycle")
    watch = commands.add_parser("watch", help="run idempotent detached gate cycles")
    watch.add_argument("--idle-seconds", type=float, default=45.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = BaseTrueTpsGate(physical_root=args.root, hcli_binary=args.hcli)
    if args.command == "watch":
        return runner.watch(idle_seconds=args.idle_seconds)
    status = runner.run_once()
    print(json.dumps({"status_path": str(runner.root_status), "models": [(row.get("model", {}).get("key"), row.get("phase")) for row in status["models"]]}, sort_keys=True))
    return 0


__all__ = [
    "BaseTrueTpsGate",
    "BaseTrueTpsGateError",
    "BenchSample",
    "DEFAULT_HCLI_BINARY",
    "DEFAULT_PHYSICAL_ROOT",
    "HCLI_RECEIPT_SCHEMA",
    "HCLI_RECEIPT_STATUS",
    "KERNEL_RECEIPT_SCHEMA",
    "KERNEL_RECEIPT_STATUS",
    "MEASURED_RUNS_PER_PROMPT",
    "MIN_MEASURED_REQUESTS",
    "MIN_TOTAL_COMPLETE_FORWARDS",
    "PROMPT_PROBES",
    "TG10_TPS",
    "TG3_TPS",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
