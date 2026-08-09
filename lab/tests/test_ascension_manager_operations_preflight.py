from __future__ import annotations

import json
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from lab.operators import ascension_manager_operations_preflight as preflight
from lab.operators import ascension_physical_gatekeeper as gatekeeper
from lab.receipts import seal


class _RoomyThreadingHTTPServer(ThreadingHTTPServer):
    # The real preflight deliberately bursts eight logical sessions at once;
    # retain enough pending sockets for the test double to model that behavior.
    request_queue_size = 32


def _write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")


class _Endpoint:
    def __init__(self, *, spec: gatekeeper.ModelSpec, runtime_seal: str, hcli_seal: str) -> None:
        self.spec = spec
        self.runtime_seal = runtime_seal
        self.hcli_seal = hcli_seal
        self.instance = 1
        self.server = _RoomyThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def _body(self) -> dict[str, Any]:
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                return json.loads(raw.decode("utf-8")) if raw else {}

            def _json(self, payload: dict[str, Any]) -> None:
                raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                if self.path != "/healthz":
                    self.send_error(404)
                    return
                self._json(
                    {
                        "ready": True,
                        "model_alone": True,
                        "fallback_count": 0,
                        "gravity_artifact_id": endpoint.spec.gravity_artifact_id,
                        "server_instance_id": f"instance-{endpoint.instance}",
                    }
                )

            def do_POST(self) -> None:  # noqa: N802
                payload = self._body()
                if self.path == "/v1/hawking/raw-decode":
                    self._json(
                        {
                            "measurement": {
                                "timing_scope": "complete_model_token_loop",
                                "uses_exact_native_runtime": True,
                                "full_token_execution": True,
                                "model_alone": True,
                                "no_fallback": True,
                                "runtime_receipt_seal_sha256": endpoint.runtime_seal,
                                "measured_token_count": 10,
                                "elapsed_seconds": 0.1,
                                "base_true_tokens_per_second": 100.0,
                            }
                        }
                    )
                    return
                if self.path == "/v1/chat/completions":
                    session_id = self.headers.get(preflight.SESSION_HEADER)
                    messages = payload.get("messages") or []
                    self._json(
                        {
                            "choices": [{"message": {"role": "assistant", "content": f"ok {session_id}"}}],
                            "usage": {"completion_tokens": 2},
                            preflight.TELEMETRY_KEY: {
                                "session_id": session_id,
                                "gravity_artifact_id": endpoint.spec.gravity_artifact_id,
                                "weight_body_id": endpoint.spec.gravity_artifact_id,
                                "weight_reuse_observed": True,
                                "no_fallback": True,
                                "context_reused": len(messages) >= 3,
                                "kv_state_bytes": 4096,
                                "context_compile_latency_ms": 1.0,
                                "tool_wait_ms": 0.25,
                                "queue_wait_ms": 0.5,
                            },
                        }
                    )
                    return
                if self.path != "/v1/hawking/manager-ops":
                    self.send_error(404)
                    return
                operation = payload.get("operation")
                if operation == "endpoint_restart":
                    endpoint.instance += 1
                response: dict[str, Any] = {
                    "operation": operation,
                    "completed": True,
                    "gravity_artifact_id": endpoint.spec.gravity_artifact_id,
                    "runtime_receipt_seal_sha256": endpoint.runtime_seal,
                    "hcli_receipt_seal_sha256": endpoint.hcli_seal,
                    "no_fallback": True,
                    "isolated_non_destructive": True,
                }
                if operation in {"acquire_quiet_benchmark_lease", "release_quiet_benchmark_lease"}:
                    response.update({"lease_id": payload.get("lease_id", "lease-1"), "exclusive_gpu": True})
                if operation == "residency_probe":
                    response.update(
                        {
                            "resident_model_body_count": 1,
                            "weight_body_id": endpoint.spec.gravity_artifact_id,
                            "logical_sessions": [1, 2, 4, 8],
                        }
                    )
                if operation == "tool_recovery_probe":
                    response.update({"tool_recovery_passed": True, "tool_wait_ms": 0.25})
                if operation == "rollback_probe":
                    response["rollback_passed"] = True
                if operation == "storage_rollback_probe":
                    response.update({"storage_rollback_passed": True, "disk_free_delta_bytes": 0})
                if operation == "endpoint_restart":
                    response["restart_requested"] = True
                self._json(response)

        return Handler


def _endpoint_contract(spec: gatekeeper.ModelSpec, port: int) -> dict[str, Any]:
    return {
        "schema": preflight.ENDPOINT_SCHEMA,
        "protocol": "openai_chat_completions_v1",
        "host": "127.0.0.1",
        "port": port,
        "model": spec.gravity_artifact_id,
        "gravity_artifact_id": spec.gravity_artifact_id,
        "health_path": "/healthz",
        "chat_path": "/v1/chat/completions",
        "operations": {
            "session_header": preflight.SESSION_HEADER,
            "response_telemetry_field": preflight.TELEMETRY_KEY,
            "raw_decode_path": "/v1/hawking/raw-decode",
            "control_path": "/v1/hawking/manager-ops",
            "control_operations": list(preflight.CONTROL_OPERATIONS),
            "control_probes_are_isolated_and_non_destructive": True,
            "single_model_body_shared_across_sessions": True,
            "no_host_shell_or_hidden_membership_access": True,
        },
    }


def _bind_ready_state(
    monkeypatch: pytest.MonkeyPatch, root: Path, endpoint: _Endpoint
) -> tuple[gatekeeper.ModelSpec, dict[str, Any], dict[str, Any]]:
    spec = gatekeeper.MODEL_SPECS[0]
    runtime = seal(
        {
            "schema": gatekeeper.RUNTIME_SCHEMA,
            "status": "PASS_EXACT_NATIVE_FULL_TOKEN_RUNTIME",
            "binding": {
                "model_id": spec.model_id,
                "runtime_executable_sha256": "d" * 64,
            },
            "runtime": {},
        }
    )
    hcli = seal(
        {
            "schema": gatekeeper.HCLI_SCHEMA,
            "status": "PASS_MEASURED_HCLI",
            "binding": {"runtime_receipt_seal_sha256": runtime["seal_sha256"]},
            "measurement": {"manager_operations_endpoint": _endpoint_contract(spec, endpoint.port)},
        }
    )
    paths = gatekeeper._paths(root, spec)
    _write(paths["runtime"], runtime)
    _write(paths["hcli"], hcli)
    suite_seal = "a" * 64

    def fake_gate(_root: Path) -> dict[str, Any]:
        return {
            "models": [
                {
                    "key": spec.key,
                    "requirements": {
                        "native_exact_full_token_runtime": {
                            "state": "PASS",
                            "seal_sha256": runtime["seal_sha256"],
                        },
                        "measured_hcli": {"state": "PASS", "seal_sha256": hcli["seal_sha256"]},
                    },
                }
            ]
        }

    monkeypatch.setattr(preflight.gatekeeper, "build_gate_status", fake_gate)
    monkeypatch.setattr(
        preflight.physical_tournament,
        "validate_suite_preflight",
        lambda _root: {"passed": True, "seal_sha256": suite_seal, "path": root / "suite.json", "reasons": []},
    )
    return spec, runtime, hcli


def test_preflight_runs_real_endpoint_sessions_but_never_writes_final_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = gatekeeper.MODEL_SPECS[0]
    endpoint = _Endpoint(spec=spec, runtime_seal="pending", hcli_seal="pending")
    endpoint.start()
    try:
        bound_spec, runtime, hcli = _bind_ready_state(monkeypatch, tmp_path / "physical", endpoint)
        endpoint.runtime_seal = runtime["seal_sha256"]
        endpoint.hcli_seal = hcli["seal_sha256"]
        binding, reasons, _ = preflight.readiness(tmp_path / "physical", bound_spec)
        assert reasons == []
        assert binding is not None

        result = preflight.run_attempt(tmp_path / "physical", bound_spec, fingerprint=binding.fingerprint)

        assert result["schema"] == preflight.RESULT_SCHEMA
        assert result["status"] == preflight.RESULT_STATUS
        assert result["preflight_complete"] is True
        assert [row["logical_sessions"] for row in result["session_measurements"]] == [1, 2, 4, 8]
        assert all(row["raw_model_tps"] == 100.0 for row in result["session_measurements"])
        assert result["endpoint_restart"] is not None
        assert result["claim_boundary"]["does_not_write_or_replace_final_manager_operations_receipt"] is True
        assert not gatekeeper._paths(tmp_path / "physical", bound_spec)["manager_operations"].exists()
        assert (tmp_path / "physical" / bound_spec.key / "agent-os" / "preflight-runs").is_dir()
    finally:
        endpoint.close()


def test_readiness_refuses_unsealed_manager_operations_endpoint_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = gatekeeper.MODEL_SPECS[0]
    endpoint = _Endpoint(spec=spec, runtime_seal="pending", hcli_seal="pending")
    endpoint.start()
    try:
        bound_spec, runtime, hcli = _bind_ready_state(monkeypatch, tmp_path / "physical", endpoint)
        endpoint.runtime_seal = runtime["seal_sha256"]
        endpoint.hcli_seal = hcli["seal_sha256"]
        hcli_path = gatekeeper._paths(tmp_path / "physical", bound_spec)["hcli"]
        malformed = seal(
            {
                "schema": gatekeeper.HCLI_SCHEMA,
                "status": "PASS_MEASURED_HCLI",
                "binding": {"runtime_receipt_seal_sha256": runtime["seal_sha256"]},
                "measurement": {},
            }
        )
        _write(hcli_path, malformed)

        monkeypatch.setattr(
            preflight.gatekeeper,
            "build_gate_status",
            lambda _root: {
                "models": [
                    {
                        "key": bound_spec.key,
                        "requirements": {
                            "native_exact_full_token_runtime": {
                                "state": "PASS",
                                "seal_sha256": runtime["seal_sha256"],
                            },
                            "measured_hcli": {"state": "PASS", "seal_sha256": malformed["seal_sha256"]},
                        },
                    }
                ]
            },
        )

        binding, reasons, _ = preflight.readiness(tmp_path / "physical", bound_spec)

        assert binding is None
        assert any("manager_operations_endpoint" in reason for reason in reasons)
    finally:
        endpoint.close()


def test_readiness_refuses_a_revoked_runtime_even_if_a_stale_gate_snapshot_says_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = gatekeeper.MODEL_SPECS[0]
    endpoint = _Endpoint(spec=spec, runtime_seal="pending", hcli_seal="pending")
    endpoint.start()
    try:
        bound_spec, runtime, hcli = _bind_ready_state(monkeypatch, tmp_path / "physical", endpoint)
        endpoint.runtime_seal = runtime["seal_sha256"]
        endpoint.hcli_seal = hcli["seal_sha256"]
        paths = gatekeeper._paths(tmp_path / "physical", bound_spec)
        raw = paths["runtime"].read_bytes()
        archive = (
            paths["runtime"].parent
            / "runtime-receipt-history"
            / f"{bound_spec.prefix}_EXACT_FULL_TOKEN_RUNTIME_RECEIPT_{runtime['seal_sha256']}.json"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(raw)
        archive_sha = hashlib.sha256(raw).hexdigest()
        _write(
            paths["runtime_supersession"],
            seal(
                {
                    "schema": gatekeeper.RUNTIME_SUPERSESSION_SCHEMA,
                    "status": "REVOKED_TEST_RUNTIME_DEFECT",
                    "recorded_at": "2026-08-08T00:00:00Z",
                    "binding": {
                        "model_id": bound_spec.model_id,
                        "canonical_runtime_receipt_path": str(paths["runtime"]),
                        "superseded_runtime_receipt_seal_sha256": runtime["seal_sha256"],
                        "defective_runtime_executable_sha256": "d" * 64,
                        "archived_runtime_receipt_path": str(archive),
                        "archived_runtime_receipt_document_sha256": archive_sha,
                    },
                    "revoked_runtime": {
                        "canonical_receipt_path": str(paths["runtime"]),
                        "canonical_receipt_seal_sha256": runtime["seal_sha256"],
                        "complete_manifest_seal_sha256": "e" * 64,
                        "model_id": bound_spec.model_id,
                        "runtime_executable_sha256": "d" * 64,
                    },
                    "historical_pass_archive_path": str(archive),
                    "historical_pass_archive_sha256": archive_sha,
                    "defect": {"class": "TEST"},
                    "invalidates": {
                        "canonical_native_runtime_pass": True,
                        "all_old_full_token_prompt_and_profile_controls_bound_to_runtime_sha": True,
                        "native_http_adapter_and_transport_handoff_bound_to_runtime_sha": True,
                        "any_hcli_tps_tg_capability_or_tournament_consumer_of_that_sha": True,
                    },
                    "required_before_reissue": ["new executable"],
                    "consumer_contract": {
                        "fail_closed_if_canonical_status_is_not_pass": True,
                        "fail_closed_if_this_supersession_revokes_the_bound_receipt_seal_or_runtime_executable_sha256": True,
                        "historical_archive_is_for_negative_science_only_not_a_gate_authority": True,
                    },
                    "claim_boundary": {"revocation": True},
                }
            ),
        )

        binding, reasons, details = preflight.readiness(tmp_path / "physical", bound_spec)

        assert binding is None
        assert any("revoked, superseded" in reason for reason in reasons)
        assert details["runtime_authority"]["state"] == "CURRENT_RUNTIME_REVOKED"
    finally:
        endpoint.close()


def test_failed_health_is_a_sealed_blocked_preflight_not_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = gatekeeper.MODEL_SPECS[0]
    endpoint = _Endpoint(spec=spec, runtime_seal="pending", hcli_seal="pending")
    endpoint.start()
    try:
        bound_spec, runtime, hcli = _bind_ready_state(monkeypatch, tmp_path / "physical", endpoint)
        endpoint.runtime_seal = runtime["seal_sha256"]
        endpoint.hcli_seal = hcli["seal_sha256"]
        binding, reasons, _ = preflight.readiness(tmp_path / "physical", bound_spec)
        assert binding is not None and not reasons
    finally:
        endpoint.close()

    result = preflight.run_attempt(tmp_path / "physical", bound_spec, fingerprint=binding.fingerprint)
    assert result["status"] == preflight.BLOCKED_RESULT_STATUS
    assert result["preflight_complete"] is False
    assert result["errors"]
