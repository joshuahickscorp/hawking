"""Live HCLI/native reproduction ladder.

This is a bounded qualification command for the current native profile.  It
does not turn a fixture into a claim: every stage records the provider
identity, request/response shape, and the exact failure boundary.  The
profile's prompt fallback remains data owned by that profile, so another
native provider can supply a different prompt contract.
"""
from __future__ import annotations

import json
import os
import selectors
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.native_gate.v1"
DEFAULT_PROMPT = "Return exactly: HAWKING_OK"


def _profile_path(profile: Optional[str], repo_root: Optional[Path]) -> Path:
    value = profile or os.environ.get("HCLI_HAWKING_NATIVE_CONFIG")
    if value:
        return Path(value).expanduser().resolve()
    root = repo_root or Path(__file__).resolve().parents[2]
    return (root / "hcli" / "hawking-native.sealed-3.14.json").resolve()


def _identity(config: Any) -> Dict[str, Any]:
    value = config.identity()
    return {
        key: value.get(key)
        for key in (
            "resident_identity",
            "provider",
            "model_id",
            "family",
            "runtime",
            "protocol",
            "artifact_root",
            "tokenizer",
            "binary",
            "resident_binary",
            "mode",
            "physical_ebpw",
            "current_runtime",
        )
        if key in value
    }


def _text_from_openai(body: Any) -> Optional[str]:
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(choice.get("text"), str):
            return choice["text"]
    if isinstance(body.get("content"), str):
        return body["content"]
    return None


def _response_summary(body: Any) -> Dict[str, Any]:
    text = body.get("generated_text", body.get("text")) if isinstance(body, dict) else None
    if text is None:
        text = _text_from_openai(body)
    usage = body.get("usage") if isinstance(body, dict) else None
    choice = {}
    if isinstance(body, dict) and isinstance(body.get("choices"), list) and body["choices"]:
        choice = body["choices"][0] if isinstance(body["choices"][0], dict) else {}
    token_ids = body.get("new_token_ids") if isinstance(body, dict) else None
    return {
        "id": body.get("id") if isinstance(body, dict) else None,
        "status": body.get("status") if isinstance(body, dict) else None,
        "finish_reason": (
            choice.get("finish_reason")
            if isinstance(choice, dict) and choice.get("finish_reason") is not None
            else body.get("finish_reason") if isinstance(body, dict) else None
        ),
        "text_len": len(text) if isinstance(text, str) else None,
        "text_preview": repr(text)[:500] if isinstance(text, str) else None,
        "generated_tokens": body.get("generated_tokens") if isinstance(body, dict) else None,
        "new_token_ids_len": len(token_ids) if isinstance(token_ids, list) else None,
        "prompt_tokens": (
            (usage or {}).get("prompt_tokens")
            if isinstance(usage, dict)
            else body.get("prompt_tokens", body.get("prompt_len"))
            if isinstance(body, dict)
            else None
        ),
        "completion_tokens": (
            (usage or {}).get("completion_tokens")
            if isinstance(usage, dict)
            else body.get("completion_tokens", body.get("generated_tokens"))
            if isinstance(body, dict)
            else None
        ),
        "fallbacks": body.get("fallbacks") if isinstance(body, dict) else None,
        "error": body.get("error") if isinstance(body, dict) else None,
    }


def _stage(
    name: str,
    *,
    config: Any,
    started: float,
    response: Any = None,
    passed: bool = False,
    **extra: Any,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "stage": name,
        "status": "PASSED" if passed else "FAILED",
        "elapsed_s": round(time.monotonic() - started, 3),
        "identity": _identity(config),
    }
    if response is not None:
        result["response"] = _response_summary(response)
    result.update(extra)
    return result


def _read_json_line(
    proc: subprocess.Popen[str],
    selector: selectors.BaseSelector,
    deadline: float,
) -> Optional[Dict[str, Any]]:
    while time.monotonic() < deadline:
        events = selector.select(max(0.05, min(1.0, deadline - time.monotonic())))
        if not events:
            if proc.poll() is not None:
                return None
            continue
        stream = proc.stdout
        if stream is None:
            return None
        line = stream.readline()
        if not line:
            return None
        body = json.loads(line)
        if not isinstance(body, dict):
            raise ValueError("native stdout record is not an object")
        return body
    return None


def _stop_process(proc: subprocess.Popen[str]) -> Tuple[Optional[int], str]:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    stderr = proc.stderr.read() if proc.stderr is not None else ""
    return proc.returncode, stderr[-1600:]


def _direct_resident_call(config: Any, prompt: str, timeout_s: float) -> Dict[str, Any]:
    env = os.environ.copy()
    env.update(config.fusion_env)
    command = [
        config.resident_binary,
        "--artifact-root",
        config.artifact_root,
        "--tokenizer",
        config.tokenizer,
        "--max-seq-len",
        str(config.max_seq_len),
        "--resident-identity",
        config.resident_identity,
    ]
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
        start_new_session=True,
    )
    selector = selectors.DefaultSelector()
    if proc.stdout is None:
        raise RuntimeError("native resident stdout is unavailable")
    selector.register(proc.stdout, selectors.EVENT_READ)
    request_id = "native-gate-a1"
    try:
        ready = _read_json_line(proc, selector, time.monotonic() + timeout_s)
        if not ready or ready.get("status") != "ready":
            raise RuntimeError(f"resident did not become ready: {ready}")
        if proc.stdin is None:
            raise RuntimeError("native resident stdin is unavailable")
        proc.stdin.write(
            json.dumps(
                {
                    "id": request_id,
                    "prompt": prompt,
                    "max_new_tokens": 32,
                    "max_seq_len": config.max_seq_len,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        proc.stdin.flush()
        response = _read_json_line(proc, selector, time.monotonic() + timeout_s)
        if not response or response.get("id") != request_id:
            raise RuntimeError(f"correlated response missing: {response}")
        return {"ready": ready, "response": response}
    finally:
        selector.close()
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        return_code, stderr_tail = _stop_process(proc)
        # Attach diagnostics to the returned body without allowing stderr to
        # become part of the model result contract.
        if "response" in locals() and isinstance(response, dict):
            response.setdefault("_gate_process", {})
            response["_gate_process"].update(
                {"returncode": return_code, "stderr_tail": stderr_tail}
            )


@contextmanager
def _temporary_generation_env(model_tokens: int) -> Iterator[None]:
    names = {
        "HCLI_MODEL_TOKENS": str(max(1, int(model_tokens))),
        "HCLI_STRUCTURED_OUTPUT_ATTEMPTS": "1",
    }
    old = {name: os.environ.get(name) for name in names}
    os.environ.update(names)
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _write_receipt(report: Dict[str, Any], emit: Optional[str], repo_root: Path) -> None:
    destination = Path(emit).expanduser() if emit else repo_root / "receipts" / "headless" / "HCLI_AGENTOS_NATIVE_GATE.json"
    if not destination.is_absolute():
        destination = repo_root / destination
    report["receipt_path"] = str(destination.resolve())
    atomic_write_json(destination, report)


def run_native_gate(
    workspace: Optional[str | os.PathLike[str]] = None,
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    profile: Optional[str] = None,
    prompt: str = DEFAULT_PROMPT,
    emit: Optional[str] = None,
    timeout_s: float = 180.0,
    model_tokens: int = 64,
) -> Dict[str, Any]:
    """Run and persist the live native reproduction ladder."""
    from hcli.backends import NoeticNativeBackend
    from hcli.controller import Controller
    from hcli.hawking_native import HawkingNativeConfig, HawkingNativeConnector, _TokenizerRenderer
    from hcli.workspace import Workspace

    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    work = Path(workspace).expanduser().resolve() if workspace else repo
    work.mkdir(parents=True, exist_ok=True)
    profile_path = _profile_path(profile, repo)
    config = HawkingNativeConfig.from_file(str(profile_path))
    rendered = _TokenizerRenderer(config).render(
        [{"role": "user", "content": prompt}],
        thinking_requested=False,
    )
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "qualification": "LIVE_NATIVE_HCLI_LADDER",
        "started_at": time.time(),
        "profile_path": str(profile_path),
        "prompt": prompt,
        "prompt_contract": {
            "chars": len(rendered.text),
            "estimated_tokens": rendered.prompt_tokens,
            "token_count_source": rendered.token_count_source,
            "thinking_qualified": rendered.thinking_qualified,
            "fallback_template": config.prompt_contract.get("fallback_template"),
        },
        "stages": [],
        "errors": [],
    }

    def record(name: str, fn: Any) -> Optional[Tuple[Any, float]]:
        started = time.monotonic()
        try:
            value = fn()
            return value, started
        except Exception as exc:  # noqa: BLE001 - the receipt must show the boundary
            report["stages"].append(
                _stage(
                    name,
                    config=config,
                    started=started,
                    passed=False,
                    error_type=type(exc).__name__,
                    error=str(exc)[:2000],
                )
            )
            report["errors"].append({"stage": name, "type": type(exc).__name__, "error": str(exc)[:2000]})
            return None

    direct_record = record("A1_direct_resident", lambda: _direct_resident_call(config, rendered.text, timeout_s))
    if direct_record is not None:
        direct, direct_started = direct_record
        response = direct["response"]
        text = response.get("generated_text", response.get("text"))
        report["stages"].append(
            _stage(
                "A1_direct_resident",
                config=config,
                started=direct_started,
                response=response,
                passed=isinstance(text, str) and "HAWKING_OK" in text,
                backend_class="subprocess-resident",
                pid=(direct.get("ready") or {}).get("resident_pid"),
                request_id=response.get("id"),
                ready=_response_summary(direct.get("ready")),
            )
        )

    def run_connector() -> Dict[str, Any]:
        connector = HawkingNativeConnector(config)
        try:
            connector.start(timeout=timeout_s)
            raw = connector.complete_payload(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 32,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=timeout_s,
            )
            return {"raw": raw, "pid": connector.pid}
        finally:
            connector.stop()

    connector_record = record("A2_hawking_native_backend", run_connector)
    if connector_record is not None:
        connector_result, connector_started = connector_record
        raw = connector_result["raw"]
        text = _text_from_openai(raw)
        report["stages"].append(
            _stage(
                "A2_hawking_native_backend",
                config=config,
                started=connector_started,
                response=raw,
                passed=isinstance(text, str) and "HAWKING_OK" in text,
                backend_class="HawkingNativeConnector",
                pid=connector_result.get("pid"),
            )
        )

    def run_backend() -> Dict[str, Any]:
        backend = NoeticNativeBackend(model_path=str(profile_path))
        try:
            backend.spawn()
            result = backend.complete(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "model": config.resident_identity,
                    "max_tokens": 32,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=timeout_s,
            )
            return {"result": result, "pid": backend.pid}
        finally:
            backend.stop()

    backend_record = record("A3_provider_abstraction", run_backend)
    if backend_record is not None:
        backend_result, backend_started = backend_record
        result = backend_result["result"]
        raw = result.raw if hasattr(result, "raw") else result
        text = result.text if hasattr(result, "text") else _text_from_openai(raw)
        report["stages"].append(
            _stage(
                "A3_provider_abstraction",
                config=config,
                started=backend_started,
                response=raw,
                passed=isinstance(text, str) and "HAWKING_OK" in text,
                backend_class="NoeticNativeBackend",
                pid=backend_result.get("pid"),
                completion_result={
                    "finish_reason": getattr(result, "finish_reason", None),
                    "degraded": list(getattr(result, "degraded", []) or []),
                },
            )
        )

    controller = Controller(work, model=str(profile_path))
    try:
        with _temporary_generation_env(model_tokens):
            plain_record = record("A4_hcli_plain_cognition", lambda: controller.complete_text(prompt))
            if plain_record is not None:
                plain, plain_started = plain_record
            else:
                plain = None
            if isinstance(plain, str):
                report["stages"].append(
                    _stage(
                        "A4_hcli_plain_cognition",
                        config=config,
                        started=plain_started,
                        passed="HAWKING_OK" in plain,
                        backend_class="Controller/Engine.complete_text",
                        text_len=len(plain),
                        text_preview=repr(plain)[:500],
                    )
                )

            structured_record = record("A5_hcli_structured_cognition", lambda: controller.execute(prompt))
            if structured_record is not None:
                structured, structured_started = structured_record
            else:
                structured = None
            if isinstance(structured, dict):
                report["stages"].append(
                    _stage(
                        "A5_hcli_structured_cognition",
                        config=config,
                        started=structured_started,
                        passed=(
                            structured.get("status") == "completed"
                            and structured.get("kind") == "answer"
                            and structured.get("content") == "HAWKING_OK"
                        ),
                        backend_class="Controller/Engine.execute",
                        result={k: structured.get(k) for k in ("kind", "content", "status", "error", "receipt")},
                        receipt=structured.get("receipt"),
                    )
                )
    finally:
        controller.shutdown()

    def run_cli() -> Dict[str, Any]:
        executable = shutil.which("hcli")
        if executable:
            command = [executable]
        else:
            command = [sys.executable, "-m", "hcli"]
        command.extend(["--model", str(profile_path), "--task", prompt])
        env = os.environ.copy()
        env["HCLI_MODEL_TOKENS"] = str(max(1, int(model_tokens)))
        env["HCLI_STRUCTURED_OUTPUT_ATTEMPTS"] = "1"
        completed = subprocess.run(
            command,
            cwd=str(repo),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-1200:],
            "stderr": completed.stderr[-1200:],
        }

    cli_record = record("A6_full_hcli_task", run_cli)
    if cli_record is not None:
        cli_result, cli_started = cli_record
        report["stages"].append(
            _stage(
                "A6_full_hcli_task",
                config=config,
                started=cli_started,
                passed=cli_result["returncode"] == 0 and "HAWKING_OK" in cli_result["stdout"],
                backend_class="hcli-command",
                **cli_result,
            )
        )

    passed_names = {stage["stage"] for stage in report["stages"] if stage.get("status") == "PASSED"}
    required = {
        "A1_direct_resident",
        "A2_hawking_native_backend",
        "A3_provider_abstraction",
        "A4_hcli_plain_cognition",
        "A5_hcli_structured_cognition",
        "A6_full_hcli_task",
    }
    report["checks"] = {name: name in passed_names for name in sorted(required)}
    report["status"] = "PASSED" if required.issubset(passed_names) else "FAILED"
    report["finished_at"] = time.time()
    _write_receipt(report, emit, repo)
    return report


__all__ = ["DEFAULT_PROMPT", "SCHEMA", "run_native_gate"]
