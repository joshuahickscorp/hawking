"""WorkUnit backend execution. One WorkUnit identity, several backends.

This is not a second DAG or scheduler. Mission still owns the loop.
Backend choice is an execution-policy decision recorded on the WorkUnit.
Grok text never marks a WorkUnit complete; a verifier must.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from .report_compiler import compile_backend_report


BACKEND_QWEN = "qwen"
BACKEND_GROK = "grok"
BACKEND_CPU = "cpu"
BACKEND_TOOL = "tool"


def select_backend_name(wu: Any) -> str:
    pref = str(
        getattr(wu, "provider", None)
        or getattr(wu, "preferred_backend", None)
        or ""
    ).strip().lower()
    if pref:
        return pref
    rc = str(getattr(wu, "resource_class", "") or "")
    if rc == "GROK":
        return BACKEND_GROK
    # Resource class is an admission constraint, not an automatic backend
    # switch. Default cognition stays on the local engine so existing
    # missions (COMPILE/TEST included) keep using execute_workunit.
    return BACKEND_QWEN


class WorkUnitExecutor:
    def __init__(
        self,
        workspace: Union[str, Path],
        engine: Any = None,
        grok_bridge: Any = None,
        providers: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.engine = engine
        self._grok = grok_bridge
        self.providers = dict(providers or {})

    def grok_bridge(self):
        if self._grok is not None:
            return self._grok
        from .grok_bridge import GrokBridge

        self._grok = GrokBridge(self.workspace)
        return self._grok

    def execute(self, wu: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = dict(context or {})
        name = select_backend_name(wu)
        wu.assigned_backend = name
        if name == BACKEND_GROK:
            return self._run_grok(wu, context)
        if name in (BACKEND_CPU, BACKEND_TOOL):
            return self._run_cpu(wu, context)
        provider = context.get("provider_instance") or self.providers.get(name)
        # A provider may intentionally be the same object as the engine (the
        # AgentOS ``resident`` alias is the normal example).  Only the
        # historical implicit ``qwen`` route is special; every explicitly
        # selected provider name must use the provider-neutral adapter so its
        # identity, capabilities, and receipt stay visible.
        if provider is not None and (provider is not self.engine or name != BACKEND_QWEN):
            return self._run_provider(provider, name, wu, context)
        return self._run_engine(wu, context, backend_name=name)

    def _run_provider(
        self,
        provider: Any,
        name: str,
        wu: Any,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Invoke a non-legacy provider without teaching AgentOS its transport."""
        started = time.time()
        workunit_call = getattr(provider, "execute_workunit", None)
        request_payload: Dict[str, Any]
        response_value: Any = None
        if callable(workunit_call):
            request_payload = {
                "unit_id": wu.id,
                "role": getattr(wu, "role", None),
                "description": getattr(wu, "description", None),
                "context": context,
            }
            raw = workunit_call(wu, context)
        else:
            prompt = str(context.get("prompt") or getattr(wu, "description", ""))
            request_payload = {
                "model": context.get("model") or name,
                "messages": [{"role": "user", "content": prompt}],
                # Keep explicit provider calls on the same bounded generation
                # budget as the engine path.  A provider-neutral WorkUnit
                # must not silently jump from HCLI_MODEL_TOKENS=64 to a
                # 1024-token request merely because it implements ``generate``
                # instead of the legacy Engine seam.
                "max_tokens": int(
                    context.get("max_tokens")
                    or os.environ.get("HCLI_MODEL_TOKENS", "1024")
                ),
            }
            payload = request_payload
            generate = getattr(provider, "generate", None)
            complete = getattr(provider, "complete", None)
            if callable(generate):
                from .providers import GenerationRequest

                response = generate(GenerationRequest.from_mapping(payload), timeout=context.get("timeout"))
                response_value = response
                raw = response.to_dict() if hasattr(response, "to_dict") else response
                if isinstance(raw, dict) and raw.get("text") is not None:
                    raw = {"content": raw.get("text"), "provider_response": raw}
            elif callable(complete):
                response = complete(payload, timeout=context.get("timeout"))
                response_value = response
                raw = getattr(response, "raw", response)
            elif callable(provider):
                raw = provider(payload)
            else:
                raise RuntimeError(f"provider {name!r} has no generate/complete/execute_workunit method")
        if not isinstance(raw, dict):
            raw = {"content": str(raw)}
        raw.setdefault("backend", name)
        raw.setdefault("provider", name)
        try:
            from .providers import ProviderReceipt, ResidentProfile

            profile = None
            profile_fn = getattr(provider, "profile", None)
            if callable(profile_fn):
                try:
                    profile = profile_fn()
                except Exception:
                    profile = None
            if isinstance(profile, ResidentProfile):
                model_id = profile.model_id
                profile_id = profile.profile_id
            else:
                identity_fn = getattr(provider, "identity", None)
                identity = identity_fn() if callable(identity_fn) else {}
                identity = identity if isinstance(identity, dict) else {}
                model_id = str(identity.get("model_id") or identity.get("model") or name)
                profile_id = None
            if response_value is not None and hasattr(response_value, "to_dict"):
                response_dict = response_value.to_dict()
            else:
                response_dict = raw
            raw.setdefault(
                "provider_receipt",
                ProviderReceipt(
                    provider=str(raw.get("provider") or name),
                    model_id=str(model_id),
                    profile_id=profile_id,
                    request=request_payload,
                    response=response_dict if isinstance(response_dict, dict) else {"value": response_dict},
                    started_at=started,
                    finished_at=time.time(),
                ).to_dict(),
            )
        except Exception as exc:  # noqa: BLE001 - provenance cannot block cognition
            raw.setdefault("provider_receipt_error", f"{type(exc).__name__}: {exc}")
        return raw

    def _run_engine(
        self,
        wu: Any,
        context: Dict[str, Any],
        *,
        backend_name: str = BACKEND_QWEN,
    ) -> Dict[str, Any]:
        engine = self.engine
        raw: Any = {}
        if engine is not None and hasattr(engine, "execute_workunit"):
            raw = engine.execute_workunit(wu, context)
        elif engine is not None and hasattr(engine, "execute"):
            raw = engine.execute(context.get("prompt") or wu.description)
        if not isinstance(raw, dict):
            raw = {"content": str(raw)}
        raw.setdefault("backend", backend_name)
        raw.setdefault("provider", backend_name)
        return raw

    # Kept as a narrow compatibility shim for callers that imported this
    # private helper while the default backend was named Qwen.
    def _run_qwen(self, wu: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_engine(wu, context, backend_name=BACKEND_QWEN)

    def _run_cpu(self, wu: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        cmd = (getattr(wu, "verifier", None) or "").strip()
        if not cmd:
            cmd = _infer_cpu_command(wu)
        if not cmd:
            return {
                "backend": BACKEND_CPU,
                "validation": {
                    "ok": False,
                    "reason": "NO_DETERMINISTIC_VALIDATION",
                },
            }
        # A verifier that cannot fail is not a verifier. verifier_pipeline
        # already knows how to spot one (`true`, `:`, `exit 0`,
        # `python3 -c 'raise SystemExit(0)'`, `sh -c true`), and the Ledger
        # refuses them -- but this executor ran them through shell=True and
        # accepted exit 0, so the same command the ledger rejects was accepted
        # here. Reuse the one detector rather than growing a second opinion.
        try:
            from .verifier_pipeline import command_is_admissible

            admissible, why = command_is_admissible(cmd)
        except Exception:
            # FAIL CLOSED. An audit caught this admitting the command when the
            # detector itself threw: a verifier that cannot be checked is not a
            # verifier we may trust, and defaulting to "admissible" hands an
            # attacker a way to bypass the check by breaking it.
            admissible, why = False, "VACUOUS_COMMAND"
        # Only the "cannot fail" verdicts apply here. command_is_admissible
        # also enforces the verifier pipeline's first-token allowlist, which is
        # deliberately narrower than what a WorkUnit verifier may legitimately
        # be -- `test -f x && grep -q nonce x` is a perfectly good check and
        # must not be refused as un-admitted.
        if not admissible and why in ("VACUOUS_COMMAND", "EMPTY_COMMAND"):
            return {
                "backend": BACKEND_CPU,
                "validation": {
                    "ok": False,
                    "reason": why,
                    "command": cmd,
                },
            }

        timeout = float(os.environ.get("HCLI_CPU_TIMEOUT", "120"))
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "backend": BACKEND_CPU,
                "validation": {
                    "ok": False,
                    "reason": f"timeout after {timeout}s",
                    "output": (exc.stdout or "") + "\n" + (exc.stderr or ""),
                },
            }
        output = (proc.stdout or "") + (proc.stderr or "")
        compact = compile_backend_report(
            backend=BACKEND_CPU,
            task_id=wu.id,
            raw_text=output,
            extra={"verifier_inputs": [cmd]},
        )
        return {
            "backend": BACKEND_CPU,
            "compact": compact,
            "validation": {
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "command": cmd,
                "output": output[-4000:],
            },
        }

    def _run_grok(self, wu: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        bridge = self.grok_bridge()
        prompt = context.get("prompt") or wu.description
        handle = bridge.consult(str(prompt), background=True)
        wu.backend_task_id = handle.task_id
        timeout = float(os.environ.get("HCLI_GROK_WAIT", "3600"))
        status = bridge.wait(handle.task_id, timeout=timeout)
        compact = {}
        compact_fn = getattr(bridge, "compact_report", None)
        if callable(compact_fn):
            try:
                compact = compact_fn(handle.task_id)
            except Exception as exc:
                compact = {"errors": [str(exc)], "task_id": handle.task_id}
        # The terminal state is consulted BEFORE the verifier is allowed to
        # decide. A verifier can pass for reasons that have nothing to do with
        # the Grok task -- it might touch a file the harness already wrote, or
        # simply exit 0 -- so letting it speak for a task that failed, errored,
        # timed out, was refused, or whose process is gone turns a failure into
        # an acceptance. `stale-running` is included: a status file left at
        # running by a dead process is not a success either.
        terminal = str(status.get("state") or "").strip().lower()
        if terminal in ("failed", "error", "errored", "timeout", "timed-out",
                        "refused", "cancelled", "stale-running", "unknown"):
            return {
                "backend": BACKEND_GROK,
                "backend_task_id": handle.task_id,
                "status": status,
                "compact": compact,
                "validation": {
                    "ok": False,
                    "reason": "GROK_TERMINAL_STATE_NOT_SUCCESSFUL",
                    "grok_state": terminal,
                },
            }

        verifier = (getattr(wu, "verifier", None) or "").strip()
        if verifier:
            try:
                verifier = verifier.format(
                    backend_task_id=handle.task_id,
                    workspace=str(self.workspace),
                    task_id=handle.task_id,
                )
                wu.verifier = verifier
            except (KeyError, IndexError, ValueError):
                pass
        validation: Dict[str, Any]
        if verifier:
            cpu = self._run_cpu(wu, context)
            validation = cpu.get("validation") or {
                "ok": False,
                "reason": "NO_DETERMINISTIC_VALIDATION",
            }
        else:
            # Grok text is evidence, not acceptance.
            validation = {
                "ok": False,
                "reason": "GROK_REQUIRES_VERIFIER",
                "grok_state": status.get("state"),
            }
        return {
            "backend": BACKEND_GROK,
            "backend_task_id": handle.task_id,
            "status": status,
            "compact": compact,
            "validation": validation,
        }


def _infer_cpu_command(wu: Any) -> str:
    desc = str(getattr(wu, "description", "") or "")
    if desc.startswith("python") or desc.startswith("pytest") or desc.startswith("cargo"):
        return desc
    return ""


# --- worker-context plug-in (added by the context-compiler lane) -------------
# Engine.execute is a root-goal API. These adapters keep the root goal out of
# worker payloads: the compiled WorkerPacket is the only user text a worker
# model sees. Kept alongside WorkUnitExecutor rather than replacing it -- the
# lane that wrote them could not see this file and shipped a module that would
# have deleted the whole worker execution layer.
import inspect
import threading
from typing import Sequence

from .engine import Engine, EngineError
from .goal import WorkerPacket


def gather_evidence_paths(engine: Any, paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Read only the listed files. Never scan the ultragoal blob."""
    listed = [str(path).strip() for path in paths if str(path).strip()]
    if not listed:
        return []
    gather = getattr(engine, "_gather_evidence", None)
    if not callable(gather):
        return []
    return gather(" ".join(listed))


def execute_workunit(self: Any, wu: Any, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Run one WorkUnit from a compiled packet. Does not compile a new root goal."""
    del wu  # identity lives on the packet / context; Engine.execute keys off prompt
    context = context or {}
    prompt = str(context.get("prompt") or "").strip()
    if not prompt:
        raise EngineError("worker path requires execute_workunit")
    if "GOAL:" in prompt:
        raise EngineError("worker packet must not contain GOAL:")
    paths = [str(item) for item in (context.get("evidence_paths") or [])]
    compiled = context.get("compiled")
    if not isinstance(compiled, dict):
        compiled = {}
    evidence = gather_evidence_paths(self, paths)

    return self.execute(prompt, evidence=evidence, compiled=compiled)


def dispatch_workunit(engine: Any, wu: Any, context: Optional[Dict[str, Any]]) -> Any:
    """Mission worker path. Missing execute_workunit is a hard error, not a dump."""
    if engine is None:
        raise EngineError("worker path requires execute_workunit")
    fn = getattr(engine, "execute_workunit", None)
    if not callable(fn):
        raise EngineError("worker path requires execute_workunit")
    return fn(wu, context)


def consult_worker(bridge: Any, packet: WorkerPacket, **kwargs: Any) -> Any:
    """Grok consult sees the compiled packet, never the root goal."""
    if not isinstance(packet, WorkerPacket):
        raise EngineError("Grok consult for a worker requires a WorkerPacket")
    if "GOAL:" in packet.prompt:
        raise EngineError("worker packet must not contain GOAL:")
    consult = getattr(bridge, "consult", None)
    if not callable(consult):
        raise EngineError("Grok consult bridge is missing consult()")
    return consult(packet.prompt, **kwargs)


def _install_engine_hooks() -> None:
    current = getattr(Engine, "_build_model_payload", None)
    if current is not None and not getattr(current, "_hcli_worker_payload", False):
        try:
            src = inspect.getsource(current)
        except (OSError, TypeError):
            src = ""
        if "del compiled" in src and "GOAL:" in src:

            def _build_model_payload(
                self: Any,
                prompt: str,
                evidence: Optional[List[Dict[str, Any]]] = None,
                compiled: Any = None,
                *,
                kind: str = "worker",
                enable_thinking: Optional[bool] = None,
                response_schema: Optional[bool] = None,
            ) -> Dict[str, Any]:
                payload = current(
                    self,
                    prompt,
                    evidence,
                    compiled,
                    enable_thinking=enable_thinking,
                    response_schema=response_schema,
                )
                if kind == "root":
                    return payload
                messages = payload.get("messages") or []
                if (
                    len(messages) >= 2
                    and messages[1].get("role") == "user"
                ):
                    user = messages[1].get("content") or ""
                    if user.startswith("GOAL:\n"):
                        messages[1]["content"] = user[len("GOAL:\n"):]
                return payload

            _build_model_payload._hcli_worker_payload = True  # type: ignore[attr-defined]
            Engine._build_model_payload = _build_model_payload

    if getattr(Engine, "execute_workunit", None) is None:
        Engine.execute_workunit = execute_workunit  # type: ignore[assignment]


_install_engine_hooks()


__all__ = [
    "consult_worker",
    "dispatch_workunit",
    "execute_workunit",
    "gather_evidence_paths",
]
