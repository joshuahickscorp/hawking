"""Operational AgentOS facade over HCLI's canonical mission authorities."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

from hcli.goal import GoalCompiler
from hcli.mission import Mission, mission_state_path
from hcli.persist import atomic_write_json
from hcli.providers import ResidentProfile, RoleRouter, profile_from_backend
from hcli.result_envelope import build_result_envelope
from hcli.runtime_iface import model_semantics_for
from hcli.tool_registry import ToolRegistry, ToolResult, default_tool_registry
from hcli.agentos.background import BackgroundJobStore


CHECKPOINT_SCHEMA = "hcli.agentos.control_checkpoint.v1"


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "hcli").is_dir() and (path / "pyproject.toml").is_file():
            return path
    # A caller may intentionally place mission state in a temporary or
    # external workspace. Keep repository-backed read tools connected to the
    # installed HCLI project instead of silently turning that workspace into
    # a fake repository root.
    here = Path(__file__).resolve()
    for path in (here.parent, *here.parents):
        if (path / "hcli").is_dir() and (path / "pyproject.toml").is_file():
            return path
    return start


class AgentOS:
    """One durable control-plane facade.

    Mission, Scheduler, DAG, MutationLock, and verifier remain the authorities
    they were before this facade existed.  AgentOS owns composition: it gives
    them a provider-neutral model surface, a typed tool registry, and a
    restart/recovery entry point.
    """

    def __init__(
        self,
        workspace: Union[str, os.PathLike[str]],
        *,
        engine: Any = None,
        controller: Any = None,
        model: Optional[str] = None,
        runtime_count: int = 1,
        repo_root: Optional[Union[str, os.PathLike[str]]] = None,
        permissions: Optional[Iterable[str]] = None,
        providers: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.repo_root = (
            Path(repo_root).expanduser().resolve()
            if repo_root is not None
            else _find_repo_root(self.workspace)
        )
        self.runtime_count = max(1, int(runtime_count))
        self.controller = controller
        if engine is None and controller is None:
            from hcli.controller import Controller

            controller = Controller(
                self.workspace,
                runtime_count=self.runtime_count,
                model=model,
            )
            self.controller = controller
        self.engine = engine or getattr(self.controller, "engine", None)
        self.role_router = RoleRouter()
        self.providers: Dict[str, Any] = dict(providers or {})
        if self.engine is not None:
            self.providers.setdefault("resident", self.engine)
            self.providers.setdefault("local", self.engine)
        self.tools: ToolRegistry = default_tool_registry(
            self.workspace,
            repo_root=self.repo_root,
            mission_root=self.workspace / ".hcli" / "mission",
            permissions=permissions,
        )
        self.background = BackgroundJobStore(
            self.workspace,
            allowed_roots=(self.repo_root,),
        )
        self.mission: Optional[Mission] = None
        self.last_result: Optional[Dict[str, Any]] = None

    @property
    def state_path(self) -> Path:
        return mission_state_path(self.workspace)

    @property
    def checkpoint_path(self) -> Path:
        return self.workspace / ".hcli" / "agentos" / "checkpoint.json"

    def _next_action(self, mission: Optional[Mission] = None) -> str:
        active = mission or self.mission
        if active is None:
            return "start a mission with AgentOS.start_mission(goal)"
        units = list(getattr(getattr(active, "scheduler", None), "units", {}).values())
        if any(getattr(unit, "status", "") == "running" for unit in units):
            return "recover running work and continue the mission"
        if any(getattr(unit, "status", "") in {"pending", "ready", "failed"} for unit in units):
            return "dispatch the next dependency-ready work unit; failed units may produce bounded repairs"
        if getattr(active, "phase", "") == "completed":
            return "inspect the result envelope and receipts; start the next mission if more work is required"
        return "reconstruct Mission.from_workspace and continue"

    def _persist_control_checkpoint(self, *, event: str, next_action: Optional[str] = None) -> Path:
        mission = self.mission
        mission_status = mission.status() if mission is not None else None
        units = {}
        if mission is not None:
            units = {
                uid: _json_safe(unit.to_dict())
                for uid, unit in getattr(mission.scheduler, "units", {}).items()
            }
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "event": event,
            "updated_at": time.time(),
            "workspace": str(self.workspace),
            "mission": {
                "id": getattr(mission, "id", None),
                "goal": getattr(mission, "goal", None),
                "phase": getattr(mission, "phase", None),
                "state": (mission_status or {}).get("state") if isinstance(mission_status, dict) else None,
                "status": _json_safe(mission_status),
                "units": units,
                "compiled": _json_safe(getattr(mission, "_compiled", None)),
            },
            "providers": _json_safe(self.provider_profiles()),
            "next_action": next_action or self._next_action(mission),
            "continuation": {
                "reconstruct": "AgentOS(workspace).recover_mission()",
                "continue": "AgentOS(workspace).continue_mission()",
                "disk_is_authority": True,
            },
            "background": self.background.list(),
        }
        atomic_write_json(self.checkpoint_path, payload)
        return self.checkpoint_path

    def compile_goal(self, goal: str) -> Dict[str, Any]:
        """Compile a goal without launching a model or mutating the workspace."""
        return GoalCompiler().compile(str(goal or ""))

    def start_mission(
        self,
        goal: str,
        *,
        units: Any = None,
        runtime_count: Optional[int] = None,
        providers: Optional[Mapping[str, Any]] = None,
    ) -> Mission:
        if self.engine is None:
            raise RuntimeError("AgentOS requires an Engine or Controller")
        count = self.runtime_count if runtime_count is None else max(1, int(runtime_count))
        pool = getattr(self.controller, "runtime_pool", None)
        selected_providers = dict(self.providers)
        selected_providers.update(dict(providers or {}))
        self.mission = Mission(
            self.workspace,
            engine=self.engine,
            units=units,
            goal=str(goal or ""),
            runtime_count=count,
            runtime_pool=pool,
            repo_root=self.repo_root,
            providers=selected_providers,
            stop_runtime_pool=False,
        )
        self.mission.checkpoint()
        self._persist_control_checkpoint(event="mission_started")
        return self.mission

    def recover_mission(self) -> Mission:
        """Reconstruct the persisted mission and mark orphaned local work safely."""
        if self.engine is None:
            raise RuntimeError("AgentOS requires an Engine or Controller")
        self.mission = Mission.from_workspace(
            self.workspace,
            engine=self.engine,
            runtime_pool=getattr(self.controller, "runtime_pool", None),
            runtime_count=self.runtime_count,
            repo_root=self.repo_root,
            providers=self.providers,
            stop_runtime_pool=False,
        )
        self._persist_control_checkpoint(event="mission_recovered")
        return self.mission

    def run(self, goal: Optional[str] = None) -> Dict[str, Any]:
        if goal is not None:
            mission = self.start_mission(goal)
        elif self.mission is not None:
            mission = self.mission
        elif self.state_path.is_file():
            mission = self.recover_mission()
        else:
            raise RuntimeError("no mission is loaded; provide a goal first")
        result = mission.run()
        self.last_result = result
        self._persist_control_checkpoint(event="mission_finished")
        return result

    def continue_mission(self) -> Dict[str, Any]:
        """Reconstruct and run the durable mission without human re-planning."""
        if self.mission is None:
            if not self.state_path.is_file():
                raise RuntimeError("no durable mission is available")
            self.recover_mission()
        return self.run()

    def register_provider(self, name: str, provider: Any) -> None:
        """Attach a provider under a stable policy name for future work units."""
        key = str(name or "").strip().lower()
        if not key or any(char.isspace() for char in key):
            raise ValueError("provider name must be a non-empty token")
        self.providers[key] = provider

    def route_role(self, role: str) -> Dict[str, Any]:
        """Return the first provider satisfying a role's capability policy."""
        values = dict(self.providers)
        pool = getattr(self.controller, "runtime_pool", None)
        for runtime in getattr(pool, "runtimes", []) or []:
            backend = getattr(runtime, "backend", None)
            if backend is None:
                continue
            identity = getattr(backend, "identity", lambda: {})()
            provider = str(identity.get("provider") or identity.get("runtime") or type(backend).__name__).lower()
            values.setdefault(provider, backend)
        return self.role_router.choose(role, values)

    def checkpoint(self) -> Optional[Path]:
        if self.mission is None:
            return None
        path = self.mission.checkpoint()
        self._persist_control_checkpoint(event="checkpoint")
        return path

    def start_background(
        self,
        argv: Iterable[str],
        *,
        cwd: Optional[Union[str, os.PathLike[str]]] = None,
        label: Optional[str] = None,
        resumable: bool = True,
        env: Optional[Mapping[str, str]] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Start shell-free background work and persist its restart record."""
        job = self.background.start(
            list(argv),
            cwd=cwd,
            label=label,
            resumable=resumable,
            env=env,
            timeout_s=timeout_s,
        )
        self._persist_control_checkpoint(event=f"background_started:{job['job_id']}")
        return job

    def background_status(self, job_id: str) -> Dict[str, Any]:
        return self.background.inspect(job_id)

    def background_jobs(self) -> list[Dict[str, Any]]:
        return self.background.list()

    def resume_background(self, job_id: str) -> Dict[str, Any]:
        job = self.background.resume(job_id)
        self._persist_control_checkpoint(event=f"background_resumed:{job['job_id']}")
        return job

    def cancel_background(self, job_id: str) -> Dict[str, Any]:
        job = self.background.cancel(job_id)
        self._persist_control_checkpoint(event=f"background_cancelled:{job_id}")
        return job

    def invoke_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> ToolResult:
        result = self.tools.invoke(name, arguments)
        # Tool results are durable evidence.  This write is AgentOS-owned and
        # stays under the mission's .hcli receipt directory; the tool itself
        # still cannot acquire a write permission by accident.
        destination = self.workspace / ".hcli" / "receipts" / "tools" / f"{result.invocation_id}.json"
        result.provenance["receipt_path"] = str(destination)
        try:
            atomic_write_json(destination, result.to_dict())
        except OSError:
            # A tool remains useful when receipt storage is unavailable; the
            # missing path is visible in its result rather than hidden.
            result.provenance.pop("receipt_path", None)
            result.provenance["receipt_error"] = str(destination)
        self._persist_control_checkpoint(event=f"tool:{result.tool}")
        return result

    def provider_profiles(self) -> Dict[str, Dict[str, Any]]:
        profiles: Dict[str, Dict[str, Any]] = {}
        pool = getattr(self.controller, "runtime_pool", None)
        for runtime in getattr(pool, "runtimes", []) or []:
            backend = getattr(runtime, "backend", None)
            if backend is None:
                continue
            profile = profile_from_backend(backend)
            profiles[profile.profile_id] = profile.to_dict()
        # Include explicitly registered specialist/frontier/vision providers
        # even when the current local pool is healthy.  A profile census that
        # drops non-primary providers cannot reproduce role routing after a
        # restart.
        for name, provider in self.providers.items():
            if provider is None:
                continue
            try:
                profile_fn = getattr(provider, "profile", None)
                profile = profile_fn() if callable(profile_fn) else None
                if isinstance(profile, ResidentProfile):
                    profiles[profile.profile_id] = profile.to_dict()
                    continue
                if isinstance(provider, ResidentProfile):
                    profiles[provider.profile_id] = provider.to_dict()
                    continue
                if isinstance(provider, Mapping):
                    candidate = provider.get("profile") or provider
                    if isinstance(candidate, Mapping):
                        parsed = ResidentProfile.from_mapping(candidate)
                        profiles[parsed.profile_id] = parsed.to_dict()
                        continue
                identity_fn = getattr(provider, "identity", None)
                identity = identity_fn() if callable(identity_fn) else {}
                if isinstance(identity, Mapping):
                    parsed = ResidentProfile.from_backend(provider, profile_id=str(name))
                    profiles[parsed.profile_id] = parsed.to_dict()
            except Exception:
                # A provider can be unavailable at census time. Keep a stable
                # placeholder rather than silently omitting its policy name.
                profiles.setdefault(str(name), {
                    "schema": "hcli.provider.profile.v1",
                    "profile_id": str(name),
                    "provider": str(name),
                    "model_id": "unknown",
                    "qualification": {"status": "UNOBSERVED"},
                })
        if profiles:
            return profiles
        model = getattr(self.controller, "model", None) if self.controller is not None else None
        if not model:
            return {}
        semantics = model_semantics_for(model)
        profile = ResidentProfile(
            profile_id=semantics.identity,
            provider=semantics.backend_kind,
            model_id=semantics.identity,
            artifact={"path": semantics.path, "bytes": semantics.bytes, "present": semantics.present},
            runtime={"backend_kind": semantics.backend_kind},
            limits={"context_length": semantics.context_length},
            qualification={"status": "not_spawned"},
        )
        return {profile.profile_id: profile.to_dict()}

    def recovery_status(self) -> Dict[str, Any]:
        if not self.state_path.is_file():
            return {"state_path": str(self.state_path), "present": False}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return {"state_path": str(self.state_path), "present": True, "valid": False, "error": str(exc)}
        units = data.get("units") if isinstance(data, dict) else {}
        running = [uid for uid, item in (units or {}).items() if isinstance(item, dict) and item.get("status") == "running"]
        pending = [uid for uid, item in (units or {}).items() if isinstance(item, dict) and item.get("status") in {"pending", "ready", "failed"}]
        from .states import workunit_state

        state_by_unit = {
            str(uid): workunit_state(type("RecoveredUnit", (), item)()).value
            for uid, item in (units or {}).items()
            if isinstance(item, dict)
        }
        checkpoint = None
        if self.checkpoint_path.is_file():
            try:
                checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                checkpoint = None
        return {
            "state_path": str(self.state_path),
            "present": True,
            "valid": isinstance(data, dict),
            "mission_id": data.get("id") if isinstance(data, dict) else None,
            "phase": data.get("phase") if isinstance(data, dict) else None,
            "checkpoint_id": data.get("checkpoint_id") if isinstance(data, dict) else None,
            "running_units": running,
            "resumable_units": pending,
            "state": (checkpoint or {}).get("mission", {}).get("state") if isinstance(checkpoint, dict) else None,
            "unit_states": state_by_unit,
            "restart_policy": "reconstruct Mission.from_workspace; rerun non-adoptable in-process units",
            "control_checkpoint": str(self.checkpoint_path),
            "next_action": self._next_action(self.mission),
        }

    def _mission_validation(self) -> Optional[Dict[str, Any]]:
        """Aggregate persisted WorkUnit verification for the result boundary.

        Mission completion is a lifecycle fact, not proof.  Only WorkUnit
        verification records can promote the aggregate envelope to ACCEPT;
        incomplete or legacy state remains UNVERIFIED.
        """
        mission = self.mission
        scheduler = getattr(mission, "scheduler", None) if mission is not None else None
        units = list(getattr(scheduler, "units", {}).values()) if scheduler is not None else []
        if not units:
            return None
        if any(getattr(unit, "status", None) == "failed" for unit in units):
            return {
                "ok": False,
                "reason": "one or more WorkUnits failed",
                "checks": [
                    {
                        "unit_id": str(getattr(unit, "id", "")),
                        "status": getattr(unit, "status", None),
                        "verification": getattr(unit, "verification", None),
                    }
                    for unit in units
                ],
            }
        if any(getattr(unit, "status", None) != "completed" for unit in units):
            return None
        validations = []
        for unit in units:
            validation = getattr(unit, "verification", None)
            if not isinstance(validation, dict):
                return None
            validations.append((unit, validation))
        sources = {
            str(validation.get("acceptance_source") or "unknown")
            for _unit, validation in validations
        }
        source = next(iter(sources)) if len(sources) == 1 else "mixed"
        checks = [
            {
                "unit_id": str(getattr(unit, "id", "")),
                "ok": validation.get("ok") is True,
                "acceptance_source": validation.get("acceptance_source"),
            }
            for unit, validation in validations
        ]
        return {
            "ok": all(validation.get("ok") is True for _unit, validation in validations),
            "checks": checks,
            "acceptance_source": source,
        }

    def result_envelope(self, result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = result or self.last_result or {}
        receipt = body.get("receipt") if isinstance(body, dict) else None
        validation = body.get("validation") if isinstance(body, dict) else None
        if validation is None:
            validation = self._mission_validation()
        return build_result_envelope(
            goal=getattr(self.mission, "goal", "") if self.mission is not None else "",
            result=body,
            validation=validation,
            runtime_provenance=(getattr(self.engine, "_runtime_provenance", lambda: [])() if self.engine is not None else []),
            model_calls=getattr(self.engine, "_model_calls", []) if self.engine is not None else [],
            receipt_path=receipt,
        )

    def status(self) -> Dict[str, Any]:
        mission_status = self.mission.status() if self.mission is not None else None
        return {
            "schema": "hcli.agentos.status.v1",
            "workspace": str(self.workspace),
            "mission": mission_status,
            "recovery": self.recovery_status(),
            "providers": self.provider_profiles(),
            "roles": self.role_router.to_dict(),
            "tools": self.tools.discover(),
            "background": self.background.list(),
            "checkpoint_path": str(self.checkpoint_path),
            "next_action": self._next_action(self.mission),
            "generated_at": time.time(),
        }


__all__ = ["AgentOS"]
