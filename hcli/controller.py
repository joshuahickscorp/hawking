from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Config
from .engine import Engine
from .events import EventBus
from .max_policy import grok_pool_snapshot
from .mission import Mission
from .models import ModelInfo, ModelRegistry
from .resources import MutationLock, can_admit, normalize_resource_class, occupancy_of
from .runtime import RuntimePool, load_observed_overlap
from .session import Session, SessionStore
from .steering import SteeringQueue
from .workunit import is_ready
from .workspace import Workspace


def _default_native_profile() -> Optional[str]:
    """Return the shipped native profile only when its physical paths exist.

    An explicit ``--model``/config/env selection always wins.  This small
    default makes the current resident the local cognition provider on this
    machine while keeping a checkout or installation without the resident
    fully usable for explicit MLX, GGUF, or remote selections.
    """
    candidate = Path(__file__).resolve().with_name("hawking-native.sealed-3.14.json")
    if not candidate.is_file():
        return None
    try:
        from .hawking_native import HawkingNativeConfig

        profile = HawkingNativeConfig.from_file(str(candidate))
        profile.validate()
    except Exception:
        return None
    return str(candidate)


def _http_json(url: str, timeout: float = 0.4) -> Any:
    """GET ``url`` and parse JSON. None on any failure. Never raises."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError):
        return None
    except Exception:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, AttributeError):
        return None


class Controller:
    """HCLI product controller and owner of the native Engine."""

    def __init__(
        self,
        workspace: Any,
        runtime_count: int = 1,
        model: Optional[str] = None,
        bus: Any = None,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        if isinstance(
            workspace,
            Workspace,
        ):
            self.workspace = workspace
        else:
            self.workspace = Workspace(
                os.fspath(workspace)
            )

        self.workspace_root = os.path.realpath(
            os.fspath(
                self.workspace.root
            )
        )

        self.runtime_count = int(
            runtime_count
        )

        if self.runtime_count < 1:
            raise ValueError(
                "runtime_count must be >= 1"
            )

        self.model = model

        self.bus = (
            bus
            if bus is not None
            else EventBus()
        )

        self.registry = (
            registry
            if registry is not None
            else ModelRegistry()
        )

        self.model_info: Optional[
            ModelInfo
        ] = None

        self.runtime_pool: Optional[
            RuntimePool
        ] = None

        self.session = Session(
            runtime_count=self.runtime_count,
            model=model,
        )
        self.session_store = SessionStore(
            self.workspace_root
        )
        self.mission: Optional[Mission] = None
        self._exit_requested = False
        self._command_handler = None
        self._ledger = None

        self._shutdown = False

        migration_env = (
            os.environ.get(
                "HCLI_MODEL_PATH"
            )
            or os.environ.get(
                "HAIDER_MODEL_PATH"
            )
            or os.environ.get(
                "HCLI_HAWKING_NATIVE_CONFIG"
            )
        )

        self.config = Config(
            self.workspace_root
        )

        project_model = self.config.layer_model(self.config.project_path)
        global_model = self.config.layer_model(self.config.global_path)
        default_native = None
        if not model and not project_model and not global_model and not migration_env:
            default_native = _default_native_profile()

        self.model_info = self.registry.resolve(
            explicit=model or default_native,
            project_config=project_model,
            global_config=global_model,
            env=migration_env,
        )

        # Keep the resolved selection in session state for every provider.
        # Remote selectors have already been reduced to a credential-free URL;
        # local/native selections are normalized by ModelRegistry as well.
        if self.model_info is not None:
            self.model = self.model_info.path
            self.session.model = self.model_info.path

        # Persist only the resolved model selection.  In particular, a
        # remote selector may have arrived with a credential-bearing query;
        # ModelRegistry has already reduced it to a credential-free path.
        self._persist_session()

        engine_model = (
            self.model_info.path
            if self.model_info is not None
            else model
            or migration_env
            or "local"
        )

        self.engine = Engine(
            workspace=self.workspace,
            event_bus=self.bus,
            runtime_provider=self.ensure_runtime_pool,
            runtime_state_provider=lambda: self.runtime_pool,
            runtime_count=self.runtime_count,
            model_name=engine_model,
        )

    @property
    def model_name(
        self,
    ) -> Optional[str]:
        if self.model_info is None:
            return (
                self.model
                if self.model
                else None
            )

        return self.model_info.display_name

    def _emit(
        self,
        event_type: str,
        payload: Optional[dict] = None,
    ) -> None:
        emit = getattr(
            self.bus,
            "emit",
            None,
        )

        if not callable(emit):
            return

        try:
            emit(
                event_type,
                payload or {},
            )
        except Exception:
            pass

    def status(
        self,
    ) -> dict:
        runtimes = []

        if self.runtime_pool is not None:
            for runtime in (
                self.runtime_pool.runtimes
            ):
                runtimes.append(
                    {
                        "index": runtime.index,
                        "pid": runtime.pid,
                        "port": runtime.port,
                        "active": runtime.active,
                        "model": runtime.model,
                    }
                )

        snapshot = {
            "workspace": self.workspace_root,
            "requested_runtimes": self.runtime_count,
            "admitted_runtimes": (
                self.runtime_pool.admitted_n
                if self.runtime_pool is not None
                else 0
            ),
            "model": (
                self.model_info.path
                if self.model_info is not None
                else self.model
            ),
            "model_name": self.model_name,
            "runtimes": runtimes,
            "engine_active": self.engine.active,
            "shutdown": self._shutdown,
        }

        goal = ""
        if self.mission is not None:
            goal = getattr(self.mission, "goal", "") or ""
        if not goal:
            session = getattr(self, "session", None)
            goal = getattr(session, "goal", "") or ""
        snapshot["goal"] = goal

        if self.mission is not None:
            mission_status = self.mission.status()
            snapshot["mission"] = mission_status
            for key in (
                "mission_id",
                "phase",
                "units_by_status",
                "active_runtimes",
                "active_decodes",
                "accepted_units_per_hour",
                "elapsed_wall",
                "last_checkpoint",
                "no_progress_warning",
            ):
                snapshot[key] = mission_status.get(key)
            snapshot["blocked_reason"] = getattr(
                self.mission, "_stop_reason", None
            )
        else:
            snapshot["blocked_reason"] = None

        units_map, units = self._scheduler_units()
        snapshot["occupancy"] = dict(occupancy_of(units))
        snapshot["blocked_units"] = sum(
            1
            for unit in units
            if unit.status in ("pending", "failed")
            and not is_ready(unit, units_map)
        )
        runtime_status = self._runtime_status()
        snapshot["runtime"] = runtime_status
        snapshot["provider"] = runtime_status.get("provider")
        # Keep the old key for clients written before provider-neutral status;
        # the value is now the generic runtime observation rather than a claim
        # that every local model is Qwen.
        snapshot["qwen"] = runtime_status
        snapshot["grok"] = grok_pool_snapshot(
            self.workspace_root, mission=self.mission
        )
        snapshot["mutation"] = self._mutation_status()
        if snapshot["mutation"].get("held"):
            snapshot["mutation"]["waiters"] = sum(
                1
                for unit in units
                if unit.status == "ready"
                and normalize_resource_class(unit.resource_class) == "MUTATION"
            )
        else:
            snapshot["mutation"]["waiters"] = 0

        last_checkpoint = snapshot.get("last_checkpoint") or 0.0
        try:
            last_checkpoint = float(last_checkpoint)
        except (TypeError, ValueError):
            last_checkpoint = 0.0
        snapshot["checkpoint_age_s"] = (
            (time.time() - last_checkpoint) if last_checkpoint else None
        )

        ledger = self._status_ledger()
        if ledger is None:
            snapshot["verifier_backlog"] = None
            snapshot["watchdog_tier"] = "unknown"
            snapshot["watchdog"] = "unknown"
            snapshot["watchdog_message"] = snapshot.get("no_progress_warning") or ""
        else:
            try:
                unverified = ledger.unverified()
                snapshot["verifier_backlog"] = len(unverified)
            except Exception:
                snapshot["verifier_backlog"] = None
            watchdog = getattr(ledger, "status", None)
            if callable(watchdog):
                try:
                    watchdog = watchdog()
                except Exception:
                    watchdog = None
            snapshot["watchdog"] = watchdog if watchdog else "unknown"
            snapshot["watchdog_tier"] = "unknown"
            message = snapshot.get("no_progress_warning") or ""
            count_fn = getattr(ledger, "consecutive_no_progress_count", None)
            if callable(count_fn):
                try:
                    count = int(count_fn() or 0)
                except (TypeError, ValueError):
                    count = 0
                if count and not message:
                    message = f"no_progress x{count}"
            snapshot["watchdog_message"] = message

        snapshot["goal"] = self.session.goal
        snapshot["session_id"] = self.session.id
        snapshot["messages"] = len(self.session.messages)
        snapshot.setdefault("grok", self._grok_pool_status())
        # NOT an unconditional reassignment: the block above already computed
        # mutation status AND its real waiter count. Overwriting it here reset
        # waiters to the hardcoded 0 in _mutation_status, so the scheduler was
        # counting a queue the operator could never see.
        snapshot.setdefault("mutation", self._mutation_status())
        snapshot.setdefault("qwen", self._qwen_pool_status())
        snapshot.setdefault("max", self._max_status())
        return snapshot

    @property
    def mutation_lock(self):
        if self.mission is not None:
            scheduler = getattr(self.mission, "scheduler", None)
            lock = getattr(scheduler, "mutation_lock", None) if scheduler is not None else None
            if lock is not None:
                return lock
        return getattr(self, "_mutation_lock", None)

    def dispatcher(self):
        """Canonical slash-command dispatcher. One owner for TUI/CLI/tests."""
        from .commands import CommandHandler

        handler = self._command_handler
        if handler is None:
            handler = CommandHandler(self)
            self._command_handler = handler
        return handler

    def _grok_pool_status(self) -> dict:
        # Use the module-level name, not a function-local import: a local
        # import re-resolves from max_policy on every call, so the snapshot
        # cannot be substituted and the truthfulness of this line cannot be
        # tested.
        try:
            return grok_pool_snapshot(self.workspace_root, self.mission)
        except Exception as exc:
            # A failure to observe is not an observation of zero. Reporting
            # active=0 here is exactly the lie this whole surface is being
            # repaired to stop telling.
            return {
                "admitted": None,
                "active": None,
                "queued": None,
                "done": None,
                "failed": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _max_status(self) -> dict:
        try:
            from .max_policy import load_equilibrium

            return load_equilibrium(self.workspace_root)
        except Exception:
            return {}

    def _scheduler_units(self):
        sched = getattr(self.mission, "scheduler", None) if self.mission is not None else None
        units_map = dict(getattr(sched, "units", None) or {})
        return units_map, list(units_map.values())

    def _status_ledger(self):
        for obj in (self, self.mission):
            if obj is None:
                continue
            for name in ("_ledger", "ledger"):
                ledger = getattr(obj, name, None)
                if ledger is not None and hasattr(ledger, "unverified"):
                    return ledger
        return None

    def _qwen_endpoint(self) -> str:
        pool = self.runtime_pool
        if pool is not None:
            for runtime in getattr(pool, "runtimes", None) or []:
                port = getattr(runtime, "port", None)
                if port:
                    return f"http://127.0.0.1:{int(port)}"
                backend = getattr(runtime, "backend", None)
                endpoint = getattr(backend, "endpoint", None)
                if callable(endpoint):
                    try:
                        value = endpoint()
                    except Exception:
                        value = None
                    if value:
                        return str(value)
        return "http://127.0.0.1:8080"

    def _gpu_decode_ready_not_admitted(self) -> int:
        units_map, units = self._scheduler_units()
        if not units:
            return 0
        sched = getattr(self.mission, "scheduler", None)
        limits = getattr(sched, "limits", None)
        if limits is None:
            return 0
        occupied = occupancy_of(units)
        queued = 0
        for unit in units:
            if normalize_resource_class(unit.resource_class) != "GPU_DECODE":
                continue
            waiting = unit.status == "ready" or is_ready(unit, units_map)
            if waiting and not can_admit("GPU_DECODE", occupied, limits):
                queued += 1
        return queued

    def _last_qwen_tps(self) -> Optional[float]:
        calls = getattr(self.engine, "_model_calls", None) or []
        if not calls:
            return None
        last = calls[-1]
        if not isinstance(last, dict):
            return None
        try:
            toks = float(last.get("completion_tokens") or 0)
            wall = float(last.get("wall_s") or 0)
        except (TypeError, ValueError):
            return None
        if toks > 0 and wall > 0:
            return toks / wall
        return None

    def _qwen_pool_status(self) -> dict:
        pool = self.runtime_pool
        runtimes = list(getattr(pool, "runtimes", []) or []) if pool is not None else []
        if runtimes:
            backend = getattr(runtimes[0], "backend", None)
            endpoint_fn = getattr(backend, "endpoint", None)
            endpoint = None
            if callable(endpoint_fn):
                try:
                    endpoint = str(endpoint_fn())
                except Exception:
                    endpoint = None
            if endpoint and not endpoint.startswith(("http://", "https://")):
                health_fn = getattr(backend, "health_snapshot", None)
                details = {}
                if callable(health_fn):
                    try:
                        details = health_fn()
                    except Exception as exc:
                        details = {"error": f"{type(exc).__name__}: {exc}"}
                identity = {}
                try:
                    identity = backend.identity()
                except Exception:
                    identity = {}
                ready = bool(details.get("ready")) if isinstance(details, dict) else False
                return {
                    "resident": len(runtimes),
                    "health": "ok" if ready else "down",
                    "active_decode": int(getattr(pool, "_in_flight", 0) or 0),
                    "queued": self._gpu_decode_ready_not_admitted(),
                    "n_ctx": identity.get("context"),
                    "prompt_tokens": None,
                    "tps": self._last_qwen_tps(),
                    "backend": identity.get("backend"),
                    "endpoint": endpoint,
                    "details": details,
                }
        base = self._qwen_endpoint().rstrip("/")
        health = _http_json(f"{base}/health", timeout=0.4)
        slots = _http_json(f"{base}/slots", timeout=0.4)
        if not isinstance(slots, list):
            return {
                "resident": 0,
                "health": "down",
                "active_decode": 0,
                "queued": 0,
                "n_ctx": None,
                "prompt_tokens": None,
                "tps": None,
            }
        processing = [
            slot for slot in slots if isinstance(slot, dict) and slot.get("is_processing")
        ]
        prompt = 0
        if slots:
            prompt = max(
                int(slot.get("n_prompt_tokens") or 0)
                for slot in slots
                if isinstance(slot, dict)
            )
        n_ctx = None
        first = slots[0] if slots else None
        if isinstance(first, dict) and first.get("n_ctx") is not None:
            try:
                n_ctx = int(first.get("n_ctx"))
            except (TypeError, ValueError):
                n_ctx = None
        health_ok = isinstance(health, dict) and health.get("status") == "ok"
        return {
            "resident": len(slots),
            "health": "ok" if health_ok else "down",
            "active_decode": len(processing),
            "queued": self._gpu_decode_ready_not_admitted(),
            "n_ctx": n_ctx,
            "prompt_tokens": prompt,
            "tps": self._last_qwen_tps(),
        }

    def _runtime_status(self) -> dict:
        """Provider-neutral runtime status with a compatibility alias."""
        pool = self.runtime_pool
        runtimes = list(getattr(pool, "runtimes", []) or []) if pool is not None else []
        if not runtimes:
            status = self._qwen_pool_status()
            selected_provider = getattr(self.model_info, "provider", None)
            status["provider"] = selected_provider or status.get("provider") or "local"
            if self.model_info is not None:
                status["model"] = self.model_info.path
                if status.get("endpoint") is None and status["provider"] == "remote":
                    try:
                        from .backends import OpenAICompatibleBackend

                        status["endpoint"] = OpenAICompatibleBackend(
                            self.model_info.path
                        ).endpoint()
                    except Exception:
                        pass
            return status
        backend = getattr(runtimes[0], "backend", None)
        identity = {}
        try:
            identity = backend.identity() if backend is not None else {}
        except Exception:
            identity = {}
        provider = str(
            identity.get("provider")
            or identity.get("runtime")
            or identity.get("backend")
            or type(backend).__name__
        )
        endpoint = None
        endpoint_fn = getattr(backend, "endpoint", None)
        if callable(endpoint_fn):
            try:
                endpoint = str(endpoint_fn())
            except Exception:
                endpoint = None
        if provider == "remote" or identity.get("backend") == "openai_compatible":
            active = sum(1 for runtime in runtimes if getattr(runtime, "active", False))
            return {
                "provider": "remote",
                "backend": identity.get("backend"),
                "health": "ok" if active else "down",
                "resident": active,
                "active_decode": int(getattr(pool, "_in_flight", 0) or 0),
                "queued": self._gpu_decode_ready_not_admitted(),
                "n_ctx": identity.get("context"),
                "prompt_tokens": None,
                "tps": self._last_qwen_tps(),
                "endpoint": endpoint,
            }
        status = self._qwen_pool_status()
        status["provider"] = provider
        status["backend"] = identity.get("backend") or status.get("backend")
        status["endpoint"] = endpoint or status.get("endpoint")
        return status

    def _mutation_lock(self):
        mission = self.mission
        sched = getattr(mission, "scheduler", None) if mission is not None else None
        lock = getattr(sched, "mutation_lock", None) if sched is not None else None
        if lock is not None:
            return lock
        return MutationLock(self.workspace_root)

    def _mutation_status(self) -> Dict[str, Any]:
        lock = self._mutation_lock()
        rec = None
        try:
            rec = lock.read() if lock is not None else None
        except Exception:
            rec = None
        held = False
        if lock is not None and rec:
            try:
                held = bool(lock.holder_is_live(rec))
            except Exception:
                held = False
        owner = rec.get("unit_id") if isinstance(rec, dict) else None
        pid = rec.get("pid") if isinstance(rec, dict) else None
        owner_display = owner
        if owner and not held:
            owner_display = "stale"
        return {
            "held": held,
            "pid": pid,
            "owner": owner,
            "owner_display": owner_display,
            "waiters": 0,
        }

    def list_models(
        self,
    ) -> list:
        return self.registry.discover()

    def select_model(
        self,
        selector: str,
    ) -> bool:
        selected = self.registry.select(
            selector
        )

        if selected is None:
            self._emit(
                "warning",
                {
                    "message": (
                        f"Model not found: "
                        f"{selector}"
                    )
                },
            )

            return False

        if self.runtime_pool is not None:
            self.runtime_pool.stop()
            self.runtime_pool = None

        self.model_info = selected
        self.model = selected.path
        self.session.model = selected.path
        self._persist_session()
        self.engine.model_name = (
            selected.path
        )

        try:
            self.config.save_global(
                {
                    "model": selected.path,
                }
            )
        except OSError:
            pass

        self._emit(
            "model_switching",
            {
                "model": selected.path,
                "display_name": (
                    selected.display_name
                ),
            },
        )

        return True

    def ensure_runtime_pool(
        self,
    ) -> RuntimePool:
        if self._shutdown:
            raise RuntimeError(
                "Controller is shut down"
            )

        model_path = (
            self.model_info.path
            if self.model_info is not None
            else os.environ.get(
                "HCLI_MODEL_PATH"
            )
            or os.environ.get(
                "HAIDER_MODEL_PATH"
            )
            or os.environ.get(
                "HCLI_HAWKING_NATIVE_CONFIG"
            )
        )

        if not model_path:
            raise RuntimeError(
                "No model selected. "
                "Use /models or /model <number|path>."
            )

        if self.runtime_pool is None:
            pool = RuntimePool(
                model_path,
                requested_n=self.runtime_count,
                workspace=self.workspace_root,
                repo_root=self.workspace_root,
                observed_overlap=load_observed_overlap(self.workspace_root),
            )

            self._emit(
                "runtime_loading",
                {
                    "requested": (
                        self.runtime_count
                    ),
                    "model": model_path,
                },
            )

            try:
                pool.start()
            except Exception:
                pool.stop()
                raise

            self.runtime_pool = pool
            self.engine.model_name = model_path

            self._emit(
                "runtime_ready",
                {
                    "requested": (
                        self.runtime_count
                    ),
                    "admitted": (
                        pool.admitted_n
                    ),
                    "runtimes": [
                        {
                            "index": (
                                runtime.index
                            ),
                            "pid": runtime.pid,
                            "port": runtime.port,
                        }
                        for runtime
                        in pool.runtimes
                    ],
                },
            )

        # A Mission can finish while leaving the Controller-owned pool in its
        # field for status/provenance.  Reconcile that pool before every
        # caller receives it: an empty or dead pool is restartable state, not
        # a reason to let the next model request fall through to a verifier.
        try:
            self.runtime_pool.start()
        except Exception:
            raise
        return self.runtime_pool

    def execute(
        self,
        prompt: str,
    ) -> dict:
        prompt = (
            prompt
            or ""
        ).strip()

        if not prompt:
            return {
                "kind": "answer",
                "content": "",
                "operations": [],
                "tests": [],
                "status": "completed",
            }

        return self.engine.execute(
            prompt
        )

    def complete_text(
        self,
        prompt: str,
    ) -> str:
        """Expose provider text for the plain-cognition qualification lane."""
        return self.engine.complete_text(prompt)

    def set_goal(
        self,
        goal: str,
    ) -> None:
        self.session.goal = goal or ""
        self._persist_session()

    def queue_steer(
        self,
        text: str,
        kind: str = "knowledge",
    ) -> Any:
        body = text or ""
        token = (kind or "knowledge").strip().lower()
        lower = body.lstrip().lower()
        for name in ("constraint", "correction", "knowledge"):
            if lower.startswith(name + ":") or lower.startswith(name + " "):
                token = name
                body = body.lstrip()[len(name) :].lstrip(": ").strip() or body
                break
        if token not in ("knowledge", "correction", "constraint"):
            token = "knowledge"
        if self.mission is not None:
            event = self.mission.append_steer(body, kind=token)
            ledger = self._ledger
            if token == "constraint" and ledger is not None:
                try:
                    queue = getattr(self.mission, "_steering", None)
                    if queue is not None:
                        queue.apply_constraint(event, ledger)
                    else:
                        ledger.apply_constraint(event)
                except Exception:
                    pass
        else:
            queue = SteeringQueue(
                self.workspace_root,
                self.session.id,
            )
            event = queue.enqueue(body, kind=token)
        self.session.steering.append(body)
        self._persist_session()
        return event

    def run_mission(
        self,
        goal: Optional[str] = None,
        units: Any = None,
        engine: Any = None,
        **kwargs: Any,
    ) -> dict:
        text = (
            goal
            if goal is not None
            else self.session.goal
        )
        self.mission = Mission(
            self.workspace_root,
            engine=engine if engine is not None else self.engine,
            units=units,
            goal=text or "",
            runtime_count=self.runtime_count,
            runtime_pool=self.runtime_pool,
            session_id=self.session.id,
            **kwargs,
        )
        self.session.mission_id = self.mission.id
        self._persist_session()
        return self.mission.run()

    def context_summary(
        self,
    ) -> str:
        return (
            f"session {self.session.id} "
            f"messages={len(self.session.messages)}"
        )

    def compact_context(
        self,
    ) -> None:
        self.session.messages = self.session.messages[-4:]
        self._persist_session()

    def clear_transcript(
        self,
    ) -> None:
        """Clear presentation/transcript only. Durable mission state stays."""
        self.session.messages = []
        self._persist_session()

    def start_ultragoal(self, goal_text: str) -> dict:
        """Create or update the durable Goal + ledger + WorkUnit DAG.

        This is not a second Goal engine. It uses GoalCompiler IR, Ledger
        obligations, and the canonical Mission/Scheduler DAG.
        """
        text = (goal_text or "").strip()
        if not text:
            raise ValueError("ultragoal text is required")
        from .goal import GoalCompiler
        from .ledger import Ledger
        from .mission import Mission, mission_state_path

        self.set_goal(text)
        compiled = GoalCompiler().compile(text)
        goal_md = Path(self.workspace_root) / ".hcli" / "GOAL.md"
        ledger: Optional[Ledger] = None
        if goal_md.is_file() and self.mission is not None:
            try:
                ledger = Ledger.parse(goal_md)
            except Exception:
                ledger = None
        if ledger is None:
            ledger = Ledger()
            ledger._preamble = f"# Ultragoal\n\n{text}\n\n"
            compiled_obs = list(compiled.get("obligations") or [])
            if compiled_obs:
                for ob in compiled_obs:
                    ob_text = str(ob.get("text") or "").strip()
                    if not ob_text:
                        continue
                    acceptance = str(ob.get("acceptance") or "").strip()
                    if not acceptance or acceptance == ob_text:
                        acceptance = (
                            "an independent check of this claim can fail; "
                            "restating the obligation is not acceptance"
                        )
                    verify = str(ob.get("verify") or "").strip()
                    if re.search(r"SystemExit\(\s*0\s*\)", verify) or verify in {
                        "true",
                        "exit 0",
                        ":",
                    }:
                        verify = ""
                    add_kwargs = {
                        "acceptance": acceptance,
                        "verify_command": verify,
                        "tier": "V1",
                        "risk": "medium",
                    }
                    oid = str(ob.get("id") or "").strip()
                    if oid:
                        add_kwargs["obligation_id"] = oid
                    ledger.add(ob_text, **add_kwargs)
            else:
                criteria = list(compiled.get("acceptance_criteria") or [])
                if not criteria:
                    criteria = [
                        "an independent check of the requested behavior "
                        "can fail; restating the request is not acceptance"
                    ]
                for sentence in criteria:
                    ledger.add(
                        sentence,
                        acceptance=(
                            "an independent check of this claim can fail; "
                            "restating the obligation is not acceptance"
                        ),
                        verify_command="",
                        tier="V1",
                        risk="medium",
                    )
        else:
            ledger._preamble = f"# Ultragoal\n\n{text}\n\n"
            ledger._dirty = True
        # save() rather than write_text(): it also records _path, and the
        # verify-receipt sidecars are resolved relative to that path.
        ledger.save(goal_md)
        self._ledger = ledger

        dag = compiled.get("workunits")
        units = dict(getattr(dag, "units", {}) or {})
        if self.mission is not None:
            self.mission.goal = text
            for uid, wu in units.items():
                if uid not in self.mission.scheduler.units:
                    self.mission.scheduler.units[uid] = wu
            self.mission.scheduler._persist()
            self.session.mission_id = self.mission.id
            self.mission.checkpoint()
        else:
            existing = mission_state_path(self.workspace_root)
            if existing.is_file():
                self.mission = Mission.from_workspace(
                    self.workspace_root,
                    engine=self.engine,
                    runtime_count=self.runtime_count,
                    runtime_pool=self.runtime_pool,
                    session_id=self.session.id,
                    quiet=True,
                )
                self.mission.goal = text
                for uid, wu in units.items():
                    if uid not in self.mission.scheduler.units:
                        self.mission.scheduler.units[uid] = wu
                self.mission.scheduler._persist()
                self.mission.checkpoint()
            else:
                self.mission = Mission(
                    self.workspace_root,
                    engine=self.engine,
                    units=units,
                    goal=text,
                    runtime_count=self.runtime_count,
                    runtime_pool=self.runtime_pool,
                    session_id=self.session.id,
                    quiet=True,
                )
                self.mission.checkpoint()
            self.session.mission_id = self.mission.id
        self._persist_session()
        return {
            "mission_id": self.mission.id,
            "goal": text,
            "ledger_path": str(goal_md),
            "obligation_ids": [ob.id for ob in ledger.obligations()],
            "workunit_ids": list(self.mission.scheduler.units.keys()),
        }

    def _persist_session(self) -> None:
        self.session_store.save(self.session)

    def start_ultragoal(self, goal_text: str) -> dict:
        """Create or update the durable Goal + ledger + WorkUnit DAG.

        This is not a second Goal engine. It uses GoalCompiler IR, Ledger
        obligations, and the canonical Mission/Scheduler DAG.
        """
        text = (goal_text or "").strip()
        if not text:
            raise ValueError("ultragoal text is required")
        from .goal import GoalCompiler
        from .ledger import Ledger
        from .mission import Mission, mission_state_path

        self.set_goal(text)
        compiled = GoalCompiler().compile(text)
        goal_md = Path(self.workspace_root) / ".hcli" / "GOAL.md"
        ledger: Optional[Ledger] = None
        if goal_md.is_file() and self.mission is not None:
            try:
                ledger = Ledger.parse(goal_md)
            except Exception:
                ledger = None
        if ledger is None:
            ledger = Ledger()
            ledger._preamble = f"# Ultragoal\n\n{text}\n\n"
            compiled_obs = list(compiled.get("obligations") or [])
            if compiled_obs:
                for ob in compiled_obs:
                    ob_text = str(ob.get("text") or "").strip()
                    if not ob_text:
                        continue
                    acceptance = str(ob.get("acceptance") or "").strip()
                    if not acceptance or acceptance == ob_text:
                        acceptance = (
                            "an independent check of this claim can fail; "
                            "restating the obligation is not acceptance"
                        )
                    verify = str(ob.get("verify") or "").strip()
                    if re.search(r"SystemExit\(\s*0\s*\)", verify) or verify in {
                        "true",
                        "exit 0",
                        ":",
                    }:
                        verify = ""
                    add_kwargs = {
                        "acceptance": acceptance,
                        "verify_command": verify,
                        "tier": "V1",
                        "risk": "medium",
                    }
                    oid = str(ob.get("id") or "").strip()
                    if oid:
                        add_kwargs["obligation_id"] = oid
                    ledger.add(ob_text, **add_kwargs)
            else:
                criteria = list(compiled.get("acceptance_criteria") or [])
                if not criteria:
                    criteria = [
                        "an independent check of the requested behavior "
                        "can fail; restating the request is not acceptance"
                    ]
                for sentence in criteria:
                    ledger.add(
                        sentence,
                        acceptance=(
                            "an independent check of this claim can fail; "
                            "restating the obligation is not acceptance"
                        ),
                        verify_command="",
                        tier="V1",
                        risk="medium",
                    )
        else:
            ledger._preamble = f"# Ultragoal\n\n{text}\n\n"
            ledger._dirty = True
        # save() rather than write_text(): it also records _path, and the
        # verify-receipt sidecars are resolved relative to that path.
        ledger.save(goal_md)
        self._ledger = ledger

        dag = compiled.get("workunits")
        units = dict(getattr(dag, "units", {}) or {})
        if self.mission is not None:
            self.mission.goal = text
            for uid, wu in units.items():
                if uid not in self.mission.scheduler.units:
                    self.mission.scheduler.units[uid] = wu
            self.mission.scheduler._persist()
            self.session.mission_id = self.mission.id
            self.mission.checkpoint()
        else:
            existing = mission_state_path(self.workspace_root)
            if existing.is_file():
                self.mission = Mission.from_workspace(
                    self.workspace_root,
                    engine=self.engine,
                    runtime_count=self.runtime_count,
                    runtime_pool=self.runtime_pool,
                    session_id=self.session.id,
                    quiet=True,
                )
                self.mission.goal = text
                for uid, wu in units.items():
                    if uid not in self.mission.scheduler.units:
                        self.mission.scheduler.units[uid] = wu
                self.mission.scheduler._persist()
                self.mission.checkpoint()
            else:
                self.mission = Mission(
                    self.workspace_root,
                    engine=self.engine,
                    units=units,
                    goal=text,
                    runtime_count=self.runtime_count,
                    runtime_pool=self.runtime_pool,
                    session_id=self.session.id,
                    quiet=True,
                )
                self.mission.checkpoint()
            self.session.mission_id = self.mission.id
        self.session_store.save(self.session)
        return {
            "mission_id": self.mission.id,
            "goal": text,
            "ledger_path": str(goal_md),
            "obligation_ids": [ob.id for ob in ledger.obligations()],
            "workunit_ids": list(self.mission.scheduler.units.keys()),
        }

    def resume_session(
        self,
        session_id: str,
    ) -> Optional[str]:
        token = (session_id or "").strip()
        loaded = None
        if token:
            loaded = self.session_store.load(token)
            if loaded is not None:
                self.session = loaded
        restored = self._restore_mission()
        if token and loaded is None and restored is None:
            return None
        if loaded is not None or restored is not None:
            self._persist_session()
        if loaded is not None:
            return loaded.id
        if restored is not None:
            return restored.id
        return None

    def _restore_mission(self) -> Optional[Mission]:
        from .mission import Mission, mission_state_path

        path = mission_state_path(self.workspace_root)
        if not path.is_file():
            return self.mission
        try:
            mission = Mission.from_workspace(
                self.workspace_root,
                engine=self.engine,
                runtime_count=self.runtime_count,
                runtime_pool=self.runtime_pool,
                session_id=self.session.id,
                quiet=True,
            )
        except Exception:
            return self.mission
        self.mission = mission
        self.session.mission_id = mission.id
        if mission.goal and not self.session.goal:
            self.session.goal = mission.goal
        goal_md = Path(self.workspace_root) / ".hcli" / "GOAL.md"
        if goal_md.is_file():
            try:
                from .ledger import Ledger

                self._ledger = Ledger.parse(goal_md)
            except Exception:
                pass
        return mission

    def request_exit(
        self,
    ) -> None:
        self._exit_requested = True
        self.shutdown()

    def cancel(
        self,
    ) -> None:
        if self.mission is not None:
            self.mission.cancel()
        self.engine.cancel()

    def handle_command(
        self,
        text: str,
    ) -> Any:
        """TUI/CLI ingress. Delegates every slash command to CommandHandler."""
        text = (text or "").strip()
        if not text:
            return None

        command, _, _rest = text.partition(" ")
        command = command.lower()
        handler = self.dispatcher()
        result = handler.handle(text)
        payload = handler.last_value

        if command == "/clear":
            self._emit(
                "transcript_cleared",
                {"kind": "transcript", "content": result or "Transcript cleared"},
            )
            return payload if payload is not None else result

        if command in ("/exit", "/quit"):
            return False

        if result is not None:
            self._emit("final_response", {"content": str(result)})
        elif command.startswith("/") and result is None and command not in (
            "/exit",
            "/quit",
        ):
            self._emit(
                "warning",
                {"message": f"Command failed: {command}"},
            )

        if payload is not None:
            return payload
        return result

    def shutdown(
        self,
    ) -> None:
        if self._shutdown:
            return

        try:
            self._persist_session()
        except Exception:
            pass

        if self.mission is not None:
            try:
                self.mission.cancel("shutdown")
            except Exception:
                pass

        if self.runtime_pool is not None:
            self.runtime_pool.stop()
            self.runtime_pool = None

        self._shutdown = True

        self._emit(
            "session_stopped",
            {},
        )
