from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Config
from .engine import Engine
from .events import EventBus
from .goal_bank import GoalBank, GoalBankError
from .knowledge import KnowledgeStore, _sanitise_text
from .max_policy import grok_pool_snapshot
from .mission import (
    TERMINAL_MISSION_PHASES,
    Mission,
    load_state,
    mission_state_path,
)
from .models import ModelInfo, ModelRegistry
from .resources import MutationLock, can_admit, normalize_resource_class, occupancy_of
from .runtime import RuntimePool, load_observed_overlap
from .session import CONTEXT_MEMORY_SCHEMA, Session, SessionStore
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


_AUTO_COMPACT_MESSAGES = 32
_MEMORY_LIST_LIMIT = 12
_MEMORY_TEXT_LIMIT = 600


def _live_mission_identity(workspace_root: Any) -> Optional[Dict[str, Any]]:
    """The running mission's id and session_id from disk, or None.

    Disk is authority. A supervising process that did not start the mission has
    no mission object, but the mission's own state file names the session whose
    steering queue it polls.
    """
    try:
        path = Path(str(workspace_root)) / ".hcli" / "mission" / "state.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        if str(data.get("phase") or "").lower() not in ("running", "paused"):
            return None
        return {"id": data.get("id"), "session_id": data.get("session_id")}
    except Exception:
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
        self.config = Config(
            self.workspace_root
        )
        self.knowledge = KnowledgeStore(
            self.workspace_root,
            archive_root=self.config.value(
                "context_archive_root",
                "HCLI_CONTEXT_ARCHIVE_ROOT",
                None,
            ),
        )
        self._knowledge_error: Optional[str] = None
        self.goal_bank = GoalBank(self.workspace_root)
        self._goal_bank_error: Optional[str] = None
        try:
            self.goal_bank.recover_inflight()
        except Exception as exc:
            # A corrupt optional queue must not make /status or a read-only
            # session unusable. Mutating queue commands report this same error
            # instead of overwriting a document we could not trust.
            self._goal_bank_error = f"{type(exc).__name__}: {exc}"
        self.mission: Optional[Mission] = None
        self._exit_requested = False
        self._command_handler = None
        self._ledger = None
        self._bank_auto_promoting = False

        self._shutdown = False

        migration_env = (
            os.environ.get(
                "HCLI_MODEL_PATH"
            )
            or os.environ.get(
                "HCLI_HAWKING_NATIVE_CONFIG"
            )
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
        self._seed_prior_knowledge()

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
        snapshot["goal_bank"] = self.goal_bank_snapshot()
        snapshot["prior_knowledge"] = self.prior_knowledge_snapshot(limit=4, max_chars=4000)
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

    def prior_knowledge_snapshot(
        self,
        *,
        limit: int = 8,
        max_chars: int = 8000,
        focus: str = "",
    ) -> Dict[str, Any]:
        """Return bounded workspace memory for a new or resumed session."""
        if self._knowledge_error:
            return {
                "schema": "hcli.workspace_knowledge.v1",
                "available": False,
                "path": str(self.knowledge.path),
                "reason": self._knowledge_error,
                "generation": 0,
                "records": [],
            }
        try:
            return self.knowledge.snapshot(
                limit=limit,
                max_chars=max_chars,
                focus=focus,
            )
        except Exception as exc:
            self._knowledge_error = f"{type(exc).__name__}: {exc}"
            return {
                "schema": "hcli.workspace_knowledge.v1",
                "available": False,
                "path": str(self.knowledge.path),
                "reason": self._knowledge_error,
                "generation": 0,
                "records": [],
            }

    def _seed_prior_knowledge(self) -> None:
        """Make the last bounded workspace facts available before turn one."""
        snapshot = self.prior_knowledge_snapshot(limit=6, max_chars=6000)
        if not snapshot.get("available") or not snapshot.get("records"):
            return
        self.session.set_memory(
            {
                "schema": CONTEXT_MEMORY_SCHEMA,
                "generation": 0,
                "prior_knowledge": snapshot,
                "retention": {
                    "source": "workspace semantic index",
                    "raw_history": "available in the gzip archive; not replayed into prompts",
                    "rule": "prior claims are context, current disk state is authority",
                },
            }
        )
        self._persist_session()

    def _record_knowledge_checkpoint(self, memory: Optional[Dict[str, Any]] = None) -> None:
        """Best-effort durable memory; never turn an optional cache into a blocker."""
        if self._knowledge_error:
            return
        try:
            value = memory if isinstance(memory, dict) else self._build_context_memory()
            self.knowledge.record_checkpoint(value)
        except Exception as exc:
            self._knowledge_error = f"{type(exc).__name__}: {exc}"

    def _record_knowledge_result(self, goal: str, result: Any) -> None:
        if self._knowledge_error:
            return
        try:
            self.knowledge.record_result(goal, result)
        except Exception as exc:
            self._knowledge_error = f"{type(exc).__name__}: {exc}"

    def _context_memory_for_turn(self, prompt: str = "") -> Dict[str, Any]:
        """Refresh cheap current-state edges around the durable memory index."""
        memory = dict(self.session.memory) if isinstance(self.session.memory, dict) else {}
        focus = " ".join(
            item
            for item in (str(self.session.goal or ""), str(prompt or ""))
            if item.strip()
        )
        prior = self.prior_knowledge_snapshot(
            limit=8,
            max_chars=7000,
            focus=focus,
        )
        if prior.get("available") and prior.get("records"):
            memory["prior_knowledge"] = prior
        bank = self.goal_bank_snapshot(queued_limit=8, recent_limit=4, display_limit=480)
        if bank.get("available") and (
            bank.get("queued_count") or bank.get("running_count") or bank.get("recent")
        ):
            memory["goal_bank"] = bank
        receipts = self._compact_receipts()
        if receipts:
            memory["receipts"] = receipts
        return memory

    def goal_bank_snapshot(
        self,
        *,
        queued_limit: int = 16,
        recent_limit: int = 6,
        display_limit: int = 640,
    ) -> Dict[str, Any]:
        """Return bounded queue state for the TUI, /status, and memory."""
        if self._goal_bank_error:
            return {
                "schema": "hcli.goal_bank.v1",
                "available": False,
                "path": str(self.goal_bank.path),
                "reason": self._goal_bank_error,
                "queued_count": 0,
                "running_count": 0,
                "queued": [],
                "running": [],
                "recent": [],
                "next": None,
            }
        try:
            return self.goal_bank.snapshot(
                queued_limit=queued_limit,
                recent_limit=recent_limit,
                display_limit=display_limit,
            )
        except Exception as exc:
            self._goal_bank_error = f"{type(exc).__name__}: {exc}"
            return {
                "schema": "hcli.goal_bank.v1",
                "available": False,
                "path": str(self.goal_bank.path),
                "reason": self._goal_bank_error,
                "queued_count": 0,
                "running_count": 0,
                "queued": [],
                "running": [],
                "recent": [],
                "next": None,
            }

    def bank_goal(self, goal: str, *, mode: str = "auto") -> Dict[str, Any]:
        """Persist a future goal without changing the active goal or steer."""
        if self._goal_bank_error:
            raise GoalBankError(self._goal_bank_error)
        item = self.goal_bank.add(goal, mode=mode)
        self._emit(
            "bank_queued",
            {
                "id": item.get("id"),
                "goal": item.get("goal"),
                "mode": item.get("mode"),
                "queued_count": self.goal_bank_snapshot().get("queued_count", 0),
            },
        )
        self._persist_session()
        return item

    def drop_banked_goal(self, selector: str) -> Optional[Dict[str, Any]]:
        if self._goal_bank_error:
            raise GoalBankError(self._goal_bank_error)
        item = self.goal_bank.drop(selector)
        if item is not None:
            self._emit(
                "bank_dropped",
                {"id": item.get("id"), "goal": item.get("goal")},
            )
            self._persist_session()
        return item

    def clear_banked_goals(self) -> int:
        if self._goal_bank_error:
            raise GoalBankError(self._goal_bank_error)
        removed = self.goal_bank.clear()
        if removed:
            self._emit("bank_cleared", {"removed": removed})
            self._persist_session()
        return removed

    def _finish_banked_goal(
        self,
        item_id: Optional[str],
        result: Any,
        *,
        error: Optional[str] = None,
    ) -> None:
        if not item_id:
            return
        try:
            item = self.goal_bank.finish(item_id, result, error=error)
        except Exception as exc:
            self._emit(
                "warning",
                {"message": f"bank completion could not be persisted: {type(exc).__name__}: {exc}"},
            )
            return
        if item is not None:
            self._emit(
                "bank_finished",
                {
                    "id": item.get("id"),
                    "goal": item.get("goal"),
                    "status": item.get("status"),
                    "error": item.get("last_error"),
                },
            )

    def _auto_start_banked_goals(self, *, runner: str) -> List[Dict[str, Any]]:
        """Drain completed-goal continuations without recursive call stacks."""
        if self._bank_auto_promoting:
            return []
        self._bank_auto_promoting = True
        promoted: List[Dict[str, Any]] = []
        current_runner = "mission" if runner == "mission" else "execute"
        try:
            while True:
                try:
                    item = self.goal_bank.claim_next()
                except Exception as exc:
                    self._goal_bank_error = f"{type(exc).__name__}: {exc}"
                    self._emit("warning", {"message": f"bank unavailable: {self._goal_bank_error}"})
                    break
                if item is None:
                    break
                item_runner = (
                    "mission"
                    if item.get("mode") == "mission" or current_runner == "mission"
                    else "execute"
                )
                self._emit(
                    "bank_started",
                    {
                        "id": item.get("id"),
                        "goal": item.get("goal"),
                        "mode": item_runner,
                        "queued_count": self.goal_bank_snapshot().get("queued_count", 0),
                    },
                )
                try:
                    if item_runner == "mission":
                        result = self.run_mission(
                            str(item.get("goal") or ""),
                            _bank_item_id=str(item.get("id") or ""),
                            _auto_promote=False,
                        )
                    else:
                        result = self.execute(
                            str(item.get("goal") or ""),
                            _bank_item_id=str(item.get("id") or ""),
                            _auto_promote=False,
                        )
                except Exception as exc:
                    self._finish_banked_goal(
                        str(item.get("id") or ""),
                        {"status": "failed"},
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    break
                result_status = (
                    str(result.get("status") or "")
                    if isinstance(result, dict)
                    else "failed"
                ).lower()
                promoted.append(
                    {
                        "id": item.get("id"),
                        "goal": item.get("goal"),
                        "mode": item_runner,
                        "status": result_status,
                    }
                )
                if result_status != "completed":
                    break
                current_runner = item_runner
        finally:
            self._bank_auto_promoting = False
        return promoted

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
        *,
        _bank_item_id: Optional[str] = None,
        _auto_promote: bool = True,
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

        self._emit(
            "activity_started",
            {"label": "working"},
        )
        self.session.append_message("user", prompt, kind="goal")
        try:
            try:
                parameters = inspect.signature(self.engine.execute).parameters
                supports_memory = (
                    "context_memory" in parameters
                    or any(
                        item.kind is inspect.Parameter.VAR_KEYWORD
                        for item in parameters.values()
                    )
                )
            except (TypeError, ValueError):
                supports_memory = True
            if supports_memory:
                result = self.engine.execute(
                    prompt,
                    context_memory=self._context_memory_for_turn(prompt),
                )
            else:
                result = self.engine.execute(prompt)
        finally:
            self._persist_session()

        if isinstance(result, dict):
            reply = result.get("content") or result.get("error") or result.get("status") or ""
        else:
            reply = result
        self.session.append_message("assistant", reply, kind="result")
        try:
            if len(self.session.messages) > _AUTO_COMPACT_MESSAGES:
                self.compact_context()
            else:
                self._persist_session()
        finally:
            # A claimed bank item must not remain ``running`` merely because
            # the session crossed an automatic compaction boundary.
            self._finish_banked_goal(_bank_item_id, result)
        if (
            _auto_promote
            and isinstance(result, dict)
            and str(result.get("status") or "").lower() == "completed"
        ):
            promoted = self._auto_start_banked_goals(runner="execute")
            if promoted:
                result["bank_started"] = promoted
        self._record_knowledge_result(prompt, result)
        return result

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
            # No mission object attached -- the DETACHED SUPERVISOR case, which the
            # Odyssey contract treats as normal: Claude attaches, injects, detaches.
            # Keying the queue on THIS process's session id wrote the steer to a file
            # the running mission never reads, and the CLI still printed a success
            # tick. Measured: mission session 3f254d6d..., steer landed in
            # c22b3aa4..., no file for the mission at all.
            #
            # Target the LIVE mission's session when one is on disk, so the steer
            # reaches the queue the mission actually polls.
            live = _live_mission_identity(self.workspace_root)
            target_session = (live or {}).get("session_id") or self.session.id
            queue = SteeringQueue(self.workspace_root, target_session)
            event = queue.enqueue(
                body,
                kind=token,
                mission_id=(live or {}).get("id"),
                source_session_id=self.session.id,
            )
        self.session.steering.append(body)
        try:
            self.knowledge.record_note(body, kind=token)
        except Exception as exc:
            self._knowledge_error = f"{type(exc).__name__}: {exc}"
        self._persist_session()
        return event

    def run_mission(
        self,
        goal: Optional[str] = None,
        units: Any = None,
        engine: Any = None,
        *,
        _bank_item_id: Optional[str] = None,
        _auto_promote: bool = True,
        **kwargs: Any,
    ) -> dict:
        text = (
            goal
            if goal is not None
            else self.session.goal
        )
        # Popped once, before the fork: both branches pass the same value, and
        # a failed adopt must not consume it on its way to the replace path.
        if "context_memory" in kwargs:
            memory = kwargs.pop("context_memory")
        else:
            memory = self._context_memory_for_turn()
        # Reissuing a goal used to MINT a new mission over the old one: the
        # constructor's Scheduler._persist() rewrites dag.json, and because
        # content_identity() ignores status, a content-identical graph is not
        # an IdentityConflict and is not retired -- completed units came back
        # `pending` and the prior id survived only in mission.log. Adopt
        # instead, but only on a verified match: same goal text, and a phase
        # the mission can still be advanced from. Agent.run() already forks
        # this way (runtime.py); it just never had the identity check because
        # its resume path is the one that passes no goal at all.
        adopt = False
        if units is None:
            try:
                prior = load_state(mission_state_path(self.workspace_root))
            except Exception:
                prior = {}
            adopt = (
                str(prior.get("goal") or "") == (text or "")
                and str(prior.get("phase") or "") not in TERMINAL_MISSION_PHASES
            )
        if adopt:
            # A corrupt state.json, a retired dag, a WorkUnit schema that moved
            # -- any of these raise in from_workspace. None of them may make
            # /mission unusable: restoring is the optimisation, minting a fresh
            # mission is still a correct answer, so fall through on failure.
            try:
                self.mission = Mission.from_workspace(
                    self.workspace_root,
                    engine=engine if engine is not None else self.engine,
                    goal=text or "",
                    runtime_count=self.runtime_count,
                    runtime_pool=self.runtime_pool,
                    session_id=self.session.id,
                    context_memory=memory,
                    **kwargs,
                )
            except Exception:
                adopt = False
            else:
                # An adopted mission inherits the on-disk obligations too. Without
                # this, /steer against a resumed mission reads a None ledger.
                self._reload_ledger()
        if not adopt:
            self.mission = Mission(
                self.workspace_root,
                engine=engine if engine is not None else self.engine,
                units=units,
                goal=text or "",
                runtime_count=self.runtime_count,
                runtime_pool=self.runtime_pool,
                session_id=self.session.id,
                context_memory=memory,
                **kwargs,
            )
        self.session.mission_id = self.mission.id
        self._persist_session()
        result = self.mission.run()
        self._finish_banked_goal(_bank_item_id, result)
        if (
            _auto_promote
            and isinstance(result, dict)
            and str(result.get("status") or "").lower() == "completed"
        ):
            promoted = self._auto_start_banked_goals(runner="mission")
            if promoted:
                result["bank_started"] = promoted
        self._record_knowledge_result(text or "mission", result)
        return result

    @staticmethod
    def _memory_text(value: Any, limit: int = _MEMORY_TEXT_LIMIT) -> str:
        # Every compaction-memory string reaches a worker prompt through here, so
        # this is where control tokens have to die. HCLI's own error text names
        # `<think>` and `reasoning_content`; pasting that into the next prompt
        # steers the resident into a reasoning-only reply, which trips the guard
        # that rejects reasoning, which records the same text again. Sanitising
        # only the knowledge store missed this path -- receipts carry `error` and
        # `blocker` straight from disk.
        text = _sanitise_text(str(value or "").strip())
        if len(text) > limit:
            return text[: limit - 1].rstrip() + "…"
        return text

    def _compact_git_state(self) -> Dict[str, Any]:
        """Capture the index and worktree separately for compaction memory."""
        root = Path(self.workspace_root)
        try:
            proc = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain=v1",
                    "--branch",
                    "--untracked-files=normal",
                ],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        if proc.returncode != 0:
            return {
                "available": False,
                "reason": self._memory_text(proc.stderr or "git status failed", 240),
            }

        branch = ""
        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []
        for line in (proc.stdout or "").splitlines():
            if line.startswith("## "):
                branch = line[3:].strip()
                continue
            if len(line) < 3:
                continue
            code = line[:2]
            path = line[3:].strip()
            if code == "??":
                untracked.append(path)
                continue
            if code[0] != " ":
                staged.append(path)
            if code[1] != " ":
                unstaged.append(path)

        def bounded(paths: list[str]) -> Dict[str, Any]:
            return {
                "count": len(paths),
                "paths": paths[:_MEMORY_LIST_LIMIT],
                "truncated": len(paths) > _MEMORY_LIST_LIMIT,
            }

        return {
            "available": True,
            "branch": branch,
            "staged": bounded(staged),
            "unstaged": bounded(unstaged),
            "untracked": bounded(untracked),
        }

    def _compact_steering(self) -> Dict[str, Any]:
        try:
            events = SteeringQueue(self.workspace_root, self.session.id).all()
        except Exception as exc:
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        rank = {"constraint": 0, "correction": 1, "knowledge": 2}
        ordered = sorted(
            events,
            key=lambda event: (
                rank.get(str(getattr(event, "kind", "knowledge")), 3),
                -float(getattr(event, "timestamp", 0.0) or 0.0),
            ),
        )
        rows = [
            {
                "id": str(getattr(event, "id", "")),
                "kind": str(getattr(event, "kind", "knowledge")),
                "applied": bool(getattr(event, "applied", False)),
                "text": self._memory_text(getattr(event, "text", "")),
            }
            for event in ordered[:_MEMORY_LIST_LIMIT]
        ]
        return {
            "available": True,
            "pending_count": sum(
                1 for event in events if not bool(getattr(event, "applied", False))
            ),
            "events": rows,
            "truncated": len(events) > _MEMORY_LIST_LIMIT,
        }

    def _compact_goal(self) -> Dict[str, Any]:
        goal = self._memory_text(self.session.goal, 2000)
        result: Dict[str, Any] = {"text": goal}
        if not goal:
            return result
        try:
            compiled = self.engine.goal_compiler.compile(self.session.goal)
        except Exception:
            try:
                from .goal import GoalCompiler

                compiled = GoalCompiler().compile(self.session.goal)
            except Exception:
                compiled = {}
        if not isinstance(compiled, dict):
            return result
        result.update(
            {
                "summary": self._memory_text(compiled.get("goal_summary"), 600),
                "invariants": [
                    self._memory_text(item)
                    for item in (compiled.get("invariants") or [])[:8]
                ],
                "acceptance": [
                    self._memory_text(item)
                    for item in (compiled.get("acceptance_criteria") or [])[:8]
                ],
                "files": [
                    self._memory_text(item, 240)
                    for item in (compiled.get("referenced_files") or [])[:12]
                ],
            }
        )
        return result

    def _compact_mission(self) -> Dict[str, Any]:
        mission = self.mission
        if mission is None:
            return {"id": self.session.mission_id, "phase": "none"}
        try:
            snapshot = mission.status()
        except Exception as exc:
            snapshot = {"error": f"{type(exc).__name__}: {exc}"}
        rows = []
        units = getattr(getattr(mission, "scheduler", None), "units", {}) or {}
        priority = {"running": 0, "failed": 1, "ready": 2, "pending": 3, "completed": 4}
        values = sorted(
            units.values(),
            key=lambda unit: priority.get(str(getattr(unit, "status", "")), 5),
        )
        for unit in values[:_MEMORY_LIST_LIMIT]:
            failure = getattr(unit, "failure_context", None)
            why = ""
            if isinstance(failure, dict):
                why = failure.get("error") or failure.get("reason") or ""
            rows.append(
                {
                    "id": str(getattr(unit, "id", "")),
                    "status": str(getattr(unit, "status", "")),
                    "role": self._memory_text(getattr(unit, "role", ""), 120),
                    "description": self._memory_text(getattr(unit, "description", ""), 320),
                    "why": self._memory_text(why, 320),
                }
            )
        state = {
            key: snapshot.get(key)
            for key in (
                "phase",
                "state",
                "units_by_status",
                "active_runtimes",
                "active_decodes",
                "accepted_units_per_hour",
                "last_checkpoint",
                "no_progress_warning",
            )
            if isinstance(snapshot, dict) and key in snapshot
        }
        return {
            "id": getattr(mission, "id", self.session.mission_id),
            "phase": snapshot.get("phase") if isinstance(snapshot, dict) else None,
            "state": state,
            "checkpoint_id": getattr(mission, "_last_checkpoint_id", None),
            "units": rows,
            "units_truncated": len(values) > _MEMORY_LIST_LIMIT,
        }

    def _compact_ledger(self) -> Dict[str, Any]:
        ledger = self._status_ledger()
        if ledger is None:
            return {"available": False}
        try:
            obligations = list(ledger.obligations())
        except Exception as exc:
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        rows = []
        for obligation in obligations[:_MEMORY_LIST_LIMIT]:
            rows.append(
                {
                    "id": str(getattr(obligation, "id", "")),
                    "status": str(getattr(obligation, "status", "")),
                    "text": self._memory_text(getattr(obligation, "text", ""), 360),
                }
            )
        return {
            "available": True,
            "terminal_blocker": self._memory_text(
                getattr(ledger, "terminal_blocker", None), 600
            ),
            "obligations": rows,
            "truncated": len(obligations) > _MEMORY_LIST_LIMIT,
        }

    def _compact_receipts(self) -> list[Dict[str, Any]]:
        directory = Path(self.workspace_root) / ".hcli" / "receipts"
        if not directory.is_dir():
            return []
        rows = []
        try:
            paths = sorted(
                directory.glob("*.json"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )[:6]
        except OSError:
            return []
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            envelope = data.get("result_envelope")
            envelope = envelope if isinstance(envelope, dict) else {}
            rows.append(
                {
                    "file": path.name,
                    "goal_id": data.get("goal_id") or path.stem,
                    "status": data.get("status"),
                    "kind": data.get("kind"),
                    # These are the parts of an old run that can change the
                    # next decision. Raw model calls and tracebacks stay in
                    # the receipt on disk, not in every future prompt.
                    "goal": self._memory_text(data.get("goal"), 480),
                    "claim": self._memory_text(
                        envelope.get("claim") or data.get("goal"), 360
                    ),
                    "verdict": self._memory_text(
                        envelope.get("verdict") or data.get("verdict"), 120
                    ),
                    "blocker": self._memory_text(
                        data.get("error") or envelope.get("blocker"), 360
                    ),
                    "next_action": self._memory_text(
                        envelope.get("next_action"), 360
                    ),
                }
            )
        return rows

    def _build_context_memory(self) -> Dict[str, Any]:
        recent = []
        for item in self.session.messages[-8:]:
            if not isinstance(item, dict):
                continue
            recent.append(
                {
                    "role": str(item.get("role") or ""),
                    "kind": str(item.get("kind") or "conversation"),
                    "content": self._memory_text(item.get("content")),
                }
            )
        return {
            "schema": CONTEXT_MEMORY_SCHEMA,
            "generation": int(self.session.compaction_count) + 1,
            "compacted_at": time.time(),
            "active_goal": self._compact_goal(),
            "mission": self._compact_mission(),
            "ledger": self._compact_ledger(),
            "steering": self._compact_steering(),
            "staging": self._compact_git_state(),
            "prior_knowledge": self.prior_knowledge_snapshot(
                limit=6,
                max_chars=6000,
            ),
            "goal_bank": self.goal_bank_snapshot(
                queued_limit=8,
                recent_limit=4,
                display_limit=480,
            ),
            "recent": recent,
            "receipts": self._compact_receipts(),
            "retention": {
                "hot_messages_kept": 4,
                "raw_history": "older messages are gzip archived on the workspace SSD; named receipts remain on disk",
                "rule": "constraints and verifier state outrank conversational recency",
            },
        }

    def context_summary(
        self,
    ) -> str:
        memory = self.session.memory if isinstance(self.session.memory, dict) else {}
        generation = memory.get("generation")
        prior = memory.get("prior_knowledge")
        prior_generation = prior.get("generation") if isinstance(prior, dict) else None
        if generation:
            memory_text = f" memory=checkpoint#{generation}"
        elif prior_generation:
            memory_text = f" memory=prior#{prior_generation}"
        else:
            memory_text = " memory=none"
        bank = self.goal_bank_snapshot()
        bank_text = ""
        if bank.get("available"):
            bank_text = f" bank={int(bank.get('queued_count') or 0)}"
        else:
            bank_text = " bank=unavailable"
        return (
            f"session {self.session.id} "
            f"messages={len(self.session.messages)}{memory_text}{bank_text}"
        )

    def compact_context(
        self,
    ) -> Dict[str, Any]:
        archive = self.session_store.archive_messages(self.session)
        memory = self._build_context_memory()
        self._record_knowledge_checkpoint(memory)
        memory["prior_knowledge"] = self.prior_knowledge_snapshot(
            limit=6,
            max_chars=6000,
        )
        memory["history_archive"] = archive
        self.session.set_memory(memory)
        self.session.compaction_count += 1
        self.session.compacted_at = str(memory.get("compacted_at"))
        self.session.messages = self.session.messages[-4:]
        self._persist_session()
        return memory

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
        self._reload_ledger()
        return mission

    def _reload_ledger(self) -> None:
        """Re-read the on-disk GOAL.md into `_ledger`.

        Any path that adopts a mission off disk must also adopt its ledger:
        `_steer` reads `self._ledger`, so a restored mission with a None
        ledger silently drops every constraint the operator files against it.
        """
        goal_md = Path(self.workspace_root) / ".hcli" / "GOAL.md"
        if goal_md.is_file():
            try:
                from .ledger import Ledger

                self._ledger = Ledger.parse(goal_md)
            except Exception:
                pass

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
            self._record_knowledge_checkpoint()
        except Exception:
            pass

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
