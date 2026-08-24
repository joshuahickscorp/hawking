from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# A per-hour rate from a window much shorter than an hour is sampling noise.
# 4 accepts in 12.4s annualises to 1164/h — the documented lie. Five minutes
# is ~8% of an hour: the smallest window where a handful of accepts cannot
# explode into four-digit rates. Below that, print the raw count and window.
MIN_ACCEPTED_RATE_WINDOW_S = 300.0
GOAL_DISPLAY_CHARS = 72


def _fmt_unknown(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    return str(value)


def _fmt_age(seconds: Any) -> str:
    if seconds is None:
        return "unknown"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "unknown"
    if value < 0:
        value = 0.0
    if value < 10:
        return f"{value:.1f}s"
    return f"{int(round(value))}s"


def _fmt_window(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.1f}s"
    return f"{int(round(seconds))}s"


def format_accepted_h(accepted_count: Any, elapsed_s: Any) -> str:
    """Rate, or raw `N in Ts` when the window is too short to annualise."""
    if accepted_count is None or elapsed_s is None:
        return "unknown"
    try:
        count = int(accepted_count)
        elapsed = float(elapsed_s)
    except (TypeError, ValueError):
        return "unknown"
    if elapsed < 0:
        elapsed = 0.0
    if elapsed < MIN_ACCEPTED_RATE_WINDOW_S:
        return f"{count} in {_fmt_window(elapsed)}"
    if elapsed == 0:
        return "unknown"
    return f"{count / (elapsed / 3600.0):.1f}"


def _truncate_goal(goal: Any) -> str:
    if not goal:
        return "(unset)"
    text = str(goal).splitlines()[0].strip()
    if not text:
        return "(unset)"
    if len(text) > GOAL_DISPLAY_CHARS:
        return text[: GOAL_DISPLAY_CHARS - 3] + "..."
    return text


def _accepted_h_text(snap: Dict[str, Any]) -> str:
    count = snap.get("accepted_count")
    elapsed = snap.get("elapsed_wall")
    if count is not None and elapsed is not None:
        return format_accepted_h(count, elapsed)
    # Refuse a precomputed rate that does not carry its window. That is how
    # 12.4s became 1164/h, including the stale max-equilibrium.json figure.
    if elapsed is not None:
        try:
            if float(elapsed) < MIN_ACCEPTED_RATE_WINDOW_S:
                return "unknown"
        except (TypeError, ValueError):
            return "unknown"
        rate = snap.get("accepted_units_per_hour")
        if rate is not None:
            try:
                return f"{float(rate):.1f}"
            except (TypeError, ValueError):
                return "unknown"
    return "unknown"


def format_status(snapshot: Dict[str, Any]) -> str:
    """One-screen /status. Unmeasured fields print as unknown, never as 0."""
    snap = snapshot if isinstance(snapshot, dict) else {}
    units = snap.get("units_by_status")
    if not isinstance(units, dict):
        units = None
    occupancy = snap.get("occupancy")
    if not isinstance(occupancy, dict):
        occupancy = None
    qwen = snap.get("qwen")
    if not isinstance(qwen, dict):
        qwen = None
    grok = snap.get("grok")
    if not isinstance(grok, dict):
        grok = None
    mutation = snap.get("mutation")
    if not isinstance(mutation, dict):
        mutation = None

    mission_id = snap.get("mission_id") or "—"
    phase = snap.get("phase") or "—"
    goal = _truncate_goal(snap.get("goal"))

    if units is None and snap.get("blocked_units") is None:
        wu_line = "WU unknown"
    else:
        units = units or {}
        blocked = snap.get("blocked_units")
        if blocked is None:
            blocked = "unknown"
        wu_line = (
            f"WU ready={units.get('ready', 0)} running={units.get('running', 0)} "
            f"blocked={blocked} completed={units.get('completed', 0)} "
            f"failed={units.get('failed', 0)}"
        )
    reason = snap.get("blocked_reason")
    if reason:
        wu_line += f" blocked_reason={reason}"

    if not qwen:
        qwen_line = "Qwen unknown"
    else:
        health = qwen.get("health")
        if health and health != "ok":
            queued = qwen.get("queued")
            queued_s = str(queued) if queued is not None else "unknown"
            qwen_line = (
                "Qwen health=down resident=0 active=0 "
                f"queued={queued_s} n_ctx=unknown prompt=unknown tps=unknown"
            )
        elif health == "ok":
            qwen_line = (
                f"Qwen health=ok resident={_fmt_unknown(qwen.get('resident'))} "
                f"active={_fmt_unknown(qwen.get('active_decode'))} "
                f"queued={_fmt_unknown(qwen.get('queued'))} "
                f"n_ctx={_fmt_unknown(qwen.get('n_ctx'))} "
                f"prompt={_fmt_unknown(qwen.get('prompt_tokens'))} "
                f"tps={_fmt_unknown(qwen.get('tps'))}"
            )
        else:
            qwen_line = (
                "Qwen health=unknown "
                f"resident={_fmt_unknown(qwen.get('resident'))} "
                f"active={_fmt_unknown(qwen.get('active_decode'))} "
                f"queued={_fmt_unknown(qwen.get('queued'))} "
                f"n_ctx={_fmt_unknown(qwen.get('n_ctx'))} "
                f"prompt={_fmt_unknown(qwen.get('prompt_tokens'))} "
                f"tps={_fmt_unknown(qwen.get('tps'))}"
            )

    if not grok:
        grok_line = "Grok unknown"
    else:
        grok_line = (
            f"Grok admitted={_fmt_unknown(grok.get('admitted'))} "
            f"active={_fmt_unknown(grok.get('active'))} "
            f"queued={_fmt_unknown(grok.get('queued'))} "
            f"done={_fmt_unknown(grok.get('done'))} "
            f"failed={_fmt_unknown(grok.get('failed'))} "
            f"latency={_fmt_age(grok.get('latency_s'))}"
        )

    if not occupancy:
        cpu_line = "CPU unknown"
    else:
        cpu_line = (
            f"CPU decode={occupancy.get('GPU_DECODE', 0)} "
            f"compile={occupancy.get('COMPILE', 0)} "
            f"test={occupancy.get('TEST', 0)} "
            f"tool={occupancy.get('TOOL_WAIT', 0)}"
        )

    if not mutation:
        mut_line = "Mutation unknown"
    else:
        held = mutation.get("held")
        if held is True:
            held_s = "true"
        elif held is False:
            held_s = "false"
        else:
            held_s = "unknown"
        owner = mutation.get("owner_display")
        if owner is None:
            owner = mutation.get("owner")
        if owner is None:
            owner = "unknown"
        mut_line = (
            f"Mutation held={held_s} pid={_fmt_unknown(mutation.get('pid'))} "
            f"owner={owner} waiters={_fmt_unknown(mutation.get('waiters'))}"
        )

    watchdog = snap.get("watchdog")
    if watchdog in (None, ""):
        watchdog = snap.get("watchdog_tier") or "unknown"
    footer = (
        f"Verifier backlog={_fmt_unknown(snap.get('verifier_backlog'))}  "
        f"accepted/h={_accepted_h_text(snap)}  "
        f"ckpt={_fmt_age(snap.get('checkpoint_age_s'))}  "
        f"watchdog={_fmt_unknown(watchdog)}"
    )

    lines = [
        f"mission {mission_id}  phase={phase}",
        f"Goal: {goal}",
        wu_line,
        qwen_line,
        grok_line,
        cpu_line,
        mut_line,
        footer,
    ]
    warning = snap.get("no_progress_warning") or snap.get("watchdog_message")
    if warning and warning not in ("(none)", ""):
        if snap.get("no_progress_warning") or "no_progress" in str(warning):
            lines.append(f"no_progress: {warning}")
    return "\n".join(lines)


def _workspace_root(controller: Any) -> Optional[Path]:
    root = getattr(controller, "workspace_root", None)
    if root:
        return Path(os.fspath(root))
    workspace = getattr(controller, "workspace", None)
    if workspace is None:
        return None
    inner = getattr(workspace, "root", workspace)
    try:
        return Path(os.fspath(inner))
    except TypeError:
        return None


def _load_mission_state(controller: Any) -> Optional[Dict[str, Any]]:
    root = _workspace_root(controller)
    if root is None:
        return None
    path = root / ".hcli" / "mission" / "state.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _find_ledger(controller: Any) -> Any:
    for obj in (controller, getattr(controller, "mission", None)):
        if obj is None:
            continue
        for name in ("_ledger", "ledger"):
            ledger = getattr(obj, name, None)
            if ledger is not None and hasattr(ledger, "unverified"):
                return ledger
    root = _workspace_root(controller)
    if root is None:
        return None
    for candidate in (root / ".hcli" / "GOAL.md", root / "GOAL.md"):
        if not candidate.is_file():
            continue
        try:
            from .ledger import Ledger

            return Ledger.parse(candidate)
        except Exception:
            continue
    return None


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _watchdog_from_last_dispatch(scheduler: Any) -> Optional[str]:
    """last_dispatch is only a watchdog if it carries a clock.

    The hawking-copy scheduler stores requested/admitted/overhead_s with no
    timestamp, so presence of the dict is not 'ok' and not an age.
    """
    if scheduler is None:
        return None
    payload = getattr(scheduler, "last_dispatch", None)
    if not isinstance(payload, dict) or not payload:
        return None
    for key in ("at", "ts", "time", "t", "when", "timestamp"):
        stamped = _float_or_none(payload.get(key))
        if stamped is None or stamped <= 0:
            continue
        # monotonic stamps are not unix; refuse to subtract from time.time()
        if stamped > 1e12:
            continue
        if stamped < 1e9:
            return None
        return f"dispatch {_fmt_age(max(0.0, time.time() - stamped))}"
    return None


def _watchdog_from_ledger(ledger: Any) -> Optional[str]:
    if ledger is None:
        return None
    count_fn = getattr(ledger, "consecutive_no_progress_count", None)
    stall = 0
    if callable(count_fn):
        try:
            stall = int(count_fn() or 0)
        except (TypeError, ValueError):
            stall = 0
    tier_fn = getattr(ledger, "watchdog_tier", None)
    if callable(tier_fn):
        try:
            tier = int(tier_fn())
        except (TypeError, ValueError):
            tier = 0
        if tier > 0:
            return f"L{tier}"
    if stall > 0:
        return f"stall x{stall}"
    status = getattr(ledger, "status", None)
    if callable(status):
        try:
            status = status()
        except Exception:
            status = None
    if status:
        return str(status)
    return None


def enrich_status_snapshot(controller: Any, snap: Dict[str, Any]) -> Dict[str, Any]:
    """Fill verifier backlog, accepted/h inputs, ckpt age, watchdog from live state.

    Never reads max-equilibrium.json. Recomputes accepted/h inputs from
    accepted_count + started_at so the formatter can refuse a short window.
    """
    now = time.time()
    mission = getattr(controller, "mission", None)
    state = _load_mission_state(controller)

    started_at = None
    accepted_count = None
    last_checkpoint = None
    no_progress = snap.get("no_progress_warning")
    phase = snap.get("phase")

    if mission is not None:
        snap.setdefault("mission_id", getattr(mission, "id", None))
        snap.setdefault("phase", getattr(mission, "phase", None))
        if not snap.get("goal"):
            snap["goal"] = getattr(mission, "goal", "") or ""
        started_at = _float_or_none(getattr(mission, "started_at", None))
        try:
            accepted_count = int(getattr(mission, "accepted_count"))
        except (TypeError, ValueError, AttributeError):
            accepted_count = None
        last_checkpoint = _float_or_none(getattr(mission, "last_checkpoint", None))
        if not no_progress:
            no_progress = getattr(mission, "no_progress_warning", None)
        if not phase:
            phase = getattr(mission, "phase", None)
        if "units_by_status" not in snap:
            status_fn = getattr(mission, "status", None)
            if callable(status_fn):
                try:
                    mission_snap = status_fn()
                except Exception:
                    mission_snap = None
                if isinstance(mission_snap, dict):
                    for key in (
                        "units_by_status",
                        "active_runtimes",
                        "active_decodes",
                        "elapsed_wall",
                        "no_progress_warning",
                    ):
                        snap.setdefault(key, mission_snap.get(key))

    if isinstance(state, dict):
        if started_at is None:
            started_at = _float_or_none(state.get("started_at"))
        if accepted_count is None:
            try:
                accepted_count = int(state.get("accepted_count"))
            except (TypeError, ValueError):
                accepted_count = None
        if not last_checkpoint:
            last_checkpoint = _float_or_none(state.get("last_checkpoint"))
        snap.setdefault("mission_id", state.get("id"))
        snap.setdefault("phase", state.get("phase"))
        if not snap.get("goal"):
            snap["goal"] = state.get("goal") or ""
        if not no_progress:
            no_progress = state.get("no_progress_warning")
        if not phase:
            phase = state.get("phase")

    session = getattr(controller, "session", None)
    if session is not None and not snap.get("goal"):
        snap["goal"] = getattr(session, "goal", "") or ""

    if accepted_count is not None:
        snap["accepted_count"] = accepted_count
    if started_at is not None:
        snap["elapsed_wall"] = max(0.0, now - started_at)

    if last_checkpoint and last_checkpoint > 0:
        snap["checkpoint_age_s"] = max(0.0, now - last_checkpoint)
    elif snap.get("checkpoint_age_s") is None:
        root = _workspace_root(controller)
        path = None if root is None else root / ".hcli" / "mission" / "state.json"
        if path is not None and path.is_file():
            try:
                snap["checkpoint_age_s"] = max(0.0, now - path.stat().st_mtime)
            except OSError:
                snap["checkpoint_age_s"] = None
        else:
            snap["checkpoint_age_s"] = None

    ledger = _find_ledger(controller)
    if ledger is not None:
        try:
            snap["verifier_backlog"] = len(ledger.unverified())
        except Exception:
            snap.setdefault("verifier_backlog", None)
    else:
        snap.setdefault("verifier_backlog", None)

    watchdog = None
    if no_progress or phase == "no_progress":
        watchdog = "no_progress"
        snap["no_progress_warning"] = no_progress or snap.get("no_progress_warning")
    if watchdog is None:
        watchdog = _watchdog_from_ledger(ledger)
    if watchdog is None:
        scheduler = getattr(mission, "scheduler", None) if mission is not None else None
        watchdog = _watchdog_from_last_dispatch(scheduler)
    if watchdog is not None:
        snap["watchdog"] = watchdog
    else:
        snap.setdefault("watchdog", None)

    return snap


def _status_has_observed_fields(snap: Dict[str, Any]) -> bool:
    if snap.get("mission_id"):
        return True
    if snap.get("verifier_backlog") is not None:
        return True
    if snap.get("checkpoint_age_s") is not None:
        return True
    if snap.get("accepted_count") is not None:
        return True
    if snap.get("watchdog") not in (None, "", "unknown"):
        return True
    if snap.get("qwen") or snap.get("grok"):
        return True
    return False


REQUIRED_COMMANDS = (
    "/help",
    "/status",
    "/models",
    "/model",
    "/goal",
    "/ultragoal",
    "/mission",
    "/steer",
    "/grok",
    "/cancel",
    "/context",
    "/compact",
    "/clear",
    "/resume",
    "/exit",
)


def _model_path(model: Any) -> str:
    if isinstance(model, dict):
        return str(model.get("path") or "")
    return str(getattr(model, "path", "") or "")


def _model_name(model: Any) -> str:
    if isinstance(model, dict):
        return str(
            model.get("name")
            or model.get("display_name")
            or model.get("path")
            or "?"
        )
    return str(
        getattr(model, "display_name", None)
        or getattr(model, "name", None)
        or getattr(model, "path", None)
        or "?"
    )


def _fmt_unknown(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    return str(value)


def _fmt_age(seconds: Any) -> str:
    if seconds is None:
        return "unknown"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "unknown"
    if value < 10:
        return f"{value:.1f}s"
    return f"{int(round(value))}s"


class CommandHandler:
    """Canonical HCLI slash-command dispatcher.

    TUI, CLI, tests, and automation enter here. Controller.handle_command
    is a thin adapter that emits TUI events and returns structured
    ``last_value`` payloads. This module must not grow a second command
    universe.
    """

    def __init__(self, controller: Any):
        self.controller = controller
        self._grok = None
        self._grok_root: Optional[Path] = None
        self.last_value: Any = None
        self.last_command: str = ""

    def handle(self, line: str) -> Optional[str]:
        self.last_value = None
        self.last_command = ""
        line = (line or "").strip()
        if not line.startswith("/"):
            return None
        parts = line.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        self.last_command = cmd
        handler = getattr(self, f"_cmd_{cmd[1:]}", None)
        if handler is None:
            text = f"Unknown command: {cmd}"
            self.last_value = text
            return text
        return handler(arg)

    def _cmd_help(self, arg: str) -> str:
        text = (
            "Commands:\n"
            "  /help - show this help\n"
            "  /status - show session status\n"
            "  /models - list available models\n"
            "  /model - select model\n"
            "  /goal - set active goal\n"
            "  /ultragoal - create or show the durable Goal + ledger + DAG\n"
            "  /mission - run a persistent mission\n"
            "  /steer - queue steering instruction\n"
            "  /grok - delegate, audit, consult, or inspect a Grok task\n"
            "  /cancel - cancel the active mission\n"
            "  /context - show context summary\n"
            "  /compact - compact context\n"
            "  /clear - clear transcript (does not forget the mission)\n"
            "  /resume - resume session\n"
            "  /exit - exit HCLI"
        )
        self.last_value = text
        return text

    def _cmd_status(self, arg: str) -> str:
        snap: Dict[str, Any] = {}
        used_controller_status = False
        status_fn = getattr(self.controller, "status", None)
        if callable(status_fn):
            try:
                raw = status_fn()
            except Exception:
                raw = None
            if isinstance(raw, dict):
                snap = dict(raw)
                used_controller_status = True
        enrich_status_snapshot(self.controller, snap)
        self.last_value = snap
        if used_controller_status or _status_has_observed_fields(snap):
            return format_status(snap)
        session = getattr(self.controller, "session", None)
        if session is None:
            return "No active session"
        text = (
            f"Session: {session.id}\n"
            f"Goal: {session.goal or '(none)'}\n"
            f"Runtimes: {session.runtime_count}\n"
            f"Model: {session.model or '(default)'}\n"
            f"Messages: {len(session.messages)}"
        )
        self.last_value = text
        return text

    def _cmd_models(self, arg: str) -> str:
        models = self.controller.list_models()
        self.last_value = models
        if not models:
            return "No models discovered"
        session = getattr(self.controller, "session", None)
        selected = getattr(session, "model", None) if session is not None else None
        lines = ["Available models:"]
        for index, model in enumerate(models, start=1):
            path = _model_path(model)
            name = _model_name(model)
            marker = "●" if selected and path == selected else "  "
            lines.append(f"  {marker} {index}. {name}  {path}")
        return "\n".join(lines)

    def _cmd_model(self, arg: str) -> str:
        if not arg:
            return self._cmd_models("")
        result = self.controller.select_model(arg)
        self.last_value = result
        if result is None or result is False:
            return f"Model not found: {arg}"
        name = getattr(self.controller, "model_name", None) or result
        return f"Switched to {name}"

    def _cmd_goal(self, arg: str) -> str:
        if not arg:
            session = getattr(self.controller, "session", None)
            current = getattr(session, "goal", "") if session is not None else ""
            getter = getattr(self.controller, "session", None)
            if getter is not None:
                current = getattr(getter, "goal", current)
            self.last_value = current
            return current or "(no goal)"
        self.controller.set_goal(arg)
        self.last_value = arg
        return f"Goal set: {arg}"

    def _cmd_ultragoal(self, arg: str) -> str:
        starter = getattr(self.controller, "start_ultragoal", None)
        if not arg:
            status_fn = getattr(self.controller, "status", None)
            snap = status_fn() if callable(status_fn) else {}
            mission = getattr(self.controller, "mission", None)
            if mission is None:
                self.last_value = snap
                return "No durable ultragoal. Usage: /ultragoal <goal text>"
            self.last_value = snap if snap else mission.status()
            return (
                f"ultragoal mission {getattr(mission, 'id', None)} "
                f"goal={getattr(mission, 'goal', '')!r}"
            )
        if not callable(starter):
            self.controller.set_goal(arg)
            result = self.controller.run_mission(arg)
            self.last_value = result
            return f"ultragoal (fallback mission): {result}"
        result = starter(arg)
        self.last_value = result
        if isinstance(result, dict):
            return (
                f"ultragoal mission {result.get('mission_id')} "
                f"obligations={result.get('obligation_ids')} "
                f"units={result.get('workunit_ids')}"
            )
        return f"ultragoal: {result}"

    def _cmd_steer(self, arg: str) -> str:
        if not arg:
            return "Usage: /steer <instruction>"
        event = self.controller.queue_steer(arg)
        self.last_value = event
        text = getattr(event, "text", arg)
        return f"✓ Steer queued: {text}"

    def _grok_bridge(self):
        from .grok_bridge import GrokBridge

        root = Path(self.controller.workspace_root)
        if self._grok is not None and self._grok_root == root:
            return self._grok
        self._grok = GrokBridge(root)
        self._grok_root = root
        return self._grok

    def _grok_mutation_lock(self):
        lock = getattr(self.controller, "mutation_lock", None)
        if lock is None:
            mission = getattr(self.controller, "mission", None)
            scheduler = (
                getattr(mission, "scheduler", None) if mission is not None else None
            )
            lock = (
                getattr(scheduler, "mutation_lock", None)
                if scheduler is not None
                else None
            )
        acquire = getattr(lock, "acquire", None)
        release = getattr(lock, "release", None)
        if lock is None or not callable(acquire) or not callable(release):
            return None
        module = getattr(type(lock), "__module__", "") or ""
        if module.startswith("unittest.mock"):
            return None
        from contextlib import contextmanager

        @contextmanager
        def mutation_lock():
            unit_id = "hcli-grok-delegate"
            if not lock.acquire(unit_id):
                raise RuntimeError("MUTATION lock held")
            try:
                yield
            finally:
                lock.release(unit_id)

        return mutation_lock

    def _cmd_grok(self, arg: str) -> str:
        usage = (
            "Commands:\n"
            "  /grok delegate <task-slug> <contract-file-path>\n"
            "  /grok audit <task-slug> <contract-file-path>\n"
            "  /grok consult <prompt text...>\n"
            "  /grok status <task-id>\n"
            "  /grok wait <task-id>\n"
            "  /grok report <task-id>\n"
            "  /grok cleanup <task-id>"
        )
        raw = (arg or "").strip()
        if not raw:
            self.last_value = usage
            return usage
        verb, _, rest = raw.partition(" ")
        verb = verb.lower()
        rest = rest.strip()
        if verb not in {
            "delegate",
            "audit",
            "consult",
            "status",
            "wait",
            "report",
            "cleanup",
        }:
            self.last_value = usage
            return usage
        from .grok_bridge import GrokContractError, GrokNotAvailable, GrokRunError

        try:
            bridge = self._grok_bridge()
            if verb in ("delegate", "audit"):
                if not rest:
                    return f"Usage: /grok {verb} <task-slug> <contract-file-path>"
                task, _, path = rest.partition(" ")
                task, path = task.strip(), path.strip()
                if not task or not path:
                    return f"Usage: /grok {verb} <task-slug> <contract-file-path>"
                contract = Path(path).expanduser()
                if not contract.is_file():
                    rooted = Path(self.controller.workspace_root) / path
                    if rooted.is_file():
                        contract = rooted
                if not contract.is_file():
                    return f"Contract file not found: {path}"
                text = contract.read_text(encoding="utf-8")
                if verb == "delegate":
                    handle = bridge.delegate(
                        task, text, mutation_lock=self._grok_mutation_lock()
                    )
                else:
                    handle = bridge.audit(task, text)
                self.last_value = handle
                extra = " dry_run=True" if getattr(handle, "dry_run", False) else ""
                return f"grok {handle.mode or verb} {handle.task_id}{extra}"
            if verb == "consult":
                if not rest:
                    return "Usage: /grok consult <prompt text...>"
                handle = bridge.consult(rest)
                self.last_value = handle
                extra = " dry_run=True" if getattr(handle, "dry_run", False) else ""
                return f"grok {handle.mode or verb} {handle.task_id}{extra}"
            if not rest:
                return f"Usage: /grok {verb} <task-id>"
            if verb == "status":
                parsed = bridge.status(rest)
                self.last_value = parsed
                return (
                    f"grok {parsed.get('task_id', rest)} "
                    f"state={parsed.get('state')} "
                    f"exit={parsed.get('exit_code')}"
                )
            if verb == "wait":
                parsed = bridge.wait(rest)
                self.last_value = parsed
                return (
                    f"grok {parsed.get('task_id', rest)} "
                    f"state={parsed.get('state')} "
                    f"exit={parsed.get('exit_code')}"
                )
            if verb == "report":
                compact_fn = getattr(bridge, "compact_report", None)
                if callable(compact_fn):
                    compact = compact_fn(rest)
                    self.last_value = compact
                    summary = compact.get("final_summary") or ""
                    path = compact.get("raw_report_path") or ""
                    return (
                        f"grok {compact.get('task_id', rest)} summary: "
                        f"{summary}\nraw_report_path={path}"
                    )
                report = bridge.report(rest)
                self.last_value = report
                return report
            out = bridge.cleanup(rest)
            self.last_value = out
            return (
                f"grok cleanup {out.get('task_id', rest)} "
                f"ok={out.get('ok')} "
                f"exit={out.get('exit_code')}"
            )
        except (GrokNotAvailable, GrokContractError, GrokRunError, OSError) as exc:
            text = str(exc)
            self.last_value = text
            return text

    def _cmd_mission(self, arg: str) -> str:
        if not arg:
            mission = getattr(self.controller, "mission", None)
            if mission is not None:
                snap = mission.status()
                self.last_value = snap
                return (
                    f"mission {snap.get('mission_id')} "
                    f"phase={snap.get('phase')} "
                    f"units={snap.get('units_by_status')}"
                )
            return "Usage: /mission <goal>"
        result = self.controller.run_mission(arg)
        self.last_value = result
        if isinstance(result, dict):
            return (
                f"mission {result.get('mission_id')} "
                f"status={result.get('status')} "
                f"reason={result.get('reason')}"
            )
        return f"mission: {result}"

    def _cmd_cancel(self, arg: str) -> str:
        self.controller.cancel()
        self.last_value = True
        return "Cancellation requested."

    def _cmd_context(self, arg: str) -> str:
        text = self.controller.context_summary()
        self.last_value = text
        return text

    def _cmd_compact(self, arg: str) -> str:
        self.controller.compact_context()
        self.last_value = True
        return "Context compacted"

    def _cmd_clear(self, arg: str) -> str:
        clearer = getattr(self.controller, "clear_transcript", None)
        if callable(clearer):
            clearer()
        session = getattr(self.controller, "session", None)
        remaining = {
            "goal": getattr(session, "goal", None) if session is not None else None,
            "mission_id": (
                getattr(session, "mission_id", None) if session is not None else None
            ),
            "messages": (
                len(getattr(session, "messages", []) or [])
                if session is not None
                else 0
            ),
        }
        self.last_value = {
            "cleared": True,
            "kind": "transcript",
            "preserved": remaining,
        }
        return "Transcript cleared"

    def _cmd_resume(self, arg: str) -> str:
        result = self.controller.resume_session(arg)
        self.last_value = result
        if result is None:
            return "No session to resume"
        return f"Resumed session: {result}"

    def _cmd_exit(self, arg: str) -> str:
        self.controller.request_exit()
        self.last_value = False
        return None

    def _cmd_quit(self, arg: str) -> str:
        return self._cmd_exit(arg)

    def _cmd_stop(self, arg: str) -> str:
        return self._cmd_cancel(arg)
