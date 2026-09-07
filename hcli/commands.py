from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .command_registry import command_names, help_text
from .paths import find_repo_root

# A per-hour rate from a window much shorter than an hour is sampling noise.
# 4 accepts in 12.4s annualises to 1164/h — the documented lie. Five minutes
# is ~8% of an hour: the smallest window where a handful of accepts cannot
# explode into four-digit rates. Below that, print the raw count and window.
MIN_ACCEPTED_RATE_WINDOW_S = 300.0
GOAL_DISPLAY_CHARS = 72
# /status is one screen of unwrapped lines. `evacuation_checkpoint_error`
# alone would spend a third of the resident line, so cap the event field.
STATUS_LINE_CHARS = 80
# One screen. Four protected tests assert /status never exceeds this.
STATUS_MAX_LINES = 10
EVENT_DISPLAY_CHARS = 20
# /land's default test command when the operator does not type one. Matches
# this repo's own documented test invocation, so a bare /land re-runs exactly
# what a human would run before committing by hand.
# -p no:cacheprovider for the same reason landing sets
# PYTHONDONTWRITEBYTECODE: a verification run that writes .pytest_cache/
# into the repo dirties the tree between the status snapshot and the
# commit, and landing then correctly refuses its own proposal as
# TAMPERED_DURING_VERIFICATION.
LAND_DEFAULT_TEST_COMMAND: Tuple[str, ...] = (
    "python3", "-m", "pytest", "hcli/", "-q", "-p", "no:cacheprovider",
)


def _fmt_unknown(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    return str(value)


def _truncate(value: Any, limit: int) -> str:
    text = _fmt_unknown(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


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


def _fmt_bytes(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(number) < 1024.0:
            return f"{number:.1f}{unit}"
        number /= 1024.0
    return f"{number:.1f}TB"


def format_machine_status(
    snapshot: Dict[str, Any], *, max_lines: int = 2
) -> List[str]:
    """The machine-scoped lines: resident daemon and ModelLake watcher.

    ``state`` is what the supervisor last wrote; ``supervisor`` is what the
    process table says now. They are printed side by side on purpose, so a
    RUNNING record whose pid is gone reads as the contradiction it is.

    ``max_lines=1`` collapses both onto one row. /status is capped at one
    screen, and a mission carrying a no_progress warning already spends that
    row -- two machine lines plus the warning put it one over. Collapsing is
    what gives way there, rather than the cap or the warning.
    """
    snap = snapshot if isinstance(snapshot, dict) else {}
    if max_lines <= 1:
        return _machine_status_one_line(snap)
    lines: List[str] = []
    resident = snap.get("resident")
    if isinstance(resident, dict) and resident.get("ambiguous"):
        # Two live records disagree; picking one and hiding the other would
        # be the silent guess this line exists to refuse.
        lines.append(
            f"Resident ambiguous: {_fmt_unknown(resident.get('root_count'))} "
            "workspaces report state, see --workspace"
        )
    elif isinstance(resident, dict):
        head = (
            f"Resident {_fmt_unknown(resident.get('state'))} "
            f"supervisor={_fmt_unknown(resident.get('supervisor'))} "
            f"cycles={_fmt_unknown(resident.get('cycles'))} event="
        )
        tail = f" age={_fmt_age(resident.get('age_s'))}"
        # The event is the one field with no natural bound, so it is the one
        # that gives way -- visibly elided -- when the rest runs long.
        # ponytail: `stop_reason` is a whole sentence and does not fit a line
        # that is already full. It stays in /status's structured last_value.
        budget = max(8, STATUS_LINE_CHARS - len(head) - len(tail))
        lines.append(
            head
            + _truncate(resident.get("last_event"), min(EVENT_DISPLAY_CHARS, budget))
            + tail
        )
    modellake = snap.get("modellake")
    if isinstance(modellake, dict) and modellake.get("ambiguous"):
        lines.append(
            f"ModelLake ambiguous: {_fmt_unknown(modellake.get('root_count'))} "
            "workspaces report state, see --workspace"
        )
    elif isinstance(modellake, dict):
        lines.append(
            f"ModelLake watcher={_fmt_unknown(modellake.get('watcher'))} "
            f"jobs={_fmt_unknown(modellake.get('jobs'))} "
            f"remaining={_fmt_bytes(modellake.get('remaining_bytes'))} "
            f"sample={_fmt_age(modellake.get('sample_age_s'))}"
        )
    if not lines:
        # Say it rather than printing nothing. "this host is quiet" and
        # "/status cannot see this host" are different facts -- and only the
        # latter means zero workspaces looked real enough to search.
        if snap.get("workspace_roots_seen") == 0:
            lines.append("Machine: no Hawking workspace visible, see --workspace")
        else:
            lines.append("Machine resident=absent modellake=absent")
    return lines


def _machine_status_one_line(snap: Dict[str, Any]) -> List[str]:
    """Both machine facts on one row, within STATUS_LINE_CHARS.

    Drops the fields a reader can get from ``/status``'s structured last_value
    (supervisor pid, sample age) and keeps the two that change decisions: what
    the resident is doing, and whether ModelLake is still pulling.
    """
    resident = snap.get("resident")
    modellake = snap.get("modellake")
    if not isinstance(resident, dict) and not isinstance(modellake, dict):
        if snap.get("workspace_roots_seen") == 0:
            return ["Machine: no Hawking workspace visible, see --workspace"]
        return ["Machine resident=absent modellake=absent"]
    if isinstance(resident, dict) and resident.get("ambiguous"):
        left = f"Machine resident=ambiguous({_fmt_unknown(resident.get('root_count'))})"
    elif isinstance(resident, dict):
        left = (
            f"Machine resident={_fmt_unknown(resident.get('state'))} "
            f"cycles={_fmt_unknown(resident.get('cycles'))}"
        )
    else:
        left = "Machine resident=absent"
    if isinstance(modellake, dict) and modellake.get("ambiguous"):
        right = f" modellake=ambiguous({_fmt_unknown(modellake.get('root_count'))})"
    elif isinstance(modellake, dict):
        right = (
            f" modellake={_fmt_unknown(modellake.get('watcher'))} "
            f"jobs={_fmt_unknown(modellake.get('jobs'))} "
            f"remaining={_fmt_bytes(modellake.get('remaining_bytes'))}"
        )
    else:
        right = " modellake=absent"
    # `remaining` outranks the event string here: it is the field that says
    # whether ModelLake is still competing for this host. The event is included
    # only if a useful amount of it fits -- `_truncate(text, 0)` returns almost
    # the whole string, so a zero budget has to omit the field, not shrink it.
    budget = STATUS_LINE_CHARS - len(left) - len(right) - len(" event=")
    event = (
        _truncate(resident.get("last_event"), min(EVENT_DISPLAY_CHARS, budget))
        if isinstance(resident, dict)
        and not resident.get("ambiguous")
        and budget >= 8
        else ""
    )
    return [left + (f" event={event}" if event else "") + right]


def format_status(snapshot: Dict[str, Any]) -> str:
    """One-screen /status. Unmeasured fields print as unknown, never as 0."""
    snap = snapshot if isinstance(snapshot, dict) else {}
    units = snap.get("units_by_status")
    if not isinstance(units, dict):
        units = None
    occupancy = snap.get("occupancy")
    if not isinstance(occupancy, dict):
        occupancy = None
    runtime = snap.get("runtime")
    if not isinstance(runtime, dict):
        runtime = None
    qwen = snap.get("qwen")
    if not isinstance(qwen, dict):
        qwen = None
    provider_status = runtime or qwen
    provider_label = "Runtime" if runtime is not None else "Qwen"
    grok = snap.get("grok")
    if not isinstance(grok, dict):
        grok = None
    mutation = snap.get("mutation")
    if not isinstance(mutation, dict):
        mutation = None

    mission_id = snap.get("mission_id") or "—"
    phase = snap.get("phase") or "—"
    goal = _truncate_goal(snap.get("goal"))
    bank = snap.get("goal_bank")
    if isinstance(bank, dict) and bank.get("available"):
        bank_count = str(int(bank.get("queued_count") or 0))
    elif isinstance(bank, dict):
        bank_count = "?"
    else:
        bank_count = "0"
    mission_line = f"mission {mission_id}  phase={phase} bank={bank_count}"

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

    if not provider_status:
        qwen_line = f"{provider_label} unknown"
    else:
        health = provider_status.get("health")
        if health and health != "ok":
            queued = provider_status.get("queued")
            queued_s = str(queued) if queued is not None else "unknown"
            head = f"{provider_label} health=down resident=0 active=0 "
            tail = f"queued={queued_s} n_ctx=unknown prompt=unknown tps=unknown"
        else:
            health_s = "ok" if health == "ok" else "unknown"
            head = (
                f"{provider_label} health={health_s} "
                f"resident={_fmt_unknown(provider_status.get('resident'))} "
                f"active={_fmt_unknown(provider_status.get('active_decode'))} "
                f"queued={_fmt_unknown(provider_status.get('queued'))} "
            )
            tail = (
                f"n_ctx={_fmt_unknown(provider_status.get('n_ctx'))} "
                f"prompt={_fmt_unknown(provider_status.get('prompt_tokens'))} "
                f"tps={_fmt_unknown(provider_status.get('tps'))}"
            )
        # n_ctx/prompt_tokens carry real token counts (up to 7 digits on a
        # long-context model) with no natural width cap, unlike the small
        # bounded counters in `head`. Same head/tail budget technique as
        # the resident line below: one flexible chunk truncated to fit,
        # rather than letting the frame wrap.
        budget = max(8, STATUS_LINE_CHARS - len(head))
        qwen_line = head + _truncate(tail, budget)

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

    warning = snap.get("no_progress_warning") or snap.get("watchdog_message")
    has_warning = bool(warning) and warning not in ("(none)", "")
    # Eight fixed rows plus an optional warning; the machine section gets what
    # is left of the one-screen budget. Deciding this here, rather than letting
    # the machine lines splice in unconditionally, is what keeps a warning-
    # carrying mission from rendering STATUS_MAX_LINES + 1.
    lines = [
        mission_line,
        f"Goal: {goal}",
        wu_line,
        qwen_line,
        grok_line,
        *format_machine_status(
            snap, max_lines=STATUS_MAX_LINES - 8 - (1 if has_warning else 0)
        ),
        cpu_line,
        mut_line,
        footer,
    ]
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


def _land_repo_root(controller: Any) -> Path:
    """Where /land's git plumbing runs.

    Uses `session_ledger.discover_repo_root`, not the plain `find_repo_root`
    every other command here uses -- see that function's docstring for why:
    `find_repo_root` silently redirects a tree that is not shaped like this
    repo to the live hawking checkout, which is unsafe for a verb (push,
    branch -f) that actually mutates git state.
    """
    from .session_ledger import discover_repo_root

    root = _workspace_root(controller)
    return discover_repo_root(root) if root is not None else find_repo_root()


# --- machine-scoped observation ------------------------------------------
# /status described only this session's controller, so a second CLI on a busy
# host printed `health=down resident=0` while a resident supervisor and three
# ModelLake downloads were running. These readers open the durable files those
# processes actually write. Anything that cannot be read stays None and prints
# as unknown; nothing here is remembered between calls.

RESIDENT_STATE_REL = (".hcli", "resident", "state.json")
# Producer: tools/odyssey/modellake_watch.py (LOCK_PATH and LOG), both under
# <repo>/workspace/campaign/odyssey. That is the watcher's own literal layout,
# not lab.layout.ODYSSEY_ROOT, which points somewhere else.
MODELLAKE_LOCK_REL = ("workspace", "campaign", "odyssey", ".modellake-watch.lock")
MODELLAKE_LOG_REL = (
    "workspace",
    "campaign",
    "odyssey",
    "downloads",
    "modellake-watch.jsonl",
)
# The watcher log grows without bound (580 MB observed). A watcher_sample row
# is a few KB and lands every 10s, so a bounded tail always contains one.
MODELLAKE_TAIL_BYTES = 128 * 1024


def _status_roots(
    controller: Any, *, repo_root: Optional[Path] = None
) -> List[Path]:
    """Where host state may live: session workspace, resolved workspace, repo.

    A session opened in a scratch directory must still see the machine, so
    this also tries `resolve_workspace()` -- the existing HCLI_WORKSPACE /
    `.hcli` ancestor walk-up, already built for exactly this and
    already wired into mission commands, but never into /status until now.
    ``repo_root`` overrides the live `find_repo_root()` call; production
    never passes it, tests use it to exercise the stamped-install fallback
    (which always resolves to *this* checkout when the test itself runs from
    inside it) without faking `__file__`.
    """
    from .runtime import resolve_workspace

    roots: List[Path] = []
    for candidate in (
        _workspace_root(controller),
        resolve_workspace(),
        repo_root if repo_root is not None else find_repo_root(),
    ):
        if candidate is not None and candidate not in roots:
            roots.append(candidate)
    return roots


# The state roots SUPERWAVE_STATE.md documents for this repo, plus `.git` for
# a fresh checkout that has not written any of them yet. Not a registry --
# the same on-disk evidence `_resident_status`/`_modellake_status` already
# read from -- just checked before trusting a guessed root at all.
# Evidence that is actually DISTINCTIVE to Hawking. The first list accepted a
# bare `.git`, so every git repository on the machine read as a Hawking
# workspace, and `resolve_workspace()` walks up to the filesystem root, so one
# stray `.hcli` left in the system tmp dir did too. On any developer box that
# made `workspace_roots_seen` non-zero with zero real roots present, which fell
# straight back to the misleading "absent" line the search exists to replace.
#
# The roadmap is unmistakable on its own. Runtime state plus receipts together
# is the other honest signature: a scratch dir has neither, a stamped install
# snapshot has no receipts, and a stray `.hcli` has no receipts beside it.
_DECISIVE_MARKERS = (Path("civilization") / "ROADMAP_STATE.json",)
_CORROBORATING_MARKERS = (Path("receipts"), Path(".hcli"))


def _looks_like_hawking_root(path: Optional[Path]) -> bool:
    """Cheap evidence *path* is a real Hawking workspace, not a guess.

    A stamped install snapshot (no vcs metadata, no state dirs) or a plain
    scratch directory fails this. "This host is quiet" and "/status cannot
    see this host" are different facts; this is what tells them apart.
    """
    if path is None:
        return False
    try:
        if any((path / marker).exists() for marker in _DECISIVE_MARKERS):
            return True
        return all((path / marker).is_dir() for marker in _CORROBORATING_MARKERS)
    except OSError:
        return False


def _visible_status_roots(
    controller: Any, *, repo_root: Optional[Path] = None
) -> List[Path]:
    """``_status_roots``, filtered to candidates that look like a real host."""
    return [
        root
        for root in _status_roots(controller, repo_root=repo_root)
        if _looks_like_hawking_root(root)
    ]


def _multiple_hits(roots: List[Path], rel: tuple) -> bool:
    """True when more than one visible root independently has this file.

    Used to tell a defined search order (this root, then that one) apart
    from an actual conflict -- two live records that disagree -- which must
    not be resolved by silently picking the first and dropping the rest.
    """
    return sum(1 for root in roots if root.joinpath(*rel).is_file()) > 1


def _multiple_modellake_hits(roots: List[Path]) -> bool:
    return (
        sum(
            1
            for root in roots
            if root.joinpath(*MODELLAKE_LOCK_REL).is_file()
            or root.joinpath(*MODELLAKE_LOG_REL).is_file()
        )
        > 1
    )


def _pid_liveness(pid: Any, token: Any) -> str:
    """``live`` / ``dead`` / ``none``. Never ``live`` on a pid alone.

    A recycled pid answers ``kill(pid, 0)`` happily, so the recorded start
    token has to match too. An unreadable token is not evidence of death --
    the pid check already passed -- so that case stays ``live``.
    """
    from .resources import pid_is_alive, process_start_token

    try:
        number = int(pid)
    except (TypeError, ValueError):
        return "none"
    if number <= 0:
        return "none"
    if not pid_is_alive(number):
        return "dead"
    if token is None:
        return "live"
    observed = process_start_token(number)
    return "live" if observed is None or str(observed) == str(token) else "dead"


def _resident_status(roots: List[Path]) -> Optional[Dict[str, Any]]:
    """The durable resident record a host supervisor left, or None."""
    for root in roots:
        path = root.joinpath(*RESIDENT_STATE_REL)
        if not path.is_file():
            continue
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(state, dict):
            continue
        updated = _float_or_none(state.get("updated_at"))
        if updated is None:
            try:
                updated = path.stat().st_mtime
            except OSError:
                updated = None
        return {
            "state": state.get("state"),
            "supervisor": _pid_liveness(
                state.get("supervisor_pid"), state.get("supervisor_start_token")
            ),
            "pid": state.get("supervisor_pid"),
            "worker": _pid_liveness(
                state.get("worker_pid"), state.get("worker_start_token")
            ),
            "cycles": state.get("cycles"),
            "last_event": state.get("last_event"),
            "stop_reason": state.get("stop_reason"),
            "age_s": None if updated is None else max(0.0, time.time() - updated),
            "state_path": str(path),
        }
    return None


def _last_watcher_sample(path: Path) -> Optional[Dict[str, Any]]:
    """Last complete ``watcher_sample`` row from a bounded tail of the JSONL."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - MODELLAKE_TAIL_BYTES))
            chunk = handle.read()
    except OSError:
        return None
    for line in reversed(chunk.decode("utf-8", "replace").splitlines()):
        if '"watcher_sample"' not in line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # The first line of the tail is usually a half row. Keep looking.
            continue
        if isinstance(row, dict) and row.get("event") == "watcher_sample":
            return row
    return None


def _sample_age_s(sample: Optional[Dict[str, Any]]) -> Optional[float]:
    stamp = (sample or {}).get("ts")
    if not stamp:
        return None
    try:
        born = datetime.fromisoformat(str(stamp)).timestamp()
    except ValueError:
        return None
    return max(0.0, time.time() - born)


def _modellake_status(roots: List[Path]) -> Optional[Dict[str, Any]]:
    """What the detached ModelLake watcher is doing, or None if it never ran.

    Job count and remaining bytes come from the watcher's own last sample, so
    a dead watcher reports its last observation with a visible sample age
    rather than a fresh-looking number.
    """
    for root in roots:
        lock = root.joinpath(*MODELLAKE_LOCK_REL)
        log = root.joinpath(*MODELLAKE_LOG_REL)
        if not lock.is_file() and not log.is_file():
            continue
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
        except (OSError, UnicodeError, ValueError):
            pid = None
        sample = _last_watcher_sample(log) or {}
        jobs = sample.get("active_jobs")
        return {
            # The watcher holds an flock on this file for its whole life, so
            # the pid in it identifies the incarnation; there is no start
            # token to check, only whether that pid is still there.
            "watcher": _pid_liveness(pid, None),
            "pid": pid,
            "jobs": len(jobs) if isinstance(jobs, list) else None,
            "job_names": jobs if isinstance(jobs, list) else None,
            "remaining_bytes": sample.get("active_remaining_bytes"),
            "free_bytes": sample.get("free_bytes"),
            "sample_age_s": _sample_age_s(sample),
        }
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

    # Machine scope. Measured here rather than defaulted, because the whole
    # point is that these outlive this session's controller. Only roots that
    # look like a real Hawking workspace count -- `workspace_roots_seen`
    # lets the formatter tell "zero visible" apart from "visible but quiet".
    roots = _visible_status_roots(controller)
    snap["workspace_roots_seen"] = len(roots)

    resident = _resident_status(roots)
    if resident is not None and _multiple_hits(roots, RESIDENT_STATE_REL):
        resident = dict(resident, ambiguous=True, root_count=len(roots))
    snap["resident"] = resident

    modellake = _modellake_status(roots)
    if modellake is not None and _multiple_modellake_hits(roots):
        modellake = dict(modellake, ambiguous=True, root_count=len(roots))
    snap["modellake"] = modellake

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
    if snap.get("runtime") or snap.get("qwen") or snap.get("grok"):
        return True
    return False


# Derived, never hand-maintained. This tuple and /help used to be two lists
# and they had already drifted: /tools, /provider and /flash-next were
# advertised by help and missing here. hcli/command_registry.py is the source.
REQUIRED_COMMANDS = command_names()


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


def _mtime(path: Path) -> float:
    """Sort key that survives a file vanishing mid-listing."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


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
        if not line.startswith(("/", "\\")):
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
        text = help_text()
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
        # The machine outlives this session either way, so the thin session
        # summary carries the same host lines the full snapshot does.
        text = "\n".join(
            [
                f"Session: {session.id}",
                f"Goal: {session.goal or '(none)'}",
                f"Runtimes: {session.runtime_count}",
                f"Model: {session.model or '(default)'}",
                f"Messages: {len(session.messages)}",
                *format_machine_status(snap),
            ]
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

    def _cmd_tools(self, arg: str) -> str:
        del arg
        from .tool_registry import default_tool_registry

        workspace = getattr(self.controller, "workspace_root", None) or os.getcwd()
        registry = default_tool_registry(workspace)
        tools = registry.discover()
        self.last_value = tools
        return "\n".join(
            f"{item['name']}  mutation={item['mutation']}"
            for item in tools
        )

    def _cmd_provider(self, arg: str) -> str:
        del arg
        from .providers import profile_from_backend

        pool = getattr(self.controller, "runtime_pool", None)
        rows = []
        for runtime in getattr(pool, "runtimes", []) or []:
            backend = getattr(runtime, "backend", None)
            if backend is not None:
                rows.append(profile_from_backend(backend).to_dict())
        if not rows:
            model = getattr(self.controller, "model", None)
            rows = [{"model_id": model, "provider": "unspawned"}] if model else []
        self.last_value = rows
        return json.dumps(rows, indent=2, sort_keys=True)

    def _cmd_flash_next(self, arg: str) -> str:
        from .flash_next import flash_next_profile

        report = flash_next_profile(arg.strip() or None)
        self.last_value = report
        profile = report["profile"]
        return (
            f"{profile['model_id']} revision="
            f"{profile['artifact']['pinned_revision']} "
            f"qualification={profile['qualification']['status']}"
        )

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

    def _cmd_bank(self, arg: str) -> str:
        """Queue a future goal, or inspect/manage the durable goal bank."""
        usage = (
            "Usage:\n"
            "  /bank <goal> - queue a future goal (auto runner)\n"
            "  /bank mission <goal> - queue a persistent Mission goal\n"
            "  /bank - show queued/running/recent goals\n"
            "  /bank drop <id|position> - remove one waiting goal\n"
            "  /bank clear - remove all waiting goals"
        )
        raw = (arg or "").strip()
        if not raw:
            snapshot = self.controller.goal_bank_snapshot()
            self.last_value = snapshot
            if not snapshot.get("available", False):
                return f"Goal bank unavailable: {snapshot.get('reason', 'unknown error')}"
            queued = snapshot.get("queued") or []
            running = snapshot.get("running") or []
            recent = snapshot.get("recent") or []
            lines = [
                f"Goal bank: queued={snapshot.get('queued_count', 0)} "
                f"running={snapshot.get('running_count', 0)}"
            ]
            if queued:
                lines.append("Queued:")
                lines.extend(
                    f"  {index}. {item.get('id')} [{item.get('mode', 'auto')}] "
                    f"{_truncate(item.get('goal'), 100)}"
                    for index, item in enumerate(queued, start=1)
                )
            if running:
                lines.append("Running:")
                lines.extend(
                    f"  {item.get('id')} [{item.get('mode', 'auto')}] "
                    f"{_truncate(item.get('goal'), 100)}"
                    for item in running
                )
            if recent:
                lines.append("Recent:")
                lines.extend(
                    f"  {item.get('id')} {item.get('status')} "
                    f"{_truncate(item.get('goal'), 100)}"
                    for item in recent
                )
            if not queued and not running and not recent:
                lines.append("  (empty)")
            return "\n".join(lines)

        verb, _, rest = raw.partition(" ")
        verb = verb.lower()
        rest = rest.strip()
        try:
            if verb in {"drop", "remove"}:
                if not rest:
                    return "Usage: /bank drop <id|position>"
                item = self.controller.drop_banked_goal(rest)
                self.last_value = item
                return (
                    f"Dropped banked goal {item.get('id')}"
                    if item is not None
                    else f"No queued goal matched: {rest}"
                )
            if verb == "clear":
                removed = self.controller.clear_banked_goals()
                self.last_value = {"removed": removed}
                return f"Cleared {removed} banked goal(s)"
            mode = "auto"
            goal = raw
            if verb == "mission":
                mode = "mission"
                goal = rest
            if not goal:
                return usage
            item = self.controller.bank_goal(goal, mode=mode)
            self.last_value = item
            snapshot = self.controller.goal_bank_snapshot()
            position = next(
                (
                    index
                    for index, queued in enumerate(snapshot.get("queued") or [], start=1)
                    if queued.get("id") == item.get("id")
                ),
                snapshot.get("queued_count", "?"),
            )
            return (
                f"Banked {item.get('id')} position={position} mode={mode}: "
                f"{_truncate(goal, 160)}"
            )
        except (ValueError, RuntimeError) as exc:
            self.last_value = str(exc)
            return f"Goal bank error: {exc}"

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

    def _paste_cache(self):
        from .paste_cache import PasteCache

        return PasteCache(_workspace_root(self.controller))

    def _cmd_context(self, arg: str) -> str:
        """Context summary, plus the disposable paste cache.

        ``drop`` and ``clear-pastes`` reach PasteCache and nothing else. Every
        id it accepts must match its ``paste_<stamp>_<sha8>`` pattern AND
        resolve to a direct child of ``<root>/.hcli/pastes``, so there is no
        spelling of this command that can delete a receipt, mission state, or
        evidence -- those live in sibling directories a paste id cannot name.
        """
        usage = (
            "Commands:\n"
            "  /context - context summary\n"
            "  /context memory - show the bounded prior-knowledge index\n"
            "  /context list - list cached pastes, newest first\n"
            "  /context drop <paste-id> - delete one cached paste\n"
            "  /context clear-pastes - delete every cached paste"
        )
        verb, _, rest = (arg or "").strip().partition(" ")
        verb, rest = verb.lower(), rest.strip()

        if not verb:
            text = self.controller.context_summary()
            try:
                text += f" pastes={len(self._paste_cache().list())}"
            except OSError:
                text += " pastes=unknown"
            self.last_value = text
            return text

        if verb == "memory":
            getter = getattr(self.controller, "prior_knowledge_snapshot", None)
            snapshot = getter() if callable(getter) else {}
            self.last_value = snapshot
            return json.dumps(snapshot, indent=2, sort_keys=True, default=str)

        if verb == "list":
            refs = self._paste_cache().list()
            self.last_value = [ref.to_dict() for ref in refs]
            if not refs:
                return "No cached pastes"
            return "\n".join(
                ["Cached pastes (newest first):"]
                + [f"  {ref.context_ref()}" for ref in refs]
            )

        if verb == "drop":
            if not rest:
                return "Usage: /context drop <paste-id>"
            try:
                dropped = self._paste_cache().drop(rest)
            except ValueError as exc:
                self.last_value = str(exc)
                return str(exc)
            self.last_value = {"dropped": [rest] if dropped else []}
            return f"Dropped paste {rest}" if dropped else f"No such paste: {rest}"

        if verb == "clear-pastes":
            # keep_last=0 is PasteCache's own way to say "empty it"; prune
            # refuses to run without an explicit policy.
            dropped = self._paste_cache().prune(keep_last=0)
            self.last_value = {"dropped": dropped}
            return f"Dropped {len(dropped)} paste(s)"

        self.last_value = usage
        return usage

    def _cmd_processes(self, arg: str) -> str:
        """Role, RSS and stop-safety for every live Hawking process.

        The operator complaint this answers is that Activity Monitor shows
        several entries called `Python` and nothing distinguishes the resident
        supervisor from a model download. Classification is by argv because the
        executable name is identical across all of them.
        """
        from .processes import render

        return render(
            width=STATUS_LINE_CHARS,
            workspace=_workspace_root(self.controller),
        )

    def _cmd_receipts(self, arg: str) -> str:
        """Durable run receipts, newest first. hcli.engine writes one per goal."""
        limit = 10
        if arg.strip():
            try:
                limit = max(1, int(arg.strip()))
            except ValueError:
                return "Usage: /receipts [count]"
        root = _workspace_root(self.controller)
        directory = None if root is None else root / ".hcli" / "receipts"
        if directory is None or not directory.is_dir():
            self.last_value = []
            return "No receipts"
        now = time.time()
        rows = []
        for path in sorted(directory.glob("*.json"), key=_mtime, reverse=True)[:limit]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                data = {}
            if not isinstance(data, dict):
                data = {}
            mtime = _mtime(path)
            rows.append(
                {
                    "path": str(path),
                    "goal_id": data.get("goal_id") or path.stem,
                    "status": data.get("status"),
                    "kind": data.get("kind"),
                    "age_s": None if mtime <= 0 else max(0.0, now - mtime),
                }
            )
        self.last_value = rows
        if not rows:
            return "No receipts"
        return "\n".join(
            ["Receipts (newest first):"]
            + [
                f"  {row['goal_id']} status={_fmt_unknown(row['status'])} "
                f"kind={_fmt_unknown(row['kind'])} age={_fmt_age(row['age_s'])}"
                for row in rows
            ]
        )

    def _cmd_compact(self, arg: str) -> str:
        memory = self.controller.compact_context()
        self.last_value = memory
        if isinstance(memory, dict):
            staging = memory.get("staging") or {}
            staged = (staging.get("staged") or {}).get("count", 0)
            unstaged = (staging.get("unstaged") or {}).get("count", 0)
            generation = memory.get("generation", "?")
            return (
                f"Context compacted checkpoint#{generation} "
                f"messages_kept=4 staged={staged} unstaged={unstaged}"
            )
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

    def _cmd_land(self, arg: str) -> str:
        """/land commit[s] accumulated work; push and merge are separate,
        explicitly-typed steps -- see `_land_push`/`_land_merge` for why.
        `/land` alone (or `/land <message>`) commits everything currently
        dirty through `hcli.landing`, which re-verifies and re-runs the test
        command itself; this method never touches git directly for that
        path, only for the push/merge verbs, neither of which is a commit.
        """
        verb, _, rest = (arg or "").strip().partition(" ")
        if verb.lower() == "push":
            return self._land_push()
        if verb.lower() == "merge":
            target = rest.strip()
            if not target:
                text = "Usage: /land merge <branch>"
                self.last_value = text
                return text
            return self._land_merge(target)
        return self._land_commit((arg or "").strip())

    def _land_commit(self, message: str) -> str:
        from .landing import propose_landing
        from .session_ledger import SessionLedger, changed_paths

        repo_root = _land_repo_root(self.controller)
        ledger = SessionLedger(repo_root, repo_root=repo_root)
        snap = ledger.snapshot()
        paths = changed_paths(repo_root)
        if not paths:
            text = "Nothing to land: working tree is clean."
            self.last_value = {"landed": False, "reason": "EMPTY_DIFF"}
            return text
        branch = self._verifier_run(repo_root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        if not message:
            message = (
                f"session checkpoint: {snap['files_changed']} file(s) changed, "
                f"+{snap['insertions']}/-{snap['deletions']} lines, "
                f"{snap['untracked']} untracked"
            )
        result = propose_landing(
            repo_root,
            branch=branch,
            allowed_paths=paths,
            test_command=list(LAND_DEFAULT_TEST_COMMAND),
            message=message,
        )
        self.last_value = result
        if result.get("landed"):
            sha = str(result.get("commit_sha") or "")[:12]
            count = len(result.get("changed_paths") or [])
            return f"Landed {sha}: {count} file(s) committed. Push with /land push."
        detail = f": {result.get('detail')}" if result.get("detail") else ""
        return f"Not landed ({result.get('reason')}){detail}"

    def _land_push(self) -> str:
        """The explicit, operator-typed push. `hcli.landing` never pushes on
        its own -- a local commit is recoverable by inspection; a push
        changes what a remote and every other clone sees, so it happens only
        when a human types this verb, never from an automatic prompt and
        never from the resident (which never imports this method at all)."""
        repo_root = _land_repo_root(self.controller)
        result = self._verifier_run(repo_root, "push", timeout=60.0)
        self.last_value = {
            "pushed": result.returncode == 0,
            "stdout": result.stdout, "stderr": result.stderr,
        }
        if result.returncode == 0:
            return "Pushed."
        return f"Push failed: {(result.stderr or result.stdout).strip()}"

    def _land_merge(self, target: str) -> str:
        """Fast-forward `target` to HEAD, ONLY when `target` is a strict
        ancestor of HEAD, and WITHOUT ever checking `target` out -- this
        working tree hosts a live daemon whose worker respawns from these
        files, so swapping them mid-cycle is a production break. Moving the
        branch pointer with `git branch -f` advances `target` without
        touching a single file on disk.
        """
        repo_root = _land_repo_root(self.controller)
        verify = self._verifier_run(repo_root, "rev-parse", "--verify", "--quiet", target)
        if verify.returncode != 0:
            text = f"Refused: no such branch {target!r}"
            self.last_value = {"merged": False, "reason": "NO_SUCH_BRANCH", "target": target}
            return text
        counts = self._verifier_run(repo_root, "rev-list", "--left-right", "--count", f"{target}...HEAD")
        parts = counts.stdout.split() if counts.returncode == 0 else []
        if counts.returncode != 0 or len(parts) != 2 or not all(p.isdigit() for p in parts):
            text = f"Refused: could not compare {target!r} with HEAD: {counts.stderr.strip()}"
            self.last_value = {"merged": False, "reason": "COMPARE_FAILED", "target": target}
            return text
        only_target, only_head = int(parts[0]), int(parts[1])
        if only_target > 0:
            text = (
                f"Refused: {target!r} is not a strict ancestor of HEAD "
                f"({only_target} commit(s) on {target!r} not on HEAD); fast-forward only."
            )
            self.last_value = {"merged": False, "reason": "NOT_FAST_FORWARD", "target": target}
            return text
        if only_head == 0:
            text = f"{target!r} is already up to date with HEAD."
            self.last_value = {"merged": False, "reason": "ALREADY_UP_TO_DATE", "target": target}
            return text
        move = self._verifier_run(repo_root, "branch", "-f", target, "HEAD")
        if move.returncode != 0:
            text = f"Refused: git branch -f failed: {move.stderr.strip()}"
            self.last_value = {"merged": False, "reason": "BRANCH_UPDATE_FAILED", "target": target}
            return text
        self.last_value = {"merged": True, "target": target, "advanced_by": only_head}
        return f"Fast-forwarded {target!r} by {only_head} commit(s). Working tree untouched."

    @staticmethod
    def _verifier_run(repo_root: Path, *args: str, timeout: float = 30.0):
        """Read-only-shaped git plumbing for /land push and /land merge, via
        the same subprocess wrapper `hcli.landing.IntegrationVerifier` uses
        for its own checks -- one way this module talks to git, not two."""
        from .landing import IntegrationVerifier

        return IntegrationVerifier()._run(repo_root, *args, timeout=timeout)


# `/flash-next` carries a dash and no Python identifier can. The dispatcher
# looks up `_cmd_` + the name, so /help advertised a command that answered
# "Unknown command: /flash-next". Bind the dashed attribute instead of
# teaching the lookup a spelling rule -- a direct
# `getattr(handler, "_cmd_" + name[1:])`, which is how the ingress tests
# probe wiring, then finds it too.
setattr(CommandHandler, "_cmd_flash-next", CommandHandler._cmd_flash_next)
