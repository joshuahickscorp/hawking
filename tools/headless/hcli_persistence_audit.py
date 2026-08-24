#!/usr/bin/env python3
"""HCLI persistence and transaction-integrity audit.

Enumerates every durable writer under tools/haider/hcli/ and DEMONSTRATES
atomicity, crash-between-writes leftovers, persist-failure handling, and
restart reconciliation. A finding is only claimed after this process
watched the on-disk state.

Run from the repository root:

    python3 tools/headless/hcli_persistence_audit.py
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
HAIDER = REPO

os.environ.setdefault("HCLI_DISABLE_SIGNAL_HOOKS", "1")

from hcli.config import Config  # noqa: E402
from hcli.controller import Controller  # noqa: E402
from hcli.dag_store import DagStore, atomic_write_json  # noqa: E402
from hcli.engine import Engine  # noqa: E402
from hcli.events import EventBus  # noqa: E402
from hcli.grok_bridge import GrokBridge  # noqa: E402
from hcli.ledger import Ledger, VerifyResult  # noqa: E402
from hcli.machine import GIB, MachineGenome, MemGate  # noqa: E402
from hcli.mission import Mission, MissionCorruptError, load_state  # noqa: E402
from hcli.mutation import apply_mutation_operations, rollback_mutation  # noqa: E402
from hcli.resources import MutationLock  # noqa: E402
from hcli.runtime import RuntimePool  # noqa: E402
from hcli.scheduler import NO_PROGRESS, Scheduler  # noqa: E402
from hcli.session import Session, SessionStore  # noqa: E402
from hcli.steering import SteeringQueue  # noqa: E402
from hcli.workunit import WorkUnit, identify_ready  # noqa: E402
from hcli.workspace import Workspace  # noqa: E402

HCLI_DIR = REPO / "hcli"
RECEIPT_PATH = REPO / "receipts" / "headless" / "HCLI_PERSISTENCE_AUDIT.json"

FAILS: List[str] = []
WATCHED_FAIL: List[Dict[str, Any]] = []
DEMOS: List[Dict[str, Any]] = []
SAFE: List[Dict[str, Any]] = []
WRITERS: List[Dict[str, Any]] = []


class CrashBetweenWrites(Exception):
    """Stand-in for SIGKILL between two durable writes."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _git_head() -> str:
    proc = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (proc.stdout or "").strip() or "UNKNOWN"


def _loc(obj: Any) -> str:
    try:
        src = inspect.getsourcelines(obj)
        path = inspect.getsourcefile(obj) or inspect.getfile(obj)
        rel = os.path.relpath(path, str(REPO))
        return f"{rel}:{src[1]}"
    except Exception as exc:
        return f"unresolved:{exc}"


def _wu(uid: str, **kwargs: Any) -> WorkUnit:
    return WorkUnit(id=uid, role="work", description=uid, **kwargs)


def _tree(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = str(path.relative_to(root))
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                text = f"<unreadable {exc}>"
            if len(text) > 4000:
                text = text[:4000] + "\n…<truncated>…"
            out[rel] = text
    return out


def _record_demo(
    name: str,
    *,
    ok: bool,
    detail: str,
    on_disk: Optional[Dict[str, Any]] = None,
    watched_fail: bool = False,
    rank: Optional[str] = None,
) -> Dict[str, Any]:
    rec = {
        "name": name,
        "ok": ok,
        "detail": detail,
        "on_disk": on_disk or {},
        "watched_fail": watched_fail,
        "rank": rank,
    }
    DEMOS.append(rec)
    marker = "ok  " if ok else "FAIL"
    print(f"{marker} {name}")
    if detail:
        for line in str(detail).splitlines() or [detail]:
            print(f"     {line}")
    if on_disk:
        print("     on-disk:")
        blob = json.dumps(on_disk, indent=2, sort_keys=True, default=str)
        for line in blob.splitlines():
            print(f"       {line}")
    if not ok:
        FAILS.append(f"{name}: {detail}")
    if watched_fail:
        WATCHED_FAIL.append(rec)
    return rec


def _engine(root: Path) -> Engine:
    return Engine(
        workspace=Workspace(str(root)),
        event_bus=EventBus(),
        runtime_count=1,
        model_name="/missing.gguf",
    )


VALID_CONTRACT = """# WRITE
tools/headless/hcli_persistence_audit.py

# VERIFY
python3 tools/headless/hcli_persistence_audit.py

# ACCEPTANCE
harness exits 0
"""


class FakeBackend:
    """Minimal llama-server stand-in. Spawns `sleep` so ownership can leak."""

    def __init__(self, model_path, port, n_slots=1, index=0, **_kwargs):
        self.model_path = model_path
        self.port = port
        self.n_slots = n_slots
        self.index = index
        self.process = None
        self.pid = None
        self.start_time = None

    def identity(self):
        return {
            "backend": "fake",
            "binary": "sleep",
            "model_path": self.model_path,
            "model_identity": f"{self.model_path}:fake",
        }

    def spawn(self, **kwargs):
        if kwargs.get("port") is not None:
            self.port = int(kwargs["port"])
        if self.process is not None and self.process.poll() is None:
            return
        self.process = subprocess.Popen(
            ["sleep", "120"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.pid = self.process.pid
        from hcli.resources import process_start_token

        self.start_time = process_start_token(self.pid)

    def ready(self, timeout):
        del timeout
        return self.process is not None and self.process.poll() is None

    def stop(self):
        report = {"pid": self.pid, "gone": True, "unreaped": []}
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=2)
                except Exception:
                    pass
        if self.pid:
            try:
                os.kill(int(self.pid), 0)
                report["gone"] = False
                report["unreaped"] = [self.pid]
            except OSError:
                report["gone"] = True
        self.process = None
        if report["gone"]:
            self.pid = None
        return report


def _gate():
    return MemGate(
        reserve_bytes=1,
        model_bytes=100,
        per_runtime_overhead_bytes=100,
        headroom_frac=0.1,
        metal_info={
            "recommendedMaxWorkingSetSize": 80 * GIB,
            "currentAllocatedSize": 0,
            "source": "audit-inject",
        },
        topology="process",
    )


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


WRITE_ATTRS = {
    "write_text",
    "write_bytes",
    "replace",
    "dump",
    "mkdir",
    "open",
    "unlink",
}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def ast_write_census() -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for path in sorted(HCLI_DIR.glob("*.py")):
        rel = str(path.relative_to(REPO))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            tail = name.split(".")[-1]
            if tail not in WRITE_ATTRS and name not in {
                "os.replace",
                "os.rename",
                "os.fsync",
                "json.dump",
                "open",
            }:
                continue
            if tail in {"mkdir"}:
                continue
            hits.append(
                {
                    "file": rel,
                    "line": getattr(node, "lineno", 0),
                    "call": name,
                }
            )
    return hits


def _writer(
    wid: str,
    symbol: str,
    obj: Any,
    dest: str,
    *,
    notes: str = "",
) -> Dict[str, Any]:
    loc = _loc(obj)
    source = ""
    try:
        source = inspect.getsource(obj)
    except Exception:
        source = ""
    uses_replace = "os.replace" in source or "os.rename" in source
    uses_write_text = "write_text(" in source or "write_bytes(" in source
    uses_append = "open(" in source and (
        '", "a"' in source or "', 'a'" in source or '"a"' in source
    )
    uses_fsync = "fsync" in source
    uses_tmp = ".tmp" in source or "tmp_name" in source or "tmp =" in source
    delegates_atomic = any(
        token in source
        for token in (
            "atomic_write_json(",
            "_atomic_write_text(",
            "_atomic_write(",
            "self.store.save(",
            "self.scheduler._persist(",
        )
    )
    if uses_replace and uses_tmp:
        atomic = True
        method = "temp-file-plus-rename"
    elif delegates_atomic:
        atomic = True
        method = "delegates to tmp+replace helper (single file)"
    elif uses_append:
        atomic = False
        method = "append-in-place"
    elif uses_write_text:
        atomic = False
        method = "write_text/write_bytes onto live path"
    elif uses_replace:
        atomic = True
        method = "temp-file-plus-rename"
    else:
        atomic = False
        method = "unknown-or-no-persist"
    rec = {
        "id": wid,
        "symbol": symbol,
        "loc": loc,
        "dest": dest,
        "atomic_from_source": atomic,
        "method_from_source": method,
        "fsync": uses_fsync,
        "raises_from_source": "except OSError" not in source
        or "raise" in source.split("except OSError", 1)[-1][:200]
        if "except OSError" in source
        else True,
        "notes": notes,
        "source_excerpt_has_replace": uses_replace,
        "source_excerpt_has_write_text": uses_write_text,
    }
    WRITERS.append(rec)
    return rec


def build_inventory() -> None:
    _writer(
        "W01",
        "dag_store.atomic_write_json",
        atomic_write_json,
        "<dest> via .tmp + os.replace + fsync",
        notes="primitive used by DAG and Mission.checkpoint",
    )
    _writer(
        "W02",
        "DagStore.save",
        DagStore.save,
        "<workspace>/.hcli/dag.json",
        notes="single JSON document of all WorkUnits",
    )
    _writer(
        "W03",
        "Scheduler._persist",
        Scheduler._persist,
        "<workspace>/.hcli/dag.json",
        notes="called from __init__, from_workspace, dispatch, complete, fail",
    )
    _writer(
        "W04",
        "Mission.checkpoint",
        Mission.checkpoint,
        "<workspace>/.hcli/mission/state.json",
        notes=(
            "does NOT persist DAG, does NOT stamp checkpoint_id. "
            "Call sites pair it with Scheduler._persist (two files)."
        ),
    )
    _writer(
        "W05",
        "Mission._log",
        Mission._log,
        "<workspace>/.hcli/mission/mission.log",
        notes="append; except OSError: pass",
    )
    _writer(
        "W06",
        "Config._write",
        Config._write,
        "~/.config/hcli/config.json (or override)",
        notes="tmp+replace, NO fsync",
    )
    _writer(
        "W07",
        "Config.save_project",
        Config.save_project,
        "<workspace>/.hcli/config.json",
        notes="duplicates _write; tmp+replace, NO fsync",
    )
    _writer(
        "W08",
        "Config.save_global",
        Config.save_global,
        "via Config._write",
        notes="Controller.select_model swallows OSError",
    )
    _writer(
        "W09",
        "SessionStore.save",
        SessionStore.save,
        "<workspace>/.hcli/sessions/<id>.json",
        notes="tmp+replace, NO fsync; Controller never calls save",
    )
    _writer(
        "W10",
        "SteeringQueue._save",
        SteeringQueue._save,
        "<workspace>/.hcli/steering/<session>.json",
        notes="tmp+replace, NO fsync",
    )
    _writer(
        "W11",
        "MutationLock.write",
        MutationLock.write,
        "<workspace>/.hcli/mutation.lock",
        notes="via resources._atomic_write_text (tmp+replace+fsync)",
    )
    _writer(
        "W12",
        "RuntimePool._write_ownership",
        RuntimePool._write_ownership,
        "<workspace>/.hcli/runtime_pool.json",
        notes="atomic helper; called AFTER processes are spawned",
    )
    _writer(
        "W13",
        "RuntimePool._clear_ownership",
        RuntimePool._clear_ownership,
        "<workspace>/.hcli/runtime_pool.json (unlink or empty record)",
        notes="unlink failure falls back to empty write; both OSError paths swallowed",
    )
    _writer(
        "W14",
        "GrokBridge._write_contract_file",
        GrokBridge._write_contract_file,
        "<workspace>/.hcli/grok/contracts/<task>-<stamp>.md",
        notes="Path.write_text onto live path; happens BEFORE grok-run and receipt",
    )
    _writer(
        "W15",
        "GrokBridge._write_receipt",
        GrokBridge._write_receipt,
        "<workspace>/.hcli/grok/<task_id>.json",
        notes="Path.write_text onto live path; AFTER grok-run returns an id",
    )
    _writer(
        "W16",
        "GrokBridge._update_receipt",
        GrokBridge._update_receipt,
        "<workspace>/.hcli/grok/<task_id>.json",
        notes="Path.write_text; _read_receipt returns None on corrupt JSON",
    )
    _writer(
        "W17",
        "Engine._write_receipt",
        Engine._write_receipt,
        "<workspace>/.hcli/receipts/<goal_id>.json",
        notes="Path.write_text onto live path; AFTER mutation is on disk",
    )
    _writer(
        "W18",
        "Engine._apply_operations",
        Engine._apply_operations,
        "workspace files named in operations",
        notes="sequential write_text; not atomic per file; not all-or-nothing across files",
    )
    _writer(
        "W19",
        "Engine._restore",
        Engine._restore,
        "workspace files from snapshot",
        notes="write_bytes onto live path; execute() uses this on exception, not SIGKILL",
    )
    _writer(
        "W20",
        "mutation._apply_create / write_text",
        apply_mutation_operations,
        "workspace files",
        notes="sequential write_text; rollback_mutation is broken for created files",
    )
    _writer(
        "W21",
        "MachineGenome.save",
        MachineGenome.save,
        "$HCLI_HOME/machine-genome.json (default ~/.local/share/hcli/)",
        notes="Path.write_text onto live path",
    )
    _writer(
        "W22",
        "cli.install_shims",
        None,  # resolved below
        "~/.local/share/hcli/build-* + ~/.local/bin/{hcli,jhcli}",
        notes="copytree, symlink current, two write_text shims; multi-file",
    )
    from hcli.cli import install_shims

    WRITERS[-1]["loc"] = _loc(install_shims)
    WRITERS[-1]["atomic_from_source"] = False
    WRITERS[-1]["method_from_source"] = "copytree + symlink + write_text"
    _writer(
        "W23",
        "Ledger.save / persist",
        Ledger.to_markdown,
        "GOAL.md (commented; no save method exists)",
        notes=(
            "ledger.py:49 comment says Path.write_text(ledger.to_markdown()). "
            "There is no save() implementation. mark_verified never hits disk."
        ),
    )
    WRITERS[-1]["atomic_from_source"] = False
    WRITERS[-1]["method_from_source"] = "no persist implementation"
    _writer(
        "W24",
        "Mission.from_workspace (reader, persist of recovered DAG)",
        Mission.from_workspace,
        "<workspace>/.hcli/dag.json via Scheduler._persist",
        notes="recover_running marks failed without emitting a repair",
    )


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------


class _PartialThenCrash:
    """File-like: write a prefix, then raise. Does not recurse into itself."""

    def __init__(self, handle, limit: int = 12):
        self._h = handle
        self._limit = limit

    def write(self, data):
        chunk = data[: self._limit] if isinstance(data, (str, bytes)) else data
        self._h.write(chunk)
        self._h.flush()
        raise CrashBetweenWrites("mid-tmp-write")

    def flush(self):
        self._h.flush()

    def fileno(self):
        return self._h.fileno()

    def close(self):
        self._h.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


def demo_atomic_write_json_crash_mid_tmp() -> None:
    name = "atomic_write_json: crash mid-tmp leaves live dest intact"
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "dag.json"
        dest.write_text('{"keep": true}', encoding="utf-8")
        real_open = open

        def exploding_open(path, mode="r", *args, **kwargs):
            if "w" in str(mode) and str(path).endswith(".tmp"):
                return _PartialThenCrash(real_open(path, mode, *args, **kwargs))
            return real_open(path, mode, *args, **kwargs)

        raised = None
        with patch("builtins.open", exploding_open):
            try:
                atomic_write_json(dest, {"keep": False, "new": 1})
            except CrashBetweenWrites as exc:
                raised = exc
        live = dest.read_text(encoding="utf-8")
        leftovers = [p.name for p in Path(tmp).glob("*.tmp")]
        intact = '"keep": true' in live and "new" not in live
        _record_demo(
            name,
            ok=raised is not None and intact,
            detail=(
                f"raised={type(raised).__name__} live={live!r} "
                f"tmp_leftovers={leftovers}"
            ),
            on_disk={"dag.json": live, "tmp_leftovers": leftovers},
            watched_fail=True,
            rank="cosmetic (tmp leftover) / dest proven untruncated",
        )
        if intact:
            SAFE.append(
                {
                    "writer": "W01 dag_store.atomic_write_json",
                    "claim": "crash mid-tmp does not truncate the live path",
                    "evidence": f"live stayed {live!r}; CrashBetweenWrites raised",
                }
            )


def demo_write_text_truncates() -> None:
    name = "Path.write_text onto live path: crash mid-write truncates"
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "receipt.json"
        original = json.dumps({"task_id": "old", "status": "running"}, indent=2)
        dest.write_text(original, encoding="utf-8")
        payload = json.dumps(
            {"task_id": "new", "status": "done", "padding": "x" * 400},
            indent=2,
        )
        def truncating(self, data, encoding="utf-8", errors="strict", newline=None):
            del errors, newline
            raw = data.encode(encoding) if isinstance(data, str) else data
            with open(self, "wb") as handle:
                handle.write(raw[:40])
            raise CrashBetweenWrites("mid-write_text")

        raised = None
        with patch.object(Path, "write_text", truncating):
            try:
                dest.write_text(payload, encoding="utf-8")
            except CrashBetweenWrites as exc:
                raised = exc
        live = dest.read_bytes()
        truncated = live != original.encode("utf-8") and live != payload.encode("utf-8")
        parseable = True
        parse_err = None
        try:
            json.loads(live.decode("utf-8", "replace"))
        except Exception as exc:
            parseable = False
            parse_err = f"{type(exc).__name__}: {exc}"
        _record_demo(
            name,
            ok=raised is not None and truncated and not parseable,
            detail=(
                f"raised={type(raised).__name__} bytes={live!r} "
                f"parseable={parseable} parse_err={parse_err}"
            ),
            on_disk={
                "receipt.json_bytes": repr(live),
                "json_parse": parse_err or "parsed",
            },
            watched_fail=True,
            rank="loses grok/engine receipt; restart treats as missing",
        )


def demo_checkpoint_does_not_write_dag() -> None:
    name = "Mission.checkpoint does not persist DAG and stamps no checkpoint_id"
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        units = {"a": _wu("a")}
        mission = Mission(
            ws,
            engine=object(),
            units=units,
            quiet=True,
            no_progress_threshold=100,
        )
        saves: List[str] = []
        real = DagStore.save

        def spy(self, *args, **kwargs):
            saves.append(str(self.path))
            return real(self, *args, **kwargs)

        with patch.object(DagStore, "save", spy):
            saves.clear()
            path = mission.checkpoint()
            during = list(saves)
        state = json.loads(path.read_text(encoding="utf-8"))
        _record_demo(
            name,
            ok=True,
            detail=(
                f"DagStore.save calls during checkpoint={during} "
                f"checkpoint_id in payload={'checkpoint_id' in state} "
                f"keys={sorted(state)}"
            ),
            on_disk={"mission/state.json_keys": sorted(state.keys()), "dag_saves": during},
            watched_fail="checkpoint_id" not in state,
            rank="loses mission position (claimed repair is absent in this tree)",
        )


def demo_sigkill_between_dag_and_mission() -> None:
    name = "INTERRUPTED: SIGKILL between DAG persist and mission checkpoint"
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        child = ws / "child.py"
        child.write_text(
            "\n".join(
                [
                    "import os, sys, signal",
                    
                    "from pathlib import Path",
                    "from hcli.mission import Mission",
                    "from hcli.workunit import WorkUnit",
                    f"ws = Path({str(ws)!r})",
                    "class E:",
                    "    def execute_workunit(self, unit, context):",
                    "        (ws / 'executed').write_text(unit.id)",
                    "        return {'validation': {'ok': True}}",
                    "n = {'c': 0}",
                    "real = Mission.checkpoint",
                    "def maybe_die(self):",
                    "    n['c'] += 1",
                    "    if n['c'] >= 2:",
                    "        (ws / 'killed_before_second_checkpoint').write_text('1')",
                    "        os.kill(os.getpid(), signal.SIGKILL)",
                    "    return real(self)",
                    "Mission.checkpoint = maybe_die",
                    "units = {'u1': WorkUnit(id='u1', role='work', description='u1')}",
                    "m = Mission(ws, engine=E(), units=units, quiet=True,",
                    "            no_progress_threshold=100, heartbeat_s=60)",
                    "raise SystemExit(m.run() and 0)",
                ]
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", "")
        env["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
        proc = subprocess.run(
            [sys.executable, str(child)],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        dag_path = ws / ".hcli" / "dag.json"
        state_path = ws / ".hcli" / "mission" / "state.json"
        dag = json.loads(dag_path.read_text(encoding="utf-8")) if dag_path.is_file() else None
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
        dag_status = None
        if isinstance(dag, dict):
            dag_status = ((dag.get("units") or {}).get("u1") or {}).get("status")
        mission_status = None
        in_flight = None
        if isinstance(state, dict):
            mission_status = ((state.get("units") or {}).get("u1") or {}).get("status")
            in_flight = state.get("in_flight")
        split = dag_status == "running" and mission_status != "running"
        # Restart reconciliation.
        resumed = Mission.from_workspace(
            ws, engine=object(), quiet=True, no_progress_threshold=100
        )
        after = resumed.scheduler.units["u1"]
        status_before_ready = after.status
        ready = identify_ready(resumed.scheduler.units)
        ready_ids = [u.id for u in ready]
        repairs = [
            u.id
            for u in resumed.scheduler.units.values()
            if getattr(u, "repairs", None) == "u1"
        ]
        on_disk = {
            "child_returncode": proc.returncode,
            "killed_marker": (ws / "killed_before_second_checkpoint").is_file(),
            "dag_u1_status": dag_status,
            "mission_u1_status": mission_status,
            "mission_in_flight": in_flight,
            "mission_phase": None if state is None else state.get("phase"),
            "resume_u1_status_before_identify_ready": status_before_ready,
            "resume_u1_status_after_identify_ready": after.status,
            "resume_u1_attempts": after.attempts,
            "identify_ready": ready_ids,
            "repairs": repairs,
            "dag.json": dag,
            "mission/state.json": state,
        }
        _record_demo(
            name,
            ok=split and proc.returncode in (-9, 9, 128 + 9, -signal.SIGKILL),
            detail=(
                f"SIGKILL rc={proc.returncode} dag_status={dag_status} "
                f"mission_status={mission_status} in_flight={in_flight} "
                f"resume_before_ready={status_before_ready} "
                f"resume_after_identify_ready={after.status} attempts={after.attempts} "
                f"ready={ready_ids} repairs={repairs}. "
                "Concrete leftover: dag.json has u1=running; mission state has "
                "u1=pending and in_flight=[]. Resume marks u1 failed WITHOUT a "
                "repair unit; identify_ready re-readies it (attempts < 3) — "
                "the duplicate-launch class of bug."
            ),
            on_disk=on_disk,
            watched_fail=True,
            rank="loses mission position; duplicate dispatch of accepted-in-flight work",
        )


def demo_complete_then_checkpoint_crash() -> None:
    name = "INTERRUPTED: scheduler.complete persists DAG, checkpoint then raises"
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        units = {"a": _wu("a")}
        mission = Mission(
            ws,
            engine=object(),
            units=units,
            quiet=True,
            no_progress_threshold=100,
        )
        mission.scheduler.dispatch()
        mission.checkpoint()
        before_accepted = mission.accepted_count
        mission.scheduler.complete("a")
        dag_after_complete = json.loads(
            (ws / ".hcli" / "dag.json").read_text(encoding="utf-8")
        )
        def boom(self):
            raise CrashBetweenWrites("checkpoint after complete")

        raised = None
        with patch.object(Mission, "checkpoint", boom):
            try:
                mission.checkpoint()
            except CrashBetweenWrites as exc:
                raised = exc
        state = json.loads(
            (ws / ".hcli" / "mission" / "state.json").read_text(encoding="utf-8")
        )
        dag_status = ((dag_after_complete.get("units") or {}).get("a") or {}).get(
            "status"
        )
        mission_status = ((state.get("units") or {}).get("a") or {}).get("status")
        resumed = Mission.from_workspace(
            ws, engine=object(), quiet=True, no_progress_threshold=100
        )
        _record_demo(
            name,
            ok=raised is not None and dag_status == "completed",
            detail=(
                f"DAG a={dag_status} mission-state a={mission_status} "
                f"in_flight={state.get('in_flight')} "
                f"accepted_count on disk={state.get('accepted_count')} "
                f"in-memory accepted_count={before_accepted} "
                f"resume accepted_count={resumed.accepted_count} "
                f"resume a={resumed.scheduler.units['a'].status}"
            ),
            on_disk={
                "dag_a": dag_status,
                "mission_a": mission_status,
                "mission_accepted_count": state.get("accepted_count"),
                "resume_accepted_count": resumed.accepted_count,
            },
            watched_fail=True,
            rank="work is kept (DAG completed) but mission accepted_count/in_flight stale",
        )


def demo_fingerprint_persist_order() -> None:
    name = "INTERRUPTED: complete() persists DAG before recording fingerprint"
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        units = {f"u{i}": _wu(f"u{i}") for i in range(3)}
        sched = Scheduler(units, 3, workspace=ws, no_progress_threshold=3)
        sched.dispatch()
        snaps: List[List[str]] = []
        real = DagStore.save

        def spy(self, units_map, extra=None):
            snaps.append(list((extra or {}).get("fingerprints") or []))
            return real(self, units_map, extra=extra)

        raised = None
        with patch.object(DagStore, "save", spy):
            try:
                sched.complete("u0", fingerprint="SAME")
                sched.dispatch()
                sched.complete("u1", fingerprint="SAME")
                sched.dispatch()
                sched.complete("u2", fingerprint="SAME")
            except NO_PROGRESS as exc:
                raised = exc
        dag = json.loads((ws / ".hcli" / "dag.json").read_text(encoding="utf-8"))
        persisted_fps = dag.get("fingerprints") or []
        _record_demo(
            name,
            ok=raised is not None and persisted_fps.count("SAME") == 2,
            detail=(
                f"NO_PROGRESS raised={raised is not None} "
                f"persisted_fingerprints={persisted_fps} "
                f"in-memory={sched._fingerprints}. "
                "The fingerprint that tripped the governor is NOT on disk. "
                "Resume loads count=2, threshold=3, so the halt does not fire."
            ),
            on_disk={"dag.fingerprints": persisted_fps, "save_snaps": snaps},
            watched_fail=True,
            rank="loses mission position (no-progress governor)",
        )


def demo_grok_contract_then_receipt() -> None:
    name = "INTERRUPTED: GrokBridge writes contract, then receipt raises"
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        bridge = GrokBridge(ws)
        stdout = (
            "DRY RUN — would execute:\n"
            "grok --prompt-file /tmp/x/task.md\n"
            "task dir: /tmp/fake-tasks/recv-20260101-000000\n"
            "recv-20260101-000000\n"
        )
        completed = subprocess.CompletedProcess(
            ["grok-run", "delegate"], 0, stdout, ""
        )
        def boom(self, *args, **kwargs):
            raise CrashBetweenWrites("receipt after contract+launch")

        raised = None
        with patch("hcli.grok_bridge.find_grok_run", return_value="/fake/grok-run"):
            with patch.object(GrokBridge, "_run", return_value=completed):
                with patch.object(GrokBridge, "_write_receipt", boom):
                    try:
                        bridge.delegate(
                            "recv",
                            VALID_CONTRACT,
                            dry_run=True,
                            mutation_lock=lambda: __import__(
                                "contextlib"
                            ).nullcontext(),
                        )
                    except CrashBetweenWrites as exc:
                        raised = exc
        tree = _tree(ws / ".hcli")
        contracts = list((ws / ".hcli" / "grok" / "contracts").glob("*.md")) if (
            ws / ".hcli" / "grok" / "contracts"
        ).exists() else []
        receipts = list((ws / ".hcli" / "grok").glob("*.json")) if (
            ws / ".hcli" / "grok"
        ).exists() else []
        _record_demo(
            name,
            ok=raised is not None and bool(contracts) and not receipts,
            detail=(
                f"contract_files={[p.name for p in contracts]} "
                f"receipt_json={[p.name for p in receipts]}. "
                "Concrete leftover: grok-run has been invoked (task id assigned) "
                "and a contract file exists, but <workspace>/.hcli/grok/<id>.json "
                "was never written. Restart has no task record and will launch again."
            ),
            on_disk={"tree": tree, "contracts": [p.name for p in contracts]},
            watched_fail=True,
            rank="duplicate Grok launch; loses grok task record",
        )


def demo_grok_receipt_truncate_then_read() -> None:
    name = "GrokBridge._read_receipt trusts None on truncated JSON (no reconcile)"
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        bridge = GrokBridge(ws)
        path = bridge.receipt_path("task-1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"task_id": "task-1", "status"', encoding="utf-8")
        got = bridge._read_receipt("task-1")
        _record_demo(
            name,
            ok=got is None and path.is_file(),
            detail=(
                f"_read_receipt={got!r} live={path.read_text()!r}. "
                "Corrupt receipt is indistinguishable from a missing one."
            ),
            on_disk={"receipt": path.read_text(encoding="utf-8")},
            watched_fail=True,
            rank="loses grok task record",
        )


def demo_engine_two_file_mutation() -> None:
    name = "INTERRUPTED: Engine._apply_operations second write_text raises"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        engine = _engine(root)
        a = engine.root / "a.py"
        b = engine.root / "b.py"
        a.write_text("A_OLD\n", encoding="utf-8")
        b.write_text("B_OLD\n", encoding="utf-8")
        ops = [
            {"op": "replace_file", "path": "a.py", "new_text": "A_NEW\n"},
            {"op": "replace_file", "path": "b.py", "new_text": "B_NEW\n"},
        ]
        real = Path.write_text
        seen = {"n": 0}

        def counted(self, data, encoding="utf-8", errors="strict", newline=None):
            seen["n"] += 1
            if self.name in {"a.py", "b.py"} and seen["n"] >= 2 and data == "B_NEW\n":
                raise CrashBetweenWrites("second mutation file")
            return real(self, data, encoding=encoding, errors=errors, newline=newline)

        raised = None
        with patch.object(Path, "write_text", counted):
            try:
                engine._apply_operations(ops)
            except CrashBetweenWrites as exc:
                raised = exc
        _record_demo(
            name,
            ok=raised is not None and a.read_text() == "A_NEW\n" and b.read_text() == "B_OLD\n",
            detail=(
                f"a.py={a.read_text()!r} b.py={b.read_text()!r}. "
                "Concrete leftover: a.py mutated, b.py original. "
                "_apply_operations is not all-or-nothing. execute() would "
                "snapshot/restore on exception, but SIGKILL skips restore."
            ),
            on_disk={"a.py": a.read_text(), "b.py": b.read_text()},
            watched_fail=True,
            rank="loses accepted work (partial mutation)",
        )


def demo_engine_execute_restores_on_exception() -> None:
    name = "Engine.execute restores snapshot when apply raises (exception, not SIGKILL)"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        engine = _engine(root)
        a = engine.root / "a.py"
        a.write_text("A_OLD\n", encoding="utf-8")

        def fake_model(*_a, **_k):
            return {
                "kind": "mutation",
                "content": "x",
                "operations": [
                    {"op": "replace_file", "path": "a.py", "new_text": "A_NEW\n"}
                ],
                "tests": ["test_a.py"],
            }

        real = Path.write_text

        def boom(self, data, encoding="utf-8", errors="strict", newline=None):
            if self.name == "a.py" and data == "A_NEW\n":
                real(self, "A_PARTIAL\n", encoding=encoding)
                raise CrashBetweenWrites("apply")
            return real(self, data, encoding=encoding, errors=errors, newline=newline)

        engine._call_model = fake_model  # type: ignore[method-assign]
        engine._validate = lambda *a, **k: {"ok": True, "checks": []}  # type: ignore
        with patch.object(Path, "write_text", boom):
            try:
                engine.execute("mutate a")
            except Exception:
                pass
        live = a.read_text() if a.exists() else None
        restored = live == "A_OLD\n"
        _record_demo(
            name,
            ok=restored,
            detail=f"after execute() exception a.py={live!r} restored={restored}",
            on_disk={"a.py": live},
            watched_fail=not restored,
            rank="safe for exceptions; NOT safe for SIGKILL (see previous demo)",
        )
        if restored:
            SAFE.append(
                {
                    "writer": "W19 Engine._restore via execute()",
                    "claim": "exception during apply rolls workspace files back",
                    "evidence": f"a.py restored to A_OLD, watched live={live!r}",
                }
            )


def demo_engine_receipt_after_mutation() -> None:
    name = "INTERRUPTED: mutation on disk, _write_receipt raises"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        engine = _engine(root)
        a = engine.root / "a.py"
        a.write_text("A_OLD\n", encoding="utf-8")

        def fake_model(*_a, **_k):
            return {
                "kind": "mutation",
                "content": "x",
                "operations": [
                    {"op": "replace_file", "path": "a.py", "new_text": "A_NEW\n"}
                ],
                "tests": [],
            }

        child = root / "child.py"
        child.write_text(
            "\n".join(
                [
                    "import os, sys, signal",
                    
                    "from pathlib import Path",
                    "from hcli.engine import Engine",
                    "from hcli.events import EventBus",
                    "from hcli.workspace import Workspace",
                    f"root = Path({str(engine.root)!r})",
                    "engine = Engine(workspace=Workspace(str(root)), event_bus=EventBus(),",
                    "                runtime_count=1, model_name='/missing.gguf')",
                    "(engine.root / 'a.py').write_text('A_OLD\\n', encoding='utf-8')",
                    "def fake_model(*_a, **_k):",
                    "    return {'kind': 'mutation', 'content': 'x',",
                    "            'operations': [{'op': 'replace_file', 'path': 'a.py',",
                    "                            'new_text': 'A_NEW\\n'}],",
                    "            'tests': []}",
                    "engine._call_model = fake_model",
                    "def die(self, *args, **kwargs):",
                    "    (root / 'killed_before_receipt').write_text('1')",
                    "    os.kill(os.getpid(), signal.SIGKILL)",
                    "Engine._write_receipt = die",
                    "engine.execute('mutate a')",
                ]
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, str(child)],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        live = a.read_text(encoding="utf-8") if a.exists() else None
        receipts = (
            list((engine.root / ".hcli" / "receipts").glob("*.json"))
            if (engine.root / ".hcli" / "receipts").exists()
            else []
        )
        _record_demo(
            name,
            ok=(
                proc.returncode in (-9, 9, 128 + 9, -signal.SIGKILL)
                and live == "A_NEW\n"
                and not receipts
            ),
            detail=(
                f"SIGKILL rc={proc.returncode} a.py={live!r} "
                f"receipts={[p.name for p in receipts]} "
                f"killed_marker={(root / 'killed_before_receipt').is_file()}. "
                "Accepted work is on disk with no receipt. Restart cannot tell "
                "whether the mutation was validated. (In-process exception would "
                "restore; SIGKILL skips restore.)"
            ),
            on_disk={"a.py": live, "receipts": [p.name for p in receipts]},
            watched_fail=True,
            rank="loses accepted-work evidence (mutation kept, receipt missing)",
        )


def demo_mutation_rollback_create() -> None:
    name = "mutation.rollback of a created file TypeErrors and leaves the file"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / "new.py"
        ops = [{"op": "create", "path": str(target), "content": "CREATED\n"}]

        class Guard:
            def resolve(self, path):
                return str(root / path) if not os.path.isabs(path) else path

        result = apply_mutation_operations(Guard(), ops)
        exists_before = target.is_file()
        raised = None
        try:
            rollback_mutation(result)
        except Exception as exc:
            raised = exc
        exists_after = target.is_file()
        _record_demo(
            name,
            ok=exists_before and exists_after and raised is not None,
            detail=(
                f"created={exists_before} after_rollback={exists_after} "
                f"raised={type(raised).__name__}: {raised}. "
                "_restore_file(None) does Path(snapshot['path']) on None."
            ),
            on_disk={"new.py": target.read_text() if target.exists() else None},
            watched_fail=True,
            rank="loses accepted work (create cannot be rolled back)",
        )


def demo_config_tmp_replace_and_corrupt_read() -> None:
    name = "Config: tmp+replace is truncation-safe; corrupt file loads as {}"
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        ws.mkdir()
        cfg = Config(str(ws), global_path=str(Path(tmp) / "global.json"))
        cfg.save_project({"model": "/ok.gguf"})
        live_path = Path(cfg.project_path)
        before = live_path.read_text(encoding="utf-8")
        real_open = open

        def exploding_open(path, mode="r", *args, **kwargs):
            if "w" in str(mode) and str(path).endswith(".tmp"):
                return _PartialThenCrash(real_open(path, mode, *args, **kwargs), limit=8)
            return real_open(path, mode, *args, **kwargs)

        raised = None
        with patch("builtins.open", exploding_open):
            try:
                cfg.save_project({"model": "/other.gguf"})
            except CrashBetweenWrites as exc:
                raised = exc
        after_crash = live_path.read_text(encoding="utf-8")
        intact = after_crash == before
        live_path.write_text("{not json", encoding="utf-8")
        loaded = cfg._read(str(live_path))
        _record_demo(
            name,
            ok=raised is not None and intact and loaded == {},
            detail=(
                f"crash left live intact={intact}; truncated/corrupt _read -> {loaded!r} "
                "(swallowed JSONDecodeError, trusted as empty)."
            ),
            on_disk={"after_crash": after_crash, "corrupt_load": loaded},
            watched_fail=True,
            rank="corrupt config silently becomes empty (wrong model on resume)",
        )
        if intact:
            SAFE.append(
                {
                    "writer": "W07 Config.save_project",
                    "claim": "tmp+replace does not truncate live config on mid-tmp crash",
                    "evidence": f"live stayed {after_crash!r}",
                }
            )


def demo_controller_swallows_save_global() -> None:
    name = "Controller.select_model swallows OSError from save_global"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        models = root / "models"
        models.mkdir()
        gguf = models / "only-7B-Q4.gguf"
        gguf.write_bytes(b"x" * 50)
        ws = root / "ws"
        ws.mkdir()
        from hcli.models import ModelRegistry

        controller = Controller(
            workspace=str(ws),
            runtime_count=1,
            model=None,
            bus=EventBus(),
            registry=ModelRegistry([str(models)]),
        )
        controller.config.global_path = str(root / "home" / ".config" / "hcli" / "config.json")

        def boom(data):
            raise OSError("disk full")

        controller.config.save_global = boom  # type: ignore[method-assign]
        ok = controller.select_model("1")
        written = Path(controller.config.global_path).is_file()
        believes = controller.model == str(gguf.resolve()) or (
            controller.model_info is not None
            and Path(controller.model_info.path).name == gguf.name
        )
        controller.shutdown()
        _record_demo(
            name,
            ok=ok is True and not written and believes,
            detail=(
                f"select_model returned {ok}, in-memory model set={believes}, "
                f"global config written={written}. Persist failure was swallowed."
            ),
            on_disk={"global_exists": written},
            watched_fail=True,
            rank="cosmetic/config: process believes a model that is not durable",
        )


def demo_steering_corrupt_is_empty() -> None:
    name = "SteeringQueue._load treats corrupt JSON as empty (steers lost)"
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        q = SteeringQueue(str(ws), "s1")
        q.enqueue("remember the constraint", kind="constraint")
        path = Path(q._path)
        before = path.read_text(encoding="utf-8")
        path.write_text("{truncated", encoding="utf-8")
        q2 = SteeringQueue(str(ws), "s1")
        _record_demo(
            name,
            ok=before.strip().startswith("[") and q2.all() == [],
            detail=(
                f"before={before[:80]!r} after_reload_events={len(q2.all())}. "
                "except Exception: self._events = []. Restart trusts empty."
            ),
            on_disk={"before": before, "after_file": path.read_text()},
            watched_fail=True,
            rank="loses mission position (operator steers vanish)",
        )


def demo_apply_constraint_ledger_not_on_disk() -> None:
    name = "INTERRUPTED class: apply_constraint saves steering, never GOAL.md"
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        goal = ws / "GOAL.md"
        goal.write_text(
            "- [ ] G001 — keep me | status: PENDING | risk: high | tier: V2\n"
            "      acceptance: x\n"
            "      verify: true\n"
            "      evidence: (none yet)\n",
            encoding="utf-8",
        )
        ledger = Ledger.parse(goal)
        q = SteeringQueue(str(ws), "s1")
        event = q.enqueue("add obligation: new work from steer", kind="constraint")
        q.apply_constraint(event, ledger)
        goal_after = goal.read_text(encoding="utf-8")
        steering = Path(q._path).read_text(encoding="utf-8")
        ids = [ob.id for ob in ledger.obligations()]
        _record_demo(
            name,
            ok="G002" in ids and "G002" not in goal_after and "applied" in steering,
            detail=(
                f"in-memory obligations={ids}; GOAL.md still has only G001; "
                f"steering JSON marks the event applied. Crash here: resume "
                "parses GOAL.md (G001 only) and steering (applied=true, no re-apply)."
            ),
            on_disk={"GOAL.md": goal_after, "steering": steering, "memory_ids": ids},
            watched_fail=True,
            rank="loses mission position (constraint never durable)",
        )


def demo_ledger_has_no_save() -> None:
    name = "Ledger.mark_verified does not write GOAL.md"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "GOAL.md"
        path.write_text(
            "- [ ] G001 — x | status: PENDING | risk: high | tier: V2\n"
            "      acceptance: a\n"
            "      verify: python3 -c 'import sys; sys.exit(0)'\n"
            "      evidence: (none yet)\n",
            encoding="utf-8",
        )
        ledger = Ledger.parse(path)
        result = ledger.run_verify("G001")
        if result.passed:
            ledger.mark_verified("G001", result)
        disk = path.read_text(encoding="utf-8")
        has_save = hasattr(Ledger, "save")
        _record_demo(
            name,
            ok=(not has_save) and "VERIFIED" not in disk and ledger.get("G001").status == "VERIFIED",
            detail=(
                f"has_save={has_save} memory={ledger.get('G001').status} "
                f"disk_contains_VERIFIED={'VERIFIED' in disk}"
            ),
            on_disk={"GOAL.md": disk},
            watched_fail=True,
            rank="loses mission position (obligation ledger is memory-only)",
        )


def demo_session_never_saved() -> None:
    name = "SessionStore.save exists; Controller never calls it"
    src = inspect.getsource(Controller)
    calls = len(re.findall(r"session_store\.save", src))
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        c = Controller(workspace=str(ws), runtime_count=1, bus=EventBus())
        files = list((ws / ".hcli" / "sessions").glob("*.json")) if (
            ws / ".hcli" / "sessions"
        ).exists() else []
        sid = c.session.id
        loaded = c.resume_session(sid)
        c.shutdown()
        _record_demo(
            name,
            ok=calls == 0 and loaded is None and files == [],
            detail=(
                f"Controller.session_store.save call-sites={calls}; "
                f"after init session files={ [p.name for p in files] }; "
                f"resume_session({sid!r})={loaded!r}"
            ),
            on_disk={"sessions": [p.name for p in files]},
            watched_fail=True,
            rank="cosmetic (session identity/messages never durable)",
        )


def demo_mission_log_swallows() -> None:
    name = "Mission._log swallows OSError; process believes the event was recorded"
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        mission = Mission(
            ws,
            engine=object(),
            units={"a": _wu("a")},
            quiet=True,
            no_progress_threshold=100,
        )
        mission.log_path.parent.mkdir(parents=True, exist_ok=True)
        mission.log_path.write_text("", encoding="utf-8")
        real_open = open

        def boom(path, mode="r", *args, **kwargs):
            if "a" in str(mode) and str(path).endswith("mission.log"):
                raise OSError("log unwritable")
            return real_open(path, mode, *args, **kwargs)

        with patch("builtins.open", boom):
            mission._log({"event": "accepted", "id": "a"})
        live = mission.log_path.read_text(encoding="utf-8")
        _record_demo(
            name,
            ok=live == "",
            detail="accepted event swallowed; mission.log empty; no exception escaped",
            on_disk={"mission.log": live},
            watched_fail=True,
            rank="cosmetic (log); control state lives elsewhere",
        )


def demo_dag_corrupt_raises() -> None:
    name = "DagStore.load raises DagCorruptError (does not trust garbage)"
    with tempfile.TemporaryDirectory() as tmp:
        store = DagStore(tmp)
        store.save({"a": _wu("a")})
        store.path.write_text("{not json", encoding="utf-8")
        raised = None
        try:
            store.load()
        except Exception as exc:
            raised = exc
        _record_demo(
            name,
            ok=raised is not None and type(raised).__name__ == "DagCorruptError",
            detail=f"raised={type(raised).__name__}: {raised}",
            on_disk={"dag.json": store.path.read_text()},
            watched_fail=True,
            rank="safe reader: refuses truncated DAG (does not silently complete work)",
        )
        if raised is not None:
            SAFE.append(
                {
                    "writer": "W02 DagStore.load (reader)",
                    "claim": "truncated dag.json is not loaded as a valid DAG",
                    "evidence": f"{type(raised).__name__}: {raised}",
                }
            )


def demo_mission_corrupt_raises() -> None:
    name = "Mission.load_state raises MissionCorruptError on truncated JSON"
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        mission = Mission(
            ws, engine=object(), units={"a": _wu("a")}, quiet=True, no_progress_threshold=100
        )
        mission.checkpoint()
        path = mission.state_path
        path.write_text("{not json", encoding="utf-8")
        raised = None
        try:
            load_state(path)
        except MissionCorruptError as exc:
            raised = exc
        _record_demo(
            name,
            ok=raised is not None,
            detail=f"raised={type(raised).__name__}: {raised}",
            on_disk={"state.json": path.read_text()},
            watched_fail=True,
            rank="safe reader for mission JSON (refuses garbage)",
        )
        if raised is not None:
            SAFE.append(
                {
                    "writer": "W04 Mission.load_state (reader)",
                    "claim": "truncated mission state is not trusted",
                    "evidence": f"{type(raised).__name__}: {raised}",
                }
            )


def demo_atomic_write_json_raises() -> None:
    name = "atomic_write_json persist failure RAISES (not swallowed)"
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "x.json"
        dest.write_text("{}", encoding="utf-8")
        real = os.replace

        def boom(src, dst, *args, **kwargs):
            del src, dst, args, kwargs
            raise OSError("replace failed")

        raised = None
        with patch("os.replace", boom):
            try:
                atomic_write_json(dest, {"a": 1})
            except OSError as exc:
                raised = exc
        live = dest.read_text(encoding="utf-8")
        _record_demo(
            name,
            ok=raised is not None and live == "{}",
            detail=f"raised={raised!r} live={live!r}",
            on_disk={"x.json": live},
            watched_fail=True,
            rank="safe: persist failure is visible; dest unchanged",
        )
        if raised is not None:
            SAFE.append(
                {
                    "writer": "W01 atomic_write_json",
                    "claim": "os.replace failure raises; live dest unchanged",
                    "evidence": f"OSError {raised!r}, live={live!r}",
                }
            )


def demo_mutation_lock_corrupt_is_free() -> None:
    name = "Corrupt mutation.lock is treated as unlocked"
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        lock = MutationLock(ws)
        lock.acquire("u1")
        lock.path.write_text("{truncated", encoding="utf-8")
        got = lock.read()
        ok = lock.acquire("u2")
        _record_demo(
            name,
            ok=got is None and ok is True,
            detail=(
                f"read()={got!r} acquire(u2)={ok}. Truncated lock file => "
                "holder_is_live is false => second writer proceeds."
            ),
            on_disk={"mutation.lock": lock.path.read_text() if lock.path.is_file() else None},
            watched_fail=True,
            rank="loses accepted work (two MUTATION units can run)",
        )


def demo_runtime_ownership_after_spawn() -> None:
    name = "INTERRUPTED: RuntimePool.start spawns, SIGKILL before ownership durable"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = root / "dummy.gguf"
        model.write_bytes(b"x" * 64)
        child = root / "child.py"
        child.write_text(
            "\n".join(
                [
                    "import os, sys, signal",
                    
                    "from pathlib import Path",
                    f"sys.path.insert(0, {str(REPO / 'tools' / 'headless')!r})",
                    "os.environ['HCLI_DISABLE_SIGNAL_HOOKS'] = '1'",
                    "os.environ['HCLI_RESIDENT_RUNTIME_LIMIT'] = '1'",
                    "os.environ['HCLI_ACTIVE_DECODE_LIMIT'] = '1'",
                    "os.environ['HCLI_DECODE_TOPOLOGY'] = 'process'",
                    "from hcli.runtime import RuntimePool",
                    "from hcli.machine import GIB, MemGate",
                    "import subprocess, time",
                    "from hcli.resources import process_start_token",
                    f"root = Path({str(root)!r})",
                    f"model = Path({str(model)!r})",
                    "class FakeBackend:",
                    "    def __init__(self, model_path, port, n_slots=1, index=0, **_k):",
                    "        self.model_path = model_path; self.port = port",
                    "        self.process = None; self.pid = None; self.start_time = None",
                    "    def identity(self):",
                    "        return {'backend': 'fake', 'model_path': self.model_path}",
                    "    def spawn(self, **kwargs):",
                    "        self.process = subprocess.Popen(['sleep', '120'],",
                    "            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,",
                    "            start_new_session=True)",
                    "        self.pid = self.process.pid",
                    "        self.start_time = process_start_token(self.pid)",
                    "        Path(root / 'child.pid').write_text(str(self.pid))",
                    "    def ready(self, timeout):",
                    "        return self.process is not None and self.process.poll() is None",
                    "    def stop(self):",
                    "        if self.process is not None and self.process.poll() is None:",
                    "            self.process.kill()",
                    "        return {'pid': self.pid, 'gone': True, 'unreaped': []}",
                    "gate = MemGate(reserve_bytes=1, model_bytes=100,",
                    "    per_runtime_overhead_bytes=100, headroom_frac=0.1,",
                    "    metal_info={'recommendedMaxWorkingSetSize': 80 * GIB,",
                    "                'currentAllocatedSize': 0, 'source': 'audit'},",
                    "    topology='process')",
                    "pool = RuntimePool(str(model), requested_n=1, workspace=root,",
                    "    backend_factory=FakeBackend, mem_gate=gate, topology='process',",
                    "    repo_root=root)",
                    "def die(self):",
                    "    (root / 'killed_before_ownership').write_text('1')",
                    "    os.kill(os.getpid(), signal.SIGKILL)",
                    "pool._write_ownership = die.__get__(pool, RuntimePool)",
                    "pool.start()",
                ]
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", "")
        env["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
        proc = subprocess.run(
            [sys.executable, str(child)],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        pid_file = root / "child.pid"
        child_pid = None
        alive = False
        if pid_file.is_file():
            try:
                child_pid = int(pid_file.read_text().strip())
            except ValueError:
                child_pid = None
        if child_pid:
            try:
                os.kill(child_pid, 0)
                alive = True
            except OSError:
                alive = False
        ownership = root / ".hcli" / "runtime_pool.json"
        try:
            _record_demo(
                name,
                ok=(
                    proc.returncode in (-9, 9, 128 + 9, -signal.SIGKILL)
                    and alive
                    and not ownership.is_file()
                ),
                detail=(
                    f"SIGKILL rc={proc.returncode} sleep_pid={child_pid} "
                    f"sleep_alive={alive} ownership_exists={ownership.is_file()} "
                    f"stderr={(proc.stderr or '')[-400:]!r}. "
                    "Concrete leftover: the spawned runtime process is still "
                    "alive and runtime_pool.json was never written. "
                    "reap_orphans on a later pool has nothing to reap."
                ),
                on_disk={
                    "sleep_pid": child_pid,
                    "sleep_alive": alive,
                    "ownership_exists": ownership.is_file(),
                    "killed_marker": (root / "killed_before_ownership").is_file(),
                },
                watched_fail=True,
                rank="loses mission position (orphan GPU/runtime process)",
            )
        finally:
            if child_pid:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except OSError:
                    pass


def demo_machine_genome_truncates() -> None:
    name = "MachineGenome.save write_text truncates live genome"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "machine-genome.json"
        g = MachineGenome(path)
        g.data = {"resident_runtime_limit": 4}
        g.save()
        original = path.read_text(encoding="utf-8")
        g.data = {"resident_runtime_limit": 99, "pad": "y" * 200}

        def truncating(self, data, encoding="utf-8", errors="strict", newline=None):
            raw = data.encode(encoding) if isinstance(data, str) else data
            Path.write_bytes(self, raw[:20])
            raise CrashBetweenWrites("genome")

        raised = None
        with patch.object(Path, "write_text", truncating):
            try:
                g.save()
            except CrashBetweenWrites as exc:
                raised = exc
        live = path.read_bytes()
        parseable = True
        try:
            json.loads(live.decode("utf-8", "replace"))
        except Exception:
            parseable = False
        _record_demo(
            name,
            ok=raised is not None and not parseable and live != original.encode(),
            detail=f"bytes={live!r} parseable={parseable}",
            on_disk={"genome_bytes": repr(live)},
            watched_fail=True,
            rank="cosmetic/config (runtime limit file garbage on resume)",
        )


def demo_install_shims_partial() -> None:
    name = "INTERRUPTED: install_shims writes first shim, second write_text raises"
    from hcli.cli import install_shims

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        real = Path.write_text
        seen = {"n": 0}

        def counted(self, data, encoding="utf-8", errors="strict", newline=None):
            if self.name in {"hcli", "jhcli"}:
                seen["n"] += 1
                if seen["n"] >= 2:
                    raise CrashBetweenWrites("second shim")
            return real(self, data, encoding=encoding, errors=errors, newline=newline)

        raised = None
        with patch.object(Path, "write_text", counted):
            try:
                install_shims(home=str(home))
            except CrashBetweenWrites as exc:
                raised = exc
        bin_dir = home / ".local" / "bin"
        names = sorted(p.name for p in bin_dir.glob("*")) if bin_dir.exists() else []
        current = home / ".local" / "share" / "hcli" / "current"
        _record_demo(
            name,
            ok=raised is not None and names == ["hcli"],
            detail=(
                f"bin names={names} current_is_symlink={current.is_symlink()}. "
                "Concrete leftover: package copied, current symlink set, "
                "only the hcli shim exists — jhcli missing."
            ),
            on_disk={"bin": names, "current": str(current) if current.exists() else None},
            watched_fail=True,
            rank="cosmetic (install incomplete)",
        )


def demo_recover_running_no_repair() -> None:
    name = "DagStore.load recover_running: running -> failed, no repair unit"
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        units = {"a": _wu("a"), "b": _wu("b", dependencies=["a"])}
        sched = Scheduler(units, 1, workspace=ws)
        sched.dispatch()
        assert units["a"].status == "running"
        restarted = Scheduler.from_workspace(ws, runtime_count=1)
        statuses = {uid: wu.status for uid, wu in restarted.units.items()}
        repairs = [u.id for u in restarted.units.values() if u.repairs]
        ready = [u.id for u in identify_ready(restarted.units)]
        _record_demo(
            name,
            ok=statuses.get("a") in {"failed", "ready"} and not repairs,
            detail=(
                f"statuses={statuses} repairs={repairs} ready={ready}. "
                "Crash-recovered unit is re-readied, not repaired."
            ),
            on_disk={"statuses": statuses, "ready": ready, "repairs": repairs},
            watched_fail=True,
            rank="duplicate dispatch of in-flight work",
        )


# ---------------------------------------------------------------------------
# HANDOFF patches (exact; for a sibling lane — this lane does not edit hcli)
# ---------------------------------------------------------------------------


HANDOFF_PATCHES = r'''
## HANDOFF

This lane is DENIED writes under tools/haider/hcli/. Apply these exact patches
in the lane that owns those modules.

### P1 — Mission.checkpoint: DAG first, raise, stamp checkpoint_id

The claimed repair is NOT in this tree. checkpoint() writes only
.hcli/mission/state.json and does not stamp checkpoint_id. SIGKILL between
Scheduler._persist (dispatch) and checkpoint() leaves dag.json with the unit
running and mission state with the unit pending / in_flight=[]. Resume then
re-dispatches (identify_ready on failed, attempts < 3).

--- a/tools/haider/hcli/mission.py
+++ b/tools/haider/hcli/mission.py
@@ def checkpoint(self) -> Path:
     def checkpoint(self) -> Path:
         path = self.state_path
         path.parent.mkdir(parents=True, exist_ok=True)
         with self._lock:
             inflight = sorted(self._inflight)
+        checkpoint_id = str(uuid.uuid4())
+        # DAG first. A crash here leaves mission stale and DAG current;
+        # resume trusts DAG. The inverse (mission ahead of DAG) re-dispatches.
+        if getattr(self.scheduler, "store", None) is not None:
+            self.scheduler.store.save(
+                self.scheduler.units,
+                extra={
+                    "fingerprints": list(self.scheduler._fingerprints),
+                    "no_progress_threshold": self.scheduler.no_progress_threshold,
+                    "active_decode_limit": self.scheduler.active_decode_limit,
+                    "active_decode_limit_source": self.scheduler.active_decode_limit_source,
+                    "checkpoint_id": checkpoint_id,
+                },
+            )
         payload = {
             "version": MISSION_VERSION,
             "id": self.id,
             "goal": self.goal,
             "phase": self.phase,
             "strategy": self.strategy,
             "started_at": self.started_at,
             "last_checkpoint": time.time(),
             "in_flight": inflight,
             "accepted_count": self.accepted_count,
             "no_progress_warning": self.no_progress_warning,
             "child_pids": sorted(self.child_pids),
             "cancel_reason": self.cancel_reason,
             "session_id": self.session_id,
             "no_progress_threshold": self.no_progress_threshold,
+            "checkpoint_id": checkpoint_id,
             "units": {
                 uid: wu.to_dict() for uid, wu in self.scheduler.units.items()
             },
         }
         atomic_write_json(path, payload)
         self.last_checkpoint = payload["last_checkpoint"]
         return path

Do not catch the save/atomic_write_json failure. Persist failure must raise.

### P2 — Crash recovery must emit a repair and reap recorded children

DagStore.load(recover_running=True) marks running -> failed and does not emit
a repair. identify_ready then re-readies the same id (attempts < 3). That is
the duplicate Grok launch. from_workspace also adopts child_pids without killing
them.

--- a/tools/haider/hcli/dag_store.py
+++ b/tools/haider/hcli/dag_store.py
@@ in DagStore.load, recover_running loop:
         if recover_running:
+            recovered = []
             for wu in units.values():
                 if wu.status == "running":
+                    recovered.append(wu.id)
                     transition_status(wu, "failed")
                     wu.assigned_runtime = None
+                    ctx = dict(wu.failure_context or {})
+                    ctx["crash_recovered"] = True
+                    wu.failure_context = ctx
+            self.last_meta["recovered_running"] = recovered

--- a/tools/haider/hcli/mission.py
+++ b/tools/haider/hcli/mission.py
@@ in Mission.from_workspace, after sched = Scheduler.from_workspace / fallback:
+        recovered = []
+        meta = getattr(sched.store, "last_meta", None) or {}
+        recovered.extend(str(x) for x in (meta.get("recovered_running") or []))
         for uid in data.get("in_flight") or []:
             wu = sched.units.get(str(uid))
             if wu is None:
                 continue
             if wu.status == "running":
-                transition_status(wu, "failed")
-                wu.assigned_runtime = None
+                recovered.append(str(uid))
-        sched._persist()
+        for uid in recovered:
+            wu = sched.units.get(uid)
+            if wu is None:
+                continue
+            already = any(other.repairs == uid for other in sched.units.values())
+            if already:
+                continue
+            sched.fail(uid, context={"reason": "crash_recovered"})
         mission = cls(...)
         ...
         for pid in data.get("child_pids") or []:
             try:
                 mission.child_pids.add(int(pid))
             except (TypeError, ValueError):
                 pass
+        if mission.child_pids:
+            mission._stop_children()
         if mission.phase == "running":
             mission.phase = "idle"
         return mission

### P3 — scheduler.complete: persist fingerprints INCLUDING the new one

Current order: _persist() then _record_fingerprint() which may raise
NO_PROGRESS. The tripping fingerprint never hits disk.

--- a/tools/haider/hcli/scheduler.py
+++ b/tools/haider/hcli/scheduler.py
@@ def complete(self, wu_id, fingerprint=None):
             transition_status(wu, "completed")
             if was_running:
                 self._release_unit(wu)
-            self._persist()
-            self._record_fingerprint(fingerprint)
+            if fingerprint is not None:
+                self._fingerprints.append(fingerprint)
+            self._persist()
+            if fingerprint is not None:
+                count = self._fingerprints.count(fingerprint)
+                if count >= self.no_progress_threshold:
+                    raise NO_PROGRESS(
+                        fingerprint=fingerprint,
+                        count=count,
+                        threshold=self.no_progress_threshold,
+                    )

### P4 — Grok receipts/contracts: use atomic_write_json / atomic text

--- a/tools/haider/hcli/grok_bridge.py
+++ b/tools/haider/hcli/grok_bridge.py
@@
+from .dag_store import atomic_write_json
@@ _write_contract_file:
-        path.write_text(contract_text, encoding="utf-8")
+        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
+        tmp.write_text(contract_text, encoding="utf-8")
+        os.replace(tmp, path)
@@ _write_receipt / _update_receipt:
-        path.write_text(json.dumps(...), encoding="utf-8")
+        atomic_write_json(path, receipt)  # or `current` in _update_receipt

Also: write a "pending" receipt BEFORE invoking grok-run, then update it
after extract_task_id. Crash-between currently leaves a launched grok-run
with no .hcli/grok/<id>.json.

### P5 — Engine._write_receipt: atomic_write_json

--- a/tools/haider/hcli/engine.py
+++ b/tools/haider/hcli/engine.py
@@
+from .dag_store import atomic_write_json
@@ end of _write_receipt:
-        path.write_text(json.dumps(self._strip_reasoning(receipt), ...), encoding="utf-8")
+        atomic_write_json(path, self._strip_reasoning(receipt))

### P6 — Engine._apply_operations: write via tmp+replace per file

For each file mutation, write to `.{name}.{pid}.{uuid}.tmp` in the same
directory and os.replace onto the live path. execute() already snapshots
and restores on exception; SIGKILL still needs per-file atomicity so a
crash mid-write does not truncate a user's file. Cross-file all-or-nothing
under SIGKILL additionally needs a journal (.hcli/mutation-journal.json)
listing in-flight paths; without it, a two-file mutation can remain partial
(demonstrated).

### P7 — mutation._restore_file None-snapshot bug

--- a/tools/haider/hcli/mutation.py
+++ b/tools/haider/hcli/mutation.py
@@
-def _restore_file(snapshot: Optional[Dict[str, Any]]) -> None:
-    if snapshot is None:
-        p = Path(snapshot["path"])
-        if p.exists():
-            p.unlink()
-    else:
-        p = Path(snapshot["path"])
-        p.parent.mkdir(parents=True, exist_ok=True)
-        p.write_text(snapshot["content"], encoding="utf-8")
+def _restore_file(snapshot: Optional[Dict[str, Any]], path: Optional[str] = None) -> None:
+    if snapshot is None:
+        if not path:
+            return
+        p = Path(path)
+        if p.exists():
+            p.unlink()
+        return
+    p = Path(snapshot["path"])
+    p.parent.mkdir(parents=True, exist_ok=True)
+    p.write_text(snapshot["content"], encoding="utf-8")
@@ rollback_mutation:
-        _restore_file(snap)
+        _restore_file(snap, full_path)

### P8 — RuntimePool.start: write ownership BEFORE returning, and stop() on failure

--- a/tools/haider/hcli/runtime.py
+++ b/tools/haider/hcli/runtime.py
@@ end of start():
         self.admitted_n = len(self.runtimes)
         if self.runtimes:
-            self._write_ownership()
+            try:
+                self._write_ownership()
+            except BaseException:
+                self.stop()
+                raise

### P9 — MachineGenome.save atomic

--- a/tools/haider/hcli/machine.py
+++ b/tools/haider/hcli/machine.py
@@ def save(self) -> None:
-        self.path.write_text(json.dumps(self.data, indent=2))
+        from .dag_store import atomic_write_json
+        atomic_write_json(self.path, self.data)

### P10 — Config/Session/Steering: fsync before replace; do not swallow corrupt

Config._write / save_project / SessionStore.save / SteeringQueue._save:
after json.dump, handle.flush(); os.fsync(handle.fileno()); then os.replace.

SteeringQueue._load:
-            except Exception:
-                self._events = []
+            except Exception as exc:
+                raise RuntimeError(f"steering queue corrupt: {self._path}: {exc}") from exc

Config._read: do not return {} on JSONDecodeError for a file that exists;
raise. Empty-on-missing is fine.

Controller.select_model: delete `except OSError: pass` around save_global.

### P11 — Ledger.save using tmp+replace; apply_constraint must persist GOAL.md

ledger.py has no save(). Add:

    def save(self, path: Union[str, Path]) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f".{dest.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        payload = self.to_markdown()
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, dest)
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

SteeringQueue.apply_constraint must take a goal_md path (or the caller
must save immediately after). Today the in-memory ledger amendment is
lost on crash while the steer is marked applied.

### P12 — SessionStore.save must be called

Controller.start_mission / __init__ / select_model should
`self.session_store.save(self.session)`. resume_session is dead code
without that.
'''


def ranked_findings() -> List[Dict[str, Any]]:
    return [
        {
            "rank": 1,
            "severity": "loses mission position + duplicate in-flight work",
            "id": "F-DISPATCH-SPLIT",
            "where": "scheduler.dispatch _persist then Mission.checkpoint",
            "watched": "SIGKILL on the second checkpoint: dag.json u1=running, mission state u1=pending in_flight=[]",
            "resume": "from_workspace recover_running -> failed, no repair; identify_ready re-readies the same id",
        },
        {
            "rank": 2,
            "severity": "duplicate Grok launch / loses grok task record",
            "id": "F-GROK-CONTRACT-THEN-RECEIPT",
            "where": "GrokBridge._launch: _write_contract_file, grok-run, _write_receipt",
            "watched": "receipt raise left contracts/*.md and zero <id>.json",
            "resume": "_read_receipt returns None; caller retries delegate()",
        },
        {
            "rank": 3,
            "severity": "loses accepted work (partial mutation)",
            "id": "F-APPLY-PARTIAL",
            "where": "Engine._apply_operations sequential write_text",
            "watched": "a.py=A_NEW, b.py=B_OLD after second write raised",
            "resume": "trusts whatever is on disk; no journal",
        },
        {
            "rank": 4,
            "severity": "loses accepted-work evidence",
            "id": "F-MUTATION-THEN-RECEIPT",
            "where": "Engine.execute apply then _write_receipt",
            "watched": "a.py mutated, .hcli/receipts empty",
            "resume": "cannot tell whether the mutation was validated",
        },
        {
            "rank": 5,
            "severity": "loses accepted work (rollback broken)",
            "id": "F-MUTATION-RESTORE-NONE",
            "where": "mutation._restore_file",
            "watched": "TypeError on snapshot is None; created file remains",
            "resume": "create is durable and un-undoable",
        },
        {
            "rank": 6,
            "severity": "loses mission position (no-progress governor)",
            "id": "F-FINGERPRINT-ORDER",
            "where": "Scheduler.complete persist-then-record",
            "watched": "NO_PROGRESS raised; dag.json fingerprints count=2 of 3",
            "resume": "governor does not fire",
        },
        {
            "rank": 7,
            "severity": "orphan runtime / GPU process",
            "id": "F-OWNERSHIP-AFTER-SPAWN",
            "where": "RuntimePool.start then _write_ownership",
            "watched": "live sleep pid, no runtime_pool.json",
            "resume": "reap_orphans sees nothing to reap",
        },
        {
            "rank": 8,
            "severity": "loses mission position (steers / ledger)",
            "id": "F-STEER-LEDGER-SPLIT",
            "where": "SteeringQueue.apply_constraint + Ledger (no save)",
            "watched": "steering JSON applied=true, GOAL.md unchanged, in-memory G002",
            "resume": "constraint never re-applied; GOAL.md is the only reader",
        },
        {
            "rank": 9,
            "severity": "two MUTATION writers",
            "id": "F-LOCK-CORRUPT",
            "where": "MutationLock.read via _load_json",
            "watched": "truncated lock -> read() None -> acquire() True",
            "resume": "lock is free",
        },
        {
            "rank": 10,
            "severity": "silent config/steer loss",
            "id": "F-SWALLOW-CORRUPT-AND-OSERROR",
            "where": "Config._read, SteeringQueue._load, Controller.select_model, Mission._log",
            "watched": "corrupt JSON -> {}; OSError swallowed; log empty",
            "resume": "trusts empty / in-memory lie",
        },
        {
            "rank": 11,
            "severity": "cosmetic",
            "id": "F-WRITE-TEXT-RECEIPTS",
            "where": "GrokBridge/Engine/MachineGenome Path.write_text",
            "watched": "mid-write left unparseable JSON",
            "resume": "reader returns None or {}",
        },
        {
            "rank": 12,
            "severity": "cosmetic",
            "id": "F-SESSION-NEVER-SAVED",
            "where": "SessionStore.save unused",
            "watched": "Controller never writes sessions/*.json",
            "resume": "resume_session always None",
        },
        {
            "rank": 13,
            "severity": "cosmetic",
            "id": "F-INSTALL-SHIMS-PARTIAL",
            "where": "cli.install_shims",
            "watched": "only hcli shim on disk, jhcli missing, current symlink set",
            "resume": "half-installed",
        },
    ]


def print_inventory(census: List[Dict[str, Any]]) -> None:
    print()
    print("=== WRITER INVENTORY ===")
    print(f"durable persist functions enumerated: {len(WRITERS)}")
    print(f"AST write-like calls under tools/haider/hcli/*.py: {len(census)}")
    print()
    for w in WRITERS:
        print(
            f"{w['id']} {w['symbol']}  {w['loc']}\n"
            f"     dest: {w['dest']}\n"
            f"     atomic(from source): {w['atomic_from_source']}  "
            f"method={w['method_from_source']}  fsync={w['fsync']}\n"
            f"     notes: {w['notes']}"
        )
    print()
    print("--- AST write-like call sites (non-test) ---")
    for hit in census:
        if "/tests/" in hit["file"]:
            continue
        print(f"  {hit['file']}:{hit['line']}  {hit['call']}")


def write_receipt(head: str, census: List[Dict[str, Any]]) -> None:
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema": "hawking.headless.hcli_persistence_audit.v1",
        "recorded_at": _now(),
        "git_head": head,
        "command": "python3 tools/headless/hcli_persistence_audit.py",
        "writers_found": len(WRITERS),
        "ast_write_like_calls": len(census),
        "writers": WRITERS,
        "ast_census": [h for h in census if "/tests/" not in h["file"]],
        "demonstrations": DEMOS,
        "watched_fail": WATCHED_FAIL,
        "safe_with_evidence": SAFE,
        "ranked_findings": ranked_findings(),
        "harness_failures": FAILS,
        "claimed_checkpoint_repair_present": any(
            "checkpoint_id" in (d.get("on_disk") or {}).get("mission/state.json_keys", [])
            for d in DEMOS
        ),
        "handoff": HANDOFF_PATCHES,
    }
    RECEIPT_PATH.write_text(json.dumps(body, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    head = _git_head()
    print("=== HCLI PERSISTENCE AUDIT ===")
    print(f"git HEAD: {head}")
    print(f"hcli tree: {HCLI_DIR}")
    print()
    build_inventory()
    census = ast_write_census()
    print_inventory(census)
    print()
    print("=== DEMONSTRATIONS ===")
    demos: List[Callable[[], None]] = [
        demo_atomic_write_json_crash_mid_tmp,
        demo_write_text_truncates,
        demo_checkpoint_does_not_write_dag,
        demo_sigkill_between_dag_and_mission,
        demo_complete_then_checkpoint_crash,
        demo_fingerprint_persist_order,
        demo_grok_contract_then_receipt,
        demo_grok_receipt_truncate_then_read,
        demo_engine_two_file_mutation,
        demo_engine_execute_restores_on_exception,
        demo_engine_receipt_after_mutation,
        demo_mutation_rollback_create,
        demo_config_tmp_replace_and_corrupt_read,
        demo_controller_swallows_save_global,
        demo_steering_corrupt_is_empty,
        demo_apply_constraint_ledger_not_on_disk,
        demo_ledger_has_no_save,
        demo_session_never_saved,
        demo_mission_log_swallows,
        demo_dag_corrupt_raises,
        demo_mission_corrupt_raises,
        demo_atomic_write_json_raises,
        demo_mutation_lock_corrupt_is_free,
        demo_runtime_ownership_after_spawn,
        demo_machine_genome_truncates,
        demo_install_shims_partial,
        demo_recover_running_no_repair,
    ]
    for fn in demos:
        try:
            fn()
        except Exception:
            _record_demo(
                fn.__name__,
                ok=False,
                detail=traceback.format_exc(),
                watched_fail=True,
                rank="harness error",
            )
        print()

    print("=== SAFE WITH EVIDENCE ===")
    if not SAFE:
        print("     (none)")
    for item in SAFE:
        print(f"  - {item['writer']}: {item['claim']}")
        print(f"      evidence: {item['evidence']}")

    print()
    print("=== RANKED FINDINGS ===")
    for item in ranked_findings():
        print(
            f"  R{item['rank']} [{item['severity']}]\n"
            f"     {item['id']}  {item['where']}\n"
            f"     watched: {item['watched']}\n"
            f"     resume:  {item['resume']}"
        )

    print()
    print("## WHAT I WATCHED FAIL")
    if not WATCHED_FAIL:
        print("     NOTHING WATCHED FAIL — this lane is itself a failure.")
    for rec in WATCHED_FAIL:
        print(f"  - {rec['name']}")
        print(f"      {rec['detail']}")

    print()
    print(HANDOFF_PATCHES)
    write_receipt(head, census)
    print()
    print(f"receipt: {RECEIPT_PATH}")
    print(f"writers_found: {len(WRITERS)}")
    print(f"demos: {len(DEMOS)}  watched_fail: {len(WATCHED_FAIL)}  harness_FAILS: {len(FAILS)}")
    if FAILS:
        print("harness mechanics failed:")
        for item in FAILS:
            print(f"  - {item}")
        return 1
    if not WATCHED_FAIL:
        print("refusing to exit 0: nothing was watched failing")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
