#!/usr/bin/env python3
"""Operator source GC watcher for the doctor_v5_ultra campaign.

Deletes a model's source staging dir only after the model is fully terminal.
All gates fail closed. This tool is the separate operator action referenced by
the campaign plan field parent_source_cleanup=disabled_separate_operator_action_only;
no campaign document authorizes deletion and none is reinterpreted as doing so.

Subcommands:
  run-once   evaluate gates for every configured model; receipt then delete
  status     print gate evaluation as JSON; no side effects
  selftest   pure synthetic-fixture battery in a tempdir; no deletion of real data
  arm        install LaunchAgent com.hawking.doctorv5ultra.source-gc (StartInterval 300)
  disarm     boot out and remove that LaunchAgent

Never signals or inspects campaign processes. Never writes campaign control files.
Writes only: source_gc/<model>_<utcstamp>.json receipts, source_gc/ledger.jsonl,
source_gc/gc.lock, and (arm only) its own LaunchAgent plist.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as _dt
import fcntl
import hashlib
import json
import os
import plistlib
import secrets
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, IO

VERSION = 1
STATE_SCHEMA = "hawking.doctor_v5_ultra_queue_state.v1"
RECEIPT_SCHEMA = "hawking.operator_source_gc_receipt.v1"
LEDGER_SCHEMA = "hawking.operator_source_gc_ledger_entry.v1"
REPORTER_SYNC_SCHEMA = "hawking.doctor_v5_ultra_reporter_sync.v1"
LAUNCH_LABEL = "com.hawking.doctorv5ultra.source-gc"
LAUNCH_INTERVAL_SECONDS = 300
TERMINAL = frozenset({"complete", "negative", "unsupported"})
CAMPAIGN_TOTAL_CELLS = 320
HEAD_TAIL_THRESHOLD = 256 * 1024 * 1024
HEAD_TAIL_WINDOW = 64 * 1024 * 1024
MAX_SMALL_JSON = 64 * 1024 * 1024
MAX_LEDGER_BYTES = 64 * 1024 * 1024
MAX_SYNC_LOG_BYTES = 512 * 1024 * 1024

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PYTHON = Path(sys.executable).resolve()

AUTHORITY = (
    "operator-directed model-terminal source GC; "
    "plan parent_source_cleanup=disabled_separate_operator_action_only; "
    "this run is that separate operator action"
)


class GcError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Config:
    root: Path
    campaign_dir: Path
    staging_root: Path
    staging_map: dict[str, Path]
    transaction_paths: tuple[Path, ...]
    campaign_total_cells: int = CAMPAIGN_TOTAL_CELLS
    full_campaign_models: frozenset[str] = frozenset({"gpt-oss-120b"})

    @property
    def queue_state_path(self) -> Path:
        return self.campaign_dir / "queue_state.json"

    @property
    def reporter_sync_path(self) -> Path:
        return self.campaign_dir / "reporter_sync.jsonl"

    @property
    def plan_path(self) -> Path:
        return self.campaign_dir / "campaign_plan.json"

    @property
    def gc_dir(self) -> Path:
        return self.campaign_dir / "source_gc"

    @property
    def ledger_path(self) -> Path:
        return self.gc_dir / "ledger.jsonl"

    @property
    def lock_path(self) -> Path:
        return self.gc_dir / "gc.lock"


def default_config() -> Config:
    campaign = ROOT / "reports" / "condense" / "doctor_v5_ultra"
    staging_root = ROOT / "scratch" / "staging"
    stage = campaign / "staged_acceleration"
    return Config(
        root=ROOT,
        campaign_dir=campaign,
        staging_root=staging_root,
        staging_map={
            "qwen2-5-14b": staging_root / "qwen-14b.partial",
            "qwen2-5-32b": staging_root / "qwen-32b.partial",
            "qwen2-5-72b": staging_root / "qwen-72b.partial",
            "gpt-oss-120b": staging_root / "gpt-oss-120b.partial",
        },
        transaction_paths=(
            stage / "retry_ceiling_disposition_reconciler_v2" / "transaction.json",
            stage / "evidence_closure_rg2" / "transaction.json",
        ),
    )


# ---------------------------------------------------------------- io helpers

def _stable_bytes(path: Path, *, ceiling: int = MAX_SMALL_JSON) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > ceiling:
            raise GcError(f"unsafe or oversized regular file: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(fd, min(remaining, 1024 * 1024))
            if not block:
                raise GcError(f"short read: {path}")
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_mode, row.st_size, row.st_mtime_ns
        )
        if identity(before) != identity(after) or len(raw) != after.st_size:
            raise GcError(f"file changed while reading: {path}")
        return raw
    finally:
        os.close(fd)


def _read_json(path: Path, *, ceiling: int = MAX_SMALL_JSON) -> dict[str, Any]:
    try:
        value = json.loads(_stable_bytes(path, ceiling=ceiling).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GcError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GcError(f"JSON object required: {path}")
    return value


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path: Path, raw: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(temporary, flags, mode)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _hash_value(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_hashed(path: Path, value: dict[str, Any], hash_field: str) -> dict[str, Any]:
    output = copy.deepcopy(value)
    output.pop(hash_field, None)
    output[hash_field] = _hash_value(output)
    raw = json.dumps(
        output, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, raw)
    return output


def _open_lock(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise GcError(f"lock unavailable: {path}")
    return handle


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _parse_iso(raw: Any, label: str) -> _dt.datetime:
    if not isinstance(raw, str):
        raise GcError(f"timestamp missing: {label}")
    try:
        value = _dt.datetime.fromisoformat(raw)
    except ValueError as exc:
        raise GcError(f"timestamp invalid: {label}: {raw!r}") from exc
    if value.tzinfo is None:
        raise GcError(f"timestamp naive: {label}: {raw!r}")
    return value


def _self_sha256() -> str:
    return hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


# --------------------------------------------------------------- file hashing

def _hash_regular_file(path: Path) -> dict[str, Any]:
    """Hash-identity row for one regular file. Head-tail 64M for large files."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise GcError(f"not a regular file: {path}")
        size = info.st_size
        digest = hashlib.sha256()

        def _read_span(offset: int, length: int) -> None:
            os.lseek(fd, offset, os.SEEK_SET)
            remaining = length
            while remaining:
                block = os.read(fd, min(remaining, 8 * 1024 * 1024))
                if not block:
                    raise GcError(f"short read: {path}")
                digest.update(block)
                remaining -= len(block)

        if size > HEAD_TAIL_THRESHOLD:
            _read_span(0, HEAD_TAIL_WINDOW)
            _read_span(size - HEAD_TAIL_WINDOW, HEAD_TAIL_WINDOW)
            key = "sha256_head_tail_64m"
        else:
            _read_span(0, size)
            key = "sha256"
        after = os.fstat(fd)
        if (after.st_size, after.st_mtime_ns) != (info.st_size, info.st_mtime_ns):
            raise GcError(f"file changed while hashing: {path}")
        mtime = _dt.datetime.fromtimestamp(
            info.st_mtime_ns / 1e9, _dt.timezone.utc
        ).isoformat()
        return {"bytes": size, "mtime": mtime, key: digest.hexdigest()}
    finally:
        os.close(fd)


def _inventory(config: Config, directory: Path) -> tuple[list[dict[str, Any]], int]:
    """Walk directory, refuse symlinks and non-regular entries, hash every file."""
    rows: list[dict[str, Any]] = []
    total = 0
    stack = [directory]
    while stack:
        current = stack.pop()
        if current.is_symlink():
            raise GcError(f"symlink refused: {current}")
        with os.scandir(current) as entries:
            for entry in sorted(entries, key=lambda e: e.path):
                path = Path(entry.path)
                if entry.is_symlink():
                    raise GcError(f"symlink refused: {path}")
                if entry.is_dir(follow_symlinks=False):
                    stack.append(path)
                elif entry.is_file(follow_symlinks=False):
                    row = _hash_regular_file(path)
                    row["path"] = str(path.relative_to(config.root))
                    rows.append(row)
                    total += row["bytes"]
                else:
                    raise GcError(f"non-regular entry refused: {path}")
    rows.sort(key=lambda row: row["path"])
    return rows, total


def _delete_tree(directory: Path) -> None:
    """Bottom-up unlink+rmdir; refuses symlinks; never leaves the tree."""
    if directory.is_symlink():
        raise GcError(f"symlink refused: {directory}")
    for current, dirnames, filenames in os.walk(directory, topdown=False):
        base = Path(current)
        for name in filenames:
            path = base / name
            if path.is_symlink():
                raise GcError(f"symlink refused: {path}")
            os.unlink(path)
        for name in dirnames:
            path = base / name
            if path.is_symlink():
                raise GcError(f"symlink refused: {path}")
            os.rmdir(path)
    os.rmdir(directory)
    _fsync_dir(directory.parent)


# -------------------------------------------------------------------- gates

def _load_queue_state(config: Config) -> dict[str, Any]:
    state = _read_json(config.queue_state_path)
    if state.get("schema") != STATE_SCHEMA:
        raise GcError(
            f"queue_state schema mismatch: {state.get('schema')!r} != {STATE_SCHEMA!r}"
        )
    cells = state.get("cells")
    if not isinstance(cells, dict) or not cells:
        raise GcError("queue_state cells missing or empty")
    for cell_id, row in cells.items():
        if not isinstance(row, dict) or not isinstance(row.get("status"), str):
            raise GcError(f"queue_state cell malformed: {cell_id}")
    return state


def _model_of(cell_id: str) -> str:
    return cell_id.split("__", 1)[0]


def _reporter_terminal_ok_cells(config: Config) -> set[str]:
    """Cell ids with an ok terminal reporter sync line. Corrupt lines refuse."""
    path = config.reporter_sync_path
    if not path.exists():
        raise GcError(f"reporter sync log missing: {path}")
    if path.is_symlink():
        raise GcError(f"symlink refused: {path}")
    if path.stat().st_size > MAX_SYNC_LOG_BYTES:
        raise GcError(f"reporter sync log oversized: {path}")
    sealed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GcError(f"reporter sync line {number} invalid JSON") from exc
            if not isinstance(entry, dict):
                raise GcError(f"reporter sync line {number} not an object")
            reason = entry.get("reason")
            if (
                isinstance(reason, str)
                and reason.startswith("terminal:")
                and entry.get("ok") is True
            ):
                sealed.add(reason[len("terminal:"):])
    return sealed


def _transaction_gate(config: Config) -> list[str]:
    reasons: list[str] = []
    for path in config.transaction_paths:
        if not path.exists():
            continue
        try:
            journal = _read_json(path)
        except GcError as exc:
            reasons.append(f"transaction unreadable: {path.name}: {exc}")
            continue
        phase = journal.get("phase")
        if phase != "complete":
            reasons.append(
                f"transaction in flight: {path.parent.name}/{path.name} phase={phase!r}"
            )
    return reasons


def evaluate_model(
    config: Config,
    model: str,
    state: dict[str, Any],
    sealed_cells: set[str],
) -> dict[str, Any]:
    """Full fail-closed gate stack for one model. Returns a status dict."""
    reasons: list[str] = []
    cells: dict[str, Any] = state["cells"]
    model_cells = {cid: row for cid, row in cells.items() if _model_of(cid) == model}
    directory = config.staging_map[model]

    if not model_cells:
        reasons.append(f"no cells for model {model}")

    non_terminal = sorted(
        cid for cid, row in model_cells.items() if row.get("status") not in TERMINAL
    )
    if non_terminal:
        reasons.append(
            f"non-terminal cells: {len(non_terminal)} (first {non_terminal[0]})"
        )

    if model in config.full_campaign_models:
        total = len(cells)
        campaign_non_terminal = sum(
            1 for row in cells.values() if row.get("status") not in TERMINAL
        )
        if total != config.campaign_total_cells:
            reasons.append(
                f"campaign cell count {total} != {config.campaign_total_cells}"
            )
        if campaign_non_terminal:
            reasons.append(
                f"campaign not sealed: {campaign_non_terminal} non-terminal cells "
                f"(hard gate for {model})"
            )

    active_cells = state.get("active_cells")
    if not isinstance(active_cells, list):
        reasons.append("active_cells missing")
    else:
        hits = [c for c in active_cells if isinstance(c, str) and _model_of(c) == model]
        if hits:
            reasons.append(f"active cell references model: {hits[0]}")

    active_children = state.get("active_children")
    if not isinstance(active_children, dict):
        reasons.append("active_children missing")
    else:
        for key, child in active_children.items():
            names = {key}
            if isinstance(child, dict) and isinstance(child.get("cell_id"), str):
                names.add(child["cell_id"])
            if any(_model_of(name) == model for name in names if isinstance(name, str)):
                reasons.append(f"active child references model: {key}")
                break

    terminal_cells = {
        cid: row for cid, row in model_cells.items() if row.get("status") in TERMINAL
    }
    unsealed = sorted(cid for cid in terminal_cells if cid not in sealed_cells)
    if unsealed:
        reasons.append(
            f"reporter not sealed for {len(unsealed)} terminal cells "
            f"(first {unsealed[0]})"
        )

    last_sync = state.get("last_reporter_sync")
    if not isinstance(last_sync, dict) or last_sync.get("ok") is not True:
        reasons.append("last_reporter_sync missing or not ok")
    else:
        completed = [
            row.get("completed_at")
            for row in terminal_cells.values()
            if row.get("completed_at")
        ]
        if terminal_cells and len(completed) != len(terminal_cells):
            reasons.append("terminal cell missing completed_at")
        elif completed:
            try:
                latest = max(_parse_iso(raw, "completed_at") for raw in completed)
                sync_at = _parse_iso(last_sync.get("at"), "last_reporter_sync.at")
                if sync_at < latest:
                    reasons.append(
                        "last_reporter_sync predates model last completed_at"
                    )
            except GcError as exc:
                reasons.append(str(exc))

    reasons.extend(_transaction_gate(config))

    dir_exists = directory.exists() or directory.is_symlink()
    if dir_exists:
        if directory.is_symlink():
            reasons.append(f"staging dir is a symlink: {directory}")
        elif not directory.is_dir():
            reasons.append(f"staging path not a directory: {directory}")
        try:
            resolved = directory.resolve(strict=False)
            staging_root = config.staging_root.resolve(strict=False)
            if resolved.parent != staging_root:
                reasons.append(f"staging dir outside staging root: {resolved}")
        except OSError as exc:
            reasons.append(f"staging dir unresolvable: {exc}")

    return {
        "model": model,
        "dir": str(directory),
        "dir_exists": dir_exists,
        "cell_count": len(model_cells),
        "terminal_count": len(terminal_cells),
        "eligible": dir_exists and not reasons,
        "already_gone": not dir_exists,
        "reasons": reasons,
    }


# ------------------------------------------------------------------- ledger

def _read_ledger_entries(config: Config) -> list[dict[str, Any]]:
    path = config.ledger_path
    if not path.exists():
        return []
    if path.is_symlink():
        raise GcError(f"symlink refused: {path}")
    if path.stat().st_size > MAX_LEDGER_BYTES:
        raise GcError(f"ledger oversized: {path}")
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GcError(f"ledger line {number} invalid JSON") from exc
            if not isinstance(entry, dict):
                raise GcError(f"ledger line {number} not an object")
            entries.append(entry)
    return entries


def verify_ledger(config: Config) -> list[dict[str, Any]]:
    """Verify every entry self-hash and the prev-hash chain. Fail closed."""
    entries = _read_ledger_entries(config)
    previous: str | None = None
    for index, entry in enumerate(entries, start=1):
        if entry.get("schema") != LEDGER_SCHEMA:
            raise GcError(f"ledger entry {index} schema mismatch")
        claimed = entry.get("entry_sha256")
        body = {k: v for k, v in entry.items() if k != "entry_sha256"}
        if not isinstance(claimed, str) or _hash_value(body) != claimed:
            raise GcError(f"ledger entry {index} self-hash mismatch")
        if entry.get("prev_entry_sha256") != previous:
            raise GcError(f"ledger entry {index} chain break")
        previous = claimed
    return entries


def _append_ledger(config: Config, body: dict[str, Any]) -> dict[str, Any]:
    entries = verify_ledger(config)
    previous = entries[-1]["entry_sha256"] if entries else None
    entry = dict(body)
    entry["schema"] = LEDGER_SCHEMA
    entry["prev_entry_sha256"] = previous
    entry["entry_sha256"] = _hash_value(entry)
    raw = json.dumps(
        entry, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    config.gc_dir.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(config.ledger_path, flags, 0o600)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(config.gc_dir)
    verify_ledger(config)
    return entry


# ------------------------------------------------------------------ actions

def _plan_sha256(config: Config) -> str | None:
    try:
        raw = _stable_bytes(config.plan_path, ceiling=MAX_SMALL_JSON)
    except (FileNotFoundError, GcError):
        return None
    return hashlib.sha256(raw).hexdigest()


def gc_model(config: Config, model: str, verdict: dict[str, Any]) -> dict[str, Any]:
    """Receipt-then-delete for one eligible model. Caller holds the gc lock."""
    directory = config.staging_map[model]
    files, total = _inventory(config, directory)
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    receipt_path = config.gc_dir / f"{model}_{stamp}.json"
    serial = 1
    while receipt_path.exists():
        serial += 1
        receipt_path = config.gc_dir / f"{model}_{stamp}_{serial}.json"
    document = {
        "schema": RECEIPT_SCHEMA,
        "authority": AUTHORITY,
        "model": model,
        "campaign_plan_sha256": _plan_sha256(config),
        "quality_claims_permitted": False,
        "recorded_at": _utc_now().isoformat(),
        "gates": {k: v for k, v in verdict.items() if k != "reasons"},
        "deleted": [
            {
                "dir": str(directory.relative_to(config.root)),
                "file_count": len(files),
                "files": files,
                "total_bytes": total,
            }
        ],
    }
    written = _write_hashed(receipt_path, document, "receipt_sha256")
    _delete_tree(directory)
    entry = _append_ledger(
        config,
        {
            "at": _utc_now().isoformat(),
            "model": model,
            "dir": str(directory.relative_to(config.root)),
            "receipt_path": str(receipt_path.relative_to(config.root)),
            "receipt_sha256": written["receipt_sha256"],
            "file_count": len(files),
            "total_bytes": total,
        },
    )
    return {
        "model": model,
        "deleted": True,
        "receipt": str(receipt_path),
        "receipt_sha256": written["receipt_sha256"],
        "ledger_entry_sha256": entry["entry_sha256"],
        "file_count": len(files),
        "total_bytes": total,
    }


def run_once(config: Config) -> dict[str, Any]:
    lock = _open_lock(config.lock_path)
    try:
        state = _load_queue_state(config)
        sealed = _reporter_terminal_ok_cells(config)
        outcomes: dict[str, Any] = {}
        for model in sorted(config.staging_map):
            verdict = evaluate_model(config, model, state, sealed)
            if verdict["already_gone"]:
                outcomes[model] = {"model": model, "deleted": False, "skip": "dir gone"}
            elif verdict["eligible"]:
                outcomes[model] = gc_model(config, model, verdict)
            else:
                outcomes[model] = {
                    "model": model,
                    "deleted": False,
                    "refused": True,
                    "reasons": verdict["reasons"],
                }
        deleted = [m for m, o in outcomes.items() if o.get("deleted")]
        freed = sum(o.get("total_bytes", 0) for o in outcomes.values())
        summary = (
            f"source-gc run-once deleted={len(deleted)} "
            f"models={','.join(deleted) if deleted else 'none'} freed_bytes={freed}"
        )
        return {"ok": True, "summary": summary, "outcomes": outcomes}
    finally:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock.close()


def status(config: Config) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema": "hawking.operator_source_gc_status.v1",
        "at": _utc_now().isoformat(),
        "launch_label": LAUNCH_LABEL,
        "self_sha256": _self_sha256(),
        "models": {},
    }
    try:
        state = _load_queue_state(config)
        sealed = _reporter_terminal_ok_cells(config)
        for model in sorted(config.staging_map):
            output["models"][model] = evaluate_model(config, model, state, sealed)
    except (GcError, FileNotFoundError, OSError) as exc:
        output["error"] = str(exc)
    try:
        output["ledger_entries"] = len(verify_ledger(config))
        output["ledger_ok"] = True
    except GcError as exc:
        output["ledger_ok"] = False
        output["ledger_error"] = str(exc)
    return output


# ----------------------------------------------------------------- launchd

def _launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_LABEL}.plist"


def _launch_agent_document(source_sha256: str, config: Config) -> bytes:
    value = {
        "Label": LAUNCH_LABEL,
        "ProgramArguments": [
            str(PYTHON), str(SCRIPT), "run-once",
            "--expected-self-sha256", source_sha256,
        ],
        "WorkingDirectory": str(config.root),
        "StartInterval": LAUNCH_INTERVAL_SECONDS,
        "RunAtLoad": False,
        "ProcessType": "Background",
        "StandardOutPath": str(config.gc_dir / "launch_agent.log"),
        "StandardErrorPath": str(config.gc_dir / "launch_agent.log"),
    }
    return plistlib.dumps(value, fmt=plistlib.FMT_XML, sort_keys=True)


def arm(config: Config) -> dict[str, Any]:
    plist_path = _launch_agent_path()
    if plist_path.exists():
        raise GcError(f"refusing to overwrite existing LaunchAgent: {plist_path}")
    source_sha256 = _self_sha256()
    _atomic_bytes(plist_path, _launch_agent_document(source_sha256, config), mode=0o600)
    target = f"gui/{os.getuid()}/{LAUNCH_LABEL}"
    process = subprocess.run(
        ["/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if process.returncode != 0:
        raise GcError(
            f"launchctl bootstrap failed ({process.returncode}): "
            + (process.stderr or process.stdout)[-500:]
        )
    return {
        "ok": True,
        "label": LAUNCH_LABEL,
        "plist": str(plist_path),
        "target": target,
        "interval_seconds": LAUNCH_INTERVAL_SECONDS,
        "pinned_self_sha256": source_sha256,
    }


def disarm() -> dict[str, Any]:
    plist_path = _launch_agent_path()
    process = subprocess.run(
        ["/bin/launchctl", "bootout", f"gui/{os.getuid()}/{LAUNCH_LABEL}"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    booted = process.returncode == 0
    removed = False
    if plist_path.exists():
        plist_path.unlink()
        removed = True
    return {"ok": True, "booted_out": booted, "plist_removed": removed}


# ---------------------------------------------------------------- selftest

def _make_fixture(
    base: Path,
    *,
    model: str = "qwen2-5-14b",
    statuses: list[str] | None = None,
    transaction_phase: str | None = None,
    reporter_ok: bool = True,
    campaign_total: int = 4,
) -> Config:
    """Synthetic campaign fixture in base. Never touches the real campaign."""
    campaign = base / "campaign"
    staging_root = base / "scratch" / "staging"
    stage = campaign / "staged_acceleration" / "retry_ceiling_disposition_reconciler_v2"
    campaign.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    statuses = statuses or ["complete", "complete", "negative", "unsupported"]
    cells: dict[str, Any] = {}
    sync_lines: list[str] = []
    for index, cell_status in enumerate(statuses):
        cell_id = f"{model}__{index}bpw__codec-control"
        cells[cell_id] = {
            "status": cell_status,
            "completed_at": (
                "2026-07-16T10:00:00+00:00" if cell_status in TERMINAL else None
            ),
        }
        if cell_status in TERMINAL and reporter_ok:
            sync_lines.append(json.dumps({
                "schema": REPORTER_SYNC_SCHEMA,
                "ok": True,
                "reason": f"terminal:{cell_id}",
                "at": "2026-07-16T11:00:00+00:00",
            }))
    while len(cells) < campaign_total:
        cell_id = f"filler-model__{len(cells)}bpw__codec-control"
        cells[cell_id] = {"status": "complete", "completed_at": "2026-07-16T10:00:00+00:00"}
        sync_lines.append(json.dumps({
            "schema": REPORTER_SYNC_SCHEMA, "ok": True,
            "reason": f"terminal:{cell_id}", "at": "2026-07-16T11:00:00+00:00",
        }))
    state = {
        "schema": STATE_SCHEMA,
        "cells": cells,
        "active_cells": [],
        "active_children": {},
        "last_reporter_sync": {"ok": True, "at": "2026-07-16T12:00:00+00:00"},
    }
    (campaign / "queue_state.json").write_text(json.dumps(state), encoding="utf-8")
    (campaign / "reporter_sync.jsonl").write_text(
        "".join(line + "\n" for line in sync_lines), encoding="utf-8"
    )
    (campaign / "campaign_plan.json").write_text("{}", encoding="utf-8")
    if transaction_phase is not None:
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "transaction.json").write_text(
            json.dumps({"phase": transaction_phase}), encoding="utf-8"
        )
    directory = staging_root / f"{model}.partial"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "shard-000.bin").write_bytes(b"a" * 4096)
    (directory / "manifest.json").write_text("{}", encoding="utf-8")
    return Config(
        root=base,
        campaign_dir=campaign,
        staging_root=staging_root,
        staging_map={model: directory},
        transaction_paths=(stage / "transaction.json",),
        campaign_total_cells=campaign_total,
    )


def selftest() -> dict[str, Any]:
    checks: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            raise GcError(f"selftest failed: {name}")
        checks.append(name)

    with tempfile.TemporaryDirectory(prefix="source-gc-selftest-") as raw:
        base = Path(raw)

        config = _make_fixture(base / "a", statuses=["complete", "pending"])
        result = run_once(config)
        outcome = result["outcomes"]["qwen2-5-14b"]
        check("refuses-non-terminal", outcome.get("refused") is True)
        check("dir-survives-refusal", config.staging_map["qwen2-5-14b"].exists())

        config = _make_fixture(base / "b", transaction_phase="state-mutated")
        outcome = run_once(config)["outcomes"]["qwen2-5-14b"]
        check("refuses-transaction-in-flight", outcome.get("refused") is True)

        config = _make_fixture(
            base / "c", model="gpt-oss-120b", campaign_total=8
        )
        config = dataclasses.replace(
            config, campaign_total_cells=CAMPAIGN_TOTAL_CELLS
        )
        outcome = run_once(config)["outcomes"]["gpt-oss-120b"]
        check("refuses-120b-before-320", outcome.get("refused") is True)

        config = _make_fixture(base / "d", transaction_phase="complete")
        directory = config.staging_map["qwen2-5-14b"]
        outcome = run_once(config)["outcomes"]["qwen2-5-14b"]
        check("deletes-when-sealed", outcome.get("deleted") is True)
        check("dir-gone", not directory.exists())
        receipt = _read_json(Path(outcome["receipt"]))
        body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        check("receipt-self-hash", _hash_value(body) == receipt["receipt_sha256"])
        check("ledger-chain", len(verify_ledger(config)) == 1)
        second = run_once(config)["outcomes"]["qwen2-5-14b"]
        check("idempotent-skip", second.get("skip") == "dir gone")

        config = _make_fixture(base / "e")
        directory = config.staging_map["qwen2-5-14b"]
        (directory / "escape").symlink_to(base)
        refused = False
        try:
            run_once(config)
        except GcError:
            refused = True
        check("symlink-refusal", refused)
        check("symlink-dir-survives", directory.exists())

    return {"ok": True, "checks": checks}


# --------------------------------------------------------------------- cli

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    once = sub.add_parser("run-once", help="evaluate gates and GC eligible models")
    once.add_argument("--expected-self-sha256", default=None)
    sub.add_parser("status", help="print gate evaluation JSON")
    sub.add_parser("selftest", help="pure synthetic battery, no real deletion")
    sub.add_parser("arm", help="install the LaunchAgent")
    sub.add_parser("disarm", help="remove the LaunchAgent")
    args = parser.parse_args(argv)

    config = default_config()
    try:
        if args.command == "run-once":
            if args.expected_self_sha256 is not None:
                actual = _self_sha256()
                if actual != args.expected_self_sha256:
                    raise GcError(
                        f"self sha256 mismatch: {actual} != {args.expected_self_sha256}"
                    )
            result = run_once(config)
            print(result["summary"])
            print(json.dumps(result["outcomes"], sort_keys=True))
        elif args.command == "status":
            print(json.dumps(status(config), sort_keys=True, indent=1))
        elif args.command == "selftest":
            print(json.dumps(selftest(), sort_keys=True))
        elif args.command == "arm":
            print(json.dumps(arm(config), sort_keys=True))
        elif args.command == "disarm":
            print(json.dumps(disarm(), sort_keys=True))
    except GcError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
