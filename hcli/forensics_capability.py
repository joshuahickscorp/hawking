"""FORENSICS as a capability: preserve evidence before cleanup, then read it.

S036 s48: after an unexpected failure, preserve the relevant evidence BEFORE
cleanup -- logs, recent mutations, artifact hashes, process/resource state --
then diagnose. A daemon that destroys failure evidence cannot improve.

The recon found the primitives scattered and uncomposed: git history, receipts,
the event-sink JSONL, file hashing, process listing. This composes them into one
TRIAGE snapshot: a compact, timestamped record of "what was the world when this
broke", written to disk so it survives the cleanup that follows. It reads the
primitives that already exist; it introduces no new capture mechanism.

WHY COMPOSE RATHER THAN LEAVE SCATTERED. In an incident the operator does not
want to remember to grab six things by hand in the right order before the
worktree is reset. One call captures the set, atomically, with hashes, so the
snapshot itself is verifiable evidence rather than a story about the evidence.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _git(root: Path, *args: str) -> str:
    try:
        done = subprocess.run(["git", "-C", str(root), *args],
                              capture_output=True, text=True, timeout=8)
        return done.stdout.strip() if done.returncode == 0 else ""
    except Exception:
        return ""


def _hash_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


@dataclass
class Snapshot:
    """The world at the moment something broke."""

    at: float
    workspace: str
    head: str = ""
    branch: str = ""
    dirty: List[str] = field(default_factory=list)
    recent_commits: List[str] = field(default_factory=list)
    changed_hashes: Dict[str, str] = field(default_factory=dict)
    recent_events: List[str] = field(default_factory=list)
    note: str = ""
    path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        lines = [f"FORENSIC SNAPSHOT at {int(self.at)}  {self.workspace}"]
        if self.head:
            lines.append(f"  HEAD {self.head} ({self.branch})")
        if self.dirty:
            lines.append(f"  DIRTY {len(self.dirty)} file(s): "
                         + ", ".join(self.dirty[:8]))
        if self.recent_events:
            lines.append(f"  LAST EVENT {self.recent_events[-1][:100]}")
        if self.path:
            lines.append(f"  preserved at {self.path}")
        return "\n".join(lines)


def capture(workspace: str, *, note: str = "",
            reason: str = "incident") -> Snapshot:
    """Preserve the failure context, then return it. Writes before it returns.

    Never raises into the caller -- forensics that crashes during an incident is
    worse than none. Every field degrades to empty rather than failing the
    snapshot.
    """
    root = Path(workspace)
    snap = Snapshot(at=time.time(), workspace=str(root), note=note)

    snap.head = _git(root, "rev-parse", "--short", "HEAD")
    snap.branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(root, "status", "--porcelain")
    # _git() strips output, so a porcelain line " M path" arrives as "M path"
    # and a fixed line[3:] then eats the first path char. Split off the 1-2
    # status columns by the first whitespace run instead, and take the rename
    # destination when present.
    def _porcelain_path(line: str) -> str:
        rest = line.split(None, 1)
        tail = rest[1] if len(rest) == 2 else line
        return tail.split(" -> ")[-1].strip().strip('"')
    snap.dirty = [_porcelain_path(line)
                  for line in status.splitlines() if line.strip()][:60]
    log = _git(root, "log", "--oneline", "-8")
    snap.recent_commits = log.splitlines()

    # hash the dirty files: what did the world look like, verifiably
    for rel in snap.dirty[:30]:
        path = root / rel
        if path.is_file():
            digest = _hash_file(path)
            if digest:
                snap.changed_hashes[rel] = digest

    events = root / ".hcli" / "mission" / "events.jsonl"
    if events.is_file():
        try:
            tail = events.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            snap.recent_events = tail
        except OSError:
            pass

    # persist BEFORE returning, so cleanup after this call cannot lose it
    out = root / ".hcli" / "forensics" / f"snapshot-{int(snap.at)}-{reason}.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(snap.to_dict(), indent=1, default=str),
                       encoding="utf-8")
        snap.path = str(out)
    except OSError:
        snap.note = (snap.note + " (could not persist)").strip()
    return snap


# --- REVERSE: what IS this file, without running it -------------------------

def identify(path: str) -> Dict[str, Any]:
    """Classify a file by its bytes -- format, and strings a human would read.

    The basic rung of REVERSE (S036 s46): metadata and magic-byte
    classification, not disassembly. Reuses the existing file_eye classifier the
    perception package already ships; falls back to a small local table if it is
    not importable, so the capability is real either way.
    """
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "error": "not a file"}
    try:
        head = p.read_bytes()[:64]
    except OSError as exc:
        return {"path": str(p), "error": str(exc)}
    kind = _magic(head)
    result: Dict[str, Any] = {
        "path": str(p),
        "bytes": p.stat().st_size,
        "kind": kind,
        "sha256_16": _hash_file(p),
    }
    try:
        from .agentos.vmcp.file_eye import classify_bytes  # type: ignore
        deep = classify_bytes(head)
        if isinstance(deep, dict):
            result["classification"] = deep
    except Exception:
        pass
    return result


_MAGIC = (
    (b"\x7fELF", "ELF executable"),
    (b"\xcf\xfa\xed\xfe", "Mach-O 64-bit"),
    (b"\xfe\xed\xfa\xce", "Mach-O 32-bit"),
    (b"MZ", "PE/DOS executable"),
    (b"\x89PNG", "PNG image"),
    (b"%PDF", "PDF"),
    (b"PK\x03\x04", "zip/jar/wheel"),
    (b"\x1f\x8b", "gzip"),
    (b"SQLite format 3", "sqlite database"),
    (b"\x00asm", "wasm module"),
)


def _magic(head: bytes) -> str:
    for prefix, name in _MAGIC:
        if head.startswith(prefix):
            return name
    try:
        head.decode("utf-8")
        return "utf-8 text"
    except UnicodeDecodeError:
        return "binary (unrecognised)"
