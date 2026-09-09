"""What Agent OS can DO, named as capability, not as package.

THE KALI LESSON (S036 s3). A capability platform is not "a computer with 600
programs". The human thinks in missions; the packages are implementation. So
HCLI must perceive "I can inspect source", not "I have ripgrep" -- and when it
asks whether it can debug, the answer is a capability with an escalation ladder
underneath, not a version string.

This module is the registry of capability OWNERS. Each owner:

  * names a domain a mission is stated in -- INSPECT, RESEARCH, BUILD, TEST,
    PROFILE, PROCESS, RECOVER, and the security domains as they are earned;
  * carries an ESCALATION LADDER (S036 s11): the cheapest sufficient rung
    first, deeper rungs when evidence demands, so the daemon does not remain
    shallow when it must go deeper nor reach for the debugger to read a file;
  * reports whether it is REACHABLE right now, by probing the primitive it
    wraps rather than by asserting -- a capability nothing can call does not
    exist, and this file refuses to claim one that is not wired.

It adds NO new tool. Every rung points at machinery that already exists
(tool_registry, engine, processes, the profilers). The value is the coherent
surface: one place that answers "what can this body do, and how deep does each
sense go", which is what an operator needs and what a package list cannot give.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path


@dataclass
class Rung:
    """One level of a capability's escalation ladder."""

    name: str
    via: str                      # the primitive that serves this rung
    cost: str                     # cheap | moderate | expensive
    note: str = ""


@dataclass
class Capability:
    """A domain the body can act in, and how deep it can go."""

    domain: str
    verb: str                     # the plain-language "I can ..."
    ladder: List[Rung]
    probe: Optional[Callable[[], bool]] = None
    tools: List[str] = field(default_factory=list)

    def reachable(self) -> bool:
        """True only when the underlying primitive answers. Never asserted."""
        if self.probe is None:
            return bool(self.ladder)
        try:
            return bool(self.probe())
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "verb": self.verb,
            "reachable": self.reachable(),
            "ladder": [{"rung": r.name, "cost": r.cost, "note": r.note}
                       for r in self.ladder],
            "tools": self.tools,
        }


def _registry_has(*names: str) -> Callable[[], bool]:
    """A probe: the tool registry actually serves every named tool."""
    def probe() -> bool:
        try:
            from .chat_tools import build_registry
            registry = build_registry(".", ".")
            return all(registry.get(name) is not None for name in names)
        except Exception:
            return False
    return probe


def _engine_can_mutate() -> bool:
    """A probe: the mutation transaction exists and is callable."""
    try:
        from .engine import Engine
        return callable(getattr(Engine, "apply_typed_mutation", None))
    except Exception:
        return False


def _processes_reachable() -> bool:
    try:
        from . import processes
        return callable(getattr(processes, "live_processes", None))
    except Exception:
        return False


def _profiler_reachable() -> bool:
    try:
        import importlib
        importlib.import_module("hcli.prefill_profile")
        return True
    except Exception:
        return False


def _debug_reachable() -> bool:
    try:
        return callable(diagnose)
    except Exception:
        return False


def _fuzz_reachable() -> bool:
    try:
        return callable(fuzz)
    except Exception:
        return False


def _forensics_reachable() -> bool:
    try:
        return callable(capture) and callable(identify)
    except Exception:
        return False


def capabilities() -> List[Capability]:
    """The body's capability surface, probed against live machinery."""
    return [
        Capability(
            domain="INSPECT", verb="inspect source and repository",
            probe=_registry_has("fs.read", "fs.search", "fs.list"),
            tools=["fs.read", "fs.search", "fs.list"],
            ladder=[
                Rung("exact file", "fs.read", "cheap",
                     "when the path is known"),
                Rung("text search", "fs.search", "cheap",
                     "escalates to source glob when a search truncates"),
                Rung("directory map", "fs.list", "cheap", ""),
                Rung("structural / AST", "MISSING", "moderate",
                     "definitions and references not yet an owned rung"),
            ]),
        Capability(
            domain="RESEARCH", verb="research the public web",
            probe=_registry_has("web.search", "web.fetch"),
            tools=["web.search", "web.fetch", "github.search",
                   "huggingface.resolve"],
            ladder=[
                Rung("search", "web.search", "cheap",
                     "decodes engine redirect wrappers to real sources"),
                Rung("fetch one page", "web.fetch", "moderate", ""),
                Rung("crawl / browse", "MISSING", "expensive",
                     "no owned browser rung on the serve path yet"),
            ]),
        Capability(
            domain="BUILD", verb="mutate source under authority",
            probe=_engine_can_mutate,
            tools=["repo.edit"],
            ladder=[
                Rung("typed create/replace/insert", "engine.apply_typed_mutation",
                     "moderate",
                     "validated in the resulting file, rolled back on a failing test"),
                Rung("red-before-green", "engine._validate", "moderate",
                     "an untested edit is unproven, never verified"),
            ]),
        Capability(
            domain="TEST", verb="run admitted tests",
            probe=_engine_can_mutate,
            tools=[],
            ladder=[
                Rung("proving test", "engine admitted-test forms", "moderate",
                     "pytest or a bare path; the refusal names the accepted forms"),
                Rung("targeted subset", "engine tests= argument", "moderate", ""),
                Rung("full regression", "pytest -q", "expensive",
                     "reserved for real blast radius"),
            ]),
        Capability(
            domain="PROFILE", verb="profile prefill and wall",
            probe=_profiler_reachable,
            tools=[],
            ladder=[
                Rung("report", "hcli report", "moderate",
                     "cold prefill, warm reuse, decode, with the warm control"),
                Rung("prefill attribution", "prefill_profile", "expensive",
                     "dispatch counts, where the wall goes"),
            ]),
        Capability(
            domain="PROCESS", verb="own, adopt and reap processes",
            probe=_processes_reachable,
            tools=[],
            ladder=[
                Rung("list live", "processes.live_processes", "cheap", ""),
                Rung("find orphaned bodies", "processes.orphaned_resident_bodies",
                     "cheap", ""),
                Rung("reap owned orphans", "processes.reap_orphaned_bodies",
                     "moderate",
                     "startup-only, never blind kill; provenance first"),
            ]),
        Capability(
            domain="DEBUG", verb="reproduce, localize and diagnose a failure",
            probe=_debug_reachable,
            tools=[],
            ladder=[
                Rung("reproduce", "engine admitted-test path", "moderate",
                     "a failure you cannot reproduce is a story, not a fact"),
                Rung("localize", "debug_capability.diagnose", "cheap",
                     "the deepest PROJECT frame, the assertion, the function"),
                Rung("source window", "debug_capability", "cheap",
                     "the failing line marked, not a 200-line dump"),
                Rung("discriminate / bisect", "MISSING", "expensive",
                     "narrowing between competing causes not yet an owned rung"),
            ]),
        Capability(
            domain="FUZZ", verb="throw malformed input at a boundary and keep crashes",
            probe=_fuzz_reachable,
            tools=[],
            ladder=[
                Rung("classic corpus", "fuzz_capability.fuzz", "moderate",
                     "empty/huge/control-byte/nesting/encoding-edge inputs"),
                Rung("seed mutation", "fuzz_capability._mutations", "moderate",
                     "byte-level mutations of a caller-supplied seed"),
                Rung("minimize", "fuzz_capability._minimize", "cheap",
                     "shrink a crash to a reproducer; noise never enters context"),
                Rung("process / coverage-guided", "MISSING", "expensive",
                     "native fuzzer with coverage feedback not yet owned"),
            ]),
        Capability(
            domain="FORENSICS", verb="preserve failure evidence before cleanup",
            probe=_forensics_reachable,
            tools=[],
            ladder=[
                Rung("triage snapshot", "forensics_capability.capture", "cheap",
                     "HEAD, dirty files with hashes, recent commits, event tail"),
                Rung("persist", "forensics_capability.capture", "cheap",
                     "written before it returns, so cleanup cannot lose it"),
                Rung("timeline reconstruction", "MISSING", "expensive",
                     "cross-source incident timeline not yet an owned rung"),
            ]),
        Capability(
            domain="REVERSE", verb="identify a file without running it",
            probe=_forensics_reachable,
            tools=[],
            ladder=[
                Rung("magic / format", "forensics_capability.identify", "cheap",
                     "ELF/Mach-O/PE/archive/image/text by header"),
                Rung("classification", "file_eye.classify_bytes", "cheap",
                     "the perception package's deeper classifier when present"),
                Rung("disassembly / symbols", "MISSING", "expensive",
                     "objdump/nm equivalence not yet owned"),
            ]),
        Capability(
            domain="RECOVER", verb="checkpoint, resume and roll back",
            probe=lambda: _engine_can_mutate(),
            tools=[],
            ladder=[
                Rung("mutation rollback", "engine._restore", "cheap",
                     "a rejected mutation leaves the file exactly as it was"),
                Rung("session checkpoint", "chat_state.checkpoint", "cheap",
                     "objective, plan, next action -- resident's own field names"),
                Rung("resume", "chat_state.resume", "cheap",
                     "verifies the repo did not move before replaying"),
            ]),
    ]


#: Domains named in the steer that have NO owned rung yet. Named honestly rather
#: than pretended into existence, so the frontier is visible instead of implied.
UNOWNED_DOMAINS = ("REPORT",)


def capability_map() -> Dict[str, Any]:
    """The whole surface, for /health and for the model's own self-knowledge."""
    owned = capabilities()
    return {
        "schema": "hcli.capabilities.v1",
        "can": {c.domain: c.verb for c in owned if c.reachable()},
        "reachable": [c.domain for c in owned if c.reachable()],
        "present_but_unreachable": [c.domain for c in owned if not c.reachable()],
        "not_yet_owned": list(UNOWNED_DOMAINS),
        "capabilities": [c.to_dict() for c in owned],
    }


def capability(domain: str) -> Optional[Capability]:
    for entry in capabilities():
        if entry.domain == domain.upper():
            return entry
    return None


# === FUZZ (s47): malformed input, keep minimized crashes (merged from fuzz_capability) ===

#: The classic corpus that breaks real parsers. Seeds are mutated on top of this.
_SEED_INPUTS: Tuple[Any, ...] = (
    "", " ", "\n", "\x00", "\xff", "\t\r\n",
    "0", "-0", "1e999", "NaN", "Infinity", "null", "true", "[]", "{}",
    "[" * 200, "{" * 200, '{"' + "a" * 10000 + '"}',
    "a" * 100000, "\ud800", "\\", '"', "'", "%s%s%s", "../" * 50,
    "\x1b[31m", "\U0001f600", b"\x00\x01\x02", b"\xff\xfe", 0, -1, 2 ** 63,
    [], {}, None, 3.14, float("inf"),
)

#: Byte/char mutations applied to a seed to widen coverage cheaply.
def _mutations(seed: Any) -> List[Any]:
    out: List[Any] = []
    if isinstance(seed, str) and seed:
        out.append(seed[:len(seed) // 2])                 # truncate
        out.append(seed + seed)                            # double
        out.append(seed.replace(seed[0], "\x00", 1))       # inject NUL
        out.append(seed.upper() + "￿")                # append noise
    if isinstance(seed, (bytes, bytearray)) and seed:
        out.append(bytes(seed[:len(seed) // 2]))
        out.append(bytes(seed) + b"\x00")
    return out


@dataclass
class Crash:
    """One reproduced invariant violation."""

    exception: str
    message: str
    minimized_repr: str
    signature: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class FuzzResult:
    target: str
    iterations: int
    crashes: List[Crash] = field(default_factory=list)
    clean: bool = True
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "iterations": self.iterations,
            "crash_count": len(self.crashes),
            "clean": self.clean,
            "crashes": [c.to_dict() for c in self.crashes],
            "note": self.note,
        }

    def summary(self) -> str:
        if self.clean:
            return (f"FUZZ {self.target} -- {self.iterations} inputs, no invariant "
                    f"violation. {self.note}").strip()
        lines = [f"FUZZ {self.target} -- {len(self.crashes)} distinct crash(es) "
                 f"in {self.iterations} inputs:"]
        for crash in self.crashes[:6]:
            lines.append(f"  {crash.exception}: {crash.message[:80]}")
            lines.append(f"    reproduce with: {crash.minimized_repr}")
        return "\n".join(lines)


def _minimize(target: Callable[[Any], Any], value: Any,
              allowed: Tuple[type, ...]) -> Any:
    """Shrink a crashing input while it keeps crashing the same way.

    Greedy and bounded: halve a string/bytes/list, drop a dict key, and keep the
    smaller input only if it still violates. A minimized input is what a human
    turns into a red test; the 10,000 that did not crash are noise.
    """
    def still_crashes(candidate: Any) -> bool:
        try:
            target(candidate)
            return False
        except Exception as exc:  # noqa: BLE001
            return not isinstance(exc, allowed)

    current = value
    for _ in range(40):
        smaller: Optional[Any] = None
        if isinstance(current, (str, bytes, bytearray)) and len(current) > 1:
            smaller = current[: len(current) // 2]
        elif isinstance(current, list) and current:
            smaller = current[: len(current) // 2]
        elif isinstance(current, dict) and current:
            smaller = dict(list(current.items())[:-1])
        if smaller is None or not still_crashes(smaller):
            break
        current = smaller
    return current


def fuzz(target: Callable[[Any], Any], *, name: str = "",
         allowed: Tuple[type, ...] = (ValueError,),
         seeds: Optional[List[Any]] = None,
         max_iterations: int = 400) -> FuzzResult:
    """Throw malformed input at `target`; collect deduplicated minimized crashes.

    `allowed` is the invariant: exceptions in this tuple are the contract (a
    parser is ALLOWED to raise ValueError on bad input). Anything else -- an
    IndexError, a RecursionError, a bare Exception -- is a crash. Returning
    normally is always fine. Deduplicated by (exception type, first frame), so
    one bug does not report a thousand times.
    """
    corpus: List[Any] = list(_SEED_INPUTS)
    for seed in (seeds or []):
        corpus.append(seed)
        corpus.extend(_mutations(seed))
    corpus = corpus[:max_iterations]

    crashes: Dict[str, Crash] = {}
    tried = 0
    for value in corpus:
        tried += 1
        try:
            target(value)
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, allowed):
                continue
            kind = type(exc).__name__
            signature = hashlib.sha256(
                f"{kind}:{str(exc)[:60]}".encode("utf-8", "replace")).hexdigest()[:12]
            if signature in crashes:
                continue
            minimized = _minimize(target, value, allowed)
            crashes[signature] = Crash(
                exception=kind, message=str(exc)[:200],
                minimized_repr=repr(minimized)[:200], signature=signature)
    result = FuzzResult(target=name or getattr(target, "__name__", "target"),
                        iterations=tried, crashes=list(crashes.values()),
                        clean=not crashes)
    if not crashes:
        result.note = f"held the invariant (only {allowed} raised) across the corpus"
    return result


# === FORENSICS (s48) + REVERSE (s46): preserve evidence; identify a file by bytes (merged from forensics_capability) ===

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


# === DEBUG (s21): reproduce, localize, diagnose (merged from debug_capability) ===

#: The last frame of a Python traceback that lives in the project, not in the
#: stdlib or site-packages -- the place a fix almost always goes.
_FRAME = re.compile(r'^\s*(?:File "|)(?P<path>[^"\n]+?\.py)"?[,:]\s*'
                    r'(?:line\s+)?(?P<line>\d+)', re.M)
_ASSERT = re.compile(r'^E\s+(?P<kind>\w*(?:Error|Exception|assert\w*))\b.*$', re.M | re.I)


@dataclass
class Failure:
    """A reproduced, localized failure, ready to reason about."""

    reproduced: bool
    command: str
    exit_code: Optional[int]
    kind: str = ""
    message: str = ""
    file: str = ""
    line: int = 0
    function: str = ""
    source_window: str = ""
    handle: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def summary(self) -> str:
        """What the model should see: actionable, not the whole dump."""
        if not self.reproduced:
            return (f"could not reproduce with `{self.command}` "
                    f"(exit {self.exit_code}): {self.note or 'no failure captured'}")
        where = f"{self.file}:{self.line}" if self.file else "location unknown"
        head = f"REPRODUCED `{self.command}` -> {self.kind or 'failure'} at {where}"
        if self.function:
            head += f" in {self.function}()"
        parts = [head]
        if self.message:
            parts.append(f"  {self.message}")
        if self.source_window:
            parts.append(self.source_window)
        if self.handle:
            parts.append(f"  full output: {self.handle}")
        return "\n".join(parts)


def _project_frame(text: str, root: Path) -> Optional[Dict[str, Any]]:
    """The deepest traceback frame inside the project. Stdlib frames are noise."""
    best = None
    for match in _FRAME.finditer(text or ""):
        raw = match.group("path")
        try:
            path = Path(raw)
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
        except Exception:
            continue
        parts = str(path)
        if any(seg in parts for seg in ("/site-packages/", "/lib/python",
                                        "/_pytest/", "/pluggy/")):
            continue
        try:
            path.relative_to(root.resolve())
        except ValueError:
            continue
        best = {"file": str(path), "line": int(match.group("line"))}
    return best


def _enclosing_function(path: Path, line: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for index in range(min(line, len(lines)) - 1, -1, -1):
        stripped = lines[index].lstrip()
        if stripped.startswith(("def ", "async def ")):
            name = stripped.split("(")[0].replace("async def ", "").replace("def ", "")
            return name.strip()
    return ""


def _source_window(path: Path, line: int, radius: int = 4) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    lo = max(0, line - radius - 1)
    hi = min(len(lines), line + radius)
    out = []
    for index in range(lo, hi):
        mark = ">>" if index == line - 1 else "  "
        out.append(f"  {mark} {index + 1:4} {lines[index]}")
    return "\n".join(out)


def diagnose(output: str, *, command: str, exit_code: Optional[int],
             root: Path, cache: Any = None) -> Failure:
    """Turn a raw failed-run output into a localized Failure.

    Pure over its inputs, so it is testable without running anything: hand it
    the captured output and it localizes. The caller owns reproduction (through
    the engine), which is what keeps ONE test authority.
    """
    reproduced = exit_code not in (0, None)
    kind = ""
    message = ""
    assertion = _ASSERT.search(output or "")
    if assertion:
        kind = assertion.group("kind")
        message = assertion.group(0).lstrip("E").strip()
    frame = _project_frame(output or "", root)
    failure = Failure(reproduced=reproduced, command=command, exit_code=exit_code,
                      kind=kind, message=message)
    if frame:
        path = Path(frame["file"])
        failure.file = str(path)
        failure.line = frame["line"]
        failure.function = _enclosing_function(path, frame["line"])
        failure.source_window = _source_window(path, frame["line"])
    if cache is not None and output and len(output) > 1500:
        try:
            ref = cache.store(output)
            failure.handle = ref.id
        except Exception:
            pass
    if not reproduced:
        failure.note = "the command exited cleanly; nothing to debug"
    elif not frame:
        failure.note = ("reproduced, but no project frame in the traceback -- "
                        "the failure is in a dependency or the harness")
    return failure
