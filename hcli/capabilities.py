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
from typing import Any, Callable, Dict, List, Optional


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
        from .debug_capability import diagnose
        return callable(diagnose)
    except Exception:
        return False


def _fuzz_reachable() -> bool:
    try:
        from .fuzz_capability import fuzz
        return callable(fuzz)
    except Exception:
        return False


def _forensics_reachable() -> bool:
    try:
        from .forensics_capability import capture, identify
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
                Rung("session checkpoint", "chat_continuity.checkpoint", "cheap",
                     "objective, plan, next action -- resident's own field names"),
                Rung("resume", "chat_continuity.resume", "cheap",
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
