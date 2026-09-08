"""Typed, permissioned tools for AgentOS.

The model is allowed to *request* a tool; it is never allowed to smuggle a
shell command, a credential, or an unbounded path operation through a generic
``exec`` escape hatch.  Every registered tool has an input schema, an output
contract, a mutation class, and a provenance-bearing result.

The default registry is intentionally conservative.  It provides enough
filesystem, Git, public research, receipt, ModelLake, VMCP, Doctor, Gravity,
accelerator, roadmap, and test-discovery surfaces to investigate a mission.
Write-capable tools can be added by an application, but are denied unless the
application explicitly grants the corresponding permission.
"""
from __future__ import annotations

import difflib
import hashlib
import html
import ipaddress
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TOOL_SCHEMA = "hcli.agentos.tool.v1"

READ_ONLY = "read_only"
RESEARCH = "research"
WORKSPACE_WRITE = "workspace_write"
REPO_WRITE = "repo_write"
REVERSIBLE_REPO = "reversible_repo"
REVERSIBLE_RUNTIME = "reversible_runtime"
COSTLY = "costly"
DESTRUCTIVE = "destructive"
EXTERNAL_WRITE = "external_write"

MUTATION_CLASSES = frozenset({
    READ_ONLY,
    RESEARCH,
    REVERSIBLE_REPO,
    REVERSIBLE_RUNTIME,
    COSTLY,
    DESTRUCTIVE,
    EXTERNAL_WRITE,
    # Compatibility labels from the first AgentOS registry revision.
    WORKSPACE_WRITE,
    REPO_WRITE,
})

_SECRET_NAME_RE = re.compile(
    r"(?i)^(?:api[_-]?key|access[_-]?token|authorization|auth|password|secret|private[_-]?key|bearer|token|key|(?:hf|gh|github|openai|anthropic)[_-]?(?:token|key))$"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b((?:api[_-]?key|access[_-]?token|authorization|password|secret|private[_-]?key|bearer|(?:hf|gh|github|openai|anthropic)[_-]?(?:token|key)))\s*[:=]\s*([^\s,;]+)"
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)\b(?:hf_[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9_-]+|github_pat_[A-Za-z0-9_:-]+|sk-[A-Za-z0-9_-]+|xox[baprs]-[A-Za-z0-9-]+)\b"
)
_SHELL_META_RE = re.compile(r"[;&|<>`$()]|\n|\r")
_SAFE_SHELL_COMMANDS = frozenset(
    {
        "cat",
        "cmp",
        "cut",
        "diff",
        "file",
        "grep",
        "head",
        "jq",
        "ls",
        "md5",
        "more",
        "realpath",
        "rg",
        "sed",
        "shasum",
        "sha256sum",
        "stat",
        "tail",
        "tr",
        "wc",
    }
)
_SAFE_GIT_COMMANDS = frozenset({"status", "diff", "log", "show", "rev-parse"})
_MAX_READ_BYTES = 2 * 1024 * 1024
#: Lowered from 20,000. With the byte cap in place a rare pattern over the repo
#: root still cost 5.7 s at 20,000 files; a code search does not need that
#: breadth, and the result already reports `truncated` so a caller can narrow
#: the root rather than be told a silent lie.
_MAX_SEARCH_FILES = 5_000
_MAX_LIST_DIRECTORIES = 20_000


#: Every string in a ToolResult is clipped to this before it reaches a caller, in
#: `to_dict`. It is the LAST truncation and for a long time the only undisclosed one:
#: fs.read clipped a 299,504-byte ledger to 65,536 and reported that, and then this
#: clipped the content to 4,001 and reported nothing. A round read the ledger, received
#: 1.3% of it, and was told it had 21.9%.
_RESULT_STRING_LIMIT = 4000


def _redact(value: Any, *, limit: int = _RESULT_STRING_LIMIT) -> Any:
    """Redact likely credentials before anything enters a result/receipt."""
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if _SECRET_NAME_RE.search(str(key)):
                out[str(key)] = "[REDACTED]"
            else:
                out[str(key)] = _redact(item, limit=limit)
        return out
    if isinstance(value, list):
        return [_redact(item, limit=limit) for item in value]
    if isinstance(value, tuple):
        return [_redact(item, limit=limit) for item in value]
    if isinstance(value, str):
        text = value
        text = _SECRET_ASSIGNMENT_RE.sub(r"\1=[REDACTED]", text)
        text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", text)
        text = _SECRET_VALUE_RE.sub("[REDACTED]", text)
        return text[:limit] + ("…" if len(text) > limit else "")
    return value


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def validate_input(value: Any, schema: Mapping[str, Any], path: str = "$") -> Optional[str]:
    """Small dependency-free JSON-schema subset used at the tool boundary."""
    if not isinstance(schema, Mapping):
        return None
    expected = schema.get("type")
    if expected:
        names = expected if isinstance(expected, list) else [expected]
        ok = any(
            (name == "object" and isinstance(value, dict))
            or (name == "array" and isinstance(value, list))
            or (name == "string" and isinstance(value, str))
            or (name == "boolean" and isinstance(value, bool))
            or (name == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (name == "number" and isinstance(value, (int, float)) and not isinstance(value, bool))
            or (name == "null" and value is None)
            for name in names
        )
        if not ok:
            return f"{path}: expected {expected}, got {_json_type(value)}"
    if isinstance(value, dict):
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                return f"{path}: missing required property {key!r}"
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            extras = [key for key in value if key not in properties]
            if extras:
                return f"{path}: unexpected properties {extras!r}"
        for key, child in properties.items():
            if key in value and isinstance(child, Mapping):
                error = validate_input(value[key], child, f"{path}.{key}")
                if error:
                    return error
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            error = validate_input(item, schema["items"], f"{path}[{index}]")
            if error:
                return error
    if isinstance(value, str) and schema.get("enum") and value not in schema["enum"]:
        return f"{path}: {value!r} is not in enum"
    return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(path: Path, roots: Iterable[Path]) -> bool:
    try:
        candidate = path.resolve(strict=False)
    except OSError:
        candidate = Path(os.path.abspath(path))
    for root in roots:
        try:
            candidate.relative_to(root.resolve(strict=False))
            return True
        except ValueError:
            continue
    return False


@dataclass(frozen=True)
class ToolContext:
    workspace: Path
    repo_root: Path
    mission_root: Optional[Path] = None
    permissions: frozenset[str] = frozenset({READ_ONLY, RESEARCH})

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", Path(self.workspace).expanduser().resolve())
        object.__setattr__(self, "repo_root", Path(self.repo_root).expanduser().resolve())
        if self.mission_root is not None:
            object.__setattr__(self, "mission_root", Path(self.mission_root).expanduser().resolve())
        object.__setattr__(self, "permissions", frozenset(str(p) for p in self.permissions))

    @property
    def read_roots(self) -> Tuple[Path, ...]:
        roots = [self.workspace, self.repo_root]
        if self.mission_root is not None:
            roots.append(self.mission_root)
        return tuple(dict.fromkeys(roots))

    @property
    def write_roots(self) -> Tuple[Path, ...]:
        """Roots where an explicitly permissioned reversible tool may write."""
        return tuple(dict.fromkeys((self.workspace, self.repo_root)))

    def resolve_read_path(self, raw: Any) -> Path:
        text = str(raw or "").strip()
        if not text:
            raise ValueError("path is required")
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        candidate = candidate.resolve(strict=False)
        if not _within(candidate, self.read_roots):
            raise PermissionError(f"path is outside the AgentOS read roots: {candidate}")
        return candidate

    def resolve_write_path(self, raw: Any) -> Path:
        text = str(raw or "").strip()
        if not text:
            raise ValueError("path is required")
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        candidate = candidate.resolve(strict=False)
        if not _within(candidate, self.write_roots):
            raise PermissionError(f"path is outside the AgentOS write roots: {candidate}")
        relative_parts = set(candidate.relative_to(self.workspace).parts) if _within(candidate, (self.workspace,)) else set()
        if relative_parts.intersection({".git", ".hcli"}):
            raise PermissionError("protected AgentOS paths are not writable through this tool")
        return candidate


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    mutation: str = READ_ONLY
    deterministic: bool = True
    timeout_s: float = 30.0
    roles: Tuple[str, ...] = ()
    resources: Tuple[str, ...] = ()
    verifier_expectations: Tuple[str, ...] = ()
    provenance: str = "hcli.tool_registry"
    alias_of: Optional[str] = None
    handler: Callable[[ToolContext, Dict[str, Any]], Any] = field(repr=False, compare=False, default=lambda _c, _a: None)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "schema": TOOL_SCHEMA,
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "mutation": self.mutation,
            "deterministic": self.deterministic,
            "timeout_s": self.timeout_s,
            "roles": list(self.roles),
            "resources": list(self.resources),
            "verifier_expectations": list(self.verifier_expectations),
            "provenance": self.provenance,
        }
        if self.alias_of:
            result["alias_of"] = self.alias_of
        return result


@dataclass
class ToolResult:
    tool: str
    invocation_id: str
    ok: bool
    value: Any = None
    error: Optional[str] = None
    failure_class: Optional[str] = None
    mutation: str = READ_ONLY
    deterministic: bool = True
    provenance: Dict[str, Any] = field(default_factory=dict)
    artifact: Optional[Dict[str, Any]] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "hcli.agentos.tool.result.v1",
            "tool": self.tool,
            "invocation_id": self.invocation_id,
            "ok": self.ok,
            "value": _redact(self.value),
            "error": _redact(self.error),
            "failure_class": self.failure_class,
            "mutation": self.mutation,
            "deterministic": self.deterministic,
            "provenance": _redact(self.provenance),
            "artifact": _redact(self.artifact),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": (
                self.finished_at - self.started_at
                if self.finished_at is not None
                else None
            ),
        }


class ToolRegistry:
    """Register and invoke typed tools under an explicit permission boundary."""

    def __init__(self, context: ToolContext):
        self.context = context
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        name = str(spec.name or "").strip()
        if not name or any(char.isspace() for char in name):
            raise ValueError(f"invalid tool name: {name!r}")
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        if spec.mutation not in MUTATION_CLASSES:
            raise ValueError(f"unknown mutation class: {spec.mutation}")
        if spec.alias_of and spec.alias_of not in self._tools:
            raise ValueError(f"alias target is not registered: {spec.alias_of}")
        self._tools[name] = spec
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(str(name or "").strip())

    def discover(
        self,
        *,
        role: Optional[str] = None,
        include_aliases: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return canonical tools with compatibility aliases as metadata.

        Aliases remain callable through :meth:`get` and :meth:`invoke`, but
        are not independent model-facing capabilities. ``include_aliases``
        is available for diagnostics and migration audits.
        """
        # Resolve alias CHAINS to their canonical root. Consolidation can make an
        # alias point at a name that is itself now an alias -- filesystem.read ->
        # fs.read -> fs -- and crediting only the direct target would leave the
        # outer name callable but advertised by nothing. A capability nobody can
        # find is the defect this campaign keeps paying for.
        def _root(name: str) -> str:
            seen = {name}
            spec = self._tools.get(name)
            while spec is not None and spec.alias_of:
                if spec.alias_of in seen:  # a cycle is a bug, not a loop to ride
                    break
                seen.add(spec.alias_of)
                name = spec.alias_of
                spec = self._tools.get(name)
            return name

        aliases: Dict[str, List[str]] = {}
        for item in self._tools.values():
            if item.alias_of:
                aliases.setdefault(_root(item.name), []).append(item.name)
        result = []
        for spec in sorted(self._tools.values(), key=lambda item: item.name):
            if spec.alias_of and not include_aliases:
                continue
            if role and spec.roles and role not in spec.roles:
                continue
            item = spec.to_dict()
            if not spec.alias_of and aliases.get(spec.name):
                item["aliases"] = sorted(aliases[spec.name])
            result.append(item)
        return result

    def capability_names(self, *, role: Optional[str] = None) -> set:
        """Every name a caller can REACH, canonical or absorbed. [S004]

        discover() answers "what are the primary tools" -- the smaller surface a
        model chooses from. This answers a different question: "can I get to X
        at all", where X may now live as an op behind a merged tool. Both are
        needed. A reachability check written against discover() alone will call
        a capability missing the moment it is consolidated, which is how a
        surface reduction gets mistaken for a capability loss.
        """
        names = set()
        for item in self.discover(role=role):
            names.add(item["name"])
            names.update(item.get("aliases") or [])
        return names

    def describe(self, focus: str = "", *, max_results: int = 12) -> Dict[str, Any]:
        """Return the smallest useful slice of the registry for one question.

        The full catalog is still available through :meth:`discover`, but
        putting every domain in every model prompt makes aliases and unrelated
        capabilities compete with the current task. This deterministic index
        lets a model ask for the exact signatures it needs after seeing only a
        compact first-round catalog.
        """
        query = str(focus or "").strip()
        terms = tuple(dict.fromkeys(re.findall(r"[a-z0-9][a-z0-9_.-]*", query.lower())))
        try:
            limit = max(1, min(32, int(max_results)))
        except (TypeError, ValueError):
            limit = 12

        scored: List[Tuple[int, str, ToolSpec]] = []
        for spec in self._tools.values():
            if spec.alias_of:
                continue
            name = spec.name.lower()
            haystack = " ".join(
                (spec.name, spec.description, *spec.roles, *spec.resources)
            ).lower()
            score = 0
            for term in terms:
                if term == name:
                    score += 100
                elif term in name:
                    score += 40
                elif term in haystack:
                    score += 10
            if score or not terms:
                scored.append((score, spec.name, spec))
        if terms:
            scored.sort(key=lambda item: (-item[0], item[1]))
        else:
            scored.sort(key=lambda item: item[1])
        chosen = [spec for _score, _name, spec in scored[:limit]]
        matches = [spec.to_dict() for spec in chosen]
        names = [spec.name for spec in chosen]
        return {
            "names": names,
            "matches": matches,
            "shown": len(matches),
            "match_count": len(scored),
            "truncated": len(scored) > limit,
            "focus": query,
            "provenance": "hcli.tool_registry.ToolRegistry.describe",
        }

    def invoke(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> ToolResult:
        invocation_id = f"tool-{uuid.uuid4()}"
        spec = self.get(name)
        started = time.time()
        if spec is None:
            return ToolResult(
                tool=str(name), invocation_id=invocation_id, ok=False,
                error=f"unknown tool: {name}", failure_class="UNKNOWN_TOOL",
                started_at=started, finished_at=time.time(),
            )
        args = dict(arguments or {})
        schema_error = validate_input(args, spec.input_schema)
        if schema_error:
            return ToolResult(
                tool=spec.name, invocation_id=invocation_id, ok=False,
                error=schema_error, failure_class="INVALID_ARGUMENTS",
                mutation=spec.mutation, deterministic=spec.deterministic,
                provenance={"source": spec.provenance},
                started_at=started, finished_at=time.time(),
            )
        # A caller may request a shorter timeout, never a longer one than the
        # tool contract declares.  Handlers that support timeouts receive the
        # bounded value; subprocess/network handlers also keep their own
        # defensive caps.
        if "timeout_s" in args:
            try:
                args["timeout_s"] = min(
                    max(0.1, float(args["timeout_s"])),
                    max(0.1, float(spec.timeout_s)),
                )
            except (TypeError, ValueError):
                return ToolResult(
                    tool=spec.name, invocation_id=invocation_id, ok=False,
                    error="timeout_s must be numeric", failure_class="INVALID_ARGUMENTS",
                    mutation=spec.mutation, deterministic=spec.deterministic,
                    provenance={"source": spec.provenance},
                    started_at=started, finished_at=time.time(),
                )
        if spec.mutation not in self.context.permissions:
            return ToolResult(
                tool=spec.name, invocation_id=invocation_id, ok=False,
                error=f"permission denied for mutation class {spec.mutation}",
                failure_class="PERMISSION_DENIED", mutation=spec.mutation,
                deterministic=spec.deterministic,
                provenance={"source": spec.provenance},
                started_at=started, finished_at=time.time(),
            )
        try:
            value = spec.handler(self.context, args)
            output_error = validate_input(value, spec.output_schema)
            if output_error:
                return ToolResult(
                    tool=spec.name, invocation_id=invocation_id, ok=False,
                    error=f"tool returned invalid output: {output_error}",
                    failure_class="INVALID_OUTPUT", mutation=spec.mutation,
                    deterministic=spec.deterministic,
                    provenance={"source": spec.provenance},
                    started_at=started, finished_at=time.time(),
                )
            finished = time.time()
            artifact = value.get("artifact") if isinstance(value, dict) else None
            return ToolResult(
                tool=spec.name, invocation_id=invocation_id, ok=True,
                value=_redact(value), mutation=spec.mutation,
                deterministic=spec.deterministic,
                provenance={"source": spec.provenance, "observed_at": finished},
                artifact=artifact if isinstance(artifact, dict) else None,
                started_at=started, finished_at=finished,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                tool=spec.name, invocation_id=invocation_id, ok=False,
                error=f"tool timed out after {spec.timeout_s}s",
                failure_class="TIMEOUT", mutation=spec.mutation,
                deterministic=spec.deterministic,
                provenance={"source": spec.provenance},
                started_at=started, finished_at=time.time(),
            )
        except Exception as exc:  # noqa: BLE001 - tool failures are receipt data
            return ToolResult(
                tool=spec.name, invocation_id=invocation_id, ok=False,
                error=f"{type(exc).__name__}: {_redact(str(exc))}",
                failure_class=type(exc).__name__, mutation=spec.mutation,
                deterministic=spec.deterministic,
                provenance={"source": spec.provenance},
                started_at=started, finished_at=time.time(),
            )


def _text_limit(value: Any, default: int = 64 * 1024, maximum: int = _MAX_READ_BYTES) -> int:
    try:
        return max(1, min(maximum, int(value if value is not None else default)))
    except (TypeError, ValueError):
        return default


#: Default number of decision-relevant entries a list tool emits. The closed-turn
#: observation budget is 500 characters; twelve compact rows fit, an unbounded
#: lake or process table does not.
_ACTIONABLE_SHOW_DEFAULT = 12


def _shown_limit(value: Any, default: int = _ACTIONABLE_SHOW_DEFAULT, maximum: int = 32) -> int:
    try:
        return max(1, min(maximum, int(value if value is not None else default)))
    except (TypeError, ValueError):
        return default


def _lead_with(payload: Mapping[str, Any], *first: str) -> Dict[str, Any]:
    """Rebuild a dict so json.dumps emits decision-relevant keys first."""
    out: Dict[str, Any] = {}
    for key in first:
        if key in payload:
            out[key] = payload[key]
    for key, value in payload.items():
        if key not in out:
            out[key] = value
    return out


_SCOPE_KEYS = ("n", "total", "count", "shown", "truncated", "truncation_note",
               "n_owed", "n_processes", "n_orphaned", "n_ranked", "n_specimens",
               "specimen_count")


def _scope_first(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Lead with the keys that say HOW MUCH, before the rows that say WHAT.

    Engine._compact_closed_observations cuts every observation to 500 chars as
    head 250 + marker + tail 208, eliding the middle. A result that leads with
    its rows therefore delivers rows and loses its own scope: the round sees
    some entries and cannot tell how many exist or whether it has them all --
    which is exactly how round 20 read a partial census as complete.

    `_lead_with` already existed for this and had 7 call sites. This is the same
    move applied by rule instead of by remembering.
    """
    present = [k for k in _SCOPE_KEYS if k in payload]
    return _lead_with(payload, *present) if present else dict(payload)


def _read_file(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    path = context.resolve_read_path(args.get("path"))
    if path.is_dir():
        # NOT FileNotFoundError. Measured: the model listed `hcli`, called
        # fs.read on it, was told the path did not exist, concluded it had the
        # path wrong, and spent five retries and eight model calls hunting a
        # path that was correct all along. An error that misdescribes the
        # situation cannot be recovered from -- say what it is and what to use.
        raise IsADirectoryError(
            f"{path} is a directory, not a file. Use fs.list to see what is "
            f"inside it, then fs.read one of the files it names."
        )
    if not path.is_file():
        # A miss is CORRECTABLE and used to be a bare traceback, so a model
        # that guessed a path just guessed it again -- measured: three
        # identical FileNotFoundError on hcli/tests/test_tool_registry.py in
        # one goal, one 60-190s model call apiece. The directory case has said
        # what to do next for a while; the file case never did.
        parent = path.parent
        if not parent.is_dir():
            raise FileNotFoundError(
                f"{path} does not exist, and neither does {parent}. Use fs.list "
                f"on a directory that does exist to find the right path."
            )
        siblings = sorted(q.name for q in parent.iterdir() if q.is_file())
        close = difflib.get_close_matches(path.name, siblings, n=3, cutoff=0.72)
        hint = (
            f" Did you mean: {', '.join(close)}?" if close
            else f" {parent} holds {len(siblings)} files; use fs.list to see them."
        )
        raise FileNotFoundError(
            f"{path} does not exist.{hint} If you meant to CREATE it, do not "
            f"read it first -- emit a create operation for that path."
        )
    limit = _text_limit(args.get("max_bytes"))
    raw = path.read_bytes()
    encoding = str(args.get("encoding") or "utf-8")

    # A WINDOW, because without one a large file can only ever be read from the
    # top. fs.read returned the first 4,001 bytes of a 188,062-byte engine.py,
    # so a model that had already located `_record_model_call` at line 3514 --
    # fs.search reports the line -- could never read it, and said so:
    # "Need to see the actual _record_model_call function ... to implement the
    # grammar_enforced field correctly". It could find the code and not look at
    # it. Lines are 1-indexed and inclusive, matching what fs.search returns.
    start = args.get("start_line")
    end = args.get("end_line")
    line_window = start is not None or end is not None
    if line_window:
        text = raw.decode(encoding, errors="replace")
        lines = text.splitlines(keepends=True)
        first = max(1, int(start or 1))
        last = min(len(lines), int(end) if end is not None else len(lines))
        selected = "".join(lines[first - 1:last]) if first <= last else ""
        body = selected.encode(encoding, errors="replace")
        clipped = body[:limit]
        return {
            "path": str(path),
            "bytes": len(raw),
            "start_line": first,
            "end_line": last,
            "total_lines": len(lines),
            **_truncation_fields(len(clipped), len(body)),
            "sha256": _sha256_bytes(raw),
            "content": clipped.decode(encoding, errors="replace"),
            "artifact": {"kind": "file", "path": str(path), "sha256": _sha256_bytes(raw), "bytes": len(raw)},
        }

    clipped = raw[:limit]
    return {
        "path": str(path),
        "bytes": len(raw),
        **_truncation_fields(len(clipped), len(raw)),
        "sha256": _sha256_bytes(raw),
        "content": clipped.decode(encoding, errors="replace"),
        "artifact": {"kind": "file", "path": str(path), "sha256": _sha256_bytes(raw), "bytes": len(raw)},
    }


def _truncation_fields(shown: int, total: int) -> Dict[str, Any]:
    """Say HOW MUCH survived, not merely that a cut happened.

    `truncated: True` is technically honest and operationally useless: it cannot tell a
    caller 99% from 1.3%. A round read the 299,504-byte Odyssey ledger through fs.read,
    received 4,001 characters, reasoned soundly over the handful of specimens in that
    1.3%, and never learned that the two bodies still owing anatomy were outside it.
    """
    # What the caller RECEIVES, not what this handler clipped to. `to_dict` redacts
    # every string down to _RESULT_STRING_LIMIT afterwards, so reporting the handler's
    # own limit overstates it by 16x on a large file -- which is exactly the failure
    # this function exists to stop, one layer up.
    delivered = min(shown, _RESULT_STRING_LIMIT)
    if delivered >= total:
        return {"truncated": False, "shown_bytes": delivered}
    pct = (delivered / total * 100.0) if total else 0.0
    return {
        "truncated": True,
        "shown_bytes": delivered,
        "truncation_note": (
            f"TRUNCATED: you are seeing {delivered} of {total} bytes ({pct:.1f}%). "
            f"Do not conclude anything about what is NOT shown. Narrow with "
            f"start/end lines, or use the purpose-built tool for this file if one exists."
        ),
    }


#: A search reads candidates whole, so an unbounded file size is an unbounded
#: search. Measured: one fs.search took 23.5 s against a 50 ms budget because
#: `glob` defaults to `*` and the walk read model artifacts under workspace/.
_MAX_SEARCH_FILE_BYTES = 2_000_000
_SEARCH_SKIP_DIRS = {
    ".git", ".venv", "__pycache__", "node_modules", ".cache",
    "target", ".worktrees",
}
_SEARCH_SKIP_SUFFIXES = {
    ".safetensors", ".bin", ".gguf", ".pt", ".pth", ".onnx", ".npy", ".npz",
    ".so", ".dylib", ".a", ".o", ".zip", ".tar", ".gz", ".zst", ".png",
    ".jpg", ".jpeg", ".pdf", ".mp4", ".mov", ".wav",
}


def _search_files(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    # `path` is an alias for `root`. fs.read, fs.list and fs.write all name
    # their location `path`; search alone called it `root`, so the consistent
    # guess was a hard schema error -- "unexpected properties ['path']" --
    # which the model read as ZERO MATCHES and reported as an absence of
    # evidence. It then hedged a correct answer because its own search had
    # apparently found nothing. One inconsistent word cost a whole round and
    # the confidence of the answer.
    root = context.resolve_read_path(args.get("root") or args.get("path") or ".")
    # A FILE is a legitimate place to search. Finding the line a symbol sits on
    # inside one known file is the whole point of searching before a windowed
    # read, and refusing it with NotADirectoryError sent the model back to
    # reading the file's head -- which is where it could not see the symbol in
    # the first place. Measured: the truncation notice tells the model to
    # "use fs.search to find the line a symbol is on, then fs.read that file
    # with start_line and end_line", and fs.search then rejected the file it
    # had just been pointed at.
    single_file = None
    if root.is_file():
        single_file = root
        root = root.parent
    elif not root.is_dir():
        raise NotADirectoryError(root)
    needle = str(args.get("pattern") or "")
    if not needle:
        raise ValueError("pattern is required")
    # Naming one file IS the filter, so it overrides any glob rather than
    # silently returning nothing when the two disagree.
    glob = single_file.name if single_file is not None else str(args.get("glob") or "*")
    limit = max(1, min(1000, int(args.get("max_results") or 100)))
    matches: List[Dict[str, Any]] = []
    files_seen = 0
    skipped_large = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames if name not in _SEARCH_SKIP_DIRS
        )
        for filename in sorted(filenames):
            if not Path(filename).match(glob):
                continue
            path = Path(dirpath) / filename
            # Bound the BYTES, not only the file count. The count was already
            # capped and a search still took 23.5 s, because `glob` defaults to
            # `*` and every candidate was read WHOLE -- including multi-gigabyte
            # model artifacts under workspace/. Source files that a search is
            # for are kilobytes; anything above the cap is a specimen, not code.
            try:
                if path.stat().st_size > _MAX_SEARCH_FILE_BYTES:
                    skipped_large += 1
                    continue
            except OSError:
                continue
            if path.suffix.lower() in _SEARCH_SKIP_SUFFIXES:
                continue
            files_seen += 1
            if files_seen > _MAX_SEARCH_FILES:
                return {"root": str(root), "pattern": needle, "matches": matches, "truncated": True, "files_seen": files_seen, "skipped_large": skipped_large}
            try:
                data = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_number, line in enumerate(data.splitlines(), 1):
                if needle in line:
                    matches.append({"path": str(path), "line": line_number, "text": line[:1000]})
                    if len(matches) >= limit:
                        return {"root": str(root), "pattern": needle, "matches": matches, "truncated": True, "files_seen": files_seen}
    return {"root": str(root), "pattern": needle, "matches": matches, "truncated": False, "files_seen": files_seen}


def _list_files(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """List files and visible directories under a read root.

    `fs.search` requires a content `pattern`, so a caller that wanted to SEE
    what is in a directory could not express it and forced search into a
    listing role instead -- the model spent an entire tool budget calling
    fs.search without `pattern`, reading the failure, and guessing again. The
    listing result keeps the historical ``files`` field and adds
    ``directories`` so a directory question does not silently omit folders.
    """
    root = context.resolve_read_path(args.get("path") or ".")
    if not root.is_dir():
        raise NotADirectoryError(root)
    glob = str(args.get("glob") or "*")
    limit = max(1, min(2000, int(args.get("max_results") or 500)))
    # Recursion is OPT-IN. "What is in this directory" is one level, and the
    # default walked the whole tree: a bare fs.list on this repo took 28.1 s
    # against fs.read's 6 ms, because the tree holds model artifacts and
    # capture directories with tens of thousands of files. A tool that costs
    # half a minute is not a tool the model can afford to look with.
    recursive = bool(args.get("recursive", False))
    entries: List[Dict[str, Any]] = []
    directories: List[Dict[str, Any]] = []
    truncated = False
    directories_seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        directories_seen += 1
        if directories_seen > _MAX_LIST_DIRECTORIES:
            truncated = True
            break
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in {".git", ".venv", "__pycache__", "node_modules"}
        )
        if len(entries) >= limit and len(directories) >= limit:
            # Both caps are full: every further stat() is work whose result is
            # thrown away. The walk used to run to completion regardless.
            truncated = True
            break
        for dirname in dirnames:
            if not Path(dirname).match(glob):
                continue
            if len(directories) >= limit:
                truncated = True
                continue
            path = Path(dirpath) / dirname
            directories.append({
                "path": str(path.relative_to(root)),
                "kind": "directory",
            })
        for filename in sorted(filenames):
            if not Path(filename).match(glob):
                continue
            if len(entries) >= limit:
                # Check the cap BEFORE the stat(). Sizing a file that will not
                # be returned is the whole cost of a large tree.
                truncated = True
                continue
            path = Path(dirpath) / filename
            try:
                size = path.stat().st_size
            except OSError:
                continue
            entries.append({"path": str(path.relative_to(root)), "bytes": size})
        if not recursive:
            break
    return {
        "root": str(root),
        "glob": glob,
        "files": entries,
        "directories": directories,
        "truncated": truncated,
        "directories_seen": directories_seen,
    }


def _git_dir(context: ToolContext, raw: Any = None) -> Path:
    path = context.resolve_read_path(raw or str(context.repo_root))
    return path if path.is_dir() else path.parent


def _run_readonly(argv: Sequence[str], *, cwd: Path, timeout: float = 30.0) -> Dict[str, Any]:
    proc = subprocess.run(
        list(argv), cwd=str(cwd), capture_output=True, text=True,
        timeout=timeout, check=False,
        env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", "LANG": "C"},
    )
    stdout = _redact(proc.stdout or "")
    stderr = _redact(proc.stderr or "")
    return {"argv": list(argv), "cwd": str(cwd), "returncode": proc.returncode, "stdout": stdout, "stderr": stderr}


def _git_status(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    cwd = _git_dir(context, args.get("path"))
    return _run_readonly(["git", "-C", str(cwd), "status", "--short", "--branch"], cwd=cwd)


def _git_log(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    cwd = _git_dir(context, args.get("path"))
    try:
        limit = max(1, min(100, int(args.get("limit") or 10)))
    except (TypeError, ValueError):
        limit = 10
    raw = _run_readonly(["git", "-C", str(cwd), "log", f"-{limit}", "--oneline", "--decorate"], cwd=cwd)
    commits: List[Dict[str, Any]] = []
    for line in str(raw.get("stdout") or "").splitlines():
        token = line.split(None, 1)
        if token:
            commits.append({"hash": token[0], "line": line[:160]})
    return _scope_first({
        "commits": commits,
        "n": len(commits),
        "shown": len(commits),
        "returncode": raw.get("returncode"),
        "stdout": raw.get("stdout"),
        "stderr": raw.get("stderr"),
        "argv": raw.get("argv"),
        "cwd": raw.get("cwd"),
    })


def _shell_readonly(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    command = str(args.get("command") or "").strip()
    if not command:
        raise ValueError("command is required")
    if _SHELL_META_RE.search(command):
        raise PermissionError("shell metacharacters are not allowed")
    argv = shlex.split(command)
    if not argv or Path(argv[0]).name not in _SAFE_SHELL_COMMANDS:
        raise PermissionError(f"command is not in the read-only allowlist: {argv[0] if argv else ''}")
    forbidden = {"-delete", "-exec", "-execdir", "--in-place", "-i"}
    if forbidden.intersection(argv[1:]):
        raise PermissionError("mutating command option is not allowed")
    # File-looking arguments must stay in the same read roots. Options and
    # grep patterns are left alone; this is a conservative boundary, not a
    # shell parser pretending to be a security sandbox.
    for token in argv[1:]:
        if token.startswith("-") or any(ch in token for ch in "*?[]"):
            continue
        candidate = Path(token).expanduser()
        if candidate.is_absolute() or token.startswith((".", "/")):
            context.resolve_read_path(token)
    return _run_readonly(argv, cwd=context.workspace)


def _host_is_public(host: str) -> bool:
    host = host.strip("[]").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        try:
            address = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            return False
    return True


def _public_url(raw: Any, *, allowed_hosts: Optional[Iterable[str]] = None) -> str:
    parsed = urllib.parse.urlparse(str(raw or ""))
    if parsed.scheme != "https" or not parsed.hostname:
        raise PermissionError("research tools require an https URL")
    if parsed.username or parsed.password:
        raise PermissionError("URL credentials are not allowed")
    query_names = {
        urllib.parse.unquote_plus(key).strip().lower()
        for key, _value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    }
    if query_names.intersection({
        "api_key", "apikey", "access_token", "authorization", "password",
        "secret", "signature", "sig", "token",
    }):
        raise PermissionError("credential-bearing URL query parameters are not allowed")
    host = parsed.hostname.lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError as exc:
        raise PermissionError("URL port is invalid") from exc
    allowed = {item.lower().rstrip(".") for item in (allowed_hosts or ())}
    if allowed and host not in allowed and not any(host.endswith("." + suffix) for suffix in allowed):
        raise PermissionError(f"host is not allowed: {host}")
    if not _host_is_public(host):
        raise PermissionError(f"host is not public: {host}")
    netloc = host if port is None else f"{host}:{port}"
    return urllib.parse.urlunparse(("https", netloc, parsed.path or "/", "", parsed.query, ""))


def _fetch(
    context: ToolContext,
    args: Dict[str, Any],
    *,
    allowed_hosts: Optional[Iterable[str]] = None,
    headers: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    del context
    url = _public_url(args.get("url"), allowed_hosts=allowed_hosts)
    limit = _text_limit(args.get("max_bytes"), default=256 * 1024)
    request_headers = {"User-Agent": "hcli-agentos-research/1"}
    request_headers.update({str(key): str(value) for key, value in (headers or {}).items()})
    request = urllib.request.Request(url, headers=request_headers, method="GET")
    started = time.time()
    class _CheckedRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            checked = _public_url(
                urllib.parse.urljoin(req.full_url, newurl),
                allowed_hosts=allowed_hosts,
            )
            return super().redirect_request(req, fp, code, msg, headers, checked)

    opener = urllib.request.build_opener(_CheckedRedirectHandler())
    with opener.open(request, timeout=min(60.0, float(args.get("timeout_s") or 30.0))) as response:
        body = response.read(limit + 1)
        final_url = _public_url(response.geturl(), allowed_hosts=allowed_hosts)
        status = getattr(response, "status", None)
        content_type = response.headers.get("Content-Type")
    clipped = body[:limit]
    return {
        "url": url,
        "final_url": final_url,
        "status": status,
        "content_type": content_type,
        "bytes_read": len(clipped),
        "truncated": len(body) > limit,
        "sha256": _sha256_bytes(clipped),
        "content": clipped.decode("utf-8", errors="replace"),
        "provenance": {"retrieved_at": started, "source_url": final_url},
    }


class _SearchResultParser(HTMLParser):
    """Small parser for the public DuckDuckGo HTML result surface."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[Dict[str, str]] = []
        self._current: Optional[Dict[str, str]] = None
        self._field: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag != "a":
            return
        fields = dict(attrs)
        classes = set(str(fields.get("class") or "").split())
        if "result__a" in classes:
            href = html.unescape(str(fields.get("href") or ""))
            if href.startswith("//"):
                href = "https:" + href
            if href.startswith("/l/"):
                query = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                href = str((query.get("uddg") or [href])[0])
            self._current = {"url": href, "title": "", "snippet": ""}
            self._field = "title"
        elif "result__snippet" in classes and self._current is not None:
            self._field = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current is not None and self._field == "title":
            self._field = None
        if tag in {"div", "td"} and self._current is not None and self._current.get("title") and self._current.get("snippet"):
            self.rows.append(self._current)
            self._current = None
            self._field = None

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._field in {"title", "snippet"}:
            self._current[self._field] = (self._current.get(self._field) or "") + data


class _BingSearchResultParser(HTMLParser):
    """Bounded parser for Bing's server-rendered ``b_algo`` result list."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[Dict[str, str]] = []
        self._current: Optional[Dict[str, str]] = None
        self._field: Optional[str] = None
        self._in_result = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        fields = dict(attrs)
        classes = set(str(fields.get("class") or "").split())
        if tag == "li" and "b_algo" in classes:
            self._current = {"url": "", "title": "", "snippet": ""}
            self._in_result = True
            self._field = None
            return
        if not self._in_result or self._current is None:
            return
        if tag == "a" and not self._current.get("url"):
            href = html.unescape(str(fields.get("href") or ""))
            if href.startswith("http://") or href.startswith("https://"):
                self._current["url"] = href
                self._field = "title"
        elif tag == "p":
            self._field = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "li" and self._in_result and self._current is not None:
            if self._current.get("url") and self._current.get("title"):
                self.rows.append(self._current)
            self._current = None
            self._field = None
            self._in_result = False

    def handle_data(self, data: str) -> None:
        if self._in_result and self._current is not None and self._field:
            self._current[self._field] = (self._current.get(self._field) or "") + data


def _search_limit(value: Any, default: int = 10, maximum: int = 50) -> int:
    try:
        return max(1, min(maximum, int(value if value is not None else default)))
    except (TypeError, ValueError):
        return default


def _web_search(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    limit = _search_limit(args.get("max_results"))
    endpoints = [
        (
            "duckduckgo-html",
            "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query),
            ("html.duckduckgo.com", "duckduckgo.com"),
            _SearchResultParser,
        ),
        (
            "bing-html",
            "https://www.bing.com/search?q=" + urllib.parse.quote_plus(query),
            ("www.bing.com", "bing.com"),
            _BingSearchResultParser,
        ),
    ]
    attempted: List[Dict[str, Any]] = []
    parser: Any = None
    fetched: Optional[Dict[str, Any]] = None
    provider = None
    endpoint = None
    for candidate_provider, candidate_endpoint, allowed_hosts, parser_type in endpoints:
        try:
            candidate = _fetch(
                context,
                {"url": candidate_endpoint, "max_bytes": min(_MAX_READ_BYTES, 512 * 1024), "timeout_s": args.get("timeout_s")},
                allowed_hosts=allowed_hosts,
            )
            body = str(candidate.get("content") or "")
            parser_candidate = parser_type()
            parser_candidate.feed(body)
            rows = getattr(parser_candidate, "rows", [])
            challenged = "anomaly-modal" in body or "captcha" in body.lower()
            attempted.append({"provider": candidate_provider, "status": "CHALLENGE" if challenged else "OK", "rows": len(rows)})
            if rows:
                parser, fetched, provider, endpoint = parser_candidate, candidate, candidate_provider, candidate_endpoint
                break
            if fetched is None:
                # Keep the first successful response for an honest no-result
                # receipt if the fallback is also empty.
                fetched, provider, endpoint, parser = candidate, candidate_provider, candidate_endpoint, parser_candidate
        except Exception as exc:  # noqa: BLE001 - search can report partial availability
            attempted.append({"provider": candidate_provider, "status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"})
    if fetched is None:
        return {
            "query": query,
            "provider": None,
            "results": [],
            "count": 0,
            "source_url": endpoint,
            "retrieved_at": None,
            "confidence": "network-unavailable",
            "unresolved": ["all configured search providers unavailable"],
            "attempted": attempted,
        }
    results: List[Dict[str, Any]] = []
    for row in getattr(parser, "rows", []):
        url = str(row.get("url") or "").strip()
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        results.append({
            "title": " ".join(str(row.get("title") or "").split()),
            "url": url,
            "snippet": " ".join(str(row.get("snippet") or "").split()),
        })
        if len(results) >= limit:
            break
    return {
        "query": query,
        "provider": provider,
        "results": results,
        "count": len(results),
        "source_url": endpoint,
        "retrieved_at": fetched.get("provenance", {}).get("retrieved_at"),
        "confidence": "source-links-extracted" if results else "no-results-parsed",
        "unresolved": [] if results else ["search response format or network availability"],
        "attempted": attempted,
    }


def _github_search(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    kind = str(args.get("kind") or "repositories").strip().lower()
    endpoint_by_kind = {
        "repositories": "repositories",
        "issues": "issues",
        "code": "code",
        "commits": "commits",
    }
    if kind not in endpoint_by_kind:
        raise ValueError("kind must be repositories, issues, code, or commits")
    per_page = _search_limit(args.get("max_results"), maximum=30)
    endpoint = "https://api.github.com/search/" + endpoint_by_kind[kind] + "?" + urllib.parse.urlencode({"q": query, "per_page": per_page})
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    fetched = _fetch(context, {"url": endpoint, "max_bytes": _MAX_READ_BYTES, "timeout_s": args.get("timeout_s")}, allowed_hosts=("api.github.com",), headers=headers)
    try:
        payload = json.loads(str(fetched.get("content") or "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError("GitHub search returned non-JSON data") from exc
    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        items = []
    reduced = []
    for item in items[:per_page]:
        if not isinstance(item, dict):
            continue
        reduced.append({key: item.get(key) for key in ("name", "full_name", "html_url", "url", "description", "default_branch", "sha", "repository") if key in item})
    return {
        "query": query,
        "kind": kind,
        "results": reduced,
        "count": len(reduced),
        "total_count": payload.get("total_count") if isinstance(payload, dict) else None,
        "source_url": endpoint,
        "retrieved_at": fetched.get("provenance", {}).get("retrieved_at"),
        "auth_available": bool(token or shutil.which("gh")),
        "authenticated": bool(token),
        "credential_values_recorded": False,
    }


def _huggingface_resolve(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    repo = str(args.get("repo") or "").strip()
    revision = str(args.get("revision") or "main").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("repo must be an owner/name Hugging Face repository")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", revision):
        raise ValueError("invalid revision")
    url = f"https://huggingface.co/api/models/{repo}/revision/{urllib.parse.quote(revision, safe='') }?blobs=true"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    result = _fetch(context, {"url": url, "max_bytes": _MAX_READ_BYTES}, headers=headers)
    try:
        payload = json.loads(result["content"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Hugging Face returned non-JSON metadata") from exc
    siblings = payload.get("siblings") if isinstance(payload, dict) else []
    files = []
    for item in siblings or []:
        if not isinstance(item, dict):
            continue
        files.append({key: item.get(key) for key in ("rfilename", "size", "lfs") if key in item})
    return {
        "repo": repo,
        "requested_revision": revision,
        "resolved_revision": payload.get("sha") if isinstance(payload, dict) else None,
        "private": payload.get("private") if isinstance(payload, dict) else None,
        "file_count": len(files),
        "files": files[:1000],
        "source_url": url,
        "retrieved_at": result.get("provenance", {}).get("retrieved_at"),
        "authenticated": bool(token or shutil.which("hf") or shutil.which("huggingface-cli")),
        "credential_values_recorded": False,
    }


def _hf_repo_revision(args: Mapping[str, Any]) -> Tuple[str, str]:
    repo = str(args.get("repo") or "").strip()
    revision = str(args.get("revision") or "main").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("repo must be an owner/name Hugging Face repository")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", revision):
        raise ValueError("invalid revision")
    return repo, revision


def _hf_file_url(args: Mapping[str, Any]) -> Tuple[str, str, str]:
    repo, revision = _hf_repo_revision(args)
    filename = str(args.get("path") or args.get("filename") or "").strip().lstrip("/")
    if not filename or "\x00" in filename or any(part in {"", ".", ".."} for part in filename.split("/")):
        raise ValueError("path must be a non-empty repository-relative file path")
    url = f"https://huggingface.co/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(filename, safe='/')}"
    return repo, revision, url


def _huggingface_fetch_file(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    repo, revision, url = _hf_file_url(args)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    fetched = _fetch(context, {"url": url, "max_bytes": _text_limit(args.get("max_bytes"), default=256 * 1024), "timeout_s": args.get("timeout_s")}, allowed_hosts=("huggingface.co",), headers=headers)
    return {
        "repo": repo,
        "requested_revision": revision,
        "path": str(args.get("path") or args.get("filename")),
        "content": fetched.get("content"),
        "bytes_read": fetched.get("bytes_read"),
        "truncated": fetched.get("truncated"),
        "sha256": fetched.get("sha256"),
        "source_url": fetched.get("final_url") or url,
        "retrieved_at": fetched.get("provenance", {}).get("retrieved_at"),
        "authenticated": bool(token or shutil.which("hf") or shutil.which("huggingface-cli")),
        "credential_values_recorded": False,
    }


def _huggingface_history(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    repo, revision = _hf_repo_revision(args)
    endpoint = f"https://huggingface.co/api/models/{repo}/commits/{urllib.parse.quote(revision, safe='')}"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    fetched = _fetch(context, {"url": endpoint, "max_bytes": min(_MAX_READ_BYTES, 1024 * 1024), "timeout_s": args.get("timeout_s")}, allowed_hosts=("huggingface.co",), headers=headers)
    try:
        payload = json.loads(str(fetched.get("content") or "[]"))
    except json.JSONDecodeError as exc:
        raise ValueError("Hugging Face commit history returned non-JSON data") from exc
    commits = payload if isinstance(payload, list) else payload.get("commits", []) if isinstance(payload, dict) else []
    return {
        "repo": repo,
        "revision": revision,
        "commits": commits[:100],
        "count": len(commits[:100]),
        "source_url": endpoint,
        "retrieved_at": fetched.get("provenance", {}).get("retrieved_at"),
    }


def _model_lake_roots(context: ToolContext) -> Tuple[Path, ...]:
    configured = os.environ.get("HCLI_MODEL_LAKE_ROOT")
    values = [Path(configured).expanduser() if configured else Path("/Volumes/corpdrive/hawking-modellake")]
    values.extend(context.write_roots)
    return tuple(path.resolve(strict=False) for path in values)


def _resolve_download_path(context: ToolContext, raw: Any) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("destination is required")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = context.workspace / candidate
    candidate = candidate.resolve(strict=False)
    if not _within(candidate, _model_lake_roots(context)):
        raise PermissionError("download destination must be inside the workspace or configured ModelLake")
    if _within(candidate, (context.workspace,)):
        try:
            relative = candidate.relative_to(context.workspace)
            if set(relative.parts).intersection({".git", ".hcli"}):
                raise PermissionError("protected AgentOS paths are not download destinations")
        except ValueError:
            pass
    return candidate


def _huggingface_download(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    if args.get("confirm") is not True:
        raise PermissionError("costly Hugging Face downloads require confirm=true")
    repo, revision, url = _hf_file_url(args)
    destination = _resolve_download_path(context, args.get("destination"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    maximum = _text_limit(args.get("max_bytes"), default=64 * 1024 * 1024, maximum=2 * 1024 * 1024 * 1024)
    prior = partial.stat().st_size if partial.is_file() else 0
    if prior > maximum:
        raise ValueError("existing partial download exceeds max_bytes")
    headers = {"User-Agent": "hcli-agentos-research/1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if prior:
        headers["Range"] = f"bytes={prior}-"
    request = urllib.request.Request(_public_url(url, allowed_hosts=("huggingface.co",)), headers=headers, method="GET")

    class _CheckedRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, response_headers, newurl):
            checked = _public_url(urllib.parse.urljoin(req.full_url, newurl), allowed_hosts=("huggingface.co",))
            return super().redirect_request(req, fp, code, msg, response_headers, checked)

    started = time.time()
    resumed = False
    try:
        opener = urllib.request.build_opener(_CheckedRedirectHandler())
        with opener.open(request, timeout=min(120.0, float(args.get("timeout_s") or 60.0))) as response:
            status = int(getattr(response, "status", 0) or 0)
            resumed = prior > 0 and status == 206
            mode = "ab" if resumed else "wb"
            if not resumed:
                prior = 0
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    if prior + int(content_length) > maximum:
                        raise ValueError("download exceeds max_bytes")
                except ValueError:
                    raise
            total = prior
            with partial.open(mode) as handle:
                while True:
                    chunk = response.read(min(1024 * 1024, maximum - total + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > maximum:
                        raise ValueError("download exceeds max_bytes")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            final_url = _public_url(response.geturl(), allowed_hosts=("huggingface.co",))
    except Exception:
        # Preserve the partial for a later explicit resume attempt.
        raise
    digest = _sha256_file(partial)
    expected = str(args.get("expected_sha256") or "").strip().lower()
    if expected and digest != expected:
        raise ValueError(f"download hash mismatch: expected {expected}, observed {digest}")
    os.replace(partial, destination)
    return {
        "repo": repo,
        "requested_revision": revision,
        "destination": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": digest,
        "resumed": resumed,
        "atomic_publish": True,
        "source_url": final_url,
        "retrieved_at": started,
        "authenticated": bool(token),
        "credential_values_recorded": False,
        "download_performed": True,
    }


def _receipt_read(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    raw = str(args.get("path") or "").strip()
    if not raw:
        raise ValueError("path is required")
    candidate = Path(raw).expanduser()
    # Named project evidence is conventionally relative to the repository,
    # while AgentOS mission receipts are relative to the workspace. Keep both
    # explicit and bounded when the two roots differ.
    if not candidate.is_absolute() and (
        raw == "receipts"
        or raw.startswith("receipts/")
        or raw == "civilization"
        or raw.startswith("civilization/")
    ):
        path = (context.repo_root / candidate).resolve(strict=False)
    else:
        path = context.resolve_read_path(raw)
    allowed = (
        context.repo_root / "receipts",
        context.workspace / ".hcli" / "receipts",
        context.workspace / ".hcli" / "mission",
    )
    if not _within(path, allowed):
        raise PermissionError("receipt path must be under receipts or .hcli state")
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        value = raw.decode("utf-8", errors="replace")
    return {"path": str(path), "sha256": _sha256_bytes(raw), "bytes": len(raw), "document": value}


def _context_recall(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Recall bounded older semantic facts without replaying the transcript."""
    from .config import Config
    from .knowledge import KnowledgeStore

    archive_root = Config(str(context.workspace)).value(
        "context_archive_root",
        "HCLI_CONTEXT_ARCHIVE_ROOT",
        None,
    )
    store = KnowledgeStore(context.workspace, archive_root=archive_root)
    return store.recall(
        str(args.get("focus") or ""),
        limit=args.get("max_results", 8),
        max_chars=args.get("max_chars", 8000),
    )


_RECEIPT_TARGETS = {
    "roadmap.read": "civilization/ROADMAP_STATE.json",
    "vmcp.capabilities": "receipts/headless/VMCP_CAPABILITY_SURFACE.json",
    "doctor.inspect": "receipts/headless/DOCTOR_TOURNAMENT.json",
    "gravity.inspect": "receipts/headless/GRAVITY_COMPILER_SEARCH.json",
    "accelerator.inspect": "receipts/headless/ACCELERATOR_MACHINE_GENOME.json",
    "modellake.status": "receipts/headless/MODEL_LAKE_ROLLING_PIPELINE.json",
}


def _target_receipt(name: str) -> Callable[[ToolContext, Dict[str, Any]], Dict[str, Any]]:
    def handler(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        target = args.get("path") or _RECEIPT_TARGETS[name]
        return _receipt_read(context, {"path": target})
    return handler


def _list_tests(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    raw = args.get("root")
    if raw is None or not str(raw).strip():
        root = (context.repo_root / "hcli/tests").resolve(strict=False)
    else:
        root = context.resolve_read_path(raw)
    if not _within(root, context.read_roots):
        raise PermissionError(f"test root is outside the AgentOS read roots: {root}")
    if not root.is_dir():
        raise NotADirectoryError(root)
    # Bounded WALK, not a bounded slice. Enumerating the whole tree and then
    # taking the first 2000 costs the whole tree: measured at 2072 ms against
    # fs.read's 4 ms. Stop when the cap is full.
    limit = max(1, min(2000, int(args.get("max_results") or 2000)))
    found: List[str] = []
    truncated = False
    for path in root.rglob("test_*.py"):
        if len(found) >= limit:
            truncated = True
            break
        if path.is_file():
            found.append(str(path))
    found.sort()
    return {
        "root": str(root),
        "count": len(found),
        "paths": found,
        "truncated": truncated,
    }


def _filesystem_write(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    path = context.resolve_write_path(args.get("path"))
    content = str(args.get("content") or "")
    overwrite = bool(args.get("overwrite", False))
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    before = _sha256_file(path) if path.is_file() else None
    with tempfile.NamedTemporaryFile("w", encoding=str(args.get("encoding") or "utf-8"), dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    after = _sha256_file(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "before_sha256": before,
        "sha256": after,
        "changed": before != after,
        "atomic_publish": True,
    }


def _git_diff(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    cwd = _git_dir(context, args.get("path"))
    raw_paths = args.get("paths") or []
    if not isinstance(raw_paths, list):
        raise ValueError("paths must be an array")
    argv = ["git", "-C", str(cwd), "diff", "--"]
    for raw in raw_paths[:100]:
        candidate = context.resolve_read_path(str(raw))
        try:
            argv.append(str(candidate.relative_to(cwd)))
        except ValueError as exc:
            raise PermissionError("git diff path must be inside the repository") from exc
    return _run_readonly(argv, cwd=cwd, timeout=30.0)


def _git_safe_revert_refusal(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    del args
    return {
        "status": "REFUSED",
        "reason": "safe checkout/revert requires a caller-owned backup and explicit destructive policy; use git.diff first",
        "mutation_performed": False,
    }


def _shell_exec(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    raw = args.get("argv")
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise ValueError("argv must be a non-empty string array")
    argv = [str(item) for item in raw]
    if any(_SHELL_META_RE.search(item) for item in argv):
        raise PermissionError("shell metacharacters are not allowed")
    command = Path(argv[0]).name
    if command in {"python", "python3", Path(sys.executable).name}:
        if "-c" in argv or "-" in argv:
            raise PermissionError("inline Python is not allowed through shell.exec")
        if "-m" in argv:
            try:
                module = argv[argv.index("-m") + 1]
            except (ValueError, IndexError):
                raise ValueError("python -m requires a module") from None
            if module not in {"pytest", "unittest", "compileall"}:
                raise PermissionError("only pytest, unittest, and compileall are allowed")
        elif len(argv) > 1:
            context.resolve_read_path(argv[1])
    elif command == "pytest":
        pass
    elif command == "cargo":
        if len(argv) < 2 or argv[1] not in {"check", "test", "metadata"}:
            raise PermissionError("cargo command is not in the reversible allowlist")
    elif command == "git":
        if len(argv) < 2 or argv[1] not in {"status", "diff", "log", "show", "rev-parse"}:
            raise PermissionError("git command is not read-only")
    else:
        raise PermissionError(f"command is not in the reversible allowlist: {command}")
    cwd = context.resolve_read_path(args.get("cwd") or ".")
    if not cwd.is_dir():
        raise NotADirectoryError(cwd)
    timeout = min(600.0, max(0.1, float(args.get("timeout_s") or 60.0)))
    return _run_readonly(argv, cwd=cwd, timeout=timeout)


def _tests_run(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    runner = str(args.get("runner") or "pytest").lower()
    root = context.resolve_read_path(args.get("root") or ".")
    if not root.is_dir():
        raise NotADirectoryError(root)
    raw_paths = args.get("paths") or []
    if not isinstance(raw_paths, list):
        raise ValueError("paths must be an array")
    paths: List[str] = []
    for raw in raw_paths[:100]:
        path = context.resolve_read_path(raw)
        try:
            paths.append(str(path.relative_to(root)))
        except ValueError as exc:
            raise PermissionError("test paths must be inside the test root") from exc
    if not paths:
        paths = ["."]
    if runner == "pytest":
        argv = [sys.executable, "-m", "pytest", "-q", *paths]
    elif runner == "unittest":
        argv = [sys.executable, "-m", "unittest", "discover", "-s", paths[0]]
    elif runner == "cargo":
        manifest = root / "Cargo.toml"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        argv = ["cargo", "test", "--manifest-path", str(manifest)]
    else:
        raise ValueError("runner must be pytest, unittest, or cargo")
    timeout = min(900.0, max(0.1, float(args.get("timeout_s") or 300.0)))
    started = time.time()
    raw = _run_readonly(argv, cwd=root, timeout=timeout)
    stdout = str(raw.get("stdout") or "")
    return {
        "verified": raw.get("returncode") == 0,
        "returncode": raw.get("returncode"),
        "runner": runner,
        "root": str(root),
        "n_stdout_chars": len(stdout),
        "stdout": stdout,
        "stderr": raw.get("stderr"),
        "argv": raw.get("argv"),
        "cwd": raw.get("cwd"),
        "started_at": started,
        "finished_at": time.time(),
    }


def _vmcp_inspect(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    from .vmcp_adapter import inspect_vmcp

    return inspect_vmcp(context.repo_root, profile=str(args.get("profile") or "core"))


CONTINUATION = "workspace/campaign/odyssey/CONTINUATION.json"

# What a restart cannot rebuild from receipts. Coverage, evidence and priors are
# all derivable -- campaign.state already derives them. Intent is not: no
# receipt records WHY this specimen was chosen over the others, what the live
# hypothesis is, or what the operator meant to do next. Those are the fields.
_CONT_FIELDS = ("objective", "active_specimen", "why_this_specimen", "active_workunit",
                "hypothesis", "next_action", "evidence_refs", "open_jobs",
                "resource_deps", "representations_ruled_out")
_CONT_REQUIRED = ("objective", "hypothesis", "next_action")


def _campaign_checkpoint(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Write the part of the campaign a restart cannot rebuild from disk. [S006 33-36]

    A checkpoint that records what receipts already record is a second copy that
    goes stale. This records only intent, and refuses a checkpoint missing the
    three fields that make it worth reading: what is being pursued, what is
    believed, and what to do next. A checkpoint with no next_action costs a
    restart exactly as much as no checkpoint.
    """
    if WORKSPACE_WRITE not in getattr(context, "permissions", frozenset()):
        return {"refused": "campaign.checkpoint needs workspace_write"}
    missing = [f for f in _CONT_REQUIRED if not str(args.get(f) or "").strip()]
    if missing:
        return {"refused": f"a checkpoint without {missing} does not shorten a restart",
                "required": list(_CONT_REQUIRED)}
    doc = {f: args.get(f) for f in _CONT_FIELDS if args.get(f) is not None}
    doc["written_at"] = int(time.time())
    path = context.repo_root / CONTINUATION
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = None
    if path.is_file():
        try:
            prior = json.loads(path.read_text()).get("written_at")
        except Exception:
            prior = None
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return {"wrote": CONTINUATION, "fields": sorted(doc), "supersedes": prior,
            "note": "campaign.state now leads with this on reattach"}


def _read_continuation(root) -> Optional[Dict[str, Any]]:
    path = root / CONTINUATION
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"unreadable": f"{type(exc).__name__}: {exc}"}


def _campaign_state(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """The campaign in one read: what is being advanced, and what blocks it. [S003 3]

    A campaign operator that must call eight tools to learn where it is will
    spend its round learning where it is. This assembles the answer from disk --
    axis coverage, the most recent evidence, the live priors -- and leads with
    the two things that decide the next move: the dominant bottleneck and how
    many priors would remove an experiment before it is run.

    It ASSEMBLES; it does not judge. Choosing the next move is the operator's job.
    """
    root = context.repo_root
    out: Dict[str, Any] = {}
    led = root / "receipts" / "future" / "G034_ODYSSEY_LEDGER.json"
    axes: Dict[str, Any] = {}
    owed_depth: List[str] = []
    if led.is_file():
        try:
            doc = json.loads(led.read_text())
            counts: Dict[str, Dict[str, int]] = {}
            for row in doc.get("specimens") or []:
                for axis, cell in (row.get("axes") or {}).items():
                    counts.setdefault(axis, {})
                    st = str(cell.get("state"))
                    counts[axis][st] = counts[axis].get(st, 0) + 1
            axes = counts
            # The depth axes are where the campaign is thin; name them first.
            for axis in ("nr_candidate", "cpu", "gpu", "tps", "nx_disposition"):
                c = counts.get(axis) or {}
                if c.get("MEASURED", 0) == 0:
                    owed_depth.append(axis)
        except Exception as exc:
            axes = {"unreadable": f"{type(exc).__name__}: {exc}"}
    priors = _odyssey_priors(context, {"show": 1})
    n_priors = priors.get("total")
    recent: List[Dict[str, Any]] = []
    rdir = root / "receipts" / "future"
    if rdir.is_dir():
        newest = sorted(rdir.glob("*.json"), key=lambda q: q.stat().st_mtime,
                        reverse=True)[:6]
        for q in newest:
            recent.append({"receipt": q.name, "mtime": int(q.stat().st_mtime)})
    cont = _read_continuation(root)
    out = {
        "continuation": cont or (
            "no checkpoint on disk; this restart costs a re-planned campaign. "
            "Write one with campaign.checkpoint at the next transition."),
        "bottleneck": (
            f"depth axes with ZERO measurements: {owed_depth}" if owed_depth else
            "no depth axis is entirely unmeasured; the bottleneck is elsewhere"),
        "depth_axes_never_measured": owed_depth,
        "axis_coverage": axes,
        "n_priors": n_priors,
        "priors_note": ("read them with odyssey.read op=priors BEFORE proposing an "
                        "experiment; a prior that already answers a question "
                        "removes it"),
        "recent_evidence": recent,
        "ledger": str(led.relative_to(root)) if led.is_file() else None,
        "this_tool_assembles_it_does_not_judge": (
            "choosing the next move is the operator's decision, not this tool's"),
    }
    return out


def _vmcp_tools(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """WHICH VMCP tools can HCLI actually call, and what is refused, and why. [S004]

    VisionMCP ships 303 tools; the HCLI bridge allowlists a read-only subset.
    Without this, a resident holding `vmcp` has a door and no idea what is
    behind it -- the same built-but-unreachable shape the campaign keeps
    finding, one level down. Callable names lead; the refusal for everything
    else names its mechanism rather than pretending the rest do not exist.
    """
    from .vmcp_adapter import VMCP_READ_ONLY_TOOLS, _candidate_source_roots, _source_tools

    callable_names = sorted(VMCP_READ_ONLY_TOOLS)
    total = None
    for root in _candidate_source_roots(context.repo_root):
        pkg = root / "visionmcp"
        if pkg.exists():
            try:
                total = len(_source_tools(pkg))
            except Exception:
                total = None
            break
    return {
        "callable": callable_names,
        "n_callable": len(callable_names),
        "n_in_visionmcp": total,
        "how": "vmcp op=query tool=<name> arguments={...}",
        "refused_mechanism": (
            "everything outside this list raises PermissionError from "
            "hcli/vmcp_adapter.py::call_vmcp -- HCLI does not expose the full "
            "VisionMCP laboratory as an untyped escape hatch; mutations and "
            "experimental tools stay behind VMCP's own governed interfaces"),
        "reopen_when": (
            "a specific VMCP tool earns a read-only case and is added to "
            "VMCP_READ_ONLY_TOOLS; the allowlist is the mechanism, not a bug"),
    }


def _vmcp_query(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    from .vmcp_adapter import call_vmcp

    raw_arguments = args.get("arguments") or {}
    if not isinstance(raw_arguments, dict):
        raise ValueError("arguments must be an object")
    return call_vmcp(
        context.repo_root,
        projects_root=context.workspace,
        profile=str(args.get("profile") or "core"),
        tool=str(args.get("tool") or ""),
        arguments=raw_arguments,
    )


def _architecture_inspect(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    from .architecture import ArchitectureRecognizer

    path = context.resolve_read_path(args.get("path"))
    architecture_atlas = None
    atlas_path = context.repo_root / "receipts" / "headless" / "ACCELERATOR_ARCHITECTURE_ATLAS.json"
    try:
        candidate = json.loads(atlas_path.read_text(encoding="utf-8"))
        if isinstance(candidate, Mapping):
            from tools.accelerator.architecture_atlas import validate_atlas

            validate_atlas(candidate)
            architecture_atlas = candidate
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        # Metadata inspection remains useful when the optional planning atlas
        # is absent or stale; it simply returns the historical plan shape.
        architecture_atlas = None
    backend = str(args.get("backend") or "").strip() or None
    return ArchitectureRecognizer(max_tensors=int(args.get("max_tensors") or 250000)).inspect(
        path,
        architecture_atlas=architecture_atlas,
        backend=backend,
    )


def _doctor_query(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    operation = str(args.get("operation") or "census").strip().lower()
    result: Dict[str, Any] = {
        "schema": "hcli.doctor.query.v1",
        "operation": operation,
        "status": "PROPOSED",
        "authority": "tools.doctor.engine",
        "external_research_used": False,
        "physical_execution": False,
    }
    target = args.get("receipt") or args.get("model")
    if target:
        from tools.doctor.engine import diagnose

        result["diagnosis"] = diagnose(target)
        result["status"] = "DIAGNOSED"
        result["producer"] = "tools.doctor.engine.diagnose"
    else:
        from tools.doctor.engine import zeros_controls

        result["doctor_controls"] = zeros_controls()
        result["status"] = "CONTROLLED_PROPOSAL"
        result["producer"] = "tools.doctor.engine.zeros_controls"
    if args.get("model"):
        result["architecture"] = _architecture_inspect(context, {"path": args["model"]})
    if args.get("receipt"):
        result["receipt"] = _receipt_read(context, {"path": args["receipt"]})
    if args.get("research_query"):
        result["research"] = _web_search(context, {"query": args["research_query"], "max_results": args.get("max_results") or 5})
        result["external_research_used"] = True
    if operation in {"techniques", "bottlenecks", "transfer", "propose_experiments", "analyze_negative", "update_laws"}:
        result["next_action"] = "run a bounded protected experiment and persist its measurement receipt"
    else:
        result["next_action"] = "supply a model/organ or a research query for a more specific Doctor proposal"
    return result


def _gravity_experiment(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema": "hcli.gravity.experiment.v1",
        "status": "HYPOTHESIS",
        "model": args.get("model"),
        "organ": args.get("organ") or "unspecified",
        "representation": args.get("representation") or "metadata-only candidate",
        "physical_execution": False,
        "capability_claim": "none",
        "negative_controls_required": True,
        "next_action": "compile a protected native experiment with a deterministic verifier",
    }
    if args.get("model"):
        result["architecture"] = _architecture_inspect(context, {"path": args["model"]})
    if args.get("receipt"):
        result["prior_evidence"] = _receipt_read(context, {"path": args["receipt"]})
    if args.get("execute"):
        from hcli.agentos.flash_representation_experiment import (
            run_flash_representation_experiment,
        )

        experiment = run_flash_representation_experiment(root=args.get("model"))
        result["experiment"] = experiment
        result["status"] = "EXECUTED"
        result["producer"] = "hcli.agentos.flash_representation_experiment.run_flash_representation_experiment"
        result["physical_execution"] = False
        result["capability_claim"] = "none"
    return result


def _accelerator_benchmark(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    if args.get("confirm") is not True:
        raise PermissionError("accelerator benchmarks require confirm=true")
    result: Dict[str, Any] = {
        "schema": "hcli.accelerator.benchmark.v1",
        "status": "RECEIPT_ONLY",
        "physical_execution": False,
        "claim_ceiling": "no speed or hardware claim without live benchmark samples",
    }
    if args.get("receipt"):
        result["receipt"] = _receipt_read(context, {"path": args["receipt"]})
    else:
        result["blocker"] = "provide a named benchmark receipt or a separately governed runner"
    return result


def _benchmark_run(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    if args.get("confirm") is not True:
        raise PermissionError("benchmarks require confirm=true")
    return _tests_run(context, {
        "runner": args.get("runner") or "pytest",
        "root": args.get("root") or ".",
        "paths": args.get("paths") or [],
        "timeout_s": args.get("timeout_s") or 600,
    })


def _frontier_escalate(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    del context
    if args.get("confirm") is not True:
        raise PermissionError("cloud frontier escalation is costly and requires confirm=true")
    from .escalation import escalate_to_frontier

    return escalate_to_frontier(
        args.get("question"),
        args.get("mission_kernel"),
        args.get("artifacts") or [],
        args.get("output_schema") or {"type": "object"},
        model=args.get("model"),
        timeout_s=float(args.get("timeout_s") or 60.0),
    )


def _git_land_propose(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """The resident's only path to a commit. Builds nothing itself -- it just
    forwards the typed proposal into ``landing.propose_landing``, which is
    governed by a deterministic verifier the resident cannot see or skip."""
    from .landing import propose_landing

    return propose_landing(
        context.repo_root,
        branch=args.get("branch"),
        allowed_paths=args.get("allowed_paths") or [],
        test_command=args.get("test_command") or [],
        message=args.get("message"),
        timeout_s=args.get("timeout_s"),
    )


def _odyssey_read(name: str):
    """Read-only Odyssey state. The driver is already running a live mission -
    O003 sealed, O010-O013 queued - so these observe it, never restart it."""

    def handler(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        from . import odyssey

        out = getattr(odyssey, name)()
        return _scope_first(out) if isinstance(out, dict) else out

    return handler


def _odyssey_cycle(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """One Odyssey cycle. Mutating and expensive, so it is gated the way every
    other costly verb here is: an explicit confirm, refused by default."""
    from . import odyssey

    if args.get("confirm") is not True:
        raise PermissionError("odyssey.cycle mutates Odyssey state and requires confirm=True")
    return odyssey.cycle(confirm=True, max_lanes=args.get("max_lanes"))


def _odyssey_gravity_gauntlet(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    from . import odyssey

    if args.get("confirm") is not True:
        raise PermissionError("odyssey.gravity_gauntlet writes search state and requires confirm=True")
    state_path = context.resolve_write_path(args["state_path"]) if args.get("state_path") else None
    receipt_dir = context.resolve_read_path(args["receipt_dir"]) if args.get("receipt_dir") else None
    return odyssey.gravity_gauntlet(
        oxx=str(args["oxx"]),
        candidate_specs=list(args["candidate_specs"]),
        budget=int(args.get("budget") or 2),
        state_path=str(state_path) if state_path else None,
        receipt_dir=str(receipt_dir) if receipt_dir else None,
        confirm=True,
    )


def _forbidden_fruit_lab(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Run the ANE probe lab and report OBSERVED placement.

    No ANE placement has ever been demonstrated on this host; the fixture has
    landed on CPU in every compute plan. This reports MLComputePlan.deviceUsage
    as observed, never the requested compute units, so a CPU result reads as CPU.
    """
    from . import forbidden_fruit

    return forbidden_fruit.run_forbidden_fruit_lab(sdk=args.get("sdk"))


def _frontier_decide(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Which frontier should run next, and why not the others.

    This is the other half of the sovereign stall guard. The loop can park
    itself; without a caller here, nothing picks up the next frontier and a
    parked frontier means an idle machine.
    """
    from . import frontier_scheduler

    return frontier_scheduler.decide().to_dict()


def _future(name: str):
    """Import a tools/future module without putting the REPO ROOT on sys.path.

    tools/future itself IS placed on sys.path, once, and that is the narrow part of the
    original intent that had to give. Modules there import their siblings by plain name --
    dense_anatomy imports lake_scheme_census, dense_sweep imports both dense_anatomy and
    campaign_memory_guard -- and registering under a private `_hcli_future_` key means
    those plain-name imports resolve against nothing. Every sidecar module with a sibling
    was therefore unreachable through this loader, which is the same
    built-but-not-connected shape this session has now hit five times.

    The repo root stays off: that would expose `hcli`, `tools`, `receipts` and the rest.
    The sidecar partition is a much narrower surface, and it is the one these modules were
    written to import from.
    """
    import importlib.util
    import pathlib
    import sys as _sys
    future_dir = pathlib.Path(__file__).resolve().parents[1] / "tools" / "future"
    if str(future_dir) not in _sys.path:
        _sys.path.append(str(future_dir))
    here = future_dir / f"{name}.py"
    if not here.is_file():
        raise FileNotFoundError(
            f"{here} is missing: this tool names a module that does not exist, which is "
            f"how a registered capability becomes unreachable without anyone noticing")
    key = f"_hcli_future_{name}"
    cached = _sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, here)
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE executing: @dataclass resolves cls.__module__ through
    # sys.modules, and without this campaign_memory_guard's Snapshot raised
    # "AttributeError: 'NoneType' object has no attribute '__dict__'" at import.
    _sys.modules[key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        _sys.modules.pop(key, None)
        raise
    return mod


def _selection_brief(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """G018's five inputs, assembled and deliberately unranked.

    Only ONE of the five was reachable from the tool surface before this:
    odyssey.ledger carries measurement debt. campaign_pareto.py computes a
    210-candidate frontier and was imported by two sidecar modules and no tool.
    A selection obligation whose inputs the selector cannot see is not a
    selection obligation.
    """
    m = _future("selection_brief")
    return m.brief(limit=_shown_limit(args.get("limit")))


def _lake_census(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Reachability and complete EBPW for every specimen, from headers only.

    Reads no payload bytes: the whole 4.29 TiB lake classifies in about twelve
    seconds. Answers, per body, whether an expert organ exists, whether it is
    pre-quantized on disk, and what it actually costs in bits per source
    parameter.
    """
    m = _future("lake_scheme_census")
    catalog = str(args.get("catalog") or "receipts/future/modellake-index/catalog.json")
    slug = str(args.get("slug") or "").strip()
    if slug:
        import json as _json
        cat = _json.load(open(catalog))
        row = next((x for x in cat["specimens"] if x["slug"] == slug), None)
        if row is None:
            raise KeyError(f"{slug} is not in {catalog}")
        out = m.classify(row["path"])
        # The catalog knew the path all along and the census kept it to itself. A round
        # that selects a body here and reaches for an anatomy tool needs a DIRECTORY, and
        # a slug is not one -- rounds 13 and 15 both handed the anatomy tool the ledger's
        # own path because it was the only path they had.
        out["snapshot"] = row["path"]
        try:
            out.update(m.accounting(row["path"], m.confirm_pack_factor(row["path"])))
        except Exception as exc:
            out["accounting_error"] = f"{type(exc).__name__}: {exc}"
        out["slug"] = slug
        return _lead_with(out, "slug", "klass", "complete_ebpw", "pack_factor", "blocked_by")
    census = m.census(catalog)
    rows = list(census.get("rows") or [])
    # A VALUE IS NOT AN OWED CELL. Round 20 read `complete_ebpw: 16.0` here,
    # concluded "16.0 EBPW is worst owed", re-derived a number the ledger had
    # already recorded as MEASURED, and closed having moved nothing: MEASURED
    # stayed at 94. Nothing in this row said the axis was already resolved, so
    # the round inferred owed-ness from the only thing it could see -- the
    # magnitude. Carry the ledger state beside the value so the two cannot be
    # confused. Missing or unreadable ledger degrades to None, never to a
    # cheerful default that would recreate the same mistake.
    owed_by_slug: Dict[str, Any] = {}
    try:
        import json as _json
        _led = _json.load(open("receipts/future/G034_ODYSSEY_LEDGER.json"))
        owed_by_slug = {
            r["slug"]: sorted(a for a, v in r["axes"].items() if v["state"] == "OWED")
            for r in _led["specimens"]
        }
    except Exception:
        owed_by_slug = {}
    compact = [
        {
            "slug": row.get("slug"),
            "klass": row.get("klass"),
            "gib": row.get("gib"),
            "complete_ebpw": row.get("complete_ebpw"),
            "ebpw_axis_state": (
                "OWED" if "ebpw" in owed_by_slug.get(row.get("slug"), [])
                else ("RESOLVED" if row.get("slug") in owed_by_slug else None)
            ),
            "owed_axes": owed_by_slug.get(row.get("slug")),
            "pack_factor": row.get("pack_factor"),
            "blocked_by": row.get("blocked_by"),
            "family": row.get("family"),
        }
        for row in rows
    ]
    top = _shown_limit(args.get("limit"))
    return {
        "rows": compact[:top],
        "n": census.get("n", len(rows)),
        "shown": min(top, len(compact)),
        "truncated": len(compact) > top,
        "tally": census.get("tally"),
        "expert_anatomy_reachable": census.get("expert_anatomy_reachable"),
        "reachable_gib": census.get("reachable_gib"),
        "blocked_gib": census.get("blocked_gib"),
    }


def _odyssey_ledger(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """What each specimen has measured, refused, or still owes.

    Every specimen owes seven axes or an explicit recorded reason. This is the
    state HCLI needs to choose what to work on next without a human naming it.
    """
    m = _future("odyssey_ledger")
    import json as _json
    path = str(args.get("path") or "receipts/future/G034_ODYSSEY_LEDGER.json")
    led = _json.load(open(path))
    prog = m.progress(led)
    slug = str(args.get("slug") or "").strip()
    if slug:
        rec = next((r for r in led["specimens"] if r["slug"] == slug), None)
        if rec is None:
            raise KeyError(f"{slug} is not in {path}")
        # `path` travels with the answer so a round can chain ledger -> record without
        # anyone pasting the literal file in. Round 7 read the ledger with fs.read --
        # and got 1.3% of it -- because the record tool's example in its prompt carried
        # the raw path, which put a file in front of it.
        snap = None
        try:
            with open("receipts/future/modellake-index/catalog.json") as fh:
                snap = next((x["path"] for x in _json.load(fh)["specimens"]
                             if x["slug"] == slug), None)
        except Exception:
            snap = None
        return {"specimen": rec, "progress": prog, "path": path, "snapshot": snap}
    if args.get("owed_only"):
        # Carry the SNAPSHOT PATH, not just the slug. Every anatomy tool takes a directory
        # and every discovery tool returned a name, with nothing in the registry converting
        # one to the other -- so a round that selected correctly still had to guess, and
        # twice guessed the ledger's own path.
        paths = {}
        try:
            with open("receipts/future/modellake-index/catalog.json") as fh:
                paths = {x["slug"]: x["path"] for x in _json.load(fh)["specimens"]}
        except Exception:
            paths = {}
        owed = [{"slug": r["slug"], "gib": r["gib"], "class": r["class"],
                 "owed": [a for a, v in r["axes"].items() if v["state"] == "OWED"],
                 "snapshot": paths.get(r["slug"])}
                for r in led["specimens"]
                if any(v["state"] == "OWED" for v in r["axes"].values())]
        owed.sort(key=lambda r: (-len(r["owed"]), r["gib"]))
        # ACTIONABLE FIRST, AND BOUNDED. The full list is 8091 characters and the
        # closed-turn compactor keeps 500 -- so with `progress` emitted first, the
        # resident received the aggregate summary and NONE of the specimen names.
        # It called this tool four times and could not choose a target, because the
        # only part that names one was in the truncated tail. A tool whose useful
        # half does not survive the caller's budget is a tool that does not work.
        top = int(args.get("limit") or 12)
        shown = min(top, len(owed))
        tail = owed[shown:]
        # DESCRIBE THE TAIL, do not make the caller go and find it. Bounding the view
        # fixed the original defect -- an 8 KB summary naming not one specimen -- and
        # created a new one: a careful caller told that 34 items are hidden goes looking
        # for them. Two consecutive rounds spent their entire observation budget on that
        # search. Round 11 correctly DERIVED that the hidden bodies must all be >= the
        # largest visible one, because this list is sorted smallest-first within an owed
        # count, and then read the 301 KB ledger in ten windows to confirm it. It should
        # not have had to do either.
        hidden = {
            "n": len(tail),
            "gib_min": min((r["gib"] for r in tail), default=None),
            "gib_max": max((r["gib"] for r in tail), default=None),
        }
        return {
            "owed": owed[:top],
            "n_owed": len(owed),
            "shown": shown,
            "hidden": hidden,
            # The list IS sorted and never said how, so "the worst" was ambiguous: most
            # axes owed, or the largest body? Say it, in the view itself.
            "ordering": ("most axes owed first, then smallest GiB first "
                         "(cheapest to measure among equals)"),
            "path": path,
            "summary": (f"{prog['axes_resolved']}/{prog['axes_total']} axes resolved "
                        f"({prog['pct']}%), {prog['specimens_complete']} specimens complete"),
        }
    return {"progress": prog, "n": len(led["specimens"]), "path": path}


def _odyssey_anatomy(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Representational anatomy of one specimen's expert organ.

    Refuses with a named mechanism rather than returning an empty anatomy: a
    body with no safetensors, no expert organ, or a pre-quantized payload is
    recorded as measured-and-impossible, never as measured-and-silent.
    """
    m = _future("representational_anatomy")
    snapshot = str(args.get("snapshot") or "").strip()
    if not snapshot:
        raise ValueError("snapshot path is required")
    layer = args.get("layer", 0)
    try:
        out = m.anatomy_from_safetensors(snapshot, layer=layer)
    except m.AnatomyUnavailable as exc:
        return {"refused": str(exc), "anatomy": None, "snapshot": snapshot}
    return _lead_with(out, "hypotheses", "layer", "scheme", "storage", "snapshot")


def _odyssey_record_measurement(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Move one ledger axis from OWED to MEASURED or REFUSED. The write door.

    odyssey.anatomy RETURNS an anatomy and persists nothing; odyssey.ingest is
    read_only; record_law and record_scar record Laws and Scars, not axis cells.
    So a round that measured something correctly saw the ledger unchanged and
    asked again -- which is what closed round 6 on the all-repeat closer after six
    successful observations and zero failures.

    The guards are odyssey_ledger's own, imported rather than restated: a value
    without a receipt is not evidence, and a refusal under 20 characters is how an
    unmeasured axis disguises itself as a finding. Those two functions existed with
    ZERO callers anywhere in the repo -- built and never connected.

    `path` is REQUIRED and has no default. A write tool whose default target is the
    campaign's own live ledger is a foot-gun, and the caller has to name what it is
    writing to.
    """
    m = _future("odyssey_ledger")
    import json as _json
    # The schema makes `path` REQUIRED, which stops a caller omitting it. It does not
    # stop an EMPTY string, and that is the case this check owns -- deleting it is
    # detectable, deleting a restatement of the schema was not.
    path = str(args.get("path") or "").strip()
    if not path:
        raise ValueError(
            "path is empty: this tool WRITES, and it will not guess which ledger")
    target = context.resolve_write_path(path) if hasattr(context, "resolve_write_path") \
        else Path(path)
    led = _json.loads(Path(target).read_text())
    slug = str(args.get("slug") or "").strip()
    rec = next((r for r in led["specimens"] if r["slug"] == slug), None)
    if rec is None:
        raise KeyError(f"{slug!r} is not a specimen in {path}")
    axis = str(args.get("axis") or "").strip()
    reason = args.get("reason")
    if reason not in (None, ""):
        text = str(reason)
        # A refusal is EVIDENCE ABOUT A SPECIMEN. The ledger being written to is never
        # evidence about anything inside it, and self-reference is how a bad tool call
        # gets stored as a scientific finding. Round 13 passed this very file to
        # odyssey.dense_anatomy as a snapshot, got a correct complaint about that
        # argument, and wrote it onto a specimen's nr_candidate axis. The existing guard
        # could not see it: the reason is long and does name a mechanism -- just not one
        # about the body.
        #
        # Deliberately narrow. "Is this reason RELEVANT" cannot be checked, and demanding
        # the slug would reject both legitimate refusals already on disk, neither of which
        # names its own.
        target_name = Path(path).name
        if target_name in text or str(target) in text or path in text:
            raise m.LedgerError(
                f"{slug}/{axis}: this refusal is about {target_name}, the ledger being "
                f"written to, not about the specimen. A tool-call error is not a finding. "
                f"Record what the BODY refuses, with the mechanism the body gave you."
            )
        # OPTIONAL, and recorded when given. 121 of 142 refusals on the live
        # ledger name a mechanism and no way back, which makes them permanent by
        # accident. Required would reject the next refusal a round writes for a
        # field it has never been asked for; accepted lets the gap close and be
        # measured while it does.
        m.refused(rec, axis, text, reopen_when=args.get("reopen_when"))
    else:
        receipt = str(args.get("receipt") or "").strip()
        # SAME invariant as the refusal guard above, other field: a cell's evidence cannot
        # be the file the cell lives in. Round 16 closed the loop and wrote
        # nr_candidate = "16.0" citing receipts/future/G034_ODYSSEY_LEDGER.json -- circular,
        # and both existing guards let it through, because the refusal guard only inspects
        # `reason` and the existence check passes on a file that obviously exists.
        if receipt and (Path(receipt).name == Path(path).name
                        or os.path.realpath(receipt) == os.path.realpath(str(target))):
            raise m.LedgerError(
                f"{slug}/{axis}: the ledger cannot be its own receipt. A cell's evidence "
                f"must be a file that records the MEASUREMENT, not the file the cell lives "
                f"in. Write one first with filesystem.write, then cite it here.")
        # Existence is enforced by odyssey_ledger.measured, but its message cannot know
        # about tools. A round that has no way to MAKE a receipt will cite whatever file it
        # already knows -- so the failure names the write path rather than adding a second
        # one. [S008 3] ONE OWNER, ONE GUARD, ONE WRITE PATH.
        if receipt and not Path(receipt).exists():
            raise m.LedgerError(
                f"{slug}/{axis}: receipt {receipt!r} does not exist yet. Write the finding "
                f"with filesystem.write (path under receipts/future/), then record the cell "
                f"citing that path.")
        # No receipt-EMPTY check here on purpose: odyssey_ledger.measured already refuses a
        # value without one, and restating a guard is how two copies drift apart. A
        # mutation that deleted a duplicate check here stayed green, which is the tell.
        m.measured(rec, axis, args.get("value"), receipt)
    tmp = Path(str(target) + ".tmp")
    tmp.write_text(_json.dumps(led, indent=1) + "\n")
    tmp.replace(target)
    prog = m.progress(led)
    return _lead_with(
        {"recorded": {"slug": slug, "axis": axis, **rec["axes"][axis]},
         "axes_resolved": prog.get("axes_resolved"),
         "axes_owed": prog.get("axes_owed"),
         "path": path},
        "recorded", "axes_owed", "axes_resolved")


def _odyssey_dense_anatomy(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Within-tensor anatomy of one DENSE specimen's organs, against a same-shape null.

    odyssey.anatomy measures the EXPERT organ and refuses a dense body by saying so:
    "no key contains 'expert'; this looks dense, not MoE". A round that read that refusal
    correctly then had nowhere to go, because dense anatomy had no door -- the capability
    measured 34 bodies for G002 through dense_anatomy.anatomy_from_safetensors and was
    simply unreachable from here. Use odyssey.anatomy for a MoE body and this for a dense
    one; each refusal names the other.

    The guard is asked BEFORE the body is opened, from a header-only size estimate. Dense
    anatomy ran 796 s at 6.59 GiB peak on the largest body in the sweep, so a STOP verdict
    has to be a refusal with its numbers rather than a machine at risk.
    """
    cmg = _future("campaign_memory_guard")
    da = _future("dense_anatomy")
    sweep = _future("dense_sweep")
    snapshot = str(args.get("snapshot") or "").strip()
    if not snapshot:
        raise ValueError("snapshot path is required")
    est = sweep.estimate_from_headers(snapshot)
    snap = cmg.sample(expected_gb=float(est.get("expected_gb") or 0.0))
    if snap.state == "STOP":
        return _lead_with({
            "refused": (f"{snapshot}: campaign guard says STOP before any payload was read "
                        f"-- free {snap.free_gb} GB, compressor {snap.compressor_gb} GB, "
                        f"swapfiles {snap.swapfiles}, this body needs about "
                        f"{est.get('expected_gb')} GB. Refusal, not a crash."),
            "anatomy": None, "guard": snap.as_dict(), "snapshot": snapshot,
        }, "refused", "guard", "snapshot")
    try:
        out = da.anatomy_from_safetensors(snapshot)
    except da.DenseAnatomyUnavailable as exc:
        return _lead_with({"refused": str(exc), "anatomy": None, "snapshot": snapshot,
                           "guard": snap.as_dict()},
                          "refused", "snapshot")
    ordering = sweep.organ_ordering_from_anatomy(out)
    return _lead_with({
        "organ_ordering": ordering,
        "hypotheses": out.get("hypotheses"),
        "n_organs": len(ordering),
        "snapshot": snapshot,
        "guard": snap.as_dict(),
        "anatomy": out,
    }, "organ_ordering", "hypotheses", "n_organs", "snapshot")


def _campaign_guard(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Host memory and swap headroom before an expensive run.

    Reads free pages, compressor size, swapfile count and resident RSS. The
    swapfile count is the live signal: vm.swapusage used is a boot high-water
    mark, not a current reading.
    """
    m = _future("campaign_memory_guard")
    snap = m.sample(expected_gb=float(args.get("expected_gb") or 0.0))
    return {"state": snap.state, "reasons": list(snap.reasons),
            "free_gb": snap.free_gb, "compressor_gb": snap.compressor_gb,
            "swapfiles": snap.swapfiles, "wired_gb": snap.wired_gb,
            "expected_gb": snap.expected_gb, "headroom_gb": snap.headroom_gb}


def _specimens_registry(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Every sealed specimen, enumerated from disk. SEALED != LOAD NOW."""
    from . import specimens

    name = str(args.get("name") or "").strip()
    if name:
        found = specimens.get(name)
        return {"name": name, "specimen": found, "found": found is not None}
    data = specimens.registry()
    rows = list(data.get("specimens") or [])
    compact = [
        {
            "id": row.get("id"),
            "size_bytes": row.get("size_bytes"),
            "model_type": (row.get("architecture") or {}).get("model_type"),
            "verified_complete": row.get("verified_complete"),
        }
        for row in rows
    ]
    top = _shown_limit(args.get("limit"))
    # SCOPE FIRST. The observation budget cuts this to 500 chars as head+tail,
    # and with `specimens` leading, n_specimens / shown / truncated fell in the
    # elided middle -- the round saw rows and could not tell how many existed.
    return _lead_with({
        "specimens": compact[:top],
        "n_specimens": data.get("n_specimens"),
        "shown": min(top, len(compact)),
        "truncated": len(compact) > top,
        "mounted": data.get("mounted"),
        "lake": data.get("lake"),
        "specimens_dir": data.get("specimens_dir"),
        "schema": data.get("schema"),
        "reason": data.get("reason"),
        "sealed_does_not_mean_resident": data.get("sealed_does_not_mean_resident"),
    }, "n_specimens", "shown", "truncated")


def _acquisition_propose(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Rank what to acquire next. Proposes and explains; never starts a
    download - that stays behind explicit confirmation elsewhere."""
    from . import acquisition

    raw = acquisition.propose()
    if not isinstance(raw, dict):
        return raw
    ranked = list(raw.get("ranked") or [])
    top = _shown_limit(None)
    leading = {
        "recommended": raw.get("recommended"),
        "recommendation_reason": raw.get("recommendation_reason"),
        "ranked": ranked[:top],
        "n_ranked": len(ranked),
        "shown": min(top, len(ranked)),
        "truncated": len(ranked) > top,
        "list_order_pick": raw.get("list_order_pick"),
        "list_order_would_redownload_sealed": raw.get("list_order_would_redownload_sealed"),
    }
    rest = {
        key: value for key, value in raw.items()
        if key not in leading
    }
    leading.update(rest)
    # Its own leading block already puts the recommendation first, which is the
    # right instinct -- but n_ranked / shown / truncated sat behind the ranked
    # rows and did not survive the 500-char observation cut.
    return _scope_first(leading)


def _odyssey_read_verb(name: str, required: Sequence[str] = ()):
    """Read-only Odyssey verbs. ``required`` names the positional arguments the
    connector declares; a verb that takes one and is wired without it raises
    TypeError on first call, so the names are forwarded rather than dropped."""

    def handler(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        from . import odyssey

        out = getattr(odyssey, name)(*(str(args[key]) for key in required))
        return _scope_first(out) if isinstance(out, dict) else out

    return handler


def _odyssey_mutating(name: str, required: Sequence[str]):
    """Odyssey verbs that change campaign state. Gated exactly like
    odyssey.cycle: refused without an explicit confirm."""

    def handler(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        from . import odyssey

        if args.get("confirm") is not True:
            raise PermissionError(f"odyssey.{name} changes Odyssey state and requires confirm=True")
        kwargs = {k: args[k] for k in required if k in args}
        for extra in ("note", "reason", "evidence", "source_oxx", "attack", "description"):
            if extra in args:
                kwargs[extra] = args[extra]
        return getattr(odyssey, name)(confirm=True, **kwargs)

    return handler


def _grok_swarm_propose(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    del context
    from .escalation import propose_swarm

    return propose_swarm(args.get("problem_statement"), args.get("lanes") or [])


def _grok_swarm_launch(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    if args.get("confirm") is not True:
        raise PermissionError("Grok swarm launches are costly and require confirm=true")
    from .escalation import launch_swarm

    return launch_swarm(
        context.workspace,
        args.get("problem_statement"),
        args.get("lanes") or [],
        mode=str(args.get("mode") or "audit"),
        dry_run=args.get("dry_run"),
    )


def _processes_list(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Every live Hawking process, classified by the native Rust authority.
    Read-only: this wraps the Python compatibility skin and nothing else. Killing a process is
    deliberately NOT reachable here -- that stays on the owned-signal path in
    hcli/agentos/resident.py (_owned_signal), which checks a
    process_start_token before it will ever send a signal.
    """
    del args
    from . import processes

    procs = list(processes.live_processes(workspace=context.workspace))
    index = [
        {
            "pid": p.pid,
            "role": p.role,
            "rss_gib": round(p.rss_bytes / 1024 ** 3, 3),
            "safe_to_stop": p.safe_to_stop,
        }
        for p in procs
    ]
    top = _shown_limit(None)
    return {
        "index": index[:top],
        "n_processes": len(procs),
        "shown": min(top, len(index)),
        "truncated": len(index) > top,
        "processes": [p.to_dict() for p in procs],
    }


def _processes_summary(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Roll-up counts and total footprint for every live Hawking process --
    the same native Rust entry point the process audit receipt uses. Read-only."""
    del args
    from . import processes

    return processes.summary(workspace=context.workspace)


def _processes_orphaned(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Resident model bodies with no live owner (reparented to pid 1, unclaimed
    by any resident state file). Enumeration only: this calls the native Rust
    inspector, never the reaper, which sends SIGTERM. Reaping stays a startup-only self-heal in
    hcli/runtime.py._reap_orphans_once, not something the model can trigger
    through the tool surface.
    """
    del args
    from . import processes

    procs = list(processes.orphaned_resident_bodies(workspace=context.workspace))
    index = [
        {
            "pid": p.pid,
            "role": p.role,
            "rss_gib": round(p.rss_bytes / 1024 ** 3, 3),
            "safe_to_stop": p.safe_to_stop,
        }
        for p in procs
    ]
    top = _shown_limit(None)
    return {
        "index": index[:top],
        "n_orphaned": len(procs),
        "shown": min(top, len(index)),
        "truncated": len(index) > top,
        "orphaned": [p.to_dict() for p in procs],
    }



# ---------------------------------------------------------------------------
# NR->NX doors. [S010 22, 35, 37, 49, 51]
#
# complete_ebpw.py, physical_emitter.py and nx_promotion.py were all built and
# then reachable from nothing but their own tests. A capability nothing calls
# does not exist, and the campaign has now been bitten by that four times. The
# three handlers below are DOORS, not new subsystems: each one calls the single
# existing authority and refuses rather than substituting a guess. No second
# biller and no second emitter is written here.
#
# A51 -- every required argument has a reachable source: physical.emit takes a
# round id that physical.rounds enumerates; nr.complete_ebpw takes either the
# sealed incumbent (no arguments) or a candidate the caller declares in full,
# which is exactly the contract complete_ebpw already refuses to guess at.
# ---------------------------------------------------------------------------

_PRIOR_SOURCES = (
    ("law_store", "receipts/future/ODYSSEY2_LAW_STORE.json", "laws"),
    ("scars", "receipts/future/CAMPAIGN_SCARS.json", "scars"),
    ("hcli_ledger", "workspace/campaign/odyssey/HCLI_LEDGER.json", "laws"),
    ("hcli_scars", "workspace/campaign/odyssey/HCLI_LEDGER.json", "scars"),
)


def _prior_row(kind: str, source: str, rec: Mapping[str, Any]) -> Dict[str, Any]:
    """One prior, actionable first: what it says, where it holds, what reopens it."""
    statement = (rec.get("statement") or rec.get("text") or rec.get("description")
                 or rec.get("scar") or "")
    # DOMAIN first, then the claim. A 400-char statement in front of the domain
    # means the domain is what the observation budget cuts -- and a law quoted
    # without its domain is how a model-local result gets applied to an
    # architecture nobody tested. Statements are clipped here on purpose; the
    # full text is one receipt.read away and the id says which.
    return {
        "id": rec.get("law_id") or rec.get("id") or rec.get("scar_id") or "?",
        "domain": rec.get("scope") or rec.get("architecture_family") or rec.get("organ_class"),
        "kind": kind,
        "says": str(statement)[:150],
        # REOPEN. What would make this false again, or measurable again.
        "reopen": str(rec.get("counterexample_requirement") or rec.get("reopen_when")
                      or rec.get("cheapest_check") or "")[:150] or None,
        "evidence": rec.get("evidence_refs") or rec.get("evidence") or rec.get("caught_by"),
        "source": source,
    }


def _odyssey_priors(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """What the campaign already knows, so a round does not re-buy it. [S001 47, 119]

    HCLI has had odyssey.record_law and odyssey.record_scar -- two WRITE doors --
    and no way to read either back. A memory you can only write to is not a
    memory, and its ledger (workspace/campaign/odyssey/HCLI_LEDGER.json) had never
    been created, so nothing had ever been recorded through it either.

    Every row leads with what the prior SAYS, then where it HOLDS (domain) and
    what would REOPEN it. A law quoted without its domain is how a
    model-local result gets applied to an architecture nobody tested.
    """
    focus = str(args.get("focus") or "").strip().lower()
    shown = _shown_limit(args.get("show"), default=10, maximum=40)
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for source, rel, key in _PRIOR_SOURCES:
        path = context.repo_root / rel
        if not path.is_file():
            missing.append(rel)
            continue
        try:
            doc = json.loads(path.read_text())
        except Exception as exc:
            missing.append(f"{rel} ({type(exc).__name__})")
            continue
        for rec in (doc.get(key) or []):
            if isinstance(rec, Mapping):
                rows.append(_prior_row(key.rstrip("s"), source, rec))
    if focus:
        rows = [r for r in rows
                if focus in json.dumps(r, default=str).lower()]
    no_domain = [r["id"] for r in rows if not r["domain"]]
    # Deliberately NOT _truncation_fields: its note is 179 characters of prose
    # about file bytes, which is both wrong here and enough on its own to push
    # the first prior's domain past the observation budget.
    return {
        "n": len(rows[:shown]),
        "total": len(rows),
        "priors": rows[:shown],
        "focus": focus or None,
        "without_domain": no_domain[:8],
        "sources_missing": missing,
        "truncated": len(rows) > shown,
        "how_to_use": ("a prior REMOVES search. Check domain before applying one "
                       "to a different architecture; check reopen before treating "
                       "it as permanent."),
    }



# ---------------------------------------------------------------------------
# HCLI AS ADVERSARIAL AUDITOR. [S002]
#
# These are the checks the supervisor was running by hand this session, made
# callable. Each one found a real defect the first time it was run manually:
# an experiment whose arms never shared a bit-depth multiset, a receipt whose
# verdict was written under a superseded gate, a capability authority with no
# caller, and a law store with two write doors and no read door.
# ---------------------------------------------------------------------------

def _odyssey_attack_law(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """OIII: generate ranked executable attacks against one recorded law.

    The adversary was reachable only from an acceptance runner, so the loop
    LAW -> ATTACK -> RESULT -> SCOPE UPDATE had no entry point from the side
    that writes the laws. It does now: HCLI records a law through
    odyssey.record_law, reads it back through odyssey.priors, and attacks it
    here. A law that emits no attack is refused rather than quietly published.

    These are SPECS, not measurements -- static, bench UNKNOWN. Running one is
    a separate act.
    """
    _future_tools_on_path(context)
    law_id = str(args.get("law_id") or "").strip()
    if not law_id:
        return {"refused": "law_id is required; odyssey.priors lists them"}
    rows = (_odyssey_priors(context, {"focus": law_id.lower(), "show": 20})
            .get("priors") or [])
    match = next((r for r in rows if str(r.get("id")) == law_id), None)
    if match is None:
        return {"refused": f"{law_id} is not a recorded law or scar",
                "hint": "odyssey.priors lists what is recorded"}
    if not match.get("domain"):
        return {"refused": f"{law_id} has no domain; a law with no stated scope "
                           "cannot be attacked on scope, and attacking it on "
                           "anything else would be attacking a guess",
                "fix": "re-record it with a domain"}
    # The campaign runs TWO scope vocabularies: the law store writes
    # ARCHITECTURE_FAMILY / GENERIC_CANDIDATE, and the adversary's ladder
    # accepts neither. Map conservatively -- never UPGRADE a scope, because a
    # law attacked at a broader scope than it was recorded at gets refuted for
    # a claim nobody made -- and refuse anything with no conservative mapping.
    ladder = {"GENERIC_VERIFIED", "FAMILY_VERIFIED", "MODEL_LOCAL",
              "ORGAN_LOCAL", "DEVICE_LOCAL", "MACHINE_LOCAL"}
    downgrade = {"ARCHITECTURE_FAMILY": "FAMILY_VERIFIED",
                 "GENERIC_CANDIDATE": "MODEL_LOCAL"}
    raw_domain = str(match["domain"])
    scope = raw_domain if raw_domain in ladder else downgrade.get(raw_domain)
    if scope is None:
        return {"refused": f"{law_id} has domain {raw_domain!r}, which is on neither "
                           "the law store's vocabulary nor the adversary's scope "
                           "ladder; attacking it would mean inventing its scope",
                "ladder": sorted(ladder), "known_mappings": downgrade}
    law = {
        "law_id": match["id"],
        "statement": match["says"],
        "scope": scope,
        "evidence_refs": (match.get("evidence") if isinstance(match.get("evidence"), list)
                          else [match.get("evidence")] if match.get("evidence") else []),
        "counterexample_requirement": match.get("reopen") or "",
        "source_model": args.get("source_model") or "UNKNOWN",
        "source_device": "UNKNOWN",
        "architecture_family": args.get("architecture_family") or "UNKNOWN",
        "organ_class": args.get("organ_class") or "UNKNOWN",
        "backend": "UNKNOWN",
        "evidence_strength": "DIAGNOSTIC_RELATIVE",
        "transfer_candidates": [],
        # Neutral default, NOT a measurement: laws recorded through
        # odyssey.record_law carry no confidence figure, and inventing a
        # confident one would bias which attacks the ranker prefers.
        "transfer_confidence": 0.5,
    }
    try:
        from tools.future import odyssey3_adversary as o3  # type: ignore
        ranked = o3.rank_attacks(o3.generate_attacks(law))
    except Exception as exc:
        return {"experiment_failed": f"{type(exc).__name__}: {exc}",
                "not_a_specimen_property": True, "law_id": law_id}
    if not ranked:
        return {"refused": f"{law_id} emitted no attack; an unattackable law is "
                           "not a published law"}
    shown = _shown_limit(args.get("show"), default=5, maximum=20)
    # Field names are the adversary's own ATTACK_SPEC_FIELDS. Guessing them
    # produced a row of nulls that looked like a working tool.
    lead = [{"attack_id": a.get("attack_id"), "family": a.get("family"),
             "cost_units": a.get("cost_units"),
             "p_refutation": a.get("p_refutation"),
             "selection_score": a.get("selection_score"),
             "falsifier": str(a.get("falsifier") or "")[:200],
             "adversarial_target": str(a.get("adversarial_target") or "")[:160],
             "scope_if_refuted": a.get("target_scope_if_refuted"),
             "command": a.get("command")}
            for a in ranked[:shown]]
    return {
        "law_id": law_id,
        "domain": raw_domain,
        "scope_used": scope,
        "scope_note": (None if scope == raw_domain else
                       f"recorded domain {raw_domain} mapped DOWN to {scope} for the "
                       "adversary's ladder; attacks are judged at the narrower scope"),
        "transfer_confidence_is_a_neutral_default": 0.5,
        "n_attacks": len(ranked),
        "attacks": lead,
        "evidence_class": "STATIC_ONLY",
        "note": "specs, not measurements; running one is a separate act",
    }


def _capability_gate(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """THE capability authority: perplexity AND n-gram diversity, G020 bars.

    Also answers "is this receipt's stored verdict still true?" -- pass the
    numbers a receipt recorded and compare. A receipt written under the
    pre-G020 bar recorded a diversity bar of 3 x dense, which for a dense r4
    of 0.3619 is 1.0857 -- ABOVE the metric's own [0,1] range, so nothing could
    fail it and the conjunction was perplexity wearing a second name.
    """
    _future_tools_on_path(context)
    try:
        need = ("ppl", "r4", "dense_ppl", "dense_r4")
        vals = {}
        for k in need:
            if args.get(k) is None:
                return {"refused": f"{k} is required; the gate is measured against "
                                   "the specimen's OWN dense parent, not a constant"}
            vals[k] = float(args[k])
        import organ_allocation as oa  # type: ignore
    except Exception as exc:
        return {"experiment_failed": f"{type(exc).__name__}: {exc}",
                "not_a_specimen_property": True}
    live = oa.gate_from_reference(
        {"ppl_full": vals["ppl"], "r4_full": vals["r4"]},
        {"ppl_full": vals["dense_ppl"], "r4_full": vals["dense_r4"]})
    out = {
        "capability_ok": live["capability_ok"],
        "ppl_ok": live["ppl_ok"],
        "diversity_ok": live["diversity_ok"],
        "gate_ppl_max": live["gate_ppl_max"],
        "gate_r4_max": live["gate_r4_max"],
        "gate_r4_capped": live["gate_r4_capped"],
        "rule": live["gate_r4_cap_rule"],
        "authority": live["evaluator"],
    }
    claimed = args.get("recorded_capability_ok")
    if claimed is not None:
        stale = bool(claimed) != bool(live["capability_ok"])
        out["stale_verdict"] = stale
        out["recorded_capability_ok"] = bool(claimed)
        out["why"] = ("the stored verdict disagrees with the live gate; the "
                      "evidence was written under a superseded bar and must be "
                      "re-derived, not cited" if stale else
                      "the stored verdict still holds under the live gate")
    return out


def _experiment_confound(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Before believing an ORDERING result, check the arms were otherwise equal.

    Reads a receipt's `arms` and reports, per arm, the total organ bytes, the
    complete EBPW and the MULTISET of bit depths. If those differ between arms,
    a difference in capability is not attributable to the ordering -- it is
    attributable to whichever of them moved. This is the check that showed
    G021's monotone arms never shared a depth multiset.
    """
    raw = args.get("receipt")
    if not raw:
        return {"refused": "receipt path is required"}
    path = context.resolve_read_path(raw) if hasattr(context, "resolve_read_path") \
        else Path(raw)
    try:
        doc = json.loads(Path(path).read_text())
    except Exception as exc:
        return {"experiment_failed": f"{type(exc).__name__}: {exc}",
                "not_a_specimen_property": True}
    arms = doc.get("arms") or {}
    if not isinstance(arms, Mapping) or not arms:
        return {"refused": f"{Path(path).name} has no `arms` object to compare"}
    # A design may deliberately include an arm that is NOT depth-matched -- a
    # uniform reference cannot be, since uniform means one spec everywhere.
    # Let the caller scope the audit to the arms actually being compared rather
    # than have the whole receipt reported confounded because of a reference arm.
    only = args.get("arms")
    if only:
        want = [str(x) for x in only]
        unknown = [x for x in want if x not in arms]
        if unknown:
            return {"refused": f"{Path(path).name} has no arms {unknown}",
                    "available": sorted(arms)}
        arms = {k: v for k, v in arms.items() if k in want}
    rows = {}
    for name, arm in arms.items():
        if not isinstance(arm, Mapping):
            continue
        per = arm.get("per_organ") or {}
        bits = sorted(int(v.get("bits")) for v in per.values()
                      if isinstance(v, Mapping) and v.get("bits") is not None)
        rows[name] = {
            "organ_bytes": arm.get("organ_bytes"),
            "complete_ebpw": arm.get("complete_ebpw"),
            "bit_depths": bits,
            "min_bits": min(bits) if bits else None,
        }
    # An experiment can declare its own comparison GROUPS. A mirror-swap
    # receipt is not one big comparison: only the two arms sharing a `pair` are
    # meant to be compared, and arms from different pairs legitimately differ in
    # total bytes because their organs differ in size. Comparing everything to
    # everything would report a correctly-designed experiment as confounded, and
    # a checker that cries wolf on a sound design gets ignored.
    groups: Dict[str, List[str]] = {}
    for name, arm in arms.items():
        if isinstance(arm, Mapping) and arm.get("pair"):
            groups.setdefault("|".join(str(x) for x in arm["pair"]), []).append(name)
    grouped = {g: names for g, names in groups.items() if len(names) > 1}

    def _uniq(key, names=None):
        pick = rows if names is None else {k: rows[k] for k in names if k in rows}
        return {json.dumps(r[key], sort_keys=True) for r in pick.values()}

    if grouped:
        per_group = {}
        for g, names in sorted(grouped.items()):
            per_group[g] = {
                "arms": sorted(names),
                "bytes_matched": len(_uniq("organ_bytes", names)) <= 1,
                "ebpw_matched": len(_uniq("complete_ebpw", names)) <= 1,
                "depth_multiset_matched": len(_uniq("bit_depths", names)) <= 1,
            }
        bad = sorted(g for g, v in per_group.items() if not all(
            (v["bytes_matched"], v["ebpw_matched"], v["depth_multiset_matched"])))
        return {
            "confounded": bool(bad),
            "compared_within_groups": True,
            "confounded_groups": bad,
            "safe_to_claim": (
                "each declared group is byte-exact, EBPW-exact and depth-exact, so a "
                "difference inside a group is attributable to the assignment alone"
                if not bad else
                "these groups are not internally matched; equalise them first"),
            "n_groups": len(per_group),
            "groups": per_group,
            "note": ("arms carry a `pair`, so only same-pair arms were compared; "
                     "across groups the organs differ in size and unequal bytes are "
                     "expected, not a defect"),
            "receipt": str(raw),
        }

    bytes_matched = len(_uniq("organ_bytes")) <= 1
    ebpw_matched = len(_uniq("complete_ebpw")) <= 1
    depth_matched = len(_uniq("bit_depths")) <= 1
    confounds = []
    if not bytes_matched:
        confounds.append("total organ bytes differ between arms")
    if not ebpw_matched:
        confounds.append("complete EBPW differs between arms")
    if not depth_matched:
        confounds.append(
            "bit-depth multiset differs between arms: an arm's result mixes its "
            "ORDERING with how deep its ramp went")
    return {
        "confounded": bool(confounds),
        "confounds": confounds,
        "safe_to_claim": ("a difference between these arms is attributable to the "
                          "assignment alone" if not confounds else
                          "NOT an ordering claim; equalise the listed dimensions first"),
        "bytes_matched": bytes_matched,
        "ebpw_matched": ebpw_matched,
        "depth_multiset_matched": depth_matched,
        "arms": rows,
        "receipt": str(raw),
    }


def _tool_reachable(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Can this tool actually be CALLED, or does an argument have no source?

    A required argument with no reachable producer is a door to nowhere, and the
    campaign has burned whole autonomous rounds discovering one at the critical
    path. For each required argument this reports whether another registered
    tool plausibly produces it, so the gap is found before it costs a round.
    """
    name = str(args.get("name") or "").strip()
    reg = _REACHABILITY_REGISTRY.get("registry")
    if reg is None:
        return {"refused": "no registry in scope"}
    spec = reg.get(name)
    if spec is None:
        return {"refused": f"{name!r} is not registered",
                "hint": "tools.catalog lists what is"}
    required = list((spec.input_schema or {}).get("required") or [])
    others = [s for n, s in _REACHABILITY_REGISTRY["specs"].items() if n != name]
    sources: Dict[str, Any] = {}
    for arg in required:
        producers = []
        for other in others:
            blob = (other.description or "").lower() + " " + json.dumps(
                other.output_schema or {}).lower()
            if arg.lower() in blob or arg.replace("_", " ") in blob:
                producers.append(other.name)
        sources[arg] = producers[:4]
    orphans = [a for a, p in sources.items() if not p]
    return {
        "name": name,
        "callable": not orphans,
        "arguments_without_a_source": orphans,
        "required": required,
        "likely_producers": sources,
        "mutation": spec.mutation,
        "note": ("every required argument has a plausible producer" if not orphans
                 else "these arguments have no producing tool; either they are "
                      "literal-derivable or this door cannot be opened"),
    }


def _claim_attack(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """What would make this claim FALSE? The cheapest refutation, not a defence.

    Every recorded law already carries its own counterexample_requirement. This
    surfaces it for one claim and, where the claim names a receipt, the
    comparability fields whose absence would sink it.
    """
    focus = str(args.get("claim") or "").strip().lower()
    if not focus:
        return {"refused": "claim text or law id is required"}
    priors = _odyssey_priors(context, {"focus": focus, "show": 6})
    rows = priors.get("priors") or []
    attacks = [
        {"id": r["id"], "domain": r["domain"],
         "falsifier": r["reopen"] or "NONE RECORDED -- a law with no counterexample "
                                     "requirement cannot be attacked and should not "
                                     "be trusted as permanent"}
        for r in rows
    ]
    generic = [
        "is the evidence about the SPECIMEN, or about a tool invocation?",
        "were the compared arms equal in everything except the named variable?",
        "was the verdict written under the gate that is live now?",
        "is a missing measurement being read as a zero cost?",
        "does the claim's domain cover the body it is being applied to?",
    ]
    return {
        "claim": focus,
        "recorded_falsifiers": attacks,
        "n_recorded": len(attacks),
        "generic_attacks": generic,
        "note": "attack before adopting; a prior with no falsifier is not a prior",
    }


def _wall_avoided(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Record science NOT run because a prior already answered it.

    Compounding is work avoided, and the campaign's earlier attempt to show it
    (rho = -0.050, p = 0.78) measured the wrong thing. This is the right thing:
    an append-only log of experiments a prior removed.
    """
    prior = str(args.get("prior") or "").strip()
    skipped = str(args.get("skipped") or "").strip()
    if not prior or not skipped:
        return {"refused": "both `prior` and `skipped` are required; an unattributed "
                           "skip is not evidence of compounding"}
    entry = {
        "prior": prior,
        "skipped": skipped,
        "saved_wall_estimate_s": args.get("saved_wall_estimate_s"),
        "confirmation": args.get("confirmation"),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "estimate_not_measurement": True,
    }
    dest = context.repo_root / "receipts" / "future" / "WALL_AVOIDED.jsonl"
    if WORKSPACE_WRITE not in getattr(context, "permissions", frozenset()):
        return {"refused": "recording a skip needs workspace_write permission",
                "entry": entry}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return {"recorded": True, "entry": entry,
            "log": str(dest.relative_to(context.repo_root))}


_REACHABILITY_REGISTRY: Dict[str, Any] = {}


def _future_tools_on_path(context: ToolContext) -> Path:
    future = context.repo_root / "tools" / "future"
    if str(future) not in sys.path:
        sys.path.insert(0, str(future))
    if str(context.repo_root) not in sys.path:
        sys.path.insert(0, str(context.repo_root))
    return future


def _round_receipt_dir(context: ToolContext) -> Path:
    return context.repo_root / ".hcli" / "receipts"


def _physical_rounds(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Round receipts the canonical physical emitter can turn into a measurement.

    A round with no model_calls had nothing physical happen in it; the emitter
    refuses those, so they are reported as not emittable rather than hidden.
    """
    shown = _shown_limit(args.get("show"))
    root = _round_receipt_dir(context)
    if not root.is_dir():
        return {
            "n": 0, "total": 0, "shown": 0, "rounds": [],
            "refused": f"no round receipts at {root}; nothing physical to emit",
        }
    paths = sorted(root.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True)
    rows: List[Dict[str, Any]] = []
    for q in paths[: max(shown * 4, 40)]:
        try:
            d = json.loads(q.read_text())
        except Exception as exc:
            rows.append({"round_id": q.stem, "emittable": False,
                         "why": f"unreadable: {type(exc).__name__}"})
            continue
        calls = d.get("model_calls") or []
        prov = (d.get("runtime_provenance") or [{}])[0]
        rows.append({
            "round_id": q.stem,
            "emittable": bool(calls),
            "n_model_calls": len(calls),
            "goal_id": d.get("goal_id"),
            "resident": ((prov.get("identity") or {}).get("resident_identity")
                         or (prov.get("identity") or {}).get("model")),
            "mtime": int(q.stat().st_mtime),
            "why": None if calls else "no model_calls -- nothing physical happened",
        })
    emittable = [r for r in rows if r.get("emittable")]
    out = {
        "rounds": emittable[:shown],
        "n": len(emittable[:shown]),
        "total": len(paths),
        "shown": len(emittable[:shown]),
        "n_emittable_scanned": len(emittable),
        "n_scanned": len(rows),
        "next": "physical.emit with one of these round_id values",
    }
    out.update(_truncation_fields(len(emittable[:shown]), len(paths)))
    return _scope_first(out)


def _physical_emit(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """THE canonical physical measurement path. tools/future/physical_emitter.py.

    Not a benchmark and not an estimate: it reads the per-call prefill/decode
    nanoseconds a real round already recorded against a real resident, and
    attaches the six comparability fields (specimen, nr, runtime, context,
    path, concurrency) whose absence made 78 of 79 historical physical
    receipts incomparable. Writing a second emitter is what this forbids.
    """
    _future_tools_on_path(context)
    raw = args.get("round_id") or args.get("round")
    if not raw:
        return {"refused": "round_id is required; physical.rounds enumerates them"}
    name = str(raw).strip()
    if "/" in name or name.startswith("."):
        return {"refused": f"round_id must be a bare receipt id, got {name!r}"}
    path = _round_receipt_dir(context) / (name if name.endswith(".json") else name + ".json")
    if not path.is_file():
        return {"refused": f"no round receipt {name}; physical.rounds lists what exists"}
    try:
        import physical_emitter  # type: ignore
        measured = physical_emitter.emit(path)
    except Exception as exc:
        # A tool failure is not a specimen refusal. [S010 A20]
        return {"experiment_failed": f"{type(exc).__name__}: {exc}",
                "not_a_specimen_property": True, "round_id": name}
    lead = {k: measured.get(k) for k in (
        "specimen", "nr", "runtime", "context", "path", "concurrency",
        "prefill_tps", "decode_tps", "gpu_share_of_prefill_wall_mean",
        "evidence_tier", "gpu_authority")}
    written = None
    if args.get("write_receipt"):
        # Reading a measurement is read-only; only PERSISTING it is a write. The
        # tool is declared read_only so a reader is not forced to hold write
        # permission to see prefill/decode tok-s, and the write is refused here
        # instead -- the permission check belongs to the byte that lands on disk.
        if WORKSPACE_WRITE not in getattr(context, "permissions", frozenset()):
            lead_refusal = "write_receipt needs workspace_write permission; " \
                           "returning the measurement without persisting it"
        else:
            lead_refusal = None
        out_dir = context.repo_root / "receipts" / "future"
        if lead_refusal is None:
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"PHYSICAL_{name}.json"
            dest.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n")
            written = str(dest.relative_to(context.repo_root))
        else:
            lead["receipt_refused"] = lead_refusal
    lead["receipt"] = written
    lead["round_id"] = name
    lead["totals"] = measured.get("totals")
    lead["n_model_calls"] = len(measured.get("per_call") or [])
    lead["claim_boundary"] = measured.get("claim_boundary")
    return lead


def _physical_measure(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Measure a body that has never been a resident, under the SAME contract.

    physical.emit reads a round receipt, which only ever exists for the
    resident. That is why 28 ModelLake bodies still owe gpu/cpu/tps. This runs
    a controlled fixed-length sweep -- fresh prefill from an empty cache every
    repeat, greedy decode, concurrency 1 -- and fills the same six contract
    fields. Timings are reported as min/median/max with the spread, never as a
    bare median.

    COSTLY: it loads and runs a model. Do not launch it beside another timing
    measurement; shared CPU invalidates both.
    """
    _future_tools_on_path(context)
    snapshot = str(args.get("snapshot") or "").strip()
    specimen = str(args.get("specimen") or "").strip()
    if not snapshot or not specimen:
        return {"refused": "snapshot and specimen are both required; a physical "
                           "receipt that cannot say WHAT it measured is not comparable"}
    root = Path(snapshot)
    if not root.is_dir():
        return {"refused": f"{snapshot} is not a directory on this host"}
    try:
        import physical_emitter  # type: ignore
        measured = physical_emitter.emit_direct(
            snapshot, specimen=specimen,
            nr=str(args.get("nr") or "source body as stored on disk, unmodified"),
            prompt_tokens=int(args.get("prompt_tokens") or 512),
            decode_tokens=int(args.get("decode_tokens") or 64),
            repeats=int(args.get("repeats") or 3),
            backend=str(args.get("backend") or "torch"))
    except Exception as exc:
        return {"experiment_failed": f"{type(exc).__name__}: {exc}",
                "not_a_specimen_property": True, "specimen": specimen}
    lead = {k: measured.get(k) for k in (
        "specimen", "nr", "runtime", "context", "path", "concurrency",
        "prefill_tps", "decode_tps", "prefill_tps_spread", "decode_tps_spread",
        "device", "evidence_tier", "gpu_authority")}
    written = None
    if args.get("write_receipt"):
        if WORKSPACE_WRITE not in getattr(context, "permissions", frozenset()):
            lead["receipt_refused"] = "write_receipt needs workspace_write permission"
        else:
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", specimen)[:80]
            # The BACKEND belongs in the filename. Without it the Metal receipt
            # silently overwrote the CPU one for the same body -- two different
            # machines' numbers competing for one path, and the first measurement
            # simply vanished.
            back = re.sub(r"[^A-Za-z0-9_.-]", "_", str(args.get("backend") or "torch"))
            dest = (context.repo_root / "receipts" / "future"
                    / f"PHYSICAL_DIRECT_{safe}__{back}.json")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n")
            written = str(dest.relative_to(context.repo_root))
    lead["receipt"] = written
    lead["claim_boundary"] = measured.get("claim_boundary")
    return lead


def _nr_complete_ebpw(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Bill a representation through the ONE accounting authority.

    tools/future/complete_ebpw.py counts every persistent part -- payload,
    codebooks, generators, bases, coefficients, indices, metadata. It REFUSES
    an unreconciled candidate rather than billing a flattering subtotal, and
    it flags a candidate that stores small but rematerializes the dense parent
    to execute. Passing a partial candidate here gets a refusal, which is the
    correct answer, not a smaller number.
    """
    _future_tools_on_path(context)
    try:
        from tools.future import complete_ebpw as ce  # type: ignore
    except Exception as exc:
        return {"experiment_failed": f"cannot import complete_ebpw: {exc}",
                "not_a_specimen_property": True}
    cand = args.get("candidate")
    if not cand:
        if not args.get("incumbent"):
            return {"refused": "pass a declared candidate, or incumbent=true to "
                               "bill the sealed resident from MIX_REPORT"}
        try:
            cand = ce.incumbent_candidate()
        except Exception as exc:
            return {"experiment_failed": f"{type(exc).__name__}: {exc}",
                    "not_a_specimen_property": True}
    try:
        billed = ce.cost(cand)
    except ce.CompleteEbpwRefused as exc:
        return {"refused": str(exc), "billed_by": ce.RECORDED_BY,
                "mechanism": "declared parts did not reconcile, or a part-like "
                             "key was undeclared; a guess is not a bill"}
    except Exception as exc:
        return {"experiment_failed": f"{type(exc).__name__}: {exc}",
                "not_a_specimen_property": True}
    lead = {
        "id": billed.get("id"),
        "complete_ebpw": billed.get("complete_ebpw"),
        "stated_total_bytes": billed.get("stated_total_bytes"),
        "parent_params": billed.get("parent_params"),
        "billed_by": ce.RECORDED_BY,
        "evidence_class": ce.EVIDENCE_CLASS,
        "claim_boundary": ce.CLAIM_BOUNDARY,
    }
    for k in ("dense_parent_rematerialization", "flag", "flags", "parts",
              "ms_total", "by_category"):
        if k in billed:
            lead[k] = billed[k]
    return lead



# ---------------------------------------------------------------------------
# SURFACE CONSOLIDATION. [S004]
#
# 60 tools became 81 became 91. Tool COUNT was already a declared non-goal, and
# a resident that must choose among 83 schemas every turn is paying a selection
# tax for a capability it could reach through a dozen doors. So: merge the
# families behind one dispatching tool each, keep EVERY old name callable as an
# alias, and let discover() show the smaller surface.
#
# The one hard rule is permission. A tool has ONE mutation class, so folding a
# destructive op into a read_only tool would silently widen what a "read" can
# do. _merge_group therefore REFUSES to merge members that disagree on
# mutation class, at registry-build time, where it cannot be missed. That is
# why git keeps land.propose and checkout-safe outside git, and why odyssey
# splits into read / record / drive rather than one god-tool.
# ---------------------------------------------------------------------------

CONSOLIDATION: Dict[str, Dict[str, Any]] = {
    "odyssey.read": {
        "summary": "Read Odyssey campaign state. One door, many views.",
        "ops": {
            "status": "odyssey.status", "ledger": "odyssey.ledger",
            "queue": "odyssey.queue", "priors": "odyssey.priors",
            "value": "odyssey.value", "economics": "odyssey.economics",
            "harvest": "odyssey.harvest", "ingest": "odyssey.ingest",
            "completions": "odyssey.completions",
            "selection_brief": "odyssey.selection_brief",
            "patient": "odyssey.patient", "anatomy": "odyssey.anatomy",
            "dense_anatomy": "odyssey.dense_anatomy",
            "attack_law": "odyssey.attack_law",
        },
    },
    "odyssey.record": {
        "summary": "Write one Odyssey result: a Law, a Scar, or a ledger axis.",
        "ops": {"law": "odyssey.record_law", "scar": "odyssey.record_scar"},
    },
    "odyssey.drive": {
        "summary": "Advance the Odyssey driver. Every op is COSTLY and needs confirm.",
        "ops": {
            "cycle": "odyssey.cycle", "retire": "odyssey.retire",
            "park_specimen": "odyssey.park_specimen",
            "add_to_eligibility": "odyssey.add_to_eligibility",
            "write_packet": "odyssey.write_packet",
            "gravity_gauntlet": "odyssey.gravity_gauntlet",
            "transfer_probe": "odyssey.create_transfer_probe",
            "adversarial_probe": "odyssey.create_adversarial_probe",
        },
    },
    "audit": {
        "summary": ("Adversarial audit. Is this verdict stale, is this experiment "
                    "confounded, is this tool actually callable, what would falsify "
                    "this claim?"),
        "ops": {"gate": "capability.gate", "confound": "experiment.confound",
                "reachable": "tool.reachable", "attack": "claim.attack"},
    },
    "fs": {
        "summary": "Read the filesystem: one file, a listing, or a search.",
        "ops": {"read": "fs.read", "list": "fs.list", "search": "fs.search"},
    },
    "git": {
        "summary": ("Inspect the repository without mutating it. Landing and "
                    "checkout stay OUTSIDE this tool because they are not reads."),
        "ops": {"status": "git.status", "log": "git.log", "diff": "git.diff"},
    },
    "processes": {
        "summary": "Live Hawking processes: a listing, a roll-up, or the orphans.",
        "ops": {"list": "processes.list", "summary": "processes.summary",
                "orphaned": "processes.orphaned"},
    },
    "physical": {
        "summary": ("Physical measurement under the ONE contract. Measuring a body "
                    "that was never a resident is physical.measure -- COSTLY, so it "
                    "stays outside."),
        "ops": {"rounds": "physical.rounds", "emit": "physical.emit"},
    },
    "lake": {
        "summary": "The ModelLake: census, mount status, specimen registry, acquisition.",
        "ops": {"census": "lake.census", "status": "modellake.status",
                "specimens": "specimens.registry", "acquire": "acquisition.propose"},
    },
    "receipt": {
        "summary": "Read a named receipt: campaign evidence, roadmap, or architecture.",
        "ops": {"read": "receipt.read", "roadmap": "roadmap.read",
                "architecture": "architecture.inspect"},
    },
    "vmcp": {
        "summary": ("Vision MCP. `tools` says what is callable and what is refused, "
                    "`query` calls one. VisionMCP ships 303 tools; the bridge "
                    "allowlists a read-only subset and names the mechanism for the rest."),
        "ops": {"tools": "vmcp.tools", "capabilities": "vmcp.capabilities",
                "inspect": "vmcp.inspect", "query": "vmcp.query"},
    },
    "web": {
        "summary": "The open web: search for pages, or fetch one.",
        "ops": {"search": "web.search", "fetch": "web.fetch"},
    },
    "github": {
        "summary": "GitHub: search, or fetch a specific object.",
        "ops": {"search": "github.search", "fetch": "github.fetch"},
    },
    "huggingface": {
        "summary": ("Hugging Face metadata. Downloading is COSTLY and stays outside "
                    "as huggingface.download."),
        "ops": {"resolve": "huggingface.resolve", "history": "huggingface.history",
                "fetch_file": "huggingface.fetch_file"},
    },
}


def _merge_group(registry: "ToolRegistry", name: str, spec: Mapping[str, Any]) -> Optional[ToolSpec]:
    """Register one dispatching tool over an existing family, or refuse.

    Refuses -- loudly, at build time -- if the members disagree on mutation
    class. A merged tool has one class; a quiet merge across classes would let
    a read-permissioned caller reach a write.
    """
    members = {op: registry.get(target) for op, target in spec["ops"].items()}
    missing = sorted(op for op, sp in members.items() if sp is None)
    if missing:
        raise RuntimeError(f"consolidation {name}: no such tool for ops {missing}")
    classes = {sp.mutation for sp in members.values()}
    if len(classes) != 1:
        raise RuntimeError(
            f"consolidation {name}: members span mutation classes {sorted(classes)}; "
            "merging them would widen what the weakest permission can reach")
    mutation = classes.pop()
    resources = tuple(sorted({r for sp in members.values() for r in sp.resources}))
    deterministic = all(sp.deterministic for sp in members.values())
    timeout = max(sp.timeout_s for sp in members.values())

    def handler(context: ToolContext, args: Dict[str, Any]) -> Any:
        op = str(args.get("op") or "").strip()
        target = members.get(op)
        if target is None:
            return {"refused": f"unknown op {op!r} for {name}",
                    "ops": sorted(members), "hint": "pass one of ops as `op`"}
        sub = {k: v for k, v in args.items() if k != "op"}
        # Keep the ORIGINAL tool's own validation. The merged schema has to be
        # permissive because ops take different arguments; the per-op contract
        # must not be lost with it.
        problem = validate_input(sub, target.input_schema)
        if problem is not None:
            return {"refused": problem, "op": op, "resolves_to": target.name,
                    "required": (target.input_schema or {}).get("required") or []}
        return target.handler(context, sub)

    lines = []
    for op in sorted(members):
        sp = members[op]
        req = (sp.input_schema or {}).get("required") or []
        lines.append(f"{op}" + (f"({','.join(req)})" if req else "") + f" -> {sp.name}")
    description = spec["summary"] + " ops: " + "; ".join(lines)
    # The merged schema carries the UNION of its members' properties, not just
    # `op`. The catalog renders a signature from the SCHEMA, so a merged tool
    # advertising only op:string would show the model a door and hide every
    # argument behind it -- the same unreachable-argument defect this campaign
    # keeps finding, reintroduced by the consolidation itself. Each property
    # says which ops accept it, so the union does not read as a free-for-all.
    props: Dict[str, Any] = {
        "op": {"type": "string", "enum": sorted(members),
               "description": "which operation; see per-op required args below"}}
    used_by: Dict[str, List[str]] = {}
    for op, sp in sorted(members.items()):
        for field, schema in ((sp.input_schema or {}).get("properties") or {}).items():
            used_by.setdefault(field, []).append(op)
            if field not in props:
                props[field] = dict(schema)
    for field, ops in used_by.items():
        req_for = sorted(o for o in ops
                         if field in ((members[o].input_schema or {}).get("required") or []))
        note = "ops: " + ",".join(sorted(ops))
        if req_for:
            note += "; REQUIRED for " + ",".join(req_for)
        existing = props[field].get("description")
        props[field]["description"] = f"{existing} ({note})" if existing else note
    merged = registry.register(ToolSpec(
        name, description,
        {"type": "object", "required": ["op"], "additionalProperties": True,
         "properties": props},
        mutation=mutation, deterministic=deterministic, resources=resources,
        timeout_s=timeout, handler=handler,
    ))
    # Absorbed names stay CALLABLE -- alias_of only hides them from discover().
    # Nothing that worked before this consolidation stops working. ToolSpec is
    # frozen, so the marker goes on a replacement carrying the same handler
    # rather than by mutating a spec other code may already hold.
    from dataclasses import replace as _replace
    for target in members.values():
        if target.name != name:
            registry._tools[target.name] = _replace(target, alias_of=name)
    return merged


def _consolidate(registry: "ToolRegistry") -> Dict[str, Any]:
    before = len(registry.discover())
    for name, spec in CONSOLIDATION.items():
        _merge_group(registry, name, spec)
    return {"before": before, "after": len(registry.discover())}


def default_tool_registry(
    workspace: str | os.PathLike[str],
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    mission_root: Optional[str | os.PathLike[str]] = None,
    permissions: Optional[Iterable[str]] = None,
) -> ToolRegistry:
    """Build the standard read/research registry for one mission."""
    ws = Path(workspace).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve() if repo_root is not None else ws
    context = ToolContext(
        ws,
        repo,
        Path(mission_root).expanduser().resolve() if mission_root is not None else None,
        frozenset(
            permissions
            if permissions is not None
            else {READ_ONLY, RESEARCH, REVERSIBLE_REPO, REVERSIBLE_RUNTIME}
        ),
    )
    registry = ToolRegistry(context)
    registry.register(ToolSpec(
        "tools.catalog",
        "Find exact typed tool signatures for a focused question; read-only and bounded.",
        {
            "type": "object",
            "required": ["focus"],
            "additionalProperties": False,
            "properties": {
                "focus": {"type": "string"},
                "max_results": {"type": "integer"},
            },
        },
        resources=("filesystem",),
        handler=lambda _context, args: registry.describe(
            args.get("focus"), max_results=args.get("max_results", 12)
        ),
    ))
    registry.register(ToolSpec(
        "lake.census",
        "Reachability and complete EBPW for every ModelLake specimen, from safetensors headers only; reads no payload bytes.",
        {"type": "object", "additionalProperties": False,
         "properties": {"catalog": {"type": "string"}, "slug": {"type": "string"},
                        "limit": {"type": "integer"}}},
        resources=("filesystem",), timeout_s=300.0,
        handler=_lake_census,
    ))
    registry.register(ToolSpec(
        "odyssey.ledger",
        "Per-specimen Odyssey axis state: what is measured, what is refused with a reason, and what is still owed.",
        {"type": "object", "additionalProperties": False,
         "properties": {"path": {"type": "string"}, "slug": {"type": "string"},
                        "owed_only": {"type": "boolean"},
                        "limit": {"type": "integer"}}},
        resources=("filesystem",),
        handler=_odyssey_ledger,
    ))
    registry.register(ToolSpec(
        "odyssey.anatomy",
        "Representational anatomy of one specimen's expert organ; refuses with a named mechanism rather than returning an empty result.",
        {"type": "object", "required": ["snapshot"], "additionalProperties": False,
         "properties": {"snapshot": {"type": "string"},
                        "layer": {"type": ["integer", "null"]}}},
        resources=("filesystem",), timeout_s=1800.0, deterministic=False,
        handler=_odyssey_anatomy,
    ))
    registry.register(ToolSpec(
        "odyssey.dense_anatomy",
        "Within-tensor organ anatomy of one DENSE specimen against a same-shape null. Use odyssey.anatomy instead for a MoE body; each refuses toward the other by name.",
        {"type": "object", "required": ["snapshot"], "additionalProperties": False,
         "properties": {"snapshot": {"type": "string"}}},
        resources=("filesystem",), timeout_s=1800.0, deterministic=False,
        handler=_odyssey_dense_anatomy,
    ))
    registry.register(ToolSpec(
        "odyssey.record_measurement",
        "Record one specimen/axis result into the Odyssey ledger: a value WITH a receipt, or a "
        "refusal whose reason names a mechanism. This is how a measured round closes. With a "
        "refusal, also give reopen_when -- the condition that would make this measurable (a float "
        "copy of the body, a runtime that supports the architecture, a machine with the memory). "
        "121 of the ledger's 142 refusals name a mechanism and no way back, which makes them "
        "permanent by accident.",
        {"type": "object", "required": ["path", "slug", "axis"],
         "additionalProperties": False,
         "properties": {"path": {"type": "string"},
                        "slug": {"type": "string"},
                        "axis": {"type": "string"},
                        "value": {},
                        "receipt": {"type": ["string", "null"]},
                        "reason": {"type": ["string", "null"]},
                        "reopen_when": {"type": ["string", "null"]}}},
        mutation=REVERSIBLE_REPO,
        resources=("filesystem",), deterministic=False,
        handler=_odyssey_record_measurement,
    ))
    registry.register(ToolSpec(
        "campaign.guard",
        "Host memory and swap headroom before an expensive run; the swapfile count is the live signal, not vm.swapusage.",
        {"type": "object", "additionalProperties": False,
         "properties": {"expected_gb": {"type": "number"}}},
        resources=("processes",), deterministic=False,
        handler=_campaign_guard,
    ))
    registry.register(ToolSpec(
        "context.recall",
        "Recall bounded prior-knowledge facts from the hot index and cold gzip archive; never replays the transcript.",
        {
            "type": "object",
            "required": ["focus"],
            "additionalProperties": False,
            "properties": {
                "focus": {"type": "string"},
                "max_results": {"type": "integer"},
                "max_chars": {"type": "integer"},
            },
        },
        resources=("filesystem", "ssd"),
        handler=_context_recall,
    ))
    path_schema = {
        "type": "object",
        "required": ["path"],
        "additionalProperties": False,
        "properties": {
            "path": {"type": "string"},
            "max_bytes": {"type": "integer"},
            "encoding": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
    }
    registry.register(ToolSpec(
        "fs.read",
        "Read one known file under an AgentOS read root. Pass start_line and "
        "end_line (1-indexed, inclusive) to read a window; fs.search reports "
        "the line a match is on, so search then read that region.",
        path_schema,
        handler=_read_file,
    ))
    registry.register(ToolSpec("filesystem.read", "Read one known file under an AgentOS read root.", path_schema, alias_of="fs.read", handler=_read_file))
    registry.register(ToolSpec(
        "fs.search", "Search bounded text files under a read root.",
        {"type": "object", "required": ["pattern"], "additionalProperties": False,
         "properties": {"pattern": {"type": "string"}, "root": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}, "max_results": {"type": "integer"}}},
        handler=_search_files,
    ))
    registry.register(ToolSpec(
        "filesystem.search", "Search bounded text files under a read root.",
        {"type": "object", "required": ["pattern"], "additionalProperties": False,
         "properties": {"pattern": {"type": "string"}, "root": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}, "max_results": {"type": "integer"}}},
        alias_of="fs.search", handler=_search_files,
    ))
    # `path` is NOT required: the handler already defaults to the workspace root
    # and the read-root check still applies, so demanding it bought no safety and
    # cost real calls. Measured: the model called fs.list with no arguments,
    # got "missing required property 'path'", and burned the round. "List the
    # repo" is the common intent and must be expressible.
    list_schema = {
        "type": "object", "required": [], "additionalProperties": False,
        "properties": {
            "path": {"type": "string"}, "glob": {"type": "string"},
            "recursive": {"type": "boolean"}, "max_results": {"type": "integer"},
        },
    }
    registry.register(ToolSpec(
        "fs.list", "List files and directory entries under a read root, optionally filtered by glob.",
        list_schema, handler=_list_files,
    ))
    registry.register(ToolSpec(
        "filesystem.list", "List files and directory entries under a read root, optionally filtered by glob.",
        list_schema, alias_of="fs.list", handler=_list_files,
    ))
    registry.register(ToolSpec(
        "filesystem.write", "Atomically write a workspace/repository file under reversible permission.",
        {"type": "object", "required": ["path", "content"], "additionalProperties": False,
         "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "overwrite": {"type": "boolean"}, "encoding": {"type": "string"}}},
        mutation=REVERSIBLE_REPO,
        resources=("filesystem", "ssd"),
        verifier_expectations=("caller must verify resulting bytes",),
        handler=_filesystem_write,
    ))
    registry.register(ToolSpec(
        "shell.readonly", "Run one allowlisted read-only command without a shell interpreter.",
        {"type": "object", "required": ["command"], "additionalProperties": False, "properties": {"command": {"type": "string"}}},
        handler=_shell_readonly,
    ))
    registry.register(ToolSpec(
        "shell.exec", "Run one typed, allowlisted reversible command without a shell interpreter.",
        {"type": "object", "required": ["argv"], "additionalProperties": False,
         "properties": {"argv": {"type": "array", "items": {"type": "string"}}, "cwd": {"type": "string"}, "timeout_s": {"type": "number"}}},
        mutation=REVERSIBLE_RUNTIME,
        resources=("cpu",),
        verifier_expectations=("returncode must be checked by the caller",),
        handler=_shell_exec,
    ))
    # G009 (receipts/sovereign/G009_reachability.json) found process truth
    # REACHABLE FROM PRODUCTION CODE (hcli/runtime.py's startup reaper,
    # hcli/commands.py's /processes command) but UNREACHABLE FROM THE MODEL:
    # no registered tool named a process, and shell.readonly above refuses
    # `ps` outright. The live goal's first law names processes as authority
    # and gave the resident no way to look at one. These three close that gap
    # by wrapping hcli/processes.py's existing read paths only -- nothing new
    # is taught to reap or signal anything. Killing a process is not reachable
    # through this registry at all; see the handler docstrings. Observation is
    # now owned by hide-backend::process_inspector.
    registry.register(ToolSpec(
        "processes.list",
        "Every live Hawking process, classified by argv: role, PID, memory, "
        "elapsed time and whether it is safe to stop. Read-only.",
        {"type": "object", "additionalProperties": False, "properties": {}},
        handler=_processes_list,
    ))
    registry.register(ToolSpec(
        "processes.summary",
        "Roll-up of live Hawking processes: count, total footprint, counts by "
        "class. The same entry point the process audit receipt uses.",
        {"type": "object", "additionalProperties": False, "properties": {}},
        handler=_processes_summary,
    ))
    registry.register(ToolSpec(
        "processes.orphaned",
        "Resident model bodies with no live owner (reparented to pid 1, "
        "unclaimed by any resident state file). Enumeration only -- does not "
        "reap; reaping is a startup-only self-heal in hcli/runtime.py.",
        {"type": "object", "additionalProperties": False, "properties": {}},
        handler=_processes_orphaned,
    ))
    registry.register(ToolSpec(
        "git.status", "Inspect repository status without mutating Git.",
        {"type": "object", "additionalProperties": False, "properties": {"path": {"type": "string"}}},
        handler=_git_status,
    ))
    registry.register(ToolSpec(
        "git.log", "Inspect recent repository commits without mutating Git.",
        {"type": "object", "additionalProperties": False, "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}},
        handler=_git_log,
    ))
    registry.register(ToolSpec(
        "git.diff", "Inspect repository changes without mutating Git.",
        {"type": "object", "additionalProperties": False, "properties": {"path": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}}},
        handler=_git_diff,
    ))
    for name in ("git.checkout-safe", "git.revert-safe", "git.checkout/revert-safe"):
        registry.register(ToolSpec(
            name, "Governed placeholder for safe checkout/revert; refuses without a caller-owned recovery plan.",
            {"type": "object", "additionalProperties": False, "properties": {"path": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}}},
            mutation=DESTRUCTIVE,
            deterministic=True,
            alias_of=None if name == "git.checkout-safe" else "git.checkout-safe",
            handler=_git_safe_revert_refusal,
        ))
    registry.register(ToolSpec(
        "git.land.propose",
        "Propose one landing candidate: a declared branch, an allowlist of "
        "changed paths, a test command, and a message. Admissibility is "
        "decided by a deterministic verifier that re-runs the test command "
        "itself and re-checks the tree; a commit happens only if every named "
        "condition holds. This is the only path from the resident to a git "
        "commit -- it never pushes.",
        {"type": "object", "required": ["branch", "allowed_paths", "test_command", "message"],
         "additionalProperties": False,
         "properties": {
             "branch": {"type": "string"},
             "allowed_paths": {"type": "array", "items": {"type": "string"}},
             "test_command": {"type": "array", "items": {"type": "string"}},
             "message": {"type": "string"},
             "timeout_s": {"type": "number"},
         }},
        mutation=REVERSIBLE_REPO,
        deterministic=False,
        resources=("filesystem", "git"),
        verifier_expectations=(
            "landed is true only when the verifier re-ran test_command itself, on this "
            "tree state, and it exited zero; a proposal cannot assert its own tests passed",
        ),
        handler=_git_land_propose,
    ))
    research_schema = {"type": "object", "required": ["url"], "additionalProperties": False, "properties": {"url": {"type": "string"}, "max_bytes": {"type": "integer"}, "timeout_s": {"type": "number"}}}
    registry.register(ToolSpec("web.fetch", "Fetch bounded public HTTPS evidence with no credentials.", research_schema, mutation=RESEARCH, deterministic=False, handler=lambda c, a: _fetch(c, a)))
    registry.register(ToolSpec("github.fetch", "Fetch bounded public GitHub HTTPS evidence with no credentials.", research_schema, mutation=RESEARCH, deterministic=False, handler=lambda c, a: _fetch(c, a, allowed_hosts=("github.com", "api.github.com", "raw.githubusercontent.com"))))
    registry.register(ToolSpec(
        "web.search", "Search the public web through a bounded, credential-free provider.",
        {"type": "object", "required": ["query"], "additionalProperties": False, "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}, "timeout_s": {"type": "number"}}},
        mutation=RESEARCH, deterministic=False, resources=("network",),
        verifier_expectations=("source URL and retrieval timestamp are required",),
        handler=_web_search,
    ))
    registry.register(ToolSpec(
        "github.search", "Search public GitHub repositories/issues/code/commits with bounded provenance.",
        {"type": "object", "required": ["query"], "additionalProperties": False, "properties": {"query": {"type": "string"}, "kind": {"type": "string", "enum": ["repositories", "issues", "code", "commits"]}, "max_results": {"type": "integer"}, "timeout_s": {"type": "number"}}},
        mutation=RESEARCH, deterministic=False, resources=("network",),
        verifier_expectations=("GitHub source URL and retrieval timestamp are required",),
        handler=_github_search,
    ))
    registry.register(ToolSpec(
        "huggingface.resolve", "Resolve a Hugging Face model revision and bounded file manifest.",
        {"type": "object", "required": ["repo"], "additionalProperties": False, "properties": {"repo": {"type": "string"}, "revision": {"type": "string"}}},
        mutation=RESEARCH, deterministic=False, handler=_huggingface_resolve,
    ))
    registry.register(ToolSpec(
        "huggingface.manifest", "Resolve a Hugging Face revision and return a bounded manifest.",
        {"type": "object", "required": ["repo"], "additionalProperties": False, "properties": {"repo": {"type": "string"}, "revision": {"type": "string"}}},
        mutation=RESEARCH, deterministic=False, resources=("network",), alias_of="huggingface.resolve", handler=_huggingface_resolve,
    ))
    registry.register(ToolSpec(
        "huggingface.fetch_file", "Fetch a bounded text/metadata file from a public or already-authorized Hugging Face repo.",
        {"type": "object", "required": ["repo", "path"], "additionalProperties": False, "properties": {"repo": {"type": "string"}, "path": {"type": "string"}, "revision": {"type": "string"}, "max_bytes": {"type": "integer"}, "timeout_s": {"type": "number"}}},
        mutation=RESEARCH, deterministic=False, resources=("network",), handler=_huggingface_fetch_file,
    ))
    registry.register(ToolSpec(
        "huggingface.history", "Inspect bounded Hugging Face commit history.",
        {"type": "object", "required": ["repo"], "additionalProperties": False, "properties": {"repo": {"type": "string"}, "revision": {"type": "string"}, "timeout_s": {"type": "number"}}},
        mutation=RESEARCH, deterministic=False, resources=("network",), handler=_huggingface_history,
    ))
    registry.register(ToolSpec(
        "huggingface.download", "Resume and atomically publish one explicitly confirmed Hugging Face file.",
        {"type": "object", "required": ["repo", "path", "destination", "confirm"], "additionalProperties": False,
         "properties": {"repo": {"type": "string"}, "path": {"type": "string"}, "revision": {"type": "string"}, "destination": {"type": "string"}, "confirm": {"type": "boolean"}, "max_bytes": {"type": "integer"}, "expected_sha256": {"type": "string"}, "timeout_s": {"type": "number"}}},
        mutation=COSTLY, deterministic=False, resources=("network", "modellake", "ssd"),
        verifier_expectations=("hash must be checked before atomic publish",), handler=_huggingface_download,
    ))
    registry.register(ToolSpec("receipt.read", "Read a JSON/text receipt under the repository or mission state roots.", path_schema, handler=_receipt_read))
    registry.register(ToolSpec("receipt.inspect", "Inspect a JSON/text receipt under the repository or mission state roots.", path_schema, alias_of="receipt.read", handler=_receipt_read))
    for name, description in (
        ("roadmap.read", "Read the persisted civilization roadmap."),
        ("vmcp.capabilities", "Read the latest VMCP capability census."),
        ("doctor.inspect", "Read the latest Doctor tournament receipt."),
        ("gravity.inspect", "Read the latest Gravity compiler/search receipt."),
        ("accelerator.inspect", "Read the latest accelerator machine receipt."),
        ("modellake.status", "Read the ModelLake rolling-pipeline receipt."),
    ):
        registry.register(ToolSpec(
            name, description,
            {"type": "object", "additionalProperties": False, "properties": {"path": {"type": "string"}}},
            handler=_target_receipt(name),
        ))
    # Odyssey and the ANE lab were built as modules and never registered, so
    # nothing a resident drives could reach them: WorkUnit.tool -> _run_tool ->
    # ToolRegistry.invoke is the only path, and a module absent from this
    # registry is absent from that path. A capability nothing calls does not
    # exist, which is the law this whole wave enforces.
    for name, description in (
        ("odyssey.status", "Read live Odyssey state: queue, patient, compiler rules, research."),
        ("odyssey.queue", "Read the Odyssey specimen queue."),
        ("odyssey.value", "Read the Odyssey value/economics ranking."),
        ("odyssey.economics", "Read Odyssey acquisition economics."),
    ):
        registry.register(ToolSpec(
            name, description,
            {"type": "object", "additionalProperties": False, "properties": {}},
            handler=_odyssey_read(name.split(".", 1)[1]),
        ))
    registry.register(ToolSpec(
        "odyssey.selection_brief",
        "The five selection inputs G018 requires -- measurement debt, cost, architecture novelty, "
        "capability state and Pareto state -- assembled per specimen and NOT RANKED. Rows come "
        "back in slug order, which carries no opinion. field_discrimination says how much each "
        "input actually separates these bodies; read_this_first names the inputs that do not. "
        "The choice, and the reason, are yours.",
        {"type": "object", "additionalProperties": False,
         "properties": {"limit": {"type": "integer"}}},
        handler=_selection_brief,
    ))
    registry.register(ToolSpec(
        "odyssey.cycle",
        "Advance the LIVE Odyssey by one cycle. Mutating; requires confirm=True.",
        {"type": "object", "additionalProperties": False,
         "required": ["confirm"],
         "properties": {"confirm": {"type": "boolean"}, "max_lanes": {"type": "integer"}}},
        mutation=COSTLY, deterministic=False, resources=("cpu",),
        handler=_odyssey_cycle,
    ))
    registry.register(ToolSpec(
        "odyssey.gravity_gauntlet",
        "Run one bounded, resumable, evidence-guided Gravity search; requires confirm=True.",
        {"type": "object", "additionalProperties": False,
         "required": ["oxx", "candidate_specs", "confirm"],
         "properties": {
             "oxx": {"type": "string"},
             "candidate_specs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
             "budget": {"type": "integer", "minimum": 1},
             "state_path": {"type": "string"},
             "receipt_dir": {"type": "string"},
             "confirm": {"type": "boolean"},
         }},
        mutation=COSTLY, deterministic=False, resources=("cpu", "ssd"),
        verifier_expectations=(
            "TARGET_HIT requires complete EBPW <= 1 plus independent capability/execution verification",
            "BUDGET_EXHAUSTED is never a pass",
        ),
        handler=_odyssey_gravity_gauntlet,
    ))
    registry.register(ToolSpec(
        "forbidden_fruit.lab",
        "Probe CPU/GPU/ANE, run the compiled fixture, report OBSERVED placement and timing.",
        # `timeout_s` was advertised here and forwarded to a handler that has no
        # such parameter, so EVERY call raised TypeError before the lab ran. The
        # lab bounds itself per step (compile 180s, run 60s, pair 120s); there is
        # no single timeout for one knob to mean.
        {"type": "object", "additionalProperties": False,
         "properties": {"sdk": {"type": "string"}}},
        mutation=REVERSIBLE_RUNTIME, deterministic=False, resources=("cpu",),
        verifier_expectations=(
            "placement is MLComputePlan.deviceUsage as observed, never the requested compute units",
        ),
        handler=_forbidden_fruit_lab,
    ))
    # WIRED HERE ON PURPOSE. Four modules were built in separate lanes, each
    # scoped to its own file, which structurally prevented any of them from
    # registering. Four verifiers then correctly refused them all on the same
    # ground: a capability nothing can call does not exist. Registration is
    # cross-cutting, so it belongs in one place rather than fragmented.
    registry.register(ToolSpec(
        "frontier.decide",
        "Which frontier should run next and why the others are waiting or parked.",
        {"type": "object", "additionalProperties": False, "properties": {}},
        handler=_frontier_decide,
    ))
    registry.register(ToolSpec(
        "specimens.registry",
        "Every sealed specimen enumerated from disk. Sealed does not mean loadable.",
        {"type": "object", "additionalProperties": False,
         "properties": {"name": {"type": "string"}, "limit": {"type": "integer"}}},
        handler=_specimens_registry,
    ))
    registry.register(ToolSpec(
        "acquisition.propose",
        "Rank what to acquire next, with destination-filesystem headroom. Never starts a download.",
        {"type": "object", "additionalProperties": False, "properties": {}},
        handler=_acquisition_propose,
    ))
    registry.register(ToolSpec(
        "odyssey.ingest",
        "Read the live mid-flight Odyssey state so HCLI can take it over without restarting it.",
        {"type": "object", "additionalProperties": False, "properties": {}},
        handler=_odyssey_read_verb("ingest"),
    ))
    # The other three read-only connectors. Same reason as the block above: a
    # verb absent from this registry is absent from the resident's only call
    # path, so it did not exist. `completions` is registered without its
    # `rebuild` flag and `harvest` without its apply flag -- both of those
    # write, and both are already what odyssey.cycle does on confirm.
    for verb, required, description in (
        ("completions", (), "List recorded Odyssey completions. Read-only; the --rebuild backfill stays on the CLI."),
        ("patient", ("oxx",), "Read one Odyssey patient packet already on disk. Never writes one."),
        ("harvest", (), "Dry-run harvest: which finished lanes would be applied. Applying is odyssey.cycle."),
    ):
        registry.register(ToolSpec(
            "odyssey." + verb, description,
            {"type": "object", "additionalProperties": False,
             "required": list(required),
             "properties": {key: {"type": "string"} for key in required}},
            deterministic=False,
            handler=_odyssey_read_verb(verb, required),
        ))
    # Targeted per-specimen driver mutations, gated like odyssey.cycle. Neither
    # is reachable through cycle: cycle retires whatever it selects itself and
    # never writes a packet for a named oxx.
    for verb, description in (
        ("retire", "Retire one named Odyssey patient. Writes the driver's own state; requires confirm=True."),
        ("write_packet", "Write the patient packet for one named Odyssey oxx. Requires confirm=True."),
    ):
        registry.register(ToolSpec(
            "odyssey." + verb, description,
            {"type": "object", "additionalProperties": False,
             "required": ["confirm", "oxx"],
             "properties": {"confirm": {"type": "boolean"}, "oxx": {"type": "string"}}},
            mutation=COSTLY, deterministic=False,
            handler=_odyssey_mutating(verb, ("oxx",)),
        ))
    for verb, required in (
        ("add_to_eligibility", ("oxx",)),
        ("park_specimen", ("oxx",)),
        ("record_law", ("text", "domain", "reopen_when")),
        ("record_scar", ("law_id", "reopen_when")),
        ("create_transfer_probe", ("law_id", "target_oxx")),
        ("create_adversarial_probe", ("law_id",)),
    ):
        props = {"confirm": {"type": "boolean"}}
        for field in ("oxx", "text", "law_id", "target_oxx", "note", "reason",
                      "evidence", "source_oxx", "attack", "description",
                      # A law without a domain is applied where nobody tested it;
                      # one without a reopen condition is permanent by accident.
                      "domain", "reopen_when"):
            props[field] = {"type": "string"}
        registry.register(ToolSpec(
            "odyssey." + verb,
            "Odyssey campaign mutation: " + verb.replace("_", " ") + ". Requires confirm=True.",
            {"type": "object", "additionalProperties": False,
             "required": ["confirm"] + list(required), "properties": props},
            mutation=COSTLY, deterministic=False,
            handler=_odyssey_mutating(verb, required),
        ))
    registry.register(ToolSpec(
        "tests.list", "Discover deterministic test files without executing them.",
        {"type": "object", "additionalProperties": False, "properties": {"root": {"type": "string"}}},
        handler=_list_tests,
    ))
    registry.register(ToolSpec(
        "tests.run", "Run a bounded pytest/unittest/cargo verification command under reversible runtime permission.",
        {"type": "object", "additionalProperties": False, "properties": {"runner": {"type": "string", "enum": ["pytest", "unittest", "cargo"]}, "root": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}, "timeout_s": {"type": "number"}}},
        mutation=REVERSIBLE_RUNTIME, deterministic=False, resources=("cpu",),
        verifier_expectations=("verified is true only for exit code zero",), handler=_tests_run,
    ))
    registry.register(ToolSpec(
        "vmcp.inspect", "Inspect the discovered VisionMCP public API and profile/tool surface without starting a worker.",
        {"type": "object", "additionalProperties": False, "properties": {"profile": {"type": "string"}}},
        mutation=READ_ONLY, deterministic=True, resources=("filesystem",), handler=_vmcp_inspect,
    ))
    registry.register(ToolSpec(
        "vmcp.query", "Call one allowlisted read/evidence tool on the discovered VisionMCP core surface.",
        {"type": "object", "required": ["tool"], "additionalProperties": False, "properties": {
            "profile": {"type": "string"},
            "tool": {"type": "string"},
            "arguments": {"type": "object"},
        }},
        mutation=READ_ONLY, deterministic=False, resources=("filesystem",), handler=_vmcp_query,
    ))
    registry.register(ToolSpec(
        "architecture.inspect", "Recognize generic model organs/topology and project the canonical accelerator atlas as planning-only hypotheses.",
        {"type": "object", "required": ["path"], "additionalProperties": False, "properties": {"path": {"type": "string"}, "max_tensors": {"type": "integer"}, "backend": {"type": "string"}}},
        mutation=READ_ONLY, deterministic=True, resources=("filesystem",), handler=_architecture_inspect,
    ))
    registry.register(ToolSpec(
        "doctor.query", "Ask the local Doctor research/evidence/planning surface for a bounded proposal.",
        {"type": "object", "additionalProperties": False, "properties": {"operation": {"type": "string"}, "model": {"type": "string"}, "organ": {"type": "string"}, "receipt": {"type": "string"}, "research_query": {"type": "string"}, "max_results": {"type": "integer"}}},
        mutation=RESEARCH, deterministic=False, resources=("filesystem", "network"),
        verifier_expectations=("Doctor proposals require a later measurement receipt",), handler=_doctor_query,
    ))
    registry.register(ToolSpec(
        "gravity.experiment", "Propose a bounded Gravity representation experiment with negative-control requirements.",
        {"type": "object", "additionalProperties": False, "properties": {"model": {"type": "string"}, "organ": {"type": "string"}, "representation": {"type": "string"}, "receipt": {"type": "string"}, "execute": {"type": "boolean"}}},
        mutation=REVERSIBLE_RUNTIME, deterministic=True, resources=("cpu", "ssd"),
        verifier_expectations=("no capability claim without native execution and protected verifier",), handler=_gravity_experiment,
    ))
    registry.register(ToolSpec(
        "accelerator.benchmark", "Inspect or explicitly authorize a physical accelerator benchmark window.",
        {"type": "object", "required": ["confirm"], "additionalProperties": False, "properties": {"confirm": {"type": "boolean"}, "receipt": {"type": "string"}}},
        mutation=COSTLY, deterministic=False, resources=("gpu", "exclusive_benchmark_window"),
        verifier_expectations=("physical claims require live samples and benchmark state",), handler=_accelerator_benchmark,
    ))
    registry.register(ToolSpec(
        "benchmark.run", "Run one explicitly confirmed bounded benchmark/test command.",
        {"type": "object", "required": ["confirm"], "additionalProperties": False, "properties": {"confirm": {"type": "boolean"}, "runner": {"type": "string", "enum": ["pytest", "unittest", "cargo"]}, "root": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}, "timeout_s": {"type": "number"}}},
        mutation=COSTLY, deterministic=False, resources=("cpu", "exclusive_benchmark_window"),
        verifier_expectations=("verified is true only for exit code zero",), handler=_benchmark_run,
    ))
    registry.register(ToolSpec(
        "roadmap.inspect", "Inspect the persisted civilization roadmap receipt.",
        {"type": "object", "additionalProperties": False, "properties": {"path": {"type": "string"}}},
        alias_of="roadmap.read", handler=_target_receipt("roadmap.read"),
    ))
    registry.register(ToolSpec(
        "benchmark.inspect", "Inspect benchmark evidence through a named receipt path.",
        {"type": "object", "required": ["path"], "additionalProperties": False, "properties": {"path": {"type": "string"}}},
        alias_of="receipt.read", handler=_receipt_read,
    ))
    artifact_item_schema = {
        "type": "object",
        "required": ["name", "content"],
        "additionalProperties": False,
        "properties": {"name": {"type": "string"}, "content": {"type": "string"}},
    }
    registry.register(ToolSpec(
        "frontier.escalate",
        "Escalate one scoped question to a cloud frontier model with a curated packet "
        "(mission kernel + named artifacts + output schema). Returns a proposal, never "
        "a fact; fails closed with no API credential configured.",
        {"type": "object", "required": ["confirm", "question", "mission_kernel"], "additionalProperties": False,
         "properties": {
             "confirm": {"type": "boolean"},
             "question": {"type": "string"},
             "mission_kernel": {"type": "string"},
             "artifacts": {"type": "array", "items": artifact_item_schema},
             "output_schema": {"type": "object"},
             "model": {"type": "string"},
             "timeout_s": {"type": "number"},
         }},
        mutation=COSTLY, deterministic=False, resources=("network", "frontier_api"),
        verifier_expectations=("frontier prose is UNVERIFIED until a local deterministic check accepts it",),
        handler=_frontier_escalate,
    ))
    lane_item_schema = {
        "type": "object",
        "required": ["name"],
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "objective": {"type": "string"},
            "write_scope": {"type": "array", "items": {"type": "string"}},
            "verify_command": {"type": "string"},
            "acceptance": {"type": "array", "items": {"type": "string"}},
            "contract_text": {"type": "string"},
        },
    }
    registry.register(ToolSpec(
        "grok.swarm.propose",
        "Validate and render up to 4 caller-authored WRITE/VERIFY lane contracts from a "
        "problem statement, without launching anything.",
        {"type": "object", "required": ["problem_statement", "lanes"], "additionalProperties": False,
         "properties": {"problem_statement": {"type": "string"}, "lanes": {"type": "array", "items": lane_item_schema}}},
        handler=_grok_swarm_propose,
    ))
    registry.register(ToolSpec(
        "grok.swarm.launch",
        "Launch a bounded (<= 4 lanes) read-only Grok swarm (audit/consult) from "
        "caller-authored lane contracts. Costly; requires confirm=true; fails closed "
        "if grok-run is not installed.",
        {"type": "object", "required": ["confirm", "problem_statement", "lanes"], "additionalProperties": False,
         "properties": {
             "confirm": {"type": "boolean"},
             "problem_statement": {"type": "string"},
             "lanes": {"type": "array", "items": lane_item_schema},
             "mode": {"type": "string", "enum": ["audit", "consult"]},
             "dry_run": {"type": "boolean"},
         }},
        mutation=COSTLY, deterministic=False, resources=("cpu", "network", "grok"),
        verifier_expectations=("each lane's grok status/report must be checked before its output counts as fact",),
        handler=_grok_swarm_launch,
    ))
    # NR->NX doors: the single biller and the single physical emitter, made
    # reachable. [S010 35, 37, 49]
    registry.register(ToolSpec(
        "odyssey.attack_law",
        "OIII: generate ranked executable attacks against one recorded law -- "
        "negative transfer, blind holdout, measurement trap, goodhart, scope. "
        "Closes the loop LAW -> ATTACK -> RESULT -> SCOPE UPDATE from the side "
        "that writes laws. A law with no domain, or one that emits no attack, "
        "is refused rather than published.",
        {"type": "object", "required": ["law_id"], "additionalProperties": False,
         "properties": {"law_id": {"type": "string"}, "show": {"type": "integer"},
                        "source_model": {"type": "string"},
                        "architecture_family": {"type": "string"},
                        "organ_class": {"type": "string"}}},
        handler=_odyssey_attack_law,
    ))
    registry.register(ToolSpec(
        "capability.gate",
        "THE capability authority: does this candidate pass perplexity AND "
        "n-gram diversity against its own dense parent, under the live G020 "
        "bars? Pass recorded_capability_ok to ask whether a receipt's stored "
        "verdict is STALE -- evidence written under a superseded bar is not "
        "evidence.",
        {"type": "object", "additionalProperties": False,
         "required": ["ppl", "r4", "dense_ppl", "dense_r4"],
         "properties": {"ppl": {"type": "number"}, "r4": {"type": "number"},
                        "dense_ppl": {"type": "number"}, "dense_r4": {"type": "number"},
                        "recorded_capability_ok": {"type": "boolean"}}},
        handler=_capability_gate,
    ))
    registry.register(ToolSpec(
        "experiment.confound",
        "Before believing an ORDERING result, check the arms were equal in "
        "everything else: total bytes, complete EBPW and the multiset of bit "
        "depths. If any of those differ, the capability difference belongs to "
        "whichever moved, not to the ordering.",
        {"type": "object", "required": ["receipt"], "additionalProperties": False,
         "properties": {"receipt": {"type": "string"},
                        "arms": {"type": "array", "items": {"type": "string"}}}},
        handler=_experiment_confound,
    ))
    registry.register(ToolSpec(
        "tool.reachable",
        "Can a named tool actually be called, or does one of its required "
        "arguments have no producing tool? Run this BEFORE planning work around "
        "a door, not after a round has burned turns discovering it is shut.",
        {"type": "object", "required": ["name"], "additionalProperties": False,
         "properties": {"name": {"type": "string"}}},
        handler=_tool_reachable,
    ))
    registry.register(ToolSpec(
        "claim.attack",
        "What would make this claim FALSE? Returns each matching law's own "
        "recorded counterexample requirement plus the standing attacks. A prior "
        "with no falsifier is not a prior.",
        {"type": "object", "required": ["claim"], "additionalProperties": False,
         "properties": {"claim": {"type": "string"}}},
        handler=_claim_attack,
    ))
    registry.register(ToolSpec(
        "wall.avoided",
        "Record an experiment NOT run because a prior already answered it. "
        "Compounding is work avoided; this is the log that measures it.",
        {"type": "object", "required": ["prior", "skipped"],
         "additionalProperties": False,
         "properties": {"prior": {"type": "string"}, "skipped": {"type": "string"},
                        "saved_wall_estimate_s": {"type": "number"},
                        "confirmation": {"type": "string"}}},
        mutation=WORKSPACE_WRITE,
        handler=_wall_avoided,
    ))
    registry.register(ToolSpec(
        "campaign.state",
        "The campaign in one read: which depth axes have ZERO measurements (the "
        "bottleneck), axis coverage across every specimen, how many priors are "
        "live, and the most recent evidence. Call this FIRST when deciding what "
        "to do next. It assembles; it does not choose.",
        {"type": "object", "additionalProperties": False, "properties": {}},
        handler=_campaign_state,
    ))
    registry.register(ToolSpec(
        "campaign.checkpoint",
        "Write the part of the campaign a restart cannot rebuild from receipts: "
        "the objective, why this specimen, the live hypothesis, and the explicit "
        "NEXT ACTION. Call it at transitions, not every turn. campaign.state "
        "reads it back first on reattach.",
        {"type": "object", "additionalProperties": False, "properties": {
            "objective": {"type": "string"},
            "active_specimen": {"type": "string"},
            "why_this_specimen": {"type": "string"},
            "active_workunit": {"type": "string"},
            "hypothesis": {"type": "string"},
            "next_action": {"type": "string"},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "open_jobs": {"type": "array", "items": {"type": "string"}},
            "resource_deps": {"type": "array", "items": {"type": "string"}},
            "representations_ruled_out": {"type": "array", "items": {"type": "string"}}},
         "required": ["objective", "hypothesis", "next_action"]},
        mutation=WORKSPACE_WRITE,
        handler=_campaign_checkpoint,
    ))
    registry.register(ToolSpec(
        "vmcp.tools",
        "Which VisionMCP tools HCLI can actually call, and the named mechanism "
        "that refuses the rest. Read this before vmcp op=query.",
        {"type": "object", "additionalProperties": False, "properties": {}},
        handler=_vmcp_tools,
    ))
    registry.register(ToolSpec(
        "odyssey.priors",
        "What the campaign already knows: recorded Laws and Scars, each with the "
        "DOMAIN it was measured on and the condition that would REOPEN it. Read "
        "this before proposing an experiment -- a prior that already answers the "
        "question removes the experiment. odyssey.record_law and "
        "odyssey.record_scar write here; this is how they are read back.",
        {"type": "object", "additionalProperties": False,
         "properties": {"focus": {"type": "string"}, "show": {"type": "integer"}}},
        handler=_odyssey_priors,
    ))
    registry.register(ToolSpec(
        "physical.rounds",
        "Round receipts that the canonical physical emitter can turn into a "
        "measurement, newest first. Use this to get a round_id for "
        "physical.emit; rounds with no model calls are excluded because "
        "nothing physical happened in them.",
        {"type": "object", "additionalProperties": False,
         "properties": {"show": {"type": "integer"}}},
        handler=_physical_rounds,
    ))
    registry.register(ToolSpec(
        "physical.emit",
        "THE canonical physical measurement for one round: fresh prefill "
        "tok/s, decode tok/s, GPU share of prefill wall, plus the six "
        "comparability fields (specimen, nr, runtime, context, path, "
        "concurrency). Real production calls against the live resident, not a "
        "synthetic benchmark and not an estimate. Set write_receipt to "
        "persist it under receipts/future/.",
        {"type": "object", "required": ["round_id"], "additionalProperties": False,
         "properties": {"round_id": {"type": "string"},
                        "write_receipt": {"type": "boolean"}}},
        timeout_s=120.0,
        handler=_physical_emit,
    ))
    registry.register(ToolSpec(
        "physical.measure",
        "Measure a ModelLake body that has never been a resident, under the "
        "SAME physical contract as physical.emit: fresh prefill tok/s and decode "
        "tok/s with min/median/max spread, plus specimen, nr, runtime, context, "
        "path and concurrency. This is how gpu/cpu/tps stop being OWED on bodies "
        "the resident never ran. COSTLY -- never run it beside another timing "
        "measurement.",
        {"type": "object", "required": ["snapshot", "specimen"],
         "additionalProperties": False,
         "properties": {"snapshot": {"type": "string"}, "specimen": {"type": "string"},
                        "nr": {"type": "string"},
                        "prompt_tokens": {"type": "integer"},
                        "decode_tokens": {"type": "integer"},
                        "repeats": {"type": "integer"},
                        "backend": {"type": "string", "enum": ["torch", "mlx"],
                                    "description": "torch = CPU float32 reference; "
                                                   "mlx = METAL at the checkpoint's "
                                                   "native dtype. Different device AND "
                                                   "different precision, so receipts "
                                                   "from the two are not comparable."},
                        "write_receipt": {"type": "boolean"}}},
        mutation=COSTLY, deterministic=False,
        resources=("cpu", "gpu", "exclusive_benchmark_window"),
        timeout_s=1800.0,
        handler=_physical_measure,
    ))
    registry.register(ToolSpec(
        "nr.complete_ebpw",
        "Complete executable bits-per-weight of a representation, billing "
        "EVERY persistent part: payload, codebooks, generators, bases, "
        "coefficients, indices, residuals, metadata. Pass incumbent=true for "
        "the sealed resident, or a fully declared candidate. An unreconciled "
        "candidate is REFUSED rather than billed low -- that refusal is the "
        "answer. This is the only accounting authority; do not compute a bpw "
        "yourself.",
        {"type": "object", "additionalProperties": False,
         "properties": {"incumbent": {"type": "boolean"},
                        "candidate": {"type": "object"}}},
        timeout_s=60.0,
        handler=_nr_complete_ebpw,
    ))
    # Snapshot AFTER every registration: taken earlier, tool.reachable would
    # report a door unreachable purely because its producer had not been
    # registered yet at snapshot time.
    _REACHABILITY_REGISTRY["registry"] = registry
    _REACHABILITY_REGISTRY["specs"] = {
        n: sp for n, sp in
        [(i["name"], registry.get(i["name"])) for i in registry.discover()]
        if sp is not None
    }
    _consolidate(registry)
    return registry


__all__ = [
    "COSTLY",
    "DESTRUCTIVE",
    "EXTERNAL_WRITE",
    "MUTATION_CLASSES",
    "READ_ONLY",
    "REPO_WRITE",
    "RESEARCH",
    "REVERSIBLE_REPO",
    "REVERSIBLE_RUNTIME",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "WORKSPACE_WRITE",
    "default_tool_registry",
    "validate_input",
]
