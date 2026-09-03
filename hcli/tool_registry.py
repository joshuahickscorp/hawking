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
_MAX_SEARCH_FILES = 20_000
_MAX_LIST_DIRECTORIES = 20_000


def _redact(value: Any, *, limit: int = 4000) -> Any:
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
    handler: Callable[[ToolContext, Dict[str, Any]], Any] = field(repr=False, compare=False, default=lambda _c, _a: None)

    def to_dict(self) -> Dict[str, Any]:
        return {
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
        self._tools[name] = spec
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(str(name or "").strip())

    def discover(self, *, role: Optional[str] = None) -> List[Dict[str, Any]]:
        result = []
        for spec in sorted(self._tools.values(), key=lambda item: item.name):
            if role and spec.roles and role not in spec.roles:
                continue
            result.append(spec.to_dict())
        return result

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
        matches = [spec.to_dict() for _score, _name, spec in scored[:limit]]
        return {
            "focus": query,
            "matches": matches,
            "match_count": len(scored),
            "truncated": len(scored) > limit,
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
        raise FileNotFoundError(path)
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
            "truncated": len(body) > limit,
            "sha256": _sha256_bytes(raw),
            "content": clipped.decode(encoding, errors="replace"),
            "artifact": {"kind": "file", "path": str(path), "sha256": _sha256_bytes(raw), "bytes": len(raw)},
        }

    clipped = raw[:limit]
    return {
        "path": str(path),
        "bytes": len(raw),
        "truncated": len(raw) > limit,
        "sha256": _sha256_bytes(raw),
        "content": clipped.decode(encoding, errors="replace"),
        "artifact": {"kind": "file", "path": str(path), "sha256": _sha256_bytes(raw), "bytes": len(raw)},
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
    if not root.is_dir():
        raise NotADirectoryError(root)
    needle = str(args.get("pattern") or "")
    if not needle:
        raise ValueError("pattern is required")
    glob = str(args.get("glob") or "*")
    limit = max(1, min(1000, int(args.get("max_results") or 100)))
    matches: List[Dict[str, Any]] = []
    files_seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in {".git", ".venv", "__pycache__"})
        for filename in sorted(filenames):
            if not Path(filename).match(glob):
                continue
            files_seen += 1
            if files_seen > _MAX_SEARCH_FILES:
                return {"root": str(root), "pattern": needle, "matches": matches, "truncated": True, "files_seen": files_seen}
            path = Path(dirpath) / filename
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
    return _run_readonly(["git", "-C", str(cwd), "log", f"-{limit}", "--oneline", "--decorate"], cwd=cwd)


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
    del context, args
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
    result = _run_readonly(argv, cwd=root, timeout=timeout)
    result.update({
        "runner": runner,
        "root": str(root),
        "started_at": started,
        "finished_at": time.time(),
        "verified": result.get("returncode") == 0,
    })
    return result


def _vmcp_inspect(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    from .vmcp_adapter import inspect_vmcp

    return inspect_vmcp(context.repo_root, profile=str(args.get("profile") or "core"))


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
        "authority": "measurement/verifier, not Doctor",
        "external_research_used": False,
        "physical_execution": False,
    }
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
        result["status"] = "REFUSED_NOT_IMPLEMENTED"
        result["blocker"] = "HCLI does not claim physical Gravity execution from a metadata tool"
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

        return getattr(odyssey, name)()

    return handler


def _odyssey_cycle(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """One Odyssey cycle. Mutating and expensive, so it is gated the way every
    other costly verb here is: an explicit confirm, refused by default."""
    from . import odyssey

    if args.get("confirm") is not True:
        raise PermissionError("odyssey.cycle mutates Odyssey state and requires confirm=True")
    return odyssey.cycle(confirm=True, max_lanes=args.get("max_lanes"))


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


def _specimens_registry(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Every sealed specimen, enumerated from disk. SEALED != LOAD NOW."""
    from . import specimens

    name = str(args.get("name") or "").strip()
    if name:
        found = specimens.get(name)
        return {"name": name, "specimen": found, "found": found is not None}
    return specimens.registry()


def _acquisition_propose(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Rank what to acquire next. Proposes and explains; never starts a
    download - that stays behind explicit confirmation elsewhere."""
    from . import acquisition

    return acquisition.propose()


def _odyssey_read_verb(name: str):
    def handler(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        from . import odyssey

        return getattr(odyssey, name)()

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
    """Every live Hawking process, classified by argv. Read-only: this wraps
    hcli.processes.live_processes() and nothing else. Killing a process is
    deliberately NOT reachable here -- that stays on the owned-signal path in
    hcli/agentos/resident.py (_owned_signal), which checks a
    process_start_token before it will ever send a signal.
    """
    del context, args
    from . import processes

    return {"processes": [p.to_dict() for p in processes.live_processes()]}


def _processes_summary(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Roll-up counts and total footprint for every live Hawking process --
    the same hcli.processes.summary() entry point the process audit receipt
    uses. Read-only."""
    del context, args
    from . import processes

    return processes.summary()


def _processes_orphaned(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Resident model bodies with no live owner (reparented to pid 1, unclaimed
    by any resident state file). Enumeration only: this calls
    hcli.processes.orphaned_resident_bodies(), never reap_orphaned_bodies(),
    which sends SIGTERM. Reaping stays a startup-only self-heal in
    hcli/runtime.py._reap_orphans_once, not something the model can trigger
    through the tool surface.
    """
    del context, args
    from . import processes

    return {"orphaned": [p.to_dict() for p in processes.orphaned_resident_bodies()]}


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
    registry.register(ToolSpec("filesystem.read", "Read one known file under an AgentOS read root.", path_schema, handler=_read_file))
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
        handler=_search_files,
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
        list_schema, handler=_list_files,
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
    # through this registry at all; see the handler docstrings.
    # G009 (receipts/sovereign/G009_reachability.json) found process truth
    # REACHABLE FROM PRODUCTION CODE (hcli/runtime.py's startup reaper,
    # hcli/commands.py's /processes command) but UNREACHABLE FROM THE MODEL:
    # no registered tool named a process, and shell.readonly above refuses
    # `ps` outright. The live goal's first law names processes as authority
    # and gave the resident no way to look at one. These three close that gap
    # by wrapping hcli/processes.py's existing read paths only -- nothing new
    # is taught to reap or signal anything. Killing a process is not reachable
    # through this registry at all; see the handler docstrings.
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
        mutation=RESEARCH, deterministic=False, resources=("network",), handler=_huggingface_resolve,
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
    registry.register(ToolSpec("receipt.inspect", "Inspect a JSON/text receipt under the repository or mission state roots.", path_schema, handler=_receipt_read))
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
        "odyssey.cycle",
        "Advance the LIVE Odyssey by one cycle. Mutating; requires confirm=True.",
        {"type": "object", "additionalProperties": False,
         "required": ["confirm"],
         "properties": {"confirm": {"type": "boolean"}, "max_lanes": {"type": "integer"}}},
        mutation=COSTLY, deterministic=False, resources=("cpu",),
        handler=_odyssey_cycle,
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
         "properties": {"name": {"type": "string"}}},
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
    for verb, required in (
        ("add_to_eligibility", ("oxx",)),
        ("park_specimen", ("oxx",)),
        ("record_law", ("text",)),
        ("record_scar", ("law_id",)),
        ("create_transfer_probe", ("law_id", "target_oxx")),
        ("create_adversarial_probe", ("law_id",)),
    ):
        props = {"confirm": {"type": "boolean"}}
        for field in ("oxx", "text", "law_id", "target_oxx", "note", "reason",
                      "evidence", "source_oxx", "attack", "description"):
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
        "doctor.query", "Ask the local Doctor evidence/planning surface for a bounded proposal.",
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
        handler=_target_receipt("roadmap.read"),
    ))
    registry.register(ToolSpec(
        "benchmark.inspect", "Inspect benchmark evidence through a named receipt path.",
        {"type": "object", "required": ["path"], "additionalProperties": False, "properties": {"path": {"type": "string"}}},
        handler=_receipt_read,
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
