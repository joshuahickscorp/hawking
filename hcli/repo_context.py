"""What the model can know about the folder you opened it in.

MEASURED CONSTRAINT, NOT A STYLE CHOICE. Injecting context is not free here: a
cold prompt token costs about what a generated token costs (~30 ms on
sealed-3.14), so a careless 4000-token dump is two minutes before the first
word. But prefix reuse on the native resident is 30x for a prefix that does not
change between turns. Both facts together dictate the layout:

    [ STABLE ..................... ] [ VOLATILE ...... ] [ conversation ]
      repo identity, tree, README     files this question    the messages
      -- byte-identical every turn     happens to touch

The stable block is paid ONCE per session and reused; only the small volatile
block is re-prefilled per question. Putting the question-specific files first --
the obvious order -- would break the shared prefix on every turn and re-pay the
whole thing, which is the difference between a 0.5 s follow-up and a 20 s one.

CWD IS CONTEXT, NOT IDENTITY. Opening HCLI inside a repository must not turn
"explain DeltaNet" into a question about this repository. So the injected block
says what the repo IS and explicitly licenses ignoring it, rather than
instructing the model that every question is about these files.

RETRIEVAL, NOT A TOOL LOOP. This gives the model what is already on disk before
it answers. It cannot go and fetch more, and it does not pretend to: the block
names the files it included so a reader can tell the difference between "the
model reasoned about this file" and "the model was never shown it".
"""
from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: Roughly 4 chars per token. These are budgets in CHARACTERS.
STABLE_BUDGET = 6000
VOLATILE_BUDGET = 6000
FILE_HEAD = 2400

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "target",
    "build", "dist", ".mypy_cache", ".pytest_cache", ".worktrees",
    "receipts", "workspace", "evidence", ".hcli",
}
CODE_SUFFIXES = {".py", ".rs", ".ts", ".tsx", ".js", ".go", ".c", ".h", ".cpp",
                 ".metal", ".swift", ".sh", ".toml", ".md", ".json", ".yaml", ".yml"}
_CODE_SUFFIX_TUPLE = tuple(CODE_SUFFIXES)
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-/]{2,}")

_CONTENT_STOP_WORDS = {
    "actual", "actions", "across", "blind", "claim", "conducting",
    "determine", "engineering", "equivalently", "evidence", "exact",
    "files", "focused", "general", "guess", "implement", "inspect",
    "isolated", "mission", "modify", "objective", "observations", "path",
    "preserves", "regression", "repair", "report", "repository",
    "reproduce", "results", "returned", "shows", "smallest", "tests",
    "tools", "unrelated", "use", "using", "whether", "without", "worktree",
}


def _validated_native_code_path(raw_path: object) -> Optional[str]:
    """Accept only the normalized relative paths the native owner emits.

    The native result is still an untrusted process boundary.  Keep the
    containment fence, but validate the common response shape with string
    operations instead of constructing a ``Path`` for every returned row.
    Native repository paths are slash-separated, normalized, and never have
    a trailing separator; rejecting those non-canonical forms is safer than
    silently normalizing a path received from a child process.
    """
    if not isinstance(raw_path, str) or not raw_path:
        return None
    if (
        raw_path.startswith("/")
        or "\\" in raw_path
        or raw_path == "."
        or raw_path == ".."
        or raw_path.startswith("../")
        or "/../" in raw_path
        or raw_path.endswith("/..")
        or "/./" in raw_path
        or raw_path.endswith("/.")
    ):
        return None
    filename = raw_path.rsplit("/", 1)[-1]
    if filename in ("", ".", "..") or not any(
        filename.endswith(suffix) and filename != suffix
        for suffix in _CODE_SUFFIX_TUPLE
    ):
        return None
    return raw_path


class _NativeGravityIndex:
    """One `hawking-gravityd` child shared by a Hawking daemon process.

    This is deliberately a transport adapter, not a second implementation of
    retrieval. The child owns index construction and query; Python only keeps
    the historical `RepoContext` result shape while the wider HCLI migration is
    in flight. It is enabled only by the daemon, so an ordinary short-lived
    `hcli` client cannot accidentally create a competing resident service.
    """

    def __init__(self, root: Path, binary: Path) -> None:
        self.root = root
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [str(binary), "--root", str(root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._tool_catalog_key: Optional[Tuple[Tuple[str, str], ...]] = None
        self._tool_dispatch_key: Optional[Tuple[Tuple[str, str], ...]] = None
        self._tool_dispatch_entries: Optional[Sequence[Dict[str, str]]] = None
        # A positive dispatch decision is stable for an immutable catalog and
        # permission set.  Keep it beside the child process so the hot path
        # does not pay a second JSONL round trip on every ToolRegistry call.
        # Negative decisions are deliberately not cached: a caller changing
        # authority must be checked against the native owner again.
        self._tool_dispatch_admission_cache: Dict[
            Tuple[str, str, Tuple[str, ...]], Dict[str, object]
        ] = {}

    def _request(self, payload: Dict[str, object]) -> Optional[Dict[str, object]]:
        with self._lock:
            if self._process.poll() is not None or self._process.stdin is None or self._process.stdout is None:
                return None
            try:
                self._process.stdin.write(
                    json.dumps(payload, separators=(",", ":")) + "\n"
                )
                self._process.stdin.flush()
                response = json.loads(self._process.stdout.readline())
            except (OSError, UnicodeError, ValueError):
                return None
        return response if isinstance(response, dict) and response.get("ok") else None

    def search(self, needle: str) -> Optional[List[str]]:
        response = self._request({"op": "search", "needle": needle, "max_results": 1000})
        if response is None:
            return None
        rows = response.get("matches") or []
        return sorted({str(row.get("path")) for row in rows if isinstance(row, dict) and row.get("path")})

    def search_files(self, path: Path, *, needle: str, glob: str,
                     max_results: int,
                     max_per_file: Optional[int]) -> Optional[Dict[str, object]]:
        payload: Dict[str, object] = {
            "op": "search_files", "path": str(path), "needle": needle,
            "glob": glob, "max_results": max_results,
        }
        if max_per_file is not None:
            payload["max_per_file"] = max_per_file
        return self._request(payload)

    def list_files(self, path: Path, *, glob: str, max_results: int,
                   recursive: bool) -> Optional[Dict[str, object]]:
        """List a contained directory through the resident Rust owner."""
        return self._request({
            "op": "list", "path": str(path), "glob": glob,
            "max_results": max_results, "recursive": recursive,
        })

    def read_file(self, path: Path, *, max_bytes: int,
                  start_line: Optional[int], end_line: Optional[int]) -> Optional[Dict[str, object]]:
        """Read one already-contained file through the resident Rust owner."""
        payload: Dict[str, object] = {
            "op": "read_file", "path": str(path), "max_bytes": max_bytes,
        }
        if start_line is not None:
            payload["start_line"] = start_line
        if end_line is not None:
            payload["end_line"] = end_line
        return self._request(payload)

    def rank_code_paths(self, suffixes: Sequence[str], words: Sequence[str],
                        *, max_results: int) -> Optional[Dict[str, object]]:
        """Rank direct source-path hints through the resident Rust owner."""
        return self._request({
            "op": "rank_code_paths", "suffixes": list(suffixes),
            "words": list(words), "max_results": max_results,
        })

    def rank_content_paths(self, terms: Sequence[str], *,
                           scope_prefix: Optional[str], test_focus: bool,
                           max_results: int) -> Optional[Dict[str, object]]:
        """Rank content-oriented repository evidence in one native request."""
        payload: Dict[str, object] = {
            "op": "rank_content_paths", "terms": list(terms),
            "test_focus": test_focus, "max_results": max_results,
        }
        if scope_prefix:
            payload["scope_prefix"] = scope_prefix
        return self._request(payload)

    def mega_kernel_plan(self, model_id: str,
                         organs: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
        """Ask the native core to normalize a model-aware MegaKernel plan."""
        return self._request({
            "op": "mega_kernel_plan",
            "model_id": str(model_id),
            "organs": list(organs),
        })

    def catalog(self, *, slug: Optional[str] = None,
                catalog: Optional[str] = None) -> Optional[Dict[str, object]]:
        payload: Dict[str, object] = {"op": "catalog"}
        if slug:
            payload["slug"] = slug
        if catalog:
            payload["catalog"] = catalog
        return self._request(payload)

    def tool_catalog(self, entries: Sequence[Dict[str, str]], terms: Sequence[str],
                     max_results: int) -> Optional[Dict[str, object]]:
        """Load one immutable catalog generation, then rank within it."""
        key = tuple((str(entry.get("name") or ""), str(entry.get("search_text") or ""))
                    for entry in entries)
        if self._tool_catalog_key != key:
            loaded = self._request({"op": "tool_catalog_load", "entries": list(entries)})
            if loaded is None:
                return None
            self._tool_catalog_key = key
        return self._request({
            "op": "tool_catalog_query", "terms": list(terms),
            "max_results": max_results,
        })

    def tool_dispatch_admit(self, entries: Sequence[Dict[str, str]], *, name: str,
                            mutation: str, permissions: Sequence[str]) -> Optional[Dict[str, object]]:
        if self._process.poll() is not None:
            self._tool_dispatch_admission_cache.clear()
            return None
        # ToolRegistry passes a cached immutable tuple on ordinary invocations.
        # Identity avoids hashing every registered tool again on the hot path;
        # independently constructed snapshots retain content-based validation.
        if self._tool_dispatch_entries is not entries:
            key = tuple((str(entry.get("name") or ""), str(entry.get("mutation") or ""))
                        for entry in entries)
        else:
            key = self._tool_dispatch_key
        if self._tool_dispatch_key != key:
            loaded = self._request({"op": "tool_dispatch_load", "entries": list(entries)})
            if loaded is None:
                return None
            self._tool_dispatch_key = key
            self._tool_dispatch_entries = entries
            self._tool_dispatch_admission_cache.clear()
        admission_key = (str(name), str(mutation), tuple(sorted(str(p) for p in permissions)))
        cached = self._tool_dispatch_admission_cache.get(admission_key)
        if cached is not None:
            return {**cached, "cached": True, "elapsed_ns": 0}
        response = self._request({
            "op": "tool_dispatch_admit", "name": name, "mutation": mutation,
            "permissions": list(permissions),
        })
        if response is not None and response.get("admitted") is True:
            self._tool_dispatch_admission_cache[admission_key] = dict(response)
        return response

    def stop(self) -> None:
        with self._lock:
            if self._process.poll() is None and self._process.stdin is not None:
                try:
                    self._process.stdin.write('{"op":"shutdown"}\n')
                    self._process.stdin.flush()
                except OSError:
                    pass
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except (OSError, subprocess.SubprocessError):
                pass


_NATIVE_GRAVITY_INDEXES: Dict[Path, _NativeGravityIndex] = {}
_NATIVE_GRAVITY_LOCK = threading.Lock()
_NATIVE_GRAVITY_BINARY_CACHE: Dict[Tuple[str, str], Path] = {}


def _native_gravity_binary(root: Path) -> Optional[Path]:
    explicit = os.environ.get("HCLI_GRAVITYD_BIN")
    explicit_path = Path(explicit).expanduser() if explicit else None
    cache_key = (str(root), explicit or "")
    cached = _NATIVE_GRAVITY_BINARY_CACHE.get(cache_key)
    if cached is not None:
        # An explicit override must not be shadowed by a fallback that was
        # found while the override was temporarily absent.
        if (
            (explicit_path is None or cached == explicit_path)
            and cached.is_file()
            and os.access(cached, os.X_OK)
        ):
            return cached
        # Do not retain a stale positive result after an uninstall or a
        # broken deployment. Negative results are deliberately not cached so
        # a later install can become visible without restarting HCLI.
        _NATIVE_GRAVITY_BINARY_CACHE.pop(cache_key, None)
    candidates = [Path(explicit).expanduser()] if explicit else []
    checkout = Path(__file__).resolve().parents[1]
    candidates.extend((
        checkout / "hawking-gravityd",
        root / "workspace" / "ops" / "build" / "rust" / "release" / "hawking-gravityd",
        root / "workspace" / "ops" / "build" / "rust" / "debug" / "hawking-gravityd",
        root / "target" / "release" / "hawking-gravityd",
        root / "target" / "debug" / "hawking-gravityd",
        checkout / "workspace" / "ops" / "build" / "rust" / "release" / "hawking-gravityd",
        checkout / "workspace" / "ops" / "build" / "rust" / "debug" / "hawking-gravityd",
    ))
    named = shutil.which("hawking-gravityd")
    if named:
        candidates.append(Path(named))
    found = next((path for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)
    if found is not None:
        _NATIVE_GRAVITY_BINARY_CACHE[cache_key] = found
    return found


def _native_gravity_index(root: Path) -> Optional[_NativeGravityIndex]:
    """Return the resident native owner with a read-mostly hot path.

    The service is immutable for the lifetime of its repository generation;
    only first creation needs the process-global lock.  Keep the common case
    as one dictionary lookup so every native query does not repeat executable
    stat/access checks and lock acquisition.  The request method still
    verifies the child process before writing, and a missing/dead owner
    returns ``None`` through the existing compatibility fallback.
    """
    current = _NATIVE_GRAVITY_INDEXES.get(root)
    if current is not None:
        return current
    binary = _native_gravity_binary(root)
    if binary is None:
        return None
    with _NATIVE_GRAVITY_LOCK:
        current = _NATIVE_GRAVITY_INDEXES.get(root)
        if current is not None:
            return current
        try:
            current = _NativeGravityIndex(root, binary)
        except OSError:
            return None
        _NATIVE_GRAVITY_INDEXES[root] = current
        return current


def _native_content_matches(root: Path, term: str) -> Optional[List[str]]:
    if os.environ.get("HCLI_NATIVE_GRAVITY") != "1":
        return None
    index = _native_gravity_index(root)
    if index is None:
        return None
    return index.search(term)


def native_modellake_catalog(root: Path, *, slug: Optional[str] = None,
                             catalog: Optional[str] = None) -> Optional[Dict[str, object]]:
    """Query the daemon-owned native ModelLake catalog, or return ``None``.

    ``None`` means the native lane is unavailable, not that a specimen is
    absent. Callers retain an explicit compatibility reader during migration.
    """
    if os.environ.get("HCLI_NATIVE_GRAVITY") != "1":
        return None
    index = _native_gravity_index(root)
    if index is None:
        return None
    return index.catalog(slug=slug, catalog=catalog)


def native_tool_catalog(root: Path, *, entries: Sequence[Dict[str, str]],
                        terms: Sequence[str], max_results: int) -> Optional[Dict[str, object]]:
    """Rank an HCLI-owned immutable tool catalog in the resident Rust child."""
    if os.environ.get("HCLI_NATIVE_GRAVITY") != "1":
        return None
    index = _native_gravity_index(root)
    if index is None:
        return None
    return index.tool_catalog(entries, terms, max_results)


def native_tool_dispatch_admit(root: Path, *, entries: Sequence[Dict[str, str]],
                                name: str, mutation: str,
                                permissions: Sequence[str]) -> Optional[Dict[str, object]]:
    """Native admission fence for a Python-owned tool invocation."""
    if os.environ.get("HCLI_NATIVE_GRAVITY") != "1":
        return None
    index = _native_gravity_index(root)
    if index is None:
        return None
    return index.tool_dispatch_admit(
        entries, name=name, mutation=mutation, permissions=permissions,
    )


def native_filesystem_list(root: Path, path: Path, *, glob: str,
                           max_results: int, recursive: bool) -> Optional[Dict[str, object]]:
    """Native bounded filesystem discovery; ``None`` preserves parity fallback."""
    if os.environ.get("HCLI_NATIVE_GRAVITY") != "1":
        return None
    # The native owner is rooted at the repository. Do not ask it to inspect a
    # separate workspace/read root: Python's exact read-root contract remains
    # the authority for that case.
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return None
    index = _native_gravity_index(root)
    if index is None:
        return None
    return index.list_files(path, glob=glob, max_results=max_results, recursive=recursive)


def native_filesystem_read(root: Path, path: Path, *, max_bytes: int,
                           start_line: Optional[int],
                           end_line: Optional[int]) -> Optional[Dict[str, object]]:
    """Native bounded UTF-8 file read; ``None`` preserves rich fallback errors."""
    if os.environ.get("HCLI_NATIVE_GRAVITY") != "1":
        return None
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return None
    index = _native_gravity_index(root)
    if index is None:
        return None
    return index.read_file(
        path, max_bytes=max_bytes, start_line=start_line, end_line=end_line,
    )


def native_rank_code_paths(root: Path, *, suffixes: Sequence[str],
                           words: Sequence[str], max_results: int) -> Optional[Dict[str, object]]:
    """Native direct-path ranking; ``None`` preserves the exact fallback."""
    if os.environ.get("HCLI_NATIVE_GRAVITY") != "1":
        return None
    index = _native_gravity_index(root)
    if index is None:
        return None
    return index.rank_code_paths(suffixes, words, max_results=max_results)


def native_rank_content_paths(root: Path, *, terms: Sequence[str],
                              scope_prefix: Optional[str], test_focus: bool,
                              max_results: int) -> Optional[Dict[str, object]]:
    """Native multi-term content ranking; ``None`` preserves the fallback."""
    if os.environ.get("HCLI_NATIVE_GRAVITY") != "1":
        return None
    index = _native_gravity_index(root)
    if index is None:
        return None
    return index.rank_content_paths(
        terms, scope_prefix=scope_prefix, test_focus=test_focus,
        max_results=max_results,
    )


def native_mega_kernel_plan(root: Path, *, model_id: str,
                            organs: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """Return the Rust-owned plan when daemon-native Gravity is available.

    None is an ordinary compatibility fallback: it means the current daemon
    either predates the endpoint or has no native owner, not that a model lacks
    a physical plan.
    """

    if os.environ.get("HCLI_NATIVE_GRAVITY") != "1":
        return None
    index = _native_gravity_index(root)
    if index is None:
        return None
    return index.mega_kernel_plan(model_id, organs)


def native_filesystem_search(root: Path, path: Path, *, needle: str, glob: str,
                             max_results: int,
                             max_per_file: Optional[int]) -> Optional[Dict[str, object]]:
    """Native bounded text search; ``None`` preserves the exact fallback."""
    if os.environ.get("HCLI_NATIVE_GRAVITY") != "1":
        return None
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return None
    index = _native_gravity_index(root)
    if index is None:
        return None
    return index.search_files(
        path, needle=needle, glob=glob, max_results=max_results,
        max_per_file=max_per_file,
    )


def prewarm_native_gravity_index(root: Path) -> bool:
    """Start the daemon-owned native index without making a request wait.

    ``hawking-gravityd`` builds its first coherent generation during child
    startup. Starting it while the resident model is loading overlaps that
    one-time work with an already-required operation; it does not claim that a
    brand-new process can search a large repository in milliseconds. Calls are
    idempotent, and the process remains governed by the parent HCLI daemon's
    normal atexit cleanup.
    """
    if os.environ.get("HCLI_NATIVE_GRAVITY") != "1":
        return False
    current = _NATIVE_GRAVITY_INDEXES.get(root)
    if current is not None and current._process.poll() is None:
        return True
    if current is not None:
        # The read-mostly query path intentionally does not pay a process
        # liveness syscall on every request; prewarm is the lifecycle boundary
        # that may replace a dead child and restore the resident owner.
        with _NATIVE_GRAVITY_LOCK:
            if _NATIVE_GRAVITY_INDEXES.get(root) is current:
                _NATIVE_GRAVITY_INDEXES.pop(root, None)
        current.stop()
    return _native_gravity_index(root) is not None


def _stop_native_gravity_indexes() -> None:
    with _NATIVE_GRAVITY_LOCK:
        indexes = list(_NATIVE_GRAVITY_INDEXES.values())
        _NATIVE_GRAVITY_INDEXES.clear()
    for index in indexes:
        index.stop()


atexit.register(_stop_native_gravity_indexes)


@dataclass
class RepoContext:
    root: Path
    name: str
    is_git: bool
    branch: str = ""
    head: str = ""

    @classmethod
    def detect(cls, start: Optional[str] = None) -> Optional["RepoContext"]:
        """The repository the user opened HCLI in, or None if this is not one."""
        here = Path(os.path.expanduser(start or os.getcwd())).resolve()
        if not here.is_dir():
            return None
        top = here
        for candidate in [here, *here.parents]:
            if (candidate / ".git").exists():
                top = candidate
                break
        else:
            # Not a git repo. A plain directory of source is still context.
            if not any(p.suffix in CODE_SUFFIXES for p in _shallow(here)):
                return None
            return cls(root=here, name=here.name, is_git=False)
        branch = _git(top, "rev-parse", "--abbrev-ref", "HEAD")
        head = _git(top, "rev-parse", "--short", "HEAD")
        return cls(root=top, name=top.name, is_git=True, branch=branch, head=head)

    # -- the two blocks -----------------------------------------------------
    def stable_block(self) -> str:
        """Byte-identical between turns, so the resident's prefix cache keeps it."""
        lines = [
            f"You are running inside the {self.name!r} repository at {self.root}.",
        ]
        if self.is_git:
            lines.append(f"git: branch {self.branch or '?'} at {self.head or '?'}")
        lines.append("")
        lines.append("Top level:")
        for entry in self._top_level():
            lines.append(f"  {entry}")
        readme = self._readme()
        if readme:
            lines.append("")
            lines.append("README (head):")
            lines.append(readme)
        lines.append("")
        lines.append(
            "Use this when the question is about this repository. When it is a "
            "general question, answer it generally -- being launched in a folder "
            "does not make every question about that folder.")
        text = "\n".join(lines)
        return text[:STABLE_BUDGET]

    def volatile_block(self, question: str) -> str:
        """The files this particular question appears to be about."""
        picked = self.files_for(question)
        if not picked:
            return ""
        lines = ["Files from this repository that may be relevant:"]
        budget = VOLATILE_BUDGET
        for path, body in picked:
            if budget <= 0:
                break
            chunk = body[:min(FILE_HEAD, budget)]
            lines.append(f"\n--- {path} ---\n{chunk}")
            budget -= len(chunk)
        lines.append(
            "\nThose are the only files you were shown. If the answer needs one "
            "that is not here, say which file you would need rather than guessing "
            "its contents.")
        return "\n".join(lines)

    # -- retrieval ----------------------------------------------------------
    def paths_for(self, question: str, limit: int = 4) -> List[str]:
        """Rank a small, explainable repository evidence set for ``question``.

        Explicit path matches still dominate.  When the operator supplies only
        an objective, content matches across several distinctive terms provide
        orientation; without that fallback Engine receives zero files and asks
        the model to mutate a repository it has never seen.
        """
        raw = {w.lower() for w in _WORD.findall(question or "")}
        # "hcli/serve.py" is one token AND its parts: the question names a path,
        # a module and a stem all at once, and each is a legitimate way to find
        # the file.
        words = set(raw)
        for token in raw:
            for part in re.split(r"[./]", token):
                if len(part) > 2:
                    words.add(part)
        words -= {"the", "this", "that", "what", "does", "file", "code", "repo",
                  "repository", "function", "where", "which", "how", "why",
                  "explain", "show", "and", "for", "with", "from"}
        # Objective boilerplate must not become filename evidence. In the
        # blind mission, words such as ``path``, ``test`` and ``tools`` ranked
        # three unrelated test modules above serve.py before content scoring
        # had a chance to discriminate.
        words -= _CONTENT_STOP_WORDS
        words.discard("hcli")
        if not words:
            return []
        # A path-shaped question can use the native owner without exporting
        # the entire source inventory over JSONL. Non-path questions retain
        # the existing content-aware Python scorer; the native result is
        # accepted only when it proves a high-confidence direct match.
        path_like = any(
            "/" in token or any(token.endswith(suffix) for suffix in CODE_SUFFIXES)
            for token in raw
        )
        content_terms = self._content_terms(question)
        if path_like and limit > 0:
            try:
                native = native_rank_code_paths(
                    self.root,
                    suffixes=tuple(sorted(CODE_SUFFIXES)),
                    words=tuple(sorted(words)),
                    max_results=limit,
                )
            except Exception:
                native = None
            if native is not None and native.get("complete") is True:
                rows = native.get("paths")
                if isinstance(rows, list):
                    ranked: List[str] = []
                    valid = True
                    for raw_path in rows:
                        candidate = _validated_native_code_path(raw_path)
                        if candidate is None:
                            valid = False
                            break
                        ranked.append(candidate)
                    if valid:
                        return ranked
        if not path_like and limit > 0 and content_terms:
            try:
                native = native_rank_content_paths(
                    self.root,
                    terms=tuple(content_terms),
                    scope_prefix=("hcli/" if re.search(r"\bhcli\b", question or "", re.IGNORECASE) else None),
                    test_focus=bool(re.search(
                        r"\b(?:test|tests|regression)\b", question or "", re.IGNORECASE,
                    )),
                    max_results=limit,
                )
            except Exception:
                native = None
            if native is not None:
                rows = native.get("paths")
                if isinstance(rows, list):
                    ranked = []
                    valid = True
                    for raw_path in rows:
                        candidate = _validated_native_code_path(raw_path)
                        if candidate is None:
                            valid = False
                            break
                        ranked.append(candidate)
                    if valid:
                        return ranked
        scores: Dict[str, int] = {}
        if path_like:
            # A path the question NAMED must outrank a file that merely shares
            # a word with it. Measured: asking about "hcli/serve.py and the
            # stream flag" returned visionmcp/ocular/stream.py, because a stem
            # hit on "stream" (10) beat a substring hit on the full path (3).
            # Exactness is the ranking, not word count. This inventory is only
            # the compatibility fallback when the native path rank is absent
            # or below its confidence threshold.
            for path in self._candidate_files():
                rel = str(path.relative_to(self.root))
                stem = path.stem.lower()
                lowered = rel.lower()
                score = 0
                for word in words:
                    if word == lowered:
                        score += 25
                    elif lowered.endswith("/" + word) or word == path.name.lower():
                        score += 20
                    elif word == stem:
                        score += 10
                    elif word in lowered:
                        score += 3
                if score:
                    scores[rel] = score

        # A directly named path is already stronger evidence than a content
        # search. Preserve the existing precision rule and avoid paying for
        # repository search when the user supplied the answer's address.
        if scores and max(scores.values()) >= 20:
            scored = sorted(scores.items(), key=lambda row: (-row[1], len(row[0])))
            best = scored[0][1]
            floor = best * 0.5
            return [rel for rel, score in scored if score >= floor][:limit]

        for term in content_terms:
            matches = self._content_matches(term)
            count = len(matches)
            if not count:
                continue
            # Rare terms discriminate. Common words still contribute, but one
            # ubiquitous token cannot crowd out a file matching several terms.
            weight = 8 if count <= 10 else 4 if count <= 50 else 2 if count <= 200 else 1
            for rel in matches:
                scores[rel] = scores.get(rel, 0) + weight

        lowered_question = (question or "").lower()
        if re.search(r"\bhcli\b", lowered_question):
            for rel in list(scores):
                if rel.startswith("hcli/"):
                    scores[rel] += 4
            scoped = {rel: score for rel, score in scores.items()
                      if rel.startswith("hcli/")}
            if scoped:
                scores = scoped
        if re.search(r"\b(?:test|tests|regression)\b", lowered_question):
            for rel in list(scores):
                if "test" in Path(rel).name.lower():
                    scores[rel] += 3

        scored = sorted(scores.items(), key=lambda row: (-row[1], len(row[0]), row[0]))
        # PRECISION OVER RECALL when the question named something exactly.
        # Measured: asking about hcli/catalog.py returned it AND three other
        # files called catalog, and the model produced a blended description of
        # all four -- fluent, confident, and about no file that exists. When the
        # best hit is far ahead, the also-rans are noise, not context.
        if scored:
            best = scored[0][1]
            floor = best * 0.5 if best >= 20 else 0
            scored = [row for row in scored if row[1] >= floor]
        return [rel for rel, _score in scored[:limit]]

    def files_for(self, question: str, limit: int = 4) -> List[Tuple[str, str]]:
        """Read the bounded paths selected by :meth:`paths_for`."""
        out: List[Tuple[str, str]] = []
        for rel in self.paths_for(question, limit=limit):
            try:
                out.append((rel, (self.root / rel).read_text(
                    encoding="utf-8", errors="replace")))
            except OSError:
                continue
        return out

    @staticmethod
    def _content_terms(question: str, limit: int = 12) -> List[str]:
        terms: List[str] = []
        seen = set()
        for raw in _WORD.findall(question or ""):
            token = raw.lower().strip("._-/")
            variants = [token, *re.split(r"[._/-]+", token)]
            for term in variants:
                if (
                    len(term) < 5
                    or term in _CONTENT_STOP_WORDS
                    or term in seen
                    # The split components are retained below. Sending the
                    # punctuation-bearing compound would force the native
                    # owner into a full substring scan instead of its token
                    # index, turning one fast request into an O(repository)
                    # query for terms such as ``h-manifesto``.
                    or any(separator in term for separator in ".-/")
                ):
                    continue
                seen.add(term)
                terms.append(term)
        # Compound and longer terms tend to be the most discriminating.
        terms.sort(key=lambda item: ("-" not in item, -len(item), item))
        return terms[:limit]

    def _content_matches(self, term: str) -> List[str]:
        native = _native_content_matches(self.root, term)
        if native is not None:
            return native
        args = [
            "rg", "-l", "-i", "-F",
            "--glob", "!receipts/**",
            "--glob", "!workspace/**",
            "--glob", "!evidence/**",
            "--glob", "!.hcli/**",
            "--glob", "!.git/**",
            "--", term, ".",
        ]
        try:
            done = subprocess.run(
                args,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        if done.returncode not in (0, 1):
            return []
        found = []
        for row in done.stdout.splitlines()[:1000]:
            rel = row.strip().removeprefix("./")
            path = Path(rel)
            if path.suffix not in CODE_SUFFIXES:
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            found.append(rel)
        return sorted(set(found))

    def _candidate_files(self) -> List[Path]:
        found: List[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames
                           if d not in SKIP_DIRS and not d.startswith(".")]
            for name in filenames:
                if Path(name).suffix in CODE_SUFFIXES:
                    found.append(Path(dirpath) / name)
            if len(found) > 20000:
                return found
        return found

    def _top_level(self) -> List[str]:
        rows = []
        for entry in sorted(_shallow(self.root), key=lambda p: p.name):
            if entry.name.startswith(".") or entry.name in SKIP_DIRS:
                continue
            if entry.is_dir():
                try:
                    count = sum(1 for _ in entry.iterdir())
                except OSError:
                    count = 0
                rows.append(f"{entry.name}/ ({count} entries)")
            else:
                rows.append(entry.name)
            if len(rows) >= 40:
                break
        return rows

    def _readme(self) -> str:
        for name in ("README.md", "README.rst", "README.txt", "README"):
            path = self.root / name
            if path.is_file():
                try:
                    return path.read_text(encoding="utf-8", errors="replace")[:1500]
                except OSError:
                    return ""
        return ""


def _shallow(root: Path) -> List[Path]:
    try:
        return list(root.iterdir())
    except OSError:
        return []


def _git(root: Path, *args: str) -> str:
    try:
        done = subprocess.run(["git", "-C", str(root), *args],
                              capture_output=True, text=True, timeout=5)
        return done.stdout.strip() if done.returncode == 0 else ""
    except Exception:
        return ""


def inject(messages: Sequence[Dict[str, str]],
           context: Optional[RepoContext]) -> List[Dict[str, str]]:
    """Put the repo in front of the conversation, stable part first.

    Returns the messages unchanged when there is no context, so the caller does
    not have to branch and a general session pays nothing.
    """
    rows = [dict(m) for m in messages]
    if context is None:
        return rows
    question = ""
    for message in reversed(rows):
        if message.get("role") == "user":
            question = str(message.get("content") or "")
            break
    blocks = [context.stable_block()]
    volatile = context.volatile_block(question)
    if volatile:
        blocks.append(volatile)
    return [{"role": "system", "content": "\n\n".join(blocks)}, *rows]
