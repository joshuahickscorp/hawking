"""Git-backed source view.

This worktree is a sparse checkout. A path missing from disk is not evidence
the file does not exist. Reads go: overlay -> working tree -> `git show HEAD`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]

_HEAD_COMMIT: str | None = None
_HEAD_PATHS: frozenset[str] | None = None
_BLOB_CACHE: dict[tuple[str, str], str | None] = {}
# Keyed by SOURCE, not global. A SourceView can read HEAD blobs or the working
# tree, and those are DIFFERENT file sets: `ls-tree HEAD` omits untracked files,
# `ls-files` includes staged-but-uncommitted ones. A single global cache let
# whichever view ran first decide what every later view saw, which is the
# order-dependent-verdict bug removed in lane s3. Do not collapse this back.
_TRACKED_PY_BY_SOURCE: dict[str, list[str]] = {}


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def head_commit() -> str:
    global _HEAD_COMMIT
    if _HEAD_COMMIT is None:
        _HEAD_COMMIT = _git("rev-parse", "HEAD").stdout.strip()
    return _HEAD_COMMIT


def head_paths() -> frozenset[str]:
    """Every path in HEAD. One `ls-tree`, not one `cat-file -e` per exists()."""
    global _HEAD_PATHS
    if _HEAD_PATHS is None:
        out = _git("ls-tree", "-r", "--name-only", "-z", "HEAD").stdout
        _HEAD_PATHS = frozenset(p for p in out.split("\0") if p)
    return _HEAD_PATHS


def _parse_cat_file_batch(data: bytes, rels: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    idx = 0
    for rel in rels:
        if idx >= len(data):
            out[rel] = None
            continue
        nl = data.find(b"\n", idx)
        if nl < 0:
            out[rel] = None
            break
        header = data[idx:nl].decode("utf-8", errors="replace")
        idx = nl + 1
        if " missing" in header:
            out[rel] = None
            continue
        parts = header.split()
        if len(parts) < 3 or parts[1] != "blob":
            out[rel] = None
            continue
        try:
            size = int(parts[2])
        except ValueError:
            out[rel] = None
            continue
        blob = data[idx : idx + size]
        idx = idx + size
        if idx < len(data) and data[idx : idx + 1] == b"\n":
            idx += 1
        out[rel] = blob.decode("utf-8", errors="replace")
    return out


def _cat_file_batch(commit: str, rels: list[str]) -> dict[str, str | None]:
    """One `git cat-file --batch` for many `commit:rel` blobs."""
    if not rels:
        return {}
    specs = "\n".join(f"{commit}:{rel}" for rel in rels) + "\n"
    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(REPO), "cat-file", "--batch"],
            input=specs.encode("utf-8"),
            capture_output=True,
            check=False,
        )
    except OSError:
        return {rel: None for rel in rels}
    return _parse_cat_file_batch(proc.stdout or b"", rels)


def prefetch_blobs(commit: str, rels: Iterable[str]) -> None:
    pending: list[str] = []
    seen: set[str] = set()
    for rel in rels:
        if not rel or rel in seen:
            continue
        seen.add(rel)
        if (commit, rel) in _BLOB_CACHE:
            continue
        pending.append(rel)
    if not pending:
        return
    fetched = _cat_file_batch(commit, pending)
    for rel, text in fetched.items():
        _BLOB_CACHE[(commit, rel)] = text


def blob_text(commit: str, rel: str) -> str | None:
    """File text at `commit:rel`, or None if that blob does not exist."""
    if not commit or not rel:
        return None
    key = (commit, rel)
    if key in _BLOB_CACHE:
        return _BLOB_CACHE[key]
    prefetch_blobs(commit, [rel])
    return _BLOB_CACHE.get(key)


class SourceView:
    """Readable snapshot of HEAD plus an optional overlay used by mutation checks."""

    def __init__(self) -> None:
        self.overlay: dict[str, str] = {}
        self._cache: dict[str, str] = {}
        self._exists_cache: dict[str, bool] = {}
        self._grep_cache: dict[str, list[str]] = {}
        self._py_files: list[str] | None = None

    def head_paths(self) -> frozenset[str]:
        return head_paths()

    def tracked_py(self) -> list[str]:
        if self._py_files is None:
            key = str(self.source)
            cached = _TRACKED_PY_BY_SOURCE.get(key)
            if cached is None:
                if key == "head":
                    out = _git("ls-tree", "-r", "--name-only", "HEAD").stdout.splitlines()
                    cached = [
                        line
                        for line in out
                        if line.endswith(".py") and line and "__pycache__" not in line
                    ]
                else:
                    out = _git("ls-files", "*.py").stdout.splitlines()
                    cached = [line for line in out if line and "__pycache__" not in line]
                _TRACKED_PY_BY_SOURCE[key] = cached
            self._py_files = cached
        files = list(self._py_files)
        for rel in self.overlay:
            if rel.endswith(".py") and rel not in files:
                files.append(rel)
        return files

    def exists(self, rel: str) -> bool:
        if rel in self.overlay:
            return True
        if rel in self._exists_cache:
            return self._exists_cache[rel]
        disk = REPO / rel
        if disk.is_file():
            self._exists_cache[rel] = True
            return True
        present = rel in head_paths()
        self._exists_cache[rel] = present
        return present

    def prefetch(self, rels: Iterable[str]) -> None:
        """Fill the read/exists caches. Overlay wins; disk next; one cat-file batch for the rest."""
        pending: list[str] = []
        seen: set[str] = set()
        for rel in rels:
            if not rel or rel in seen:
                continue
            seen.add(rel)
            if rel in self.overlay or rel in self._cache:
                self._exists_cache[rel] = True
                continue
            disk = REPO / rel
            if disk.is_file():
                try:
                    self._cache[rel] = disk.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    self._cache[rel] = ""
                self._exists_cache[rel] = True
                continue
            pending.append(rel)
        if not pending:
            return
        commit = head_commit()
        prefetch_blobs(commit, pending)
        for rel in pending:
            text = _BLOB_CACHE.get((commit, rel))
            if text is None:
                self._exists_cache[rel] = False
                self._cache[rel] = ""
            else:
                self._exists_cache[rel] = True
                self._cache[rel] = text

    def read(self, rel: str) -> str:
        if rel in self.overlay:
            return self.overlay[rel]
        if rel in self._cache:
            return self._cache[rel]
        self.prefetch([rel])
        return self._cache.get(rel, "")

    def grep_files(self, needle: str) -> list[str]:
        """Candidate files whose HEAD blob contains `needle`. Overlay files are always included."""
        if not needle:
            return []
        if needle not in self._grep_cache:
            self.prefetch_grep([needle])
        hits = list(self._grep_cache.get(needle) or [])
        for rel in self.overlay:
            if rel.endswith(".py") and rel not in hits:
                hits.append(rel)
        return hits

    def prefetch_grep(self, needles: list[str]) -> None:
        """One `git grep -F` for many needles; fills `_grep_cache` (HEAD blobs)."""
        pending = [n for n in needles if n and n not in self._grep_cache]
        if not pending:
            return
        args = ["grep", "-n", "-F"]
        for n in pending:
            args.extend(["-e", n])
        args.extend(["HEAD", "--", "*.py"])
        cp = _git(*args, check=False)
        hits: dict[str, list[str]] = {n: [] for n in pending}
        seen: dict[str, set[str]] = {n: set() for n in pending}
        for line in cp.stdout.splitlines():
            if not line:
                continue
            if line.startswith("HEAD:"):
                line = line[len("HEAD:") :]
            # path:lineno:text — lineno is forced by -n.
            rel, sep, rest = line.partition(":")
            lineno, sep2, text = rest.partition(":")
            if not sep or not sep2 or not lineno.isdigit():
                rel, _, text = line.partition(":")
            if not rel:
                continue
            blob = text if sep2 else line
            for n in pending:
                if n in blob and rel not in seen[n]:
                    seen[n].add(rel)
                    hits[n].append(rel)
        for n in pending:
            self._grep_cache[n] = hits[n]

    def path(self, rel: str) -> Path:
        return REPO / rel


def classify_symbol(text: str, symbol: str) -> tuple[str | None, int | None]:
    """Classify a module-level name: function / class / assignment / None.

    A NAME that is only assigned (a constant, a string, a comment) is not an
    invocable implementing symbol. Call-site analysis must not treat it as one.
    """
    import ast

    if not text or not symbol:
        return None, None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _classify_symbol_lines(text, symbol)
    # Module-level first so a constant named like a nested helper cannot
    # masquerade as an invocable, and a method still counts as a function.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
            return "function", int(node.lineno)
        if isinstance(node, ast.ClassDef) and node.name == symbol:
            return "class", int(node.lineno)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    return "assignment", int(node.lineno)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == symbol
        ):
            return "assignment", int(node.lineno)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
            return "function", int(node.lineno)
        if isinstance(node, ast.ClassDef) and node.name == symbol:
            return "class", int(node.lineno)
    return None, None


def _classify_symbol_lines(text: str, symbol: str) -> tuple[str | None, int | None]:
    prefix_def = f"def {symbol}("
    prefix_async = f"async def {symbol}("
    prefix_cls = f"class {symbol}"
    assign = f"{symbol} ="
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith(prefix_def) or stripped.startswith(prefix_async):
            return "function", i
        if stripped.startswith(prefix_cls) and (
            len(stripped) == len(prefix_cls)
            or stripped[len(prefix_cls)] in "(: \t"
        ):
            return "class", i
        if stripped.startswith(assign):
            return "assignment", i
    return None, None


def definition_line(text: str, symbol: str) -> int | None:
    """First invocable `def`/`class` line for `symbol`, or None.

    Assignments (constants) are not definitions of a callable capability.
    """
    kind, line = classify_symbol(text, symbol)
    if kind in ("function", "class"):
        return line
    return None


def iter_py_rel(view: SourceView, extra: Iterable[str] = ()) -> list[str]:
    files = list(view.tracked_py())
    for rel in extra:
        if rel not in files:
            files.append(rel)
    return files
