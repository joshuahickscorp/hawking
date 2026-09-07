"""Client for the hawking-index python-facts JSON surface.

 The query binary is now owned by the `hawking-index` package. This client
 accepts either the compatibility executable name or the primary index name:

    hawking-index-query python-facts --git-head --commit <sha> --repo <repo>
    hawking-index python-facts --git-head --commit <sha> --repo <repo>

Schema: hawking.index.python_facts.v1

The dump is built once per SourceView (overlay included) from the named git
commit's blobs (default HEAD), never the working tree. A sparse worktree
where hcli/ is absent from disk still indexes those files. Untracked and
uncommitted files are invisible. Every file fact carries the commit it was
parsed from.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tools.roadmap.gitfs import REPO, SourceView, head_commit

SCHEMA = "hawking.index.python_facts.v1"

# Set ROADMAP_REACH_BACKEND=ast to force the old capability_reachability path.
# index = require the rust dump; auto = index if the binary exists else ast.
_BACKEND_ENV = "ROADMAP_REACH_BACKEND"


def backend() -> str:
    raw = (os.environ.get(_BACKEND_ENV) or "auto").strip().lower()
    if raw in {"ast", "cr", "python"}:
        return "ast"
    if raw in {"index", "rust"}:
        return "index"
    return "auto"


def _bin_candidates() -> list[Path]:
    out: list[Path] = []
    for key in ("HAWKING_INDEX_QUERY_BIN", "HAWKING_INDEX_BIN"):
        env = os.environ.get(key)
        if env:
            out.append(Path(env))
    roots = [
        REPO / "workspace" / "ops" / "build" / "rust",
        REPO / "target",
    ]
    names = ("hawking-index-query", "hawking-index")
    for root in roots:
        for kind in ("release", "debug"):
            for name in names:
                out.append(root / kind / name)
    which = shutil.which("hawking-index-query") or shutil.which("hawking-index")
    if which:
        out.append(Path(which))
    return out


def find_index_bin() -> Path | None:
    for cand in _bin_candidates():
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def resolve_backend() -> str:
    choice = backend()
    if choice == "ast":
        return "ast"
    if choice == "index":
        if find_index_bin() is None:
            raise FileNotFoundError(
                "ROADMAP_REACH_BACKEND=index but hawking-index-query (or "
                "hawking-index) is not built. cargo build -p hawking-index-query "
                "--bin hawking-index-query --release "
                "(CARGO_TARGET_DIR=workspace/ops/build/rust)"
            )
        return "index"
    return "index" if find_index_bin() is not None else "ast"


def _overlay_ndjson(view: SourceView) -> str:
    lines = []
    for rel, text in view.overlay.items():
        if not rel.endswith(".py"):
            continue
        lines.append(json.dumps({"path": rel, "content": text}, ensure_ascii=False))
    return "\n".join(lines)


def catalog_watch_names() -> list[str]:
    """Symbol / module-stem names the auditor will query. Shrinks the dump."""
    from tools.roadmap import catalog

    names: set[str] = set()
    for table in (catalog.GATES, catalog.GENES):
        for probe in table.values():
            for spec in probe.get("symbols") or []:
                if spec.get("symbol"):
                    names.add(spec["symbol"])
            for mod in probe.get("modules") or []:
                names.add(mod.rsplit(".", 1)[-1])
                names.add(mod)
            for path in probe.get("code_paths") or []:
                names.add(Path(path).stem)
    return sorted(n for n in names if n)


_FACTS_MEMO: dict[tuple[Any, ...], dict[str, Any]] = {}


def code_digest() -> str:
    """Hash of tools/roadmap production modules so a mutated auditor busts caches."""
    h = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for p in sorted(root.glob("*.py")):
        if p.name.startswith("test_") or p.name == "conftest.py":
            continue
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def artifact_session_dir() -> Path:
    """Per-PROCESS artifact cache. DO NOT make this stable across invocations.

    It looks like an obvious win -- the audit cache key is
    (head_commit, include_assemble, roadmap, code_digest), which is content-
    addressed and flock-protected, so a shared directory would let one 27-second
    graph build serve every later invocation at the same commit.

    It would also be silently wrong. SourceView.prefetch reads the WORKING TREE
    first and only falls back to HEAD blobs for paths not on disk, so the auditor
    reads receipts/acceptance/*.json off disk -- and the key contains no digest of
    them. Write an acceptance receipt, regenerate at the same commit, and a stable
    cache would serve the graph from BEFORE the receipt existed: the gate would
    stay open and the roadmap would say so, with nothing to indicate why.

    The per-PID directory is what currently prevents that. Making it stable
    requires first putting every worktree input the auditor reads into the key.
    """
    sid = os.environ.get("ROADMAP_ARTIFACT_SESSION")
    if not sid:
        sid = str(os.getpid())
        os.environ["ROADMAP_ARTIFACT_SESSION"] = sid
    d = Path(tempfile.gettempdir()) / f"hawking-roadmap-art-{sid}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _overlay_fingerprint(view: SourceView) -> str:
    if not view.overlay:
        return "-"
    h = hashlib.sha256()
    for path in sorted(view.overlay):
        h.update(path.encode())
        h.update(b"\0")
        h.update(view.overlay[path].encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def index_bin_stamp() -> str:
    """Identity of the index binary. A rebuild must bust the facts cache.

    The dump is a pure function of (commit, watch names, BINARY): rebuild the
    binary with different behaviour at the same commit and a key that omits it
    would serve the old answer. size+mtime_ns detects any rebuild without
    hashing 30 MB on every call.
    """
    b = find_index_bin()
    if b is None:
        return "-"
    try:
        st = b.stat()
    except OSError:
        return "-"
    return f"{st.st_size}:{st.st_mtime_ns}"


def _facts_key(view: SourceView, git_head: bool) -> tuple[Any, ...]:
    commit = head_commit() if git_head else None
    return (commit, _overlay_fingerprint(view), git_head, code_digest(), index_bin_stamp())


#: How many cached dumps to keep. Each is ~9 MB; 254 were materialised on this
#: machine while only 78 were distinct.
FACTS_CACHE_KEEP = 12


def facts_cache_dir() -> Path:
    """STABLE across processes, unlike artifact_session_dir().

    This one is safe to share and the audit cache is not, and the difference is
    what each reads. The facts dump runs the binary with `--git-head --commit
    <sha>`, which reads the git OBJECT STORE: an uncommitted file is verifiably
    absent from its output. So the dump depends only on the commit, the watch
    names (in code_digest) and the binary (in index_bin_stamp) -- all in the key.

    The audit that consumes it also reads acceptance receipts off the WORKING
    TREE and keys on none of them, which is why artifact_session_dir stays
    per-PID. Sharing the expensive deterministic half costs nothing and does not
    touch the unsound half.

    Measured waste from not doing this: 254 dumps materialised, 78 distinct, so
    176 recomputations of a byte-identical 9 MB artifact at ~24 s each.
    """
    d = Path(os.environ.get("HAWKING_FACTS_CACHE")
             or Path(tempfile.gettempdir()) / "hawking-roadmap-facts")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _reap_facts_cache(keep: int = FACTS_CACHE_KEEP) -> None:
    """Keep the newest `keep` dumps. ~9 MB each; unbounded growth is not free."""
    try:
        entries = sorted(facts_cache_dir().glob("facts-*.json"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return
    for stale in entries[keep:]:
        for victim in (stale, stale.with_suffix(".lock"), stale.with_suffix(".tmp")):
            try:
                victim.unlink()
            except OSError:
                pass


def _wrap_dump(dump: dict[str, Any], bin_path: Path, commit: str | None) -> dict[str, Any]:
    raw_files = dump.get("files") or []
    if isinstance(raw_files, dict):
        by_path = {str(k): v for k, v in raw_files.items() if k}
    else:
        by_path = {f["path"]: f for f in raw_files if f.get("path")}
    return {
        "schema": dump["schema"],
        "commit": dump.get("commit") or commit,
        "files": by_path,
        "file_count": len(by_path),
        "bin": str(bin_path),
    }


def _invoke_dump(view: SourceView, *, git_head: bool) -> dict[str, Any]:
    bin_path = find_index_bin()
    if bin_path is None:
        raise FileNotFoundError("hawking-index-query binary not found")
    commit = head_commit() if git_head else None
    cmd = [str(bin_path), "python-facts"]
    if git_head:
        assert commit is not None
        cmd.extend(["--git-head", "--commit", commit, "--repo", str(REPO)])
    for name in catalog_watch_names():
        cmd.extend(["--watch", name])
    overlay = _overlay_ndjson(view)
    cp = subprocess.run(
        cmd,
        input=overlay,
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd)} exited {cp.returncode}: {cp.stderr[-4000:]}"
        )
    dump = json.loads(cp.stdout)
    if dump.get("schema") != SCHEMA:
        raise RuntimeError(
            f"python-facts schema {dump.get('schema')!r} != {SCHEMA}; "
            "r1/r2 JSON surfaces drifted"
        )
    return _wrap_dump(dump, bin_path, commit)


def _merge_head_and_overlay(
    head: dict[str, Any], overlay: dict[str, Any], commit: str
) -> dict[str, Any]:
    files = dict(head["files"])
    files.update(overlay["files"])
    return {
        "schema": head["schema"],
        "commit": commit,
        "files": files,
        "file_count": len(files),
        "bin": head.get("bin") or overlay.get("bin"),
    }


def load_python_facts(view: SourceView, *, git_head: bool = True) -> dict[str, Any]:
    """Run the rust dump against this view. Result is cached on the view,
    in-process, and on disk for this pytest session (xdist workers share it).

    Overlay views reuse a cached HEAD dump and dump only the overlay files
    (equivalent to `--git-head` plus overlay NDJSON: overlay paths win and
    do not inherit the commit stamp).
    """
    cached = getattr(view, "_python_facts", None)
    if cached is not None:
        return cached
    key = _facts_key(view, git_head)
    memo = _FACTS_MEMO.get(key)
    if memo is not None:
        view._python_facts = memo  # type: ignore[attr-defined]
        return memo

    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:24]
    disk = facts_cache_dir() / f"facts-{digest}.json"
    lockp = disk.with_suffix(".lock")
    with open(lockp, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        memo = _FACTS_MEMO.get(key)
        if memo is None and disk.is_file():
            loaded = json.loads(disk.read_text())
            memo = _wrap_dump(loaded, Path(str(loaded.get("bin") or "")), loaded.get("commit"))
            _FACTS_MEMO[key] = memo
        if memo is None:
            if git_head and view.overlay:
                head_view = SourceView()
                head = load_python_facts(head_view, git_head=True)
                overlay = _invoke_dump(view, git_head=False)
                memo = _merge_head_and_overlay(head, overlay, head_commit())
            else:
                memo = _invoke_dump(view, git_head=git_head)
            tmp = disk.with_suffix(".tmp")
            tmp.write_text(json.dumps(memo))
            tmp.replace(disk)
            _FACTS_MEMO[key] = memo
            _reap_facts_cache()
    view._python_facts = memo  # type: ignore[attr-defined]
    return memo


def facts_for(view: SourceView) -> dict[str, Any] | None:
    if resolve_backend() != "index":
        return None
    return load_python_facts(view)


def warmup(view: SourceView) -> dict[str, Any] | None:
    return facts_for(view)


def file_facts(dump: dict[str, Any], rel: str) -> dict[str, Any] | None:
    files = dump.get("files") or {}
    return files.get(rel)


def module_name_of_rel(rel: str) -> str:
    parts = list(Path(rel).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def sibling_rel(importer_rel: str, stem: str) -> str:
    """Repo-relative path of `stem.py` next to `importer_rel`."""
    parent = importer_rel.rsplit("/", 1)[0] if "/" in importer_rel else ""
    if parent in ("", "."):
        return f"{stem}.py"
    return f"{parent}/{stem}.py"


def _known_has(rel: str, known_files: set[str] | frozenset[str] | None) -> bool:
    """Does `rel` exist in the source of truth (HEAD dump, never the worktree)."""
    if known_files is not None:
        return rel in known_files
    from tools.roadmap.gitfs import _git

    return _git("cat-file", "-e", f"HEAD:{rel}", check=False).returncode == 0


def _resolved_from_modules(
    importer_rel: str,
    imp: dict[str, Any],
    *,
    known_files: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Port of tools.roadmap.capability_reachability._resolved_from_modules."""
    bases: list[str] = []
    level = int(imp.get("level") or 0)
    if imp.get("form") == "from" and level > 0:
        importer_mod = module_name_of_rel(importer_rel)
        parts = importer_mod.split(".") if importer_mod else []
        is_init = Path(importer_rel).name == "__init__.py"
        base_parts = parts if is_init else parts[:-1]
        if level > 1:
            cut = level - 1
            base_parts = base_parts[: max(0, len(base_parts) - cut)]
        base = ".".join(base_parts)
        mod = f"{base}.{imp['module']}" if imp.get("module") else base
        if mod:
            bases.append(mod)
    else:
        mod = imp.get("module") or ""
        if mod:
            bases.append(mod)
            if "." not in mod:
                sib = sibling_rel(importer_rel, mod)
                if sib != importer_rel and _known_has(sib, known_files):
                    bases.append(module_name_of_rel(sib))
    return bases


def import_targets_and_binds(
    importer_rel: str,
    imp: dict[str, Any],
    *,
    known_files: set[str] | frozenset[str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Port of CR import-target + bound_names construction for one statement."""
    targets: list[str] = []
    binds: list[tuple[str, str]] = []
    if imp.get("form") == "import":
        for alias in imp.get("names") or []:
            name = alias.get("name") or ""
            if not name:
                continue
            asname = alias.get("asname")
            targets.append(name)
            if "." not in name:
                sib = sibling_rel(importer_rel, name.split(".")[0])
                if sib != importer_rel and _known_has(sib, known_files):
                    targets.append(module_name_of_rel(sib))
            local = asname or name.split(".")[0]
            binds.append((local, name))
    else:
        for mod in _resolved_from_modules(importer_rel, imp, known_files=known_files):
            targets.append(mod)
            for alias in imp.get("names") or []:
                aname = alias.get("name") or ""
                if not aname:
                    continue
                targets.append(f"{mod}.{aname}")
                local = alias.get("asname") or aname
                binds.append((local, f"{mod}.{aname}"))
    return targets, binds


def classify_from_facts(
    ff: dict[str, Any] | None, symbol: str
) -> tuple[str | None, int | None]:
    """Same order as gitfs.classify_symbol: module-level first, then nested."""
    if not ff or not symbol:
        return None, None
    defs = ff.get("definitions") or []
    for d in defs:
        if d.get("name") == symbol and d.get("scope") == "module":
            kind = d.get("kind")
            line = d.get("line")
            return (str(kind) if kind else None, int(line) if line else None)
    for d in defs:
        if d.get("name") == symbol and d.get("kind") in ("function", "class"):
            kind = d.get("kind")
            line = d.get("line")
            return (str(kind) if kind else None, int(line) if line else None)
    return None, None
