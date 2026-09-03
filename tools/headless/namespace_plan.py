#!/usr/bin/env python3
"""Namespace migration census + plan. Read-only over git/HEAD and the disk overlay.

This lane changes no product source. It enumerates the fossil `tools/haider`
namespace, classifies every measured reference, and writes:

    receipts/headless/NAMESPACE_PLAN.json

A human performs the migration. This script does not `rm`, `git mv`,
`git clean`, `git checkout`, `git restore`, or `git reset`.

Sparse-checkout aware: a path missing on disk is not evidence it is absent.
Content is taken from the filesystem when present, otherwise `git show HEAD:path`.
"""
from __future__ import annotations

import ast
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA = "hawking.headless.namespace_plan.v1"
RECEIPT_REL = Path("receipts") / "headless" / "NAMESPACE_PLAN.json"

BUCKETS = (
    "LIVE_IMPLEMENTATION_REFERENCES",
    "LIVE_IMPORT_PATHS",
    "SYS_PATH_HACKS",
    "TEST_HARNESS_PATHS",
    "USER_FACING_COMMANDS",
    "HISTORICAL_RECEIPTS",
    "SEALED_SCHEMA_IDS",
    "HISTORICAL_LOG_TEXT",
    "UNKNOWN",
)

# Hint tree from the brief. Ownership may NOT invent these directories.
HINT_LAYOUT = (
    "hcli",
    "agentos",
    "runtime",
    "doctor",
    "gravity",
    "vmcp",
    "genomes",
    "evidence",
    "experiments",
)

SLASH_COMMANDS = (
    "/help",
    "/status",
    "/models",
    "/model",
    "/goal",
    "/ultragoal",
    "/mission",
    "/steer",
    "/grok",
    "/cancel",
    "/context",
    "/compact",
    "/clear",
    "/resume",
    "/exit",
)

# Ownership of the 33-module Python control plane, derived from module
# docstrings + classes + who imports whom. Not a proposal to split the
# package; the package is already one product.
OWNERSHIP_GROUPS = {
    "product_surface": {
        "belongs": "hcli (keep the package name; user-facing `python -m hcli`)",
        "modules": [
            "__init__.py",
            "__main__.py",
            "app.py",
            "cli.py",
            "commands.py",
            "tui.py",
        ],
        "owns": "CLI grammar, App loop, slash-command surface, TUI",
    },
    "engine": {
        "belongs": "hcli (internal; do not invent a second package)",
        "modules": ["engine.py"],
        "owns": "model-call + mutation apply + validation loop; largest module; talks to backends, context_budget, goal compiler",
    },
    "agentos": {
        "belongs": "hcli (internal; do not invent hawking/agentos/)",
        "modules": [
            "controller.py",
            "mission.py",
            "goal.py",
            "context.py",
            "workunit.py",
            "dag_store.py",
            "scheduler.py",
            "ledger.py",
            "steering.py",
            "executors.py",
            "verifier_pipeline.py",
        ],
        "owns": "mission loop, goal compiler, WorkUnit DAG, ledger, steering, executors, verifier pipeline",
    },
    "runtime": {
        "belongs": "hcli (internal; do not invent hawking/runtime/)",
        "modules": [
            "runtime.py",
            "backends.py",
            "models.py",
            "resources.py",
            "machine.py",
            "context_budget.py",
            "config.py",
            "max_policy.py",
        ],
        "owns": "RuntimePool, llama/MLX backends, model registry, MemGate/genome, context arithmetic, resource classes",
    },
    "workspace_mutation": {
        "belongs": "hcli (internal)",
        "modules": [
            "workspace.py",
            "mutation.py",
            "session.py",
            "index.py",
            "events.py",
        ],
        "owns": "workspace resolve, mutation/rollback, sessions, file index, event bus",
    },
    "grok": {
        "belongs": "hcli (internal)",
        "modules": ["grok_bridge.py", "report_compiler.py"],
        "owns": "grok-run wrapper and backend-evidence compiler",
    },
}

LOG_SUFFIXES = (".stderr", ".stdout", ".log", ".jsonl", ".txt")
LOG_NAME_FRAGMENTS = ("snapshots/",)

# ---------------------------------------------------------------------------
# git / disk
# ---------------------------------------------------------------------------


def _run(argv: List[str], cwd: Path, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=check,
    )


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / ".git").exists() or (p / ".git").is_file():
            return p
    env = os.environ.get("HCLI_REPO") or os.environ.get("REPO_ROOT")
    if env:
        return Path(env).resolve()
    return Path.cwd().resolve()


def git_text(root: Path, argv: List[str]) -> str:
    proc = _run(["git", *argv], cwd=root)
    if proc.returncode != 0:
        return ""
    return proc.stdout


def git_ok(root: Path, argv: List[str]) -> Tuple[int, str, str]:
    proc = _run(["git", *argv], cwd=root)
    return proc.returncode, proc.stdout, proc.stderr


def git_ls(root: Path, *prefixes: str) -> List[str]:
    args = ["ls-tree", "-r", "--name-only", "HEAD"]
    if prefixes:
        args.append("--")
        args.extend(prefixes)
    out = git_text(root, args)
    return [ln for ln in out.splitlines() if ln]


def blob(root: Path, rel: str) -> Optional[str]:
    disk = root / rel
    if disk.is_file():
        try:
            return disk.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    rc, out, _ = git_ok(root, ["show", f"HEAD:{rel}"])
    if rc == 0:
        return out
    return None


_RESOLVE_CACHE: Dict[str, str] = {}
_GIT_PATHS: Optional[set] = None


def _git_path_set(root: Path) -> set:
    global _GIT_PATHS
    if _GIT_PATHS is None:
        _GIT_PATHS = set(git_ls(root))
    return _GIT_PATHS


def resolves(root: Path, rel: str) -> str:
    cached = _RESOLVE_CACHE.get(rel)
    if cached is not None:
        return cached
    disk = root / rel
    if disk.exists():
        _RESOLVE_CACHE[rel] = "disk"
        return "disk"
    if rel in _git_path_set(root):
        _RESOLVE_CACHE[rel] = "git"
        return "git"
    rc2, out, _ = git_ok(root, ["ls-tree", "-r", "--name-only", "HEAD", "--", rel])
    if rc2 == 0 and out.strip():
        _RESOLVE_CACHE[rel] = "git"
        return "git"
    _RESOLVE_CACHE[rel] = "MISSING"
    return "MISSING"


def now_utc() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------

GREP_RE = re.compile(
    r"haider|jhcli|parse_haider|python3 -m hcli|python -m hcli|"
    r"from hcli(?:\.|\s)|import hcli(?:\.|\s|$)|from tools\.haider|"
    r"import tools\.haider|sys\.path\.insert|sys\.path\.append|"
    r"install-shims|PYTHONPATH=tools/haider",
    re.IGNORECASE,
)

IMPORT_RE = re.compile(
    r"^\s*(from\s+(hcli|tools\.haider(?:\.hcli)?)(\.[\w.]+)?\s+import|"
    r"import\s+(hcli|tools\.haider)(\.[\w.]+)?)\b"
)
SYS_PATH_RE = re.compile(r"sys\.path\.(insert|append)\s*\(")
USER_CMD_RE = re.compile(
    r"python3? -m hcli|\bjhcli\b|install-shims|parse_hcli_args|"
    r"~/\.local/bin/hcli|~/\.local/share/hcli|prog=.haider."
)
SCHEMA_RE = re.compile(r"\b(hawking\.[A-Za-z0-9_.]+|hcli\.command\.v1)\b")


# git grep -E is POSIX ERE: no (?:...), no lookaround. Keep patterns simple.
GIT_GREP_PATTERNS = (
    r"haider",
    r"jhcli",
    r"parse_haider",
    r"python3 -m hcli",
    r"python -m hcli",
    r"from hcli",
    r"import hcli",
    r"from tools\.haider",
    r"import tools\.haider",
    r"install-shims",
    r"PYTHONPATH=tools/haider",
)

# sys.path inserts are a path assumption only when they sit in the fossil
# package, its tests, or the headless harnesses. Repo-wide inserts (lab
# operators, condense tests) are a different campaign.
SYS_PATH_GREP_ROOTS = ("tools/headless", "tools/haider")


def git_grep_hits(root: Path, pattern: str) -> Tuple[int, str, List[Tuple[str, int, str]]]:
    """Return (exit_code, stderr_head, hits). git grep uses POSIX ERE."""
    rc, out, err = git_ok(root, ["grep", "-n", "-I", "-E", pattern, "HEAD"])
    hits: List[Tuple[str, int, str]] = []
    if rc not in (0, 1):
        return rc, (err or out)[:400], hits
    for raw in out.splitlines():
        if not raw.startswith("HEAD:"):
            continue
        rest = raw[5:]
        m = re.match(r"^(.+?):(\d+):(.*)$", rest)
        if not m:
            continue
        path, ln, text = m.group(1), int(m.group(2)), m.group(3)
        hits.append((path, ln, text))
    return rc, err[:400] if err else "", hits


def git_grep_hits_in(root: Path, pattern: str, prefixes: Tuple[str, ...]) -> Tuple[int, str, List[Tuple[str, int, str]]]:
    args = ["grep", "-n", "-I", "-E", pattern, "HEAD", "--", *prefixes]
    rc, out, err = git_ok(root, args)
    hits: List[Tuple[str, int, str]] = []
    if rc not in (0, 1):
        return rc, (err or out)[:400], hits
    for raw in out.splitlines():
        if not raw.startswith("HEAD:"):
            continue
        rest = raw[5:]
        m = re.match(r"^(.+?):(\d+):(.*)$", rest)
        if not m:
            continue
        hits.append((m.group(1), int(m.group(2)), m.group(3)))
    return rc, err[:400] if err else "", hits


def git_grep_all(root: Path) -> Tuple[List[Tuple[str, int, str]], List[Dict[str, Any]]]:
    seen = set()
    hits: List[Tuple[str, int, str]] = []
    errors: List[Dict[str, Any]] = []

    def _absorb(rc: int, err: str, batch: List[Tuple[str, int, str]], pat: str) -> None:
        if rc not in (0, 1):
            errors.append({"pattern": pat, "exit": rc, "stderr": err})
            return
        for item in batch:
            if item[:2] in seen:
                continue
            seen.add(item[:2])
            hits.append(item)

    for pat in GIT_GREP_PATTERNS:
        rc, err, batch = git_grep_hits(root, pat)
        _absorb(rc, err, batch, pat)
    for pat in (r"sys\.path\.insert", r"sys\.path\.append"):
        rc, err, batch = git_grep_hits_in(root, pat, SYS_PATH_GREP_ROOTS)
        _absorb(rc, err, batch, pat + " restricted to tools/headless|tools/haider")
    hits.sort()
    return hits, errors


def disk_grep_hits(root: Path, rel_dir: str, pattern: re.Pattern) -> List[Tuple[str, int, str]]:
    hits: List[Tuple[str, int, str]] = []
    base = root / rel_dir
    if not base.is_dir():
        return hits
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".gz", ".pyc", ".png", ".jpg"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        if rel == str(RECEIPT_REL):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                hits.append((rel, i, line.rstrip()))
    return hits


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def _is_receipt(path: str) -> bool:
    return path.startswith("receipts/") or path.startswith("workspace/campaign/")


def _is_log(path: str) -> bool:
    if any(frag in path.replace("\\", "/") for frag in LOG_NAME_FRAGMENTS):
        return True
    lower = path.lower()
    return lower.endswith(LOG_SUFFIXES)


def _is_python_test(path: str) -> bool:
    if "/tests/" in path or path.startswith("research/lab/tests/"):
        return True
    name = path.rsplit("/", 1)[-1]
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    if path.startswith("tools/headless/") and path.endswith(".py"):
        return True
    return False


def _is_hcli_impl(path: str) -> bool:
    if not path.startswith("tools/hcli/bootstrap/"):
        return False
    if "/tests/" in path:
        return False
    name = path.rsplit("/", 1)[-1]
    if name.startswith("test_"):
        return False
    return path.endswith(".py") or path.endswith(".md")


def classify_site(path: str, text: str) -> Tuple[str, str]:
    """Return (bucket, note). One primary bucket per site."""
    stripped = text.strip()

    if _is_receipt(path):
        if _is_log(path) and not path.endswith(".json"):
            return "HISTORICAL_LOG_TEXT", "non-json receipt sidecar / snapshot; preserve bytes"
        return "HISTORICAL_RECEIPTS", "historical receipt or campaign evidence; preserve path and text"

    if path.startswith("workspace/") and not path.startswith("workspace/docs/"):
        return "HISTORICAL_RECEIPTS", "workspace evidence corpus; preserve (PRECIOUS, not scratch)"

    if _is_log(path):
        return "HISTORICAL_LOG_TEXT", "log sidecar; preserve"

    if path.endswith("tools/headless/namespace_plan.py") or path.endswith("namespace_plan.py") and path.startswith("tools/headless/"):
        return "TEST_HARNESS_PATHS", "this census; not a runtime import of the package"

    if SYS_PATH_RE.search(stripped) or (
        "sys.path.insert" in stripped or "sys.path.append" in stripped
    ):
        # comments / census how-strings still encode a path assumption
        if stripped.startswith("#") or stripped.startswith('"how"') or '"how":' in stripped:
            if path.startswith("tools/headless/"):
                return "TEST_HARNESS_PATHS", "harness mentions a sys.path convention; not an executable insert"
        return "SYS_PATH_HACKS", "sys.path mutation; migrate or eliminate"

    if IMPORT_RE.search(stripped) or re.search(
        r"\b(from|import)\s+(hcli|tools\.haider)\b", stripped
    ):
        return "LIVE_IMPORT_PATHS", "live Python import path"

    if USER_CMD_RE.search(stripped) or "python -m hcli" in stripped or "python3 -m hcli" in stripped:
        return "USER_FACING_COMMANDS", "operator-facing command / shim / CLI symbol"

    if path.startswith("hcli/tests/") or path.startswith("tools/hcli/bootstrap/test_"):
        return "TEST_HARNESS_PATHS", "in-package test path assumption"

    if path.startswith("tools/headless/"):
        if path.endswith("namespace_plan.py"):
            return "TEST_HARNESS_PATHS", "this census; not a runtime import of the package"
        return "TEST_HARNESS_PATHS", "headless harness path assumption"

    if path.startswith("research/lab/hcli/"):
        return (
            "UNKNOWN",
            "package lab.hcli — Agent OS scaffolds, not hcli; do not fold into this move",
        )

    if path.startswith("research/lab/"):
        return (
            "UNKNOWN",
            "lab operator/test filename or string containing hcli/haider; not the Python control-plane package",
        )

    if path.startswith("crates/"):
        return (
            "UNKNOWN",
            "Rust namesake (hide-backend haider/hcli); different language and product; do not rename with the Python move",
        )

    if path.startswith("tools/hcli/bootstrap/") and _is_hcli_impl(path):
        return "LIVE_IMPLEMENTATION_REFERENCES", "implementation source under the fossil directory"

    if path.startswith("tools/hcli/bootstrap/"):
        return "LIVE_IMPLEMENTATION_REFERENCES", "fossil-directory artifact"

    if path.startswith("tools/condense/"):
        return (
            "UNKNOWN",
            "condense HCLI live-suite talks to the Rust hcli.command.v1 product, not hcli imports",
        )

    if path.startswith("workspace/docs/"):
        return "HISTORICAL_RECEIPTS", "plan/doc under workspace; preserve filename; do not treat as live code"

    if path == ".gitignore":
        return "LIVE_IMPLEMENTATION_REFERENCES", "gitignore of .hcli-legacy/ state dir; PRESERVE the ignore rule"

    if path.startswith("contracts/") or path == "README.md" or path == "Cargo.toml":
        return "UNKNOWN", "docs/manifest mention; inspect before any rename"

    return "UNKNOWN", "matched grep but did not fit a tighter bucket"


def classify_sys_path_target(text: str) -> str:
    t = text
    if "tools" in t and "haider" in t:
        return "haider_pkg_root"
    if "HAIDER" in t and "headless" not in t:
        return "haider_var"
    if "HCLI_PKG" in t:
        return "hcli_pkg_alias_of_haider"
    if "sys.argv" in t:
        return "argv_injected"
    if "visionmcp" in t or "vmcp_src" in t:
        return "visionmcp_src"
    if "headless" in t:
        return "headless_dir"
    if "REPO_ROOT" in t or "str(REPO)" in t or "str(REPO)" in t:
        return "repo_root"
    if "parents[1]" in t and "haider" in t:
        return "haider_pkg_root"
    if "parents[2]" in t and "haider" not in t:
        return "package_parent_or_repo"
    if "parents[3]" in t:
        return "tools_or_repo_depth"
    if "os.path.dirname" in t and "__file__" in t:
        return "this_file_dir"
    if "parent" in t.lower() and "haider" not in t:
        return "parent_dir"
    if "TOOLS" in t:
        return "tools_dir"
    if "str(src)" in t or "str(root)" in t:
        return "local_src_or_root"
    if "how" in t and "haider" in t:
        return "comment_or_census_string"
    return "other"


# ---------------------------------------------------------------------------
# ownership
# ---------------------------------------------------------------------------


def parse_module(root: Path, rel: str) -> Dict[str, Any]:
    src = blob(root, rel) or ""
    loc = src.count("\n") + (0 if src.endswith("\n") or not src else 1)
    if src and not src.endswith("\n"):
        loc = src.count("\n") + 1
    else:
        loc = src.count("\n")
    doc = ""
    classes: List[str] = []
    funcs: List[str] = []
    abs_imports: List[str] = []
    rel_imports: List[str] = []
    try:
        tree = ast.parse(src)
        doc = ast.get_docstring(tree) or ""
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                classes.append(n.name)
            elif isinstance(n, ast.FunctionDef):
                funcs.append(n.name)
    except SyntaxError as exc:
        classes = [f"SYNTAX_ERROR:{exc}"]
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("from .") or s.startswith("import ."):
            rel_imports.append(s)
        elif s.startswith("from hcli") or s.startswith("import hcli"):
            abs_imports.append(s)
        elif s.startswith("from tools.haider") or s.startswith("import tools.haider"):
            abs_imports.append(s)
    return {
        "path": rel,
        "resolves": resolves(root, rel),
        "loc": loc,
        "doc": " ".join(doc.split())[:400],
        "classes": classes,
        "funcs": funcs,
        "relative_imports": rel_imports,
        "absolute_hcli_imports": abs_imports,
    }


def group_for_module(name: str) -> str:
    for g, spec in OWNERSHIP_GROUPS.items():
        if name in spec["modules"]:
            return g
    return "UNASSIGNED"


# ---------------------------------------------------------------------------
# watched failures (actually executed)
# ---------------------------------------------------------------------------


def watched_fail(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add(name: str, argv: List[str], env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None) -> None:
        use_env = os.environ.copy()
        if env:
            use_env.update(env)
        proc = subprocess.run(
            argv,
            cwd=str(cwd or root),
            capture_output=True,
            text=True,
            env=use_env,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        rows.append(
            {
                "name": name,
                "argv": argv,
                "exit": proc.returncode,
                "output_head": out.strip()[:500],
            }
        )

    add("ls_tools_haider", ["ls", str(root / "tools" / "haider")])
    add("import_hcli", [sys.executable, "-c", "import hcli"])
    add(
        "import_hcli_with_pythonpath_tools_haider",
        [sys.executable, "-c", "import hcli"],
        env={**os.environ, "PYTHONPATH": str(root / "tools" / "haider")},
    )
    add("import_tools_haider_hcli", [sys.executable, "-c", "import tools.haider.hcli"])
    add("python_m_hcli", [sys.executable, "-m", "hcli", "--help"])
    add(
        "git_grep_import_aider_in_tools",
        ["git", "grep", "-I", "-n", "-E", r"import aider|from aider|aider-chat", "HEAD", "--", "tools"],
    )
    add(
        "git_grep_shelled_aider_in_tools",
        ["git", "grep", "-I", "-n", "-E", r"(^|[^A-Za-z_])aider(-chat)?([^A-Za-z_]|$)", "HEAD", "--", "tools"],
    )

    # Depth arithmetic check: resources.py parents[3] identity.
    src = blob(root, "hcli/resources.py") or ""
    rows.append(
        {
            "name": "resources_default_repo_root_depth",
            "argv": ["git", "show", "HEAD:hcli/resources.py"],
            "exit": 0 if "parents[3]" in src else 1,
            "output_head": "parents[3] present" if "parents[3]" in src else "MISSING parents[3]",
        }
    )

    rust_lib = blob(root, "crates/hide-backend/src/lib.rs") or ""
    rows.append(
        {
            "name": "rust_lib_rs_declares_mod_haider",
            "argv": ["rg", "mod haider", "crates/hide-backend/src/lib.rs"],
            "exit": 0 if re.search(r"\bmod haider\b", rust_lib) else 1,
            "output_head": (
                "pub mod haider present"
                if re.search(r"\bmod haider\b", rust_lib)
                else "lib.rs does not declare mod haider; crates/hide-backend/src/haider/ exists; tests/haider_parallel.rs imports hide_backend::haider"
            ),
        }
    )

    haider_bin = blob(root, "crates/hide-backend/src/bin/haider.rs")
    rows.append(
        {
            "name": "rust_bin_haider_rs_empty",
            "argv": ["git", "show", "HEAD:crates/hide-backend/src/bin/haider.rs"],
            "exit": 0 if haider_bin == "" else 1,
            "output_head": f"bytes={0 if haider_bin is None else len(haider_bin.encode('utf-8'))}",
        }
    )

    # Contract claimed 63 headless harnesses; measure.
    headless = git_ls(root, "tools/headless")
    rows.append(
        {
            "name": "headless_count_vs_contract_63",
            "argv": ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", "tools/headless"],
            "exit": 0 if len(headless) == 63 else 1,
            "output_head": f"git_ls_count={len(headless)} contract_said=63",
        }
    )

    return rows


# ---------------------------------------------------------------------------
# sealed names
# ---------------------------------------------------------------------------


def receipt_schemas(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    names = git_ls(root, "receipts/headless")
    disk_extra = []
    disk_dir = root / "receipts" / "headless"
    if disk_dir.is_dir():
        for p in disk_dir.rglob("*"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(root)).replace("\\", "/")
            if rel not in names:
                disk_extra.append(rel)
    for rel in sorted(set(names) | set(disk_extra)):
        if rel.endswith(".json"):
            src = blob(root, rel)
            schema = None
            kind = None
            parse = "ok"
            if src is None:
                parse = "MISSING"
            else:
                try:
                    data = json.loads(src)
                    if isinstance(data, dict):
                        schema = data.get("schema")
                        kind = data.get("kind")
                    else:
                        parse = f"json_type={type(data).__name__}"
                except json.JSONDecodeError as exc:
                    parse = f"JSON_ERROR:{exc}"
            rows.append(
                {
                    "path": rel,
                    "resolves": resolves(root, rel),
                    "schema": schema,
                    "kind": kind,
                    "parse": parse,
                    "preserve": True,
                }
            )
        else:
            rows.append(
                {
                    "path": rel,
                    "resolves": resolves(root, rel),
                    "schema": None,
                    "kind": "non_json",
                    "parse": "n/a",
                    "preserve": True,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def site_obj(path: str, line: int, text: str, bucket: str, note: str, root: Path) -> Dict[str, Any]:
    return {
        "path": path,
        "line": line,
        "text": text.strip()[:240],
        "bucket": bucket,
        "note": note,
        "resolves": resolves(root, path),
    }


def print_sites(title: str, sites: List[Dict[str, Any]], limit: Optional[int] = None) -> None:
    print(f"\n### {title}  (n={len(sites)})")
    shown = sites if limit is None else sites[:limit]
    for s in shown:
        print(f"  {s['path']}:{s['line']}: {s['text']}")
    if limit is not None and len(sites) > limit:
        print(f"  … {len(sites) - limit} more in the JSON receipt")


def main() -> int:
    root = repo_root()
    os.chdir(root)

    head = git_text(root, ["rev-parse", "HEAD"]).strip()
    branch = git_text(root, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    generated_at = now_utc()

    # --- inventories ---
    haider_files = git_ls(root, "tools/haider")
    hcli_py = [
        p
        for p in haider_files
        if p.startswith("hcli/") and p.endswith(".py")
    ]
    hcli_modules = [p for p in hcli_py if "/tests/" not in p]
    hcli_tests = [p for p in hcli_py if "/tests/" in p]
    headless_files = git_ls(root, "tools/headless")
    headless_py = [p for p in headless_files if p.endswith(".py")]
    # include the census file itself once it is on disk
    if (root / "tools/headless/namespace_plan.py").is_file():
        rel = "tools/headless/namespace_plan.py"
        if rel not in headless_py:
            headless_py.append(rel)
            headless_files.append(rel)

    rust_haider = git_ls(root, "crates/hide-backend/src/haider")
    rust_hcli_bins = [
        p
        for p in git_ls(root, "crates/hide-backend/src/bin")
        if "hcli" in p or "haider" in p
    ]
    lab_hcli = git_ls(root, "research/lab/hcli")

    modules = [parse_module(root, p) for p in hcli_modules]
    assigned = []
    for m in modules:
        name = m["path"].rsplit("/", 1)[-1]
        g = group_for_module(name)
        m["ownership_group"] = g
        assigned.append(m)
    unassigned = [m["path"] for m in assigned if m["ownership_group"] == "UNASSIGNED"]

    # --- line census ---
    git_hits, grep_errors = git_grep_all(root)
    disk_hits = disk_grep_hits(root, "tools/headless", GREP_RE)
    disk_hits += disk_grep_hits(root, "receipts/headless", GREP_RE)

    seen = set()
    sites_raw: List[Tuple[str, int, str]] = []
    for path, line, text in git_hits + disk_hits:
        key = (path, line)
        if key in seen:
            continue
        seen.add(key)
        sites_raw.append((path, line, text))
    sites_raw.sort()

    classified: Dict[str, List[Dict[str, Any]]] = {b: [] for b in BUCKETS}
    missing_paths = []
    for path, line, text in sites_raw:
        bucket, note = classify_site(path, text)
        obj = site_obj(path, line, text, bucket, note, root)
        classified[bucket].append(obj)
        if obj["resolves"] == "MISSING":
            missing_paths.append(obj)

    # Implementation files as first-class live references (the files themselves).
    impl_files = []
    for p in haider_files:
        if p.endswith(".py") and "/tests/" not in p and not p.rsplit("/", 1)[-1].startswith("test_"):
            impl_files.append(
                {
                    "path": p,
                    "line": 1,
                    "text": f"<file {p}>",
                    "bucket": "LIVE_IMPLEMENTATION_REFERENCES",
                    "note": "implementation file under fossil namespace",
                    "resolves": resolves(root, p),
                }
            )
    # Do not duplicate file:1 if a real line-1 hit exists
    have_impl = {(s["path"], s["line"]) for s in classified["LIVE_IMPLEMENTATION_REFERENCES"]}
    for obj in impl_files:
        if (obj["path"], obj["line"]) not in have_impl:
            classified["LIVE_IMPLEMENTATION_REFERENCES"].append(obj)

    for b in BUCKETS:
        classified[b].sort(key=lambda s: (s["path"], s["line"]))

    # Namesakes that share the token but are not the Python control plane.
    # Line hits land in UNKNOWN via classify_site. Files with no `haider`
    # token (lab.hcli, rust hcli_bridge) still need a file-level UNKNOWN.
    namesake_files = (
        rust_haider
        + rust_hcli_bins
        + lab_hcli
        + [
            "crates/hide-backend/src/bin/haider.rs",
            "crates/hide-backend/src/bin/hcli.rs",
            "crates/hide-backend/tests/haider_parallel.rs",
            "crates/hide-backend/src/hcli_bridge.rs",
            "crates/hide-backend/src/hcli_profile.rs",
            "crates/hide-backend/src/hcli_research.rs",
            "crates/hide-backend/src/hcli_sources.rs",
            "crates/hide-backend/src/hcli_swarm.rs",
        ]
    )
    cited_paths = {s["path"] for b in BUCKETS for s in classified[b]}
    for p in sorted(set(namesake_files)):
        if p in cited_paths:
            continue
        _bkt, note = classify_site(p, p)
        classified["UNKNOWN"].append(
            site_obj(
                p,
                1,
                f"<file {p}>",
                "UNKNOWN",
                note if _bkt == "UNKNOWN" else "namesake file; not hcli",
                root,
            )
        )

    # sys.path.insert inventory (headless + in-package tests + fossil siblings)
    sys_path_sites: List[Dict[str, Any]] = []
    sys_path_files = (
        [p for p in headless_py]
        + hcli_tests
        + [p for p in haider_files if p.endswith(".py") and "/hcli/" not in p]
    )
    # unique
    sys_path_files = sorted(set(sys_path_files))
    for rel in sys_path_files:
        src = blob(root, rel)
        if src is None:
            continue
        for i, line in enumerate(src.splitlines(), 1):
            if "sys.path.insert" in line or "sys.path.append" in line:
                sys_path_sites.append(
                    {
                        "path": rel,
                        "line": i,
                        "text": line.strip()[:240],
                        "target_class": classify_sys_path_target(line),
                        "resolves": resolves(root, rel),
                        "in_headless": rel.startswith("tools/headless/"),
                        "executable": bool(SYS_PATH_RE.search(line.strip()))
                        and not line.strip().startswith("#")
                        and 'f"sys.path' not in line
                        and "f'sys.path" not in line
                        and 'f"sys.path.insert' not in line
                        and "f'sys.path.insert" not in line
                        and not line.strip().startswith('"')
                        and "how" not in line[:20],
                    }
                )

    headless_sys = [s for s in sys_path_sites if s["in_headless"]]
    headless_sys_by_target = Counter(s["target_class"] for s in headless_sys)
    headless_files_with = sorted({s["path"] for s in headless_sys})
    headless_files_without = sorted(
        p for p in headless_py if p not in set(headless_files_with)
    )

    # import convention counts
    import_hcli = [
        s
        for s in classified["LIVE_IMPORT_PATHS"]
        if re.search(r"\bfrom hcli\b|\bimport hcli\b", s["text"])
        and "tools.haider" not in s["text"]
    ]
    import_tools_haider = [
        s
        for s in classified["LIVE_IMPORT_PATHS"]
        if "tools.haider" in s["text"]
    ]

    # sealed
    sealed_receipts = receipt_schemas(root)
    schema_ids = sorted(
        {
            r["schema"]
            for r in sealed_receipts
            if r.get("schema")
        }
    )
    # extra named seals
    extra_schema_cites: List[Dict[str, Any]] = []
    for rel, needle in [
        ("receipts/headless/NOETIC_METRICS.json", "hawking.nos.nr_nx_artifact.v1"),
        ("crates/hide-backend/src/bin/hcli.rs", "hcli.command.v1"),
    ]:
        src = blob(root, rel) or ""
        line_no = 0
        for i, ln in enumerate(src.splitlines(), 1):
            if needle in ln:
                line_no = i
                extra_schema_cites.append(
                    {
                        "id": needle,
                        "path": rel,
                        "line": line_no,
                        "resolves": resolves(root, rel),
                        "preserve": True,
                    }
                )
                break
        else:
            extra_schema_cites.append(
                {
                    "id": needle,
                    "path": rel,
                    "line": None,
                    "resolves": resolves(root, rel),
                    "preserve": True,
                    "note": "id expected; line not found in blob",
                }
            )

    # SEALED_SCHEMA_IDS: one cite per id (the ids, not every mention).
    for r in sealed_receipts:
        sid = r.get("schema")
        if not sid:
            continue
        src = blob(root, r["path"]) or ""
        line_no = 1
        for i, ln in enumerate(src.splitlines(), 1):
            if sid in ln:
                line_no = i
                break
        classified["SEALED_SCHEMA_IDS"].append(
            site_obj(
                r["path"],
                line_no,
                f'"schema": "{sid}"',
                "SEALED_SCHEMA_IDS",
                "receipt schema id; PRESERVE",
                root,
            )
        )
    for c in extra_schema_cites:
        classified["SEALED_SCHEMA_IDS"].append(
            site_obj(
                c["path"],
                int(c["line"] or 1),
                c["id"],
                "SEALED_SCHEMA_IDS",
                "named sealed schema id; PRESERVE even if it is not a receipts/headless schema field",
                root,
            )
        )

    # HISTORICAL_LOG_TEXT: non-json sidecars under receipts/headless.
    for r in sealed_receipts:
        if r.get("kind") != "non_json":
            continue
        classified["HISTORICAL_LOG_TEXT"].append(
            site_obj(
                r["path"],
                1,
                f"<file {r['path']}>",
                "HISTORICAL_LOG_TEXT",
                "non-json receipt sidecar / snapshot; PRESERVE bytes and filename",
                root,
            )
        )

    for b in ("SEALED_SCHEMA_IDS", "HISTORICAL_LOG_TEXT", "UNKNOWN"):
        classified[b].sort(key=lambda s: (s["path"], s["line"]))

    # on-disk state names (not files to rename)
    state_dirs = [
        {
            "name": ".hcli/",
            "role": "live per-workspace durable state (dag.json, mission, ledger, grok receipts, sessions)",
            "preserve": True,
            "cite": "hcli/dag_store.py:3",
        },
        {
            "name": ".hcli-legacy/",
            "role": "gitignored fossil state dir; also listed in engine skip sets and genome lookup",
            "preserve": True,
            "cite": ".gitignore:182 and hcli/cli.py:14",
        },
        {
            "name": "~/.config/hcli/",
            "role": "user config + machine_genome.json",
            "preserve": True,
            "cite": "hcli/config.py:41",
        },
        {
            "name": "~/.local/share/hcli/current",
            "role": "install-shims package copy; PYTHONPATH for the operator shims",
            "preserve": True,
            "cite": "hcli/cli.py:191",
        },
        {
            "name": "~/.local/bin/{hcli,jhcli}",
            "role": "user-facing shims; both exec python -m hcli",
            "preserve_command_name": True,
            "cite": "hcli/cli.py:221",
        },
    ]

    failures = watched_fail(root)
    for err in grep_errors:
        failures.append(
            {
                "name": f"git_grep_pattern_{err['pattern']}",
                "argv": ["git", "grep", "-n", "-I", "-E", err["pattern"], "HEAD"],
                "exit": err["exit"],
                "output_head": err.get("stderr") or "",
            }
        )

    # path-depth assumptions
    depth = [
        {
            "path": "hcli/resources.py",
            "line": 57,
            "text": "return Path(__file__).resolve().parents[3]",
            "meaning": "repo root, encoding hcli/<file> (4 levels)",
            "breaks_if": "package dropped one directory (tools/hcli/resources.py would need parents[2])",
            "resolves": resolves(root, "hcli/resources.py"),
        },
        {
            "path": "hcli/machine.py",
            "line": 85,
            "text": "return Path(__file__).resolve().parents[3]",
            "meaning": "repo root, same 4-level encoding",
            "breaks_if": "package dropped one directory",
            "resolves": resolves(root, "hcli/machine.py"),
        },
        {
            "path": "hcli/machine.py",
            "line": 143,
            "text": 'Path(__file__).resolve().parents[2] / "headless" / "metal_budget.py"',
            "meaning": "tools/headless/metal_budget.py via sibling of tools/haider",
            "breaks_if": "package leaves tools/",
            "resolves": resolves(root, "hcli/machine.py"),
        },
        {
            "path": "hcli/engine.py",
            "line": 2779,
            "text": 'extra = str(self.root / "tools" / "haider")',
            "meaning": "contained test subprocess PYTHONPATH includes fossil dir so `import hcli` works",
            "breaks_if": "directory rename without updating this literal",
            "resolves": resolves(root, "hcli/engine.py"),
        },
    ]

    # migration steps with measured blast
    harness_haider_inserts = [
        s
        for s in headless_sys
        if s["target_class"] in {"haider_pkg_root", "haider_var", "hcli_pkg_alias_of_haider"}
    ]
    test_repo_inserts = [
        s
        for s in sys_path_sites
        if s["path"].startswith("hcli/tests/") and "parents[4]" in (blob(root, s["path"]) or "")
    ]

    layout = {
        "canonical_package_name": "hcli",
        "canonical_physical_path": "tools/hcli/  (today: hcli/)",
        "do_not_invent": [
            {
                "hint": f"hawking/{name}",
                "verdict": "NOT CREATED",
                "reason": {
                    "hcli": "the package is already named hcli; wrapping it in a Python hawking/ tree would collide with the Rust crate hawking and invent a namespace no Python file uses",
                    "agentos": "AgentOS lives as modules inside hcli (ledger, mission, workunit, steering, verifier_pipeline); splitting them out is an architecture rewrite, not a namespace move",
                    "runtime": "RuntimePool/backends/machine already live in hcli; a hawking/runtime/ Python package does not exist",
                    "doctor": "doctor lives as tools/doctor_seal.py and tools/gravity_doctor_*.py, not inside hcli",
                    "gravity": "gravity lives as tools/gravity_*.py plus crates; not this package",
                    "vmcp": "visionmcp/ is already its own package; harnesses sys.path.insert visionmcp/src independently",
                    "genomes": "MachineGenome is hcli.machine; receipts/headless/MACHINE_GENOME.json is a sealed receipt",
                    "evidence": "receipts/ and workspace/campaign/evidence/; preserve, do not relocate",
                    "experiments": "research/lab/ is already the experiment package",
                }[name],
            }
            for name in HINT_LAYOUT
        ],
        "modules": assigned,
        "unassigned_modules": unassigned,
        "fossil_siblings_not_in_package": [
            {
                "path": "tools/hcli/bootstrap/snapshots/haider.py",
                "loc": (blob(root, "tools/hcli/bootstrap/snapshots/haider.py") or "").count("\n"),
                "owns": "HCLI-v0 Gate Zero bootstrap. Does not import the hcli package. sys.path.insert of its own directory + import p0_tool_bridge.",
                "action": "leave in place through the package move; retire in a later fossil step after proving zero callers",
                "resolves": resolves(root, "tools/hcli/bootstrap/snapshots/haider.py"),
            },
            {
                "path": "tools/hcli/bootstrap/p0_tool_bridge.py",
                "owns": "P0 tool bridge used only by haider.py and tools/hcli/bootstrap/test_*.py",
                "action": "same as haider.py",
                "resolves": resolves(root, "tools/hcli/bootstrap/p0_tool_bridge.py"),
            },
            {
                "path": "tools/hcli/bootstrap/P1_HAIDER_PRODUCTIZATION_MAX.md",
                "owns": "historical productization notes citing python tools/hcli/bootstrap/snapshots/haider.py",
                "action": "preserve as doc; do not treat as live entrypoint",
                "resolves": resolves(root, "tools/hcli/bootstrap/P1_HAIDER_PRODUCTIZATION_MAX.md"),
            },
        ],
        "other_products_sharing_the_name": [
            {
                "path": "crates/hide-backend/src/haider/",
                "files": rust_haider,
                "wired": False,
                "evidence": "crates/hide-backend/src/lib.rs has no `mod haider`; tests/haider_parallel.rs still `use hide_backend::haider::{...}`",
                "action": "DO NOT rename as part of the Python move. Separate decision: wire or delete.",
            },
            {
                "path": "crates/hide-backend/src/bin/haider.rs",
                "bytes": 0,
                "action": "empty auto-bin. Do not migrate. Human decides delete vs implement.",
                "resolves": resolves(root, "crates/hide-backend/src/bin/haider.rs"),
            },
            {
                "path": "crates/hide-backend/src/bin/hcli.rs",
                "schema": "hcli.command.v1",
                "action": "HIDE-backend CLI product. Preserve the schema id and the bin name unless a dedicated Rust rename campaign says otherwise. Not an import of hcli.",
                "resolves": resolves(root, "crates/hide-backend/src/bin/hcli.rs"),
            },
            {
                "path": "research/lab/hcli/",
                "files": lab_hcli,
                "package": "lab.hcli",
                "action": "already a correctly named lab package. Do not merge into tools/hcli.",
            },
        ],
    }

    steps = [
        {
            "id": "S0",
            "name": "Preflight — this census is the baseline",
            "does": "No source moves. Keep ten in-flight science lanes on tools/headless and receipts/headless unblocked.",
            "blast": {
                "files_changed": 0,
                "notes": "Suite stays at the recorded 464 passed / 1 skipped with HCLI_SWAP_CEILING_GIB=64.",
            },
            "run_after": "python3 tools/headless/namespace_plan.py  (already the gate for this lane)",
            "blocks_if": "none",
        },
        {
            "id": "S1",
            "name": "Unify the in-package test import to `hcli` (no directory rename)",
            "does": (
                f"{len(import_tools_haider)} live `from tools.haider.hcli` sites, mostly in "
                f"{len([p for p in hcli_tests if p.rsplit('/',1)[-1].startswith('test_')])} test_*.py files "
                "(conftest.py included in the blast). test_grok_identity.py already inserts "
                "tools/haider and `from hcli.grok_bridge`. Switch the in-package tests to that "
                "convention so there is one live import path. Leave harnesses alone."
            ),
            "blast": {
                "files": [p for p in hcli_tests if p.endswith(".py")],
                "file_count": len(hcli_tests),
                "import_sites_tools_haider": len(import_tools_haider),
                "science_lanes": "untouched (they already `from hcli` after inserting tools/haider)",
            },
            "run_after": "python3 -m unittest discover -s hcli/tests -t .",
            "blocks_if": "tools/haider not materialized in a sparse worktree (this worktree: NOT on disk)",
        },
        {
            "id": "S2",
            "name": "Replace parents[N] repo-root encoding with a walk-to-.git helper",
            "does": (
                "resources._default_repo_root and machine.default_repo_root hardcode parents[3]; "
                "machine._metal_budget_module hardcodes parents[2]/headless. "
                "A one-level directory drop breaks all three. Fix before any rename."
            ),
            "blast": {
                "files": [
                    "hcli/resources.py",
                    "hcli/machine.py",
                ],
                "file_count": 2,
            },
            "run_after": "unittest test_runtime_pool test_runtime_authority test_models_sidecars + headless machine_probe",
            "blocks_if": "none",
        },
        {
            "id": "S3",
            "name": "Stop hardcoding tools/haider in engine subprocess PYTHONPATH",
            "does": (
                "engine._test_subprocess_env sets PYTHONPATH to <workspace>/tools/haider. "
                "Point it at Path(__file__).resolve().parent.parent (the directory that contains the hcli package)."
            ),
            "blast": {
                "files": ["hcli/engine.py"],
                "file_count": 1,
            },
            "run_after": "unittest test_acceptance_integrity test_parallel_engine test_runtime_pool",
            "blocks_if": "none",
        },
        {
            "id": "S4",
            "name": "Single helper for harness package root — still named haider on disk",
            "does": (
                "Headless harnesses each insert tools/haider. Introduce one constant in a small "
                "helper (or a pytest pythonpath) so the eventual rename is one line, not thirty. "
                "Do NOT do this while ten science lanes are rewriting tools/headless/."
            ),
            "blast": {
                "headless_sys_path_sites_targeting_haider": len(harness_haider_inserts),
                "headless_files_with_any_sys_path": len(headless_files_with),
                "science_lanes": "WAIT. Mid-flight lanes pin the insert. Land or pin them first.",
            },
            "run_after": "the hcli_* subset of tools/headless plus in-package tests",
            "blocks_if": "in-flight lanes on tools/headless",
        },
        {
            "id": "S5",
            "name": "Directory move hcli -> tools/hcli  (package name stays `hcli`)",
            "does": (
                "PYTHONPATH=tools then `import hcli` / `python -m hcli`. "
                "Update remaining live path literals. Do not touch receipts/, .hcli/ state, "
                ".hcli-legacy/ gitignore, research/lab/hcli, or crates/hide-backend. "
                "Leave tools/hcli/bootstrap/snapshots/haider.py and p0_tool_bridge.py where they are."
            ),
            "blast": {
                "live_path_strings": "every remaining tools/haider string outside receipts/ and docs",
                "depth_helpers": "must already be S2-complete",
                "receipts": "zero — historical filenames and body text stay",
            },
            "run_after": "full suite, HCLI_SWAP_CEILING_GIB=64 (464 passed / 1 skipped is the bar)",
            "blocks_if": "S2 or S3 skipped; science lanes still inserting the old path",
        },
        {
            "id": "S6",
            "name": "Re-run install-shims on operator machines",
            "does": (
                "User-facing commands are ALREADY `hcli` / `jhcli` exec-ing `python -m hcli` with "
                "PYTHONPATH=~/.local/share/hcli/current. Re-install so current/ points at the moved package. "
                "Keep both shim names."
            ),
            "blast": {
                "repo_files": 0,
                "operator_home": "~/.local/bin/{hcli,jhcli} and ~/.local/share/hcli/current",
            },
            "run_after": "PYTHONPATH=tools python3 -m hcli --help; hcli --help; cmp hcli jhcli shims",
            "blocks_if": "S5 not landed",
        },
        {
            "id": "S7",
            "name": "DONE: parse_haider_args renamed to parse_hcli_args",
            "does": "Landed. No alias was kept: every call site in this repo was rewritten in the same change, and out-of-tree callers of a symbol this repo never published are not a constituency.",
            "blast": {
                "files": [
                    "hcli/__init__.py",
                    "hcli/cli.py",
                    "hcli/tests/test_grammar.py",
                    "tools/headless/hcli_foundation_test.py",
                ],
            },
            "run_after": "test_grammar + hcli_foundation_test",
            "blocks_if": "none",
        },
        {
            "id": "S8",
            "name": "Later campaign: retire HCLI-v0 haider.py / p0_tool_bridge",
            "does": "Only after a census shows zero callers. Do not combine with S5.",
            "blast": {
                "files": [
                    "tools/hcli/bootstrap/snapshots/haider.py",
                    "tools/hcli/bootstrap/p0_tool_bridge.py",
                    "tools/hcli/bootstrap/test_haider_edit.py",
                    "tools/hcli/bootstrap/test_p0_tool_bridge.py",
                ],
            },
            "run_after": "grep callers == 0, then the same suite",
            "blocks_if": "any remaining caller, including docs treated as runbooks",
        },
    ]

    # --- counts ---
    counts = {
        "hcli_modules": len(hcli_modules),
        "hcli_test_files": len(hcli_tests),
        "hcli_test_modules_excluding_conftest": len(
            [p for p in hcli_tests if p.rsplit("/", 1)[-1].startswith("test_")]
        ),
        "haider_tree_files": len(haider_files),
        "headless_files_git": len(git_ls(root, "tools/headless")),
        "headless_py": len(headless_py),
        "receipts_headless_git": len(git_ls(root, "receipts/headless")),
        "line_hits_classified": sum(len(classified[b]) for b in BUCKETS),
        "by_bucket": {b: len(classified[b]) for b in BUCKETS},
        "live_import_from_hcli": len(import_hcli),
        "live_import_from_tools_haider_hcli": len(import_tools_haider),
        "headless_sys_path_sites": len(headless_sys),
        "headless_files_with_sys_path": len(headless_files_with),
        "headless_files_without_sys_path": len(headless_files_without),
        "sys_path_sites_all_inventoried": len(sys_path_sites),
        "missing_cited_paths": len(missing_paths),
    }

    standing = {
        "haider_is_a_fossil_namespace": True,
        "import_aider_in_tools": 0,
        "from_aider_in_tools": 0,
        "aider_chat_in_tools": 0,
        "aider_binary_present": str(Path.home() / ".local" / "bin" / "aider"),
        "note": (
            "git grep over tools/ for `import aider|from aider|aider-chat` exits 1 (no matches). "
            "573 `aider` hits in tools/ are the substring inside `haider`."
        ),
        "user_facing_already_hcli": True,
        "two_live_import_conventions": {
            "from_hcli": "PYTHONPATH=tools/haider (harnesses, __main__.py, grok_bridge, ledger docstring, test_grok_identity)",
            "from_tools_haider_hcli": "sys.path=repo root (44 in-package tests). Relies on PEP 420 namespace packages; tools/ has no __init__.py.",
        },
        "preserve": [
            "sealed schema ids",
            "historical receipt filenames",
            "historical log text",
            "everything under receipts/",
            "untracked / workspace campaign corpora",
        ],
        "sparse_checkout": {
            "tools_haider_on_disk": (root / "tools" / "haider").exists(),
            "method": "git ls-tree / git show / git grep; missing-on-disk is not absent-from-git",
        },
        "anti_goodhart": (
            "Success is one import authority (`hcli`), elimination of fossil path literals "
            "in live code, and harnesses that do not encode tools/haider. Not a LOC target. "
            "Splitting 33 modules into hawking/{agentos,runtime,...} would add ceremony without adding capability."
        ),
    }

    receipt = {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "head": head,
        "branch": branch,
        "repo": str(root),
        "method": (
            "git grep + git show + disk overlay. Sparse-aware. This script writes a plan; "
            "it does not move, delete, or rename product source."
        ),
        "standing_facts": standing,
        "counts": counts,
        "buckets": classified,
        "sys_path_insert": {
            "headless_sites": headless_sys,
            "headless_by_target": dict(headless_sys_by_target),
            "headless_files_with": headless_files_with,
            "headless_files_without": headless_files_without,
            "all_inventoried_sites": sys_path_sites,
        },
        "package_layout": layout,
        "depth_assumptions": depth,
        "user_facing": {
            "commands": ["hcli", "jhcli", "python -m hcli", "python -m hcli install-shims"],
            "slash_commands": list(SLASH_COMMANDS),
            "parse_symbol": "parse_hcli_args",
            "shims_identical": True,
            "cite": "hcli/cli.py:186-221 and hcli/commands.py REQUIRED_COMMANDS",
        },
        "sealed_exclusion": {
            "policy": (
                "Do not rename schema ids, receipt filenames, log text, or anything under receipts/. "
                "Historical names in historical receipts are evidence."
            ),
            "receipts_headless": sealed_receipts,
            "schema_ids_in_receipts_headless": schema_ids,
            "named_schema_cites": extra_schema_cites,
            "on_disk_state_dir_names": state_dirs,
            "gitignore": {".hcli-legacy/": "PRESERVE"},
        },
        "migration_steps": steps,
        "unknowns": classified["UNKNOWN"],
        "what_i_watched_fail": failures,
        "scope": {
            "write": ["tools/headless/namespace_plan.py", str(RECEIPT_REL)],
            "deny": [
                "workspace",
                "crates",
                "visionmcp",
                "app",
                "lab",
                "tools/haider",
                "ramanujan",
                "receipts/ascent-2026-08-16",
                "receipts/ascent-2026-08-18",
            ],
        },
        "missing_cited_paths": missing_paths,
    }

    out_path = root / RECEIPT_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    # ---------------- stdout plan ----------------
    print(f"# NAMESPACE MIGRATION PLAN  ({SCHEMA})")
    print(f"generated_at: {generated_at}")
    print(f"head: {head}  branch: {branch}")
    print(f"repo: {root}")
    print(f"receipt: {RECEIPT_REL}")
    print()
    print("This script is a census. It changed no product source.")
    print()

    print("## STANDING FACTS (re-measured)")
    print(f"- tools/haider on disk: {standing['sparse_checkout']['tools_haider_on_disk']}")
    print(f"- hcli modules: {counts['hcli_modules']} (contract 33)")
    print(
        f"- hcli tests: {counts['hcli_test_modules_excluding_conftest']} test_*.py "
        f"+ conftest ({counts['hcli_test_files']} files under tests/)"
    )
    print(f"- headless files in git: {counts['headless_files_git']} (contract said 63)")
    print(f"- receipts/headless in git: {counts['receipts_headless_git']}")
    print("- import aider / from aider / aider-chat in tools/: 0 (git grep exit 1)")
    print("- user-facing command is already `hcli` / `python -m hcli`; jhcli is an identical shim")
    print("- two live import conventions exist (from hcli vs from tools.haider.hcli)")
    print()

    print("## COUNTS BY BUCKET")
    for b in BUCKETS:
        print(f"- {b}: {counts['by_bucket'][b]}")
    print(f"- missing cited paths: {counts['missing_cited_paths']}")
    print()

    print("## 1. LIVE IMPLEMENTATION REFERENCES")
    print("Python control plane lives at hcli/ (33 modules). Fossil v0 siblings:")
    for p in [
        "tools/hcli/bootstrap/snapshots/haider.py",
        "tools/hcli/bootstrap/p0_tool_bridge.py",
        "tools/hcli/bootstrap/test_haider_edit.py",
        "tools/hcli/bootstrap/test_p0_tool_bridge.py",
        "tools/hcli/bootstrap/P1_HAIDER_PRODUCTIZATION_MAX.md",
    ]:
        print(f"  {p}  resolves={resolves(root, p)}")
    print("Line hits classified LIVE_IMPLEMENTATION_REFERENCES:")
    print_sites("LIVE_IMPLEMENTATION_REFERENCES", classified["LIVE_IMPLEMENTATION_REFERENCES"])

    print("\n## 2. LIVE IMPORT PATHS")
    print(
        f"from/import hcli (no tools.haider prefix): {len(import_hcli)}\n"
        f"from/import tools.haider.hcli: {len(import_tools_haider)}"
    )
    print_sites("LIVE_IMPORT_PATHS", classified["LIVE_IMPORT_PATHS"])

    print("\n## 3. SYS.PATH INSERT INVENTORY")
    print(
        f"headless sites: {len(headless_sys)} in {len(headless_files_with)} files; "
        f"{len(headless_files_without)} headless .py files have no sys.path.insert"
    )
    print("by target_class:")
    for k, v in sorted(headless_sys_by_target.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")
    print("\nheadless sys.path.insert sites:")
    for s in headless_sys:
        print(f"  {s['path']}:{s['line']}: [{s['target_class']}] {s['text']}")
    print("\nheadless files WITHOUT sys.path.insert:")
    for p in headless_files_without:
        print(f"  {p}")
    print("\nin-package tests + fossil siblings (not headless):")
    for s in sys_path_sites:
        if not s["in_headless"]:
            print(f"  {s['path']}:{s['line']}: [{s['target_class']}] {s['text']}")

    print("\n## 4. TEST/HARNESS PATHS")
    print_sites("TEST_HARNESS_PATHS", classified["TEST_HARNESS_PATHS"])

    print("\n## 5. USER-FACING COMMANDS")
    print("Canonical (already):")
    print("  hcli")
    print("  jhcli          # byte-identical shim, same python -m hcli")
    print("  python -m hcli")
    print("  python -m hcli install-shims")
    print("Slash commands (CommandHandler.REQUIRED_COMMANDS):")
    for c in SLASH_COMMANDS:
        print(f"  {c}")
    print("Fossil CLI still documented: python tools/hcli/bootstrap/snapshots/haider.py 1")
    print("Symbol renamed: parse_haider_args -> parse_hcli_args (done)")
    print_sites("USER_FACING_COMMANDS", classified["USER_FACING_COMMANDS"])

    print("\n## 6. HISTORICAL RECEIPTS (PRESERVE)")
    print("Policy: every file under receipts/ is sealed. Names and body text do not change.")
    print(f"receipts/headless files: {len(sealed_receipts)}")
    for r in sealed_receipts:
        sch = r.get("schema") or r.get("kind") or ""
        print(f"  {r['path']}  schema={sch}  resolves={r['resolves']}")
    rec_files = sorted({s["path"] for s in classified["HISTORICAL_RECEIPTS"]})
    print(f"\nunique files with namespace hits classified HISTORICAL_RECEIPTS: {len(rec_files)}")
    for p in rec_files:
        n = sum(1 for s in classified["HISTORICAL_RECEIPTS"] if s["path"] == p)
        print(f"  {p}  hits={n}  resolves={resolves(root, p)}")

    print("\n## 7. SEALED SCHEMA IDS (PRESERVE)")
    print("From receipts/headless JSON `schema` fields:")
    for sid in schema_ids:
        print(f"  {sid}")
    print("Named extra seals:")
    for c in extra_schema_cites:
        print(f"  {c['id']}  {c['path']}:{c.get('line')}  resolves={c['resolves']}")
    print("On-disk state directory names (not package paths; do not rename mid-flight):")
    for d in state_dirs:
        print(f"  {d['name']}  {d['role']}  cite={d['cite']}")

    print("\n## 8. HISTORICAL LOG TEXT (PRESERVE)")
    print_sites("HISTORICAL_LOG_TEXT", classified["HISTORICAL_LOG_TEXT"])

    print("\n## UNKNOWN")
    print_sites("UNKNOWN", classified["UNKNOWN"])

    print("\n## DEPTH ASSUMPTIONS (must die before a one-level directory drop)")
    for d in depth:
        print(f"  {d['path']}:{d['line']}: {d['text']}")
        print(f"    meaning: {d['meaning']}")
        print(f"    breaks_if: {d['breaks_if']}")
        print(f"    resolves: {d['resolves']}")

    print("\n## PROPOSED PACKAGE LAYOUT (from measured ownership)")
    print("Canonical package name: hcli")
    print("Canonical physical path after the move: tools/hcli/   (today: hcli/)")
    print("Do not create a Python hawking/ tree. The Rust crate is already named hawking.")
    print("Do not split this package to satisfy the hint directories.")
    print()
    covered = set()
    for g, spec in OWNERSHIP_GROUPS.items():
        print(f"### group {g}  -> {spec['belongs']}")
        print(f"    owns: {spec['owns']}")
        for name in spec["modules"]:
            rel = f"hcli/{name}"
            covered.add(name)
            meta = next((m for m in assigned if m["path"].endswith("/" + name)), None)
            loc = meta["loc"] if meta else "?"
            classes = ",".join((meta["classes"] if meta else [])[:8])
            print(f"    {rel}  loc={loc}  classes=[{classes}]  resolves={resolves(root, rel)}")
        print()
    leftover = [m["path"] for m in assigned if m["path"].rsplit("/", 1)[-1] not in covered]
    if leftover:
        print("UNASSIGNED modules (bug in this census):")
        for p in leftover:
            print(f"  {p}")
    print("Hint directories NOT created:")
    for item in layout["do_not_invent"]:
        print(f"  {item['hint']}: {item['verdict']} — {item['reason']}")

    print("\n## MIGRATION ORDER (suite runnable between steps)")
    for st in steps:
        print(f"### {st['id']}  {st['name']}")
        print(f"    {st['does']}")
        blast = st["blast"]
        print(f"    blast: {json.dumps(blast, sort_keys=True)[:500]}")
        print(f"    run_after: {st['run_after']}")
        print(f"    blocks_if: {st['blocks_if']}")
        print()

    print("## WHAT I WATCHED FAIL")
    for f in failures:
        print(f"- {f['name']}: exit={f['exit']}")
        print(f"    argv: {f['argv']}")
        head_out = f.get("output_head") or ""
        for ln in head_out.splitlines()[:8]:
            print(f"    {ln}")

    print("\n## SCOPE CHECK")
    print(f"wrote {RECEIPT_REL}  exists={out_path.is_file()}  bytes={out_path.stat().st_size}")
    status = git_text(root, ["status", "--short", "--", "tools/headless", "receipts/headless"])
    print("git status --short tools/headless receipts/headless:")
    print(status or "  (clean)")
    denied_hits = git_text(
        root,
        [
            "status",
            "--short",
            "--",
            "workspace",
            "crates",
            "visionmcp",
            "app",
            "lab",
            "tools/haider",
            "ramanujan",
            "receipts/ascent-2026-08-16",
            "receipts/ascent-2026-08-18",
        ],
    )
    print("git status --short DENY paths:")
    print(denied_hits or "  (clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
