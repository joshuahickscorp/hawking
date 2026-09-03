#!/usr/bin/env python3
"""Deterministic code graph over live HCLI Python plus headless harnesses.

Ordinary import graphs miss the edges that dominate cost in this campaign:
subprocess spawns (grok-run, llama-server, git, python -m hcli), persistence
through shared on-disk documents, named tools, runtime topology, and source
mutation. This census walks AST, not importlib, and reads sparse-missing
files through git so a hole on disk is not treated as absence.

    python3 tools/headless/code_graph.py

Writes receipts/headless/CODE_GRAPH.json. Byte-identical across reruns of
the same tree (no wall-clock fields). Does not modify any other path.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCHEMA = "hawking.headless.code_graph.v1"
REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts" / "headless" / "CODE_GRAPH.json"

CENSUS_ROOTS = ("hcli", "tools/headless")

STDLIB: Set[str] = set(getattr(sys, "stdlib_module_names", ())) | {
    "__future__",
    "typing_extensions",
}

THIRD_PARTY_ROOTS = {
    "numpy",
    "np",
    "mlx",
    "mlx_lm",
    "torch",
    "pytest",
    "yaml",
    "requests",
    "PIL",
    "cv2",
    "openai",
    "anthropic",
    "grok",
    "httpx",
    "pydantic",
    "fastapi",
    "flask",
    "sklearn",
    "scipy",
    "pandas",
    "rich",
    "click",
    "tqdm",
    "regex",
    "orjson",
    "msgpack",
    "zstandard",
    "blake3",
    "prompt_toolkit",
    "prompt_toolkit.history",
}

# Sibling/installable packages this tree talks to but does not vendor.
EXTERNAL_PACKAGE_ROOTS = {
    "visionmcp",
}

TOOL_BINARIES = {
    "grok-run": "tool",
    "grok": "tool",
    "aider": "tool",
    "aider-chat": "tool",
    "cargo": "tool",
    "pytest": "tool",
    "rg": "tool",
    "blender": "tool",
    "colmap": "tool",
    "ffmpeg": "tool",
    "git": "tool",
    "hcli": "runtime",
    "jhcli": "runtime",
    "llama-server": "runtime",
    "llama-cli": "runtime",
    "mlx_lm.server": "runtime",
    "sysctl": "runtime",
    "vm_stat": "runtime",
    "memory_pressure": "runtime",
    "ps": "runtime",
    "pkill": "runtime",
    "kill": "runtime",
    "pgrep": "runtime",
    "which": "tool",
    "bash": "subprocess",
    "sh": "subprocess",
    "zsh": "subprocess",
    "python": "runtime",
    "python3": "runtime",
    "codesign": "tool",
    "xcrun": "tool",
    "metal": "runtime",
    "ollama": "runtime",
}

GROK_SUBCOMMANDS = {
    "delegate",
    "audit",
    "consult",
    "status",
    "wait",
    "cleanup",
    "report",
    "cancel",
}

GIT_MUTATION_SUBCOMMANDS = {
    "add",
    "commit",
    "mv",
    "rm",
    "checkout",
    "restore",
    "reset",
    "clean",
    "rebase",
    "merge",
    "cherry-pick",
    "stash",
    "push",
    "tag",
    "update-index",
    "worktree",
}

PERSIST_WRITE_FNAMES = {
    "open",
    "io.open",
    "os.replace",
    "os.rename",
    "os.unlink",
    "os.remove",
    "os.fsync",
    "os.mkdir",
    "os.makedirs",
    "json.dump",
    "atomic_write_json",
    "_atomic_write_text",
    "_atomic_write",
    "shutil.copy",
    "shutil.copy2",
    "shutil.copytree",
    "shutil.move",
    "shutil.rmtree",
}
PERSIST_WRITE_TAILS = {
    "write_text",
    "write_bytes",
    "mkdir",
    "makedirs",
    "unlink",
    "touch",
}
PERSIST_READ_TAILS = {
    "read_text",
    "read_bytes",
}
SUBPROCESS_TAILS = {
    "run",
    "Popen",
    "call",
    "check_output",
    "check_call",
    "getoutput",
    "getstatusoutput",
    "system",
    "popen",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "create_subprocess_exec",
    "create_subprocess_shell",
}
DYNAMIC_IMPORT_TAILS = {
    "import_module",
    "spec_from_file_location",
    "__import__",
    "run_path",
    "run_module",
    "load_module",
    "module_from_spec",
}
MUTATION_TAILS = {
    "apply_mutation_operations",
    "rollback_mutation",
    "_apply_replace",
    "_apply_insert",
    "_apply_create",
    "_apply_replace",
}

DEST_FAMILIES = (
    (".hcli/dag.json", re.compile(r"dag\.json")),
    (".hcli/mission/state.json", re.compile(r"mission.*state\.json|state\.json")),
    (".hcli/mission/mission.log", re.compile(r"mission\.log")),
    (".hcli/sessions", re.compile(r"\.hcli.*sessions")),
    (".hcli/grok", re.compile(r"\.hcli.*grok")),
    (".hcli/mutation.lock", re.compile(r"mutation\.lock")),
    ("~/.config/hcli/config.json", re.compile(r"config/hcli/config\.json|hcli/config\.json")),
    ("~/.config/hcli/machine_genome.json", re.compile(r"machine_genome\.json")),
    ("receipts/headless", re.compile(r"receipts/headless")),
    (".hcli-legacy/", re.compile(r"\.hcli-legacy/")),
    ("receipts/", re.compile(r"receipts/")),
)

CENSUS_PY_RE = re.compile(
    r"(?:tools/hcli|tools/headless)/[A-Za-z0-9_./-]+\.py"
)
RECEIPT_RE = re.compile(r"receipts/headless/[A-Za-z0-9_.-]+\.json")
HCLI_REL_RE = re.compile(r"\.hcli/[A-Za-z0-9_./-]+")


def _posix(path: str) -> str:
    return str(path).replace("\\", "/")


def git_head() -> str:
    proc = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (proc.stdout or "").strip() or "UNKNOWN"


def git_ls_tree(*prefixes: str) -> List[str]:
    args = ["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", "HEAD", "--", *prefixes]
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def git_show(rel: str) -> Optional[bytes]:
    proc = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def git_ls_tree_all_py() -> List[str]:
    proc = subprocess.run(
        ["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip().endswith(".py")
    ]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_test_path(rel: str) -> bool:
    p = _posix(rel)
    name = Path(p).name
    return (
        "/tests/" in f"/{p}/"
        or name.endswith("_test.py")
        or name.startswith("test_")
        or name == "conftest.py"
    )


def module_kind(rel: str) -> str:
    p = _posix(rel)
    test = is_test_path(p)
    if p.startswith("hcli/"):
        return "hcli_test" if test else "hcli_product"
    if p.startswith("tools/hcli/bootstrap/"):
        name = Path(p).name
        if name in {"haider.py", "p0_tool_bridge.py"}:
            return "hcli_fossil"
        return "hcli_fossil_test" if test else "hcli_fossil"
    if p.startswith("tools/headless/"):
        return "headless_test" if test else "headless_harness"
    return "other_test" if test else "other"


def is_product_kind(kind: str) -> bool:
    return kind in {"hcli_product", "hcli_fossil"}


def dotted_candidates(rel: str) -> List[str]:
    p = _posix(rel)
    if p.endswith("/__init__.py"):
        p = p[: -len("/__init__.py")]
    elif p.endswith(".py"):
        p = p[: -len(".py")]
    parts = p.split("/")
    names = [".".join(parts), ".".join(parts[-1:])]
    if parts[:2] == ["tools", "hcli"] and len(parts) > 2:
        names.append(".".join(parts[2:]))
        names.append(".".join(parts[1:]))
    if parts[:1] == ["tools"] and len(parts) > 1:
        names.append(".".join(parts[1:]))
    # unique, stable order
    out: List[str] = []
    seen: Set[str] = set()
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return call_name(node.func)
    return ""


def const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def all_strings(node: ast.AST) -> List[str]:
    out: List[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if child.value:
                out.append(child.value)
        elif isinstance(child, ast.JoinedStr):
            parts = [const_str(v) for v in child.values]
            if all(p is not None for p in parts):
                out.append("".join(parts))  # type: ignore[arg-type]
    return out


def eval_path_expr(node: ast.AST, env: Dict[str, str]) -> Optional[str]:
    if node is None:
        return None
    s = const_str(node)
    if s is not None:
        return s
    if isinstance(node, ast.JoinedStr):
        parts: List[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                inner = eval_path_expr(v.value, env)
                if inner is None:
                    parts.append("${}")
                else:
                    parts.append(inner)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Attribute):
        base = eval_path_expr(node.value, env)
        if base is None:
            # Path.home() handled via Call; shutil.which etc. no
            if node.attr == "home" and call_name(node.value) in {"Path", "pathlib.Path"}:
                return "~"
            return None
        if node.attr == "parent":
            if base in {"", "."}:
                return ""
            parent = str(Path(base).parent)
            return "" if parent == "." else parent
        if node.attr in {"resolve", "absolute", "as_posix"}:
            return base
        return None
    if isinstance(node, ast.Subscript):
        base_node = node.value
        # Path(...).parents[N]
        if isinstance(base_node, ast.Attribute) and base_node.attr == "parents":
            root = eval_path_expr(base_node.value, env)
            if root is None:
                return None
            sl = node.slice
            n: Optional[int] = None
            if isinstance(sl, ast.Constant) and isinstance(sl.value, int):
                n = sl.value
            if n is None:
                return None
            p = Path(root) if root else Path(".")
            try:
                got = p.parents[n]
            except IndexError:
                return None
            sgot = "" if str(got) == "." else str(got)
            return _posix(sgot)
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
        left = eval_path_expr(node.left, env)
        right = eval_path_expr(node.right, env)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if left in {"", "."}:
            return _posix(right)
        return _posix(str(Path(left) / right))
    if isinstance(node, ast.Call):
        fname = call_name(node.func)
        if fname in {"Path", "pathlib.Path"}:
            if not node.args:
                return None
            inner = eval_path_expr(node.args[0], env)
            return inner
        if fname in {"os.path.join", "posixpath.join"}:
            parts = [eval_path_expr(a, env) for a in node.args]
            if any(p is None for p in parts) or not parts:
                return None
            return _posix(str(Path(parts[0]) / Path(*parts[1:]))) if len(parts) > 1 else parts[0]
        if fname in {"os.path.dirname"}:
            inner = eval_path_expr(node.args[0], env) if node.args else None
            if inner is None:
                return None
            parent = str(Path(inner).parent)
            return "" if parent == "." else parent
        if fname in {"os.path.expanduser"}:
            inner = eval_path_expr(node.args[0], env) if node.args else None
            if inner is None:
                return None
            if inner.startswith("~"):
                return inner
            return inner
        if fname in {"os.path.abspath", "os.path.realpath", "os.path.normpath"}:
            return eval_path_expr(node.args[0], env) if node.args else None
        if fname in {"str"}:
            return eval_path_expr(node.args[0], env) if node.args else None
        if fname.endswith(".resolve") or fname.endswith(".absolute"):
            return eval_path_expr(node.func.value, env) if isinstance(node.func, ast.Attribute) else None  # type: ignore[attr-defined]
        if fname in {"Path.home", "pathlib.Path.home"}:
            return "~"
        if fname.endswith(".home"):
            return "~"
        # Path(__file__).resolve().parents[N] already handled via Subscript
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"replace", "format"}:
            return None
    if isinstance(node, ast.IfExp):
        return eval_path_expr(node.body, env) or eval_path_expr(node.orelse, env)
    return None


def module_env(tree: ast.AST, rel: str) -> Dict[str, str]:
    env: Dict[str, str] = {
        "__file__": rel,
        "__name__": dotted_candidates(rel)[0] if dotted_candidates(rel) else rel,
    }
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if not isinstance(node, ast.Assign):
            continue
        value = eval_path_expr(node.value, env)
        if value is None:
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                env[tgt.id] = value
    return env


def rel_from_repo(path_str: str) -> Optional[str]:
    s = _posix(path_str)
    if s.startswith("~"):
        return s
    for root in CENSUS_ROOTS:
        if s == root or s.startswith(root + "/"):
            return s
    if s.startswith("tools/") or s.startswith("receipts/") or s.startswith("research/lab/"):
        return s
    # strip an absolute prefix that contains the repo
    marker = "/tools/"
    idx = s.find(marker)
    if idx >= 0:
        # may be .../repo/tools/...
        cut = s[idx + 1 :]  # tools/...
        return cut
    marker = "/receipts/"
    idx = s.find(marker)
    if idx >= 0:
        return s[idx + 1 :]
    return s if "/" not in s or s.startswith(".") else s


def dest_family(strings: Sequence[str], evaluated: Optional[str]) -> Optional[str]:
    blob = " ".join(strings)
    if evaluated:
        blob = evaluated + " " + blob
        ev = _posix(evaluated)
        if ev.startswith("receipts/headless/") or "/receipts/headless/" in ev:
            return "receipts/headless"
        if ".hcli/" in ev:
            rest = ev.split(".hcli/", 1)[1]
            return ".hcli/" + rest.split("${")[0].rstrip("/")
        if ev.startswith("~/.config/hcli"):
            return ev
    for fam, rx in DEST_FAMILIES:
        if rx.search(blob):
            return fam
    for s in strings:
        m = RECEIPT_RE.search(s)
        if m:
            return "receipts/headless"
        m = HCLI_REL_RE.search(s)
        if m:
            return m.group(0)
    if evaluated:
        ev = _posix(evaluated)
        if "receipts/" in ev:
            return "receipts/"
        if ev.endswith(".json") or ev.endswith(".jsonl") or ev.endswith(".log"):
            return rel_from_repo(ev) or ev
    return None


def open_is_write(call: ast.Call) -> Optional[bool]:
    """True if write, False if read, None if unknown."""
    mode = None
    if len(call.args) >= 2:
        mode = const_str(call.args[1])
    for kw in call.keywords:
        if kw.arg == "mode":
            mode = const_str(kw.value) or mode
    if mode is None:
        return False
    if any(ch in mode for ch in "wax+"):
        return True
    return False


def list_first_arg_elts(call: ast.Call) -> List[ast.AST]:
    if not call.args:
        return []
    arg0 = call.args[0]
    if isinstance(arg0, (ast.List, ast.Tuple)):
        return list(arg0.elts)
    return [arg0]


def is_persist_write_call(fname: str, tail: str) -> bool:
    if fname in {"json.dumps", "json.loads"}:
        return False
    if fname in PERSIST_WRITE_FNAMES:
        return True
    if tail in PERSIST_WRITE_TAILS:
        return True
    if fname in {"json.dump"}:
        return True
    if fname in {"os.replace", "os.rename"}:
        return True
    return False


def is_persist_read_call(fname: str, tail: str) -> bool:
    if fname in {"open", "io.open"}:
        return True
    if fname in {"json.load"}:
        return True
    if tail in PERSIST_READ_TAILS:
        return True
    return False


def symbolic_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = symbolic_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def persist_destination(call: ast.Call, fname: str, env: Dict[str, str]) -> Optional[str]:
    """Best-effort dest: evaluated path, else a symbolic receiver/arg name."""
    tail = fname.split(".")[-1]

    def resolve_receiver() -> Optional[str]:
        if not isinstance(call.func, ast.Attribute):
            return None
        recv_node = call.func.value
        got = eval_path_expr(recv_node, env)
        if got:
            return got
        # Path(x).write_text — receiver is a Path(...) call
        if isinstance(recv_node, ast.Call) and call_name(recv_node.func) in {"Path", "pathlib.Path"}:
            if recv_node.args:
                return eval_path_expr(recv_node.args[0], env) or symbolic_name(recv_node.args[0])
        return symbolic_name(recv_node)

    receiver = resolve_receiver()
    arg0 = eval_path_expr(call.args[0], env) if call.args else None
    if arg0 is None and call.args:
        arg0 = symbolic_name(call.args[0])
    arg1 = None
    if tail == "dump" and len(call.args) >= 2:
        arg1 = eval_path_expr(call.args[1], env) or symbolic_name(call.args[1])
    if fname in {"atomic_write_json", "_atomic_write_text", "_atomic_write", "os.replace", "os.rename", "os.unlink", "os.remove", "os.mkdir", "os.makedirs"}:
        return arg0 or receiver
    if fname in {"open", "io.open"}:
        return arg0 or receiver
    if tail in {"write_text", "write_bytes", "read_text", "read_bytes", "mkdir", "makedirs", "unlink", "touch"}:
        # Content is arg0; dest is the receiver. Never take the content string.
        return receiver
    if fname == "json.dump":
        return arg1 or receiver
    return arg0 or receiver


def enclosing_functions(tree: ast.AST) -> List[Tuple[int, int, str, List[str]]]:
    spans: List[Tuple[int, int, str, List[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = int(getattr(node, "end_lineno", None) or node.lineno)
            spans.append((node.lineno, end, node.name, all_strings(node)))
    return spans


def infer_binary_from_context(line: int, spans: Sequence[Tuple[int, int, str, List[str]]]) -> Optional[str]:
    best: Optional[Tuple[int, int, str, List[str]]] = None
    for span in spans:
        a, b, _name, _strs = span
        if a <= line <= b and (best is None or (b - a) < (best[1] - best[0])):
            best = span
    if best is None:
        return None
    hits = []
    for s in best[3]:
        base = os.path.basename(s)
        if s in TOOL_BINARIES:
            hits.append(s)
        elif base in TOOL_BINARIES:
            hits.append(base)
    # unique preserving order
    uniq = []
    for h in hits:
        if h not in uniq:
            uniq.append(h)
    if len(uniq) == 1:
        return uniq[0]
    # prefer distinctive over python/bash
    distinctive = [h for h in uniq if h not in {"python", "python3", "bash", "sh", "zsh", "which"}]
    if len(distinctive) == 1:
        return distinctive[0]
    return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    watched: List[Dict[str, Any]] = []
    git_all = git_ls_tree_all_py()
    by_top: Dict[str, int] = defaultdict(int)
    for p in git_all:
        by_top[p.split("/", 1)[0]] += 1
    git_census = [
        p
        for p in git_ls_tree(*CENSUS_ROOTS)
        if p.endswith(".py") and "__pycache__" not in p.split("/")
    ]
    disk_census: List[str] = []
    for root in CENSUS_ROOTS:
        d = REPO / root
        if not d.is_dir():
            watched.append(
                {
                    "what": f"disk walk of {root}",
                    "result": "ABSENT",
                    "reason": f"{root} is not a directory in this worktree (sparse hole or missing)",
                }
            )
            continue
        for p in d.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            disk_census.append(_posix(str(p.relative_to(REPO))))

    paths = sorted(set(git_census) | set(disk_census))
    git_set = set(git_census)
    disk_set = set(disk_census)

    files: Dict[str, Dict[str, Any]] = {}
    for rel in paths:
        disk_path = REPO / rel
        disk_bytes: Optional[bytes] = None
        if disk_path.is_file():
            try:
                disk_bytes = disk_path.read_bytes()
            except OSError as exc:
                watched.append(
                    {
                        "what": f"read {rel} from disk",
                        "result": "FAIL",
                        "reason": str(exc),
                    }
                )
        git_bytes = None
        if rel in git_set:
            git_bytes = git_show(rel)
            if git_bytes is None:
                watched.append(
                    {
                        "what": f"git show HEAD:{rel}",
                        "result": "FAIL",
                        "reason": "git show exited nonzero",
                    }
                )
        origin = "UNKNOWN"
        data: Optional[bytes] = None
        if disk_bytes is not None and git_bytes is not None:
            if disk_bytes == git_bytes:
                origin = "disk"
                data = disk_bytes
            else:
                origin = "disk_differs_from_git"
                data = disk_bytes  # untracked work is real work
                watched.append(
                    {
                        "what": f"{rel} disk vs git",
                        "result": "DIFFERS",
                        "reason": "working tree bytes differ from HEAD; graph uses disk",
                    }
                )
        elif disk_bytes is not None:
            origin = "disk_untracked"
            data = disk_bytes
        elif git_bytes is not None:
            origin = "git_sparse"
            data = git_bytes
        else:
            watched.append(
                {
                    "what": f"load {rel}",
                    "result": "FAIL",
                    "reason": "neither disk nor git produced bytes",
                }
            )
            continue
        assert data is not None
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
            watched.append(
                {
                    "what": f"decode {rel}",
                    "result": "REPLACE",
                    "reason": "not strict utf-8; decoded with replacement",
                }
            )
        files[rel] = {
            "path": rel,
            "origin": origin,
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "text": text,
            "in_git": rel in git_set,
            "on_disk": rel in disk_set,
        }

    inventory = {
        "git_python_files": len(git_all),
        "git_python_by_top_level": dict(sorted(by_top.items())),
        "census_roots": list(CENSUS_ROOTS),
        "census_in_git": len(git_set),
        "census_on_disk": len(disk_set),
        "census_union": len(paths),
        "out_of_census_python": len(git_all)
        - sum(1 for p in git_all if any(p == r or p.startswith(r + "/") for r in CENSUS_ROOTS)),
        "sibling_namespaces_not_parsed": {
            "research/lab/hcli": [p for p in git_all if p.startswith("research/lab/hcli/")],
            "note": (
                "research/lab/hcli is a parallel namespace in git. Not parsed (out of census). "
                "A later lane that claims a single HCLI authority must compare it, not ignore it."
            ),
        },
    }
    return files, inventory, watched


# ---------------------------------------------------------------------------
# Parse one module
# ---------------------------------------------------------------------------


def try_parse(rel: str, text: str) -> Tuple[Optional[ast.AST], Optional[str]]:
    try:
        return ast.parse(text, filename=rel), None
    except SyntaxError as exc:
        return None, f"SyntaxError: {exc.msg} line {exc.lineno}"


def has_main_guard(tree: ast.AST) -> bool:
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        # if __name__ == "__main__"
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            left = test.left
            right = test.comparators[0]
            names = {const_str(left), const_str(right)}
            ids = set()
            if isinstance(left, ast.Name):
                ids.add(left.id)
            if isinstance(right, ast.Name):
                ids.add(right.id)
            if "__main__" in names and "__name__" in ids:
                return True
    return False


def has_def_main(tree: ast.AST) -> bool:
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return True
    return False


def is_reexport_only(tree: ast.AST) -> bool:
    if not isinstance(tree, ast.Module):
        return False
    saw_import = False
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # docstring
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            saw_import = True
            continue
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "__all__":
                continue
            return False
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            continue
        return False
    return saw_import


def is_thin_entrypoint(tree: ast.AST) -> bool:
    """__main__.py shape: import main, maybe sys.exit(main())."""
    if not isinstance(tree, ast.Module):
        return False
    stmts = [
        n
        for n in tree.body
        if not (
            (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
            or (isinstance(n, ast.ImportFrom) and n.module == "__future__")
        )
    ]
    if not stmts:
        return False
    if not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in stmts):
        return False
    leftover = [n for n in stmts if not isinstance(n, (ast.Import, ast.ImportFrom))]
    if not leftover:
        return True
    for n in leftover:
        if isinstance(n, ast.If) and has_main_guard(ast.Module(body=[n], type_ignores=[])):
            continue
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
            continue
        return False
    return True


def classify_import_root(name: str) -> str:
    root = name.split(".", 1)[0]
    if root in STDLIB or name in STDLIB:
        return "stdlib"
    if root in THIRD_PARTY_ROOTS:
        return "third_party"
    if root in EXTERNAL_PACKAGE_ROOTS:
        return "external_package"
    return "unknown"


class Extractor:
    def __init__(
        self,
        files: Dict[str, Dict[str, Any]],
        path_index: Dict[str, str],
        dotted_index: Dict[str, str],
    ) -> None:
        self.files = files
        self.path_index = path_index  # posix path -> posix path (identity)
        self.dotted_index = dotted_index  # dotted name -> posix path
        self.unknowns: List[Dict[str, Any]] = []

    def resolve_dotted(self, name: str, from_rel: str, level: int = 0) -> Optional[str]:
        if level and level > 0:
            base = Path(from_rel).parent
            for _ in range(level - 1):
                base = base.parent
            if name:
                cand = _posix(str(base / name.replace(".", "/")))
            else:
                cand = _posix(str(base / "__init__.py"))
                if cand in self.path_index:
                    return cand
                return None
            for guess in (cand + ".py", cand + "/__init__.py"):
                if guess in self.path_index:
                    return guess
            return None
        if not name:
            return None
        if name in self.dotted_index:
            return self.dotted_index[name]
        # walk prefixes: a.b.c -> a/b/c.py, a/b/c/__init__.py, then parent
        parts = name.split(".")
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in self.dotted_index:
                return self.dotted_index[prefix]
        # last-resort: sibling in same directory
        sib = _posix(str(Path(from_rel).parent / (parts[0] + ".py")))
        if sib in self.path_index:
            return sib
        return None

    def note_unknown(self, rel: str, kind: str, reason: str, line: int = 0) -> None:
        self.unknowns.append(
            {
                "path": rel,
                "kind": kind,
                "reason": reason,
                "line": line,
            }
        )

    def extract(self, rel: str, rec: Dict[str, Any]) -> Dict[str, Any]:
        text: str = rec["text"]
        tree, err = try_parse(rel, text)
        kind = module_kind(rel)
        facts: Dict[str, Any] = {
            "path": rel,
            "kind": kind,
            "origin": rec["origin"],
            "sha256": rec["sha256"],
            "bytes": rec["bytes"],
            "lines": text.count("\n") + (0 if text.endswith("\n") or not text else 1),
            "in_git": rec["in_git"],
            "on_disk": rec["on_disk"],
            "parse_error": err,
            "entrypoints": [],
            "reexport_only": False,
            "thin_entrypoint": False,
            "shebang": text.startswith("#!"),
            "has_main_guard": False,
            "has_def_main": False,
            "dotted_names": dotted_candidates(rel),
            "sys_path_inserts": [],
            "imports": [],
            "slash_commands": [],
            "named_tools": [],
        }
        if err or tree is None:
            self.note_unknown(rel, "parse", err or "ast.parse returned None")
            return facts

        facts["has_main_guard"] = has_main_guard(tree)
        facts["has_def_main"] = has_def_main(tree)
        facts["reexport_only"] = is_reexport_only(tree)
        facts["thin_entrypoint"] = Path(rel).name == "__main__.py" or is_thin_entrypoint(tree)
        env = module_env(tree, rel)

        entry: List[str] = []
        if Path(rel).name == "__main__.py":
            entry.append("python -m")
        if facts["has_main_guard"]:
            entry.append("__main__")
        if facts["has_def_main"] and facts["has_main_guard"]:
            entry.append("main()")
        if rel.endswith("hcli/cli.py") and facts["has_def_main"]:
            entry.append("hcli.cli:main")
        facts["entrypoints"] = entry

        imports: List[Dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "line": node.lineno,
                            "form": f"import {alias.name}",
                            "module": alias.name,
                            "names": [alias.asname or alias.name],
                            "level": 0,
                        }
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names = [a.name for a in node.names]
                stars = any(a.name == "*" for a in node.names)
                form = f"from {'.' * node.level}{mod} import {', '.join(names)}"
                imports.append(
                    {
                        "line": node.lineno,
                        "form": form,
                        "module": mod,
                        "names": names,
                        "level": node.level,
                        "star": stars,
                    }
                )
        facts["imports"] = imports

        # sys.path inserts
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fname = call_name(node.func)
            if fname.endswith("path.insert") or fname in {"sys.path.insert", "sys.path.append"}:
                ev = None
                if len(node.args) >= 2:
                    ev = eval_path_expr(node.args[1] if "insert" in fname else node.args[0], env)
                elif node.args:
                    ev = eval_path_expr(node.args[-1], env)
                facts["sys_path_inserts"].append(
                    {
                        "line": node.lineno,
                        "call": fname,
                        "path": ev or "UNKNOWN",
                        "strings": all_strings(node),
                    }
                )

        # slash commands + ALL_TOOLS
        for node in tree.body if isinstance(tree, ast.Module) else []:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in {
                        "REQUIRED_COMMANDS",
                        "ALL_TOOLS",
                        "READONLY_TOOLS",
                        "DEFAULT_WORKER_TASKS",
                    }:
                        strs = all_strings(node.value)
                        if tgt.id == "REQUIRED_COMMANDS":
                            facts["slash_commands"] = sorted(set(strs))
                        else:
                            facts["named_tools"] = sorted(set(facts["named_tools"]) | set(strs))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_cmd_"):
                cmd = "/" + node.name[len("_cmd_") :]
                if cmd not in facts["slash_commands"]:
                    facts["slash_commands"].append(cmd)
        facts["slash_commands"] = sorted(set(facts["slash_commands"]))

        rec["_tree"] = tree
        rec["_env"] = env
        rec["_facts"] = facts
        return facts


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------


def build_indexes(files: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    path_index = {p: p for p in files}
    dotted: Dict[str, str] = {}
    for p in files:
        for name in dotted_candidates(p):
            # first wins only if not already bound to a more specific file
            dotted.setdefault(name, p)
    return path_index, dotted


def extract_edges(
    files: Dict[str, Dict[str, Any]],
    extractor: Extractor,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    edges: Dict[str, List[Dict[str, Any]]] = {
        "import": [],
        "subprocess": [],
        "persistence": [],
        "tool": [],
        "runtime": [],
        "mutation": [],
    }
    persist_by_dest: Dict[str, Set[str]] = defaultdict(set)

    def add(kind: str, edge: Dict[str, Any]) -> None:
        edges[kind].append(edge)

    def resolve_census_string(s: str) -> Optional[str]:
        s = _posix(s.strip().strip("\"'"))
        if s in files:
            return s
        m = CENSUS_PY_RE.search(s)
        if m and m.group(0) in files:
            return m.group(0)
        return None

    for rel, rec in files.items():
        facts = rec.get("_facts")
        tree = rec.get("_tree")
        env = rec.get("_env") or {"__file__": rel}
        if not facts or tree is None:
            continue

        # ----- imports -----
        for imp in facts["imports"]:
            level = int(imp.get("level") or 0)
            mod = imp.get("module") or ""
            resolved = extractor.resolve_dotted(mod, rel, level=level)
            dst_kind = "census_module"
            dst: str
            if resolved:
                dst = resolved
            else:
                if level:
                    extractor.note_unknown(
                        rel,
                        "import",
                        f"unresolved relative import {imp['form']}",
                        imp["line"],
                    )
                    dst = f"UNKNOWN:{imp['form']}"
                    dst_kind = "unknown"
                else:
                    root = (mod or "").split(".", 1)[0]
                    cls = classify_import_root(mod or root)
                    if cls == "unknown":
                        # maybe repo python outside census
                        guess_paths = []
                        if mod:
                            dotted_as_path = mod.replace(".", "/") + ".py"
                            for prefix in ("", "tools/", "research/lab/", "visionmcp/src/"):
                                guess_paths.append(prefix + dotted_as_path)
                            guess_paths.append(mod.replace(".", "/") + "/__init__.py")
                            guess_paths.append("tools/" + dotted_as_path)
                        found = None
                        for g in guess_paths:
                            g = _posix(g)
                            if (REPO / g).is_file() or g in extractor.path_index:
                                found = g
                                break
                        # git existence for outside-census
                        if found is None and mod:
                            candidate = "tools/" + mod.replace(".", "/") + ".py"
                            found = candidate  # verified later
                            # only accept if we can prove it
                            if not (REPO / candidate).is_file():
                                # leave as unknown unless it is a known census miss
                                found = None
                        if found:
                            dst = found
                            dst_kind = "repo_outside_census"
                        else:
                            dst = mod or "UNKNOWN"
                            dst_kind = cls if cls != "unknown" else "unknown"
                            if dst_kind == "unknown" and mod:
                                extractor.note_unknown(
                                    rel,
                                    "import",
                                    f"unresolved import {imp['form']}",
                                    imp["line"],
                                )
                    else:
                        dst = mod
                        dst_kind = cls
            edge = {
                "src": rel,
                "dst": dst,
                "dst_kind": dst_kind,
                "line": imp["line"],
                "form": imp["form"],
                "names": imp.get("names") or [],
            }
            if dst_kind == "census_module":
                add("import", edge)
            else:
                # keep external on the module record only; still a real import
                facts.setdefault("imports_external", []).append(edge)

        init = "hcli/__init__.py"
        if init in files and not rel.startswith("hcli/"):
            forms = [(imp.get("form") or "") for imp in facts.get("imports") or []]
            if any(
                f.startswith("from hcli") or f.startswith("import hcli") or "tools.haider.hcli" in f
                for f in forms
            ):
                add(
                    "import",
                    {
                        "src": rel,
                        "dst": init,
                        "dst_kind": "census_module",
                        "line": 0,
                        "form": "implicit package init (import hcli.* executes hcli/__init__.py)",
                        "names": [],
                        "implicit_package_init": True,
                    },
                )

        func_spans = enclosing_functions(tree)

        # ----- calls -----
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fname = call_name(node.func)
            tail = fname.split(".")[-1] if fname else ""
            line = getattr(node, "lineno", 0)
            strings = all_strings(node)
            evaluated_first = eval_path_expr(node.args[0], env) if node.args else None

            # shutil.which / which("llama-server")
            if tail == "which" or fname in {"shutil.which", "os.which"}:
                target = const_str(node.args[0]) if node.args else None
                if target:
                    cls = TOOL_BINARIES.get(target, "tool")
                    add(
                        "tool",
                        {
                            "src": rel,
                            "dst": target,
                            "dst_kind": "binary",
                            "line": line,
                            "call": fname,
                            "evidence": f"{fname}({target!r})",
                        },
                    )
                    if cls == "runtime":
                        add(
                            "runtime",
                            {
                                "src": rel,
                                "dst": target,
                                "dst_kind": "binary",
                                "line": line,
                                "call": fname,
                                "evidence": f"{fname}({target!r})",
                            },
                        )

            # dynamic import
            if tail in DYNAMIC_IMPORT_TAILS or fname in {"__import__", "importlib.import_module"}:
                target_mod = None
                if node.args:
                    target_mod = const_str(node.args[0]) or eval_path_expr(node.args[0], env)
                resolved = None
                if target_mod:
                    resolved = extractor.resolve_dotted(target_mod, rel, 0)
                    if resolved is None:
                        resolved = resolve_census_string(target_mod)
                dst = resolved or target_mod or "UNKNOWN"
                dst_kind = "census_module" if resolved else ("unknown" if dst == "UNKNOWN" else "dynamic")
                if dst == "UNKNOWN":
                    extractor.note_unknown(rel, "runtime", f"{fname} target not statically recoverable", line)
                add(
                    "runtime",
                    {
                        "src": rel,
                        "dst": dst,
                        "dst_kind": dst_kind,
                        "line": line,
                        "call": fname,
                        "evidence": f"{fname}({target_mod!r})" if target_mod else fname,
                    },
                )
                if resolved:
                    add(
                        "import",
                        {
                            "src": rel,
                            "dst": resolved,
                            "dst_kind": "census_module",
                            "line": line,
                            "form": f"{fname}({target_mod!r})",
                            "names": [],
                            "dynamic": True,
                        },
                    )

            if tail in {"exec", "eval"} and fname in {"exec", "eval", "builtins.exec", "builtins.eval"}:
                add(
                    "runtime",
                    {
                        "src": rel,
                        "dst": "UNKNOWN:dynamic_code",
                        "dst_kind": "unknown",
                        "line": line,
                        "call": fname,
                        "evidence": fname,
                    },
                )

            # subprocess
            is_sub = False
            if fname.startswith("subprocess.") and tail in SUBPROCESS_TAILS:
                is_sub = True
            if fname in {"os.system", "os.popen", "os.execv", "os.execve", "os.execvp", "os.execvpe"}:
                is_sub = True
            if tail in {"Popen", "create_subprocess_exec", "create_subprocess_shell"}:
                is_sub = True
            if fname in {"asyncio.run"}:
                is_sub = False
            if is_sub:
                argv_lits: List[str] = []
                argv_syms: List[str] = []
                for elt in list_first_arg_elts(node):
                    v = const_str(elt) or eval_path_expr(elt, env)
                    if v:
                        argv_lits.append(v)
                    else:
                        sym = symbolic_name(elt)
                        if sym:
                            argv_syms.append(sym)
                argv_lits.extend(s for s in strings if s not in argv_lits)
                binary = None
                def _ok_token(t: str) -> bool:
                    if not t or t.startswith("-"):
                        return False
                    if "\n" in t or len(t) > 80:
                        return False
                    return True

                tokens = [t for t in argv_lits if _ok_token(t)]
                if tokens:
                    binary = os.path.basename(tokens[0])
                    if tokens[0] in TOOL_BINARIES:
                        binary = tokens[0]
                    if tokens[0] in GROK_SUBCOMMANDS:
                        binary = "grok-run"
                if binary in {"-c", "-m", "-lc"}:
                    binary = None
                census_hits = [s for s in argv_lits if resolve_census_string(s)]
                module_hits = [resolve_census_string(s) for s in argv_lits]
                module_hits = [m for m in module_hits if m]
                m_hcli = "-m" in argv_lits and any(a == "hcli" or a.startswith("hcli.") for a in argv_lits)
                dst: str
                dst_kind: str
                inferred = None
                if not binary and not module_hits and not m_hcli:
                    inferred = infer_binary_from_context(line, func_spans)
                    if inferred:
                        binary = inferred
                if module_hits:
                    dst = module_hits[0]
                    dst_kind = "census_module"
                elif m_hcli:
                    dst = "hcli/__main__.py"
                    dst_kind = "census_module" if "hcli/__main__.py" in files else "runtime"
                elif binary:
                    dst = binary
                    dst_kind = "binary"
                elif argv_syms:
                    dst = "symbolic:" + argv_syms[0]
                    dst_kind = "symbolic"
                    extractor.note_unknown(
                        rel,
                        "subprocess",
                        f"{fname} argv[0] is {argv_syms[0]!r} (not a literal)",
                        line,
                    )
                else:
                    dst = "UNRESOLVED_ARGV"
                    dst_kind = "unresolved"
                    extractor.note_unknown(
                        rel,
                        "subprocess",
                        f"{fname} argv not statically recoverable",
                        line,
                    )
                sub_edge = {
                    "src": rel,
                    "dst": dst,
                    "dst_kind": dst_kind,
                    "line": line,
                    "call": fname,
                    "argv_literals": argv_lits[:12],
                    "evidence": fname,
                }
                add("subprocess", sub_edge)
                # classify further
                git_sub = None
                if binary == "git" or (argv_lits and os.path.basename(argv_lits[0]) == "git"):
                    for a in argv_lits[1:]:
                        if a.startswith("-"):
                            continue
                        git_sub = a
                        break
                    add(
                        "tool",
                        {
                            "src": rel,
                            "dst": "git" + (f" {git_sub}" if git_sub else ""),
                            "dst_kind": "binary",
                            "line": line,
                            "call": fname,
                            "evidence": " ".join(argv_lits[:8]),
                        },
                    )
                    if git_sub in GIT_MUTATION_SUBCOMMANDS:
                        add(
                            "mutation",
                            {
                                "src": rel,
                                "dst": f"git {git_sub}",
                                "dst_kind": "vcs",
                                "line": line,
                                "call": fname,
                                "evidence": " ".join(argv_lits[:8]),
                            },
                        )
                elif binary in TOOL_BINARIES:
                    cls = TOOL_BINARIES[binary]
                    add(
                        "tool" if cls in {"tool", "subprocess"} else cls,
                        {
                            "src": rel,
                            "dst": binary,
                            "dst_kind": "binary",
                            "line": line,
                            "call": fname,
                            "evidence": " ".join(argv_lits[:8]),
                        },
                    )
                    if cls == "runtime":
                        add(
                            "runtime",
                            {
                                "src": rel,
                                "dst": binary,
                                "dst_kind": "binary",
                                "line": line,
                                "call": fname,
                                "evidence": " ".join(argv_lits[:8]),
                            },
                        )
                        add(
                            "tool",
                            {
                                "src": rel,
                                "dst": binary,
                                "dst_kind": "binary",
                                "line": line,
                                "call": fname,
                                "evidence": " ".join(argv_lits[:8]),
                            },
                        )
                    if binary in {"python", "python3"} and module_hits:
                        add(
                            "runtime",
                            {
                                "src": rel,
                                "dst": module_hits[0],
                                "dst_kind": "census_module",
                                "line": line,
                                "call": fname,
                                "evidence": " ".join(argv_lits[:8]),
                            },
                        )
                elif m_hcli:
                    add(
                        "runtime",
                        {
                            "src": rel,
                            "dst": "hcli/__main__.py",
                            "dst_kind": "census_module" if "hcli/__main__.py" in files else "runtime",
                            "line": line,
                            "call": fname,
                            "evidence": "python -m hcli",
                        },
                    )
                if binary == "aider" or "aider-chat" in argv_lits:
                    add(
                        "tool",
                        {
                            "src": rel,
                            "dst": "aider",
                            "dst_kind": "binary",
                            "line": line,
                            "call": fname,
                            "evidence": " ".join(argv_lits[:8]),
                        },
                    )

            # persistence — file I/O only. json.dumps / str.replace / environ.copy are not dests.
            write_like = is_persist_write_call(fname, tail)
            read_like = is_persist_read_call(fname, tail)
            if fname in {"open", "io.open"} or tail == "open":
                write_like = True
                read_like = True
            if write_like or read_like:
                if fname in {"open", "io.open"} or tail == "open":
                    w = open_is_write(node)
                    direction = "write" if w else "read"
                elif write_like:
                    direction = "write"
                else:
                    direction = "read"
                dest = persist_destination(node, fname, env)
                if dest:
                    dest = rel_from_repo(dest) or dest
                if dest in {"utf-8", "wb", "rb", "w", "r", "a", "x", "wt", "rt", "ab", "bytes", "True", "False"}:
                    dest = None
                if dest and not dest.startswith("symbolic:"):
                    path_like = (
                        "/" in dest
                        or dest.startswith("~")
                        or dest.startswith(".")
                        or dest.endswith((".json", ".jsonl", ".log", ".lock", ".md", ".txt", ".py"))
                    )
                    if not path_like:
                        dest = f"symbolic:{dest}"
                family = dest_family(strings, dest if dest and dest != evaluated_first else evaluated_first)
                if dest is None:
                    dest = family
                if dest is None:
                    # classified: we saw a writer, dest is a named expression we could not eval
                    recv = None
                    if isinstance(node.func, ast.Attribute):
                        recv = symbolic_name(node.func.value)
                    dest = f"symbolic:{recv}" if recv else f"symbolic:{fname}"
                    # Not UNKNOWN — the destination has a name. Follow-up would eval it.
                add(
                    "persistence",
                    {
                        "src": rel,
                        "dst": dest,
                        "dst_kind": "path",
                        "family": family,
                        "direction": direction,
                        "line": line,
                        "call": fname,
                        "evidence": fname,
                    },
                )
                if direction == "write" and not str(dest).startswith("symbolic:"):
                    persist_by_dest[dest].add(rel)

            # mutation API
            if tail in MUTATION_TAILS or fname.endswith("apply_mutation_operations") or fname.endswith("rollback_mutation"):
                add(
                    "mutation",
                    {
                        "src": rel,
                        "dst": "hcli.mutation",
                        "dst_kind": "api",
                        "line": line,
                        "call": fname,
                        "evidence": fname,
                    },
                )
            if tail == "MutationLock" or fname.endswith("MutationLock"):
                add(
                    "mutation",
                    {
                        "src": rel,
                        "dst": "hcli.resources.MutationLock",
                        "dst_kind": "api",
                        "line": line,
                        "call": fname,
                        "evidence": fname,
                    },
                )

            # urllib to localhost = runtime
            if "urlopen" in tail or fname.endswith("urlopen"):
                add(
                    "runtime",
                    {
                        "src": rel,
                        "dst": "http",
                        "dst_kind": "network",
                        "line": line,
                        "call": fname,
                        "strings": strings[:6],
                        "evidence": fname,
                    },
                )

        # named tools from ALL_TOOLS become tool edges
        for tool in facts.get("named_tools") or []:
            add(
                "tool",
                {
                    "src": rel,
                    "dst": tool,
                    "dst_kind": "named_tool",
                    "line": 0,
                    "call": "ALL_TOOLS",
                    "evidence": tool,
                },
            )
        for cmd in facts.get("slash_commands") or []:
            add(
                "tool",
                {
                    "src": rel,
                    "dst": f"hcli.cmd:{cmd}",
                    "dst_kind": "slash_command",
                    "line": 0,
                    "call": "CommandHandler",
                    "evidence": cmd,
                },
            )

        # string-literal census module mentions inside subprocess already handled;
        # also catch python -m hcli in string constants of this module (shell recipes)
        for s in re.findall(r"python3?\s+-m\s+hcli", rec["text"]):
            add(
                "runtime",
                {
                    "src": rel,
                    "dst": "hcli/__main__.py",
                    "dst_kind": "census_module" if "hcli/__main__.py" in files else "runtime",
                    "line": 0,
                    "call": "string-literal",
                    "evidence": s,
                },
            )

    # Shared dests are recorded as hubs in findings, not as a clique of module-module
    # edges (that clique exploded to thousands of edges on family tags like receipts/).
    coupling: List[Dict[str, Any]] = []
    for dest, writers in persist_by_dest.items():
        if len(writers) < 2:
            continue
        coupling.append(
            {
                "dst": dest,
                "writers": sorted(writers),
                "writer_count": len(writers),
            }
        )

    # dedup edges
    for kind, lst in edges.items():
        seen = set()
        uniq = []
        for e in lst:
            key = (
                e.get("src"),
                e.get("dst"),
                e.get("line"),
                e.get("call"),
                e.get("form"),
                e.get("direction"),
                e.get("via"),
            )
            if key in seen:
                continue
            seen.add(key)
            uniq.append(e)
        uniq.sort(key=lambda e: (e.get("src") or "", e.get("dst") or "", int(e.get("line") or 0), e.get("call") or ""))
        edges[kind] = uniq

    return edges, coupling


# ---------------------------------------------------------------------------
# Graph findings
# ---------------------------------------------------------------------------


def sccs(nodes: Sequence[str], pairs: Sequence[Tuple[str, str]]) -> List[List[str]]:
    node_set = set(nodes)
    adj: Dict[str, List[str]] = defaultdict(list)
    radj: Dict[str, List[str]] = defaultdict(list)
    self_loops: Set[str] = set()
    for a, b in pairs:
        if a not in node_set or b not in node_set:
            continue
        if a == b:
            self_loops.add(a)
            continue
        adj[a].append(b)
        radj[b].append(a)
    seen: Set[str] = set()
    order: List[str] = []

    def dfs(u: str) -> None:
        seen.add(u)
        for v in adj[u]:
            if v not in seen:
                dfs(v)
        order.append(u)

    for n in nodes:
        if n not in seen:
            dfs(n)
    seen.clear()
    comps: List[List[str]] = []

    def rdfs(u: str, acc: List[str]) -> None:
        seen.add(u)
        acc.append(u)
        for v in radj[u]:
            if v not in seen:
                rdfs(v, acc)

    for u in reversed(order):
        if u not in seen:
            acc: List[str] = []
            rdfs(u, acc)
            comps.append(sorted(acc))
    cycles = [c for c in comps if len(c) > 1]
    for n in sorted(self_loops):
        cycles.append([n])
    cycles.sort(key=lambda c: (len(c), c))
    return cycles


def reachable(roots: Iterable[str], pairs: Sequence[Tuple[str, str]]) -> Set[str]:
    adj: Dict[str, List[str]] = defaultdict(list)
    for a, b in pairs:
        adj[a].append(b)
    seen: Set[str] = set()
    dq = deque(r for r in roots if r)
    while dq:
        u = dq.popleft()
        if u in seen:
            continue
        seen.add(u)
        for v in adj[u]:
            if v not in seen:
                dq.append(v)
    return seen


def findings(
    files: Dict[str, Dict[str, Any]],
    edges: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    nodes = sorted(files)
    facts = {p: files[p].get("_facts") or {} for p in nodes}

    import_pairs = [
        (e["src"], e["dst"])
        for e in edges["import"]
        if e.get("dst_kind") == "census_module" and e["dst"] in files
    ]
    combined_pairs = list(import_pairs)
    for kind in ("subprocess", "runtime", "mutation", "tool"):
        for e in edges[kind]:
            if e.get("dst_kind") == "census_module" and e.get("dst") in files:
                combined_pairs.append((e["src"], e["dst"]))

    in_deg: Dict[str, int] = defaultdict(int)
    out_deg: Dict[str, int] = defaultdict(int)
    in_from: Dict[str, Set[str]] = defaultdict(set)
    out_to: Dict[str, Set[str]] = defaultdict(set)
    in_from_product: Dict[str, Set[str]] = defaultdict(set)
    in_from_nontest: Dict[str, Set[str]] = defaultdict(set)
    for a, b in import_pairs:
        if a == b:
            continue
        out_deg[a] += 1
        in_deg[b] += 1
        out_to[a].add(b)
        in_from[b].add(a)
        if is_product_kind(facts[a].get("kind") or ""):
            in_from_product[b].add(a)
        if not is_test_path(a):
            in_from_nontest[b].add(a)

    import_cycles = sccs(nodes, import_pairs)
    combined_cycles = sccs(nodes, combined_pairs)

    product_entry = []
    harness_entry = []
    for p, f in facts.items():
        kind = f.get("kind")
        entries = f.get("entrypoints") or []
        if kind == "hcli_product" and (
            Path(p).name in {"__main__.py", "cli.py", "app.py"} or "python -m" in entries or "hcli.cli:main" in entries
        ):
            product_entry.append(p)
        if kind == "hcli_fossil" and entries:
            product_entry.append(p)
        if kind == "hcli_product" and Path(p).name == "cli.py":
            product_entry.append(p)
        if p == "hcli/__init__.py":
            product_entry.append(p)
        if kind in {"headless_harness", "headless_test"} and entries:
            harness_entry.append(p)
    product_entry = sorted(set(product_entry))
    harness_entry = sorted(set(harness_entry))
    all_entry = sorted(set(product_entry) | set(harness_entry) | {p for p, f in facts.items() if f.get("entrypoints")})

    seen_from_product = reachable(product_entry, combined_pairs)
    seen_from_any = reachable(all_entry, combined_pairs)

    unreachable_from_product = sorted(
        p
        for p in nodes
        if p not in seen_from_product and is_product_kind(facts[p].get("kind") or "")
    )
    unreachable_from_any = sorted(p for p in nodes if p not in seen_from_any)

    reexport = sorted(p for p, f in facts.items() if f.get("reexport_only"))
    thin = sorted(p for p, f in facts.items() if f.get("thin_entrypoint"))

    one_caller = []
    for p, f in facts.items():
        if is_test_path(p):
            continue
        callers = sorted(in_from_nontest.get(p) or [])
        if len(callers) == 1:
            one_caller.append(
                {
                    "path": p,
                    "kind": f.get("kind"),
                    "caller": callers[0],
                    "reexport_only": bool(f.get("reexport_only")),
                    "lines": f.get("lines"),
                    "outbound_census": sorted(out_to.get(p) or []),
                }
            )
    one_caller.sort(key=lambda r: r["path"])
    WRAPPER_MAX_LINES = 80
    one_caller_wrappers = [
        r
        for r in one_caller
        if r.get("reexport_only") or int(r.get("lines") or 0) <= WRAPPER_MAX_LINES
    ]

    zero_caller_product = [
        {
            "path": p,
            "kind": facts[p].get("kind"),
            "reexport_only": bool(facts[p].get("reexport_only")),
            "entrypoints": facts[p].get("entrypoints") or [],
            "lines": facts[p].get("lines"),
        }
        for p in nodes
        if not is_test_path(p)
        and not (in_from_nontest.get(p))
        and facts[p].get("kind") in {"hcli_product", "hcli_fossil", "headless_harness"}
    ]

    leaf_fossils = []
    for p, f in facts.items():
        kind = f.get("kind")
        if kind not in {"hcli_product", "hcli_fossil"}:
            continue
        if is_test_path(p):
            continue
        inbound = in_from_product.get(p) or set()
        outbound = out_to.get(p) or set()
        name = Path(p).name
        if name == "__init__.py":
            continue
        fossilish = (
            kind == "hcli_fossil"
            or name in {"index.py", "context.py", "haider.py", "p0_tool_bridge.py"}
            or (not inbound and not f.get("entrypoints") and kind == "hcli_product")
        )
        if fossilish:
            leaf_fossils.append(
                {
                    "path": p,
                    "kind": kind,
                    "inbound_product": sorted(inbound),
                    "outbound_census": sorted(outbound),
                    "inbound_nontest": sorted(in_from_nontest.get(p) or []),
                    "reexport_only": bool(f.get("reexport_only")),
                    "entrypoints": f.get("entrypoints") or [],
                    "reason": (
                        "fossil namespace (tools/hcli/bootstrap/*.py, disconnected from hcli package)"
                        if kind == "hcli_fossil"
                        else "product module with no inbound product imports"
                        if not inbound
                        else "named leaf"
                    ),
                }
            )
    leaf_fossils.sort(key=lambda r: r["path"])

    hubs_in = sorted(
        ({"path": p, "in_degree": in_deg[p], "importers": sorted(in_from[p])} for p in nodes if in_deg[p] >= 5),
        key=lambda r: (-r["in_degree"], r["path"]),
    )
    hubs_out = sorted(
        ({"path": p, "out_degree": out_deg[p], "imported": sorted(out_to[p])} for p in nodes if out_deg[p] >= 5),
        key=lambda r: (-r["out_degree"], r["path"]),
    )

    # dual import identity
    forms_hcli = 0
    forms_tools = 0
    for p, f in facts.items():
        for imp in f.get("imports") or []:
            form = imp.get("form") or ""
            if form.startswith("from hcli.") or form.startswith("import hcli"):
                forms_hcli += 1
            if "tools.haider.hcli" in form:
                forms_tools += 1

    llama_spawners = sorted(
        {
            e["src"]
            for e in edges["runtime"] + edges["subprocess"]
            if e.get("dst") == "llama-server"
        }
    )
    grok_spawners = sorted(
        {
            e["src"]
            for e in edges["tool"] + edges["subprocess"]
            if str(e.get("dst") or "").startswith("grok-run") or e.get("dst") == "grok-run"
        }
    )
    sys_path_files = sorted(p for p, f in facts.items() if f.get("sys_path_inserts"))

    hcli_imports_headless = [
        e
        for e in edges["import"]
        if e["src"].startswith("tools/hcli/bootstrap/") and str(e.get("dst") or "").startswith("tools/headless/")
    ]
    outside_imports = []
    for p, f in facts.items():
        for e in f.get("imports_external") or []:
            if e.get("dst_kind") == "repo_outside_census":
                outside_imports.append(e)

    accidental = [
        {
            "id": "dual_import_identity",
            "severity": "high",
            "detail": (
                "The same files are imported as `hcli.*` (sys.path=tools/haider, "
                f"{forms_hcli} statements) and as `tools.haider.hcli.*` "
                f"(sys.path=repo root, {forms_tools} statements). Two dotted identities, "
                "one path. Tests mostly use the latter; harnesses mostly use the former."
            ),
            "hcli_star_statements": forms_hcli,
            "tools_haider_hcli_statements": forms_tools,
        },
        {
            "id": "sys_path_inserts",
            "severity": "medium",
            "detail": "Modules that mutate sys.path to load HCLI. Accidental coupling surface.",
            "files": sys_path_files,
            "count": len(sys_path_files),
        },
        {
            "id": "duplicate_llama_server_spawn",
            "severity": "high" if len(llama_spawners) > 1 else "info",
            "detail": (
                "Modules that spawn or resolve llama-server. More than one product spawn "
                "site is a second runtime authority."
            ),
            "files": llama_spawners,
        },
        {
            "id": "grok_run_spawn",
            "severity": "info",
            "detail": "Modules that invoke grok-run (tool edge import graphs miss).",
            "files": grok_spawners,
        },
        {
            "id": "hcli_imports_headless",
            "severity": "high" if hcli_imports_headless else "info",
            "detail": "Control-plane modules importing harnesses — that would invert the layering.",
            "edges": hcli_imports_headless,
        },
        {
            "id": "repo_outside_census_imports",
            "severity": "medium" if outside_imports else "info",
            "detail": "Census modules importing repo Python outside tools/haider + tools/headless.",
            "edges": [
                {"src": e["src"], "dst": e["dst"], "line": e["line"], "form": e.get("form")}
                for e in outside_imports
            ],
        },
        {
            "id": "fossil_haider_disconnected",
            "severity": "info",
            "detail": (
                "tools/hcli/bootstrap/snapshots/haider.py and p0_tool_bridge.py are a separate process "
                "from the hcli package. Standing fact: haider is a fossil namespace. "
                "The graph must show them disconnected from hcli.*, not rename them."
            ),
            "files": [p for p in nodes if p.startswith("tools/hcli/bootstrap/") and not p.startswith("hcli/")],
        },
        {
            "id": "lab_hcli_sibling",
            "severity": "medium",
            "detail": (
                "research/lab/hcli/ exists in git as a sibling namespace and was not parsed "
                "(out of census, and not materialized in this sparse worktree). "
                "UNKNOWN whether it still imports or duplicates hcli."
            ),
            "path": "research/lab/hcli/",
        },
        {
            "id": "duplicate_mutation_authority",
            "severity": "high",
            "detail": (
                "hcli/mutation.py (apply_mutation_operations) has no inbound "
                "import from any hcli product module. Engine implements mutation itself "
                "via _atomic_write_text. Two mutation authorities; tests and "
                "hcli_persistence_audit.py are the only census importers of mutation.py."
            ),
            "mutation_module": "hcli/mutation.py",
            "engine_module": "hcli/engine.py",
            "product_importers": sorted(in_from_product.get("hcli/mutation.py") or []),
            "nontest_importers": sorted(in_from_nontest.get("hcli/mutation.py") or []),
        },
    ]

    # shared persistence hubs
    via_counts: Dict[str, Set[str]] = defaultdict(set)
    for e in edges["persistence"]:
        if e.get("dst_kind") != "path":
            continue
        if e.get("direction") != "write":
            continue
        dest = str(e.get("dst") or "")
        if dest.startswith("symbolic:"):
            continue
        path_like = (
            "/" in dest
            or dest.startswith("~")
            or dest.startswith(".")
            or dest.endswith((".json", ".jsonl", ".log", ".lock", ".md", ".txt", ".py"))
        )
        if not path_like:
            continue
        via_counts[dest].add(e["src"])
    persist_hubs = [
        {"dest": d, "modules": sorted(ms), "module_count": len(ms)}
        for d, ms in sorted(via_counts.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        if len(ms) >= 2
    ]

    return {
        "import_cycles": import_cycles,
        "combined_cycles": combined_cycles,
        "product_entrypoints": product_entry,
        "harness_entrypoints": harness_entry,
        "all_entrypoints": all_entry,
        "unreachable_from_product_entrypoints": unreachable_from_product,
        "unreachable_from_any_entrypoint": unreachable_from_any,
        "reexport_only_modules": reexport,
        "thin_entrypoints": thin,
        "one_caller_modules": one_caller,
        "one_caller_wrappers": one_caller_wrappers,
        "zero_caller_nontest": zero_caller_product,
        "leaf_fossils": leaf_fossils,
        "giant_hubs": {"by_in_degree": hubs_in[:15], "by_out_degree": hubs_out[:15]},
        "accidental_coupling": accidental,
        "shared_persistence_hubs": persist_hubs,
        "in_degree": {p: in_deg[p] for p in nodes if in_deg[p]},
        "out_degree": {p: out_deg[p] for p in nodes if out_deg[p]},
    }


# ---------------------------------------------------------------------------
# Watch things fail (honest, executed)
# ---------------------------------------------------------------------------


def watch_failures(files: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    controller = REPO / "hcli/controller.py"
    try:
        controller.read_text(encoding="utf-8")
        rows.append(
            {
                "what": "read hcli/controller.py from disk",
                "result": "UNEXPECTED_OK",
                "reason": "file is on disk in this worktree; sparse hole was expected",
            }
        )
    except FileNotFoundError:
        rows.append(
            {
                "what": "read hcli/controller.py from disk",
                "result": "FAIL",
                "reason": "FileNotFoundError — sparse hole. git show is the authority here.",
            }
        )

    blob = git_show("hcli/controller.py")
    rows.append(
        {
            "what": "git show HEAD:hcli/controller.py",
            "result": "OK" if blob else "FAIL",
            "reason": f"{len(blob)} bytes" if blob else "empty/nonzero",
        }
    )

    try:
        import hcli  # type: ignore  # noqa: F401
        rows.append(
            {
                "what": "import hcli (no sys.path hack)",
                "result": "UNEXPECTED_OK",
                "reason": str(getattr(hcli, "__file__", None)),
            }
        )
    except Exception as exc:
        rows.append(
            {
                "what": "import hcli (no sys.path hack)",
                "result": "FAIL",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )

    # aider must not be required; record whether the fossil binary exists
    which = subprocess.run(["bash", "-lc", "command -v aider || true"], capture_output=True, text=True)
    rows.append(
        {
            "what": "command -v aider",
            "result": "PRESENT" if (which.stdout or "").strip() else "ABSENT",
            "reason": (which.stdout or "").strip() or "aider not on PATH in this shell",
        }
    )

    # prove we did not treat a sparse hole as absence of haider modules
    haider_n = sum(1 for p in files if p.startswith("tools/hcli/bootstrap/"))
    haider_disk = sum(1 for p in files if p.startswith("tools/hcli/bootstrap/") and files[p]["on_disk"])
    rows.append(
        {
            "what": "haider module presence under sparse checkout",
            "result": "OK" if haider_n and haider_disk == 0 else ("MIXED" if haider_disk else "FAIL"),
            "reason": f"parsed {haider_n} haider modules; on_disk={haider_disk} (0 is the sparse-hole case)",
        }
    )
    return rows


def strip_private(files: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    modules = []
    for p in sorted(files):
        f = files[p].get("_facts") or {}
        modules.append(
            {
                "path": p,
                "kind": f.get("kind"),
                "origin": f.get("origin"),
                "sha256": f.get("sha256"),
                "bytes": f.get("bytes"),
                "lines": f.get("lines"),
                "in_git": f.get("in_git"),
                "on_disk": f.get("on_disk"),
                "parse_error": f.get("parse_error"),
                "entrypoints": f.get("entrypoints") or [],
                "reexport_only": bool(f.get("reexport_only")),
                "thin_entrypoint": bool(f.get("thin_entrypoint")),
                "shebang": bool(f.get("shebang")),
                "dotted_names": f.get("dotted_names") or [],
                "sys_path_inserts": f.get("sys_path_inserts") or [],
                "slash_commands": f.get("slash_commands") or [],
                "named_tools": f.get("named_tools") or [],
                "import_count": len(f.get("imports") or []),
                "imports_external": f.get("imports_external") or [],
            }
        )
    return modules


def path_resolution_check(
    files: Dict[str, Dict[str, Any]],
    edges: Dict[str, List[Dict[str, Any]]],
    git_all: Sequence[str],
) -> List[Dict[str, Any]]:
    git_set = set(git_all)
    problems = []
    for p, rec in files.items():
        if rec["on_disk"]:
            if not (REPO / p).is_file():
                problems.append({"path": p, "problem": "on_disk flag set but file missing"})
            continue
        if rec["in_git"] and p not in git_set and not p.endswith("code_graph.py"):
            problems.append({"path": p, "problem": "in_git flag set but not in git ls-tree"})
        if rec["origin"] == "git_sparse":
            if git_show(p) is None:
                problems.append({"path": p, "problem": "git_sparse origin but git show failed"})
    for kind, lst in edges.items():
        for e in lst:
            src = e.get("src")
            if src and src not in files:
                problems.append({"path": src, "problem": f"{kind} edge src not in census"})
            dst = e.get("dst")
            if e.get("dst_kind") == "census_module" and dst not in files:
                # allow __main__ if present
                problems.append(
                    {
                        "path": str(dst),
                        "problem": f"{kind} edge dst_kind=census_module but dst not in census (src={src})",
                    }
                )
    return problems


def atomic_write_json(path: Path, obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(str(tmp), str(path))
    return sha256_bytes(data.encode("utf-8"))


def print_report(graph: Dict[str, Any]) -> None:
    c = graph["counts"]
    print(f"schema: {graph['schema']}")
    print(f"wrote: receipts/headless/CODE_GRAPH.json")
    print(f"sha256: {graph['receipt_sha256']}")
    print(f"git_head: {graph['git_head']}")
    print(f"census_roots: {', '.join(graph['census_roots'])}")
    print()
    print("counts:")
    for k in (
        "modules",
        "import_edges",
        "subprocess_edges",
        "persistence_edges",
        "tool_edges",
        "runtime_edges",
        "mutation_edges",
        "cycles",
        "combined_cycles",
        "entrypoints",
        "one_caller_wrappers",
        "one_caller_modules",
        "reexport_only_modules",
        "unreachable_from_product",
        "leaf_fossils",
        "unknowns",
    ):
        print(f"  {k}: {c[k]}")
    print()
    print("cycles (import graph):")
    cycles = graph["findings"]["import_cycles"]
    if not cycles:
        print("  (none)")
    else:
        for cyc in cycles:
            print("  - " + " -> ".join(cyc + [cyc[0]]))
    print()
    print("combined cycles (import+subprocess+runtime+mutation+tool module edges):")
    cc = graph["findings"]["combined_cycles"]
    if not cc:
        print("  (none)")
    else:
        for cyc in cc:
            print("  - " + " -> ".join(cyc + [cyc[0]]))
    print()
    print("entrypoints (product):")
    for p in graph["findings"]["product_entrypoints"]:
        print(f"  - {p}")
    print()
    print("reexport_only_modules:")
    for p in graph["findings"]["reexport_only_modules"]:
        print(f"  - {p}")
    print()
    print("one_caller_wrappers (reexport or <=80 lines, exactly one non-test caller):")
    oc = graph["findings"]["one_caller_wrappers"]
    if not oc:
        print("  (none)")
    else:
        for row in oc:
            extra = " reexport" if row.get("reexport_only") else ""
            print(f"  - {row['path']}  caller={row['caller']}  lines={row['lines']}{extra}")
    print()
    print("one_caller_modules (exactly one non-test caller, any size):")
    om = graph["findings"].get("one_caller_modules") or []
    if not om:
        print("  (none)")
    else:
        for row in om:
            print(f"  - {row['path']}  caller={row['caller']}  lines={row['lines']}")
    print()
    print("leaf_fossils:")
    for row in graph["findings"]["leaf_fossils"]:
        print(f"  - {row['path']}  ({row['reason']})")
    print()
    print("WHAT I WATCHED FAIL")
    for row in graph["what_i_watched_fail"]:
        print(f"  [{row['result']}] {row['what']}: {row['reason']}")
    print()
    notes = graph.get("edge_class_notes") or {}
    print("edge class notes:")
    for k, v in notes.items():
        print(f"  {k}: {v}")


def main() -> int:
    os.chdir(str(REPO))
    files, inventory, watched_discover = discover()
    path_index, dotted_index = build_indexes(files)
    extractor = Extractor(files, path_index, dotted_index)
    for rel, rec in sorted(files.items()):
        extractor.extract(rel, rec)
    edges, _coupling = extract_edges(files, extractor)
    found = findings(files, edges)
    watched = watched_discover + watch_failures(files)

    git_all = git_ls_tree_all_py()
    resolve_problems = path_resolution_check(files, edges, git_all)
    for prob in resolve_problems:
        extractor.note_unknown(prob["path"], "path_resolution", prob["problem"])

    # Drop bulky per-module source
    modules = strip_private(files)

    edge_notes = {}
    for kind in ("import", "subprocess", "persistence", "tool", "runtime", "mutation"):
        n = len(edges[kind])
        if n == 0:
            edge_notes[kind] = (
                f"EMPTY — walker found zero {kind} edges. "
                "If this class should exist, the extractor missed a call shape."
            )
        else:
            edge_notes[kind] = f"populated ({n})"

    counts = {
        "modules": len(modules),
        "import_edges": len(edges["import"]),
        "subprocess_edges": len(edges["subprocess"]),
        "persistence_edges": len(edges["persistence"]),
        "tool_edges": len(edges["tool"]),
        "runtime_edges": len(edges["runtime"]),
        "mutation_edges": len(edges["mutation"]),
        "cycles": len(found["import_cycles"]),
        "combined_cycles": len(found["combined_cycles"]),
        "entrypoints": len(found["all_entrypoints"]),
        "product_entrypoints": len(found["product_entrypoints"]),
        "one_caller_wrappers": len(found["one_caller_wrappers"]),
        "one_caller_modules": len(found["one_caller_modules"]),
        "reexport_only_modules": len(found["reexport_only_modules"]),
        "unreachable_from_product": len(found["unreachable_from_product_entrypoints"]),
        "leaf_fossils": len(found["leaf_fossils"]),
        "unknowns": len(extractor.unknowns),
        "path_resolution_problems": len(resolve_problems),
    }

    graph: Dict[str, Any] = {
        "schema": SCHEMA,
        "git_head": git_head(),
        "census_roots": list(CENSUS_ROOTS),
        "scope": {
            "includes": [
                "tools/hcli/bootstrap/**/*.py — live HCLI control plane plus fossil wrappers (read via git when sparse-missing)",
                "tools/headless/**/*.py — harnesses and headless tests",
            ],
            "excludes": [
                "research/lab/, research/ramanujan/, workspace/, visionmcp/, app/ — out of this lane's census",
                "tools/condense, tools/graph, tools/odyssey, tools/gravity_*.py — out of census; outbound edges to them are recorded as repo_outside_census",
                "historical receipts under receipts/ — evidence, not code; dests are persistence targets, never rewritten here",
            ],
            "sparse_checkout": (
                "A path missing on disk is not evidence it does not exist. "
                "This walker uses git ls-tree + git show for holes."
            ),
        },
        "inventory": {
            "git_python_files": inventory["git_python_files"],
            "git_python_by_top_level": inventory["git_python_by_top_level"],
            "census_in_git": inventory["census_in_git"],
            "census_on_disk": inventory["census_on_disk"],
            "census_union": inventory["census_union"],
            "out_of_census_python": inventory["out_of_census_python"],
            "sibling_namespaces_not_parsed": {
                "research/lab/hcli": inventory["sibling_namespaces_not_parsed"]["research/lab/hcli"],
                "note": inventory["sibling_namespaces_not_parsed"]["note"],
            },
        },
        "counts": counts,
        "edge_class_notes": edge_notes,
        "findings": {
            "import_cycles": found["import_cycles"],
            "combined_cycles": found["combined_cycles"],
            "product_entrypoints": found["product_entrypoints"],
            "harness_entrypoints": found["harness_entrypoints"],
            "all_entrypoints": found["all_entrypoints"],
            "unreachable_from_product_entrypoints": found["unreachable_from_product_entrypoints"],
            "unreachable_from_any_entrypoint": found["unreachable_from_any_entrypoint"],
            "reexport_only_modules": found["reexport_only_modules"],
            "thin_entrypoints": found["thin_entrypoints"],
            "one_caller_modules": found["one_caller_modules"],
            "one_caller_wrappers": found["one_caller_wrappers"],
            "zero_caller_nontest": found["zero_caller_nontest"],
            "leaf_fossils": found["leaf_fossils"],
            "giant_hubs": found["giant_hubs"],
            "accidental_coupling": found["accidental_coupling"],
            "shared_persistence_hubs": found["shared_persistence_hubs"],
        },
        "unknowns": extractor.unknowns,
        "path_resolution_problems": resolve_problems,
        "what_i_watched_fail": watched,
        "modules": modules,
        "edges": edges,
        "in_degree": found["in_degree"],
        "out_degree": found["out_degree"],
    }

    digest = atomic_write_json(RECEIPT, graph)
    # receipt_sha256 is of the file including everything except we need it inside...
    # Put it in after write would break determinism of the file vs printed digest.
    # Store the digest of the graph WITHOUT receipt_sha256, then rewrite once with it?
    # Acceptance: run twice, identical output. So the file must not contain a
    # field that depends on its own bytes. Print sha256 of the written file
    # after the fact; do not embed it.
    graph_for_print = dict(graph)
    graph_for_print["receipt_sha256"] = digest
    print_report(graph_for_print)
    return 0


if __name__ == "__main__":
    sys.exit(main())
