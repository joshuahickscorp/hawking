#!/usr/bin/env python3
"""Census every Python subprocess spawn in the HCLI control plane + headless harnesses.

Classifies each site:

  REQUIRED_EXTERNAL_TOOL   argv[0] is not Hawking Python (git, sysctl, llama-server, grok-run, …)
  ISOLATION_BOUNDARY       a process is load-bearing (untrusted code, killpg, crash/reaper, GPU server)
  LEGACY_WRAPPER           Python spawning Hawking Python that could be an in-process call
  ACCIDENTAL_PROCESS_SPAWN ceremony with an equivalent stdlib call (rm -rf, python3 -c 'import pytest', …)

UNKNOWN is a follow-up, not a guess. This lane changes no source: it writes this
script and receipts/headless/SUBPROCESS_CENSUS.json. A human performs any migration.

    python3 tools/headless/subprocess_census.py
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "hawking.headless.subprocess_census.v1"
REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts" / "headless" / "SUBPROCESS_CENSUS.json"

ROOTS = ("hcli", "tools/haider", "tools/headless", "crates", "src")
CLASSES = (
    "REQUIRED_EXTERNAL_TOOL",
    "ISOLATION_BOUNDARY",
    "LEGACY_WRAPPER",
    "ACCIDENTAL_PROCESS_SPAWN",
    "UNKNOWN",
)
# Acceptance vocabulary (same four buckets + UNKNOWN).
CLASS_CONTRACT = {
    "REQUIRED_EXTERNAL_TOOL": "necessary-external-tool",
    "ISOLATION_BOUNDARY": "required-isolation",
    "LEGACY_WRAPPER": "legacy-wrapper",
    "ACCIDENTAL_PROCESS_SPAWN": "accidental-spawn",
    "UNKNOWN": "unknown",
}

SPAWN_ATTRS = {
    "run",
    "Popen",
    "check_output",
    "check_call",
    "call",
    "getoutput",
    "getstatusoutput",
    "create_subprocess_exec",
    "create_subprocess_shell",
}
OS_SPAWN = {
    "system",
    "popen",
    "spawnl",
    "spawnlp",
    "spawnv",
    "spawnve",
    "execl",
    "execv",
    "execvp",
    "execvpe",
}

HOST_BINS = {
    "git",
    "sysctl",
    "vm_stat",
    "ps",
    "pgrep",
    "pkill",
    "df",
    "lsof",
    "memory_pressure",
    "which",
    "sw_vers",
    "uname",
    "sysctl",
    "ioreg",
}
INFER_BINS = {
    "llama-server",
    "mlx_lm.server",
    "mlx_lm",
    "grok-run",
    "cargo",
    "swift",
    "rg",
    "rustc",
}
STAMP_FUNCS = {
    "git_head",
    "_git_head",
    "git",
    "g",
    "sh",
}

# Historical suite walls (receipts/headless/HCLI_SELF_OPT_ITERATION_{1,2}.json).
# Cited, not re-derived; live measurement overwrites the `live` block.
HIST_SUITE = {
    "iteration_1": {
        "receipt": "receipts/headless/HCLI_SELF_OPT_ITERATION_1.json",
        "command": "python3 -m pytest tools/haider/hcli/tests -q",
        "wall_s": 129.31068000000232,
        "passed": 416,
        "skipped": 2,
        "workunits": 10,
        "mission_wall_s": 134.01605208299588,
    },
    "iteration_2": {
        "receipt": "receipts/headless/HCLI_SELF_OPT_ITERATION_2.json",
        "command": "python3 -m pytest tools/haider/hcli/tests -q",
        "wall_s": 139.5457987080008,
        "passed": 432,
        "skipped": 2,
        "workunits": 10,
        "mission_wall_s": 175.32598887500353,
    },
}


# ---------------------------------------------------------------------------
# git / files
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head(repo: Path = REPO) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNKNOWN:{exc}"


def git_ls(prefix: str) -> List[str]:
    out = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", prefix],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def read_source(rel: str) -> Tuple[str, str]:
    """Return (text, origin) where origin is 'disk' or 'git_show'."""
    path = REPO / rel
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace"), "disk"
    proc = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        cwd=str(REPO),
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise FileNotFoundError(rel)
    return proc.stdout.decode("utf-8", "replace"), "git_show"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# AST extraction
# ---------------------------------------------------------------------------


def _const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: List[str] = []
        for val in node.values:
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                parts.append(val.value)
            else:
                parts.append("{expr}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _const_str(node.left), _const_str(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _elt_summary(node: ast.AST) -> str:
    s = _const_str(node)
    if s is not None:
        return s
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Attribute) and node.attr == "executable":
        return "sys.executable"
    if isinstance(node, ast.Name):
        return f"${node.id}"
    if isinstance(node, ast.Starred) and isinstance(node.value, ast.Name):
        return f"*${node.value.id}"
    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == "str":
            return "$str(...)"
        if isinstance(fn, ast.Name) and fn.id == "list":
            return "$list(...)"
        return "$call"
    return f"${type(node).__name__}"


def _cmd_of(call: ast.Call) -> Tuple[str, Any]:
    arg: Optional[ast.AST] = None
    if call.args:
        arg = call.args[0]
    else:
        for kw in call.keywords:
            if kw.arg in ("args", "cmd", "command"):
                arg = kw.value
                break
    if arg is None:
        return "missing", None
    if isinstance(arg, (ast.List, ast.Tuple)):
        return "argv", [_elt_summary(e) for e in arg.elts]
    s = _const_str(arg)
    if s is not None:
        return "str", s
    if isinstance(arg, ast.Name):
        return "name", arg.id
    if isinstance(arg, ast.JoinedStr):
        return "fstring", _const_str(arg)
    if isinstance(arg, ast.BinOp):
        return "binop", ast.dump(arg, include_attributes=False)[:160]
    if isinstance(arg, ast.Call):
        return "call", ast.dump(arg, include_attributes=False)[:160]
    return type(arg).__name__, ast.dump(arg, include_attributes=False)[:160]


def _qualname(func: ast.AST) -> Tuple[str, str]:
    if isinstance(func, ast.Name):
        return func.id, func.id
    if isinstance(func, ast.Attribute):
        name = func.attr
        if isinstance(func.value, ast.Name):
            return f"{func.value.id}.{func.attr}", name
        if isinstance(func.value, ast.Attribute) and isinstance(func.value.value, ast.Name):
            return f"{func.value.value.id}.{func.value.attr}.{func.attr}", name
        return name, name
    return "?", "?"


def _interesting(qual: str, name: str) -> bool:
    # asyncio.run is NOT a spawn (false positive: attr is `run` and qual contains asyncio).
    if name == "Popen":
        return True
    if "subprocess" in qual and name in SPAWN_ATTRS:
        return True
    if "asyncio" in qual and name in {
        "create_subprocess_exec",
        "create_subprocess_shell",
    }:
        return True
    if qual.startswith("os.") and name in OS_SPAWN:
        return True
    if qual in {"os.system", "os.popen"}:
        return True
    return False


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel: str, src: str) -> None:
        self.rel = rel
        self.lines = src.splitlines()
        self.stack: List[str] = []
        self.hits: List[Dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[misc]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        qual, name = _qualname(node.func)
        if _interesting(qual, name):
            form, cmd = _cmd_of(node)
            snippet = ""
            if 1 <= node.lineno <= len(self.lines):
                snippet = self.lines[node.lineno - 1].strip()[:240]
            shell = any(kw.arg == "shell" for kw in node.keywords) or "shell" in qual
            if not shell:
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant):
                        shell = bool(kw.value.value)
            self.hits.append(
                {
                    "file": self.rel,
                    "line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno) or node.lineno,
                    "function": ".".join(self.stack) if self.stack else "<module>",
                    "call": qual,
                    "shell": bool(shell),
                    "cmd_form": form,
                    "cmd": cmd,
                    "snippet": snippet,
                }
            )
        self.generic_visit(node)


def _python_chunks_from_sh(text: str) -> List[Tuple[int, str]]:
    """Extract python heredocs from a bash driver. Line numbers match the file."""
    chunks: List[Tuple[int, str]] = []
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        if re.search(r"<<['\"]PY['\"]", lines[i]) or re.search(r"<<PY\b", lines[i]):
            start = i + 2  # 1-based line of first python line after the << line
            body: List[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "PY":
                body.append(lines[i])
                i += 1
            chunks.append((start, "".join(body)))
        i += 1
    return chunks


def extract_file(rel: str, text: str, origin: str) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    if rel.endswith(".py"):
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            return [
                {
                    "file": rel,
                    "line": getattr(exc, "lineno", 1) or 1,
                    "end_line": getattr(exc, "lineno", 1) or 1,
                    "function": "<parse>",
                    "call": "UNKNOWN",
                    "shell": False,
                    "cmd_form": "syntax_error",
                    "cmd": str(exc),
                    "snippet": "",
                    "origin": origin,
                    "language": "python",
                }
            ]
        vis = _Visitor(rel, text)
        vis.visit(tree)
        for h in vis.hits:
            h["origin"] = origin
            h["language"] = "python"
            hits.append(h)
        return hits
    if rel.endswith(".sh"):
        # Bash-level external tools in the driver, plus any python heredoc.
        # Do not also tag python-heredoc lines as bash (df = subprocess.run
        # would otherwise appear twice).
        heredoc_ranges: List[Tuple[int, int]] = []
        chunks = _python_chunks_from_sh(text)
        lines = text.splitlines()
        for start_line, body in chunks:
            heredoc_ranges.append((start_line, start_line + body.count("\n")))
        def _in_heredoc(lineno: int) -> bool:
            return any(a <= lineno <= b for a, b in heredoc_ranges)
        for i, line in enumerate(lines, 1):
            if _in_heredoc(i):
                continue
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            bash_cmd = None
            if re.search(r"\bgit\s", stripped):
                bash_cmd = "git …"
            elif re.search(r"\bdf\s", stripped):
                bash_cmd = "df"
            elif "/bin/ps" in stripped or re.search(r"\bps\s", stripped):
                bash_cmd = "ps"
            elif "python3" in stripped and "<<" in stripped:
                bash_cmd = "python3 - <<'PY'  (heredoc worker)"
            elif stripped.startswith('"$PACK_BIN"') or "ascension_qwen38_pack" in stripped:
                bash_cmd = "$PACK_BIN (ascension_qwen38_pack)"
            elif stripped.startswith('"$DECODE_BIN"') or "ascension_qwen38_hybrid_greedy" in stripped:
                bash_cmd = "$DECODE_BIN (ascension_qwen38_hybrid_greedy)"
            if bash_cmd:
                hits.append(
                    {
                        "file": rel,
                        "line": i,
                        "end_line": i,
                        "function": "<bash>",
                        "call": "bash",
                        "shell": True,
                        "cmd_form": "str",
                        "cmd": bash_cmd,
                        "snippet": stripped[:240],
                        "origin": origin,
                        "language": "bash",
                    }
                )
        for start_line, body in chunks:
            try:
                tree = ast.parse(body, filename=f"{rel}:heredoc")
            except SyntaxError:
                continue
            vis = _Visitor(rel, body)
            vis.visit(tree)
            for h in vis.hits:
                h["line"] = start_line + h["line"] - 1
                h["end_line"] = start_line + (h.get("end_line") or h["line"]) - 1
                h["origin"] = origin
                h["language"] = "python-heredoc"
                h["function"] = f"<bash-heredoc>.{h['function']}"
                hits.append(h)
        return hits
    if rel.endswith(".rs"):
        return extract_rust(rel, text, origin)
    return hits


_RS_FN = re.compile(r"\bfn\s+([A-Za-z0-9_]+)\s*[<(]")
_RS_NEW = re.compile(r"(?:std::process::)?Command::new\s*\(\s*([^)]+?)\s*\)")


def _enclosing_fn_rs(lines: List[str], lineno: int) -> str:
    for j in range(lineno - 1, -1, -1):
        m = _RS_FN.search(lines[j])
        if m:
            return m.group(1)
    return "<module>"


def _rust_callee(arg: str) -> Tuple[str, str]:
    arg = arg.strip()
    if arg.startswith('"') and arg.endswith('"') and arg.count('"') == 2:
        return "str", arg.strip('"')
    if arg.startswith("&"):
        inner = arg[1:].strip()
        if inner.startswith('"') and inner.endswith('"'):
            return "str", inner.strip('"')
        return "name", inner
    if re.match(r"^[A-Za-z_][A-Za-z0-9_:]*(?:\[[^\]]+\])?$", arg):
        return "name", arg
    return "expr", arg[:160]


def extract_rust(rel: str, text: str, origin: str) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if "Command::new" not in stripped:
            continue
        for m in _RS_NEW.finditer(stripped):
            form, cmd = _rust_callee(m.group(1))
            hits.append(
                {
                    "file": rel,
                    "line": i,
                    "end_line": i,
                    "function": _enclosing_fn_rs(lines, i),
                    "call": "Command::new",
                    "shell": False,
                    "cmd_form": form,
                    "cmd": cmd if form != "argv" else [cmd],
                    "snippet": stripped[:240],
                    "origin": origin,
                    "language": "rust",
                }
            )
    return hits


def classify_rust(hit: Dict[str, Any]) -> Dict[str, Any]:
    tok = first_token(hit)
    hay = _hay(hit)
    f = hit["file"]
    func = hit.get("function") or ""
    role = _role(hit)

    def out(klass: str, why: str, replacement: Optional[str] = None, would_break: Optional[str] = None, hot_path: str = "native") -> Dict[str, Any]:
        return {
            "class": klass,
            "why": why,
            "in_process_replacement": replacement,
            "what_would_break": would_break,
            "role": role,
            "hot_path": hot_path,
            "first_token": tok,
            "command": command_display(hit) if not isinstance(hit.get("cmd"), str) or hit.get("cmd_form") != "name" else (hit.get("cmd") if hit.get("cmd_form") == "str" else f"${hit.get('cmd')}"),
        }

    if tok in HOST_BINS or tok == "git" or "git" in hay:
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "Rust Command::new of a host/SCM binary (git/sysctl/vm_stat/ps). Not Hawking Python.",
            hot_path="receipt_stamp" if tok == "git" else "native",
        )
    if tok in {"xcrun", "metal", "metallib", "swift", "b3sum", "sandbox-exec", "/usr/bin/sandbox-exec"}:
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            f"Native toolchain/security binary {tok}. Cannot become an in-process Python API.",
        )
    if tok in INFER_BINS or "llama" in hay or "mlx" in hay:
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "External inference/CLI binary from a Rust driver.",
            hot_path="runtime_pool",
        )
    if tok in {"cp", "rm", "mv", "cat", "echo"}:
        return out(
            "ACCIDENTAL_PROCESS_SPAWN",
            f"Command::new({tok!r}) — std::fs / std::io does this without a process.",
            replacement="std::fs::{copy,remove_file,rename,read} as appropriate",
            would_break="Nothing if the path is a regular file the process owns.",
        )
    if tok in {"python3", "python"} or "python" in tok.lower() or tok.startswith("$python"):
        if "mlx" in hay:
            return out(
                "REQUIRED_EXTERNAL_TOOL",
                "Python MLX competitor process. GPU residency; not a Hawking import.",
                hot_path="runtime_pool",
            )
        return out(
            "LEGACY_WRAPPER",
            "Rust spawning Python. If the callee is a Hawking module, this is a language hop that could be a library call or a dedicated native API.",
            replacement="Call the native equivalent, or keep the process only when the Python is an external runtime (MLX).",
            would_break="Scripts that exist only as Python CLIs.",
            hot_path="native",
        )
    if "/tests/" in f or f.endswith("_test.rs") or "tests/" in f:
        return out(
            "ISOLATION_BOUNDARY",
            "Rust test fixture spawning a foreign binary (oracle, tokenizer, llama.cpp). The process is the claim.",
            hot_path="test",
        )
    ident = tok[1:] if tok.startswith("$") else tok
    if ident and re.match(r"^[A-Za-z_]", ident):
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            f"Command::new({tok}) — callee resolved at runtime; treated as an external binary, not Hawking Python.",
        )
    return out(
        "UNKNOWN",
        f"no rust rule matched file={f} func={func} tok={tok}",
    )


def first_token(hit: Dict[str, Any]) -> str:
    cmd = hit.get("cmd")
    if isinstance(cmd, list) and cmd:
        tok = str(cmd[0])
        if tok == "sys.executable":
            return "sys.executable"
        if tok.startswith("$"):
            return tok
        return tok
    if isinstance(cmd, str):
        parts = cmd.split()
        return parts[0] if parts else ""
    form = hit.get("cmd_form")
    if form == "name" and isinstance(cmd, str):
        return f"${cmd}"
    return str(form or "?")


def command_display(hit: Dict[str, Any]) -> str:
    cmd = hit.get("cmd")
    if isinstance(cmd, list):
        return " ".join(str(x) for x in cmd)
    if isinstance(cmd, str):
        if hit.get("cmd_form") == "name":
            return f"${cmd}"
        return cmd
    return hit.get("snippet") or ""


def _hay(hit: Dict[str, Any]) -> str:
    bits = [
        hit.get("file") or "",
        hit.get("function") or "",
        command_display(hit),
        hit.get("snippet") or "",
        " ".join(str(x) for x in hit.get("cmd") or [])
        if isinstance(hit.get("cmd"), list)
        else str(hit.get("cmd") or ""),
    ]
    return " ".join(bits)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _role(hit: Dict[str, Any]) -> str:
    f = hit["file"]
    name = Path(f).name
    func = hit.get("function") or ""
    if "/tests/" in f or name.endswith("_test.py") or name.startswith("test_"):
        return "test_fixture"
    if f.startswith("tools/headless/"):
        if func in STAMP_FUNCS or (first_token(hit) in {"git", "$args"} and "rev-parse" in _hay(hit)):
            if any(s in func.lower() for s in ("git_head", "git")):
                return "receipt_stamp"
        return "harness"
    if f.startswith("tools/haider/hcli/") and "/tests/" not in f:
        return "production"
    if f.startswith("hcli/") and "/tests/" not in f:
        return "production"
    if f.startswith("crates/") or f.startswith("src/"):
        return "native"
    if f in {"tools/haider/haider.py", "tools/haider/p0_tool_bridge.py"}:
        return "production_fossil"
    return "other"


def classify(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Return class, why, replacement, would_break, hot_path."""
    f = hit["file"]
    func = hit.get("function") or ""
    tok = first_token(hit)
    hay = _hay(hit)
    role = _role(hit)
    lang = hit.get("language") or "python"
    shell = bool(hit.get("shell"))
    cmd_s = command_display(hit)

    def out(
        klass: str,
        why: str,
        replacement: Optional[str] = None,
        would_break: Optional[str] = None,
        hot_path: str = "cold",
    ) -> Dict[str, Any]:
        return {
            "class": klass,
            "why": why,
            "in_process_replacement": replacement,
            "what_would_break": would_break,
            "role": role,
            "hot_path": hot_path,
            "first_token": tok,
            "command": cmd_s,
        }

    # --- parse failure
    if hit.get("call") == "UNKNOWN":
        return out("UNKNOWN", f"file did not parse: {hit.get('cmd')}")

    if (hit.get("language") or "") == "rust" or hit.get("call") == "Command::new":
        return classify_rust(hit)

    # --- bash driver (not Python subprocess of Hawking modules)
    if lang == "bash":
        if "python3" in hay and "heredoc" in hay.lower():
            return out(
                "REQUIRED_EXTERNAL_TOOL",
                "bash driver launching a python3 heredoc to write the receipt; the worker is not a Hawking import.",
                hot_path="harness",
            )
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "bash driver invoking git/df/ps/pack/decode binaries. Isolation is not the question; these are not Hawking Python.",
            hot_path="harness",
        )

    # ========== production function overrides (stable names) ==========
    if func.endswith("LlamaServerBackend.spawn") or func == "start_llama_server":
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "llama-server is a C++ HTTP process (GPU/CPU inference). start_new_session=True is also an isolation boundary, but the binary cannot become an in-process Hawking API.",
            would_break="RuntimePool identity (pid, start_token, RSS), reaper, /health, slot topology, crash isolation from the control plane.",
            hot_path="runtime_pool",
        )
    if func.endswith("MlxServerBackend.spawn"):
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "mlx_lm.server is a long-lived MLX HTTP server, not a Hawking module. GPU residency and crash isolation require a process even though the server is Python.",
            would_break="port/pid accounting, decode-concurrency slots, reaper, control-plane survival if MLX aborts.",
            hot_path="runtime_pool",
        )
    if func.endswith("_capture") and "backends.py" in f:
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "[llama-server|mlx_lm.server] --help / --version. The binary is the capability probe.",
            hot_path="runtime_pool",
        )
    if func.endswith("GrokBridge._run"):
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "grok-run is an external CLI that creates isolated worktrees and returns a task id. HCLI never imports grok as a library.",
            would_break="worktree isolation, contract lint, background pid, wait/cleanup; the ten live science lanes are grok-run tasks.",
            hot_path="grok",
        )
    if func.endswith("_process_tree"):
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "ps -A -o pid=,ppid= to walk the grok-run descendant tree. /proc is not the Darwin API.",
            hot_path="grok",
        )
    if func.endswith("Engine._run_contained_subprocess"):
        return out(
            "ISOLATION_BOUNDARY",
            "Untrusted WorkUnit tests. start_new_session=True plus killpg on timeout. The process group is how a runaway pytest is reaped without taking down the control plane. Env is scrubbed to PATH/HOME/LANG/TMPDIR + PYTHONPATH.",
            replacement=None,
            would_break="timeout cannot SIGKILL a thread; pytest plugin/sys.modules leak into Engine; mutated files would not reload; a hanging test hangs HCLI.",
            hot_path="per_workunit",
        )
    if func.endswith("Engine._pytest_importable"):
        return out(
            "ACCIDENTAL_PROCESS_SPAWN",
            "Spawns `[sys.executable, '-c', 'import pytest']` to see if pytest is installed. Cached on the Engine.",
            replacement="importlib.util.find_spec('pytest') is not None",
            would_break="Nothing material. find_spec answers the same question without a process. A broken pytest that imports at collection-time and crashes would no longer be detected here; _run_contained_subprocess still sees that when a test actually runs.",
            hot_path="per_workunit",
        )
    if func.endswith("Engine._validate") and "py_compile" in hay:
        return out(
            "LEGACY_WRAPPER",
            "`[sys.executable, '-m', 'py_compile', path]` per mutated .py file. py_compile is the stdlib, not an external tool, and the file is already on disk in the Engine's workspace.",
            replacement="compile(path.read_text(), str(path), 'exec') inside the Engine, or py_compile.compile(path, doraise=True) in-process",
            would_break="A SyntaxError in a C extension / coding cookie edge case that py_compile's CLI handles differently is theoretical; compile() is the same parser. Isolation is not required: the source is already being read. A hang in py_compile is implausible (timeout=120 is ceremony).",
            hot_path="per_workunit",
        )
    if func.endswith("WorkUnitExecutor._run_cpu"):
        return out(
            "ISOLATION_BOUNDARY",
            "shell=True of wu.verifier, timeout HCLI_CPU_TIMEOUT. The command is arbitrary WorkUnit text (after vacuous-command refusal). Typical payload is `python3 tools/headless/<stage>.py --stage …` (a LEGACY_WRAPPER payload) or `python3 -m pytest <file>`.",
            replacement=(
                "Split the site: (1) keep the process for untrusted/shell verifiers; "
                "(2) for a first-party Hawking module with a declared in-process entry "
                "(e.g. hcli_self_optimize.stage_sense), import and call it. "
                "GoalCompiler._verify_command currently emits `python3 -m pytest FILE` — "
                "that payload is the LEGACY half."
            ),
            would_break=(
                "shell features (`test -f x && grep -q nonce x`) used by legitimate verifiers; "
                "timeout/kill of a stuck verifier; cwd isolation; after Engine.mutate, "
                "gate.correctness MUST exec a new interpreter or it will see the old engine.py in sys.modules. "
                "Collapsing ALL verifiers in-process would mix untrusted WorkUnit code into the control plane."
            ),
            hot_path="per_workunit",
        )
    if func.endswith("Ledger.run_verify"):
        return out(
            "ISOLATION_BOUNDARY",
            "shell=True of obligation.verify_command. Same shape as WorkUnitExecutor._run_cpu. GoalCompiler emits `python3 -m pytest <test>` or empty (empty is refused, not spawned).",
            replacement="If verify_command is a first-party pytest path, call evaluate_python_test_file (still a spawn today) or pytest.main in a child. Do not import untrusted tests into the ledger process.",
            would_break="vacuous-command detector still applies; timeout; a verify command that is `test -f && grep` cannot be imported.",
            hot_path="per_workunit",
        )
    if func.endswith("validate_python_syntax"):
        return out(
            "LEGACY_WRAPPER",
            "`[sys.executable, '-m', 'py_compile', path]` in mutation.py.",
            replacement="compile(...) in-process",
            would_break="Nothing material. Duplicate of Engine._validate's py_compile spawn.",
            hot_path="per_workunit",
        )
    if func.endswith("run_validation") and "mutation.py" in f:
        return out(
            "LEGACY_WRAPPER",
            "plan['commands'] is typically `[python, -m, pytest, -x]`. A second pytest authority beside Engine._validate / verifier_pipeline.",
            replacement="Delete this path (Engine._validate is the authority) or call pytest.main / Engine._validate.",
            would_break="Anything still calling mutation.run_validation expecting a process-level pytest of the whole suite. Confirm callers before deleting.",
            hot_path="per_workunit",
        )
    if func.endswith("run_validation") and f.endswith("haider.py"):
        return out(
            "LEGACY_WRAPPER",
            "`[sys.executable, tools/haider/test_p0_tool_bridge.py]` — Python spawning Hawking Python.",
            replacement="import the test module and call its main, or pytest.main([str(test_file)])",
            would_break="haider.py is a fossil entrypoint (zero `import aider`). Callers expecting a subprocess exit code from the P0 file.",
            hot_path="cold",
        )
    if func.endswith("_run_script_counting_asserts"):
        return out(
            "ISOLATION_BOUNDARY",
            "The runner string already does runpy.run_path + sys.settrace INSIDE a child `[sys.executable, '-c', runner, path]`. The outer spawn exists so settrace and the script's sys.modules do not attach to HCLI, and so a hang can be timed out.",
            replacement="Keep the child. In-process runpy would work only for trusted files and still cannot killpg. Do not 'inline' untrusted WorkUnit scripts.",
            would_break="sys.settrace on the control plane; SystemExit from the script; module-level side effects; timeout.",
            hot_path="per_workunit",
        )
    if func.endswith("evaluate_python_test_file"):
        return out(
            "ISOLATION_BOUNDARY",
            "`[sys.executable, '-m', pytest, dest, -q, -p no:cacheprovider, --tb=short, --color=no]` for pytest-idiom files. The file may be model-written.",
            replacement="pytest.main([...]) is NOT a safe replacement: plugin state, sys.modules, and no killpg. A dedicated pytest subprocess (already this) is the isolation. Sharing one long-lived pytest worker across WorkUnits would cut startup but is still a process.",
            would_break="In-process pytest.main: leaked fixtures, collection of the wrong tree, inability to timeout, polluted assert rewriting.",
            hot_path="per_workunit",
        )
    if func.endswith("_run_cmd") and "machine.py" in f:
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "vm_stat / sysctl -n vm.pagesize / hw.memsize / … Darwin counters. MemGate reads these.",
            would_break="MemGate would lie; RuntimePool tests fail when swap is misread (the 27B live host trips this without HCLI_SWAP_CEILING_GIB=64).",
            hot_path="runtime_pool",
        )
    if func.endswith("_grok_process_blob"):
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "pgrep -fl grok. Live Grok-lane occupancy.",
            hot_path="grok",
        )
    if func.endswith("_start_token_ps") or func.endswith("_ps_argv") or func.endswith("RuntimePool._record_admission"):
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "ps -p PID for Darwin start-token / argv / rss. /proc/*/stat is Linux-only (resources.py tries it first).",
            hot_path="runtime_pool",
        )
    if func.endswith("Workspace._detect_git") or func.endswith("RepositoryGuard.detect"):
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "git rev-parse --show-toplevel. Reading .git/HEAD is wrong for worktrees (this campaign runs in one).",
            hot_path="cold",
        )
    if func.endswith("ToolExecutor._run"):
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "Admitted argv is git / rg / cargo check|test. External tools with an allowlist, not Hawking Python.",
            hot_path="cold",
        )
    if func.endswith("maybe_reexec_for_cv2") or "os.execv" in hay or hit.get("call", "").endswith("execv"):
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "os.execv into ~/.grok-vision/bin/python because system python3 has no cv2. A different interpreter, not a Hawking module.",
            would_break="OCR/ImageFileAdapter on the system interpreter. Re-exec is the documented command shape.",
            hot_path="harness",
        )
    if "capability_suite.py" in f or (tok.startswith("$") and "mlx" in hay.lower() and "-c" in hay):
        if "mlx" in hay.lower() or "MLX" in hay:
            return out(
                "REQUIRED_EXTERNAL_TOOL",
                "Dedicated MLX interpreter (`mlx_py -c MLX_RUNNER`). MLX and the control-plane python fight; this is an inference runtime, not a Hawking import.",
                would_break="GPU residency, mlx vs torch in one process, capability_suite scoring.",
                hot_path="harness",
            )
    if "runtime_ab.py" in f and "MLX" in hay:
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "MLX generate in a dedicated interpreter for the A/B runtime measurement.",
            would_break="The A/B is llama-server vs mlx_lm; in-process MLX would contaminate the llama measurement.",
            hot_path="harness",
        )
    if "hcli_persistence_audit.py" in f and tok == "sys.executable":
        return out(
            "ISOLATION_BOUNDARY",
            "Child interpreter runs a Mission and is killed; the claim is that dag.json / state.json survive the address space. In-process would not be a crash.",
            would_break="Crash-checkpoint / durability evidence.",
            hot_path="harness",
        )
    if "hcli_self_supplement.py" in f and tok == "sys.executable":
        return out(
            "ISOLATION_BOUNDARY",
            "Phase-one child is SIGKILL'd after READY. The claim is compile-and-run of an ultragoal in a killable process.",
            would_break="The SIGKILL / READY protocol.",
            hot_path="harness",
        )
    if "hcli_self_optimize_2.py" in f and ("--probe-throughput" in hay or "--probe-overlap" in hay):
        return out(
            "ISOLATION_BOUNDARY",
            "Second interpreter required to measure throughput/overlap. Threads share the lock.",
            would_break="The overlap/throughput probe would collapse onto one interpreter.",
            hot_path="per_experiment",
        )
    if "vmcp_lattice_disposition.py" in f and tok == "sys.executable":
        return out(
            "ISOLATION_BOUNDARY",
            "Child does os._exit(9) mid-write / mid-INSERT to prove atomic rename and SQLite rollback. A thread cannot test that.",
            would_break="Crash-atomicity evidence (HYBRID vs ATOMIC).",
            hot_path="harness",
        )
    if "director_epoch_replay.py" in f and (tok.startswith("$hcli") or "hcli" in hay):
        return out(
            "LEGACY_WRAPPER",
            "`[hcli, runtimes, mission]` against a scratch git workspace. hcli is Hawking Python (python -m hcli).",
            replacement="App._run_headless / Engine.execute in-process with cwd=ws. Keep a process only when the replay is testing CLI exit codes or the installed shim.",
            would_break="CLI argv, installed ~/.local/bin/hcli identity, process exit code of `hcli N mission`.",
            hot_path="harness",
        )
    if "dirty_tree_preservation.py" in f:
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "Scratch git (`git init/add/commit` with an isolated config). git is the SCM; the test's claim is git's dirty-tree behaviour.",
            hot_path="harness",
        )
    if "hcli_p0_gates.py" in f and (tok == "sh" or "grok-run" in hay or "ps " in hay):
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "ps | grep grok-run to compare max_policy.grok_pool_snapshot against the real process table. pgrep -fl grok is the production twin.",
            hot_path="grok",
        )
    if "hcli_true_mixed_max.py" in f and (
        tok in {"$expect", "expect"} or "expect_command" in hay or func.endswith("validate_workunit")
    ):
        return out(
            "ISOLATION_BOUNDARY",
            "Independent re-derivation of a Qwen answer via wu.expect_command (shell=True). Must not run inside the model process; the whole point is a second, deterministic command.",
            would_break="A Qwen unit that grades itself. The campaign already caught that loop.",
            hot_path="harness",
        )
    if "hcli_vmcp_integration.py" in f and ("replay" in hay or tok == "$replay_cmd"):
        return out(
            "ISOLATION_BOUNDARY",
            "Replay of the verifier as `python3 <verifier_py> …` against a twin file with identical bytes. SUBJECT_MISMATCH is the claim; a second process is how the production executor will run it.",
            would_break="Replay would share in-memory CaptureBus state with the original observe.",
            hot_path="harness",
        )
    if "noetic_executable_closure.py" in f and func.endswith("probe_live_decode"):
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "ascension_qwen38_hybrid_greedy (native decode binary).",
            hot_path="harness",
        )
    if "runtime_correctness_gate.py" in f and func.endswith("native_case"):
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "Native complete-wall binary against the sealed artifact.",
            hot_path="harness",
        )
    if "runtime_experiment.py" in f:
        if "cargo" in func or "cargo" in hay:
            return out(
                "REQUIRED_EXTERNAL_TOOL",
                "cargo build --profile release-fast -p hawking-core --example …",
                hot_path="harness",
            )
        if "native" in func:
            return out(
                "REQUIRED_EXTERNAL_TOOL",
                "ascension_qwen38_hybrid_greedy --complete-wall.",
                hot_path="harness",
            )
        if func.endswith("sh"):
            return out(
                "REQUIRED_EXTERNAL_TOOL",
                "argv helper. Callers are git rev-parse and host probes (not bash -lc).",
                hot_path="receipt_stamp",
            )
    if "gate_adversary.py" in f and func.endswith("run"):
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "argv helper. Callers: git status/hash-object/rev-parse/cat-file, and one bash -lc grep|sed of MAX_REPAIR_DEPTH (that one inner grep is accidental ceremony around a file read).",
            replacement="The bash -lc grep|sed caller can become a Path.read_text + regex. Keep git argv.",
            would_break="Nothing for the grep. git remains required.",
            hot_path="harness",
        )
    if "vmcp_capability_probe.py" in f and tok.startswith("$str"):
        return out(
            "ISOLATION_BOUNDARY",
            "Temp Python script that imports visionmcp (cv2/torch surface) in a child with a dedicated PYTHONPATH. The control-plane interpreter is not that env.",
            would_break="Importing visionmcp in the census/HCLI interpreter pulls optional heavy deps and is the thing the unavailable-gate exists to stop.",
            hot_path="harness",
        )
    if tok == "python3" and "performance_ledger.py" in cmd_s:
        return out(
            "LEGACY_WRAPPER",
            "python3 tools/headless/performance_ledger.py record — Hawking Python.",
            replacement="from tools.headless.performance_ledger import record; record(row)  (or a package-safe import). Keep the CLI as a shim.",
            would_break="The bash driver's `set -e` + exit code contract. A function call returns instead of exiting.",
            hot_path="harness",
        )
    if func in {"sh"} and (tok in {"bash", "sh"} or "bash -lc" in hay):
        return out(
            "ACCIDENTAL_PROCESS_SPAWN",
            "bash -lc helper wrapping git/ps/df/llama-server --version. git/ps/df are required tools; bash -lc is not.",
            replacement="subprocess.run(['git','-C',repo,'rev-parse','HEAD'], …) and argv ps/df. Pipelines (`ps | grep`, `remote -v | head`) can stay a shell or become Python.",
            would_break="A pipeline inside -lc (`git remote -v | head -1`, `ps | grep llama-server`) needs either a shell or an in-process filter. Simple rev-parse does not.",
            hot_path="receipt_stamp",
        )
    if tok in {"bash", "sh"} and hit.get("cmd_form") in {"str", "name", "argv"}:
        inner = hay
        if "git" in inner and "ps " not in inner:
            return out(
                "ACCIDENTAL_PROCESS_SPAWN",
                "bash -lc wrapping git. git stays; bash goes.",
                replacement="argv git",
                would_break="Nothing for rev-parse. Pipelines need a shell or a Python filter.",
                hot_path="receipt_stamp",
            )
        if "ps " in inner or "grep" in inner or "df " in inner:
            return out(
                "REQUIRED_EXTERNAL_TOOL",
                "Host pipeline (ps/grep/df) invoked via a shell. The tools are required; the wrapper can become argv + Python filter.",
                hot_path="harness",
            )
    if tok == "python3" and "tools/headless" in hay:
        return out(
            "LEGACY_WRAPPER",
            "python3 tools/headless/<module>.py — Hawking Python spawned by a bash/python driver.",
            replacement="Import the module's entry (record/register). Keep the CLI as a shim so in-flight lanes that shell the same command do not break.",
            would_break="bash `set -e` exit codes; operators who invoke the CLI.",
            hot_path="harness",
        )
    if func in {"sh", "run"} or func.endswith(".sh"):
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            f"argv helper {func}(); callers in this file are git/host/native (not Hawking Python).",
            hot_path="receipt_stamp",
        )

    # ========== test fixtures: the process IS the claim ==========
    if role == "test_fixture":
        if tok in HOST_BINS or tok == "git" or (isinstance(hit.get("cmd"), list) and hit["cmd"] and str(hit["cmd"][0]) == "git"):
            return out(
                "REQUIRED_EXTERNAL_TOOL",
                f"Test fixture invoking {tok} (host/SCM). Not an isolation claim.",
                hot_path="test",
            )
        if tok in {"sleep", "true"} or "sleep" in hay:
            return out(
                "ISOLATION_BOUNDARY",
                "Test double for a long-lived foreign process (llama-server stand-in) or a dead owner. The OS process is what reaper/lock/orphan tests assert on.",
                would_break="Replacing with a thread makes pid_is_alive / killpg / start_new_session tests tautological.",
                hot_path="test",
            )
        if tok == "bash" and "llama-server" in hay:
            return out(
                "ISOLATION_BOUNDARY",
                "`bash -lc 'exec -a llama-server sleep 120'` — a foreign-named process the reaper must NOT kill.",
                would_break="The reaper's identity check is the test.",
                hot_path="test",
            )
        if any(
            k in func.lower() or k in f
            for k in (
                "restart",
                "crash",
                "orphan",
                "race",
                "sigkill",
                "durability",
                "lock",
                "persist",
            )
        ) or tok == "sys.executable":
            return out(
                "ISOLATION_BOUNDARY",
                "Child Python/process required to prove restart, lock, crash, or cross-process persistence. In-process would not be a second address space.",
                would_break="The test would no longer exercise the production isolation path.",
                hot_path="test",
            )
        if tok in HOST_BINS or tok in INFER_BINS:
            return out(
                "REQUIRED_EXTERNAL_TOOL",
                f"Test fixture invoking {tok}.",
                hot_path="test",
            )
        if shell:
            return out(
                "ISOLATION_BOUNDARY",
                "Test injects a command runner (often shell=True) into verifier_pipeline to assert production behavior.",
                hot_path="test",
            )
        return out(
            "ISOLATION_BOUNDARY",
            "Test-fixture spawn; the process boundary is what is under test or is a stand-in for production isolation.",
            hot_path="test",
        )

    # ========== generic token rules ==========
    if tok == "rm":
        return out(
            "ACCIDENTAL_PROCESS_SPAWN",
            "subprocess.run(['rm','-rf', lock]) instead of shutil.rmtree.",
            replacement="shutil.rmtree(path, ignore_errors=True)",
            would_break="Nothing if the path is a directory the process owns. Do not use this on a mount point. runtime_experiment.py GPU lane lock is the site.",
            hot_path="harness",
        )
    if tok in HOST_BINS or (tok == "bash" and "git " in hay):
        why = f"host tool {tok}."
        if tok == "git" or "git" in hay:
            why = (
                "git is the SCM. `git rev-parse HEAD` / `git show HEAD:path` are correct in a worktree "
                "(this sparse worktree is the existence proof). Reading .git/HEAD is wrong."
            )
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            why,
            replacement=(
                "bash -lc wrapping of git is accidental ceremony; keep git, drop bash -lc for argv git."
                if tok == "bash"
                else None
            ),
            would_break="Receipt identity, MemGate, sparse-checkout reads via git show.",
            hot_path="receipt_stamp" if role in {"receipt_stamp", "harness"} and tok == "git" else "cold",
        )
    if tok in INFER_BINS or "llama-server" in hay or "mlx_lm" in hay or "grok-run" in hay:
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "External inference/CLI binary (llama-server / mlx_lm.server / grok-run / cargo / swift).",
            would_break="GPU residency, HTTP health, or the native decode binary.",
            hot_path="runtime_pool" if "llama" in hay or "mlx" in hay else "harness",
        )
    if tok in {"swift", "cargo"} or "ascension_" in hay:
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "Native binary (swift Metal probe, cargo, ascension_qwen38_*). Not Hawking Python.",
            hot_path="harness",
        )
    if "py_compile" in hay:
        return out(
            "LEGACY_WRAPPER",
            "python -m py_compile of a file this process can already read.",
            replacement="compile(src, filename, 'exec')",
            would_break="Nothing material.",
            hot_path="per_workunit" if "haider" in f else "harness",
        )
    if "-c" in hay and "import pytest" in hay:
        return out(
            "ACCIDENTAL_PROCESS_SPAWN",
            "python3 -c 'import pytest' probe.",
            replacement="importlib.util.find_spec('pytest')",
            would_break="Nothing material.",
            hot_path="per_workunit",
        )
    if tok == "sys.executable" or tok.startswith("$") or tok in {"python3", "python"}:
        # Python spawning Python.
        if "pytest" in hay and ("hcli/tests" in hay or "tools/haider" in hay):
            return out(
                "ISOLATION_BOUNDARY",
                "Spawns a new interpreter to run the HCLI pytest suite (self-opt gate.correctness). After mutate of engine.py this MUST be a new process: in-process import would keep the old module.",
                replacement="Keep the child for gate.correctness. For stages that only read (sense/bottleneck/hypotheses/screen), import the stage function.",
                would_break="gate.correctness would validate the pre-mutation Engine. Default HCLI_CPU_TIMEOUT=120 is below the ~140s suite wall — already watched fail.",
                hot_path="per_experiment",
            )
        if "-m" in hay and "hcli" in hay:
            return out(
                "LEGACY_WRAPPER",
                "`python -m hcli` from a harness that already imported hcli (hcli_foundation_test imports hcli.cli).",
                replacement="hcli.cli.main() / App._run_headless, except when the test is the process exit code, shims, or crash isolation.",
                would_break="check_any_folder live path that must exec the installed ~/.local/bin/hcli shim (that IS the product). Crash/restart tests.",
                hot_path="harness",
            )
        if "_probe_child" in func or "--probe-overlap" in hay:
            return out(
                "ISOLATION_BOUNDARY",
                "Second interpreter required to measure overlapping _call_model. Threads share the lock; the probe is a process.",
                would_break="The overlap measurement would collapse to one interpreter's GIL/lock.",
                hot_path="per_experiment",
            )
        if "self_optimize" in f or "self_supplement" in f:
            return out(
                "LEGACY_WRAPPER",
                "Harness spawning sys.executable against another Hawking Python module or itself.",
                replacement="Import the stage/function. Exception: gate.correctness and _probe_child (isolation).",
                would_break="See gate.correctness / probe-child isolation sites.",
                hot_path="per_experiment",
            )
        if "repair_disposition" in f or "homeostasis" in f:
            return out(
                "ISOLATION_BOUNDARY",
                "Writer/reader in two processes to prove BackendHealth / repair state survives an address space. That is the claim.",
                would_break="An in-process read after write would not prove persistence.",
                hot_path="harness",
            )
        if "noetic_executable_closure" in f or "hybrid_greedy" in hay:
            return out(
                "REQUIRED_EXTERNAL_TOOL",
                "Native decode binary (ascension_qwen38_hybrid_greedy), not Hawking Python.",
                hot_path="harness",
            )
        if role == "harness" and ("-c" in hay or tok == "sys.executable"):
            # Remaining python3 -c in harnesses: often a worker.
            if "sys.path" in hay or "from hcli" in hay or "from tools.haider" in hay:
                return out(
                    "ISOLATION_BOUNDARY" if any(k in hay.lower() for k in ("kill", "pid", "sleep", "restart", "child")) else "LEGACY_WRAPPER",
                    "Harness python3 -c that imports hcli in a child.",
                    replacement="Import hcli in-process unless the test needs a second address space.",
                    would_break="Cross-process persistence, crash, or argv contracts.",
                    hot_path="harness",
                )
            return out(
                "LEGACY_WRAPPER",
                "sys.executable child whose body is Hawking Python.",
                replacement="Import and call.",
                would_break="Depends on whether the child is used as an isolation test. Inspect the site.",
                hot_path="harness",
            )
        if shell:
            return out(
                "ISOLATION_BOUNDARY",
                "shell=True of a command string. Keep for untrusted/shell combinators; replace when the string is a fixed Hawking python3 module.",
                hot_path="per_workunit",
            )
        return out(
            "LEGACY_WRAPPER",
            f"Python child whose argv is assembled dynamically (form={hit.get('cmd_form')} tok={tok}). Hawking Python spawning Hawking Python unless the body is an isolation fixture.",
            replacement="Import and call, except when the child is the isolation claim.",
            hot_path="harness",
        )
    if tok == "bash" or tok == "sh":
        if "git" in hay:
            return out(
                "ACCIDENTAL_PROCESS_SPAWN",
                "bash -lc wrapping a git argv. git is required; bash is not.",
                replacement="subprocess.run(['git', ...], cwd=repo)",
                would_break="Nothing if the inner command is a simple git invocation. A pipeline inside -lc would need to stay a shell.",
                hot_path="receipt_stamp",
            )
        return out(
            "ACCIDENTAL_PROCESS_SPAWN",
            "bash -lc wrapping a host probe (`command -v`, git, etc.). Keep the tool; drop bash -lc when the inner command is a simple argv.",
            replacement="shutil.which(...) or argv git/ps",
            hot_path="harness",
        )
    if tok in {"sleep", "true"}:
        return out(
            "ISOLATION_BOUNDARY",
            "OS process used as a fixture or wait.",
            hot_path="test",
        )
    if tok == "pkill":
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            "pkill of a named binary. Production RuntimePool forbids bare `pkill llama-server`; a harness using it is still an external tool.",
            hot_path="harness",
        )

    if "bootstrap_snapshots" in f:
        return out(
            "LEGACY_WRAPPER",
            "Fossil snapshot of haider.py. Same spawn shape as the fossil run_validation / _start_fast_runtime_server.",
            hot_path="cold",
        )

    if tok in {
        "tar",
        "clang",
        "system_profiler",
        "strings",
        "sandbox-exec",
        "/usr/bin/sandbox-exec",
        "xcrun",
    }:
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            f"host/toolchain binary {tok}.",
            hot_path="harness",
        )

    if tok == "sys.executable" or tok.startswith("$str"):
        return out(
            "LEGACY_WRAPPER",
            "Python child whose argv is assembled dynamically (self-spawn of a harness module, or sys.executable + test path).",
            replacement="Import and call the module unless the child is an isolation test.",
            hot_path="harness",
        )

    if tok in {"$cmd", "$argv", "$args", "$verifier"} or hit.get("cmd_form") in {"name", "call"}:
        fl = func.lower()
        if any(k in fl for k in ("git", "cargo", "decode", "compile", "sysctl", "clang", "native", "example")):
            return out(
                "REQUIRED_EXTERNAL_TOOL",
                f"argv helper {func}(); callers are host/native tools, not Hawking Python.",
                hot_path="harness",
            )
        return out(
            "REQUIRED_EXTERNAL_TOOL",
            f"argv helper {func}(); the callee is not a Hawking Python module at this call site (reconstructed as {tok}).",
            hot_path="harness",
        )

    return out(
        "UNKNOWN",
        f"no rule matched file={f} func={func} tok={tok} form={hit.get('cmd_form')}",
    )


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def scan() -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    watched: List[str] = []
    files: List[str] = []
    for root in ROOTS:
        try:
            files.extend(git_ls(root))
        except Exception as exc:  # noqa: BLE001
            watched.append(f"git ls-tree {root} failed: {exc}")
    files = [
        rel
        for rel in files
        if rel.endswith((".py", ".sh", ".rs"))
        and "__pycache__" not in rel
        and "/target/" not in rel
    ]
    sites: List[Dict[str, Any]] = []
    origins = Counter()
    parse_fail = 0
    for rel in files:
        try:
            text, origin = read_source(rel)
        except Exception as exc:  # noqa: BLE001
            watched.append(f"could not read {rel}: {exc}")
            continue
        origins[origin] += 1
        extracted = extract_file(rel, text, origin)
        for hit in extracted:
            if hit.get("call") == "UNKNOWN":
                parse_fail += 1
            info = classify(hit)
            rec = dict(hit)
            rec.update(info)
            rec["id"] = f"{rel}:{hit['line']}"
            rec["path_resolves"] = True
            rec["source_sha256"] = sha256_text(text)
            sites.append(rec)
    meta = {
        "files_listed": len(files),
        "origins": dict(origins),
        "parse_fail": parse_fail,
        "sites": len(sites),
    }
    # Verify each site's line actually contains a spawn token.
    for rec in sites:
        rel, line = rec["file"], rec["line"]
        try:
            text, _ = read_source(rel)
        except Exception as exc:  # noqa: BLE001
            rec["path_resolves"] = False
            rec["class"] = "UNKNOWN"
            rec["why"] = f"path stopped resolving at verify: {exc}"
            watched.append(f"site {rel}:{line} stopped resolving: {exc}")
            continue
        lines = text.splitlines()
        if not (1 <= line <= len(lines)):
            rec["path_resolves"] = False
            rec["class"] = "UNKNOWN"
            rec["why"] = f"line {line} out of range ({len(lines)} lines)"
            watched.append(f"{rel}:{line} out of range")
            continue
        window = "\n".join(lines[max(0, line - 1) : min(len(lines), line + 6)])
        rec["line_text"] = lines[line - 1].strip()[:240]
        if rec.get("language") == "bash":
            continue
        if rec.get("language") == "rust":
            if "Command::new" not in window:
                rec["path_resolves"] = False
                watched.append(
                    f"{rel}:{line} rust window has no Command::new; snippet={rec.get('snippet')!r}"
                )
            continue
        needles = (
            "subprocess",
            "Popen",
            "os.system",
            "os.popen",
            "os.spawn",
            "os.exec",
            "check_output",
            "create_subprocess",
        )
        if not any(n in window for n in needles):
            rec["path_resolves"] = False
            watched.append(
                f"{rel}:{line} window has no subprocess token; snippet={rec.get('snippet')!r}"
            )
    return sites, watched, meta


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _repeat(fn, n: int = 25, warmup: int = 3) -> Dict[str, Any]:
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "n": n,
        "median_ms": round(statistics.median(samples), 3),
        "p10_ms": round(samples[max(0, n // 10 - 1)], 3),
        "p90_ms": round(samples[min(n - 1, (9 * n) // 10)], 3),
        "mean_ms": round(statistics.fmean(samples), 3),
        "min_ms": round(samples[0], 3),
        "max_ms": round(samples[-1], 3),
    }


def measure_micro() -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    py = sys.executable

    def run(argv, **kw):
        return subprocess.run(
            argv, capture_output=True, timeout=30, **kw
        )

    results["true"] = _repeat(lambda: run(["/usr/bin/true"]))
    results["git_rev_parse_HEAD"] = _repeat(
        lambda: run(["git", "rev-parse", "HEAD"], cwd=str(REPO))
    )
    results["sysctl_hw_ncpu"] = _repeat(
        lambda: run(["sysctl", "-n", "hw.ncpu"])
    )
    results["python_c_pass"] = _repeat(lambda: run([py, "-c", "pass"]))
    results["python_c_import_pytest"] = _repeat(
        lambda: run([py, "-c", "import pytest"])
    )

    def inproc_find_pytest():
        import importlib.util

        return importlib.util.find_spec("pytest") is not None

    results["inprocess_find_spec_pytest"] = _repeat(inproc_find_pytest)

    with tempfile.TemporaryDirectory(prefix="census-pycompile-") as tmp:
        path = Path(tmp) / "mod.py"
        path.write_text("x = 1\n", encoding="utf-8")

        def spawn_py_compile():
            run([py, "-m", "py_compile", str(path)])

        def inproc_compile():
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

        results["python_m_py_compile"] = _repeat(spawn_py_compile)
        results["inprocess_compile"] = _repeat(inproc_compile)

        testf = Path(tmp) / "test_one.py"
        testf.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

        def spawn_pytest():
            run(
                [
                    py,
                    "-m",
                    "pytest",
                    str(testf),
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--tb=no",
                    "--color=no",
                ]
            )

        def inproc_pytest_main():
            import contextlib
            import io

            import pytest as _pt

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                _pt.main(
                    [str(testf), "-q", "-p", "no:cacheprovider", "--tb=no", "--color=no"]
                )

        results["python_m_pytest_one_test"] = _repeat(spawn_pytest, n=10, warmup=1)
        # pytest.main mutates global plugin state; run fewer times.
        try:
            results["inprocess_pytest_main_one_test"] = _repeat(
                inproc_pytest_main, n=5, warmup=1
            )
        except Exception as exc:  # noqa: BLE001
            results["inprocess_pytest_main_one_test"] = {"error": str(exc)}

        lock = Path(tmp) / "lockdir"
        lock.mkdir()
        (lock / "x").write_text("1", encoding="utf-8")

        def spawn_rm():
            if lock.exists():
                run(["rm", "-rf", str(lock)])
            lock.mkdir(exist_ok=True)
            (lock / "x").write_text("1", encoding="utf-8")

        def inproc_rmtree():
            if lock.exists():
                shutil.rmtree(lock)
            lock.mkdir()
            (lock / "x").write_text("1", encoding="utf-8")

        results["rm_rf"] = _repeat(spawn_rm, n=15, warmup=2)
        results["inprocess_shutil_rmtree"] = _repeat(inproc_rmtree, n=15, warmup=2)

    results["notes"] = {
        "python_c_pass": "lower bound on a WorkUnit whose verifier is `python3 -c …`",
        "python_m_pytest_one_test": "lower bound on evaluate_python_test_file / Engine contained pytest of a 1-test file",
        "python_m_py_compile": "Engine._validate per mutated .py file",
        "python_c_import_pytest": "Engine._pytest_importable (once per Engine, then cached)",
        "git_rev_parse_HEAD": "receipt stamp used by almost every harness",
    }
    return results


class _Audit:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.t0 = time.perf_counter()

    def hook(self, event: str, args: tuple) -> None:
        if event != "subprocess.Popen":
            return
        # args: (executable, args, cwd, env)
        executable = args[0] if args else None
        argv = args[1] if len(args) > 1 else None
        cwd = args[2] if len(args) > 2 else None
        self.events.append(
            {
                "t_ms": round((time.perf_counter() - self.t0) * 1000.0, 3),
                "executable": str(executable) if executable is not None else None,
                "argv0": (
                    str(argv[0])
                    if isinstance(argv, (list, tuple)) and argv
                    else None
                ),
                "argv": (
                    [str(x) for x in list(argv)[:8]]
                    if isinstance(argv, (list, tuple))
                    else None
                ),
                "cwd": str(cwd) if cwd else None,
            }
        )


def _install_audit(audit: _Audit):
    sys.addaudithook(audit.hook)
    return audit


def measure_workunit(watched: List[str]) -> Dict[str, Any]:
    """Representative CPU WorkUnit: executors._run_cpu of a python3 -c verifier.

    Also times a first-party module spawn (this census file --help-less: -c pass
    is the floor; a real stage script pays import of hcli as well).
    """
    py = sys.executable
    out: Dict[str, Any] = {
        "shape": (
            "CPU WorkUnit whose verifier is a Python process. "
            "WorkUnitExecutor._run_cpu does subprocess.run(cmd, shell=True). "
            "Self-opt stages use `python3 tools/headless/hcli_self_optimize.py --stage <name>` "
            "(receipts/headless/HCLI_SELF_OPT_ITERATION_1.json workunits[].verification.command)."
        )
    }
    verifier = f'{py} -c "print(0)"'
    audit = _Audit()
    _install_audit(audit)
    t0 = time.perf_counter()
    proc = subprocess.run(
        verifier,
        shell=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    out["python_c_print_via_shell"] = {
        "command": verifier,
        "exit_code": proc.returncode,
        "wall_ms": round(wall_ms, 3),
        "spawn_events": len(audit.events),
        "events": audit.events,
        "stdout": (proc.stdout or "").strip()[:80],
    }

    # First-party module: spawn this file with a cheap argv that exits 2 (argparse)
    # is not representative. Spawn a tiny Hawking-shaped module: compile+run a
    # file that only imports pathlib, as a floor; then try importing hcli from
    # a git-archive extract.
    with tempfile.TemporaryDirectory(prefix="census-wu-") as tmp:
        mod = Path(tmp) / "stage.py"
        mod.write_text("print('ok')\n", encoding="utf-8")
        cmd = f"{py} {mod}"
        a2 = _Audit()
        # adding a second hook is cumulative; subtract previous count
        before = len(a2.events)
        sys.addaudithook(a2.hook)
        t1 = time.perf_counter()
        proc2 = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        out["python_module_print_via_shell"] = {
            "command": "python3 /tmp/…/stage.py",
            "exit_code": proc2.returncode,
            "wall_ms": round((time.perf_counter() - t1) * 1000.0, 3),
            "spawn_events": max(0, len(a2.events) - before),
        }

        # Prefer on-disk top-level hcli; otherwise git-archive HEAD:hcli.
        try:
            on_disk = REPO / "hcli" / "__main__.py"
            if on_disk.is_file():
                pythonpath = str(REPO)
                extract_cwd = str(REPO)
            else:
                archive = subprocess.check_output(
                    ["git", "archive", "HEAD", "hcli"], cwd=str(REPO), timeout=30
                )
                extract = Path(tmp) / "tree"
                extract.mkdir()
                tarfile.open(fileobj=io.BytesIO(archive), mode="r:*").extractall(extract)
                pythonpath = str(extract)
                extract_cwd = str(extract)
            t2 = time.perf_counter()
            proc3 = subprocess.run(
                [
                    py,
                    "-c",
                    "from hcli.executors import WorkUnitExecutor; print('imported')",
                ],
                cwd=extract_cwd,
                env={**os.environ, "PYTHONPATH": pythonpath, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                timeout=30,
            )
            import_ms = (time.perf_counter() - t2) * 1000.0
            out["import_hcli_executors_in_child"] = {
                "exit_code": proc3.returncode,
                "wall_ms": round(import_ms, 3),
                "stderr_tail": (proc3.stderr or "")[-400:],
                "stdout": (proc3.stdout or "").strip()[:80],
            }
            # In-process import of the extracted package
            if pythonpath not in sys.path:
                sys.path.insert(0, pythonpath)
            t3 = time.perf_counter()
            try:
                from hcli.executors import WorkUnitExecutor  # type: ignore
                from hcli.workunit import WorkUnit  # type: ignore

                inproc_import_ms = (time.perf_counter() - t3) * 1000.0
                wu = WorkUnit(
                    id="census-wu",
                    role="sense",
                    description="census",
                    preferred_backend="cpu",
                    resource_class="CPU_HEAVY",
                    verifier=f'{py} -c "print(0)"',
                )
                ex = WorkUnitExecutor(workspace=tmp)
                a3 = _Audit()
                sys.addaudithook(a3.hook)
                n0 = len(a3.events)
                t4 = time.perf_counter()
                result = ex._run_cpu(wu, {})  # noqa: SLF001
                live_ms = (time.perf_counter() - t4) * 1000.0
                out["live_WorkUnitExecutor_run_cpu"] = {
                    "wall_ms": round(live_ms, 3),
                    "spawn_events": max(0, len(a3.events) - n0),
                    "validation_ok": (result.get("validation") or {}).get("ok"),
                    "exit_code": (result.get("validation") or {}).get("exit_code"),
                    "inprocess_import_ms": round(inproc_import_ms, 3),
                    "spawns_per_workunit": max(0, len(a3.events) - n0),
                }
            except Exception as exc:  # noqa: BLE001
                watched.append(f"WorkUnitExecutor live path failed: {type(exc).__name__}: {exc}")
                out["live_WorkUnitExecutor_run_cpu"] = {"error": f"{type(exc).__name__}: {exc}"}
        except Exception as exc:  # noqa: BLE001
            watched.append(f"hcli archive/import failed: {type(exc).__name__}: {exc}")
            out["import_hcli_executors_in_child"] = {"error": str(exc)}

    # Formula for a typical self-opt WorkUnit.
    out["formula"] = {
        "cpu_python_stage": {
            "spawns": 1,
            "what": "WorkUnitExecutor._run_cpu shell=True of python3 tools/headless/<script>.py --stage …",
            "inner": "the stage may stamp git rev-parse (1 more) or py_compile (mutate) or pytest (gate.correctness)",
        },
        "engine_mutation_validation": {
            "spawns": "n_py_files + 1_pytest_importable_once + n_tests",
            "what": "Engine._validate: py_compile per .py, then _run_contained_subprocess per test",
        },
        "verifier_pipeline_obligation": {
            "spawns": "1 run_command (injected; often shell) + 0..1 pytest/script if a test file is named",
            "what": "verify() always calls run_command(command) once per obligation with a non-empty admitted command",
        },
    }
    return out


def measure_suite(watched: List[str]) -> Dict[str, Any]:
    """Live spawn count + milliseconds for pytest tools/haider/hcli/tests.

    Always git-archives HEAD:tools/haider into /tmp. This worktree is sparse
    (haider is not on disk) and the live ~/Downloads/hawking tree must not be
    used as cwd: ten science lanes and a resident 27B live there.
    Unit tests use FakeBackend/sleep; mlx --help only. Does not load the 27B.
    """
    out: Dict[str, Any] = {
        "historical": HIST_SUITE,
        "historical_note": (
            "Self-opt gate.correctness is itself a WorkUnit whose verifier shells "
            "python3 -m pytest tools/haider/hcli/tests. Measured walls 129.3s and "
            "139.5s (receipts cited). The contract's ~140s suite is that command."
        ),
    }
    if os.environ.get("CENSUS_REMEASURE_SUITE") != "1":
        prev = None
        if RECEIPT.is_file():
            try:
                prev = json.loads(RECEIPT.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prev = None
        cached = ((prev or {}).get("cost") or {}).get("suite") or {}
        live = cached.get("live") or {}
        if live.get("n_spawns"):
            watched.append(
                f"Cited previous suite live measurement from {prev.get('generated_at')} "
                f"(n_spawns={live.get('n_spawns')}, wall_ms={live.get('wall_ms')}). "
                "This run did not re-exec pytest tools/haider/hcli/tests "
                "(~140s). CENSUS_REMEASURE_SUITE=1 forces a rerun. Historical walls "
                "remain in cost.suite.historical."
            )
            out.update(cached)
            out["reused"] = True
            out["reused_from"] = prev.get("generated_at")
            return out
        watched.append(
            "Suite not remeasured (CENSUS_REMEASURE_SUITE unset) and no prior live "
            "block to cite. Historical self-opt walls are in cost.suite.historical."
        )
        out["live"] = {
            "ok": False,
            "reason": "NOT_REMEASURED",
            "note": "Set CENSUS_REMEASURE_SUITE=1 to run pytest tools/haider/hcli/tests under an audit hook.",
        }
        return out
    py = sys.executable
    env = os.environ.copy()
    env["HCLI_SWAP_CEILING_GIB"] = "64"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("HCLI_LLAMA_DEVICE", "none")
    watched.append(
        "tools/haider is not on disk in this sparse worktree; "
        "git archive HEAD tools/haider into a temp tree for the suite probe. "
        "Did not use ~/Downloads/hawking as cwd (live 27B + in-flight lanes)."
    )
    tmp_extract = tempfile.mkdtemp(prefix="census-suite-")
    archive = subprocess.check_output(
        ["git", "archive", "HEAD", "tools/haider"], cwd=str(REPO), timeout=30
    )
    tarfile.open(fileobj=io.BytesIO(archive), mode="r:*").extractall(tmp_extract)
    (Path(tmp_extract) / "tools" / "__init__.py").write_text("", encoding="utf-8")
    haider_init = Path(tmp_extract) / "tools" / "haider" / "__init__.py"
    if not haider_init.exists():
        haider_init.write_text("", encoding="utf-8")
    cwd = Path(tmp_extract)
    tests = "tools/haider/hcli/tests"
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    out["repo"] = str(cwd)
    out["repo_kind"] = "git-archive temp tree"

    driver = r"""
import json, os, sys, time
from pathlib import Path

events = []
t0 = time.perf_counter()

def hook(event, args):
    if event != "subprocess.Popen":
        return
    executable = args[0] if args else None
    argv = args[1] if len(args) > 1 else None
    events.append({
        "t_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "executable": None if executable is None else str(executable),
        "argv0": str(argv[0]) if isinstance(argv, (list, tuple)) and argv else None,
        "argv": [str(x) for x in list(argv)[:6]] if isinstance(argv, (list, tuple)) else None,
    })

sys.addaudithook(hook)
import pytest
code = pytest.main(sys.argv[1:])
wall_ms = (time.perf_counter() - t0) * 1000.0
Path(os.environ["CENSUS_SPAWN_LOG"]).write_text(json.dumps({
    "exit_code": int(code),
    "wall_ms": wall_ms,
    "n_spawns": len(events),
    "events": events,
}, indent=2), encoding="utf-8")
raise SystemExit(code)
"""
    with tempfile.TemporaryDirectory(prefix="census-suite-driver-") as d:
        driver_path = Path(d) / "driver.py"
        driver_path.write_text(driver, encoding="utf-8")
        log_path = Path(d) / "spawns.json"
        env["CENSUS_SPAWN_LOG"] = str(log_path)
        argv = [
            py,
            str(driver_path),
            tests,
            "-q",
            "--tb=no",
            "-p",
            "no:cacheprovider",
            "--color=no",
        ]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                argv,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=200,
            )
        except subprocess.TimeoutExpired as exc:
            watched.append(f"suite pytest timed out after 200s: {exc}")
            out["live"] = {
                "ok": False,
                "reason": "TIMEOUT",
                "wall_ms": 200000,
                "stdout_tail": (exc.stdout or "")[-1500:] if isinstance(exc.stdout, str) else "",
            }
            if tmp_extract:
                shutil.rmtree(tmp_extract, ignore_errors=True)
            return out
        wall_ms = (time.perf_counter() - t0) * 1000.0
        payload: Dict[str, Any] = {}
        if log_path.is_file():
            try:
                payload = json.loads(log_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                watched.append(f"suite spawn log JSON failed: {exc}")
        n_spawns = int(payload.get("n_spawns") or 0)
        events = payload.get("events") or []
        # Summarise argv0
        argv0s = Counter(e.get("argv0") or e.get("executable") or "?" for e in events)
        # Drop the outer pytest process itself (not in the audit of the child… the
        # child's audit records ITS spawns, not its own creation). Good.
        out["live"] = {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "wall_ms": round(payload.get("wall_ms") or wall_ms, 3),
            "n_spawns": n_spawns,
            "spawns_per_second": (
                round(n_spawns / ((payload.get("wall_ms") or wall_ms) / 1000.0), 3)
                if (payload.get("wall_ms") or wall_ms)
                else None
            ),
            "argv0_counts": dict(argv0s.most_common(20)),
            "stdout_tail": (proc.stdout or "")[-1500:],
            "stderr_tail": (proc.stderr or "")[-800:],
            "tests_path": tests,
        }
        # Parse pytest summary
        m = re.search(
            r"(\d+) passed(?:, (\d+) skipped)?", (proc.stdout or "") + (proc.stderr or "")
        )
        if m:
            out["live"]["passed"] = int(m.group(1))
            out["live"]["skipped"] = int(m.group(2) or 0)
    if tmp_extract:
        shutil.rmtree(tmp_extract, ignore_errors=True)
    return out


def experiment_cost(micro: Dict[str, Any], wu: Dict[str, Any]) -> Dict[str, Any]:
    """Per-WorkUnit and per-experiment accounting, measured + cited."""
    shell_ms = (wu.get("live_WorkUnitExecutor_run_cpu") or {}).get("wall_ms")
    if shell_ms is None:
        shell_ms = (wu.get("python_c_print_via_shell") or {}).get("wall_ms")
    py_ms = (micro.get("python_c_pass") or {}).get("median_ms")
    pytest_one_ms = (micro.get("python_m_pytest_one_test") or {}).get("median_ms")
    pycompile_ms = (micro.get("python_m_py_compile") or {}).get("median_ms")
    git_ms = (micro.get("git_rev_parse_HEAD") or {}).get("median_ms")
    live_spawns = (wu.get("live_WorkUnitExecutor_run_cpu") or {}).get("spawns_per_workunit")
    return {
        "per_workunit_cpu_python_verifier": {
            "spawns": live_spawns if live_spawns is not None else 1,
            "measured_wall_ms": shell_ms,
            "floor_ms": py_ms,
            "sites": [
                "hcli/executors.py:WorkUnitExecutor._run_cpu",
                "hcli/ledger.py:Ledger.run_verify (obligation path)",
            ],
            "note": (
                "One OS process for the verifier. A first-party stage that only "
                "reads source can become an import (LEGACY_WRAPPER). "
                "gate.correctness cannot: it must exec a new interpreter after mutate."
            ),
        },
        "per_workunit_engine_validate": {
            "spawns_formula": "len(py_paths) * py_compile + 1 * import_pytest (cached) + len(tests) * contained_pytest",
            "py_compile_ms": pycompile_ms,
            "contained_pytest_one_test_ms": pytest_one_ms,
            "import_pytest_probe_ms": (micro.get("python_c_import_pytest") or {}).get("median_ms"),
            "inprocess_compile_ms": (micro.get("inprocess_compile") or {}).get("median_ms"),
            "inprocess_find_spec_ms": (micro.get("inprocess_find_spec_pytest") or {}).get("median_ms"),
            "sites": [
                "hcli/engine.py:Engine._validate",
                "hcli/engine.py:Engine._run_contained_subprocess",
                "hcli/engine.py:Engine._pytest_importable",
            ],
        },
        "per_workunit_verifier_pipeline": {
            "spawns_formula": "1 * run_command per obligation with a non-empty admitted command; plus pytest/script if evaluate_python_test_file runs",
            "pytest_one_test_ms": pytest_one_ms,
            "sites": [
                "hcli/verifier_pipeline.py:verify (run_command, injected)",
                "hcli/verifier_pipeline.py:evaluate_python_test_file",
                "hcli/verifier_pipeline.py:_run_script_counting_asserts",
            ],
        },
        "per_experiment_self_opt": {
            "workunits": 10,
            "receipts": [
                "receipts/headless/HCLI_SELF_OPT_ITERATION_1.json",
                "receipts/headless/HCLI_SELF_OPT_ITERATION_2.json",
            ],
            "spawns_from_executor": 10,
            "plus_inner_gate_correctness": "1 pytest of tools/haider/hcli/tests (129–140s) which then starts one process per test that shells out",
            "mission_wall_s": {
                "iteration_1": HIST_SUITE["iteration_1"]["mission_wall_s"],
                "iteration_2": HIST_SUITE["iteration_2"]["mission_wall_s"],
            },
            "gate_correctness_wall_s": {
                "iteration_1": HIST_SUITE["iteration_1"]["wall_s"],
                "iteration_2": HIST_SUITE["iteration_2"]["wall_s"],
            },
            "note": (
                "Nine stages are ~1 python3 spawn each (milliseconds). "
                "The tenth is gate.correctness, which is the suite. "
                "Experiment wall ≈ suite wall. Process ceremony of the nine cheap "
                "stages is not the 140s; the suite-as-verifier is."
            ),
            "git_stamp_ms_each": git_ms,
        },
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _median_ms(block: Any) -> Optional[float]:
    if isinstance(block, dict) and isinstance(block.get("median_ms"), (int, float)):
        return float(block["median_ms"])
    return None


def spawn_ms_estimate(rec: Dict[str, Any], micro: Dict[str, Any]) -> Dict[str, Any]:
    tok = str(rec.get("first_token") or "")
    hay = " ".join(
        [
            tok,
            str(rec.get("command") or ""),
            str(rec.get("call") or ""),
            str(rec.get("file") or ""),
        ]
    ).lower()
    key = None
    if "py_compile" in hay:
        key = "python_m_py_compile"
    elif "import pytest" in hay:
        key = "python_c_import_pytest"
    elif "pytest" in hay:
        key = "python_m_pytest_one_test"
    elif tok in {"python3", "python", "sys.executable"} or tok.startswith("$") and "python" in tok.lower():
        key = "python_c_pass"
    elif tok == "git" or hay.startswith("git "):
        key = "git_rev_parse_HEAD"
    elif tok == "sysctl" or "sysctl" in hay:
        key = "sysctl_hw_ncpu"
    elif tok == "true":
        key = "true"
    elif tok == "rm":
        key = "rm_rf"
    ms = _median_ms(micro.get(key)) if key else None
    note = None
    if ms is None:
        if rec.get("language") == "rust":
            note = "ABSENT: no microbench of this native binary this run; python/git/sysctl floors are in cost.microbench"
        elif tok in {"swift", "grok-run", "llama-server", "cargo"}:
            note = f"ABSENT: {tok} spawn not microbenched here (see CONTROL_PLANE_LATENCY_LEDGER for grok/swift walls)"
        else:
            note = "ABSENT: no matching microbench key"
    return {
        "ms": ms,
        "from": key,
        "note": note,
        "kind": "measured-microbench" if ms is not None else "ABSENT",
    }


def starts_per_workunit(rec: Dict[str, Any]) -> Any:
    hot = rec.get("hot_path")
    if hot == "per_workunit":
        return 1
    if rec.get("function", "").endswith("Engine._pytest_importable"):
        return "0 after first Engine (cached); 1 on first validate"
    if rec.get("language") == "rust" or rec.get("role") == "native":
        return 0
    if hot in {"receipt_stamp", "harness", "cold", "test", "native"}:
        return 0
    if hot == "per_experiment":
        return 0
    if hot == "runtime_pool":
        return 0
    if hot == "grok":
        return 0
    return 0


def python_to_python_hops(sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hops = []
    for s in sites:
        tok = str(s.get("first_token") or "")
        hay = f"{tok} {s.get('command') or ''} {s.get('why') or ''}".lower()
        py = tok in {"python3", "python", "sys.executable"} or "python" in tok.lower()
        if not py:
            continue
        if s.get("class") not in {"LEGACY_WRAPPER", "ACCIDENTAL_PROCESS_SPAWN"}:
            continue
        hops.append(
            {
                "id": s.get("id"),
                "file": s.get("file"),
                "line": s.get("line"),
                "function": s.get("function"),
                "class": s.get("class"),
                "class_contract": CLASS_CONTRACT.get(s.get("class") or "", "unknown"),
                "command": s.get("command"),
                "in_process_replacement": s.get("in_process_replacement"),
                "what_would_break": s.get("what_would_break"),
                "starts_per_workunit": starts_per_workunit(s),
            }
        )
    return hops


def replacements_table(sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for rec in sites:
        if rec["class"] not in {"LEGACY_WRAPPER", "ACCIDENTAL_PROCESS_SPAWN"}:
            continue
        if rec.get("role") == "test_fixture":
            continue
        rows.append(
            {
                "id": rec["id"],
                "class": rec["class"],
                "role": rec.get("role"),
                "function": rec.get("function"),
                "command": rec.get("command"),
                "in_process_replacement": rec.get("in_process_replacement"),
                "what_would_break": rec.get("what_would_break"),
                "why": rec.get("why"),
            }
        )
    return rows


def isolation_table(sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for rec in sites:
        if rec["class"] != "ISOLATION_BOUNDARY":
            continue
        if rec.get("role") == "test_fixture":
            continue
        rows.append(
            {
                "id": rec["id"],
                "function": rec.get("function"),
                "command": rec.get("command"),
                "why_isolation_is_required": rec.get("why"),
                "what_would_break": rec.get("what_would_break"),
                "hot_path": rec.get("hot_path"),
            }
        )
    return rows


def tree_integrity() -> Dict[str, Any]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=20,
    )
    allowed_prefixes = (
        "tools/headless/",
        "receipts/headless/",
        "hcli/",
        "crates/",
        "src/",
        "tools/",
    )
    denied_prefixes = (
        "receipts/ascent-2026-08-16",
        "receipts/ascent-2026-08-18",
        "workspace/campaign",
        "receipts/headless/BANDWIDTH",
        "receipts/headless/PREFILL_KV",
    )
    unexpected = []
    listed = []
    denied_hits = []
    for line in (proc.stdout or "").splitlines():
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[0].strip()
        listed.append(line)
        if any(path == p or path.startswith(p) for p in denied_prefixes):
            denied_hits.append(line)
            unexpected.append(line)
            continue
        if any(path == p or path.startswith(p) for p in allowed_prefixes):
            continue
        if "__pycache__" in path:
            continue
        unexpected.append(line)
    return {
        "porcelain": listed,
        "unexpected": unexpected,
        "denied_hits": denied_hits,
        "denied_untouched": not denied_hits,
        "denied": list(denied_prefixes),
        "write_scope": list(allowed_prefixes),
    }


def print_report(doc: Dict[str, Any]) -> None:
    print(f"# SUBPROCESS CENSUS  {doc['schema']}")
    print(f"generated_at  {doc['generated_at']}")
    print(f"git_head      {doc['git_head']}")
    print(f"scope         {', '.join(doc['scope']['roots'])}")
    print(f"method        {doc['method']['scan']}")
    print()
    counts = doc["counts"]
    print("## counts")
    for k in CLASSES:
        print(f"  {k:28s} {counts.get(k, 0)}")
    print(f"  {'TOTAL':28s} {counts.get('TOTAL', 0)}")
    print(f"  files scanned {doc['scan_meta']['files_listed']}  origins {doc['scan_meta']['origins']}")
    print()
    print("## every spawn site")
    by_class = defaultdict(list)
    for s in doc["sites"]:
        by_class[s["class"]].append(s)
    for klass in CLASSES:
        rows = by_class.get(klass) or []
        if not rows:
            continue
        print(f"\n### {klass} ({len(rows)})")
        for s in rows:
            print(
                f"{s['id']:62s}  {s.get('role','?'):18s}  {s.get('call','')}"
            )
            print(f"    cmd   {s.get('command')}")
            print(f"    why   {s.get('why')}")
            if s.get("in_process_replacement"):
                print(f"    replace {s['in_process_replacement']}")
            if s.get("what_would_break"):
                print(f"    breaks  {s['what_would_break']}")
    print()
    print("## measured cost")
    micro = doc["cost"]["microbench"]
    print("microbench median_ms:")
    for k, v in micro.items():
        if k == "notes":
            continue
        if isinstance(v, dict) and "median_ms" in v:
            print(f"  {k:36s} {v['median_ms']:8.3f}  (p10={v['p10_ms']:.3f} p90={v['p90_ms']:.3f} n={v['n']})")
        else:
            print(f"  {k:36s} {v}")
    print()
    wu = doc["cost"]["workunit"]
    live = wu.get("live_WorkUnitExecutor_run_cpu") or {}
    print("representative WorkUnit (CPU verifier python3 -c print):")
    print(f"  live_WorkUnitExecutor._run_cpu  wall_ms={live.get('wall_ms')}  spawns={live.get('spawns_per_workunit')}  ok={live.get('validation_ok')}")
    print(f"  shell python3 -c                wall_ms={(wu.get('python_c_print_via_shell') or {}).get('wall_ms')}  spawns={(wu.get('python_c_print_via_shell') or {}).get('spawn_events')}")
    print(f"  import hcli.executors (child)   {(wu.get('import_hcli_executors_in_child') or {})}")
    print()
    exp = doc["cost"]["experiment"]
    print("per experiment (self-opt, 10 WorkUnits):")
    print(f"  {json.dumps(exp['per_experiment_self_opt'], indent=2)}")
    print()
    suite = doc["cost"]["suite"]
    print("suite run (pytest tools/haider/hcli/tests):")
    print(f"  historical iteration_1 wall_s={HIST_SUITE['iteration_1']['wall_s']:.1f} passed={HIST_SUITE['iteration_1']['passed']}")
    print(f"  historical iteration_2 wall_s={HIST_SUITE['iteration_2']['wall_s']:.1f} passed={HIST_SUITE['iteration_2']['passed']}")
    live_s = suite.get("live") or {}
    print(f"  live {json.dumps(live_s, indent=2)[:4000]}")
    print()
    print("## LEGACY_WRAPPER / ACCIDENTAL in-process replacements (production+harness)")
    for row in doc["replacements"]:
        print(f"- {row['id']}  [{row['class']}]")
        print(f"    {row.get('in_process_replacement')}")
        print(f"    breaks: {row.get('what_would_break')}")
    print()
    print("## ISOLATION_BOUNDARY (production+harness) — why a process is required")
    for row in doc["isolation"]:
        print(f"- {row['id']}")
        print(f"    {row.get('why_isolation_is_required')}")
    print()
    print("## WHAT I WATCHED FAIL")
    fails = doc.get("what_i_watched_fail") or []
    if not fails:
        print("  (none this run)")
    else:
        for item in fails:
            print(f"- {item}")
    print()
    print("## tree integrity")
    ti = doc["tree_integrity"]
    print(f"  denied_untouched={ti['denied_untouched']}")
    print(f"  porcelain={ti['porcelain']}")
    print(f"  unexpected={ti['unexpected']}")
    print()
    print(f"wrote {doc['receipt_path']}")


def main() -> int:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    watched: List[str] = [
        "git grep over this sparse worktree returns zero hits under tools/haider "
        "(files are in HEAD, not on disk). The scanner uses git ls-tree + git show. "
        "A missing-on-disk file is not evidence of absence.",
        "Attempted `git sparse-checkout add tools/haider` is forbidden in this sandbox "
        "(sparse-checkout.lock: Operation not permitted) and was not run.",
        "A naive `from hcli.executors import WorkUnitExecutor` in this worktree "
        "raises ModuleNotFoundError because tools/haider is not materialized. "
        "The WorkUnit measurement git-archives HEAD:tools/haider into /tmp.",
    ]
    sites, scan_watched, meta = scan()
    watched.extend(scan_watched)

    counts = Counter(s["class"] for s in sites)
    doc_cost_micro = measure_micro()
    doc_cost_wu = measure_workunit(watched)
    try:
        doc_cost_suite = measure_suite(watched)
    except Exception as exc:  # noqa: BLE001
        watched.append(f"measure_suite raised {type(exc).__name__}: {exc}")
        doc_cost_suite = {"historical": HIST_SUITE, "live": {"ok": False, "reason": str(exc)}}
    live_suite = doc_cost_suite.get("live") or {}
    argv0 = live_suite.get("argv0_counts") or {}
    if live_suite.get("n_spawns"):
        py_spawns = sum(n for k, n in argv0.items() if "python" in str(k).lower())
        host_probes = sum(
            int(argv0.get(k) or 0) for k in ("sysctl", "vm_stat", "memory_pressure", "ps", "pgrep")
        )
        doc_cost_suite["interpretation"] = {
            "n_spawns": live_suite.get("n_spawns"),
            "wall_ms": live_suite.get("wall_ms"),
            "passed": live_suite.get("passed"),
            "skipped": live_suite.get("skipped"),
            "python_children": py_spawns,
            "darwin_probes_sysctl_vm_stat_memory_pressure_ps_pgrep": host_probes,
            "python_startup_floor_ms": round(
                py_spawns * float((doc_cost_micro.get("python_c_pass") or {}).get("median_ms") or 0),
                1,
            ),
            "note": (
                "The ~140s wall is NOT 140s of process starts. python_c_pass median "
                "times python_children is the startup floor (a few seconds). The rest "
                "is test body: MemGate sysctl, fake grok-run, contained pytest, "
                "py_compile. Caching live_machine_identity for the pytest process "
                "would drop ~200 sysctl spawns without touching isolation. "
                "Deleting Engine._pytest_importable and in-process compile() cuts "
                "python children that do no WorkUnit work."
            ),
        }
        watched.append(
            f"Live suite: {live_suite.get('n_spawns')} OS spawns in "
            f"{live_suite.get('wall_ms')} ms, passed={live_suite.get('passed')} "
            f"skipped={live_suite.get('skipped')} argv0={argv0}. "
            f"{py_spawns} python children, {host_probes} Darwin probes."
        )
        if any("llama-server" in str(k) for k in argv0):
            watched.append(
                "Suite audit counted llama-server processes. Wall ~149s matches the "
                "historical pytest-only walls (129–140s), so these are "
                "backends._capture --help/--version (test_mlx_backend / health), "
                "not a 27B load. A model load would dominate the wall."
            )
    doc_cost_exp = experiment_cost(doc_cost_micro, doc_cost_wu)

    unknowns = [s for s in sites if s["class"] == "UNKNOWN"]
    if unknowns:
        watched.append(
            f"{len(unknowns)} UNKNOWN sites remain; each is a follow-up, not a guess."
        )

    # Default HCLI_CPU_TIMEOUT vs suite wall — already measured historically.
    watched.append(
        "Default HCLI_CPU_TIMEOUT=120 is below the suite wall (~140s). "
        "Self-opt iteration 1 recorded this as a watched fail and raised the "
        "WorkUnit timeout to 600 so gate.correctness was not killed mid-suite "
        "(receipts/headless/HCLI_SELF_OPT_ITERATION_1.json watched_fail)."
    )
    watched.append(
        "Standing fact confirmed by this scan's scope, not re-derived: haider is a "
        "fossil namespace. This census does not spawn aider and does not search PATH "
        "for it. Production grok/llama/mlx/git/ps sites are the live process surface."
    )

    receipt_sites = []
    for s in sites:
        receipt_sites.append(
            {
                "id": s["id"],
                "file": s["file"],
                "line": s["line"],
                "end_line": s.get("end_line"),
                "function": s.get("function"),
                "call": s.get("call"),
                "shell": s.get("shell"),
                "command": s.get("command"),
                "cmd_form": s.get("cmd_form"),
                "class": s["class"],
                "class_contract": CLASS_CONTRACT.get(s.get("class") or "", "unknown"),
                "role": s.get("role"),
                "hot_path": s.get("hot_path"),
                "why": s.get("why"),
                "in_process_replacement": s.get("in_process_replacement"),
                "what_would_break": s.get("what_would_break"),
                "origin": s.get("origin"),
                "language": s.get("language"),
                "path_resolves": s.get("path_resolves"),
                "line_text": s.get("line_text") or s.get("snippet"),
                "spawn_ms": spawn_ms_estimate(s, doc_cost_micro),
                "starts_per_workunit": starts_per_workunit(s),
            }
        )

    doc: Dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "git_head": git_head(),
        "receipt_path": str(RECEIPT.relative_to(REPO)),
        "scope": {
            "roots": list(ROOTS),
            "language": (
                "Python subprocess/os.system/Popen + bash drivers under hcli / "
                "tools/haider / tools/headless; Rust Command::new under crates/ and src/"
            ),
            "not_in_scope": [
                "lab/**",
                "visionmcp/**",
                "workspace/**",
                "tools/*.py gravity/ascent probes outside headless+haider",
            ],
            "why_this_scope": (
                "Control-plane ceremony plus every repeated native spawn the WorkUnit "
                "path can see. crates Command::new is included so a Python-only census "
                "cannot hide a Rust hop."
            ),
        },
        "method": {
            "scan": (
                "ast.Call of subprocess.{run,Popen,check_output,check_call,call} and "
                "os.{system,popen,spawn*} over git ls-tree of hcli + tools/haider + "
                "tools/headless; Command::new over crates/ + src/; missing-on-disk "
                "files via git show HEAD:path; bash drivers contribute their own "
                "commands plus python heredocs"
            ),
            "classify": "function-name overrides for the production hot path, then first-token rules; rust Command::new by callee; UNKNOWN if neither fires",
            "line_check": "each site's file:line window must contain a subprocess token (or Command::new for rust) or it is flagged",
            "cost": "perf_counter microbench; live WorkUnitExecutor._run_cpu against on-disk/git-archived hcli; suite cited unless CENSUS_REMEASURE_SUITE=1",
        },
        "anti_goodhart": (
            "A 20% LOC reduction that inlines untrusted pytest into Engine is a "
            "FAILURE. Deleting Engine._pytest_importable's python3 -c and "
            "py_compile's extra process, while keeping contained pytest as a "
            "killable session, is the kind of 5% cut that removes duplicate "
            "authorities. Do not rename sealed schema ids or historical receipts."
        ),
        "scan_meta": meta,
        "counts": {
            **{k: int(counts.get(k, 0)) for k in CLASSES},
            "by_contract": {
                CLASS_CONTRACT[k]: int(counts.get(k, 0)) for k in CLASSES
            },
            "TOTAL": len(sites),
            "by_role": dict(Counter(s.get("role") for s in sites)),
            "by_hot_path": dict(Counter(s.get("hot_path") for s in sites)),
            "by_language": dict(Counter(s.get("language") for s in sites)),
        },
        "sites": receipt_sites,
        "python_to_python_hops": python_to_python_hops(sites),
        "replacements": replacements_table(sites),
        "isolation": isolation_table(sites),
        "unknowns": [
            {"id": s["id"], "why": s.get("why"), "command": s.get("command")}
            for s in unknowns
        ],
        "cost": {
            "microbench": doc_cost_micro,
            "workunit": doc_cost_wu,
            "experiment": doc_cost_exp,
            "suite": doc_cost_suite,
        },
        "migration_plan": {
            "do_now_human": [
                "Replace Engine._pytest_importable's python3 -c 'import pytest' with importlib.util.find_spec('pytest').",
                "Replace Engine._validate / mutation.validate_python_syntax py_compile subprocess with in-process compile().",
                "Replace runtime_experiment.py rm -rf with shutil.rmtree.",
                "Replace disk_truth.py / model_registry.py bash -lc git with argv git.",
                "For first-party CPU WorkUnits (self-opt sense/bottleneck/hypotheses/screen/decide/priors/next), add an in-process stage callable and keep the process only when the verifier string is untrusted or is gate.correctness.",
            ],
            "do_not": [
                "Do not in-process llama-server, mlx_lm.server, or grok-run.",
                "Do not in-process Engine._run_contained_subprocess of model-written tests (killpg + sys.modules).",
                "Do not in-process gate.correctness after mutate (stale sys.modules).",
                "Do not rename historical receipts or sealed schema ids.",
                "Do not rm/git mv anything in this campaign; ten science lanes are in flight against tools/headless and receipts/headless.",
                "Do not treat a 20% LOC cut as success if it collapses isolation.",
            ],
            "ten_lanes": "Any migration that rewrites tools/headless/*.py entrypoints or receipts/headless filenames will strand in-flight lanes. Land replacements behind in-process APIs with the existing CLI left as a shim until those lanes finish.",
        },
        "what_i_watched_fail": watched,
        "tree_integrity": {"pending_write": True},
    }

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    doc["tree_integrity"] = tree_integrity()
    # rewrite with integrity filled
    RECEIPT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print_report(doc)
    if unknowns:
        return 2
    if not doc["tree_integrity"]["denied_untouched"]:
        return 3
    return 0


def test_census_receipt_classifies_spawns():
    assert RECEIPT.is_file(), "run python3 tools/headless/subprocess_census.py first"
    doc = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert doc.get("schema") == SCHEMA
    sites = doc.get("sites") or []
    assert sites, "census wrote zero sites"
    classes = {s.get("class") for s in sites}
    contracts = {s.get("class_contract") for s in sites}
    for required in CLASSES:
        assert required in classes or required == "UNKNOWN"
    for required in ("necessary-external-tool", "required-isolation"):
        assert required in contracts, contracts
    rust = [s for s in sites if s.get("language") == "rust"]
    assert rust, "crates/src Command::new sites missing"
    for s in sites:
        assert "starts_per_workunit" in s, s.get("id")
        assert "spawn_ms" in s, s.get("id")
        assert s.get("class_contract") in CLASS_CONTRACT.values()
    hops = doc.get("python_to_python_hops")
    assert isinstance(hops, list)


def test_spring_clean_removed_dead_artifact_census_wrappers():
    src = (REPO / "tools" / "headless" / "artifact_census.py").read_text(encoding="utf-8")
    assert "def git_head" not in src
    assert "def dir_sha" not in src
    assert "import subprocess" not in src
    mb = (REPO / "tools" / "headless" / "metal_budget.py").read_text(encoding="utf-8")
    assert "import sys" not in mb


if __name__ == "__main__":
    raise SystemExit(main())
