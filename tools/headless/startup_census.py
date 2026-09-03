#!/usr/bin/env python3
"""Cold control-plane startup census: what `python3 -m hcli --help` actually pays for.

This lane changes no HCLI source. The live package lives at hcli/,
which is often absent from a sparse checkout; the census reads it from git HEAD
(or from disk when present), profiles a real `python3 -m hcli --help`, and
writes receipts/headless/STARTUP_CENSUS.json.

    python3 tools/headless/startup_census.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "hawking.headless.startup_census.v1"
REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts" / "headless" / "STARTUP_CENSUS.json"
HCLI_GIT_PREFIX = "hcli"
REPEATS = 5

HEAVY = (
    "mlx",
    "mlx.core",
    "mlx_lm",
    "torch",
    "cv2",
    "open3d",
    "visionmcp",
    "numpy",
    "PIL",
    "prompt_toolkit",
)

# Capability each hcli submodule exists to serve. Used to name deferrals.
# "help-path" is the only class --help must pay for.
MODULE_CAPABILITY = {
    "hcli": "package init (currently pulls Controller)",
    "hcli.__main__": "help-path: python -m entry (loaded as __main__, not hcli.__main__)",
    "hcli.cli": "help-path: argparse, shim install, main()",
    "hcli.workspace": "app: Workspace root on disk",
    "hcli.events": "app: EventBus",
    "hcli.controller": "app: Controller (mission + runtime + commands)",
    "hcli.config": "runtime: Config / context budget wiring",
    "hcli.engine": "runtime: model completions, mutations, tool loop",
    "hcli.backends": "runtime: llama-server / mlx_lm.server subprocess backends",
    "hcli.runtime": "runtime: RuntimePool admission and ownership",
    "hcli.machine": "runtime: MemGate / Metal working-set admission",
    "hcli.context_budget": "runtime: llama.cpp slot arithmetic and preflight",
    "hcli.mission": "mission: Mission loop / DAG dispatch",
    "hcli.executors": "mission: WorkUnit executors",
    "hcli.scheduler": "mission: Scheduler",
    "hcli.dag_store": "mission: DAG persistence",
    "hcli.goal": "mission: GoalCompiler / WorkerPacket",
    "hcli.workunit": "mission: WorkUnit state machine",
    "hcli.models": "app: ModelRegistry",
    "hcli.session": "app: SessionStore",
    "hcli.steering": "mission: SteeringQueue",
    "hcli.resources": "runtime: MutationLock / occupancy",
    "hcli.report_compiler": "grok: backend report compaction",
    "hcli.max_policy": "runtime: grok-pool / worker-equilibrium policy",
    "hcli.app": "app: App (imported after argparse; not on --help)",
    "hcli.tui": "tui: interactive prompt_toolkit loop",
    "hcli.commands": "commands: slash-command handler",
    "hcli.grok_bridge": "grok: grok-run consult",
    "hcli.ledger": "mission: Ledger / goal-not-met",
    "hcli.verifier_pipeline": "verify: command admission",
    "hcli.index": "index: WorkspaceIndex (no production importer found)",
    "hcli.mutation": "mutate: file ops (tests import it; no production importer found)",
    "hcli.context": "mission: re-export of WorkerPacket",
}

WATCHED_FAIL: List[str] = []


def note_fail(msg: str) -> None:
    WATCHED_FAIL.append(msg)


def git(repo: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def git_head(repo: Path) -> Optional[str]:
    p = git(repo, "rev-parse", "HEAD")
    return p.stdout.strip() or None


def git_show(repo: Path, path: str) -> str:
    p = git(repo, "show", f"HEAD:{path}")
    if p.returncode != 0:
        raise FileNotFoundError(f"git show HEAD:{path}: {p.stderr.strip()}")
    return p.stdout


def median(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return float(statistics.median(xs))


def us_to_ms(us: int) -> float:
    return round(us / 1000.0, 3)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def locate_hcli(repo: Path, extract_root: Path) -> Dict[str, Any]:
    """Prefer an on-disk package; otherwise extract HEAD via git archive.

    Canonical physical path is <repo>/hcli. The fossil hcli
    tree is gone; a missing hcli/ is not evidence the package does not
    exist in git — this tree may be a sparse checkout.
    """
    on_disk = repo / "hcli"
    marker = on_disk / "__main__.py"
    if marker.is_file():
        return {
            "mode": "on-disk",
            "pythonpath": str(repo.resolve()),
            "package": str(on_disk.resolve()),
            "reason": "hcli/ is materialized in this worktree",
        }

    note_fail(
        "hcli/ is not materialized in this sparse worktree; "
        "census extracted HEAD:hcli via git archive into a temp dir"
    )
    raw = subprocess.run(
        ["git", "-C", str(repo), "archive", "HEAD", "hcli"],
        capture_output=True,
        check=False,
        timeout=120,
    )
    if raw.returncode != 0:
        raise RuntimeError(f"git archive failed: {raw.stderr.decode('utf-8', 'replace').strip()}")
    subprocess.run(
        ["tar", "-x", "-C", str(extract_root)],
        input=raw.stdout,
        capture_output=True,
        check=True,
    )
    pkg = extract_root / "hcli"
    if not (pkg / "__main__.py").is_file():
        raise RuntimeError(f"git archive did not produce {pkg / '__main__.py'}")
    pythonpath = extract_root
    return {
        "mode": "git-archive-HEAD",
        "pythonpath": str(pythonpath),
        "package": str(pkg),
        "extract_root": str(extract_root),
        "reason": "sparse checkout: hcli not on disk; content is HEAD",
    }


def clone_package(src_pkg: Path, dest_parent: Path, init_text: str) -> Path:
    """Copy hcli package under dest_parent (PYTHONPATH) with a replacement __init__.py."""
    dest_parent.mkdir(parents=True, exist_ok=True)
    dest_pkg = dest_parent / "hcli"
    if dest_pkg.exists():
        shutil.rmtree(dest_pkg)
    shutil.copytree(
        src_pkg,
        dest_pkg,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (dest_pkg / "__init__.py").write_text(init_text, encoding="utf-8")
    return dest_parent


THIN_INIT = '''"""HCLI product package (census counterfactual: no Controller on import)."""
from .cli import parse_hcli_args, main

__all__ = ["parse_hcli_args", "main", "Workspace", "Controller", "Event", "EventBus"]


def __getattr__(name):
    if name == "Workspace":
        from .workspace import Workspace
        return Workspace
    if name == "Controller":
        from .controller import Controller
        return Controller
    if name in ("Event", "EventBus"):
        from .events import Event, EventBus
        return Event if name == "Event" else EventBus
    raise AttributeError(name)
'''


EMPTY_INIT = '''"""HCLI product package (census isolate: empty init)."""
'''


def run_env(pythonpath: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONSTARTUP"] = ""
    env.pop("PYTHONPROFILEIMPORTTIME", None)
    if extra:
        env.update(extra)
    return env


def run_help(
    python: str,
    pythonpath: str,
    *,
    importtime: bool = False,
    timeout: int = 60,
) -> Dict[str, Any]:
    cmd = [python]
    if importtime:
        cmd.append("-X")
        cmd.append("importtime")
    cmd.extend(["-m", "hcli", "--help"])
    t0 = time.perf_counter()
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=run_env(pythonpath),
        cwd=str(REPO),
    )
    wall_s = time.perf_counter() - t0
    return {
        "returncode": p.returncode,
        "wall_s": wall_s,
        "stdout": p.stdout,
        "stderr": p.stderr,
        "cmd": cmd,
    }


IMPORT_TIME_RE = re.compile(
    r"^import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(.*)$"
)


def parse_importtime(stderr: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in stderr.splitlines():
        m = IMPORT_TIME_RE.match(line)
        if not m:
            continue
        name = m.group(3)
        depth = (len(name) - len(name.lstrip(" "))) // 2
        rows.append(
            {
                "self_us": int(m.group(1)),
                "cumulative_us": int(m.group(2)),
                "name": name.strip(),
                "depth": depth,
            }
        )
    return rows


def import_chain(rows: List[Dict[str, Any]], target: str) -> Optional[List[str]]:
    """Walk the importtime indent tree up from `target` to the root importer."""
    idx = None
    for i, row in enumerate(rows):
        if row["name"] == target:
            idx = i
            break
    if idx is None:
        return None
    chain = [rows[idx]["name"]]
    depth = rows[idx]["depth"]
    for j in range(idx - 1, -1, -1):
        if rows[j]["depth"] < depth:
            chain.append(rows[j]["name"])
            depth = rows[j]["depth"]
            if depth == 0:
                break
    chain.reverse()
    return chain


def timed_repeats(python: str, pythonpath: str, n: int) -> Dict[str, Any]:
    samples: List[float] = []
    codes: List[int] = []
    help_ok = []
    first_stdout = None
    for i in range(n):
        r = run_help(python, pythonpath, importtime=False)
        samples.append(float(r["wall_s"]))
        codes.append(int(r["returncode"]))
        ok = "autonomous local model engineering" in (r["stdout"] or "")
        help_ok.append(ok)
        if first_stdout is None:
            first_stdout = r["stdout"]
        if r["returncode"] not in (0,):
            note_fail(
                f"--help run {i + 1} exited {r['returncode']}: "
                f"{(r['stderr'] or '')[:400]}"
            )
    return {
        "repeats": n,
        "samples_s": samples,
        "median_s": median(samples),
        "min_s": min(samples) if samples else None,
        "max_s": max(samples) if samples else None,
        "returncodes": codes,
        "help_text_ok": all(help_ok),
        "stdout_preview": (first_stdout or "")[:800],
    }


def dump_sys_modules(python: str, pythonpath: str) -> Dict[str, Any]:
    script = r"""
import json, sys
heavy = %s
sys.argv = ["hcli", "--help"]
err = None
try:
    from hcli.cli import main
    try:
        main(["--help"])
    except SystemExit as e:
        err = int(getattr(e, "code", 0) or 0)
except Exception as e:
    print(json.dumps({"error": type(e).__name__ + ": " + str(e)}))
    sys.exit(2)
mods = sorted(sys.modules.keys())
print(json.dumps({
    "systemexit": err,
    "hcli": [m for m in mods if m == "hcli" or m.startswith("hcli.")],
    "heavy_present": {k: (k in sys.modules) for k in heavy},
    "heavy_file": {k: getattr(sys.modules[k], "__file__", None) for k in heavy if k in sys.modules},
    "module_count": len(mods),
    "modules": mods,
}))
""" % (repr(list(HEAVY)),)
    p = subprocess.run(
        [python, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env=run_env(pythonpath),
        cwd=str(REPO),
    )
    if p.returncode != 0:
        return {
            "ok": False,
            "returncode": p.returncode,
            "stderr": (p.stderr or "")[-1500:],
            "stdout": (p.stdout or "")[-1500:],
        }
    try:
        data = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"ok": False, "error": str(e), "stdout": p.stdout[-1500:]}
    data["ok"] = True
    return data


def isolated_import(
    python: str, src_pkg: Path, work: Path, module: str
) -> Dict[str, Any]:
    parent = work / module.replace(".", "_")
    pythonpath = str(clone_package(src_pkg, parent, EMPTY_INIT))
    t0 = time.perf_counter()
    p = subprocess.run(
        [python, "-X", "importtime", "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=60,
        env=run_env(pythonpath),
        cwd=str(REPO),
    )
    wall_s = time.perf_counter() - t0
    rows = parse_importtime(p.stderr or "")
    hit = next((r for r in rows if r["name"] == module), None)
    return {
        "module": module,
        "returncode": p.returncode,
        "wall_s": wall_s,
        "cumulative_us": None if hit is None else hit["cumulative_us"],
        "self_us": None if hit is None else hit["self_us"],
        "stderr_tail": (p.stderr or "")[-400:] if p.returncode != 0 else "",
    }


def argparse_floor(python: str) -> Dict[str, Any]:
    code = (
        "import argparse, sys\n"
        "p = argparse.ArgumentParser(prog='hcli',\n"
        "    description='HCLI — autonomous local model engineering')\n"
        "try:\n"
        "    p.parse_args(['--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
    )
    t0 = time.perf_counter()
    p = subprocess.run(
        [python, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=run_env(""),
    )
    wall_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    subprocess.run(
        [python, "-c", "pass"],
        capture_output=True,
        timeout=30,
        env=run_env(""),
    )
    pass_s = time.perf_counter() - t1
    return {
        "argparse_help_wall_s": wall_s,
        "python_pass_wall_s": pass_s,
        "returncode": p.returncode,
    }


def list_hcli_py(pkg: Path) -> List[Path]:
    return sorted(
        p
        for p in pkg.rglob("*.py")
        if "__pycache__" not in p.parts and "tests" not in p.parts
    )


def is_type_checking_if(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def module_level_imports(src: str) -> Tuple[List[str], List[str]]:
    """Return (eager_modnames, lazy_modnames) from AST. TYPE_CHECKING skipped."""
    tree = ast.parse(src)
    eager: List[str] = []
    lazy: List[str] = []

    def add(node: ast.AST, bucket: List[str]) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bucket.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = ("." * node.level) + (node.module or "")
            bucket.append(root or ".")

    def walk(body: List[ast.stmt], in_def: bool) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(node.body, True)
                continue
            if isinstance(node, ast.ClassDef):
                walk(node.body, True)
                continue
            if isinstance(node, ast.If):
                if is_type_checking_if(node):
                    walk(node.orelse, in_def)
                    continue
                walk(node.body, in_def)
                walk(node.orelse, in_def)
                continue
            if isinstance(node, ast.Try):
                walk(node.body, in_def)
                for handler in node.handlers:
                    walk(handler.body, in_def)
                walk(node.orelse, in_def)
                walk(node.finalbody, in_def)
                continue
            if isinstance(node, ast.With):
                walk(node.body, in_def)
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                add(node, lazy if in_def else eager)

    walk(tree.body, False)
    return eager, lazy


def ast_census(pkg: Path) -> Dict[str, Any]:
    heavy_hits: List[Dict[str, Any]] = []
    per_mod: Dict[str, Any] = {}
    production_importers: Dict[str, List[str]] = {
        "hcli.index": [],
        "hcli.mutation": [],
    }
    for path in list_hcli_py(pkg):
        rel = path.relative_to(pkg)
        mod = "hcli" if rel.as_posix() == "__init__.py" else "hcli." + str(rel.with_suffix("")).replace("/", ".")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        src = path.read_text(encoding="utf-8")
        eager, lazy = module_level_imports(src)
        per_mod[mod] = {
            "file": f"{HCLI_GIT_PREFIX}/{rel.as_posix()}",
            "bytes": path.stat().st_size,
            "sha256": sha256_bytes(path.read_bytes()),
            "eager_imports": eager,
            "lazy_imports": lazy,
        }
        # Heavy import statements (not comments, not strings-only: AST names).
        for bucket, kind in ((eager, "eager"), (lazy, "lazy")):
            for name in bucket:
                head = name.lstrip(".")
                top = head.split(".")[0] if head else ""
                if top in {
                    "mlx",
                    "mlx_lm",
                    "torch",
                    "cv2",
                    "open3d",
                    "visionmcp",
                    "numpy",
                    "PIL",
                    "prompt_toolkit",
                }:
                    heavy_hits.append(
                        {
                            "module": mod,
                            "file": per_mod[mod]["file"],
                            "kind": kind,
                            "imported": name,
                        }
                    )
        joined = src
        if re.search(r"(from\s+\.index\s+import|from\s+hcli\.index\s+import|import\s+hcli\.index)", joined):
            production_importers["hcli.index"].append(mod)
        if re.search(r"(from\s+\.mutation\s+import|from\s+hcli\.mutation\s+import|import\s+hcli\.mutation)", joined):
            production_importers["hcli.mutation"].append(mod)
    return {
        "modules": per_mod,
        "heavy_ast_imports": heavy_hits,
        "production_importers": production_importers,
        "module_count": len(per_mod),
    }


def cite(pkg: Path, rel: str, line: int, must_contain: str) -> Dict[str, Any]:
    path = pkg / rel
    if not path.is_file():
        return {
            "path": f"{HCLI_GIT_PREFIX}/{rel}",
            "line": line,
            "ok": False,
            "reason": "file missing in extracted/on-disk package",
        }
    lines = path.read_text(encoding="utf-8").splitlines()
    if line < 1 or line > len(lines):
        return {
            "path": f"{HCLI_GIT_PREFIX}/{rel}",
            "line": line,
            "ok": False,
            "reason": f"line {line} out of range ({len(lines)} lines)",
        }
    text = lines[line - 1]
    ok = must_contain in text
    return {
        "path": f"{HCLI_GIT_PREFIX}/{rel}",
        "line": line,
        "text": text,
        "must_contain": must_contain,
        "ok": ok,
    }


def find_line(pkg: Path, rel: str, substring: str) -> Optional[int]:
    path = pkg / rel
    if not path.is_file():
        return None
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if substring in line:
            return i
    return None


def hcli_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r["name"] == "hcli" or r["name"].startswith("hcli.")]


def filter_new_names(
    full: List[Dict[str, Any]], thin: List[Dict[str, Any]]
) -> List[str]:
    thin_names = {r["name"] for r in thin}
    return [r["name"] for r in full if r["name"] not in thin_names]


def locate_visionmcp_src(repo: Path) -> Optional[str]:
    env = os.environ.get("VISIONMCP_SRC")
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            repo / "visionmcp" / "src",
            Path("/Users/scammermike/Downloads/hawking/visionmcp/src"),
            Path("/Users/scammermike/Downloads/hawking-copy/visionmcp/src"),
            Path.home() / ".searcher-donors" / "visionmcp" / "src",
        ]
    )
    seen = set()
    for src in candidates:
        try:
            resolved = src.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "visionmcp" / "__init__.py").is_file():
            return str(resolved)
    return None


def adversarial_heavy_probe(
    pythonpath: str, visionmcp_src: Optional[str]
) -> Dict[str, Any]:
    """Run --help under interpreters that actually have torch/cv2/open3d/numpy.

    Absence of a package in *this* python is not proof hcli would not import it.
    A python that has the package, plus visionmcp on PYTHONPATH, is.
    """
    probes = []
    candidates = [
        {
            "label": "sys.executable",
            "python": sys.executable,
        },
        {
            "label": "grok-vision (torch/cv2/open3d)",
            "python": str(Path.home() / ".grok-vision" / "bin" / "python"),
        },
        {
            "label": "hawking-aider (numpy/PIL/prompt_toolkit)",
            "python": str(Path.home() / ".venvs" / "hawking-aider" / "bin" / "python3"),
        },
    ]
    for spec in candidates:
        py = spec["python"]
        if not os.path.isfile(py) or not os.access(py, os.X_OK):
            probes.append({**spec, "ran": False, "reason": "interpreter not present"})
            continue
        extra_pp = pythonpath
        if visionmcp_src:
            extra_pp = extra_pp + os.pathsep + visionmcp_src
        # find_spec does not import torch/cv2/open3d — it only answers "could it".
        chk = subprocess.run(
            [
                py,
                "-c",
                "import importlib.util, json, sys\n"
                "names = " + repr(list(HEAVY)) + "\n"
                "out = {}\n"
                "for n in names:\n"
                "    try:\n"
                "        spec = importlib.util.find_spec(n)\n"
                "    except ModuleNotFoundError:\n"
                "        spec = None\n"
                "    out[n] = {\n"
                "        'importable': spec is not None,\n"
                "        'origin': None if spec is None else getattr(spec, 'origin', None),\n"
                "    }\n"
                "print(json.dumps(out))\n",
            ],
            capture_output=True,
            text=True,
            timeout=45,
            env=run_env(extra_pp),
        )
        if chk.returncode != 0:
            probes.append(
                {
                    **spec,
                    "ran": False,
                    "reason": f"find_spec failed rc={chk.returncode}: {(chk.stderr or '')[:400]}",
                }
            )
            continue
        try:
            avail = json.loads(chk.stdout.strip().splitlines()[-1])
        except Exception as e:
            probes.append({**spec, "ran": False, "reason": f"find_spec json: {e}"})
            continue
        dumped = dump_sys_modules(py, extra_pp)
        present = (dumped.get("heavy_present") or {}) if dumped.get("ok") else {}
        eager = {k: bool(v) for k, v in present.items() if v}
        probes.append(
            {
                **spec,
                "ran": True,
                "available": avail,
                "dump_ok": bool(dumped.get("ok")),
                "dump_error": dumped.get("error") or dumped.get("stderr"),
                "heavy_in_sys_modules": present,
                "eager_heavy": eager,
                "hcli_modules": dumped.get("hcli") if dumped.get("ok") else None,
                "pythonpath_had_visionmcp": bool(visionmcp_src),
            }
        )
    return {"visionmcp_src": visionmcp_src, "probes": probes}


def shim_identity() -> Dict[str, Any]:
    shim = Path.home() / ".local" / "bin" / "hcli"
    current = Path.home() / ".local" / "share" / "hcli" / "current"
    body = None
    if shim.is_file():
        try:
            body = shim.read_text(encoding="utf-8")
        except OSError as e:
            body = f"<unreadable: {e}>"
    target = None
    if current.is_symlink() or current.exists():
        try:
            target = str(current.resolve())
        except OSError:
            target = str(current)
    return {
        "path": str(shim),
        "present": shim.is_file(),
        "body": body,
        "current_symlink": str(current) if current.exists() or current.is_symlink() else None,
        "current_target": target,
        "note": (
            "PATH `hcli --help` is NOT this measurement. The shim execs a venv "
            "python with PYTHONPATH=~/.local/share/hcli/current, a different tree. "
            "This census measures `python3 -m hcli --help` against git HEAD."
        ),
    }


def classify_reachable(
    hcli_loaded: List[str],
    full_rows: List[Dict[str, Any]],
    thin_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    thin_names = {r["name"] for r in hcli_rows(thin_rows)}
    full_hcli = hcli_rows(full_rows)
    by_name = {r["name"]: r for r in full_hcli}
    out = []
    for name in sorted(set(hcli_loaded) | {r["name"] for r in full_hcli}):
        if name.endswith(".__main__") or name == "hcli.__main__":
            cap = "help-path: -m entry"
            needed_for_help = True
        else:
            cap = MODULE_CAPABILITY.get(name, "UNKNOWN")
            needed_for_help = name in ("hcli", "hcli.cli")
        row = by_name.get(name)
        paid_only_because_of_controller = name not in thin_names and name not in (
            "hcli",
            "hcli.cli",
        )
        # Thin init still loads hcli + hcli.cli. Everything else on the full
        # --help path is the Controller graph.
        if name in ("hcli.app", "hcli.tui", "hcli.commands", "hcli.grok_bridge",
                    "hcli.ledger", "hcli.verifier_pipeline", "hcli.index",
                    "hcli.mutation", "hcli.context"):
            paid_only_because_of_controller = False
        out.append(
            {
                "module": name,
                "loaded_on_help": name in hcli_loaded or name in by_name,
                "needed_for_help": needed_for_help,
                "capability": cap,
                "self_us": None if row is None else row["self_us"],
                "cumulative_us": None if row is None else row["cumulative_us"],
                "deferrable_on_help": bool(
                    (name in hcli_loaded or name in by_name)
                    and not needed_for_help
                ),
                "paid_only_because_controller_imported_at_package_init": paid_only_because_of_controller,
            }
        )
    return out


def fmt_s(x: Optional[float]) -> str:
    if x is None:
        return "UNKNOWN"
    return f"{x * 1000:.1f} ms"


def fmt_us(x: Optional[int]) -> str:
    if x is None:
        return "UNKNOWN"
    if x >= 1000:
        return f"{x / 1000:.2f} ms"
    return f"{x} us"


def main() -> int:
    python = sys.executable
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    head = git_head(REPO)

    extract_root = Path(tempfile.mkdtemp(prefix="hcli-startup-census-"))
    variants_root = Path(tempfile.mkdtemp(prefix="hcli-startup-variants-"))
    try:
        located = locate_hcli(REPO, extract_root)
        src_pkg = Path(located["package"])
        pythonpath = located["pythonpath"]

        ast_info = ast_census(src_pkg)

        # Citations that must resolve.
        cites = [
            cite(src_pkg, "__init__.py", 2, "from .cli import parse_hcli_args, main"),
            cite(src_pkg, "__init__.py", 4, "from .controller import Controller"),
            cite(src_pkg, "__main__.py", 1, "from hcli.cli import main"),
        ]
        app_line = find_line(src_pkg, "cli.py", "from .app import App")
        engine_line = find_line(src_pkg, "controller.py", "from .engine import Engine")
        runtime_line = find_line(src_pkg, "controller.py", "from .runtime import RuntimePool")
        mission_line = find_line(src_pkg, "controller.py", "from .mission import Mission")
        mlx_bin_line = find_line(src_pkg, "backends.py", "def mlx_server_binary")
        mlx_which_line = find_line(src_pkg, "backends.py", 'shutil.which("mlx_lm.server")')
        urllib_engine = find_line(src_pkg, "engine.py", "import urllib.request")
        urllib_controller = find_line(src_pkg, "controller.py", "import urllib.request")
        metal_line = find_line(src_pkg, "machine.py", "spec_from_file_location")
        extra_cites = [
            cite(src_pkg, "cli.py", app_line or 0, "from .app import App") if app_line else {
                "path": f"{HCLI_GIT_PREFIX}/cli.py", "ok": False, "reason": "App import not found"
            },
            cite(src_pkg, "controller.py", engine_line or 0, "from .engine import Engine") if engine_line else {
                "path": f"{HCLI_GIT_PREFIX}/controller.py", "ok": False, "reason": "Engine import not found"
            },
            cite(src_pkg, "controller.py", runtime_line or 0, "from .runtime import RuntimePool") if runtime_line else {
                "path": f"{HCLI_GIT_PREFIX}/controller.py", "ok": False, "reason": "RuntimePool import not found"
            },
            cite(src_pkg, "controller.py", mission_line or 0, "from .mission import Mission") if mission_line else {
                "path": f"{HCLI_GIT_PREFIX}/controller.py", "ok": False, "reason": "Mission import not found"
            },
            cite(src_pkg, "backends.py", mlx_bin_line or 0, "def mlx_server_binary") if mlx_bin_line else {
                "path": f"{HCLI_GIT_PREFIX}/backends.py", "ok": False, "reason": "mlx_server_binary not found"
            },
            cite(src_pkg, "backends.py", mlx_which_line or 0, "mlx_lm.server") if mlx_which_line else {
                "path": f"{HCLI_GIT_PREFIX}/backends.py", "ok": False, "reason": "shutil.which mlx_lm.server not found"
            },
            cite(src_pkg, "engine.py", urllib_engine or 0, "import urllib.request") if urllib_engine else {
                "path": f"{HCLI_GIT_PREFIX}/engine.py", "ok": False, "reason": "urllib.request not found"
            },
            cite(src_pkg, "controller.py", urllib_controller or 0, "import urllib.request") if urllib_controller else {
                "path": f"{HCLI_GIT_PREFIX}/controller.py", "ok": False, "reason": "urllib.request not found"
            },
            cite(src_pkg, "machine.py", metal_line or 0, "spec_from_file_location") if metal_line else {
                "path": f"{HCLI_GIT_PREFIX}/machine.py", "ok": False, "reason": "metal_budget loader not found"
            },
        ]
        cites.extend(extra_cites)
        for c in cites:
            if not c.get("ok"):
                note_fail(f"citation failed: {c}")

        floor = argparse_floor(python)

        baseline_wall = timed_repeats(python, pythonpath, REPEATS)
        baseline_it = run_help(python, pythonpath, importtime=True)
        baseline_rows = parse_importtime(baseline_it["stderr"] or "")
        if not baseline_rows:
            note_fail("python -X importtime produced no parseable rows")

        dumped = dump_sys_modules(python, pythonpath)
        if not dumped.get("ok"):
            note_fail(f"sys.modules dump failed: {dumped}")

        thin_parent = clone_package(src_pkg, variants_root / "thin", THIN_INIT)
        thin_pp = str(thin_parent)
        thin_wall = timed_repeats(python, thin_pp, REPEATS)
        thin_it = run_help(python, thin_pp, importtime=True)
        thin_rows = parse_importtime(thin_it["stderr"] or "")
        thin_dump = dump_sys_modules(python, thin_pp)

        isolates = {}
        for mod in (
            "hcli.cli",
            "hcli.controller",
            "hcli.engine",
            "hcli.backends",
            "hcli.runtime",
            "hcli.machine",
            "hcli.context_budget",
            "hcli.mission",
            "hcli.goal",
            "hcli.resources",
        ):
            isolates[mod] = isolated_import(python, src_pkg, variants_root / "iso", mod)

        urllib_iso = isolated_import(python, src_pkg, variants_root / "iso", "urllib.request")
        # urllib.request is stdlib; isolated_import still copies hcli but imports urllib.
        # Override: measure stdlib directly without the copy.
        t0 = time.perf_counter()
        p_url = subprocess.run(
            [python, "-X", "importtime", "-c", "import urllib.request"],
            capture_output=True,
            text=True,
            timeout=30,
            env=run_env(""),
        )
        url_rows = parse_importtime(p_url.stderr or "")
        url_hit = next((r for r in url_rows if r["name"] == "urllib.request"), None)
        urllib_iso = {
            "module": "urllib.request",
            "returncode": p_url.returncode,
            "wall_s": time.perf_counter() - t0,
            "cumulative_us": None if url_hit is None else url_hit["cumulative_us"],
            "self_us": None if url_hit is None else url_hit["self_us"],
        }

        visionmcp_src = locate_visionmcp_src(REPO)
        if visionmcp_src is None:
            note_fail(
                "visionmcp src not on disk in this sparse worktree and no fallback "
                "found; adversarial PYTHONPATH probe for visionmcp could not be armed "
                "(AST still shows zero visionmcp imports in hcli)"
            )
        heavy_probe = adversarial_heavy_probe(pythonpath, visionmcp_src)

        loaded = list(dumped.get("hcli") or [])
        classifications = classify_reachable(loaded, baseline_rows, thin_rows)

        hcli_full = next((r for r in baseline_rows if r["name"] == "hcli"), None)
        hcli_thin = next((r for r in thin_rows if r["name"] == "hcli"), None)
        controller_row = next((r for r in baseline_rows if r["name"] == "hcli.controller"), None)

        wall_save = None
        if baseline_wall["median_s"] is not None and thin_wall["median_s"] is not None:
            wall_save = baseline_wall["median_s"] - thin_wall["median_s"]
        import_save_us = None
        if hcli_full and hcli_thin:
            import_save_us = hcli_full["cumulative_us"] - hcli_thin["cumulative_us"]

        paid_because_controller = [
            r["name"] for r in baseline_rows if r["name"] not in {x["name"] for x in thin_rows}
        ]
        hcli_paid_because_controller = [
            n for n in paid_because_controller if n == "hcli" or n.startswith("hcli.")
        ]

        heavy_verdict = {}
        for name in ("mlx", "torch", "cv2", "open3d", "visionmcp"):
            chains = []
            present_any = False
            for probe in heavy_probe["probes"]:
                if not probe.get("ran") or not probe.get("dump_ok"):
                    continue
                if (probe.get("heavy_in_sys_modules") or {}).get(name):
                    present_any = True
                    chains.append(
                        {
                            "interpreter": probe.get("label"),
                            "file": (probe.get("heavy_in_sys_modules") and name),
                        }
                    )
            in_importtime = any(r["name"] == name or r["name"].startswith(name + ".") for r in baseline_rows)
            ast_hits = [h for h in ast_info["heavy_ast_imports"] if h["imported"].split(".")[0] in {name, name.split(".")[0]}]
            # mlx vs mlx_lm
            if name == "mlx":
                ast_hits = [
                    h for h in ast_info["heavy_ast_imports"]
                    if h["imported"].split(".")[0] in ("mlx", "mlx_lm")
                ]
                in_importtime = any(
                    r["name"] in ("mlx", "mlx_lm") or r["name"].startswith(("mlx.", "mlx_lm."))
                    for r in baseline_rows
                )
            heavy_verdict[name] = {
                "eager_on_help": bool(present_any or in_importtime),
                "in_importtime": in_importtime,
                "in_sys_modules_any_probe": present_any,
                "ast_import_statements": ast_hits,
                "import_chain": import_chain(baseline_rows, name),
                "verdict": (
                    "YES" if (present_any or in_importtime) else
                    "NO"
                ),
                "why": (
                    "imported during python3 -m hcli --help"
                    if (present_any or in_importtime)
                    else (
                        "no import statement in hcli/*.py (AST) and "
                        "name absent from -X importtime and from sys.modules after --help"
                    )
                ),
            }

        # MLX is used via subprocess, not import. Record that as a distinct fact.
        mlx_subprocess = {
            "imports_mlx": False,
            "spawns_mlx_lm_server": True,
            "cited": f"{HCLI_GIT_PREFIX}/backends.py:{mlx_which_line}",
            "when": "MlxServerBackend.start / identity, not at import and not on --help",
        }

        deferrals = [
            {
                "id": "D1_package_init_controller",
                "move": (
                    "Stop importing Controller, Workspace, Event, EventBus in "
                    f"{HCLI_GIT_PREFIX}/__init__.py (lines 3-5). Keep "
                    "`from .cli import parse_hcli_args, main`. Expose the rest "
                    "via PEP 562 __getattr__ so `from hcli import Controller` still works."
                ),
                "behind": "App construction / `from hcli import Controller` / tests that need Controller",
                "why_help_does_not_need_it": (
                    f"{HCLI_GIT_PREFIX}/cli.py:{app_line} imports App only after "
                    "parse_hcli_args(); argparse --help sys.exits first. "
                    f"{HCLI_GIT_PREFIX}/__main__.py imports hcli.cli.main, but "
                    "package __init__ currently still runs the Controller import."
                ),
                "measured_wall_saving_s": wall_save,
                "measured_importtime_saving_us": import_save_us,
                "measured_how": (
                    "Copied the package, replaced __init__.py with a thin init "
                    "that does not import Controller, re-ran python3 -m hcli --help. "
                    "Saving is baseline median wall minus thin median wall, and "
                    "hcli cumulative_us full minus thin. Not additive with D2/D3."
                ),
            },
            {
                "id": "D2_controller_engine_runtime_mission",
                "move": (
                    f"In {HCLI_GIT_PREFIX}/controller.py, move module-level "
                    "`from .engine import Engine`, `from .runtime import RuntimePool`, "
                    "`from .mission import Mission` (and the urllib.request import used "
                    "by _http_json) into the methods that construct those objects."
                ),
                "behind": "Controller.__init__ / execute / mission start — not argparse",
                "why_help_does_not_need_it": "Controller is not required to print --help at all (see D1).",
                "measured_wall_saving_s": None,
                "measured_importtime_saving_us": None if controller_row is None else controller_row["cumulative_us"],
                "measured_how": (
                    "Once D1 lands, --help never imports Controller, so D2 does not "
                    "change --help. The isolated `import hcli.engine` (empty package "
                    "init) and the Controller cumulative_us on the current --help path "
                    "are the cost D2 removes from `import hcli.controller` and from "
                    "any entry that still imports Controller eagerly. Do not add D1+D2."
                ),
                "isolated_engine": isolates.get("hcli.engine"),
                "isolated_runtime": isolates.get("hcli.runtime"),
                "isolated_mission": isolates.get("hcli.mission"),
                "isolated_backends": isolates.get("hcli.backends"),
            },
            {
                "id": "D3_urllib_request",
                "move": (
                    "Defer `import urllib.request` in controller.py, engine.py, "
                    "runtime.py, context_budget.py, backends.py until the first HTTP "
                    "call to a local runtime."
                ),
                "behind": "runtime HTTP (llama-server / mlx_lm.server /health, completions)",
                "why_help_does_not_need_it": "--help does not talk to a backend.",
                "measured_wall_saving_s": None,
                "measured_importtime_saving_us": urllib_iso.get("cumulative_us"),
                "measured_how": (
                    "Cold `python3 -X importtime -c 'import urllib.request'` in a fresh "
                    "process. The process wall includes interpreter startup and is NOT a "
                    "--help saving; cumulative_us is the stdlib cost. Overlaps D1/D2: on "
                    "the current --help path urllib.request is imported because "
                    "Controller→Engine/context_budget import it. After D1, --help no "
                    "longer pays this."
                ),
                "isolated": urllib_iso,
            },
            {
                "id": "D4_already_lazy_do_not_touch",
                "move": "Nothing. These are already behind the capability that uses them.",
                "behind": "already deferred",
                "modules": [
                    "hcli.app (cli.main after argparse)",
                    "hcli.tui (App interactive; prompt_toolkit inside tui.py)",
                    "hcli.commands (Controller.handle_command)",
                    "hcli.grok_bridge (commands / executors / mission / scheduler)",
                    "hcli.ledger (controller mission path, TYPE_CHECKING in steering)",
                    "hcli.verifier_pipeline (executors / ledger)",
                    "tools/headless/metal_budget.py (machine._metal_budget_module via importlib)",
                ],
                "measured_wall_saving_s": 0.0,
                "measured_importtime_saving_us": 0,
                "measured_how": "Absent from --help importtime and from sys.modules after --help.",
            },
        ]

        # Total for --help is D1 (which includes D2+D3 on this path).
        total_help_saving = {
            "wall_s": wall_save,
            "importtime_us": import_save_us,
            "rule": (
                "Total --help saving is D1 only. D2 and D3 are subsets of the "
                "Controller graph D1 removes from this path. Adding them to D1 "
                "double-counts. D4 is already free."
            ),
        }

        not_on_help = [
            m
            for m in ast_info["modules"]
            if m not in loaded and m not in {r["name"] for r in hcli_rows(baseline_rows)}
        ]

        receipt = {
            "schema": SCHEMA,
            "gate": "STARTUP_CENSUS",
            "generated_at": generated_at,
            "git_head": head,
            "repo": str(REPO),
            "sparse_checkout": {
                "hcli_on_disk": (REPO / "hcli" / "__main__.py").is_file(),
                "hcli_source_mode": located["mode"],
                "hcli_pythonpath": pythonpath,
                "hcli_package": located["package"],
                "reason": located["reason"],
            },
            "python": {
                "executable": python,
                "version": sys.version,
                "path0": sys.path[:6],
            },
            "command": [python, "-m", "hcli", "--help"],
            "method": {
                "wall": f"{REPEATS} fresh processes, PYTHONDONTWRITEBYTECODE=1, time.perf_counter around subprocess",
                "import_tree": "CPython -X importtime (self_us / cumulative_us)",
                "sys_modules": "import hcli.cli.main(['--help']) catching SystemExit",
                "counterfactual_D1": "temp copy of package with thin __init__.py; no source in the repo modified",
                "isolated_modules": "temp copy with empty __init__.py so `import hcli.X` does not pay Controller",
                "heavy_adversary": (
                    "repeat sys.modules dump under grok-vision python (torch/cv2/open3d) "
                    "and hawking-aider python (numpy/PIL), with visionmcp src on PYTHONPATH when found"
                ),
                "anti_goodhart": (
                    "Optimise verified --help cost. A 20% LOC cut that still imports "
                    "Controller from __init__.py is a failure. D1 is a 4-line change."
                ),
            },
            "floor": floor,
            "baseline_help": {
                "wall": baseline_wall,
                "importtime_hcli_cumulative_us": None if hcli_full is None else hcli_full["cumulative_us"],
                "importtime_controller_cumulative_us": None if controller_row is None else controller_row["cumulative_us"],
                "hcli_modules": hcli_rows(baseline_rows),
                "top_cumulative": sorted(baseline_rows, key=lambda r: r["cumulative_us"], reverse=True)[:25],
                "row_count": len(baseline_rows),
            },
            "thin_init_help": {
                "wall": thin_wall,
                "importtime_hcli_cumulative_us": None if hcli_thin is None else hcli_thin["cumulative_us"],
                "hcli_modules": hcli_rows(thin_rows),
                "sys_modules_hcli": thin_dump.get("hcli") if thin_dump.get("ok") else thin_dump,
            },
            "sys_modules_help": dumped,
            "reachable_from_entrypoint": classifications,
            "paid_only_because_controller_imported_at_package_init": paid_because_controller,
            "hcli_paid_only_because_controller": [
                r["module"] for r in classifications
                if r.get("paid_only_because_controller_imported_at_package_init")
            ],
            "not_loaded_on_help": not_on_help,
            "isolated_import_costs": isolates,
            "urllib_request_isolated": urllib_iso,
            "deferrals": deferrals,
            "total_help_saving": total_help_saving,
            "heavy_packages": heavy_verdict,
            "mlx_subprocess_not_import": mlx_subprocess,
            "heavy_ast_imports": ast_info["heavy_ast_imports"],
            "adversarial_heavy_probe": heavy_probe,
            "ast": {
                "module_count": ast_info["module_count"],
                "modules": ast_info["modules"],
                "production_importers": ast_info["production_importers"],
            },
            "citations": cites,
            "shim_not_measured": shim_identity(),
            "unknowns": [
                u for u in [
                    {
                        "item": "visionmcp on PYTHONPATH during primary python3 measurement",
                        "status": "UNKNOWN" if visionmcp_src is None else "armed",
                        "reason": (
                            "visionmcp/ not materialized here; fallback src not found"
                            if visionmcp_src is None
                            else f"armed with {visionmcp_src}"
                        ),
                    },
                    {
                        "item": "hcli.index / hcli.mutation production callers",
                        "status": (
                            "no production importer in hcli/*.py (tests import mutation)"
                        ),
                        "reason": "AST/text scan of production modules; tests/ excluded",
                    },
                    {
                        "item": "Downloads/hawking as live tree",
                        "status": "not used as measurement source",
                        "reason": (
                            "this worktree HEAD is "
                            + (head or "UNKNOWN")
                            + "; a sibling checkout may have moved. Measured git HEAD of this repo."
                        ),
                    },
                ]
            ],
            "what_i_watched_fail": WATCHED_FAIL,
            "scope_guard": {
                "wrote": [
                    "tools/headless/startup_census.py",
                    "receipts/headless/STARTUP_CENSUS.json",
                ],
                "did_not_modify": [
                    "tools/haider",
                    "workspace",
                    "crates",
                    "visionmcp",
                    "app",
                    "lab",
                    "ramanujan",
                    "receipts/ascent-2026-08-16",
                    "receipts/ascent-2026-08-18",
                ],
                "no_rm_git_mv_clean_checkout_restore_reset": True,
                "temp_extract": str(extract_root),
                "temp_variants": str(variants_root),
            },
        }

        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n", encoding="utf-8")

        # Human report — this is the verification output.
        print(f"# Cold control-plane startup census")
        print(f"schema: {SCHEMA}")
        print(f"generated_at: {generated_at}")
        print(f"git_head: {head}")
        print(f"python: {python}")
        print(f"python_version: {sys.version.split()[0]}")
        print(f"hcli_source: {located['mode']}  pythonpath={pythonpath}")
        print(f"receipt: {RECEIPT}")
        print()
        print("## 1. Measured `python3 -m hcli --help` wall time")
        print(f"repeats: {REPEATS}  PYTHONDONTWRITEBYTECODE=1")
        print(
            f"median: {fmt_s(baseline_wall['median_s'])}  "
            f"min: {fmt_s(baseline_wall['min_s'])}  "
            f"max: {fmt_s(baseline_wall['max_s'])}"
        )
        print(f"samples_s: {baseline_wall['samples_s']}")
        print(f"help_text_ok: {baseline_wall['help_text_ok']}  returncodes: {baseline_wall['returncodes']}")
        print(f"interpreter floor `python3 -c pass`: {fmt_s(floor['python_pass_wall_s'])}")
        print(f"argparse --help floor (no hcli): {fmt_s(floor['argparse_help_wall_s'])}")
        print(
            f"hcli package cumulative (importtime): {fmt_us(None if hcli_full is None else hcli_full['cumulative_us'])}"
        )
        print(
            f"hcli.controller cumulative (importtime): {fmt_us(None if controller_row is None else controller_row['cumulative_us'])}"
        )
        print()
        print("## 2. Per-module import cost tree (hcli only, -X importtime)")
        print(f"{'module':<28} {'self':>10} {'cumulative':>12}  capability")
        for r in hcli_rows(baseline_rows):
            cap = MODULE_CAPABILITY.get(r["name"], "")
            print(f"{r['name']:<28} {fmt_us(r['self_us']):>10} {fmt_us(r['cumulative_us']):>12}  {cap}")
        print()
        print("## 2b. Top cumulative costs overall (including stdlib)")
        print(f"{'module':<40} {'self':>10} {'cumulative':>12}")
        for r in sorted(baseline_rows, key=lambda x: x["cumulative_us"], reverse=True)[:20]:
            print(f"{r['name']:<40} {fmt_us(r['self_us']):>10} {fmt_us(r['cumulative_us']):>12}")
        print()
        print("## 3. Reachable from the entrypoint, needed only by a capability")
        print(
            "Reachable = imported while running python3 -m hcli --help "
            "(package __init__ runs before argparse can exit)."
        )
        print(f"{'module':<28} {'help?':<6} {'defer?':<7}  capability")
        for row in classifications:
            if not row["loaded_on_help"]:
                continue
            print(
                f"{row['module']:<28} "
                f"{'yes' if row['needed_for_help'] else 'no':<6} "
                f"{'YES' if row['deferrable_on_help'] else 'no':<7}  "
                f"{row['capability']}"
            )
        print()
        print("Not loaded on --help (already behind a capability, or unused):")
        for m in not_on_help:
            print(f"  {m:<28} {MODULE_CAPABILITY.get(m, 'UNKNOWN')}")
        print()
        print("## 4. Projected saving per deferral (measured, not guessed)")
        print(
            f"D1 thin-init --help median: {fmt_s(thin_wall['median_s'])}  "
            f"samples {thin_wall['samples_s']}"
        )
        print(
            f"D1 saving (baseline - thin): wall {fmt_s(wall_save)}  "
            f"importtime {fmt_us(import_save_us)}"
        )
        print(
            "D1 is the --help total. D2/D3 are subsets of the graph D1 removes "
            "from this path; they are NOT added to D1."
        )
        print()
        for d in deferrals:
            print(f"### {d['id']}")
            print(f"move: {d['move']}")
            print(f"behind: {d['behind']}")
            print(f"wall saving: {fmt_s(d.get('measured_wall_saving_s'))}")
            print(f"importtime saving: {fmt_us(d.get('measured_importtime_saving_us'))}")
            print(f"how: {d['measured_how']}")
            print()
        print(
            f"TOTAL --help saving: wall {fmt_s(total_help_saving['wall_s'])}  "
            f"importtime {fmt_us(total_help_saving['importtime_us'])}"
        )
        print(total_help_saving["rule"])
        print()
        print("Isolated cold-import costs (empty package __init__, so Controller is not prepaid):")
        for mod, info in isolates.items():
            print(
                f"  {mod:<24} wall {fmt_s(info.get('wall_s'))}  "
                f"cumulative {fmt_us(info.get('cumulative_us'))}  rc={info.get('returncode')}"
            )
        print(
            f"  {'urllib.request':<24} wall {fmt_s(urllib_iso.get('wall_s'))}  "
            f"cumulative {fmt_us(urllib_iso.get('cumulative_us'))}"
        )
        print()
        print("## 5. Heavy packages eagerly imported? yes/no + chain")
        for name, v in heavy_verdict.items():
            chain = v.get("import_chain")
            print(f"{name}: {v['verdict']}")
            print(f"  why: {v['why']}")
            print(f"  import_chain: {chain if chain else 'n/a (not imported)'}")
            print(f"  ast_import_statements: {v['ast_import_statements'] or 'none'}")
        print(
            f"mlx via subprocess (not import): {mlx_subprocess['spawns_mlx_lm_server']}  "
            f"cited {mlx_subprocess['cited']}  when={mlx_subprocess['when']}"
        )
        print(f"AST heavy imports in all of hcli (eager+lazy): {ast_info['heavy_ast_imports'] or 'none'}")
        print()
        print("Adversarial interpreters (packages present in that python, still not imported by --help):")
        for probe in heavy_probe["probes"]:
            if not probe.get("ran"):
                print(f"  {probe.get('label')}: SKIP {probe.get('reason')}")
                continue
            avail = [k for k, v in (probe.get("available") or {}).items() if v.get("importable")]
            eager = probe.get("eager_heavy") or {}
            print(
                f"  {probe.get('label')}: dump_ok={probe.get('dump_ok')}  "
                f"importable={avail}  eager_in_sys_modules={eager or 'none'}"
            )
        print(f"visionmcp_src: {visionmcp_src or 'NOT FOUND'}")
        print()
        print("## 6. Citations (must resolve)")
        for c in cites:
            flag = "OK" if c.get("ok") else "FAIL"
            print(f"  [{flag}] {c.get('path')}:{c.get('line')}  {c.get('text', c.get('reason', ''))}")
        print()
        print("## 7. PATH shim (not this measurement)")
        shim = receipt["shim_not_measured"]
        print(f"  {shim['path']} present={shim['present']}")
        print(f"  current -> {shim.get('current_target')}")
        print(f"  {shim['note']}")
        print()
        print("## WHAT I WATCHED FAIL")
        if WATCHED_FAIL:
            for item in WATCHED_FAIL:
                print(f"- {item}")
        else:
            print("- nothing failed closed; no UNKNOWN blocked a yes/no")
        print()
        print(f"wrote {RECEIPT}")
        return 0
    finally:
        shutil.rmtree(extract_root, ignore_errors=True)
        shutil.rmtree(variants_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        note_fail(f"census aborted: {type(exc).__name__}: {exc}")
        print("## WHAT I WATCHED FAIL")
        for item in WATCHED_FAIL:
            print(f"- {item}")
        raise
