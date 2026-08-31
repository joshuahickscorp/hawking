"""TRIAL BUILD FREEZER — seal the autonomy driver's substrate so a mid-run edit cannot be claimed as the original interval.

An autonomy trial that hot-patches its own substrate mid-run measured a machine
that no longer exists. That already happened here: a 1h rerun was killed and
recorded INVALIDATED_BY_SUBSTRATE_MUTATION because specimen_verify.py was
edited while the loop was invoking it as a subprocess. Improving the test and
then claiming the original interval is the failure this module exists to make
impossible.

freeze(trial_id) hashes the files the driver can actually reach: the AST import
graph from autonomy_run.py, plus every literal tools/future/<name>.py string in
that driver's subprocess argv lists. A glob of tools/future/*.py would hash
files the driver never touches and falsely invalidate a trial; hashing only
autonomy_run.py would miss the subprocess it shells out to.

verify_unchanged(manifest) recomputes those digests and returns CLEAN or
INVALIDATED_BY_SUBSTRATE_MUTATION naming every path whose digest moved. A
deleted graph file is reported, not skipped. A file the graph never named
does not invalidate. An empty or malformed freeze is refused, never CLEAN.

This is STATIC_ONLY. It does not measure hardware, take a GPU lease, or
re-run the trial. It cannot establish that a CLEAN verdict means the trial
was scientific — only that the sealed substrate did not move.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import ast
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from tools.future._common import (
    HARDWARE_FIELDS,
    REPO,
    git,
    sha256_file,
    write_receipt,
    _assert_no_hardware_claims,
)
from tools.future import autonomy_trial as at
from tools.future import frontiers as fr
from tools.future import resident_identity as ri

RECEIPT = "HCLI_AUTONOMY_BUILD.json"
SCHEMA = "hawking.future.trial_freeze.v1"
DRIVER_REL = "tools/future/autonomy_run.py"
FUTURE_DIR = REPO / "tools" / "future"
VERDICT_CLEAN = "CLEAN"
VERDICT_INVALIDATED = "INVALIDATED_BY_SUBSTRATE_MUTATION"
UNAVAILABLE = "UNAVAILABLE"
GIT_DIRTY_TIMEOUT_S = 45

# Exact argv-literal match. A longer sentence that merely mentions a path is
# recovered_implementation prose, not a subprocess target.
_PY_REL = re.compile(r"^tools/future/[A-Za-z0-9_]+\.py$")
_RECEIPT_REL = re.compile(r"^receipts/future/[A-Za-z0-9_.-]+\.json$")

# Receipts autonomy_run.run() actually opens before the loop starts. FRONTIER_REL
# is an attribute in the driver (at.FRONTIER_REL), so it is taken from the trial
# module rather than restated.
STARTUP_RECEIPTS: tuple[str, ...] = (
    at.FRONTIER_REL,
    "receipts/future/ORCHESTRATION_BINDINGS.json",
    "receipts/future/RESIDENT_IDENTITY.json",
)


class FreezeRefused(ValueError):
    """A freeze that would look like success without a sealed substrate is refused."""


# ---------------------------------------------------------------------------
# Import graph + argv literals
# ---------------------------------------------------------------------------


def _modname(rel: str) -> str:
    p = Path(rel)
    if p.name == "__init__.py":
        return ".".join(p.parent.parts)
    return ".".join(p.with_suffix("").parts)


def _rel_from_mod(mod: str) -> str | None:
    if mod == "tools.future":
        return "tools/future/__init__.py"
    if not mod.startswith("tools.future."):
        return None
    rest = mod[len("tools.future.") :]
    if not rest or not re.fullmatch(r"[A-Za-z0-9_.]+", rest):
        return None
    return "tools/future/" + rest.replace(".", "/") + ".py"


def _future_import_rels(tree: ast.AST, current_mod: str) -> list[str]:
    """tools.future modules named by Import / ImportFrom, including relatives.

    Star-imports are not followed: guessing their names would invent reachability.
    Dynamic importlib.import_module(f"tools.future.{name}") is not a static edge.
    """
    out: list[str] = []
    parts = current_mod.split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                rel = _rel_from_mod(alias.name)
                if rel:
                    out.append(rel)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            if node.level > len(parts):
                continue
            parent = parts[: -node.level]
            pkg = ".".join(parent)
            if node.module:
                absmod = ".".join(parent + node.module.split("."))
                rel = _rel_from_mod(absmod)
                if rel:
                    out.append(rel)
            else:
                if pkg == "tools.future" or pkg.startswith("tools.future."):
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        absmod = f"{pkg}.{alias.name}" if pkg else alias.name
                        rel = _rel_from_mod(absmod)
                        if rel:
                            out.append(rel)
            continue
        if node.module == "tools.future":
            out.append("tools/future/__init__.py")
            for alias in node.names:
                if alias.name == "*":
                    continue
                rel = _rel_from_mod(f"tools.future.{alias.name}")
                if rel:
                    out.append(rel)
        elif node.module and node.module.startswith("tools.future."):
            rel = _rel_from_mod(node.module)
            if rel:
                out.append(rel)
    seen: set[str] = set()
    ordered: list[str] = []
    for rel in out:
        if rel not in seen:
            seen.add(rel)
            ordered.append(rel)
    return ordered


def _list_string_slots(node: ast.AST) -> list[str | None]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    slots: list[str | None] = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            slots.append(elt.value)
        else:
            slots.append(None)
    return slots


def _driver_argv_py(tree: ast.AST) -> tuple[list[str], list[str]]:
    """Literal tools/future/<name>.py strings in list/tuple literals of the driver.

    `mentioned` is every such string (the contract: argv lists). `executed` is
    the subset whose previous slot is not a flag, i.e. the script python3 runs
    rather than a --ignore path. Executed scripts are import-walked; mentioned
    -only files are hashed and not followed.
    """
    mentioned: list[str] = []
    executed: list[str] = []
    seen_m: set[str] = set()
    seen_e: set[str] = set()
    for node in ast.walk(tree):
        slots = _list_string_slots(node)
        if not slots:
            continue
        for i, value in enumerate(slots):
            if not value or not _PY_REL.match(value):
                continue
            if value not in seen_m:
                seen_m.add(value)
                mentioned.append(value)
            prev = slots[i - 1] if i else None
            flag_value = isinstance(prev, str) and prev.startswith("-")
            if not flag_value and value not in seen_e:
                seen_e.add(value)
                executed.append(value)
    return mentioned, executed


def _hash_rel(rel: str) -> dict[str, Any]:
    path = REPO / rel
    if not path.is_file():
        return {
            "path": rel,
            "sha256": None,
            "state": "ABSENT",
            "why": "named by import or argv but not on disk",
        }
    try:
        digest = sha256_file(path)
    except OSError as exc:
        return {
            "path": rel,
            "sha256": None,
            "state": "UNREADABLE",
            "why": f"{type(exc).__name__}: {exc}",
        }
    return {"path": rel, "sha256": digest, "state": "HASHED"}


def _parse_rel(rel: str) -> tuple[ast.AST | None, str | None]:
    path = REPO / rel
    if not path.is_file():
        return None, "ABSENT"
    try:
        return ast.parse(path.read_text()), None
    except SyntaxError as exc:
        return None, f"SyntaxError: {exc}"
    except (OSError, UnicodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def driver_reachability() -> dict[str, Any]:
    """Every tools/future/*.py the autonomy driver can reach, and how.

    Not a glob. Walk starts at autonomy_run.py. Import edges are recursive.
    Subprocess argv .py literals are taken from the driver only; executed
    scripts are themselves import-walked, mentioned-only scripts are hashed.
    """
    driver = REPO / DRIVER_REL
    if not driver.is_file():
        raise FreezeRefused(f"driver missing on disk: {DRIVER_REL}")

    files: dict[str, dict[str, Any]] = {}

    def add(rel: str, route: str) -> dict[str, Any]:
        rec = files.get(rel)
        if rec is None:
            rec = _hash_rel(rel)
            rec["routes"] = []
            files[rel] = rec
        if route not in rec["routes"]:
            rec["routes"].append(route)
            rec["routes"].sort()
        return rec

    add(DRIVER_REL, "driver")
    queue = [DRIVER_REL]
    queued = {DRIVER_REL}
    argv_mentioned: list[str] = []
    argv_executed: list[str] = []

    while queue:
        rel = queue.pop()
        tree, err = _parse_rel(rel)
        if tree is None:
            if err and err != "ABSENT" and files[rel].get("state") == "HASHED":
                files[rel]["state"] = "UNREADABLE"
                files[rel]["why"] = err
                files[rel]["sha256"] = None
            continue
        for imported in _future_import_rels(tree, _modname(rel)):
            add(imported, "import")
            if imported not in queued:
                queued.add(imported)
                queue.append(imported)
        if rel != DRIVER_REL:
            continue
        mentioned, executed = _driver_argv_py(tree)
        argv_mentioned = mentioned
        argv_executed = executed
        for path in mentioned:
            add(path, "subprocess")
        for path in executed:
            if path not in queued:
                queued.add(path)
                queue.append(path)

    glob_py = sorted(p.name for p in FUTURE_DIR.glob("*.py"))
    hashed = [r for r in files.values() if r.get("state") == "HASHED"]
    by_route: dict[str, list[str]] = {"driver": [], "import": [], "subprocess": []}
    for rec in files.values():
        for route in rec.get("routes") or []:
            by_route.setdefault(route, []).append(rec["path"])
    for route in by_route:
        by_route[route] = sorted(set(by_route[route]))

    return {
        "driver": DRIVER_REL,
        "files": [files[k] for k in sorted(files)],
        "by_route": by_route,
        "argv_mentioned": argv_mentioned,
        "argv_executed": argv_executed,
        "n_graph": len(files),
        "n_hashed": len(hashed),
        "n_future_py_on_disk": len(glob_py),
        "glob_would_hash": glob_py,
        "rule": (
            "AST import graph from autonomy_run.py plus literal "
            "tools/future/<name>.py strings in that driver's subprocess argv "
            "lists; not tools/future/*.py"
        ),
    }


# ---------------------------------------------------------------------------
# Startup receipts, identity, frontier, git
# ---------------------------------------------------------------------------


def _startup_receipts() -> list[dict[str, Any]]:
    """Receipts the driver reads before the loop. Absence is recorded, never hashed as success."""
    rows: list[dict[str, Any]] = []
    for rel in STARTUP_RECEIPTS:
        rec = _hash_rel(rel)
        rec["role"] = "driver_startup_read"
        rows.append(rec)
    return rows


def _slot_value(field: Any) -> Any:
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


def _identity_slot() -> dict[str, Any]:
    """Resident identity from resident_identity.py. No resident process is invented."""
    source = UNAVAILABLE
    ident: dict[str, Any] | None = None
    why: list[str] = []
    try:
        ident = ri.load()
        source = "tools.future.resident_identity.load"
    except (ri.IdentityRejectedError, OSError, FileNotFoundError) as exc:
        why.append(f"load: {type(exc).__name__}: {exc}")
        try:
            ident = ri.collect()
            source = "tools.future.resident_identity.collect"
        except Exception as exc2:  # collect is the last honest path; do not invent identity
            why.append(f"collect: {type(exc2).__name__}: {exc2}")
            ident = None

    process = {
        "value": UNAVAILABLE,
        "why": (
            "this sidecar does not start a resident model process; "
            "autonomy_run records resident_model_cognition UNAVAILABLE as a "
            "measurement of that fact, not as a quoted blocker"
        ),
    }
    if ident is None:
        return {
            "resident_identity": {
                "value": UNAVAILABLE,
                "source": UNAVAILABLE,
                "why": why,
            },
            "model_identity": {
                "value": UNAVAILABLE,
                "why": ["no resident identity document could be loaded or collected"] + why,
            },
            "resident_model_process": process,
        }

    family = _slot_value(ident.get("model_family"))
    nx = _slot_value(ident.get("nx_id"))
    model_id: Any = None
    if isinstance(nx, dict):
        model_id = nx.get("model_id")
    elif isinstance(nx, str):
        model_id = nx
    if model_id in (None, "", ri.UNKNOWN):
        model_slot = {
            "value": UNAVAILABLE,
            "why": [
                "no resident model id on disk; UNKNOWN/absent is not a model identity"
            ],
            "nx_id": nx if nx not in (None, "") else UNAVAILABLE,
            "model_family": family if family not in (None, "", ri.UNKNOWN) else UNAVAILABLE,
        }
    else:
        model_slot = {
            "value": model_id,
            "model_family": family if family not in (None, "", ri.UNKNOWN) else UNAVAILABLE,
            "source": source,
        }
    return {
        "resident_identity": {
            "source": source,
            "residency_status": ident.get("residency_status", UNAVAILABLE),
            "model_family": family if family not in (None, "") else UNAVAILABLE,
            "nx_id": nx if nx not in (None, "") else UNAVAILABLE,
            "gpu_authority": bool(ident.get("gpu_authority")) if "gpu_authority" in ident else False,
        },
        "model_identity": model_slot,
        "resident_model_process": process,
    }


def _frontier_slot() -> dict[str, Any]:
    """Item ids + states from frontiers.load_book(). Lanes are imported, never restated."""
    try:
        book = fr.load_book()
    except Exception as exc:
        return {
            "digest": UNAVAILABLE,
            "items": [],
            "why": f"frontiers.load_book failed: {type(exc).__name__}: {exc}",
            "available_lanes": list(fr.THIS_HOST_LANES),
            "blocked_lanes": list(fr.BLOCKED_ON_THIS_HOST),
        }
    sleeping_ids: set[str] = set()
    try:
        for unit in book.sleeping_units():
            uid = unit.get("id")
            if uid:
                sleeping_ids.add(str(uid))
    except Exception as exc:
        return {
            "digest": UNAVAILABLE,
            "items": [],
            "why": f"sleeping_units failed: {type(exc).__name__}: {exc}",
            "available_lanes": list(fr.THIS_HOST_LANES),
            "blocked_lanes": list(fr.BLOCKED_ON_THIS_HOST),
        }
    rows: list[dict[str, str]] = []
    for item in book.items:
        iid = str(item.get("id") or "")
        kind = str(item.get("kind") or "")
        if not iid:
            continue
        if iid in sleeping_ids or kind == "BLOCKED":
            state = "SLEEPING"
        else:
            state = kind
        rows.append({"id": iid, "kind": kind, "state": state})
    rows.sort(key=lambda r: r["id"])
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "digest": hashlib.sha256(blob).hexdigest(),
        "items": rows,
        "n_items": len(rows),
        "available_lanes": list(fr.THIS_HOST_LANES),
        "blocked_lanes": list(fr.BLOCKED_ON_THIS_HOST),
        "why": None,
    }


def _git_slot() -> dict[str, Any]:
    """HEAD, branch, dirty. A timeout is UNAVAILABLE, never a fake clean tree."""
    head = git("rev-parse", "HEAD") or UNAVAILABLE
    branch = git("rev-parse", "--abbrev-ref", "HEAD") or UNAVAILABLE
    dirty: Any = UNAVAILABLE
    why = None
    sample: list[str] = []
    try:
        proc = subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--",
                "tools/future",
                "receipts/future",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
            timeout=GIT_DIRTY_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        why = (
            f"git status timed out after {GIT_DIRTY_TIMEOUT_S}s; "
            "dirty is UNAVAILABLE rather than claimed clean"
        )
        return {
            "head": head,
            "branch": branch,
            "dirty": UNAVAILABLE,
            "dirty_scope": "tools/future + receipts/future",
            "why": why,
            "dirty_sample": [],
        }
    except OSError as exc:
        why = f"git status unrunnable: {type(exc).__name__}: {exc}"
        return {
            "head": head,
            "branch": branch,
            "dirty": UNAVAILABLE,
            "dirty_scope": "tools/future + receipts/future",
            "why": why,
            "dirty_sample": [],
        }
    if proc.returncode not in (0, 1):
        why = f"git status exit {proc.returncode}: {(proc.stderr or '')[:200]}"
        dirty = UNAVAILABLE
    else:
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        dirty = bool(lines)
        sample = lines[:12]
    return {
        "head": head,
        "branch": branch,
        "dirty": dirty,
        "dirty_scope": "tools/future + receipts/future",
        "why": why,
        "dirty_sample": sample,
    }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _snapshot() -> dict[str, Any]:
    reach = driver_reachability()
    identity = _identity_slot()
    frontier = _frontier_slot()
    git_state = _git_slot()
    snap = {
        "schema": SCHEMA,
        "driver": DRIVER_REL,
        "code_files": reach["files"],
        "routes": reach["by_route"],
        "argv_mentioned": reach["argv_mentioned"],
        "argv_executed": reach["argv_executed"],
        "n_code_files": reach["n_graph"],
        "n_hashed": reach["n_hashed"],
        "n_future_py_on_disk": reach["n_future_py_on_disk"],
        "graph_rule": reach["rule"],
        "startup_receipts": _startup_receipts(),
        "resident_identity": identity["resident_identity"],
        "model_identity": identity["model_identity"],
        "resident_model_process": identity["resident_model_process"],
        "frontier_digest": frontier["digest"],
        "frontier_items": frontier["items"],
        "frontier_why": frontier.get("why"),
        "available_lanes": frontier["available_lanes"],
        "blocked_lanes": frontier["blocked_lanes"],
        "git": git_state,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "verdicts": [VERDICT_CLEAN, VERDICT_INVALIDATED],
    }
    _assert_no_hardware_claims(snap)
    return snap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def freeze(trial_id: str) -> dict[str, Any]:
    """Seal the current driver substrate for one named trial. Does not write."""
    if trial_id not in at.TRIAL_IDS:
        raise FreezeRefused(
            f"unknown trial_id {trial_id!r}; known {at.TRIAL_IDS}"
        )
    snap = _snapshot()
    snap["trial_id"] = trial_id
    snap["freeze_time"] = _now()
    return snap


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO / path


def verify_unchanged(manifest: Any) -> dict[str, Any]:
    """Recompute digests of the frozen code files. Deleted files are named.

    Verdict is about CODE. Startup receipts are provenance of freeze-time
    reads; the loop itself rewrites some of them, so receipt drift is reported
    as receipt_drift and does not, by itself, produce INVALIDATED.
    """
    if not isinstance(manifest, Mapping):
        return {
            "verdict": VERDICT_INVALIDATED,
            "moved_paths": [],
            "moved": [],
            "receipt_drift": [],
            "why": "manifest is not a mapping; refusing CLEAN",
        }
    files = manifest.get("code_files")
    if not isinstance(files, list) or not files:
        return {
            "verdict": VERDICT_INVALIDATED,
            "moved_paths": [],
            "moved": [],
            "receipt_drift": [],
            "why": "manifest names no code_files; an empty freeze is not CLEAN",
        }

    moved: list[dict[str, Any]] = []
    for raw in files:
        if not isinstance(raw, Mapping):
            moved.append(
                {
                    "path": "<non-object code_files entry>",
                    "frozen_sha256": None,
                    "current_sha256": None,
                    "state": "MALFORMED",
                }
            )
            continue
        rel = str(raw.get("path") or "")
        if not rel:
            moved.append(
                {
                    "path": "<missing path>",
                    "frozen_sha256": raw.get("sha256"),
                    "current_sha256": None,
                    "state": "MALFORMED",
                }
            )
            continue
        frozen = raw.get("sha256")
        path = _resolve(rel)
        if not path.is_file():
            current = None
            state = "MISSING"
        else:
            try:
                current = sha256_file(path)
                state = "HASHED"
            except OSError as exc:
                current = None
                state = f"UNREADABLE:{type(exc).__name__}"
        if frozen != current:
            moved.append(
                {
                    "path": rel,
                    "frozen_sha256": frozen,
                    "current_sha256": current,
                    "state": "MISSING" if current is None and frozen is not None else (
                        "APPEARED" if frozen is None and current is not None else "CHANGED"
                    ),
                }
            )

    receipt_drift: list[dict[str, Any]] = []
    for raw in manifest.get("startup_receipts") or []:
        if not isinstance(raw, Mapping):
            continue
        rel = str(raw.get("path") or "")
        if not rel:
            continue
        frozen = raw.get("sha256")
        path = _resolve(rel)
        current = sha256_file(path) if path.is_file() else None
        if frozen != current:
            receipt_drift.append(
                {
                    "path": rel,
                    "frozen_sha256": frozen,
                    "current_sha256": current,
                    "state": "MISSING" if current is None else "CHANGED",
                }
            )

    if moved:
        verdict = VERDICT_INVALIDATED
        why = f"{len(moved)} code path(s) moved"
    else:
        verdict = VERDICT_CLEAN
        why = "every frozen code digest matches; no graph file is missing"
    return {
        "verdict": verdict,
        "moved_paths": [m["path"] for m in moved],
        "moved": moved,
        "n_moved": len(moved),
        "receipt_drift": receipt_drift,
        "why": why,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


def build() -> Path:
    snap = _snapshot()
    ts = _now()
    frozen_builds = [
        {**snap, "trial_id": tid, "freeze_time": ts} for tid in at.TRIAL_IDS
    ]
    both = sorted(
        set(snap["routes"].get("import") or [])
        & set(snap["routes"].get("subprocess") or [])
    )
    sub_only = sorted(
        set(snap["routes"].get("subprocess") or [])
        - set(snap["routes"].get("import") or [])
        - set(snap["routes"].get("driver") or [])
    )
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Seal the autonomy driver's reachable substrate so a trial cannot "
            "claim an interval whose code moved underneath it."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "driver": DRIVER_REL,
        "trial_ids": list(at.TRIAL_IDS),
        "frozen_builds": frozen_builds,
        "graph_rule": snap["graph_rule"],
        "files_by_route": snap["routes"],
        "files_found_by_both_routes": both,
        "files_found_by_subprocess_only": sub_only,
        "n_code_files": snap["n_code_files"],
        "n_hashed": snap["n_hashed"],
        "n_future_py_on_disk": snap["n_future_py_on_disk"],
        "glob_refused": (
            f"tools/future/*.py is {snap['n_future_py_on_disk']} files; "
            f"the driver graph is {snap['n_code_files']}. A glob would false-"
            "invalidate on unrelated edits."
        ),
        "startup_receipts": snap["startup_receipts"],
        "resident_identity": snap["resident_identity"],
        "model_identity": snap["model_identity"],
        "resident_model_process": snap["resident_model_process"],
        "frontier_digest": snap["frontier_digest"],
        "available_lanes": snap["available_lanes"],
        "blocked_lanes": snap["blocked_lanes"],
        "git": snap["git"],
        "freeze_time": ts,
        "recovered_implementation": [
            "tools/future/autonomy_run.py is the walk root; INVALIDATED_RUNS already names the 1h kill this freezer makes recomputable",
            "tools/future/autonomy_trial.py TRIAL_IDS and FRONTIER_REL (startup receipt, not restated)",
            "tools/future/resident_identity.py load()/collect() for identity; UNKNOWN/UNAVAILABLE where no model exists",
            "tools/future/frontiers.py load_book() item ids+states; THIS_HOST_LANES and BLOCKED_ON_THIS_HOST, never a second lane list",
            "tools/future/_common.py sha256_file / write_receipt / HARDWARE_FIELDS / git()",
            "tools/future/evidence_snapshot.py pin-and-hash pattern was read, not forked; freeze hashes driver-startup receipts, not the snapshot corpus",
        ],
        "gaps_closed": [
            "import-graph freeze of the autonomy driver (AST walk + subprocess argv literals)",
            "verify_unchanged returns CLEAN or INVALIDATED_BY_SUBSTRATE_MUTATION with the exact moved paths",
            "a graph file deleted between freeze and verify is reported, not skipped",
            "a file the graph never named does not invalidate a trial",
            "unknown trial_id and empty manifests refuse rather than look CLEAN",
        ],
        "negative_findings": [
            "resident_model_process is UNAVAILABLE: this sidecar does not start a resident model process",
            "startup receipt bytes are sealed at freeze; the loop itself rewrites some of them, so receipt drift is reported separately and is not by itself INVALIDATED_BY_SUBSTRATE_MUTATION",
            "git dirty is scoped to tools/future + receipts/future; a timeout is UNAVAILABLE, never claimed clean",
            "star-imports and importlib.import_module of formatted names are not static edges and are not followed",
            "SAFE_CAPABILITIES is a declared invoke list, not executed reachability; naming a module there does not put it in this graph",
            "this module is not in orchestration.BINDINGS (that table is outside this lane's WRITE list)",
        ],
        "resident_callable": {
            "entry_point": "tools.future.trial_freeze.freeze(trial_id) / verify_unchanged(manifest)",
            "workunit": "one CPU_ANALYSIS unit; seal the autonomy driver substrate and refuse a moved graph",
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.CHILD_RESIDENT.launch",
            "fails_closed": (
                "unknown trial_id raises FreezeRefused; empty/malformed manifest "
                "is INVALIDATED_BY_SUBSTRATE_MUTATION not CLEAN; missing graph "
                "files are named; git timeout is UNAVAILABLE; no hardware field "
                "may be numeric"
            ),
        },
    }
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        if key in doc and isinstance(doc[key], (int, float)):
            raise FreezeRefused(f"hardware field {key} leaked into the freeze receipt")
    return write_receipt(RECEIPT, doc, "tools/future/trial_freeze.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--freeze", metavar="TRIAL")
    ap.add_argument("--verify", metavar="MANIFEST_JSON")
    a = ap.parse_args()
    if a.freeze:
        print(json.dumps(freeze(a.freeze), indent=1, sort_keys=True))
        return 0
    if a.verify:
        path = Path(a.verify)
        if not path.is_file():
            print(
                json.dumps(
                    {
                        "verdict": VERDICT_INVALIDATED,
                        "why": f"manifest not found: {path}",
                    },
                    indent=1,
                    sort_keys=True,
                )
            )
            return 2
        try:
            man = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            print(
                json.dumps(
                    {
                        "verdict": VERDICT_INVALIDATED,
                        "why": f"manifest unreadable: {type(exc).__name__}: {exc}",
                    },
                    indent=1,
                    sort_keys=True,
                )
            )
            return 2
        out = verify_unchanged(man)
        print(json.dumps(out, indent=1, sort_keys=True))
        return 0 if out["verdict"] == VERDICT_CLEAN else 2
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
