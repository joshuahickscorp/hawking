#!/usr/bin/env python3
"""Final DeepSeek V4.1 Flash qualification on a disposable mini-Odyssey.

The harness copies bounded, text-only current Hawking source/evidence into a
temporary Git repository, gives one exact OpenRouter model a real WorkUnit,
and lets the same provider instance perform a second self-critique pass.  The
worker may only use the current safe read/write/test/status/diff registry
surface.  The temporary repository and model prose are discarded; stdout is
limited to redacted metrics and findings suitable for a benchmark receipt.

This file never starts a model runtime, downloads a model, touches ModelLake,
or writes the canonical Odyssey state.  Its only persistent artifact should be
the separately reviewed benchmark receipt produced after a successful run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hawking.remote_cognition import (  # noqa: E402
    OpenRouterProvider,
    RemoteCostPolicy,
    _keychain_resolver_from_environment,
)
from hawking.tool_registry import (  # noqa: E402
    READ_ONLY,
    REVERSIBLE_REPO,
    REVERSIBLE_RUNTIME,
    default_tool_registry,
)


MODEL_ID = "deepseek/deepseek-v4.1-flash"
PROVIDER = "openrouter"
HARD_BUDGET_USD = 0.50
MAX_TOOL_CALLS = 32
MAX_CRITIQUE_TOOL_CALLS = 6
MAX_TOKENS = 560
MAX_CRITIQUE_TOKENS = 460
LANE_TIMEOUT_S = 90.0
SAFE_PERMISSIONS = frozenset({READ_ONLY, REVERSIBLE_REPO, REVERSIBLE_RUNTIME})

IMPLEMENTATION_TOOLS: Tuple[str, ...] = (
    "tools.catalog",
    "fs.list",
    "fs.search",
    "fs.read",
    "filesystem.write",
    "tests.list",
    "tests.run",
    "git.status",
    "git.diff",
)
CRITIQUE_TOOLS: Tuple[str, ...] = (
    "fs.read",
    "git.status",
    "git.diff",
)

TEXT_SUFFIXES = frozenset({
    ".md", ".txt", ".py", ".rs", ".toml", ".json", ".yaml", ".yml",
    ".sh", ".jinja", ".html", ".js", ".ts", ".lock",
})
SKIP_DIRS = frozenset({
    ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", ".cache", ".worktrees", "target", "build", "dist",
    "downloads", "session_artifacts", ".ruff_cache",
})

COPY_TREES: Tuple[str, ...] = (
    "hawking",
    "crates/hawking-core/src",
    "crates/hide-backend/src",
    "tools/odyssey",
    "tools/roadmap",
    "tools/verify",
)
COPY_FILES: Tuple[str, ...] = (
    "AGENTS.md",
    "H-MANIFESTO.md",
    "CLAUDE-CODEX-EXODUS.md",
    "Cargo.toml",
    "Cargo.lock",
    "pyproject.toml",
    "civilization/sovereign-goal.txt",
    "docs/CURRENT_ARCHITECTURE.md",
    "docs/HAWKING_DELEGATION.md",
    "docs/HAWKING_NEXT_GOAL_CONTINUITY_2026-09-10.md",
    "docs/ULTRAGOAL_HANDOFF_2026-09-09.md",
    "docs/ULTRAGOAL_HANDOFF_2026-09-08.md",
    "docs/odyssey-i.md",
    "tools/odyssey_ctl.py",
    "tools/odyssey_costmodel.py",
    "workspace/campaign/odyssey/ODYSSEY_STATE.json",
    "workspace/campaign/odyssey/ODYSSEY.md",
    "receipts/odyssey-i/MODELLAKE_SPECIMEN_DISPOSITIONS.json",
    "receipts/odyssey-i/MODELLAKE_SCHEDULING_OVERRIDE_20260910.json",
    "receipts/future/ODYSSEY_MISSION_CONTROLLER.json",
    "receipts/future/HAWKING_NATIVE_NR_ADMISSION_OWNER_20260912.json",
    "receipts/future/LFM2_RUNTIME_ADMISSION_BLOCKED_20260911.json",
    "receipts/headless/MODEL_LAKE_ROLLING_PIPELINE.json",
)

ARCHITECTURE_SIGNALS: Tuple[str, ...] = (
    "source",
    "uncollapsed",
    "collapsed",
    "nr",
    "candidate",
    "star",
    "nx",
    "blocked",
    "active_owner",
    "source_hash",
    "nr_revision",
    "nx_revision",
    "receipt",
    "ebpw",
    "throughput",
    "qualification",
    "next_experiment",
)
ADVERSARIAL_SIGNALS: Tuple[str, ...] = (
    "no_nr",
    "qualifying",
    "multiple",
    "owner",
    "stale",
    "blocked",
    "target",
    "ambiguous",
    "hash",
)
CRITIQUE_SIGNALS: Tuple[str, ...] = (
    "duplicate",
    "owner",
    "unsupported",
    "100",
    "scale",
    "safetensors",
    "nr",
    "nx",
    "merge",
    "discard",
    "not merge",
)


def _model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", model_id).strip("-").lower()


def _copy_one(relative: str, root: Path) -> bool:
    source = REPO / relative
    if not source.is_file() or source.is_symlink():
        return False
    try:
        if source.stat().st_size > 750_000:
            return False
    except OSError:
        return False
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _copy_tree(relative: str, root: Path) -> int:
    source_root = REPO / relative
    if not source_root.is_dir():
        return 0
    copied = 0
    for directory, dirnames, filenames in os.walk(source_root):
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in SKIP_DIRS and not (Path(directory) / name).is_symlink()
        )
        for filename in sorted(filenames):
            source = Path(directory) / filename
            if source.is_symlink() or source.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                if source.stat().st_size > 750_000:
                    continue
            except OSError:
                continue
            destination = root / source.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1
    return copied


def _git_capture(root: Path, args: Sequence[str]) -> Dict[str, Any]:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
        env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", "LANG": "C"},
    )
    return {
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "")[:8000],
        "stderr": (completed.stderr or "")[:2000],
    }


def _fixture() -> str:
    return json.dumps(
        {
            "schema": "hawking.disposable.modellake.fixture.v1",
            "purpose": "adversarial organization proof only; not a production registry",
            "artifacts": [
                {
                    "source_id": "model-alpha",
                    "source_hash": "sha-source-alpha",
                    "source_state": "SOURCE",
                    "nr": None,
                    "nx": [],
                    "active_owner": None,
                    "receipts": [],
                },
                {
                    "source_id": "model-beta",
                    "source_hash": "sha-source-beta",
                    "source_state": "COLLAPSED",
                    "nr": {"revision": "nr-beta-1", "available": True},
                    "nx": [],
                    "active_owner": "lane-a",
                    "receipts": [{"kind": "capability", "qualified": False}],
                },
                {
                    "source_id": "model-gamma",
                    "source_hash": "sha-source-gamma",
                    "source_state": "COLLAPSED",
                    "nr": {"revision": "nr-gamma-4", "available": True},
                    "nx": [
                        {"revision": "nx-gamma-m3-1", "target": "m3-ultra"},
                        {"revision": "nx-gamma-gpu-2", "target": "external-gpu"},
                    ],
                    "active_owner": None,
                    "receipts": [{"kind": "capability", "qualified": True}],
                },
                {
                    "source_id": "model-delta",
                    "source_hash": "sha-source-delta-current",
                    "source_state": "CANDIDATE",
                    "nr": {"revision": "nr-delta-2", "available": True},
                    "nx": [],
                    "active_owner": "another-active-lane",
                    "receipts": [{"kind": "ebpw", "qualified": True}],
                },
                {
                    "source_id": "model-epsilon",
                    "source_hash": "sha-source-epsilon-current",
                    "source_state": "STAR_CANDIDATE",
                    "nr": {"revision": "nr-epsilon-1", "available": True},
                    "nx": [],
                    "active_owner": None,
                    "source_hash_at_nr": "sha-source-epsilon-old",
                    "receipts": [{"kind": "throughput", "qualified": True}],
                },
                {
                    "source_id": "model-zeta",
                    "source_hash": "sha-source-zeta",
                    "source_state": "BLOCKED",
                    "nr": {"revision": "nr-zeta-1", "available": True},
                    "nx": [],
                    "active_owner": None,
                    "blocked_reason": "target hardware unavailable",
                    "receipts": [],
                },
                {
                    "source_id": "model-alpha-collapsed",
                    "source_name": "model-alpha",
                    "source_hash": "sha-source-alpha-collapsed",
                    "source_state": "COLLAPSED",
                    "nr": {"revision": "nr-alpha-2", "available": True},
                    "nx": [],
                    "active_owner": None,
                    "receipts": [{"kind": "integrity", "qualified": True}],
                },
            ],
        },
        indent=2,
    ) + "\n"


def _seed_disposable_repo(root: Path) -> Dict[str, Any]:
    copied = 0
    for relative in COPY_TREES:
        copied += _copy_tree(relative, root)
    copied_files = [relative for relative in COPY_FILES if _copy_one(relative, root)]

    (root / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\ndisposable/test_result.txt\n",
        encoding="utf-8",
    )
    (root / "pytest.ini").write_text(
        "[pytest]\naddopts = -q\ntestpaths = disposable\n",
        encoding="utf-8",
    )
    disposable = root / "disposable"
    disposable.mkdir(parents=True, exist_ok=True)
    (disposable / "modellake_fixture.json").write_text(_fixture(), encoding="utf-8")
    (disposable / "model_lake_selection.py").write_text(
        "# DeepSeek disposable implementation placeholder.\n",
        encoding="utf-8",
    )
    (disposable / "test_model_lake_selection.py").write_text(
        "# DeepSeek disposable test placeholder.\n",
        encoding="utf-8",
    )
    (disposable / "__init__.py").write_text("", encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True, capture_output=True, text=True)
    revision = _git_capture(root, ["rev-parse", "--verify", "HEAD"])
    return {
        "copied_source_files": copied,
        "copied_evidence_files": len(copied_files),
        "source_revision": revision.get("stdout", "").strip() or None,
        "implementation_path": "disposable/model_lake_selection.py",
        "test_path": "disposable/test_model_lake_selection.py",
    }


def _parse_json_text(value: Any) -> Dict[str, Any]:
    if not isinstance(value, str):
        return {}
    candidate = value.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _safe_code(exc: BaseException) -> str:
    return str(getattr(exc, "code", None) or type(exc).__name__)


def _trace_rows(result: Optional[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    raw = result.get("tool_trace") if isinstance(result, Mapping) else None
    return [row for row in raw if isinstance(row, Mapping)] if isinstance(raw, list) else []


def _trace_value(row: Mapping[str, Any]) -> Mapping[str, Any]:
    result = row.get("result")
    if not isinstance(result, Mapping):
        return {}
    value = result.get("value")
    return value if isinstance(value, Mapping) else {}


def _relative_path(root: Path, raw: Any) -> Optional[str]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return str(candidate.resolve(strict=False).relative_to(root.resolve()))
    except ValueError:
        return None


def _hash_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _usage(results: Iterable[Optional[Mapping[str, Any]]]) -> Dict[str, Any]:
    totals: Dict[str, float] = {}
    reasoning_tokens: Optional[float] = None
    for result in results:
        usage = result.get("usage") if isinstance(result, Mapping) else None
        if not isinstance(usage, Mapping):
            continue
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"):
            try:
                value = usage.get(key)
                if value is not None:
                    totals[key] = totals.get(key, 0.0) + float(value)
            except (TypeError, ValueError):
                continue
        for key in ("reasoning_tokens", "output_reasoning_tokens"):
            try:
                value = usage.get(key)
                if value is not None:
                    reasoning_tokens = (reasoning_tokens or 0.0) + float(value)
                    break
            except (TypeError, ValueError):
                continue
    normalized: Dict[str, Any] = {}
    for key, value in totals.items():
        normalized[key] = int(value) if value.is_integer() else round(value, 8)
    normalized["reasoning_tokens"] = (
        int(reasoning_tokens) if reasoning_tokens is not None and reasoning_tokens.is_integer()
        else round(reasoning_tokens, 8) if reasoning_tokens is not None else None
    )
    return normalized


def _provider_turns(results: Iterable[Optional[Mapping[str, Any]]]) -> int:
    turns = 0
    for result in results:
        if not isinstance(result, Mapping):
            continue
        receipts = result.get("remote_cognition")
        if isinstance(receipts, list):
            turns += len(receipts)
    return turns


def _provider_retries(results: Iterable[Optional[Mapping[str, Any]]]) -> int:
    retries = 0
    for result in results:
        if not isinstance(result, Mapping):
            continue
        receipts = result.get("remote_cognition")
        if isinstance(receipts, list):
            for receipt in receipts:
                if isinstance(receipt, Mapping):
                    try:
                        retries += int(receipt.get("retries") or 0)
                    except (TypeError, ValueError):
                        continue
    return retries


def _model_text(results: Iterable[Optional[Mapping[str, Any]]]) -> str:
    chunks: List[str] = []
    for result in results:
        if not isinstance(result, Mapping):
            continue
        chunks.append(str(result.get("content") or ""))
        for key in ("observations", "operations", "tests", "questions"):
            chunks.append(json.dumps(result.get(key), sort_keys=True, default=str))
    return "\n".join(chunks).lower()


def _successful_tools(rows: Iterable[Mapping[str, Any]]) -> List[str]:
    return sorted({
        str(row.get("tool")) for row in rows
        if row.get("dispatched") is True and row.get("ok") is True and row.get("tool")
    })


def _run_authoritative_test(root: Path) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "disposable/test_model_lake_selection.py"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=60.0,
            env={
                "PATH": os.environ.get("PATH", ""),
                "LC_ALL": "C",
                "LANG": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            },
        )
        stdout = (completed.stdout or "")[-5000:]
        stderr = (completed.stderr or "")[-3000:]
        combined = f"{stdout}\n{stderr}"
        passed = re.search(r"(\d+) passed", combined)
        failed = re.search(r"(\d+) failed", combined)
        return {
            "verified": completed.returncode == 0,
            "returncode": completed.returncode,
            "passed": int(passed.group(1)) if passed else None,
            "failed": int(failed.group(1)) if failed else None,
            "wall_s": round(time.monotonic() - started, 3),
            "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
        }
    except subprocess.TimeoutExpired:
        return {
            "verified": False,
            "returncode": None,
            "passed": None,
            "failed": None,
            "wall_s": round(time.monotonic() - started, 3),
            "timeout": True,
        }


def _write_test_observation(root: Path, test_result: Mapping[str, Any]) -> None:
    # Only a bounded central observation is made available to the critique.
    # The file is ignored and disappears with the temporary repository.
    path = root / "disposable" / "test_result.txt"
    path.write_text(json.dumps(dict(test_result), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _grade(
    *,
    root: Path,
    seed: Mapping[str, Any],
    first: Optional[Mapping[str, Any]],
    critique: Optional[Mapping[str, Any]],
    first_error: Optional[str],
    critique_error: Optional[str],
    authoritative_test: Mapping[str, Any],
    provider_cost: float,
    wall_s: float,
) -> Dict[str, Any]:
    first_rows = _trace_rows(first)
    critique_rows = _trace_rows(critique)
    rows = first_rows + critique_rows
    successful = _successful_tools(rows)
    attempted = sorted({str(row.get("tool")) for row in rows if row.get("tool")})
    current_reads = sorted({
        path for row in rows if str(row.get("tool")) == "fs.read"
        for path in [_relative_path(root, (row.get("arguments") or {}).get("path"))]
        if path and not path.startswith("disposable/")
    })
    all_read_paths = sorted({
        path for row in rows if str(row.get("tool")) == "fs.read"
        for path in [_relative_path(root, (row.get("arguments") or {}).get("path"))]
        if path
    })
    search_count = sum(
        1 for row in rows
        if str(row.get("tool")) == "fs.search" and row.get("dispatched") is True and row.get("ok") is True
    )
    write_rows = [
        row for row in rows
        if str(row.get("tool")) == "filesystem.write" and row.get("dispatched") is True
    ]
    test_rows = [
        row for row in first_rows
        if str(row.get("tool")) == "tests.run" and row.get("dispatched") is True
    ]
    test_outcomes = [bool(_trace_value(row).get("verified")) for row in test_rows]
    repair_iterations = max(0, len(test_outcomes) - 1)
    repaired = bool(
        any(outcome is False for outcome in test_outcomes[:-1])
        and test_outcomes
        and test_outcomes[-1] is True
    )

    git_status_rows = [row for row in rows if str(row.get("tool")) == "git.status"]
    git_diff_rows = [row for row in rows if str(row.get("tool")) == "git.diff"]
    git_diff_grounded = any(
        row.get("ok") is True and "model_lake_selection.py" in str(_trace_value(row).get("stdout") or "")
        for row in git_diff_rows
    )
    status_observed = any(row.get("ok") is True for row in git_status_rows)
    diff_observed = any(row.get("ok") is True for row in git_diff_rows)

    current = _git_capture(root, ["diff", "--name-only"])
    # The baseline is intentionally staged before the worker starts so the
    # read-only git.diff tool can show edits to tracked proof placeholders.
    # Staged names are therefore not worker changes; only the worktree diff
    # and genuinely untracked paths belong in this grade.
    changed_paths = sorted({
        line.strip() for line in current.get("stdout", "").splitlines() if line.strip()
    })
    status = _git_capture(root, ["status", "--short"])
    for line in status.get("stdout", "").splitlines():
        if len(line) < 4:
            continue
        index_flag, worktree_flag = line[0], line[1]
        # `A ` is the staged disposable baseline and is deliberately ignored;
        # ` M`, `AM`, deletions, and `??` represent worktree changes.
        if not (worktree_flag != " " or (index_flag == "?" and worktree_flag == "?")):
            continue
        path = line[3:].strip()
        if path and path not in changed_paths:
            changed_paths.append(path)
    changed_paths = sorted(set(changed_paths))
    source_snapshot_changed = any(not path.startswith("disposable/") for path in changed_paths)
    disposable_changes = [path for path in changed_paths if path.startswith("disposable/")]

    implementation = root / "disposable" / "model_lake_selection.py"
    tests = root / "disposable" / "test_model_lake_selection.py"
    implementation_text = implementation.read_text(encoding="utf-8", errors="replace") if implementation.is_file() else ""
    test_text = tests.read_text(encoding="utf-8", errors="replace") if tests.is_file() else ""
    code_text = f"{implementation_text}\n{test_text}\n{_model_text((first, critique))}".lower()
    architecture_hits = [signal for signal in ARCHITECTURE_SIGNALS if signal in code_text]
    adversarial_hits = [signal for signal in ADVERSARIAL_SIGNALS if signal in code_text]
    test_count = len(re.findall(r"^\s*def\s+test_[A-Za-z0-9_]+", test_text, flags=re.MULTILINE))
    critique_text = _model_text((critique,))
    critique_hits = [signal for signal in CRITIQUE_SIGNALS if signal in critique_text]
    critique_reads_diff = any(
        str(row.get("tool")) == "git.diff" and row.get("ok") is True
        for row in critique_rows
    )
    critique_reads_tests = any(
        str(row.get("tool")) == "fs.read"
        and "disposable/test_result.txt" in str((row.get("arguments") or {}).get("path") or "")
        and row.get("ok") is True
        for row in critique_rows
    )

    # Objective score: tool evidence is weighted separately from prose. A
    # model cannot receive architecture points by merely promising a write.
    tool_points = 0.0
    if "tools.catalog" in successful:
        tool_points += 3.0
    if "fs.list" in successful:
        tool_points += 2.0
    tool_points += min(4.0, 2.0 * search_count)
    tool_points += min(6.0, 1.5 * len(current_reads))
    tool_points += min(4.0, 2.0 * len({
        _relative_path(root, (row.get("arguments") or {}).get("path"))
        for row in write_rows
        if _relative_path(root, (row.get("arguments") or {}).get("path"))
    }))
    if test_rows:
        tool_points += 2.0
    if status_observed:
        tool_points += 2.0
    if diff_observed:
        tool_points += 2.0

    navigation_points = min(15.0, len(current_reads) * 2.0)
    if any(path.startswith("receipts/") or path.startswith("docs/") for path in current_reads):
        navigation_points = min(15.0, navigation_points + 2.0)
    implementation_points = 0.0
    if implementation.is_file() and tests.is_file() and "disposable/model_lake_selection.py" in disposable_changes:
        implementation_points += 5.0
    implementation_points += min(6.0, len(architecture_hits) * 0.5)
    if "modellake_fixture.json" in " ".join(all_read_paths):
        implementation_points += 2.0
    if not source_snapshot_changed:
        implementation_points += 2.0

    test_points = 8.0 if authoritative_test.get("verified") is True else 0.0
    test_points += min(4.0, max(0, test_count - 2) * 0.5)
    test_points += min(3.0, len(adversarial_hits) * (3.0 / max(1, len(ADVERSARIAL_SIGNALS))))
    repair_points = 5.0 if repaired else 2.0 if test_outcomes and test_outcomes[-1] else 0.0
    git_points = 0.0
    if status_observed:
        git_points += 2.0
    if diff_observed:
        git_points += 2.0
    if git_diff_grounded:
        git_points += 2.0
    if disposable_changes and not source_snapshot_changed:
        git_points += 2.0
    architecture_points = min(10.0, len(architecture_hits) * (10.0 / max(1, len(ARCHITECTURE_SIGNALS))))
    if {"source", "candidate", "blocked"}.issubset(set(architecture_hits)):
        architecture_points = min(10.0, architecture_points + 2.0)
    self_critique_points = 0.0
    if critique and critique_rows:
        self_critique_points += 2.0
    if critique_reads_diff:
        self_critique_points += 2.0
    if critique_reads_tests:
        self_critique_points += 1.0
    self_critique_points += min(2.0, len(critique_hits) * 0.25)
    score = round(min(100.0, tool_points + navigation_points + implementation_points + test_points + repair_points + git_points + architecture_points + self_critique_points), 2)

    usage = _usage((first, critique))
    return {
        "model": MODEL_ID,
        "provider": PROVIDER,
        "status": "completed" if first and critique and authoritative_test.get("verified") is True else "incomplete",
        "error_code": {"implementation": first_error, "self_critique": critique_error},
        "overall_score": score,
        "tool_completion": {
            "requested": list(IMPLEMENTATION_TOOLS),
            "completed_successfully": successful,
            "attempted": attempted,
            "tool_call_count": len(rows),
            "search_calls": search_count,
            "targeted_source_reads": len(current_reads),
            "files_read": all_read_paths[:100],
            "write_calls": len(write_rows),
            "test_calls": len(test_rows),
            "repair_iterations": repair_iterations,
            "hawking_completion": dict(first.get("completion") or {})
            if isinstance(first, Mapping) and isinstance(first.get("completion"), Mapping)
            else None,
        },
        "repository_navigation": {
            "current_source_reads": current_reads[:100],
            "current_source_read_count": len(current_reads),
            "source_snapshot_files_copied": seed.get("copied_source_files"),
            "evidence_files_copied": seed.get("copied_evidence_files"),
        },
        "implementation": {
            "files_changed": disposable_changes,
            "source_snapshot_changed": source_snapshot_changed,
            "architecture_signals": architecture_hits,
            "adversarial_cases_covered": adversarial_hits,
            "test_count_in_disposable_file": test_count,
        },
        "tests": {
            "worker_reported_outcomes": test_outcomes,
            "authoritative_verified": authoritative_test.get("verified") is True,
            "authoritative_passed": authoritative_test.get("passed"),
            "authoritative_failed": authoritative_test.get("failed"),
            "authoritative_wall_s": authoritative_test.get("wall_s"),
            "repair_observed": repaired,
        },
        "git": {
            "status_observed": status_observed,
            "diff_observed": diff_observed,
            "diff_grounded_in_implementation": git_diff_grounded,
            "changed_paths": changed_paths[:100],
            "scope_clean": bool(disposable_changes) and not source_snapshot_changed,
        },
        "self_critique": {
            "completed": bool(critique and critique_rows),
            "read_test_observation": critique_reads_tests,
            "read_git_diff": critique_reads_diff,
            "signals": critique_hits,
        },
        "usage": usage,
        "provider_turns": _provider_turns((first, critique)),
        "retries": _provider_retries((first, critique)),
        "observed_cost_usd": round(max(float(provider_cost), float(usage.get("cost_usd") or 0.0)), 8),
        "wall_s": round(wall_s, 3),
        "output_discarded": True,
    }


def _run_qualification(resolver: Any) -> Dict[str, Any]:
    started = time.monotonic()
    model_slug = _model_slug(MODEL_ID)
    with tempfile.TemporaryDirectory(prefix=f"hawking-final-{model_slug}-") as temporary:
        root = Path(temporary).resolve()
        seed = _seed_disposable_repo(root)
        registry = default_tool_registry(
            root,
            repo_root=root,
            permissions=SAFE_PERMISSIONS,
        )
        provider = OpenRouterProvider(
            MODEL_ID,
            workspace=root,
            cost_policy=RemoteCostPolicy(
                max_cost_usd=HARD_BUDGET_USD,
                mission_limit_usd=HARD_BUDGET_USD,
                monthly_limit_usd=9.99,
            ),
            key_resolver=resolver,
            timeout=LANE_TIMEOUT_S,
            max_attempts=2,
            backoff_s=0.5,
            max_backoff_s=2.0,
            remote_slot="final-deepseek-qualification",
        )
        implementation_prompt = f"""Perform one realistic but disposable Hawking mini-Odyssey qualification as the exact model {MODEL_ID}.

The repository available through the tools is a bounded copy of the CURRENT dirty Hawking source,
docs, receipts, lifecycle code, and ModelLake evidence. Infer ownership and boundaries from those
files; do not rely on this prompt for the answer. The only production-like input fixture is
disposable/modellake_fixture.json. Work only in the disposable repository. Never use network,
downloads, shells, native runtimes, Flash/Pulsar, HCLI, AgentOS, HIDE, Grok, or any unoffered tool.
Do not edit any copied current Hawking source or receipt. The experiment is thrown away.

Current tool names are authoritative. repo.search/repo.read may not exist; use the typed names that
tools.catalog actually exposes. You must perform real calls, not narrate hypothetical calls.

Navigation requirements:
1. Call tools.catalog for the current read/write/test/git surface.
2. Call fs.list on the repository root with recursive=true or on hawking with recursive=true.
3. Make at least two distinct fs.search calls for ownership/lifecycle terms such as ModelLake,
   active_owner, promotion, Goal, or WorkUnit.
4. Make at least four targeted fs.read calls of files returned by search/list. Use line windows;
   do not dump whole large files. Read at least one current source file, one current doc, and one
   current receipt/evidence file.
5. Call tests.list to discover the test surface before implementation.

Implementation requirements:
6. After the archaeology, use filesystem.write to overwrite exactly disposable/model_lake_selection.py
   with a small dependency-free helper that normalizes the fixture and exposes a clear selection
   view for READY/COLLAPSED, SOURCE/NOT YET COLLAPSED, CANDIDATES, and BLOCKED. Preserve identity
   relationships and explicit fields for source hash, NR revision, NX revision/target, receipts,
   device qualification, active owner, and next experiment. Do not create a second production
   registry; this is a disposable representation only.
7. Use filesystem.write again to overwrite exactly disposable/test_model_lake_selection.py with
   objective pytest tests. Cover several adversarial cases: source without NR, NR without a
   qualifying receipt, multiple NX revisions from one NR, another active owner, stale source hash,
   blocked candidate, ambiguous source/collapsed names, and target-specific NX availability.
8. Call tests.run on the disposable test file. If it fails, read the bounded observation, repair
   the implementation or tests with filesystem.write, and rerun tests.run. Do not hide a failure.
9. Call git.status and git.diff after the final test. The diff must be limited to disposable proof
   files. Do not commit or land anything.

Return one JSON object with exactly these top-level fields: observations, operations, tests,
questions, status, content. Keep it concise. State which current Hawking owners you found, what
the disposable helper proves, what remained unsupported, and whether the result should be merged
(it should not be merged just because this experiment passes)."""
        workunit = SimpleNamespace(
            id="deepseek-final-mini-odyssey",
            role="frontier_model_lake_qualification",
            description="Disposable ModelLake organization proof with tests and repair.",
        )
        first: Optional[Dict[str, Any]] = None
        first_error: Optional[str] = None
        try:
            first = provider.execute_workunit(
                workunit,
                {
                    "prompt": implementation_prompt,
                    "objective": "Prove a disposable ModelLake organization helper using current Hawking evidence.",
                    "scope": ["temporary repository only", "read current evidence", "write disposable proof", "run bounded tests"],
                    "authority": "current Hawking source, live tool results, and authoritative pytest result",
                    "tests": ["disposable/test_model_lake_selection.py passes independently"],
                    "stop": ["hard cost cap $0.50", "no production changes", "no external model/runtime operations"],
                    "tool_registry": registry,
                    "hawking_tools": True,
                    "tool_write_authority": True,
                    "mutation_lock_held": True,
                    "allowed_tool_names": list(IMPLEMENTATION_TOOLS),
                    "completion_contract": {
                        "required_successful_tools": [
                            "filesystem.write", "tests.run", "git.status", "git.diff",
                        ],
                        "require_final_tests_pass": True,
                        "require_structured_result": True,
                        "max_continuation_turns": 8,
                        "read_only_tool_call_limit": 10,
                        "reminder": (
                            "The disposable WorkUnit is not complete. Hawking has "
                            "not accepted the prose stop. Continue with real "
                            "filesystem.write, tests.run, git.status, and git.diff "
                            "evidence, then return the required JSON object."
                        ),
                    },
                    "max_tool_calls": MAX_TOOL_CALLS,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.1,
                    "timeout": LANE_TIMEOUT_S,
                    "estimated_cost_usd": 0.005,
                    "remote_cost_authorization": {"max_usd": HARD_BUDGET_USD},
                },
            )
        except Exception as exc:  # keep the evidence envelope even on a bounded failure
            first_error = _safe_code(exc)

        authoritative_test = _run_authoritative_test(root)
        _write_test_observation(root, authoritative_test)

        critique_prompt = """You are the SAME DeepSeek worker completing the final self-critique pass.

The first pass attempted a disposable ModelLake organization helper in
disposable/model_lake_selection.py and tests in disposable/test_model_lake_selection.py. Inspect
those files, disposable/test_result.txt, and the actual Git diff with the offered tools. Do not
write anything in this pass. Answer with one JSON object using exactly these top-level fields:
observations, operations, tests, questions, status, content.

Your critique must explicitly address: what was duplicated; which existing Hawking owner should
absorb the idea instead of a second subsystem; unsupported assumptions; what fails at 100+ artifacts;
what makes direct safetensors -> NR -> NX difficult; and what must NOT be merged from this disposable
experiment. Distinguish worker claims from the authoritative test result and diff. Use several
targeted reads rather than dumping the repository."""
        critique_unit = SimpleNamespace(
            id="deepseek-final-mini-odyssey-critique",
            role="frontier_model_lake_self_critique",
            description="Inspect disposable implementation, test evidence, and diff; critique mergeability.",
        )
        critique: Optional[Dict[str, Any]] = None
        critique_error: Optional[str] = None
        try:
            critique = provider.execute_workunit(
                critique_unit,
                {
                    "prompt": critique_prompt,
                    "objective": "Critique the disposable proof against current Hawking ownership and evidence rules.",
                    "scope": ["read-only critique of disposable files and bounded test observation"],
                    "authority": "actual disposable Git diff and independently run pytest result",
                    "tests": ["read disposable/test_result.txt and inspect the test result"],
                    "stop": ["no writes", "hard cost cap $0.50"],
                    "tool_registry": registry,
                    "hawking_tools": True,
                    "tool_write_authority": False,
                    "mutation_lock_held": False,
                    "allowed_tool_names": list(CRITIQUE_TOOLS),
                    "max_tool_calls": MAX_CRITIQUE_TOOL_CALLS,
                    "max_tokens": MAX_CRITIQUE_TOKENS,
                    "temperature": 0.1,
                    "timeout": LANE_TIMEOUT_S,
                    "estimated_cost_usd": 0.005,
                    "remote_cost_authorization": {"max_usd": HARD_BUDGET_USD},
                },
            )
        except Exception as exc:
            critique_error = _safe_code(exc)

        return _grade(
            root=root,
            seed=seed,
            first=first,
            critique=critique,
            first_error=first_error,
            critique_error=critique_error,
            authoritative_test=authoritative_test,
            provider_cost=provider.cost_policy.spent_usd,
            wall_s=time.monotonic() - started,
        )


def _catalog_projection(row: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"id": row.get("id"), "name": row.get("name")}
    for key in (
        "context_length", "pricing", "input_modalities", "output_modalities",
        "supported_parameters", "architecture", "canonical_slug",
    ):
        if key in row:
            result[key] = row.get(key)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-usd", type=float, default=HARD_BUDGET_USD)
    args = parser.parse_args(argv)
    budget = max(0.0, float(args.budget_usd))
    if budget < HARD_BUDGET_USD:
        print(json.dumps({
            "status": "refused",
            "reason": "requested budget is below the final qualification hard cap",
            "budget_usd": budget,
            "hard_budget_usd": HARD_BUDGET_USD,
        }, sort_keys=True))
        return 2

    resolver = _keychain_resolver_from_environment()
    key_present = False
    if callable(resolver):
        try:
            key_present = bool(resolver())
        except Exception:
            key_present = False
    print(f"OPENROUTER_KEY_PRESENT={'PASS' if key_present else 'FAIL'}", flush=True)
    if not key_present:
        return 2

    catalog_provider = OpenRouterProvider(
        "__catalog__",
        key_resolver=resolver,
        timeout=LANE_TIMEOUT_S,
        max_attempts=2,
        backoff_s=0.5,
        max_backoff_s=2.0,
        catalog_ttl_s=0.0,
    )
    try:
        catalog = catalog_provider.list_models(force=True, timeout=LANE_TIMEOUT_S)
    except Exception as exc:
        print(json.dumps({
            "status": "catalog_failed",
            "model": MODEL_ID,
            "provider": PROVIDER,
            "error_code": _safe_code(exc),
            "output_discarded": True,
        }, sort_keys=True))
        return 1

    model_row = next(
        (dict(row) for row in catalog if isinstance(row, Mapping) and str(row.get("id") or "") == MODEL_ID),
        None,
    )
    if model_row is None:
        print(json.dumps({
            "status": "model_not_in_live_catalog",
            "model": MODEL_ID,
            "provider": PROVIDER,
            "catalog_count": len(catalog),
            "output_discarded": True,
        }, sort_keys=True))
        return 1

    print(json.dumps({
        "status": "launching_final_qualification",
        "model": MODEL_ID,
        "provider": PROVIDER,
        "catalog_count": len(catalog),
        "live_catalog_model": _catalog_projection(model_row),
        "hard_budget_usd": HARD_BUDGET_USD,
        "production_changes_allowed": False,
        "parallel_workers": 1,
    }, sort_keys=True), flush=True)

    report = _run_qualification(resolver)
    report["live_catalog_model"] = _catalog_projection(model_row)
    report["catalog_count"] = len(catalog)
    report["budget_usd"] = budget
    report["hard_budget_usd"] = HARD_BUDGET_USD
    report["authoritative_tree_changed"] = False
    report["canonical_modellake_changed"] = False
    report["sealed_receipts_changed"] = False
    report["output_discarded"] = True
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
