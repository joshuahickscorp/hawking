#!/usr/bin/env python3.12
"""Black-box behaviour matrix runner for the rebuild constitution.

Usage:
  python3.12 tools/verify/blackbox.py [--domain X] [--only-runnable] [--json]
  python3.12 tools/verify/blackbox.py --help

Anti-gaming rule: exit non-zero if any check that was previously runnable
becomes unrunnable (fixture/command vanished). Silent skips on formerly
runnable checks are failures.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.layout import evidence_dir, resolve_workspace_path

MATRIX_PATH = evidence_dir("rebuild") / "REBUILD_BLACKBOX_TEST_MATRIX.json"
CONSTITUTION_PATH = evidence_dir("rebuild") / "REBUILD_BEHAVIOUR_CONSTITUTION.json"
BASELINE_RUNNABLE_PATH = ROOT / "tools" / "verify" / "blackbox_runnable_baseline.json"
DEFAULT_TIMEOUT_S = 60

# The matrix is sealed evidence: its commands keep the paths that were true
# when the matrix was recorded.  Translate only recognized workspace roots at
# execution time, retaining the exact recorded command in reports.
_LOGICAL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?P<path>(?:adapters|control|evidence|packs|profiles|prompts|receipts|reports|tests|vendor)/"
    r"[A-Za-z0-9_./-]+)"
)
_LAYOUT_PATH_ROOTS = frozenset(
    {
        "adapters",
        "control",
        "evidence",
        "packs",
        "profiles",
        "prompts",
        "receipts",
        "reports",
        "tests",
        "vendor",
        "workspace",
    }
)


def resolve_command_paths(command: str) -> str:
    """Map sealed, historical command paths to live files without mutating evidence."""
    return _LOGICAL_PATH_RE.sub(
        lambda match: str(resolve_workspace_path(match.group("path"))), command
    )


def resolve_layout_arg(path: Path) -> Path:
    """Accept a historical layout path for CLI input while keeping other paths literal."""
    if path.is_absolute() or not path.parts or path.parts[0] not in _LAYOUT_PATH_ROOTS:
        return path
    return resolve_workspace_path(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def shell_ok(command: str, timeout: int = DEFAULT_TIMEOUT_S) -> tuple[int, str, str]:
    """Run a matrix command; return (rc, stdout, stderr).

    Prefer argv form without an extra login shell when the command is a simple
    ``python3.12 -c ...`` so 100+ runnable checks stay fast.
    """
    import shlex

    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    try:
        argv = shlex.split(command)
    except ValueError:
        argv = []

    try:
        if argv and argv[0] in ("python3.12", "python3", "python"):
            proc = subprocess.run(
                argv,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        else:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = (e.stderr or "") if isinstance(e.stderr, str) else f"timeout after {timeout}s"
        return 124, out, err
    except OSError as e:
        return 127, "", str(e)


def evaluate_check(check: dict[str, Any]) -> dict[str, Any]:
    """Execute one matrix check. Returns result dict with status pass|fail|skip."""
    bid = check.get("behaviour_id", "?")
    domain = check.get("domain", "?")
    command = (check.get("command") or "").strip()
    assertion = check.get("assertion") or ""
    declared_runnable = bool(check.get("runnable_now"))
    blocker = check.get("blocker")
    fixture = check.get("fixture")
    fixture_exists = check.get("fixture_exists")

    result: dict[str, Any] = {
        "behaviour_id": bid,
        "domain": domain,
        "declared_runnable_now": declared_runnable,
        "command": command,
        "status": "skip",
        "reason": None,
        "exit_code": None,
        "seconds": None,
    }

    if not declared_runnable:
        result["status"] = "skip"
        result["reason"] = blocker or "matrix marks runnable_now=false"
        return result

    # Runnable checks must still be runnable: fixture present if claimed.
    if fixture and fixture not in ("none", "n/a", "N/A") and fixture_exists is False:
        result["status"] = "fail"
        result["reason"] = (
            f"anti-gaming: declared runnable but fixture_exists=false ({fixture})"
        )
        return result

    if not command:
        result["status"] = "fail"
        result["reason"] = "anti-gaming: runnable check has empty command"
        return result

    resolved_command = resolve_command_paths(command)
    if resolved_command != command:
        result["resolved_command"] = resolved_command

    t0 = time.monotonic()
    rc, stdout, stderr = shell_ok(resolved_command)
    result["seconds"] = round(time.monotonic() - t0, 3)
    result["exit_code"] = rc
    result["stdout_tail"] = stdout[-500:]
    result["stderr_tail"] = stderr[-500:]

    if rc != 0:
        # If the command failed because a fixture vanished, call it unrunnable fail.
        combined = (stdout + "\n" + stderr).lower()
        if any(
            x in combined
            for x in (
                "no such file",
                "not found",
                "cannot find",
                "missing",
                "errno 2",
            )
        ):
            result["status"] = "fail"
            result["reason"] = (
                "anti-gaming: previously runnable check is now unrunnable "
                f"(exit {rc}): {stderr.strip() or stdout.strip() or assertion}"
            )
            return result
        result["status"] = "fail"
        result["reason"] = f"command failed exit={rc}: {stderr.strip() or stdout.strip() or assertion}"
        return result

    result["status"] = "pass"
    result["reason"] = "exit 0"
    return result


def load_baseline_runnable_ids() -> set[str]:
    """Ids that were runnable when the constitution was frozen.

    Prefer the committed baseline file; if absent, treat matrix runnable_now
    as the baseline (first run writes the file).
    """
    if BASELINE_RUNNABLE_PATH.is_file():
        data = load_json(BASELINE_RUNNABLE_PATH)
        return set(data.get("runnable_behaviour_ids") or [])
    return set()


def write_baseline_if_absent(runnable_ids: list[str]) -> None:
    if BASELINE_RUNNABLE_PATH.is_file():
        return
    BASELINE_RUNNABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_RUNNABLE_PATH.write_text(
        json.dumps(
            {
                "schema": "hawking.rebuild.blackbox_runnable_baseline.v1",
                "runnable_behaviour_ids": sorted(runnable_ids),
                "note": (
                    "Frozen set of behaviour ids with runnable_now=true at constitution "
                    "seal. If a later matrix marks one false or its command fails as "
                    "missing, blackbox.py must fail (anti-gaming)."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run REBUILD_BLACKBOX_TEST_MATRIX.json checks."
    )
    parser.add_argument("--domain", help="Only run checks in this domain id")
    parser.add_argument(
        "--only-runnable",
        action="store_true",
        help="Execute only checks with runnable_now=true (still enforces anti-gaming)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit machine-readable JSON report on stdout",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=MATRIX_PATH,
        help="Path to matrix JSON",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help="Per-check timeout seconds (default 60)",
    )
    args = parser.parse_args(argv)
    args.matrix = resolve_layout_arg(args.matrix)

    if not args.matrix.is_file():
        print(f"error: matrix not found: {args.matrix}", file=sys.stderr)
        return 2

    matrix = load_json(args.matrix)
    if matrix.get("schema") != "hawking.rebuild.blackbox_test_matrix.v1":
        print(
            f"error: unexpected matrix schema {matrix.get('schema')!r}",
            file=sys.stderr,
        )
        return 2

    checks: list[dict[str, Any]] = list(matrix.get("checks") or [])
    if args.domain:
        checks = [c for c in checks if c.get("domain") == args.domain]

    matrix_runnable_ids = [
        c["behaviour_id"] for c in (matrix.get("checks") or []) if c.get("runnable_now")
    ]
    write_baseline_if_absent(matrix_runnable_ids)
    baseline_ids = load_baseline_runnable_ids()
    if not baseline_ids:
        baseline_ids = set(matrix_runnable_ids)

    # Anti-gaming: every baseline runnable id must still be runnable_now in matrix.
    matrix_by_id = {c["behaviour_id"]: c for c in (matrix.get("checks") or [])}
    anti_gaming_failures: list[dict[str, Any]] = []
    for bid in sorted(baseline_ids):
        c = matrix_by_id.get(bid)
        if c is None:
            anti_gaming_failures.append(
                {
                    "behaviour_id": bid,
                    "domain": "?",
                    "declared_runnable_now": False,
                    "command": "",
                    "status": "fail",
                    "reason": "anti-gaming: baseline runnable id missing from matrix",
                    "exit_code": None,
                    "seconds": None,
                }
            )
            continue
        if not c.get("runnable_now"):
            anti_gaming_failures.append(
                {
                    "behaviour_id": bid,
                    "domain": c.get("domain"),
                    "declared_runnable_now": False,
                    "command": c.get("command"),
                    "status": "fail",
                    "reason": (
                        "anti-gaming: check was previously runnable but "
                        f"runnable_now is now false (blocker={c.get('blocker')!r})"
                    ),
                    "exit_code": None,
                    "seconds": None,
                }
            )

    if args.only_runnable:
        to_run = [c for c in checks if c.get("runnable_now")]
    else:
        to_run = checks

    results: list[dict[str, Any]] = list(anti_gaming_failures)
    for check in to_run:
        # Avoid double-running baseline demotions already recorded.
        if any(
            r["behaviour_id"] == check.get("behaviour_id") and r["status"] == "fail"
            for r in anti_gaming_failures
        ):
            continue
        results.append(evaluate_check(check))

    n_pass = sum(1 for r in results if r["status"] == "pass")
    n_fail = sum(1 for r in results if r["status"] == "fail")
    n_skip = sum(1 for r in results if r["status"] == "skip")

    # Domain rollup from constitution if present.
    domain_counts: dict[str, int] = {}
    if CONSTITUTION_PATH.is_file():
        const = load_json(CONSTITUTION_PATH)
        for d in const.get("domains") or []:
            domain_counts[d["id"]] = d.get("behaviour_count", 0)

    report = {
        "schema": "hawking.rebuild.blackbox_report.v1",
        "matrix_schema": matrix.get("schema"),
        "commit": matrix.get("commit"),
        "only_runnable": args.only_runnable,
        "domain_filter": args.domain,
        "baseline_runnable_count": len(baseline_ids),
        "summary": {
            "pass": n_pass,
            "fail": n_fail,
            "skip": n_skip,
            "total_results": len(results),
        },
        "constitution_domain_counts": domain_counts,
        "results": results,
    }

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print("blackbox matrix report")
        print(f"  matrix: {args.matrix}")
        print(f"  only_runnable: {args.only_runnable}")
        if args.domain:
            print(f"  domain: {args.domain}")
        print(
            f"  pass={n_pass} fail={n_fail} skip={n_skip} "
            f"(baseline_runnable={len(baseline_ids)})"
        )
        if domain_counts:
            print("  constitution behaviours per domain:")
            for k, v in domain_counts.items():
                print(f"    {k}: {v}")
        for r in results:
            if r["status"] == "skip" and args.only_runnable:
                continue
            mark = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[r["status"]]
            reason = r.get("reason") or ""
            print(f"  [{mark}] {r['behaviour_id']} {reason}")

    # Exit non-zero on any fail (includes anti-gaming).
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
