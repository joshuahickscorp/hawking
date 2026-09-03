#!/usr/bin/env python3
"""Cold-read a HAWKING_HEADLESS_COMPLETION_V3 handoff.

The test knows nothing except what the handoff file says. For every claim it:

  1. resolves the evidence path (this worktree disk, git HEAD, primary checkout)
  2. runs the reproducing command from the repository root

A claim whose evidence is missing, whose command is missing or unrunnable, or
whose command exits non-zero is UNSUPPORTED.

This test must be able to FAIL. After the real claims it injects a negative
control that points at a deleted receipt and a command that exits non-zero.
If that control comes back SUPPORTED, the validator is vacuous and the run
fails even if every real claim passed.

Staging is promoted to `.hcli-legacy/HAWKING_HEADLESS_COMPLETION_V3.json` only
when every real claim is SUPPORTED and the negative control is UNSUPPORTED.

Run:
    python3 tools/headless/handoff_cold_read_test.py
    python3 tools/headless/handoff_cold_read_test.py --handoff PATH
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGING_REL = ".hcli-legacy/HAWKING_HEADLESS_COMPLETION_V3.staging.json"
FINAL_REL = ".hcli-legacy/HAWKING_HEADLESS_COMPLETION_V3.json"

NEGATIVE_CONTROL = {
    "id": "NEGATIVE_CONTROL_DELETED_RECEIPT",
    "kind": "capability",
    "statement": (
        "deliberately broken: points at a receipt that does not exist and a "
        "command that exits non-zero"
    ),
    "evidence_path": "receipts/headless/DOES_NOT_EXIST_NEGATIVE_CONTROL.json",
    "reproducing_command": [
        "python3",
        "-c",
        "raise SystemExit('negative control: receipt missing')",
    ],
}


def git(args: Sequence[str], cwd: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def primary_checkout(repo: Path = REPO) -> Optional[Path]:
    r = git(["rev-parse", "--git-common-dir"], cwd=repo)
    if r.returncode != 0 or not (r.stdout or "").strip():
        return None
    cd = r.stdout.strip()
    p = Path(cd)
    if not p.is_absolute():
        p = (repo / p).resolve()
    else:
        p = p.resolve()
    root = p.parent if p.name == ".git" else p.parent
    return root if root.is_dir() else None


def resolve_evidence(rel: str, repo: Path = REPO) -> Tuple[bool, str]:
    """A missing file in a sparse checkout is not evidence it does not exist."""
    if not rel or not str(rel).strip():
        return False, "empty evidence_path"
    p = repo / rel
    if p.exists():
        return True, f"disk:{p}"
    r = git(["cat-file", "-e", f"HEAD:{rel}"], cwd=repo)
    if r.returncode == 0:
        return True, f"git:HEAD:{rel}"
    prim = primary_checkout(repo)
    if prim is not None:
        cand = prim / rel
        if cand.exists():
            return True, f"primary:{cand}"
    return False, "missing"


def find_handoff(explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (REPO / p).resolve()
        if not p.is_file():
            raise SystemExit(f"handoff not found: {p}")
        return p
    env = os.environ.get("HANDOFF_PATH", "").strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = (REPO / p).resolve()
        if not p.is_file():
            raise SystemExit(f"HANDOFF_PATH not found: {p}")
        return p
    staging = REPO / STAGING_REL
    if staging.is_file():
        return staging
    final = REPO / FINAL_REL
    if final.is_file():
        return final
    raise SystemExit(
        "no handoff found; run python3 tools/headless/handoff_builder.py first "
        f"(looked for {STAGING_REL} and {FINAL_REL})"
    )


def load_handoff(path: Path) -> Dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"handoff is not JSON: {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise SystemExit(f"handoff is not a JSON object: {path}")
    return doc


def claims_of(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = doc.get("claims")
    if not isinstance(raw, list):
        raise SystemExit("handoff has no claims[] list")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SystemExit(f"claims[{i}] is not an object")
        out.append(item)
    return out


def run_command(
    cmd: Any,
    repo: Path,
    handoff: Path,
    timeout: float,
) -> Tuple[int, str, str, str]:
    """Return (exit_code, how, stdout_tail, stderr_tail). how describes the run."""
    env = os.environ.copy()
    env["HANDOFF_PATH"] = str(handoff)
    env["PYTHONPATH"] = (
        str(HERE)
        + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    if cmd is None:
        return 127, "missing-command", "", "claim has no reproducing_command"
    if isinstance(cmd, str):
        if not cmd.strip():
            return 127, "empty-command", "", "reproducing_command is empty"
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
            )
        except subprocess.TimeoutExpired as exc:
            return 124, "timeout", (exc.stdout or "")[-2000:], (exc.stderr or "")[-2000:]
        except OSError as exc:
            return 127, "unrunnable", "", str(exc)
        return proc.returncode, "shell", (proc.stdout or "")[-2000:], (proc.stderr or "")[-2000:]
    if isinstance(cmd, list) and cmd and all(isinstance(x, str) for x in cmd):
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return 124, "timeout", (exc.stdout or "")[-2000:], (exc.stderr or "")[-2000:]
        except OSError as exc:
            return 127, "unrunnable", "", str(exc)
        return proc.returncode, "argv", (proc.stdout or "")[-2000:], (proc.stderr or "")[-2000:]
    return 127, "bad-command", "", f"reproducing_command has unusable type {type(cmd).__name__}"


def timeout_for(claim: Dict[str, Any], default: float) -> float:
    cmd = claim.get("reproducing_command")
    blob = " ".join(cmd) if isinstance(cmd, list) else str(cmd or "")
    if blob.endswith("_test.py") or " _test.py" in blob or blob.endswith("_test.py"):
        return max(default, 120.0)
    if blob.endswith("_test.py"):
        return max(default, 120.0)
    if isinstance(cmd, list) and cmd and str(cmd[-1]).endswith("_test.py"):
        return max(default, 120.0)
    return default


def evaluate_claim(
    claim: Dict[str, Any],
    repo: Path,
    handoff: Path,
    default_timeout: float,
) -> Dict[str, Any]:
    cid = str(claim.get("id") or "?")
    kind = str(claim.get("kind") or "?")
    evidence = str(claim.get("evidence_path") or "")
    cmd = claim.get("reproducing_command")
    ev_ok, ev_how = resolve_evidence(evidence, repo=repo)
    cmd_missing = cmd in (None, "", [], {})
    if cmd_missing:
        rc, how, out, err = 127, "missing-command", "", "no reproducing_command"
    else:
        rc, how, out, err = run_command(
            cmd, repo, handoff, timeout_for(claim, default_timeout)
        )
    supported = bool(ev_ok) and rc == 0 and not cmd_missing
    reasons = []
    if not ev_ok:
        reasons.append(f"evidence {ev_how}: {evidence}")
    if cmd_missing:
        reasons.append("no reproducing_command")
    elif rc != 0:
        reasons.append(f"command exit {rc} ({how})")
    return {
        "id": cid,
        "kind": kind,
        "statement": claim.get("statement"),
        "evidence_path": evidence,
        "evidence_resolved": ev_ok,
        "evidence_via": ev_how,
        "command_exit": rc,
        "command_how": how,
        "supported": supported,
        "verdict": "SUPPORTED" if supported else "UNSUPPORTED",
        "reasons": reasons,
        "stdout_tail": out,
        "stderr_tail": err,
    }


def print_row(row: Dict[str, Any]) -> None:
    flag = "SUPPORTED  " if row["supported"] else "UNSUPPORTED"
    print(f"{flag}  {row['id']:28}  {row['kind']:14}  {row['evidence_path']}")
    stmt = row.get("statement") or ""
    if stmt:
        print(f"            {stmt}")
    if not row["supported"]:
        for r in row.get("reasons") or []:
            print(f"            reason: {r}")
        err = (row.get("stderr_tail") or "").strip()
        if err:
            tail = err.splitlines()[-8:]
            for ln in tail:
                print(f"            stderr: {ln}")
        out = (row.get("stdout_tail") or "").strip()
        if out and not row["supported"]:
            tail = out.splitlines()[-4:]
            for ln in tail:
                print(f"            stdout: {ln}")


def promote(staging: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(final.suffix + ".tmp")
    shutil.copyfile(staging, tmp)
    os.replace(tmp, final)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", default="", help="path to the handoff JSON")
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="seconds per reproducing command (tests get at least 120)",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="do not copy staging to the durable path even on success",
    )
    parser.add_argument(
        "--negative-only",
        action="store_true",
        help="run only the injected negative control (for demonstrating failure)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    t0 = time.time()
    if args.negative_only:
        row = evaluate_claim(NEGATIVE_CONTROL, REPO, REPO / STAGING_REL, args.timeout)
        print("## NEGATIVE CONTROL (standalone)")
        print_row(row)
        if row["supported"]:
            print("VACUOUS: negative control was SUPPORTED — validator is broken")
            return 1
        print("negative control correctly reported UNSUPPORTED")
        return 0

    path = find_handoff(args.handoff)
    print(f"cold-read: {path}")
    doc = load_handoff(path)
    schema = doc.get("schema")
    print(f"schema: {schema}")
    print(f"generated_at: {doc.get('generated_at')}")
    print(f"builder: {doc.get('builder')}")
    print()

    claims = claims_of(doc)
    rows = []
    for claim in claims:
        row = evaluate_claim(claim, REPO, path, args.timeout)
        print_row(row)
        rows.append(row)

    print()
    print("## NEGATIVE CONTROL")
    neg = evaluate_claim(NEGATIVE_CONTROL, REPO, path, args.timeout)
    print_row(neg)
    print()

    supported = sum(1 for r in rows if r["supported"])
    unsupported = sum(1 for r in rows if not r["supported"])
    by_kind: Dict[str, Dict[str, int]] = {}
    for r in rows:
        slot = by_kind.setdefault(r["kind"], {"supported": 0, "unsupported": 0})
        slot["supported" if r["supported"] else "unsupported"] += 1

    print("## COUNTS")
    print(f"real claims: {len(rows)}")
    print(f"SUPPORTED:   {supported}")
    print(f"UNSUPPORTED: {unsupported}")
    for kind in sorted(by_kind):
        s = by_kind[kind]
        print(f"  {kind:14}  supported={s['supported']}  unsupported={s['unsupported']}")
    print(
        "negative control: "
        + neg["verdict"]
        + (" (required)" if not neg["supported"] else " (VALIDATOR IS VACUOUS)")
    )
    print(f"elapsed_s: {time.time() - t0:.1f}")

    failures = [r for r in rows if not r["supported"]]
    if failures:
        print()
        print("## REAL CLAIMS THAT DID NOT SURVIVE")
        for r in failures:
            print(f"- {r['id']}: {r['statement']}")
            for reason in r.get("reasons") or []:
                print(f"    {reason}")

    if neg["supported"]:
        print()
        print(
            "FAIL: the negative control was SUPPORTED. A handoff validator "
            "that has never been seen to reject anything validates nothing."
        )
        return 1
    if failures:
        print()
        print("FAIL: one or more real claims are UNSUPPORTED")
        return 1

    print()
    print("all real claims SUPPORTED; negative control UNSUPPORTED")
    staging = (REPO / STAGING_REL).resolve()
    final = (REPO / FINAL_REL).resolve()
    tested = path.resolve()
    if args.no_promote:
        print("not promoting (--no-promote)")
        return 0
    if tested == final:
        print(f"already at durable path: {final}")
        return 0
    if tested == staging or tested.exists():
        promote(tested, final)
        print(f"promoted -> {final}")
        return 0
    print("not promoting: handoff path is neither staging nor durable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
