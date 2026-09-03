#!/usr/bin/env python3.12
"""Probe the local formal toolchain and write an honest self-test receipt.

Runs now, under LIGHT_ONLY, because probing costs nothing.  It does not install
anything: `elan`/Lean is roughly a gigabyte and a Mathlib build is CPU-heavy for a long
time, which is exactly what must not run beside MOP tonight.

Each tool is reported as INSTALLED with its real version, or NOT_INSTALLED.  Nothing is
marked ready on the strength of an intention.  A missing tool is a missing tool.

    python3.12 -m ramanujan.toolchain_selftest
"""
from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ramanujan.layout import AUDITS_ROOT, CONTAINER_ROOT, REPO_ROOT, RUNTIME_RECORDS_ROOT
from ramanujan.limits import LimitRegistry

ROOT = REPO_ROOT
OUT = RUNTIME_RECORDS_ROOT / "RAMANUJAN_TOOLCHAIN_SELFTEST.json"

# (name, probe argv, why Ramanujan needs it)
BINARIES = [
    ("elan", ["elan", "--version"], "Lean toolchain manager; pins the Lean version"),
    ("lean", ["lean", "--version"], "the proof assistant; Tier 3 machine checks run here"),
    ("lake", ["lake", "--version"], "Lean build tool; fetches and builds Mathlib"),
    ("z3", ["z3", "--version"], "SMT solver; counterexample and constraint search"),
    ("cvc5", ["cvc5", "--version"], "second SMT solver; disagreement between solvers is a signal"),
    ("gp", ["gp", "--version"], "PARI/GP; number-theoretic computation"),
    ("gap", ["gap", "--version"], "group theory and combinatorics"),
    ("cadical", ["cadical", "--version"], "SAT solver"),
]

MODULES = [
    ("sympy", "symbolic algebra; the informal-to-formal bridge"),
    ("numpy", "numeric computation; already present for the Hawking oracle"),
    ("scipy", "numeric computation"),
    ("z3", "Z3 Python bindings"),
    ("pysat", "SAT interfaces"),
]


# Tools install to their own prefixes and are not necessarily on this process's PATH.
# elan puts lean/lake/elan in ~/.elan/bin, and probing PATH alone reported all three
# missing minutes after installing and machine-checking a theorem with them. Same class of
# false negative as the venv modules: a probe-path artifact reported as a toolchain gap.
EXTRA_BIN_DIRS = [
    Path.home() / ".elan/bin",
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
]


def _which(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for d in EXTRA_BIN_DIRS:
        cand = d / name
        if cand.is_file():
            return str(cand)
    return None


def probe_binary(name: str, argv: list[str]) -> dict:
    path = _which(name)
    if not path:
        return {"status": "NOT_INSTALLED", "path": None, "version": None}
    argv = [path, *argv[1:]]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        version = (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else None
    except Exception as e:  # a binary that exists but will not run is not installed for our purposes
        return {"status": "PRESENT_BUT_UNUSABLE", "path": path, "version": None, "error": str(e)}
    return {"status": "INSTALLED", "path": path, "version": version}


# The formal-tool Python packages live in the project venv, not in whatever interpreter
# happens to run this script. Probing only the current interpreter reported sympy, z3 and
# pysat missing while all three were installed -- a probe-path artifact reported as a
# toolchain gap, which is exactly the kind of false negative this file exists to avoid.
VENVS = [
    REPO_ROOT / ".venv/glm52/bin/python",
]


def probe_module(name: str) -> dict:
    try:
        mod = __import__(name)
        return {"status": "INSTALLED", "version": getattr(mod, "__version__", "unknown"),
                "where": "current interpreter"}
    except ImportError:
        pass
    for py in VENVS:
        if not py.is_file():
            continue
        r = subprocess.run(
            [str(py), "-c",
             f"import {name} as m; print(getattr(m, '__version__', 'ok'))"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            return {"status": "INSTALLED", "version": r.stdout.strip(), "where": str(py)}
    return {"status": "NOT_INSTALLED", "version": None}


def probe_q0_clean_container() -> dict:
    """Report the actual Q0 authority separately from host conveniences.

    Q0 is defined by its offline, pinned clean-container replay.  Missing host
    solvers are useful operational gaps, but must not rewrite a hash-bound
    successful container replay as ``BLOCKED``.  This probe does not replay or
    trust a stale receipt: it requires the current replay leaf digest and the
    Q0 closure's binding to the current evidence-bundle seal.
    """
    replay = CONTAINER_ROOT / "REPLAY_RECEIPT.json"
    bundle = AUDITS_ROOT / "RAMANUJAN_Q0_EVIDENCE_BUNDLE.json"
    closure = AUDITS_ROOT / "RAMANUJAN_Q0_CLOSURE.json"
    try:
        replay_doc = json.loads(replay.read_text(encoding="utf-8"))
        bundle_doc = json.loads(bundle.read_text(encoding="utf-8"))
        closure_doc = json.loads(closure.read_text(encoding="utf-8"))
        replay_hash = hashlib.sha256(replay.read_bytes()).hexdigest()
        expected_hash = bundle_doc["leaf_sha256"]["ramanujan/container/REPLAY_RECEIPT.json"]
        expected_bundle_seal = closure_doc["evidence_bundle"]["seal_sha256"]
        if (
            replay_doc.get("schema") == "hawking.ramanujan.clean_proof_replay_receipt.v1"
            and replay_doc.get("status") == "REPLAY_OK"
            and replay_doc.get("network") == "none"
            and replay_hash == expected_hash
            and bundle_doc.get("seal_sha256") == expected_bundle_seal
            and closure_doc.get("status") == "PROVEN"
        ):
            return {
                "status": "PROVEN_OFFLINE_HASH_BOUND",
                "why": "Current REPLAY_OK --network=none leaf matches the Q0 evidence bundle and closure seal.",
            }
        return {
            "status": "STALE_OR_UNBOUND",
            "why": "The container replay, evidence bundle, or closure no longer forms the current Q0 hash chain.",
        }
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "MISSING_OR_INVALID", "why": f"Q0 container authority unavailable: {exc}"}


def main() -> int:
    binaries = {n: {**probe_binary(n, a), "needed_for": why} for n, a, why in BINARIES}
    modules = {n: {**probe_module(n), "needed_for": why} for n, why in MODULES}

    missing = [n for n, d in binaries.items() if d["status"] != "INSTALLED"]
    missing += [n for n, d in modules.items() if d["status"] != "INSTALLED"]
    q0_container = probe_q0_clean_container()
    research_authorized = LimitRegistry().research_authorized()

    doc = {
        "schema": "hawking.ramanujan.toolchain_selftest.v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {"platform": sys.platform, "python": sys.version.split()[0]},
        "resource_mode_when_run": "LIGHT_ONLY",
        "binaries": binaries,
        "python_modules": modules,
        "missing": missing,
        "verdict": "TOOLCHAIN_INCOMPLETE" if missing else "TOOLCHAIN_READY",
        "q0_reproducibility": q0_container,
        "authority": {
            "RAMANUJAN_RESEARCH_AUTHORIZED": research_authorized,
            "fixture_execution_only": True,
            "note": "A host-tool probe is not Ramanujan research authorization or Math-Frozen evidence.",
        },
        "host_convenience_gap": {
            "status": "INCOMPLETE" if missing else "READY",
            "why": "Host solver coverage is operationally incomplete; it is not the Q0 clean-container authority.",
        },
        "honest_note": "This probes; it does not install. Installing Lean plus Mathlib is a "
        "multi-gigabyte download and a long CPU-heavy build, which is precisely what "
        "HAWKING_RESOURCE_MODE=LIGHT_ONLY forbids while MOP owns the machine.",
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{doc['verdict']}: {len(missing)} missing -> {', '.join(missing) or 'none'}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
