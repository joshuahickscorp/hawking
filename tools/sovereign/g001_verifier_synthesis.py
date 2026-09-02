#!/usr/bin/env python3
"""G001 producer: goal compilation synthesises verifiers with no named test.

Compiles REAL obligations through ``GoalCompiler.compile`` -- none of which
names a test file -- and then RUNS each synthesised verifier three times inside
a hermetic scratch git repo:

  1. capability absent   -> must exit non-zero  (red before green)
  2. capability present  -> must exit zero      (the gate can be discharged)
  3. capability removed  -> must exit non-zero  (negative control)

Every recorded outcome is an exit code from a real subprocess. Nothing here
asserts a result it did not observe, and if any phase disagrees with the
expectation the receipt is NOT written: a red gate beats a false green.

    python3 tools/sovereign/g001_verifier_synthesis.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hcli.goal import GoalCompiler  # noqa: E402
from hcli.verifier_pipeline import command_is_admissible  # noqa: E402

RECEIPT = REPO / "receipts/sovereign/G001_verifier_synthesis.json"
COMMAND = "python3 tools/sovereign/g001_verifier_synthesis.py"

# A real goal. Every section is an obligation, and NOT ONE names a test file --
# which under the old `_verify_command` meant every one of them compiled to an
# empty verifier and could never reach VERIFIED.
GOAL = """# Sovereign resident hardening

## Atomic mission persistence
hcli/persist.py must expose `atomic_write` so a crash mid-save cannot corrupt
the mission file.

## Live swap probe
hcli/resident.py must provide a `SwapoutsProbe` class that reads the live
delta, because the boot high-water mark is not a live reading.

## Receipt source constant
hcli/ledger.py must define `RECEIPT_SOURCE` at module scope so every receipt
names the producer that wrote it.

## Reachability report
tools/reach.py must expose `report_unreachable()` and a `CallGraph` class, or
a capability with no call site reads as present when it is not.
"""

# The scratch repo BEFORE any capability exists. Each file is real Python, and
# none of it satisfies any of the four claims above.
BASELINE = {
    "hcli/persist.py": "MISSION_FILE = 'mission.json'\n\n\ndef save(doc):\n    return doc\n",
    "hcli/ledger.py": "def load(path):\n    return path\n",
    "hcli/resident.py": "def cycle():\n    return 0\n",
    "tools/reach.py": "def scan(root):\n    return []\n",
}

# What discharging each obligation actually looks like on disk, keyed by the
# obligation's compiled title.
CAPABILITY = {
    "Atomic mission persistence": {
        "hcli/persist.py": BASELINE["hcli/persist.py"]
        + "\n\ndef atomic_write(path, data):\n"
        "    tmp = str(path) + '.tmp'\n"
        "    open(tmp, 'w').write(data)\n"
        "    import os\n"
        "    os.replace(tmp, path)\n",
    },
    "Live swap probe": {
        "hcli/resident.py": BASELINE["hcli/resident.py"]
        + "\n\nclass SwapoutsProbe:\n"
        "    def delta(self, before, after):\n"
        "        return after - before\n",
    },
    "Receipt source constant": {
        "hcli/ledger.py": "RECEIPT_SOURCE = 'Ledger.run_verify'\n\n"
        + BASELINE["hcli/ledger.py"],
    },
    "Reachability report": {
        "tools/reach.py": BASELINE["tools/reach.py"]
        + "\n\nclass CallGraph:\n    edges = ()\n\n\n"
        "def report_unreachable(graph):\n    return []\n",
    },
}


def _write(root: Path, files: dict) -> None:
    for rel, body in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )


def _run(command: str, root: Path) -> int:
    proc = subprocess.run(
        command, shell=True, cwd=root, capture_output=True, text=True, timeout=120
    )
    return int(proc.returncode)


def _scratch(root: Path) -> None:
    """A real git work tree holding the pre-capability world."""
    _write(root, BASELINE)
    _git(root, "init", "-q", ".")
    _git(root, "config", "user.email", "g001@sovereign.local")
    _git(root, "config", "user.name", "g001")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "baseline without the claimed capabilities")


def main() -> int:
    compiled = GoalCompiler().compile(GOAL)
    obligations = compiled.get("obligations") or []
    sections = {
        s["title"]: "\n".join(s["body"]).strip()
        for s in GoalCompiler()._extract_sections(GOAL)
    }

    cases = []
    failures = []
    for ob in obligations:
        title = str(ob.get("text") or "").strip()
        verifier = str(ob.get("verify") or "").strip()
        patch = CAPABILITY.get(title)
        if patch is None:
            failures.append(f"{ob.get('id')}: no capability defined for {title!r}")
            continue
        if not verifier:
            failures.append(f"{ob.get('id')} {title!r}: compiler synthesised no verifier")
            continue
        admitted, why = command_is_admissible(verifier)
        if not admitted:
            failures.append(f"{ob.get('id')} {title!r}: verifier refused ({why})")
            continue

        root = Path(tempfile.mkdtemp(prefix="g001-"))
        try:
            _scratch(root)
            exit_absent = _run(verifier, root)
            _write(root, patch)
            _git(root, "add", "-A")
            exit_present = _run(verifier, root)
            _write(root, {rel: BASELINE[rel] for rel in patch})
            _git(root, "add", "-A")
            exit_removed = _run(verifier, root)
        finally:
            shutil.rmtree(root, ignore_errors=True)

        if not (exit_absent != 0 and exit_present == 0 and exit_removed != 0):
            failures.append(
                f"{ob.get('id')} {title!r}: absent={exit_absent} "
                f"present={exit_present} removed={exit_removed} "
                "(wanted non-zero / zero / non-zero)"
            )
            continue

        cases.append(
            {
                "obligation_id": ob.get("id"),
                "obligation_text": title,
                "goal_text": f"{title}\n{sections.get(title, '')}".strip(),
                "names_a_test_file": False,
                "verifier": verifier,
                "verifier_admissible": True,
                "exit_capability_absent": exit_absent,
                "exit_capability_present": exit_present,
                "exit_capability_removed": exit_removed,
                "red_before_green": exit_absent != 0 and exit_present == 0,
                "negative_control_failed": exit_removed != 0,
                "measured": True,
            }
        )

    if failures or not cases:
        for line in failures or ["compiler produced no obligations"]:
            print(f"G001 NOT DISCHARGED: {line}", file=sys.stderr)
        print("no receipt written; the gate stays red", file=sys.stderr)
        return 1

    receipt = {
        "schema": "sovereign.gate.G001.verifier_synthesis/1",
        "produced_by": "tools/sovereign/g001_verifier_synthesis.py",
        "produced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": COMMAND,
        "status": "completed",
        "goal_text_compiled": GOAL,
        "obligations_compiled": len(obligations),
        "cases_measured": len(cases),
        "distinct_verifiers": len({c["verifier"] for c in cases}),
        "how_measured": (
            "each synthesised verifier ran three times as a real subprocess in "
            "a fresh scratch git work tree: capability absent, capability "
            "present, capability removed. The recorded integers are the actual "
            "exit codes, not expectations."
        ),
        "evidence": [
            f"{c['obligation_id']} {c['obligation_text']!r}: exit "
            f"{c['exit_capability_absent']} absent -> "
            f"{c['exit_capability_present']} present -> "
            f"{c['exit_capability_removed']} removed"
            for c in cases
        ],
        "cases": cases,
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RECEIPT.relative_to(REPO)} ({len(cases)} measured cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
