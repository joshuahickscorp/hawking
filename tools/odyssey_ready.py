#!/usr/bin/env python3
"""Odyssey I required graph: can the machine start the first campaign?

Each capability is READY only when a named module exists AND has a live caller
or entrypoint. Existence alone is not readiness -- this campaign has already
found four capabilities that were built, declared and structurally unreachable.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: capability -> (module that must exist, symbol that must be defined)
#: Paths were located by search, not guessed. A guessed path reports MISSING for
#: a capability the machine has, which is the same false negative that made the
#: first classifier call live code dead.
REQUIRED = {
    "enumerate_specimens":    ("tools/odyssey/inventory.py", None),
    "identify_architecture":  ("tools/odyssey/arch_recognizer.py", "recognize"),
    "inspect_tensors":        ("tools/headless/noetic_organ_census.py", None),
    "stream_from_modellake":  ("hcli/agentos/modellake_supervisor.py", None),
    "cheap_structural_probe": ("tools/odyssey/specimen_open.py", None),
    "gravity_experiments":    ("tools/gravity_verify_source.py", None),
    "capability_tests":       ("tools/odyssey/performance_qualification.py", None),
    "physical_benchmark":     ("tools/odyssey/runtime_authority.py", None),
    "persist_candidates":     ("tools/odyssey/contracts.py", None),
    "emit_receipts":          ("hcli/agentos/modellake_receipts.py", None),
    "derive_laws_scars":      ("tools/future/campaign_scars.py", None),
    "compare_candidates":     ("tools/odyssey/tournament.py", None),
    "schedule_followups":     ("tools/future/frontiers.py", None),
}


def callers(stem: str, self_path: str) -> int:
    out = subprocess.run(["git", "grep", "-l", "--fixed-strings", stem, "--", "."],
                         cwd=REPO, capture_output=True, text=True).stdout.split()
    return len([o for o in out
                if o != self_path
                and not o.startswith(("receipts/", "workspace/campaign/", "research/receipts/"))])


def classify(module: str, symbol: str | None) -> tuple[str, str]:
    p = REPO / module
    if not p.is_file():
        alt = list((REPO / Path(module).parent).glob(f"*{Path(module).stem.split('_')[0]}*.py")) \
            if (REPO / Path(module).parent).is_dir() else []
        if alt:
            return "PARTIAL", f"absent; nearest present: {alt[0].relative_to(REPO)}"
        return "MISSING", "module absent"
    n = callers(Path(module).stem, module)
    if symbol:
        try:
            if symbol not in p.read_text(encoding="utf-8", errors="replace"):
                return "PARTIAL", f"present but {symbol}() not defined"
        except OSError:
            return "PARTIAL", "unreadable"
    if n == 0:
        return "PARTIAL", "present but no live caller"
    return "READY", f"{n} callers"


def main() -> int:
    rows = {k: classify(m, s) for k, (m, s) in REQUIRED.items()}
    counts: dict[str, int] = {}
    for state, _ in rows.values():
        counts[state] = counts.get(state, 0) + 1
    width = max(len(k) for k in rows)
    for k, (state, why) in rows.items():
        print(f"  {k:<{width}}  {state:8}  {why}")
    print("\n  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    (REPO / ".hcli" / "odyssey_ready.json").write_text(
        json.dumps({k: {"state": v[0], "why": v[1], "module": REQUIRED[k][0]}
                    for k, v in rows.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
