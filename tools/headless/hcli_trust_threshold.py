#!/usr/bin/env python3
"""The 14 conditions that gate entry into self-supplement and self-optimization.

From steer S002. The point is not to accumulate green ticks: it is to answer one
question honestly — is this substrate trustworthy enough to let it start
changing itself?

Each condition resolves against a RECEIPT on disk, not against a memory of
having done the work, and a condition whose receipt is missing, stale or
negative reports as NOT MET with the reason. A condition that cannot be
evaluated is NOT met; "cannot tell" never counts as "yes".

    python3 tools/headless/hcli_trust_threshold.py
    python3 tools/headless/hcli_trust_threshold.py --json

Exit 0 only when all 14 are physically true.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = REPO_ROOT / "receipts" / "headless"


def _load(name: str) -> Optional[Dict[str, Any]]:
    p = RECEIPTS / name
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _passed(rec: Optional[Dict[str, Any]]) -> bool:
    if not rec:
        return False
    result = str(rec.get("result") or "").upper()
    if result:
        return result.startswith("PASS")
    # Harness-shaped receipts carry a failed list instead of a result string.
    if "failed" in rec:
        return not rec.get("failed")
    # Gate-suite receipts carry green/red/inconclusive lists.
    if "red" in rec or "green" in rec:
        return not rec.get("red")
    return False


def _cond(name: str, receipt: str, extra: Optional[Callable[[Dict[str, Any]], Tuple[bool, str]]] = None):
    def run() -> Dict[str, Any]:
        rec = _load(receipt)
        if rec is None:
            return {"met": False, "why": f"receipt {receipt} is missing or unparseable"}
        # The extra predicate is the authority when one is supplied: a receipt
        # can be well-formed and still not say what the condition needs.
        if extra is not None:
            ok2, why2 = extra(rec)
            return {"met": ok2, "why": why2}
        ok = _passed(rec)
        stage = rec.get("stage")
        why = f"{receipt}: result={rec.get('result') or rec.get('failed')}"
        if stage and not ok:
            why += f" (stage: {stage})"
        return {"met": ok, "why": why}

    return name, run


def _equilibrium_measured(rec: Dict[str, Any]) -> Tuple[bool, str]:
    n = (rec.get("finding") or {}).get("measured_equilibrium_active_decodes")
    ceiling = (rec.get("finding") or {}).get("ceiling_vs_single_decoder")
    return (
        bool(n),
        f"measured equilibrium {n} concurrent decodes at {ceiling}x a single decoder",
    )


def _max_equilibrium_measured(rec: Dict[str, Any]) -> Tuple[bool, str]:
    """Met when a USEFUL equilibrium was derived, not merely a maximum reached."""
    n = rec.get("useful_equilibrium")
    rungs = rec.get("rungs") or []
    rates = [
        (r.get("requested_concurrency"), r.get("verified_units_per_hour")) for r in rungs
    ]
    if not n or len(rungs) < 2:
        return False, f"no equilibrium derived from {len(rungs)} rung(s)"
    # A single rung, or a rung set where nothing was rejected, has not located a knee.
    rejected = [r for r in rungs if r.get("rejected_as_equilibrium")]
    return (
        True,
        f"useful equilibrium c={n} from rungs {rates}; "
        f"{len(rejected)} rung(s) rejected for insufficient gain",
    )


def _no_red_gates(rec: Dict[str, Any]) -> Tuple[bool, str]:
    red = rec.get("red") or []
    incon = rec.get("inconclusive") or []
    return (
        not red,
        f"{len(rec.get('green') or [])} green, {len(red)} red, {len(incon)} inconclusive"
        + (f" (red: {red})" if red else ""),
    )


def _ingress_crossed(rec: Dict[str, Any]) -> Tuple[bool, str]:
    served = rec.get("prompt_tokens_server_counted")
    return (
        bool(served) and int(served) > 24000,
        f"{served} server-counted prompt tokens through the canonical path",
    )


CONDITIONS: List[Tuple[str, Callable[[], Dict[str, Any]]]] = [
    _cond("1. root context authority works",
          "HCLI_CONTEXT_AUTHORITY.json"),
    _cond("2. durable Goal/DAG/WorkUnit identity works",
          "HCLI_RESTART_RESUME.json"),
    _cond("3. dependency scheduling works",
          "MAX_DEPENDENCY_AND_ISOLATION.json"),
    _cond("4. bounded repair/retry behaviour exists",
          "REPAIR_HOMEOSTASIS.json"),
    _cond("5. verifier authority is non-vacuous",
          "AGENTOS_VERIFIER_AUTHORITY.json"),
    _cond("6. shared mutation is single-writer",
          "AGENTOS_SINGLE_WRITER.json"),
    _cond("7. Grok lifecycle is durable and reconcilable",
          "AGENTOS_GROK_WORKUNITS.json"),
    _cond("8. local Qwen execution is measured",
          "QWEN_MAX_EQUILIBRIUM.json", _equilibrium_measured),
    _cond("9. mixed backend failure is isolated",
          "MAX_DEPENDENCY_AND_ISOLATION.json"),
    _cond("10. checkpoint/restart is coherent",
          "HCLI_RESTART_RESUME.json"),
    _cond("11. steering is durable",
          "HCLI_RESTART_RESUME.json"),
    _cond("12. status exposes meaningful physical state",
          "HCLI_STATUS_OBSERVABILITY.json"),
    _cond("13. MAX has a measured useful equilibrium",
          "HCLI_MIXED_MAX.json", _max_equilibrium_measured),
    _cond("14. no known P0/P1 defect can explode work or manufacture VERIFIED state",
          "P0_GATES.json", _no_red_gates),
]

# Not one of the 14, but the single strongest end-to-end fact and worth carrying
# in the same receipt.
SUPPORTING = [_cond("long-context root ingress crossed 24K through the canonical path",
                    "HCLI_LONG_CONTEXT_ROOT_INGRESS.json", _ingress_crossed)]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default="receipts/headless/TRUST_THRESHOLD.json")
    args = ap.parse_args(argv)

    rows = []
    for name, run in CONDITIONS:
        rows.append(dict(condition=name, **run()))
    support = [dict(condition=n, **r()) for n, r in SUPPORTING]

    unmet = [r for r in rows if not r["met"]]
    report = {
        "gate": "TRUST_THRESHOLD",
        "source": "steer S002",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "conditions": rows,
        "supporting": support,
        "met": len(rows) - len(unmet),
        "total": len(rows),
        "unmet": [r["condition"] for r in unmet],
        "result": "PASS" if not unmet else "NOT_MET",
        "meaning": "PASS means the substrate is trustworthy enough to begin self-supplement and "
        "self-optimization. It does NOT mean the substrate is finished; anything beyond these 14 is "
        "input to the optimization loop rather than a prerequisite for it.",
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for r in rows:
            print(f"{'MET ' if r['met'] else 'NO  '} {r['condition']}")
            print(f"       {r['why']}")
        for r in support:
            print(f"{'  + ' if r['met'] else '  - '} {r['condition']}")
            print(f"       {r['why']}")
        print(f"\n{report['met']}/{report['total']} conditions met -> {report['result']}")
        if unmet:
            print("unmet: " + ", ".join(report["unmet"]))

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"receipt: {out}")
    return 0 if not unmet else 1


if __name__ == "__main__":
    raise SystemExit(main())
