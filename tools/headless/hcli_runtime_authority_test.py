#!/usr/bin/env python3
"""Runtime resource authority: is the genome an authority or a hardcoded number?

The failure this gate exists to prevent is the tempting one. The Qwen decode
ladder measured a useful equilibrium of 4, while `machine_genome.json` says
`active_decode_limit: 2`. The cheap "fix" is to open the genome and type 4.
That would replace a measured fact with an asserted one and leave nothing able
to tell the difference later -- a repeat of the campaign that hand-tuned a
constant and then measured against its own edit.

So the properties checked here are:

  1. the genome was NOT hand-edited to agree with the ladder
  2. a stale or foreign genome is DETECTABLE, not silently trusted
  3. the long-context ingress still works, so an authority change cannot
     quietly regress the 37586-token path that took real work to open
  4. the ~5% single-stream penalty from an oversized pool has a recorded
     disposition rather than being forgotten

    python3 tools/headless/hcli_runtime_authority_test.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
GENOME = Path.home() / ".config" / "hcli" / "machine_genome.json"

RESULTS: List[Dict[str, Any]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"{'ok  ' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")


def main() -> int:
    # ---- 1. the genome is not hand-edited --------------------------------
    genome = json.loads(GENOME.read_text()) if GENOME.is_file() else {}
    limit = genome.get("active_decode_limit")
    check(
        "the genome was NOT hand-edited to match the measured ladder",
        limit == 2,
        f"active_decode_limit={limit} (the ladder peaked at 4; the genome still "
        f"reports what its own probe measured with --max-runtimes 3, which "
        f"could not have observed 4)",
    )
    check(
        "the genome names the probe that produced it, so promotion has a path",
        "machine_probe" in str(genome.get("measured_by", "")),
        f"measured_by={genome.get('measured_by')}",
    )
    check(
        "the genome carries the receipt it came from",
        bool(genome.get("source_receipt")),
        f"source_receipt={genome.get('source_receipt')}",
    )

    # ---- 2. staleness is detectable --------------------------------------
    from hcli.machine import assess_genome_freshness

    fresh = assess_genome_freshness(genome)
    check(
        "a matching current genome assesses FRESH",
        getattr(fresh, "status", None) == "FRESH" and not getattr(fresh, "reasons", None),
        f"status={getattr(fresh, 'status', None)} reasons={getattr(fresh, 'reasons', None)}",
    )

    foreign = dict(genome)
    foreign["machine"] = dict(genome.get("machine") or {})
    foreign["machine"]["hw_model"] = "Mac00,0-not-this-box"
    stale_machine = assess_genome_freshness(foreign)
    check(
        "a genome from a DIFFERENT machine is refused, with a reason",
        getattr(stale_machine, "status", None) == "STALE"
        and bool(getattr(stale_machine, "reasons", None)),
        f"status={getattr(stale_machine, 'status', None)} "
        f"reasons={getattr(stale_machine, 'reasons', None)}",
    )

    old = dict(genome)
    old["generated_at"] = "2020-01-01T00:00:00Z"
    stale_age = assess_genome_freshness(old)
    check(
        "a genome older than the horizon is refused, with a reason",
        getattr(stale_age, "status", None) == "STALE"
        and any("horizon" in r for r in (getattr(stale_age, "reasons", None) or [])),
        f"status={getattr(stale_age, 'status', None)} "
        f"reasons={getattr(stale_age, 'reasons', None)}",
    )

    # ---- 3. the long-context ingress has not regressed --------------------
    # This is the anti-regression the obligation asks for. It is a real
    # subprocess with a real exit code, not an import check.
    ingress = REPO / "tools/headless/hcli_long_context_ingress.py"
    proc = subprocess.run(
        [sys.executable, str(ingress)],
        cwd=str(REPO), capture_output=True, text=True, timeout=900,
    )
    tail = (proc.stdout or proc.stderr or "").strip().splitlines()
    check(
        "the long-context ingress still passes through the canonical path",
        proc.returncode == 0,
        f"exit={proc.returncode}; {tail[-1] if tail else 'no output'}",
    )

    # ---- 4. the oversized-allocation penalty has a disposition ------------
    from hcli.runtime import slot_allocation_decision

    limit = int(genome.get("active_decode_limit") or 2)
    over = slot_allocation_decision(
        5, topology="slot", active_decode_limit=limit, requested_n=1)
    check(
        "allocating five slots for ONE decode is recorded at the decision, not silent",
        bool(over.get("oversized")) or bool(over.get("single_stream_cost")),
        f"oversized={over.get('oversized')} "
        f"single_stream_cost={over.get('single_stream_cost')}",
    )
    modest = slot_allocation_decision(
        1, topology="slot", active_decode_limit=limit, requested_n=1)
    check(
        "NEGATIVE CONTROL: a single slot for one decode is NOT flagged oversized",
        not modest.get("oversized"),
        f"oversized={modest.get('oversized')}",
    )

    failed = [r for r in RESULTS if not r["ok"]]
    receipt = {
        "gate": "HCLI_RUNTIME_AUTHORITY",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "genome_is_a_prior_not_a_constant":
            "The genome says active_decode_limit 2. The Qwen ladder measured a "
            "useful equilibrium of 4 with aggregate scaling 1.264. Those do not "
            "contradict each other: the genome was probed with --max-runtimes 3 "
            "and could not have observed 4. Promotion belongs to "
            "tools/headless/machine_probe.py re-run with a higher ceiling, NOT "
            "to an editor. This gate fails if anyone types the number in.",
        "single_stream_penalty": {
            "relative_loss": 0.0522,
            "basis": "5-slot c=1 22.489 tok/s vs 2-slot 23.727 tok/s",
            "kind": "PRIOR from receipts/headless/QWEN_MAX_EQUILIBRIUM*.json, "
                    "not a live measurement of any one spawn",
            "cause": "NOT_DETERMINED. What is ruled out: it is not decode "
                     "arithmetic, since the same weights and the same single "
                     "sequence are involved either way. The remaining candidates "
                     "are KV reservation per slot and scheduler overhead across "
                     "idle slots; distinguishing them needs a slot-count sweep "
                     "at fixed context, which is not run here.",
            "actionable": "pass requested_n=1 for a single deep decode; extra "
                          "resident slots are justified by warm state and prefix "
                          "locality, not by aggregate tok/s",
        },
        "results": RESULTS,
        "failed": [r["name"] for r in failed],
        "result": "PASS" if not failed else "FAIL",
    }
    out = REPO / "receipts/headless/RUNTIME_AUTHORITY.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    print(f"receipt: {out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
