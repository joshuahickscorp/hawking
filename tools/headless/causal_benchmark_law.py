#!/usr/bin/env python3
"""N012 — a benchmark a no-op would also pass is invalid.

S017 §37 makes this a law for every GPU optimization, and it is not theoretical.
The parent campaign's self-optimization harness compared two arms that were
executing IDENTICAL code, because `sys.path` ordering made the file being
inspected differ from the module being executed. Its REFUSED verdicts were
vacuous. Before that, an earlier benchmark fanned completions straight at a live
llama-server and never entered the mutated pool at all, so a no-op would have
scored the same.

So a GPU speed claim must carry proof that the thing it changed actually ran:

    kernel_hash        identity of the code that executed
    dispatch_count     the graph actually changed shape
    sentinel           something only the candidate path can produce
    noop_control       a candidate that changes nothing must NOT score
    bad_control        a deliberately worse candidate must be REJECTED

This harness AUDITS the GPU claims already on disk against that law and reports
which ones would survive it. It is deliberately applied retroactively: a law that
only governs future work leaves the existing claims unexamined.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
R = REPO / "receipts" / "headless"
OUT = R / "CAUSAL_BENCHMARK_LAW.json"

# GPU speed/shape claims made in this campaign and its parent.
CLAIMS = [
    ("NOETIC_DISPATCH_FUSION", "964 -> 756 dispatches, 33.717 -> 35.670 tok/s"),
    ("NOETIC_FUSED_SUBBIT", "2-bit MLP on the fused graph beats the incumbent on both axes"),
    ("AFFINE2_G64_LSFIT", "LS fit + g64 specialization, 26.84 -> 32.84 tok/s unfused"),
    ("AFFINE2_NATIVE_MLP", "native 2-bit MLP at 3.4574 EBPW"),
    ("NOETIC_Q3_MLP_Q4_ATTN", "q3 MLP + q4 attention at 3.6165 EBPW"),
    ("GPU_LEDGER", "incumbent is bandwidth-bound, 468.9 GB/s"),
    ("NOETIC_MULTISESSION", "one shared body; concurrency ceiling 1.32x"),
    ("NOETIC_PARENT_A", "sealed leader: 3.1393 EBPW, 756 dispatches"),
    ("NATIVE_2BIT_MLP", "FULL-MLP native 2-bit, fused SwiGLU, dense_w=0"),
]

# What counts as evidence for each requirement. Substring match on the receipt
# JSON is deliberately permissive: the point is to find claims with NO trace of a
# control at all, not to grade phrasing.
REQUIREMENTS: dict[str, dict[str, Any]] = {
    "kernel_identity": {
        "keys": ["kernel", "kernel_hash", "shader_evidence", "kernels", "metal_source_sha"],
        "why": "which code actually executed",
    },
    "dispatch_count": {
        "keys": ["dispatch", "dispatches_per_token", "dispatch_count"],
        "why": "the execution graph actually changed shape",
    },
    "sentinel": {
        # Pointer/identity evidence is a sentinel too. Omitting it produced a
        # FALSE NEGATIVE on NOETIC_MULTISESSION, which proves one shared body via
        # `weights_ptr_shared` and distinct per-session buffer pointers -- exactly
        # a fact only the candidate path can produce. An audit that misreports a
        # lane which did the work properly is worse than no audit.
        "keys": ["expanded_to_q4", "dense_w_materialized", "fallbacks", "census",
                 "n_affine", "counters", "weights_ptr_shared", "buffer_identities",
                 "one_body_not_n_copies", "closure_sha256"],
        "why": "something only the candidate path can produce",
    },
    "noop_control": {
        "keys": ["noop", "no_op", "no-op", "unfused", "before", "undegraded", "incumbent"],
        "why": "a candidate that changes nothing must not score",
    },
    "bad_control": {
        "keys": ["bad_control", "runtime_div", "degenerate", "zeroed", "deletion",
                 "scale_trap", "lost_to_incumbent"],
        "why": "a deliberately worse candidate must be rejected",
    },
}


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()[:12]


def audit(name: str, claim: str) -> dict[str, Any]:
    p = R / f"{name}.json"
    if not p.is_file():
        return {"receipt": name, "claim": claim, "state": "ABSENT",
                "reason": "receipt not on disk"}
    body = p.read_text().lower()
    have, missing = {}, []
    for req, spec in REQUIREMENTS.items():
        hit = next((k for k in spec["keys"] if k.lower() in body), None)
        have[req] = hit
        if hit is None:
            missing.append(req)
    return {
        "receipt": name,
        "claim": claim,
        "evidence_found": have,
        "missing": missing,
        "verdict": "SURVIVES_THE_LAW" if not missing else "INCOMPLETE_UNDER_THE_LAW",
        "would_a_noop_pass": bool("noop_control" in missing or "sentinel" in missing),
    }


def main() -> int:
    rows = [audit(n, c) for n, c in CLAIMS]
    survives = [r for r in rows if r.get("verdict") == "SURVIVES_THE_LAW"]
    incomplete = [r for r in rows if r.get("verdict") == "INCOMPLETE_UNDER_THE_LAW"]
    noop_risk = [r for r in rows if r.get("would_a_noop_pass")]

    receipt = {
        "schema": "hawking.headless.causal_benchmark_law.v1",
        "obligation": "N012 (S017 §37)",
        "git_head": git_head(),
        "law": (
            "A benchmark that would show the same win on a NO-OP is invalid. Every "
            "GPU optimization must prove the changed kernel or operator executed: "
            "kernel identity, dispatch count, a candidate-specific sentinel, a no-op "
            "control and an intentionally bad control."
        ),
        "why_this_is_not_theoretical": [
            "A self-optimization harness compared two arms running IDENTICAL code "
            "because sys.path ordering made the inspected file differ from the "
            "executed module (G021_SCRATCH_IMPORT_SHADOW). Its REFUSED verdicts were "
            "vacuous.",
            "An earlier benchmark fanned completions at a live llama-server and never "
            "entered the mutated pool, so a no-op would have scored identically.",
        ],
        "requirements": {k: v["why"] for k, v in REQUIREMENTS.items()},
        "audited": rows,
        "counts": {
            "claims": len(rows),
            "survives": len(survives),
            "incomplete": len(incomplete),
            "noop_would_pass": len(noop_risk),
        },
        "applied_retroactively": (
            "Deliberately. A law that governs only future work leaves every existing "
            "claim unexamined, and the existing claims are the ones the campaign is "
            "currently reasoning from."
        ),
        "limits": [
            "This audits for the PRESENCE of control evidence in a receipt, not for "
            "its correctness. A receipt can name a no-op control and still have run "
            "it wrongly.",
            "Substring matching is permissive on purpose: it finds claims with no "
            "trace of a control, not claims whose wording differs.",
        ],
    }
    OUT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"claims {len(rows)}  survives {len(survives)}  incomplete {len(incomplete)}")
    for r in rows:
        v = r.get("verdict", r.get("state"))
        miss = ",".join(r.get("missing", []))
        print(f"  {r['receipt']:<28} {v:<26} {miss}")
    print(f"receipt: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
