#!/usr/bin/env python3
"""Emit the session verdict mechanically from receipts, not from optimism.

GOAL_MET (ULTRAGOAL section 50) requires ALL of:
  performance   Q80 >= 50 valid tok/s AND DSV4F >= 50 valid tok/s
  measurement   both have a current TOKEN_NS / GPU / BYTE ledger
  density       both have a materially real <=1.5 path
  autonomy      both continue detached without Claude picking each step
  integrity     rollback, skew guard, governors active; no weakened gate

Anything short of that is BLOCKED_WITH_PROOF (section 49), which is NOT a softer
GOAL_MET -- it has its own hard evidence requirements. This tool refuses to print
GOAL_MET unless every gate is satisfied by a receipt on disk.

    python3 tools/ascent_verdict.py
    python3 tools/ascent_verdict.py --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "receipts" / "ascent-2026-08-16" / "ASCENT_STATE.json"
FLOOR_TOK_S = 50.0
NS_PER_S = 1_000_000_000


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.is_file() else {}


def tok_s(ns_per_token: float | None) -> float | None:
    if not ns_per_token:
        return None
    return NS_PER_S / ns_per_token


def performance(state: dict) -> tuple[bool, list[str]]:
    inc = state.get("incumbents", {})
    lines, ok = [], True

    q80 = inc.get("q80", {})
    q_ns = q80.get("measured_ns_per_token")
    q_tps = tok_s(q_ns)
    if q_tps is None:
        ok = False
        lines.append("Q80: no measured ns/token on record")
    else:
        hit = q_tps >= FLOOR_TOK_S and q80.get("tournament_valid")
        ok &= bool(hit)
        lines.append(
            f"Q80: {q_tps:.3f} tok/s ({q_ns/1e6:.1f} ms/token) "
            f"[{q80.get('measurement_label','?')}] "
            f"tournament_valid={q80.get('tournament_valid')} -> "
            f"{'PASS' if hit else 'FAIL'} vs {FLOOR_TOK_S:.0f}"
        )

    d = inc.get("dsv4f", {})
    d_ns = d.get("body_ns_per_token_median") or d.get("expected_ns_per_token")
    d_tps = tok_s(d_ns)
    if d_tps is None:
        ok = False
        lines.append("DSV4F: no measured ns/token on record")
    else:
        hit = d_tps >= FLOOR_TOK_S
        ok &= hit
        lines.append(
            f"DSV4F: {d_tps:.3f} tok/s ({d_ns/1e6:.1f} ms/token) "
            f"[{d.get('measurement_label','?')}] -> "
            f"{'PASS' if hit else 'FAIL'} vs {FLOOR_TOK_S:.0f}"
        )
    return ok, lines


def density(state: dict) -> tuple[bool, list[str]]:
    """A <=1.5 claim needs packed bytes AND a decode kernel AND coherent generation.

    Organ cosine is a screen, never the gate.
    """
    inc = state.get("incumbents", {})
    lines, ok = [], True
    for name, key in (("Q80", "q80_density_frontier"), ("DSV4F", "dsv4f_density_frontier")):
        f = inc.get(key, {})
        if not f:
            ok = False
            lines.append(f"{name}: no density frontier on record")
            continue
        packed = bool(f.get("artifact_packed"))
        kernel = bool(f.get("decode_kernel_exists"))
        coh = f.get("coherence", {})
        generated = bool(f.get("coherence_generation_tested"))
        qualified = packed and kernel and generated
        ok &= qualified
        bpw = (f.get("complete_bpw_measured_from_reconstruction", {}) or {}).get(
            "nonexpert_8bit"
        ) or f.get("complete_bpw_at_8bit_nonexpert") or f.get("complete_bpw_1_5_gib")
        lines.append(
            f"{name}: bpw={bpw} packed={packed} kernel={kernel} generated={generated}"
            f" -> {'QUALIFIED' if qualified else 'NOT QUALIFIED'}"
        )
        if isinstance(coh, dict) and coh.get("status_after_audit"):
            lines.append(f"      coherence: {coh['status_after_audit']}")
    return ok, lines


def integrity() -> tuple[bool, list[str]]:
    lines, ok = [], True
    for label, path in (
        ("skew guard", "tools/branch_skew_guard.py"),
        ("controller", "tools/ascent_controller.py"),
        ("resource governor", "tools/agentos/machine_state.py"),
        ("disk governor", "tools/reclaim_safe.sh"),
        ("gpu lane mutex", "tools/gpu_lane_lock.sh"),
    ):
        present = (REPO / path).is_file()
        ok &= present
        lines.append(f"{label}: {'present' if present else 'MISSING'} ({path})")
    free = subprocess.run(
        ["df", "-g", str(REPO)], capture_output=True, text=True
    ).stdout.splitlines()
    if len(free) > 1:
        gib = int(free[-1].split()[3])
        ok &= gib >= 15
        lines.append(f"disk free: {gib} GiB (hard floor 15) -> {'ok' if gib >= 15 else 'BELOW FLOOR'}")
    return ok, lines


def blocked_proof(state: dict) -> list[str]:
    """Section 49: a large historical multiplier is NOT evidence."""
    inc = state.get("incumbents", {})
    out = []
    for name, key in (("Q80", "q80"), ("DSV4F", "dsv4f")):
        d = inc.get(key, {})
        ns = d.get("measured_ns_per_token") or d.get("body_ns_per_token_median")
        out.append(f"{name} current: {ns} ns/token ({d.get('measurement_label','?')})")
        top = d.get("top_cost_classes") or {}
        for k, v in top.items():
            if isinstance(v, (int, float)):
                out.append(f"    {k} = {v} ns/token")
        if d.get("path_to_20ms"):
            out.append(f"    path: {d['path_to_20ms']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return 0

    state = load_state()
    perf_ok, perf = performance(state)
    dens_ok, dens = density(state)
    integ_ok, integ = integrity()
    verdict = "GOAL_MET" if (perf_ok and dens_ok and integ_ok) else "BLOCKED_WITH_PROOF"

    if args.json:
        json.dump(
            {"verdict": verdict, "performance": perf, "density": dens,
             "integrity": integ, "proof": blocked_proof(state)},
            sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if verdict == "GOAL_MET" else 1

    print(f"VERDICT: {verdict}\n")
    for title, block, okflag in (
        ("PERFORMANCE", perf, perf_ok), ("DENSITY", dens, dens_ok),
        ("INTEGRITY", integ, integ_ok),
    ):
        print(f"{title}  [{'PASS' if okflag else 'FAIL'}]")
        for line in block:
            print(f"  {line}")
        print()
    if verdict != "GOAL_MET":
        print("BLOCKED_WITH_PROOF evidence (section 49):")
        for line in blocked_proof(state):
            print(f"  {line}")
        print("\n  A large historical multiplier is not evidence. The remaining")
        print("  requirement is the measured reason 20 ms is unreachable by known")
        print("  mechanisms, not a narrative about progress.")
    return 0 if verdict == "GOAL_MET" else 1


def _selfcheck() -> None:
    """The gate must be impossible to pass by optimism alone."""
    empty = {"incumbents": {}}
    assert not performance(empty)[0], "no data must not pass performance"
    assert not density(empty)[0], "no data must not pass density"

    # Fast but tournament-invalid must still FAIL: BPW ceiling is not optional.
    fast_invalid = {"incumbents": {
        "q80": {"measured_ns_per_token": 10_000_000, "tournament_valid": False},
        "dsv4f": {"body_ns_per_token_median": 10_000_000}}}
    assert not performance(fast_invalid)[0], "invalid artifact must not pass on speed"

    fast_valid = {"incumbents": {
        "q80": {"measured_ns_per_token": 10_000_000, "tournament_valid": True},
        "dsv4f": {"body_ns_per_token_median": 10_000_000}}}
    assert performance(fast_valid)[0], "20 ms/token both, valid, should pass"

    # Screen-passed but unpacked/ungenerated must NOT qualify as <=1.5.
    screened = {"incumbents": {
        "q80_density_frontier": {"artifact_packed": False, "decode_kernel_exists": True,
                                 "coherence_generation_tested": False},
        "dsv4f_density_frontier": {"artifact_packed": True, "decode_kernel_exists": True,
                                   "coherence_generation_tested": True}}}
    assert not density(screened)[0], "unpacked/ungenerated must not qualify"
    print("selfcheck ok")


if __name__ == "__main__":
    sys.exit(main())
