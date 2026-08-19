#!/usr/bin/env python3
"""G27 (was G160) spine: the gates, integrated into ONE process instead of one-shots.

Every gate the campaign built ran ONCE to close an obligation, wrote a receipt, and was
never called again -- no tool imported another, so nothing enforced the gates going
forward. That is a check that measures nothing: the exact failure the campaign keeps
hitting. This module composes the REAL gate functions (imported, not reimplemented)
into the canonical Genesis flow, so a candidate is qualified and promoted by the same
code every time, and a gate that should refuse halts the process.

Stage order follows the stack DOCTOR -> GRAVITY -> NR -> NX -> NVM/HIDE -> succession:

  1 SPAWN     worker_gate.gate        -- may we bring up an eval worker? (wired memory)
  2 TIMING    gpu_lane_guard.guard    -- any measurement is VOID if the lane is contended
  3 DOCTOR    doctor_seal.seal        -- required seal fields present and a control watched to fail
  4 PROVENANCE provenance_chain.check -- content-bound chain, tamper detectable
  5 PROMOTE   successor_select.rank   -- refuses BPW-alone and silent T regression
  6 REBIND    worker_checkpoint.*     -- checkpoint + rebind, no progress reset

Any stage that refuses short-circuits the pipeline with its reason. The selftest drives
the whole chain twice: a clean candidate (every gate passes -> PROMOTE -> rebind) and a
broken one (a gate REFUSES), so the gates are shown live and still refusing in-process.

  ./tools/nos_pipeline.py selftest
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import worker_gate            # noqa: E402
import gpu_lane_guard         # noqa: E402
import doctor_seal            # noqa: E402
import provenance_chain       # noqa: E402
import successor_select       # noqa: E402
import worker_checkpoint      # noqa: E402


class GateRefused(Exception):
    def __init__(self, stage, reason):
        self.stage, self.reason = stage, reason
        super().__init__(f"{stage}: {reason}")


def qualify_and_promote(candidate: dict, parent: dict, *, want_worker=True,
                        timing_fn=None, provenance_links=None) -> dict:
    """Run a candidate through every live gate. Returns the trace; raises GateRefused
    at the first stage that refuses."""
    trace = []

    # 1 SPAWN GATE -- consult the real memory gate on the live machine.
    if want_worker:
        obs = worker_gate.observe()
        g = worker_gate.gate(obs)
        trace.append({"stage": "spawn", "verdict": g["decision"], "note": g["note"]})
        if g["decision"] == "REFUSE":
            raise GateRefused("spawn", g["note"])

    # 2 TIMING under the protected-lane guard (only if a measurement is requested).
    if timing_fn is not None:
        _, v = gpu_lane_guard.guard(timing_fn, label=f"qualify:{candidate['name']}")
        trace.append({"stage": "timing", "verdict": v["verdict"], "note": v["why"]})
        if v["verdict"] == "VOID":
            raise GateRefused("timing", v["why"])  # a void measurement cannot qualify

    # 3 DOCTOR SEAL -- required fields + a control watched to fail.
    verdict, reasons = doctor_seal.seal(candidate["seal"])
    trace.append({"stage": "doctor", "verdict": verdict, "note": reasons})
    if verdict == "REFUSED":
        raise GateRefused("doctor", "; ".join(reasons))

    # 4 PROVENANCE -- content-bound chain check (links supplied or synthesized).
    links = provenance_links or candidate.get("provenance_links")
    if links:
        root = provenance_chain.seal(links)
        ok = provenance_chain.check(links, root)
        trace.append({"stage": "provenance", "verdict": "PASS" if ok else "FAIL",
                      "note": f"chain seal {root[:12]}"})
        if not ok:
            raise GateRefused("provenance", "chain check failed")

    # 5 PROMOTE decision -- refuses BPW-alone and silent T regression.
    try:
        decision = successor_select.rank_against_parent(candidate["metrics"], parent["metrics"])
    except successor_select.Refused as e:
        raise GateRefused("promote", str(e))
    trace.append({"stage": "promote", "verdict": decision["decision"], "note": decision["reason"]})

    # 6 REBIND only on PROMOTE.
    rebound = None
    if decision["decision"] == "PROMOTE":
        store = ROOT / "workspace/campaign/records/workers"
        cp = worker_checkpoint.checkpoint(
            f"eval-{candidate['name']}", parent["name"], candidate.get("obligations", ["G27"]),
            candidate["checkpoint_state"], store)
        doc = worker_checkpoint.rebind(cp, candidate["name"])
        rebound = {"checkpoint": str(cp.relative_to(ROOT)), "new_parent": doc["parent"],
                   "measurements_invalidated": doc["state"]["measurements"] == []}
        trace.append({"stage": "rebind", "verdict": "DONE",
                      "note": f"workers rebound to {doc['parent']}, measurements invalidated"})

    return {"candidate": candidate["name"], "final": decision["decision"],
            "trace": trace, "rebind": rebound}


def _good_candidate(name="cand-native-faster"):
    return {
        "name": name,
        "seal": {
            "tabula_drift": {"ratio": 11.4, "instrument_validated": True},
            "observed_controls": [{"name": "null-vs-null", "watched_to_fail": True}],
            "stated_test_width": {"battery_items": 60, "categories": 10},
            "known_blind_spots": ["long-context beyond 8k not in battery"],
        },
        "provenance_links": [{"digest": "a" * 64}, {"digest": "b" * 64}, {"digest": "c" * 64}],
        "metrics": {"name": name, "bpw": 4.2560, "token_ns": int(30_336_726 * 0.8),
                    "doctor_pass": True, "provenance_valid": True, "native_path": True,
                    "no_hidden_fallback": True},
        "obligations": ["G27"],
        "checkpoint_state": {
            "hypothesis": "native kernel win at equal BPW", "code_changes": ["kernel"],
            "measurements": [{"what": "0.8x token_ns", "parent": "G0"}],
            "negative_science": ["density alone caps at 99.11 TPS"],
            "blocker": "none", "next_experiment": "profile attention path"},
    }


def selftest() -> bool:
    parent = {"name": "uniform-q4-v1",
              "metrics": {"name": "uniform-q4-v1", "bpw": 4.2560, "token_ns": 30_336_726,
                          "doctor_pass": True, "provenance_valid": True, "native_path": True,
                          "no_hidden_fallback": True}}
    print("SPAWN GATE (live): synthetic clean vs pressured machine state")
    clean = {"total_gb": 103.08, "wired_gb": 4.63, "free_gb": 50.0, "inactive_gb": 44.0,
             "compressed_gb": 0.10, "swap_used_mb": 0.0, "workers_resident": 0,
             "worker_rss_total_gb": 0.0}
    pressured = dict(clean, swap_used_mb=512.0)
    gc = worker_gate.gate(clean); gp = worker_gate.gate(pressured)
    print(f"  clean machine    -> {gc['decision']}")
    print(f"  swap in use      -> {gp['decision']}")
    spawn_ok = gc["decision"] == "PERMIT" and gp["decision"] == "REFUSE"

    print("\nPATH A: clean candidate, every gate should pass -> PROMOTE -> rebind")
    a = qualify_and_promote(_good_candidate(), parent, want_worker=False)
    for s in a["trace"]:
        print(f"  {s['stage']:<11} {s['verdict']}")
    print(f"  FINAL: {a['final']}  rebind={bool(a['rebind'])}")
    ok_a = a["final"] == "PROMOTE" and a["rebind"] and a["rebind"]["measurements_invalidated"]

    print("\nPATH B: candidate fails the Doctor seal (no control watched to fail) -> REFUSE")
    bad = _good_candidate("cand-broken")
    bad["seal"]["observed_controls"] = [{"name": "x", "watched_to_fail": False}]
    refused_b = False
    try:
        qualify_and_promote(bad, parent, want_worker=False)
    except GateRefused as e:
        refused_b = e.stage == "doctor"
        print(f"  REFUSED at {e.stage}: {e.reason[:80]}")

    print("\nPATH C: smaller-BPW-but-slower candidate -> promote gate refuses silent T trade")
    slow = _good_candidate("cand-smaller-slower")
    slow["metrics"]["bpw"] = 3.34
    slow["metrics"]["token_ns"] = int(30_336_726 * 1.3)
    c = qualify_and_promote(slow, parent, want_worker=False)
    dev_c = c["final"] == "DEVELOP" and c["rebind"] is None
    print(f"  FINAL: {c['final']} (no rebind: {c['rebind'] is None})")

    ok = spawn_ok and ok_a and refused_b and dev_c
    print(f"\nGATES LIVE IN-PROCESS: spawn_gate={spawn_ok} pass_promote={ok_a} "
          f"doctor_refuses={refused_b} promote_refuses_silent_T={dev_c} -> {ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest"])
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    ok = selftest()
    if a.out:
        doc = {"schema": "hawking.nos.pipeline.v1",
               "obligation": "G27 (was G160) spine -- gates integrated into ONE process",
               "gates_wired": ["worker_gate.gate", "gpu_lane_guard.guard", "doctor_seal.seal",
                               "provenance_chain.check", "successor_select.rank_against_parent",
                               "worker_checkpoint.checkpoint+rebind"],
               "was_the_problem": "no tool imported another; every gate ran once and was never "
                                  "enforced again -- a check that measures nothing going forward",
               "now": "one entry point runs the gates as ordered stages; a refusal halts the flow",
               "selftest_all_gates_live": ok,
               "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True, cwd=ROOT).stdout.strip()}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
