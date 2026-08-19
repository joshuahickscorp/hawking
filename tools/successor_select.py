#!/usr/bin/env python3
"""G151: successor selection as CODE, not prose.

The prime directives forbid two failure modes the campaign has already committed in
narration: promoting on BPW alone, and trading away execution speed T silently for a
smaller representation. This is the ranking function that mechanically refuses both,
with a unit test over CONSTRUCTED candidate vectors so the refusals are demonstrated,
not asserted.

Rules, in order:
  1. HARD GATE. A candidate that fails Doctor, lacks valid provenance, executes a
     hidden fallback, or has no native path is not rankable at all -- it is refused
     before any score is computed. No number buys past the gate.
  2. NO BPW-ALONE. Between two gated candidates, a smaller BPW never wins on its own;
     it must not regress T (token_ns) beyond a stated tolerance. A smaller-and-slower
     candidate is kept for development, never promoted over the parent.
  3. NO SILENT T TRADE. If a candidate improves representation (B down) but regresses
     execution (T up), the function REFUSES to rank it above the parent and says so.
     T may be traded only explicitly, by passing allow_t_regression.

  ./tools/successor_select.py --out receipts/.../G151_SUCCESSOR_SELECT.json
"""
from __future__ import annotations
import argparse, datetime, json, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]

HARD = ["doctor_pass", "provenance_valid", "native_path", "no_hidden_fallback"]


class Refused(Exception):
    pass


def gate(c: dict):
    missing = [k for k in HARD if not c.get(k)]
    if missing:
        raise Refused(f"{c['name']}: hard gate failed on {missing}")


def rank_against_parent(cand: dict, parent: dict, t_tol: float = 0.02,
                        allow_t_regression: bool = False) -> dict:
    """Decide promote / develop / refuse for cand vs the resident parent."""
    gate(cand)
    b_better = cand["bpw"] < parent["bpw"]
    # token_ns: lower is better. Regression = cand slower than parent beyond tolerance.
    t_regress = cand["token_ns"] > parent["token_ns"] * (1 + t_tol)
    t_better = cand["token_ns"] < parent["token_ns"] * (1 - t_tol)

    if t_regress and not allow_t_regression:
        return {"name": cand["name"], "decision": "DEVELOP",
                "reason": f"B {'down' if b_better else 'up'} but T regressed "
                          f"{cand['token_ns']/parent['token_ns']:.3f}x > tol {t_tol} -- "
                          "refusing to trade T silently; kept for development, not promoted"}
    if t_better:
        return {"name": cand["name"], "decision": "PROMOTE",
                "reason": f"native T win {parent['token_ns']/cand['token_ns']:.3f}x at "
                          f"{'lower' if b_better else 'equal-or-higher'} BPW"}
    if b_better and not t_regress:
        return {"name": cand["name"], "decision": "PROMOTE",
                "reason": "smaller BPW at no T regression (Pareto-dominates on B, ties on T)"}
    if b_better and t_regress and allow_t_regression:
        return {"name": cand["name"], "decision": "PROMOTE",
                "reason": f"smaller BPW with T regressed {cand['token_ns']/parent['token_ns']:.3f}x, "
                          "trade EXPLICITLY authorized via allow_t_regression"}
    return {"name": cand["name"], "decision": "DEVELOP",
            "reason": "neither a native T win nor a BPW gain without T cost"}


def unit_test() -> dict:
    parent = {"name": "G0", "bpw": 4.2560, "token_ns": 30_336_726,
              "doctor_pass": True, "provenance_valid": True, "native_path": True,
              "no_hidden_fallback": True}
    cases = []

    # A: smaller BPW, slower -> DEVELOP, never PROMOTE (the silent-T-trade refusal)
    a = dict(parent, name="A-smaller-slower", bpw=3.34, token_ns=int(30_336_726 * 1.3))
    ra = rank_against_parent(a, parent)
    cases.append(("smaller BPW but slower must NOT promote", ra["decision"] == "DEVELOP", ra))

    # B: same BPW, genuine native speed win -> PROMOTE
    b = dict(parent, name="B-native-faster", bpw=4.2560, token_ns=int(30_336_726 * 0.8))
    rb = rank_against_parent(b, parent)
    cases.append(("native T win must promote", rb["decision"] == "PROMOTE", rb))

    # C: smaller BPW, equal speed -> PROMOTE (Pareto on B, tie on T)
    c = dict(parent, name="C-smaller-equal", bpw=3.34, token_ns=30_336_726)
    rc = rank_against_parent(c, parent)
    cases.append(("smaller BPW at equal T may promote", rc["decision"] == "PROMOTE", rc))

    # D: dramatically smaller BPW, fails Doctor -> REFUSED at the gate (no number saves it)
    d = dict(parent, name="D-tiny-but-broken", bpw=1.0, token_ns=1, doctor_pass=False)
    refused = False
    try:
        rank_against_parent(d, parent)
    except Refused:
        refused = True
    cases.append(("tiny BPW that fails Doctor must be refused at the gate", refused, "REFUSED"))

    # E: smaller BPW, slower, but T regression EXPLICITLY allowed -> may promote
    e = dict(parent, name="E-explicit-trade", bpw=2.5, token_ns=int(30_336_726 * 1.3))
    re = rank_against_parent(e, parent, allow_t_regression=True)
    cases.append(("explicit T trade is allowed to promote", re["decision"] == "PROMOTE", re))

    passed = all(ok for _, ok, _ in cases)
    return {"passed": passed, "cases": [{"claim": c, "ok": ok,
                                         "result": r if isinstance(r, str) else r["decision"]}
                                        for c, ok, r in cases]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r = unit_test()
    for c in r["cases"]:
        print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['claim']} -> {c['result']}")
    print(f"unit test {'PASS' if r['passed'] else 'FAIL'}")
    doc = {
        "schema": "hawking.nos.successor_select.v1",
        "obligation": "G151 -- successor selection implemented as CODE, not prose",
        "started": start,
        "hard_gate": HARD,
        "rules": ["hard gate before any score", "no BPW-alone promotion",
                  "no silent T regression -- DEVELOP not PROMOTE unless explicitly allowed"],
        "unit_test": r,
        "refuses_bpw_alone": any(c["claim"].startswith("smaller BPW but slower") and c["ok"]
                                 for c in r["cases"]),
        "refuses_silent_t_trade": r["cases"][0]["ok"],
        "gate_beats_any_number": r["cases"][3]["ok"],
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
        "ended": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")
    return r["passed"]


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
