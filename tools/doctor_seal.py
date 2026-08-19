#!/usr/bin/env python3
"""G124: the Doctor sealer. No PASS without Tabula drift, controls, width, blind spots.

A seal is a claim that a candidate is still the patient. This campaign has recorded
enough ways that claim goes wrong that the sealer refuses on structure, before it
ever looks at a score:

  tabula_drift        the patient is abliterated and quantization puts the removed
                      direction back monotonically. A capability score cannot see
                      that, so a seal without it is not a seal.
  observed_controls   a control that was never WATCHED TO FAIL proves nothing. This
                      session alone found a battery scoring 0.00 on its reference, a
                      curvature rule degenerate on its grid, and an answer key that
                      marked the reference model wrong.
  stated_test_width   "10/10" without the width is unreadable. G046 and G048
                      measured a ten-item gate as too narrow to certify equivalence.
  known_blind_spots   an omission that is not declared reads as an absence of
                      problems. G041 already recorded this for the T axis.

Refusing on a missing field is cheap. Discovering afterwards that a PASS rested on
an unstated blind spot is not.

  ./tools/doctor_seal.py --self-test        # four refusals and one pass
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "receipts/ascent-2026-08-16"
REQUIRED = ["tabula_drift", "observed_controls", "stated_test_width", "known_blind_spots"]


def seal(candidate: dict):
    """Returns (verdict, reasons). PASS requires all four fields to be present AND
    substantive -- an empty list or a null is treated as absent, because a field
    filled with nothing is the same omission wearing a key."""
    missing = []
    for f in REQUIRED:
        v = candidate.get(f)
        if v is None or (isinstance(v, (list, dict, str)) and len(v) == 0):
            missing.append(f)
    if missing:
        return "REFUSED", [f"missing or empty required field: {f}" for f in missing]

    warn = []
    td = candidate["tabula_drift"]
    if not td.get("instrument_validated", False):
        warn.append("tabula_drift present but its instrument is NOT validated -- G123 recovers the "
                    "direction with a 200x null separation and its ladder misses the recorded "
                    "11-12/25-27/64-67 by a constant 2.5x. The drift number is PROVISIONAL.")
    oc = candidate["observed_controls"]
    if not any(c.get("watched_to_fail") for c in oc):
        return "REFUSED", ["observed_controls present but NONE was watched to fail; a control that "
                           "has never separated is decoration"]
    return ("PASS_WITH_WARNINGS" if warn else "PASS"), warn


def real_candidate():
    return {
        "candidate": "uniform-q4-v1",
        "tabula_drift": {
            "drift_x_vs_parent": 28.63, "codec": "q4_group64", "layer": 63,
            "direction_agreement_abs_cos": 0.9983, "null_abs_cos": 0.0049,
            "instrument_validated": False,
            "source": "receipts/ascent-2026-08-16/G123_TABULA_DRIFT.json"},
        "observed_controls": [
            {"control": "mixed-q4down-v1 known-bad on the six-dimension battery",
             "watched_to_fail": True,
             "how_it_failed": "truncated on 100% of items in every dimension -- it never closes a "
                              "think block, which a score-only battery would have reported as "
                              "0.00 correct rather than never terminates",
             "source": "G076"},
            {"control": "prose|prose same-class ceiling beside every cross-class overlap",
             "watched_to_fail": False, "source": "G079"}],
        "stated_test_width": {
            "capability_items": 53, "dimensions": 6, "tasks_scored_for_utility": 30,
            "vocabulary_coverage_fraction": 0.0012846,
            "note": "six dimensions scored separately and never averaged; utility measured as wall "
                    "time per VERIFIED task, not TPS"},
        "known_blind_spots": [
            "activation statistics are PROSE-fitted; a prose profile costs 11.07% mean and 31.28% "
            "worst against a matched profile on non-prose classes (G081/G083)",
            "vocabulary coverage is 0.128% -- nothing is known about the 99.87% of tokens never "
            "produced, and special tokens carry 1.6-2.3x the row error of ordinary ones (G090)",
            "no per-kernel timing exists, so 7.8 ms of the token -- 25.5% -- is unattributed "
            "(G108/G122)",
            "the Tabula ladder misses the recorded values by a constant 2.5x (G123)"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    cases, results = [], []
    full = real_candidate()
    for f in REQUIRED:
        c = json.loads(json.dumps(full)); c.pop(f)
        cases.append((f"without {f}", c))
    cases.append(("complete", full))
    print(f"{'case':<32}{'verdict':<22}reasons")
    for name, c in cases:
        v, why = seal(c)
        results.append({"case": name, "verdict": v, "reasons": why})
        print(f"{name:<32}{v:<22}{why[0][:70] if why else ''}")
        for extra in why[1:]:
            print(f"{'':<54}{extra[:70]}")
    refused = sum(1 for r in results if r["verdict"] == "REFUSED")
    print(f"\nrefusals: {refused}/{len(REQUIRED)} required fields")
    doc = {"schema": "hawking.nos.doctor_seal.v1",
           "obligation": "G124 -- Tabula drift is part of the seal; no PASS without it",
           "required_fields": REQUIRED, "self_test": results,
           "refusals": refused,
           "the_real_candidate_does_not_get_a_clean_PASS": (
               "uniform-q4-v1 with every field populated seals as PASS_WITH_WARNINGS, not PASS, "
               "because G123's Tabula instrument is not validated -- its ladder misses the recorded "
               "values by a constant 2.5x. The sealer says so instead of emitting a clean PASS on a "
               "provisional number."),
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, cwd=ROOT).stdout.strip()}
    if a.out:
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
