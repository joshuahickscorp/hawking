#!/usr/bin/env python3
"""Protected test for the PerformanceLedger gate (directive §25).

    "Promotions without comparable measurements are forbidden."

The gate is only real if it refuses. Every check here was written by first
constructing the situation the gate must reject and confirming it rejects it,
AND by constructing the situation it must allow and confirming it allows it — a
gate that is always closed detects nothing, exactly like a gate that is always
open.

Runs with plain python3 or under pytest. No GPU, no model, no network.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("perfledger", HERE / "performance_ledger.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}: {detail}")
        FAILS.append(f"{name}: {detail}")


def complete_row(**over):
    row = {
        "generation": "gen-a",
        "model_identity": "/models/a.gguf",
        "runtime_build": "llama b9430",
        "source_sha": "deadbeef",
        "context": 8192,
        "decode_tps": 22.0,
        "complete_token_wall_ns": int(1e9 / 22.0),
        "workers_resident": 1,
        "workers_decoding": 1,
        "measurement_reps": 3,
        "measurement_spread_pct": 2.0,
    }
    row.update(over)
    return row


def test_refuses_when_candidate_lacks_measurements():
    inc = complete_row()
    cand = complete_row(generation="gen-b", model_identity="/models/b.gguf",
                        decode_tps=None, complete_token_wall_ns=None)
    d = pl.can_promote(cand, inc)
    check("refuses a candidate with no decode measurement", d["allowed"] is False, str(d))
    check("names the missing axis",
          any("decode_tps" in r for r in d.get("reasons", [])), str(d.get("reasons")))


def test_refuses_when_incumbent_lacks_measurements():
    inc = complete_row(measurement_reps=None, measurement_spread_pct=None)
    cand = complete_row(generation="gen-b", model_identity="/models/b.gguf", decode_tps=30.0,
                        complete_token_wall_ns=int(1e9 / 30.0))
    d = pl.can_promote(cand, inc)
    check("refuses when the INCUMBENT was never measured properly", d["allowed"] is False, str(d))


def test_refuses_incomparable_environments():
    inc = complete_row()
    cand = complete_row(generation="gen-b", model_identity="/models/b.gguf",
                        decode_tps=40.0, complete_token_wall_ns=int(1e9 / 40.0),
                        workers_decoding=4)
    d = pl.can_promote(cand, inc)
    check("refuses a 4-decoder candidate against a 1-decoder incumbent",
          d["allowed"] is False, str(d))
    check("says which axis made them incomparable",
          any("workers_decoding" in r for r in d.get("reasons", [])), str(d.get("reasons")))

    cand2 = complete_row(generation="gen-c", model_identity="/models/c.gguf",
                         decode_tps=40.0, complete_token_wall_ns=int(1e9 / 40.0),
                         runtime_build="llama b9999")
    d2 = pl.can_promote(cand2, inc)
    check("refuses across different runtime builds", d2["allowed"] is False, str(d2))


def test_allows_a_comparable_pair():
    """The gate must be capable of saying yes, or it is not a gate."""
    inc = complete_row()
    cand = complete_row(generation="gen-b", model_identity="/models/b.gguf",
                        decode_tps=33.0, complete_token_wall_ns=int(1e9 / 33.0))
    d = pl.can_promote(cand, inc)
    check("allows a properly measured, comparable candidate", d["allowed"] is True, str(d))
    check("reports the gain", d.get("decode_tps_gain_pct", 0) > 0, str(d))
    check("verdict is ELIGIBLE, not PROMOTED",
          str(d.get("verdict", "")).startswith("ELIGIBLE"), str(d.get("verdict")))
    check("says capability gates still apply", "Doctor" in str(d.get("note", "")), str(d.get("note")))


def test_gain_inside_the_spread_is_noise():
    inc = complete_row(measurement_spread_pct=8.0)
    cand = complete_row(generation="gen-b", model_identity="/models/b.gguf",
                        decode_tps=22.0 * 1.03,  # +3%, inside an 8% spread
                        complete_token_wall_ns=int(1e9 / (22.0 * 1.03)),
                        measurement_spread_pct=8.0)
    d = pl.can_promote(cand, inc)
    check("a +3% gain inside an 8% spread is refused as noise",
          d["allowed"] is True and d["gain_within_noise"] is True
          and str(d["verdict"]).startswith("REJECT"), str(d))


def test_slower_candidate_is_rejected():
    inc = complete_row()
    cand = complete_row(generation="gen-b", model_identity="/models/b.gguf",
                        decode_tps=18.0, complete_token_wall_ns=int(1e9 / 18.0))
    d = pl.can_promote(cand, inc)
    check("a slower candidate is rejected",
          str(d.get("verdict", "")).startswith("REJECT"), str(d.get("verdict")))


def test_change_axis_exempts_only_the_declared_axis():
    """The axis mechanism must widen exactly one thing, not become a bypass."""
    inc = complete_row(runtime_build="llama b9430", quantization="Q5_K")
    cand = complete_row(generation="gen-b", model_identity="/models/b.mlx",
                        runtime_build="mlx", quantization="4bit",
                        decode_tps=35.5, complete_token_wall_ns=int(1e9 / 35.5))
    d_model = pl.can_promote(cand, inc, axis="model")
    check("axis=model still refuses a changed runtime", d_model["allowed"] is False, str(d_model))
    d_rt = pl.can_promote(cand, inc, axis="runtime")
    check("axis=runtime allows the runtime to differ", d_rt["allowed"] is True, str(d_rt))
    check("the decision records which axis was declared",
          d_rt.get("change_axis") == "runtime", str(d_rt.get("change_axis")))
    check("the decision records what was held equal",
          "workers_decoding" in (d_rt.get("held_equal") or []), str(d_rt.get("held_equal")))

    # a declared axis must NOT excuse a difference outside it
    cand_wide = complete_row(generation="gen-c", model_identity="/models/c.mlx",
                             runtime_build="mlx", workers_decoding=4,
                             decode_tps=40.0, complete_token_wall_ns=int(1e9 / 40.0))
    d_wide = pl.can_promote(cand_wide, inc, axis="runtime")
    check("axis=runtime does NOT excuse a concurrency difference",
          d_wide["allowed"] is False, str(d_wide))
    check("the refusal names the field and the axis",
          any("workers_decoding" in r and "axis=runtime" in r for r in d_wide.get("reasons", [])),
          str(d_wide.get("reasons")))

    d_bad = pl.can_promote(cand, inc, axis="everything")
    check("an unknown axis is refused rather than treated as permissive",
          d_bad["allowed"] is False, str(d_bad))


def test_ledger_is_append_only(tmpdir=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        orig = pl.LEDGER
        try:
            pl.LEDGER = Path(td) / "PERFORMANCE_LEDGER.jsonl"
            r = complete_row()
            first = pl.append(r)
            pl.append(r)                       # same content -> must not duplicate
            rows = pl.load()
            check("appending an identical row does not duplicate it", len(rows) == 1, str(rows))
            second = pl.append(complete_row(generation="gen-b", model_identity="/models/b.gguf"))
            check("a genuinely different row does append", len(pl.load()) == 2)
            check("ids are stable and distinct", first != second, f"{first} {second}")
            # the file must only ever have grown
            text = pl.LEDGER.read_text().splitlines()
            check("every row is one line of valid json",
                  all(l.strip().startswith("{") for l in text if l.strip()), str(text))
        finally:
            pl.LEDGER = orig


def main() -> int:
    for fn in sorted([f for n, f in globals().items() if n.startswith("test_")],
                     key=lambda f: f.__code__.co_firstlineno):
        fn()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        for f in FAILS:
            print("  " + f)
        return 1
    print("\nall performance-ledger gate checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
