"""Byte totals and I/O time for the 48h Odyssey phase plan, recomputed live.

``receipts/runtime/ODYSSEY_48H_PLAN.md`` states tier sizes and per-phase I/O
time as a snapshot. A markdown table nobody can verify is a wish -- this
module recomputes the same totals from ``hcli.specimens.registry()`` (a fresh
stat-only walk of the live lake, never a hand-maintained list) so a test can
catch the doc going stale the moment the lake's real composition drifts from
what it claims.

Tier boundaries and the "top 3" deferral are the operator's 2026-09-05
policy (encoded here, not re-litigated): tiers are named by GiB (2**30),
matching how the operator's own figures were derived (`du`-style binary
units, confirmed by matching the given top-3 sizes to within 0.1 GiB).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hcli import specimens

GIB = 1024**3
TIB = 1024**4
SEQ_READ_BPS = 118_000_000  # 118 MB/s sequential, measured against this USB volume

# Tier upper bounds in GiB: A_tiny <=8, B_mid <=40, C_large <=80, D_giant >80.
TIERS = (("A_tiny", 8 * GIB), ("B_mid", 40 * GIB), ("C_large", 80 * GIB), ("D_giant", None))

# Operator decision 2026-09-05: the top 3 by size are deferred from
# execution-class Odyssey work (not dropped -- stay in the static census).
DEFERRED_GIANTS = frozenset({
    "moonshotai/Kimi-K3",
    "thinkingmachines/Inkling-Small",
    "windowsxp811203/Qwen3.8-Flash-Next-Abliterated",
})


def tier_of(size_bytes: int) -> str:
    for name, upper in TIERS:
        if upper is None or size_bytes <= upper:
            return name
    raise AssertionError("unreachable")  # D_giant's upper is None, always matches


def _bucket(rows: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {name: [] for name, _ in TIERS}
    for s in rows:
        buckets[tier_of(s["size_bytes"])].append(s)
    return buckets


def snapshot() -> dict:
    """Live totals: per-tier n/bytes, deferred giants, and the three phases.

    Raises if the lake is not mounted -- callers that need to tolerate that
    check ``hcli.specimens.registry()['mounted']`` themselves first.
    """
    reg = specimens.registry()
    if not reg["mounted"]:
        raise RuntimeError(f"ModelLake not mounted at {reg['lake']}")
    rows = reg["specimens"]
    buckets = _bucket(rows)

    tiers = {
        name: {"n": len(b), "bytes": sum(s["size_bytes"] for s in b)}
        for name, b in buckets.items()
    }

    deferred_rows = [s for s in buckets["D_giant"] if s["repo"] in DEFERRED_GIANTS]
    deferred_bytes = sum(s["size_bytes"] for s in deferred_rows)
    phase3_bytes = tiers["D_giant"]["bytes"] - deferred_bytes
    phase1_bytes = tiers["A_tiny"]["bytes"] + tiers["B_mid"]["bytes"]
    phase2_bytes = tiers["C_large"]["bytes"]

    def hours(b: int) -> float:
        return b / SEQ_READ_BPS / 3600

    return {
        "n_specimens": len(rows),
        "total_bytes": sum(s["size_bytes"] for s in rows),
        "tiers": tiers,
        "deferred_n": len(deferred_rows),
        "deferred_bytes": deferred_bytes,
        "phase1_n": tiers["A_tiny"]["n"] + tiers["B_mid"]["n"],
        "phase1_bytes": phase1_bytes,
        "phase1_hours": hours(phase1_bytes),
        "phase2_n": tiers["C_large"]["n"],
        "phase2_bytes": phase2_bytes,
        "phase2_hours": hours(phase2_bytes),
        "phase3_n": tiers["D_giant"]["n"] - len(deferred_rows),
        "phase3_bytes": phase3_bytes,
        "phase3_hours": hours(phase3_bytes),
        "total_hours": hours(sum(s["size_bytes"] for s in rows)),
    }


# --- gravity-gauntlet cost evidence (measured 2026-09-05) -----------------
# receipts/runtime/GRAVITY_GAUNTLET_COST.md turns these into a schedule.
# Two numbers below are measured straight from receipts/odyssey-i/O0*_GRAVITY_*.json
# (test_lake_phases.py::test_gravity_mix_evidence_constants_match_the_receipts
# recomputes them live and fails if they drift). A third real cost -- the
# mlx_lm.convert build/quantize step that runs before every probe -- has no
# timestamp anywhere in this repo's receipts or logs and is NOT estimated
# here; gauntlet_hours() reports it as None, not a guess.

# doctor.wall_s across the 108 completed gravity-mix probes this repo has
# (5 specimens, 7B-30B params, tiers A_tiny/B_mid): PROBE TIME ONLY, does not
# include the antecedent convert step.
GRAVITY_PROBE_WALL_S = {
    "n": 108, "min": 6.271, "median": 15.986,
    "mean": 16.362648148148146, "max": 45.579,
}

# mixes actually run per specimen in the one real gravity battery this repo
# has: O001=23 O003=23 O004=21 O005=18 O006=23 (grid over bits x group-size x
# attn/mlp- or expert-protection, plus 3 mixed-bit combos). None reached the
# sub-bit target -- best complete_bpw was 2.1918-4.6419 against a <=1.00 bar
# -- so this is a LOWER BOUND on what a target-seeking search costs, not a
# validated search-to-target size.
GRAVITY_BATTERY_MIXES = {"n_specimens": 5, "counts": (23, 23, 21, 18, 23), "mean": 21.6}


def gauntlet_hours(snap: dict | None = None) -> dict:
    """Per-tier gravity-gauntlet time: measured lake I/O + measured probe
    time x the one measured battery size. Excludes the unmeasured
    mlx_lm.convert build/quantize stage -- every tier's total here is a
    LOWER BOUND on the real schedule, not the real schedule.
    """
    snap = snap if snap is not None else snapshot()
    mixes = GRAVITY_BATTERY_MIXES["mean"]
    probe_s = GRAVITY_PROBE_WALL_S["mean"]

    def row(n: int, b: int) -> dict:
        io_h = b / SEQ_READ_BPS / 3600
        probe_h = n * mixes * probe_s / 3600
        return {
            "n": n,
            "bytes": b,
            "io_hours": io_h,
            "probe_hours_lower_bound": probe_h,
            "build_hours": None,  # UNKNOWN -- no receipt or log times mlx_lm.convert
            "known_hours_lower_bound": io_h + probe_h,
        }

    t = snap["tiers"]
    return {
        "A_tiny": row(t["A_tiny"]["n"], t["A_tiny"]["bytes"]),
        "B_mid": row(t["B_mid"]["n"], t["B_mid"]["bytes"]),
        "C_large": row(t["C_large"]["n"], t["C_large"]["bytes"]),
        "D_giant_minus_deferred": row(snap["phase3_n"], snap["phase3_bytes"]),
    }


def _self_check() -> None:
    fake = [
        {"repo": "a", "size_bytes": 1 * GIB},          # A_tiny
        {"repo": "b", "size_bytes": 20 * GIB},          # B_mid
        {"repo": "c", "size_bytes": 60 * GIB},          # C_large
        {"repo": "moonshotai/Kimi-K3", "size_bytes": 1454 * GIB},  # D_giant, deferred
        {"repo": "d", "size_bytes": 90 * GIB},          # D_giant, not deferred
    ]
    b = _bucket(fake)
    assert [len(b[n]) for n, _ in TIERS] == [1, 1, 1, 2], b
    assert tier_of(8 * GIB) == "A_tiny"       # boundary is inclusive
    assert tier_of(8 * GIB + 1) == "B_mid"
    print("lake_phases self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    elif "--gauntlet" in sys.argv:
        print(json.dumps(gauntlet_hours(), indent=2))
    else:
        print(json.dumps(snapshot(), indent=2))
