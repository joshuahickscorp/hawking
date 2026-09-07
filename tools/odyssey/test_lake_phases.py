"""ODYSSEY_48H_PLAN.md embeds a snapshot; this proves it hasn't gone stale.

The doc's numbers are checked against a *live* re-scan of the lake
(`lake_phases.snapshot()`), not against each other -- a plan document nobody
can verify is a wish. Skipped when the ModelLake volume isn't mounted, same
convention as hcli/test_specimens.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools.odyssey import lake_phases as lp

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "receipts" / "runtime" / "ODYSSEY_48H_PLAN.md"
REAL_SPECIMENS_DIR = Path("/Volumes/corpdrive/hawking-modellake/specimens")

TOLERANCE = 0.03  # 3% relative drift is normal lake churn; more means re-generate the doc


def _doc_snapshot() -> dict:
    text = PLAN.read_text()
    m = re.search(r"```json\n(\{.*?\})\n```", text, re.DOTALL)
    assert m, f"{PLAN} has no ```json snapshot block to check"
    return json.loads(m.group(1))


def _close(a: float, b: float, tol: float = TOLERANCE) -> bool:
    if a == b == 0:
        return True
    return abs(a - b) <= tol * max(abs(a), abs(b))


def test_tier_boundaries_are_inclusive_at_the_stated_gib():
    assert lp.tier_of(8 * lp.GIB) == "A_tiny"
    assert lp.tier_of(8 * lp.GIB + 1) == "B_mid"
    assert lp.tier_of(40 * lp.GIB) == "B_mid"
    assert lp.tier_of(40 * lp.GIB + 1) == "C_large"
    assert lp.tier_of(80 * lp.GIB) == "C_large"
    assert lp.tier_of(80 * lp.GIB + 1) == "D_giant"


@pytest.mark.skipif(not REAL_SPECIMENS_DIR.is_dir(), reason="ModelLake volume not mounted")
def test_plan_document_agrees_with_the_live_lake():
    doc = _doc_snapshot()
    live = lp.snapshot()

    assert doc["n_specimens"] == live["n_specimens"], (
        "specimen count in the doc no longer matches the lake -- regenerate "
        "ODYSSEY_48H_PLAN.md"
    )

    scalar_fields = [
        "total_bytes", "deferred_bytes",
        "phase1_bytes", "phase1_hours",
        "phase2_bytes", "phase2_hours",
        "phase3_bytes", "phase3_hours",
    ]
    for field in scalar_fields:
        assert _close(doc[field], live[field]), (
            f"{field}: doc says {doc[field]}, live lake says {live[field]} "
            f"(>{TOLERANCE:.0%} apart) -- regenerate ODYSSEY_48H_PLAN.md"
        )

    for tier in ("A_tiny", "B_mid", "C_large", "D_giant"):
        assert doc["tiers"][tier]["n"] == live["tiers"][tier]["n"], tier
        assert _close(doc["tiers"][tier]["bytes"], live["tiers"][tier]["bytes"]), tier


@pytest.mark.skipif(not REAL_SPECIMENS_DIR.is_dir(), reason="ModelLake volume not mounted")
def test_deferred_giants_are_excluded_from_phase3():
    live = lp.snapshot()
    assert live["deferred_n"] == 3
    assert live["phase3_n"] == live["tiers"]["D_giant"]["n"] - 3


# --- gravity-gauntlet cost (receipts/runtime/GRAVITY_GAUNTLET_COST.md) ----

GRAVITY_DIR = REPO / "receipts" / "odyssey-i"
GAUNTLET_DOC = REPO / "receipts" / "runtime" / "GRAVITY_GAUNTLET_COST.md"


def _live_gravity_mix_evidence() -> dict:
    """Recompute the probe-time and mixes-per-specimen evidence straight from
    the odyssey-i gravity receipts -- the same source lake_phases.py's
    GRAVITY_PROBE_WALL_S / GRAVITY_BATTERY_MIXES constants are built from, so
    a hand-edited constant that drifts from the receipts gets caught the same
    way test_plan_document_agrees_with_the_live_lake catches a stale lake
    snapshot.
    """
    from collections import defaultdict

    by_oxx: dict[str, list[float]] = defaultdict(list)
    for f in sorted(GRAVITY_DIR.glob("O0*_GRAVITY_*.json")):
        oxx = f.name.split("_GRAVITY_")[0]
        d = json.loads(f.read_text())
        w = d.get("doctor", {}).get("wall_s")
        if w is not None:
            by_oxx[oxx].append(w)
    probe_s = [w for ws in by_oxx.values() for w in ws]
    # only oxx codes with a real battery (>=15 mixes), not a one-off probe
    counts = [len(ws) for ws in by_oxx.values() if len(ws) >= 15]
    return {
        "probe_n": len(probe_s),
        "probe_min": min(probe_s),
        "probe_max": max(probe_s),
        "probe_mean": sum(probe_s) / len(probe_s),
        "mixes_n_specimens": len(counts),
        "mixes_mean": sum(counts) / len(counts),
    }


def test_gravity_mix_evidence_constants_match_the_receipts():
    live = _live_gravity_mix_evidence()
    assert lp.GRAVITY_PROBE_WALL_S["n"] == live["probe_n"]
    assert lp.GRAVITY_PROBE_WALL_S["min"] == live["probe_min"]
    assert lp.GRAVITY_PROBE_WALL_S["max"] == live["probe_max"]
    assert _close(lp.GRAVITY_PROBE_WALL_S["mean"], live["probe_mean"], tol=1e-9)
    assert lp.GRAVITY_BATTERY_MIXES["n_specimens"] == live["mixes_n_specimens"]
    assert _close(lp.GRAVITY_BATTERY_MIXES["mean"], live["mixes_mean"], tol=1e-9)


def test_gauntlet_hours_excludes_the_unmeasured_build_stage():
    """The mlx_lm.convert build/quantize step has no timestamp anywhere in
    this repo -- gauntlet_hours() must say so with None, never a guessed
    number standing in for a measurement (HARD RULE: UNKNOWN stays UNKNOWN).
    """
    g = lp.gauntlet_hours({
        "tiers": {
            "A_tiny": {"n": 21, "bytes": 73456681480},
            "B_mid": {"n": 19, "bytes": 337064183935},
            "C_large": {"n": 5, "bytes": 314263794536},
            "D_giant": {"n": 11, "bytes": 3987905281873},
        },
        "phase3_n": 8,
        "phase3_bytes": 1534938184222,
    })
    assert set(g) == {"A_tiny", "B_mid", "C_large", "D_giant_minus_deferred"}
    for name, row in g.items():
        assert row["build_hours"] is None, f"{name}: build time is unmeasured, must stay None"
        assert row["known_hours_lower_bound"] == pytest.approx(
            row["io_hours"] + row["probe_hours_lower_bound"]
        )
    # A_tiny: 21 specimens, mean battery of 21.6 mixes, mean probe 16.362648...s
    assert g["A_tiny"]["n"] == 21
    assert g["A_tiny"]["probe_hours_lower_bound"] == pytest.approx(
        21 * lp.GRAVITY_BATTERY_MIXES["mean"] * lp.GRAVITY_PROBE_WALL_S["mean"] / 3600
    )
    # D_giant row uses phase3 (minus the 3 deferred giants), not the raw tier
    assert g["D_giant_minus_deferred"]["n"] == 8


@pytest.mark.skipif(not REAL_SPECIMENS_DIR.is_dir(), reason="ModelLake volume not mounted")
def test_gauntlet_cost_doc_agrees_with_the_live_lake():
    assert GAUNTLET_DOC.is_file(), f"{GAUNTLET_DOC} does not exist"
    text = GAUNTLET_DOC.read_text()
    m = re.search(r"```json\n(\{.*?\})\n```", text, re.DOTALL)
    assert m, f"{GAUNTLET_DOC} has no ```json snapshot block to check"
    doc = json.loads(m.group(1))
    live = lp.gauntlet_hours()
    for tier in ("A_tiny", "B_mid", "C_large", "D_giant_minus_deferred"):
        assert doc[tier]["n"] == live[tier]["n"], tier
        assert _close(doc[tier]["known_hours_lower_bound"], live[tier]["known_hours_lower_bound"]), (
            tier, doc[tier], live[tier]
        )
