"""G033 regression pins.

The defect these exist for: `complete_ebpw` was computed from hardcoded per-organ
rates and published under a physical name, so adding 1,288,519,664 bytes to an
artifact left the reported density unmoved at 2.596957 while the real physical
figure was 2.980217.
"""
import json
from pathlib import Path

import pytest

import subprocess, sys

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools/odyssey/metric_audit.py"
CLEAN = Path("/Users/scammermike/noetic/CLEAN_REBUILD_A/mix_hetero_n041_floors")

# THE SUITE RUNS THE TOOL. It used to read two checked-in receipts, and that was
# wrong in three separate ways, each proven by running:
#
#   1. Source mutations were INVISIBLE. Republishing a design constant under its
#      old physical name, and stripping the blocking gate off an unmeasurable
#      metric, both left every test passing -- because the tests only ever read a
#      JSON file the mutated code had not regenerated.
#   2. A missing receipt SILENTLY SKIPPED THE WHOLE MODULE. `12 passed` became
#      `12 skipped`, which in a large suite reads as green.
#   3. The canary guard named METRIC_MUTATION_CANARIES.json while the tool emits
#      PHYSICAL_METRIC_CANARIES.json, so the canary assertions were pinned to a
#      stale artifact from an earlier code version.
#
# Running the tool costs seconds (the canaries mutate an APFS clone, not the real
# artifact) and is done ONCE per session.


@pytest.fixture(scope="session")
def _receipts(tmp_path_factory):
    if not CLEAN.is_dir():
        pytest.skip(f"the audited artifact is absent: {CLEAN}. This is a real missing "
                    f"precondition, not a receipt that was never built.")
    d = tmp_path_factory.mktemp("metric_audit")
    a_out, c_out = d / "audit.json", d / "canaries.json"
    r = subprocess.run([sys.executable, str(TOOL), "--emit-audit", str(a_out),
                        "--emit-canaries", str(c_out)],
                       capture_output=True, text=True, cwd=str(REPO))
    assert a_out.is_file() and c_out.is_file(), (
        f"metric_audit.py did not emit its receipts (exit {r.returncode}):\n"
        f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return json.load(open(a_out)), json.load(open(c_out))


@pytest.fixture
def audit(_receipts):
    return _receipts[0]


@pytest.fixture
def canary(_receipts):
    return _receipts[1]


def test_every_metric_is_classified_by_where_its_value_comes_from(audit, canary):
    a = audit
    for m in a["metrics"]:
        assert m["classification"].startswith(
            ("DESIGN_EXPECTED_", "ARTIFACT_PHYSICAL_", "RUNTIME_MEASURED_")), m
        assert m["source_of_value"].strip(), m


def test_all_three_classes_are_populated(audit, canary):
    """An audit that finds only one class is not an audit."""
    a = audit
    assert a["n_design_expected"] >= 1
    assert a["n_artifact_physical"] >= 3
    assert a["n_runtime_measured"] >= 3


def test_complete_ebpw_is_physical_not_design(audit, canary):
    a = audit
    row = next(m for m in a["metrics"] if m["metric"] == "complete_ebpw")
    assert row["classification"] == "ARTIFACT_PHYSICAL_complete_ebpw"
    assert "payload_bytes" in row["source_of_value"]


def test_no_design_constant_wears_a_PHYSICAL_or_RUNTIME_name(audit, canary):
    """THE LAW, replacing an assertion that named one specific frozen metric.

    The old test asserted `active_ebpw_per_token` IS frozen. G059 un-froze it --
    it is now derived from real bytes -- so the old test failed for the RIGHT
    reason, and pinning the law to one metric's name was the defect: it would
    have gone stale again on the next metric. This asserts the invariant
    instead, so it survives the list changing.
    """
    a = audit
    for m in a["metrics"]:
        if m["classification"].startswith("DESIGN_EXPECTED_"):
            assert m["metric"].startswith("DESIGN_EXPECTED_"), (
                f"{m['metric']} is a design constant published under a physical/runtime "
                f"name -- exactly the bug where adding 1,288,519,664 bytes to an artifact "
                f"did not move complete_ebpw")
    assert a["still_frozen_and_flagged"] == [], a["still_frozen_and_flagged"]


def test_every_design_metric_carries_a_GATE_or_a_MEASURED_TWIN(audit, canary):
    """A design constant is legitimate only if it says why it cannot be measured
    (blocking_gate) or points at the real measurement beside it (measured_twin).
    Without one of those it is a number with no accountability.

    dram_bytes_per_token is the live case: it carries a blocking_gate naming root
    powermetrics and MTLCounterSampleBuffer instrumentation, and NO twin, because
    nothing on this box can measure it. Renamed rather than fabricated."""
    a = audit
    for m in a["metrics"]:
        if not m["classification"].startswith("DESIGN_EXPECTED_"):
            continue
        assert m.get("blocking_gate") or m.get("measured_twin"), (
            f"{m['metric']} is a design constant with neither a blocking gate nor a "
            f"measured twin")
    assert a["design_metrics_without_gate_or_measured_twin"] == []


def test_five_canaries_all_executed_and_passed(audit, canary):
    c = canary
    names = {r["canary"][0] for r in c["canaries"]}
    assert names == {"A", "B", "C", "D", "E"}
    assert c["n"] == 5 and c["n_passed"] == 5
    assert c["pass"] is True


def test_canaries_A_and_B_are_exact_not_merely_directional(audit, canary):
    """'went up' is not enough. The delta must equal 8*bytes/params exactly."""
    c = canary
    for r in c["canaries"]:
        if r["canary"][0] in ("A", "B"):
            assert r["exact"] is True, r


def test_canary_C_proves_a_genome_edit_cannot_move_a_physical_number(audit, canary):
    c = canary
    r = next(x for x in c["canaries"] if x["canary"].startswith("C_"))
    assert r["before"] == r["after"]


def test_canary_D_surfaces_the_design_physical_mismatch(audit, canary):
    """This is the exact shape of the original defect."""
    c = canary
    r = next(x for x in c["canaries"] if x["canary"].startswith("D_"))
    assert r["physical_after"] > r["physical_before"]
    assert r["design_equals_original"] is True
    assert r["mismatch_surfaced"] is True


def test_canary_E_states_what_it_did_not_do(audit, canary):
    """E verifies the fallback channel exists; it does not induce a fallback."""
    c = canary
    r = next(x for x in c["canaries"] if x["canary"].startswith("E_"))
    assert r["not_a_python_literal"] is True
    assert "does not INDUCE" in r["honest_limitation"]


def test_canaries_left_the_clone_at_its_baseline(audit, canary):
    c = canary
    assert c["restore_check"]["restored_to_baseline"] is True
    assert c["restore_check"]["baseline_bytes"] == c["restore_check"]["final_bytes"]


def test_receipts_are_machine_generated(audit, canary):
    for r in (audit, canary):
        assert r["hand_authored"] is False
        assert r["generated_by"] == "tools/odyssey/metric_audit.py"
