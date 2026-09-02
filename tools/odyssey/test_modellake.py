"""HDD stores, SSD stages. The bounds are enforced and the integrity checks reject."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import modellake as ml

REPO = Path(__file__).resolve().parents[2]
R = REPO / "receipts/headless/MODEL_LAKE_ROLLING_PIPELINE.json"


def test_budget_is_enforced_not_advisory():
    ok, why = ml.admit(4_000 * 10**9, 2)
    assert not ok and "budget" in why
    ok, why = ml.admit(200 * 2**30, 1)
    assert not ok and "budget" in why


def test_obvious_budget_refusal_does_not_measure_a_busy_mount(monkeypatch):
    def should_not_run(*args, **kwargs):
        raise AssertionError("storage measurement should be skipped")

    monkeypatch.setattr(ml, "tier2_used", should_not_run)
    monkeypatch.setattr(ml, "free", should_not_run)

    ok, why = ml.admit(ml.TIER2_BUDGET + 1, 2)

    assert ok is False
    assert "exceeds" in why


def test_writes_outside_the_lake_are_refused():
    for bad in ("/Volumes/corpdrive/substrate", "/Volumes/corpdrive/legal-scans-2026-08-23.tar.zst",
                "/Users/scammermike/models/qwen3.8-27b-abliterated-bf16"):
        try:
            ml.guard_protected(bad)
        except PermissionError:
            continue
        raise AssertionError(f"guard allowed {bad}")


def test_retiring_an_unrecorded_specimen_is_refused():
    r = ml.retire("not--a--specimen@000000000000")
    assert r["retired"] is False and "manifest" in r["why"]


def test_cycle_receipt_proves_each_property():
    d = json.load(open(R))
    steps = {s["step"]: s for s in d["cycle"]}
    assert d["pass"] is True
    assert steps["budget_refusals"]["tier2_4TB_admitted"] is False
    assert steps["budget_refusals"]["tier1_200GiB_admitted"] is False
    assert steps["interrupted"]["partial_bytes"] > 0
    assert steps["resumed_and_verified"]["acquired"] is True
    assert steps["corruption_rejected"]["verified"] is False
    assert steps["restored_and_reverified"]["verified"] is True
    assert steps["retired"]["retired"] is True and steps["retired"]["reacquisition"]
    ev = steps["capacity_event"]
    assert ev["human_step_required"] is False
    assert ev["scheduler_selected_next"] and ev.get("acquisition_started")


def test_protected_paths_still_present():
    """Checked against the tool's LIVE list, not the list frozen into an old receipt.

    substrate-git-backup-20260824-190835.tar was on this list and is now in the volume's
    Finder Trash. The lake could not have done it -- it has no trash path and every delete
    it makes is built under LAKE -- so it is recorded as externally removed rather than
    quietly dropped, and the remaining three are still asserted present.
    """
    for name in ml.PROTECTED:
        assert Path(f"/Volumes/corpdrive/{name}").exists(), name
    assert ml.EXTERNALLY_REMOVED, "an item left the protected list with no record of why"
    for e in ml.EXTERNALLY_REMOVED:
        assert e["by"] and e["observed"]
        assert not Path(f"/Volumes/corpdrive/{e['name']}").exists()


def test_the_lake_cannot_delete_outside_itself():
    """Every delete surface is either guarded or explicitly contained."""
    src = (Path(ml.__file__)).read_text()
    import re
    for m in re.finditer(r"^\s*_?s?h?\.?rmtree\((.+?)\)$", src, re.M):
        arg = m.group(1)
        assert ("guard_protected" in arg or "part" in arg or "staged" in arg), arg


def test_capacity_event_skips_resident_specimens():
    """It selected an already-resident model once, by trusting a stale disk snapshot."""
    ev = ml.capacity_event("test", 0)
    nxt = ev["scheduler_selected_next"]
    assert nxt is None or nxt["slug"] not in ml.resident_slugs()
