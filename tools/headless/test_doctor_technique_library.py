"""N052: the hardened general Doctor library keeps techniques as general capabilities;
a Qwen failure is never a prune reason (S028)."""
import json, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parents[2] / "receipts" / "headless"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load():
    import doctor_technique_library as m
    m.main()
    return json.loads((R / "DOCTOR_TECHNIQUE_LIBRARY.json").read_text())


def test_generated_and_merged():
    d = _load()
    assert d["hand_authored"] is False
    assert d["n_techniques"] == len(d["techniques"])
    # merges N043 (15) + N046 recommended, deduplicated
    assert d["n_techniques"] >= 30


def test_every_technique_has_applicability_and_qwen_datapoint():
    d = _load()
    for tq in d["techniques"]:
        assert "applicability" in tq and tq["applicability"], tq["name"]
        assert "qwen_datapoint" in tq, tq["name"]


def test_no_prune_for_a_merely_qwen_reason():
    """The core S028 invariant: prune only for superseded / strictly-dominated /
    Metal-infeasible -- never because it failed on Qwen's MLP."""
    d = _load()
    for tq in d["techniques"]:
        dec = str(tq.get("decision") or tq.get("keep_or_prune") or "KEEP").upper()
        if dec.startswith("PRUNE"):
            reason = str(tq.get("prune_reason") or tq.get("reason") or "").lower()
            assert "qwen" not in reason and "failed on" not in reason, \
                f"{tq['name']} pruned for a Qwen reason: {reason}"


def test_general_not_qwen_only():
    d = _load()
    assert d.get("not_qwen_only")
    # paper mechanisms present as general capabilities
    names = {t["id"].lower() for t in d["techniques"]}
    assert any("spin" in n for n in names)  # SpinQuant (rotations)
    # our Qwen-failed campaign results are KEPT as scars for other architectures, not deleted
    scars = d.get("campaign_scars") or {}
    assert "binary" in scars and str(scars["binary"].get("decision", "")).upper().startswith("KEEP")
    assert "shared_basis" in scars and str(scars["shared_basis"].get("decision", "")).upper().startswith("KEEP")
