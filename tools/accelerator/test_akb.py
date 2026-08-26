#!/usr/bin/env python3
"""Negative controls for the Accelerator Knowledge Base validator.

The discipline this file is built around: a validator that has never refused
anything is assumed vacuous, and so is one that refuses everything. Every negative
control below therefore asserts BOTH halves --

    the unmutated entry VALIDATES, and the mutated one is REFUSED

-- because a validator stuck in the refuse position would pass a suite that only
ever checked for refusals. The paired assertion is the whole point; do not
"simplify" it away.

The second hazard this file guards is the one this repo has actually shipped three
times: a suite that passes while doing nothing. So the corpus tests assert a real
receipt count and a real law count rather than iterating a possibly-empty list, and
nothing here is allowed to skip.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import akb


# --------------------------------------------------------------------------- helpers

def law(law_id: str) -> dict:
    """A deep copy of one curated law, so a mutation cannot leak between tests."""
    for entry in akb.LAWS:
        if entry["law_id"] == law_id:
            return copy.deepcopy(entry)
    raise AssertionError(f"no law {law_id!r}; the negative controls are pinned to real laws")


def fake_root(tmp_path: Path, name: str, receipt: dict) -> Path:
    """A throwaway repo root holding one synthetic receipt."""
    (tmp_path / "receipts/headless").mkdir(parents=True, exist_ok=True)
    (tmp_path / "receipts/headless" / name).write_text(json.dumps(receipt))
    return tmp_path


MINIMAL_IDENTITIES = {k: {"status": "ABSENT", "reason": "synthetic fixture"}
                      for k in ("experiment", "machine", "device", "model",
                                "representation", "kernel", "runtime", "transport")}


# --------------------------------------------------------------------------- the corpus is real

def test_corpus_is_the_real_receipt_corpus_not_a_fixture():
    paths = akb.corpus()
    assert len(paths) >= 70, (
        f"only {len(paths)} ACCELERATOR receipts found. This suite grades the REAL corpus; "
        f"an empty or truncated glob must fail loudly, never skip.")
    assert all(p.exists() for p in paths)


def test_build_runs_over_the_real_corpus_and_yields_laws():
    built = akb.build()
    assert built["corpus_size"] >= 70
    assert len(built["entries"]) >= 15, "an empty AKB would otherwise pass every test below"
    assert built["receipts_yielding_laws"] >= 15


def test_every_entry_carries_all_eleven_axes():
    for entry in akb.build()["entries"]:
        assert set(entry["applicability"]) == set(akb.AXES), entry["law_id"]


def test_supersession_is_modelled_from_both_mechanisms():
    sup = akb.superseding_corpus()
    assert len(sup) >= 10, f"only {len(sup)} superseded receipts found in a corpus known to amend itself"
    flat = [c for cites in sup.values() for c in cites]
    assert any("AMEND" in c.upper() for c in flat), "the AMENDED_IN_PLACE mechanism did not fire"
    assert any("boundary_this_closes" in c for c in flat), "the boundary_this_closes mechanism did not fire"


def test_no_superseded_receipt_is_served_as_an_active_law():
    built = akb.build()
    sup = built["supersession_in_corpus"]
    for entry in akb.active(built):
        for rel in entry["source_receipts"]:
            assert Path(rel).name not in sup, f"{entry['law_id']} serves a superseded receipt as ACTIVE"


def test_refuted_and_negative_results_are_first_class():
    built = akb.build()
    assert built["entries_by_status"].get("REFUTED", 0) >= 1, "an AKB that holds only wins is a marketing document"
    assert built["negative_results"] >= 1


def test_every_unextracted_receipt_carries_a_reason():
    for item in akb.build()["unextracted"]:
        assert item["reason_code"] != "UNCLASSIFIED", item["receipt"]
        assert item["reason"], item["receipt"]


def test_corpus_is_partitioned_with_nothing_lost():
    built = akb.build()
    cited = {Path(r).name for e in built["entries"] for r in e["source_receipts"]}
    unext = {Path(u["receipt"]).name for u in built["unextracted"]}
    assert cited & unext == set(), "a receipt cannot be both extracted and unextracted"
    assert cited | unext == {p.name for p in akb.corpus()}, "a receipt fell out of the partition"


# --------------------------------------------------------------------------- NC1: UNSCOPED without breadth

def test_nc1_unscoped_shape_with_no_basis_is_refused():
    base = law("AKB-SCAN-VS-CUMSUM")
    akb.validate(copy.deepcopy(base))                      # the unmutated entry VALIDATES

    bad = copy.deepcopy(base)
    bad["applicability"]["SHAPE"] = akb.UNSCOPED
    with pytest.raises(akb.Refused) as exc:
        akb.validate(bad)
    assert "SHAPE is UNSCOPED with no unscoped_basis" in str(exc.value)


def test_nc1_unscoped_shape_whose_receipt_measured_one_shape_is_refused():
    base = law("AKB-SCAN-VS-CUMSUM")
    akb.validate(copy.deepcopy(base))

    bad = copy.deepcopy(base)
    bad["applicability"]["SHAPE"] = akb.UNSCOPED
    # a real field of the real source receipt -- but it holds ONE value, not a range
    bad["unscoped_basis"] = {
        "SHAPE": "receipts/headless/ACCELERATOR_SCAN.json#result.performance.16777216.gbps_2n.mlx_cumsum"}
    with pytest.raises(akb.Refused) as exc:
        akb.validate(bad)
    assert "shows only 1 value" in str(exc.value)


def test_nc1_a_genuinely_broad_axis_is_allowed():
    """The mirror of NC1: breadth that IS evidenced must pass, or the rule is just a ban."""
    akb.validate(law("AKB-DISPATCH-VS-SUBMISSION"))
    akb.validate(law("AKB-ORGAN-FLOOR-DOES-NOT-TRANSFER"))


# --------------------------------------------------------------------------- NC2: bare "X is faster"

@pytest.mark.parametrize("statement", [
    "The AIR scan is faster than mx.cumsum.",
    "Fusion is better than materialising intermediates.",
    "simdgroup matrix ops are slower in attention.",
    "tg256_ept2 is the fastest variant.",
])
def test_nc2_bare_present_tense_comparative_is_refused(statement):
    base = law("AKB-SCAN-VS-CUMSUM")
    akb.validate(copy.deepcopy(base))

    bad = copy.deepcopy(base)
    bad["statement"] = statement
    with pytest.raises(akb.Refused) as exc:
        akb.validate(bad)
    assert "bare present-tense comparative" in str(exc.value)


def test_nc2_every_shipped_statement_survives_its_own_rule():
    for entry in akb.LAWS:
        assert not akb.BARE_COMPARATIVE.search(entry["statement"]), entry["law_id"]


# --------------------------------------------------------------------------- NC3: citation does not resolve

def test_nc3_source_receipt_that_does_not_exist_is_refused():
    base = law("AKB-MACHINE-BANDWIDTH")
    akb.validate(copy.deepcopy(base))

    bad = copy.deepcopy(base)
    bad["source_receipts"] = ["receipts/headless/ACCELERATOR_NO_SUCH_RECEIPT.json"]
    with pytest.raises(akb.Refused) as exc:
        akb.validate(bad)
    assert "names a receipt that does not exist" in str(exc.value)


def test_nc3_citation_into_a_field_the_receipt_does_not_have_is_refused():
    base = law("AKB-MACHINE-BANDWIDTH")
    akb.validate(copy.deepcopy(base))

    bad = copy.deepcopy(base)
    bad["citations"] = ["receipts/headless/ACCELERATOR_MACHINE_GENOME.json#result.measured_bandwidth.median_tflops"]
    with pytest.raises(akb.Refused) as exc:
        akb.validate(bad)
    assert "does not resolve" in str(exc.value)


# --------------------------------------------------------------------------- NC4: superseded served as ACTIVE

def test_nc4_active_on_an_amended_receipt_is_refused():
    base = law("AKB-ORGAN-FLOOR-DOES-NOT-TRANSFER")
    assert base["status"] == "CONDITIONAL"
    akb.validate(copy.deepcopy(base))                      # CONDITIONAL is accepted

    bad = copy.deepcopy(base)
    bad["status"] = "ACTIVE"                               # the only change
    with pytest.raises(akb.Refused) as exc:
        akb.validate(bad)
    assert "is superseded by" in str(exc.value)
    assert "AMENDED_IN_PLACE_2026_08_25" in str(exc.value)


def test_nc4_active_on_a_receipt_closed_by_a_later_one_is_refused():
    base = law("AKB-SCAN-VS-CUMSUM")
    akb.validate(copy.deepcopy(base))

    bad = copy.deepcopy(base)
    # ACCELERATOR_RUNTIME_GATE.json is named by ACCELERATOR_RUNTIME_EXECUTES.json's
    # boundary_this_closes -- superseded by the OTHER mechanism, not by an AMEND key.
    bad["source_receipts"] = ["receipts/headless/ACCELERATOR_RUNTIME_GATE.json"]
    bad["citations"] = []
    with pytest.raises(akb.Refused) as exc:
        akb.validate(bad)
    assert "boundary_this_closes" in str(exc.value)


def test_nc4_superseded_status_needs_a_link():
    bad = law("AKB-SCAN-VS-CUMSUM")
    bad["status"] = "SUPERSEDED"
    bad["superseded_by"] = None
    with pytest.raises(akb.Refused) as exc:
        akb.validate(bad)
    assert "SUPERSEDED with no superseded_by" in str(exc.value)


# --------------------------------------------------------------------------- NC5: Measured on a failed run

def test_nc5_measured_on_a_receipt_that_did_not_pass_is_refused(tmp_path):
    """No receipt in the real corpus records pass: false, so this control is synthetic.

    That is stated rather than worked around: the rule still has to be watched firing,
    and a fixture is the only way to watch it on a corpus where every run passed.
    """
    good = {"schema": "hawking.accelerator.receipt.v1", "pass": True,
            "identities": MINIMAL_IDENTITIES, "result": {"n": 1}}
    failed = dict(good, **{"pass": False})
    root = fake_root(tmp_path, "ACCELERATOR_PASSED.json", good)
    fake_root(tmp_path, "ACCELERATOR_FAILED.json", failed)

    base = law("AKB-MACHINE-BANDWIDTH")
    base["applicability"]["MACHINE"] = akb.NONE       # the fixture records every identity ABSENT
    base["citations"] = []
    base["source_receipts"] = ["receipts/headless/ACCELERATOR_PASSED.json"]
    akb.validate(copy.deepcopy(base), superseded={}, root=root)   # passes on the passing receipt

    bad = copy.deepcopy(base)
    bad["source_receipts"] = ["receipts/headless/ACCELERATOR_FAILED.json"]
    with pytest.raises(akb.Refused) as exc:
        akb.validate(bad, superseded={}, root=root)
    assert "records pass: false" in str(exc.value)


# --------------------------------------------------------------------------- structural refusals

def test_a_missing_axis_is_refused():
    bad = law("AKB-SCAN-VS-CUMSUM")
    del bad["applicability"]["ORGAN"]
    with pytest.raises(akb.Refused) as exc:
        akb.validate(bad)
    assert "missing axes ['ORGAN']" in str(exc.value)


def test_an_invented_axis_is_refused():
    bad = law("AKB-SCAN-VS-CUMSUM")
    bad["applicability"]["VIBES"] = "good"
    with pytest.raises(akb.Refused) as exc:
        akb.validate(bad)
    assert "axes that do not exist" in str(exc.value)


def test_none_on_an_axis_the_receipt_actually_measured_is_refused():
    """NONE means the receipt recorded that identity ABSENT. It is not a way to skip an axis."""
    base = law("AKB-MACHINE-BANDWIDTH")
    akb.validate(copy.deepcopy(base))

    bad = copy.deepcopy(base)
    bad["applicability"]["MACHINE"] = akb.NONE   # the genome receipt very much has a machine
    with pytest.raises(akb.Refused) as exc:
        akb.validate(bad)
    assert "claims NONE but" in str(exc.value)


def test_an_unknown_evidence_class_is_refused():
    bad = law("AKB-SCAN-VS-CUMSUM")
    bad["evidence_class"] = "VibesBased"
    with pytest.raises(akb.Refused) as exc:
        akb.validate(bad)
    assert "is not an evidence class" in str(exc.value)


def test_a_missing_required_field_is_refused():
    bad = law("AKB-SCAN-VS-CUMSUM")
    del bad["confidence_basis"]
    with pytest.raises(akb.Refused) as exc:
        akb.validate(bad)
    assert "missing required field 'confidence_basis'" in str(exc.value)


# --------------------------------------------------------------------------- the validator is not stuck refusing

def test_every_shipped_law_validates():
    """The anti-vacuity mirror of every test above: a validator stuck in the refuse
    position would pass all of them and fail this one."""
    sup = akb.superseding_corpus()
    for entry in akb.LAWS:
        akb.validate(copy.deepcopy(entry), superseded=sup)
