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


def test_membership_no_longer_depends_ONLY_on_the_filename():
    """THIS TEST ASSERTED THE OPPOSITE AND WAS RIGHT TO, UNTIL IT WASN'T.

    It used to require `the corpus scope IS a filename prefix and says so`. That was
    true when written and is a TRANSIENT FACT ENCODED AS A LAW -- the same shape this
    program caught once before, when a test asserted unmeasured gaps outnumber
    measured ones and kept failing after the ledger moved past it.

    S032 §13 required membership to stop depending on what a file was named. It now
    has two routes and the build reports which one each member arrived by."""
    b = akb.build()
    m = b["membership_routes"]
    assert m["declared_count"] > 0, (
        "no receipt declares itself; membership is still purely a filename glob")
    # the four that were INVISIBLE are in, and they are in BY DECLARING
    for name in ("TOKEN_EXECUTION_ATLAS_COUNTS.json", "TOKEN_GRAPH_REDUCTION_TIMED.json",
                 "CAPABILITY_FUSED_GRAPH_CLEARED.json",
                 "FUSION_GAIN_IS_LENGTH_INDEPENDENT.json"):
        assert name in m["declared"], (name, m["declared"])
        assert name not in m["legacy_glob_only"], name


def test_an_INCOMPLETE_declaration_does_not_buy_membership():
    """A receipt that names its domain and nothing else would join the corpus while
    telling the reader nothing about what its evidence covers. All six scopes or
    none -- the same rule the AKB applies to an entry's eleven applicability axes."""
    import json, tempfile, pathlib as _p
    full = {"evidence_domain": "accelerator", "civilization": "I-D_ACCELERATOR",
            "program": "x", "machine_scope": "x", "representation_scope": "x",
            "kernel_scope": "x"}
    with tempfile.TemporaryDirectory() as td:
        f = _p.Path(td) / "r.json"
        f.write_text(json.dumps({"akb_registration": full}))
        assert akb.registration(f) is not None, "a complete declaration was refused"
        for drop in akb.REGISTRATION_KEYS:
            partial = {k: v for k, v in full.items() if k != drop}
            f.write_text(json.dumps({"akb_registration": partial}))
            assert akb.registration(f) is None, f"a declaration missing {drop!r} bought membership"
        # and a declaration for ANOTHER domain is not this lane's evidence
        f.write_text(json.dumps({"akb_registration": {**full, "evidence_domain": "q80"}}))
        assert akb.registration(f) is None, "another campaign's declaration bought membership"


def test_the_legacy_glob_route_is_REPORTED_so_it_can_shrink():
    """83 of 88 members still arrive by filename. That number is the size of the
    remaining name dependence; reporting it is what makes it shrinkable, and a
    silent one is a gap nobody can close."""
    m = akb.build()["membership_routes"]
    assert m["legacy_glob_only_count"] + m["declared_count"] == akb.build()["corpus_size"]
    assert m["legacy_glob_only_count"] > 0, (
        "if this ever hits zero the legacy route is dead and both it and this test "
        "should go -- that is a good failure, not a bad one")


def test_the_named_gap_list_cannot_go_stale_silently():
    """If a receipt named in the gap list is ever brought INTO the corpus -- by a
    rename or now by a declaration -- it must LEAVE the list rather than sit in it
    forever as a false alarm. A gap list that only grows is one nobody reads."""
    b = akb.build()
    inside = {p.name for p in akb.corpus()}
    assert not (set(b["known_accelerator_outside_scope"]) & inside)


def test_a_NONE_that_could_not_be_checked_is_REPORTED_not_silently_skipped():
    """The NONE grounding check skips a receipt with no identities block. Skipping
    SILENTLY is the check that cannot fail: an ungrounded NONE on MACHINE reads
    exactly like a grounded one, and NONE on MACHINE turns an M3 Ultra result into
    a universal. Found by mutating this lane's own newest law."""
    import json
    p = akb.RH / "ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json"
    if not p.is_file():
        return
    built = akb.build()
    assert built["none_claims_not_grounded_count"] == 0, (
        "a law claims NONE on an identity-backed axis that nothing checked: "
        f"{[n for e in built['entries'] for n in e.get('none_claims_not_grounded', [])]}")

    # ANTI-VACUITY. Zero is only a result if the counter can move.
    backup = p.read_text()
    doc = json.loads(backup)
    doc.pop("identities", None)
    p.write_text(json.dumps(doc, indent=2))
    try:
        broken = akb.build()
    finally:
        p.write_text(backup)
    assert broken["none_claims_not_grounded_count"] >= 1, (
        "stripping the identities block off a cited receipt did not raise the "
        "ungrounded-NONE count, so the counter reports nothing and the zero above "
        "means nothing")
    assert akb.build()["none_claims_not_grounded_count"] == 0, "restore failed"


def test_a_NONE_contradicted_by_a_PRESENT_identity_is_REFUSED():
    """The other direction. Reporting the unverifiable ones is worthless if a
    verifiable over-claim still passes."""
    import copy
    law = next(l for l in akb.LAWS
               if l["law_id"] == "AKB-DISPATCH-COUNT-DOES-NOT-PREDICT-COST")
    superseded = akb.superseding_corpus(akb.corpus())
    assert akb.validate(copy.deepcopy(law), superseded=superseded) is not None
    for axis in ("MACHINE", "RUNTIME", "MODEL", "KERNEL"):
        bad = copy.deepcopy(law)
        bad["applicability"][axis] = akb.NONE
        try:
            akb.validate(bad, superseded=superseded)
        except akb.Refused:
            continue
        raise AssertionError(f"{axis}=NONE survived though the receipt records it PRESENT")


def test_a_value_that_READS_as_a_sentinel_but_is_not_one_is_REFUSED():
    """'NONE -- the bound holds for any kernel' looks like NONE to a reviewer and
    is a named value to the grounding check, so it claims the breadth of a
    sentinel while escaping the rule that grounds one. Written by accident in this
    lane's own bandwidth-ceiling law and caught before it shipped."""
    import copy
    law = next(l for l in akb.LAWS
               if l["law_id"] == "AKB-BANDWIDTH-CEILING-BOUNDS-ACCEPTED-TPS")
    superseded = akb.superseding_corpus(akb.corpus())
    assert akb.validate(copy.deepcopy(law), superseded=superseded) is not None

    for axis, prose in [("KERNEL", "NONE -- the bound is over bytes, any kernel"),
                        ("SHAPE", "UNSCOPED across every length we tried"),
                        ("ORGAN", "UNKNOWN, nobody has looked at this")]:
        bad = copy.deepcopy(law)
        bad["applicability"][axis] = prose
        try:
            akb.validate(bad, superseded=superseded)
        except akb.Refused:
            continue
        raise AssertionError(f"{axis}={prose!r} survived; the sentinel guard is decoration")


def test_the_sentinel_guard_did_not_break_the_bare_sentinels():
    """Anti-vacuity partner. A guard that refused NONE itself would pass the test
    above and make every honest entry unwritable."""
    built = akb.build()
    used = {v for e in built["entries"] for v in e["applicability"].values()}
    assert akb.NONE in used, "no entry uses the bare NONE sentinel any more"
