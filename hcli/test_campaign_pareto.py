"""Eleven-axis campaign frontier: partial candidates, live recompute, real receipts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "future"))
import campaign_pareto as C  # noqa: E402
from pareto_disposition import dominates as g033_dominates  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RECEIPTS = REPO / "receipts" / "future"
MULTIAXIS = RECEIPTS / "O003_PARETO_MULTIAXIS.json"
SRC = REPO / "tools" / "future" / "campaign_pareto.py"


def test_g033_six_axes_are_not_a_strict_subset_of_the_eleven():
    """G033 carries active_bytes_per_token, which is not a campaign axis."""
    assert "active_bytes_per_token" in C.G033_AXES
    assert "active_bytes_per_token" not in C.CAMPAIGN_AXES
    assert C.g033_is_strict_subset() is False
    for a in ("capability", "complete_ebpw", "tps", "prefill", "resident_memory",
              "temporary_memory", "state", "reliability", "role_suitability",
              "representation_complexity", "transfer_value"):
        assert a in C.CAMPAIGN_AXES


def test_epistemic_labels_are_six_distinct_states():
    assert C.EPISTEMIC == (
        "architecturally_supported", "implemented", "wired",
        "accepted", "verified", "physically_measured",
    )
    assert len(set(C.EPISTEMIC)) == 6


def test_g033_skip_none_is_the_defect_this_module_closes():
    """G033 lets a candidate that never measured TPS dominate a poor TPS.

    That is the 'quietly rewards not measuring' failure. Coverage-subset
    refuses it, and both points stay on the frontier.
    """
    missing = {"capability_passes": True, "complete_ebpw": 2.0}
    poor = {"capability_passes": True, "complete_ebpw": 3.0, "decode_tok_s": 10.0}
    assert g033_dominates(missing, poor), "G033 no longer exhibits the skip-None defect; update the pin"

    a = {"organism": "T", "capability": True, "complete_ebpw": 2.0}
    b = {"organism": "T", "capability": True, "complete_ebpw": 3.0, "tps": 10.0}
    assert not C.dominates(a, b)
    live = C.frontier([a, b])
    assert a in live and b in live


def test_a_missing_axis_is_not_a_zero_and_is_not_a_pass():
    # Treating missing TPS as 0 would let any positive TPS dominate it on that
    # axis. Coverage-subset does not invent a zero, so the better-ebpw partial
    # is not excluded by a worse-ebpw point that merely has a TPS number.
    missing = {"organism": "T", "complete_ebpw": 2.0}
    measured = {"organism": "T", "complete_ebpw": 3.0, "tps": 10.0}
    assert not C.has_value(missing, "tps")
    assert C.has_value(measured, "tps")
    assert 0 not in (missing.get("tps"),)
    assert not C.dominates(measured, missing)
    assert not C.dominates(missing, measured)


def test_a_partial_candidate_is_excluded_only_when_covered_and_worse():
    """Not silently dropped for being partial; dropped when a covering point is better."""
    partial = {"organism": "T", "complete_ebpw": 3.0}
    covering = {"organism": "T", "complete_ebpw": 2.0, "tps": 50.0}
    assert C.dominates(covering, partial)
    assert partial not in C.frontier([covering, partial])
    # the same partial, unique-best on its only axis, is RETAINED
    best_partial = {"organism": "T", "complete_ebpw": 1.0}
    other = {"organism": "T", "complete_ebpw": 2.0, "tps": 50.0}
    fr = C.frontier([best_partial, other])
    assert best_partial in fr and other in fr


def test_a_strictly_worse_arm_is_dominated():
    good = {"organism": "T", "capability": True, "complete_ebpw": 3.0, "tps": 130.0,
            "representation_complexity": 1, "transfer_value": 2}
    worse = dict(good, complete_ebpw=4.0, tps=120.0)
    assert C.dominates(good, worse) and not C.dominates(worse, good)
    assert C.frontier([good, worse]) == [good]


def test_a_tradeoff_is_retained_the_frontier_is_not_a_ranking():
    cheap_slow = {"organism": "T", "complete_ebpw": 2.0, "tps": 40.0}
    heavy_fast = {"organism": "T", "complete_ebpw": 8.0, "tps": 200.0}
    fr = C.frontier([cheap_slow, heavy_fast])
    assert cheap_slow in fr and heavy_fast in fr
    assert len(fr) == 2


def test_a_capability_failure_never_dominates_a_pass():
    failer = {"organism": "T", "capability": False, "complete_ebpw": 1.0,
              "representation_complexity": 0, "transfer_value": 9}
    passer = {"organism": "T", "capability": True, "complete_ebpw": 9.0,
              "representation_complexity": 9, "transfer_value": 0}
    assert not C.dominates(failer, passer)
    fr = C.frontier([failer, passer])
    assert failer in fr and passer in fr


def test_built_is_not_capability_preserving():
    """A representation being BUILT / Doctor-passing is not the capability gate."""
    for blob in (
        {"spec": "q2-g32-experts", "complete_ebpw": 3.06,
         "execution_complete": True, "capability_status": "CANDIDATE_PASS"},
        {"spec": "q2-g32-experts", "complete_ebpw": 3.06, "verdict": "CANDIDATE_PASS"},
        {"spec": "x", "complete_ebpw": 2.0, "ppl_passes": True},  # ppl alone
    ):
        cap, _ = C._capability_from(blob)
        assert cap is None, blob
    # both gate numbers: conjunction, not either.
    cap, epi = C._capability_from({"ppl": 3.5128, "median_4gram": 0.6667})
    assert cap is False and epi == "physically_measured"
    cap, _ = C._capability_from({"ppl": 3.1353, "median_4gram": 0.1075})
    assert cap is True


def test_different_organisms_never_dominate():
    dead = {"organism": "O003", "complete_ebpw": 0.87, "capability": False}
    other = {"organism": "microsoft--bitnet", "complete_ebpw": 3.908}
    assert not C.dominates(dead, other) and not C.dominates(other, dead)
    assert len(C.frontier([dead, other])) == 2


def test_false_and_zero_are_values_none_is_not():
    p = {"capability": False, "transfer_value": 0.0, "tps": None}
    assert C.has_value(p, "capability")
    assert C.has_value(p, "transfer_value")
    assert not C.has_value(p, "tps")
    assert not C.has_value(p, "prefill")


def test_source_recomputes_frontier_and_does_not_echo_a_stored_list():
    """A test that asserted against a stored frontier was vacuous here before."""
    src = SRC.read_text()
    assert "front = frontier(scored)" in src
    assert "return doc.get(\"frontier\")" not in src
    assert "P[\"frontier\"]" not in src
    # The only read of a receipt key named frontier is PARETO_MEASURED's
    # executed-config list, which is then fed through frontier(scored).
    assert "harvest_measured_execution" in src


def test_live_frontier_is_not_the_stored_multiaxis_list():
    stored = json.loads(MULTIAXIS.read_text())["frontier"]
    doc = C.harvest()
    assert doc["recomputed"] is True
    o003 = [p["name"] for p in doc["frontier"] if p["organism"] == "O003"]
    assert o003 != stored, (
        "live O003 frontier equalled the stored MULTIAXIS list; the tool is "
        "echoing a receipt rather than recomputing"
    )
    # Changing a live number must move the frontier: the function is of the
    # points, not of a stored list. a covers b and is cheaper, so b is out;
    # worsen a's ebpw past b and they become a tradeoff (b cannot cover TPS).
    pts = [{"organism": "U", "complete_ebpw": 1.0, "tps": 10.0, "name": "a"},
           {"organism": "U", "complete_ebpw": 2.0, "name": "b"}]
    assert [p["name"] for p in C.frontier(pts)] == ["a"]
    worsened = [dict(pts[0], complete_ebpw=4.0), pts[1]]
    names = {p["name"] for p in C.frontier(worsened)}
    assert names == {"a", "b"}


def _by_name(doc, organism, name):
    for p in doc["scored"]:
        if p["organism"] == organism and p["name"] == name:
            return p
    return None


def test_known_o003_points_are_harvested_from_receipts_not_the_contract():
    """Contract numbers enter only when a receipt confirms them."""
    doc = C.harvest()
    q3 = _by_name(doc, "O003", "q3-g128-experts")
    assert q3 is not None, "q3-g128-experts missing from harvest"
    ebpw = q3["values"]["complete_ebpw"]
    assert abs(ebpw - 3.2831) < 1e-3, ebpw
    assert q3["values"].get("capability") is True
    assert any("GAP_PROBE" in p or "NON_EXPERT_FLOOR" in p or "GRAVITY" in p
               for p in q3["provenance"])

    pq4 = _by_name(doc, "O003", "pqpercal4k512")
    assert pq4 is not None
    assert abs(pq4["values"]["complete_ebpw"] - 2.4059) < 1e-3
    assert pq4["values"].get("capability") is False  # likelihood good, r4 fails

    pq2 = _by_name(doc, "O003", "pqpercal2k16")
    assert pq2 is not None
    assert abs(pq2["values"]["complete_ebpw"] - 2.1864) < 1e-3
    assert pq2["values"].get("capability") is False  # diversity better, ppl fails

    dead = _by_name(doc, "O003", "binarypercal-g512")
    assert dead is not None
    assert abs(dead["values"]["complete_ebpw"] - 1.3639) < 1e-3
    assert dead["values"].get("capability") is False
    # The contract said ~1.35; the receipt is 1.3639. We do not round it to 1.35.


def test_gravity_tps_specimen_is_not_ingested_as_tps():
    doc = C.harvest()
    # Every Gravity receipt for O003 carried tps_specimen=51.627. That constant
    # is not a per-variant measurement.
    for p in doc["scored"]:
        if p["organism"] != "O003":
            continue
        tps = (p["values"] or {}).get("tps")
        if tps is None:
            continue
        assert abs(tps - 51.627) > 0.01, f"{p['name']} ingested Gravity tps_specimen"
        meta = (p["axis_meta"].get("tps") or {})
        assert meta.get("epistemic") == "physically_measured"


def test_real_receipts_produce_a_frontier_with_provenance():
    doc = C.harvest()
    assert doc["n_frontier"] >= 1
    assert doc["n_scored"] >= 10
    o003 = [p for p in doc["frontier"] if p["organism"] == "O003"]
    assert o003, "O003 contributed no frontier point"
    for p in o003:
        assert p["provenance"], p
        assert p["support"], p
        assert "values" in p
    # q3-g128-experts is the lowest capability-preserving point on disk; a
    # cheaper passer would have to beat 3.2831 with the conjunction gate.
    passers = [p for p in doc["scored"]
               if p["organism"] == "O003" and p["values"].get("capability") is True]
    assert passers
    lowest = min(passers, key=lambda p: p["values"]["complete_ebpw"])
    assert lowest["name"] == "q3-g128-experts"
    assert abs(lowest["values"]["complete_ebpw"] - 3.2831) < 1e-3


def test_measurement_debt_names_axes_with_no_values():
    doc = C.harvest()
    debt = doc["measurement_debt"]
    assert "unmeasured_for_everyone" in debt
    for a in debt["unmeasured_for_everyone"]:
        assert debt["n_with_axis"][a] == 0
        assert a in C.CAMPAIGN_AXES
    # `reliability` LEFT this list when G016 landed: reliability_axis.py derives
    # accepted_rate = is_accepted_work / n_runs from the engine receipts and
    # campaign_pareto harvests it onto serving identities. Asserting it is still
    # unmeasured would now pin the debt in place and turn a fix into a failure.
    # `state` left it too, when the G016 state axis landed: state_axis.py derives
    # per-token state cost from each body's own config and campaign_pareto harvests it.
    assert "role_suitability" in debt["unmeasured_for_everyone"], debt["n_with_axis"]
    # The axes that MOVED must stay measured. This is the regression guard: without it
    # either could silently fall back to zero coverage and only the shrinking list above
    # would notice, which is the "reward for not measuring" this frontier was corrected
    # to remove in the first place.
    for a in ("reliability", "state"):
        assert a not in debt["unmeasured_for_everyone"], debt["n_with_axis"]
        assert debt["n_with_axis"][a] > 0, debt["n_with_axis"]


def test_disposition_is_reused_not_forked():
    on = C.disposition({"on_frontier": True, "contributions": ["x"]})
    off = C.disposition({"on_frontier": False, "contributions": ["a transfer Law"]})
    assert on["action"] == "RETAIN" and on["earns_executable_synthesis"] is True
    assert off["action"] == "SEAL_AND_RELEASE" and off["earns_executable_synthesis"] is False


def test_partial_rule_is_coverage_subset():
    src = SRC.read_text()
    assert "support_b.issubset(support_a)" in src
    assert "PARTIAL_RULE" in src
    assert "not a zero" in C.PARTIAL_RULE
    assert "missing TPS cannot dominate" in C.PARTIAL_RULE


def test_every_axis_is_populated_or_explicitly_absent():
    """G011 acceptance: absent is a recorded OWED/REFUSED, not a dropped key."""
    doc = C.harvest()
    for p in doc["frontier"] + doc["scored"][:20]:
        meta = p["axis_meta"]
        assert set(meta) == set(C.CAMPAIGN_AXES), p["id"]
        for axis in C.CAMPAIGN_AXES:
            state = meta[axis]["state"]
            assert state in (C.MEASURED, C.REFUSED, C.OWED), (p["id"], axis, state)
            if state == C.MEASURED:
                assert axis in p["values"]
            else:
                assert axis not in p["values"]
                assert meta[axis].get("reason"), (p["id"], axis)


def test_disclosing_owed_axes_does_not_change_domination():
    a = {"organism": "T", "complete_ebpw": 2.0, "axis_meta": {}}
    b = {"organism": "T", "complete_ebpw": 3.0, "tps": 10.0, "axis_meta": {}}
    C.disclose_axes(a)
    C.disclose_axes(b)
    assert a["axis_meta"]["tps"]["state"] == C.OWED
    assert not C.has_value(a, "tps")
    assert not C.dominates(a, b)
    assert a in C.frontier([a, b]) and b in C.frontier([a, b])
