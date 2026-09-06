from __future__ import annotations

import json

import pytest

from hcli.gravity_gauntlet import (
    BUDGET_EXHAUSTED,
    PROVEN_UNABLE,
    TARGET_HIT,
    GravityGauntlet,
    candidate_space,
)


def measured(*, complete: float, verdict: str = "CANDIDATE_PASS", full: bool = False, **extra):
    d = {
        "complete_bpw": complete,
        "accounting": {"complete_bpw": complete, "complete_bytes": 1000},
        "labels": {"complete_bpw": "MEASURED"},
        "_evidence": "MEASURED (complete executable accounting)",
        "verdict": verdict,
        "wall_s": 0.25,
        "utilization": {"validated_fields": ["prefill_wall_s", "decode_wall_s"]},
        "nr_release_verified": True,
    }
    if full:
        d.update({"capability_ok": True, "execution_complete": True, "verifier_independent": True, "magnitude_ratio": 1.0})
    d.update(extra)
    if "capability_ok" in extra:
        d["capability_ok"] = extra["capability_ok"]
    return d


def test_evidence_guides_second_candidate_and_budget_is_not_a_pass(tmp_path):
    cs = candidate_space("O003", ["q3-g32-experts", "q2-g32-experts", "q2-g64-experts"])
    seen = []

    def evaluate(c):
        seen.append(c.spec)
        return measured(complete={"q3-g32-experts": 3.9, "q2-g32-experts": 3.1}[c.spec])

    state = GravityGauntlet(tmp_path / "state.json", "O003", cs, budget=2).run(evaluate)
    assert state["terminal"]["disposition"] == BUDGET_EXHAUSTED
    assert seen == ["q3-g32-experts", "q2-g32-experts"]
    assert state["iterations"][1]["candidate"]["parent_id"] == state["iterations"][0]["candidate"]["id"]
    assert state["terminal"]["best_complete_ebpw"] == 3.1
    assert state["iterations"][0]["decision"]["next_candidate_id"] == state["iterations"][1]["candidate"]["id"]


def test_target_hit_requires_independent_complete_execution(tmp_path):
    cs = candidate_space("O003", ["q2-g32-experts"])
    state = GravityGauntlet(tmp_path / "state.json", "O003", cs, budget=1).run(
        lambda _c: measured(complete=0.8, full=True)
    )
    assert state["terminal"]["disposition"] == TARGET_HIT

    state2 = GravityGauntlet(tmp_path / "state2.json", "O003", cs, budget=1).run(
        lambda _c: measured(complete=0.8, full=False)
    )
    assert state2["terminal"]["disposition"] == BUDGET_EXHAUSTED


def test_magnitude_destroyed_negative_control_fails_even_with_direction(tmp_path):
    cs = candidate_space("O003", ["negative-control-0.01W"])
    state = GravityGauntlet(tmp_path / "state.json", "O003", cs, budget=1).run(
        lambda _c: measured(complete=0.01, full=True, magnitude_ratio=0.01, direction_similarity=0.999)
    )
    obs = state["iterations"][0]["observation"]
    assert obs["magnitude_adequacy"]["verdict"] == "REJECTED_MAGNITUDE_DESTROYED"
    assert state["terminal"]["disposition"] == BUDGET_EXHAUSTED


def test_resume_preserves_history_and_does_not_repeat(tmp_path):
    path = tmp_path / "state.json"
    cs = candidate_space("O003", ["q3-g32-experts", "q2-g32-experts"])
    first = GravityGauntlet(path, "O003", cs, budget=2).run(lambda _c: measured(complete=3.0), max_steps=1)
    assert first["terminal"] is None
    seen = []
    resumed = GravityGauntlet(path, "O003", cs, budget=2).run(lambda c: seen.append(c.spec) or measured(complete=2.0))
    assert seen == ["q2-g32-experts"]
    assert len(resumed["iterations"]) == 2
    json.loads(path.read_text())


def test_proven_unable_requires_a_real_bound(tmp_path):
    cs = candidate_space("O003", ["q3-g32-experts"])
    engine = GravityGauntlet(tmp_path / "state.json", "O003", cs, budget=1)
    try:
        engine.finalize_proven_unable({"proven": True})
    except ValueError:
        pass
    else:
        raise AssertionError("missing bound fields must refuse PROVEN_UNABLE")
    state = engine.finalize_proven_unable({
        "proven": True,
        "upper_bound_complete_ebpw": 2.0,
        "limiting_mechanism": "measured persistent routing metadata floor",
        "measured_evidence": ["receipt-a"],
        "assumptions": ["candidate family is closed"],
        "search_region": "q2-q4 group32/64",
        "reopen_condition": "new executable family or lower metadata floor",
    })
    assert state["terminal"]["disposition"] == PROVEN_UNABLE


def test_hcli_wrapper_is_the_mutation_boundary(tmp_path):
    from hcli import odyssey

    with pytest.raises(PermissionError):
        odyssey.gravity_gauntlet("O003", ["q3-g32-experts"], state_path=str(tmp_path / "no.json"), receipt_dir="receipts/odyssey-i")
    state = odyssey.gravity_gauntlet(
        "O003",
        ["q3-g32-experts", "q2-g32-experts"],
        budget=2,
        state_path=str(tmp_path / "state.json"),
        receipt_dir="receipts/odyssey-i",
        confirm=True,
    )
    assert state["terminal"]["disposition"] == BUDGET_EXHAUSTED


def test_unparseable_spec_is_not_given_a_fabricated_precision():
    """A spec with no q-form carries NO precision information. Returning 4 makes
    an unrunnable string indistinguishable from q4 and feeds a number nobody
    measured into the search order."""
    from hcli.gravity_gauntlet import _spec_bits, _spec_group

    assert _spec_bits("q2-g64-experts") == 2
    assert _spec_group("q2-g64-experts") == 64
    assert _spec_bits("outlier0.005-g128") is None
    assert _spec_bits("totally-unparseable") is None
    assert _spec_group("totally-unparseable") is None


def test_specs_without_precision_do_not_outrank_measured_ones():
    """Ordering must not interleave a no-precision class at a made-up bit depth."""
    from hcli.gravity_gauntlet import _candidate_priority, Candidate

    def c(spec):
        return Candidate(id=spec, specimen="O003", spec=spec, parent_id=None,
                         mutation="m", expected_effect="e")

    ranked = sorted(["q4-g64-experts", "q2-g64-experts", "outlier0.005-g128"],
                    key=lambda s: _candidate_priority(c(s), capability_signal=True))
    assert ranked[-1] == "outlier0.005-g128", ranked


def test_best_never_crowns_a_capability_losing_candidate(tmp_path):
    """A search whose headline 'best' is a model that cannot generate is
    advertising a broken artifact. Lowest EBPW is not best on its own."""
    cs = candidate_space("O003", ["q3-g64-experts", "q2-g128-experts"])
    scores = {"q3-g64-experts": (3.50, True), "q2-g128-experts": (2.40, False)}

    def evaluate(c):
        ebpw, cap = scores[c.spec]
        return measured(complete=ebpw, magnitude_ratio=1.0, capability_ok=cap,
                        verdict="CANDIDATE_PASS" if cap else "CAPABILITY_LOSS")

    state = GravityGauntlet(tmp_path / "s.json", "O003", cs, budget=2).run(evaluate)
    assert state["terminal"]["best_candidate_id"] == "O003-q3-g64-experts", state["terminal"]
    assert state["terminal"]["best_complete_ebpw"] == 3.50
