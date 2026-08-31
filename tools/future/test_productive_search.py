"""Launch count is not autonomy quality.

Run four launched 51 units, made 155 decisions and repeated no ask - a real
improvement over 4 launches and 94 identical questions. Then the launch list: 44
of the 51 were WU.HAWKING.health_probe.NNN. Fifty-one launches, about seven real
questions. The no-idle law cannot see that; it is busy, not productive.
"""
from __future__ import annotations

import pytest

from tools.future import productive_search as ps


def _u(uid, **kw):
    return {"id": uid, "frontier_id": kw.pop("frontier", "FT.X"), **kw}


def test_an_indexed_probe_series_is_one_causal_question():
    a = ps.causal_question_id(_u("WU.HAWKING.health_probe.007"))
    b = ps.causal_question_id(_u("WU.HAWKING.health_probe.008"))
    assert a == b, "a numeric suffix is not a new question"


def test_a_different_objective_is_a_different_question():
    a = ps.causal_question_id(_u("WU.HAWKING.health_probe.007"))
    b = ps.causal_question_id(_u("WU.EXEC.fold_addqx_ab_status"))
    assert a != b


def test_identity_uses_the_named_fields_not_a_prompt_hash():
    """Same id, different hypothesis and discriminator - two questions."""
    a = ps.causal_question_id(_u("WU.X.1", hypothesis_family="occupancy",
                                 verifier="sweep"))
    b = ps.causal_question_id(_u("WU.X.1", hypothesis_family="memory_level",
                                 verifier="unroll"))
    assert a != b


def test_a_unit_with_no_causal_identity_is_refused():
    with pytest.raises(ps.SearchRefused, match="no causal identity"):
        ps.causal_question_id({"id": "", "frontier_id": ""})


def test_the_run_four_window_is_degenerate():
    """51 launches, 44 health probes. This is the case the obligation names."""
    units = [_u(f"WU.HAWKING.health_probe.{i:03d}") for i in range(44)]
    units += [_u(x) for x in (
        "WU.HAWKING.resident_identity_pin", "WU.EXEC.fold_addqx_ab_status",
        "WU.HAWKING.fusion_env_applied", "WU.HAWKING.no_wait_scheduler",
        "WU.PROBE.decode_arith_cost", "WU.HYPB.005.encode",
        "WU.SUBAGENT.receipt_wait_probe",
    )]
    got = ps.classify_window(units)
    assert got["verdict"] == ps.DEGENERATE
    assert got["dominant_family"] == "WU.HAWKING.health_probe"
    assert got["dominant_family_share"] == pytest.approx(44 / 51, abs=1e-3)
    assert "86%" in " ".join(got["why"])


def test_four_decisive_experiments_across_four_uncertainties_are_productive():
    units = [_u(x) for x in ("WU.A.occupancy", "WU.B.memory_level",
                             "WU.C.alu_roofline", "WU.D.cache_knee")]
    assert ps.classify_window(units)["verdict"] == ps.PRODUCTIVE


def test_a_narrow_window_needs_several_questions_not_just_variety():
    """Two families is variety; it is not several uncertainties."""
    units = [_u("WU.A.1"), _u("WU.B.1")]
    assert ps.classify_window(units)["verdict"] == ps.DEGENERATE


def test_an_empty_window_is_refused_rather_than_passed():
    with pytest.raises(ps.SearchRefused, match="not evidence"):
        ps.classify_window([])


def test_the_first_health_probe_is_free_and_repeats_need_a_reason():
    units = [_u(f"WU.HAWKING.health_probe.{i}") for i in range(3)]
    b = ps.health_probe_budget(units)
    assert b["n_probes"] == 3 and b["n_unjustified_repeats"] == 2
    ok = ps.health_probe_budget(units, justifications={
        "WU.HAWKING.health_probe.1": "new_mutation",
        "WU.HAWKING.health_probe.2": "observed_anomaly",
    })
    assert ok["n_unjustified_repeats"] == 0


def test_an_invented_justification_is_not_accepted():
    units = [_u("WU.HAWKING.health_probe.0"), _u("WU.HAWKING.health_probe.1")]
    b = ps.health_probe_budget(units, justifications={
        "WU.HAWKING.health_probe.1": "it is cheap and safe",
    })
    assert b["n_unjustified_repeats"] == 1


def test_no_low_information_busy_loop_is_named_beside_no_runnable_idle():
    inv = ps.build()["invariants"]
    assert "NO_RUNNABLE_IDLE" in inv and "NO_LOW_INFORMATION_BUSY_LOOP" in inv
    assert "useful UNCERTAINTY" in inv["NO_LOW_INFORMATION_BUSY_LOOP"]


def test_the_receipt_does_not_claim_the_questions_were_the_right_ones():
    cb = ps.build()["claim_boundary"]
    assert "not whether they were the RIGHT ones" in cb or "RIGHT ones" in cb
