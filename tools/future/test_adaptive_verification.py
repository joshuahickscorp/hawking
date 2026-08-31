"""Negative controls for adaptive multi-fidelity verification.

A guard nobody has watched fail is not a guard. The four required refusals:

  * a candidate killed at stage 1 launches zero later stages (assert the
    later stages were not called, not just that the verdict is negative)
  * a candidate that survives every stage is NOT reported as verified
  * a stage that cannot run (missing input) is not silently treated as passed
  * the ladder refuses a candidate the negative index already killed,
    before running ANY stage
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from tools.future import adaptive_verification as av
from tools.future import flash_schools as fs
from tools.future import meta_funnel as mf
from tools.future import negative_index as ni
from tools.future._common import (
    RECEIPTS,
    HardwareClaimError,
    _assert_no_hardware_claims,
)


def _clear(_proposal: dict) -> None:
    return None


def _plan():
    return av._plan()


def _spy_funnel() -> tuple[mf.Funnel, list[str]]:
    funnel = mf.Funnel()
    advanced: list[str] = []
    original = funnel.advance

    def wrapped(candidate, gate):
        g = mf.resolve_gate(gate)
        advanced.append(g.name)
        return original(candidate, gate)

    funnel.advance = wrapped  # type: ignore[method-assign]
    return funnel, advanced


def test_build_emits_sealed_receipt():
    out = av.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "ADAPTIVE_VERIFICATION.json"
    assert doc["schema"] == av.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert doc["seal_sha256"] == hashlib.sha256(blob).hexdigest()
    _assert_no_hardware_claims(doc)
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert doc["resident_callable"]["entry_point"]
    assert doc["resident_callable"]["receipt"] == f"receipts/future/{av.RECEIPT}"
    assert doc["resident_callable"]["frontier"] == "FT.VERIFICATION.repro"
    assert doc["flash_schools_cheapest_falsifier_field"] is True
    assert "cheapest_falsifier" in fs.CANDIDATE_FIELDS
    assert doc["counts"]["funnel_gates"] == len(mf.GATES) == 9
    assert doc["counts"]["proofs_held"] == 5
    for name, proof in doc["proofs"].items():
        assert proof["holds"] is True, name
    workunits = [u["workunit"] for u in doc["funnel_child_workunits"]]
    assert workunits == [av.funnel_workunit(g) for g in mf.GATES]


def test_selftest_aliases_build():
    out = av.selftest()
    assert out.name == av.RECEIPT


def test_module_parses():
    src = Path(av.__file__).read_text()
    ast.parse(src)
    for needle in ("raise NotImplementedError", "TODO", "\n    pass\n"):
        assert needle not in src


def test_ladder_is_cheapest_first_and_states_what_it_cannot_decide():
    rows = av.ladder(av._candidate("ladder.plain"))
    assert [r["name"] for r in rows] == [g.name for g in mf.GATES]
    assert [r["cost_rank"] for r in rows] == sorted(r["cost_rank"] for r in rows)
    for row, gate in zip(rows, mf.GATES):
        assert row["cannot_decide"] == gate.passing_does_not_prove
        assert row["can_decide"] == gate.passing_proves
        assert row["passing_proves"] == "not yet dead at this stage"
        assert row["origin"] == av.FUNNEL_ORIGIN
        assert row["workunit"] == av.funnel_workunit(gate)
        assert row["cost_class"] == gate.cost_class

    named = av.ladder(
        av._candidate("ladder.named", cheapest="Kill if the cheap observation fires.")
    )
    assert named[0]["name"] == av.FALSIFIER_NAME
    assert named[0]["cost_rank"] == 0
    assert named[0]["cost_class"] == av.FALSIFIER_COST_CLASS
    assert named[0]["cannot_decide"]
    assert [r["name"] for r in named[1:]] == [g.name for g in mf.GATES]
    ranks = [r["cost_rank"] for r in named]
    assert ranks == sorted(ranks)
    assert ranks[0] < ranks[1]


def test_saved_names_are_real_funnel_gates_not_an_invented_count():
    units = av.funnel_child_workunits()
    assert len(units) == len(mf.GATES) == 9
    for unit, gate in zip(units, mf.GATES):
        assert unit["gate_id"] == gate.id
        assert unit["gate_name"] == gate.name
        assert unit["cost_class"] == gate.cost_class
        assert unit["required_input"] == gate.required_input
        assert unit["origin"] == "tools/future/meta_funnel.py:GATES"
        assert unit["workunit"] == f"future.meta_funnel.gate.{gate.id}.{gate.name}"


def test_kill_at_stage_1_does_not_call_later_stages():
    """NEGATIVE CONTROL: later stages must not be invoked.

    A malformed allocation dies at analytical_structure_screen. Watching
    only the verdict would miss a bug that still ran teacher-fit.
    """
    called: list[str] = []
    funnel, advanced = _spy_funnel()
    cand = av._candidate(
        "neg.kill.gate1",
        allocation_plan={"unit": "TOTAL_EXECUTABLE_INFORMATION", "regions": []},
    )
    result = av.screen(
        cand,
        refuse_fn=_clear,
        funnel=funnel,
        on_stage=lambda s, _c: called.append(s.name),
    )
    later = [g.name for g in mf.GATES if g.id > 1]
    assert result["verdict"] == av.VERDICT_KILLED
    assert result["killed_by"] == "analytical_structure_screen"
    assert result["verified"] is False
    assert called == ["analytical_structure_screen"]
    assert advanced == ["analytical_structure_screen"]
    assert all(name not in called for name in later)
    assert all(name not in advanced for name in later)
    assert result["cost"]["later_stages_launched"] == 0
    assert result["cost"]["funnel_gates_launched"] == 1
    saved = av.saved(cand, screen_result=result)
    assert [u["gate_name"] for u in saved] == later
    assert [u["gate_id"] for u in saved] == [g.id for g in mf.GATES if g.id > 1]
    assert all(u["origin"] == av.FUNNEL_ORIGIN for u in saved)
    assert all("not_launched_because" in u for u in saved)


def test_kill_at_cheapest_falsifier_launches_zero_funnel_gates():
    """NEGATIVE CONTROL: the expensive child work is the nine funnel gates.

    A cheap observation that fires must not call Funnel.advance at all.
    """
    called: list[str] = []
    funnel, advanced = _spy_funnel()
    cand = av._candidate(
        "neg.kill.falsifier",
        cheapest="Kill if the cheap observation already fires.",
        observation={"fired": True, "mechanism": "cheap observation killed it"},
        **av._all_pass_inputs(),
    )
    result = av.screen(
        cand,
        refuse_fn=_clear,
        funnel=funnel,
        on_stage=lambda s, _c: called.append(s.name),
    )
    assert result["verdict"] == av.VERDICT_KILLED
    assert result["killed_by"] == av.FALSIFIER_NAME
    assert result["verified"] is False
    assert called == [av.FALSIFIER_NAME]
    assert advanced == []
    assert result["cost"]["funnel_gates_launched"] == 0
    assert result["cost"]["later_stages_launched"] == 0
    saved = result["saved"]
    assert [u["gate_name"] for u in saved] == [g.name for g in mf.GATES]
    assert [u["workunit"] for u in saved] == [av.funnel_workunit(g) for g in mf.GATES]
    assert av.saved(cand, screen_result=result) == saved


def test_survive_every_stage_is_not_verified():
    """NEGATIVE CONTROL: a full walk is not-yet-dead, never verified."""
    called: list[str] = []
    funnel, advanced = _spy_funnel()
    cand = av._candidate("neg.survive", **av._all_pass_inputs())
    result = av.screen(
        cand,
        refuse_fn=_clear,
        funnel=funnel,
        on_stage=lambda s, _c: called.append(s.name),
    )
    assert called == [g.name for g in mf.GATES]
    assert advanced == [g.name for g in mf.GATES]
    assert result["verdict"] == av.VERDICT_NOT_YET_DEAD
    assert result["verified"] is False
    assert result.get("status") not in {"VERIFIED", "PASSED_ALL", "VERIFIED_ALL"}
    assert result["killed_by"] is None
    assert result["saved"] == []
    blob = json.dumps(result, sort_keys=True)
    assert '"VERIFIED"' not in blob
    for row in result["stages_run"]:
        assert row["verdict"] == av.VERDICT_NOT_YET_DEAD
        assert row["proves"] == "not yet dead at this stage"
        assert row["does_not_prove"]
        assert row["passing_proves"] == "not yet dead at this stage"


def test_passing_cheapest_falsifier_is_not_evidence():
    """A cheap pass proves nothing except not-yet-dead, then the funnel runs."""
    called: list[str] = []
    cand = av._candidate(
        "neg.falsifier.pass",
        cheapest="Kill if the cheap observation fires.",
        observation={"fired": False},
        teacher_corpus="NOT_BUILT",
    )
    result = av.screen(
        cand,
        refuse_fn=_clear,
        on_stage=lambda s, _c: called.append(s.name),
    )
    assert called[0] == av.FALSIFIER_NAME
    assert result["stages_run"][0]["verdict"] == av.VERDICT_NOT_YET_DEAD
    assert result["stages_run"][0]["proves"] == "not yet dead at this stage"
    assert "capability" in result["stages_run"][0]["does_not_prove"].lower() or (
        "promotion" in result["stages_run"][0]["does_not_prove"].lower()
    )
    assert result["verdict"] == av.VERDICT_REFUSED
    assert result["verified"] is False
    assert result["refused_by"] == "real_teacher_fit"


def test_missing_input_is_not_silently_passed():
    """NEGATIVE CONTROL: absent teacher corpus must REFUSE, not PASS."""
    called: list[str] = []
    funnel, advanced = _spy_funnel()
    cand = av._candidate("neg.missing.teacher")
    result = av.screen(
        cand,
        refuse_fn=_clear,
        funnel=funnel,
        on_stage=lambda s, _c: called.append(s.name),
    )
    later = [g.name for g in mf.GATES if g.id > 2]
    assert result["verdict"] == av.VERDICT_REFUSED
    assert result["refused_by"] == "real_teacher_fit"
    assert result["verified"] is False
    assert called == ["analytical_structure_screen", "real_teacher_fit"]
    assert advanced == called
    assert all(name not in called for name in later)
    last = result["stages_run"][-1]
    assert last["raw_verdict"] == av.VERDICT_REFUSED
    assert last["verdict"] != "PASSED"
    assert last["input_state"] == "NOT_BUILT"
    assert "NOT_BUILT" in result["reason"]
    assert [u["gate_name"] for u in result["saved"]] == later


def test_named_falsifier_without_observation_is_not_a_pass():
    """A prose cheapest_falsifier with no observation cannot be skipped past."""
    called: list[str] = []
    funnel, advanced = _spy_funnel()
    cand = av._candidate(
        "neg.falsifier.absent",
        cheapest="Kill if the cheap observation fires.",
        **av._all_pass_inputs(),
    )
    result = av.screen(
        cand,
        refuse_fn=_clear,
        funnel=funnel,
        on_stage=lambda s, _c: called.append(s.name),
    )
    assert result["verdict"] == av.VERDICT_REFUSED
    assert result["refused_by"] == av.FALSIFIER_NAME
    assert result["verified"] is False
    assert called == [av.FALSIFIER_NAME]
    assert advanced == []
    assert result["cost"]["funnel_gates_launched"] == 0
    assert [u["gate_name"] for u in result["saved"]] == [g.name for g in mf.GATES]


def test_negative_index_refuses_before_any_stage():
    """NEGATIVE CONTROL: a dead scar must not enter the ladder."""
    called: list[str] = []
    funnel, advanced = _spy_funnel()

    def refuse(_proposal):
        return {
            "refused": True,
            "scar_id": "TEST-SCAR-DEAD",
            "source_path": "tools/future/negative_index.py",
            "reason": "known-dead hypothesis; rediscovery is not free",
            "hypothesis_family": "cross_expert_structure",
        }

    cand = av._candidate("neg.dead.index", **av._all_pass_inputs())
    result = av.screen(
        cand,
        refuse_fn=refuse,
        funnel=funnel,
        on_stage=lambda s, _c: called.append(s.name),
    )
    assert result["verdict"] == av.VERDICT_REFUSED_DEAD
    assert result["refused_by"] == "negative_index"
    assert result["verified"] is False
    assert result["killed_by"] is None
    assert called == []
    assert advanced == []
    assert result["stages_run"] == []
    assert result["cost"]["stages_executed"] == 0
    assert result["cost"]["funnel_gates_launched"] == 0
    assert result["cost"]["later_stages_launched"] == 0
    assert [u["gate_name"] for u in result["saved"]] == [g.name for g in mf.GATES]
    assert all(
        "negative index already killed" in u["not_launched_because"]
        for u in result["saved"]
    )


def test_default_screen_calls_refuse_if_dead(monkeypatch):
    """DECLARED CAPABILITY != EXECUTED CAPABILITY: the index lookup must run."""
    hits: list[dict] = []

    def fake(proposal, scars=None):
        hits.append(dict(proposal))
        return {
            "refused": True,
            "scar_id": "PATCHED",
            "source_path": "patched",
            "reason": "patched hit",
        }

    monkeypatch.setattr(ni, "refuse_if_dead", fake)
    called: list[str] = []
    cand = av._candidate("neg.default.index")
    result = av.screen(cand, on_stage=lambda s, _c: called.append(s.name))
    assert hits, "screen() did not call negative_index.refuse_if_dead"
    assert result["verdict"] == av.VERDICT_REFUSED_DEAD
    assert called == []


def test_screen_invokes_funnel_advance_not_a_named_list():
    """Naming meta_funnel.GATES is not evidence Funnel.advance ran."""
    funnel, advanced = _spy_funnel()
    cand = av._candidate(
        "neg.invoke.funnel",
        teacher_corpus={"status": "FAILED", "fit_passed": False, "mechanism": "null failed"},
    )
    result = av.screen(cand, refuse_fn=_clear, funnel=funnel)
    assert advanced == ["analytical_structure_screen", "real_teacher_fit"]
    assert result["verdict"] == av.VERDICT_KILLED
    assert result["killed_by"] == "real_teacher_fit"
    assert result["scar_id"]
    assert funnel.scars and funnel.scars[0]["gate_name"] == "real_teacher_fit"


def test_hardware_claim_guard_still_bites():
    doc = {"tps": 12.0}
    try:
        _assert_no_hardware_claims(doc)
    except HardwareClaimError:
        return
    raise AssertionError("hardware claim guard did not fire on tps=12.0")


def test_recovered_families_do_not_report_verification():
    families, _provenance = mf.recover_families()
    assert families
    for fam in families:
        result = av.screen(dict(fam), refuse_fn=_clear)
        assert result["verified"] is False
        assert result["verdict"] != "VERIFIED"
        for unit in result["saved"]:
            assert unit["origin"] == av.FUNNEL_ORIGIN
            assert unit["workunit"].startswith("future.meta_funnel.gate.")
