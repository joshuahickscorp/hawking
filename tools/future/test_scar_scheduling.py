"""Scar scheduling: admission gate, four-way propagation, negative controls.

A guard nobody has watched fail is not a guard. Each of the four
propagation effects is asserted on its own; the receipt existing is not
the proof.
"""
from __future__ import annotations

import json

from tools.future import negative_index as ni
from tools.future import odyssey2_law_store as ols
from tools.future import odyssey3_adversary as o3
from tools.future import scar_scheduling as ss
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


def _fresh_env() -> tuple[ss.ScarScheduler, dict, dict]:
    fx = ss.four_way_fixture()
    sch = ss.ScarScheduler(
        include_corpus=False,
        declared_edges=list(fx["declared_edges"]),
        laws=[dict(fx["law"])],
    )
    result = sch.propagate_failure(fx["failure"])
    return sch, fx, result


def test_build_emits_sealed_receipt():
    out = ss.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "SCAR_SCHEDULING.json"
    assert doc["schema"] == "hawking.future.scar_scheduling.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]["can_hcli_invoke"] is True
    assert doc["resident_callable"]["entry_point"]
    assert doc["resident_callable"]["workunit_emitted"]["id"] == ss.WORKUNIT_ID
    assert doc["resident_callable"]["receipt"] == "receipts/future/SCAR_SCHEDULING.json"
    assert doc["resident_callable"]["frontier_fed"]
    assert doc["resident_callable"]["fail_closed"]
    assert doc["workunit"]["id"] == ss.WORKUNIT_ID
    assert doc["workunit"]["classification"] == "STATIC_ONLY"
    assert doc["confidence_reduction_rule"] == ss.CONFIDENCE_REDUCTION_RULE
    prevented = doc["experiments_prevented"]
    assert prevented["n_total"] == (
        prevented["n_refused_admissions"] + prevented["n_invalidated_dependents"]
    )
    assert prevented["n_total"] >= 1
    ingest = doc["known_scars_ingest"]
    assert ingest["n_already_present"] + ingest["n_newly_added"] == ingest["n_input"]
    _assert_no_hardware_claims(doc)


def test_selftest_runs_and_seals():
    out = ss.selftest()
    doc = json.loads(out.read_text())
    assert doc["schema"] == ss.SCHEMA
    assert doc["seal_sha256"]
    assert doc["four_way_proof"]["scar_stored"] is True
    assert doc["corpus_negative_control"]["dead"]["decision"] == ss.DECISION_REFUSED
    assert doc["corpus_negative_control"]["near_miss_organ_rep"]["decision"] == ss.DECISION_ADMITTED


def test_admit_refuses_known_dead_and_admits_near_miss():
    """NEGATIVE CONTROL: the gate fires on a live corpus dead hypothesis
    and still admits a structurally different proposal."""
    sch = ss.ScarScheduler(include_corpus=True)
    dead = sch.admit(
        {
            "id": "wu.dead.cross-expert",
            "model": "qwen3-235b-a22b",
            "organ": "gate",
            "hypothesis_family": "cross_expert_structure",
        }
    )
    assert dead["decision"] == ss.DECISION_REFUSED, dead
    assert dead["scar_id"]
    assert dead["source_path"]
    assert dead["hypothesis_family"] == "cross_expert_structure"
    assert dead["workunit"]["status"] == "blocked"
    assert dead["workunit"]["classification"] == "SCAR_REFUSED"
    assert dead["workunit"]["failure_context"]["scar_id"] == dead["scar_id"]
    known = (
        "NEGATIVE_TRANSFER_ATLAS.json",
        "NEGATIVE_SCIENCE.json",
        "NOETIC_NEGATIVE_SCIENCE.json",
        "QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json",
        "DOCTOR_NEGATIVE_TRANSFER_ATLAS.json",
    )
    assert any(k in str(dead["source_path"]) for k in known), dead["source_path"]

    near = sch.admit(
        {
            "id": "wu.near.lmhead-q4",
            "model": "qwen3-235b-a22b",
            "organ": "lm_head",
            "representation": "q4",
            "hypothesis_family": "cross_expert_structure",
        }
    )
    assert near["decision"] == ss.DECISION_ADMITTED, near

    other = sch.admit(
        {
            "id": "wu.live.hwir",
            "model": "qwen3-235b-a22b",
            "organ": "gate",
            "hypothesis_family": "hwir_node_types",
        }
    )
    assert other["decision"] == ss.DECISION_ADMITTED, other


def test_admit_is_first_class_outcome_not_exception():
    sch = ss.ScarScheduler(include_corpus=True)
    # Must return, not raise, on a known-dead hypothesis.
    outcome = sch.admit(
        {
            "id": "wu.no-raise",
            "model": "qwen3-235b-a22b",
            "organ": "gate",
            "hypothesis_family": "cross_expert_structure",
        }
    )
    assert outcome["decision"] == ss.DECISION_REFUSED
    assert "workunit" in outcome
    assert outcome["workunit"]["failure_context"]["decision"] == ss.DECISION_REFUSED


def test_fresh_failure_stores_scar_with_six_keys():
    """(a) store the scar with model/organ/representation/machine/hypothesis/mechanism."""
    _sch, fx, result = _fresh_env()
    assert result["scar_stored"] is True
    scar = result["scar"]
    assert scar["model"] not in (None, "", ni.UNRECORDED)
    assert scar["organ"] == "gate"
    assert scar["representation"] == "kronecker"
    assert scar["machine"] == "m3_ultra"
    assert scar["hypothesis_family"] == "kronecker"
    assert scar["failure_mechanism"]
    assert scar["keys_filled"] >= 6
    assert scar["refuse_eligible"] is True
    assert scar["scar_id"]


def test_fresh_failure_invalidates_named_dependents():
    """(b) invalidate dependent candidates — name them."""
    _sch, fx, result = _fresh_env()
    ids = result["invalidated_candidate_ids"]
    assert ss.FOUR_WAY_CHILD_A in ids
    assert ss.FOUR_WAY_CHILD_B in ids
    assert ss.FOUR_WAY_UNRELATED not in ids
    named = {row["candidate_id"]: row for row in result["invalidated_candidates"]}
    assert named[ss.FOUR_WAY_CHILD_A]["scar_id"] == result["scar"]["scar_id"]
    assert named[ss.FOUR_WAY_CHILD_B]["failed_candidate_id"] == ss.FOUR_WAY_PARENT
    assert named[ss.FOUR_WAY_CHILD_A]["reason"]


def test_fresh_failure_reduces_law_confidence_by_stated_rule():
    """(c) REDUCE the confidence of any law that predicted success, by a stated rule."""
    _sch, fx, result = _fresh_env()
    reds = result["confidence_reductions"]
    assert reds, "no law that predicted success was reduced"
    red = reds[0]
    assert red["law_id"] == fx["law"]["law_id"]
    assert red["before"] == fx["confidence_before"]
    assert red["after"] == fx["confidence_after"]
    assert red["factor"] == ss.CONFIDENCE_REDUCTION_FACTOR
    assert red["floor"] == ss.CONFIDENCE_FLOOR
    assert red["rule"] == ss.CONFIDENCE_REDUCTION_RULE
    assert red["after"] == max(
        ss.CONFIDENCE_FLOOR,
        round(red["before"] * ss.CONFIDENCE_REDUCTION_FACTOR, 4),
    )


def test_fresh_failure_updates_odyssey2_scope_and_emits_odyssey3_implication():
    """(d) update Odyssey II scope and generate an Odyssey III implication."""
    _sch, fx, result = _fresh_env()
    scopes = result["odyssey2_scope_updates"]
    assert scopes, "Odyssey II scope was not updated"
    upd = scopes[0]
    assert upd["law_id"] == fx["law"]["law_id"]
    assert upd["scope_before"] == "ARCHITECTURE_FAMILY"
    assert upd["scope_after"] == "MODEL_LOCAL"
    assert upd["moved"] is True
    assert upd["direction"] == "DOWN"

    impl = result["odyssey3_implication"]
    assert impl.get("moved") is True, impl
    assert impl["direction"] == "DOWN"
    assert impl["verdict"] == "REFUTED"
    assert impl["attack_id"]
    assert o3.is_downgrade(impl["scope_before"], impl["scope_after"])
    assert impl["scope_before"] != impl["scope_after"]


def test_writing_a_json_is_not_the_four_way_proof():
    """Writing a failed JSON is explicitly NOT sufficient — effects live on the scheduler."""
    sch, _fx, result = _fresh_env()
    assert result["scar_stored"] is True
    assert sch.extra_scars, "scar must land in the in-memory index, not only a file"
    assert sch.invalidated, "dependents must be named on the scheduler"
    assert sch.confidence_reductions, "confidence reductions must be on the scheduler"
    assert sch.scope_updates, "O2 scope updates must be on the scheduler"
    assert sch.o3_implications, "O3 implication must be on the scheduler"
    # A receipt write is a later report of these effects, not the effects.
    assert "scar_id" in result["scar"]


def test_specificity_organ_representation_pair():
    """A scar about one organ/representation pair must not block a near-miss."""
    sch, fx, _result = _fresh_env()
    dead = sch.admit(fx["dead_unit"])
    assert dead["decision"] == ss.DECISION_REFUSED, dead
    near = sch.admit(fx["near_miss_unit"])
    assert near["decision"] == ss.DECISION_ADMITTED, near
    # Same organ, different representation is also a near-miss.
    same_organ_other_rep = sch.admit(
        {
            "id": "wu.near.gate-q4",
            "model": ss.FOUR_WAY_MODEL,
            "organ": "gate",
            "representation": "q4",
            "hypothesis_family": "kronecker",
        }
    )
    assert same_organ_other_rep["decision"] == ss.DECISION_ADMITTED, same_organ_other_rep


def test_ingest_known_scars_reports_present_vs_added():
    sch = ss.ScarScheduler(include_corpus=True)
    report = sch.ingest_known_scars()
    assert report["n_already_present"] + report["n_newly_added"] == report["n_input"]
    # Cope with either recovery path: handoff present or unresolved.
    assert report["handoff_source"] in {
        "disk",
        "pinned_snapshot",
        "git",
        "unresolved",
        "unreadable",
        "git_unreadable",
    }
    if report["n_input"]:
        names = [r["name"] for r in report["records"]]
        assert names == sorted(names)
        for rec in report["records"]:
            assert rec["status"] in {"already_present", "newly_added"}
            keys = rec["keys"]
            for field in (
                "model",
                "organ",
                "representation",
                "machine",
                "hypothesis_family",
                "failure_mechanism",
            ):
                assert field in keys
            if rec.get("receipt"):
                # Seals travel with the record when the source had them.
                assert "receipt_seal_sha256" in rec
        # Second ingest is idempotent: everything already present.
        again = sch.ingest_known_scars()
        assert again["n_newly_added"] == 0
        assert again["n_already_present"] == again["n_input"]


def test_experiments_prevented_metric():
    sch, fx, result = _fresh_env()
    dead = sch.admit(fx["dead_unit"])
    assert dead["decision"] == ss.DECISION_REFUSED
    near = sch.admit(fx["near_miss_unit"])
    assert near["decision"] == ss.DECISION_ADMITTED
    metric = sch.experiments_prevented()
    assert metric["n_refused_admissions"] >= 1
    assert metric["n_invalidated_dependents"] == len(result["invalidated_candidate_ids"])
    assert metric["n_total"] == metric["n_refused_admissions"] + metric["n_invalidated_dependents"]
    assert metric["n_unique"] >= 1
    assert fx["dead_unit"]["id"] in metric["refused_workunit_ids"]
    assert ss.FOUR_WAY_CHILD_A in metric["invalidated_candidate_ids"]
    assert metric["citations"]
    # Near-miss is not a prevented experiment.
    assert fx["near_miss_unit"]["id"] not in metric["refused_workunit_ids"]


def test_idempotent_propagation_does_not_double_halve():
    fx = ss.four_way_fixture()
    sch = ss.ScarScheduler(
        include_corpus=False,
        declared_edges=list(fx["declared_edges"]),
        laws=[dict(fx["law"])],
    )
    first = sch.propagate_failure(fx["failure"])
    second = sch.propagate_failure(fx["failure"])
    assert second["duplicate"] is True
    assert first["confidence_reductions"][0]["after"] == 0.32
    assert second["confidence_reductions"][0]["after"] == 0.32
    assert sch.laws[0]["transfer_confidence"]["value"] == 0.32


def test_fail_closed_when_index_raises():
    class Boom(ss.ScarScheduler):
        def pool(self):
            raise RuntimeError("index unavailable")

    sch = Boom(include_corpus=False)
    outcome = sch.admit(
        {
            "id": "wu.fail-closed",
            "model": "qwen3-235b-a22b",
            "organ": "gate",
            "hypothesis_family": "cross_expert_structure",
        }
    )
    assert outcome["decision"] == ss.DECISION_REFUSED
    assert outcome["fail_closed"] is True
    assert "index unavailable" in outcome["reason"]
    assert outcome["workunit"]["classification"] == "SCAR_REFUSED"


def test_plumbing_unit_without_hypothesis_still_admits():
    sch = ss.ScarScheduler(include_corpus=False)
    outcome = sch.admit({"id": "wu.plumbing", "role": "science", "description": "copy a receipt"})
    assert outcome["decision"] == ss.DECISION_ADMITTED


def test_invalidated_dependent_is_refused_on_admit():
    sch, fx, result = _fresh_env()
    child = sch.admit(
        {
            "id": ss.FOUR_WAY_CHILD_A,
            "candidate_id": ss.FOUR_WAY_CHILD_A,
            "model": ss.FOUR_WAY_MODEL,
            "organ": "gate",
            "representation": "kronecker",
            "hypothesis_family": "kronecker",
        }
    )
    assert child["decision"] == ss.DECISION_REFUSED
    assert child["scar_id"] == result["scar"]["scar_id"]


def test_receipt_has_no_hardware_numeric_claims():
    out = ss.build()
    doc = json.loads(out.read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                    raise AssertionError(f"hardware field {here}={v!r}")
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)
    assert doc["evidence_class"] == "STATIC_ONLY"


def test_handoff_load_copes_with_either_state():
    loaded = ss.load_handoff()
    # Never treat sparse absence as proof the handoff does not exist.
    assert loaded.get("present") in {True, None}
    assert "known_scars" in loaded
    assert isinstance(loaded["known_scars"], list)
    if loaded.get("present") is True:
        n = len(loaded["known_scars"])
        ingest = ss.ScarScheduler(include_corpus=False).ingest_known_scars(loaded)
        assert ingest["n_input"] == n
        assert ingest["n_already_present"] + ingest["n_newly_added"] == n
    else:
        assert loaded.get("source") in {"unresolved", "unreadable", "git_unreadable"}
        assert loaded.get("searched")


def test_demote_mapping_never_widens():
    for scope in ("MODEL_LOCAL", "ARCHITECTURE_FAMILY", "GENERIC_CANDIDATE", "GENERIC_VERIFIED"):
        d = ss.demote_odyssey2_scope(scope)
        if d["scope_before"] in ols.SCOPES and d["scope_after"] in ols.SCOPES:
            assert ols.SCOPES.index(d["scope_after"]) <= ols.SCOPES.index(d["scope_before"])
    machine = ss.demote_odyssey2_scope("MACHINE_LOCAL")
    assert machine["moved"] is False
    assert machine["scope_after"] == "MACHINE_LOCAL"


def test_emit_resident_workunit_is_static_only():
    unit = ss.emit_resident_workunit()
    assert unit["classification"] == "STATIC_ONLY"
    assert unit["effect_class"] == "READ_ONLY"
    assert unit["resource_class"] == "STATIC_ANALYSIS"
    assert unit["output_receipt_path"] == "receipts/future/SCAR_SCHEDULING.json"
    assert unit["command"][0] == "python3"
