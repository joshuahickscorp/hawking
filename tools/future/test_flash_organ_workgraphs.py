"""Tests for Flash organ-school funnel WorkGraphs.

Negative controls (must actually fail if the guard is removed):
  * a node whose input receipt is absent SLEEPS with a wake condition, never runs
  * no cross-school edge exists over the union of all fourteen graphs
  * a weight organ whose census family is empty is reported as such, not given
    a complete-looking empty graph
  * COMPLETE is refused unless the named receipt is a file on disk
A guard nobody has watched fail is not a guard.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.future import flash_organ_workgraphs as fog
from tools.future import flash_schools as fs
from tools.future import meta_funnel as mf
from tools.future._common import RECEIPTS, REPO, _assert_no_hardware_claims


def _graphs(**kwargs):
    inv = kwargs.pop("inventory", fs.organ_inventory())
    teacher = kwargs.pop("teacher_state", fog.flash_teacher_corpus_state())
    return fog.build_all_graphs(inventory=inv, teacher_state=teacher, **kwargs)


def _emitted(graphs):
    return [g for g in graphs if g.get("graph_emitted")]


def test_build_emits_sealed_receipt():
    out = fog.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "FLASH_ORGAN_WORKGRAPHS.json"
    assert doc["schema"] == "hawking.future.flash_organ_workgraphs.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert len(doc["seal_sha256"]) == 64
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert doc["seal_sha256"] == hashlib.sha256(blob).hexdigest()
    _assert_no_hardware_claims(doc)
    assert doc["n_funnel_stages"] == 11
    assert doc["counts"]["schools_in_catalog"] == len(fs.SCHOOL_CATALOG)
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]["entry_point"].endswith("build_all_graphs()")
    assert doc["resident_callable"]["frontier"] == "FT.MODEL_REPRESENTATION.meta-gates-3-9"
    assert doc["negative_controls"]["all_fired"] is True
    assert doc["era_vocabulary"]["no_era_vi"] is True
    assert doc["era_vocabulary"]["no_odyssey_iv"] is True


def test_selftest_aliases_build():
    assert fog.selftest().name == "FLASH_ORGAN_WORKGRAPHS.json"


def test_eleven_funnel_stages_map_onto_meta_funnel_plus_two_gaps():
    names = [s.name for s in fog.FUNNEL_STAGES]
    assert names == [
        "analytic_structure",
        "tiny_synthetic_sanity",
        "real_teacher_fit",
        "held_out_numerical",
        "route_stability",
        "state_stability",
        "logit_token",
        "capability_subset",
        "physical_nr_lowering",
        "native_nx",
        "complete_ebpw",
    ]
    assert len(fog.FUNNEL_STAGES) == 11
    meta_mapped = [s for s in fog.FUNNEL_STAGES if s.meta_funnel_gate is not None]
    assert len(meta_mapped) == 9
    assert len(mf.GATES) == 9
    by_gate = {s.meta_funnel_gate: s for s in meta_mapped}
    assert by_gate[1].name == "analytic_structure"
    assert by_gate[2].name == "real_teacher_fit"
    assert by_gate[2].meta_funnel_gate == mf.GATES_BY_NAME["real_teacher_fit"].id
    gaps = [s.name for s in fog.FUNNEL_STAGES if s.meta_funnel_gate is None]
    assert gaps == ["tiny_synthetic_sanity", "state_stability"]
    for s in fog.FUNNEL_STAGES:
        assert s.verifier
        assert s.sleeps_when
        assert s.wake_condition
        assert s.passing_proves
        assert s.passing_does_not_prove
        d = s.as_dict()
        assert d["evidence_class"] == "STATIC_ONLY"
        assert d["gpu_authority"] is False


def test_one_graph_per_catalog_school_and_unknown_school_raises():
    graphs = _graphs()
    assert [g["school"] for g in graphs] == list(fs.SCHOOL_CATALOG)
    emitted = _emitted(graphs)
    assert emitted, "census unreachability must still emit graphs, not fake emptiness"
    for g in emitted:
        assert g["n_nodes"] == 11
        assert g["n_edges"] == 10
        assert g["independent"] is True
        assert g["schedulable_alone"] is True
        assert g["looks_complete"] is False
        assert g["evidence_class"] == "STATIC_ONLY"
        assert g["gpu_authority"] is False
        ids = [n["id"] for n in g["nodes"]]
        assert ids == [fog.node_id(g["school"], s.name) for s in fog.FUNNEL_STAGES]
        for e in g["edges"]:
            assert e["school"] == g["school"]
            assert fog.school_of(e["from"]) == g["school"]
            assert fog.school_of(e["to"]) == g["school"]
    with pytest.raises(fs.UnknownSchoolError):
        fog.build_school_graph("ERA_VI_DOES_NOT_EXIST")


def test_no_cross_school_edges_over_the_union():
    """NEGATIVE CONTROL: independence is watched to fire."""
    graphs = _graphs()
    edges = fog.union_edges(graphs)
    assert edges, "emitted graphs must have intra-school funnel edges"
    cross = fog.cross_school_edges(graphs)
    assert cross == [], cross
    schools = {e["school"] for e in edges}
    assert schools <= set(fs.SCHOOL_CATALOG)
    for e in edges:
        assert fog.school_of(e["from"]) == fog.school_of(e["to"]) == e["school"]
    # A synthetic cross-school edge must be detected, not swallowed.
    poisoned = list(graphs)
    if not _emitted(poisoned):
        pytest.fail("need at least one emitted graph to poison")
    fake = {
        "school": "POISON",
        "graph_emitted": True,
        "edges": [
            {
                "from": "ROUTED_EXPERTS:analytic_structure",
                "to": "NGRAM:tiny_synthetic_sanity",
                "school": "ROUTED_EXPERTS",
            }
        ],
        "nodes": [],
    }
    assert fog.cross_school_edges([*poisoned, fake]) != []


def test_absent_input_receipt_sleeps_and_never_runs():
    """NEGATIVE CONTROL: missing teacher corpus SLEEPS; run_node raises."""
    graphs = _graphs()
    teacher_nodes = [
        n
        for g in _emitted(graphs)
        for n in g["nodes"]
        if n["stage"] == "real_teacher_fit"
    ]
    assert teacher_nodes, "every emitted graph has a teacher-fit node"
    assert all(n["status"] == fog.ST_SLEEPING for n in teacher_nodes)
    assert all(n["input_state"] in {"NOT_BUILT", "ABSENT", "NOT_ON_DISK"} for n in teacher_nodes)
    assert all(n["wake_condition"] for n in teacher_nodes)
    sample = teacher_nodes[0]
    with pytest.raises(fog.NodeAsleep) as exc:
        fog.run_node(sample)
    assert sample["id"] in str(exc.value)
    assert exc.value.wake_condition
    # Injected missing receipt on a prefix-independent later stage also sleeps
    # when no ancestor has slept yet: probe everything absent except we still
    # have in-process prefix. Teacher is the first sleeper.
    def nobody(rel: str) -> bool:
        return False
    g = fog.build_school_graph(
        "ROUTER",
        receipt_probe=nobody,
        teacher_state={
            "state": "NOT_BUILT",
            "reason": "injected absence",
            "on_disk": [],
        },
    )
    assert g["graph_emitted"] is True
    teacher = next(n for n in g["nodes"] if n["stage"] == "real_teacher_fit")
    assert teacher["status"] == fog.ST_SLEEPING
    held = next(n for n in g["nodes"] if n["stage"] == "held_out_numerical")
    assert held["status"] == fog.ST_UNREACHABLE
    assert held["unreachable_because"] == teacher["id"]


def test_nodes_past_a_sleeper_are_unreachable_not_pending():
    graphs = _graphs()
    for g in _emitted(graphs):
        teacher = next(n for n in g["nodes"] if n["stage"] == "real_teacher_fit")
        assert teacher["status"] == fog.ST_SLEEPING
        after = [n for n in g["nodes"] if n["index"] > teacher["index"]]
        assert after
        for n in after:
            assert n["status"] == fog.ST_UNREACHABLE, (g["school"], n["id"], n["status"])
            assert n["status"] != "pending"
            assert n["unreachable_because"] == teacher["id"]
            with pytest.raises(fog.NodeUnreachable):
                fog.run_node(n)
        prefix = [n for n in g["nodes"] if n["stage"] in fog.PREFIX_STAGES]
        assert prefix
        for n in prefix:
            assert n["status"] in {fog.ST_READY, fog.ST_COMPLETE, fog.ST_AWAITING}
            assert n["status"] not in {fog.ST_SLEEPING, fog.ST_UNREACHABLE}


def test_empty_census_family_is_reported_not_a_complete_looking_graph():
    """NEGATIVE CONTROL: empty family_summary is not a silent empty success."""
    empty_inv = {
        "census_source": "pinned_snapshot",
        "by_family": {},
        "budget_by_family": {},
        "families": [],
        "router_tensor_bytes": None,
    }
    g = fog.build_school_graph("ROUTED_EXPERTS", inventory=empty_inv)
    assert g["empty_census_family"] is True
    assert g["graph_emitted"] is False
    assert g["looks_complete"] is False
    assert g["nodes"] is None
    assert g["edges"] is None
    assert g["n_nodes"] == 0
    assert "routed_experts" in (g["reason"] or "")
    # Function organs declare an empty family tuple on purpose and still
    # emit a graph. That is not this control.
    router = fog.build_school_graph("ROUTER", inventory=empty_inv)
    assert router["empty_census_family"] is False
    assert router["graph_emitted"] is True
    assert router["function_organ"] is True
    assert router["n_nodes"] == 11
    assert router["looks_complete"] is False
    # Unreachable census is coped, not reported as empty.
    missing_inv = {
        "census_source": "unavailable",
        "by_family": {},
        "budget_by_family": {},
        "families": [],
    }
    coped = fog.build_school_graph("NGRAM", inventory=missing_inv)
    assert coped["empty_census_family"] is False
    assert coped["graph_emitted"] is True
    assert coped["census"]["kind"] == "census_unreachable"


def test_refuse_complete_without_receipt_on_disk():
    """NEGATIVE CONTROL: COMPLETE is not a status you can assert."""
    graphs = _graphs()
    ready = next(
        n
        for g in _emitted(graphs)
        for n in g["nodes"]
        if n["status"] == fog.ST_READY
    )
    missing = REPO / "receipts" / "future" / "DOES_NOT_EXIST_ORGAN_WG.json"
    assert not missing.is_file()
    with pytest.raises(fog.CompleteWithoutReceipt):
        fog.mark_complete(ready, missing)
    # A SLEEPING node cannot be completed even with a real file.
    teacher = next(
        n
        for g in _emitted(graphs)
        for n in g["nodes"]
        if n["status"] == fog.ST_SLEEPING
    )
    with pytest.raises(fog.CompleteWithoutReceipt):
        fog.mark_complete(teacher, Path(__file__))
    unreachable = next(
        n
        for g in _emitted(graphs)
        for n in g["nodes"]
        if n["status"] == fog.ST_UNREACHABLE
    )
    with pytest.raises(fog.CompleteWithoutReceipt):
        fog.mark_complete(unreachable, Path(__file__))
    # Evaluator itself never returns COMPLETE unless the probe says the file exists.
    for n in fog.all_nodes(graphs):
        if n["status"] == fog.ST_COMPLETE:
            assert n["completion_receipt"]
            assert (REPO / n["completion_receipt"]).is_file()


def test_ready_prefix_is_runnable_and_does_not_claim_teacher_fit():
    g = fog.build_school_graph("POSITIONAL_STRUCTURE")
    assert g["graph_emitted"] is True
    # Which stage sits at the ready frontier depends on which receipts are on
    # disk: once analytic_structure has its receipt the frontier advances to
    # tiny_synthetic_sanity. Pinning a stage name asserts the state of a sparse
    # worktree. What must hold is that every runnable pre-teacher node runs, and
    # that none of them claims teacher fit or marks itself complete.
    pre_teacher = [n for n in g["nodes"]
                   if n["stage"] in ("analytic_structure", "tiny_synthetic_sanity")]
    assert pre_teacher, "the graph has no pre-teacher stages at all"
    ready = [n for n in pre_teacher if n["status"] == fog.ST_READY]
    assert ready, "no pre-teacher stage is runnable; the prefix is not usable"
    for node in ready:
        assert node["schedulable_now"] is True
        r = fog.run_node(node)
        assert r["ran"] is True
        assert r["not_teacher_fit"] is True, "a pre-teacher node claimed teacher fit"
        assert r["marked_complete"] is False, "a node marked itself complete"
        if node["stage"] == "analytic_structure":
            assert r["verdict"] in {"PASSED", "KILLED", "REFUSED"}
        else:
            assert r["used_teacher_corpus"] is False
            assert r["admitted"] == r["n_candidates"]


def test_schools_do_not_block_each_other_in_ready_workgraph():
    graphs = _graphs()
    ready = fog.emit_ready_units(graphs)
    assert ready["cross_school_ready_deps"] == []
    assert ready["sleeping_organ_nodes_converted"] == 0
    assert ready["gpu_authority"] is False
    # Independence is the invariant, not which stage happens to be at the
    # frontier. Every emitted school must contribute exactly one ready unit, and
    # no school may wait on another -- whichever stage each has reached.
    emitted_schools = {g["school"] for g in _emitted(graphs)}
    assert emitted_schools, "no school emitted a graph"
    by_school = {}
    for uid in ready["admitted_ids"]:
        by_school.setdefault(fog.school_of(uid), []).append(uid)
    assert set(by_school) == emitted_schools, "a school contributed no ready unit"
    for school, uids in by_school.items():
        assert len(uids) == 1, f"{school} emitted {len(uids)} ready units, not one"
        assert uids[0] in ready["ready_ids"]


def test_function_organs_still_get_a_graph():
    assert fog.FUNCTION_ORGANS == frozenset({"ROUTER", "KV_STATE", "DECODING", "MTP_SPECULATION"})
    for school in sorted(fog.FUNCTION_ORGANS):
        g = fog.build_school_graph(school)
        assert g["function_organ"] is True
        assert g["graph_emitted"] is True
        assert g["n_nodes"] == 11
        assert g["teacher_fit"]["status"] == fog.ST_SLEEPING


def test_extends_existing_modules_not_a_fork():
    rec = fog.recovered_implementation()
    paths = [row["path"] for row in rec["landed_siblings_extended"]]
    assert "tools/future/flash_schools.py" in paths
    assert "tools/future/meta_funnel.py" in paths
    assert "tools/future/workgraph.py" in paths
    assert "tools/future/expert_bank_school.py" in paths
    assert "tools/future/ngram_school.py" in paths
    assert "tools/future/moe_physical_school.py" in paths
    assert "tools/future/router_science.py" in paths
    assert rec["not_duplicating"]
    assert fog.FUNNEL_STAGES[0].meta_funnel_gate == 1
    # Catalog identity: we did not invent a fifteenth school.
    assert [g["school"] for g in fog.build_all_graphs()] == list(fs.SCHOOL_CATALOG)


def test_teacher_corpus_contract_is_not_a_corpus():
    st = fog.flash_teacher_corpus_state()
    assert st["contract_is_not_a_corpus"] is True
    assert st["glm_teacher_forced_is_wrong_specimen"] is True
    if st["state"] == "PRESENT":
        assert st["on_disk"]
    else:
        assert st["state"] == "NOT_BUILT"
        assert not st["on_disk"]


def test_state_stability_negative_and_positive_evaluators():
    killed, reason = fog._eval_state_stability({}, {"state_identity": False, "mechanism": "kv desync"})
    assert killed == "KILLED"
    assert "kv desync" in reason
    passed, _ = fog._eval_state_stability({}, {"status": "PASSED", "state_identity": True})
    assert passed == "PASSED"
    refused, _ = fog._eval_state_stability({}, {"status": "NOT_MEASURED"})
    assert refused == "REFUSED"
