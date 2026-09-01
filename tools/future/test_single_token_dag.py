"""The token is a chain, and that is what makes it a finding rather than a drawing.

S025 §4-5 asks how much GPU work is serial by SEMANTICS versus serial only
because this executor schedules it that way. The answer decides where G009's
missing capacity can possibly be: if the serialization were artificial,
reordering would recover it; if the token is a true chain, no reordering will and
the cause is inside the kernels.
"""
from __future__ import annotations

import pytest

from tools.future import single_token_dag as dag


def test_an_edge_without_a_reason_is_refused():
    """An unexplained edge makes every critical path downstream unfalsifiable."""
    with pytest.raises(dag.DagRefused, match="say WHY"):
        dag._edge("a", "b", "")
    with pytest.raises(dag.DagRefused, match="say WHY"):
        dag._edge("a", "b", "because")


def test_an_unknown_edge_kind_is_refused():
    with pytest.raises(dag.DagRefused, match="unknown edge kind"):
        dag._edge("a", "b", "a long enough reason to pass the length bar", "MAYBE")


def test_every_edge_names_its_read_and_write_relationship():
    for e in dag.token_edges():
        assert len(e["why"]) >= 20, e
        assert e["kind"] in {dag.TRUE_DEPENDENCY, dag.ARTIFICIAL, dag.UNKNOWN}


def test_the_token_is_a_chain_with_no_artificial_barrier():
    s = dag.slack()
    assert s["n_true_dependency"] == s["n_edges"]
    assert s["largest_artificial_barriers"] == []
    assert s["unknown_edges"] == []
    assert s["theoretically_overlapable_ns"] == 0


def test_overlap_efficiency_is_none_not_zero():
    """0.0 would read as a failure to exploit slack that does not exist."""
    s = dag.slack()
    assert s["overlap_efficiency"] is None
    assert "divide by zero" in s["overlap_efficiency_is_none_because"]


def test_the_critical_path_is_the_whole_measured_token():
    s = dag.slack()
    assert s["critical_path_ns"] == s["total_gpu_work_ns"]
    assert s["measured_total_ms"] == pytest.approx(26.70, abs=0.05)


def test_every_measured_row_is_bound_to_a_node():
    """Measured time outside the graph makes the critical path wrong by that much."""
    n = dag.node_times()
    assert n["unmapped_rows"] == []
    assert n["n_rows"] == len(n["row_to_node"])
    assert sum(n["ms_by_node"].values()) == pytest.approx(dag.slack()["measured_total_ms"], abs=0.01)


def test_q4_remainder_is_the_mixer_out_projection_and_cites_its_source():
    """It needed reading, not guessing: it is out_proj/o_proj for all 64 layers."""
    n = dag.node_times()
    assert n["row_to_node"]["q4_remainder"] == "mixer.out"
    assert "qwen38_hybrid_decode.rs:5310" in n["cites"]["q4_remainder"]


def test_the_fusions_that_already_took_the_slack_are_recorded():
    """gate/up/swiglu, q/k/v, residual/rmsnorm and ba_to_decay were independent."""
    fused = {tuple(f["was"]) for f in dag.FUSED_INDEPENDENCE}
    assert ("gate_proj", "up_proj", "swiglu") in fused
    assert ("q_proj", "k_proj", "v_proj") in fused
    for f in dag.FUSED_INDEPENDENCE:
        assert f["why_independent"], f["now"]


def test_the_reading_does_not_overclaim_about_within_kernel_slack():
    r = dag.reading()
    assert r["finding"] == "THE_TOKEN_IS_A_CHAIN"
    joined = " ".join(r["what_this_does_not_prove"])
    assert "WITHIN a kernel" in joined
    assert "heads, groups and row blocks" in joined


def test_the_reordering_scar_is_emitted_by_the_producer():
    """A scar in a hand-edited receipt does not survive a rebuild."""
    doc = dag.build()
    scars = doc["scars"]
    assert len(scars) == 1
    s = scars[0]
    assert s["family"] == "TOP_LEVEL_TOKEN_REORDERING_HAS_NO_CURRENT_SLACK"
    assert s["authority"] == "S026 §4"
    assert "11 edges" in s["mechanism"]


def test_the_scar_is_computed_from_the_slack_not_typed(monkeypatch):
    """If overlapable work ever appears, the scar must refuse to be emitted."""
    real = dag.slack()
    monkeypatch.setattr(
        dag, "slack",
        lambda: {**real, "theoretically_overlapable_ns": 1_000_000})
    with pytest.raises(dag.DagRefused, match="reopen condition has fired"):
        dag.reordering_scar()


def test_the_scar_does_not_claim_the_gpu_is_saturated():
    s = dag.reordering_scar()
    assert "a claim that the GPU is saturated" in s["not"]
    assert "450-580 aggregate GB/s" in s["not"], "the oracle that motivates it"
    assert "says only WHERE IT IS NOT" in s["not"]


def test_the_scar_carries_a_reopen_condition():
    s = dag.reordering_scar()
    assert "speculative drafting" in s["reopen"]
    assert "recomputes the slack and raises" in s["reopen"]
