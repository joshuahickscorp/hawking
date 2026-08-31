"""Nine classes, discriminators written before the runs, and one already dead.

S025 §13 is explicit about the trap: do NOT call the effect "serial dependency"
merely because concurrency helped. The tree exists so that a class can be killed
by evidence rather than by narrative, and class H - the only one that would have
made "serial dependency" literally true at the dispatch grain - is the one the
DAG already killed.
"""
from __future__ import annotations

import pytest

from tools.future import multistream_capacity_tree as tree


def test_a_class_without_a_cheap_discriminator_is_refused():
    with pytest.raises(tree.TreeRefused, match="cheap discriminator"):
        tree._cls(id="x", claim="c", max_payoff_ms=None, discriminator="short",
                  kills_if="a", reopens_if="b")


def test_kill_and_reopen_criteria_are_mandatory():
    long = "a discriminator long enough to clear the forty character bar here"
    with pytest.raises(tree.TreeRefused, match="BEFORE the run"):
        tree._cls(id="x", claim="c", max_payoff_ms=None, discriminator=long,
                  kills_if="", reopens_if="b")


def test_a_kill_must_name_its_evidence():
    long = "a discriminator long enough to clear the forty character bar here"
    with pytest.raises(tree.TreeRefused, match="without naming the evidence"):
        tree._cls(id="x", claim="c", max_payoff_ms=None, discriminator=long,
                  kills_if="a", reopens_if="b", status=tree.KILLED)


def test_all_nine_classes_are_present_and_each_can_be_reopened():
    cs = tree.classes()
    assert len(cs) == 9
    for c in cs:
        assert c["reopens_if"], c["id"]
        assert c["kills_if"], c["id"]


def test_class_h_is_dead_and_cites_the_dag():
    h = next(c for c in tree.classes() if c["id"].startswith("H_"))
    assert h["status"] == tree.KILLED
    assert "SINGLE_TOKEN_PARALLEL_SLACK" in h["killed_by"]
    assert "0 artificial barriers" in h["killed_by"]


def test_the_tree_refuses_to_claim_h_dead_if_the_dag_disagrees(monkeypatch, tmp_path):
    """The kill is only as good as the receipt behind it."""
    fake = tmp_path / "slack.json"
    fake.write_text('{"theoretically_overlapable_ns": 5}')
    monkeypatch.setattr(tree, "DAG_REL", str(fake.relative_to(tree.REPO))
                        if str(fake).startswith(str(tree.REPO)) else tree.DAG_REL)
    if str(fake).startswith(str(tree.REPO)):
        with pytest.raises(tree.TreeRefused, match="class H is NOT dead"):
            tree.summary()


def test_the_parent_class_exists_to_be_split_not_tested():
    a = next(c for c in tree.classes() if c["id"].startswith("A_"))
    assert "SPLIT, not tested" in a["cheapest_discriminator"]
    assert "redundant rather than false" in a["kills_if"]


def test_payoffs_are_null_rather_than_invented():
    for c in tree.classes():
        assert c["max_payoff_ms"] is None, (
            f"{c['id']} carries a payoff number nothing has bounded yet"
        )


def test_the_summary_refuses_to_name_the_cause():
    s = tree.summary()
    assert s["n_killed"] == 1
    assert s["n_open"] + s["n_sharpened"] + s["n_blocked_on_tooling"] == 8
    assert "concurrency helping is the OBSERVATION" in s["do_not_call_it_serial_dependency"]


def test_the_next_cheapest_are_the_within_kernel_classes():
    """The DAG pointed the search at the kernels; the tree must follow.

    NOT a fixed list - as classes resolve the frontier moves, and pinning the
    day's ordering would fail every time the campaign advanced. What must hold is
    that the next cheapest are still WITHIN-KERNEL classes and never the killed
    executor-serialization one.
    """
    nxt = tree.summary()["next_cheapest"]
    assert nxt, "a tree with open classes must name what to run next"
    assert all(not x.startswith("H_") for x in nxt), "the killed class cannot return"
    within_kernel = ("B_", "C_", "D_", "E_", "F_", "G_", "I_")
    assert all(x.startswith(within_kernel) for x in nxt)


def test_class_d_is_sharpened_by_the_alu_roofline_not_killed():
    """ARM A strips arithmetic at IDENTICAL bytes and MLP jumps 1.5089x.

    That is the opposite of this class's kill criterion, so it narrows rather
    than dying - and what it narrows to is a NUMBER: about 1.5x available at
    constant bytes, inside the 1.24-1.60x the concurrency ladder measured from a
    completely different direction.
    """
    d = next(c for c in tree.classes() if c["id"].startswith("D_"))
    assert d["status"] == tree.SHARPENED
    assert d["killed_by"] is None
    ev = d["sharpened_by"]
    assert "MLP_ALU_ROOFLINE" in ev
    assert "1.5089x" in ev and "1.0427" in ev
    assert "ratios hold under load" in ev


def test_a_sharpened_class_must_name_its_evidence():
    long = "a discriminator long enough to clear the forty character bar here"
    with pytest.raises(tree.TreeRefused, match="SHARPENED without naming"):
        tree._cls(id="x", claim="c", max_payoff_ms=None, discriminator=long,
                  kills_if="a", reopens_if="b", status=tree.SHARPENED)


def test_the_two_independent_probes_agree_on_the_magnitude():
    """1.5x from stripping arithmetic, 1.24-1.60x from a second stream."""
    d = next(c for c in tree.classes() if c["id"].startswith("D_"))
    assert "1.24-1.60x" in d["sharpened_by"]
    assert "different direction" in d["sharpened_by"]


def test_the_summary_counts_sharpened_separately_from_killed():
    """Conflating them would let a narrowed class read as a dead one."""
    s = tree.summary()
    assert s["n_killed"] == 1
    assert "D_instruction_dependency_chain" in s["sharpened"]
    assert s["n_sharpened"] >= 1 and s["n_open"] >= 1


def test_class_b_was_already_swept_and_is_not_flat():
    """GEOMETRY_TABLE ran this discriminator: 11 shapes, 8 distinct winners."""
    b = next(c for c in tree.classes() if c["id"].startswith("B_"))
    assert b["status"] == tree.SHARPENED
    ev = b["sharpened_by"]
    assert "GEOMETRY_TABLE" in ev
    assert "flat is FALSE" in ev and "EIGHT distinct launch winners" in ev
    assert "must not be assumed" in ev, "the open sub-question must stay open"


def test_class_c_is_blocked_on_tooling_not_silently_open():
    """registers_per_thread is null because the reflection does not carry it."""
    c = next(x for x in tree.classes() if x["id"].startswith("C_"))
    assert c["status"] == tree.BLOCKED_ON_TOOLING
    ev = c["sharpened_by"]
    assert "registers_per_thread NULL" in ev
    assert "xcrun metal is not on PATH" in ev
    assert "neither confirmed nor killed" in ev
    assert "toolchain that reports the register count" in c["reopens_if"]


def test_blocked_on_tooling_also_requires_its_evidence():
    long = "a discriminator long enough to clear the forty character bar here"
    with pytest.raises(tree.TreeRefused, match="without naming"):
        tree._cls(id="x", claim="c", max_payoff_ms=None, discriminator=long,
                  kills_if="a", reopens_if="b", status=tree.BLOCKED_ON_TOOLING)


def test_the_four_statuses_are_counted_apart():
    """Killed, sharpened, blocked-on-tooling and open are different things."""
    s = tree.summary()
    assert s["n_killed"] + s["n_sharpened"] + s["n_blocked_on_tooling"] + s["n_open"] == 9
    assert s["n_killed"] == 1 and s["n_blocked_on_tooling"] == 1


def test_the_next_cheapest_moved_on_as_classes_resolved():
    """B and D are no longer the frontier; E, F and G are."""
    nxt = tree.summary()["next_cheapest"]
    assert not any(x.startswith(("B_", "D_", "C_")) for x in nxt)
    assert nxt[0].startswith("E_memory_level")


def test_arithmetic_is_a_throughput_term_not_a_dependency_term():
    """Removing arithmetic buys 1.51x; making it parallel buys nothing.

    ARM A strips the arithmetic at identical bytes: 1.5089x. The ILP ladder
    splits the same 16-FMA inner loop into 2, 4 and 8 independent accumulator
    chains, all bit-identical: 328.5, 325.5, 327.4 GB/s - flat inside 1%. The
    machine is not stalling on the chain, it is spending issue slots.
    """
    d = next(c for c in tree.classes() if c["id"].startswith("D_"))
    ev = d["sharpened_by"]
    assert "MLP_ISSUE_RATE_LADDER" in ev
    assert "328.5" in ev and "327.4" in ev and "flat inside 1%" in ev
    assert "THROUGHPUT" in ev and "not dependency-chain LATENCY" in ev
    assert "1.3333" in ev, "the remedy must point at the decode tax"


def test_class_e_is_protected_from_its_lookalike():
    """The ILP ladder varies accumulator chains, not the load pipeline.

    Closing E on it would be the easiest available mistake, so the class names
    the receipt that does NOT answer it.
    """
    e = next(c for c in tree.classes() if c["id"].startswith("E_"))
    assert e["status"] == tree.OPEN
    assert e["sharpened_by"] is None
    guard = e["not_answered_by"]
    assert "MLP_ISSUE_RATE_LADDER" in guard
    assert "ACCUMULATOR CHAINS" in guard
    assert "not the load pipeline" in guard


def test_only_e_carries_a_lookalike_guard_so_far():
    guarded = [c["id"] for c in tree.classes() if c.get("not_answered_by")]
    assert guarded == ["E_memory_level_parallelism"]
