"""Fusion Planner pins. G054 -- multi-domain topology, placement, and
transfer-vs-recompute planning on top of humf.py's existing move-vs-recompute
planner. See fusion_planner.py's module docstring for what this extends and
what it deliberately does not implement."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
import fusion_planner as fp  # noqa: E402
from fusion_wire import HEADER_SIZE  # noqa: E402
from humf import MemoryClass  # noqa: E402


def _opt(plan, action):
    return next(o for o in plan.options if o["action"] == action)


# ---------------------------------------------------------------- topology shapes


def test_apple_alone_has_exactly_one_domain_and_no_links():
    t = fp.topology_apple_alone()
    assert set(t.domains) == {"APPLE"}
    assert t.domains["APPLE"] is True
    assert t.neighbors("APPLE") == []


def test_apple_plus_one_spark_shape():
    t = fp.topology_apple_plus_sparks(1)
    assert set(t.domains) == {"APPLE", "SPARK0"}
    assert t.domains["SPARK0"] is False
    link = t.link("APPLE", "SPARK0")
    assert link is not None and link.physical is False


def test_apple_plus_two_sparks_shape_has_a_spark_superdomain_link():
    t = fp.topology_apple_plus_sparks(2)
    assert set(t.domains) == {"APPLE", "SPARK0", "SPARK1"}
    link = t.link("SPARK0", "SPARK1")
    assert link is not None
    assert "SUPERDOMAIN" in link.note


def test_apple_spark_egpu_shape_egpu_only_attaches_to_apple():
    t = fp.topology_apple_spark_egpu()
    assert set(t.domains) == {"APPLE", "SPARK0", "EGPU0"}
    assert t.link("APPLE", "EGPU0") is not None
    assert t.link("SPARK0", "EGPU0") is None


def test_no_hardcoded_max_node_count_five_sparks_full_mesh():
    """The task's explicit requirement: no ceiling on node count. Five Sparks
    all get an Apple link AND a full SPARK_SUPERDOMAIN mesh with each other,
    using the exact same code path that handles one or two."""
    t = fp.topology_apple_plus_sparks(5)
    sparks = [f"SPARK{i}" for i in range(5)]
    assert set(t.domains) == {"APPLE", *sparks}
    for i in range(5):
        assert t.link("APPLE", sparks[i]) is not None
        for j in range(i + 1, 5):
            assert t.link(sparks[i], sparks[j]) is not None


def test_duplicate_domain_refused():
    t = fp.topology_apple_alone()
    with pytest.raises(fp.FusionPlannerError, match="already"):
        t.add_domain("APPLE", physical=True)


def test_link_to_unknown_domain_refused():
    t = fp.topology_apple_alone()
    with pytest.raises(fp.FusionPlannerError, match="not a domain"):
        t.add_link("APPLE", "GHOST", bandwidth_gb_s=1.0, latency_s=0.0, physical=False)


def test_duplicate_link_refused():
    t = fp.topology_apple_plus_sparks(1)
    with pytest.raises(fp.FusionPlannerError, match="already exists"):
        t.add_link("APPLE", "SPARK0", bandwidth_gb_s=1.0, latency_s=0.0, physical=False)


# --------------------------------------------------------- shortest_path / SIMULATED law


def test_route_within_the_same_domain_is_zero_cost():
    t = fp.topology_apple_alone()
    r = t.shortest_path("APPLE", "APPLE", 1 << 30)
    assert r.total_time_s == 0.0
    assert r.hops == ()


def test_all_physical_hops_route_is_measured():
    """Positive control on the SIMULATED law: a route over links that are
    ALL physical=True is stamped MEASURED, matching humf._transfer_cost's
    own rule (physical AND physical -> MEASURED)."""
    t = fp.Topology()
    t.add_domain("APPLE", physical=True)
    t.add_domain("APPLE2", physical=True)
    t.add_link("APPLE", "APPLE2", bandwidth_gb_s=589.73, latency_s=0.0, physical=True)
    r = t.shortest_path("APPLE", "APPLE2", 1 << 20)
    assert r.cost_provenance == "MEASURED"


def test_a_single_simulated_hop_taints_the_whole_route_simulated():
    """Negative control: no Spark hardware exists on this machine, so ANY
    route that touches a Spark link must never be laundered as MEASURED."""
    t = fp.topology_apple_plus_sparks(1)
    r = t.shortest_path("APPLE", "SPARK0", 1 << 20)
    assert r.cost_provenance == "SIMULATED"


def test_unreachable_domain_raises_rather_than_returning_infinity():
    t = fp.topology_apple_alone()
    t.add_domain("ISLAND", physical=False)
    with pytest.raises(fp.FusionPlannerError, match="no route"):
        t.shortest_path("APPLE", "ISLAND", 100)


def test_shortest_path_refuses_a_domain_not_in_the_topology():
    t = fp.topology_apple_alone()
    with pytest.raises(fp.FusionPlannerError, match="not a domain"):
        t.shortest_path("APPLE", "GHOST", 100)


# ------------------------------------------- Spark superdomain routing: BOTH directions


def test_spark_to_spark_does_not_route_through_apple_when_direct_is_better():
    """Positive: the default topology's SPARK_SUPERDOMAIN link (100 GB/s) is
    far faster than relaying through Apple (12 GB/s each way), so the
    planner must use the direct one-hop path."""
    t = fp.topology_apple_plus_sparks(2)
    r = t.shortest_path("SPARK0", "SPARK1", 10 << 30)
    assert r.path == ("SPARK0", "SPARK1")
    assert len(r.hops) == 1
    via_apple = t.shortest_path("SPARK0", "APPLE", 10 << 30).total_time_s + \
        t.shortest_path("APPLE", "SPARK1", 10 << 30).total_time_s
    assert r.total_time_s < via_apple


def test_spark_to_spark_routes_through_apple_when_the_direct_link_is_worse():
    """Negative: one direction alone proves nothing about a shortest-path
    chooser. Here the direct SPARK0<->SPARK1 link is deliberately degraded
    below the two-hop relay through Apple, and the planner must choose the
    relay -- Dijkstra earns its keep, not a hardcoded 'prefer direct' rule."""
    t = fp.Topology()
    t.add_domain("APPLE", physical=True)
    t.add_domain("SPARK0", physical=False)
    t.add_domain("SPARK1", physical=False)
    t.add_link("APPLE", "SPARK0", bandwidth_gb_s=12.0, latency_s=2.5e-4, physical=False)
    t.add_link("APPLE", "SPARK1", bandwidth_gb_s=12.0, latency_s=2.5e-4, physical=False)
    t.add_link("SPARK0", "SPARK1", bandwidth_gb_s=0.001, latency_s=1.0, physical=False,
               note="deliberately degraded direct link")
    r = t.shortest_path("SPARK0", "SPARK1", 1 << 20)
    assert r.path == ("SPARK0", "APPLE", "SPARK1")
    assert len(r.hops) == 2


# --------------------------------------------------------------------- placement


def _obj(identity, mc, gran, nbytes, home=None, consumers=()):
    return fp.SemanticObject(identity, mc, gran, nbytes, home_hint=home, consumers=consumers)


def test_immutable_organ_is_replicated_into_every_remote_consumer():
    t = fp.topology_apple_plus_sparks(1)
    objs = [_obj("W", MemoryClass.IMMUTABLE_WEIGHTS, fp.Granularity.ORGAN, 1 << 30,
                 home="APPLE", consumers=("APPLE", "SPARK0"))]
    placements = fp.place_objects(t, objs)
    p = placements["W"]
    assert p.home == "APPLE"
    assert p.replicas == ("SPARK0",)


def test_immutable_tensor_is_not_eagerly_replicated_the_contrast_case():
    """Same memory class, same remote consumer, ONLY the granularity
    differs -- TENSOR stays single-copy, exactly the ping-pong-avoidance
    preference the task names."""
    t = fp.topology_apple_plus_sparks(1)
    objs = [_obj("w_tile", MemoryClass.IMMUTABLE_WEIGHTS, fp.Granularity.TENSOR, 4096,
                 home="APPLE", consumers=("APPLE", "SPARK0"))]
    placements = fp.place_objects(t, objs)
    assert placements["w_tile"].replicas == ()


def test_kv_state_never_replicated_even_with_a_remote_consumer():
    t = fp.topology_apple_plus_sparks(1)
    objs = [_obj("KV", MemoryClass.KV_STATE, fp.Granularity.LAYER_GROUP, 1 << 20,
                 home="APPLE", consumers=("APPLE", "SPARK0"))]
    placements = fp.place_objects(t, objs)
    assert placements["KV"].replicas == ()
    assert "mutable" in placements["KV"].reason


def test_recurrent_state_never_replicated_either():
    t = fp.topology_apple_plus_sparks(1)
    objs = [_obj("REC", MemoryClass.RECURRENT_STATE, fp.Granularity.MODEL, 1 << 20,
                 home="APPLE", consumers=("APPLE", "SPARK0"))]
    placements = fp.place_objects(t, objs)
    assert placements["REC"].replicas == ()


def test_immutable_object_with_no_remote_consumer_stays_single_copy():
    t = fp.topology_apple_plus_sparks(1)
    objs = [_obj("W", MemoryClass.IMMUTABLE_WEIGHTS, fp.Granularity.MODEL, 1 << 30,
                 home="APPLE", consumers=("APPLE",))]
    placements = fp.place_objects(t, objs)
    assert placements["W"].replicas == ()


def test_default_home_is_the_first_domain_when_no_hint_is_given():
    t = fp.topology_apple_alone()
    objs = [_obj("X", MemoryClass.METADATA, fp.Granularity.TENSOR, 8)]
    placements = fp.place_objects(t, objs)
    assert placements["X"].home == "APPLE"


def test_placement_refuses_an_unknown_home_domain():
    t = fp.topology_apple_alone()
    objs = [_obj("X", MemoryClass.METADATA, fp.Granularity.TENSOR, 8, home="GHOST")]
    with pytest.raises(fp.FusionPlannerError, match="not in this topology"):
        fp.place_objects(t, objs)


def test_placement_refuses_an_unknown_consumer_domain():
    t = fp.topology_apple_alone()
    objs = [_obj("W", MemoryClass.IMMUTABLE_WEIGHTS, fp.Granularity.ORGAN, 1 << 20,
                 home="APPLE", consumers=("GHOST",))]
    with pytest.raises(fp.FusionPlannerError, match="not in this topology"):
        fp.place_objects(t, objs)


def test_place_objects_refuses_an_empty_topology():
    with pytest.raises(fp.FusionPlannerError, match="empty topology"):
        fp.place_objects(fp.Topology(), [_obj("X", MemoryClass.METADATA, fp.Granularity.TENSOR, 8)])


# --------------------------------------------------------- the planner's choice
#
# Numbers below were checked against fusion_planner.plan_dependency's actual
# output before being pinned (topology_apple_plus_sparks(1) defaults: 12
# GB/s, 2.5e-4s latency), specifically BECAUSE MOVE_COMPUTE is offered
# unconditionally: with output_bytes=0 its cost is ~one link latency
# (dispatching a 42-byte fusion_wire packet), so it is cheap enough to beat
# a plain MOVE_DATA whenever the object is not tiny. Every scenario below is
# built to make ONE specific action win or lose ON PURPOSE, not by accident
# of that baseline.


def _t():
    return fp.topology_apple_plus_sparks(1)


def test_already_resident_when_home_equals_need_domain():
    q = fp.DependencyQuery("X", "APPLE", "APPLE", nbytes=100, memory_class=MemoryClass.SCRATCH)
    plan = fp.plan_dependency(_t(), q)
    assert plan.action == "ALREADY_RESIDENT"
    assert plan.cost_s == 0.0


def test_mutable_class_never_offers_replicate_as_an_option_at_all():
    q = fp.DependencyQuery("KV", "APPLE", "SPARK0", nbytes=1 << 20, memory_class=MemoryClass.KV_STATE)
    plan = fp.plan_dependency(_t(), q)
    assert "REPLICATE" not in {o["action"] for o in plan.options}


# MOVE_DATA -----------------------------------------------------------------

def test_move_data_wins_a_tiny_object_undercuts_move_computes_dispatch_floor():
    """MOVE_COMPUTE's floor is one dispatch packet (HEADER_SIZE=42 bytes); an
    object smaller than that transfers for less than the dispatch itself
    costs, so plain MOVE_DATA wins outright."""
    q = fp.DependencyQuery("X", "APPLE", "SPARK0", nbytes=1, memory_class=MemoryClass.KV_STATE)
    plan = fp.plan_dependency(_t(), q)
    assert plan.action == "MOVE_DATA"
    assert _opt(plan, "MOVE_DATA")["cost_s"] < _opt(plan, "MOVE_COMPUTE")["cost_s"]


def test_move_data_loses_to_a_cheap_recompute():
    q = fp.DependencyQuery("ACT", "APPLE", "SPARK0", nbytes=1 << 30,
                           memory_class=MemoryClass.ACTIVATIONS, recompute_cost_s=1e-6)
    plan = fp.plan_dependency(_t(), q)
    assert plan.action == "RECOMPUTE"
    assert _opt(plan, "MOVE_DATA")["cost_s"] > plan.cost_s


# MOVE_COMPUTE ----------------------------------------------------------------

def test_move_compute_wins_a_huge_object_with_a_tiny_result():
    q = fp.DependencyQuery("W", "APPLE", "SPARK0", nbytes=1 << 34,
                           memory_class=MemoryClass.KV_STATE)   # output_bytes=0 default
    plan = fp.plan_dependency(_t(), q)
    assert plan.action == "MOVE_COMPUTE"
    assert plan.cost_s < _opt(plan, "MOVE_DATA")["cost_s"]


def test_move_compute_loses_when_the_result_is_bigger_than_the_input():
    q = fp.DependencyQuery("X", "APPLE", "SPARK0", nbytes=1,
                           memory_class=MemoryClass.KV_STATE, output_bytes=1 << 30)
    plan = fp.plan_dependency(_t(), q)
    assert plan.action == "MOVE_DATA"
    assert _opt(plan, "MOVE_COMPUTE")["cost_s"] > plan.cost_s


# RECOMPUTE -------------------------------------------------------------------

def test_recompute_wins_when_cheap():
    q = fp.DependencyQuery("ACT", "APPLE", "SPARK0", nbytes=1 << 30,
                           memory_class=MemoryClass.ACTIVATIONS, recompute_cost_s=1e-6)
    plan = fp.plan_dependency(_t(), q)
    assert plan.action == "RECOMPUTE"


def test_recompute_loses_when_expensive():
    q = fp.DependencyQuery("ACT", "APPLE", "SPARK0", nbytes=1 << 30,
                           memory_class=MemoryClass.ACTIVATIONS, recompute_cost_s=5.0)
    plan = fp.plan_dependency(_t(), q)
    assert plan.action != "RECOMPUTE"
    assert _opt(plan, "RECOMPUTE")["cost_s"] > plan.cost_s


# REPACK ------------------------------------------------------------------

def test_repack_wins_when_it_shrinks_the_transfer_enough():
    q = fp.DependencyQuery("ACT", "APPLE", "SPARK0", nbytes=1 << 30, output_bytes=1 << 30,
                           memory_class=MemoryClass.ACTIVATIONS,
                           repack_bytes=1 << 20, repack_cost_s=0.001)
    plan = fp.plan_dependency(_t(), q)
    assert plan.action == "REPACK"
    assert plan.cost_s < _opt(plan, "MOVE_DATA")["cost_s"]
    assert plan.cost_s < _opt(plan, "MOVE_COMPUTE")["cost_s"]


def test_repack_loses_when_the_repack_itself_is_too_expensive():
    q = fp.DependencyQuery("ACT", "APPLE", "SPARK0", nbytes=1 << 20,
                           memory_class=MemoryClass.ACTIVATIONS,
                           repack_bytes=1024, repack_cost_s=10.0)
    plan = fp.plan_dependency(_t(), q)
    assert plan.action != "REPACK"
    assert _opt(plan, "REPACK")["cost_s"] > plan.cost_s


# REPLICATE -----------------------------------------------------------------

def test_replicate_wins_a_cost_tie_against_move_data_for_immutable_data():
    """REPLICATE and MOVE_DATA move the identical bytes at the identical
    cost -- the permanent law (fastest transfer is transfer proven
    unnecessary) says the tie is broken toward keeping the data, since it
    can never go stale."""
    q = fp.DependencyQuery("W", "APPLE", "SPARK0", nbytes=1 << 20, output_bytes=1 << 20,
                           memory_class=MemoryClass.IMMUTABLE_WEIGHTS)
    plan = fp.plan_dependency(_t(), q)
    assert plan.action == "REPLICATE"
    assert _opt(plan, "REPLICATE")["cost_s"] == _opt(plan, "MOVE_DATA")["cost_s"]


def test_replicate_loses_to_a_cheap_recompute():
    q = fp.DependencyQuery("W", "APPLE", "SPARK0", nbytes=1 << 30,
                           memory_class=MemoryClass.IMMUTABLE_WEIGHTS, recompute_cost_s=1e-6)
    plan = fp.plan_dependency(_t(), q)
    assert plan.action == "RECOMPUTE"
    assert _opt(plan, "REPLICATE")["cost_s"] > plan.cost_s


# WAIT ------------------------------------------------------------------

def test_wait_wins_when_the_in_flight_transfer_is_almost_done():
    q = fp.DependencyQuery("ACT", "APPLE", "SPARK0", nbytes=1 << 20,
                           memory_class=MemoryClass.ACTIVATIONS, in_flight_eta_s=1e-7)
    plan = fp.plan_dependency(_t(), q)
    assert plan.action == "WAIT"


def test_wait_loses_when_the_in_flight_transfer_is_stalled():
    q = fp.DependencyQuery("ACT", "APPLE", "SPARK0", nbytes=1 << 20,
                           memory_class=MemoryClass.ACTIVATIONS, in_flight_eta_s=100.0,
                           recompute_cost_s=1e-6)
    plan = fp.plan_dependency(_t(), q)
    assert plan.action == "RECOMPUTE"
    assert _opt(plan, "WAIT")["cost_s"] > plan.cost_s


# PREFETCH ------------------------------------------------------------------

def test_prefetch_wins_when_fully_hidden_by_the_overlap_window():
    q = fp.DependencyQuery("ACT", "APPLE", "SPARK0", nbytes=1 << 20,
                           memory_class=MemoryClass.ACTIVATIONS, overlap_window_s=10.0)
    plan = fp.plan_dependency(_t(), q)
    assert plan.action == "PREFETCH"
    assert plan.cost_s == 0.0


def test_prefetch_loses_when_there_is_no_overlap_window():
    q = fp.DependencyQuery("ACT", "APPLE", "SPARK0", nbytes=1 << 20,
                           memory_class=MemoryClass.ACTIVATIONS, overlap_window_s=0.0,
                           recompute_cost_s=1e-6)
    plan = fp.plan_dependency(_t(), q)
    assert plan.action == "RECOMPUTE"
    assert _opt(plan, "PREFETCH")["cost_s"] > plan.cost_s


# tie-break law ---------------------------------------------------------------

def test_tie_break_prefers_avoiding_transfer_entirely():
    """THE permanent law, checked directly: an exact cost tie between
    MOVE_DATA and RECOMPUTE is broken toward RECOMPUTE, which moves zero
    bytes -- not toward whichever happened to be built first."""
    t = _t()
    move_route = t.shortest_path("APPLE", "SPARK0", 1 << 20)
    q = fp.DependencyQuery("ACT", "APPLE", "SPARK0", nbytes=1 << 20, output_bytes=1 << 20,
                           memory_class=MemoryClass.ACTIVATIONS,
                           recompute_cost_s=move_route.total_time_s)
    plan = fp.plan_dependency(t, q)
    assert plan.action == "RECOMPUTE"
    assert _opt(plan, "MOVE_DATA")["cost_s"] == pytest.approx(plan.cost_s)


def test_all_provenance_is_simulated_over_a_spark_link():
    q = fp.DependencyQuery("KV", "APPLE", "SPARK0", nbytes=1 << 20, memory_class=MemoryClass.KV_STATE)
    plan = fp.plan_dependency(_t(), q)
    assert plan.rests_on_simulated_numbers is True
    assert all(o["cost_provenance"] == "SIMULATED" for o in plan.options
               if o["action"] != "RECOMPUTE")


# ------------------------------------------------------------------ collectives


def test_collective_is_direct_for_two_participants_no_manufactured_distinction():
    t = fp.topology_apple_plus_sparks(1)
    plan = fp.plan_collective(t, fp.CollectiveOp.BROADCAST, ["APPLE", "SPARK0"],
                              message_bytes=1 << 20)
    assert plan.algorithm == "DIRECT"
    assert plan.crossover_bytes is None
    assert plan.ring_cost_s == plan.tree_cost_s == plan.cost_s


def test_collective_crossover_is_computed_and_flips_the_algorithm_choice():
    """The task's explicit requirement: report the crossover as a computed
    quantity, and prove it actually governs the choice by sampling on both
    sides of it -- not just asserting a formula in isolation."""
    t = fp.topology_apple_plus_sparks(3)   # APPLE + 3 Sparks -> p=4 group
    domains = ["APPLE", "SPARK0", "SPARK1", "SPARK2"]
    probe = fp.plan_collective(t, fp.CollectiveOp.ALLREDUCE, domains, message_bytes=1)
    assert probe.crossover_bytes is not None and probe.crossover_bytes > 0

    small = fp.plan_collective(t, fp.CollectiveOp.ALLREDUCE, domains,
                               message_bytes=int(probe.crossover_bytes * 0.1))
    large = fp.plan_collective(t, fp.CollectiveOp.ALLREDUCE, domains,
                               message_bytes=int(probe.crossover_bytes * 10))
    assert small.algorithm == "TREE"
    assert large.algorithm == "RING"
    assert small.reason and large.reason   # the planner REPORTS which and why


def test_collective_degenerate_p_where_one_algorithm_dominates_everywhere():
    """p=3 is a genuine edge case for this cost model (ring's latency-hop
    count equals tree's exactly), so ring dominates for every positive
    message size and there is no crossover. Reported as None, not a bogus
    or divide-by-zero number."""
    t = fp.topology_apple_plus_sparks(2)
    domains = ["APPLE", "SPARK0", "SPARK1"]
    plan = fp.plan_collective(t, fp.CollectiveOp.BROADCAST, domains, message_bytes=1 << 30)
    assert plan.crossover_bytes is None
    assert plan.algorithm == "RING"


def test_collective_provenance_is_simulated_with_a_spark_participant():
    t = fp.topology_apple_plus_sparks(1)
    plan = fp.plan_collective(t, fp.CollectiveOp.BROADCAST, ["APPLE", "SPARK0"],
                              message_bytes=1 << 20)
    assert plan.cost_provenance == "SIMULATED"


def test_collective_provenance_is_measured_when_every_hop_is_physical():
    t = fp.Topology()
    for name in ("A", "B", "C"):
        t.add_domain(name, physical=True)
    for a, b in (("A", "B"), ("B", "C"), ("C", "A")):
        t.add_link(a, b, bandwidth_gb_s=589.73, latency_s=0.0, physical=True)
    plan = fp.plan_collective(t, fp.CollectiveOp.ALLGATHER, ["A", "B", "C"], message_bytes=1 << 20)
    assert plan.cost_provenance == "MEASURED"


def test_collective_alpha_reflects_an_unconnected_pair_routed_through_apple():
    """Topology genuinely drives the collective's cost, not just the
    point-to-point planner: SPARK0<->EGPU0 has no direct link, so that ring
    edge must relay through APPLE, and the group's alpha is strictly worse
    than an all-direct triangle built from identical per-hop numbers."""
    te = fp.topology_apple_spark_egpu(n_sparks=1)
    with_egpu = fp.plan_collective(te, fp.CollectiveOp.ALLREDUCE,
                                   ["APPLE", "SPARK0", "EGPU0"], message_bytes=1)

    t2 = fp.Topology()
    for name in ("APPLE", "SPARK0", "EGPU0"):
        t2.add_domain(name, physical=False)
    for a, b in (("APPLE", "SPARK0"), ("SPARK0", "EGPU0"), ("EGPU0", "APPLE")):
        t2.add_link(a, b, bandwidth_gb_s=12.0, latency_s=2.5e-4, physical=False)
    all_direct = fp.plan_collective(t2, fp.CollectiveOp.ALLREDUCE,
                                    ["APPLE", "SPARK0", "EGPU0"], message_bytes=1)

    assert with_egpu.alpha_s > all_direct.alpha_s


@pytest.mark.parametrize("op", list(fp.CollectiveOp))
def test_every_collective_op_is_plannable(op):
    t = fp.topology_apple_plus_sparks(2)
    plan = fp.plan_collective(t, op, ["APPLE", "SPARK0", "SPARK1"], message_bytes=1 << 20)
    assert plan.op == op.value
    assert plan.cost_s >= 0.0


def test_collective_needs_at_least_two_participants():
    t = fp.topology_apple_alone()
    with pytest.raises(fp.FusionPlannerError, match="at least 2"):
        fp.plan_collective(t, fp.CollectiveOp.BROADCAST, ["APPLE"], message_bytes=100)


def test_collective_refuses_a_duplicate_participant():
    t = fp.topology_apple_plus_sparks(1)
    with pytest.raises(fp.FusionPlannerError, match="duplicate"):
        fp.plan_collective(t, fp.CollectiveOp.BROADCAST, ["APPLE", "APPLE"], message_bytes=100)


def test_collective_refuses_an_unknown_participant_domain():
    t = fp.topology_apple_plus_sparks(1)
    with pytest.raises(fp.FusionPlannerError, match="not a domain"):
        fp.plan_collective(t, fp.CollectiveOp.BROADCAST, ["APPLE", "GHOST"], message_bytes=100)


def test_dispatch_packet_size_matches_fusion_wires_real_header_not_an_invented_constant():
    """MOVE_COMPUTE's dispatch cost is not a made-up number -- it is
    literally fusion_wire.HEADER_SIZE, the same 42-byte command packet the
    parallel protocol lane already defined."""
    assert fp._DISPATCH_PACKET_BYTES == HEADER_SIZE == 42
