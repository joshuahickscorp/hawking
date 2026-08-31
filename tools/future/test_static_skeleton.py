"""Tests for the static skeleton + dynamic-slot validator.

The load-bearing test is the negative control: an edge set that depends on
a routed activation VALUE must be refused, while the same shape indexed
only by EXPERT_ID must be accepted. A guard nobody has watched fail is
not a guard.
"""
from __future__ import annotations

import json

import pytest

from tools.future import static_skeleton as sk
from tools.future._common import RECEIPTS, HardwareClaimError, _assert_no_hardware_claims


def test_slot_kinds_are_exactly_the_five():
    assert sk.SLOT_KINDS == (
        "EXPERT_ID",
        "TOKEN_POSITION",
        "SAMPLING",
        "REPRESENTATION_FRAGMENT",
        "STATE_VALUE",
    )
    assert len(sk.SLOT_KIND_SET) == 5


def test_validate_accepts_expert_id_indexed_skeleton():
    result = sk.validate(sk.legal_expert_id_skeleton())
    assert result.accepted, result.errors
    assert result.errors == ()


def test_validate_rejects_activation_value_gated_edges():
    """NEGATIVE CONTROL: edge set depends on a routed activation VALUE.

    Not an expert id. The validator must fire. If this assertion is ever
    inverted, graph replay is unguarded.
    """
    illegal = sk.illegal_activation_gated_skeleton()
    result = sk.validate(illegal)
    assert result.accepted is False
    assert result.errors, "refusal must name the offending edge"
    blob = " ".join(result.errors)
    assert "VALUE_GATED" in blob
    assert "routed_activation_value" in blob
    assert "cannot be replayed" in blob
    # The legal twin still passes, so the validator is discriminating
    # and not just always-refusing.
    assert sk.validate(sk.legal_expert_id_skeleton()).accepted is True


def test_validate_rejects_activation_gated_dispatch_count():
    result = sk.validate(sk.illegal_activation_gated_dispatch_count())
    assert result.accepted is False
    blob = " ".join(result.errors)
    assert "dispatch_count_gated_on" in blob
    assert "routed_activation_value" in blob


def test_require_valid_raises_on_illegal():
    with pytest.raises(sk.SkeletonRefused) as ei:
        sk.require_valid(sk.illegal_activation_gated_skeleton())
    assert "VALUE_GATED" in str(ei.value)


def test_validate_rejects_unknown_slot_kind():
    bad = sk.Skeleton(
        name="unknown_kind",
        slots=(sk.Slot(name="foo", kind="ACTIVATION_MAGNITUDE", lo=0, hi=7),),
        nodes=(
            sk.Node(
                id="n",
                kind="op",
                existence="SLOT_INDEXED",
                slot="foo",
                dispatch_count=1,
            ),
        ),
        edges=(),
    )
    result = sk.validate(bad)
    assert result.accepted is False
    assert any("ACTIVATION_MAGNITUDE" in e for e in result.errors)


def test_validate_rejects_unbounded_slot():
    bad = sk.Skeleton(
        name="unbounded",
        slots=(sk.Slot(name="expert_id", kind="EXPERT_ID"),),
        nodes=(),
        edges=(),
    )
    result = sk.validate(bad)
    assert result.accepted is False
    assert any("not bounded" in e for e in result.errors)


def test_validate_rejects_slot_indexed_without_declared_slot():
    bad = sk.Skeleton(
        name="missing_slot",
        slots=(),
        nodes=(
            sk.Node(
                id="bank",
                kind="expert_body",
                existence="SLOT_INDEXED",
                slot="expert_id",
                dispatch_count=1,
            ),
        ),
        edges=(),
    )
    result = sk.validate(bad)
    assert result.accepted is False
    assert any("undeclared slot" in e for e in result.errors)


def test_validate_accepts_all_five_slot_kinds_together():
    skeleton = sk.Skeleton(
        name="all_five",
        slots=(
            sk.Slot(name="expert_id", kind="EXPERT_ID", lo=0, hi=511, dispatch_bound=10),
            sk.Slot(name="token_position", kind="TOKEN_POSITION", lo=0, hi=41),
            sk.Slot(name="sampling_mode", kind="SAMPLING", lo=0, hi=3),
            sk.Slot(name="row_window", kind="REPRESENTATION_FRAGMENT", lo=0, hi=127),
            sk.Slot(name="dn_state", kind="STATE_VALUE", shape=(128, 128), dtype="f32"),
        ),
        nodes=(
            sk.Node(id="router", kind="router", dispatch_count=1),
            sk.Node(
                id="expert_bank",
                kind="expert_body",
                existence="SLOT_INDEXED",
                slot="expert_id",
                binds_slots=("row_window",),
                dispatch_count=10,
                dispatch_count_from_slot="expert_id",
            ),
            sk.Node(
                id="recurrent",
                kind="deltanet",
                binds_slots=("token_position", "dn_state"),
                dispatch_count=1,
            ),
            sk.Node(
                id="sample",
                kind="sample",
                binds_slots=("sampling_mode",),
                dispatch_count=1,
            ),
        ),
        edges=(
            sk.Edge(src="router", dst="expert_bank", existence="SLOT_INDEXED", slot="expert_id"),
            sk.Edge(src="expert_bank", dst="recurrent"),
            sk.Edge(src="recurrent", dst="sample"),
        ),
    )
    result = sk.validate(skeleton)
    assert result.accepted, result.errors
    frac = sk.static_fraction(skeleton)
    assert frac["value_gated_units"] == 0
    assert frac["replayable_fraction"] == pytest.approx(1.0)
    assert 0.0 < frac["topology_static_fraction"] < 1.0


def test_static_fraction_counts_gated_edges():
    frac_legal = sk.static_fraction(sk.legal_expert_id_skeleton())
    frac_illegal = sk.static_fraction(sk.illegal_activation_gated_skeleton())
    assert frac_legal["value_gated_units"] == 0
    assert frac_illegal["value_gated_units"] == 2
    assert frac_illegal["replayable_fraction"] < 1.0


def test_longest_path_is_node_count_not_time():
    path = sk.longest_path(sk.legal_expert_id_skeleton())
    assert path["is_dag"] is True
    assert path["unit"] == "nodes"
    assert path["length_nodes"] == 2
    assert "ns" not in path["note"]


def test_skeleton_from_physical_graph_is_static_and_valid():
    pg = {
        "schema": "hcli.physical_graph.v1",
        "model_id": "wrap_test",
        "qualification": "PLAN_ONLY",
        "computation": [
            {"id": "a", "kind": "computation"},
            {"id": "b", "kind": "native_kernel_dispatch", "dispatches_per_sample": 1},
        ],
        "dependencies": [{"from": "a", "to": "b", "kind": "dataflow"}],
    }
    skeleton = sk.skeleton_from_physical_graph(pg)
    result = sk.validate(skeleton)
    assert result.accepted, result.errors
    assert all(n.existence == "STATIC" for n in skeleton.nodes)
    assert all(e.existence == "STATIC" for e in skeleton.edges)
    assert skeleton.slots == ()
    frac = sk.static_fraction(skeleton)
    assert frac["topology_static_fraction"] == 1.0


def test_ledger_layer_46_is_fully_static_when_ledger_loads():
    ledger = sk._load_optional(sk.LEDGER_REL)
    if ledger is None:
        pytest.skip("DISPATCH_LEDGER.json not in git or disk")
    skeleton = sk.skeleton_from_ledger_layer(ledger, 46)
    result = sk.validate(skeleton)
    assert result.accepted, result.errors
    assert len(skeleton.nodes) == 12
    assert all(n.existence == "STATIC" for n in skeleton.nodes)
    assert all(e.existence == "STATIC" for e in skeleton.edges)
    frac = sk.static_fraction(skeleton)
    assert frac["topology_static_fraction"] == 1.0
    assert frac["value_gated_units"] == 0
    kinds = {s.kind for s in skeleton.slots}
    assert "STATE_VALUE" in kinds
    assert "EXPERT_ID" not in kinds
    path = sk.longest_path(skeleton)
    assert path["is_dag"] is True
    assert path["length_nodes"] >= 8
    # Layer 30 is the same DeltaNet 12-node shape.
    l30 = sk.skeleton_from_ledger_layer(ledger, 30)
    assert [n.id for n in l30.nodes] == [n.id for n in skeleton.nodes]


def test_flash_moe_component_is_replayable_not_all_static():
    selection = sk._load_optional(sk.FLASH_ROUTER_SEL_REL)
    expert = sk._load_optional(sk.FLASH_EXPERT_GRAPH_REL)
    if selection is None or expert is None:
        pytest.skip("Flash Noetic receipts not in git or disk")
    skeleton = sk.flash_moe_component_skeleton(selection, expert)
    result = sk.validate(skeleton)
    assert result.accepted, result.errors
    kinds = {s.kind for s in skeleton.slots}
    assert "EXPERT_ID" in kinds
    assert "REPRESENTATION_FRAGMENT" in kinds
    frac = sk.static_fraction(skeleton)
    assert frac["value_gated_units"] == 0
    assert frac["replayable_fraction"] == pytest.approx(1.0)
    assert frac["slot_indexed_units"] > 0
    assert frac["topology_static_fraction"] < 1.0
    bank = next(n for n in skeleton.nodes if n.id == "expert_bank")
    assert bank.existence == "SLOT_INDEXED"
    assert bank.dispatch_count == 10


def test_backends_declare_availability_without_hardware_numbers():
    backends = sk.backend_usability()
    for name in ("METAL", "FPGA", "CUDA", "ANE_GRAPH_SEGMENT"):
        assert name in backends
        row = backends[name]
        assert row["physical_device_authority"] is False
        assert "slot_becomes_indirection" in row
        for kind in sk.SLOT_KINDS:
            assert kind in row["slot_becomes_indirection"]
    _assert_no_hardware_claims(backends)
    assert backends["FPGA"]["physical_board_present"] in (False, None)
    assert backends["ANE_GRAPH_SEGMENT"]["present_on_disk"] is False


def test_build_emits_sealed_receipt():
    out = sk.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "STATIC_SKELETON.json"
    assert doc["schema"] == "hawking.future.static_skeleton.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["slot_kinds"] == list(sk.SLOT_KINDS)
    assert doc["discrimination_selftest"]["guard_watched_failing"] is True
    assert doc["discrimination_selftest"]["legal_expert_id_accepted"] is True
    assert doc["discrimination_selftest"]["illegal_activation_gated_edges_refused"] is True
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    _assert_no_hardware_claims(doc)
    # HardwareClaimError is the boundary: a number in a hardware field
    # must still be refused by write_receipt's checker.
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims({"tps": 12.0})
