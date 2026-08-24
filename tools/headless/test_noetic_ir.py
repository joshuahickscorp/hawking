"""Noetic IR: at least one family expressed and EXECUTED from the IR.

`python3 -m pytest tools/headless -q` must write receipts/headless/NOETIC_IR.json
and exit 0. An IR with no executing node is a schema, not an IR.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools"))

from noetic_ir import (  # noqa: E402
    CANNOT_EXPRESS,
    EXECUTORS,
    IR_KIND,
    MACHINE_SPECIFIC,
    NODE_TYPES,
    PLANTED_SHARED_BASIS_BYTES,
    RECEIPT,
    SCHEMA,
    SEALED_BPW,
    SOURCE_PARAM_COUNT,
    MachineRefusal,
    SharedPool,
    UnexecutableNode,
    account,
    build,
    check_lowering,
    execute_node,
    make_node,
    pack_grouped_absmax_q4,
    planted_basis_node,
    this_machine_stub,
    validate_semantic,
    write_receipt,
)

RECEIPT_DOC = None


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        RECEIPT_DOC = build()
        write_receipt(RECEIPT_DOC)
    return RECEIPT_DOC


def test_harness_writes_receipt():
    doc = receipt()
    assert RECEIPT.is_file(), f"missing {RECEIPT}"
    on_disk = json.loads(RECEIPT.read_text())
    assert on_disk["schema"] == SCHEMA
    assert doc["schema"] == SCHEMA
    assert doc["ir_kind"] == IR_KIND


def test_receipt_names_an_executed_family():
    doc = receipt()
    assert doc["executed_families"], "no family was executed — this is a schema, not an IR"
    assert "grouped_absmax_q4" in doc["executed_families"]
    flag = doc["flagship"]
    assert flag["family"] == "grouped_absmax_q4"
    assert flag["y_from_ir"]
    assert flag["y_direct"]
    assert flag["y_from_ir"] != flag["y_direct"] or flag["max_abs_diff"] == 0.0
    assert flag["match_atol_1e5"] is True
    assert flag["max_abs_diff"] < 1e-5


def test_ir_output_compared_against_direct():
    doc = receipt()
    for ex in doc["executions"]:
        assert "y_from_ir" in ex and "y_direct" in ex, ex["family"]
        assert "max_abs_diff" in ex and "rel_l2" in ex
        assert ex["match_atol_1e5"] is True, (
            f"{ex['family']}: ir {ex['y_from_ir']} vs direct {ex['y_direct']} "
            f"diff={ex['max_abs_diff']}"
        )
        assert len(ex["y_from_ir"]) == len(ex["y_direct"])


def test_self_check_all_true():
    doc = receipt()
    failed = [k for k, v in doc["self_check"].items() if v is not True]
    assert not failed, failed


def test_semantic_side_refuses_machine_fields():
    """G103 teeth: kernel / threadgroup belong in NX, not in the semantic node."""
    doc = receipt()
    teeth = doc["semantic_vs_machine_teeth"]
    assert teeth["machine_field_on_semantic_rejected"] is True
    joined = " ".join(teeth["problems"])
    assert "threadgroup_size" in joined or "kernel" in joined
    for node in doc["graph"]["nodes"]:
        for k, _path in _walk(node["semantic"]):
            assert k.lower() not in MACHINE_SPECIFIC, (
                f"semantic node {node['semantic']['id']} carries machine field {k}"
            )


def _walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def test_machine_lowering_genome_mismatch_is_refused():
    """An NX that could load anywhere has failed. Metal lowerings are not portable."""
    doc = receipt()
    assert "REFUSED" in doc["semantic_vs_machine_teeth"]["metal_lowering_genome_mismatch"]
    with pytest.raises(MachineRefusal):
        check_lowering({
            "kind": "metal_kernel",
            "portable": False,
            "compiled_for_machine_genome": "gpu_cores=40",
            "required_here": "gpu_cores=60",
        })
    # The CPU interpreter IS portable: it is the semantic function, not an NX.
    check_lowering(this_machine_stub())


def test_each_node_type_is_justified_by_an_experiment():
    for kind, spec in NODE_TYPES.items():
        assert spec["experiment"], kind
        assert spec["receipts"], kind
        assert spec["family"]


def test_tensor_train_is_not_a_node():
    assert "tensor_train" not in NODE_TYPES
    assert "tensor_train" not in EXECUTORS
    assert any("tensor_train" in c["what"] for c in CANNOT_EXPRESS)


def test_planted_1gb_moves_bpw_and_cannot_execute():
    """The G103 hole: validate() ok and BPW stuck at 4.252735126866492.

    This IR charges the 1 GB (BPW moves) and refuses to execute it.
    """
    doc = receipt()
    acc = doc["accounting"]
    assert acc["bpw_moved"] is True
    assert acc["planted_bytes"] == PLANTED_SHARED_BASIS_BYTES
    assert acc["bpw_delta"] == pytest.approx(
        8.0 * PLANTED_SHARED_BASIS_BYTES / SOURCE_PARAM_COUNT
    )
    assert acc["after_1gb_shared_basis"]["complete_bpw"] != SEALED_BPW
    assert acc["planted_execute"]
    assert "no interpreter" in acc["planted_execute"]

    pool = SharedPool()
    planted = planted_basis_node(pool)
    with pytest.raises(UnexecutableNode):
        execute_node(planted, np.zeros(1, dtype=np.float32))
    ok, problems = validate_semantic([planted])
    assert ok, problems  # it is a legal SEMANTIC node — accounting, not execution


def test_grouped_absmax_fused_matches_reconstructed_matvec():
    rng = np.random.RandomState(7)
    W = rng.randn(16, 64).astype(np.float32)
    x = rng.randn(64).astype(np.float32)
    node = make_node(
        "unit.q4", "grouped_absmax", pack_grouped_absmax_q4(W),
        lowering=this_machine_stub(),
    )
    y_ir = execute_node(node, x)
    from noetic_ir import reconstruct_grouped_absmax_q4
    y_direct = reconstruct_grouped_absmax_q4(node["payload"]) @ x
    assert np.allclose(y_ir, y_direct, atol=1e-5)
    # Quantisation vs source is a different computation and is NOT required to be ~0.
    y_src = W @ x
    assert np.linalg.norm(y_ir - y_src) / np.linalg.norm(y_src) > 1e-6


def test_low_rank_uv_does_not_form_W():
    """y = L @ (R @ x) equals (L @ R) @ x without the IR path writing W."""
    rng = np.random.RandomState(11)
    L = rng.randn(10, 3).astype(np.float32)
    R = rng.randn(3, 20).astype(np.float32)
    x = rng.randn(20).astype(np.float32)
    from noetic_ir import pack_low_rank_uv
    node = make_node(
        "unit.uv", "low_rank_uv", pack_low_rank_uv(L, R),
        lowering=this_machine_stub(),
    )
    y_ir = execute_node(node, x)
    y_oracle = (L @ R) @ x
    assert np.allclose(y_ir, y_oracle, atol=1e-5)
    assert "L" in node["payload"] and "R" in node["payload"]
    assert "W" not in node["payload"]


def test_did_not_load_a_second_27b():
    doc = receipt()
    assert doc["did_not_load_27b"] is True
    assert doc["opened_model_paths"] == []


def test_cannot_express_names_the_nr_hole():
    doc = receipt()
    blob = json.dumps(doc["cannot_express"]) + json.dumps(doc["accounting"]["nr_hole"])
    assert "1 GB" in blob or "1_000_000_000" in blob or "SharedBasis" in blob
    assert f"{SEALED_BPW}" in json.dumps(doc["accounting"]["nr_hole"])


def test_already_approximates_cites_nr_nx_gravity_ir():
    doc = receipt()
    names = " ".join(row["what"] for row in doc["already_approximates_the_ir"])
    assert "noetic_representation" in names
    assert "noetic_executable_genome" in names
    assert "gravity_ir" in names
    # gravity_ir Node has no execute — it approximates accounting, not execution
    from gravity_ir import Node
    n = Node("QuantTensor", kernel="qwen_uniform_q4_group64_matvec_geo_tpr64_tg128")
    assert not hasattr(n, "execute")
    assert isinstance(n.kernel, str)


def test_account_charges_pool_once():
    pool = SharedPool()
    a = planted_basis_node(pool)
    b = planted_basis_node(pool)  # same content_id → counted once
    cost = account([a, b], pool)
    assert cost["shared_bytes"] == PLANTED_SHARED_BASIS_BYTES
    assert cost["total_bytes"] == PLANTED_SHARED_BASIS_BYTES
