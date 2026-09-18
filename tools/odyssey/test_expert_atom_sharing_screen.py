from __future__ import annotations

import numpy as np
import pytest

from tools.odyssey.expert_atom_sharing_screen import (
    _aggregate_replication_bank,
    _canonical_sha256,
    _accounting,
    _canonicalize_triples,
    _selected_neurons,
    _triple_outputs,
    screen_matrices,
    verify_replication_receipt,
)
from tools.odyssey import expert_atom_sharing_screen as atom_screen


def test_selected_neurons_are_deterministic_and_unique() -> None:
    np.testing.assert_array_equal(_selected_neurons(8, 4), np.array([0, 2, 4, 6]))
    assert np.unique(_selected_neurons(17, 8)).size == 8


def test_reciprocal_up_down_scaling_preserves_gated_output() -> None:
    rng = np.random.default_rng(20260912)
    gate = rng.normal(size=(3, 6)).astype(np.float32)
    up = rng.normal(size=(3, 6)).astype(np.float32)
    down = rng.normal(size=(3, 6)).astype(np.float32)
    activation = rng.normal(size=6).astype(np.float32)
    canonical = _canonicalize_triples(gate, up, down)[:3]
    np.testing.assert_allclose(
        _triple_outputs((gate, up, down), activation),
        _triple_outputs(canonical, activation),
        rtol=2e-5,
        atol=2e-5,
    )


def test_accounting_bills_dictionary_assignments_and_support() -> None:
    result = _accounting(expert_count=2, atom_count=128, width=2560)
    assert result["source_exact_bf16_scope_bytes"] == 2 * 128 * 3 * 2560 * 2
    assert result["shared_bf16_atom_dictionary_bytes"] == 128 * 3 * 2560 * 2
    assert result["packed_atom_assignment_bits"] == 128 * 7
    assert result["reciprocal_scaling_descriptor_bytes"] == 0
    assert result["complete_bytes"] > result["shared_bf16_atom_dictionary_bytes"]
    assert result["byte_advantage_percent"] > 20.0


def test_screen_is_deterministic_and_keeps_scope_explicit() -> None:
    rng = np.random.default_rng(20260912)
    gate_ups = [rng.normal(size=(8, 6)).astype(np.float32) for _ in range(2)]
    downs = [rng.normal(size=(6, 4)).astype(np.float32) for _ in range(2)]
    kwargs = {
        "expert_ids": [3, 7],
        "route_weights": [0.75, 0.25],
        "activation": rng.normal(size=6).astype(np.float32),
        "selected_count": 4,
        "probe_count": 4,
        "output_probe_dim": 4,
        "seed": 11,
    }
    first = screen_matrices(gate_ups, downs, **kwargs)
    second = screen_matrices(gate_ups, downs, **kwargs)
    assert first["variant"]["family"] == "lawful_nonlinear_gated_moe_atom_sharing"
    assert first["accounting"]["scope_boundary"].startswith("two experts")
    assert first["matching"]["assignments"] == second["matching"]["assignments"]
    assert first["outcome"]["status"] in {"SOURCE_OUTPUT_DISCRIMINATOR_SURVIVES", "FIXED_FAMILY_NEGATIVE"}
    assert first["source_activation_output"]["full_route"]["matched"]["finite"] is True
    assert first["function_probe_output"]["route_weighted"]["shuffled_null"]["finite"] is True


def test_screen_requires_two_experts() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        screen_matrices(
            [np.ones((8, 6), dtype=np.float32)],
            [np.ones((6, 4), dtype=np.float32)],
            expert_ids=[1],
            route_weights=[1.0],
            activation=np.ones(6, dtype=np.float32),
            selected_count=4,
        )


def _pair_result(*, survives: bool, matched: float = 0.5, shuffled: float = 0.7) -> dict:
    return {
        "outcome": {
            "status": (
                "SOURCE_OUTPUT_DISCRIMINATOR_SURVIVES" if survives else "FIXED_FAMILY_NEGATIVE"
            ),
            "source_output_matched_null_advantage": survives,
        },
        "accounting": {
            "complete_scoped_ebpw": 8.25,
            "complete_bytes": 100,
            "source_exact_bf16_scope_bytes": 200,
        },
        "source_activation_output": {
            "full_route": {
                "matched": {"relative_l2": matched},
                "shuffled_null": {"relative_l2": shuffled},
            }
        },
    }


def test_replication_bank_requires_all_three_new_disjoint_pairs() -> None:
    experts = list(range(8))
    positive = _aggregate_replication_bank(
        [_pair_result(survives=True) for _ in range(4)],
        expert_ids=experts,
        seed=1337,
    )
    assert positive["replicated"] is True
    assert positive["independent_survivor_count"] == 3
    assert [row["expert_ids"] for row in positive["pairs"]] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert [row["seed"] for row in positive["pairs"]] == [1337, 1338, 1339, 1340]
    assert positive["aggregate_accounting"]["byte_advantage_percent"] == 50.0

    mixed = _aggregate_replication_bank(
        [
            _pair_result(survives=True),
            _pair_result(survives=True),
            _pair_result(survives=False, matched=0.8, shuffled=0.7),
            _pair_result(survives=True),
        ],
        expert_ids=experts,
        seed=1337,
    )
    assert mixed["replicated"] is False
    assert mixed["status"] == "NOT_REPLICATED_ACROSS_PREDECLARED_TOP8_ROUTE_BANK"


def test_replication_bank_refuses_duplicate_or_partial_expert_sets() -> None:
    results = [_pair_result(survives=True) for _ in range(4)]
    with pytest.raises(ValueError, match="eight experts"):
        _aggregate_replication_bank(results, expert_ids=list(range(7)), seed=1337)
    with pytest.raises(ValueError, match="distinct"):
        _aggregate_replication_bank(results, expert_ids=[0, 1, 2, 3, 4, 5, 6, 6], seed=1337)


def test_replication_receipt_verifier_recomputes_seal_and_pair_aggregate() -> None:
    experts = list(range(8))
    replication = _aggregate_replication_bank(
        [
            _pair_result(survives=True),
            _pair_result(survives=True),
            _pair_result(survives=False, matched=0.8, shuffled=0.7),
            _pair_result(survives=True),
        ],
        expert_ids=experts,
        seed=1337,
    )
    producer_path = atom_screen.Path(atom_screen.__file__).resolve()
    document = {
        "schema": atom_screen.REPLICATION_SCHEMA,
        "status": "SOURCE_BOUND_NONLINEAR_ATOM_SHARING_REPLICATION_COMPLETE",
        "producer": {
            "path": str(producer_path),
            "sha256": atom_screen._sha256_bytes(producer_path.read_bytes()),
        },
        "source": {
            "specimen": str(atom_screen.DEFAULT_SPEC),
            "verified_source_shard_ledger": {
                "verification": "sha256_exact_file_bytes (attested by sealed ledger)"
            },
            "sealed_route_top8_expert_ids": experts,
        },
        "replication": replication,
        "model_loaded": False,
        "gpu_session_started": False,
    }
    document["seal_sha256"] = _canonical_sha256(document)
    assert verify_replication_receipt(document)["replication"]["replicated"] is False

    document["replication"]["independent_survivor_count"] = 3
    with pytest.raises(ValueError, match="seal mismatch"):
        verify_replication_receipt(document)
