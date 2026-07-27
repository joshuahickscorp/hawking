#!/usr/bin/env python3.12
"""Focused contracts for the bounded compact-MLA sparse fixture."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_compact_mla_fixture as compact_fixture  # noqa: E402


def test_fp64_authority_materializes_physical_weights_without_f32_matvec() -> None:
    class TinySource:
        def tensor(self, name: str) -> np.ndarray:
            assert name == "weight"
            return np.asarray([[0.25, -0.5]], dtype=np.float32)

        def matvec(self, *_args: object) -> np.ndarray:
            raise AssertionError("the FP64 authority must not use the f32 PQ matvec")

    weights = compact_fixture._materialize_fp64_authority_weights(
        TinySource(), ["weight"]  # type: ignore[arg-type]
    )
    assert weights["weight"].dtype == np.float64
    assert weights["weight"].tolist() == [[0.25, -0.5]]


def test_sparse_schedule_and_direct_u8_names_cover_every_expert_stage() -> None:
    config = compact_fixture.CONFIG
    assert config["num_hidden_layers"] == 1
    assert config["mlp_layer_types"] == ["sparse"]
    assert config["first_k_dense_replace"] == 0
    assert config["num_experts_per_tok"] == 2

    names = compact_fixture._direct_u8_tensor_names()
    assert len(names) == 29
    assert len(set(names)) == len(names)
    for expert in range(config["n_routed_experts"]):
        for projection in ("gate", "up", "down"):
            assert (
                f"model.layers.0.mlp.experts.{expert}.{projection}_proj.weight"
                in names
            )
    for projection in ("gate", "up", "down"):
        assert (
            f"model.layers.0.mlp.shared_experts.{projection}_proj.weight" in names
        )


def test_authority_record_freezes_last_token_logits_dsa_and_router_choices() -> None:
    logits = np.arange(2 * 16, dtype=np.float64).reshape(1, 2, 16)
    trace = {
        "final_main_topk": np.asarray([[[0], [1, 0]]], dtype=object),
        "layers": [
            {
                "mlp": {
                    "kind": "sparse",
                    "topk_indices": np.asarray([[[2, 3], [5, 4]]]),
                    "router_logits": np.arange(16, dtype=np.float64).reshape(1, 2, 8),
                    "topk_weights": np.asarray([[[1.0, 1.5], [1.25, 1.25]]]),
                }
            }
        ],
    }

    authority = compact_fixture._authority_record([7, 9], logits, trace)
    assert authority["tokens"] == [7, 9]
    assert authority["logits"] == [float(value) for value in logits[0, -1]]
    assert authority["final_topk"] == [1, 0]
    assert authority["expert_choices"] == [[5, 4]]
    assert authority["router_logits"] == [
        [float(value) for value in trace["layers"][0]["mlp"]["router_logits"][0, -1]]
    ]
    assert authority["expert_weights"] == [[1.25, 1.25]]
