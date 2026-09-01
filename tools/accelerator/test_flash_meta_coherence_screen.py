from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from flash_meta_coherence_screen import (  # noqa: E402
    DEFAULT_META_BUDGET,
    fit_shared_latent_program,
    heldout_split,
    load_teacher_capture_binding,
    load_meta_budget_binding,
    load_teacher_states,
    sha256,
)


MODEL = "Qwen/Qwen3.8-Flash-Next"
PINNED_REVISION = "34567a4712bc9766c4449e2e98e4468bfa24d915"


def _source_identity():
    return {
        "model": MODEL,
        "pinned_revision": PINNED_REVISION,
        "artifact_root": "/source/Qwen--Qwen3.8-Flash-Next@34567a4712bc",
        "safetensors_index": {
            "path": "/source/Qwen--Qwen3.8-Flash-Next@34567a4712bc/model.safetensors.index.json",
            "sha256": "a" * 64,
            "bf16_tensor_count": 1658,
        },
        "config": {
            "path": "/source/Qwen--Qwen3.8-Flash-Next@34567a4712bc/config.json",
            "sha256": "b" * 64,
        },
    }


def _zero_row_hash():
    return hashlib.sha256(np.zeros(2560, dtype="<f4").tobytes()).hexdigest()


def test_holdout_split_leaves_a_fit_row_for_small_probe():
    fit, heldout = heldout_split(4)
    assert fit.tolist() == [False, True, True, True]
    assert heldout.tolist() == [True, False, False, False]


def test_teacher_loader_deduplicates_rows_before_any_holdout_split(tmp_path):
    rows = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "state.f32"
    rows.tofile(path)

    states, metadata = load_teacher_states([path], width=4)

    assert states.shape == (2, 4)
    assert metadata["raw_rows"] == 3
    assert metadata["rows"] == 2
    assert metadata["unique_row_hashes"] == 2


def test_factor_budget_is_priced_against_bf16_source_not_float32_working_set():
    rng = np.random.default_rng(7)
    states = rng.normal(size=(20, 64)).astype(np.float32)
    weights = rng.normal(size=(2, 3, 64)).astype(np.float32)
    fit_rows, heldout_rows = heldout_split(states.shape[0])

    row = fit_shared_latent_program(
        states,
        weights,
        rank=4,
        fit_rows=fit_rows,
        heldout_rows=heldout_rows,
    )

    expected_source_bytes = weights.size * 2
    expected_factor_bytes = (4 * states.shape[1] + 2 * weights.shape[1] * 4) * 2
    assert row["selected_dense_source_bytes"] == expected_source_bytes
    assert row["selected_dense_loaded_f32_bytes"] == weights.nbytes
    assert row["diagnostic_factor_bytes"] == expected_factor_bytes
    assert row["diagnostic_factor_equivalent_bpw"] == (
        expected_factor_bytes * 8.0 / weights.size
    )
    assert row["first_surface_failure"] == (
        row["surface_failure_gates"][0] if row["surface_failure_gates"] else None
    )


def test_teacher_binding_requires_the_post_hyperconnection_mlp_input(tmp_path):
    state = tmp_path / "teacher.f32"
    np.zeros((2, 2560), dtype=np.float32).tofile(state)
    receipt = tmp_path / "teacher.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "hawking.flash.meta_teacher_trace.v1",
                "status": "CAPTURED_SOURCE_MLP_INPUT_NOT_CAPABILITY_PROVEN",
                "model": MODEL,
                "pinned_revision": PINNED_REVISION,
                "source_identity": _source_identity(),
                "teacher_trace": {
                    "layer": 4,
                    "surface": "model.language_model.layers.4.mlp_input",
                    "organ": "layer_4.routed_experts.gate_up_proj",
                    "dtype": "F32_LE",
                    "width": 2560,
                    "state_path": str(state),
                    "state_sha256": sha256(state),
                    "rows": 2,
                    "raw_rows": 2,
                    "unique_rows": 1,
                    "source_pipeline": "dense source-BF16 teacher path",
                },
                "execution": {
                    "source_bf16_authority": True,
                    "dense_prefix": True,
                    "dense_layer4": True,
                    "source_index_sha256": "a" * 64,
                },
                "rows": [
                    {
                        "row": 0,
                        "token_id": 10,
                        "layer3_state_sha256": _zero_row_hash(),
                        "layer4_mlp_input_sha256": _zero_row_hash(),
                        "layer4_output_sha256": _zero_row_hash(),
                        "layer4_output_surface": "layer_4.final_state",
                        "route_ids": list(range(10)),
                    },
                    {
                        "row": 1,
                        "token_id": 11,
                        "layer3_state_sha256": _zero_row_hash(),
                        "layer4_mlp_input_sha256": _zero_row_hash(),
                        "layer4_output_sha256": _zero_row_hash(),
                        "layer4_output_surface": "layer_4.final_state",
                        "route_ids": list(range(1, 11)),
                    },
                ],
                "route_audit": {
                    "rows": 2,
                    "unique_ordered_topk_sets": 2,
                    "route_union": list(range(11)),
                },
                "token_ids": [10, 11],
            }
        )
    )

    binding = load_teacher_capture_binding(receipt, [state])

    assert binding["surface"] == "model.language_model.layers.4.mlp_input"
    assert binding["rows_declared"] == 2
    assert binding["source_authority"] is True
    assert binding["route_union"] == list(range(11))

    with pytest.raises(ValueError, match=r"layers\.5"):
        load_teacher_capture_binding(receipt, [state], expected_layer=5)


def test_meta_budget_binding_is_detached_and_fail_closed():
    binding = load_meta_budget_binding(DEFAULT_META_BUDGET)

    assert binding["metric"] == "meta_bpw"
    assert binding["prospective_target"] < 1.0
    assert binding["physical_ebpw"] is None


def test_teacher_binding_rejects_row_count_drift(tmp_path):
    state = tmp_path / "teacher.f32"
    np.zeros((2, 2560), dtype=np.float32).tofile(state)
    receipt = tmp_path / "teacher.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "hawking.flash.meta_teacher_trace.v1",
                "status": "CAPTURED_SOURCE_MLP_INPUT_NOT_CAPABILITY_PROVEN",
                "model": MODEL,
                "pinned_revision": PINNED_REVISION,
                "source_identity": _source_identity(),
                "teacher_trace": {
                    "layer": 4,
                    "surface": "model.language_model.layers.4.mlp_input",
                    "organ": "layer_4.routed_experts.gate_up_proj",
                    "dtype": "F32_LE",
                    "width": 2560,
                    "state_path": str(state),
                    "state_sha256": sha256(state),
                    "rows": 1,
                    "raw_rows": 2,
                },
                "execution": {
                    "source_bf16_authority": True,
                    "dense_prefix": True,
                    "dense_layer4": True,
                    "source_index_sha256": "a" * 64,
                },
                "rows": [
                    {"route_ids": list(range(10))},
                    {"route_ids": list(range(1, 11))},
                ],
                "route_audit": {
                    "rows": 2,
                    "unique_ordered_topk_sets": 2,
                    "route_union": list(range(11)),
                },
                "token_ids": [10, 11],
            }
        )
    )

    with pytest.raises(ValueError, match="teacher row counts are invalid"):
        load_teacher_capture_binding(receipt, [state])
