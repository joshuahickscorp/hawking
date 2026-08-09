"""Focused non-GPU tests for the HQ30GR2 all-layer comparison receipt."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.operators import ascension_qwen30_quality_repack_all_layer_comparison as comparison
from lab.receipts import seal, verify


def _write(path: Path, document: dict[str, object]) -> dict[str, object]:
    sealed = seal(document)
    path.write_text(json.dumps(sealed, sort_keys=True) + "\n", encoding="utf-8")
    return sealed


def _sha(character: str) -> str:
    return character * 64


def _top(ids: list[int], *, offset: float = 0.0) -> list[dict[str, object]]:
    return [
        {"token_id": token, "logit": 10.0 - rank + offset, "logit_bits": 0x41000000 - rank}
        for rank, token in enumerate(ids)
    ]


def _logits(ids: list[int], digest: str) -> dict[str, object]:
    return {"vocab_rows": 32, "full_f32le_sha256": _sha(digest), "top_k": _top(ids)}


def _step(*, position: int, token: int, sampled: int, digest: str, e0: bool, ids: list[int]) -> dict[str, object]:
    return {
        "position": position,
        "input_token_id": token,
        "sampled_token_id": sampled,
        "route_ids_u32le_sha256": _sha(digest),
        "l0_expert0_selected": e0,
        "l0_expert_ids": ids,
        "all_layers_route_captured": 48,
    }


def _prefix(*, candidate: bool) -> dict[str, object]:
    prefix_ids = [3, 4, 5, 6, 7, 8, 9, 10] if candidate else [1, 3, 4, 5, 6, 7, 8, 9]
    return {
        "exact_prefix_token_forwards": 369,
        "all_layer_route_captures": 369 * 48,
        "layers_per_forward": 48,
        "route_trace_sha256": _sha("b" if candidate else "a"),
        "l0_expert0_selected_positions": [337],
        "target_position_step": _step(
            position=337,
            token=12,
            sampled=3,
            digest="d" if candidate else "c",
            e0=True,
            ids=[117, 97, 99, 126, 0, 37, 74, 24],
        ),
        "final_prefix_step": _step(
            position=368,
            token=13,
            sampled=3 if candidate else 1,
            digest="f" if candidate else "e",
            e0=False,
            ids=[16, 100, 1, 109, 104, 51, 76, 119],
        ),
        "final_logits": _logits(prefix_ids, "b" if candidate else "a"),
    }


def _continuation(*, candidate: bool) -> dict[str, object]:
    ids = [3, 4, 5, 6, 7, 8, 9, 10] if candidate else [2, 3, 4, 5, 6, 7, 8, 9]
    return {
        "additional_forwards": 1,
        "step": _step(
            position=369,
            token=1,
            sampled=3 if candidate else 2,
            digest="d" if candidate else "c",
            e0=False,
            ids=[7, 30, 98, 87, 120, 45, 114, 53],
        ),
        "final_logits": _logits(ids, "f" if candidate else "e"),
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    inner = tmp_path / "inner.json"
    outer = tmp_path / "outer.json"
    chain = tmp_path / "chain.json"
    chain_receipt = tmp_path / "chain-receipt.json"
    chain_doc = _write(
        chain_receipt,
        {
            "schema": comparison.CHAIN_SCHEMA,
            "status": comparison.CHAIN_STATUS,
            "assessment": {
                "all_three_selected_positions_complete_mlp_down_output_improved_vs_source": True,
                "actual_route_reach": {
                    "selected_positions": {"literal_hawking": 337, "json_status": 321, "python_add": 339}
                },
            },
        },
    )
    _write(
        chain,
        {
            "schema": comparison.CHAIN_CURRENT_SCHEMA,
            "status": comparison.CHAIN_CURRENT_STATUS,
            "chain_receipt": {"path": str(chain_receipt.resolve()), "seal_sha256": chain_doc["seal_sha256"]},
        },
    )
    inner_doc = _write(
        inner,
        {
            "schema": comparison.INNER_SCHEMA,
            "status": comparison.INNER_STATUS,
            "artifact_binding": {
                "candidate_manifest_seal_sha256": _sha("1"),
                "candidate_admission_receipt_seal_sha256": _sha("2"),
                "control_manifest_seal_sha256": _sha("3"),
                "control_runtime_receipt_seal_sha256": _sha("4"),
            },
            "exact_trace_execution": {
                "probe_id": "literal_hawking",
                "source_template_token_count": 369,
                "source_template_token_ids_u32le_sha256": _sha("5"),
                "forced_continuation": {"forced_token_id": 1},
            },
            "metal_execution_policy": {
                "timing_or_benchmarking_allowed": False,
                "hcli_or_server_allowed": False,
                "tps_or_tg_claim_allowed": False,
                "coherence_claim_allowed": False,
                "capability_claim_allowed": False,
                "tournament_claim_allowed": False,
            },
            "structural_witnesses": {
                "control_scalar_path": _prefix(candidate=False),
                "candidate_typed_hq30gr2_path": _prefix(candidate=True),
                "control_forced_continuation": _continuation(candidate=False),
                "candidate_forced_continuation": _continuation(candidate=True),
                "typed_l0_e0_sparse_interception": {
                    "device_sparse_gate_up_encodes": 1,
                    "matching_l0_e0_route_selections": 1,
                    "direct_fallback_for_sparse_residual_forbidden": True,
                    "scalar_control_topology_for_all_unchanged_organs": True,
                    "selected_residual_organs": ["gate", "up"],
                },
            },
        },
    )
    _write(
        outer,
        {
            "schema": comparison.OUTER_SCHEMA,
            "status": comparison.OUTER_STATUS,
            "inner_probe_capture": {
                "binding_valid": True,
                "metal_performed": True,
                "receipt": {"path": str(inner.resolve()), "sha256": __import__("hashlib").sha256(inner.read_bytes()).hexdigest()},
            },
        },
    )
    assert inner_doc["seal_sha256"]
    return inner, outer, chain


def test_build_comparison_is_explicitly_non_promotable(tmp_path: Path) -> None:
    inner, outer, chain = _inputs(tmp_path)
    result = comparison.build_comparison(diagnostic_path=inner, outer_path=outer, chain_current_path=chain)
    verify(result)
    assert result["status"] == comparison.STATUS
    assert result["classification"]["candidate_runtime_promotion"] == "NOT_ELIGIBLE"
    assert result["classification"]["allowed_next_scope"] == "BROADER_DIAGNOSTIC_ONLY"
    assert result["divergence"]["bounded_top_k"]["exact_prefix"]["shared_token_count"] == 7
    assert result["divergence"]["route_trace"]["target_l0_e0_step"]["l0_expert_ids_equal"] is True
    assert result["observed_candidate_local_effect"]["typed_sparse_device_encodes"] == 1


def test_rejects_mismatched_forced_token(tmp_path: Path) -> None:
    inner, outer, chain = _inputs(tmp_path)
    document = json.loads(inner.read_text(encoding="utf-8"))
    document.pop("seal_sha256")
    document["structural_witnesses"]["candidate_forced_continuation"]["step"]["input_token_id"] = 99
    _write(inner, document)
    outer_document = json.loads(outer.read_text(encoding="utf-8"))
    outer_document.pop("seal_sha256")
    outer_document["inner_probe_capture"]["receipt"]["sha256"] = __import__("hashlib").sha256(inner.read_bytes()).hexdigest()
    _write(outer, outer_document)
    with pytest.raises(comparison.AllLayerComparisonError, match="forced continuation input token"):
        comparison.build_comparison(diagnostic_path=inner, outer_path=outer, chain_current_path=chain)


def test_rejects_top_k_duplicate_token() -> None:
    with pytest.raises(comparison.AllLayerComparisonError, match="repeats token ID"):
        comparison._top_k(
            [
                {"token_id": 1 if index < 2 else index, "logit": 10.0 - index, "logit_bits": index}
                for index in range(8)
            ],
            label="fixture",
        )
