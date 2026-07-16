from __future__ import annotations

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
MODULE_PATH = CONDENSE / "spec_receipt_contract.py"
SPEC = importlib.util.spec_from_file_location("spec_receipt_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _workloads(*, exact: bool = True, speedup: float = 1.10) -> list[dict]:
    return [
        {
            "workload": workload,
            "prompts": 10,
            "scored_tokens": 1024,
            "speedup_lcb": speedup,
            "quality_passed": True,
            "exact_target_commit": exact,
        }
        for workload in sorted(MODULE.WORKLOADS)
    ]


def test_requirements_cover_every_named_spec_schema() -> None:
    req = MODULE.requirements()
    assert set(req["payload_schemas"]) == MODULE.SCHEMAS


def test_parity_and_curve_require_all_eight_batches() -> None:
    parity = {
        "runtime_path": "stored",
        "model_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "kernel": "strand_bitslice_gemm_small_stored",
        "coverage": {
            "expected_all_linear": 196,
            "mapped": 196,
            "gpu_resident": 196,
            "residual_gpu_resident": 0,
        },
        "target_identity": {
            "reference": "tq_single_token_greedy",
            "verifier": "tq_batch_major_b1_b8",
            "greedy_tie_break": "canonical_qwen_argmax",
            "all_owned_projections_tq_native": True,
        },
        "default_change_requested": False,
        "batches": [
            {
                "b": b,
                "prompts": 20,
                "generated_tokens_per_prompt": 256,
                "exact_token_match": True,
                "mismatches": 0,
                "skipped": 0,
            }
            for b in range(1, 9)
        ],
    }
    assert MODULE.validate_payload("hawking.spec_tq_batched_parity.v1", parity) == []
    parity["batches"].pop()
    assert MODULE.validate_payload("hawking.spec_tq_batched_parity.v1", parity)

    curve = {
        "runtime_path": "hashed",
        "model_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "physical_counters_measured": True,
        "counter_attestation": MODULE.physical_counter_attestation.stamp({
            "schema": MODULE.physical_counter_attestation.SCHEMA,
        }),
        "curve_transform": "none-observed-raw-ratios",
        "curve_method": {
            "warmups_per_batch": 3,
            "independent_repeats_per_batch": 5,
            "paired_interleaved_baseline": True,
            "monotone_transform_applied": False,
            "ucb_method": "paired_bootstrap_95",
            "confidence_level": 0.95,
            "phase_markers_sha256": "c" * 64,
        },
        "counter_sources": {
            "energy_source": "powermetrics",
            "gpu_time_source": "metal_trace",
            "bytes_source": "gpu_counters",
        },
        "default_change_requested": False,
        "batches": [
            {
                "b": b,
                "trials": 5,
                "median_ns": b * 100,
                "p95_ns": b * 110,
                "ucb_ns": b * 120,
                "raw_total_forward_equiv": float(b),
                "total_forward_equiv": float(b),
                "energy_j_total": b / 10,
                "gpu_time_ns": b * 1000,
                "bytes": {
                    field: (b * 100 if field == "weights" else 0)
                    for field in MODULE.appendix_contract.BYTE_FIELDS
                },
            }
            for b in range(1, 9)
        ],
    }
    assert MODULE.validate_payload("hawking.spec_verifier_curve.v1", curve) == []
    curve["batches"][4]["total_forward_equiv"] = 0.5
    assert MODULE.validate_payload("hawking.spec_verifier_curve.v1", curve)


def test_proposer_tree_and_composition_fail_closed() -> None:
    proposer = {
        "proposer": "retrieval",
        "workload": "code",
        "prompts": 10,
        "scored_tokens": 1024,
        "held_out_exact_target": True,
        "lookup_and_miss_cost_charged": True,
        "draft_lengths": [
            {"k": k, "proposed_tokens": 100, "accepted_tokens": 50, "lookup_ns": 1, "miss_ns": 1}
            for k in range(2, 8)
        ],
    }
    assert MODULE.validate_payload("hawking.spec_proposer_oracle.v1", proposer) == []
    proposer["lookup_and_miss_cost_charged"] = False
    assert MODULE.validate_payload("hawking.spec_proposer_oracle.v1", proposer)

    tree = {
        "tree_width": 4,
        "parity_mismatches": 0,
        "rollback_tests_passed": True,
        "cpu_fallback_used_as_speed_evidence": False,
        "longest_argmax_confirmed_prefix": True,
    }
    assert MODULE.validate_payload("hawking.spec_tree_verify.v1", tree) == []
    tree["parity_mismatches"] = 1
    assert MODULE.validate_payload("hawking.spec_tree_verify.v1", tree)

    composition = {"target_runtime_path": "computed", "workloads": _workloads()}
    assert MODULE.validate_payload("hawking.spec_composition_gate.v1", composition) == []
    composition["workloads"][0]["speedup_lcb"] = 1.09
    assert MODULE.validate_payload("hawking.spec_composition_gate.v1", composition)


def test_learned_receipt_forbids_placeholder_tokens() -> None:
    payload = {
        "architecture": "p_eagle_style_parallel_mtp",
        "placeholder_token_count": 0,
        "trained_on_served_target_distribution": True,
        "workloads": [
            {
                "workload": workload,
                "draft_latency_ns": 1,
                "proposed_tokens": 100,
                "accepted_tokens": 50,
            }
            for workload in sorted(MODULE.WORKLOADS)
        ],
    }
    assert MODULE.validate_payload("hawking.spec_parallel_head.v1", payload) == []
    payload["placeholder_token_count"] = 7
    assert MODULE.validate_payload("hawking.spec_parallel_head.v1", payload)
