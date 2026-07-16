from __future__ import annotations

import copy
import hashlib
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from physical_counter_fixtures import attestation, execution_authority, phase_markers
MODULE_PATH = CONDENSE / "spec_tq_runner.py"
SPEC = importlib.util.spec_from_file_location("spec_tq_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _raw() -> dict:
    prompts = 20
    pairs = []
    for phase, count in (("warmup", 3), ("trial", 5)):
        for iteration in range(count):
            order = sorted(
                range(1, 9),
                key=lambda batch: hashlib.sha256(
                    f"{'2' * 64}:{phase}:{iteration}:{batch}".encode()
                ).digest(),
            )
            for batch in order:
                pairs.append({
                    "phase": phase,
                    "batch": batch,
                    "iteration": iteration,
                    "first_role": "baseline" if (iteration + batch) % 2 == 0 else "verifier",
                    "comparison_role": "verifier",
                })
    markers = phase_markers(pairs=pairs)
    trial_by_batch = {
        batch: [
            row for row in pairs if row["phase"] == "trial" and row["batch"] == batch
        ]
        for batch in range(1, 9)
    }
    rows = []
    protocol_rows = []
    for batch in range(1, 9):
        count = ((256 + batch - 1) // batch) * batch
        baseline = [1_000_000 + repeat for repeat in range(5)]
        verifier = [1_000_000 + batch * 100_000 + repeat for repeat in range(5)]
        rows.append({
            "b": batch,
            "prompts": prompts,
            "generated_tokens_per_prompt": count,
            "values_compared": prompts * count * 5 * 2,
            "exact_token_match": True,
            "mismatches": 0,
            "skipped": 0,
            "baseline_greedy_wall_ns": baseline,
            "verifier_wall_ns": verifier,
        })
        protocol_rows.append({
            "b": batch,
            "repeats": [
                {
                    "repeat": repeat,
                    "baseline_wall_ns": baseline[repeat],
                    "verifier_wall_ns": verifier[repeat],
                    "phase_marker_sha256": trial_by_batch[batch][repeat]["phase_marker_sha256"],
                    "exact_token_match": True,
                    "mismatches": 0,
                    "skipped": 0,
                }
                for repeat in range(5)
            ],
        })
    parity_cell, curve_cell, _ = MODULE._cells("stored", "CORPUS")
    return {
        "schema": MODULE.RAW_SCHEMA,
        "source_commit": "a" * 40,
        "model": {
            "path": "/models/qwen.gguf",
            "sha256": "b" * 64,
            "size_bytes": 1000,
            "family": "qwen_dense",
        },
        "artifact": {
            "path": "/models/qwen.tq",
            "sha256": "c" * 64,
            "size_bytes": 900,
        },
        "matrix_identity": {
            "runtime_path": "stored",
            "parity_cell_id": parity_cell["id"],
            "curve_cell_id": curve_cell["id"],
            "model_sha256": "b" * 64,
            "artifact_sha256": "c" * 64,
            "tokenizer_sha256": "e" * 64,
            "prompt_set_sha256": "d" * 64,
        },
        "prompt_set": {
            "path": "/corpus/prompts.json",
            "sha256": "d" * 64,
            "schema": "hawking.spec_token_prompts.v1",
            "prompts": prompts,
            "tokenizer_sha256": "e" * 64,
        },
        "tokenizer": {
            "source": "tokenizer_json",
            "path": "/models/tokenizer.json",
            "sha256": "e" * 64,
            "size_bytes": 800,
        },
        "device": {"name": "Test Metal", "profile": "Studio-M3Ultra-96"},
        "runtime_path": "stored",
        "kernel": MODULE.RUNTIME_KERNEL["stored"],
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
        "profile_flags": {
            "vocab_prune": 32000,
            "q4k_lmhead": True,
            "q4k_predec": True,
            "prefix_cache": False,
        },
        "batches": rows,
        "measurement_protocol": {
            "warmups_per_batch": 3,
            "independent_repeats_per_batch": 5,
            "randomized_balanced_batch_order": True,
            "paired_interleaved_baseline": True,
            "baseline_reused_across_batches": False,
            "phase_marker_schema": MODULE.physical_counter_attestation.PHASE_MARKERS_SCHEMA,
            "phase_markers_sha256": markers["phase_markers_sha256"],
            "monotone_transform_applied": False,
            "batches": protocol_rows,
        },
        "phase_markers": markers,
        "physical_counters": {"measured": False},
        "default_change_requested": False,
    }


def _snapshot(swap: float = 100.0) -> dict:
    return {"pressure_level": 1, "pressure_name": "normal", "swap_used_mb": swap}


def _bundle(tmp_path: pathlib.Path) -> dict:
    raw = _raw()
    return MODULE.build_bundle(
        raw,
        resource_before=_snapshot(),
        resource_after=_snapshot(),
        thermal_before="nominal",
        thermal_after="nominal",
        execution_authority=execution_authority(tmp_path, raw),
    )


def _counters(bundle: dict, tmp_path: pathlib.Path) -> tuple[dict, dict]:
    raw = bundle["raw_probe"]
    protocol = {
        row["b"]: row for row in raw["measurement_protocol"]["batches"]
    }
    counters = {
        "schema": MODULE.COUNTER_SCHEMA,
        "raw_bundle_sha256": bundle["raw_bundle_sha256"],
        "artifact_sha256": bundle["raw_probe"]["artifact"]["sha256"],
        "runtime_path": "stored",
        "phase_markers_sha256": raw["phase_markers"]["phase_markers_sha256"],
        "batches": [
            {
                "b": batch,
                "repeats": [
                    {
                        "repeat": repeat,
                        "phase_marker_sha256": row["phase_marker_sha256"],
                        "energy_j": 0.2 + batch / 100,
                        "gpu_time_ns": 1_000_000 * batch + repeat,
                        "physical_bytes": 1_000_000 * batch + repeat,
                    }
                    for repeat, row in enumerate(protocol[batch]["repeats"])
                ],
            }
            for batch in range(1, 9)
        ],
    }
    counter_attestation = attestation(
        tmp_path, bundle=bundle, counter_payload=counters,
        domains=("energy", "gpu_time", "physical_bytes"), sample_count=160,
    )
    return counters, counter_attestation


def test_raw_bundle_and_both_receipts_are_strict(tmp_path: pathlib.Path) -> None:
    raw = _raw()
    assert MODULE.validate_raw(raw) == []
    bundle = _bundle(tmp_path)
    assert MODULE.validate_bundle(bundle) == []
    counters, counter_attestation = _counters(bundle, tmp_path)
    assert MODULE.validate_counters(counters, counter_attestation, bundle) == []
    parity, curve = MODULE.finalize_receipts(bundle, counters, counter_attestation)
    assert MODULE.spec_receipt_contract.validate_receipt(parity) == []
    assert MODULE.spec_receipt_contract.validate_receipt(curve) == []
    assert parity["bindings"]["target_sha256"] == "c" * 64
    assert curve["bindings"]["parent_receipt_sha256"] == [parity["receipt_sha256"]]
    totals = [row["total_forward_equiv"] for row in curve["experiment_payload"]["batches"]]
    assert totals == sorted(totals)


def test_parity_skip_swap_growth_and_counter_rebinding_fail(tmp_path: pathlib.Path) -> None:
    raw = _raw()
    raw["batches"][3]["skipped"] = 1
    assert MODULE.validate_raw(raw)

    growth = MODULE.build_bundle(
        _raw(),
        resource_before=_snapshot(100.0),
        resource_after=_snapshot(101.0),
        thermal_before="nominal",
        thermal_after="nominal",
        execution_authority=execution_authority(tmp_path, _raw()),
    )
    assert "runner observed swap growth" in MODULE.validate_bundle(growth)

    bundle = _bundle(tmp_path)
    counters, counter_attestation = _counters(bundle, tmp_path)
    counters["raw_bundle_sha256"] = "0" * 64
    assert "counters are not bound to the raw bundle" in MODULE.validate_counters(
        counters, counter_attestation, bundle,
    )


def test_bundle_tamper_cannot_retain_stamp(tmp_path: pathlib.Path) -> None:
    bundle = _bundle(tmp_path)
    tampered = copy.deepcopy(bundle)
    tampered["raw_probe"]["coverage"]["gpu_resident"] -= 1
    errors = MODULE.validate_bundle(tampered)
    assert "raw_bundle_sha256 mismatch" in errors
    assert "raw TQ all-linear GPU coverage is incomplete" in errors


def test_phase_protocol_tampering_and_sparse_malformed_raw_fail_closed() -> None:
    raw = _raw()
    raw["measurement_protocol"]["batches"][0]["repeats"][0][
        "phase_marker_sha256"
    ] = raw["measurement_protocol"]["batches"][1]["repeats"][0][
        "phase_marker_sha256"
    ]
    raw["phase_markers"]["pairs"][0]["first_role"] = "verifier"
    errors = MODULE.validate_raw(raw)
    assert any("reused" in error or "baseline/verifier" in error for error in errors)
    assert any("first_role" in error or "phase_markers_sha256" in error for error in errors)
    for malformed in ({"phase_markers": {}}, {"benchmark": {}}, {"batches": [None] * 8}):
        assert MODULE.validate_raw(malformed)
