"""Focused contract test for the sealed DSV4F child-baseline bundle."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import deepseek_v4_gravity as gravity
from lab.receipts import seal, verify


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _source_config() -> dict:
    return {
        "model_type": "deepseek_v4",
        "torch_dtype": "bfloat16",
        "expert_dtype": "fp4",
        "hidden_size": 16,
        "num_hidden_layers": 43,
        "num_attention_heads": 2,
        "head_dim": 8,
        "q_lora_rank": 4,
        "o_lora_rank": 4,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "vocab_size": 64,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 4,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "use_cache": True,
    }


def _artifact(root: Path, *, full: bool, config: dict) -> tuple[Path, dict]:
    artifact = root / ("full.gravity" if full else "diagnostic.gravity")
    config_path = artifact / "metadata" / "config.json"
    _write_json(config_path, config)
    source = {
        "repository": gravity.REPOSITORY,
        "revision": gravity.REVISION,
        "metadata_assets": {
            "config.json": {"sha256": gravity._sha256(config_path.read_bytes())}
        },
    }
    if full:
        document = seal(
            {
                "schema": gravity.FULL_ARTIFACT_SCHEMA,
                "status": "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY",
                "source": source,
                "storage": {"source_parent_retained": False},
                "runtime_adapter": {
                    "id": None,
                    "registration": None,
                    "device": None,
                    "metal_dispatches": 0,
                    "capability_status": "full_artifact_streamed_runtime_pending",
                },
            }
        )
    else:
        document = seal(
            {
                "schema": gravity.ARTIFACT_SCHEMA,
                "status": "DIAGNOSTIC_SEALED_LOADABLE_BY_V4_NUMPY_ADAPTER",
                "source": source,
                "storage": {"source_parent_retained": False},
                "runtime_adapter": {
                    "id": "test.diagnostic",
                    "registration": "test",
                    "device": "cpu",
                    "metal_dispatches": 0,
                    "capability_status": "diagnostic_cpu_only_not_tg_eligible",
                },
                "diagnostic_scope": {"selected_layer": 4},
            }
        )
    _write_json(artifact / "manifest.json", document)
    return artifact, document


def _profile_stages() -> dict:
    return {
        name: {
            "label": name,
            "execution_statuses": {"executed": 1},
            "cpu_duration_ms": {"p50": 1.0, "p95": 1.5, "p99": 1.9},
            "cpu_wall_elapsed_ms": {"p50": 1.1, "p95": 1.6, "p99": 2.0},
            "bytes_read_estimate_total": 10,
            "bytes_written_estimate_total": 5,
            "fp_operations_estimate_total": 20,
            "integer_bit_operations_estimate_total": 2,
            "dispatches_total": 0,
            "command_buffers_total": 0,
            "waits_total": 0,
        }
        for name in dict.fromkeys(gravity._COMPLETE_TOKEN_PROFILE_STAGES)
    }


def _trace_record(position: int) -> dict:
    components = {
        name: {"executed": True}
        for name in (
            "embedding",
            "attention",
            "hc_attention",
            "hc_ffn",
            "routed_experts",
            "shared_expert",
            "head",
        )
    }
    components["router"] = {
        "executed": True,
        "kind": "score_router",
        "selected_route_ids": [1, 3],
    }
    components.update({"metal_dispatches": 0, "numeric_parity_v2_1": "not_proven"})
    return {"position": position, "component_execution": components}


def test_freeze_child_baseline_writes_only_the_fixed_sealed_bundle(tmp_path: Path) -> None:
    config = _source_config()
    full_artifact, full_manifest = _artifact(tmp_path, full=True, config=config)
    diagnostic_artifact, diagnostic_manifest = _artifact(tmp_path, full=False, config=config)
    full_seal = full_manifest["seal_sha256"]
    diagnostic_seal = diagnostic_manifest["seal_sha256"]

    profile = _write_json(
        tmp_path / gravity._CANONICAL_COMPLETE_TOKEN_PROFILE_V3,
        seal(
            {
                "schema": "hawking.gravity.deepseek_v4.complete_token_profile_receipt.v1",
                "status": "SEALED_REAL_LAYER4_CPU_DIAGNOSTIC_PROFILE_NOT_BASE_TRUE_TPS",
                "artifact": {"seal_sha256": diagnostic_seal},
                "claim_boundary": {"base_true_tps": False, "metal_dispatch": False},
                "profile_run": {"route_frequency": {"1": 2, "3": 2}},
                "aggregate": {
                    "real_diagnostic_forward_count": 2,
                    "stages": _profile_stages(),
                    "timing_accounting": {"other_share_percent": 0.0, "status": "PASS"},
                    "gpu_dispatch_accounting": {
                        "command_buffers_total": 0,
                        "dispatches_total": 0,
                        "waits_total": 0,
                    },
                    "endpoint_hcli_streaming": {"status": "NOT_MEASURED"},
                    "complete_token_wall_elapsed_ms": {"p50": 2.0, "p95": 2.5, "p99": 3.0},
                    "complete_token_cpu_duration_ms": {"p50": 1.5, "p95": 2.0, "p99": 2.5},
                },
            }
        ),
    )
    hcli = _write_json(
        tmp_path / gravity._CANONICAL_HCLI_LIVE_SUITE_V2,
        seal(
            {
                "schema": "hawking.gravity.deepseek_v4.hcli_live_suite.v1",
                "status": "HCLI_LIVE_SUITE_EVIDENCE_SEALED_DIAGNOSTIC_ONLY",
                "artifact": {"seal_sha256": diagnostic_seal},
                "endpoint": {"runtime_context": {"metal_dispatches": 0}},
                "claim_boundary": {"full_43_layer_runtime": False, "base_true_tps": False},
                "evidence": [],
            }
        ),
    )
    blocker = _write_json(
        tmp_path / "blocker-v2.json",
        seal(
            {
                "schema": "hawking.gravity.deepseek_v4.full_runtime_blocker.v1",
                "status": "FULL_STREAMED_RUNTIME_NO_REGISTERED_43_LAYER_ADAPTER",
                "artifact": {
                    "manifest_seal_sha256": full_seal,
                    "repository": gravity.REPOSITORY,
                    "revision": gravity.REVISION,
                    "source_parent_retained": False,
                },
                "storage_accounting": {
                    "raw_artifact_eviction_authorized": False,
                    "raw_parent_materialized": False,
                    "protected_floor_bytes": gravity.MIN_FREE_FLOOR_BYTES,
                    "eviction_rule": "retain raw stream",
                },
                "first_missing_milestone": {"stage": "native_v4_adapter"},
                "missing_execution_grammar": {"required_runtime_semantics": ["native V4"]},
            }
        ),
    )
    fp8_probe = _write_json(
        tmp_path / "fp8-metal-component-probe.json",
        seal(
            {
                "schema": "hawking.gravity.deepseek_v4.fp8_e4m3fn_e8m0_metal_component_probe.v1",
                "status": "PASS_REAL_METAL_COMPONENT_PARITY_NOT_FULL_RUNTIME",
                "artifact": {"manifest_seal_sha256": full_seal},
                "source": {
                    "repository": gravity.REPOSITORY,
                    "revision": gravity.REVISION,
                    "weight": {
                        "name": "layers.0.attn.wq_a.weight",
                        "dtype": "F8_E4M3",
                        "shape": [2, 16],
                    },
                    "scale": {
                        "name": "layers.0.attn.wq_a.scale",
                        "dtype": "F8_E8M0",
                        "shape": [1, 1],
                    },
                },
                "scope": {
                    "component": "one FP8 matvec",
                    "not_a_full_model_load": True,
                    "not_a_generation_or_TPS_claim": True,
                    "not_a_registered_43_layer_runtime_adapter": True,
                },
                "parity": {"status": "PASS", "max_abs_error": 0.0, "max_relative_error": 0.0},
                "metal": {
                    "fallback": False,
                    "gpu_dispatches": 1,
                    "device": "test Metal",
                    "kernel": "test.fp8",
                    "command_buffers": 1,
                    "compute_encoders": 1,
                    "timing": {"gpu_duration_us": 1},
                },
            }
        ),
    )
    trace = _write_json(
        tmp_path / "component-trace.json",
        seal(
            {
                "schema": "hawking.gravity.deepseek_v4.diagnostic_generation.v1",
                "status": "FIRST_TOKEN_GENERATED_DIAGNOSTIC",
                "artifact_seal_sha256": diagnostic_seal,
                "result": {
                    "stats": {"fallback": "cpu diagnostic"},
                    "trace": [_trace_record(0), _trace_record(1)],
                },
            }
        ),
    )
    tps_gate = _write_json(
        tmp_path / "base-tps-gate.json",
        seal(
            {
                "schema": "hawking.gravity.deepseek_v4.base_tps_gate.v1",
                "status": "BASE_TRUE_TPS_WITHHELD",
                "artifact": {"seal_sha256": diagnostic_seal},
                "observed_cpu_diagnostic_measurement": {
                    "classification": "DIAGNOSTIC_CPU_ONLY_NOT_BASE_TRUE_TPS",
                    "complete_forward_tps": 1.0,
                    "decode_ms": 1000.0,
                },
                "physical_blockers": ["CPU diagnostic"],
            }
        ),
    )
    reverify = _write_json(
        tmp_path / "full-reverify.json",
        seal(
            {
                "schema": "hawking.gravity.deepseek_v4.full_reverify.v1",
                "status": "FULL_MODEL_STREAM_FULLY_REVERIFIED_RUNTIME_PENDING",
                "artifact_seal_sha256": full_seal,
                "full_chunk_verification": {"sha256_verified": True},
            }
        ),
    )
    out_dir = tmp_path / "baseline"
    result = gravity.freeze_child_baseline(
        full_artifact_dir=full_artifact,
        diagnostic_artifact_dir=diagnostic_artifact,
        complete_token_profile=profile,
        hcli_live_suite=hcli,
        full_runtime_blocker=blocker,
        fp8_metal_component_probe=fp8_probe,
        component_trace=trace,
        base_tps_gate=tps_gate,
        full_reverify=reverify,
        out_dir=out_dir,
    )

    verify(result, label="baseline freeze result")
    assert {path.name for path in out_dir.iterdir()} == set(gravity._CHILD_BASELINE_FILENAMES)
    for filename in gravity._CHILD_BASELINE_FILENAMES:
        document = json.loads((out_dir / filename).read_text(encoding="utf-8"))
        verify(document, label=filename)
        assert document["claim_boundary"]["full_43_layer_runtime"] is False
        assert document["claim_boundary"]["base_true_tps"] is False
        assert document["claim_boundary"]["full_43_layer_metal_dispatch"] is False
        assert document["claim_boundary"]["component_only_fp8_metal_probe"] is True
    scoreboard = json.loads((out_dir / "DSV4F_100TPS_SCOREBOARD.json").read_text(encoding="utf-8"))
    assert scoreboard["metrics"]["BASE_TRUE_TPS"]["value"] is None
    assert scoreboard["first_required_change_before_a_real_base_run"] == "native_v4_adapter"
    assert gravity.freeze_child_baseline(
        full_artifact_dir=full_artifact,
        diagnostic_artifact_dir=diagnostic_artifact,
        complete_token_profile=profile,
        hcli_live_suite=hcli,
        full_runtime_blocker=blocker,
        fp8_metal_component_probe=fp8_probe,
        component_trace=trace,
        base_tps_gate=tps_gate,
        full_reverify=reverify,
        out_dir=out_dir,
    )["seal_sha256"] == result["seal_sha256"]
    (out_dir / "unexpected.txt").write_text("refuse", encoding="utf-8")
    with pytest.raises(gravity.DeepSeekV4GravityError, match="fixed child baseline bundle"):
        gravity.freeze_child_baseline(
            full_artifact_dir=full_artifact,
            diagnostic_artifact_dir=diagnostic_artifact,
            complete_token_profile=profile,
            hcli_live_suite=hcli,
            full_runtime_blocker=blocker,
            fp8_metal_component_probe=fp8_probe,
            component_trace=trace,
            base_tps_gate=tps_gate,
            full_reverify=reverify,
            out_dir=out_dir,
        )
