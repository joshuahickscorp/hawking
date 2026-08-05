"""Offline contract tests for the consolidated three-generation pipeline."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import frankenstein_pipeline as pipeline  # noqa: E402
from lab.receipts import seal, verify  # noqa: E402


def _write(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path


def _inputs(root: Path) -> pipeline.PipelineInputs:
    full_manifest = seal(
        {
            "schema": pipeline.FULL_MANIFEST_SCHEMA,
            "status": "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY",
            "source": {
                "repository": pipeline.DEEPSEEK_REPOSITORY,
                "revision": pipeline.DEEPSEEK_REVISION,
            },
        }
    )
    manifest_path = _write(root / "full-manifest.json", full_manifest)
    return pipeline.PipelineInputs(
        public_winner=_write(
            root / "winner.json",
            seal(
                {
                    "schema": pipeline.WINNER_SCHEMA,
                    "status": "FROZEN",
                    "source": {
                        "repository": pipeline.DEEPSEEK_REPOSITORY,
                        "revision": pipeline.DEEPSEEK_REVISION,
                    },
                    "profile": {
                        "transport": "direct_presigned_range",
                        "scheduler_shape": "eight_files_low_per_file_concurrency",
                        "connection_reuse": True,
                    },
                    "real_stream_application": {
                        "outer_source_windows_maximum": 8,
                        "source_cache_bytes": 0,
                    },
                }
            ),
        ),
        full_manifest=manifest_path,
        full_reverify=_write(
            root / "reverify.json",
            seal(
                {
                    "schema": pipeline.FULL_REVERIFY_SCHEMA,
                    "status": "FULL_MODEL_STREAM_FULLY_REVERIFIED_RUNTIME_PENDING",
                }
            ),
        ),
        full_runtime_blocker=_write(
            root / "blocker.json",
            seal(
                {
                    "schema": pipeline.FULL_BLOCKER_SCHEMA,
                    "status": "FULL_STREAMED_RUNTIME_NO_REGISTERED_43_LAYER_ADAPTER",
                    "artifact": {"manifest_seal_sha256": full_manifest["seal_sha256"]},
                    "storage_accounting": {"raw_artifact_eviction_authorized": False},
                }
            ),
        ),
        child_baseline=_write(
            root / "child.json",
            seal(
                {
                    "schema": pipeline.CHILD_BASELINE_SCHEMA,
                    "status": "DSV4F_CHILD_BASELINE_FROZEN_FULL_STREAM_RUNTIME_PENDING",
                    "claim_boundary": {
                        "full_43_layer_runtime": False,
                        "full_43_layer_metal_dispatch": False,
                        "base_true_tps": False,
                        "direct_weight_transplant": False,
                        "kimi_or_glm_donor_weights_present": False,
                        "kimi_or_glm_training_performed": False,
                    },
                }
            ),
        ),
        latent_bridge=_write(
            root / "bridge.json",
            seal(
                {
                    "schema": pipeline.LATENT_BRIDGE_SCHEMA,
                    "status": "DSV4F_FUTURE_BRIDGE_INTERFACES_DECLARED_NO_DONOR_INHERITANCE",
                }
            ),
        ),
        hcli_live_suite=_write(
            root / "hcli.json",
            seal(
                {
                    "schema": pipeline.HCLI_SUITE_SCHEMA,
                    "status": "HCLI_LIVE_SUITE_EVIDENCE_SEALED_DIAGNOSTIC_ONLY",
                    "claim_boundary": {"full_43_layer_runtime": False},
                }
            ),
        ),
        base_tps_gate=_write(
            root / "tps.json",
            seal(
                {
                    "schema": pipeline.TPS_GATE_SCHEMA,
                    "status": "BASE_TRUE_TPS_WITHHELD",
                }
            ),
        ),
        glm_decision=_write(
            root / "glm.json",
            {
                "schema": pipeline.GLM_DECISION_SCHEMA,
                "decision": "FUNCTIONAL_PARTIAL_ONLY",
                "glm_full_stream": "DO_NOT_STREAM",
                "gates": {
                    name: {"passes": False}
                    for name in (
                        "FS2_next_layer_propagation",
                        "FS4_cross_layer_sharing",
                        "FS7_full_stream_admission",
                    )
                },
            },
        ),
        cascade_decision=_write(
            root / "cascade.json",
            {
                "schema": pipeline.CASCADE_SCHEMA,
                "answer": "NO",
                "verdict": "FUNCTIONAL_PARTIAL_ONLY",
            },
        ),
        kimi_ladder=_write(
            root / "ladder.json",
            {
                "schema": pipeline.LADDER_SCHEMA,
                "rungs": [
                    {
                        "rung": "F8",
                        "official_repo": "UNRESOLVED",
                        "revision": "UNRESOLVED",
                        "license": "UNRESOLVED",
                        "release_status": "PENDING_OFFICIAL_WEIGHTS",
                        "readiness_stage": "A0",
                    }
                ],
            },
        ),
        kimi_k26_release=_write(
            root / "k26.json",
            seal(
                {
                    "schema": "hawking.kimi_k26.source_release_for_glm52.v1",
                    "status": "RECONCILED_ALREADY_RELEASED",
                    "source": {"repo": "moonshotai/Kimi-K2.6"},
                }
            ),
        ),
        ramanujan_gate=_write(
            root / "ramanujan-gate.json",
            {
                "schema": pipeline.RAMANUJAN_GATE_SCHEMA,
                "status": "BLOCKED_ON_HAWKING_COMPLETION",
                "authority": {
                    "ramanujan_research_authorized": False,
                    "production_authority": False,
                },
            },
        ),
        ramanujan_offline_manifest=_write(
            root / "ramanujan-offline.json",
            {
                "schema": pipeline.RAMANUJAN_OFFLINE_SCHEMA,
                "status": "LOCAL_SOURCES_PARTIALLY_GENERATED",
            },
        ),
    )


def test_preflight_binds_every_lane_but_refuses_promotion(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    plan = pipeline.build_plan(inputs=inputs, workspace=tmp_path)

    verify(plan, label="Frankenstein plan")
    assert plan["status"] == "BLOCKED_BY_REQUIRED_GATES"
    assert plan["stages"][0]["id"] == "PUBLIC_XET_PATH"
    assert plan["stages"][0]["state"] == "COMPLETED_FROZEN"
    assert {stage["id"] for stage in plan["stages"] if stage["state"] == "BLOCKED"} >= {
        "DEEPSEEK_RUNTIME",
        "KIMI_K3_ADMISSION",
        "GLM_MATH_DIRECTOR",
        "ROUTE_AWARE_TRANSFER",
        "FOUR_WAY_ABLATION",
    }
    assert plan["transfer_contract"]["direct_weight_transplant"] is False
    assert "teacher_hidden_states" in plan["transfer_contract"]["prohibited_payloads"]

    result = pipeline.write_preflight(
        inputs=inputs,
        workspace=tmp_path,
        out=tmp_path / "plan.json",
        progress=tmp_path / "progress.jsonl",
    )
    assert result["plan_seal_sha256"] == plan["seal_sha256"]
    assert len((tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_tampered_sealed_winner_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    winner = json.loads(inputs.public_winner.read_text(encoding="utf-8"))
    winner["status"] = "FROZEN_BUT_TAMPERED"
    inputs.public_winner.write_text(json.dumps(winner), encoding="utf-8")

    with pytest.raises(pipeline.FrankensteinPipelineError, match="seal mismatch"):
        pipeline.build_plan(inputs=inputs, workspace=tmp_path)


def test_sustained_winner_binds_back_to_its_sealed_base_winner(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    base = json.loads(inputs.public_winner.read_text(encoding="utf-8"))
    followup = seal(
        {
            "schema": pipeline.SUSTAINED_FOLLOWUP_SCHEMA,
            "status": "COMPLETE_PROMOTED",
            "base_frozen_winner": {
                "path": str(inputs.public_winner),
                "seal_sha256": base["seal_sha256"],
            },
        }
    )
    followup_path = _write(tmp_path / "followup.json", followup)
    sustained = seal(
        {
            "schema": pipeline.SUSTAINED_WINNER_SCHEMA,
            "status": "FROZEN",
            "followup_path": str(followup_path),
            "followup_seal_sha256": followup["seal_sha256"],
            "base_frozen_winner_seal_sha256": base["seal_sha256"],
            "profile": {
                "transport": "direct_presigned_range",
                "scheduler_shape": "dynamic_work_stealing",
                "connection_reuse": True,
            },
            "real_stream_application": {
                "outer_source_windows_maximum": 8,
                "source_cache_bytes": 0,
            },
        }
    )
    sustained_path = _write(tmp_path / "sustained-winner.json", sustained)
    plan = pipeline.build_plan(inputs=replace(inputs, public_winner=sustained_path), workspace=tmp_path)
    assert plan["public_path"]["winner_schema"] == pipeline.SUSTAINED_WINNER_SCHEMA
    assert plan["input_bindings"]["public_xet_base_winner"]["receipt_seal_sha256"] == base["seal_sha256"]
