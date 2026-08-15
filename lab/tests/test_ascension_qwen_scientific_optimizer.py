"""Focused tests for the additive Qwen scientific optimizer lane."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from lab.operators.ascension_qwen_scientific_optimizer import (
    COMPONENT_PROFILE_SCHEMA,
    EXPERIMENT_SCHEMA,
    FRONTIER_SCHEMA,
    MODEL_SPECS,
    QwenScientificOptimizer,
    RUNTIME_SCHEMA,
    RUNTIME_STATUS,
    SCHEMA,
)
from lab.receipts import seal, verify


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_payload(codec: str) -> bytes:
    header = json.dumps({"schema": codec}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return b"HGRAVU01" + len(header).to_bytes(4, "little") + header + b"bounded-component-body"


def _populate_model(physical: Path, key: str, prefix: str, *, heartbeat: int = 1) -> None:
    base = physical / key
    evolution = base / "evolution"
    codec = "hawking.gravity.uniform_group.v1"
    artifact = evolution / "artifacts" / f"{key}.gravity"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(_artifact_payload(codec))
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    candidate = seal(
        {
            "schema": "unit.candidate.v1",
            "candidate_id": f"{key}-component",
            "artifact": {"path": str(artifact), "sha256": artifact_sha, "codec": codec},
            "representation": {"family": "uniform_q4_group64"},
        }
    )
    candidate_path = evolution / "candidates" / f"{key}-component.json"
    _write_json(candidate_path, candidate)
    champion = {
        "candidate_id": f"{key}-component",
        "record_path": str(candidate_path),
        "artifact_path": str(artifact),
        "artifact_sha256": artifact_sha,
        "artifact_bytes": artifact.stat().st_size,
        "physical_bpw": 1.2,
    }
    _write_json(
        evolution / "CHAMPIONS.json",
        seal(
            {
                "schema": "unit.champions.v1",
                "current_fastest_component": champion,
                "current_lowest_bpw_component": champion,
                "current_capable": {"candidate": None, "status": "UNSET"},
            }
        ),
    )
    _write_json(evolution / "PARETO_FRONTIER.json", seal({"schema": "unit.frontier.v1", "entries": []}))
    _write_json(
        evolution / "SOURCE_CONTENT_IDENTITY.json",
        seal({"schema": "unit.source.v1", "status": "IMMUTABLE_SOURCE_CONTENT_IDENTITY_BOUND"}),
    )
    _write_json(
        base / "complete-gravity" / f"{prefix}_CURRENT_SOURCE_SHARD_REVALIDATION.json",
        seal({"schema": "unit.revalidation.v1", "status": "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED"}),
    )
    _write_json(
        base / "complete-gravity" / f"{prefix}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json",
        seal(
            {
                "schema": "unit.complete-manifest.v1",
                "status": "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED",
                "complete_physical_bpw_ledger": {"complete_physical_bpw": 1.2},
            }
        ),
    )
    _write_json(
        base / "complete-gravity" / f"{prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json",
        seal({"schema": "unit.admission.v1", "status": "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"}),
    )
    _write_json(
        base / "state-kv" / f"{prefix}_STATE_KV_STATUS.json",
        seal({"schema": "unit.state-kv.v1", "status": "COMPONENT_EVIDENCE_ONLY"}),
    )
    _write_json(
        base / "complete-gravity" / f"{prefix}_COMPLETE_GRAVITY_STATUS.json",
        {"schema": "unit.pack-status.v1", "phase": "EARNED_COMPLETE_PHYSICAL_BINARY_CANDIDATE_UNQUALIFIED", "pid": os.getpid(), "progress": {"completed_tensors": 17}},
    )
    _write_json(
        base / "complete-runtime" / f"{prefix}_COMPLETE_RUNTIME_STATUS.json",
        {"schema": "unit.runtime-watchdog.v1", "phase": "BLOCKED_ON_NATIVE_DECODER", "pid": os.getpid(), "heartbeat": heartbeat},
    )
    legacy = "QWEN30_REAL_CAMPAIGN_STATUS.json" if key == "qwen30" else "QWEN80_PHYSICAL_CAMPAIGN_STATUS.json"
    _write_json(
        base / legacy,
        {
            "schema": "unit.worker-status.v1",
            "phase": "EVOLVING_PHYSICAL_CANDIDATE",
            "heartbeat": heartbeat,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "last_material_progress_at": "2026-08-08T00:00:00Z",
            "population": {"candidate_count": 4, "completed_candidate_count": 4},
            "current_experiment": {"candidate_id": f"{key}-current", "representation": "uniform_q4_group64"},
        },
    )


def _populate_shared(physical: Path) -> None:
    shared = physical / "qwen-family" / "dual-gravity"
    for name in (
        "ASCENSION_REPRESENTATION_GENOME.jsonl",
        "ASCENSION_KERNEL_GENOME.jsonl",
        "ASCENSION_SCHEDULER_GENOME.jsonl",
        "ASCENSION_NEGATIVE_SCIENCE.jsonl",
    ):
        _write_jsonl(
            shared / name,
            seal(
                {
                    "schema": "unit.shared-knowledge.v1",
                    "record_id": f"unit:{name}",
                    "status": "UNIT",
                    "recorded_at": "2026-08-08T00:00:00Z",
                }
            ),
        )


def _fixture_root(tmp_path: Path) -> Path:
    physical = tmp_path / "physical"
    _populate_model(physical, "qwen30", "QWEN30")
    _populate_model(physical, "qwen80", "QWEN80")
    _populate_shared(physical)
    return physical


def _experiment_documents(physical: Path) -> list[dict[str, object]]:
    return [
        verify(json.loads(path.read_text(encoding="utf-8")), label=str(path))
        for path in sorted((physical / "qwen-family" / "scientific-optimizer" / "experiments").glob("*.json"))
    ]


def test_optimizer_reads_all_required_evidence_and_runs_bounded_component_profiles(tmp_path: Path) -> None:
    physical = _fixture_root(tmp_path)
    optimizer = QwenScientificOptimizer(physical_root=physical)

    status = optimizer.run_cycle()

    assert status["schema"] == SCHEMA
    assert status["status"] == "REAL_EVIDENCE_DRIVEN_OPTIMIZATION_ADVANCING"
    experiments = _experiment_documents(physical)
    assert len(experiments) == 2
    for experiment in experiments:
        assert experiment["schema"] == EXPERIMENT_SCHEMA
        assert experiment["status"] == "BLOCKED_NATIVE_RUNTIME"
        reads = experiment["required_pre_experiment_reads"]
        assert {
            "current_complete_champion",
            "current_capability_frontier",
            "current_bpw_frontier",
            "current_complete_token_profile",
            "kernel_genome",
            "representation_genome",
            "scheduler_genome",
            "negative_science",
            "state_kv",
            "peer_latest_results",
        } <= set(reads)
        reasoning = experiment["reasoning"]
        assert len(reasoning["mechanisms"]) >= 3
        assert set(reasoning["conditions"]) == {"PASS", "FAIL", "REOPEN_LATER"}
        assert experiment["safe_component_profile_candidate"]["artifact_sha256"]

    profiles = list((physical / "qwen-family" / "scientific-optimizer" / "component-profiles").glob("*.json"))
    assert len(profiles) == 2
    for path in profiles:
        profile = verify(json.loads(path.read_text(encoding="utf-8")), label=str(path))
        assert profile["schema"] == COMPONENT_PROFILE_SCHEMA
        assert profile["status"] == "PASS_SOURCE_BOUND_PACKED_COMPONENT_IO_PROFILE_NOT_MODEL_TPS"
        assert profile["metrics"]["artifact_read_mib_per_second"] > 0
        assert "base_true_tps" not in profile["metrics"]

    frontier_path = physical / "qwen-family" / "scientific-optimizer" / "QWEN_SCIENTIFIC_OPTIMIZER_FRONTIER.json"
    frontier = verify(json.loads(frontier_path.read_text(encoding="utf-8")), label=str(frontier_path))
    assert frontier["schema"] == FRONTIER_SCHEMA
    assert set(frontier["component_io_frontier"]) == {"qwen30", "qwen80"}


def test_optimizer_rejects_heartbeat_only_worker_liveness_and_does_not_rerun_profile(tmp_path: Path) -> None:
    physical = _fixture_root(tmp_path)
    optimizer = QwenScientificOptimizer(physical_root=physical)
    optimizer.run_cycle()
    before = sorted((physical / "qwen-family" / "scientific-optimizer" / "component-profiles").glob("*.json"))

    for spec in MODEL_SPECS:
        worker_path = physical / spec.key / spec.legacy_worker_status
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        worker["heartbeat"] += 1
        _write_json(worker_path, worker)
    status = optimizer.run_cycle()
    after = sorted((physical / "qwen-family" / "scientific-optimizer" / "component-profiles").glob("*.json"))

    assert status["status"] == "AWAITING_MATERIAL_FRONTIER_CHANGE"
    assert after == before
    assert {row["state"] for row in status["worker_liveness"].values()} == {"HEARTBEAT_ONLY_REJECTED"}


def test_optimizer_rejects_candidate_selection_churn_without_committed_frontier_change() -> None:
    observation = {
        "worker": {"document": {"heartbeat": 9}},
        "process": {"worker": {"state": "PID_ALIVE", "pid": 42, "alive": True}},
        "material_marker": {"completed_candidate_count": 4, "runtime_receipt_seal": "stable"},
        "activity_marker": {"current_experiment_id": "candidate-b", "current_representation": "binary"},
    }
    previous = {
        "heartbeat": 8,
        "material_marker": {"completed_candidate_count": 4, "runtime_receipt_seal": "stable"},
        "activity_marker": {"current_experiment_id": "candidate-a", "current_representation": "binary"},
    }

    liveness = QwenScientificOptimizer._material_liveness(observation, previous)

    assert liveness["state"] == "HEARTBEAT_OR_SELECTION_ONLY_REJECTED"
    assert "selection churn" in liveness["claim_boundary"]


def test_new_peer_ledger_row_does_not_superficially_rerun_same_artifact_profile(tmp_path: Path) -> None:
    physical = _fixture_root(tmp_path)
    optimizer = QwenScientificOptimizer(physical_root=physical)
    optimizer.run_cycle()
    before = sorted((physical / "qwen-family" / "scientific-optimizer" / "component-profiles").glob("*.json"))

    kernel = physical / "qwen-family" / "dual-gravity" / "ASCENSION_KERNEL_GENOME.jsonl"
    with kernel.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(seal({"schema": "unit.peer.v1", "record_id": "peer:new-kernel", "status": "PASS"})) + "\n")
    optimizer.run_cycle()
    after = sorted((physical / "qwen-family" / "scientific-optimizer" / "component-profiles").glob("*.json"))

    assert after == before


def test_direct_packed_unscored_generation_becomes_capability_quality_blocker(tmp_path: Path) -> None:
    optimizer = QwenScientificOptimizer(physical_root=tmp_path / "physical")
    task = optimizer._native_runtime_task(
        MODEL_SPECS[0],
        {
            "direct_packed_generation": {
                "state": "OBSERVED_UNSEALED_DIRECT_PACKED_AUTOREGRESSIVE_TRACE_REQUIRES_CAPABILITY_QUALITY_GATE",
                "completion_text_unscored": "_urenne",
            },
            "current_complete_champion": {"manifest": {"complete_physical_bpw": 1.130366}},
            "source_authority": {},
            "state_kv": {},
        },
    )

    assert task["status"] == "BLOCKED_CAPABILITY_COHERENCE_AFTER_DIRECT_PACKED_GENERATION"
    assert task["blocker_category"] == "capability"
    assert len(task["mechanisms"]) >= 3
    assert set(task["conditions"]) == {"PASS", "FAIL", "REOPEN_LATER"}


def test_canonical_template_evidence_is_narrowly_ingested_and_emits_three_mechanism_task(tmp_path: Path) -> None:
    physical = _fixture_root(tmp_path)
    base = physical / "qwen30"
    _write_json(
        base / "complete-runtime" / "QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json",
        seal({"schema": RUNTIME_SCHEMA, "status": RUNTIME_STATUS, "runtime": {"all_layers_executed": True}}),
    )
    _write_json(
        base / "complete-runtime" / "QWEN30_HCLI_UNQUALIFIED_CHAT_SSE_TRANSPORT_RECEIPT.json",
        seal(
            {
                "schema": "unit.qwen30.transport.v1",
                "status": "EARNED_DIRECT_PACKED_NATIVE_CHAT_SSE_TRANSPORT_HCLI_UNQUALIFIED",
                "measurement": {"coherence": "UNSCORED_NOT_A_CAPABILITY_EVALUATION", "clean_tps": "NOT_MEASURED"},
            }
        ),
    )
    _write_json(
        base / "complete-runtime" / "QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY_RECEIPT.json",
        seal(
            {
                "schema": "unit.qwen30.simd.v1",
                "status": "REJECTED_QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY",
                "failures": ["template token mismatch"],
            }
        ),
    )
    _write_json(
        base / "complete-token-profiler" / "QWEN30_DIRECT_PACKED_GATE_UP_PAIR_COMPONENT_RECEIPT.json",
        seal(
            {
                "schema": "unit.qwen30.gate-up.v1",
                "status": "EARNED_SOURCE_BOUND_DIRECT_PACKED_GATE_UP_PAIR_COMPONENT_NOT_MODEL_TPS",
                "timing": {"p50_component_host_wall_speedup_ratio": 1.66},
            }
        ),
    )
    optimizer = QwenScientificOptimizer(physical_root=physical)
    shared = optimizer._shared_knowledge()
    qwen30 = optimizer._observe_model(MODEL_SPECS[0], shared)

    task = optimizer._qwen30_canonical_template_task(qwen30)
    assert task is not None
    assert task["status"] == "BLOCKED_CANONICAL_TEMPLATE_COHERENCE_KERNEL_INTEGRATION_DIAGNOSIS"
    assert len(task["mechanisms"]) == 3
    assert set(task["conditions"]) == {"PASS", "FAIL", "REOPEN_LATER"}
    assert "Qwen80" in task["prior_initialization"]["qwen80_transfer_rule"]

    evidence, advanced = optimizer._ingest_qwen30_canonical_template_evidence(qwen30)
    assert advanced is True
    assert evidence is not None
    assert evidence["status"] == "INGESTED_RUNTIME_TRANSPORT_REJECTION_AND_COMPONENT_KERNEL_EVIDENCE"
    assert "not a coherence" in evidence["claim_boundary"]

    negative_rows = [json.loads(line) for line in (physical / "qwen-family" / "dual-gravity" / "ASCENSION_NEGATIVE_SCIENCE.jsonl").read_text().splitlines()]
    assert any(row.get("mechanism") == "packed_binary_simdgroup_template_parity" for row in negative_rows)
    kernel_rows = [json.loads(line) for line in (physical / "qwen-family" / "dual-gravity" / "ASCENSION_KERNEL_GENOME.jsonl").read_text().splitlines()]
    assert any(row.get("mechanism") == "direct_packed_gate_up_pair_one_dispatch" for row in kernel_rows)


def test_paired_swiglu_control_is_component_only_and_exact_when_payloads_match() -> None:
    import numpy as np

    generator = np.random.default_rng(7)
    gate = generator.standard_normal((8, 16), dtype=np.float32)
    up = generator.standard_normal((8, 16), dtype=np.float32)
    measurement = QwenScientificOptimizer._paired_swiglu_control_quality(gate, up, gate, up)

    assert measurement["paired_vs_two_projection_direct_binary"]["exact_float32_equivalent"] is True
    assert measurement["paired_vs_two_projection_direct_binary"]["max_abs"] == 0.0
    assert measurement["source_to_direct_binary_swiglu"]["relative_l2"] == 0.0
    assert "not prompt/template parity" in measurement["claim_boundary"]


def test_router_residual_win_is_only_a_full_accounting_gated_repack_proposal(tmp_path: Path) -> None:
    optimizer = QwenScientificOptimizer(physical_root=tmp_path / "physical")
    quality = seal(
        {
            "schema": "unit.quality.v1",
            "status": "PASS_SOURCE_BOUND_SENSITIVE_ROUTER_REPRESENTATION_QUALITY_DIAGNOSTIC_NOT_CAPABILITY",
            "receipt_path": str(tmp_path / "quality.json"),
            "winner": {"name": "binary_sign_scale_sparse_fp16_residual_0.0025", "component_physical_bpw": 1.2574},
        }
    )
    proposal = optimizer._write_qwen30_repack_proposal(
        quality,
        {
            "current_complete_champion": {
                "manifest": {"path": "manifest", "seal_sha256": "a" * 64, "complete_physical_bpw": 1.130366},
                "admission": {"path": "admission", "seal_sha256": "b" * 64},
            }
        },
    )

    assert proposal is not None
    assert proposal["status"] == "PROPOSED_NOT_APPLIED_COMPLETE_ACCOUNTING_AND_CAPABILITY_RETEST_REQUIRED"
    assert proposal["baseline_control"]["replacement_forbidden_until_all_acceptance_gates_pass"] is True
    assert proposal["hard_full_artifact_accounting_gate"]["required_complete_physical_bpw_max"] == 1.5
    assert proposal["post_runtime_capability_retest_gate"]["state"] == "REQUIRED_AFTER_ANY_MATERIAL_REPRESENTATION_OR_RUNTIME_CHANGE"


def test_optimizer_seals_narrow_negative_science_on_corrupt_component_artifact(tmp_path: Path) -> None:
    physical = _fixture_root(tmp_path)
    candidate = physical / "qwen30" / "evolution" / "candidates" / "qwen30-component.json"
    document = verify(json.loads(candidate.read_text(encoding="utf-8")), label=str(candidate))
    document["artifact"]["sha256"] = "0" * 64
    _write_json(candidate, seal({key: value for key, value in document.items() if key != "seal_sha256"}))

    QwenScientificOptimizer(physical_root=physical).run_cycle()

    profiles = [
        verify(json.loads(path.read_text(encoding="utf-8")), label=str(path))
        for path in (physical / "qwen-family" / "scientific-optimizer" / "component-profiles").glob("*.json")
    ]
    failed = [row for row in profiles if row["model"] == "qwen30"]
    assert len(failed) == 1
    assert failed[0]["status"] == "FAIL_SOURCE_BOUND_PACKED_COMPONENT_IO_PROFILE"
    negative_path = physical / "qwen-family" / "dual-gravity" / "ASCENSION_NEGATIVE_SCIENCE.jsonl"
    negatives = [json.loads(line) for line in negative_path.read_text(encoding="utf-8").splitlines()]
    assert any(row.get("record_id", "").startswith("negative-optimizer:qwen30-") for row in negatives)
