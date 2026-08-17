"""In-process fixtures for lineage tests. Not a production surface."""
from __future__ import annotations

from typing import Any

from lab.lineage.canon import labeled_sha
from lab.lineage.identity import (
    DEFAULT_BENCHMARK_FINGERPRINT,
    DEFAULT_CAPABILITY_CONTRACT,
    GENESIS_COMPLETE_TOKEN_NS,
    GenesisInstance,
    Invoker,
    make_qwen38_genesis,
)
from lab.lineage.promotion import REQUIRED_PROTECTED_TESTS
from lab.lineage.state import LineageState
from lab.lineage.transfer import parent_research_payload


EXTERNAL = Invoker(principal="lineage_gate", identity="protected-lineage-gate")

PARENT_REPS = (
    GENESIS_COMPLETE_TOKEN_NS,
    GENESIS_COMPLETE_TOKEN_NS + 40_000,
    GENESIS_COMPLETE_TOKEN_NS - 25_000,
)
CHILD_WALL = 30_000_000
CHILD_REPS = (30_010_000, 29_980_000, 30_010_000)


def make_child(
    parent: GenesisInstance,
    *,
    instance_id: str = "genesis-child-g1",
    generation: int = 1,
    complete_token_ns: int = CHILD_WALL,
    representation_bpw: float | None = None,
    capability: dict[str, float] | None = None,
    silent_fallback_ids: tuple[str, ...] = (),
    benchmark_fingerprint: str | None = None,
    artifact_sha: str | None = None,
    runtime_sha: str | None = None,
    kernel_genome_sha: str | None = None,
    lane: str = "representation",
) -> GenesisInstance:
    return GenesisInstance(
        instance_id=instance_id,
        generation=generation,
        artifact_sha=artifact_sha or labeled_sha("artifact/child-g1"),
        runtime_sha=runtime_sha or labeled_sha("runtime/child-g1"),
        kernel_genome_sha=kernel_genome_sha or labeled_sha("genome/child-g1"),
        representation_bpw=parent.representation_bpw if representation_bpw is None else representation_bpw,
        complete_token_ns=complete_token_ns,
        capability=dict(DEFAULT_CAPABILITY_CONTRACT if capability is None else capability),
        silent_fallback_ids=silent_fallback_ids,
        benchmark_fingerprint=benchmark_fingerprint or parent.benchmark_fingerprint,
        identity=dict(parent.identity),
        lane=lane,
        valid=True,
    )


def passing_evidence(parent: GenesisInstance, child: GenesisInstance) -> dict[str, Any]:
    return {
        "measurement": {
            "artifact_sha": child.artifact_sha,
            "complete_token_ns_reps": list(CHILD_REPS),
            "parent_complete_token_ns_reps": list(PARENT_REPS),
            "regime": "warm",
            "timing_authority": "MTLCommandBuffer GPUStartTime/GPUEndTime after wait",
            "benchmark_fingerprint": DEFAULT_BENCHMARK_FINGERPRINT,
            "paired": True,
            "alternating_reps": 3,
        },
        "representation": {"bpw": child.representation_bpw, "receipt_ref": "rep-child-g1"},
        "genome": {
            "runtime_sha": child.runtime_sha,
            "kernel_genome_sha": child.kernel_genome_sha,
            "receipt_ref": "genome-child-g1",
        },
        "artifact_receipt": {"sha": child.artifact_sha},
        "protected_tests": [{"name": name, "status": "PASS"} for name in REQUIRED_PROTECTED_TESTS],
        "state_transfer": {
            "checksum_verified": True,
            "checksum_sha256": labeled_sha("xfer/child-g1"),
        },
        "rollback_artifact": {"valid": True, "instance_id": parent.instance_id},
        "parent_contract": {
            "capability": dict(parent.capability),
            "benchmark_fingerprint": parent.benchmark_fingerprint,
            "silent_fallback_ids": list(parent.silent_fallback_ids),
            "complete_token_ns": parent.complete_token_ns,
            "representation_bpw": parent.representation_bpw,
            "tps": parent.tps,
        },
        "child_tps": child.tps,
        "new_silent_fallbacks": [],
    }


def armed_lineage() -> tuple[LineageState, GenesisInstance, GenesisInstance, dict[str, Any], Invoker]:
    parent = make_qwen38_genesis()
    state = LineageState()
    state.install(parent)
    child = make_child(parent)
    state.nominate(child)
    return state, parent, child, passing_evidence(parent, child), EXTERNAL


def science_payload() -> dict[str, Any]:
    return parent_research_payload(
        next_bottleneck="weight_addressing 21.293 ms (60.44%) — attack the addressing assumption, not the 97.6% ceiling",
        task_graph=[{"id": "rung-1", "target_ns": 10_000_000}],
        memories=["qwen38 seated 2026-08-16", "q80 disqualified"],
        genomes={"runtime": "ascension_qwen38_hybrid_greedy"},
        negative_science=[{"id": "NS-001", "class": "UNREACHABLE"}],
        active_hypotheses=[{"id": "H-per-layer", "text": "per-layer codec unexplored"}],
        grok_findings=[{"id": "G024", "text": "qwen38 token-ns ledger closed"}],
        open_experiments=[{"id": "E-per-head", "status": "open"}],
    )
