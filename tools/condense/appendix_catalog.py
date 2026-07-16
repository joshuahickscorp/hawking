#!/usr/bin/env python3.12
"""Machine-readable coverage map for the 25 Appendix optimization sectors."""

from __future__ import annotations

import json
import sys


SCHEMA = "hawking.appendix_capability_catalog.v1"
CURRENCIES = {"arithmetic", "bytes", "state", "serial", "communication", "coordination"}
STATES = {
    "active_corpus",
    "built_measure",
    "partial_measure",
    "scaffolded",
    "oracle_first",
    "parked_by_evidence",
    "architecture_research",
}


def _sector(
    number: int,
    name: str,
    currencies: tuple[str, ...],
    state: str,
    tq_relevance: str,
    evidence: tuple[str, ...],
    next_gate: str,
    *,
    heavy: bool,
) -> dict:
    return {
        "id": f"S{number:02d}",
        "number": number,
        "name": name,
        "currencies": list(currencies),
        "hawking_state": state,
        "tq_relevance": tq_relevance,
        "evidence": list(evidence),
        "next_gate": next_gate,
        "requires_exclusive_heavy_lease": heavy,
        "mutates_active_corpus": False,
    }


def build_catalog() -> dict:
    s = [
        _sector(1, "model_size_and_shape", ("arithmetic", "bytes"), "active_corpus", "direct",
                ("current multi-scale ladder",), "compare quality-adjusted serving by scale", heavy=True),
        _sector(2, "knowledge_distillation", ("arithmetic", "bytes"), "oracle_first", "training",
                ("computational_efficiency_paradigms_2026_07_11.md",), "teacher/student trace oracle before training", heavy=True),
        _sector(3, "model_cascades_and_routing", ("arithmetic", "bytes", "coordination"), "partial_measure", "indirect",
                ("hawking-orch routing", "capability frontier"), "calibrated abstention and retry-inclusive cost receipt", heavy=True),
        _sector(4, "adaptive_computation", ("arithmetic", "serial"), "oracle_first", "training",
                ("post-hoc sparsity kill ledger",), "layer/token skip oracle with exact fallback", heavy=True),
        _sector(5, "mixture_of_experts", ("arithmetic", "bytes", "communication"), "partial_measure", "direct",
                ("MoE loader and expert paging", "hot/cold expert negative evidence"), "routing locality and paging trace", heavy=True),
        _sector(6, "numeric_precision", ("arithmetic", "bytes", "state"), "active_corpus", "direct",
                ("TQ ladder", "Doctor", "KV precision experiments"), "artifact-bound quality and byte-roofline receipt", heavy=True),
        _sector(7, "sparsity_and_pruning", ("arithmetic", "bytes"), "parked_by_evidence", "direct",
                ("dead_levers.md", "FFN block-sparsity oracle"), "new trained structured-sparsity oracle only", heavy=True),
        _sector(8, "attention_architecture", ("arithmetic", "bytes", "state"), "architecture_research", "indirect",
                ("Qwen attention traces",), "trained GQA/local/indexable-attention model comparison", heavy=True),
        _sector(9, "alternative_sequence_architectures", ("arithmetic", "state", "serial"), "built_measure", "indirect",
                ("RWKV-7 runtime", "Mamba-2 runtime"), "same-workload capability-efficiency comparison", heavy=True),
        _sector(10, "context_and_token_count", ("arithmetic", "bytes", "state", "serial"), "scaffolded", "indirect",
                ("Appendix token-budget baseline",), "hold quality and answer length while sweeping prompt/RAG/output budgets", heavy=True),
        _sector(11, "kv_cache_engineering", ("bytes", "state", "coordination"), "partial_measure", "indirect",
                ("F16 and INT4 KV paths", "state persistence"), "real-generation parity across context/batch strata", heavy=True),
        _sector(12, "prefix_and_state_reuse", ("arithmetic", "bytes", "state"), "built_measure", "indirect",
                ("RAM prefix cache", "disk prefix state"), "hit-rate and byte-second ledger on corpus prompts", heavy=True),
        _sector(13, "speculative_and_multi_token_decode", ("serial", "bytes", "coordination"), "scaffolded", "direct",
                ("spec_decode_reentry_appendix_2026_07_14.md",), "TQ-native B=1..8 parity before proposer work", heavy=True),
        _sector(14, "kernel_and_compiler_engineering", ("arithmetic", "bytes", "coordination"), "built_measure", "direct",
                ("Metal fused kernels", "TQ runtime modes"), "per-shape bytes/occupancy/latency autotune receipt", heavy=True),
        _sector(15, "batch_construction", ("bytes", "coordination"), "partial_measure", "direct",
                ("multi-sequence paths",), "continuous-batch SLO sweep on served TQ", heavy=True),
        _sector(16, "runtime_scheduling", ("serial", "coordination"), "partial_measure", "indirect",
                ("queue and resource gates",), "mixed prefill/decode useful-token scheduler receipt", heavy=True),
        _sector(17, "prefill_decode_separation", ("communication", "coordination", "state"), "parked_by_evidence", "indirect",
                ("single-SoC shared-memory topology",), "KV-transfer oracle must beat same-device scheduling", heavy=True),
        _sector(18, "parallelism", ("arithmetic", "communication", "coordination"), "oracle_first", "indirect",
                ("single-node target",), "communication-inclusive model before adding devices", heavy=True),
        _sector(19, "interconnect_and_placement", ("bytes", "communication", "state"), "partial_measure", "direct",
                ("unified-memory runtime", "expert paging"), "tier and placement bytes plus pressure receipt", heavy=True),
        _sector(20, "cpu_accelerator_coordination", ("communication", "coordination"), "built_measure", "direct",
                ("TokenCommandBuffer", "buffer reuse", "GPU sampling paths"), "phase timing with host waits and launch counts", heavy=True),
        _sector(21, "result_level_caching", ("arithmetic", "bytes", "state"), "parked_by_evidence", "indirect",
                ("exact prefix reuse", "semantic-cache no-uplift oracle"), "real-corpus invalidation/privacy-aware oracle", heavy=True),
        _sector(22, "decoding_policy", ("arithmetic", "serial"), "scaffolded", "indirect",
                ("greedy and sampling engine",), "quality/length/candidate-count factorial baseline", heavy=True),
        _sector(23, "hardware_specialization", ("arithmetic", "bytes", "communication"), "active_corpus", "direct",
                ("TQ direct Metal execution", "compute-for-memory Appendix"), "runtime-path parity then device roofline", heavy=True),
        _sector(24, "power_and_thermal_behavior", ("arithmetic", "coordination"), "partial_measure", "direct",
                ("energy and thermal probes",), "accepted tokens/s and joules/token under stable thermal strata", heavy=True),
        _sector(25, "training_for_deployment", ("arithmetic", "bytes", "state"), "architecture_research", "training",
                ("Doctor corpus", "TQ served distribution"), "trace-bound QAT/structured/drafter training oracle", heavy=True),
    ]
    return {
        "schema": SCHEMA,
        "source_sector_count": 25,
        "active_run_is_primary_corpus": True,
        "execution_supported": False,
        "sectors": s,
    }


def _selftest() -> int:
    first = build_catalog()
    assert first == build_catalog()
    sectors = first["sectors"]
    assert len(sectors) == 25
    assert [item["number"] for item in sectors] == list(range(1, 26))
    assert len({item["id"] for item in sectors}) == 25
    for item in sectors:
        assert set(item["currencies"]) <= CURRENCIES
        assert item["hawking_state"] in STATES
        assert item["tq_relevance"] in {"direct", "indirect", "training"}
        assert item["next_gate"]
        assert item["mutates_active_corpus"] is False
    print("appendix_catalog.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--selftest"]:
        return _selftest()
    if argv == ["--catalog"]:
        print(json.dumps(build_catalog(), indent=2, sort_keys=True))
        return 0
    print("usage: appendix_catalog.py --catalog | --selftest", file=sys.stderr)
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
