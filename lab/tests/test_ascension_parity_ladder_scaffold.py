"""P0–P13 parity-ladder scaffold tests for Qwen3 bootstrap families.

These tests assert harness contracts and honest refusal — they do **not** load
Qwen weights or run Gravity. Live rung bodies are filled after stream.

Reference discipline: DSV4F NumericParityV21Only + PASS_FULL_STACK / REJECT_*.
"""

from __future__ import annotations

import pytest

from lab.operators.ascension_parity_ladder import (
    FORBIDDEN_UNIVERSAL_BPW_REQUIREMENT,
    GRAVITY_LADDER_ORDER,
    MODEL_LADDER_PIPELINE,
    ModelFamily,
    PARITY_RUNGS,
    ParityClassification,
    ParityLadderHarness,
    QWEN3_MOE_STAGES,
    QWEN3_NEXT_ARCHITECTURE_REQUIREMENTS,
    QWEN3_NEXT_STATE_GATES,
    RungStatus,
    all_family_scaffold_receipts,
    promote_rung_status,
    verify_ladder_receipt,
)
from lab.receipts import verify


# ---------------------------------------------------------------------------
# Inventory / schema contracts
# ---------------------------------------------------------------------------


def test_parity_ladder_has_exactly_fourteen_rungs_p0_through_p13() -> None:
    assert len(PARITY_RUNGS) == 14
    assert [r.id for r in PARITY_RUNGS] == [f"P{i}" for i in range(14)]
    assert PARITY_RUNGS[0].name == "tokenizer/template"
    assert PARITY_RUNGS[13].name == "restart/reload"
    assert PARITY_RUNGS[0].requires_weights is False
    assert PARITY_RUNGS[9].requires_gpu is True
    assert PARITY_RUNGS[9].requires_fallback_zero is True


def test_model_ladder_pipeline_matches_bible_section_30() -> None:
    assert MODEL_LADDER_PIPELINE[0] == "DISCOVER"
    assert MODEL_LADDER_PIPELINE[-1] == "ROTATE"
    assert "PARITY" in MODEL_LADDER_PIPELINE
    assert "GRAVITY" in MODEL_LADDER_PIPELINE
    assert len(MODEL_LADDER_PIPELINE) == 15


def test_gravity_ladder_order_and_forbidden_universal_bpw() -> None:
    assert [s.value for s in GRAVITY_LADDER_ORDER] == [
        "source_authority",
        "quality_anchor",
        "performance_anchor",
        "gravity_equilibrium_artifact",
    ]
    assert FORBIDDEN_UNIVERSAL_BPW_REQUIREMENT == 1.5


def test_numeric_parity_v2_1_is_not_exact_storage() -> None:
    assert ParityClassification.NUMERIC_PARITY_V2_1_ONLY.is_exact_storage() is False
    assert ParityClassification.EXACT_STORAGE.is_exact_storage() is True
    assert (
        ParityClassification.NUMERIC_PARITY_V2_1_ONLY.as_str()
        == "NUMERIC_PARITY_V2_1_ONLY"
    )


# ---------------------------------------------------------------------------
# Family parameterization
# ---------------------------------------------------------------------------


def test_qwen3_moe_30b_harness_stages_and_bootstrap_target() -> None:
    h = ParityLadderHarness(family=ModelFamily.QWEN3_MOE)
    assert h.family_key == "QWEN3_MOE"
    assert h.stages() == QWEN3_MOE_STAGES
    assert "router_top_k" in h.stages()
    assert "attention" in h.stages()
    target = h.bootstrap_target()
    assert target["hf_id"] == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert target["role"] == "executor"
    assert h.state_gates() == ()


def test_qwen3_next_80b_has_distinct_stages_and_state_gates() -> None:
    h = ParityLadderHarness(family=ModelFamily.QWEN3_NEXT)
    stages = h.stages()
    assert "gated_deltanet_state" in stages
    assert "deltanet_update" in stages
    assert "hybrid_schedule_slot" in stages
    assert "router_top10" in stages
    assert "state_memory_accounting" in stages
    assert len(h.state_gates()) == 7
    assert [g.id for g in h.state_gates()] == [f"SG{i}" for i in range(7)]
    reqs = set(h.architecture_requirements())
    assert "hybrid_3_deltanet_1_gated_attention_schedule" in reqs
    assert "routing_512_expert_top10" in reqs
    assert set(QWEN3_NEXT_ARCHITECTURE_REQUIREMENTS) == reqs


# ---------------------------------------------------------------------------
# Stub rung functions (P0–P13) — names ready for live fill-in
# ---------------------------------------------------------------------------


def _moe() -> ParityLadderHarness:
    return ParityLadderHarness(family=ModelFamily.QWEN3_MOE, weights_present=False)


def test_p0_tokenizer_template_scaffold() -> None:
    """P0 tokenizer/template — may run without weights once tokenizer is local."""
    h = _moe()
    rung = PARITY_RUNGS[0]
    receipt = h.stub_rung_receipt(rung)
    verify(receipt, label="P0")
    assert receipt["rung"]["id"] == "P0"
    assert receipt["implementation"]["body_filled"] is False
    # Without weights, tokenizer-only still scaffold-pending (no tokenizer file yet)
    assert receipt["status"] in {
        RungStatus.SCAFFOLD_PENDING.value,
        RungStatus.PASS_SCAFFOLD_CONTRACT.value,
    }


def test_p1_embedding_norm_scaffold() -> None:
    receipt = _moe().stub_rung_receipt(PARITY_RUNGS[1])
    verify(receipt, label="P1")
    assert receipt["rung"]["id"] == "P1"
    assert receipt["status"] == RungStatus.REJECT_WEIGHTS_ABSENT.value


def test_p2_qkv_rope_kv_scaffold() -> None:
    receipt = _moe().stub_rung_receipt(PARITY_RUNGS[2])
    verify(receipt, label="P2")
    assert receipt["rung"]["id"] == "P2"
    assert receipt["status"] == RungStatus.REJECT_WEIGHTS_ABSENT.value


def test_p3_attention_scaffold() -> None:
    receipt = _moe().stub_rung_receipt(PARITY_RUNGS[3])
    verify(receipt, label="P3")
    assert receipt["rung"]["id"] == "P3"
    assert receipt["status"] == RungStatus.REJECT_WEIGHTS_ABSENT.value


def test_p4_router_top_k_scaffold() -> None:
    receipt = _moe().stub_rung_receipt(PARITY_RUNGS[4])
    verify(receipt, label="P4")
    assert receipt["rung"]["id"] == "P4"
    assert receipt["rung"]["name"] == "router/top-k"
    assert receipt["status"] == RungStatus.REJECT_WEIGHTS_ABSENT.value


def test_p5_one_expert_scaffold() -> None:
    receipt = _moe().stub_rung_receipt(PARITY_RUNGS[5])
    verify(receipt, label="P5")
    assert receipt["rung"]["id"] == "P5"
    assert receipt["status"] == RungStatus.REJECT_WEIGHTS_ABSENT.value


def test_p6_full_moe_scaffold() -> None:
    receipt = _moe().stub_rung_receipt(PARITY_RUNGS[6])
    verify(receipt, label="P6")
    assert receipt["rung"]["id"] == "P6"
    assert receipt["status"] == RungStatus.REJECT_WEIGHTS_ABSENT.value


def test_p7_one_layer_scaffold() -> None:
    receipt = _moe().stub_rung_receipt(PARITY_RUNGS[7])
    verify(receipt, label="P7")
    assert receipt["rung"]["id"] == "P7"
    assert receipt["status"] == RungStatus.REJECT_WEIGHTS_ABSENT.value


def test_p8_early_middle_late_scaffold() -> None:
    receipt = _moe().stub_rung_receipt(PARITY_RUNGS[8])
    verify(receipt, label="P8")
    assert receipt["rung"]["id"] == "P8"
    assert receipt["status"] == RungStatus.REJECT_WEIGHTS_ABSENT.value


def test_p9_first_token_scaffold() -> None:
    receipt = _moe().stub_rung_receipt(PARITY_RUNGS[9])
    verify(receipt, label="P9")
    assert receipt["rung"]["id"] == "P9"
    assert receipt["rung"]["requires_gpu"] is True
    assert receipt["rung"]["requires_fallback_zero"] is True
    assert receipt["status"] == RungStatus.REJECT_WEIGHTS_ABSENT.value


def test_p10_continuation_full_logits_scaffold() -> None:
    receipt = _moe().stub_rung_receipt(PARITY_RUNGS[10])
    verify(receipt, label="P10")
    assert receipt["rung"]["id"] == "P10"
    assert receipt["status"] == RungStatus.REJECT_WEIGHTS_ABSENT.value


def test_p11_tool_json_edit_behavior_scaffold() -> None:
    receipt = _moe().stub_rung_receipt(PARITY_RUNGS[11])
    verify(receipt, label="P11")
    assert receipt["rung"]["id"] == "P11"
    assert receipt["rung"]["capability_rung"] is True
    assert receipt["status"] == RungStatus.REJECT_WEIGHTS_ABSENT.value


def test_p12_long_generation_scaffold() -> None:
    receipt = _moe().stub_rung_receipt(PARITY_RUNGS[12])
    verify(receipt, label="P12")
    assert receipt["rung"]["id"] == "P12"
    assert receipt["status"] == RungStatus.REJECT_WEIGHTS_ABSENT.value


def test_p13_restart_reload_scaffold() -> None:
    receipt = _moe().stub_rung_receipt(PARITY_RUNGS[13])
    verify(receipt, label="P13")
    assert receipt["rung"]["id"] == "P13"
    assert receipt["status"] == RungStatus.REJECT_WEIGHTS_ABSENT.value
    assert receipt["parity_classification"] == ParityClassification.SCAFFOLD_PENDING.value
    assert receipt["parity_is_exact_storage"] is False


# ---------------------------------------------------------------------------
# Promotion vocabulary + sealed ladder receipt
# ---------------------------------------------------------------------------


def test_promote_rung_status_honest_refusal_and_full_stack() -> None:
    assert (
        promote_rung_status(
            parity_pass=True, fallback_count=None, gpu_dispatches=0, full_stack=True
        )
        == RungStatus.SCAFFOLD_PENDING
    )
    assert (
        promote_rung_status(
            parity_pass=True, fallback_count=1, gpu_dispatches=10, full_stack=True
        )
        == RungStatus.REJECT_FALLBACK_NONZERO
    )
    assert (
        promote_rung_status(
            parity_pass=True, fallback_count=0, gpu_dispatches=0, full_stack=True
        )
        == RungStatus.REJECT_NO_REAL_GPU_DISPATCH
    )
    assert (
        promote_rung_status(
            parity_pass=False, fallback_count=0, gpu_dispatches=5, full_stack=True
        )
        == RungStatus.REJECT_PARITY
    )
    assert (
        promote_rung_status(
            parity_pass=True, fallback_count=0, gpu_dispatches=5, full_stack=False
        )
        == RungStatus.PASS_NUMERIC_V2_1_ONLY
    )
    assert (
        promote_rung_status(
            parity_pass=True, fallback_count=0, gpu_dispatches=5, full_stack=True
        )
        == RungStatus.PASS_FULL_STACK
    )


def test_scaffold_ladder_receipt_seals_and_never_claims_live_work() -> None:
    h = ParityLadderHarness(family=ModelFamily.QWEN3_MOE)
    receipt = h.scaffold_ladder_receipt()
    verify_ladder_receipt(receipt)
    assert receipt["status"] == RungStatus.PASS_SCAFFOLD_CONTRACT.value
    assert receipt["rung_count"] == 14
    assert receipt["honesty"]["live_model_work"] is False
    assert receipt["honesty"]["qwen_download"] is False
    assert receipt["claim_boundary"]["base_true_tps"] is False
    assert receipt["claim_boundary"]["universal_1_5_bpw_required"] is False
    assert receipt["forbidden_universal_bpw"] == 1.5
    # Nested rung receipts are individually sealed
    for rung_doc in receipt["rungs"]:
        verify(rung_doc, label=rung_doc["rung"]["id"])


def test_gravity_ladder_and_pipeline_receipts_seal() -> None:
    h = ParityLadderHarness(family=ModelFamily.QWEN3_MOE)
    g = h.gravity_ladder_receipt()
    p = h.model_ladder_pipeline_receipt()
    verify(g, label="gravity ladder")
    verify(p, label="model ladder pipeline")
    assert g["universal_1_5_bpw_forbidden"] is True
    assert p["pipeline"][-1] == "ROTATE"
    assert "TG_RUNG_DESCENT" in p["rotation_triggers"]
    assert "TWO_FAILED_ARCHITECTURES" in p["rotation_triggers"]


def test_all_family_scaffold_receipts_cover_moe_and_next() -> None:
    all_rx = all_family_scaffold_receipts()
    assert set(all_rx) == {"QWEN3_MOE", "QWEN3_NEXT"}
    for fam, doc in all_rx.items():
        verify(doc, label=fam)
        assert doc["family"] == fam
        assert doc["honesty"]["live_model_work"] is False


def test_claim_boundary_forbids_inflating_numeric_to_exact() -> None:
    h = ParityLadderHarness(family=ModelFamily.QWEN3_MOE)
    receipt = h.stub_rung_receipt(
        PARITY_RUNGS[7],
        status=RungStatus.PASS_NUMERIC_V2_1_ONLY,
        parity=ParityClassification.NUMERIC_PARITY_V2_1_ONLY,
    )
    verify(receipt, label="P7 numeric")
    assert receipt["parity_is_exact_storage"] is False
    assert receipt["claim_boundary"]["exact_storage_parity"] is False
    assert receipt["claim_boundary"]["full_stack_runtime"] is False


@pytest.mark.parametrize("gate", list(QWEN3_NEXT_STATE_GATES))
def test_qwen3_next_state_gate_stubs_refuse_without_weights(gate) -> None:
    h = ParityLadderHarness(family=ModelFamily.QWEN3_NEXT, weights_present=False)
    receipt = h.stub_state_gate_receipt(gate)
    verify(receipt, label=gate.id)
    assert receipt["status"] == RungStatus.REJECT_WEIGHTS_ABSENT.value
    assert receipt["implementation"]["body_filled"] is False
