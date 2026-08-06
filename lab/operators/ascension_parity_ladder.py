"""Family-parameterized parity ladder harness (scaffold).

Generalizes the DeepSeek-V4-Flash (DSV4F) discipline that landed tonight:

- rung ladder P0..P13 with honest claim boundaries
- ``NumericParityV21Only`` classification (not exact-storage until sealed)
- status vocabulary: ``PASS_FULL_STACK`` / ``REJECT_*`` / ``*_WITHHELD``
- receipt schemas + seals (``lab.receipts.seal``)
- fallback=0 + real GPU dispatch requirements on capability rungs
- complete-token stage inventory parameterized by architecture family

Reference implementations (read-only; do not edit here):
- ``crates/hawking-core/src/gravity_deepseek_v4_p4b_device.rs``
  (``DeepSeekV4P4bParityClassification::NumericParityV21Only``)
- ``lab/operators/deepseek_v4_gravity.py``
  (``_COMPLETE_TOKEN_PROFILE_STAGES``, ``BASE_TRUE_TPS_WITHHELD``, claim_boundary)
- ``tools/condense/tests/test_deepseek_v4_complete_token_profile.py``
- ``tools/condense/tests/test_deepseek_v4_child_baseline.py``
- ``lab/operators/frankenstein_teacher_forced_executor.py`` (``PASS_FULL_STACK``)

This module is **scaffold only**. It never downloads Qwen weights, never opens a
Gravity artifact, and never claims BASE_TRUE_TPS. When weights are absent the
harness returns honest ``REJECT_WEIGHTS_ABSENT`` / ``SCAFFOLD_PENDING`` statuses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from lab.receipts import seal, verify

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

PARITY_LADDER_RECEIPT_SCHEMA = "hawking.ascension.parity_ladder_receipt.v1"
RUNG_RECEIPT_SCHEMA = "hawking.ascension.parity_rung_receipt.v1"
GRAVITY_LADDER_RECEIPT_SCHEMA = "hawking.ascension.gravity_ladder_receipt.v1"
STATE_GATE_RECEIPT_SCHEMA = "hawking.ascension.state_gate_receipt.v1"
FAMILY_KERNEL_RECEIPT_SCHEMA = "hawking.ascension.family_kernel_plan.v1"
MODEL_LADDER_PIPELINE_SCHEMA = "hawking.ascension.model_ladder_pipeline.v1"

# ---------------------------------------------------------------------------
# Family keys (bible §29)
# ---------------------------------------------------------------------------


class ModelFamily(str, Enum):
    """Architecture-family keys for the shared parity harness."""

    QWEN3_MOE = "QWEN3_MOE"
    QWEN3_NEXT = "QWEN3_NEXT"
    DEEPSEEK_V4 = "DEEPSEEK_V4"
    LLAMA = "LLAMA"
    MISTRAL_MIXTRAL = "MISTRAL_MIXTRAL"
    STATE_SPACE_HYBRID = "STATE_SPACE_HYBRID"


# Bootstrap targets (bible §8 / §9) — identity pins filled at download time.
BOOTSTRAP_TARGETS: dict[str, dict[str, Any]] = {
    ModelFamily.QWEN3_MOE.value: {
        "role": "executor",
        "hf_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "scale_label": "30B-A3B",
        "routing": {"routed_experts": None, "top_k": 8, "shared_expert": True},
        "attention": "standard_gqa_or_mha",
        "notes": "First self-optimization vehicle; first production HCLI model.",
    },
    ModelFamily.QWEN3_NEXT.value: {
        "role": "reviewer",
        "hf_id": "Qwen/Qwen3-Coder-Next",  # pin revision at stream time
        "scale_label": "80B-class",
        "routing": {"routed_experts": 512, "top_k": 10, "shared_expert": True},
        "attention": "hybrid_3_deltanet_1_gated_attention",
        "notes": "Distinct architecture family; exact-model path before generalization.",
    },
}


# ---------------------------------------------------------------------------
# Parity classification (generalizes NumericParityV21Only)
# ---------------------------------------------------------------------------


class ParityClassification(str, Enum):
    """Honesty labels for numeric comparison strength.

    Mirrors ``DeepSeekV4P4bParityClassification::NumericParityV21Only``:
    not convertible to exact-storage until a sealed e2e receipt is earned.
    """

    SCAFFOLD_PENDING = "SCAFFOLD_PENDING"
    NUMERIC_PARITY_V2_1_ONLY = "NUMERIC_PARITY_V2_1_ONLY"
    EXACT_STORAGE = "EXACT_STORAGE"
    REJECTED = "REJECTED"

    def is_exact_storage(self) -> bool:
        return self is ParityClassification.EXACT_STORAGE

    def as_str(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Status vocabulary (generalizes PASS_FULL_STACK / REJECT_* / WITHHELD)
# ---------------------------------------------------------------------------


class RungStatus(str, Enum):
    """Terminal status for one parity-ladder rung or gauntlet step."""

    # Scaffold / pre-weight honesty
    SCAFFOLD_PENDING = "SCAFFOLD_PENDING"
    REJECT_WEIGHTS_ABSENT = "REJECT_WEIGHTS_ABSENT"
    REJECT_ARTIFACT_ABSENT = "REJECT_ARTIFACT_ABSENT"

    # Execution honesty (DSV4F pattern)
    REJECT_FALLBACK_NONZERO = "REJECT_FALLBACK_NONZERO"
    REJECT_NO_REAL_GPU_DISPATCH = "REJECT_NO_REAL_GPU_DISPATCH"
    REJECT_PARITY = "REJECT_PARITY"
    REJECT_CAPABILITY = "REJECT_CAPABILITY"
    REJECT_CLAIM_BOUNDARY = "REJECT_CLAIM_BOUNDARY"
    REJECT_PROMOTION = "REJECT_PROMOTION"

    # Partial / diagnostic (never promote as full stack)
    PASS_SCAFFOLD_CONTRACT = "PASS_SCAFFOLD_CONTRACT"
    PASS_DIAGNOSTIC_ONLY = "PASS_DIAGNOSTIC_ONLY"
    PASS_NUMERIC_V2_1_ONLY = "PASS_NUMERIC_V2_1_ONLY"
    PASS_PARTIAL = "PASS_PARTIAL"
    PASS_FULL_STACK = "PASS_FULL_STACK"

    # Metric honesty
    BASE_TRUE_TPS_WITHHELD = "BASE_TRUE_TPS_WITHHELD"
    METRIC_WITHHELD = "METRIC_WITHHELD"

    # Human gates
    TG3_REVIEW_REQUIRED = "TG3_REVIEW_REQUIRED"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"


# ---------------------------------------------------------------------------
# P0–P13 rungs (bible §8) — shared skeleton; family stages differ
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParityRung:
    id: str
    index: int
    name: str
    description: str
    requires_weights: bool = True
    requires_gpu: bool = False
    requires_fallback_zero: bool = False
    capability_rung: bool = False


PARITY_RUNGS: tuple[ParityRung, ...] = (
    ParityRung("P0", 0, "tokenizer/template", "Tokenizer + chat template admission", requires_weights=False),
    ParityRung("P1", 1, "embedding/norm", "Embedding + RMSNorm"),
    ParityRung("P2", 2, "QKV/RoPE/KV", "QKV projection, RoPE, KV cache write"),
    ParityRung("P3", 3, "attention", "Attention (or DeltaNet schedule for Next)"),
    ParityRung("P4", 4, "router/top-k", "Router + top-k expert selection"),
    ParityRung("P5", 5, "one expert", "Single expert gate/up/act/down"),
    ParityRung("P6", 6, "full MoE", "Full MoE route gather/combine (+ shared expert)"),
    ParityRung("P7", 7, "one layer", "Complete one decoder layer residual"),
    ParityRung("P8", 8, "early/middle/late", "Layer positions across depth"),
    ParityRung(
        "P9",
        9,
        "first token",
        "First-token generation path",
        requires_gpu=True,
        requires_fallback_zero=True,
    ),
    ParityRung(
        "P10",
        10,
        "continuation/full logits",
        "Continuation + full logits parity",
        requires_gpu=True,
        requires_fallback_zero=True,
    ),
    ParityRung(
        "P11",
        11,
        "tool/JSON/edit behavior",
        "Tool calls, JSON, edit behavior",
        requires_gpu=True,
        requires_fallback_zero=True,
        capability_rung=True,
    ),
    ParityRung(
        "P12",
        12,
        "long generation",
        "Long generation stability",
        requires_gpu=True,
        requires_fallback_zero=True,
        capability_rung=True,
    ),
    ParityRung(
        "P13",
        13,
        "restart/reload",
        "Session restart + weight reload",
        requires_gpu=True,
        requires_fallback_zero=True,
        capability_rung=True,
    ),
)

assert len(PARITY_RUNGS) == 14
assert [r.id for r in PARITY_RUNGS] == [f"P{i}" for i in range(14)]


# ---------------------------------------------------------------------------
# Complete-token stages by family (generalizes _COMPLETE_TOKEN_PROFILE_STAGES)
# ---------------------------------------------------------------------------

# Qwen3-MoE / 30B executor (bible §8 build list + profiler §11)
QWEN3_MOE_STAGES: tuple[str, ...] = (
    "tokenizer_template",
    "embedding",
    "norm",
    "qkv",
    "rope",
    "kv_state_read_write",
    "attention",
    "router_top_k",
    "expert_gather",
    "gate_up",
    "activation",
    "down",
    "shared_expert",
    "route_combine",
    "residual",
    "final_norm",
    "lm_head",
    "topk_sampling",
    "endpoint_hcli_streaming",
    "runtime_bookkeeping",
)

# Qwen3-Next / 80B hybrid (bible §9)
QWEN3_NEXT_STAGES: tuple[str, ...] = (
    "tokenizer_template",
    "embedding",
    "norm",
    "qkv_or_deltanet_proj",
    "gated_deltanet_state",
    "deltanet_update",
    "hybrid_schedule_slot",  # 3× DeltaNet / 1× gated attention
    "gated_attention",
    "kv_state_read_write",
    "router_top10",
    "expert_gather",
    "gate_up",
    "activation",
    "down",
    "shared_expert",
    "route_combine",
    "residual",
    "final_norm",
    "lm_head",
    "topk_sampling",
    "endpoint_hcli_streaming",
    "state_memory_accounting",
    "runtime_bookkeeping",
)

# DSV4F reference inventory kept for cross-family comparison only
DEEPSEEK_V4_STAGES_REFERENCE: tuple[str, ...] = (
    "tokenizer_template",
    "embedding",
    "mhc_state_control",
    "norm",
    "qkv",
    "compressed_sparse_attention",
    "index_heads_topk_index",
    "kv_state_read_write",
    "router_top6",
    "expert_gather",
    "gate_up",
    "activation",
    "down",
    "shared_expert",
    "route_combine",
    "residual",
    "lm_head",
    "topk_sampling",
    "endpoint_hcli_streaming",
    "runtime_bookkeeping",
)

FAMILY_STAGES: dict[str, tuple[str, ...]] = {
    ModelFamily.QWEN3_MOE.value: QWEN3_MOE_STAGES,
    ModelFamily.QWEN3_NEXT.value: QWEN3_NEXT_STAGES,
    ModelFamily.DEEPSEEK_V4.value: DEEPSEEK_V4_STAGES_REFERENCE,
}


# ---------------------------------------------------------------------------
# Qwen3-Next state gates (bible §9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateGate:
    id: str
    name: str
    description: str


QWEN3_NEXT_STATE_GATES: tuple[StateGate, ...] = (
    StateGate("SG0", "state_initialization", "Gated DeltaNet state zero/init contract"),
    StateGate("SG1", "chunk_recurrent_equivalence", "Chunk vs recurrent update equivalence"),
    StateGate("SG2", "incremental_decode_parity", "Incremental decode parity vs full recompute"),
    StateGate("SG3", "context_extension", "Context extension preserves state continuity"),
    StateGate("SG4", "restart_reset", "Restart/reset clears or reloads state correctly"),
    StateGate("SG5", "long_generation_stability", "Long generation state stability"),
    StateGate("SG6", "state_memory_accounting", "State + KV memory accounting ledger"),
)

# Qwen3-Next architecture requirements (bible §9)
QWEN3_NEXT_ARCHITECTURE_REQUIREMENTS: tuple[str, ...] = (
    "qwen3_next_parser",
    "gated_deltanet_state",
    "deltanet_update_kernels",
    "hybrid_3_deltanet_1_gated_attention_schedule",
    "gated_attention",
    "routing_512_expert_top10",
    "shared_expert",
    "state_kv_management",
    "hybrid_command_graph",
    "gravity_support",
)


# ---------------------------------------------------------------------------
# Gravity ladder (bible §8) — not a universal 1.5-BPW mandate
# ---------------------------------------------------------------------------


class GravityLadderStage(str, Enum):
    SOURCE_AUTHORITY = "source_authority"
    QUALITY_ANCHOR = "quality_anchor"
    PERFORMANCE_ANCHOR = "performance_anchor"
    GRAVITY_EQUILIBRIUM = "gravity_equilibrium_artifact"


GRAVITY_LADDER_ORDER: tuple[GravityLadderStage, ...] = (
    GravityLadderStage.SOURCE_AUTHORITY,
    GravityLadderStage.QUALITY_ANCHOR,
    GravityLadderStage.PERFORMANCE_ANCHOR,
    GravityLadderStage.GRAVITY_EQUILIBRIUM,
)

# Explicit anti-requirement from bible §8
FORBIDDEN_UNIVERSAL_BPW_REQUIREMENT = 1.5


# ---------------------------------------------------------------------------
# Model ladder pipeline + rotation (bible §30–§31)
# ---------------------------------------------------------------------------

MODEL_LADDER_PIPELINE: tuple[str, ...] = (
    "DISCOVER",
    "PREFLIGHT",
    "RESEARCH_DISTINCTION",
    "DOWNLOAD_STREAM",
    "GRAVITY",
    "LOAD",
    "PARITY",
    "CAPABILITY",
    "PROFILE",
    "OPTIMIZE",
    "REVIEW",
    "REPORT",
    "SEAL",
    "EVICT",
    "ROTATE",
)


class RotationTrigger(str, Enum):
    TG_RUNG_DESCENT = "TG_RUNG_DESCENT"  # A: descended ≥1 named TG rung
    TWO_FAILED_ARCHITECTURES = "TWO_FAILED_ARCHITECTURES"  # B: two fails + roofline + sealed bottleneck + next change
    TG3_HUMAN_REVIEW = "TG3_HUMAN_REVIEW"  # always stop at TG3


# ---------------------------------------------------------------------------
# Default claim boundary (generalizes DSV4F claim_boundary honesty)
# ---------------------------------------------------------------------------


def default_claim_boundary(*, family: str, weights_present: bool = False) -> dict[str, Any]:
    """Honest defaults: nothing is claimed until a sealed receipt flips a field."""
    return {
        "family": family,
        "scaffold_only": True,
        "weights_present": weights_present,
        "source_cpu_parity": False,
        "numeric_parity_v2_1": False,
        "exact_storage_parity": False,
        "first_token_parity": False,
        "full_stack_runtime": False,
        "base_true_tps": False,
        "accelerated_accepted_tps": False,
        "block_executed_tps": False,
        "prefill_tps": False,
        "ttft": False,
        "gpu_dispatches": 0,
        "fallback_count": None,  # unknown until measured; must be 0 to promote
        "hcli_endpoint_exercised": False,
        "capability_protected": False,
        "gravity_equilibrium": False,
        "universal_1_5_bpw_required": False,  # forbidden mandate — always False
        "qwen_download_performed": False,
        "live_model_work": False,
    }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class ParityLadderHarness:
    """Reusable parity ladder parameterized by model family.

    Does not load weights. Callers fill rung bodies once artifacts exist.
    """

    family: ModelFamily
    weights_present: bool = False
    artifact_present: bool = False
    gpu_available: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.family, str):
            self.family = ModelFamily(self.family)

    @property
    def family_key(self) -> str:
        return self.family.value

    def stages(self) -> tuple[str, ...]:
        stages = FAMILY_STAGES.get(self.family_key)
        if stages is None:
            raise KeyError(f"no complete-token stage inventory for family {self.family_key}")
        return stages

    def bootstrap_target(self) -> dict[str, Any]:
        return dict(BOOTSTRAP_TARGETS.get(self.family_key, {}))

    def rungs(self) -> tuple[ParityRung, ...]:
        return PARITY_RUNGS

    def state_gates(self) -> tuple[StateGate, ...]:
        if self.family is ModelFamily.QWEN3_NEXT:
            return QWEN3_NEXT_STATE_GATES
        return ()

    def architecture_requirements(self) -> tuple[str, ...]:
        if self.family is ModelFamily.QWEN3_NEXT:
            return QWEN3_NEXT_ARCHITECTURE_REQUIREMENTS
        if self.family is ModelFamily.QWEN3_MOE:
            return (
                "qwen3_moe_parser",
                "tokenizer_template",
                "embedding_rmsnorm",
                "qkv_rope_kv",
                "attention",
                "router_top8",
                "expert_path",
                "shared_expert",
                "residual_final_norm_lm_head",
                "hcli_streaming",
                "gravity_support",
            )
        return ()

    def evaluate_rung_preconditions(self, rung: ParityRung) -> RungStatus:
        """Honest preflight for a rung before any numeric work.

        Mirrors DSV4F refusal: no fake PASS when weights/GPU/fallback gate fails.
        """
        if not self.weights_present and rung.requires_weights:
            return RungStatus.REJECT_WEIGHTS_ABSENT
        if rung.requires_weights and not self.artifact_present:
            # tokenizer-only rungs may run without a Gravity artifact
            return RungStatus.REJECT_ARTIFACT_ABSENT
        if rung.requires_gpu and not self.gpu_available:
            return RungStatus.REJECT_NO_REAL_GPU_DISPATCH
        # Scaffold path until live bodies are filled
        return RungStatus.SCAFFOLD_PENDING

    def stub_rung_receipt(
        self,
        rung: ParityRung,
        *,
        status: RungStatus | None = None,
        parity: ParityClassification = ParityClassification.SCAFFOLD_PENDING,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = status or self.evaluate_rung_preconditions(rung)
        body: dict[str, Any] = {
            "schema": RUNG_RECEIPT_SCHEMA,
            "status": resolved.value,
            "family": self.family_key,
            "rung": {
                "id": rung.id,
                "index": rung.index,
                "name": rung.name,
                "description": rung.description,
                "requires_weights": rung.requires_weights,
                "requires_gpu": rung.requires_gpu,
                "requires_fallback_zero": rung.requires_fallback_zero,
                "capability_rung": rung.capability_rung,
            },
            "parity_classification": parity.value,
            "parity_is_exact_storage": parity.is_exact_storage(),
            "metal": {
                "fallback": None,
                "fallback_count": None,
                "gpu_dispatches": 0,
                "command_buffers": 0,
                "compute_encoders": 0,
                "note": "scaffold — no live dispatch",
            },
            "metrics": {
                "max_abs_error": None,
                "max_relative_error": None,
                "status": RungStatus.METRIC_WITHHELD.value,
            },
            "claim_boundary": default_claim_boundary(
                family=self.family_key, weights_present=self.weights_present
            ),
            "implementation": {
                "body_filled": False,
                "test_stub": True,
                "fill_when": "Qwen weights streamed + Gravity artifact present",
            },
            "reference_discipline": {
                "numeric_parity_v2_1": "crates/hawking-core/src/gravity_deepseek_v4_p4b_device.rs",
                "complete_token_profile": "lab/operators/deepseek_v4_gravity.py::_COMPLETE_TOKEN_PROFILE_STAGES",
                "pass_full_stack": "lab/operators/frankenstein_teacher_forced_executor.py",
                "tps_withheld": "tools/condense/tests/test_deepseek_v4_child_baseline.py",
            },
        }
        if extra:
            body.update(dict(extra))
        return seal(body)

    def scaffold_ladder_receipt(self) -> dict[str, Any]:
        """Sealed inventory of all 14 rungs for this family (no live execution)."""
        rung_receipts = [self.stub_rung_receipt(r) for r in self.rungs()]
        body: dict[str, Any] = {
            "schema": PARITY_LADDER_RECEIPT_SCHEMA,
            "status": RungStatus.PASS_SCAFFOLD_CONTRACT.value,
            "family": self.family_key,
            "bootstrap_target": self.bootstrap_target(),
            "rung_count": len(rung_receipts),
            "rung_ids": [r.id for r in self.rungs()],
            "rungs": rung_receipts,
            "complete_token_stages": list(self.stages()),
            "architecture_requirements": list(self.architecture_requirements()),
            "state_gates": [
                {"id": g.id, "name": g.name, "description": g.description}
                for g in self.state_gates()
            ],
            "gravity_ladder": [s.value for s in GRAVITY_LADDER_ORDER],
            "gravity_target": "lowest capable, runnable equilibrium",
            "forbidden_universal_bpw": FORBIDDEN_UNIVERSAL_BPW_REQUIREMENT,
            "model_ladder_pipeline": list(MODEL_LADDER_PIPELINE),
            "claim_boundary": default_claim_boundary(
                family=self.family_key, weights_present=self.weights_present
            ),
            "honesty": {
                "live_model_work": False,
                "qwen_download": False,
                "gravity_against_real_weights": False,
                "benchmark_execution": False,
                "reason": "Proto-Frankenstein offload gate; harness scaffold only",
            },
        }
        return seal(body)

    def stub_state_gate_receipt(self, gate: StateGate) -> dict[str, Any]:
        return seal(
            {
                "schema": STATE_GATE_RECEIPT_SCHEMA,
                "status": RungStatus.SCAFFOLD_PENDING.value
                if self.weights_present
                else RungStatus.REJECT_WEIGHTS_ABSENT.value,
                "family": self.family_key,
                "gate": {
                    "id": gate.id,
                    "name": gate.name,
                    "description": gate.description,
                },
                "parity_classification": ParityClassification.SCAFFOLD_PENDING.value,
                "claim_boundary": default_claim_boundary(
                    family=self.family_key, weights_present=self.weights_present
                ),
                "implementation": {"body_filled": False, "test_stub": True},
            }
        )

    def gravity_ladder_receipt(self) -> dict[str, Any]:
        stages = []
        for stage in GRAVITY_LADDER_ORDER:
            stages.append(
                {
                    "stage": stage.value,
                    "status": RungStatus.SCAFFOLD_PENDING.value,
                    "artifact": None,
                    "note": "fill after stream + Gravity co-design",
                }
            )
        return seal(
            {
                "schema": GRAVITY_LADDER_RECEIPT_SCHEMA,
                "status": RungStatus.PASS_SCAFFOLD_CONTRACT.value,
                "family": self.family_key,
                "stages": stages,
                "target": "lowest capable, runnable equilibrium",
                "universal_1_5_bpw_forbidden": True,
                "claim_boundary": default_claim_boundary(
                    family=self.family_key, weights_present=self.weights_present
                ),
            }
        )

    def model_ladder_pipeline_receipt(self) -> dict[str, Any]:
        return seal(
            {
                "schema": MODEL_LADDER_PIPELINE_SCHEMA,
                "status": RungStatus.PASS_SCAFFOLD_CONTRACT.value,
                "family": self.family_key,
                "pipeline": list(MODEL_LADDER_PIPELINE),
                "current_phase": "PREFLIGHT",
                "rotation_triggers": [t.value for t in RotationTrigger],
                "rotation_rule": {
                    "A": "model descends at least one named TG rung",
                    "B": (
                        "two materially different optimization architectures fail, "
                        "same-model roofline measured, repeated bottleneck sealed, "
                        "smallest next representation change named"
                    ),
                    "TG3": "always stop for human/controller review",
                },
                "selection_questions": [
                    "What new physical or architectural distinction does this test?",
                    "Which existing kernel grammar should transfer?",
                    "Which new operator or state contract is required?",
                    "What evidence would make the model redundant?",
                ],
                "claim_boundary": default_claim_boundary(
                    family=self.family_key, weights_present=self.weights_present
                ),
            }
        )


def promote_rung_status(
    *,
    parity_pass: bool,
    fallback_count: int | None,
    gpu_dispatches: int,
    full_stack: bool,
    capability_pass: bool = True,
) -> RungStatus:
    """Promotion decision vocabulary (DSV4F / broker_kernel_ab pattern).

    - fallback != 0 → REJECT_FALLBACK_NONZERO
    - gpu_dispatches == 0 on GPU path → REJECT_NO_REAL_GPU_DISPATCH
    - parity fail → REJECT_PARITY
    - full stack only when all gates pass
    """
    if fallback_count is None:
        return RungStatus.SCAFFOLD_PENDING
    if fallback_count != 0:
        return RungStatus.REJECT_FALLBACK_NONZERO
    if gpu_dispatches <= 0:
        return RungStatus.REJECT_NO_REAL_GPU_DISPATCH
    if not parity_pass:
        return RungStatus.REJECT_PARITY
    if not capability_pass:
        return RungStatus.REJECT_CAPABILITY
    if full_stack:
        return RungStatus.PASS_FULL_STACK
    return RungStatus.PASS_NUMERIC_V2_1_ONLY


def verify_ladder_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return verify(receipt, label="ascension parity ladder receipt")


def all_family_scaffold_receipts() -> dict[str, dict[str, Any]]:
    """Inventory receipts for bootstrap families (30B MoE + 80B Next)."""
    out: dict[str, dict[str, Any]] = {}
    for family in (ModelFamily.QWEN3_MOE, ModelFamily.QWEN3_NEXT):
        harness = ParityLadderHarness(family=family)
        out[family.value] = harness.scaffold_ladder_receipt()
    return out


__all__ = [
    "BOOTSTRAP_TARGETS",
    "DEEPSEEK_V4_STAGES_REFERENCE",
    "FAMILY_STAGES",
    "FORBIDDEN_UNIVERSAL_BPW_REQUIREMENT",
    "GRAVITY_LADDER_ORDER",
    "GRAVITY_LADDER_RECEIPT_SCHEMA",
    "GravityLadderStage",
    "MODEL_LADDER_PIPELINE",
    "MODEL_LADDER_PIPELINE_SCHEMA",
    "ModelFamily",
    "PARITY_LADDER_RECEIPT_SCHEMA",
    "PARITY_RUNGS",
    "ParityClassification",
    "ParityLadderHarness",
    "ParityRung",
    "QWEN3_MOE_STAGES",
    "QWEN3_NEXT_ARCHITECTURE_REQUIREMENTS",
    "QWEN3_NEXT_STAGES",
    "QWEN3_NEXT_STATE_GATES",
    "RUNG_RECEIPT_SCHEMA",
    "RotationTrigger",
    "RungStatus",
    "STATE_GATE_RECEIPT_SCHEMA",
    "StateGate",
    "all_family_scaffold_receipts",
    "default_claim_boundary",
    "promote_rung_status",
    "verify_ladder_receipt",
]
