//! Qwen dense / MoE family.
//!
//! Stage A source-header validated. Higher grades (small checkpoint / full parent)
//! require on-disk parents; the cited integration tests skip without them.

use crate::abi::{
    AbiField, AbiListField, ContextLimits, Evidence, EvidenceKind, FamilyAbi, FamilyAdapter,
    FamilyDescriptor, ProviderAvailability,
};
use crate::support_level::SupportLevel;

pub struct QwenFamily;

const ALIASES: &[&str] = &["qwen", "qwen2", "qwen2moe", "qwen-moe", "qwen3"];

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "crates/hawking-core/src/model/qwen_dense.rs",
        claim: "in-tree QwenDense engine",
        kind: EvidenceKind::Description,
    },
    Evidence {
        path: "crates/hawking-core/src/model/qwen_moe.rs",
        claim: "in-tree QwenMoE engine",
        kind: EvidenceKind::Description,
    },
    Evidence {
        path: "adapters/receipts/ADAPTER_QWEN_RECEIPT.json",
        claim: "Stage A: official config/tokenizer/safetensors header parsed and mapped",
        kind: EvidenceKind::SourceHeader,
    },
    Evidence {
        path: "crates/hawking-core/tests/integration_greedy_64.rs",
        claim: "greedy integration path (skips when no model on disk — not live grade evidence)",
        kind: EvidenceKind::FullParentValidation,
    },
    Evidence {
        path: "crates/hawking-core/tests/cpu_backend_parity.rs",
        claim: "CPU backend parity (skips when no qwen0.5b weights)",
        kind: EvidenceKind::FullParentValidation,
    },
    Evidence {
        path: "crates/hawking-core/tests/qwen_tq_serve_parity.rs",
        claim: "TQ serve parity (#[ignore] + skips without weights/sidecar)",
        kind: EvidenceKind::SmallCheckpointRun,
    },
    Evidence {
        path: "crates/hawking-core/src/model/mod.rs",
        claim: "load_engine dispatches qwen2/qwen2moe",
        kind: EvidenceKind::Description,
    },
];

const GAPS: &[&str] = &[
    "not PRODUCTION: no standing production parity receipt under continuous serve",
    "large MoE parents (235B/397B) are campaign-side, not this registry's PRODUCTION claim",
    "integration/parity tests skip or are #[ignore] without on-disk parents",
];

const SOURCE_CLASSES: &[&str] = &[
    "gguf.qwen2",
    "gguf.qwen2moe",
    "gguf.general.architecture=qwen2|qwen2moe",
];
const CODECS: &[&str] = &["gguf", "tq (feature-gated)"];
const PRECISIONS: &[&str] = &["F16", "BF16", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "TQ"];
const LIMITS: &[&str] = &[
    "no PRODUCTION standing receipt",
    "235B/397B MoE parents not claimed FULL_PARENT at this registry grade for those sizes",
];

const ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::some(
        "GGUF qwen2 dense + qwen2moe: llama-family-like blk.{l} names; MoE uses \
         ffn_gate_inp router and expert tensors per GGUF qwen2moe layout",
    ),
    tokenizer: AbiField::some("gguf.tokenizer (Qwen vocab embedded)"),
    chat_template: AbiField::null(
        "chat template applied at serve/prompt layer; not declared inside engine modules",
    ),
    attention_or_state: AbiField::some("GQA causal attention"),
    topology: AbiField::some("dense (QwenDense) and MoE (QwenMoE) both in-tree"),
    normalization: AbiField::some("RMSNorm"),
    positional_encoding: AbiField::some("RoPE (qwen rope_theta from metadata)"),
    kv_or_state_format: AbiField::some("per-layer K/V cache"),
    context_limits: ContextLimits::unknown(
        "context from GGUF/EngineConfig; large-parent long-context not sealed as family limit",
    ),
    artifact_codecs: AbiListField::some(CODECS),
    providers: ProviderAvailability {
        cpu: AbiField::some("available — CPU backend parity tests"),
        metal: AbiField::some("partial — Metal path used in serve when configured"),
        cuda: AbiField::null("no in-tree CUDA provider"),
    },
    fabric_partition_boundaries: AbiField::null(
        "no Fabric multi-node partition declaration for qwen family",
    ),
    capability_limitations: LIMITS,
    source_precision_classes: AbiListField::some(PRECISIONS),
};

impl FamilyAdapter for QwenFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        FamilyDescriptor {
            id: "qwen",
            aliases: ALIASES,
            display_name: "Qwen (dense + MoE)",
            // Demoted from FullParentValidated then SmallRealCheckpoint: cited tests skip
            // or are #[ignore] without on-disk parents. Stage A source-header receipt is live.
            level: SupportLevel::SourceHeaderValidated,
            evidence: EVIDENCE,
            module: "crates/hawking-core/src/model/qwen_dense.rs",
            executes: true,
            serve_registered: true,
            gaps: GAPS,
            abi: ABI,
        }
    }
}
