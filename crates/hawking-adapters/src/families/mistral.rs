//! Mistral / Mixtral family.
//!
//! Dense Mistral GGUF shares the llama loader (`load_engine` maps `mistral` →
//! LlamaDense). Mixtral MoE was extracted to the hawking-adapters-extra pack.

use crate::abi::{
    AbiField, AbiListField, ContextLimits, Evidence, EvidenceKind, FamilyAbi, FamilyAdapter,
    FamilyDescriptor, ProviderAvailability,
};
use crate::support_level::SupportLevel;

pub struct MistralFamily;

const ALIASES: &[&str] = &["mistral", "mixtral", "mistral_mixtral"];

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "crates/hawking-core/src/model/mod.rs",
        claim: "dense mistral arch string routes to LlamaDense",
        kind: EvidenceKind::Description,
    },
    Evidence {
        path: "adapters/receipts/ADAPTER_MISTRAL_MIXTRAL_RECEIPT.json",
        claim: "Stage A: official config/tokenizer/safetensors header parsed and mapped",
        kind: EvidenceKind::SourceHeader,
    },
    Evidence {
        path: "crates/hawking-core/tests/llama32_smoke.rs",
        claim: "dense llama-family small checkpoint path (mistral shares LlamaDense; skips without GGUF)",
        kind: EvidenceKind::SmallCheckpointRun,
    },
    Evidence {
        path: "packs/hawking-adapters-extra.json",
        claim: "mixtral extracted off-tree to adapters-extra pack",
        kind: EvidenceKind::Description,
    },
    Evidence {
        path: "crates/hawking-seed-c/src/providers/adapters.rs",
        claim: "seed-c ArchAdapter::mixtral is declarative plan-only (does not execute)",
        kind: EvidenceKind::Description,
    },
];

const GAPS: &[&str] = &[
    "mixtral MoE not in shipping load_engine",
    "seed-c ArchAdapter does not execute",
    "no PRODUCTION receipt",
    "SMALL_REAL_CHECKPOINT smoke skips when no GGUF is on disk",
];

const SOURCE_CLASSES: &[&str] = &[
    "gguf.mistral",
    "gguf.general.architecture=mistral",
    "gguf.mixtral (pack only)",
    "seed-c.ArchAdapter.mixtral",
];
const CODECS: &[&str] = &["gguf", "pack (mixtral extra)"];
const PRECISIONS: &[&str] = &["F16", "BF16", "Q4_K", "Q5_K", "Q8_0"];
const LIMITS: &[&str] = &[
    "mixtral requires adapters-extra hydrate",
    "dense path only for serve-registered execution",
    "no PRODUCTION standing receipt",
];

const ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::some(
        "dense mistral: llama-family GGUF names via LlamaDense; mixtral: pack-defined MoE expert tensors",
    ),
    tokenizer: AbiField::some("gguf.tokenizer (dense); pack tokenizer for mixtral"),
    chat_template: AbiField::null("chat template at serve/prompt layer"),
    attention_or_state: AbiField::some("GQA causal attention (dense); mixtral MoE experts (pack)"),
    topology: AbiField::some("dense transformer (shipping); Mixtral MoE (pack only)"),
    normalization: AbiField::some("RMSNorm"),
    positional_encoding: AbiField::some("RoPE"),
    kv_or_state_format: AbiField::some("per-layer K/V cache (dense path)"),
    context_limits: ContextLimits::unknown(
        "no sealed family-level context limit receipt for mistral/mixtral",
    ),
    artifact_codecs: AbiListField::some(CODECS),
    providers: ProviderAvailability {
        cpu: AbiField::some("available for dense mistral via LlamaDense"),
        metal: AbiField::some("partial — shared llama metal path"),
        cuda: AbiField::null("no in-tree CUDA provider"),
    },
    fabric_partition_boundaries: AbiField::null(
        "no Fabric partition declaration for mistral/mixtral",
    ),
    capability_limitations: LIMITS,
    source_precision_classes: AbiListField::some(PRECISIONS),
};

impl FamilyAdapter for MistralFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        FamilyDescriptor {
            id: "mistral_mixtral",
            aliases: ALIASES,
            display_name: "Mistral / Mixtral",
            // Demoted: only live evidence is Stage A source-header receipt.
            // llama32_smoke skips without GGUF; mixtral is pack/extracted.
            level: SupportLevel::SourceHeaderValidated,
            evidence: EVIDENCE,
            module: "crates/hawking-core/src/model/llama.rs (+ pack mixtral)",
            executes: true,
            serve_registered: true,
            gaps: GAPS,
            abi: ABI,
        }
    }
}
