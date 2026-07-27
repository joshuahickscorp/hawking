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
        path: "crates/hawking-core/tests/llama32_smoke.rs",
        claim: "dense llama-family small checkpoint path (mistral shares LlamaDense)",
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
];

const SOURCE_CLASSES: &[&str] = &[
    "gguf.mistral",
    "gguf.general.architecture=mistral",
    "gguf.mixtral (pack only)",
    "seed-c.ArchAdapter.mixtral",
];
const CODECS: &[&str] = &["gguf"];
const PRECISIONS: &[&str] = &["F16", "Q4_K", "Q5_K", "Q6_K", "Q8_0"];
const LIMITS: &[&str] = &[
    "mixtral MoE requires pack hydrate — not shipping",
    "dense mistral only is serve-registered",
];

const ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::some(
        "dense: same llama-family GGUF names; mixtral (pack): blk.{l}.ffn_gate_inp + \
         per-expert ffn_down.{e} (split experts, not fused *_exps)",
    ),
    tokenizer: AbiField::some("gguf.tokenizer"),
    chat_template: AbiField::null("chat template not owned by family engine"),
    attention_or_state: AbiField::some("GQA causal attention (dense mistral); mixtral same attention + MoE FFN"),
    topology: AbiField::some(
        "dense mistral (shipping); mixtral MoE 8 experts top-2 (pack only, declarative seed-c)",
    ),
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
            level: SupportLevel::SmallRealCheckpoint,
            evidence: EVIDENCE,
            module: "crates/hawking-core/src/model/llama.rs (+ pack mixtral)",
            executes: true,
            serve_registered: true,
            gaps: GAPS,
            abi: ABI,
        }
    }
}
