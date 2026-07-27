//! Gemma family — extracted to hawking-adapters-extra pack.

use crate::abi::{
    AbiField, AbiListField, ContextLimits, Evidence, EvidenceKind, FamilyAbi, FamilyAdapter,
    FamilyDescriptor, ProviderAvailability,
};
use crate::support_level::SupportLevel;

pub struct GemmaFamily;

const ALIASES: &[&str] = &["gemma", "gemma2"];

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "packs/hawking-adapters-extra.json",
        claim: "gemma2 extracted off-tree",
        kind: EvidenceKind::Description,
    },
    Evidence {
        path: "crates/hawking-core/tests/gemma2_smoke.rs",
        claim: "smoke test remains but load_engine rejects unknown gemma2 arch without pack",
        kind: EvidenceKind::Description,
    },
    Evidence {
        path: "crates/hawking-seed-c/src/providers/adapters.rs",
        claim: "seed-c ArchAdapter::gemma2 is declarative plan-only",
        kind: EvidenceKind::Description,
    },
];

const GAPS: &[&str] = &[
    "module not in shipping load_engine",
    "pack hydrate required to execute",
    "not PRODUCTION",
];

const SOURCE_CLASSES: &[&str] = &[
    "gguf.gemma2",
    "seed-c.ArchAdapter.gemma2",
    "pack.hawking-adapters-extra.gemma2",
];
const CODECS: &[&str] = &["gguf (pack only)"];
const PRECISIONS: &[&str] = &["F16", "Q4_K", "Q8_0"];
const LIMITS: &[&str] = &[
    "not in shipping load_engine",
    "ArchAdapter is plan-summary only",
    "logit softcap + pre/post FFN norm exceptions",
];

const ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::some(
        "GGUF llama-family tensor names; exceptions: attn_logit_softcap, final_logit_softcap, pre+post ffn norm",
    ),
    tokenizer: AbiField::some("gguf.tokenizer (declared in seed-c ArchAdapter)"),
    chat_template: AbiField::null("not owned by ArchAdapter plan path"),
    attention_or_state: AbiField::some("GQA with logit soft-capping (declared)"),
    topology: AbiField::some("dense transformer"),
    normalization: AbiField::some("RMSNorm with pre+post FFN norm exception"),
    positional_encoding: AbiField::some("RoPE"),
    kv_or_state_format: AbiField::some("standard K/V cache (declared; not shipping execute)"),
    context_limits: ContextLimits::unknown("pack not hydrated in shipping tree"),
    artifact_codecs: AbiListField::some(CODECS),
    providers: ProviderAvailability {
        cpu: AbiField::null("shipping load_engine rejects gemma2 without pack"),
        metal: AbiField::null("not shipping"),
        cuda: AbiField::null("not shipping"),
    },
    fabric_partition_boundaries: AbiField::null("not serve-registered"),
    capability_limitations: LIMITS,
    source_precision_classes: AbiListField::some(PRECISIONS),
};

impl FamilyAdapter for GemmaFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        FamilyDescriptor {
            id: "gemma",
            aliases: ALIASES,
            display_name: "Gemma 2",
            level: SupportLevel::Declared,
            evidence: EVIDENCE,
            module: "packs/hawking-adapters-extra (gemma2)",
            executes: false,
            serve_registered: false,
            gaps: GAPS,
            abi: ABI,
        }
    }
}
