//! Phi family — extracted to hawking-adapters-extra pack.

use crate::abi::{
    AbiField, AbiListField, ContextLimits, Evidence, EvidenceKind, FamilyAbi, FamilyAdapter,
    FamilyDescriptor, ProviderAvailability,
};
use crate::support_level::SupportLevel;

pub struct PhiFamily;

const ALIASES: &[&str] = &["phi", "phi3"];

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "packs/hawking-adapters-extra.json",
        claim: "phi3 extracted off-tree",
        kind: EvidenceKind::Description,
    },
    Evidence {
        path: "crates/hawking-core/tests/phi3_smoke.rs",
        claim: "smoke test remains; arch not in shipping load_engine",
        kind: EvidenceKind::Description,
    },
    Evidence {
        path: "crates/hawking-seed-c/src/providers/adapters.rs",
        claim: "seed-c ArchAdapter::phi3 is declarative plan-only",
        kind: EvidenceKind::Description,
    },
];

const GAPS: &[&str] = &[
    "module not in shipping load_engine",
    "pack hydrate required to execute",
    "not PRODUCTION",
];

const SOURCE_CLASSES: &[&str] = &[
    "gguf.phi3",
    "seed-c.ArchAdapter.phi3",
    "pack.hawking-adapters-extra.phi3",
];
const CODECS: &[&str] = &["gguf (pack only)"];
const PRECISIONS: &[&str] = &["F16", "Q4_K", "Q8_0"];
const LIMITS: &[&str] = &[
    "not in shipping load_engine",
    "fused qkv and gate_up exceptions",
];

const ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::some(
        "GGUF llama-family names with fused qkv + fused gate_up exceptions (split at plan time)",
    ),
    tokenizer: AbiField::some("gguf.tokenizer (declared in seed-c ArchAdapter)"),
    chat_template: AbiField::null("not owned by ArchAdapter plan path"),
    attention_or_state: AbiField::some("GQA/MHA causal attention (declared)"),
    topology: AbiField::some("dense transformer"),
    normalization: AbiField::some("RMSNorm"),
    positional_encoding: AbiField::some("RoPE"),
    kv_or_state_format: AbiField::some("standard K/V cache (declared; not shipping execute)"),
    context_limits: ContextLimits::unknown("pack not hydrated in shipping tree"),
    artifact_codecs: AbiListField::some(CODECS),
    providers: ProviderAvailability {
        cpu: AbiField::null("shipping load_engine rejects phi3 without pack"),
        metal: AbiField::null("not shipping"),
        cuda: AbiField::null("not shipping"),
    },
    fabric_partition_boundaries: AbiField::null("not serve-registered"),
    capability_limitations: LIMITS,
    source_precision_classes: AbiListField::some(PRECISIONS),
};

impl FamilyAdapter for PhiFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        FamilyDescriptor {
            id: "phi",
            aliases: ALIASES,
            display_name: "Phi-3",
            level: SupportLevel::Declared,
            evidence: EVIDENCE,
            module: "packs/hawking-adapters-extra (phi3)",
            executes: false,
            serve_registered: false,
            gaps: GAPS,
            abi: ABI,
        }
    }
}
