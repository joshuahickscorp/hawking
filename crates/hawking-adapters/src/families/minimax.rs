//! MiniMax family — declared only; no in-tree execution path.

use crate::abi::{
    AbiField, AbiListField, ContextLimits, Evidence, EvidenceKind, FamilyAbi, FamilyAdapter,
    FamilyDescriptor, ProviderAvailability,
};
use crate::support_level::SupportLevel;

pub struct MiniMaxFamily;

const ALIASES: &[&str] = &["minimax", "minimax_m3", "minimax-m3"];

const EVIDENCE: &[Evidence] = &[Evidence {
    path: "FABRIC_BRIDGE_ARCHAEOLOGY.md",
    claim: "family listed in bridge archaeology; no serve path found",
    kind: EvidenceKind::Description,
}];

const GAPS: &[&str] = &[
    "no in-tree engine module",
    "not serve-registered",
    "not PRODUCTION",
];

const SOURCE_CLASSES: &[&str] = &["hf.config (MiniMax-M3 — ladder F3, declared only)"];
const LIMITS: &[&str] = &[
    "block-sparse top-k attention not in shipping ArchAdapter table",
    "no execution path",
];

const ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::null(
        "no in-tree tensor map; source uses MiniMax sparse attention + MoE HF names",
    ),
    tokenizer: AbiField::null("HF tokenizers JSON on source release; not wired"),
    chat_template: AbiField::null("not wired"),
    attention_or_state: AbiField::some(
        "declared: block-sparse top-k attention (MiniMax Sparse Attention) + GQA",
    ),
    topology: AbiField::some("declared: MoE 128 local experts top-4 + shared; early dense layers"),
    normalization: AbiField::null("not mapped in-tree"),
    positional_encoding: AbiField::some("declared: partial RoPE (partial_rotary_factor 0.5) on source"),
    kv_or_state_format: AbiField::null("not implemented"),
    context_limits: ContextLimits::unknown("no in-tree validation; source config not loaded here"),
    artifact_codecs: AbiListField::null("no codec path in shipping tree"),
    providers: ProviderAvailability {
        cpu: AbiField::null("no engine"),
        metal: AbiField::null("no engine"),
        cuda: AbiField::null("no engine"),
    },
    fabric_partition_boundaries: AbiField::null("not serve-registered"),
    capability_limitations: LIMITS,
    source_precision_classes: AbiListField::some(&["BF16", "F32"]),
};

impl FamilyAdapter for MiniMaxFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        FamilyDescriptor {
            id: "minimax",
            aliases: ALIASES,
            display_name: "MiniMax",
            level: SupportLevel::Declared,
            evidence: EVIDENCE,
            module: "(none — declared only)",
            executes: false,
            serve_registered: false,
            gaps: GAPS,
            abi: ABI,
        }
    }
}
