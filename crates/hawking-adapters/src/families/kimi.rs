//! Kimi family.
//!
//! Reference / synthetic only, not serve-registered.
//! KIMI_K26_ADAPTER_TWIN.json = SYNTHETIC_CPU_REFERENCE_AND_BOUND_REAL_SOURCE_METAL_K1.

use crate::abi::{
    AbiField, AbiListField, ContextLimits, Evidence, EvidenceKind, FamilyAbi, FamilyAdapter,
    FamilyDescriptor, ProviderAvailability,
};
use crate::support_level::SupportLevel;

pub struct KimiFamily;

const ALIASES: &[&str] = &["kimi", "kimi_k2", "kimi_k26", "moonshot"];

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "KIMI_K26_ADAPTER_TWIN.json",
        claim: "synthetic CPU reference + bound real-source metal K1 twin",
        kind: EvidenceKind::SyntheticParity,
    },
    Evidence {
        path: "crates/hawking-core/src/model/mod.rs",
        claim: "load_engine has no kimi arch arm (not serve-registered)",
        kind: EvidenceKind::Description,
    },
];

const GAPS: &[&str] = &[
    "not serve-registered in load_engine",
    "no SMALL_REAL_CHECKPOINT sealed receipt for full generate path",
    "not PRODUCTION",
];

const SOURCE_CLASSES: &[&str] = &[
    "hf.config (Kimi-K2.x text core DeepseekV3-like)",
    "adapter_twin.kimi_k26",
];
const CODECS: &[&str] = &["safetensors (source)", "synthetic twin"];
const PRECISIONS: &[&str] = &["INT4 group-32 (routed experts)", "BF16", "F32"];
const LIMITS: &[&str] = &[
    "no load_engine arch arm",
    "tokenizer is tiktoken (not HF tokenizers JSON)",
    "multimodal vision tower out of scope for this family entry",
];

const ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::some(
        "DeepseekV3-like MLA MoE names on text core; twin maps a reduced synthetic namespace",
    ),
    tokenizer: AbiField::some("tiktoken (tiktoken.model + tokenization_kimi.py) — not HF tokenizers JSON"),
    chat_template: AbiField::null(
        "chat template bound to source release; not sealed in adapter twin as serve template",
    ),
    attention_or_state: AbiField::some("MLA (DeepseekV3-like text core)"),
    topology: AbiField::some("MoE 384 routed / top-8 / 1 shared (source claim; twin is reduced)"),
    normalization: AbiField::some("RMSNorm (source family)"),
    positional_encoding: AbiField::some("RoPE theta 50000 (source text_config)"),
    kv_or_state_format: AbiField::some("MLA latent KV (source); twin uses synthetic state"),
    context_limits: ContextLimits::unknown(
        "source max_position_embeddings not sealed as serve-validated limit",
    ),
    artifact_codecs: AbiListField::some(CODECS),
    providers: ProviderAvailability {
        cpu: AbiField::some("synthetic CPU reference only"),
        metal: AbiField::some("partial — bound real-source metal K1 twin, not full serve"),
        cuda: AbiField::null("no CUDA path declared"),
    },
    fabric_partition_boundaries: AbiField::null("not serve-registered; no Fabric placement"),
    capability_limitations: LIMITS,
    source_precision_classes: AbiListField::some(PRECISIONS),
};

impl FamilyAdapter for KimiFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        FamilyDescriptor {
            id: "kimi",
            aliases: ALIASES,
            display_name: "Kimi K2.x",
            level: SupportLevel::SyntheticParity,
            evidence: EVIDENCE,
            module: "KIMI_K26_ADAPTER_TWIN.json (reference twin; no in-tree serve module)",
            executes: false,
            serve_registered: false,
            gaps: GAPS,
            abi: ABI,
        }
    }
}
