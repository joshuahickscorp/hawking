//! DeepSeek V2 family (GGUF deepseek2).

use crate::abi::{
    AbiField, AbiListField, ContextLimits, Evidence, EvidenceKind, FamilyAbi, FamilyAdapter,
    FamilyDescriptor, ProviderAvailability,
};
use crate::support_level::SupportLevel;

pub struct DeepSeekFamily;

const ALIASES: &[&str] = &["deepseek", "deepseek2", "deepseek_v2"];

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "crates/hawking-core/src/model/deepseek_v2.rs",
        claim: "in-tree DeepSeekV2 engine",
        kind: EvidenceKind::Description,
    },
    Evidence {
        path: "crates/hawking-core/tests/cpu_backend_parity_deepseek.rs",
        claim: "CPU backend parity for deepseek path",
        kind: EvidenceKind::SmallCheckpointRun,
    },
    Evidence {
        path: "crates/hawking-core/src/model/mod.rs",
        claim: "load_engine dispatches deepseek2",
        kind: EvidenceKind::Description,
    },
    // The grade names the evidence, so the grade must cite it. Stage A of the family
    // gauntlet parsed DeepSeek's official config, tokenizer and safetensors header; that
    // is what SOURCE_HEADER_VALIDATED means, and it is the only evidence here that is
    // unconditionally true. The cpu_backend_parity test above skips when no weights are
    // on disk, and none are.
    Evidence {
        path: "adapters/receipts/ADAPTER_DEEPSEEK_RECEIPT.json",
        claim: "Stage A: official config/tokenizer/safetensors header parsed and mapped",
        kind: EvidenceKind::SourceHeader,
    },
];

const GAPS: &[&str] = &[
    "not FULL_PARENT_VALIDATED: no sealed full-size parent receipt in registry evidence",
    "not PRODUCTION",
    "DeepSeek V3/V4 MLA+DSA ladder rungs are NOT this family's shipping GGUF deepseek2 path",
];

const SOURCE_CLASSES: &[&str] = &["gguf.deepseek2", "gguf.general.architecture=deepseek2"];
const CODECS: &[&str] = &["gguf"];
const PRECISIONS: &[&str] = &["F16", "Q4_K", "Q5_K", "Q6_K", "Q8_0"];
const LIMITS: &[&str] = &[
    "shipping path is GGUF deepseek2 — not V3/V4 safetensors ladder",
    "no PRODUCTION receipt",
];

const ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::some(
        "GGUF deepseek2 tensor names as consumed by DeepSeekV2 engine",
    ),
    tokenizer: AbiField::some("gguf.tokenizer"),
    chat_template: AbiField::null("chat template not owned by DeepSeekV2 engine module"),
    attention_or_state: AbiField::some("MLA-style / multi-latent attention as implemented in deepseek_v2"),
    topology: AbiField::some("MoE (DeepSeek V2 GGUF layout) as implemented in-tree"),
    normalization: AbiField::some("RMSNorm"),
    positional_encoding: AbiField::some("RoPE (MLA rope dims)"),
    kv_or_state_format: AbiField::some("MLA compressed KV / latent cache per engine"),
    context_limits: ContextLimits::unknown(
        "context from GGUF metadata; no sealed family-level long-context receipt",
    ),
    artifact_codecs: AbiListField::some(CODECS),
    providers: ProviderAvailability {
        cpu: AbiField::some("available — cpu_backend_parity_deepseek"),
        metal: AbiField::some("partial — shared metal path when configured"),
        cuda: AbiField::null("no in-tree CUDA provider"),
    },
    fabric_partition_boundaries: AbiField::null(
        "no Fabric partition declaration for deepseek2",
    ),
    capability_limitations: LIMITS,
    source_precision_classes: AbiListField::some(PRECISIONS),
};

impl FamilyAdapter for DeepSeekFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        FamilyDescriptor {
            id: "deepseek",
            aliases: ALIASES,
            display_name: "DeepSeek V2",
            // Demoted: cpu_backend_parity_deepseek.rs skips when weights are absent.
            level: SupportLevel::SourceHeaderValidated,
            evidence: EVIDENCE,
            module: "crates/hawking-core/src/model/deepseek_v2.rs",
            executes: true,
            serve_registered: true,
            gaps: GAPS,
            abi: ABI,
        }
    }
}
