//! GLM family (flagship via .gravity / glm_moe_dsa).
//!
//! SMALL_REAL_CHECKPOINT with M04_SEALED receipt in GLM52_FLAGSHIP_ADAPTER_PARITY.json.
//! Not PRODUCTION.

use crate::abi::{
    AbiField, AbiListField, ContextLimits, Evidence, EvidenceKind, FamilyAbi, FamilyAdapter,
    FamilyDescriptor, ProviderAvailability,
};
use crate::support_level::SupportLevel;

pub struct GlmFamily;

const ALIASES: &[&str] = &["glm", "glm52", "glm_moe_dsa", "glm-5.2"];

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "GLM52_FLAGSHIP_ADAPTER_PARITY.json",
        claim: "M04_SEALED: Rust adapter vs oracle on real flagship .gravity shards",
        kind: EvidenceKind::SmallCheckpointRun,
    },
    Evidence {
        path: "crates/hawking-core/src/model/gravity_engine.rs",
        claim: "GravityEngine dispatches glm_moe_dsa",
        kind: EvidenceKind::Description,
    },
    Evidence {
        path: "crates/hawking-core/tests/gravity_engine_registry.rs",
        claim: "registry path for .gravity artifacts",
        kind: EvidenceKind::Description,
    },
    // Stage C: this test runs UNCONDITIONALLY. Unlike the weight-gated tests, it does
    // not skip -- the fixture is committed. Real container, real codec, real oracle.
    Evidence {
        path: "crates/hawking-core/tests/gravity_glm_forward.rs",
        claim: "unconditional: real .gravity container decoded, complete token executed, matches numpy oracle reading the same container (3 passed, 0.04s)",
        kind: EvidenceKind::RealTensorDecode,
    },
];

const GAPS: &[&str] = &[
    "not PRODUCTION",
    "gravity_glm.rs is another lane's sealed path — not claimed as open production serve",
    "full parent source safetensors not the parity authority (gravity bytes are)",
];

const SOURCE_CLASSES: &[&str] = &[
    "gravity.glm_moe_dsa",
    "hf.config (GLM-5.2 — campaign ladder, not serve authority)",
];
const CODECS: &[&str] = &["gravity"];
const PRECISIONS: &[&str] = &["gravity-packed", "F8_E4M3 (source parent)", "BF16 (source parent)"];
const LIMITS: &[&str] = &[
    "execution is .gravity only for glm_moe_dsa — not raw HF safetensors serve",
    "MLA+DSA indexer shapes are gravity-contract specific",
    "no PRODUCTION standing receipt",
];

const ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::some(
        "gravity glm_moe_dsa shard layout (not GGUF llama names); MLA/DSA packed tensors \
         per gravity functional codec",
    ),
    tokenizer: AbiField::null(
        "tokenizer bound at serve/campaign layer for GLM flagship; not re-declared in gravity engine",
    ),
    chat_template: AbiField::null(
        "chat_template.jinja lives with source release; not owned by gravity_engine",
    ),
    attention_or_state: AbiField::some("MLA + DSA lightning indexer (glm_moe_dsa)"),
    topology: AbiField::some("MoE (routed experts + shared) with dense early layers — gravity contract"),
    normalization: AbiField::some("RMSNorm (gravity contract)"),
    positional_encoding: AbiField::some("RoPE on MLA rope dims (gravity contract)"),
    kv_or_state_format: AbiField::some("compact MLA cache / gravity-resident state layouts"),
    context_limits: ContextLimits::known(1_048_576, "declared_from_source_config_unvalidated_at_serve"),
    artifact_codecs: AbiListField::some(CODECS),
    providers: ProviderAvailability {
        cpu: AbiField::some("partial — CPU oracle/parity paths exist for gravity shards"),
        metal: AbiField::some("partial — Metal gravity kernels used in sealed parity lanes"),
        cuda: AbiField::null("no in-tree CUDA provider for glm_moe_dsa"),
    },
    fabric_partition_boundaries: AbiField::some(
        "gravity shard boundaries are the partition unit; Fabric multi-node placement not declared",
    ),
    capability_limitations: LIMITS,
    source_precision_classes: AbiListField::some(PRECISIONS),
};

impl FamilyAdapter for GlmFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        FamilyDescriptor {
            id: "glm",
            aliases: ALIASES,
            display_name: "GLM (gravity glm_moe_dsa)",
            level: SupportLevel::SmallRealCheckpoint,
            evidence: EVIDENCE,
            module: "crates/hawking-core/src/model/gravity_engine.rs",
            executes: true,
            serve_registered: true,
            gaps: GAPS,
            abi: ABI,
        }
    }
}
