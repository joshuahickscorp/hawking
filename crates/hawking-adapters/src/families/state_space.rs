//! State-space family (RWKV7 in-tree; Mamba2 extracted).

use crate::abi::{
    AbiField, AbiListField, ContextLimits, Evidence, EvidenceKind, FamilyAbi, FamilyAdapter,
    FamilyDescriptor, ProviderAvailability,
};
use crate::support_level::SupportLevel;

pub struct StateSpaceFamily;

const ALIASES: &[&str] = &["state_space", "rwkv7", "rwkv", "mamba2", "mamba"];

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "crates/hawking-core/src/model/rwkv7.rs",
        claim: "in-tree RwkvSeven engine",
        kind: EvidenceKind::Description,
    },
    Evidence {
        path: "crates/hawking-core/tests/rwkv7_parity.rs",
        claim: "RWKV7 parity + load_engine routing",
        kind: EvidenceKind::SmallCheckpointRun,
    },
    Evidence {
        path: "packs/hawking-adapters-extra.json",
        claim: "mamba2 extracted off-tree",
        kind: EvidenceKind::Description,
    },
];

const GAPS: &[&str] = &[
    "mamba2 not in shipping load_engine",
    "not PRODUCTION",
    "family spans RWKV (executes) and Mamba2 (declared pack only)",
];

const SOURCE_CLASSES: &[&str] = &[
    "gguf.rwkv7",
    "gguf.mamba2 (pack only)",
    "seed-c.ArchAdapter.mamba2",
];
const CODECS: &[&str] = &["gguf"];
const PRECISIONS: &[&str] = &["F16", "Q4_K", "Q8_0"];
const LIMITS: &[&str] = &[
    "mamba2 requires IR Ssm extension and pack hydrate",
    "RWKV is time-mix state, not attention KV",
];

const ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::some(
        "RWKV7: time-mix / channel-mix GGUF names per rwkv7 engine; \
         mamba2: SSM tensors (pack; non_transformer ArchAdapter refuses dense plan)",
    ),
    tokenizer: AbiField::some("gguf.tokenizer"),
    chat_template: AbiField::null("not owned by state-space engines"),
    attention_or_state: AbiField::some(
        "RWKV7 recurrent time-mix state (shipping); mamba2 selective SSM (pack only)",
    ),
    topology: AbiField::some("non-transformer state-space (dense residual backbone)"),
    normalization: AbiField::some("architecture-specific (RWKV LayerNorm/RMS variants per engine)"),
    positional_encoding: AbiField::some("none (recurrent state carries sequence)"),
    kv_or_state_format: AbiField::some(
        "RWKV wkv/state tensors; mamba2 SSM state (not attention KV)",
    ),
    context_limits: ContextLimits::unknown(
        "recurrent models have soft length limits; no sealed family receipt",
    ),
    artifact_codecs: AbiListField::some(CODECS),
    providers: ProviderAvailability {
        cpu: AbiField::some("available for rwkv7"),
        metal: AbiField::some("partial — when configured for rwkv7"),
        cuda: AbiField::null("no in-tree CUDA provider"),
    },
    fabric_partition_boundaries: AbiField::null("no Fabric partition for state-space family"),
    capability_limitations: LIMITS,
    source_precision_classes: AbiListField::some(PRECISIONS),
};

impl FamilyAdapter for StateSpaceFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        FamilyDescriptor {
            id: "state_space",
            aliases: ALIASES,
            display_name: "State-space (RWKV7 + Mamba2)",
            // Demoted: rwkv7_parity.rs skips when weights are absent.
            // No source-header receipt exists: state-space families were not in the Stage A
            // gauntlet's five, and the rwkv7_parity test it previously cited skips without
            // weights. DECLARED is what the evidence actually supports.
            level: SupportLevel::Declared,
            evidence: EVIDENCE,
            module: "crates/hawking-core/src/model/rwkv7.rs",
            executes: true,
            serve_registered: true,
            gaps: GAPS,
            abi: ABI,
        }
    }
}
