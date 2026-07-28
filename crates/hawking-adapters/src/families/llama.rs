//! Llama family (Llama-2/3.x dense via GGUF + gravity llama).

use crate::abi::{
    AbiField, AbiListField, ContextLimits, Evidence, EvidenceKind, FamilyAbi, FamilyAdapter,
    FamilyDescriptor, ProviderAvailability,
};
use crate::support_level::SupportLevel;

pub struct LlamaFamily;

const ALIASES: &[&str] = &["llama", "llama2", "llama3", "llama3.2"];

const EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "crates/hawking-core/src/model/llama.rs",
        claim: "in-tree LlamaDense engine module",
        kind: EvidenceKind::Description,
    },
    Evidence {
        path: "adapters/receipts/ADAPTER_LLAMA_RECEIPT.json",
        claim: "Stage A: official config/tokenizer/safetensors header parsed and mapped",
        kind: EvidenceKind::SourceHeader,
    },
    Evidence {
        path: "crates/hawking-core/tests/llama32_smoke.rs",
        claim: "small-parent greedy smoke when GGUF present (skips when absent — not live grade evidence)",
        kind: EvidenceKind::SmallCheckpointRun,
    },
    Evidence {
        path: "crates/hawking-core/tests/gravity_llama_forward.rs",
        claim: "gravity llama forward vs frozen oracle when off-tree .gravity present (skips when absent)",
        kind: EvidenceKind::RealTensorDecode,
    },
    Evidence {
        path: "crates/hawking-core/src/model/mod.rs",
        claim: "load_engine dispatches llama|mistral GGUF arch strings",
        kind: EvidenceKind::Description,
    },
];

const GAPS: &[&str] = &[
    "no standing PRODUCTION parity receipt",
    "smoke and gravity_llama_forward skip when weights/artifacts are absent",
    "REAL_TENSOR_DECODE / SMALL_REAL_CHECKPOINT require committed fixtures or on-disk parents",
];

const SOURCE_CLASSES: &[&str] = &["gguf.llama", "gguf.general.architecture=llama", "gravity.llama"];
const CODECS: &[&str] = &["gguf", "gravity"];
const PRECISIONS: &[&str] = &["F16", "BF16", "Q4_K", "Q5_K", "Q6_K", "Q8_0"];
const LIMITS: &[&str] = &[
    "no PRODUCTION standing receipt",
    "gravity llama artifact is off-tree (CampaignS08 / HAWKING_GRAVITY_LLAMA_ARTIFACT)",
];

const ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::some(
        "GGUF llama: token_embd, output_norm, output, blk.{l}.attn_*/ffn_*; gravity.llama codec for .gravity",
    ),
    tokenizer: AbiField::some("gguf.tokenizer (vocab embedded in GGUF)"),
    chat_template: AbiField::null(
        "chat template applied at serve/prompt layer; not declared inside LlamaDense",
    ),
    attention_or_state: AbiField::some("GQA / MHA causal attention per GGUF metadata"),
    topology: AbiField::some("dense transformer"),
    normalization: AbiField::some("RMSNorm"),
    positional_encoding: AbiField::some("RoPE (rope_base from GGUF metadata)"),
    kv_or_state_format: AbiField::some("per-layer K/V cache; layout owned by Engine/KV runtime"),
    context_limits: ContextLimits::unknown(
        "max context is GGUF-metadata / EngineConfig driven; no sealed family-level limit receipt",
    ),
    artifact_codecs: AbiListField::some(CODECS),
    providers: ProviderAvailability {
        cpu: AbiField::some("available — CPU backend path exercises LlamaDense"),
        metal: AbiField::some("partial — Metal kernels used when configured; not family-grade sealed"),
        cuda: AbiField::null("no in-tree CUDA provider for this family"),
    },
    fabric_partition_boundaries: AbiField::null(
        "Fabric partition placement not declared for llama; serve is single-process today",
    ),
    capability_limitations: LIMITS,
    source_precision_classes: AbiListField::some(PRECISIONS),
};

impl FamilyAdapter for LlamaFamily {
    fn descriptor(&self) -> FamilyDescriptor {
        FamilyDescriptor {
            id: "llama",
            aliases: ALIASES,
            display_name: "Llama (dense GGUF + gravity)",
            // Demoted: llama32_smoke and gravity_llama_forward skip without on-disk
            // weights/artifacts. Stage A source-header receipt is the live evidence.
            level: SupportLevel::SourceHeaderValidated,
            evidence: EVIDENCE,
            module: "crates/hawking-core/src/model/llama.rs",
            executes: true,
            serve_registered: true,
            gaps: GAPS,
            abi: ABI,
        }
    }
}
