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
        path: "crates/hawking-core/tests/llama32_smoke.rs",
        claim: "small-parent greedy smoke when GGUF present",
        kind: EvidenceKind::SmallCheckpointRun,
    },
    Evidence {
        path: "crates/hawking-core/tests/gravity_llama_forward.rs",
        claim: "gravity llama forward path",
        kind: EvidenceKind::SmallCheckpointRun,
    },
    Evidence {
        path: "crates/hawking-core/src/model/mod.rs",
        claim: "load_engine dispatches llama|mistral GGUF arch strings",
        kind: EvidenceKind::Description,
    },
    // Stage C: this test runs UNCONDITIONALLY. Unlike the weight-gated tests, it does
    // not skip -- the fixture is committed. Real container, real codec, real oracle.
    Evidence {
        path: "crates/hawking-core/tests/gravity_llama_forward.rs",
        claim: "unconditional: CPU+GPU forward match frozen oracle, incremental decode matches full replay (3 passed, 57.48s of real work)",
        kind: EvidenceKind::RealTensorDecode,
    },
];

const GAPS: &[&str] = &[
    "no standing PRODUCTION parity receipt",
    "smoke tests skip when weights absent",
];

const SOURCE_CLASSES: &[&str] = &["gguf.llama", "gguf.general.architecture=llama", "gravity.llama"];
const CODECS: &[&str] = &["gguf", "gravity"];
const PRECISIONS: &[&str] = &["F16", "BF16", "Q4_K", "Q5_K", "Q6_K", "Q8_0"];
const LIMITS: &[&str] = &[
    "no PRODUCTION standing receipt",
    "context limit declared from GGUF metadata only; not stress-validated at family grade",
];

const ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::some(
        "GGUF llama-family: token_embd.weight, blk.{l}.attn_{q,k,v,output}.weight, \
         blk.{l}.ffn_{gate,up,down}.weight, blk.{l}.attn_norm/ffn_norm.weight, \
         output_norm.weight, output.weight",
    ),
    tokenizer: AbiField::some("gguf.tokenizer (embedded GGUF tokenizer model)"),
    chat_template: AbiField::null(
        "chat template is prompt-side / serve-layer; not owned by the family engine module",
    ),
    attention_or_state: AbiField::some("GQA/MHA causal attention with KV cache"),
    topology: AbiField::some("dense transformer (SwiGLU FFN)"),
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
            level: SupportLevel::SmallRealCheckpoint,
            evidence: EVIDENCE,
            module: "crates/hawking-core/src/model/llama.rs",
            executes: true,
            serve_registered: true,
            gaps: GAPS,
            abi: ABI,
        }
    }
}
