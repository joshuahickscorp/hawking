//! Architecture-family registry data authority.
//!
//! One counted table of [`FamilyDescriptor`] rows. No per-family modules,
//! trait objects, match arms, or runtime JSON.

use crate::abi::{
    AbiField, AbiListField, ContextLimits, Evidence, EvidenceKind, FamilyAbi, FamilyDescriptor,
    ProviderAvailability,
};
use crate::support_level::SupportLevel;

// Shared exact values (descriptor bytes unchanged).
const PACK_MODULE_GAPS: &[&str] = &[
    "module not in shipping load_engine",
    "pack hydrate required to execute",
    "not PRODUCTION",
];
const NO_ENGINE_PROVIDERS: ProviderAvailability = ProviderAvailability {
    cpu: AbiField::null("no engine"),
    metal: AbiField::null("no engine"),
    cuda: AbiField::null("no engine"),
};
const FABRIC_NOT_SERVE: AbiField = AbiField::null("not serve-registered");

/// Shared DECLARED pack ABI shell for gemma / phi.
const fn declared_pack_abi(
    source_config_classes: &'static [&'static str],
    tensor_namespace_rules: AbiField,
    attention_or_state: AbiField,
    normalization: AbiField,
    cpu_reject_reason: &'static str,
    capability_limitations: &'static [&'static str],
) -> FamilyAbi {
    FamilyAbi {
        source_config_classes: AbiListField::some(source_config_classes),
        tensor_namespace_rules,
        tokenizer: AbiField::some("gguf.tokenizer (declared in seed-c ArchAdapter)"),
        chat_template: AbiField::null("not owned by ArchAdapter plan path"),
        attention_or_state,
        topology: AbiField::some("dense transformer"),
        normalization,
        positional_encoding: AbiField::some("RoPE"),
        kv_or_state_format: AbiField::some("standard K/V cache (declared; not shipping execute)"),
        context_limits: ContextLimits::unknown("pack not hydrated in shipping tree"),
        artifact_codecs: AbiListField::some(&["gguf (pack only)"]),
        providers: ProviderAvailability {
            cpu: AbiField::null(cpu_reject_reason),
            metal: AbiField::null("not shipping"),
            cuda: AbiField::null("not shipping"),
        },
        fabric_partition_boundaries: FABRIC_NOT_SERVE,
        capability_limitations,
        source_precision_classes: AbiListField::some(&["F16", "Q4_K", "Q8_0"]),
    }
}

const LLAMA_EVIDENCE: &[Evidence] = &[
    Evidence { path: "crates/hawking-core/src/model/llama.rs", claim: "in-tree LlamaDense engine module", kind: EvidenceKind::Description },
    Evidence { path: "adapters/receipts/ADAPTER_LLAMA_RECEIPT.json", claim: "Stage A: official config/tokenizer/safetensors header parsed and mapped", kind: EvidenceKind::SourceHeader },
    Evidence { path: "crates/hawking-core/tests/llama32_smoke.rs", claim: "small-parent greedy smoke when GGUF present (skips when absent — not live grade evidence)", kind: EvidenceKind::SmallCheckpointRun },
    Evidence { path: "crates/hawking-core/tests/gravity_llama_forward.rs", claim: "gravity llama forward vs frozen oracle when off-tree .gravity present (skips when absent)", kind: EvidenceKind::RealTensorDecode },
    Evidence { path: "crates/hawking-core/src/model/mod.rs", claim: "load_engine dispatches llama|mistral GGUF arch strings", kind: EvidenceKind::Description },
];
const LLAMA_GAPS: &[&str] = &[
    "no standing PRODUCTION parity receipt",
    "smoke and gravity_llama_forward skip when weights/artifacts are absent",
    "REAL_TENSOR_DECODE / SMALL_REAL_CHECKPOINT require committed fixtures or on-disk parents",
    // Measured 2026-07-30 on Llama-3.2-1B-Instruct-Q4_K_M, evidence/tg/TG_EXTERNAL_BASELINE.json.
    // `executes` stays true by its own definition, a forward does run, but these
    // two are what a reader needs before treating that as working support.
    "executes on CPU only: dispatches_per_forward=0 and device_id=null, so no Metal path is reached",
    "output is degenerate on both a raw prompt and a correct Llama-3 chat template, while llama.cpp reads the same GGUF at 338 tok/s against our 3.63",
];
const LLAMA_SOURCE_CLASSES: &[&str] = &[
    "gguf.llama",
    "gguf.general.architecture=llama",
    "gravity.llama",
];
const LLAMA_LIMITS: &[&str] = &[
    "no PRODUCTION standing receipt",
    "gravity llama artifact is off-tree (CampaignS08 / HAWKING_GRAVITY_LLAMA_ARTIFACT)",
];
const LLAMA_ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(LLAMA_SOURCE_CLASSES),
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
    artifact_codecs: AbiListField::some(&["gguf", "gravity"]),
    providers: ProviderAvailability {
        cpu: AbiField::some("available — CPU backend path exercises LlamaDense"),
        metal: AbiField::some(
            "partial — Metal kernels used when configured; not family-grade sealed",
        ),
        cuda: AbiField::null("no in-tree CUDA provider for this family"),
    },
    fabric_partition_boundaries: AbiField::null(
        "Fabric partition placement not declared for llama; serve is single-process today",
    ),
    capability_limitations: LLAMA_LIMITS,
    source_precision_classes: AbiListField::some(&["F16", "BF16", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]),
};
const MISTRAL_EVIDENCE: &[Evidence] = &[
    Evidence { path: "crates/hawking-core/src/model/mod.rs", claim: "dense mistral arch string routes to LlamaDense", kind: EvidenceKind::Description },
    Evidence { path: "adapters/receipts/ADAPTER_MISTRAL_MIXTRAL_RECEIPT.json", claim: "Stage A: official config/tokenizer/safetensors header parsed and mapped", kind: EvidenceKind::SourceHeader },
    Evidence { path: "crates/hawking-core/tests/llama32_smoke.rs", claim: "dense llama-family small checkpoint path (mistral shares LlamaDense; skips without GGUF)", kind: EvidenceKind::SmallCheckpointRun },
    Evidence { path: "packs/hawking-adapters-extra.json", claim: "mixtral extracted off-tree to adapters-extra pack", kind: EvidenceKind::Description },
    Evidence { path: "crates/hawking-seed-c/src/providers/adapters.rs", claim: "seed-c ArchAdapter::mixtral is declarative plan-only (does not execute)", kind: EvidenceKind::Description },
];
const MISTRAL_GAPS: &[&str] = &[
    "mixtral MoE not in shipping load_engine",
    "seed-c ArchAdapter does not execute",
    "no PRODUCTION receipt",
    "SMALL_REAL_CHECKPOINT smoke skips when no GGUF is on disk",
];
const MISTRAL_SOURCE_CLASSES: &[&str] = &[
    "gguf.mistral",
    "gguf.general.architecture=mistral",
    "gguf.mixtral (pack only)",
    "seed-c.ArchAdapter.mixtral",
];
const MISTRAL_LIMITS: &[&str] = &[
    "mixtral requires adapters-extra hydrate",
    "dense path only for serve-registered execution",
    "no PRODUCTION standing receipt",
];
const MISTRAL_ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(MISTRAL_SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::some(
        "dense mistral: llama-family GGUF names via LlamaDense; mixtral: pack-defined MoE expert tensors",
    ),
    tokenizer: AbiField::some("gguf.tokenizer (dense); pack tokenizer for mixtral"),
    chat_template: AbiField::null("chat template at serve/prompt layer"),
    attention_or_state: AbiField::some("GQA causal attention (dense); mixtral MoE experts (pack)"),
    topology: AbiField::some("dense transformer (shipping); Mixtral MoE (pack only)"),
    normalization: AbiField::some("RMSNorm"),
    positional_encoding: AbiField::some("RoPE"),
    kv_or_state_format: AbiField::some("per-layer K/V cache (dense path)"),
    context_limits: ContextLimits::unknown(
        "no sealed family-level context limit receipt for mistral/mixtral",
    ),
    artifact_codecs: AbiListField::some(&["gguf", "pack (mixtral extra)"]),
    providers: ProviderAvailability {
        cpu: AbiField::some("available for dense mistral via LlamaDense"),
        metal: AbiField::some("partial — shared llama metal path"),
        cuda: AbiField::null("no in-tree CUDA provider"),
    },
    fabric_partition_boundaries: AbiField::null(
        "no Fabric partition declaration for mistral/mixtral",
    ),
    capability_limitations: MISTRAL_LIMITS,
    source_precision_classes: AbiListField::some(&["F16", "BF16", "Q4_K", "Q5_K", "Q8_0"]),
};
const QWEN_EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "crates/hawking-core/src/model/qwen_dense.rs",
        claim: "in-tree QwenDense engine",
        kind: EvidenceKind::Description,
    },
    Evidence {
        path: "crates/hawking-core/src/model/qwen_moe.rs",
        claim: "in-tree QwenMoE engine",
        kind: EvidenceKind::Description,
    },
    Evidence {
        path: "adapters/receipts/ADAPTER_QWEN_RECEIPT.json",
        claim: "Stage A: official config/tokenizer/safetensors header parsed and mapped",
        kind: EvidenceKind::SourceHeader,
    },
    Evidence {
        path: "crates/hawking-core/tests/integration_greedy_64.rs",
        claim: "greedy integration path (skips when no model on disk — not live grade evidence)",
        kind: EvidenceKind::FullParentValidation,
    },
    Evidence {
        path: "crates/hawking-core/tests/cpu_backend_parity.rs",
        claim: "CPU backend parity (skips when no qwen0.5b weights)",
        kind: EvidenceKind::FullParentValidation,
    },
    Evidence {
        path: "crates/hawking-core/tests/qwen_tq_serve_parity.rs",
        claim: "TQ serve parity (#[ignore] + skips without weights/sidecar)",
        kind: EvidenceKind::SmallCheckpointRun,
    },
    Evidence {
        path: "crates/hawking-core/src/model/mod.rs",
        claim: "load_engine dispatches qwen2/qwen2moe",
        kind: EvidenceKind::Description,
    },
];
const QWEN_GAPS: &[&str] = &[
    "not PRODUCTION: no standing production parity receipt under continuous serve",
    "large MoE parents (235B/397B) are campaign-side, not this registry's PRODUCTION claim",
    "integration/parity tests skip or are #[ignore] without on-disk parents",
];
const QWEN_SOURCE_CLASSES: &[&str] = &[
    "gguf.qwen2",
    "gguf.qwen2moe",
    "gguf.general.architecture=qwen2|qwen2moe",
];
const QWEN_LIMITS: &[&str] = &[
    "no PRODUCTION standing receipt",
    "235B/397B MoE parents not claimed FULL_PARENT at this registry grade for those sizes",
];
const QWEN_ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(QWEN_SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::some(
        "GGUF qwen2 dense + qwen2moe: llama-family-like blk.{l} names; MoE uses \
         ffn_gate_inp router and expert tensors per GGUF qwen2moe layout",
    ),
    tokenizer: AbiField::some("gguf.tokenizer (Qwen vocab embedded)"),
    chat_template: AbiField::null(
        "chat template applied at serve/prompt layer; not declared inside engine modules",
    ),
    attention_or_state: AbiField::some("GQA causal attention"),
    topology: AbiField::some("dense (QwenDense) and MoE (QwenMoE) both in-tree"),
    normalization: AbiField::some("RMSNorm"),
    positional_encoding: AbiField::some("RoPE (qwen rope_theta from metadata)"),
    kv_or_state_format: AbiField::some("per-layer K/V cache"),
    context_limits: ContextLimits::unknown(
        "context from GGUF/EngineConfig; large-parent long-context not sealed as family limit",
    ),
    artifact_codecs: AbiListField::some(&["gguf", "tq (feature-gated)"]),
    providers: ProviderAvailability {
        cpu: AbiField::some("available — CPU backend parity tests"),
        metal: AbiField::some("partial — Metal path used in serve when configured"),
        cuda: AbiField::null("no in-tree CUDA provider"),
    },
    fabric_partition_boundaries: AbiField::null(
        "no Fabric multi-node partition declaration for qwen family",
    ),
    capability_limitations: QWEN_LIMITS,
    source_precision_classes: AbiListField::some(&[
        "F16", "BF16", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "TQ",
    ]),
};
const GLM_EVIDENCE: &[Evidence] = &[
    Evidence { path: "evidence/glm52/GLM52_FLAGSHIP_ADAPTER_PARITY.json", claim: "M04_SEALED: Rust adapter vs oracle on real flagship .gravity shards", kind: EvidenceKind::SmallCheckpointRun },
    Evidence { path: "crates/hawking-core/src/model/gravity_engine.rs", claim: "GravityEngine dispatches glm_moe_dsa", kind: EvidenceKind::Description },
    Evidence { path: "crates/hawking-core/tests/gravity_engine_registry.rs", claim: "registry path for .gravity artifacts", kind: EvidenceKind::Description },
    // Stage C: this test runs UNCONDITIONALLY. Unlike the weight-gated tests, it does
    // not skip -- the fixture is committed. Real container, real codec, real oracle.
    Evidence { path: "crates/hawking-core/tests/gravity_glm_forward.rs", claim: "unconditional: real .gravity container decoded, complete token executed, matches numpy oracle reading the same container (3 passed, 0.04s)", kind: EvidenceKind::RealTensorDecode },
];
const GLM_GAPS: &[&str] = &[
    "not PRODUCTION",
    "gravity_glm.rs is another lane's sealed path — not claimed as open production serve",
    "full parent source safetensors not the parity authority (gravity bytes are)",
];
const GLM_SOURCE_CLASSES: &[&str] = &[
    "gravity.glm_moe_dsa",
    "hf.config (GLM-5.2 — campaign ladder, not serve authority)",
];
const GLM_PRECISIONS: &[&str] = &[
    "gravity-packed",
    "F8_E4M3 (source parent)",
    "BF16 (source parent)",
];
const GLM_LIMITS: &[&str] = &[
    "execution is .gravity only for glm_moe_dsa — not raw HF safetensors serve",
    "MLA+DSA indexer shapes are gravity-contract specific",
    "no PRODUCTION standing receipt",
];
const GLM_ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(GLM_SOURCE_CLASSES),
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
    topology: AbiField::some(
        "MoE (routed experts + shared) with dense early layers — gravity contract",
    ),
    normalization: AbiField::some("RMSNorm (gravity contract)"),
    positional_encoding: AbiField::some("RoPE on MLA rope dims (gravity contract)"),
    kv_or_state_format: AbiField::some("compact MLA cache / gravity-resident state layouts"),
    context_limits: ContextLimits::known(1_048_576, "declared_from_source_config_unvalidated_at_serve"),
    artifact_codecs: AbiListField::some(&["gravity"]),
    providers: ProviderAvailability {
        cpu: AbiField::some("partial — CPU oracle/parity paths exist for gravity shards"),
        metal: AbiField::some("partial — Metal gravity kernels used in sealed parity lanes"),
        cuda: AbiField::null("no in-tree CUDA provider for glm_moe_dsa"),
    },
    fabric_partition_boundaries: AbiField::some(
        "gravity shard boundaries are the partition unit; Fabric multi-node placement not declared",
    ),
    capability_limitations: GLM_LIMITS,
    source_precision_classes: AbiListField::some(GLM_PRECISIONS),
};
const DEEPSEEK_EVIDENCE: &[Evidence] = &[
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
const DEEPSEEK_GAPS: &[&str] = &[
    "not FULL_PARENT_VALIDATED: no sealed full-size parent receipt in registry evidence",
    "not PRODUCTION",
    "DeepSeek V3/V4 MLA+DSA ladder rungs are NOT this family's shipping GGUF deepseek2 path",
];
const DEEPSEEK_SOURCE_CLASSES: &[&str] = &["gguf.deepseek2", "gguf.general.architecture=deepseek2"];
const DEEPSEEK_LIMITS: &[&str] = &[
    "shipping path is GGUF deepseek2 — not V3/V4 safetensors ladder",
    "no PRODUCTION receipt",
];
const DEEPSEEK_ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(DEEPSEEK_SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::some(
        "GGUF deepseek2 tensor names as consumed by DeepSeekV2 engine",
    ),
    tokenizer: AbiField::some("gguf.tokenizer"),
    chat_template: AbiField::null("chat template not owned by DeepSeekV2 engine module"),
    attention_or_state: AbiField::some(
        "MLA-style / multi-latent attention as implemented in deepseek_v2",
    ),
    topology: AbiField::some("MoE (DeepSeek V2 GGUF layout) as implemented in-tree"),
    normalization: AbiField::some("RMSNorm"),
    positional_encoding: AbiField::some("RoPE (MLA rope dims)"),
    kv_or_state_format: AbiField::some("MLA compressed KV / latent cache per engine"),
    context_limits: ContextLimits::unknown(
        "context from GGUF metadata; no sealed family-level long-context receipt",
    ),
    artifact_codecs: AbiListField::some(&["gguf"]),
    providers: ProviderAvailability {
        cpu: AbiField::some("available — cpu_backend_parity_deepseek"),
        metal: AbiField::some("partial — shared metal path when configured"),
        cuda: AbiField::null("no in-tree CUDA provider"),
    },
    fabric_partition_boundaries: AbiField::null("no Fabric partition declaration for deepseek2"),
    capability_limitations: DEEPSEEK_LIMITS,
    source_precision_classes: AbiListField::some(&["F16", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]),
};
const KIMI_EVIDENCE: &[Evidence] = &[
    Evidence {
        path: "evidence/kimi-k26/KIMI_K26_ADAPTER_TWIN.json",
        claim: "synthetic CPU reference + bound real-source metal K1 twin",
        kind: EvidenceKind::SyntheticParity,
    },
    Evidence {
        path: "crates/hawking-core/src/model/mod.rs",
        claim: "load_engine has no kimi arch arm (not serve-registered)",
        kind: EvidenceKind::Description,
    },
];
const KIMI_GAPS: &[&str] = &[
    "not serve-registered in load_engine",
    "no SMALL_REAL_CHECKPOINT sealed receipt for full generate path",
    "not PRODUCTION",
];
const KIMI_SOURCE_CLASSES: &[&str] = &[
    "hf.config (Kimi-K2.x text core DeepseekV3-like)",
    "adapter_twin.kimi_k26",
];
const KIMI_LIMITS: &[&str] = &[
    "no load_engine arch arm",
    "tokenizer is tiktoken (not HF tokenizers JSON)",
    "multimodal vision tower out of scope for this family entry",
];
const KIMI_ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(KIMI_SOURCE_CLASSES),
    tensor_namespace_rules: AbiField::some(
        "DeepseekV3-like MLA MoE names on text core; twin maps a reduced synthetic namespace",
    ),
    tokenizer: AbiField::some(
        "tiktoken (tiktoken.model + tokenization_kimi.py) — not HF tokenizers JSON",
    ),
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
    artifact_codecs: AbiListField::some(&["safetensors (source)", "synthetic twin"]),
    providers: ProviderAvailability {
        cpu: AbiField::some("synthetic CPU reference only"),
        metal: AbiField::some("partial — bound real-source metal K1 twin, not full serve"),
        cuda: AbiField::null("no CUDA path declared"),
    },
    fabric_partition_boundaries: AbiField::null("not serve-registered; no Fabric placement"),
    capability_limitations: KIMI_LIMITS,
    source_precision_classes: AbiListField::some(&[
        "INT4 group-32 (routed experts)",
        "BF16",
        "F32",
    ]),
};
const MINIMAX_EVIDENCE: &[Evidence] = &[Evidence {
    path: "evidence/fabric/FABRIC_BRIDGE_ARCHAEOLOGY.md",
    claim: "family listed in bridge archaeology; no serve path found",
    kind: EvidenceKind::Description,
}];
const MINIMAX_GAPS: &[&str] = &[
    "no in-tree engine module",
    "not serve-registered",
    "not PRODUCTION",
];
const MINIMAX_SOURCE_CLASSES: &[&str] = &["hf.config (MiniMax-M3 — ladder F3, declared only)"];
const MINIMAX_LIMITS: &[&str] = &[
    "block-sparse top-k attention not in shipping ArchAdapter table",
    "no execution path",
];
const MINIMAX_ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(MINIMAX_SOURCE_CLASSES),
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
    positional_encoding: AbiField::some(
        "declared: partial RoPE (partial_rotary_factor 0.5) on source",
    ),
    kv_or_state_format: AbiField::null("not implemented"),
    context_limits: ContextLimits::unknown("no in-tree validation; source config not loaded here"),
    artifact_codecs: AbiListField::null("no codec path in shipping tree"),
    providers: NO_ENGINE_PROVIDERS,
    fabric_partition_boundaries: FABRIC_NOT_SERVE,
    capability_limitations: MINIMAX_LIMITS,
    source_precision_classes: AbiListField::some(&["BF16", "F32"]),
};
const GEMMA_EVIDENCE: &[Evidence] = &[
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
const GEMMA_SOURCE_CLASSES: &[&str] = &[
    "gguf.gemma2",
    "seed-c.ArchAdapter.gemma2",
    "pack.hawking-adapters-extra.gemma2",
];
const GEMMA_LIMITS: &[&str] = &[
    "not in shipping load_engine",
    "ArchAdapter is plan-summary only",
    "logit softcap + pre/post FFN norm exceptions",
];
const GEMMA_ABI: FamilyAbi = declared_pack_abi(
    GEMMA_SOURCE_CLASSES,
    AbiField::some(
        "GGUF llama-family tensor names; exceptions: attn_logit_softcap, final_logit_softcap, pre+post ffn norm",
    ),
    AbiField::some("GQA with logit soft-capping (declared)"),
    AbiField::some("RMSNorm with pre+post FFN norm exception"),
    "shipping load_engine rejects gemma2 without pack",
    GEMMA_LIMITS,
);
const PHI_EVIDENCE: &[Evidence] = &[
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
const PHI_SOURCE_CLASSES: &[&str] = &[
    "gguf.phi3",
    "seed-c.ArchAdapter.phi3",
    "pack.hawking-adapters-extra.phi3",
];
const PHI_LIMITS: &[&str] = &[
    "not in shipping load_engine",
    "fused qkv and gate_up exceptions",
];
const PHI_ABI: FamilyAbi = declared_pack_abi(
    PHI_SOURCE_CLASSES,
    AbiField::some(
        "GGUF llama-family names with fused qkv + fused gate_up exceptions (split at plan time)",
    ),
    AbiField::some("GQA/MHA causal attention (declared)"),
    AbiField::some("RMSNorm"),
    "shipping load_engine rejects phi3 without pack",
    PHI_LIMITS,
);
const STATE_SPACE_EVIDENCE: &[Evidence] = &[
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
const STATE_SPACE_GAPS: &[&str] = &[
    "mamba2 not in shipping load_engine",
    "not PRODUCTION",
    "family spans RWKV (executes) and Mamba2 (declared pack only)",
];
const STATE_SPACE_SOURCE_CLASSES: &[&str] = &[
    "gguf.rwkv7",
    "gguf.mamba2 (pack only)",
    "seed-c.ArchAdapter.mamba2",
];
const STATE_SPACE_LIMITS: &[&str] = &[
    "mamba2 requires IR Ssm extension and pack hydrate",
    "RWKV is time-mix state, not attention KV",
];
const STATE_SPACE_ABI: FamilyAbi = FamilyAbi {
    source_config_classes: AbiListField::some(STATE_SPACE_SOURCE_CLASSES),
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
    artifact_codecs: AbiListField::some(&["gguf"]),
    providers: ProviderAvailability {
        cpu: AbiField::some("available for rwkv7"),
        metal: AbiField::some("partial — when configured for rwkv7"),
        cuda: AbiField::null("no in-tree CUDA provider"),
    },
    fabric_partition_boundaries: AbiField::null("no Fabric partition for state-space family"),
    capability_limitations: STATE_SPACE_LIMITS,
    source_precision_classes: AbiListField::some(&["F16", "Q4_K", "Q8_0"]),
};

// Sole family table (stable authoring order; registry emits id-sorted)
/// All built-in family descriptors. Registration order is authoring order;
/// [`crate::registry::FamilyRegistry`] emits via BTreeMap id order.
pub const FAMILY_TABLE: &[FamilyDescriptor] = &[
    FamilyDescriptor {
        id: "llama",
        aliases: &["llama", "llama2", "llama3", "llama3.2"],
        display_name: "Llama (dense GGUF + gravity)",
        // Demoted: llama32_smoke and gravity_llama_forward skip without on-disk
        // weights/artifacts. Stage A source-header receipt is the live evidence.
        level: SupportLevel::SourceHeaderValidated,
        evidence: LLAMA_EVIDENCE,
        module: "crates/hawking-core/src/model/llama.rs",
        executes: true,
        serve_registered: true,
        gaps: LLAMA_GAPS,
        abi: LLAMA_ABI,
    },
    FamilyDescriptor {
        id: "mistral_mixtral",
        aliases: &["mistral", "mixtral", "mistral_mixtral"],
        display_name: "Mistral / Mixtral",
        // Demoted: only live evidence is Stage A source-header receipt.
        // llama32_smoke skips without GGUF; mixtral is pack/extracted.
        level: SupportLevel::SourceHeaderValidated,
        evidence: MISTRAL_EVIDENCE,
        module: "crates/hawking-core/src/model/llama.rs (+ pack mixtral)",
        executes: true,
        serve_registered: true,
        gaps: MISTRAL_GAPS,
        abi: MISTRAL_ABI,
    },
    FamilyDescriptor {
        id: "qwen",
        aliases: &["qwen", "qwen2", "qwen2moe", "qwen-moe", "qwen3"],
        display_name: "Qwen (dense + MoE)",
        // Demoted from FullParentValidated then SmallRealCheckpoint: cited tests skip
        // or are #[ignore] without on-disk parents. Stage A source-header receipt is live.
        level: SupportLevel::SourceHeaderValidated,
        evidence: QWEN_EVIDENCE,
        module: "crates/hawking-core/src/model/qwen_dense.rs",
        executes: true,
        serve_registered: true,
        gaps: QWEN_GAPS,
        abi: QWEN_ABI,
    },
    FamilyDescriptor {
        id: "glm",
        aliases: &["glm", "glm52", "glm_moe_dsa", "glm-5.2"],
        display_name: "GLM (gravity glm_moe_dsa)",
        level: SupportLevel::SmallRealCheckpoint,
        evidence: GLM_EVIDENCE,
        module: "crates/hawking-core/src/model/gravity_engine.rs",
        executes: true,
        serve_registered: true,
        gaps: GLM_GAPS,
        abi: GLM_ABI,
    },
    FamilyDescriptor {
        id: "deepseek",
        aliases: &["deepseek", "deepseek2", "deepseek_v2"],
        display_name: "DeepSeek V2",
        // Demoted: cpu_backend_parity_deepseek.rs skips when weights are absent.
        level: SupportLevel::SourceHeaderValidated,
        evidence: DEEPSEEK_EVIDENCE,
        module: "crates/hawking-core/src/model/deepseek_v2.rs",
        executes: true,
        serve_registered: true,
        gaps: DEEPSEEK_GAPS,
        abi: DEEPSEEK_ABI,
    },
    FamilyDescriptor {
        id: "kimi",
        aliases: &["kimi", "kimi_k2", "kimi_k26", "moonshot"],
        display_name: "Kimi K2.x",
        level: SupportLevel::SyntheticParity,
        evidence: KIMI_EVIDENCE,
        module: "KIMI_K26_ADAPTER_TWIN.json (reference twin; no in-tree serve module)",
        executes: false,
        serve_registered: false,
        gaps: KIMI_GAPS,
        abi: KIMI_ABI,
    },
    FamilyDescriptor {
        id: "minimax",
        aliases: &["minimax", "minimax_m3", "minimax-m3"],
        display_name: "MiniMax",
        level: SupportLevel::Declared,
        evidence: MINIMAX_EVIDENCE,
        module: "(none — declared only)",
        executes: false,
        serve_registered: false,
        gaps: MINIMAX_GAPS,
        abi: MINIMAX_ABI,
    },
    FamilyDescriptor {
        id: "gemma",
        aliases: &["gemma", "gemma2"],
        display_name: "Gemma 2",
        level: SupportLevel::Declared,
        evidence: GEMMA_EVIDENCE,
        module: "packs/hawking-adapters-extra (gemma2)",
        executes: false,
        serve_registered: false,
        gaps: PACK_MODULE_GAPS,
        abi: GEMMA_ABI,
    },
    FamilyDescriptor {
        id: "phi",
        aliases: &["phi", "phi3"],
        display_name: "Phi-3",
        level: SupportLevel::Declared,
        evidence: PHI_EVIDENCE,
        module: "packs/hawking-adapters-extra (phi3)",
        executes: false,
        serve_registered: false,
        gaps: PACK_MODULE_GAPS,
        abi: PHI_ABI,
    },
    FamilyDescriptor {
        id: "state_space",
        aliases: &["state_space", "rwkv7", "rwkv", "mamba2", "mamba"],
        display_name: "State-space (RWKV7 + Mamba2)",
        // Demoted: rwkv7_parity.rs skips when weights are absent.
        // No source-header receipt exists: state-space families were not in the Stage A
        // gauntlet's five, and the rwkv7_parity test it previously cited skips without
        // weights. DECLARED is what the evidence actually supports.
        level: SupportLevel::Declared,
        evidence: STATE_SPACE_EVIDENCE,
        module: "crates/hawking-core/src/model/rwkv7.rs",
        executes: true,
        serve_registered: true,
        gaps: STATE_SPACE_GAPS,
        abi: STATE_SPACE_ABI,
    },
];
