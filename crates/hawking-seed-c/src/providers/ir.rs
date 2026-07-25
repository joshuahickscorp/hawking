//! **The one architecture execution IR.** Every [`super::adapters::ArchAdapter`] emits an
//! [`OperationPlan`]: an ordered sequence of typed ops, each carrying its inputs, outputs, shape
//! relations, required tensor roles, and the config fields that parameterize it.
//!
//! This is the ARCHITECTURE-level plan IR, not a second runtime. [`crate::ir`] stays the ONE runtime
//! register IR that the CPU/Metal bit-identity path executes; an `OperationPlan` describes what an
//! architecture *is* (so it can be validated, diffed, and sealed into a launch packet as JSON) and
//! lowers to `crate::ir::Plan` only for the families that runtime already executes.
//!
//! Two hard boundaries are enforced here:
//! 1. **Capability/profile**: an op declares a [`Capability`]; a [`PlanProfile`] declares what it allows.
//!    A dense text-core claim can never silently contain MoE or multimodal work.
//! 2. **Multimodal is a different type**: vision ops live in [`VisionOp`], so a `Vec<Op>` cannot
//!    represent them at all. A text-core plan is multimodal-free by construction, not by convention.
//!
//! Families marked PROVISIONAL have no official config field binding yet; they are declared so a plan
//! can name them honestly, and `validate()` still refuses an incoherent one. Provisional never means
//! "measured".

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

fn err(msg: String) -> crate::Error {
    crate::Error::Adapter(format!("ir: {msg}"))
}

/// Model-wide dimensions every op is checked against (official config: hidden_size, vocab_size,
/// num_hidden_layers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dims {
    pub hidden: usize,
    pub vocab: usize,
    pub n_layers: usize,
}

/// The value a op reads or writes. Symbolic, not a register allocation (that is the runtime's job).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Slot {
    Tokens,
    Hidden,
    Normed,
    Q,
    K,
    V,
    KvLatent,
    Index,
    AttnOut,
    RouterWeights,
    ExpertOut,
    Logits,
    Sampled,
    Pixels,
    VisionEmbed,
}

/// A tensor role a plan declares as present in the source. Ops reference roles, never file names;
/// name mapping stays in [`super::adapters::TensorNames`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorRole {
    TokenEmbd,
    OutputHead,
    NormWeight,
    LinearWeight,
    CompactWeight,
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    KvADown,
    KvBUp,
    QDown,
    QUp,
    Indexer,
    AttnSink,
    ShortConv,
    DeltaGate,
    Router,
    RouterBias,
    ExpertGate,
    ExpertUp,
    ExpertDown,
    SharedExpert,
    LatentBasis,
    HyperConnection,
    MtpHead,
    VisionEncoder,
    VisionProjector,
}

/// What an op needs the profile to permit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Dense text core: embedding, norms, linears, attention, activation, residual, sampling.
    Core,
    /// Routed sparsity: router, experts, shared expert, latent MoE, weighted combine.
    Moe,
    /// Vision tower and projector.
    Multimodal,
}

/// The claim a plan makes about itself. `validate()` refuses any op outside the declared profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanProfile {
    TextCoreDense,
    TextCoreMoe,
    TextVision,
}

impl PlanProfile {
    pub fn allows(self, c: Capability) -> bool {
        matches!(
            (self, c),
            (_, Capability::Core)
                | (PlanProfile::TextCoreMoe, Capability::Moe)
                | (PlanProfile::TextVision, Capability::Moe)
                | (PlanProfile::TextVision, Capability::Multimodal)
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormKind {
    /// rms_norm_eps (llama, qwen3, deepseek).
    RmsNorm,
    /// layer_norm_epsilon (gpt-oss vision-free core, phi).
    LayerNorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeScaling {
    /// rope_scaling absent.
    None,
    /// rope_scaling.type = "linear", factor.
    Linear { factor_x1000: u32 },
    /// rope_scaling.type = "yarn", factor + original_max_position_embeddings.
    Yarn { factor_x1000: u32, original_ctx: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActKind {
    /// hidden_act = "silu" with gate/up (llama, qwen3, deepseek).
    SiluGated,
    /// hidden_act = "gelu"/"gelu_new" with gate/up.
    GeluGated,
    /// gpt-oss clamped swiglu: swiglu_limit, alpha (official gpt-oss reference implementation).
    SwigluClamped { limit_x1000: u32, alpha_x1000: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterScoring {
    /// scoring_func = "softmax" (mixtral, qwen3-moe, olmoe).
    Softmax,
    /// scoring_func = "sigmoid" (DeepSeek V3).
    Sigmoid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactFormat {
    /// gpt-oss MXFP4 blocks + scales (declared in the safetensors dtype table).
    Mxfp4,
    /// the Seed's ternary compact operator ([`crate::subbit`]).
    Ternary,
    /// residual vector quantization (sub-bit foundry).
    Rvq,
}

/// The text-core operation families. Multimodal is deliberately NOT representable here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    /// Token embedding lookup. Every transformer (config: vocab_size, hidden_size).
    Embed { vocab: usize, hidden: usize },
    /// lm_head projection to vocab; `tied` = tie_word_embeddings (gemma, qwen3-small).
    FinalProjection { vocab: usize, hidden: usize, tied: bool },
    /// Pre/post normalization (config: rms_norm_eps or layer_norm_epsilon). `n` = normalized width,
    /// hidden for block norms, head_dim for qk-norm (qwen3, olmoe).
    Norm { kind: NormKind, n: usize, eps_x1e9: u64 },
    /// A dense projection. `role` binds it to the tensor it consumes.
    Linear { role: TensorRole, in_features: usize, out_features: usize, bias: bool },
    /// A projection executed directly on the compressed representation, never densified
    /// (gpt-oss MXFP4 experts; the Seed's sub-bit compact operator).
    CompactLinear { role: TensorRole, in_features: usize, out_features: usize, block: usize, format: CompactFormat },
    /// Rotary embedding (config: rope_theta, rope_scaling). llama, qwen3, deepseek.
    RoPE { n_heads: usize, head_dim: usize, base_x1000: u64, scaling: RopeScaling },
    /// Full multi-head attention, num_key_value_heads == num_attention_heads (gpt-2 era, phi3).
    MHA { n_heads: usize, head_dim: usize },
    /// Grouped-query attention (config: num_key_value_heads); sliding_window when declared (gemma2, gpt-oss).
    GQA { n_heads: usize, n_kv_heads: usize, head_dim: usize, sliding_window: Option<usize> },
    /// Multi-head latent attention: DeepSeek V3/V3.2 kv_lora_rank, q_lora_rank,
    /// qk_nope_head_dim, qk_rope_head_dim, v_head_dim.
    MLA {
        n_heads: usize,
        kv_lora_rank: usize,
        q_lora_rank: Option<usize>,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
    },
    /// Sparse attention over an indexer-selected token set: DeepSeek V3.2-Exp lightning indexer
    /// (index_n_heads, index_head_dim, index_topk).
    SparseAttention { n_heads: usize, head_dim: usize, index_topk: usize, indexer_heads: usize, indexer_dim: usize },
    /// Compressed + selected + sliding branches: Native Sparse Attention (DeepSeek, arXiv 2502.11089).
    CompressedSparseAttention { n_heads: usize, head_dim: usize, block: usize, compress_stride: usize, select_blocks: usize, window: usize },
    /// PROVISIONAL: KV compressed harder than MLA (rank <= hidden/8). No official config field binds
    /// this yet; declared so a plan can name the claim without pretending it is measured.
    HeavilyCompressedAttention { n_heads: usize, head_dim: usize, kv_rank: usize },
    /// MiniMax hybrid lightning/softmax attention (config: attn_type_list, block-wise decay).
    MiniMaxSparseAttention { n_heads: usize, head_dim: usize, block: usize, window: usize },
    /// Gated linear recurrent attention (Gated DeltaNet, arXiv 2412.06464); Qwen3-Next linear layers
    /// (linear_num_value_heads, linear_conv_kernel_dim).
    DeltaNet { n_heads: usize, head_dim: usize, expand_v: usize, conv_kernel: usize },
    /// PROVISIONAL: Kimi Delta Attention (Kimi Linear / K3 provisional); KDA with a short conv branch.
    KimiDeltaAttention { n_heads: usize, head_dim: usize, conv_kernel: usize, short_conv: usize },
    /// Residual add of the attention sublayer output; `scale_x1000` != 1000 only when the config
    /// declares a residual multiplier (IBM Granite residual_multiplier).
    AttentionResidual { scale_x1000: u32 },
    /// Learned per-head attention sinks (gpt-oss `sinks` tensor; StreamingLLM arXiv 2309.17453).
    AttentionSink { n_sinks: usize },
    /// Reuse of one index selection across heads or layers (DeepSeek V3.2's indexer is MQA-shared;
    /// cross-layer sharing is PROVISIONAL).
    IndexShare { share_group: usize, source_layer: usize },
    /// Expert routing (config: n_routed_experts, num_experts_per_tok, scoring_func, norm_topk_prob,
    /// n_group / topk_group for DeepSeek V3 group-limited routing).
    Router {
        n_experts: usize,
        top_k: usize,
        scoring: RouterScoring,
        norm_topk_prob: bool,
        n_groups: Option<usize>,
        group_topk: Option<usize>,
        bias: bool,
    },
    /// The selected experts' FFN (config: moe_intermediate_size). mixtral, qwen3-moe, gpt-oss, deepseek.
    Experts { n_experts: usize, top_k: usize, intermediate: usize, act: ActKind },
    /// Always-on shared expert (DeepSeek n_shared_experts; Qwen3-Next shared_expert_intermediate_size).
    SharedExpert { intermediate: usize, gated: bool },
    /// PROVISIONAL: experts factorized through a shared latent basis. No official config binding.
    LatentMoE { n_experts: usize, top_k: usize, latent_rank: usize },
    /// Combine selected expert outputs by router weights (config: routed_scaling_factor, DeepSeek V3).
    WeightedCombine { normalize: bool, routed_scaling_x1000: u32 },
    /// The FFN nonlinearity (config: hidden_act; gpt-oss swiglu_limit).
    Activation { kind: ActKind },
    /// Generic residual add.
    Residual { scale_x1000: u32 },
    /// PROVISIONAL: manifold/hyper-connections replacing the single residual stream
    /// (Hyper-Connections, arXiv 2409.19606). `expansion_rate` = n.
    HyperConnection { expansion_rate: usize, dynamic: bool },
    /// Multi-token prediction heads (DeepSeek V3 num_nextn_predict_layers).
    MTP { n_predict: usize, share_embed: bool },
    /// Token selection from logits. Sampling parameters are runtime-owned, not plan-owned.
    Sample { vocab: usize },
}

/// Multimodal ops. A separate type so a text-core `Vec<Op>` cannot contain one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionOp {
    /// Vision tower (Qwen2.5-VL vision_config: patch_size, depth, num_heads, hidden_size).
    VisionEncoder { patch: usize, hidden: usize, layers: usize, heads: usize },
    /// Vision-to-text merger/projector (Qwen2.5-VL spatial_merge_size, out_hidden_size).
    VisionProjector { in_dim: usize, out_dim: usize, merge_size: usize },
}

impl Op {
    pub fn family(&self) -> &'static str {
        match self {
            Op::Embed { .. } => "embed",
            Op::FinalProjection { .. } => "final_projection",
            Op::Norm { .. } => "norm",
            Op::Linear { .. } => "linear",
            Op::CompactLinear { .. } => "compact_linear",
            Op::RoPE { .. } => "rope",
            Op::MHA { .. } => "mha",
            Op::GQA { .. } => "gqa",
            Op::MLA { .. } => "mla",
            Op::SparseAttention { .. } => "sparse_attention",
            Op::CompressedSparseAttention { .. } => "compressed_sparse_attention",
            Op::HeavilyCompressedAttention { .. } => "heavily_compressed_attention",
            Op::MiniMaxSparseAttention { .. } => "minimax_sparse_attention",
            Op::DeltaNet { .. } => "deltanet",
            Op::KimiDeltaAttention { .. } => "kimi_delta_attention",
            Op::AttentionResidual { .. } => "attention_residual",
            Op::AttentionSink { .. } => "attention_sink",
            Op::IndexShare { .. } => "index_share",
            Op::Router { .. } => "router",
            Op::Experts { .. } => "experts",
            Op::SharedExpert { .. } => "shared_expert",
            Op::LatentMoE { .. } => "latent_moe",
            Op::WeightedCombine { .. } => "weighted_combine",
            Op::Activation { .. } => "activation",
            Op::Residual { .. } => "residual",
            Op::HyperConnection { .. } => "hyper_connection",
            Op::MTP { .. } => "mtp",
            Op::Sample { .. } => "sample",
        }
    }

    pub fn capability(&self) -> Capability {
        match self {
            Op::Router { .. } | Op::Experts { .. } | Op::SharedExpert { .. } | Op::LatentMoE { .. } | Op::WeightedCombine { .. } => Capability::Moe,
            _ => Capability::Core,
        }
    }

    /// True for families with no official config-field binding yet.
    pub fn provisional(&self) -> bool {
        matches!(
            self,
            Op::HeavilyCompressedAttention { .. } | Op::KimiDeltaAttention { .. } | Op::LatentMoE { .. } | Op::HyperConnection { .. }
        )
    }

    pub fn inputs(&self) -> &'static [Slot] {
        use Slot::*;
        match self {
            Op::Embed { .. } => &[Tokens],
            Op::FinalProjection { .. } => &[Normed],
            Op::Norm { .. } => &[Hidden],
            Op::Linear { .. } | Op::CompactLinear { .. } => &[Normed],
            Op::RoPE { .. } => &[Q, K],
            Op::MHA { .. } | Op::GQA { .. } | Op::SparseAttention { .. } | Op::CompressedSparseAttention { .. } | Op::MiniMaxSparseAttention { .. } | Op::DeltaNet { .. } | Op::KimiDeltaAttention { .. } => &[Q, K, V],
            Op::MLA { .. } | Op::HeavilyCompressedAttention { .. } => &[Q, KvLatent],
            Op::AttentionSink { .. } => &[AttnOut],
            Op::IndexShare { .. } => &[Index],
            Op::AttentionResidual { .. } => &[Hidden, AttnOut],
            Op::Router { .. } => &[Normed],
            Op::Experts { .. } | Op::LatentMoE { .. } => &[Normed, RouterWeights],
            Op::SharedExpert { .. } => &[Normed],
            Op::WeightedCombine { .. } => &[ExpertOut, RouterWeights],
            Op::Activation { .. } => &[Normed],
            Op::Residual { .. } | Op::HyperConnection { .. } => &[Hidden],
            Op::MTP { .. } => &[Hidden],
            Op::Sample { .. } => &[Logits],
        }
    }

    pub fn outputs(&self) -> &'static [Slot] {
        use Slot::*;
        match self {
            Op::Embed { .. } => &[Hidden],
            Op::FinalProjection { .. } => &[Logits],
            Op::Norm { .. } => &[Normed],
            Op::Linear { .. } | Op::CompactLinear { .. } => &[Normed],
            Op::RoPE { .. } => &[Q, K],
            Op::MHA { .. } | Op::GQA { .. } | Op::MLA { .. } | Op::HeavilyCompressedAttention { .. } | Op::MiniMaxSparseAttention { .. } | Op::DeltaNet { .. } | Op::KimiDeltaAttention { .. } | Op::AttentionSink { .. } => &[AttnOut],
            Op::SparseAttention { .. } | Op::CompressedSparseAttention { .. } => &[AttnOut, Index],
            Op::IndexShare { .. } => &[Index],
            Op::AttentionResidual { .. } | Op::Residual { .. } | Op::HyperConnection { .. } | Op::WeightedCombine { .. } => &[Hidden],
            Op::Router { .. } => &[RouterWeights],
            Op::Experts { .. } | Op::LatentMoE { .. } | Op::SharedExpert { .. } => &[ExpertOut],
            Op::Activation { .. } => &[Normed],
            Op::MTP { .. } => &[Logits],
            Op::Sample { .. } => &[Sampled],
        }
    }

    /// The tensor roles this op requires the source to declare.
    pub fn roles(&self) -> Vec<TensorRole> {
        use TensorRole::*;
        match self {
            Op::Embed { .. } => vec![TokenEmbd],
            Op::FinalProjection { tied, .. } => {
                if *tied {
                    vec![TokenEmbd]
                } else {
                    vec![OutputHead]
                }
            }
            Op::Norm { .. } => vec![NormWeight],
            Op::Linear { role, .. } | Op::CompactLinear { role, .. } => vec![*role],
            Op::RoPE { .. } | Op::Residual { .. } | Op::AttentionResidual { .. } | Op::Activation { .. } | Op::Sample { .. } | Op::WeightedCombine { .. } => vec![],
            Op::MHA { .. } | Op::GQA { .. } | Op::MiniMaxSparseAttention { .. } => vec![AttnQ, AttnK, AttnV, AttnO],
            Op::MLA { q_lora_rank, .. } => {
                let mut r = vec![KvADown, KvBUp, AttnO];
                if q_lora_rank.is_some() {
                    r.push(QDown);
                    r.push(QUp);
                } else {
                    r.push(AttnQ);
                }
                r
            }
            Op::HeavilyCompressedAttention { .. } => vec![KvADown, KvBUp, AttnQ, AttnO],
            Op::SparseAttention { .. } | Op::CompressedSparseAttention { .. } => vec![AttnQ, AttnK, AttnV, AttnO, Indexer],
            Op::DeltaNet { .. } => vec![AttnQ, AttnK, AttnV, AttnO, ShortConv, DeltaGate],
            Op::KimiDeltaAttention { .. } => vec![AttnQ, AttnK, AttnV, AttnO, ShortConv, DeltaGate],
            Op::AttentionSink { .. } => vec![AttnSink],
            Op::IndexShare { .. } => vec![Indexer],
            Op::Router { bias, .. } => {
                if *bias {
                    vec![Router, RouterBias]
                } else {
                    vec![Router]
                }
            }
            Op::Experts { .. } => vec![ExpertGate, ExpertUp, ExpertDown],
            Op::SharedExpert { .. } => vec![SharedExpert],
            Op::LatentMoE { .. } => vec![Router, LatentBasis],
            Op::HyperConnection { .. } => vec![HyperConnection],
            Op::MTP { .. } => vec![MtpHead],
        }
    }

    /// Shape relations that must hold for the op to be legal against the model dims.
    pub fn check_shapes(&self, d: &Dims) -> Result<()> {
        let want = |ok: bool, why: &str| if ok { Ok(()) } else { Err(err(format!("{}: {why}", self.family()))) };
        match self {
            Op::Embed { vocab, hidden } => want(*vocab == d.vocab && *hidden == d.hidden, "embed shape != model dims"),
            Op::FinalProjection { vocab, hidden, .. } => want(*vocab == d.vocab && *hidden == d.hidden, "lm_head shape != model dims"),
            Op::Norm { n, eps_x1e9, .. } => want(*n > 0 && *n <= d.hidden && *eps_x1e9 > 0, "norm width must be in 1..=hidden with eps > 0"),
            Op::Linear { in_features, out_features, .. } => want(*in_features > 0 && *out_features > 0, "linear dims must be non-zero"),
            Op::CompactLinear { in_features, out_features, block, .. } => want(
                *block > 0 && *in_features > 0 && *out_features > 0 && in_features % block == 0,
                "compact linear in_features must be a whole number of blocks",
            ),
            Op::RoPE { n_heads, head_dim, base_x1000, .. } => want(*n_heads > 0 && *head_dim % 2 == 0 && *base_x1000 > 0, "rope needs even head_dim and a positive base"),
            Op::MHA { n_heads, head_dim } => want(n_heads * head_dim == d.hidden, "n_heads*head_dim != hidden"),
            Op::GQA { n_heads, n_kv_heads, head_dim, .. } => want(
                *n_kv_heads > 0 && n_heads % n_kv_heads == 0 && n_heads * head_dim == d.hidden,
                "n_heads must be a multiple of n_kv_heads and n_heads*head_dim == hidden",
            ),
            Op::MLA { n_heads, kv_lora_rank, q_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim } => want(
                *n_heads > 0
                    && *kv_lora_rank > 0
                    && *kv_lora_rank < d.hidden
                    && qk_nope_head_dim + qk_rope_head_dim > 0
                    && *v_head_dim > 0
                    && q_lora_rank.map(|r| r > 0 && r < d.hidden).unwrap_or(true),
                "kv/q lora ranks must compress (0 < rank < hidden) and qk/v head dims be non-zero",
            ),
            Op::SparseAttention { n_heads, head_dim, index_topk, indexer_heads, indexer_dim } => want(
                n_heads * head_dim == d.hidden && *index_topk > 0 && *indexer_heads > 0 && *indexer_dim > 0,
                "indexer must select a positive top-k and n_heads*head_dim == hidden",
            ),
            Op::CompressedSparseAttention { n_heads, head_dim, block, compress_stride, select_blocks, window } => want(
                n_heads * head_dim == d.hidden && *block > 0 && *compress_stride > 0 && compress_stride <= block && *select_blocks > 0 && *window > 0,
                "compress stride must fit the block and every branch must be non-empty",
            ),
            Op::HeavilyCompressedAttention { n_heads, head_dim, kv_rank } => want(
                n_heads * head_dim == d.hidden && *kv_rank > 0 && kv_rank * 8 <= d.hidden,
                "heavy compression means kv_rank <= hidden/8",
            ),
            Op::MiniMaxSparseAttention { n_heads, head_dim, block, window } => want(n_heads * head_dim == d.hidden && *block > 0 && *window > 0, "block and window must be non-zero"),
            Op::DeltaNet { n_heads, head_dim, expand_v, conv_kernel } => want(
                *n_heads > 0 && *head_dim > 0 && *expand_v >= 1 && *conv_kernel > 0 && n_heads * head_dim * expand_v >= d.hidden,
                "value expansion must cover hidden and conv kernel be non-zero",
            ),
            Op::KimiDeltaAttention { n_heads, head_dim, conv_kernel, short_conv } => want(
                *n_heads > 0 && *head_dim > 0 && *conv_kernel > 0 && *short_conv > 0,
                "kda heads and both conv widths must be non-zero",
            ),
            Op::AttentionResidual { scale_x1000 } | Op::Residual { scale_x1000 } => want(*scale_x1000 > 0, "residual scale must be positive"),
            Op::AttentionSink { n_sinks } => want(*n_sinks > 0, "sinks must be non-empty"),
            Op::IndexShare { share_group, source_layer } => want(*share_group > 0 && *source_layer < d.n_layers, "share group must be non-zero and source layer in range"),
            Op::Router { n_experts, top_k, n_groups, group_topk, .. } => {
                let groups_ok = match (n_groups, group_topk) {
                    (Some(g), Some(t)) => *g > 0 && n_experts % g == 0 && *t > 0 && t <= g,
                    (None, None) => true,
                    _ => false,
                };
                want(*top_k > 0 && top_k <= n_experts && groups_ok, "top_k must be in 1..=n_experts and group routing consistent")
            }
            Op::Experts { n_experts, top_k, intermediate, .. } => want(*top_k > 0 && top_k <= n_experts && *intermediate > 0, "top_k must be in 1..=n_experts with a non-zero intermediate"),
            Op::SharedExpert { intermediate, .. } => want(*intermediate > 0, "shared expert intermediate must be non-zero"),
            Op::LatentMoE { n_experts, top_k, latent_rank } => want(*top_k > 0 && top_k <= n_experts && *latent_rank > 0 && *latent_rank < d.hidden, "latent rank must compress and top_k fit n_experts"),
            Op::WeightedCombine { routed_scaling_x1000, .. } => want(*routed_scaling_x1000 > 0, "routed scaling must be positive"),
            Op::Activation { kind } => match kind {
                ActKind::SwigluClamped { limit_x1000, alpha_x1000 } => want(*limit_x1000 > 0 && *alpha_x1000 > 0, "clamped swiglu needs a positive limit and alpha"),
                _ => Ok(()),
            },
            Op::HyperConnection { expansion_rate, .. } => want(*expansion_rate >= 1, "hyper-connection expansion rate must be >= 1"),
            Op::MTP { n_predict, .. } => want(*n_predict >= 1, "mtp must predict at least one extra token"),
            Op::Sample { vocab } => want(*vocab == d.vocab, "sample vocab != model vocab"),
        }
    }
}

impl VisionOp {
    pub fn family(&self) -> &'static str {
        match self {
            VisionOp::VisionEncoder { .. } => "vision_encoder",
            VisionOp::VisionProjector { .. } => "vision_projector",
        }
    }
    pub fn capability(&self) -> Capability {
        Capability::Multimodal
    }
    pub fn inputs(&self) -> &'static [Slot] {
        match self {
            VisionOp::VisionEncoder { .. } => &[Slot::Pixels],
            VisionOp::VisionProjector { .. } => &[Slot::VisionEmbed],
        }
    }
    pub fn outputs(&self) -> &'static [Slot] {
        match self {
            VisionOp::VisionEncoder { .. } => &[Slot::VisionEmbed],
            VisionOp::VisionProjector { .. } => &[Slot::Hidden],
        }
    }
    pub fn roles(&self) -> Vec<TensorRole> {
        match self {
            VisionOp::VisionEncoder { .. } => vec![TensorRole::VisionEncoder],
            VisionOp::VisionProjector { .. } => vec![TensorRole::VisionProjector],
        }
    }
    pub fn check_shapes(&self, d: &Dims) -> Result<()> {
        match self {
            VisionOp::VisionEncoder { patch, hidden, layers, heads } => {
                if *patch > 0 && *hidden > 0 && *layers > 0 && *heads > 0 && hidden % heads == 0 {
                    Ok(())
                } else {
                    Err(err("vision_encoder: dims must be non-zero and hidden divisible by heads".into()))
                }
            }
            VisionOp::VisionProjector { in_dim, out_dim, merge_size } => {
                if *in_dim > 0 && *merge_size > 0 && *out_dim == d.hidden {
                    Ok(())
                } else {
                    Err(err("vision_projector: must project into the text hidden size".into()))
                }
            }
        }
    }
}

/// An ordered plan an adapter emits. Sealed into launch packets as JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationPlan {
    pub arch: String,
    pub profile: PlanProfile,
    pub dims: Dims,
    /// Tensor roles the source declares. Every role an op requires must appear here.
    pub declared_roles: BTreeSet<TensorRole>,
    pub ops: Vec<Op>,
    /// Multimodal ops. Must be empty unless `profile` is `TextVision`.
    #[serde(default)]
    pub vision: Vec<VisionOp>,
}

impl OperationPlan {
    pub fn new(arch: &str, profile: PlanProfile, dims: Dims, ops: Vec<Op>) -> Self {
        OperationPlan { arch: arch.into(), profile, dims, declared_roles: BTreeSet::new(), ops, vision: Vec::new() }
    }

    /// Declare the tensor roles the source provides.
    pub fn with_roles(mut self, roles: impl IntoIterator<Item = TensorRole>) -> Self {
        self.declared_roles.extend(roles);
        self
    }

    /// Declare the roles every op in the plan requires (convenience for adapters that trust their own
    /// tensor map). Validation of undeclared roles then only catches vision/text mixing.
    pub fn declaring_own_roles(mut self) -> Self {
        let roles: Vec<TensorRole> = self.ops.iter().flat_map(|o| o.roles()).chain(self.vision.iter().flat_map(|o| o.roles())).collect();
        self.declared_roles.extend(roles);
        self
    }

    pub fn with_vision(mut self, vision: Vec<VisionOp>) -> Self {
        self.vision = vision;
        self
    }

    /// Every op family in the plan, in order (text core then vision).
    pub fn families(&self) -> Vec<&'static str> {
        self.ops.iter().map(|o| o.family()).chain(self.vision.iter().map(|o| o.family())).collect()
    }

    /// Legality: profile admits every op, shape relations hold, every referenced tensor role is
    /// declared, MoE order is coherent, and the text-core/multimodal boundary is respected.
    pub fn validate(&self) -> Result<()> {
        if self.ops.is_empty() {
            return Err(err("plan is empty".into()));
        }
        if !matches!(self.ops.first(), Some(Op::Embed { .. })) {
            return Err(err("plan must start with embed".into()));
        }
        if !matches!(self.ops.last(), Some(Op::Sample { .. })) {
            return Err(err("plan must end with sample".into()));
        }

        // multimodal boundary, both directions
        match (self.profile, self.vision.is_empty()) {
            (PlanProfile::TextVision, true) => return Err(err("text_vision profile declares no vision ops".into())),
            (p, false) if p != PlanProfile::TextVision => {
                return Err(err(format!("{} vision ops in a text-core plan ({:?})", self.vision.len(), p)))
            }
            _ => {}
        }

        let mut last_router_top_k: Option<usize> = None;
        for op in &self.ops {
            if !self.profile.allows(op.capability()) {
                return Err(err(format!("op {} needs {:?}, not allowed by profile {:?}", op.family(), op.capability(), self.profile)));
            }
            op.check_shapes(&self.dims)?;
            for r in op.roles() {
                if !self.declared_roles.contains(&r) {
                    return Err(err(format!("op {} references undeclared tensor role {:?}", op.family(), r)));
                }
            }
            match op {
                Op::Router { top_k, .. } => last_router_top_k = Some(*top_k),
                Op::Experts { top_k, .. } | Op::LatentMoE { top_k, .. } => match last_router_top_k {
                    None => return Err(err(format!("{} without a preceding router", op.family()))),
                    Some(k) if k != *top_k => return Err(err(format!("{} top_k {top_k} != router top_k {k}", op.family()))),
                    _ => {}
                },
                _ => {}
            }
        }

        for op in &self.vision {
            if !self.profile.allows(op.capability()) {
                return Err(err(format!("op {} needs multimodal capability", op.family())));
            }
            op.check_shapes(&self.dims)?;
            for r in op.roles() {
                if !self.declared_roles.contains(&r) {
                    return Err(err(format!("op {} references undeclared tensor role {:?}", op.family(), r)));
                }
            }
        }
        Ok(())
    }

    /// Families with no official config binding yet. A launch packet must surface these.
    pub fn provisional_families(&self) -> Vec<&'static str> {
        self.ops.iter().filter(|o| o.provisional()).map(|o| o.family()).collect()
    }

    pub fn summary(&self) -> String {
        format!(
            "{} [{:?}] {} ops, {} vision, {} roles, {} layers, hidden {}, provisional: {}",
            self.arch,
            self.profile,
            self.ops.len(),
            self.vision.len(),
            self.declared_roles.len(),
            self.dims.n_layers,
            self.dims.hidden,
            if self.provisional_families().is_empty() { "none".to_string() } else { self.provisional_families().join(",") }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims() -> Dims {
        Dims { hidden: 2048, vocab: 151936, n_layers: 4 }
    }

    /// A Qwen3-style MoE layer: rms-norm, GQA with qk-norm, softmax router (top-8 of 128), experts,
    /// weighted combine.
    fn qwen3_moe_ops() -> Vec<Op> {
        vec![
            Op::Embed { vocab: 151936, hidden: 2048 },
            Op::Norm { kind: NormKind::RmsNorm, n: 2048, eps_x1e9: 1_000_000 },
            Op::Linear { role: TensorRole::AttnQ, in_features: 2048, out_features: 2048, bias: false },
            Op::Linear { role: TensorRole::AttnK, in_features: 2048, out_features: 512, bias: false },
            Op::Linear { role: TensorRole::AttnV, in_features: 2048, out_features: 512, bias: false },
            Op::Norm { kind: NormKind::RmsNorm, n: 128, eps_x1e9: 1_000_000 },
            Op::RoPE { n_heads: 16, head_dim: 128, base_x1000: 10_000_000, scaling: RopeScaling::None },
            Op::GQA { n_heads: 16, n_kv_heads: 4, head_dim: 128, sliding_window: None },
            Op::Linear { role: TensorRole::AttnO, in_features: 2048, out_features: 2048, bias: false },
            Op::AttentionResidual { scale_x1000: 1000 },
            Op::Norm { kind: NormKind::RmsNorm, n: 2048, eps_x1e9: 1_000_000 },
            Op::Router { n_experts: 128, top_k: 8, scoring: RouterScoring::Softmax, norm_topk_prob: true, n_groups: None, group_topk: None, bias: false },
            Op::Experts { n_experts: 128, top_k: 8, intermediate: 768, act: ActKind::SiluGated },
            Op::WeightedCombine { normalize: true, routed_scaling_x1000: 1000 },
            Op::Residual { scale_x1000: 1000 },
            Op::Norm { kind: NormKind::RmsNorm, n: 2048, eps_x1e9: 1_000_000 },
            Op::FinalProjection { vocab: 151936, hidden: 2048, tied: false },
            Op::Sample { vocab: 151936 },
        ]
    }

    fn qwen3_moe_plan() -> OperationPlan {
        OperationPlan::new("qwen3-moe", PlanProfile::TextCoreMoe, dims(), qwen3_moe_ops()).declaring_own_roles()
    }

    #[test]
    fn legal_qwen3_moe_plan_validates() {
        let p = qwen3_moe_plan();
        p.validate().unwrap();
        assert_eq!(p.ops.len(), 18);
        assert!(p.provisional_families().is_empty());
        assert!(p.families().contains(&"router"));
    }

    #[test]
    fn unknown_op_for_profile_is_rejected() {
        let mut p = qwen3_moe_plan();
        p.profile = PlanProfile::TextCoreDense;
        let e = p.validate().unwrap_err().to_string();
        assert!(e.contains("router") && e.contains("Moe"), "{e}");
    }

    #[test]
    fn shape_relation_violation_is_rejected() {
        let mut ops = qwen3_moe_ops();
        // n_heads*head_dim (16*64) != hidden (2048)
        ops[7] = Op::GQA { n_heads: 16, n_kv_heads: 4, head_dim: 64, sliding_window: None };
        let p = OperationPlan::new("qwen3-moe", PlanProfile::TextCoreMoe, dims(), ops).declaring_own_roles();
        assert!(p.validate().unwrap_err().to_string().contains("gqa"));
    }

    #[test]
    fn mla_rank_must_compress() {
        let d = dims();
        let bad = Op::MLA { n_heads: 16, kv_lora_rank: 4096, q_lora_rank: None, qk_nope_head_dim: 128, qk_rope_head_dim: 64, v_head_dim: 128 };
        assert!(bad.check_shapes(&d).is_err());
        let good = Op::MLA { n_heads: 16, kv_lora_rank: 512, q_lora_rank: Some(1536), qk_nope_head_dim: 128, qk_rope_head_dim: 64, v_head_dim: 128 };
        good.check_shapes(&d).unwrap();
        assert!(good.roles().contains(&TensorRole::QDown));
    }

    #[test]
    fn undeclared_tensor_role_is_rejected() {
        let p = OperationPlan::new("qwen3-moe", PlanProfile::TextCoreMoe, dims(), qwen3_moe_ops())
            .with_roles([TensorRole::TokenEmbd, TensorRole::NormWeight]);
        assert!(p.validate().unwrap_err().to_string().contains("undeclared tensor role"));
    }

    #[test]
    fn text_core_cannot_carry_vision() {
        let p = qwen3_moe_plan().with_vision(vec![VisionOp::VisionProjector { in_dim: 1280, out_dim: 2048, merge_size: 2 }]);
        assert!(p.validate().unwrap_err().to_string().contains("text-core"));
        // and the type-level half: Vec<Op> has no vision variant, so ops can never hold one.
        assert!(!p.families()[..p.ops.len()].contains(&"vision_projector"));
    }

    #[test]
    fn multimodal_profile_requires_vision_ops() {
        let mut p = qwen3_moe_plan();
        p.profile = PlanProfile::TextVision;
        assert!(p.validate().unwrap_err().to_string().contains("no vision ops"));

        let ok = p
            .clone()
            .with_vision(vec![
                VisionOp::VisionEncoder { patch: 14, hidden: 1280, layers: 32, heads: 16 },
                VisionOp::VisionProjector { in_dim: 1280, out_dim: 2048, merge_size: 2 },
            ])
            .declaring_own_roles();
        ok.validate().unwrap();
        // a projector that does not land in the text hidden size is illegal
        let bad = p.with_vision(vec![VisionOp::VisionProjector { in_dim: 1280, out_dim: 999, merge_size: 2 }]).declaring_own_roles();
        assert!(bad.validate().is_err());
    }

    #[test]
    fn experts_must_follow_a_matching_router() {
        let mut ops = qwen3_moe_ops();
        ops.remove(11); // drop the router
        let p = OperationPlan::new("x", PlanProfile::TextCoreMoe, dims(), ops).declaring_own_roles();
        assert!(p.validate().unwrap_err().to_string().contains("without a preceding router"));

        let mut ops = qwen3_moe_ops();
        ops[12] = Op::Experts { n_experts: 128, top_k: 4, intermediate: 768, act: ActKind::SiluGated };
        let p = OperationPlan::new("x", PlanProfile::TextCoreMoe, dims(), ops).declaring_own_roles();
        assert!(p.validate().unwrap_err().to_string().contains("router top_k"));
    }

    #[test]
    fn provisional_families_are_surfaced() {
        let op = Op::KimiDeltaAttention { n_heads: 16, head_dim: 128, conv_kernel: 4, short_conv: 4 };
        assert!(op.provisional());
        let mut ops = qwen3_moe_ops();
        ops.insert(8, op);
        let p = OperationPlan::new("kimi-provisional", PlanProfile::TextCoreMoe, dims(), ops).declaring_own_roles();
        p.validate().unwrap();
        assert_eq!(p.provisional_families(), vec!["kimi_delta_attention"]);
    }

    #[test]
    fn plan_round_trips_as_json_for_a_launch_packet() {
        let p = qwen3_moe_plan();
        let json = serde_json::to_string(&p).unwrap();
        let back: OperationPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
        back.validate().unwrap();
        assert!(json.contains("\"router\""));
    }

    #[test]
    fn every_declared_family_is_reachable() {
        // 28 text-core families + 2 vision = the 30 the IR is allowed to have.
        let all: Vec<&'static str> = vec![
            Op::Embed { vocab: 1, hidden: 1 }.family(),
            Op::FinalProjection { vocab: 1, hidden: 1, tied: true }.family(),
            Op::Norm { kind: NormKind::LayerNorm, n: 1, eps_x1e9: 1 }.family(),
            Op::Linear { role: TensorRole::LinearWeight, in_features: 1, out_features: 1, bias: false }.family(),
            Op::CompactLinear { role: TensorRole::CompactWeight, in_features: 32, out_features: 1, block: 32, format: CompactFormat::Mxfp4 }.family(),
            Op::RoPE { n_heads: 1, head_dim: 2, base_x1000: 1, scaling: RopeScaling::Linear { factor_x1000: 4000 } }.family(),
            Op::MHA { n_heads: 1, head_dim: 1 }.family(),
            Op::GQA { n_heads: 1, n_kv_heads: 1, head_dim: 1, sliding_window: Some(128) }.family(),
            Op::MLA { n_heads: 1, kv_lora_rank: 1, q_lora_rank: None, qk_nope_head_dim: 1, qk_rope_head_dim: 1, v_head_dim: 1 }.family(),
            Op::SparseAttention { n_heads: 1, head_dim: 1, index_topk: 1, indexer_heads: 1, indexer_dim: 1 }.family(),
            Op::CompressedSparseAttention { n_heads: 1, head_dim: 1, block: 1, compress_stride: 1, select_blocks: 1, window: 1 }.family(),
            Op::HeavilyCompressedAttention { n_heads: 1, head_dim: 1, kv_rank: 1 }.family(),
            Op::MiniMaxSparseAttention { n_heads: 1, head_dim: 1, block: 1, window: 1 }.family(),
            Op::DeltaNet { n_heads: 1, head_dim: 1, expand_v: 1, conv_kernel: 1 }.family(),
            Op::KimiDeltaAttention { n_heads: 1, head_dim: 1, conv_kernel: 1, short_conv: 1 }.family(),
            Op::AttentionResidual { scale_x1000: 1000 }.family(),
            Op::AttentionSink { n_sinks: 1 }.family(),
            Op::IndexShare { share_group: 1, source_layer: 0 }.family(),
            Op::Router { n_experts: 1, top_k: 1, scoring: RouterScoring::Sigmoid, norm_topk_prob: false, n_groups: None, group_topk: None, bias: true }.family(),
            Op::Experts { n_experts: 1, top_k: 1, intermediate: 1, act: ActKind::GeluGated }.family(),
            Op::SharedExpert { intermediate: 1, gated: true }.family(),
            Op::LatentMoE { n_experts: 1, top_k: 1, latent_rank: 1 }.family(),
            Op::WeightedCombine { normalize: true, routed_scaling_x1000: 1 }.family(),
            Op::Activation { kind: ActKind::SwigluClamped { limit_x1000: 7000, alpha_x1000: 1702 } }.family(),
            Op::Residual { scale_x1000: 1000 }.family(),
            Op::HyperConnection { expansion_rate: 4, dynamic: true }.family(),
            Op::MTP { n_predict: 1, share_embed: true }.family(),
            Op::Sample { vocab: 1 }.family(),
            VisionOp::VisionEncoder { patch: 1, hidden: 1, layers: 1, heads: 1 }.family(),
            VisionOp::VisionProjector { in_dim: 1, out_dim: 1, merge_size: 1 }.family(),
        ];
        assert_eq!(all.len(), 30);
        let unique: BTreeSet<&'static str> = all.iter().copied().collect();
        assert_eq!(unique.len(), 30);
    }
}
