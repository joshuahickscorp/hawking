//! Compact, source-bound tensor anchors for every DeepSeek-V4-Flash base layer.
//!
//! The sealed stream contains nearly seventy thousand tensor descriptors.  A
//! future decoder needs exact names, physical shapes, native pair geometry,
//! gate data, mHC controls, and compression controls for all forty-three base
//! layers, but retaining a second `String` for each descriptor would be both
//! wasteful and easy to desynchronise from the artifact.  This module instead
//! stores one small descriptor per layer and derives names on demand.
//!
//! [`verify_deepseek_v4_layer_source_anchors`] performs a fail-closed,
//! metadata-only walk through every base-layer binding supplied by an already
//! admitted [`DeepSeekV4FullStreamReader`].  It never reads a tensor payload,
//! uploads a buffer, allocates device state, performs a forward pass, or
//! exposes an engine/endpoint/throughput claim.  Content-byte verification
//! remains the reader's explicit `verify_tensor` / `verify_all_chunks`
//! responsibility.

use serde_json::Value;

use crate::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, DeepSeekV4TensorMetadata, NativeScalePairKind,
    PINNED_REPOSITORY, PINNED_REVISION,
};
use crate::{Error, Result};

/// Number of decoder layers in the source child body.  The separate MTP
/// auxiliary layer is intentionally excluded from this anchor surface.
pub const DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT: usize = 43;
/// Number of source-routed experts in every base MoE layer.
pub const DSV4F_LAYER_SOURCE_ANCHOR_ROUTED_EXPERT_COUNT: usize = 256;
/// Number of routed experts selected by the source gate per token.
pub const DSV4F_LAYER_SOURCE_ANCHOR_TOP_K: usize = 6;
/// Hidden width used by all base-layer state, norm, and gate bindings.
pub const DSV4F_LAYER_SOURCE_ANCHOR_HIDDEN_SIZE: usize = 4_096;

const VOCAB_SIZE: usize = 129_280;
const HC_MULT: usize = 4;
const HC_SINKHORN_ITERS: usize = 20;
const HASH_LAYER_COUNT: usize = 3;
const EXPECTED_TOTAL_TENSOR_COUNT: usize = 69_187;
const EXPECTED_BASE_TENSOR_COUNT: usize = 67_612;
const EXPECTED_EXCLUDED_MTP_TENSOR_COUNT: usize = 1_575;
const EXPECTED_TOTAL_TENSOR_BYTES: u64 = 159_609_485_896;
const EXPECTED_FP8_PAIR_COUNT: usize = 375;
const EXPECTED_FP4_PAIR_COUNT: usize = 33_792;
const METADATA_READ_BOUND_BYTES: usize = 64 * 1024;

const OFFICIAL_INFERENCE_MODEL_PY_SHA256: &str =
    "ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71";
const OFFICIAL_INFERENCE_KERNEL_PY_SHA256: &str =
    "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2";
const OFFICIAL_INFERENCE_CONFIG_JSON_SHA256: &str =
    "6cc6f816ca73a8d38750194e330398e4f6955b4b45f674f7d29c96da14ccb733";
const OFFICIAL_MODEL_CONFIG_JSON_SHA256: &str =
    "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818";

const SHAPE_F32_64: &[u64] = &[64];
const SHAPE_BF16_512: &[u64] = &[512];
const SHAPE_BF16_1024: &[u64] = &[1_024];
const SHAPE_BF16_4096: &[u64] = &[4_096];
const SHAPE_F32_24: &[u64] = &[24];
const SHAPE_F32_3: &[u64] = &[3];
const SHAPE_F32_256: &[u64] = &[256];
const SHAPE_BF16_GATE: &[u64] = &[256, 4_096];
const SHAPE_I64_TID2EID: &[u64] = &[129_280, 6];
const SHAPE_F32_HC_FN: &[u64] = &[24, 16_384];
const SHAPE_F32_RATIO4_APE: &[u64] = &[4, 1_024];
const SHAPE_BF16_RATIO4_COMPRESSOR_WEIGHT: &[u64] = &[1_024, 4_096];
const SHAPE_F32_RATIO128_APE: &[u64] = &[128, 512];
const SHAPE_BF16_RATIO128_COMPRESSOR_WEIGHT: &[u64] = &[512, 4_096];
const SHAPE_BF16_COMPRESSOR_NORM: &[u64] = &[512];
const SHAPE_F32_INDEXER_APE: &[u64] = &[4, 256];
const SHAPE_BF16_INDEXER_COMPRESSOR_WEIGHT: &[u64] = &[256, 4_096];
const SHAPE_BF16_INDEXER_NORM: &[u64] = &[128];
const SHAPE_BF16_INDEXER_WEIGHTS_PROJ: &[u64] = &[64, 4_096];

const SHAPE_FP8_WQ_A_WEIGHT: &[u64] = &[1_024, 4_096];
const SHAPE_FP8_WQ_A_SCALE: &[u64] = &[8, 32];
const SHAPE_FP8_WQ_B_WEIGHT: &[u64] = &[32_768, 1_024];
const SHAPE_FP8_WQ_B_SCALE: &[u64] = &[256, 8];
const SHAPE_FP8_WKV_WEIGHT: &[u64] = &[512, 4_096];
const SHAPE_FP8_WKV_SCALE: &[u64] = &[4, 32];
const SHAPE_FP8_WO_A_WEIGHT: &[u64] = &[8_192, 4_096];
const SHAPE_FP8_WO_A_SCALE: &[u64] = &[64, 32];
const SHAPE_FP8_WO_B_WEIGHT: &[u64] = &[4_096, 8_192];
const SHAPE_FP8_WO_B_SCALE: &[u64] = &[32, 64];
const SHAPE_FP8_SHARED_W1_W3_WEIGHT: &[u64] = &[2_048, 4_096];
const SHAPE_FP8_SHARED_W1_W3_SCALE: &[u64] = &[16, 32];
const SHAPE_FP8_SHARED_W2_WEIGHT: &[u64] = &[4_096, 2_048];
const SHAPE_FP8_SHARED_W2_SCALE: &[u64] = &[32, 16];
const SHAPE_FP8_INDEXER_WQ_B_WEIGHT: &[u64] = &[8_192, 1_024];
const SHAPE_FP8_INDEXER_WQ_B_SCALE: &[u64] = &[64, 8];
const SHAPE_FP4_W1_W3_WEIGHT: &[u64] = &[2_048, 2_048];
const SHAPE_FP4_W1_W3_SCALE: &[u64] = &[2_048, 128];
const SHAPE_FP4_W2_WEIGHT: &[u64] = &[4_096, 1_024];
const SHAPE_FP4_W2_SCALE: &[u64] = &[4_096, 64];

#[derive(Debug, Clone, Copy)]
struct TensorExpectation {
    dtype: &'static str,
    shape: &'static [u64],
}

#[derive(Debug, Clone, Copy)]
struct NativePairExpectation {
    kind: NativeScalePairKind,
    weight: TensorExpectation,
    scale: TensorExpectation,
    out_rows: u64,
    packed_k: u64,
    logical_k: u64,
    scale_rows: u64,
    scale_cols: u64,
}

const FP8_WQ_A: NativePairExpectation = NativePairExpectation {
    kind: NativeScalePairKind::Fp8E4M3fn,
    weight: TensorExpectation {
        dtype: "F8_E4M3",
        shape: SHAPE_FP8_WQ_A_WEIGHT,
    },
    scale: TensorExpectation {
        dtype: "F8_E8M0",
        shape: SHAPE_FP8_WQ_A_SCALE,
    },
    out_rows: 1_024,
    packed_k: 4_096,
    logical_k: 4_096,
    scale_rows: 8,
    scale_cols: 32,
};
const FP8_WQ_B: NativePairExpectation = NativePairExpectation {
    kind: NativeScalePairKind::Fp8E4M3fn,
    weight: TensorExpectation {
        dtype: "F8_E4M3",
        shape: SHAPE_FP8_WQ_B_WEIGHT,
    },
    scale: TensorExpectation {
        dtype: "F8_E8M0",
        shape: SHAPE_FP8_WQ_B_SCALE,
    },
    out_rows: 32_768,
    packed_k: 1_024,
    logical_k: 1_024,
    scale_rows: 256,
    scale_cols: 8,
};
const FP8_WKV: NativePairExpectation = NativePairExpectation {
    kind: NativeScalePairKind::Fp8E4M3fn,
    weight: TensorExpectation {
        dtype: "F8_E4M3",
        shape: SHAPE_FP8_WKV_WEIGHT,
    },
    scale: TensorExpectation {
        dtype: "F8_E8M0",
        shape: SHAPE_FP8_WKV_SCALE,
    },
    out_rows: 512,
    packed_k: 4_096,
    logical_k: 4_096,
    scale_rows: 4,
    scale_cols: 32,
};
const FP8_WO_A: NativePairExpectation = NativePairExpectation {
    kind: NativeScalePairKind::Fp8E4M3fn,
    weight: TensorExpectation {
        dtype: "F8_E4M3",
        shape: SHAPE_FP8_WO_A_WEIGHT,
    },
    scale: TensorExpectation {
        dtype: "F8_E8M0",
        shape: SHAPE_FP8_WO_A_SCALE,
    },
    out_rows: 8_192,
    packed_k: 4_096,
    logical_k: 4_096,
    scale_rows: 64,
    scale_cols: 32,
};
const FP8_WO_B: NativePairExpectation = NativePairExpectation {
    kind: NativeScalePairKind::Fp8E4M3fn,
    weight: TensorExpectation {
        dtype: "F8_E4M3",
        shape: SHAPE_FP8_WO_B_WEIGHT,
    },
    scale: TensorExpectation {
        dtype: "F8_E8M0",
        shape: SHAPE_FP8_WO_B_SCALE,
    },
    out_rows: 4_096,
    packed_k: 8_192,
    logical_k: 8_192,
    scale_rows: 32,
    scale_cols: 64,
};
const FP8_SHARED_W1_W3: NativePairExpectation = NativePairExpectation {
    kind: NativeScalePairKind::Fp8E4M3fn,
    weight: TensorExpectation {
        dtype: "F8_E4M3",
        shape: SHAPE_FP8_SHARED_W1_W3_WEIGHT,
    },
    scale: TensorExpectation {
        dtype: "F8_E8M0",
        shape: SHAPE_FP8_SHARED_W1_W3_SCALE,
    },
    out_rows: 2_048,
    packed_k: 4_096,
    logical_k: 4_096,
    scale_rows: 16,
    scale_cols: 32,
};
const FP8_SHARED_W2: NativePairExpectation = NativePairExpectation {
    kind: NativeScalePairKind::Fp8E4M3fn,
    weight: TensorExpectation {
        dtype: "F8_E4M3",
        shape: SHAPE_FP8_SHARED_W2_WEIGHT,
    },
    scale: TensorExpectation {
        dtype: "F8_E8M0",
        shape: SHAPE_FP8_SHARED_W2_SCALE,
    },
    out_rows: 4_096,
    packed_k: 2_048,
    logical_k: 2_048,
    scale_rows: 32,
    scale_cols: 16,
};
const FP8_INDEXER_WQ_B: NativePairExpectation = NativePairExpectation {
    kind: NativeScalePairKind::Fp8E4M3fn,
    weight: TensorExpectation {
        dtype: "F8_E4M3",
        shape: SHAPE_FP8_INDEXER_WQ_B_WEIGHT,
    },
    scale: TensorExpectation {
        dtype: "F8_E8M0",
        shape: SHAPE_FP8_INDEXER_WQ_B_SCALE,
    },
    out_rows: 8_192,
    packed_k: 1_024,
    logical_k: 1_024,
    scale_rows: 64,
    scale_cols: 8,
};
const FP4_W1_W3: NativePairExpectation = NativePairExpectation {
    kind: NativeScalePairKind::Fp4E2M1fnX2,
    weight: TensorExpectation {
        dtype: "I8",
        shape: SHAPE_FP4_W1_W3_WEIGHT,
    },
    scale: TensorExpectation {
        dtype: "F8_E8M0",
        shape: SHAPE_FP4_W1_W3_SCALE,
    },
    out_rows: 2_048,
    packed_k: 2_048,
    logical_k: 4_096,
    scale_rows: 2_048,
    scale_cols: 128,
};
const FP4_W2: NativePairExpectation = NativePairExpectation {
    kind: NativeScalePairKind::Fp4E2M1fnX2,
    weight: TensorExpectation {
        dtype: "I8",
        shape: SHAPE_FP4_W2_WEIGHT,
    },
    scale: TensorExpectation {
        dtype: "F8_E8M0",
        shape: SHAPE_FP4_W2_SCALE,
    },
    out_rows: 4_096,
    packed_k: 1_024,
    logical_k: 2_048,
    scale_rows: 4_096,
    scale_cols: 64,
};

/// Immutable artifact and source identities used by the compact anchors.
/// These values are copied from the admitted reader; no stream data is held.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4LayerSourceIdentity {
    pub artifact_root: String,
    pub repository: String,
    pub revision: String,
    pub manifest_seal_sha256: String,
    pub manifest_file_sha256: String,
    pub restart_seal_sha256: String,
    pub inference_model_py_sha256: String,
    pub inference_kernel_py_sha256: String,
    pub inference_config_json_sha256: String,
    pub model_config_json_sha256: String,
}

/// Exact source geometry used to derive all compact per-layer descriptors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4LayerSourceGeometry {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub base_layer_count: usize,
    pub excluded_mtp_layer_count: usize,
    pub hash_layer_count: usize,
    pub routed_expert_count: usize,
    pub shared_expert_count: usize,
    pub activated_experts_per_token: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub attention_head_count: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub sliding_window_tokens: usize,
    /// Only the base-body entries are retained.  The omitted final source
    /// entry belongs to the excluded MTP auxiliary layer.
    pub base_compression_ratios: Vec<usize>,
}

/// The source gate transform bound by the authenticated `model.py` and
/// config assets.  The rational scale avoids turning an exact source binding
/// into an imprecise floating-point comparison in the descriptor API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV4GateAlgorithmBinding {
    pub score_transform: &'static str,
    pub selected_weights_use_pre_bias_scores: bool,
    pub non_softmax_weights_are_normalized: bool,
    pub route_scale_numerator: u8,
    pub route_scale_denominator: u8,
}

/// Source-defined attention/cache compression family for one base layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4LayerCompressionMode {
    SlidingWindowOnly,
    Ratio4WithIndexer,
    Ratio128,
}

impl DeepSeekV4LayerCompressionMode {
    pub const fn ratio(self) -> usize {
        match self {
            Self::SlidingWindowOnly => 0,
            Self::Ratio4WithIndexer => 4,
            Self::Ratio128 => 128,
        }
    }

    pub const fn has_attention_compressor(self) -> bool {
        !matches!(self, Self::SlidingWindowOnly)
    }

    pub const fn has_indexer(self) -> bool {
        matches!(self, Self::Ratio4WithIndexer)
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SlidingWindowOnly => "sliding_window_only",
            Self::Ratio4WithIndexer => "ratio_4_with_indexer",
            Self::Ratio128 => "ratio_128",
        }
    }
}

/// Exact gate data selection for one base layer.  Hash routing applies only
/// to the first three base layers; it still uses the common BF16 score matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4LayerGateMode {
    HashTokenIdToExpertIds,
    LearnedScoresWithSelectionBias,
}

impl DeepSeekV4LayerGateMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HashTokenIdToExpertIds => "hash_token_id_to_expert_ids",
            Self::LearnedScoresWithSelectionBias => "learned_scores_with_selection_bias",
        }
    }
}

/// FP8 attention-control projection represented by a native weight/scale
/// pair in every base layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4LayerControlProjection {
    WqA,
    WqB,
    Wkv,
    WoA,
    WoB,
}

impl DeepSeekV4LayerControlProjection {
    const ALL: [Self; 5] = [Self::WqA, Self::WqB, Self::Wkv, Self::WoA, Self::WoB];

    const fn suffix(self) -> &'static str {
        match self {
            Self::WqA => "attn.wq_a.weight",
            Self::WqB => "attn.wq_b.weight",
            Self::Wkv => "attn.wkv.weight",
            Self::WoA => "attn.wo_a.weight",
            Self::WoB => "attn.wo_b.weight",
        }
    }

    const fn expectation(self) -> NativePairExpectation {
        match self {
            Self::WqA => FP8_WQ_A,
            Self::WqB => FP8_WQ_B,
            Self::Wkv => FP8_WKV,
            Self::WoA => FP8_WO_A,
            Self::WoB => FP8_WO_B,
        }
    }
}

/// One source SwiGLU projection.  W1/W3 have the same source-native geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4LayerExpertProjection {
    W1,
    W2,
    W3,
}

impl DeepSeekV4LayerExpertProjection {
    const ALL: [Self; 3] = [Self::W1, Self::W2, Self::W3];

    const fn suffix(self) -> &'static str {
        match self {
            Self::W1 => "w1.weight",
            Self::W2 => "w2.weight",
            Self::W3 => "w3.weight",
        }
    }

    const fn fp8_shared_expectation(self) -> NativePairExpectation {
        match self {
            Self::W1 | Self::W3 => FP8_SHARED_W1_W3,
            Self::W2 => FP8_SHARED_W2,
        }
    }

    const fn fp4_routed_expectation(self) -> NativePairExpectation {
        match self {
            Self::W1 | Self::W3 => FP4_W1_W3,
            Self::W2 => FP4_W2,
        }
    }
}

/// One of the source hyper-connection control triplets in each layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4LayerMhcStage {
    Attention,
    FeedForward,
}

impl DeepSeekV4LayerMhcStage {
    const ALL: [Self; 2] = [Self::Attention, Self::FeedForward];

    const fn stem(self) -> &'static str {
        match self {
            Self::Attention => "hc_attn",
            Self::FeedForward => "hc_ffn",
        }
    }
}

/// Tensor family inside a source compressor binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4LayerCompressorTensor {
    Ape,
    Wkv,
    Wgate,
    Norm,
}

impl DeepSeekV4LayerCompressorTensor {
    const ALL: [Self; 4] = [Self::Ape, Self::Wkv, Self::Wgate, Self::Norm];

    const fn suffix(self) -> &'static str {
        match self {
            Self::Ape => "ape",
            Self::Wkv => "wkv.weight",
            Self::Wgate => "wgate.weight",
            Self::Norm => "norm.weight",
        }
    }
}

/// Compact, metadata-only anchor for a single non-pair tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4TensorSourceAnchor {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<u64>,
    pub bytes: u64,
}

/// Compact, metadata-only source-native weight/scale pair anchor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4NativePairSourceAnchor {
    pub kind: NativeScalePairKind,
    pub weight: DeepSeekV4TensorSourceAnchor,
    pub scale: DeepSeekV4TensorSourceAnchor,
    pub out_rows: u64,
    pub packed_k: u64,
    pub logical_k: u64,
    pub scale_rows: u64,
    pub scale_cols: u64,
}

/// Exact mHC control tensor names and shapes for one stage in one layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4MhcSourceBinding {
    pub stage: DeepSeekV4LayerMhcStage,
    pub fn_tensor: DeepSeekV4TensorSourceAnchor,
    pub base_tensor: DeepSeekV4TensorSourceAnchor,
    pub scale_tensor: DeepSeekV4TensorSourceAnchor,
}

/// Exact score-matrix plus layer-specific route-data binding for a source
/// gate.  For learned layers `route_data` is the selection-only F32 bias;
/// for hash layers it is the I64 token-id to expert-id table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4GateSourceBinding {
    pub mode: DeepSeekV4LayerGateMode,
    pub score_weight: DeepSeekV4TensorSourceAnchor,
    pub route_data: DeepSeekV4TensorSourceAnchor,
    pub activated_experts_per_token: usize,
    pub algorithm: DeepSeekV4GateAlgorithmBinding,
}

/// Source compression controls for one base layer.  `None` values are
/// intentional for ratio-zero layers and avoid materialising missing names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4CompressionSourceBinding {
    pub mode: DeepSeekV4LayerCompressionMode,
    pub attention_compressor: Option<[DeepSeekV4TensorSourceAnchor; 4]>,
    pub indexer_compressor: Option<[DeepSeekV4TensorSourceAnchor; 4]>,
    pub indexer_wq_b: Option<DeepSeekV4NativePairSourceAnchor>,
    pub indexer_weights_proj: Option<DeepSeekV4TensorSourceAnchor>,
}

/// One compact base-layer descriptor.  It stores no tensor names; all tensor
/// and pair names are derived from `layer` and a validated role on demand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4LayerSourceAnchor {
    pub layer: usize,
    pub compression: DeepSeekV4LayerCompressionMode,
    pub gate_mode: DeepSeekV4LayerGateMode,
    /// Includes scale tensors and all 256 × 3 routed expert native pairs.
    pub tensor_count: usize,
}

impl DeepSeekV4LayerSourceAnchor {
    /// Build the exact source binding for one common FP8 attention projection.
    pub fn control_pair(
        &self,
        projection: DeepSeekV4LayerControlProjection,
    ) -> DeepSeekV4NativePairSourceAnchor {
        native_pair_anchor(
            format!("layers.{}.{}", self.layer, projection.suffix()),
            projection.expectation(),
        )
    }

    /// Build the exact source binding for one FP8 shared-expert projection.
    pub fn shared_expert_pair(
        &self,
        projection: DeepSeekV4LayerExpertProjection,
    ) -> DeepSeekV4NativePairSourceAnchor {
        native_pair_anchor(
            format!(
                "layers.{}.ffn.shared_experts.{}",
                self.layer,
                projection.suffix()
            ),
            projection.fp8_shared_expectation(),
        )
    }

    /// Build the exact source binding for one FP4 routed-expert projection.
    /// The request is fail-closed for out-of-range expert identifiers.
    pub fn routed_expert_pair(
        &self,
        expert: usize,
        projection: DeepSeekV4LayerExpertProjection,
    ) -> Result<DeepSeekV4NativePairSourceAnchor> {
        if expert >= DSV4F_LAYER_SOURCE_ANCHOR_ROUTED_EXPERT_COUNT {
            return Err(anchor_error(format!(
                "layer {} routed expert {expert} is outside 0..{}",
                self.layer, DSV4F_LAYER_SOURCE_ANCHOR_ROUTED_EXPERT_COUNT
            )));
        }
        Ok(native_pair_anchor(
            format!(
                "layers.{}.ffn.experts.{expert}.{}",
                self.layer,
                projection.suffix()
            ),
            projection.fp4_routed_expectation(),
        ))
    }

    /// Exact score and route-data tensor bindings for this layer's gate.
    pub fn gate_binding(&self) -> DeepSeekV4GateSourceBinding {
        let route_data = match self.gate_mode {
            DeepSeekV4LayerGateMode::HashTokenIdToExpertIds => tensor_anchor(
                format!("layers.{}.ffn.gate.tid2eid", self.layer),
                TensorExpectation {
                    dtype: "I64",
                    shape: SHAPE_I64_TID2EID,
                },
            ),
            DeepSeekV4LayerGateMode::LearnedScoresWithSelectionBias => tensor_anchor(
                format!("layers.{}.ffn.gate.bias", self.layer),
                TensorExpectation {
                    dtype: "F32",
                    shape: SHAPE_F32_256,
                },
            ),
        };
        DeepSeekV4GateSourceBinding {
            mode: self.gate_mode,
            score_weight: tensor_anchor(
                format!("layers.{}.ffn.gate.weight", self.layer),
                TensorExpectation {
                    dtype: "BF16",
                    shape: SHAPE_BF16_GATE,
                },
            ),
            route_data,
            activated_experts_per_token: DSV4F_LAYER_SOURCE_ANCHOR_TOP_K,
            algorithm: gate_algorithm_binding(),
        }
    }

    /// Exact source mHC triplet for attention or feed-forward residual mixing.
    pub fn mhc_binding(&self, stage: DeepSeekV4LayerMhcStage) -> DeepSeekV4MhcSourceBinding {
        let prefix = format!("layers.{}.{}", self.layer, stage.stem());
        DeepSeekV4MhcSourceBinding {
            stage,
            fn_tensor: tensor_anchor(
                format!("{prefix}_fn"),
                TensorExpectation {
                    dtype: "F32",
                    shape: SHAPE_F32_HC_FN,
                },
            ),
            base_tensor: tensor_anchor(
                format!("{prefix}_base"),
                TensorExpectation {
                    dtype: "F32",
                    shape: SHAPE_F32_24,
                },
            ),
            scale_tensor: tensor_anchor(
                format!("{prefix}_scale"),
                TensorExpectation {
                    dtype: "F32",
                    shape: SHAPE_F32_3,
                },
            ),
        }
    }

    /// Exact source compression/indexer tensor bindings for this layer.
    pub fn compression_binding(&self) -> DeepSeekV4CompressionSourceBinding {
        let attention_compressor = self
            .compression
            .has_attention_compressor()
            .then(|| compressor_anchor_array(self.layer, false, self.compression));
        let has_indexer = self.compression.has_indexer();
        DeepSeekV4CompressionSourceBinding {
            mode: self.compression,
            attention_compressor,
            indexer_compressor: has_indexer
                .then(|| compressor_anchor_array(self.layer, true, self.compression)),
            indexer_wq_b: has_indexer.then(|| {
                native_pair_anchor(
                    format!("layers.{}.attn.indexer.wq_b.weight", self.layer),
                    FP8_INDEXER_WQ_B,
                )
            }),
            indexer_weights_proj: has_indexer.then(|| {
                tensor_anchor(
                    format!("layers.{}.attn.indexer.weights_proj.weight", self.layer),
                    TensorExpectation {
                        dtype: "BF16",
                        shape: SHAPE_BF16_INDEXER_WEIGHTS_PROJ,
                    },
                )
            }),
        }
    }

    /// Non-pair attention/norm binding by a stable source suffix.  This is
    /// intentionally narrow; native pairs, gate, mHC, and compression each
    /// retain typed accessors above.
    pub fn common_tensor(&self, kind: DeepSeekV4LayerCommonTensor) -> DeepSeekV4TensorSourceAnchor {
        let (suffix, expectation) = match kind {
            DeepSeekV4LayerCommonTensor::AttentionSink => (
                "attn.attn_sink",
                TensorExpectation {
                    dtype: "F32",
                    shape: SHAPE_F32_64,
                },
            ),
            DeepSeekV4LayerCommonTensor::AttentionQNorm => (
                "attn.q_norm.weight",
                TensorExpectation {
                    dtype: "BF16",
                    shape: SHAPE_BF16_1024,
                },
            ),
            DeepSeekV4LayerCommonTensor::AttentionKvNorm => (
                "attn.kv_norm.weight",
                TensorExpectation {
                    dtype: "BF16",
                    shape: SHAPE_BF16_512,
                },
            ),
            DeepSeekV4LayerCommonTensor::AttentionNorm => (
                "attn_norm.weight",
                TensorExpectation {
                    dtype: "BF16",
                    shape: SHAPE_BF16_4096,
                },
            ),
            DeepSeekV4LayerCommonTensor::FeedForwardNorm => (
                "ffn_norm.weight",
                TensorExpectation {
                    dtype: "BF16",
                    shape: SHAPE_BF16_4096,
                },
            ),
        };
        tensor_anchor(format!("layers.{}.{}", self.layer, suffix), expectation)
    }
}

/// Stable non-pair tensor names shared by every base layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4LayerCommonTensor {
    AttentionSink,
    AttentionQNorm,
    AttentionKvNorm,
    AttentionNorm,
    FeedForwardNorm,
}

impl DeepSeekV4LayerCommonTensor {
    const ALL: [Self; 5] = [
        Self::AttentionSink,
        Self::AttentionQNorm,
        Self::AttentionKvNorm,
        Self::AttentionNorm,
        Self::FeedForwardNorm,
    ];
}

/// The validated compact descriptor set for the 43-layer base child body.
/// It is deliberately reader-independent after construction, so a future
/// executor can retain only these few kilobytes plus its own bounded caches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4LayerSourceAnchors {
    identity: DeepSeekV4LayerSourceIdentity,
    geometry: DeepSeekV4LayerSourceGeometry,
    layers: Vec<DeepSeekV4LayerSourceAnchor>,
    base_tensor_count: usize,
    excluded_mtp_tensor_count: usize,
}

impl DeepSeekV4LayerSourceAnchors {
    pub fn identity(&self) -> &DeepSeekV4LayerSourceIdentity {
        &self.identity
    }

    pub fn geometry(&self) -> &DeepSeekV4LayerSourceGeometry {
        &self.geometry
    }

    pub fn layers(&self) -> &[DeepSeekV4LayerSourceAnchor] {
        &self.layers
    }

    pub fn layer(&self, layer: usize) -> Result<&DeepSeekV4LayerSourceAnchor> {
        self.layers.get(layer).ok_or_else(|| {
            anchor_error(format!(
                "layer {layer} is outside the {}-layer base body",
                DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT
            ))
        })
    }

    pub fn base_tensor_count(&self) -> usize {
        self.base_tensor_count
    }

    pub fn excluded_mtp_tensor_count(&self) -> usize {
        self.excluded_mtp_tensor_count
    }

    /// Re-run the complete metadata walk and require the same immutable
    /// artifact/source identity and compact descriptor result.  It never
    /// trusts a descriptor set across an artifact or source revision change.
    pub fn verify_against(&self, reader: &DeepSeekV4FullStreamReader) -> Result<()> {
        let current = verify_deepseek_v4_layer_source_anchors(reader)?;
        if self != &current {
            return Err(anchor_error(
                "reader bindings differ from the immutable layer-source anchors",
            ));
        }
        Ok(())
    }
}

/// Fully validate every base-layer source binding from an admitted reader and
/// return a compact descriptor set.  This is metadata-only: no tensor bytes
/// are read and no model data is uploaded or retained.
pub fn verify_deepseek_v4_layer_source_anchors(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<DeepSeekV4LayerSourceAnchors> {
    let identity = verify_identity(reader)?;
    let geometry = verify_geometry(reader)?;
    verify_global_base_tensors(reader)?;
    verify_inventory(reader)?;

    let mut layers = Vec::with_capacity(DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT);
    let mut base_tensor_count = 6usize;
    for layer in 0..geometry.base_layer_count {
        let compression = compression_mode_for_ratio(
            *geometry
                .base_compression_ratios
                .get(layer)
                .ok_or_else(|| anchor_error(format!("missing base compression ratio for layer {layer}")))?,
            layer,
        )?;
        let gate_mode = if layer < geometry.hash_layer_count {
            DeepSeekV4LayerGateMode::HashTokenIdToExpertIds
        } else {
            DeepSeekV4LayerGateMode::LearnedScoresWithSelectionBias
        };
        verify_layer(reader, layer, compression, gate_mode)?;
        let tensor_count = base_layer_tensor_count(compression);
        base_tensor_count = base_tensor_count
            .checked_add(tensor_count)
            .ok_or_else(|| anchor_error("base tensor count overflow"))?;
        layers.push(DeepSeekV4LayerSourceAnchor {
            layer,
            compression,
            gate_mode,
            tensor_count,
        });
    }

    let excluded_mtp_tensor_count = reader
        .tensor_count()
        .checked_sub(base_tensor_count)
        .ok_or_else(|| anchor_error("base layer inventory exceeds reader inventory"))?;
    if base_tensor_count != EXPECTED_BASE_TENSOR_COUNT
        || excluded_mtp_tensor_count != EXPECTED_EXCLUDED_MTP_TENSOR_COUNT
    {
        return Err(anchor_error(format!(
            "base/MTP tensor partition differs from the pinned source: base={base_tensor_count}, excluded_mtp={excluded_mtp_tensor_count}"
        )));
    }

    Ok(DeepSeekV4LayerSourceAnchors {
        identity,
        geometry,
        layers,
        base_tensor_count,
        excluded_mtp_tensor_count,
    })
}

fn verify_identity(reader: &DeepSeekV4FullStreamReader) -> Result<DeepSeekV4LayerSourceIdentity> {
    let source = reader.source_identity();
    if source.repository != PINNED_REPOSITORY || source.revision != PINNED_REVISION {
        return Err(anchor_error(format!(
            "reader source is {}/{} rather than pinned {}/{}",
            source.repository, source.revision, PINNED_REPOSITORY, PINNED_REVISION
        )));
    }
    let identity = DeepSeekV4LayerSourceIdentity {
        artifact_root: reader.artifact_root().display().to_string(),
        repository: source.repository.clone(),
        revision: source.revision.clone(),
        manifest_seal_sha256: reader.manifest_seal_sha256().to_owned(),
        manifest_file_sha256: reader.manifest_file_sha256().to_owned(),
        restart_seal_sha256: reader.restart_seal_sha256().to_owned(),
        inference_model_py_sha256: reader
            .source_metadata_asset_sha256("inference/model.py")?
            .to_owned(),
        inference_kernel_py_sha256: reader
            .source_metadata_asset_sha256("inference/kernel.py")?
            .to_owned(),
        inference_config_json_sha256: reader
            .source_metadata_asset_sha256("inference/config.json")?
            .to_owned(),
        model_config_json_sha256: reader.source_metadata_asset_sha256("config.json")?.to_owned(),
    };
    if identity.inference_model_py_sha256 != OFFICIAL_INFERENCE_MODEL_PY_SHA256
        || identity.inference_kernel_py_sha256 != OFFICIAL_INFERENCE_KERNEL_PY_SHA256
        || identity.inference_config_json_sha256 != OFFICIAL_INFERENCE_CONFIG_JSON_SHA256
        || identity.model_config_json_sha256 != OFFICIAL_MODEL_CONFIG_JSON_SHA256
    {
        return Err(anchor_error(
            "authenticated model/kernel/config source hashes differ from the pinned layer anchors",
        ));
    }
    if identity.manifest_seal_sha256.is_empty()
        || identity.manifest_file_sha256.is_empty()
        || identity.restart_seal_sha256.is_empty()
    {
        return Err(anchor_error("reader identity has an empty artifact seal binding"));
    }
    Ok(identity)
}

fn verify_geometry(reader: &DeepSeekV4FullStreamReader) -> Result<DeepSeekV4LayerSourceGeometry> {
    let model = read_verified_json(reader, "config.json")?;
    let inference = read_verified_json(reader, "inference/config.json")?;

    expect_string(&model, "model_type", "deepseek_v4", "config.json")?;
    expect_string(&model, "torch_dtype", "bfloat16", "config.json")?;
    expect_string(&model, "expert_dtype", "fp4", "config.json")?;
    expect_string(&model, "scoring_func", "sqrtsoftplus", "config.json")?;
    expect_string(&model, "topk_method", "noaux_tc", "config.json")?;
    expect_bool(&model, "norm_topk_prob", true, "config.json")?;
    expect_f64(&model, "routed_scaling_factor", 1.5, "config.json")?;
    expect_string(&inference, "dtype", "fp8", "inference/config.json")?;
    expect_string(&inference, "expert_dtype", "fp4", "inference/config.json")?;
    expect_string(&inference, "scale_fmt", "ue8m0", "inference/config.json")?;
    expect_f64(&inference, "route_scale", 1.5, "inference/config.json")?;

    let geometry = DeepSeekV4LayerSourceGeometry {
        hidden_size: required_usize(&model, "hidden_size", "config.json")?,
        vocab_size: required_usize(&model, "vocab_size", "config.json")?,
        base_layer_count: required_usize(&model, "num_hidden_layers", "config.json")?,
        excluded_mtp_layer_count: required_usize(
            &model,
            "num_nextn_predict_layers",
            "config.json",
        )?,
        hash_layer_count: required_usize(&model, "num_hash_layers", "config.json")?,
        routed_expert_count: required_usize(&model, "n_routed_experts", "config.json")?,
        shared_expert_count: required_usize(&model, "n_shared_experts", "config.json")?,
        activated_experts_per_token: required_usize(
            &model,
            "num_experts_per_tok",
            "config.json",
        )?,
        hc_mult: required_usize(&model, "hc_mult", "config.json")?,
        hc_sinkhorn_iters: required_usize(&model, "hc_sinkhorn_iters", "config.json")?,
        q_lora_rank: required_usize(&model, "q_lora_rank", "config.json")?,
        o_lora_rank: required_usize(&model, "o_lora_rank", "config.json")?,
        attention_head_count: required_usize(&model, "num_attention_heads", "config.json")?,
        head_dim: required_usize(&model, "head_dim", "config.json")?,
        rope_head_dim: required_usize(&model, "qk_rope_head_dim", "config.json")?,
        sliding_window_tokens: required_usize(&model, "sliding_window", "config.json")?,
        base_compression_ratios: required_usize_array(
            &inference,
            "compress_ratios",
            "inference/config.json",
        )?,
    };
    if geometry.hidden_size != DSV4F_LAYER_SOURCE_ANCHOR_HIDDEN_SIZE
        || geometry.vocab_size != VOCAB_SIZE
        || geometry.base_layer_count != DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT
        || geometry.excluded_mtp_layer_count != 1
        || geometry.hash_layer_count != HASH_LAYER_COUNT
        || geometry.routed_expert_count != DSV4F_LAYER_SOURCE_ANCHOR_ROUTED_EXPERT_COUNT
        || geometry.shared_expert_count != 1
        || geometry.activated_experts_per_token != DSV4F_LAYER_SOURCE_ANCHOR_TOP_K
        || geometry.hc_mult != HC_MULT
        || geometry.hc_sinkhorn_iters != HC_SINKHORN_ITERS
        || geometry.q_lora_rank != 1_024
        || geometry.o_lora_rank != 1_024
        || geometry.attention_head_count != 64
        || geometry.head_dim != 512
        || geometry.rope_head_dim != 64
        || geometry.sliding_window_tokens != 128
    {
        return Err(anchor_error(
            "source config geometry differs from the pinned 43-layer child body",
        ));
    }

    verify_duplicate_config_usize(
        &model,
        "vocab_size",
        &inference,
        "vocab_size",
        geometry.vocab_size,
    )?;
    verify_duplicate_config_usize(
        &model,
        "hidden_size",
        &inference,
        "dim",
        geometry.hidden_size,
    )?;
    verify_duplicate_config_usize(
        &model,
        "num_hidden_layers",
        &inference,
        "n_layers",
        geometry.base_layer_count,
    )?;
    verify_duplicate_config_usize(
        &model,
        "num_hash_layers",
        &inference,
        "n_hash_layers",
        geometry.hash_layer_count,
    )?;
    verify_duplicate_config_usize(
        &model,
        "n_routed_experts",
        &inference,
        "n_routed_experts",
        geometry.routed_expert_count,
    )?;
    verify_duplicate_config_usize(
        &model,
        "n_shared_experts",
        &inference,
        "n_shared_experts",
        geometry.shared_expert_count,
    )?;
    verify_duplicate_config_usize(
        &model,
        "num_experts_per_tok",
        &inference,
        "n_activated_experts",
        geometry.activated_experts_per_token,
    )?;
    verify_duplicate_config_usize(&model, "hc_mult", &inference, "hc_mult", geometry.hc_mult)?;
    verify_duplicate_config_usize(
        &model,
        "hc_sinkhorn_iters",
        &inference,
        "hc_sinkhorn_iters",
        geometry.hc_sinkhorn_iters,
    )?;
    verify_duplicate_config_usize(
        &model,
        "q_lora_rank",
        &inference,
        "q_lora_rank",
        geometry.q_lora_rank,
    )?;
    verify_duplicate_config_usize(
        &model,
        "o_lora_rank",
        &inference,
        "o_lora_rank",
        geometry.o_lora_rank,
    )?;
    verify_duplicate_config_usize(
        &model,
        "num_attention_heads",
        &inference,
        "n_heads",
        geometry.attention_head_count,
    )?;
    verify_duplicate_config_usize(
        &model,
        "head_dim",
        &inference,
        "head_dim",
        geometry.head_dim,
    )?;
    verify_duplicate_config_usize(
        &model,
        "qk_rope_head_dim",
        &inference,
        "rope_head_dim",
        geometry.rope_head_dim,
    )?;
    let inference_window = required_usize(&inference, "window_size", "inference/config.json")?;
    if inference_window != geometry.sliding_window_tokens {
        return Err(anchor_error(format!(
            "config sliding window mismatch: config.json={} inference/config.json={inference_window}",
            geometry.sliding_window_tokens
        )));
    }

    let all_compression_ratios = geometry.base_compression_ratios.clone();
    if all_compression_ratios.len()
        != DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT + geometry.excluded_mtp_layer_count
    {
        return Err(anchor_error(format!(
            "inference compression schedule has {} entries, expected {} base plus {} MTP",
            all_compression_ratios.len(),
            DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT,
            geometry.excluded_mtp_layer_count
        )));
    }
    for (index, ratio) in all_compression_ratios.iter().copied().enumerate() {
        let expected = expected_compression_ratio(index);
        if ratio != expected {
            return Err(anchor_error(format!(
                "compression ratio at source layer {index} is {ratio}, expected {expected}"
            )));
        }
    }
    let mut geometry = geometry;
    geometry
        .base_compression_ratios
        .truncate(DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT);
    Ok(geometry)
}

fn verify_inventory(reader: &DeepSeekV4FullStreamReader) -> Result<()> {
    if reader.tensor_count() != EXPECTED_TOTAL_TENSOR_COUNT
        || reader.tensor_bytes() != EXPECTED_TOTAL_TENSOR_BYTES
        || reader.native_scale_pair_count_for(NativeScalePairKind::Fp8E4M3fn)
            != EXPECTED_FP8_PAIR_COUNT
        || reader.native_scale_pair_count_for(NativeScalePairKind::Fp4E2M1fnX2)
            != EXPECTED_FP4_PAIR_COUNT
        || reader.native_scale_pair_count() != EXPECTED_FP8_PAIR_COUNT + EXPECTED_FP4_PAIR_COUNT
    {
        return Err(anchor_error(format!(
            "admitted inventory differs from the pinned source: tensors={}, bytes={}, fp8_pairs={}, fp4_pairs={}, total_pairs={}",
            reader.tensor_count(),
            reader.tensor_bytes(),
            reader.native_scale_pair_count_for(NativeScalePairKind::Fp8E4M3fn),
            reader.native_scale_pair_count_for(NativeScalePairKind::Fp4E2M1fnX2),
            reader.native_scale_pair_count(),
        )));
    }
    Ok(())
}

fn verify_global_base_tensors(reader: &DeepSeekV4FullStreamReader) -> Result<()> {
    verify_tensor(
        reader,
        "embed.weight",
        TensorExpectation {
            dtype: "BF16",
            shape: &[VOCAB_SIZE as u64, DSV4F_LAYER_SOURCE_ANCHOR_HIDDEN_SIZE as u64],
        },
    )?;
    verify_tensor(
        reader,
        "norm.weight",
        TensorExpectation {
            dtype: "BF16",
            shape: SHAPE_BF16_4096,
        },
    )?;
    verify_tensor(
        reader,
        "head.weight",
        TensorExpectation {
            dtype: "BF16",
            shape: &[VOCAB_SIZE as u64, DSV4F_LAYER_SOURCE_ANCHOR_HIDDEN_SIZE as u64],
        },
    )?;
    verify_tensor(
        reader,
        "hc_head_fn",
        TensorExpectation {
            dtype: "F32",
            shape: &[4, 16_384],
        },
    )?;
    verify_tensor(
        reader,
        "hc_head_base",
        TensorExpectation {
            dtype: "F32",
            shape: &[4],
        },
    )?;
    verify_tensor(
        reader,
        "hc_head_scale",
        TensorExpectation {
            dtype: "F32",
            shape: &[1],
        },
    )
}

fn verify_layer(
    reader: &DeepSeekV4FullStreamReader,
    layer: usize,
    compression: DeepSeekV4LayerCompressionMode,
    gate_mode: DeepSeekV4LayerGateMode,
) -> Result<()> {
    let compact = DeepSeekV4LayerSourceAnchor {
        layer,
        compression,
        gate_mode,
        tensor_count: base_layer_tensor_count(compression),
    };
    for kind in DeepSeekV4LayerCommonTensor::ALL {
        let tensor = compact.common_tensor(kind);
        verify_tensor_anchor(reader, &tensor)?;
    }
    for projection in DeepSeekV4LayerControlProjection::ALL {
        verify_native_pair_anchor(reader, &compact.control_pair(projection))?;
    }
    let gate = compact.gate_binding();
    verify_tensor_anchor(reader, &gate.score_weight)?;
    verify_tensor_anchor(reader, &gate.route_data)?;
    for projection in DeepSeekV4LayerExpertProjection::ALL {
        verify_native_pair_anchor(reader, &compact.shared_expert_pair(projection))?;
    }
    for expert in 0..DSV4F_LAYER_SOURCE_ANCHOR_ROUTED_EXPERT_COUNT {
        for projection in DeepSeekV4LayerExpertProjection::ALL {
            verify_native_pair_anchor(reader, &compact.routed_expert_pair(expert, projection)?)?;
        }
    }
    for stage in DeepSeekV4LayerMhcStage::ALL {
        let mhc = compact.mhc_binding(stage);
        verify_tensor_anchor(reader, &mhc.fn_tensor)?;
        verify_tensor_anchor(reader, &mhc.base_tensor)?;
        verify_tensor_anchor(reader, &mhc.scale_tensor)?;
    }
    let compression = compact.compression_binding();
    if let Some(tensors) = &compression.attention_compressor {
        for tensor in tensors {
            verify_tensor_anchor(reader, tensor)?;
        }
    }
    if let Some(tensors) = &compression.indexer_compressor {
        for tensor in tensors {
            verify_tensor_anchor(reader, tensor)?;
        }
    }
    if let Some(pair) = &compression.indexer_wq_b {
        verify_native_pair_anchor(reader, pair)?;
    }
    if let Some(tensor) = &compression.indexer_weights_proj {
        verify_tensor_anchor(reader, tensor)?;
    }
    Ok(())
}

fn verify_tensor_anchor(
    reader: &DeepSeekV4FullStreamReader,
    anchor: &DeepSeekV4TensorSourceAnchor,
) -> Result<()> {
    verify_tensor(
        reader,
        &anchor.name,
        TensorExpectation {
            dtype: anchor.dtype,
            shape: &anchor.shape,
        },
    )
}

fn verify_native_pair_anchor(
    reader: &DeepSeekV4FullStreamReader,
    anchor: &DeepSeekV4NativePairSourceAnchor,
) -> Result<()> {
    let pair = reader.native_scale_pair(&anchor.weight.name)?;
    if pair.kind != anchor.kind
        || pair.out_rows != anchor.out_rows
        || pair.packed_k != anchor.packed_k
        || pair.logical_k != anchor.logical_k
        || pair.scale_rows != anchor.scale_rows
        || pair.scale_cols != anchor.scale_cols
        || pair.scale.name != anchor.scale.name
    {
        return Err(anchor_error(format!(
            "{}: native pair geometry differs from compact source anchor",
            anchor.weight.name
        )));
    }
    verify_tensor_anchor(reader, &anchor.weight)?;
    verify_tensor_anchor(reader, &anchor.scale)
}

fn verify_tensor(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    expected: TensorExpectation,
) -> Result<()> {
    let metadata = reader.tensor_metadata(name)?;
    let expected_bytes = tensor_bytes(expected)?;
    if metadata.name != name
        || metadata.dtype != expected.dtype
        || metadata.shape.as_slice() != expected.shape
        || metadata.bytes != expected_bytes
        || metadata.segments.is_empty()
    {
        return Err(anchor_error(format!(
            "{name}: metadata differs from required {} {:?} / {expected_bytes} bytes",
            expected.dtype, expected.shape
        )));
    }
    Ok(())
}

fn tensor_bytes(expected: TensorExpectation) -> Result<u64> {
    let elements = expected.shape.iter().try_fold(1_u64, |total, dim| {
        total
            .checked_mul(*dim)
            .ok_or_else(|| anchor_error("expected tensor element count overflow"))
    })?;
    let element_bytes = match expected.dtype {
        "BF16" => 2,
        "F32" => 4,
        "I64" => 8,
        "F8_E4M3" | "F8_E8M0" | "I8" => 1,
        other => return Err(anchor_error(format!("unknown expected dtype {other:?}"))),
    };
    elements
        .checked_mul(element_bytes)
        .ok_or_else(|| anchor_error("expected tensor byte count overflow"))
}

fn native_pair_anchor(
    weight_name: String,
    expectation: NativePairExpectation,
) -> DeepSeekV4NativePairSourceAnchor {
    let scale_name = weight_name
        .strip_suffix(".weight")
        .map(|prefix| format!("{prefix}.scale"))
        .expect("all pinned native pair weight names end in .weight");
    DeepSeekV4NativePairSourceAnchor {
        kind: expectation.kind,
        weight: tensor_anchor(weight_name, expectation.weight),
        scale: tensor_anchor(scale_name, expectation.scale),
        out_rows: expectation.out_rows,
        packed_k: expectation.packed_k,
        logical_k: expectation.logical_k,
        scale_rows: expectation.scale_rows,
        scale_cols: expectation.scale_cols,
    }
}

fn tensor_anchor(name: String, expectation: TensorExpectation) -> DeepSeekV4TensorSourceAnchor {
    DeepSeekV4TensorSourceAnchor {
        name,
        dtype: expectation.dtype,
        shape: expectation.shape.to_vec(),
        bytes: tensor_bytes(expectation).expect("pinned source tensor geometry is bounded"),
    }
}

fn compressor_anchor_array(
    layer: usize,
    indexer: bool,
    compression: DeepSeekV4LayerCompressionMode,
) -> [DeepSeekV4TensorSourceAnchor; 4] {
    let prefix = if indexer {
        format!("layers.{layer}.attn.indexer.compressor")
    } else {
        format!("layers.{layer}.attn.compressor")
    };
    let expectations = compressor_expectations(indexer, compression)
        .expect("compact compressor only requested for source-supported modes");
    std::array::from_fn(|index| {
        tensor_anchor(
            format!("{}.{}", prefix, DeepSeekV4LayerCompressorTensor::ALL[index].suffix()),
            expectations[index],
        )
    })
}

fn compressor_expectations(
    indexer: bool,
    compression: DeepSeekV4LayerCompressionMode,
) -> Option<[TensorExpectation; 4]> {
    if indexer {
        return matches!(compression, DeepSeekV4LayerCompressionMode::Ratio4WithIndexer).then_some([
            TensorExpectation {
                dtype: "F32",
                shape: SHAPE_F32_INDEXER_APE,
            },
            TensorExpectation {
                dtype: "BF16",
                shape: SHAPE_BF16_INDEXER_COMPRESSOR_WEIGHT,
            },
            TensorExpectation {
                dtype: "BF16",
                shape: SHAPE_BF16_INDEXER_COMPRESSOR_WEIGHT,
            },
            TensorExpectation {
                dtype: "BF16",
                shape: SHAPE_BF16_INDEXER_NORM,
            },
        ]);
    }
    match compression {
        DeepSeekV4LayerCompressionMode::SlidingWindowOnly => None,
        DeepSeekV4LayerCompressionMode::Ratio4WithIndexer => Some([
            TensorExpectation {
                dtype: "F32",
                shape: SHAPE_F32_RATIO4_APE,
            },
            TensorExpectation {
                dtype: "BF16",
                shape: SHAPE_BF16_RATIO4_COMPRESSOR_WEIGHT,
            },
            TensorExpectation {
                dtype: "BF16",
                shape: SHAPE_BF16_RATIO4_COMPRESSOR_WEIGHT,
            },
            TensorExpectation {
                dtype: "BF16",
                shape: SHAPE_BF16_COMPRESSOR_NORM,
            },
        ]),
        DeepSeekV4LayerCompressionMode::Ratio128 => Some([
            TensorExpectation {
                dtype: "F32",
                shape: SHAPE_F32_RATIO128_APE,
            },
            TensorExpectation {
                dtype: "BF16",
                shape: SHAPE_BF16_RATIO128_COMPRESSOR_WEIGHT,
            },
            TensorExpectation {
                dtype: "BF16",
                shape: SHAPE_BF16_RATIO128_COMPRESSOR_WEIGHT,
            },
            TensorExpectation {
                dtype: "BF16",
                shape: SHAPE_BF16_COMPRESSOR_NORM,
            },
        ]),
    }
}

fn gate_algorithm_binding() -> DeepSeekV4GateAlgorithmBinding {
    DeepSeekV4GateAlgorithmBinding {
        score_transform: "sqrt_softplus",
        selected_weights_use_pre_bias_scores: true,
        non_softmax_weights_are_normalized: true,
        route_scale_numerator: 3,
        route_scale_denominator: 2,
    }
}

fn base_layer_tensor_count(compression: DeepSeekV4LayerCompressionMode) -> usize {
    // 1 sink + 2 attention norms + 2 block norms + 5 FP8 pairs + gate pair
    // + 3 FP8 shared pairs + 6 mHC tensors + 256 × 3 FP4 routed pairs.
    let fixed = 1_565;
    fixed
        + match compression {
            DeepSeekV4LayerCompressionMode::SlidingWindowOnly => 0,
            DeepSeekV4LayerCompressionMode::Ratio128 => 4,
            DeepSeekV4LayerCompressionMode::Ratio4WithIndexer => 11,
        }
}

fn compression_mode_for_ratio(
    ratio: usize,
    layer: usize,
) -> Result<DeepSeekV4LayerCompressionMode> {
    match ratio {
        0 => Ok(DeepSeekV4LayerCompressionMode::SlidingWindowOnly),
        4 => Ok(DeepSeekV4LayerCompressionMode::Ratio4WithIndexer),
        128 => Ok(DeepSeekV4LayerCompressionMode::Ratio128),
        other => Err(anchor_error(format!(
            "base layer {layer} has unsupported source compression ratio {other}"
        ))),
    }
}

fn expected_compression_ratio(source_layer: usize) -> usize {
    match source_layer {
        0 | 1 | 43 => 0,
        index if index % 2 == 0 => 4,
        _ => 128,
    }
}

fn read_verified_json(reader: &DeepSeekV4FullStreamReader, asset: &str) -> Result<Value> {
    let raw = reader.read_verified_metadata_asset(asset, METADATA_READ_BOUND_BYTES)?;
    serde_json::from_slice(&raw)
        .map_err(|error| anchor_error(format!("{asset}: invalid authenticated JSON: {error}")))
}

fn required_usize(value: &Value, key: &str, label: &str) -> Result<usize> {
    let raw = value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| anchor_error(format!("{label}: missing unsigned integer {key:?}")))?;
    usize::try_from(raw)
        .map_err(|_| anchor_error(format!("{label}: {key:?}={raw} does not fit host usize")))
}

fn required_usize_array(value: &Value, key: &str, label: &str) -> Result<Vec<usize>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| anchor_error(format!("{label}: missing integer array {key:?}")))?
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let raw = item.as_u64().ok_or_else(|| {
                anchor_error(format!("{label}: {key:?}[{index}] is not an unsigned integer"))
            })?;
            usize::try_from(raw).map_err(|_| {
                anchor_error(format!(
                    "{label}: {key:?}[{index}]={raw} does not fit host usize"
                ))
            })
        })
        .collect()
}

fn expect_string(value: &Value, key: &str, expected: &str, label: &str) -> Result<()> {
    let actual = value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anchor_error(format!("{label}: missing string {key:?}")))?;
    if actual != expected {
        return Err(anchor_error(format!(
            "{label}: {key:?}={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn expect_bool(value: &Value, key: &str, expected: bool, label: &str) -> Result<()> {
    let actual = value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| anchor_error(format!("{label}: missing bool {key:?}")))?;
    if actual != expected {
        return Err(anchor_error(format!(
            "{label}: {key:?}={actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn expect_f64(value: &Value, key: &str, expected: f64, label: &str) -> Result<()> {
    let actual = value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| anchor_error(format!("{label}: missing numeric {key:?}")))?;
    if actual != expected {
        return Err(anchor_error(format!(
            "{label}: {key:?}={actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn verify_duplicate_config_usize(
    model: &Value,
    model_key: &str,
    inference: &Value,
    inference_key: &str,
    expected: usize,
) -> Result<()> {
    let model_value = required_usize(model, model_key, "config.json")?;
    let inference_value = required_usize(inference, inference_key, "inference/config.json")?;
    if model_value != expected || inference_value != expected {
        return Err(anchor_error(format!(
            "config disagreement: config.json.{model_key}={model_value}, inference/config.json.{inference_key}={inference_value}, expected {expected}"
        )));
    }
    Ok(())
}

fn anchor_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!(
        "DeepSeek-V4 layer source anchors: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_schedule_and_tensor_partition_are_exact() {
        let mut total = 6usize;
        for layer in 0..DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT {
            let mode = compression_mode_for_ratio(expected_compression_ratio(layer), layer)
                .expect("pinned ratio");
            total += base_layer_tensor_count(mode);
        }
        assert_eq!(total, EXPECTED_BASE_TENSOR_COUNT);
        assert_eq!(
            EXPECTED_BASE_TENSOR_COUNT + EXPECTED_EXCLUDED_MTP_TENSOR_COUNT,
            EXPECTED_TOTAL_TENSOR_COUNT
        );
        assert_eq!(expected_compression_ratio(0), 0);
        assert_eq!(expected_compression_ratio(1), 0);
        assert_eq!(expected_compression_ratio(2), 4);
        assert_eq!(expected_compression_ratio(3), 128);
        assert_eq!(expected_compression_ratio(42), 4);
        assert_eq!(expected_compression_ratio(43), 0);
    }

    #[test]
    fn compact_layer_descriptor_derives_exact_gate_mhc_and_native_pair_names() {
        let hash_layer = DeepSeekV4LayerSourceAnchor {
            layer: 2,
            compression: DeepSeekV4LayerCompressionMode::Ratio4WithIndexer,
            gate_mode: DeepSeekV4LayerGateMode::HashTokenIdToExpertIds,
            tensor_count: base_layer_tensor_count(DeepSeekV4LayerCompressionMode::Ratio4WithIndexer),
        };
        assert_eq!(hash_layer.gate_binding().score_weight.name, "layers.2.ffn.gate.weight");
        assert_eq!(hash_layer.gate_binding().route_data.name, "layers.2.ffn.gate.tid2eid");
        assert_eq!(
            hash_layer
                .mhc_binding(DeepSeekV4LayerMhcStage::Attention)
                .fn_tensor
                .name,
            "layers.2.hc_attn_fn"
        );
        let pair = hash_layer.control_pair(DeepSeekV4LayerControlProjection::WqB);
        assert_eq!(pair.weight.name, "layers.2.attn.wq_b.weight");
        assert_eq!(pair.scale.name, "layers.2.attn.wq_b.scale");
        assert_eq!(pair.logical_k, 1_024);
        let compression = hash_layer.compression_binding();
        assert_eq!(compression.mode.ratio(), 4);
        assert_eq!(
            compression.attention_compressor.as_ref().expect("ratio 4")[0].name,
            "layers.2.attn.compressor.ape"
        );
        assert_eq!(
            compression.indexer_compressor.as_ref().expect("indexer")[3].name,
            "layers.2.attn.indexer.compressor.norm.weight"
        );

        let learned_layer = DeepSeekV4LayerSourceAnchor {
            layer: 3,
            compression: DeepSeekV4LayerCompressionMode::Ratio128,
            gate_mode: DeepSeekV4LayerGateMode::LearnedScoresWithSelectionBias,
            tensor_count: base_layer_tensor_count(DeepSeekV4LayerCompressionMode::Ratio128),
        };
        assert_eq!(learned_layer.gate_binding().route_data.name, "layers.3.ffn.gate.bias");
        assert!(learned_layer.compression_binding().indexer_compressor.is_none());
        assert_eq!(
            learned_layer
                .routed_expert_pair(17, DeepSeekV4LayerExpertProjection::W2)
                .expect("routed expert")
                .weight
                .name,
            "layers.3.ffn.experts.17.w2.weight"
        );
    }

    #[test]
    fn routed_expert_bounds_and_ratio_zero_compression_fail_closed() {
        let layer = DeepSeekV4LayerSourceAnchor {
            layer: 0,
            compression: DeepSeekV4LayerCompressionMode::SlidingWindowOnly,
            gate_mode: DeepSeekV4LayerGateMode::HashTokenIdToExpertIds,
            tensor_count: base_layer_tensor_count(DeepSeekV4LayerCompressionMode::SlidingWindowOnly),
        };
        assert!(layer
            .routed_expert_pair(
                DSV4F_LAYER_SOURCE_ANCHOR_ROUTED_EXPERT_COUNT,
                DeepSeekV4LayerExpertProjection::W1,
            )
            .is_err());
        let compression = layer.compression_binding();
        assert!(compression.attention_compressor.is_none());
        assert!(compression.indexer_compressor.is_none());
        assert!(compression.indexer_wq_b.is_none());
        assert!(compression.indexer_weights_proj.is_none());
    }

    #[test]
    fn tensor_byte_grammar_matches_native_physical_storage() {
        assert_eq!(tensor_bytes(FP8_WQ_A.weight).expect("fp8 bytes"), 4_194_304);
        assert_eq!(tensor_bytes(FP8_WQ_A.scale).expect("fp8 scale bytes"), 256);
        assert_eq!(tensor_bytes(FP4_W1_W3.weight).expect("fp4 bytes"), 4_194_304);
        assert_eq!(tensor_bytes(FP4_W1_W3.scale).expect("fp4 scale bytes"), 262_144);
        assert_eq!(FP4_W1_W3.logical_k, 4_096);
        assert_eq!(FP4_W1_W3.packed_k, 2_048);
    }
}
