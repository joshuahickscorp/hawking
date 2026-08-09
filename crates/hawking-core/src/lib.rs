#![allow(clippy::all)]
pub mod artifact;
pub mod attn;
pub mod backend;
/// Broker-kernel A/B promotion gate + receipt-backed cost registry.
/// Planning/groundwork only; does not touch the DeepSeek-V4 forward lane.
pub mod broker_kernel_ab;
pub mod cache;
pub mod cost_ledger;
pub mod gguf;
pub mod gravity;
pub mod gravity_deepseek;
/// Read-only admission and verified range reader for the sealed DeepSeek-V4
/// full source stream.  This intentionally has no Engine, serving, Metal, or
/// forward integration: admitting bytes is not permission to execute a model.
pub mod gravity_deepseek_v4;
/// CPU-only source-algorithm oracle for DeepSeek-V4 FP8 activation
/// quantization and the bounded layer-0 WQ-A checkpoint.  This has no engine,
/// Metal, forward, or serving integration.
pub mod gravity_deepseek_v4_act_quant;
/// Parameterized ratio-zero attention device plan (layer/position/growing KV).
/// Resolves source tensor names and refuses ratio-4/128 cleanly.
pub mod gravity_deepseek_v4_attention_device;
/// Parameterized BOS/position-0 attention device graph for any base layer
/// 1..42. Window-KV only (compressed slots empty at BOS). Not a full causal
/// compressed graph, Engine, HCLI, serve, or TPS claim.
#[cfg(target_os = "macos")]
pub mod gravity_deepseek_v4_bos_layer_attention_device;
/// Bounded, source-backed preparation context for a future DeepSeek-V4 native
/// causal loop. It stages real source rows/operators and cache residency but
/// intentionally has no Engine, Metal dispatch, forward, or serving surface.
pub mod gravity_deepseek_v4_execution_context;
/// Bounded source-chunk-backed routed-expert cache for the admitted
/// DeepSeek-V4 full stream.  Storage only; no runtime or execution surface.
pub mod gravity_deepseek_v4_expert_cache;
/// Final mHC-head merge, RMSNorm, and greedy LM-head (host authority + optional
/// device gemv). Not an Engine, serve path, or exact-storage parity claim.
pub mod gravity_deepseek_v4_final_head;
/// Multi-token / non-BOS growing-KV attention (empty-compressed specialization
/// for ratio-4/128). Capture path for PROTO_FRANKENSTEIN_V0 bridge sites.
#[cfg(target_os = "macos")]
pub mod gravity_deepseek_v4_fullseq_attention_device;
/// CPU-only, source-algorithm-derived DeepSeek-V4 layer-0 position-zero
/// attention checkpoint.  It is not an Engine, upstream-runtime parity, or
/// runtime/TPS evidence.
pub mod gravity_deepseek_v4_layer0_attention;
/// Bounded source-derived layer-0 position-one causal-KV continuation oracle.
/// This has no full forward, serving, or TPS claim.
pub mod gravity_deepseek_v4_layer0_continuation;
/// CPU-only source-algorithm continuation of the DeepSeek-V4 layer-0
/// attention checkpoint through hash routing and the MoE branch.  This has no
/// Engine, Metal, serving, upstream-runtime-parity, or TPS claim.
pub mod gravity_deepseek_v4_layer0_moe;
/// CPU-only source-derived successor from the sealed layer-0 position-one
/// complete-attention mHC state through the hash-routed MoE FFN.  It is a
/// bounded parity target, not a registered decoder runtime or TPS surface.
pub mod gravity_deepseek_v4_layer0_position1_ffn;
/// CPU-only, source-algorithm-derived DeepSeek-V4 tokenizer-to-layer-0
/// Hyper-Connection prefix checkpoint.  It intentionally ends at the BF16
/// WQ-A input and has no upstream-runtime, Metal, forwarding, or TPS claim.
pub mod gravity_deepseek_v4_layer0_prefix;
/// Bounded macOS-only, caller-owned-context continuation from the layer-0
/// child boundary through layer-1/BOS ratio-zero attention. It is not a full
/// forward, Engine, HCLI endpoint, parity receipt, or TPS result.
#[cfg(target_os = "macos")]
pub mod gravity_deepseek_v4_layer1_attention_device;
/// General per-layer device plan resolved from source anchors: compression
/// mode, gate mode, and honest refusal for unimplemented ratio-4/128 paths.
pub mod gravity_deepseek_v4_layer_plan;
/// Incremental bounded source-staging scheduler for the future DeepSeek-V4
/// native layer loop. It has no default device encoder or causal runtime.
pub mod gravity_deepseek_v4_layer_scheduler;
/// Compact, source-bound per-layer tensor anchors for every DeepSeek-V4-Flash
/// base layer (0..42). Metadata-only; no Engine, Metal, or forward surface.
pub mod gravity_deepseek_v4_layer_source_anchors;
/// Bounded verifier for the opt-in, source-bound Torch F32 Gate calibration
/// target used by the position-zero diagnostic. It has no Metal, runtime,
/// route-default, or TPS surface.
pub mod gravity_deepseek_v4_p0_gate_calibration;
/// Reusable bounded Metal sink for the DeepSeek-V4 layer-0 P3A mHC/norm/Q
/// source path. It is deliberately below the Engine and causal-runtime gate.
pub mod gravity_deepseek_v4_p3a_stage_sink;
/// Reusable bounded all-device P4B layer-0 position-one attention graph. It
/// retains a caller-owned-context P7 handoff with Numeric Parity V2.1 only;
/// it makes no exact-storage, decoder-runtime, endpoint, or TPS claim.
#[cfg(target_os = "macos")]
pub mod gravity_deepseek_v4_p4b_device;
/// Reusable macOS-only, source-bound P6 device graph. It consumes P7's
/// caller-owned Metal context and BF16 predecessor buffer but does not itself
/// establish a runtime, endpoint, parity receipt, or TPS result.
#[cfg(target_os = "macos")]
pub mod gravity_deepseek_v4_p6_device;
/// Source-lease and device-handoff contract for bounded DeepSeek-V4 P7.
/// This module itself stages no commands; the macOS-only device executor
/// below owns the P7 kernels. Neither is an Engine, parity receipt, or
/// runtime claim.
pub mod gravity_deepseek_v4_p7_composition;
/// Bounded macOS-only P4B -> P7 -> P6 -> P7 device composition for the
/// layer-0/position-1 diagnostic. It preserves the P4B V2.1-only label and
/// has no decoder-runtime, endpoint, or TPS claim.
#[cfg(target_os = "macos")]
pub mod gravity_deepseek_v4_p7_device;
/// Immutable source-hash-bound ABI, storage, kernel-slot, residency, and
/// bridge-point sidecar for a future DeepSeek-V4 runtime. It cannot alter the
/// sealed stream, register an Engine, expose HCLI, or claim TPS.
pub mod gravity_deepseek_v4_runtime_binding;
/// Source-hash-bound DeepSeek-V4 base-body topology and bounded staging data
/// plane.  This is intentionally non-Engine/non-servable until a complete
/// 43-layer causal runtime is implemented and parity-gated.
pub mod gravity_deepseek_v4_runtime_spine;
/// Byte-bounded authenticated cache of source-native DeepSeek-V4 tensor
/// ranges for a future Metal executor. It is storage-only: no device upload,
/// forward, Engine, serving, or TPS surface exists here.
pub mod gravity_deepseek_v4_verified_tensor_cache;
pub mod gravity_glm;
#[cfg(target_os = "macos")]
pub mod gravity_glm_resident;
pub mod gravity_llama;
pub mod json_constrain;
pub mod kernel_bench;
pub mod kernels;
pub mod metal;
pub mod mixed_quant_store;
pub mod model;
pub mod moe;
/// Numeric Parity Contract V2.1 — condition-aware hybrid metrics + FP64 authority.
/// See root `NUMERIC_PARITY_V2_1.md`.
pub mod numeric_parity;
pub mod profile;
pub mod q4k_fast;
pub mod quant;
pub mod quant_tier_map;
pub mod sample;
pub mod sidecar;
// speculate extracted to the hawking-speculate crate (NUCLEAR PASTA).
// Re-export so integration tests (and any in-tree callers) that still path
// through `hawking_core::speculate::…` keep compiling.
pub use hawking_speculate as speculate;
pub mod stateful;
pub mod tokenizer;
/// TQ (Trellis-Quant): `.tq` decode + activation-RHT CPU serving reference, built
/// on the absorbed strand-quant codec. Behind the `tq` feature so default builds
/// are byte-identical.
#[cfg(feature = "tq")]
pub mod tq;
/// TQ GPU bitslice decode→GEMV: the Metal port of the STRAND G4 bitslice kernel,
/// held bit-identical to the `tq`/strand-quant CPU oracle. Behind `tq`.
#[cfg(feature = "tq")]
pub(crate) mod tq_gpu;
#[cfg(all(feature = "tq", target_os = "macos"))]
pub use tq_gpu::{gpu_decode_q12, TqDeviceHarness};
/// Public, `BitsliceEntry`-free entry point to the TQ GPU bitslice decode (the
/// parity gate's surface). macOS + `tq` only.
#[cfg(feature = "tq")]
pub use tq_gpu::{
    TqCodebookSource, TqGpuAdmission, TqGpuIneligibility, TqMetadataMode, TqRuntimePath,
    TqRuntimeRecipe, TqRuntimeTraffic,
};
pub mod vocab_prune;

mod error;
pub use error::{Error, Result};

mod engine;
pub use engine::{
    Engine, EngineConfig, GenStats, GenerateRequest, SamplingParams, SpeculateMode, StopReason,
    StreamEvent,
};

/// `true` when env var `name` is set to "1". The codebase's standard
/// on/off toggle for `HAWKING_*` levers.
pub fn env_on(name: &str) -> bool {
    std::env::var(name).map(|v| v == "1").unwrap_or(false)
}

/// `true` unless env var `name` is explicitly set to a disable token
/// (`0`, `false`, `off`, `no`, case-insensitive). The opt-OUT counterpart
/// to [`env_on`]: a default-ON lever stays on when the var is unset, and
/// is disabled only by an explicit disable token. Any other value (e.g.
/// `1`, `true`) leaves it on.
pub fn env_opt_out(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    }
}

/// Parse env var `name` as usize, falling back to `default` when unset
/// or unparseable.
pub fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
