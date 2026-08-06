//! Immutable sidecar contract for a future DeepSeek-V4 native runtime.
//!
//! This module binds an already-admitted, sealed full source stream to the
//! *shape* of a future runtime without registering one.  It deliberately has
//! no `Engine`, Metal, HCLI, token loop, manifest writer, artifact writer, or
//! throughput surface.  In particular, serializing this value records an
//! immutable sidecar only; it cannot promote the full-stream manifest.

use std::path::Path;

use serde::Serialize;

use crate::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, NativeScalePairKind, PINNED_REPOSITORY, PINNED_REVISION,
};
use crate::gravity_deepseek_v4_runtime_spine::{
    DSV4F_BASE_LAYER_COUNT, DSV4F_HIDDEN_SIZE, DSV4F_MTP_LAYER_COUNT, DSV4F_ROUTED_EXPERT_COUNT,
    DSV4F_TOP_K_EXPERTS, DSV4F_VOCAB_SIZE, MAX_STAGED_OPERATOR_BYTES,
    PROVISIONAL_CONTROL_RESIDENT_CEILING_BYTES, PROVISIONAL_ROUTED_EXPERT_COLD_CEILING_BYTES,
    PROVISIONAL_ROUTED_EXPERT_HOT_CEILING_BYTES,
};
use crate::{Error, Result};

/// Stable schema of an optional serialized runtime-binding sidecar.  The
/// schema is intentionally separate from, and never written into, the sealed
/// full-stream manifest.
pub const DSV4F_RUNTIME_BINDING_SCHEMA: &str =
    "hawking.gravity.deepseek_v4.runtime_binding_sidecar.v1";
/// This is a contract status, not a runtime-readiness status.
pub const DSV4F_RUNTIME_BINDING_STATUS: &str = "SOURCE_BOUND_NON_RUNTIME_SIDECAR";
/// ABI identifier reserved for a later source-faithful 43-layer executor.
pub const DSV4F_RUNTIME_ABI: &str = "hawking.deepseek_v4.native_causal_abi.v1";

const OFFICIAL_INFERENCE_MODEL_PY_SHA256: &str =
    "ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71";
const OFFICIAL_INFERENCE_KERNEL_PY_SHA256: &str =
    "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2";
const OFFICIAL_INFERENCE_CONFIG_JSON_SHA256: &str =
    "6cc6f816ca73a8d38750194e330398e4f6955b4b45f674f7d29c96da14ccb733";
const OFFICIAL_MODEL_CONFIG_JSON_SHA256: &str =
    "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818";

/// Identity that must remain identical before a future runtime may consume a
/// sidecar.  It binds the logical manifest seal, physical canonical manifest,
/// restart receipt, repository/revision, and the source grammar assets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeepSeekV4RuntimeBindingIdentity {
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

/// Explicit future execution ABI.  Its fields describe source geometry only;
/// no executor is constructed by this module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeepSeekV4RuntimeAbi {
    pub identifier: &'static str,
    pub base_layer_count: usize,
    pub excluded_mtp_layer_count: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub routed_expert_count: usize,
    pub activated_experts_per_token: usize,
    pub causal_state_abi: &'static str,
    pub token_alignment_abi: &'static str,
}

/// Source-native storage representation that a future runtime must preserve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DeepSeekV4SourceTensorRepresentation {
    Bf16LittleEndian,
    F32LittleEndian,
    I64LittleEndian,
    Fp8E4M3fnWithUe8m0Block128,
    Fp4E2M1fnX2WithUe8m0LogicalK32,
}

impl DeepSeekV4SourceTensorRepresentation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16LittleEndian => "bf16_little_endian",
            Self::F32LittleEndian => "f32_little_endian",
            Self::I64LittleEndian => "i64_little_endian",
            Self::Fp8E4M3fnWithUe8m0Block128 => "fp8_e4m3fn_ue8m0_block_128x128",
            Self::Fp4E2M1fnX2WithUe8m0LogicalK32 => "fp4_e2m1fn_x2_ue8m0_logical_k32",
        }
    }
}

/// The native representation grammar is bound to pair counts from the sealed
/// reader.  It does not decode or re-quantize any tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeepSeekV4SourceTensorGrammar {
    pub representations: Vec<DeepSeekV4SourceTensorRepresentation>,
    pub fp8_control_or_shared_pair_count: usize,
    pub fp4_routed_expert_pair_count: usize,
}

/// Explicit bounded-source residency policy for a future executor.  These are
/// policy ceilings inherited from the source data plane, not measured resident
/// bytes, cache hit rates, or an allocation performed by this sidecar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeepSeekV4CacheResidencyPolicy {
    pub policy_identifier: &'static str,
    pub source_parent_retention_permitted: bool,
    pub all_model_materialization_permitted: bool,
    pub maximum_single_host_stage_bytes: usize,
    pub provisional_control_resident_ceiling_bytes: u64,
    pub provisional_routed_expert_hot_ceiling_bytes: u64,
    pub provisional_routed_expert_cold_ceiling_bytes: u64,
    pub causal_kv_or_mhc_state_allocated: bool,
}

/// Stable ABI slot names for future source-native kernels.  A slot is not a
/// compiled or dispatchable kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DeepSeekV4KernelIdentifier {
    Fp8ControlMatvec,
    Fp4RoutedExpertMatvec,
    GateAndTopK,
    MhcState,
    CompressedAttentionAndIndex,
    RouteGather,
    SharedExpert,
    RouteWeightedCombine,
    LmHeadSampling,
}

impl DeepSeekV4KernelIdentifier {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fp8ControlMatvec => "dsv4f.native.fp8_control_matvec.v1",
            Self::Fp4RoutedExpertMatvec => "dsv4f.native.fp4_routed_expert_matvec.v1",
            Self::GateAndTopK => "dsv4f.native.gate_topk.v1",
            Self::MhcState => "dsv4f.native.mhc_state.v1",
            Self::CompressedAttentionAndIndex => "dsv4f.native.compressed_attention_index.v1",
            Self::RouteGather => "dsv4f.native.route_gather.v1",
            Self::SharedExpert => "dsv4f.native.shared_expert.v1",
            Self::RouteWeightedCombine => "dsv4f.native.route_weighted_combine.v1",
            Self::LmHeadSampling => "dsv4f.native.lm_head_sampling.v1",
        }
    }
}

/// One ABI kernel slot.  It is deliberately marked unavailable until a real
/// runtime has independently compiled, dispatched, and parity-gated it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeepSeekV4KernelBinding {
    pub identifier: DeepSeekV4KernelIdentifier,
    pub source_representation: Option<DeepSeekV4SourceTensorRepresentation>,
    pub dispatchable: bool,
}

/// Bounded telemetry/transplant tap names reserved for later inheritance
/// adapters.  All locations are read-only and preserve token alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DeepSeekV4ProtectedBridgeLocationId {
    PreNormHidden,
    PostAttentionHidden,
    PreRouterHidden,
    RouterLogits,
    RouteSelection,
    PostMoeHidden,
    MhcState,
    AttentionOrIndexState,
    FinalHidden,
    LmHeadLogits,
    ToolActionDecision,
}

impl DeepSeekV4ProtectedBridgeLocationId {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PreNormHidden => "pre_norm_hidden",
            Self::PostAttentionHidden => "post_attention_hidden",
            Self::PreRouterHidden => "pre_router_hidden",
            Self::RouterLogits => "router_logits",
            Self::RouteSelection => "route_selection",
            Self::PostMoeHidden => "post_moe_hidden",
            Self::MhcState => "mhc_state",
            Self::AttentionOrIndexState => "attention_or_index_state",
            Self::FinalHidden => "final_hidden",
            Self::LmHeadLogits => "lm_head_logits",
            Self::ToolActionDecision => "tool_action_decision",
        }
    }
}

/// The only source region a protected bridge may observe.  It does not grant
/// direct weight replacement or make the location live at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DeepSeekV4BridgeLayerScope {
    EveryBaseLayer,
    FinalBodyOnly,
    HcliEffectBoundary,
}

/// A protected location is a stable contract point, not a current latent
/// export.  Raw hidden-state retention remains outside this sidecar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeepSeekV4ProtectedBridgeLocation {
    pub identifier: DeepSeekV4ProtectedBridgeLocationId,
    pub layer_scope: DeepSeekV4BridgeLayerScope,
    pub token_aligned: bool,
    pub direct_weight_transplant_permitted: bool,
    pub runtime_export_available: bool,
}

/// Explicitly non-runtime flags.  Their false values are part of the sidecar
/// contract and may not be promoted by mutation of this data structure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeepSeekV4RuntimeBindingCapabilities {
    pub sealed_reader_identity_bound: bool,
    pub source_native_grammar_bound: bool,
    pub runtime_abi_declared: bool,
    pub cache_residency_policy_declared: bool,
    pub kernel_identifiers_declared: bool,
    pub protected_bridge_locations_declared: bool,
    pub manifest_mutation_permitted: bool,
    pub manifest_runtime_promotion_permitted: bool,
    pub engine_registered: bool,
    pub causal_forward_available: bool,
    pub metal_dispatches_available: bool,
    pub hcli_endpoint_available: bool,
    pub numeric_parity_v21_passed: bool,
    pub base_true_tps_eligible: bool,
}

/// Immutable source-bound runtime sidecar.  The type intentionally exposes
/// accessors only; callers cannot edit its identity, grant capabilities, or
/// promote the manifest through this module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeepSeekV4RuntimeBinding {
    schema: &'static str,
    status: &'static str,
    identity: DeepSeekV4RuntimeBindingIdentity,
    abi: DeepSeekV4RuntimeAbi,
    tensor_grammar: DeepSeekV4SourceTensorGrammar,
    cache_residency_policy: DeepSeekV4CacheResidencyPolicy,
    kernel_bindings: Vec<DeepSeekV4KernelBinding>,
    protected_bridge_locations: Vec<DeepSeekV4ProtectedBridgeLocation>,
    capabilities: DeepSeekV4RuntimeBindingCapabilities,
}

impl DeepSeekV4RuntimeBinding {
    /// Bind a sidecar to the immutable identity already admitted by `reader`.
    /// This operation only reads reader metadata; it neither writes nor
    /// changes the sealed stream.
    pub fn bind(reader: &DeepSeekV4FullStreamReader) -> Result<Self> {
        let identity = identity_from_reader(reader)?;
        let fp8_pairs = reader.native_scale_pair_count_for(NativeScalePairKind::Fp8E4M3fn);
        let fp4_pairs = reader.native_scale_pair_count_for(NativeScalePairKind::Fp4E2M1fnX2);
        validate_native_pair_inventory(fp8_pairs, fp4_pairs, reader.native_scale_pair_count())?;

        Ok(Self::from_verified_identity(identity, fp8_pairs, fp4_pairs))
    }

    pub fn schema(&self) -> &'static str {
        self.schema
    }

    pub fn status(&self) -> &'static str {
        self.status
    }

    pub fn identity(&self) -> &DeepSeekV4RuntimeBindingIdentity {
        &self.identity
    }

    pub fn abi(&self) -> &DeepSeekV4RuntimeAbi {
        &self.abi
    }

    pub fn tensor_grammar(&self) -> &DeepSeekV4SourceTensorGrammar {
        &self.tensor_grammar
    }

    pub fn cache_residency_policy(&self) -> &DeepSeekV4CacheResidencyPolicy {
        &self.cache_residency_policy
    }

    pub fn kernel_bindings(&self) -> &[DeepSeekV4KernelBinding] {
        &self.kernel_bindings
    }

    pub fn protected_bridge_locations(&self) -> &[DeepSeekV4ProtectedBridgeLocation] {
        &self.protected_bridge_locations
    }

    pub fn capabilities(&self) -> &DeepSeekV4RuntimeBindingCapabilities {
        &self.capabilities
    }

    /// Require that a second already-admitted reader has precisely the same
    /// immutable identity and native representation inventory.
    pub fn verify_reader(&self, reader: &DeepSeekV4FullStreamReader) -> Result<()> {
        let observed = Self::bind(reader)?;
        if self.identity != observed.identity || self.tensor_grammar != observed.tensor_grammar {
            return Err(binding_error(
                "sealed reader identity or source-native representation grammar differs from this sidecar",
            ));
        }
        Ok(())
    }

    /// Re-admit `root` and compare it to this sidecar.  This is the only
    /// supported way to detect an on-disk manifest/restart/source mutation;
    /// it never writes, repairs, or promotes the artifact.
    pub fn verify_current_stream(&self, root: impl AsRef<Path>) -> Result<()> {
        let reader = DeepSeekV4FullStreamReader::admit(root)?;
        self.verify_reader(&reader)
    }

    /// Sidecars may never rewrite a sealed manifest or turn its explicitly
    /// non-runtime status into runtime readiness.  A future runtime needs a
    /// separate sealed receipt and public admission decision.
    pub fn reject_manifest_mutation_or_runtime_promotion(&self) -> Result<()> {
        Err(binding_error(
            "runtime-binding sidecars are immutable and cannot mutate or promote the sealed full-stream manifest",
        ))
    }

    fn from_verified_identity(
        identity: DeepSeekV4RuntimeBindingIdentity,
        fp8_pairs: usize,
        fp4_pairs: usize,
    ) -> Self {
        Self {
            schema: DSV4F_RUNTIME_BINDING_SCHEMA,
            status: DSV4F_RUNTIME_BINDING_STATUS,
            identity,
            abi: DeepSeekV4RuntimeAbi {
                identifier: DSV4F_RUNTIME_ABI,
                base_layer_count: DSV4F_BASE_LAYER_COUNT,
                excluded_mtp_layer_count: DSV4F_MTP_LAYER_COUNT,
                hidden_size: DSV4F_HIDDEN_SIZE,
                vocab_size: DSV4F_VOCAB_SIZE,
                routed_expert_count: DSV4F_ROUTED_EXPERT_COUNT,
                activated_experts_per_token: DSV4F_TOP_K_EXPERTS,
                causal_state_abi: "dsv4f.causal.kv_mhc_state.v1",
                token_alignment_abi: "dsv4f.token_alignment.source_token_index.v1",
            },
            tensor_grammar: DeepSeekV4SourceTensorGrammar {
                representations: vec![
                    DeepSeekV4SourceTensorRepresentation::Bf16LittleEndian,
                    DeepSeekV4SourceTensorRepresentation::F32LittleEndian,
                    DeepSeekV4SourceTensorRepresentation::I64LittleEndian,
                    DeepSeekV4SourceTensorRepresentation::Fp8E4M3fnWithUe8m0Block128,
                    DeepSeekV4SourceTensorRepresentation::Fp4E2M1fnX2WithUe8m0LogicalK32,
                ],
                fp8_control_or_shared_pair_count: fp8_pairs,
                fp4_routed_expert_pair_count: fp4_pairs,
            },
            cache_residency_policy: DeepSeekV4CacheResidencyPolicy {
                policy_identifier: "dsv4f.bounded_source_cache_residency.v1",
                source_parent_retention_permitted: false,
                all_model_materialization_permitted: false,
                maximum_single_host_stage_bytes: MAX_STAGED_OPERATOR_BYTES,
                provisional_control_resident_ceiling_bytes:
                    PROVISIONAL_CONTROL_RESIDENT_CEILING_BYTES,
                provisional_routed_expert_hot_ceiling_bytes:
                    PROVISIONAL_ROUTED_EXPERT_HOT_CEILING_BYTES,
                provisional_routed_expert_cold_ceiling_bytes:
                    PROVISIONAL_ROUTED_EXPERT_COLD_CEILING_BYTES,
                causal_kv_or_mhc_state_allocated: false,
            },
            kernel_bindings: kernel_bindings(),
            protected_bridge_locations: protected_bridge_locations(),
            capabilities: non_runtime_capabilities(),
        }
    }
}

fn identity_from_reader(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<DeepSeekV4RuntimeBindingIdentity> {
    let source = reader.source_identity();
    let identity = DeepSeekV4RuntimeBindingIdentity {
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
        model_config_json_sha256: reader
            .source_metadata_asset_sha256("config.json")?
            .to_owned(),
    };
    validate_identity(&identity)?;
    Ok(identity)
}

fn validate_identity(identity: &DeepSeekV4RuntimeBindingIdentity) -> Result<()> {
    if identity.repository != PINNED_REPOSITORY || identity.revision != PINNED_REVISION {
        return Err(binding_error(
            "reader repository/revision differs from the pinned DeepSeek-V4 source identity",
        ));
    }
    for (label, hash) in [
        ("manifest seal", identity.manifest_seal_sha256.as_str()),
        ("manifest file", identity.manifest_file_sha256.as_str()),
        ("restart receipt", identity.restart_seal_sha256.as_str()),
        (
            "inference/model.py",
            identity.inference_model_py_sha256.as_str(),
        ),
        (
            "inference/kernel.py",
            identity.inference_kernel_py_sha256.as_str(),
        ),
        (
            "inference/config.json",
            identity.inference_config_json_sha256.as_str(),
        ),
        ("config.json", identity.model_config_json_sha256.as_str()),
    ] {
        if !is_lower_sha256(hash) {
            return Err(binding_error(format!(
                "{label} hash is not a lowercase SHA-256"
            )));
        }
    }
    if identity.inference_model_py_sha256 != OFFICIAL_INFERENCE_MODEL_PY_SHA256
        || identity.inference_kernel_py_sha256 != OFFICIAL_INFERENCE_KERNEL_PY_SHA256
        || identity.inference_config_json_sha256 != OFFICIAL_INFERENCE_CONFIG_JSON_SHA256
        || identity.model_config_json_sha256 != OFFICIAL_MODEL_CONFIG_JSON_SHA256
    {
        return Err(binding_error(
            "reader source grammar hashes differ from the source-native runtime ABI anchors",
        ));
    }
    Ok(())
}

fn validate_native_pair_inventory(
    fp8_pairs: usize,
    fp4_pairs: usize,
    total_pairs: usize,
) -> Result<()> {
    if fp8_pairs == 0 || fp4_pairs == 0 || fp8_pairs.checked_add(fp4_pairs) != Some(total_pairs) {
        return Err(binding_error(
            "sealed reader lacks a complete native FP8/FP4 scale-pair grammar",
        ));
    }
    Ok(())
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn kernel_bindings() -> Vec<DeepSeekV4KernelBinding> {
    use DeepSeekV4KernelIdentifier as Id;
    use DeepSeekV4SourceTensorRepresentation as Repr;

    [
        (Id::Fp8ControlMatvec, Some(Repr::Fp8E4M3fnWithUe8m0Block128)),
        (
            Id::Fp4RoutedExpertMatvec,
            Some(Repr::Fp4E2M1fnX2WithUe8m0LogicalK32),
        ),
        (Id::GateAndTopK, Some(Repr::F32LittleEndian)),
        (Id::MhcState, Some(Repr::F32LittleEndian)),
        (
            Id::CompressedAttentionAndIndex,
            Some(Repr::Fp8E4M3fnWithUe8m0Block128),
        ),
        (Id::RouteGather, Some(Repr::I64LittleEndian)),
        (Id::SharedExpert, Some(Repr::Fp8E4M3fnWithUe8m0Block128)),
        (Id::RouteWeightedCombine, Some(Repr::F32LittleEndian)),
        (Id::LmHeadSampling, Some(Repr::Bf16LittleEndian)),
    ]
    .into_iter()
    .map(
        |(identifier, source_representation)| DeepSeekV4KernelBinding {
            identifier,
            source_representation,
            dispatchable: false,
        },
    )
    .collect()
}

fn protected_bridge_locations() -> Vec<DeepSeekV4ProtectedBridgeLocation> {
    use DeepSeekV4BridgeLayerScope as Scope;
    use DeepSeekV4ProtectedBridgeLocationId as Id;

    [
        (Id::PreNormHidden, Scope::EveryBaseLayer),
        (Id::PostAttentionHidden, Scope::EveryBaseLayer),
        (Id::PreRouterHidden, Scope::EveryBaseLayer),
        (Id::RouterLogits, Scope::EveryBaseLayer),
        (Id::RouteSelection, Scope::EveryBaseLayer),
        (Id::PostMoeHidden, Scope::EveryBaseLayer),
        (Id::MhcState, Scope::EveryBaseLayer),
        (Id::AttentionOrIndexState, Scope::EveryBaseLayer),
        (Id::FinalHidden, Scope::FinalBodyOnly),
        (Id::LmHeadLogits, Scope::FinalBodyOnly),
        (Id::ToolActionDecision, Scope::HcliEffectBoundary),
    ]
    .into_iter()
    .map(
        |(identifier, layer_scope)| DeepSeekV4ProtectedBridgeLocation {
            identifier,
            layer_scope,
            token_aligned: true,
            direct_weight_transplant_permitted: false,
            runtime_export_available: false,
        },
    )
    .collect()
}

fn non_runtime_capabilities() -> DeepSeekV4RuntimeBindingCapabilities {
    DeepSeekV4RuntimeBindingCapabilities {
        sealed_reader_identity_bound: true,
        source_native_grammar_bound: true,
        runtime_abi_declared: true,
        cache_residency_policy_declared: true,
        kernel_identifiers_declared: true,
        protected_bridge_locations_declared: true,
        manifest_mutation_permitted: false,
        manifest_runtime_promotion_permitted: false,
        engine_registered: false,
        causal_forward_available: false,
        metal_dispatches_available: false,
        hcli_endpoint_available: false,
        numeric_parity_v21_passed: false,
        base_true_tps_eligible: false,
    }
}

fn binding_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!("DeepSeek-V4 runtime binding: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_identity() -> DeepSeekV4RuntimeBindingIdentity {
        DeepSeekV4RuntimeBindingIdentity {
            repository: PINNED_REPOSITORY.to_owned(),
            revision: PINNED_REVISION.to_owned(),
            manifest_seal_sha256: "a".repeat(64),
            manifest_file_sha256: "b".repeat(64),
            restart_seal_sha256: "c".repeat(64),
            inference_model_py_sha256: OFFICIAL_INFERENCE_MODEL_PY_SHA256.to_owned(),
            inference_kernel_py_sha256: OFFICIAL_INFERENCE_KERNEL_PY_SHA256.to_owned(),
            inference_config_json_sha256: OFFICIAL_INFERENCE_CONFIG_JSON_SHA256.to_owned(),
            model_config_json_sha256: OFFICIAL_MODEL_CONFIG_JSON_SHA256.to_owned(),
        }
    }

    #[test]
    fn sidecar_declares_source_native_abi_without_a_runtime() {
        let binding = DeepSeekV4RuntimeBinding::from_verified_identity(valid_identity(), 7, 13);
        assert_eq!(binding.schema(), DSV4F_RUNTIME_BINDING_SCHEMA);
        assert_eq!(binding.status(), DSV4F_RUNTIME_BINDING_STATUS);
        assert_eq!(binding.abi().identifier, DSV4F_RUNTIME_ABI);
        assert_eq!(binding.abi().base_layer_count, DSV4F_BASE_LAYER_COUNT);
        assert_eq!(binding.tensor_grammar().fp8_control_or_shared_pair_count, 7);
        assert_eq!(binding.tensor_grammar().fp4_routed_expert_pair_count, 13);
        assert_eq!(binding.kernel_bindings().len(), 9);
        assert_eq!(binding.protected_bridge_locations().len(), 11);
        assert!(binding
            .kernel_bindings()
            .iter()
            .all(|kernel| !kernel.dispatchable));
        assert!(binding
            .protected_bridge_locations()
            .iter()
            .all(|location| !location.direct_weight_transplant_permitted));
    }

    #[test]
    fn identity_validation_fails_closed_on_manifest_or_source_drift() {
        let mut manifest_drift = valid_identity();
        manifest_drift.manifest_file_sha256.replace_range(0..1, "G");
        assert!(validate_identity(&manifest_drift).is_err());

        let mut source_drift = valid_identity();
        source_drift.inference_model_py_sha256 = "d".repeat(64);
        assert!(validate_identity(&source_drift).is_err());
    }

    #[test]
    fn native_pair_inventory_fails_closed_on_incomplete_grammar() {
        assert!(validate_native_pair_inventory(0, 1, 1).is_err());
        assert!(validate_native_pair_inventory(1, 0, 1).is_err());
        assert!(validate_native_pair_inventory(1, 1, 3).is_err());
        assert!(validate_native_pair_inventory(1, 1, 2).is_ok());
    }

    #[test]
    fn sidecar_refuses_manifest_promotion_and_runtime_claims() {
        let binding = DeepSeekV4RuntimeBinding::from_verified_identity(valid_identity(), 1, 1);
        let capabilities = binding.capabilities();
        assert!(capabilities.sealed_reader_identity_bound);
        assert!(!capabilities.manifest_mutation_permitted);
        assert!(!capabilities.manifest_runtime_promotion_permitted);
        assert!(!capabilities.engine_registered);
        assert!(!capabilities.causal_forward_available);
        assert!(!capabilities.metal_dispatches_available);
        assert!(!capabilities.hcli_endpoint_available);
        assert!(!capabilities.numeric_parity_v21_passed);
        assert!(!capabilities.base_true_tps_eligible);
        assert!(binding
            .reject_manifest_mutation_or_runtime_promotion()
            .is_err());
    }
}
