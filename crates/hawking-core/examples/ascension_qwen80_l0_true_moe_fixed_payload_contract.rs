//! Static, source-bound L0 true-MoE fixed-suffix payload contract for Qwen80.
//!
//! This is deliberately only a CPU-side description of future immutable
//! buffers and command-buffer bindings.  It does not open an artifact, create
//! a Metal context, register a shader, dispatch work, or promote component
//! evidence into a layer/token/HCLI/TPS claim.  The ten routed-expert payloads,
//! retained `first_residual`, and expected top-10 route witness remain the
//! responsibility of the existing typed all-ten route plan/bridge.

use serde::Serialize;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "hawking.ascension.qwen80_l0_true_moe_fixed_payload_contract.v1";
const STATUS: &str = "PREPARED_QWEN80_L0_TRUE_MOE_FIXED_SUFFIX_PAYLOAD_PLAN_NOT_EXECUTED";
const EXECUTION_STATUS: &str = "PREPARED_NOT_EXECUTED";

const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const CURRENT_MANIFEST_SEAL: &str =
    "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const CURRENT_MANIFEST_DOCUMENT_SHA256: &str =
    "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10";
const CURRENT_ADMISSION_RECEIPT_SEAL: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";
const SOURCE_BODY_AUDIT_SEAL: &str =
    "c572b2270b623b8677c374b43c89ddd729de135c25721488bb874b184ff8c3d4";
const SOURCE_REVALIDATION_SEAL: &str =
    "541b16fca1d4805ecba356face97b4e8de1accdeb21e98ee0c13b70ab0746c45";
const SOURCE_CONFIG_SHA256: &str =
    "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8";
const SOURCE_SHARD: &str = "model-00001-of-00040.safetensors";
const SOURCE_SHARD_SHA256: &str =
    "8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a";

const LAYER: u32 = 0;
const HIDDEN: u32 = 2_048;
const INTERMEDIATE: u32 = 512;
const EXPERTS: u32 = 512;
const TOP_K: u32 = 10;
const GROUP_SIZE: u32 = 128;
const RMS_EPSILON: &str = "1e-6";

const POST_NORM_NAME: &str = "model.layers.0.post_attention_layernorm.weight";
const ROUTER_NAME: &str = "model.layers.0.mlp.gate.weight";
const SHARED_GATE_NAME: &str = "model.layers.0.mlp.shared_expert.gate_proj.weight";
const SHARED_UP_NAME: &str = "model.layers.0.mlp.shared_expert.up_proj.weight";
const SHARED_DOWN_NAME: &str = "model.layers.0.mlp.shared_expert.down_proj.weight";
const SHARED_SCALAR_GATE_NAME: &str = "model.layers.0.mlp.shared_expert_gate.weight";

const POST_NORM_ARTIFACT_SHA256: &str =
    "a00ba60c88bd0d5dcf77e4c1fad05d83ddb6feec844ee3bbc65480fffd5a1fa7";
const ROUTER_ARTIFACT_SHA256: &str =
    "582725c1fa47c62b0f109216e8c2c40533b2931a583f4a41dfa34477deda45f4";
const SHARED_GATE_ARTIFACT_SHA256: &str =
    "92172dc4463a3a0610460ecf768427f6c9c8da04b43a73e904ca1fa36bc79aa6";
const SHARED_UP_ARTIFACT_SHA256: &str =
    "9d76293fa8abf4ccc2611d77386060671107e83dfd4458b5fddd5e345f24b4c4";
const SHARED_DOWN_ARTIFACT_SHA256: &str =
    "acf137a00b364f9c490e1282f18632465f05323b89903a5617162437b1ff500b";
const SHARED_SCALAR_GATE_ARTIFACT_SHA256: &str =
    "a40ff8a3f4e4b7e990a4672470cbd028b0c96b1cb15acd40aa3b8b2e2215096c";

const ROUTE_PLAN_SCHEMA: &str = "hawking.ascension.qwen80_all_ten_routed_expert_binding_plan.v1";
const ROUTE_PLAN_STATUS: &str =
    "SOURCE_BOUND_ALL_TEN_ROUTED_EXPERT_PLAN_READY_NOT_EXECUTED_NOT_COMBINED";
const SOURCE_TOKEN_ROUTE_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_all_ten_route_plan_authority.v1";
const SOURCE_TOKEN_ROUTE_AUTHORITY_STATUS: &str =
    "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_ALL_TEN_ROUTE_PLAN_READY_FOR_NEW_TYPED_BRIDGE";
const FIRST_RESIDUAL_SCHEMA: &str = "hawking.ascension.qwen80_first_residual_outer_capture.v1";
const FIRST_RESIDUAL_STATUS: &str = "CAPTURED_QWEN80_FIRST_RESIDUAL_STRICT_MATH_COMPONENT_ONLY";
const TYPED_BRIDGE_SCHEMA: &str = "hawking.ascension.qwen80_all_ten_true_moe_source_bridge.v1";
const TYPED_BRIDGE_STATUS: &str =
    "SEALED_CURRENT_ADMITTED_QWEN80_ALL_TEN_TRUE_MOE_SOURCE_BRIDGE_READY_FOR_DEVICE_LEASE";
const SOURCE_TOKEN_TYPED_BRIDGE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_all_ten_true_moe_source_bridge.v1";
const SOURCE_TOKEN_TYPED_BRIDGE_STATUS: &str =
    "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_SOURCE_BRIDGE_READY_FOR_OUTER_PREFLIGHT";

/// The suffix payloads and 14-dispatch ABI are invariant, but the exact
/// authority producing the ten route payloads is not.  Keep the historical
/// fixture route path separate from the token-1/zero-state path so a static
/// plan can never silently promote one into the other.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RouteAuthorityFamily {
    HistoricalFixture,
    SourceToken,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SourceBinding {
    model_id: &'static str,
    model_key: &'static str,
    source_repository: &'static str,
    source_revision: &'static str,
    source_config_sha256: &'static str,
    source_shard: &'static str,
    source_shard_sha256: &'static str,
    manifest_schema: &'static str,
    manifest_seal_sha256: &'static str,
    manifest_document_sha256: &'static str,
    admission_receipt_seal_sha256: &'static str,
    source_body_audit_seal_sha256: &'static str,
    source_revalidation_seal_sha256: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct Geometry {
    layer: u32,
    hidden: u32,
    intermediate: u32,
    experts: u32,
    top_k: u32,
    group_size: u32,
    rms_epsilon: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct CompactPayload {
    role: &'static str,
    tensor_name: &'static str,
    tensor_artifact_sha256: &'static str,
    source_shard: &'static str,
    source_shard_sha256: &'static str,
    shape: Vec<u32>,
    elements: u64,
    group_size: u32,
    group_count: u64,
    scale_element_type: &'static str,
    scale_bytes: u64,
    sign_element_type: &'static str,
    sign_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum BufferAuthority {
    FixedPayload,
    CommandScratch,
    ExternalAllTenRouteBridge,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct BufferAbi {
    name: &'static str,
    element_type: &'static str,
    shape: Vec<u32>,
    byte_len: u64,
    authority: BufferAuthority,
    purpose: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct BufferBinding {
    index: u32,
    name: &'static str,
    element_type: &'static str,
    shape: Vec<u32>,
    byte_len: u64,
    authority: BufferAuthority,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ScalarBinding {
    index: u32,
    element_type: &'static str,
    literal: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct DispatchAbi {
    ordinal: u32,
    kernel: &'static str,
    grid: [u32; 3],
    threadgroup: [u32; 3],
    purpose: &'static str,
    buffers: Vec<BufferBinding>,
    scalars: Vec<ScalarBinding>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ExternalRouteAuthority {
    route_plan_schema: &'static str,
    route_plan_status: &'static str,
    first_residual_schema: &'static str,
    first_residual_status: &'static str,
    typed_bridge_schema: &'static str,
    typed_bridge_status: &'static str,
    required_external_buffers: Vec<&'static str>,
    required_route_payload_buffers: Vec<&'static str>,
    route_payloads_materialized_here: bool,
    first_residual_materialized_here: bool,
    expected_topk_witness_materialized_here: bool,
    route_tensor_sha256s_materialized_here: bool,
    required_same_input_provenance: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ClaimBoundary {
    artifact_scan_or_payload_open_performed: bool,
    metal_context_or_dispatch_performed: bool,
    runtime_watcher_server_registry_or_hcli_changed: bool,
    token_or_tps_claim: bool,
    execution_status: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct FixedSuffixPayloadPlan {
    schema: &'static str,
    status: &'static str,
    source_binding: SourceBinding,
    geometry: Geometry,
    fixed_payloads: Vec<CompactPayload>,
    buffer_catalog: Vec<BufferAbi>,
    fixed_14_dispatch_abi: Vec<DispatchAbi>,
    external_authority: ExternalRouteAuthority,
    claim_boundary: ClaimBoundary,
}

fn source_binding() -> SourceBinding {
    SourceBinding {
        model_id: MODEL_ID,
        model_key: MODEL_KEY,
        source_repository: SOURCE_REPOSITORY,
        source_revision: SOURCE_REVISION,
        source_config_sha256: SOURCE_CONFIG_SHA256,
        source_shard: SOURCE_SHARD,
        source_shard_sha256: SOURCE_SHARD_SHA256,
        manifest_schema: MANIFEST_SCHEMA,
        manifest_seal_sha256: CURRENT_MANIFEST_SEAL,
        manifest_document_sha256: CURRENT_MANIFEST_DOCUMENT_SHA256,
        admission_receipt_seal_sha256: CURRENT_ADMISSION_RECEIPT_SEAL,
        source_body_audit_seal_sha256: SOURCE_BODY_AUDIT_SEAL,
        source_revalidation_seal_sha256: SOURCE_REVALIDATION_SEAL,
    }
}

fn geometry() -> Geometry {
    Geometry {
        layer: LAYER,
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        experts: EXPERTS,
        top_k: TOP_K,
        group_size: GROUP_SIZE,
        rms_epsilon: RMS_EPSILON,
    }
}

fn compact_payload(
    role: &'static str,
    tensor_name: &'static str,
    tensor_artifact_sha256: &'static str,
    shape: &[u32],
) -> CompactPayload {
    let elements = shape.iter().map(|&dim| u64::from(dim)).product::<u64>();
    assert_eq!(
        elements % u64::from(GROUP_SIZE),
        0,
        "{tensor_name} cannot be represented by group-{GROUP_SIZE} direct packing"
    );
    let group_count = elements / u64::from(GROUP_SIZE);
    CompactPayload {
        role,
        tensor_name,
        tensor_artifact_sha256,
        source_shard: SOURCE_SHARD,
        source_shard_sha256: SOURCE_SHARD_SHA256,
        shape: shape.to_vec(),
        elements,
        group_size: GROUP_SIZE,
        group_count,
        scale_element_type: "u16",
        scale_bytes: group_count * 2,
        sign_element_type: "u8",
        sign_bytes: elements / 8,
    }
}

fn fixed_payloads() -> Vec<CompactPayload> {
    vec![
        compact_payload(
            "post_attention_layernorm",
            POST_NORM_NAME,
            POST_NORM_ARTIFACT_SHA256,
            &[HIDDEN],
        ),
        compact_payload(
            "router_gate",
            ROUTER_NAME,
            ROUTER_ARTIFACT_SHA256,
            &[EXPERTS, HIDDEN],
        ),
        compact_payload(
            "shared_gate",
            SHARED_GATE_NAME,
            SHARED_GATE_ARTIFACT_SHA256,
            &[INTERMEDIATE, HIDDEN],
        ),
        compact_payload(
            "shared_up",
            SHARED_UP_NAME,
            SHARED_UP_ARTIFACT_SHA256,
            &[INTERMEDIATE, HIDDEN],
        ),
        compact_payload(
            "shared_down",
            SHARED_DOWN_NAME,
            SHARED_DOWN_ARTIFACT_SHA256,
            &[HIDDEN, INTERMEDIATE],
        ),
        compact_payload(
            "shared_expert_gate",
            SHARED_SCALAR_GATE_NAME,
            SHARED_SCALAR_GATE_ARTIFACT_SHA256,
            &[1, HIDDEN],
        ),
    ]
}

fn element_bytes(element_type: &str) -> u64 {
    match element_type {
        "u8" => 1,
        "u16" => 2,
        "u32" | "f32" => 4,
        _ => panic!("unsupported ABI element type {element_type}"),
    }
}

fn buffer(
    name: &'static str,
    element_type: &'static str,
    shape: &[u32],
    authority: BufferAuthority,
    purpose: &'static str,
) -> BufferAbi {
    let elements = shape.iter().map(|&dim| u64::from(dim)).product::<u64>();
    BufferAbi {
        name,
        element_type,
        shape: shape.to_vec(),
        byte_len: elements * element_bytes(element_type),
        authority,
        purpose,
    }
}

fn buffer_catalog() -> Vec<BufferAbi> {
    let vector_sign_bytes = HIDDEN / 8;
    let vector_groups = HIDDEN / GROUP_SIZE;
    let matrix_sign_bytes = INTERMEDIATE * HIDDEN / 8;
    let matrix_groups = INTERMEDIATE * HIDDEN / GROUP_SIZE;
    vec![
        buffer(
            "first_residual",
            "f32",
            &[HIDDEN],
            BufferAuthority::ExternalAllTenRouteBridge,
            "retained real [2048] DeltaNet first-residual buffer",
        ),
        buffer(
            "expected_route_ids",
            "u32",
            &[TOP_K],
            BufferAuthority::ExternalAllTenRouteBridge,
            "retained source selected top-10 route IDs",
        ),
        buffer(
            "expected_route_weights",
            "f32",
            &[TOP_K],
            BufferAuthority::ExternalAllTenRouteBridge,
            "retained source normalized top-10 route weights",
        ),
        buffer(
            "postnorm_signs",
            "u8",
            &[vector_sign_bytes],
            BufferAuthority::FixedPayload,
            "direct-packed post-attention RMSNorm signs",
        ),
        buffer(
            "postnorm_scales",
            "u16",
            &[vector_groups],
            BufferAuthority::FixedPayload,
            "direct-packed post-attention RMSNorm group scales",
        ),
        buffer(
            "postnorm_hidden",
            "f32",
            &[HIDDEN],
            BufferAuthority::CommandScratch,
            "post-attention RMSNorm output",
        ),
        buffer(
            "router_signs",
            "u8",
            &[matrix_sign_bytes],
            BufferAuthority::FixedPayload,
            "direct-packed router gate signs",
        ),
        buffer(
            "router_scales",
            "u16",
            &[matrix_groups],
            BufferAuthority::FixedPayload,
            "direct-packed router gate group scales",
        ),
        buffer(
            "router_logits",
            "f32",
            &[EXPERTS],
            BufferAuthority::CommandScratch,
            "all 512 router logits",
        ),
        buffer(
            "router_probabilities",
            "f32",
            &[EXPERTS],
            BufferAuthority::CommandScratch,
            "all 512 router probabilities",
        ),
        buffer(
            "router_route_ids",
            "u32",
            &[TOP_K],
            BufferAuthority::CommandScratch,
            "computed top-10 route IDs for the route guard",
        ),
        buffer(
            "router_route_weights",
            "f32",
            &[TOP_K],
            BufferAuthority::CommandScratch,
            "computed normalized top-10 route weights",
        ),
        buffer(
            "route_guard",
            "u32",
            &[1],
            BufferAuthority::CommandScratch,
            "all-ten external route-plan equality witness",
        ),
        buffer(
            "route_gate_signs",
            "u8",
            &[TOP_K, matrix_sign_bytes],
            BufferAuthority::ExternalAllTenRouteBridge,
            "external direct-packed gate signs for exactly ten planned experts",
        ),
        buffer(
            "route_gate_scales",
            "u16",
            &[TOP_K, matrix_groups],
            BufferAuthority::ExternalAllTenRouteBridge,
            "external direct-packed gate group scales for exactly ten planned experts",
        ),
        buffer(
            "route_up_signs",
            "u8",
            &[TOP_K, matrix_sign_bytes],
            BufferAuthority::ExternalAllTenRouteBridge,
            "external direct-packed up signs for exactly ten planned experts",
        ),
        buffer(
            "route_up_scales",
            "u16",
            &[TOP_K, matrix_groups],
            BufferAuthority::ExternalAllTenRouteBridge,
            "external direct-packed up group scales for exactly ten planned experts",
        ),
        buffer(
            "route_down_signs",
            "u8",
            &[TOP_K, matrix_sign_bytes],
            BufferAuthority::ExternalAllTenRouteBridge,
            "external direct-packed down signs for exactly ten planned experts",
        ),
        buffer(
            "route_down_scales",
            "u16",
            &[TOP_K, matrix_groups],
            BufferAuthority::ExternalAllTenRouteBridge,
            "external direct-packed down group scales for exactly ten planned experts",
        ),
        buffer(
            "route_gate",
            "f32",
            &[TOP_K, INTERMEDIATE],
            BufferAuthority::CommandScratch,
            "all-ten gate projection outputs",
        ),
        buffer(
            "route_up",
            "f32",
            &[TOP_K, INTERMEDIATE],
            BufferAuthority::CommandScratch,
            "all-ten up projection outputs",
        ),
        buffer(
            "route_activated",
            "f32",
            &[TOP_K, INTERMEDIATE],
            BufferAuthority::CommandScratch,
            "all-ten SiLU(gate) * up outputs",
        ),
        buffer(
            "route_weighted",
            "f32",
            &[TOP_K, HIDDEN],
            BufferAuthority::CommandScratch,
            "all-ten source-weighted down outputs",
        ),
        buffer(
            "shared_gate_signs",
            "u8",
            &[matrix_sign_bytes],
            BufferAuthority::FixedPayload,
            "direct-packed shared gate signs",
        ),
        buffer(
            "shared_gate_scales",
            "u16",
            &[matrix_groups],
            BufferAuthority::FixedPayload,
            "direct-packed shared gate group scales",
        ),
        buffer(
            "shared_up_signs",
            "u8",
            &[matrix_sign_bytes],
            BufferAuthority::FixedPayload,
            "direct-packed shared up signs",
        ),
        buffer(
            "shared_up_scales",
            "u16",
            &[matrix_groups],
            BufferAuthority::FixedPayload,
            "direct-packed shared up group scales",
        ),
        buffer(
            "shared_down_signs",
            "u8",
            &[matrix_sign_bytes],
            BufferAuthority::FixedPayload,
            "direct-packed shared down signs",
        ),
        buffer(
            "shared_down_scales",
            "u16",
            &[matrix_groups],
            BufferAuthority::FixedPayload,
            "direct-packed shared down group scales",
        ),
        buffer(
            "shared_scalar_signs",
            "u8",
            &[vector_sign_bytes],
            BufferAuthority::FixedPayload,
            "direct-packed shared scalar gate signs",
        ),
        buffer(
            "shared_scalar_scales",
            "u16",
            &[vector_groups],
            BufferAuthority::FixedPayload,
            "direct-packed shared scalar gate group scales",
        ),
        buffer(
            "shared_gate",
            "f32",
            &[INTERMEDIATE],
            BufferAuthority::CommandScratch,
            "shared gate projection output",
        ),
        buffer(
            "shared_up",
            "f32",
            &[INTERMEDIATE],
            BufferAuthority::CommandScratch,
            "shared up projection output",
        ),
        buffer(
            "shared_activated",
            "f32",
            &[INTERMEDIATE],
            BufferAuthority::CommandScratch,
            "shared SiLU(gate) * up output",
        ),
        buffer(
            "shared_output",
            "f32",
            &[HIDDEN],
            BufferAuthority::CommandScratch,
            "shared down projection output",
        ),
        buffer(
            "shared_scalar_logit",
            "f32",
            &[1],
            BufferAuthority::CommandScratch,
            "shared expert scalar gate logit",
        ),
        buffer(
            "gated_shared",
            "f32",
            &[HIDDEN],
            BufferAuthority::CommandScratch,
            "sigmoid-gated shared expert output",
        ),
        buffer(
            "routed_sum",
            "f32",
            &[HIDDEN],
            BufferAuthority::CommandScratch,
            "fixed source-order sum of ten route outputs",
        ),
        buffer(
            "second_residual",
            "f32",
            &[HIDDEN],
            BufferAuthority::CommandScratch,
            "routed sum + gated shared + retained first residual",
        ),
    ]
}

fn bound(catalog: &[BufferAbi], index: u32, name: &'static str) -> BufferBinding {
    let buffer = catalog
        .iter()
        .find(|buffer| buffer.name == name)
        .unwrap_or_else(|| panic!("missing buffer ABI for {name}"));
    BufferBinding {
        index,
        name: buffer.name,
        element_type: buffer.element_type,
        shape: buffer.shape.clone(),
        byte_len: buffer.byte_len,
        authority: buffer.authority,
    }
}

fn u32_scalar(index: u32, literal: &'static str) -> ScalarBinding {
    ScalarBinding {
        index,
        element_type: "u32",
        literal,
    }
}

fn f32_scalar(index: u32, literal: &'static str) -> ScalarBinding {
    ScalarBinding {
        index,
        element_type: "f32",
        literal,
    }
}

fn dispatch(
    ordinal: u32,
    kernel: &'static str,
    grid: [u32; 3],
    threadgroup: [u32; 3],
    purpose: &'static str,
    buffers: Vec<BufferBinding>,
    scalars: Vec<ScalarBinding>,
) -> DispatchAbi {
    DispatchAbi {
        ordinal,
        kernel,
        grid,
        threadgroup,
        purpose,
        buffers,
        scalars,
    }
}

fn fixed_14_dispatch_abi(catalog: &[BufferAbi]) -> Vec<DispatchAbi> {
    vec![
        dispatch(
            1,
            "qwen80_postnorm_router_top10_rmsnorm",
            [256, 1, 1],
            [256, 1, 1],
            "post-attention RMSNorm(first_residual) -> postnorm_hidden",
            vec![
                bound(catalog, 0, "first_residual"),
                bound(catalog, 1, "postnorm_signs"),
                bound(catalog, 2, "postnorm_scales"),
                bound(catalog, 3, "postnorm_hidden"),
            ],
            vec![
                u32_scalar(4, "2048"),
                u32_scalar(5, "128"),
                f32_scalar(6, "1e-6"),
            ],
        ),
        dispatch(
            2,
            "qwen80_postnorm_router_top10_matvec",
            [256, EXPERTS, 1],
            [256, 1, 1],
            "direct-packed router gate -> all 512 logits",
            vec![
                bound(catalog, 0, "router_signs"),
                bound(catalog, 1, "router_scales"),
                bound(catalog, 2, "postnorm_hidden"),
                bound(catalog, 3, "router_logits"),
            ],
            vec![
                u32_scalar(4, "512"),
                u32_scalar(5, "2048"),
                u32_scalar(6, "128"),
            ],
        ),
        dispatch(
            3,
            "qwen80_postnorm_router_top10_select",
            [1, 1, 1],
            [1, 1, 1],
            "source tie-policy top-10 IDs and normalized weights",
            vec![
                bound(catalog, 0, "router_logits"),
                bound(catalog, 1, "router_probabilities"),
                bound(catalog, 2, "router_route_ids"),
                bound(catalog, 3, "router_route_weights"),
            ],
            vec![
                u32_scalar(4, "512"),
                u32_scalar(5, "10"),
                f32_scalar(6, "0.0"),
            ],
        ),
        dispatch(
            4,
            "qwen80_all_ten_routed_wave_route_guard",
            [1, 1, 1],
            [1, 1, 1],
            "reject a router result that differs from the retained all-ten plan",
            vec![
                bound(catalog, 0, "router_route_ids"),
                bound(catalog, 1, "router_route_weights"),
                bound(catalog, 2, "expected_route_ids"),
                bound(catalog, 3, "expected_route_weights"),
                bound(catalog, 4, "route_guard"),
            ],
            vec![u32_scalar(5, "10"), f32_scalar(6, "2e-5")],
        ),
        dispatch(
            5,
            "qwen80_all_ten_routed_wave_gate_up",
            [256, INTERMEDIATE, TOP_K],
            [256, 1, 1],
            "all ten external direct-packed gate/up projections, route in Z",
            vec![
                bound(catalog, 0, "route_gate_signs"),
                bound(catalog, 1, "route_gate_scales"),
                bound(catalog, 2, "route_up_signs"),
                bound(catalog, 3, "route_up_scales"),
                bound(catalog, 4, "postnorm_hidden"),
                bound(catalog, 5, "route_gate"),
                bound(catalog, 6, "route_up"),
            ],
            vec![
                u32_scalar(7, "10"),
                u32_scalar(8, "512"),
                u32_scalar(9, "2048"),
                u32_scalar(10, "128"),
            ],
        ),
        dispatch(
            6,
            "qwen80_all_ten_routed_wave_swiglu",
            [INTERMEDIATE, TOP_K, 1],
            [256, 1, 1],
            "all ten SiLU(gate) * up activations",
            vec![
                bound(catalog, 0, "route_gate"),
                bound(catalog, 1, "route_up"),
                bound(catalog, 2, "route_activated"),
            ],
            vec![u32_scalar(3, "10"), u32_scalar(4, "512")],
        ),
        dispatch(
            7,
            "qwen80_all_ten_routed_wave_down_weighted",
            [256, HIDDEN, TOP_K],
            [256, 1, 1],
            "all ten external direct-packed down projections and source route weights",
            vec![
                bound(catalog, 0, "route_down_signs"),
                bound(catalog, 1, "route_down_scales"),
                bound(catalog, 2, "route_activated"),
                bound(catalog, 3, "router_route_weights"),
                bound(catalog, 4, "route_weighted"),
            ],
            vec![
                u32_scalar(5, "10"),
                u32_scalar(6, "2048"),
                u32_scalar(7, "512"),
                u32_scalar(8, "128"),
            ],
        ),
        dispatch(
            8,
            "qwen80_shared_expert_wave_gate_up",
            [256, INTERMEDIATE, 1],
            [256, 1, 1],
            "direct-packed fixed shared gate/up projections",
            vec![
                bound(catalog, 0, "shared_gate_signs"),
                bound(catalog, 1, "shared_gate_scales"),
                bound(catalog, 2, "shared_up_signs"),
                bound(catalog, 3, "shared_up_scales"),
                bound(catalog, 4, "postnorm_hidden"),
                bound(catalog, 5, "shared_gate"),
                bound(catalog, 6, "shared_up"),
            ],
            vec![
                u32_scalar(7, "512"),
                u32_scalar(8, "2048"),
                u32_scalar(9, "128"),
            ],
        ),
        dispatch(
            9,
            "qwen80_shared_expert_wave_swiglu",
            [INTERMEDIATE, 1, 1],
            [256, 1, 1],
            "fixed shared SiLU(gate) * up activation",
            vec![
                bound(catalog, 0, "shared_gate"),
                bound(catalog, 1, "shared_up"),
                bound(catalog, 2, "shared_activated"),
            ],
            vec![u32_scalar(3, "512")],
        ),
        dispatch(
            10,
            "qwen80_shared_expert_wave_down",
            [256, HIDDEN, 1],
            [256, 1, 1],
            "direct-packed fixed shared down projection",
            vec![
                bound(catalog, 0, "shared_down_signs"),
                bound(catalog, 1, "shared_down_scales"),
                bound(catalog, 2, "shared_activated"),
                bound(catalog, 3, "shared_output"),
            ],
            vec![
                u32_scalar(4, "2048"),
                u32_scalar(5, "512"),
                u32_scalar(6, "128"),
            ],
        ),
        dispatch(
            11,
            "qwen80_shared_expert_wave_scalar_gate",
            [256, 1, 1],
            [256, 1, 1],
            "direct-packed fixed shared scalar gate",
            vec![
                bound(catalog, 0, "shared_scalar_signs"),
                bound(catalog, 1, "shared_scalar_scales"),
                bound(catalog, 2, "postnorm_hidden"),
                bound(catalog, 3, "shared_scalar_logit"),
            ],
            vec![u32_scalar(4, "2048"), u32_scalar(5, "128")],
        ),
        dispatch(
            12,
            "qwen80_shared_expert_wave_apply_sigmoid_gate",
            [HIDDEN, 1, 1],
            [256, 1, 1],
            "sigmoid-gated fixed shared output",
            vec![
                bound(catalog, 0, "shared_output"),
                bound(catalog, 1, "shared_scalar_logit"),
                bound(catalog, 2, "gated_shared"),
            ],
            vec![u32_scalar(3, "2048")],
        ),
        dispatch(
            13,
            "qwen80_moe_wave_aggregate_second_residual_route_sum",
            [HIDDEN, 1, 1],
            [256, 1, 1],
            "fixed source-order sum of route[0] through route[9]",
            vec![
                bound(catalog, 0, "route_weighted"),
                bound(catalog, 1, "routed_sum"),
            ],
            vec![u32_scalar(2, "10"), u32_scalar(3, "2048")],
        ),
        dispatch(
            14,
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
            [HIDDEN, 1, 1],
            [256, 1, 1],
            "routed sum + fixed shared result + externally retained first residual",
            vec![
                bound(catalog, 0, "routed_sum"),
                bound(catalog, 1, "gated_shared"),
                bound(catalog, 2, "first_residual"),
                bound(catalog, 3, "second_residual"),
            ],
            vec![u32_scalar(4, "2048")],
        ),
    ]
}

fn external_authority_for(family: RouteAuthorityFamily) -> ExternalRouteAuthority {
    let (route_plan_schema, route_plan_status, typed_bridge_schema, typed_bridge_status, provenance) =
        match family {
            RouteAuthorityFamily::HistoricalFixture => (
                ROUTE_PLAN_SCHEMA,
                ROUTE_PLAN_STATUS,
                TYPED_BRIDGE_SCHEMA,
                TYPED_BRIDGE_STATUS,
                "typed bridge must bind retained [2048] first_residual, route plan, and future command graph to one real input",
            ),
            RouteAuthorityFamily::SourceToken => (
                SOURCE_TOKEN_ROUTE_AUTHORITY_SCHEMA,
                SOURCE_TOKEN_ROUTE_AUTHORITY_STATUS,
                SOURCE_TOKEN_TYPED_BRIDGE_SCHEMA,
                SOURCE_TOKEN_TYPED_BRIDGE_STATUS,
                "source-token typed bridge must bind token 1, zero L0 DeltaNet state, retained [2048] first_residual, exact all-ten authority, and future command graph to one real input",
            ),
        };
    ExternalRouteAuthority {
        route_plan_schema,
        route_plan_status,
        first_residual_schema: FIRST_RESIDUAL_SCHEMA,
        first_residual_status: FIRST_RESIDUAL_STATUS,
        typed_bridge_schema,
        typed_bridge_status,
        required_external_buffers: vec![
            "first_residual",
            "expected_route_ids",
            "expected_route_weights",
            "route_gate_signs",
            "route_gate_scales",
            "route_up_signs",
            "route_up_scales",
            "route_down_signs",
            "route_down_scales",
        ],
        required_route_payload_buffers: vec![
            "route_gate_signs",
            "route_gate_scales",
            "route_up_signs",
            "route_up_scales",
            "route_down_signs",
            "route_down_scales",
        ],
        route_payloads_materialized_here: false,
        first_residual_materialized_here: false,
        expected_topk_witness_materialized_here: false,
        route_tensor_sha256s_materialized_here: false,
        required_same_input_provenance: provenance,
    }
}

fn claim_boundary() -> ClaimBoundary {
    ClaimBoundary {
        artifact_scan_or_payload_open_performed: false,
        metal_context_or_dispatch_performed: false,
        runtime_watcher_server_registry_or_hcli_changed: false,
        token_or_tps_claim: false,
        execution_status: EXECUTION_STATUS,
    }
}

fn prepared_plan_for(family: RouteAuthorityFamily) -> FixedSuffixPayloadPlan {
    let buffer_catalog = buffer_catalog();
    FixedSuffixPayloadPlan {
        schema: SCHEMA,
        status: STATUS,
        source_binding: source_binding(),
        geometry: geometry(),
        fixed_payloads: fixed_payloads(),
        fixed_14_dispatch_abi: fixed_14_dispatch_abi(&buffer_catalog),
        buffer_catalog,
        external_authority: external_authority_for(family),
        claim_boundary: claim_boundary(),
    }
}

fn prepared_plan() -> FixedSuffixPayloadPlan {
    prepared_plan_for(RouteAuthorityFamily::HistoricalFixture)
}

fn validate_plan_for(
    plan: &FixedSuffixPayloadPlan,
    family: RouteAuthorityFamily,
) -> Result<(), String> {
    if plan.schema != SCHEMA || plan.status != STATUS {
        return Err("fixed payload contract schema/status drifted".into());
    }
    if plan.source_binding != source_binding() {
        return Err("fixed payload source/manifest/admission binding drifted".into());
    }
    if plan.geometry != geometry() {
        return Err("fixed payload geometry drifted".into());
    }
    if plan.fixed_payloads != fixed_payloads() {
        return Err("fixed payload tensor shape, SHA, or compact byte geometry drifted".into());
    }

    let expected_catalog = buffer_catalog();
    if plan.buffer_catalog != expected_catalog {
        return Err("fixed payload buffer catalog drifted".into());
    }
    if plan.fixed_14_dispatch_abi != fixed_14_dispatch_abi(&expected_catalog) {
        return Err("fixed 14-dispatch buffer ABI/order drifted".into());
    }
    if plan.external_authority != external_authority_for(family) {
        return Err("external all-ten route authority drifted".into());
    }
    if plan.claim_boundary != claim_boundary() {
        return Err("contract must remain prepared/not-executed only".into());
    }

    for dispatch in &plan.fixed_14_dispatch_abi {
        for (expected_index, binding) in dispatch.buffers.iter().enumerate() {
            if binding.index != expected_index as u32 {
                return Err(format!(
                    "{} buffer {} occupies {}, expected {}",
                    dispatch.kernel, binding.name, binding.index, expected_index
                ));
            }
            let Some(catalog) = plan
                .buffer_catalog
                .iter()
                .find(|catalog| catalog.name == binding.name)
            else {
                return Err(format!(
                    "{} references absent buffer {}",
                    dispatch.kernel, binding.name
                ));
            };
            if catalog.element_type != binding.element_type
                || catalog.shape != binding.shape
                || catalog.byte_len != binding.byte_len
                || catalog.authority != binding.authority
            {
                return Err(format!(
                    "{} binding {} diverges from catalog geometry",
                    dispatch.kernel, binding.name
                ));
            }
        }
    }

    for route_payload in &plan.external_authority.required_route_payload_buffers {
        let Some(buffer) = plan
            .buffer_catalog
            .iter()
            .find(|buffer| buffer.name == *route_payload)
        else {
            return Err(format!(
                "required external route payload {route_payload} is absent"
            ));
        };
        if buffer.authority != BufferAuthority::ExternalAllTenRouteBridge {
            return Err(format!(
                "route payload {route_payload} cannot be promoted into fixed payload authority"
            ));
        }
    }

    if plan
        .fixed_payloads
        .iter()
        .any(|payload| payload.role.starts_with("route_"))
    {
        return Err("routed-expert payloads cannot join the fixed suffix contract".into());
    }
    Ok(())
}

fn validate_plan(plan: &FixedSuffixPayloadPlan) -> Result<(), String> {
    validate_plan_for(plan, RouteAuthorityFamily::HistoricalFixture)
}

#[derive(Debug)]
enum OutputMode {
    Stdout,
    NewAbsoluteFile(PathBuf),
}

fn parse_output_mode(arguments: &[String]) -> Result<(OutputMode, RouteAuthorityFamily), String> {
    let mut family = RouteAuthorityFamily::HistoricalFixture;
    let mut print_plan = false;
    let mut output_path: Option<PathBuf> = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--source-token" => {
                if family == RouteAuthorityFamily::SourceToken {
                    return Err("--source-token was repeated".into());
                }
                family = RouteAuthorityFamily::SourceToken;
                index += 1;
            }
            "--print-plan" => {
                if print_plan {
                    return Err("--print-plan was repeated".into());
                }
                print_plan = true;
                index += 1;
            }
            "--out" => {
                if output_path.is_some() {
                    return Err("--out was repeated".into());
                }
                let raw_path = arguments.get(index + 1).ok_or("--out requires a value")?;
                let path = PathBuf::from(raw_path);
                if !path.is_absolute() {
                    return Err("--out must name an absolute new file".into());
                }
                let Some(parent) = path.parent() else {
                    return Err("--out path has no parent directory".into());
                };
                if !parent.is_dir() {
                    return Err("--out parent directory must already exist".into());
                }
                if path.exists() {
                    return Err("--out refuses to overwrite an existing file".into());
                }
                output_path = Some(path);
                index += 2;
            }
            value => {
                return Err(format!(
                    "unsupported argument {value:?}; usage: ascension_qwen80_l0_true_moe_fixed_payload_contract [--source-token] [--print-plan | --out NEW_ABSOLUTE_FILE]"
                ))
            }
        }
    }
    if print_plan && output_path.is_some() {
        return Err("--print-plan and --out are mutually exclusive".into());
    }
    Ok((
        output_path.map_or(OutputMode::Stdout, OutputMode::NewAbsoluteFile),
        family,
    ))
}

fn raw_plan_bytes(plan: &FixedSuffixPayloadPlan) -> Result<Vec<u8>, String> {
    let mut bytes = serde_json::to_vec_pretty(plan)
        .map_err(|error| format!("static Qwen80 L0 plan serialization failed: {error}"))?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Writes an unsealed static authority document. `create_new` is intentional:
/// a future lease must bind this immutable document rather than allowing an
/// earlier prepared plan to be replaced in place.
fn write_new_raw_plan(path: &Path, plan: &FixedSuffixPayloadPlan) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("raw plan path must be absolute".into());
    }
    let bytes = raw_plan_bytes(plan)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create new raw plan {}: {error}", path.display()))?;
    output
        .write_all(&bytes)
        .map_err(|error| format!("cannot write raw plan {}: {error}", path.display()))?;
    output
        .sync_all()
        .map_err(|error| format!("cannot sync raw plan {}: {error}", path.display()))?;
    Ok(())
}

fn main() {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let (output_mode, family) = parse_output_mode(&arguments).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });

    let plan = prepared_plan_for(family);
    validate_plan_for(&plan, family)
        .expect("static Qwen80 L0 fixed payload contract must validate");
    match output_mode {
        OutputMode::Stdout => print!(
            "{}",
            String::from_utf8(raw_plan_bytes(&plan).expect("static Qwen80 L0 plan must serialize"))
                .expect("static Qwen80 L0 plan JSON must be UTF-8")
        ),
        OutputMode::NewAbsoluteFile(path) => {
            write_new_raw_plan(&path, &plan)
                .unwrap_or_else(|error| panic!("cannot emit prepared raw plan: {error}"));
            println!("{}", path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named_buffer<'a>(plan: &'a FixedSuffixPayloadPlan, name: &str) -> &'a BufferAbi {
        plan.buffer_catalog
            .iter()
            .find(|buffer| buffer.name == name)
            .unwrap_or_else(|| panic!("missing {name}"))
    }

    #[test]
    fn prepared_plan_is_source_bound_and_not_executed() {
        let plan = prepared_plan();
        validate_plan(&plan).unwrap();
        assert_eq!(plan.schema, SCHEMA);
        assert_eq!(plan.status, STATUS);
        assert_eq!(plan.claim_boundary.execution_status, EXECUTION_STATUS);
        assert!(!plan.claim_boundary.artifact_scan_or_payload_open_performed);
        assert!(!plan.claim_boundary.metal_context_or_dispatch_performed);
        assert!(!plan.claim_boundary.token_or_tps_claim);
    }

    #[test]
    fn six_fixed_payloads_have_exact_group128_compact_geometries() {
        let plan = prepared_plan();
        assert_eq!(plan.fixed_payloads.len(), 6);
        assert_eq!(
            plan.fixed_payloads
                .iter()
                .map(|payload| payload.sign_bytes)
                .sum::<u64>(),
            524_800
        );
        assert_eq!(
            plan.fixed_payloads
                .iter()
                .map(|payload| payload.scale_bytes)
                .sum::<u64>(),
            65_600
        );
        for payload in &plan.fixed_payloads {
            assert_eq!(payload.group_size, GROUP_SIZE);
            assert_eq!(payload.elements / GROUP_SIZE as u64, payload.group_count);
            assert_eq!(payload.scale_bytes, payload.group_count * 2);
            assert_eq!(payload.sign_bytes, payload.elements / 8);
            assert_eq!(payload.source_shard_sha256, SOURCE_SHARD_SHA256);
        }
        assert_eq!(plan.fixed_payloads[0].shape, vec![HIDDEN]);
        assert_eq!(plan.fixed_payloads[1].shape, vec![EXPERTS, HIDDEN]);
        assert_eq!(plan.fixed_payloads[5].shape, vec![1, HIDDEN]);
    }

    #[test]
    fn all_non_route_f32_and_u32_buffer_geometries_are_explicit() {
        let plan = prepared_plan();
        let f32_or_u32 = plan
            .buffer_catalog
            .iter()
            .filter(|buffer| matches!(buffer.element_type, "f32" | "u32"))
            .collect::<Vec<_>>();
        assert_eq!(f32_or_u32.len(), 21);
        assert_eq!(named_buffer(&plan, "first_residual").shape, vec![HIDDEN]);
        assert_eq!(named_buffer(&plan, "first_residual").byte_len, 8_192);
        assert_eq!(named_buffer(&plan, "router_logits").shape, vec![EXPERTS]);
        assert_eq!(named_buffer(&plan, "router_route_ids").shape, vec![TOP_K]);
        assert_eq!(
            named_buffer(&plan, "route_weighted").shape,
            vec![TOP_K, HIDDEN]
        );
        assert_eq!(named_buffer(&plan, "route_weighted").byte_len, 81_920);
        assert_eq!(named_buffer(&plan, "shared_scalar_logit").byte_len, 4);
        assert_eq!(named_buffer(&plan, "second_residual").shape, vec![HIDDEN]);
    }

    #[test]
    fn exact_fourteen_dispatches_and_key_binding_positions_are_retained() {
        let plan = prepared_plan();
        assert_eq!(plan.fixed_14_dispatch_abi.len(), 14);
        let names = plan
            .fixed_14_dispatch_abi
            .iter()
            .map(|dispatch| dispatch.kernel)
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "qwen80_postnorm_router_top10_rmsnorm",
                "qwen80_postnorm_router_top10_matvec",
                "qwen80_postnorm_router_top10_select",
                "qwen80_all_ten_routed_wave_route_guard",
                "qwen80_all_ten_routed_wave_gate_up",
                "qwen80_all_ten_routed_wave_swiglu",
                "qwen80_all_ten_routed_wave_down_weighted",
                "qwen80_shared_expert_wave_gate_up",
                "qwen80_shared_expert_wave_swiglu",
                "qwen80_shared_expert_wave_down",
                "qwen80_shared_expert_wave_scalar_gate",
                "qwen80_shared_expert_wave_apply_sigmoid_gate",
                "qwen80_moe_wave_aggregate_second_residual_route_sum",
                "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
            ]
        );
        assert_eq!(
            plan.fixed_14_dispatch_abi[0].buffers[0].name,
            "first_residual"
        );
        assert_eq!(
            plan.fixed_14_dispatch_abi[3].buffers[2].name,
            "expected_route_ids"
        );
        assert_eq!(
            plan.fixed_14_dispatch_abi[4].buffers[0].authority,
            BufferAuthority::ExternalAllTenRouteBridge
        );
        assert_eq!(
            plan.fixed_14_dispatch_abi[7].buffers[0].name,
            "shared_gate_signs"
        );
        assert_eq!(
            plan.fixed_14_dispatch_abi[13].buffers[3].name,
            "second_residual"
        );
    }

    #[test]
    fn validator_rejects_compact_payload_byte_drift() {
        let mut plan = prepared_plan();
        plan.fixed_payloads[1].sign_bytes += 1;
        let error = validate_plan(&plan).unwrap_err();
        assert!(error.contains("compact byte geometry"));
    }

    #[test]
    fn validator_rejects_external_route_payload_promotion() {
        let mut plan = prepared_plan();
        let route_gate_signs = plan
            .buffer_catalog
            .iter_mut()
            .find(|buffer| buffer.name == "route_gate_signs")
            .unwrap();
        route_gate_signs.authority = BufferAuthority::FixedPayload;
        let error = validate_plan(&plan).unwrap_err();
        assert!(error.contains("buffer catalog"));
    }

    #[test]
    fn validator_rejects_dispatch_order_drift() {
        let mut plan = prepared_plan();
        plan.fixed_14_dispatch_abi.swap(0, 1);
        let error = validate_plan(&plan).unwrap_err();
        assert!(error.contains("14-dispatch"));
    }

    #[test]
    fn external_authority_remains_unmaterialized_and_same_input_bound() {
        let plan = prepared_plan();
        let authority = &plan.external_authority;
        assert_eq!(authority.route_plan_schema, ROUTE_PLAN_SCHEMA);
        assert_eq!(authority.first_residual_schema, FIRST_RESIDUAL_SCHEMA);
        assert_eq!(authority.typed_bridge_schema, TYPED_BRIDGE_SCHEMA);
        assert!(!authority.route_payloads_materialized_here);
        assert!(!authority.first_residual_materialized_here);
        assert!(!authority.expected_topk_witness_materialized_here);
        assert!(!authority.route_tensor_sha256s_materialized_here);
        assert!(authority
            .required_same_input_provenance
            .contains("one real input"));
    }

    #[test]
    fn source_token_variant_retains_exact_suffix_but_refuses_legacy_authority_family() {
        let source_token = prepared_plan_for(RouteAuthorityFamily::SourceToken);
        validate_plan_for(&source_token, RouteAuthorityFamily::SourceToken).unwrap();
        assert_eq!(source_token.fixed_payloads, prepared_plan().fixed_payloads);
        assert_eq!(
            source_token.fixed_14_dispatch_abi,
            prepared_plan().fixed_14_dispatch_abi
        );
        assert_eq!(
            source_token.external_authority.route_plan_schema,
            SOURCE_TOKEN_ROUTE_AUTHORITY_SCHEMA
        );
        assert_eq!(
            source_token.external_authority.typed_bridge_schema,
            SOURCE_TOKEN_TYPED_BRIDGE_SCHEMA
        );
        assert!(validate_plan(&source_token).is_err());
        assert!(source_token
            .external_authority
            .required_same_input_provenance
            .contains("token 1"));
    }

    #[test]
    fn output_mode_only_accepts_stdout_or_a_new_absolute_path() {
        assert!(matches!(
            parse_output_mode(&[]),
            Ok((OutputMode::Stdout, RouteAuthorityFamily::HistoricalFixture))
        ));
        assert!(matches!(
            parse_output_mode(&["--print-plan".into()]),
            Ok((OutputMode::Stdout, RouteAuthorityFamily::HistoricalFixture))
        ));
        assert!(matches!(
            parse_output_mode(&["--source-token".into()]),
            Ok((OutputMode::Stdout, RouteAuthorityFamily::SourceToken))
        ));
        assert!(parse_output_mode(&["--out".into(), "relative.json".into()]).is_err());
        assert!(parse_output_mode(&["--other".into()]).is_err());

        let directory = tempfile::tempdir().unwrap();
        let new_path = directory.path().join("fixed-payload-plan.json");
        let (mode, family) =
            parse_output_mode(&["--out".into(), new_path.to_string_lossy().into_owned()]).unwrap();
        assert!(matches!(mode, OutputMode::NewAbsoluteFile(path) if path == new_path));
        assert_eq!(family, RouteAuthorityFamily::HistoricalFixture);

        let source_path = directory
            .path()
            .join("source-token-fixed-payload-plan.json");
        let (mode, family) = parse_output_mode(&[
            "--source-token".into(),
            "--out".into(),
            source_path.to_string_lossy().into_owned(),
        ])
        .unwrap();
        assert!(matches!(mode, OutputMode::NewAbsoluteFile(path) if path == source_path));
        assert_eq!(family, RouteAuthorityFamily::SourceToken);

        let missing_parent = directory.path().join("absent-parent").join("plan.json");
        let error = parse_output_mode(&[
            "--out".into(),
            missing_parent.to_string_lossy().into_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("parent directory"));

        std::fs::write(&new_path, b"existing").unwrap();
        let error = parse_output_mode(&["--out".into(), new_path.to_string_lossy().into_owned()])
            .unwrap_err();
        assert!(error.contains("refuses to overwrite"));
    }

    #[test]
    fn raw_plan_output_replays_exact_static_authority_and_never_overwrites() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("fixed-payload-plan.json");
        let plan = prepared_plan();
        let expected_bytes = raw_plan_bytes(&plan).unwrap();
        write_new_raw_plan(&output, &plan).unwrap();

        let written = std::fs::read(&output).unwrap();
        assert_eq!(written, expected_bytes);
        let replay: serde_json::Value = serde_json::from_slice(&written).unwrap();
        assert_eq!(replay["schema"], SCHEMA);
        assert_eq!(replay["status"], STATUS);
        assert_eq!(
            replay["source_binding"]["manifest_seal_sha256"],
            CURRENT_MANIFEST_SEAL
        );
        assert_eq!(replay["fixed_payloads"].as_array().unwrap().len(), 6);
        assert_eq!(
            replay["fixed_14_dispatch_abi"].as_array().unwrap().len(),
            14
        );
        assert_eq!(
            replay["claim_boundary"]["execution_status"],
            EXECUTION_STATUS
        );

        let error = write_new_raw_plan(&output, &plan).unwrap_err();
        assert!(error.contains("cannot create new raw plan"));
    }
}
