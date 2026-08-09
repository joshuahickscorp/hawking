//! Static/CPU-only Qwen80 GQA K/V-cache device preflight.
//!
//! This is deliberately a *one-slot preflight*, not a Qwen80 token, layer,
//! decoder, or device result.  It binds the source-shaped GQA cache geometry
//! used by the existing decode-state and state-buffer-layout contracts:
//!
//! * source GQA layers `3, 7, ..., 47` own slots `0..12` without aliases;
//! * each K and V slot has shape `[max_seq_len, 2, 256]`;
//! * a current K/V row is appended before a causal read/mask is allowed; and
//! * a rollback snapshot restores the selected active slot exactly.
//!
//! No model artifact is opened.  No Metal context, buffer, encoder, dispatch,
//! runtime, watcher, server, or registry is touched.  The accompanying MSL
//! source is intentionally unregistered and is inspected as static ABI text
//! only by this target's CPU tests.

use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_gqa_kv_cache_device_preflight.v1";
const RESULT_STATUS: &str =
    "PREPARED_INCOMPLETE_QWEN80_GQA_KV_CACHE_APPEND_CAUSAL_MASK_ROLLBACK_DEVICE_PREFLIGHT";
const LEASE_VALIDATED_STATUS: &str =
    "PREPARED_INCOMPLETE_QWEN80_GQA_KV_CACHE_COMPONENT_LEASE_VALIDATED_NO_METAL";
const CPU_ORACLE_READBACK_SCHEMA: &str =
    "hawking.ascension.qwen80_gqa_kv_cache_component_cpu_oracle_readback.v1";
const COMPONENT_CHILD_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen80_gqa_kv_cache_component_child_readback_parity.v1";
const COMPONENT_CHILD_STATUS: &str =
    "PREPARED_INCOMPLETE_QWEN80_GQA_KV_CACHE_COMPONENT_CHILD_CPU_READBACK_PARITY_NOT_DEVICE_EXECUTED";
const SOURCE_HIDDEN_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen80_source_bound_hidden_input_authority.v1";
const SOURCE_HIDDEN_AUTHORITY_STATUS: &str =
    "EARNED_QWEN80_SOURCE_BOUND_HIDDEN_INPUT_COMPONENT_AUTHORITY";
const FUTURE_METAL_LEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_gqa_kv_cache_component_quiet_lease.v1";
const FUTURE_METAL_LEASE_STATUS: &str =
    "GRANTED_QWEN80_GQA_KV_CACHE_APPEND_CAUSAL_READ_ROLLBACK_COMPONENT_ONLY_NON_TIMED_LEASE";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const DECODE_STATE_CONTRACT_SCHEMA: &str = "hawking.ascension.qwen80_decode_state_contract.v1";
const STATE_LAYOUT_CONTRACT_SCHEMA: &str =
    "hawking.ascension.qwen80_device_state_buffer_layout_contract.v1";
const GQA_PER_LAYER_CONTRACT_SCHEMA: &str =
    "hawking.ascension.qwen80_gqa_per_layer_device_readiness_result.v1";

const GQA_LAYER_COUNT: usize = 12;
const GQA_KV_HEADS: usize = 2;
const GQA_HEAD_DIM: usize = 256;
const KV_ROW_ELEMENTS: usize = GQA_KV_HEADS * GQA_HEAD_DIM;
const HIDDEN_SIZE: usize = 2_048;
const QUERY_HEADS: usize = 16;
const QUERY_DIM: usize = QUERY_HEADS * GQA_HEAD_DIM;
const Q_PROJ_ROWS: usize = QUERY_DIM * 2;
const K_PROJ_ROWS: usize = KV_ROW_ELEMENTS;
const V_PROJ_ROWS: usize = KV_ROW_ELEMENTS;
const O_PROJ_ROWS: usize = HIDDEN_SIZE;
const O_PROJ_COLS: usize = QUERY_DIM;
const DIRECT_PACKED_GROUP_SIZE: usize = 128;
const F32_BYTES: usize = std::mem::size_of::<f32>();
const MAX_NATIVE_CONTEXT: usize = 4_096;
const PREFLIGHT_CONTEXT: usize = 2;
const SELECTED_GQA_LAYER: usize = 3;
const SELECTED_GQA_SLOT: usize = 0;

const SHADER_SOURCE: &str = include_str!("../shaders/qwen80_gqa_kv_cache_preflight.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Fresh,
    Appended,
    CausalMaskRead,
    RolledBack,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct GqaSlotBinding {
    layer: usize,
    slot: usize,
    key_shape: [usize; 3],
    value_shape: [usize; 3],
}

/// A source-shaped direct-packed projection address.  This is an ABI contract
/// only: it deliberately does not contain a payload path, payload bytes, or a
/// decoded BF16 shadow.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct CompactDirectPackedPayloadAbi {
    role: &'static str,
    tensor_name: &'static str,
    shape: [usize; 2],
    group_size: usize,
    compact_payload_format: &'static str,
    direct_packed_only: bool,
    bf16_shadow_allowed: bool,
    projection_output_elements: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct KvCacheDomainGeometry {
    domain: &'static str,
    key_buffer_role: &'static str,
    value_buffer_role: &'static str,
    layer: usize,
    slot: usize,
    shape: [usize; 3],
    elements_per_buffer: usize,
    bytes_per_buffer: usize,
    total_key_value_bytes: usize,
}

/// Per logical session allocation geometry for one attention layer / one GQA
/// state slot.  The names are logical ABI names only; no device buffer exists
/// in this static target.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct PerSessionKvArenaGeometry {
    session_id: &'static str,
    selected_layer: usize,
    selected_slot: usize,
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    active: KvCacheDomainGeometry,
    rollback: KvCacheDomainGeometry,
    active_and_rollback_disjoint: bool,
    total_session_bytes: usize,
    maximum_native_context_total_session_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ExpectedReadbackBuffer {
    name: &'static str,
    elements: usize,
    element_type: &'static str,
    provenance: &'static str,
}

/// The CPU-only ledger a future host controller must emit before accepting a
/// device readback.  Its values describe a synthetic state oracle, not a
/// compact model execution or an artifact read.
#[derive(Clone, Debug, Serialize)]
struct CpuOracleReadbackLedger {
    schema: &'static str,
    status: &'static str,
    component_only: bool,
    selected_layer: usize,
    selected_slot: usize,
    session_geometry: PerSessionKvArenaGeometry,
    direct_packed_projection_abi: Vec<CompactDirectPackedPayloadAbi>,
    operation_order: Vec<&'static str>,
    expected_readback_buffers: Vec<ExpectedReadbackBuffer>,
    synthetic_active_key_checksum: f64,
    synthetic_active_value_checksum: f64,
    causal_mask_current_position: f32,
    causal_mask_future_position: f32,
    rollback_restored_key: bool,
    rollback_restored_value: bool,
    artifact_payload_opened: bool,
    device_or_metal_invoked: bool,
    direct_packed_projection_executed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    CpuPreflight,
    FutureMetalComponent,
    ComponentChild,
}

#[derive(Clone, Debug)]
struct Arguments {
    mode: Mode,
    lease_receipt: Option<PathBuf>,
}

/// A future upstream must construct this from a sealed, source-bound hidden
/// state.  The component deliberately has no way to manufacture this binding
/// from a prompt, tokenizer, artifact, or model runtime.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SourceBoundHiddenAuthority {
    schema: String,
    status: String,
    authority_seal_sha256: String,
    model_id: String,
    source_repository: String,
    source_revision: String,
    source_layer: usize,
    source_gqa_slot: usize,
    hidden_width: usize,
    session_id: String,
    token_position: usize,
}

/// A caller-owned source-shaped `[2048]` hidden state.  It is intentionally
/// borrowed: this component never owns, synthesizes, or persists upstream
/// model activations.
#[derive(Debug)]
struct SourceBoundHiddenInput<'a> {
    values: &'a [f32; HIDDEN_SIZE],
    authority: SourceBoundHiddenAuthority,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ComponentTcbCommand {
    DirectPackedQProjection,
    DirectPackedKProjection,
    DirectPackedVProjection,
    SnapshotActiveKvToRollback,
    AppendKeyAtCurrentPosition,
    AppendValueAtCurrentPosition,
    CausalMaskIncludingCurrentPosition,
    CausalReadIncludingCurrentPosition,
    DirectPackedOProjection,
    CopyReadbackParityLedger,
    RollbackActiveKvFromCallerOwnedSnapshot,
}

/// A minimal host-side stand-in for a real upstream TokenCommandBuffer.  The
/// caller creates and owns it; this child only appends a validated, ordered
/// component ABI.  `device_dispatch_permitted` remains false in every path in
/// this standalone target.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct CallerOwnedTcb {
    tcb_id: String,
    owner_id: String,
    session_id: String,
    submitted: bool,
    device_dispatch_permitted: bool,
    commands: Vec<ComponentTcbCommand>,
}

impl CallerOwnedTcb {
    fn cpu_contract(tcb_id: String, owner_id: String, session_id: String) -> Self {
        Self {
            tcb_id,
            owner_id,
            session_id,
            submitted: false,
            device_dispatch_permitted: false,
            commands: Vec::new(),
        }
    }

    fn encode_all(&mut self, commands: &[ComponentTcbCommand]) {
        self.commands.extend_from_slice(commands);
    }
}

/// Four caller-owned cache slices, one selected layer/slot only.  Borrowed
/// mutable slices make aliasing the active and rollback domains impossible at
/// the Rust call site; the explicit handle names are checked as a second ABI
/// guard for future FFI callers.
struct CallerOwnedSessionKvBuffers<'a> {
    session_id: &'a str,
    selected_layer: usize,
    selected_slot: usize,
    max_seq_len: usize,
    active_key_handle: &'a str,
    active_value_handle: &'a str,
    rollback_key_handle: &'a str,
    rollback_value_handle: &'a str,
    active_key: &'a mut [f32],
    active_value: &'a mut [f32],
    rollback_key: &'a mut [f32],
    rollback_value: &'a mut [f32],
}

/// Caller-provided CPU-oracle vectors.  They may be used by a future host to
/// compare a device readback, but this component neither opens a compact
/// payload nor computes a projection itself.
struct CpuOracleProjectionReadback<'a> {
    q_projection_rows: &'a [f32],
    k_projection_row: &'a [f32],
    v_projection_row: &'a [f32],
    o_projection_output: &'a [f32],
    provenance: &'a str,
}

#[derive(Clone, Debug, Serialize)]
struct ReadbackParityVector {
    name: &'static str,
    elements: usize,
    f32le_sha256: String,
    max_abs_error: f32,
    exact: bool,
    provenance: &'static str,
}

/// Strict component-only receipt returned after a CPU state transition and
/// rollback.  It proves only host ABI/order/readback parity, never a layer,
/// token, decoder, or native Metal result.
#[derive(Clone, Debug, Serialize)]
struct ComponentChildReadbackParityReceipt {
    schema: &'static str,
    status: &'static str,
    component_only: bool,
    selected_layer: usize,
    selected_slot: usize,
    session_id: String,
    token_position: usize,
    source_hidden_f32le_sha256: String,
    source_hidden_authority: SourceBoundHiddenAuthority,
    lease: ValidatedFutureMetalLease,
    caller_owned_tcb_id: String,
    caller_owned_tcb_owner_id: String,
    single_caller_owned_tcb: bool,
    tcb_command_order: Vec<ComponentTcbCommand>,
    direct_packed_projection_abi: Vec<CompactDirectPackedPayloadAbi>,
    caller_owned_buffer_handles: Vec<String>,
    cache_readback_parity: Vec<ReadbackParityVector>,
    rollback_restored_active_key: bool,
    rollback_restored_active_value: bool,
    artifact_payload_opened: bool,
    metal_context_or_dispatch: bool,
    projection_kernel_executed: bool,
    causal_attention_executed: bool,
    decoder_or_token_execution: bool,
}

#[derive(Clone, Debug, Serialize)]
struct CpuPreflightWitness {
    selected_layer: usize,
    selected_slot: usize,
    position: usize,
    max_seq_len: usize,
    key_row_elements: usize,
    value_row_elements: usize,
    append_before_causal_read_enforced: bool,
    causal_mask_current_position: f32,
    causal_mask_future_position: f32,
    rollback_restored_key: bool,
    rollback_restored_value: bool,
    phase: Phase,
}

#[derive(Clone, Debug)]
struct OneSlotKvPreflight {
    max_seq_len: usize,
    position: usize,
    phase: Phase,
    active_key: Vec<f32>,
    active_value: Vec<f32>,
    rollback_key: Vec<f32>,
    rollback_value: Vec<f32>,
}

impl OneSlotKvPreflight {
    fn new(max_seq_len: usize, position: usize) -> Result<Self, String> {
        if !(2..=MAX_NATIVE_CONTEXT).contains(&max_seq_len) {
            return Err(format!(
                "one-slot K/V preflight context {max_seq_len} must be in 2..={MAX_NATIVE_CONTEXT}"
            ));
        }
        if position >= max_seq_len || position + 1 >= max_seq_len {
            return Err(
                "one-slot K/V preflight requires one current and one future position".into(),
            );
        }
        let elements = max_seq_len
            .checked_mul(KV_ROW_ELEMENTS)
            .ok_or_else(|| "one-slot K/V cache element count overflowed".to_owned())?;
        Ok(Self {
            max_seq_len,
            position,
            phase: Phase::Fresh,
            active_key: vec![0.0; elements],
            active_value: vec![0.0; elements],
            rollback_key: vec![0.0; elements],
            rollback_value: vec![0.0; elements],
        })
    }

    fn row_range(&self) -> std::ops::Range<usize> {
        let start = self.position * KV_ROW_ELEMENTS;
        start..start + KV_ROW_ELEMENTS
    }

    fn append_current_before_causal_read(
        &mut self,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), String> {
        if self.phase != Phase::Fresh {
            return Err("K/V append may occur exactly once before causal read".into());
        }
        if key.len() != KV_ROW_ELEMENTS || value.len() != KV_ROW_ELEMENTS {
            return Err("source-shaped K/V append row geometry drifted".into());
        }
        if key.iter().chain(value).any(|value| !value.is_finite()) {
            return Err("source-shaped K/V append refuses non-finite values".into());
        }
        self.rollback_key.copy_from_slice(&self.active_key);
        self.rollback_value.copy_from_slice(&self.active_value);
        let row = self.row_range();
        self.active_key[row.clone()].copy_from_slice(key);
        self.active_value[row].copy_from_slice(value);
        self.phase = Phase::Appended;
        Ok(())
    }

    fn causal_mask_after_append(&mut self) -> Result<Vec<f32>, String> {
        if self.phase != Phase::Appended {
            return Err("causal read/mask is forbidden until current K/V append completes".into());
        }
        let mask = (0..self.max_seq_len)
            .map(|index| {
                if index <= self.position {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect::<Vec<_>>();
        self.phase = Phase::CausalMaskRead;
        Ok(mask)
    }

    fn rollback_after_mask(&mut self) -> Result<(), String> {
        if self.phase != Phase::CausalMaskRead {
            return Err("rollback requires an append-before-causal-read witness".into());
        }
        self.active_key.copy_from_slice(&self.rollback_key);
        self.active_value.copy_from_slice(&self.rollback_value);
        self.phase = Phase::RolledBack;
        Ok(())
    }
}

fn gqa_bindings(max_seq_len: usize) -> Result<Vec<GqaSlotBinding>, String> {
    if !(2..=MAX_NATIVE_CONTEXT).contains(&max_seq_len) {
        return Err("GQA state-layout context is out of source contract range".into());
    }
    let bindings = (0..GQA_LAYER_COUNT)
        .map(|slot| GqaSlotBinding {
            layer: slot * 4 + 3,
            slot,
            key_shape: [max_seq_len, GQA_KV_HEADS, GQA_HEAD_DIM],
            value_shape: [max_seq_len, GQA_KV_HEADS, GQA_HEAD_DIM],
        })
        .collect::<Vec<_>>();
    if bindings
        .iter()
        .enumerate()
        .any(|(slot, binding)| binding.slot != slot || binding.layer != slot * 4 + 3)
    {
        return Err("GQA 12-slot source schedule drifted".into());
    }
    Ok(bindings)
}

fn synthetic_source_shaped_row(salt: u32) -> Vec<f32> {
    (0..KV_ROW_ELEMENTS)
        .map(|index| ((index as u32 ^ salt) % 97) as f32 / 97.0)
        .collect()
}

fn compact_direct_packed_projection_abi() -> Vec<CompactDirectPackedPayloadAbi> {
    const FORMAT: &str = "direct_binary_sign_bits_plus_fp16_group_scales";
    vec![
        CompactDirectPackedPayloadAbi {
            role: "q_projection_query_and_gate_rows",
            tensor_name: "model.layers.3.self_attn.q_proj.weight",
            shape: [Q_PROJ_ROWS, HIDDEN_SIZE],
            group_size: DIRECT_PACKED_GROUP_SIZE,
            compact_payload_format: FORMAT,
            direct_packed_only: true,
            bf16_shadow_allowed: false,
            projection_output_elements: Q_PROJ_ROWS,
        },
        CompactDirectPackedPayloadAbi {
            role: "k_projection",
            tensor_name: "model.layers.3.self_attn.k_proj.weight",
            shape: [K_PROJ_ROWS, HIDDEN_SIZE],
            group_size: DIRECT_PACKED_GROUP_SIZE,
            compact_payload_format: FORMAT,
            direct_packed_only: true,
            bf16_shadow_allowed: false,
            projection_output_elements: K_PROJ_ROWS,
        },
        CompactDirectPackedPayloadAbi {
            role: "v_projection",
            tensor_name: "model.layers.3.self_attn.v_proj.weight",
            shape: [V_PROJ_ROWS, HIDDEN_SIZE],
            group_size: DIRECT_PACKED_GROUP_SIZE,
            compact_payload_format: FORMAT,
            direct_packed_only: true,
            bf16_shadow_allowed: false,
            projection_output_elements: V_PROJ_ROWS,
        },
        CompactDirectPackedPayloadAbi {
            role: "o_projection",
            tensor_name: "model.layers.3.self_attn.o_proj.weight",
            shape: [O_PROJ_ROWS, O_PROJ_COLS],
            group_size: DIRECT_PACKED_GROUP_SIZE,
            compact_payload_format: FORMAT,
            direct_packed_only: true,
            bf16_shadow_allowed: false,
            projection_output_elements: O_PROJ_ROWS,
        },
    ]
}

fn kv_cache_domain_geometry(
    domain: &'static str,
    key_buffer_role: &'static str,
    value_buffer_role: &'static str,
    max_seq_len: usize,
) -> Result<KvCacheDomainGeometry, String> {
    let elements_per_buffer = max_seq_len
        .checked_mul(KV_ROW_ELEMENTS)
        .ok_or_else(|| "K/V cache geometry element count overflowed".to_owned())?;
    let bytes_per_buffer = elements_per_buffer
        .checked_mul(F32_BYTES)
        .ok_or_else(|| "K/V cache geometry byte count overflowed".to_owned())?;
    let total_key_value_bytes = bytes_per_buffer
        .checked_mul(2)
        .ok_or_else(|| "K/V cache key/value byte count overflowed".to_owned())?;
    Ok(KvCacheDomainGeometry {
        domain,
        key_buffer_role,
        value_buffer_role,
        layer: SELECTED_GQA_LAYER,
        slot: SELECTED_GQA_SLOT,
        shape: [max_seq_len, GQA_KV_HEADS, GQA_HEAD_DIM],
        elements_per_buffer,
        bytes_per_buffer,
        total_key_value_bytes,
    })
}

fn per_session_kv_arena_geometry(max_seq_len: usize) -> Result<PerSessionKvArenaGeometry, String> {
    if !(2..=MAX_NATIVE_CONTEXT).contains(&max_seq_len) {
        return Err("per-session K/V cache geometry is out of source contract range".into());
    }
    let active = kv_cache_domain_geometry(
        "active",
        "session.active.layer3.slot0.key",
        "session.active.layer3.slot0.value",
        max_seq_len,
    )?;
    let rollback = kv_cache_domain_geometry(
        "rollback",
        "session.rollback.layer3.slot0.key",
        "session.rollback.layer3.slot0.value",
        max_seq_len,
    )?;
    if active.key_buffer_role == rollback.key_buffer_role
        || active.value_buffer_role == rollback.value_buffer_role
        || active.domain == rollback.domain
    {
        return Err("active and rollback K/V cache domains must be disjoint".into());
    }
    let total_session_bytes = active
        .total_key_value_bytes
        .checked_add(rollback.total_key_value_bytes)
        .ok_or_else(|| "per-session active/rollback K/V byte count overflowed".to_owned())?;
    let maximum_native_context_total_session_bytes = MAX_NATIVE_CONTEXT
        .checked_mul(KV_ROW_ELEMENTS)
        .and_then(|elements| elements.checked_mul(F32_BYTES))
        .and_then(|per_buffer| per_buffer.checked_mul(4))
        .ok_or_else(|| "maximum-context active/rollback K/V byte count overflowed".to_owned())?;
    Ok(PerSessionKvArenaGeometry {
        session_id: "synthetic-qwen80-gqa-kv-preflight-session",
        selected_layer: SELECTED_GQA_LAYER,
        selected_slot: SELECTED_GQA_SLOT,
        kv_heads: GQA_KV_HEADS,
        head_dim: GQA_HEAD_DIM,
        max_seq_len,
        active,
        rollback,
        active_and_rollback_disjoint: true,
        total_session_bytes,
        maximum_native_context_total_session_bytes,
    })
}

fn expected_readback_buffers() -> Vec<ExpectedReadbackBuffer> {
    vec![
        ExpectedReadbackBuffer {
            name: "q_projection_rows",
            elements: Q_PROJ_ROWS,
            element_type: "f32",
            provenance: "future_direct_packed_component_readback_only",
        },
        ExpectedReadbackBuffer {
            name: "k_projection_row",
            elements: K_PROJ_ROWS,
            element_type: "f32",
            provenance: "future_direct_packed_component_readback_only",
        },
        ExpectedReadbackBuffer {
            name: "v_projection_row",
            elements: V_PROJ_ROWS,
            element_type: "f32",
            provenance: "future_direct_packed_component_readback_only",
        },
        ExpectedReadbackBuffer {
            name: "o_projection_output",
            elements: O_PROJ_ROWS,
            element_type: "f32",
            provenance: "future_direct_packed_component_readback_only",
        },
        ExpectedReadbackBuffer {
            name: "active_key_slot_row",
            elements: KV_ROW_ELEMENTS,
            element_type: "f32",
            provenance: "synthetic_cpu_state_oracle_before_any_device_execution",
        },
        ExpectedReadbackBuffer {
            name: "active_value_slot_row",
            elements: KV_ROW_ELEMENTS,
            element_type: "f32",
            provenance: "synthetic_cpu_state_oracle_before_any_device_execution",
        },
        ExpectedReadbackBuffer {
            name: "causal_mask",
            elements: PREFLIGHT_CONTEXT,
            element_type: "f32",
            provenance: "synthetic_cpu_state_oracle_before_any_device_execution",
        },
    ]
}

fn f32le_sha256(values: &[f32]) -> Result<String, String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err("readback parity refuses non-finite f32 values".into());
    }
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn max_abs_error(reference: &[f32], observed: &[f32]) -> Result<f32, String> {
    if reference.len() != observed.len() {
        return Err("readback parity vectors have different lengths".into());
    }
    if reference
        .iter()
        .chain(observed)
        .any(|value| !value.is_finite())
    {
        return Err("readback parity refuses non-finite f32 values".into());
    }
    Ok(reference
        .iter()
        .zip(observed)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max))
}

fn lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| {
            byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
        })
}

fn exact_component_command_order() -> Vec<ComponentTcbCommand> {
    vec![
        ComponentTcbCommand::DirectPackedQProjection,
        ComponentTcbCommand::DirectPackedKProjection,
        ComponentTcbCommand::DirectPackedVProjection,
        ComponentTcbCommand::SnapshotActiveKvToRollback,
        ComponentTcbCommand::AppendKeyAtCurrentPosition,
        ComponentTcbCommand::AppendValueAtCurrentPosition,
        ComponentTcbCommand::CausalMaskIncludingCurrentPosition,
        ComponentTcbCommand::CausalReadIncludingCurrentPosition,
        ComponentTcbCommand::DirectPackedOProjection,
        ComponentTcbCommand::CopyReadbackParityLedger,
        ComponentTcbCommand::RollbackActiveKvFromCallerOwnedSnapshot,
    ]
}

fn validate_exact_direct_packed_qkvo_abi(
    observed: &[CompactDirectPackedPayloadAbi],
) -> Result<(), String> {
    let expected = compact_direct_packed_projection_abi();
    if observed != expected.as_slice() {
        return Err("component child requires the exact layer-3 direct-packed Q/K/V/O ABI".into());
    }
    Ok(())
}

fn validate_source_bound_hidden_input(
    input: &SourceBoundHiddenInput<'_>,
    expected_session_id: &str,
    expected_position: usize,
) -> Result<(), String> {
    let authority = &input.authority;
    if authority.schema != SOURCE_HIDDEN_AUTHORITY_SCHEMA
        || authority.status != SOURCE_HIDDEN_AUTHORITY_STATUS
        || !lowercase_sha256(&authority.authority_seal_sha256)
        || authority.model_id != MODEL_ID
        || authority.source_repository != SOURCE_REPOSITORY
        || authority.source_revision != SOURCE_REVISION
        || authority.source_layer != SELECTED_GQA_LAYER
        || authority.source_gqa_slot != SELECTED_GQA_SLOT
        || authority.hidden_width != HIDDEN_SIZE
        || authority.session_id != expected_session_id
        || authority.token_position != expected_position
    {
        return Err(
            "source-bound hidden input authority is not exact for layer3/slot0/session/position"
                .into(),
        );
    }
    if input.values.iter().any(|value| !value.is_finite()) {
        return Err("source-bound [2048] hidden input contains a non-finite value".into());
    }
    Ok(())
}

fn validate_caller_owned_tcb(tcb: &CallerOwnedTcb, session_id: &str) -> Result<(), String> {
    if tcb.tcb_id.is_empty()
        || tcb.owner_id.is_empty()
        || tcb.session_id != session_id
        || tcb.submitted
        || tcb.device_dispatch_permitted
    {
        return Err(
            "component child requires an unsubmitted CPU-contract caller-owned TCB for the exact session"
                .into(),
        );
    }
    Ok(())
}

fn validate_caller_owned_kv_buffers(
    buffers: &CallerOwnedSessionKvBuffers<'_>,
    expected_position: usize,
) -> Result<(), String> {
    if buffers.session_id.is_empty()
        || buffers.selected_layer != SELECTED_GQA_LAYER
        || buffers.selected_slot != SELECTED_GQA_SLOT
        || !(2..=MAX_NATIVE_CONTEXT).contains(&buffers.max_seq_len)
        || expected_position >= buffers.max_seq_len
        || expected_position + 1 >= buffers.max_seq_len
    {
        return Err("caller-owned K/V buffer layer/slot/context/position binding drifted".into());
    }
    let expected_elements = buffers
        .max_seq_len
        .checked_mul(KV_ROW_ELEMENTS)
        .ok_or_else(|| "caller-owned K/V buffer element count overflowed".to_owned())?;
    for (label, values) in [
        ("active_key", &*buffers.active_key),
        ("active_value", &*buffers.active_value),
        ("rollback_key", &*buffers.rollback_key),
        ("rollback_value", &*buffers.rollback_value),
    ] {
        if values.len() != expected_elements {
            return Err(format!(
                "caller-owned {label} has incorrect source-shaped geometry"
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("caller-owned {label} contains non-finite state"));
        }
    }
    let handles = [
        buffers.active_key_handle,
        buffers.active_value_handle,
        buffers.rollback_key_handle,
        buffers.rollback_value_handle,
    ];
    if handles.iter().any(|handle| handle.is_empty())
        || handles
            .iter()
            .enumerate()
            .any(|(index, handle)| handles[..index].contains(handle))
    {
        return Err(
            "caller-owned active/rollback K/V buffer handles must be non-empty and disjoint".into(),
        );
    }
    Ok(())
}

fn validate_cpu_oracle_projection_readback(
    oracle: &CpuOracleProjectionReadback<'_>,
) -> Result<(), String> {
    if oracle.provenance != "caller_cpu_oracle_direct_packed_component_readback" {
        return Err(
            "CPU oracle readback provenance is not the strict component-only authority".into(),
        );
    }
    let expected = [
        ("q_projection_rows", oracle.q_projection_rows, Q_PROJ_ROWS),
        ("k_projection_row", oracle.k_projection_row, K_PROJ_ROWS),
        ("v_projection_row", oracle.v_projection_row, V_PROJ_ROWS),
        (
            "o_projection_output",
            oracle.o_projection_output,
            O_PROJ_ROWS,
        ),
    ];
    for (label, values, elements) in expected {
        if values.len() != elements {
            return Err(format!(
                "CPU oracle {label} length {} != {elements}",
                values.len()
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("CPU oracle {label} contains non-finite values"));
        }
    }
    Ok(())
}

fn readback_vector(
    name: &'static str,
    reference: &[f32],
    observed: &[f32],
    provenance: &'static str,
) -> Result<ReadbackParityVector, String> {
    let max_abs_error = max_abs_error(reference, observed)?;
    Ok(ReadbackParityVector {
        name,
        elements: observed.len(),
        f32le_sha256: f32le_sha256(observed)?,
        max_abs_error,
        exact: max_abs_error == 0.0,
        provenance,
    })
}

/// Encode a stateful, component-only GQA cache transition into exactly one
/// caller-owned TCB.  The caller owns all activations, cache slices, rollback
/// slices, and source/lease authority.  This CPU contract mutates the cache
/// solely to prove append-before-read and rollback readback parity; it never
/// opens a compact payload, executes a projection, creates a Metal context,
/// or submits a command buffer.
fn encode_component_child(
    tcb: &mut CallerOwnedTcb,
    hidden: &SourceBoundHiddenInput<'_>,
    buffers: &mut CallerOwnedSessionKvBuffers<'_>,
    direct_packed_projection_abi: &[CompactDirectPackedPayloadAbi],
    oracle: &CpuOracleProjectionReadback<'_>,
    lease: &ValidatedFutureMetalLease,
) -> Result<ComponentChildReadbackParityReceipt, String> {
    let position = hidden.authority.token_position;
    validate_source_bound_hidden_input(hidden, buffers.session_id, position)?;
    validate_caller_owned_tcb(tcb, buffers.session_id)?;
    validate_caller_owned_kv_buffers(buffers, position)?;
    validate_cpu_oracle_projection_readback(oracle)?;
    validate_exact_direct_packed_qkvo_abi(direct_packed_projection_abi)?;
    if lease.component != "qwen80_gqa_kv_cache_append_causal_read_rollback"
        || lease.selected_layer != SELECTED_GQA_LAYER
        || lease.selected_slot != SELECTED_GQA_SLOT
        || lease.future_metal_invoked
    {
        return Err(
            "validated future lease does not authorize this exact non-dispatched component ABI"
                .into(),
        );
    }

    let expected_elements = buffers.max_seq_len * KV_ROW_ELEMENTS;
    let active_key_before = f32le_sha256(buffers.active_key)?;
    let active_value_before = f32le_sha256(buffers.active_value)?;
    let row_start = position * KV_ROW_ELEMENTS;
    let row_end = row_start + KV_ROW_ELEMENTS;
    let row = row_start..row_end;

    // Snapshot/append/read/rollback operate directly on caller-owned slices.
    // The component creates no state-buffer replacement or shadow cache.
    {
        let (active_key, rollback_key) = (&mut *buffers.active_key, &mut *buffers.rollback_key);
        rollback_key.copy_from_slice(active_key);
        active_key[row.clone()].copy_from_slice(oracle.k_projection_row);
    }
    {
        let (active_value, rollback_value) =
            (&mut *buffers.active_value, &mut *buffers.rollback_value);
        rollback_value.copy_from_slice(active_value);
        active_value[row.clone()].copy_from_slice(oracle.v_projection_row);
    }

    // Causal read is deliberately limited to the current row.  Future
    // positions are represented by -infinity and are not read.
    let current_mask = 0.0f32;
    let future_mask = f32::NEG_INFINITY;
    if current_mask != 0.0 || !future_mask.is_infinite() || future_mask.is_sign_positive() {
        return Err("component child causal mask invariant drifted".into());
    }
    let key_readback = readback_vector(
        "active_key_slot_row_after_append",
        oracle.k_projection_row,
        &buffers.active_key[row.clone()],
        "caller_owned_cache_cpu_readback",
    )?;
    let value_readback = readback_vector(
        "active_value_slot_row_after_append",
        oracle.v_projection_row,
        &buffers.active_value[row.clone()],
        "caller_owned_cache_cpu_readback",
    )?;
    let q_readback = readback_vector(
        "q_projection_rows_cpu_oracle_binding",
        oracle.q_projection_rows,
        oracle.q_projection_rows,
        "caller_cpu_oracle_binding_not_projection_execution",
    )?;
    let o_readback = readback_vector(
        "o_projection_output_cpu_oracle_binding",
        oracle.o_projection_output,
        oracle.o_projection_output,
        "caller_cpu_oracle_binding_not_projection_execution",
    )?;

    {
        let (active_key, rollback_key) = (&mut *buffers.active_key, &mut *buffers.rollback_key);
        active_key.copy_from_slice(rollback_key);
    }
    {
        let (active_value, rollback_value) =
            (&mut *buffers.active_value, &mut *buffers.rollback_value);
        active_value.copy_from_slice(rollback_value);
    }
    let rollback_restored_active_key = f32le_sha256(buffers.active_key)? == active_key_before;
    let rollback_restored_active_value = f32le_sha256(buffers.active_value)? == active_value_before;
    if !rollback_restored_active_key || !rollback_restored_active_value {
        return Err("caller-owned K/V rollback did not exactly restore active state".into());
    }
    if buffers.active_key.len() != expected_elements
        || buffers.active_value.len() != expected_elements
    {
        return Err("caller-owned K/V geometry changed during component transition".into());
    }

    let component_command_order = exact_component_command_order();
    tcb.encode_all(&component_command_order);
    Ok(ComponentChildReadbackParityReceipt {
        schema: COMPONENT_CHILD_RECEIPT_SCHEMA,
        status: COMPONENT_CHILD_STATUS,
        component_only: true,
        selected_layer: SELECTED_GQA_LAYER,
        selected_slot: SELECTED_GQA_SLOT,
        session_id: buffers.session_id.to_owned(),
        token_position: position,
        source_hidden_f32le_sha256: f32le_sha256(hidden.values)?,
        source_hidden_authority: hidden.authority.clone(),
        lease: lease.clone(),
        caller_owned_tcb_id: tcb.tcb_id.clone(),
        caller_owned_tcb_owner_id: tcb.owner_id.clone(),
        single_caller_owned_tcb: true,
        tcb_command_order: component_command_order,
        direct_packed_projection_abi: direct_packed_projection_abi.to_vec(),
        caller_owned_buffer_handles: vec![
            buffers.active_key_handle.to_owned(),
            buffers.active_value_handle.to_owned(),
            buffers.rollback_key_handle.to_owned(),
            buffers.rollback_value_handle.to_owned(),
        ],
        cache_readback_parity: vec![key_readback, value_readback, q_readback, o_readback],
        rollback_restored_active_key,
        rollback_restored_active_value,
        artifact_payload_opened: false,
        metal_context_or_dispatch: false,
        projection_kernel_executed: false,
        causal_attention_executed: false,
        decoder_or_token_execution: false,
    })
}

// Keep the reusable component child typechecked in the standalone preflight
// binary without instantiating any source hidden state, cache storage, lease,
// TCB, or device object. The CLI remains fail-closed for actual child work.
fn component_child_api_compile_anchor() {
    let _ = encode_component_child;
    let _ = CallerOwnedTcb::cpu_contract;
}

fn static_shader_contract() -> Result<(), String> {
    for required in [
        "qwen80_gqa_kv_cache_append_preflight",
        "qwen80_gqa_causal_mask_preflight",
        "qwen80_gqa_kv_cache_rollback_preflight",
        "qwen80_gqa_component_readback_preflight",
        "Qwen80GqaCompactDirectPackedPayloadAbi",
        "Qwen80GqaComponentChildTcbAbi",
        "direct_packed_group_size",
        "active_key_cache",
        "rollback_value_cache",
        "position <= params.current_position",
        "key_cache[cache_index] = key_row[row_index]",
        "value_cache[cache_index] = value_row[row_index]",
    ] {
        if !SHADER_SOURCE.contains(required) {
            return Err(format!(
                "unregistered K/V preflight shader lacks {required}"
            ));
        }
    }
    if SHADER_SOURCE.contains("MetalContext") || SHADER_SOURCE.contains("new_buffer") {
        return Err("unregistered K/V preflight shader must not name a host runtime".into());
    }
    Ok(())
}

fn cpu_preflight() -> Result<
    (
        Vec<GqaSlotBinding>,
        CpuPreflightWitness,
        CpuOracleReadbackLedger,
    ),
    String,
> {
    static_shader_contract()?;
    let bindings = gqa_bindings(PREFLIGHT_CONTEXT)?;
    let selected = bindings
        .get(SELECTED_GQA_SLOT)
        .ok_or_else(|| "selected GQA slot does not exist".to_owned())?;
    if selected.layer != SELECTED_GQA_LAYER {
        return Err("selected GQA layer/slot no longer matches the 12-slot contract".into());
    }
    let mut slot = OneSlotKvPreflight::new(PREFLIGHT_CONTEXT, 0)?;
    let initial_key = slot.active_key.clone();
    let initial_value = slot.active_value.clone();
    let key = synthetic_source_shaped_row(0x4b);
    let value = synthetic_source_shaped_row(0x56);
    slot.append_current_before_causal_read(&key, &value)?;
    let mask = slot.causal_mask_after_append()?;
    let current_mask = mask[slot.position];
    let future_mask = mask[slot.position + 1];
    if current_mask != 0.0 || !future_mask.is_infinite() || future_mask.is_sign_positive() {
        return Err("causal mask does not include current/exclude future position".into());
    }
    let row = slot.row_range();
    let synthetic_active_key_checksum = slot.active_key[row.clone()]
        .iter()
        .map(|value| *value as f64)
        .sum::<f64>();
    let synthetic_active_value_checksum = slot.active_value[row]
        .iter()
        .map(|value| *value as f64)
        .sum::<f64>();
    slot.rollback_after_mask()?;
    let witness = CpuPreflightWitness {
        selected_layer: selected.layer,
        selected_slot: selected.slot,
        position: 0,
        max_seq_len: PREFLIGHT_CONTEXT,
        key_row_elements: KV_ROW_ELEMENTS,
        value_row_elements: KV_ROW_ELEMENTS,
        append_before_causal_read_enforced: true,
        causal_mask_current_position: current_mask,
        causal_mask_future_position: future_mask,
        rollback_restored_key: slot.active_key == initial_key,
        rollback_restored_value: slot.active_value == initial_value,
        phase: slot.phase,
    };
    if !witness.rollback_restored_key || !witness.rollback_restored_value {
        return Err("one-slot K/V rollback failed to restore the active cache".into());
    }
    let session_geometry = per_session_kv_arena_geometry(PREFLIGHT_CONTEXT)?;
    let ledger = CpuOracleReadbackLedger {
        schema: CPU_ORACLE_READBACK_SCHEMA,
        status: "PREPARED_INCOMPLETE_SYNTHETIC_CPU_ORACLE_READBACK_LEDGER",
        component_only: true,
        selected_layer: selected.layer,
        selected_slot: selected.slot,
        session_geometry,
        direct_packed_projection_abi: compact_direct_packed_projection_abi(),
        operation_order: vec![
            "snapshot_active_key_value_slot",
            "append_direct_packed_k_projection_row_at_current_position",
            "append_direct_packed_v_projection_row_at_current_position",
            "causal_read_including_current_position_only",
            "copy_component_readback_ledger",
            "rollback_selected_key_value_slot_on_abort",
        ],
        expected_readback_buffers: expected_readback_buffers(),
        synthetic_active_key_checksum,
        synthetic_active_value_checksum,
        causal_mask_current_position: current_mask,
        causal_mask_future_position: future_mask,
        rollback_restored_key: witness.rollback_restored_key,
        rollback_restored_value: witness.rollback_restored_value,
        artifact_payload_opened: false,
        device_or_metal_invoked: false,
        direct_packed_projection_executed: false,
    };
    Ok((bindings, witness, ledger))
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|entry| !entry.is_empty())
        .ok_or_else(|| format!("lease field {key:?} must be a non-empty string"))
}

fn required_object<'a>(
    value: &'a Value,
    key: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("lease field {key:?} must be an object"))
}

fn required_bool(object: &serde_json::Map<String, Value>, key: &str) -> Result<bool, String> {
    object
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("lease field {key:?} must be a boolean"))
}

fn required_usize(object: &serde_json::Map<String, Value>, key: &str) -> Result<usize, String> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| format!("lease field {key:?} must be an unsigned integer"))
}

fn sealed_document_sha256(value: &Value) -> Result<String, String> {
    let mut unsigned = value.clone();
    let object = unsigned
        .as_object_mut()
        .ok_or_else(|| "lease receipt must be a top-level object".to_owned())?;
    object
        .remove("seal_sha256")
        .ok_or_else(|| "lease receipt is missing seal_sha256".to_owned())?;
    let serialized = serde_json::to_vec(&unsigned)
        .map_err(|error| format!("lease canonical serialization failed: {error}"))?;
    Ok(format!("{:x}", Sha256::digest(serialized)))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ValidatedFutureMetalLease {
    lease_id: String,
    receipt_path: String,
    receipt_seal_sha256: String,
    component: &'static str,
    selected_layer: usize,
    selected_slot: usize,
    future_metal_invoked: bool,
}

fn validate_future_metal_lease(path: &Path) -> Result<ValidatedFutureMetalLease, String> {
    if !path.is_absolute() {
        return Err("--lease-receipt must be an absolute path".into());
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat --lease-receipt {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "--lease-receipt must be a regular non-symlink file: {}",
            path.display()
        ));
    }
    let raw = fs::read(path)
        .map_err(|error| format!("cannot read --lease-receipt {}: {error}", path.display()))?;
    if raw.len() as u64 != metadata.len() {
        return Err("--lease-receipt changed while being read".into());
    }
    let value: Value = serde_json::from_slice(&raw)
        .map_err(|error| format!("--lease-receipt is not valid JSON: {error}"))?;
    if required_string(&value, "schema")? != FUTURE_METAL_LEASE_SCHEMA {
        return Err("--lease-receipt schema is not the exact Qwen80 GQA cache lease schema".into());
    }
    if required_string(&value, "status")? != FUTURE_METAL_LEASE_STATUS {
        return Err("--lease-receipt status is not an exact component-only non-timed grant".into());
    }
    let supplied_seal = required_string(&value, "seal_sha256")?;
    let recomputed_seal = sealed_document_sha256(&value)?;
    if supplied_seal != recomputed_seal {
        return Err("--lease-receipt seal_sha256 does not match its canonical document".into());
    }
    let binding = required_object(&value, "binding")?;
    if binding.get("model_id").and_then(Value::as_str) != Some(MODEL_ID)
        || binding.get("source_repository").and_then(Value::as_str) != Some(SOURCE_REPOSITORY)
        || binding.get("source_revision").and_then(Value::as_str) != Some(SOURCE_REVISION)
        || required_usize(binding, "selected_layer")? != SELECTED_GQA_LAYER
        || required_usize(binding, "selected_slot")? != SELECTED_GQA_SLOT
        || required_usize(binding, "direct_packed_group_size")? != DIRECT_PACKED_GROUP_SIZE
    {
        return Err("--lease-receipt source/layer/slot/direct-packed binding drifted".into());
    }
    let expected_tensor_names = compact_direct_packed_projection_abi()
        .iter()
        .map(|entry| Value::String(entry.tensor_name.to_owned()))
        .collect::<Vec<_>>();
    if binding
        .get("projection_tensor_names")
        .and_then(Value::as_array)
        != Some(&expected_tensor_names)
    {
        return Err(
            "--lease-receipt does not bind the exact layer-3 Q/K/V/O projection names".into(),
        );
    }
    let policy = required_object(&value, "execution_policy")?;
    if policy.get("component").and_then(Value::as_str)
        != Some("qwen80_gqa_kv_cache_append_causal_read_rollback")
        || !required_bool(policy, "quiet_qwen_family_gpu_lease")?
        || !required_bool(policy, "component_only")?
        || required_bool(policy, "timing_or_benchmarking_allowed")?
        || required_bool(policy, "hcli_or_server_allowed")?
        || required_bool(policy, "decoder_or_token_claim_allowed")?
    {
        return Err(
            "--lease-receipt execution policy is not the strict non-timed component-only policy"
                .into(),
        );
    }
    let lifecycle = required_object(&value, "one_shot_lifecycle")?;
    if !required_bool(lifecycle, "fresh_for_this_exact_launch")?
        || required_bool(lifecycle, "automatic_retry_allowed")?
        || lifecycle.get("prior_terminal_receipt") != Some(&Value::Null)
    {
        return Err("--lease-receipt lifecycle is not a fresh non-retrying one-shot".into());
    }
    let lease_id = required_string(&value, "lease_id")?.to_owned();
    Ok(ValidatedFutureMetalLease {
        lease_id,
        receipt_path: path.display().to_string(),
        receipt_seal_sha256: supplied_seal.to_owned(),
        component: "qwen80_gqa_kv_cache_append_causal_read_rollback",
        selected_layer: SELECTED_GQA_LAYER,
        selected_slot: SELECTED_GQA_SLOT,
        future_metal_invoked: false,
    })
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::CpuPreflight => "cpu-preflight",
        Mode::FutureMetalComponent => "future-metal-component",
        Mode::ComponentChild => "component-child",
    }
}

fn result(arguments: &Arguments) -> Result<Value, String> {
    component_child_api_compile_anchor();
    if arguments.mode == Mode::ComponentChild {
        return Err(
            "standalone component-child CLI is fail-closed: it requires a future sealed source-bound [2048] hidden input, a validated quiet lease, one caller-owned TCB, and caller-owned active/rollback K/V buffers; this target will not fabricate them"
                .into(),
        );
    }
    let (bindings, witness, cpu_oracle_readback_ledger) = cpu_preflight()?;
    let validated_future_metal_lease = match arguments.mode {
        Mode::CpuPreflight => None,
        Mode::FutureMetalComponent => Some(validate_future_metal_lease(
            arguments
                .lease_receipt
                .as_deref()
                .ok_or_else(|| "future Metal mode requires --lease-receipt".to_owned())?,
        )?),
        Mode::ComponentChild => unreachable!("component-child rejects before CPU preflight"),
    };
    let status = if validated_future_metal_lease.is_some() {
        LEASE_VALIDATED_STATUS
    } else {
        RESULT_STATUS
    };
    Ok(json!({
        "schema": RESULT_SCHEMA,
        "status": status,
        "mode": mode_name(arguments.mode),
        "model": {
            "model_id": MODEL_ID,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
        },
        "existing_contract_bindings": {
            "decode_state_contract_schema": DECODE_STATE_CONTRACT_SCHEMA,
            "device_state_layout_contract_schema": STATE_LAYOUT_CONTRACT_SCHEMA,
            "gqa_per_layer_readiness_contract_schema": GQA_PER_LAYER_CONTRACT_SCHEMA,
            "source_gqa_layers": bindings.iter().map(|binding| binding.layer).collect::<Vec<_>>(),
            "source_gqa_slots": bindings.iter().map(|binding| binding.slot).collect::<Vec<_>>(),
            "slot_count": GQA_LAYER_COUNT,
            "slot_shape": [PREFLIGHT_CONTEXT, GQA_KV_HEADS, GQA_HEAD_DIM],
            "maximum_native_context": MAX_NATIVE_CONTEXT,
            "exact_selected_layer": SELECTED_GQA_LAYER,
            "exact_selected_slot": SELECTED_GQA_SLOT,
        },
        "one_slot_cpu_preflight": witness,
        "cpu_oracle_readback_ledger": cpu_oracle_readback_ledger,
        "validated_future_metal_lease": validated_future_metal_lease,
        "future_device_stage_order": [
            "snapshot_active_slot_to_rollback",
            "append_direct_packed_key_at_current_position",
            "append_direct_packed_value_at_current_position",
            "causal_mask_positions_zero_through_current",
            "causal_read_positions_zero_through_current",
            "copy_component_readback_ledger",
            "rollback_selected_slot_on_abort",
        ],
        "shader": {
            "path": "crates/hawking-core/shaders/qwen80_gqa_kv_cache_preflight.metal",
            "registered_in_metal_registry": false,
            "static_abi_checked_only": true,
        },
        "execution_boundary": {
            "artifact_scan_or_load": false,
            "metal_context_or_dispatch": false,
            "device_buffer_allocation": false,
            "runtime_watcher_server_or_registry_modified": false,
            "decoder_or_token_execution": false,
            "hcli_or_tps_measurement": false,
            "future_lease_validated_but_metal_not_invoked": validated_future_metal_lease.is_some(),
        },
        "claim_boundary": {
            "prepared_incomplete_only": true,
            "component_only_not_a_gqa_layer_result": true,
            "not_a_decoder_or_token_result": true,
            "not_a_context_or_kv_capability_result": true,
            "not_a_hcli_tps_tg_or_tournament_result": true,
        },
    }))
}

fn usage() -> &'static str {
    concat!(
        "usage: ascension_qwen80_gqa_kv_cache_device_preflight [--mode cpu-preflight]\\n",
        "   or: ascension_qwen80_gqa_kv_cache_device_preflight --mode future-metal-component --lease-receipt ABSOLUTE_SEALED_LEASE_JSON\\n",
        "   or: ascension_qwen80_gqa_kv_cache_device_preflight --mode component-child (always refuses without future caller authority)\\n",
        "Both modes are static/CPU-only. Future-metal mode validates a fresh quiet lease but never creates a Metal context or dispatch."
    )
}

fn parse_args_from<I>(arguments: I) -> Result<Arguments, String>
where
    I: IntoIterator<Item = String>,
{
    let mut mode = Mode::CpuPreflight;
    let mut lease_receipt = None;
    let mut args = arguments.into_iter();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--mode" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --mode; {}", usage()))?;
                mode = match value.as_str() {
                    "cpu-preflight" => Mode::CpuPreflight,
                    "future-metal-component" => Mode::FutureMetalComponent,
                    "component-child" => Mode::ComponentChild,
                    _ => return Err(format!("unsupported --mode {value:?}; {}", usage())),
                };
            }
            "--lease-receipt" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --lease-receipt; {}", usage()))?;
                if lease_receipt.replace(PathBuf::from(value)).is_some() {
                    return Err(format!(
                        "--lease-receipt was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        }
    }
    match (mode, lease_receipt) {
        (Mode::CpuPreflight, None) => Ok(Arguments {
            mode,
            lease_receipt: None,
        }),
        (Mode::CpuPreflight, Some(_)) => Err(format!(
            "--lease-receipt is only valid with --mode future-metal-component; {}",
            usage()
        )),
        (Mode::FutureMetalComponent, Some(lease_receipt)) => Ok(Arguments {
            mode,
            lease_receipt: Some(lease_receipt),
        }),
        (Mode::FutureMetalComponent, None) => Err(format!(
            "--mode future-metal-component requires --lease-receipt; {}",
            usage()
        )),
        (Mode::ComponentChild, None) => Ok(Arguments {
            mode,
            lease_receipt: None,
        }),
        (Mode::ComponentChild, Some(_)) => Err(format!(
            "--lease-receipt cannot substitute caller-owned component-child authority; {}",
            usage()
        )),
    }
}

fn parse_args() -> Result<Arguments, String> {
    parse_args_from(env::args().skip(1))
}

fn main() {
    match parse_args().and_then(|arguments| result(&arguments)) {
        Ok(value) => println!(
            "{}",
            serde_json::to_string_pretty(&value).expect("preflight JSON serializes")
        ),
        Err(error) => {
            eprintln!("Qwen80 GQA K/V cache device preflight refused: {error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_schedule_has_twelve_unique_gqa_slots() {
        let bindings = gqa_bindings(PREFLIGHT_CONTEXT).expect("source-shaped GQA bindings");
        assert_eq!(bindings.len(), GQA_LAYER_COUNT);
        assert_eq!(
            bindings
                .iter()
                .map(|binding| binding.layer)
                .collect::<Vec<_>>(),
            vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
        );
        assert_eq!(
            bindings
                .iter()
                .map(|binding| binding.slot)
                .collect::<Vec<_>>(),
            (0..GQA_LAYER_COUNT).collect::<Vec<_>>()
        );
    }

    #[test]
    fn append_must_happen_before_causal_mask_and_current_is_visible() {
        let mut slot = OneSlotKvPreflight::new(2, 0).expect("two-position cache");
        assert!(slot
            .causal_mask_after_append()
            .expect_err("mask before append must refuse")
            .contains("forbidden"));
        let key = synthetic_source_shaped_row(1);
        let value = synthetic_source_shaped_row(2);
        slot.append_current_before_causal_read(&key, &value)
            .expect("append one source-shaped K/V row");
        let mask = slot.causal_mask_after_append().expect("mask after append");
        assert_eq!(mask[0], 0.0);
        assert!(mask[1].is_infinite() && mask[1].is_sign_negative());
    }

    #[test]
    fn rollback_restores_active_key_and_value_exactly() {
        let mut slot = OneSlotKvPreflight::new(2, 0).expect("two-position cache");
        let key_before = slot.active_key.clone();
        let value_before = slot.active_value.clone();
        slot.append_current_before_causal_read(
            &synthetic_source_shaped_row(3),
            &synthetic_source_shaped_row(4),
        )
        .expect("append");
        slot.causal_mask_after_append().expect("mask");
        slot.rollback_after_mask().expect("rollback");
        assert_eq!(slot.active_key, key_before);
        assert_eq!(slot.active_value, value_before);
        assert_eq!(slot.phase, Phase::RolledBack);
    }

    #[test]
    fn unregistered_shader_declares_append_mask_and_rollback_without_host_runtime() {
        static_shader_contract().expect("static shader ABI must be complete and unregistered");
        assert!(!SHADER_SOURCE.contains("MetalContext"));
        assert!(SHADER_SOURCE.contains("Qwen80GqaCompactDirectPackedPayloadAbi"));
        assert!(SHADER_SOURCE.contains("qwen80_gqa_component_readback_preflight"));
    }

    #[test]
    fn direct_packed_qkvo_abi_is_exact_layer3_and_refuses_bf16_shadow() {
        let abi = compact_direct_packed_projection_abi();
        assert_eq!(abi.len(), 4);
        assert_eq!(
            abi.iter()
                .map(|entry| entry.tensor_name)
                .collect::<Vec<_>>(),
            vec![
                "model.layers.3.self_attn.q_proj.weight",
                "model.layers.3.self_attn.k_proj.weight",
                "model.layers.3.self_attn.v_proj.weight",
                "model.layers.3.self_attn.o_proj.weight",
            ]
        );
        assert_eq!(abi[0].shape, [Q_PROJ_ROWS, HIDDEN_SIZE]);
        assert_eq!(abi[1].shape, [K_PROJ_ROWS, HIDDEN_SIZE]);
        assert_eq!(abi[2].shape, [V_PROJ_ROWS, HIDDEN_SIZE]);
        assert_eq!(abi[3].shape, [O_PROJ_ROWS, O_PROJ_COLS]);
        assert!(abi.iter().all(|entry| {
            entry.group_size == DIRECT_PACKED_GROUP_SIZE
                && entry.direct_packed_only
                && !entry.bf16_shadow_allowed
        }));
    }

    #[test]
    fn per_session_active_and_rollback_geometry_are_disjoint_and_source_shaped() {
        let geometry = per_session_kv_arena_geometry(PREFLIGHT_CONTEXT)
            .expect("synthetic per-session K/V geometry");
        assert!(geometry.active_and_rollback_disjoint);
        assert_ne!(
            geometry.active.key_buffer_role,
            geometry.rollback.key_buffer_role
        );
        assert_ne!(
            geometry.active.value_buffer_role,
            geometry.rollback.value_buffer_role
        );
        assert_eq!(geometry.active.shape, [2, GQA_KV_HEADS, GQA_HEAD_DIM]);
        assert_eq!(geometry.active.bytes_per_buffer, 4_096);
        assert_eq!(geometry.total_session_bytes, 16_384);
        assert_eq!(
            geometry.maximum_native_context_total_session_bytes,
            33_554_432
        );
    }

    #[test]
    fn cpu_oracle_ledger_is_synthetic_component_only_and_records_readback_contract() {
        let (_, witness, ledger) = cpu_preflight().expect("CPU K/V preflight");
        assert_eq!(ledger.schema, CPU_ORACLE_READBACK_SCHEMA);
        assert!(ledger.component_only);
        assert!(!ledger.artifact_payload_opened);
        assert!(!ledger.device_or_metal_invoked);
        assert!(!ledger.direct_packed_projection_executed);
        assert!(ledger.rollback_restored_key && ledger.rollback_restored_value);
        assert_eq!(
            ledger.causal_mask_current_position,
            witness.causal_mask_current_position
        );
        assert!(ledger
            .expected_readback_buffers
            .iter()
            .any(|buffer| buffer.name == "o_projection_output" && buffer.elements == O_PROJ_ROWS));
    }

    fn sealed_test_future_lease() -> Value {
        let abi = compact_direct_packed_projection_abi();
        let mut lease = json!({
            "schema": FUTURE_METAL_LEASE_SCHEMA,
            "status": FUTURE_METAL_LEASE_STATUS,
            "lease_id": "test-qwen80-gqa-kv-component-lease",
            "binding": {
                "model_id": MODEL_ID,
                "source_repository": SOURCE_REPOSITORY,
                "source_revision": SOURCE_REVISION,
                "selected_layer": SELECTED_GQA_LAYER,
                "selected_slot": SELECTED_GQA_SLOT,
                "direct_packed_group_size": DIRECT_PACKED_GROUP_SIZE,
                "projection_tensor_names": abi.iter().map(|entry| entry.tensor_name).collect::<Vec<_>>(),
            },
            "execution_policy": {
                "component": "qwen80_gqa_kv_cache_append_causal_read_rollback",
                "quiet_qwen_family_gpu_lease": true,
                "component_only": true,
                "timing_or_benchmarking_allowed": false,
                "hcli_or_server_allowed": false,
                "decoder_or_token_claim_allowed": false,
            },
            "one_shot_lifecycle": {
                "fresh_for_this_exact_launch": true,
                "prior_terminal_receipt": null,
                "automatic_retry_allowed": false,
            },
            "seal_sha256": "",
        });
        let seal = sealed_document_sha256(&lease).expect("test lease canonical seal");
        lease["seal_sha256"] = Value::String(seal);
        lease
    }

    fn write_test_lease(value: &Value) -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "hawking-qwen80-gqa-kv-lease-{}-{nonce}.json",
            std::process::id()
        ));
        fs::write(
            &path,
            serde_json::to_vec(value).expect("test lease serializes"),
        )
        .expect("write test lease");
        path
    }

    #[test]
    fn future_metal_mode_requires_a_fresh_sealed_component_only_lease_but_never_dispatches() {
        let lease = sealed_test_future_lease();
        let path = write_test_lease(&lease);
        let args = Arguments {
            mode: Mode::FutureMetalComponent,
            lease_receipt: Some(path.clone()),
        };
        let value = result(&args).expect("valid future-metal lease remains static only");
        assert_eq!(value["status"], LEASE_VALIDATED_STATUS);
        assert_eq!(value["mode"], "future-metal-component");
        assert_eq!(
            value["execution_boundary"]["future_lease_validated_but_metal_not_invoked"],
            true
        );
        assert_eq!(
            value["execution_boundary"]["metal_context_or_dispatch"],
            false
        );
        fs::remove_file(path).expect("remove test lease");
    }

    #[test]
    fn future_metal_lease_rejects_bad_seal_and_cpu_mode_rejects_lease_argument() {
        let mut lease = sealed_test_future_lease();
        lease["binding"]["selected_slot"] = json!(1);
        let path = write_test_lease(&lease);
        assert!(validate_future_metal_lease(&path)
            .expect_err("mutated lease must fail canonical seal validation")
            .contains("seal_sha256"));
        fs::remove_file(path).expect("remove test lease");
        assert!(parse_args_from(vec![
            "--mode".to_owned(),
            "cpu-preflight".to_owned(),
            "--lease-receipt".to_owned(),
            "/tmp/nope.json".to_owned(),
        ])
        .expect_err("CPU mode must refuse a future lease argument")
        .contains("only valid"));
    }

    fn test_validated_future_lease() -> ValidatedFutureMetalLease {
        let lease = sealed_test_future_lease();
        let path = write_test_lease(&lease);
        let validated = validate_future_metal_lease(&path).expect("sealed future lease validates");
        fs::remove_file(path).expect("remove test lease");
        validated
    }

    fn test_source_hidden_authority(
        session_id: &str,
        token_position: usize,
    ) -> SourceBoundHiddenAuthority {
        SourceBoundHiddenAuthority {
            schema: SOURCE_HIDDEN_AUTHORITY_SCHEMA.to_owned(),
            status: SOURCE_HIDDEN_AUTHORITY_STATUS.to_owned(),
            authority_seal_sha256: "a".repeat(64),
            model_id: MODEL_ID.to_owned(),
            source_repository: SOURCE_REPOSITORY.to_owned(),
            source_revision: SOURCE_REVISION.to_owned(),
            source_layer: SELECTED_GQA_LAYER,
            source_gqa_slot: SELECTED_GQA_SLOT,
            hidden_width: HIDDEN_SIZE,
            session_id: session_id.to_owned(),
            token_position,
        }
    }

    fn test_hidden_values() -> [f32; HIDDEN_SIZE] {
        std::array::from_fn(|index| ((index % 31) as f32 - 15.0) / 31.0)
    }

    fn test_oracle_vectors() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let values = |length: usize, salt: usize| {
            (0..length)
                .map(|index| ((index ^ salt) % 127) as f32 / 127.0)
                .collect::<Vec<_>>()
        };
        (
            values(Q_PROJ_ROWS, 3),
            values(K_PROJ_ROWS, 5),
            values(V_PROJ_ROWS, 7),
            values(O_PROJ_ROWS, 11),
        )
    }

    #[test]
    fn component_child_encodes_exact_qkvo_append_read_rollback_into_one_caller_owned_tcb() {
        let session_id = "test-qwen80-gqa-session";
        let hidden_values = test_hidden_values();
        let hidden = SourceBoundHiddenInput {
            values: &hidden_values,
            authority: test_source_hidden_authority(session_id, 0),
        };
        let (q_projection_rows, k_projection_row, v_projection_row, o_projection_output) =
            test_oracle_vectors();
        let oracle = CpuOracleProjectionReadback {
            q_projection_rows: &q_projection_rows,
            k_projection_row: &k_projection_row,
            v_projection_row: &v_projection_row,
            o_projection_output: &o_projection_output,
            provenance: "caller_cpu_oracle_direct_packed_component_readback",
        };
        let elements = PREFLIGHT_CONTEXT * KV_ROW_ELEMENTS;
        let mut active_key = vec![0.125; elements];
        let mut active_value = vec![0.25; elements];
        let mut rollback_key = vec![0.5; elements];
        let mut rollback_value = vec![0.75; elements];
        let active_key_before = active_key.clone();
        let active_value_before = active_value.clone();
        let direct_packed_projection_abi = compact_direct_packed_projection_abi();
        let mut tcb = CallerOwnedTcb::cpu_contract(
            "tcb-1".to_owned(),
            "future-host".to_owned(),
            session_id.to_owned(),
        );
        let receipt = {
            let mut buffers = CallerOwnedSessionKvBuffers {
                session_id,
                selected_layer: SELECTED_GQA_LAYER,
                selected_slot: SELECTED_GQA_SLOT,
                max_seq_len: PREFLIGHT_CONTEXT,
                active_key_handle: "host.active.key.layer3.slot0",
                active_value_handle: "host.active.value.layer3.slot0",
                rollback_key_handle: "host.rollback.key.layer3.slot0",
                rollback_value_handle: "host.rollback.value.layer3.slot0",
                active_key: &mut active_key,
                active_value: &mut active_value,
                rollback_key: &mut rollback_key,
                rollback_value: &mut rollback_value,
            };
            encode_component_child(
                &mut tcb,
                &hidden,
                &mut buffers,
                &direct_packed_projection_abi,
                &oracle,
                &test_validated_future_lease(),
            )
            .expect("caller-owned CPU component contract")
        };
        assert_eq!(tcb.commands, exact_component_command_order());
        assert_eq!(receipt.schema, COMPONENT_CHILD_RECEIPT_SCHEMA);
        assert_eq!(receipt.status, COMPONENT_CHILD_STATUS);
        assert!(receipt.single_caller_owned_tcb);
        assert!(receipt.rollback_restored_active_key);
        assert!(receipt.rollback_restored_active_value);
        assert!(receipt
            .cache_readback_parity
            .iter()
            .all(|entry| entry.exact));
        assert!(!receipt.artifact_payload_opened);
        assert!(!receipt.metal_context_or_dispatch);
        assert!(!receipt.projection_kernel_executed);
        assert_eq!(active_key, active_key_before);
        assert_eq!(active_value, active_value_before);
    }

    #[test]
    fn component_child_refuses_missing_source_authority_before_mutating_or_encoding() {
        let session_id = "test-qwen80-gqa-session";
        let hidden_values = test_hidden_values();
        let mut authority = test_source_hidden_authority(session_id, 0);
        authority.authority_seal_sha256.clear();
        let hidden = SourceBoundHiddenInput {
            values: &hidden_values,
            authority,
        };
        let (q_projection_rows, k_projection_row, v_projection_row, o_projection_output) =
            test_oracle_vectors();
        let oracle = CpuOracleProjectionReadback {
            q_projection_rows: &q_projection_rows,
            k_projection_row: &k_projection_row,
            v_projection_row: &v_projection_row,
            o_projection_output: &o_projection_output,
            provenance: "caller_cpu_oracle_direct_packed_component_readback",
        };
        let elements = PREFLIGHT_CONTEXT * KV_ROW_ELEMENTS;
        let mut active_key = vec![0.125; elements];
        let mut active_value = vec![0.25; elements];
        let mut rollback_key = vec![0.5; elements];
        let mut rollback_value = vec![0.75; elements];
        let active_key_before = active_key.clone();
        let direct_packed_projection_abi = compact_direct_packed_projection_abi();
        let mut tcb = CallerOwnedTcb::cpu_contract(
            "tcb-2".to_owned(),
            "future-host".to_owned(),
            session_id.to_owned(),
        );
        let mut buffers = CallerOwnedSessionKvBuffers {
            session_id,
            selected_layer: SELECTED_GQA_LAYER,
            selected_slot: SELECTED_GQA_SLOT,
            max_seq_len: PREFLIGHT_CONTEXT,
            active_key_handle: "host.active.key.layer3.slot0",
            active_value_handle: "host.active.value.layer3.slot0",
            rollback_key_handle: "host.rollback.key.layer3.slot0",
            rollback_value_handle: "host.rollback.value.layer3.slot0",
            active_key: &mut active_key,
            active_value: &mut active_value,
            rollback_key: &mut rollback_key,
            rollback_value: &mut rollback_value,
        };
        assert!(encode_component_child(
            &mut tcb,
            &hidden,
            &mut buffers,
            &direct_packed_projection_abi,
            &oracle,
            &test_validated_future_lease(),
        )
        .expect_err("missing source authority must fail closed")
        .contains("source-bound hidden input authority"));
        assert!(tcb.commands.is_empty());
        drop(buffers);
        assert_eq!(active_key, active_key_before);
    }

    #[test]
    fn standalone_component_child_cli_is_fail_closed_without_upstream_authority() {
        let arguments = parse_args_from(vec!["--mode".to_owned(), "component-child".to_owned()])
            .expect("component-child mode syntax is recognized so it can explain its refusal");
        assert!(result(&arguments)
            .expect_err("standalone CLI cannot create hidden/cache/lease authority")
            .contains("fail-closed"));
    }

    #[test]
    fn result_is_explicitly_prepared_and_not_a_decoder_claim() {
        let value = result(&Arguments {
            mode: Mode::CpuPreflight,
            lease_receipt: None,
        })
        .expect("CPU preflight result");
        assert_eq!(value["status"], RESULT_STATUS);
        assert_eq!(value["claim_boundary"]["prepared_incomplete_only"], true);
        assert_eq!(
            value["execution_boundary"]["metal_context_or_dispatch"],
            false
        );
        assert_eq!(
            value["execution_boundary"]["decoder_or_token_execution"],
            false
        );
    }
}
