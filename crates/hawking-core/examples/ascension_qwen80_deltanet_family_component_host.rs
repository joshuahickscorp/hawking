//! CPU/build-only Qwen3-Coder-Next 36-slot DeltaNet family component host.
//!
//! This program consumes metadata from the permanent 48-layer payload/schedule
//! authority plus the existing L0 strict-bridge *source file*.  It emits a
//! create-new, static ABI plan for every source-selected DeltaNet layer:
//! `0, 1, 2, 4, ..., 46` / slots `0..35`.
//!
//! It deliberately does not open an artifact payload, allocate a Metal
//! buffer, create a command buffer, dispatch a kernel, take a lease, mutate a
//! registry/runtime/watcher/server, execute a model, or measure HCLI/TPS/TG.
//! The source bridge is evidence of the existing L0 implementation shape only;
//! it is not promoted into an all-layer device result.  The emitted plan is a
//! reusable component-host readiness artifact, not a complete layer, token,
//! decoder, or server claim.
//!
//! ```text
//! cargo run -p hawking-core --example ascension_qwen80_deltanet_family_component_host -- \
//!   --schedule-authority /absolute/path/QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY.json \
//!   --l0-strict-bridge-source /absolute/path/ascension_qwen80_first_residual_bridge_device.rs \
//!   --out /absolute/new/QWEN80_DELTANET_FAMILY_COMPONENT_HOST.json
//! ```

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_deltanet_family_component_host.v1";
const RESULT_STATUS: &str = "PREPARED_QWEN80_36_DELTANET_GENERIC_COMPONENT_HOST_NOT_EXECUTED";
const EXECUTION_STATUS: &str = "PREPARED_NOT_EXECUTED";
const SCHEDULE_SCHEMA: &str = "hawking.ascension.qwen80_48_layer_payload_schedule_authority.v1";
const SCHEDULE_STATUS: &str = "PREPARED_QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY_NOT_EXECUTED";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_SEAL: &str = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const DESCRIPTOR_DOCUMENT_SHA: &str =
    "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10";
const HIDDEN: usize = 2_048;
const GROUP_SIZE: usize = 128;
const RMS_EPSILON: f64 = 1.0e-6;
const LAYER_COUNT: usize = 48;
const DELTANET_LAYERS: usize = 36;
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 32;
const VALUES_PER_KEY_HEAD: usize = VALUE_HEADS / KEY_HEADS;
const KEY_HEAD_DIM: usize = 128;
const VALUE_HEAD_DIM: usize = 128;
const QKVZ_ROWS: usize = 12_288;
const BA_ROWS: usize = 64;
const CONV_CHANNELS: usize = 8_192;
const CONV_KERNEL: usize = 4;
const CONV_HISTORY: usize = CONV_KERNEL - 1;
const CONV_SLOT_ELEMENTS: usize = CONV_CHANNELS * CONV_HISTORY;
const RECURRENT_SLOT_ELEMENTS: usize = VALUE_HEADS * KEY_HEAD_DIM * VALUE_HEAD_DIM;
const VALUE_ELEMENTS: usize = VALUE_HEADS * VALUE_HEAD_DIM;
const F32_BYTES: usize = std::mem::size_of::<f32>();
const L0_REQUIRED_SOURCE_ANCHORS: [&str; 6] = [
    "Qwen80SourceInputFirstResidualEncoder",
    "encode_source_token_first_linear_deltanet_into",
    "TokenCommandBuffer",
    "Qwen80SourceInputL0TrueMoeGraph",
    "PREFIX_DISPATCHES: usize = 9",
    "prefix_dispatches != 9",
];

#[derive(Clone, Debug, Eq, PartialEq)]
struct Args {
    schedule_authority: PathBuf,
    l0_strict_bridge_source: PathBuf,
    out: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
struct ScheduleAuthority {
    schema: String,
    status: String,
    source_authority: ScheduleSourceAuthority,
    geometry: ScheduleGeometry,
    layers: Vec<ScheduleLayer>,
    deltanet_state_slots: Vec<ScheduleStateSlot>,
    all_48_layers_scheduled: bool,
    all_descriptors_source_artifact_bound: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct ScheduleSourceAuthority {
    model_id: String,
    model_key: String,
    source_repository: String,
    source_revision: String,
    descriptor_inventory_document_sha256: String,
    descriptor_inventory_seal_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct ScheduleGeometry {
    layer_count: usize,
    hidden_size: usize,
    linear_key_heads: usize,
    linear_value_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_conv_kernel: usize,
    direct_pack_group_size: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Mixer {
    DeltaNet,
    Gqa,
}

impl Mixer {
    fn expected(layer: usize) -> Self {
        if layer % 4 == 3 {
            Self::Gqa
        } else {
            Self::DeltaNet
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct ScheduleLayer {
    layer: usize,
    mixer: Mixer,
    state_slot: ScheduleStateSlot,
    input_layernorm: ScheduleTensor,
    mixer_tensor_execution_order: Vec<String>,
    mixer_tensors: Vec<ScheduleTensor>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct ScheduleStateSlot {
    layer: usize,
    slot: usize,
    domain: String,
    device_buffers_required_before_execution: Vec<String>,
    rollback_buffers_required_before_execution: Vec<String>,
    state_materialized_by_this_plan: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct ScheduleTensor {
    role: String,
    tensor_name: String,
    shape: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct DeltaNetLayerSlot {
    layer: usize,
    slot: usize,
}

fn expected_deltanet_schedule() -> Vec<DeltaNetLayerSlot> {
    (0..LAYER_COUNT)
        .filter(|layer| Mixer::expected(*layer) == Mixer::DeltaNet)
        .enumerate()
        .map(|(slot, layer)| DeltaNetLayerSlot { layer, slot })
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct DirectPackedTensorAbi {
    role: String,
    tensor_name: String,
    shape: Vec<usize>,
    required_before_encoding: bool,
    payload_opened_or_uploaded_by_this_plan: bool,
}

fn expected_direct_packed_tensors(layer: usize) -> Vec<DirectPackedTensorAbi> {
    let prefix = format!("model.layers.{layer}.linear_attn");
    vec![
        DirectPackedTensorAbi {
            role: "input_layernorm".into(),
            tensor_name: format!("model.layers.{layer}.input_layernorm.weight"),
            shape: vec![HIDDEN],
            required_before_encoding: true,
            payload_opened_or_uploaded_by_this_plan: false,
        },
        DirectPackedTensorAbi {
            role: "deltanet_in_proj_qkvz".into(),
            tensor_name: format!("{prefix}.in_proj_qkvz.weight"),
            shape: vec![QKVZ_ROWS, HIDDEN],
            required_before_encoding: true,
            payload_opened_or_uploaded_by_this_plan: false,
        },
        DirectPackedTensorAbi {
            role: "deltanet_in_proj_ba".into(),
            tensor_name: format!("{prefix}.in_proj_ba.weight"),
            shape: vec![BA_ROWS, HIDDEN],
            required_before_encoding: true,
            payload_opened_or_uploaded_by_this_plan: false,
        },
        DirectPackedTensorAbi {
            role: "deltanet_causal_conv1d".into(),
            tensor_name: format!("{prefix}.conv1d.weight"),
            shape: vec![CONV_CHANNELS, 1, CONV_KERNEL],
            required_before_encoding: true,
            payload_opened_or_uploaded_by_this_plan: false,
        },
        DirectPackedTensorAbi {
            role: "deltanet_a_log".into(),
            tensor_name: format!("{prefix}.A_log"),
            shape: vec![VALUE_HEADS],
            required_before_encoding: true,
            payload_opened_or_uploaded_by_this_plan: false,
        },
        DirectPackedTensorAbi {
            role: "deltanet_dt_bias".into(),
            tensor_name: format!("{prefix}.dt_bias"),
            shape: vec![VALUE_HEADS],
            required_before_encoding: true,
            payload_opened_or_uploaded_by_this_plan: false,
        },
        DirectPackedTensorAbi {
            role: "deltanet_gated_rmsnorm".into(),
            tensor_name: format!("{prefix}.norm.weight"),
            shape: vec![VALUE_HEAD_DIM],
            required_before_encoding: true,
            payload_opened_or_uploaded_by_this_plan: false,
        },
        DirectPackedTensorAbi {
            role: "deltanet_out_proj".into(),
            tensor_name: format!("{prefix}.out_proj.weight"),
            shape: vec![HIDDEN, VALUE_ELEMENTS],
            required_before_encoding: true,
            payload_opened_or_uploaded_by_this_plan: false,
        },
    ]
}

fn expected_mixer_order() -> Vec<String> {
    vec![
        "deltanet_in_proj_qkvz".into(),
        "deltanet_in_proj_ba".into(),
        "deltanet_causal_conv1d".into(),
        "deltanet_a_log".into(),
        "deltanet_dt_bias".into(),
        "deltanet_gated_rmsnorm".into(),
        "deltanet_out_proj".into(),
    ]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum BufferAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct CallerOwnedBufferAbi {
    name: &'static str,
    access: BufferAccess,
    element_type: &'static str,
    family_allocation_shape: Vec<usize>,
    selected_slot_shape: Vec<usize>,
    selected_slot_offset_elements: usize,
    selected_slot_elements: usize,
    selected_slot_bytes: usize,
    retain_through_same_token_command_buffer_fence: bool,
    allocation_or_upload_performed_by_this_plan: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum DeltaNetDispatchOperation {
    InputRmsNorm,
    QkvzSignScaleMatvec,
    BaSignScaleMatvec,
    QkvzRearrangeCausalConvL2AndConvStateCommit,
    BaToDecayBeta,
    GatedDeltaRecurrenceAndRecurrentStateCommit,
    ZGatedRmsNorm,
    OutProjectionSignScaleMatvec,
    AddFirstResidual,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct DispatchAbi {
    ordinal: usize,
    operation: DeltaNetDispatchOperation,
    direct_packed_roles: Vec<&'static str>,
    caller_buffer_names: Vec<&'static str>,
    scalar_contract: Vec<String>,
    commits_conv_state_once: bool,
    commits_recurrent_state_once: bool,
    uses_same_open_token_command_buffer: bool,
    commits_or_fences_by_this_plan: bool,
}

fn dispatch_abi(slot: usize) -> Vec<DispatchAbi> {
    let conv_offset = slot * CONV_SLOT_ELEMENTS;
    let recurrent_offset = slot * RECURRENT_SLOT_ELEMENTS;
    vec![
        DispatchAbi {
            ordinal: 0,
            operation: DeltaNetDispatchOperation::InputRmsNorm,
            direct_packed_roles: vec!["input_layernorm"],
            caller_buffer_names: vec!["hidden_input", "normalized_hidden"],
            scalar_contract: vec![
                format!("hidden_elements={HIDDEN}"),
                format!("group_size={GROUP_SIZE}"),
                format!("rms_epsilon={RMS_EPSILON}"),
            ],
            commits_conv_state_once: false,
            commits_recurrent_state_once: false,
            uses_same_open_token_command_buffer: true,
            commits_or_fences_by_this_plan: false,
        },
        DispatchAbi {
            ordinal: 1,
            operation: DeltaNetDispatchOperation::QkvzSignScaleMatvec,
            direct_packed_roles: vec!["deltanet_in_proj_qkvz"],
            caller_buffer_names: vec!["normalized_hidden", "qkvz_output"],
            scalar_contract: vec![
                format!("rows={QKVZ_ROWS}"),
                format!("columns={HIDDEN}"),
                format!("group_size={GROUP_SIZE}"),
            ],
            commits_conv_state_once: false,
            commits_recurrent_state_once: false,
            uses_same_open_token_command_buffer: true,
            commits_or_fences_by_this_plan: false,
        },
        DispatchAbi {
            ordinal: 2,
            operation: DeltaNetDispatchOperation::BaSignScaleMatvec,
            direct_packed_roles: vec!["deltanet_in_proj_ba"],
            caller_buffer_names: vec!["normalized_hidden", "ba_output"],
            scalar_contract: vec![
                format!("rows={BA_ROWS}"),
                format!("columns={HIDDEN}"),
                format!("group_size={GROUP_SIZE}"),
            ],
            commits_conv_state_once: false,
            commits_recurrent_state_once: false,
            uses_same_open_token_command_buffer: true,
            commits_or_fences_by_this_plan: false,
        },
        DispatchAbi {
            ordinal: 3,
            operation: DeltaNetDispatchOperation::QkvzRearrangeCausalConvL2AndConvStateCommit,
            direct_packed_roles: vec!["deltanet_causal_conv1d"],
            caller_buffer_names: vec![
                "qkvz_output",
                "deltanet_conv_history_active",
                "repeated_query",
                "repeated_key",
                "convolved_value",
                "z_gate",
            ],
            scalar_contract: vec![
                format!("conv_slot={slot}"),
                format!("conv_state_offset_elements={conv_offset}"),
                format!("conv_channels={CONV_CHANNELS}"),
                format!("prior_tokens={CONV_HISTORY}"),
                format!("conv_kernel={CONV_KERNEL}"),
                format!("key_heads={KEY_HEADS}"),
                format!("values_per_key_head={VALUES_PER_KEY_HEAD}"),
                format!("key_head_dim={KEY_HEAD_DIM}"),
                format!("value_head_dim={VALUE_HEAD_DIM}"),
            ],
            commits_conv_state_once: true,
            commits_recurrent_state_once: false,
            uses_same_open_token_command_buffer: true,
            commits_or_fences_by_this_plan: false,
        },
        DispatchAbi {
            ordinal: 4,
            operation: DeltaNetDispatchOperation::BaToDecayBeta,
            direct_packed_roles: vec!["deltanet_a_log", "deltanet_dt_bias"],
            caller_buffer_names: vec!["ba_output", "decay", "beta"],
            scalar_contract: vec![
                format!("key_heads={KEY_HEADS}"),
                format!("values_per_key_head={VALUES_PER_KEY_HEAD}"),
                format!("group_size={GROUP_SIZE}"),
            ],
            commits_conv_state_once: false,
            commits_recurrent_state_once: false,
            uses_same_open_token_command_buffer: true,
            commits_or_fences_by_this_plan: false,
        },
        DispatchAbi {
            ordinal: 5,
            operation: DeltaNetDispatchOperation::GatedDeltaRecurrenceAndRecurrentStateCommit,
            direct_packed_roles: vec![],
            caller_buffer_names: vec![
                "deltanet_recurrent_active",
                "repeated_query",
                "repeated_key",
                "convolved_value",
                "decay",
                "beta",
                "recurrent_output",
            ],
            scalar_contract: vec![
                format!("recurrent_slot={slot}"),
                format!("recurrent_state_offset_elements={recurrent_offset}"),
                format!("value_heads={VALUE_HEADS}"),
                format!("key_head_dim={KEY_HEAD_DIM}"),
                format!("value_head_dim={VALUE_HEAD_DIM}"),
            ],
            commits_conv_state_once: false,
            commits_recurrent_state_once: true,
            uses_same_open_token_command_buffer: true,
            commits_or_fences_by_this_plan: false,
        },
        DispatchAbi {
            ordinal: 6,
            operation: DeltaNetDispatchOperation::ZGatedRmsNorm,
            direct_packed_roles: vec!["deltanet_gated_rmsnorm"],
            caller_buffer_names: vec!["recurrent_output", "z_gate", "gated_output"],
            scalar_contract: vec![
                format!("value_heads={VALUE_HEADS}"),
                format!("value_head_dim={VALUE_HEAD_DIM}"),
                format!("group_size={GROUP_SIZE}"),
                format!("rms_epsilon={RMS_EPSILON}"),
            ],
            commits_conv_state_once: false,
            commits_recurrent_state_once: false,
            uses_same_open_token_command_buffer: true,
            commits_or_fences_by_this_plan: false,
        },
        DispatchAbi {
            ordinal: 7,
            operation: DeltaNetDispatchOperation::OutProjectionSignScaleMatvec,
            direct_packed_roles: vec!["deltanet_out_proj"],
            caller_buffer_names: vec!["gated_output", "mixer_output"],
            scalar_contract: vec![
                format!("rows={HIDDEN}"),
                format!("columns={VALUE_ELEMENTS}"),
                format!("group_size={GROUP_SIZE}"),
            ],
            commits_conv_state_once: false,
            commits_recurrent_state_once: false,
            uses_same_open_token_command_buffer: true,
            commits_or_fences_by_this_plan: false,
        },
        DispatchAbi {
            ordinal: 8,
            operation: DeltaNetDispatchOperation::AddFirstResidual,
            direct_packed_roles: vec![],
            caller_buffer_names: vec!["hidden_input", "mixer_output", "first_residual_output"],
            scalar_contract: vec![format!("hidden_elements={HIDDEN}")],
            commits_conv_state_once: false,
            commits_recurrent_state_once: false,
            uses_same_open_token_command_buffer: true,
            commits_or_fences_by_this_plan: false,
        },
    ]
}

fn caller_owned_buffers(slot: usize) -> Vec<CallerOwnedBufferAbi> {
    let buffer = |name: &'static str,
                  access: BufferAccess,
                  family_allocation_shape: Vec<usize>,
                  selected_slot_shape: Vec<usize>,
                  offset: usize| {
        let elements = selected_slot_shape.iter().product::<usize>();
        CallerOwnedBufferAbi {
            name,
            access,
            element_type: "f32",
            family_allocation_shape,
            selected_slot_shape,
            selected_slot_offset_elements: offset,
            selected_slot_elements: elements,
            selected_slot_bytes: elements * F32_BYTES,
            retain_through_same_token_command_buffer_fence: true,
            allocation_or_upload_performed_by_this_plan: false,
        }
    };
    vec![
        buffer(
            "hidden_input",
            BufferAccess::ReadOnly,
            vec![HIDDEN],
            vec![HIDDEN],
            0,
        ),
        buffer(
            "normalized_hidden",
            BufferAccess::WriteOnly,
            vec![HIDDEN],
            vec![HIDDEN],
            0,
        ),
        buffer(
            "qkvz_output",
            BufferAccess::WriteOnly,
            vec![QKVZ_ROWS],
            vec![QKVZ_ROWS],
            0,
        ),
        buffer(
            "ba_output",
            BufferAccess::WriteOnly,
            vec![BA_ROWS],
            vec![BA_ROWS],
            0,
        ),
        buffer(
            "repeated_query",
            BufferAccess::WriteOnly,
            vec![VALUE_ELEMENTS],
            vec![VALUE_ELEMENTS],
            0,
        ),
        buffer(
            "repeated_key",
            BufferAccess::WriteOnly,
            vec![VALUE_ELEMENTS],
            vec![VALUE_ELEMENTS],
            0,
        ),
        buffer(
            "convolved_value",
            BufferAccess::WriteOnly,
            vec![VALUE_ELEMENTS],
            vec![VALUE_ELEMENTS],
            0,
        ),
        buffer(
            "z_gate",
            BufferAccess::WriteOnly,
            vec![VALUE_ELEMENTS],
            vec![VALUE_ELEMENTS],
            0,
        ),
        buffer(
            "decay",
            BufferAccess::WriteOnly,
            vec![VALUE_HEADS],
            vec![VALUE_HEADS],
            0,
        ),
        buffer(
            "beta",
            BufferAccess::WriteOnly,
            vec![VALUE_HEADS],
            vec![VALUE_HEADS],
            0,
        ),
        buffer(
            "deltanet_conv_history_active",
            BufferAccess::ReadWrite,
            vec![DELTANET_LAYERS, CONV_CHANNELS, CONV_HISTORY],
            vec![CONV_CHANNELS, CONV_HISTORY],
            slot * CONV_SLOT_ELEMENTS,
        ),
        buffer(
            "deltanet_recurrent_active",
            BufferAccess::ReadWrite,
            vec![DELTANET_LAYERS, VALUE_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM],
            vec![VALUE_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM],
            slot * RECURRENT_SLOT_ELEMENTS,
        ),
        buffer(
            "deltanet_conv_history_rollback",
            BufferAccess::ReadWrite,
            vec![DELTANET_LAYERS, CONV_CHANNELS, CONV_HISTORY],
            vec![CONV_CHANNELS, CONV_HISTORY],
            slot * CONV_SLOT_ELEMENTS,
        ),
        buffer(
            "deltanet_recurrent_rollback",
            BufferAccess::ReadWrite,
            vec![DELTANET_LAYERS, VALUE_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM],
            vec![VALUE_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM],
            slot * RECURRENT_SLOT_ELEMENTS,
        ),
        buffer(
            "recurrent_output",
            BufferAccess::WriteOnly,
            vec![VALUE_ELEMENTS],
            vec![VALUE_ELEMENTS],
            0,
        ),
        buffer(
            "gated_output",
            BufferAccess::WriteOnly,
            vec![VALUE_ELEMENTS],
            vec![VALUE_ELEMENTS],
            0,
        ),
        buffer(
            "mixer_output",
            BufferAccess::WriteOnly,
            vec![HIDDEN],
            vec![HIDDEN],
            0,
        ),
        buffer(
            "first_residual_output",
            BufferAccess::WriteOnly,
            vec![HIDDEN],
            vec![HIDDEN],
            0,
        ),
    ]
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct StateJournalContract {
    active_conv_slot_offset_elements: usize,
    active_recurrent_slot_offset_elements: usize,
    rollback_conv_slot_offset_elements: usize,
    rollback_recurrent_slot_offset_elements: usize,
    snapshot_before_dispatch_zero_required: bool,
    conv_state_committed_once_at_dispatch_ordinal: usize,
    recurrent_state_committed_once_at_dispatch_ordinal: usize,
    restore_both_rollback_ranges_on_rejected_token_required: bool,
    state_slices_may_alias_other_deltanet_slots: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct DeltaNetLayerComponentAbi {
    layer: usize,
    slot: usize,
    component_only: bool,
    synthetic_input_accepted: bool,
    direct_packed_tensors: Vec<DirectPackedTensorAbi>,
    caller_owned_buffers: Vec<CallerOwnedBufferAbi>,
    state_journal: StateJournalContract,
    same_token_command_buffer_required: bool,
    total_dispatches: usize,
    ordered_dispatches: Vec<DispatchAbi>,
}

fn build_layer_component(layer: usize, slot: usize) -> Result<DeltaNetLayerComponentAbi, String> {
    let expected = expected_deltanet_schedule()
        .into_iter()
        .find(|entry| entry.layer == layer)
        .ok_or_else(|| format!("layer {layer} is not a source-selected DeltaNet layer"))?;
    if slot != expected.slot {
        return Err(format!(
            "layer {layer} requires DeltaNet slot {}; got {slot}",
            expected.slot
        ));
    }
    let ordered_dispatches = dispatch_abi(slot);
    let conv_offset = slot * CONV_SLOT_ELEMENTS;
    let recurrent_offset = slot * RECURRENT_SLOT_ELEMENTS;
    let component = DeltaNetLayerComponentAbi {
        layer,
        slot,
        component_only: true,
        synthetic_input_accepted: false,
        direct_packed_tensors: expected_direct_packed_tensors(layer),
        caller_owned_buffers: caller_owned_buffers(slot),
        state_journal: StateJournalContract {
            active_conv_slot_offset_elements: conv_offset,
            active_recurrent_slot_offset_elements: recurrent_offset,
            rollback_conv_slot_offset_elements: conv_offset,
            rollback_recurrent_slot_offset_elements: recurrent_offset,
            snapshot_before_dispatch_zero_required: true,
            conv_state_committed_once_at_dispatch_ordinal: 3,
            recurrent_state_committed_once_at_dispatch_ordinal: 5,
            restore_both_rollback_ranges_on_rejected_token_required: true,
            state_slices_may_alias_other_deltanet_slots: false,
        },
        same_token_command_buffer_required: true,
        total_dispatches: ordered_dispatches.len(),
        ordered_dispatches,
    };
    validate_component_abi(&component)?;
    Ok(component)
}

fn validate_component_abi(component: &DeltaNetLayerComponentAbi) -> Result<(), String> {
    let expected = expected_deltanet_schedule()
        .into_iter()
        .find(|entry| entry.layer == component.layer)
        .ok_or_else(|| format!("component names non-DeltaNet layer {}", component.layer))?;
    if component.slot != expected.slot
        || !component.component_only
        || component.synthetic_input_accepted
    {
        return Err("component layer/slot or component-only boundary drifted".into());
    }
    if component.direct_packed_tensors != expected_direct_packed_tensors(component.layer) {
        return Err("component direct-packed tensor ABI drifted".into());
    }
    let expected_dispatches = dispatch_abi(component.slot);
    if component.ordered_dispatches != expected_dispatches
        || component.total_dispatches != 9
        || !component.same_token_command_buffer_required
    {
        return Err("component nine-dispatch same-command-buffer ABI drifted".into());
    }
    let expected_conv_offset = component.slot * CONV_SLOT_ELEMENTS;
    let expected_recurrent_offset = component.slot * RECURRENT_SLOT_ELEMENTS;
    if component.state_journal.active_conv_slot_offset_elements != expected_conv_offset
        || component.state_journal.rollback_conv_slot_offset_elements != expected_conv_offset
        || component
            .state_journal
            .active_recurrent_slot_offset_elements
            != expected_recurrent_offset
        || component
            .state_journal
            .rollback_recurrent_slot_offset_elements
            != expected_recurrent_offset
        || !component
            .state_journal
            .snapshot_before_dispatch_zero_required
        || component
            .state_journal
            .conv_state_committed_once_at_dispatch_ordinal
            != 3
        || component
            .state_journal
            .recurrent_state_committed_once_at_dispatch_ordinal
            != 5
        || !component
            .state_journal
            .restore_both_rollback_ranges_on_rejected_token_required
        || component
            .state_journal
            .state_slices_may_alias_other_deltanet_slots
    {
        return Err("component state journal/slot isolation ABI drifted".into());
    }
    if component.caller_owned_buffers != caller_owned_buffers(component.slot) {
        return Err("component caller-owned hidden/conv/recurrent buffer ABI drifted".into());
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ScheduleAuthorityEvidence {
    canonical_path: String,
    document_sha256: String,
    schema: String,
    status: String,
    descriptor_inventory_document_sha256: String,
    descriptor_inventory_seal_sha256: String,
    metadata_only: bool,
    artifact_payload_opened_or_scanned_by_this_program: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct L0StrictBridgeSourceEvidence {
    canonical_path: String,
    source_sha256: String,
    bytes: usize,
    required_anchors: Vec<&'static str>,
    source_read_as_static_external_evidence_only: bool,
    source_executed_by_this_program: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SourceIdentity {
    model_id: &'static str,
    model_key: &'static str,
    source_repository: &'static str,
    source_revision: &'static str,
    descriptor_inventory_document_sha256: &'static str,
    descriptor_inventory_seal_sha256: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct FutureCpuDeviceParityReceiptContract {
    schema: &'static str,
    future_receipt_status: &'static str,
    source_and_artifact_bound_non_fixture_input_required: bool,
    synthetic_input_rejected: bool,
    full_layer_or_decoder_promotion_rejected: bool,
    same_token_command_buffer_required: bool,
    exact_nine_dispatches_required: bool,
    no_fallback_required: bool,
    required_f32le_sha256_fields: Vec<&'static str>,
    required_max_abs_error_fields: Vec<&'static str>,
    required_state_transition_fields: Vec<&'static str>,
    receipt_materialized_by_this_plan: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ClaimBoundary {
    component_host_only: bool,
    complete_transformer_layer_earned: bool,
    complete_token_or_decoder_earned: bool,
    gravity_server_launch_precondition_satisfied: bool,
    artifact_payload_open_or_scan_performed: bool,
    metal_device_or_dispatch_performed: bool,
    lease_registry_runtime_watcher_server_or_hcli_changed: bool,
    model_execution_or_token_generation_performed: bool,
    tps_or_tg_measurement_performed: bool,
    execution_status: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct FamilyComponentHostPlan {
    schema: &'static str,
    status: &'static str,
    source_identity: SourceIdentity,
    permanent_schedule_authority: ScheduleAuthorityEvidence,
    l0_strict_bridge_source_evidence: L0StrictBridgeSourceEvidence,
    exact_deltanet_layer_slots: Vec<DeltaNetLayerSlot>,
    generic_component_host_parameterization: &'static str,
    all_36_layer_components: Vec<DeltaNetLayerComponentAbi>,
    future_cpu_device_parity_receipt: FutureCpuDeviceParityReceiptContract,
    claim_boundary: ClaimBoundary,
    unsealed_preimage_sha256: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct FutureParityReceiptCandidate {
    schema: String,
    receipt_seal_sha256: String,
    layer: usize,
    slot: usize,
    source_and_artifact_bound_input: bool,
    synthetic_input: bool,
    component_only: bool,
    full_layer_or_decoder_promotion: bool,
    same_token_command_buffer: bool,
    dispatch_count: usize,
    fallback_used: bool,
    input_hidden_f32le_sha256: String,
    cpu_first_residual_f32le_sha256: String,
    device_first_residual_f32le_sha256: String,
    cpu_next_conv_state_f32le_sha256: String,
    device_next_conv_state_f32le_sha256: String,
    cpu_next_recurrent_state_f32le_sha256: String,
    device_next_recurrent_state_f32le_sha256: String,
    first_residual_max_abs_error: f64,
    conv_state_max_abs_error: f64,
    recurrent_state_max_abs_error: f64,
    conv_state_committed_once: bool,
    recurrent_state_committed_once: bool,
    rollback_bytes_captured: bool,
}

#[allow(dead_code)]
fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

#[allow(dead_code)]
fn validate_future_parity_receipt(
    candidate: &FutureParityReceiptCandidate,
    component: &DeltaNetLayerComponentAbi,
) -> Result<(), String> {
    if candidate.schema != "hawking.ascension.qwen80_deltanet_family_cpu_device_parity_receipt.v1"
        || !is_lower_sha256(&candidate.receipt_seal_sha256)
        || candidate.layer != component.layer
        || candidate.slot != component.slot
    {
        return Err("future parity receipt schema, seal, layer, or slot drifted".into());
    }
    if !candidate.source_and_artifact_bound_input
        || candidate.synthetic_input
        || !candidate.component_only
        || candidate.full_layer_or_decoder_promotion
        || !candidate.same_token_command_buffer
        || candidate.dispatch_count != component.total_dispatches
        || candidate.fallback_used
    {
        return Err(
            "future parity receipt violates source/non-fixture/component-only boundary".into(),
        );
    }
    for (label, value) in [
        ("input hidden", &candidate.input_hidden_f32le_sha256),
        (
            "CPU first residual",
            &candidate.cpu_first_residual_f32le_sha256,
        ),
        (
            "device first residual",
            &candidate.device_first_residual_f32le_sha256,
        ),
        (
            "CPU next convolution state",
            &candidate.cpu_next_conv_state_f32le_sha256,
        ),
        (
            "device next convolution state",
            &candidate.device_next_conv_state_f32le_sha256,
        ),
        (
            "CPU next recurrent state",
            &candidate.cpu_next_recurrent_state_f32le_sha256,
        ),
        (
            "device next recurrent state",
            &candidate.device_next_recurrent_state_f32le_sha256,
        ),
    ] {
        if !is_lower_sha256(value) {
            return Err(format!("future parity receipt has invalid {label} digest"));
        }
    }
    if !candidate.first_residual_max_abs_error.is_finite()
        || !candidate.conv_state_max_abs_error.is_finite()
        || !candidate.recurrent_state_max_abs_error.is_finite()
        || candidate.first_residual_max_abs_error < 0.0
        || candidate.conv_state_max_abs_error < 0.0
        || candidate.recurrent_state_max_abs_error < 0.0
        || !candidate.conv_state_committed_once
        || !candidate.recurrent_state_committed_once
        || !candidate.rollback_bytes_captured
    {
        return Err("future parity receipt lacks finite parity/state/rollback evidence".into());
    }
    Ok(())
}

fn validate_schedule_authority(authority: &ScheduleAuthority) -> Result<(), String> {
    if authority.schema != SCHEDULE_SCHEMA || authority.status != SCHEDULE_STATUS {
        return Err("schedule authority schema or prepared-only status drifted".into());
    }
    let source = &authority.source_authority;
    if source.model_id != MODEL_ID
        || source.model_key != MODEL_KEY
        || source.source_repository != SOURCE_REPOSITORY
        || source.source_revision != SOURCE_REVISION
        || source.descriptor_inventory_document_sha256 != DESCRIPTOR_DOCUMENT_SHA
        || source.descriptor_inventory_seal_sha256 != MANIFEST_SEAL
    {
        return Err("schedule authority source/artifact identity drifted".into());
    }
    let geometry = &authority.geometry;
    if geometry.layer_count != LAYER_COUNT
        || geometry.hidden_size != HIDDEN
        || geometry.linear_key_heads != KEY_HEADS
        || geometry.linear_value_heads != VALUE_HEADS
        || geometry.linear_key_head_dim != KEY_HEAD_DIM
        || geometry.linear_value_head_dim != VALUE_HEAD_DIM
        || geometry.linear_conv_kernel != CONV_KERNEL
        || geometry.direct_pack_group_size != GROUP_SIZE
        || !authority.all_48_layers_scheduled
        || !authority.all_descriptors_source_artifact_bound
    {
        return Err(
            "schedule authority 48-layer DeltaNet geometry or descriptor binding drifted".into(),
        );
    }
    if authority.layers.len() != LAYER_COUNT {
        return Err("schedule authority does not contain exactly 48 layers".into());
    }
    for (layer_index, layer) in authority.layers.iter().enumerate() {
        if layer.layer != layer_index || layer.mixer != Mixer::expected(layer_index) {
            return Err(format!(
                "schedule authority mixer cadence drifted at layer {layer_index}"
            ));
        }
    }
    let expected_schedule = expected_deltanet_schedule();
    if authority.deltanet_state_slots.len() != DELTANET_LAYERS {
        return Err("schedule authority does not contain exactly 36 DeltaNet state slots".into());
    }
    let mut state_layers = BTreeSet::new();
    let mut state_slots = BTreeSet::new();
    for (expected, state) in expected_schedule
        .iter()
        .zip(&authority.deltanet_state_slots)
    {
        validate_schedule_state_slot(state, expected.layer, expected.slot)?;
        state_layers.insert(state.layer);
        state_slots.insert(state.slot);
    }
    if state_layers.len() != DELTANET_LAYERS || state_slots.len() != DELTANET_LAYERS {
        return Err("schedule authority DeltaNet states duplicate a layer or slot".into());
    }
    for expected in expected_schedule {
        let layer = authority
            .layers
            .get(expected.layer)
            .ok_or_else(|| format!("schedule authority omits layer {}", expected.layer))?;
        if layer.mixer != Mixer::DeltaNet {
            return Err(format!(
                "schedule authority maps layer {} as non-DeltaNet",
                expected.layer
            ));
        }
        validate_schedule_state_slot(&layer.state_slot, expected.layer, expected.slot)?;
        validate_schedule_tensors(layer)?;
    }
    Ok(())
}

fn validate_schedule_state_slot(
    state: &ScheduleStateSlot,
    expected_layer: usize,
    expected_slot: usize,
) -> Result<(), String> {
    if state.layer != expected_layer
        || state.slot != expected_slot
        || state.domain != "delta_net_conv_and_recurrent"
        || state.device_buffers_required_before_execution
            != ["deltanet_conv_history", "deltanet_recurrent_state"]
        || state.rollback_buffers_required_before_execution
            != [
                "deltanet_conv_history_rollback",
                "deltanet_recurrent_state_rollback",
            ]
        || state.state_materialized_by_this_plan
    {
        return Err(format!(
            "schedule authority state mapping drifted for DeltaNet layer {expected_layer} / slot {expected_slot}"
        ));
    }
    Ok(())
}

fn validate_schedule_tensors(layer: &ScheduleLayer) -> Result<(), String> {
    let expected = expected_direct_packed_tensors(layer.layer);
    let input = &expected[0];
    if layer.input_layernorm.role != input.role
        || layer.input_layernorm.tensor_name != input.tensor_name
        || layer.input_layernorm.shape != input.shape
        || layer.mixer_tensor_execution_order != expected_mixer_order()
        || layer.mixer_tensors.len() != expected.len() - 1
    {
        return Err(format!(
            "schedule authority direct-packed input/QKVZ/BA/conv/recurrent/gated-norm/out order drifted at layer {}",
            layer.layer
        ));
    }
    for (actual, expected) in layer.mixer_tensors.iter().zip(expected.iter().skip(1)) {
        if actual.role != expected.role
            || actual.tensor_name != expected.tensor_name
            || actual.shape != expected.shape
        {
            return Err(format!(
                "schedule authority direct-packed tensor binding drifted at layer {} role {}",
                layer.layer, expected.role
            ));
        }
    }
    Ok(())
}

fn validate_l0_strict_bridge_source(bytes: &[u8]) -> Result<(), String> {
    let source = std::str::from_utf8(bytes)
        .map_err(|_| "L0 strict bridge source is not UTF-8 Rust source".to_owned())?;
    for anchor in L0_REQUIRED_SOURCE_ANCHORS {
        if !source.contains(anchor) {
            return Err(format!(
                "L0 strict bridge source lacks required anchor {anchor:?}"
            ));
        }
    }
    Ok(())
}

fn regular_file_bytes(path: &Path, label: &str) -> Result<(PathBuf, Vec<u8>), String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be an absolute path"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    let canonical = fs::canonicalize(path)
        .map_err(|error| format!("cannot canonicalize {label} {}: {error}", path.display()))?;
    let bytes = fs::read(&canonical)
        .map_err(|error| format!("cannot read {label} {}: {error}", canonical.display()))?;
    Ok((canonical, bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn build_plan(
    schedule_path: PathBuf,
    schedule_bytes: &[u8],
    authority: &ScheduleAuthority,
    bridge_path: PathBuf,
    bridge_bytes: &[u8],
) -> Result<FamilyComponentHostPlan, String> {
    validate_schedule_authority(authority)?;
    validate_l0_strict_bridge_source(bridge_bytes)?;
    let exact_deltanet_layer_slots = expected_deltanet_schedule();
    let all_36_layer_components = exact_deltanet_layer_slots
        .iter()
        .map(|entry| build_layer_component(entry.layer, entry.slot))
        .collect::<Result<Vec<_>, _>>()?;
    if all_36_layer_components.len() != DELTANET_LAYERS {
        return Err("generic host did not construct all 36 DeltaNet ABI slots".into());
    }
    let mut plan = FamilyComponentHostPlan {
        schema: RESULT_SCHEMA,
        status: RESULT_STATUS,
        source_identity: SourceIdentity {
            model_id: MODEL_ID,
            model_key: MODEL_KEY,
            source_repository: SOURCE_REPOSITORY,
            source_revision: SOURCE_REVISION,
            descriptor_inventory_document_sha256: DESCRIPTOR_DOCUMENT_SHA,
            descriptor_inventory_seal_sha256: MANIFEST_SEAL,
        },
        permanent_schedule_authority: ScheduleAuthorityEvidence {
            canonical_path: schedule_path.display().to_string(),
            document_sha256: sha256_hex(schedule_bytes),
            schema: authority.schema.clone(),
            status: authority.status.clone(),
            descriptor_inventory_document_sha256: authority
                .source_authority
                .descriptor_inventory_document_sha256
                .clone(),
            descriptor_inventory_seal_sha256: authority
                .source_authority
                .descriptor_inventory_seal_sha256
                .clone(),
            metadata_only: true,
            artifact_payload_opened_or_scanned_by_this_program: false,
        },
        l0_strict_bridge_source_evidence: L0StrictBridgeSourceEvidence {
            canonical_path: bridge_path.display().to_string(),
            source_sha256: sha256_hex(bridge_bytes),
            bytes: bridge_bytes.len(),
            required_anchors: L0_REQUIRED_SOURCE_ANCHORS.to_vec(),
            source_read_as_static_external_evidence_only: true,
            source_executed_by_this_program: false,
        },
        exact_deltanet_layer_slots,
        generic_component_host_parameterization:
            "validated (layer, slot) for exactly source-selected DeltaNet layers; caller supplies retained hidden, compact direct-packed payloads, active+rollback conv/recurrent state, scratch, and one open TokenCommandBuffer",
        all_36_layer_components,
        future_cpu_device_parity_receipt: FutureCpuDeviceParityReceiptContract {
            schema: "hawking.ascension.qwen80_deltanet_family_cpu_device_parity_receipt.v1",
            future_receipt_status:
                "FUTURE_SEALED_SOURCE_ARTIFACT_BOUND_NON_FIXTURE_DELTANET_COMPONENT_PARITY_REQUIRED",
            source_and_artifact_bound_non_fixture_input_required: true,
            synthetic_input_rejected: true,
            full_layer_or_decoder_promotion_rejected: true,
            same_token_command_buffer_required: true,
            exact_nine_dispatches_required: true,
            no_fallback_required: true,
            required_f32le_sha256_fields: vec![
                "input_hidden_f32le_sha256",
                "cpu_first_residual_f32le_sha256",
                "device_first_residual_f32le_sha256",
                "cpu_next_conv_state_f32le_sha256",
                "device_next_conv_state_f32le_sha256",
                "cpu_next_recurrent_state_f32le_sha256",
                "device_next_recurrent_state_f32le_sha256",
            ],
            required_max_abs_error_fields: vec![
                "first_residual_max_abs_error",
                "conv_state_max_abs_error",
                "recurrent_state_max_abs_error",
            ],
            required_state_transition_fields: vec![
                "conv_state_committed_once",
                "recurrent_state_committed_once",
                "rollback_bytes_captured",
            ],
            receipt_materialized_by_this_plan: false,
        },
        claim_boundary: ClaimBoundary {
            component_host_only: true,
            complete_transformer_layer_earned: false,
            complete_token_or_decoder_earned: false,
            gravity_server_launch_precondition_satisfied: false,
            artifact_payload_open_or_scan_performed: false,
            metal_device_or_dispatch_performed: false,
            lease_registry_runtime_watcher_server_or_hcli_changed: false,
            model_execution_or_token_generation_performed: false,
            tps_or_tg_measurement_performed: false,
            execution_status: EXECUTION_STATUS,
        },
        unsealed_preimage_sha256: String::new(),
    };
    plan.unsealed_preimage_sha256 = sha256_hex(
        &serde_json::to_vec(&plan)
            .map_err(|error| format!("cannot serialize component-host preimage: {error}"))?,
    );
    Ok(plan)
}

fn write_new_json(path: &Path, value: &FamilyComponentHostPlan) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("--out must be an absolute path".into());
    }
    let parent = path.parent().ok_or("--out has no parent")?;
    let parent_metadata = fs::symlink_metadata(parent)
        .map_err(|error| format!("cannot stat --out parent {}: {error}", parent.display()))?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
        return Err("--out parent must be an existing non-symlink directory".into());
    }
    let serialized = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize component-host plan: {error}"))?;
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|error| format!("cannot create --out {}: {error}", path.display()))?;
    output
        .write_all(&serialized)
        .and_then(|_| output.write_all(b"\n"))
        .and_then(|_| output.sync_all())
        .map_err(|error| format!("cannot durably write --out {}: {error}", path.display()))
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_deltanet_family_component_host \\
--schedule-authority ABSOLUTE_JSON \\
--l0-strict-bridge-source ABSOLUTE_RUST_SOURCE \\
--out ABSOLUTE_NEW_JSON"
}

fn parse_args() -> Result<Args, String> {
    let mut values = std::collections::BTreeMap::<String, PathBuf>::new();
    let mut arguments = env::args().skip(1);
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--schedule-authority" | "--l0-strict-bridge-source" | "--out" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| format!("{flag} requires a value; {}", usage()))?;
                if values.insert(flag.clone(), PathBuf::from(value)).is_some() {
                    return Err(format!("{flag} repeated; {}", usage()));
                }
            }
            "--help" | "-h" => return Err(usage().into()),
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        }
    }
    let required = |flag: &str| -> Result<PathBuf, String> {
        let path = values
            .get(flag)
            .cloned()
            .ok_or_else(|| format!("missing {flag}; {}", usage()))?;
        if !path.is_absolute() {
            return Err(format!("{flag} must be an absolute path"));
        }
        Ok(path)
    };
    Ok(Args {
        schedule_authority: required("--schedule-authority")?,
        l0_strict_bridge_source: required("--l0-strict-bridge-source")?,
        out: required("--out")?,
    })
}

fn run(args: &Args) -> Result<(), String> {
    let (schedule_path, schedule_bytes) =
        regular_file_bytes(&args.schedule_authority, "--schedule-authority")?;
    let authority: ScheduleAuthority = serde_json::from_slice(&schedule_bytes)
        .map_err(|error| format!("cannot parse --schedule-authority: {error}"))?;
    let (bridge_path, bridge_bytes) =
        regular_file_bytes(&args.l0_strict_bridge_source, "--l0-strict-bridge-source")?;
    let plan = build_plan(
        schedule_path,
        &schedule_bytes,
        &authority,
        bridge_path,
        &bridge_bytes,
    )?;
    write_new_json(&args.out, &plan)
}

fn main() {
    let result = parse_args().and_then(|args| run(&args));
    if let Err(error) = result {
        eprintln!("ascension_qwen80_deltanet_family_component_host: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule_state(layer: usize, slot: usize) -> ScheduleStateSlot {
        ScheduleStateSlot {
            layer,
            slot,
            domain: "delta_net_conv_and_recurrent".into(),
            device_buffers_required_before_execution: vec![
                "deltanet_conv_history".into(),
                "deltanet_recurrent_state".into(),
            ],
            rollback_buffers_required_before_execution: vec![
                "deltanet_conv_history_rollback".into(),
                "deltanet_recurrent_state_rollback".into(),
            ],
            state_materialized_by_this_plan: false,
        }
    }

    fn schedule_tensor(binding: &DirectPackedTensorAbi) -> ScheduleTensor {
        ScheduleTensor {
            role: binding.role.clone(),
            tensor_name: binding.tensor_name.clone(),
            shape: binding.shape.clone(),
        }
    }

    fn valid_schedule_authority() -> ScheduleAuthority {
        let mut delta_slot = 0;
        let mut layers = Vec::with_capacity(LAYER_COUNT);
        for layer in 0..LAYER_COUNT {
            let mixer = Mixer::expected(layer);
            let state_slot = if mixer == Mixer::DeltaNet {
                let slot = delta_slot;
                delta_slot += 1;
                schedule_state(layer, slot)
            } else {
                ScheduleStateSlot {
                    layer,
                    slot: layer / 4,
                    domain: "gqa_kv".into(),
                    device_buffers_required_before_execution: vec!["gqa_key_cache".into()],
                    rollback_buffers_required_before_execution: vec![
                        "gqa_key_cache_rollback".into()
                    ],
                    state_materialized_by_this_plan: false,
                }
            };
            let tensors = if mixer == Mixer::DeltaNet {
                expected_direct_packed_tensors(layer)
            } else {
                Vec::new()
            };
            layers.push(ScheduleLayer {
                layer,
                mixer,
                state_slot,
                input_layernorm: tensors
                    .first()
                    .map(schedule_tensor)
                    .unwrap_or(ScheduleTensor {
                        role: "gqa_unused_input".into(),
                        tensor_name: format!("model.layers.{layer}.input_layernorm.weight"),
                        shape: vec![HIDDEN],
                    }),
                mixer_tensor_execution_order: (mixer == Mixer::DeltaNet)
                    .then(expected_mixer_order)
                    .unwrap_or_default(),
                mixer_tensors: tensors.iter().skip(1).map(schedule_tensor).collect(),
            });
        }
        ScheduleAuthority {
            schema: SCHEDULE_SCHEMA.into(),
            status: SCHEDULE_STATUS.into(),
            source_authority: ScheduleSourceAuthority {
                model_id: MODEL_ID.into(),
                model_key: MODEL_KEY.into(),
                source_repository: SOURCE_REPOSITORY.into(),
                source_revision: SOURCE_REVISION.into(),
                descriptor_inventory_document_sha256: DESCRIPTOR_DOCUMENT_SHA.into(),
                descriptor_inventory_seal_sha256: MANIFEST_SEAL.into(),
            },
            geometry: ScheduleGeometry {
                layer_count: LAYER_COUNT,
                hidden_size: HIDDEN,
                linear_key_heads: KEY_HEADS,
                linear_value_heads: VALUE_HEADS,
                linear_key_head_dim: KEY_HEAD_DIM,
                linear_value_head_dim: VALUE_HEAD_DIM,
                linear_conv_kernel: CONV_KERNEL,
                direct_pack_group_size: GROUP_SIZE,
            },
            deltanet_state_slots: expected_deltanet_schedule()
                .into_iter()
                .map(|entry| schedule_state(entry.layer, entry.slot))
                .collect(),
            layers,
            all_48_layers_scheduled: true,
            all_descriptors_source_artifact_bound: true,
        }
    }

    fn valid_future_receipt(layer: usize, slot: usize) -> FutureParityReceiptCandidate {
        FutureParityReceiptCandidate {
            schema: "hawking.ascension.qwen80_deltanet_family_cpu_device_parity_receipt.v1".into(),
            receipt_seal_sha256: "1".repeat(64),
            layer,
            slot,
            source_and_artifact_bound_input: true,
            synthetic_input: false,
            component_only: true,
            full_layer_or_decoder_promotion: false,
            same_token_command_buffer: true,
            dispatch_count: 9,
            fallback_used: false,
            input_hidden_f32le_sha256: "2".repeat(64),
            cpu_first_residual_f32le_sha256: "3".repeat(64),
            device_first_residual_f32le_sha256: "4".repeat(64),
            cpu_next_conv_state_f32le_sha256: "5".repeat(64),
            device_next_conv_state_f32le_sha256: "6".repeat(64),
            cpu_next_recurrent_state_f32le_sha256: "7".repeat(64),
            device_next_recurrent_state_f32le_sha256: "8".repeat(64),
            first_residual_max_abs_error: 0.0,
            conv_state_max_abs_error: 0.0,
            recurrent_state_max_abs_error: 0.0,
            conv_state_committed_once: true,
            recurrent_state_committed_once: true,
            rollback_bytes_captured: true,
        }
    }

    #[test]
    fn exact_source_schedule_is_thirty_six_deltanet_slots() {
        let schedule = expected_deltanet_schedule();
        assert_eq!(schedule.len(), 36);
        assert_eq!(schedule[0], DeltaNetLayerSlot { layer: 0, slot: 0 });
        assert_eq!(schedule[3], DeltaNetLayerSlot { layer: 4, slot: 3 });
        assert_eq!(
            schedule[35],
            DeltaNetLayerSlot {
                layer: 46,
                slot: 35
            }
        );
        validate_schedule_authority(&valid_schedule_authority()).unwrap();
    }

    #[test]
    fn generic_layer_slot_abi_binds_direct_packed_state_and_nine_dispatches() {
        let component = build_layer_component(4, 3).unwrap();
        assert_eq!(component.direct_packed_tensors.len(), 8);
        assert_eq!(component.total_dispatches, 9);
        assert_eq!(
            component.ordered_dispatches[3].operation,
            DeltaNetDispatchOperation::QkvzRearrangeCausalConvL2AndConvStateCommit
        );
        assert_eq!(
            component.state_journal.active_conv_slot_offset_elements,
            3 * CONV_SLOT_ELEMENTS
        );
        assert_eq!(
            component
                .state_journal
                .active_recurrent_slot_offset_elements,
            3 * RECURRENT_SLOT_ELEMENTS
        );
        assert!(component
            .caller_owned_buffers
            .iter()
            .all(|buffer| buffer.retain_through_same_token_command_buffer_fence));
    }

    #[test]
    fn rejects_non_deltanet_layer_or_wrong_slot() {
        assert!(build_layer_component(3, 0).is_err());
        assert!(build_layer_component(4, 4).is_err());
    }

    #[test]
    fn rejects_schedule_tampering_of_deltanet_tensor_or_state_mapping() {
        let mut authority = valid_schedule_authority();
        authority.layers[4].mixer_tensors[0].shape = vec![12_287, HIDDEN];
        assert!(validate_schedule_authority(&authority).is_err());

        let mut authority = valid_schedule_authority();
        authority.deltanet_state_slots[3].slot = 4;
        assert!(validate_schedule_authority(&authority).is_err());
    }

    #[test]
    fn l0_bridge_source_requires_real_prefix_anchors() {
        let valid = L0_REQUIRED_SOURCE_ANCHORS.join("\n");
        validate_l0_strict_bridge_source(valid.as_bytes()).unwrap();
        assert!(validate_l0_strict_bridge_source(b"synthetic bridge").is_err());
    }

    #[test]
    fn future_receipt_rejects_synthetic_or_full_layer_promotion() {
        let component = build_layer_component(0, 0).unwrap();
        validate_future_parity_receipt(&valid_future_receipt(0, 0), &component).unwrap();

        let mut synthetic = valid_future_receipt(0, 0);
        synthetic.synthetic_input = true;
        assert!(validate_future_parity_receipt(&synthetic, &component).is_err());

        let mut promoted = valid_future_receipt(0, 0);
        promoted.full_layer_or_decoder_promotion = true;
        assert!(validate_future_parity_receipt(&promoted, &component).is_err());
    }

    #[test]
    fn schedule_authority_does_not_promote_the_component_host() {
        let authority = valid_schedule_authority();
        let bridge = L0_REQUIRED_SOURCE_ANCHORS.join("\n");
        let plan = build_plan(
            PathBuf::from("/tmp/qwen80-schedule.json"),
            b"schedule-metadata-only",
            &authority,
            PathBuf::from("/tmp/qwen80-l0-bridge.rs"),
            bridge.as_bytes(),
        )
        .unwrap();
        assert_eq!(plan.status, RESULT_STATUS);
        assert_eq!(plan.all_36_layer_components.len(), 36);
        assert!(!plan.claim_boundary.complete_transformer_layer_earned);
        assert!(!plan.claim_boundary.complete_token_or_decoder_earned);
        assert!(!plan.claim_boundary.metal_device_or_dispatch_performed);
        assert!(
            !plan
                .future_cpu_device_parity_receipt
                .receipt_materialized_by_this_plan
        );
    }
}
