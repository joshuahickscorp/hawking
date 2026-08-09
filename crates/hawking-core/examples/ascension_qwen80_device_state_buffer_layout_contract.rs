//! CPU-only Qwen3-Coder-Next per-session device-state-buffer layout contract.
//!
//! This standalone target translates the existing Qwen80 hybrid decode-state
//! contract into an exact allocation target for a future decoder command
//! graph. It computes metadata only: no artifact is opened, no Metal buffer
//! is allocated, and no runtime, watcher, server, HCLI request, or benchmark
//! is touched.
//!
//! For each logical session it requires distinct active and rollback arenas:
//! 36 DeltaNet convolution slices [8192,3], 36 DeltaNet recurrent slices
//! [32,128,128], and 12 GQA K plus 12 GQA V slices [max_seq_len,2,256].
//! Every slot has an exact element offset, end capacity, byte span, and
//! source-schedule owner. Rollback mirrors every active domain in a separate
//! arena and may only commit after the entire 48-layer plus terminal path
//! succeeds.
//!
//! This result is a future allocation target only. It remains NOT_READY until
//! a separately sealed physical capture proves actual device allocation,
//! per-session isolation, full state transition parity, and transactional
//! rollback behavior.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "hawking.ascension.qwen80_device_state_buffer_layout_contract.v1";
const SOURCE_STATE_CONTRACT_SCHEMA: &str = "hawking.ascension.qwen80_decode_state_contract.v1";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_SEAL: &str = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const ADMISSION_RECEIPT_SEAL: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";
const MAX_NATIVE_CONTEXT: usize = 4_096;
const LAYER_COUNT: usize = 48;
const DELTANET_LAYERS: usize = 36;
const GQA_LAYERS: usize = 12;
const F32_BYTES: usize = 4;
const DELTANET_CONV_CHANNELS: usize = 8_192;
const DELTANET_CONV_HISTORY_TOKENS: usize = 3;
const DELTANET_VALUE_HEADS: usize = 32;
const DELTANET_KEY_HEAD_DIM: usize = 128;
const DELTANET_VALUE_HEAD_DIM: usize = 128;
const GQA_KV_HEADS: usize = 2;
const GQA_HEAD_DIM: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
enum Mixer {
    DeltaNet,
    Gqa,
}

impl Mixer {
    fn expected_for_layer(layer: usize) -> Self {
        if layer % 4 == 3 {
            Self::Gqa
        } else {
            Self::DeltaNet
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
enum StateDomain {
    DeltaNetConv,
    DeltaNetRecurrent,
    GqaKey,
    GqaValue,
}

impl StateDomain {
    fn as_str(self) -> &'static str {
        match self {
            Self::DeltaNetConv => "deltanet_conv_history",
            Self::DeltaNetRecurrent => "deltanet_recurrent",
            Self::GqaKey => "gqa_key_cache",
            Self::GqaValue => "gqa_value_cache",
        }
    }

    fn slot_shape(self, max_seq_len: usize) -> Vec<usize> {
        match self {
            Self::DeltaNetConv => vec![DELTANET_CONV_CHANNELS, DELTANET_CONV_HISTORY_TOKENS],
            Self::DeltaNetRecurrent => vec![
                DELTANET_VALUE_HEADS,
                DELTANET_KEY_HEAD_DIM,
                DELTANET_VALUE_HEAD_DIM,
            ],
            Self::GqaKey | Self::GqaValue => vec![max_seq_len, GQA_KV_HEADS, GQA_HEAD_DIM],
        }
    }

    fn slots(self) -> usize {
        match self {
            Self::DeltaNetConv | Self::DeltaNetRecurrent => DELTANET_LAYERS,
            Self::GqaKey | Self::GqaValue => GQA_LAYERS,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
enum Arena {
    Active,
    Rollback,
}

impl Arena {
    fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Rollback => "rollback",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SourceIdentity {
    model_id: String,
    model_key: String,
    source_repository: String,
    source_revision: String,
    manifest_seal_sha256: String,
    admission_receipt_seal_sha256: String,
    source_state_contract_schema: &'static str,
}

impl SourceIdentity {
    fn exact() -> Self {
        Self {
            model_id: MODEL_ID.into(),
            model_key: MODEL_KEY.into(),
            source_repository: SOURCE_REPOSITORY.into(),
            source_revision: SOURCE_REVISION.into(),
            manifest_seal_sha256: MANIFEST_SEAL.into(),
            admission_receipt_seal_sha256: ADMISSION_RECEIPT_SEAL.into(),
            source_state_contract_schema: SOURCE_STATE_CONTRACT_SCHEMA,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct BufferAllocation {
    arena: Arena,
    domain: StateDomain,
    allocation_id: String,
    element_type: &'static str,
    bytes_per_element: usize,
    slot_shape: Vec<usize>,
    slots: usize,
    slot_stride_elements: usize,
    allocation_capacity_elements: usize,
    allocation_capacity_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SlotRange {
    domain: StateDomain,
    arena: Arena,
    allocation_id: String,
    slot: usize,
    offset_elements: usize,
    capacity_elements: usize,
    offset_bytes: usize,
    capacity_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct LayerStateBinding {
    layer: usize,
    mixer: Mixer,
    state_slot: usize,
    active_ranges: Vec<SlotRange>,
    rollback_ranges: Vec<SlotRange>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SessionStateLayout {
    session_id: String,
    max_seq_len: usize,
    active_allocations: Vec<BufferAllocation>,
    rollback_allocations: Vec<BufferAllocation>,
    layer_bindings: Vec<LayerStateBinding>,
    active_total_elements: usize,
    active_total_bytes: usize,
    rollback_total_elements: usize,
    rollback_total_bytes: usize,
    per_session_total_elements: usize,
    per_session_total_bytes: usize,
    rollback_protocol: Vec<&'static str>,
}

#[derive(Serialize)]
struct BufferSummary {
    domain: &'static str,
    slot_shape: Vec<usize>,
    slot_count: usize,
    slot_stride_elements: usize,
    active_capacity_elements: usize,
    active_capacity_bytes: usize,
    rollback_capacity_elements: usize,
    rollback_capacity_bytes: usize,
}

#[derive(Serialize)]
struct LayoutReport {
    schema: &'static str,
    status: &'static str,
    ready_for_decoder_graph: bool,
    actual_device_allocation_performed: bool,
    actual_device_state_parity_performed: bool,
    actual_rollback_bytes_captured: bool,
    source_identity: SourceIdentity,
    max_seq_len: usize,
    native_max_seq_len: usize,
    schedule_layers: usize,
    deltanet_layers: usize,
    gqa_layers: usize,
    buffers: Vec<BufferSummary>,
    session_layout: SessionStateLayout,
    contract_checks: ContractChecks,
    required_before_ready: Vec<&'static str>,
    claim_boundary: Vec<&'static str>,
    unsealed_preimage_sha256: String,
}

#[derive(Serialize)]
struct ContractChecks {
    max_seq_len_within_native_bound: bool,
    exact_48_layer_schedule: bool,
    exact_36_deltanet_and_12_gqa_slots: bool,
    exact_offsets_and_capacities: bool,
    active_and_rollback_allocations_disjoint: bool,
    layer_slot_mapping_valid: bool,
    rollback_mirrors_active_layout: bool,
    no_device_allocation_or_runtime_claim: bool,
}

struct Args {
    max_seq_len: usize,
    session_id: String,
    out: PathBuf,
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{label} overflowed"))
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("{label} overflowed"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn product(shape: &[usize], label: &str) -> Result<usize, String> {
    shape
        .iter()
        .copied()
        .try_fold(1usize, |total, extent| checked_mul(total, extent, label))
}

fn validate_request(max_seq_len: usize, session_id: &str) -> Result<(), String> {
    if max_seq_len == 0 || max_seq_len > MAX_NATIVE_CONTEXT {
        return Err(format!(
            "max_seq_len must be within 1..={MAX_NATIVE_CONTEXT}, got {max_seq_len}"
        ));
    }
    if session_id.trim().is_empty() {
        return Err("session_id must be non-empty".into());
    }
    Ok(())
}

fn stride_elements(domain: StateDomain, max_seq_len: usize) -> Result<usize, String> {
    product(&domain.slot_shape(max_seq_len), domain.as_str())
}

fn allocation(
    session_id: &str,
    arena: Arena,
    domain: StateDomain,
    max_seq_len: usize,
) -> Result<BufferAllocation, String> {
    let stride = stride_elements(domain, max_seq_len)?;
    let capacity = checked_mul(domain.slots(), stride, "state allocation capacity")?;
    Ok(BufferAllocation {
        arena,
        domain,
        allocation_id: format!(
            "qwen80/session={session_id}/arena={}/domain={}",
            arena.as_str(),
            domain.as_str()
        ),
        element_type: "f32",
        bytes_per_element: F32_BYTES,
        slot_shape: domain.slot_shape(max_seq_len),
        slots: domain.slots(),
        slot_stride_elements: stride,
        allocation_capacity_elements: capacity,
        allocation_capacity_bytes: checked_mul(capacity, F32_BYTES, "state allocation bytes")?,
    })
}

fn expected_allocations(
    session_id: &str,
    arena: Arena,
    max_seq_len: usize,
) -> Result<Vec<BufferAllocation>, String> {
    [
        StateDomain::DeltaNetConv,
        StateDomain::DeltaNetRecurrent,
        StateDomain::GqaKey,
        StateDomain::GqaValue,
    ]
    .into_iter()
    .map(|domain| allocation(session_id, arena, domain, max_seq_len))
    .collect()
}

fn allocation_for<'a>(
    allocations: &'a [BufferAllocation],
    domain: StateDomain,
    arena: Arena,
) -> Result<&'a BufferAllocation, String> {
    allocations
        .iter()
        .find(|allocation| allocation.domain == domain && allocation.arena == arena)
        .ok_or_else(|| format!("missing {} {} allocation", arena.as_str(), domain.as_str()))
}

fn slot_range(allocation: &BufferAllocation, slot: usize) -> Result<SlotRange, String> {
    if slot >= allocation.slots {
        return Err(format!(
            "{} {} slot {slot} is outside 0..{}",
            allocation.arena.as_str(),
            allocation.domain.as_str(),
            allocation.slots - 1
        ));
    }
    let offset_elements = checked_mul(
        slot,
        allocation.slot_stride_elements,
        "state slot offset elements",
    )?;
    let capacity_elements = checked_add(
        offset_elements,
        allocation.slot_stride_elements,
        "state slot capacity elements",
    )?;
    if capacity_elements > allocation.allocation_capacity_elements {
        return Err("state slot range exceeds its allocation capacity".into());
    }
    Ok(SlotRange {
        domain: allocation.domain,
        arena: allocation.arena,
        allocation_id: allocation.allocation_id.clone(),
        slot,
        offset_elements,
        capacity_elements,
        offset_bytes: checked_mul(offset_elements, F32_BYTES, "state slot offset bytes")?,
        capacity_bytes: checked_mul(capacity_elements, F32_BYTES, "state slot capacity bytes")?,
    })
}

fn layer_binding(
    layer: usize,
    active: &[BufferAllocation],
    rollback: &[BufferAllocation],
) -> Result<LayerStateBinding, String> {
    let mixer = Mixer::expected_for_layer(layer);
    let (slot, domains) = match mixer {
        Mixer::DeltaNet => (
            layer - layer / 4,
            [StateDomain::DeltaNetConv, StateDomain::DeltaNetRecurrent],
        ),
        Mixer::Gqa => (layer / 4, [StateDomain::GqaKey, StateDomain::GqaValue]),
    };
    let active_ranges = domains
        .iter()
        .copied()
        .map(|domain| slot_range(allocation_for(active, domain, Arena::Active)?, slot))
        .collect::<Result<Vec<_>, _>>()?;
    let rollback_ranges = domains
        .iter()
        .copied()
        .map(|domain| slot_range(allocation_for(rollback, domain, Arena::Rollback)?, slot))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LayerStateBinding {
        layer,
        mixer,
        state_slot: slot,
        active_ranges,
        rollback_ranges,
    })
}

fn total_capacity(allocations: &[BufferAllocation]) -> Result<(usize, usize), String> {
    let elements = allocations.iter().try_fold(0usize, |total, allocation| {
        checked_add(
            total,
            allocation.allocation_capacity_elements,
            "state arena total elements",
        )
    })?;
    let bytes = checked_mul(elements, F32_BYTES, "state arena total bytes")?;
    Ok((elements, bytes))
}

fn expected_layout(max_seq_len: usize, session_id: &str) -> Result<SessionStateLayout, String> {
    validate_request(max_seq_len, session_id)?;
    let active_allocations = expected_allocations(session_id, Arena::Active, max_seq_len)?;
    let rollback_allocations = expected_allocations(session_id, Arena::Rollback, max_seq_len)?;
    let layer_bindings = (0..LAYER_COUNT)
        .map(|layer| layer_binding(layer, &active_allocations, &rollback_allocations))
        .collect::<Result<Vec<_>, _>>()?;
    let (active_total_elements, active_total_bytes) = total_capacity(&active_allocations)?;
    let (rollback_total_elements, rollback_total_bytes) = total_capacity(&rollback_allocations)?;
    let per_session_total_elements = checked_add(
        active_total_elements,
        rollback_total_elements,
        "per-session active+rollback elements",
    )?;
    let per_session_total_bytes = checked_add(
        active_total_bytes,
        rollback_total_bytes,
        "per-session active+rollback bytes",
    )?;
    let layout = SessionStateLayout {
        session_id: session_id.into(),
        max_seq_len,
        active_allocations,
        rollback_allocations,
        layer_bindings,
        active_total_elements,
        active_total_bytes,
        rollback_total_elements,
        rollback_total_bytes,
        per_session_total_elements,
        per_session_total_bytes,
        rollback_protocol: vec![
            "checkpoint: copy every active state allocation into its same-domain rollback allocation before mutation",
            "execute: each of 48 layers mutates only its active source-owned state ranges",
            "commit: advance session state only after layer 47 and valid terminal feedback complete",
            "rollback: restore every active allocation from its rollback mirror on any failed layer, parity failure, sampler rejection, or interrupted token",
            "isolation: never share active or rollback allocation_id values across logical sessions",
        ],
    };
    validate_layout(&layout)?;
    Ok(layout)
}

fn validate_allocation(
    observed: &BufferAllocation,
    expected: &BufferAllocation,
) -> Result<(), String> {
    if observed != expected {
        return Err(format!(
            "{} {} allocation geometry/identity drifted",
            expected.arena.as_str(),
            expected.domain.as_str()
        ));
    }
    Ok(())
}

fn validate_ranges(
    observed: &[SlotRange],
    expected: &[SlotRange],
    label: &str,
) -> Result<(), String> {
    if observed != expected {
        return Err(format!(
            "{label} state ranges drifted from exact offset/capacity target"
        ));
    }
    let mut allocation_ids = BTreeSet::new();
    for range in observed {
        if !allocation_ids.insert(range.allocation_id.as_str()) {
            return Err(format!("{label} has duplicated allocation ids"));
        }
        if range.capacity_elements <= range.offset_elements
            || range.capacity_bytes <= range.offset_bytes
            || range.offset_bytes != range.offset_elements * F32_BYTES
            || range.capacity_bytes != range.capacity_elements * F32_BYTES
        {
            return Err(format!("{label} has invalid byte/element capacity"));
        }
    }
    Ok(())
}

fn validate_layout(layout: &SessionStateLayout) -> Result<(), String> {
    validate_request(layout.max_seq_len, &layout.session_id)?;
    let expected = {
        // Construct manually to avoid recursive validation through expected_layout.
        let active_allocations =
            expected_allocations(&layout.session_id, Arena::Active, layout.max_seq_len)?;
        let rollback_allocations =
            expected_allocations(&layout.session_id, Arena::Rollback, layout.max_seq_len)?;
        let layer_bindings = (0..LAYER_COUNT)
            .map(|layer| layer_binding(layer, &active_allocations, &rollback_allocations))
            .collect::<Result<Vec<_>, _>>()?;
        let (active_total_elements, active_total_bytes) = total_capacity(&active_allocations)?;
        let (rollback_total_elements, rollback_total_bytes) =
            total_capacity(&rollback_allocations)?;
        SessionStateLayout {
            session_id: layout.session_id.clone(),
            max_seq_len: layout.max_seq_len,
            active_allocations,
            rollback_allocations,
            layer_bindings,
            active_total_elements,
            active_total_bytes,
            rollback_total_elements,
            rollback_total_bytes,
            per_session_total_elements: checked_add(
                active_total_elements,
                rollback_total_elements,
                "expected per-session elements",
            )?,
            per_session_total_bytes: checked_add(
                active_total_bytes,
                rollback_total_bytes,
                "expected per-session bytes",
            )?,
            rollback_protocol: vec![
                "checkpoint: copy every active state allocation into its same-domain rollback allocation before mutation",
                "execute: each of 48 layers mutates only its active source-owned state ranges",
                "commit: advance session state only after layer 47 and valid terminal feedback complete",
                "rollback: restore every active allocation from its rollback mirror on any failed layer, parity failure, sampler rejection, or interrupted token",
                "isolation: never share active or rollback allocation_id values across logical sessions",
            ],
        }
    };
    if layout.active_allocations.len() != 4 || layout.rollback_allocations.len() != 4 {
        return Err("exact layout requires four active plus four rollback allocations".into());
    }
    for observed in &layout.active_allocations {
        let expected_allocation =
            allocation_for(&expected.active_allocations, observed.domain, Arena::Active)?;
        validate_allocation(observed, expected_allocation)?;
    }
    for observed in &layout.rollback_allocations {
        let expected_allocation = allocation_for(
            &expected.rollback_allocations,
            observed.domain,
            Arena::Rollback,
        )?;
        validate_allocation(observed, expected_allocation)?;
    }
    let active_ids = layout
        .active_allocations
        .iter()
        .map(|allocation| allocation.allocation_id.as_str())
        .collect::<BTreeSet<_>>();
    let rollback_ids = layout
        .rollback_allocations
        .iter()
        .map(|allocation| allocation.allocation_id.as_str())
        .collect::<BTreeSet<_>>();
    if active_ids.len() != 4 || rollback_ids.len() != 4 || !active_ids.is_disjoint(&rollback_ids) {
        return Err(
            "active and rollback allocation identifiers must be unique and disjoint".into(),
        );
    }
    if layout.layer_bindings.len() != LAYER_COUNT {
        return Err("layout must bind exactly 48 layers".into());
    }
    let mut deltanet_slots = BTreeSet::new();
    let mut gqa_slots = BTreeSet::new();
    for (expected_layer, observed) in layout.layer_bindings.iter().enumerate() {
        let expected_binding = &expected.layer_bindings[expected_layer];
        if observed.layer != expected_layer
            || observed.mixer != Mixer::expected_for_layer(expected_layer)
            || observed.state_slot != expected_binding.state_slot
        {
            return Err(format!(
                "layer {expected_layer} schedule/slot mapping drifted from 3xDeltaNet/1xGQA"
            ));
        }
        validate_ranges(
            &observed.active_ranges,
            &expected_binding.active_ranges,
            "active layer binding",
        )?;
        validate_ranges(
            &observed.rollback_ranges,
            &expected_binding.rollback_ranges,
            "rollback layer binding",
        )?;
        match observed.mixer {
            Mixer::DeltaNet => {
                if !deltanet_slots.insert(observed.state_slot) {
                    return Err("DeltaNet state slot is reused".into());
                }
            }
            Mixer::Gqa => {
                if !gqa_slots.insert(observed.state_slot) {
                    return Err("GQA state slot is reused".into());
                }
            }
        }
    }
    if deltanet_slots.len() != DELTANET_LAYERS || gqa_slots.len() != GQA_LAYERS {
        return Err("layout does not own exact 36 DeltaNet + 12 GQA slots".into());
    }
    if layout.active_total_elements != expected.active_total_elements
        || layout.active_total_bytes != expected.active_total_bytes
        || layout.rollback_total_elements != expected.rollback_total_elements
        || layout.rollback_total_bytes != expected.rollback_total_bytes
        || layout.per_session_total_elements != expected.per_session_total_elements
        || layout.per_session_total_bytes != expected.per_session_total_bytes
        || layout.rollback_protocol != expected.rollback_protocol
    {
        return Err("layout total capacity or rollback protocol drifted".into());
    }
    Ok(())
}

fn buffer_summary(layout: &SessionStateLayout) -> Result<Vec<BufferSummary>, String> {
    [
        StateDomain::DeltaNetConv,
        StateDomain::DeltaNetRecurrent,
        StateDomain::GqaKey,
        StateDomain::GqaValue,
    ]
    .into_iter()
    .map(|domain| {
        let active = allocation_for(&layout.active_allocations, domain, Arena::Active)?;
        let rollback = allocation_for(&layout.rollback_allocations, domain, Arena::Rollback)?;
        Ok(BufferSummary {
            domain: domain.as_str(),
            slot_shape: active.slot_shape.clone(),
            slot_count: active.slots,
            slot_stride_elements: active.slot_stride_elements,
            active_capacity_elements: active.allocation_capacity_elements,
            active_capacity_bytes: active.allocation_capacity_bytes,
            rollback_capacity_elements: rollback.allocation_capacity_elements,
            rollback_capacity_bytes: rollback.allocation_capacity_bytes,
        })
    })
    .collect()
}

fn contract_checks(layout: &SessionStateLayout) -> ContractChecks {
    let max_seq_len_within_native_bound =
        layout.max_seq_len > 0 && layout.max_seq_len <= MAX_NATIVE_CONTEXT;
    let exact_48_layer_schedule =
        layout
            .layer_bindings
            .iter()
            .enumerate()
            .all(|(layer, binding)| {
                binding.layer == layer && binding.mixer == Mixer::expected_for_layer(layer)
            });
    let exact_36_deltanet_and_12_gqa_slots = layout
        .layer_bindings
        .iter()
        .filter(|binding| binding.mixer == Mixer::DeltaNet)
        .count()
        == DELTANET_LAYERS
        && layout
            .layer_bindings
            .iter()
            .filter(|binding| binding.mixer == Mixer::Gqa)
            .count()
            == GQA_LAYERS;
    let exact_offsets_and_capacities = validate_layout(layout).is_ok();
    let active_ids = layout
        .active_allocations
        .iter()
        .map(|allocation| allocation.allocation_id.as_str())
        .collect::<BTreeSet<_>>();
    let rollback_ids = layout
        .rollback_allocations
        .iter()
        .map(|allocation| allocation.allocation_id.as_str())
        .collect::<BTreeSet<_>>();
    let active_and_rollback_allocations_disjoint =
        active_ids.len() == 4 && rollback_ids.len() == 4 && active_ids.is_disjoint(&rollback_ids);
    let layer_slot_mapping_valid = exact_offsets_and_capacities;
    let rollback_mirrors_active_layout = layout.active_allocations.iter().all(|active| {
        allocation_for(&layout.rollback_allocations, active.domain, Arena::Rollback)
            .map(|rollback| {
                rollback.slot_shape == active.slot_shape
                    && rollback.slots == active.slots
                    && rollback.slot_stride_elements == active.slot_stride_elements
                    && rollback.allocation_capacity_elements == active.allocation_capacity_elements
                    && rollback.allocation_capacity_bytes == active.allocation_capacity_bytes
            })
            .unwrap_or(false)
    });
    ContractChecks {
        max_seq_len_within_native_bound,
        exact_48_layer_schedule,
        exact_36_deltanet_and_12_gqa_slots,
        exact_offsets_and_capacities,
        active_and_rollback_allocations_disjoint,
        layer_slot_mapping_valid,
        rollback_mirrors_active_layout,
        no_device_allocation_or_runtime_claim: true,
    }
}

fn report(max_seq_len: usize, session_id: &str) -> Result<LayoutReport, String> {
    let session_layout = expected_layout(max_seq_len, session_id)?;
    let checks = contract_checks(&session_layout);
    if !checks.max_seq_len_within_native_bound
        || !checks.exact_48_layer_schedule
        || !checks.exact_36_deltanet_and_12_gqa_slots
        || !checks.exact_offsets_and_capacities
        || !checks.active_and_rollback_allocations_disjoint
        || !checks.layer_slot_mapping_valid
        || !checks.rollback_mirrors_active_layout
    {
        return Err("derived state-buffer layout failed its exact contract checks".into());
    }
    let mut report = LayoutReport {
        schema: SCHEMA,
        status:
            "NOT_READY_NO_DEVICE_ALLOCATION_NO_STATE_PARITY_NO_ROLLBACK_CAPTURE_QWEN80_PER_SESSION_BUFFER_LAYOUT_CONTRACT",
        ready_for_decoder_graph: false,
        actual_device_allocation_performed: false,
        actual_device_state_parity_performed: false,
        actual_rollback_bytes_captured: false,
        source_identity: SourceIdentity::exact(),
        max_seq_len,
        native_max_seq_len: MAX_NATIVE_CONTEXT,
        schedule_layers: LAYER_COUNT,
        deltanet_layers: DELTANET_LAYERS,
        gqa_layers: GQA_LAYERS,
        buffers: buffer_summary(&session_layout)?,
        session_layout,
        contract_checks: checks,
        required_before_ready: vec![
            "Allocate all eight named f32 buffers for a concrete logical session on the device; do not alias active and rollback arenas or any two sessions.",
            "Bind every source-selected DeltaNet and GQA layer to its exact active offsets/capacities, then prove those bindings against admitted source/artifact identity.",
            "Capture CPU/device parity for state bytes and hidden outputs across all 48 layers for more than one decode position.",
            "Prove atomic checkpoint, commit, and restore of all four state domains, including failure after a partial layer sequence and before/after terminal sampling.",
            "Only then integrate valid tail-masked terminal feedback and independently qualify a full decoder token, HCLI, and clean TPS/TG evidence.",
        ],
        claim_boundary: vec![
            "This contract calculates buffer metadata only. It does not allocate a device buffer, move a state byte, execute a Qwen80 layer, produce a token, or inspect an artifact.",
            "The active/rollback identifiers are required future allocation names, not evidence that memory exists or rollback works.",
            "The result remains NOT_READY and cannot qualify Qwen80 for runtime, server residency, HCLI, BASE_TRUE_TPS, TG, capability, Agent OS, or tournament actions.",
        ],
        unsealed_preimage_sha256: String::new(),
    };
    report.unsealed_preimage_sha256 =
        sha256_hex(&serde_json::to_vec(&report).map_err(|error| error.to_string())?);
    Ok(report)
}

fn write_report_atomic(path: &Path, report: &LayoutReport) -> Result<(), Box<dyn Error>> {
    let parent = path.parent().ok_or("output path has no parent")?;
    if !parent.is_dir() {
        return Err(format!("output parent is missing: {}", parent.display()).into());
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(report)?)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_device_state_buffer_layout_contract \
--max-seq-len 1..4096 --session-id NONEMPTY --out ABSOLUTE_PATH"
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut max_seq_len = MAX_NATIVE_CONTEXT;
    let mut session_id = None;
    let mut out = None;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("missing value after {flag}; {}", usage()))?;
        match flag.as_str() {
            "--max-seq-len" => {
                max_seq_len = value
                    .parse::<usize>()
                    .map_err(|_| "--max-seq-len must be an unsigned integer")?;
            }
            "--session-id" => {
                if session_id.replace(value).is_some() {
                    return Err("--session-id repeated".into());
                }
            }
            "--out" => {
                if out.replace(PathBuf::from(value)).is_some() {
                    return Err("--out repeated".into());
                }
            }
            _ => return Err(format!("unsupported option {flag:?}; {}", usage()).into()),
        }
    }
    let session_id = session_id.ok_or("missing --session-id")?;
    let out = out.ok_or("missing --out")?;
    if !out.is_absolute() {
        return Err("--out must be absolute".into());
    }
    validate_request(max_seq_len, &session_id)?;
    Ok(Args {
        max_seq_len,
        session_id,
        out,
    })
}

fn main() {
    let result = parse_args().and_then(|args| {
        let rendered = report(args.max_seq_len, &args.session_id)?;
        write_report_atomic(&args.out, &rendered)
    });
    if let Err(error) = result {
        eprintln!("ascension_qwen80_device_state_buffer_layout_contract: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_context_layout_has_exact_source_geometry_and_capacity() {
        let layout = expected_layout(MAX_NATIVE_CONTEXT, "state-layout-test").unwrap();
        assert_eq!(layout.layer_bindings.len(), 48);
        assert_eq!(layout.active_total_elements, 70_090_752);
        assert_eq!(layout.active_total_bytes, 280_363_008);
        assert_eq!(layout.rollback_total_elements, 70_090_752);
        assert_eq!(layout.rollback_total_bytes, 280_363_008);
        assert_eq!(layout.per_session_total_elements, 140_181_504);
        assert_eq!(layout.per_session_total_bytes, 560_726_016);
        assert_eq!(
            allocation_for(
                &layout.active_allocations,
                StateDomain::DeltaNetConv,
                Arena::Active
            )
            .unwrap()
            .slot_stride_elements,
            24_576
        );
        assert_eq!(
            allocation_for(
                &layout.active_allocations,
                StateDomain::DeltaNetRecurrent,
                Arena::Active
            )
            .unwrap()
            .slot_stride_elements,
            524_288
        );
        assert_eq!(
            allocation_for(
                &layout.active_allocations,
                StateDomain::GqaKey,
                Arena::Active
            )
            .unwrap()
            .slot_stride_elements,
            2_097_152
        );
    }

    #[test]
    fn exact_schedule_maps_deltanet_and_gqa_slots_without_aliases() {
        let layout = expected_layout(16, "schedule-test").unwrap();
        assert_eq!(layout.layer_bindings[0].mixer, Mixer::DeltaNet);
        assert_eq!(layout.layer_bindings[0].state_slot, 0);
        assert_eq!(layout.layer_bindings[2].state_slot, 2);
        assert_eq!(layout.layer_bindings[3].mixer, Mixer::Gqa);
        assert_eq!(layout.layer_bindings[3].state_slot, 0);
        assert_eq!(layout.layer_bindings[47].mixer, Mixer::Gqa);
        assert_eq!(layout.layer_bindings[47].state_slot, 11);
        validate_layout(&layout).unwrap();
    }

    #[test]
    fn rejects_zero_and_above_native_max_context() {
        assert!(expected_layout(0, "bad-context").is_err());
        assert!(expected_layout(MAX_NATIVE_CONTEXT + 1, "bad-context").is_err());
        assert!(expected_layout(1, "").is_err());
    }

    #[test]
    fn rejects_wrong_offset_or_capacity() {
        let mut layout = expected_layout(64, "offset-test").unwrap();
        layout.layer_bindings[1].active_ranges[0].offset_elements += 1;
        assert!(validate_layout(&layout).is_err());

        let mut layout = expected_layout(64, "capacity-test").unwrap();
        layout.active_allocations[0].allocation_capacity_elements -= 1;
        assert!(validate_layout(&layout).is_err());
    }

    #[test]
    fn rejects_active_rollback_alias_and_wrong_slot_mapping() {
        let mut layout = expected_layout(8, "rollback-test").unwrap();
        layout.rollback_allocations[0].allocation_id =
            layout.active_allocations[0].allocation_id.clone();
        assert!(validate_layout(&layout).is_err());

        let mut layout = expected_layout(8, "slot-test").unwrap();
        layout.layer_bindings[4].state_slot = 0;
        assert!(validate_layout(&layout).is_err());
    }

    #[test]
    fn rollback_layout_exactly_mirrors_active_domain_geometry() {
        let layout = expected_layout(256, "mirror-test").unwrap();
        let checks = contract_checks(&layout);
        assert!(checks.rollback_mirrors_active_layout);
        assert!(checks.active_and_rollback_allocations_disjoint);
        assert!(checks.exact_offsets_and_capacities);
        assert_eq!(
            layout.layer_bindings[3].active_ranges[0].offset_elements,
            layout.layer_bindings[3].rollback_ranges[0].offset_elements
        );
        assert_ne!(
            layout.layer_bindings[3].active_ranges[0].allocation_id,
            layout.layer_bindings[3].rollback_ranges[0].allocation_id
        );
    }

    #[test]
    fn distinct_logical_sessions_have_disjoint_future_allocation_ids() {
        let first = expected_layout(128, "logical-session-a").unwrap();
        let second = expected_layout(128, "logical-session-b").unwrap();
        let first_ids = first
            .active_allocations
            .iter()
            .chain(&first.rollback_allocations)
            .map(|allocation| allocation.allocation_id.as_str())
            .collect::<BTreeSet<_>>();
        let second_ids = second
            .active_allocations
            .iter()
            .chain(&second.rollback_allocations)
            .map(|allocation| allocation.allocation_id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(first_ids.len(), 8);
        assert_eq!(second_ids.len(), 8);
        assert!(first_ids.is_disjoint(&second_ids));
    }

    #[test]
    fn report_is_explicitly_not_ready_without_actual_device_allocation() {
        let report = report(32, "not-ready-test").unwrap();
        assert!(!report.ready_for_decoder_graph);
        assert!(!report.actual_device_allocation_performed);
        assert!(!report.actual_device_state_parity_performed);
        assert!(!report.actual_rollback_bytes_captured);
        assert!(report.status.contains("NOT_READY"));
        assert!(report.contract_checks.no_device_allocation_or_runtime_claim);
    }
}
