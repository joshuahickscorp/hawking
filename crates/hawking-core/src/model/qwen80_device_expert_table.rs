//! Qwen80 512-way device expert table — Q30 device-expert-table pattern.
//!
//! One per-layer table of 512 gate/up/down gpuAddresses. Device `route_ids`
//! select the top-10 experts. The host does not bind a payload per expert.
//!
//! Three historical bind strategies (all still callable):
//!
//! - [`write_top10_address_table`]: 10-entry table, caller remaps route ids
//!   to `0..10`. No payload memcpy. Was the default token-loop rewrite.
//! - [`write_compact_selected_table`]: memcpy top-10 payloads into compact
//!   slabs (~16.7 MiB/layer) and write a 512-entry table so real route ids
//!   still index. Env-gated (`HAWKING_QWEN80_COMPACT_EXPERT_SLABS=1`).
//! - [`write_selected_expert_table`]: 512-entry table pointing at already-
//!   resident per-expert buffers. Unused by the live decode loop.
//!
//! The live default is now [`super::device_residency`]: one persistent
//! all-layer table (`48 * 512` entries), payloads in an LRU pool, token
//! loop writes only route ids. The Q4 triplet is a [`ResidentPayload`]
//! impl — the table/pool do not know about group-64 codes.

use super::device_residency::{AddressTableGeometry, PayloadLayout};
use super::qwen80_complete_runtime::{
    QWEN80_EXPERTS, QWEN80_HIDDEN, QWEN80_LAYERS, QWEN80_MOE_INTERMEDIATE,
};
use super::qwen80_uniform_q4_hybrid_decode::Qwen80UniformQ4StreamingCatalog;
use super::qwen_complete_binary::UNIFORM_Q4_GROUP_SIZE;
use crate::{Error, Result};
use std::time::Instant;

pub const QWEN80_DEVICE_EXPERT_KIND_UNIFORM_Q4: u32 = 3;
pub const QWEN80_DEVICE_EXPERT_TRIPLET_READY: u32 = 0b111;
pub const QWEN80_DEVICE_PROJ_GATE: u32 = 0;
pub const QWEN80_DEVICE_PROJ_UP: u32 = 1;
pub const QWEN80_DEVICE_PROJ_DOWN: u32 = 2;
pub const QWEN80_EXPERT_TABLE_TOP_K: usize = 10;

/// Default **off**. Set `HAWKING_QWEN80_COMPACT_EXPERT_SLABS=1` to restore
/// the host memcpy of top-10 payloads into compact slabs (the pre-velocity
/// pack path). Address-table bind is the default.
pub fn qwen80_compact_expert_slabs_enabled() -> bool {
    match std::env::var("HAWKING_QWEN80_COMPACT_EXPERT_SLABS") {
        Ok(raw) => {
            let trimmed = raw.trim();
            !(trimmed.is_empty()
                || trimmed.eq_ignore_ascii_case("0")
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off")
                || trimmed.eq_ignore_ascii_case("no"))
        }
        Err(_) => false,
    }
}

/// Default **on**. Set `HAWKING_QWEN80_DEVICE_EXPERT_TABLE=0` / `false` /
/// `off` / `no` to restore the host per-expert bind path for A/B.
pub fn qwen80_device_expert_table_enabled() -> bool {
    for key in [
        "HAWKING_QWEN80_DEVICE_EXPERT_TABLE",
        "HAWKING_QWEN80_DEVICE_ROUTE",
    ] {
        if let Ok(raw) = std::env::var(key) {
            let trimmed = raw.trim();
            if trimmed.eq_ignore_ascii_case("0")
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off")
                || trimmed.eq_ignore_ascii_case("no")
            {
                return false;
            }
            if !trimmed.is_empty() {
                return true;
            }
        }
    }
    true
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen80ExpertTableKernel {
    Serial,
    Rowblock,
    Simdgroup,
}

pub fn qwen80_expert_table_kernel() -> Qwen80ExpertTableKernel {
    match std::env::var("HAWKING_QWEN80_Q4_KERNEL")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "serial" => Qwen80ExpertTableKernel::Serial,
        "rowblock" => Qwen80ExpertTableKernel::Rowblock,
        "simdgroup" | "fast" => Qwen80ExpertTableKernel::Simdgroup,
        // Measured at the live [512,2048]/[2048,512] top_k=10 geometry:
        // simdgroup 3.67 tok/s, serial 3.45, rowblock 3.25 (token-identical).
        _ => Qwen80ExpertTableKernel::Simdgroup,
    }
}

pub const QWEN80_EXPERT_TABLE_KERNELS: [&str; 5] = [
    "qwen80_expert_table_uniform_q4_matvec_serial",
    "qwen80_expert_table_uniform_q4_matvec_rowblock",
    "qwen80_expert_table_uniform_q4_matvec_simdgroup",
    "qwen80_expert_table_silu_mul",
    "qwen80_expert_table_weighted_sum",
];

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Qwen80DeviceExpertTensorRef {
    pub primary_address: u64,
    pub secondary_address: u64,
    pub rows: u32,
    pub cols: u32,
    pub rank: u32,
    pub kind: u32,
    pub generation: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Qwen80DeviceExpertTriplet {
    pub gate: Qwen80DeviceExpertTensorRef,
    pub up: Qwen80DeviceExpertTensorRef,
    pub down: Qwen80DeviceExpertTensorRef,
    pub ready_mask: u32,
    pub generation: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Qwen80DeviceExpertMatvecParams {
    pub n_experts: u32,
    pub experts_per_token: u32,
    pub generation: u32,
    pub projection: u32,
    pub group_size: u32,
    pub max_rows: u32,
    pub input_base_elems: u32,
    pub input_stride_elems: u32,
    pub output_base_elems: u32,
    pub output_stride_elems: u32,
    pub pad0: u32,
    pub pad1: u32,
}

const _: () = assert!(std::mem::size_of::<Qwen80DeviceExpertTensorRef>() == 40);
const _: () = assert!(std::mem::size_of::<Qwen80DeviceExpertTriplet>() == 128);
const _: () = assert!(std::mem::size_of::<Qwen80DeviceExpertMatvecParams>() == 48);

unsafe impl bytemuck::Pod for Qwen80DeviceExpertTensorRef {}
unsafe impl bytemuck::Zeroable for Qwen80DeviceExpertTensorRef {}
unsafe impl bytemuck::Pod for Qwen80DeviceExpertTriplet {}
unsafe impl bytemuck::Zeroable for Qwen80DeviceExpertTriplet {}
unsafe impl bytemuck::Pod for Qwen80DeviceExpertMatvecParams {}
unsafe impl bytemuck::Zeroable for Qwen80DeviceExpertMatvecParams {}

fn table_error(message: impl Into<String>) -> Error {
    Error::Model(format!("qwen80 device expert table: {}", message.into()))
}

/// One Q4 expert projection: 512×2048 or 2048×512, same element count.
pub const QWEN80_EXPERT_PROJ_ELEMENTS: usize = QWEN80_MOE_INTERMEDIATE * QWEN80_HIDDEN;
pub const QWEN80_EXPERT_PROJ_GROUPS: usize = QWEN80_EXPERT_PROJ_ELEMENTS / UNIFORM_Q4_GROUP_SIZE;
pub const QWEN80_EXPERT_CODE_BYTES: usize = QWEN80_EXPERT_PROJ_GROUPS * (UNIFORM_Q4_GROUP_SIZE / 2);
pub const QWEN80_EXPERT_SCALE_BYTES: usize = QWEN80_EXPERT_PROJ_GROUPS * 2;
pub const QWEN80_EXPERT_TRIPLET_PAYLOAD_BYTES: u64 =
    3 * (QWEN80_EXPERT_CODE_BYTES as u64 + QWEN80_EXPERT_SCALE_BYTES as u64);

/// Q4-harness layout for the generic residency table. Isolated here so the
/// pool itself stays representation-agnostic.
pub fn qwen80_q4_payload_layout() -> PayloadLayout {
    PayloadLayout::triplet(
        QWEN80_DEVICE_EXPERT_KIND_UNIFORM_Q4,
        6,
        QWEN80_EXPERT_TRIPLET_PAYLOAD_BYTES,
    )
}

pub fn qwen80_q4_address_geometry() -> AddressTableGeometry {
    AddressTableGeometry {
        n_layers: QWEN80_LAYERS,
        n_experts: QWEN80_EXPERTS,
        layout: qwen80_q4_payload_layout(),
    }
}

fn expert_proj_name(layer: usize, expert: usize, proj: &str) -> String {
    format!("model.layers.{layer}.mlp.experts.{expert}.{proj}.weight")
}

fn tensor_ref(
    primary_address: u64,
    secondary_address: u64,
    rows: u32,
    cols: u32,
    generation: u32,
) -> Qwen80DeviceExpertTensorRef {
    Qwen80DeviceExpertTensorRef {
        primary_address,
        secondary_address,
        rows,
        cols,
        rank: 0,
        kind: QWEN80_DEVICE_EXPERT_KIND_UNIFORM_Q4,
        generation,
        pad: 0,
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub struct Qwen80ExpertGpuTriplet {
    pub gate_codes: crate::metal::PinnedBuffer,
    pub gate_scales: crate::metal::PinnedBuffer,
    pub up_codes: crate::metal::PinnedBuffer,
    pub up_scales: crate::metal::PinnedBuffer,
    pub down_codes: crate::metal::PinnedBuffer,
    pub down_scales: crate::metal::PinnedBuffer,
}

#[cfg(target_os = "macos")]
impl Qwen80ExpertGpuTriplet {
    pub fn resources(&self) -> [&crate::metal::PinnedBuffer; 6] {
        [
            &self.gate_codes,
            &self.gate_scales,
            &self.up_codes,
            &self.up_scales,
            &self.down_codes,
            &self.down_scales,
        ]
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub struct Qwen80DeviceExpertTableLease {
    pub table: crate::metal::PinnedBuffer,
    pub resources: Vec<crate::metal::PinnedBuffer>,
    pub generation: u32,
    pub n_experts: usize,
    pub layer: usize,
    pub table_byte_offset: u64,
    pub build_secs: f64,
}

#[cfg(target_os = "macos")]
impl Qwen80DeviceExpertTableLease {
    pub fn from_residency_bind(bind: super::device_residency::LayerBind) -> Self {
        Self {
            table: bind.table,
            resources: bind.resources,
            generation: bind.generation,
            n_experts: bind.n_experts,
            layer: bind.layer,
            table_byte_offset: bind.table_byte_offset,
            build_secs: 0.0,
        }
    }
}

#[cfg(target_os = "macos")]
impl super::device_residency::ResidentPayload for Qwen80ExpertGpuTriplet {
    fn payload_bytes(&self) -> u64 {
        QWEN80_EXPERT_TRIPLET_PAYLOAD_BYTES
    }

    fn write_entry(&self, generation: u32, dst: &mut [u8]) {
        let trip = triplet_from_gpu(self, generation);
        let bytes = bytemuck::bytes_of(&trip);
        dst[..bytes.len()].copy_from_slice(bytes);
    }

    fn append_resources(&self, dst: &mut Vec<crate::metal::PinnedBuffer>) {
        for resource in self.resources() {
            dst.push(resource.clone());
        }
    }
}

#[cfg(target_os = "macos")]
pub struct Qwen80DeviceExpertWaveWorkspace {
    pub input: crate::metal::PinnedBuffer,
    pub gate: crate::metal::PinnedBuffer,
    pub up: crate::metal::PinnedBuffer,
    pub activated: crate::metal::PinnedBuffer,
    pub down: crate::metal::PinnedBuffer,
    pub route_ids: crate::metal::PinnedBuffer,
    pub route_weights: crate::metal::PinnedBuffer,
    pub combined: crate::metal::PinnedBuffer,
}

#[cfg(target_os = "macos")]
impl Qwen80DeviceExpertWaveWorkspace {
    pub fn allocate(context: &crate::metal::MetalContext) -> Result<Self> {
        let hidden_bytes = QWEN80_HIDDEN * std::mem::size_of::<f32>();
        let mid_bytes =
            QWEN80_EXPERT_TABLE_TOP_K * QWEN80_MOE_INTERMEDIATE * std::mem::size_of::<f32>();
        let down_bytes = QWEN80_EXPERT_TABLE_TOP_K * QWEN80_HIDDEN * std::mem::size_of::<f32>();
        Ok(Self {
            input: context.new_buffer_checked(hidden_bytes)?,
            gate: context.new_buffer_checked(mid_bytes)?,
            up: context.new_buffer_checked(mid_bytes)?,
            activated: context.new_buffer_checked(mid_bytes)?,
            down: context.new_buffer_checked(down_bytes)?,
            route_ids: context
                .new_buffer_checked(QWEN80_EXPERT_TABLE_TOP_K * std::mem::size_of::<u32>())?,
            route_weights: context
                .new_buffer_checked(QWEN80_EXPERT_TABLE_TOP_K * std::mem::size_of::<f32>())?,
            combined: context.new_buffer_checked(hidden_bytes)?,
        })
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct SlabView {
    gate_codes: *mut u8,
    gate_scales: *mut u8,
    up_codes: *mut u8,
    up_scales: *mut u8,
    down_codes: *mut u8,
    down_scales: *mut u8,
}

#[cfg(target_os = "macos")]
unsafe impl Send for SlabView {}
#[cfg(target_os = "macos")]
unsafe impl Sync for SlabView {}

#[cfg(target_os = "macos")]
pub fn build_qwen80_device_expert_table(
    context: &crate::metal::MetalContext,
    catalog: &Qwen80UniformQ4StreamingCatalog,
    layer: usize,
) -> Result<Qwen80DeviceExpertTableLease> {
    let started = Instant::now();
    let generation = u32::try_from(layer.saturating_add(1))
        .map_err(|_| table_error("layer generation overflows u32"))?;
    let n_experts = QWEN80_EXPERTS;
    let code_slab = n_experts
        .checked_mul(QWEN80_EXPERT_CODE_BYTES)
        .ok_or_else(|| table_error("code slab overflows"))?;
    let scale_slab = n_experts
        .checked_mul(QWEN80_EXPERT_SCALE_BYTES)
        .ok_or_else(|| table_error("scale slab overflows"))?;

    let gate_codes = context.new_buffer_checked(code_slab)?;
    let gate_scales = context.new_buffer_checked(scale_slab)?;
    let up_codes = context.new_buffer_checked(code_slab)?;
    let up_scales = context.new_buffer_checked(scale_slab)?;
    let down_codes = context.new_buffer_checked(code_slab)?;
    let down_scales = context.new_buffer_checked(scale_slab)?;

    let ptrs = SlabView {
        gate_codes: gate_codes.contents() as *mut u8,
        gate_scales: gate_scales.contents() as *mut u8,
        up_codes: up_codes.contents() as *mut u8,
        up_scales: up_scales.contents() as *mut u8,
        down_codes: down_codes.contents() as *mut u8,
        down_scales: down_scales.contents() as *mut u8,
    };

    let workers = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 28))
        .unwrap_or(8);
    let chunk = n_experts.div_ceil(workers);
    let errors = std::sync::Mutex::new(Vec::<String>::new());
    std::thread::scope(|scope| {
        for worker in 0..workers {
            let start = worker * chunk;
            let end = (start + chunk).min(n_experts);
            if start >= end {
                continue;
            }
            let errors = &errors;
            scope.spawn(move || {
                for expert in start..end {
                    if let Err(error) = fill_one_expert(catalog, layer, expert, ptrs) {
                        if let Ok(mut slot) = errors.lock() {
                            slot.push(error.to_string());
                        }
                        return;
                    }
                }
            });
        }
    });
    let collected = errors
        .into_inner()
        .map_err(|_| table_error("worker error lock poisoned"))?;
    if let Some(first) = collected.into_iter().next() {
        return Err(table_error(first));
    }

    let gate_code_base = gate_codes.gpu_address();
    let gate_scale_base = gate_scales.gpu_address();
    let up_code_base = up_codes.gpu_address();
    let up_scale_base = up_scales.gpu_address();
    let down_code_base = down_codes.gpu_address();
    let down_scale_base = down_scales.gpu_address();

    let mut entries = vec![Qwen80DeviceExpertTriplet::default(); n_experts];
    for expert in 0..n_experts {
        let code_off = (expert * QWEN80_EXPERT_CODE_BYTES) as u64;
        let scale_off = (expert * QWEN80_EXPERT_SCALE_BYTES) as u64;
        entries[expert] = Qwen80DeviceExpertTriplet {
            gate: tensor_ref(
                gate_code_base + code_off,
                gate_scale_base + scale_off,
                QWEN80_MOE_INTERMEDIATE as u32,
                QWEN80_HIDDEN as u32,
                generation,
            ),
            up: tensor_ref(
                up_code_base + code_off,
                up_scale_base + scale_off,
                QWEN80_MOE_INTERMEDIATE as u32,
                QWEN80_HIDDEN as u32,
                generation,
            ),
            down: tensor_ref(
                down_code_base + code_off,
                down_scale_base + scale_off,
                QWEN80_HIDDEN as u32,
                QWEN80_MOE_INTERMEDIATE as u32,
                generation,
            ),
            ready_mask: QWEN80_DEVICE_EXPERT_TRIPLET_READY,
            generation,
        };
    }
    let table = context.new_buffer_with_bytes_checked(bytemuck::cast_slice(&entries))?;
    Ok(Qwen80DeviceExpertTableLease {
        table,
        resources: vec![
            gate_codes,
            gate_scales,
            up_codes,
            up_scales,
            down_codes,
            down_scales,
        ],
        generation,
        n_experts,
        layer,
        table_byte_offset: 0,
        build_secs: started.elapsed().as_secs_f64(),
    })
}

#[cfg(target_os = "macos")]
fn fill_one_expert(
    catalog: &Qwen80UniformQ4StreamingCatalog,
    layer: usize,
    expert: usize,
    ptrs: SlabView,
) -> Result<()> {
    copy_proj(
        catalog,
        &expert_proj_name(layer, expert, "gate_proj"),
        QWEN80_MOE_INTERMEDIATE,
        QWEN80_HIDDEN,
        unsafe { ptrs.gate_codes.add(expert * QWEN80_EXPERT_CODE_BYTES) },
        unsafe { ptrs.gate_scales.add(expert * QWEN80_EXPERT_SCALE_BYTES) },
    )?;
    copy_proj(
        catalog,
        &expert_proj_name(layer, expert, "up_proj"),
        QWEN80_MOE_INTERMEDIATE,
        QWEN80_HIDDEN,
        unsafe { ptrs.up_codes.add(expert * QWEN80_EXPERT_CODE_BYTES) },
        unsafe { ptrs.up_scales.add(expert * QWEN80_EXPERT_SCALE_BYTES) },
    )?;
    copy_proj(
        catalog,
        &expert_proj_name(layer, expert, "down_proj"),
        QWEN80_HIDDEN,
        QWEN80_MOE_INTERMEDIATE,
        unsafe { ptrs.down_codes.add(expert * QWEN80_EXPERT_CODE_BYTES) },
        unsafe { ptrs.down_scales.add(expert * QWEN80_EXPERT_SCALE_BYTES) },
    )?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn copy_proj(
    catalog: &Qwen80UniformQ4StreamingCatalog,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
    codes_dst: *mut u8,
    scales_dst: *mut u8,
) -> Result<()> {
    let packed = catalog.load_packed(name)?;
    let (rows, cols) = packed.rows_cols()?;
    if rows != expected_rows || cols != expected_cols {
        return Err(table_error(format!(
            "{name} geometry {rows}x{cols} != {expected_rows}x{expected_cols}"
        )));
    }
    let codes = packed.codes()?;
    let scales = packed.scales()?;
    if codes.len() != QWEN80_EXPERT_CODE_BYTES || scales.len() != QWEN80_EXPERT_SCALE_BYTES {
        return Err(table_error(format!(
            "{name} payload sizes codes={} scales={} drifted",
            codes.len(),
            scales.len()
        )));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(codes.as_ptr(), codes_dst, QWEN80_EXPERT_CODE_BYTES);
        std::ptr::copy_nonoverlapping(scales.as_ptr(), scales_dst, QWEN80_EXPERT_SCALE_BYTES);
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn upload_proj(
    context: &crate::metal::MetalContext,
    catalog: &Qwen80UniformQ4StreamingCatalog,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<(crate::metal::PinnedBuffer, crate::metal::PinnedBuffer)> {
    let packed = catalog.load_packed(name)?;
    let (rows, cols) = packed.rows_cols()?;
    if rows != expected_rows || cols != expected_cols {
        return Err(table_error(format!(
            "{name} geometry {rows}x{cols} != {expected_rows}x{expected_cols}"
        )));
    }
    let codes = packed.codes()?;
    let scales = packed.scales()?;
    if codes.len() != QWEN80_EXPERT_CODE_BYTES || scales.len() != QWEN80_EXPERT_SCALE_BYTES {
        return Err(table_error(format!(
            "{name} payload sizes codes={} scales={} drifted",
            codes.len(),
            scales.len()
        )));
    }
    Ok((
        context.new_buffer_with_bytes_checked(codes)?,
        context.new_buffer_with_bytes_checked(scales)?,
    ))
}

/// Upload one expert's three Q4 projections. Used by the paged selected-10 cache.
#[cfg(target_os = "macos")]
pub fn upload_qwen80_expert_triplet(
    context: &crate::metal::MetalContext,
    catalog: &Qwen80UniformQ4StreamingCatalog,
    layer: usize,
    expert: usize,
) -> Result<Qwen80ExpertGpuTriplet> {
    let (gate_codes, gate_scales) = upload_proj(
        context,
        catalog,
        &expert_proj_name(layer, expert, "gate_proj"),
        QWEN80_MOE_INTERMEDIATE,
        QWEN80_HIDDEN,
    )?;
    let (up_codes, up_scales) = upload_proj(
        context,
        catalog,
        &expert_proj_name(layer, expert, "up_proj"),
        QWEN80_MOE_INTERMEDIATE,
        QWEN80_HIDDEN,
    )?;
    let (down_codes, down_scales) = upload_proj(
        context,
        catalog,
        &expert_proj_name(layer, expert, "down_proj"),
        QWEN80_HIDDEN,
        QWEN80_MOE_INTERMEDIATE,
    )?;
    Ok(Qwen80ExpertGpuTriplet {
        gate_codes,
        gate_scales,
        up_codes,
        up_scales,
        down_codes,
        down_scales,
    })
}

#[cfg(target_os = "macos")]
fn triplet_from_gpu(trip: &Qwen80ExpertGpuTriplet, generation: u32) -> Qwen80DeviceExpertTriplet {
    Qwen80DeviceExpertTriplet {
        gate: tensor_ref(
            trip.gate_codes.gpu_address(),
            trip.gate_scales.gpu_address(),
            QWEN80_MOE_INTERMEDIATE as u32,
            QWEN80_HIDDEN as u32,
            generation,
        ),
        up: tensor_ref(
            trip.up_codes.gpu_address(),
            trip.up_scales.gpu_address(),
            QWEN80_MOE_INTERMEDIATE as u32,
            QWEN80_HIDDEN as u32,
            generation,
        ),
        down: tensor_ref(
            trip.down_codes.gpu_address(),
            trip.down_scales.gpu_address(),
            QWEN80_HIDDEN as u32,
            QWEN80_MOE_INTERMEDIATE as u32,
            generation,
        ),
        ready_mask: QWEN80_DEVICE_EXPERT_TRIPLET_READY,
        generation,
    }
}

/// Write a 512-entry table whose selected route ids are ready. Other slots
/// stay zero (not ready); the kernel only indexes the live route_ids.
#[cfg(target_os = "macos")]
pub fn write_selected_expert_table(
    context: &crate::metal::MetalContext,
    layer: usize,
    selected: &[(u32, &Qwen80ExpertGpuTriplet)],
    table: Option<crate::metal::PinnedBuffer>,
) -> Result<Qwen80DeviceExpertTableLease> {
    let generation = u32::try_from(layer.saturating_add(1))
        .map_err(|_| table_error("layer generation overflows u32"))?;
    let mut entries = vec![Qwen80DeviceExpertTriplet::default(); QWEN80_EXPERTS];
    let mut resources = Vec::with_capacity(selected.len().saturating_mul(6));
    for &(expert, trip) in selected {
        let index = expert as usize;
        if index >= QWEN80_EXPERTS {
            return Err(table_error(format!(
                "selected expert {expert} is outside 0..512"
            )));
        }
        entries[index] = triplet_from_gpu(trip, generation);
        for resource in trip.resources() {
            resources.push(resource.clone());
        }
    }
    let table = if let Some(existing) = table {
        crate::metal::MetalContext::write_buffer_bytes(&existing, bytemuck::cast_slice(&entries));
        existing
    } else {
        context.new_buffer_with_bytes_checked(bytemuck::cast_slice(&entries))?
    };
    Ok(Qwen80DeviceExpertTableLease {
        table,
        resources,
        generation,
        n_experts: QWEN80_EXPERTS,
        layer,
        table_byte_offset: 0,
        build_secs: 0.0,
    })
}

/// Ten-entry address table. Caller remaps live route ids to `0..10` so the
/// kernel indexes this compact table. No payload memcpy.
#[cfg(target_os = "macos")]
pub fn write_top10_address_table(
    context: &crate::metal::MetalContext,
    layer: usize,
    selected: &[(u32, &Qwen80ExpertGpuTriplet)],
    table: Option<crate::metal::PinnedBuffer>,
) -> Result<Qwen80DeviceExpertTableLease> {
    if selected.len() != QWEN80_EXPERT_TABLE_TOP_K {
        return Err(table_error("top-10 address table requires exactly 10 experts"));
    }
    let generation = u32::try_from(layer.saturating_add(1))
        .map_err(|_| table_error("layer generation overflows u32"))?;
    let mut entries = [Qwen80DeviceExpertTriplet::default(); QWEN80_EXPERT_TABLE_TOP_K];
    let mut resources = Vec::with_capacity(selected.len().saturating_mul(6));
    for (slot, &(_, trip)) in selected.iter().enumerate() {
        entries[slot] = triplet_from_gpu(trip, generation);
        for resource in trip.resources() {
            resources.push(resource.clone());
        }
    }
    let table = if let Some(existing) = table {
        crate::metal::MetalContext::write_buffer_bytes(&existing, bytemuck::cast_slice(&entries));
        existing
    } else {
        context.new_buffer_with_bytes_checked(bytemuck::cast_slice(&entries))?
    };
    Ok(Qwen80DeviceExpertTableLease {
        table,
        resources,
        generation,
        n_experts: QWEN80_EXPERT_TABLE_TOP_K,
        layer,
        table_byte_offset: 0,
        build_secs: 0.0,
    })
}

/// Six compact slabs that hold the live top-10 experts for one layer.
#[cfg(target_os = "macos")]
pub struct Qwen80CompactExpertSlabs {
    pub gate_codes: crate::metal::PinnedBuffer,
    pub gate_scales: crate::metal::PinnedBuffer,
    pub up_codes: crate::metal::PinnedBuffer,
    pub up_scales: crate::metal::PinnedBuffer,
    pub down_codes: crate::metal::PinnedBuffer,
    pub down_scales: crate::metal::PinnedBuffer,
}

#[cfg(target_os = "macos")]
impl Qwen80CompactExpertSlabs {
    pub fn allocate(context: &crate::metal::MetalContext) -> Result<Self> {
        Ok(Self {
            gate_codes: context
                .new_buffer_checked(QWEN80_EXPERT_TABLE_TOP_K * QWEN80_EXPERT_CODE_BYTES)?,
            gate_scales: context
                .new_buffer_checked(QWEN80_EXPERT_TABLE_TOP_K * QWEN80_EXPERT_SCALE_BYTES)?,
            up_codes: context
                .new_buffer_checked(QWEN80_EXPERT_TABLE_TOP_K * QWEN80_EXPERT_CODE_BYTES)?,
            up_scales: context
                .new_buffer_checked(QWEN80_EXPERT_TABLE_TOP_K * QWEN80_EXPERT_SCALE_BYTES)?,
            down_codes: context
                .new_buffer_checked(QWEN80_EXPERT_TABLE_TOP_K * QWEN80_EXPERT_CODE_BYTES)?,
            down_scales: context
                .new_buffer_checked(QWEN80_EXPERT_TABLE_TOP_K * QWEN80_EXPERT_SCALE_BYTES)?,
        })
    }

    pub fn resources(&self) -> [&crate::metal::PinnedBuffer; 6] {
        [
            &self.gate_codes,
            &self.gate_scales,
            &self.up_codes,
            &self.up_scales,
            &self.down_codes,
            &self.down_scales,
        ]
    }
}

#[cfg(target_os = "macos")]
fn copy_buf(
    dst: &crate::metal::PinnedBuffer,
    dst_off: usize,
    src: &crate::metal::PinnedBuffer,
    len: usize,
) {
    unsafe {
        let d = (dst.contents() as *mut u8).add(dst_off);
        let s = src.contents() as *const u8;
        std::ptr::copy_nonoverlapping(s, d, len);
    }
}

/// Pack the selected top-10 into compact slabs and write a 512-entry table
/// so device `route_ids` (real expert ids) still index the 512-way table.
#[cfg(target_os = "macos")]
pub fn write_compact_selected_table(
    context: &crate::metal::MetalContext,
    slabs: &Qwen80CompactExpertSlabs,
    layer: usize,
    selected: &[(u32, &Qwen80ExpertGpuTriplet)],
    table: Option<crate::metal::PinnedBuffer>,
) -> Result<Qwen80DeviceExpertTableLease> {
    if selected.len() != QWEN80_EXPERT_TABLE_TOP_K {
        return Err(table_error("compact table requires exactly top-10 experts"));
    }
    let generation = u32::try_from(layer.saturating_add(1))
        .map_err(|_| table_error("layer generation overflows u32"))?;
    for (slot, &(_, trip)) in selected.iter().enumerate() {
        copy_buf(
            &slabs.gate_codes,
            slot * QWEN80_EXPERT_CODE_BYTES,
            &trip.gate_codes,
            QWEN80_EXPERT_CODE_BYTES,
        );
        copy_buf(
            &slabs.gate_scales,
            slot * QWEN80_EXPERT_SCALE_BYTES,
            &trip.gate_scales,
            QWEN80_EXPERT_SCALE_BYTES,
        );
        copy_buf(
            &slabs.up_codes,
            slot * QWEN80_EXPERT_CODE_BYTES,
            &trip.up_codes,
            QWEN80_EXPERT_CODE_BYTES,
        );
        copy_buf(
            &slabs.up_scales,
            slot * QWEN80_EXPERT_SCALE_BYTES,
            &trip.up_scales,
            QWEN80_EXPERT_SCALE_BYTES,
        );
        copy_buf(
            &slabs.down_codes,
            slot * QWEN80_EXPERT_CODE_BYTES,
            &trip.down_codes,
            QWEN80_EXPERT_CODE_BYTES,
        );
        copy_buf(
            &slabs.down_scales,
            slot * QWEN80_EXPERT_SCALE_BYTES,
            &trip.down_scales,
            QWEN80_EXPERT_SCALE_BYTES,
        );
    }
    let mut entries = vec![Qwen80DeviceExpertTriplet::default(); QWEN80_EXPERTS];
    for (slot, &(expert, _)) in selected.iter().enumerate() {
        let index = expert as usize;
        if index >= QWEN80_EXPERTS {
            return Err(table_error(format!(
                "selected expert {expert} is outside 0..512"
            )));
        }
        let code_off = (slot * QWEN80_EXPERT_CODE_BYTES) as u64;
        let scale_off = (slot * QWEN80_EXPERT_SCALE_BYTES) as u64;
        entries[index] = Qwen80DeviceExpertTriplet {
            gate: tensor_ref(
                slabs.gate_codes.gpu_address() + code_off,
                slabs.gate_scales.gpu_address() + scale_off,
                QWEN80_MOE_INTERMEDIATE as u32,
                QWEN80_HIDDEN as u32,
                generation,
            ),
            up: tensor_ref(
                slabs.up_codes.gpu_address() + code_off,
                slabs.up_scales.gpu_address() + scale_off,
                QWEN80_MOE_INTERMEDIATE as u32,
                QWEN80_HIDDEN as u32,
                generation,
            ),
            down: tensor_ref(
                slabs.down_codes.gpu_address() + code_off,
                slabs.down_scales.gpu_address() + scale_off,
                QWEN80_HIDDEN as u32,
                QWEN80_MOE_INTERMEDIATE as u32,
                generation,
            ),
            ready_mask: QWEN80_DEVICE_EXPERT_TRIPLET_READY,
            generation,
        };
    }
    let table = if let Some(existing) = table {
        crate::metal::MetalContext::write_buffer_bytes(&existing, bytemuck::cast_slice(&entries));
        existing
    } else {
        context.new_buffer_with_bytes_checked(bytemuck::cast_slice(&entries))?
    };
    Ok(Qwen80DeviceExpertTableLease {
        table,
        resources: slabs.resources().into_iter().cloned().collect(),
        generation,
        n_experts: QWEN80_EXPERTS,
        layer,
        table_byte_offset: 0,
        build_secs: 0.0,
    })
}

#[cfg(target_os = "macos")]
fn matvec_dispatch_shape(
    kernel: Qwen80ExpertTableKernel,
    top_k: usize,
    max_rows: usize,
) -> Result<(&'static str, (u32, u32, u32), (u32, u32, u32))> {
    match kernel {
        Qwen80ExpertTableKernel::Serial => {
            let total = top_k
                .checked_mul(max_rows)
                .ok_or_else(|| table_error("serial grid overflows"))?;
            let total_u32 =
                u32::try_from(total).map_err(|_| table_error("serial grid overflows u32"))?;
            Ok((
                "qwen80_expert_table_uniform_q4_matvec_serial",
                (total_u32, 1, 1),
                (total_u32.min(256).max(1), 1, 1),
            ))
        }
        Qwen80ExpertTableKernel::Rowblock => {
            let groups = max_rows.div_ceil(256 * 4);
            let grid_x = top_k
                .checked_mul(groups)
                .and_then(|v| v.checked_mul(256))
                .ok_or_else(|| table_error("rowblock grid overflows"))?;
            let grid_x_u32 =
                u32::try_from(grid_x).map_err(|_| table_error("rowblock grid overflows u32"))?;
            Ok((
                "qwen80_expert_table_uniform_q4_matvec_rowblock",
                (grid_x_u32, 1, 1),
                (256, 1, 1),
            ))
        }
        Qwen80ExpertTableKernel::Simdgroup => {
            let groups = max_rows.div_ceil(8);
            let grid_x = top_k
                .checked_mul(groups)
                .and_then(|v| v.checked_mul(256))
                .ok_or_else(|| table_error("simdgroup grid overflows"))?;
            let grid_x_u32 =
                u32::try_from(grid_x).map_err(|_| table_error("simdgroup grid overflows u32"))?;
            Ok((
                "qwen80_expert_table_uniform_q4_matvec_simdgroup",
                (grid_x_u32, 1, 1),
                (256, 1, 1),
            ))
        }
    }
}

/// Additive hybrid-graph dispatch site: encode the 512-way top-10 Q4 wave
/// onto an already-open token command buffer.
#[cfg(target_os = "macos")]
pub fn dispatch_qwen80_device_expert_table_tcb(
    tcb: &mut crate::metal::TokenCommandBuffer<'_>,
    lease: &Qwen80DeviceExpertTableLease,
    workspace: &Qwen80DeviceExpertWaveWorkspace,
    kernel: Qwen80ExpertTableKernel,
) -> Result<u64> {
    dispatch_qwen80_device_expert_table_wave(tcb, lease, workspace, kernel)
}

#[cfg(target_os = "macos")]
pub fn dispatch_qwen80_device_expert_table_wave(
    tcb: &mut crate::metal::TokenCommandBuffer<'_>,
    lease: &Qwen80DeviceExpertTableLease,
    workspace: &Qwen80DeviceExpertWaveWorkspace,
    kernel: Qwen80ExpertTableKernel,
) -> Result<u64> {
    if (lease.n_experts != QWEN80_EXPERTS && lease.n_experts != QWEN80_EXPERT_TABLE_TOP_K)
        || lease.resources.len() < 6
        || lease.resources.len() % 6 != 0
    {
        return Err(table_error("lease is incomplete"));
    }
    let mut dispatches = 0u64;
    encode_matvec(
        tcb,
        lease,
        workspace,
        QWEN80_DEVICE_PROJ_GATE,
        &workspace.input,
        &workspace.gate,
        0,
        0,
        0,
        QWEN80_MOE_INTERMEDIATE,
        QWEN80_MOE_INTERMEDIATE,
        kernel,
        true,
    )?;
    dispatches += 1;
    encode_matvec(
        tcb,
        lease,
        workspace,
        QWEN80_DEVICE_PROJ_UP,
        &workspace.input,
        &workspace.up,
        0,
        0,
        0,
        QWEN80_MOE_INTERMEDIATE,
        QWEN80_MOE_INTERMEDIATE,
        kernel,
        true,
    )?;
    dispatches += 1;

    let silu_elems = (QWEN80_EXPERT_TABLE_TOP_K * QWEN80_MOE_INTERMEDIATE) as u32;
    let gate = workspace.gate.clone();
    let up = workspace.up.clone();
    let activated = workspace.activated.clone();
    tcb.dispatch_threads(
        "qwen80_expert_table_silu_mul",
        (silu_elems, 1, 1),
        (silu_elems.min(256).max(1), 1, 1),
        move |encoder| {
            encoder.set_buffer(0, Some(&gate), 0);
            encoder.set_buffer(1, Some(&up), 0);
            encoder.set_buffer(2, Some(&activated), 0);
            encoder.set_bytes(3, 4, &silu_elems as *const u32 as *const _);
        },
    )?;
    dispatches += 1;

    encode_matvec(
        tcb,
        lease,
        workspace,
        QWEN80_DEVICE_PROJ_DOWN,
        &workspace.activated,
        &workspace.down,
        0,
        QWEN80_MOE_INTERMEDIATE,
        0,
        QWEN80_HIDDEN,
        QWEN80_HIDDEN,
        kernel,
        true,
    )?;
    dispatches += 1;

    let hidden = QWEN80_HIDDEN as u32;
    let routes = QWEN80_EXPERT_TABLE_TOP_K as u32;
    let down = workspace.down.clone();
    let weights = workspace.route_weights.clone();
    let combined = workspace.combined.clone();
    tcb.dispatch_threads(
        "qwen80_expert_table_weighted_sum",
        (hidden, 1, 1),
        (hidden.min(256).max(1), 1, 1),
        move |encoder| {
            encoder.set_buffer(0, Some(&down), 0);
            encoder.set_buffer(1, Some(&weights), 0);
            encoder.set_buffer(2, Some(&combined), 0);
            encoder.set_bytes(3, 4, &hidden as *const u32 as *const _);
            encoder.set_bytes(4, 4, &routes as *const u32 as *const _);
        },
    )?;
    Ok(dispatches + 1)
}

#[cfg(target_os = "macos")]
fn encode_matvec(
    tcb: &mut crate::metal::TokenCommandBuffer<'_>,
    lease: &Qwen80DeviceExpertTableLease,
    workspace: &Qwen80DeviceExpertWaveWorkspace,
    projection: u32,
    input: &crate::metal::PinnedBuffer,
    output: &crate::metal::PinnedBuffer,
    input_base_elems: usize,
    input_stride_elems: usize,
    output_base_elems: usize,
    output_stride_elems: usize,
    max_rows: usize,
    kernel: Qwen80ExpertTableKernel,
    declare_resources: bool,
) -> Result<()> {
    let params = Qwen80DeviceExpertMatvecParams {
        n_experts: lease.n_experts as u32,
        experts_per_token: QWEN80_EXPERT_TABLE_TOP_K as u32,
        generation: lease.generation,
        projection,
        group_size: UNIFORM_Q4_GROUP_SIZE as u32,
        max_rows: max_rows as u32,
        input_base_elems: input_base_elems as u32,
        input_stride_elems: input_stride_elems as u32,
        output_base_elems: output_base_elems as u32,
        output_stride_elems: output_stride_elems as u32,
        pad0: 0,
        pad1: 0,
    };
    let (name, grid, tg) = matvec_dispatch_shape(kernel, QWEN80_EXPERT_TABLE_TOP_K, max_rows)?;
    let route_ids = workspace.route_ids.clone();
    let table = lease.table.clone();
    let table_byte_offset = lease.table_byte_offset;
    let input = input.clone();
    let output = output.clone();
    let resources = if declare_resources {
        lease.resources.clone()
    } else {
        Vec::new()
    };
    tcb.dispatch_threads(name, grid, tg, move |encoder| {
        encoder.set_buffer(0, Some(&route_ids), 0);
        encoder.set_buffer(1, Some(&table), table_byte_offset);
        encoder.set_buffer(2, Some(&input), 0);
        encoder.set_buffer(3, Some(&output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&params) as u64,
            &params as *const _ as *const _,
        );
        if !resources.is_empty() {
            let mut refs: Vec<&metal::ResourceRef> = Vec::with_capacity(resources.len());
            for resource in &resources {
                refs.push(resource);
            }
            encoder.use_resources(&refs, metal::MTLResourceUsage::Read);
        }
    })
}

#[cfg(target_os = "macos")]
pub fn run_qwen80_routed_expert_table(
    context: &crate::metal::MetalContext,
    lease: &Qwen80DeviceExpertTableLease,
    workspace: &Qwen80DeviceExpertWaveWorkspace,
    route_ids: &[u32],
    route_weights: &[f32],
    input: &[f32],
    output: &mut [f32],
    kernel: Qwen80ExpertTableKernel,
) -> Result<u64> {
    if route_ids.len() != QWEN80_EXPERT_TABLE_TOP_K
        || route_weights.len() != QWEN80_EXPERT_TABLE_TOP_K
    {
        return Err(table_error("route vectors must be top-10"));
    }
    if input.len() != QWEN80_HIDDEN || output.len() != QWEN80_HIDDEN {
        return Err(table_error("hidden geometry drifted"));
    }
    if route_ids.iter().any(|&id| id as usize >= lease.n_experts) {
        return Err(table_error("route id is outside the 512-way table"));
    }
    crate::metal::MetalContext::write_buffer_bytes(
        &workspace.route_ids,
        bytemuck::cast_slice(route_ids),
    );
    crate::metal::MetalContext::write_buffer_bytes(
        &workspace.route_weights,
        bytemuck::cast_slice(route_weights),
    );
    crate::metal::MetalContext::write_buffer_bytes(&workspace.input, bytemuck::cast_slice(input));
    let mut tcb = crate::metal::TokenCommandBuffer::new(context);
    let dispatches = dispatch_qwen80_device_expert_table_wave(&mut tcb, lease, workspace, kernel)?;
    tcb.commit_and_wait()?;
    let observed = unsafe {
        std::slice::from_raw_parts(workspace.combined.contents() as *const f32, QWEN80_HIDDEN)
    };
    output.copy_from_slice(observed);
    if output.iter().any(|value| !value.is_finite()) {
        return Err(table_error(
            "routed expert table produced a non-finite hidden",
        ));
    }
    Ok(dispatches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_expert_table_abi_matches_metal_static_asserts() {
        assert_eq!(std::mem::size_of::<Qwen80DeviceExpertTensorRef>(), 40);
        assert_eq!(std::mem::size_of::<Qwen80DeviceExpertTriplet>(), 128);
        assert_eq!(std::mem::size_of::<Qwen80DeviceExpertMatvecParams>(), 48);
        assert_eq!(QWEN80_EXPERT_PROJ_ELEMENTS, 512 * 2048);
        assert_eq!(QWEN80_EXPERT_CODE_BYTES, 524_288);
        assert_eq!(QWEN80_EXPERT_SCALE_BYTES, 32_768);
        assert_eq!(QWEN80_EXPERT_TRIPLET_PAYLOAD_BYTES, 1_671_168);
        assert_eq!(qwen80_q4_address_geometry().table_bytes(), 3_145_728);
        assert_eq!(QWEN80_EXPERT_TABLE_KERNELS.len(), 5);
    }

    #[test]
    fn device_expert_table_flag_defaults_on_and_opt_out() {
        let previous = std::env::var("HAWKING_QWEN80_DEVICE_EXPERT_TABLE").ok();
        std::env::remove_var("HAWKING_QWEN80_DEVICE_EXPERT_TABLE");
        std::env::remove_var("HAWKING_QWEN80_DEVICE_ROUTE");
        assert!(qwen80_device_expert_table_enabled());
        std::env::set_var("HAWKING_QWEN80_DEVICE_EXPERT_TABLE", "0");
        assert!(!qwen80_device_expert_table_enabled());
        std::env::set_var("HAWKING_QWEN80_DEVICE_EXPERT_TABLE", "false");
        assert!(!qwen80_device_expert_table_enabled());
        std::env::set_var("HAWKING_QWEN80_DEVICE_EXPERT_TABLE", "1");
        assert!(qwen80_device_expert_table_enabled());
        match previous {
            Some(value) => std::env::set_var("HAWKING_QWEN80_DEVICE_EXPERT_TABLE", value),
            None => std::env::remove_var("HAWKING_QWEN80_DEVICE_EXPERT_TABLE"),
        }
        std::env::remove_var("HAWKING_QWEN80_DEVICE_ROUTE");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn device_expert_table_layer0_matches_host_for_fixed_routes() {
        use crate::kernels::silu_mul;
        use crate::model::qwen80_uniform_q4_hybrid_decode::discover_qwen80_uniform_q4_root;
        let Some(root) = discover_qwen80_uniform_q4_root() else {
            return;
        };
        let Ok(context) = crate::metal::MetalContext::new() else {
            return;
        };
        let catalog = Qwen80UniformQ4StreamingCatalog::open(&root).unwrap();
        let lease = build_qwen80_device_expert_table(&context, &catalog, 0).unwrap();
        let wave = Qwen80DeviceExpertWaveWorkspace::allocate(&context).unwrap();
        let input: Vec<f32> = (0..QWEN80_HIDDEN)
            .map(|i| ((i as f32) * 0.017).sin() * 0.05)
            .collect();
        let route_ids = [0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let route_weights = [0.1f32; 10];
        let mut device_out = vec![0.0f32; QWEN80_HIDDEN];
        run_qwen80_routed_expert_table(
            &context,
            &lease,
            &wave,
            &route_ids,
            &route_weights,
            &input,
            &mut device_out,
            Qwen80ExpertTableKernel::Rowblock,
        )
        .unwrap();

        let mut host_out = vec![0.0f32; QWEN80_HIDDEN];
        for (&expert, &weight) in route_ids.iter().zip(route_weights.iter()) {
            let gate_p = catalog
                .load_packed(&expert_proj_name(0, expert as usize, "gate_proj"))
                .unwrap();
            let up_p = catalog
                .load_packed(&expert_proj_name(0, expert as usize, "up_proj"))
                .unwrap();
            let down_p = catalog
                .load_packed(&expert_proj_name(0, expert as usize, "down_proj"))
                .unwrap();
            let mut gate = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
            let mut up = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
            let mut act = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
            let mut down = vec![0.0f32; QWEN80_HIDDEN];
            gate_p.matvec(&input, &mut gate).unwrap();
            up_p.matvec(&input, &mut up).unwrap();
            silu_mul(&gate, &up, &mut act);
            down_p.matvec(&act, &mut down).unwrap();
            for (dst, value) in host_out.iter_mut().zip(down) {
                *dst += value * weight;
            }
        }
        let mut max_abs = 0.0f32;
        for (d, h) in device_out.iter().zip(host_out.iter()) {
            max_abs = max_abs.max((d - h).abs());
        }
        assert!(
            max_abs < 1e-4,
            "layer-0 table vs host max_abs={max_abs} device[0]={} host[0]={}",
            device_out[0],
            host_out[0]
        );
    }
}
