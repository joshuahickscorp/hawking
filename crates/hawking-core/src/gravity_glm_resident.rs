//! GPU-resident decode state for GLM-5.2.
//!
//! Prerequisite lane for Temporal Gravity command-buffer collapse. The host
//! path keeps the residual stream and per-layer KV / DSA caches as host
//! `Vec<f32>`, so every projection must finish and return a host vector before
//! the next host loop can run (~1,171 `commit_and_wait`s per flagship token).
//!
//! This module keeps those tensors in device (Metal shared) buffers across a
//! token: activations, KV, index keys, router logits / top-k / expert offsets.
//! By default, discrete decisions (stable top-k, noaux_tc groups, sparse
//! softmax) still use the same host arithmetic as
//! [`crate::gravity_glm::forward_impl`] so token identity is bit-exact against
//! the host-state path; they read device-mapped memory in place rather than
//! owning a separate host cache. Projection outputs are written straight into
//! those buffers and are not copied into host `Vec`s as the cache of record.
//!
//! `lm_head` is once per token. Default: host dense or PQ via
//! [`GpuWeightCache::matvec`]. With [`crate::gravity_glm::GPU_LM_HEAD_ENV`]=1,
//! final RMSNorm + projection + greedy argmax + top-k diagnostics share one
//! device command buffer, so the residual stream stays on device at the head
//! boundary. A native.bf16 head also stays device-resident with no per-token
//! widening of its 1.90 GB table; an opt-in PQ head lets bounded direct-u8
//! fixtures exercise the same final graph. Default readback is **token + top-k
//! only**; full logits require `HAWKING_GLM_GPU_LM_HEAD_FULL_LOGITS=1`. The same
//! flag keeps other rank-2 `native.bf16` matvecs (indexer, router) as device
//! bf16.
//!
//! **Expert-wave** (`HAWKING_GLM_GPU_EXPERT_WAVE=1`, default off): opt-in collapse
//! of each MLP layer to one command buffer (`gate + up → SiLU → down` and MoE
//! weighted combine). The default three-`matvec_batch` path is unchanged when
//! the flag is unset (Parity V2.1 item 6). The additional default-off
//! `HAWKING_GLM_GPU_EXPERT_WAVE_CONCURRENT=1` groups independent gate/up and
//! down projections in concurrent Metal encoders while preserving dependency
//! boundaries and ordered weighted combine. The pure device wave appends its
//! residual add before the same commit and returns no activation to the host.
//!
//! **Compact MLA** (`HAWKING_GLM_GPU_COMPACT_MLA=1`, default off): replaces
//! persistent expanded per-head K/V with normalized MLA latent + shared RoPE
//! tail and executes append → absorbed K → ranked attention → absorbed V →
//! o_proj in one five-dispatch command buffer after host DSA ranking.
//!
//! **Device DSA** (`HAWKING_GLM_GPU_DEVICE_DSA=1`, default off, requires compact
//! MLA): when every dependent projection is device-encodable, folds input/q/kv
//! RMSNorm, q_a/kv_a/q_b, compact query/key RoPE, the full indexer, exact radix
//! rank, and compact attention into one command buffer. Host-native projections
//! retain the qualified host-prelude fallback. The final rank is read only for
//! diagnostics after attention, never as an attention dependency.
//!
//! Gated by [`GPU_RESIDENT_STATE_ENV`] (`HAWKING_GLM_GPU_RESIDENT_STATE`), default
//! off, so the host-state path remains the parity oracle.

#![cfg(target_os = "macos")]

use crate::gravity::matvec_dense;
use crate::gravity_glm::gpu::{
    encode_argmax_f32, encode_gemv_native_bf16_seq, encode_sample_topk_f32,
    record_routed_tensor_representation, GpuTensor, GpuWeightCache,
};
use crate::gravity_glm::{
    gpu_compact_mla_enabled, gpu_device_router_enabled, gpu_expert_wave_concurrent_enabled,
    gpu_expert_wave_enabled, gpu_lm_head_enabled, gpu_lm_head_full_logits_enabled, rope_cos_sin,
    rope_interleaved, topk_desc, BoundedLru, GlmArch, GlmTrace, WeightAccess,
    GPU_LM_HEAD_DIAG_TOPK, RESIDENT_RUNTIME_INITIAL_KV_CAPACITY_TOKENS,
};
use crate::metal::{MetalContext, TokenCommandBuffer};
use crate::{Error, Result};
use metal::{Buffer, MTLResourceUsage};
use std::cell::Cell;
use std::sync::Mutex;

// Flag + static wait estimators live on `gravity_glm` so non-Metal unit tests
// can see them: `GPU_RESIDENT_STATE_ENV`, `gpu_resident_state_enabled`,
// `estimate_host_state_waits_per_token`, `estimate_resident_waits_per_token`.

fn write_f32(buf: &Buffer, src: &[f32]) {
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), buf.contents() as *mut f32, src.len());
    }
}

fn zero_f32(buf: &Buffer, n: usize) -> Result<()> {
    let bytes = n
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Gravity(format!("zero_f32 byte size overflow: {n} elements")))?;
    if bytes as u64 > buf.length() {
        return Err(Error::Gravity(format!(
            "zero_f32 byte range 0..{bytes} exceeds buffer length {}",
            buf.length()
        )));
    }
    unsafe {
        std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes);
    }
    Ok(())
}

fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n).to_vec() }
}

fn read_u32(buf: &Buffer, n: usize) -> Vec<u32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const u32, n).to_vec() }
}

#[allow(dead_code)]
const DEVICE_EXPERT_TENSOR_KIND_PQ: u32 = 1;
#[allow(dead_code)]
const DEVICE_EXPERT_TENSOR_KIND_NATIVE_BF16: u32 = 2;
#[allow(dead_code)]
const DEVICE_EXPERT_TRIPLET_READY: u32 = 0b111;
#[allow(dead_code)]
const DEVICE_EXPERT_TABLE_MAX_EXPERTS: usize = 256;

/// Tagged device pointer plus projection geometry. The byte layout is frozen
/// against `GravityDeviceExpertTensorRef` in `gravity_pq.metal`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
struct DeviceExpertTensorRef {
    primary_address: u64,
    secondary_address: u64,
    dim: u32,
    subspaces: u32,
    sub: u32,
    card: u32,
    rows: u32,
    cols: u32,
    nchunk: u32,
    bits: u32,
    kind: u32,
    generation: u32,
}

/// One routed expert's gate/up/down descriptor. An entry is ready only when
/// all three projections were cloned into the immutable snapshot lease.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
struct DeviceExpertTriplet {
    gate: DeviceExpertTensorRef,
    up: DeviceExpertTensorRef,
    down: DeviceExpertTensorRef,
    ready_mask: u32,
    generation: u32,
}

const _: [(); 56] = [(); std::mem::size_of::<DeviceExpertTensorRef>()];
const _: [(); 8] = [(); std::mem::align_of::<DeviceExpertTensorRef>()];
const _: [(); 176] = [(); std::mem::size_of::<DeviceExpertTriplet>()];
const _: [(); 8] = [(); std::mem::align_of::<DeviceExpertTriplet>()];

#[allow(dead_code)]
struct DeviceExpertTableLease {
    table: Buffer,
    /// Cloned Metal handles keep every indirectly referenced buffer alive even
    /// if its logical LRU entry is evicted after this snapshot is built.
    resources: Vec<Buffer>,
    generation: u32,
    n_experts: usize,
    ready_entries: usize,
}

#[allow(dead_code)]
fn device_expert_tensor_ref(
    tensor: &GpuTensor,
    generation: u32,
) -> Option<(DeviceExpertTensorRef, Vec<Buffer>)> {
    match tensor {
        GpuTensor::Pq {
            codebooks,
            codes,
            params,
        } => Some((
            DeviceExpertTensorRef {
                primary_address: codebooks.gpu_address(),
                secondary_address: codes.gpu_address(),
                dim: params.dim,
                subspaces: params.subspaces,
                sub: params.sub,
                card: params.card,
                rows: params.rows,
                cols: params.cols,
                nchunk: params.nchunk,
                bits: params.bits,
                kind: DEVICE_EXPERT_TENSOR_KIND_PQ,
                generation,
            },
            vec![codebooks.clone(), codes.clone()],
        )),
        GpuTensor::NativeGpuBf16 { buf, rows, cols } => Some((
            DeviceExpertTensorRef {
                primary_address: buf.gpu_address(),
                rows: *rows,
                cols: *cols,
                kind: DEVICE_EXPERT_TENSOR_KIND_NATIVE_BF16,
                generation,
                ..DeviceExpertTensorRef::default()
            },
            vec![buf.clone()],
        )),
        GpuTensor::NativeCpu(_) => None,
    }
}

/// Snapshot the currently resident routed triplets for one layer.
///
/// The caller owns the cache guard while this walks the name-keyed LRU. Each
/// ready entry clones its backing Metal buffers into the returned lease, then
/// the descriptor bytes are uploaded once and never patched in place.
#[allow(dead_code)]
fn build_device_expert_table_snapshot(
    ctx: &MetalContext,
    cache: &BoundedLru<GpuTensor>,
    mlp_prefix: &str,
    n_experts: usize,
    generation: u32,
) -> Result<DeviceExpertTableLease> {
    if n_experts == 0 || n_experts > DEVICE_EXPERT_TABLE_MAX_EXPERTS {
        return Err(Error::Gravity(format!(
            "device expert table supports 1..={DEVICE_EXPERT_TABLE_MAX_EXPERTS} experts, got \
             {n_experts}"
        )));
    }
    if generation == 0 {
        return Err(Error::Gravity(
            "device expert table generation 0 is reserved for missing entries".into(),
        ));
    }

    let mut entries = vec![DeviceExpertTriplet::default(); n_experts];
    let mut resources = Vec::new();
    let mut ready_entries = 0usize;
    for (expert, entry) in entries.iter_mut().enumerate() {
        let expert_prefix = format!("{mlp_prefix}.experts.{expert}");
        let gate_name = format!("{expert_prefix}.gate_proj.weight");
        let up_name = format!("{expert_prefix}.up_proj.weight");
        let down_name = format!("{expert_prefix}.down_proj.weight");
        let Some((gate, gate_resources)) = cache
            .get(&gate_name)
            .and_then(|tensor| device_expert_tensor_ref(tensor, generation))
        else {
            continue;
        };
        let Some((up, up_resources)) = cache
            .get(&up_name)
            .and_then(|tensor| device_expert_tensor_ref(tensor, generation))
        else {
            continue;
        };
        let Some((down, down_resources)) = cache
            .get(&down_name)
            .and_then(|tensor| device_expert_tensor_ref(tensor, generation))
        else {
            continue;
        };
        *entry = DeviceExpertTriplet {
            gate,
            up,
            down,
            ready_mask: DEVICE_EXPERT_TRIPLET_READY,
            generation,
        };
        resources.extend(gate_resources);
        resources.extend(up_resources);
        resources.extend(down_resources);
        ready_entries += 1;
    }
    let table = ctx.new_buffer_with_bytes_checked(bytemuck::cast_slice(&entries))?;
    Ok(DeviceExpertTableLease {
        table,
        resources,
        generation,
        n_experts,
        ready_entries,
    })
}

#[allow(dead_code)]
fn build_single_device_expert_snapshot(
    ctx: &MetalContext,
    gate_tensor: &GpuTensor,
    up_tensor: &GpuTensor,
    down_tensor: &GpuTensor,
    generation: u32,
) -> Result<DeviceExpertTableLease> {
    if generation == 0 {
        return Err(Error::Gravity(
            "device expert table generation 0 is reserved for missing entries".into(),
        ));
    }
    let (gate, gate_resources) = device_expert_tensor_ref(gate_tensor, generation)
        .ok_or_else(|| Error::Gravity("single expert gate is not device-resident".into()))?;
    let (up, up_resources) = device_expert_tensor_ref(up_tensor, generation)
        .ok_or_else(|| Error::Gravity("single expert up is not device-resident".into()))?;
    let (down, down_resources) = device_expert_tensor_ref(down_tensor, generation)
        .ok_or_else(|| Error::Gravity("single expert down is not device-resident".into()))?;
    let entry = DeviceExpertTriplet {
        gate,
        up,
        down,
        ready_mask: DEVICE_EXPERT_TRIPLET_READY,
        generation,
    };
    let mut resources =
        Vec::with_capacity(gate_resources.len() + up_resources.len() + down_resources.len());
    resources.extend(gate_resources);
    resources.extend(up_resources);
    resources.extend(down_resources);
    Ok(DeviceExpertTableLease {
        table: ctx.new_buffer_with_bytes_checked(bytemuck::bytes_of(&entry))?,
        resources,
        generation,
        n_experts: 1,
        ready_entries: 1,
    })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceExpertTableValidateParams {
    n_experts: u32,
    experts_per_token: u32,
    generation: u32,
    required_kind: u32,
    hidden: u32,
    intermediate: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceExpertTableMatvecParams {
    n_experts: u32,
    experts_per_token: u32,
    generation: u32,
    execution_position: u32,
    projection: u32,
    rows: u32,
    cols: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceExpertTableAxpyParams {
    n: u32,
    experts_per_token: u32,
    execution_position: u32,
    use_router_weight: u32,
}

const _: [(); 24] = [(); std::mem::size_of::<DeviceExpertTableValidateParams>()];
const _: [(); 28] = [(); std::mem::size_of::<DeviceExpertTableMatvecParams>()];
const _: [(); 16] = [(); std::mem::size_of::<DeviceExpertTableAxpyParams>()];

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn encode_device_expert_table_validate(
    tcb: &mut TokenCommandBuffer<'_>,
    lease: &DeviceExpertTableLease,
    expert_indices: &Buffer,
    expert_exec_slots: &Buffer,
    miss_mask: &Buffer,
    experts_per_token: usize,
    hidden: usize,
    intermediate: usize,
    required_kind: u32,
) -> Result<()> {
    if experts_per_token == 0 || experts_per_token > 32 {
        return Err(Error::Gravity(format!(
            "device expert table validation requires 1..=32 selected experts, got \
             {experts_per_token}"
        )));
    }
    let selected_bytes = experts_per_token
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| Error::Gravity("device expert selection byte overflow".into()))?
        as u64;
    if expert_indices.length() < selected_bytes
        || expert_exec_slots.length() < selected_bytes
        || miss_mask.length() < std::mem::size_of::<u32>() as u64
    {
        return Err(Error::Gravity(
            "device expert table validation received an undersized selection or miss buffer".into(),
        ));
    }
    let expect_table = lease
        .n_experts
        .checked_mul(std::mem::size_of::<DeviceExpertTriplet>())
        .ok_or_else(|| Error::Gravity("device expert table byte overflow".into()))?
        as u64;
    if lease.table.length() != expect_table {
        return Err(Error::Gravity(format!(
            "device expert table has {} B, expected exactly {expect_table} B",
            lease.table.length()
        )));
    }
    let params = DeviceExpertTableValidateParams {
        n_experts: lease.n_experts as u32,
        experts_per_token: experts_per_token as u32,
        generation: lease.generation,
        required_kind,
        hidden: hidden as u32,
        intermediate: intermediate as u32,
    };
    let indices = expert_indices.clone();
    let slots = expert_exec_slots.clone();
    let table = lease.table.clone();
    let miss = miss_mask.clone();
    tcb.dispatch_threads(
        "gravity_glm_expert_table_validate",
        (1, 1, 1),
        (1, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&indices), 0);
            enc.set_buffer(1, Some(&slots), 0);
            enc.set_buffer(2, Some(&table), 0);
            enc.set_buffer(3, Some(&miss), 0);
            enc.set_bytes(
                4,
                std::mem::size_of_val(&params) as u64,
                &params as *const _ as *const _,
            );
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn encode_device_expert_table_pq_matvec(
    tcb: &mut TokenCommandBuffer<'_>,
    lease: &DeviceExpertTableLease,
    expert_indices: &Buffer,
    expert_exec_slots: &Buffer,
    miss_mask: &Buffer,
    experts_per_token: usize,
    execution_position: usize,
    projection: u32,
    x: &Buffer,
    rows: usize,
    cols: usize,
    y: &Buffer,
) -> Result<()> {
    if execution_position >= experts_per_token || projection > 2 {
        return Err(Error::Gravity(format!(
            "invalid device expert table matvec position/projection: position \
             {execution_position}/{experts_per_token}, projection {projection}"
        )));
    }
    let x_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Gravity("device expert matvec input byte overflow".into()))?
        as u64;
    let y_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Gravity("device expert matvec output byte overflow".into()))?
        as u64;
    if x.length() < x_bytes || y.length() < y_bytes {
        return Err(Error::Gravity(format!(
            "device expert table matvec buffer too small: x={}/{} B y={}/{} B",
            x.length(),
            x_bytes,
            y.length(),
            y_bytes
        )));
    }
    let params = DeviceExpertTableMatvecParams {
        n_experts: lease.n_experts as u32,
        experts_per_token: experts_per_token as u32,
        generation: lease.generation,
        execution_position: execution_position as u32,
        projection,
        rows: rows as u32,
        cols: cols as u32,
    };
    let indices = expert_indices.clone();
    let slots = expert_exec_slots.clone();
    let table = lease.table.clone();
    let miss = miss_mask.clone();
    let xb = x.clone();
    let yb = y.clone();
    let resources = lease.resources.clone();
    const TG: u32 = 256;
    let n_tg = (rows as u32).div_ceil(8);
    tcb.dispatch_threads(
        "gravity_glm_expert_table_pq_matvec",
        (n_tg * TG, 1, 1),
        (TG, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&indices), 0);
            enc.set_buffer(1, Some(&slots), 0);
            enc.set_buffer(2, Some(&table), 0);
            enc.set_buffer(3, Some(&miss), 0);
            enc.set_buffer(4, Some(&xb), 0);
            enc.set_buffer(5, Some(&yb), 0);
            enc.set_bytes(
                6,
                std::mem::size_of_val(&params) as u64,
                &params as *const _ as *const _,
            );
            let mut refs: Vec<&metal::ResourceRef> = Vec::with_capacity(resources.len());
            for resource in &resources {
                refs.push(resource);
            }
            enc.use_resources(&refs, MTLResourceUsage::Read);
        },
    )
}

fn require_f32_elements(buffer: &Buffer, elements: usize, label: &str) -> Result<()> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Gravity(format!("{label} byte overflow")))? as u64;
    if buffer.length() < bytes {
        return Err(Error::Gravity(format!(
            "{label} has {} B, needs at least {bytes} B",
            buffer.length()
        )));
    }
    Ok(())
}

#[allow(dead_code)]
fn encode_device_expert_table_zero(
    tcb: &mut TokenCommandBuffer<'_>,
    output: &Buffer,
    miss_mask: &Buffer,
    n: usize,
) -> Result<()> {
    require_f32_elements(output, n, "device expert guarded zero output")?;
    if miss_mask.length() < std::mem::size_of::<u32>() as u64 {
        return Err(Error::Gravity(
            "device expert guarded zero miss buffer is undersized".into(),
        ));
    }
    let output = output.clone();
    let miss = miss_mask.clone();
    let n = n as u32;
    const TG: u32 = 256;
    tcb.dispatch_threads(
        "gravity_glm_expert_table_zero_f32",
        (n.div_ceil(TG) * TG, 1, 1),
        (TG, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&output), 0);
            enc.set_buffer(1, Some(&miss), 0);
            enc.set_bytes(2, 4, &n as *const u32 as *const _);
        },
    )
}

#[allow(dead_code)]
fn encode_device_expert_table_silu_mul(
    tcb: &mut TokenCommandBuffer<'_>,
    gate: &Buffer,
    up: &Buffer,
    output: &Buffer,
    miss_mask: &Buffer,
    n: usize,
) -> Result<()> {
    require_f32_elements(gate, n, "device expert guarded SiLU gate")?;
    require_f32_elements(up, n, "device expert guarded SiLU up")?;
    require_f32_elements(output, n, "device expert guarded SiLU output")?;
    let gate = gate.clone();
    let up = up.clone();
    let output = output.clone();
    let miss = miss_mask.clone();
    let n = n as u32;
    const TG: u32 = 256;
    tcb.dispatch_threads(
        "gravity_glm_expert_table_silu_mul_f32",
        (n.div_ceil(TG) * TG, 1, 1),
        (TG, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&gate), 0);
            enc.set_buffer(1, Some(&up), 0);
            enc.set_buffer(2, Some(&output), 0);
            enc.set_buffer(3, Some(&miss), 0);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn encode_device_expert_table_axpy(
    tcb: &mut TokenCommandBuffer<'_>,
    output: &Buffer,
    input: &Buffer,
    expert_weights: &Buffer,
    expert_exec_slots: &Buffer,
    miss_mask: &Buffer,
    n: usize,
    experts_per_token: usize,
    execution_position: usize,
    use_router_weight: bool,
) -> Result<()> {
    require_f32_elements(output, n, "device expert guarded AXPY output")?;
    require_f32_elements(input, n, "device expert guarded AXPY input")?;
    if use_router_weight {
        require_f32_elements(
            expert_weights,
            experts_per_token,
            "device expert guarded AXPY weights",
        )?;
        let slot_bytes = experts_per_token
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| Error::Gravity("device expert AXPY slot byte overflow".into()))?
            as u64;
        if expert_exec_slots.length() < slot_bytes || execution_position >= experts_per_token {
            return Err(Error::Gravity(
                "device expert guarded AXPY received invalid execution slots".into(),
            ));
        }
    }
    let params = DeviceExpertTableAxpyParams {
        n: n as u32,
        experts_per_token: experts_per_token as u32,
        execution_position: execution_position as u32,
        use_router_weight: use_router_weight as u32,
    };
    let output = output.clone();
    let input = input.clone();
    let weights = expert_weights.clone();
    let slots = expert_exec_slots.clone();
    let miss = miss_mask.clone();
    const TG: u32 = 256;
    tcb.dispatch_threads(
        "gravity_glm_expert_table_axpy_f32",
        ((n as u32).div_ceil(TG) * TG, 1, 1),
        (TG, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&output), 0);
            enc.set_buffer(1, Some(&input), 0);
            enc.set_buffer(2, Some(&weights), 0);
            enc.set_buffer(3, Some(&slots), 0);
            enc.set_buffer(4, Some(&miss), 0);
            enc.set_bytes(
                5,
                std::mem::size_of_val(&params) as u64,
                &params as *const _ as *const _,
            );
        },
    )
}

#[allow(dead_code)]
fn encode_device_expert_table_residual_add(
    tcb: &mut TokenCommandBuffer<'_>,
    residual: &Buffer,
    expert_output: &Buffer,
    miss_mask: &Buffer,
    n: usize,
) -> Result<()> {
    require_f32_elements(residual, n, "device expert guarded residual")?;
    require_f32_elements(expert_output, n, "device expert guarded residual input")?;
    let residual = residual.clone();
    let expert_output = expert_output.clone();
    let miss = miss_mask.clone();
    let n = n as u32;
    const TG: u32 = 256;
    tcb.dispatch_threads(
        "gravity_glm_expert_table_residual_add_f32",
        (n.div_ceil(TG) * TG, 1, 1),
        (TG, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&residual), 0);
            enc.set_buffer(1, Some(&expert_output), 0);
            enc.set_buffer(2, Some(&miss), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
        },
    )
}

/// Per-layer expanded device K/V cache. DSA index keys have independent
/// session ownership so a future compact K/V layout can reuse them.
struct LayerGpuCache {
    keys: Buffer,
    values: Buffer,
}

struct ExpandedResidentCache {
    layers: Vec<LayerGpuCache>,
    capacity: usize,
}

/// Production-unreachable owner for the compact MLA core state.
///
/// This candidate deliberately remains separate from [`LayerGpuCache`] until
/// the complete compact attention path can replace (rather than accompany)
/// expanded K/V. Constructing the ordinary [`ResidentSession`] therefore
/// allocates none of these buffers.
#[allow(dead_code)]
struct CompactLayerGpuCache {
    latents: Buffer,
    rope_tails: Buffer,
}

#[allow(dead_code)]
struct CompactResidentCache {
    layers: Vec<CompactLayerGpuCache>,
    capacity: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentAttentionLayout {
    Expanded,
    Compact,
}

/// Exactly one attention-cache representation is owned by a session.
///
/// Keeping compact and expanded state in an enum makes a side-by-side
/// flagship allocation unrepresentable. The ordinary constructor selects
/// `Expanded`; compact construction remains private and runtime-unreachable
/// until the compact forward path is wired end to end.
enum ResidentAttentionState {
    Expanded(ExpandedResidentCache),
    Compact(CompactResidentCache),
}

const MIN_SEQUENCE_CAPACITY: usize = 4;

fn checked_sequence_bytes(elements: usize, element_bytes: usize, what: &str) -> Result<usize> {
    elements.checked_mul(element_bytes).ok_or_else(|| {
        Error::Gravity(format!(
            "{what}: sequence buffer size overflow ({elements} elements x {element_bytes} bytes)"
        ))
    })
}

fn grown_sequence_capacity(current: usize, need: usize) -> Result<usize> {
    if need <= current {
        return Ok(current);
    }
    need.checked_next_power_of_two()
        .map(|cap| cap.max(8))
        .ok_or_else(|| {
            Error::Gravity(format!(
                "resident sequence capacity overflow: current={current}, need={need}"
            ))
        })
}

fn active_sequence_len(position: usize, capacity: usize, owner: &str) -> Result<usize> {
    let need = position.checked_add(1).ok_or_else(|| {
        Error::Gravity(format!(
            "{owner}: position {position} cannot be represented as a sequence length"
        ))
    })?;
    if need > capacity {
        return Err(Error::Gravity(format!(
            "{owner}: position {position} needs {need} elements, capacity is {capacity}"
        )));
    }
    Ok(need)
}

impl ExpandedResidentCache {
    fn layer_bytes(capacity: usize, heads: usize, width: usize, what: &str) -> Result<usize> {
        let elements = capacity
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| {
                Error::Gravity(format!(
                    "{what}: expanded cache element count overflow ({capacity} x {heads} x {width})"
                ))
            })?;
        checked_sequence_bytes(elements, std::mem::size_of::<f32>(), what)
    }

    fn new(ctx: &MetalContext, arch: &GlmArch, initial_cap: usize) -> Result<Self> {
        let capacity = initial_cap.max(MIN_SEQUENCE_CAPACITY);
        let key_bytes =
            Self::layer_bytes(capacity, arch.n_heads, arch.qk_dim(), "expanded MLA keys")?;
        let value_bytes = Self::layer_bytes(
            capacity,
            arch.n_heads,
            arch.v_head_dim,
            "expanded MLA values",
        )?;
        let mut layers = Vec::with_capacity(arch.n_layers);
        for _ in 0..arch.n_layers {
            layers.push(LayerGpuCache {
                keys: ctx.new_buffer_checked(key_bytes)?,
                values: ctx.new_buffer_checked(value_bytes)?,
            });
        }
        Ok(Self { layers, capacity })
    }

    fn reserve(
        &mut self,
        ctx: &MetalContext,
        arch: &GlmArch,
        need: usize,
        seq_len: usize,
    ) -> Result<()> {
        if self.layers.len() != arch.n_layers {
            return Err(Error::Gravity(format!(
                "resident expanded cache layer count {} != architecture {}",
                self.layers.len(),
                arch.n_layers
            )));
        }
        if seq_len > self.capacity {
            return Err(Error::Gravity(format!(
                "resident expanded cache seq_len {seq_len} exceeds capacity {}",
                self.capacity
            )));
        }
        let capacity = grown_sequence_capacity(self.capacity, need)?;
        if capacity == self.capacity {
            return Ok(());
        }
        let key_bytes = Self::layer_bytes(
            capacity,
            arch.n_heads,
            arch.qk_dim(),
            "expanded MLA key growth",
        )?;
        let value_bytes = Self::layer_bytes(
            capacity,
            arch.n_heads,
            arch.v_head_dim,
            "expanded MLA value growth",
        )?;
        let key_copy_bytes = Self::layer_bytes(
            seq_len,
            arch.n_heads,
            arch.qk_dim(),
            "expanded MLA key copy",
        )?;
        let value_copy_bytes = Self::layer_bytes(
            seq_len,
            arch.n_heads,
            arch.v_head_dim,
            "expanded MLA value copy",
        )?;
        let mut next_layers = Vec::with_capacity(arch.n_layers);
        for layer in &self.layers {
            let next_keys = ctx.new_buffer_checked(key_bytes)?;
            let next_values = ctx.new_buffer_checked(value_bytes)?;
            if seq_len > 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        layer.keys.contents() as *const u8,
                        next_keys.contents() as *mut u8,
                        key_copy_bytes,
                    );
                    std::ptr::copy_nonoverlapping(
                        layer.values.contents() as *const u8,
                        next_values.contents() as *mut u8,
                        value_copy_bytes,
                    );
                }
            }
            next_layers.push(LayerGpuCache {
                keys: next_keys,
                values: next_values,
            });
        }
        self.layers = next_layers;
        self.capacity = capacity;
        Ok(())
    }
}

impl CompactResidentCache {
    fn layer_bytes(capacity: usize, width: usize, what: &str) -> Result<usize> {
        let elements = capacity.checked_mul(width).ok_or_else(|| {
            Error::Gravity(format!(
                "{what}: compact cache element count overflow ({capacity} x {width})"
            ))
        })?;
        checked_sequence_bytes(elements, std::mem::size_of::<f32>(), what)
    }

    fn new(ctx: &MetalContext, arch: &GlmArch, initial_cap: usize) -> Result<Self> {
        let capacity = initial_cap.max(MIN_SEQUENCE_CAPACITY);
        let latent_bytes =
            Self::layer_bytes(capacity, arch.kv_lora_rank, "compact MLA latent cache")?;
        let rope_bytes =
            Self::layer_bytes(capacity, arch.qk_rope_head_dim, "compact MLA rope cache")?;
        let mut layers = Vec::with_capacity(arch.n_layers);
        for _ in 0..arch.n_layers {
            layers.push(CompactLayerGpuCache {
                latents: ctx.new_buffer_checked(latent_bytes)?,
                rope_tails: ctx.new_buffer_checked(rope_bytes)?,
            });
        }
        Ok(Self { layers, capacity })
    }

    fn reserve(
        &mut self,
        ctx: &MetalContext,
        arch: &GlmArch,
        need: usize,
        seq_len: usize,
    ) -> Result<()> {
        if self.layers.len() != arch.n_layers {
            return Err(Error::Gravity(format!(
                "compact MLA cache layer count {} != architecture {}",
                self.layers.len(),
                arch.n_layers
            )));
        }
        if seq_len > self.capacity {
            return Err(Error::Gravity(format!(
                "compact MLA cache seq_len {} exceeds capacity {}",
                seq_len, self.capacity
            )));
        }
        let capacity = grown_sequence_capacity(self.capacity, need)?;
        if capacity == self.capacity {
            return Ok(());
        }
        let latent_bytes = Self::layer_bytes(
            capacity,
            arch.kv_lora_rank,
            "compact MLA latent cache growth",
        )?;
        let rope_bytes = Self::layer_bytes(
            capacity,
            arch.qk_rope_head_dim,
            "compact MLA rope cache growth",
        )?;
        let latent_copy_bytes =
            Self::layer_bytes(seq_len, arch.kv_lora_rank, "compact MLA latent cache copy")?;
        let rope_copy_bytes = Self::layer_bytes(
            seq_len,
            arch.qk_rope_head_dim,
            "compact MLA rope cache copy",
        )?;
        let mut next_layers = Vec::with_capacity(arch.n_layers);
        for layer in &self.layers {
            let next_latents = ctx.new_buffer_checked(latent_bytes)?;
            let next_rope_tails = ctx.new_buffer_checked(rope_bytes)?;
            if seq_len > 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        layer.latents.contents() as *const u8,
                        next_latents.contents() as *mut u8,
                        latent_copy_bytes,
                    );
                    std::ptr::copy_nonoverlapping(
                        layer.rope_tails.contents() as *const u8,
                        next_rope_tails.contents() as *mut u8,
                        rope_copy_bytes,
                    );
                }
            }
            next_layers.push(CompactLayerGpuCache {
                latents: next_latents,
                rope_tails: next_rope_tails,
            });
        }
        self.layers = next_layers;
        self.capacity = capacity;
        Ok(())
    }
}

impl ResidentAttentionState {
    fn new(
        ctx: &MetalContext,
        arch: &GlmArch,
        initial_cap: usize,
        layout: ResidentAttentionLayout,
    ) -> Result<Self> {
        match layout {
            ResidentAttentionLayout::Expanded => Ok(Self::Expanded(ExpandedResidentCache::new(
                ctx,
                arch,
                initial_cap,
            )?)),
            ResidentAttentionLayout::Compact => Ok(Self::Compact(CompactResidentCache::new(
                ctx,
                arch,
                initial_cap,
            )?)),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Expanded(cache) => cache.capacity,
            Self::Compact(cache) => cache.capacity,
        }
    }

    fn is_compact(&self) -> bool {
        matches!(self, Self::Compact(_))
    }

    fn reserve(
        &mut self,
        ctx: &MetalContext,
        arch: &GlmArch,
        need: usize,
        seq_len: usize,
    ) -> Result<usize> {
        match self {
            Self::Expanded(cache) => {
                cache.reserve(ctx, arch, need, seq_len)?;
                Ok(cache.capacity)
            }
            Self::Compact(cache) => {
                cache.reserve(ctx, arch, need, seq_len)?;
                Ok(cache.capacity)
            }
        }
    }

    fn expanded_layer(&self, layer: usize) -> Result<&LayerGpuCache> {
        match self {
            Self::Expanded(cache) => cache.layers.get(layer).ok_or_else(|| {
                Error::Gravity(format!(
                    "resident expanded cache layer {layer} out of range {}",
                    cache.layers.len()
                ))
            }),
            Self::Compact(_) => Err(Error::Gravity(
                "compact resident attention state reached the expanded forward path".into(),
            )),
        }
    }

    fn compact_layer(&self, layer: usize) -> Result<&CompactLayerGpuCache> {
        match self {
            Self::Compact(cache) => cache.layers.get(layer).ok_or_else(|| {
                Error::Gravity(format!(
                    "resident compact cache layer {layer} out of range {}",
                    cache.layers.len()
                ))
            }),
            Self::Expanded(_) => Err(Error::Gravity(
                "expanded resident attention state reached the compact forward path".into(),
            )),
        }
    }
}

/// Host workspaces whose lengths track the resident sequence capacity.
///
/// Keeping these vectors at their reserved length means index scoring,
/// stable top-k selection, and sparse-attention masking do not allocate
/// sequence-sized temporaries after [`ResidentSession::reserve`] succeeds.
#[derive(Debug)]
struct HostSequenceScratch {
    index_scores: Vec<f32>,
    selection_indices: Vec<usize>,
    attention_allowed: Vec<u8>,
    attention_scores: Vec<f32>,
}

impl HostSequenceScratch {
    fn new(capacity: usize) -> Self {
        Self {
            index_scores: vec![0.0; capacity],
            selection_indices: vec![0; capacity],
            attention_allowed: vec![0; capacity],
            attention_scores: vec![f32::NEG_INFINITY; capacity],
        }
    }

    fn grow_preserving(&mut self, capacity: usize) {
        if capacity <= self.index_scores.len() {
            return;
        }
        self.index_scores.resize(capacity, 0.0);
        self.selection_indices.resize(capacity, 0);
        self.attention_allowed.resize(capacity, 0);
        self.attention_scores.resize(capacity, f32::NEG_INFINITY);
    }
}

/// Sequence-sized DSA/index-selection scratch owned by one resident session.
///
/// `ActPool` is model-global and fixed-size, so sequence-dependent buffers do
/// not belong there. This workspace grows in lockstep with the session's KV
/// caches and is then reused serially by every layer.
struct SequenceScratch {
    index_scores_device: Buffer,
    host: HostSequenceScratch,
    capacity: usize,
    device_score_len: usize,
}

impl SequenceScratch {
    fn new(ctx: &MetalContext, initial_cap: usize) -> Result<Self> {
        let capacity = initial_cap.max(MIN_SEQUENCE_CAPACITY);
        let bytes = checked_sequence_bytes(
            capacity,
            std::mem::size_of::<f32>(),
            "resident index scores",
        )?;
        Ok(Self {
            index_scores_device: ctx.new_buffer_checked(bytes)?,
            host: HostSequenceScratch::new(capacity),
            capacity,
            device_score_len: 0,
        })
    }

    fn reserve(&mut self, ctx: &MetalContext, need: usize) -> Result<()> {
        let capacity = grown_sequence_capacity(self.capacity, need)?;
        if capacity == self.capacity {
            return Ok(());
        }

        let bytes = checked_sequence_bytes(
            capacity,
            std::mem::size_of::<f32>(),
            "resident index scores",
        )?;
        let next = ctx.new_buffer_checked(bytes)?;
        if self.device_score_len > 0 {
            let copy_bytes = checked_sequence_bytes(
                self.device_score_len,
                std::mem::size_of::<f32>(),
                "resident index score copy",
            )?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.index_scores_device.contents() as *const u8,
                    next.contents() as *mut u8,
                    copy_bytes,
                );
            }
        }
        self.host.grow_preserving(capacity);
        self.index_scores_device = next;
        self.capacity = capacity;
        Ok(())
    }

    fn active_len(&self, position: usize) -> Result<usize> {
        active_sequence_len(position, self.capacity, "resident sequence scratch")
    }

    fn store_index_scores(&mut self, len: usize) -> Result<()> {
        if len > self.capacity || len > self.host.index_scores.len() {
            return Err(Error::Gravity(format!(
                "resident index score write needs {len} elements, capacity is {}",
                self.capacity
            )));
        }
        let bytes = checked_sequence_bytes(
            len,
            std::mem::size_of::<f32>(),
            "resident index score write",
        )?;
        if bytes as u64 > self.index_scores_device.length() {
            return Err(Error::Gravity(format!(
                "resident index score write needs {bytes} bytes, device buffer has {}",
                self.index_scores_device.length()
            )));
        }
        write_f32(&self.index_scores_device, &self.host.index_scores[..len]);
        self.device_score_len = len;
        Ok(())
    }
}

/// DSA/index state has an independent capacity and recovery path from MLA K/V.
///
/// A failed attention-cache growth may leave attention at the requested
/// capacity while this owner remains unchanged. Retrying `reserve` therefore
/// rechecks and repairs DSA state independently instead of returning early
/// from the attention capacity alone.
struct DsaIndexState {
    index_keys: Vec<Buffer>,
    sequence_scratch: SequenceScratch,
    shared_topk: Option<Vec<usize>>,
    ranked_indices: Option<Buffer>,
    device_selection: bool,
    ranked_capacity: usize,
    capacity: usize,
}

impl DsaIndexState {
    fn new(
        ctx: &MetalContext,
        arch: &GlmArch,
        initial_cap: usize,
        compact_rank_upload: bool,
        device_dsa: bool,
    ) -> Result<Self> {
        if device_dsa && !compact_rank_upload {
            return Err(Error::Gravity(
                "device DSA requires compact ranked-index state".into(),
            ));
        }
        let capacity = initial_cap.max(MIN_SEQUENCE_CAPACITY);
        let elements = capacity.checked_mul(arch.index_head_dim).ok_or_else(|| {
            Error::Gravity(format!(
                "resident DSA index-key element count overflow: {capacity} x {}",
                arch.index_head_dim
            ))
        })?;
        let bytes = checked_sequence_bytes(elements, std::mem::size_of::<f32>(), "DSA index keys")?;
        let mut index_keys = Vec::with_capacity(arch.n_layers);
        for _ in 0..arch.n_layers {
            index_keys.push(ctx.new_buffer_checked(bytes)?);
        }
        let ranked_capacity = arch.index_topk.max(1);
        let ranked_indices = if compact_rank_upload {
            let ranked_bytes = checked_sequence_bytes(
                ranked_capacity,
                std::mem::size_of::<u32>(),
                "DSA ranked indices",
            )?;
            Some(ctx.new_buffer_checked(ranked_bytes)?)
        } else {
            None
        };
        Ok(Self {
            index_keys,
            sequence_scratch: SequenceScratch::new(ctx, capacity)?,
            shared_topk: None,
            ranked_indices,
            device_selection: device_dsa,
            ranked_capacity: if compact_rank_upload {
                ranked_capacity
            } else {
                0
            },
            capacity,
        })
    }

    fn reset(&mut self) {
        self.shared_topk = None;
        self.sequence_scratch.device_score_len = 0;
    }

    fn reserve(
        &mut self,
        ctx: &MetalContext,
        arch: &GlmArch,
        need: usize,
        seq_len: usize,
    ) -> Result<()> {
        if self.index_keys.len() != arch.n_layers {
            return Err(Error::Gravity(format!(
                "resident index cache layer count {} != architecture {}",
                self.index_keys.len(),
                arch.n_layers
            )));
        }
        if self.sequence_scratch.capacity != self.capacity {
            return Err(Error::Gravity(format!(
                "resident DSA scratch capacity {} != index capacity {}",
                self.sequence_scratch.capacity, self.capacity
            )));
        }
        if seq_len > self.capacity {
            return Err(Error::Gravity(format!(
                "resident DSA seq_len {seq_len} exceeds capacity {}",
                self.capacity
            )));
        }
        let capacity = grown_sequence_capacity(self.capacity, need)?;
        if capacity == self.capacity {
            return Ok(());
        }
        let elements = capacity.checked_mul(arch.index_head_dim).ok_or_else(|| {
            Error::Gravity(format!(
                "resident DSA index-key growth overflow: {capacity} x {}",
                arch.index_head_dim
            ))
        })?;
        let bytes =
            checked_sequence_bytes(elements, std::mem::size_of::<f32>(), "DSA index-key growth")?;
        let copy_elements = seq_len.checked_mul(arch.index_head_dim).ok_or_else(|| {
            Error::Gravity(format!(
                "resident DSA index-key copy overflow: {seq_len} x {}",
                arch.index_head_dim
            ))
        })?;
        let copy_bytes = checked_sequence_bytes(
            copy_elements,
            std::mem::size_of::<f32>(),
            "DSA index-key copy",
        )?;
        let mut next_index_keys = Vec::with_capacity(arch.n_layers);
        for index_keys in &self.index_keys {
            let next = ctx.new_buffer_checked(bytes)?;
            if seq_len > 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        index_keys.contents() as *const u8,
                        next.contents() as *mut u8,
                        copy_bytes,
                    );
                }
            }
            next_index_keys.push(next);
        }
        self.sequence_scratch.reserve(ctx, capacity)?;
        self.index_keys = next_index_keys;
        self.capacity = capacity;
        Ok(())
    }

    fn store_ranked_indices(&self, ranked: &[usize]) -> Result<()> {
        if ranked.len() > self.ranked_capacity {
            return Err(Error::Gravity(format!(
                "resident DSA rank upload needs {} slots, capacity is {}",
                ranked.len(),
                self.ranked_capacity
            )));
        }
        let mut checked = Vec::with_capacity(ranked.len());
        for &index in ranked {
            checked.push(u32::try_from(index).map_err(|_| {
                Error::Gravity(format!("resident DSA rank index {index} exceeds u32"))
            })?);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                checked.as_ptr(),
                self.ranked_indices
                    .as_ref()
                    .ok_or_else(|| {
                        Error::Gravity(
                            "resident DSA rank upload requested without compact state".into(),
                        )
                    })?
                    .contents() as *mut u32,
                checked.len(),
            );
        }
        Ok(())
    }

    fn ranked_indices(&self) -> Result<&Buffer> {
        self.ranked_indices.as_ref().ok_or_else(|| {
            Error::Gravity("resident compact attention has no DSA rank buffer".into())
        })
    }

    fn device_selection_enabled(&self) -> bool {
        self.device_selection
    }
}

/// Reuses the session's O(sequence-length) index workspace for [`topk_desc`].
///
/// The index tie-break makes the comparator a total order even though the
/// backing sort is unstable, preserving the reference's ascending-index
/// result for equal finite scores without allocating a stable-sort merge
/// buffer. The returned O(k) result remains owned, matching the existing
/// resident-path interface.
fn topk_desc_with_scratch(
    values: &[f32],
    k: usize,
    selection_indices: &mut [usize],
) -> Result<Vec<usize>> {
    if selection_indices.len() < values.len() {
        return Err(Error::Gravity(format!(
            "resident top-k selection needs {} indices, scratch has {}",
            values.len(),
            selection_indices.len()
        )));
    }
    let indices = &mut selection_indices[..values.len()];
    for (index, slot) in indices.iter_mut().enumerate() {
        *slot = index;
    }
    indices.sort_unstable_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    Ok(indices[..k.min(indices.len())].to_vec())
}

/// Device-resident working set for one generation.
pub struct ResidentSession {
    attention: ResidentAttentionState,
    dsa: DsaIndexState,
    pub seq_len: usize,
    waits: Cell<u64>,
}

impl ResidentSession {
    pub fn new(ctx: &MetalContext, arch: &GlmArch, initial_cap: usize) -> Result<Self> {
        Self::new_with_layout(
            ctx,
            arch,
            initial_cap,
            ResidentAttentionLayout::Expanded,
            false,
        )
    }

    #[allow(dead_code)]
    fn new_compact(ctx: &MetalContext, arch: &GlmArch, initial_cap: usize) -> Result<Self> {
        Self::new_with_layout(
            ctx,
            arch,
            initial_cap,
            ResidentAttentionLayout::Compact,
            false,
        )
    }

    fn new_with_layout(
        ctx: &MetalContext,
        arch: &GlmArch,
        initial_cap: usize,
        layout: ResidentAttentionLayout,
        device_dsa: bool,
    ) -> Result<Self> {
        if device_dsa && layout != ResidentAttentionLayout::Compact {
            return Err(Error::Gravity(
                "resident device DSA requires compact attention layout".into(),
            ));
        }
        let cap = initial_cap.max(MIN_SEQUENCE_CAPACITY);
        let attention = ResidentAttentionState::new(ctx, arch, cap, layout)?;
        Ok(Self {
            attention,
            dsa: DsaIndexState::new(
                ctx,
                arch,
                cap,
                layout == ResidentAttentionLayout::Compact,
                device_dsa,
            )?,
            seq_len: 0,
            waits: Cell::new(0),
        })
    }

    pub fn reset(&mut self) {
        self.dsa.reset();
        self.seq_len = 0;
        self.waits.set(0);
    }

    pub fn waits(&self) -> u64 {
        self.waits.get()
    }

    fn reserve(&mut self, ctx: &MetalContext, arch: &GlmArch, need: usize) -> Result<()> {
        self.attention.reserve(ctx, arch, need, self.seq_len)?;
        self.dsa.reserve(ctx, arch, need, self.seq_len)?;
        Ok(())
    }
}

/// Activation / scratch pool (device buffers, reused every token).
pub struct ActPool {
    pub x: Buffer,
    h: Buffer,
    q_a: Buffer,
    q_resid: Buffer,
    q: Buffer,
    compressed: Buffer,
    k_latent: Buffer,
    kv: Buffer,
    queries: Buffer,
    context: Buffer,
    o: Buffer,
    // DSA / router scratch kept on device as the cache of record
    idx_q: Buffer,
    idx_k_raw: Buffer,
    idx_head_w: Buffer,
    router_logits: Buffer,
    router_bias: Buffer,
    router_scores: Buffer,
    router_corrected: Buffer,
    expert_idx: Buffer,
    expert_w: Buffer,
    /// Permutation of score-ranked expert slots in ascending expert-ID order.
    expert_exec_slots: Buffer,
    // Expert scratch (sized for future device-side expert chaining; the
    // batched path currently uses matvec_batch into host Vecs for the three
    // co-issued waits that match the host oracle).
    #[allow(dead_code)]
    gate: Buffer,
    #[allow(dead_code)]
    up: Buffer,
    #[allow(dead_code)]
    act: Buffer,
    #[allow(dead_code)]
    down: Buffer,
    final_norm_weight: Buffer,
    final_hidden: Buffer,
    /// Device logits for device-resident lm_head (vocab-sized). Stay on device;
    /// host only reads them under `HAWKING_GLM_GPU_LM_HEAD_FULL_LOGITS=1`.
    logits: Buffer,
    /// Single u32 greedy token from on-device argmax (token-only readback).
    sample_token: Buffer,
    /// Diagnostic top-k indices over device logits (`GPU_LM_HEAD_DIAG_TOPK`).
    head_topk_idx: Buffer,
    /// Diagnostic top-k values over device logits.
    head_topk_val: Buffer,
    #[allow(dead_code)]
    gate_cap: usize,
    /// Lazily allocated only for the default-off compact MLA path.
    compact_attention_scratch: Mutex<Option<CompactAttentionScratch>>,
    /// Lazily allocated only for device DSA's normalization/RoPE graph.
    device_dsa_transform_scratch: Mutex<Option<DeviceDsaTransformScratch>>,
    /// Lazily allocated only when compact attention's projection prelude can
    /// stay entirely device-side.
    device_attention_prelude_scratch: Mutex<Option<DeviceAttentionPreludeScratch>>,
    /// Grow-once scratch for the default-off expert-wave candidate. Keeping
    /// this lazy preserves the default path's allocation and residency shape.
    expert_wave_scratch: Mutex<Option<ExpertWaveScratch>>,
}

struct CompactAttentionScratch {
    query_nope: Buffer,
    query_rope: Buffer,
    key_rope: Buffer,
    query_latent: Buffer,
    n_heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    latent_dim: usize,
}

impl CompactAttentionScratch {
    fn new(ctx: &MetalContext, arch: &GlmArch) -> Result<Self> {
        Ok(Self {
            query_nope: ctx.new_buffer_checked(arch.n_heads * arch.qk_nope_head_dim * 4)?,
            query_rope: ctx.new_buffer_checked(arch.n_heads * arch.qk_rope_head_dim * 4)?,
            key_rope: ctx.new_buffer_checked(arch.qk_rope_head_dim * 4)?,
            query_latent: ctx.new_buffer_checked(arch.n_heads * arch.kv_lora_rank * 4)?,
            n_heads: arch.n_heads,
            nope_dim: arch.qk_nope_head_dim,
            rope_dim: arch.qk_rope_head_dim,
            latent_dim: arch.kv_lora_rank,
        })
    }

    fn matches(&self, arch: &GlmArch) -> bool {
        self.n_heads == arch.n_heads
            && self.nope_dim == arch.qk_nope_head_dim
            && self.rope_dim == arch.qk_rope_head_dim
            && self.latent_dim == arch.kv_lora_rank
    }
}

struct DeviceDsaTransformScratch {
    query: Buffer,
    cos: Buffer,
    sin: Buffer,
    norm_weight: Buffer,
    norm_bias: Buffer,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
}

impl DeviceDsaTransformScratch {
    fn new(ctx: &MetalContext, arch: &GlmArch) -> Result<Self> {
        if arch.index_n_heads == 0
            || arch.index_head_dim == 0
            || arch.qk_rope_head_dim == 0
            || arch.qk_rope_head_dim % 2 != 0
            || arch.qk_rope_head_dim > arch.index_head_dim
        {
            return Err(Error::Gravity(format!(
                "device DSA transform scratch has invalid geometry: heads={} head_dim={} rope_dim={}",
                arch.index_n_heads, arch.index_head_dim, arch.qk_rope_head_dim
            )));
        }
        let rope_half = arch.qk_rope_head_dim / 2;
        Ok(Self {
            query: ctx.new_buffer_checked(
                arch.index_n_heads
                    .checked_mul(arch.index_head_dim)
                    .and_then(|elements| elements.checked_mul(4))
                    .ok_or_else(|| {
                        Error::Gravity(
                            "device DSA transformed-query scratch byte size overflow".into(),
                        )
                    })?,
            )?,
            cos: ctx.new_buffer_checked(rope_half * 4)?,
            sin: ctx.new_buffer_checked(rope_half * 4)?,
            norm_weight: ctx.new_buffer_checked(arch.index_head_dim * 4)?,
            norm_bias: ctx.new_buffer_checked(arch.index_head_dim * 4)?,
            n_heads: arch.index_n_heads,
            head_dim: arch.index_head_dim,
            rope_dim: arch.qk_rope_head_dim,
        })
    }

    fn matches(&self, arch: &GlmArch) -> bool {
        self.n_heads == arch.index_n_heads
            && self.head_dim == arch.index_head_dim
            && self.rope_dim == arch.qk_rope_head_dim
    }
}

struct DeviceAttentionPreludeScratch {
    input_norm_weight: Buffer,
    q_norm_weight: Buffer,
    kv_norm_weight: Buffer,
    cos: Buffer,
    sin: Buffer,
    hidden: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    rope_dim: usize,
}

impl DeviceAttentionPreludeScratch {
    fn new(ctx: &MetalContext, arch: &GlmArch) -> Result<Self> {
        if arch.hidden == 0
            || arch.q_lora_rank == 0
            || arch.kv_lora_rank == 0
            || arch.qk_rope_head_dim == 0
            || arch.qk_rope_head_dim % 2 != 0
        {
            return Err(Error::Gravity(format!(
                "device attention prelude scratch has invalid geometry: hidden={} q_lora={} kv_lora={} rope_dim={}",
                arch.hidden, arch.q_lora_rank, arch.kv_lora_rank, arch.qk_rope_head_dim
            )));
        }
        let rope_half = arch.qk_rope_head_dim / 2;
        Ok(Self {
            input_norm_weight: ctx.new_buffer_checked(arch.hidden * 4)?,
            q_norm_weight: ctx.new_buffer_checked(arch.q_lora_rank * 4)?,
            kv_norm_weight: ctx.new_buffer_checked(arch.kv_lora_rank * 4)?,
            cos: ctx.new_buffer_checked(rope_half * 4)?,
            sin: ctx.new_buffer_checked(rope_half * 4)?,
            hidden: arch.hidden,
            q_lora_rank: arch.q_lora_rank,
            kv_lora_rank: arch.kv_lora_rank,
            rope_dim: arch.qk_rope_head_dim,
        })
    }

    fn matches(&self, arch: &GlmArch) -> bool {
        self.hidden == arch.hidden
            && self.q_lora_rank == arch.q_lora_rank
            && self.kv_lora_rank == arch.kv_lora_rank
            && self.rope_dim == arch.qk_rope_head_dim
    }
}

struct ExpertWaveScratch {
    expert_capacity: usize,
    intermediate_capacity: usize,
    hidden_capacity: usize,
    gate: Vec<Buffer>,
    up: Vec<Buffer>,
    act: Vec<Buffer>,
    down: Vec<Buffer>,
    combined: Buffer,
}

impl ExpertWaveScratch {
    fn new(
        ctx: &MetalContext,
        expert_capacity: usize,
        intermediate_capacity: usize,
        hidden_capacity: usize,
    ) -> Result<Self> {
        if expert_capacity == 0 || intermediate_capacity == 0 || hidden_capacity == 0 {
            return Err(Error::Gravity(format!(
                "expert-wave scratch dimensions must be nonzero: experts={expert_capacity} \
                 intermediate={intermediate_capacity} hidden={hidden_capacity}"
            )));
        }
        let inter_bytes = intermediate_capacity.checked_mul(4).ok_or_else(|| {
            Error::Gravity(format!(
                "expert-wave intermediate scratch byte size overflow: {intermediate_capacity}"
            ))
        })?;
        let hidden_bytes = hidden_capacity.checked_mul(4).ok_or_else(|| {
            Error::Gravity(format!(
                "expert-wave hidden scratch byte size overflow: {hidden_capacity}"
            ))
        })?;
        let allocate_many = |count: usize, bytes: usize| -> Result<Vec<Buffer>> {
            (0..count).map(|_| ctx.new_buffer_checked(bytes)).collect()
        };
        Ok(Self {
            expert_capacity,
            intermediate_capacity,
            hidden_capacity,
            gate: allocate_many(expert_capacity, inter_bytes)?,
            up: allocate_many(expert_capacity, inter_bytes)?,
            act: allocate_many(expert_capacity, inter_bytes)?,
            down: allocate_many(expert_capacity, hidden_bytes)?,
            combined: ctx.new_buffer_checked(hidden_bytes)?,
        })
    }

    fn fits(&self, experts: usize, intermediate: usize, hidden: usize) -> bool {
        experts <= self.expert_capacity
            && intermediate <= self.intermediate_capacity
            && hidden <= self.hidden_capacity
    }
}

impl ActPool {
    pub fn new(ctx: &MetalContext, arch: &GlmArch) -> Result<Self> {
        let h = arch.hidden;
        let qk = arch.qk_dim();
        let gate_cap = (h * 32).max(4096);
        Ok(Self {
            x: ctx.new_buffer_checked(h * 4)?,
            h: ctx.new_buffer_checked(h * 4)?,
            q_a: ctx.new_buffer_checked(arch.q_lora_rank * 4)?,
            q_resid: ctx.new_buffer_checked(arch.q_lora_rank * 4)?,
            q: ctx.new_buffer_checked(arch.n_heads * qk * 4)?,
            compressed: ctx.new_buffer_checked((arch.kv_lora_rank + arch.qk_rope_head_dim) * 4)?,
            k_latent: ctx.new_buffer_checked(arch.kv_lora_rank * 4)?,
            kv: ctx
                .new_buffer_checked(arch.n_heads * (arch.qk_nope_head_dim + arch.v_head_dim) * 4)?,
            queries: ctx.new_buffer_checked(arch.n_heads * qk * 4)?,
            context: ctx.new_buffer_checked(arch.n_heads * arch.v_head_dim * 4)?,
            o: ctx.new_buffer_checked(h * 4)?,
            idx_q: ctx.new_buffer_checked(arch.index_n_heads * arch.index_head_dim * 4)?,
            idx_k_raw: ctx.new_buffer_checked(arch.index_head_dim * 4)?,
            idx_head_w: ctx.new_buffer_checked(arch.index_n_heads * 4)?,
            router_logits: ctx.new_buffer_checked(arch.n_routed_experts * 4)?,
            router_bias: ctx.new_buffer_checked(arch.n_routed_experts * 4)?,
            router_scores: ctx.new_buffer_checked(arch.n_routed_experts * 4)?,
            router_corrected: ctx.new_buffer_checked(arch.n_routed_experts * 4)?,
            expert_idx: ctx.new_buffer_checked(arch.num_experts_per_tok.max(1) * 4)?,
            expert_w: ctx.new_buffer_checked(arch.num_experts_per_tok.max(1) * 4)?,
            expert_exec_slots: ctx.new_buffer_checked(arch.num_experts_per_tok.max(1) * 4)?,
            gate: ctx.new_buffer_checked(gate_cap * 4)?,
            up: ctx.new_buffer_checked(gate_cap * 4)?,
            act: ctx.new_buffer_checked(gate_cap * 4)?,
            down: ctx.new_buffer_checked(h * 4)?,
            final_norm_weight: ctx.new_buffer_checked(h * 4)?,
            final_hidden: ctx.new_buffer_checked(h * 4)?,
            logits: ctx.new_buffer_checked(arch.vocab_size * 4)?,
            sample_token: ctx.new_buffer_checked(4)?,
            head_topk_idx: ctx.new_buffer_checked((GPU_LM_HEAD_DIAG_TOPK as usize) * 4)?,
            head_topk_val: ctx.new_buffer_checked((GPU_LM_HEAD_DIAG_TOPK as usize) * 4)?,
            gate_cap,
            compact_attention_scratch: Mutex::new(None),
            device_dsa_transform_scratch: Mutex::new(None),
            device_attention_prelude_scratch: Mutex::new(None),
            expert_wave_scratch: Mutex::new(None),
        })
    }

    fn ensure_compact_attention_scratch(
        &self,
        ctx: &MetalContext,
        arch: &GlmArch,
    ) -> Result<std::sync::MutexGuard<'_, Option<CompactAttentionScratch>>> {
        let mut scratch = self
            .compact_attention_scratch
            .lock()
            .expect("compact attention scratch");
        if !scratch.as_ref().is_some_and(|state| state.matches(arch)) {
            *scratch = Some(CompactAttentionScratch::new(ctx, arch)?);
        }
        Ok(scratch)
    }

    fn ensure_device_dsa_transform_scratch(
        &self,
        ctx: &MetalContext,
        arch: &GlmArch,
    ) -> Result<std::sync::MutexGuard<'_, Option<DeviceDsaTransformScratch>>> {
        let mut scratch = self
            .device_dsa_transform_scratch
            .lock()
            .expect("device DSA transform scratch");
        if !scratch.as_ref().is_some_and(|state| state.matches(arch)) {
            *scratch = Some(DeviceDsaTransformScratch::new(ctx, arch)?);
        }
        Ok(scratch)
    }

    fn ensure_device_attention_prelude_scratch(
        &self,
        ctx: &MetalContext,
        arch: &GlmArch,
    ) -> Result<std::sync::MutexGuard<'_, Option<DeviceAttentionPreludeScratch>>> {
        let mut scratch = self
            .device_attention_prelude_scratch
            .lock()
            .expect("device attention prelude scratch");
        if !scratch.as_ref().is_some_and(|state| state.matches(arch)) {
            *scratch = Some(DeviceAttentionPreludeScratch::new(ctx, arch)?);
        }
        Ok(scratch)
    }

    fn ensure_expert_wave_scratch(
        &self,
        ctx: &MetalContext,
        experts: usize,
        intermediate: usize,
        hidden: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<ExpertWaveScratch>>> {
        let mut scratch = self
            .expert_wave_scratch
            .lock()
            .expect("expert-wave scratch");
        let fits = scratch
            .as_ref()
            .is_some_and(|s| s.fits(experts, intermediate, hidden));
        if !fits {
            let expert_capacity = scratch
                .as_ref()
                .map_or(experts, |s| s.expert_capacity.max(experts));
            let intermediate_capacity = scratch
                .as_ref()
                .map_or(intermediate, |s| s.intermediate_capacity.max(intermediate));
            let hidden_capacity = scratch
                .as_ref()
                .map_or(hidden, |s| s.hidden_capacity.max(hidden));
            *scratch = Some(ExpertWaveScratch::new(
                ctx,
                expert_capacity,
                intermediate_capacity,
                hidden_capacity,
            )?);
        }
        Ok(scratch)
    }
}

fn commit(tcb: Option<TokenCommandBuffer<'_>>, waits: &Cell<u64>) -> Result<()> {
    if let Some(buf) = tcb {
        // When the cost ledger is recording, commit folds metal_encode /
        // metal_submit / metal_synchronize + GPU timestamps. Off path is
        // the historical uninstrumented flush (single atomic load).
        buf.commit_and_wait()?;
        waits.set(waits.get().saturating_add(1));
    }
    Ok(())
}

fn record_dense_matvec_ops(rows: u64, cols: u64) {
    let fp = rows.saturating_mul(cols).saturating_mul(2);
    crate::cost_ledger::record_source_modelled_operations(fp, 0, 0, 0, fp);
}

fn record_pq_matvec_ops(params: crate::gravity_glm::gpu::PqParams) {
    let rows = params.rows as u64;
    let dense_fp = rows.saturating_mul(params.cols as u64).saturating_mul(2);
    // Kernel source executes one FMA per logical weight plus a 32-lane
    // simd_sum (31 adds/row). pq_index's visible bit arithmetic is a
    // documented 15-op lower bound per row/chunk/subspace lookup.
    let fp = dense_fp.saturating_add(rows.saturating_mul(31));
    let lookups = rows
        .saturating_mul(params.nchunk as u64)
        .saturating_mul(params.subspaces as u64);
    crate::cost_ledger::record_source_modelled_operations(
        fp,
        lookups.saturating_mul(15),
        0,
        0,
        dense_fp,
    );
}

fn matvecs_are_device_encodable(weights: &GpuWeightCache, names: &[&str]) -> Result<bool> {
    let mut cache = weights.cache.lock().expect("gpu weight cache");
    for &name in names {
        weights.ensure_many_locked(&mut cache, &[name])?;
        if !matches!(
            cache.get(name).expect("ensured device-encodability probe"),
            GpuTensor::Pq { .. } | GpuTensor::NativeGpuBf16 { .. }
        ) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Matvec into a device buffer. Host-native weights run on the host into the
/// shared buffer (no wait). PQ and device-resident bf16 encode into `tcb` and
/// need a later commit.
fn matvec_into<'a>(
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    weights: &GpuWeightCache,
    name: &str,
    x: &Buffer,
    x_len: usize,
    y: &Buffer,
) -> Result<()> {
    crate::cost_ledger::record_matvec_call();
    let mut cache = weights.cache.lock().expect("gpu weight cache");
    weights.ensure_many_locked(&mut cache, &[name])?;
    let tensor = cache.get(name).expect("ensured");
    record_routed_tensor_representation(name, tensor);
    match tensor {
        GpuTensor::NativeCpu(w) => {
            // Host-oracle path (flag off): do not change default billing or
            // numerics. Active-byte category partition for native.f32 widen is
            // owned by WeightAccess::matvec when that path is used — still
            // bill here so the resident path is not a blind hole when the
            // ledger is on.
            crate::cost_ledger::record_active_bytes_for(name, (w.len() * 4) as u64);
            record_dense_matvec_ops((w.len() / x_len) as u64, x_len as u64);
            let x_host = read_f32(x, x_len);
            let y_host = matvec_dense(w, &x_host, name)?;
            write_f32(y, &y_host);
            Ok(())
        }
        GpuTensor::NativeGpuBf16 { buf, rows, cols } => {
            if x_len != *cols as usize {
                return Err(Error::Gravity(format!(
                    "resident matvec {name}: x_len {x_len} != cols {cols}"
                )));
            }
            // Bill stored bf16 length (no f32 widen tax).
            crate::cost_ledger::record_active_bytes_for(name, buf.length());
            record_dense_matvec_ops(*rows as u64, *cols as u64);
            let tcb = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
            // MetalEncode charged at TCB commit from dispatch_threads wall.
            encode_gemv_native_bf16_seq(tcb, buf, *rows, *cols, x, y)
        }
        GpuTensor::Pq {
            codebooks,
            codes,
            params,
        } => {
            if x_len != params.cols as usize {
                return Err(Error::Gravity(format!(
                    "resident matvec {name}: x_len {x_len} != cols {}",
                    params.cols
                )));
            }
            crate::cost_ledger::record_active_bytes_for(name, codebooks.length() + codes.length());
            record_pq_matvec_ops(*params);
            let tcb = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
            const TG: u32 = 256;
            let n_tg = params.rows.div_ceil(8);
            let params = *params;
            let cb = codebooks.clone();
            let co = codes.clone();
            tcb.dispatch_threads("gravity_pq_matvec", (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
                enc.set_buffer(0, Some(&cb), 0);
                enc.set_buffer(1, Some(&co), 0);
                enc.set_buffer(2, Some(x), 0);
                enc.set_buffer(3, Some(y), 0);
                enc.set_bytes(
                    4,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
            })?;
            Ok(())
        }
    }
}

fn rmsnorm_into(x: &Buffer, x_len: usize, weight: &[f32], eps: f32, out: &Buffer) {
    let _norm = crate::cost_ledger::Scope::new(crate::cost_ledger::Bucket::Norm);
    crate::cost_ledger::record_source_modelled_operations((4 * x_len) as u64, 0, 0, 1, 0);
    let xv = read_f32(x, x_len);
    let mean_sq = xv.iter().map(|v| v * v).sum::<f32>() / x_len as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    let y: Vec<f32> = xv.iter().zip(weight).map(|(v, w)| v * inv * *w).collect();
    write_f32(out, &y);
}

fn residual_add(x: &Buffer, add: &Buffer, n: usize) {
    let _state = crate::cost_ledger::Scope::new(crate::cost_ledger::Bucket::ResidualAndState);
    crate::cost_ledger::record_source_modelled_operations(n as u64, 0, 0, 0, 0);
    let mut xv = read_f32(x, n);
    let av = read_f32(add, n);
    for (a, b) in xv.iter_mut().zip(&av) {
        *a += *b;
    }
    write_f32(x, &xv);
}

enum ResidentDsaSelection {
    Host(Vec<usize>),
    Device { len: usize },
}

impl ResidentDsaSelection {
    fn len(&self) -> usize {
        match self {
            Self::Host(indices) => indices.len(),
            Self::Device { len } => *len,
        }
    }

    fn host_indices(&self) -> Option<&[usize]> {
        match self {
            Self::Host(indices) => Some(indices),
            Self::Device { .. } => None,
        }
    }
}

/// One generation step (or prefill) with decode state on device.
pub fn forward_resident(
    weights: &GpuWeightCache,
    arch: &GlmArch,
    session: &mut ResidentSession,
    pool: &ActPool,
    tokens: &[u32],
    start_pos: usize,
) -> Result<(Vec<f32>, GlmTrace, u64)> {
    use crate::cost_ledger::{self, Bucket};

    if tokens.is_empty() {
        return Err(Error::Gravity("forward_resident: no tokens".into()));
    }
    let ctx = &weights.ctx;
    let a = arch;
    let qk = a.qk_dim();
    let required_sequence = start_pos.checked_add(tokens.len()).ok_or_else(|| {
        Error::Gravity(format!(
            "forward_resident: sequence length overflow ({start_pos} + {})",
            tokens.len()
        ))
    })?;
    {
        let _kv = cost_ledger::Scope::new(Bucket::KvUpdate);
        session.reserve(ctx, arch, required_sequence)?;
    }
    let waits_before = session.waits.get();
    let mut logits = Vec::new();
    let mut trace = GlmTrace::default();

    for (step, &token) in tokens.iter().enumerate() {
        let pos = start_pos + step;
        if token as usize >= a.vocab_size {
            return Err(Error::Gravity(format!(
                "token {token} out of range for vocab_size {}",
                a.vocab_size
            )));
        }

        {
            let _embedding = cost_ledger::Scope::new(Bucket::EmbeddingAndPosition);
            let emb = weights.row("model.embed_tokens.weight", token as usize, a.hidden)?;
            write_f32(&pool.x, &emb);
        }
        let (cos, sin) = {
            let _position = cost_ledger::Scope::new(Bucket::EmbeddingAndPosition);
            rope_cos_sin(arch, pos)
        };
        let device_dsa = session.dsa.device_selection_enabled();
        let mut shared_topk = session.dsa.shared_topk.clone();
        trace.expert_choices.clear();

        for layer in 0..a.n_layers {
            let p = format!("model.layers.{layer}");
            let attn_p = format!("{p}.self_attn");
            let compact_attention = session.attention.is_compact();
            let mut tcb: Option<TokenCommandBuffer<'_>> = None;

            // Attention + IndexShare: projections, DSA indexer, sparse attend,
            // o_proj residual. Nested metal/norm/kv buckets steal exclusive time.
            let topk = {
                let _attn = cost_ledger::Scope::new(Bucket::AttentionAndIndexShare);

                let input_norm_name = format!("{p}.input_layernorm.weight");
                let q_a_name = format!("{attn_p}.q_a_proj.weight");
                let kv_a_name = format!("{attn_p}.kv_a_proj_with_mqa.weight");
                let q_norm_name = format!("{attn_p}.q_a_layernorm.weight");
                let kv_norm_name = format!("{attn_p}.kv_a_layernorm.weight");
                let q_b_name = format!("{attn_p}.q_b_proj.weight");
                let mut graph_projection_names =
                    vec![q_a_name.as_str(), kv_a_name.as_str(), q_b_name.as_str()];
                let indexer_names = if a.indexer_types[layer] == "full" {
                    let idx = format!("{attn_p}.indexer");
                    Some([
                        format!("{idx}.wq_b.weight"),
                        format!("{idx}.wk.weight"),
                        format!("{idx}.weights_proj.weight"),
                    ])
                } else {
                    None
                };
                if let Some(indexer_names) = &indexer_names {
                    graph_projection_names.extend(indexer_names.iter().map(String::as_str));
                }
                let closed_attention_prelude = device_dsa
                    && compact_attention
                    && matvecs_are_device_encodable(weights, &graph_projection_names)?;

                let k_rot = if closed_attention_prelude {
                    let w_in = weights.dense(&input_norm_name)?;
                    let w_q = weights.dense(&q_norm_name)?;
                    let w_kv = weights.dense(&kv_norm_name)?;
                    if w_in.len() != a.hidden
                        || w_q.len() != a.q_lora_rank
                        || w_kv.len() != a.kv_lora_rank
                    {
                        return Err(Error::Gravity(format!(
                            "device attention prelude norm geometry at layer {layer}: input={} expected={} q={} expected={} kv={} expected={}",
                            w_in.len(),
                            a.hidden,
                            w_q.len(),
                            a.q_lora_rank,
                            w_kv.len(),
                            a.kv_lora_rank
                        )));
                    }
                    let mut compact_guard = pool.ensure_compact_attention_scratch(ctx, a)?;
                    let compact_scratch = compact_guard
                        .as_mut()
                        .expect("compact attention scratch initialized");
                    let mut prelude_guard = pool.ensure_device_attention_prelude_scratch(ctx, a)?;
                    let prelude = prelude_guard
                        .as_mut()
                        .expect("device attention prelude scratch initialized");
                    write_f32(&prelude.input_norm_weight, &w_in);
                    write_f32(&prelude.q_norm_weight, &w_q);
                    write_f32(&prelude.kv_norm_weight, &w_kv);
                    write_f32(&prelude.cos, &cos);
                    write_f32(&prelude.sin, &sin);
                    cost_ledger::record_source_modelled_operations(
                        (4 * (a.hidden + a.q_lora_rank + a.kv_lora_rank)) as u64,
                        0,
                        0,
                        3,
                        0,
                    );

                    {
                        let wave = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
                        route_segment_primitives::encode_rmsnorm(
                            wave,
                            &pool.x,
                            &prelude.input_norm_weight,
                            &pool.h,
                            a.hidden,
                            a.rms_norm_eps,
                        )?;
                    }
                    matvec_into(
                        &mut tcb, ctx, weights, &q_a_name, &pool.h, a.hidden, &pool.q_a,
                    )?;
                    matvec_into(
                        &mut tcb,
                        ctx,
                        weights,
                        &kv_a_name,
                        &pool.h,
                        a.hidden,
                        &pool.compressed,
                    )?;
                    {
                        let wave = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
                        route_segment_primitives::encode_rmsnorm(
                            wave,
                            &pool.q_a,
                            &prelude.q_norm_weight,
                            &pool.q_resid,
                            a.q_lora_rank,
                            a.rms_norm_eps,
                        )?;
                        route_segment_primitives::encode_rmsnorm(
                            wave,
                            &pool.compressed,
                            &prelude.kv_norm_weight,
                            &pool.k_latent,
                            a.kv_lora_rank,
                            a.rms_norm_eps,
                        )?;
                        route_segment_primitives::encode_rope_interleaved(
                            wave,
                            &pool.compressed,
                            a.kv_lora_rank,
                            &compact_scratch.key_rope,
                            0,
                            &prelude.cos,
                            &prelude.sin,
                            1,
                            a.qk_rope_head_dim,
                            a.qk_rope_head_dim,
                            a.qk_rope_head_dim,
                        )?;
                    }
                    matvec_into(
                        &mut tcb,
                        ctx,
                        weights,
                        &q_b_name,
                        &pool.q_resid,
                        a.q_lora_rank,
                        &pool.q,
                    )?;
                    {
                        let wave = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
                        route_segment_primitives::encode_copy_head_prefix(
                            wave,
                            &pool.q,
                            &compact_scratch.query_nope,
                            a.n_heads,
                            a.qk_nope_head_dim,
                            a.qk_rope_head_dim,
                        )?;
                        route_segment_primitives::encode_rope_interleaved(
                            wave,
                            &pool.q,
                            a.qk_nope_head_dim,
                            &compact_scratch.query_rope,
                            0,
                            &prelude.cos,
                            &prelude.sin,
                            a.n_heads,
                            a.qk_rope_head_dim,
                            qk,
                            a.qk_rope_head_dim,
                        )?;
                    }
                    Vec::new()
                } else {
                    let w_in = weights.dense(&input_norm_name)?;
                    rmsnorm_into(&pool.x, a.hidden, &w_in, a.rms_norm_eps, &pool.h);

                    // Q path
                    matvec_into(
                        &mut tcb, ctx, weights, &q_a_name, &pool.h, a.hidden, &pool.q_a,
                    )?;
                    // KV-a is independent of q_a — co-issue before the wait.
                    matvec_into(
                        &mut tcb,
                        ctx,
                        weights,
                        &kv_a_name,
                        &pool.h,
                        a.hidden,
                        &pool.compressed,
                    )?;
                    commit(tcb.take(), &session.waits)?;

                    let w_q = weights.dense(&q_norm_name)?;
                    rmsnorm_into(
                        &pool.q_a,
                        a.q_lora_rank,
                        &w_q,
                        a.rms_norm_eps,
                        &pool.q_resid,
                    );

                    let compressed =
                        read_f32(&pool.compressed, a.kv_lora_rank + a.qk_rope_head_dim);
                    let w_kv = weights.dense(&kv_norm_name)?;
                    let k_latent = {
                        let _norm = cost_ledger::Scope::new(Bucket::Norm);
                        let x = &compressed[..a.kv_lora_rank];
                        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
                        let inv = 1.0 / (mean_sq + a.rms_norm_eps).sqrt();
                        x.iter()
                            .zip(&w_kv)
                            .map(|(v, w)| v * inv * w)
                            .collect::<Vec<_>>()
                    };
                    write_f32(&pool.k_latent, &k_latent);
                    let k_rot = rope_interleaved(&compressed[a.kv_lora_rank..], &cos, &sin);

                    matvec_into(
                        &mut tcb,
                        ctx,
                        weights,
                        &q_b_name,
                        &pool.q_resid,
                        a.q_lora_rank,
                        &pool.q,
                    )?;
                    if !compact_attention {
                        matvec_into(
                            &mut tcb,
                            ctx,
                            weights,
                            &format!("{attn_p}.kv_b_proj.weight"),
                            &pool.k_latent,
                            a.kv_lora_rank,
                            &pool.kv,
                        )?;
                    }
                    commit(tcb.take(), &session.waits)?;
                    k_rot
                };

                // Expanded path: materialize and append per-head K/V. Compact
                // mode postpones its latent/RoPE append into the five-dispatch
                // DAG after the stable DSA rank is available.
                if !compact_attention {
                    let _kv = cost_ledger::Scope::new(Bucket::KvUpdate);
                    let kv = read_f32(&pool.kv, a.n_heads * (a.qk_nope_head_dim + a.v_head_dim));
                    let cache = session.attention.expanded_layer(layer)?;
                    let per = a.qk_nope_head_dim + a.v_head_dim;
                    let mut keys_pos = Vec::with_capacity(a.n_heads * qk);
                    let mut vals_pos = Vec::with_capacity(a.n_heads * a.v_head_dim);
                    for head in 0..a.n_heads {
                        let src = &kv[head * per..(head + 1) * per];
                        keys_pos.extend_from_slice(&src[..a.qk_nope_head_dim]);
                        keys_pos.extend_from_slice(&k_rot);
                        vals_pos.extend_from_slice(&src[a.qk_nope_head_dim..]);
                    }
                    let k_off = pos * a.n_heads * qk;
                    let v_off = pos * a.n_heads * a.v_head_dim;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            keys_pos.as_ptr(),
                            (cache.keys.contents() as *mut f32).add(k_off),
                            keys_pos.len(),
                        );
                        std::ptr::copy_nonoverlapping(
                            vals_pos.as_ptr(),
                            (cache.values.contents() as *mut f32).add(v_off),
                            vals_pos.len(),
                        );
                    }
                }

                // Expanded path query layout. Compact mode packs only the
                // content and rotated-RoPE components into lazy scratch.
                if !compact_attention {
                    let q = read_f32(&pool.q, a.n_heads * qk);
                    let mut queries = vec![0f32; a.n_heads * qk];
                    cost_ledger::record_allocation((queries.len() * 4) as u64);
                    for head in 0..a.n_heads {
                        let src = &q[head * qk..(head + 1) * qk];
                        let dst = &mut queries[head * qk..(head + 1) * qk];
                        dst[..a.qk_nope_head_dim].copy_from_slice(&src[..a.qk_nope_head_dim]);
                        dst[a.qk_nope_head_dim..].copy_from_slice(&rope_interleaved(
                            &src[a.qk_nope_head_dim..],
                            &cos,
                            &sin,
                        ));
                    }
                    write_f32(&pool.queries, &queries);
                }

                let topk = match a.indexer_types[layer].as_str() {
                    "full" => {
                        if device_dsa {
                            let len = indexer_topk_device(
                                weights,
                                arch,
                                &attn_p,
                                pool,
                                &session.dsa.index_keys[layer],
                                &session.dsa.sequence_scratch.index_scores_device,
                                session.dsa.ranked_indices()?,
                                session.dsa.capacity,
                                pos,
                                &cos,
                                &sin,
                                &mut tcb,
                                ctx,
                            )?;
                            ResidentDsaSelection::Device { len }
                        } else {
                            let index_keys = &session.dsa.index_keys[layer];
                            let scratch = &mut session.dsa.sequence_scratch;
                            let t = indexer_topk(
                                weights,
                                arch,
                                &attn_p,
                                pool,
                                index_keys,
                                session.dsa.capacity,
                                scratch,
                                pos,
                                &cos,
                                &sin,
                                &mut tcb,
                                ctx,
                                &session.waits,
                            )?;
                            shared_topk = Some(t.clone());
                            session.dsa.shared_topk = Some(t.clone());
                            ResidentDsaSelection::Host(t)
                        }
                    }
                    "shared" => {
                        if device_dsa {
                            let n_keys = active_sequence_len(
                                pos,
                                session.dsa.capacity,
                                "resident shared device DSA index cache",
                            )?;
                            ResidentDsaSelection::Device {
                                len: a.index_topk.min(n_keys),
                            }
                        } else {
                            ResidentDsaSelection::Host(shared_topk.clone().ok_or_else(|| {
                                Error::Gravity(format!(
                                    "layer {layer} shares an index but no earlier layer computed one"
                                ))
                            })?)
                        }
                    }
                    other => {
                        return Err(Error::Gravity(format!(
                            "layer {layer}: unknown indexer type {other:?}"
                        )))
                    }
                };

                if compact_attention {
                    compact_attend_into(
                        weights,
                        a,
                        &session.attention,
                        &session.dsa,
                        pool,
                        layer,
                        pos,
                        &topk,
                        &k_rot,
                        &cos,
                        &sin,
                        closed_attention_prelude,
                        &mut tcb,
                        ctx,
                        &session.waits,
                    )?;
                } else {
                    // Sparse attend over expanded device-resident K/V.
                    let cache = session.attention.expanded_layer(layer)?;
                    let scratch = &mut session.dsa.sequence_scratch;
                    let host_topk = topk.host_indices().ok_or_else(|| {
                        Error::Gravity(
                            "device DSA reached expanded attention without compact admission"
                                .into(),
                        )
                    })?;
                    let context = sparse_attend(
                        a,
                        pool,
                        cache,
                        session.attention.capacity(),
                        scratch,
                        pos,
                        host_topk,
                        qk,
                    )?;
                    write_f32(&pool.context, &context);

                    matvec_into(
                        &mut tcb,
                        ctx,
                        weights,
                        &format!("{attn_p}.o_proj.weight"),
                        &pool.context,
                        a.n_heads * a.v_head_dim,
                        &pool.o,
                    )?;
                    commit(tcb.take(), &session.waits)?;
                }
                if closed_attention_prelude {
                    let _state = cost_ledger::Scope::new(Bucket::ResidualAndState);
                    cost_ledger::record_source_modelled_operations(a.hidden as u64, 0, 0, 0, 0);
                    let wave = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
                    route_segment_primitives::encode_residual_add_inplace(
                        wave, &pool.x, &pool.o, a.hidden,
                    )?;
                    commit(tcb.take(), &session.waits)?;
                } else {
                    residual_add(&pool.x, &pool.o, a.hidden);
                }
                topk
            };

            let w_post = weights.dense(&format!("{p}.post_attention_layernorm.weight"))?;
            rmsnorm_into(&pool.x, a.hidden, &w_post, a.rms_norm_eps, &pool.h);

            match a.mlp_layer_types[layer].as_str() {
                "dense" => {
                    let _dense = cost_ledger::Scope::new(Bucket::DenseExperts);
                    let prefix = format!("{p}.mlp");
                    let out = mlp_one(
                        weights,
                        &prefix,
                        &pool.h,
                        &pool.x,
                        a.hidden,
                        pool,
                        &mut tcb,
                        ctx,
                        &session.waits,
                    )?;
                    if let MlpWaveResult::Host(out) = out {
                        write_f32(&pool.o, &out);
                        residual_add(&pool.x, &pool.o, a.hidden);
                    }
                }
                "sparse" => {
                    let prefix = format!("{p}.mlp");
                    let device_router = gpu_device_router_enabled();
                    // Router gate plus optional exact device noaux_tc selection.
                    {
                        let _route = cost_ledger::Scope::new(Bucket::Routing);
                        matvec_into(
                            &mut tcb,
                            ctx,
                            weights,
                            &format!("{prefix}.gate.weight"),
                            &pool.h,
                            a.hidden,
                            &pool.router_logits,
                        )?;
                        if device_router {
                            let bias =
                                weights.dense(&format!("{prefix}.gate.e_score_correction_bias"))?;
                            if bias.len() != a.n_routed_experts {
                                return Err(Error::Gravity(format!(
                                    "device router bias at layer {layer}: {} values, expected {}",
                                    bias.len(),
                                    a.n_routed_experts
                                )));
                            }
                            write_f32(&pool.router_bias, &bias);
                            cost_ledger::record_transfer(
                                (bias.len() * std::mem::size_of::<f32>()) as u64,
                                true,
                                "router_bias_upload",
                            );
                            cost_ledger::record_source_modelled_operations(
                                (4 * a.n_routed_experts
                                    + 2 * a.n_group
                                    + a.num_experts_per_tok * a.n_routed_experts)
                                    as u64,
                                0,
                                0,
                                a.n_routed_experts as u64,
                                0,
                            );
                            let wave = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
                            route_segment_primitives::encode_router_select_noaux(
                                wave,
                                &pool.router_logits,
                                &pool.router_bias,
                                &pool.router_scores,
                                &pool.router_corrected,
                                &pool.expert_idx,
                                &pool.expert_w,
                                &pool.expert_exec_slots,
                                a.n_routed_experts,
                                a.n_group,
                                a.topk_group,
                                a.num_experts_per_tok,
                                a.norm_topk_prob,
                                a.routed_scaling_factor,
                            )?;
                        }
                        commit(tcb.take(), &session.waits)?;
                    }

                    let (indices, moe_weights) = if device_router {
                        let indices = read_u32(&pool.expert_idx, a.num_experts_per_tok)
                            .into_iter()
                            .map(|index| index as usize)
                            .collect::<Vec<_>>();
                        if let Some(index) = indices
                            .iter()
                            .copied()
                            .find(|&index| index >= a.n_routed_experts)
                        {
                            return Err(Error::Gravity(format!(
                                "device router returned expert {index}, but layer {layer} has {} experts",
                                a.n_routed_experts
                            )));
                        }
                        let moe_weights = read_f32(&pool.expert_w, a.num_experts_per_tok);
                        cost_ledger::record_transfer(
                            (a.num_experts_per_tok
                                * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>()))
                                as u64,
                            false,
                            "router_selection_download",
                        );
                        (indices, moe_weights)
                    } else {
                        router_select(weights, a, &prefix, pool)?
                    };
                    // Residency: expert selection + weights live on device.
                    if !device_router {
                        let _route_state = cost_ledger::Scope::new(Bucket::Routing);
                        let idx_u: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                idx_u.as_ptr(),
                                pool.expert_idx.contents() as *mut u32,
                                idx_u.len(),
                            );
                        }
                        write_f32(&pool.expert_w, &moe_weights);
                    }
                    trace.expert_choices.push(indices.clone());

                    // Ascending expert order (float-add associativity), then
                    // shared last — same as host `routed_moe` / `batched_mlp`.
                    let mut order: Vec<usize> = (0..indices.len()).collect();
                    order.sort_by_key(|&s| indices[s]);
                    let prefixes: Vec<String> = order
                        .iter()
                        .map(|&slot| format!("{prefix}.experts.{}", indices[slot]))
                        .chain(std::iter::once(format!("{prefix}.shared_experts")))
                        .collect();
                    // Expert-wave (flagged, default off): one CB for gate/up/SiLU/
                    // down/weighted combine. Default three-batch path is unchanged.
                    // RoutedExperts owns co-batch CPU glue; metal_* steals GPU waits.
                    let routed = {
                        let _routed = cost_ledger::Scope::new(Bucket::RoutedExperts);
                        if gpu_expert_wave_enabled() {
                            let scales: Vec<f32> = order
                                .iter()
                                .map(|&slot| moe_weights[slot])
                                .chain(std::iter::once(1.0f32))
                                .collect();
                            moe_device_wave(
                                weights,
                                &prefixes,
                                &scales,
                                &pool.h,
                                &pool.x,
                                a.hidden,
                                pool,
                                &mut tcb,
                                ctx,
                                &session.waits,
                            )?
                        } else {
                            let mut outs = batched_mlp(
                                weights,
                                &prefixes,
                                &pool.h,
                                a.hidden,
                                pool,
                                &mut tcb,
                                ctx,
                                &session.waits,
                            )?;
                            let shared = outs.pop().expect("shared last");
                            let mut routed = {
                                let _r = cost_ledger::Scope::new(Bucket::RoutedExperts);
                                let mut routed = vec![0f32; a.hidden];
                                cost_ledger::record_allocation((routed.len() * 4) as u64);
                                cost_ledger::record_source_modelled_operations(
                                    (2usize
                                        .saturating_mul(routed.len())
                                        .saturating_mul(outs.len()))
                                        as u64,
                                    0,
                                    0,
                                    0,
                                    0,
                                );
                                for (out, &slot) in outs.iter().zip(&order) {
                                    for (r, o) in routed.iter_mut().zip(out) {
                                        *r += o * moe_weights[slot];
                                    }
                                }
                                routed
                            };
                            {
                                let _shared = cost_ledger::Scope::new(Bucket::SharedExperts);
                                cost_ledger::record_source_modelled_operations(
                                    routed.len() as u64,
                                    0,
                                    0,
                                    0,
                                    0,
                                );
                                for (r, s) in routed.iter_mut().zip(&shared) {
                                    *r += *s;
                                }
                            }
                            MlpWaveResult::Host(routed)
                        }
                    };
                    if let MlpWaveResult::Host(routed) = routed {
                        write_f32(&pool.o, &routed);
                        residual_add(&pool.x, &pool.o, a.hidden);
                    }
                }
                other => {
                    return Err(Error::Gravity(format!(
                        "layer {layer}: unknown MLP type {other:?}"
                    )))
                }
            }

            if layer + 1 == a.n_layers {
                trace.final_topk = match topk {
                    ResidentDsaSelection::Host(indices) => indices,
                    ResidentDsaSelection::Device { len } => {
                        read_u32(session.dsa.ranked_indices()?, len)
                            .into_iter()
                            .map(|index| index as usize)
                            .collect()
                    }
                };
            }
        }

        // lm_head once per token. A device head appends final RMSNorm, logits,
        // greedy argmax, and diagnostic top-k into one command buffer. The
        // flagship native.bf16 path is selected whenever that tensor is device
        // resident; default-off GPU_LM_HEAD also permits a PQ head so bounded
        // complete-token fixtures exercise the same final graph.
        let waits_before_head = session.waits.get();
        {
            let _head = crate::cost_ledger::Scope::new(crate::cost_ledger::Bucket::FinalHead);
            enum DeviceHead {
                NativeBf16 {
                    weight: Buffer,
                    rows: u32,
                    cols: u32,
                },
                Pq {
                    codebooks: Buffer,
                    codes: Buffer,
                    params: crate::gravity_glm::gpu::PqParams,
                },
            }

            let mut cache = weights.cache.lock().expect("gpu weight cache");
            weights.ensure_many_locked(&mut cache, &["lm_head.weight"])?;
            let device_head = match cache.get("lm_head.weight").expect("ensured lm_head") {
                GpuTensor::NativeGpuBf16 { buf, rows, cols } => {
                    if a.hidden != *cols as usize {
                        return Err(Error::Gravity(format!(
                            "lm_head device path: hidden {} != cols {cols}",
                            a.hidden
                        )));
                    }
                    if a.vocab_size != *rows as usize {
                        return Err(Error::Gravity(format!(
                            "lm_head device path: vocab {} != rows {rows}",
                            a.vocab_size
                        )));
                    }
                    crate::cost_ledger::record_matvec_call();
                    crate::cost_ledger::record_active_bytes_for("lm_head.weight", buf.length());
                    crate::cost_ledger::record_source_modelled_operations(
                        2u64.saturating_mul(*rows as u64)
                            .saturating_mul(*cols as u64),
                        0,
                        0,
                        0,
                        2u64.saturating_mul(*rows as u64)
                            .saturating_mul(*cols as u64),
                    );
                    Some(DeviceHead::NativeBf16 {
                        weight: buf.clone(),
                        rows: *rows,
                        cols: *cols,
                    })
                }
                GpuTensor::Pq {
                    codebooks,
                    codes,
                    params,
                } if gpu_lm_head_enabled() => {
                    if a.hidden != params.cols as usize {
                        return Err(Error::Gravity(format!(
                            "lm_head device PQ path: hidden {} != cols {}",
                            a.hidden, params.cols
                        )));
                    }
                    if a.vocab_size != params.rows as usize {
                        return Err(Error::Gravity(format!(
                            "lm_head device PQ path: vocab {} != rows {}",
                            a.vocab_size, params.rows
                        )));
                    }
                    crate::cost_ledger::record_matvec_call();
                    crate::cost_ledger::record_active_bytes_for(
                        "lm_head.weight",
                        codebooks.length() + codes.length(),
                    );
                    record_pq_matvec_ops(*params);
                    Some(DeviceHead::Pq {
                        codebooks: codebooks.clone(),
                        codes: codes.clone(),
                        params: *params,
                    })
                }
                GpuTensor::NativeCpu(_) | GpuTensor::Pq { .. } => None,
            };
            drop(cache);

            let w_norm = weights.dense("model.norm.weight")?;
            if w_norm.len() != a.hidden {
                return Err(Error::Gravity(format!(
                    "final RMSNorm weight has {} values, expected {}",
                    w_norm.len(),
                    a.hidden
                )));
            }
            if let Some(device_head) = device_head {
                write_f32(&pool.final_norm_weight, &w_norm);
                cost_ledger::record_transfer(
                    (w_norm.len() * std::mem::size_of::<f32>()) as u64,
                    true,
                    "final_norm_weight_upload",
                );
                let mut tcb = TokenCommandBuffer::new(ctx);
                {
                    let _norm = cost_ledger::Scope::new(Bucket::Norm);
                    cost_ledger::record_source_modelled_operations(
                        (4 * a.hidden) as u64,
                        0,
                        0,
                        1,
                        0,
                    );
                    route_segment_primitives::encode_rmsnorm(
                        &mut tcb,
                        &pool.x,
                        &pool.final_norm_weight,
                        &pool.final_hidden,
                        a.hidden,
                        a.rms_norm_eps,
                    )?;
                }
                let rows = match device_head {
                    DeviceHead::NativeBf16 { weight, rows, cols } => {
                        encode_gemv_native_bf16_seq(
                            &mut tcb,
                            &weight,
                            rows,
                            cols,
                            &pool.final_hidden,
                            &pool.logits,
                        )?;
                        rows
                    }
                    DeviceHead::Pq {
                        codebooks,
                        codes,
                        params,
                    } => {
                        encode_pq_matvec_device(
                            &mut tcb,
                            &codebooks,
                            &codes,
                            params,
                            &pool.final_hidden,
                            &pool.logits,
                        )?;
                        params.rows
                    }
                };
                {
                    let _sampling = cost_ledger::Scope::new(cost_ledger::Bucket::Sampling);
                    encode_argmax_f32(&mut tcb, &pool.logits, rows, &pool.sample_token)?;
                    encode_sample_topk_f32(
                        &mut tcb,
                        &pool.logits,
                        rows,
                        GPU_LM_HEAD_DIAG_TOPK,
                        &pool.head_topk_idx,
                        &pool.head_topk_val,
                    )?;
                    let rounds = GPU_LM_HEAD_DIAG_TOPK as u64 + 1;
                    cost_ledger::record_source_modelled_operations(
                        0,
                        0,
                        rounds
                            .saturating_mul(rows as u64)
                            .saturating_add(rounds.saturating_mul(255)),
                        0,
                        0,
                    );
                }
                tcb.commit_and_wait()?;
                session.waits.set(session.waits.get().saturating_add(1));

                {
                    let _sampling = cost_ledger::Scope::new(cost_ledger::Bucket::Sampling);
                    let tok = read_u32(&pool.sample_token, 1)[0];
                    let k = GPU_LM_HEAD_DIAG_TOPK as usize;
                    let topk_idx = read_u32(&pool.head_topk_idx, k);
                    let topk_val = read_f32(&pool.head_topk_val, k);
                    crate::cost_ledger::record_transfer(
                        (4 + k * 4 + k * 4) as u64,
                        false,
                        "lm_head_token_diag_download",
                    );
                    trace.sample_token = Some(tok);
                    trace.head_topk_idx = topk_idx;
                    trace.head_topk_val = topk_val;
                }

                if gpu_lm_head_full_logits_enabled() {
                    logits = read_f32(&pool.logits, a.vocab_size);
                    crate::cost_ledger::record_transfer(
                        (a.vocab_size * 4) as u64,
                        false,
                        "lm_head_y_download",
                    );
                    trace.head_full_logits_readback = true;
                } else {
                    logits = Vec::new();
                    trace.head_full_logits_readback = false;
                }
            } else {
                rmsnorm_into(
                    &pool.x,
                    a.hidden,
                    &w_norm,
                    a.rms_norm_eps,
                    &pool.final_hidden,
                );
                let hidden = read_f32(&pool.final_hidden, a.hidden);
                logits = weights.matvec("lm_head.weight", &hidden)?;
                trace.head_full_logits_readback = true;
                if session.waits.get() == waits_before_head {
                    let mut cache = weights.cache.lock().expect("gpu weight cache");
                    weights.ensure_many_locked(&mut cache, &["lm_head.weight"])?;
                    if matches!(cache.get("lm_head.weight"), Some(GpuTensor::Pq { .. })) {
                        session.waits.set(session.waits.get().saturating_add(1));
                    }
                }
            }
        }
    }

    session.seq_len = required_sequence;
    let waits = session.waits.get().saturating_sub(waits_before);
    Ok((logits, trace, waits))
}

fn router_select(
    weights: &GpuWeightCache,
    a: &GlmArch,
    prefix: &str,
    pool: &ActPool,
) -> Result<(Vec<usize>, Vec<f32>)> {
    // Host-side noaux_tc arithmetic after the gate matvec. Nested under any
    // open parent; when none is open this is the exclusive Routing line.
    let _route = crate::cost_ledger::Scope::new(crate::cost_ledger::Bucket::Routing);
    let logits = read_f32(&pool.router_logits, a.n_routed_experts);
    let scores: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
    crate::cost_ledger::record_source_modelled_operations(
        (3 * a.n_routed_experts) as u64,
        0,
        0,
        a.n_routed_experts as u64,
        0,
    );
    crate::cost_ledger::record_allocation((scores.len() * 4) as u64);
    write_f32(&pool.router_scores, &scores);
    let bias = weights.dense(&format!("{prefix}.gate.e_score_correction_bias"))?;
    let corrected: Vec<f32> = scores.iter().zip(&bias).map(|(s, b)| s + b).collect();
    crate::cost_ledger::record_source_modelled_operations(corrected.len() as u64, 0, 0, 0, 0);
    crate::cost_ledger::record_allocation((corrected.len() * 4) as u64);
    write_f32(&pool.router_corrected, &corrected);
    let per_group = a.n_routed_experts / a.n_group;
    let group_scores: Vec<f32> = (0..a.n_group)
        .map(|g| {
            let slice = &corrected[g * per_group..(g + 1) * per_group];
            topk_desc(slice, 2.min(per_group))
                .iter()
                .map(|&i| slice[i])
                .sum()
        })
        .collect();
    let chosen = topk_desc(&group_scores, a.topk_group);
    let mut choice = vec![f32::NEG_INFINITY; a.n_routed_experts];
    for &g in &chosen {
        for e in g * per_group..(g + 1) * per_group {
            choice[e] = corrected[e];
        }
    }
    let indices = topk_desc(&choice, a.num_experts_per_tok);
    let mut weights_out: Vec<f32> = indices.iter().map(|&i| scores[i]).collect();
    crate::cost_ledger::record_allocation(
        ((group_scores.len() + choice.len() + weights_out.len()) * 4) as u64,
    );
    if a.norm_topk_prob {
        let total: f32 = weights_out.iter().sum::<f32>() + 1e-20;
        crate::cost_ledger::record_source_modelled_operations(
            (2 * weights_out.len() + 1) as u64,
            0,
            0,
            0,
            0,
        );
        for w in weights_out.iter_mut() {
            *w /= total;
        }
    }
    crate::cost_ledger::record_source_modelled_operations(weights_out.len() as u64, 0, 0, 0, 0);
    for w in weights_out.iter_mut() {
        *w *= a.routed_scaling_factor;
    }
    Ok((indices, weights_out))
}

fn sparse_attend(
    a: &GlmArch,
    pool: &ActPool,
    cache: &LayerGpuCache,
    cache_capacity: usize,
    scratch: &mut SequenceScratch,
    pos: usize,
    topk: &[usize],
    qk: usize,
) -> Result<Vec<f32>> {
    let n_keys = active_sequence_len(pos, cache_capacity, "resident attention cache")?;
    let scratch_len = scratch.active_len(pos)?;
    if scratch_len != n_keys {
        return Err(Error::Gravity(format!(
            "resident sparse attention capacity mismatch: cache={n_keys}, scratch={scratch_len}"
        )));
    }
    let keys = unsafe {
        std::slice::from_raw_parts(cache.keys.contents() as *const f32, n_keys * a.n_heads * qk)
    };
    let values = unsafe {
        std::slice::from_raw_parts(
            cache.values.contents() as *const f32,
            n_keys * a.n_heads * a.v_head_dim,
        )
    };
    let queries = read_f32(&pool.queries, a.n_heads * qk);
    let HostSequenceScratch {
        attention_allowed,
        attention_scores,
        ..
    } = &mut scratch.host;
    let allow = &mut attention_allowed[..n_keys];
    allow.fill(0);
    for &t in topk {
        if t <= pos && t < n_keys {
            allow[t] = 1;
        }
    }
    let selected = allow.iter().filter(|&&v| v != 0).count() as u64;
    let heads = a.n_heads as u64;
    let per_selected_fp = (2 * qk + 4 + 2 * a.v_head_dim) as u64;
    crate::cost_ledger::record_source_modelled_operations(
        heads
            .saturating_mul(selected)
            .saturating_mul(per_selected_fp),
        0,
        heads.saturating_mul(n_keys as u64),
        heads.saturating_mul(selected),
        0,
    );
    let scale = (qk as f32).powf(-0.5);
    let mut context = vec![0f32; a.n_heads * a.v_head_dim];
    let scores = &mut attention_scores[..n_keys];
    scores.fill(f32::NEG_INFINITY);
    crate::cost_ledger::record_allocation((context.len() * 4) as u64);
    for head in 0..a.n_heads {
        let qh = &queries[head * qk..(head + 1) * qk];
        let mut best = f32::NEG_INFINITY;
        for t in 0..n_keys {
            if allow[t] == 0 {
                scores[t] = f32::NEG_INFINITY;
                continue;
            }
            let off = (t * a.n_heads + head) * qk;
            let dot: f32 = qh
                .iter()
                .zip(&keys[off..off + qk])
                .map(|(x, y)| x * y)
                .sum();
            scores[t] = dot * scale;
            best = best.max(scores[t]);
        }
        let mut total = 0f32;
        for s in scores.iter_mut() {
            *s = if s.is_finite() {
                (*s - best).exp()
            } else {
                0.0
            };
            total += *s;
        }
        let out = &mut context[head * a.v_head_dim..(head + 1) * a.v_head_dim];
        for (t, &prob) in scores.iter().enumerate() {
            if prob == 0.0 {
                continue;
            }
            let w = prob / total;
            let off = (t * a.n_heads + head) * a.v_head_dim;
            for (o, v) in out.iter_mut().zip(&values[off..off + a.v_head_dim]) {
                *o += w * v;
            }
        }
    }
    Ok(context)
}

#[allow(clippy::too_many_arguments)]
fn compact_attend_into<'a>(
    weights: &GpuWeightCache,
    a: &GlmArch,
    attention: &ResidentAttentionState,
    dsa: &DsaIndexState,
    pool: &ActPool,
    layer: usize,
    pos: usize,
    topk: &ResidentDsaSelection,
    k_rot: &[f32],
    cos: &[f32],
    sin: &[f32],
    device_inputs_ready: bool,
    pending: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<()> {
    let cache = attention.compact_layer(layer)?;
    let n_keys = active_sequence_len(pos, attention.capacity(), "compact MLA attention cache")?;
    let n_allow = topk.len();
    if n_allow > 2048 {
        return Err(Error::Gravity(format!(
            "compact MLA ranked attention supports at most 2048 positions, got {}",
            n_allow
        )));
    }
    if let Some(host_ranked) = topk.host_indices() {
        dsa.store_ranked_indices(host_ranked)?;
    } else if !dsa.device_selection_enabled() {
        return Err(Error::Gravity(
            "compact MLA received device-ranked DSA without device selection state".into(),
        ));
    }

    let mut scratch_guard = pool.ensure_compact_attention_scratch(ctx, a)?;
    let scratch = scratch_guard
        .as_mut()
        .expect("compact attention scratch initialized");
    let qk = a.qk_dim();
    if device_inputs_ready {
        if topk.host_indices().is_some() || !dsa.device_selection_enabled() {
            return Err(Error::Gravity(
                "device compact-attention inputs require device-ranked DSA".into(),
            ));
        }
        if k_rot.len() != 0 {
            return Err(Error::Gravity(
                "device compact-attention inputs unexpectedly carried host key RoPE".into(),
            ));
        }
    } else {
        let query = read_f32(&pool.q, a.n_heads * qk);
        let mut query_nope = vec![0.0f32; a.n_heads * a.qk_nope_head_dim];
        let mut query_rope = vec![0.0f32; a.n_heads * a.qk_rope_head_dim];
        for head in 0..a.n_heads {
            let source = &query[head * qk..(head + 1) * qk];
            let nope_out =
                &mut query_nope[head * a.qk_nope_head_dim..(head + 1) * a.qk_nope_head_dim];
            nope_out.copy_from_slice(&source[..a.qk_nope_head_dim]);
            let rotated = rope_interleaved(&source[a.qk_nope_head_dim..], cos, sin);
            let rope_out =
                &mut query_rope[head * a.qk_rope_head_dim..(head + 1) * a.qk_rope_head_dim];
            rope_out.copy_from_slice(&rotated);
        }
        write_f32(&scratch.query_nope, &query_nope);
        write_f32(&scratch.query_rope, &query_rope);
        write_f32(&scratch.key_rope, k_rot);
    }

    let attn_p = format!("model.layers.{layer}.self_attn");
    let kv_name = format!("{attn_p}.kv_b_proj.weight");
    let o_name = format!("{attn_p}.o_proj.weight");
    let (kv_codebooks, kv_codes, kv_params, o_codebooks, o_codes, o_params) = {
        let mut weight_cache = weights.cache.lock().expect("gpu weight cache");
        weights.ensure_many_locked(&mut weight_cache, &[&kv_name, &o_name])?;
        let (kv_codebooks, kv_codes, kv_params) =
            match weight_cache.get(&kv_name).expect("ensured compact kv_b") {
                tensor @ GpuTensor::Pq {
                    codebooks,
                    codes,
                    params,
                } => {
                    record_routed_tensor_representation(&kv_name, tensor);
                    (codebooks.clone(), codes.clone(), *params)
                }
                _ => {
                    return Err(Error::Gravity(format!(
                        "compact MLA requires PQ tensor {kv_name}"
                    )))
                }
            };
        let (o_codebooks, o_codes, o_params) =
            match weight_cache.get(&o_name).expect("ensured compact o_proj") {
                tensor @ GpuTensor::Pq {
                    codebooks,
                    codes,
                    params,
                } => {
                    record_routed_tensor_representation(&o_name, tensor);
                    (codebooks.clone(), codes.clone(), *params)
                }
                _ => {
                    return Err(Error::Gravity(format!(
                        "compact MLA requires PQ tensor {o_name}"
                    )))
                }
            };
        (
            kv_codebooks,
            kv_codes,
            kv_params,
            o_codebooks,
            o_codes,
            o_params,
        )
    };

    crate::gravity_glm::gpu::validate_compact_mla_layer_params(a, layer, kv_params, o_params)?;
    let row_stride = a
        .qk_nope_head_dim
        .checked_add(a.v_head_dim)
        .ok_or_else(|| Error::Gravity("compact MLA KV row stride overflow".into()))?;

    crate::cost_ledger::record_active_bytes_for(
        &kv_name,
        (kv_codebooks.length() + kv_codes.length()).saturating_mul(2),
    );
    record_pq_matvec_ops(kv_params);
    crate::cost_ledger::record_active_bytes_for(&o_name, o_codebooks.length() + o_codes.length());
    record_pq_matvec_ops(o_params);
    let selected = n_allow as u64;
    let attention_fp = (a.n_heads as u64)
        .saturating_mul(selected)
        .saturating_mul((4 * a.kv_lora_rank + 2 * a.qk_rope_head_dim + 6) as u64);
    crate::cost_ledger::record_source_modelled_operations(
        attention_fp,
        0,
        (a.n_heads as u64).saturating_mul(selected),
        (a.n_heads as u64).saturating_mul(selected),
        0,
    );

    let tcb = pending.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
    let dispatches_before = tcb.dispatch_count();
    route_segment_primitives::encode_mla_append_compact(
        tcb,
        &pool.k_latent,
        &scratch.key_rope,
        &cache.latents,
        &cache.rope_tails,
        a.kv_lora_rank,
        a.qk_rope_head_dim,
        pos,
    )?;
    route_segment_primitives::encode_pq_k_transpose_heads(
        tcb,
        &kv_codebooks,
        &kv_codes,
        &scratch.query_nope,
        &scratch.query_latent,
        a.n_heads,
        a.qk_nope_head_dim,
        row_stride,
        a.kv_lora_rank,
        kv_params.dim as usize,
        kv_params.sub as usize,
        kv_params.card as usize,
        kv_params.bits as usize,
        kv_params.nchunk as usize,
    )?;
    route_segment_primitives::encode_compact_ranked_attention(
        tcb,
        &scratch.query_latent,
        &scratch.query_rope,
        &cache.latents,
        &cache.rope_tails,
        dsa.ranked_indices()?,
        &scratch.query_latent,
        a.n_heads,
        a.kv_lora_rank,
        a.qk_rope_head_dim,
        n_keys,
        n_allow,
        (qk as f32).powf(-0.5),
    )?;
    route_segment_primitives::encode_pq_v_rows_heads(
        tcb,
        &kv_codebooks,
        &kv_codes,
        &scratch.query_latent,
        &pool.context,
        a.n_heads,
        row_stride,
        a.qk_nope_head_dim,
        a.v_head_dim,
        a.kv_lora_rank,
        kv_params.dim as usize,
        kv_params.sub as usize,
        kv_params.card as usize,
        kv_params.bits as usize,
        kv_params.nchunk as usize,
    )?;
    encode_pq_matvec_device(
        tcb,
        &o_codebooks,
        &o_codes,
        o_params,
        &pool.context,
        &pool.o,
    )?;
    let compact_dispatches = tcb.dispatch_count().saturating_sub(dispatches_before);
    if compact_dispatches != 5 {
        return Err(Error::Gravity(format!(
            "compact MLA expected five dispatches, encoded {}",
            compact_dispatches
        )));
    }
    if device_inputs_ready {
        Ok(())
    } else {
        commit(pending.take(), waits)
    }
}

#[allow(clippy::too_many_arguments)]
fn indexer_topk<'a>(
    weights: &GpuWeightCache,
    arch: &GlmArch,
    attn_p: &str,
    pool: &ActPool,
    index_key_buffer: &Buffer,
    cache_capacity: usize,
    scratch: &mut SequenceScratch,
    pos: usize,
    cos: &[f32],
    sin: &[f32],
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<Vec<usize>> {
    let a = arch;
    let (ih, idim, rot) = (a.index_n_heads, a.index_head_dim, a.qk_rope_head_dim);
    let idx = format!("{attn_p}.indexer");

    matvec_into(
        tcb,
        ctx,
        weights,
        &format!("{idx}.wq_b.weight"),
        &pool.q_resid,
        a.q_lora_rank,
        &pool.idx_q,
    )?;
    matvec_into(
        tcb,
        ctx,
        weights,
        &format!("{idx}.wk.weight"),
        &pool.h,
        a.hidden,
        &pool.idx_k_raw,
    )?;
    commit(tcb.take(), waits)?;

    let k_raw = read_f32(&pool.idx_k_raw, idim);
    let kw = weights.dense(&format!("{idx}.k_norm.weight"))?;
    let kb = weights.dense(&format!("{idx}.k_norm.bias"))?;
    let k = {
        let n = k_raw.len() as f32;
        let mean = k_raw.iter().sum::<f32>() / n;
        let var = k_raw.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + 1e-6).sqrt();
        (0..k_raw.len())
            .map(|i| (k_raw[i] - mean) * inv * kw[i] + kb[i])
            .collect::<Vec<_>>()
    };
    let mut k_full = rope_interleaved(&k[..rot], cos, sin);
    k_full.extend_from_slice(&k[rot..]);
    unsafe {
        std::ptr::copy_nonoverlapping(
            k_full.as_ptr(),
            (index_key_buffer.contents() as *mut f32).add(pos * idim),
            idim,
        );
    }

    let q = read_f32(&pool.idx_q, ih * idim);
    let mut q_full = vec![0f32; ih * idim];
    for h in 0..ih {
        let src = &q[h * idim..(h + 1) * idim];
        let rotated = rope_interleaved(&src[..rot], cos, sin);
        q_full[h * idim..h * idim + rot].copy_from_slice(&rotated);
        q_full[h * idim + rot..(h + 1) * idim].copy_from_slice(&src[rot..]);
    }

    matvec_into(
        tcb,
        ctx,
        weights,
        &format!("{idx}.weights_proj.weight"),
        &pool.h,
        a.hidden,
        &pool.idx_head_w,
    )?;
    commit(tcb.take(), waits)?;
    let head_scale = (ih as f32).powf(-0.5);
    let mut head_weights = read_f32(&pool.idx_head_w, ih);
    for w in head_weights.iter_mut() {
        *w *= head_scale;
    }

    let n_keys = active_sequence_len(pos, cache_capacity, "resident index-key cache")?;
    let scratch_len = scratch.active_len(pos)?;
    if scratch_len != n_keys {
        return Err(Error::Gravity(format!(
            "resident indexer capacity mismatch: cache={n_keys}, scratch={scratch_len}"
        )));
    }
    let dim_scale = (idim as f32).powf(-0.5);
    let index_keys = unsafe {
        std::slice::from_raw_parts(index_key_buffer.contents() as *const f32, n_keys * idim)
    };
    crate::cost_ledger::record_source_modelled_operations(
        (n_keys as u64)
            .saturating_mul(ih as u64)
            .saturating_mul((2 * idim + 3) as u64),
        0,
        (n_keys as u64).saturating_mul(ih as u64),
        0,
        0,
    );
    let topk = {
        let HostSequenceScratch {
            index_scores,
            selection_indices,
            ..
        } = &mut scratch.host;
        let index_scores = &mut index_scores[..n_keys];
        index_scores.fill(0.0);
        for (t, score) in index_scores.iter_mut().enumerate() {
            let key = &index_keys[t * idim..(t + 1) * idim];
            let mut acc = 0f32;
            for h in 0..ih {
                let qh = &q_full[h * idim..(h + 1) * idim];
                let dot: f32 = qh.iter().zip(key).map(|(x, y)| x * y).sum();
                acc += head_weights[h] * (dot * dim_scale).max(0.0);
            }
            *score = acc;
        }
        for (t, score) in index_scores.iter_mut().enumerate() {
            if t > pos {
                *score = f32::NEG_INFINITY;
            }
        }
        topk_desc_with_scratch(
            index_scores,
            a.index_topk.min(n_keys),
            &mut selection_indices[..n_keys],
        )?
    };
    scratch.store_index_scores(n_keys)?;
    Ok(topk)
}

/// Default-off device DSA path.
///
/// `wq_b + wk + weights_proj → affine LayerNorm → q/k RoPE assembly → DSA
/// scores → exact radix top-k` stays in the caller's open command buffer.
/// Compact attention appends to the same graph and consumes the ranked u32
/// buffer directly, so no projection, score, or rank readback lies on the
/// attention dependency path.
#[allow(clippy::too_many_arguments)]
fn indexer_topk_device<'a>(
    weights: &GpuWeightCache,
    arch: &GlmArch,
    attn_p: &str,
    pool: &ActPool,
    index_key_buffer: &Buffer,
    score_buffer: &Buffer,
    ranked_indices: &Buffer,
    cache_capacity: usize,
    pos: usize,
    cos: &[f32],
    sin: &[f32],
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
) -> Result<usize> {
    let a = arch;
    let (ih, idim, rot) = (a.index_n_heads, a.index_head_dim, a.qk_rope_head_dim);
    let idx = format!("{attn_p}.indexer");

    matvec_into(
        tcb,
        ctx,
        weights,
        &format!("{idx}.wq_b.weight"),
        &pool.q_resid,
        a.q_lora_rank,
        &pool.idx_q,
    )?;
    matvec_into(
        tcb,
        ctx,
        weights,
        &format!("{idx}.wk.weight"),
        &pool.h,
        a.hidden,
        &pool.idx_k_raw,
    )?;
    let kw = weights.dense(&format!("{idx}.k_norm.weight"))?;
    let kb = weights.dense(&format!("{idx}.k_norm.bias"))?;
    if kw.len() != idim || kb.len() != idim {
        return Err(Error::Gravity(format!(
            "device DSA affine parameters for {idx}: weight={} bias={} expected={idim}",
            kw.len(),
            kb.len()
        )));
    }
    if cos.len() != rot / 2 || sin.len() != rot / 2 {
        return Err(Error::Gravity(format!(
            "device DSA RoPE tables: cos={} sin={} expected={}",
            cos.len(),
            sin.len(),
            rot / 2
        )));
    }
    let mut transform_guard = pool.ensure_device_dsa_transform_scratch(ctx, a)?;
    let transform = transform_guard
        .as_mut()
        .expect("device DSA transform scratch initialized");
    write_f32(&transform.norm_weight, &kw);
    write_f32(&transform.norm_bias, &kb);
    write_f32(&transform.cos, cos);
    write_f32(&transform.sin, sin);

    matvec_into(
        tcb,
        ctx,
        weights,
        &format!("{idx}.weights_proj.weight"),
        &pool.h,
        a.hidden,
        &pool.idx_head_w,
    )?;

    let n_keys = active_sequence_len(pos, cache_capacity, "resident device DSA index cache")?;
    let k = a.index_topk.min(n_keys);
    crate::cost_ledger::record_source_modelled_operations(
        (n_keys as u64)
            .saturating_mul(ih as u64)
            .saturating_mul((2 * idim + 3) as u64),
        0,
        (n_keys as u64).saturating_mul(ih as u64),
        0,
        0,
    );
    let wave = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
    route_segment_primitives::encode_layernorm_affine(
        wave,
        &pool.idx_k_raw,
        &transform.norm_weight,
        &transform.norm_bias,
        &pool.idx_k_raw,
        idim,
        1e-6,
    )?;
    let key_offset = pos.checked_mul(idim).ok_or_else(|| {
        Error::Gravity(format!(
            "device DSA index-key offset overflow: position={pos} dim={idim}"
        ))
    })?;
    route_segment_primitives::encode_rope_prefix_tail(
        wave,
        &pool.idx_k_raw,
        0,
        index_key_buffer,
        key_offset,
        &transform.cos,
        &transform.sin,
        1,
        rot,
        idim,
        idim,
    )?;
    route_segment_primitives::encode_rope_prefix_tail(
        wave,
        &pool.idx_q,
        0,
        &transform.query,
        0,
        &transform.cos,
        &transform.sin,
        ih,
        rot,
        idim,
        idim,
    )?;
    route_segment_primitives::encode_dsa_scores(
        wave,
        &transform.query,
        index_key_buffer,
        &pool.idx_head_w,
        score_buffer,
        n_keys,
        ih,
        idim,
        pos,
        (idim as f32).powf(-0.5),
        (ih as f32).powf(-0.5),
    )?;
    route_segment_primitives::encode_radix_topk(wave, score_buffer, ranked_indices, n_keys, k)?;
    Ok(k)
}

#[allow(clippy::too_many_arguments)]
fn mlp_one<'a>(
    weights: &GpuWeightCache,
    prefix: &str,
    x: &Buffer,
    residual: &Buffer,
    x_len: usize,
    pool: &ActPool,
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<MlpWaveResult> {
    // Expert-wave: one CB for dense MLP. Default path below is unchanged.
    if gpu_expert_wave_enabled() {
        return moe_device_wave(
            weights,
            &[prefix.to_string()],
            &[1.0f32],
            x,
            residual,
            x_len,
            pool,
            tcb,
            ctx,
            waits,
        );
    }
    let mut outs = batched_mlp(
        weights,
        &[prefix.to_string()],
        x,
        x_len,
        pool,
        tcb,
        ctx,
        waits,
    )?;
    outs.pop()
        .map(MlpWaveResult::Host)
        .ok_or_else(|| Error::Gravity("mlp_one empty".into()))
}

/// Gate/up/down co-issued across all prefixes via `matvec_batch` — three waits
/// total for the whole expert set (matches host `batched_mlp`). The residual
/// `x` and KV stay on device; per-expert gate/up/act vectors are ephemeral
/// because each down_proj takes a different input.
///
/// **Default resident path. Do not edit for expert-wave.** The flagged collapse
/// lives in [`moe_device_wave`]. Changing this function is a Parity V2.1 item 6
/// regression.
#[allow(clippy::too_many_arguments)]
fn batched_mlp<'a>(
    weights: &GpuWeightCache,
    prefixes: &[String],
    x: &Buffer,
    x_len: usize,
    _pool: &ActPool,
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    _ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<Vec<Vec<f32>>> {
    if prefixes.is_empty() {
        return Ok(Vec::new());
    }
    // Flush any pending attention/router encodes before the batch path, which
    // commits on its own.
    commit(tcb.take(), waits)?;
    let x_host = read_f32(x, x_len);
    let gate_names: Vec<String> = prefixes
        .iter()
        .map(|p| format!("{p}.gate_proj.weight"))
        .collect();
    let up_names: Vec<String> = prefixes
        .iter()
        .map(|p| format!("{p}.up_proj.weight"))
        .collect();
    let gate_calls: Vec<(&str, &[f32])> = gate_names
        .iter()
        .map(|n| (n.as_str(), x_host.as_slice()))
        .collect();
    let up_calls: Vec<(&str, &[f32])> = up_names
        .iter()
        .map(|n| (n.as_str(), x_host.as_slice()))
        .collect();
    let gate_outs = weights.matvec_batch(&gate_calls)?;
    waits.set(waits.get().saturating_add(1));
    let up_outs = weights.matvec_batch(&up_calls)?;
    waits.set(waits.get().saturating_add(1));
    let acts: Vec<Vec<f32>> = gate_outs
        .iter()
        .zip(&up_outs)
        .map(|(g, u)| {
            g.iter()
                .zip(u)
                .map(|(gv, uv)| (gv / (1.0 + (-gv).exp())) * uv)
                .collect()
        })
        .collect();
    let activation_elements = gate_outs.iter().map(Vec::len).sum::<usize>() as u64;
    crate::cost_ledger::record_source_modelled_operations(
        activation_elements.saturating_mul(4),
        0,
        0,
        activation_elements,
        0,
    );
    crate::cost_ledger::record_allocation(activation_elements.saturating_mul(4));
    let down_names: Vec<String> = prefixes
        .iter()
        .map(|p| format!("{p}.down_proj.weight"))
        .collect();
    let down_calls: Vec<(&str, &[f32])> = down_names
        .iter()
        .zip(&acts)
        .map(|(n, a)| (n.as_str(), a.as_slice()))
        .collect();
    let downs = weights.matvec_batch(&down_calls)?;
    waits.set(waits.get().saturating_add(1));
    Ok(downs)
}

// ── Typed route-segment primitives (default path untouched) ────────────────

/// Typed ABI boundary for GLM resident kernels.
///
/// These wrappers only append work to a caller-owned [`TokenCommandBuffer`].
/// They never submit, wait, or inspect flags. The compact append/K/attention/V
/// subset participates in [`forward_resident`] only after the default-off
/// compact session layout is selected; the ordinary expanded path is unchanged.
///
/// The existing MLA append and sparse-attention shaders use the expanded
/// `[position][head][qk/value]` cache. They are transitional correctness
/// scaffolding, not the 32K cache solution: the compact design must consume
/// normalized 512-wide MLA latent plus the shared 64-wide RoPE tail, or
/// reconstruct expanded K/V only for selected positions.
#[allow(dead_code)]
mod route_segment_primitives {
    use super::*;

    const TG: u32 = 256;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct GlmRopeParams {
        pub n_heads: u32,
        pub rotary_dim: u32,
        pub in_stride: u32,
        pub out_stride: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct GlmMlaAppendParams {
        pub n_heads: u32,
        pub qk_nope: u32,
        pub qk_rope: u32,
        pub v_dim: u32,
        pub pos: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct GlmMlaCompactAppendParams {
        pub latent_dim: u32,
        pub rope_dim: u32,
        pub pos: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct GlmPqKTransposeParams {
        pub n_heads: u32,
        pub key_rows: u32,
        pub row_stride: u32,
        pub latent_dim: u32,
        pub pq_dim: u32,
        pub pq_sub: u32,
        pub pq_nchunk: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(super) struct GlmCompactRankedAttnParams {
        pub n_heads: u32,
        pub latent_dim: u32,
        pub rope_dim: u32,
        pub n_keys: u32,
        pub n_allow: u32,
        pub scale: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct GlmPqVRowsParams {
        pub n_heads: u32,
        pub row_stride: u32,
        pub value_row_offset: u32,
        pub value_rows: u32,
        pub latent_dim: u32,
        pub pq_dim: u32,
        pub pq_sub: u32,
        pub pq_nchunk: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct GlmBuildQParams {
        pub n_heads: u32,
        pub qk_nope: u32,
        pub qk_rope: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(super) struct GlmDsaParams {
        pub n_keys: u32,
        pub n_heads: u32,
        pub head_dim: u32,
        pub pos: u32,
        pub dim_scale: f32,
        pub head_scale: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct GlmTopkParams {
        pub n: u32,
        pub k: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct GlmSortU32Params {
        pub n: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(super) struct GlmSparseAttnParams {
        pub n_heads: u32,
        pub qk_dim: u32,
        pub v_dim: u32,
        pub n_keys: u32,
        pub n_allow: u32,
        pub scale: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(super) struct GlmRouterSelectParams {
        pub n_experts: u32,
        pub n_group: u32,
        pub topk_group: u32,
        pub experts_per_token: u32,
        pub norm_topk_prob: u32,
        pub routed_scaling_factor: f32,
    }

    const _: [(); 16] = [(); std::mem::size_of::<GlmRopeParams>()];
    const _: [(); 20] = [(); std::mem::size_of::<GlmMlaAppendParams>()];
    const _: [(); 12] = [(); std::mem::size_of::<GlmMlaCompactAppendParams>()];
    const _: [(); 28] = [(); std::mem::size_of::<GlmPqKTransposeParams>()];
    const _: [(); 24] = [(); std::mem::size_of::<GlmCompactRankedAttnParams>()];
    const _: [(); 32] = [(); std::mem::size_of::<GlmPqVRowsParams>()];
    const _: [(); 12] = [(); std::mem::size_of::<GlmBuildQParams>()];
    const _: [(); 24] = [(); std::mem::size_of::<GlmDsaParams>()];
    const _: [(); 8] = [(); std::mem::size_of::<GlmTopkParams>()];
    const _: [(); 4] = [(); std::mem::size_of::<GlmSortU32Params>()];
    const _: [(); 24] = [(); std::mem::size_of::<GlmSparseAttnParams>()];
    const _: [(); 24] = [(); std::mem::size_of::<GlmRouterSelectParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmRopeParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmMlaAppendParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmMlaCompactAppendParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmPqKTransposeParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmCompactRankedAttnParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmPqVRowsParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmBuildQParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmDsaParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmTopkParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmSortU32Params>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmSparseAttnParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmRouterSelectParams>()];

    fn u32_arg(value: usize, what: &str) -> Result<u32> {
        u32::try_from(value)
            .map_err(|_| Error::Gravity(format!("{what}: {value} does not fit the Metal u32 ABI")))
    }

    fn checked_add(a: usize, b: usize, what: &str) -> Result<usize> {
        a.checked_add(b)
            .ok_or_else(|| Error::Gravity(format!("{what}: size overflow ({a} + {b})")))
    }

    fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize> {
        a.checked_mul(b)
            .ok_or_else(|| Error::Gravity(format!("{what}: size overflow ({a} x {b})")))
    }

    fn require_range(
        buffer: &Buffer,
        element_offset: usize,
        elements: usize,
        element_bytes: usize,
        what: &str,
    ) -> Result<u64> {
        let offset = checked_mul(element_offset, element_bytes, what)?;
        let bytes = checked_mul(elements, element_bytes, what)?;
        let end = checked_add(offset, bytes, what)?;
        if end as u64 > buffer.length() {
            return Err(Error::Gravity(format!(
                "{what}: needs byte range [{offset}, {end}), buffer has {} bytes",
                buffer.length()
            )));
        }
        Ok(offset as u64)
    }

    fn require_f32(
        buffer: &Buffer,
        element_offset: usize,
        elements: usize,
        what: &str,
    ) -> Result<u64> {
        require_range(
            buffer,
            element_offset,
            elements,
            std::mem::size_of::<f32>(),
            what,
        )
    }

    fn grid_1d(elements: u32, what: &str) -> Result<(u32, u32, u32)> {
        let groups = elements.div_ceil(TG);
        let width = groups
            .checked_mul(TG)
            .ok_or_else(|| Error::Gravity(format!("{what}: rounded Metal grid width overflow")))?;
        Ok((width, 1, 1))
    }

    fn strided_elements(count: usize, stride: usize, width: usize, what: &str) -> Result<usize> {
        if count == 0 {
            return Ok(0);
        }
        let preceding = checked_mul(count - 1, stride, what)?;
        checked_add(preceding, width, what)
    }

    pub(super) fn encode_rmsnorm(
        tcb: &mut TokenCommandBuffer<'_>,
        x: &Buffer,
        weight: &Buffer,
        out: &Buffer,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        require_f32(x, 0, n, "gravity_rmsnorm_f32 x")?;
        require_f32(weight, 0, n, "gravity_rmsnorm_f32 weight")?;
        require_f32(out, 0, n, "gravity_rmsnorm_f32 out")?;
        let n = u32_arg(n, "gravity_rmsnorm_f32 n")?;
        let xb = x.clone();
        let wb = weight.clone();
        let ob = out.clone();
        tcb.dispatch_threads("gravity_rmsnorm_f32", (TG, 1, 1), (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&xb), 0);
            enc.set_buffer(1, Some(&wb), 0);
            enc.set_buffer(2, Some(&ob), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
            enc.set_bytes(4, 4, &eps as *const f32 as *const _);
            enc.set_threadgroup_memory_length(0, (TG as u64) * 4);
        })
    }

    pub(super) fn encode_layernorm_affine(
        tcb: &mut TokenCommandBuffer<'_>,
        x: &Buffer,
        weight: &Buffer,
        bias: &Buffer,
        out: &Buffer,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        require_f32(x, 0, n, "gravity_layernorm_affine_f32 x")?;
        require_f32(weight, 0, n, "gravity_layernorm_affine_f32 weight")?;
        require_f32(bias, 0, n, "gravity_layernorm_affine_f32 bias")?;
        require_f32(out, 0, n, "gravity_layernorm_affine_f32 out")?;
        let n = u32_arg(n, "gravity_layernorm_affine_f32 n")?;
        let xb = x.clone();
        let wb = weight.clone();
        let bb = bias.clone();
        let ob = out.clone();
        tcb.dispatch_threads(
            "gravity_layernorm_affine_f32",
            (TG, 1, 1),
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&xb), 0);
                enc.set_buffer(1, Some(&wb), 0);
                enc.set_buffer(2, Some(&bb), 0);
                enc.set_buffer(3, Some(&ob), 0);
                enc.set_bytes(4, 4, &n as *const u32 as *const _);
                enc.set_bytes(5, 4, &eps as *const f32 as *const _);
                enc.set_threadgroup_memory_length(0, (TG as u64) * 4);
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_rope_interleaved(
        tcb: &mut TokenCommandBuffer<'_>,
        x: &Buffer,
        input_element_offset: usize,
        out: &Buffer,
        output_element_offset: usize,
        cos: &Buffer,
        sin: &Buffer,
        n_heads: usize,
        rotary_dim: usize,
        in_stride: usize,
        out_stride: usize,
    ) -> Result<()> {
        if n_heads == 0 || rotary_dim == 0 {
            return Ok(());
        }
        if rotary_dim % 2 != 0 || in_stride < rotary_dim || out_stride < rotary_dim {
            return Err(Error::Gravity(format!(
                "gravity_rope_interleaved_f32 invalid geometry: heads={n_heads}, rotary_dim={rotary_dim}, in_stride={in_stride}, out_stride={out_stride}"
            )));
        }
        let input_len = strided_elements(
            n_heads,
            in_stride,
            rotary_dim,
            "gravity_rope_interleaved_f32 input",
        )?;
        let output_len = strided_elements(
            n_heads,
            out_stride,
            rotary_dim,
            "gravity_rope_interleaved_f32 output",
        )?;
        let input_byte_offset = require_f32(
            x,
            input_element_offset,
            input_len,
            "gravity_rope_interleaved_f32 input",
        )?;
        let output_byte_offset = require_f32(
            out,
            output_element_offset,
            output_len,
            "gravity_rope_interleaved_f32 output",
        )?;
        require_f32(cos, 0, rotary_dim / 2, "gravity_rope_interleaved_f32 cos")?;
        require_f32(sin, 0, rotary_dim / 2, "gravity_rope_interleaved_f32 sin")?;
        let params = GlmRopeParams {
            n_heads: u32_arg(n_heads, "gravity_rope_interleaved_f32 n_heads")?,
            rotary_dim: u32_arg(rotary_dim, "gravity_rope_interleaved_f32 rotary_dim")?,
            in_stride: u32_arg(in_stride, "gravity_rope_interleaved_f32 in_stride")?,
            out_stride: u32_arg(out_stride, "gravity_rope_interleaved_f32 out_stride")?,
        };
        let threads = params
            .n_heads
            .checked_mul(params.rotary_dim / 2)
            .ok_or_else(|| Error::Gravity("gravity_rope_interleaved_f32 grid overflow".into()))?;
        let grid = grid_1d(threads, "gravity_rope_interleaved_f32")?;
        let xb = x.clone();
        let ob = out.clone();
        let cb = cos.clone();
        let sb = sin.clone();
        tcb.dispatch_threads(
            "gravity_rope_interleaved_f32",
            grid,
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&xb), input_byte_offset);
                enc.set_buffer(1, Some(&ob), output_byte_offset);
                enc.set_buffer(2, Some(&cb), 0);
                enc.set_buffer(3, Some(&sb), 0);
                enc.set_bytes(
                    4,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_rope_prefix_tail(
        tcb: &mut TokenCommandBuffer<'_>,
        x: &Buffer,
        input_element_offset: usize,
        out: &Buffer,
        output_element_offset: usize,
        cos: &Buffer,
        sin: &Buffer,
        n_heads: usize,
        rotary_dim: usize,
        in_stride: usize,
        out_stride: usize,
    ) -> Result<()> {
        if n_heads == 0 || out_stride == 0 {
            return Ok(());
        }
        if rotary_dim == 0
            || rotary_dim % 2 != 0
            || in_stride < out_stride
            || out_stride < rotary_dim
        {
            return Err(Error::Gravity(format!(
                "gravity_rope_prefix_tail_f32 invalid geometry: heads={n_heads}, rotary_dim={rotary_dim}, in_stride={in_stride}, out_stride={out_stride}"
            )));
        }
        if x.contents() == out.contents() {
            return Err(Error::Gravity(
                "gravity_rope_prefix_tail_f32 requires non-aliasing input/output".into(),
            ));
        }
        let input_len = strided_elements(
            n_heads,
            in_stride,
            out_stride,
            "gravity_rope_prefix_tail_f32 input",
        )?;
        let output_len = checked_mul(n_heads, out_stride, "gravity_rope_prefix_tail_f32 output")?;
        let input_byte_offset = require_f32(
            x,
            input_element_offset,
            input_len,
            "gravity_rope_prefix_tail_f32 input",
        )?;
        let output_byte_offset = require_f32(
            out,
            output_element_offset,
            output_len,
            "gravity_rope_prefix_tail_f32 output",
        )?;
        require_f32(cos, 0, rotary_dim / 2, "gravity_rope_prefix_tail_f32 cos")?;
        require_f32(sin, 0, rotary_dim / 2, "gravity_rope_prefix_tail_f32 sin")?;
        let params = GlmRopeParams {
            n_heads: u32_arg(n_heads, "gravity_rope_prefix_tail_f32 n_heads")?,
            rotary_dim: u32_arg(rotary_dim, "gravity_rope_prefix_tail_f32 rotary_dim")?,
            in_stride: u32_arg(in_stride, "gravity_rope_prefix_tail_f32 in_stride")?,
            out_stride: u32_arg(out_stride, "gravity_rope_prefix_tail_f32 out_stride")?,
        };
        let threads = params
            .n_heads
            .checked_mul(params.out_stride)
            .ok_or_else(|| Error::Gravity("gravity_rope_prefix_tail_f32 grid overflow".into()))?;
        let grid = grid_1d(threads, "gravity_rope_prefix_tail_f32")?;
        let xb = x.clone();
        let ob = out.clone();
        let cb = cos.clone();
        let sb = sin.clone();
        tcb.dispatch_threads(
            "gravity_rope_prefix_tail_f32",
            grid,
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&xb), input_byte_offset);
                enc.set_buffer(1, Some(&ob), output_byte_offset);
                enc.set_buffer(2, Some(&cb), 0);
                enc.set_buffer(3, Some(&sb), 0);
                enc.set_bytes(
                    4,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
            },
        )
    }

    pub(super) fn encode_copy_tail(
        tcb: &mut TokenCommandBuffer<'_>,
        src: &Buffer,
        dst: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        require_f32(src, src_offset, n, "gravity_copy_tail_f32 src")?;
        require_f32(dst, dst_offset, n, "gravity_copy_tail_f32 dst")?;
        let src_offset = u32_arg(src_offset, "gravity_copy_tail_f32 src_off")?;
        let dst_offset = u32_arg(dst_offset, "gravity_copy_tail_f32 dst_off")?;
        let n = u32_arg(n, "gravity_copy_tail_f32 n")?;
        let grid = grid_1d(n, "gravity_copy_tail_f32")?;
        let sb = src.clone();
        let db = dst.clone();
        tcb.dispatch_threads("gravity_copy_tail_f32", grid, (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&sb), 0);
            enc.set_buffer(1, Some(&db), 0);
            enc.set_bytes(2, 4, &src_offset as *const u32 as *const _);
            enc.set_bytes(3, 4, &dst_offset as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_mla_append_kv_expanded(
        tcb: &mut TokenCommandBuffer<'_>,
        kv: &Buffer,
        k_rot: &Buffer,
        keys: &Buffer,
        values: &Buffer,
        n_heads: usize,
        qk_nope: usize,
        qk_rope: usize,
        v_dim: usize,
        position: usize,
    ) -> Result<()> {
        let qk = checked_add(qk_nope, qk_rope, "gravity_glm_mla_append_kv qk")?;
        let per_kv = checked_add(qk_nope, v_dim, "gravity_glm_mla_append_kv per_kv")?;
        let key_elems = checked_mul(n_heads, qk, "gravity_glm_mla_append_kv key elements")?;
        let value_elems = checked_mul(n_heads, v_dim, "gravity_glm_mla_append_kv value elements")?;
        let total = checked_add(key_elems, value_elems, "gravity_glm_mla_append_kv grid")?;
        if total == 0 {
            return Ok(());
        }
        require_f32(
            kv,
            0,
            checked_mul(n_heads, per_kv, "gravity_glm_mla_append_kv kv")?,
            "gravity_glm_mla_append_kv kv",
        )?;
        require_f32(k_rot, 0, qk_rope, "gravity_glm_mla_append_kv k_rot")?;
        let positions = checked_add(position, 1, "gravity_glm_mla_append_kv position")?;
        require_f32(
            keys,
            0,
            checked_mul(positions, key_elems, "gravity_glm_mla_append_kv keys")?,
            "gravity_glm_mla_append_kv expanded keys",
        )?;
        require_f32(
            values,
            0,
            checked_mul(positions, value_elems, "gravity_glm_mla_append_kv values")?,
            "gravity_glm_mla_append_kv expanded values",
        )?;
        let params = GlmMlaAppendParams {
            n_heads: u32_arg(n_heads, "gravity_glm_mla_append_kv n_heads")?,
            qk_nope: u32_arg(qk_nope, "gravity_glm_mla_append_kv qk_nope")?,
            qk_rope: u32_arg(qk_rope, "gravity_glm_mla_append_kv qk_rope")?,
            v_dim: u32_arg(v_dim, "gravity_glm_mla_append_kv v_dim")?,
            pos: u32_arg(position, "gravity_glm_mla_append_kv pos")?,
        };
        let grid = grid_1d(
            u32_arg(total, "gravity_glm_mla_append_kv total")?,
            "gravity_glm_mla_append_kv",
        )?;
        let kvb = kv.clone();
        let krb = k_rot.clone();
        let kb = keys.clone();
        let vb = values.clone();
        tcb.dispatch_threads("gravity_glm_mla_append_kv", grid, (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&kvb), 0);
            enc.set_buffer(1, Some(&krb), 0);
            enc.set_buffer(2, Some(&kb), 0);
            enc.set_buffer(3, Some(&vb), 0);
            enc.set_bytes(
                4,
                std::mem::size_of_val(&params) as u64,
                &params as *const _ as *const _,
            );
        })
    }

    /// Encode one compact MLA cache append.
    ///
    /// This stores the normalized KV latent and shared rotated RoPE tail
    /// directly as `[position][dimension]`. It does not expand either value
    /// across attention heads.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_mla_append_compact(
        tcb: &mut TokenCommandBuffer<'_>,
        latent: &Buffer,
        k_rot: &Buffer,
        latent_cache: &Buffer,
        rope_cache: &Buffer,
        latent_dim: usize,
        rope_dim: usize,
        position: usize,
    ) -> Result<()> {
        let total = checked_add(latent_dim, rope_dim, "gravity_glm_mla_append_compact grid")?;
        if total == 0 {
            return Ok(());
        }
        require_f32(
            latent,
            0,
            latent_dim,
            "gravity_glm_mla_append_compact latent",
        )?;
        require_f32(k_rot, 0, rope_dim, "gravity_glm_mla_append_compact k_rot")?;
        let positions = checked_add(position, 1, "gravity_glm_mla_append_compact position")?;
        require_f32(
            latent_cache,
            0,
            checked_mul(
                positions,
                latent_dim,
                "gravity_glm_mla_append_compact latent cache",
            )?,
            "gravity_glm_mla_append_compact latent cache",
        )?;
        require_f32(
            rope_cache,
            0,
            checked_mul(
                positions,
                rope_dim,
                "gravity_glm_mla_append_compact rope cache",
            )?,
            "gravity_glm_mla_append_compact rope cache",
        )?;
        let params = GlmMlaCompactAppendParams {
            latent_dim: u32_arg(latent_dim, "gravity_glm_mla_append_compact latent_dim")?,
            rope_dim: u32_arg(rope_dim, "gravity_glm_mla_append_compact rope_dim")?,
            pos: u32_arg(position, "gravity_glm_mla_append_compact pos")?,
        };
        let grid = grid_1d(
            u32_arg(total, "gravity_glm_mla_append_compact total")?,
            "gravity_glm_mla_append_compact",
        )?;
        let lb = latent.clone();
        let rb = k_rot.clone();
        let lcb = latent_cache.clone();
        let rcb = rope_cache.clone();
        tcb.dispatch_threads(
            "gravity_glm_mla_append_compact",
            grid,
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&lb), 0);
                enc.set_buffer(1, Some(&rb), 0);
                enc.set_buffer(2, Some(&lcb), 0);
                enc.set_buffer(3, Some(&rcb), 0);
                enc.set_bytes(
                    4,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
            },
        )
    }

    /// Encode `W_key^T @ query_nope` per head directly from a bits=8,
    /// single-subspace gravity-pq matrix.
    ///
    /// Logical key rows are the first `key_rows` rows inside each
    /// `row_stride`-wide head block. The inner reduction is compensated in
    /// ascending key-row order.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_pq_k_transpose_heads(
        tcb: &mut TokenCommandBuffer<'_>,
        codebooks: &Buffer,
        codes: &Buffer,
        query_nope: &Buffer,
        query_latent: &Buffer,
        n_heads: usize,
        key_rows: usize,
        row_stride: usize,
        latent_dim: usize,
        pq_dim: usize,
        pq_sub: usize,
        pq_card: usize,
        pq_bits: usize,
        pq_nchunk: usize,
    ) -> Result<()> {
        if n_heads == 0 || key_rows == 0 || latent_dim == 0 {
            return Ok(());
        }
        if row_stride < key_rows {
            return Err(Error::Gravity(format!(
                "gravity_pq_k_transpose_heads row_stride {row_stride} < key_rows {key_rows}"
            )));
        }
        if pq_dim == 0 || pq_sub == 0 || pq_card != 256 || pq_bits != 8 {
            return Err(Error::Gravity(format!(
                "gravity_pq_k_transpose_heads requires direct-u8 bits=8 cardinality=256, got dim={pq_dim}, sub={pq_sub}, card={pq_card}, bits={pq_bits}"
            )));
        }
        if pq_dim != pq_sub {
            return Err(Error::Gravity(format!(
                "gravity_pq_k_transpose_heads requires one subspace with dim == sub, got dim={pq_dim}, sub={pq_sub}"
            )));
        }
        let represented_cols = checked_mul(
            pq_nchunk,
            pq_dim,
            "gravity_pq_k_transpose_heads represented columns",
        )?;
        if represented_cols != latent_dim {
            return Err(Error::Gravity(format!(
                "gravity_pq_k_transpose_heads latent_dim {latent_dim} != pq_nchunk {pq_nchunk} * pq_dim {pq_dim}"
            )));
        }
        require_range(
            codebooks,
            0,
            checked_mul(pq_card, pq_sub, "gravity_pq_k_transpose_heads codebook")?,
            std::mem::size_of::<half::f16>(),
            "gravity_pq_k_transpose_heads codebook",
        )?;
        let last_head_base = checked_mul(
            n_heads - 1,
            row_stride,
            "gravity_pq_k_transpose_heads last head",
        )?;
        let rows_touched = checked_add(
            last_head_base,
            key_rows,
            "gravity_pq_k_transpose_heads rows touched",
        )?;
        require_range(
            codes,
            0,
            checked_mul(
                rows_touched,
                pq_nchunk,
                "gravity_pq_k_transpose_heads codes",
            )?,
            std::mem::size_of::<u8>(),
            "gravity_pq_k_transpose_heads codes",
        )?;
        require_f32(
            query_nope,
            0,
            checked_mul(n_heads, key_rows, "gravity_pq_k_transpose_heads query")?,
            "gravity_pq_k_transpose_heads query",
        )?;
        let outputs = checked_mul(n_heads, latent_dim, "gravity_pq_k_transpose_heads output")?;
        require_f32(
            query_latent,
            0,
            outputs,
            "gravity_pq_k_transpose_heads output",
        )?;
        let params = GlmPqKTransposeParams {
            n_heads: u32_arg(n_heads, "gravity_pq_k_transpose_heads n_heads")?,
            key_rows: u32_arg(key_rows, "gravity_pq_k_transpose_heads key_rows")?,
            row_stride: u32_arg(row_stride, "gravity_pq_k_transpose_heads row_stride")?,
            latent_dim: u32_arg(latent_dim, "gravity_pq_k_transpose_heads latent_dim")?,
            pq_dim: u32_arg(pq_dim, "gravity_pq_k_transpose_heads pq_dim")?,
            pq_sub: u32_arg(pq_sub, "gravity_pq_k_transpose_heads pq_sub")?,
            pq_nchunk: u32_arg(pq_nchunk, "gravity_pq_k_transpose_heads pq_nchunk")?,
        };
        let grid = grid_1d(
            u32_arg(outputs, "gravity_pq_k_transpose_heads outputs")?,
            "gravity_pq_k_transpose_heads",
        )?;
        let cbb = codebooks.clone();
        let cib = codes.clone();
        let qb = query_nope.clone();
        let ob = query_latent.clone();
        tcb.dispatch_threads(
            "gravity_pq_k_transpose_heads",
            grid,
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&cbb), 0);
                enc.set_buffer(1, Some(&cib), 0);
                enc.set_buffer(2, Some(&qb), 0);
                enc.set_buffer(3, Some(&ob), 0);
                enc.set_bytes(
                    4,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
            },
        )
    }

    /// Encode compact absorbed MLA attention over stable DSA-ranked positions.
    ///
    /// Each head computes content scores from the compact latent cache,
    /// appends the shared RoPE score, normalizes in the supplied rank order,
    /// and produces one probability-weighted latent. `query_latent` and
    /// `weighted_latent` may be the same buffer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_compact_ranked_attention(
        tcb: &mut TokenCommandBuffer<'_>,
        query_latent: &Buffer,
        query_rope: &Buffer,
        latent_cache: &Buffer,
        rope_cache: &Buffer,
        ranked_indices: &Buffer,
        weighted_latent: &Buffer,
        n_heads: usize,
        latent_dim: usize,
        rope_dim: usize,
        n_keys: usize,
        n_allow: usize,
        scale: f32,
    ) -> Result<()> {
        const MAX_ALLOW: usize = 2048;
        if n_heads == 0 || latent_dim == 0 {
            return Ok(());
        }
        if n_allow > MAX_ALLOW {
            return Err(Error::Gravity(format!(
                "gravity_glm_compact_ranked_attn supports n_allow <= {MAX_ALLOW}, got {n_allow}"
            )));
        }
        let query_elements = checked_mul(
            n_heads,
            latent_dim,
            "gravity_glm_compact_ranked_attn query latent",
        )?;
        require_f32(
            query_latent,
            0,
            query_elements,
            "gravity_glm_compact_ranked_attn query latent",
        )?;
        require_f32(
            query_rope,
            0,
            checked_mul(
                n_heads,
                rope_dim,
                "gravity_glm_compact_ranked_attn query rope",
            )?,
            "gravity_glm_compact_ranked_attn query rope",
        )?;
        require_f32(
            latent_cache,
            0,
            checked_mul(
                n_keys,
                latent_dim,
                "gravity_glm_compact_ranked_attn latent cache",
            )?,
            "gravity_glm_compact_ranked_attn latent cache",
        )?;
        require_f32(
            rope_cache,
            0,
            checked_mul(
                n_keys,
                rope_dim,
                "gravity_glm_compact_ranked_attn rope cache",
            )?,
            "gravity_glm_compact_ranked_attn rope cache",
        )?;
        require_range(
            ranked_indices,
            0,
            n_allow,
            std::mem::size_of::<u32>(),
            "gravity_glm_compact_ranked_attn ranked indices",
        )?;
        require_f32(
            weighted_latent,
            0,
            query_elements,
            "gravity_glm_compact_ranked_attn weighted latent",
        )?;
        let params = GlmCompactRankedAttnParams {
            n_heads: u32_arg(n_heads, "gravity_glm_compact_ranked_attn n_heads")?,
            latent_dim: u32_arg(latent_dim, "gravity_glm_compact_ranked_attn latent_dim")?,
            rope_dim: u32_arg(rope_dim, "gravity_glm_compact_ranked_attn rope_dim")?,
            n_keys: u32_arg(n_keys, "gravity_glm_compact_ranked_attn n_keys")?,
            n_allow: u32_arg(n_allow, "gravity_glm_compact_ranked_attn n_allow")?,
            scale,
        };
        let grid_width = params.n_heads.checked_mul(TG).ok_or_else(|| {
            Error::Gravity("gravity_glm_compact_ranked_attn grid overflow".into())
        })?;
        let shmem = checked_mul(
            n_allow.max(1),
            std::mem::size_of::<f32>(),
            "gravity_glm_compact_ranked_attn threadgroup memory",
        )?;
        let qlb = query_latent.clone();
        let qrb = query_rope.clone();
        let lcb = latent_cache.clone();
        let rcb = rope_cache.clone();
        let rib = ranked_indices.clone();
        let wlb = weighted_latent.clone();
        tcb.dispatch_threads(
            "gravity_glm_compact_ranked_attn",
            (grid_width, 1, 1),
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&qlb), 0);
                enc.set_buffer(1, Some(&qrb), 0);
                enc.set_buffer(2, Some(&lcb), 0);
                enc.set_buffer(3, Some(&rcb), 0);
                enc.set_buffer(4, Some(&rib), 0);
                enc.set_buffer(5, Some(&wlb), 0);
                enc.set_bytes(
                    6,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem as u64);
            },
        )
    }

    /// Encode the per-head value-row window directly from a bits=8,
    /// single-subspace gravity-pq K/V matrix.
    ///
    /// One SIMD group owns each logical value row and matches the generic PQ
    /// matvec lane/chunk reduction order. Compact MLA calls this from its
    /// default-off five-dispatch attention DAG.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_pq_v_rows_heads(
        tcb: &mut TokenCommandBuffer<'_>,
        codebooks: &Buffer,
        codes: &Buffer,
        weighted_latent: &Buffer,
        context: &Buffer,
        n_heads: usize,
        row_stride: usize,
        value_row_offset: usize,
        value_rows: usize,
        latent_dim: usize,
        pq_dim: usize,
        pq_sub: usize,
        pq_card: usize,
        pq_bits: usize,
        pq_nchunk: usize,
    ) -> Result<()> {
        if n_heads == 0 || value_rows == 0 || latent_dim == 0 {
            return Ok(());
        }
        let row_end = checked_add(
            value_row_offset,
            value_rows,
            "gravity_pq_v_rows_heads value row window",
        )?;
        if row_end > row_stride {
            return Err(Error::Gravity(format!(
                "gravity_pq_v_rows_heads value window [{value_row_offset}, {row_end}) exceeds row_stride {row_stride}"
            )));
        }
        if pq_dim == 0 || pq_sub == 0 || pq_card != 256 || pq_bits != 8 {
            return Err(Error::Gravity(format!(
                "gravity_pq_v_rows_heads requires direct-u8 bits=8 cardinality=256, got dim={pq_dim}, sub={pq_sub}, card={pq_card}, bits={pq_bits}"
            )));
        }
        if pq_dim != pq_sub {
            return Err(Error::Gravity(format!(
                "gravity_pq_v_rows_heads requires one subspace with dim == sub, got dim={pq_dim}, sub={pq_sub}"
            )));
        }
        let represented_cols = checked_mul(
            pq_nchunk,
            pq_dim,
            "gravity_pq_v_rows_heads represented columns",
        )?;
        if represented_cols != latent_dim {
            return Err(Error::Gravity(format!(
                "gravity_pq_v_rows_heads latent_dim {latent_dim} != pq_nchunk {pq_nchunk} * pq_dim {pq_dim}"
            )));
        }
        require_range(
            codebooks,
            0,
            checked_mul(pq_card, pq_sub, "gravity_pq_v_rows_heads codebook")?,
            std::mem::size_of::<half::f16>(),
            "gravity_pq_v_rows_heads codebook",
        )?;
        let last_head_base =
            checked_mul(n_heads - 1, row_stride, "gravity_pq_v_rows_heads last head")?;
        let rows_touched = checked_add(
            last_head_base,
            row_end,
            "gravity_pq_v_rows_heads rows touched",
        )?;
        require_range(
            codes,
            0,
            checked_mul(rows_touched, pq_nchunk, "gravity_pq_v_rows_heads codes")?,
            std::mem::size_of::<u8>(),
            "gravity_pq_v_rows_heads codes",
        )?;
        require_f32(
            weighted_latent,
            0,
            checked_mul(
                n_heads,
                latent_dim,
                "gravity_pq_v_rows_heads weighted latent",
            )?,
            "gravity_pq_v_rows_heads weighted latent",
        )?;
        let outputs = checked_mul(n_heads, value_rows, "gravity_pq_v_rows_heads output")?;
        require_f32(context, 0, outputs, "gravity_pq_v_rows_heads context")?;
        let params = GlmPqVRowsParams {
            n_heads: u32_arg(n_heads, "gravity_pq_v_rows_heads n_heads")?,
            row_stride: u32_arg(row_stride, "gravity_pq_v_rows_heads row_stride")?,
            value_row_offset: u32_arg(
                value_row_offset,
                "gravity_pq_v_rows_heads value_row_offset",
            )?,
            value_rows: u32_arg(value_rows, "gravity_pq_v_rows_heads value_rows")?,
            latent_dim: u32_arg(latent_dim, "gravity_pq_v_rows_heads latent_dim")?,
            pq_dim: u32_arg(pq_dim, "gravity_pq_v_rows_heads pq_dim")?,
            pq_sub: u32_arg(pq_sub, "gravity_pq_v_rows_heads pq_sub")?,
            pq_nchunk: u32_arg(pq_nchunk, "gravity_pq_v_rows_heads pq_nchunk")?,
        };
        const OUTPUTS_PER_THREADGROUP: usize = (TG / 32) as usize;
        let grid_threads = outputs
            .div_ceil(OUTPUTS_PER_THREADGROUP)
            .checked_mul(TG as usize)
            .ok_or_else(|| Error::Gravity("gravity_pq_v_rows_heads grid overflow".into()))?;
        let grid = (
            u32_arg(grid_threads, "gravity_pq_v_rows_heads grid threads")?,
            1,
            1,
        );
        let cbb = codebooks.clone();
        let cib = codes.clone();
        let wlb = weighted_latent.clone();
        let cb = context.clone();
        tcb.dispatch_threads("gravity_pq_v_rows_heads", grid, (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&cbb), 0);
            enc.set_buffer(1, Some(&cib), 0);
            enc.set_buffer(2, Some(&wlb), 0);
            enc.set_buffer(3, Some(&cb), 0);
            enc.set_bytes(
                4,
                std::mem::size_of_val(&params) as u64,
                &params as *const _ as *const _,
            );
        })
    }

    pub(super) fn encode_build_queries(
        tcb: &mut TokenCommandBuffer<'_>,
        q: &Buffer,
        q_rope_rot: &Buffer,
        queries: &Buffer,
        n_heads: usize,
        qk_nope: usize,
        qk_rope: usize,
    ) -> Result<()> {
        let qk = checked_add(qk_nope, qk_rope, "gravity_glm_build_queries qk")?;
        let total = checked_mul(n_heads, qk, "gravity_glm_build_queries total")?;
        if total == 0 {
            return Ok(());
        }
        require_f32(q, 0, total, "gravity_glm_build_queries q")?;
        require_f32(
            q_rope_rot,
            0,
            checked_mul(n_heads, qk_rope, "gravity_glm_build_queries q_rope_rot")?,
            "gravity_glm_build_queries q_rope_rot",
        )?;
        require_f32(queries, 0, total, "gravity_glm_build_queries queries")?;
        let params = GlmBuildQParams {
            n_heads: u32_arg(n_heads, "gravity_glm_build_queries n_heads")?,
            qk_nope: u32_arg(qk_nope, "gravity_glm_build_queries qk_nope")?,
            qk_rope: u32_arg(qk_rope, "gravity_glm_build_queries qk_rope")?,
        };
        let grid = grid_1d(
            u32_arg(total, "gravity_glm_build_queries total")?,
            "gravity_glm_build_queries",
        )?;
        let qb = q.clone();
        let rb = q_rope_rot.clone();
        let ob = queries.clone();
        tcb.dispatch_threads("gravity_glm_build_queries", grid, (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&qb), 0);
            enc.set_buffer(1, Some(&rb), 0);
            enc.set_buffer(2, Some(&ob), 0);
            enc.set_bytes(
                3,
                std::mem::size_of_val(&params) as u64,
                &params as *const _ as *const _,
            );
        })
    }

    pub(super) fn encode_copy_head_prefix(
        tcb: &mut TokenCommandBuffer<'_>,
        q: &Buffer,
        prefix: &Buffer,
        n_heads: usize,
        qk_nope: usize,
        qk_rope: usize,
    ) -> Result<()> {
        let qk = checked_add(qk_nope, qk_rope, "gravity_copy_head_prefix_f32 qk")?;
        let input = checked_mul(n_heads, qk, "gravity_copy_head_prefix_f32 input")?;
        let output = checked_mul(n_heads, qk_nope, "gravity_copy_head_prefix_f32 output")?;
        if output == 0 {
            return Ok(());
        }
        require_f32(q, 0, input, "gravity_copy_head_prefix_f32 q")?;
        require_f32(prefix, 0, output, "gravity_copy_head_prefix_f32 prefix")?;
        let params = GlmBuildQParams {
            n_heads: u32_arg(n_heads, "gravity_copy_head_prefix_f32 n_heads")?,
            qk_nope: u32_arg(qk_nope, "gravity_copy_head_prefix_f32 qk_nope")?,
            qk_rope: u32_arg(qk_rope, "gravity_copy_head_prefix_f32 qk_rope")?,
        };
        let grid = grid_1d(
            u32_arg(output, "gravity_copy_head_prefix_f32 output")?,
            "gravity_copy_head_prefix_f32",
        )?;
        let qb = q.clone();
        let pb = prefix.clone();
        tcb.dispatch_threads(
            "gravity_copy_head_prefix_f32",
            grid,
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&qb), 0);
                enc.set_buffer(1, Some(&pb), 0);
                enc.set_bytes(
                    2,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
            },
        )
    }

    pub(super) fn encode_append_index_key(
        tcb: &mut TokenCommandBuffer<'_>,
        k_full: &Buffer,
        index_keys: &Buffer,
        position: usize,
        head_dim: usize,
    ) -> Result<()> {
        if head_dim == 0 {
            return Ok(());
        }
        require_f32(k_full, 0, head_dim, "gravity_glm_append_index_key k_full")?;
        let positions = checked_add(position, 1, "gravity_glm_append_index_key position")?;
        require_f32(
            index_keys,
            0,
            checked_mul(
                positions,
                head_dim,
                "gravity_glm_append_index_key index_keys",
            )?,
            "gravity_glm_append_index_key index_keys",
        )?;
        let position = u32_arg(position, "gravity_glm_append_index_key pos")?;
        let head_dim = u32_arg(head_dim, "gravity_glm_append_index_key idim")?;
        let grid = grid_1d(head_dim, "gravity_glm_append_index_key")?;
        let kb = k_full.clone();
        let ib = index_keys.clone();
        tcb.dispatch_threads(
            "gravity_glm_append_index_key",
            grid,
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&kb), 0);
                enc.set_buffer(1, Some(&ib), 0);
                enc.set_bytes(2, 4, &position as *const u32 as *const _);
                enc.set_bytes(3, 4, &head_dim as *const u32 as *const _);
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_dsa_scores(
        tcb: &mut TokenCommandBuffer<'_>,
        q_full: &Buffer,
        index_keys: &Buffer,
        head_weights: &Buffer,
        scores: &Buffer,
        n_keys: usize,
        n_heads: usize,
        head_dim: usize,
        position: usize,
        dim_scale: f32,
        head_scale: f32,
    ) -> Result<()> {
        if n_keys == 0 {
            return Ok(());
        }
        require_f32(
            q_full,
            0,
            checked_mul(n_heads, head_dim, "gravity_glm_dsa_scores q_full")?,
            "gravity_glm_dsa_scores q_full",
        )?;
        require_f32(
            index_keys,
            0,
            checked_mul(n_keys, head_dim, "gravity_glm_dsa_scores index_keys")?,
            "gravity_glm_dsa_scores index_keys",
        )?;
        require_f32(
            head_weights,
            0,
            n_heads,
            "gravity_glm_dsa_scores head_weights",
        )?;
        require_f32(scores, 0, n_keys, "gravity_glm_dsa_scores scores")?;
        let params = GlmDsaParams {
            n_keys: u32_arg(n_keys, "gravity_glm_dsa_scores n_keys")?,
            n_heads: u32_arg(n_heads, "gravity_glm_dsa_scores n_heads")?,
            head_dim: u32_arg(head_dim, "gravity_glm_dsa_scores head_dim")?,
            pos: u32_arg(position, "gravity_glm_dsa_scores pos")?,
            dim_scale,
            head_scale,
        };
        let grid = grid_1d(params.n_keys, "gravity_glm_dsa_scores")?;
        let qb = q_full.clone();
        let kb = index_keys.clone();
        let wb = head_weights.clone();
        let sb = scores.clone();
        tcb.dispatch_threads("gravity_glm_dsa_scores", grid, (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&qb), 0);
            enc.set_buffer(1, Some(&kb), 0);
            enc.set_buffer(2, Some(&wb), 0);
            enc.set_buffer(3, Some(&sb), 0);
            enc.set_bytes(
                4,
                std::mem::size_of_val(&params) as u64,
                &params as *const _ as *const _,
            );
        })
    }

    pub(super) fn encode_stable_topk(
        tcb: &mut TokenCommandBuffer<'_>,
        values: &Buffer,
        indices: &Buffer,
        selected_scratch: &Buffer,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if n == 0 || k == 0 {
            return Ok(());
        }
        let out_len = k.min(n);
        require_f32(values, 0, n, "gravity_glm_stable_topk_f32 values")?;
        require_range(
            indices,
            0,
            out_len,
            std::mem::size_of::<u32>(),
            "gravity_glm_stable_topk_f32 indices",
        )?;
        require_range(
            selected_scratch,
            0,
            n,
            std::mem::size_of::<u8>(),
            "gravity_glm_stable_topk_f32 selected",
        )?;
        let params = GlmTopkParams {
            n: u32_arg(n, "gravity_glm_stable_topk_f32 n")?,
            k: u32_arg(k, "gravity_glm_stable_topk_f32 k")?,
        };
        let vb = values.clone();
        let ib = indices.clone();
        let sb = selected_scratch.clone();
        tcb.dispatch_threads(
            "gravity_glm_stable_topk_f32",
            (1, 1, 1),
            (1, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&vb), 0);
                enc.set_buffer(1, Some(&ib), 0);
                enc.set_buffer(2, Some(&sb), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
            },
        )
    }

    /// One-threadgroup exact radix-select + bitonic-rank candidate.
    ///
    /// Output order is identical to [`encode_stable_topk`]: descending score,
    /// lower position first on ties. The fixed 16 KiB rank workspace admits
    /// at most 2048 indices, matching compact attention's bound.
    pub(super) fn encode_radix_topk(
        tcb: &mut TokenCommandBuffer<'_>,
        values: &Buffer,
        indices: &Buffer,
        n: usize,
        k: usize,
    ) -> Result<()> {
        const MAX_K: usize = 2048;
        if n == 0 || k == 0 {
            return Ok(());
        }
        if k > MAX_K {
            return Err(Error::Gravity(format!(
                "gravity_glm_radix_topk_f32 supports k <= {MAX_K}, got {k}"
            )));
        }
        let out_len = k.min(n);
        require_f32(values, 0, n, "gravity_glm_radix_topk_f32 values")?;
        require_range(
            indices,
            0,
            out_len,
            std::mem::size_of::<u32>(),
            "gravity_glm_radix_topk_f32 indices",
        )?;
        let params = GlmTopkParams {
            n: u32_arg(n, "gravity_glm_radix_topk_f32 n")?,
            k: u32_arg(k, "gravity_glm_radix_topk_f32 k")?,
        };
        let vb = values.clone();
        let ib = indices.clone();
        tcb.dispatch_threads(
            "gravity_glm_radix_topk_f32",
            (TG, 1, 1),
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&vb), 0);
                enc.set_buffer(1, Some(&ib), 0);
                enc.set_bytes(
                    2,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
            },
        )
    }

    /// Sort unique stable-top-k position IDs into ascending host accumulation
    /// order. Bounded to the flagship `index_topk <= 2048` contract.
    ///
    /// The kernel uses one 256-thread group and one power-of-two-padded u32
    /// array in dynamic threadgroup memory (maximum 8 KiB). Input and output
    /// may be the same Metal buffer.
    pub(super) fn encode_sort_positions_ascending(
        tcb: &mut TokenCommandBuffer<'_>,
        score_ordered_indices: &Buffer,
        ascending_indices: &Buffer,
        k: usize,
    ) -> Result<()> {
        const MAX_K: usize = 2048;
        if k == 0 {
            return Ok(());
        }
        if k > MAX_K {
            return Err(Error::Gravity(format!(
                "gravity_glm_sort_u32_ascending supports k <= {MAX_K}, got {k}"
            )));
        }
        require_range(
            score_ordered_indices,
            0,
            k,
            std::mem::size_of::<u32>(),
            "gravity_glm_sort_u32_ascending input",
        )?;
        require_range(
            ascending_indices,
            0,
            k,
            std::mem::size_of::<u32>(),
            "gravity_glm_sort_u32_ascending output",
        )?;
        let padded = k.checked_next_power_of_two().ok_or_else(|| {
            Error::Gravity("gravity_glm_sort_u32_ascending padded width overflow".into())
        })?;
        let shmem = checked_mul(
            padded,
            std::mem::size_of::<u32>(),
            "gravity_glm_sort_u32_ascending threadgroup memory",
        )?;
        let params = GlmSortU32Params {
            n: u32_arg(k, "gravity_glm_sort_u32_ascending n")?,
        };
        let input = score_ordered_indices.clone();
        let output = ascending_indices.clone();
        tcb.dispatch_threads(
            "gravity_glm_sort_u32_ascending",
            (TG, 1, 1),
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&input), 0);
                enc.set_buffer(1, Some(&output), 0);
                enc.set_bytes(
                    2,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem as u64);
            },
        )
    }

    /// Encode transitional expanded-cache sparse attention.
    ///
    /// `allow_idx` must contain unique positions in ascending position order
    /// to preserve the current host accumulation order. The stable-top-k
    /// output is score-ordered and must first pass through
    /// [`encode_sort_positions_ascending`].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_sparse_attention_expanded_ascending_allow(
        tcb: &mut TokenCommandBuffer<'_>,
        queries: &Buffer,
        keys: &Buffer,
        values: &Buffer,
        allow_idx: &Buffer,
        context: &Buffer,
        n_heads: usize,
        qk_dim: usize,
        v_dim: usize,
        n_keys: usize,
        n_allow: usize,
        scale: f32,
    ) -> Result<()> {
        if n_heads == 0 || qk_dim == 0 || v_dim == 0 {
            return Ok(());
        }
        require_f32(
            queries,
            0,
            checked_mul(n_heads, qk_dim, "gravity_glm_sparse_attn queries")?,
            "gravity_glm_sparse_attn queries",
        )?;
        require_f32(
            keys,
            0,
            checked_mul(
                checked_mul(n_keys, n_heads, "gravity_glm_sparse_attn keys")?,
                qk_dim,
                "gravity_glm_sparse_attn keys",
            )?,
            "gravity_glm_sparse_attn expanded keys",
        )?;
        require_f32(
            values,
            0,
            checked_mul(
                checked_mul(n_keys, n_heads, "gravity_glm_sparse_attn values")?,
                v_dim,
                "gravity_glm_sparse_attn values",
            )?,
            "gravity_glm_sparse_attn expanded values",
        )?;
        require_range(
            allow_idx,
            0,
            n_allow,
            std::mem::size_of::<u32>(),
            "gravity_glm_sparse_attn allow_idx",
        )?;
        require_f32(
            context,
            0,
            checked_mul(n_heads, v_dim, "gravity_glm_sparse_attn context")?,
            "gravity_glm_sparse_attn context",
        )?;
        let params = GlmSparseAttnParams {
            n_heads: u32_arg(n_heads, "gravity_glm_sparse_attn n_heads")?,
            qk_dim: u32_arg(qk_dim, "gravity_glm_sparse_attn qk_dim")?,
            v_dim: u32_arg(v_dim, "gravity_glm_sparse_attn v_dim")?,
            n_keys: u32_arg(n_keys, "gravity_glm_sparse_attn n_keys")?,
            n_allow: u32_arg(n_allow, "gravity_glm_sparse_attn n_allow")?,
            scale,
        };
        let grid_width = params
            .n_heads
            .checked_mul(TG)
            .ok_or_else(|| Error::Gravity("gravity_glm_sparse_attn grid overflow".into()))?;
        let shmem = checked_mul(
            n_allow.max(1),
            std::mem::size_of::<f32>(),
            "gravity_glm_sparse_attn threadgroup memory",
        )?;
        let qb = queries.clone();
        let kb = keys.clone();
        let vb = values.clone();
        let ab = allow_idx.clone();
        let cb = context.clone();
        tcb.dispatch_threads(
            "gravity_glm_sparse_attn",
            (grid_width, 1, 1),
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&qb), 0);
                enc.set_buffer(1, Some(&kb), 0);
                enc.set_buffer(2, Some(&vb), 0);
                enc.set_buffer(3, Some(&ab), 0);
                enc.set_buffer(4, Some(&cb), 0);
                enc.set_bytes(
                    5,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem as u64);
            },
        )
    }

    pub(super) fn encode_router_correction(
        tcb: &mut TokenCommandBuffer<'_>,
        logits: &Buffer,
        bias: &Buffer,
        scores: &Buffer,
        corrected: &Buffer,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        require_f32(logits, 0, n, "gravity_glm_router_correct logits")?;
        require_f32(bias, 0, n, "gravity_glm_router_correct bias")?;
        require_f32(scores, 0, n, "gravity_glm_router_correct scores")?;
        require_f32(corrected, 0, n, "gravity_glm_router_correct corrected")?;
        let n = u32_arg(n, "gravity_glm_router_correct n")?;
        let grid = grid_1d(n, "gravity_glm_router_correct")?;
        let lb = logits.clone();
        let bb = bias.clone();
        let sb = scores.clone();
        let cb = corrected.clone();
        tcb.dispatch_threads("gravity_glm_router_correct", grid, (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&lb), 0);
            enc.set_buffer(1, Some(&bb), 0);
            enc.set_buffer(2, Some(&sb), 0);
            enc.set_buffer(3, Some(&cb), 0);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_router_select_noaux(
        tcb: &mut TokenCommandBuffer<'_>,
        logits: &Buffer,
        bias: &Buffer,
        scores: &Buffer,
        corrected: &Buffer,
        expert_indices: &Buffer,
        expert_weights: &Buffer,
        expert_exec_slots: &Buffer,
        n_experts: usize,
        n_group: usize,
        topk_group: usize,
        experts_per_token: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
    ) -> Result<()> {
        const MAX_GROUPS: usize = 64;
        const MAX_EXPERTS_PER_TOKEN: usize = 64;
        if n_experts == 0
            || n_group == 0
            || n_experts % n_group != 0
            || topk_group == 0
            || topk_group > n_group
            || experts_per_token == 0
            || n_group > MAX_GROUPS
            || experts_per_token > MAX_EXPERTS_PER_TOKEN
        {
            return Err(Error::Gravity(format!(
                "gravity_glm_router_select_noaux_f32 unsupported geometry: experts={n_experts} groups={n_group} topk_group={topk_group} experts_per_token={experts_per_token}"
            )));
        }
        let selectable_experts = topk_group.checked_mul(n_experts / n_group).ok_or_else(|| {
            Error::Gravity(
                "gravity_glm_router_select_noaux_f32 selectable geometry overflow".into(),
            )
        })?;
        if experts_per_token > selectable_experts {
            return Err(Error::Gravity(format!(
                "gravity_glm_router_select_noaux_f32 experts_per_token={experts_per_token} exceeds selected-group capacity {selectable_experts}"
            )));
        }
        require_f32(
            logits,
            0,
            n_experts,
            "gravity_glm_router_select_noaux_f32 logits",
        )?;
        require_f32(
            bias,
            0,
            n_experts,
            "gravity_glm_router_select_noaux_f32 bias",
        )?;
        require_f32(
            scores,
            0,
            n_experts,
            "gravity_glm_router_select_noaux_f32 scores",
        )?;
        require_f32(
            corrected,
            0,
            n_experts,
            "gravity_glm_router_select_noaux_f32 corrected",
        )?;
        require_range(
            expert_indices,
            0,
            experts_per_token,
            std::mem::size_of::<u32>(),
            "gravity_glm_router_select_noaux_f32 expert indices",
        )?;
        require_f32(
            expert_weights,
            0,
            experts_per_token,
            "gravity_glm_router_select_noaux_f32 expert weights",
        )?;
        require_range(
            expert_exec_slots,
            0,
            experts_per_token,
            std::mem::size_of::<u32>(),
            "gravity_glm_router_select_noaux_f32 expert execution slots",
        )?;
        let params = GlmRouterSelectParams {
            n_experts: u32_arg(n_experts, "gravity_glm_router_select_noaux_f32 n_experts")?,
            n_group: u32_arg(n_group, "gravity_glm_router_select_noaux_f32 n_group")?,
            topk_group: u32_arg(topk_group, "gravity_glm_router_select_noaux_f32 topk_group")?,
            experts_per_token: u32_arg(
                experts_per_token,
                "gravity_glm_router_select_noaux_f32 experts_per_token",
            )?,
            norm_topk_prob: u32::from(norm_topk_prob),
            routed_scaling_factor,
        };
        let lb = logits.clone();
        let bb = bias.clone();
        let sb = scores.clone();
        let cb = corrected.clone();
        let ib = expert_indices.clone();
        let wb = expert_weights.clone();
        let eb = expert_exec_slots.clone();
        tcb.dispatch_threads(
            "gravity_glm_router_select_noaux_f32",
            (1, 1, 1),
            (1, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&lb), 0);
                enc.set_buffer(1, Some(&bb), 0);
                enc.set_buffer(2, Some(&sb), 0);
                enc.set_buffer(3, Some(&cb), 0);
                enc.set_buffer(4, Some(&ib), 0);
                enc.set_buffer(5, Some(&wb), 0);
                enc.set_buffer(6, Some(&eb), 0);
                enc.set_bytes(
                    7,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
            },
        )
    }

    /// Encode the residual add `x[i] += y[i]`.
    ///
    /// `x` and `y` may be the same Metal buffer: the shader assigns exactly
    /// one thread to each element and performs no cross-element access.
    pub(super) fn encode_residual_add_inplace(
        tcb: &mut TokenCommandBuffer<'_>,
        x: &Buffer,
        y: &Buffer,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let n = u32_arg(n, "gravity_add_inplace_f32 n")?;
        let grid = grid_1d(n, "gravity_add_inplace_f32")?;
        require_f32(x, 0, n as usize, "gravity_add_inplace_f32 x")?;
        require_f32(y, 0, n as usize, "gravity_add_inplace_f32 y")?;
        let xb = x.clone();
        let yb = y.clone();
        tcb.dispatch_threads("gravity_add_inplace_f32", grid, (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&xb), 0);
            enc.set_buffer(1, Some(&yb), 0);
            enc.set_bytes(2, 4, &n as *const u32 as *const _);
        })
    }

    pub(super) fn encode_zero(
        tcb: &mut TokenCommandBuffer<'_>,
        buffer: &Buffer,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        require_f32(buffer, 0, n, "gravity_zero_f32 buffer")?;
        let n = u32_arg(n, "gravity_zero_f32 n")?;
        let grid = grid_1d(n, "gravity_zero_f32")?;
        let xb = buffer.clone();
        tcb.dispatch_threads("gravity_zero_f32", grid, (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&xb), 0);
            enc.set_bytes(1, 4, &n as *const u32 as *const _);
        })
    }
}

// ── Expert-wave (flagged; default path above is untouched) ─────────────────

fn encode_silu_mul_f32(
    tcb: &mut TokenCommandBuffer<'_>,
    gate: &Buffer,
    up: &Buffer,
    out: &Buffer,
    n: u32,
) -> Result<()> {
    crate::cost_ledger::record_source_modelled_operations(
        (n as u64).saturating_mul(4),
        0,
        0,
        n as u64,
        0,
    );
    const TG: u32 = 256;
    let n_u = n;
    let g = gate.clone();
    let u = up.clone();
    let o = out.clone();
    tcb.dispatch_threads(
        "gravity_silu_mul_f32",
        (n.div_ceil(TG) * TG, 1, 1),
        (TG, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&g), 0);
            enc.set_buffer(1, Some(&u), 0);
            enc.set_buffer(2, Some(&o), 0);
            enc.set_bytes(3, 4, &n_u as *const u32 as *const _);
        },
    )
}

fn encode_axpy_f32(
    tcb: &mut TokenCommandBuffer<'_>,
    y: &Buffer,
    x: &Buffer,
    scale: f32,
    n: u32,
) -> Result<()> {
    crate::cost_ledger::record_source_modelled_operations((n as u64).saturating_mul(2), 0, 0, 0, 0);
    const TG: u32 = 256;
    let s = scale;
    let n_u = n;
    let yb = y.clone();
    let xb = x.clone();
    tcb.dispatch_threads(
        "gravity_axpy_f32",
        (n.div_ceil(TG) * TG, 1, 1),
        (TG, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&yb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_bytes(2, 4, &s as *const f32 as *const _);
            enc.set_bytes(3, 4, &n_u as *const u32 as *const _);
        },
    )
}

fn encode_pq_matvec_device(
    tcb: &mut TokenCommandBuffer<'_>,
    codebooks: &Buffer,
    codes: &Buffer,
    params: crate::gravity_glm::gpu::PqParams,
    x: &Buffer,
    y: &Buffer,
) -> Result<()> {
    const TG: u32 = 256;
    let n_tg = params.rows.div_ceil(8);
    let p = params;
    let cb = codebooks.clone();
    let co = codes.clone();
    let xb = x.clone();
    let yb = y.clone();
    tcb.dispatch_threads(
        "gravity_pq_matvec",
        (n_tg * TG, 1, 1),
        (TG, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&cb), 0);
            enc.set_buffer(1, Some(&co), 0);
            enc.set_buffer(2, Some(&xb), 0);
            enc.set_buffer(3, Some(&yb), 0);
            enc.set_bytes(
                4,
                std::mem::size_of_val(&p) as u64,
                &p as *const _ as *const _,
            );
        },
    )
}

/// Encode one weight matvec (device x → device y) into an open command buffer.
/// Host-native weights are applied immediately into `y` (no encode).
fn encode_weight_matvec(
    tcb: &mut TokenCommandBuffer<'_>,
    weights: &GpuWeightCache,
    name: &str,
    x: &Buffer,
    x_len: usize,
    y: &Buffer,
) -> Result<()> {
    crate::cost_ledger::record_matvec_call();
    let mut cache = weights.cache.lock().expect("gpu weight cache");
    weights.ensure_many_locked(&mut cache, &[name])?;
    let tensor = cache.get(name).expect("ensured");
    record_routed_tensor_representation(name, tensor);
    match tensor {
        GpuTensor::NativeCpu(w) => {
            crate::cost_ledger::record_active_bytes_for(name, (w.len() * 4) as u64);
            record_dense_matvec_ops((w.len() / x_len) as u64, x_len as u64);
            let x_host = read_f32(x, x_len);
            let y_host = matvec_dense(w, &x_host, name)?;
            write_f32(y, &y_host);
            Ok(())
        }
        GpuTensor::NativeGpuBf16 { buf, rows, cols } => {
            if x_len != *cols as usize {
                return Err(Error::Gravity(format!(
                    "expert-wave matvec {name}: x_len {x_len} != cols {cols}"
                )));
            }
            crate::cost_ledger::record_active_bytes_for(name, buf.length());
            record_dense_matvec_ops(*rows as u64, *cols as u64);
            encode_gemv_native_bf16_seq(tcb, buf, *rows, *cols, x, y)
        }
        GpuTensor::Pq {
            codebooks,
            codes,
            params,
        } => {
            if x_len != params.cols as usize {
                return Err(Error::Gravity(format!(
                    "expert-wave matvec {name}: x_len {x_len} != cols {}",
                    params.cols
                )));
            }
            crate::cost_ledger::record_active_bytes_for(name, codebooks.length() + codes.length());
            record_pq_matvec_ops(*params);
            encode_pq_matvec_device(tcb, codebooks, codes, *params, x, y)
        }
    }
}

/// Isolated device path: **gate + up → SiLU → down → weighted combine** in one
/// command buffer (one `commit_and_wait`). Flagged via
/// [`crate::gravity_glm::GPU_EXPERT_WAVE_ENV`]; never called from the default
/// resident path.
///
/// `scales[i]` multiplies prefix `i`'s down projection into the sum (MoE router
/// weights for routed experts, `1.0` for shared / dense). Accumulation order
/// matches the host: prefixes are already sorted ascending-expert then shared.
///
/// Requires every gate/up/down weight to be device-resident (`Pq` or
/// `NativeGpuBf16`). Host-native tensors fall back to a single host pass with
/// one wait tick (tiny fixtures without `HAWKING_GLM_GPU_LM_HEAD`); the pure
/// device path is what flagship PQ experts hit.
enum MlpWaveResult {
    /// Host-native fallback or the ordinary three-batch path: caller applies
    /// the residual add exactly as before.
    Host(Vec<f32>),
    /// Pure device wave appended the residual add before its existing commit.
    DeviceResidualApplied,
}

#[allow(clippy::too_many_arguments)]
fn moe_device_wave<'a>(
    weights: &GpuWeightCache,
    prefixes: &[String],
    scales: &[f32],
    x: &Buffer,
    residual: &Buffer,
    x_len: usize,
    pool: &ActPool,
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<MlpWaveResult> {
    if prefixes.is_empty() {
        return Ok(MlpWaveResult::Host(vec![0f32; x_len]));
    }
    if scales.len() != prefixes.len() {
        return Err(Error::Gravity(format!(
            "expert-wave: scales.len() {} != prefixes.len() {}",
            scales.len(),
            prefixes.len()
        )));
    }
    // Flush pending attention encodes; this path owns the next commit.
    commit(tcb.take(), waits)?;

    // Pin every projection for this layer before encoding so LRU cannot drop
    // a tensor mid-wave (same invariant as matvec_batch).
    let mut all_names: Vec<String> = Vec::with_capacity(prefixes.len() * 3);
    for p in prefixes {
        all_names.push(format!("{p}.gate_proj.weight"));
        all_names.push(format!("{p}.up_proj.weight"));
        all_names.push(format!("{p}.down_proj.weight"));
    }
    {
        let name_refs: Vec<&str> = all_names.iter().map(String::as_str).collect();
        let mut cache = weights.cache.lock().expect("gpu weight cache");
        weights.ensure_many_locked(&mut cache, &name_refs)?;
    }

    // Device-resident weights only on the pure CB path. Host-native needs the
    // host fallback (cannot encode silu→down dependence without a wait).
    let all_device = {
        let cache = weights.cache.lock().expect("gpu weight cache");
        all_names.iter().all(|n| {
            matches!(
                cache.get(n),
                Some(GpuTensor::Pq { .. }) | Some(GpuTensor::NativeGpuBf16 { .. })
            )
        })
    };
    if !all_device {
        return moe_wave_host_fallback(weights, prefixes, scales, x, x_len, waits)
            .map(MlpWaveResult::Host);
    }

    // Discover intermediate width from the first gate projection.
    let inter = {
        let cache = weights.cache.lock().expect("gpu weight cache");
        let gname = format!("{}.gate_proj.weight", prefixes[0]);
        match cache.get(&gname).expect("ensured gate") {
            GpuTensor::Pq { params, .. } => params.rows as usize,
            GpuTensor::NativeGpuBf16 { rows, .. } => *rows as usize,
            GpuTensor::NativeCpu(_) => {
                return Err(Error::Gravity(
                    "expert-wave: gate is NativeCpu after device check".into(),
                ));
            }
        }
    };

    let n_exp = prefixes.len();
    let scratch_guard = pool.ensure_expert_wave_scratch(ctx, n_exp, inter, x_len)?;
    let scratch = scratch_guard
        .as_ref()
        .expect("expert-wave scratch ensured above");

    // Preserve the original host-zero semantics while avoiding its temporary
    // `Vec`: this candidate changes resource lifetime, not dispatch shape.
    zero_f32(&scratch.combined, x_len)?;
    let mut wave = TokenCommandBuffer::new(ctx);
    let concurrent_projections = gpu_expert_wave_concurrent_enabled();
    if concurrent_projections {
        wave.begin_concurrent_group()?;
    }
    for (i, p) in prefixes.iter().enumerate() {
        encode_weight_matvec(
            &mut wave,
            weights,
            &format!("{p}.gate_proj.weight"),
            x,
            x_len,
            &scratch.gate[i],
        )?;
        encode_weight_matvec(
            &mut wave,
            weights,
            &format!("{p}.up_proj.weight"),
            x,
            x_len,
            &scratch.up[i],
        )?;
    }
    if concurrent_projections {
        wave.end_concurrent_group()?;
    }
    for i in 0..n_exp {
        encode_silu_mul_f32(
            &mut wave,
            &scratch.gate[i],
            &scratch.up[i],
            &scratch.act[i],
            inter as u32,
        )?;
    }
    if concurrent_projections {
        wave.begin_concurrent_group()?;
    }
    for (i, p) in prefixes.iter().enumerate() {
        encode_weight_matvec(
            &mut wave,
            weights,
            &format!("{p}.down_proj.weight"),
            &scratch.act[i],
            inter,
            &scratch.down[i],
        )?;
    }
    if concurrent_projections {
        wave.end_concurrent_group()?;
    }
    // Weighted combine in prefix order (associativity matches host).
    for i in 0..n_exp {
        encode_axpy_f32(
            &mut wave,
            &scratch.combined,
            &scratch.down[i],
            scales[i],
            x_len as u32,
        )?;
    }
    // The expert output is consumed only by the residual add. Keep both on
    // device and append that exact elementwise operation before the wave's
    // existing commit; no new command buffer or wait.
    route_segment_primitives::encode_residual_add_inplace(
        &mut wave,
        residual,
        &scratch.combined,
        x_len,
    )?;

    // Encode/submit/sync + dispatch count fold at TCB commit when ledger on.
    wave.commit_and_wait()?;
    waits.set(waits.get().saturating_add(1));
    Ok(MlpWaveResult::DeviceResidualApplied)
}

/// Host-side gate/up/SiLU/down/combine used when expert weights are not
/// device-resident. Still bills **one** wait for the wave accounting contract
/// (flag on ⇒ collapsed MLP drain count); no intermediate readback loop.
fn moe_wave_host_fallback(
    weights: &GpuWeightCache,
    prefixes: &[String],
    scales: &[f32],
    x: &Buffer,
    x_len: usize,
    waits: &Cell<u64>,
) -> Result<Vec<f32>> {
    let x_host = read_f32(x, x_len);
    let mut combined = vec![0f32; x_len];
    for (p, &scale) in prefixes.iter().zip(scales.iter()) {
        let gate = weights.matvec(&format!("{p}.gate_proj.weight"), &x_host)?;
        let up = weights.matvec(&format!("{p}.up_proj.weight"), &x_host)?;
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let down = weights.matvec(&format!("{p}.down_proj.weight"), &act)?;
        for (c, d) in combined.iter_mut().zip(&down) {
            *c += *d * scale;
        }
    }
    // One wait tick: collapsed accounting for the flag-on path.
    waits.set(waits.get().saturating_add(1));
    Ok(combined)
}

/// Holds the long-lived resident state for a [`crate::gravity_glm::gpu::GravityGlmGpu`].
pub struct ResidentRuntime {
    pub session: Mutex<ResidentSession>,
    pub pool: ActPool,
}

impl ResidentRuntime {
    pub fn new(ctx: &MetalContext, arch: &GlmArch) -> Result<Self> {
        let compact_mla = gpu_compact_mla_enabled();
        let device_dsa = crate::gravity_glm::gpu_device_dsa_enabled();
        if device_dsa && !compact_mla {
            return Err(Error::Gravity(format!(
                "{} requires {}=1",
                crate::gravity_glm::GPU_DEVICE_DSA_ENV,
                crate::gravity_glm::GPU_COMPACT_MLA_ENV
            )));
        }
        Self::new_with_compact_mla(ctx, arch, compact_mla, device_dsa)
    }

    /// Construct the layout already admitted by the model opener. Capturing
    /// the flag once prevents a process-environment race from selecting a
    /// compact session after the header preflight decision was made.
    pub(crate) fn new_with_compact_mla(
        ctx: &MetalContext,
        arch: &GlmArch,
        compact_mla: bool,
        device_dsa: bool,
    ) -> Result<Self> {
        let layout = if compact_mla {
            ResidentAttentionLayout::Compact
        } else {
            ResidentAttentionLayout::Expanded
        };
        let session = ResidentSession::new_with_layout(
            ctx,
            arch,
            RESIDENT_RUNTIME_INITIAL_KV_CAPACITY_TOKENS,
            layout,
            device_dsa,
        )?;
        Ok(Self {
            session: Mutex::new(session),
            pool: ActPool::new(ctx, arch)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::route_segment_primitives::*;
    use super::*;
    use crate::numeric_parity::{score_pair, Bounds};

    fn tiny_arch() -> GlmArch {
        GlmArch {
            n_layers: 1,
            hidden: 4,
            n_heads: 1,
            q_lora_rank: 2,
            kv_lora_rank: 2,
            qk_nope_head_dim: 1,
            qk_rope_head_dim: 1,
            v_head_dim: 1,
            index_n_heads: 1,
            index_head_dim: 1,
            index_topk: 2,
            n_routed_experts: 2,
            n_group: 1,
            topk_group: 1,
            num_experts_per_tok: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            vocab_size: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            indexer_types: vec!["full".into()],
            mlp_layer_types: vec!["dense".into()],
        }
    }

    fn f32_buffer(ctx: &MetalContext, values: &[f32]) -> Buffer {
        let buffer = ctx
            .new_buffer_checked(values.len() * std::mem::size_of::<f32>())
            .expect("f32 test buffer");
        write_f32(&buffer, values);
        buffer
    }

    fn filled_f32_buffer(ctx: &MetalContext, len: usize, value: f32) -> Buffer {
        f32_buffer(ctx, &vec![value; len])
    }

    fn u32_buffer(ctx: &MetalContext, values: &[u32]) -> Buffer {
        let buffer = ctx
            .new_buffer_checked(values.len() * std::mem::size_of::<u32>())
            .expect("u32 test buffer");
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                buffer.contents() as *mut u32,
                values.len(),
            );
        }
        buffer
    }

    fn empty_u8_buffer(ctx: &MetalContext, len: usize) -> Buffer {
        ctx.new_buffer_checked(len).expect("u8 test buffer")
    }

    fn f16_buffer(ctx: &MetalContext, values: &[half::f16]) -> Buffer {
        ctx.new_buffer_with_bytes_checked(bytemuck::cast_slice(values))
            .expect("f16 test buffer")
    }

    fn assert_v21_pair(label: &str, host: &[f32], device: &[f32], reference: &[f64]) {
        let score = score_pair(host, device, reference, &Bounds::continuous_only());
        assert!(
            score.pass,
            "{label}: Numeric Parity V2.1 failure against FP64 authority; host={:?}, device={:?}",
            score.host.failures, score.device.failures
        );
    }

    fn topk_desc_f64(values: &[f64], k: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..values.len()).collect();
        indices.sort_by(|&a, &b| {
            values[b]
                .partial_cmp(&values[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        indices.truncate(k);
        indices
    }

    fn deterministic_fixture_f32(mut state: u32, len: usize, scale: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = ((state >> 8) as f32) * (1.0 / 8_388_608.0) - 1.0;
            out.push(unit * scale);
        }
        out
    }

    fn direct_u8_pq_tensor(
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        salt: usize,
    ) -> (GpuTensor, Vec<f32>) {
        const DIM: usize = 32;
        const CARD: usize = 256;
        assert_eq!(cols % DIM, 0);
        let nchunk = cols / DIM;
        let codebooks: Vec<half::f16> = (0..CARD * DIM)
            .map(|flat| {
                let code = flat / DIM;
                let element = flat % DIM;
                let positive = ((code * 17 + element * 13 + salt * 19) % 47) + 1;
                half::f16::from_f32(positive as f32 * (1.0 / 256.0))
            })
            .collect();
        let mut codes = vec![0u8; rows * nchunk + 4];
        let mut dense = vec![0.0f32; rows * cols];
        for row in 0..rows {
            for chunk in 0..nchunk {
                let code = (row * nchunk + chunk + salt * 7) % CARD;
                codes[row * nchunk + chunk] = code as u8;
                for element in 0..DIM {
                    dense[row * cols + chunk * DIM + element] =
                        codebooks[code * DIM + element].to_f32();
                }
            }
        }
        let codebooks = f16_buffer(ctx, &codebooks);
        let codes = ctx
            .new_buffer_with_bytes_checked(&codes)
            .expect("direct-u8 codes");
        (
            GpuTensor::Pq {
                codebooks,
                codes,
                params: crate::gravity_glm::gpu::PqParams {
                    dim: DIM as u32,
                    subspaces: 1,
                    sub: DIM as u32,
                    card: CARD as u32,
                    rows: rows as u32,
                    cols: cols as u32,
                    nchunk: nchunk as u32,
                    bits: 8,
                },
            },
            dense,
        )
    }

    fn gpu_tensor_bytes(tensor: &GpuTensor) -> u64 {
        match tensor {
            GpuTensor::Pq {
                codebooks, codes, ..
            } => codebooks.length() + codes.length(),
            GpuTensor::NativeGpuBf16 { buf, .. } => buf.length(),
            GpuTensor::NativeCpu(values) => (values.len() * std::mem::size_of::<f32>()) as u64,
        }
    }

    fn fixture_mlp_f32(
        gate_weights: &[f32],
        up_weights: &[f32],
        down_weights: &[f32],
        hidden: usize,
        intermediate: usize,
        x: &[f32],
    ) -> Vec<f32> {
        let gate = matvec_dense(gate_weights, x, "fixture gate").expect("fixture gate");
        let up = matvec_dense(up_weights, x, "fixture up").expect("fixture up");
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| (gate / (1.0 + (-gate).exp())) * up)
            .collect();
        assert_eq!(act.len(), intermediate);
        assert_eq!(down_weights.len(), hidden * intermediate);
        matvec_dense(down_weights, &act, "fixture down").expect("fixture down")
    }

    fn fixture_mlp_f64(
        gate_weights: &[f32],
        up_weights: &[f32],
        down_weights: &[f32],
        hidden: usize,
        intermediate: usize,
        x: &[f32],
    ) -> Vec<f64> {
        let matvec = |weights: &[f32], cols: usize, input: &[f64]| {
            weights
                .chunks_exact(cols)
                .map(|row| {
                    row.iter()
                        .zip(input)
                        .map(|(&weight, &activation)| weight as f64 * activation)
                        .sum::<f64>()
                })
                .collect::<Vec<_>>()
        };
        let x64: Vec<f64> = x.iter().map(|&value| value as f64).collect();
        let gate = matvec(gate_weights, hidden, &x64);
        let up = matvec(up_weights, hidden, &x64);
        let act: Vec<f64> = gate
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| (gate / (1.0 + (-gate).exp())) * up)
            .collect();
        assert_eq!(act.len(), intermediate);
        matvec(down_weights, intermediate, &act)
    }

    #[test]
    fn device_expert_table_hit_is_indirect_leased_and_miss_is_fail_closed() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const HIDDEN: usize = 32;
        const INTERMEDIATE: usize = 32;
        const GENERATION: u32 = 7;
        const PREFIX: &str = "model.layers.0.mlp";

        let mut cache = BoundedLru::new(110_000).expect("bounded cache");
        let mut items = Vec::new();
        let mut gate_authorities = std::collections::HashMap::new();
        for &expert in &[0usize, 2usize] {
            for (projection, rows, cols, projection_salt) in [
                ("gate_proj", INTERMEDIATE, HIDDEN, 1usize),
                ("up_proj", INTERMEDIATE, HIDDEN, 2usize),
                ("down_proj", HIDDEN, INTERMEDIATE, 3usize),
            ] {
                let (tensor, dense) =
                    direct_u8_pq_tensor(&ctx, rows, cols, expert * 11 + projection_salt);
                if projection == "gate_proj" {
                    gate_authorities.insert(expert, dense);
                }
                let bytes = gpu_tensor_bytes(&tensor);
                items.push((
                    format!("{PREFIX}.experts.{expert}.{projection}.weight"),
                    tensor,
                    bytes,
                ));
            }
        }
        cache
            .admit_pinned(items, &std::collections::HashSet::new())
            .expect("admit selected triplets");
        let lease = build_device_expert_table_snapshot(&ctx, &cache, PREFIX, 4, GENERATION)
            .expect("immutable expert table");
        assert_eq!(std::mem::size_of::<DeviceExpertTensorRef>(), 56);
        assert_eq!(std::mem::size_of::<DeviceExpertTriplet>(), 176);
        assert_eq!(
            DEVICE_EXPERT_TABLE_MAX_EXPERTS * std::mem::size_of::<DeviceExpertTriplet>(),
            45_056
        );
        assert_eq!(lease.table.length(), (4 * 176) as u64);
        assert_eq!(lease.ready_entries, 2);
        assert_eq!(lease.resources.len(), 12);
        let table_entries = unsafe {
            std::slice::from_raw_parts(
                lease.table.contents() as *const DeviceExpertTriplet,
                lease.n_experts,
            )
        };
        assert_eq!(table_entries[0].ready_mask, DEVICE_EXPERT_TRIPLET_READY);
        assert_eq!(table_entries[1], DeviceExpertTriplet::default());
        assert_eq!(table_entries[2].generation, GENERATION);

        // Logical LRU eviction after the immutable snapshot cannot free a
        // leased Metal resource before this command completes.
        let evicting = GpuTensor::NativeGpuBf16 {
            buf: ctx.new_buffer_checked(20_000).expect("evicting buffer"),
            rows: 100,
            cols: 100,
        };
        let evicting_bytes = gpu_tensor_bytes(&evicting);
        cache
            .admit_pinned(
                vec![("unrelated.native.weight".into(), evicting, evicting_bytes)],
                &std::collections::HashSet::new(),
            )
            .expect("bounded eviction");
        assert!(
            !cache.contains(&format!("{PREFIX}.experts.0.gate_proj.weight")),
            "oldest logical entry should have been evicted"
        );
        assert!(cache.high_water_bytes() <= cache.budget_bytes());

        // Score-ranked IDs [2,0], device execution slots [1,0]: execution
        // position zero must indirectly address expert 0 without a host ID
        // read or concrete expert buffer binding.
        let expert_indices = u32_buffer(&ctx, &[2, 0]);
        let execution_slots = u32_buffer(&ctx, &[1, 0]);
        let miss_mask = u32_buffer(&ctx, &[u32::MAX]);
        let x_values: Vec<f32> = deterministic_fixture_f32(0x51A7_2026, HIDDEN, 0.25)
            .into_iter()
            .map(|value| value.abs() + 0.125)
            .collect();
        let x = f32_buffer(&ctx, &x_values);
        let y = filled_f32_buffer(&ctx, INTERMEDIATE, -9_999.0);
        let mut hit = TokenCommandBuffer::new(&ctx);
        encode_device_expert_table_validate(
            &mut hit,
            &lease,
            &expert_indices,
            &execution_slots,
            &miss_mask,
            2,
            HIDDEN,
            INTERMEDIATE,
            DEVICE_EXPERT_TENSOR_KIND_PQ,
        )
        .expect("validate resident hit");
        encode_device_expert_table_pq_matvec(
            &mut hit,
            &lease,
            &expert_indices,
            &execution_slots,
            &miss_mask,
            2,
            0,
            0,
            &x,
            INTERMEDIATE,
            HIDDEN,
            &y,
        )
        .expect("indirect gate projection");
        assert_eq!(hit.dispatch_count(), 2);
        hit.commit_and_wait().expect("resident-hit command");
        assert_eq!(read_u32(&miss_mask, 1), vec![0]);

        let weights = &gate_authorities[&0];
        let host = matvec_dense(weights, &x_values, "expert-0 gate authority")
            .expect("host gate comparator");
        let authority: Vec<f64> = weights
            .chunks_exact(HIDDEN)
            .map(|row| {
                row.iter()
                    .zip(&x_values)
                    .map(|(&weight, &activation)| weight as f64 * activation as f64)
                    .sum()
            })
            .collect();
        let device = read_f32(&y, INTERMEDIATE);
        let score = score_pair(&host, &device, &authority, &Bounds::continuous_only());
        eprintln!(
            "device expert table indirect gate: rel_l2={:.3e} meaningful={:.3e} \
             greedy={} top5={}",
            score.device.continuous.relative_l2,
            score.device.continuous.max_meaningful_rel,
            score.device.discrete.greedy_match,
            score.device.discrete.top_k_exact_match
        );
        assert!(
            score.pass,
            "device expert table indirect gate failed V2.1: host={:?}, device={:?}",
            score.host.failures, score.device.failures
        );

        // Expert 1 has no ready triplet. Validation writes bit 0 and the
        // following indirect projection must leave its destination untouched.
        let missing_indices = u32_buffer(&ctx, &[2, 1]);
        let missing_slots = u32_buffer(&ctx, &[1, 0]);
        let missing_mask = u32_buffer(&ctx, &[u32::MAX]);
        let sentinel: Vec<f32> = (0..INTERMEDIATE)
            .map(|index| 1_000.0 + index as f32)
            .collect();
        let missing_y = f32_buffer(&ctx, &sentinel);
        let before_bits: Vec<u32> = sentinel.iter().map(|value| value.to_bits()).collect();
        let mut miss = TokenCommandBuffer::new(&ctx);
        encode_device_expert_table_validate(
            &mut miss,
            &lease,
            &missing_indices,
            &missing_slots,
            &missing_mask,
            2,
            HIDDEN,
            INTERMEDIATE,
            DEVICE_EXPERT_TENSOR_KIND_PQ,
        )
        .expect("validate resident miss");
        encode_device_expert_table_pq_matvec(
            &mut miss,
            &lease,
            &missing_indices,
            &missing_slots,
            &missing_mask,
            2,
            0,
            0,
            &x,
            INTERMEDIATE,
            HIDDEN,
            &missing_y,
        )
        .expect("suppressed missing projection");
        miss.commit_and_wait().expect("resident-miss command");
        assert_eq!(read_u32(&missing_mask, 1), vec![1]);
        let after_bits: Vec<u32> = read_f32(&missing_y, INTERMEDIATE)
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(
            after_bits, before_bits,
            "a table miss must not mutate the projection destination"
        );
    }

    #[test]
    fn device_expert_table_complete_wave_is_ordered_and_residual_miss_is_fail_closed() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const HIDDEN: usize = 32;
        const INTERMEDIATE: usize = 32;
        const GENERATION: u32 = 11;
        const PREFIX: &str = "model.layers.0.mlp";

        let mut cache = BoundedLru::new(200_000).expect("bounded routed cache");
        let mut items = Vec::new();
        let mut dense = std::collections::HashMap::<String, Vec<f32>>::new();
        for &expert in &[0usize, 2usize] {
            for (projection, rows, cols, projection_salt) in [
                ("gate_proj", INTERMEDIATE, HIDDEN, 1usize),
                ("up_proj", INTERMEDIATE, HIDDEN, 2usize),
                ("down_proj", HIDDEN, INTERMEDIATE, 3usize),
            ] {
                let key = format!("{expert}.{projection}");
                let (tensor, authority) =
                    direct_u8_pq_tensor(&ctx, rows, cols, expert * 11 + projection_salt);
                dense.insert(key, authority);
                let bytes = gpu_tensor_bytes(&tensor);
                items.push((
                    format!("{PREFIX}.experts.{expert}.{projection}.weight"),
                    tensor,
                    bytes,
                ));
            }
        }
        cache
            .admit_pinned(items, &std::collections::HashSet::new())
            .expect("admit routed triplets");
        let routed_lease = build_device_expert_table_snapshot(&ctx, &cache, PREFIX, 4, GENERATION)
            .expect("routed table snapshot");

        let (shared_gate, shared_gate_dense) = direct_u8_pq_tensor(&ctx, INTERMEDIATE, HIDDEN, 101);
        let (shared_up, shared_up_dense) = direct_u8_pq_tensor(&ctx, INTERMEDIATE, HIDDEN, 102);
        let (shared_down, shared_down_dense) = direct_u8_pq_tensor(&ctx, HIDDEN, INTERMEDIATE, 103);
        let shared_lease = build_single_device_expert_snapshot(
            &ctx,
            &shared_gate,
            &shared_up,
            &shared_down,
            GENERATION,
        )
        .expect("shared expert snapshot");

        // Router score order is expert [2,0], while execution order is [0,2].
        // The weights remain aligned to score slots: expert 0 receives 0.7.
        let expert_indices = u32_buffer(&ctx, &[2, 0]);
        let execution_slots = u32_buffer(&ctx, &[1, 0]);
        let expert_weights = f32_buffer(&ctx, &[0.3, 0.7]);
        let shared_indices = u32_buffer(&ctx, &[0]);
        let shared_slots = u32_buffer(&ctx, &[0]);
        let x_values: Vec<f32> = deterministic_fixture_f32(0xE771_2026, HIDDEN, 0.2)
            .into_iter()
            .map(|value| value.abs() + 0.125)
            .collect();
        let x = f32_buffer(&ctx, &x_values);
        let routed_gate: Vec<Buffer> = (0..2)
            .map(|_| filled_f32_buffer(&ctx, INTERMEDIATE, -7_000.0))
            .collect();
        let routed_up: Vec<Buffer> = (0..2)
            .map(|_| filled_f32_buffer(&ctx, INTERMEDIATE, -7_100.0))
            .collect();
        let routed_act: Vec<Buffer> = (0..2)
            .map(|_| filled_f32_buffer(&ctx, INTERMEDIATE, -7_200.0))
            .collect();
        let routed_down: Vec<Buffer> = (0..2)
            .map(|_| filled_f32_buffer(&ctx, HIDDEN, -7_300.0))
            .collect();
        let shared_gate_out = filled_f32_buffer(&ctx, INTERMEDIATE, -7_400.0);
        let shared_up_out = filled_f32_buffer(&ctx, INTERMEDIATE, -7_500.0);
        let shared_act = filled_f32_buffer(&ctx, INTERMEDIATE, -7_600.0);
        let shared_down_out = filled_f32_buffer(&ctx, HIDDEN, -7_700.0);
        let combined = filled_f32_buffer(&ctx, HIDDEN, -7_800.0);
        let residual_values = deterministic_fixture_f32(0x5E51_DA1, HIDDEN, 0.1);
        let residual = f32_buffer(&ctx, &residual_values);

        let encode_wave = |wave: &mut TokenCommandBuffer<'_>,
                           selected_indices: &Buffer,
                           selected_slots: &Buffer,
                           miss_mask: &Buffer|
         -> Result<()> {
            encode_device_expert_table_validate(
                wave,
                &routed_lease,
                selected_indices,
                selected_slots,
                miss_mask,
                2,
                HIDDEN,
                INTERMEDIATE,
                DEVICE_EXPERT_TENSOR_KIND_PQ,
            )?;
            encode_device_expert_table_zero(wave, &combined, miss_mask, HIDDEN)?;
            for execution_position in 0..2 {
                encode_device_expert_table_pq_matvec(
                    wave,
                    &routed_lease,
                    selected_indices,
                    selected_slots,
                    miss_mask,
                    2,
                    execution_position,
                    0,
                    &x,
                    INTERMEDIATE,
                    HIDDEN,
                    &routed_gate[execution_position],
                )?;
                encode_device_expert_table_pq_matvec(
                    wave,
                    &routed_lease,
                    selected_indices,
                    selected_slots,
                    miss_mask,
                    2,
                    execution_position,
                    1,
                    &x,
                    INTERMEDIATE,
                    HIDDEN,
                    &routed_up[execution_position],
                )?;
                encode_device_expert_table_silu_mul(
                    wave,
                    &routed_gate[execution_position],
                    &routed_up[execution_position],
                    &routed_act[execution_position],
                    miss_mask,
                    INTERMEDIATE,
                )?;
                encode_device_expert_table_pq_matvec(
                    wave,
                    &routed_lease,
                    selected_indices,
                    selected_slots,
                    miss_mask,
                    2,
                    execution_position,
                    2,
                    &routed_act[execution_position],
                    HIDDEN,
                    INTERMEDIATE,
                    &routed_down[execution_position],
                )?;
                encode_device_expert_table_axpy(
                    wave,
                    &combined,
                    &routed_down[execution_position],
                    &expert_weights,
                    selected_slots,
                    miss_mask,
                    HIDDEN,
                    2,
                    execution_position,
                    true,
                )?;
            }

            // Shared expert is host-known and always scheduled after all
            // routed execution positions, but its pointers are still leased
            // and indirectly dereferenced through the same triplet ABI.
            encode_device_expert_table_pq_matvec(
                wave,
                &shared_lease,
                &shared_indices,
                &shared_slots,
                miss_mask,
                1,
                0,
                0,
                &x,
                INTERMEDIATE,
                HIDDEN,
                &shared_gate_out,
            )?;
            encode_device_expert_table_pq_matvec(
                wave,
                &shared_lease,
                &shared_indices,
                &shared_slots,
                miss_mask,
                1,
                0,
                1,
                &x,
                INTERMEDIATE,
                HIDDEN,
                &shared_up_out,
            )?;
            encode_device_expert_table_silu_mul(
                wave,
                &shared_gate_out,
                &shared_up_out,
                &shared_act,
                miss_mask,
                INTERMEDIATE,
            )?;
            encode_device_expert_table_pq_matvec(
                wave,
                &shared_lease,
                &shared_indices,
                &shared_slots,
                miss_mask,
                1,
                0,
                2,
                &shared_act,
                HIDDEN,
                INTERMEDIATE,
                &shared_down_out,
            )?;
            encode_device_expert_table_axpy(
                wave,
                &combined,
                &shared_down_out,
                &expert_weights,
                &shared_slots,
                miss_mask,
                HIDDEN,
                1,
                0,
                false,
            )?;
            encode_device_expert_table_residual_add(wave, &residual, &combined, miss_mask, HIDDEN)
        };

        let miss_mask = u32_buffer(&ctx, &[u32::MAX]);
        let mut hit = TokenCommandBuffer::new(&ctx);
        encode_wave(&mut hit, &expert_indices, &execution_slots, &miss_mask)
            .expect("encode complete table hit");
        assert_eq!(hit.dispatch_count(), 18);
        hit.commit_and_wait().expect("complete table hit command");
        assert_eq!(read_u32(&miss_mask, 1), vec![0]);

        let routed_0_f32 = fixture_mlp_f32(
            &dense["0.gate_proj"],
            &dense["0.up_proj"],
            &dense["0.down_proj"],
            HIDDEN,
            INTERMEDIATE,
            &x_values,
        );
        let routed_2_f32 = fixture_mlp_f32(
            &dense["2.gate_proj"],
            &dense["2.up_proj"],
            &dense["2.down_proj"],
            HIDDEN,
            INTERMEDIATE,
            &x_values,
        );
        let shared_f32 = fixture_mlp_f32(
            &shared_gate_dense,
            &shared_up_dense,
            &shared_down_dense,
            HIDDEN,
            INTERMEDIATE,
            &x_values,
        );
        let mut host = residual_values.clone();
        for index in 0..HIDDEN {
            let mut expert_output = 0.0f32;
            expert_output += routed_0_f32[index] * 0.7f32;
            expert_output += routed_2_f32[index] * 0.3f32;
            expert_output += shared_f32[index];
            host[index] += expert_output;
        }

        let routed_0_f64 = fixture_mlp_f64(
            &dense["0.gate_proj"],
            &dense["0.up_proj"],
            &dense["0.down_proj"],
            HIDDEN,
            INTERMEDIATE,
            &x_values,
        );
        let routed_2_f64 = fixture_mlp_f64(
            &dense["2.gate_proj"],
            &dense["2.up_proj"],
            &dense["2.down_proj"],
            HIDDEN,
            INTERMEDIATE,
            &x_values,
        );
        let shared_f64 = fixture_mlp_f64(
            &shared_gate_dense,
            &shared_up_dense,
            &shared_down_dense,
            HIDDEN,
            INTERMEDIATE,
            &x_values,
        );
        let authority: Vec<f64> = (0..HIDDEN)
            .map(|index| {
                residual_values[index] as f64
                    + routed_0_f64[index] * 0.7f32 as f64
                    + routed_2_f64[index] * 0.3f32 as f64
                    + shared_f64[index]
            })
            .collect();
        let device = read_f32(&residual, HIDDEN);
        let score = score_pair(&host, &device, &authority, &Bounds::continuous_only());
        eprintln!(
            "device expert table complete wave: rel_l2={:.3e} meaningful={:.3e} \
             greedy={} top5={} dispatches=18 waits=1",
            score.device.continuous.relative_l2,
            score.device.continuous.max_meaningful_rel,
            score.device.discrete.greedy_match,
            score.device.discrete.top_k_exact_match
        );
        assert!(
            score.pass,
            "device expert table complete wave failed V2.1: host={:?}, device={:?}",
            score.host.failures, score.device.failures
        );

        // A routed miss must suppress every scratch write and the residual
        // mutation across the complete already-encoded wave.
        let all_outputs: Vec<&Buffer> = routed_gate
            .iter()
            .chain(&routed_up)
            .chain(&routed_act)
            .chain(&routed_down)
            .chain([
                &shared_gate_out,
                &shared_up_out,
                &shared_act,
                &shared_down_out,
                &combined,
                &residual,
            ])
            .collect();
        for (buffer_index, buffer) in all_outputs.iter().enumerate() {
            let sentinel: Vec<f32> = (0..HIDDEN)
                .map(|index| 10_000.0 + (buffer_index * HIDDEN + index) as f32)
                .collect();
            write_f32(buffer, &sentinel);
        }
        let before: Vec<Vec<u32>> = all_outputs
            .iter()
            .map(|buffer| {
                read_f32(buffer, HIDDEN)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect()
            })
            .collect();
        let missing_indices = u32_buffer(&ctx, &[2, 1]);
        let missing_slots = u32_buffer(&ctx, &[1, 0]);
        let missing_mask = u32_buffer(&ctx, &[u32::MAX]);
        let mut miss = TokenCommandBuffer::new(&ctx);
        encode_wave(&mut miss, &missing_indices, &missing_slots, &missing_mask)
            .expect("encode complete table miss");
        assert_eq!(miss.dispatch_count(), 18);
        miss.commit_and_wait().expect("complete table miss command");
        assert_eq!(read_u32(&missing_mask, 1), vec![1]);
        let after: Vec<Vec<u32>> = all_outputs
            .iter()
            .map(|buffer| {
                read_f32(buffer, HIDDEN)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect()
            })
            .collect();
        assert_eq!(
            after, before,
            "a table miss must suppress every expert scratch and residual write"
        );
    }

    #[test]
    fn route_segment_parameter_abis_are_frozen_and_ranges_fail_closed() {
        assert_eq!(std::mem::size_of::<GlmRopeParams>(), 16);
        assert_eq!(std::mem::offset_of!(GlmRopeParams, n_heads), 0);
        assert_eq!(std::mem::offset_of!(GlmRopeParams, rotary_dim), 4);
        assert_eq!(std::mem::offset_of!(GlmRopeParams, in_stride), 8);
        assert_eq!(std::mem::offset_of!(GlmRopeParams, out_stride), 12);

        assert_eq!(std::mem::size_of::<GlmMlaAppendParams>(), 20);
        assert_eq!(std::mem::offset_of!(GlmMlaAppendParams, n_heads), 0);
        assert_eq!(std::mem::offset_of!(GlmMlaAppendParams, pos), 16);

        assert_eq!(std::mem::size_of::<GlmMlaCompactAppendParams>(), 12);
        assert_eq!(
            std::mem::offset_of!(GlmMlaCompactAppendParams, latent_dim),
            0
        );
        assert_eq!(std::mem::offset_of!(GlmMlaCompactAppendParams, pos), 8);

        assert_eq!(std::mem::size_of::<GlmPqKTransposeParams>(), 28);
        assert_eq!(std::mem::offset_of!(GlmPqKTransposeParams, n_heads), 0);
        assert_eq!(std::mem::offset_of!(GlmPqKTransposeParams, latent_dim), 12);
        assert_eq!(std::mem::offset_of!(GlmPqKTransposeParams, pq_nchunk), 24);

        assert_eq!(std::mem::size_of::<GlmCompactRankedAttnParams>(), 24);
        assert_eq!(std::mem::offset_of!(GlmCompactRankedAttnParams, n_heads), 0);
        assert_eq!(
            std::mem::offset_of!(GlmCompactRankedAttnParams, n_allow),
            16
        );
        assert_eq!(std::mem::offset_of!(GlmCompactRankedAttnParams, scale), 20);

        assert_eq!(std::mem::size_of::<GlmPqVRowsParams>(), 32);
        assert_eq!(std::mem::offset_of!(GlmPqVRowsParams, n_heads), 0);
        assert_eq!(std::mem::offset_of!(GlmPqVRowsParams, value_row_offset), 8);
        assert_eq!(std::mem::offset_of!(GlmPqVRowsParams, latent_dim), 16);
        assert_eq!(std::mem::offset_of!(GlmPqVRowsParams, pq_nchunk), 28);

        assert_eq!(std::mem::size_of::<GlmBuildQParams>(), 12);
        assert_eq!(std::mem::offset_of!(GlmBuildQParams, qk_rope), 8);

        assert_eq!(std::mem::size_of::<GlmDsaParams>(), 24);
        assert_eq!(std::mem::offset_of!(GlmDsaParams, pos), 12);
        assert_eq!(std::mem::offset_of!(GlmDsaParams, dim_scale), 16);
        assert_eq!(std::mem::offset_of!(GlmDsaParams, head_scale), 20);

        assert_eq!(std::mem::size_of::<GlmTopkParams>(), 8);
        assert_eq!(std::mem::offset_of!(GlmTopkParams, k), 4);

        assert_eq!(std::mem::size_of::<GlmSortU32Params>(), 4);
        assert_eq!(std::mem::align_of::<GlmSortU32Params>(), 4);
        assert_eq!(std::mem::offset_of!(GlmSortU32Params, n), 0);

        assert_eq!(std::mem::size_of::<GlmSparseAttnParams>(), 24);
        assert_eq!(std::mem::offset_of!(GlmSparseAttnParams, n_allow), 16);
        assert_eq!(std::mem::offset_of!(GlmSparseAttnParams, scale), 20);

        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let one = f32_buffer(&ctx, &[1.0]);
        let mut tcb = TokenCommandBuffer::new(&ctx);
        let error = encode_rmsnorm(&mut tcb, &one, &one, &one, 2, 1e-6)
            .expect_err("undersized buffers must be rejected before encoding");
        assert!(error.to_string().contains("byte range"));
        assert_eq!(tcb.dispatch_count(), 0);
    }

    #[test]
    fn route_segment_pq_k_transpose_is_deterministic_and_passes_v21() {
        let shader = include_str!("../shaders/gravity_pq.metal");
        assert!(shader.contains("kernel void gravity_pq_k_transpose_heads("));
        let registry = include_str!("metal/mod.rs");
        assert!(registry
            .contains("\"gravity_pq_k_transpose_heads\" => \"gravity_pq_k_transpose_heads\""));

        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let (n_heads, key_rows, row_stride) = (2usize, 3usize, 5usize);
        let (latent_dim, pq_dim, pq_sub, pq_card, pq_nchunk) =
            (8usize, 4usize, 4usize, 256usize, 2usize);
        let codebook_prefix = [
            0.25, -0.5, 0.75, 1.0, -1.0, 0.125, 0.5, -0.25, 1.5, -0.75, 0.25, 0.625, -0.375, 1.25,
            -1.5, 0.875,
        ];
        let mut codebook = vec![half::f16::ZERO; pq_card * pq_sub];
        for (dst, value) in codebook.iter_mut().zip(codebook_prefix) {
            *dst = half::f16::from_f32(value);
        }
        let rows_touched = (n_heads - 1) * row_stride + key_rows;
        let codes: Vec<u8> = (0..rows_touched * pq_nchunk)
            .map(|index| ((index * 3 + index / 2) % 4) as u8)
            .collect();
        let query = vec![0.75, -1.25, 0.5, -0.625, 1.5, 0.25];
        let codebookb = f16_buffer(&ctx, &codebook);
        let codesb = ctx
            .new_buffer_with_bytes_checked(&codes)
            .expect("PQ code test buffer");
        let queryb = f32_buffer(&ctx, &query);
        let output_a = filled_f32_buffer(&ctx, n_heads * latent_dim, f32::NAN);
        let output_b = filled_f32_buffer(&ctx, n_heads * latent_dim, f32::NAN);

        let mut tcb = TokenCommandBuffer::new(&ctx);
        for output in [&output_a, &output_b] {
            encode_pq_k_transpose_heads(
                &mut tcb, &codebookb, &codesb, &queryb, output, n_heads, key_rows, row_stride,
                latent_dim, pq_dim, pq_sub, pq_card, 8, pq_nchunk,
            )
            .expect("encode direct PQ K transpose");
        }
        assert_eq!(tcb.dispatch_count(), 2);
        tcb.commit_and_wait()
            .expect("PQ K transpose command buffer");

        let mut host = vec![0.0f32; n_heads * latent_dim];
        let mut authority = vec![0.0f64; n_heads * latent_dim];
        for head in 0..n_heads {
            for col in 0..latent_dim {
                let chunk = col / pq_dim;
                let within = col % pq_dim;
                let mut host_acc = 0.0f32;
                let mut authority_acc = 0.0f64;
                for key_row in 0..key_rows {
                    let row = head * row_stride + key_row;
                    let code = codes[row * pq_nchunk + chunk] as usize;
                    let weight = codebook[code * pq_sub + within].to_f32();
                    let q = query[head * key_rows + key_row];
                    host_acc = weight.mul_add(q, host_acc);
                    authority_acc += weight as f64 * q as f64;
                }
                host[head * latent_dim + col] = host_acc;
                authority[head * latent_dim + col] = authority_acc;
            }
        }
        let device_a = read_f32(&output_a, host.len());
        let device_b = read_f32(&output_b, host.len());
        assert_eq!(
            device_a, device_b,
            "one-thread-per-output reduction must be bit-stable"
        );
        assert_v21_pair("direct PQ K transpose", &host, &device_a, &authority);

        let mut rejected = TokenCommandBuffer::new(&ctx);
        let error = encode_pq_k_transpose_heads(
            &mut rejected,
            &codebookb,
            &codesb,
            &queryb,
            &output_a,
            n_heads,
            key_rows,
            row_stride,
            latent_dim,
            pq_dim,
            2,
            pq_card,
            8,
            pq_nchunk,
        )
        .expect_err("multi-subspace-like geometry must fail before dispatch");
        assert!(error.to_string().contains("dim == sub"));
        assert_eq!(rejected.dispatch_count(), 0);

        let error = encode_pq_k_transpose_heads(
            &mut rejected,
            &codebookb,
            &codesb,
            &queryb,
            &output_a,
            n_heads,
            key_rows,
            row_stride,
            latent_dim,
            pq_dim,
            pq_sub,
            pq_card,
            7,
            pq_nchunk,
        )
        .expect_err("packed non-byte codes must fail before dispatch");
        assert!(error.to_string().contains("bits=8"));
        assert_eq!(rejected.dispatch_count(), 0);

        let short_codes = ctx
            .new_buffer_with_bytes_checked(&codes[..codes.len() - 1])
            .expect("short code test buffer");
        let error = encode_pq_k_transpose_heads(
            &mut rejected,
            &codebookb,
            &short_codes,
            &queryb,
            &output_a,
            n_heads,
            key_rows,
            row_stride,
            latent_dim,
            pq_dim,
            pq_sub,
            pq_card,
            8,
            pq_nchunk,
        )
        .expect_err("undersized PQ codes must fail before dispatch");
        assert!(error.to_string().contains("byte range"));
        assert_eq!(rejected.dispatch_count(), 0);
    }

    #[test]
    fn route_segment_pq_k_transpose_flagship_geometry_passes_v21() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let (n_heads, key_rows, row_stride) = (64usize, 192usize, 448usize);
        let (latent_dim, pq_dim, pq_sub, pq_card, pq_nchunk) =
            (512usize, 32usize, 32usize, 256usize, 16usize);
        let codebook: Vec<half::f16> = (0..pq_card * pq_sub)
            .map(|index| {
                let signed = ((index * 17 + index / 31) % 257) as i32 - 128;
                half::f16::from_f32(signed as f32 * (1.0 / 512.0))
            })
            .collect();
        let rows_touched = (n_heads - 1) * row_stride + key_rows;
        assert_eq!(rows_touched, 28_416);
        let codes: Vec<u8> = (0..rows_touched * pq_nchunk)
            .map(|index| {
                let row = index / pq_nchunk;
                let chunk = index % pq_nchunk;
                ((row * 11 + chunk * 17 + row / 7) & 255) as u8
            })
            .collect();
        let query: Vec<f32> = (0..n_heads * key_rows)
            .map(|index| {
                let signed = ((index * 13 + index / 29) % 127) as i32 - 63;
                signed as f32 * (1.0 / 128.0)
            })
            .collect();
        let codebookb = f16_buffer(&ctx, &codebook);
        let codesb = ctx
            .new_buffer_with_bytes_checked(&codes)
            .expect("flagship PQ codes");
        let queryb = f32_buffer(&ctx, &query);
        let output = filled_f32_buffer(&ctx, n_heads * latent_dim, f32::NAN);

        let mut tcb = TokenCommandBuffer::new(&ctx);
        encode_pq_k_transpose_heads(
            &mut tcb, &codebookb, &codesb, &queryb, &output, n_heads, key_rows, row_stride,
            latent_dim, pq_dim, pq_sub, pq_card, 8, pq_nchunk,
        )
        .expect("encode flagship direct PQ K transpose");
        assert_eq!(tcb.dispatch_count(), 1);
        tcb.commit_and_wait()
            .expect("flagship PQ K transpose command buffer");

        let mut host = vec![0.0f32; n_heads * latent_dim];
        let mut authority = vec![0.0f64; n_heads * latent_dim];
        for head in 0..n_heads {
            for col in 0..latent_dim {
                let chunk = col / pq_dim;
                let within = col % pq_dim;
                let mut host_acc = 0.0f32;
                let mut authority_acc = 0.0f64;
                for key_row in 0..key_rows {
                    let row = head * row_stride + key_row;
                    let code = codes[row * pq_nchunk + chunk] as usize;
                    let weight = codebook[code * pq_sub + within].to_f32();
                    let q = query[head * key_rows + key_row];
                    host_acc = weight.mul_add(q, host_acc);
                    authority_acc += weight as f64 * q as f64;
                }
                host[head * latent_dim + col] = host_acc;
                authority[head * latent_dim + col] = authority_acc;
            }
        }
        assert_v21_pair(
            "flagship direct PQ K transpose",
            &host,
            &read_f32(&output, host.len()),
            &authority,
        );
    }

    #[test]
    fn route_segment_compact_ranked_attention_is_stable_alias_safe_and_v21() {
        let shader = include_str!("../shaders/gravity_pq.metal");
        assert!(shader.contains("kernel void gravity_glm_compact_ranked_attn("));
        let registry = include_str!("metal/mod.rs");
        assert!(registry.contains(
            "\"gravity_glm_compact_ranked_attn\" => \"gravity_glm_compact_ranked_attn\""
        ));

        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let (n_heads, latent_dim, rope_dim, n_keys) = (2usize, 8usize, 4usize, 10usize);
        let ranked = [6u32, 2, 5, 9, 8, 0, 3];
        let scale = 0.5f32;
        let query_latent: Vec<f32> = (0..n_heads * latent_dim)
            .map(|index| {
                let signed = ((index * 11 + index / 3) % 29) as i32 - 14;
                signed as f32 * (1.0 / 32.0)
            })
            .collect();
        let query_rope: Vec<f32> = (0..n_heads * rope_dim)
            .map(|index| {
                let signed = ((index * 7 + 3) % 17) as i32 - 8;
                signed as f32 * (1.0 / 16.0)
            })
            .collect();
        let latent_cache: Vec<f32> = (0..n_keys * latent_dim)
            .map(|index| {
                let signed = ((index * 13 + index / 5) % 37) as i32 - 18;
                0.75 + signed as f32 * (1.0 / 64.0)
            })
            .collect();
        let rope_cache: Vec<f32> = (0..n_keys * rope_dim)
            .map(|index| {
                let signed = ((index * 5 + index / 4) % 23) as i32 - 11;
                signed as f32 * (1.0 / 32.0)
            })
            .collect();

        let mut host = vec![0.0f32; n_heads * latent_dim];
        let mut authority = vec![0.0f64; n_heads * latent_dim];
        for head in 0..n_heads {
            let mut scores = vec![0.0f32; ranked.len()];
            let mut authority_scores = vec![0.0f64; ranked.len()];
            for (slot, &token) in ranked.iter().enumerate() {
                let token = token as usize;
                let mut dot = 0.0f32;
                let mut authority_dot = 0.0f64;
                for dim in 0..latent_dim {
                    let q = query_latent[head * latent_dim + dim];
                    let k = latent_cache[token * latent_dim + dim];
                    dot = q.mul_add(k, dot);
                    authority_dot += q as f64 * k as f64;
                }
                for dim in 0..rope_dim {
                    let q = query_rope[head * rope_dim + dim];
                    let k = rope_cache[token * rope_dim + dim];
                    dot = q.mul_add(k, dot);
                    authority_dot += q as f64 * k as f64;
                }
                scores[slot] = dot * scale;
                authority_scores[slot] = authority_dot * scale as f64;
            }
            let best = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let authority_best = authority_scores
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let exponentials: Vec<f32> = scores.iter().map(|&score| (score - best).exp()).collect();
            let authority_exponentials: Vec<f64> = authority_scores
                .iter()
                .map(|&score| (score - authority_best).exp())
                .collect();
            let total: f32 = exponentials.iter().sum();
            let authority_total: f64 = authority_exponentials.iter().sum();
            for dim in 0..latent_dim {
                let mut acc = 0.0f32;
                let mut authority_acc = 0.0f64;
                for (slot, &token) in ranked.iter().enumerate() {
                    let value = latent_cache[token as usize * latent_dim + dim];
                    acc = (exponentials[slot] / total).mul_add(value, acc);
                    authority_acc +=
                        (authority_exponentials[slot] / authority_total) * value as f64;
                }
                host[head * latent_dim + dim] = acc;
                authority[head * latent_dim + dim] = authority_acc;
            }
        }

        let query_latent_a = f32_buffer(&ctx, &query_latent);
        let query_latent_alias = f32_buffer(&ctx, &query_latent);
        let query_rope_buffer = f32_buffer(&ctx, &query_rope);
        let latent_cache_buffer = f32_buffer(&ctx, &latent_cache);
        let rope_cache_buffer = f32_buffer(&ctx, &rope_cache);
        let ranked_buffer = u32_buffer(&ctx, &ranked);
        let output_a = filled_f32_buffer(&ctx, host.len(), f32::NAN);
        let output_b = filled_f32_buffer(&ctx, host.len(), f32::NAN);

        let mut tcb = TokenCommandBuffer::new(&ctx);
        for output in [&output_a, &output_b] {
            encode_compact_ranked_attention(
                &mut tcb,
                &query_latent_a,
                &query_rope_buffer,
                &latent_cache_buffer,
                &rope_cache_buffer,
                &ranked_buffer,
                output,
                n_heads,
                latent_dim,
                rope_dim,
                n_keys,
                ranked.len(),
                scale,
            )
            .expect("encode compact ranked attention");
        }
        encode_compact_ranked_attention(
            &mut tcb,
            &query_latent_alias,
            &query_rope_buffer,
            &latent_cache_buffer,
            &rope_cache_buffer,
            &ranked_buffer,
            &query_latent_alias,
            n_heads,
            latent_dim,
            rope_dim,
            n_keys,
            ranked.len(),
            scale,
        )
        .expect("encode in-place compact ranked attention");
        assert_eq!(tcb.dispatch_count(), 3);
        tcb.commit_and_wait()
            .expect("compact ranked attention command buffer");

        let device_a = read_f32(&output_a, host.len());
        let device_b = read_f32(&output_b, host.len());
        let device_alias = read_f32(&query_latent_alias, host.len());
        assert_eq!(device_a, device_b, "repeated dispatch must be bit-stable");
        assert_eq!(
            device_a, device_alias,
            "query/weighted-latent alias must be exact"
        );
        assert_v21_pair("compact ranked attention", &host, &device_a, &authority);

        let mut rejected = TokenCommandBuffer::new(&ctx);
        let error = encode_compact_ranked_attention(
            &mut rejected,
            &query_latent_a,
            &query_rope_buffer,
            &latent_cache_buffer,
            &rope_cache_buffer,
            &ranked_buffer,
            &output_a,
            n_heads,
            latent_dim,
            rope_dim,
            n_keys,
            2049,
            scale,
        )
        .expect_err("oversized selection must fail before dispatch");
        assert!(error.to_string().contains("n_allow <= 2048"));
        assert_eq!(rejected.dispatch_count(), 0);

        let short_ranked = u32_buffer(&ctx, &ranked[..ranked.len() - 1]);
        let error = encode_compact_ranked_attention(
            &mut rejected,
            &query_latent_a,
            &query_rope_buffer,
            &latent_cache_buffer,
            &rope_cache_buffer,
            &short_ranked,
            &output_a,
            n_heads,
            latent_dim,
            rope_dim,
            n_keys,
            ranked.len(),
            scale,
        )
        .expect_err("undersized ranked-index buffer must fail before dispatch");
        assert!(error.to_string().contains("byte range"));
        assert_eq!(rejected.dispatch_count(), 0);
    }

    #[test]
    fn route_segment_pq_v_rows_is_deterministic_fail_closed_and_v21() {
        let shader = include_str!("../shaders/gravity_pq.metal");
        assert!(shader.contains("kernel void gravity_pq_v_rows_heads("));
        let registry = include_str!("metal/mod.rs");
        assert!(registry.contains("\"gravity_pq_v_rows_heads\" => \"gravity_pq_v_rows_heads\""));

        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let (n_heads, row_stride, value_row_offset, value_rows) = (2usize, 5usize, 2usize, 3usize);
        let (latent_dim, pq_dim, pq_sub, pq_card, pq_nchunk) =
            (8usize, 4usize, 4usize, 256usize, 2usize);
        let codebook_prefix = [
            0.25, -0.5, 0.75, 1.0, -1.0, 0.125, 0.5, -0.25, 1.5, -0.75, 0.25, 0.625, -0.375, 1.25,
            -1.5, 0.875,
        ];
        let mut codebook = vec![half::f16::ZERO; pq_card * pq_sub];
        for (dst, value) in codebook.iter_mut().zip(codebook_prefix) {
            *dst = half::f16::from_f32(value);
        }
        let rows_touched = (n_heads - 1) * row_stride + value_row_offset + value_rows;
        let codes: Vec<u8> = (0..rows_touched * pq_nchunk)
            .map(|index| ((index * 5 + index / 3) % 4) as u8)
            .collect();
        let weighted_latent = [
            0.75, -1.25, 0.5, -0.625, 1.5, 0.25, -0.375, 0.875, -0.5, 1.25, 0.625, -0.75, 0.375,
            -1.5, 0.25, 1.0,
        ];
        let codebookb = f16_buffer(&ctx, &codebook);
        let codesb = ctx
            .new_buffer_with_bytes_checked(&codes)
            .expect("PQ V-row code test buffer");
        let weightedb = f32_buffer(&ctx, &weighted_latent);
        let output_a = filled_f32_buffer(&ctx, n_heads * value_rows, f32::NAN);
        let output_b = filled_f32_buffer(&ctx, n_heads * value_rows, f32::NAN);

        let mut tcb = TokenCommandBuffer::new(&ctx);
        for output in [&output_a, &output_b] {
            encode_pq_v_rows_heads(
                &mut tcb,
                &codebookb,
                &codesb,
                &weightedb,
                output,
                n_heads,
                row_stride,
                value_row_offset,
                value_rows,
                latent_dim,
                pq_dim,
                pq_sub,
                pq_card,
                8,
                pq_nchunk,
            )
            .expect("encode direct PQ V rows");
        }
        assert_eq!(tcb.dispatch_count(), 2);
        tcb.commit_and_wait()
            .expect("direct PQ V-row command buffer");

        let mut host = vec![0.0f32; n_heads * value_rows];
        let mut authority = vec![0.0f64; n_heads * value_rows];
        for head in 0..n_heads {
            for value_row in 0..value_rows {
                let source_row = head * row_stride + value_row_offset + value_row;
                let mut host_acc = 0.0f32;
                let mut authority_acc = 0.0f64;
                for chunk in 0..pq_nchunk {
                    let code = codes[source_row * pq_nchunk + chunk] as usize;
                    for within in 0..pq_sub {
                        let weight = codebook[code * pq_sub + within].to_f32();
                        let x = weighted_latent[head * latent_dim + chunk * pq_dim + within];
                        host_acc = weight.mul_add(x, host_acc);
                        authority_acc += weight as f64 * x as f64;
                    }
                }
                let output = head * value_rows + value_row;
                host[output] = host_acc;
                authority[output] = authority_acc;
            }
        }
        let device_a = read_f32(&output_a, host.len());
        let device_b = read_f32(&output_b, host.len());
        assert_eq!(device_a, device_b, "repeated V-row dispatch must be exact");
        assert_v21_pair("direct PQ V rows", &host, &device_a, &authority);

        let mut rejected = TokenCommandBuffer::new(&ctx);
        let error = encode_pq_v_rows_heads(
            &mut rejected,
            &codebookb,
            &codesb,
            &weightedb,
            &output_a,
            n_heads,
            row_stride,
            value_row_offset,
            value_rows,
            latent_dim,
            pq_dim,
            pq_sub,
            pq_card,
            7,
            pq_nchunk,
        )
        .expect_err("packed non-byte codes must fail before dispatch");
        assert!(error.to_string().contains("bits=8"));
        assert_eq!(rejected.dispatch_count(), 0);

        let error = encode_pq_v_rows_heads(
            &mut rejected,
            &codebookb,
            &codesb,
            &weightedb,
            &output_a,
            n_heads,
            row_stride,
            value_row_offset + 1,
            value_rows,
            latent_dim,
            pq_dim,
            pq_sub,
            pq_card,
            8,
            pq_nchunk,
        )
        .expect_err("out-of-stride value window must fail before dispatch");
        assert!(error.to_string().contains("exceeds row_stride"));
        assert_eq!(rejected.dispatch_count(), 0);

        let short_codes = ctx
            .new_buffer_with_bytes_checked(&codes[..codes.len() - 1])
            .expect("short V-row codes");
        let error = encode_pq_v_rows_heads(
            &mut rejected,
            &codebookb,
            &short_codes,
            &weightedb,
            &output_a,
            n_heads,
            row_stride,
            value_row_offset,
            value_rows,
            latent_dim,
            pq_dim,
            pq_sub,
            pq_card,
            8,
            pq_nchunk,
        )
        .expect_err("undersized V-row codes must fail before dispatch");
        assert!(error.to_string().contains("byte range"));
        assert_eq!(rejected.dispatch_count(), 0);
    }

    #[test]
    fn route_segment_pq_v_rows_flagship_geometry_passes_v21() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let (n_heads, row_stride, value_row_offset, value_rows) =
            (64usize, 448usize, 192usize, 256usize);
        let (latent_dim, pq_dim, pq_sub, pq_card, pq_nchunk) =
            (512usize, 32usize, 32usize, 256usize, 16usize);
        let codebook: Vec<half::f16> = (0..pq_card * pq_sub)
            .map(|index| {
                let signed = ((index * 19 + index / 23) % 257) as i32 - 128;
                half::f16::from_f32(signed as f32 * (1.0 / 512.0))
            })
            .collect();
        let rows_touched = (n_heads - 1) * row_stride + value_row_offset + value_rows;
        assert_eq!(rows_touched, 28_672);
        let codes: Vec<u8> = (0..rows_touched * pq_nchunk)
            .map(|index| {
                let row = index / pq_nchunk;
                let chunk = index % pq_nchunk;
                ((row * 13 + chunk * 29 + row / 11) & 255) as u8
            })
            .collect();
        let weighted_latent: Vec<f32> = (0..n_heads * latent_dim)
            .map(|index| {
                let signed = ((index * 17 + index / 31) % 127) as i32 - 63;
                signed as f32 * (1.0 / 128.0)
            })
            .collect();
        let codebookb = f16_buffer(&ctx, &codebook);
        let codesb = ctx
            .new_buffer_with_bytes_checked(&codes)
            .expect("flagship V-row codes");
        let weightedb = f32_buffer(&ctx, &weighted_latent);
        let output = filled_f32_buffer(&ctx, n_heads * value_rows, f32::NAN);

        let mut tcb = TokenCommandBuffer::new(&ctx);
        encode_pq_v_rows_heads(
            &mut tcb,
            &codebookb,
            &codesb,
            &weightedb,
            &output,
            n_heads,
            row_stride,
            value_row_offset,
            value_rows,
            latent_dim,
            pq_dim,
            pq_sub,
            pq_card,
            8,
            pq_nchunk,
        )
        .expect("encode flagship direct PQ V rows");
        assert_eq!(tcb.dispatch_count(), 1);
        tcb.commit_and_wait()
            .expect("flagship direct PQ V-row command buffer");

        let mut host = vec![0.0f32; n_heads * value_rows];
        let mut authority = vec![0.0f64; n_heads * value_rows];
        for head in 0..n_heads {
            for value_row in 0..value_rows {
                let source_row = head * row_stride + value_row_offset + value_row;
                let mut host_acc = 0.0f32;
                let mut authority_acc = 0.0f64;
                for chunk in 0..pq_nchunk {
                    let code = codes[source_row * pq_nchunk + chunk] as usize;
                    for within in 0..pq_sub {
                        let weight = codebook[code * pq_sub + within].to_f32();
                        let x = weighted_latent[head * latent_dim + chunk * pq_dim + within];
                        host_acc = weight.mul_add(x, host_acc);
                        authority_acc += weight as f64 * x as f64;
                    }
                }
                let output_index = head * value_rows + value_row;
                host[output_index] = host_acc;
                authority[output_index] = authority_acc;
            }
        }
        assert_v21_pair(
            "flagship direct PQ V rows",
            &host,
            &read_f32(&output, host.len()),
            &authority,
        );
    }

    #[test]
    fn compact_absorbed_three_dispatch_chain_preserves_ranked_v21_contract() {
        const TOKENS: usize = 11;
        const HEADS: usize = 3;
        const LATENT: usize = 17;
        const NOPE: usize = 13;
        const ROPE: usize = 4;
        const VALUE: usize = 9;
        const ROW_STRIDE: usize = NOPE + VALUE;
        const CARD: usize = 256;
        const SELECTED: usize = 7;

        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let latents = deterministic_fixture_f32(0x1020_3040, TOKENS * LATENT, 0.8);
        let rope_keys = deterministic_fixture_f32(0x5566_7788, TOKENS * ROPE, 0.6);
        let query_nope = deterministic_fixture_f32(0x90ab_cdef, HEADS * NOPE, 0.7);
        let query_rope = deterministic_fixture_f32(0x3141_5926, HEADS * ROPE, 0.5);
        let key_weight: Vec<f32> =
            deterministic_fixture_f32(0x2718_2818, HEADS * NOPE * LATENT, 0.35)
                .into_iter()
                .map(|value| half::f16::from_f32(value).to_f32())
                .collect();
        let value_weight: Vec<f32> =
            deterministic_fixture_f32(0xdead_beef, HEADS * VALUE * LATENT, 0.4)
                .into_iter()
                .map(|value| half::f16::from_f32(value).to_f32())
                .collect();
        let ranked = [6u32, 2, 5, 9, 8, 0, 3];
        let mut ascending = ranked;
        ascending.sort_unstable();
        assert_eq!(ascending, [0, 2, 3, 5, 6, 8, 9]);

        // One direct-u8 code per logical row. Each row receives a unique
        // f16 codebook entry, so this tiny fixture exactly exercises the same
        // D==sub, S=1 addressing used by the flagship D32 matrix.
        let mut codebook = vec![half::f16::ZERO; CARD * LATENT];
        let mut codes = vec![0u8; HEADS * ROW_STRIDE];
        for head in 0..HEADS {
            for key_row in 0..NOPE {
                let source_row = head * ROW_STRIDE + key_row;
                codes[source_row] = source_row as u8;
                for latent in 0..LATENT {
                    codebook[source_row * LATENT + latent] =
                        half::f16::from_f32(key_weight[(head * NOPE + key_row) * LATENT + latent]);
                }
            }
            for value_row in 0..VALUE {
                let source_row = head * ROW_STRIDE + NOPE + value_row;
                codes[source_row] = source_row as u8;
                for latent in 0..LATENT {
                    codebook[source_row * LATENT + latent] = half::f16::from_f32(
                        value_weight[(head * VALUE + value_row) * LATENT + latent],
                    );
                }
            }
        }

        // Expanded f32 source formulation and a shared f64 authority.
        let selected_ascending: Vec<usize> = ascending
            .iter()
            .map(|&position| position as usize)
            .collect();
        let scale = ((NOPE + ROPE) as f32).powf(-0.5);
        let mut expanded = vec![0.0f32; HEADS * VALUE];
        let mut authority = vec![0.0f64; HEADS * VALUE];
        for head in 0..HEADS {
            let mut logits = vec![0.0f32; SELECTED];
            let mut authority_logits = vec![0.0f64; SELECTED];
            let mut expanded_values = vec![0.0f32; SELECTED * VALUE];
            let mut authority_values = vec![0.0f64; SELECTED * VALUE];
            for (slot, &token) in selected_ascending.iter().enumerate() {
                let mut dot = 0.0f32;
                let mut authority_dot = 0.0f64;
                for key_row in 0..NOPE {
                    let mut key = 0.0f32;
                    let mut authority_key = 0.0f64;
                    for latent in 0..LATENT {
                        let weight = key_weight[(head * NOPE + key_row) * LATENT + latent];
                        let value = latents[token * LATENT + latent];
                        key = weight.mul_add(value, key);
                        authority_key += weight as f64 * value as f64;
                    }
                    let query = query_nope[head * NOPE + key_row];
                    dot = query.mul_add(key, dot);
                    authority_dot += query as f64 * authority_key;
                }
                for rope in 0..ROPE {
                    let query = query_rope[head * ROPE + rope];
                    let key = rope_keys[token * ROPE + rope];
                    dot = query.mul_add(key, dot);
                    authority_dot += query as f64 * key as f64;
                }
                logits[slot] = dot * scale;
                authority_logits[slot] = authority_dot * scale as f64;

                for value_row in 0..VALUE {
                    let mut value = 0.0f32;
                    let mut authority_value = 0.0f64;
                    for latent in 0..LATENT {
                        let weight = value_weight[(head * VALUE + value_row) * LATENT + latent];
                        let input = latents[token * LATENT + latent];
                        value = weight.mul_add(input, value);
                        authority_value += weight as f64 * input as f64;
                    }
                    expanded_values[slot * VALUE + value_row] = value;
                    authority_values[slot * VALUE + value_row] = authority_value;
                }
            }
            let best = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let authority_best = authority_logits
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let exponentials: Vec<f32> = logits.iter().map(|&score| (score - best).exp()).collect();
            let authority_exponentials: Vec<f64> = authority_logits
                .iter()
                .map(|&score| (score - authority_best).exp())
                .collect();
            let total: f32 = exponentials.iter().sum();
            let authority_total: f64 = authority_exponentials.iter().sum();
            for slot in 0..SELECTED {
                let probability = exponentials[slot] / total;
                let authority_probability = authority_exponentials[slot] / authority_total;
                for value_row in 0..VALUE {
                    let output = head * VALUE + value_row;
                    expanded[output] = probability
                        .mul_add(expanded_values[slot * VALUE + value_row], expanded[output]);
                    authority[output] +=
                        authority_probability * authority_values[slot * VALUE + value_row];
                }
            }
        }

        let codebookb = f16_buffer(&ctx, &codebook);
        let codesb = ctx
            .new_buffer_with_bytes_checked(&codes)
            .expect("chain direct-u8 codes");
        let query_nopeb = f32_buffer(&ctx, &query_nope);
        let query_ropeb = f32_buffer(&ctx, &query_rope);
        let latentsb = f32_buffer(&ctx, &latents);
        let rope_keysb = f32_buffer(&ctx, &rope_keys);
        let rankedb = u32_buffer(&ctx, &ranked);
        let ascendingb = u32_buffer(&ctx, &ascending);
        let query_latent_ranked = filled_f32_buffer(&ctx, HEADS * LATENT, f32::NAN);
        let query_latent_ascending = filled_f32_buffer(&ctx, HEADS * LATENT, f32::NAN);
        let context_ranked = filled_f32_buffer(&ctx, HEADS * VALUE, f32::NAN);
        let context_ascending = filled_f32_buffer(&ctx, HEADS * VALUE, f32::NAN);

        let mut tcb = TokenCommandBuffer::new(&ctx);
        for (indices, query_latent, context) in [
            (&rankedb, &query_latent_ranked, &context_ranked),
            (&ascendingb, &query_latent_ascending, &context_ascending),
        ] {
            encode_pq_k_transpose_heads(
                &mut tcb,
                &codebookb,
                &codesb,
                &query_nopeb,
                query_latent,
                HEADS,
                NOPE,
                ROW_STRIDE,
                LATENT,
                LATENT,
                LATENT,
                CARD,
                8,
                1,
            )
            .expect("encode chain K transpose");
            encode_compact_ranked_attention(
                &mut tcb,
                query_latent,
                &query_ropeb,
                &latentsb,
                &rope_keysb,
                indices,
                query_latent,
                HEADS,
                LATENT,
                ROPE,
                TOKENS,
                SELECTED,
                scale,
            )
            .expect("encode chain compact ranked attention");
            encode_pq_v_rows_heads(
                &mut tcb,
                &codebookb,
                &codesb,
                query_latent,
                context,
                HEADS,
                ROW_STRIDE,
                NOPE,
                VALUE,
                LATENT,
                LATENT,
                LATENT,
                CARD,
                8,
                1,
            )
            .expect("encode chain direct PQ V rows");
        }
        assert_eq!(tcb.dispatch_count(), 6);
        tcb.commit_and_wait()
            .expect("compact absorbed three-dispatch chains");

        let ranked_context = read_f32(&context_ranked, expanded.len());
        let ascending_context = read_f32(&context_ascending, expanded.len());
        let mut bounds = Bounds::continuous_only();
        bounds.top_k = 5;
        let ranked_pair = score_pair(&expanded, &ranked_context, &authority, &bounds);
        assert!(
            ranked_pair.pass,
            "ranked compact chain must pass V2.1: {ranked_pair:#?}"
        );
        assert!(
            ranked_pair.device.discrete.greedy_match
                && ranked_pair.device.discrete.top_k_exact_match,
            "ranked chain final-context decisions must be exact"
        );
        let ascending_pair = score_pair(&expanded, &ascending_context, &authority, &bounds);
        assert!(
            ascending_pair.pass,
            "the f16-codebook/FMA chain's ascending diagnostic must remain \
             independently characterized: {ascending_pair:#?}"
        );
        assert_ne!(
            ranked_context, ascending_context,
            "selected-position traversal order must remain numerically observable"
        );
    }

    #[test]
    fn compact_absorbed_five_dispatch_dag_passes_v21_without_readback() {
        const TOKENS: usize = 5;
        const HEADS: usize = 2;
        const LATENT: usize = 8;
        const NOPE: usize = 3;
        const ROPE: usize = 2;
        const VALUE: usize = 4;
        const CONTEXT: usize = HEADS * VALUE;
        const ROW_STRIDE: usize = NOPE + VALUE;
        const CARD: usize = 256;

        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let latents = deterministic_fixture_f32(0x1357_2468, TOKENS * LATENT, 0.7);
        let rope_keys = deterministic_fixture_f32(0x2468_1357, TOKENS * ROPE, 0.5);
        let query_nope = deterministic_fixture_f32(0xabcd_0123, HEADS * NOPE, 0.6);
        let query_rope = deterministic_fixture_f32(0x7654_3210, HEADS * ROPE, 0.4);
        let ranked = [4u32, 1, 3];
        let scale = ((NOPE + ROPE) as f32).powf(-0.5);

        let mut kv_codebook = vec![half::f16::ZERO; CARD * LATENT];
        let mut kv_codes = vec![0u8; HEADS * ROW_STRIDE];
        let kv_weights: Vec<f32> =
            deterministic_fixture_f32(0xcafe_babe, HEADS * ROW_STRIDE * LATENT, 0.3)
                .into_iter()
                .map(|value| half::f16::from_f32(value).to_f32())
                .collect();
        for row in 0..HEADS * ROW_STRIDE {
            kv_codes[row] = row as u8;
            for latent in 0..LATENT {
                kv_codebook[row * LATENT + latent] =
                    half::f16::from_f32(kv_weights[row * LATENT + latent]);
            }
        }

        // Identity o_proj in the same direct-u8 single-chunk representation.
        let mut o_codebook = vec![half::f16::ZERO; CARD * CONTEXT];
        let mut o_codes = vec![0u8; CONTEXT + 4];
        for row in 0..CONTEXT {
            o_codes[row] = row as u8;
            o_codebook[row * CONTEXT + row] = half::f16::ONE;
        }

        let mut host_query_latent = vec![0.0f32; HEADS * LATENT];
        let mut authority_query_latent = vec![0.0f64; HEADS * LATENT];
        for head in 0..HEADS {
            for latent in 0..LATENT {
                for key_row in 0..NOPE {
                    let weight = kv_weights[(head * ROW_STRIDE + key_row) * LATENT + latent];
                    let query = query_nope[head * NOPE + key_row];
                    let output = head * LATENT + latent;
                    host_query_latent[output] = weight.mul_add(query, host_query_latent[output]);
                    authority_query_latent[output] += weight as f64 * query as f64;
                }
            }
        }
        let mut host_weighted = vec![0.0f32; HEADS * LATENT];
        let mut authority_weighted = vec![0.0f64; HEADS * LATENT];
        for head in 0..HEADS {
            let mut logits = vec![0.0f32; ranked.len()];
            let mut authority_logits = vec![0.0f64; ranked.len()];
            for (slot, &token) in ranked.iter().enumerate() {
                let token = token as usize;
                for latent in 0..LATENT {
                    logits[slot] = host_query_latent[head * LATENT + latent]
                        .mul_add(latents[token * LATENT + latent], logits[slot]);
                    authority_logits[slot] += authority_query_latent[head * LATENT + latent]
                        * latents[token * LATENT + latent] as f64;
                }
                for rope in 0..ROPE {
                    logits[slot] = query_rope[head * ROPE + rope]
                        .mul_add(rope_keys[token * ROPE + rope], logits[slot]);
                    authority_logits[slot] += query_rope[head * ROPE + rope] as f64
                        * rope_keys[token * ROPE + rope] as f64;
                }
                logits[slot] *= scale;
                authority_logits[slot] *= scale as f64;
            }
            let best = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let authority_best = authority_logits
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let exponentials: Vec<f32> = logits.iter().map(|&score| (score - best).exp()).collect();
            let authority_exponentials: Vec<f64> = authority_logits
                .iter()
                .map(|&score| (score - authority_best).exp())
                .collect();
            let total: f32 = exponentials.iter().sum();
            let authority_total: f64 = authority_exponentials.iter().sum();
            for (slot, &token) in ranked.iter().enumerate() {
                let token = token as usize;
                for latent in 0..LATENT {
                    let output = head * LATENT + latent;
                    host_weighted[output] = (exponentials[slot] / total)
                        .mul_add(latents[token * LATENT + latent], host_weighted[output]);
                    authority_weighted[output] += (authority_exponentials[slot] / authority_total)
                        * latents[token * LATENT + latent] as f64;
                }
            }
        }
        let mut host = vec![0.0f32; CONTEXT];
        let mut authority = vec![0.0f64; CONTEXT];
        for head in 0..HEADS {
            for value_row in 0..VALUE {
                let source_row = head * ROW_STRIDE + NOPE + value_row;
                let output = head * VALUE + value_row;
                for latent in 0..LATENT {
                    let weight = kv_weights[source_row * LATENT + latent];
                    host[output] =
                        weight.mul_add(host_weighted[head * LATENT + latent], host[output]);
                    authority[output] += weight as f64 * authority_weighted[head * LATENT + latent];
                }
            }
        }

        let kv_codebookb = f16_buffer(&ctx, &kv_codebook);
        let kv_codesb = ctx
            .new_buffer_with_bytes_checked(&kv_codes)
            .expect("five-dispatch KV codes");
        let o_codebookb = f16_buffer(&ctx, &o_codebook);
        let o_codesb = ctx
            .new_buffer_with_bytes_checked(&o_codes)
            .expect("five-dispatch o_proj codes");
        let query_nopeb = f32_buffer(&ctx, &query_nope);
        let query_ropeb = f32_buffer(&ctx, &query_rope);
        let latent_cacheb = filled_f32_buffer(&ctx, TOKENS * LATENT, f32::NAN);
        let rope_cacheb = filled_f32_buffer(&ctx, TOKENS * ROPE, f32::NAN);
        write_f32(&latent_cacheb, &latents[..(TOKENS - 1) * LATENT]);
        write_f32(&rope_cacheb, &rope_keys[..(TOKENS - 1) * ROPE]);
        let current_latentb = f32_buffer(&ctx, &latents[(TOKENS - 1) * LATENT..TOKENS * LATENT]);
        let current_ropeb = f32_buffer(&ctx, &rope_keys[(TOKENS - 1) * ROPE..TOKENS * ROPE]);
        let rankedb = u32_buffer(&ctx, &ranked);
        let query_latentb = filled_f32_buffer(&ctx, HEADS * LATENT, f32::NAN);
        let contextb = filled_f32_buffer(&ctx, CONTEXT, f32::NAN);
        let hiddenb = filled_f32_buffer(&ctx, CONTEXT, f32::NAN);

        let mut tcb = TokenCommandBuffer::new(&ctx);
        encode_mla_append_compact(
            &mut tcb,
            &current_latentb,
            &current_ropeb,
            &latent_cacheb,
            &rope_cacheb,
            LATENT,
            ROPE,
            TOKENS - 1,
        )
        .expect("encode compact append");
        encode_pq_k_transpose_heads(
            &mut tcb,
            &kv_codebookb,
            &kv_codesb,
            &query_nopeb,
            &query_latentb,
            HEADS,
            NOPE,
            ROW_STRIDE,
            LATENT,
            LATENT,
            LATENT,
            CARD,
            8,
            1,
        )
        .expect("encode absorbed K");
        encode_compact_ranked_attention(
            &mut tcb,
            &query_latentb,
            &query_ropeb,
            &latent_cacheb,
            &rope_cacheb,
            &rankedb,
            &query_latentb,
            HEADS,
            LATENT,
            ROPE,
            TOKENS,
            ranked.len(),
            scale,
        )
        .expect("encode ranked compact attention");
        encode_pq_v_rows_heads(
            &mut tcb,
            &kv_codebookb,
            &kv_codesb,
            &query_latentb,
            &contextb,
            HEADS,
            ROW_STRIDE,
            NOPE,
            VALUE,
            LATENT,
            LATENT,
            LATENT,
            CARD,
            8,
            1,
        )
        .expect("encode absorbed V");
        encode_pq_matvec_device(
            &mut tcb,
            &o_codebookb,
            &o_codesb,
            crate::gravity_glm::gpu::PqParams {
                dim: CONTEXT as u32,
                subspaces: 1,
                sub: CONTEXT as u32,
                card: CARD as u32,
                rows: CONTEXT as u32,
                cols: CONTEXT as u32,
                nchunk: 1,
                bits: 8,
            },
            &contextb,
            &hiddenb,
        )
        .expect("encode unchanged o_proj");
        assert_eq!(tcb.dispatch_count(), 5);
        tcb.commit_and_wait()
            .expect("five-dispatch compact attention DAG");

        assert_eq!(read_f32(&latent_cacheb, latents.len()), latents);
        assert_eq!(read_f32(&rope_cacheb, rope_keys.len()), rope_keys);
        let device = read_f32(&hiddenb, CONTEXT);
        let mut bounds = Bounds::continuous_only();
        bounds.top_k = 5;
        let pair = score_pair(&host, &device, &authority, &bounds);
        assert!(pair.pass, "five-dispatch compact DAG V2.1: {pair:#?}");
        assert!(
            pair.device.discrete.greedy_match && pair.device.discrete.top_k_exact_match,
            "five-dispatch final decisions must be exact"
        );
    }

    #[test]
    fn route_segment_reorder_is_exact_at_edges_and_after_tied_topk() {
        let shader = include_str!("../shaders/gravity_pq.metal");
        assert!(shader.contains("kernel void gravity_glm_sort_u32_ascending("));
        let registry = include_str!("metal/mod.rs");
        assert!(registry
            .contains("\"gravity_glm_sort_u32_ascending\" => \"gravity_glm_sort_u32_ascending\""));

        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let edge_sizes = [1usize, 2, 3, 31, 32, 33, 255, 256, 257, 1023, 2047, 2048];
        let mut fixtures = Vec::new();
        let mut tcb = TokenCommandBuffer::new(&ctx);

        let dummy = u32_buffer(&ctx, &[0]);
        encode_sort_positions_ascending(&mut tcb, &dummy, &dummy, 0)
            .expect("k=0 is an encode no-op");
        assert_eq!(tcb.dispatch_count(), 0);

        for &k in &edge_sizes {
            let mut input: Vec<u32> = (0..k as u32).collect();
            for index in 0..k {
                let peer = (index.wrapping_mul(37).wrapping_add(11)) % k;
                input.swap(index, peer);
            }
            if k == 3 {
                input = vec![32767, 0, 8192];
            }
            let input_buffer = u32_buffer(&ctx, &input);
            let output_buffer = if k == 33 {
                input_buffer.clone()
            } else {
                u32_buffer(&ctx, &vec![u32::MAX; k])
            };
            encode_sort_positions_ascending(&mut tcb, &input_buffer, &output_buffer, k)
                .expect("encode bounded ascending reorder");
            let mut expected = input.clone();
            expected.sort_unstable();
            fixtures.push((output_buffer, expected));
        }
        assert_eq!(tcb.dispatch_count(), edge_sizes.len());
        tcb.commit_and_wait().expect("edge reorder command buffer");
        for (output, expected) in fixtures {
            assert_eq!(
                read_u32(&output, expected.len()),
                expected,
                "GPU reorder must be exact"
            );
        }

        let oversized_input = u32_buffer(&ctx, &vec![0; 2049]);
        let oversized_output = u32_buffer(&ctx, &vec![0; 2049]);
        let mut oversized_tcb = TokenCommandBuffer::new(&ctx);
        let error = encode_sort_positions_ascending(
            &mut oversized_tcb,
            &oversized_input,
            &oversized_output,
            2049,
        )
        .expect_err("k above the flagship bound must fail before encode");
        assert!(error.to_string().contains("k <= 2048"));
        assert_eq!(oversized_tcb.dispatch_count(), 0);

        let tied_values = vec![3.0, 7.0, 7.0, -1.0, 7.0, 2.0, 9.0, 9.0, 0.0, 9.0];
        let k = 7usize;
        let values = f32_buffer(&ctx, &tied_values);
        let score_order = u32_buffer(&ctx, &vec![u32::MAX; k]);
        let selected = empty_u8_buffer(&ctx, tied_values.len());
        let ascending = u32_buffer(&ctx, &vec![u32::MAX; k]);
        let mut chain = TokenCommandBuffer::new(&ctx);
        encode_stable_topk(
            &mut chain,
            &values,
            &score_order,
            &selected,
            tied_values.len(),
            k,
        )
        .expect("encode tied stable top-k");
        encode_sort_positions_ascending(&mut chain, &score_order, &ascending, k)
            .expect("encode top-k reorder");
        assert_eq!(chain.dispatch_count(), 2);
        chain.commit_and_wait().expect("top-k reorder chain");

        let expected_score_order: Vec<u32> = topk_desc(&tied_values, k)
            .into_iter()
            .map(|index| index as u32)
            .collect();
        assert_eq!(read_u32(&score_order, k), expected_score_order);
        let mut expected_ascending = expected_score_order;
        expected_ascending.sort_unstable();
        assert_eq!(read_u32(&ascending, k), expected_ascending);
    }

    #[test]
    fn radix_topk_is_exact_at_32k_2048_with_ties_and_signed_zero() {
        let shader = include_str!("../shaders/gravity_pq.metal");
        assert!(shader.contains("kernel void gravity_glm_radix_topk_f32("));
        let registry = include_str!("metal/mod.rs");
        assert!(
            registry.contains("\"gravity_glm_radix_topk_f32\" => \"gravity_glm_radix_topk_f32\"")
        );

        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const N: usize = 32768;
        let mut state = 0x1bad_f00du32;
        let mut values = Vec::with_capacity(N);
        for index in 0..N {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let bucket = ((state >> 8) % 4096) as i32 - 2048;
            let mut score = bucket as f32 * (1.0 / 128.0);
            if index % 127 == 0 {
                score = 7.0; // many exact ties, lower position must rank first
            }
            values.push(score);
        }
        values[3] = -0.0;
        values[4] = 0.0;
        values[17] = f32::INFINITY;
        values[18] = f32::NEG_INFINITY;

        let values_buffer = f32_buffer(&ctx, &values);
        let ks = [1usize, 3, 33, 2048, 2048, 2048];
        let mut outputs = Vec::new();
        let mut tcb = TokenCommandBuffer::new(&ctx);
        for &k in &ks {
            let output = u32_buffer(&ctx, &vec![u32::MAX; k]);
            encode_radix_topk(&mut tcb, &values_buffer, &output, values.len(), k)
                .expect("encode 32K radix top-k");
            outputs.push((k, output));
        }
        assert_eq!(tcb.dispatch_count(), ks.len());
        tcb.commit_and_wait()
            .expect("32K radix top-k command buffer");

        for (k, output) in outputs {
            let expected: Vec<u32> = topk_desc(&values, k)
                .into_iter()
                .map(|index| index as u32)
                .collect();
            assert_eq!(
                read_u32(&output, k),
                expected,
                "radix rank mismatch at k={k}"
            );
        }

        let oversized = u32_buffer(&ctx, &vec![0; 2049]);
        let mut rejected = TokenCommandBuffer::new(&ctx);
        let error = encode_radix_topk(&mut rejected, &values_buffer, &oversized, N, 2049)
            .expect_err("radix k above compact bound must fail before encode");
        assert!(error.to_string().contains("k <= 2048"));
        assert_eq!(rejected.dispatch_count(), 0);
    }

    #[test]
    #[ignore = "bounded Metal timing; run explicitly on a free-enough GPU"]
    fn benchmark_radix_topk_32k_2048_against_serial_oracle() {
        let ctx = MetalContext::new().expect("Metal device for bounded top-k benchmark");
        const N: usize = 32768;
        const K: usize = 2048;
        const ITERS: usize = 5;
        let mut state = 0x5eed_1234u32;
        let values: Vec<f32> = (0..N)
            .map(|index| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                if index % 113 == 0 {
                    3.5
                } else {
                    (((state >> 8) % 16384) as i32 - 8192) as f32 * (1.0 / 256.0)
                }
            })
            .collect();
        let expected: Vec<u32> = topk_desc(&values, K)
            .into_iter()
            .map(|index| index as u32)
            .collect();
        let values_buffer = f32_buffer(&ctx, &values);
        let serial_output = u32_buffer(&ctx, &vec![u32::MAX; K]);
        let serial_selected = empty_u8_buffer(&ctx, N);
        let radix_output = u32_buffer(&ctx, &vec![u32::MAX; K]);

        let run_serial = || {
            let mut tcb = TokenCommandBuffer::new(&ctx);
            encode_stable_topk(
                &mut tcb,
                &values_buffer,
                &serial_output,
                &serial_selected,
                N,
                K,
            )
            .expect("encode serial stable top-k");
            tcb.commit_and_wait().expect("serial stable top-k");
        };
        let run_radix = || {
            let mut tcb = TokenCommandBuffer::new(&ctx);
            encode_radix_topk(&mut tcb, &values_buffer, &radix_output, N, K)
                .expect("encode radix top-k");
            tcb.commit_and_wait().expect("radix top-k");
        };

        run_serial();
        run_radix();
        assert_eq!(read_u32(&serial_output, K), expected);
        assert_eq!(read_u32(&radix_output, K), expected);

        let mut serial_us = Vec::with_capacity(ITERS);
        let mut radix_us = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let started = std::time::Instant::now();
            run_serial();
            serial_us.push(started.elapsed().as_secs_f64() * 1e6);
        }
        for _ in 0..ITERS {
            let started = std::time::Instant::now();
            run_radix();
            radix_us.push(started.elapsed().as_secs_f64() * 1e6);
        }
        serial_us.sort_by(f64::total_cmp);
        radix_us.sort_by(f64::total_cmp);
        let serial_median = serial_us[ITERS / 2];
        let radix_median = radix_us[ITERS / 2];
        eprintln!(
            "device DSA top-k N={N} K={K}: serial_us={serial_us:?} radix_us={radix_us:?} \
             median_speedup={:.3}x",
            serial_median / radix_median
        );
        assert!(
            radix_median < serial_median,
            "parallel radix median {radix_median:.3} us did not beat serial {serial_median:.3} us"
        );
    }

    #[test]
    fn route_segment_residual_add_is_exact_alias_safe_and_fail_closed() {
        let shader = include_str!("../shaders/gravity_pq.metal");
        assert!(shader.contains("kernel void gravity_add_inplace_f32("));
        let registry = include_str!("metal/mod.rs");
        assert!(registry.contains("\"gravity_add_inplace_f32\" => \"gravity_add_inplace_f32\""));

        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let x: Vec<f32> = (0..257)
            .map(|index| ((index % 31) as i32 - 15) as f32 * 0.5)
            .collect();
        let y: Vec<f32> = (0..257)
            .map(|index| (((index * 7) % 29) as i32 - 14) as f32 * 0.25)
            .collect();
        let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| x + y).collect();
        let xb = f32_buffer(&ctx, &x);
        let yb = f32_buffer(&ctx, &y);

        let alias_values = vec![1.5, -2.0, 0.25, 4.0];
        let alias = f32_buffer(&ctx, &alias_values);
        let mut tcb = TokenCommandBuffer::new(&ctx);
        encode_residual_add_inplace(&mut tcb, &xb, &yb, x.len())
            .expect("encode residual add over a rounded grid");
        encode_residual_add_inplace(&mut tcb, &alias, &alias, alias_values.len())
            .expect("encode explicitly supported full-buffer alias");
        assert_eq!(tcb.dispatch_count(), 2);
        tcb.commit_and_wait().expect("residual add command buffer");
        assert_eq!(read_f32(&xb, x.len()), expected);
        assert_eq!(
            read_f32(&alias, alias_values.len()),
            vec![3.0, -4.0, 0.5, 8.0]
        );

        let one = f32_buffer(&ctx, &[1.0]);
        let mut rejected = TokenCommandBuffer::new(&ctx);
        encode_residual_add_inplace(&mut rejected, &one, &one, 0)
            .expect("zero elements are an encode no-op");
        assert_eq!(rejected.dispatch_count(), 0);

        let error = encode_residual_add_inplace(&mut rejected, &one, &one, 2)
            .expect_err("undersized buffers must fail before dispatch");
        assert!(error.to_string().contains("byte range"));
        assert_eq!(rejected.dispatch_count(), 0);

        let error = encode_residual_add_inplace(&mut rejected, &one, &one, u32::MAX as usize)
            .expect_err("rounded grid overflow must fail before dispatch");
        assert!(error
            .to_string()
            .contains("rounded Metal grid width overflow"));
        assert_eq!(rejected.dispatch_count(), 0);

        let oversized_n = (u32::MAX as usize)
            .checked_add(1)
            .expect("usize exceeds the Metal u32 ABI on supported hosts");
        let error = encode_residual_add_inplace(&mut rejected, &one, &one, oversized_n)
            .expect_err("oversized geometry must fail before dispatch");
        assert!(error.to_string().contains("does not fit the Metal u32 ABI"));
        assert_eq!(rejected.dispatch_count(), 0);
    }

    #[test]
    fn route_segment_norm_rope_copy_and_zero_match_host_and_f64() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };

        let x = vec![0.5, -1.25, 2.0, 0.75, -0.125, 3.5, -2.25];
        let weight = vec![0.75, -0.5, 1.25, 0.875, -1.5, 0.25, 2.0];
        let bias = vec![0.1, -0.2, 0.3, -0.4, 0.05, 0.125, -0.25];
        let xb = f32_buffer(&ctx, &x);
        let wb = f32_buffer(&ctx, &weight);
        let bb = f32_buffer(&ctx, &bias);
        let rms_out = filled_f32_buffer(&ctx, x.len(), f32::NAN);
        let affine_out = filled_f32_buffer(&ctx, x.len(), f32::NAN);

        let rope_input = vec![
            91.0, 92.0, 1.0, 2.0, 3.0, 4.0, // head 0, rotate tail
            81.0, 82.0, -1.5, 0.5, 2.5, -3.0, // head 1
        ];
        let cos = vec![0.875, -0.25];
        let sin = vec![0.125, 0.75];
        let rope_in = f32_buffer(&ctx, &rope_input);
        let cosb = f32_buffer(&ctx, &cos);
        let sinb = f32_buffer(&ctx, &sin);
        let rope_out = filled_f32_buffer(&ctx, 8, f32::NAN);

        let copy_src = f32_buffer(&ctx, &[10.0, 11.0, 12.0, 13.0, 14.0]);
        let copy_dst = filled_f32_buffer(&ctx, 7, -9.0);
        let zero_out = filled_f32_buffer(&ctx, 9, 7.0);

        let mut tcb = TokenCommandBuffer::new(&ctx);
        encode_rmsnorm(&mut tcb, &xb, &wb, &rms_out, x.len(), 1e-6).expect("encode rms");
        encode_layernorm_affine(&mut tcb, &xb, &wb, &bb, &affine_out, x.len(), 1e-6)
            .expect("encode affine layernorm");
        encode_rope_interleaved(
            &mut tcb, &rope_in, 2, &rope_out, 0, &cosb, &sinb, 2, 4, 6, 4,
        )
        .expect("encode GLM RoPE");
        encode_copy_tail(&mut tcb, &copy_src, &copy_dst, 1, 3, 3).expect("encode copy tail");
        encode_zero(&mut tcb, &zero_out, 9).expect("encode zero");
        assert_eq!(tcb.dispatch_count(), 5);
        tcb.commit_and_wait().expect("primitive command buffer");

        let mean_sq_f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv_f32 = 1.0 / (mean_sq_f32 + 1e-6).sqrt();
        let rms_host: Vec<f32> = x
            .iter()
            .zip(&weight)
            .map(|(v, w)| v * inv_f32 * w)
            .collect();
        let mean_sq_f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
        let inv_f64 = 1.0 / (mean_sq_f64 + 1e-6f64).sqrt();
        let rms_f64: Vec<f64> = x
            .iter()
            .zip(&weight)
            .map(|(&v, &w)| (v as f64) * inv_f64 * (w as f64))
            .collect();
        assert_v21_pair("rmsnorm", &rms_host, &read_f32(&rms_out, x.len()), &rms_f64);

        let mean_f32 = x.iter().sum::<f32>() / x.len() as f32;
        let var_f32 = x
            .iter()
            .map(|v| (v - mean_f32) * (v - mean_f32))
            .sum::<f32>()
            / x.len() as f32;
        let affine_inv_f32 = 1.0 / (var_f32 + 1e-6).sqrt();
        let affine_host: Vec<f32> = x
            .iter()
            .zip(&weight)
            .zip(&bias)
            .map(|((&v, &w), &b)| (v - mean_f32) * affine_inv_f32 * w + b)
            .collect();
        let mean_f64 = x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64;
        let var_f64 = x
            .iter()
            .map(|&v| {
                let d = v as f64 - mean_f64;
                d * d
            })
            .sum::<f64>()
            / x.len() as f64;
        let affine_inv_f64 = 1.0 / (var_f64 + 1e-6f64).sqrt();
        let affine_f64: Vec<f64> = x
            .iter()
            .zip(&weight)
            .zip(&bias)
            .map(|((&v, &w), &b)| (v as f64 - mean_f64) * affine_inv_f64 * w as f64 + b as f64)
            .collect();
        assert_v21_pair(
            "affine layernorm",
            &affine_host,
            &read_f32(&affine_out, x.len()),
            &affine_f64,
        );

        let mut rope_host = Vec::new();
        let mut rope_f64 = Vec::new();
        for head in 0..2 {
            let base = head * 6 + 2;
            rope_host.extend(rope_interleaved(&rope_input[base..base + 4], &cos, &sin));
            let src = &rope_input[base..base + 4];
            for i in 0..2 {
                rope_f64.push(
                    src[2 * i] as f64 * cos[i] as f64 - src[2 * i + 1] as f64 * sin[i] as f64,
                );
            }
            for i in 0..2 {
                rope_f64.push(
                    src[2 * i + 1] as f64 * cos[i] as f64 + src[2 * i] as f64 * sin[i] as f64,
                );
            }
        }
        assert_v21_pair(
            "interleaved RoPE",
            &rope_host,
            &read_f32(&rope_out, 8),
            &rope_f64,
        );

        assert_eq!(
            read_f32(&copy_dst, 7),
            vec![-9.0, -9.0, -9.0, 11.0, 12.0, 13.0, -9.0]
        );
        assert_eq!(read_f32(&zero_out, 9), vec![0.0; 9]);
    }

    #[test]
    fn route_segment_rope_prefix_tail_matches_host_and_fails_closed() {
        let shader = include_str!("../shaders/gravity_pq.metal");
        assert!(shader.contains("kernel void gravity_rope_prefix_tail_f32("));
        let registry = include_str!("metal/mod.rs");
        assert!(registry
            .contains("\"gravity_rope_prefix_tail_f32\" => \"gravity_rope_prefix_tail_f32\""));

        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 91.0, 92.0, // head 0: rotary prefix + tail
            -1.5, 0.5, 2.5, -3.0, 81.0, 82.0, // head 1
        ];
        let cos = vec![0.875, -0.25];
        let sin = vec![0.125, 0.75];
        let input_buffer = f32_buffer(&ctx, &input);
        let cos_buffer = f32_buffer(&ctx, &cos);
        let sin_buffer = f32_buffer(&ctx, &sin);
        let output = filled_f32_buffer(&ctx, 14, -99.0);

        let mut tcb = TokenCommandBuffer::new(&ctx);
        encode_rope_prefix_tail(
            &mut tcb,
            &input_buffer,
            0,
            &output,
            1,
            &cos_buffer,
            &sin_buffer,
            2,
            4,
            6,
            6,
        )
        .expect("encode RoPE prefix plus tail");
        assert_eq!(tcb.dispatch_count(), 1);
        tcb.commit_and_wait().expect("RoPE prefix plus tail");

        let mut expected = vec![-99.0];
        let mut authority = Vec::new();
        for head in 0..2 {
            let source = &input[head * 6..(head + 1) * 6];
            expected.extend(rope_interleaved(&source[..4], &cos, &sin));
            expected.extend_from_slice(&source[4..]);
            for i in 0..2 {
                authority.push(
                    source[2 * i] as f64 * cos[i] as f64 - source[2 * i + 1] as f64 * sin[i] as f64,
                );
            }
            for i in 0..2 {
                authority.push(
                    source[2 * i + 1] as f64 * cos[i] as f64 + source[2 * i] as f64 * sin[i] as f64,
                );
            }
            authority.extend(source[4..].iter().map(|&value| value as f64));
        }
        expected.push(-99.0);
        let actual = read_f32(&output, expected.len());
        assert_eq!(actual.first(), Some(&-99.0));
        assert_eq!(actual.last(), Some(&-99.0));
        assert_v21_pair(
            "RoPE prefix plus tail",
            &expected[1..expected.len() - 1],
            &actual[1..actual.len() - 1],
            &authority,
        );

        let mut rejected = TokenCommandBuffer::new(&ctx);
        let error = encode_rope_prefix_tail(
            &mut rejected,
            &input_buffer,
            0,
            &input_buffer,
            0,
            &cos_buffer,
            &sin_buffer,
            2,
            4,
            6,
            6,
        )
        .expect_err("in-place prefix assembly is not alias safe");
        assert!(error.to_string().contains("non-aliasing"));
        assert_eq!(rejected.dispatch_count(), 0);
    }

    #[test]
    fn route_segment_mla_build_and_index_append_match_host_exactly() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let (n_heads, qk_nope, qk_rope, v_dim, position) = (2usize, 2usize, 2usize, 2usize, 1usize);
        let qk = qk_nope + qk_rope;
        let kv = vec![1.0, 2.0, 11.0, 12.0, 3.0, 4.0, 13.0, 14.0];
        let k_rot = vec![21.0, 22.0];
        let kvb = f32_buffer(&ctx, &kv);
        let krb = f32_buffer(&ctx, &k_rot);
        let keys = filled_f32_buffer(&ctx, 3 * n_heads * qk, -99.0);
        let values = filled_f32_buffer(&ctx, 3 * n_heads * v_dim, -77.0);
        let latent = vec![0.25, -1.5, 3.75];
        let latentb = f32_buffer(&ctx, &latent);
        let compact_latent = filled_f32_buffer(&ctx, 3 * latent.len(), -66.0);
        let compact_rope = filled_f32_buffer(&ctx, 3 * k_rot.len(), -44.0);

        let q = vec![1.0, 2.0, 90.0, 91.0, 3.0, 4.0, 92.0, 93.0];
        let q_rot = vec![31.0, 32.0, 33.0, 34.0];
        let qb = f32_buffer(&ctx, &q);
        let qrb = f32_buffer(&ctx, &q_rot);
        let queries = filled_f32_buffer(&ctx, n_heads * qk, f32::NAN);
        let query_nope = filled_f32_buffer(&ctx, n_heads * qk_nope, f32::NAN);

        let k_full = vec![41.0, 42.0, 43.0, 44.0];
        let kfb = f32_buffer(&ctx, &k_full);
        let index_keys = filled_f32_buffer(&ctx, 3 * k_full.len(), -55.0);

        let mut tcb = TokenCommandBuffer::new(&ctx);
        encode_mla_append_kv_expanded(
            &mut tcb, &kvb, &krb, &keys, &values, n_heads, qk_nope, qk_rope, v_dim, position,
        )
        .expect("encode expanded MLA append");
        encode_mla_append_compact(
            &mut tcb,
            &latentb,
            &krb,
            &compact_latent,
            &compact_rope,
            latent.len(),
            k_rot.len(),
            position,
        )
        .expect("encode compact MLA append");
        encode_build_queries(&mut tcb, &qb, &qrb, &queries, n_heads, qk_nope, qk_rope)
            .expect("encode build queries");
        encode_copy_head_prefix(&mut tcb, &qb, &query_nope, n_heads, qk_nope, qk_rope)
            .expect("encode compact query prefix");
        encode_append_index_key(&mut tcb, &kfb, &index_keys, 2, k_full.len())
            .expect("encode index-key append");
        assert_eq!(tcb.dispatch_count(), 5);
        tcb.commit_and_wait().expect("MLA primitive command buffer");

        let mut expected_keys = vec![-99.0; 3 * n_heads * qk];
        let mut expected_values = vec![-77.0; 3 * n_heads * v_dim];
        for head in 0..n_heads {
            let kv_base = head * (qk_nope + v_dim);
            let key_base = (position * n_heads + head) * qk;
            expected_keys[key_base..key_base + qk_nope]
                .copy_from_slice(&kv[kv_base..kv_base + qk_nope]);
            expected_keys[key_base + qk_nope..key_base + qk].copy_from_slice(&k_rot);
            let value_base = (position * n_heads + head) * v_dim;
            expected_values[value_base..value_base + v_dim]
                .copy_from_slice(&kv[kv_base + qk_nope..kv_base + qk_nope + v_dim]);
        }
        assert_eq!(read_f32(&keys, expected_keys.len()), expected_keys);
        assert_eq!(read_f32(&values, expected_values.len()), expected_values);
        assert_eq!(
            read_f32(&compact_latent, 3 * latent.len()),
            vec![-66.0, -66.0, -66.0, 0.25, -1.5, 3.75, -66.0, -66.0, -66.0]
        );
        assert_eq!(
            read_f32(&compact_rope, 3 * k_rot.len()),
            vec![-44.0, -44.0, 21.0, 22.0, -44.0, -44.0]
        );

        let expected_queries = vec![1.0, 2.0, 31.0, 32.0, 3.0, 4.0, 33.0, 34.0];
        assert_eq!(read_f32(&queries, expected_queries.len()), expected_queries);
        assert_eq!(
            read_f32(&query_nope, n_heads * qk_nope),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            read_f32(&index_keys, 3 * k_full.len()),
            vec![-55.0, -55.0, -55.0, -55.0, -55.0, -55.0, -55.0, -55.0, 41.0, 42.0, 43.0, 44.0,]
        );

        let mut rejected = TokenCommandBuffer::new(&ctx);
        encode_mla_append_compact(
            &mut rejected,
            &latentb,
            &krb,
            &compact_latent,
            &compact_rope,
            0,
            0,
            usize::MAX,
        )
        .expect("empty compact append is an encode no-op");
        assert_eq!(rejected.dispatch_count(), 0);
        let too_small = filled_f32_buffer(&ctx, latent.len(), 0.0);
        let error = encode_mla_append_compact(
            &mut rejected,
            &latentb,
            &krb,
            &too_small,
            &compact_rope,
            latent.len(),
            k_rot.len(),
            position,
        )
        .expect_err("undersized compact latent cache must fail before dispatch");
        assert!(error.to_string().contains("byte range"));
        assert_eq!(rejected.dispatch_count(), 0);

        let short_prefix = filled_f32_buffer(&ctx, n_heads * qk_nope - 1, 0.0);
        let error =
            encode_copy_head_prefix(&mut rejected, &qb, &short_prefix, n_heads, qk_nope, qk_rope)
                .expect_err("undersized compact query prefix must fail before dispatch");
        assert!(error.to_string().contains("byte range"));
        assert_eq!(rejected.dispatch_count(), 0);
    }

    #[test]
    fn route_segment_dsa_topk_sparse_and_router_pass_v21_and_exact_ids() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let (n_keys, n_heads, head_dim) = (5usize, 2usize, 3usize);
        let q_full = vec![1.0, -0.5, 2.0, -1.5, 0.25, 0.75];
        let index_keys = vec![
            0.5, 1.0, -0.25, 2.0, -0.5, 1.0, 2.0, -0.5, 1.0, 0.125, 0.25, 0.5, -0.75, 1.5, -1.0,
        ];
        let head_weights = vec![0.75, -0.25];
        let head_scale = (n_heads as f32).powf(-0.5);
        let dim_scale = (head_dim as f32).powf(-0.5);
        let dsa_position = n_keys - 2;
        let qfb = f32_buffer(&ctx, &q_full);
        let ikb = f32_buffer(&ctx, &index_keys);
        let hwb = f32_buffer(&ctx, &head_weights);
        let dsa_scores = filled_f32_buffer(&ctx, n_keys, f32::NAN);
        let topk_indices = u32_buffer(&ctx, &[u32::MAX; 3]);
        let selected = empty_u8_buffer(&ctx, n_keys);

        let qk_dim = 3usize;
        let v_dim = 2usize;
        let queries = vec![0.5, -1.0, 1.5, -0.75, 0.25, 2.0];
        let sparse_keys: Vec<f32> = (0..n_keys * n_heads * qk_dim)
            .map(|i| ((i as f32 * 0.173).sin() * 1.25) + 0.05)
            .collect();
        let sparse_values: Vec<f32> = (0..n_keys * n_heads * v_dim)
            .map(|i| ((i as f32 * 0.219).cos() * 0.75) - 0.1)
            .collect();
        let allow = vec![0u32, 2, 4];
        let queryb = f32_buffer(&ctx, &queries);
        let sparse_keyb = f32_buffer(&ctx, &sparse_keys);
        let sparse_valueb = f32_buffer(&ctx, &sparse_values);
        let allowb = u32_buffer(&ctx, &allow);
        let context = filled_f32_buffer(&ctx, n_heads * v_dim, f32::NAN);
        let sparse_scale = (qk_dim as f32).powf(-0.5);

        let logits = vec![-4.0, -0.25, 0.0, 1.25, 5.0, 0.75];
        let bias = vec![0.125, -0.05, 0.2, -0.125, 0.01, 0.3];
        let logitb = f32_buffer(&ctx, &logits);
        let biasb = f32_buffer(&ctx, &bias);
        let router_scores = filled_f32_buffer(&ctx, logits.len(), f32::NAN);
        let corrected = filled_f32_buffer(&ctx, logits.len(), f32::NAN);
        let router_indices = u32_buffer(&ctx, &[u32::MAX; 3]);
        let router_weights = filled_f32_buffer(&ctx, 3, f32::NAN);
        let router_exec_slots = u32_buffer(&ctx, &[u32::MAX; 3]);

        let mut tcb = TokenCommandBuffer::new(&ctx);
        encode_dsa_scores(
            &mut tcb,
            &qfb,
            &ikb,
            &hwb,
            &dsa_scores,
            n_keys,
            n_heads,
            head_dim,
            dsa_position,
            dim_scale,
            head_scale,
        )
        .expect("encode DSA scores");
        encode_stable_topk(&mut tcb, &dsa_scores, &topk_indices, &selected, n_keys, 3)
            .expect("encode exact stable top-k");
        encode_sparse_attention_expanded_ascending_allow(
            &mut tcb,
            &queryb,
            &sparse_keyb,
            &sparse_valueb,
            &allowb,
            &context,
            n_heads,
            qk_dim,
            v_dim,
            n_keys,
            allow.len(),
            sparse_scale,
        )
        .expect("encode expanded sparse attention");
        encode_router_correction(
            &mut tcb,
            &logitb,
            &biasb,
            &router_scores,
            &corrected,
            logits.len(),
        )
        .expect("encode router correction");
        encode_router_select_noaux(
            &mut tcb,
            &logitb,
            &biasb,
            &router_scores,
            &corrected,
            &router_indices,
            &router_weights,
            &router_exec_slots,
            logits.len(),
            3,
            2,
            3,
            true,
            1.25,
        )
        .expect("encode exact noaux router selection");
        assert_eq!(tcb.dispatch_count(), 5);
        tcb.commit_and_wait()
            .expect("decision primitive command buffer");

        let mut dsa_host = vec![0.0f32; n_keys];
        let mut dsa_f64 = vec![0.0f64; n_keys];
        for key_index in 0..n_keys {
            if key_index > dsa_position {
                dsa_host[key_index] = f32::NEG_INFINITY;
                dsa_f64[key_index] = f64::NEG_INFINITY;
                continue;
            }
            let key = &index_keys[key_index * head_dim..(key_index + 1) * head_dim];
            let mut host_acc = 0.0f32;
            let mut authority_acc = 0.0f64;
            for head in 0..n_heads {
                let query = &q_full[head * head_dim..(head + 1) * head_dim];
                let host_dot = query.iter().zip(key).map(|(x, y)| x * y).sum::<f32>();
                let scaled_weight = head_weights[head] * head_scale;
                host_acc += scaled_weight * (host_dot * dim_scale).max(0.0);
                let authority_dot = query
                    .iter()
                    .zip(key)
                    .map(|(&x, &y)| x as f64 * y as f64)
                    .sum::<f64>();
                authority_acc += (head_weights[head] * head_scale) as f64
                    * (authority_dot * dim_scale as f64).max(0.0);
            }
            dsa_host[key_index] = host_acc;
            dsa_f64[key_index] = authority_acc;
        }
        let dsa_device = read_f32(&dsa_scores, n_keys);
        assert_v21_pair(
            "DSA scores",
            &dsa_host[..=dsa_position],
            &dsa_device[..=dsa_position],
            &dsa_f64[..=dsa_position],
        );
        assert_eq!(dsa_device[dsa_position + 1], f32::NEG_INFINITY);
        let topk_device: Vec<usize> = read_u32(&topk_indices, 3)
            .into_iter()
            .map(|v| v as usize)
            .collect();
        assert_eq!(topk_device, topk_desc(&dsa_host, 3));
        assert_eq!(topk_device, topk_desc_f64(&dsa_f64, 3));
        assert!(
            topk_device.iter().position(|&index| index == 1)
                < topk_device.iter().position(|&index| index == 2),
            "the lower index must win the exact DSA score tie"
        );

        let mut sparse_host = vec![0.0f32; n_heads * v_dim];
        let mut sparse_f64 = vec![0.0f64; n_heads * v_dim];
        for head in 0..n_heads {
            let query = &queries[head * qk_dim..(head + 1) * qk_dim];
            let mut host_logits = Vec::new();
            let mut authority_logits = Vec::new();
            for &position in &allow {
                let position = position as usize;
                let key_base = (position * n_heads + head) * qk_dim;
                let key = &sparse_keys[key_base..key_base + qk_dim];
                host_logits
                    .push(query.iter().zip(key).map(|(x, y)| x * y).sum::<f32>() * sparse_scale);
                authority_logits.push(
                    query
                        .iter()
                        .zip(key)
                        .map(|(&x, &y)| x as f64 * y as f64)
                        .sum::<f64>()
                        * sparse_scale as f64,
                );
            }
            let host_best = host_logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut host_probs: Vec<f32> =
                host_logits.iter().map(|v| (v - host_best).exp()).collect();
            let host_total = host_probs.iter().sum::<f32>();
            for value in &mut host_probs {
                *value /= host_total;
            }
            let authority_best = authority_logits
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut authority_probs: Vec<f64> = authority_logits
                .iter()
                .map(|v| (v - authority_best).exp())
                .collect();
            let authority_total = authority_probs.iter().sum::<f64>();
            for value in &mut authority_probs {
                *value /= authority_total;
            }
            for (slot, &position) in allow.iter().enumerate() {
                let value_base = (position as usize * n_heads + head) * v_dim;
                for dim in 0..v_dim {
                    sparse_host[head * v_dim + dim] +=
                        host_probs[slot] * sparse_values[value_base + dim];
                    sparse_f64[head * v_dim + dim] +=
                        authority_probs[slot] * sparse_values[value_base + dim] as f64;
                }
            }
        }
        assert_v21_pair(
            "expanded sparse attention",
            &sparse_host,
            &read_f32(&context, sparse_host.len()),
            &sparse_f64,
        );

        let router_host: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
        let router_f64: Vec<f64> = logits
            .iter()
            .map(|&l| 1.0 / (1.0 + (-(l as f64)).exp()))
            .collect();
        assert_v21_pair(
            "router sigmoid",
            &router_host,
            &read_f32(&router_scores, logits.len()),
            &router_f64,
        );
        let corrected_host: Vec<f32> = router_host.iter().zip(&bias).map(|(s, b)| s + b).collect();
        let corrected_f64: Vec<f64> = router_f64
            .iter()
            .zip(&bias)
            .map(|(s, &b)| s + b as f64)
            .collect();
        assert_v21_pair(
            "router correction",
            &corrected_host,
            &read_f32(&corrected, logits.len()),
            &corrected_f64,
        );

        let per_group = 2;
        let group_scores: Vec<f32> = corrected_host
            .chunks_exact(per_group)
            .map(|group| {
                topk_desc(group, 2)
                    .into_iter()
                    .map(|index| group[index])
                    .sum()
            })
            .collect();
        let chosen_groups = topk_desc(&group_scores, 2);
        let mut expert_choice = vec![f32::NEG_INFINITY; logits.len()];
        for group in chosen_groups {
            expert_choice[group * per_group..(group + 1) * per_group]
                .copy_from_slice(&corrected_host[group * per_group..(group + 1) * per_group]);
        }
        let expected_indices = topk_desc(&expert_choice, 3);
        assert_eq!(
            read_u32(&router_indices, 3)
                .into_iter()
                .map(|index| index as usize)
                .collect::<Vec<_>>(),
            expected_indices,
            "device noaux selection must preserve stable lower-index ties"
        );
        let mut expected_exec_slots: Vec<u32> = (0..expected_indices.len() as u32).collect();
        expected_exec_slots.sort_by_key(|&slot| expected_indices[slot as usize]);
        assert_eq!(
            read_u32(&router_exec_slots, 3),
            expected_exec_slots,
            "device execution slots must sort selected experts by ascending ID"
        );
        let mut expected_weights: Vec<f32> = expected_indices
            .iter()
            .map(|&index| router_host[index])
            .collect();
        let total = expected_weights.iter().sum::<f32>() + 1e-20;
        for weight in &mut expected_weights {
            *weight = (*weight / total) * 1.25;
        }
        let mut authority_weights: Vec<f64> = expected_indices
            .iter()
            .map(|&index| router_f64[index])
            .collect();
        let authority_total = authority_weights.iter().sum::<f64>() + 1e-20;
        for weight in &mut authority_weights {
            *weight = (*weight / authority_total) * 1.25;
        }
        assert_v21_pair(
            "router selected weights",
            &expected_weights,
            &read_f32(&router_weights, 3),
            &authority_weights,
        );

        let mut rejected = TokenCommandBuffer::new(&ctx);
        let error = encode_router_select_noaux(
            &mut rejected,
            &logitb,
            &biasb,
            &router_scores,
            &corrected,
            &router_indices,
            &router_weights,
            &router_exec_slots,
            logits.len(),
            4,
            2,
            3,
            true,
            1.25,
        )
        .expect_err("non-divisible expert groups must fail before dispatch");
        assert!(error.to_string().contains("unsupported geometry"));
        assert_eq!(rejected.dispatch_count(), 0);

        let tie_logits = f32_buffer(&ctx, &[0.0; 4]);
        let tie_bias = f32_buffer(&ctx, &[0.0; 4]);
        let tie_scores = filled_f32_buffer(&ctx, 4, f32::NAN);
        let tie_corrected = filled_f32_buffer(&ctx, 4, f32::NAN);
        let tie_indices = u32_buffer(&ctx, &[u32::MAX; 2]);
        let tie_weights = filled_f32_buffer(&ctx, 2, f32::NAN);
        let tie_exec_slots = u32_buffer(&ctx, &[u32::MAX; 2]);
        let mut tie_tcb = TokenCommandBuffer::new(&ctx);
        encode_router_select_noaux(
            &mut tie_tcb,
            &tie_logits,
            &tie_bias,
            &tie_scores,
            &tie_corrected,
            &tie_indices,
            &tie_weights,
            &tie_exec_slots,
            4,
            2,
            1,
            2,
            false,
            1.0,
        )
        .expect("encode tied router");
        tie_tcb.commit_and_wait().expect("tied router command");
        assert_eq!(
            read_u32(&tie_indices, 2),
            vec![0, 1],
            "lower group and expert indices must win exact ties"
        );
        assert_eq!(read_f32(&tie_weights, 2), vec![0.5, 0.5]);
        assert_eq!(read_u32(&tie_exec_slots, 2), vec![0, 1]);
    }

    #[test]
    fn expert_wave_scratch_is_lazy_reused_and_grows_monotonically() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let arch = tiny_arch();
        let pool = ActPool::new(&ctx, &arch).expect("activation pool");
        assert_eq!(
            pool.final_norm_weight.length(),
            (arch.hidden * std::mem::size_of::<f32>()) as u64
        );
        assert_eq!(
            pool.final_hidden.length(),
            (arch.hidden * std::mem::size_of::<f32>()) as u64
        );
        assert_eq!(
            pool.expert_exec_slots.length(),
            (arch.num_experts_per_tok.max(1) * std::mem::size_of::<u32>()) as u64
        );
        assert!(
            pool.expert_wave_scratch
                .lock()
                .expect("scratch lock")
                .is_none(),
            "default path must not allocate expert-wave scratch"
        );

        let first_address = {
            let scratch = pool
                .ensure_expert_wave_scratch(&ctx, 2, 7, 4)
                .expect("initial scratch");
            let scratch = scratch.as_ref().expect("scratch allocated");
            assert_eq!(scratch.expert_capacity, 2);
            assert_eq!(scratch.intermediate_capacity, 7);
            assert_eq!(scratch.hidden_capacity, 4);
            assert_eq!(scratch.gate.len(), 2);
            scratch.combined.gpu_address()
        };
        {
            let scratch = pool
                .ensure_expert_wave_scratch(&ctx, 1, 6, 4)
                .expect("reuse adequate scratch");
            let scratch = scratch.as_ref().expect("scratch retained");
            assert_eq!(
                scratch.combined.gpu_address(),
                first_address,
                "adequate scratch must retain its Metal resources"
            );
        }
        {
            let scratch = pool
                .ensure_expert_wave_scratch(&ctx, 3, 9, 8)
                .expect("grow scratch");
            let scratch = scratch.as_ref().expect("scratch grown");
            assert_eq!(scratch.expert_capacity, 3);
            assert_eq!(scratch.intermediate_capacity, 9);
            assert_eq!(scratch.hidden_capacity, 8);
            assert_eq!(scratch.gate.len(), 3);
            assert_ne!(
                scratch.combined.gpu_address(),
                first_address,
                "growth must replace the undersized Metal resources"
            );
        }

        let error = match ExpertWaveScratch::new(&ctx, 1, usize::MAX, 1) {
            Ok(_) => panic!("overflowing f32 scratch geometry must fail before allocation"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("byte size overflow"));
    }

    #[test]
    fn expert_wave_reused_accumulator_is_zeroed_and_combined_in_host_order() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let n = 257usize;
        let first: Vec<f32> = (0..n)
            .map(|i| ((i % 19) as i32 - 9) as f32 * 0.25)
            .collect();
        let second: Vec<f32> = (0..n)
            .map(|i| (((i * 7) % 23) as i32 - 11) as f32 * 0.125)
            .collect();
        let first_scale = 0.75f32;
        let second_scale = -1.25f32;
        let expected: Vec<f32> = first
            .iter()
            .zip(&second)
            .map(|(&a, &b)| {
                let mut value = 0.0f32;
                value += a * first_scale;
                value += b * second_scale;
                value
            })
            .collect();
        let first_buffer = f32_buffer(&ctx, &first);
        let second_buffer = f32_buffer(&ctx, &second);
        let combined = f32_buffer(&ctx, &vec![91.0; n]);

        for stale in [91.0f32, -37.5f32] {
            write_f32(&combined, &vec![stale; n]);
            zero_f32(&combined, n).expect("zero reused accumulator");
            let mut wave = TokenCommandBuffer::new(&ctx);
            encode_axpy_f32(&mut wave, &combined, &first_buffer, first_scale, n as u32)
                .expect("encode first ordered combine");
            encode_axpy_f32(&mut wave, &combined, &second_buffer, second_scale, n as u32)
                .expect("encode second ordered combine");
            wave.commit_and_wait().expect("reused combine wave");
            assert_eq!(
                read_f32(&combined, n),
                expected,
                "stale accumulator contents must not leak across wave reuse"
            );
        }
    }

    #[test]
    fn sequence_scratch_covers_8k_boundary_and_32k_last_position() {
        let cases = [
            (8191usize, 8192usize),
            (8192usize, 16384usize),
            (32767usize, 32768usize),
        ];
        let mut host = HostSequenceScratch::new(64);
        let mut capacity = 64usize;

        for (position, expected_capacity) in cases {
            let need = position.checked_add(1).expect("fixture length");
            capacity = grown_sequence_capacity(capacity, need).expect("capacity growth");
            assert_eq!(capacity, expected_capacity);
            host.grow_preserving(capacity);

            let active =
                active_sequence_len(position, capacity, "test scratch").expect("position fits");
            assert_eq!(active, need);
            assert!(
                checked_sequence_bytes(active, std::mem::size_of::<f32>(), "test active")
                    .expect("active bytes")
                    <= checked_sequence_bytes(
                        capacity,
                        std::mem::size_of::<f32>(),
                        "test capacity"
                    )
                    .expect("capacity bytes")
            );

            // These are the exact four sequence-sized writes used by the
            // resident indexer/selection path.
            host.index_scores[active - 1] = position as f32;
            host.selection_indices[active - 1] = position;
            host.attention_allowed[active - 1] = 1;
            host.attention_scores[active - 1] = -(position as f32);
        }

        assert!(
            active_sequence_len(8192, 8192, "old fixed ActPool score buffer").is_err(),
            "the former fixed buffer must be recognized as too small at position 8192"
        );
    }

    #[test]
    fn host_scratch_growth_preserves_state_and_adequate_reserve_is_allocation_free() {
        let mut host = HostSequenceScratch::new(8192);
        host.index_scores[0] = 1.25;
        host.index_scores[8191] = -7.5;
        host.selection_indices[8191] = 4096;
        host.attention_allowed[8191] = 1;
        host.attention_scores[8191] = 3.75;

        host.grow_preserving(16384);
        assert_eq!(host.index_scores[0], 1.25);
        assert_eq!(host.index_scores[8191], -7.5);
        assert_eq!(host.selection_indices[8191], 4096);
        assert_eq!(host.attention_allowed[8191], 1);
        assert_eq!(host.attention_scores[8191], 3.75);

        host.index_scores[8192] = 9.5;
        host.grow_preserving(32768);
        assert_eq!(host.index_scores[8192], 9.5);

        let pointers = (
            host.index_scores.as_ptr(),
            host.selection_indices.as_ptr(),
            host.attention_allowed.as_ptr(),
            host.attention_scores.as_ptr(),
        );
        let capacities = (
            host.index_scores.capacity(),
            host.selection_indices.capacity(),
            host.attention_allowed.capacity(),
            host.attention_scores.capacity(),
        );
        host.grow_preserving(32768);
        assert_eq!(
            pointers,
            (
                host.index_scores.as_ptr(),
                host.selection_indices.as_ptr(),
                host.attention_allowed.as_ptr(),
                host.attention_scores.as_ptr(),
            ),
            "an adequate reserve must not replace any sequence workspace"
        );
        assert_eq!(
            capacities,
            (
                host.index_scores.capacity(),
                host.selection_indices.capacity(),
                host.attention_allowed.capacity(),
                host.attention_scores.capacity(),
            ),
            "an adequate reserve must not allocate more host capacity"
        );
    }

    #[test]
    fn reused_sequence_topk_matches_numeric_parity_oracle() {
        let mut values = vec![f32::NEG_INFINITY; 32768];
        values[0] = 2.0;
        values[8191] = 8.0;
        values[8192] = 8.0;
        values[16384] = -1.0;
        values[32767] = 7.0;
        let expected = topk_desc(&values, 4);
        let mut selection = vec![usize::MAX; values.len()];
        let pointer = selection.as_ptr();
        let capacity = selection.capacity();

        let actual = topk_desc_with_scratch(&values, 4, &mut selection).expect("scratch selection");
        assert_eq!(actual, expected);
        assert_eq!(actual[..2], [8191, 8192], "lower index wins a score tie");

        let again =
            topk_desc_with_scratch(&values, 4, &mut selection).expect("scratch selection reuse");
        assert_eq!(again, expected);
        assert_eq!(selection.as_ptr(), pointer);
        assert_eq!(selection.capacity(), capacity);
    }

    #[test]
    fn device_index_score_growth_copies_prior_state() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let mut scratch = SequenceScratch::new(&ctx, 8192).expect("initial scratch");
        scratch.host.index_scores[0] = 0.25;
        scratch.host.index_scores[8191] = -3.5;
        scratch
            .store_index_scores(8192)
            .expect("store initial score state");

        scratch.reserve(&ctx, 8193).expect("grow past 8K");
        assert_eq!(scratch.capacity, 16384);
        assert_eq!(scratch.device_score_len, 8192);
        let copied = read_f32(&scratch.index_scores_device, 8192);
        assert_eq!(copied[0], 0.25);
        assert_eq!(copied[8191], -3.5);

        let device_contents = scratch.index_scores_device.contents();
        scratch
            .reserve(&ctx, 16384)
            .expect("adequate reserve is a no-op");
        assert_eq!(scratch.index_scores_device.contents(), device_contents);
    }

    #[test]
    fn resident_growth_preserves_kv_index_keys_and_scores_through_32k_reserve() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let arch = tiny_arch();
        let qk = arch.qk_dim();
        let mut session = ResidentSession::new(&ctx, &arch, 8192).expect("initial session");
        let expanded_layers = match &session.attention {
            ResidentAttentionState::Expanded(cache) => &cache.layers,
            ResidentAttentionState::Compact(_) => panic!("default session must be expanded"),
        };
        assert_eq!(session.dsa.index_keys.len(), expanded_layers.len());

        {
            let cache = session.attention.expanded_layer(0).expect("expanded layer");
            unsafe {
                *(cache.keys.contents() as *mut f32) = 1.0;
                *(cache.keys.contents() as *mut f32).add(8191 * qk + (qk - 1)) = 2.0;
                *(cache.values.contents() as *mut f32).add(8191) = 3.0;
                *(session.dsa.index_keys[0].contents() as *mut f32).add(8191) = 4.0;
            }
        }
        session.dsa.sequence_scratch.host.index_scores[0] = 5.0;
        session.dsa.sequence_scratch.host.index_scores[8191] = 6.0;
        session
            .dsa
            .sequence_scratch
            .store_index_scores(8192)
            .expect("store 8K scores");
        session.seq_len = 8192;

        session
            .reserve(&ctx, &arch, 8193)
            .expect("grow beyond the old 8K limit");
        assert_eq!(session.attention.capacity(), 16384);
        assert_eq!(session.dsa.sequence_scratch.capacity, 16384);
        {
            let cache = session.attention.expanded_layer(0).expect("expanded layer");
            unsafe {
                assert_eq!(*(cache.keys.contents() as *const f32), 1.0);
                assert_eq!(
                    *(cache.keys.contents() as *const f32).add(8191 * qk + (qk - 1)),
                    2.0
                );
                assert_eq!(*(cache.values.contents() as *const f32).add(8191), 3.0);
                assert_eq!(
                    *(session.dsa.index_keys[0].contents() as *const f32).add(8191),
                    4.0
                );
            }
        }
        let scores = read_f32(&session.dsa.sequence_scratch.index_scores_device, 8192);
        assert_eq!(scores[0], 5.0);
        assert_eq!(scores[8191], 6.0);

        {
            let cache = session.attention.expanded_layer(0).expect("expanded layer");
            unsafe {
                *(cache.keys.contents() as *mut f32).add(8192 * qk) = 7.0;
                *(cache.values.contents() as *mut f32).add(8192) = 8.0;
                *(session.dsa.index_keys[0].contents() as *mut f32).add(8192) = 9.0;
            }
        }
        session.dsa.sequence_scratch.host.index_scores[8192] = 10.0;
        session
            .dsa
            .sequence_scratch
            .store_index_scores(8193)
            .expect("store post-8K score");
        session.seq_len = 8193;

        session
            .reserve(&ctx, &arch, 32768)
            .expect("reserve through position 32767");
        assert_eq!(session.attention.capacity(), 32768);
        assert_eq!(session.dsa.sequence_scratch.capacity, 32768);
        {
            let cache = session.attention.expanded_layer(0).expect("expanded layer");
            unsafe {
                assert_eq!(*(cache.keys.contents() as *const f32), 1.0);
                assert_eq!(
                    *(cache.keys.contents() as *const f32).add(8191 * qk + (qk - 1)),
                    2.0
                );
                assert_eq!(
                    *(session.dsa.index_keys[0].contents() as *const f32).add(8191),
                    4.0
                );
                assert_eq!(*(cache.keys.contents() as *const f32).add(8192 * qk), 7.0);
                assert_eq!(*(cache.values.contents() as *const f32).add(8192), 8.0);
                assert_eq!(
                    *(session.dsa.index_keys[0].contents() as *const f32).add(8192),
                    9.0
                );
            }
        }
        let scores = read_f32(&session.dsa.sequence_scratch.index_scores_device, 8193);
        assert_eq!(scores[0], 5.0);
        assert_eq!(scores[8191], 6.0);
        assert_eq!(scores[8192], 10.0);
    }

    #[test]
    fn resident_compact_layout_excludes_expanded_kv_and_grows_with_index_state() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let arch = tiny_arch();
        let default =
            ResidentSession::new(&ctx, &arch, 8).expect("default expanded resident session");
        assert!(matches!(
            default.attention,
            ResidentAttentionState::Expanded(_)
        ));
        assert!(default.dsa.ranked_indices.is_none());
        assert!(!default.dsa.device_selection_enabled());

        let mut compact =
            ResidentSession::new_compact(&ctx, &arch, 8).expect("compact resident session");
        assert!(compact.attention.expanded_layer(0).is_err());
        assert_eq!(compact.dsa.index_keys.len(), arch.n_layers);
        assert!(compact.dsa.ranked_indices.is_some());
        assert!(!compact.dsa.device_selection_enabled());
        {
            let ResidentAttentionState::Compact(cache) = &mut compact.attention else {
                panic!("compact constructor must own only compact MLA state");
            };
            assert_eq!(cache.layers.len(), arch.n_layers);
            assert_eq!(cache.capacity, 8);
            assert_eq!(
                cache.layers[0].latents.length(),
                (8 * arch.kv_lora_rank * 4) as u64
            );
            assert_eq!(
                cache.layers[0].rope_tails.length(),
                (8 * arch.qk_rope_head_dim * 4) as u64
            );
            unsafe {
                *(cache.layers[0].latents.contents() as *mut f32) = 11.0;
                *(cache.layers[0].rope_tails.contents() as *mut f32) = 12.0;
            }
        }
        unsafe {
            *(compact.dsa.index_keys[0].contents() as *mut f32) = 13.0;
        }
        compact.seq_len = 8;
        compact
            .reserve(&ctx, &arch, 9)
            .expect("compact and index state grow together");

        let ResidentAttentionState::Compact(cache) = &compact.attention else {
            panic!("compact reserve must not replace the selected layout");
        };
        assert_eq!(cache.capacity, 16);
        assert_eq!(compact.dsa.sequence_scratch.capacity, 16);
        assert_eq!(
            compact.dsa.index_keys[0].length(),
            (16 * arch.index_head_dim * 4) as u64
        );
        unsafe {
            assert_eq!(*(cache.layers[0].latents.contents() as *const f32), 11.0);
            assert_eq!(*(cache.layers[0].rope_tails.contents() as *const f32), 12.0);
            assert_eq!(*(compact.dsa.index_keys[0].contents() as *const f32), 13.0);
        }
    }

    #[test]
    fn device_dsa_mode_is_compact_only_and_adds_no_sequence_scratch() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let arch = tiny_arch();
        assert!(
            ResidentSession::new_with_layout(
                &ctx,
                &arch,
                8,
                ResidentAttentionLayout::Expanded,
                true,
            )
            .is_err(),
            "device DSA must not allocate under expanded attention"
        );

        let mut device = ResidentSession::new_with_layout(
            &ctx,
            &arch,
            8,
            ResidentAttentionLayout::Compact,
            true,
        )
        .expect("compact device DSA session");
        assert!(device.dsa.device_selection_enabled());
        assert_eq!(
            device.dsa.ranked_indices().unwrap().length(),
            (arch.index_topk.max(1) * 4) as u64,
            "device DSA reuses compact MLA's fixed ranked-index buffer"
        );
        let ranked_before = device.dsa.ranked_indices().unwrap().contents();
        device.seq_len = 8;
        device
            .reserve(&ctx, &arch, 9)
            .expect("device DSA state growth");
        assert_eq!(device.dsa.capacity, 16);
        assert_eq!(
            device.dsa.ranked_indices().unwrap().contents(),
            ranked_before,
            "ranked output is O(index_topk), not O(sequence), and does not grow"
        );
        assert!(device.dsa.device_selection_enabled());

        assert!(
            DeviceDsaTransformScratch::new(&ctx, &arch).is_err(),
            "zero/odd synthetic RoPE geometry must fail before allocation"
        );
        let mut transform_arch = arch.clone();
        transform_arch.index_head_dim = 4;
        transform_arch.qk_rope_head_dim = 2;
        let pool = ActPool::new(&ctx, &transform_arch).expect("default activation pool");
        assert!(
            pool.device_dsa_transform_scratch
                .lock()
                .expect("device DSA transform scratch")
                .is_none(),
            "ordinary activation-pool construction must allocate no device DSA transform buffers"
        );
        assert!(
            pool.device_attention_prelude_scratch
                .lock()
                .expect("device attention prelude scratch")
                .is_none(),
            "ordinary activation-pool construction must allocate no device attention prelude buffers"
        );
        let mut transform_guard = pool
            .ensure_device_dsa_transform_scratch(&ctx, &transform_arch)
            .expect("lazy device DSA transform scratch");
        let transform = transform_guard
            .as_mut()
            .expect("device DSA transform scratch initialized");
        assert_eq!(
            transform.query.length(),
            (transform_arch.index_n_heads * transform_arch.index_head_dim * 4) as u64
        );
        assert_eq!(
            transform.cos.length(),
            (transform_arch.qk_rope_head_dim / 2 * 4) as u64
        );
        assert_eq!(
            transform.norm_weight.length(),
            (transform_arch.index_head_dim * 4) as u64
        );
        drop(transform_guard);

        let mut prelude_guard = pool
            .ensure_device_attention_prelude_scratch(&ctx, &transform_arch)
            .expect("lazy device attention prelude scratch");
        let prelude = prelude_guard
            .as_mut()
            .expect("device attention prelude scratch initialized");
        assert_eq!(
            prelude.input_norm_weight.length(),
            (transform_arch.hidden * 4) as u64
        );
        assert_eq!(
            prelude.q_norm_weight.length(),
            (transform_arch.q_lora_rank * 4) as u64
        );
        assert_eq!(
            prelude.kv_norm_weight.length(),
            (transform_arch.kv_lora_rank * 4) as u64
        );
        assert_eq!(
            prelude.cos.length(),
            (transform_arch.qk_rope_head_dim / 2 * 4) as u64
        );
    }

    #[test]
    fn resident_reserve_repairs_dsa_after_attention_only_growth() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let arch = tiny_arch();
        let mut session =
            ResidentSession::new(&ctx, &arch, 8).expect("initial expanded resident session");
        unsafe {
            *(session.dsa.index_keys[0].contents() as *mut f32) = 17.0;
        }
        session.seq_len = 8;

        session
            .attention
            .reserve(&ctx, &arch, 9, session.seq_len)
            .expect("simulate attention owner completing before DSA");
        assert_eq!(session.attention.capacity(), 16);
        assert_eq!(session.dsa.capacity, 8);

        session
            .reserve(&ctx, &arch, 9)
            .expect("retry independently repairs DSA owner");
        assert_eq!(session.attention.capacity(), 16);
        assert_eq!(session.dsa.capacity, 16);
        assert_eq!(session.dsa.sequence_scratch.capacity, 16);
        unsafe {
            assert_eq!(*(session.dsa.index_keys[0].contents() as *const f32), 17.0);
        }
    }

    #[test]
    fn dsa_rank_upload_preserves_stable_order_and_fails_closed() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let mut arch = tiny_arch();
        arch.index_topk = 3;
        let dsa = DsaIndexState::new(&ctx, &arch, 8, true, false).expect("DSA state");
        dsa.store_ranked_indices(&[4, 1, 3])
            .expect("stable ranked upload");
        assert_eq!(
            read_u32(dsa.ranked_indices().expect("rank buffer"), 3),
            vec![4, 1, 3]
        );

        let too_many = dsa
            .store_ranked_indices(&[0, 1, 2, 3])
            .expect_err("rank upload beyond fixed capacity must fail");
        assert!(too_many.to_string().contains("capacity"));

        if usize::BITS > u32::BITS {
            let overflow = dsa
                .store_ranked_indices(&[u32::MAX as usize + 1])
                .expect_err("rank upload beyond u32 must fail");
            assert!(overflow.to_string().contains("exceeds u32"));
        }
        let expanded_dsa =
            DsaIndexState::new(&ctx, &arch, 8, false, false).expect("expanded DSA state");
        assert!(expanded_dsa.ranked_indices.is_none());
        assert!(expanded_dsa.store_ranked_indices(&[0]).is_err());
    }

    #[test]
    fn compact_mla_cache_owner_grows_through_32k_and_preserves_state() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let arch = tiny_arch();
        let mut cache =
            CompactResidentCache::new(&ctx, &arch, 8192).expect("initial compact cache");
        assert_eq!(cache.layers.len(), arch.n_layers);
        assert_eq!(cache.capacity, 8192);
        assert_eq!(
            cache.layers[0].latents.length(),
            (8192 * arch.kv_lora_rank * 4) as u64
        );
        assert_eq!(
            cache.layers[0].rope_tails.length(),
            (8192 * arch.qk_rope_head_dim * 4) as u64
        );

        unsafe {
            let latents = cache.layers[0].latents.contents() as *mut f32;
            *latents = 1.0;
            *latents.add(8191 * arch.kv_lora_rank + (arch.kv_lora_rank - 1)) = 2.0;
            let rope = cache.layers[0].rope_tails.contents() as *mut f32;
            *rope = 3.0;
            *rope.add(8191 * arch.qk_rope_head_dim) = 4.0;
        }
        let latent_pointer = cache.layers[0].latents.contents();
        let rope_pointer = cache.layers[0].rope_tails.contents();
        cache
            .reserve(&ctx, &arch, 8192, 8192)
            .expect("adequate compact reserve");
        assert_eq!(cache.layers[0].latents.contents(), latent_pointer);
        assert_eq!(cache.layers[0].rope_tails.contents(), rope_pointer);

        cache
            .reserve(&ctx, &arch, 8193, 8192)
            .expect("grow compact cache beyond 8K");
        assert_eq!(cache.capacity, 16384);
        unsafe {
            let latents = cache.layers[0].latents.contents() as *mut f32;
            assert_eq!(*latents, 1.0);
            assert_eq!(
                *latents.add(8191 * arch.kv_lora_rank + (arch.kv_lora_rank - 1)),
                2.0
            );
            let rope = cache.layers[0].rope_tails.contents() as *mut f32;
            assert_eq!(*rope, 3.0);
            assert_eq!(*rope.add(8191 * arch.qk_rope_head_dim), 4.0);
            *latents.add(8192 * arch.kv_lora_rank) = 5.0;
            *latents.add(8192 * arch.kv_lora_rank + (arch.kv_lora_rank - 1)) = 6.0;
            *rope.add(8192 * arch.qk_rope_head_dim) = 7.0;
        }
        cache
            .reserve(&ctx, &arch, 32768, 8193)
            .expect("grow compact cache through 32K");
        assert_eq!(cache.capacity, 32768);
        unsafe {
            let latents = cache.layers[0].latents.contents() as *const f32;
            assert_eq!(*latents, 1.0);
            assert_eq!(
                *latents.add(8191 * arch.kv_lora_rank + (arch.kv_lora_rank - 1)),
                2.0
            );
            assert_eq!(*latents.add(8192 * arch.kv_lora_rank), 5.0);
            assert_eq!(
                *latents.add(8192 * arch.kv_lora_rank + (arch.kv_lora_rank - 1)),
                6.0
            );
            let rope = cache.layers[0].rope_tails.contents() as *const f32;
            assert_eq!(*rope, 3.0);
            assert_eq!(*rope.add(8191 * arch.qk_rope_head_dim), 4.0);
            assert_eq!(*rope.add(8192 * arch.qk_rope_head_dim), 7.0);
        }
        let invalid_seq_len = cache.capacity + 1;
        let error = cache
            .reserve(&ctx, &arch, invalid_seq_len, invalid_seq_len)
            .expect_err("invalid compact cache ownership state must fail closed");
        assert!(error.to_string().contains("seq_len"));
    }
}
