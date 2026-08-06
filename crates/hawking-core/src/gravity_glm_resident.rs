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
//! **Final-head replay** (`HAWKING_GLM_GPU_LM_HEAD_ICB=1`, default off):
//! captures final RMSNorm, native-BF16 or PQ projection, greedy argmax, and
//! diagnostic top-k once into a four-command compute ICB. All scalar arguments
//! live in one persistent buffer; stable-address warm tokens replay without
//! rebinding while retaining exact norm/head/sampling ledger composition.
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
//! `HAWKING_GLM_GPU_COMPACT_ATTENTION_ICB=1` additionally replays the
//! nine-command input/q/kv prelude, the full indexer's six fixed-grid
//! transforms, and the fixed-grid radix/compact-attention/residual post-score
//! DAG. Full-indexer layers group the contiguous nine- and six-command
//! pre-score ICBs behind one direct encoder. Exact active-length DSA scoring
//! remains directly encoded between the pre-score and post-score boundaries.
//!
//! Gated by [`GPU_RESIDENT_STATE_ENV`] (`HAWKING_GLM_GPU_RESIDENT_STATE`), default
//! off, so the host-state path remains the parity oracle.

#![cfg(target_os = "macos")]

use crate::gravity::matvec_dense;
use crate::gravity_glm::gpu::{
    encode_activation_aware_matvec, encode_argmax_f32, encode_gemv_native_bf16_seq,
    encode_sample_topk_f32, record_activation_aware_matvec_ops,
    record_routed_tensor_representation, routed_pq_representation, GpuTensor, GpuWeightCache,
};
use crate::gravity_glm::{
    gpu_compact_attention_icb_enabled, gpu_compact_mla_enabled, gpu_device_router_enabled,
    gpu_expert_table_hit_enabled, gpu_expert_table_icb_enabled, gpu_expert_wave_concurrent_enabled,
    gpu_expert_wave_enabled, gpu_lm_head_enabled, gpu_lm_head_full_logits_enabled,
    gpu_lm_head_icb_enabled, rope_cos_sin, rope_interleaved, topk_desc, BoundedLru, GlmArch,
    GlmTrace, WeightAccess, GPU_LM_HEAD_DIAG_TOPK, RESIDENT_RUNTIME_INITIAL_KV_CAPACITY_TOKENS,
};
use crate::metal::{
    MetalContext, ReplayBufferBinding, ReplayComputeStage, ReplayResourceDeclaration,
    ReplayableComputeGraph, TokenCommandBuffer,
};
use crate::{Error, Result};
use metal::{Buffer, MTLResourceUsage};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// Flag + static wait estimators live on `gravity_glm` so non-Metal unit tests
// can see them: `GPU_RESIDENT_STATE_ENV`, `gpu_resident_state_enabled`,
// `estimate_host_state_waits_per_token`, `estimate_resident_waits_per_token`.

/// Opt-in device-only SiLU on the **ordinary** three-batch resident MLP.
///
/// Default **off**. When set, `batched_mlp` keeps gate/up/act on device:
/// `gate + up` encode → device `silu(g)*u` → `down` consume, with no host
/// gate/up materialization and no activation re-upload. This is **not**
/// expert-wave: weighted combine and residual stay on the caller path, and
/// the expert-wave flag remains independently off.
pub const GPU_DEVICE_ONLY_MLP_ENV: &str = "HAWKING_GLM_GPU_DEVICE_ONLY_MLP";

/// Test-only poison: zero the device SiLU output so causal mutation fails.
/// Has no effect unless [`GPU_DEVICE_ONLY_MLP_ENV`] is also on.
pub const GPU_DEVICE_ONLY_MLP_POISON_ENV: &str = "HAWKING_GLM_GPU_DEVICE_ONLY_MLP_POISON";

/// Whether [`GPU_DEVICE_ONLY_MLP_ENV`] requests device-only SiLU on ordinary MLP.
pub fn gpu_device_only_mlp_enabled() -> bool {
    crate::env_on(GPU_DEVICE_ONLY_MLP_ENV)
}

fn gpu_device_only_mlp_poison_enabled() -> bool {
    crate::env_on(GPU_DEVICE_ONLY_MLP_POISON_ENV)
}

/// Process-global hit counter (works with ledger off). Reset via
/// [`reset_device_only_mlp_probe`].
static DEVICE_ONLY_MLP_HITS: AtomicU64 = AtomicU64::new(0);
static DEVICE_ONLY_MLP_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Reset probe counters used by the device-only MLP acceptance test.
pub fn reset_device_only_mlp_probe() {
    DEVICE_ONLY_MLP_HITS.store(0, Ordering::Relaxed);
    DEVICE_ONLY_MLP_FALLBACKS.store(0, Ordering::Relaxed);
}

/// Times the ordinary three-batch path took the device-only SiLU hit.
pub fn device_only_mlp_hits() -> u64 {
    DEVICE_ONLY_MLP_HITS.load(Ordering::Relaxed)
}

/// Times the flag was on but the path fell back to host SiLU.
pub fn device_only_mlp_fallbacks() -> u64 {
    DEVICE_ONLY_MLP_FALLBACKS.load(Ordering::Relaxed)
}

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
const DEVICE_EXPERT_TENSOR_KIND_ANY_SUPPORTED: u32 = 0;
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

#[derive(Clone)]
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

#[derive(Clone)]
struct PersistentDeviceExpertLayer {
    routed: DeviceExpertTableLease,
    shared: DeviceExpertTableLease,
    intermediate: usize,
    metrics: DeviceExpertLayerMetrics,
    routed_dispatch_mode: DeviceExpertDispatchMode,
    shared_dispatch_mode: DeviceExpertDispatchMode,
    replay_graph: Arc<Mutex<Option<CachedDeviceExpertReplayGraph>>>,
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
        GpuTensor::NativeCpu(_) | GpuTensor::ActivationAware { .. } => None,
    }
}

/// Snapshot the currently resident routed triplets for one layer.
///
/// The caller owns the cache guard while this walks the name-keyed LRU. Each
/// ready entry clones its backing Metal buffers into the returned lease, then
/// the descriptor bytes are uploaded once and never patched in place.
fn build_device_expert_table_snapshot_filtered(
    ctx: &MetalContext,
    cache: &BoundedLru<GpuTensor>,
    mlp_prefix: &str,
    n_experts: usize,
    generation: u32,
    selected_experts: Option<&[usize]>,
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
    if let Some(selected) = selected_experts {
        if let Some(expert) = selected.iter().copied().find(|&expert| expert >= n_experts) {
            return Err(Error::Gravity(format!(
                "device expert table selected expert {expert} exceeds layer extent {n_experts}"
            )));
        }
    }

    let mut entries = vec![DeviceExpertTriplet::default(); n_experts];
    let mut resources = Vec::new();
    let mut ready_entries = 0usize;
    for (expert, entry) in entries.iter_mut().enumerate() {
        if selected_experts.is_some_and(|selected| !selected.contains(&expert)) {
            continue;
        }
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
fn build_device_expert_table_snapshot(
    ctx: &MetalContext,
    cache: &BoundedLru<GpuTensor>,
    mlp_prefix: &str,
    n_experts: usize,
    generation: u32,
) -> Result<DeviceExpertTableLease> {
    build_device_expert_table_snapshot_filtered(ctx, cache, mlp_prefix, n_experts, generation, None)
}

fn build_selected_device_expert_table_snapshot(
    ctx: &MetalContext,
    cache: &BoundedLru<GpuTensor>,
    mlp_prefix: &str,
    n_experts: usize,
    generation: u32,
    selected_experts: &[usize],
) -> Result<DeviceExpertTableLease> {
    build_device_expert_table_snapshot_filtered(
        ctx,
        cache,
        mlp_prefix,
        n_experts,
        generation,
        Some(selected_experts),
    )
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
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DeviceExpertTableValidateParams {
    n_experts: u32,
    experts_per_token: u32,
    generation: u32,
    required_kind: u32,
    hidden: u32,
    intermediate: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DeviceExpertTableMatvecParams {
    n_experts: u32,
    experts_per_token: u32,
    generation: u32,
    execution_position: u32,
    projection: u32,
    rows: u32,
    cols: u32,
    allow_other_kind: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DeviceExpertTableAxpyParams {
    n: u32,
    experts_per_token: u32,
    execution_position: u32,
    use_router_weight: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DeviceExpertTraceCopyParams {
    count: u32,
    destination_offset: u32,
}

const _: [(); 24] = [(); std::mem::size_of::<DeviceExpertTableValidateParams>()];
const _: [(); 32] = [(); std::mem::size_of::<DeviceExpertTableMatvecParams>()];
const _: [(); 16] = [(); std::mem::size_of::<DeviceExpertTableAxpyParams>()];
const _: [(); 8] = [(); std::mem::size_of::<DeviceExpertTraceCopyParams>()];

fn encode_device_expert_trace_copy(
    tcb: &mut TokenCommandBuffer<'_>,
    expert_indices: &Buffer,
    expert_trace: &Buffer,
    count: usize,
    destination_offset: usize,
) -> Result<()> {
    let source_bytes = count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| Error::Gravity("device expert trace source byte overflow".into()))?;
    let trace_elements = destination_offset
        .checked_add(count)
        .ok_or_else(|| Error::Gravity("device expert trace range overflow".into()))?;
    let trace_bytes = trace_elements
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| Error::Gravity("device expert trace byte overflow".into()))?;
    if expert_indices.length() < source_bytes as u64 || expert_trace.length() < trace_bytes as u64 {
        return Err(Error::Gravity(format!(
            "device expert trace buffers are undersized: source={}/{} B trace={}/{} B",
            expert_indices.length(),
            source_bytes,
            expert_trace.length(),
            trace_bytes
        )));
    }
    let params = DeviceExpertTraceCopyParams {
        count: count as u32,
        destination_offset: destination_offset as u32,
    };
    let indices = expert_indices.clone();
    let trace = expert_trace.clone();
    const TG: u32 = 32;
    tcb.dispatch_threads(
        "gravity_glm_expert_trace_copy",
        ((count as u32).div_ceil(TG) * TG, 1, 1),
        (TG, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&indices), 0);
            enc.set_buffer(1, Some(&trace), 0);
            enc.set_bytes(
                2,
                std::mem::size_of_val(&params) as u64,
                &params as *const _ as *const _,
            );
        },
    )
}

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
    allow_other_kind: bool,
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
        allow_other_kind: u32::from(allow_other_kind),
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

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn encode_device_expert_table_native_bf16_matvec(
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
    allow_other_kind: bool,
) -> Result<()> {
    if execution_position >= experts_per_token || projection > 2 {
        return Err(Error::Gravity(format!(
            "invalid native device expert table matvec position/projection: position \
             {execution_position}/{experts_per_token}, projection {projection}"
        )));
    }
    let x_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Gravity("native device expert matvec input byte overflow".into()))?
        as u64;
    let y_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Gravity("native device expert matvec output byte overflow".into()))?
        as u64;
    if x.length() < x_bytes || y.length() < y_bytes {
        return Err(Error::Gravity(format!(
            "native device expert table matvec buffer too small: x={}/{} B y={}/{} B",
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
        allow_other_kind: u32::from(allow_other_kind),
    };
    let indices = expert_indices.clone();
    let slots = expert_exec_slots.clone();
    let table = lease.table.clone();
    let miss = miss_mask.clone();
    let xb = x.clone();
    let yb = y.clone();
    let resources = lease.resources.clone();
    const TG: u32 = 256;
    let grid = (rows as u32).div_ceil(TG) * TG;
    tcb.dispatch_threads(
        "gravity_glm_expert_table_native_bf16_matvec",
        (grid, 1, 1),
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

#[allow(clippy::too_many_arguments)]
fn encode_device_expert_table_matvec(
    tcb: &mut TokenCommandBuffer<'_>,
    mode: DeviceExpertDispatchMode,
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
    match mode {
        DeviceExpertDispatchMode::PqOnly => encode_device_expert_table_pq_matvec(
            tcb,
            lease,
            expert_indices,
            expert_exec_slots,
            miss_mask,
            experts_per_token,
            execution_position,
            projection,
            x,
            rows,
            cols,
            y,
            false,
        ),
        DeviceExpertDispatchMode::NativeBf16Only => encode_device_expert_table_native_bf16_matvec(
            tcb,
            lease,
            expert_indices,
            expert_exec_slots,
            miss_mask,
            experts_per_token,
            execution_position,
            projection,
            x,
            rows,
            cols,
            y,
            false,
        ),
        DeviceExpertDispatchMode::Heterogeneous => {
            encode_device_expert_table_pq_matvec(
                tcb,
                lease,
                expert_indices,
                expert_exec_slots,
                miss_mask,
                experts_per_token,
                execution_position,
                projection,
                x,
                rows,
                cols,
                y,
                true,
            )?;
            encode_device_expert_table_native_bf16_matvec(
                tcb,
                lease,
                expert_indices,
                expert_exec_slots,
                miss_mask,
                experts_per_token,
                execution_position,
                projection,
                x,
                rows,
                cols,
                y,
                true,
            )
        }
    }
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
    /// Four-byte guarded-wave hit/miss result. Validation overwrites it before
    /// every cache-indexed expert wave.
    expert_miss_mask: Buffer,
    /// Device-side diagnostic trace, indexed as layer × experts-per-token.
    /// Cache-indexed hits never download selections on the layer critical path.
    expert_trace: Buffer,
    /// Host-known singleton selection used by the shared-expert table lease.
    shared_expert_idx: Buffer,
    shared_expert_slot: Buffer,
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
    /// Grow-once scratch for the default-off **ordinary** device-only SiLU MLP
    /// path. Separate from expert-wave so that sealed-negative wave code is
    /// never entered from this lane.
    device_only_mlp_scratch: Mutex<Option<DeviceOnlyMlpScratch>>,
    /// At most one selected R4 route per layer. This bounds lease-pinned
    /// expert resources to the previous route footprint and lets warm hits
    /// reuse descriptor tables without a per-token rebuild/upload.
    persistent_expert_layers: Mutex<Vec<Option<PersistentDeviceExpertLayer>>>,
    /// Lazily captured fixed-shape final norm → lm-head → sampling graph.
    /// The key includes every bound GPU address, so warm tokens reuse the ICB
    /// without allocating and any storage change rebuilds it fail-closed.
    final_head_replay: Mutex<Option<CachedFinalHeadReplayGraph>>,
    /// One lazily captured fixed-grid compact-attention DAG per layer. Cache
    /// growth changes persistent addresses and therefore rebuilds only the
    /// affected layer entry.
    compact_attention_replay_layers: Mutex<Vec<Option<CachedCompactAttentionReplayGraph>>>,
    /// One lazily captured fixed-grid full-indexer pre-score DAG per layer.
    /// The exact active-length score stage remains direct.
    device_dsa_pre_score_replay_layers: Mutex<Vec<Option<CachedDeviceDsaPreScoreReplayGraph>>>,
    /// One lazily captured input/q/kv projection prelude per compact layer.
    /// Full-indexer and shared-indexer layers have the same prelude geometry.
    attention_prelude_replay_layers: Mutex<Vec<Option<CachedAttentionPreludeReplayGraph>>>,
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

/// Scratch for ordinary device-only SiLU three-batch MLP (not expert-wave).
struct DeviceOnlyMlpScratch {
    expert_capacity: usize,
    intermediate_capacity: usize,
    hidden_capacity: usize,
    gate: Vec<Buffer>,
    up: Vec<Buffer>,
    act: Vec<Buffer>,
    down: Vec<Buffer>,
}

impl DeviceOnlyMlpScratch {
    fn new(
        ctx: &MetalContext,
        expert_capacity: usize,
        intermediate_capacity: usize,
        hidden_capacity: usize,
    ) -> Result<Self> {
        if expert_capacity == 0 || intermediate_capacity == 0 || hidden_capacity == 0 {
            return Err(Error::Gravity(format!(
                "device-only MLP scratch has zero capacity: experts={expert_capacity} \
                 intermediate={intermediate_capacity} hidden={hidden_capacity}"
            )));
        }
        let inter_bytes = intermediate_capacity.checked_mul(4).ok_or_else(|| {
            Error::Gravity(format!(
                "device-only MLP intermediate scratch byte size overflow: {intermediate_capacity}"
            ))
        })?;
        let hidden_bytes = hidden_capacity.checked_mul(4).ok_or_else(|| {
            Error::Gravity(format!(
                "device-only MLP hidden scratch byte size overflow: {hidden_capacity}"
            ))
        })?;
        let mut gate = Vec::with_capacity(expert_capacity);
        let mut up = Vec::with_capacity(expert_capacity);
        let mut act = Vec::with_capacity(expert_capacity);
        let mut down = Vec::with_capacity(expert_capacity);
        for _ in 0..expert_capacity {
            gate.push(ctx.new_buffer_checked(inter_bytes)?);
            up.push(ctx.new_buffer_checked(inter_bytes)?);
            act.push(ctx.new_buffer_checked(inter_bytes)?);
            down.push(ctx.new_buffer_checked(hidden_bytes)?);
        }
        Ok(Self {
            expert_capacity,
            intermediate_capacity,
            hidden_capacity,
            gate,
            up,
            act,
            down,
        })
    }

    fn fits(&self, experts: usize, intermediate: usize, hidden: usize) -> bool {
        experts <= self.expert_capacity
            && intermediate <= self.intermediate_capacity
            && hidden <= self.hidden_capacity
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
            expert_miss_mask: ctx.new_buffer_checked(4)?,
            expert_trace: ctx.new_buffer_checked(
                arch.n_layers
                    .max(1)
                    .checked_mul(arch.num_experts_per_tok.max(1))
                    .and_then(|elements| elements.checked_mul(4))
                    .ok_or_else(|| Error::Gravity("expert trace byte size overflow".into()))?,
            )?,
            shared_expert_idx: ctx.new_buffer_with_bytes_checked(bytemuck::bytes_of(&0u32))?,
            shared_expert_slot: ctx.new_buffer_with_bytes_checked(bytemuck::bytes_of(&0u32))?,
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
            device_only_mlp_scratch: Mutex::new(None),
            persistent_expert_layers: Mutex::new((0..arch.n_layers).map(|_| None).collect()),
            final_head_replay: Mutex::new(None),
            compact_attention_replay_layers: Mutex::new((0..arch.n_layers).map(|_| None).collect()),
            device_dsa_pre_score_replay_layers: Mutex::new(
                (0..arch.n_layers).map(|_| None).collect(),
            ),
            attention_prelude_replay_layers: Mutex::new((0..arch.n_layers).map(|_| None).collect()),
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

    fn ensure_device_only_mlp_scratch(
        &self,
        ctx: &MetalContext,
        experts: usize,
        intermediate: usize,
        hidden: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<DeviceOnlyMlpScratch>>> {
        let mut scratch = self
            .device_only_mlp_scratch
            .lock()
            .expect("device-only MLP scratch");
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
            *scratch = Some(DeviceOnlyMlpScratch::new(
                ctx,
                expert_capacity,
                intermediate_capacity,
                hidden_capacity,
            )?);
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
        GpuTensor::ActivationAware {
            coefficients,
            basis,
            params,
        } => {
            if x_len != params.cols as usize {
                return Err(Error::Gravity(format!(
                    "resident matvec {name}: x_len {x_len} != cols {}",
                    params.cols
                )));
            }
            crate::cost_ledger::record_active_bytes_for(
                name,
                coefficients.length() + basis.length(),
            );
            record_activation_aware_matvec_ops(*params);
            let latent =
                ctx.new_buffer_checked(params.rank as usize * std::mem::size_of::<f32>())?;
            let tcb = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
            encode_activation_aware_matvec(tcb, coefficients, basis, *params, x, &latent, y)
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
        let device_expert_table = gpu_expert_table_hit_enabled();
        let mut device_expert_trace_layers = 0usize;
        let mut shared_topk = session.dsa.shared_topk.clone();
        trace.expert_choices.clear();

        for layer in 0..a.n_layers {
            let p = format!("model.layers.{layer}");
            let attn_p = format!("{p}.self_attn");
            let compact_attention = session.attention.is_compact();
            let mut tcb: Option<TokenCommandBuffer<'_>> = None;
            let mut deferred_attention_prelude: Option<Arc<ReplayableComputeGraph>> = None;

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

                    let replay_projections = if gpu_compact_attention_icb_enabled() {
                        device_replay_projection_triplet(
                            weights,
                            [q_a_name.as_str(), kv_a_name.as_str(), q_b_name.as_str()],
                        )?
                    } else {
                        None
                    };
                    if let Some(projections) = replay_projections {
                        record_device_replay_projection_cost(&q_a_name, &projections[0]);
                        record_device_replay_projection_cost(&kv_a_name, &projections[1]);
                        record_device_replay_projection_cost(&q_b_name, &projections[2]);
                        let inputs = AttentionPreludeReplayInputs {
                            layer,
                            hidden: a.hidden,
                            q_lora_rank: a.q_lora_rank,
                            kv_lora_rank: a.kv_lora_rank,
                            n_heads: a.n_heads,
                            qk_nope_dim: a.qk_nope_head_dim,
                            rope_dim: a.qk_rope_head_dim,
                            rms_norm_eps: a.rms_norm_eps,
                            projections: &projections,
                            x: &pool.x,
                            h: &pool.h,
                            q_a: &pool.q_a,
                            compressed: &pool.compressed,
                            q_resid: &pool.q_resid,
                            k_latent: &pool.k_latent,
                            q: &pool.q,
                            input_norm_weight: &prelude.input_norm_weight,
                            q_norm_weight: &prelude.q_norm_weight,
                            kv_norm_weight: &prelude.kv_norm_weight,
                            cos: &prelude.cos,
                            sin: &prelude.sin,
                            key_rope: &compact_scratch.key_rope,
                            query_nope: &compact_scratch.query_nope,
                            query_rope: &compact_scratch.query_rope,
                        };
                        let key = inputs.key();
                        let mut replay_layers = pool
                            .attention_prelude_replay_layers
                            .lock()
                            .expect("attention prelude replay layers");
                        let replay_layer_count = replay_layers.len();
                        let slot = replay_layers.get_mut(layer).ok_or_else(|| {
                            Error::Gravity(format!(
                                "attention prelude replay layer {layer} exceeds pool extent {replay_layer_count}"
                            ))
                        })?;
                        if !slot.as_ref().is_some_and(|entry| entry.key == key) {
                            *slot = Some(build_attention_prelude_replay_graph(ctx, &inputs)?);
                        }
                        let replay = slot
                            .as_ref()
                            .expect("attention prelude replay graph just populated");
                        if a.indexer_types[layer] == "full" {
                            deferred_attention_prelude = Some(Arc::clone(&replay.graph));
                        } else {
                            let wave = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
                            wave.execute_replayable_graph(&replay.graph)?;
                        }
                    } else {
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
                                layer,
                                &session.dsa.index_keys[layer],
                                &session.dsa.sequence_scratch.index_scores_device,
                                session.dsa.ranked_indices()?,
                                session.dsa.capacity,
                                pos,
                                &cos,
                                &sin,
                                closed_attention_prelude,
                                deferred_attention_prelude.as_deref(),
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
                    if !gpu_compact_attention_icb_enabled() {
                        let wave = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
                        route_segment_primitives::encode_residual_add_inplace(
                            wave, &pool.x, &pool.o, a.hidden,
                        )?;
                    }
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
                            if device_expert_table {
                                encode_device_expert_trace_copy(
                                    wave,
                                    &pool.expert_idx,
                                    &pool.expert_trace,
                                    a.num_experts_per_tok,
                                    device_expert_trace_layers
                                        .checked_mul(a.num_experts_per_tok)
                                        .ok_or_else(|| {
                                            Error::Gravity(
                                                "device expert trace layer offset overflow".into(),
                                            )
                                        })?,
                                )?;
                                device_expert_trace_layers =
                                    device_expert_trace_layers.saturating_add(1);
                            }
                        }
                        if !device_expert_table {
                            commit(tcb.take(), &session.waits)?;
                        }
                    }

                    let table_wave = if device_expert_table {
                        let generation = u32::try_from(layer.saturating_add(1)).map_err(|_| {
                            Error::Gravity(format!(
                                "device expert table layer generation overflow: {layer}"
                            ))
                        })?;
                        let _routed = cost_ledger::Scope::new(Bucket::RoutedExperts);
                        Some(moe_device_table_wave(
                            weights,
                            &prefix,
                            layer,
                            a.hidden,
                            a.num_experts_per_tok,
                            a.n_routed_experts,
                            generation,
                            &pool.h,
                            &pool.x,
                            pool,
                            &mut tcb,
                            ctx,
                            &session.waits,
                        )?)
                    } else {
                        None
                    };
                    let (table_hit, table_miss) = match table_wave {
                        Some(DeviceExpertTableWaveResult::Hit) => (true, false),
                        Some(DeviceExpertTableWaveResult::Miss(mask)) => {
                            debug_assert_ne!(mask, 0);
                            (false, true)
                        }
                        Some(DeviceExpertTableWaveResult::Unsupported) => {
                            // Shared layout could not use the guarded direct-u8
                            // graph. Commit router + trace before the ordinary
                            // host-known selection/fallback path.
                            commit(tcb.take(), &session.waits)?;
                            (false, false)
                        }
                        None => (false, false),
                    };

                    if !table_hit {
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
                        if !device_expert_table {
                            trace.expert_choices.push(indices.clone());
                        }

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
                        if table_miss {
                            let generation =
                                u32::try_from(layer.saturating_add(1)).map_err(|_| {
                                    Error::Gravity(format!(
                                        "device expert table refresh generation overflow: {layer}"
                                    ))
                                })?;
                            refresh_persistent_device_expert_layer(
                                weights,
                                &prefix,
                                layer,
                                a.hidden,
                                a.n_routed_experts,
                                generation,
                                &indices,
                                pool,
                                ctx,
                            )?;
                        }
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
                GpuTensor::NativeCpu(_)
                | GpuTensor::Pq { .. }
                | GpuTensor::ActivationAware { .. } => None,
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
                {
                    let _norm = cost_ledger::Scope::new(Bucket::Norm);
                    cost_ledger::record_source_modelled_operations(
                        (4 * a.hidden) as u64,
                        0,
                        0,
                        1,
                        0,
                    );
                }
                let rows = device_head.rows();
                let mut tcb = TokenCommandBuffer::new(ctx);
                if gpu_lm_head_icb_enabled() {
                    let key = final_head_replay_key(&device_head, pool, a.hidden, a.rms_norm_eps);
                    let mut cached = pool
                        .final_head_replay
                        .lock()
                        .expect("final-head replay graph");
                    if !cached.as_ref().is_some_and(|entry| entry.key == key) {
                        let graph = build_final_head_replay_graph(
                            ctx,
                            &device_head,
                            pool,
                            a.hidden,
                            a.rms_norm_eps,
                        )?;
                        *cached = Some(CachedFinalHeadReplayGraph { key, graph });
                    }
                    tcb.execute_replayable_graph(
                        &cached
                            .as_ref()
                            .expect("final-head replay graph just populated")
                            .graph,
                    )?;
                } else {
                    {
                        let _norm = cost_ledger::Scope::new(Bucket::Norm);
                        route_segment_primitives::encode_rmsnorm(
                            &mut tcb,
                            &pool.x,
                            &pool.final_norm_weight,
                            &pool.final_hidden,
                            a.hidden,
                            a.rms_norm_eps,
                        )?;
                    }
                    match &device_head {
                        DeviceHead::NativeBf16 { weight, rows, cols } => {
                            encode_gemv_native_bf16_seq(
                                &mut tcb,
                                weight,
                                *rows,
                                *cols,
                                &pool.final_hidden,
                                &pool.logits,
                            )?;
                        }
                        DeviceHead::Pq {
                            codebooks,
                            codes,
                            params,
                        } => {
                            encode_pq_matvec_device(
                                &mut tcb,
                                codebooks,
                                codes,
                                *params,
                                &pool.final_hidden,
                                &pool.logits,
                            )?;
                        }
                    }
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
                    }
                }
                {
                    let _sampling = cost_ledger::Scope::new(cost_ledger::Bucket::Sampling);
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

        if device_expert_table {
            let trace_elements = device_expert_trace_layers
                .checked_mul(a.num_experts_per_tok)
                .ok_or_else(|| Error::Gravity("device expert trace readback overflow".into()))?;
            let flat = read_u32(&pool.expert_trace, trace_elements);
            trace.expert_choices = flat
                .chunks_exact(a.num_experts_per_tok)
                .map(|layer| layer.iter().map(|&expert| expert as usize).collect())
                .collect();
            crate::cost_ledger::record_transfer(
                (trace_elements * std::mem::size_of::<u32>()) as u64,
                false,
                "device_expert_trace_download",
            );
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
    let ranked_indices = dsa.ranked_indices()?;
    if gpu_compact_attention_icb_enabled() && device_inputs_ready {
        let inputs = CompactAttentionReplayInputs {
            layer,
            hidden: a.hidden,
            n_heads: a.n_heads,
            latent_dim: a.kv_lora_rank,
            rope_dim: a.qk_rope_head_dim,
            key_rows: a.qk_nope_head_dim,
            row_stride,
            value_rows: a.v_head_dim,
            max_allow: a.index_topk,
            scale: (qk as f32).powf(-0.5),
            kv_params,
            o_params,
            k_latent: &pool.k_latent,
            key_rope: &scratch.key_rope,
            latent_cache: &cache.latents,
            rope_cache: &cache.rope_tails,
            kv_codebooks: &kv_codebooks,
            kv_codes: &kv_codes,
            query_nope: &scratch.query_nope,
            query_latent: &scratch.query_latent,
            query_rope: &scratch.query_rope,
            scores: (a.indexer_types[layer] == "full")
                .then_some(&dsa.sequence_scratch.index_scores_device),
            ranked_indices,
            context: &pool.context,
            o_codebooks: &o_codebooks,
            o_codes: &o_codes,
            output: &pool.o,
            residual: Some(&pool.x),
        };
        let key = inputs.key();
        let mut replay_layers = pool
            .compact_attention_replay_layers
            .lock()
            .expect("compact-attention replay layers");
        let replay_layer_count = replay_layers.len();
        let slot = replay_layers.get_mut(layer).ok_or_else(|| {
            Error::Gravity(format!(
                "compact-attention replay layer {layer} exceeds pool extent {replay_layer_count}"
            ))
        })?;
        if !slot.as_ref().is_some_and(|entry| entry.key == key) {
            *slot = Some(build_compact_attention_replay_graph(
                ctx, &inputs, pos, n_keys, n_allow,
            )?);
        }
        let replay = slot
            .as_ref()
            .expect("compact-attention replay graph just populated");
        replay.update_dynamic_parameters(pos, n_keys, n_allow)?;
        tcb.execute_replayable_graph(&replay.graph)?;
    } else {
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
            ranked_indices,
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
    }
    let compact_dispatches = tcb.dispatch_count().saturating_sub(dispatches_before);
    let expected_dispatches = if gpu_compact_attention_icb_enabled() && device_inputs_ready {
        6usize.saturating_add(usize::from(a.indexer_types[layer] == "full"))
    } else {
        5
    };
    if compact_dispatches != expected_dispatches {
        return Err(Error::Gravity(format!(
            "compact MLA expected {expected_dispatches} dispatches, encoded {compact_dispatches}"
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
/// `wq_b + wk + weights_proj → affine LayerNorm → q/k RoPE assembly` groups
/// with the deferred attention-prelude ICB when available. Exact active-length
/// DSA scores stay directly encoded in the caller's open command buffer; the
/// post-score replay starts with radix top-k and consumes its ranked u32 buffer
/// directly, so no projection, score, or rank readback lies on the attention
/// dependency path.
#[allow(clippy::too_many_arguments)]
fn indexer_topk_device<'a>(
    weights: &GpuWeightCache,
    arch: &GlmArch,
    attn_p: &str,
    pool: &ActPool,
    layer: usize,
    index_key_buffer: &Buffer,
    score_buffer: &Buffer,
    ranked_indices: &Buffer,
    cache_capacity: usize,
    pos: usize,
    cos: &[f32],
    sin: &[f32],
    replay_inputs_ready: bool,
    deferred_prelude_replay: Option<&ReplayableComputeGraph>,
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
) -> Result<usize> {
    let a = arch;
    let (ih, idim, rot) = (a.index_n_heads, a.index_head_dim, a.qk_rope_head_dim);
    let idx = format!("{attn_p}.indexer");
    let wq_name = format!("{idx}.wq_b.weight");
    let wk_name = format!("{idx}.wk.weight");
    let head_weight_name = format!("{idx}.weights_proj.weight");
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
    let projection_names = [
        wq_name.as_str(),
        wk_name.as_str(),
        head_weight_name.as_str(),
    ];
    let replay_projections = if gpu_compact_attention_icb_enabled() && replay_inputs_ready {
        device_replay_projection_triplet(weights, projection_names)?
    } else {
        None
    };
    if let Some(projections) = replay_projections {
        record_device_replay_projection_cost(&wq_name, &projections[0]);
        record_device_replay_projection_cost(&wk_name, &projections[1]);
        record_device_replay_projection_cost(&head_weight_name, &projections[2]);
        let inputs = DeviceDsaPreScoreReplayInputs {
            layer,
            n_heads: ih,
            head_dim: idim,
            rope_dim: rot,
            norm_eps: 1e-6,
            projections: &projections,
            q_resid: &pool.q_resid,
            h: &pool.h,
            idx_q: &pool.idx_q,
            idx_k_raw: &pool.idx_k_raw,
            idx_head_w: &pool.idx_head_w,
            norm_weight: &transform.norm_weight,
            norm_bias: &transform.norm_bias,
            cos: &transform.cos,
            sin: &transform.sin,
            query: &transform.query,
            index_keys: index_key_buffer,
        };
        let key = inputs.key();
        let mut replay_layers = pool
            .device_dsa_pre_score_replay_layers
            .lock()
            .expect("device DSA pre-score replay layers");
        let replay_layer_count = replay_layers.len();
        let slot = replay_layers.get_mut(layer).ok_or_else(|| {
            Error::Gravity(format!(
                "device DSA pre-score replay layer {layer} exceeds pool extent {replay_layer_count}"
            ))
        })?;
        if !slot.as_ref().is_some_and(|entry| entry.key == key) {
            *slot = Some(build_device_dsa_pre_score_replay_graph(ctx, &inputs, pos)?);
        }
        let replay = slot
            .as_ref()
            .expect("device DSA pre-score replay graph just populated");
        replay.update_position(pos)?;
        let wave = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
        if let Some(prelude) = deferred_prelude_replay {
            wave.execute_replayable_graphs(&[prelude, &replay.graph])?;
        } else {
            wave.execute_replayable_graph(&replay.graph)?;
        }
    } else {
        if let Some(prelude) = deferred_prelude_replay {
            let wave = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
            wave.execute_replayable_graph(prelude)?;
        }
        matvec_into(
            tcb,
            ctx,
            weights,
            &wq_name,
            &pool.q_resid,
            a.q_lora_rank,
            &pool.idx_q,
        )?;
        matvec_into(
            tcb,
            ctx,
            weights,
            &wk_name,
            &pool.h,
            a.hidden,
            &pool.idx_k_raw,
        )?;
        matvec_into(
            tcb,
            ctx,
            weights,
            &head_weight_name,
            &pool.h,
            a.hidden,
            &pool.idx_head_w,
        )?;
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
        route_segment_primitives::encode_rope_prefix_tail_positioned(
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
    }
    let wave = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
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
    if !(gpu_compact_attention_icb_enabled() && replay_inputs_ready) {
        route_segment_primitives::encode_radix_topk(wave, score_buffer, ranked_indices, n_keys, k)?;
    }
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
///
/// When [`gpu_device_only_mlp_enabled`] is set, the ordinary three-batch path
/// uses device SiLU (`gravity_silu_mul_f32`) so gate/up never materialize on
/// the host and activations are not re-uploaded for down. Expert-wave remains
/// a separate sealed-negative path and is never entered from here.
#[allow(clippy::too_many_arguments)]
fn batched_mlp<'a>(
    weights: &GpuWeightCache,
    prefixes: &[String],
    x: &Buffer,
    x_len: usize,
    pool: &ActPool,
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<Vec<Vec<f32>>> {
    if prefixes.is_empty() {
        return Ok(Vec::new());
    }
    if gpu_device_only_mlp_enabled() {
        match batched_mlp_device_only(weights, prefixes, x, x_len, pool, tcb, ctx, waits) {
            Ok(downs) => return Ok(downs),
            Err(e) => {
                // Fail closed only on hard errors; soft "cannot encode" falls
                // through to the host SiLU baseline below.
                let msg = e.to_string();
                if msg.contains("device-only MLP refuse") {
                    DEVICE_ONLY_MLP_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                    // continue to host path
                } else {
                    return Err(e);
                }
            }
        }
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
    // Physical proof counters: host path materializes gate/up for SiLU.
    let gate_up_bytes = (gate_outs.iter().map(Vec::len).sum::<usize>()
        + up_outs.iter().map(Vec::len).sum::<usize>())
    .saturating_mul(std::mem::size_of::<f32>()) as u64;
    crate::cost_ledger::record_mlp_gate_up_download(gate_up_bytes);
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
    let act_bytes = activation_elements.saturating_mul(std::mem::size_of::<f32>() as u64);
    crate::cost_ledger::record_mlp_activation_upload(act_bytes);
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

/// Ordinary three-batch MLP with device SiLU: gate/up encode → device
/// `silu(g)*u` → down encode. Never calls `moe_device_wave` or expert-wave
/// scratch. Returns host down vectors for the existing weighted-combine
/// caller (residual stays on the ordinary path).
///
/// **Command-buffer topology:** appends into the caller's open
/// [`TokenCommandBuffer`] (same pattern as `matvec_into` / residual encodes).
/// Gate, up, SiLU, and device-encodable downs share that single pending CB;
/// one `commit` fences the whole MLP. Private per-stage waves are forbidden —
/// each `TokenCommandBuffer::new` is a physical CB, and Drop used to
/// auto-commit empties (extra submit/wait with no useful work).
#[allow(clippy::too_many_arguments)]
fn batched_mlp_device_only<'a>(
    weights: &GpuWeightCache,
    prefixes: &[String],
    x: &Buffer,
    x_len: usize,
    pool: &ActPool,
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<Vec<Vec<f32>>> {
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

    let inter = {
        let cache = weights.cache.lock().expect("gpu weight cache");
        let gname = format!("{}.gate_proj.weight", prefixes[0]);
        match cache.get(&gname).expect("ensured gate") {
            GpuTensor::Pq { params, .. } => params.rows as usize,
            GpuTensor::NativeGpuBf16 { rows, .. } => *rows as usize,
            GpuTensor::ActivationAware { params, .. } => params.rows as usize,
            GpuTensor::NativeCpu(w) => {
                if x_len == 0 {
                    return Err(Error::Gravity(
                        "device-only MLP refuse: zero x_len for NativeCpu gate".into(),
                    ));
                }
                w.len() / x_len
            }
        }
    };
    if inter == 0 {
        return Err(Error::Gravity(
            "device-only MLP refuse: zero intermediate width".into(),
        ));
    }

    let n_exp = prefixes.len();
    let scratch_guard = pool.ensure_device_only_mlp_scratch(ctx, n_exp, inter, x_len)?;
    let scratch = scratch_guard
        .as_ref()
        .expect("device-only MLP scratch ensured");

    // Verify every projection is present (admission already pinned them).
    {
        let cache = weights.cache.lock().expect("gpu weight cache");
        for name in &all_names {
            if cache.get(name).is_none() {
                return Err(Error::Gravity(format!(
                    "device-only MLP refuse: missing tensor {name}"
                )));
            }
        }
    }

    // Classify host-native projections. NativeCpu matvecs `read_f32` their
    // inputs on the CPU, so any producer still pending in `tcb` must be
    // fenced first. Device-encodable projections append into the open CB.
    let mut host_gate_up = false;
    let mut host_down: Vec<bool> = Vec::with_capacity(n_exp);
    {
        let cache = weights.cache.lock().expect("gpu weight cache");
        for p in prefixes {
            let g = format!("{p}.gate_proj.weight");
            let u = format!("{p}.up_proj.weight");
            let d = format!("{p}.down_proj.weight");
            if matches!(cache.get(&g), Some(GpuTensor::NativeCpu(_)))
                || matches!(cache.get(&u), Some(GpuTensor::NativeCpu(_)))
            {
                host_gate_up = true;
            }
            host_down.push(matches!(
                cache.get(&d).expect("ensured down"),
                GpuTensor::NativeCpu(_)
            ));
        }
    }
    let any_host_down = host_down.iter().any(|&h| h);
    let poison = gpu_device_only_mlp_poison_enabled();

    // Host gate/up read `x` immediately. If prior attention/router work is
    // still pending in the open CB and wrote `x` (or its producers) on device,
    // fence it first — same safety as the host three-batch path's entry commit.
    // Pure device gate/up/silu/down skips this so SiLU co-issues with whatever
    // is already open (the topology fix vs a private per-MLP wave).
    if host_gate_up {
        commit(tcb.take(), waits)?;
    }

    // Encode gate + up + SiLU (+ device downs) into the caller's open CB.
    // NativeCpu projections write scratch immediately (no dispatch); device
    // projections and SiLU append dispatches. Do **not** open a private wave —
    // each `TokenCommandBuffer::new` is a physical CB, and empty Drop used to
    // auto-submit (the 1→5 per-token regression shape).
    {
        let wave = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
        for (i, p) in prefixes.iter().enumerate() {
            encode_weight_matvec(
                wave,
                weights,
                &format!("{p}.gate_proj.weight"),
                x,
                x_len,
                &scratch.gate[i],
            )?;
            encode_weight_matvec(
                wave,
                weights,
                &format!("{p}.up_proj.weight"),
                x,
                x_len,
                &scratch.up[i],
            )?;
        }
        for i in 0..n_exp {
            encode_silu_mul_f32(
                wave,
                &scratch.gate[i],
                &scratch.up[i],
                &scratch.act[i],
                inter as u32,
            )?;
        }
        // Device downs share this CB when the host does not need to mutate
        // activations between SiLU and down (poison path).
        if !poison {
            for (i, p) in prefixes.iter().enumerate() {
                if host_down[i] {
                    continue;
                }
                encode_weight_matvec(
                    wave,
                    weights,
                    &format!("{p}.down_proj.weight"),
                    &scratch.act[i],
                    inter,
                    &scratch.down[i],
                )?;
            }
        }
    }

    // One fence for this MLP's device encodes (and any co-issued prior work
    // when gate/up stayed on device). Required before host-native downs,
    // poison, or reading device downs back for the weighted combine.
    let needs_fence =
        tcb.as_ref().is_some_and(|w| w.dispatch_count() > 0) || any_host_down || poison;
    if needs_fence {
        commit(tcb.take(), waits)?;
    }

    if poison {
        // Causal mutation after the real SiLU fence: overwrite activations so
        // any test requiring logit/token identity against the healthy path
        // fails, while still counting a device-only hit.
        let poison_vec = vec![1.0f32; inter];
        for i in 0..n_exp {
            write_f32(&scratch.act[i], &poison_vec);
        }
        // Device downs after poison: encode into a fresh open CB (same
        // get_or_insert pattern — not a private throwaway wave).
        let encoded_poison_down = {
            let wave = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
            let before = wave.dispatch_count();
            for (i, p) in prefixes.iter().enumerate() {
                if host_down[i] {
                    continue;
                }
                encode_weight_matvec(
                    wave,
                    weights,
                    &format!("{p}.down_proj.weight"),
                    &scratch.act[i],
                    inter,
                    &scratch.down[i],
                )?;
            }
            wave.dispatch_count() > before
        };
        if encoded_poison_down {
            commit(tcb.take(), waits)?;
        }
    }

    // Host-native downs: activations are fenced; matvec_into writes scratch
    // without opening a CB for NativeCpu (and without a Drop-auto-commit).
    for (i, p) in prefixes.iter().enumerate() {
        if !host_down[i] {
            continue;
        }
        matvec_into(
            tcb,
            ctx,
            weights,
            &format!("{p}.down_proj.weight"),
            &scratch.act[i],
            inter,
            &scratch.down[i],
        )?;
    }

    // Read only down outputs for the existing host weighted combine.
    // Gate/up/act are never downloaded — transfer counters stay zero.
    let mut downs = Vec::with_capacity(n_exp);
    for i in 0..n_exp {
        downs.push(read_f32(&scratch.down[i], x_len));
    }
    crate::cost_ledger::record_device_only_mlp_hit();
    DEVICE_ONLY_MLP_HITS.fetch_add(1, Ordering::Relaxed);
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
    #[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
    pub(super) struct GlmRopeParams {
        pub n_heads: u32,
        pub rotary_dim: u32,
        pub in_stride: u32,
        pub out_stride: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
    pub(super) struct GlmPositionedRopeParams {
        pub n_heads: u32,
        pub rotary_dim: u32,
        pub in_stride: u32,
        pub out_stride: u32,
        pub input_element_offset: u32,
        pub output_element_offset: u32,
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
    #[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
    pub(super) struct GlmMlaCompactAppendParams {
        pub latent_dim: u32,
        pub rope_dim: u32,
        pub pos: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
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
    #[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    pub(super) struct GlmCompactRankedAttnParams {
        pub n_heads: u32,
        pub latent_dim: u32,
        pub rope_dim: u32,
        pub n_keys: u32,
        pub n_allow: u32,
        pub scale: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
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
    #[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
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
    #[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
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
    const _: [(); 24] = [(); std::mem::size_of::<GlmPositionedRopeParams>()];
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
    const _: [(); 4] = [(); std::mem::align_of::<GlmPositionedRopeParams>()];
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

    /// Replay-safe indexer RoPE assembly. Unlike [`encode_rope_prefix_tail`],
    /// both element offsets live in the parameter ABI and the full buffers are
    /// bound at offset zero, so position can change without rebuilding an ICB.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_rope_prefix_tail_positioned(
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
                "gravity_rope_prefix_tail_positioned_f32 invalid geometry: heads={n_heads}, rotary_dim={rotary_dim}, in_stride={in_stride}, out_stride={out_stride}"
            )));
        }
        if x.contents() == out.contents() {
            return Err(Error::Gravity(
                "gravity_rope_prefix_tail_positioned_f32 requires non-aliasing input/output".into(),
            ));
        }
        let input_len = strided_elements(
            n_heads,
            in_stride,
            out_stride,
            "gravity_rope_prefix_tail_positioned_f32 input",
        )?;
        let output_len = checked_mul(
            n_heads,
            out_stride,
            "gravity_rope_prefix_tail_positioned_f32 output",
        )?;
        require_f32(
            x,
            input_element_offset,
            input_len,
            "gravity_rope_prefix_tail_positioned_f32 input",
        )?;
        require_f32(
            out,
            output_element_offset,
            output_len,
            "gravity_rope_prefix_tail_positioned_f32 output",
        )?;
        require_f32(
            cos,
            0,
            rotary_dim / 2,
            "gravity_rope_prefix_tail_positioned_f32 cos",
        )?;
        require_f32(
            sin,
            0,
            rotary_dim / 2,
            "gravity_rope_prefix_tail_positioned_f32 sin",
        )?;
        let params = GlmPositionedRopeParams {
            n_heads: u32_arg(n_heads, "gravity_rope_prefix_tail_positioned_f32 n_heads")?,
            rotary_dim: u32_arg(
                rotary_dim,
                "gravity_rope_prefix_tail_positioned_f32 rotary_dim",
            )?,
            in_stride: u32_arg(
                in_stride,
                "gravity_rope_prefix_tail_positioned_f32 in_stride",
            )?,
            out_stride: u32_arg(
                out_stride,
                "gravity_rope_prefix_tail_positioned_f32 out_stride",
            )?,
            input_element_offset: u32_arg(
                input_element_offset,
                "gravity_rope_prefix_tail_positioned_f32 input offset",
            )?,
            output_element_offset: u32_arg(
                output_element_offset,
                "gravity_rope_prefix_tail_positioned_f32 output offset",
            )?,
        };
        let threads = params
            .n_heads
            .checked_mul(params.out_stride)
            .ok_or_else(|| {
                Error::Gravity("gravity_rope_prefix_tail_positioned_f32 grid overflow".into())
            })?;
        let grid = grid_1d(threads, "gravity_rope_prefix_tail_positioned_f32")?;
        let xb = x.clone();
        let ob = out.clone();
        let cb = cos.clone();
        let sb = sin.clone();
        tcb.dispatch_threads(
            "gravity_rope_prefix_tail_positioned_f32",
            grid,
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&xb), 0);
                enc.set_buffer(1, Some(&ob), 0);
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
        GpuTensor::ActivationAware {
            coefficients,
            basis,
            params,
        } => {
            if x_len != params.cols as usize {
                return Err(Error::Gravity(format!(
                    "expert-wave matvec {name}: x_len {x_len} != cols {}",
                    params.cols
                )));
            }
            crate::cost_ledger::record_active_bytes_for(
                name,
                coefficients.length() + basis.length(),
            );
            record_activation_aware_matvec_ops(*params);
            let latent = weights
                .ctx
                .new_buffer_checked(params.rank as usize * std::mem::size_of::<f32>())?;
            encode_activation_aware_matvec(tcb, coefficients, basis, *params, x, &latent, y)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceExpertTableWaveResult {
    Hit,
    Miss(u32),
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceExpertDispatchMode {
    PqOnly,
    NativeBf16Only,
    Heterogeneous,
}

#[derive(Clone)]
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

impl DeviceHead {
    fn rows(&self) -> u32 {
        match self {
            Self::NativeBf16 { rows, .. } => *rows,
            Self::Pq { params, .. } => params.rows,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinalHeadReplayGeometry {
    NativeBf16 { rows: u32, cols: u32 },
    Pq(crate::gravity_glm::gpu::PqParams),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FinalHeadReplayKey {
    geometry: FinalHeadReplayGeometry,
    hidden: usize,
    rms_norm_eps_bits: u32,
    buffer_addresses: [u64; 9],
}

struct CachedFinalHeadReplayGraph {
    key: FinalHeadReplayKey,
    graph: ReplayableComputeGraph,
}

#[derive(Default)]
struct ReplayParameterArena {
    bytes: Vec<u8>,
}

impl ReplayParameterArena {
    fn push<T: bytemuck::Pod>(&mut self, value: &T) -> usize {
        let align = std::mem::align_of::<T>();
        let padding = (align - (self.bytes.len() % align)) % align;
        self.bytes
            .resize(self.bytes.len().saturating_add(padding), 0);
        let offset = self.bytes.len();
        self.bytes.extend_from_slice(bytemuck::bytes_of(value));
        offset
    }

    fn finish(self, ctx: &MetalContext, label: &str) -> Result<Buffer> {
        if self.bytes.is_empty() {
            return Err(Error::Gravity(format!(
                "{label} has no persistent parameters"
            )));
        }
        let buffer = ctx.new_buffer_with_bytes_checked(&self.bytes)?;
        crate::cost_ledger::record_allocation(buffer.length());
        Ok(buffer)
    }
}

fn write_replay_parameter<T: bytemuck::Pod>(
    buffer: &Buffer,
    offset: usize,
    value: &T,
    label: &str,
) -> Result<()> {
    let bytes = bytemuck::bytes_of(value);
    let end = offset
        .checked_add(bytes.len())
        .ok_or_else(|| Error::Gravity(format!("{label} parameter offset overflow")))?;
    if end as u64 > buffer.length() {
        return Err(Error::Gravity(format!(
            "{label} parameter range [{offset}, {end}) exceeds {} bytes",
            buffer.length()
        )));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            (buffer.contents() as *mut u8).add(offset),
            bytes.len(),
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceReplayProjectionGeometry {
    NativeBf16 { rows: u32, cols: u32 },
    Pq(crate::gravity_glm::gpu::PqParams),
}

#[derive(Clone)]
enum DeviceReplayProjection {
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

impl DeviceReplayProjection {
    fn geometry(&self) -> DeviceReplayProjectionGeometry {
        match self {
            Self::NativeBf16 { rows, cols, .. } => DeviceReplayProjectionGeometry::NativeBf16 {
                rows: *rows,
                cols: *cols,
            },
            Self::Pq { params, .. } => DeviceReplayProjectionGeometry::Pq(*params),
        }
    }

    fn rows_cols(&self) -> (u32, u32) {
        match self.geometry() {
            DeviceReplayProjectionGeometry::NativeBf16 { rows, cols } => (rows, cols),
            DeviceReplayProjectionGeometry::Pq(params) => (params.rows, params.cols),
        }
    }

    fn append_addresses(&self, addresses: &mut Vec<u64>) {
        match self {
            Self::NativeBf16 { weight, .. } => addresses.push(weight.gpu_address()),
            Self::Pq {
                codebooks, codes, ..
            } => {
                addresses.push(codebooks.gpu_address());
                addresses.push(codes.gpu_address());
            }
        }
    }
}

fn device_replay_projection_triplet(
    weights: &GpuWeightCache,
    names: [&str; 3],
) -> Result<Option<[DeviceReplayProjection; 3]>> {
    let mut cache = weights.cache.lock().expect("gpu weight cache");
    weights.ensure_many_locked(&mut cache, &names)?;
    let projection = |name: &str| -> Option<DeviceReplayProjection> {
        let tensor = cache.get(name).expect("ensured replay projection");
        record_routed_tensor_representation(name, tensor);
        match tensor {
            GpuTensor::NativeGpuBf16 { buf, rows, cols } => {
                Some(DeviceReplayProjection::NativeBf16 {
                    weight: buf.clone(),
                    rows: *rows,
                    cols: *cols,
                })
            }
            GpuTensor::Pq {
                codebooks,
                codes,
                params,
            } => Some(DeviceReplayProjection::Pq {
                codebooks: codebooks.clone(),
                codes: codes.clone(),
                params: *params,
            }),
            GpuTensor::NativeCpu(_) | GpuTensor::ActivationAware { .. } => None,
        }
    };
    let Some(first) = projection(names[0]) else {
        return Ok(None);
    };
    let Some(second) = projection(names[1]) else {
        return Ok(None);
    };
    let Some(third) = projection(names[2]) else {
        return Ok(None);
    };
    Ok(Some([first, second, third]))
}

fn record_device_replay_projection_cost(name: &str, projection: &DeviceReplayProjection) {
    crate::cost_ledger::record_matvec_call();
    match projection {
        DeviceReplayProjection::NativeBf16 { weight, rows, cols } => {
            crate::cost_ledger::record_active_bytes_for(name, weight.length());
            record_dense_matvec_ops(*rows as u64, *cols as u64);
        }
        DeviceReplayProjection::Pq {
            codebooks,
            codes,
            params,
        } => {
            crate::cost_ledger::record_active_bytes_for(
                name,
                codebooks.length().saturating_add(codes.length()),
            );
            record_pq_matvec_ops(*params);
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ReplayProjectionParameterOffsets {
    NativeBf16 { rows: usize, cols: usize },
    Pq { params: usize },
}

fn append_replay_projection_parameters(
    parameters: &mut ReplayParameterArena,
    projection: &DeviceReplayProjection,
) -> ReplayProjectionParameterOffsets {
    match projection {
        DeviceReplayProjection::NativeBf16 { rows, cols, .. } => {
            ReplayProjectionParameterOffsets::NativeBf16 {
                rows: parameters.push(rows),
                cols: parameters.push(cols),
            }
        }
        DeviceReplayProjection::Pq { params, .. } => ReplayProjectionParameterOffsets::Pq {
            params: parameters.push(params),
        },
    }
}

fn build_replay_projection_stage(
    projection: &DeviceReplayProjection,
    input: &Buffer,
    output: &Buffer,
    parameter_buffer: &Buffer,
    offsets: ReplayProjectionParameterOffsets,
    label: &str,
) -> Result<ReplayComputeStage> {
    const TG: u32 = 256;
    let stage = match (projection, offsets) {
        (
            DeviceReplayProjection::NativeBf16 { weight, rows, .. },
            ReplayProjectionParameterOffsets::NativeBf16 {
                rows: rows_offset,
                cols: cols_offset,
            },
        ) => ReplayComputeStage::new(
            "gemv_native_bf16_seq",
            (replay_grid(*rows, TG, TG, label)?, 1, 1),
            (TG, 1, 1),
            vec![
                ReplayBufferBinding::read(0, weight, 0),
                ReplayBufferBinding::read(1, input, 0),
                ReplayBufferBinding::write(2, output, 0),
                ReplayBufferBinding::read(3, parameter_buffer, rows_offset),
                ReplayBufferBinding::read(4, parameter_buffer, cols_offset),
            ],
        ),
        (
            DeviceReplayProjection::Pq {
                codebooks,
                codes,
                params,
            },
            ReplayProjectionParameterOffsets::Pq {
                params: params_offset,
            },
        ) => ReplayComputeStage::new(
            "gravity_pq_matvec",
            (replay_grid(params.rows, 8, TG, label)?, 1, 1),
            (TG, 1, 1),
            vec![
                ReplayBufferBinding::read(0, codebooks, 0),
                ReplayBufferBinding::read(1, codes, 0),
                ReplayBufferBinding::read(2, input, 0),
                ReplayBufferBinding::write(3, output, 0),
                ReplayBufferBinding::read(4, parameter_buffer, params_offset),
            ],
        ),
        _ => {
            return Err(Error::Gravity(format!(
                "{label} projection geometry and parameter layout disagree"
            )))
        }
    };
    Ok(stage.with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AttentionPreludeReplayKey {
    layer: usize,
    hidden: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    n_heads: usize,
    qk_nope_dim: usize,
    rope_dim: usize,
    rms_norm_eps_bits: u32,
    projection_geometry: [DeviceReplayProjectionGeometry; 3],
    buffer_addresses: Vec<u64>,
}

struct CachedAttentionPreludeReplayGraph {
    key: AttentionPreludeReplayKey,
    graph: Arc<ReplayableComputeGraph>,
}

struct AttentionPreludeReplayInputs<'a> {
    layer: usize,
    hidden: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    n_heads: usize,
    qk_nope_dim: usize,
    rope_dim: usize,
    rms_norm_eps: f32,
    projections: &'a [DeviceReplayProjection; 3],
    x: &'a Buffer,
    h: &'a Buffer,
    q_a: &'a Buffer,
    compressed: &'a Buffer,
    q_resid: &'a Buffer,
    k_latent: &'a Buffer,
    q: &'a Buffer,
    input_norm_weight: &'a Buffer,
    q_norm_weight: &'a Buffer,
    kv_norm_weight: &'a Buffer,
    cos: &'a Buffer,
    sin: &'a Buffer,
    key_rope: &'a Buffer,
    query_nope: &'a Buffer,
    query_rope: &'a Buffer,
}

impl AttentionPreludeReplayInputs<'_> {
    fn key(&self) -> AttentionPreludeReplayKey {
        let mut buffer_addresses = Vec::with_capacity(21);
        for projection in self.projections {
            projection.append_addresses(&mut buffer_addresses);
        }
        buffer_addresses.extend([
            self.x.gpu_address(),
            self.h.gpu_address(),
            self.q_a.gpu_address(),
            self.compressed.gpu_address(),
            self.q_resid.gpu_address(),
            self.k_latent.gpu_address(),
            self.q.gpu_address(),
            self.input_norm_weight.gpu_address(),
            self.q_norm_weight.gpu_address(),
            self.kv_norm_weight.gpu_address(),
            self.cos.gpu_address(),
            self.sin.gpu_address(),
            self.key_rope.gpu_address(),
            self.query_nope.gpu_address(),
            self.query_rope.gpu_address(),
        ]);
        AttentionPreludeReplayKey {
            layer: self.layer,
            hidden: self.hidden,
            q_lora_rank: self.q_lora_rank,
            kv_lora_rank: self.kv_lora_rank,
            n_heads: self.n_heads,
            qk_nope_dim: self.qk_nope_dim,
            rope_dim: self.rope_dim,
            rms_norm_eps_bits: self.rms_norm_eps.to_bits(),
            projection_geometry: [
                self.projections[0].geometry(),
                self.projections[1].geometry(),
                self.projections[2].geometry(),
            ],
            buffer_addresses,
        }
    }
}

fn build_attention_prelude_replay_graph(
    ctx: &MetalContext,
    inputs: &AttentionPreludeReplayInputs<'_>,
) -> Result<CachedAttentionPreludeReplayGraph> {
    const TG: u32 = 256;
    let qk = inputs
        .qk_nope_dim
        .checked_add(inputs.rope_dim)
        .ok_or_else(|| Error::Gravity("attention prelude replay qk overflow".into()))?;
    if inputs.hidden == 0
        || inputs.q_lora_rank == 0
        || inputs.kv_lora_rank == 0
        || inputs.n_heads == 0
        || inputs.qk_nope_dim == 0
        || inputs.rope_dim == 0
        || inputs.rope_dim % 2 != 0
    {
        return Err(Error::Gravity(format!(
            "attention prelude replay has invalid geometry: hidden={} q_lora={} kv_lora={} heads={} qk_nope={} rope={}",
            inputs.hidden,
            inputs.q_lora_rank,
            inputs.kv_lora_rank,
            inputs.n_heads,
            inputs.qk_nope_dim,
            inputs.rope_dim
        )));
    }
    let (q_a_rows, q_a_cols) = inputs.projections[0].rows_cols();
    let (kv_a_rows, kv_a_cols) = inputs.projections[1].rows_cols();
    let (q_b_rows, q_b_cols) = inputs.projections[2].rows_cols();
    let expected_q_b_rows = inputs
        .n_heads
        .checked_mul(qk)
        .ok_or_else(|| Error::Gravity("attention prelude replay q_b rows overflow".into()))?;
    let expected_kv_rows = inputs
        .kv_lora_rank
        .checked_add(inputs.rope_dim)
        .ok_or_else(|| Error::Gravity("attention prelude replay kv_a rows overflow".into()))?;
    if q_a_rows as usize != inputs.q_lora_rank
        || q_a_cols as usize != inputs.hidden
        || kv_a_rows as usize != expected_kv_rows
        || kv_a_cols as usize != inputs.hidden
        || q_b_rows as usize != expected_q_b_rows
        || q_b_cols as usize != inputs.q_lora_rank
    {
        return Err(Error::Gravity(format!(
            "attention prelude replay projection mismatch: q_a={q_a_rows}x{q_a_cols}, kv_a={kv_a_rows}x{kv_a_cols}, q_b={q_b_rows}x{q_b_cols}; expected {}x{}, {}x{}, {}x{}",
            inputs.q_lora_rank,
            inputs.hidden,
            expected_kv_rows,
            inputs.hidden,
            expected_q_b_rows,
            inputs.q_lora_rank
        )));
    }

    let mut parameters = ReplayParameterArena::default();
    let projection_offsets = [
        append_replay_projection_parameters(&mut parameters, &inputs.projections[0]),
        append_replay_projection_parameters(&mut parameters, &inputs.projections[1]),
        append_replay_projection_parameters(&mut parameters, &inputs.projections[2]),
    ];
    let hidden = replay_u32(inputs.hidden, "attention prelude replay hidden")?;
    let q_lora = replay_u32(inputs.q_lora_rank, "attention prelude replay q_lora")?;
    let kv_lora = replay_u32(inputs.kv_lora_rank, "attention prelude replay kv_lora")?;
    let hidden_offset = parameters.push(&hidden);
    let q_lora_offset = parameters.push(&q_lora);
    let kv_lora_offset = parameters.push(&kv_lora);
    let eps_offset = parameters.push(&inputs.rms_norm_eps);
    let key_rope_params = route_segment_primitives::GlmRopeParams {
        n_heads: 1,
        rotary_dim: replay_u32(inputs.rope_dim, "attention prelude replay RoPE dimension")?,
        in_stride: replay_u32(inputs.rope_dim, "attention prelude replay key input stride")?,
        out_stride: replay_u32(
            inputs.rope_dim,
            "attention prelude replay key output stride",
        )?,
    };
    let key_rope_parameter_offset = parameters.push(&key_rope_params);
    let copy_params = route_segment_primitives::GlmBuildQParams {
        n_heads: replay_u32(inputs.n_heads, "attention prelude replay head count")?,
        qk_nope: replay_u32(inputs.qk_nope_dim, "attention prelude replay qk_nope")?,
        qk_rope: key_rope_params.rotary_dim,
    };
    let copy_parameter_offset = parameters.push(&copy_params);
    let query_rope_params = route_segment_primitives::GlmRopeParams {
        n_heads: copy_params.n_heads,
        rotary_dim: key_rope_params.rotary_dim,
        in_stride: replay_u32(qk, "attention prelude replay query input stride")?,
        out_stride: key_rope_params.rotary_dim,
    };
    let query_rope_parameter_offset = parameters.push(&query_rope_params);
    let parameter_buffer = parameters.finish(ctx, "attention prelude replay graph")?;

    let input_norm = ReplayComputeStage::new(
        "gravity_rmsnorm_f32",
        (TG, 1, 1),
        (TG, 1, 1),
        vec![
            ReplayBufferBinding::read(0, inputs.x, 0),
            ReplayBufferBinding::read(1, inputs.input_norm_weight, 0),
            ReplayBufferBinding::write(2, inputs.h, 0),
            ReplayBufferBinding::read(3, &parameter_buffer, hidden_offset),
            ReplayBufferBinding::read(4, &parameter_buffer, eps_offset),
        ],
    )
    .with_threadgroup_memory_length(0, TG as usize * 4)
    .with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare);
    let q_a = build_replay_projection_stage(
        &inputs.projections[0],
        inputs.h,
        inputs.q_a,
        &parameter_buffer,
        projection_offsets[0],
        "attention prelude replay q_a",
    )?
    .with_barrier_before();
    let kv_a = build_replay_projection_stage(
        &inputs.projections[1],
        inputs.h,
        inputs.compressed,
        &parameter_buffer,
        projection_offsets[1],
        "attention prelude replay kv_a",
    )?
    .with_barrier_before();
    let q_norm = ReplayComputeStage::new(
        "gravity_rmsnorm_f32",
        (TG, 1, 1),
        (TG, 1, 1),
        vec![
            ReplayBufferBinding::read(0, inputs.q_a, 0),
            ReplayBufferBinding::read(1, inputs.q_norm_weight, 0),
            ReplayBufferBinding::write(2, inputs.q_resid, 0),
            ReplayBufferBinding::read(3, &parameter_buffer, q_lora_offset),
            ReplayBufferBinding::read(4, &parameter_buffer, eps_offset),
        ],
    )
    .with_threadgroup_memory_length(0, TG as usize * 4)
    .with_barrier_before()
    .with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare);
    let kv_norm = ReplayComputeStage::new(
        "gravity_rmsnorm_f32",
        (TG, 1, 1),
        (TG, 1, 1),
        vec![
            ReplayBufferBinding::read(0, inputs.compressed, 0),
            ReplayBufferBinding::read(1, inputs.kv_norm_weight, 0),
            ReplayBufferBinding::write(2, inputs.k_latent, 0),
            ReplayBufferBinding::read(3, &parameter_buffer, kv_lora_offset),
            ReplayBufferBinding::read(4, &parameter_buffer, eps_offset),
        ],
    )
    .with_threadgroup_memory_length(0, TG as usize * 4)
    .with_barrier_before()
    .with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare);
    let key_input_byte_offset = inputs
        .kv_lora_rank
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Gravity("attention prelude key RoPE offset overflow".into()))?;
    let key_rope_threads = key_rope_params.rotary_dim / 2;
    let key_rope = ReplayComputeStage::new(
        "gravity_rope_interleaved_f32",
        (
            replay_grid(
                key_rope_threads,
                TG,
                TG,
                "attention prelude replay key RoPE",
            )?,
            1,
            1,
        ),
        (TG, 1, 1),
        vec![
            ReplayBufferBinding::read(0, inputs.compressed, key_input_byte_offset),
            ReplayBufferBinding::write(1, inputs.key_rope, 0),
            ReplayBufferBinding::read(2, inputs.cos, 0),
            ReplayBufferBinding::read(3, inputs.sin, 0),
            ReplayBufferBinding::read(4, &parameter_buffer, key_rope_parameter_offset),
        ],
    )
    .with_barrier_before()
    .with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare);
    let q_b = build_replay_projection_stage(
        &inputs.projections[2],
        inputs.q_resid,
        inputs.q,
        &parameter_buffer,
        projection_offsets[2],
        "attention prelude replay q_b",
    )?
    .with_barrier_before();
    let copy_elements = copy_params
        .n_heads
        .checked_mul(copy_params.qk_nope)
        .ok_or_else(|| Error::Gravity("attention prelude replay prefix grid overflow".into()))?;
    let copy_prefix = ReplayComputeStage::new(
        "gravity_copy_head_prefix_f32",
        (
            replay_grid(
                copy_elements,
                TG,
                TG,
                "attention prelude replay query prefix",
            )?,
            1,
            1,
        ),
        (TG, 1, 1),
        vec![
            ReplayBufferBinding::read(0, inputs.q, 0),
            ReplayBufferBinding::write(1, inputs.query_nope, 0),
            ReplayBufferBinding::read(2, &parameter_buffer, copy_parameter_offset),
        ],
    )
    .with_barrier_before()
    .with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare);
    let query_input_byte_offset = inputs
        .qk_nope_dim
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Gravity("attention prelude query RoPE offset overflow".into()))?;
    let query_rope_threads = query_rope_params
        .n_heads
        .checked_mul(query_rope_params.rotary_dim / 2)
        .ok_or_else(|| Error::Gravity("attention prelude query RoPE grid overflow".into()))?;
    let query_rope = ReplayComputeStage::new(
        "gravity_rope_interleaved_f32",
        (
            replay_grid(
                query_rope_threads,
                TG,
                TG,
                "attention prelude replay query RoPE",
            )?,
            1,
            1,
        ),
        (TG, 1, 1),
        vec![
            ReplayBufferBinding::read(0, inputs.q, query_input_byte_offset),
            ReplayBufferBinding::write(1, inputs.query_rope, 0),
            ReplayBufferBinding::read(2, inputs.cos, 0),
            ReplayBufferBinding::read(3, inputs.sin, 0),
            ReplayBufferBinding::read(4, &parameter_buffer, query_rope_parameter_offset),
        ],
    )
    .with_barrier_before()
    .with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare);

    let graph = ReplayableComputeGraph::new(
        ctx,
        vec![
            input_norm,
            q_a,
            kv_a,
            q_norm,
            kv_norm,
            key_rope,
            q_b,
            copy_prefix,
            query_rope,
        ],
    )?;
    Ok(CachedAttentionPreludeReplayGraph {
        key: inputs.key(),
        graph: Arc::new(graph),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeviceDsaPreScoreReplayKey {
    layer: usize,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    q_lora_rank: usize,
    hidden: usize,
    norm_eps_bits: u32,
    projection_geometry: [DeviceReplayProjectionGeometry; 3],
    buffer_addresses: Vec<u64>,
}

struct CachedDeviceDsaPreScoreReplayGraph {
    key: DeviceDsaPreScoreReplayKey,
    graph: ReplayableComputeGraph,
    parameter_buffer: Buffer,
    positioned_rope_parameter_offset: usize,
}

impl CachedDeviceDsaPreScoreReplayGraph {
    fn update_position(&self, position: usize) -> Result<()> {
        let output_element_offset = position.checked_mul(self.key.head_dim).ok_or_else(|| {
            Error::Gravity(format!(
                "device DSA replay index-key offset overflow: position={position} dim={}",
                self.key.head_dim
            ))
        })?;
        let params = route_segment_primitives::GlmPositionedRopeParams {
            n_heads: 1,
            rotary_dim: replay_u32(self.key.rope_dim, "device DSA replay RoPE dimension")?,
            in_stride: replay_u32(self.key.head_dim, "device DSA replay input stride")?,
            out_stride: replay_u32(self.key.head_dim, "device DSA replay output stride")?,
            input_element_offset: 0,
            output_element_offset: replay_u32(
                output_element_offset,
                "device DSA replay index-key offset",
            )?,
        };
        write_replay_parameter(
            &self.parameter_buffer,
            self.positioned_rope_parameter_offset,
            &params,
            "device DSA positioned key RoPE",
        )?;
        crate::cost_ledger::record_transfer(
            std::mem::size_of_val(&params) as u64,
            true,
            "device_dsa_pre_score_icb_parameter_update",
        );
        Ok(())
    }
}

struct DeviceDsaPreScoreReplayInputs<'a> {
    layer: usize,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    norm_eps: f32,
    projections: &'a [DeviceReplayProjection; 3],
    q_resid: &'a Buffer,
    h: &'a Buffer,
    idx_q: &'a Buffer,
    idx_k_raw: &'a Buffer,
    idx_head_w: &'a Buffer,
    norm_weight: &'a Buffer,
    norm_bias: &'a Buffer,
    cos: &'a Buffer,
    sin: &'a Buffer,
    query: &'a Buffer,
    index_keys: &'a Buffer,
}

impl DeviceDsaPreScoreReplayInputs<'_> {
    fn key(&self) -> DeviceDsaPreScoreReplayKey {
        let mut buffer_addresses = Vec::with_capacity(18);
        for projection in self.projections {
            projection.append_addresses(&mut buffer_addresses);
        }
        buffer_addresses.extend([
            self.q_resid.gpu_address(),
            self.h.gpu_address(),
            self.idx_q.gpu_address(),
            self.idx_k_raw.gpu_address(),
            self.idx_head_w.gpu_address(),
            self.norm_weight.gpu_address(),
            self.norm_bias.gpu_address(),
            self.cos.gpu_address(),
            self.sin.gpu_address(),
            self.query.gpu_address(),
            self.index_keys.gpu_address(),
        ]);
        DeviceDsaPreScoreReplayKey {
            layer: self.layer,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
            rope_dim: self.rope_dim,
            q_lora_rank: self.projections[0].rows_cols().1 as usize,
            hidden: self.projections[1].rows_cols().1 as usize,
            norm_eps_bits: self.norm_eps.to_bits(),
            projection_geometry: [
                self.projections[0].geometry(),
                self.projections[1].geometry(),
                self.projections[2].geometry(),
            ],
            buffer_addresses,
        }
    }
}

fn build_device_dsa_pre_score_replay_graph(
    ctx: &MetalContext,
    inputs: &DeviceDsaPreScoreReplayInputs<'_>,
    position: usize,
) -> Result<CachedDeviceDsaPreScoreReplayGraph> {
    const TG: u32 = 256;
    if inputs.n_heads == 0
        || inputs.head_dim == 0
        || inputs.rope_dim == 0
        || inputs.rope_dim % 2 != 0
        || inputs.rope_dim > inputs.head_dim
    {
        return Err(Error::Gravity(format!(
            "device DSA pre-score replay has invalid geometry: heads={} head_dim={} rope_dim={}",
            inputs.n_heads, inputs.head_dim, inputs.rope_dim
        )));
    }
    let (wq_rows, q_lora_rank) = inputs.projections[0].rows_cols();
    let (wk_rows, hidden) = inputs.projections[1].rows_cols();
    let (head_rows, head_cols) = inputs.projections[2].rows_cols();
    let expected_q_rows = replay_u32(
        inputs
            .n_heads
            .checked_mul(inputs.head_dim)
            .ok_or_else(|| Error::Gravity("device DSA replay query rows overflow".into()))?,
        "device DSA replay query rows",
    )?;
    if wq_rows != expected_q_rows
        || wk_rows != inputs.head_dim as u32
        || head_rows != inputs.n_heads as u32
        || head_cols != hidden
        || q_lora_rank == 0
        || hidden == 0
    {
        return Err(Error::Gravity(format!(
            "device DSA pre-score projection mismatch: wq={wq_rows}x{q_lora_rank}, wk={wk_rows}x{hidden}, head={head_rows}x{head_cols}, expected wq rows={expected_q_rows}, wk rows={}, head={}x{hidden}",
            inputs.head_dim, inputs.n_heads
        )));
    }
    let output_element_offset = position.checked_mul(inputs.head_dim).ok_or_else(|| {
        Error::Gravity(format!(
            "device DSA replay index-key offset overflow: position={position} dim={}",
            inputs.head_dim
        ))
    })?;
    let required_index_elements = output_element_offset
        .checked_add(inputs.head_dim)
        .ok_or_else(|| Error::Gravity("device DSA replay index-key extent overflow".into()))?;
    if (required_index_elements as u64).saturating_mul(4) > inputs.index_keys.length() {
        return Err(Error::Gravity(format!(
            "device DSA replay index-key extent {required_index_elements} exceeds {} f32 elements",
            inputs.index_keys.length() / 4
        )));
    }

    let mut parameters = ReplayParameterArena::default();
    let projection_offsets = [
        append_replay_projection_parameters(&mut parameters, &inputs.projections[0]),
        append_replay_projection_parameters(&mut parameters, &inputs.projections[1]),
        append_replay_projection_parameters(&mut parameters, &inputs.projections[2]),
    ];
    let head_dim = replay_u32(inputs.head_dim, "device DSA replay head dimension")?;
    let norm_n_offset = parameters.push(&head_dim);
    let norm_eps_offset = parameters.push(&inputs.norm_eps);
    let positioned_rope = route_segment_primitives::GlmPositionedRopeParams {
        n_heads: 1,
        rotary_dim: replay_u32(inputs.rope_dim, "device DSA replay RoPE dimension")?,
        in_stride: head_dim,
        out_stride: head_dim,
        input_element_offset: 0,
        output_element_offset: replay_u32(
            output_element_offset,
            "device DSA replay index-key offset",
        )?,
    };
    let positioned_rope_parameter_offset = parameters.push(&positioned_rope);
    let query_rope = route_segment_primitives::GlmRopeParams {
        n_heads: replay_u32(inputs.n_heads, "device DSA replay head count")?,
        rotary_dim: positioned_rope.rotary_dim,
        in_stride: head_dim,
        out_stride: head_dim,
    };
    let query_rope_offset = parameters.push(&query_rope);
    let parameter_buffer = parameters.finish(ctx, "device DSA pre-score replay graph")?;

    let mut stages = vec![
        build_replay_projection_stage(
            &inputs.projections[0],
            inputs.q_resid,
            inputs.idx_q,
            &parameter_buffer,
            projection_offsets[0],
            "device DSA replay wq_b",
        )?,
        build_replay_projection_stage(
            &inputs.projections[1],
            inputs.h,
            inputs.idx_k_raw,
            &parameter_buffer,
            projection_offsets[1],
            "device DSA replay wk",
        )?,
        build_replay_projection_stage(
            &inputs.projections[2],
            inputs.h,
            inputs.idx_head_w,
            &parameter_buffer,
            projection_offsets[2],
            "device DSA replay head weights",
        )?,
    ];
    let norm = ReplayComputeStage::new(
        "gravity_layernorm_affine_f32",
        (TG, 1, 1),
        (TG, 1, 1),
        vec![
            ReplayBufferBinding::read(0, inputs.idx_k_raw, 0),
            ReplayBufferBinding::read(1, inputs.norm_weight, 0),
            ReplayBufferBinding::read(2, inputs.norm_bias, 0),
            ReplayBufferBinding::write(3, inputs.idx_k_raw, 0),
            ReplayBufferBinding::read(4, &parameter_buffer, norm_n_offset),
            ReplayBufferBinding::read(5, &parameter_buffer, norm_eps_offset),
        ],
    )
    .with_threadgroup_memory_length(0, TG as usize * 4)
    .with_barrier_before()
    .with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare);
    let key_rope_grid = replay_grid(head_dim, TG, TG, "device DSA replay key RoPE")?;
    let key_rope = ReplayComputeStage::new(
        "gravity_rope_prefix_tail_positioned_f32",
        (key_rope_grid, 1, 1),
        (TG, 1, 1),
        vec![
            ReplayBufferBinding::read(0, inputs.idx_k_raw, 0),
            ReplayBufferBinding::write(1, inputs.index_keys, 0),
            ReplayBufferBinding::read(2, inputs.cos, 0),
            ReplayBufferBinding::read(3, inputs.sin, 0),
            ReplayBufferBinding::read(4, &parameter_buffer, positioned_rope_parameter_offset),
        ],
    )
    .with_barrier_before()
    .with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare);
    let query_rope_elements = replay_u32(
        inputs
            .n_heads
            .checked_mul(inputs.head_dim)
            .ok_or_else(|| Error::Gravity("device DSA replay query RoPE grid overflow".into()))?,
        "device DSA replay query RoPE elements",
    )?;
    let query_rope = ReplayComputeStage::new(
        "gravity_rope_prefix_tail_f32",
        (
            replay_grid(query_rope_elements, TG, TG, "device DSA replay query RoPE")?,
            1,
            1,
        ),
        (TG, 1, 1),
        vec![
            ReplayBufferBinding::read(0, inputs.idx_q, 0),
            ReplayBufferBinding::write(1, inputs.query, 0),
            ReplayBufferBinding::read(2, inputs.cos, 0),
            ReplayBufferBinding::read(3, inputs.sin, 0),
            ReplayBufferBinding::read(4, &parameter_buffer, query_rope_offset),
        ],
    )
    .with_barrier_before()
    .with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare);
    stages.extend([norm, key_rope, query_rope]);
    let graph = ReplayableComputeGraph::new(ctx, stages)?;
    Ok(CachedDeviceDsaPreScoreReplayGraph {
        key: inputs.key(),
        graph,
        parameter_buffer,
        positioned_rope_parameter_offset,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CompactAttentionReplayKey {
    layer: usize,
    hidden: usize,
    n_heads: usize,
    latent_dim: usize,
    rope_dim: usize,
    key_rows: usize,
    row_stride: usize,
    value_rows: usize,
    max_allow: usize,
    scale_bits: u32,
    kv_params: crate::gravity_glm::gpu::PqParams,
    o_params: crate::gravity_glm::gpu::PqParams,
    include_radix: bool,
    include_residual: bool,
    buffer_addresses: [u64; 16],
}

struct CachedCompactAttentionReplayGraph {
    key: CompactAttentionReplayKey,
    graph: ReplayableComputeGraph,
    parameter_buffer: Buffer,
    radix_parameter_offset: Option<usize>,
    append_parameter_offset: usize,
    ranked_parameter_offset: usize,
    score_capacity: usize,
}

impl CachedCompactAttentionReplayGraph {
    fn update_dynamic_parameters(
        &self,
        position: usize,
        n_keys: usize,
        n_allow: usize,
    ) -> Result<()> {
        if n_allow > self.key.max_allow {
            return Err(Error::Gravity(format!(
                "compact-attention replay n_allow {n_allow} exceeds captured bound {}",
                self.key.max_allow
            )));
        }
        if self.radix_parameter_offset.is_some() && n_keys > self.score_capacity {
            return Err(Error::Gravity(format!(
                "compact-attention replay active key count {n_keys} exceeds score capacity {}",
                self.score_capacity
            )));
        }
        let append = route_segment_primitives::GlmMlaCompactAppendParams {
            latent_dim: replay_u32(self.key.latent_dim, "compact replay latent dimension")?,
            rope_dim: replay_u32(self.key.rope_dim, "compact replay RoPE dimension")?,
            pos: replay_u32(position, "compact replay position")?,
        };
        let ranked = route_segment_primitives::GlmCompactRankedAttnParams {
            n_heads: replay_u32(self.key.n_heads, "compact replay head count")?,
            latent_dim: replay_u32(self.key.latent_dim, "compact replay latent dimension")?,
            rope_dim: replay_u32(self.key.rope_dim, "compact replay RoPE dimension")?,
            n_keys: replay_u32(n_keys, "compact replay active key count")?,
            n_allow: replay_u32(n_allow, "compact replay selected key count")?,
            scale: f32::from_bits(self.key.scale_bits),
        };
        write_replay_parameter(
            &self.parameter_buffer,
            self.append_parameter_offset,
            &append,
            "compact-attention append",
        )?;
        write_replay_parameter(
            &self.parameter_buffer,
            self.ranked_parameter_offset,
            &ranked,
            "compact-attention ranked",
        )?;
        let mut transfer_bytes = std::mem::size_of_val(&append) + std::mem::size_of_val(&ranked);
        if let Some(offset) = self.radix_parameter_offset {
            let radix = route_segment_primitives::GlmTopkParams {
                n: replay_u32(n_keys, "compact replay radix input count")?,
                k: replay_u32(n_allow, "compact replay radix selected count")?,
            };
            write_replay_parameter(
                &self.parameter_buffer,
                offset,
                &radix,
                "compact-attention radix",
            )?;
            transfer_bytes = transfer_bytes.saturating_add(std::mem::size_of_val(&radix));
        }
        crate::cost_ledger::record_transfer(
            transfer_bytes as u64,
            true,
            "compact_attention_icb_parameter_update",
        );
        Ok(())
    }
}

struct CompactAttentionReplayInputs<'a> {
    layer: usize,
    hidden: usize,
    n_heads: usize,
    latent_dim: usize,
    rope_dim: usize,
    key_rows: usize,
    row_stride: usize,
    value_rows: usize,
    max_allow: usize,
    scale: f32,
    kv_params: crate::gravity_glm::gpu::PqParams,
    o_params: crate::gravity_glm::gpu::PqParams,
    k_latent: &'a Buffer,
    key_rope: &'a Buffer,
    latent_cache: &'a Buffer,
    rope_cache: &'a Buffer,
    kv_codebooks: &'a Buffer,
    kv_codes: &'a Buffer,
    query_nope: &'a Buffer,
    query_latent: &'a Buffer,
    query_rope: &'a Buffer,
    scores: Option<&'a Buffer>,
    ranked_indices: &'a Buffer,
    context: &'a Buffer,
    o_codebooks: &'a Buffer,
    o_codes: &'a Buffer,
    output: &'a Buffer,
    residual: Option<&'a Buffer>,
}

impl CompactAttentionReplayInputs<'_> {
    fn key(&self) -> CompactAttentionReplayKey {
        CompactAttentionReplayKey {
            layer: self.layer,
            hidden: self.hidden,
            n_heads: self.n_heads,
            latent_dim: self.latent_dim,
            rope_dim: self.rope_dim,
            key_rows: self.key_rows,
            row_stride: self.row_stride,
            value_rows: self.value_rows,
            max_allow: self.max_allow,
            scale_bits: self.scale.to_bits(),
            kv_params: self.kv_params,
            o_params: self.o_params,
            include_radix: self.scores.is_some(),
            include_residual: self.residual.is_some(),
            buffer_addresses: [
                self.k_latent.gpu_address(),
                self.key_rope.gpu_address(),
                self.latent_cache.gpu_address(),
                self.rope_cache.gpu_address(),
                self.kv_codebooks.gpu_address(),
                self.kv_codes.gpu_address(),
                self.query_nope.gpu_address(),
                self.query_latent.gpu_address(),
                self.query_rope.gpu_address(),
                self.scores.map_or(0, |buffer| buffer.gpu_address()),
                self.ranked_indices.gpu_address(),
                self.context.gpu_address(),
                self.o_codebooks.gpu_address(),
                self.o_codes.gpu_address(),
                self.output.gpu_address(),
                self.residual.map_or(0, |buffer| buffer.gpu_address()),
            ],
        }
    }
}

fn build_compact_attention_replay_graph(
    ctx: &MetalContext,
    inputs: &CompactAttentionReplayInputs<'_>,
    position: usize,
    n_keys: usize,
    n_allow: usize,
) -> Result<CachedCompactAttentionReplayGraph> {
    const TG: u32 = 256;
    if inputs.max_allow == 0 || inputs.max_allow > 2048 || n_allow > inputs.max_allow {
        return Err(Error::Gravity(format!(
            "compact-attention replay requires 1 <= max_allow <= 2048 and n_allow <= max_allow, got max_allow={} n_allow={n_allow}",
            inputs.max_allow
        )));
    }
    let expected_row_stride = inputs
        .key_rows
        .checked_add(inputs.value_rows)
        .ok_or_else(|| Error::Gravity("compact replay row stride overflow".into()))?;
    if inputs.row_stride != expected_row_stride {
        return Err(Error::Gravity(format!(
            "compact-attention replay row_stride {} != key_rows {} + value_rows {}",
            inputs.row_stride, inputs.key_rows, inputs.value_rows
        )));
    }
    if inputs.hidden == 0 || inputs.o_params.rows as usize != inputs.hidden {
        return Err(Error::Gravity(format!(
            "compact-attention replay hidden {} != o_proj rows {}",
            inputs.hidden, inputs.o_params.rows
        )));
    }
    let score_capacity = inputs.scores.map_or(0, |scores| {
        (scores.length() / std::mem::size_of::<f32>() as u64) as usize
    });
    if inputs.scores.is_some() && n_keys > score_capacity {
        return Err(Error::Gravity(format!(
            "compact-attention replay active key count {n_keys} exceeds score capacity {score_capacity}"
        )));
    }
    if let Some(residual) = inputs.residual {
        let required = inputs
            .hidden
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                Error::Gravity("compact-attention replay residual extent overflow".into())
            })?;
        if required as u64 > residual.length() {
            return Err(Error::Gravity(format!(
                "compact-attention replay residual needs {required} bytes, buffer has {}",
                residual.length()
            )));
        }
    }

    let append = route_segment_primitives::GlmMlaCompactAppendParams {
        latent_dim: replay_u32(inputs.latent_dim, "compact replay latent dimension")?,
        rope_dim: replay_u32(inputs.rope_dim, "compact replay RoPE dimension")?,
        pos: replay_u32(position, "compact replay position")?,
    };
    let k_transpose = route_segment_primitives::GlmPqKTransposeParams {
        n_heads: replay_u32(inputs.n_heads, "compact replay head count")?,
        key_rows: replay_u32(inputs.key_rows, "compact replay key rows")?,
        row_stride: replay_u32(inputs.row_stride, "compact replay row stride")?,
        latent_dim: replay_u32(inputs.latent_dim, "compact replay latent dimension")?,
        pq_dim: inputs.kv_params.dim,
        pq_sub: inputs.kv_params.sub,
        pq_nchunk: inputs.kv_params.nchunk,
    };
    let ranked = route_segment_primitives::GlmCompactRankedAttnParams {
        n_heads: replay_u32(inputs.n_heads, "compact replay head count")?,
        latent_dim: replay_u32(inputs.latent_dim, "compact replay latent dimension")?,
        rope_dim: replay_u32(inputs.rope_dim, "compact replay RoPE dimension")?,
        n_keys: replay_u32(n_keys, "compact replay active key count")?,
        n_allow: replay_u32(n_allow, "compact replay selected key count")?,
        scale: inputs.scale,
    };
    let v_rows = route_segment_primitives::GlmPqVRowsParams {
        n_heads: replay_u32(inputs.n_heads, "compact replay head count")?,
        row_stride: replay_u32(inputs.row_stride, "compact replay row stride")?,
        value_row_offset: replay_u32(inputs.key_rows, "compact replay value row offset")?,
        value_rows: replay_u32(inputs.value_rows, "compact replay value rows")?,
        latent_dim: replay_u32(inputs.latent_dim, "compact replay latent dimension")?,
        pq_dim: inputs.kv_params.dim,
        pq_sub: inputs.kv_params.sub,
        pq_nchunk: inputs.kv_params.nchunk,
    };

    let mut parameters = ReplayParameterArena::default();
    let radix_parameter_offset = inputs.scores.map(|_| {
        parameters.push(&route_segment_primitives::GlmTopkParams {
            n: ranked.n_keys,
            k: ranked.n_allow,
        })
    });
    let append_parameter_offset = parameters.push(&append);
    let k_parameter_offset = parameters.push(&k_transpose);
    let ranked_parameter_offset = parameters.push(&ranked);
    let v_parameter_offset = parameters.push(&v_rows);
    let o_parameter_offset = parameters.push(&inputs.o_params);
    let residual_parameter_offset = inputs
        .residual
        .map(|_| parameters.push(&inputs.o_params.rows));
    let parameter_buffer = parameters.finish(ctx, "compact-attention replay graph")?;

    let append_grid = replay_grid(
        replay_u32(
            inputs
                .latent_dim
                .checked_add(inputs.rope_dim)
                .ok_or_else(|| Error::Gravity("compact replay append grid overflow".into()))?,
            "compact replay append elements",
        )?,
        TG,
        TG,
        "compact replay append",
    )?;
    let k_outputs = inputs
        .n_heads
        .checked_mul(inputs.latent_dim)
        .ok_or_else(|| Error::Gravity("compact replay K output count overflow".into()))?;
    let k_grid = replay_grid(
        replay_u32(k_outputs, "compact replay K outputs")?,
        TG,
        TG,
        "compact replay K transpose",
    )?;
    let ranked_grid = replay_u32(inputs.n_heads, "compact replay head count")?
        .checked_mul(TG)
        .ok_or_else(|| Error::Gravity("compact replay ranked grid overflow".into()))?;
    let v_outputs = inputs
        .n_heads
        .checked_mul(inputs.value_rows)
        .ok_or_else(|| Error::Gravity("compact replay V output count overflow".into()))?;
    let v_grid = replay_grid(
        replay_u32(v_outputs, "compact replay V outputs")?,
        8,
        TG,
        "compact replay V rows",
    )?;
    let o_grid = replay_grid(inputs.o_params.rows, 8, TG, "compact replay o_proj")?;
    let ranked_threadgroup_bytes = inputs
        .max_allow
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Gravity("compact replay ranked memory overflow".into()))?;

    let mut stages = Vec::with_capacity(
        5usize
            .saturating_add(usize::from(inputs.scores.is_some()))
            .saturating_add(usize::from(inputs.residual.is_some())),
    );
    if let (Some(scores), Some(radix_offset)) = (inputs.scores, radix_parameter_offset) {
        stages.push(
            ReplayComputeStage::new(
                "gravity_glm_radix_topk_f32",
                (TG, 1, 1),
                (TG, 1, 1),
                vec![
                    ReplayBufferBinding::read(0, scores, 0),
                    ReplayBufferBinding::write(1, inputs.ranked_indices, 0),
                    ReplayBufferBinding::read(2, &parameter_buffer, radix_offset),
                ],
            )
            .with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare),
        );
    }
    stages.extend([
        ReplayComputeStage::new(
            "gravity_glm_mla_append_compact",
            (append_grid, 1, 1),
            (TG, 1, 1),
            vec![
                ReplayBufferBinding::read(0, inputs.k_latent, 0),
                ReplayBufferBinding::read(1, inputs.key_rope, 0),
                ReplayBufferBinding::write(2, inputs.latent_cache, 0),
                ReplayBufferBinding::write(3, inputs.rope_cache, 0),
                ReplayBufferBinding::read(4, &parameter_buffer, append_parameter_offset),
            ],
        )
        .with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare),
        ReplayComputeStage::new(
            "gravity_pq_k_transpose_heads",
            (k_grid, 1, 1),
            (TG, 1, 1),
            vec![
                ReplayBufferBinding::read(0, inputs.kv_codebooks, 0),
                ReplayBufferBinding::read(1, inputs.kv_codes, 0),
                ReplayBufferBinding::read(2, inputs.query_nope, 0),
                ReplayBufferBinding::write(3, inputs.query_latent, 0),
                ReplayBufferBinding::read(4, &parameter_buffer, k_parameter_offset),
            ],
        )
        .with_barrier_before()
        .with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare),
        ReplayComputeStage::new(
            "gravity_glm_compact_ranked_attn",
            (ranked_grid, 1, 1),
            (TG, 1, 1),
            vec![
                ReplayBufferBinding::read(0, inputs.query_latent, 0),
                ReplayBufferBinding::read(1, inputs.query_rope, 0),
                ReplayBufferBinding::read(2, inputs.latent_cache, 0),
                ReplayBufferBinding::read(3, inputs.rope_cache, 0),
                ReplayBufferBinding::read(4, inputs.ranked_indices, 0),
                ReplayBufferBinding::write(5, inputs.query_latent, 0),
                ReplayBufferBinding::read(6, &parameter_buffer, ranked_parameter_offset),
            ],
        )
        .with_threadgroup_memory_length(0, ranked_threadgroup_bytes)
        .with_barrier_before()
        .with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare),
        ReplayComputeStage::new(
            "gravity_pq_v_rows_heads",
            (v_grid, 1, 1),
            (TG, 1, 1),
            vec![
                ReplayBufferBinding::read(0, inputs.kv_codebooks, 0),
                ReplayBufferBinding::read(1, inputs.kv_codes, 0),
                ReplayBufferBinding::read(2, inputs.query_latent, 0),
                ReplayBufferBinding::write(3, inputs.context, 0),
                ReplayBufferBinding::read(4, &parameter_buffer, v_parameter_offset),
            ],
        )
        .with_barrier_before()
        .with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare),
        ReplayComputeStage::new(
            "gravity_pq_matvec",
            (o_grid, 1, 1),
            (TG, 1, 1),
            vec![
                ReplayBufferBinding::read(0, inputs.o_codebooks, 0),
                ReplayBufferBinding::read(1, inputs.o_codes, 0),
                ReplayBufferBinding::read(2, inputs.context, 0),
                ReplayBufferBinding::write(3, inputs.output, 0),
                ReplayBufferBinding::read(4, &parameter_buffer, o_parameter_offset),
            ],
        )
        .with_barrier_before()
        .with_ledger_stage(crate::cost_ledger::GpuStage::AttentionAndIndexShare),
    ]);
    if let (Some(residual), Some(residual_offset)) = (inputs.residual, residual_parameter_offset) {
        stages.push(
            ReplayComputeStage::new(
                "gravity_add_inplace_f32",
                (
                    replay_grid(
                        replay_u32(inputs.hidden, "compact replay residual elements")?,
                        TG,
                        TG,
                        "compact replay residual",
                    )?,
                    1,
                    1,
                ),
                (TG, 1, 1),
                vec![
                    ReplayBufferBinding::read_write(0, residual, 0),
                    ReplayBufferBinding::read(1, inputs.output, 0),
                    ReplayBufferBinding::read(2, &parameter_buffer, residual_offset),
                ],
            )
            .with_barrier_before()
            .with_ledger_stage(crate::cost_ledger::GpuStage::Other),
        );
    }
    let graph = ReplayableComputeGraph::new(ctx, stages)?;
    Ok(CachedCompactAttentionReplayGraph {
        key: inputs.key(),
        graph,
        parameter_buffer,
        radix_parameter_offset,
        append_parameter_offset,
        ranked_parameter_offset,
        score_capacity,
    })
}

fn final_head_replay_key(
    head: &DeviceHead,
    pool: &ActPool,
    hidden: usize,
    rms_norm_eps: f32,
) -> FinalHeadReplayKey {
    let (geometry, primary, secondary) = match head {
        DeviceHead::NativeBf16 { weight, rows, cols } => (
            FinalHeadReplayGeometry::NativeBf16 {
                rows: *rows,
                cols: *cols,
            },
            weight.gpu_address(),
            0,
        ),
        DeviceHead::Pq {
            codebooks,
            codes,
            params,
        } => (
            FinalHeadReplayGeometry::Pq(*params),
            codebooks.gpu_address(),
            codes.gpu_address(),
        ),
    };
    FinalHeadReplayKey {
        geometry,
        hidden,
        rms_norm_eps_bits: rms_norm_eps.to_bits(),
        buffer_addresses: [
            pool.x.gpu_address(),
            pool.final_norm_weight.gpu_address(),
            pool.final_hidden.gpu_address(),
            pool.logits.gpu_address(),
            pool.sample_token.gpu_address(),
            pool.head_topk_idx.gpu_address(),
            pool.head_topk_val.gpu_address(),
            primary,
            secondary,
        ],
    }
}

fn build_final_head_replay_graph(
    ctx: &MetalContext,
    head: &DeviceHead,
    pool: &ActPool,
    hidden: usize,
    rms_norm_eps: f32,
) -> Result<ReplayableComputeGraph> {
    const TG: u32 = 256;
    let hidden_u32 = replay_u32(hidden, "final-head hidden size")?;
    let rows = head.rows();
    if rows == 0 {
        return Err(Error::Gravity(
            "final-head replay graph has zero vocabulary rows".into(),
        ));
    }

    let mut parameters = ReplayParameterArena::default();
    let hidden_offset = parameters.push(&hidden_u32);
    let eps_offset = parameters.push(&rms_norm_eps);
    let sample_n_offset = parameters.push(&rows);
    let sample_k = GPU_LM_HEAD_DIAG_TOPK.min(64);
    let sample_k_offset = parameters.push(&sample_k);
    let head_offsets = match head {
        DeviceHead::NativeBf16 { rows, cols, .. } => {
            let rows_offset = parameters.push(rows);
            let cols_offset = parameters.push(cols);
            (rows_offset, Some(cols_offset))
        }
        DeviceHead::Pq { params, .. } => (parameters.push(params), None),
    };
    let parameter_buffer = parameters.finish(ctx, "final-head replay graph")?;

    let norm = ReplayComputeStage::new(
        "gravity_rmsnorm_f32",
        (TG, 1, 1),
        (TG, 1, 1),
        vec![
            ReplayBufferBinding::read(0, &pool.x, 0),
            ReplayBufferBinding::read(1, &pool.final_norm_weight, 0),
            ReplayBufferBinding::write(2, &pool.final_hidden, 0),
            ReplayBufferBinding::read(3, &parameter_buffer, hidden_offset),
            ReplayBufferBinding::read(4, &parameter_buffer, eps_offset),
        ],
    )
    .with_threadgroup_memory_length(0, TG as usize * 4)
    .with_ledger_stage(crate::cost_ledger::GpuStage::KvAndNorm);

    let head_stage = match head {
        DeviceHead::NativeBf16 {
            weight,
            rows,
            cols: _,
        } => {
            let grid = replay_grid(*rows, TG, TG, "native final-head replay")?;
            ReplayComputeStage::new(
                "gemv_native_bf16_seq",
                (grid, 1, 1),
                (TG, 1, 1),
                vec![
                    ReplayBufferBinding::read(0, weight, 0),
                    ReplayBufferBinding::read(1, &pool.final_hidden, 0),
                    ReplayBufferBinding::write(2, &pool.logits, 0),
                    ReplayBufferBinding::read(3, &parameter_buffer, head_offsets.0),
                    ReplayBufferBinding::read(
                        4,
                        &parameter_buffer,
                        head_offsets
                            .1
                            .expect("native final-head replay has cols parameter"),
                    ),
                ],
            )
        }
        DeviceHead::Pq {
            codebooks,
            codes,
            params,
        } => {
            let grid = replay_grid(params.rows, 8, TG, "PQ final-head replay")?;
            ReplayComputeStage::new(
                "gravity_pq_matvec",
                (grid, 1, 1),
                (TG, 1, 1),
                vec![
                    ReplayBufferBinding::read(0, codebooks, 0),
                    ReplayBufferBinding::read(1, codes, 0),
                    ReplayBufferBinding::read(2, &pool.final_hidden, 0),
                    ReplayBufferBinding::write(3, &pool.logits, 0),
                    ReplayBufferBinding::read(4, &parameter_buffer, head_offsets.0),
                ],
            )
        }
    }
    .with_barrier_before()
    .with_ledger_stage(crate::cost_ledger::GpuStage::FinalHead);

    let argmax = ReplayComputeStage::new(
        "sample_argmax_f32",
        (TG, 1, 1),
        (TG, 1, 1),
        vec![
            ReplayBufferBinding::read(0, &pool.logits, 0),
            ReplayBufferBinding::write(1, &pool.sample_token, 0),
            ReplayBufferBinding::read(2, &parameter_buffer, sample_n_offset),
        ],
    )
    .with_threadgroup_memory_length(0, TG as usize * 4)
    .with_threadgroup_memory_length(1, TG as usize * 4)
    .with_barrier_before()
    .with_ledger_stage(crate::cost_ledger::GpuStage::Sampling);

    let topk = ReplayComputeStage::new(
        "sample_topk",
        (TG, 1, 1),
        (TG, 1, 1),
        vec![
            ReplayBufferBinding::read(0, &pool.logits, 0),
            ReplayBufferBinding::write(1, &pool.head_topk_idx, 0),
            ReplayBufferBinding::write(2, &pool.head_topk_val, 0),
            ReplayBufferBinding::read(3, &parameter_buffer, sample_n_offset),
            ReplayBufferBinding::read(4, &parameter_buffer, sample_k_offset),
        ],
    )
    .with_threadgroup_memory_length(0, TG as usize * 4)
    .with_threadgroup_memory_length(1, TG as usize * 4)
    .with_threadgroup_memory_length(2, 64 * 4)
    .with_barrier_before()
    .with_ledger_stage(crate::cost_ledger::GpuStage::Sampling);

    ReplayableComputeGraph::new(ctx, vec![norm, head_stage, argmax, topk])
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeviceExpertReplayKey {
    generation: u32,
    experts_per_token: usize,
    hidden: usize,
    intermediate: usize,
    routed_dispatch_mode: DeviceExpertDispatchMode,
    shared_dispatch_mode: DeviceExpertDispatchMode,
    buffer_addresses: Vec<u64>,
}

struct CachedDeviceExpertReplayGraph {
    key: DeviceExpertReplayKey,
    graph: ReplayableComputeGraph,
}

#[derive(Clone, Copy)]
enum DeviceExpertProjectionMetrics {
    Pq {
        params: crate::gravity_glm::gpu::PqParams,
        bytes: u64,
        representation: crate::cost_ledger::RoutedWeightRepresentation,
    },
    NativeBf16 {
        rows: u32,
        cols: u32,
        bytes: u64,
    },
}

impl DeviceExpertProjectionMetrics {
    fn rows_cols(self) -> (usize, usize) {
        match self {
            Self::Pq { params, .. } => (params.rows as usize, params.cols as usize),
            Self::NativeBf16 { rows, cols, .. } => (rows as usize, cols as usize),
        }
    }

    fn bytes(self) -> u64 {
        match self {
            Self::Pq { bytes, .. } | Self::NativeBf16 { bytes, .. } => bytes,
        }
    }

    fn is_pq(self) -> bool {
        matches!(self, Self::Pq { .. })
    }
}

#[derive(Clone)]
struct DeviceExpertLayerMetrics {
    routed: Vec<[DeviceExpertProjectionMetrics; 3]>,
    shared: [DeviceExpertProjectionMetrics; 3],
}

impl DeviceExpertLayerMetrics {
    #[cfg(test)]
    fn dispatch_mode(&self) -> DeviceExpertDispatchMode {
        Self::dispatch_mode_for(
            self.routed
                .iter()
                .flatten()
                .chain(self.shared.iter())
                .copied(),
        )
        .expect("shared expert triplet is nonempty")
    }

    fn routed_dispatch_mode(&self) -> DeviceExpertDispatchMode {
        Self::dispatch_mode_for(self.routed.iter().flatten().copied())
            // The initial empty table cannot hit. Its provisional mode only
            // determines which guarded no-op kernel follows validation.
            .unwrap_or(DeviceExpertDispatchMode::PqOnly)
    }

    fn shared_dispatch_mode(&self) -> DeviceExpertDispatchMode {
        Self::dispatch_mode_for(self.shared.iter().copied())
            .expect("shared expert triplet is nonempty")
    }

    fn dispatch_mode_for(
        metrics: impl Iterator<Item = DeviceExpertProjectionMetrics>,
    ) -> Option<DeviceExpertDispatchMode> {
        let mut saw_pq = false;
        let mut saw_native = false;
        for metric in metrics {
            if metric.is_pq() {
                saw_pq = true;
            } else {
                saw_native = true;
            }
        }
        Some(match (saw_pq, saw_native) {
            (true, false) => DeviceExpertDispatchMode::PqOnly,
            (false, true) => DeviceExpertDispatchMode::NativeBf16Only,
            (true, true) => DeviceExpertDispatchMode::Heterogeneous,
            (false, false) => return None,
        })
    }
}

fn device_expert_projection_metrics(tensor: &GpuTensor) -> Option<DeviceExpertProjectionMetrics> {
    use crate::cost_ledger::RoutedWeightRepresentation;

    match tensor {
        GpuTensor::Pq {
            codebooks,
            codes,
            params,
        } => {
            let representation = routed_pq_representation(params);
            if !matches!(
                representation,
                RoutedWeightRepresentation::R4 | RoutedWeightRepresentation::R0
            ) || params.rows == 0
                || params.cols == 0
                || params.bits == 0
                || params.bits > 8
                || params.subspaces == 0
                || params.sub == 0
                || params.dim != params.subspaces.checked_mul(params.sub)?
                || params.card != 1u32.checked_shl(params.bits)?
                || params.nchunk == 0
                || params.cols != params.nchunk.checked_mul(params.dim)?
            {
                return None;
            }
            let codebook_bytes = u64::from(params.subspaces)
                .checked_mul(u64::from(params.card))?
                .checked_mul(u64::from(params.sub))?
                .checked_mul(2)?;
            let index_count = u64::from(params.rows)
                .checked_mul(u64::from(params.nchunk))?
                .checked_mul(u64::from(params.subspaces))?;
            let packed_bytes = index_count
                .checked_mul(u64::from(params.bits))?
                .div_ceil(8)
                .checked_add(4)?;
            if codebooks.length() < codebook_bytes || codes.length() < packed_bytes {
                return None;
            }
            Some(DeviceExpertProjectionMetrics::Pq {
                params: *params,
                bytes: codebooks.length().saturating_add(codes.length()),
                representation,
            })
        }
        GpuTensor::NativeGpuBf16 { buf, rows, cols } => {
            if *rows == 0 || *cols == 0 {
                return None;
            }
            let required = u64::from(*rows)
                .checked_mul(u64::from(*cols))?
                .checked_mul(2)?;
            if buf.length() < required {
                return None;
            }
            Some(DeviceExpertProjectionMetrics::NativeBf16 {
                rows: *rows,
                cols: *cols,
                bytes: buf.length(),
            })
        }
        GpuTensor::NativeCpu(_) | GpuTensor::ActivationAware { .. } => None,
    }
}

fn device_expert_triplet_metrics(
    gate: &GpuTensor,
    up: &GpuTensor,
    down: &GpuTensor,
    hidden: usize,
) -> Option<(usize, [DeviceExpertProjectionMetrics; 3])> {
    let metrics = [
        device_expert_projection_metrics(gate)?,
        device_expert_projection_metrics(up)?,
        device_expert_projection_metrics(down)?,
    ];
    let (gate_rows, gate_cols) = metrics[0].rows_cols();
    let (up_rows, up_cols) = metrics[1].rows_cols();
    let (down_rows, down_cols) = metrics[2].rows_cols();
    if gate_rows == 0
        || gate_cols != hidden
        || up_rows != gate_rows
        || up_cols != hidden
        || down_rows != hidden
        || down_cols != gate_rows
    {
        return None;
    }
    Some((gate_rows, metrics))
}

fn record_device_expert_projection_cost(
    name: &str,
    metric: DeviceExpertProjectionMetrics,
    routed: bool,
) {
    use crate::cost_ledger::{self, RoutedWeightRepresentation};

    cost_ledger::record_matvec_call();
    cost_ledger::record_active_bytes_for(name, metric.bytes());
    match metric {
        DeviceExpertProjectionMetrics::Pq {
            params,
            bytes,
            representation,
        } => {
            if routed {
                cost_ledger::record_routed_weight_representation(name, representation, bytes);
            }
            record_pq_matvec_ops(params);
        }
        DeviceExpertProjectionMetrics::NativeBf16 { rows, cols, bytes } => {
            if routed {
                cost_ledger::record_routed_weight_representation(
                    name,
                    RoutedWeightRepresentation::NativeBf16,
                    bytes,
                );
            }
            record_dense_matvec_ops(rows as u64, cols as u64);
        }
    }
}

fn record_device_expert_table_hit_costs(
    mlp_prefix: &str,
    hidden: usize,
    intermediate: usize,
    metrics: &DeviceExpertLayerMetrics,
) {
    use crate::cost_ledger;

    let projection_names = ["gate_proj", "up_proj", "down_proj"];
    for (execution_position, triplet) in metrics.routed.iter().enumerate() {
        for projection in 0..3 {
            let name = format!(
                "{mlp_prefix}.experts.device_slot_{execution_position}.{}.weight",
                projection_names[projection]
            );
            record_device_expert_projection_cost(&name, triplet[projection], true);
        }
    }
    for projection in 0..3 {
        let name = format!(
            "{mlp_prefix}.shared_experts.{}.weight",
            projection_names[projection]
        );
        record_device_expert_projection_cost(&name, metrics.shared[projection], false);
    }

    let expert_count = metrics.routed.len().saturating_add(1) as u64;
    cost_ledger::record_source_modelled_operations(
        expert_count
            .saturating_mul((4usize.saturating_mul(intermediate)) as u64)
            .saturating_add(expert_count.saturating_mul((2usize.saturating_mul(hidden)) as u64))
            .saturating_add(hidden as u64),
        0,
        0,
        expert_count.saturating_mul(intermediate as u64),
        0,
    );
}

#[allow(clippy::too_many_arguments)]
fn build_persistent_device_expert_layer(
    weights: &GpuWeightCache,
    mlp_prefix: &str,
    hidden: usize,
    n_routed_experts: usize,
    generation: u32,
    selected_experts: &[usize],
    ctx: &MetalContext,
) -> Result<Option<PersistentDeviceExpertLayer>> {
    let shared_prefix = format!("{mlp_prefix}.shared_experts");
    let shared_gate_name = format!("{shared_prefix}.gate_proj.weight");
    let shared_up_name = format!("{shared_prefix}.up_proj.weight");
    let shared_down_name = format!("{shared_prefix}.down_proj.weight");
    let mut names = vec![
        shared_gate_name.clone(),
        shared_up_name.clone(),
        shared_down_name.clone(),
    ];
    for &expert in selected_experts {
        if expert >= n_routed_experts {
            return Err(Error::Gravity(format!(
                "persistent device expert route selected {expert}, but layer has \
                 {n_routed_experts} experts"
            )));
        }
        let prefix = format!("{mlp_prefix}.experts.{expert}");
        names.push(format!("{prefix}.gate_proj.weight"));
        names.push(format!("{prefix}.up_proj.weight"));
        names.push(format!("{prefix}.down_proj.weight"));
    }
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();

    let (routed, shared, intermediate, metrics, routed_dispatch_mode, shared_dispatch_mode) = {
        let mut cache = weights.cache.lock().expect("gpu weight cache");
        weights.ensure_many_locked(&mut cache, &name_refs)?;
        let shared_gate = cache.get(&shared_gate_name).expect("ensured shared gate");
        let shared_up = cache.get(&shared_up_name).expect("ensured shared up");
        let shared_down = cache.get(&shared_down_name).expect("ensured shared down");
        let Some((intermediate, shared_metrics)) =
            device_expert_triplet_metrics(shared_gate, shared_up, shared_down, hidden)
        else {
            return Ok(None);
        };

        // The selected IDs are host-known only on the guarded miss path. Bind
        // their exact representation/extent metadata to the immutable lease;
        // a later hit can only reference these ready entries, so its ledger is
        // exact without another ID or metrics readback.
        let mut routed_metrics = Vec::with_capacity(selected_experts.len());
        for &expert in selected_experts {
            let prefix = format!("{mlp_prefix}.experts.{expert}");
            let gate = cache
                .get(&format!("{prefix}.gate_proj.weight"))
                .expect("ensured routed gate");
            let up = cache
                .get(&format!("{prefix}.up_proj.weight"))
                .expect("ensured routed up");
            let down = cache
                .get(&format!("{prefix}.down_proj.weight"))
                .expect("ensured routed down");
            let Some((routed_intermediate, triplet_metrics)) =
                device_expert_triplet_metrics(gate, up, down, hidden)
            else {
                return Ok(None);
            };
            if routed_intermediate != intermediate {
                return Ok(None);
            }
            routed_metrics.push(triplet_metrics);
        }

        let shared = build_single_device_expert_snapshot(
            ctx,
            shared_gate,
            shared_up,
            shared_down,
            generation,
        )?;
        let routed = build_selected_device_expert_table_snapshot(
            ctx,
            &cache,
            mlp_prefix,
            n_routed_experts,
            generation,
            selected_experts,
        )?;
        let metrics = DeviceExpertLayerMetrics {
            routed: routed_metrics,
            shared: shared_metrics,
        };
        let routed_dispatch_mode = metrics.routed_dispatch_mode();
        let shared_dispatch_mode = metrics.shared_dispatch_mode();
        (
            routed,
            shared,
            intermediate,
            metrics,
            routed_dispatch_mode,
            shared_dispatch_mode,
        )
    };

    let snapshot_bytes = routed.table.length().saturating_add(shared.table.length());
    crate::cost_ledger::record_transfer(
        snapshot_bytes,
        true,
        "device_expert_table_snapshot_upload",
    );
    crate::cost_ledger::record_allocation(snapshot_bytes);
    Ok(Some(PersistentDeviceExpertLayer {
        routed,
        shared,
        intermediate,
        metrics,
        routed_dispatch_mode,
        shared_dispatch_mode,
        replay_graph: Arc::new(Mutex::new(None)),
    }))
}

#[allow(clippy::too_many_arguments)]
fn persistent_device_expert_layer(
    weights: &GpuWeightCache,
    mlp_prefix: &str,
    layer: usize,
    hidden: usize,
    n_routed_experts: usize,
    generation: u32,
    pool: &ActPool,
    ctx: &MetalContext,
) -> Result<Option<PersistentDeviceExpertLayer>> {
    let mut layers = pool
        .persistent_expert_layers
        .lock()
        .expect("persistent device expert layers");
    if layer >= layers.len() {
        return Err(Error::Gravity(format!(
            "persistent device expert layer {layer} exceeds pool extent {}",
            layers.len()
        )));
    }
    if let Some(state) = &layers[layer] {
        return Ok(Some(state.clone()));
    }
    let state = build_persistent_device_expert_layer(
        weights,
        mlp_prefix,
        hidden,
        n_routed_experts,
        generation,
        &[],
        ctx,
    )?;
    if let Some(state) = &state {
        layers[layer] = Some(state.clone());
    }
    Ok(state)
}

#[allow(clippy::too_many_arguments)]
fn refresh_persistent_device_expert_layer(
    weights: &GpuWeightCache,
    mlp_prefix: &str,
    layer: usize,
    hidden: usize,
    n_routed_experts: usize,
    generation: u32,
    selected_experts: &[usize],
    pool: &ActPool,
    ctx: &MetalContext,
) -> Result<()> {
    let Some(state) = build_persistent_device_expert_layer(
        weights,
        mlp_prefix,
        hidden,
        n_routed_experts,
        generation,
        selected_experts,
        ctx,
    )?
    else {
        return Ok(());
    };
    let mut layers = pool
        .persistent_expert_layers
        .lock()
        .expect("persistent device expert layers");
    let layer_count = layers.len();
    let slot = layers.get_mut(layer).ok_or_else(|| {
        Error::Gravity(format!(
            "persistent device expert refresh layer {layer} exceeds pool extent {}",
            layer_count
        ))
    })?;
    *slot = Some(state);
    Ok(())
}

struct DeviceExpertReplayStageSpec {
    kernel: &'static str,
    grid: (u32, u32, u32),
    threadgroup: (u32, u32, u32),
    bindings: Vec<ReplayBufferBinding>,
    parameter_index: usize,
    parameter_offset: usize,
}

#[derive(Default)]
struct DeviceExpertReplayPlan {
    stages: Vec<DeviceExpertReplayStageSpec>,
    parameters: Vec<u8>,
}

impl DeviceExpertReplayPlan {
    fn push<T: bytemuck::Pod>(
        &mut self,
        kernel: &'static str,
        grid: (u32, u32, u32),
        threadgroup: (u32, u32, u32),
        bindings: Vec<ReplayBufferBinding>,
        parameter_index: usize,
        parameters: &T,
    ) {
        let align = std::mem::align_of::<T>();
        let padding = (align - (self.parameters.len() % align)) % align;
        self.parameters
            .resize(self.parameters.len().saturating_add(padding), 0);
        let parameter_offset = self.parameters.len();
        self.parameters
            .extend_from_slice(bytemuck::bytes_of(parameters));
        self.stages.push(DeviceExpertReplayStageSpec {
            kernel,
            grid,
            threadgroup,
            bindings,
            parameter_index,
            parameter_offset,
        });
    }

    fn finish(
        self,
        ctx: &MetalContext,
        indirect_resources: Vec<ReplayResourceDeclaration>,
    ) -> Result<ReplayableComputeGraph> {
        if self.parameters.is_empty() {
            return Err(Error::Gravity(
                "device expert replay graph has no persistent parameters".into(),
            ));
        }
        let parameter_buffer = ctx.new_buffer_with_bytes_checked(&self.parameters)?;
        crate::cost_ledger::record_allocation(parameter_buffer.length());
        let stages = self
            .stages
            .into_iter()
            .enumerate()
            .map(|(stage_index, mut spec)| {
                spec.bindings.push(ReplayBufferBinding::read(
                    spec.parameter_index,
                    &parameter_buffer,
                    spec.parameter_offset,
                ));
                let stage = ReplayComputeStage::new(
                    spec.kernel,
                    spec.grid,
                    spec.threadgroup,
                    spec.bindings,
                );
                if stage_index == 0 {
                    stage
                } else {
                    stage.with_barrier_before()
                }
            })
            .collect();
        ReplayableComputeGraph::new_with_resources(ctx, stages, indirect_resources)
    }
}

fn replay_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::Gravity(format!("{label} {value} exceeds the Metal u32 ABI")))
}

fn replay_grid(n: u32, divisor: u32, threads: u32, label: &str) -> Result<u32> {
    n.div_ceil(divisor)
        .checked_mul(threads)
        .ok_or_else(|| Error::Gravity(format!("{label} grid size overflow")))
}

fn device_expert_replay_key(
    generation: u32,
    experts_per_token: usize,
    hidden: usize,
    intermediate: usize,
    routed_dispatch_mode: DeviceExpertDispatchMode,
    shared_dispatch_mode: DeviceExpertDispatchMode,
    routed: &DeviceExpertTableLease,
    shared: &DeviceExpertTableLease,
    x: &Buffer,
    residual: &Buffer,
    pool: &ActPool,
    scratch: &ExpertWaveScratch,
) -> DeviceExpertReplayKey {
    let mut buffer_addresses = Vec::new();
    visit_device_expert_replay_buffers(
        experts_per_token,
        routed,
        shared,
        x,
        residual,
        pool,
        scratch,
        |buffer| buffer_addresses.push(buffer.gpu_address()),
    );
    DeviceExpertReplayKey {
        generation,
        experts_per_token,
        hidden,
        intermediate,
        routed_dispatch_mode,
        shared_dispatch_mode,
        buffer_addresses,
    }
}

#[allow(clippy::too_many_arguments)]
fn visit_device_expert_replay_buffers(
    experts_per_token: usize,
    routed: &DeviceExpertTableLease,
    shared: &DeviceExpertTableLease,
    x: &Buffer,
    residual: &Buffer,
    pool: &ActPool,
    scratch: &ExpertWaveScratch,
    mut visit: impl FnMut(&Buffer),
) {
    visit(&routed.table);
    for resource in &routed.resources {
        visit(resource);
    }
    visit(&shared.table);
    for resource in &shared.resources {
        visit(resource);
    }
    visit(x);
    visit(residual);
    visit(&pool.expert_idx);
    visit(&pool.expert_exec_slots);
    visit(&pool.expert_miss_mask);
    visit(&pool.expert_w);
    visit(&pool.shared_expert_idx);
    visit(&pool.shared_expert_slot);
    visit(&scratch.combined);
    for position in 0..=experts_per_token {
        visit(&scratch.gate[position]);
        visit(&scratch.up[position]);
        visit(&scratch.act[position]);
        visit(&scratch.down[position]);
    }
}

#[allow(clippy::too_many_arguments)]
fn device_expert_replay_key_matches(
    key: &DeviceExpertReplayKey,
    generation: u32,
    experts_per_token: usize,
    hidden: usize,
    intermediate: usize,
    routed_dispatch_mode: DeviceExpertDispatchMode,
    shared_dispatch_mode: DeviceExpertDispatchMode,
    routed: &DeviceExpertTableLease,
    shared: &DeviceExpertTableLease,
    x: &Buffer,
    residual: &Buffer,
    pool: &ActPool,
    scratch: &ExpertWaveScratch,
) -> bool {
    if key.generation != generation
        || key.experts_per_token != experts_per_token
        || key.hidden != hidden
        || key.intermediate != intermediate
        || key.routed_dispatch_mode != routed_dispatch_mode
        || key.shared_dispatch_mode != shared_dispatch_mode
    {
        return false;
    }
    let mut index = 0usize;
    let mut matches = true;
    visit_device_expert_replay_buffers(
        experts_per_token,
        routed,
        shared,
        x,
        residual,
        pool,
        scratch,
        |buffer| {
            matches &= key
                .buffer_addresses
                .get(index)
                .is_some_and(|&address| address == buffer.gpu_address());
            index = index.saturating_add(1);
        },
    );
    matches && index == key.buffer_addresses.len()
}

#[allow(clippy::too_many_arguments)]
fn push_device_expert_replay_matvec(
    plan: &mut DeviceExpertReplayPlan,
    mode: DeviceExpertDispatchMode,
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
            "invalid replay device expert position/projection: \
             {execution_position}/{experts_per_token}, projection {projection}"
        )));
    }
    require_f32_elements(x, cols, "replay device expert matvec input")?;
    require_f32_elements(y, rows, "replay device expert matvec output")?;
    let rows_u32 = replay_u32(rows, "replay device expert rows")?;
    let cols_u32 = replay_u32(cols, "replay device expert cols")?;
    let parameters = |allow_other_kind: bool| DeviceExpertTableMatvecParams {
        n_experts: lease.n_experts as u32,
        experts_per_token: experts_per_token as u32,
        generation: lease.generation,
        execution_position: execution_position as u32,
        projection,
        rows: rows_u32,
        cols: cols_u32,
        allow_other_kind: u32::from(allow_other_kind),
    };
    let bindings = || {
        vec![
            ReplayBufferBinding::read(0, expert_indices, 0),
            ReplayBufferBinding::read(1, expert_exec_slots, 0),
            ReplayBufferBinding::read(2, &lease.table, 0),
            ReplayBufferBinding::read_write(3, miss_mask, 0),
            ReplayBufferBinding::read(4, x, 0),
            ReplayBufferBinding::write(5, y, 0),
        ]
    };
    match mode {
        DeviceExpertDispatchMode::PqOnly => plan.push(
            "gravity_glm_expert_table_pq_matvec",
            (
                replay_grid(rows_u32, 8, 256, "replay device expert PQ matvec")?,
                1,
                1,
            ),
            (256, 1, 1),
            bindings(),
            6,
            &parameters(false),
        ),
        DeviceExpertDispatchMode::NativeBf16Only => plan.push(
            "gravity_glm_expert_table_native_bf16_matvec",
            (
                replay_grid(rows_u32, 256, 256, "replay native device expert matvec")?,
                1,
                1,
            ),
            (256, 1, 1),
            bindings(),
            6,
            &parameters(false),
        ),
        DeviceExpertDispatchMode::Heterogeneous => {
            plan.push(
                "gravity_glm_expert_table_pq_matvec",
                (
                    replay_grid(rows_u32, 8, 256, "replay heterogeneous PQ matvec")?,
                    1,
                    1,
                ),
                (256, 1, 1),
                bindings(),
                6,
                &parameters(true),
            );
            plan.push(
                "gravity_glm_expert_table_native_bf16_matvec",
                (
                    replay_grid(rows_u32, 256, 256, "replay heterogeneous native matvec")?,
                    1,
                    1,
                ),
                (256, 1, 1),
                bindings(),
                6,
                &parameters(true),
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_device_expert_replay_graph(
    ctx: &MetalContext,
    routed_dispatch_mode: DeviceExpertDispatchMode,
    shared_dispatch_mode: DeviceExpertDispatchMode,
    routed: &DeviceExpertTableLease,
    shared: &DeviceExpertTableLease,
    experts_per_token: usize,
    hidden: usize,
    intermediate: usize,
    x: &Buffer,
    residual: &Buffer,
    pool: &ActPool,
    scratch: &ExpertWaveScratch,
) -> Result<ReplayableComputeGraph> {
    if experts_per_token == 0
        || experts_per_token > 32
        || routed.generation == 0
        || shared.generation == 0
        || shared.n_experts != 1
        || scratch.gate.len() <= experts_per_token
        || scratch.up.len() <= experts_per_token
        || scratch.act.len() <= experts_per_token
        || scratch.down.len() <= experts_per_token
    {
        return Err(Error::Gravity(
            "device expert replay graph received an invalid lease or scratch geometry".into(),
        ));
    }
    let selected_bytes = experts_per_token
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| Error::Gravity("device expert replay selection byte overflow".into()))?
        as u64;
    let routed_table_bytes = routed
        .n_experts
        .checked_mul(std::mem::size_of::<DeviceExpertTriplet>())
        .ok_or_else(|| Error::Gravity("device expert replay routed table overflow".into()))?
        as u64;
    if routed.table.length() != routed_table_bytes
        || shared.table.length() != std::mem::size_of::<DeviceExpertTriplet>() as u64
        || pool.expert_idx.length() < selected_bytes
        || pool.expert_exec_slots.length() < selected_bytes
        || pool.expert_w.length() < experts_per_token as u64 * 4
        || pool.expert_miss_mask.length() < 4
        || pool.shared_expert_idx.length() < 4
        || pool.shared_expert_slot.length() < 4
    {
        return Err(Error::Gravity(
            "device expert replay graph received an undersized table or selection buffer".into(),
        ));
    }
    require_f32_elements(x, hidden, "device expert replay input")?;
    require_f32_elements(residual, hidden, "device expert replay residual")?;
    require_f32_elements(
        &scratch.combined,
        hidden,
        "device expert replay combined output",
    )?;
    for position in 0..=experts_per_token {
        require_f32_elements(
            &scratch.gate[position],
            intermediate,
            "device expert replay gate scratch",
        )?;
        require_f32_elements(
            &scratch.up[position],
            intermediate,
            "device expert replay up scratch",
        )?;
        require_f32_elements(
            &scratch.act[position],
            intermediate,
            "device expert replay activation scratch",
        )?;
        require_f32_elements(
            &scratch.down[position],
            hidden,
            "device expert replay down scratch",
        )?;
    }

    let hidden_u32 = replay_u32(hidden, "device expert replay hidden")?;
    let intermediate_u32 = replay_u32(intermediate, "device expert replay intermediate")?;
    let experts_u32 = replay_u32(experts_per_token, "device expert replay experts")?;
    let mut plan = DeviceExpertReplayPlan::default();
    let validate = DeviceExpertTableValidateParams {
        n_experts: routed.n_experts as u32,
        experts_per_token: experts_u32,
        generation: routed.generation,
        required_kind: match routed_dispatch_mode {
            DeviceExpertDispatchMode::PqOnly => DEVICE_EXPERT_TENSOR_KIND_PQ,
            DeviceExpertDispatchMode::NativeBf16Only => DEVICE_EXPERT_TENSOR_KIND_NATIVE_BF16,
            DeviceExpertDispatchMode::Heterogeneous => DEVICE_EXPERT_TENSOR_KIND_ANY_SUPPORTED,
        },
        hidden: hidden_u32,
        intermediate: intermediate_u32,
    };
    plan.push(
        "gravity_glm_expert_table_validate",
        (1, 1, 1),
        (1, 1, 1),
        vec![
            ReplayBufferBinding::read(0, &pool.expert_idx, 0),
            ReplayBufferBinding::read(1, &pool.expert_exec_slots, 0),
            ReplayBufferBinding::read(2, &routed.table, 0),
            ReplayBufferBinding::write(3, &pool.expert_miss_mask, 0),
        ],
        4,
        &validate,
    );
    plan.push(
        "gravity_glm_expert_table_zero_f32",
        (
            replay_grid(hidden_u32, 256, 256, "device expert replay guarded zero")?,
            1,
            1,
        ),
        (256, 1, 1),
        vec![
            ReplayBufferBinding::write(0, &scratch.combined, 0),
            ReplayBufferBinding::read(1, &pool.expert_miss_mask, 0),
        ],
        2,
        &hidden_u32,
    );

    for execution_position in 0..experts_per_token {
        push_device_expert_replay_matvec(
            &mut plan,
            routed_dispatch_mode,
            routed,
            &pool.expert_idx,
            &pool.expert_exec_slots,
            &pool.expert_miss_mask,
            experts_per_token,
            execution_position,
            0,
            x,
            intermediate,
            hidden,
            &scratch.gate[execution_position],
        )?;
        push_device_expert_replay_matvec(
            &mut plan,
            routed_dispatch_mode,
            routed,
            &pool.expert_idx,
            &pool.expert_exec_slots,
            &pool.expert_miss_mask,
            experts_per_token,
            execution_position,
            1,
            x,
            intermediate,
            hidden,
            &scratch.up[execution_position],
        )?;
        plan.push(
            "gravity_glm_expert_table_silu_mul_f32",
            (
                replay_grid(intermediate_u32, 256, 256, "device expert replay SiLU")?,
                1,
                1,
            ),
            (256, 1, 1),
            vec![
                ReplayBufferBinding::read(0, &scratch.gate[execution_position], 0),
                ReplayBufferBinding::read(1, &scratch.up[execution_position], 0),
                ReplayBufferBinding::write(2, &scratch.act[execution_position], 0),
                ReplayBufferBinding::read(3, &pool.expert_miss_mask, 0),
            ],
            4,
            &intermediate_u32,
        );
        push_device_expert_replay_matvec(
            &mut plan,
            routed_dispatch_mode,
            routed,
            &pool.expert_idx,
            &pool.expert_exec_slots,
            &pool.expert_miss_mask,
            experts_per_token,
            execution_position,
            2,
            &scratch.act[execution_position],
            hidden,
            intermediate,
            &scratch.down[execution_position],
        )?;
        let axpy = DeviceExpertTableAxpyParams {
            n: hidden_u32,
            experts_per_token: experts_u32,
            execution_position: execution_position as u32,
            use_router_weight: 1,
        };
        plan.push(
            "gravity_glm_expert_table_axpy_f32",
            (
                replay_grid(hidden_u32, 256, 256, "device expert replay routed AXPY")?,
                1,
                1,
            ),
            (256, 1, 1),
            vec![
                ReplayBufferBinding::read_write(0, &scratch.combined, 0),
                ReplayBufferBinding::read(1, &scratch.down[execution_position], 0),
                ReplayBufferBinding::read(2, &pool.expert_w, 0),
                ReplayBufferBinding::read(3, &pool.expert_exec_slots, 0),
                ReplayBufferBinding::read(4, &pool.expert_miss_mask, 0),
            ],
            5,
            &axpy,
        );
    }

    let shared_position = experts_per_token;
    push_device_expert_replay_matvec(
        &mut plan,
        shared_dispatch_mode,
        shared,
        &pool.shared_expert_idx,
        &pool.shared_expert_slot,
        &pool.expert_miss_mask,
        1,
        0,
        0,
        x,
        intermediate,
        hidden,
        &scratch.gate[shared_position],
    )?;
    push_device_expert_replay_matvec(
        &mut plan,
        shared_dispatch_mode,
        shared,
        &pool.shared_expert_idx,
        &pool.shared_expert_slot,
        &pool.expert_miss_mask,
        1,
        0,
        1,
        x,
        intermediate,
        hidden,
        &scratch.up[shared_position],
    )?;
    plan.push(
        "gravity_glm_expert_table_silu_mul_f32",
        (
            replay_grid(
                intermediate_u32,
                256,
                256,
                "device expert replay shared SiLU",
            )?,
            1,
            1,
        ),
        (256, 1, 1),
        vec![
            ReplayBufferBinding::read(0, &scratch.gate[shared_position], 0),
            ReplayBufferBinding::read(1, &scratch.up[shared_position], 0),
            ReplayBufferBinding::write(2, &scratch.act[shared_position], 0),
            ReplayBufferBinding::read(3, &pool.expert_miss_mask, 0),
        ],
        4,
        &intermediate_u32,
    );
    push_device_expert_replay_matvec(
        &mut plan,
        shared_dispatch_mode,
        shared,
        &pool.shared_expert_idx,
        &pool.shared_expert_slot,
        &pool.expert_miss_mask,
        1,
        0,
        2,
        &scratch.act[shared_position],
        hidden,
        intermediate,
        &scratch.down[shared_position],
    )?;
    let shared_axpy = DeviceExpertTableAxpyParams {
        n: hidden_u32,
        experts_per_token: 1,
        execution_position: 0,
        use_router_weight: 0,
    };
    plan.push(
        "gravity_glm_expert_table_axpy_f32",
        (
            replay_grid(hidden_u32, 256, 256, "device expert replay shared AXPY")?,
            1,
            1,
        ),
        (256, 1, 1),
        vec![
            ReplayBufferBinding::read_write(0, &scratch.combined, 0),
            ReplayBufferBinding::read(1, &scratch.down[shared_position], 0),
            ReplayBufferBinding::read(2, &pool.expert_w, 0),
            ReplayBufferBinding::read(3, &pool.shared_expert_slot, 0),
            ReplayBufferBinding::read(4, &pool.expert_miss_mask, 0),
        ],
        5,
        &shared_axpy,
    );
    plan.push(
        "gravity_glm_expert_table_residual_add_f32",
        (
            replay_grid(hidden_u32, 256, 256, "device expert replay residual add")?,
            1,
            1,
        ),
        (256, 1, 1),
        vec![
            ReplayBufferBinding::read_write(0, residual, 0),
            ReplayBufferBinding::read(1, &scratch.combined, 0),
            ReplayBufferBinding::read(2, &pool.expert_miss_mask, 0),
        ],
        3,
        &hidden_u32,
    );

    let indirect_resources = routed
        .resources
        .iter()
        .chain(shared.resources.iter())
        .map(ReplayResourceDeclaration::read)
        .collect();
    plan.finish(ctx, indirect_resources)
}

/// Append the cache-indexed routed/shared expert graph after an already
/// encoded device-router selection.
///
/// A hit commits router + trace + validation + all expert work + residual as
/// one command buffer and downloads only the four-byte miss mask. A miss
/// commits the same guarded graph, whose validation prevents every subsequent
/// write, and lets the caller replay through the qualified host-known wave.
/// Unsupported shared-expert layouts leave the router command buffer open for
/// the caller's ordinary selection readback.
#[allow(clippy::too_many_arguments)]
fn moe_device_table_wave<'a>(
    weights: &GpuWeightCache,
    mlp_prefix: &str,
    layer: usize,
    hidden: usize,
    experts_per_token: usize,
    n_routed_experts: usize,
    generation: u32,
    x: &Buffer,
    residual: &Buffer,
    pool: &ActPool,
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<DeviceExpertTableWaveResult> {
    if generation == 0 {
        return Err(Error::Gravity(
            "device expert production table requires a nonzero generation".into(),
        ));
    }
    if experts_per_token == 0 || experts_per_token > 32 {
        return Ok(DeviceExpertTableWaveResult::Unsupported);
    }
    if tcb.is_none() {
        return Err(Error::Gravity(
            "device expert table wave requires an open router command buffer".into(),
        ));
    }

    let Some(layer_state) = persistent_device_expert_layer(
        weights,
        mlp_prefix,
        layer,
        hidden,
        n_routed_experts,
        generation,
        pool,
        ctx,
    )?
    else {
        return Ok(DeviceExpertTableWaveResult::Unsupported);
    };
    let replay_graph_cache = layer_state.replay_graph.clone();
    let routed_lease = layer_state.routed;
    let shared_lease = layer_state.shared;
    let intermediate = layer_state.intermediate;
    let metrics = layer_state.metrics;
    let routed_dispatch_mode = layer_state.routed_dispatch_mode;
    let shared_dispatch_mode = layer_state.shared_dispatch_mode;

    let scratch_guard =
        pool.ensure_expert_wave_scratch(ctx, experts_per_token + 1, intermediate, hidden)?;
    let scratch = scratch_guard
        .as_ref()
        .expect("device expert table scratch ensured");
    let wave = tcb
        .as_mut()
        .expect("device expert table router command buffer");

    if gpu_expert_table_icb_enabled() {
        let mut cached = replay_graph_cache
            .lock()
            .expect("device expert replay graph");
        let cache_hit = cached.as_ref().is_some_and(|state| {
            device_expert_replay_key_matches(
                &state.key,
                generation,
                experts_per_token,
                hidden,
                intermediate,
                routed_dispatch_mode,
                shared_dispatch_mode,
                &routed_lease,
                &shared_lease,
                x,
                residual,
                pool,
                scratch,
            )
        });
        if !cache_hit {
            let key = device_expert_replay_key(
                generation,
                experts_per_token,
                hidden,
                intermediate,
                routed_dispatch_mode,
                shared_dispatch_mode,
                &routed_lease,
                &shared_lease,
                x,
                residual,
                pool,
                scratch,
            );
            let graph = build_device_expert_replay_graph(
                ctx,
                routed_dispatch_mode,
                shared_dispatch_mode,
                &routed_lease,
                &shared_lease,
                experts_per_token,
                hidden,
                intermediate,
                x,
                residual,
                pool,
                scratch,
            )?;
            *cached = Some(CachedDeviceExpertReplayGraph { key, graph });
        }
        wave.execute_replayable_graph(
            &cached
                .as_ref()
                .expect("device expert replay graph constructed")
                .graph,
        )?;
    } else {
        encode_device_expert_table_validate(
            wave,
            &routed_lease,
            &pool.expert_idx,
            &pool.expert_exec_slots,
            &pool.expert_miss_mask,
            experts_per_token,
            hidden,
            intermediate,
            match routed_dispatch_mode {
                DeviceExpertDispatchMode::PqOnly => DEVICE_EXPERT_TENSOR_KIND_PQ,
                DeviceExpertDispatchMode::NativeBf16Only => DEVICE_EXPERT_TENSOR_KIND_NATIVE_BF16,
                DeviceExpertDispatchMode::Heterogeneous => DEVICE_EXPERT_TENSOR_KIND_ANY_SUPPORTED,
            },
        )?;
        encode_device_expert_table_zero(wave, &scratch.combined, &pool.expert_miss_mask, hidden)?;

        for execution_position in 0..experts_per_token {
            encode_device_expert_table_matvec(
                wave,
                routed_dispatch_mode,
                &routed_lease,
                &pool.expert_idx,
                &pool.expert_exec_slots,
                &pool.expert_miss_mask,
                experts_per_token,
                execution_position,
                0,
                x,
                intermediate,
                hidden,
                &scratch.gate[execution_position],
            )?;
            encode_device_expert_table_matvec(
                wave,
                routed_dispatch_mode,
                &routed_lease,
                &pool.expert_idx,
                &pool.expert_exec_slots,
                &pool.expert_miss_mask,
                experts_per_token,
                execution_position,
                1,
                x,
                intermediate,
                hidden,
                &scratch.up[execution_position],
            )?;
            encode_device_expert_table_silu_mul(
                wave,
                &scratch.gate[execution_position],
                &scratch.up[execution_position],
                &scratch.act[execution_position],
                &pool.expert_miss_mask,
                intermediate,
            )?;
            encode_device_expert_table_matvec(
                wave,
                routed_dispatch_mode,
                &routed_lease,
                &pool.expert_idx,
                &pool.expert_exec_slots,
                &pool.expert_miss_mask,
                experts_per_token,
                execution_position,
                2,
                &scratch.act[execution_position],
                hidden,
                intermediate,
                &scratch.down[execution_position],
            )?;
            encode_device_expert_table_axpy(
                wave,
                &scratch.combined,
                &scratch.down[execution_position],
                &pool.expert_w,
                &pool.expert_exec_slots,
                &pool.expert_miss_mask,
                hidden,
                experts_per_token,
                execution_position,
                true,
            )?;
        }

        let shared_position = experts_per_token;
        encode_device_expert_table_matvec(
            wave,
            shared_dispatch_mode,
            &shared_lease,
            &pool.shared_expert_idx,
            &pool.shared_expert_slot,
            &pool.expert_miss_mask,
            1,
            0,
            0,
            x,
            intermediate,
            hidden,
            &scratch.gate[shared_position],
        )?;
        encode_device_expert_table_matvec(
            wave,
            shared_dispatch_mode,
            &shared_lease,
            &pool.shared_expert_idx,
            &pool.shared_expert_slot,
            &pool.expert_miss_mask,
            1,
            0,
            1,
            x,
            intermediate,
            hidden,
            &scratch.up[shared_position],
        )?;
        encode_device_expert_table_silu_mul(
            wave,
            &scratch.gate[shared_position],
            &scratch.up[shared_position],
            &scratch.act[shared_position],
            &pool.expert_miss_mask,
            intermediate,
        )?;
        encode_device_expert_table_matvec(
            wave,
            shared_dispatch_mode,
            &shared_lease,
            &pool.shared_expert_idx,
            &pool.shared_expert_slot,
            &pool.expert_miss_mask,
            1,
            0,
            2,
            &scratch.act[shared_position],
            hidden,
            intermediate,
            &scratch.down[shared_position],
        )?;
        encode_device_expert_table_axpy(
            wave,
            &scratch.combined,
            &scratch.down[shared_position],
            &pool.expert_w,
            &pool.shared_expert_slot,
            &pool.expert_miss_mask,
            hidden,
            1,
            0,
            false,
        )?;
        encode_device_expert_table_residual_add(
            wave,
            residual,
            &scratch.combined,
            &pool.expert_miss_mask,
            hidden,
        )?;
    }

    commit(tcb.take(), waits)?;
    let miss_mask = read_u32(&pool.expert_miss_mask, 1)[0];
    crate::cost_ledger::record_transfer(4, false, "device_expert_table_miss_mask_download");
    if miss_mask == 0 {
        if metrics.routed.len() != experts_per_token {
            return Err(Error::Gravity(format!(
                "device expert table hit admitted {} routed metric triplets for \
                 {experts_per_token} execution positions",
                metrics.routed.len()
            )));
        }
        record_device_expert_table_hit_costs(mlp_prefix, hidden, intermediate, &metrics);
        Ok(DeviceExpertTableWaveResult::Hit)
    } else {
        Ok(DeviceExpertTableWaveResult::Miss(miss_mask))
    }
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
                Some(GpuTensor::Pq { .. })
                    | Some(GpuTensor::NativeGpuBf16 { .. })
                    | Some(GpuTensor::ActivationAware { .. })
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
            GpuTensor::ActivationAware { params, .. } => params.rows as usize,
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
    fn f64_authority_matvec(weights: &[f32], cols: usize, x: &[f32]) -> Vec<f64> {
        weights
            .chunks_exact(cols)
            .map(|row| row.iter().zip(x).map(|(&w, &a)| w as f64 * a as f64).sum())
            .collect()
    }
    fn assert_v21_gate(label: &str, host: &[f32], device: &[f32], authority: &[f64]) {
        let score = score_pair(host, device, authority, &Bounds::continuous_only());
        eprintln!(
            "{label}: rel_l2={:.3e} meaningful={:.3e} greedy={} top5={}",
            score.device.continuous.relative_l2,
            score.device.continuous.max_meaningful_rel,
            score.device.discrete.greedy_match,
            score.device.discrete.top_k_exact_match
        );
        assert!(
            score.pass,
            "{label} failed V2.1: host={:?}, device={:?}",
            score.host.failures, score.device.failures
        );
    }
    fn single_route_bufs(ctx: &MetalContext) -> (Buffer, Buffer, Buffer) {
        (
            u32_buffer(ctx, &[0]),
            u32_buffer(ctx, &[0]),
            u32_buffer(ctx, &[u32::MAX]),
        )
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
    fn pack_msb_indices(indices: &[usize], bits: usize) -> Vec<u8> {
        assert!((1..=8).contains(&bits));
        let mut packed = vec![0u8; (indices.len() * bits).div_ceil(8) + 4];
        for (index, &value) in indices.iter().enumerate() {
            assert!(value < (1usize << bits));
            let bit_offset = index * bits;
            for bit in 0..bits {
                let source = (value >> (bits - 1 - bit)) & 1;
                if source != 0 {
                    let destination = bit_offset + bit;
                    packed[destination / 8] |= 1u8 << (7 - destination % 8);
                }
            }
        }
        packed
    }
    fn packed_r0_pq_tensor(
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        salt: usize,
    ) -> (GpuTensor, Vec<f32>) {
        const DIM: usize = 8;
        const CARD: usize = 128;
        const BITS: usize = 7;
        assert_eq!(cols % DIM, 0);
        let nchunk = cols / DIM;
        let codebooks: Vec<half::f16> = (0..CARD * DIM)
            .map(|flat| {
                let code = flat / DIM;
                let element = flat % DIM;
                let positive = ((code * 11 + element * 7 + salt * 13) % 61) + 1;
                half::f16::from_f32(positive as f32 * (1.0 / 192.0))
            })
            .collect();
        let indices: Vec<usize> = (0..rows * nchunk)
            .map(|flat| (flat * 29 + salt * 17 + flat / nchunk * 3) % CARD)
            .collect();
        let codes = pack_msb_indices(&indices, BITS);
        let mut dense = vec![0.0f32; rows * cols];
        for row in 0..rows {
            for chunk in 0..nchunk {
                let code = indices[row * nchunk + chunk];
                for element in 0..DIM {
                    dense[row * cols + chunk * DIM + element] =
                        codebooks[code * DIM + element].to_f32();
                }
            }
        }
        let codebooks = f16_buffer(ctx, &codebooks);
        let codes = ctx
            .new_buffer_with_bytes_checked(&codes)
            .expect("packed-r0 codes");
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
                    bits: BITS as u32,
                },
            },
            dense,
        )
    }
    fn native_bf16_tensor(
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        salt: usize,
    ) -> (GpuTensor, Vec<f32>) {
        let bits: Vec<u16> = (0..rows * cols)
            .map(|flat| {
                let positive = ((flat * 13 + salt * 17 + flat / cols * 5) % 63) + 1;
                let value = positive as f32 * (1.0 / 64.0);
                (value.to_bits() >> 16) as u16
            })
            .collect();
        let dense: Vec<f32> = bits
            .iter()
            .map(|&value| f32::from_bits((value as u32) << 16))
            .collect();
        let buf = ctx
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&bits))
            .expect("native-bf16 weights");
        (
            GpuTensor::NativeGpuBf16 {
                buf,
                rows: rows as u32,
                cols: cols as u32,
            },
            dense,
        )
    }
    fn gpu_tensor_bytes(tensor: &GpuTensor) -> u64 {
        match tensor {
            GpuTensor::Pq {
                codebooks, codes, ..
            } => codebooks.length() + codes.length(),
            GpuTensor::ActivationAware {
                coefficients,
                basis,
                ..
            } => coefficients.length() + basis.length(),
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
        let selected_only =
            build_selected_device_expert_table_snapshot(&ctx, &cache, PREFIX, 4, GENERATION, &[2])
                .expect("selected-only immutable expert table");
        assert_eq!(selected_only.ready_entries, 1);
        assert_eq!(selected_only.resources.len(), 6);
        let selected_entries = unsafe {
            std::slice::from_raw_parts(
                selected_only.table.contents() as *const DeviceExpertTriplet,
                selected_only.n_experts,
            )
        };
        assert_eq!(selected_entries[0], DeviceExpertTriplet::default());
        assert_eq!(selected_entries[2].ready_mask, DEVICE_EXPERT_TRIPLET_READY);
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
            false,
        )
        .expect("indirect gate projection");
        assert_eq!(hit.dispatch_count(), 2);
        hit.commit_and_wait().expect("resident-hit command");
        assert_eq!(read_u32(&miss_mask, 1), vec![0]);
        let weights = &gate_authorities[&0];
        let host = matvec_dense(weights, &x_values, "expert-0 gate authority")
            .expect("host gate comparator");
        let authority = f64_authority_matvec(&weights, HIDDEN, &x_values);
        let device = read_f32(&y, INTERMEDIATE);
        assert_v21_gate(
            "device expert table indirect gate",
            &host,
            &device,
            &authority,
        );
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
            false,
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
    fn device_expert_table_packed_r0_is_indirect_and_invalid_bits_fail_closed() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const HIDDEN: usize = 32;
        const INTERMEDIATE: usize = 24;
        const GENERATION: u32 = 13;
        let (gate, gate_dense) = packed_r0_pq_tensor(&ctx, INTERMEDIATE, HIDDEN, 1);
        let (up, _) = packed_r0_pq_tensor(&ctx, INTERMEDIATE, HIDDEN, 2);
        let (down, _) = packed_r0_pq_tensor(&ctx, HIDDEN, INTERMEDIATE, 3);
        let lease = build_single_device_expert_snapshot(&ctx, &gate, &up, &down, GENERATION)
            .expect("packed-r0 immutable expert table");
        assert_eq!(lease.ready_entries, 1);
        assert_eq!(lease.resources.len(), 6);
        let (expert_indices, execution_slots, miss_mask) = single_route_bufs(&ctx);
        let x_values: Vec<f32> = deterministic_fixture_f32(0x70_2026, HIDDEN, 0.5)
            .into_iter()
            .map(|value| value.abs() + 0.0625)
            .collect();
        let x = f32_buffer(&ctx, &x_values);
        let y = filled_f32_buffer(&ctx, INTERMEDIATE, -4_096.0);
        let mut hit = TokenCommandBuffer::new(&ctx);
        encode_device_expert_table_validate(
            &mut hit,
            &lease,
            &expert_indices,
            &execution_slots,
            &miss_mask,
            1,
            HIDDEN,
            INTERMEDIATE,
            DEVICE_EXPERT_TENSOR_KIND_PQ,
        )
        .expect("validate packed-r0 hit");
        encode_device_expert_table_pq_matvec(
            &mut hit,
            &lease,
            &expert_indices,
            &execution_slots,
            &miss_mask,
            1,
            0,
            0,
            &x,
            INTERMEDIATE,
            HIDDEN,
            &y,
            false,
        )
        .expect("packed-r0 indirect gate");
        hit.commit_and_wait().expect("packed-r0 command");
        assert_eq!(read_u32(&miss_mask, 1), vec![0]);
        let host = matvec_dense(&gate_dense, &x_values, "packed-r0 host")
            .expect("packed-r0 host comparator");
        let authority = f64_authority_matvec(&gate_dense, HIDDEN, &x_values);
        let device = read_f32(&y, INTERMEDIATE);
        assert_v21_gate("packed-r0 indirect gate", &host, &device, &authority);
        assert_eq!(
            topk_desc_f64(&authority, 5),
            topk_desc_f64(
                &device.iter().map(|&value| value as f64).collect::<Vec<_>>(),
                5
            ),
            "packed-r0 top-5 must remain exact"
        );
        let invalid_table = unsafe {
            std::slice::from_raw_parts_mut(
                lease.table.contents() as *mut DeviceExpertTriplet,
                lease.n_experts,
            )
        };
        invalid_table[0].gate.bits = 9;
        let invalid_mask = u32_buffer(&ctx, &[u32::MAX]);
        let sentinel: Vec<f32> = (0..INTERMEDIATE)
            .map(|index| -2_000.0 - index as f32)
            .collect();
        let invalid_y = f32_buffer(&ctx, &sentinel);
        let mut invalid = TokenCommandBuffer::new(&ctx);
        encode_device_expert_table_validate(
            &mut invalid,
            &lease,
            &expert_indices,
            &execution_slots,
            &invalid_mask,
            1,
            HIDDEN,
            INTERMEDIATE,
            DEVICE_EXPERT_TENSOR_KIND_PQ,
        )
        .expect("validate invalid packed descriptor");
        encode_device_expert_table_pq_matvec(
            &mut invalid,
            &lease,
            &expert_indices,
            &execution_slots,
            &invalid_mask,
            1,
            0,
            0,
            &x,
            INTERMEDIATE,
            HIDDEN,
            &invalid_y,
            false,
        )
        .expect("suppressed invalid packed projection");
        invalid.commit_and_wait().expect("invalid packed command");
        assert_eq!(read_u32(&invalid_mask, 1), vec![1]);
        assert_eq!(
            read_f32(&invalid_y, INTERMEDIATE)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            sentinel
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "an invalid packed descriptor must not mutate the destination"
        );
    }
    #[test]
    fn device_expert_table_native_bf16_is_exact_and_invalid_descriptor_fails_closed() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const HIDDEN: usize = 31;
        const INTERMEDIATE: usize = 23;
        const GENERATION: u32 = 17;
        let (gate, gate_dense) = native_bf16_tensor(&ctx, INTERMEDIATE, HIDDEN, 1);
        let (up, _) = native_bf16_tensor(&ctx, INTERMEDIATE, HIDDEN, 2);
        let (down, _) = native_bf16_tensor(&ctx, HIDDEN, INTERMEDIATE, 3);
        let lease = build_single_device_expert_snapshot(&ctx, &gate, &up, &down, GENERATION)
            .expect("native-bf16 immutable expert table");
        assert_eq!(lease.ready_entries, 1);
        assert_eq!(lease.resources.len(), 3);
        let (expert_indices, execution_slots, miss_mask) = single_route_bufs(&ctx);
        let x_values: Vec<f32> = deterministic_fixture_f32(0xBF16_2026, HIDDEN, 0.25)
            .into_iter()
            .map(|value| value.abs() + 0.03125)
            .collect();
        let x = f32_buffer(&ctx, &x_values);
        let y = filled_f32_buffer(&ctx, INTERMEDIATE, -8_192.0);
        let mut hit = TokenCommandBuffer::new(&ctx);
        encode_device_expert_table_validate(
            &mut hit,
            &lease,
            &expert_indices,
            &execution_slots,
            &miss_mask,
            1,
            HIDDEN,
            INTERMEDIATE,
            DEVICE_EXPERT_TENSOR_KIND_NATIVE_BF16,
        )
        .expect("validate native-bf16 hit");
        encode_device_expert_table_native_bf16_matvec(
            &mut hit,
            &lease,
            &expert_indices,
            &execution_slots,
            &miss_mask,
            1,
            0,
            0,
            &x,
            INTERMEDIATE,
            HIDDEN,
            &y,
            false,
        )
        .expect("native-bf16 indirect gate");
        hit.commit_and_wait().expect("native-bf16 command");
        assert_eq!(read_u32(&miss_mask, 1), vec![0]);
        let host = matvec_dense(&gate_dense, &x_values, "native-bf16 host")
            .expect("native-bf16 host comparator");
        let authority = f64_authority_matvec(&gate_dense, HIDDEN, &x_values);
        let device = read_f32(&y, INTERMEDIATE);
        assert_eq!(
            device
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            host.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            "native-bf16 indirect projection must match sequential host bits"
        );
        assert_v21_gate("native-bf16 indirect gate", &host, &device, &authority);
        let invalid_table = unsafe {
            std::slice::from_raw_parts_mut(
                lease.table.contents() as *mut DeviceExpertTriplet,
                lease.n_experts,
            )
        };
        invalid_table[0].gate.secondary_address = 1;
        let invalid_mask = u32_buffer(&ctx, &[u32::MAX]);
        let sentinel: Vec<f32> = (0..INTERMEDIATE)
            .map(|index| 4_000.0 + index as f32)
            .collect();
        let invalid_y = f32_buffer(&ctx, &sentinel);
        let mut invalid = TokenCommandBuffer::new(&ctx);
        encode_device_expert_table_validate(
            &mut invalid,
            &lease,
            &expert_indices,
            &execution_slots,
            &invalid_mask,
            1,
            HIDDEN,
            INTERMEDIATE,
            DEVICE_EXPERT_TENSOR_KIND_NATIVE_BF16,
        )
        .expect("validate invalid native descriptor");
        encode_device_expert_table_native_bf16_matvec(
            &mut invalid,
            &lease,
            &expert_indices,
            &execution_slots,
            &invalid_mask,
            1,
            0,
            0,
            &x,
            INTERMEDIATE,
            HIDDEN,
            &invalid_y,
            false,
        )
        .expect("suppressed invalid native projection");
        invalid.commit_and_wait().expect("invalid native command");
        assert_eq!(read_u32(&invalid_mask, 1), vec![1]);
        assert_eq!(
            read_f32(&invalid_y, INTERMEDIATE)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            sentinel
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "an invalid native descriptor must not mutate the destination"
        );
    }
    #[test]
    fn device_expert_table_heterogeneous_triplet_executes_and_fails_closed() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const HIDDEN: usize = 32;
        const INTERMEDIATE: usize = 32;
        const GENERATION: u32 = 19;
        let (gate, gate_dense) = packed_r0_pq_tensor(&ctx, INTERMEDIATE, HIDDEN, 4);
        let (up, up_dense) = native_bf16_tensor(&ctx, INTERMEDIATE, HIDDEN, 5);
        let (down, down_dense) = direct_u8_pq_tensor(&ctx, HIDDEN, INTERMEDIATE, 6);
        let lease = build_single_device_expert_snapshot(&ctx, &gate, &up, &down, GENERATION)
            .expect("heterogeneous immutable expert table");
        assert_eq!(lease.ready_entries, 1);
        assert_eq!(lease.resources.len(), 5);
        let (_, triplet_metrics) =
            device_expert_triplet_metrics(&gate, &up, &down, HIDDEN).expect("mixed metrics");
        let metrics = DeviceExpertLayerMetrics {
            routed: vec![triplet_metrics],
            shared: triplet_metrics,
        };
        assert_eq!(
            metrics.dispatch_mode(),
            DeviceExpertDispatchMode::Heterogeneous
        );
        let (expert_indices, execution_slots, miss_mask) = single_route_bufs(&ctx);
        let x_values: Vec<f32> = deterministic_fixture_f32(0xA11C_E019, HIDDEN, 0.125)
            .into_iter()
            .map(|value| value.abs() + 0.03125)
            .collect();
        let x = f32_buffer(&ctx, &x_values);
        let gate_out = filled_f32_buffer(&ctx, INTERMEDIATE, -1.0);
        let up_out = filled_f32_buffer(&ctx, INTERMEDIATE, -2.0);
        let act = filled_f32_buffer(&ctx, INTERMEDIATE, -3.0);
        let down_out = filled_f32_buffer(&ctx, HIDDEN, -4.0);
        let mut hit = TokenCommandBuffer::new(&ctx);
        encode_device_expert_table_validate(
            &mut hit,
            &lease,
            &expert_indices,
            &execution_slots,
            &miss_mask,
            1,
            HIDDEN,
            INTERMEDIATE,
            DEVICE_EXPERT_TENSOR_KIND_ANY_SUPPORTED,
        )
        .expect("validate heterogeneous hit");
        encode_device_expert_table_matvec(
            &mut hit,
            DeviceExpertDispatchMode::Heterogeneous,
            &lease,
            &expert_indices,
            &execution_slots,
            &miss_mask,
            1,
            0,
            0,
            &x,
            INTERMEDIATE,
            HIDDEN,
            &gate_out,
        )
        .expect("heterogeneous gate");
        encode_device_expert_table_matvec(
            &mut hit,
            DeviceExpertDispatchMode::Heterogeneous,
            &lease,
            &expert_indices,
            &execution_slots,
            &miss_mask,
            1,
            0,
            1,
            &x,
            INTERMEDIATE,
            HIDDEN,
            &up_out,
        )
        .expect("heterogeneous up");
        encode_device_expert_table_silu_mul(
            &mut hit,
            &gate_out,
            &up_out,
            &act,
            &miss_mask,
            INTERMEDIATE,
        )
        .expect("heterogeneous silu");
        encode_device_expert_table_matvec(
            &mut hit,
            DeviceExpertDispatchMode::Heterogeneous,
            &lease,
            &expert_indices,
            &execution_slots,
            &miss_mask,
            1,
            0,
            2,
            &act,
            HIDDEN,
            INTERMEDIATE,
            &down_out,
        )
        .expect("heterogeneous down");
        assert_eq!(hit.dispatch_count(), 8);
        hit.commit_and_wait().expect("heterogeneous command");
        assert_eq!(read_u32(&miss_mask, 1), vec![0]);
        let host = fixture_mlp_f32(
            &gate_dense,
            &up_dense,
            &down_dense,
            HIDDEN,
            INTERMEDIATE,
            &x_values,
        );
        let authority = fixture_mlp_f64(
            &gate_dense,
            &up_dense,
            &down_dense,
            HIDDEN,
            INTERMEDIATE,
            &x_values,
        );
        let device = read_f32(&down_out, HIDDEN);
        assert_v21_gate("heterogeneous triplet", &host, &device, &authority);
        let mut replay_arch = tiny_arch();
        replay_arch.hidden = HIDDEN;
        replay_arch.n_routed_experts = 1;
        replay_arch.num_experts_per_tok = 1;
        let replay_pool = ActPool::new(&ctx, &replay_arch).expect("heterogeneous replay pool");
        unsafe {
            (replay_pool.expert_idx.contents() as *mut u32).write(0);
            (replay_pool.expert_exec_slots.contents() as *mut u32).write(0);
            (replay_pool.expert_miss_mask.contents() as *mut u32).write(u32::MAX);
        }
        write_f32(&replay_pool.expert_w, &[0.25]);
        let replay_scratch =
            ExpertWaveScratch::new(&ctx, 2, INTERMEDIATE, HIDDEN).expect("heterogeneous scratch");
        let residual_values = deterministic_fixture_f32(0x1CB0_0019, HIDDEN, 0.05);
        let replay_residual = f32_buffer(&ctx, &residual_values);
        let replay_graph = build_device_expert_replay_graph(
            &ctx,
            DeviceExpertDispatchMode::Heterogeneous,
            DeviceExpertDispatchMode::Heterogeneous,
            &lease,
            &lease,
            1,
            HIDDEN,
            INTERMEDIATE,
            &x,
            &replay_residual,
            &replay_pool,
            &replay_scratch,
        )
        .expect("heterogeneous complete-wave replay graph");
        assert_eq!(replay_graph.command_count(), 19);
        let mut replay = TokenCommandBuffer::new(&ctx);
        replay
            .execute_replayable_graph(&replay_graph)
            .expect("execute heterogeneous complete wave");
        assert_eq!(replay.dispatch_count(), 19);
        replay
            .commit_and_wait()
            .expect("heterogeneous complete-wave command");
        assert_eq!(read_u32(&replay_pool.expert_miss_mask, 1), vec![0]);
        let replay_host: Vec<f32> = residual_values
            .iter()
            .zip(&host)
            .map(|(&residual, &expert)| residual + expert * 0.25 + expert)
            .collect();
        let replay_authority: Vec<f64> = residual_values
            .iter()
            .zip(&authority)
            .map(|(&residual, &expert)| residual as f64 + expert * 0.25 + expert)
            .collect();
        let replay_device = read_f32(&replay_residual, HIDDEN);
        let replay_score = score_pair(
            &replay_host,
            &replay_device,
            &replay_authority,
            &Bounds::continuous_only(),
        );
        assert!(
            replay_score.pass,
            "heterogeneous replay wave failed V2.1: host={:?}, device={:?}",
            replay_score.host.failures, replay_score.device.failures
        );
        let invalid_table = unsafe {
            std::slice::from_raw_parts_mut(
                lease.table.contents() as *mut DeviceExpertTriplet,
                lease.n_experts,
            )
        };
        invalid_table[0].up.kind = 99;
        let invalid_mask = u32_buffer(&ctx, &[u32::MAX]);
        let sentinel: Vec<f32> = (0..INTERMEDIATE)
            .map(|index| 8_000.0 + index as f32)
            .collect();
        let invalid_gate = f32_buffer(&ctx, &sentinel);
        let mut invalid = TokenCommandBuffer::new(&ctx);
        encode_device_expert_table_validate(
            &mut invalid,
            &lease,
            &expert_indices,
            &execution_slots,
            &invalid_mask,
            1,
            HIDDEN,
            INTERMEDIATE,
            DEVICE_EXPERT_TENSOR_KIND_ANY_SUPPORTED,
        )
        .expect("validate unsupported mixed member");
        encode_device_expert_table_matvec(
            &mut invalid,
            DeviceExpertDispatchMode::Heterogeneous,
            &lease,
            &expert_indices,
            &execution_slots,
            &invalid_mask,
            1,
            0,
            0,
            &x,
            INTERMEDIATE,
            HIDDEN,
            &invalid_gate,
        )
        .expect("suppress mixed triplet after validation miss");
        invalid
            .commit_and_wait()
            .expect("invalid heterogeneous command");
        assert_eq!(read_u32(&invalid_mask, 1), vec![1]);
        assert_eq!(
            read_f32(&invalid_gate, INTERMEDIATE)
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            sentinel
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "an unsupported heterogeneous triplet must not mutate the destination"
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
                    false,
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
                    false,
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
                    false,
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
                false,
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
                false,
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
                false,
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
        assert_v21_gate(
            "device expert table complete wave",
            &host,
            &device,
            &authority,
        );
        let mut replay_arch = tiny_arch();
        replay_arch.hidden = HIDDEN;
        replay_arch.n_routed_experts = 4;
        replay_arch.num_experts_per_tok = 2;
        let replay_pool = ActPool::new(&ctx, &replay_arch).expect("replay activation pool");
        unsafe {
            std::ptr::copy_nonoverlapping(
                [2u32, 0].as_ptr(),
                replay_pool.expert_idx.contents() as *mut u32,
                2,
            );
            std::ptr::copy_nonoverlapping(
                [1u32, 0].as_ptr(),
                replay_pool.expert_exec_slots.contents() as *mut u32,
                2,
            );
            (replay_pool.expert_miss_mask.contents() as *mut u32).write(u32::MAX);
        }
        write_f32(&replay_pool.expert_w, &[0.3, 0.7]);
        let replay_scratch =
            ExpertWaveScratch::new(&ctx, 3, INTERMEDIATE, HIDDEN).expect("replay scratch");
        let replay_residual = f32_buffer(&ctx, &residual_values);
        let replay_graph = build_device_expert_replay_graph(
            &ctx,
            DeviceExpertDispatchMode::PqOnly,
            DeviceExpertDispatchMode::PqOnly,
            &routed_lease,
            &shared_lease,
            2,
            HIDDEN,
            INTERMEDIATE,
            &x,
            &replay_residual,
            &replay_pool,
            &replay_scratch,
        )
        .expect("complete-wave replay graph");
        assert_eq!(replay_graph.command_count(), 18);
        let replay_key_before = device_expert_replay_key(
            GENERATION,
            2,
            HIDDEN,
            INTERMEDIATE,
            DeviceExpertDispatchMode::PqOnly,
            DeviceExpertDispatchMode::PqOnly,
            &routed_lease,
            &shared_lease,
            &x,
            &replay_residual,
            &replay_pool,
            &replay_scratch,
        );
        let _ = ctx.drain_stats();
        let mut replay_hit = TokenCommandBuffer::new(&ctx);
        replay_hit
            .execute_replayable_graph(&replay_graph)
            .expect("execute complete-wave replay hit");
        assert_eq!(replay_hit.dispatch_count(), 18);
        replay_hit
            .commit_and_wait()
            .expect("complete-wave replay hit command");
        assert_eq!(read_u32(&replay_pool.expert_miss_mask, 1), vec![0]);
        let replay_device = read_f32(&replay_residual, HIDDEN);
        let replay_score = score_pair(
            &host,
            &replay_device,
            &authority,
            &Bounds::continuous_only(),
        );
        assert!(
            replay_score.pass,
            "replayed device expert complete wave failed V2.1: host={:?}, device={:?}",
            replay_score.host.failures, replay_score.device.failures
        );
        assert_eq!(
            replay_device
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            device
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "ICB and direct complete waves must be bit-identical"
        );
        let replay_outputs: Vec<&Buffer> = replay_scratch
            .gate
            .iter()
            .chain(&replay_scratch.up)
            .chain(&replay_scratch.act)
            .chain(&replay_scratch.down)
            .chain([&replay_scratch.combined, &replay_residual])
            .collect();
        for (buffer_index, buffer) in replay_outputs.iter().enumerate() {
            let elements = if buffer.length() >= (INTERMEDIATE * 4) as u64 {
                INTERMEDIATE
            } else {
                HIDDEN
            };
            let sentinel: Vec<f32> = (0..elements)
                .map(|index| 20_000.0 + (buffer_index * INTERMEDIATE + index) as f32)
                .collect();
            write_f32(buffer, &sentinel);
        }
        let replay_before: Vec<Vec<u32>> = replay_outputs
            .iter()
            .map(|buffer| {
                read_f32(buffer, HIDDEN)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect()
            })
            .collect();
        unsafe {
            std::ptr::copy_nonoverlapping(
                [2u32, 1].as_ptr(),
                replay_pool.expert_idx.contents() as *mut u32,
                2,
            );
            (replay_pool.expert_miss_mask.contents() as *mut u32).write(u32::MAX);
        }
        let mut replay_miss = TokenCommandBuffer::new(&ctx);
        replay_miss
            .execute_replayable_graph(&replay_graph)
            .expect("execute complete-wave replay miss");
        assert_eq!(replay_miss.dispatch_count(), 18);
        replay_miss
            .commit_and_wait()
            .expect("complete-wave replay miss command");
        assert_eq!(read_u32(&replay_pool.expert_miss_mask, 1), vec![1]);
        let replay_after: Vec<Vec<u32>> = replay_outputs
            .iter()
            .map(|buffer| {
                read_f32(buffer, HIDDEN)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect()
            })
            .collect();
        assert_eq!(
            replay_after, replay_before,
            "a replayed table miss must suppress every scratch and residual write"
        );
        assert!(
            device_expert_replay_key_matches(
                &replay_key_before,
                GENERATION,
                2,
                HIDDEN,
                INTERMEDIATE,
                DeviceExpertDispatchMode::PqOnly,
                DeviceExpertDispatchMode::PqOnly,
                &routed_lease,
                &shared_lease,
                &x,
                &replay_residual,
                &replay_pool,
                &replay_scratch
            ),
            "selection content changes must not invalidate stable-address replay"
        );
        assert_eq!(
            ctx.drain_stats(),
            (0, 0, 0),
            "replaying a captured expert graph must not allocate buffers"
        );
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
    fn device_expert_table_hit_costs_cover_routed_shared_and_elementwise_work() {
        use crate::cost_ledger::RoutedWeightRepresentation;
        use crate::gravity_glm::gpu::PqParams;
        crate::cost_ledger::set_enabled(true);
        let _ = crate::cost_ledger::end_token();
        assert!(crate::cost_ledger::begin_token());
        let r4 = |rows, cols, bytes| DeviceExpertProjectionMetrics::Pq {
            params: PqParams {
                dim: 32,
                subspaces: 1,
                sub: 32,
                card: 256,
                rows,
                cols,
                nchunk: cols / 32,
                bits: 8,
            },
            bytes,
            representation: RoutedWeightRepresentation::R4,
        };
        let triplet = [r4(64, 32, 100), r4(64, 32, 120), r4(32, 64, 80)];
        let metrics = DeviceExpertLayerMetrics {
            routed: vec![triplet, triplet],
            shared: triplet,
        };
        assert_eq!(metrics.dispatch_mode(), DeviceExpertDispatchMode::PqOnly);
        record_device_expert_table_hit_costs("model.layers.0.mlp", 32, 64, &metrics);
        let report = crate::cost_ledger::end_token().expect("table-hit cost report");
        crate::cost_ledger::set_enabled(false);
        assert_eq!(report.counters.matvec_calls, 9);
        assert_eq!(report.counters.active_bytes_read, 900);
        assert_eq!(
            report.counters.active_bytes_by_category["routed_experts"].as_u64(),
            Some(600)
        );
        assert_eq!(
            report.counters.active_bytes_by_category["shared_experts"].as_u64(),
            Some(300)
        );
        assert_eq!(
            report.counters.routed_representations.r4_projection_touches,
            6
        );
        assert_eq!(report.counters.routed_representations.r4_active_bytes, 600);
        assert_eq!(report.counters.dense_equivalent_fp_operations, 36_864);
        assert_eq!(report.counters.source_modelled_fp_operations, 52_736);
        assert_eq!(
            report
                .counters
                .source_modelled_integer_bitwise_ops_lower_bound,
            8_640
        );
        assert_eq!(report.counters.source_modelled_transcendentals, 192);
    }
    #[test]
    fn device_expert_table_heterogeneous_hit_costs_are_route_exact() {
        use crate::cost_ledger::RoutedWeightRepresentation;
        use crate::gravity_glm::gpu::PqParams;
        let pq = |params, bytes, representation| DeviceExpertProjectionMetrics::Pq {
            params,
            bytes,
            representation,
        };
        let triplet = [
            pq(
                PqParams {
                    dim: 32,
                    subspaces: 1,
                    sub: 32,
                    card: 256,
                    rows: 64,
                    cols: 32,
                    nchunk: 1,
                    bits: 8,
                },
                100,
                RoutedWeightRepresentation::R4,
            ),
            pq(
                PqParams {
                    dim: 8,
                    subspaces: 1,
                    sub: 8,
                    card: 128,
                    rows: 64,
                    cols: 32,
                    nchunk: 4,
                    bits: 7,
                },
                50,
                RoutedWeightRepresentation::R0,
            ),
            DeviceExpertProjectionMetrics::NativeBf16 {
                rows: 32,
                cols: 64,
                bytes: 4_096,
            },
        ];
        let metrics = DeviceExpertLayerMetrics {
            routed: vec![triplet, triplet],
            shared: triplet,
        };
        assert_eq!(
            metrics.dispatch_mode(),
            DeviceExpertDispatchMode::Heterogeneous
        );
        crate::cost_ledger::set_enabled(true);
        let _ = crate::cost_ledger::end_token();
        assert!(crate::cost_ledger::begin_token());
        record_device_expert_table_hit_costs("model.layers.0.mlp", 32, 64, &metrics);
        let report = crate::cost_ledger::end_token().expect("heterogeneous cost report");
        crate::cost_ledger::set_enabled(false);
        assert_eq!(report.counters.matvec_calls, 9);
        assert_eq!(report.counters.active_bytes_read, 12_738);
        assert_eq!(
            report.counters.active_bytes_by_category["routed_experts"].as_u64(),
            Some(8_492)
        );
        assert_eq!(
            report.counters.active_bytes_by_category["shared_experts"].as_u64(),
            Some(4_246)
        );
        let routed = &report.counters.routed_representations;
        assert_eq!(routed.r4_projection_touches, 2);
        assert_eq!(routed.r4_active_bytes, 200);
        assert_eq!(routed.r0_projection_touches, 2);
        assert_eq!(routed.r0_active_bytes, 100);
        assert_eq!(routed.native_bf16_projection_touches, 2);
        assert_eq!(routed.native_bf16_active_bytes, 8_192);
        assert_eq!(routed.other_projection_touches, 0);
        assert_eq!(routed.other_active_bytes, 0);
    }
    #[test]
    fn route_segment_parameter_abis_are_frozen_and_ranges_fail_closed() {
        assert_eq!(std::mem::size_of::<GlmRopeParams>(), 16);
        assert_eq!(std::mem::offset_of!(GlmRopeParams, n_heads), 0);
        assert_eq!(std::mem::offset_of!(GlmRopeParams, rotary_dim), 4);
        assert_eq!(std::mem::offset_of!(GlmRopeParams, in_stride), 8);
        assert_eq!(std::mem::offset_of!(GlmRopeParams, out_stride), 12);
        assert_eq!(std::mem::size_of::<GlmPositionedRopeParams>(), 24);
        assert_eq!(std::mem::offset_of!(GlmPositionedRopeParams, n_heads), 0);
        assert_eq!(
            std::mem::offset_of!(GlmPositionedRopeParams, input_element_offset),
            16
        );
        assert_eq!(
            std::mem::offset_of!(GlmPositionedRopeParams, output_element_offset),
            20
        );
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
    fn compact_absorbed_post_score_dag_passes_v21_without_readback() {
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
        let kv_params = crate::gravity_glm::gpu::PqParams {
            dim: LATENT as u32,
            subspaces: 1,
            sub: LATENT as u32,
            card: CARD as u32,
            rows: (HEADS * ROW_STRIDE) as u32,
            cols: LATENT as u32,
            nchunk: 1,
            bits: 8,
        };
        let o_params = crate::gravity_glm::gpu::PqParams {
            dim: CONTEXT as u32,
            subspaces: 1,
            sub: CONTEXT as u32,
            card: CARD as u32,
            rows: CONTEXT as u32,
            cols: CONTEXT as u32,
            nchunk: 1,
            bits: 8,
        };
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
            o_params,
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
        let replay_latent_cache = filled_f32_buffer(&ctx, TOKENS * LATENT, f32::NAN);
        let replay_rope_cache = filled_f32_buffer(&ctx, TOKENS * ROPE, f32::NAN);
        write_f32(&replay_latent_cache, &latents[..(TOKENS - 1) * LATENT]);
        write_f32(&replay_rope_cache, &rope_keys[..(TOKENS - 1) * ROPE]);
        let replay_query_latent = filled_f32_buffer(&ctx, HEADS * LATENT, f32::NAN);
        let replay_context = filled_f32_buffer(&ctx, CONTEXT, f32::NAN);
        let replay_hidden = filled_f32_buffer(&ctx, CONTEXT, f32::NAN);
        let replay_scores = f32_buffer(&ctx, &[0.0, 4.0, 1.0, 3.0, 5.0]);
        let replay_ranked = u32_buffer(&ctx, &vec![u32::MAX; ranked.len()]);
        let replay_residual = filled_f32_buffer(&ctx, CONTEXT, 0.0);
        let replay_inputs = CompactAttentionReplayInputs {
            layer: 0,
            hidden: CONTEXT,
            n_heads: HEADS,
            latent_dim: LATENT,
            rope_dim: ROPE,
            key_rows: NOPE,
            row_stride: ROW_STRIDE,
            value_rows: VALUE,
            max_allow: ranked.len(),
            scale,
            kv_params,
            o_params,
            k_latent: &current_latentb,
            key_rope: &current_ropeb,
            latent_cache: &replay_latent_cache,
            rope_cache: &replay_rope_cache,
            kv_codebooks: &kv_codebookb,
            kv_codes: &kv_codesb,
            query_nope: &query_nopeb,
            query_latent: &replay_query_latent,
            query_rope: &query_ropeb,
            scores: Some(&replay_scores),
            ranked_indices: &replay_ranked,
            context: &replay_context,
            o_codebooks: &o_codebookb,
            o_codes: &o_codesb,
            output: &replay_hidden,
            residual: Some(&replay_residual),
        };
        let replay = build_compact_attention_replay_graph(
            &ctx,
            &replay_inputs,
            TOKENS - 1,
            TOKENS,
            ranked.len(),
        )
        .expect("capture seven-dispatch post-score graph");
        assert_eq!(replay.graph.command_count(), 7);
        let direct_hidden_bits: Vec<u32> = device.iter().map(|value| value.to_bits()).collect();
        let _ = ctx.drain_stats();
        for iteration in 0..2 {
            write_f32(&replay_query_latent, &vec![f32::NAN; HEADS * LATENT]);
            write_f32(&replay_context, &vec![f32::NAN; CONTEXT]);
            write_f32(&replay_hidden, &vec![f32::NAN; CONTEXT]);
            write_f32(&replay_residual, &vec![0.0; CONTEXT]);
            replay
                .update_dynamic_parameters(TOKENS - 1, TOKENS, ranked.len())
                .expect("update compact-attention replay scalars");
            let mut replay_tcb = TokenCommandBuffer::new(&ctx);
            replay_tcb
                .execute_replayable_graph(&replay.graph)
                .expect("execute compact-attention replay");
            assert_eq!(replay_tcb.dispatch_count(), 7);
            replay_tcb
                .commit_and_wait()
                .expect("compact-attention replay command");
            assert_eq!(
                read_f32(&replay_hidden, CONTEXT)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                direct_hidden_bits,
                "compact-attention ICB replay {iteration} must be bit-exact to direct encoding"
            );
            assert_eq!(
                read_u32(&replay_ranked, ranked.len()),
                ranked,
                "post-score replay {iteration} must preserve exact radix order"
            );
            assert_eq!(
                read_f32(&replay_residual, CONTEXT)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                direct_hidden_bits,
                "post-score replay {iteration} residual must consume the exact o_proj output"
            );
        }
        assert_eq!(
            ctx.drain_stats(),
            (0, 0, 0),
            "warm compact-attention graph replays must not allocate buffers"
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
                score = 7.0;
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
            91.0, 92.0, 1.0, 2.0, 3.0, 4.0, 81.0, 82.0, -1.5, 0.5, 2.5, -3.0,
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
        assert!(shader.contains("kernel void gravity_rope_prefix_tail_positioned_f32("));
        let registry = include_str!("metal/mod.rs");
        assert!(registry
            .contains("\"gravity_rope_prefix_tail_f32\" => \"gravity_rope_prefix_tail_f32\""));
        assert!(registry.contains("\"gravity_rope_prefix_tail_positioned_f32\""));
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 91.0, 92.0, -1.5, 0.5, 2.5, -3.0, 81.0, 82.0,
        ];
        let cos = vec![0.875, -0.25];
        let sin = vec![0.125, 0.75];
        let input_buffer = f32_buffer(&ctx, &input);
        let cos_buffer = f32_buffer(&ctx, &cos);
        let sin_buffer = f32_buffer(&ctx, &sin);
        let output = filled_f32_buffer(&ctx, 14, -99.0);
        let positioned_output = filled_f32_buffer(&ctx, 14, -99.0);
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
        encode_rope_prefix_tail_positioned(
            &mut tcb,
            &input_buffer,
            0,
            &positioned_output,
            1,
            &cos_buffer,
            &sin_buffer,
            2,
            4,
            6,
            6,
        )
        .expect("encode replay-safe positioned RoPE prefix plus tail");
        assert_eq!(tcb.dispatch_count(), 2);
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
        let positioned_actual = read_f32(&positioned_output, expected.len());
        assert_eq!(
            positioned_actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "position-scalar and binding-offset RoPE paths must be bit-identical"
        );
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
    fn device_dsa_pre_score_icb_replays_six_fixed_grid_commands_bit_exact() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const HEADS: usize = 2;
        const HEAD_DIM: usize = 4;
        const ROPE_DIM: usize = 2;
        const Q_LORA: usize = 3;
        const HIDDEN: usize = 3;
        const CAPACITY: usize = 3;
        const POSITION: usize = 1;
        let projection_from = |tensor: GpuTensor| match tensor {
            GpuTensor::NativeGpuBf16 { buf, rows, cols } => DeviceReplayProjection::NativeBf16 {
                weight: buf,
                rows,
                cols,
            },
            _ => panic!("native replay projection fixture"),
        };
        let projections = [
            projection_from(native_bf16_tensor(&ctx, HEADS * HEAD_DIM, Q_LORA, 11).0),
            projection_from(native_bf16_tensor(&ctx, HEAD_DIM, HIDDEN, 17).0),
            projection_from(native_bf16_tensor(&ctx, HEADS, HIDDEN, 23).0),
        ];
        let q_resid = f32_buffer(&ctx, &[0.5, -1.25, 2.0]);
        let h = f32_buffer(&ctx, &[1.5, 0.25, -0.75]);
        let norm_weight = f32_buffer(&ctx, &[1.0, 0.75, 1.25, 0.5]);
        let norm_bias = f32_buffer(&ctx, &[0.125, -0.25, 0.5, -0.75]);
        let cos = f32_buffer(&ctx, &[0.875]);
        let sin = f32_buffer(&ctx, &[0.125]);
        let direct_q = filled_f32_buffer(&ctx, HEADS * HEAD_DIM, f32::NAN);
        let direct_k = filled_f32_buffer(&ctx, HEAD_DIM, f32::NAN);
        let direct_head_w = filled_f32_buffer(&ctx, HEADS, f32::NAN);
        let direct_query = filled_f32_buffer(&ctx, HEADS * HEAD_DIM, f32::NAN);
        let direct_index_keys = filled_f32_buffer(&ctx, CAPACITY * HEAD_DIM, -99.0);
        let mut direct = TokenCommandBuffer::new(&ctx);
        for (projection, input, output) in [
            (&projections[0], &q_resid, &direct_q),
            (&projections[1], &h, &direct_k),
            (&projections[2], &h, &direct_head_w),
        ] {
            let DeviceReplayProjection::NativeBf16 { weight, rows, cols } = projection else {
                unreachable!("native fixture");
            };
            encode_gemv_native_bf16_seq(&mut direct, weight, *rows, *cols, input, output)
                .expect("encode direct pre-score projection");
        }
        encode_layernorm_affine(
            &mut direct,
            &direct_k,
            &norm_weight,
            &norm_bias,
            &direct_k,
            HEAD_DIM,
            1e-6,
        )
        .expect("encode direct index-key norm");
        encode_rope_prefix_tail_positioned(
            &mut direct,
            &direct_k,
            0,
            &direct_index_keys,
            POSITION * HEAD_DIM,
            &cos,
            &sin,
            1,
            ROPE_DIM,
            HEAD_DIM,
            HEAD_DIM,
        )
        .expect("encode direct positioned key RoPE");
        encode_rope_prefix_tail(
            &mut direct,
            &direct_q,
            0,
            &direct_query,
            0,
            &cos,
            &sin,
            HEADS,
            ROPE_DIM,
            HEAD_DIM,
            HEAD_DIM,
        )
        .expect("encode direct query RoPE");
        assert_eq!(direct.dispatch_count(), 6);
        direct.commit_and_wait().expect("direct pre-score graph");
        let replay_q = filled_f32_buffer(&ctx, HEADS * HEAD_DIM, f32::NAN);
        let replay_k = filled_f32_buffer(&ctx, HEAD_DIM, f32::NAN);
        let replay_head_w = filled_f32_buffer(&ctx, HEADS, f32::NAN);
        let replay_query = filled_f32_buffer(&ctx, HEADS * HEAD_DIM, f32::NAN);
        let replay_index_keys = filled_f32_buffer(&ctx, CAPACITY * HEAD_DIM, -99.0);
        let replay_inputs = DeviceDsaPreScoreReplayInputs {
            layer: 0,
            n_heads: HEADS,
            head_dim: HEAD_DIM,
            rope_dim: ROPE_DIM,
            norm_eps: 1e-6,
            projections: &projections,
            q_resid: &q_resid,
            h: &h,
            idx_q: &replay_q,
            idx_k_raw: &replay_k,
            idx_head_w: &replay_head_w,
            norm_weight: &norm_weight,
            norm_bias: &norm_bias,
            cos: &cos,
            sin: &sin,
            query: &replay_query,
            index_keys: &replay_index_keys,
        };
        let replay = build_device_dsa_pre_score_replay_graph(&ctx, &replay_inputs, POSITION)
            .expect("capture six-command pre-score graph");
        assert_eq!(replay.graph.command_count(), 6);
        let _ = ctx.drain_stats();
        replay
            .update_position(POSITION)
            .expect("update first replay position");
        let mut replay_tcb = TokenCommandBuffer::new(&ctx);
        replay_tcb
            .execute_replayable_graph(&replay.graph)
            .expect("execute pre-score replay");
        assert_eq!(replay_tcb.dispatch_count(), 6);
        replay_tcb.commit_and_wait().expect("pre-score replay");
        for (label, expected, actual, len) in [
            ("index query", &direct_q, &replay_q, HEADS * HEAD_DIM),
            ("index key", &direct_k, &replay_k, HEAD_DIM),
            ("head weights", &direct_head_w, &replay_head_w, HEADS),
            (
                "rotated query",
                &direct_query,
                &replay_query,
                HEADS * HEAD_DIM,
            ),
            (
                "index-key cache",
                &direct_index_keys,
                &replay_index_keys,
                CAPACITY * HEAD_DIM,
            ),
        ] {
            assert_eq!(
                read_f32(actual, len)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                read_f32(expected, len)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{label} must be bit-exact between direct and replay encoding"
            );
        }
        replay
            .update_position(POSITION + 1)
            .expect("update second replay position");
        let mut second = TokenCommandBuffer::new(&ctx);
        second
            .execute_replayable_graph(&replay.graph)
            .expect("execute changed-position pre-score replay");
        second
            .commit_and_wait()
            .expect("changed-position pre-score replay");
        assert_eq!(
            read_f32(&replay_index_keys, CAPACITY * HEAD_DIM)
                [(POSITION + 1) * HEAD_DIM..(POSITION + 2) * HEAD_DIM],
            read_f32(&replay_index_keys, CAPACITY * HEAD_DIM)
                [POSITION * HEAD_DIM..(POSITION + 1) * HEAD_DIM],
            "dynamic position scalar must retarget the stable index-key buffer"
        );
        assert_eq!(
            ctx.drain_stats(),
            (0, 0, 0),
            "warm pre-score graph replays must not allocate buffers"
        );
    }
    #[test]
    fn attention_prelude_icb_replays_nine_fixed_grid_commands_bit_exact() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const HIDDEN: usize = 4;
        const Q_LORA: usize = 3;
        const KV_LORA: usize = 3;
        const HEADS: usize = 2;
        const NOPE: usize = 2;
        const ROPE: usize = 2;
        const QK: usize = NOPE + ROPE;
        let projection_from = |tensor: GpuTensor| match tensor {
            GpuTensor::NativeGpuBf16 { buf, rows, cols } => DeviceReplayProjection::NativeBf16 {
                weight: buf,
                rows,
                cols,
            },
            _ => panic!("native replay projection fixture"),
        };
        let projections = [
            projection_from(native_bf16_tensor(&ctx, Q_LORA, HIDDEN, 31).0),
            projection_from(native_bf16_tensor(&ctx, KV_LORA + ROPE, HIDDEN, 37).0),
            projection_from(native_bf16_tensor(&ctx, HEADS * QK, Q_LORA, 41).0),
        ];
        let input_norm_weight = f32_buffer(&ctx, &[1.0, 0.75, 1.25, 0.5]);
        let q_norm_weight = f32_buffer(&ctx, &[0.625, 1.375, 0.875]);
        let kv_norm_weight = f32_buffer(&ctx, &[1.125, 0.5, 1.5]);
        let cos = f32_buffer(&ctx, &[0.875]);
        let sin = f32_buffer(&ctx, &[0.125]);
        let direct_x = filled_f32_buffer(&ctx, HIDDEN, f32::NAN);
        let direct_h = filled_f32_buffer(&ctx, HIDDEN, f32::NAN);
        let direct_q_a = filled_f32_buffer(&ctx, Q_LORA, f32::NAN);
        let direct_compressed = filled_f32_buffer(&ctx, KV_LORA + ROPE, f32::NAN);
        let direct_q_resid = filled_f32_buffer(&ctx, Q_LORA, f32::NAN);
        let direct_k_latent = filled_f32_buffer(&ctx, KV_LORA, f32::NAN);
        let direct_q = filled_f32_buffer(&ctx, HEADS * QK, f32::NAN);
        let direct_key_rope = filled_f32_buffer(&ctx, ROPE, f32::NAN);
        let direct_query_nope = filled_f32_buffer(&ctx, HEADS * NOPE, f32::NAN);
        let direct_query_rope = filled_f32_buffer(&ctx, HEADS * ROPE, f32::NAN);
        let replay_x = filled_f32_buffer(&ctx, HIDDEN, f32::NAN);
        let replay_h = filled_f32_buffer(&ctx, HIDDEN, f32::NAN);
        let replay_q_a = filled_f32_buffer(&ctx, Q_LORA, f32::NAN);
        let replay_compressed = filled_f32_buffer(&ctx, KV_LORA + ROPE, f32::NAN);
        let replay_q_resid = filled_f32_buffer(&ctx, Q_LORA, f32::NAN);
        let replay_k_latent = filled_f32_buffer(&ctx, KV_LORA, f32::NAN);
        let replay_q = filled_f32_buffer(&ctx, HEADS * QK, f32::NAN);
        let replay_key_rope = filled_f32_buffer(&ctx, ROPE, f32::NAN);
        let replay_query_nope = filled_f32_buffer(&ctx, HEADS * NOPE, f32::NAN);
        let replay_query_rope = filled_f32_buffer(&ctx, HEADS * ROPE, f32::NAN);
        let replay_inputs = AttentionPreludeReplayInputs {
            layer: 0,
            hidden: HIDDEN,
            q_lora_rank: Q_LORA,
            kv_lora_rank: KV_LORA,
            n_heads: HEADS,
            qk_nope_dim: NOPE,
            rope_dim: ROPE,
            rms_norm_eps: 1e-6,
            projections: &projections,
            x: &replay_x,
            h: &replay_h,
            q_a: &replay_q_a,
            compressed: &replay_compressed,
            q_resid: &replay_q_resid,
            k_latent: &replay_k_latent,
            q: &replay_q,
            input_norm_weight: &input_norm_weight,
            q_norm_weight: &q_norm_weight,
            kv_norm_weight: &kv_norm_weight,
            cos: &cos,
            sin: &sin,
            key_rope: &replay_key_rope,
            query_nope: &replay_query_nope,
            query_rope: &replay_query_rope,
        };
        let replay = build_attention_prelude_replay_graph(&ctx, &replay_inputs)
            .expect("capture nine-command attention prelude");
        assert_eq!(replay.graph.command_count(), 9);
        let snapshots = |buffers: [(&Buffer, usize); 9]| {
            buffers
                .into_iter()
                .map(|(buffer, len)| {
                    read_f32(buffer, len)
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let _ = ctx.drain_stats();
        for x_values in [[0.5f32, -1.25, 2.0, 0.75], [1.5f32, 0.25, -0.75, 2.25]] {
            write_f32(&direct_x, &x_values);
            let mut direct = TokenCommandBuffer::new(&ctx);
            encode_rmsnorm(
                &mut direct,
                &direct_x,
                &input_norm_weight,
                &direct_h,
                HIDDEN,
                1e-6,
            )
            .expect("encode direct input norm");
            for (projection, input, output) in [
                (&projections[0], &direct_h, &direct_q_a),
                (&projections[1], &direct_h, &direct_compressed),
            ] {
                let DeviceReplayProjection::NativeBf16 { weight, rows, cols } = projection else {
                    unreachable!("native fixture");
                };
                encode_gemv_native_bf16_seq(&mut direct, weight, *rows, *cols, input, output)
                    .expect("encode direct prelude projection");
            }
            encode_rmsnorm(
                &mut direct,
                &direct_q_a,
                &q_norm_weight,
                &direct_q_resid,
                Q_LORA,
                1e-6,
            )
            .expect("encode direct q norm");
            encode_rmsnorm(
                &mut direct,
                &direct_compressed,
                &kv_norm_weight,
                &direct_k_latent,
                KV_LORA,
                1e-6,
            )
            .expect("encode direct kv norm");
            encode_rope_interleaved(
                &mut direct,
                &direct_compressed,
                KV_LORA,
                &direct_key_rope,
                0,
                &cos,
                &sin,
                1,
                ROPE,
                ROPE,
                ROPE,
            )
            .expect("encode direct key RoPE");
            let DeviceReplayProjection::NativeBf16 { weight, rows, cols } = &projections[2] else {
                unreachable!("native fixture");
            };
            encode_gemv_native_bf16_seq(
                &mut direct,
                weight,
                *rows,
                *cols,
                &direct_q_resid,
                &direct_q,
            )
            .expect("encode direct q_b");
            encode_copy_head_prefix(
                &mut direct,
                &direct_q,
                &direct_query_nope,
                HEADS,
                NOPE,
                ROPE,
            )
            .expect("encode direct query prefix");
            encode_rope_interleaved(
                &mut direct,
                &direct_q,
                NOPE,
                &direct_query_rope,
                0,
                &cos,
                &sin,
                HEADS,
                ROPE,
                QK,
                ROPE,
            )
            .expect("encode direct query RoPE");
            assert_eq!(direct.dispatch_count(), 9);
            direct.commit_and_wait().expect("direct attention prelude");
            let expected = snapshots([
                (&direct_h, HIDDEN),
                (&direct_q_a, Q_LORA),
                (&direct_compressed, KV_LORA + ROPE),
                (&direct_q_resid, Q_LORA),
                (&direct_k_latent, KV_LORA),
                (&direct_q, HEADS * QK),
                (&direct_key_rope, ROPE),
                (&direct_query_nope, HEADS * NOPE),
                (&direct_query_rope, HEADS * ROPE),
            ]);
            write_f32(&replay_x, &x_values);
            let mut replay_tcb = TokenCommandBuffer::new(&ctx);
            replay_tcb
                .execute_replayable_graph(&replay.graph)
                .expect("execute attention prelude replay");
            assert_eq!(replay_tcb.dispatch_count(), 9);
            replay_tcb
                .commit_and_wait()
                .expect("attention prelude replay");
            let actual = snapshots([
                (&replay_h, HIDDEN),
                (&replay_q_a, Q_LORA),
                (&replay_compressed, KV_LORA + ROPE),
                (&replay_q_resid, Q_LORA),
                (&replay_k_latent, KV_LORA),
                (&replay_q, HEADS * QK),
                (&replay_key_rope, ROPE),
                (&replay_query_nope, HEADS * NOPE),
                (&replay_query_rope, HEADS * ROPE),
            ]);
            assert_eq!(
                actual, expected,
                "nine-command replay must be bit-exact to direct encoding"
            );
        }
        assert_eq!(
            ctx.drain_stats(),
            (0, 0, 0),
            "warm attention prelude replays must not allocate buffers"
        );
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
        let expert_trace = u32_buffer(&ctx, &[u32::MAX; 6]);
        let mut trace_tcb = TokenCommandBuffer::new(&ctx);
        encode_device_expert_trace_copy(&mut trace_tcb, &tie_indices, &expert_trace, 2, 3)
            .expect("encode deferred expert trace");
        trace_tcb
            .commit_and_wait()
            .expect("deferred expert trace command");
        assert_eq!(
            read_u32(&expert_trace, 6),
            vec![u32::MAX, u32::MAX, u32::MAX, 0, 1, u32::MAX]
        );
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
        assert_eq!(
            pool.expert_trace.length(),
            (arch.n_layers.max(1) * arch.num_experts_per_tok.max(1) * std::mem::size_of::<u32>())
                as u64
        );
        assert_eq!(
            pool.expert_miss_mask.length(),
            std::mem::size_of::<u32>() as u64
        );
        assert_eq!(read_u32(&pool.shared_expert_idx, 1), vec![0]);
        assert_eq!(read_u32(&pool.shared_expert_slot, 1), vec![0]);
        {
            let layers = pool
                .persistent_expert_layers
                .lock()
                .expect("persistent expert layers");
            assert_eq!(layers.len(), arch.n_layers);
            assert!(
                layers.iter().all(Option::is_none),
                "default path must not build or lease an expert descriptor table"
            );
        }
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
    fn native_final_head_icb_replays_fixed_graph_bit_exact_without_allocations() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let arch = tiny_arch();
        let pool = ActPool::new(&ctx, &arch).expect("final-head activation pool");
        let norm_weight = [1.0f32, 0.75, 1.25, 0.5];
        write_f32(&pool.final_norm_weight, &norm_weight);
        let weights: Vec<f32> = (0..arch.vocab_size)
            .flat_map(|row| {
                (0..arch.hidden)
                    .map(move |col| ((row + 1) as f32 * 0.0625) - (col as f32 * 0.03125))
            })
            .collect();
        let bf16_bits: Vec<u16> = weights
            .iter()
            .map(|value| half::bf16::from_f32(*value).to_bits())
            .collect();
        let weight = ctx
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&bf16_bits))
            .expect("native bf16 final-head weight");
        let head = DeviceHead::NativeBf16 {
            weight,
            rows: arch.vocab_size as u32,
            cols: arch.hidden as u32,
        };
        let run_direct = |x: &[f32]| {
            write_f32(&pool.x, x);
            let mut tcb = TokenCommandBuffer::new(&ctx);
            encode_rmsnorm(
                &mut tcb,
                &pool.x,
                &pool.final_norm_weight,
                &pool.final_hidden,
                arch.hidden,
                arch.rms_norm_eps,
            )
            .expect("encode direct final norm");
            let DeviceHead::NativeBf16 { weight, rows, cols } = &head else {
                unreachable!("native fixture");
            };
            encode_gemv_native_bf16_seq(
                &mut tcb,
                weight,
                *rows,
                *cols,
                &pool.final_hidden,
                &pool.logits,
            )
            .expect("encode direct native head");
            encode_argmax_f32(&mut tcb, &pool.logits, *rows, &pool.sample_token)
                .expect("encode direct argmax");
            encode_sample_topk_f32(
                &mut tcb,
                &pool.logits,
                *rows,
                GPU_LM_HEAD_DIAG_TOPK,
                &pool.head_topk_idx,
                &pool.head_topk_val,
            )
            .expect("encode direct top-k");
            tcb.commit_and_wait().expect("direct final-head graph");
            (
                read_f32(&pool.final_hidden, arch.hidden)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                read_f32(&pool.logits, arch.vocab_size)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                read_u32(&pool.sample_token, 1),
                read_u32(&pool.head_topk_idx, GPU_LM_HEAD_DIAG_TOPK as usize),
                read_f32(&pool.head_topk_val, GPU_LM_HEAD_DIAG_TOPK as usize)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
            )
        };
        let first_x = [0.5f32, -1.0, 1.5, 0.25];
        let direct_first = run_direct(&first_x);
        let graph =
            build_final_head_replay_graph(&ctx, &head, &pool, arch.hidden, arch.rms_norm_eps)
                .expect("capture native final-head graph");
        assert_eq!(graph.command_count(), 4);
        let key_before = final_head_replay_key(&head, &pool, arch.hidden, arch.rms_norm_eps);
        let _ = ctx.drain_stats();
        for (x, direct) in [
            (first_x.as_slice(), direct_first),
            (
                [1.25f32, -0.5, 0.125, 2.0].as_slice(),
                run_direct(&[1.25, -0.5, 0.125, 2.0]),
            ),
        ] {
            write_f32(&pool.x, x);
            write_f32(&pool.final_hidden, &vec![f32::NAN; arch.hidden]);
            write_f32(&pool.logits, &vec![f32::NAN; arch.vocab_size]);
            let mut replay = TokenCommandBuffer::new(&ctx);
            replay
                .execute_replayable_graph(&graph)
                .expect("execute native final-head replay");
            assert_eq!(replay.dispatch_count(), 4);
            replay
                .commit_and_wait()
                .expect("native final-head replay command");
            let actual = (
                read_f32(&pool.final_hidden, arch.hidden)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                read_f32(&pool.logits, arch.vocab_size)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                read_u32(&pool.sample_token, 1),
                read_u32(&pool.head_topk_idx, GPU_LM_HEAD_DIAG_TOPK as usize),
                read_f32(&pool.head_topk_val, GPU_LM_HEAD_DIAG_TOPK as usize)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                actual, direct,
                "ICB and direct native final-head graphs must be bit-identical"
            );
        }
        assert_eq!(
            final_head_replay_key(&head, &pool, arch.hidden, arch.rms_norm_eps),
            key_before,
            "activation content changes must preserve the stable-address replay key"
        );
        assert_eq!(
            ctx.drain_stats(),
            (0, 0, 0),
            "warm final-head graph replays must not allocate buffers"
        );
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
                true
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
                .lock().expect("device attention prelude scratch")
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
