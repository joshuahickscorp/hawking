// Qwen30 device-indexed expert address table.
//
// Host populates one Gravity-style per-layer table of 128 expert triplets
// (gate/up/down) with Metal gpuAddress values. Device route_ids select which
// expert each of the top-k routes executes — without a host readback.
//
// Layout is frozen against the Rust structs in qwen30_complete_runtime.rs.
// Every indirectly referenced buffer must also be declared via useResources.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint QWEN30_EXPERT_KIND_BINARY = 1u;
constant constexpr uint QWEN30_EXPERT_KIND_HGRAVS = 2u;
constant constexpr uint QWEN30_EXPERT_TRIPLET_READY = 7u;

struct Qwen30DeviceExpertTensorRef {
    const device uchar *primary;
    const device uchar *secondary;
    uint rows;
    uint cols;
    uint rank;
    uint kind;
    uint generation;
    uint pad;
};

struct Qwen30DeviceExpertTriplet {
    Qwen30DeviceExpertTensorRef gate;
    Qwen30DeviceExpertTensorRef up;
    Qwen30DeviceExpertTensorRef down;
    uint ready_mask;
    uint generation;
};

static_assert(sizeof(Qwen30DeviceExpertTensorRef) == 40,
              "Qwen30DeviceExpertTensorRef ABI drift");
static_assert(sizeof(Qwen30DeviceExpertTriplet) == 128,
              "Qwen30DeviceExpertTriplet ABI drift");

struct Qwen30DeviceExpertMatvecParams {
    uint n_experts;
    uint experts_per_token;
    uint generation;
    uint execution_position;
    uint projection; // 0=gate, 1=up, 2=down
    uint group_size;
    uint input_offset_elems;
    uint output_offset_elems;
};

static_assert(sizeof(Qwen30DeviceExpertMatvecParams) == 32,
              "Qwen30DeviceExpertMatvecParams ABI drift");

struct Qwen30DeviceExpertPairedParams {
    uint n_experts;
    uint experts_per_token;
    uint generation;
    uint execution_position;
    uint group_size;
    uint output_offset_elems;
    uint pad0;
    uint pad1;
};

static_assert(sizeof(Qwen30DeviceExpertPairedParams) == 32,
              "Qwen30DeviceExpertPairedParams ABI drift");

struct Qwen30DeviceExpertHgravsParams {
    uint n_experts;
    uint experts_per_token;
    uint generation;
    uint execution_position;
    uint projection; // 0=gate, 1=up, 2=down
    uint stage;     // 0 = R@x -> mid, 1 = L@mid -> y
    uint input_offset_elems;
    uint output_offset_elems;
};

static_assert(sizeof(Qwen30DeviceExpertHgravsParams) == 32,
              "Qwen30DeviceExpertHgravsParams ABI drift");

static inline const device Qwen30DeviceExpertTensorRef *
qwen30_select_projection(const device Qwen30DeviceExpertTriplet &entry, uint projection)
{
    if (projection == 0u) {
        return &entry.gate;
    }
    if (projection == 1u) {
        return &entry.up;
    }
    return &entry.down;
}

// Scalar-control binary sign/scale matvec indexed by device route_ids.
// One thread owns one output row; accumulation matches qwen_binary_sign_scale_matvec.
kernel void qwen30_expert_table_binary_matvec(
    const device uint *route_ids [[buffer(0)]],
    const device Qwen30DeviceExpertTriplet *table [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Qwen30DeviceExpertMatvecParams &p [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (p.execution_position >= p.experts_per_token) {
        return;
    }
    uint expert = route_ids[p.execution_position];
    if (expert >= p.n_experts) {
        return;
    }
    const device Qwen30DeviceExpertTriplet &entry = table[expert];
    const device Qwen30DeviceExpertTensorRef *tensor =
        qwen30_select_projection(entry, p.projection);
    if (entry.ready_mask != QWEN30_EXPERT_TRIPLET_READY ||
        entry.generation != p.generation ||
        tensor->generation != p.generation ||
        tensor->kind != QWEN30_EXPERT_KIND_BINARY ||
        tensor->primary == nullptr ||
        tensor->secondary == nullptr ||
        tensor->rows == 0u ||
        tensor->cols == 0u ||
        p.group_size == 0u) {
        return;
    }
    if (row >= tensor->rows) {
        return;
    }

    const device uchar *signs = tensor->primary;
    const device half *scales =
        reinterpret_cast<const device half *>(tensor->secondary);
    const uint groups_per_row = (tensor->cols + p.group_size - 1u) / p.group_size;
    const device float *x = input + p.input_offset_elems;
    device float *y = output + p.output_offset_elems;

    float sum = 0.0f;
    const uint row_base = row * tensor->cols;
    const uint scale_base = row * groups_per_row;
    for (uint group = 0u; group < groups_per_row; ++group) {
        const uint group_start = group * p.group_size;
        const uint group_end = min(group_start + p.group_size, tensor->cols);
        const float scale = float(scales[scale_base + group]);
        for (uint col = group_start; col < group_end; ++col) {
            const uint flat = row_base + col;
            const uchar byte = signs[flat >> 3u];
            const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
            sum += (positive ? scale : -scale) * x[col];
        }
    }
    y[row] = sum;
}

// Simdgroup candidate geometry: same products as scalar, simd_sum reduction.
// Grid: (ceil(rows / 8) * 256, 1, 1), threadgroup (256, 1, 1).
kernel void qwen30_expert_table_binary_matvec_simdgroup(
    const device uint *route_ids [[buffer(0)]],
    const device Qwen30DeviceExpertTriplet *table [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Qwen30DeviceExpertMatvecParams &p [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (p.execution_position >= p.experts_per_token) {
        return;
    }
    uint expert = route_ids[p.execution_position];
    if (expert >= p.n_experts) {
        return;
    }
    const device Qwen30DeviceExpertTriplet &entry = table[expert];
    const device Qwen30DeviceExpertTensorRef *tensor =
        qwen30_select_projection(entry, p.projection);
    if (entry.ready_mask != QWEN30_EXPERT_TRIPLET_READY ||
        entry.generation != p.generation ||
        tensor->generation != p.generation ||
        tensor->kind != QWEN30_EXPERT_KIND_BINARY ||
        tensor->primary == nullptr ||
        tensor->secondary == nullptr ||
        tensor->rows == 0u ||
        tensor->cols == 0u ||
        p.group_size == 0u) {
        return;
    }
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= tensor->rows) {
        return;
    }

    const device uchar *signs = tensor->primary;
    const device half *scales =
        reinterpret_cast<const device half *>(tensor->secondary);
    const device float *x = input + p.input_offset_elems;
    device float *y = output + p.output_offset_elems;

    float sum = 0.0f;
    const uint row_base = row * tensor->cols;
    const uint scale_base = row * ((tensor->cols + p.group_size - 1u) / p.group_size);
    for (uint col = simd_lane; col < tensor->cols; col += 32u) {
        const uint flat = row_base + col;
        const float scale = float(scales[scale_base + col / p.group_size]);
        const uchar byte = signs[flat >> 3u];
        const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
        sum += (positive ? scale : -scale) * x[col];
    }
    sum = simd_sum(sum);
    if (simd_lane == 0u) {
        y[row] = sum;
    }
}

// Paired gate/up SwiGLU with scalar-order arithmetic, table-indexed.
// Matches qwen_direct_packed_gate_up_swiglu_paired_scalar_order_candidate.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline float qwen30_table_paired_scalar_order_value(
    const device uchar *signs,
    const device half *scales,
    uint element,
    uint group_size)
{
    const uint group = element / group_size;
    const uint bit = element % group_size;
    const uchar packed = signs[group * (group_size / 8u) + bit / 8u];
    const bool positive = ((packed >> (bit & 7u)) & 1u) != 0u;
    const float scale = float(scales[group]);
    return positive ? scale : -scale;
}

kernel void qwen30_expert_table_paired_gate_up_swiglu(
    const device uint *route_ids [[buffer(0)]],
    const device Qwen30DeviceExpertTriplet *table [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device float *activation [[buffer(3)]],
    constant Qwen30DeviceExpertPairedParams &p [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (p.execution_position >= p.experts_per_token) {
        return;
    }
    uint expert = route_ids[p.execution_position];
    if (expert >= p.n_experts) {
        return;
    }
    const device Qwen30DeviceExpertTriplet &entry = table[expert];
    if (entry.ready_mask != QWEN30_EXPERT_TRIPLET_READY ||
        entry.generation != p.generation ||
        entry.gate.kind != QWEN30_EXPERT_KIND_BINARY ||
        entry.up.kind != QWEN30_EXPERT_KIND_BINARY ||
        entry.gate.primary == nullptr ||
        entry.gate.secondary == nullptr ||
        entry.up.primary == nullptr ||
        entry.up.secondary == nullptr ||
        entry.gate.rows == 0u ||
        entry.gate.cols == 0u ||
        entry.gate.rows != entry.up.rows ||
        entry.gate.cols != entry.up.cols ||
        p.group_size == 0u) {
        return;
    }
    if (row >= entry.gate.rows) {
        return;
    }

    const device uchar *gate_signs = entry.gate.primary;
    const device half *gate_scales =
        reinterpret_cast<const device half *>(entry.gate.secondary);
    const device uchar *up_signs = entry.up.primary;
    const device half *up_scales =
        reinterpret_cast<const device half *>(entry.up.secondary);
    device float *out = activation + p.output_offset_elems;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint row_base = row * entry.gate.cols;
    for (uint col = 0u; col < entry.gate.cols; ++col) {
        const float x = input[col];
        const float gate_weight = qwen30_table_paired_scalar_order_value(
            gate_signs, gate_scales, row_base + col, p.group_size);
        const float up_weight = qwen30_table_paired_scalar_order_value(
            up_signs, up_scales, row_base + col, p.group_size);
        const float gate_product = gate_weight * x;
        const float up_product = up_weight * x;
        gate_sum = gate_sum + gate_product;
        up_sum = up_sum + up_product;
    }
    out[row] = (gate_sum / (1.0f + exp(-gate_sum))) * up_sum;
}

#pragma clang fp reassociate(on)
#pragma clang fp contract(on)

// HGRAVS01 stage gemv via device pointers.
// stage 0: mid = R @ x  (weight = secondary, rows=rank, cols=cols)
// stage 1: y   = L @ mid (weight = primary, rows=rows, cols=rank)
// Grid: (rows * 256, 1, 1), threadgroup (256, 1, 1) — matches gemv_f32_moe.
kernel void qwen30_expert_table_hgravs_gemv(
    const device uint *route_ids [[buffer(0)]],
    const device Qwen30DeviceExpertTriplet *table [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Qwen30DeviceExpertHgravsParams &p [[buffer(4)]],
    threadgroup float *shmem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (p.execution_position >= p.experts_per_token) {
        return;
    }
    uint expert = route_ids[p.execution_position];
    if (expert >= p.n_experts) {
        return;
    }
    const device Qwen30DeviceExpertTriplet &entry = table[expert];
    const device Qwen30DeviceExpertTensorRef *tensor =
        qwen30_select_projection(entry, p.projection);
    if (entry.ready_mask != QWEN30_EXPERT_TRIPLET_READY ||
        entry.generation != p.generation ||
        tensor->generation != p.generation ||
        tensor->kind != QWEN30_EXPERT_KIND_HGRAVS ||
        tensor->primary == nullptr ||
        tensor->secondary == nullptr ||
        tensor->rank == 0u ||
        tensor->rows == 0u ||
        tensor->cols == 0u) {
        return;
    }

    uint rows = 0u;
    uint cols = 0u;
    const device float *w = nullptr;
    if (p.stage == 0u) {
        // R is [rank, cols]
        rows = tensor->rank;
        cols = tensor->cols;
        w = reinterpret_cast<const device float *>(tensor->secondary);
    } else {
        // L is [rows, rank]
        rows = tensor->rows;
        cols = tensor->rank;
        w = reinterpret_cast<const device float *>(tensor->primary);
    }
    if (gid >= rows || w == nullptr) {
        return;
    }

    const device float *x = input + p.input_offset_elems;
    device float *y = output + p.output_offset_elems;
    const device float *row_w = w + (uint64_t)gid * (uint64_t)cols;

    float partial = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        partial += row_w[c] * x[c];
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        y[gid] = shmem[0];
    }
}
