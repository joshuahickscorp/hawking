// Qwen30 device-indexed expert address table — multi-route fused.
//
// Host populates one Gravity-style per-layer table of 128 expert triplets
// (gate/up/down) with Metal gpuAddress values. Device route_ids select which
// expert each of the top-k routes executes — without a host readback.
//
// Fusion contract (intersection of gemv_f32_moe + device table lookup):
// one dispatch covers all experts_per_token routes for a single organ stage.
// Threadgroups own (route, row); route_ids[route] indexes the table. This
// restores the control path's multi-expert-per-dispatch shape while keeping
// selection on device.
//
// Layout is frozen against the Rust structs in qwen30_complete_runtime.rs.
// Every indirectly referenced buffer must also be declared via useResources.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint QWEN30_EXPERT_KIND_BINARY = 1u;
constant constexpr uint QWEN30_EXPERT_KIND_HGRAVS = 2u;
// Uniform Q4 group-64 codes + FP16 scales (primary = codes, secondary = scales).
// Must match QWEN30_DEVICE_EXPERT_KIND_UNIFORM_Q4 in qwen30_complete_runtime.rs.
constant constexpr uint QWEN30_EXPERT_KIND_UNIFORM_Q4 = 3u;
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

// Multi-route fused binary matvec params.
// Grid covers experts_per_token * max_rows output rows (or simdgroup tiles).
struct Qwen30DeviceExpertMatvecParams {
    uint n_experts;
    uint experts_per_token;
    uint generation;
    uint projection; // 0=gate, 1=up, 2=down
    uint group_size;
    uint max_rows; // rows per route in the grid (intermediate or hidden)
    uint input_base_elems;
    uint input_stride_elems; // 0 when all routes share one input (e.g. x_norm)
    uint output_base_elems;
    uint output_stride_elems; // intermediate or hidden
    uint pad0;
    uint pad1;
};

static_assert(sizeof(Qwen30DeviceExpertMatvecParams) == 48,
              "Qwen30DeviceExpertMatvecParams ABI drift");

struct Qwen30DeviceExpertPairedParams {
    uint n_experts;
    uint experts_per_token;
    uint generation;
    uint group_size;
    uint max_rows; // intermediate rows per route
    uint output_base_elems;
    uint output_stride_elems; // intermediate
    uint pad0;
};

static_assert(sizeof(Qwen30DeviceExpertPairedParams) == 32,
              "Qwen30DeviceExpertPairedParams ABI drift");

struct Qwen30DeviceExpertHgravsParams {
    uint n_experts;
    uint experts_per_token;
    uint generation;
    uint projection; // 0=gate, 1=up, 2=down
    uint stage;     // 0 = R@x -> mid, 1 = L@mid -> y
    uint max_rows;  // grid rows per route (mid_stride / intermediate / hidden)
    uint input_base_elems;
    uint input_stride_elems;
    uint output_base_elems;
    uint output_stride_elems;
    uint pad0;
    uint pad1;
};

static_assert(sizeof(Qwen30DeviceExpertHgravsParams) == 48,
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

// Serial one-thread-per-(route,row) oracle for bit-identity A/B.
// Grid: (experts_per_token * max_rows, 1, 1).
kernel void qwen30_expert_table_binary_matvec_serial(
    const device uint *route_ids [[buffer(0)]],
    const device Qwen30DeviceExpertTriplet *table [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Qwen30DeviceExpertMatvecParams &p [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (p.max_rows == 0u || p.experts_per_token == 0u) {
        return;
    }
    const uint route = tid / p.max_rows;
    const uint row = tid % p.max_rows;
    if (route >= p.experts_per_token) {
        return;
    }
    const uint expert = route_ids[route];
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
    if (row >= tensor->rows || row >= p.max_rows) {
        return;
    }

    const device uchar *signs = tensor->primary;
    const device half *scales =
        reinterpret_cast<const device half *>(tensor->secondary);
    const uint groups_per_row = (tensor->cols + p.group_size - 1u) / p.group_size;
    const uint in_off = p.input_base_elems + route * p.input_stride_elems;
    const uint out_off = p.output_base_elems + route * p.output_stride_elems;
    const device float *x = input + in_off;
    device float *y = output + out_off;

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

// Serial one-thread-per-(route,row) uniform-Q4 oracle for bit-identity A/B.
// Accumulation order matches `qwen_uniform_q4_group64_matvec` exactly:
// groups left-to-right, even nibble then odd nibble within each code byte.
// Grid: (experts_per_token * max_rows, 1, 1).
// Codes live at tensor->primary; FP16 scales at tensor->secondary.
// group_size must be 64 (UNIFORM_Q4_GROUP_SIZE); code bytes per group = 32.
kernel void qwen30_expert_table_uniform_q4_matvec_serial(
    const device uint *route_ids [[buffer(0)]],
    const device Qwen30DeviceExpertTriplet *table [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Qwen30DeviceExpertMatvecParams &p [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (p.max_rows == 0u || p.experts_per_token == 0u) {
        return;
    }
    const uint route = tid / p.max_rows;
    const uint row = tid % p.max_rows;
    if (route >= p.experts_per_token) {
        return;
    }
    const uint expert = route_ids[route];
    if (expert >= p.n_experts) {
        return;
    }
    const device Qwen30DeviceExpertTriplet &entry = table[expert];
    const device Qwen30DeviceExpertTensorRef *tensor =
        qwen30_select_projection(entry, p.projection);
    if (entry.ready_mask != QWEN30_EXPERT_TRIPLET_READY ||
        entry.generation != p.generation ||
        tensor->generation != p.generation ||
        tensor->kind != QWEN30_EXPERT_KIND_UNIFORM_Q4 ||
        tensor->primary == nullptr ||
        tensor->secondary == nullptr ||
        tensor->rows == 0u ||
        tensor->cols == 0u ||
        p.group_size == 0u) {
        return;
    }
    if (row >= tensor->rows || row >= p.max_rows) {
        return;
    }

    const device uchar *codes = tensor->primary;
    const device half *scales =
        reinterpret_cast<const device half *>(tensor->secondary);
    const uint groups_per_row = (tensor->cols + p.group_size - 1u) / p.group_size;
    // code bytes per group = group_size / 2 (nibble packing).
    const uint code_bytes_per_group = p.group_size >> 1u;
    const uint in_off = p.input_base_elems + route * p.input_stride_elems;
    const uint out_off = p.output_base_elems + route * p.output_stride_elems;
    const device float *x = input + in_off;
    device float *y = output + out_off;

    float sum = 0.0f;
    const uint row_group_base = row * groups_per_row;
    for (uint group = 0u; group < groups_per_row; ++group) {
        const uint group_start = group * p.group_size;
        const uint group_end = min(group_start + p.group_size, tensor->cols);
        const uint group_base = row_group_base + group;
        const uint code_base = group_base * code_bytes_per_group;
        const float scale = float(scales[group_base]);
        for (uint col = group_start; col < group_end; ++col) {
            const uint local_col = col - group_start;
            const uchar packed = codes[code_base + (local_col >> 1u)];
            const uchar nibble = (local_col & 1u) == 0u
                ? (packed & 0x0fu)
                : (packed >> 4u);
            const int q = int(nibble) - 8;
            sum += float(q) * scale * x[col];
        }
    }
    y[row] = sum;
}

// Default binary sign/scale matvec, fused across top-k routes.
// Matches qwen_binary_sign_scale_matvec: one simdgroup per (route, row),
// contiguous 32-col tiles, simd_sum reduction.
// Grid: (experts_per_token * ceil(max_rows / 8) * 256, 1, 1), TG (256, 1, 1).
kernel void qwen30_expert_table_binary_matvec(
    const device uint *route_ids [[buffer(0)]],
    const device Qwen30DeviceExpertTriplet *table [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Qwen30DeviceExpertMatvecParams &p [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (p.max_rows == 0u || p.experts_per_token == 0u) {
        return;
    }
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    const uint groups_per_route = (p.max_rows + kSimdgroupsPerThreadgroup - 1u) /
                                  kSimdgroupsPerThreadgroup;
    if (groups_per_route == 0u) {
        return;
    }
    const uint route = group_id / groups_per_route;
    const uint group_in_route = group_id % groups_per_route;
    if (route >= p.experts_per_token) {
        return;
    }
    const uint expert = route_ids[route];
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
    const uint row = group_in_route * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= tensor->rows || row >= p.max_rows) {
        return;
    }

    const device uchar *signs = tensor->primary;
    const device half *scales =
        reinterpret_cast<const device half *>(tensor->secondary);
    const uint in_off = p.input_base_elems + route * p.input_stride_elems;
    const uint out_off = p.output_base_elems + route * p.output_stride_elems;
    const device float *x = input + in_off;
    device float *y = output + out_off;

    float partial = 0.0f;
    const uint row_base = row * tensor->cols;
    const uint scale_base =
        row * ((tensor->cols + p.group_size - 1u) / p.group_size);
    for (uint base = 0u; base < tensor->cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= tensor->cols) {
            continue;
        }
        const float scale = float(scales[scale_base + col / p.group_size]);
        const uint flat = row_base + col;
        const uchar byte = signs[flat >> 3u];
        const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
        partial += (positive ? scale : -scale) * x[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        y[row] = partial;
    }
}

// Register-blocked row variants for the fused expert-table binary matvec.
// Same association contract as qwen_binary_sign_scale_matvec_rowblock*: each
// simdgroup owns R consecutive rows of one route; within a row the 32-wide
// tile order matches the R=1 default (bit-identical by construction).
// Grid: (experts_per_token * ceil(max_rows / (8*R)) * 256, 1, 1), TG (256,1,1).

kernel void qwen30_expert_table_binary_matvec_rowblock2(
    const device uint *route_ids [[buffer(0)]],
    const device Qwen30DeviceExpertTriplet *table [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Qwen30DeviceExpertMatvecParams &p [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (p.max_rows == 0u || p.experts_per_token == 0u) {
        return;
    }
    constexpr uint R = 2u;
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kRowsPerTg = kSimdgroupsPerThreadgroup * R;
    const uint groups_per_route = (p.max_rows + kRowsPerTg - 1u) / kRowsPerTg;
    if (groups_per_route == 0u) {
        return;
    }
    const uint route = group_id / groups_per_route;
    const uint group_in_route = group_id % groups_per_route;
    if (route >= p.experts_per_token) {
        return;
    }
    const uint expert = route_ids[route];
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
    const uint row0 = group_in_route * kRowsPerTg + simd_id * R;
    if (row0 >= tensor->rows || row0 >= p.max_rows) {
        return;
    }
    const uint row1 = row0 + 1u;
    const bool has1 = row1 < tensor->rows && row1 < p.max_rows;
    const uint r1 = has1 ? row1 : row0;

    const device uchar *signs = tensor->primary;
    const device half *scales =
        reinterpret_cast<const device half *>(tensor->secondary);
    const uint in_off = p.input_base_elems + route * p.input_stride_elems;
    const uint out_off = p.output_base_elems + route * p.output_stride_elems;
    const device float *x = input + in_off;
    device float *y = output + out_off;
    const uint groups_per_row =
        (tensor->cols + p.group_size - 1u) / p.group_size;

    float a0 = 0.0f;
    float a1 = 0.0f;
    const uint rb0 = row0 * tensor->cols;
    const uint rb1 = r1 * tensor->cols;
    const uint sb0 = row0 * groups_per_row;
    const uint sb1 = r1 * groups_per_row;

    for (uint base = 0u; base < tensor->cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= tensor->cols) {
            continue;
        }
        const float xv = x[col];
        const uint g = col / p.group_size;
        {
            const float scale = float(scales[sb0 + g]);
            const uint flat = rb0 + col;
            const uchar byte = signs[flat >> 3u];
            a0 += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * xv;
        }
        {
            const float scale = float(scales[sb1 + g]);
            const uint flat = rb1 + col;
            const uchar byte = signs[flat >> 3u];
            a1 += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * xv;
        }
    }
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    if (simd_lane == 0u) {
        y[row0] = a0;
        if (has1) {
            y[row1] = a1;
        }
    }
}

kernel void qwen30_expert_table_binary_matvec_rowblock4(
    const device uint *route_ids [[buffer(0)]],
    const device Qwen30DeviceExpertTriplet *table [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Qwen30DeviceExpertMatvecParams &p [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (p.max_rows == 0u || p.experts_per_token == 0u) {
        return;
    }
    constexpr uint R = 4u;
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kRowsPerTg = kSimdgroupsPerThreadgroup * R;
    const uint groups_per_route = (p.max_rows + kRowsPerTg - 1u) / kRowsPerTg;
    if (groups_per_route == 0u) {
        return;
    }
    const uint route = group_id / groups_per_route;
    const uint group_in_route = group_id % groups_per_route;
    if (route >= p.experts_per_token) {
        return;
    }
    const uint expert = route_ids[route];
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
    const uint row0 = group_in_route * kRowsPerTg + simd_id * R;
    if (row0 >= tensor->rows || row0 >= p.max_rows) {
        return;
    }
    const uint row1 = row0 + 1u, row2 = row0 + 2u, row3 = row0 + 3u;
    const bool has1 = row1 < tensor->rows && row1 < p.max_rows;
    const bool has2 = row2 < tensor->rows && row2 < p.max_rows;
    const bool has3 = row3 < tensor->rows && row3 < p.max_rows;
    const uint r1 = has1 ? row1 : row0;
    const uint r2 = has2 ? row2 : row0;
    const uint r3 = has3 ? row3 : row0;

    const device uchar *signs = tensor->primary;
    const device half *scales =
        reinterpret_cast<const device half *>(tensor->secondary);
    const uint in_off = p.input_base_elems + route * p.input_stride_elems;
    const uint out_off = p.output_base_elems + route * p.output_stride_elems;
    const device float *x = input + in_off;
    device float *y = output + out_off;
    const uint groups_per_row =
        (tensor->cols + p.group_size - 1u) / p.group_size;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const uint rb0 = row0 * tensor->cols, rb1 = r1 * tensor->cols;
    const uint rb2 = r2 * tensor->cols, rb3 = r3 * tensor->cols;
    const uint sb0 = row0 * groups_per_row, sb1 = r1 * groups_per_row;
    const uint sb2 = r2 * groups_per_row, sb3 = r3 * groups_per_row;

    for (uint base = 0u; base < tensor->cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= tensor->cols) {
            continue;
        }
        const float xv = x[col];
        const uint g = col / p.group_size;
        {
            const float scale = float(scales[sb0 + g]);
            const uint flat = rb0 + col;
            const uchar byte = signs[flat >> 3u];
            a0 += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * xv;
        }
        {
            const float scale = float(scales[sb1 + g]);
            const uint flat = rb1 + col;
            const uchar byte = signs[flat >> 3u];
            a1 += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * xv;
        }
        {
            const float scale = float(scales[sb2 + g]);
            const uint flat = rb2 + col;
            const uchar byte = signs[flat >> 3u];
            a2 += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * xv;
        }
        {
            const float scale = float(scales[sb3 + g]);
            const uint flat = rb3 + col;
            const uchar byte = signs[flat >> 3u];
            a3 += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * xv;
        }
    }
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    a2 = simd_sum(a2);
    a3 = simd_sum(a3);
    if (simd_lane == 0u) {
        y[row0] = a0;
        if (has1) y[row1] = a1;
        if (has2) y[row2] = a2;
        if (has3) y[row3] = a3;
    }
}

kernel void qwen30_expert_table_binary_matvec_rowblock8(
    const device uint *route_ids [[buffer(0)]],
    const device Qwen30DeviceExpertTriplet *table [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Qwen30DeviceExpertMatvecParams &p [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    if (p.max_rows == 0u || p.experts_per_token == 0u) {
        return;
    }
    constexpr uint R = 8u;
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kRowsPerTg = kSimdgroupsPerThreadgroup * R;
    const uint groups_per_route = (p.max_rows + kRowsPerTg - 1u) / kRowsPerTg;
    if (groups_per_route == 0u) {
        return;
    }
    const uint route = group_id / groups_per_route;
    const uint group_in_route = group_id % groups_per_route;
    if (route >= p.experts_per_token) {
        return;
    }
    const uint expert = route_ids[route];
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
    const uint row0 = group_in_route * kRowsPerTg + simd_id * R;
    if (row0 >= tensor->rows || row0 >= p.max_rows) {
        return;
    }
    uint rid[8];
    bool has[8];
    rid[0] = row0;
    has[0] = true;
    for (uint r = 1u; r < R; ++r) {
        rid[r] = row0 + r;
        has[r] = rid[r] < tensor->rows && rid[r] < p.max_rows;
        if (!has[r]) {
            rid[r] = row0;
        }
    }

    const device uchar *signs = tensor->primary;
    const device half *scales =
        reinterpret_cast<const device half *>(tensor->secondary);
    const uint in_off = p.input_base_elems + route * p.input_stride_elems;
    const uint out_off = p.output_base_elems + route * p.output_stride_elems;
    const device float *x = input + in_off;
    device float *y = output + out_off;
    const uint groups_per_row =
        (tensor->cols + p.group_size - 1u) / p.group_size;

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint rb[8];
    uint sb[8];
    for (uint r = 0u; r < R; ++r) {
        rb[r] = rid[r] * tensor->cols;
        sb[r] = rid[r] * groups_per_row;
    }

    for (uint base = 0u; base < tensor->cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= tensor->cols) {
            continue;
        }
        const float xv = x[col];
        const uint g = col / p.group_size;
        for (uint r = 0u; r < R; ++r) {
            const float scale = float(scales[sb[r] + g]);
            const uint flat = rb[r] + col;
            const uchar byte = signs[flat >> 3u];
            acc[r] += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * xv;
        }
    }
    for (uint r = 0u; r < R; ++r) {
        acc[r] = simd_sum(acc[r]);
    }
    if (simd_lane == 0u) {
        for (uint r = 0u; r < R; ++r) {
            if (has[r]) {
                y[row0 + r] = acc[r];
            }
        }
    }
}

// Strided-lane simdgroup A/B entry (lane owns col, col+32, ...). Same grid as
// the tiled default; retained for --packed-matvec-kernel simdgroup-candidate.
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
    if (p.max_rows == 0u || p.experts_per_token == 0u) {
        return;
    }
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint groups_per_route = (p.max_rows + kSimdgroupsPerThreadgroup - 1u) /
                                  kSimdgroupsPerThreadgroup;
    if (groups_per_route == 0u) {
        return;
    }
    const uint route = group_id / groups_per_route;
    const uint group_in_route = group_id % groups_per_route;
    if (route >= p.experts_per_token) {
        return;
    }
    const uint expert = route_ids[route];
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
    const uint row = group_in_route * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= tensor->rows || row >= p.max_rows) {
        return;
    }

    const device uchar *signs = tensor->primary;
    const device half *scales =
        reinterpret_cast<const device half *>(tensor->secondary);
    const uint in_off = p.input_base_elems + route * p.input_stride_elems;
    const uint out_off = p.output_base_elems + route * p.output_stride_elems;
    const device float *x = input + in_off;
    device float *y = output + out_off;

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

// Paired gate/up SwiGLU with scalar-order arithmetic, table-indexed, fused
// across top-k routes. Matches
// qwen_direct_packed_gate_up_swiglu_paired_scalar_order_candidate.
// Grid: (experts_per_token * max_rows, 1, 1).
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
    uint tid [[thread_position_in_grid]])
{
    if (p.max_rows == 0u || p.experts_per_token == 0u) {
        return;
    }
    const uint route = tid / p.max_rows;
    const uint row = tid % p.max_rows;
    if (route >= p.experts_per_token) {
        return;
    }
    const uint expert = route_ids[route];
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
    if (row >= entry.gate.rows || row >= p.max_rows) {
        return;
    }

    const device uchar *gate_signs = entry.gate.primary;
    const device half *gate_scales =
        reinterpret_cast<const device half *>(entry.gate.secondary);
    const device uchar *up_signs = entry.up.primary;
    const device half *up_scales =
        reinterpret_cast<const device half *>(entry.up.secondary);
    device float *out =
        activation + p.output_base_elems + route * p.output_stride_elems;

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

// HGRAVS01 stage gemv via device pointers, fused across top-k routes.
// stage 0: mid = R @ x  (weight = secondary, rows=rank, cols=cols)
// stage 1: y   = L @ mid (weight = primary, rows=rows, cols=rank)
// Grid: (experts_per_token * max_rows * 256, 1, 1), TG (256, 1, 1)
// — same per-row reduction as gemv_f32_moe, with route packed into the grid.
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
    if (p.max_rows == 0u || p.experts_per_token == 0u) {
        return;
    }
    const uint route = gid / p.max_rows;
    const uint row = gid % p.max_rows;
    if (route >= p.experts_per_token) {
        return;
    }
    const uint expert = route_ids[route];
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
    if (row >= rows || row >= p.max_rows || w == nullptr) {
        return;
    }

    const uint in_off = p.input_base_elems + route * p.input_stride_elems;
    const uint out_off = p.output_base_elems + route * p.output_stride_elems;
    const device float *x = input + in_off;
    device float *y = output + out_off;
    const device float *row_w = w + (uint64_t)row * (uint64_t)cols;

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
        y[row] = shmem[0];
    }
}

// Register-blocked HGRAVS: one threadgroup owns R consecutive rows of one
// route. Each thread keeps R independent partials (same column stride as the
// R=1 kernel), then tree-reduces each row independently through shared memory
// in row order. Per-row accumulation and reduction order match the R=1 kernel,
// so outputs are bit-identical by construction.
// Grid: (experts_per_token * ceil(max_rows / R) * 256, 1, 1), TG (256, 1, 1).
// shmem: tg_size floats (reused across the R sequential reductions).

kernel void qwen30_expert_table_hgravs_gemv_rowblock2(
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
    if (p.max_rows == 0u || p.experts_per_token == 0u) {
        return;
    }
    constexpr uint R = 2u;
    const uint groups_per_route = (p.max_rows + R - 1u) / R;
    if (groups_per_route == 0u) {
        return;
    }
    const uint route = gid / groups_per_route;
    const uint group_in_route = gid % groups_per_route;
    if (route >= p.experts_per_token) {
        return;
    }
    const uint expert = route_ids[route];
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
        rows = tensor->rank;
        cols = tensor->cols;
        w = reinterpret_cast<const device float *>(tensor->secondary);
    } else {
        rows = tensor->rows;
        cols = tensor->rank;
        w = reinterpret_cast<const device float *>(tensor->primary);
    }
    const uint row0 = group_in_route * R;
    if (row0 >= rows || row0 >= p.max_rows || w == nullptr) {
        return;
    }
    const uint row1 = row0 + 1u;
    const bool has1 = row1 < rows && row1 < p.max_rows;

    const uint in_off = p.input_base_elems + route * p.input_stride_elems;
    const uint out_off = p.output_base_elems + route * p.output_stride_elems;
    const device float *x = input + in_off;
    device float *y = output + out_off;
    const device float *w0 = w + (uint64_t)row0 * (uint64_t)cols;
    const device float *w1 = has1 ? (w + (uint64_t)row1 * (uint64_t)cols) : w0;

    float p0 = 0.0f;
    float p1 = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        const float xv = x[c];
        p0 += w0[c] * xv;
        p1 += w1[c] * xv;
    }

    // Reduce row0, then row1, reusing shmem — same tree as the R=1 kernel.
    shmem[tid] = p0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        y[row0] = shmem[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (has1) {
        shmem[tid] = p1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                shmem[tid] += shmem[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            y[row1] = shmem[0];
        }
    }
}

kernel void qwen30_expert_table_hgravs_gemv_rowblock4(
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
    if (p.max_rows == 0u || p.experts_per_token == 0u) {
        return;
    }
    constexpr uint R = 4u;
    const uint groups_per_route = (p.max_rows + R - 1u) / R;
    if (groups_per_route == 0u) {
        return;
    }
    const uint route = gid / groups_per_route;
    const uint group_in_route = gid % groups_per_route;
    if (route >= p.experts_per_token) {
        return;
    }
    const uint expert = route_ids[route];
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
        rows = tensor->rank;
        cols = tensor->cols;
        w = reinterpret_cast<const device float *>(tensor->secondary);
    } else {
        rows = tensor->rows;
        cols = tensor->rank;
        w = reinterpret_cast<const device float *>(tensor->primary);
    }
    const uint row0 = group_in_route * R;
    if (row0 >= rows || row0 >= p.max_rows || w == nullptr) {
        return;
    }
    uint rid[4];
    bool has[4];
    rid[0] = row0;
    has[0] = true;
    for (uint r = 1u; r < R; ++r) {
        rid[r] = row0 + r;
        has[r] = rid[r] < rows && rid[r] < p.max_rows;
        if (!has[r]) {
            rid[r] = row0;
        }
    }

    const uint in_off = p.input_base_elems + route * p.input_stride_elems;
    const uint out_off = p.output_base_elems + route * p.output_stride_elems;
    const device float *x = input + in_off;
    device float *y = output + out_off;
    const device float *wr[4];
    for (uint r = 0u; r < R; ++r) {
        wr[r] = w + (uint64_t)rid[r] * (uint64_t)cols;
    }

    float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint c = tid; c < cols; c += tg_size) {
        const float xv = x[c];
        for (uint r = 0u; r < R; ++r) {
            partial[r] += wr[r][c] * xv;
        }
    }

    for (uint r = 0u; r < R; ++r) {
        if (!has[r]) {
            continue;
        }
        shmem[tid] = partial[r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                shmem[tid] += shmem[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            y[row0 + r] = shmem[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void qwen30_expert_table_hgravs_gemv_rowblock8(
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
    if (p.max_rows == 0u || p.experts_per_token == 0u) {
        return;
    }
    constexpr uint R = 8u;
    const uint groups_per_route = (p.max_rows + R - 1u) / R;
    if (groups_per_route == 0u) {
        return;
    }
    const uint route = gid / groups_per_route;
    const uint group_in_route = gid % groups_per_route;
    if (route >= p.experts_per_token) {
        return;
    }
    const uint expert = route_ids[route];
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
        rows = tensor->rank;
        cols = tensor->cols;
        w = reinterpret_cast<const device float *>(tensor->secondary);
    } else {
        rows = tensor->rows;
        cols = tensor->rank;
        w = reinterpret_cast<const device float *>(tensor->primary);
    }
    const uint row0 = group_in_route * R;
    if (row0 >= rows || row0 >= p.max_rows || w == nullptr) {
        return;
    }
    uint rid[8];
    bool has[8];
    rid[0] = row0;
    has[0] = true;
    for (uint r = 1u; r < R; ++r) {
        rid[r] = row0 + r;
        has[r] = rid[r] < rows && rid[r] < p.max_rows;
        if (!has[r]) {
            rid[r] = row0;
        }
    }

    const uint in_off = p.input_base_elems + route * p.input_stride_elems;
    const uint out_off = p.output_base_elems + route * p.output_stride_elems;
    const device float *x = input + in_off;
    device float *y = output + out_off;
    const device float *wr[8];
    for (uint r = 0u; r < R; ++r) {
        wr[r] = w + (uint64_t)rid[r] * (uint64_t)cols;
    }

    float partial[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint c = tid; c < cols; c += tg_size) {
        const float xv = x[c];
        for (uint r = 0u; r < R; ++r) {
            partial[r] += wr[r][c] * xv;
        }
    }

    for (uint r = 0u; r < R; ++r) {
        if (!has[r]) {
            continue;
        }
        shmem[tid] = partial[r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                shmem[tid] += shmem[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            y[row0 + r] = shmem[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
