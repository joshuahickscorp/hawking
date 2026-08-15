// Qwen80 device-indexed 512-way expert address table — Q30 pattern, 512/top-10.
//
// Host populates one per-layer table of 512 expert triplets (gate/up/down)
// with Metal gpuAddress values. Device route_ids select which of the top-10
// experts each organ executes — without a host bind per expert.
//
// ABI is frozen against the Rust structs in qwen80_device_expert_table.rs
// and is layout-identical to the Q30 device expert table so the same
// uniform-Q4 accumulation order applies. Every indirectly referenced
// slab must also be declared via useResources.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint QWEN80_EXPERT_KIND_UNIFORM_Q4 = 3u;
constant constexpr uint QWEN80_EXPERT_TRIPLET_READY = 7u;

struct Qwen80DeviceExpertTensorRef {
    const device uchar *primary;
    const device uchar *secondary;
    uint rows;
    uint cols;
    uint rank;
    uint kind;
    uint generation;
    uint pad;
};

struct Qwen80DeviceExpertTriplet {
    Qwen80DeviceExpertTensorRef gate;
    Qwen80DeviceExpertTensorRef up;
    Qwen80DeviceExpertTensorRef down;
    uint ready_mask;
    uint generation;
};

static_assert(sizeof(Qwen80DeviceExpertTensorRef) == 40,
              "Qwen80DeviceExpertTensorRef ABI drift");
static_assert(sizeof(Qwen80DeviceExpertTriplet) == 128,
              "Qwen80DeviceExpertTriplet ABI drift");

struct Qwen80DeviceExpertMatvecParams {
    uint n_experts;
    uint experts_per_token;
    uint generation;
    uint projection; // 0=gate, 1=up, 2=down
    uint group_size;
    uint max_rows;
    uint input_base_elems;
    uint input_stride_elems;
    uint output_base_elems;
    uint output_stride_elems;
    uint pad0;
    uint pad1;
};

static_assert(sizeof(Qwen80DeviceExpertMatvecParams) == 48,
              "Qwen80DeviceExpertMatvecParams ABI drift");

static inline const device Qwen80DeviceExpertTensorRef *
qwen80_select_projection(const device Qwen80DeviceExpertTriplet &entry, uint projection)
{
    if (projection == 0u) {
        return &entry.gate;
    }
    if (projection == 1u) {
        return &entry.up;
    }
    return &entry.down;
}

static inline float qwen80_uniform_q4_weight(
    const device uchar *codes,
    const device half *scales,
    uint row_group_base,
    uint col,
    uint group_size)
{
    const uint group = col / group_size;
    const uint local = col - group * group_size;
    const uint code_bytes_per_group = group_size >> 1u;
    const uint group_base = row_group_base + group;
    const uchar packed = codes[group_base * code_bytes_per_group + (local >> 1u)];
    const uchar nibble = ((local & 1u) == 0u) ? (packed & 0x0fu) : (packed >> 4u);
    return float(int(nibble) - 8) * float(scales[group_base]);
}

// Serial one-thread-per-(route,row) oracle. Accumulation order matches
// `qwen_uniform_q4_group64_matvec` and the Q30 serial table kernel:
// groups left-to-right, even nibble then odd nibble within each code byte.
// Grid: (experts_per_token * max_rows, 1, 1).
kernel void qwen80_expert_table_uniform_q4_matvec_serial(
    const device uint *route_ids [[buffer(0)]],
    const device Qwen80DeviceExpertTriplet *table [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Qwen80DeviceExpertMatvecParams &p [[buffer(4)]],
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
    const device Qwen80DeviceExpertTriplet &entry = table[expert];
    const device Qwen80DeviceExpertTensorRef *tensor =
        qwen80_select_projection(entry, p.projection);
    if (entry.ready_mask != QWEN80_EXPERT_TRIPLET_READY ||
        entry.generation != p.generation ||
        tensor->generation != p.generation ||
        tensor->kind != QWEN80_EXPERT_KIND_UNIFORM_Q4 ||
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

// Bit-identical occupancy: 4 serial rows per thread, 256 threads, 1024 rows/TG.
// Grid: (experts_per_token * ceil(max_rows / 1024) * 256, 1, 1), TG (256,1,1).
kernel void qwen80_expert_table_uniform_q4_matvec_rowblock(
    const device uint *route_ids [[buffer(0)]],
    const device Qwen80DeviceExpertTriplet *table [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Qwen80DeviceExpertMatvecParams &p [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    if (p.max_rows == 0u || p.experts_per_token == 0u) {
        return;
    }
    constexpr uint R = 4u;
    constexpr uint kThreads = 256u;
    constexpr uint kRowsPerTg = kThreads * R;
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
    const device Qwen80DeviceExpertTriplet &entry = table[expert];
    const device Qwen80DeviceExpertTensorRef *tensor =
        qwen80_select_projection(entry, p.projection);
    if (entry.ready_mask != QWEN80_EXPERT_TRIPLET_READY ||
        entry.generation != p.generation ||
        tensor->generation != p.generation ||
        tensor->kind != QWEN80_EXPERT_KIND_UNIFORM_Q4 ||
        tensor->primary == nullptr ||
        tensor->secondary == nullptr ||
        tensor->rows == 0u ||
        tensor->cols == 0u ||
        p.group_size == 0u) {
        return;
    }
    const uint row0 = group_in_route * kRowsPerTg + lid * R;
    if (row0 >= tensor->rows || row0 >= p.max_rows) {
        return;
    }

    const device uchar *codes = tensor->primary;
    const device half *scales =
        reinterpret_cast<const device half *>(tensor->secondary);
    const uint groups_per_row = (tensor->cols + p.group_size - 1u) / p.group_size;
    const uint in_off = p.input_base_elems + route * p.input_stride_elems;
    const uint out_off = p.output_base_elems + route * p.output_stride_elems;
    const device float *x = input + in_off;
    device float *y = output + out_off;

    uint rid[4];
    bool has[4];
    rid[0] = row0;
    has[0] = true;
    for (uint r = 1u; r < R; ++r) {
        rid[r] = row0 + r;
        has[r] = rid[r] < tensor->rows && rid[r] < p.max_rows;
        if (!has[r]) {
            rid[r] = row0;
        }
    }
    uint rgb[4];
    for (uint r = 0u; r < R; ++r) {
        rgb[r] = rid[r] * groups_per_row;
    }

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint group = 0u; group < groups_per_row; ++group) {
        const uint group_start = group * p.group_size;
        const uint group_end = min(group_start + p.group_size, tensor->cols);
        for (uint col = group_start; col < group_end; ++col) {
            const float xv = x[col];
            for (uint r = 0u; r < R; ++r) {
                acc[r] += qwen80_uniform_q4_weight(
                    codes, scales, rgb[r], col, p.group_size) * xv;
            }
        }
    }
    for (uint r = 0u; r < R; ++r) {
        if (has[r]) {
            y[rid[r]] = acc[r];
        }
    }
}

// Fast path: one simdgroup per (route, row). Not bit-identical to serial.
// Grid: (experts_per_token * ceil(max_rows / 8) * 256, 1, 1), TG (256,1,1).
kernel void qwen80_expert_table_uniform_q4_matvec_simdgroup(
    const device uint *route_ids [[buffer(0)]],
    const device Qwen80DeviceExpertTriplet *table [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant Qwen80DeviceExpertMatvecParams &p [[buffer(4)]],
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
    const device Qwen80DeviceExpertTriplet &entry = table[expert];
    const device Qwen80DeviceExpertTensorRef *tensor =
        qwen80_select_projection(entry, p.projection);
    if (entry.ready_mask != QWEN80_EXPERT_TRIPLET_READY ||
        entry.generation != p.generation ||
        tensor->generation != p.generation ||
        tensor->kind != QWEN80_EXPERT_KIND_UNIFORM_Q4 ||
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

    const device uchar *codes = tensor->primary;
    const device half *scales =
        reinterpret_cast<const device half *>(tensor->secondary);
    const uint groups_per_row = (tensor->cols + p.group_size - 1u) / p.group_size;
    const uint in_off = p.input_base_elems + route * p.input_stride_elems;
    const uint out_off = p.output_base_elems + route * p.output_stride_elems;
    const device float *x = input + in_off;
    device float *y = output + out_off;
    const uint row_group_base = row * groups_per_row;

    float partial = 0.0f;
    for (uint base = 0u; base < tensor->cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= tensor->cols) {
            continue;
        }
        partial += qwen80_uniform_q4_weight(
            codes, scales, row_group_base, col, p.group_size) * x[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        y[row] = partial;
    }
}

// Route-major SwiGLU: SiLU(gate) * up over top_k * intermediate elements.
kernel void qwen80_expert_table_silu_mul(
    const device float *gate [[buffer(0)]],
    const device float *up [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= elements) {
        return;
    }
    const float g = gate[id];
    output[id] = (g / (1.0f + exp(-g))) * up[id];
}

// Weighted sum of top-k route-major hidden slices into a fresh output.
kernel void qwen80_expert_table_weighted_sum(
    const device float *routed [[buffer(0)]],
    const device float *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &hidden [[buffer(3)]],
    constant uint &routes [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= hidden) {
        return;
    }
    float value = 0.0f;
    for (uint route = 0u; route < routes; ++route) {
        value = fma(weights[route], routed[route * hidden + id], value);
    }
    output[id] = value;
}
