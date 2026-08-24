// N017 — pure streaming-read / streaming-write microbenchmarks.
// No model math. Accumulators exist only to keep loads/stores live.
// nbytes is ulong so a single dispatch can cover >2 GiB (the old unique_once
// kernel took uint nbytes and could not name a 13.6 GB point).

#include <metal_stdlib>
using namespace metal;

struct RoofParams {
    ulong nbytes;
    uint nthreads;
    uint iters;
    uint stride_bytes;
    uint nbufs;
};

inline void unique_slice(ulong nvec, uint tid, uint nthreads,
                         thread ulong &start, thread ulong &count) {
    if (nthreads == 0u || nvec == 0ul) {
        start = 0;
        count = 0;
        return;
    }
    const ulong mine = nvec / (ulong)nthreads;
    const ulong extra = nvec % (ulong)nthreads;
    start = (ulong)tid * mine + min((ulong)tid, extra);
    count = mine + ((ulong)tid < extra ? 1ul : 0ul);
}

inline float sum4(float4 v) { return v.x + v.y + v.z + v.w; }

kernel void roof_fill_u32(
    device uint *data [[buffer(0)]],
    constant ulong &nwords [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]])
{
    if (nthreads == 0u) {
        return;
    }
    for (ulong i = (ulong)tid; i < nwords; i += (ulong)nthreads) {
        data[i] = (uint)(i * 0x9E3779B9ul + 1ul);
    }
}

// Sequential unique-once, scalar float (4 B / load).
kernel void roof_seq_f1(
    device const float *data [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant RoofParams &p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.nthreads) {
        return;
    }
    ulong start, count;
    unique_slice(p.nbytes / 4ul, tid, p.nthreads, start, count);
    float acc = 0.0f;
    for (uint pass = 0u; pass < p.iters; ++pass) {
        ulong i = start;
        const ulong end = start + count;
        for (; i + 8ul <= end; i += 8ul) {
            acc += data[i] + data[i + 1ul] + data[i + 2ul] + data[i + 3ul]
                 + data[i + 4ul] + data[i + 5ul] + data[i + 6ul] + data[i + 7ul];
        }
        for (; i < end; ++i) {
            acc += data[i];
        }
    }
    out[tid] = acc;
}

// Sequential unique-once, float2 (8 B / load).
kernel void roof_seq_f2(
    device const float2 *data [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant RoofParams &p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.nthreads) {
        return;
    }
    ulong start, count;
    unique_slice(p.nbytes / 8ul, tid, p.nthreads, start, count);
    float acc = 0.0f;
    for (uint pass = 0u; pass < p.iters; ++pass) {
        ulong i = start;
        const ulong end = start + count;
        for (; i + 4ul <= end; i += 4ul) {
            float2 a = data[i];
            float2 b = data[i + 1ul];
            float2 c = data[i + 2ul];
            float2 d = data[i + 3ul];
            acc += a.x + a.y + b.x + b.y + c.x + c.y + d.x + d.y;
        }
        for (; i < end; ++i) {
            float2 v = data[i];
            acc += v.x + v.y;
        }
    }
    out[tid] = acc;
}

// Sequential unique-once, float4 (16 B / load). Widest natural vector.
kernel void roof_seq_f4(
    device const float4 *data [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant RoofParams &p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.nthreads) {
        return;
    }
    ulong start, count;
    unique_slice(p.nbytes / 16ul, tid, p.nthreads, start, count);
    float acc = 0.0f;
    for (uint pass = 0u; pass < p.iters; ++pass) {
        ulong i = start;
        const ulong end = start + count;
        for (; i + 8ul <= end; i += 8ul) {
            acc += sum4(data[i]) + sum4(data[i + 1ul]) + sum4(data[i + 2ul])
                 + sum4(data[i + 3ul]) + sum4(data[i + 4ul]) + sum4(data[i + 5ul])
                 + sum4(data[i + 6ul]) + sum4(data[i + 7ul]);
        }
        for (; i < end; ++i) {
            acc += sum4(data[i]);
        }
    }
    out[tid] = acc;
}

// Sequential unique-once, float4x4 (64 B / load). Four consecutive float4s.
kernel void roof_seq_f4x4(
    device const float4x4 *data [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant RoofParams &p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.nthreads) {
        return;
    }
    ulong start, count;
    unique_slice(p.nbytes / 64ul, tid, p.nthreads, start, count);
    float acc = 0.0f;
    for (uint pass = 0u; pass < p.iters; ++pass) {
        for (ulong i = start; i < start + count; ++i) {
            float4x4 m = data[i];
            acc += sum4(m[0]) + sum4(m[1]) + sum4(m[2]) + sum4(m[3]);
        }
    }
    out[tid] = acc;
}

// Sequential unique-once, 8 consecutive float4s (128 B / inner step).
// Widest per-thread streaming pattern this ISA usefully offers.
kernel void roof_seq_f4x8(
    device const float4 *data [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant RoofParams &p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.nthreads) {
        return;
    }
    ulong start, count;
    unique_slice(p.nbytes / 16ul, tid, p.nthreads, start, count);
    float acc = 0.0f;
    for (uint pass = 0u; pass < p.iters; ++pass) {
        ulong i = start;
        const ulong end = start + count;
        for (; i + 8ul <= end; i += 8ul) {
            acc += sum4(data[i]) + sum4(data[i + 1ul]) + sum4(data[i + 2ul])
                 + sum4(data[i + 3ul]) + sum4(data[i + 4ul]) + sum4(data[i + 5ul])
                 + sum4(data[i + 6ul]) + sum4(data[i + 7ul]);
        }
        for (; i < end; ++i) {
            acc += sum4(data[i]);
        }
    }
    out[tid] = acc;
}

// Sequential unique-once via simdgroup_load of an 8x8 float tile (256 B / simdgroup).
// Store to threadgroup so the loaded values stay live without indexing a
// simdgroup matrix from an arbitrary thread (that is not a documented op).
kernel void roof_seq_simd8x8(
    device const float *data [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant RoofParams &p [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    threadgroup float tile_mem[32 * 64];
    if (tid >= p.nthreads || tpg < 32u || (tpg / 32u) > 32u) {
        return;
    }
    const uint nsg_per_tg = tpg / 32u;
    const uint tg_index = tid / tpg;
    const uint global_sg = tg_index * nsg_per_tg + sg;
    const uint nsg = p.nthreads / 32u;
    ulong start, count;
    unique_slice(p.nbytes / 256ul, global_sg, nsg, start, count);
    float acc = 0.0f;
    threadgroup float *sh = tile_mem + sg * 64;
    for (uint pass = 0u; pass < p.iters; ++pass) {
        for (ulong tile = start; tile < start + count; ++tile) {
            simdgroup_float8x8 mat;
            simdgroup_load(mat, data + tile * 64ul, 8, ulong2(0, 0));
            simdgroup_store(mat, sh, 8, ulong2(0, 0));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            acc += sh[lane] + sh[lane + 32u];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    out[tid] = acc;
}

// Strided float4. Each load steps `stride_bytes` (must be multiple of 16).
// Bytes moved = nthreads * iters * 16, NOT unique coverage.
kernel void roof_stride_f4(
    device const float4 *data [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant RoofParams &p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.nthreads || p.stride_bytes < 16u || p.nbytes < 16ul) {
        return;
    }
    const ulong nvec = p.nbytes / 16ul;
    const ulong stride_vec = (ulong)p.stride_bytes / 16ul;
    ulong off = ((ulong)tid) % nvec;
    float acc = 0.0f;
    const uint loads = p.iters;
    for (uint i = 0u; i < loads; ++i) {
        acc += sum4(data[off]);
        off += stride_vec;
        if (off >= nvec) {
            off -= nvec;
        }
    }
    out[tid] = acc;
}

// Gather: hashed (not sequential) float4 loads. Bytes moved = nthreads * iters * 16.
kernel void roof_gather_f4(
    device const float4 *data [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant RoofParams &p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.nthreads || p.nbytes < 16ul) {
        return;
    }
    const ulong nvec = p.nbytes / 16ul;
    ulong seed = (ulong)tid * 0x9E3779B97F4A7C15ul + 1ul;
    float acc = 0.0f;
    for (uint i = 0u; i < p.iters; ++i) {
        seed = seed * 0xBF58476D1CE4E5B9ul + 1ul;
        const ulong idx = seed % nvec;
        acc += sum4(data[idx]);
    }
    out[tid] = acc;
}

// Indexed gather: indices[i] is a float4 slot. Unique-once over the index list.
kernel void roof_gather_indexed_f4(
    device const float4 *data [[buffer(0)]],
    device const uint *indices [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant RoofParams &p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.nthreads) {
        return;
    }
    const ulong nidx = p.nbytes / 4ul; // index buffer bytes, one uint each
    ulong start, count;
    unique_slice(nidx, tid, p.nthreads, start, count);
    float acc = 0.0f;
    for (uint pass = 0u; pass < p.iters; ++pass) {
        for (ulong i = start; i < start + count; ++i) {
            acc += sum4(data[(ulong)indices[i]]);
        }
    }
    out[tid] = acc;
}

// Four buffers, unique-once concatenated in logical order.
kernel void roof_multi_f4(
    device const float4 *a [[buffer(0)]],
    device const float4 *b [[buffer(1)]],
    device const float4 *c [[buffer(2)]],
    device const float4 *d [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant RoofParams &p [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.nthreads) {
        return;
    }
    const ulong nvec_each = p.nbytes / 16ul; // nbytes is per-buffer
    ulong start, count;
    unique_slice(nvec_each, tid, p.nthreads, start, count);
    float acc = 0.0f;
    for (uint pass = 0u; pass < p.iters; ++pass) {
        for (ulong i = start; i < start + count; ++i) {
            acc += sum4(a[i]) + sum4(b[i]) + sum4(c[i]) + sum4(d[i]);
        }
    }
    out[tid] = acc;
}

// Write-only unique-once float4.
kernel void roof_write_f4(
    device float4 *data [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant RoofParams &p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.nthreads) {
        return;
    }
    ulong start, count;
    unique_slice(p.nbytes / 16ul, tid, p.nthreads, start, count);
    float acc = 0.0f;
    const float4 v = float4(1.0f, 2.0f, 3.0f, 4.0f);
    for (uint pass = 0u; pass < p.iters; ++pass) {
        for (ulong i = start; i < start + count; ++i) {
            data[i] = v;
            acc += (float)i;
        }
    }
    out[tid] = acc;
}

// Read src, write dst, unique-once. Physical traffic = 2 * nbytes * iters.
kernel void roof_readwrite_f4(
    device const float4 *src [[buffer(0)]],
    device float4 *dst [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant RoofParams &p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.nthreads) {
        return;
    }
    ulong start, count;
    unique_slice(p.nbytes / 16ul, tid, p.nthreads, start, count);
    float acc = 0.0f;
    for (uint pass = 0u; pass < p.iters; ++pass) {
        for (ulong i = start; i < start + count; ++i) {
            float4 v = src[i];
            dst[i] = v;
            acc += sum4(v);
        }
    }
    out[tid] = acc;
}

// Deliberately-bad control: claims to stream p.nbytes but each thread
// reloads the same 16-byte slot. A no-op-passing benchmark would divide
// claimed nbytes by GPU time and report a fantasy GB/s. The harness
// rejects that number.
kernel void roof_bad_control(
    device const volatile float4 *data [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant RoofParams &p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.nthreads || p.nbytes < 16ul) {
        return;
    }
    const ulong nvec = p.nbytes / 16ul;
    const ulong slot = (ulong)(tid & 3u) % nvec;
    float acc = 0.0f;
    for (uint i = 0u; i < p.iters; ++i) {
        acc += sum4(data[slot]);
    }
    out[tid] = acc;
}
