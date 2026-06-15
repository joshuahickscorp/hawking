// strand_bitslice_staged.metal — NEW decode variant (prototype, NOT yet wired into BitsliceGpu)
//
// GOAL: close the bitslice decode bandwidth gap (measured ~51-70% of peak at k3 L7).
// The deployed `strand_bitslice_decode` is WRITE-BOUND: each thread owns one 256-weight
// block and stores out_q12[gidx*256 + j] for j=0..256. Within a 32-lane SIMD-group,
// adjacent lanes (block gidx, gidx+1) write addresses 256 ints = 1024 BYTES apart, so a
// 32-lane store fans out to ~32 separate cache-line transactions instead of 1-2. The 4 B/w
// Q12 output buffer is ~85% of decode-only traffic (per shaders/README.md), so that
// scatter is the dominant component of the 26% gap.
//
// FIX (this file): keep the EXACT serial trellis walk (bit-for-bit identical reconstruct),
// but route the per-block outputs through a THREADGROUP STAGING TILE and then flush the
// tile to device memory COALESCED — consecutive lanes writing consecutive global
// addresses. One threadgroup owns a CONTIGUOUS run of `bg` (= threads_per_threadgroup)
// blocks, so the tile covers out[base .. base + bg*256) which is contiguous in the output.
//
// Identity is preserved because:
//   * the per-block walk, LUT lookup, eff/off reconstruct are byte-identical to
//     strand_bitslice_decode (same ops, same order within a block);
//   * each block writes ONLY its own [local*256 .. local*256+n) slot of the tile, then the
//     tile is copied verbatim to out[obase0 + s]; obase0 = first block's out_off, and
//     because block_plans lays out_off = prefix_n (contiguous, ascending), the tile maps
//     1:1 onto a contiguous output region with no gaps or overlaps for full (n==256) blocks.
//
// LAYOUT CONTRACT (host must honor — see integration plan):
//   * grid = ceil(n_blocks / bg) threadgroups, bg = threads_per_threadgroup (e.g. 64).
//   * blocks are CONTIGUOUS and out_off is strictly ascending by n (true for block_plans).
//   * threadgroup memory: sh_lut (2^L ints) + sh_tile (bg*256 ints). Host picks bg so
//     bg*256*4 + (2^L)*4 <= 32768 (M3 TG limit). bg=64 => 64 KB tile — TOO BIG; use bg<=24.
//     For k3 L7 (512 B LUT): bg<=31 (31*1024+512=32256). For k2 L12 (16 KB LUT): bg<=16.
//     A safe single value for BOTH configs is bg=16 (16*1024 + 16384 = 32768, exactly at cap)
//     or bg=8 for headroom. The host clamps bg.
//
// This variant handles the common case where every block in the run has n==256 (the deploy
// invariant: cols % 256 == 0, full blocks). The LAST threadgroup may have a partial final
// block (n<256) and/or fewer than bg live blocks; both are handled by writing exactly n
// entries per block and flushing exactly the covered span.

#include <metal_stdlib>
using namespace metal;

struct BitsliceEntry {
    uint bit_offset;
    uint init_state;
    uint out_off;
    uint n;
    int  eff[8];
    int  off[8];
    uint d;
};

static inline uint bss_load_u32_le(device const uchar* p, uint widx) {
    uint b = widx << 2;
    return (uint)p[b]
         | ((uint)p[b + 1] << 8)
         | ((uint)p[b + 2] << 16)
         | ((uint)p[b + 3] << 24);
}

// Coalesced-write staged decode. tile_blocks = threads_per_threadgroup (= bg).
kernel void strand_bitslice_decode_staged(
    device   const uchar*          w_bits   [[buffer(0)]],
    device         int*            out_q12  [[buffer(1)]],
    device   const BitsliceEntry*  tbl      [[buffer(2)]],
    constant       uint&           n_blocks [[buffer(3)]],
    constant       uint&           k_bits   [[buffer(4)]],
    constant       uint&           l_bits   [[buffer(5)]],
    device   const int*            lut_q12  [[buffer(6)]],
    threadgroup    int*            sh_lut   [[threadgroup(0)]],
    threadgroup    int*            sh_tile  [[threadgroup(1)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]])
{
    uint lut_n = 1u << l_bits;
    for (uint s = tid; s < lut_n; s += tgs) sh_lut[s] = lut_q12[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint state_mask = lut_n - 1u;
    uint input_mask = (1u << k_bits) - 1u;

    // This threadgroup owns blocks [base_block .. base_block + tgs).
    uint base_block = gid * tgs;
    uint my_block   = base_block + tid;

    // First block's output base — the coalesced flush anchor. All blocks in this TG write a
    // CONTIGUOUS output region because block_plans assigns out_off = running prefix sum of n.
    uint obase0 = tbl[base_block].out_off;

    // Each live thread serially walks its block into the tile at [tid*256 .. tid*256 + n).
    if (my_block < n_blocks) {
        device const BitsliceEntry* e = &tbl[my_block];
        uint  state    = e->init_state & state_mask;
        uint  n        = e->n;
        uint  word_idx = e->bit_offset >> 5;
        uint  bit_in_w = e->bit_offset & 31u;
        ulong acc      = (ulong)(bss_load_u32_le(w_bits, word_idx) >> bit_in_w);
        uint  have     = 32u - bit_in_w;
        threadgroup int* dst = sh_tile + (ulong)tid * 256u;
        for (uint j = 0; j < n; ++j) {
            if (have < k_bits) {
                ulong nxt = (ulong)bss_load_u32_le(w_bits, ++word_idx);
                acc |= nxt << have;
                have += 32u;
            }
            uint sym = (uint)acc & input_mask;
            acc >>= k_bits;
            have -= k_bits;
            state = ((state << k_bits) | sym) & state_mask;
            int q  = sh_lut[state];
            uint sb = j >> 5;
            int w  = (int)(((long)e->eff[sb] * (long)q) >> 16) + e->off[sb];
            dst[j] = w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Coalesced flush. tile slot s maps 1:1 to global offset obase0 + s, because:
    //   - blocks in this TG are contiguous and out_off is the ascending prefix sum of n;
    //   - every block except possibly the LAST block of the whole tensor has n==256, so its
    //     tile span (local*256 .. local*256+256) equals its global span; the per-thread tile
    //     base tid*256 therefore lands exactly at (out_off - obase0).
    // The final partial block (n<256) is the last live block in its TG, so span stops at its
    // true end and no hole is ever flushed. (Proven on CPU: tests/staged_layout.rs.)
    //
    // Every thread independently computes the flush length from the table (one extra device
    // read each — cheap, and avoids a shared-memory broadcast + its barrier):
    // span = (last_live.out_off + last_live.n) - obase0.
    uint last = base_block + tgs - 1u;
    if (last >= n_blocks) last = n_blocks - 1u;
    device const BitsliceEntry* le = &tbl[last];
    uint span = (le->out_off + le->n) - obase0;

    for (uint s = tid; s < span; s += tgs) {
        out_q12[obase0 + s] = sh_tile[s];
    }
}

kernel void strand_bitslice_staged_sizeof(device uint* out [[buffer(0)]]) {
    out[0] = (uint)sizeof(BitsliceEntry);
}
