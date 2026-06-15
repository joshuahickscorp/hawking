// strand_bitslice_decode.cu
// =============================================================================
// CUDA (nvrtc) port of the Metal G4 bitslice decode kernel
// (shaders/strand_bitslice.metal :: strand_bitslice_decode).
//
// PURPOSE
// -------
// Decode STRAND trellis-coded weights to the canonical Q12 integer domain on an
// NVIDIA GPU. The output of this kernel MUST be BYTE-IDENTICAL to:
//   * the CPU reference  strand_quant::decode::decode_tensor_fixed_with_lut, and
//   * the Metal kernel    strand_bitslice_decode (shaders/strand_bitslice.metal).
// on every input. This bit-exactness IS the product moat: the same .strand file
// decodes to the same integers on Apple GPU, NVIDIA GPU, and scalar CPU.
//
// DETERMINISM CONTRACT (why this is provably identical — see host module + the
// determinism argument in the task return):
//   1. NO FLOATING POINT anywhere in the decode path. Every operation is on
//      uint32 / int32 / uint64 / int64 with C/CUDA's fixed two's-complement,
//      well-defined widths. There is no fast-math, no FMA, no rounding mode, no
//      transcendental — nothing the compiler or the GPU may reorder or
//      re-associate in a value-changing way. Integer add/shift/and/or/mul are
//      bit-exact and associative-invariant across nvcc, nvrtc, clang, and rustc.
//   2. The frozen reconstruction is exactly  w = (int)(((int64)eff * (int64)q) >> 16) + off.
//      `eff` (int32) and `off` (int32) are hoisted HOST-SIDE (identical i32
//      values baked into the BitsliceEntry table by bake_bitslice_entries; they
//      come from eff_scale_q / eff_min_q, both integer). `q = lut[state]` is an
//      int32 read from a host-supplied integer LUT. The product is computed in
//      int64 to match the CPU's `(i64 * i64) >> 16` and Metal's `(long*long)>>16`
//      EXACTLY (same width, same arithmetic-shift-right of a signed value).
//   3. The bit reader is the byte-for-byte analog of CPU WordBitReader /
//      WordReader and Metal bs_load_u32_le: little-endian u32 word loads, a u64
//      accumulator, pop k bits LSB-first. Same word index, same shift counts,
//      same masks.
//   4. State recursion  state = ((state << k) | sym) & state_mask  is integer and
//      width-exact (uint32 wrap is irrelevant: state_mask <= 2^14-1 keeps it in
//      14 bits, far inside u32).
//   5. Per-block independence: one thread fully decodes one block from its baked
//      init_state. No cross-thread communication after the LUT load, so thread
//      scheduling / warp divergence cannot change any output value. The LUT load
//      is a pure broadcast of identical data followed by __syncthreads().
//
// SHARED MEMORY (compile-time switch: -DLUT_IN_SMEM=1)
// ----------------------------------------------------
// When `LUT_IN_SMEM` is defined the integer LUT (2^l_bits int32) is staged once
// per thread block into dynamic shared memory, exactly like Metal's
// `threadgroup int* sh_lut`, and the walk reads hit shared memory.
//   * L<=12  -> <=16 KB, fits the default 48 KB/block shared budget on all archs.
//   * L=13   -> 32 KB, still fits the default 48 KB.
//   * L=14   -> 64 KB, EXCEEDS the 48 KB default. The host then compiles the
//     GLOBAL-LUT variant (LUT_IN_SMEM undefined): the kernel reads the LUT from
//     global memory — byte-identical, just slower (L1/L2 cache the hot table).
// The frozen-state STRAND deploy configs (k3/L7=512 B, k2/L6=256 B, k4/L8=1 KB,
// k2/L12=16 KB) all sit at or below 16 KB, so the moat path is always smem.
//
// IMPORTANT: this source is the SINGLE SOURCE OF TRUTH for the kernel — the Rust
// host module `include_str!`s this exact file (mirroring how `strand_bitslice.metal`
// is `include_str!`'d into `metal.rs`), so there is no second copy to drift.
//
// LAUNCH CONTRACT (mirrors the Metal dispatch in BitsliceGpu::decode_q12):
//   grid  = ( ceil(n_blocks / 256), 1, 1 )
//   block = ( 256, 1, 1 )
//   dynamic shared = LUT_IN_SMEM ? (1u << l_bits) * sizeof(int) : 0
// Scalar params (n_blocks/k_bits/l_bits) are passed BY POINTER (length-1 device
// buffers), matching the host's `htod_sync_copy(&[v])` staging — kept as the
// Metal [[buffer(n)]] slot shape so the two backends line up 1:1.
//
// This source is compiled at runtime by nvrtc (compile_ptx), exactly like the
// CUDA encode backend (strand-quant/src/cuda_backend.rs). It therefore must NOT
// #include any system header (nvrtc has no default include path) — it uses only
// builtin integer types (no intrinsics needed: integer ops only).
// =============================================================================

// Layout MUST match the Rust #[repr(C)] BitsliceEntry (host module) AND the
// Metal `struct BitsliceEntry` field-for-field. All members are 4-byte scalars,
// so the natural C layout is: 4*u32 + 8*i32 + 8*i32 = 80 bytes, no padding.
// __align__(4) makes the alignment explicit and matches Rust's repr(C) for a
// struct whose largest scalar is 4 bytes. The host asserts sizeof == 80 against
// the kernel (entry_sizeof probe) before any decode, identical to the Metal
// `strand_bitslice_entry_sizeof` guard.
struct __align__(4) BitsliceEntry {
    unsigned int bit_offset;   // start bit of this block in the packed payload
    unsigned int init_state;   // pre-resolved start state (tail-biting folded in host-side)
    unsigned int out_off;      // first output index for this block
    unsigned int n;            // number of symbols (weights) in this block
    int          eff[8];       // per-sub-block effective scale_q (i32), index = j>>5
    int          off[8];       // per-sub-block additive offset (i32, affine-min); 0 if unused
};

// Little-endian u32 word load from a byte buffer, by WORD index.
// Byte-for-byte analog of CPU load_u32_le and Metal bs_load_u32_le. The host
// pads the payload up to a 4-byte boundary + 8 slack bytes (mirrors
// upload_payload) so an over-read by the look-ahead is always in-bounds and
// reads the host-written zero padding — same as the CPU reader's zero-fill past
// end. We index a const unsigned char* and assemble the 4 bytes explicitly so
// endianness is fixed regardless of host/device byte order.
__device__ __forceinline__ unsigned int
bs_load_u32_le(const unsigned char* p, unsigned int widx) {
    unsigned int b = widx << 2;
    return  (unsigned int)p[b]
         | ((unsigned int)p[b + 1] << 8)
         | ((unsigned int)p[b + 2] << 16)
         | ((unsigned int)p[b + 3] << 24);
}

extern "C" __global__ void strand_bitslice_decode(
    const unsigned char* w_bits,      // [packed payload bytes]   (buffer 0)
    int*                 out_q12,      // [total] decoded Q12 ints (buffer 1)
    const BitsliceEntry* tbl,          // [n_blocks]               (buffer 2)
    const unsigned int*  n_blocks_p,   // length-1 device buffer   (buffer 3)
    const unsigned int*  k_bits_p,     // length-1 device buffer   (buffer 4)
    const unsigned int*  l_bits_p,     // length-1 device buffer   (buffer 5)
    const int*           lut_q12)      // [1<<l_bits] integer LUT  (buffer 6)
{
    const unsigned int n_blocks = *n_blocks_p;
    const unsigned int k_bits   = *k_bits_p;
    const unsigned int l_bits   = *l_bits_p;
    const unsigned int lut_n    = 1u << l_bits;

    const unsigned int gidx = blockIdx.x * blockDim.x + threadIdx.x; // thread_position_in_grid

#ifdef LUT_IN_SMEM
    // Dynamic shared memory: the integer LUT, staged cooperatively by the block.
    // Identical values for every block; strided so all 256 threads participate,
    // exactly like the Metal `for (s = tid; s < lut_n; s += tgs)` loop. Read-only
    // after the barrier — no value-path mutation, so scheduling cannot perturb it.
    extern __shared__ int sh_lut[];
    for (unsigned int s = threadIdx.x; s < lut_n; s += blockDim.x) {
        sh_lut[s] = lut_q12[s];
    }
    __syncthreads();   // == threadgroup_barrier(mem_flags::mem_threadgroup)
    const int* LUT = sh_lut;
#else
    // Global-LUT variant (used only at L=14 where the smem table exceeds 48 KB).
    // Byte-identical: same i32 LUT values, just read from global memory.
    const int* LUT = lut_q12;
#endif

    // One thread decodes one block. Extra threads in the last partial block-grid
    // simply return (under LUT_IN_SMEM they have already done their share of the
    // LUT load before this early-out).
    if (gidx >= n_blocks) {
        return;
    }

    const unsigned int state_mask = lut_n - 1u;          // L-bit state mask
    const unsigned int input_mask = (1u << k_bits) - 1u; // k-bit symbol mask

    const BitsliceEntry e = tbl[gidx];

    unsigned int state    = e.init_state & state_mask;
    const unsigned int n  = e.n;
    const unsigned int obase = e.out_off;

    // Bit reader init — exact analog of WordBitReader::new / Metal lines 51-55.
    unsigned int word_idx       = e.bit_offset >> 5;
    const unsigned int bit_in_w = e.bit_offset & 31u;
    unsigned long long acc = (unsigned long long)(bs_load_u32_le(w_bits, word_idx) >> bit_in_w);
    unsigned int have      = 32u - bit_in_w;

    for (unsigned int j = 0u; j < n; ++j) {
        // Refill: pull the next LE word when fewer than k bits remain.
        if (have < k_bits) {
            unsigned long long nxt = (unsigned long long)bs_load_u32_le(w_bits, ++word_idx);
            acc |= nxt << have;
            have += 32u;
        }
        const unsigned int sym = (unsigned int)acc & input_mask;
        acc >>= k_bits;
        have -= k_bits;

        // Trellis state recursion (integer, width-exact).
        state = ((state << k_bits) | sym) & state_mask;

        const int q = LUT[state];

        // Per-sub-block side info: SUB_BLOCK = 32, so the sub index is j >> 5.
        // bake_bitslice_entries gates n <= 256 => at most 8 sub-blocks => index
        // in [0,7], always within eff[8]/off[8].
        const unsigned int sb = j >> 5;
        const int es = e.eff[sb];

        // FROZEN integer reconstruction. The int64 product + arithmetic >>16
        // matches the CPU reconstruct_q ((i64*i64)>>SCALE_SHIFT, SCALE_SHIFT=16)
        // and the Metal ((long)es*(long)q)>>16 EXACTLY. The cast back to int
        // truncates to 32 bits identically to the CPU `as i32` / Metal `(int)`.
        const int w = (int)(((long long)es * (long long)q) >> 16) + e.off[sb];

        out_q12[obase + j] = w;
    }
}

// sizeof probe — analog of Metal strand_bitslice_entry_sizeof. The host launches
// this with a 1-thread grid and asserts out[0] == sizeof(Rust BitsliceEntry)
// before any decode, so a layout drift between the Rust struct and this .cu
// struct is caught immediately (it would otherwise corrupt the tbl stride).
extern "C" __global__ void strand_bitslice_entry_sizeof(unsigned int* out) {
    out[0] = (unsigned int)sizeof(BitsliceEntry);
}
