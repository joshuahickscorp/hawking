// SCRATCH EXPERIMENT (gate-coopwindow): de-risk the COOPERATIVE WINDOWED decode kernel
// for the bandwidth-bound, float-free, computed-codebook pivot.
//
// Hypotheses under test (all measured vs the EXISTING fused B=1 path as baseline):
//   H1 "computed codebook removes the gather helps?": replace `q = sh_lut[state]`
//      (2^L i32 staged in shared mem, data-dependent index) with the float-free
//      computed codebook  q = quantile[hash_state(state)]  (hash = ~7 ALU ops, quantile
//      a 128/4096 i16 table). Same block-per-thread layout, NO barrier/shmem-LUT for L7.
//   H2 "cooperative windowed decode": threadgroup-per-block; threads decode the block's
//      weights IN PARALLEL using the window property (weight j's state = last L bits of
//      [init_state ++ block_bits] ending at (j+1)*k), computed codebook per lane, MAC into
//      per-thread partials, SIMD+threadgroup reduction. No serial recurrence.
//
// DETERMINISM: the computed codebook reproduces codebook_lut[state] BIT-IDENTICALLY because
// codebook_lut[s] == quantile_lut[hash_state(s)] by construction (codebook.rs:97-101). The
// kernel computes hash_state on the fly (pure integer) and indexes the frozen quantile table.
// Identity is gated vs decode_tensor_fixed before any timing is trusted.
//
// Self-contained: own MSL, own GPU wrapper, own bakes. Touches NO tracked file. Reference
// timing uses the existing public BitsliceGpu::bench_matvec for apples-to-apples on the same
// ffn_down shape the prior gates used.

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("gate-coopwindow: Metal is macOS-only; nothing to run on this target.");
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod macos {
    use metal::{Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize, NSUInteger};
    use strand_decode_kernel::block_walk::gate_proto::{machine_stamp, synth_encoded};
    use strand_decode_kernel::metal::{bake_bitslice_entries, BitsliceEntry, BitsliceGpu, StrandGpu};
    use strand_quant::codebook::{codebook_lut, quantile_lut};
    use strand_quant::decode::decode_tensor_fixed;
    use strand_quant::TrellisConfig;

    // The frozen quantile table as i16 (range +-15025 fits i16). Indexed by hash_state(state).
    fn quantile_i16(l_bits: u32) -> Vec<i16> {
        quantile_lut(l_bits).iter().map(|&v| v as i16).collect()
    }

    // ============================================================================
    // Self-contained Metal: computed-codebook block-per-thread (K1) and the
    // cooperative windowed (K2) gemv-partials kernels + a copy of reduce_rows.
    //
    // hash_state (MUST match codebook.rs:85-95 exactly), 32-bit ALU. The reference uses
    // u64 multiplies masked to l_bits; for l<=16 the low-32 product bits suffice because
    // the result is masked to l_bits (<=12 here). We do the multiply in 32-bit and mask.
    // We VALIDATE this against the i64 reference by the identity gate (below) — if the
    // 32-bit hash ever diverged from the i64 hash for some state, identity would fail.
    // ============================================================================
    const MSL: &str = r#"
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

static inline uint bs_load_u32_le(device const uchar* p, uint widx) {
    uint b = widx << 2;
    return (uint)p[b] | ((uint)p[b+1]<<8) | ((uint)p[b+2]<<16) | ((uint)p[b+3]<<24);
}

// hash_state mirror. r = max(l/2,1). Constants are the low 32 bits of the i64 consts;
// because the result is masked to l_bits (<=12), only the low l_bits of the product
// matter and 32-bit multiply low-bits are exact for those. (Gated by identity.)
static inline uint hash_state(uint s, uint l_bits) {
    uint mask = (1u << l_bits) - 1u;
    uint r = max(l_bits >> 1, 1u);
    uint h = s & mask;
    h = (h ^ (h >> r)) & mask;
    h = (h * 0x4F6CDD1Du) & mask;   // low32 of 0x2545F4914F6CDD1D
    h = (h ^ (h >> r)) & mask;
    h = (h * 0x7F4A7C15u) & mask;   // low32 of 0x9E3779B97F4A7C15
    return h & mask;
}

// ---- K1: computed-codebook, block-per-thread (drop-in shape vs the reference fused) ----
// One thread per block. NO sh_lut, NO barrier. q = quantile[hash_state(state)].
// quantile table staged in shared mem ONCE as i16 (256 B for L7, 8 KB for L12) — still a
// small shared read, but the index is hash(state) not state, and the table is i16.
kernel void k1_computed_blockthread(
    device   const uchar*          w_bits   [[buffer(0)]],
    device   const float*          x        [[buffer(1)]],
    device         float*          partials [[buffer(2)]],
    device   const BitsliceEntry*  tbl      [[buffer(3)]],
    constant       uint&           n_blocks [[buffer(4)]],
    constant       uint&           cols     [[buffer(5)]],
    constant       uint&           k_bits   [[buffer(6)]],
    constant       uint&           l_bits   [[buffer(7)]],
    device   const short*          quant    [[buffer(8)]],
    threadgroup    short*          sh_q     [[threadgroup(0)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint gidx [[thread_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]])
{
    uint qn = 1u << l_bits;
    for (uint s = tid; s < qn; s += tgs) sh_q[s] = quant[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gidx >= n_blocks) return;

    uint state_mask = qn - 1u;
    uint input_mask = (1u << k_bits) - 1u;
    const float Q12_TO_F32 = 1.0f / 4096.0f;

    device const BitsliceEntry* e = &tbl[gidx];
    uint state = e->init_state & state_mask;
    uint n = e->n;
    uint col0 = e->out_off % cols;
    uint word_idx = e->bit_offset >> 5;
    uint bit_in_w = e->bit_offset & 31u;
    ulong acc = (ulong)(bs_load_u32_le(w_bits, word_idx) >> bit_in_w);
    uint have = 32u - bit_in_w;

    float partial = 0.0f;
    for (uint j = 0; j < n; ++j) {
        if (have < k_bits) { ulong nxt=(ulong)bs_load_u32_le(w_bits, ++word_idx); acc|=nxt<<have; have+=32u; }
        uint sym = (uint)acc & input_mask; acc >>= k_bits; have -= k_bits;
        state = ((state << k_bits) | sym) & state_mask;
        int q = (int)sh_q[hash_state(state, l_bits)];
        int w = (int)(((long)e->eff[j>>5]*(long)q)>>16) + e->off[j>>5];
        partial += (float)w * Q12_TO_F32 * x[col0 + j];
    }
    partials[gidx] = partial;
}

// ---- windowed state closed form (CPU-verified, /tmp window2.py) -----------------------
// The state USED for output j (the codebook index) is the POST-update recurrence state =
// last l_bits of [init ++ sym_0 ++ ... ++ sym_j], where in the recurrence the NEWEST symbol
// is in the LOW bits. Closed form: state_j = (init<<((j+1)k) | Σ_i sym_i<<((j-i)k)) & mask.
// Efficient: only the most recent m=ceil(l/k) symbols matter (symbol i sits at bit (j-i)k;
// for (j-i)k>=l it is masked out), plus init ONLY when (j+1)k < l (the first few outputs).
// `base_bit` is the absolute bit (in w_bits / sh_bits frame) of symbol 0.
static inline uint windowed_state_dev(device const uchar* w_bits, uint base_bit, uint init,
                                      uint j, uint k, uint l) {
    uint mask = (1u << l) - 1u;
    uint m = (l + k - 1u) / k;                 // recent symbols that can land within l bits
    uint lo = (j + 1u > m) ? (j + 1u - m) : 0u;
    uint val = 0u;
    for (uint i = lo; i <= j; ++i) {
        uint sb = base_bit + i * k;
        uint wi = sb >> 5; uint bo = sb & 31u;
        ulong chunk = ((ulong)bs_load_u32_le(w_bits, wi) | ((ulong)bs_load_u32_le(w_bits, wi+1u) << 32)) >> bo;
        uint sym = (uint)chunk & ((1u << k) - 1u);
        val |= sym << ((j - i) * k);
    }
    if ((j + 1u) * k < l) val |= (init << ((j + 1u) * k));
    return val & mask;
}
// ---- K2 decode-only: windowed decode to Q12 (identity gate for the windowed math) -----
// One SIMD-group per block; lanes decode j=lane,lane+32,... via the window property and
// write out_q12[out_off+j]. Proves the windowed state == serial recurrence bit-for-bit.
kernel void k2_windowed_decode(
    device   const uchar*          w_bits   [[buffer(0)]],
    device         int*            out_q12  [[buffer(1)]],
    device   const BitsliceEntry*  tbl      [[buffer(2)]],
    constant       uint&           n_blocks [[buffer(3)]],
    constant       uint&           k_bits   [[buffer(4)]],
    constant       uint&           l_bits   [[buffer(5)]],
    device   const short*          quant    [[buffer(6)]],
    threadgroup    short*          sh_q     [[threadgroup(0)]],
    uint tid   [[thread_position_in_threadgroup]],
    uint gid   [[threadgroup_position_in_grid]],
    uint tgs   [[threads_per_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint nsg   [[simdgroups_per_threadgroup]])
{
    uint qn = 1u << l_bits;
    for (uint s = tid; s < qn; s += tgs) sh_q[s] = quant[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint blk = gid * nsg + sgid;
    if (blk >= n_blocks) return;
    uint state_mask = qn - 1u;
    device const BitsliceEntry* e = &tbl[blk];
    uint n=e->n; uint init=e->init_state&state_mask; uint obase=e->out_off; uint base_bit=e->bit_offset;
    for (uint j = lane; j < n; j += 32u) {
        uint state = windowed_state_dev(w_bits, base_bit, init, j, k_bits, l_bits);
        int q = (int)sh_q[hash_state(state, l_bits)];
        int w = (int)(((long)e->eff[j>>5]*(long)q)>>16) + e->off[j>>5];
        out_q12[obase + j] = w;
    }
}

// ---- K2: cooperative WINDOWED decode, BG blocks per threadgroup -----------------------
// grid = ceil(n_blocks/BG) threadgroups, TPB = BG*W threads (W = SIMD width 32, one
// SIMD-group per block). The quant table is staged ONCE per threadgroup (amortized over BG
// blocks — fixes the per-block table re-stage traffic that killed the naive 1-block-per-TG
// variant). Each SIMD-group g owns block (gid*BG + g): its 32 lanes decode weights
// j = lane, lane+32, ... INDEPENDENTLY via the window property (computed codebook, no serial
// recurrence), MAC into a per-lane partial, then a SIMD reduction (simd_sum, no barrier)
// produces the block's partial. Payload is read directly from device (cache-resident); no
// sh_bits stage (the windowed read touches only m+1 recent symbols/weight, fully cached).
kernel void k2_coop_windowed(
    device   const uchar*          w_bits   [[buffer(0)]],
    device   const float*          x        [[buffer(1)]],
    device         float*          partials [[buffer(2)]],
    device   const BitsliceEntry*  tbl      [[buffer(3)]],
    constant       uint&           n_blocks [[buffer(4)]],
    constant       uint&           cols     [[buffer(5)]],
    constant       uint&           k_bits   [[buffer(6)]],
    constant       uint&           l_bits   [[buffer(7)]],
    device   const short*          quant    [[buffer(8)]],
    threadgroup    short*          sh_q     [[threadgroup(0)]],
    uint tid   [[thread_position_in_threadgroup]],
    uint gid   [[threadgroup_position_in_grid]],
    uint tgs   [[threads_per_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint nsg   [[simdgroups_per_threadgroup]])
{
    uint qn = 1u << l_bits;
    for (uint s = tid; s < qn; s += tgs) sh_q[s] = quant[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint blk = gid * nsg + sgid;             // this SIMD-group's block
    if (blk >= n_blocks) return;

    uint state_mask = qn - 1u;
    const float Q12_TO_F32 = 1.0f / 4096.0f;

    device const BitsliceEntry* e = &tbl[blk];
    uint n = e->n;
    uint init = e->init_state & state_mask;
    uint col0 = e->out_off % cols;
    uint base_bit = e->bit_offset;

    float partial = 0.0f;
    for (uint j = lane; j < n; j += 32u) {
        uint state = windowed_state_dev(w_bits, base_bit, init, j, k_bits, l_bits);
        int q = (int)sh_q[hash_state(state, l_bits)];
        int w = (int)(((long)e->eff[j>>5]*(long)q)>>16) + e->off[j>>5];
        partial += (float)w * Q12_TO_F32 * x[col0 + j];
    }
    float blk_sum = simd_sum(partial);       // SIMD reduction, no barrier
    if (lane == 0u) partials[blk] = blk_sum;
}

// ---- decode-only variants (Q12 out), to isolate decode from MAC ----------------------
kernel void k1_computed_decode(
    device   const uchar*          w_bits   [[buffer(0)]],
    device         int*            out_q12  [[buffer(1)]],
    device   const BitsliceEntry*  tbl      [[buffer(2)]],
    constant       uint&           n_blocks [[buffer(3)]],
    constant       uint&           k_bits   [[buffer(4)]],
    constant       uint&           l_bits   [[buffer(5)]],
    device   const short*          quant    [[buffer(6)]],
    threadgroup    short*          sh_q     [[threadgroup(0)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint gidx [[thread_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]])
{
    uint qn = 1u << l_bits;
    for (uint s = tid; s < qn; s += tgs) sh_q[s] = quant[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gidx >= n_blocks) return;
    uint state_mask = qn - 1u; uint input_mask = (1u << k_bits) - 1u;
    device const BitsliceEntry* e = &tbl[gidx];
    uint state=e->init_state&state_mask; uint n=e->n; uint obase=e->out_off;
    uint word_idx=e->bit_offset>>5; uint bit_in_w=e->bit_offset&31u;
    ulong acc=(ulong)(bs_load_u32_le(w_bits,word_idx)>>bit_in_w); uint have=32u-bit_in_w;
    for (uint j=0;j<n;++j){
        if(have<k_bits){ulong nxt=(ulong)bs_load_u32_le(w_bits,++word_idx);acc|=nxt<<have;have+=32u;}
        uint sym=(uint)acc&input_mask;acc>>=k_bits;have-=k_bits;
        state=((state<<k_bits)|sym)&state_mask;
        int q=(int)sh_q[hash_state(state,l_bits)];
        int w=(int)(((long)e->eff[j>>5]*(long)q)>>16)+e->off[j>>5];
        out_q12[obase+j]=w;
    }
}

kernel void reduce_rows(
    device const float* partials [[buffer(0)]],
    device       float* y        [[buffer(1)]],
    constant     uint&  rows     [[buffer(2)]],
    constant     uint&  bpr      [[buffer(3)]],
    uint gidx [[thread_position_in_grid]])
{
    if (gidx >= rows) return;
    float acc=0.0f; uint base=gidx*bpr;
    for (uint b=0;b<bpr;++b) acc+=partials[base+b];
    y[gidx]=acc;
}

kernel void sizeof_probe(device uint* out [[buffer(0)]]) { out[0]=(uint)sizeof(BitsliceEntry); }
"#;

    struct Gpu {
        device: Device,
        queue: CommandQueue,
        k1_blk: ComputePipelineState,
        k1_dec: ComputePipelineState,
        k2_coop: ComputePipelineState,
        k2_dec: ComputePipelineState,
        reduce: ComputePipelineState,
        szprobe: ComputePipelineState,
    }

    impl Gpu {
        fn new() -> Option<Self> {
            let device = Device::system_default()?;
            let lib = match device.new_library_with_source(MSL, &CompileOptions::new()) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("[gate-coopwindow] compile error: {e}");
                    return None;
                }
            };
            let p = |n: &str| -> Option<ComputePipelineState> {
                let f = lib.get_function(n, None).ok()?;
                device.new_compute_pipeline_state_with_function(&f).ok()
            };
            let g = Self {
                k1_blk: p("k1_computed_blockthread")?,
                k1_dec: p("k1_computed_decode")?,
                k2_coop: p("k2_coop_windowed")?,
                k2_dec: p("k2_windowed_decode")?,
                reduce: p("reduce_rows")?,
                szprobe: p("sizeof_probe")?,
                queue: device.new_command_queue(),
                device,
            };
            let sz = g.probe_sizeof();
            assert_eq!(sz as usize, std::mem::size_of::<BitsliceEntry>(), "GPU sizeof(BitsliceEntry)={sz} != host {}", std::mem::size_of::<BitsliceEntry>());
            Some(g)
        }
        fn upload<T: Copy>(&self, d: &[T]) -> Buffer {
            let n = (d.len() * std::mem::size_of::<T>()).max(4);
            let b = self.device.new_buffer(n as NSUInteger, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(d.as_ptr() as *const u8, b.contents() as *mut u8, d.len() * std::mem::size_of::<T>());
            }
            b
        }
        fn alloc(&self, n: usize) -> Buffer {
            self.device.new_buffer(n.max(4) as NSUInteger, MTLResourceOptions::StorageModeShared)
        }
        fn upload_payload(&self, bits: &[u8]) -> Buffer {
            let n = bits.len().div_ceil(4) * 4 + 8;
            let b = self.alloc(n);
            unsafe {
                let d = b.contents() as *mut u8;
                std::ptr::write_bytes(d, 0, n);
                std::ptr::copy_nonoverlapping(bits.as_ptr(), d, bits.len());
            }
            b
        }
        fn probe_sizeof(&self) -> u32 {
            let out = self.alloc(4);
            let cmd = self.queue.new_command_buffer();
            let e = cmd.new_compute_command_encoder();
            e.set_compute_pipeline_state(&self.szprobe);
            e.set_buffer(0, Some(&out), 0);
            let one = MTLSize { width: 1, height: 1, depth: 1 };
            e.dispatch_thread_groups(one, one);
            e.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            unsafe { *(out.contents() as *const u32) }
        }

        // K1 decode-only identity
        #[allow(clippy::too_many_arguments)]
        fn k1_decode(&self, payload: &[u8], tbl: &[BitsliceEntry], quant: &[i16], total: usize, k: u32, l: u32) -> Vec<i32> {
            let w = self.upload_payload(payload);
            let out = self.alloc(total * 4);
            let t = self.upload(tbl);
            let nb = self.upload(&[tbl.len() as u32]);
            let kb = self.upload(&[k]);
            let lb = self.upload(&[l]);
            let q = self.upload(quant);
            let cmd = self.queue.new_command_buffer();
            let e = cmd.new_compute_command_encoder();
            e.set_compute_pipeline_state(&self.k1_dec);
            e.set_buffer(0, Some(&w), 0);
            e.set_buffer(1, Some(&out), 0);
            e.set_buffer(2, Some(&t), 0);
            e.set_buffer(3, Some(&nb), 0);
            e.set_buffer(4, Some(&kb), 0);
            e.set_buffer(5, Some(&lb), 0);
            e.set_buffer(6, Some(&q), 0);
            e.set_threadgroup_memory_length(0, ((1usize << l) * 2) as NSUInteger);
            let groups = MTLSize { width: (tbl.len() as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
            let tpg = MTLSize { width: 256, height: 1, depth: 1 };
            e.dispatch_thread_groups(groups, tpg);
            e.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            unsafe { std::slice::from_raw_parts(out.contents() as *const i32, total) }.to_vec()
        }

        // K2 windowed decode-only identity
        #[allow(clippy::too_many_arguments)]
        fn k2_decode(&self, payload: &[u8], tbl: &[BitsliceEntry], quant: &[i16], total: usize, k: u32, l: u32) -> Vec<i32> {
            const BG: u64 = 8;
            let w = self.upload_payload(payload);
            let out = self.alloc(total * 4);
            let t = self.upload(tbl);
            let nb = self.upload(&[tbl.len() as u32]);
            let kb = self.upload(&[k]);
            let lb = self.upload(&[l]);
            let q = self.upload(quant);
            let cmd = self.queue.new_command_buffer();
            let e = cmd.new_compute_command_encoder();
            e.set_compute_pipeline_state(&self.k2_dec);
            e.set_buffer(0, Some(&w), 0);
            e.set_buffer(1, Some(&out), 0);
            e.set_buffer(2, Some(&t), 0);
            e.set_buffer(3, Some(&nb), 0);
            e.set_buffer(4, Some(&kb), 0);
            e.set_buffer(5, Some(&lb), 0);
            e.set_buffer(6, Some(&q), 0);
            e.set_threadgroup_memory_length(0, ((1usize << l) * 2) as NSUInteger);
            let groups = MTLSize { width: (tbl.len() as u64).div_ceil(BG) as NSUInteger, height: 1, depth: 1 };
            let tpg = MTLSize { width: (BG * 32) as NSUInteger, height: 1, depth: 1 };
            e.dispatch_thread_groups(groups, tpg);
            e.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            unsafe { std::slice::from_raw_parts(out.contents() as *const i32, total) }.to_vec()
        }

        // K1 fused matvec
        #[allow(clippy::too_many_arguments)]
        fn k1_matvec(&self, b: &Bufs, rows: u32, l: u32) -> (Vec<f32>, f64, usize) {
            self.run_partials(&self.k1_blk, b, rows, l, /*coop=*/ false)
        }
        #[allow(clippy::too_many_arguments)]
        fn k2_matvec(&self, b: &Bufs, rows: u32, l: u32) -> (Vec<f32>, f64, usize) {
            self.run_partials(&self.k2_coop, b, rows, l, /*coop=*/ true)
        }

        fn run_partials(&self, pipe: &ComputePipelineState, b: &Bufs, rows: u32, l: u32, coop: bool) -> (Vec<f32>, f64, usize) {
            // returns (y, dt_best_over_internal, tpb)
            // coop: BG blocks per threadgroup (one 32-lane SIMD-group per block) so the quant
            // table stages once per TG and amortizes. BG=8 -> 256 threads/TG.
            const BG: u64 = 8;
            let tpb: u64 = if coop { BG * 32 } else { 256 };
            let dispatch = || {
                let cmd = self.queue.new_command_buffer();
                {
                    let e = cmd.new_compute_command_encoder();
                    e.set_compute_pipeline_state(pipe);
                    e.set_buffer(0, Some(&b.w), 0);
                    e.set_buffer(1, Some(&b.x), 0);
                    e.set_buffer(2, Some(&b.partials), 0);
                    e.set_buffer(3, Some(&b.tbl), 0);
                    e.set_buffer(4, Some(&b.nb), 0);
                    e.set_buffer(5, Some(&b.cols), 0);
                    e.set_buffer(6, Some(&b.k), 0);
                    e.set_buffer(7, Some(&b.l), 0);
                    e.set_buffer(8, Some(&b.q), 0);
                    if coop {
                        e.set_threadgroup_memory_length(0, ((1u64 << l) * 2) as NSUInteger); // sh_q i16
                        let groups = MTLSize { width: (b.n_blocks as u64).div_ceil(BG) as NSUInteger, height: 1, depth: 1 };
                        let t = MTLSize { width: tpb as NSUInteger, height: 1, depth: 1 };
                        e.dispatch_thread_groups(groups, t);
                    } else {
                        e.set_threadgroup_memory_length(0, ((1u64 << l) * 2) as NSUInteger);
                        let groups = MTLSize { width: (b.n_blocks as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
                        let t = MTLSize { width: 256, height: 1, depth: 1 };
                        e.dispatch_thread_groups(groups, t);
                    }
                    e.end_encoding();
                }
                {
                    let e = cmd.new_compute_command_encoder();
                    e.set_compute_pipeline_state(&self.reduce);
                    e.set_buffer(0, Some(&b.partials), 0);
                    e.set_buffer(1, Some(&b.y), 0);
                    e.set_buffer(2, Some(&b.rows), 0);
                    e.set_buffer(3, Some(&b.bpr), 0);
                    let groups = MTLSize { width: (rows as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
                    let t = MTLSize { width: 256, height: 1, depth: 1 };
                    e.dispatch_thread_groups(groups, t);
                    e.end_encoding();
                }
                cmd.commit();
                cmd.wait_until_completed();
            };
            // warm
            dispatch();
            let mut best = f64::INFINITY;
            for _ in 0..30 {
                let t0 = std::time::Instant::now();
                dispatch();
                let dt = t0.elapsed().as_secs_f64();
                if dt > 0.0 && dt < best {
                    best = dt;
                }
            }
            let y = unsafe { std::slice::from_raw_parts(b.y.contents() as *const f32, rows as usize) }.to_vec();
            (y, best, tpb as usize)
        }
    }

    struct Bufs {
        w: Buffer,
        x: Buffer,
        partials: Buffer,
        y: Buffer,
        tbl: Buffer,
        nb: Buffer,
        cols: Buffer,
        k: Buffer,
        l: Buffer,
        q: Buffer,
        rows: Buffer,
        bpr: Buffer,
        n_blocks: u32,
    }

    #[allow(clippy::too_many_arguments)]
    fn make_bufs(g: &Gpu, payload: &[u8], tbl: &[BitsliceEntry], quant: &[i16], rows: u32, cols: u32, k: u32, l: u32, x: &[f32]) -> Bufs {
        Bufs {
            w: g.upload_payload(payload),
            x: g.upload(x),
            partials: g.alloc(tbl.len() * 4),
            y: g.alloc(rows as usize * 4),
            tbl: g.upload(tbl),
            nb: g.upload(&[tbl.len() as u32]),
            cols: g.upload(&[cols]),
            k: g.upload(&[k]),
            l: g.upload(&[l]),
            q: g.upload(quant),
            rows: g.upload(&[rows]),
            bpr: g.upload(&[cols / 256]),
            n_blocks: tbl.len() as u32,
        }
    }

    fn check_fused(y: &[f32], want: &[i32], x: &[f32], out_f: usize, in_f: usize) -> (bool, f64) {
        let inv = 1.0f32 / 4096.0;
        let mut ok = true;
        let mut maxrel = 0.0f64;
        for r in (0..out_f).step_by(257) {
            let row = &want[r * in_f..(r + 1) * in_f];
            let mut acc = 0.0f32;
            for i in 0..in_f {
                acc += (row[i] as f32) * inv * x[i];
            }
            let denom = acc.abs().max(1e-3);
            let rel = ((y[r] - acc).abs() / denom) as f64;
            maxrel = maxrel.max(rel);
            if rel >= 1e-3 {
                ok = false;
            }
        }
        (ok, maxrel)
    }

    pub fn run() {
        let Some(g) = Gpu::new() else {
            println!("gate-coopwindow: no Metal / compile failed.");
            return;
        };
        let Some(refg) = BitsliceGpu::new() else {
            println!("gate-coopwindow: reference BitsliceGpu unavailable.");
            return;
        };
        let peak = StrandGpu::new().map(|s| s.bench_peak_bw(64 << 20, 5)).unwrap_or(f64::NAN);
        println!("== gate-coopwindow: computed-codebook + cooperative windowed decode ==");
        println!("  {}", machine_stamp());
        println!(
            "  device props: k1_blk maxTPT={} execWidth={}  k2_coop maxTPT={} execWidth={}",
            g.k1_blk.max_total_threads_per_threadgroup(),
            g.k1_blk.thread_execution_width(),
            g.k2_coop.max_total_threads_per_threadgroup(),
            g.k2_coop.thread_execution_width()
        );
        println!("  measured streaming peak: {:.1} GB/s", peak / 1e9);

        let (out_f, in_f) = (18944usize, 3584usize);
        let total = out_f * in_f;
        let x: Vec<f32> = (0..in_f).map(|i| (i as f32 * 0.05).sin()).collect();
        println!("  ffn_down {out_f}x{in_f} = {:.1}M weights\n", total as f64 / 1e6);

        for (cfg, label) in [(TrellisConfig::for_bpw(3.0), "k3 L7 (3-bit deploy)"), (TrellisConfig::for_bpw_l(2.0, 12), "k2 L12 (2-bit reopen)")] {
            let enc = synth_encoded(total, cfg.k_bits, 256);
            let tbl = bake_bitslice_entries(&enc, &cfg).expect("bake");
            let want = decode_tensor_fixed(&enc, &cfg);
            let quant = quantile_i16(cfg.l_bits);
            let cb = codebook_lut(cfg.l_bits);
            let lutref = cb; // reference fused uses the precomputed codebook i32

            // ---- IDENTITY: computed codebook == codebook_lut (and decode == reference) ----
            // 1) host check: quant[hash(s)] == codebook_lut[s] for all s.
            {
                let n = 1usize << cfg.l_bits;
                let mut ok = true;
                for s in 0..n {
                    let h = hash_host(s, cfg.l_bits);
                    if quant[h] as i32 != cb[s] {
                        ok = false;
                        break;
                    }
                }
                assert!(ok, "host computed codebook != codebook_lut at {label}");
            }
            // 2) GPU decode (computed, serial) == decode_tensor_fixed
            let got = g.k1_decode(&enc.bits, &tbl, &quant, total, cfg.k_bits, cfg.l_bits);
            let dec_ok = got == want;
            if !dec_ok {
                let idx = got.iter().zip(want.iter()).position(|(a, b)| a != b).unwrap_or(usize::MAX);
                println!("  !! K1 decode identity FAIL at {label}: first diff i={idx} GPU={} CPU={}", got.get(idx).copied().unwrap_or(0), want.get(idx).copied().unwrap_or(0));
            }
            // 3) GPU WINDOWED decode (parallel, no recurrence) == decode_tensor_fixed
            let gotw = g.k2_decode(&enc.bits, &tbl, &quant, total, cfg.k_bits, cfg.l_bits);
            let win_ok = gotw == want;
            if !win_ok {
                let idx = gotw.iter().zip(want.iter()).position(|(a, b)| a != b).unwrap_or(usize::MAX);
                println!("  !! K2 WINDOWED decode identity FAIL at {label}: first diff i={idx} GPU={} CPU={}", gotw.get(idx).copied().unwrap_or(0), want.get(idx).copied().unwrap_or(0));
            }

            // ---- reference fused B=1 (existing kernel) baseline ----
            let ref_dt = refg.bench_matvec(&enc.bits, &tbl, lutref, out_f as u32, in_f as u32, cfg.k_bits, cfg.l_bits, &x, 30);
            let ref_gws = total as f64 / ref_dt / 1e9;

            // ---- K1 computed-codebook fused ----
            let bufs = make_bufs(&g, &enc.bits, &tbl, &quant, out_f as u32, in_f as u32, cfg.k_bits, cfg.l_bits, &x);
            let (y1, dt1, _) = g.k1_matvec(&bufs, out_f as u32, cfg.l_bits);
            let (ok1, rel1) = check_fused(&y1, &want, &x, out_f, in_f);
            let g1 = total as f64 / dt1 / 1e9;

            // ---- K2 cooperative windowed fused ----
            let (y2, dt2, tpb2) = g.k2_matvec(&bufs, out_f as u32, cfg.l_bits);
            let (ok2, rel2) = check_fused(&y2, &want, &x, out_f, in_f);
            let g2 = total as f64 / dt2 / 1e9;

            let payload_bytes = (total as f64) * (cfg.k_bits as f64) / 8.0;
            let io = ((in_f + out_f) * 4) as f64;
            let tbl_bytes = (tbl.len() * std::mem::size_of::<BitsliceEntry>()) as f64;
            let bw = |dt: f64| 100.0 * ((payload_bytes + tbl_bytes + io) / dt) / peak;

            println!("  [{label}]  serial-decode-id={}  windowed-decode-id={}", if dec_ok { "PASS" } else { "FAIL" }, if win_ok { "PASS" } else { "FAIL" });
            println!("    reference fused B=1 : {:>7.3} ms  {:>6.2} Gw/s  {:>5.1}% peak  (1.00x)", ref_dt * 1e3, ref_gws, bw(ref_dt));
            println!(
                "    K1 computed-cbook   : {:>7.3} ms  {:>6.2} Gw/s  {:>5.1}% peak  ({:.2}x)  fused-id={} (rel {:.1e})",
                dt1 * 1e3,
                g1,
                bw(dt1),
                g1 / ref_gws,
                if ok1 { "PASS" } else { "FAIL" },
                rel1
            );
            println!(
                "    K2 coop-windowed tpb{tpb2}: {:>6.3} ms  {:>6.2} Gw/s  {:>5.1}% peak  ({:.2}x)  fused-id={} (rel {:.1e})\n",
                dt2 * 1e3,
                g2,
                bw(dt2),
                g2 / ref_gws,
                if ok2 { "PASS" } else { "FAIL" },
                rel2
            );
        }
        println!("  NOTE: %peak uses payload+table+io traffic. Computed codebook removes the per-thread");
        println!("  data-dependent 2^L i32 shmem LUT (L12: 16KB->8KB i16 quant). K2 tests within-block parallelism.");
    }

    fn hash_host(s: usize, l_bits: u32) -> usize {
        let mask = (1usize << l_bits) - 1;
        let r = (l_bits / 2).max(1) as usize;
        let mut h = s & mask;
        h = (h ^ (h >> r)) & mask;
        h = h.wrapping_mul(0x2545_F491_4F6C_DD1D) & mask;
        h = (h ^ (h >> r)) & mask;
        h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15) & mask;
        h & mask
    }
}
