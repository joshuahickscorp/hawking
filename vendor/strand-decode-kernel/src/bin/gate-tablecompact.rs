// SCRATCH / DIAGNOSTIC (untracked). Additive compact-table experiment.
//
// Tests whether storing a COMPACT per-block entry (scale_q + raw 6-bit sub-scale
// codes) and expanding eff[sb]/off[sb] INSIDE the kernel reduces B=1 fused memory
// traffic enough to raise throughput — bit-identically.
//
// This file is fully self-contained: its own Metal shader source, its own compact
// entry struct + bake, its own GPU wrapper. It touches NO tracked file. The
// reference (expanded 84-B table) path is driven through the existing public
// strand_decode_kernel::metal API for an apples-to-apples comparison on the same
// shapes the prior agent used.

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("gate-tablecompact: Metal is macOS-only; nothing to gate on this target.");
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod macos {
    use metal::{
        Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
        MTLSize, NSUInteger,
    };

    use strand_decode_kernel::block_walk::gate_proto::{machine_stamp, synth_encoded};
    use strand_decode_kernel::metal::{bake_bitslice_entries, BitsliceEntry, BitsliceGpu, StrandGpu};
    use strand_quant::codebook::codebook_lut;
    use strand_quant::decode::{decode_tensor_fixed, eff_min_q, eff_scale_q};
    use strand_quant::encode::{n_sub_blocks, unpack_sub_scales, EncodedTensor};
    use strand_quant::TrellisConfig;

    // ============================================================================
    // Compact entry. 40 B. The 8 mult codes (u8 each, only 6 bits used) pack into
    // 2 u32s; ditto the 8 min codes. We never have more than 8 sub-blocks here
    // because n<=256 and SUB_BLOCK=32 => n_sub<=8.
    // ============================================================================
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct CompactEntry {
        bit_offset: u32,
        init_state: u32,
        out_off: u32,
        n: u32,
        scale_q: i32,
        min_base_q: i32,
        mult_codes: [u32; 2], // 8 u8 codes little-endian packed
        min_codes: [u32; 2],
    }

    const _: () = assert!(std::mem::size_of::<CompactEntry>() == 40);

    #[inline]
    fn pack8(codes: &[u8]) -> [u32; 2] {
        let mut b = [0u8; 8];
        for (i, &c) in codes.iter().enumerate().take(8) {
            b[i] = c;
        }
        [
            u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        ]
    }

    /// Compact bake. We derive bit_offset / init_state / out_off / n from the
    /// EXISTING reference bake so the bitstream addressing and tail-biting walk are
    /// byte-for-byte identical — the only thing this experiment changes is HOW the
    /// per-sub-block scale/offset is stored (raw codes vs pre-expanded). The codes
    /// come straight from the EncodedTensor (scale_q, sub_scales, min_base_q, mins),
    /// matching decode_tensor_fixed exactly.
    fn bake_compact(enc: &EncodedTensor, cfg: &TrellisConfig) -> Option<Vec<CompactEntry>> {
        let expanded = bake_bitslice_entries(enc, cfg)?; // None if any block n>256
        assert_eq!(expanded.len(), enc.blocks.len());
        let has_affine = enc.has_affine_min;
        let mut out = Vec::with_capacity(enc.blocks.len());
        for (ref_e, blk) in expanded.iter().zip(enc.blocks.iter()) {
            let n_sub = n_sub_blocks(blk.n as usize);
            assert!(n_sub <= 8, "compact entry holds <=8 sub-blocks (n<=256)");
            let mult_codes = unpack_sub_scales(&blk.sub_scales, n_sub);
            let min_codes = if has_affine {
                unpack_sub_scales(&blk.mins, n_sub)
            } else {
                Vec::new()
            };
            out.push(CompactEntry {
                bit_offset: ref_e.bit_offset,
                init_state: ref_e.init_state,
                out_off: ref_e.out_off,
                n: ref_e.n as u32,
                scale_q: blk.scale_q,
                min_base_q: if has_affine { blk.min_base_q } else { 0 },
                mult_codes: pack8(&mult_codes),
                min_codes: pack8(&min_codes),
            });
        }
        Some(out)
    }

    /// Host-side mirror of the in-kernel expansion, used only as a CPU cross-check
    /// of the eff/off formulas before trusting the GPU (belt-and-suspenders; the
    /// GPU identity vs decode_tensor_fixed is the real gate).
    #[allow(dead_code)]
    fn host_eff_off(e: &CompactEntry, sb: usize) -> (i32, i32) {
        let mb = e.mult_codes[sb >> 2].to_le_bytes();
        let mult_code = mb[sb & 3];
        let nb = e.min_codes[sb >> 2].to_le_bytes();
        let min_code = nb[sb & 3];
        (
            eff_scale_q(e.scale_q, mult_code),
            eff_min_q(e.min_base_q, min_code),
        )
    }

    // ============================================================================
    // Self-contained Metal shader: compact-entry kernels (decode + gemv partials),
    // plus a sizeof probe and the SAME row-reduce the real path uses (copied so we
    // don't reach into the library's private pipelines).
    //
    // EXPANSION MATH (must match strand-quant/src/decode.rs EXACTLY):
    //   eff_scale_q(scale_q, code) = (scale_q * ((code & 0x3F) + 1)) >> 6
    //   eff_min_q(min_base_q, code):
    //       mag = code & 0x1F; if mag==0 -> 0
    //       base = |min_base_q|; signed = (code&0x20)? base*mag : -(base*mag)
    //       result = signed / 31   (truncating toward zero, like Rust i64 '/')
    //   per-weight: w = (eff[sb] * q) >> 16 + off[sb]      (SCALE_SHIFT=16)
    // ============================================================================
    const COMPACT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct CompactEntry {
    uint bit_offset;
    uint init_state;
    uint out_off;
    uint n;
    int  scale_q;
    int  min_base_q;
    uint mult_codes[2];   // 8 packed u8 codes
    uint min_codes[2];
};

static inline uint bs_load_u32_le(device const uchar* p, uint widx) {
    uint b = widx << 2;
    return (uint)p[b]
         | ((uint)p[b + 1] << 8)
         | ((uint)p[b + 2] << 16)
         | ((uint)p[b + 3] << 24);
}

static inline uchar code_at(const thread uint codes[2], uint sb) {
    uint word = codes[sb >> 2];
    return (uchar)((word >> ((sb & 3u) * 8u)) & 0xFFu);
}

// eff_scale_q: (scale_q * ((code & 0x3F) + 1)) >> 6
static inline int expand_eff(int scale_q, uchar code) {
    long mult = (long)((uint)(code & 0x3Fu) + 1u);
    return (int)(((long)scale_q * mult) >> 6);
}

// eff_min_q: see formula above. Division is truncating toward zero.
static inline int expand_off(int min_base_q, uchar code) {
    long mag = (long)(code & 0x1Fu);
    if (mag == 0) return 0;
    long base = (long)(uint)abs(min_base_q);   // unsigned_abs
    long signed_v = (code & 0x20u) ? (base * mag) : -(base * mag);
    return (int)(signed_v / 31);
}

kernel void compact_entry_sizeof(device uint* out [[buffer(0)]]) {
    out[0] = (uint)sizeof(CompactEntry);
}

// ---- decode (Q12 stream), compact entry --------------------------------------
kernel void compact_decode(
    device   const uchar*         w_bits   [[buffer(0)]],
    device         int*           out_q12  [[buffer(1)]],
    device   const CompactEntry*  tbl      [[buffer(2)]],
    constant       uint&          n_blocks [[buffer(3)]],
    constant       uint&          k_bits   [[buffer(4)]],
    constant       uint&          l_bits   [[buffer(5)]],
    device   const int*           lut_q12  [[buffer(6)]],
    threadgroup    int*           sh_lut   [[threadgroup(0)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint gidx [[thread_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]])
{
    uint lut_n = 1u << l_bits;
    for (uint s = tid; s < lut_n; s += tgs) sh_lut[s] = lut_q12[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gidx >= n_blocks) return;

    uint state_mask = lut_n - 1u;
    uint input_mask = (1u << k_bits) - 1u;

    device const CompactEntry* e = &tbl[gidx];
    uint mult[2] = { e->mult_codes[0], e->mult_codes[1] };
    uint mins[2] = { e->min_codes[0],  e->min_codes[1]  };
    int  scale_q = e->scale_q;
    int  min_base_q = e->min_base_q;

    uint  state    = e->init_state & state_mask;
    uint  n        = e->n;
    uint  obase    = e->out_off;
    uint  word_idx = e->bit_offset >> 5;
    uint  bit_in_w = e->bit_offset & 31u;
    ulong acc      = (ulong)(bs_load_u32_le(w_bits, word_idx) >> bit_in_w);
    uint  have     = 32u - bit_in_w;

    int   es = expand_eff(scale_q, code_at(mult, 0));
    int   of = expand_off(min_base_q, code_at(mins, 0));
    uint  cur_sb = 0u;

    for (uint j = 0; j < n; ++j) {
        if (have < k_bits) {
            ulong nxt = (ulong)bs_load_u32_le(w_bits, ++word_idx);
            acc |= nxt << have;
            have += 32u;
        }
        uint sym = (uint)acc & input_mask;
        acc >>= k_bits;
        have -= k_bits;
        state = ((state << k_bits) | sym) & state_mask;

        uint sb = j >> 5;
        if (sb != cur_sb) {                       // expand once per sub-block
            cur_sb = sb;
            es = expand_eff(scale_q, code_at(mult, sb));
            of = expand_off(min_base_q, code_at(mins, sb));
        }
        int q = sh_lut[state];
        int w = (int)(((long)es * (long)q) >> 16) + of;
        out_q12[obase + j] = w;
    }
}

// ---- gemv partials (B=1 fused), compact entry --------------------------------
kernel void compact_gemv_partials(
    device   const uchar*         w_bits   [[buffer(0)]],
    device   const float*         x        [[buffer(1)]],
    device         float*         partials [[buffer(2)]],
    device   const CompactEntry*  tbl      [[buffer(3)]],
    constant       uint&          n_blocks [[buffer(4)]],
    constant       uint&          cols     [[buffer(5)]],
    constant       uint&          k_bits   [[buffer(6)]],
    constant       uint&          l_bits   [[buffer(7)]],
    device   const int*           lut_q12  [[buffer(8)]],
    threadgroup    int*           sh_lut   [[threadgroup(0)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint gidx [[thread_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]])
{
    uint lut_n = 1u << l_bits;
    for (uint s = tid; s < lut_n; s += tgs) sh_lut[s] = lut_q12[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gidx >= n_blocks) return;

    uint state_mask = lut_n - 1u;
    uint input_mask = (1u << k_bits) - 1u;
    const float Q12_TO_F32 = 1.0f / 4096.0f;

    device const CompactEntry* e = &tbl[gidx];
    uint mult[2] = { e->mult_codes[0], e->mult_codes[1] };
    uint mins[2] = { e->min_codes[0],  e->min_codes[1]  };
    int  scale_q = e->scale_q;
    int  min_base_q = e->min_base_q;

    uint  state    = e->init_state & state_mask;
    uint  n        = e->n;
    uint  col0     = e->out_off % cols;
    uint  word_idx = e->bit_offset >> 5;
    uint  bit_in_w = e->bit_offset & 31u;
    ulong acc      = (ulong)(bs_load_u32_le(w_bits, word_idx) >> bit_in_w);
    uint  have     = 32u - bit_in_w;

    int   es = expand_eff(scale_q, code_at(mult, 0));
    int   of = expand_off(min_base_q, code_at(mins, 0));
    uint  cur_sb = 0u;

    float partial = 0.0f;
    for (uint j = 0; j < n; ++j) {
        if (have < k_bits) {
            ulong nxt = (ulong)bs_load_u32_le(w_bits, ++word_idx);
            acc |= nxt << have;
            have += 32u;
        }
        uint sym = (uint)acc & input_mask;
        acc >>= k_bits;
        have -= k_bits;
        state = ((state << k_bits) | sym) & state_mask;

        uint sb = j >> 5;
        if (sb != cur_sb) {
            cur_sb = sb;
            es = expand_eff(scale_q, code_at(mult, sb));
            of = expand_off(min_base_q, code_at(mins, sb));
        }
        int q = sh_lut[state];
        int w = (int)(((long)es * (long)q) >> 16) + of;
        partial += (float)w * Q12_TO_F32 * x[col0 + j];
    }
    partials[gidx] = partial;
}

// row reduce (copy of strand_bitslice_reduce_rows)
kernel void compact_reduce_rows(
    device const float* partials [[buffer(0)]],
    device       float* y        [[buffer(1)]],
    constant     uint&  rows     [[buffer(2)]],
    constant     uint&  bpr      [[buffer(3)]],
    uint gidx [[thread_position_in_grid]])
{
    if (gidx >= rows) return;
    float acc = 0.0f;
    uint base = gidx * bpr;
    for (uint b = 0; b < bpr; ++b) acc += partials[base + b];
    y[gidx] = acc;
}
"#;

    struct CompactGpu {
        device: Device,
        queue: CommandQueue,
        decode: ComputePipelineState,
        gemv_partials: ComputePipelineState,
        reduce_rows: ComputePipelineState,
        sizeof_probe: ComputePipelineState,
    }

    impl CompactGpu {
        fn new() -> Option<Self> {
            let device = Device::system_default()?;
            let lib = match device.new_library_with_source(COMPACT_MSL, &CompileOptions::new()) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("[gate-tablecompact] compact shader compile error: {e}");
                    return None;
                }
            };
            let pipe = |name: &str| -> Option<ComputePipelineState> {
                let f = lib.get_function(name, None).ok()?;
                device.new_compute_pipeline_state_with_function(&f).ok()
            };
            let decode = pipe("compact_decode")?;
            let gemv_partials = pipe("compact_gemv_partials")?;
            let reduce_rows = pipe("compact_reduce_rows")?;
            let sizeof_probe = pipe("compact_entry_sizeof")?;
            let queue = device.new_command_queue();
            let gpu = Self { device, queue, decode, gemv_partials, reduce_rows, sizeof_probe };
            let gpu_sz = gpu.gpu_sizeof();
            assert_eq!(
                gpu_sz as usize,
                std::mem::size_of::<CompactEntry>(),
                "GPU sizeof(CompactEntry)={gpu_sz} != host {} — tbl stride would diverge",
                std::mem::size_of::<CompactEntry>()
            );
            Some(gpu)
        }

        fn upload<T: Copy>(&self, data: &[T]) -> Buffer {
            let byte_len = (data.len() * std::mem::size_of::<T>()).max(4);
            let buf = self
                .device
                .new_buffer(byte_len as NSUInteger, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    buf.contents() as *mut u8,
                    data.len() * std::mem::size_of::<T>(),
                );
            }
            buf
        }

        fn alloc(&self, byte_len: usize) -> Buffer {
            self.device
                .new_buffer(byte_len.max(4) as NSUInteger, MTLResourceOptions::StorageModeShared)
        }

        fn upload_payload(&self, bits: &[u8]) -> Buffer {
            let padded_len = bits.len().div_ceil(4) * 4 + 8;
            let buf = self.alloc(padded_len);
            unsafe {
                let dst = buf.contents() as *mut u8;
                std::ptr::write_bytes(dst, 0, padded_len);
                std::ptr::copy_nonoverlapping(bits.as_ptr(), dst, bits.len());
            }
            buf
        }

        fn gpu_sizeof(&self) -> u32 {
            let out = self.alloc(4);
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.sizeof_probe);
            enc.set_buffer(0, Some(&out), 0);
            let one = MTLSize { width: 1, height: 1, depth: 1 };
            enc.dispatch_thread_groups(one, one);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            unsafe { *(out.contents() as *const u32) }
        }

        #[allow(clippy::too_many_arguments)]
        fn decode_q12(
            &self,
            payload: &[u8],
            tbl: &[CompactEntry],
            lut: &[i32],
            total: usize,
            k_bits: u32,
            l_bits: u32,
        ) -> Vec<i32> {
            let w_buf = self.upload_payload(payload);
            let out_buf = self.alloc(total * 4);
            let tbl_buf = self.upload(tbl);
            let nb_buf = self.upload(&[tbl.len() as u32]);
            let k_buf = self.upload(&[k_bits]);
            let l_buf = self.upload(&[l_bits]);
            let lut_buf = self.upload(lut);
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.decode);
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(&out_buf), 0);
            enc.set_buffer(2, Some(&tbl_buf), 0);
            enc.set_buffer(3, Some(&nb_buf), 0);
            enc.set_buffer(4, Some(&k_buf), 0);
            enc.set_buffer(5, Some(&l_buf), 0);
            enc.set_buffer(6, Some(&lut_buf), 0);
            enc.set_threadgroup_memory_length(0, ((1usize << l_bits) * 4) as NSUInteger);
            let groups = MTLSize {
                width: (tbl.len() as u64).div_ceil(256) as NSUInteger,
                height: 1,
                depth: 1,
            };
            let tpg = MTLSize { width: 256, height: 1, depth: 1 };
            enc.dispatch_thread_groups(groups, tpg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let ptr = out_buf.contents() as *const i32;
            unsafe { std::slice::from_raw_parts(ptr, total) }.to_vec()
        }

        // ---- fused B=1 matvec (gemv partials + reduce) -------------------------
        #[allow(clippy::too_many_arguments)]
        fn matvec_bufs(
            &self,
            payload: &[u8],
            tbl: &[CompactEntry],
            lut: &[i32],
            rows: u32,
            cols: u32,
            k_bits: u32,
            l_bits: u32,
            x: &[f32],
        ) -> CompactBufs {
            CompactBufs {
                w: self.upload_payload(payload),
                x: self.upload(x),
                partials: self.alloc(tbl.len() * 4),
                y: self.alloc(rows as usize * 4),
                tbl: self.upload(tbl),
                n_blocks: self.upload(&[tbl.len() as u32]),
                cols: self.upload(&[cols]),
                k: self.upload(&[k_bits]),
                l: self.upload(&[l_bits]),
                lut: self.upload(lut),
                rows: self.upload(&[rows]),
                bpr: self.upload(&[cols / 256]),
            }
        }

        fn matvec_dispatch(&self, b: &CompactBufs, n_blocks: u32, rows: u32, l_bits: u32) {
            let cmd = self.queue.new_command_buffer();
            {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.gemv_partials);
                enc.set_buffer(0, Some(&b.w), 0);
                enc.set_buffer(1, Some(&b.x), 0);
                enc.set_buffer(2, Some(&b.partials), 0);
                enc.set_buffer(3, Some(&b.tbl), 0);
                enc.set_buffer(4, Some(&b.n_blocks), 0);
                enc.set_buffer(5, Some(&b.cols), 0);
                enc.set_buffer(6, Some(&b.k), 0);
                enc.set_buffer(7, Some(&b.l), 0);
                enc.set_buffer(8, Some(&b.lut), 0);
                enc.set_threadgroup_memory_length(0, ((1usize << l_bits) * 4) as NSUInteger);
                let groups = MTLSize {
                    width: (n_blocks as u64).div_ceil(256) as NSUInteger,
                    height: 1,
                    depth: 1,
                };
                let tpg = MTLSize { width: 256, height: 1, depth: 1 };
                enc.dispatch_thread_groups(groups, tpg);
                enc.end_encoding();
            }
            {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.reduce_rows);
                enc.set_buffer(0, Some(&b.partials), 0);
                enc.set_buffer(1, Some(&b.y), 0);
                enc.set_buffer(2, Some(&b.rows), 0);
                enc.set_buffer(3, Some(&b.bpr), 0);
                let groups = MTLSize {
                    width: (rows as u64).div_ceil(256) as NSUInteger,
                    height: 1,
                    depth: 1,
                };
                let tpg = MTLSize { width: 256, height: 1, depth: 1 };
                enc.dispatch_thread_groups(groups, tpg);
                enc.end_encoding();
            }
            cmd.commit();
            cmd.wait_until_completed();
        }

        #[allow(clippy::too_many_arguments)]
        fn matvec(
            &self,
            payload: &[u8],
            tbl: &[CompactEntry],
            lut: &[i32],
            rows: u32,
            cols: u32,
            k_bits: u32,
            l_bits: u32,
            x: &[f32],
        ) -> Vec<f32> {
            let bufs = self.matvec_bufs(payload, tbl, lut, rows, cols, k_bits, l_bits, x);
            self.matvec_dispatch(&bufs, tbl.len() as u32, rows, l_bits);
            let ptr = bufs.y.contents() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, rows as usize) }.to_vec()
        }

        #[allow(clippy::too_many_arguments)]
        fn bench_matvec(
            &self,
            payload: &[u8],
            tbl: &[CompactEntry],
            lut: &[i32],
            rows: u32,
            cols: u32,
            k_bits: u32,
            l_bits: u32,
            x: &[f32],
            iters: usize,
        ) -> f64 {
            let bufs = self.matvec_bufs(payload, tbl, lut, rows, cols, k_bits, l_bits, x);
            let mut best = f64::INFINITY;
            for _ in 0..iters {
                let t0 = std::time::Instant::now();
                self.matvec_dispatch(&bufs, tbl.len() as u32, rows, l_bits);
                let dt = t0.elapsed().as_secs_f64();
                if dt > 0.0 && dt < best {
                    best = dt;
                }
            }
            best
        }
    }

    struct CompactBufs {
        w: Buffer,
        x: Buffer,
        partials: Buffer,
        y: Buffer,
        tbl: Buffer,
        n_blocks: Buffer,
        cols: Buffer,
        k: Buffer,
        l: Buffer,
        lut: Buffer,
        rows: Buffer,
        bpr: Buffer,
    }

    // ============================================================================
    // Identity gate: compact decode == decode_tensor_fixed, across configs/variants
    // including affine-min (exercises off[]) and tail-biting.
    // ============================================================================
    fn identity(gpu: &CompactGpu) -> Result<usize, String> {
        use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};
        let configs = [
            (TrellisConfig::for_bpw(3.0), "k3 L7"),
            (TrellisConfig::for_bpw(2.0), "k2 L6"),
            (TrellisConfig::for_bpw(4.0), "k4 L8"),
            (TrellisConfig::for_bpw_l(2.0, 12), "k2 L12"),
            (TrellisConfig::for_bpw_l(2.0, 5), "k2 L5 (fold)"),
            (TrellisConfig::for_bpw_l(4.0, 4), "k4 L4 (fold)"),
        ];
        let mut checked = 0usize;
        for (cfg, label) in &configs {
            let lut = codebook_lut(cfg.l_bits);
            for seed in 0..16u64 {
                let n = 1 + (seed as usize * 211) % 4096;
                let w: Vec<f32> = (0..n)
                    .map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5)
                    .collect();
                let variants = [
                    encode_tensor(&w, cfg),
                    encode_tensor_with(&w, cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
                    encode_tensor_with(&w, cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
                    encode_tensor_with(
                        &w,
                        cfg,
                        &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
                    ),
                ];
                for enc in &variants {
                    let Some(tbl) = bake_compact(enc, cfg) else {
                        // n>256 only when block_len>256; synth/encode uses 256, so this
                        // shouldn't trip, but guard anyway.
                        continue;
                    };
                    // host-formula cross-check on block 0 (cheap)
                    if let (Some(ce), Some(blk)) = (tbl.first(), enc.blocks.first()) {
                        let n_sub = n_sub_blocks(blk.n as usize);
                        let mults = unpack_sub_scales(&blk.sub_scales, n_sub);
                        for (sb, &m) in mults.iter().enumerate() {
                            let (he, _ho) = host_eff_off(ce, sb);
                            let want = eff_scale_q(blk.scale_q, m);
                            if he != want {
                                return Err(format!(
                                    "host expand mismatch {label} sb={sb}: {he} != {want}"
                                ));
                            }
                        }
                    }
                    let got = gpu.decode_q12(&enc.bits, &tbl, lut, enc.total, cfg.k_bits, cfg.l_bits);
                    let want = decode_tensor_fixed(enc, cfg);
                    if got != want {
                        let idx = got
                            .iter()
                            .zip(want.iter())
                            .position(|(a, b)| a != b)
                            .unwrap_or(usize::MAX);
                        let (g, r) = if idx < got.len() { (got[idx], want[idx]) } else { (0, 0) };
                        return Err(format!(
                            "IDENTITY VIOLATION {label} n={n} seed={seed} tail={} affine={}: \
                             first diff @ i={idx} GPU={g} vs CPU={r}",
                            enc.tail_biting, enc.has_affine_min
                        ));
                    }
                    checked += 1;
                }
            }
        }
        Ok(checked)
    }

    pub fn run() {
        let Some(cgpu) = CompactGpu::new() else {
            println!("gate-tablecompact: no Metal device / compact shader compile failed.");
            return;
        };
        let Some(rgpu) = BitsliceGpu::new() else {
            println!("gate-tablecompact: BitsliceGpu (reference) unavailable.");
            return;
        };
        println!("== gate-tablecompact: compact per-block table experiment ==");
        println!("  {}", machine_stamp());
        let host_ref = std::mem::size_of::<BitsliceEntry>();
        let host_compact = std::mem::size_of::<CompactEntry>();
        println!(
            "  sizeof: reference BitsliceEntry = {host_ref} B (GPU {}), compact CompactEntry = {host_compact} B (GPU {})",
            rgpu.gpu_entry_sizeof(),
            cgpu.gpu_sizeof(),
        );

        // ---- identity gate FIRST ----
        print!("  identity (compact decode == decode_tensor_fixed): ");
        match identity(&cgpu) {
            Ok(n) => println!("PASS — {n} config x variant x seed cells byte-identical"),
            Err(e) => {
                println!("FAIL");
                println!("  !! {e}");
                println!("  VERDICT: compact expansion is NOT exact — do not ship. (finding)");
                return;
            }
        }

        // ---- bench, same shape the prior agent used: ffn_down 18944x3584 ----
        let (out_f, in_f) = (18944usize, 3584usize);
        let total = out_f * in_f;
        let peak = StrandGpu::new().map(|g| g.bench_peak_bw(64 << 20, 5)).unwrap_or(f64::NAN);
        println!(
            "\n  ffn_down {out_f}x{in_f} = {:.1}M weights;  measured streaming peak {:.1} GB/s",
            total as f64 / 1e6,
            peak / 1e9
        );

        let x: Vec<f32> = (0..in_f).map(|i| (i as f32 * 0.05).sin()).collect();

        let cells = [
            (TrellisConfig::for_bpw(3.0), "k3 L7 (3-bit deploy)"),
            (TrellisConfig::for_bpw_l(2.0, 12), "k2 L12 (2-bit reopen)"),
        ];

        println!(
            "\n  {:<24} {:>10} {:>10} {:>8}   {:>10} {:>10} {:>8}   {:>7}",
            "config", "ref ms", "ref Gw/s", "ref %pk", "cmp ms", "cmp Gw/s", "cmp %pk", "ratio"
        );

        for (cfg, label) in cells {
            let enc = synth_encoded(total, cfg.k_bits, 256);
            let lut = codebook_lut(cfg.l_bits);

            // reference (expanded 84-B) path via the library
            let ref_tbl = bake_bitslice_entries(&enc, &cfg).expect("ref bake");
            let want = decode_tensor_fixed(&enc, &cfg);

            // compact path
            let cmp_tbl = bake_compact(&enc, &cfg).expect("compact bake");

            // ---- identity on the bench shape, BOTH paths ----
            let cmp_decode = cgpu.decode_q12(&enc.bits, &cmp_tbl, lut, total, cfg.k_bits, cfg.l_bits);
            assert_eq!(cmp_decode, want, "compact decode identity violated on bench shape at {label}");

            // fused y identity (compact) vs CPU, probe rows
            let y_cmp = cgpu.matvec(
                &enc.bits, &cmp_tbl, lut, out_f as u32, in_f as u32, cfg.k_bits, cfg.l_bits, &x,
            );
            let inv = 1.0f32 / 4096.0;
            for r in (0..out_f).step_by(997) {
                let row = &want[r * in_f..(r + 1) * in_f];
                let mut acc = 0.0f32;
                for i in 0..in_f {
                    acc += (row[i] as f32) * inv * x[i];
                }
                let denom = acc.abs().max(1e-3);
                assert!(
                    (y_cmp[r] - acc).abs() / denom < 1e-3,
                    "compact fused y diverged at {label} row {r}: GPU {} vs CPU {acc}",
                    y_cmp[r]
                );
            }

            // ---- timings ----
            let dt_ref = rgpu.bench_matvec(
                &enc.bits, &ref_tbl, lut, out_f as u32, in_f as u32, cfg.k_bits, cfg.l_bits, &x, 20,
            );
            let dt_cmp = cgpu.bench_matvec(
                &enc.bits, &cmp_tbl, lut, out_f as u32, in_f as u32, cfg.k_bits, cfg.l_bits, &x, 20,
            );

            let ref_gws = total as f64 / dt_ref / 1e9;
            let cmp_gws = total as f64 / dt_cmp / 1e9;

            // modeled traffic (payload + table + x/y). table term differs.
            let payload = (total as f64) * (cfg.k_bits as f64) / 8.0;
            let io = ((in_f + out_f) * 4) as f64;
            let ref_tbl_bytes = (ref_tbl.len() * host_ref) as f64;
            let cmp_tbl_bytes = (cmp_tbl.len() * host_compact) as f64;
            let ref_bytes = payload + ref_tbl_bytes + io;
            let cmp_bytes = payload + cmp_tbl_bytes + io;
            let ref_pct = 100.0 * (ref_bytes / dt_ref) / peak;
            let cmp_pct = 100.0 * (cmp_bytes / dt_cmp) / peak;
            let ratio = cmp_gws / ref_gws;

            println!(
                "  {:<24} {:>10.3} {:>10.2} {:>7.1}%   {:>10.3} {:>10.2} {:>7.1}%   {:>6.2}x",
                label, dt_ref * 1e3, ref_gws, ref_pct, dt_cmp * 1e3, cmp_gws, cmp_pct, ratio
            );

            // traffic model detail
            let nblk = ref_tbl.len();
            println!(
                "      table: {host_ref} B/blk -> {host_compact} B/blk over {nblk} blocks  \
                 ({:.1} MB -> {:.1} MB).  modeled fused traffic {:.1} MB -> {:.1} MB ({:+.1}%).  \
                 table share of traffic {:.0}% -> {:.0}%",
                ref_tbl_bytes / 1e6,
                cmp_tbl_bytes / 1e6,
                ref_bytes / 1e6,
                cmp_bytes / 1e6,
                100.0 * (cmp_bytes - ref_bytes) / ref_bytes,
                100.0 * ref_tbl_bytes / ref_bytes,
                100.0 * cmp_tbl_bytes / cmp_bytes,
            );
        }

        println!(
            "\n  VERDICT RULE (committed before run): compact raises B=1 fused throughput >~1.2x at\n\
             \x20 k=2 (the table-heaviest case) => table was binding, BUILD IT FOR REAL (then compact\n\
             \x20 further + cooperative shared-mem payload loads). <1.2x => table wasn't the bottleneck;\n\
             \x20 payload-read coalescing is the next thing."
        );
    }
}
