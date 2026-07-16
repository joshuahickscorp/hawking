#![allow(unsafe_code)]

use metal::{Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize, NSUInteger};

use strand_quant::codebook::codebook_lut;
use strand_quant::decode::eff_scale_q;
use strand_quant::encode::{n_sub_blocks, unpack_sub_scales, EncodedTensor};
use strand_quant::format::TensorHeaderV2;

use crate::loader::StrandModel;

const GEMV_MSL: &str = include_str!("../shaders/strand_trellis_gemv.metal");

const PROBE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct BlockEntry {
    uint  bit_offset;
    uint  init_state;
    int   scale_q;
    int   eff[8];
    ushort n;
    ushort d;
    uint  _pad;
};
kernel void strand_blockentry_sizeof(device uint* out [[buffer(0)]]) {
    out[0] = (uint)sizeof(BlockEntry);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockEntry {
    pub bit_offset: u32,

    pub init_state: u32,

    pub scale_q: i32,

    pub eff: [i32; 8],

    pub n: u16,

    pub d: u16,

    pub _pad: u32,
}

pub struct StrandGpu {
    device: Device,
    queue: CommandQueue,
    fused: ComputePipelineState,

    #[allow(dead_code)]
    predec: ComputePipelineState,
    blockentry_sizeof: ComputePipelineState,
}

impl StrandGpu {
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let opts = CompileOptions::new();

        let lib = match device.new_library_with_source(GEMV_MSL, &opts) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[strand-decode-kernel] GEMV shader compile error: {e}");
                return None;
            }
        };
        let probe_lib = match device.new_library_with_source(PROBE_MSL, &opts) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[strand-decode-kernel] probe shader compile error: {e}");
                return None;
            }
        };

        let pipeline = |lib: &metal::Library, name: &str| -> Option<ComputePipelineState> {
            let f = lib.get_function(name, None).ok()?;
            device.new_compute_pipeline_state_with_function(&f).ok()
        };

        let fused = pipeline(&lib, "strand_trellis_gemv")?;
        let predec = pipeline(&lib, "strand_trellis_gemv_predec")?;
        let blockentry_sizeof = pipeline(&probe_lib, "strand_blockentry_sizeof")?;
        let queue = device.new_command_queue();

        Some(Self { device, queue, fused, predec, blockentry_sizeof })
    }

    pub fn gpu_blockentry_sizeof(&self) -> u32 {
        let out = self.alloc_shared(4);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.blockentry_sizeof);
        enc.set_buffer(0, Some(&out), 0);
        let one = MTLSize { width: 1, height: 1, depth: 1 };
        enc.dispatch_thread_groups(one, one);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let p = out.contents() as *const u32;
        unsafe { *p }
    }

    fn upload<T: Copy>(&self, data: &[T]) -> Buffer {
        let byte_len = (data.len() * std::mem::size_of::<T>()).max(4);
        let buf = self.device.new_buffer(byte_len as NSUInteger, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.contents() as *mut u8, data.len() * std::mem::size_of::<T>());
        }
        buf
    }

    fn upload_scalar(&self, v: u32) -> Buffer {
        self.upload(&[v])
    }

    fn alloc_shared(&self, byte_len: usize) -> Buffer {
        self.device.new_buffer(byte_len.max(4) as NSUInteger, MTLResourceOptions::StorageModeShared)
    }

    fn read_f32(&self, buf: &Buffer, len: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemv_fused(&self, payload: &[u8], tbl: &[BlockEntry], lut: &[i32], rows: u32, cols: u32, k_bits: u32, l_bits: u32, x_rht: &[f32]) -> Vec<f32> {
        assert_eq!(cols % 256, 0, "cols must be a multiple of 256 (STRICT deploy)");
        let bpr = cols / 256;
        assert_eq!(tbl.len() as u32, rows * bpr, "block table must be row-major with stride bpr");
        assert_eq!(x_rht.len() as u32, cols, "x_rht length must equal cols");

        let w_buf = self.upload(payload);
        let x_buf = self.upload(x_rht);
        let y_buf = self.alloc_shared(rows as usize * std::mem::size_of::<f32>());
        let rows_buf = self.upload_scalar(rows);
        let cols_buf = self.upload_scalar(cols);
        let tbl_buf = self.upload(tbl);
        let k_buf = self.upload_scalar(k_bits);
        let l_buf = self.upload_scalar(l_bits);
        let lut_buf = self.upload(lut);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.fused);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&y_buf), 0);
        enc.set_buffer(3, Some(&rows_buf), 0);
        enc.set_buffer(4, Some(&cols_buf), 0);
        enc.set_buffer(5, Some(&tbl_buf), 0);
        enc.set_buffer(6, Some(&k_buf), 0);
        enc.set_buffer(7, Some(&l_buf), 0);
        enc.set_buffer(8, Some(&lut_buf), 0);

        let lut_n = 1usize << l_bits;
        enc.set_threadgroup_memory_length(0, (lut_n * std::mem::size_of::<i32>()) as NSUInteger);
        enc.set_threadgroup_memory_length(1, (256 * std::mem::size_of::<f32>()) as NSUInteger);

        let groups = MTLSize { width: rows as NSUInteger, height: 1, depth: 1 };
        let tpg = MTLSize { width: 256, height: 1, depth: 1 };
        enc.dispatch_thread_groups(groups, tpg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        self.read_f32(&y_buf, rows as usize)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bench_gemv(&self, payload: &[u8], tbl: &[BlockEntry], lut: &[i32], rows: u32, cols: u32, k_bits: u32, l_bits: u32, x_rht: &[f32], iters: usize) -> f64 {
        let bpr = cols / 256;
        assert_eq!(tbl.len() as u32, rows * bpr);
        let w_buf = self.upload(payload);
        let x_buf = self.upload(x_rht);
        let y_buf = self.alloc_shared(rows as usize * 4);
        let rows_buf = self.upload_scalar(rows);
        let cols_buf = self.upload_scalar(cols);
        let tbl_buf = self.upload(tbl);
        let k_buf = self.upload_scalar(k_bits);
        let l_buf = self.upload_scalar(l_bits);
        let lut_buf = self.upload(lut);
        let lut_n = 1usize << l_bits;
        let groups = MTLSize { width: rows as NSUInteger, height: 1, depth: 1 };
        let tpg = MTLSize { width: 256, height: 1, depth: 1 };
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.fused);
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&y_buf), 0);
            enc.set_buffer(3, Some(&rows_buf), 0);
            enc.set_buffer(4, Some(&cols_buf), 0);
            enc.set_buffer(5, Some(&tbl_buf), 0);
            enc.set_buffer(6, Some(&k_buf), 0);
            enc.set_buffer(7, Some(&l_buf), 0);
            enc.set_buffer(8, Some(&lut_buf), 0);
            enc.set_threadgroup_memory_length(0, (lut_n * 4) as NSUInteger);
            enc.set_threadgroup_memory_length(1, (256 * 4) as NSUInteger);
            enc.dispatch_thread_groups(groups, tpg);
            enc.end_encoding();
            let t0 = std::time::Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            let dt = t0.elapsed().as_secs_f64();
            if dt > 0.0 && dt < best {
                best = dt;
            }
        }
        best
    }

    pub fn bench_peak_bw(&self, n_floats: usize, iters: usize) -> f64 {
        const SRC: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void peak_read(device const float* a [[buffer(0)]],
                              device float* out      [[buffer(1)]],
                              constant uint& n        [[buffer(2)]],
                              constant uint& gsz      [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
            float acc = 0.0;
            for (uint i = gid; i < n; i += gsz) acc += a[i];
            out[gid] = acc;
        }"#;
        let lib = self.device.new_library_with_source(SRC, &CompileOptions::new()).expect("peak_read compile");
        let f = lib.get_function("peak_read", None).expect("peak_read fn");
        let pipe = self.device.new_compute_pipeline_state_with_function(&f).expect("peak_read pipeline");
        let a = self.alloc_shared(n_floats * 4);
        let gsz: u32 = 1 << 16;
        let out = self.alloc_shared(gsz as usize * 4);
        let n_buf = self.upload_scalar(n_floats as u32);
        let gsz_buf = self.upload_scalar(gsz);
        let tpg = MTLSize { width: 256, height: 1, depth: 1 };
        let groups = MTLSize { width: (gsz / 256) as NSUInteger, height: 1, depth: 1 };
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&a), 0);
            enc.set_buffer(1, Some(&out), 0);
            enc.set_buffer(2, Some(&n_buf), 0);
            enc.set_buffer(3, Some(&gsz_buf), 0);
            enc.dispatch_thread_groups(groups, tpg);
            enc.end_encoding();
            let t0 = std::time::Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            let dt = t0.elapsed().as_secs_f64();
            if dt > 0.0 && dt < best {
                best = dt;
            }
        }
        (n_floats as f64 * 4.0) / best
    }
}

pub fn bake_block_entries(hdr: &TensorHeaderV2, enc: &EncodedTensor) -> Vec<BlockEntry> {
    assert!(
        !hdr.has_affine_min,
        "bake_block_entries: affine-min not supported by the deploy kernel (3-bit ⇒ has_affine_min=false); \
         4-bit needs a parallel eff_min_q off[8] + kernel add"
    );
    assert_eq!(hdr.vec_dim, 1, "deploy kernel is scalar (vec_dim=1); B.7 vector trellis is future work");
    assert_eq!(hdr.table.len(), enc.blocks.len(), "table/blocks length mismatch");

    let mut out = Vec::with_capacity(enc.blocks.len());
    for (rec, blk) in hdr.table.iter().zip(enc.blocks.iter()) {
        let n_sub = n_sub_blocks(blk.n as usize);
        let mults = unpack_sub_scales(&blk.sub_scales, n_sub);
        let mut eff = [0i32; 8];
        for (s, &m) in mults.iter().enumerate().take(8) {
            eff[s] = eff_scale_q(blk.scale_q, m);
        }
        debug_assert_eq!(rec.scale_q, blk.scale_q);
        debug_assert_eq!(rec.init_state, blk.init_state);
        out.push(BlockEntry { bit_offset: rec.bit_offset as u32, init_state: rec.init_state, scale_q: rec.scale_q, eff, n: blk.n as u16, d: hdr.vec_dim as u16, _pad: 0 });
    }
    out
}

pub fn gpu_matvec_named(gpu: &StrandGpu, model: &StrandModel, name: &str, x_rht: &[f32]) -> Option<Vec<f32>> {
    let hdr = model.tensor_header(name)?.clone();
    if hdr.shape.len() < 2 {
        return None;
    }
    let rows = hdr.shape[0] as u32;
    let cols = hdr.shape[1] as u32;
    if cols % 256 != 0 {
        return None;
    }
    let cfg = model.config_for(&hdr);
    let enc = model.encoded_tensor(name)?;
    let payload = model.view(name)?.payload.to_vec();
    let tbl = bake_block_entries(&hdr, &enc);
    let lut = codebook_lut(cfg.l_bits);
    Some(gpu.gemv_fused(&payload, &tbl, lut, rows, cols, cfg.k_bits, cfg.l_bits, x_rht))
}

use crate::block_walk::{block_init_state, block_plans, SideInfo};
use strand_quant::decode::decode_lean_with_lut;

const BITSLICE_MSL: &str = include_str!("../shaders/strand_bitslice.metal");

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BitsliceEntry {
    pub bit_offset: u32,

    pub init_state: u32,

    pub out_off: u32,

    pub n: u32,

    pub eff: [i32; 8],

    pub off: [i32; 8],

    /// Vector dim (1 = scalar). Kept last so the first 80 B are byte-identical to the
    /// scalar layout the original kernels read. d>1 selects the `_vec` kernels.
    pub d: u32,
}

pub struct BitsliceGpu {
    device: Device,
    queue: CommandQueue,
    decode: ComputePipelineState,
    /// Computed-codebook scalar decode (Variant A): inline integer Acklam, no LUT
    /// gather / staging. Byte-identical to `decode` on the frozen Gaussian codebook.
    decode_computed: ComputePipelineState,

    gemv_partials: ComputePipelineState,

    reduce_rows: ComputePipelineState,

    gemm_partials_b4: ComputePipelineState,
    gemm_partials_b16: ComputePipelineState,
    gemm_partials_b64: ComputePipelineState,

    reduce_rows_gemm: ComputePipelineState,
    entry_sizeof: ComputePipelineState,

    decode_vec: ComputePipelineState,
    gemv_partials_vec: ComputePipelineState,
}

impl BitsliceGpu {
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let opts = CompileOptions::new();
        let lib = match device.new_library_with_source(BITSLICE_MSL, &opts) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[strand-decode-kernel] bitslice shader compile error: {e}");
                return None;
            }
        };
        let pipeline = |name: &str| -> Option<ComputePipelineState> {
            let f = lib.get_function(name, None).ok()?;
            device.new_compute_pipeline_state_with_function(&f).ok()
        };
        let decode = pipeline("strand_bitslice_decode")?;
        let decode_computed = pipeline("strand_bitslice_decode_computed")?;
        let gemv_partials = pipeline("strand_bitslice_gemv_partials")?;
        let reduce_rows = pipeline("strand_bitslice_reduce_rows")?;
        let gemm_partials_b4 = pipeline("strand_bitslice_gemm_partials_b4")?;
        let gemm_partials_b16 = pipeline("strand_bitslice_gemm_partials_b16")?;
        let gemm_partials_b64 = pipeline("strand_bitslice_gemm_partials_b64")?;
        let reduce_rows_gemm = pipeline("strand_bitslice_reduce_rows_gemm")?;
        let entry_sizeof = pipeline("strand_bitslice_entry_sizeof")?;
        let decode_vec = pipeline("strand_bitslice_decode_vec")?;
        let gemv_partials_vec = pipeline("strand_bitslice_gemv_partials_vec")?;
        let queue = device.new_command_queue();
        let gpu = Self {
            device,
            queue,
            decode,
            decode_computed,
            gemv_partials,
            reduce_rows,
            gemm_partials_b4,
            gemm_partials_b16,
            gemm_partials_b64,
            reduce_rows_gemm,
            entry_sizeof,
            decode_vec,
            gemv_partials_vec,
        };

        let gpu_sz = gpu.gpu_entry_sizeof();
        assert_eq!(gpu_sz as usize, std::mem::size_of::<BitsliceEntry>(), "GPU sizeof(BitsliceEntry)={gpu_sz} != host {} — tbl stride would diverge", std::mem::size_of::<BitsliceEntry>());
        Some(gpu)
    }

    pub fn gpu_entry_sizeof(&self) -> u32 {
        let out = self.alloc(4);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.entry_sizeof);
        enc.set_buffer(0, Some(&out), 0);
        let one = MTLSize { width: 1, height: 1, depth: 1 };
        enc.dispatch_thread_groups(one, one);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        unsafe { *(out.contents() as *const u32) }
    }

    fn upload<T: Copy>(&self, data: &[T]) -> Buffer {
        let byte_len = (data.len() * std::mem::size_of::<T>()).max(4);
        let buf = self.device.new_buffer(byte_len as NSUInteger, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.contents() as *mut u8, data.len() * std::mem::size_of::<T>());
        }
        buf
    }

    fn alloc(&self, byte_len: usize) -> Buffer {
        self.device.new_buffer(byte_len.max(4) as NSUInteger, MTLResourceOptions::StorageModeShared)
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

    pub fn decode_q12(&self, payload: &[u8], tbl: &[BitsliceEntry], lut: &[i32], total: usize, k_bits: u32, l_bits: u32) -> Vec<i32> {
        assert_eq!(lut.len(), 1usize << l_bits, "LUT must have 2^L entries");
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
        let groups = MTLSize { width: (tbl.len() as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
        let tpg = MTLSize { width: 256, height: 1, depth: 1 };
        enc.dispatch_thread_groups(groups, tpg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = out_buf.contents() as *const i32;
        unsafe { std::slice::from_raw_parts(ptr, total) }.to_vec()
    }

    /// Computed-codebook scalar decode (Variant A). Identical output to
    /// [`Self::decode_q12`] on the frozen Gaussian codebook, but the kernel
    /// synthesises each codebook value inline (integer Acklam) instead of
    /// gathering a staged `2^L` LUT — only the tiny `tail` prefix is uploaded.
    /// `tail` must be `strand_quant::codebook::tail_left_prefix_q12(l_bits)`.
    pub fn decode_q12_computed(&self, payload: &[u8], tbl: &[BitsliceEntry], tail: &[i32], total: usize, k_bits: u32, l_bits: u32) -> Vec<i32> {
        assert!(l_bits <= 16, "computed codebook kernel asserts L <= 16, got {l_bits}");
        let w_buf = self.upload_payload(payload);
        let out_buf = self.alloc(total * 4);
        let tbl_buf = self.upload(tbl);
        let nb_buf = self.upload(&[tbl.len() as u32]);
        let k_buf = self.upload(&[k_bits]);
        let l_buf = self.upload(&[l_bits]);
        let tail_buf = self.upload(tail);
        let tlen_buf = self.upload(&[tail.len() as u32]);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.decode_computed);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        enc.set_buffer(2, Some(&tbl_buf), 0);
        enc.set_buffer(3, Some(&nb_buf), 0);
        enc.set_buffer(4, Some(&k_buf), 0);
        enc.set_buffer(5, Some(&l_buf), 0);
        enc.set_buffer(6, Some(&tail_buf), 0);
        enc.set_buffer(7, Some(&tlen_buf), 0);
        // No threadgroup staging: the codebook is computed per state.
        let groups = MTLSize { width: (tbl.len() as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
        let tpg = MTLSize { width: 256, height: 1, depth: 1 };
        enc.dispatch_thread_groups(groups, tpg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = out_buf.contents() as *const i32;
        unsafe { std::slice::from_raw_parts(ptr, total) }.to_vec()
    }

    pub fn bench_decode(&self, payload: &[u8], tbl: &[BitsliceEntry], lut: &[i32], total: usize, k_bits: u32, l_bits: u32, iters: usize) -> f64 {
        let w_buf = self.upload_payload(payload);
        let out_buf = self.alloc(total * 4);
        let tbl_buf = self.upload(tbl);
        let nb_buf = self.upload(&[tbl.len() as u32]);
        let k_buf = self.upload(&[k_bits]);
        let l_buf = self.upload(&[l_bits]);
        let lut_buf = self.upload(lut);
        let groups = MTLSize { width: (tbl.len() as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
        let tpg = MTLSize { width: 256, height: 1, depth: 1 };
        let mut best = f64::INFINITY;
        for _ in 0..iters {
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
            enc.dispatch_thread_groups(groups, tpg);
            enc.end_encoding();
            let t0 = std::time::Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            let dt = t0.elapsed().as_secs_f64();
            if dt > 0.0 && dt < best {
                best = dt;
            }
        }
        best
    }
}

// ---- B.7 vector trellis (d>1) dispatch ---------------------------------------
impl BitsliceGpu {
    /// Vector-d decode. `lut` must have `2^L * d` entries; `tbl` must be a vec bake
    /// (`bake_bitslice_entries_vec`). Output is `total` Q12 ints, byte-identical to
    /// `decode_tensor_fixed_with_lut_vec`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_q12_vec(&self, payload: &[u8], tbl: &[BitsliceEntry], lut: &[i32], total: usize, k_bits: u32, l_bits: u32, d: usize) -> Vec<i32> {
        assert_eq!(lut.len(), (1usize << l_bits) * d, "vec LUT must have 2^L * d entries");
        let w_buf = self.upload_payload(payload);
        let out_buf = self.alloc(total * 4);
        let tbl_buf = self.upload(tbl);
        let nb_buf = self.upload(&[tbl.len() as u32]);
        let k_buf = self.upload(&[k_bits]);
        let l_buf = self.upload(&[l_bits]);
        let lut_buf = self.upload(lut);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.decode_vec);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        enc.set_buffer(2, Some(&tbl_buf), 0);
        enc.set_buffer(3, Some(&nb_buf), 0);
        enc.set_buffer(4, Some(&k_buf), 0);
        enc.set_buffer(5, Some(&l_buf), 0);
        enc.set_buffer(6, Some(&lut_buf), 0);
        enc.set_threadgroup_memory_length(0, ((1usize << l_bits) * d * 4) as NSUInteger);
        let groups = MTLSize { width: (tbl.len() as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
        let tpg = MTLSize { width: 256, height: 1, depth: 1 };
        enc.dispatch_thread_groups(groups, tpg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = out_buf.contents() as *const i32;
        unsafe { std::slice::from_raw_parts(ptr, total) }.to_vec()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bench_decode_vec(&self, payload: &[u8], tbl: &[BitsliceEntry], lut: &[i32], total: usize, k_bits: u32, l_bits: u32, d: usize, iters: usize) -> f64 {
        let w_buf = self.upload_payload(payload);
        let out_buf = self.alloc(total * 4);
        let tbl_buf = self.upload(tbl);
        let nb_buf = self.upload(&[tbl.len() as u32]);
        let k_buf = self.upload(&[k_bits]);
        let l_buf = self.upload(&[l_bits]);
        let lut_buf = self.upload(lut);
        let groups = MTLSize { width: (tbl.len() as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
        let tpg = MTLSize { width: 256, height: 1, depth: 1 };
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.decode_vec);
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(&out_buf), 0);
            enc.set_buffer(2, Some(&tbl_buf), 0);
            enc.set_buffer(3, Some(&nb_buf), 0);
            enc.set_buffer(4, Some(&k_buf), 0);
            enc.set_buffer(5, Some(&l_buf), 0);
            enc.set_buffer(6, Some(&lut_buf), 0);
            enc.set_threadgroup_memory_length(0, ((1usize << l_bits) * d * 4) as NSUInteger);
            enc.dispatch_thread_groups(groups, tpg);
            enc.end_encoding();
            let t0 = std::time::Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            let dt = t0.elapsed().as_secs_f64();
            if dt > 0.0 && dt < best {
                best = dt;
            }
        }
        best
    }

    /// Vector-d fused y=Wx, B=1. Two passes: `_vec` gemv partials then the (d-agnostic)
    /// row reduce. Returns `rows` floats.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_vec(&self, payload: &[u8], tbl: &[BitsliceEntry], lut: &[i32], rows: u32, cols: u32, k_bits: u32, l_bits: u32, d: usize, x: &[f32]) -> Vec<f32> {
        assert_eq!(cols % 256, 0, "bitslice matvec_vec: cols must be a multiple of 256");
        assert_eq!(x.len() as u32, cols, "x length must equal cols");
        let bpr = cols / 256;
        assert_eq!(tbl.len() as u32, rows * bpr, "tbl must cover rows*bpr blocks");
        let bufs = self.matvec_buffers(payload, tbl, lut, rows, cols, k_bits, l_bits, x);
        self.matvec_dispatch_vec(&bufs, tbl.len() as u32, rows, l_bits, d);
        let ptr = bufs.y.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, rows as usize) }.to_vec()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bench_matvec_vec(&self, payload: &[u8], tbl: &[BitsliceEntry], lut: &[i32], rows: u32, cols: u32, k_bits: u32, l_bits: u32, d: usize, x: &[f32], iters: usize) -> f64 {
        let bufs = self.matvec_buffers(payload, tbl, lut, rows, cols, k_bits, l_bits, x);
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t0 = std::time::Instant::now();
            self.matvec_dispatch_vec(&bufs, tbl.len() as u32, rows, l_bits, d);
            let dt = t0.elapsed().as_secs_f64();
            if dt > 0.0 && dt < best {
                best = dt;
            }
        }
        best
    }

    fn matvec_dispatch_vec(&self, b: &BitsliceMatvecBufs, n_blocks: u32, rows: u32, l_bits: u32, d: usize) {
        let cmd = self.queue.new_command_buffer();
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.gemv_partials_vec);
            enc.set_buffer(0, Some(&b.w), 0);
            enc.set_buffer(1, Some(&b.x), 0);
            enc.set_buffer(2, Some(&b.partials), 0);
            enc.set_buffer(3, Some(&b.tbl), 0);
            enc.set_buffer(4, Some(&b.n_blocks), 0);
            enc.set_buffer(5, Some(&b.cols), 0);
            enc.set_buffer(6, Some(&b.k), 0);
            enc.set_buffer(7, Some(&b.l), 0);
            enc.set_buffer(8, Some(&b.lut), 0);
            enc.set_threadgroup_memory_length(0, ((1usize << l_bits) * d * 4) as NSUInteger);
            let groups = MTLSize { width: (n_blocks as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
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
            let groups = MTLSize { width: (rows as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
            let tpg = MTLSize { width: 256, height: 1, depth: 1 };
            enc.dispatch_thread_groups(groups, tpg);
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();
    }
}

impl BitsliceGpu {
    #[allow(clippy::too_many_arguments)]
    pub fn matvec(&self, payload: &[u8], tbl: &[BitsliceEntry], lut: &[i32], rows: u32, cols: u32, k_bits: u32, l_bits: u32, x: &[f32]) -> Vec<f32> {
        assert_eq!(cols % 256, 0, "bitslice matvec: cols must be a multiple of 256");
        assert_eq!(x.len() as u32, cols, "x length must equal cols");
        let bpr = cols / 256;
        assert_eq!(tbl.len() as u32, rows * bpr, "tbl must cover rows*bpr blocks");
        let bufs = self.matvec_buffers(payload, tbl, lut, rows, cols, k_bits, l_bits, x);
        self.matvec_dispatch(&bufs, tbl.len() as u32, rows, l_bits);
        let ptr = bufs.y.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, rows as usize) }.to_vec()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bench_matvec(&self, payload: &[u8], tbl: &[BitsliceEntry], lut: &[i32], rows: u32, cols: u32, k_bits: u32, l_bits: u32, x: &[f32], iters: usize) -> f64 {
        let bufs = self.matvec_buffers(payload, tbl, lut, rows, cols, k_bits, l_bits, x);
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

    fn matvec_buffers(&self, payload: &[u8], tbl: &[BitsliceEntry], lut: &[i32], rows: u32, cols: u32, k_bits: u32, l_bits: u32, x: &[f32]) -> BitsliceMatvecBufs {
        BitsliceMatvecBufs {
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

    fn matvec_dispatch(&self, b: &BitsliceMatvecBufs, n_blocks: u32, rows: u32, l_bits: u32) {
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
            let groups = MTLSize { width: (n_blocks as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
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
            let groups = MTLSize { width: (rows as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
            let tpg = MTLSize { width: 256, height: 1, depth: 1 };
            enc.dispatch_thread_groups(groups, tpg);
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();
    }
}

struct BitsliceMatvecBufs {
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

// ---- WHOLE-TOKEN COMMAND BUFFER (Wave-4 speed bet 2) ---------------------------------------
// A "token pass" dispatches every projection tensor for one decode step. The deployed path
// runs one `new_command_buffer -> encode -> commit -> wait_until_completed` per tensor, so a
// Qwen-0.5B token costs ~7 tensors/layer * 24 layers = 168 commits + 168 GPU round-trips. The
// research note (docs/STRAND-speed-moonshot-research.md §D) flags this as mandatory to fix
// before dismantle integration: a 3x-faster kernel can lose the win to per-tensor commits.
//
// `MatvecToken` is an opaque pre-prepared (buffers + dims) tensor handle so the gate can build
// the whole token graph ONCE and then time two execution shapes against identical work:
//   * run_token_per_tensor  — one command buffer per tensor (baseline, today's shape)
//   * run_token_one_buffer  — all tensors' partials+reduce in ONE command buffer, one commit,
//                             one wait_until_completed (the target)
// Both produce the same per-tensor y outputs. This is additive: the production decode path is
// untouched; only the scratch gate calls these.
pub struct MatvecToken {
    bufs: BitsliceMatvecBufs,
    n_blocks: u32,
    rows: u32,
    l_bits: u32,
}

impl MatvecToken {
    /// Read back this tensor's output vector (length `rows`). Only valid after a run.
    pub fn y(&self) -> Vec<f32> {
        let ptr = self.bufs.y.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, self.rows as usize) }.to_vec()
    }
}

impl BitsliceGpu {
    /// Pre-prepare one tensor of a token pass (upload payload/tbl/lut/x, alloc partials/y).
    /// Mirrors `matvec_buffers` exactly so the two execution shapes time identical work.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_token_tensor(&self, payload: &[u8], tbl: &[BitsliceEntry], lut: &[i32], rows: u32, cols: u32, k_bits: u32, l_bits: u32, x: &[f32]) -> MatvecToken {
        assert_eq!(cols % 256, 0, "prepare_token_tensor: cols must be a multiple of 256");
        assert_eq!(x.len() as u32, cols, "x length must equal cols");
        let bpr = cols / 256;
        assert_eq!(tbl.len() as u32, rows * bpr, "tbl must cover rows*bpr blocks");
        MatvecToken { bufs: self.matvec_buffers(payload, tbl, lut, rows, cols, k_bits, l_bits, x), n_blocks: tbl.len() as u32, rows, l_bits }
    }

    fn encode_token_tensor(&self, cmd: &metal::CommandBufferRef, t: &MatvecToken) {
        let b = &t.bufs;
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
            enc.set_threadgroup_memory_length(0, ((1usize << t.l_bits) * 4) as NSUInteger);
            let groups = MTLSize { width: (t.n_blocks as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
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
            let groups = MTLSize { width: (t.rows as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
            let tpg = MTLSize { width: 256, height: 1, depth: 1 };
            enc.dispatch_thread_groups(groups, tpg);
            enc.end_encoding();
        }
    }

    /// BASELINE token pass: one command buffer per tensor, commit+wait each (today's shape).
    pub fn run_token_per_tensor(&self, tokens: &[MatvecToken]) {
        for t in tokens {
            let cmd = self.queue.new_command_buffer();
            self.encode_token_tensor(cmd, t);
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    /// TARGET token pass: encode ALL tensors into ONE command buffer, single commit + single
    /// wait. Within-buffer ordering provides the dependency barrier between each tensor's
    /// partials and its reduce stage (sequential compute encoders on one buffer execute in
    /// order on Apple GPUs).
    pub fn run_token_one_buffer(&self, tokens: &[MatvecToken]) {
        let cmd = self.queue.new_command_buffer();
        for t in tokens {
            self.encode_token_tensor(cmd, t);
        }
        cmd.commit();
        cmd.wait_until_completed();
    }
}

impl BitsliceGpu {
    fn gemm_pipeline(&self, batch: usize) -> &ComputePipelineState {
        match batch {
            4 => &self.gemm_partials_b4,
            16 => &self.gemm_partials_b16,
            64 => &self.gemm_partials_b64,
            _ => panic!("bitslice gemm: batch must be one of 4/16/64 (got {batch})"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm(&self, payload: &[u8], tbl: &[BitsliceEntry], lut: &[i32], rows: u32, cols: u32, k_bits: u32, l_bits: u32, xs: &[f32], batch: usize) -> Vec<f32> {
        assert_eq!(cols % 256, 0, "bitslice gemm: cols must be a multiple of 256");
        assert_eq!(xs.len(), batch * cols as usize, "xs must be batch x in_features");
        let bpr = cols / 256;
        assert_eq!(tbl.len() as u32, rows * bpr, "tbl must cover rows*bpr blocks");
        let bufs = self.gemm_buffers(payload, tbl, lut, rows, cols, k_bits, l_bits, xs, batch);
        self.gemm_dispatch(&bufs, tbl.len() as u32, rows, l_bits, batch);
        let ptr = bufs.y.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, rows as usize * batch) }.to_vec()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bench_gemm(&self, payload: &[u8], tbl: &[BitsliceEntry], lut: &[i32], rows: u32, cols: u32, k_bits: u32, l_bits: u32, xs: &[f32], batch: usize, iters: usize) -> f64 {
        let bufs = self.gemm_buffers(payload, tbl, lut, rows, cols, k_bits, l_bits, xs, batch);
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t0 = std::time::Instant::now();
            self.gemm_dispatch(&bufs, tbl.len() as u32, rows, l_bits, batch);
            let dt = t0.elapsed().as_secs_f64();
            if dt > 0.0 && dt < best {
                best = dt;
            }
        }
        best
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_buffers(&self, payload: &[u8], tbl: &[BitsliceEntry], lut: &[i32], rows: u32, cols: u32, k_bits: u32, l_bits: u32, xs: &[f32], batch: usize) -> BitsliceGemmBufs {
        let cols_u = cols as usize;
        let mut xt = vec![0.0f32; xs.len()];
        for b in 0..batch {
            for c in 0..cols_u {
                xt[c * batch + b] = xs[b * cols_u + c];
            }
        }
        BitsliceGemmBufs {
            w: self.upload_payload(payload),
            xt: self.upload(&xt),
            partials: self.alloc(tbl.len() * batch * 4),
            y: self.alloc(rows as usize * batch * 4),
            tbl: self.upload(tbl),
            n_blocks: self.upload(&[tbl.len() as u32]),
            cols: self.upload(&[cols]),
            k: self.upload(&[k_bits]),
            l: self.upload(&[l_bits]),
            lut: self.upload(lut),
            rows: self.upload(&[rows]),
            bpr: self.upload(&[cols / 256]),
            batch: self.upload(&[batch as u32]),
        }
    }

    fn gemm_dispatch(&self, b: &BitsliceGemmBufs, n_blocks: u32, rows: u32, l_bits: u32, batch: usize) {
        let cmd = self.queue.new_command_buffer();
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(self.gemm_pipeline(batch));
            enc.set_buffer(0, Some(&b.w), 0);
            enc.set_buffer(1, Some(&b.xt), 0);
            enc.set_buffer(2, Some(&b.partials), 0);
            enc.set_buffer(3, Some(&b.tbl), 0);
            enc.set_buffer(4, Some(&b.n_blocks), 0);
            enc.set_buffer(5, Some(&b.cols), 0);
            enc.set_buffer(6, Some(&b.k), 0);
            enc.set_buffer(7, Some(&b.l), 0);
            enc.set_buffer(8, Some(&b.lut), 0);
            enc.set_threadgroup_memory_length(0, ((1usize << l_bits) * 4) as NSUInteger);
            let groups = MTLSize { width: (n_blocks as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
            let tpg = MTLSize { width: 256, height: 1, depth: 1 };
            enc.dispatch_thread_groups(groups, tpg);
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.reduce_rows_gemm);
            enc.set_buffer(0, Some(&b.partials), 0);
            enc.set_buffer(1, Some(&b.y), 0);
            enc.set_buffer(2, Some(&b.rows), 0);
            enc.set_buffer(3, Some(&b.bpr), 0);
            enc.set_buffer(4, Some(&b.batch), 0);
            let n_out = rows as u64 * batch as u64;
            let groups = MTLSize { width: n_out.div_ceil(256) as NSUInteger, height: 1, depth: 1 };
            let tpg = MTLSize { width: 256, height: 1, depth: 1 };
            enc.dispatch_thread_groups(groups, tpg);
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();
    }
}

struct BitsliceGemmBufs {
    w: Buffer,
    xt: Buffer,
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
    batch: Buffer,
}

pub struct BitslicePrepared {
    w: Buffer,
    tbl: Buffer,
    lut: Buffer,
    out: Buffer,
    nb_buf: Buffer,
    k_buf: Buffer,
    l_buf: Buffer,
    n_blocks: u32,
    total: usize,
    k_bits: u32,
    l_bits: u32,
    payload_len: usize,
    lut_len: usize,
}

impl BitsliceGpu {
    pub fn prepare(&self, enc: &EncodedTensor, cfg: &strand_quant::TrellisConfig) -> Option<BitslicePrepared> {
        if cfg.vec_dim() > 1 {
            return None;
        }
        let tbl = bake_bitslice_entries(enc, cfg)?;
        let lut = codebook_lut(cfg.l_bits);
        Some(BitslicePrepared {
            w: self.upload_payload(&enc.bits),
            tbl: self.upload(&tbl),
            lut: self.upload(lut),
            out: self.alloc(enc.total * 4),
            nb_buf: self.upload(&[tbl.len() as u32]),
            k_buf: self.upload(&[cfg.k_bits]),
            l_buf: self.upload(&[cfg.l_bits]),
            n_blocks: tbl.len() as u32,
            total: enc.total,
            k_bits: cfg.k_bits,
            l_bits: cfg.l_bits,
            payload_len: enc.bits.len(),
            lut_len: lut.len(),
        })
    }

    pub fn decode_q12_prepared(&self, p: &BitslicePrepared) -> Vec<i32> {
        self.dispatch_prepared(p);
        let ptr = p.out.contents() as *const i32;
        unsafe { std::slice::from_raw_parts(ptr, p.total) }.to_vec()
    }

    pub fn dispatch_prepared(&self, p: &BitslicePrepared) {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.decode);
        enc.set_buffer(0, Some(&p.w), 0);
        enc.set_buffer(1, Some(&p.out), 0);
        enc.set_buffer(2, Some(&p.tbl), 0);
        enc.set_buffer(3, Some(&p.nb_buf), 0);
        enc.set_buffer(4, Some(&p.k_buf), 0);
        enc.set_buffer(5, Some(&p.l_buf), 0);
        enc.set_buffer(6, Some(&p.lut), 0);
        enc.set_threadgroup_memory_length(0, ((1usize << p.l_bits) * 4) as NSUInteger);
        let groups = MTLSize { width: (p.n_blocks as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
        let tpg = MTLSize { width: 256, height: 1, depth: 1 };
        enc.dispatch_thread_groups(groups, tpg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    pub fn encode_prepared(&self, cmd: &metal::CommandBufferRef, p: &BitslicePrepared) {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.decode);
        enc.set_buffer(0, Some(&p.w), 0);
        enc.set_buffer(1, Some(&p.out), 0);
        enc.set_buffer(2, Some(&p.tbl), 0);
        enc.set_buffer(3, Some(&p.nb_buf), 0);
        enc.set_buffer(4, Some(&p.k_buf), 0);
        enc.set_buffer(5, Some(&p.l_buf), 0);
        enc.set_buffer(6, Some(&p.lut), 0);
        enc.set_threadgroup_memory_length(0, ((1usize << p.l_bits) * 4) as NSUInteger);
        let groups = MTLSize { width: (p.n_blocks as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
        let tpg = MTLSize { width: 256, height: 1, depth: 1 };
        enc.dispatch_thread_groups(groups, tpg);
        enc.end_encoding();
    }

    pub fn dispatch_prepared_all(&self, tensors: &[BitslicePrepared]) {
        let cmd = self.queue.new_command_buffer();
        for p in tensors {
            self.encode_prepared(cmd, p);
        }
        cmd.commit();
        cmd.wait_until_completed();
    }
}

impl BitslicePrepared {
    pub fn total(&self) -> usize {
        self.total
    }

    pub fn config_bits(&self) -> (u32, u32) {
        (self.k_bits, self.l_bits)
    }

    pub fn read_out(&self) -> Vec<i32> {
        let ptr = self.out.contents() as *const i32;
        unsafe { std::slice::from_raw_parts(ptr, self.total) }.to_vec()
    }

    pub fn gpu_bytes(&self) -> usize {
        let padded_payload = self.payload_len.div_ceil(4) * 4 + 8;
        padded_payload + self.n_blocks as usize * std::mem::size_of::<BitsliceEntry>() + self.lut_len * 4 + self.total * 4 + 3 * 4
    }
}

pub fn bake_bitslice_entries(enc: &EncodedTensor, cfg: &strand_quant::TrellisConfig) -> Option<Vec<BitsliceEntry>> {
    if enc.blocks.iter().any(|b| b.n > 256) {
        return None;
    }
    let k = cfg.k_bits as usize;
    let plans = block_plans(enc, k);
    let mut out = Vec::with_capacity(enc.blocks.len());
    for (blk, plan) in enc.blocks.iter().zip(plans.iter()) {
        let side = SideInfo::hoist(blk, enc.has_affine_min);
        let mut eff = [0i32; 8];
        let mut off = [0i32; 8];
        eff[..side.n_sub].copy_from_slice(side.eff());
        off[..side.n_sub].copy_from_slice(side.off());
        let init = block_init_state(blk, &enc.bits, plan.start_bit, cfg, enc.tail_biting);
        out.push(BitsliceEntry { bit_offset: plan.start_bit as u32, init_state: init as u32, out_off: plan.out_off as u32, n: blk.n, eff, off, d: 1 });
    }
    Some(out)
}

/// Vector-trellis (d>1) bake. Differs from the scalar bake only in the bitstream
/// arithmetic: each block consumes `n_steps = ceil(n/d)` symbols (so the per-block
/// `start_bit` accumulates `n_steps * k`, NOT `n * k`), while `out_off` still
/// accumulates the true output count `n`. The init-state walk (tail-biting) reads
/// `n_steps` symbols. This mirrors `decode_tensor_fixed_with_lut_vec` exactly.
pub fn bake_bitslice_entries_vec(enc: &EncodedTensor, cfg: &strand_quant::TrellisConfig) -> Option<Vec<BitsliceEntry>> {
    if enc.blocks.iter().any(|b| b.n > 256) {
        return None;
    }
    let d = cfg.vec_dim();
    let k = cfg.k_bits as usize;
    let mask = cfg.state_mask();
    let input_mask = cfg.num_inputs() - 1;

    let mut out = Vec::with_capacity(enc.blocks.len());
    let mut start_bit = 0usize;
    let mut out_off = 0usize;
    for blk in &enc.blocks {
        let n = blk.n as usize;
        let n_steps = n.div_ceil(d);

        let side = SideInfo::hoist(blk, enc.has_affine_min);
        let mut eff = [0i32; 8];
        let mut off = [0i32; 8];
        eff[..side.n_sub].copy_from_slice(side.eff());
        off[..side.n_sub].copy_from_slice(side.off());

        // init state: tail-biting walks n_steps symbols, else stored init_state.
        let nk = n_steps * k;
        let init = if enc.tail_biting && nk >= cfg.l_bits as usize {
            let mut s = 0usize;
            let mut c = start_bit;
            for _ in 0..n_steps {
                let sym = strand_quant::trellis::read_bits(&enc.bits, c, cfg.k_bits) & input_mask;
                c += k;
                s = ((s << k) | sym) & mask;
            }
            s
        } else {
            blk.init_state as usize & mask
        };

        out.push(BitsliceEntry { bit_offset: start_bit as u32, init_state: init as u32, out_off: out_off as u32, n: blk.n, eff, off, d: d as u32 });
        start_bit += n_steps * k;
        out_off += n;
    }
    Some(out)
}

pub fn bitslice_decode_q12(gpu: &BitsliceGpu, enc: &EncodedTensor, cfg: &strand_quant::TrellisConfig) -> Vec<i32> {
    bitslice_decode_q12_with_lut(gpu, enc, cfg, codebook_lut(cfg.l_bits))
}

pub fn bitslice_decode_q12_with_lut(gpu: &BitsliceGpu, enc: &EncodedTensor, cfg: &strand_quant::TrellisConfig, lut: &[i32]) -> Vec<i32> {
    if cfg.vec_dim() > 1 {
        return decode_lean_with_lut(enc, cfg, lut);
    }
    let Some(tbl) = bake_bitslice_entries(enc, cfg) else {
        return decode_lean_with_lut(enc, cfg, lut);
    };
    gpu.decode_q12(&enc.bits, &tbl, lut, enc.total, cfg.k_bits, cfg.l_bits)
}

/// Computed-codebook scalar GPU decode (Variant A): the codebook value is
/// synthesised inline on the GPU (integer Acklam) — no `2^L` LUT staged. Output
/// is byte-identical to [`bitslice_decode_q12`] on the frozen Gaussian codebook.
/// Falls back to the CPU lean decoder for the vector path (which uses an arbitrary
/// LUT that is not the computed Gaussian codebook) or when the bake is rejected.
pub fn bitslice_decode_q12_computed(gpu: &BitsliceGpu, enc: &EncodedTensor, cfg: &strand_quant::TrellisConfig) -> Vec<i32> {
    if cfg.vec_dim() > 1 {
        return decode_lean_with_lut(enc, cfg, codebook_lut(cfg.l_bits));
    }
    let Some(tbl) = bake_bitslice_entries(enc, cfg) else {
        return decode_lean_with_lut(enc, cfg, codebook_lut(cfg.l_bits));
    };
    let tail = strand_quant::codebook::tail_left_prefix_q12(cfg.l_bits);
    gpu.decode_q12_computed(&enc.bits, &tbl, &tail, enc.total, cfg.k_bits, cfg.l_bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use strand_quant::decode::decode_lean;
    use strand_quant::encode::encode_tensor;
    use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2};
    use strand_quant::TrellisConfig;

    fn write_tiny_v2(name: &str, rows: u64, cols: u64, cfg: &TrellisConfig, enc: &EncodedTensor) -> std::path::PathBuf {
        let shape = [rows, cols];
        let pt = PackedTensorV2 {
            base: PackedTensor { name, shape: &shape, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc },
            block_len: cfg.block_len as u32,
        };
        let buf = write_strand_v2(&[pt], [0u8; 32], true).expect("write_strand_v2");
        let mut path = std::env::temp_dir();
        let pid = std::process::id();
        let uniq = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        path.push(format!("strand_metal_{name}_{pid}_{uniq}.strand"));
        let mut f = std::fs::File::create(&path).expect("create temp .strand");
        f.write_all(&buf).expect("write temp .strand");
        f.sync_all().ok();
        path
    }

    #[test]
    fn gpu_q12_matches_cpu_decode_lean() {
        let Some(gpu) = StrandGpu::new() else {
            eprintln!("[strand-decode-kernel] no Metal device; skipping gpu_q12_matches_cpu_decode_lean");
            return;
        };

        let gpu_sz = gpu.gpu_blockentry_sizeof();
        assert_eq!(
            gpu_sz as usize,
            std::mem::size_of::<BlockEntry>(),
            "GPU sizeof(BlockEntry)={gpu_sz} != host size_of::<BlockEntry>()={}; \
             the row-major tbl stride would diverge",
            std::mem::size_of::<BlockEntry>()
        );

        for &(rows, cols) in &[(2u64, 256u64), (2u64, 512u64)] {
            let n = (rows * cols) as usize;
            let weights: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 0.7).collect();
            let cfg = TrellisConfig::for_bpw(3.0);
            let enc = encode_tensor(&weights, &cfg);
            assert!(!enc.has_affine_min, "3-bit deploy must have affine-min off");
            let path = write_tiny_v2("w", rows, cols, &cfg, &enc);
            let model = StrandModel::open(&path).expect("open");

            let cpu_q12 = decode_lean(&enc, &cfg);
            let lut = codebook_lut(cfg.l_bits);
            let hdr = model.tensor_header("w").unwrap().clone();
            let enc_back = model.encoded_tensor("w").unwrap();
            let tbl = bake_block_entries(&hdr, &enc_back);
            let payload = model.view("w").unwrap().payload.to_vec();

            let cols_u = cols as usize;
            let probe_cols = [0usize, 1, 31, 32, 200, cols_u / 2, cols_u - 1];
            for &c in probe_cols.iter().filter(|&&c| c < cols_u) {
                let mut x_rht = vec![0.0f32; cols_u];
                x_rht[c] = 1.0;
                let y = gpu.gemv_fused(&payload, &tbl, lut, rows as u32, cols as u32, cfg.k_bits, cfg.l_bits, &x_rht);
                for r in 0..rows as usize {
                    let recovered = (y[r] * 4096.0).round() as i32;
                    let expected = cpu_q12[r * cols_u + c];
                    assert_eq!(recovered, expected, "rows={rows} cols={cols} row={r} col={c}: GPU Q12 {recovered} != CPU {expected}");
                }
            }

            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn gpu_y_matches_cpu_matvec() {
        let Some(gpu) = StrandGpu::new() else {
            eprintln!("[strand-decode-kernel] no Metal device; skipping gpu_y_matches_cpu_matvec");
            return;
        };
        let (rows, cols) = (3u64, 512u64);
        let n = (rows * cols) as usize;
        let weights: Vec<f32> = (0..n).map(|i| (i as f32 * 0.021).cos() * 0.4).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&weights, &cfg);
        let path = write_tiny_v2("w", rows, cols, &cfg, &enc);
        let model = StrandModel::open(&path).expect("open");

        let x: Vec<f32> = (0..cols as usize).map(|i| (i as f32 * 0.05).sin()).collect();
        let y_gpu = gpu_matvec_named(&gpu, &model, "w", &x).expect("gpu_matvec_named");
        let y_cpu = crate::gemv::matvec_named(&model, "w", &x).expect("cpu matvec_named");
        assert_eq!(y_gpu.len(), y_cpu.len());
        for r in 0..rows as usize {
            let denom = y_cpu[r].abs().max(1e-3);
            assert!((y_gpu[r] - y_cpu[r]).abs() / denom < 1e-4, "row {r}: GPU {} vs CPU {} (rel)", y_gpu[r], y_cpu[r]);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bitslice_q12_matches_decode_tensor_fixed() {
        use strand_quant::decode::decode_tensor_fixed;
        use strand_quant::encode::{encode_tensor_with, EncodeOpts};

        let Some(gpu) = BitsliceGpu::new() else {
            eprintln!("[strand-decode-kernel] no Metal device; skipping bitslice identity test");
            return;
        };

        let configs = [
            TrellisConfig::for_bpw(3.0),
            TrellisConfig::for_bpw(2.0),
            TrellisConfig::for_bpw(4.0),
            TrellisConfig::for_bpw_l(2.0, 12),
            TrellisConfig::for_bpw_l(2.0, 5),
            TrellisConfig::for_bpw_l(4.0, 4),
        ];
        for cfg in configs {
            for seed in 0..8u64 {
                let n = 1 + (seed as usize * 211) % 2048;
                let w: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect();
                let variants = [
                    encode_tensor(&w, &cfg),
                    encode_tensor_with(&w, &cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
                    encode_tensor_with(&w, &cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
                    encode_tensor_with(&w, &cfg, &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() }),
                ];
                for enc in &variants {
                    let got = bitslice_decode_q12(&gpu, enc, &cfg);
                    let want = decode_tensor_fixed(enc, &cfg);
                    assert_eq!(got, want, "bitslice GPU diverged: k={} L={} n={n} seed={seed} tail={} affine={}", cfg.k_bits, cfg.l_bits, enc.tail_biting, enc.has_affine_min);
                }
            }
        }
    }

    #[test]
    fn bitslice_gemm_identity_and_reference() {
        use strand_quant::decode::decode_tensor_fixed;

        let Some(gpu) = BitsliceGpu::new() else {
            eprintln!("[strand-decode-kernel] no Metal device; skipping bitslice gemm test");
            return;
        };

        let (rows, cols) = (8usize, 512usize);
        let total = rows * cols;
        let w: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.0137).sin() * 0.5).collect();
        for cfg in [TrellisConfig::for_bpw(3.0), TrellisConfig::for_bpw_l(2.0, 12)] {
            let enc = encode_tensor(&w, &cfg);
            let want = decode_tensor_fixed(&enc, &cfg);
            let tbl = bake_bitslice_entries(&enc, &cfg).expect("bake");
            let lut = codebook_lut(cfg.l_bits);

            for &batch in &[4usize, 16, 64] {
                let mut xs = vec![0.0f32; batch * cols];
                let probe: Vec<usize> = (0..batch).map(|b| (7 * b + 3) % cols).collect();
                for (b, &c) in probe.iter().enumerate() {
                    xs[b * cols + c] = 1.0;
                }
                let y = gpu.gemm(&enc.bits, &tbl, lut, rows as u32, cols as u32, cfg.k_bits, cfg.l_bits, &xs, batch);
                for r in 0..rows {
                    for (b, &c) in probe.iter().enumerate() {
                        let recovered = (y[r * batch + b] * 4096.0).round() as i32;
                        let expected = want[r * cols + c];
                        assert_eq!(recovered, expected, "GEMM one-hot Q12 diverged: k={} L={} B={batch} r={r} c={c}", cfg.k_bits, cfg.l_bits);
                    }
                }

                let xs: Vec<f32> = (0..batch * cols).map(|i| ((i as f32) * 0.031).cos()).collect();
                let y = gpu.gemm(&enc.bits, &tbl, lut, rows as u32, cols as u32, cfg.k_bits, cfg.l_bits, &xs, batch);
                let inv = 1.0f32 / 4096.0;
                for r in 0..rows {
                    for b in 0..batch {
                        let mut acc = 0.0f32;
                        for blk in 0..cols / 256 {
                            let mut p = 0.0f32;
                            for j in 0..256 {
                                let c = blk * 256 + j;
                                p += (want[r * cols + c] as f32) * inv * xs[b * cols + c];
                            }
                            acc += p;
                        }
                        let denom = acc.abs().max(1e-3);
                        assert!((y[r * batch + b] - acc).abs() / denom < 1e-3, "GEMM y diverged: k={} L={} B={batch} r={r} b={b}: GPU {} vs CPU {acc}", cfg.k_bits, cfg.l_bits, y[r * batch + b]);
                    }
                }
            }
        }
    }

    #[test]
    fn bitslice_prepared_decode_is_bit_identical() {
        use strand_quant::decode::decode_tensor_fixed;
        use strand_quant::encode::{encode_tensor_with, EncodeOpts};

        let Some(gpu) = BitsliceGpu::new() else {
            eprintln!("[strand-decode-kernel] no Metal device; skipping prepared GPU test");
            return;
        };

        let mut prepared = Vec::new();
        let mut wants = Vec::new();
        for (i, cfg) in [TrellisConfig::for_bpw(3.0), TrellisConfig::for_bpw_l(2.0, 12), TrellisConfig::for_bpw_l(2.0, 5)].into_iter().enumerate() {
            let n = 700 + i * 531;
            let w: Vec<f32> = (0..n).map(|j| ((j as f32) * 0.0117).sin() * 0.4).collect();
            let enc = encode_tensor_with(&w, &cfg, &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() });
            let want = decode_tensor_fixed(&enc, &cfg);
            let p = gpu.prepare(&enc, &cfg).expect("prepare");
            assert!(p.gpu_bytes() > 0);
            assert_eq!(gpu.decode_q12_prepared(&p), want, "prepared GPU decode diverged (cfg {i})");
            prepared.push(p);
            wants.push(want);
        }
        gpu.dispatch_prepared_all(&prepared);
        for (p, want) in prepared.iter().zip(wants.iter()) {
            assert_eq!(&p.read_out(), want, "batched prepared dispatch diverged");
        }
        let vcfg = TrellisConfig::for_bpw(3.0).with_vec_dim(2);
        let venc = encode_tensor(&(0..600).map(|i| (i as f32 * 0.01).cos()).collect::<Vec<_>>(), &vcfg);
        assert!(gpu.prepare(&venc, &vcfg).is_none(), "vec trellis must fall back");
    }
}
