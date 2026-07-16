#![allow(unsafe_code)]

use metal::{Buffer, CommandQueue, CompileOptions, Device, MTLResourceOptions, MTLSize, NSUInteger};

use crate::codebook::codebook_lut;
use crate::decode::eff_min_q;
use crate::encode::{build_sub_levels, choose_affine_min, choose_scale_q, choose_sub_scales, n_sub_blocks, pack_sub_scales, BlockMeta, EncodeOpts, EncodedTensor, SUB_BLOCK, SUB_SCALE_UNITY};
use crate::trellis::{push_bits, TrellisConfig};

const TROPICAL_MSL: &str = include_str!("../shaders/strand_tropical_encode.metal");
const SEARCH_MSL: &str = include_str!("../shaders/strand_scale_search.metal");

const SEARCH_TG_THREADS: usize = 256;

const MAX_BACK_BYTES: usize = 256 * 1024 * 1024;

const STRETCH_TG_THREADS: usize = 256;

const MAX_GPU_STATES: usize = 1 << 12;

#[repr(C)]
#[derive(Clone, Copy)]
struct TropicalParams {
    block_len: u32,
    num_states: u32,
    k_bits: u32,

    n_sub: u32,
    sub_size: u32,
    max_block_len: u32,
    tail_bite: u32,

    use_device_cost: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SearchParams {
    block_len: u32,
    num_states: u32,
    k_bits: u32,
    n_sub: u32,
    sub_size: u32,
    max_block_len: u32,
    n_sub_max: u32,
    adaptive: u32,
    affine_min: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

struct BlockPrep {
    blen: usize,
    scale_q: i32,
    mults: Vec<u8>,
    min_base_q: i32,
    min_codes: Vec<u8>,

    levels_f32: Vec<f32>,
    tail_bite: bool,
}

pub struct TropicalEncoder {
    device: Device,
    queue: CommandQueue,
    pipeline: metal::ComputePipelineState,
    search_pipeline: metal::ComputePipelineState,
    max_threads: usize,
    max_tg_bytes: usize,
}

impl TropicalEncoder {
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;

        let opts = CompileOptions::new();

        opts.set_fast_math_enabled(false);

        let build = |src: &str, name: &str| -> Option<metal::ComputePipelineState> {
            let lib = match device.new_library_with_source(src, &opts) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("[strand-quant] {name} shader compile error: {e}");
                    return None;
                }
            };
            let func = match lib.get_function(name, None) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("[strand-quant] {name} function lookup error: {e}");
                    return None;
                }
            };
            match device.new_compute_pipeline_state_with_function(&func) {
                Ok(p) => Some(p),
                Err(e) => {
                    eprintln!("[strand-quant] {name} pipeline error: {e}");
                    None
                }
            }
        };
        let pipeline = build(TROPICAL_MSL, "tropical_encode_block")?;
        let search_pipeline = build(SEARCH_MSL, "scale_search_block")?;

        let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
        let max_tg_bytes = device.max_threadgroup_memory_length() as usize;
        let queue = device.new_command_queue();
        eprintln!("[strand-quant] tropical encode ready: {} (max_threads_per_tg={max_threads}, tg_mem={max_tg_bytes}B, fast-math OFF, scale-search lane compiled)", device.name(),);
        Some(Self { device, queue, pipeline, search_pipeline, max_threads, max_tg_bytes })
    }

    pub fn supports(&self, cfg: &TrellisConfig) -> bool {
        cfg.vec_dim() == 1 && cfg.num_states() <= MAX_GPU_STATES
    }

    fn viterbi_geometry(&self, num_states: usize) -> (usize, bool) {
        let tg_fits = 2 * num_states * std::mem::size_of::<f32>() + 16 <= self.max_tg_bytes;
        if tg_fits {
            (num_states.min(self.max_threads), false)
        } else {
            (STRETCH_TG_THREADS.min(self.max_threads), true)
        }
    }

    pub fn encode_tensor(&self, weights: &[f32], cfg: &TrellisConfig, opts: &EncodeOpts) -> Option<EncodedTensor> {
        if !self.supports(cfg) {
            return None;
        }
        if weights.is_empty() {
            return Some(EncodedTensor { bits: Vec::new(), blocks: Vec::new(), total: 0, has_rht_seed: false, tail_biting: opts.tail_biting, has_affine_min: opts.affine_min });
        }

        let num_states = cfg.num_states();
        let n_sub_max = cfg.block_len.div_ceil(SUB_BLOCK);
        let n_blocks = weights.len().div_ceil(cfg.block_len);

        let back_bytes_per_block = cfg.block_len * num_states;
        let batch_blocks = (MAX_BACK_BYTES / back_bytes_per_block.max(1)).clamp(1, n_blocks);

        let lut = codebook_lut(cfg.l_bits);
        let lut_buf = self.upload(lut);

        let mut bits = Vec::new();
        let mut bit_cursor = 0usize;
        let mut blocks_meta: Vec<BlockMeta> = Vec::with_capacity(n_blocks);

        let timing = std::env::var_os("STRAND_TROPICAL_TIMING").is_some_and(|v| v == "1");
        let (mut t_pack, mut t_gpu, mut t_asm) = (0.0f64, 0.0f64, 0.0f64);

        let mut bi_base = 0usize;
        while bi_base < n_blocks {
            let bi_end = (bi_base + batch_blocks).min(n_blocks);
            let w_start = bi_base * cfg.block_len;
            let w_end = (bi_end * cfg.block_len).min(weights.len());

            let t0 = std::time::Instant::now();
            let blens: Vec<usize> = (bi_base..bi_end)
                .map(|bi| {
                    let lo = bi * cfg.block_len;
                    ((lo + cfg.block_len).min(weights.len())) - lo
                })
                .collect();
            t_pack += t0.elapsed().as_secs_f64();

            let t0 = std::time::Instant::now();
            let out = self.run_batch_full(&weights[w_start..w_end], &blens, cfg, opts, n_sub_max, &lut_buf)?;
            t_gpu += t0.elapsed().as_secs_f64();

            let t0 = std::time::Instant::now();
            for (i, &blen) in blens.iter().enumerate() {
                let p_off = i * cfg.block_len;
                for &sym in &out.paths[p_off..p_off + blen] {
                    push_bits(&mut bits, &mut bit_cursor, sym as usize, cfg.k_bits);
                }
                let n_sub = n_sub_blocks(blen);
                let m_off = i * n_sub_max;
                blocks_meta.push(BlockMeta {
                    scale_q: out.scales[i],
                    sub_scales: if opts.adaptive || opts.affine_min { pack_sub_scales(&out.mults[m_off..m_off + n_sub]) } else { Vec::new() },
                    min_base_q: out.min_bases[i],
                    mins: if opts.affine_min { pack_sub_scales(&out.min_codes[m_off..m_off + n_sub]) } else { Vec::new() },
                    init_state: out.inits[i],
                    n: blen as u32,
                });
            }
            t_asm += t0.elapsed().as_secs_f64();
            bi_base = bi_end;
        }

        if timing {
            let tot = (t_pack + t_gpu + t_asm).max(1e-12);
            eprintln!(
                "[tropical timing/full-gpu] N={} pack={:.3}s ({:.0}%) gpu={:.3}s ({:.0}%) assemble={:.3}s ({:.0}%)",
                weights.len(),
                t_pack,
                100.0 * t_pack / tot,
                t_gpu,
                100.0 * t_gpu / tot,
                t_asm,
                100.0 * t_asm / tot,
            );
        }

        Some(EncodedTensor { bits, blocks: blocks_meta, total: weights.len(), has_rht_seed: false, tail_biting: opts.tail_biting, has_affine_min: opts.affine_min })
    }

    fn run_batch_full(&self, batch_weights: &[f32], blens: &[usize], cfg: &TrellisConfig, opts: &EncodeOpts, n_sub_max: usize, lut_buf: &Buffer) -> Option<FullBatchOut> {
        let n = blens.len();
        let num_states = cfg.num_states();
        let mbl = cfg.block_len;
        let (tg_threads, dev_cost_rows) = self.viterbi_geometry(num_states);

        let mut w_pad = vec![0.0f32; n * mbl];
        {
            let mut src = 0usize;
            for (i, &blen) in blens.iter().enumerate() {
                w_pad[i * mbl..i * mbl + blen].copy_from_slice(&batch_weights[src..src + blen]);
                src += blen;
            }
        }
        let w_buf = self.upload(&w_pad);

        let search_params: Vec<SearchParams> = blens
            .iter()
            .map(|&blen| SearchParams {
                block_len: blen as u32,
                num_states: num_states as u32,
                k_bits: cfg.k_bits,
                n_sub: n_sub_blocks(blen) as u32,
                sub_size: SUB_BLOCK as u32,
                max_block_len: mbl as u32,
                n_sub_max: n_sub_max as u32,
                adaptive: opts.adaptive as u32,
                affine_min: opts.affine_min as u32,
                _p0: 0,
                _p1: 0,
                _p2: 0,
            })
            .collect();
        let tropical_params: Vec<TropicalParams> = blens
            .iter()
            .map(|&blen| {
                let nk = cfg.num_steps(blen) * cfg.k_bits as usize;
                TropicalParams {
                    block_len: blen as u32,
                    num_states: num_states as u32,
                    k_bits: cfg.k_bits,
                    n_sub: n_sub_max as u32,
                    sub_size: SUB_BLOCK as u32,
                    max_block_len: mbl as u32,
                    tail_bite: (opts.tail_biting && nk >= cfg.l_bits as usize) as u32,
                    use_device_cost: dev_cost_rows as u32,
                }
            })
            .collect();
        let sp_buf = self.upload(&search_params);
        let tp_buf = self.upload(&tropical_params);

        let lv_stride = n_sub_max * num_states;
        let levels_buf = self.device.new_buffer((n * lv_stride * std::mem::size_of::<f32>()).max(4) as NSUInteger, MTLResourceOptions::StorageModePrivate);
        let back_buf = self.device.new_buffer((n * mbl * num_states).max(4) as NSUInteger, MTLResourceOptions::StorageModePrivate);

        let dev_cost_buf = self.device.new_buffer(if dev_cost_rows { (n * 2 * num_states * std::mem::size_of::<f32>()) as NSUInteger } else { 4 }, MTLResourceOptions::StorageModePrivate);

        let path_buf = self.alloc_shared(n * mbl);
        let init_buf = self.alloc_shared(n * std::mem::size_of::<u32>());
        let scale_buf = self.alloc_shared(n * std::mem::size_of::<i32>());
        let mult_buf = self.alloc_shared(n * n_sub_max);
        let minb_buf = self.alloc_shared(n * std::mem::size_of::<i32>());
        let minc_buf = self.alloc_shared(n * n_sub_max);

        let cmd = self.queue.new_command_buffer();

        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.search_pipeline);
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(lut_buf), 0);
            enc.set_buffer(2, Some(&levels_buf), 0);
            enc.set_buffer(3, Some(&sp_buf), 0);
            enc.set_buffer(4, Some(&scale_buf), 0);
            enc.set_buffer(5, Some(&mult_buf), 0);
            enc.set_buffer(6, Some(&minb_buf), 0);
            enc.set_buffer(7, Some(&minc_buf), 0);

            let shf_bytes = ((n_sub_max * 65 + 4) * std::mem::size_of::<f32>()) as NSUInteger;
            let shi_bytes = ((2 * n_sub_max + 4) * std::mem::size_of::<i32>()) as NSUInteger;
            let shw_bytes = (mbl * std::mem::size_of::<f32>()) as NSUInteger;
            enc.set_threadgroup_memory_length(0, shf_bytes);
            enc.set_threadgroup_memory_length(1, shi_bytes);
            enc.set_threadgroup_memory_length(2, shw_bytes);
            let tg = SEARCH_TG_THREADS.min(self.search_pipeline.max_total_threads_per_threadgroup() as usize);
            enc.dispatch_thread_groups(MTLSize { width: n as NSUInteger, height: 1, depth: 1 }, MTLSize { width: tg as NSUInteger, height: 1, depth: 1 });
            enc.end_encoding();
        }

        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline);
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(&levels_buf), 0);
            enc.set_buffer(2, Some(&back_buf), 0);
            enc.set_buffer(3, Some(&tp_buf), 0);
            enc.set_buffer(4, Some(&path_buf), 0);
            enc.set_buffer(5, Some(&init_buf), 0);
            enc.set_buffer(6, Some(&dev_cost_buf), 0);

            let cost_bytes = if dev_cost_rows { 16 } else { (num_states * std::mem::size_of::<f32>()) as NSUInteger };
            enc.set_threadgroup_memory_length(0, cost_bytes);
            enc.set_threadgroup_memory_length(1, cost_bytes);
            enc.set_threadgroup_memory_length(2, 16);
            enc.dispatch_thread_groups(MTLSize { width: n as NSUInteger, height: 1, depth: 1 }, MTLSize { width: tg_threads as NSUInteger, height: 1, depth: 1 });
            enc.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();

        Some(FullBatchOut {
            paths: self.read_u8(&path_buf, n * mbl)?,
            inits: self.read_u32(&init_buf, n)?,
            scales: self.read_i32(&scale_buf, n)?,
            mults: self.read_u8(&mult_buf, n * n_sub_max)?,
            min_bases: self.read_i32(&minb_buf, n)?,
            min_codes: self.read_u8(&minc_buf, n * n_sub_max)?,
        })
    }

    pub fn encode_tensor_prep_cpu(&self, weights: &[f32], cfg: &TrellisConfig, opts: &EncodeOpts) -> Option<EncodedTensor> {
        if !self.supports(cfg) {
            return None;
        }
        if weights.is_empty() {
            return Some(EncodedTensor { bits: Vec::new(), blocks: Vec::new(), total: 0, has_rht_seed: false, tail_biting: opts.tail_biting, has_affine_min: opts.affine_min });
        }

        let num_states = cfg.num_states();
        let n_sub_max = cfg.block_len.div_ceil(SUB_BLOCK);
        let n_blocks = weights.len().div_ceil(cfg.block_len);

        let back_bytes_per_block = cfg.block_len * num_states;
        let batch_blocks = (MAX_BACK_BYTES / back_bytes_per_block.max(1)).clamp(1, n_blocks);

        let mut bits = Vec::new();
        let mut bit_cursor = 0usize;
        let mut blocks_meta: Vec<BlockMeta> = Vec::with_capacity(n_blocks);

        let timing = std::env::var_os("STRAND_TROPICAL_TIMING").is_some_and(|v| v == "1");
        let (mut t_prep, mut t_gpu, mut t_asm) = (0.0f64, 0.0f64, 0.0f64);

        let mut bi_base = 0usize;
        while bi_base < n_blocks {
            let bi_end = (bi_base + batch_blocks).min(n_blocks);
            let w_start = bi_base * cfg.block_len;
            let w_end = (bi_end * cfg.block_len).min(weights.len());

            let t0 = std::time::Instant::now();
            let preps = prep_blocks_parallel(&weights[w_start..w_end], cfg, opts, num_states, n_sub_max);
            t_prep += t0.elapsed().as_secs_f64();

            let t0 = std::time::Instant::now();
            let (paths, inits) = self.run_batch(&weights[w_start..w_end], &preps, cfg, n_sub_max)?;
            t_gpu += t0.elapsed().as_secs_f64();

            let t0 = std::time::Instant::now();
            for (i, prep) in preps.iter().enumerate() {
                let p_off = i * cfg.block_len;
                for &sym in &paths[p_off..p_off + prep.blen] {
                    push_bits(&mut bits, &mut bit_cursor, sym as usize, cfg.k_bits);
                }
                blocks_meta.push(BlockMeta {
                    scale_q: prep.scale_q,
                    sub_scales: if opts.adaptive || opts.affine_min { pack_sub_scales(&prep.mults) } else { Vec::new() },
                    min_base_q: prep.min_base_q,
                    mins: if opts.affine_min { pack_sub_scales(&prep.min_codes) } else { Vec::new() },
                    init_state: inits[i],
                    n: prep.blen as u32,
                });
            }
            t_asm += t0.elapsed().as_secs_f64();
            bi_base = bi_end;
        }

        if timing {
            let tot = (t_prep + t_gpu + t_asm).max(1e-12);
            eprintln!(
                "[tropical timing] N={} prep={:.3}s ({:.0}%) gpu={:.3}s ({:.0}%) assemble={:.3}s ({:.0}%)",
                weights.len(),
                t_prep,
                100.0 * t_prep / tot,
                t_gpu,
                100.0 * t_gpu / tot,
                t_asm,
                100.0 * t_asm / tot,
            );
        }

        Some(EncodedTensor { bits, blocks: blocks_meta, total: weights.len(), has_rht_seed: false, tail_biting: opts.tail_biting, has_affine_min: opts.affine_min })
    }

    fn run_batch(&self, batch_weights: &[f32], preps: &[BlockPrep], cfg: &TrellisConfig, n_sub_max: usize) -> Option<(Vec<u8>, Vec<u32>)> {
        let n = preps.len();
        let num_states = cfg.num_states();
        let mbl = cfg.block_len;
        let (tg_threads, dev_cost_rows) = self.viterbi_geometry(num_states);

        let mut w_pad = vec![0.0f32; n * mbl];
        {
            let mut src = 0usize;
            for (i, p) in preps.iter().enumerate() {
                w_pad[i * mbl..i * mbl + p.blen].copy_from_slice(&batch_weights[src..src + p.blen]);
                src += p.blen;
            }
        }
        let w_buf = self.upload(&w_pad);

        let lv_stride = n_sub_max * num_states;
        let mut lv_flat = vec![0.0f32; n * lv_stride];
        for (i, p) in preps.iter().enumerate() {
            lv_flat[i * lv_stride..(i + 1) * lv_stride].copy_from_slice(&p.levels_f32);
        }
        let lv_buf = self.upload(&lv_flat);

        let back_buf = self.device.new_buffer((n * mbl * num_states).max(4) as NSUInteger, MTLResourceOptions::StorageModePrivate);

        let dev_cost_buf = self.device.new_buffer(if dev_cost_rows { (n * 2 * num_states * std::mem::size_of::<f32>()) as NSUInteger } else { 4 }, MTLResourceOptions::StorageModePrivate);

        let params: Vec<TropicalParams> = preps
            .iter()
            .map(|p| TropicalParams {
                block_len: p.blen as u32,
                num_states: num_states as u32,
                k_bits: cfg.k_bits,
                n_sub: n_sub_max as u32,
                sub_size: SUB_BLOCK as u32,
                max_block_len: mbl as u32,
                tail_bite: p.tail_bite as u32,
                use_device_cost: dev_cost_rows as u32,
            })
            .collect();
        let params_buf = self.upload(&params);

        let path_buf = self.alloc_shared(n * mbl);
        let init_buf = self.alloc_shared(n * std::mem::size_of::<u32>());

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&lv_buf), 0);
        enc.set_buffer(2, Some(&back_buf), 0);
        enc.set_buffer(3, Some(&params_buf), 0);
        enc.set_buffer(4, Some(&path_buf), 0);
        enc.set_buffer(5, Some(&init_buf), 0);
        enc.set_buffer(6, Some(&dev_cost_buf), 0);

        let cost_bytes = if dev_cost_rows { 16 } else { (num_states * std::mem::size_of::<f32>()) as NSUInteger };
        enc.set_threadgroup_memory_length(0, cost_bytes);
        enc.set_threadgroup_memory_length(1, cost_bytes);
        enc.set_threadgroup_memory_length(2, 16);
        enc.dispatch_thread_groups(MTLSize { width: n as NSUInteger, height: 1, depth: 1 }, MTLSize { width: tg_threads as NSUInteger, height: 1, depth: 1 });
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let paths = self.read_u8(&path_buf, n * mbl)?;
        let inits = self.read_u32(&init_buf, n)?;
        Some((paths, inits))
    }

    fn upload<T: Copy>(&self, data: &[T]) -> Buffer {
        let byte_len = data.len() * std::mem::size_of::<T>();
        let buf = self.device.new_buffer(byte_len.max(4) as NSUInteger, MTLResourceOptions::StorageModeShared);

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.contents() as *mut u8, byte_len);
        }
        buf
    }

    fn alloc_shared(&self, byte_len: usize) -> Buffer {
        self.device.new_buffer(byte_len.max(4) as NSUInteger, MTLResourceOptions::StorageModeShared)
    }

    fn read_u8(&self, buf: &Buffer, len: usize) -> Option<Vec<u8>> {
        let ptr = buf.contents() as *const u8;
        if ptr.is_null() {
            return None;
        }

        Some(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec())
    }

    fn read_u32(&self, buf: &Buffer, len: usize) -> Option<Vec<u32>> {
        let ptr = buf.contents() as *const u32;
        if ptr.is_null() {
            return None;
        }

        Some(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec())
    }

    fn read_i32(&self, buf: &Buffer, len: usize) -> Option<Vec<i32>> {
        let ptr = buf.contents() as *const i32;
        if ptr.is_null() {
            return None;
        }

        Some(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec())
    }
}

struct FullBatchOut {
    paths: Vec<u8>,

    inits: Vec<u32>,

    scales: Vec<i32>,

    mults: Vec<u8>,

    min_bases: Vec<i32>,

    min_codes: Vec<u8>,
}

fn prep_blocks_parallel(weights: &[f32], cfg: &TrellisConfig, opts: &EncodeOpts, num_states: usize, n_sub_max: usize) -> Vec<BlockPrep> {
    let lut = codebook_lut(cfg.l_bits);
    let chunks: Vec<&[f32]> = weights.chunks(cfg.block_len).collect();
    let n = chunks.len();
    let nthreads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1).min(n.max(1));
    let per = n.div_ceil(nthreads);

    let prep_one = |chunk: &[f32]| -> BlockPrep {
        let scale_q = choose_scale_q(chunk, lut, cfg);
        let mults = if opts.adaptive { choose_sub_scales(chunk, scale_q, lut, cfg) } else { vec![SUB_SCALE_UNITY; n_sub_blocks(chunk.len())] };
        let (min_base_q, min_codes) = if opts.affine_min { choose_affine_min(chunk, scale_q, &mults, lut, cfg) } else { (0, Vec::new()) };
        let mins_eff: Vec<i32> = min_codes.iter().map(|&c| eff_min_q(min_base_q, c)).collect();

        let sub_levels = build_sub_levels(scale_q, &mults, &mins_eff, lut, num_states);
        let mut levels_f32 = vec![0.0f32; n_sub_max * num_states];
        for (j, row) in sub_levels.iter().enumerate() {
            for (s, &v) in row.iter().enumerate() {
                levels_f32[j * num_states + s] = v as f32;
            }
        }
        let nk = cfg.num_steps(chunk.len()) * cfg.k_bits as usize;
        let tail_bite = opts.tail_biting && nk >= cfg.l_bits as usize;
        BlockPrep { blen: chunk.len(), scale_q, mults, min_base_q, min_codes, levels_f32, tail_bite }
    };

    if nthreads <= 1 {
        return chunks.iter().map(|c| prep_one(c)).collect();
    }
    let mut out: Vec<Option<BlockPrep>> = (0..n).map(|_| None).collect();
    std::thread::scope(|s| {
        for (slot_chunk, work_chunk) in out.chunks_mut(per).zip(chunks.chunks(per)) {
            s.spawn(move || {
                for (slot, chunk) in slot_chunk.iter_mut().zip(work_chunk.iter()) {
                    *slot = Some(prep_one(chunk));
                }
            });
        }
    });
    out.into_iter().map(|p| p.expect("prep slot filled")).collect()
}
