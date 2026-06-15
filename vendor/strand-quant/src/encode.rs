
use wide::{f32x4, f64x4, CmpLt};

use crate::codebook::QUANTILE_SHIFT;
use crate::decode::{eff_scale_q, reconstruct_q, SCALE_SHIFT};
use crate::trellis::{push_bits, read_bits, TrellisConfig};

#[cfg(any(target_os = "macos", feature = "cuda"))]
use std::sync::OnceLock;

#[cfg(target_os = "macos")]
static METAL: OnceLock<Option<crate::metal_backend::MetalViterbi>> = OnceLock::new();
#[cfg(target_os = "macos")]
fn metal_viterbi() -> Option<&'static crate::metal_backend::MetalViterbi> {
    METAL.get_or_init(|| crate::metal_backend::MetalViterbi::new()).as_ref()
}

#[cfg(feature = "cuda")]
static CUDA: OnceLock<Option<crate::cuda_backend::CudaViterbi>> = OnceLock::new();
#[cfg(feature = "cuda")]
fn cuda_viterbi() -> Option<&'static crate::cuda_backend::CudaViterbi> {
    CUDA.get_or_init(|| crate::cuda_backend::CudaViterbi::new()).as_ref()
}

pub const SUB_BLOCK: usize = 32;

pub const SUB_SCALE_UNITY: u8 = 63;

pub const AFFINE_MIN_LEVELS: u32 = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockMeta {
    
    pub scale_q: i32,
    
    pub sub_scales: Vec<u8>,
    
    pub min_base_q: i32,
    
    pub mins: Vec<u8>,
    
    pub init_state: u32,
    
    pub n: u32,
}

pub fn pack_sub_scales(mults: &[u8]) -> Vec<u8> {
    let total_bits = mults.len() * 6;
    let mut bytes = vec![0u8; total_bits.div_ceil(8)];
    let mut cursor = 0usize;
    for &m in mults {
        let v = m as usize;
        for b in 0..6 {
            if (v >> b) & 1 != 0 {
                bytes[cursor >> 3] |= 1u8 << (cursor & 7);
            }
            cursor += 1;
        }
    }
    bytes
}

#[inline]
pub fn unpack_sub_scales(bytes: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut cursor = 0usize;
    for _ in 0..n {
        let mut v = 0u8;
        for b in 0..6 {
            let bit_idx = cursor + b;
            let byte = bit_idx >> 3;
            let bit = if byte < bytes.len() {
                (bytes[byte] >> (bit_idx & 7)) & 1
            } else {
                0
            };
            v |= bit << b;
        }
        out.push(v);
        cursor += 6;
    }
    out
}

#[inline]
pub fn n_sub_blocks(n: usize) -> usize {
    n.div_ceil(SUB_BLOCK)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedTensor {
    
    pub bits: Vec<u8>,
    
    pub blocks: Vec<BlockMeta>,
    
    pub total: usize,
    
    pub has_rht_seed: bool,
    
    pub tail_biting: bool,
    
    pub has_affine_min: bool,
}

pub const RHT_SEED_BITS: usize = 64;

impl EncodedTensor {
    
    pub fn payload_bpw(&self, cfg: &TrellisConfig) -> f64 {
        cfg.k_bits as f64 / cfg.vec_dim() as f64
    }

    pub fn index_symbols(&self, cfg: &TrellisConfig) -> Vec<u8> {
        let k = cfg.k_bits;
        let input_mask = cfg.num_inputs() - 1;
        let mut out = Vec::with_capacity(self.total);
        let mut bit_cursor = 0usize;
        for blk in &self.blocks {
            for _ in 0..cfg.num_steps(blk.n as usize) {
                let sym = read_bits(&self.bits, bit_cursor, k) & input_mask;
                bit_cursor += k as usize;
                out.push(sym as u8);
            }
        }
        out
    }

    fn block_side_bits(&self, cfg: &TrellisConfig) -> usize {
        let mut bits = 16; 
        for b in &self.blocks {
            let n_sub = n_sub_blocks(b.n as usize);
            let affine = if self.has_affine_min { 6 * n_sub } else { 0 };
            let nk = cfg.num_steps(b.n as usize) * cfg.k_bits as usize;
            let tail_bit = self.tail_biting && nk >= cfg.l_bits as usize;
            let init_bits = if tail_bit { 0 } else { cfg.l_bits as usize };
            bits += 32 + 6 * n_sub + affine + init_bits;
        }
        bits
    }

    pub fn total_bpw(&self, cfg: &TrellisConfig) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let payload: usize = self
            .blocks
            .iter()
            .map(|b| cfg.num_steps(b.n as usize) * cfg.k_bits as usize)
            .sum();
        let side = self.block_side_bits(cfg);
        let rht = if self.has_rht_seed { RHT_SEED_BITS } else { 0 };
        (payload + side + rht) as f64 / self.total as f64
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EncodeOpts {
    
    pub adaptive: bool,
    
    pub tail_biting: bool,
    
    pub affine_min: bool,
    
    pub silence_bonus: f64,
    
    pub entropy_bonus_scale: f64,
    
    pub entropy_bonus_two_pass: bool,
}

impl Default for EncodeOpts {
    fn default() -> Self {
        EncodeOpts {
            adaptive: true,
            tail_biting: false,
            affine_min: false,
            silence_bonus: 0.0,
            entropy_bonus_scale: 0.0,
            entropy_bonus_two_pass: false,
        }
    }
}

pub fn encode_tensor(weights: &[f32], cfg: &TrellisConfig) -> EncodedTensor {
    encode_tensor_with(weights, cfg, &EncodeOpts::default())
}

pub fn encode_tensor_opts(weights: &[f32], cfg: &TrellisConfig, adaptive: bool) -> EncodedTensor {
    encode_tensor_with(
        weights,
        cfg,
        &EncodeOpts {
            adaptive,
            ..Default::default()
        },
    )
}

pub fn compute_block_entropy(syms: &[u32], k_bits: u8) -> f64 {
    if syms.is_empty() || k_bits == 0 {
        return 1.0; 
    }
    let num_levels = 1usize << k_bits;
    
    let n = syms.len();
    let mut counts_stack = [0u32; 16]; 
    let counts: &mut [u32] = if num_levels <= 16 {
        &mut counts_stack[..num_levels]
    } else {
        
        return 0.0;
    };
    for &s in syms {
        let idx = (s as usize) & (num_levels - 1);
        
        counts[idx] = counts[idx].saturating_add(1);
    }
    
    let n_f64 = n as f64;
    let k_f64 = k_bits as f64;
    let mut h_nats = 0.0f64; 
    for &c in counts.iter().take(num_levels) {
        if c == 0 {
            continue;
        }
        let p = (c as f64) / n_f64;
        h_nats -= p * p.ln(); 
    }
    
    let h_normalised = h_nats / (k_f64 * core::f64::consts::LN_2);
    
    let h_clamped = h_normalised.clamp(0.0, 1.0);
    1.0 - h_clamped
}

pub fn extract_block_symbols(enc: &EncodedTensor, b: usize, cfg: &TrellisConfig) -> Vec<u32> {
    if b >= enc.blocks.len() {
        return Vec::new();
    }
    
    let mut bit_offset = 0usize;
    for blk in enc.blocks.iter().take(b) {
        bit_offset += cfg.num_steps(blk.n as usize) * cfg.k_bits as usize;
    }
    let n_steps = cfg.num_steps(enc.blocks[b].n as usize);
    let k = cfg.k_bits as usize;
    let input_mask = cfg.num_inputs() - 1;
    let mut syms = Vec::with_capacity(n_steps);
    for i in 0..n_steps {
        let sym = read_bits(&enc.bits, bit_offset + i * k, cfg.k_bits) & input_mask;
        syms.push(sym as u32);
    }
    syms
}

pub fn encode_tensor_with(weights: &[f32], cfg: &TrellisConfig, opts: &EncodeOpts) -> EncodedTensor {
    
    if cfg.vec_dim() > 1 {
        // Scalar codebook sourced per `cfg.codebook_mode` (byte-identical under
        // either mode, Variant A exact), then expanded to the vector LUT.
        let scalar = cfg.codebook();
        let lut = vector_lut_from_scalar(&scalar, cfg.vec_dim());
        return encode_tensor_with_lut(weights, cfg, opts, &lut);
    }
    
    let gpu_eligible = !opts.tail_biting && !opts.affine_min
        && std::env::var_os("STRAND_NO_GPU").is_none()
        && !f32_metric_from_env();

    #[cfg(target_os = "macos")]
    if gpu_eligible {
        if let Some(enc) = encode_tensor_with_metal(weights, cfg, opts) {
            return enc;
        }
    }

    #[cfg(feature = "cuda")]
    if gpu_eligible {
        if let Some(enc) = encode_tensor_with_cuda(weights, cfg, opts) {
            return enc;
        }
    }

    encode_tensor_with_cpu(weights, cfg, opts)
}

fn encode_tensor_with_cpu(weights: &[f32], cfg: &TrellisConfig, opts: &EncodeOpts) -> EncodedTensor {
    // Codebook sourced per `cfg.codebook_mode`. Under `ComputedAcklam` this is a
    // freshly materialised `Vec` whose entries are byte-identical to the frozen
    // table (Variant A), so every encode metric below is unchanged. The encoder
    // keeps the `&[i32]` interface; only the *source* of the array differs.
    let lut = cfg.codebook();
    encode_tensor_with_lut(weights, cfg, opts, &lut)
}

pub fn f32_metric_from_env() -> bool {
    matches!(std::env::var_os("STRAND_F32_METRIC"), Some(v) if v == "1")
}

pub fn f32_search_from_env() -> bool {
    matches!(std::env::var_os("STRAND_F32_SEARCH"), Some(v) if v == "1")
}

pub fn encode_tensor_with_lut(
    weights: &[f32],
    cfg: &TrellisConfig,
    opts: &EncodeOpts,
    lut: &[i32],
) -> EncodedTensor {
    let f32_metric = f32_metric_from_env();
    encode_tensor_with_lut_metric_search(
        weights,
        cfg,
        opts,
        lut,
        f32_metric,
        f32_metric && f32_search_from_env(),
    )
}

pub fn encode_tensor_with_lut_metric(
    weights: &[f32],
    cfg: &TrellisConfig,
    opts: &EncodeOpts,
    lut: &[i32],
    f32_metric: bool,
) -> EncodedTensor {
    encode_tensor_with_lut_metric_search(weights, cfg, opts, lut, f32_metric, false)
}

pub fn encode_tensor_with_lut_metric_search(
    weights: &[f32],
    cfg: &TrellisConfig,
    opts: &EncodeOpts,
    lut: &[i32],
    f32_metric: bool,
    f32_search: bool,
) -> EncodedTensor {
    if cfg.vec_dim() > 1 {
        return encode_tensor_with_lut_vec(weights, cfg, opts, lut);
    }

    let psi_active = opts.entropy_bonus_scale != 0.0;

    if psi_active && opts.entropy_bonus_two_pass {
        
        let pass1_opts = EncodeOpts {
            silence_bonus: 0.0,
            entropy_bonus_scale: 0.0,
            entropy_bonus_two_pass: false,
            ..*opts
        };
        let pass1 = encode_tensor_with_lut_metric_search(
            weights, cfg, &pass1_opts, lut, f32_metric, f32_search,
        );
        
        let compressibilities: Vec<f64> = (0..pass1.blocks.len())
            .map(|b| {
                let syms = extract_block_symbols(&pass1, b, cfg);
                compute_block_entropy(&syms, cfg.k_bits as u8)
            })
            .collect();
        
        let num_states = cfg.num_states();
        let mut bits = Vec::new();
        let mut bit_cursor = 0usize;
        let mut blocks = Vec::new();
        let mut back_buf: Vec<u32> = vec![u32::MAX; cfg.block_len * num_states];

        for (bi, chunk) in weights.chunks(cfg.block_len).enumerate() {
            let scale_q = if f32_search {
                choose_scale_q_f32(chunk, lut, cfg)
            } else {
                choose_scale_q(chunk, lut, cfg)
            };
            let mults = if opts.adaptive {
                if f32_search {
                    choose_sub_scales_f32(chunk, scale_q, lut, cfg)
                } else {
                    choose_sub_scales(chunk, scale_q, lut, cfg)
                }
            } else {
                vec![SUB_SCALE_UNITY; n_sub_blocks(chunk.len())]
            };
            let (min_base_q, min_codes) = if opts.affine_min {
                if f32_search {
                    choose_affine_min_f32(chunk, scale_q, &mults, lut, cfg)
                } else {
                    choose_affine_min(chunk, scale_q, &mults, lut, cfg)
                }
            } else {
                (0, Vec::new())
            };
            let mins_eff: Vec<i32> = min_codes
                .iter()
                .map(|&c| crate::decode::eff_min_q(min_base_q, c))
                .collect();
            
            let entropy_bonus = opts.entropy_bonus_scale * compressibilities[bi];
            let total_bonus = opts.silence_bonus + entropy_bonus;
            let (path, init_state) = viterbi_path_buf(
                chunk, scale_q, &mults, &mins_eff, lut, cfg,
                opts.tail_biting, f32_metric, total_bonus, &mut back_buf,
            );
            for &sym in &path {
                push_bits(&mut bits, &mut bit_cursor, sym as usize, cfg.k_bits);
            }
            blocks.push(BlockMeta {
                scale_q,
                sub_scales: pack_sub_scales(&mults),
                min_base_q,
                mins: if opts.affine_min { pack_sub_scales(&min_codes) } else { Vec::new() },
                init_state: init_state as u32,
                n: chunk.len() as u32,
            });
        }

        return EncodedTensor {
            bits,
            blocks,
            total: weights.len(),
            has_rht_seed: false,
            tail_biting: opts.tail_biting,
            has_affine_min: opts.affine_min,
        };
    }

    let mut bits = Vec::new();
    let mut bit_cursor = 0usize;
    let mut blocks = Vec::new();

    let num_states = cfg.num_states();
    let mut back_buf: Vec<u32> = vec![u32::MAX; cfg.block_len * num_states];

    const ROLLING_WINDOW: usize = 8;
    let mut rolling_buf = [0.0f64; ROLLING_WINDOW];
    let mut rolling_pos = 0usize;   
    let mut rolling_filled = 0usize; 

    for chunk in weights.chunks(cfg.block_len) {
        let scale_q = if f32_search {
            choose_scale_q_f32(chunk, lut, cfg)
        } else {
            choose_scale_q(chunk, lut, cfg)
        };
        let mults = if opts.adaptive {
            if f32_search {
                choose_sub_scales_f32(chunk, scale_q, lut, cfg)
            } else {
                choose_sub_scales(chunk, scale_q, lut, cfg)
            }
        } else {
            vec![SUB_SCALE_UNITY; n_sub_blocks(chunk.len())]
        };
        let (min_base_q, min_codes) = if opts.affine_min {
            if f32_search {
                choose_affine_min_f32(chunk, scale_q, &mults, lut, cfg)
            } else {
                choose_affine_min(chunk, scale_q, &mults, lut, cfg)
            }
        } else {
            (0, Vec::new())
        };
        let mins_eff: Vec<i32> = min_codes
            .iter()
            .map(|&c| crate::decode::eff_min_q(min_base_q, c))
            .collect();

        let entropy_bonus = if psi_active {
            let rolling_mean = if rolling_filled == 0 {
                0.0
            } else {
                let sum: f64 = rolling_buf[..rolling_filled.min(ROLLING_WINDOW)]
                    .iter()
                    .sum();
                sum / rolling_filled.min(ROLLING_WINDOW) as f64
            };
            opts.entropy_bonus_scale * rolling_mean
        } else {
            0.0
        };

        let total_bonus = opts.silence_bonus + entropy_bonus;
        let (path, init_state) = viterbi_path_buf(
            chunk, scale_q, &mults, &mins_eff, lut, cfg,
            opts.tail_biting, f32_metric, total_bonus, &mut back_buf,
        );

        if psi_active {
            let syms: Vec<u32> = path.iter().copied().collect();
            let c = compute_block_entropy(&syms, cfg.k_bits as u8);
            rolling_buf[rolling_pos % ROLLING_WINDOW] = c;
            rolling_pos += 1;
            if rolling_filled < ROLLING_WINDOW {
                rolling_filled += 1;
            }
        }

        for &sym in &path {
            push_bits(&mut bits, &mut bit_cursor, sym as usize, cfg.k_bits);
        }
        blocks.push(BlockMeta {
            scale_q,
            sub_scales: pack_sub_scales(&mults),
            min_base_q,
            mins: if opts.affine_min { pack_sub_scales(&min_codes) } else { Vec::new() },
            init_state: init_state as u32,
            n: chunk.len() as u32,
        });
    }

    EncodedTensor {
        bits,
        blocks,
        total: weights.len(),
        has_rht_seed: false,
        tail_biting: opts.tail_biting,
        has_affine_min: opts.affine_min,
    }
}

pub fn vector_lut_from_scalar(scalar: &[i32], d: usize) -> Vec<i32> {
    let d = d.max(1);
    let mut out = Vec::with_capacity(scalar.len() * d);
    for &v in scalar {
        for _ in 0..d {
            out.push(v);
        }
    }
    out
}

pub fn encode_tensor_with_lut_vec(
    weights: &[f32],
    cfg: &TrellisConfig,
    opts: &EncodeOpts,
    lut: &[i32],
) -> EncodedTensor {
    let d = cfg.vec_dim();
    let num_states = cfg.num_states();
    debug_assert_eq!(lut.len(), num_states * d, "vector LUT must be [2^L * d]");

    let mut bits = Vec::new();
    let mut bit_cursor = 0usize;
    let mut blocks = Vec::new();

    let max_steps = cfg.num_steps(cfg.block_len);
    let mut back_buf: Vec<u32> = vec![u32::MAX; max_steps * num_states];

    for chunk in weights.chunks(cfg.block_len) {
        let scale_q = choose_scale_q_vec(chunk, lut, cfg);
        let mults = if opts.adaptive {
            choose_sub_scales_vec(chunk, scale_q, lut, cfg)
        } else {
            vec![SUB_SCALE_UNITY; n_sub_blocks(chunk.len())]
        };
        let (min_base_q, min_codes) = if opts.affine_min {
            choose_affine_min_vec(chunk, scale_q, &mults, lut, cfg)
        } else {
            (0, Vec::new())
        };
        let mins_eff: Vec<i32> = min_codes
            .iter()
            .map(|&c| crate::decode::eff_min_q(min_base_q, c))
            .collect();

        let (path, init_state) = viterbi_path_buf_vec(
            chunk, scale_q, &mults, &mins_eff, lut, cfg,
            opts.tail_biting, &mut back_buf,
        );
        for &sym in &path {
            push_bits(&mut bits, &mut bit_cursor, sym as usize, cfg.k_bits);
        }
        blocks.push(BlockMeta {
            scale_q,
            sub_scales: pack_sub_scales(&mults),
            min_base_q,
            mins: if opts.affine_min { pack_sub_scales(&min_codes) } else { Vec::new() },
            init_state: init_state as u32,
            n: chunk.len() as u32,
        });
    }

    EncodedTensor {
        bits,
        blocks,
        total: weights.len(),
        has_rht_seed: false,
        tail_biting: opts.tail_biting,
        has_affine_min: opts.affine_min,
    }
}

#[cfg(target_os = "macos")]
fn encode_tensor_with_metal(
    weights: &[f32],
    cfg: &TrellisConfig,
    opts: &EncodeOpts,
) -> Option<EncodedTensor> {
    let m = metal_viterbi()?;
    // Codebook sourced per `cfg.codebook_mode`. Under `ComputedAcklam` this is a
    // freshly materialised `Vec` whose entries are byte-identical to the frozen
    // table (Variant A), so every encode metric below is unchanged. The encoder
    // keeps the `&[i32]` interface; only the *source* of the array differs.
    let lut_cb = cfg.codebook();
    let lut: &[i32] = &lut_cb;
    let num_states = cfg.num_states();
    let n_sub_per_block = cfg.block_len.div_ceil(SUB_BLOCK);
    let q_to_real = 1.0f32 / (1u32 << QUANTILE_SHIFT) as f32;

    struct BlockPrep {
        chunk_offset: usize,
        chunk_len: usize,
        scale_q: i32,
        mults: Vec<u8>,
        levels_f32: Vec<f32>,
    }

    let build_levels = |scale_q: i32, mults: &[u8]| -> Vec<f32> {
        let mut lv = vec![0.0f32; n_sub_per_block * num_states];
        for (j, &mult) in mults.iter().enumerate() {
            let es = eff_scale_q(scale_q, mult);
            let base = j * num_states;
            for s in 0..num_states {
                lv[base + s] = (reconstruct_q(es, lut[s]) as f32) * q_to_real;
            }
        }
        lv
    };

    let n_blocks = weights.len().div_ceil(cfg.block_len);
    let mut preps: Vec<BlockPrep> = Vec::with_capacity(n_blocks);
    let mut offset = 0usize;
    for chunk in weights.chunks(cfg.block_len) {
        let scale_q = choose_scale_q(chunk, lut, cfg);
        let mults = if opts.adaptive {
            choose_sub_scales(chunk, scale_q, lut, cfg)
        } else {
            vec![SUB_SCALE_UNITY; n_sub_blocks(chunk.len())]
        };
        let levels_f32 = build_levels(scale_q, &mults);
        preps.push(BlockPrep { chunk_offset: offset, chunk_len: chunk.len(), scale_q, mults, levels_f32 });
        offset += chunk.len();
    }

    const MAX_BACK_BYTES: usize = 512 * 1024 * 1024;
    let bytes_per_block = cfg.block_len * num_states * std::mem::size_of::<u32>();
    let batch_size = (MAX_BACK_BYTES / bytes_per_block.max(1)).max(64).min(n_blocks);

    let input_mask = (1usize << cfg.k_bits) - 1;
    let mut all_paths: Vec<Vec<u32>> = vec![Vec::new(); n_blocks];
    let mut all_init_states: Vec<usize> = vec![0usize; n_blocks];

    let mut bi_base = 0;
    while bi_base < n_blocks {
        let bi_end = (bi_base + batch_size).min(n_blocks);
        let w_start = preps[bi_base].chunk_offset;
        let w_end = if bi_end < n_blocks { preps[bi_end].chunk_offset } else { weights.len() };
        let batch_weights = &weights[w_start..w_end];
        let batch_lens: Vec<usize> = preps[bi_base..bi_end].iter().map(|p| p.chunk_len).collect();

        let sub_levels: Vec<f32> = preps[bi_base..bi_end]
            .iter().flat_map(|p| p.levels_f32.iter().copied()).collect();
        let gpu = m.run_blocks(
            batch_weights, &sub_levels, &batch_lens, cfg.block_len, num_states, cfg.k_bits as u32,
        )?;
        let mbl = gpu.max_block_len;

        for (i, prep) in preps[bi_base..bi_end].iter().enumerate() {
            let blen = prep.chunk_len;
            let fc_base = i * num_states;
            let back_base = i * mbl * num_states;
            let terminal = (0..num_states)
                .min_by(|&a, &b| {
                    gpu.final_cost[fc_base + a]
                        .partial_cmp(&gpu.final_cost[fc_base + b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            let mut path = vec![0u32; blen];
            let mut state = terminal;
            for step in (0..blen).rev() {
                path[step] = (state & input_mask) as u32;
                state = gpu.back_flat[back_base + step * num_states + state] as usize;
            }
            all_paths[bi_base + i] = path;
            all_init_states[bi_base + i] = state;
        }
        bi_base = bi_end;
    }

    let mut bits = Vec::new();
    let mut bit_cursor = 0usize;
    let mut blocks = Vec::new();
    for (bi, prep) in preps.iter().enumerate() {
        for &sym in &all_paths[bi] {
            push_bits(&mut bits, &mut bit_cursor, sym as usize, cfg.k_bits);
        }
        blocks.push(BlockMeta {
            scale_q: prep.scale_q,
            sub_scales: pack_sub_scales(&prep.mults),
            min_base_q: 0,
            mins: Vec::new(),
            init_state: all_init_states[bi] as u32,
            n: prep.chunk_len as u32,
        });
    }

    Some(EncodedTensor {
        bits,
        blocks,
        total: weights.len(),
        has_rht_seed: false,
        tail_biting: false,
        has_affine_min: false,
    })
}

#[cfg(feature = "cuda")]
fn encode_tensor_with_cuda(
    weights: &[f32],
    cfg: &TrellisConfig,
    opts: &EncodeOpts,
) -> Option<EncodedTensor> {
    let cu = cuda_viterbi()?;
    // Codebook sourced per `cfg.codebook_mode`. Under `ComputedAcklam` this is a
    // freshly materialised `Vec` whose entries are byte-identical to the frozen
    // table (Variant A), so every encode metric below is unchanged. The encoder
    // keeps the `&[i32]` interface; only the *source* of the array differs.
    let lut_cb = cfg.codebook();
    let lut: &[i32] = &lut_cb;
    let num_states = cfg.num_states();
    let n_sub_per_block = cfg.block_len.div_ceil(SUB_BLOCK);
    let q_to_real = 1.0f32 / (1u32 << QUANTILE_SHIFT) as f32;

    struct BlockPrep {
        chunk_offset: usize,
        chunk_len: usize,
        scale_q: i32,
        mults: Vec<u8>,
        levels_f32: Vec<f32>,
    }

    let build_levels = |scale_q: i32, mults: &[u8]| -> Vec<f32> {
        let mut lv = vec![0.0f32; n_sub_per_block * num_states];
        for (j, &mult) in mults.iter().enumerate() {
            let es = eff_scale_q(scale_q, mult);
            let base = j * num_states;
            for s in 0..num_states {
                lv[base + s] = (reconstruct_q(es, lut[s]) as f32) * q_to_real;
            }
        }
        lv
    };

    let n_blocks = weights.len().div_ceil(cfg.block_len);
    let mut preps: Vec<BlockPrep> = Vec::with_capacity(n_blocks);
    let mut offset = 0usize;
    for chunk in weights.chunks(cfg.block_len) {
        let scale_q = choose_scale_q(chunk, lut, cfg);
        let mults = if opts.adaptive {
            choose_sub_scales(chunk, scale_q, lut, cfg)
        } else {
            vec![SUB_SCALE_UNITY; n_sub_blocks(chunk.len())]
        };
        let levels_f32 = build_levels(scale_q, &mults);
        preps.push(BlockPrep { chunk_offset: offset, chunk_len: chunk.len(), scale_q, mults, levels_f32 });
        offset += chunk.len();
    }

    const MAX_BACK_BYTES_CUDA: usize = 512 * 1024 * 1024; 
    let bytes_per_block = cfg.block_len * num_states * std::mem::size_of::<u32>();
    let batch_size = (MAX_BACK_BYTES_CUDA / bytes_per_block.max(1)).max(64).min(n_blocks);

    let input_mask = (1usize << cfg.k_bits) - 1;
    let mut all_paths: Vec<Vec<u32>> = vec![Vec::new(); n_blocks];
    let mut all_init_states: Vec<usize> = vec![0usize; n_blocks];

    let mut bi_base = 0;
    while bi_base < n_blocks {
        let bi_end = (bi_base + batch_size).min(n_blocks);
        let w_start = preps[bi_base].chunk_offset;
        let w_end = if bi_end < n_blocks { preps[bi_end].chunk_offset } else { weights.len() };
        let batch_weights = &weights[w_start..w_end];
        let batch_lens: Vec<usize> = preps[bi_base..bi_end].iter().map(|p| p.chunk_len).collect();

        let sub_levels: Vec<f32> = preps[bi_base..bi_end]
            .iter().flat_map(|p| p.levels_f32.iter().copied()).collect();
        let gpu = cu.run_blocks(
            batch_weights, &sub_levels, &batch_lens, cfg.block_len, num_states, cfg.k_bits as u32,
        )?;
        let mbl = gpu.max_block_len;

        for (i, prep) in preps[bi_base..bi_end].iter().enumerate() {
            let blen = prep.chunk_len;
            let fc_base = i * num_states;
            let back_base = i * mbl * num_states;
            let terminal = (0..num_states)
                .min_by(|&a, &b| {
                    gpu.final_cost[fc_base + a]
                        .partial_cmp(&gpu.final_cost[fc_base + b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            let mut path = vec![0u32; blen];
            let mut state = terminal;
            for step in (0..blen).rev() {
                path[step] = (state & input_mask) as u32;
                state = gpu.back_flat[back_base + step * num_states + state] as usize;
            }
            all_paths[bi_base + i] = path;
            all_init_states[bi_base + i] = state;
        }
        bi_base = bi_end;
    }

    let mut bits = Vec::new();
    let mut bit_cursor = 0usize;
    let mut blocks = Vec::new();
    for (bi, prep) in preps.iter().enumerate() {
        for &sym in &all_paths[bi] {
            push_bits(&mut bits, &mut bit_cursor, sym as usize, cfg.k_bits);
        }
        blocks.push(BlockMeta {
            scale_q: prep.scale_q,
            sub_scales: pack_sub_scales(&prep.mults),
            min_base_q: 0,
            mins: Vec::new(),
            init_state: all_init_states[bi] as u32,
            n: prep.chunk_len as u32,
        });
    }

    Some(EncodedTensor {
        bits,
        blocks,
        total: weights.len(),
        has_rht_seed: false,
        tail_biting: false,
        has_affine_min: false,
    })
}

#[inline]
fn level_real(scale_real: f64, q_q12: i32) -> f64 {
    scale_real * (q_q12 as f64) / (1u32 << QUANTILE_SHIFT) as f64
}

pub(crate) fn choose_scale_q(weights: &[f32], lut: &[i32], cfg: &TrellisConfig) -> i32 {
    if weights.is_empty() {
        return 0;
    }
    let absmax = weights.iter().fold(0.0f64, |m, &w| m.max(w.abs() as f64));
    if absmax == 0.0 {
        return 0;
    }
    let q_max = (lut[lut.len() - 1] as f64) / (1u32 << QUANTILE_SHIFT) as f64;
    let q_max = if q_max > 0.0 { q_max } else { 1.0 };
    let seed = absmax / q_max;

    const MULTS: [f64; 11] = [
        0.55, 0.65, 0.75, 0.85, 0.92, 1.0, 1.08, 1.18, 1.30, 1.45, 1.65,
    ];
    let mut best_scale = seed;
    let mut best_mse = f64::INFINITY;
    for &m in &MULTS {
        let s = seed * m;
        let mse = greedy_replay_mse(weights, s, lut, cfg);
        if mse < best_mse {
            best_mse = mse;
            best_scale = s;
        }
    }

    let scale_q = (best_scale * (1u64 << SCALE_SHIFT) as f64).round();
    scale_q.clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

pub(crate) fn choose_sub_scales(chunk: &[f32], scale_q: i32, lut: &[i32], cfg: &TrellisConfig) -> Vec<u8> {
    let n_sub = n_sub_blocks(chunk.len());
    let mut mults = Vec::with_capacity(n_sub);
    for sb in 0..n_sub {
        let lo = sb * SUB_BLOCK;
        let hi = (lo + SUB_BLOCK).min(chunk.len());
        let sub = &chunk[lo..hi];
        if sub.iter().all(|&w| w == 0.0) {
            mults.push(SUB_SCALE_UNITY); 
            continue;
        }
        let mut best_c = SUB_SCALE_UNITY;
        let mut best_mse = f64::INFINITY;
        for c in 0u8..=63 {
            let es = eff_scale_q(scale_q, c);
            if es == 0 {
                continue;
            }
            let es_real = (es as f64) / (1u64 << SCALE_SHIFT) as f64;
            let mse = greedy_replay_mse(sub, es_real, lut, cfg);
            if mse < best_mse {
                best_mse = mse;
                best_c = c;
            }
        }
        mults.push(best_c);
    }
    mults
}

pub(crate) fn choose_affine_min(
    chunk: &[f32],
    scale_q: i32,
    mults: &[u8],
    lut: &[i32],
    cfg: &TrellisConfig,
) -> (i32, Vec<u8>) {
    let n_sub = n_sub_blocks(chunk.len());
    let means: Vec<f64> = (0..n_sub)
        .map(|sb| {
            let lo = sb * SUB_BLOCK;
            let hi = (lo + SUB_BLOCK).min(chunk.len());
            let s = &chunk[lo..hi];
            if s.is_empty() {
                0.0
            } else {
                s.iter().map(|&w| w as f64).sum::<f64>() / s.len() as f64
            }
        })
        .collect();
    let base_abs = means
        .iter()
        .copied()
        .fold(0.0f64, |b, m| if m.abs() > b { m.abs() } else { b });
    if base_abs < 1e-12 {
        return (0, vec![0u8; n_sub]);
    }
    let min_base_q = (base_abs * (1u32 << QUANTILE_SHIFT) as f64).round() as i32;
    let q_to_real = 1.0f64 / (1u32 << QUANTILE_SHIFT) as f64;
    let mut codes = Vec::with_capacity(n_sub);
    for (sb, &mult) in mults.iter().enumerate() {
        let lo = sb * SUB_BLOCK;
        let hi = (lo + SUB_BLOCK).min(chunk.len());
        let sub = &chunk[lo..hi];
        let es = eff_scale_q(scale_q, mult);
        let es_real = (es as f64) / (1u64 << SCALE_SHIFT) as f64;
        let positive_side = means[sb] >= 0.0;
        let code_range = if positive_side { 32u8..=63 } else { 0u8..=31 };
        let mut best_c = if positive_side { 32 } else { 0 }; 
        let mut best_mse = f64::INFINITY;
        for c in code_range {
            let off_real = (crate::decode::eff_min_q(min_base_q, c) as f64) * q_to_real;
            let mse = greedy_replay_mse_off(sub, es_real, off_real, lut, cfg);
            if mse < best_mse {
                best_mse = mse;
                best_c = c;
            }
        }
        codes.push(best_c);
    }
    (min_base_q, codes)
}

fn greedy_replay_mse(weights: &[f32], scale_real: f64, lut: &[i32], cfg: &TrellisConfig) -> f64 {
    greedy_replay_mse_off(weights, scale_real, 0.0, lut, cfg)
}

fn greedy_replay_mse_off(
    weights: &[f32],
    scale_real: f64,
    offset: f64,
    lut: &[i32],
    cfg: &TrellisConfig,
) -> f64 {
    let mut state = 0usize;
    let mut acc = 0.0f64;
    for &w in weights {
        let target = w as f64;
        let mut best_in = 0usize;
        let mut best_err = f64::INFINITY;
        for inp in 0..cfg.num_inputs() {
            let ns = cfg.next_state(state, inp);
            let lvl = level_real(scale_real, lut[ns]) + offset;
            let e = (target - lvl) * (target - lvl);
            if e < best_err {
                best_err = e;
                best_in = inp;
            }
        }
        state = cfg.next_state(state, best_in);
        acc += best_err;
    }
    acc
}

const Q12_INV_F32: f32 = 1.0 / (1u32 << QUANTILE_SHIFT) as f32; 
const S16_INV_F32: f32 = 1.0 / (1u64 << SCALE_SHIFT) as f32;

#[inline]
fn round_pos_f32_to_i32(x: f32) -> i32 {
    if !x.is_finite() || x >= 2147483648.0f32 {
        return i32::MAX;
    }
    let xf = x.floor();
    let r = (xf as i64) + i64::from(x - xf >= 0.5);
    r.min(i32::MAX as i64) as i32
}

fn greedy_replay_mse_off_f32(
    weights: &[f32],
    scale: f32,
    offset: f32,
    lut: &[i32],
    cfg: &TrellisConfig,
) -> f32 {
    let mut state = 0usize;
    let mut acc = 0.0f32;
    for &w in weights {
        let mut best_in = 0usize;
        let mut best_err = f32::INFINITY;
        for inp in 0..cfg.num_inputs() {
            let ns = cfg.next_state(state, inp);
            let q = (lut[ns] as f32) * Q12_INV_F32;
            let t1 = scale * q;
            let lvl = t1 + offset;
            let d = w - lvl;
            let e = d * d;
            if e < best_err {
                best_err = e;
                best_in = inp;
            }
        }
        state = cfg.next_state(state, best_in);
        acc += best_err;
    }
    acc
}

pub(crate) fn choose_scale_q_f32(weights: &[f32], lut: &[i32], cfg: &TrellisConfig) -> i32 {
    if weights.is_empty() {
        return 0;
    }
    let mut absmax = 0.0f32;
    for &w in weights {
        let a = w.abs();
        if a > absmax {
            absmax = a;
        }
    }
    if absmax == 0.0 {
        return 0;
    }
    let q_max = {
        let q = (lut[lut.len() - 1] as f32) * Q12_INV_F32;
        if q > 0.0 { q } else { 1.0 }
    };
    let seed = absmax / q_max;

    const MULTS_F32: [f32; 11] = [
        0.55, 0.65, 0.75, 0.85, 0.92, 1.0, 1.08, 1.18, 1.30, 1.45, 1.65,
    ];
    let mut best_scale = seed;
    let mut best_mse = f32::INFINITY;
    for &m in &MULTS_F32 {
        let s = seed * m;
        let mse = greedy_replay_mse_off_f32(weights, s, 0.0, lut, cfg);
        if mse < best_mse {
            best_mse = mse;
            best_scale = s;
        }
    }
    round_pos_f32_to_i32(best_scale * 65536.0f32)
}

pub(crate) fn choose_sub_scales_f32(
    chunk: &[f32],
    scale_q: i32,
    lut: &[i32],
    cfg: &TrellisConfig,
) -> Vec<u8> {
    let n_sub = n_sub_blocks(chunk.len());
    let mut mults = Vec::with_capacity(n_sub);
    for sb in 0..n_sub {
        let lo = sb * SUB_BLOCK;
        let hi = (lo + SUB_BLOCK).min(chunk.len());
        let sub = &chunk[lo..hi];
        if sub.iter().all(|&w| w == 0.0) {
            mults.push(SUB_SCALE_UNITY);
            continue;
        }
        let mut best_c = SUB_SCALE_UNITY;
        let mut best_mse = f32::INFINITY;
        for c in 0u8..=63 {
            let es = eff_scale_q(scale_q, c);
            if es == 0 {
                continue;
            }
            let es_real = (es as f32) * S16_INV_F32;
            let mse = greedy_replay_mse_off_f32(sub, es_real, 0.0, lut, cfg);
            if mse < best_mse {
                best_mse = mse;
                best_c = c;
            }
        }
        mults.push(best_c);
    }
    mults
}

pub(crate) fn choose_affine_min_f32(
    chunk: &[f32],
    scale_q: i32,
    mults: &[u8],
    lut: &[i32],
    cfg: &TrellisConfig,
) -> (i32, Vec<u8>) {
    let n_sub = n_sub_blocks(chunk.len());
    let means: Vec<f32> = (0..n_sub)
        .map(|sb| {
            let lo = sb * SUB_BLOCK;
            let hi = (lo + SUB_BLOCK).min(chunk.len());
            let s = &chunk[lo..hi];
            if s.is_empty() {
                0.0
            } else {
                let mut sum = 0.0f32;
                for &w in s {
                    sum += w;
                }
                sum / (s.len() as f32)
            }
        })
        .collect();
    let base_abs = means
        .iter()
        .copied()
        .fold(0.0f32, |b, m| if m.abs() > b { m.abs() } else { b });
    if base_abs < 1e-12f32 {
        return (0, vec![0u8; n_sub]);
    }
    let min_base_q = round_pos_f32_to_i32(base_abs * 4096.0f32);
    let mut codes = Vec::with_capacity(n_sub);
    for (sb, &mult) in mults.iter().enumerate() {
        let lo = sb * SUB_BLOCK;
        let hi = (lo + SUB_BLOCK).min(chunk.len());
        let sub = &chunk[lo..hi];
        let es = eff_scale_q(scale_q, mult);
        let es_real = (es as f32) * S16_INV_F32;
        let positive_side = means[sb] >= 0.0;
        let code_range = if positive_side { 32u8..=63 } else { 0u8..=31 };
        let mut best_c = if positive_side { 32 } else { 0 };
        let mut best_mse = f32::INFINITY;
        for c in code_range {
            let off_real = (crate::decode::eff_min_q(min_base_q, c) as f32) * Q12_INV_F32;
            let mse = greedy_replay_mse_off_f32(sub, es_real, off_real, lut, cfg);
            if mse < best_mse {
                best_mse = mse;
                best_c = c;
            }
        }
        codes.push(best_c);
    }
    (min_base_q, codes)
}

#[allow(clippy::too_many_arguments)]
fn viterbi_path_buf(
    weights: &[f32],
    scale_q: i32,
    mults: &[u8],
    mins_eff: &[i32],
    lut: &[i32],
    cfg: &TrellisConfig,
    tail_biting: bool,
    f32_metric: bool,
    silence_bonus: f64,
    back_buf: &mut Vec<u32>,
) -> (Vec<u32>, usize) {
    let n = weights.len();
    let num_states = cfg.num_states();
    if n == 0 {
        return (Vec::new(), 0);
    }

    let sub_levels = build_sub_levels(scale_q, mults, mins_eff, lut, num_states);

    let nk = n * cfg.k_bits as usize;
    let can_tail_bite = tail_biting && nk >= cfg.l_bits as usize;

    let needed = n * num_states;
    if back_buf.len() < needed {
        back_buf.resize(needed, u32::MAX);
    }

    if f32_metric {
        let sub_levels_f32: Vec<Vec<f32>> = sub_levels
            .iter()
            .map(|v| v.iter().map(|&x| x as f32).collect())
            .collect();
        if !can_tail_bite {
            return backtrack_buf_f32(weights, &sub_levels_f32, cfg, None, None, back_buf);
        }
        let (final_s, _) = viterbi_forward_f32(weights, &sub_levels_f32, cfg, None);
        let path = backtrack_buf_f32(
            weights, &sub_levels_f32, cfg, Some(final_s), Some(final_s), back_buf,
        );
        return (path.0, final_s);
    }

    if silence_bonus != 0.0 {
        if !can_tail_bite {
            return backtrack_buf_with_bonus(
                weights, &sub_levels, cfg, None, None, silence_bonus, back_buf,
            );
        }
        let (final_s, _) =
            viterbi_forward_with_bonus(weights, &sub_levels, cfg, None, silence_bonus);
        let path = backtrack_buf_with_bonus(
            weights,
            &sub_levels,
            cfg,
            Some(final_s),
            Some(final_s),
            silence_bonus,
            back_buf,
        );
        return (path.0, final_s);
    }

    if !can_tail_bite {
        return backtrack_buf(weights, &sub_levels, cfg, None, None, back_buf);
    }

    let (final_s, _) = viterbi_forward(weights, &sub_levels, cfg, None);
    let path = backtrack_buf(weights, &sub_levels, cfg, Some(final_s), Some(final_s), back_buf);
    (path.0, final_s)
}

pub(crate) fn build_sub_levels(
    scale_q: i32,
    mults: &[u8],
    mins_eff: &[i32],
    lut: &[i32],
    num_states: usize,
) -> Vec<Vec<f64>> {
    let q_to_real = 1.0f64 / (1u32 << QUANTILE_SHIFT) as f64;
    mults
        .iter()
        .enumerate()
        .map(|(j, &m)| {
            let es = eff_scale_q(scale_q, m);
            let off = *mins_eff.get(j).unwrap_or(&0);
            (0..num_states)
                .map(|s| ((reconstruct_q(es, lut[s]) + off) as f64) * q_to_real)
                .collect::<Vec<f64>>()
        })
        .collect()
}

macro_rules! gen_step_dist {
    ($name:ident, $ty:ty) => {
        #[inline]
        fn $name(target: $ty, levels: &[$ty], dist: &mut [$ty]) {
            debug_assert_eq!(levels.len(), dist.len());
            for (d, &lvl) in dist.iter_mut().zip(levels.iter()) {
                let diff = target - lvl;
                *d = diff * diff;
            }
        }
    };
}
gen_step_dist!(step_dist_f64, f64);
gen_step_dist!(step_dist_f32, f32);

macro_rules! gen_relax_step {
    ($name:ident, $ty:ty, $vty:ty) => {
        #[inline]
        fn $name(
            cost: &[$ty],
            dist: &[$ty],
            next_cost: &mut [$ty],
            mut back_row: Option<&mut [u32]>,
            k: usize,
        ) {
            let num_states = cost.len();
            let num_inputs = 1usize << k;
            let n_groups = num_states >> k;
            debug_assert_eq!(dist.len(), num_states);
            debug_assert_eq!(next_cost.len(), num_states);
            if num_inputs >= 4 {
                for g in 0..n_groups {
                    let ns_base = g << k;
                    for off in (0..num_inputs).step_by(4) {
                        let i0 = ns_base + off;
                        let d_v =
                            <$vty>::from([dist[i0], dist[i0 + 1], dist[i0 + 2], dist[i0 + 3]]);
                        let mut best_v = <$vty>::splat(cost[g]) + d_v;
                        let mut best_t = <$vty>::splat(0 as $ty);
                        for t in 1..num_inputs {
                            let v = <$vty>::splat(cost[g + t * n_groups]) + d_v;
                            let m = v.cmp_lt(best_v);
                            best_v = m.blend(v, best_v);
                            best_t = m.blend(<$vty>::splat(t as $ty), best_t);
                        }
                        next_cost[i0..i0 + 4].copy_from_slice(&best_v.to_array());
                        if let Some(row) = back_row.as_deref_mut() {
                            let bt = best_t.to_array();
                            for lane in 0..4 {
                                row[i0 + lane] = (g + (bt[lane] as usize) * n_groups) as u32;
                            }
                        }
                    }
                }
            } else {
                for g in 0..n_groups {
                    for j in 0..num_inputs {
                        let ns = (g << k) + j;
                        let d = dist[ns];
                        let mut best = cost[g] + d;
                        let mut best_p = g;
                        for t in 1..num_inputs {
                            let p = g + t * n_groups;
                            let v = cost[p] + d;
                            if v < best {
                                best = v;
                                best_p = p;
                            }
                        }
                        next_cost[ns] = best;
                        if let Some(row) = back_row.as_deref_mut() {
                            row[ns] = best_p as u32;
                        }
                    }
                }
            }
        }
    };
}
gen_relax_step!(relax_step_f64, f64, f64x4);
gen_relax_step!(relax_step_f32, f32, f32x4);

macro_rules! gen_viterbi_sweeps {
    ($forward:ident, $backtrack:ident, $dist_fn:ident, $relax:ident, $pick:ident, $ty:ty) => {
        fn $forward(
            weights: &[f32],
            sub_levels: &[Vec<$ty>],
            cfg: &TrellisConfig,
            pin_start: Option<usize>,
        ) -> (usize, $ty) {
            let num_states = cfg.num_states();
            let inf = <$ty>::INFINITY;
            let mut cost: Vec<$ty> = match pin_start {
                Some(s0) => {
                    let mut c = vec![inf; num_states];
                    c[s0] = 0 as $ty;
                    c
                }
                None => vec![0 as $ty; num_states],
            };
            let k = cfg.k_bits as usize;
            let mut next_cost: Vec<$ty> = vec![inf; num_states];
            let mut dist: Vec<$ty> = vec![0 as $ty; num_states];
            for (i, &w) in weights.iter().enumerate() {
                $dist_fn(w as $ty, &sub_levels[i / SUB_BLOCK], &mut dist);
                $relax(&cost, &dist, &mut next_cost, None, k);
                std::mem::swap(&mut cost, &mut next_cost);
            }
            $pick(&cost)
        }

        fn $backtrack(
            weights: &[f32],
            sub_levels: &[Vec<$ty>],
            cfg: &TrellisConfig,
            pin_start: Option<usize>,
            final_s: Option<usize>,
            back_buf: &mut [u32],
        ) -> (Vec<u32>, usize) {
            let n = weights.len();
            let num_states = cfg.num_states();
            let inf = <$ty>::INFINITY;
            let mut cost: Vec<$ty> = match pin_start {
                Some(s0) => {
                    let mut c = vec![inf; num_states];
                    c[s0] = 0 as $ty;
                    c
                }
                None => vec![0 as $ty; num_states],
            };
            let k = cfg.k_bits as usize;
            let mut next_cost: Vec<$ty> = vec![inf; num_states];
            let mut dist: Vec<$ty> = vec![0 as $ty; num_states];
            for (i, &w) in weights.iter().enumerate() {
                $dist_fn(w as $ty, &sub_levels[i / SUB_BLOCK], &mut dist);
                let row = &mut back_buf[i * num_states..(i + 1) * num_states];
                $relax(&cost, &dist, &mut next_cost, Some(row), k);
                std::mem::swap(&mut cost, &mut next_cost);
            }
            let start_at = final_s.unwrap_or_else(|| $pick(&cost).0);
            let input_mask = cfg.num_inputs() - 1;
            let mut path = vec![0u32; n];
            let mut s = start_at;
            for i in (0..n).rev() {
                path[i] = (s & input_mask) as u32;
                s = back_buf[i * num_states + s] as usize;
            }
            (path, s)
        }
    };
}
gen_viterbi_sweeps!(
    viterbi_forward, backtrack_buf, step_dist_f64, relax_step_f64, pick_terminal, f64
);
gen_viterbi_sweeps!(
    viterbi_forward_f32, backtrack_buf_f32, step_dist_f32, relax_step_f32, pick_terminal_f32, f32
);

fn pick_terminal_f32(cost: &[f32]) -> (usize, f32) {
    let mut final_s = 0usize;
    let mut best = f32::INFINITY;
    for (s, &c) in cost.iter().enumerate() {
        if c < best {
            best = c;
            final_s = s;
        }
    }
    (final_s, best)
}

fn pick_terminal(cost: &[f64]) -> (usize, f64) {
    let inf = f64::INFINITY;
    let mut final_s = 0usize;
    let mut best = inf;
    for (s, &c) in cost.iter().enumerate() {
        if c < best {
            best = c;
            final_s = s;
        }
    }
    (final_s, best)
}

fn viterbi_forward_with_bonus(
    weights: &[f32],
    sub_levels: &[Vec<f64>],
    cfg: &TrellisConfig,
    pin_start: Option<usize>,
    silence_bonus: f64,
) -> (usize, f64) {
    let num_states = cfg.num_states();
    let k = cfg.k_bits as usize;
    let num_inputs = 1usize << k;
    let n_groups = num_states >> k;
    let inf = f64::INFINITY;
    let mut cost: Vec<f64> = match pin_start {
        Some(s0) => {
            let mut c = vec![inf; num_states];
            c[s0] = 0.0;
            c
        }
        None => vec![0.0f64; num_states],
    };
    let mut next_cost: Vec<f64> = vec![inf; num_states];
    for (i, &w) in weights.iter().enumerate() {
        let target = w as f64;
        let levels = &sub_levels[i / SUB_BLOCK];
        next_cost.fill(inf);
        for g in 0..n_groups {
            let ns_base = g << k;
            for j in 0..num_inputs {
                let ns = ns_base + j;
                let lvl = levels[ns];
                let diff = target - lvl;
                let mut d = diff * diff;
                if lvl == 0.0 {
                    d -= silence_bonus;
                }
                let mut best = cost[g] + d;
                let mut best_p = g;
                for t in 1..num_inputs {
                    let p = g + t * n_groups;
                    let v = cost[p] + d;
                    if v < best {
                        best = v;
                        best_p = p;
                    }
                }
                let _ = best_p;
                if best < next_cost[ns] {
                    next_cost[ns] = best;
                }
            }
        }
        std::mem::swap(&mut cost, &mut next_cost);
    }
    pick_terminal(&cost)
}

fn backtrack_buf_with_bonus(
    weights: &[f32],
    sub_levels: &[Vec<f64>],
    cfg: &TrellisConfig,
    pin_start: Option<usize>,
    final_s: Option<usize>,
    silence_bonus: f64,
    back_buf: &mut [u32],
) -> (Vec<u32>, usize) {
    let n = weights.len();
    let num_states = cfg.num_states();
    let k = cfg.k_bits as usize;
    let num_inputs = 1usize << k;
    let n_groups = num_states >> k;
    let inf = f64::INFINITY;
    let mut cost: Vec<f64> = match pin_start {
        Some(s0) => {
            let mut c = vec![inf; num_states];
            c[s0] = 0.0;
            c
        }
        None => vec![0.0f64; num_states],
    };
    let mut next_cost: Vec<f64> = vec![inf; num_states];
    for (i, &w) in weights.iter().enumerate() {
        let target = w as f64;
        let levels = &sub_levels[i / SUB_BLOCK];
        let row = &mut back_buf[i * num_states..(i + 1) * num_states];
        next_cost.fill(inf);
        for row_entry in row.iter_mut() {
            *row_entry = 0u32;
        }
        for g in 0..n_groups {
            let ns_base = g << k;
            for j in 0..num_inputs {
                let ns = ns_base + j;
                let lvl = levels[ns];
                let diff = target - lvl;
                let mut d = diff * diff;
                if lvl == 0.0 {
                    d -= silence_bonus;
                }
                let mut best = cost[g] + d;
                let mut best_p = g;
                for t in 1..num_inputs {
                    let p = g + t * n_groups;
                    let v = cost[p] + d;
                    if v < best {
                        best = v;
                        best_p = p;
                    }
                }
                next_cost[ns] = best;
                row[ns] = best_p as u32;
            }
        }
        std::mem::swap(&mut cost, &mut next_cost);
    }
    let start_at = final_s.unwrap_or_else(|| pick_terminal(&cost).0);
    let input_mask = num_inputs - 1;
    let mut path = vec![0u32; n];
    let mut s = start_at;
    for i in (0..n).rev() {
        path[i] = (s & input_mask) as u32;
        s = back_buf[i * num_states + s] as usize;
    }
    (path, s)
}

fn viterbi_forward_reference(
    weights: &[f32],
    sub_levels: &[Vec<f64>],
    cfg: &TrellisConfig,
    pin_start: Option<usize>,
) -> (usize, f64) {
    let num_states = cfg.num_states();
    let inf = f64::INFINITY;
    let mut cost = match pin_start {
        Some(s0) => {
            let mut c = vec![inf; num_states];
            c[s0] = 0.0;
            c
        }
        None => vec![0.0f64; num_states],
    };
    let num_inputs = cfg.num_inputs();
    let k = cfg.k_bits as usize;
    let use_simd = num_inputs >= 4;
    let mut next_cost = vec![inf; num_states];
    for (i, &w) in weights.iter().enumerate() {
        let target = w as f64;
        let levels = &sub_levels[i / SUB_BLOCK];
        next_cost.fill(inf);
        if use_simd {
            let target_v = f64x4::splat(target);
            let chunks = num_inputs / 4;
            for (s, &c) in cost.iter().enumerate() {
                if c == inf { continue; }
                let c_v = f64x4::splat(c);
                let ns_base = (s << k) & (num_states - 1);
                let lvl = &levels[ns_base..ns_base + num_inputs];
                let nc_dst = &mut next_cost[ns_base..ns_base + num_inputs];
                for ch in 0..chunks {
                    let off = ch * 4;
                    let lv = f64x4::from([lvl[off], lvl[off+1], lvl[off+2], lvl[off+3]]);
                    let d_v = target_v - lv;
                    let nc_v = c_v + d_v * d_v;
                    let old_v = f64x4::from([nc_dst[off], nc_dst[off+1], nc_dst[off+2], nc_dst[off+3]]);
                    let nc_a = nc_v.to_array();
                    let old_a = old_v.to_array();
                    for lane in 0..4 {
                        if nc_a[lane] < old_a[lane] { nc_dst[off + lane] = nc_a[lane]; }
                    }
                }
            }
        } else {
            for (s, &c) in cost.iter().enumerate() {
                if c == inf { continue; }
                for inp in 0..num_inputs {
                    let ns = cfg.next_state(s, inp);
                    let d = target - levels[ns];
                    let nc = c + d * d;
                    if nc < next_cost[ns] { next_cost[ns] = nc; }
                }
            }
        }
        std::mem::swap(&mut cost, &mut next_cost);
    }
    pick_terminal(&cost)
}

#[allow(clippy::ptr_arg)] 
fn backtrack_buf_reference(
    weights: &[f32],
    sub_levels: &[Vec<f64>],
    cfg: &TrellisConfig,
    pin_start: Option<usize>,
    final_s: Option<usize>,
    back_buf: &mut Vec<u32>,
) -> (Vec<u32>, usize) {
    let n = weights.len();
    let num_states = cfg.num_states();
    let inf = f64::INFINITY;
    let mut cost = match pin_start {
        Some(s0) => {
            let mut c = vec![inf; num_states];
            c[s0] = 0.0;
            c
        }
        None => vec![0.0f64; num_states],
    };
    let num_inputs = cfg.num_inputs();
    let k = cfg.k_bits as usize;
    let use_simd = num_inputs >= 4;
    let mut next_cost = vec![inf; num_states];
    for (i, &w) in weights.iter().enumerate() {
        let target = w as f64;
        let levels = &sub_levels[i / SUB_BLOCK];
        let row = &mut back_buf[i * num_states..(i + 1) * num_states];
        row.fill(u32::MAX);
        next_cost.fill(inf);
        if use_simd {
            let target_v = f64x4::splat(target);
            let chunks = num_inputs / 4;
            for (s, &c) in cost.iter().enumerate() {
                if c == inf { continue; }
                let c_v = f64x4::splat(c);
                let s_u32 = s as u32;
                let ns_base = (s << k) & (num_states - 1);
                let lvl = &levels[ns_base..ns_base + num_inputs];
                let nc_dst = &mut next_cost[ns_base..ns_base + num_inputs];
                let back = &mut row[ns_base..ns_base + num_inputs];
                for ch in 0..chunks {
                    let off = ch * 4;
                    let lv = f64x4::from([lvl[off], lvl[off+1], lvl[off+2], lvl[off+3]]);
                    let d_v = target_v - lv;
                    let nc_v = c_v + d_v * d_v;
                    let nc_a = nc_v.to_array();
                    for lane in 0..4 {
                        if nc_a[lane] < nc_dst[off + lane] {
                            nc_dst[off + lane] = nc_a[lane];
                            back[off + lane] = s_u32;
                        }
                    }
                }
            }
        } else {
            for (s, &c) in cost.iter().enumerate() {
                if c == inf { continue; }
                for inp in 0..num_inputs {
                    let ns = cfg.next_state(s, inp);
                    let d = target - levels[ns];
                    let nc = c + d * d;
                    if nc < next_cost[ns] {
                        next_cost[ns] = nc;
                        row[ns] = s as u32;
                    }
                }
            }
        }
        std::mem::swap(&mut cost, &mut next_cost);
    }
    let start_at = final_s.unwrap_or_else(|| pick_terminal(&cost).0);
    let input_mask = cfg.num_inputs() - 1;
    let mut path = vec![0u32; n];
    let mut s = start_at;
    for i in (0..n).rev() {
        path[i] = (s & input_mask) as u32;
        let pred = back_buf[i * num_states + s];
        debug_assert_ne!(pred, u32::MAX, "backtrack hit an unreached state");
        s = pred as usize;
    }
    (path, s)
}

#[allow(clippy::too_many_arguments)]
fn viterbi_path_buf_reference(
    weights: &[f32],
    scale_q: i32,
    mults: &[u8],
    mins_eff: &[i32],
    lut: &[i32],
    cfg: &TrellisConfig,
    tail_biting: bool,
    back_buf: &mut Vec<u32>,
) -> (Vec<u32>, usize) {
    let n = weights.len();
    let num_states = cfg.num_states();
    if n == 0 {
        return (Vec::new(), 0);
    }
    let sub_levels = build_sub_levels(scale_q, mults, mins_eff, lut, num_states);
    let nk = n * cfg.k_bits as usize;
    let can_tail_bite = tail_biting && nk >= cfg.l_bits as usize;
    let needed = n * num_states;
    if back_buf.len() < needed {
        back_buf.resize(needed, u32::MAX);
    }
    if !can_tail_bite {
        return backtrack_buf_reference(weights, &sub_levels, cfg, None, None, back_buf);
    }
    let (final_s, _) = viterbi_forward_reference(weights, &sub_levels, cfg, None);
    let path =
        backtrack_buf_reference(weights, &sub_levels, cfg, Some(final_s), Some(final_s), back_buf);
    (path.0, final_s)
}

#[doc(hidden)]
pub fn encode_tensor_with_lut_reference(
    weights: &[f32],
    cfg: &TrellisConfig,
    opts: &EncodeOpts,
    lut: &[i32],
) -> EncodedTensor {
    if cfg.vec_dim() > 1 {
        return encode_tensor_with_lut_vec_reference(weights, cfg, opts, lut);
    }
    let mut bits = Vec::new();
    let mut bit_cursor = 0usize;
    let mut blocks = Vec::new();
    let num_states = cfg.num_states();
    let mut back_buf: Vec<u32> = vec![u32::MAX; cfg.block_len * num_states];

    for chunk in weights.chunks(cfg.block_len) {
        let scale_q = choose_scale_q(chunk, lut, cfg);
        let mults = if opts.adaptive {
            choose_sub_scales(chunk, scale_q, lut, cfg)
        } else {
            vec![SUB_SCALE_UNITY; n_sub_blocks(chunk.len())]
        };
        let (min_base_q, min_codes) = if opts.affine_min {
            choose_affine_min(chunk, scale_q, &mults, lut, cfg)
        } else {
            (0, Vec::new())
        };
        let mins_eff: Vec<i32> = min_codes
            .iter()
            .map(|&c| crate::decode::eff_min_q(min_base_q, c))
            .collect();
        let (path, init_state) = viterbi_path_buf_reference(
            chunk, scale_q, &mults, &mins_eff, lut, cfg, opts.tail_biting, &mut back_buf,
        );
        for &sym in &path {
            push_bits(&mut bits, &mut bit_cursor, sym as usize, cfg.k_bits);
        }
        blocks.push(BlockMeta {
            scale_q,
            sub_scales: pack_sub_scales(&mults),
            min_base_q,
            mins: if opts.affine_min { pack_sub_scales(&min_codes) } else { Vec::new() },
            init_state: init_state as u32,
            n: chunk.len() as u32,
        });
    }

    EncodedTensor {
        bits,
        blocks,
        total: weights.len(),
        has_rht_seed: false,
        tail_biting: opts.tail_biting,
        has_affine_min: opts.affine_min,
    }
}

#[inline]
fn vec_level_real(scale_real: f64, off_real: f64, lut: &[i32], d: usize, s: usize, j: usize) -> f64 {
    scale_real * (lut[s * d + j] as f64) / (1u32 << QUANTILE_SHIFT) as f64 + off_real
}

fn greedy_replay_sse_vec_off(
    chunk: &[f32],
    scale_real: f64,
    off_real: f64,
    lut: &[i32],
    cfg: &TrellisConfig,
) -> f64 {
    let d = cfg.vec_dim();
    let n = chunk.len();
    let n_steps = n.div_ceil(d);
    let mut state = 0usize;
    let mut acc = 0.0f64;
    for t in 0..n_steps {
        let base_i = t * d;
        let present = (n - base_i).min(d);
        let mut best_in = 0usize;
        let mut best_err = f64::INFINITY;
        for inp in 0..cfg.num_inputs() {
            let ns = cfg.next_state(state, inp);
            let mut e = 0.0f64;
            for j in 0..present {
                let target = chunk[base_i + j] as f64;
                let lvl = vec_level_real(scale_real, off_real, lut, d, ns, j);
                let diff = target - lvl;
                e += diff * diff;
            }
            if e < best_err {
                best_err = e;
                best_in = inp;
            }
        }
        state = cfg.next_state(state, best_in);
        acc += best_err;
    }
    acc
}

fn choose_scale_q_vec(chunk: &[f32], lut: &[i32], cfg: &TrellisConfig) -> i32 {
    if chunk.is_empty() {
        return 0;
    }
    let absmax = chunk.iter().fold(0.0f64, |m, &w| m.max(w.abs() as f64));
    if absmax == 0.0 {
        return 0;
    }
    let q_max = lut
        .iter()
        .fold(0.0f64, |m, &q| m.max((q as f64).abs()))
        / (1u32 << QUANTILE_SHIFT) as f64;
    let q_max = if q_max > 0.0 { q_max } else { 1.0 };
    let seed = absmax / q_max;

    const MULTS: [f64; 11] = [
        0.55, 0.65, 0.75, 0.85, 0.92, 1.0, 1.08, 1.18, 1.30, 1.45, 1.65,
    ];
    let mut best_scale = seed;
    let mut best_sse = f64::INFINITY;
    for &m in &MULTS {
        let s = seed * m;
        let sse = greedy_replay_sse_vec_off(chunk, s, 0.0, lut, cfg);
        if sse < best_sse {
            best_sse = sse;
            best_scale = s;
        }
    }
    let scale_q = (best_scale * (1u64 << SCALE_SHIFT) as f64).round();
    scale_q.clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

fn choose_sub_scales_vec(chunk: &[f32], scale_q: i32, lut: &[i32], cfg: &TrellisConfig) -> Vec<u8> {
    let n_sub = n_sub_blocks(chunk.len());
    let mut mults = Vec::with_capacity(n_sub);
    for sb in 0..n_sub {
        let lo = sb * SUB_BLOCK;
        let hi = (lo + SUB_BLOCK).min(chunk.len());
        let sub = &chunk[lo..hi];
        if sub.iter().all(|&w| w == 0.0) {
            mults.push(SUB_SCALE_UNITY);
            continue;
        }
        let mut best_c = SUB_SCALE_UNITY;
        let mut best_sse = f64::INFINITY;
        for c in 0u8..=63 {
            let es = eff_scale_q(scale_q, c);
            if es == 0 {
                continue;
            }
            let es_real = (es as f64) / (1u64 << SCALE_SHIFT) as f64;
            let sse = greedy_replay_sse_vec_off(sub, es_real, 0.0, lut, cfg);
            if sse < best_sse {
                best_sse = sse;
                best_c = c;
            }
        }
        mults.push(best_c);
    }
    mults
}

fn choose_affine_min_vec(
    chunk: &[f32],
    scale_q: i32,
    mults: &[u8],
    lut: &[i32],
    cfg: &TrellisConfig,
) -> (i32, Vec<u8>) {
    let n_sub = n_sub_blocks(chunk.len());
    let means: Vec<f64> = (0..n_sub)
        .map(|sb| {
            let lo = sb * SUB_BLOCK;
            let hi = (lo + SUB_BLOCK).min(chunk.len());
            let s = &chunk[lo..hi];
            if s.is_empty() {
                0.0
            } else {
                s.iter().map(|&w| w as f64).sum::<f64>() / s.len() as f64
            }
        })
        .collect();
    let base_abs = means
        .iter()
        .copied()
        .fold(0.0f64, |b, m| if m.abs() > b { m.abs() } else { b });
    if base_abs < 1e-12 {
        return (0, vec![0u8; n_sub]);
    }
    let min_base_q = (base_abs * (1u32 << QUANTILE_SHIFT) as f64).round() as i32;
    let q_to_real = 1.0f64 / (1u32 << QUANTILE_SHIFT) as f64;
    let mut codes = Vec::with_capacity(n_sub);
    for (sb, &mult) in mults.iter().enumerate() {
        let lo = sb * SUB_BLOCK;
        let hi = (lo + SUB_BLOCK).min(chunk.len());
        let sub = &chunk[lo..hi];
        let es = eff_scale_q(scale_q, mult);
        let es_real = (es as f64) / (1u64 << SCALE_SHIFT) as f64;
        let positive_side = means[sb] >= 0.0;
        let code_range = if positive_side { 32u8..=63 } else { 0u8..=31 };
        let mut best_c = if positive_side { 32 } else { 0 };
        let mut best_sse = f64::INFINITY;
        for c in code_range {
            let off_real = (crate::decode::eff_min_q(min_base_q, c) as f64) * q_to_real;
            let sse = greedy_replay_sse_vec_off(sub, es_real, off_real, lut, cfg);
            if sse < best_sse {
                best_sse = sse;
                best_c = c;
            }
        }
        codes.push(best_c);
    }
    (min_base_q, codes)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn step_state_dist(
    chunk: &[f32],
    base_i: usize,
    present: usize,
    sub_levels: &[Vec<f64>],
    d: usize,
    num_states: usize,
    dist: &mut [f64],
) {
    for s in 0..num_states {
        let mut e = 0.0f64;
        for j in 0..present {
            let i = base_i + j;
            let target = chunk[i] as f64;
            let lvl = sub_levels[i / SUB_BLOCK][s * d + j];
            let diff = target - lvl;
            e += diff * diff;
        }
        dist[s] = e;
    }
}

fn build_sub_levels_vec(
    scale_q: i32,
    mults: &[u8],
    mins_eff: &[i32],
    lut: &[i32],
    cfg: &TrellisConfig,
) -> Vec<Vec<f64>> {
    let d = cfg.vec_dim();
    let num_states = cfg.num_states();
    let q_to_real = 1.0f64 / (1u32 << QUANTILE_SHIFT) as f64;
    mults
        .iter()
        .enumerate()
        .map(|(sb, &m)| {
            let es = eff_scale_q(scale_q, m);
            let off = *mins_eff.get(sb).unwrap_or(&0);
            let mut lv = vec![0.0f64; num_states * d];
            for s in 0..num_states {
                for j in 0..d {
                    lv[s * d + j] = (reconstruct_q(es, lut[s * d + j]) + off) as f64 * q_to_real;
                }
            }
            lv
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn viterbi_path_buf_vec(
    chunk: &[f32],
    scale_q: i32,
    mults: &[u8],
    mins_eff: &[i32],
    lut: &[i32],
    cfg: &TrellisConfig,
    tail_biting: bool,
    back_buf: &mut Vec<u32>,
) -> (Vec<u32>, usize) {
    let d = cfg.vec_dim();
    let n = chunk.len();
    let num_states = cfg.num_states();
    let n_steps = n.div_ceil(d);
    if n == 0 {
        return (Vec::new(), 0);
    }

    let sub_levels = build_sub_levels_vec(scale_q, mults, mins_eff, lut, cfg);

    let nk = n_steps * cfg.k_bits as usize;
    let can_tail_bite = tail_biting && nk >= cfg.l_bits as usize;

    let needed = n_steps * num_states;
    if back_buf.len() < needed {
        back_buf.resize(needed, u32::MAX);
    }

    if !can_tail_bite {
        return vec_backtrack(chunk, &sub_levels, cfg, None, None, back_buf);
    }
    
    let (final_s, _) = vec_forward(chunk, &sub_levels, cfg, None);
    let path = vec_backtrack(chunk, &sub_levels, cfg, Some(final_s), Some(final_s), back_buf);
    (path.0, final_s)
}

fn vec_forward(
    chunk: &[f32],
    sub_levels: &[Vec<f64>],
    cfg: &TrellisConfig,
    pin_start: Option<usize>,
) -> (usize, f64) {
    let d = cfg.vec_dim();
    let n = chunk.len();
    let n_steps = n.div_ceil(d);
    let num_states = cfg.num_states();
    let inf = f64::INFINITY;
    let mut cost = match pin_start {
        Some(s0) => {
            let mut c = vec![inf; num_states];
            c[s0] = 0.0;
            c
        }
        None => vec![0.0f64; num_states],
    };
    let k = cfg.k_bits as usize;
    let mut next_cost = vec![inf; num_states];
    let mut dist = vec![0.0f64; num_states];
    for t in 0..n_steps {
        let base_i = t * d;
        let present = (n - base_i).min(d);
        step_state_dist(chunk, base_i, present, sub_levels, d, num_states, &mut dist);
        relax_step_f64(&cost, &dist, &mut next_cost, None, k);
        std::mem::swap(&mut cost, &mut next_cost);
    }
    pick_terminal(&cost)
}

fn vec_backtrack(
    chunk: &[f32],
    sub_levels: &[Vec<f64>],
    cfg: &TrellisConfig,
    pin_start: Option<usize>,
    final_s: Option<usize>,
    back_buf: &mut [u32],
) -> (Vec<u32>, usize) {
    let d = cfg.vec_dim();
    let n = chunk.len();
    let n_steps = n.div_ceil(d);
    let num_states = cfg.num_states();
    let inf = f64::INFINITY;
    let mut cost = match pin_start {
        Some(s0) => {
            let mut c = vec![inf; num_states];
            c[s0] = 0.0;
            c
        }
        None => vec![0.0f64; num_states],
    };
    let k = cfg.k_bits as usize;
    let mut next_cost = vec![inf; num_states];
    let mut dist = vec![0.0f64; num_states];
    for t in 0..n_steps {
        let base_i = t * d;
        let present = (n - base_i).min(d);
        step_state_dist(chunk, base_i, present, sub_levels, d, num_states, &mut dist);
        let row = &mut back_buf[t * num_states..(t + 1) * num_states];
        relax_step_f64(&cost, &dist, &mut next_cost, Some(row), k);
        std::mem::swap(&mut cost, &mut next_cost);
    }
    let start_at = final_s.unwrap_or_else(|| pick_terminal(&cost).0);
    let input_mask = cfg.num_inputs() - 1;
    let mut path = vec![0u32; n_steps];
    let mut s = start_at;
    for t in (0..n_steps).rev() {
        path[t] = (s & input_mask) as u32;
        s = back_buf[t * num_states + s] as usize;
    }
    (path, s)
}

fn vec_forward_reference(
    chunk: &[f32],
    sub_levels: &[Vec<f64>],
    cfg: &TrellisConfig,
    pin_start: Option<usize>,
) -> (usize, f64) {
    let d = cfg.vec_dim();
    let n = chunk.len();
    let n_steps = n.div_ceil(d);
    let num_states = cfg.num_states();
    let inf = f64::INFINITY;
    let mut cost = match pin_start {
        Some(s0) => {
            let mut c = vec![inf; num_states];
            c[s0] = 0.0;
            c
        }
        None => vec![0.0f64; num_states],
    };
    let num_inputs = cfg.num_inputs();
    let k = cfg.k_bits as usize;
    let mut next_cost = vec![inf; num_states];
    let mut dist = vec![0.0f64; num_states];
    for t in 0..n_steps {
        let base_i = t * d;
        let present = (n - base_i).min(d);
        step_state_dist(chunk, base_i, present, sub_levels, d, num_states, &mut dist);
        next_cost.fill(inf);
        for (s, &c) in cost.iter().enumerate() {
            if c == inf {
                continue;
            }
            let ns_base = (s << k) & (num_states - 1);
            for off in 0..num_inputs {
                let ns = ns_base | off;
                let nc = c + dist[ns];
                if nc < next_cost[ns] {
                    next_cost[ns] = nc;
                }
            }
        }
        std::mem::swap(&mut cost, &mut next_cost);
    }
    pick_terminal(&cost)
}

fn vec_backtrack_reference(
    chunk: &[f32],
    sub_levels: &[Vec<f64>],
    cfg: &TrellisConfig,
    pin_start: Option<usize>,
    final_s: Option<usize>,
    back_buf: &mut [u32],
) -> (Vec<u32>, usize) {
    let d = cfg.vec_dim();
    let n = chunk.len();
    let n_steps = n.div_ceil(d);
    let num_states = cfg.num_states();
    let inf = f64::INFINITY;
    let mut cost = match pin_start {
        Some(s0) => {
            let mut c = vec![inf; num_states];
            c[s0] = 0.0;
            c
        }
        None => vec![0.0f64; num_states],
    };
    let num_inputs = cfg.num_inputs();
    let k = cfg.k_bits as usize;
    let mut next_cost = vec![inf; num_states];
    let mut dist = vec![0.0f64; num_states];
    for t in 0..n_steps {
        let base_i = t * d;
        let present = (n - base_i).min(d);
        step_state_dist(chunk, base_i, present, sub_levels, d, num_states, &mut dist);
        let row = &mut back_buf[t * num_states..(t + 1) * num_states];
        row.fill(u32::MAX);
        next_cost.fill(inf);
        for (s, &c) in cost.iter().enumerate() {
            if c == inf {
                continue;
            }
            let s_u32 = s as u32;
            let ns_base = (s << k) & (num_states - 1);
            for off in 0..num_inputs {
                let ns = ns_base | off;
                let nc = c + dist[ns];
                if nc < next_cost[ns] {
                    next_cost[ns] = nc;
                    row[ns] = s_u32;
                }
            }
        }
        std::mem::swap(&mut cost, &mut next_cost);
    }
    let start_at = final_s.unwrap_or_else(|| pick_terminal(&cost).0);
    let input_mask = num_inputs - 1;
    let mut path = vec![0u32; n_steps];
    let mut s = start_at;
    for t in (0..n_steps).rev() {
        path[t] = (s & input_mask) as u32;
        let pred = back_buf[t * num_states + s];
        debug_assert_ne!(pred, u32::MAX, "vec backtrack hit an unreached state");
        s = pred as usize;
    }
    (path, s)
}

#[allow(clippy::too_many_arguments)]
fn viterbi_path_buf_vec_reference(
    chunk: &[f32],
    scale_q: i32,
    mults: &[u8],
    mins_eff: &[i32],
    lut: &[i32],
    cfg: &TrellisConfig,
    tail_biting: bool,
    back_buf: &mut Vec<u32>,
) -> (Vec<u32>, usize) {
    let d = cfg.vec_dim();
    let n = chunk.len();
    let num_states = cfg.num_states();
    let n_steps = n.div_ceil(d);
    if n == 0 {
        return (Vec::new(), 0);
    }
    let sub_levels = build_sub_levels_vec(scale_q, mults, mins_eff, lut, cfg);
    let nk = n_steps * cfg.k_bits as usize;
    let can_tail_bite = tail_biting && nk >= cfg.l_bits as usize;
    let needed = n_steps * num_states;
    if back_buf.len() < needed {
        back_buf.resize(needed, u32::MAX);
    }
    if !can_tail_bite {
        return vec_backtrack_reference(chunk, &sub_levels, cfg, None, None, back_buf);
    }
    let (final_s, _) = vec_forward_reference(chunk, &sub_levels, cfg, None);
    let path =
        vec_backtrack_reference(chunk, &sub_levels, cfg, Some(final_s), Some(final_s), back_buf);
    (path.0, final_s)
}

#[doc(hidden)]
pub fn encode_tensor_with_lut_vec_reference(
    weights: &[f32],
    cfg: &TrellisConfig,
    opts: &EncodeOpts,
    lut: &[i32],
) -> EncodedTensor {
    let d = cfg.vec_dim();
    let num_states = cfg.num_states();
    debug_assert_eq!(lut.len(), num_states * d, "vector LUT must be [2^L * d]");

    let mut bits = Vec::new();
    let mut bit_cursor = 0usize;
    let mut blocks = Vec::new();
    let max_steps = cfg.num_steps(cfg.block_len);
    let mut back_buf: Vec<u32> = vec![u32::MAX; max_steps * num_states];

    for chunk in weights.chunks(cfg.block_len) {
        let scale_q = choose_scale_q_vec(chunk, lut, cfg);
        let mults = if opts.adaptive {
            choose_sub_scales_vec(chunk, scale_q, lut, cfg)
        } else {
            vec![SUB_SCALE_UNITY; n_sub_blocks(chunk.len())]
        };
        let (min_base_q, min_codes) = if opts.affine_min {
            choose_affine_min_vec(chunk, scale_q, &mults, lut, cfg)
        } else {
            (0, Vec::new())
        };
        let mins_eff: Vec<i32> = min_codes
            .iter()
            .map(|&c| crate::decode::eff_min_q(min_base_q, c))
            .collect();
        let (path, init_state) = viterbi_path_buf_vec_reference(
            chunk, scale_q, &mults, &mins_eff, lut, cfg, opts.tail_biting, &mut back_buf,
        );
        for &sym in &path {
            push_bits(&mut bits, &mut bit_cursor, sym as usize, cfg.k_bits);
        }
        blocks.push(BlockMeta {
            scale_q,
            sub_scales: pack_sub_scales(&mults),
            min_base_q,
            mins: if opts.affine_min { pack_sub_scales(&min_codes) } else { Vec::new() },
            init_state: init_state as u32,
            n: chunk.len() as u32,
        });
    }

    EncodedTensor {
        bits,
        blocks,
        total: weights.len(),
        has_rht_seed: false,
        tail_biting: opts.tail_biting,
        has_affine_min: opts.affine_min,
    }
}
