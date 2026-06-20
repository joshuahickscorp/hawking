
use crate::block_walk::{
    block_init_state, block_plans, exceeds_max_sub, BlockPlan, SideInfo, WordReader,
};
use crate::loader::StrandModel;
use rayon::prelude::*;
use strand_quant::codebook::codebook_lut;
use strand_quant::decode::reconstruct_q;
use strand_quant::encode::{BlockMeta, EncodedTensor, SUB_BLOCK};
use strand_quant::TrellisConfig;

pub fn decode_q12_par(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<i32> {
    decode_q12_par_with_lut(enc, cfg, codebook_lut(cfg.l_bits))
}

pub fn decode_q12_par_counted(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    counts: Option<&mut [u32]>,
) -> Vec<i32> {
    decode_q12_par_with_lut_counted(enc, cfg, codebook_lut(cfg.l_bits), counts)
}

pub fn decode_q12_par_with_lut(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32]) -> Vec<i32> {
    if cfg.vec_dim() > 1 || exceeds_max_sub(enc) {
        return crate::gemv::decode_q12_fast_with_lut(enc, cfg, lut);
    }

    let num_states = cfg.num_states();
    debug_assert_eq!(lut.len(), num_states, "scalar LUT must have num_states entries");
    let fold = SUB_BLOCK >= num_states;
    let has_affine = enc.has_affine_min;
    let tail_biting = enc.tail_biting;
    let k = cfg.k_bits as usize;

    let plans = block_plans(enc, k);
    let mut out = vec![0i32; enc.total];

    let mut slices: Vec<&mut [i32]> = Vec::with_capacity(enc.blocks.len());
    let mut rest: &mut [i32] = &mut out;
    for blk in &enc.blocks {
        let (head, tail) = rest.split_at_mut(blk.n as usize);
        slices.push(head);
        rest = tail;
    }

    enc.blocks
        .par_iter()
        .zip(plans.par_iter())
        .zip(slices.par_iter_mut())
        .for_each(|((blk, plan), dst)| {
            let mut folded: Vec<i32> = Vec::new();
            decode_block_inline(
                blk, plan, &enc.bits, cfg, lut, fold, has_affine, tail_biting, &mut folded, dst,
            );
        });

    out
}

pub fn decode_q12_par_with_lut_counted(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
    counts: Option<&mut [u32]>,
) -> Vec<i32> {
    
    let Some(counts) = counts else {
        return decode_q12_par_with_lut(enc, cfg, lut);
    };

    if cfg.vec_dim() > 1 || exceeds_max_sub(enc) {
        
        let out = crate::gemv::decode_q12_fast_with_lut(enc, cfg, lut);
        for b in 0..enc.blocks.len().min(counts.len()) {
            counts[b] = counts[b].saturating_add(1);
        }
        return out;
    }

    let num_states = cfg.num_states();
    debug_assert_eq!(lut.len(), num_states, "scalar LUT must have num_states entries");
    let fold = SUB_BLOCK >= num_states;
    let has_affine = enc.has_affine_min;
    let tail_biting = enc.tail_biting;
    let k = cfg.k_bits as usize;

    let plans = block_plans(enc, k);
    let mut out = vec![0i32; enc.total];

    let mut slices: Vec<&mut [i32]> = Vec::with_capacity(enc.blocks.len());
    let mut rest: &mut [i32] = &mut out;
    for blk in &enc.blocks {
        let (head, tail) = rest.split_at_mut(blk.n as usize);
        slices.push(head);
        rest = tail;
    }

    let mut folded: Vec<i32> = Vec::new();
    for (b, ((blk, plan), dst)) in enc.blocks
        .iter()
        .zip(plans.iter())
        .zip(slices.iter_mut())
        .enumerate()
    {
        decode_block_inline(
            blk, plan, &enc.bits, cfg, lut, fold, has_affine, tail_biting, &mut folded, dst,
        );
        
        if b < counts.len() {
            counts[b] = counts[b].saturating_add(1);
        }
    }

    out
}

#[allow(unsafe_code)]
#[inline]
fn decode_block_inline(
    blk: &BlockMeta,
    plan: &BlockPlan,
    bits: &[u8],
    cfg: &TrellisConfig,
    lut: &[i32],
    fold: bool,
    has_affine: bool,
    tail_biting: bool,
    folded: &mut Vec<i32>,
    dst: &mut [i32],
) {
    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let num_states = cfg.num_states();
    let n = blk.n as usize;
    let start_bit = plan.start_bit;
    debug_assert_eq!(plan.out_off * (k as usize), start_bit);

    let side = SideInfo::hoist(blk, has_affine);
    let (eff, off) = (side.eff(), side.off());
    let n_sub = side.n_sub;
    let mut state = block_init_state(blk, bits, start_bit, cfg, tail_biting);

    let mut reader = WordReader::new(bits, start_bit);

    if fold {
        folded.clear();
        folded.resize(n_sub * num_states, 0);
        for (sb, &es) in eff.iter().enumerate() {
            let base = sb * num_states;
            for s in 0..num_states {
                folded[base + s] = reconstruct_q(es, lut[s]);
            }
        }
        let mut i = 0usize;
        for (sb, &o) in off.iter().enumerate() {
            let base = sb * num_states;
            let end = (i + SUB_BLOCK).min(n);
            while i < end {
                let sym = reader.pop(k) & input_mask;
                state = ((state << k) | sym) & mask;
                let v = unsafe { *folded.get_unchecked(base + state) };
                unsafe { *dst.get_unchecked_mut(i) = v + o };
                i += 1;
            }
        }
    } else {
        let lut_ptr = lut.as_ptr();
        let mut i = 0usize;
        for (&es, &o) in eff.iter().zip(off.iter()) {
            let end = (i + SUB_BLOCK).min(n);
            while i < end {
                let sym = reader.pop(k) & input_mask;
                state = ((state << k) | sym) & mask;
                let q = unsafe { *lut_ptr.add(state) };
                unsafe { *dst.get_unchecked_mut(i) = reconstruct_q(es, q) + o };
                i += 1;
            }
        }
    }
}

pub fn decode_tensor_q12_par(model: &StrandModel, name: &str) -> Option<Vec<i32>> {
    let hdr = model.tensor_header(name)?;
    let cfg = model.config_for(hdr);
    let enc = model.encoded_tensor(name)?;
    Some(decode_q12_par(&enc, &cfg))
}

pub fn matvec_named_par(model: &StrandModel, name: &str, x: &[f32]) -> Option<Vec<f32>> {
    let hdr = model.tensor_header(name)?;
    if hdr.shape.len() < 2 {
        return None;
    }
    let out_features = hdr.shape[0] as usize;
    let in_features = hdr.shape[1] as usize;
    if x.len() != in_features {
        return None;
    }
    let cfg = model.config_for(hdr);
    let enc = model.encoded_tensor(name)?;
    let w = decode_q12_par(&enc, &cfg);
    if w.len() != out_features * in_features {
        return None;
    }
    let inv = 1.0f32 / 4096.0;
    
    let y: Vec<f32> = (0..out_features)
        .into_par_iter()
        .map(|o| {
            let row = &w[o * in_features..(o + 1) * in_features];
            let mut acc = 0.0f32;
            for i in 0..in_features {
                acc += (row[i] as f32) * inv * x[i];
            }
            acc
        })
        .collect();
    Some(y)
}

#[cfg(target_arch = "aarch64")]
pub const SIMD_LANES: usize = 4;

pub fn decode_q12_simd(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<i32> {
    decode_q12_simd_with_lut(enc, cfg, codebook_lut(cfg.l_bits))
}

#[cfg(target_arch = "aarch64")]
pub fn decode_q12_simd_with_lut(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32]) -> Vec<i32> {
    if cfg.vec_dim() > 1 || exceeds_max_sub(enc) {
        return decode_q12_par_with_lut(enc, cfg, lut);
    }
    let num_states = cfg.num_states();
    let fold = SUB_BLOCK >= num_states;
    if fold {
        return decode_q12_par_with_lut(enc, cfg, lut);
    }

    let k = cfg.k_bits;
    let mask = cfg.state_mask() as u32;
    let input_mask = (cfg.num_inputs() - 1) as u32;
    let has_affine = enc.has_affine_min;
    let tail_biting = enc.tail_biting;
    let plans = block_plans(enc, k as usize);

    let mut out = vec![0i32; enc.total];
    let mut slices: Vec<&mut [i32]> = Vec::with_capacity(enc.blocks.len());
    {
        let mut rest: &mut [i32] = &mut out;
        for blk in &enc.blocks {
            let (head, tail) = rest.split_at_mut(blk.n as usize);
            slices.push(head);
            rest = tail;
        }
    }

    let n_blocks = enc.blocks.len();
    let blocks: &[BlockMeta] = &enc.blocks;
    let bits: &[u8] = &enc.bits;

    let block_chunks: Vec<&[BlockMeta]> = blocks.chunks(SIMD_LANES).collect();
    let plan_chunks: Vec<&[BlockPlan]> = plans.chunks(SIMD_LANES).collect();
    let mut slice_groups: Vec<Vec<&mut [i32]>> = Vec::with_capacity(block_chunks.len());
    {
        let mut it = slices.into_iter();
        for ch in &block_chunks {
            let mut g: Vec<&mut [i32]> = Vec::with_capacity(ch.len());
            for _ in 0..ch.len() {
                g.push(it.next().expect("slice per block"));
            }
            slice_groups.push(g);
        }
    }
    let _ = n_blocks;

    block_chunks
        .into_par_iter()
        .zip(plan_chunks.into_par_iter())
        .zip(slice_groups.into_par_iter())
        .for_each(|((bch, pch), mut sg)| {
            if bch.len() == SIMD_LANES {
                
                unsafe {
                    decode_4_blocks_neon(
                        bch, pch, bits, cfg, lut, k, mask, input_mask, has_affine, tail_biting,
                        &mut sg,
                    );
                }
            } else {
                let mut folded: Vec<i32> = Vec::new();
                for ((blk, plan), dst) in bch.iter().zip(pch.iter()).zip(sg.iter_mut()) {
                    decode_block_inline(
                        blk, plan, bits, cfg, lut, false, has_affine, tail_biting, &mut folded,
                        dst,
                    );
                }
            }
        });

    out
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
unsafe fn decode_4_blocks_neon(
    blocks: &[BlockMeta],
    plans: &[BlockPlan],
    bits: &[u8],
    cfg: &TrellisConfig,
    lut: &[i32],
    k: u32,
    mask: u32,
    input_mask: u32,
    has_affine: bool,
    tail_biting: bool,
    dst: &mut [&mut [i32]],
) {
    use core::arch::aarch64::*;

    debug_assert_eq!(blocks.len(), SIMD_LANES);

    let sides: [SideInfo; SIMD_LANES] =
        core::array::from_fn(|lane| SideInfo::hoist(&blocks[lane], has_affine));
    let mut n_lane = [0usize; SIMD_LANES];
    for lane in 0..SIMD_LANES {
        n_lane[lane] = blocks[lane].n as usize;
    }

    let mut state_arr = [0u32; SIMD_LANES];
    let mut readers: Vec<WordReader> = Vec::with_capacity(SIMD_LANES);
    for lane in 0..SIMD_LANES {
        let start_bit = plans[lane].start_bit;
        state_arr[lane] =
            block_init_state(&blocks[lane], bits, start_bit, cfg, tail_biting) as u32;
        readers.push(WordReader::new(bits, start_bit));
    }

    let max_n = *n_lane.iter().max().unwrap();
    let mut state_v = vld1q_u32(state_arr.as_ptr());
    let mask_v = vdupq_n_u32(mask);

    let lut_ptr = lut.as_ptr();
    for i in 0..max_n {
        let mut sym_arr = [0u32; SIMD_LANES];
        for lane in 0..SIMD_LANES {
            if i < n_lane[lane] {
                sym_arr[lane] = (readers[lane].pop(k) as u32) & input_mask;
            }
        }
        let sym_v = vld1q_u32(sym_arr.as_ptr());
        
        let shifted = vshlq_n_u32_dynamic(state_v, k);
        state_v = vandq_u32(vorrq_u32(shifted, sym_v), mask_v);
        vst1q_u32(state_arr.as_mut_ptr(), state_v);

        for lane in 0..SIMD_LANES {
            if i < n_lane[lane] {
                let st = state_arr[lane] as usize;
                
                let q = *lut_ptr.add(st);
                let sb = i / SUB_BLOCK;
                let es = sides[lane].eff[sb];
                let o = sides[lane].off[sb];
                *dst[lane].get_unchecked_mut(i) = reconstruct_q(es, q) + o;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_code)]
unsafe fn vshlq_n_u32_dynamic(v: core::arch::aarch64::uint32x4_t, k: u32) -> core::arch::aarch64::uint32x4_t {
    use core::arch::aarch64::*;
    let cnt = vdupq_n_s32(k as i32);
    vshlq_u32(v, cnt)
}

#[cfg(not(target_arch = "aarch64"))]
pub fn decode_q12_simd_with_lut(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32]) -> Vec<i32> {
    decode_q12_par_with_lut(enc, cfg, lut)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemv::decode_q12_fast;
    use strand_quant::decode::decode_lean as ref_decode_lean;
    use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};

    #[test]
    fn par_and_simd_decode_are_bit_identical() {
        let configs = [
            (TrellisConfig::for_bpw(3.0), false),
            (TrellisConfig::for_bpw(2.0), false),
            (TrellisConfig::for_bpw(4.0), false),
            (TrellisConfig::for_bpw_l(2.0, 5), true),
            (TrellisConfig::for_bpw_l(4.0, 4), true),
            (TrellisConfig::for_bpw_l(3.0, 5), true),
        ];
        for (cfg, fold_expected) in configs {
            assert_eq!(
                SUB_BLOCK >= cfg.num_states(),
                fold_expected,
                "fold gate mismatch for L={}",
                cfg.l_bits
            );
            for seed in 0..96u64 {
                let n = 1 + (seed as usize * 53) % 4096;
                let w: Vec<f32> = (0..n)
                    .map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5)
                    .collect();
                let variants = [
                    encode_tensor(&w, &cfg),
                    encode_tensor_with(
                        &w,
                        &cfg,
                        &EncodeOpts { tail_biting: true, ..Default::default() },
                    ),
                    encode_tensor_with(
                        &w,
                        &cfg,
                        &EncodeOpts { affine_min: true, ..Default::default() },
                    ),
                    encode_tensor_with(
                        &w,
                        &cfg,
                        &EncodeOpts {
                            tail_biting: true,
                            affine_min: true,
                            ..Default::default()
                        },
                    ),
                ];
                for enc in &variants {
                    let refr = ref_decode_lean(enc, &cfg);
                    let fast = decode_q12_fast(enc, &cfg);
                    assert_eq!(fast, refr, "precondition: fast != lean");
                    let par = decode_q12_par(enc, &cfg);
                    assert_eq!(
                        par, refr,
                        "PAR decode diverged: L={} k={} n={} seed={} tail={} affine={}",
                        cfg.l_bits, cfg.k_bits, n, seed, enc.tail_biting, enc.has_affine_min
                    );
                    let simd = decode_q12_simd(enc, &cfg);
                    assert_eq!(
                        simd, refr,
                        "SIMD decode diverged: L={} k={} n={} seed={} tail={} affine={}",
                        cfg.l_bits, cfg.k_bits, n, seed, enc.tail_biting, enc.has_affine_min
                    );
                }
            }
        }
    }

    #[test]
    fn par_simd_bit_identical_large() {
        let cfg = TrellisConfig::for_bpw(3.0);
        let n = 256 * 257 + 17;
        let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0011).sin() * 0.6).collect();
        let enc = encode_tensor(&w, &cfg);
        let refr = ref_decode_lean(&enc, &cfg);
        assert_eq!(decode_q12_fast(&enc, &cfg), refr);
        assert_eq!(decode_q12_par(&enc, &cfg), refr, "par large");
        assert_eq!(decode_q12_simd(&enc, &cfg), refr, "simd large");
    }

    #[test]
    fn par_and_simd_vector_trellis_bit_identical() {
        use strand_quant::decode::decode_lean_with_lut;
        use strand_quant::encode::encode_tensor_with_lut;
        use strand_quant::learned_codebook::train_state_vector_lut;

        let cfg = TrellisConfig::for_bpw(4.0).with_vec_dim(2);
        assert_eq!(cfg.vec_dim(), 2);
        assert_eq!(cfg.k_bits, 4);
        for seed in 0..48u64 {
            let n = 2 + (seed as usize * 213) % 6000;
            let w: Vec<f32> =
                (0..n).map(|i| ((i as f32 + seed as f32) * 0.0091).cos() * 0.6).collect();
            let lut = train_state_vector_lut(&w, cfg.l_bits, 2, 0xABCD ^ seed, 30);
            assert_eq!(lut.len(), cfg.num_states() * 2, "vector LUT must be [2^L * d]");
            for tail in [false, true] {
                let opts = EncodeOpts { tail_biting: tail, ..Default::default() };
                let enc = encode_tensor_with_lut(&w, &cfg, &opts, &lut);
                assert_eq!(enc.total, n);
                let refr = decode_lean_with_lut(&enc, &cfg, &lut);
                assert_eq!(
                    decode_q12_par_with_lut(&enc, &cfg, &lut),
                    refr,
                    "PAR vec_dim=2 n={n} seed={seed} tail={tail}"
                );
                assert_eq!(
                    decode_q12_simd_with_lut(&enc, &cfg, &lut),
                    refr,
                    "SIMD vec_dim=2 n={n} seed={seed} tail={tail}"
                );
            }
        }
    }

    #[test]
    fn par_decode_via_v2_matches_fast() {
        use std::io::Write;
        use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2};

        fn write_tiny_v2(name: &str, rows: u64, cols: u64, cfg: &TrellisConfig, enc: &EncodedTensor) -> std::path::PathBuf {
            let shape = [rows, cols];
            let pt = PackedTensorV2 {
                base: PackedTensor {
                    name,
                    shape: &shape,
                    rht_seed: 0,
                    l_bits: cfg.l_bits as u8,
                    k_bits: cfg.k_bits as u8,
                    vec_dim: cfg.vec_dim() as u8,
                    enc,
                },
                block_len: cfg.block_len as u32,
            };
            let buf = write_strand_v2(&[pt], [0u8; 32], true).expect("write_strand_v2");
            let mut path = std::env::temp_dir();
            let uniq = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            path.push(format!("strand_gemvpar_e2e_{name}_{}_{uniq}.strand", std::process::id()));
            let mut f = std::fs::File::create(&path).expect("create temp .strand");
            f.write_all(&buf).expect("write");
            f.sync_all().ok();
            path
        }

        for &(rows, cols) in &[(4u64, 256u64), (8u64, 512u64), (7u64, 768u64)] {
            let n = (rows * cols) as usize;
            let weights: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 0.7).collect();
            let cfg = TrellisConfig::for_bpw(3.0);
            let enc = encode_tensor(&weights, &cfg);
            let path = write_tiny_v2("w", rows, cols, &cfg, &enc);
            let model = StrandModel::open(&path).expect("open");
            let q_par = decode_tensor_q12_par(&model, "w").expect("par");
            let q_fast = crate::gemv::decode_tensor_q12_fast(&model, "w").expect("fast");
            assert_eq!(q_par, q_fast, "par vs fast mmap (scalar rows={rows} cols={cols})");
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn matvec_named_par_matches_fast_via_v2() {
        use std::io::Write;
        use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2};

        let (rows, cols) = (9usize, 512usize);
        let weights: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.123).sin()).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&weights, &cfg);
        let shape = [rows as u64, cols as u64];
        let pt = PackedTensorV2 {
            base: PackedTensor {
                name: "w",
                shape: &shape,
                rht_seed: 0,
                l_bits: cfg.l_bits as u8,
                k_bits: cfg.k_bits as u8,
                vec_dim: cfg.vec_dim() as u8,
                enc: &enc,
            },
            block_len: cfg.block_len as u32,
        };
        let buf = write_strand_v2(&[pt], [0u8; 32], true).expect("write_strand_v2");
        let mut path = std::env::temp_dir();
        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        path.push(format!("strand_gemvpar_mv_{}_{uniq}.strand", std::process::id()));
        std::fs::File::create(&path).unwrap().write_all(&buf).unwrap();

        let model = StrandModel::open(&path).expect("open");
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.07).cos()).collect();
        let y_par = matvec_named_par(&model, "w", &x).expect("matvec_named_par");
        let y_fast = crate::gemv::matvec_named_fast(&model, "w", &x).expect("matvec_named_fast");
        assert_eq!(y_par.len(), y_fast.len());
        for o in 0..rows {
            assert_eq!(y_par[o].to_bits(), y_fast[o].to_bits(), "row {o} y mismatch");
        }
        assert!(matvec_named_par(&model, "w", &[0.0f32; 7]).is_none());
        assert!(matvec_named_par(&model, "missing", &x).is_none());
        let _ = std::fs::remove_file(&path);
    }
}
