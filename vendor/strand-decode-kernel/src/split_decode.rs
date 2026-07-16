use crate::block_walk::{block_init_state, block_plans, exceeds_max_sub, SideInfo, WordReader};
use rayon::prelude::*;
use strand_quant::codebook::codebook_lut;
use strand_quant::decode::{reconstruct_q, SCALE_SHIFT};
use strand_quant::encode::{EncodedTensor, SUB_BLOCK};
use strand_quant::TrellisConfig;

const _: () = assert!(SCALE_SHIFT == 16, "NEON pass 2 hardcodes the Q16 scale shift");

pub fn decode_q12_split(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<i32> {
    decode_q12_split_with_lut(enc, cfg, codebook_lut(cfg.l_bits))
}

pub fn decode_q12_split_with_lut(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32]) -> Vec<i32> {
    if cfg.vec_dim() > 1 || exceeds_max_sub(enc) {
        return crate::gemv::decode_q12_fast_with_lut(enc, cfg, lut);
    }
    debug_assert_eq!(lut.len(), cfg.num_states(), "scalar LUT must have num_states entries");

    let k = cfg.k_bits as usize;
    let has_affine = enc.has_affine_min;
    let tail_biting = enc.tail_biting;
    let plans = block_plans(enc, k);

    let mut out = vec![0i32; enc.total];
    let mut rest: &mut [i32] = &mut out;
    for (blk, plan) in enc.blocks.iter().zip(plans.iter()) {
        let (dst, tail) = rest.split_at_mut(plan.n);
        rest = tail;
        split_pass1(blk, &enc.bits, plan.start_bit, cfg, lut, tail_biting, dst);
        let side = SideInfo::hoist(blk, has_affine);
        split_pass2(&side, dst);
    }
    out
}

pub fn decode_q12_split_par(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<i32> {
    decode_q12_split_par_with_lut(enc, cfg, codebook_lut(cfg.l_bits))
}

pub fn decode_q12_split_par_with_lut(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32]) -> Vec<i32> {
    if cfg.vec_dim() > 1 || exceeds_max_sub(enc) {
        return crate::gemv::decode_q12_fast_with_lut(enc, cfg, lut);
    }
    debug_assert_eq!(lut.len(), cfg.num_states(), "scalar LUT must have num_states entries");

    let k = cfg.k_bits as usize;
    let has_affine = enc.has_affine_min;
    let tail_biting = enc.tail_biting;
    let plans = block_plans(enc, k);

    let mut out = vec![0i32; enc.total];

    let mut slices: Vec<&mut [i32]> = Vec::with_capacity(enc.blocks.len());
    let mut rest: &mut [i32] = &mut out;
    for blk in &enc.blocks {
        let (head, tail) = rest.split_at_mut(blk.n as usize);
        slices.push(head);
        rest = tail;
    }

    enc.blocks.par_iter().zip(plans.par_iter()).zip(slices.par_iter_mut()).for_each(|((blk, plan), dst)| {
        split_pass1(blk, &enc.bits, plan.start_bit, cfg, lut, tail_biting, dst);
        let side = SideInfo::hoist(blk, has_affine);
        split_pass2(&side, dst);
    });

    out
}

#[allow(unsafe_code)]
#[inline]
fn split_pass1(blk: &strand_quant::encode::BlockMeta, bits: &[u8], start_bit: usize, cfg: &TrellisConfig, lut: &[i32], tail_biting: bool, dst: &mut [i32]) {
    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let mut state = block_init_state(blk, bits, start_bit, cfg, tail_biting);
    let mut reader = WordReader::new(bits, start_bit);
    let lut_ptr = lut.as_ptr();
    for slot in dst.iter_mut() {
        let sym = reader.pop(k) & input_mask;
        state = ((state << k) | sym) & mask;

        *slot = unsafe { *lut_ptr.add(state) };
    }
}

#[inline]
fn split_pass2(side: &SideInfo, dst: &mut [i32]) {
    let n = dst.len();
    let mut i = 0usize;
    for sb in 0..side.n_sub {
        let end = (i + SUB_BLOCK).min(n);
        scale_apply(&mut dst[i..end], side.eff[sb], side.off[sb]);
        i = end;
    }
    debug_assert_eq!(i, n, "sub-block segments must cover the block exactly");
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn scale_apply(seg: &mut [i32], es: i32, off: i32) {
    unsafe { scale_apply_neon(seg, es, off) }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[target_feature(enable = "neon")]
unsafe fn scale_apply_neon(seg: &mut [i32], es: i32, off: i32) {
    use core::arch::aarch64::*;
    let es2 = vdup_n_s32(es);
    let offv = vdupq_n_s32(off);
    let len = seg.len();
    let n4 = len & !3;
    let p = seg.as_mut_ptr();
    let mut i = 0usize;
    while i < n4 {
        let q = vld1q_s32(p.add(i));
        let plo = vmull_s32(vget_low_s32(q), es2);
        let phi = vmull_s32(vget_high_s32(q), es2);
        let r = vcombine_s32(vmovn_s64(vshrq_n_s64::<16>(plo)), vmovn_s64(vshrq_n_s64::<16>(phi)));
        vst1q_s32(p.add(i), vaddq_s32(r, offv));
        i += 4;
    }
    while i < len {
        *seg.get_unchecked_mut(i) = reconstruct_q(es, *seg.get_unchecked(i)) + off;
        i += 1;
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn scale_apply(seg: &mut [i32], es: i32, off: i32) {
    for q in seg.iter_mut() {
        *q = reconstruct_q(es, *q) + off;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strand_quant::decode::{decode_tensor_fixed, decode_tensor_fixed_with_lut};
    use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};

    #[test]
    fn split_decode_is_bit_identical() {
        let configs = [
            TrellisConfig::for_bpw(3.0),
            TrellisConfig::for_bpw(2.0),
            TrellisConfig::for_bpw(4.0),
            TrellisConfig::for_bpw_l(2.0, 12),
            TrellisConfig::for_bpw_l(2.0, 5),
            TrellisConfig::for_bpw_l(4.0, 4),
            TrellisConfig::for_bpw_l(3.0, 5),
        ];
        for cfg in &configs {
            for seed in 0..48u64 {
                let n = 1 + (seed as usize * 97) % 4096;
                let w: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect();
                let variants = [
                    encode_tensor(&w, cfg),
                    encode_tensor_with(&w, cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
                    encode_tensor_with(&w, cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
                    encode_tensor_with(&w, cfg, &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() }),
                ];
                for enc in &variants {
                    let reference = decode_tensor_fixed(enc, cfg);
                    assert_eq!(
                        decode_q12_split(enc, cfg),
                        reference,
                        "SPLIT decode diverged: L={} k={} n={} seed={} tail={} affine={}",
                        cfg.l_bits,
                        cfg.k_bits,
                        n,
                        seed,
                        enc.tail_biting,
                        enc.has_affine_min
                    );
                    assert_eq!(
                        decode_q12_split_par(enc, cfg),
                        reference,
                        "SPLIT-PAR decode diverged: L={} k={} n={} seed={} tail={} affine={}",
                        cfg.l_bits,
                        cfg.k_bits,
                        n,
                        seed,
                        enc.tail_biting,
                        enc.has_affine_min
                    );
                }
            }
        }
    }

    #[test]
    fn split_bit_identical_large() {
        for cfg in [TrellisConfig::for_bpw(3.0), TrellisConfig::for_bpw_l(2.0, 12)] {
            let n = 256 * 257 + 17;
            let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0011).sin() * 0.6).collect();
            let enc = encode_tensor(&w, &cfg);
            let reference = decode_tensor_fixed(&enc, &cfg);
            assert_eq!(decode_q12_split(&enc, &cfg), reference, "split large L={}", cfg.l_bits);
            assert_eq!(decode_q12_split_par(&enc, &cfg), reference, "split-par large L={}", cfg.l_bits);
        }
    }

    #[test]
    fn split_vector_trellis_fallback() {
        let cfg = TrellisConfig::for_bpw(3.0).with_vec_dim(2);
        let (ns, d) = (cfg.num_states(), cfg.vec_dim());
        let lut: Vec<i32> = (0..ns * d).map(|i| ((i as u32).wrapping_mul(2654435761) >> 20) as i32 - 2048).collect();
        let w: Vec<f32> = (0..700).map(|i| ((i as f32) * 0.011).cos() * 0.3).collect();
        let enc = encode_tensor(&w, &cfg);
        let want = decode_tensor_fixed_with_lut(&enc, &cfg, &lut);
        assert_eq!(decode_q12_split_with_lut(&enc, &cfg, &lut), want, "split vec fallback");
        assert_eq!(decode_q12_split_par_with_lut(&enc, &cfg, &lut), want, "split-par vec fallback");
    }
}
