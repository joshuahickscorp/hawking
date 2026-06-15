
use strand_quant::codebook::codebook_lut;
use strand_quant::decode::reconstruct_q;
use strand_quant::encode::{EncodedTensor, SUB_BLOCK};
use strand_quant::TrellisConfig;

use crate::block_walk::{block_init_state, block_plans, exceeds_max_sub, BlockPlan, SideInfo, WordReader};
use crate::gemv::decode_q12_fast_with_lut;

const LANES: usize = 8;

pub fn decode_q12_neonlut(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<i32> {
    decode_q12_neonlut_with_lut(enc, cfg, codebook_lut(cfg.l_bits))
}

pub fn decode_q12_neonlut_with_lut(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
) -> Vec<i32> {
    decode_dispatch(enc, cfg, lut, false)
}

pub fn decode_q12_neonlut_scalar_gather(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<i32> {
    decode_dispatch(enc, cfg, codebook_lut(cfg.l_bits), true)
}

fn decode_dispatch(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
    force_scalar_gather: bool,
) -> Vec<i32> {
    
    let in_envelope = cfg.vec_dim() == 1
        && cfg.l_bits <= 8
        && lut.len() == cfg.num_states()
        && !exceeds_max_sub(enc)
        && lut.iter().all(|&v| v >= i16::MIN as i32 && v <= i16::MAX as i32);
    if !in_envelope {
        return decode_q12_fast_with_lut(enc, cfg, lut);
    }
    #[cfg(target_arch = "aarch64")]
    {
        return aarch64_impl::decode(enc, cfg, lut, force_scalar_gather);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = force_scalar_gather;
        decode_q12_fast_with_lut(enc, cfg, lut)
    }
}

fn decode_block_scalar(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
    plan: &BlockPlan,
    blk_idx: usize,
    out: &mut [i32],
) {
    let blk = &enc.blocks[blk_idx];
    let k = cfg.k_bits;
    let mask = cfg.state_mask();
    let input_mask = cfg.num_inputs() - 1;
    let n = plan.n;

    let side = SideInfo::hoist(blk, enc.has_affine_min);
    let mut state = block_init_state(blk, &enc.bits, plan.start_bit, cfg, enc.tail_biting);
    let mut reader = WordReader::new(&enc.bits, plan.start_bit);

    let mut i = 0usize;
    for (&es, &o) in side.eff().iter().zip(side.off().iter()) {
        let end = (i + SUB_BLOCK).min(n);
        while i < end {
            let sym = reader.pop(k) & input_mask;
            state = ((state << k) | sym) & mask;
            out[plan.out_off + i] = reconstruct_q(es, lut[state]) + o;
            i += 1;
        }
    }
}

fn scale_pass(
    qbuf: &[i16],
    sides: &[SideInfo],
    plans: &[BlockPlan],
    n: usize,
    out: &mut [i32],
) {
    for (lane, (side, plan)) in sides.iter().zip(plans.iter()).enumerate() {
        let base = plan.out_off;
        let mut i = 0usize;
        for (&es, &o) in side.eff().iter().zip(side.off().iter()) {
            let end = (i + SUB_BLOCK).min(n);
            while i < end {
                let q = qbuf[i * LANES + lane] as i32;
                out[base + i] = reconstruct_q(es, q) + o;
                i += 1;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64_impl {
    use super::*;
    use std::arch::aarch64::*;

    #[inline(always)]
    fn pop_lane(bytes: &[u8], acc: &mut u64, have: &mut u32, widx: &mut usize, k: u32) -> u16 {
        if *have < k {
            *widx += 1;
            let nxt = WordReader::load_u32_le(bytes, *widx) as u64;
            *acc |= nxt << *have;
            *have += 32;
        }
        let sym = (*acc & ((1u64 << k) - 1)) as u16;
        *acc >>= k;
        *have -= k;
        sym
    }

    pub(super) fn decode(
        enc: &EncodedTensor,
        cfg: &TrellisConfig,
        lut: &[i32],
        force_scalar_gather: bool,
    ) -> Vec<i32> {
        let plans = block_plans(enc, cfg.k_bits as usize);
        let mut out = vec![0i32; enc.total];
        if enc.blocks.is_empty() {
            return out;
        }

        let num_states = cfg.num_states();
        let mut lut16 = [0i16; 256];
        for (d, &v) in lut16.iter_mut().zip(lut.iter()) {
            *d = v as i16;
        }

        let use_tbl = !force_scalar_gather && num_states * 2 <= 256;

        let mut qbuf: Vec<i16> = Vec::new();
        let mut b = 0usize;
        while b < enc.blocks.len() {
            let group_ok = b + LANES <= enc.blocks.len()
                && (b..b + LANES).all(|j| plans[j].n == plans[b].n);
            if !group_ok {
                decode_block_scalar(enc, cfg, lut, &plans[b], b, &mut out);
                b += 1;
                continue;
            }
            let n = plans[b].n;
            qbuf.clear();
            qbuf.resize(n * LANES, 0);

            let mut states = [0u16; LANES];
            let mut acc = [0u64; LANES];
            let mut have = [0u32; LANES];
            let mut widx = [0usize; LANES];
            let mut sides: Vec<SideInfo> = Vec::with_capacity(LANES);
            for l in 0..LANES {
                let plan = &plans[b + l];
                let blk = &enc.blocks[b + l];
                states[l] = block_init_state(blk, &enc.bits, plan.start_bit, cfg, enc.tail_biting)
                    as u16;
                let r = WordReader::new(&enc.bits, plan.start_bit);
                acc[l] = r.acc;
                have[l] = r.have;
                widx[l] = r.word_idx;
                sides.push(SideInfo::hoist(blk, enc.has_affine_min));
            }

            if use_tbl {
                
                unsafe {
                    match num_states * 2 {
                        0..=64 => simd_steps_tbl::<1>(
                            enc, cfg, &lut16, n, &mut states, &mut acc, &mut have, &mut widx,
                            &mut qbuf,
                        ),
                        65..=128 => simd_steps_tbl::<2>(
                            enc, cfg, &lut16, n, &mut states, &mut acc, &mut have, &mut widx,
                            &mut qbuf,
                        ),
                        _ => simd_steps_tbl::<4>(
                            enc, cfg, &lut16, n, &mut states, &mut acc, &mut have, &mut widx,
                            &mut qbuf,
                        ),
                    }
                }
            } else {
                scalar_gather_steps(
                    enc, cfg, &lut16, n, &mut states, &mut acc, &mut have, &mut widx, &mut qbuf,
                );
            }

            scale_pass(&qbuf, &sides, &plans[b..b + LANES], n, &mut out);
            b += LANES;
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "neon")]
    unsafe fn simd_steps_tbl<const NT: usize>(
        enc: &EncodedTensor,
        cfg: &TrellisConfig,
        lut16: &[i16; 256],
        n: usize,
        states: &mut [u16; LANES],
        acc: &mut [u64; LANES],
        have: &mut [u32; LANES],
        widx: &mut [usize; LANES],
        qbuf: &mut [i16],
    ) {
        let k = cfg.k_bits;
        let bytes: &[u8] = &enc.bits;
        let mask_v = vdupq_n_u16(cfg.state_mask() as u16);
        let kshift = vdupq_n_s16(k as i16);
        let bias = vdupq_n_u16(256);

        let tp = lut16.as_ptr() as *const u8;
        let lt = |o: usize| -> uint8x16x4_t {
            uint8x16x4_t(
                vld1q_u8(tp.add(o)),
                vld1q_u8(tp.add(o + 16)),
                vld1q_u8(tp.add(o + 32)),
                vld1q_u8(tp.add(o + 48)),
            )
        };
        let t0 = lt(0);
        let t1 = if NT >= 2 { lt(64) } else { t0 };
        let t2 = if NT >= 4 { lt(128) } else { t0 };
        let t3 = if NT >= 4 { lt(192) } else { t0 };
        let sub64 = vdupq_n_u8(64);
        let sub128 = vdupq_n_u8(128);
        let sub192 = vdupq_n_u8(192);

        let mut state_v = vld1q_u16(states.as_ptr());
        let qp = qbuf.as_mut_ptr();
        for step in 0..n {
            let mut syms = [0u16; LANES];
            
            for l in 0..LANES {
                syms[l] = pop_lane(bytes, &mut acc[l], &mut have[l], &mut widx[l], k);
            }
            let sym_v = vld1q_u16(syms.as_ptr());
            state_v = vandq_u16(vorrq_u16(vshlq_u16(state_v, kshift), sym_v), mask_v);

            let idx = vmlaq_n_u16(bias, state_v, 514);
            let idx_b = vreinterpretq_u8_u16(idx);
            let mut r = vqtbl4q_u8(t0, idx_b);
            if NT >= 2 {
                r = vorrq_u8(r, vqtbl4q_u8(t1, vsubq_u8(idx_b, sub64)));
            }
            if NT >= 4 {
                r = vorrq_u8(r, vqtbl4q_u8(t2, vsubq_u8(idx_b, sub128)));
                r = vorrq_u8(r, vqtbl4q_u8(t3, vsubq_u8(idx_b, sub192)));
            }
            vst1q_s16(qp.add(step * LANES), vreinterpretq_s16_u8(r));
        }
        vst1q_u16(states.as_mut_ptr(), state_v);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn scalar_gather_steps(
        enc: &EncodedTensor,
        cfg: &TrellisConfig,
        lut16: &[i16; 256],
        n: usize,
        states: &mut [u16; LANES],
        acc: &mut [u64; LANES],
        have: &mut [u32; LANES],
        widx: &mut [usize; LANES],
        qbuf: &mut [i16],
    ) {
        let k = cfg.k_bits;
        let mask = cfg.state_mask() as u16;
        let bytes: &[u8] = &enc.bits;
        for step in 0..n {
            let row = step * LANES;
            for l in 0..LANES {
                let sym = pop_lane(bytes, &mut acc[l], &mut have[l], &mut widx[l], k);
                states[l] = ((states[l] << k) | sym) & mask;
                qbuf[row + l] = lut16[states[l] as usize];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strand_quant::decode::{decode_lean, decode_tensor_fixed};
    use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};

    #[test]
    fn neonlut_is_bit_identical() {
        let configs = [
            TrellisConfig::for_bpw(3.0),        
            TrellisConfig::for_bpw(2.0),        
            TrellisConfig::for_bpw(4.0),        
            TrellisConfig::for_bpw_l(2.0, 7),   
            TrellisConfig::for_bpw_l(3.0, 6),   
            TrellisConfig::for_bpw_l(3.0, 8),   
            TrellisConfig::for_bpw_l(2.0, 5),   
            TrellisConfig::for_bpw_l(2.0, 12),  
        ];
        for cfg in configs {
            for seed in 0..48u64 {
                
                let n = 1 + (seed as usize * 211) % 4096;
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
                        &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
                    ),
                ];
                for enc in &variants {
                    let reference = decode_tensor_fixed(enc, &cfg);
                    assert_eq!(decode_lean(enc, &cfg), reference, "lean != fixed (harness)");
                    assert_eq!(
                        decode_q12_neonlut(enc, &cfg),
                        reference,
                        "neonlut diverged: L={} k={} n={n} seed={seed} tail={} affine={}",
                        cfg.l_bits, cfg.k_bits, enc.tail_biting, enc.has_affine_min
                    );
                    assert_eq!(
                        decode_q12_neonlut_scalar_gather(enc, &cfg),
                        reference,
                        "neonlut-scalar-gather diverged: L={} k={} n={n} seed={seed} tail={} affine={}",
                        cfg.l_bits, cfg.k_bits, enc.tail_biting, enc.has_affine_min
                    );
                }
            }
        }
    }

    #[test]
    fn frozen_luts_fit_i16_through_l8() {
        for l in 4..=8u32 {
            let lut = strand_quant::codebook::codebook_lut(l);
            let max = lut.iter().map(|v| v.unsigned_abs()).max().unwrap();
            assert!(max <= i16::MAX as u32, "L={l} max |q| {max} overflows i16");
        }
    }
}
