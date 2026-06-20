
use crate::block_walk::{block_init_state, exceeds_max_sub, SideInfo, WordReader};
use rayon::prelude::*;
use strand_quant::codebook::codebook_lut;
use strand_quant::decode::reconstruct_q;
use strand_quant::encode::{EncodedTensor, SUB_BLOCK};
use strand_quant::trellis::TrellisConfig;

struct Lane<'a> {
    reader: WordReader<'a>,
    state: usize,
    side: SideInfo,
    dst: &'a mut [i32],
    n: usize,
    done: usize,
}

impl<'a> Lane<'a> {
    
    fn new(
        blk: &strand_quant::encode::BlockMeta,
        start_bit: usize,
        bits: &'a [u8],
        cfg: &TrellisConfig,
        has_affine: bool,
        tail_biting: bool,
        dst: &'a mut [i32],
    ) -> Self {
        let n = blk.n as usize;
        let side = SideInfo::hoist(blk, has_affine);
        let state = block_init_state(blk, bits, start_bit, cfg, tail_biting);
        Lane { reader: WordReader::new(bits, start_bit), state, side, dst, n, done: 0 }
    }

    #[inline(always)]
    fn step(&mut self, k: u32, input_mask: usize, mask: usize, lut: &[i32]) {
        let sym = self.reader.pop(k) & input_mask;
        self.state = ((self.state << k) | sym) & mask;
        
        let q = unsafe { *lut.get_unchecked(self.state) };
        let sb = self.done / SUB_BLOCK;
        let es = unsafe { *self.side.eff.get_unchecked(sb) };
        let o = unsafe { *self.side.off.get_unchecked(sb) };
        unsafe { *self.dst.get_unchecked_mut(self.done) = reconstruct_q(es, q) + o };
        self.done += 1;
    }
}

fn decode_chunk_interleaved<const S: usize>(lanes: &mut [Lane<'_>], cfg: &TrellisConfig, lut: &[i32]) {
    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;

    if lanes.len() == S {
        
        let bytes: &[u8] = lanes[0].reader.bytes;
        let mut acc = [0u64; S];
        let mut have = [0u32; S];
        let mut widx = [0usize; S];
        let mut state = [0usize; S];
        let mut done = [0usize; S];
        for l in 0..S {
            acc[l] = lanes[l].reader.acc;
            have[l] = lanes[l].reader.have;
            widx[l] = lanes[l].reader.word_idx;
            state[l] = lanes[l].state;
            done[l] = lanes[l].done;
        }

        'rounds: loop {
            for l in 0..S {
                if lanes[l].n - done[l] < SUB_BLOCK {
                    break 'rounds;
                }
            }
            
            let mut es = [0i32; S];
            let mut of = [0i32; S];
            let mut dp = [std::ptr::null_mut::<i32>(); S];
            for l in 0..S {
                let sb = done[l] / SUB_BLOCK;
                es[l] = lanes[l].side.eff[sb];
                of[l] = lanes[l].side.off[sb];
                
                dp[l] = unsafe { lanes[l].dst.as_mut_ptr().add(done[l]) };
            }
            for j in 0..SUB_BLOCK {
                for l in 0..S {
                    
                    if have[l] < k {
                        widx[l] += 1;
                        let nxt = WordReader::load_u32_le(bytes, widx[l]) as u64;
                        acc[l] |= nxt << have[l];
                        have[l] += 32;
                    }
                    let sym = (acc[l] & ((1u64 << k) - 1)) as usize & input_mask;
                    acc[l] >>= k;
                    have[l] -= k;
                    state[l] = ((state[l] << k) | sym) & mask;
                    
                    let q = unsafe { *lut.get_unchecked(state[l]) };
                    unsafe { *dp[l].add(j) = reconstruct_q(es[l], q) + of[l] };
                }
            }
            for l in 0..S {
                done[l] += SUB_BLOCK;
            }
        }

        for l in 0..S {
            lanes[l].reader.acc = acc[l];
            lanes[l].reader.have = have[l];
            lanes[l].reader.word_idx = widx[l];
            lanes[l].state = state[l];
            lanes[l].done = done[l];
        }
    }
    
    loop {
        let mut any = false;
        for lane in lanes.iter_mut() {
            if lane.done < lane.n {
                lane.step(k, input_mask, mask, lut);
                any = true;
            }
        }
        if !any {
            break;
        }
    }
}

pub fn decode_q12_interleave<const S: usize>(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<i32> {
    decode_q12_interleave_with_lut::<S>(enc, cfg, codebook_lut(cfg.l_bits))
}

pub fn decode_q12_interleave_with_lut<const S: usize>(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
) -> Vec<i32> {
    
    if cfg.vec_dim() > 1
        || SUB_BLOCK >= cfg.num_states()
        || exceeds_max_sub(enc)
    {
        return crate::gemv::decode_q12_fast_with_lut(enc, cfg, lut);
    }

    let k = cfg.k_bits as usize;
    let mut out = vec![0i32; enc.total];

    let mut start_bits = Vec::with_capacity(enc.blocks.len());
    let mut acc_bits = 0usize;
    for blk in &enc.blocks {
        start_bits.push(acc_bits);
        acc_bits += blk.n as usize * k;
    }

    let mut rest: &mut [i32] = &mut out;
    for (blocks, bits_off) in enc.blocks.chunks(S).zip(start_bits.chunks(S)) {
        let mut lanes: Vec<Lane<'_>> = Vec::with_capacity(blocks.len());
        for (blk, &sb) in blocks.iter().zip(bits_off.iter()) {
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(blk.n as usize);
            rest = tail;
            lanes.push(Lane::new(blk, sb, &enc.bits, cfg, enc.has_affine_min, enc.tail_biting, head));
        }
        decode_chunk_interleaved::<S>(&mut lanes, cfg, lut);
    }

    out
}

pub fn decode_q12_interleave_par<const S: usize>(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
) -> Vec<i32> {
    decode_q12_interleave_par_with_lut::<S>(enc, cfg, codebook_lut(cfg.l_bits))
}

pub fn decode_q12_interleave_par_with_lut<const S: usize>(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
) -> Vec<i32> {
    if cfg.vec_dim() > 1
        || SUB_BLOCK >= cfg.num_states()
        || exceeds_max_sub(enc)
    {
        return crate::gemv::decode_q12_fast_with_lut(enc, cfg, lut);
    }

    let k = cfg.k_bits as usize;
    let mut out = vec![0i32; enc.total];

    let mut start_bits = Vec::with_capacity(enc.blocks.len());
    let mut acc_bits = 0usize;
    for blk in &enc.blocks {
        start_bits.push(acc_bits);
        acc_bits += blk.n as usize * k;
    }
    
    let n_chunks = enc.blocks.len().div_ceil(S);
    let mut chunked_dsts: Vec<Vec<&mut [i32]>> = Vec::with_capacity(n_chunks);
    let mut rest: &mut [i32] = &mut out;
    for blocks in enc.blocks.chunks(S) {
        let mut dsts = Vec::with_capacity(blocks.len());
        for blk in blocks {
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(blk.n as usize);
            rest = tail;
            dsts.push(head);
        }
        chunked_dsts.push(dsts);
    }

    enc.blocks
        .par_chunks(S)
        .zip(start_bits.par_chunks(S))
        .zip(chunked_dsts.into_par_iter())
        .for_each(|((blocks, bits_off), dsts)| {
            let mut lanes: Vec<Lane<'_>> = blocks
                .iter()
                .zip(bits_off.iter())
                .zip(dsts.into_iter())
                .map(|((blk, &sb), dst)| {
                    Lane::new(blk, sb, &enc.bits, cfg, enc.has_affine_min, enc.tail_biting, dst)
                })
                .collect();
            decode_chunk_interleaved::<S>(&mut lanes, cfg, lut);
        });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use strand_quant::decode::decode_tensor_fixed;
    use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};

    #[test]
    fn interleave_is_bit_identical() {
        let configs = [
            TrellisConfig::for_bpw(2.0),       
            TrellisConfig::for_bpw(3.0),       
            TrellisConfig::for_bpw(4.0),       
            TrellisConfig::for_bpw_l(2.0, 12), 
        ];
        for cfg in &configs {
            for seed in 0..24u64 {
                
                let n = 1 + (seed as usize * 211) % 1500;
                let w: Vec<f32> = (0..n)
                    .map(|i| ((i as f32 + seed as f32) * 0.0173).sin() * 0.4)
                    .collect();
                let variants = [
                    encode_tensor(&w, cfg),
                    encode_tensor_with(
                        &w,
                        cfg,
                        &EncodeOpts { tail_biting: true, ..Default::default() },
                    ),
                    encode_tensor_with(
                        &w,
                        cfg,
                        &EncodeOpts { affine_min: true, ..Default::default() },
                    ),
                ];
                for enc in &variants {
                    let want = decode_tensor_fixed(enc, cfg);
                    assert_eq!(decode_q12_interleave::<2>(enc, cfg), want, "S=2 L={}", cfg.l_bits);
                    assert_eq!(decode_q12_interleave::<4>(enc, cfg), want, "S=4 L={}", cfg.l_bits);
                    assert_eq!(decode_q12_interleave::<8>(enc, cfg), want, "S=8 L={}", cfg.l_bits);
                    assert_eq!(
                        decode_q12_interleave_par::<4>(enc, cfg),
                        want,
                        "par S=4 L={}",
                        cfg.l_bits
                    );
                    assert_eq!(
                        decode_q12_interleave_par::<8>(enc, cfg),
                        want,
                        "par S=8 L={}",
                        cfg.l_bits
                    );
                }
            }
        }
    }

    #[test]
    fn fallback_configs_identical() {
        let w: Vec<f32> = (0..700).map(|i| ((i as f32) * 0.011).cos() * 0.3).collect();

        let fold_cfg = TrellisConfig::for_bpw_l(2.0, 5); 
        let enc = encode_tensor(&w, &fold_cfg);
        let want = decode_tensor_fixed(&enc, &fold_cfg);
        assert_eq!(decode_q12_interleave::<4>(&enc, &fold_cfg), want);
        assert_eq!(decode_q12_interleave_par::<8>(&enc, &fold_cfg), want);

        use strand_quant::decode::decode_tensor_fixed_with_lut;
        let vec_cfg = TrellisConfig::for_bpw(3.0).with_vec_dim(2);
        let (ns, d) = (vec_cfg.num_states(), vec_cfg.vec_dim());
        let lut: Vec<i32> = (0..ns * d)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 20) as i32 - 2048)
            .collect();
        let encv = encode_tensor(&w, &vec_cfg);
        let want_v = decode_tensor_fixed_with_lut(&encv, &vec_cfg, &lut);
        assert_eq!(decode_q12_interleave_with_lut::<4>(&encv, &vec_cfg, &lut), want_v);
        assert_eq!(decode_q12_interleave_par_with_lut::<8>(&encv, &vec_cfg, &lut), want_v);
    }
}
