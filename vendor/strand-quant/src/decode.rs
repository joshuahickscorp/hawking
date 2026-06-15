
use crate::codebook::QUANTILE_SHIFT;
use crate::encode::EncodedTensor;
use crate::trellis::{read_bits, TrellisConfig};

pub const SCALE_SHIFT: u32 = 16;

#[inline]
pub fn reconstruct_q(scale_q: i32, quantile_q: i32) -> i32 {
    let prod = (scale_q as i64) * (quantile_q as i64);
    (prod >> SCALE_SHIFT) as i32
}

pub const SUB_SCALE_SHIFT: u32 = 6;

#[inline]
pub fn eff_scale_q(scale_q: i32, code: u8) -> i32 {
    let mult = (code as i64 & 0x3F) + 1; 
    (((scale_q as i64) * mult) >> SUB_SCALE_SHIFT) as i32
}

#[inline]
pub fn eff_min_q(min_base_q: i32, code: u8) -> i32 {
    let mag = (code & 0x1F) as i64; 
    if mag == 0 {
        return 0;
    }
    let base = (min_base_q.unsigned_abs()) as i64;
    let signed = if code & 0x20 != 0 { base * mag } else { -(base * mag) };
    (signed / 31) as i32
}

pub fn decode_tensor_fixed(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<i32> {
    // Codebook sourced per `cfg.codebook_mode`. `StoredLut` borrows the frozen
    // `&'static` table (zero alloc); `ComputedAcklam` materialises the identical
    // integers via the pure-integer Acklam path. Byte-for-byte the same scalar
    // codebook either way (Variant A is contract-tested exact).
    let lut = cfg.codebook();
    decode_tensor_fixed_with_lut(enc, cfg, &lut)
}

pub fn decode_tensor_fixed_with_lut(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
) -> Vec<i32> {
    
    if cfg.vec_dim() > 1 {
        return decode_tensor_fixed_with_lut_vec(enc, cfg, lut);
    }
    use crate::encode::{n_sub_blocks, unpack_sub_scales, SUB_BLOCK};

    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;

    let mut out = Vec::with_capacity(enc.total);
    let mut bit_cursor = 0usize;

    for blk in &enc.blocks {
        
        let n_sub = n_sub_blocks(blk.n as usize);
        let mults = unpack_sub_scales(&blk.sub_scales, n_sub);
        let eff: Vec<i32> = mults.iter().map(|&m| eff_scale_q(blk.scale_q, m)).collect();
        
        let offs: Vec<i32> = if enc.has_affine_min {
            let codes = unpack_sub_scales(&blk.mins, n_sub);
            codes.iter().map(|&c| eff_min_q(blk.min_base_q, c)).collect()
        } else {
            Vec::new()
        };

        let nk = (blk.n as usize) * (k as usize);
        let start_state = if enc.tail_biting && nk >= cfg.l_bits as usize {
            let mut s = 0usize;
            let mut c = bit_cursor;
            for _ in 0..blk.n as usize {
                let sym = read_bits(&enc.bits, c, k) & input_mask;
                c += k as usize;
                s = ((s << k) | sym) & mask;
            }
            s
        } else {
            blk.init_state as usize & mask
        };

        let mut state = start_state;
        for i in 0..blk.n as usize {
            let sym = read_bits(&enc.bits, bit_cursor, k) & input_mask;
            bit_cursor += k as usize;
            state = ((state << k) | sym) & mask;
            
            let q = lut[state];
            
            let es = eff[i / SUB_BLOCK];
            let off = offs.get(i / SUB_BLOCK).copied().unwrap_or(0);
            out.push(reconstruct_q(es, q) + off);
        }
    }
    out
}

#[inline]
fn load_u32_le(bytes: &[u8], wi: usize) -> u32 {
    let b = wi * 4;
    let g = |o: usize| -> u32 { if b + o < bytes.len() { bytes[b + o] as u32 } else { 0 } };
    g(0) | (g(1) << 8) | (g(2) << 16) | (g(3) << 24)
}

pub(crate) struct WordBitReader<'a> {
    bytes: &'a [u8],
    word_idx: usize,
    acc: u64,
    have: u32,
}

impl<'a> WordBitReader<'a> {
    
    #[inline]
    pub(crate) fn new(bytes: &'a [u8], start_bit: usize) -> Self {
        let word_idx = start_bit >> 5;
        let bit_in_w = (start_bit & 31) as u32;
        let acc = (load_u32_le(bytes, word_idx) as u64) >> bit_in_w;
        WordBitReader { bytes, word_idx, acc, have: 32 - bit_in_w }
    }

    #[inline]
    pub(crate) fn pop(&mut self, k: u32) -> usize {
        if self.have < k {
            self.word_idx += 1;
            let nxt = load_u32_le(self.bytes, self.word_idx) as u64;
            self.acc |= nxt << self.have;
            self.have += 32;
        }
        let sym = (self.acc & ((1u64 << k) - 1)) as usize;
        self.acc >>= k;
        self.have -= k;
        sym
    }
}

pub fn decode_lean(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<i32> {
    // Codebook sourced per `cfg.codebook_mode` (see `decode_tensor_fixed`); the
    // scalar codebook is byte-identical under either mode (Variant A exact).
    let lut = cfg.codebook();
    decode_lean_with_lut(enc, cfg, &lut)
}

pub fn decode_lean_with_lut(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32]) -> Vec<i32> {
    if cfg.vec_dim() > 1 {
        return decode_tensor_fixed_with_lut_vec(enc, cfg, lut);
    }
    use crate::encode::{n_sub_blocks, unpack_sub_scales, SUB_BLOCK};

    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    
    let fold = SUB_BLOCK >= cfg.num_states();

    let mut out = Vec::with_capacity(enc.total);
    
    let mut reader = WordBitReader::new(&enc.bits, 0);
    
    let mut folded: Vec<i32> = Vec::new();

    for blk in &enc.blocks {
        let n_sub = n_sub_blocks(blk.n as usize);
        let mults = unpack_sub_scales(&blk.sub_scales, n_sub);
        let eff: Vec<i32> = mults.iter().map(|&m| eff_scale_q(blk.scale_q, m)).collect();
        let offs: Vec<i32> = if enc.has_affine_min {
            let codes = unpack_sub_scales(&blk.mins, n_sub);
            codes.iter().map(|&c| eff_min_q(blk.min_base_q, c)).collect()
        } else {
            Vec::new()
        };

        let nk = (blk.n as usize) * (k as usize);
        let start_state = if enc.tail_biting && nk >= cfg.l_bits as usize {
            
            let bit_cursor = out.len() * (k as usize);
            let mut s = 0usize;
            let mut c = bit_cursor;
            for _ in 0..blk.n as usize {
                let sym = read_bits(&enc.bits, c, k) & input_mask;
                c += k as usize;
                s = ((s << k) | sym) & mask;
            }
            s
        } else {
            blk.init_state as usize & mask
        };

        let mut state = start_state;
        if fold {
            
            let ns = cfg.num_states();
            folded.clear();
            folded.resize(n_sub * ns, 0);
            for (sb, &es) in eff.iter().enumerate() {
                let base = sb * ns;
                for s in 0..ns {
                    folded[base + s] = reconstruct_q(es, lut[s]);
                }
            }
            for i in 0..blk.n as usize {
                let sym = reader.pop(k) & input_mask;
                state = ((state << k) | sym) & mask;
                let sb = i / SUB_BLOCK;
                let off = offs.get(sb).copied().unwrap_or(0);
                out.push(folded[sb * ns + state] + off);
            }
        } else {
            for i in 0..blk.n as usize {
                let sym = reader.pop(k) & input_mask;
                state = ((state << k) | sym) & mask;
                let q = lut[state];
                let es = eff[i / SUB_BLOCK];
                let off = offs.get(i / SUB_BLOCK).copied().unwrap_or(0);
                out.push(reconstruct_q(es, q) + off);
            }
        }
    }
    out
}

pub fn decode_tensor_fixed_with_lut_vec(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
) -> Vec<i32> {
    use crate::encode::{n_sub_blocks, unpack_sub_scales, SUB_BLOCK};

    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let d = cfg.vec_dim();

    let mut out = Vec::with_capacity(enc.total);
    let mut bit_cursor = 0usize;

    for blk in &enc.blocks {
        let n = blk.n as usize;
        let n_sub = n_sub_blocks(n);
        let mults = unpack_sub_scales(&blk.sub_scales, n_sub);
        let eff: Vec<i32> = mults.iter().map(|&m| eff_scale_q(blk.scale_q, m)).collect();
        let offs: Vec<i32> = if enc.has_affine_min {
            let codes = unpack_sub_scales(&blk.mins, n_sub);
            codes.iter().map(|&c| eff_min_q(blk.min_base_q, c)).collect()
        } else {
            Vec::new()
        };

        let n_steps = n.div_ceil(d);

        let nk = n_steps * (k as usize);
        let start_state = if enc.tail_biting && nk >= cfg.l_bits as usize {
            let mut s = 0usize;
            let mut c = bit_cursor;
            for _ in 0..n_steps {
                let sym = read_bits(&enc.bits, c, k) & input_mask;
                c += k as usize;
                s = ((s << k) | sym) & mask;
            }
            s
        } else {
            blk.init_state as usize & mask
        };

        let mut state = start_state;
        let mut produced = 0usize; 
        for _ in 0..n_steps {
            let sym = read_bits(&enc.bits, bit_cursor, k) & input_mask;
            bit_cursor += k as usize;
            state = ((state << k) | sym) & mask;
            
            let base = state * d;
            
            let remaining = n - produced;
            let emit = remaining.min(d);
            for j in 0..emit {
                let i = produced + j;
                let q = lut[base + j];
                let es = eff[i / SUB_BLOCK];
                let off = offs.get(i / SUB_BLOCK).copied().unwrap_or(0);
                out.push(reconstruct_q(es, q) + off);
            }
            produced += emit;
        }
    }
    out
}

const Q12_TO_F32: f32 = 1.0 / (1u32 << QUANTILE_SHIFT) as f32;

pub fn decode_tensor(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<f32> {
    decode_tensor_fixed(enc, cfg)
        .into_iter()
        .map(|q| (q as f32) * Q12_TO_F32)
        .collect()
}

#[cfg(test)]
mod lean_tests {
    use super::*;
    use crate::encode::{encode_tensor, encode_tensor_with, EncodeOpts, SUB_BLOCK};

    #[test]
    fn decode_lean_is_bit_identical() {
        
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
                
                let n = 1 + (seed as usize * 37) % 2048;
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
                        &EncodeOpts {
                            affine_min: true,
                            ..Default::default()
                        },
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
                    assert_eq!(
                        decode_lean(enc, &cfg),
                        decode_tensor_fixed(enc, &cfg),
                        "decode_lean diverged: L={} k={} n={} seed={} tail={} affine={}",
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
    fn word_reader_matches_read_bits() {
        
        let bytes: Vec<u8> = (0..257u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        for k in 1..=6u32 {
            for start in 0..40usize {
                let mut reader = WordBitReader::new(&bytes, start);
                let total_bits = bytes.len() * 8;
                
                let n_syms = (total_bits - start) / k as usize + 4;
                let mut cursor = start;
                for s in 0..n_syms {
                    let want = read_bits(&bytes, cursor, k);
                    let got = reader.pop(k);
                    assert_eq!(got, want, "k={k} start={start} sym#{s}");
                    cursor += k as usize;
                }
            }
        }
    }
}
