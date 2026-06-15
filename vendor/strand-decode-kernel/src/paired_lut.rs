
use crate::block_walk::{
    block_init_state, block_plans, exceeds_max_sub, BlockPlan, SideInfo, WordReader,
};
use rayon::prelude::*;
use strand_quant::codebook::codebook_lut;
use strand_quant::decode::reconstruct_q;
use strand_quant::encode::{BlockMeta, EncodedTensor, SUB_BLOCK};
use strand_quant::TrellisConfig;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct PairEntry {
    pub q1: i32,
    pub q2: i32,
}

pub struct PairTable {
    entries: Vec<PairEntry>,
    lut: Vec<i32>,
    
    idx_mask: usize,
    pub l_bits: u32,
    pub k_bits: u32,
}

impl PairTable {
    
    pub fn build(cfg: &TrellisConfig) -> Self {
        Self::build_with_lut(cfg, codebook_lut(cfg.l_bits))
    }

    pub fn build_with_lut(cfg: &TrellisConfig, lut: &[i32]) -> Self {
        assert_eq!(cfg.vec_dim(), 1, "paired table is scalar-trellis only");
        assert_eq!(lut.len(), cfg.num_states(), "scalar LUT must have 2^L entries");
        let l = cfg.l_bits;
        let k = cfg.k_bits;
        let n = 1usize << (l + k);
        let mask = cfg.state_mask();
        let mut entries = Vec::with_capacity(n);
        for idx in 0..n {
            
            entries.push(PairEntry { q1: lut[idx >> k], q2: lut[idx & mask] });
        }
        PairTable { entries, lut: lut.to_vec(), idx_mask: n - 1, l_bits: l, k_bits: k }
    }

    pub fn size_bytes(&self) -> usize {
        self.entries.len() * core::mem::size_of::<PairEntry>()
    }

    #[inline(always)]
    pub(crate) fn entries_slice(&self) -> &[PairEntry] {
        &self.entries
    }

    #[inline(always)]
    pub(crate) fn lut_slice(&self) -> &[i32] {
        &self.lut
    }

    #[inline(always)]
    pub(crate) fn index_mask(&self) -> usize {
        self.idx_mask
    }
}

pub fn decode_q12_paired(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<i32> {
    decode_q12_paired_with_lut(enc, cfg, codebook_lut(cfg.l_bits))
}

pub fn decode_q12_paired_with_lut(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
) -> Vec<i32> {
    if cfg.vec_dim() > 1 || exceeds_max_sub(enc) {
        return crate::gemv::decode_q12_fast_with_lut(enc, cfg, lut);
    }
    let table = PairTable::build_with_lut(cfg, lut);
    decode_q12_paired_with_table(enc, cfg, &table)
}

pub fn decode_q12_paired_with_table(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    table: &PairTable,
) -> Vec<i32> {
    assert_eq!(cfg.vec_dim(), 1, "paired decode is scalar-trellis only");
    assert_eq!((table.l_bits, table.k_bits), (cfg.l_bits, cfg.k_bits), "table/cfg mismatch");
    if exceeds_max_sub(enc) {
        return crate::gemv::decode_q12_fast_with_lut(enc, cfg, &table.lut);
    }

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
    for ((blk, plan), dst) in enc.blocks.iter().zip(plans.iter()).zip(slices.iter_mut()) {
        decode_block_paired(
            blk,
            plan,
            &enc.bits,
            cfg,
            table,
            enc.has_affine_min,
            enc.tail_biting,
            dst,
        );
    }
    out
}

pub fn decode_q12_paired_par(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<i32> {
    if cfg.vec_dim() > 1 || exceeds_max_sub(enc) {
        return crate::gemv_par::decode_q12_par(enc, cfg);
    }
    let table = PairTable::build(cfg);
    decode_q12_paired_par_with_table(enc, cfg, &table)
}

pub fn decode_q12_paired_par_with_table(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    table: &PairTable,
) -> Vec<i32> {
    assert_eq!(cfg.vec_dim(), 1, "paired decode is scalar-trellis only");
    assert_eq!((table.l_bits, table.k_bits), (cfg.l_bits, cfg.k_bits), "table/cfg mismatch");
    if exceeds_max_sub(enc) {
        return crate::gemv::decode_q12_fast_with_lut(enc, cfg, &table.lut);
    }

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
            decode_block_paired(
                blk,
                plan,
                &enc.bits,
                cfg,
                table,
                enc.has_affine_min,
                enc.tail_biting,
                dst,
            );
        });
    out
}

#[allow(unsafe_code, clippy::too_many_arguments)]
#[inline]
fn decode_block_paired(
    blk: &BlockMeta,
    plan: &BlockPlan,
    bits: &[u8],
    cfg: &TrellisConfig,
    table: &PairTable,
    has_affine: bool,
    tail_biting: bool,
    dst: &mut [i32],
) {
    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let k2 = 2 * k;
    let kmask = cfg.num_inputs() - 1; 
    let idx_mask = table.idx_mask;
    let n = blk.n as usize;

    let side = SideInfo::hoist(blk, has_affine);
    let (eff, off) = (side.eff(), side.off());
    let mut state = block_init_state(blk, bits, plan.start_bit, cfg, tail_biting);
    let mut reader = WordReader::new(bits, plan.start_bit);

    let entries = table.entries.as_ptr();
    let lut_ptr = table.lut.as_ptr();

    macro_rules! pair_step {
        ($i:expr, $es:expr, $o:expr) => {{
            let raw = reader.pop(k2);
            let sp = ((raw & kmask) << k) | (raw >> k);
            let t = (state << k2) | sp;
            
            let e = unsafe { *entries.add(t & idx_mask) };
            state = t & mask;
            unsafe {
                *dst.get_unchecked_mut($i) = reconstruct_q($es, e.q1) + $o;
                *dst.get_unchecked_mut($i + 1) = reconstruct_q($es, e.q2) + $o;
            }
        }};
    }

    let mut i = 0usize;
    for (&es, &o) in eff.iter().zip(off.iter()) {
        let end = (i + SUB_BLOCK).min(n);
        if end - i == SUB_BLOCK {
            
            for _ in 0..SUB_BLOCK / 2 {
                pair_step!(i, es, o);
                i += 2;
            }
        } else {
            while i + 1 < end {
                pair_step!(i, es, o);
                i += 2;
            }
            
            if i < end {
                let sym = reader.pop(k) & kmask;
                state = ((state << k as usize) | sym) & mask;
                
                let q = unsafe { *lut_ptr.add(state) };
                unsafe { *dst.get_unchecked_mut(i) = reconstruct_q(es, q) + o };
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strand_quant::decode::{decode_tensor_fixed, decode_tensor_fixed_with_lut};
    use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};

    #[test]
    fn pop_pair_equals_two_pops() {
        let bytes: Vec<u8> =
            (0..301u32).map(|i| (i.wrapping_mul(2654435761) >> 11) as u8).collect();
        for k in [2u32, 3, 4] {
            for start in 0..16usize {
                let mut r1 = WordReader::new(&bytes, start);
                let mut r2 = WordReader::new(&bytes, start);
                for _ in 0..600 {
                    let raw = r1.pop(2 * k);
                    let a = r2.pop(k);
                    let b = r2.pop(k);
                    assert_eq!(raw & ((1 << k) - 1), a, "first symbol must be the LOW k bits");
                    assert_eq!(raw >> k, b, "second symbol must be the HIGH k bits");
                }
            }
        }
    }

    #[test]
    fn paired_decode_is_bit_identical() {
        let configs = [
            TrellisConfig::for_bpw(3.0),       
            TrellisConfig::for_bpw(2.0),       
            TrellisConfig::for_bpw(4.0),       
            TrellisConfig::for_bpw_l(2.0, 12), 
            TrellisConfig::for_bpw_l(2.0, 5),  
            TrellisConfig::for_bpw_l(3.0, 5),  
            TrellisConfig::for_bpw_l(4.0, 4),  
        ];
        for cfg in configs {
            let table = PairTable::build(&cfg);
            assert_eq!(table.size_bytes(), (1usize << (cfg.l_bits + cfg.k_bits)) * 8);
            for seed in 0..64u64 {
                
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
                    let reference = decode_tensor_fixed(enc, &cfg);
                    let ctx = format!(
                        "L={} k={} n={} seed={} tail={} affine={}",
                        cfg.l_bits, cfg.k_bits, n, seed, enc.tail_biting, enc.has_affine_min
                    );
                    assert_eq!(decode_q12_paired(enc, &cfg), reference, "paired: {ctx}");
                    assert_eq!(
                        decode_q12_paired_with_table(enc, &cfg, &table),
                        reference,
                        "paired+table: {ctx}"
                    );
                    assert_eq!(
                        decode_q12_paired_par_with_table(enc, &cfg, &table),
                        reference,
                        "paired-par: {ctx}"
                    );
                }
            }
        }
    }

    #[test]
    fn paired_bit_identical_large_odd() {
        for (cfg, n) in [
            (TrellisConfig::for_bpw(3.0), 256 * 257 + 17),
            (TrellisConfig::for_bpw_l(2.0, 12), 256 * 129 + 33),
        ] {
            let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0011).sin() * 0.6).collect();
            let enc = encode_tensor(&w, &cfg);
            let reference = decode_tensor_fixed(&enc, &cfg);
            let table = PairTable::build(&cfg);
            assert_eq!(decode_q12_paired_with_table(&enc, &cfg, &table), reference);
            assert_eq!(decode_q12_paired_par_with_table(&enc, &cfg, &table), reference);
        }
    }

    #[test]
    fn paired_with_custom_lut_bit_identical() {
        let cfg = TrellisConfig::for_bpw(3.0);
        let ns = cfg.num_states();
        let lut: Vec<i32> = (0..ns)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 18) as i32 - 8192)
            .collect();
        let w: Vec<f32> = (0..3001).map(|i| ((i as f32) * 0.017).cos() * 0.4).collect();
        let enc = encode_tensor(&w, &cfg);
        let want = decode_tensor_fixed_with_lut(&enc, &cfg, &lut);
        assert_eq!(decode_q12_paired_with_lut(&enc, &cfg, &lut), want, "custom-LUT paired");
    }

    #[test]
    fn paired_vec_fallback_bit_identical() {
        let cfg = TrellisConfig::for_bpw(3.0).with_vec_dim(2);
        let (ns, d) = (cfg.num_states(), cfg.vec_dim());
        let lut: Vec<i32> = (0..ns * d)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 20) as i32 - 2048)
            .collect();
        let w: Vec<f32> = (0..700).map(|i| ((i as f32) * 0.011).cos() * 0.3).collect();
        let enc = encode_tensor(&w, &cfg);
        let want = decode_tensor_fixed_with_lut(&enc, &cfg, &lut);
        assert_eq!(decode_q12_paired_with_lut(&enc, &cfg, &lut), want, "paired vec fallback");
    }
}
