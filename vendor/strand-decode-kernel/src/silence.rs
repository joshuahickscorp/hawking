
use crate::block_walk::{block_init_state, block_plans, exceeds_max_sub, SideInfo, WordReader};
use strand_quant::codebook::codebook_lut;
use strand_quant::decode::reconstruct_q;
use strand_quant::encode::{n_sub_blocks, unpack_sub_scales, EncodedTensor, SUB_BLOCK};
use strand_quant::TrellisConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockClass {
    
    Live,
    
    SilentZero,
    
    SilentConst,
}

pub struct SilenceMask {
    
    pub class: Vec<BlockClass>,
    
    pub consts: Vec<Option<Vec<i32>>>,
}

impl SilenceMask {
    
    pub fn build(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32]) -> Self {
        let n_blocks = enc.blocks.len();
        if cfg.vec_dim() > 1 || exceeds_max_sub(enc) {
            return SilenceMask { class: vec![BlockClass::Live; n_blocks], consts: vec![None; n_blocks] };
        }
        let w = crate::gemv::decode_q12_fast_with_lut(enc, cfg, lut);
        let mut class = Vec::with_capacity(n_blocks);
        let mut consts = Vec::with_capacity(n_blocks);
        let mut off = 0usize;
        for blk in &enc.blocks {
            let n = blk.n as usize;
            let vals = &w[off..off + n];
            let n_sub = n_sub_blocks(n);
            let mut c = Vec::with_capacity(n_sub);
            let mut constant = true;
            'sub: for sb in 0..n_sub {
                let s = sb * SUB_BLOCK;
                let e = (s + SUB_BLOCK).min(n);
                let v0 = vals[s];
                for &v in &vals[s + 1..e] {
                    if v != v0 {
                        constant = false;
                        break 'sub;
                    }
                }
                c.push(v0);
            }
            if constant {
                if c.iter().all(|&v| v == 0) {
                    class.push(BlockClass::SilentZero);
                } else {
                    class.push(BlockClass::SilentConst);
                }
                consts.push(Some(c));
            } else {
                class.push(BlockClass::Live);
                consts.push(None);
            }
            off += n;
        }
        SilenceMask { class, consts }
    }

    pub fn n_silent_zero(&self) -> usize {
        self.class.iter().filter(|c| **c == BlockClass::SilentZero).count()
    }

    pub fn n_silent_const(&self) -> usize {
        self.class.iter().filter(|c| **c == BlockClass::SilentConst).count()
    }
}

pub fn classify_strong(enc: &EncodedTensor) -> Vec<bool> {
    enc.blocks
        .iter()
        .map(|blk| {
            let n_sub = n_sub_blocks(blk.n as usize);
            let mults = unpack_sub_scales(&blk.sub_scales, n_sub);
            mults
                .iter()
                .all(|&m| strand_quant::decode::eff_scale_q(blk.scale_q, m) == 0)
        })
        .collect()
}

pub fn zero_nearest_q12(lut: &[i32]) -> i32 {
    lut.iter().map(|v| v.abs()).min().unwrap_or(0)
}

pub fn decode_q12_silence(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
    mask: &SilenceMask,
) -> Vec<i32> {
    if cfg.vec_dim() > 1 || exceeds_max_sub(enc) {
        return strand_quant::decode::decode_lean_with_lut(enc, cfg, lut);
    }
    debug_assert_eq!(mask.class.len(), enc.blocks.len(), "mask/tensor block count mismatch");

    let mask_bits = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let num_states = cfg.num_states();
    let fold = SUB_BLOCK >= num_states;
    let has_affine = enc.has_affine_min;
    let plans = block_plans(enc, k as usize);

    let mut out: Vec<i32> = Vec::with_capacity(enc.total);
    let mut folded: Vec<i32> = Vec::new();

    for (b, blk) in enc.blocks.iter().enumerate() {
        let n = blk.n as usize;
        if let Some(consts) = &mask.consts[b] {
            let mut i = 0usize;
            for &c in consts {
                let end = (i + SUB_BLOCK).min(n);
                while i < end {
                    out.push(c);
                    i += 1;
                }
            }
            continue;
        }
        let plan = plans[b];
        let side = SideInfo::hoist(blk, has_affine);
        let (eff, off) = (side.eff(), side.off());
        let n_sub = side.n_sub;
        let mut reader = WordReader::new(&enc.bits, plan.start_bit);
        let mut state = block_init_state(blk, &enc.bits, plan.start_bit, cfg, enc.tail_biting);
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
                    state = ((state << k) | sym) & mask_bits;
                    out.push(folded[base + state] + o);
                    i += 1;
                }
            }
        } else {
            let mut i = 0usize;
            for (&es, &o) in eff.iter().zip(off.iter()) {
                let end = (i + SUB_BLOCK).min(n);
                while i < end {
                    let sym = reader.pop(k) & input_mask;
                    state = ((state << k) | sym) & mask_bits;
                    let q = lut[state];
                    out.push(reconstruct_q(es, q) + o);
                    i += 1;
                }
            }
        }
    }
    out
}

pub fn matvec_silence(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
    mask: &SilenceMask,
    out_features: usize,
    in_features: usize,
    x: &[f32],
) -> Vec<f32> {
    assert_eq!(x.len(), in_features, "x must have in_features entries");
    let w = decode_q12_silence(enc, cfg, lut, mask);
    assert_eq!(w.len(), out_features * in_features, "decoded weight count mismatch");
    let inv = 1.0f32 / 4096.0;
    let mut y = vec![0.0f32; out_features];
    for o in 0..out_features {
        let row = &w[o * in_features..(o + 1) * in_features];
        let mut acc = 0.0f32;
        for i in 0..in_features {
            acc += (row[i] as f32) * inv * x[i];
        }
        y[o] = acc;
    }
    y
}

pub fn matvec_silence_skip(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
    mask: &SilenceMask,
    out_features: usize,
    in_features: usize,
    x: &[f32],
) -> Vec<f32> {
    assert_eq!(x.len(), in_features, "x must have in_features entries");
    let w = decode_q12_silence(enc, cfg, lut, mask);
    assert_eq!(w.len(), out_features * in_features, "decoded weight count mismatch");

    let plans = block_plans(enc, cfg.k_bits as usize);
    let mut zero_ranges: Vec<(usize, usize)> = Vec::new();
    for (b, plan) in plans.iter().enumerate() {
        if mask.class.get(b) == Some(&BlockClass::SilentZero) {
            let (s, e) = (plan.out_off, plan.out_off + plan.n);
            match zero_ranges.last_mut() {
                Some(last) if last.1 == s => last.1 = e,
                _ => zero_ranges.push((s, e)),
            }
        }
    }

    let inv = 1.0f32 / 4096.0;
    let mut y = vec![0.0f32; out_features];
    let mut zi = 0usize;
    for o in 0..out_features {
        let row_s = o * in_features;
        let row_e = row_s + in_features;
        while zi > 0 && zero_ranges[zi - 1].1 > row_s {
            zi -= 1;
        }
        let mut acc = 0.0f32;
        let mut i = row_s;
        let mut zj = zi;
        while i < row_e {
            while zj < zero_ranges.len() && zero_ranges[zj].1 <= i {
                zj += 1;
            }
            let (live_end, next_i) = match zero_ranges.get(zj) {
                Some(&(zs, ze)) if zs < row_e => {
                    if zs <= i {
                        i = ze.min(row_e);
                        continue;
                    }
                    (zs, zs)
                }
                _ => (row_e, row_e),
            };
            while i < live_end {
                acc += (w[i] as f32) * inv * x[i - row_s];
                i += 1;
            }
            i = next_i.max(i);
        }
        y[o] = acc;
        zi = zj;
    }
    y
}

pub struct TensorCensus {
    pub n_weights: usize,
    pub n_blocks: usize,
    pub n_sub: usize,
    
    pub n_silent_zero: usize,
    
    pub n_silent_const: usize,
    
    pub n_zero_level_blocks: usize,
    
    pub n_strong_silent: usize,
    
    pub n_sub_code0: usize,
    
    pub n_sub_code_max: usize,
    
    pub n_scaleq_zero: usize,
    
    pub state_hist: Vec<u64>,
    
    pub zero_level_visits: u64,
    
    pub n_silent_with_outlier: usize,
}

pub fn census_tensor(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
    outlier_idx: &[u32],
) -> TensorCensus {
    let lut = if lut.is_empty() { codebook_lut(cfg.l_bits) } else { lut };
    let num_states = cfg.num_states();
    let zmin = zero_nearest_q12(lut);
    let mask = SilenceMask::build(enc, cfg, lut);
    let strong = classify_strong(enc);

    let mut c = TensorCensus {
        n_weights: enc.total,
        n_blocks: enc.blocks.len(),
        n_sub: 0,
        n_silent_zero: mask.n_silent_zero(),
        n_silent_const: mask.n_silent_const(),
        n_zero_level_blocks: 0,
        n_strong_silent: strong.iter().filter(|&&s| s).count(),
        n_sub_code0: 0,
        n_sub_code_max: 0,
        n_scaleq_zero: enc.blocks.iter().filter(|b| b.scale_q == 0).count(),
        state_hist: vec![0u64; num_states],
        zero_level_visits: 0,
        n_silent_with_outlier: 0,
    };

    for blk in &enc.blocks {
        let n_sub = n_sub_blocks(blk.n as usize);
        c.n_sub += n_sub;
        for code in unpack_sub_scales(&blk.sub_scales, n_sub) {
            if code & 0x3F == 0 {
                c.n_sub_code0 += 1;
            }
            if code & 0x3F == 0x3F {
                c.n_sub_code_max += 1;
            }
        }
    }

    if cfg.vec_dim() == 1 {
        let mask_bits = cfg.state_mask();
        let k = cfg.k_bits;
        let input_mask = cfg.num_inputs() - 1;
        let plans = block_plans(enc, k as usize);
        for (b, blk) in enc.blocks.iter().enumerate() {
            let plan = plans[b];
            let mut reader = WordReader::new(&enc.bits, plan.start_bit);
            let mut state = block_init_state(blk, &enc.bits, plan.start_bit, cfg, enc.tail_biting);
            let mut all_zero_level = true;
            for _ in 0..blk.n as usize {
                let sym = reader.pop(k) & input_mask;
                state = ((state << k) | sym) & mask_bits;
                c.state_hist[state] += 1;
                if lut[state].abs() == zmin {
                    c.zero_level_visits += 1;
                } else {
                    all_zero_level = false;
                }
            }
            if all_zero_level && blk.n > 0 {
                c.n_zero_level_blocks += 1;
            }
        }
    }

    if !outlier_idx.is_empty() {
        let plans = block_plans(enc, cfg.k_bits as usize);
        for (b, plan) in plans.iter().enumerate() {
            if mask.class[b] == crate::silence::BlockClass::Live {
                continue;
            }
            let s = plan.out_off as u32;
            let e = (plan.out_off + plan.n) as u32;
            let lo = outlier_idx.partition_point(|&i| i < s);
            if lo < outlier_idx.len() && outlier_idx[lo] < e {
                c.n_silent_with_outlier += 1;
            }
        }
    }

    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use strand_quant::decode::decode_tensor_fixed;
    use strand_quant::encode::{encode_tensor, encode_tensor_with, BlockMeta, EncodeOpts};

    fn planted_weights(n: usize, block_len: usize, seed: u64) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let b = i / block_len;
                let last = (n - 1) / block_len;
                if b == 0 || b == 3 || b == last {
                    0.0
                } else {
                    ((i as f32 + seed as f32) * 0.0137).sin() * 0.5
                }
            })
            .collect()
    }

    fn synth_with_scaleq_zero(total: usize, k: u32, block_len: usize) -> EncodedTensor {
        let mut enc = crate::block_walk::gate_proto::synth_encoded(total, k, block_len);
        for b in [1usize, 2usize] {
            if b < enc.blocks.len() {
                let old: BlockMeta = enc.blocks[b].clone();
                enc.blocks[b] = BlockMeta { scale_q: 0, ..old };
            }
        }
        enc
    }

    #[test]
    fn silence_decode_is_byte_identical_and_detects_planted_blocks() {
        let configs = [
            TrellisConfig::for_bpw(3.0),
            TrellisConfig::for_bpw(2.0),
            TrellisConfig::for_bpw_l(2.0, 12),
            TrellisConfig::for_bpw_l(2.0, 5),
        ];
        for cfg in configs {
            let lut = codebook_lut(cfg.l_bits);
            for &n in &[2048usize, 1000, 257] {
                let w = planted_weights(n, cfg.block_len, 7);
                let variants = [
                    encode_tensor(&w, &cfg),
                    encode_tensor_with(&w, &cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
                    encode_tensor_with(&w, &cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
                ];
                for enc in &variants {
                    let mask = SilenceMask::build(enc, &cfg, lut);
                    let got = decode_q12_silence(enc, &cfg, lut, &mask);
                    let want = decode_tensor_fixed(enc, &cfg);
                    assert_eq!(got, want, "silence decode diverged L={} n={n}", cfg.l_bits);
                    if !enc.has_affine_min {
                        assert!(
                            mask.n_silent_zero() + mask.n_silent_const() >= 1,
                            "no silence detected on planted tensor L={} n={n}",
                            cfg.l_bits
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn scaleq_zero_blocks_are_strong_silent_and_exact() {
        let cfg = TrellisConfig::for_bpw_l(2.0, 12);
        let lut = codebook_lut(cfg.l_bits);
        let enc = synth_with_scaleq_zero(1024, cfg.k_bits, cfg.block_len);
        let strong = classify_strong(&enc);
        assert!(strong[1] && strong[2], "scale_q=0 blocks must be strong-silent");
        let mask = SilenceMask::build(&enc, &cfg, lut);
        assert!(mask.n_silent_zero() >= 2, "strong-silent blocks must classify SilentZero");
        assert_eq!(
            decode_q12_silence(&enc, &cfg, lut, &mask),
            decode_tensor_fixed(&enc, &cfg),
            "scale_q=0 silence decode diverged"
        );
    }

    #[test]
    fn matvec_silence_variants_are_bit_equal() {
        
        for &(rows, cols) in &[(8usize, 896usize), (4, 256), (5, 320)] {
            let cfg = TrellisConfig::for_bpw_l(2.0, 12);
            let lut = codebook_lut(cfg.l_bits);
            let w = planted_weights(rows * cols, cfg.block_len, 11);
            let enc = encode_tensor(&w, &cfg);
            let mask = SilenceMask::build(&enc, &cfg, lut);
            let x: Vec<f32> = (0..cols)
                .map(|i| match i % 5 {
                    0 => -((i as f32) * 0.013).cos(),
                    1 => 0.0,
                    2 => -0.0,
                    3 => f32::MIN_POSITIVE / 2.0,
                    _ => ((i as f32) * 0.07).cos(),
                })
                .collect();
            let y_ref = crate::matvec(&enc, &cfg, None, rows, cols, &x);
            let y_a = matvec_silence(&enc, &cfg, lut, &mask, rows, cols, &x);
            let y_b = matvec_silence_skip(&enc, &cfg, lut, &mask, rows, cols, &x);
            for o in 0..rows {
                assert_eq!(y_a[o].to_bits(), y_ref[o].to_bits(), "matvec_silence row {o} ({rows}x{cols})");
                assert_eq!(y_b[o].to_bits(), y_ref[o].to_bits(), "matvec_silence_skip row {o} ({rows}x{cols})");
            }
        }
    }

    #[test]
    fn census_counts_are_consistent() {
        let cfg = TrellisConfig::for_bpw_l(2.0, 12);
        let lut = codebook_lut(cfg.l_bits);
        let w = planted_weights(2048, cfg.block_len, 3);
        let enc = encode_tensor(&w, &cfg);
        let c = census_tensor(&enc, &cfg, lut, &[]);
        assert_eq!(c.n_weights, 2048);
        assert_eq!(c.n_blocks, enc.blocks.len());
        let visits: u64 = c.state_hist.iter().sum();
        assert_eq!(visits, 2048, "every weight visits exactly one state");
        assert!(c.n_silent_zero >= 1, "planted zero blocks must show in the census");
        let c2 = census_tensor(&enc, &cfg, lut, &[5u32]);
        assert_eq!(c2.n_silent_with_outlier, 1);
    }
}
