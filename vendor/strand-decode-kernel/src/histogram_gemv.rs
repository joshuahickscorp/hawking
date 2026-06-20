
use crate::block_walk::{
    block_init_state, block_plans, exceeds_max_sub, BlockPlan, SideInfo, WordReader,
};
use rayon::prelude::*;
use strand_quant::codebook::codebook_lut;
use strand_quant::decode::reconstruct_q;
use strand_quant::encode::{EncodedTensor, SUB_BLOCK};
use strand_quant::TrellisConfig;

pub fn histogram_gemm(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: Option<&[i32]>,
    out_features: usize,
    in_features: usize,
    xs: &[f32],
    batch: usize,
) -> Vec<f32> {
    histogram_gemm_impl(enc, cfg, lut, out_features, in_features, xs, batch, false)
}

pub fn histogram_gemm_scalar(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: Option<&[i32]>,
    out_features: usize,
    in_features: usize,
    xs: &[f32],
    batch: usize,
) -> Vec<f32> {
    histogram_gemm_impl(enc, cfg, lut, out_features, in_features, xs, batch, true)
}

pub fn histogram_matvec(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: Option<&[i32]>,
    out_features: usize,
    in_features: usize,
    x: &[f32],
) -> Vec<f32> {
    histogram_gemm(enc, cfg, lut, out_features, in_features, x, 1)
}

pub fn histogram_dot_q12(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: Option<&[i32]>,
    out_features: usize,
    in_features: usize,
    xq: &[i32],
) -> Vec<i64> {
    assert_eq!(xq.len(), in_features, "xq must have in_features entries");
    assert_eq!(
        enc.total,
        out_features * in_features,
        "encoded weight count != out_features * in_features"
    );
    if cfg.vec_dim() > 1 || exceeds_max_sub(enc) {
        return dot_q12_reference(enc, cfg, lut, out_features, in_features, xq);
    }
    let lut_resolved: &[i32] = match lut {
        Some(l) => l,
        None => codebook_lut(cfg.l_bits),
    };
    let num_states = cfg.num_states();
    debug_assert_eq!(lut_resolved.len(), num_states);

    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let has_affine = enc.has_affine_min;
    let tail_biting = enc.tail_biting;
    let plans = block_plans(enc, k as usize);

    let mut y = vec![0i64; out_features];
    let mut bucket = vec![0i64; num_states];
    let mut stamp = vec![0u32; num_states];
    let mut touched: Vec<u32> = Vec::with_capacity(num_states.min(4096));
    let mut epoch: u32 = 1;
    let mut cur_es = 0i32;
    let mut cur_off = 0i32;
    let mut have_region = false;

    let mut acc: i64 = 0;
    let mut col = 0usize;
    let mut row = 0usize;

    macro_rules! flush_i64 {
        () => {
            if !touched.is_empty() {
                for &v in touched.iter() {
                    let q = reconstruct_q(cur_es, lut_resolved[v as usize]) + cur_off;
                    acc += (q as i64) * bucket[v as usize];
                }
                touched.clear();
                epoch = epoch.wrapping_add(1);
                if epoch == 0 {
                    stamp.fill(0);
                    epoch = 1;
                }
            }
        };
    }

    for (bi, plan) in plans.iter().enumerate() {
        let blk = &enc.blocks[bi];
        let n = plan.n;
        let side = SideInfo::hoist(blk, has_affine);
        let mut state = block_init_state(blk, &enc.bits, plan.start_bit, cfg, tail_biting);
        let mut reader = WordReader::new(&enc.bits, plan.start_bit);
        let mut i = 0usize;
        for sb in 0..side.n_sub {
            let es = side.eff[sb];
            let o = side.off[sb];
            if !have_region {
                cur_es = es;
                cur_off = o;
                have_region = true;
            } else if es != cur_es || o != cur_off {
                flush_i64!();
                cur_es = es;
                cur_off = o;
            }
            let end = (i + SUB_BLOCK).min(n);
            while i < end {
                let sym = reader.pop(k) & input_mask;
                state = ((state << k) | sym) & mask;
                i += 1;
                let xv = xq[col] as i64;
                if stamp[state] != epoch {
                    stamp[state] = epoch;
                    touched.push(state as u32);
                    bucket[state] = xv;
                } else {
                    bucket[state] += xv;
                }
                col += 1;
                if col == in_features {
                    flush_i64!();
                    y[row] = acc;
                    acc = 0;
                    col = 0;
                    row += 1;
                }
            }
        }
    }
    debug_assert_eq!(row, out_features, "must finish exactly out_features rows");
    debug_assert_eq!(col, 0, "tensor must end on a row boundary");
    y
}

pub fn dot_q12_reference(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: Option<&[i32]>,
    out_features: usize,
    in_features: usize,
    xq: &[i32],
) -> Vec<i64> {
    assert_eq!(xq.len(), in_features, "xq must have in_features entries");
    let w = crate::decode_weights_q12(enc, cfg, lut);
    assert_eq!(w.len(), out_features * in_features, "decoded weight count mismatch");
    let mut y = vec![0i64; out_features];
    for o in 0..out_features {
        let row = &w[o * in_features..(o + 1) * in_features];
        let mut acc = 0i64;
        for i in 0..in_features {
            acc += (row[i] as i64) * (xq[i] as i64);
        }
        y[o] = acc;
    }
    y
}

#[derive(Debug, Clone, Copy)]
pub struct HistogramStats {
    pub weights: u64,
    pub regions: u64,
    
    pub occupied: u64,
}

impl HistogramStats {
    pub fn avg_region_len(&self) -> f64 {
        self.weights as f64 / (self.regions.max(1)) as f64
    }
    
    pub fn mul_ratio(&self) -> f64 {
        self.occupied as f64 / (self.weights.max(1)) as f64
    }
}

pub fn histogram_stats(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    in_features: usize,
) -> HistogramStats {
    assert!(
        cfg.vec_dim() == 1 && !exceeds_max_sub(enc),
        "histogram_stats instruments the fast walk only"
    );
    let num_states = cfg.num_states();
    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let has_affine = enc.has_affine_min;
    let tail_biting = enc.tail_biting;
    let plans = block_plans(enc, k as usize);

    let mut stamp = vec![0u32; num_states];
    let mut epoch: u32 = 1;
    let mut cur_touched: u64 = 0;
    let mut cur_es = 0i32;
    let mut cur_off = 0i32;
    let mut have_region = false;
    let mut stats = HistogramStats { weights: 0, regions: 0, occupied: 0 };
    let mut col = 0usize;

    macro_rules! flush_stat {
        () => {
            if cur_touched > 0 {
                stats.regions += 1;
                stats.occupied += cur_touched;
                cur_touched = 0;
                epoch = epoch.wrapping_add(1);
                if epoch == 0 {
                    stamp.fill(0);
                    epoch = 1;
                }
            }
        };
    }

    for (bi, plan) in plans.iter().enumerate() {
        let blk = &enc.blocks[bi];
        let n = plan.n;
        let side = SideInfo::hoist(blk, has_affine);
        let mut state = block_init_state(blk, &enc.bits, plan.start_bit, cfg, tail_biting);
        let mut reader = WordReader::new(&enc.bits, plan.start_bit);
        let mut i = 0usize;
        for sb in 0..side.n_sub {
            let es = side.eff[sb];
            let o = side.off[sb];
            if !have_region {
                cur_es = es;
                cur_off = o;
                have_region = true;
            } else if es != cur_es || o != cur_off {
                flush_stat!();
                cur_es = es;
                cur_off = o;
            }
            let end = (i + SUB_BLOCK).min(n);
            while i < end {
                let sym = reader.pop(k) & input_mask;
                state = ((state << k) | sym) & mask;
                i += 1;
                if stamp[state] != epoch {
                    stamp[state] = epoch;
                    cur_touched += 1;
                }
                stats.weights += 1;
                col += 1;
                if col == in_features {
                    flush_stat!();
                    col = 0;
                }
            }
        }
    }
    flush_stat!();
    let _ = (cur_touched, epoch);
    stats
}

#[allow(clippy::too_many_arguments)]
fn histogram_gemm_impl(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: Option<&[i32]>,
    out_features: usize,
    in_features: usize,
    xs: &[f32],
    batch: usize,
    force_scalar: bool,
) -> Vec<f32> {
    assert!(batch >= 1, "batch must be >= 1");
    assert_eq!(xs.len(), batch * in_features, "xs must be batch x in_features");
    assert_eq!(
        enc.total,
        out_features * in_features,
        "encoded weight count != out_features * in_features"
    );
    if cfg.vec_dim() > 1 || exceeds_max_sub(enc) {
        return crate::fused::fused_gemm(enc, cfg, lut, out_features, in_features, xs, batch);
    }
    let lut_resolved: &[i32] = match lut {
        Some(l) => l,
        None => codebook_lut(cfg.l_bits),
    };
    debug_assert_eq!(lut_resolved.len(), cfg.num_states());

    let mut xt = vec![0.0f32; xs.len()];
    for b in 0..batch {
        for i in 0..in_features {
            xt[i * batch + b] = xs[b * in_features + i];
        }
    }

    let plans = block_plans(enc, cfg.k_bits as usize);
    let mut y = vec![0.0f32; out_features * batch];
    let threads = rayon::current_num_threads().max(1);
    let rows_per_group = out_features.div_ceil(threads * 8).max(1);

    y.par_chunks_mut(rows_per_group * batch).enumerate().for_each(|(gi, yg)| {
        let r0 = gi * rows_per_group;
        let mut b_off = 0usize;
        while b_off < batch {
            let rem = batch - b_off;
            let step = if rem >= 64 {
                hist_chunk::<64, 16>(
                    enc, cfg, lut_resolved, &plans, in_features, batch, b_off, &xt, r0, yg,
                    force_scalar,
                );
                64
            } else if rem >= 16 {
                hist_chunk::<16, 4>(
                    enc, cfg, lut_resolved, &plans, in_features, batch, b_off, &xt, r0, yg,
                    force_scalar,
                );
                16
            } else if rem >= 4 {
                hist_chunk::<4, 1>(
                    enc, cfg, lut_resolved, &plans, in_features, batch, b_off, &xt, r0, yg,
                    force_scalar,
                );
                4
            } else {
                hist_rows::<1>(enc, cfg, lut_resolved, &plans, in_features, batch, b_off, &xt, r0, yg);
                1
            };
            b_off += step;
        }
    });
    y
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn hist_chunk<const B: usize, const NV: usize>(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
    plans: &[BlockPlan],
    in_features: usize,
    batch: usize,
    b_off: usize,
    xt: &[f32],
    r0: usize,
    yg: &mut [f32],
    force_scalar: bool,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if !force_scalar {
            unsafe {
                
                hist_rows_neon::<B, NV>(enc, cfg, lut, plans, in_features, batch, b_off, xt, r0, yg);
            }
            return;
        }
    }
    let _ = force_scalar;
    hist_rows::<B>(enc, cfg, lut, plans, in_features, batch, b_off, xt, r0, yg);
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn flush_hist<const B: usize>(
    touched: &mut Vec<u32>,
    stamp: &mut [u32],
    epoch: &mut u32,
    bucket: &[f32],
    lut: &[i32],
    es: i32,
    off: i32,
    acc: &mut [f32; B],
) {
    if touched.is_empty() {
        return;
    }
    let inv = 1.0f32 / 4096.0;
    for &v in touched.iter() {
        let q = reconstruct_q(es, lut[v as usize]) + off;
        let wf = (q as f32) * inv;
        let row = &bucket[v as usize * B..v as usize * B + B];
        for (a, &bv) in acc.iter_mut().zip(row.iter()) {
            *a += wf * bv;
        }
    }
    touched.clear();
    *epoch = epoch.wrapping_add(1);
    if *epoch == 0 {
        stamp.fill(0);
        *epoch = 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn hist_rows<const B: usize>(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
    plans: &[BlockPlan],
    in_features: usize,
    batch: usize,
    b_off: usize,
    xt: &[f32],
    r0: usize,
    yg: &mut [f32],
) {
    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let num_states = cfg.num_states();
    let has_affine = enc.has_affine_min;
    let tail_biting = enc.tail_biting;

    let nrows = yg.len() / batch;
    let g0 = r0 * in_features;
    let g1 = g0 + nrows * in_features;

    let mut bi = plans.partition_point(|p| p.out_off + p.n <= g0);

    let mut bucket = vec![0.0f32; num_states * B];
    let mut stamp = vec![0u32; num_states];
    let mut touched: Vec<u32> = Vec::with_capacity(num_states.min(4096));
    let mut epoch: u32 = 1;
    let mut cur_es = 0i32;
    let mut cur_off = 0i32;
    let mut have_region = false;

    let mut acc = [0.0f32; B];
    let mut col = 0usize;
    let mut row_rel = 0usize;

    'blocks: while bi < plans.len() {
        let plan = plans[bi];
        if plan.out_off >= g1 {
            break;
        }
        let blk = &enc.blocks[bi];
        let n = plan.n;

        let side = SideInfo::hoist(blk, has_affine);
        let n_sub = side.n_sub;
        let mut state = block_init_state(blk, &enc.bits, plan.start_bit, cfg, tail_biting);

        let mut reader = WordReader::new(&enc.bits, plan.start_bit);
        let mut i = 0usize;
        for sb in 0..n_sub {
            let es = side.eff[sb];
            let o = side.off[sb];
            if !have_region {
                cur_es = es;
                cur_off = o;
                have_region = true;
            } else if es != cur_es || o != cur_off {
                flush_hist::<B>(&mut touched, &mut stamp, &mut epoch, &bucket, lut, cur_es, cur_off, &mut acc);
                cur_es = es;
                cur_off = o;
            }
            let end = (i + SUB_BLOCK).min(n);
            while i < end {
                let sym = reader.pop(k) & input_mask;
                state = ((state << k) | sym) & mask;
                let g = plan.out_off + i;
                i += 1;
                if g < g0 {
                    continue;
                }
                if g >= g1 {
                    break 'blocks;
                }
                let xb = &xt[col * batch + b_off..col * batch + b_off + B];
                
                let rb = &mut bucket[state * B..state * B + B];
                if stamp[state] != epoch {
                    stamp[state] = epoch;
                    touched.push(state as u32);
                    rb.copy_from_slice(xb);
                } else {
                    for (r, &xv) in rb.iter_mut().zip(xb.iter()) {
                        *r += xv;
                    }
                }
                col += 1;
                if col == in_features {
                    flush_hist::<B>(
                        &mut touched, &mut stamp, &mut epoch, &bucket, lut, cur_es, cur_off,
                        &mut acc,
                    );
                    let yo = row_rel * batch + b_off;
                    yg[yo..yo + B].copy_from_slice(&acc);
                    acc = [0.0f32; B];
                    col = 0;
                    row_rel += 1;
                }
            }
        }
        bi += 1;
    }
    debug_assert_eq!(row_rel, nrows, "group must finish exactly its rows");
    debug_assert_eq!(col, 0, "group must end on a row boundary");
    debug_assert!(touched.is_empty(), "buckets must drain at the final row end");
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
unsafe fn hist_rows_neon<const B: usize, const NV: usize>(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
    plans: &[BlockPlan],
    in_features: usize,
    batch: usize,
    b_off: usize,
    xt: &[f32],
    r0: usize,
    yg: &mut [f32],
) {
    use core::arch::aarch64::*;
    debug_assert_eq!(NV * 4, B, "NV must be B/4");

    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let num_states = cfg.num_states();
    let has_affine = enc.has_affine_min;
    let tail_biting = enc.tail_biting;
    let inv = 1.0f32 / 4096.0;

    let nrows = yg.len() / batch;
    let g0 = r0 * in_features;
    let g1 = g0 + nrows * in_features;

    let mut bi = plans.partition_point(|p| p.out_off + p.n <= g0);

    let mut bucket = vec![0.0f32; num_states * B];
    let bptr = bucket.as_mut_ptr();
    let mut stamp = vec![0u32; num_states];
    let mut touched: Vec<u32> = Vec::with_capacity(num_states.min(4096));
    let mut epoch: u32 = 1;
    let mut cur_es = 0i32;
    let mut cur_off = 0i32;
    let mut have_region = false;

    let mut acc: [float32x4_t; NV] = [vdupq_n_f32(0.0); NV];
    let mut col = 0usize;
    let mut row_rel = 0usize;

    macro_rules! flush_neon {
        () => {
            if !touched.is_empty() {
                for &v in touched.iter() {
                    let q = reconstruct_q(cur_es, *lut.get_unchecked(v as usize)) + cur_off;
                    let wv = vdupq_n_f32((q as f32) * inv);
                    let rp = bptr.add(v as usize * B) as *const f32;
                    for (l, a) in acc.iter_mut().enumerate() {
                        let bv = vld1q_f32(rp.add(l * 4));
                        *a = vaddq_f32(*a, vmulq_f32(wv, bv));
                    }
                }
                touched.clear();
                epoch = epoch.wrapping_add(1);
                if epoch == 0 {
                    stamp.fill(0);
                    epoch = 1;
                }
            }
        };
    }

    'blocks: while bi < plans.len() {
        let plan = plans[bi];
        if plan.out_off >= g1 {
            break;
        }
        let blk = &enc.blocks[bi];
        let n = plan.n;

        let side = SideInfo::hoist(blk, has_affine);
        let n_sub = side.n_sub;
        let mut state = block_init_state(blk, &enc.bits, plan.start_bit, cfg, tail_biting);

        let mut reader = WordReader::new(&enc.bits, plan.start_bit);
        let mut i = 0usize;
        for sb in 0..n_sub {
            let es = side.eff[sb];
            let o = side.off[sb];
            if !have_region {
                cur_es = es;
                cur_off = o;
                have_region = true;
            } else if es != cur_es || o != cur_off {
                flush_neon!();
                cur_es = es;
                cur_off = o;
            }
            let end = (i + SUB_BLOCK).min(n);
            while i < end {
                let sym = reader.pop(k) & input_mask;
                state = ((state << k) | sym) & mask;
                let g = plan.out_off + i;
                i += 1;
                if g < g0 { continue; }
                if g >= g1 { break 'blocks; }
                let xp = xt.as_ptr().add(col * batch + b_off);
                let rp = bptr.add(state * B);
                if *stamp.get_unchecked(state) != epoch {
                    *stamp.get_unchecked_mut(state) = epoch;
                    touched.push(state as u32);
                    for l in 0..NV {
                        vst1q_f32(rp.add(l * 4), vld1q_f32(xp.add(l * 4)));
                    }
                } else {
                    for l in 0..NV {
                        let bv = vld1q_f32(rp.add(l * 4));
                        let xv = vld1q_f32(xp.add(l * 4));
                        vst1q_f32(rp.add(l * 4), vaddq_f32(bv, xv));
                    }
                }
                col += 1;
                if col == in_features {
                    flush_neon!();
                    let yo = row_rel * batch + b_off;
                    let yp = yg.as_mut_ptr().add(yo);
                    for (l, a) in acc.iter_mut().enumerate() {
                        vst1q_f32(yp.add(l * 4), *a);
                        *a = vdupq_n_f32(0.0);
                    }
                    col = 0;
                    row_rel += 1;
                }
            }
        }
        bi += 1;
    }
    debug_assert_eq!(row_rel, nrows, "group must finish exactly its rows");
    debug_assert_eq!(col, 0, "group must end on a row boundary");
    debug_assert!(touched.is_empty(), "buckets must drain at the final row end");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::{fused_gemm, fused_matvec};
    use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};

    fn synth_x(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed) * 0.0713).cos()).collect()
    }

    fn synth_xq(n: usize, seed: u64) -> Vec<i32> {
        
        (0..n)
            .map(|i| {
                let h = (i as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(seed);
                ((h >> 40) as i32 & 0x1FFF) - 4096
            })
            .collect()
    }

    #[test]
    fn histogram_dot_q12_bit_equal_reference() {
        let configs = [
            TrellisConfig::for_bpw(3.0),       
            TrellisConfig::for_bpw(2.0),       
            TrellisConfig::for_bpw(4.0),       
            TrellisConfig::for_bpw_l(2.0, 5),  
            TrellisConfig::for_bpw_l(2.0, 12), 
        ];
        let shapes = [(8usize, 256usize), (37, 300), (64, 200), (5, 512), (3, 97)];
        for cfg in &configs {
            for &(rows, cols) in &shapes {
                let n = rows * cols;
                let w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0137).sin() * 0.5).collect();
                let variants = [
                    encode_tensor(&w, cfg),
                    encode_tensor_with(&w, cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
                    encode_tensor_with(&w, cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
                    encode_tensor_with(
                        &w,
                        cfg,
                        &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
                    ),
                ];
                let xq = synth_xq(cols, 7);
                for enc in &variants {
                    let y_h = histogram_dot_q12(enc, cfg, None, rows, cols, &xq);
                    let y_r = dot_q12_reference(enc, cfg, None, rows, cols, &xq);
                    assert_eq!(
                        y_h, y_r,
                        "integer histogram != reference dot: L={} k={} rows={rows} cols={cols} tail={} affine={}",
                        cfg.l_bits, cfg.k_bits, enc.tail_biting, enc.has_affine_min
                    );
                }
            }
        }
    }

    #[test]
    fn histogram_gemm_within_tolerance_of_fused() {
        let configs = [
            TrellisConfig::for_bpw(3.0),
            TrellisConfig::for_bpw_l(2.0, 12),
            TrellisConfig::for_bpw_l(2.0, 5), 
        ];
        for cfg in &configs {
            for &(rows, cols) in &[(16usize, 256usize), (37, 300), (9, 1024)] {
                let n = rows * cols;
                let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).sin() * 0.6).collect();
                let enc = encode_tensor_with(
                    &w,
                    cfg,
                    &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
                );
                let wq = crate::decode_weights_q12(&enc, cfg, None);
                let inv = 1.0f32 / 4096.0;
                for &batch in &[1usize, 4, 16] {
                    let mut xs = Vec::with_capacity(batch * cols);
                    for b in 0..batch {
                        xs.extend(synth_x(cols, b as f32 * 3.1 + 0.2));
                    }
                    let y_h = histogram_gemm(&enc, cfg, None, rows, cols, &xs, batch);
                    let y_f = fused_gemm(&enc, cfg, None, rows, cols, &xs, batch);
                    for o in 0..rows {
                        for b in 0..batch {
                            let xb = &xs[b * cols..(b + 1) * cols];
                            let row = &wq[o * cols..(o + 1) * cols];
                            let mut absum = 0.0f32;
                            for i in 0..cols {
                                absum += ((row[i] as f32) * inv * xb[i]).abs();
                            }
                            let bound = 1e-3 * absum + 1e-4;
                            let err = (y_h[o * batch + b] - y_f[o * batch + b]).abs();
                            assert!(
                                err <= bound,
                                "histogram vs fused beyond order bound at row {o} col {b}: \
                                 err {err:.3e} > bound {bound:.3e} \
                                 (L={} k={} rows={rows} cols={cols} batch={batch})",
                                cfg.l_bits,
                                cfg.k_bits
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn histogram_columns_bit_equal_matvec() {
        let cfg = TrellisConfig::for_bpw(3.0);
        let (rows, cols) = (37usize, 300usize); 
        let n = rows * cols;
        let w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0091).sin() * 0.4).collect();
        let enc = encode_tensor_with(
            &w,
            &cfg,
            &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
        );
        for &batch in &[1usize, 3, 4, 5, 8, 16, 21, 64, 65] {
            let mut xs = Vec::with_capacity(batch * cols);
            for b in 0..batch {
                xs.extend(synth_x(cols, b as f32 * 2.3 + 0.7));
            }
            let y = histogram_gemm(&enc, &cfg, None, rows, cols, &xs, batch);
            for b in 0..batch {
                let xb = &xs[b * cols..(b + 1) * cols];
                let y1 = histogram_matvec(&enc, &cfg, None, rows, cols, xb);
                for o in 0..rows {
                    assert_eq!(
                        y[o * batch + b].to_bits(),
                        y1[o].to_bits(),
                        "gemm col {b} row {o} != matvec (batch={batch})"
                    );
                }
            }
        }
    }

    #[test]
    fn neon_bucket_add_bit_equal_scalar() {
        let configs = [TrellisConfig::for_bpw(3.0), TrellisConfig::for_bpw_l(2.0, 12)];
        for cfg in &configs {
            for &(rows, cols) in &[(16usize, 256usize), (37, 300)] {
                let n = rows * cols;
                let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0113).sin() * 0.6).collect();
                for opts in [
                    EncodeOpts::default(),
                    EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
                ] {
                    let enc = encode_tensor_with(&w, cfg, &opts);
                    for &batch in &[4usize, 5, 16, 21, 64] {
                        let mut xs = Vec::with_capacity(batch * cols);
                        for b in 0..batch {
                            xs.extend(synth_x(cols, b as f32 * 1.9 + 0.4));
                        }
                        let y_n = histogram_gemm(&enc, cfg, None, rows, cols, &xs, batch);
                        let y_s = histogram_gemm_scalar(&enc, cfg, None, rows, cols, &xs, batch);
                        for (i, (a, b)) in y_n.iter().zip(y_s.iter()).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "NEON vs scalar histogram diverged at {i}: L={} k={} batch={batch}",
                                cfg.l_bits,
                                cfg.k_bits
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fallback_delegation_matches() {
        let cfg = TrellisConfig::for_bpw_l(2.0, 8).with_vec_dim(2);
        let (rows, cols) = (6usize, 128usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.017).sin()).collect();
        let enc = encode_tensor(&w, &cfg);
        let vlut: Vec<i32> =
            (0..cfg.num_states() * cfg.vec_dim()).map(|i| ((i * 37) % 8192) as i32 - 4096).collect();
        let x = synth_x(cols, 4.0);
        let y_h = histogram_matvec(&enc, &cfg, Some(&vlut), rows, cols, &x);
        let y_f = fused_matvec(&enc, &cfg, Some(&vlut), rows, cols, &x);
        for o in 0..rows {
            assert_eq!(y_h[o].to_bits(), y_f[o].to_bits(), "float fallback row {o}");
        }
        let xq = synth_xq(cols, 11);
        assert_eq!(
            histogram_dot_q12(&enc, &cfg, Some(&vlut), rows, cols, &xq),
            dot_q12_reference(&enc, &cfg, Some(&vlut), rows, cols, &xq),
            "integer fallback"
        );
    }

    #[test]
    fn stats_sane_and_bounded() {
        let cfg = TrellisConfig::for_bpw(3.0);
        let (rows, cols) = (16usize, 512usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.031).sin() * 0.7).collect();
        let enc = encode_tensor_with(
            &w,
            &cfg,
            &EncodeOpts { affine_min: true, ..Default::default() },
        );
        let s = histogram_stats(&enc, &cfg, cols);
        assert_eq!(s.weights, (rows * cols) as u64);
        assert!(s.regions >= rows as u64, "row ends force at least one region per row");
        assert!(s.mul_ratio() <= 1.0 + 1e-12, "occupied can never exceed weights");
        assert!(s.avg_region_len() > 0.0);
    }
}
