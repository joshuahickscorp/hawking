
use crate::block_walk::{
    block_init_state, block_plans, exceeds_max_sub, BlockPlan, SideInfo, WordReader,
};
use rayon::prelude::*;
use strand_quant::codebook::codebook_lut;
use strand_quant::decode::reconstruct_q;
use strand_quant::encode::{EncodedTensor, SUB_BLOCK};
use strand_quant::TrellisConfig;

pub fn fused_matvec(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: Option<&[i32]>,
    out_features: usize,
    in_features: usize,
    x: &[f32],
) -> Vec<f32> {
    fused_gemm(enc, cfg, lut, out_features, in_features, x, 1)
}

pub fn fused_gemm(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: Option<&[i32]>,
    out_features: usize,
    in_features: usize,
    xs: &[f32],
    batch: usize,
) -> Vec<f32> {
    fused_gemm_impl(enc, cfg, lut, out_features, in_features, xs, batch, false, false).0
}

pub fn fused_gemm_scalar_mac(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: Option<&[i32]>,
    out_features: usize,
    in_features: usize,
    xs: &[f32],
    batch: usize,
) -> Vec<f32> {
    fused_gemm_impl(enc, cfg, lut, out_features, in_features, xs, batch, false, true).0
}

pub fn fused_gemm_factored(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: Option<&[i32]>,
    out_features: usize,
    in_features: usize,
    xs: &[f32],
    batch: usize,
) -> Vec<f32> {
    assert!(batch >= 1, "batch must be >= 1");
    assert_eq!(xs.len(), batch * in_features, "xs must be batch x in_features");
    assert_eq!(
        enc.total,
        out_features * in_features,
        "encoded weight count != out_features * in_features"
    );
    if cfg.vec_dim() > 1 || exceeds_max_sub(enc) {
        return fused_gemm(enc, cfg, lut, out_features, in_features, xs, batch);
    }
    let lut_resolved: &[i32] = match lut {
        Some(l) => l,
        None => codebook_lut(cfg.l_bits),
    };
    
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
                factored_chunk::<64>(enc, cfg, lut_resolved, &plans, in_features, batch, b_off, &xt, r0, yg);
                64
            } else if rem >= 16 {
                factored_chunk::<16>(enc, cfg, lut_resolved, &plans, in_features, batch, b_off, &xt, r0, yg);
                16
            } else if rem >= 4 {
                factored_chunk::<4>(enc, cfg, lut_resolved, &plans, in_features, batch, b_off, &xt, r0, yg);
                4
            } else {
                factored_chunk::<1>(enc, cfg, lut_resolved, &plans, in_features, batch, b_off, &xt, r0, yg);
                1
            };
            b_off += step;
        }
    });
    y
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn factored_chunk<const B: usize>(
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
    if enc.has_affine_min {
        fused_rows_factored::<B, true>(enc, cfg, lut, plans, in_features, batch, b_off, xt, r0, yg);
    } else {
        fused_rows_factored::<B, false>(enc, cfg, lut, plans, in_features, batch, b_off, xt, r0, yg);
    }
}

#[allow(unsafe_code, clippy::too_many_arguments)]
fn fused_rows_factored<const B: usize, const AFFINE: bool>(
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
    let has_affine = enc.has_affine_min;
    let tail_biting = enc.tail_biting;
    debug_assert_eq!(has_affine, AFFINE);
    
    let scale_to_f = 1.0f32 / (1u64 << (16 + 12)) as f32;
    let off_to_f = 1.0f32 / 4096.0;

    let nrows = yg.len() / batch;
    let g0 = r0 * in_features;
    let g1 = g0 + nrows * in_features;

    let mut bi = plans.partition_point(|p| p.out_off + p.n <= g0);

    let mut acc = [0.0f32; B];
    let mut sacc = [0.0f32; B]; 
    let mut xsum = [0.0f32; B]; 
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
        let lut_ptr = lut.as_ptr();
        let mut i = 0usize;
        for sb in 0..n_sub {
            let es_f = (side.eff[sb] as f32) * scale_to_f;
            let off_f = (side.off[sb] as f32) * off_to_f;
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
                
                let q_raw = unsafe { *lut_ptr.add(state) } as f32;
                let xb = &xt[col * batch + b_off..col * batch + b_off + B];
                for ((s, xs_a), &xv) in sacc.iter_mut().zip(xsum.iter_mut()).zip(xb.iter()) {
                    *s += q_raw * xv;
                    if AFFINE {
                        *xs_a += xv;
                    }
                }
                col += 1;
                if col == in_features {
                    flush_factored::<B, AFFINE>(&mut acc, &mut sacc, &mut xsum, es_f, off_f);
                    let yo = row_rel * batch + b_off;
                    yg[yo..yo + B].copy_from_slice(&acc);
                    acc = [0.0f32; B];
                    col = 0;
                    row_rel += 1;
                }
            }
            flush_factored::<B, AFFINE>(&mut acc, &mut sacc, &mut xsum, es_f, off_f);
        }
        bi += 1;
    }
    debug_assert_eq!(row_rel, nrows, "group must finish exactly its rows");
    debug_assert_eq!(col, 0, "group must end on a row boundary");
}

#[inline(always)]
fn flush_factored<const B: usize, const AFFINE: bool>(
    acc: &mut [f32; B],
    sacc: &mut [f32; B],
    xsum: &mut [f32; B],
    es_f: f32,
    off_f: f32,
) {
    for b in 0..B {
        acc[b] += es_f * sacc[b];
        if AFFINE {
            acc[b] += off_f * xsum[b];
        }
        sacc[b] = 0.0;
        if AFFINE {
            xsum[b] = 0.0;
        }
    }
}

pub fn fused_gemm_with_q12(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: Option<&[i32]>,
    out_features: usize,
    in_features: usize,
    xs: &[f32],
    batch: usize,
) -> (Vec<f32>, Vec<i32>) {
    let (y, q) = fused_gemm_impl(enc, cfg, lut, out_features, in_features, xs, batch, true, false);
    (y, q.expect("debug path materializes q12"))
}

#[allow(clippy::too_many_arguments)]
fn fused_gemm_impl(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: Option<&[i32]>,
    out_features: usize,
    in_features: usize,
    xs: &[f32],
    batch: usize,
    want_q12: bool,
    force_scalar_mac: bool,
) -> (Vec<f32>, Option<Vec<i32>>) {
    assert!(batch >= 1, "batch must be >= 1");
    assert_eq!(xs.len(), batch * in_features, "xs must be batch x in_features");
    assert_eq!(
        enc.total,
        out_features * in_features,
        "encoded weight count != out_features * in_features"
    );

    if cfg.vec_dim() > 1 || exceeds_max_sub(enc) {
        return fused_fallback(enc, cfg, lut, out_features, in_features, xs, batch, want_q12);
    }

    let lut_resolved: &[i32] = match lut {
        Some(l) => l,
        None => codebook_lut(cfg.l_bits),
    };
    debug_assert_eq!(lut_resolved.len(), cfg.num_states(), "scalar LUT must have num_states entries");

    let mut xt = vec![0.0f32; xs.len()];
    for b in 0..batch {
        for i in 0..in_features {
            xt[i * batch + b] = xs[b * in_features + i];
        }
    }

    let plans = block_plans(enc, cfg.k_bits as usize);
    let mut y = vec![0.0f32; out_features * batch];
    let mut q12: Option<Vec<i32>> = if want_q12 { Some(vec![0i32; enc.total]) } else { None };

    let threads = rayon::current_num_threads().max(1);
    let rows_per_group = out_features.div_ceil(threads * 8).max(1);

    match q12.as_mut() {
        Some(q) => {
            y.par_chunks_mut(rows_per_group * batch)
                .zip(q.par_chunks_mut(rows_per_group * in_features))
                .enumerate()
                .for_each(|(gi, (yg, qg))| {
                    let r0 = gi * rows_per_group;
                    fused_group(
                        enc, cfg, lut_resolved, &plans, in_features, batch, &xt, r0, yg,
                        Some(qg), force_scalar_mac,
                    );
                });
        }
        None => {
            y.par_chunks_mut(rows_per_group * batch).enumerate().for_each(|(gi, yg)| {
                let r0 = gi * rows_per_group;
                fused_group(
                    enc, cfg, lut_resolved, &plans, in_features, batch, &xt, r0, yg, None,
                    force_scalar_mac,
                );
            });
        }
    }

    (y, q12)
}

#[allow(clippy::too_many_arguments)]
fn fused_group(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: &[i32],
    plans: &[BlockPlan],
    in_features: usize,
    batch: usize,
    xt: &[f32],
    r0: usize,
    yg: &mut [f32],
    mut qg: Option<&mut [i32]>,
    force_scalar_mac: bool,
) {
    let mut b_off = 0usize;
    while b_off < batch {
        let rem = batch - b_off;
        let q = if b_off == 0 { qg.as_deref_mut() } else { None };
        let step = if rem >= 64 {
            rows_chunk::<64, 16>(
                enc, cfg, lut, plans, in_features, batch, b_off, xt, r0, yg, q,
                force_scalar_mac,
            );
            64
        } else if rem >= 16 {
            rows_chunk::<16, 4>(
                enc, cfg, lut, plans, in_features, batch, b_off, xt, r0, yg, q,
                force_scalar_mac,
            );
            16
        } else if rem >= 4 {
            rows_chunk::<4, 1>(
                enc, cfg, lut, plans, in_features, batch, b_off, xt, r0, yg, q,
                force_scalar_mac,
            );
            4
        } else {
            fused_rows::<1>(enc, cfg, lut, plans, in_features, batch, b_off, xt, r0, yg, q);
            1
        };
        b_off += step;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn rows_chunk<const B: usize, const NV: usize>(
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
    q12: Option<&mut [i32]>,
    force_scalar_mac: bool,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if !force_scalar_mac {
            unsafe {
                
                fused_rows_neon::<B, NV>(
                    enc, cfg, lut, plans, in_features, batch, b_off, xt, r0, yg, q12,
                );
            }
            return;
        }
    }
    let _ = force_scalar_mac;
    fused_rows::<B>(enc, cfg, lut, plans, in_features, batch, b_off, xt, r0, yg, q12);
}

#[allow(unsafe_code, clippy::too_many_arguments)]
fn fused_rows<const B: usize>(
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
    mut q12: Option<&mut [i32]>,
) {
    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let num_states = cfg.num_states();
    let fold = SUB_BLOCK >= num_states; 
    let has_affine = enc.has_affine_min;
    let tail_biting = enc.tail_biting;
    let inv = 1.0f32 / 4096.0;

    let nrows = yg.len() / batch;
    let g0 = r0 * in_features;
    let g1 = g0 + nrows * in_features;

    let mut bi = plans.partition_point(|p| p.out_off + p.n <= g0);

    let mut acc = [0.0f32; B];
    let mut col = 0usize;
    let mut row_rel = 0usize;
    let mut folded: Vec<i32> = Vec::new();

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

        if fold {
            folded.clear();
            folded.resize(n_sub * num_states, 0);
            for (sb, &es) in side.eff().iter().enumerate() {
                let base = sb * num_states;
                for s in 0..num_states {
                    folded[base + s] = reconstruct_q(es, lut[s]);
                }
            }
        }

        let mut reader = WordReader::new(&enc.bits, plan.start_bit);
        let lut_ptr = lut.as_ptr();
        let mut i = 0usize;
        for sb in 0..n_sub {
            let es = side.eff[sb];
            let o = side.off[sb];
            let base = sb * num_states;
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
                
                let q = if fold {
                    (unsafe { *folded.get_unchecked(base + state) }) + o
                } else {
                    reconstruct_q(es, unsafe { *lut_ptr.add(state) }) + o
                };
                if let Some(qd) = q12.as_deref_mut() {
                    qd[g - g0] = q;
                }
                let wf = (q as f32) * inv;
                let xb = &xt[col * batch + b_off..col * batch + b_off + B];
                for (a, &xv) in acc.iter_mut().zip(xb.iter()) {
                    *a += wf * xv;
                }
                col += 1;
                if col == in_features {
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
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
unsafe fn fused_rows_neon<const B: usize, const NV: usize>(
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
    mut q12: Option<&mut [i32]>,
) {
    use core::arch::aarch64::*;
    debug_assert_eq!(NV * 4, B, "NV must be B/4");

    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let num_states = cfg.num_states();
    let fold = SUB_BLOCK >= num_states;
    let has_affine = enc.has_affine_min;
    let tail_biting = enc.tail_biting;
    let inv = 1.0f32 / 4096.0;

    let nrows = yg.len() / batch;
    let g0 = r0 * in_features;
    let g1 = g0 + nrows * in_features;

    let mut bi = plans.partition_point(|p| p.out_off + p.n <= g0);

    let mut acc: [float32x4_t; NV] = [vdupq_n_f32(0.0); NV];
    let mut col = 0usize;
    let mut row_rel = 0usize;
    let mut folded: Vec<i32> = Vec::new();

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

        if fold {
            folded.clear();
            folded.resize(n_sub * num_states, 0);
            for (sb, &es) in side.eff().iter().enumerate() {
                let base = sb * num_states;
                for s in 0..num_states {
                    folded[base + s] = reconstruct_q(es, lut[s]);
                }
            }
        }

        let mut reader = WordReader::new(&enc.bits, plan.start_bit);
        let lut_ptr = lut.as_ptr();
        let mut i = 0usize;
        for sb in 0..n_sub {
            let es = side.eff[sb];
            let o = side.off[sb];
            let base = sb * num_states;
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
                
                let q = if fold {
                    (*folded.get_unchecked(base + state)) + o
                } else {
                    reconstruct_q(es, *lut_ptr.add(state)) + o
                };
                if let Some(qd) = q12.as_deref_mut() {
                    qd[g - g0] = q;
                }
                let wv = vdupq_n_f32((q as f32) * inv);
                let xp = xt.as_ptr().add(col * batch + b_off);
                for (v, a) in acc.iter_mut().enumerate() {
                    let xv = vld1q_f32(xp.add(v * 4));
                    #[cfg(not(feature = "neon-fma"))]
                    {
                        *a = vaddq_f32(*a, vmulq_f32(wv, xv));
                    }
                    #[cfg(feature = "neon-fma")]
                    {
                        *a = vfmaq_f32(*a, wv, xv);
                    }
                }
                col += 1;
                if col == in_features {
                    let yo = row_rel * batch + b_off;
                    let yp = yg.as_mut_ptr().add(yo);
                    for (v, a) in acc.iter_mut().enumerate() {
                        vst1q_f32(yp.add(v * 4), *a);
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
}

#[allow(clippy::too_many_arguments)]
fn fused_fallback(
    enc: &EncodedTensor,
    cfg: &TrellisConfig,
    lut: Option<&[i32]>,
    out_features: usize,
    in_features: usize,
    xs: &[f32],
    batch: usize,
    want_q12: bool,
) -> (Vec<f32>, Option<Vec<i32>>) {
    let w = crate::decode_weights_q12(enc, cfg, lut);
    let inv = 1.0f32 / 4096.0;
    let mut y = vec![0.0f32; out_features * batch];
    for o in 0..out_features {
        let row = &w[o * in_features..(o + 1) * in_features];
        let mut accs = vec![0.0f32; batch];
        for (i, &q) in row.iter().enumerate() {
            let wf = (q as f32) * inv;
            for (a, b) in accs.iter_mut().zip(0..batch) {
                *a += wf * xs[b * in_features + i];
            }
        }
        y[o * batch..(o + 1) * batch].copy_from_slice(&accs);
    }
    let q12 = if want_q12 { Some(w) } else { None };
    (y, q12)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemv_par::decode_q12_par;
    use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};

    fn synth_x(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed) * 0.0713).cos()).collect()
    }

    #[test]
    fn fused_matvec_bit_equals_reference_matvec() {
        let configs = [
            TrellisConfig::for_bpw(3.0),        
            TrellisConfig::for_bpw(2.0),        
            TrellisConfig::for_bpw(4.0),        
            TrellisConfig::for_bpw_l(2.0, 5),   
            TrellisConfig::for_bpw_l(3.0, 5),   
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
                let x = synth_x(cols, 1.0);
                for enc in &variants {
                    let y_fused = fused_matvec(enc, cfg, None, rows, cols, &x);
                    let y_ref = crate::matvec(enc, cfg, None, rows, cols, &x);
                    assert_eq!(y_fused.len(), y_ref.len());
                    for o in 0..rows {
                        assert_eq!(
                            y_fused[o].to_bits(),
                            y_ref[o].to_bits(),
                            "y[{o}] fused vs reference: L={} k={} rows={rows} cols={cols} tail={} affine={}",
                            cfg.l_bits, cfg.k_bits, enc.tail_biting, enc.has_affine_min
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fused_hidden_q12_byte_identical_to_decode_q12_par() {
        let configs = [
            TrellisConfig::for_bpw(3.0),
            TrellisConfig::for_bpw_l(2.0, 12), 
            TrellisConfig::for_bpw_l(2.0, 5),  
        ];
        for cfg in &configs {
            for &(rows, cols) in &[(16usize, 256usize), (37, 300), (9, 1024)] {
                let n = rows * cols;
                let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).sin() * 0.6).collect();
                for opts in [
                    EncodeOpts::default(),
                    EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
                ] {
                    let enc = encode_tensor_with(&w, cfg, &opts);
                    let x = synth_x(cols, 2.0);
                    let (_y, q12) = fused_gemm_with_q12(&enc, cfg, None, rows, cols, &x, 1);
                    let q_ref = decode_q12_par(&enc, cfg);
                    assert_eq!(
                        q12, q_ref,
                        "hidden Q12 diverged: L={} k={} rows={rows} cols={cols}",
                        cfg.l_bits, cfg.k_bits
                    );
                }
            }
        }
    }

    #[cfg(not(feature = "neon-fma"))]
    #[test]
    fn fused_gemm_columns_bit_equal_fused_matvec() {
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
                xs.extend(synth_x(cols, b as f32 * 3.7 + 0.5));
            }
            let y = fused_gemm(&enc, &cfg, None, rows, cols, &xs, batch);
            assert_eq!(y.len(), rows * batch);
            for b in 0..batch {
                let xb = &xs[b * cols..(b + 1) * cols];
                let y1 = fused_matvec(&enc, &cfg, None, rows, cols, xb);
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

    #[cfg(not(feature = "neon-fma"))]
    #[test]
    fn g2b_neon_mac_bit_equal_scalar_mac() {
        let configs = [
            TrellisConfig::for_bpw(3.0),       
            TrellisConfig::for_bpw_l(2.0, 12), 
            TrellisConfig::for_bpw_l(2.0, 5),  
        ];
        for cfg in &configs {
            for &(rows, cols) in &[(16usize, 256usize), (37, 300), (9, 1024)] {
                let n = rows * cols;
                let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0113).sin() * 0.6).collect();
                for opts in [
                    EncodeOpts::default(),
                    EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
                ] {
                    let enc = encode_tensor_with(&w, cfg, &opts);
                    for &batch in &[4usize, 5, 16, 21, 64, 65] {
                        let mut xs = Vec::with_capacity(batch * cols);
                        for b in 0..batch {
                            xs.extend(synth_x(cols, b as f32 * 2.9 + 0.3));
                        }
                        let y_neon = fused_gemm(&enc, cfg, None, rows, cols, &xs, batch);
                        let y_scalar =
                            fused_gemm_scalar_mac(&enc, cfg, None, rows, cols, &xs, batch);
                        assert_eq!(y_neon.len(), y_scalar.len());
                        for (i, (a, b)) in y_neon.iter().zip(y_scalar.iter()).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "G2b NEON vs scalar MAC diverged at flat index {i}: L={} k={} \
                                 rows={rows} cols={cols} batch={batch} tail={} affine={}",
                                cfg.l_bits, cfg.k_bits, enc.tail_biting, enc.has_affine_min
                            );
                        }
                    }
                    let xs4: Vec<f32> =
                        (0..4 * cols).map(|i| ((i as f32) * 0.031).cos()).collect();
                    let (_y, q12) = fused_gemm_with_q12(&enc, cfg, None, rows, cols, &xs4, 4);
                    assert_eq!(
                        q12,
                        decode_q12_par(&enc, cfg),
                        "G2b hidden Q12 diverged: L={} k={}",
                        cfg.l_bits,
                        cfg.k_bits
                    );
                }
            }
        }
    }

    #[test]
    fn fused_fallback_matches_reference() {
        let cfg = TrellisConfig::for_bpw_l(2.0, 8).with_vec_dim(2); 
        let (rows, cols) = (6usize, 128usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.017).sin()).collect();
        let enc = encode_tensor(&w, &cfg);
        let x = synth_x(cols, 4.0);
        let vlut: Vec<i32> =
            (0..cfg.num_states() * cfg.vec_dim()).map(|i| ((i * 37) % 8192) as i32 - 4096).collect();
        let y_fused = fused_matvec(&enc, &cfg, Some(&vlut), rows, cols, &x);
        let y_ref = crate::matvec(&enc, &cfg, Some(&vlut), rows, cols, &x);
        for o in 0..rows {
            assert_eq!(y_fused[o].to_bits(), y_ref[o].to_bits(), "fallback row {o}");
        }
        let (_y, q12) = fused_gemm_with_q12(&enc, &cfg, Some(&vlut), rows, cols, &x, 1);
        assert_eq!(q12, crate::decode_weights_q12(&enc, &cfg, Some(&vlut)), "fallback q12");
    }
}
