use crate::block_walk::{block_init_state, exceeds_max_sub, SideInfo, WordReader};
use crate::gemv_par::decode_q12_par_with_lut;
use crate::loader::StrandModel;
use crate::paired_lut::PairTable;
use rayon::prelude::*;
use strand_quant::codebook::codebook_lut;
use strand_quant::decode::reconstruct_q;
use strand_quant::encode::{n_sub_blocks, EncodedTensor, SUB_BLOCK};
use strand_quant::TrellisConfig;

pub struct PreparedBlocks {
    out_off: Vec<usize>,

    init_state: Vec<u32>,

    sub_off: Vec<u32>,

    eff: Vec<i32>,

    off: Vec<i32>,
}

impl PreparedBlocks {
    fn build(enc: &EncodedTensor, cfg: &TrellisConfig) -> Self {
        let k = cfg.k_bits as usize;
        let nb = enc.blocks.len();
        let has_affine = enc.has_affine_min;

        let mut out_off = Vec::with_capacity(nb + 1);
        let mut sub_off = Vec::with_capacity(nb + 1);
        out_off.push(0usize);
        sub_off.push(0u32);
        let (mut pn, mut ps) = (0usize, 0u32);
        for blk in &enc.blocks {
            pn += blk.n as usize;
            ps += n_sub_blocks(blk.n as usize) as u32;
            out_off.push(pn);
            sub_off.push(ps);
        }
        debug_assert_eq!(pn, enc.total);

        let bits: &[u8] = &enc.bits;
        let init_state: Vec<u32> = enc.blocks.par_iter().enumerate().map(|(b, blk)| block_init_state(blk, bits, out_off[b] * k, cfg, enc.tail_biting) as u32).collect();

        let total_sub = ps as usize;
        let mut eff = vec![0i32; total_sub];
        let mut off = if has_affine { vec![0i32; total_sub] } else { Vec::new() };
        for (b, blk) in enc.blocks.iter().enumerate() {
            let side = SideInfo::hoist(blk, has_affine);
            let s0 = sub_off[b] as usize;
            let s1 = sub_off[b + 1] as usize;
            debug_assert_eq!(s1 - s0, side.n_sub);
            eff[s0..s1].copy_from_slice(&side.eff[..side.n_sub]);
            if has_affine {
                off[s0..s1].copy_from_slice(&side.off[..side.n_sub]);
            }
        }

        PreparedBlocks { out_off, init_state, sub_off, eff, off }
    }

    pub fn bytes(&self) -> usize {
        self.out_off.len() * std::mem::size_of::<usize>() + self.init_state.len() * 4 + self.sub_off.len() * 4 + self.eff.len() * 4 + self.off.len() * 4
    }

    #[inline(always)]
    fn n_blocks(&self) -> usize {
        self.init_state.len()
    }

    #[inline(always)]
    fn block_n(&self, b: usize) -> usize {
        self.out_off[b + 1] - self.out_off[b]
    }
}

pub struct PreparedTensor {
    enc: EncodedTensor,
    cfg: TrellisConfig,
    lut: Option<Vec<i32>>,
    fast: Option<PreparedBlocks>,
    shape: Option<(usize, usize)>,
}

impl PreparedTensor {
    pub fn new(enc: EncodedTensor, cfg: TrellisConfig) -> Self {
        Self::with_lut(enc, cfg, None)
    }

    pub fn with_lut(enc: EncodedTensor, cfg: TrellisConfig, lut: Option<Vec<i32>>) -> Self {
        let fast = if cfg.vec_dim() > 1 || exceeds_max_sub(&enc) { None } else { Some(PreparedBlocks::build(&enc, &cfg)) };
        PreparedTensor { enc, cfg, lut, fast, shape: None }
    }

    pub fn with_shape(mut self, out_features: usize, in_features: usize) -> Self {
        self.shape = Some((out_features, in_features));
        self
    }

    pub fn shape(&self) -> Option<(usize, usize)> {
        self.shape
    }

    pub fn enc(&self) -> &EncodedTensor {
        &self.enc
    }

    pub fn cfg(&self) -> &TrellisConfig {
        &self.cfg
    }

    pub fn is_fast_path(&self) -> bool {
        self.fast.is_some()
    }

    pub fn prepared_bytes(&self) -> usize {
        self.fast.as_ref().map(PreparedBlocks::bytes).unwrap_or(0) + self.lut_bytes()
    }

    pub fn prepared_bytes_per_weight(&self) -> f64 {
        if self.enc.total == 0 {
            return 0.0;
        }
        self.prepared_bytes() as f64 / self.enc.total as f64
    }

    pub fn resident_bytes(&self) -> usize {
        let blocks_bytes: usize = self.enc.blocks.iter().map(|b| std::mem::size_of_val(b) + b.sub_scales.capacity() + b.mins.capacity()).sum();
        self.enc.bits.capacity() + blocks_bytes + self.prepared_bytes()
    }

    fn lut_bytes(&self) -> usize {
        self.lut.as_ref().map(|l| l.len() * 4).unwrap_or(0)
    }

    #[inline(always)]
    fn lut_resolved(&self) -> &[i32] {
        match &self.lut {
            Some(l) => l.as_slice(),
            None => codebook_lut(self.cfg.l_bits),
        }
    }
}

pub struct PreparedModel {
    tensors: Vec<(String, PreparedTensor)>,
}

impl PreparedModel {
    pub fn from_model(model: &StrandModel) -> Result<Self, String> {
        let names: Vec<String> = model.tensor_names().map(|s| s.to_string()).collect();
        let mut tensors = Vec::with_capacity(names.len());
        for name in names {
            let hdr = model.tensor_header(&name).ok_or_else(|| format!("prepare: tensor {name:?} vanished"))?;
            let cfg = model.config_for(hdr);
            let shape = if hdr.shape.len() >= 2 { Some((hdr.shape[0] as usize, hdr.shape[1] as usize)) } else { None };
            let enc = model.encoded_tensor_checked(&name)?;
            let mut p = PreparedTensor::new(enc, cfg);
            if let Some((o, i)) = shape {
                p = p.with_shape(o, i);
            }
            tensors.push((name, p));
        }
        Ok(PreparedModel { tensors })
    }

    pub fn get(&self, name: &str) -> Option<&PreparedTensor> {
        self.tensors.iter().find(|(n, _)| n == name).map(|(_, p)| p)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &PreparedTensor)> {
        self.tensors.iter().map(|(n, p)| (n.as_str(), p))
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn prepared_bytes(&self) -> usize {
        self.tensors.iter().map(|(_, p)| p.prepared_bytes()).sum()
    }

    pub fn resident_bytes(&self) -> usize {
        self.tensors.iter().map(|(_, p)| p.resident_bytes()).sum()
    }
}

pub fn decode_q12_par_prepared_counted(p: &PreparedTensor, counts: Option<&mut [u32]>) -> Vec<i32> {
    let Some(counts) = counts else {
        return decode_q12_par_prepared(p);
    };

    let lut = p.lut_resolved();
    let Some(fb) = &p.fast else {
        let out = decode_q12_par_with_lut(&p.enc, &p.cfg, lut);
        for b in 0..p.enc.blocks.len().min(counts.len()) {
            counts[b] = counts[b].saturating_add(1);
        }
        return out;
    };

    let cfg = &p.cfg;
    let enc = &p.enc;
    let num_states = cfg.num_states();
    debug_assert_eq!(lut.len(), num_states, "scalar LUT must have num_states entries");
    let fold = SUB_BLOCK >= num_states;
    let k = cfg.k_bits;
    let mask = cfg.state_mask();
    let input_mask = cfg.num_inputs() - 1;
    let has_affine = enc.has_affine_min;

    let mut out = vec![0i32; enc.total];

    let mut slices: Vec<&mut [i32]> = Vec::with_capacity(fb.n_blocks());
    let mut rest: &mut [i32] = &mut out;
    for b in 0..fb.n_blocks() {
        let (head, tail) = rest.split_at_mut(fb.block_n(b));
        slices.push(head);
        rest = tail;
    }

    let mut folded: Vec<i32> = Vec::new();
    for (b, dst) in slices.iter_mut().enumerate() {
        replay_block(&enc.bits, lut, fb, b, k, mask, input_mask, num_states, fold, has_affine, &mut folded, dst);

        if b < counts.len() {
            counts[b] = counts[b].saturating_add(1);
        }
    }

    out
}

pub fn decode_q12_par_prepared(p: &PreparedTensor) -> Vec<i32> {
    let lut = p.lut_resolved();
    let Some(fb) = &p.fast else {
        return decode_q12_par_with_lut(&p.enc, &p.cfg, lut);
    };
    let cfg = &p.cfg;
    let enc = &p.enc;
    let num_states = cfg.num_states();
    debug_assert_eq!(lut.len(), num_states, "scalar LUT must have num_states entries");
    let fold = SUB_BLOCK >= num_states;
    let k = cfg.k_bits;
    let mask = cfg.state_mask();
    let input_mask = cfg.num_inputs() - 1;
    let has_affine = enc.has_affine_min;

    let mut out = vec![0i32; enc.total];

    let mut slices: Vec<&mut [i32]> = Vec::with_capacity(fb.n_blocks());
    let mut rest: &mut [i32] = &mut out;
    for b in 0..fb.n_blocks() {
        let (head, tail) = rest.split_at_mut(fb.block_n(b));
        slices.push(head);
        rest = tail;
    }

    slices.par_iter_mut().enumerate().for_each(|(b, dst)| {
        let mut folded: Vec<i32> = Vec::new();
        replay_block(&enc.bits, lut, fb, b, k, mask, input_mask, num_states, fold, has_affine, &mut folded, dst);
    });

    out
}

#[allow(unsafe_code, clippy::too_many_arguments)]
#[inline]
fn replay_block(
    bits: &[u8],
    lut: &[i32],
    fb: &PreparedBlocks,
    b: usize,
    k: u32,
    mask: usize,
    input_mask: usize,
    num_states: usize,
    fold: bool,
    has_affine: bool,
    folded: &mut Vec<i32>,
    dst: &mut [i32],
) {
    let n = fb.block_n(b);
    let start_bit = fb.out_off[b] * (k as usize);
    let s0 = fb.sub_off[b] as usize;
    let s1 = fb.sub_off[b + 1] as usize;
    let mut state = fb.init_state[b] as usize;
    let mut reader = WordReader::new(bits, start_bit);

    if fold {
        folded.clear();
        folded.resize((s1 - s0) * num_states, 0);
        for (j, &es) in fb.eff[s0..s1].iter().enumerate() {
            let base = j * num_states;
            for s in 0..num_states {
                folded[base + s] = reconstruct_q(es, lut[s]);
            }
        }
        let mut i = 0usize;
        for j in 0..(s1 - s0) {
            let o = if has_affine { fb.off[s0 + j] } else { 0 };
            let base = j * num_states;
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
        for sb in s0..s1 {
            let es = unsafe { *fb.eff.get_unchecked(sb) };
            let o = if has_affine { unsafe { *fb.off.get_unchecked(sb) } } else { 0 };
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

pub fn decode_q12_prepared_paired(p: &PreparedTensor, table: &PairTable) -> Vec<i32> {
    let Some(fb) = compose_fast_route(p, table) else {
        return crate::gemv::decode_q12_fast_with_lut(&p.enc, &p.cfg, p.lut_resolved());
    };
    let (k, mask, input_mask, has_affine) = compose_consts(p);
    let mut out = vec![0i32; p.enc.total];
    let mut slices = split_block_slices(fb, &mut out);
    for (b, dst) in slices.iter_mut().enumerate() {
        replay_block_paired(&p.enc.bits, table, fb, b, k, mask, input_mask, has_affine, dst);
    }
    out
}

pub fn decode_q12_prepared_paired_par(p: &PreparedTensor, table: &PairTable) -> Vec<i32> {
    let Some(fb) = compose_fast_route(p, table) else {
        return decode_q12_par_with_lut(&p.enc, &p.cfg, p.lut_resolved());
    };
    let (k, mask, input_mask, has_affine) = compose_consts(p);
    let mut out = vec![0i32; p.enc.total];
    let mut slices = split_block_slices(fb, &mut out);
    slices.par_iter_mut().enumerate().for_each(|(b, dst)| {
        replay_block_paired(&p.enc.bits, table, fb, b, k, mask, input_mask, has_affine, dst);
    });
    out
}

fn compose_fast_route<'a>(p: &'a PreparedTensor, table: &PairTable) -> Option<&'a PreparedBlocks> {
    assert_eq!((table.l_bits, table.k_bits), (p.cfg.l_bits, p.cfg.k_bits), "pair table / prepared tensor config mismatch");

    let fb = p.fast.as_ref()?;
    assert_eq!(table.lut_slice(), p.lut_resolved(), "pair table built from a different LUT than the prepared tensor resolves");
    Some(fb)
}

#[inline(always)]
fn compose_consts(p: &PreparedTensor) -> (u32, usize, usize, bool) {
    (p.cfg.k_bits, p.cfg.state_mask(), p.cfg.num_inputs() - 1, p.enc.has_affine_min)
}

fn split_block_slices<'a>(fb: &PreparedBlocks, out: &'a mut [i32]) -> Vec<&'a mut [i32]> {
    let mut slices: Vec<&mut [i32]> = Vec::with_capacity(fb.n_blocks());
    let mut rest = out;
    for b in 0..fb.n_blocks() {
        let (head, tail) = rest.split_at_mut(fb.block_n(b));
        slices.push(head);
        rest = tail;
    }
    slices
}

#[allow(unsafe_code, clippy::too_many_arguments)]
#[inline]
fn replay_block_paired(bits: &[u8], table: &PairTable, fb: &PreparedBlocks, b: usize, k: u32, mask: usize, input_mask: usize, has_affine: bool, dst: &mut [i32]) {
    let k2 = 2 * k;
    let idx_mask = table.index_mask();
    let n = fb.block_n(b);
    let start_bit = fb.out_off[b] * (k as usize);
    let s0 = fb.sub_off[b] as usize;
    let s1 = fb.sub_off[b + 1] as usize;

    let mut state = fb.init_state[b] as usize;
    let mut reader = WordReader::new(bits, start_bit);

    let entries = table.entries_slice().as_ptr();
    let lut_ptr = table.lut_slice().as_ptr();
    let eff_ptr = fb.eff.as_ptr();
    let off_ptr = fb.off.as_ptr();

    macro_rules! pair_step {
        ($i:expr, $es:expr, $o:expr) => {{
            let raw = reader.pop(k2);
            let sp = ((raw & input_mask) << k) | (raw >> k);
            let t = (state << k2) | sp;

            let e = unsafe { *entries.add(t & idx_mask) };
            state = t & mask;
            unsafe {
                *dst.get_unchecked_mut($i) = reconstruct_q($es, e.q1) + $o;
                *dst.get_unchecked_mut($i + 1) = reconstruct_q($es, e.q2) + $o;
            }
        }};
    }

    let k = k as usize;
    let mut i = 0usize;
    for sb in s0..s1 {
        let es = unsafe { *eff_ptr.add(sb) };
        let o = if has_affine { unsafe { *off_ptr.add(sb) } } else { 0 };
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
                let sym = reader.pop(k as u32) & input_mask;
                state = ((state << k) | sym) & mask;

                let q = unsafe { *lut_ptr.add(state) };
                unsafe { *dst.get_unchecked_mut(i) = reconstruct_q(es, q) + o };
                i += 1;
            }
        }
    }
}

pub fn fused_gemm_prepared(p: &PreparedTensor, out_features: usize, in_features: usize, xs: &[f32], batch: usize) -> Vec<f32> {
    fused_prepared_impl(p, out_features, in_features, xs, batch, false).0
}

pub fn fused_gemm_prepared_with_q12(p: &PreparedTensor, out_features: usize, in_features: usize, xs: &[f32], batch: usize) -> (Vec<f32>, Vec<i32>) {
    let (y, q) = fused_prepared_impl(p, out_features, in_features, xs, batch, true);
    (y, q.expect("debug path materializes q12"))
}

fn fused_prepared_impl(p: &PreparedTensor, out_features: usize, in_features: usize, xs: &[f32], batch: usize, want_q12: bool) -> (Vec<f32>, Option<Vec<i32>>) {
    assert!(batch >= 1, "batch must be >= 1");
    assert_eq!(xs.len(), batch * in_features, "xs must be batch x in_features");
    assert_eq!(p.enc.total, out_features * in_features, "encoded weight count != out_features * in_features");

    let Some(fb) = &p.fast else {
        return if want_q12 {
            let (y, q) = crate::fused::fused_gemm_with_q12(&p.enc, &p.cfg, p.lut.as_deref(), out_features, in_features, xs, batch);
            (y, Some(q))
        } else {
            (crate::fused::fused_gemm(&p.enc, &p.cfg, p.lut.as_deref(), out_features, in_features, xs, batch), None)
        };
    };

    let lut = p.lut_resolved();
    debug_assert_eq!(lut.len(), p.cfg.num_states());

    let mut xt = vec![0.0f32; xs.len()];
    for b in 0..batch {
        for i in 0..in_features {
            xt[i * batch + b] = xs[b * in_features + i];
        }
    }

    let mut y = vec![0.0f32; out_features * batch];
    let mut q12: Option<Vec<i32>> = if want_q12 { Some(vec![0i32; p.enc.total]) } else { None };

    let threads = rayon::current_num_threads().max(1);
    let rows_per_group = out_features.div_ceil(threads * 8).max(1);

    match q12.as_mut() {
        Some(q) => {
            y.par_chunks_mut(rows_per_group * batch).zip(q.par_chunks_mut(rows_per_group * in_features)).enumerate().for_each(|(gi, (yg, qg))| {
                let r0 = gi * rows_per_group;
                prep_group(p, fb, lut, in_features, batch, &xt, r0, yg, Some(qg));
            });
        }
        None => {
            y.par_chunks_mut(rows_per_group * batch).enumerate().for_each(|(gi, yg)| {
                let r0 = gi * rows_per_group;
                prep_group(p, fb, lut, in_features, batch, &xt, r0, yg, None);
            });
        }
    }

    (y, q12)
}

#[allow(clippy::too_many_arguments)]
fn prep_group(p: &PreparedTensor, fb: &PreparedBlocks, lut: &[i32], in_features: usize, batch: usize, xt: &[f32], r0: usize, yg: &mut [f32], mut qg: Option<&mut [i32]>) {
    let mut b_off = 0usize;
    while b_off < batch {
        let rem = batch - b_off;
        let q = if b_off == 0 { qg.as_deref_mut() } else { None };
        let step = if rem >= 64 {
            prep_chunk::<64, 16>(p, fb, lut, in_features, batch, b_off, xt, r0, yg, q);
            64
        } else if rem >= 16 {
            prep_chunk::<16, 4>(p, fb, lut, in_features, batch, b_off, xt, r0, yg, q);
            16
        } else if rem >= 4 {
            prep_chunk::<4, 1>(p, fb, lut, in_features, batch, b_off, xt, r0, yg, q);
            4
        } else {
            prep_rows::<1>(p, fb, lut, in_features, batch, b_off, xt, r0, yg, q);
            1
        };
        b_off += step;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn prep_chunk<const B: usize, const NV: usize>(
    p: &PreparedTensor,
    fb: &PreparedBlocks,
    lut: &[i32],
    in_features: usize,
    batch: usize,
    b_off: usize,
    xt: &[f32],
    r0: usize,
    yg: &mut [f32],
    q12: Option<&mut [i32]>,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        prep_rows_neon::<B, NV>(p, fb, lut, in_features, batch, b_off, xt, r0, yg, q12);
    }
    #[cfg(not(target_arch = "aarch64"))]
    prep_rows::<B>(p, fb, lut, in_features, batch, b_off, xt, r0, yg, q12);
}

#[allow(unsafe_code, clippy::too_many_arguments)]
fn prep_rows<const B: usize>(p: &PreparedTensor, fb: &PreparedBlocks, lut: &[i32], in_features: usize, batch: usize, b_off: usize, xt: &[f32], r0: usize, yg: &mut [f32], mut q12: Option<&mut [i32]>) {
    let cfg = &p.cfg;
    let enc = &p.enc;
    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let num_states = cfg.num_states();
    let fold = SUB_BLOCK >= num_states;
    let has_affine = enc.has_affine_min;
    let inv = 1.0f32 / 4096.0;

    let nrows = yg.len() / batch;
    let g0 = r0 * in_features;
    let g1 = g0 + nrows * in_features;
    let n_blocks = fb.n_blocks();

    let mut bi = fb.out_off[1..].partition_point(|&e| e <= g0);

    let out_off_ptr = fb.out_off.as_ptr();
    let sub_off_ptr = fb.sub_off.as_ptr();
    let init_ptr = fb.init_state.as_ptr();
    let eff_ptr = fb.eff.as_ptr();
    let off_ptr = fb.off.as_ptr();

    let mut acc = [0.0f32; B];
    let mut col = 0usize;
    let mut row_rel = 0usize;
    let mut folded: Vec<i32> = Vec::new();

    'blocks: while bi < n_blocks {
        let o0 = unsafe { *out_off_ptr.add(bi) };
        if o0 >= g1 {
            break;
        }
        let n = unsafe { *out_off_ptr.add(bi + 1) } - o0;
        let start_bit = o0 * (k as usize);
        let s0 = unsafe { *sub_off_ptr.add(bi) } as usize;
        let s1 = unsafe { *sub_off_ptr.add(bi + 1) } as usize;
        let mut state = unsafe { *init_ptr.add(bi) } as usize;

        if fold {
            folded.clear();
            folded.resize((s1 - s0) * num_states, 0);
            for (j, &es) in fb.eff[s0..s1].iter().enumerate() {
                let base = j * num_states;
                for s in 0..num_states {
                    folded[base + s] = reconstruct_q(es, lut[s]);
                }
            }
        }

        let mut reader = WordReader::new(&enc.bits, start_bit);
        let lut_ptr = lut.as_ptr();
        let mut i = 0usize;
        for sb in s0..s1 {
            let es = unsafe { *eff_ptr.add(sb) };
            let o = if has_affine { unsafe { *off_ptr.add(sb) } } else { 0 };
            let base = (sb - s0) * num_states;
            let end = (i + SUB_BLOCK).min(n);
            while i < end {
                let sym = reader.pop(k) & input_mask;
                state = ((state << k) | sym) & mask;
                let g = o0 + i;
                i += 1;
                if g < g0 {
                    continue;
                }
                if g >= g1 {
                    break 'blocks;
                }

                let q = if fold { (unsafe { *folded.get_unchecked(base + state) }) + o } else { reconstruct_q(es, unsafe { *lut_ptr.add(state) }) + o };
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
unsafe fn prep_rows_neon<const B: usize, const NV: usize>(
    p: &PreparedTensor,
    fb: &PreparedBlocks,
    lut: &[i32],
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

    let cfg = &p.cfg;
    let enc = &p.enc;
    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let num_states = cfg.num_states();
    let fold = SUB_BLOCK >= num_states;
    let has_affine = enc.has_affine_min;
    let inv = 1.0f32 / 4096.0;

    let nrows = yg.len() / batch;
    let g0 = r0 * in_features;
    let g1 = g0 + nrows * in_features;
    let n_blocks = fb.n_blocks();

    let mut bi = fb.out_off[1..].partition_point(|&e| e <= g0);

    let out_off_ptr = fb.out_off.as_ptr();
    let sub_off_ptr = fb.sub_off.as_ptr();
    let init_ptr = fb.init_state.as_ptr();
    let eff_ptr = fb.eff.as_ptr();
    let off_ptr = fb.off.as_ptr();

    let mut acc: [float32x4_t; NV] = [vdupq_n_f32(0.0); NV];
    let mut col = 0usize;
    let mut row_rel = 0usize;
    let mut folded: Vec<i32> = Vec::new();

    'blocks: while bi < n_blocks {
        let o0 = *out_off_ptr.add(bi);
        if o0 >= g1 {
            break;
        }
        let n = *out_off_ptr.add(bi + 1) - o0;
        let start_bit = o0 * (k as usize);
        let s0 = *sub_off_ptr.add(bi) as usize;
        let s1 = *sub_off_ptr.add(bi + 1) as usize;
        let mut state = *init_ptr.add(bi) as usize;

        if fold {
            folded.clear();
            folded.resize((s1 - s0) * num_states, 0);
            for (j, &es) in fb.eff[s0..s1].iter().enumerate() {
                let base = j * num_states;
                for s in 0..num_states {
                    folded[base + s] = reconstruct_q(es, lut[s]);
                }
            }
        }

        let mut reader = WordReader::new(&enc.bits, start_bit);
        let lut_ptr = lut.as_ptr();
        let mut i = 0usize;
        for sb in s0..s1 {
            let es = *eff_ptr.add(sb);
            let o = if has_affine { *off_ptr.add(sb) } else { 0 };
            let base = (sb - s0) * num_states;
            let end = (i + SUB_BLOCK).min(n);
            while i < end {
                let sym = reader.pop(k) & input_mask;
                state = ((state << k) | sym) & mask;
                let g = o0 + i;
                i += 1;
                if g < g0 {
                    continue;
                }
                if g >= g1 {
                    break 'blocks;
                }

                let q = if fold { (*folded.get_unchecked(base + state)) + o } else { reconstruct_q(es, *lut_ptr.add(state)) + o };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::{fused_gemm, fused_gemm_with_q12};
    use crate::gemv_par::decode_q12_par;
    use strand_quant::decode::decode_tensor_fixed;
    use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};

    fn synth_x(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed) * 0.0713).cos()).collect()
    }

    #[test]
    fn prepared_decode_is_bit_identical() {
        let configs = [
            TrellisConfig::for_bpw(3.0),
            TrellisConfig::for_bpw(2.0),
            TrellisConfig::for_bpw(4.0),
            TrellisConfig::for_bpw_l(2.0, 12),
            TrellisConfig::for_bpw_l(2.0, 5),
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
                    let refr = decode_tensor_fixed(enc, cfg);
                    let p = PreparedTensor::new(enc.clone(), cfg.clone());
                    assert!(p.is_fast_path());
                    assert_eq!(
                        decode_q12_par_prepared(&p),
                        refr,
                        "PREPARED decode diverged: L={} k={} n={} seed={} tail={} affine={}",
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
    fn prepared_fused_bit_equals_fused_gemm() {
        let configs = [TrellisConfig::for_bpw(3.0), TrellisConfig::for_bpw_l(2.0, 12), TrellisConfig::for_bpw_l(2.0, 5)];
        for cfg in &configs {
            for &(rows, cols) in &[(16usize, 256usize), (37, 300), (9, 1024)] {
                let n = rows * cols;
                let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0113).sin() * 0.6).collect();
                for opts in [EncodeOpts::default(), EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() }] {
                    let enc = encode_tensor_with(&w, cfg, &opts);
                    let p = PreparedTensor::new(enc.clone(), cfg.clone());
                    for &batch in &[1usize, 3, 4, 5, 16, 21, 64, 65] {
                        let mut xs = Vec::with_capacity(batch * cols);
                        for b in 0..batch {
                            xs.extend(synth_x(cols, b as f32 * 2.9 + 0.3));
                        }
                        let y_prep = fused_gemm_prepared(&p, rows, cols, &xs, batch);
                        let y_ref = fused_gemm(&enc, cfg, None, rows, cols, &xs, batch);
                        assert_eq!(y_prep.len(), y_ref.len());
                        for (i, (a, b)) in y_prep.iter().zip(y_ref.iter()).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "prepared fused diverged at flat index {i}: L={} k={} \
                                 rows={rows} cols={cols} batch={batch} tail={} affine={}",
                                cfg.l_bits,
                                cfg.k_bits,
                                enc.tail_biting,
                                enc.has_affine_min
                            );
                        }
                    }
                    for &batch in &[1usize, 4] {
                        let xs: Vec<f32> = (0..batch * cols).map(|i| ((i as f32) * 0.031).cos()).collect();
                        let (_y, q12) = fused_gemm_prepared_with_q12(&p, rows, cols, &xs, batch);
                        assert_eq!(q12, decode_q12_par(&enc, cfg), "prepared hidden Q12 diverged: L={} k={} batch={batch}", cfg.l_bits, cfg.k_bits);
                        let y_dbg = fused_gemm_prepared_with_q12(&p, rows, cols, &xs, batch).0;
                        let y_plain = fused_gemm_prepared(&p, rows, cols, &xs, batch);
                        for (a, b) in y_dbg.iter().zip(y_plain.iter()) {
                            assert_eq!(a.to_bits(), b.to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_paired_is_bit_identical() {
        let configs = [
            TrellisConfig::for_bpw(3.0),
            TrellisConfig::for_bpw(2.0),
            TrellisConfig::for_bpw(4.0),
            TrellisConfig::for_bpw_l(2.0, 12),
            TrellisConfig::for_bpw_l(2.0, 5),
            TrellisConfig::for_bpw_l(3.0, 5),
        ];
        for cfg in &configs {
            let table = PairTable::build(cfg);
            for seed in 0..32u64 {
                let n = 1 + (seed as usize * 131) % 4096;
                let w: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect();
                let variants = [
                    encode_tensor(&w, cfg),
                    encode_tensor_with(&w, cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
                    encode_tensor_with(&w, cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
                    encode_tensor_with(&w, cfg, &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() }),
                ];
                for enc in &variants {
                    let refr = decode_tensor_fixed(enc, cfg);
                    let p = PreparedTensor::new(enc.clone(), cfg.clone());
                    assert!(p.is_fast_path());
                    let ctx = format!("L={} k={} n={} seed={} tail={} affine={}", cfg.l_bits, cfg.k_bits, n, seed, enc.tail_biting, enc.has_affine_min);
                    assert_eq!(decode_q12_prepared_paired(&p, &table), refr, "compose: {ctx}");
                    assert_eq!(decode_q12_prepared_paired_par(&p, &table), refr, "compose-par: {ctx}");
                }
            }
        }
    }

    #[test]
    fn prepared_paired_fallback_and_custom_lut() {
        use strand_quant::decode::decode_tensor_fixed_with_lut;

        let vcfg = TrellisConfig::for_bpw(3.0).with_vec_dim(2);
        let (ns, d) = (vcfg.num_states(), vcfg.vec_dim());
        let vlut: Vec<i32> = (0..ns * d).map(|i| ((i as u32).wrapping_mul(2654435761) >> 20) as i32 - 2048).collect();
        let w: Vec<f32> = (0..700).map(|i| ((i as f32) * 0.011).cos() * 0.3).collect();
        let enc = encode_tensor(&w, &vcfg);
        let want = decode_tensor_fixed_with_lut(&enc, &vcfg, &vlut);
        let p = PreparedTensor::with_lut(enc, vcfg.clone(), Some(vlut));
        assert!(!p.is_fast_path());

        let scalar_table = PairTable::build(&TrellisConfig::for_bpw(3.0));
        assert_eq!(decode_q12_prepared_paired(&p, &scalar_table), want, "compose vec fallback");
        assert_eq!(decode_q12_prepared_paired_par(&p, &scalar_table), want, "compose-par vec fallback");

        let cfg = TrellisConfig::for_bpw(3.0);
        let ns = cfg.num_states();
        let lut: Vec<i32> = (0..ns).map(|i| ((i as u32).wrapping_mul(2654435761) >> 18) as i32 - 8192).collect();
        let w: Vec<f32> = (0..3001).map(|i| ((i as f32) * 0.017).cos() * 0.4).collect();
        let enc = encode_tensor(&w, &cfg);
        let want = decode_tensor_fixed_with_lut(&enc, &cfg, &lut);
        let table = PairTable::build_with_lut(&cfg, &lut);
        let p = PreparedTensor::with_lut(enc, cfg, Some(lut));
        assert!(p.is_fast_path());
        assert_eq!(decode_q12_prepared_paired(&p, &table), want, "compose custom LUT");
        assert_eq!(decode_q12_prepared_paired_par(&p, &table), want, "compose-par custom LUT");
    }

    #[test]
    #[should_panic(expected = "different LUT")]
    fn prepared_paired_rejects_mismatched_lut_table() {
        let cfg = TrellisConfig::for_bpw(3.0);
        let w: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.01).sin()).collect();
        let enc = encode_tensor(&w, &cfg);
        let p = PreparedTensor::new(enc, cfg.clone());
        let wrong: Vec<i32> = (0..cfg.num_states()).map(|i| i as i32).collect();
        let table = PairTable::build_with_lut(&cfg, &wrong);
        let _ = decode_q12_prepared_paired(&p, &table);
    }

    #[test]
    fn prepared_vector_fallback_bit_identical() {
        use strand_quant::decode::decode_tensor_fixed_with_lut;

        let cfg = TrellisConfig::for_bpw(3.0).with_vec_dim(2);
        let (ns, d) = (cfg.num_states(), cfg.vec_dim());
        let lut: Vec<i32> = (0..ns * d).map(|i| ((i as u32).wrapping_mul(2654435761) >> 20) as i32 - 2048).collect();
        let w: Vec<f32> = (0..700).map(|i| ((i as f32) * 0.011).cos() * 0.3).collect();
        let enc = encode_tensor(&w, &cfg);
        let want = decode_tensor_fixed_with_lut(&enc, &cfg, &lut);
        let p = PreparedTensor::with_lut(enc, cfg, Some(lut));
        assert!(!p.is_fast_path());
        assert_eq!(p.prepared_bytes(), p.lut_bytes(), "fallback holds no flat arrays");
        assert_eq!(decode_q12_par_prepared(&p), want, "prepared vec fallback");
    }

    #[test]
    fn loader_prepare_round_trips() {
        use std::io::Write;
        use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2};

        let (rows, cols) = (4u64, 256u64);
        let n = (rows * cols) as usize;
        let weights: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 0.7).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor_with(&weights, &cfg, &EncodeOpts { tail_biting: true, ..Default::default() });
        let shape = [rows, cols];
        let pt = PackedTensorV2 {
            base: PackedTensor { name: "w", shape: &shape, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc },
            block_len: cfg.block_len as u32,
        };
        let buf = write_strand_v2(&[pt], [0u8; 32], true).expect("write_strand_v2");
        let mut path = std::env::temp_dir();
        let uniq = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        path.push(format!("strand_prepared_{}_{uniq}.strand", std::process::id()));
        std::fs::File::create(&path).unwrap().write_all(&buf).unwrap();

        let model = StrandModel::open(&path).expect("open");
        let p = model.prepared_tensor("w").expect("prepared_tensor");
        assert_eq!(p.shape(), Some((rows as usize, cols as usize)));
        assert!(p.is_fast_path());
        assert!(p.prepared_bytes() > 0);
        assert!(p.resident_bytes() > p.prepared_bytes());
        let q_prep = decode_q12_par_prepared(&p);
        let q_cold = crate::gemv_par::decode_tensor_q12_par(&model, "w").expect("cold");
        assert_eq!(q_prep, q_cold, "prepared vs cold mmap decode");

        let pm = model.prepare().expect("prepare");
        assert_eq!(pm.len(), 1);
        assert!(!pm.is_empty());
        assert!(pm.prepared_bytes() > 0);
        assert!(pm.resident_bytes() >= pm.prepared_bytes());
        let q_pm = decode_q12_par_prepared(pm.get("w").expect("get"));
        assert_eq!(q_pm, q_cold, "PreparedModel decode vs cold");
        assert!(pm.get("missing").is_none());
        assert_eq!(pm.iter().count(), 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn prepared_bytes_match_documented_bill() {
        let cfg = TrellisConfig::for_bpw(3.0);
        let n = 256 * 64;
        let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0017).sin() * 0.5).collect();

        let enc = encode_tensor(&w, &cfg);
        let p = PreparedTensor::new(enc, cfg.clone());

        let expect = 65 * 8 + 64 * 4 + 65 * 4 + 64 * 8 * 4;
        assert_eq!(p.prepared_bytes(), expect);
        assert!((p.prepared_bytes_per_weight() - expect as f64 / n as f64).abs() < 1e-12);

        let enc_a = encode_tensor_with(&w, &cfg, &EncodeOpts { affine_min: true, ..Default::default() });
        if enc_a.has_affine_min {
            let pa = PreparedTensor::new(enc_a, cfg);
            assert_eq!(pa.prepared_bytes(), expect + 64 * 8 * 4, "affine adds off[]");
        }
    }
}
