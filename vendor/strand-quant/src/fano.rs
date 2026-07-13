use wide::{f64x4, CmpLt};

use crate::decode::eff_min_q;
use crate::encode::{
    build_sub_levels, choose_affine_min, choose_scale_q, choose_sub_scales, n_sub_blocks,
    pack_sub_scales, BlockMeta, EncodeOpts, EncodedTensor, SUB_BLOCK, SUB_SCALE_UNITY,
};
use crate::trellis::{push_bits, TrellisConfig};

const PRUNE_MARGIN: f64 = 1e-9;

#[derive(Clone, Copy, Debug)]
pub struct BlockPruneStat {
    pub n: u32,
    pub win_cost: f64,
    pub greedy_ub: f64,
    pub floor: f64,
    pub expanded: u64,
    pub total: u64,
}

#[derive(Clone, Debug, Default)]
pub struct PruneReport {
    pub blocks: Vec<BlockPruneStat>,
}

impl PruneReport {
    pub fn expansion_ratio(&self) -> f64 {
        let e: u64 = self.blocks.iter().map(|b| b.expanded).sum();
        let t: u64 = self.blocks.iter().map(|b| b.total).sum();
        if t == 0 {
            1.0
        } else {
            e as f64 / t as f64
        }
    }
}

pub fn encode_tensor_pruned(
    weights: &[f32],
    cfg: &TrellisConfig,
    opts: &EncodeOpts,
    lut: &[i32],
) -> (EncodedTensor, PruneReport) {
    assert_eq!(
        cfg.vec_dim(),
        1,
        "encode_tensor_pruned: scalar trellis only (vec_dim == 1)"
    );
    let num_states = cfg.num_states();
    let mut bits = Vec::new();
    let mut bit_cursor = 0usize;
    let mut blocks = Vec::new();
    let mut report = PruneReport::default();

    let mut back_buf: Vec<u32> = vec![u32::MAX; cfg.block_len * num_states];
    let mut sorted_levels: Vec<Vec<f64>> = Vec::new();

    for chunk in weights.chunks(cfg.block_len) {
        let scale_q = choose_scale_q(chunk, lut, cfg);
        let mults = if opts.adaptive {
            choose_sub_scales(chunk, scale_q, lut, cfg)
        } else {
            vec![SUB_SCALE_UNITY; n_sub_blocks(chunk.len())]
        };
        let (min_base_q, min_codes) = if opts.affine_min {
            choose_affine_min(chunk, scale_q, &mults, lut, cfg)
        } else {
            (0, Vec::new())
        };
        let mins_eff: Vec<i32> = min_codes
            .iter()
            .map(|&c| eff_min_q(min_base_q, c))
            .collect();
        let sub_levels = build_sub_levels(scale_q, &mults, &mins_eff, lut, num_states);

        let ub = greedy_path_cost(chunk, &sub_levels, cfg);
        let ub_eff = ub * (1.0 + PRUNE_MARGIN);

        sorted_levels.clear();
        for lv in &sub_levels {
            let mut s = lv.clone();
            s.sort_by(f64::total_cmp);
            sorted_levels.push(s);
        }
        let n = chunk.len();
        let mut floors = vec![0.0f64; n];
        for (i, &w) in chunk.iter().enumerate() {
            floors[i] = min_dist(w as f64, &sorted_levels[i / SUB_BLOCK]);
        }
        let mut rem = vec![0.0f64; n];
        let mut acc = 0.0f64;
        for i in (0..n).rev() {
            rem[i] = acc;
            acc += floors[i];
        }
        let floor_total = acc;

        let nk = n * cfg.k_bits as usize;
        let can_tail_bite = opts.tail_biting && nk >= cfg.l_bits as usize;
        let needed = n * num_states;
        if back_buf.len() < needed {
            back_buf.resize(needed, u32::MAX);
        }

        let (path, init_state, win_cost, expanded, total) = if can_tail_bite {
            let s1 = pruned_sweep(chunk, &sub_levels, cfg, None, None, ub_eff, &rem);
            let inf_rem = vec![0.0f64; n];
            let mut s2 = pruned_sweep(
                chunk,
                &sub_levels,
                cfg,
                Some((s1.terminal, &mut back_buf)),
                Some(s1.terminal),
                f64::INFINITY,
                &inf_rem,
            );
            let (path, init_state) = s2.traceback.take().expect("pass 2 records");
            (
                path,
                init_state,
                s2.start_cost,
                s1.expanded + s2.expanded,
                s1.total + s2.total,
            )
        } else {
            let mut s = pruned_sweep(
                chunk,
                &sub_levels,
                cfg,
                Some((usize::MAX, &mut back_buf)),
                None,
                ub_eff,
                &rem,
            );
            let (path, init_state) = s.traceback.take().expect("recording sweep");
            (path, init_state, s.start_cost, s.expanded, s.total)
        };

        for &sym in &path {
            push_bits(&mut bits, &mut bit_cursor, sym as usize, cfg.k_bits);
        }
        blocks.push(BlockMeta {
            scale_q,
            sub_scales: if opts.adaptive || opts.affine_min {
                pack_sub_scales(&mults)
            } else {
                Vec::new()
            },
            min_base_q,
            mins: if opts.affine_min {
                pack_sub_scales(&min_codes)
            } else {
                Vec::new()
            },
            init_state: init_state as u32,
            n: n as u32,
        });
        report.blocks.push(BlockPruneStat {
            n: n as u32,
            win_cost,
            greedy_ub: ub,
            floor: floor_total,
            expanded,
            total,
        });
    }

    (
        EncodedTensor {
            bits,
            blocks,
            total: weights.len(),
            has_rht_seed: false,
            tail_biting: opts.tail_biting,
            has_affine_min: opts.affine_min,
        },
        report,
    )
}

#[inline]
fn min_dist(target: f64, sorted: &[f64]) -> f64 {
    let p = sorted.partition_point(|&l| l < target);
    let mut best = f64::INFINITY;
    if p < sorted.len() {
        let d = target - sorted[p];
        best = d * d;
    }
    if p > 0 {
        let d = target - sorted[p - 1];
        let v = d * d;
        if v < best {
            best = v;
        }
    }
    if best.is_finite() {
        best
    } else {
        0.0
    }
}

fn greedy_path_cost(chunk: &[f32], sub_levels: &[Vec<f64>], cfg: &TrellisConfig) -> f64 {
    let mut state = 0usize;
    let mut acc = 0.0f64;
    for (i, &w) in chunk.iter().enumerate() {
        let levels = &sub_levels[i / SUB_BLOCK];
        let target = w as f64;
        let mut best_in = 0usize;
        let mut best_d = f64::INFINITY;
        for inp in 0..cfg.num_inputs() {
            let ns = cfg.next_state(state, inp);
            let diff = target - levels[ns];
            let d = diff * diff;
            if d < best_d {
                best_d = d;
                best_in = inp;
            }
        }
        state = cfg.next_state(state, best_in);
        acc += best_d;
    }
    acc
}

struct SweepResult {
    terminal: usize,
    start_cost: f64,
    traceback: Option<(Vec<u32>, usize)>,
    expanded: u64,
    total: u64,
}

fn pruned_sweep(
    weights: &[f32],
    sub_levels: &[Vec<f64>],
    cfg: &TrellisConfig,
    record: Option<(usize, &mut [u32])>,
    final_s: Option<usize>,
    ub_eff: f64,
    rem: &[f64],
) -> SweepResult {
    let n = weights.len();
    let num_states = cfg.num_states();
    let k = cfg.k_bits as usize;
    let num_inputs = cfg.num_inputs();
    let n_groups = num_states >> k;
    let inf = f64::INFINITY;

    let (pin_start, mut back): (Option<usize>, Option<&mut [u32]>) = match record {
        Some((pin, buf)) => (if pin == usize::MAX { None } else { Some(pin) }, Some(buf)),
        None => (None, None),
    };

    let mut cost: Vec<f64> = match pin_start {
        Some(s0) => {
            let mut c = vec![inf; num_states];
            c[s0] = 0.0;
            c
        }
        None => vec![0.0f64; num_states],
    };
    let mut next_cost: Vec<f64> = vec![inf; num_states];
    let mut expanded = 0u64;
    let total = (n as u64) * (num_states as u64);

    for (i, &w) in weights.iter().enumerate() {
        let target = w as f64;
        let levels = &sub_levels[i / SUB_BLOCK];
        let row: Option<&mut [u32]> = back
            .as_deref_mut()
            .map(|b| &mut b[i * num_states..(i + 1) * num_states]);
        let mut row = row;

        for g in 0..n_groups {
            let mut any = false;
            for t in 0..num_inputs {
                if cost[g + t * n_groups] < inf {
                    any = true;
                    break;
                }
            }
            let ns_base = g << k;
            if !any {
                for j in 0..num_inputs {
                    next_cost[ns_base + j] = inf;
                }

                continue;
            }
            expanded += num_inputs as u64;

            if num_inputs >= 4 {
                for off in (0..num_inputs).step_by(4) {
                    let i0 = ns_base + off;
                    let d0 = {
                        let d = target - levels[i0];
                        d * d
                    };
                    let d1 = {
                        let d = target - levels[i0 + 1];
                        d * d
                    };
                    let d2 = {
                        let d = target - levels[i0 + 2];
                        d * d
                    };
                    let d3 = {
                        let d = target - levels[i0 + 3];
                        d * d
                    };
                    let d_v = f64x4::from([d0, d1, d2, d3]);
                    let mut best_v = f64x4::splat(cost[g]) + d_v;
                    let mut best_t = f64x4::splat(0.0);
                    for t in 1..num_inputs {
                        let v = f64x4::splat(cost[g + t * n_groups]) + d_v;
                        let m = v.cmp_lt(best_v);
                        best_v = m.blend(v, best_v);
                        best_t = m.blend(f64x4::splat(t as f64), best_t);
                    }
                    next_cost[i0..i0 + 4].copy_from_slice(&best_v.to_array());
                    if let Some(r) = row.as_deref_mut() {
                        let bt = best_t.to_array();
                        for lane in 0..4 {
                            r[i0 + lane] = (g + (bt[lane] as usize) * n_groups) as u32;
                        }
                    }
                }
            } else {
                for j in 0..num_inputs {
                    let ns = ns_base + j;
                    let diff = target - levels[ns];
                    let d = diff * diff;
                    let mut best = cost[g] + d;
                    let mut best_p = g;
                    for t in 1..num_inputs {
                        let p = g + t * n_groups;
                        let v = cost[p] + d;
                        if v < best {
                            best = v;
                            best_p = p;
                        }
                    }
                    next_cost[ns] = best;
                    if let Some(r) = row.as_deref_mut() {
                        r[ns] = best_p as u32;
                    }
                }
            }
        }

        if ub_eff.is_finite() {
            let thresh = ub_eff - rem[i];
            for c in next_cost.iter_mut() {
                if *c > thresh {
                    *c = inf;
                }
            }
        }
        std::mem::swap(&mut cost, &mut next_cost);
    }

    let mut terminal = 0usize;
    let mut best = inf;
    for (s, &c) in cost.iter().enumerate() {
        if c < best {
            best = c;
            terminal = s;
        }
    }
    let start_at = final_s.unwrap_or(terminal);
    let start_cost = cost[start_at];

    let traceback = back.map(|b| {
        let input_mask = cfg.num_inputs() - 1;
        let mut path = vec![0u32; n];
        let mut s = start_at;
        for i in (0..n).rev() {
            path[i] = (s & input_mask) as u32;
            s = b[i * num_states + s] as usize;
        }
        (path, s)
    });

    SweepResult {
        terminal,
        start_cost,
        traceback,
        expanded,
        total,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FanoParams {
    pub bias_scale: f64,
    pub budget_mult: f64,
}

impl Default for FanoParams {
    fn default() -> Self {
        FanoParams {
            bias_scale: 1.0,
            budget_mult: 8.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BlockFanoStat {
    pub n: u32,
    pub pops: u64,
    pub budget_exhausted: bool,
    pub path_cost: f64,
}

#[derive(Clone, Debug, Default)]
pub struct FanoReport {
    pub blocks: Vec<BlockFanoStat>,
}

#[derive(Clone, Copy, PartialEq)]
struct Ord64(f64);
impl Eq for Ord64 {}
impl PartialOrd for Ord64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ord64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

pub fn encode_tensor_fano(
    weights: &[f32],
    cfg: &TrellisConfig,
    opts: &EncodeOpts,
    lut: &[i32],
    params: &FanoParams,
) -> (EncodedTensor, FanoReport) {
    assert_eq!(cfg.vec_dim(), 1, "encode_tensor_fano: scalar trellis only");
    assert!(
        !opts.tail_biting,
        "encode_tensor_fano: tail-biting unsupported"
    );
    let num_states = cfg.num_states();
    let num_inputs = cfg.num_inputs();
    let mut bits = Vec::new();
    let mut bit_cursor = 0usize;
    let mut blocks = Vec::new();
    let mut report = FanoReport::default();

    let mut dom_cost: Vec<f64> = Vec::new();
    let mut dom_gen: Vec<u32> = Vec::new();
    let mut gen = 0u32;

    for chunk in weights.chunks(cfg.block_len) {
        let scale_q = choose_scale_q(chunk, lut, cfg);
        let mults = if opts.adaptive {
            choose_sub_scales(chunk, scale_q, lut, cfg)
        } else {
            vec![SUB_SCALE_UNITY; n_sub_blocks(chunk.len())]
        };
        let (min_base_q, min_codes) = if opts.affine_min {
            choose_affine_min(chunk, scale_q, &mults, lut, cfg)
        } else {
            (0, Vec::new())
        };
        let mins_eff: Vec<i32> = min_codes
            .iter()
            .map(|&c| eff_min_q(min_base_q, c))
            .collect();
        let sub_levels = build_sub_levels(scale_q, &mults, &mins_eff, lut, num_states);

        let n = chunk.len();
        let greedy_ub = greedy_path_cost(chunk, &sub_levels, cfg);
        let bias = if n > 0 {
            params.bias_scale * greedy_ub / n as f64
        } else {
            0.0
        };
        let budget = ((params.budget_mult * n as f64).ceil() as u64).max(n as u64);

        let table = (n + 1) * num_states;
        if dom_cost.len() < table {
            dom_cost.resize(table, f64::INFINITY);
            dom_gen.resize(table, 0);
        }
        gen = gen.wrapping_add(1);
        if gen == 0 {
            dom_gen.iter_mut().for_each(|g| *g = 0);
            gen = 1;
        }

        let mut arena: Vec<FanoNode> = Vec::with_capacity(4 * n + 4);
        arena.push(FanoNode {
            state: 0,
            step: 0,
            sym: 0,
            parent: u32::MAX,
        });
        let mut heap: std::collections::BinaryHeap<(std::cmp::Reverse<Ord64>, u32)> =
            std::collections::BinaryHeap::new();
        heap.push((std::cmp::Reverse(Ord64(0.0)), 0));
        let mut node_cost: Vec<f64> = vec![0.0];

        let mut pops = 0u64;
        let mut done: Option<(u32, f64)> = None;
        let mut deepest: (u32, u32) = (0, 0);

        while let Some((std::cmp::Reverse(_adj), idx)) = heap.pop() {
            pops += 1;
            let (state, step) = {
                let nd = &arena[idx as usize];
                (nd.state as usize, nd.step as usize)
            };
            let c = node_cost[idx as usize];
            let key = step * num_states + state;
            let seen = dom_gen[key] == gen;
            if seen && dom_cost[key] < c {
                if pops >= budget {
                    break;
                }
                continue;
            }
            dom_gen[key] = gen;
            dom_cost[key] = c;

            if step == n {
                done = Some((idx, c));
                break;
            }
            if arena[idx as usize].step >= deepest.0 {
                deepest = (arena[idx as usize].step, idx);
            }
            let target = chunk[step] as f64;
            let levels = &sub_levels[step / SUB_BLOCK];
            for inp in 0..num_inputs {
                let ns = cfg.next_state(state, inp);
                let diff = target - levels[ns];
                let nc = c + diff * diff;
                let nkey = (step + 1) * num_states + ns;
                if dom_gen[nkey] == gen && dom_cost[nkey] <= nc {
                    continue;
                }
                let nidx = arena.len() as u32;
                arena.push(FanoNode {
                    state: ns as u32,
                    step: (step + 1) as u32,
                    sym: inp as u8,
                    parent: idx,
                });
                node_cost.push(nc);
                let adj = nc - bias * (step + 1) as f64;
                heap.push((std::cmp::Reverse(Ord64(adj)), nidx));
            }
            if pops >= budget {
                break;
            }
        }

        let budget_exhausted = done.is_none();
        let (mut path, path_cost) = match done {
            Some((idx, c)) => (walk_path(&arena, idx, n), c),
            None => {
                let (dstep, didx) = deepest;
                let mut p = walk_path(&arena, didx, dstep as usize);
                let mut state = arena[didx as usize].state as usize;
                let mut c = node_cost[didx as usize];
                for step in dstep as usize..n {
                    let target = chunk[step] as f64;
                    let levels = &sub_levels[step / SUB_BLOCK];
                    let mut best_in = 0usize;
                    let mut best_d = f64::INFINITY;
                    for inp in 0..num_inputs {
                        let ns = cfg.next_state(state, inp);
                        let diff = target - levels[ns];
                        let d = diff * diff;
                        if d < best_d {
                            best_d = d;
                            best_in = inp;
                        }
                    }
                    state = cfg.next_state(state, best_in);
                    p.push(best_in as u32);
                    c += best_d;
                }
                (p, c)
            }
        };
        debug_assert_eq!(path.len(), n);
        path.truncate(n);

        for &sym in &path {
            push_bits(&mut bits, &mut bit_cursor, sym as usize, cfg.k_bits);
        }
        blocks.push(BlockMeta {
            scale_q,
            sub_scales: if opts.adaptive || opts.affine_min {
                pack_sub_scales(&mults)
            } else {
                Vec::new()
            },
            min_base_q,
            mins: if opts.affine_min {
                pack_sub_scales(&min_codes)
            } else {
                Vec::new()
            },
            init_state: 0,
            n: n as u32,
        });
        report.blocks.push(BlockFanoStat {
            n: n as u32,
            pops,
            budget_exhausted,
            path_cost,
        });
    }

    (
        EncodedTensor {
            bits,
            blocks,
            total: weights.len(),
            has_rht_seed: false,
            tail_biting: false,
            has_affine_min: opts.affine_min,
        },
        report,
    )
}

fn walk_path(arena: &[FanoNode], idx: u32, len: usize) -> Vec<u32> {
    let mut out = vec![0u32; len];
    let mut i = idx as usize;
    let mut pos = len;
    while pos > 0 {
        pos -= 1;
        out[pos] = arena[i].sym as u32;
        i = arena[i].parent as usize;
    }
    out
}

struct FanoNode {
    state: u32,
    step: u32,
    sym: u8,
    parent: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::codebook_lut;
    use crate::decode::decode_tensor;
    use crate::encode::encode_tensor_with_lut;

    fn splitmix64(x: &mut u64) -> u64 {
        *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn normal_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        let mut out = Vec::with_capacity(n + 1);
        while out.len() < n {
            let u1 = ((splitmix64(&mut s) >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
            let u2 = (splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64;
            let r = (-2.0 * u1.ln()).sqrt();
            let th = 2.0 * std::f64::consts::PI * u2;
            out.push((r * th.cos()) as f32);
            out.push((r * th.sin()) as f32);
        }
        out.truncate(n);
        out
    }

    #[test]
    fn pruned_is_byte_identical_smoke() {
        for (k, l) in [(2u32, 8u32), (3, 8), (2, 12), (4, 8)] {
            let cfg = TrellisConfig::new(l, k, 256);
            let lut = codebook_lut(cfg.l_bits);
            for &n in &[1024usize, 257, 17] {
                let w = normal_vec(n, 0xFA90_0000 + (k as u64) << 8 | l as u64);
                for opts in [
                    EncodeOpts::default(),
                    EncodeOpts {
                        adaptive: true,
                        tail_biting: true,
                        affine_min: false,
                        silence_bonus: 0.0,
                        entropy_bonus_scale: 0.0,
                        entropy_bonus_two_pass: false,
                    },
                ] {
                    let (pruned, rep) = encode_tensor_pruned(&w, &cfg, &opts, lut);
                    let live = encode_tensor_with_lut(&w, &cfg, &opts, lut);
                    assert_eq!(pruned, live, "k={k} l={l} n={n} tb={}", opts.tail_biting);
                    assert!(rep.expansion_ratio() <= 1.0 + 1e-12);
                }
            }
        }
    }

    #[test]
    fn fano_decodes_and_is_sane() {
        let cfg = TrellisConfig::new(8, 2, 256);
        let lut = codebook_lut(cfg.l_bits);
        let w = normal_vec(2048, 0xFA90_FFFF);
        let opts = EncodeOpts::default();
        let (enc, rep) = encode_tensor_fano(&w, &cfg, &opts, lut, &FanoParams::default());
        assert_eq!(enc.total, w.len());
        let recon = decode_tensor(&enc, &cfg);
        assert_eq!(recon.len(), w.len());
        let exact = encode_tensor_with_lut(&w, &cfg, &opts, lut);
        let r_exact = rel_rms(&w, &decode_tensor(&exact, &cfg));
        let r_fano = rel_rms(&w, &recon);
        assert!(
            r_fano < r_exact * 2.0,
            "fano rel-RMS {r_fano} vs viterbi {r_exact}"
        );
        assert_eq!(rep.blocks.len(), enc.blocks.len());
    }

    fn rel_rms(w: &[f32], r: &[f32]) -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (&a, &b) in w.iter().zip(r.iter()) {
            let d = (a - b) as f64;
            num += d * d;
            den += (a as f64) * (a as f64);
        }
        (num / den.max(1e-30)).sqrt()
    }
}
