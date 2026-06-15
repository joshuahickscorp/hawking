// bit-ledger.rs — Lane A: THE BIT LEDGER / ENTROPY MICROSCOPE
//
// READ-ONLY analysis. Given a model safetensors + a quant config, this re-encodes
// every quantizable linear tensor (reusing the production encode path) and then
// decomposes the EXACT bit spend per component (payload, scale, sub-scale, init,
// outlier positions, outlier values, v2 table, padding, rht seed), and computes
// the EMPIRICAL Shannon entropy of the side-info streams to estimate how many of
// those bits a C2 entropy coder could recover.
//
// It does NOT implement C2. It only measures whether C2 is worth building, per
// the compression-map §3.3 gate.
//
// Usage:
//   bit-ledger --in <safetensors> --bits 2 --l 12 --outlier-channel 1 \
//              [--threads N] [--only <substr>] [--csv <path>] [--md <path>]
//
// Recommended: STRAND_NO_GPU=1 for deterministic CPU encode.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use strand_quant::encode::{
    encode_tensor_with, n_sub_blocks, unpack_sub_scales, EncodedTensor, EncodeOpts,
    RHT_SEED_BITS, SUB_BLOCK,
};
use strand_quant::format::BlockOffsetRecord;
use strand_quant::gate_utils::{is_quantizable_linear, rht_seed_for};
use strand_quant::outlier_wire::{idx_bits_for, OutlierWire};
use strand_quant::rht::{rht_forward_rows, RhtConfig};
use strand_quant::safetensor_io::SafeTensors;
use strand_quant::TrellisConfig;

// ---------------------------------------------------------------------------
// Tensor-class taxonomy (attention vs FFN-up/gate vs FFN-down).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum TClass {
    Attn,
    FfnUpGate,
    FfnDown,
    Other,
}

impl TClass {
    fn of(name: &str) -> TClass {
        if name.ends_with("q_proj.weight")
            || name.ends_with("k_proj.weight")
            || name.ends_with("v_proj.weight")
            || name.ends_with("o_proj.weight")
        {
            TClass::Attn
        } else if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") {
            TClass::FfnUpGate
        } else if name.ends_with("down_proj.weight") {
            TClass::FfnDown
        } else {
            TClass::Other
        }
    }
    fn label(&self) -> &'static str {
        match self {
            TClass::Attn => "attn",
            TClass::FfnUpGate => "ffn_up_gate",
            TClass::FfnDown => "ffn_down",
            TClass::Other => "other",
        }
    }
}

// ---------------------------------------------------------------------------
// Entropy accumulators: histogram-based order-0 Shannon entropy, plus
// context-by-position entropy for sub-scales and predictor-residual entropy.
// All entropies are reported in BITS PER SYMBOL.
// ---------------------------------------------------------------------------

/// Order-0 empirical Shannon entropy in bits/symbol from a count histogram.
fn entropy_bits_from_counts(counts: &HashMap<i64, u64>) -> (f64, u64) {
    let total: u64 = counts.values().sum();
    if total == 0 {
        return (0.0, 0);
    }
    let n = total as f64;
    let mut h = 0.0f64;
    for &c in counts.values() {
        if c == 0 {
            continue;
        }
        let p = c as f64 / n;
        h -= p * p.log2();
    }
    (h.max(0.0), total)
}

/// A streaming order-0 entropy collector over i64-keyed symbols.
#[derive(Default, Clone)]
struct Hist {
    counts: HashMap<i64, u64>,
}
impl Hist {
    fn push(&mut self, sym: i64) {
        *self.counts.entry(sym).or_insert(0) += 1;
    }
    fn merge(&mut self, other: &Hist) {
        for (&k, &v) in &other.counts {
            *self.counts.entry(k).or_insert(0) += v;
        }
    }
    fn entropy(&self) -> (f64, u64) {
        entropy_bits_from_counts(&self.counts)
    }
}

// ---------------------------------------------------------------------------
// Per-tensor raw bit-spend ledger (exact, matches block_side_bits + v2 + outl).
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct RawBits {
    n_weights: u64,
    n_blocks: u64,
    payload: u64,
    scale: u64,     // 32/block
    sub_scale: u64, // 6 * n_sub / block
    init_state: u64,
    affine_min: u64,
    rht_seed: u64,
    header_const: u64, // 16/tensor in v1 block_side_bits
    // v2 deploy-format overhead
    v2_table: u64, // 128 bits/block (BlockOffsetRecord = 16 B)
    // outlier side-channel (separable)
    outl_pos: u64,
    outl_val: u64,
    outl_header: u64,
}
impl RawBits {
    fn merge(&mut self, o: &RawBits) {
        self.n_weights += o.n_weights;
        self.n_blocks += o.n_blocks;
        self.payload += o.payload;
        self.scale += o.scale;
        self.sub_scale += o.sub_scale;
        self.init_state += o.init_state;
        self.affine_min += o.affine_min;
        self.rht_seed += o.rht_seed;
        self.header_const += o.header_const;
        self.v2_table += o.v2_table;
        self.outl_pos += o.outl_pos;
        self.outl_val += o.outl_val;
        self.outl_header += o.outl_header;
    }
}

// ---------------------------------------------------------------------------
// Per-tensor entropy collectors (raw bits vs entropy-floor bits per component).
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct EntColl {
    // scale_q stream (order-0, and prev-block delta residual)
    scale_raw_o0: Hist,
    scale_delta: Hist,
    // sub-scale: pooled order-0, plus 8 per-position histograms (context-by-position)
    sub_pooled: Hist,
    sub_by_pos: Vec<Hist>, // index = sub-scale position within a 256 block
    sub_super_resid: Hist, // sub - super(mode of block) residual
    // init state
    init: Hist,
    // outliers
    outl_pos: Hist,      // absolute index (order-0 — usually ~uniform)
    outl_pos_gap: Hist,  // gap between consecutive sorted indices (predictor)
    outl_val: Hist,      // residual code value
    // counts for billed-bit denominators
    n_scale: u64,
    n_sub: u64,
    n_init: u64,
    n_outl: u64,
    idx_bits: u32,
    val_bits: u32,
    l_bits: u32,
}
impl EntColl {
    fn new(l_bits: u32) -> Self {
        EntColl {
            scale_raw_o0: Hist::default(),
            scale_delta: Hist::default(),
            sub_pooled: Hist::default(),
            sub_by_pos: vec![Hist::default(); (256 / SUB_BLOCK) + 2],
            sub_super_resid: Hist::default(),
            init: Hist::default(),
            outl_pos: Hist::default(),
            outl_pos_gap: Hist::default(),
            outl_val: Hist::default(),
            n_scale: 0,
            n_sub: 0,
            n_init: 0,
            n_outl: 0,
            idx_bits: 0,
            val_bits: 0,
            l_bits,
        }
    }
    fn merge(&mut self, o: &EntColl) {
        self.scale_raw_o0.merge(&o.scale_raw_o0);
        self.scale_delta.merge(&o.scale_delta);
        self.sub_pooled.merge(&o.sub_pooled);
        for (a, b) in self.sub_by_pos.iter_mut().zip(o.sub_by_pos.iter()) {
            a.merge(b);
        }
        self.sub_super_resid.merge(&o.sub_super_resid);
        self.init.merge(&o.init);
        self.outl_pos.merge(&o.outl_pos);
        self.outl_pos_gap.merge(&o.outl_pos_gap);
        self.outl_val.merge(&o.outl_val);
        self.n_scale += o.n_scale;
        self.n_sub += o.n_sub;
        self.n_init += o.n_init;
        self.n_outl += o.n_outl;
        if o.idx_bits != 0 {
            self.idx_bits = o.idx_bits;
        }
        if o.val_bits != 0 {
            self.val_bits = o.val_bits;
        }
    }

    /// Context-by-position sub-scale entropy in bits/symbol: weighted average of
    /// per-position order-0 entropies.
    fn sub_ctx_pos_entropy(&self) -> f64 {
        let mut hsum = 0.0f64;
        let mut nsum = 0u64;
        for h in &self.sub_by_pos {
            let (e, n) = h.entropy();
            hsum += e * n as f64;
            nsum += n;
        }
        if nsum == 0 {
            0.0
        } else {
            hsum / nsum as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Per-tensor analysis.
// ---------------------------------------------------------------------------

struct TensorAnalysis {
    name: String,
    class: TClass,
    raw: RawBits,
    ent: EntColl,
}

#[allow(clippy::too_many_arguments)]
fn analyze_one(
    name: &str,
    gt: &[f32],
    in_features: usize,
    cfg: &TrellisConfig,
    use_rht: bool,
    affine_min: bool,
    tail_biting: bool,
    outlier_pct: f64,
    outlier_bits: u32,
) -> TensorAnalysis {
    let n_total = gt.len();

    // --- Outlier extraction (mirror quantize-model::quantize_one exactly) ---
    let outlier: Option<OutlierWire> = if outlier_pct > 0.0 {
        let k = ((outlier_pct / 100.0) * n_total as f64).round() as usize;
        if k == 0 || n_total == 0 {
            None
        } else {
            let mut order: Vec<usize> = (0..n_total).collect();
            order.sort_unstable_by(|&a, &b| {
                gt[b].abs().partial_cmp(&gt[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
            });
            let idx: Vec<usize> = order[..k].to_vec();
            let omax = idx.iter().fold(0f32, |m, &i| m.max(gt[i].abs())).max(1e-12);
            let ob = outlier_bits.clamp(2, 16);
            let levels = ((1i64 << (ob - 1)) - 1) as f32;
            let codes: Vec<i32> = idx.iter().map(|&i| (gt[i] / omax * levels).round() as i32).collect();
            Some(OutlierWire::from_selection(n_total, idx, codes, omax, ob))
        }
    } else {
        None
    };

    // bulk = gt with outliers zeroed (pre-RHT), then RHT — same as production.
    let bulk_gt: Vec<f32>;
    let job_gt: &[f32] = if let Some(w) = &outlier {
        let mut b = gt.to_vec();
        for &(i, _) in &w.entries {
            b[i as usize] = 0.0;
        }
        bulk_gt = b;
        &bulk_gt
    } else {
        gt
    };

    let work: Vec<f32> = if use_rht {
        let rcfg = RhtConfig::from_seed(rht_seed_for(name));
        rht_forward_rows(job_gt, &rcfg, in_features)
    } else {
        job_gt.to_vec()
    };

    let opts = EncodeOpts { adaptive: true, tail_biting, affine_min, ..EncodeOpts::default() };
    let enc: EncodedTensor = encode_tensor_with(&work, cfg, &opts);

    // --- Raw bit decomposition (exact) ---
    let mut raw = RawBits::default();
    raw.n_weights = n_total as u64;
    raw.n_blocks = enc.blocks.len() as u64;
    raw.header_const = 16; // v1 block_side_bits constant per tensor
    raw.rht_seed = if use_rht { RHT_SEED_BITS as u64 } else { 0 };

    let mut ent = EntColl::new(cfg.l_bits);

    let mut prev_scale: Option<i64> = None;
    for blk in &enc.blocks {
        let n = blk.n as usize;
        let n_sub = n_sub_blocks(n);
        let steps = cfg.num_steps(n);
        raw.payload += (steps * cfg.k_bits as usize) as u64;
        raw.scale += 32;
        raw.sub_scale += (6 * n_sub) as u64;
        if affine_min {
            raw.affine_min += (6 * n_sub) as u64;
        }
        let nk = steps * cfg.k_bits as usize;
        let tail = tail_biting && nk >= cfg.l_bits as usize;
        if !tail {
            raw.init_state += cfg.l_bits as u64;
        }
        raw.v2_table += (BlockOffsetRecord::SIZE * 8) as u64; // 128 bits/block

        // --- entropy collection ---
        let sq = blk.scale_q as i64;
        ent.scale_raw_o0.push(sq);
        if let Some(p) = prev_scale {
            ent.scale_delta.push(sq - p);
        }
        prev_scale = Some(sq);
        ent.n_scale += 1;

        // sub-scales: unpack the 6-bit codes
        let subs = unpack_sub_scales(&blk.sub_scales, n_sub);
        // super-scale = block mode (most common code) — a cheap per-block predictor
        let super_code = block_mode(&subs);
        for (pos, &c) in subs.iter().enumerate() {
            let v = c as i64;
            ent.sub_pooled.push(v);
            let pidx = pos.min(ent.sub_by_pos.len() - 1);
            ent.sub_by_pos[pidx].push(v);
            ent.sub_super_resid.push(v - super_code as i64);
            ent.n_sub += 1;
        }

        if !tail {
            ent.init.push(blk.init_state as i64);
            ent.n_init += 1;
        }
    }

    // --- Outlier raw + entropy ---
    if let Some(w) = &outlier {
        let count = w.entries.len() as u64;
        raw.outl_pos += count * w.idx_bits as u64;
        raw.outl_val += count * w.val_bits as u64;
        raw.outl_header += 96; // 12-byte per-tensor OUTL record header (count,omax,idx,val,reserved)
        ent.idx_bits = w.idx_bits;
        ent.val_bits = w.val_bits;
        ent.n_outl = count;
        let mut prev_i: Option<u32> = None;
        for &(i, code) in &w.entries {
            ent.outl_pos.push(i as i64);
            if let Some(p) = prev_i {
                ent.outl_pos_gap.push((i - p) as i64);
            } else {
                ent.outl_pos_gap.push(i as i64);
            }
            prev_i = Some(i);
            ent.outl_val.push(code as i64);
        }
    } else if outlier_pct > 0.0 {
        // tensor too small for any outlier — record idx_bits for billing context
        ent.idx_bits = idx_bits_for(n_total);
        ent.val_bits = outlier_bits.clamp(2, 16);
    }

    TensorAnalysis { name: name.to_string(), class: TClass::of(name), raw, ent }
}

fn block_mode(codes: &[u8]) -> u8 {
    let mut counts = [0u32; 64];
    for &c in codes {
        counts[(c as usize) & 63] += 1;
    }
    let mut best = 0u8;
    let mut best_c = 0u32;
    for (i, &c) in counts.iter().enumerate() {
        if c > best_c {
            best_c = c;
            best = i as u8;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Recoverable-bpw arithmetic. For a component:
//   raw_b_per_w     = raw_bits / n_weights
//   entropy_b_per_w = entropy_bits_per_symbol * n_symbols / n_weights
//   recoverable     = raw_b_per_w - entropy_b_per_w   (>= 0; C2 ceiling)
// ---------------------------------------------------------------------------

struct CompReport {
    label: String,
    raw_b_per_w: f64,
    ent_b_per_w: f64,
    recoverable_b_per_w: f64,
    entropy_bits_per_sym: f64,
    raw_bits_per_sym: f64,
    note: String,
}

fn build_component_reports(raw: &RawBits, ent: &EntColl) -> Vec<CompReport> {
    let nw = raw.n_weights.max(1) as f64;
    let mut out = Vec::new();

    // scale_q: raw is 32 bits/block. Best of {order-0, prev-block-delta}.
    {
        let (h0, n0) = ent.scale_raw_o0.entropy();
        let (hd, _nd) = ent.scale_delta.entropy();
        // delta has one fewer symbol per tensor; bill against block count for parity
        let best_h = h0.min(hd);
        let predictor = if hd < h0 { "prev-block delta" } else { "order-0" };
        let nsym = n0 as f64;
        let raw_bps = 32.0;
        let raw_bpw = raw.scale as f64 / nw;
        let ent_bpw = best_h * nsym / nw;
        out.push(CompReport {
            label: "scale".into(),
            raw_b_per_w: raw_bpw,
            ent_b_per_w: ent_bpw,
            recoverable_b_per_w: (raw_bpw - ent_bpw).max(0.0),
            entropy_bits_per_sym: best_h,
            raw_bits_per_sym: raw_bps,
            note: format!("predictor={predictor}, H0={h0:.2}, Hdelta={hd:.2}"),
        });
    }

    // sub-scales: raw is 6 bits/code. Best of {order-0 pooled, ctx-by-position, super-residual}.
    {
        let (h_pool, n) = ent.sub_pooled.entropy();
        let h_ctx = ent.sub_ctx_pos_entropy();
        let (h_super, _) = ent.sub_super_resid.entropy();
        let best_h = h_pool.min(h_ctx).min(h_super);
        let predictor = if best_h == h_ctx {
            "ctx-by-position"
        } else if best_h == h_super {
            "super-residual"
        } else {
            "order-0"
        };
        let nsym = n as f64;
        let raw_bpw = raw.sub_scale as f64 / nw;
        let ent_bpw = best_h * nsym / nw;
        out.push(CompReport {
            label: "sub_scale".into(),
            raw_b_per_w: raw_bpw,
            ent_b_per_w: ent_bpw,
            recoverable_b_per_w: (raw_bpw - ent_bpw).max(0.0),
            entropy_bits_per_sym: best_h,
            raw_bits_per_sym: 6.0,
            note: format!(
                "predictor={predictor}, H_pool={h_pool:.2}, H_ctx={h_ctx:.2}, H_super={h_super:.2}"
            ),
        });
    }

    // init state: raw is l_bits/block (when billed).
    {
        let (h, n) = ent.init.entropy();
        let nsym = n as f64;
        let raw_bpw = raw.init_state as f64 / nw;
        let ent_bpw = h * nsym / nw;
        out.push(CompReport {
            label: "init_state".into(),
            raw_b_per_w: raw_bpw,
            ent_b_per_w: ent_bpw,
            recoverable_b_per_w: (raw_bpw - ent_bpw).max(0.0),
            entropy_bits_per_sym: h,
            raw_bits_per_sym: ent.l_bits as f64,
            note: format!("L={}", ent.l_bits),
        });
    }

    // outlier positions: raw is idx_bits/entry. Best of {order-0 abs, gap predictor}.
    {
        let (h_abs, n) = ent.outl_pos.entropy();
        let (h_gap, _) = ent.outl_pos_gap.entropy();
        let best_h = h_abs.min(h_gap);
        let predictor = if h_gap < h_abs { "gap" } else { "abs order-0" };
        let nsym = n as f64;
        let raw_bpw = raw.outl_pos as f64 / nw;
        let ent_bpw = best_h * nsym / nw;
        out.push(CompReport {
            label: "outl_pos".into(),
            raw_b_per_w: raw_bpw,
            ent_b_per_w: ent_bpw,
            recoverable_b_per_w: (raw_bpw - ent_bpw).max(0.0),
            entropy_bits_per_sym: best_h,
            raw_bits_per_sym: ent.idx_bits as f64,
            note: format!("predictor={predictor}, H_abs={h_abs:.2}, H_gap={h_gap:.2}"),
        });
    }

    // outlier values: raw is val_bits/entry.
    {
        let (h, n) = ent.outl_val.entropy();
        let nsym = n as f64;
        let raw_bpw = raw.outl_val as f64 / nw;
        let ent_bpw = h * nsym / nw;
        out.push(CompReport {
            label: "outl_val".into(),
            raw_b_per_w: raw_bpw,
            ent_b_per_w: ent_bpw,
            recoverable_b_per_w: (raw_bpw - ent_bpw).max(0.0),
            entropy_bits_per_sym: h,
            raw_bits_per_sym: ent.val_bits as f64,
            note: String::new(),
        });
    }

    out
}

// ---------------------------------------------------------------------------
// CLI.
// ---------------------------------------------------------------------------

struct Args {
    input: String,
    bits: u32,
    l: u32,
    rht: bool,
    affine_mode_on: Option<bool>,
    tail_biting: bool,
    threads: usize,
    only: Option<String>,
    outlier_pct: f64,
    outlier_bits: u32,
    csv: Option<String>,
    md: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        input: String::new(),
        bits: 2,
        l: 0,
        rht: true,
        affine_mode_on: None, // auto
        tail_biting: false,
        threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
        only: None,
        outlier_pct: 0.0,
        outlier_bits: 8,
        csv: None,
        md: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--in" | "--input" => a.input = it.next().expect("--in needs a path"),
            "--bits" => a.bits = it.next().expect("--bits").parse().expect("bits int"),
            "--l" => a.l = it.next().expect("--l").parse().expect("l int"),
            "--no-rht" => a.rht = false,
            "--affine-min" => {
                a.affine_mode_on = match it.next().as_deref() {
                    Some("on") => Some(true),
                    Some("off") => Some(false),
                    Some("auto") | None => None,
                    Some(x) => panic!("--affine-min expects on|off|auto, got {x}"),
                }
            }
            "--tail-biting" => a.tail_biting = true,
            "--threads" => a.threads = it.next().expect("--threads").parse().expect("threads int"),
            "--only" => a.only = Some(it.next().expect("--only needs a substr")),
            "--outlier-channel" => {
                a.outlier_pct = it.next().expect("--outlier-channel PCT").parse().expect("f64")
            }
            "--outlier-bits" => {
                a.outlier_bits = it.next().expect("--outlier-bits N").parse().expect("int")
            }
            "--csv" => a.csv = Some(it.next().expect("--csv needs a path")),
            "--md" => a.md = Some(it.next().expect("--md needs a path")),
            "-h" | "--help" => {
                eprintln!(
                    "bit-ledger --in <safetensors> --bits <2..6> [--l N] [--no-rht] \
                     [--affine-min on|off|auto] [--tail-biting] [--threads N] [--only substr] \
                     [--outlier-channel PCT] [--outlier-bits N] [--csv path] [--md path]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown arg {other}"),
        }
    }
    assert!(!a.input.is_empty(), "--in is required");
    assert!((2..=6).contains(&a.bits), "--bits must be 2..6");
    a
}

fn fmt_bpw(x: f64) -> String {
    format!("{x:.5}")
}

fn main() {
    let args = parse_args();
    let t0 = Instant::now();

    let cfg = if args.l != 0 {
        TrellisConfig::for_bpw_l(args.bits as f64, args.l)
    } else {
        TrellisConfig::for_bpw(args.bits as f64)
    };
    let affine_min = args.affine_mode_on.unwrap_or(args.bits >= 4);
    let tail_biting = args.tail_biting || affine_min;

    eprintln!(
        "[bit-ledger] in={} bits={} L={} k={} rht={} affine_min={} tail_biting={} outlier_pct={} outlier_bits={} threads={} GPU={}",
        args.input, args.bits, cfg.l_bits, cfg.k_bits, args.rht, affine_min, tail_biting,
        args.outlier_pct, args.outlier_bits, args.threads,
        if std::env::var_os("STRAND_NO_GPU").is_some() { "OFF(NO_GPU)" } else { "auto" }
    );

    let st = SafeTensors::open(&args.input).expect("open input safetensors");

    struct Job {
        name: String,
        gt: Vec<f32>,
        in_features: usize,
    }
    let only_match = |n: &str| args.only.as_ref().map(|s| n.contains(s.as_str())).unwrap_or(true);
    let mut jobs: Vec<Job> = Vec::new();
    for name in &st.order {
        let t = &st.tensors[name];
        if is_quantizable_linear(name, &t.shape) && only_match(name) {
            let in_features = *t.shape.last().unwrap() as usize;
            jobs.push(Job { name: name.clone(), gt: st.to_f32(t), in_features });
        }
    }
    eprintln!("[bit-ledger] {} quantizable linear tensors", jobs.len());
    assert!(!jobs.is_empty(), "no quantizable tensors matched");

    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<TensorAnalysis>> = Mutex::new(Vec::new());
    let done = AtomicUsize::new(0);
    let nthreads = args.threads.max(1);
    std::thread::scope(|scope| {
        for _ in 0..nthreads {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= jobs.len() {
                    break;
                }
                let job = &jobs[i];
                let ta = analyze_one(
                    &job.name,
                    &job.gt,
                    job.in_features,
                    &cfg,
                    args.rht,
                    affine_min,
                    tail_biting,
                    args.outlier_pct,
                    args.outlier_bits,
                );
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!("[bit-ledger] [{d}/{}] {}", jobs.len(), job.name);
                results.lock().unwrap().push(ta);
            });
        }
    });

    let mut analyses = results.into_inner().unwrap();
    let order_idx: HashMap<&str, usize> =
        st.order.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
    analyses.sort_by_key(|a| order_idx[a.name.as_str()]);

    // --- Aggregate per-class and whole-model ---
    let mut class_raw: HashMap<TClass, RawBits> = HashMap::new();
    let mut class_ent: HashMap<TClass, EntColl> = HashMap::new();
    let mut model_raw = RawBits::default();
    let mut model_ent = EntColl::new(cfg.l_bits);
    for a in &analyses {
        class_raw.entry(a.class).or_default().merge(&a.raw);
        class_ent
            .entry(a.class)
            .or_insert_with(|| EntColl::new(cfg.l_bits))
            .merge(&a.ent);
        model_raw.merge(&a.raw);
        model_ent.merge(&a.ent);
    }

    // --- CSV ---
    let mut csv = String::new();
    csv.push_str("scope,name,class,n_weights,n_blocks,payload_bpw,scale_bpw,sub_scale_bpw,init_bpw,affine_bpw,rht_bpw,header_bpw,v2_table_bpw,outl_pos_bpw,outl_val_bpw,outl_hdr_bpw,total_encoded_bpw,total_deploy_bpw\n");
    let row = |scope: &str, name: &str, class: &str, r: &RawBits| -> String {
        let nw = r.n_weights.max(1) as f64;
        let payload = r.payload as f64 / nw;
        let scale = r.scale as f64 / nw;
        let sub = r.sub_scale as f64 / nw;
        let init = r.init_state as f64 / nw;
        let aff = r.affine_min as f64 / nw;
        let rht = r.rht_seed as f64 / nw;
        let hdr = r.header_const as f64 / nw;
        let v2 = r.v2_table as f64 / nw;
        let op = r.outl_pos as f64 / nw;
        let ov = r.outl_val as f64 / nw;
        let oh = r.outl_header as f64 / nw;
        // "encoded tensor" bpw = v1 encoded artifact (payload+scale+sub+init+affine+rht+hdr) + outlier channel
        let encoded = payload + scale + sub + init + aff + rht + hdr + op + ov + oh;
        // "deploy" bpw = encoded + v2 random-access table
        let deploy = encoded + v2;
        format!(
            "{scope},{name},{class},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.n_weights, r.n_blocks,
            fmt_bpw(payload), fmt_bpw(scale), fmt_bpw(sub), fmt_bpw(init), fmt_bpw(aff),
            fmt_bpw(rht), fmt_bpw(hdr), fmt_bpw(v2), fmt_bpw(op), fmt_bpw(ov), fmt_bpw(oh),
            fmt_bpw(encoded), fmt_bpw(deploy)
        )
    };
    for a in &analyses {
        csv.push_str(&row("tensor", &a.name, a.class.label(), &a.raw));
    }
    for cls in [TClass::Attn, TClass::FfnUpGate, TClass::FfnDown, TClass::Other] {
        if let Some(r) = class_raw.get(&cls) {
            csv.push_str(&row("class", cls.label(), cls.label(), r));
        }
    }
    csv.push_str(&row("model", "WHOLE_MODEL", "all", &model_raw));

    // Entropy CSV section
    csv.push_str("\nentropy_scope,component,raw_bits_per_sym,entropy_bits_per_sym,raw_bpw,entropy_bpw,recoverable_bpw,note\n");
    let push_ent = |csv: &mut String, scope: &str, raw: &RawBits, ent: &EntColl| {
        for c in build_component_reports(raw, ent) {
            csv.push_str(&format!(
                "{scope},{},{:.4},{:.4},{},{},{},{}\n",
                c.label, c.raw_bits_per_sym, c.entropy_bits_per_sym,
                fmt_bpw(c.raw_b_per_w), fmt_bpw(c.ent_b_per_w),
                fmt_bpw(c.recoverable_b_per_w),
                c.note.replace(',', ";")
            ));
        }
    };
    for cls in [TClass::Attn, TClass::FfnUpGate, TClass::FfnDown] {
        if let (Some(r), Some(e)) = (class_raw.get(&cls), class_ent.get(&cls)) {
            push_ent(&mut csv, cls.label(), r, e);
        }
    }
    push_ent(&mut csv, "WHOLE_MODEL", &model_raw, &model_ent);

    let csv_path = args.csv.clone().unwrap_or_else(|| {
        format!("bit-ledger-q{}-l{}.csv", args.bits, cfg.l_bits)
    });
    fs::write(&csv_path, &csv).expect("write CSV");
    eprintln!("[bit-ledger] wrote {csv_path}");

    // --- Summary to stdout ---
    let reports = build_component_reports(&model_raw, &model_ent);
    let nw = model_raw.n_weights.max(1) as f64;
    let total_encoded = (model_raw.payload + model_raw.scale + model_raw.sub_scale
        + model_raw.init_state + model_raw.affine_min + model_raw.rht_seed + model_raw.header_const
        + model_raw.outl_pos + model_raw.outl_val + model_raw.outl_header) as f64 / nw;
    let total_deploy = total_encoded + model_raw.v2_table as f64 / nw;

    println!("\n================ BIT LEDGER — WHOLE MODEL ================");
    println!("config: q{} L={} k={} rht={} affine_min={} tail_biting={} outlier_pct={}",
        args.bits, cfg.l_bits, cfg.k_bits, args.rht, affine_min, tail_biting, args.outlier_pct);
    println!("weights={} blocks={}", model_raw.n_weights, model_raw.n_blocks);
    println!("\n-- raw bit spend (bpw) --");
    println!("  payload      {:>9}", fmt_bpw(model_raw.payload as f64 / nw));
    println!("  scale        {:>9}", fmt_bpw(model_raw.scale as f64 / nw));
    println!("  sub_scale    {:>9}", fmt_bpw(model_raw.sub_scale as f64 / nw));
    println!("  init_state   {:>9}", fmt_bpw(model_raw.init_state as f64 / nw));
    println!("  affine_min   {:>9}", fmt_bpw(model_raw.affine_min as f64 / nw));
    println!("  rht_seed     {:>9}", fmt_bpw(model_raw.rht_seed as f64 / nw));
    println!("  header       {:>9}", fmt_bpw(model_raw.header_const as f64 / nw));
    println!("  outl_pos     {:>9}", fmt_bpw(model_raw.outl_pos as f64 / nw));
    println!("  outl_val     {:>9}", fmt_bpw(model_raw.outl_val as f64 / nw));
    println!("  outl_hdr     {:>9}", fmt_bpw(model_raw.outl_header as f64 / nw));
    println!("  v2_table     {:>9}  (deploy-only)", fmt_bpw(model_raw.v2_table as f64 / nw));
    println!("  ----");
    println!("  TOTAL encoded {:>9} bpw", fmt_bpw(total_encoded));
    println!("  TOTAL deploy  {:>9} bpw (incl v2 table)", fmt_bpw(total_deploy));

    println!("\n-- entropy microscope: recoverable bpw per component --");
    println!("  {:<12} {:>10} {:>10} {:>10} {:>12}  {}", "component", "raw_bpw", "ent_bpw", "recov_bpw", "H_bits/sym", "note");
    let mut c2_scale_sub = 0.0f64;
    for c in &reports {
        println!("  {:<12} {:>10} {:>10} {:>10} {:>12}  {}",
            c.label, fmt_bpw(c.raw_b_per_w), fmt_bpw(c.ent_b_per_w),
            fmt_bpw(c.recoverable_b_per_w), format!("{:.3}", c.entropy_bits_per_sym), c.note);
        if c.label == "scale" || c.label == "sub_scale" {
            c2_scale_sub += c.recoverable_b_per_w;
        }
    }

    // --- GATE VERDICT ---
    // map §3.3: implement C2 only if >=0.01 B/w from scale/sub-scale,
    // or >=0.04 B/w incl. stream-mode table.
    let recov_init = reports.iter().find(|c| c.label == "init_state").map(|c| c.recoverable_b_per_w).unwrap_or(0.0);
    let recov_op = reports.iter().find(|c| c.label == "outl_pos").map(|c| c.recoverable_b_per_w).unwrap_or(0.0);
    let recov_ov = reports.iter().find(|c| c.label == "outl_val").map(|c| c.recoverable_b_per_w).unwrap_or(0.0);
    let v2_bpw = model_raw.v2_table as f64 / nw;
    let c2_with_stream = c2_scale_sub + v2_bpw; // stream-mode can drop the whole v2 table

    println!("\n-- GATE (map §3.3): C2 ships iff scale/sub >= 0.01 B/w OR incl-stream-table >= 0.04 B/w --");
    println!("  recoverable scale+sub_scale   = {} B/w", fmt_bpw(c2_scale_sub));
    println!("  + stream-mode v2-table drop    = {} B/w (table itself {})", fmt_bpw(c2_with_stream), fmt_bpw(v2_bpw));
    println!("  recoverable init_state        = {} B/w", fmt_bpw(recov_init));
    println!("  recoverable outl_pos          = {} B/w", fmt_bpw(recov_op));
    println!("  recoverable outl_val          = {} B/w", fmt_bpw(recov_ov));
    let gate_scale_sub = c2_scale_sub >= 0.01;
    let gate_stream = c2_with_stream >= 0.04;
    let verdict = if gate_scale_sub || gate_stream { "CLEARS GATE" } else { "FAILS GATE" };
    println!("  VERDICT: C2 {verdict}  (scale/sub>=0.01: {gate_scale_sub}; incl-stream>=0.04: {gate_stream})");

    eprintln!("[bit-ledger] done in {:.1}s", t0.elapsed().as_secs_f64());

    if let Some(md) = &args.md {
        let mut s = String::new();
        s.push_str(&format!("# bit-ledger q{} L={}\n\n", args.bits, cfg.l_bits));
        s.push_str(&format!("weights={} blocks={}, total encoded {} bpw, total deploy {} bpw\n\n",
            model_raw.n_weights, model_raw.n_blocks, fmt_bpw(total_encoded), fmt_bpw(total_deploy)));
        s.push_str("| component | raw_bpw | ent_bpw | recoverable_bpw | H bits/sym | note |\n|---|---:|---:|---:|---:|---|\n");
        for c in &reports {
            s.push_str(&format!("| {} | {} | {} | {} | {:.3} | {} |\n",
                c.label, fmt_bpw(c.raw_b_per_w), fmt_bpw(c.ent_b_per_w),
                fmt_bpw(c.recoverable_b_per_w), c.entropy_bits_per_sym, c.note));
        }
        s.push_str(&format!("\nGATE: scale+sub recoverable = {} B/w; incl stream-table = {} B/w; verdict = {}\n",
            fmt_bpw(c2_scale_sub), fmt_bpw(c2_with_stream), verdict));
        fs::write(md, &s).expect("write md");
        eprintln!("[bit-ledger] wrote {md}");
    }
}
