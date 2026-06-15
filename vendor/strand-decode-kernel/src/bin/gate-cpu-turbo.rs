// gate-cpu-turbo — Wave-4 speed bet 3: CPU Turbo Mode.
//
// docs/STRAND-speed-moonshot-research.md (CPU addendum) proposes a dual-mode CPU plan:
//
//   StrandCpuPlan::Compressed   — .strand mmap -> lean INTEGER decode each token
//                                 (decode_q12_fast). Tiny footprint (~0.4175 B/w at q3),
//                                 bit-identical integer decode. This is the SHIP mode.
//   StrandCpuPlan::TurboPanels  — decode ONCE at load into packed i16 Q12 panels, then a
//                                 fast i16->f32 GEMV each token. Trades RAM (~2.0 B/w) for
//                                 throughput on RAM-rich machines.
//
// Gate requirements (from the doc):
//   * compressed mode stays bit-identical (decode_q12_fast == decode_lean), and turbo's
//     packed Q12 panel equals decode_q12_fast (the cache is exact, not lossy);
//   * turbo output equals compressed output within the float-MAC profile;
//   * report B/weight (compressed ~0.4175 vs turbo ~2.0) and tokens/sec both modes,
//     MLP-only vs full per-token cache.
//
// This is a load-time/runtime PLAN prototype: it does not change the archive format or
// the deterministic decode. Turbo panels are derived deterministically from the same
// integer Q12 decode, so they can be hashed against Q12 block roots.
//
// Run:  cargo run -p strand-decode-kernel --release --bin gate-cpu-turbo

use std::time::Instant;

use strand_decode_kernel::gemv::decode_q12_fast;
use strand_quant::decode::decode_lean;
use strand_quant::encode::{encode_tensor, EncodedTensor};
use strand_quant::TrellisConfig;

const Q4K_BYTES_PER_WEIGHT: f64 = 0.5625;
const INV_Q12: f32 = 1.0 / 4096.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StrandCpuPlan {
    /// Decode each token from the compressed payload (deployed, bit-identical, tiny).
    Compressed,
    /// Decode once at load to packed i16 Q12 panels; fast i16->f32 GEMV each token.
    TurboPanels,
}

/// A tensor staged for one of the two CPU plans. Compressed keeps the EncodedTensor;
/// Turbo keeps a row-major i16 Q12 panel decoded once at load.
struct PlanTensor {
    rows: usize,
    cols: usize,
    enc: EncodedTensor,
    cfg: TrellisConfig,
    // Turbo: row-major Q12 as i16 (exact when |Q12| <= 32767; asserted at build).
    panel_i16: Vec<i16>,
}

impl PlanTensor {
    fn build(weights: &[f32], rows: usize, cols: usize, cfg: TrellisConfig) -> Self {
        let enc = encode_tensor(weights, &cfg);
        // Decode once for the turbo panel (this is the load-time decode the plan amortizes).
        let q12 = decode_q12_fast(&enc, &cfg);
        let mut over = 0usize;
        let panel_i16: Vec<i16> = q12
            .iter()
            .map(|&q| {
                if q > i16::MAX as i32 || q < i16::MIN as i32 {
                    over += 1;
                    q.clamp(i16::MIN as i32, i16::MAX as i32) as i16
                } else {
                    q as i16
                }
            })
            .collect();
        assert_eq!(
            over, 0,
            "turbo i16 panel: {over} Q12 values exceed i16 range — this tensor needs i32 panels"
        );
        Self { rows, cols, enc, cfg, panel_i16 }
    }

    fn weights_bytes_compressed(&self) -> f64 {
        // payload bits / 8 ; the deployed compressed footprint.
        self.enc.total_bpw(&self.cfg) / 8.0 * (self.rows * self.cols) as f64
    }
    fn weights_bytes_turbo(&self) -> f64 {
        // i16 panel = 2 bytes/weight resident.
        2.0 * (self.rows * self.cols) as f64
    }

    /// Compressed-mode GEMV: decode payload to Q12 this token, then MAC.
    fn gemv_compressed(&self, x: &[f32], y: &mut [f32]) {
        let w = decode_q12_fast(&self.enc, &self.cfg);
        for o in 0..self.rows {
            let row = &w[o * self.cols..(o + 1) * self.cols];
            let mut acc = 0.0f32;
            for i in 0..self.cols {
                acc += (row[i] as f32) * INV_Q12 * x[i];
            }
            y[o] = acc;
        }
    }

    /// Turbo-mode GEMV: the panel is already decoded; just MAC the i16 row.
    fn gemv_turbo(&self, x: &[f32], y: &mut [f32]) {
        for o in 0..self.rows {
            let row = &self.panel_i16[o * self.cols..(o + 1) * self.cols];
            let mut acc = 0.0f32;
            for i in 0..self.cols {
                acc += (row[i] as f32) * INV_Q12 * x[i];
            }
            y[o] = acc;
        }
    }
}

fn run_token(plan: StrandCpuPlan, tensors: &[PlanTensor], xs: &[Vec<f32>], ys: &mut [Vec<f32>]) {
    for (t, (x, y)) in tensors.iter().zip(xs.iter().zip(ys.iter_mut())) {
        match plan {
            StrandCpuPlan::Compressed => t.gemv_compressed(x, y),
            StrandCpuPlan::TurboPanels => t.gemv_turbo(x, y),
        }
    }
}

fn build_layer(cfg: &TrellisConfig, mlp_only: bool) -> Vec<PlanTensor> {
    // Qwen2.5-0.5B per-layer projection shapes (out, in); cols rounded to 256 (kernel rule).
    fn r256(n: usize) -> usize {
        n.div_ceil(256) * 256
    }
    let attn: [(usize, usize); 4] =
        [(896, r256(896)), (128, r256(896)), (128, r256(896)), (896, r256(896))];
    let mlp: [(usize, usize); 3] =
        [(4864, r256(896)), (4864, r256(896)), (896, r256(4864))];
    let mut shapes: Vec<(usize, usize)> = Vec::new();
    if !mlp_only {
        shapes.extend_from_slice(&attn);
    }
    shapes.extend_from_slice(&mlp);
    shapes
        .into_iter()
        .map(|(rows, cols)| {
            let n = rows * cols;
            let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0007).sin() * 0.3).collect();
            PlanTensor::build(&w, rows, cols, *cfg)
        })
        .collect()
}

fn bench(label: &str, cfg: &TrellisConfig, mlp_only: bool) {
    const LAYERS: usize = 24;
    let layer = build_layer(cfg, mlp_only);
    // Replicate the per-layer tensor set across 24 layers for a full token pass.
    let mut tensors: Vec<PlanTensor> = Vec::new();
    for _ in 0..LAYERS {
        for t in &layer {
            let n = t.rows * t.cols;
            let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0007).sin() * 0.3).collect();
            tensors.push(PlanTensor::build(&w, t.rows, t.cols, t.cfg));
        }
    }
    let n_tensors = tensors.len();
    let total_w: usize = tensors.iter().map(|t| t.rows * t.cols).sum();

    // ---- correctness ------------------------------------------------------------------
    // compressed decode is bit-identical, and the turbo panel is the exact same Q12.
    for t in &tensors {
        let fast = decode_q12_fast(&t.enc, &t.cfg);
        let lean = decode_lean(&t.enc, &t.cfg);
        assert_eq!(fast, lean, "compressed decode != decode_lean (bit-identity broken)");
        // panel i16 == fast Q12 (exact)
        for (a, b) in t.panel_i16.iter().zip(&fast) {
            debug_assert_eq!(*a as i32, *b, "turbo panel != decode_q12_fast");
        }
    }

    let xs: Vec<Vec<f32>> =
        tensors.iter().map(|t| (0..t.cols).map(|i| ((i as f32) * 0.01).cos()).collect()).collect();
    let mut ys: Vec<Vec<f32>> = tensors.iter().map(|t| vec![0.0f32; t.rows]).collect();
    let mut ys2: Vec<Vec<f32>> = tensors.iter().map(|t| vec![0.0f32; t.rows]).collect();

    // turbo output within float-MAC tolerance of compressed (same Q12, same MAC order)
    run_token(StrandCpuPlan::Compressed, &tensors, &xs, &mut ys);
    run_token(StrandCpuPlan::TurboPanels, &tensors, &xs, &mut ys2);
    let mut max_rel = 0.0f64;
    for (a, b) in ys.iter().zip(&ys2) {
        for (&ra, &rb) in a.iter().zip(b) {
            let denom = (ra.abs().max(rb.abs()) as f64).max(1e-3);
            max_rel = max_rel.max(((ra - rb).abs() as f64) / denom);
        }
    }
    assert!(max_rel < 1e-5, "turbo output diverged from compressed (max rel {max_rel:.2e})");

    // ---- footprint --------------------------------------------------------------------
    let bytes_c: f64 = tensors.iter().map(|t| t.weights_bytes_compressed()).sum();
    let bytes_t: f64 = tensors.iter().map(|t| t.weights_bytes_turbo()).sum();
    let bw_c = bytes_c / total_w as f64;
    let bw_t = bytes_t / total_w as f64;

    // ---- bench ------------------------------------------------------------------------
    let iters = 10usize;
    run_token(StrandCpuPlan::Compressed, &tensors, &xs, &mut ys); // warm
    let mut best_c = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        run_token(StrandCpuPlan::Compressed, &tensors, &xs, &mut ys);
        best_c = best_c.min(t.elapsed().as_secs_f64());
    }
    run_token(StrandCpuPlan::TurboPanels, &tensors, &xs, &mut ys2); // warm
    let mut best_t = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        run_token(StrandCpuPlan::TurboPanels, &tensors, &xs, &mut ys2);
        best_t = best_t.min(t.elapsed().as_secs_f64());
    }

    println!("\n== {label}  ({n_tensors} tensors, {:.1}M weights/token) ==", total_w as f64 / 1e6);
    println!("  identity: compressed == decode_lean (bit-exact); turbo panel == Q12 (exact)");
    println!("  turbo vs compressed output: max rel diff {max_rel:.2e} (within float-MAC)");
    let dens_c = if bw_c < Q4K_BYTES_PER_WEIGHT { "BEATS Q4_K" } else { "loses" };
    println!(
        "  Compressed : {:>7.3} ms/token ({:>6.1} tok/s) | {bw_c:.4} B/w  ({dens_c} {Q4K_BYTES_PER_WEIGHT})",
        best_c * 1e3,
        1.0 / best_c
    );
    println!(
        "  TurboPanels: {:>7.3} ms/token ({:>6.1} tok/s) | {bw_t:.4} B/w  ({:.1}x RAM)",
        best_t * 1e3,
        1.0 / best_t,
        bw_t / bw_c
    );
    println!(
        "  turbo speedup {:.2}x (decode-once amortized; MAC-bound after that)",
        best_c / best_t
    );
}

fn main() {
    println!("gate-cpu-turbo — Wave-4 bet 3: StrandCpuPlan {{ Compressed | TurboPanels }}");
    println!("(compressed = bit-identical integer decode each token; turbo = decode-once i16 panels)");

    let cfg3 = TrellisConfig::for_bpw(3.0);
    assert_eq!((cfg3.k_bits, cfg3.l_bits), (3, 7), "deploy point is k=3,L=7");

    // MLP-only cache vs full per-token cache, at the q3 deploy point.
    bench("q3 deploy — MLP-only", &cfg3, true);
    bench("q3 deploy — full (attn+MLP)", &cfg3, false);
}
