// Forward integration study (deliverable #4): is PER-COLUMN-sign RHT a viable cheap-serving
// substitute for the shipped PER-ROW (per-element-sign) RHT?
//
// Why it matters: the STRAND weights are stored RHT'd, so inference is y = W·x = decode(W_rht)·T(x).
//   - per-ROW RHT (sign per element): T(x) differs for every output row -> out_features activation
//     transforms PER TOKEN -> the ~1 tok/s serving wall.
//   - per-COLUMN RHT (sign per column): T(x) = rht_forward(x) is the SAME for all rows -> ONE
//     transform per token -> ~6-8 tok/s, shippable.
// The only open question is QUALITY: does sharing the sign across rows (less randomization) cost
// quant accuracy vs per-row? This test measures it on a realistic weight, and proves the
// per-column GEMV identity (so dismantle can wire it correctly).
use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor_with, EncodeOpts};
use strand_quant::rht::{rht_forward, rht_forward_cols, rht_forward_rows, rht_inverse_cols, rht_inverse_rows, RhtConfig};
use strand_quant::TrellisConfig;

// deterministic realistic-ish weight: small Gaussian-like bulk + sparse heavy tails (like a real
// FFN-down matrix), no rng (CI-stable).
fn realistic_weight(out_f: usize, in_f: usize, seed: u64) -> Vec<f32> {
    let n = out_f * in_f;
    (0..n)
        .map(|i| {
            let a = ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed) as f32 * 2.3283e-10;
            let g = (a * 0.0006).sin() * 0.5 + (a * 0.013).cos() * 0.5; // bulk ~N(0,~0.02)
            let bulk = g * 0.02;
            if i % 211 == 0 {
                bulk + ((i % 7) as f32 - 3.0) * 0.15 // ~0.5% heavy-tail outliers
            } else {
                bulk
            }
        })
        .collect()
}

fn quant_recon_rht(w_rht: &[f32], cfg: &TrellisConfig) -> Vec<f32> {
    let enc = encode_tensor_with(w_rht, cfg, &EncodeOpts::default());
    decode_tensor_fixed(&enc, cfg)
        .iter()
        .map(|&q| q as f32 / 4096.0) // Q12 -> f32 (matches the fused-GEMV Q12_TO_F32)
        .collect()
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*x as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn matvec(w: &[f32], x: &[f32], out_f: usize, in_f: usize) -> Vec<f32> {
    (0..out_f)
        .map(|r| {
            let mut s = 0.0f32;
            for c in 0..in_f {
                s += w[r * in_f + c] * x[c];
            }
            s
        })
        .collect()
}

#[test]
fn rht_per_column_serving_feasibility() {
    let (out_f, in_f) = (256usize, 2048usize);
    let w = realistic_weight(out_f, in_f, 0xC0FFEE);
    let rcfg = RhtConfig::from_seed(0xA5A5_1234_DEAD_0001);

    println!("\n=== per-column RHT serving feasibility (ffn-down {out_f}x{in_f}) ===");
    for (bpw, l, name) in [(2.0f64, 12u32, "2-bit (k2 L12)"), (3.0, 7, "3-bit (k3 L7)")] {
        let cfg = TrellisConfig::for_bpw_l(bpw, l);

        // --- quant quality: none vs per-row vs per-column ---
        let rms_none = rel_rms(&w, &quant_recon_rht(&w, &cfg));

        let w_rows = rht_forward_rows(&w, &rcfg, in_f);
        let recon_rows = rht_inverse_rows(&quant_recon_rht(&w_rows, &cfg), &rcfg, in_f);
        let rms_rows = rel_rms(&w, &recon_rows);

        let w_cols = rht_forward_cols(&w, &rcfg, in_f);
        let recon_cols_rht = quant_recon_rht(&w_cols, &cfg);
        let recon_cols = rht_inverse_cols(&recon_cols_rht, &rcfg, in_f);
        let rms_cols = rel_rms(&w, &recon_cols);

        let penalty = (rms_cols - rms_rows) / rms_rows * 100.0;
        println!("  {name}: rel-RMS  none={:.3}%  per-row={:.3}%  per-COLUMN={:.3}%   col-vs-row penalty={:+.1}%", rms_none * 100.0, rms_rows * 100.0, rms_cols * 100.0, penalty);

        // --- per-column GEMV identity: y = decode(W_rht_col) · T(x), T(x)=rht_forward(x) ONCE ---
        let x = realistic_weight(1, in_f, 0xBEEF); // one activation row
        let y_orig = matvec(&w, &x, out_f, in_f); // true W·x
        let y_recon_spatial = matvec(&recon_cols, &x, out_f, in_f); // dequant weights, spatial
        let tx = rht_forward(&x, &rcfg); // the ONE shared activation transform
        let y_strand = matvec(&recon_cols_rht, &tx, out_f, in_f); // RHT-domain GEMV (what dismantle runs)

        let id_err = rel_rms(&y_recon_spatial, &y_strand); // orthogonal identity (should be ~0)
        let quant_err = rel_rms(&y_orig, &y_strand); // vs true output (quant cost)
        println!("    GEMV identity (RHT-domain == spatial): rel-err {:.2e}   output-vs-true: {:.3}%   activation transforms/token: per-row={out_f}  per-COLUMN=1", id_err, quant_err * 100.0);

        assert!(id_err < 1e-3, "{name}: per-column GEMV identity broke ({id_err:.2e})");
    }

    println!("  verdict: if col-vs-row penalty is small (~<10%), per-column buys ~{out_f}x cheaper");
    println!("           serving (1 vs {out_f} activation transforms/token) at ~the same quality.\n");
}
