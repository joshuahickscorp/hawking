
use std::time::Instant;
use strand_decode_kernel::{footprint_bytes, matvec};
use strand_quant::encode::encode_tensor;
use strand_quant::TrellisConfig;

fn main() {
    let out = 1024usize;
    let inf = 1024usize;
    let nw = out * inf;
    let bpws = [3.0f64, 2.0, 1.5];
    println!(
        "strand-decode-kernel bench — {out}x{inf} layer ({nw} weights), reference decode+matvec\n"
    );
    let x: Vec<f32> = (0..inf).map(|i| (i as f32 * 0.01).sin()).collect();
    let weights: Vec<f32> = (0..nw).map(|i| (i as f32 * 0.0007).sin()).collect();
    let bf16 = footprint_bytes(nw, 16.0);

    for &bpw in &bpws {
        let cfg = TrellisConfig::for_bpw(bpw);
        let enc = encode_tensor(&weights, &cfg);
        let _ = matvec(&enc, &cfg, None, out, inf, &x); 
        let iters = 100;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = matvec(&enc, &cfg, None, out, inf, &x);
        }
        let secs = t0.elapsed().as_secs_f64();
        let gmacs = (nw * iters) as f64 / secs / 1e9;
        let bytes = footprint_bytes(nw, bpw);
        println!(
            "  bpw={:.1}  {:6.2} GMAC/s   weights={:4} KB   {:4.1}x smaller than bf16  (~{:4.1}x less weight traffic)",
            bpw,
            gmacs,
            bytes / 1024,
            bf16 as f64 / bytes as f64,
            16.0 / bpw
        );
    }
    println!(
        "\nnote: reference decode+matmul (no SIMD/GPU yet). The packed-int kernel is the optimization\n\
         target; this establishes the correctness baseline + the bandwidth story the fast path inherits."
    );
}
