use std::time::Instant;

use strand_quant::{encode_tensor_with, EncodeOpts, TrellisConfig};

fn bench(label: &str, weights: &[f32], cfg: &TrellisConfig, opts: &EncodeOpts, iters: u32) -> f64 {
    let n = weights.len();

    let _ = encode_tensor_with(&weights[..256.min(n)], cfg, opts);

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = encode_tensor_with(weights, cfg, opts);
    }
    let elapsed = t0.elapsed();
    let per_iter = elapsed / iters;
    let mw_s = (n as f64 * iters as f64) / elapsed.as_secs_f64() / 1e6;
    println!("  {label:<12} avg={per_iter:.1?}  {mw_s:.3} Mw/s");
    mw_s
}

fn main() {
    const N: usize = 256 * 1024;

    let weights: Vec<f32> = (0..N).map(|i| ((i as f32) * 0.003_141_592_6).sin()).collect();

    for bpw in [3.0_f64, 4.0] {
        let cfg = TrellisConfig::for_bpw(bpw);
        println!("{bpw:.0} bpw (L={}, {} states, N={N}):", cfg.l_bits, cfg.num_states());

        let gpu_opts = EncodeOpts { adaptive: true, ..Default::default() };
        let gpu = bench("GPU", &weights, &cfg, &gpu_opts, 5);

        let cpu_opts = EncodeOpts { adaptive: true, tail_biting: true, ..Default::default() };
        let cpu = bench("CPU SIMD", &weights, &cfg, &cpu_opts, 3);

        println!("  GPU speedup: {:.1}×\n", gpu / cpu);
    }
}
