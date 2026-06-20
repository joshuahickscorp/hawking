
use std::time::Instant;

use strand_decode_kernel::block_walk::gate_proto::{machine_stamp, synth_encoded};
use strand_decode_kernel::gemv::decode_q12_fast;
use strand_decode_kernel::neon_lut::{decode_q12_neonlut, decode_q12_neonlut_scalar_gather};
use strand_decode_kernel::split_decode::decode_q12_split;
use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts, EncodedTensor};
use strand_quant::TrellisConfig;

type DecodeFn = fn(&EncodedTensor, &TrellisConfig) -> Vec<i32>;

const KERNELS: &[(&str, DecodeFn)] = &[
    ("neonlut", decode_q12_neonlut),
    ("neonlut-sg", decode_q12_neonlut_scalar_gather),
];

fn configs() -> Vec<(TrellisConfig, &'static str)> {
    vec![
        (TrellisConfig::for_bpw(3.0), "k3 L7 (3-bit flagship, 4-table tree)"),
        (TrellisConfig::for_bpw(2.0), "k2 L6 (2-table tree)"),
        (TrellisConfig::for_bpw(4.0), "k4 L8 (scalar-gather lanes)"),
        (TrellisConfig::for_bpw_l(2.0, 7), "k2 L7"),
        (TrellisConfig::for_bpw_l(3.0, 6), "k3 L6"),
        (TrellisConfig::for_bpw_l(3.0, 8), "k3 L8"),
        (TrellisConfig::for_bpw_l(2.0, 5), "k2 L5 (fold, 1-table tree)"),
        (TrellisConfig::for_bpw_l(2.0, 12), "k2 L12 (out of envelope → fallback)"),
    ]
}

fn identity_matrix() -> usize {
    let mut checked = 0usize;
    for (cfg, label) in configs() {
        for seed in 0..24u64 {
            
            let n = 1 + (seed as usize * 211) % 4096;
            let w: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5)
                .collect();
            let variants = [
                encode_tensor(&w, &cfg),
                encode_tensor_with(&w, &cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
                encode_tensor_with(&w, &cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
                encode_tensor_with(
                    &w,
                    &cfg,
                    &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
                ),
            ];
            for enc in &variants {
                let reference = decode_tensor_fixed(enc, &cfg);
                for (name, f) in KERNELS {
                    let got = f(enc, &cfg);
                    assert_eq!(
                        got, reference,
                        "IDENTITY VIOLATION: `{name}` diverged from decode_tensor_fixed \
                         at {label}, n={n}, seed={seed}, tail={}, affine={} — release blocker",
                        enc.tail_biting, enc.has_affine_min
                    );
                    checked += 1;
                }
            }
        }
    }
    checked
}

fn bench_one(name: &str, f: DecodeFn, enc: &EncodedTensor, cfg: &TrellisConfig) -> f64 {
    
    let out = f(enc, cfg);
    assert_eq!(out.len(), enc.total);
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let out = f(enc, cfg);
        let dt = t.elapsed().as_secs_f64();
        std::hint::black_box(&out);
        best = best.min(dt);
    }
    let gws = enc.total as f64 / best / 1e9;
    println!("    {name:<12} {:>8.1} ms   {gws:>6.3} Gw/s", best * 1e3);
    gws
}

fn main() {
    let bench = std::env::args().any(|a| a == "--bench");

    println!("gate-neonlut: identity matrix (refuses perf without identity)…");
    let cells = identity_matrix();
    println!("IDENTITY OK — {cells} kernel×config×variant×length cells byte-identical to decode_tensor_fixed\n");

    if !bench {
        println!("(pass --bench for the machine-stamped perf sweep)");
        return;
    }

    let total = 18944usize * 3584;
    println!("bench: ffn_down {total} weights, synth, best-of-3, single-thread");
    println!("{}\n", machine_stamp());

    let points = [
        (TrellisConfig::for_bpw_l(2.0, 6), "k2 L6"),
        (TrellisConfig::for_bpw(3.0), "k3 L7 (3-bit flagship)"),
        (TrellisConfig::for_bpw(4.0), "k4 L8"),
    ];
    for (cfg, label) in points {
        let enc = synth_encoded(total, cfg.k_bits, cfg.block_len);
        println!("  {label}:");
        let base = bench_one("fast", decode_q12_fast, &enc, &cfg);
        let tbl = bench_one("neonlut", decode_q12_neonlut, &enc, &cfg);
        let sg = bench_one("neonlut-sg", decode_q12_neonlut_scalar_gather, &enc, &cfg);
        let split = bench_one("split", decode_q12_split, &enc, &cfg);
        println!(
            "    → neonlut {0:.2}× of fast, {1:.2}× of split; tbl-tree {2:.2}× of scalar-gather\n",
            tbl / base,
            tbl / split,
            tbl / sg
        );
    }
}
