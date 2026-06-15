
use std::time::Instant;

use strand_decode_kernel::block_walk::gate_proto::{canonical_configs, machine_stamp, synth_encoded};
use strand_decode_kernel::gemv::decode_q12_fast;
use strand_decode_kernel::gemv_par::decode_q12_par;
use strand_decode_kernel::paired_lut::{
    decode_q12_paired, decode_q12_paired_par_with_table, decode_q12_paired_with_table, PairTable,
};
use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};
use strand_quant::TrellisConfig;

fn identity_matrix() -> usize {
    let mut checked = 0usize;
    for (cfg, label) in canonical_configs() {
        let table = PairTable::build(&cfg);
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
                for (name, got) in [
                    ("paired", decode_q12_paired(enc, &cfg)),
                    ("paired+table", decode_q12_paired_with_table(enc, &cfg, &table)),
                    ("paired-par", decode_q12_paired_par_with_table(enc, &cfg, &table)),
                ] {
                    assert_eq!(
                        got, reference,
                        "IDENTITY VIOLATION: `{name}` diverged from decode_tensor_fixed at \
                         {label}, n={n}, seed={seed}, tail={}, affine={} — release blocker",
                        enc.tail_biting, enc.has_affine_min
                    );
                    checked += 1;
                }
            }
        }
    }
    checked
}

fn bench() {
    let (out_f, in_f) = (18944usize, 3584usize);
    let total = out_f * in_f;
    println!(
        "\n== bench (best-of-3, ffn_down {out_f}x{in_f} = {:.1}M weights) ==",
        total as f64 / 1e6
    );
    println!("  {}", machine_stamp());

    for (cfg, label) in [
        (TrellisConfig::for_bpw(3.0), "3-bit deploy (k=3, L=7)"),
        (TrellisConfig::for_bpw_l(2.0, 12), "2-bit reopen (k=2, L=12)"),
    ] {
        println!("  -- {label} --");
        
        let tb = Instant::now();
        let table = PairTable::build(&cfg);
        let build_s = tb.elapsed().as_secs_f64();
        println!(
            "  pair table: {} entries, {:.1} KB, built in {:.3} ms (load-time only)",
            1usize << (cfg.l_bits + cfg.k_bits),
            table.size_bytes() as f64 / 1024.0,
            build_s * 1e3
        );

        let enc = synth_encoded(total, cfg.k_bits, 256);
        let mut results: Vec<(&str, f64)> = Vec::new();
        let mut run = |name: &'static str, f: &dyn Fn() -> Vec<i32>| {
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                let out = f();
                let dt = t.elapsed().as_secs_f64();
                std::hint::black_box(&out);
                best = best.min(dt);
            }
            results.push((name, best));
        };
        run("fast (1-thread baseline)", &|| decode_q12_fast(&enc, &cfg));
        run("paired (1-thread)", &|| decode_q12_paired_with_table(&enc, &cfg, &table));
        run("par (rayon baseline)", &|| decode_q12_par(&enc, &cfg));
        run("paired-par (rayon)", &|| decode_q12_paired_par_with_table(&enc, &cfg, &table));

        let base_1t = results[0].1;
        let base_par = results[2].1;
        for (i, (name, best)) in results.iter().enumerate() {
            let ratio = match i {
                0 | 1 => base_1t / best,
                _ => base_par / best,
            };
            println!(
                "  {name:<26} {:>8.1} ms   {:>6.2} Gw/s   {:.2}x vs its baseline",
                best * 1e3,
                total as f64 / best / 1e9,
                ratio
            );
        }
    }
}

fn main() {
    println!("gate-paired — SCHISM #3 (paired-step LUT) identity gate");
    let t = Instant::now();
    let checked = identity_matrix();
    println!(
        "identity: {checked} kernel×config×variant cells byte-identical to \
         decode_tensor_fixed in {:.1}s ✓",
        t.elapsed().as_secs_f64()
    );

    if std::env::args().any(|a| a == "--bench") {
        
        bench();
    } else {
        println!("(pass --bench for the machine-stamped throughput sweep)");
    }
}
