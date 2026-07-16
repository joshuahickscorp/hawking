use std::time::Instant;

use strand_decode_kernel::block_walk::gate_proto::{canonical_configs, machine_stamp, synth_encoded};
use strand_decode_kernel::gemv::decode_q12_fast;
use strand_decode_kernel::gemv_par::decode_q12_par;
use strand_decode_kernel::paired_lut::{decode_q12_paired_par_with_table, decode_q12_paired_with_table, PairTable};
use strand_decode_kernel::prepared::{decode_q12_par_prepared, decode_q12_prepared_paired, decode_q12_prepared_paired_par, PreparedTensor};
use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};
use strand_quant::TrellisConfig;

fn identity_matrix() -> usize {
    let mut checked = 0usize;
    for (cfg, label) in canonical_configs() {
        let table = PairTable::build(&cfg);
        for seed in 0..24u64 {
            let n = 1 + (seed as usize * 211) % 4096;
            let w: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect();
            let variants = [
                encode_tensor(&w, &cfg),
                encode_tensor_with(&w, &cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
                encode_tensor_with(&w, &cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
                encode_tensor_with(&w, &cfg, &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() }),
            ];
            for enc in &variants {
                let reference = decode_tensor_fixed(enc, &cfg);
                let p = PreparedTensor::new(enc.clone(), cfg.clone());
                for (name, got) in [("compose", decode_q12_prepared_paired(&p, &table)), ("compose-par", decode_q12_prepared_paired_par(&p, &table))] {
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
    println!("\n== bench (best-of-3, ffn_down {out_f}x{in_f} = {:.1}M weights) ==", total as f64 / 1e6);
    println!("  {}", machine_stamp());

    for (cfg, label) in [
        (TrellisConfig::for_bpw(3.0), "3-bit deploy (k=3, L=7) — THE envelope point"),
        (TrellisConfig::for_bpw_l(3.0, 10), "k=3, L=10 — envelope edge (64 KB table)"),
        (TrellisConfig::for_bpw_l(2.0, 12), "2-bit reopen (k=2, L=12) — OUT of envelope (128 KB)"),
    ] {
        let table = PairTable::build(&cfg);
        println!("\n  -- {label} | pair table {:.0} KB --", table.size_bytes() as f64 / 1024.0);

        for tail in [false, true] {
            let mut enc = synth_encoded(total, cfg.k_bits, 256);
            enc.tail_biting = tail;
            let tp = Instant::now();
            let p = PreparedTensor::new(enc.clone(), cfg.clone());
            let prep_ms = tp.elapsed().as_secs_f64() * 1e3;
            assert!(p.is_fast_path());

            let mut rows: Vec<(&str, bool, f64)> = Vec::new();
            let mut run = |name: &'static str, is_par: bool, f: &dyn Fn() -> Vec<i32>| {
                let mut best = f64::INFINITY;
                for _ in 0..3 {
                    let t = Instant::now();
                    let out = f();
                    let dt = t.elapsed().as_secs_f64();
                    std::hint::black_box(&out);
                    best = best.min(dt);
                }
                rows.push((name, is_par, best));
            };

            run("1T fast (baseline)", false, &|| decode_q12_fast(&enc, &cfg));
            run("1T paired alone", false, &|| decode_q12_paired_with_table(&enc, &cfg, &table));
            run("1T COMPOSE prep+paired", false, &|| decode_q12_prepared_paired(&p, &table));

            run("PAR par (baseline)", true, &|| decode_q12_par(&enc, &cfg));
            run("PAR prepared alone", true, &|| decode_q12_par_prepared(&p));
            run("PAR paired alone", true, &|| decode_q12_paired_par_with_table(&enc, &cfg, &table));
            run("PAR COMPOSE prep+paired", true, &|| decode_q12_prepared_paired_par(&p, &table));

            let base_1t = rows[0].2;
            let base_par = rows[3].2;
            println!("    tail-biting {} (prepare {:.0} ms one-time, {:.4} B/w prepared):", if tail { "ON " } else { "off" }, prep_ms, p.prepared_bytes_per_weight());
            for (name, is_par, best) in &rows {
                let base = if *is_par { base_par } else { base_1t };
                println!("      {name:<26} {:>8.1} ms   {:>6.2} Gw/s   {:.2}x vs its baseline", best * 1e3, total as f64 / best / 1e9, base / best);
            }
        }
    }
    println!(
        "\n  verdict key: compose ≈ prepared-ratio × paired-ratio ⇒ multiplicative; \
         ≈ max(solo) ⇒ one win subsumes the other; < max(solo) ⇒ interference \
         (the G0 register-pressure lesson)."
    );
}

fn main() {
    println!("gate-compose — prepared × paired composition identity gate");
    let t = Instant::now();
    let checked = identity_matrix();
    println!(
        "identity: {checked} compose×config×variant cells byte-identical to \
         decode_tensor_fixed in {:.1}s ✓",
        t.elapsed().as_secs_f64()
    );

    if std::env::args().any(|a| a == "--bench") {
        bench();
    } else {
        println!("(pass --bench for the machine-stamped composition sweep)");
    }
}
