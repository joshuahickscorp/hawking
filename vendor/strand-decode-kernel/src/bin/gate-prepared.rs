
use std::io::Write;
use std::time::Instant;

use strand_decode_kernel::block_walk::gate_proto::{canonical_configs, machine_stamp, synth_encoded};
use strand_decode_kernel::fused::fused_gemm;
use strand_decode_kernel::gemv_par::{decode_q12_par, decode_tensor_q12_par};
use strand_decode_kernel::loader::StrandModel;
use strand_decode_kernel::prepared::{
    decode_q12_par_prepared, fused_gemm_prepared, fused_gemm_prepared_with_q12, PreparedTensor,
};
use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts, EncodedTensor};
use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2};
use strand_quant::TrellisConfig;

fn identity_matrix() -> usize {
    let mut checked = 0usize;
    for (cfg, label) in canonical_configs() {
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
                let p = PreparedTensor::new(enc.clone(), cfg);
                
                assert_eq!(
                    decode_q12_par_prepared(&p),
                    reference,
                    "IDENTITY VIOLATION: prepared decode diverged from decode_tensor_fixed \
                     at {label}, n={n}, seed={seed}, tail={}, affine={} — release blocker",
                    enc.tail_biting, enc.has_affine_min
                );
                checked += 1;
                
                for &batch in &[1usize, 4] {
                    let xs = vec![0.5f32; batch * n];
                    let (y_p, q_p) = fused_gemm_prepared_with_q12(&p, 1, n, &xs, batch);
                    assert_eq!(
                        q_p,
                        decode_q12_par(enc, &cfg),
                        "IDENTITY VIOLATION: prepared fused hidden-Q12 diverged at {label}, \
                         n={n}, seed={seed}, B={batch}",
                    );
                    let y_ref = fused_gemm(enc, &cfg, None, 1, n, &xs, batch);
                    for (i, (a, b)) in y_p.iter().zip(y_ref.iter()).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "IDENTITY VIOLATION: prepared fused y diverged at {label}, n={n}, \
                             seed={seed}, B={batch}, flat index {i}",
                        );
                    }
                    checked += 2;
                }
            }
        }
    }
    
    {
        let cfg = TrellisConfig::for_bpw(3.0);
        let (rows, cols) = (37usize, 300usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.0091).sin() * 0.4).collect();
        let enc = encode_tensor_with(
            &w,
            &cfg,
            &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
        );
        let p = PreparedTensor::new(enc.clone(), cfg);
        for &batch in &[1usize, 3, 5, 16, 21, 64, 65] {
            let xs: Vec<f32> =
                (0..batch * cols).map(|i| ((i as f32) * 0.0713).cos()).collect();
            let y_p = fused_gemm_prepared(&p, rows, cols, &xs, batch);
            let y_r = fused_gemm(&enc, &cfg, None, rows, cols, &xs, batch);
            for (i, (a, b)) in y_p.iter().zip(y_r.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "shaped fused diverged: B={batch} idx={i}");
            }
            checked += 1;
        }
    }
    
    {
        use strand_quant::decode::decode_tensor_fixed_with_lut;
        let cfg = TrellisConfig::for_bpw(3.0).with_vec_dim(2);
        let (ns, d) = (cfg.num_states(), cfg.vec_dim());
        let lut: Vec<i32> = (0..ns * d)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 20) as i32 - 2048)
            .collect();
        let w: Vec<f32> = (0..700).map(|i| ((i as f32) * 0.011).cos() * 0.3).collect();
        let enc = encode_tensor(&w, &cfg);
        let want = decode_tensor_fixed_with_lut(&enc, &cfg, &lut);
        let p = PreparedTensor::with_lut(enc, cfg, Some(lut));
        assert!(!p.is_fast_path(), "vec tensor must prepare as fallback-routed");
        assert_eq!(decode_q12_par_prepared(&p), want, "prepared vec fallback");
        checked += 1;
    }
    checked
}

fn memory_bill() {
    println!("\n== prepared-form memory bill (ffn_down 18944x3584 = 67.9M weights) ==");
    for (cfg, label) in [
        (TrellisConfig::for_bpw(3.0), "3-bit deploy (k3 L7)"),
        (TrellisConfig::for_bpw_l(2.0, 12), "2-bit reopen (k2 L12)"),
    ] {
        let total = 18944usize * 3584;
        let enc = synth_encoded(total, cfg.k_bits, 256);
        let payload = enc.bits.len();
        let t = Instant::now();
        let p = PreparedTensor::new(enc, cfg);
        let dt = t.elapsed().as_secs_f64();
        println!(
            "  {label}: prepared {:.1} MB = {:.4} B/w (payload {:.1} MB = {:.4} B/w; \
             prepared/payload {:.0}%) | resident {:.1} MB | prepare {:.0} ms (tail-bite OFF)",
            p.prepared_bytes() as f64 / 1e6,
            p.prepared_bytes_per_weight(),
            payload as f64 / 1e6,
            payload as f64 / total as f64,
            100.0 * p.prepared_bytes() as f64 / payload as f64,
            p.resident_bytes() as f64 / 1e6,
            dt * 1e3,
        );
        
        let mut enc_tb = synth_encoded(total, cfg.k_bits, 256);
        enc_tb.tail_biting = true;
        let t = Instant::now();
        let ptb = PreparedTensor::new(enc_tb, cfg);
        let dtb = t.elapsed().as_secs_f64();
        assert_eq!(ptb.prepared_bytes(), p.prepared_bytes());
        println!(
            "    tail-bite ON: same bytes; prepare {:.0} ms (one-time prescan, paid never again)",
            dtb * 1e3
        );
    }
    println!("  (affine-min adds 0.125 B/w for off[]; these synth tensors carry none)");
}

fn best_of<F: FnMut()>(mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn write_temp_v2(name: &str, rows: u64, cols: u64, cfg: &TrellisConfig, enc: &EncodedTensor) -> std::path::PathBuf {
    let shape = [rows, cols];
    let pt = PackedTensorV2 {
        base: PackedTensor {
            name,
            shape: &shape,
            rht_seed: 0,
            l_bits: cfg.l_bits as u8,
            k_bits: cfg.k_bits as u8,
            vec_dim: cfg.vec_dim() as u8,
            enc,
        },
        block_len: cfg.block_len as u32,
    };
    let buf = write_strand_v2(&[pt], [0u8; 32], true).expect("write_strand_v2");
    let mut path = std::env::temp_dir();
    let uniq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    path.push(format!("strand_gate_prepared_{}_{uniq}.strand", std::process::id()));
    let mut f = std::fs::File::create(&path).expect("create temp .strand");
    f.write_all(&buf).expect("write");
    f.sync_all().ok();
    path
}

fn bench_single(cfg: &TrellisConfig, label: &str) {
    let (out_f, in_f) = (18944usize, 3584usize);
    let total = out_f * in_f;
    println!("  -- {label}: single tensor ffn_down {out_f}x{in_f} ({:.1}M w) --", total as f64 / 1e6);
    for tail in [false, true] {
        let mut enc = synth_encoded(total, cfg.k_bits, 256);
        enc.tail_biting = tail;
        let path = write_temp_v2("w", out_f as u64, in_f as u64, cfg, &enc);
        let model = StrandModel::open(&path).expect("open temp v2");

        let t = Instant::now();
        let p = PreparedTensor::new(enc.clone(), *cfg);
        let prep_ms = t.elapsed().as_secs_f64() * 1e3;

        let t_loader = best_of(|| {
            let q = decode_tensor_q12_par(&model, "w").expect("loader-cold decode");
            std::hint::black_box(&q);
        });
        let t_cold = best_of(|| {
            let q = decode_q12_par(&enc, cfg);
            std::hint::black_box(&q);
        });
        let t_prep = best_of(|| {
            let q = decode_q12_par_prepared(&p);
            std::hint::black_box(&q);
        });
        println!(
            "    tail={}: loader-cold {:>6.1} ms ({:.2} Gw/s) | enc-cold {:>6.1} ms ({:.2} Gw/s) | \
             prepared {:>6.1} ms ({:.2} Gw/s) = {:.2}x enc-cold, {:.2}x loader-cold | prepare once {:.0} ms",
            if tail { "ON " } else { "off" },
            t_loader * 1e3,
            total as f64 / t_loader / 1e9,
            t_cold * 1e3,
            total as f64 / t_cold / 1e9,
            t_prep * 1e3,
            total as f64 / t_prep / 1e9,
            t_cold / t_prep,
            t_loader / t_prep,
            prep_ms,
        );
        let _ = std::fs::remove_file(&path);
    }
}

fn bench_model_loop(cfg: &TrellisConfig, label: &str) {
    let (out_f, in_f) = (4096usize, 3584usize); 
    let total = out_f * in_f;
    println!(
        "  -- {label}: 24-tensor model loop ({out_f}x{in_f} each, {:.0}M w/token) --",
        24.0 * total as f64 / 1e6
    );
    for tail in [false, true] {
        let encs: Vec<EncodedTensor> = (0..24)
            .map(|_| {
                let mut e = synth_encoded(total, cfg.k_bits, 256);
                e.tail_biting = tail;
                e
            })
            .collect();
        let t = Instant::now();
        let preps: Vec<PreparedTensor> =
            encs.iter().map(|e| PreparedTensor::new(e.clone(), *cfg)).collect();
        let prep_ms = t.elapsed().as_secs_f64() * 1e3;
        let prep_mb: usize = preps.iter().map(|p| p.prepared_bytes()).sum();

        let t_cold = best_of(|| {
            for e in &encs {
                let q = decode_q12_par(e, cfg);
                std::hint::black_box(&q);
            }
        });
        let t_prep = best_of(|| {
            for p in &preps {
                let q = decode_q12_par_prepared(p);
                std::hint::black_box(&q);
            }
        });
        let gw = 24.0 * total as f64;
        println!(
            "    tail={}: cold {:>7.1} ms/token ({:.2} Gw/s) | prepared {:>7.1} ms/token \
             ({:.2} Gw/s) = {:.2}x | prepare once {:.0} ms, +{:.1} MB resident",
            if tail { "ON " } else { "off" },
            t_cold * 1e3,
            gw / t_cold / 1e9,
            t_prep * 1e3,
            gw / t_prep / 1e9,
            t_cold / t_prep,
            prep_ms,
            prep_mb as f64 / 1e6,
        );
    }
}

fn bench_fused(cfg: &TrellisConfig, label: &str) {
    let (out_f, in_f) = (18944usize, 3584usize);
    let total = out_f * in_f;
    println!("  -- {label}: fused GEMM ffn_down {out_f}x{in_f} --");
    for tail in [false, true] {
        let mut enc = synth_encoded(total, cfg.k_bits, 256);
        enc.tail_biting = tail;
        let p = PreparedTensor::new(enc.clone(), *cfg);
        for &batch in &[1usize, 64] {
            let xs: Vec<f32> =
                (0..batch * in_f).map(|i| ((i as f32) * 0.0713).cos()).collect();
            let t_cold = best_of(|| {
                let y = fused_gemm(&enc, cfg, None, out_f, in_f, &xs, batch);
                std::hint::black_box(&y);
            });
            let t_prep = best_of(|| {
                let y = fused_gemm_prepared(&p, out_f, in_f, &xs, batch);
                std::hint::black_box(&y);
            });
            println!(
                "    tail={} B={batch:<2}: cold {:>7.1} ms | prepared {:>7.1} ms = {:.2}x",
                if tail { "ON " } else { "off" },
                t_cold * 1e3,
                t_prep * 1e3,
                t_cold / t_prep,
            );
        }
    }
}

fn ab64(cfg: &TrellisConfig, label: &str) {
    let (out_f, in_f) = (18944usize, 3584usize);
    let total = out_f * in_f;
    let enc = synth_encoded(total, cfg.k_bits, 256);
    let p = PreparedTensor::new(enc.clone(), *cfg);
    let batch = 64usize;
    let xs: Vec<f32> = (0..batch * in_f).map(|i| ((i as f32) * 0.0713).cos()).collect();
    println!("  -- {label}: interleaved A/B, fused B=64, tail=off --");
    for round in 0..6 {
        let t = Instant::now();
        let y = fused_gemm(&enc, cfg, None, out_f, in_f, &xs, batch);
        let t_cold = t.elapsed().as_secs_f64();
        std::hint::black_box(&y);
        let t = Instant::now();
        let y = fused_gemm_prepared(&p, out_f, in_f, &xs, batch);
        let t_prep = t.elapsed().as_secs_f64();
        std::hint::black_box(&y);
        println!(
            "    round {round}: cold {:>6.1} ms | prepared {:>6.1} ms ({:.2}x)",
            t_cold * 1e3,
            t_prep * 1e3,
            t_cold / t_prep
        );
    }
}

fn science_job_alive() -> bool {
    std::process::Command::new("pgrep")
        .args(["-f", "strand-delta|strand-qat|quantize-model|strand-7b-ppl"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn main() {
    println!("gate-prepared — SCHISM #4: the prepared-model layer (identity → bill → bench)");
    let t = Instant::now();
    let checked = identity_matrix();
    println!(
        "identity: {} prepared cells byte/bit-identical (decode vs decode_tensor_fixed, \
         fused q12 vs decode_q12_par, fused y vs fused_gemm) in {:.1}s ✓",
        checked,
        t.elapsed().as_secs_f64()
    );

    memory_bill();

    if std::env::args().any(|a| a == "--ab64") {
        println!("\n== A/B diagnosis (interleaved, B=64) ==");
        println!("  {}", machine_stamp());
        ab64(&TrellisConfig::for_bpw(3.0), "3-bit deploy");
        ab64(&TrellisConfig::for_bpw_l(2.0, 12), "2-bit reopen");
        return;
    }

    if std::env::args().any(|a| a == "--bench") {
        
        if !std::env::args().any(|a| a == "--no-wait") {
            while science_job_alive() {
                println!("co-running STRAND science job detected — waiting 60 s (pass --no-wait to override)…");
                std::thread::sleep(std::time::Duration::from_secs(60));
            }
        }
        println!("\n== bench (best-of-3, per-call latency) ==");
        println!("  {}", machine_stamp());
        for (cfg, label) in [
            (TrellisConfig::for_bpw(3.0), "3-bit deploy"),
            (TrellisConfig::for_bpw_l(2.0, 12), "2-bit reopen"),
        ] {
            bench_single(&cfg, label);
            bench_model_loop(&cfg, label);
            bench_fused(&cfg, label);
        }
    } else {
        println!("(pass --bench for the machine-stamped latency comparison)");
    }
}
